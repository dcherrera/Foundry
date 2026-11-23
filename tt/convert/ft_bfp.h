/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_bfp.h
 * @brief Host-side block-floating-point tile packer: fp32 tiles to and from the
 *   Tenstorrent BFP8/BFP4/BFP2 byte layout the unpacker reads from L1, plus BFP6_B, a
 *   DRAM-capacity format the unpacker does NOT read (qwen36 spec §4.1 / §4.1b).
 *
 * FORMATS (tt-isa-documentation, FloatBitPatterns.md "BFP"): every group of 16
 * datums shares one 8-bit exponent. A datum is a sign bit followed by an
 * unsigned magnitude with no implicit leading one: 7 magnitude bits for BFP8,
 * 5 for BFP6, 3 for BFP4, 1 for BFP2. The "b" formats store an FP32-style biased exponent (bias 127);
 * the "a" formats store an FP16-style 5-bit exponent (bias 15) zero-extended
 * to a byte.
 *
 * TILE LAYOUT: callers hand in 32x32 tiles stored row-major within each tile.
 * The packer walks the tile as four 16x16 faces (top-left, top-right,
 * bottom-left, bottom-right) and, inside each face, row by row; each face row
 * of 16 values is one block. A packed tile is the 64 shared-exponent bytes
 * (one per block, Packers/FormatConversion.md: "a separate output stream ...
 * one byte per 16 datums") followed by the datums in the same block order: one
 * byte each for BFP8, two per byte (low nibble first) for BFP4, four per byte
 * (lowest bit pair first) for BFP2. BFP6_B datums are a little-endian 6-bit
 * stream (datum i at stream bits 6i..6i+5, so 3 bytes hold 4 datums); the
 * other layouts (planes, fold) are reached with ft_bfp6_relayout_tiles().
 * Unpacking restores the row-major tile.
 *
 * BFP6_B -> BFP8_B IS EXACT: same exponent byte, BFP8 datum = BFP6 datum << 2
 * (sign lands on bit 7, the 5-bit magnitude on bits 6..2). ft_bfp6_expand_tiles()
 * is the host reference every card expander is byte-compared against.
 *
 * ROUNDING: fp32 to datum takes the 24-bit significand (implicit one
 * included), aligns it to the block's shared exponent with a truncating right
 * shift, then rounds the aligned value to the datum width, nearest with ties
 * to even, saturating at the largest magnitude instead of carrying. Zeros and
 * denormals pack to datum 0, and a datum whose magnitude rounds to 0 never
 * carries a sign. For the "a" formats a value below the 5-bit exponent range
 * is first raised to +-2^-15. The shared exponent is the largest exponent
 * field in the block (clamped to 0..31 for "a"); infinity and NaN take part
 * like any other value.
 *
 * DECODING follows FloatBitPatterns.md's BFP-to-float routine, except that a
 * sign-only datum (magnitude 0) decodes to -0 rather than -2^128/-2^16, and an
 * exponent that underflows wraps rather than clamps; both match the reference.
 *
 * @status Complete. Bit-exact against reference/ft_bfp_reference_vectors.h.
 * @issues None
 * @todo None
 */

#ifndef FT_BFP_H
#define FT_BFP_H

#include <stddef.h>
#include <stdint.h>

/** The block-float formats this packer produces and consumes. */
typedef enum {
    FT_BFP_BFP8_B = 0, /**< 8-bit datums, 8-bit FP32-biased shared exponent. */
    FT_BFP_BFP4_B = 1, /**< 4-bit datums, 8-bit FP32-biased shared exponent. */
    FT_BFP_BFP8_A = 2, /**< 8-bit datums, 5-bit FP16-biased shared exponent. */
    FT_BFP_BFP4_A = 3, /**< 4-bit datums, 5-bit FP16-biased shared exponent. */
    FT_BFP_BFP6_B = 4, /**< 6-bit datums, 8-bit exponent. DRAM-only: no unpacker reads it. */
    FT_BFP_BFP2_B = 5  /**< 2-bit datums (sign + 1 magnitude bit), 8-bit exponent. */
} ft_bfp_format;

/** Byte layouts of a BFP6_B tile's 768 datum bytes (the 64 exponent bytes lead either way). */
typedef enum {
    /** Little-endian 6-bit stream: 3 bytes = 4 datums. ft_bfp_pack_tiles() writes this. */
    FT_BFP6_LAYOUT_STREAM = 0,
    /** A BFP4_B-shaped plane of sign + high 3 magnitude bits (512 B, low nibble first),
     *  then a plane of the low 2 magnitude bits (256 B, lowest pair first). */
    FT_BFP6_LAYOUT_PLANES = 1,
    /** FOLD (qwen36 plan S3): every 16 datums (one 16-byte group of the BFP8_B tile they
     *  expand to, output words o0..o3) are 12 bytes w0..w2 with w_i = o_i's top six bits of
     *  each byte (o_i & 0xFCFCFCFC) plus, in each byte's low two bits, a 2-bit slice of the
     *  same byte of o3: w0 holds o3's bits 7..6, w1 bits 5..4, w2 bits 3..2. Expansion is
     *  o_i = w_i & 0xFCFCFCFC and o3 = ((w0 & m) << 6) | ((w1 & m) << 4) | ((w2 & m) << 2),
     *  m = 0x03030303: eleven ALU ops a group (the stream layout takes ~46). Words are
     *  little-endian. ft_bfp6_fold_from_stream() is the fast host conversion. */
    FT_BFP6_LAYOUT_FOLD = 2
} ft_bfp6_layout;

/** Return codes of the tile-level calls. */
enum {
    FT_BFP_OK = 0,
    FT_BFP_EINVAL = -3
};

/** Datums sharing one exponent byte. */
#define FT_BFP_BLOCK_VALUES 16

/** Exponent bytes at the head of every packed 32x32 tile (1024 / 16). */
#define FT_BFP_TILE_EXPONENT_BYTES 64

/** Bytes of one packed 32x32 tile; 0 for an unknown format. */
size_t ft_bfp_tile_bytes(ft_bfp_format format);

/** Magnitude bits per datum (the sign bit excluded); 0 for an unknown format. */
uint32_t ft_bfp_mantissa_bits(ft_bfp_format format);

/**
 * Pack `tile_count` row-major 32x32 fp32 tiles into `out`, which must hold
 * tile_count * ft_bfp_tile_bytes(format) bytes. Zero tiles is a no-op.
 * Returns FT_BFP_OK, or FT_BFP_EINVAL for an unknown format or a NULL buffer.
 */
int ft_bfp_pack_tiles(ft_bfp_format format, const float *tiles, size_t tile_count, void *out);

/** Inverse of ft_bfp_pack_tiles: decode packed tiles back to row-major fp32. */
int ft_bfp_unpack_tiles(ft_bfp_format format, const void *packed, size_t tile_count, float *tiles_out);

/**
 * Encode one fp32 bit pattern as a datum under `shared_exp`, which is in the
 * format's own exponent encoding (8-bit for "b", 5-bit for "a"). Counts ties.
 */
uint8_t ft_bfp_convert_float_bits(ft_bfp_format format, uint32_t input_bits, uint32_t shared_exp);

/** Decode one datum under `shared_exp` to fp32 bits, as the hardware reads it. */
uint32_t ft_bfp_convert_datum(ft_bfp_format format, uint8_t datum, uint32_t shared_exp);

/** The shared exponent byte for 16 fp32 bit patterns, in the format's encoding. */
uint8_t ft_bfp_block_shared_exponent(ft_bfp_format format, const uint32_t *block_bits);

/**
 * Diagnostic counters since the last reset: rounding ties met by
 * ft_bfp_convert_float_bits (and therefore by packing), and how many of them
 * rounded the magnitude up. Either pointer may be NULL. Not thread-safe.
 */
void ft_bfp_tie_counters(uint64_t *ties_seen, uint64_t *ties_rounded_up);

/** Zero both tie counters. */
void ft_bfp_tie_counters_reset(void);

/**
 * Rewrite `tile_count` packed BFP6_B tiles from one datum layout to the other. The
 * exponent bytes are copied unchanged. `in` and `out` must not overlap.
 * Returns FT_BFP_OK or FT_BFP_EINVAL.
 */
int ft_bfp6_relayout_tiles(ft_bfp6_layout from, ft_bfp6_layout to, const void *in,
                           size_t tile_count, void *out);

/**
 * The exact BFP6_B -> BFP8_B expansion: each 832 B tile in `layout` becomes the 1088 B
 * BFP8_B tile of the same values (exponents copied, datum << 2). Returns FT_BFP_OK or
 * FT_BFP_EINVAL.
 */
int ft_bfp6_expand_tiles(ft_bfp6_layout layout, const void *in, size_t tile_count, void *out);

/**
 * Stream -> fold for `tile_count` BFP6_B tiles, word-parallel (the load-time conversion of an
 * artifact's expert stacks). Byte-identical to ft_bfp6_relayout_tiles(STREAM, FOLD, ...).
 * `in` and `out` must not overlap.
 * @return FT_BFP_OK, or FT_BFP_EINVAL on a NULL pointer with a nonzero count.
 */
int ft_bfp6_fold_from_stream(const void *in, size_t tile_count, void *out);

#endif
