/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_bfp.c
 * @brief Block-floating-point (BFP8/BFP4, "a" and "b" exponent flavours; BFP6_B and
 *   BFP2_B) tile packer and unpacker, the BFP6_B layout rewrite and the exact BFP6_B ->
 *   BFP8_B expansion. See ft_bfp.h for the layout and rounding contract.
 *
 * Format meanings: tt-isa-documentation FloatBitPatterns.md ("BFP", including
 * the hardware's BFP-to-BF16/FP16 conversion routine that the decoder here
 * follows). Memory layout: Packers/FormatConversion.md (separate exponent
 * stream, one byte per 16 datums; two BFP4 datums per byte).
 *
 * @status Complete. test_bfp, test_convert and test_artifact_cache green.
 * @issues The device packer rounds ties away from zero and derives BFP4 by
 *   truncating BFP8 (Packers/FormatConversion.md); this host packer rounds
 *   ties to even straight from the aligned fp32 significand, because that is
 *   what the committed reference vectors and the shipped weight artifacts
 *   encode. Likewise a sign-only datum decodes to -0, not -2^128.
 * @todo None
 */

#include "ft_bfp.h"

#include <string.h>

/** fp32 bit-field geometry (IEEE-754 binary32). */
#define FP32_EXPONENT_SHIFT 23u
#define FP32_EXPONENT_MASK 0xffu
#define FP32_FRACTION_MASK 0x7fffffu
#define FP32_IMPLICIT_ONE 0x800000u
#define FP32_SIGNIFICAND_BITS 24u

/** An "a" exponent is FP16-biased (15); adding this re-biases it to FP32 (127). */
#define EXPONENT_A_TO_B_REBIAS 112u
#define EXPONENT_A_MAX 31u

/** 32x32 tile walked as a 2x2 grid of 16x16 faces. */
#define TILE_SIDE 32u
#define FACE_SIDE 16u
#define TILE_VALUES (TILE_SIDE * TILE_SIDE)
#define TILE_BLOCKS (TILE_VALUES / FT_BFP_BLOCK_VALUES)

static uint64_t tie_count_seen;
static uint64_t tie_count_rounded_up;

/** Static description of one format. */
typedef struct {
    uint32_t magnitude_bits;     /**< datum bits below the sign bit */
    int uses_five_bit_exponent;  /**< 1 for the "a" flavour */
} bfp_format_shape;

static int lookup_format_shape(ft_bfp_format format, bfp_format_shape *shape)
{
    switch (format) {
    case FT_BFP_BFP8_B: shape->magnitude_bits = 7; shape->uses_five_bit_exponent = 0; return 1;
    case FT_BFP_BFP4_B: shape->magnitude_bits = 3; shape->uses_five_bit_exponent = 0; return 1;
    case FT_BFP_BFP8_A: shape->magnitude_bits = 7; shape->uses_five_bit_exponent = 1; return 1;
    case FT_BFP_BFP4_A: shape->magnitude_bits = 3; shape->uses_five_bit_exponent = 1; return 1;
    case FT_BFP_BFP6_B: shape->magnitude_bits = 5; shape->uses_five_bit_exponent = 0; return 1;
    case FT_BFP_BFP2_B: shape->magnitude_bits = 1; shape->uses_five_bit_exponent = 0; return 1;
    }
    return 0;
}

size_t ft_bfp_tile_bytes(ft_bfp_format format)
{
    bfp_format_shape shape;

    if (!lookup_format_shape(format, &shape))
        return 0;
    return FT_BFP_TILE_EXPONENT_BYTES + TILE_VALUES * (shape.magnitude_bits + 1u) / 8u;
}

uint32_t ft_bfp_mantissa_bits(ft_bfp_format format)
{
    bfp_format_shape shape;

    return lookup_format_shape(format, &shape) ? shape.magnitude_bits : 0;
}

/** The block exponent re-expressed with the FP32 bias, whatever the format. */
static uint32_t shared_exponent_as_fp32_bias(const bfp_format_shape *shape, uint32_t shared_exp)
{
    return shape->uses_five_bit_exponent ? shared_exp + EXPONENT_A_TO_B_REBIAS : shared_exp;
}

uint8_t ft_bfp_block_shared_exponent(ft_bfp_format format, const uint32_t *block_bits)
{
    bfp_format_shape shape;
    uint32_t largest = 0;

    if (!lookup_format_shape(format, &shape) || !block_bits)
        return 0;

    /* Zeros and denormals carry exponent field 0, so they never raise the max. */
    for (uint32_t i = 0; i < FT_BFP_BLOCK_VALUES; i++) {
        uint32_t exponent = (block_bits[i] >> FP32_EXPONENT_SHIFT) & FP32_EXPONENT_MASK;

        if (exponent > largest)
            largest = exponent;
    }

    if (!shape.uses_five_bit_exponent)
        return (uint8_t)largest;
    if (largest <= EXPONENT_A_TO_B_REBIAS)
        return 0;
    if (largest - EXPONENT_A_TO_B_REBIAS > EXPONENT_A_MAX)
        return (uint8_t)EXPONENT_A_MAX;
    return (uint8_t)(largest - EXPONENT_A_TO_B_REBIAS);
}

/**
 * Drop the low `shift` bits (1..24) of an aligned significand, rounding to
 * nearest with ties to even, and record any tie in the diagnostic counters.
 */
static uint32_t round_off_low_bits_ties_even(uint32_t aligned, uint32_t shift)
{
    uint32_t kept = aligned >> shift;
    uint32_t remainder = aligned & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1u);

    if (remainder > half)
        return kept + 1u;
    if (remainder == half) {
        tie_count_seen++;
        if (kept & 1u) {
            tie_count_rounded_up++;
            return kept + 1u;
        }
    }
    return kept;
}

/**
 * Encoding rules, each one pinned by the reference vectors:
 *  - zero and denormal (exponent field 0) give datum 0 with no sign;
 *  - for the "a" formats a value below the 5-bit range (fp32 exponent < 112)
 *    is first raised to the smallest 5-bit exponent with its fraction cleared,
 *    i.e. it becomes +-2^-15 before block alignment;
 *  - the significand (implicit one included) is aligned to the block exponent
 *    by a TRUNCATING right shift, and only then rounded (ties to even) to the
 *    datum width — bits lost in the alignment do not act as sticky bits;
 *  - a rounding carry past the largest magnitude saturates instead of carrying;
 *  - infinity and NaN are encoded like any other exponent-255 value;
 *  - a magnitude that rounds to 0 drops the sign.
 */
uint8_t ft_bfp_convert_float_bits(ft_bfp_format format, uint32_t input_bits, uint32_t shared_exp)
{
    bfp_format_shape shape;
    uint32_t exponent = (input_bits >> FP32_EXPONENT_SHIFT) & FP32_EXPONENT_MASK;
    uint32_t sign = input_bits >> 31;
    uint32_t block_exponent, max_magnitude, magnitude;

    if (!lookup_format_shape(format, &shape) || exponent == 0)
        return 0; /* zero and denormal pack to +0 */

    block_exponent = shared_exponent_as_fp32_bias(&shape, shared_exp);
    if (shape.uses_five_bit_exponent && exponent < EXPONENT_A_TO_B_REBIAS) {
        exponent = EXPONENT_A_TO_B_REBIAS;
        input_bits &= ~FP32_FRACTION_MASK;
    }
    max_magnitude = (1u << shape.magnitude_bits) - 1u;

    if (exponent > block_exponent) {
        magnitude = max_magnitude; /* larger than the block can hold: saturate */
    } else {
        uint32_t significand = (input_bits & FP32_FRACTION_MASK) | FP32_IMPLICIT_ONE;
        uint32_t alignment = block_exponent - exponent;
        uint32_t aligned = alignment >= 32u ? 0u : significand >> alignment;

        magnitude = round_off_low_bits_ties_even(aligned, FP32_SIGNIFICAND_BITS - shape.magnitude_bits);
        if (magnitude > max_magnitude)
            magnitude = max_magnitude;
    }

    if (magnitude == 0)
        return 0;
    return (uint8_t)((sign << shape.magnitude_bits) | magnitude);
}

/**
 * Decode one datum the way FloatBitPatterns.md's BFP8ToBF16 does, widened to
 * fp32: normalise the magnitude so its leading one becomes the implicit bit,
 * lower the exponent by the shift, and keep the remaining bits as fraction.
 * The exponent arithmetic wraps in 32 bits exactly as the reference does, so
 * an underflowing exponent spills into the sign bit rather than clamping, and
 * a sign-only datum decodes to -0 (the document lists -2^128 / -2^16 for it).
 */
uint32_t ft_bfp_convert_datum(ft_bfp_format format, uint8_t datum, uint32_t shared_exp)
{
    bfp_format_shape shape;
    uint32_t sign, magnitude, exponent, leading_zeros = 0;

    if (!lookup_format_shape(format, &shape))
        return 0;

    /* Widen a narrower datum to the BFP8 position, as BFP4ToBF16 does: sign to bit 7 and
     * the magnitude's top bit to bit 6. Exact, so every width shares the BFP8 decode. */
    if (shape.magnitude_bits < 7u) {
        uint32_t width = shape.magnitude_bits + 1u;

        datum = (uint8_t)((datum & ((1u << width) - 1u)) << (8u - width));
    }

    sign = (uint32_t)datum >> 7;
    magnitude = (uint32_t)datum & 0x7fu;
    if (magnitude == 0)
        return sign << 31; /* +0, or -0 for a sign-only datum */

    while ((magnitude & 0x40u) == 0) {
        magnitude <<= 1;
        leading_zeros++;
    }
    exponent = shared_exponent_as_fp32_bias(&shape, shared_exp) - leading_zeros;

    return (sign << 31) | (exponent << FP32_EXPONENT_SHIFT) | ((magnitude & 0x3fu) << 17);
}

/** Row-major offset inside a tile of the first value of packed block `block`. */
static uint32_t block_origin_in_tile(uint32_t block)
{
    uint32_t face = block / FACE_SIDE;
    uint32_t face_row = block % FACE_SIDE;
    uint32_t tile_row = (face / 2u) * FACE_SIDE + face_row;
    uint32_t tile_col = (face % 2u) * FACE_SIDE;

    return tile_row * TILE_SIDE + tile_col;
}

/** Datum bit offset of `index` in a 6-bit little-endian stream. */
#define BFP6_STREAM_BITS 6u

/** Store datum `index` of a tile's datum stream (see ft_bfp.h for each width's order). */
static void store_datum(unsigned char *datums, uint32_t index, uint8_t datum, uint32_t magnitude_bits)
{
    uint32_t width = magnitude_bits + 1u;

    if (width == 8u) {
        datums[index] = datum;
    } else if (width == BFP6_STREAM_BITS) {
        uint32_t bit = index * BFP6_STREAM_BITS;
        uint32_t pair = (uint32_t)(datum & 0x3fu) << (bit % 8u);

        datums[bit / 8u] = (unsigned char)((datums[bit / 8u] & ~(0x3fu << (bit % 8u))) | pair);
        if (bit % 8u > 2u)
            datums[bit / 8u + 1u] = (unsigned char)((datums[bit / 8u + 1u] & ~(0x3fu >> (8u - bit % 8u))) |
                                                    (pair >> 8));
    } else {
        uint32_t per_byte = 8u / width;
        uint32_t shift = (index % per_byte) * width;
        uint32_t mask = ((1u << width) - 1u) << shift;

        datums[index / per_byte] =
            (unsigned char)((datums[index / per_byte] & ~mask) | (((uint32_t)datum << shift) & mask));
    }
}

static uint8_t load_datum(const unsigned char *datums, uint32_t index, uint32_t magnitude_bits)
{
    uint32_t width = magnitude_bits + 1u;

    if (width == 8u)
        return datums[index];
    if (width == BFP6_STREAM_BITS) {
        uint32_t bit = index * BFP6_STREAM_BITS;
        uint32_t window = datums[bit / 8u];

        if (bit % 8u > 2u)
            window |= (uint32_t)datums[bit / 8u + 1u] << 8;
        return (uint8_t)((window >> (bit % 8u)) & 0x3fu);
    }
    return (uint8_t)((datums[index / (8u / width)] >> ((index % (8u / width)) * width)) & ((1u << width) - 1u));
}

static void pack_one_tile(ft_bfp_format format, const bfp_format_shape *shape, const float *tile, unsigned char *out)
{
    unsigned char *datums = out + FT_BFP_TILE_EXPONENT_BYTES;

    memset(datums, 0, TILE_VALUES * (shape->magnitude_bits + 1u) / 8u);

    for (uint32_t block = 0; block < TILE_BLOCKS; block++) {
        const float *source = tile + block_origin_in_tile(block);
        uint32_t block_bits[FT_BFP_BLOCK_VALUES];
        uint8_t shared;

        memcpy(block_bits, source, sizeof(block_bits));
        shared = ft_bfp_block_shared_exponent(format, block_bits);
        out[block] = shared;
        for (uint32_t i = 0; i < FT_BFP_BLOCK_VALUES; i++) {
            uint8_t datum = ft_bfp_convert_float_bits(format, block_bits[i], shared);

            store_datum(datums, block * FT_BFP_BLOCK_VALUES + i, datum, shape->magnitude_bits);
        }
    }
}

/**
 * ft_bfp_convert_datum() split into its datum half and its exponent half, for the unpack hot
 * path: for every datum value, `base` = sign | fraction bits and `shift` = its leading zeros;
 * the decoded bits are then `base | ((E - shift) << 23)` (E the block's fp32-biased exponent),
 * or `base` alone for a zero magnitude — the same expression ft_bfp_convert_datum() evaluates,
 * so the output is bit-identical (test_bfp / test_bfp6 compare every datum and exponent).
 */
typedef struct {
    uint32_t base[256];
    uint32_t shift[256];
    uint8_t zero[256];
} bfp_datum_table;

static void datum_table_build(const bfp_format_shape *shape, bfp_datum_table *table)
{
    for (uint32_t raw = 0; raw < 256u; raw++) {
        uint32_t datum = raw, sign, magnitude, leading_zeros = 0;

        if (shape->magnitude_bits < 7u) {
            uint32_t width = shape->magnitude_bits + 1u;

            datum = (raw & ((1u << width) - 1u)) << (8u - width) & 0xffu;
        }
        sign = datum >> 7;
        magnitude = datum & 0x7fu;
        table->zero[raw] = magnitude == 0u;
        while (magnitude && (magnitude & 0x40u) == 0) {
            magnitude <<= 1;
            leading_zeros++;
        }
        table->base[raw] = (sign << 31) | (magnitude ? ((magnitude & 0x3fu) << 17) : 0u);
        table->shift[raw] = leading_zeros;
    }
}

static void unpack_one_tile(const bfp_format_shape *shape, const bfp_datum_table *table,
                            const unsigned char *packed, float *tile)
{
    const unsigned char *datums = packed + FT_BFP_TILE_EXPONENT_BYTES;

    for (uint32_t block = 0; block < TILE_BLOCKS; block++) {
        float *target = tile + block_origin_in_tile(block);
        const uint32_t exponent = shared_exponent_as_fp32_bias(shape, packed[block]);

        for (uint32_t i = 0; i < FT_BFP_BLOCK_VALUES; i++) {
            uint8_t datum = load_datum(datums, block * FT_BFP_BLOCK_VALUES + i, shape->magnitude_bits);
            uint32_t bits = table->zero[datum] ? table->base[datum]
                                               : table->base[datum] |
                                                     ((exponent - table->shift[datum]) << FP32_EXPONENT_SHIFT);

            memcpy(&target[i], &bits, sizeof(bits));
        }
    }
}

int ft_bfp_pack_tiles(ft_bfp_format format, const float *tiles, size_t tile_count, void *out)
{
    bfp_format_shape shape;
    size_t tile_bytes;

    if (!lookup_format_shape(format, &shape))
        return FT_BFP_EINVAL;
    if (tile_count == 0)
        return FT_BFP_OK;
    if (!tiles || !out)
        return FT_BFP_EINVAL;

    tile_bytes = ft_bfp_tile_bytes(format);
    for (size_t t = 0; t < tile_count; t++)
        pack_one_tile(format, &shape, tiles + t * TILE_VALUES, (unsigned char *)out + t * tile_bytes);
    return FT_BFP_OK;
}

int ft_bfp_unpack_tiles(ft_bfp_format format, const void *packed, size_t tile_count, float *tiles_out)
{
    bfp_format_shape shape;
    size_t tile_bytes;

    if (!lookup_format_shape(format, &shape))
        return FT_BFP_EINVAL;
    if (tile_count == 0)
        return FT_BFP_OK;
    if (!packed || !tiles_out)
        return FT_BFP_EINVAL;

    tile_bytes = ft_bfp_tile_bytes(format);
    {
        bfp_datum_table table;

        datum_table_build(&shape, &table);
        for (size_t t = 0; t < tile_count; t++)
            unpack_one_tile(&shape, &table, (const unsigned char *)packed + t * tile_bytes,
                            tiles_out + t * TILE_VALUES);
    }
    return FT_BFP_OK;
}

void ft_bfp_tie_counters(uint64_t *ties_seen, uint64_t *ties_rounded_up)
{
    if (ties_seen)
        *ties_seen = tie_count_seen;
    if (ties_rounded_up)
        *ties_rounded_up = tie_count_rounded_up;
}

void ft_bfp_tie_counters_reset(void)
{
    tie_count_seen = 0;
    tie_count_rounded_up = 0;
}

/* ================================================================
 * BFP6_B: datum layouts and the exact expansion to BFP8_B
 * ================================================================ */

#define BFP6_TILE_BYTES (FT_BFP_TILE_EXPONENT_BYTES + TILE_VALUES * 6u / 8u)
#define BFP8_TILE_BYTES (FT_BFP_TILE_EXPONENT_BYTES + TILE_VALUES)
#define BFP6_PLANE_HIGH_BYTES (TILE_VALUES / 2u)

/** Datum `index` (sign + 5-bit magnitude) of one tile's datum bytes in `layout`. */
/** FOLD: datum `index` lives in group index / 16 (12 bytes); its 16-byte BFP8 group's word
 *  (index % 16) / 4 and byte index % 4. Words 0..2 keep the datum in the top six bits of
 *  their byte; word 3's datum is split over the low two bits of the same byte of words 0..2. */
static uint8_t bfp6_fold_load(const unsigned char *datums, uint32_t index)
{
    const unsigned char *group = datums + (index / 16u) * 12u;
    const uint32_t word = (index % 16u) / 4u, byte = index % 4u;

    if (word < 3u)
        return (uint8_t)(group[word * 4u + byte] >> 2);
    return (uint8_t)(((group[byte] & 3u) << 4) | ((group[4u + byte] & 3u) << 2) |
                     (group[8u + byte] & 3u));
}

static void bfp6_fold_store(unsigned char *datums, uint32_t index, uint8_t datum)
{
    unsigned char *group = datums + (index / 16u) * 12u;
    const uint32_t word = (index % 16u) / 4u, byte = index % 4u;

    if (word < 3u) {
        group[word * 4u + byte] = (unsigned char)((group[word * 4u + byte] & 3u) | ((uint32_t)datum << 2));
        return;
    }
    for (uint32_t w = 0; w < 3u; w++)
        group[w * 4u + byte] = (unsigned char)((group[w * 4u + byte] & 0xFCu) |
                                               (((uint32_t)datum >> (4u - 2u * w)) & 3u));
}

static uint8_t bfp6_load(ft_bfp6_layout layout, const unsigned char *datums, uint32_t index)
{
    uint32_t high, low;

    if (layout == FT_BFP6_LAYOUT_STREAM)
        return load_datum(datums, index, 5u);
    if (layout == FT_BFP6_LAYOUT_FOLD)
        return bfp6_fold_load(datums, index);
    high = (datums[index / 2u] >> ((index % 2u) * 4u)) & 0x0fu;
    low = (datums[BFP6_PLANE_HIGH_BYTES + index / 4u] >> ((index % 4u) * 2u)) & 0x03u;
    return (uint8_t)((high << 2) | low);
}

static void bfp6_store(ft_bfp6_layout layout, unsigned char *datums, uint32_t index, uint8_t datum)
{
    unsigned char *high, *low;

    if (layout == FT_BFP6_LAYOUT_STREAM) {
        store_datum(datums, index, datum, 5u);
        return;
    }
    if (layout == FT_BFP6_LAYOUT_FOLD) {
        bfp6_fold_store(datums, index, datum);
        return;
    }
    high = &datums[index / 2u];
    low = &datums[BFP6_PLANE_HIGH_BYTES + index / 4u];
    *high = (unsigned char)((*high & ~(0x0fu << ((index % 2u) * 4u))) | (((uint32_t)datum >> 2) << ((index % 2u) * 4u)));
    *low = (unsigned char)((*low & ~(0x03u << ((index % 4u) * 2u))) | (((uint32_t)datum & 3u) << ((index % 4u) * 2u)));
}

static int bfp6_layout_valid(ft_bfp6_layout layout)
{
    return layout == FT_BFP6_LAYOUT_STREAM || layout == FT_BFP6_LAYOUT_PLANES ||
           layout == FT_BFP6_LAYOUT_FOLD;
}

int ft_bfp6_relayout_tiles(ft_bfp6_layout from, ft_bfp6_layout to, const void *in,
                           size_t tile_count, void *out)
{
    if (!bfp6_layout_valid(from) || !bfp6_layout_valid(to))
        return FT_BFP_EINVAL;
    if (tile_count == 0)
        return FT_BFP_OK;
    if (!in || !out)
        return FT_BFP_EINVAL;
    for (size_t t = 0; t < tile_count; t++) {
        const unsigned char *source = (const unsigned char *)in + t * BFP6_TILE_BYTES;
        unsigned char *target = (unsigned char *)out + t * BFP6_TILE_BYTES;

        memcpy(target, source, FT_BFP_TILE_EXPONENT_BYTES);
        memset(target + FT_BFP_TILE_EXPONENT_BYTES, 0, BFP6_TILE_BYTES - FT_BFP_TILE_EXPONENT_BYTES);
        for (uint32_t i = 0; i < TILE_VALUES; i++)
            bfp6_store(to, target + FT_BFP_TILE_EXPONENT_BYTES, i,
                       bfp6_load(from, source + FT_BFP_TILE_EXPONENT_BYTES, i));
    }
    return FT_BFP_OK;
}

int ft_bfp6_expand_tiles(ft_bfp6_layout layout, const void *in, size_t tile_count, void *out)
{
    if (!bfp6_layout_valid(layout))
        return FT_BFP_EINVAL;
    if (tile_count == 0)
        return FT_BFP_OK;
    if (!in || !out)
        return FT_BFP_EINVAL;
    for (size_t t = 0; t < tile_count; t++) {
        const unsigned char *source = (const unsigned char *)in + t * BFP6_TILE_BYTES;
        unsigned char *target = (unsigned char *)out + t * BFP8_TILE_BYTES;

        memcpy(target, source, FT_BFP_TILE_EXPONENT_BYTES);
        for (uint32_t i = 0; i < TILE_VALUES; i++)
            target[FT_BFP_TILE_EXPONENT_BYTES + i] =
                (unsigned char)(bfp6_load(layout, source + FT_BFP_TILE_EXPONENT_BYTES, i) << 2);
    }
    return FT_BFP_OK;
}

static uint32_t bfp6_load32le(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void bfp6_store32le(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}

/** Four stream datums (bits 0..23 of x) -> one BFP8 word (each datum << 2 in its byte). */
static uint32_t bfp6_stream_word_to_bfp8(uint32_t x)
{
    x = ((x & 0x00FFF000u) << 4) | (x & 0x00000FFFu);
    x = ((x & 0x0FC00FC0u) << 2) | (x & 0x003F003Fu);
    return x << 2;
}

int ft_bfp6_fold_from_stream(const void *in, size_t tile_count, void *out)
{
    const uint32_t mask = 0x03030303u;

    if (tile_count == 0)
        return FT_BFP_OK;
    if (!in || !out)
        return FT_BFP_EINVAL;
    for (size_t t = 0; t < tile_count; t++) {
        const unsigned char *source = (const unsigned char *)in + t * BFP6_TILE_BYTES;
        unsigned char *target = (unsigned char *)out + t * BFP6_TILE_BYTES;

        memcpy(target, source, FT_BFP_TILE_EXPONENT_BYTES);
        source += FT_BFP_TILE_EXPONENT_BYTES;
        target += FT_BFP_TILE_EXPONENT_BYTES;
        for (uint32_t group = 0; group < TILE_VALUES / 16u; group++, source += 12, target += 12) {
            const uint32_t w0 = bfp6_load32le(source), w1 = bfp6_load32le(source + 4),
                           w2 = bfp6_load32le(source + 8);
            const uint32_t o0 = bfp6_stream_word_to_bfp8(w0 & 0x00FFFFFFu);
            const uint32_t o1 = bfp6_stream_word_to_bfp8((w0 >> 24) | ((w1 & 0x0000FFFFu) << 8));
            const uint32_t o2 = bfp6_stream_word_to_bfp8((w1 >> 16) | ((w2 & 0x000000FFu) << 16));
            const uint32_t o3 = bfp6_stream_word_to_bfp8(w2 >> 8);

            bfp6_store32le(target, o0 | ((o3 >> 6) & mask));
            bfp6_store32le(target + 4, o1 | ((o3 >> 4) & mask));
            bfp6_store32le(target + 8, o2 | ((o3 >> 2) & mask));
        }
    }
    return FT_BFP_OK;
}
