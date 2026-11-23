/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_kv_block_float.h
 * @brief Block-float KV rows: one face row (16 bf16 values) to a Bfp8_b / Bfp4_b exponent
 *   byte plus datums, and back. Integer-only and header-only, so the KV append writer
 *   (NCRISC, resident_place_writer.c) and the host (export/import, tests) share ONE
 *   definition of the cache's numerics.
 *
 * WHY A KV ROW CAN BE PACKED BY A COPY KERNEL (ft_tt_op_memory.h §1.6). A face row of a K or
 * V tile is 16 head dimensions of ONE position (F1: position on the tile's H axis). Its
 * shared exponent therefore covers only that position, so an append packs the new rows and
 * never re-scales an older one. The writer has the rows in bf16 in its CB page; packing 32
 * values is a few hundred integer instructions.
 *
 * NUMERICS: bit-exact with convert/ft_bfp.c (the weight packer, whose bytes the unpacker
 * already reads on every matmul) on bf16 inputs: shared exponent = the largest exponent
 * field of the 16; the significand (implicit one included) aligned to it by a TRUNCATING
 * right shift, then rounded to the datum width nearest-even, saturating; zero, denormal and
 * a magnitude that rounds to 0 give datum 0 with no sign. tests/test_kv_block_float.c pins it.
 *
 * TILE LAYOUT (ft_bfp.h): 64 exponent bytes (face f row r at f*16 + r), then the datums in
 * the same face-row order: 16 bytes a face row for Bfp8_b, 8 for Bfp4_b (low nibble first).
 *
 * @status Host-tested bit-exact vs ft_bfp.c; on silicon through the KV append.
 * @issues None
 * @todo None
 */

#ifndef FT_KV_BLOCK_FLOAT_H
#define FT_KV_BLOCK_FLOAT_H

#include <stdint.h>

#ifndef FT_KV_BF_INLINE
#ifdef __cplusplus
#define FT_KV_BF_INLINE inline
#else
#define FT_KV_BF_INLINE static inline
#endif
#endif

/** KV element formats. The values are the wire encoding (memory flags, Pantheon's knob). */
enum { kKvFormatBf16 = 0, kKvFormatBfp8 = 1, kKvFormatBfp4 = 2 };

enum { kKvBfRowValues = 16 };     /* values sharing one exponent: one face row */
enum { kKvBfExponentBytes = 64 }; /* exponent section at the head of every tile */
enum { kKvBfBf16TileBytes = 2048 };
enum { kKvBfBfp8TileBytes = 1088 };
enum { kKvBfBfp4TileBytes = 576 };

/** Bytes of one 32x32 tile in `format`; 0 for an unknown format. */
FT_KV_BF_INLINE uint32_t ft_kv_bf_tile_bytes(uint32_t format)
{
    return format == kKvFormatBf16   ? (uint32_t)kKvBfBf16TileBytes
           : format == kKvFormatBfp8 ? (uint32_t)kKvBfBfp8TileBytes
           : format == kKvFormatBfp4 ? (uint32_t)kKvBfBfp4TileBytes
                                     : 0u;
}

/** Magnitude bits of one datum (sign excluded): 7 Bfp8_b, 3 Bfp4_b. */
FT_KV_BF_INLINE uint32_t ft_kv_bf_magnitude_bits(uint32_t format)
{
    return format == kKvFormatBfp4 ? 3u : 7u;
}

/** Datum bytes of one face row: 16 Bfp8_b, 8 Bfp4_b. */
FT_KV_BF_INLINE uint32_t ft_kv_bf_row_datum_bytes(uint32_t format)
{
    return format == kKvFormatBfp4 ? 8u : 16u;
}

/** Byte offset, inside a packed tile, of face `face` row `face_row`'s exponent. */
FT_KV_BF_INLINE uint32_t ft_kv_bf_exponent_offset(uint32_t face, uint32_t face_row)
{
    return face * 16u + face_row;
}

/** Byte offset, inside a packed tile, of face `face` row `face_row`'s first datum. */
FT_KV_BF_INLINE uint32_t ft_kv_bf_datum_offset(uint32_t format, uint32_t face, uint32_t face_row)
{
    const uint32_t row_bytes = ft_kv_bf_row_datum_bytes(format);

    return (uint32_t)kKvBfExponentBytes + (face * 16u + face_row) * row_bytes;
}

/** One bf16 value as a datum under `shared_exponent` (ft_bfp_convert_float_bits on bf16). */
FT_KV_BF_INLINE uint32_t ft_kv_bf_datum_from_bf16(uint32_t bits, uint32_t shared_exponent,
                                                  uint32_t magnitude_bits)
{
    const uint32_t exponent = (bits >> 7) & 0xFFu;
    const uint32_t drop = 24u - magnitude_bits;
    const uint32_t largest = (1u << magnitude_bits) - 1u;
    uint32_t aligned, kept, remainder, half;

    if (exponent == 0u)
        return 0u;
    aligned = shared_exponent - exponent >= 24u
                  ? 0u
                  : (((bits & 0x7Fu) | 0x80u) << 16) >> (shared_exponent - exponent);
    kept = aligned >> drop;
    remainder = aligned & ((1u << drop) - 1u);
    half = 1u << (drop - 1u);
    if (remainder > half || (remainder == half && (kept & 1u) != 0u))
        kept++;
    if (kept > largest)
        kept = largest;
    return kept == 0u ? 0u : (((bits >> 15) & 1u) << magnitude_bits) | kept;
}

/**
 * Pack 16 bf16 values (one face row) into `datums` (16 or 8 bytes) and return the shared
 * exponent byte.
 */
FT_KV_BF_INLINE uint32_t ft_kv_bf_pack_row(uint32_t format, const uint16_t *values,
                                           uint8_t *datums)
{
    const uint32_t magnitude_bits = ft_kv_bf_magnitude_bits(format);
    uint32_t shared = 0u;

    for (uint32_t i = 0; i < (uint32_t)kKvBfRowValues; i++) {
        const uint32_t exponent = ((uint32_t)values[i] >> 7) & 0xFFu;

        if (exponent > shared)
            shared = exponent;
    }
    for (uint32_t i = 0; i < (uint32_t)kKvBfRowValues; i++) {
        const uint32_t datum = ft_kv_bf_datum_from_bf16(values[i], shared, magnitude_bits);

        if (format == kKvFormatBfp4) {
            if ((i & 1u) == 0u)
                datums[i >> 1] = (uint8_t)datum;
            else
                datums[i >> 1] = (uint8_t)(datums[i >> 1] | (datum << 4));
        } else {
            datums[i] = (uint8_t)datum;
        }
    }
    return shared;
}

/** One datum back to fp32 bits, as the unpacker reads it (ft_bfp_convert_datum). */
FT_KV_BF_INLINE uint32_t ft_kv_bf_datum_to_fp32(uint32_t format, uint32_t datum,
                                                uint32_t shared_exponent)
{
    uint32_t sign, magnitude, leading_zeros = 0u;

    if (format == kKvFormatBfp4)
        datum = (datum & 0xFu) << 4;
    sign = (datum >> 7) & 1u;
    magnitude = datum & 0x7Fu;
    if (magnitude == 0u)
        return sign << 31;
    while ((magnitude & 0x40u) == 0u) {
        magnitude <<= 1;
        leading_zeros++;
    }
    return (sign << 31) | ((shared_exponent - leading_zeros) << 23) | ((magnitude & 0x3Fu) << 17);
}

/** Unpack one face row of a packed tile to 16 fp32 bit patterns. */
FT_KV_BF_INLINE void ft_kv_bf_unpack_row(uint32_t format, const uint8_t *tile, uint32_t face,
                                         uint32_t face_row, uint32_t *out_bits)
{
    const uint32_t shared = tile[ft_kv_bf_exponent_offset(face, face_row)];
    const uint8_t *datums = tile + ft_kv_bf_datum_offset(format, face, face_row);

    for (uint32_t i = 0; i < (uint32_t)kKvBfRowValues; i++) {
        const uint32_t datum =
            format == kKvFormatBfp4 ? (uint32_t)(datums[i >> 1] >> ((i & 1u) * 4u)) & 0xFu
                                    : (uint32_t)datums[i];

        out_bits[i] = ft_kv_bf_datum_to_fp32(format, datum, shared);
    }
}

/**
 * Pack tile rows [row0, row0 + row_count) of a face-ordered bf16 tile into a packed tile
 * in place: those rows' exponents and datums are replaced, every other byte is kept.
 */
FT_KV_BF_INLINE void ft_kv_bf_pack_tile_rows(uint32_t format, const uint16_t *bf16_tile,
                                             uint8_t *packed_tile, uint32_t row0,
                                             uint32_t row_count)
{
    for (uint32_t row = row0; row < row0 + row_count; row++) {
        for (uint32_t half = 0; half < 2u; half++) {
            const uint32_t face = (row / 16u) * 2u + half;
            const uint32_t face_row = row % 16u;
            const uint16_t *values = bf16_tile + face * 256u + face_row * 16u;

            packed_tile[ft_kv_bf_exponent_offset(face, face_row)] = (uint8_t)ft_kv_bf_pack_row(
                format, values, packed_tile + ft_kv_bf_datum_offset(format, face, face_row));
        }
    }
}

/* ===== Whole-tile pack (a prefill append that owns all 32 rows) =========================
 *
 * SAME BYTES AS ft_kv_bf_pack_tile_rows(format, tile, packed, 0, 32), in word-wide integer
 * code (the per-row loop above costs ~36 NCRISC instructions a value). The datum rule above,
 * rewritten for shared >= 9: with m8 = the significand incl. the implicit one and
 * bias = shared - magnitude_bits - 1, the magnitude is RNE(x / 2^9) for
 * x = m8 << max(exponent - bias, 0), saturated. The shift is at most magnitude_bits + 1, so x
 * fits 16 bits; a clamped shift leaves x = m8 < 2^8, which rounds to 0 -- exactly the values
 * (zero, denormal, or more than 2^(magnitude_bits+1) below the row max) the rule sends to 0.
 * A row whose shared exponent is below 9 (all zero, or every value below 2^-118) takes the
 * scalar ft_kv_bf_pack_row. tests/test_kv_block_float_tile.c checks the rule exhaustively
 * (every bf16 x every shared exponent) and the tile bytes against ft_kv_bf_pack_tile_rows.
 *
 * WORD ORDER: a face-ordered tile's face rows are consecutive (face f row r is face row
 * f*16 + r: 8 bf16 words), and so are their exponent bytes and datums, so a tile is 64 rows
 * walked in order. Little-endian words, as both the host and the RISC-V cores are. */

enum { kKvBfFastMinShared = 9 };

/* -Os outlines the 15-instruction datum below (a call per value); forced inline it is. */
#if defined(__GNUC__) && !defined(__cplusplus)
#define FT_KV_BF_HOT static inline __attribute__((always_inline))
#else
#define FT_KV_BF_HOT FT_KV_BF_INLINE
#endif

/** Datum magnitude of the bf16 in the low 16 bits of `bits` (higher bits ignored). */
FT_KV_BF_HOT uint32_t ft_kv_bf_fast_magnitude(uint32_t bits, int32_t bias,
                                                 uint32_t magnitude_bits)
{
    int32_t shift = (int32_t)((bits >> 7) & 0xFFu) - bias;
    uint32_t x;

    if (shift < 0)
        shift = 0;
    x = ((bits & 0x7Fu) | 0x80u) << shift;
    x = (x + 255u + ((x >> 9) & 1u)) >> 9;
    return x - (x >> magnitude_bits); /* the one overflow, 2^magnitude_bits, saturates */
}

/** Shared exponent of one face row (8 words, 16 bf16): the largest exponent field. */
FT_KV_BF_INLINE uint32_t ft_kv_bf_row_shared_words(const uint32_t *words)
{
    uint32_t high = 0u, low = 0u;

    for (uint32_t i = 0; i < 8u; i++) {
        const uint32_t fields = words[i] & 0x7F807F80u;

        high = fields > high ? fields : high; /* the high half decides the compare */
        low = (fields << 16) > low ? (fields << 16) : low;
    }
    high >>= 16;
    low >>= 16;
    return (high > low ? high : low) >> 7;
}

/** A row below the fast path's range: the scalar packer, its datums repacked into words. */
FT_KV_BF_INLINE uint32_t ft_kv_bf_pack_row_words_scalar(uint32_t format, const uint32_t *words,
                                                        uint32_t *datum_words)
{
    uint16_t values[kKvBfRowValues];
    uint8_t datums[16];
    const uint32_t count = ft_kv_bf_row_datum_bytes(format) / 4u;
    uint32_t shared;

    for (uint32_t i = 0; i < 8u; i++) {
        values[2u * i] = (uint16_t)words[i];
        values[2u * i + 1u] = (uint16_t)(words[i] >> 16);
    }
    shared = ft_kv_bf_pack_row(format, values, datums);
    for (uint32_t i = 0; i < count; i++)
        datum_words[i] = (uint32_t)datums[4u * i] | ((uint32_t)datums[4u * i + 1u] << 8) |
                         ((uint32_t)datums[4u * i + 2u] << 16) |
                         ((uint32_t)datums[4u * i + 3u] << 24);
    return shared;
}

/** Bfp8_b face row: 4 datum words; returns the exponent byte. */
FT_KV_BF_INLINE uint32_t ft_kv_bf_pack_row_bfp8_words(const uint32_t *words,
                                                      uint32_t *datum_words)
{
    const uint32_t shared = ft_kv_bf_row_shared_words(words);
    int32_t bias;

    if (shared < (uint32_t)kKvBfFastMinShared)
        return ft_kv_bf_pack_row_words_scalar(kKvFormatBfp8, words, datum_words);
    bias = (int32_t)shared - 8;
    for (uint32_t q = 0; q < 4u; q++) {
        const uint32_t a = words[2u * q], b = words[2u * q + 1u];
        /* Datum = sign (bit 7) | magnitude; the sign is cleared below where the magnitude
         * is 0 (zero carries no sign). */
        const uint32_t d =
            (ft_kv_bf_fast_magnitude(a, bias, 7u) | ((a >> 8) & 0x80u)) |
            ((ft_kv_bf_fast_magnitude(a >> 16, bias, 7u) | ((a >> 24) & 0x80u)) << 8) |
            ((ft_kv_bf_fast_magnitude(b, bias, 7u) | ((b >> 8) & 0x80u)) << 16) |
            ((ft_kv_bf_fast_magnitude(b >> 16, bias, 7u) | ((b >> 24) & 0x80u)) << 24);
        const uint32_t magnitude = d & 0x7F7F7F7Fu;
        const uint32_t nonzero = (magnitude + 0x7F7F7F7Fu) & 0x80808080u;

        datum_words[q] = magnitude | (d & nonzero);
    }
    return shared;
}

/** Bfp4_b face row: 2 datum words (low nibble first); returns the exponent byte. */
FT_KV_BF_INLINE uint32_t ft_kv_bf_pack_row_bfp4_words(const uint32_t *words,
                                                      uint32_t *datum_words)
{
    const uint32_t shared = ft_kv_bf_row_shared_words(words);
    int32_t bias;

    if (shared < (uint32_t)kKvBfFastMinShared)
        return ft_kv_bf_pack_row_words_scalar(kKvFormatBfp4, words, datum_words);
    bias = (int32_t)shared - 4;
    for (uint32_t q = 0; q < 2u; q++) {
        uint32_t d = 0u, magnitude, nonzero;

        for (uint32_t k = 0; k < 4u; k++) {
            const uint32_t w = words[4u * q + k];
            const uint32_t pair =
                (ft_kv_bf_fast_magnitude(w, bias, 3u) | ((w >> 12) & 0x8u)) |
                ((ft_kv_bf_fast_magnitude(w >> 16, bias, 3u) | ((w >> 28) & 0x8u)) << 4);

            d |= pair << (8u * k);
        }
        magnitude = d & 0x77777777u;
        nonzero = (magnitude + 0x77777777u) & 0x88888888u;
        datum_words[q] = magnitude | (d & nonzero);
    }
    return shared;
}

/**
 * Pack a whole face-ordered bf16 tile (512 words) into a Bfp8_b / Bfp4_b tile (272 / 144
 * words): every exponent and datum is written, nothing of the destination is read.
 */
FT_KV_BF_INLINE void ft_kv_bf_pack_tile_words(uint32_t format, const uint32_t *bf16_words,
                                              uint32_t *packed_words)
{
    uint32_t *datums = packed_words + (uint32_t)kKvBfExponentBytes / 4u;

    for (uint32_t row = 0; row < 64u; row += 4u) {
        uint32_t exponents = 0u;

        if (format == kKvFormatBfp4) {
            for (uint32_t k = 0; k < 4u; k++, bf16_words += 8, datums += 2)
                exponents |= ft_kv_bf_pack_row_bfp4_words(bf16_words, datums) << (8u * k);
        } else {
            for (uint32_t k = 0; k < 4u; k++, bf16_words += 8, datums += 4)
                exponents |= ft_kv_bf_pack_row_bfp8_words(bf16_words, datums) << (8u * k);
        }
        packed_words[row / 4u] = exponents;
    }
}

#endif /* FT_KV_BLOCK_FLOAT_H */
