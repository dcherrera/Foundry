/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_kv_block_float_tile.c
 * @brief Off-hardware gate for the whole-tile KV pack (ft_kv_block_float.h,
 *   ft_kv_bf_pack_tile_words): its bytes must equal the per-row packer's
 *   (ft_kv_bf_pack_tile_rows), the rule the cache's quality gates were measured on.
 *
 *   1. the fast datum rule against ft_kv_bf_datum_from_bf16 for EVERY bf16 and every shared
 *      exponent of the fast range (9..255), both formats;
 *   2. the shared exponent and the face-row pack against ft_kv_bf_pack_row on random and
 *      edge rows (zero rows, denormals, shared < 9, ties, saturation, spread exponents);
 *   3. whole tiles against ft_kv_bf_pack_tile_rows(.., 0, 32): random tiles, plus real K/V
 *      tiles when a fixture is given (raw face-ordered bf16 tiles, 2048 B each: item 6.1's
 *      dump_tiles_short.bin);
 *   4. append sequences, the writer's two paths: a cache filled by appends (aligned bands,
 *      single rows, chunks starting/ending mid-tile, a chunk crossing tiles, a whole-tile
 *      descriptor with fewer rows) where a piece covering a whole tile takes the whole-tile
 *      pack and every other piece the per-row read-modify-write, against the same appends
 *      all per-row; K Bfp8_b and V Bfp4_b side by side.
 *
 * Build: make -f Makefile.kvtile && ./build-kvtile/test_kv_block_float_tile [tiles.bin]
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels/ft_kv_block_float.h"

static int g_checks;
static int g_failures;

#define CHECK(condition, ...)                                                                   \
    do {                                                                                        \
        g_checks++;                                                                             \
        if (!(condition)) {                                                                     \
            g_failures++;                                                                       \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                \
            fprintf(stderr, __VA_ARGS__);                                                       \
            fputc('\n', stderr);                                                                \
        }                                                                                       \
    } while (0)

enum { kTileValues = 1024, kTileWords = 512, kPackedWordsMax = 272 };

static const uint32_t k_formats[2] = { kKvFormatBfp8, kKvFormatBfp4 };
static const char *const k_names[2] = { "bfp8", "bfp4" };

static uint32_t g_rng = 0x9E3779B9u;

static uint32_t next_random(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

/** One bf16 of row kind `kind` around exponent `center`. */
static uint16_t row_value(uint32_t kind, uint32_t center, uint32_t index)
{
    const uint32_t r = next_random();
    const uint32_t sign = (r & 1u) << 15;
    uint32_t exponent;

    switch (kind) {
    case 0: /* K/V-like: exponents within ~2^-8 of the row centre */
        exponent = center - (r >> 1) % 9u;
        break;
    case 1: /* wide spread: many values clamp to 0 */
        exponent = center - (r >> 1) % 40u;
        break;
    case 2: /* all zero (shared 0: the scalar row) */
        return (uint16_t)(((r & 3u) == 0u) ? sign : 0u);
    case 3: /* denormals and tiny values (shared < 9: the scalar row) */
        exponent = (r >> 1) % 9u;
        break;
    case 4: /* the row max at mantissa 0x7F (saturates) and the rest near it */
        if (index == 3u)
            return (uint16_t)(sign | (center << 7) | 0x7Fu);
        exponent = center - (r >> 1) % 3u;
        break;
    case 5: /* ties for both formats: low mantissa bits patterned */
        exponent = center - (r >> 1) % 6u;
        return (uint16_t)(sign | ((exponent & 0xFFu) << 7) | ((r >> 8) & 0x70u) |
                          (index & 1u ? 0x08u : 0x00u));
    case 6: /* uniform over every finite bf16 */
        exponent = (r >> 1) % 255u;
        break;
    default: /* mixes of zeros and normal values */
        if ((r & 3u) == 0u)
            return (uint16_t)sign;
        exponent = center - (r >> 1) % 12u;
        break;
    }
    if ((int32_t)exponent < 0)
        exponent = 0;
    return (uint16_t)(sign | ((exponent & 0xFFu) << 7) | ((r >> 9) & 0x7Fu));
}

static void make_row(uint16_t values[16], uint32_t kind)
{
    /* Centres across the range a projection output lives in, plus the exponent extremes. */
    uint32_t center = 100u + next_random() % 40u;

    if ((next_random() & 15u) == 0u)
        center = 9u + next_random() % 246u;
    if (center > 254u)
        center = 254u;
    for (uint32_t i = 0; i < 16u; i++)
        values[i] = row_value(kind, center, i);
}

static void to_words(const uint16_t *values, uint32_t *words, uint32_t value_count)
{
    for (uint32_t i = 0; i < value_count / 2u; i++)
        words[i] = (uint32_t)values[2u * i] | ((uint32_t)values[2u * i + 1u] << 16);
}

static void words_to_bytes(const uint32_t *words, uint8_t *bytes, uint32_t byte_count)
{
    for (uint32_t i = 0; i < byte_count; i++)
        bytes[i] = (uint8_t)(words[i / 4u] >> (8u * (i % 4u)));
}

/** The fast rule's whole datum (sign included, zero unsigned) for one value. */
static uint32_t fast_datum(uint32_t bits, uint32_t shared, uint32_t magnitude_bits)
{
    const uint32_t magnitude =
        ft_kv_bf_fast_magnitude(bits, (int32_t)shared - (int32_t)magnitude_bits - 1,
                                magnitude_bits);

    return magnitude == 0u ? 0u : (((bits >> 15) & 1u) << magnitude_bits) | magnitude;
}

/* ---- 1. the rule, exhaustively ------------------------------------------------------------ */

static void check_rule_exhaustive(void)
{
    for (uint32_t f = 0; f < 2u; f++) {
        const uint32_t magnitude_bits = ft_kv_bf_magnitude_bits(k_formats[f]);
        uint64_t compared = 0u, wrong = 0u;

        for (uint32_t shared = kKvBfFastMinShared; shared < 256u; shared++) {
            for (uint32_t bits = 0; bits < 65536u; bits++) {
                const uint32_t exponent = (bits >> 7) & 0xFFu;

                if (exponent > shared)
                    continue;
                compared++;
                if (fast_datum(bits, shared, magnitude_bits) !=
                    ft_kv_bf_datum_from_bf16(bits, shared, magnitude_bits)) {
                    if (wrong < 3u)
                        fprintf(stderr, "  %s: bits 0x%04x shared %u: fast 0x%02x rule 0x%02x\n",
                                k_names[f], bits, shared, fast_datum(bits, shared, magnitude_bits),
                                ft_kv_bf_datum_from_bf16(bits, shared, magnitude_bits));
                    wrong++;
                }
            }
        }
        CHECK(wrong == 0u, "%s: fast rule differs on %llu of %llu (value, shared) pairs",
              k_names[f], (unsigned long long)wrong, (unsigned long long)compared);
        printf("  %s: rule exhaustive, %llu (bf16, shared 9..255) pairs, %llu differ\n",
               k_names[f], (unsigned long long)compared, (unsigned long long)wrong);
    }
}

/* ---- 2. rows ------------------------------------------------------------------------------ */

static void check_rows(void)
{
    const uint32_t rows = 200000u;

    for (uint32_t f = 0; f < 2u; f++) {
        const uint32_t format = k_formats[f];
        const uint32_t datum_bytes = ft_kv_bf_row_datum_bytes(format);
        uint32_t wrong_shared = 0u, wrong_rows = 0u, slow_rows = 0u;

        for (uint32_t n = 0; n < rows; n++) {
            uint16_t values[16];
            uint32_t words[8], datum_words[4];
            uint8_t want[16], got[16];
            uint32_t want_exponent, got_exponent, max_exponent = 0u;

            make_row(values, n % 8u);
            to_words(values, words, 16u);
            for (uint32_t i = 0; i < 16u; i++) {
                const uint32_t e = ((uint32_t)values[i] >> 7) & 0xFFu;

                max_exponent = e > max_exponent ? e : max_exponent;
            }
            if (ft_kv_bf_row_shared_words(words) != max_exponent)
                wrong_shared++;
            if (max_exponent < (uint32_t)kKvBfFastMinShared)
                slow_rows++;
            memset(datum_words, 0xA5, sizeof(datum_words));
            want_exponent = ft_kv_bf_pack_row(format, values, want);
            got_exponent = format == kKvFormatBfp4
                               ? ft_kv_bf_pack_row_bfp4_words(words, datum_words)
                               : ft_kv_bf_pack_row_bfp8_words(words, datum_words);
            words_to_bytes(datum_words, got, datum_bytes);
            if (got_exponent != want_exponent || memcmp(got, want, datum_bytes) != 0) {
                if (wrong_rows < 3u)
                    fprintf(stderr, "  %s row %u (kind %u) differs: exponent %u vs %u\n",
                            k_names[f], n, n % 8u, got_exponent, want_exponent);
                wrong_rows++;
            }
        }
        CHECK(wrong_shared == 0u, "%s: shared exponent wrong on %u rows", k_names[f],
              wrong_shared);
        CHECK(wrong_rows == 0u, "%s: %u of %u rows differ from ft_kv_bf_pack_row", k_names[f],
              wrong_rows, rows);
        CHECK(slow_rows > 1000u, "%s: only %u rows exercised the scalar fallback", k_names[f],
              slow_rows);
        printf("  %s: %u rows (8 kinds, %u via the scalar fallback), %u differ\n", k_names[f],
               rows, slow_rows, wrong_rows);
    }
}

/* ---- 3. tiles ----------------------------------------------------------------------------- */

/** Whole-tile pack vs the per-row pack of rows 0..31; returns true when identical. */
static bool tile_matches(uint32_t format, const uint16_t tile[kTileValues])
{
    const uint32_t bytes = ft_kv_bf_tile_bytes(format);
    uint32_t words[kTileWords], packed_words[kPackedWordsMax];
    uint8_t want[kKvBfBfp8TileBytes], got[kKvBfBfp8TileBytes];

    memset(want, 0x5A, sizeof(want));
    memset(packed_words, 0xC3, sizeof(packed_words));
    ft_kv_bf_pack_tile_rows(format, tile, want, 0u, 32u);
    to_words(tile, words, kTileValues);
    ft_kv_bf_pack_tile_words(format, words, packed_words);
    words_to_bytes(packed_words, got, bytes);
    return memcmp(got, want, bytes) == 0;
}

static void make_tile(uint16_t tile[kTileValues], uint32_t salt)
{
    for (uint32_t row = 0; row < 64u; row++)
        make_row(tile + row * 16u, (row + salt) % 8u);
}

static void check_tiles(const char *fixture)
{
    const uint32_t random_tiles = 2000u;
    uint16_t *dumped = NULL;
    uint32_t dumped_tiles = 0u;

    if (fixture) {
        FILE *file = fopen(fixture, "rb");
        long size = 0;

        if (file && fseek(file, 0, SEEK_END) == 0 && (size = ftell(file)) > 0 &&
            fseek(file, 0, SEEK_SET) == 0 && size % 2048 == 0) {
            dumped = malloc((size_t)size);
            if (dumped && fread(dumped, 1, (size_t)size, file) == (size_t)size)
                dumped_tiles = (uint32_t)(size / 2048);
        }
        if (file)
            fclose(file);
        CHECK(dumped_tiles > 0u, "fixture %s: no tiles read", fixture);
    }
    for (uint32_t f = 0; f < 2u; f++) {
        uint32_t wrong = 0u, wrong_dumped = 0u;

        for (uint32_t t = 0; t < random_tiles; t++) {
            uint16_t tile[kTileValues];

            make_tile(tile, t);
            if (!tile_matches(k_formats[f], tile))
                wrong++;
        }
        for (uint32_t t = 0; t < dumped_tiles; t++)
            if (!tile_matches(k_formats[f], dumped + (size_t)t * kTileValues))
                wrong_dumped++;
        CHECK(wrong == 0u, "%s: %u of %u random tiles differ from the per-row pack",
              k_names[f], wrong, random_tiles);
        CHECK(wrong_dumped == 0u, "%s: %u of %u dumped tiles differ from the per-row pack",
              k_names[f], wrong_dumped, dumped_tiles);
        printf("  %s: whole tiles vs per-row: %u random, %u dumped K/V tiles; %u + %u differ\n",
               k_names[f], random_tiles, dumped_tiles, wrong, wrong_dumped);
    }
    free(dumped);
}

/* ---- 4. append sequences: the writer's two paths ------------------------------------------ */

enum { kCachePositionTiles = 6 }; /* 192 positions, one dim tile */

typedef struct append_case {
    uint32_t first;     /**< first position */
    uint32_t count;     /**< positions */
    bool whole_tiles;   /**< the descriptor owns its tile rows (kMemFlagDstWholeTiles) */
} append_case;

/** Source rows for positions [first, first+count): position p's bf16 face rows. */
static void source_rows(uint16_t (*rows)[32], uint32_t first, uint32_t count, uint32_t salt)
{
    for (uint32_t p = 0; p < count; p++) {
        make_row(rows[first + p], (first + p + salt) % 8u);
        make_row(rows[first + p] + 16, (first + p + salt + 3u) % 8u);
    }
}

/** The writer's CB page for one tile: rows [row0, row0+count) placed, the rest zero. */
static void make_page(uint16_t page[kTileValues], uint16_t (*rows)[32], uint32_t tile_first,
                      uint32_t row0, uint32_t count)
{
    memset(page, 0, kTileValues * sizeof(uint16_t));
    for (uint32_t r = row0; r < row0 + count; r++)
        for (uint32_t half = 0; half < 2u; half++)
            memcpy(page + ((r / 16u) * 2u + half) * 256u + (r % 16u) * 16u,
                   rows[tile_first + r] + half * 16u, 16u * sizeof(uint16_t));
}

/** Apply one descriptor's tile piece to `cache` (packed tiles back to back). `fast` follows the
 *  writer: a piece that owns its whole tile takes the whole-tile pack; otherwise rows. */
static void apply_piece(uint32_t format, uint8_t *cache, uint16_t (*rows)[32], uint32_t tile,
                        uint32_t row0, uint32_t count, bool whole_tiles, bool fast)
{
    const uint32_t bytes = ft_kv_bf_tile_bytes(format);
    const bool owns = whole_tiles || (row0 == 0u && count == 32u);
    uint16_t page[kTileValues];

    make_page(page, rows, tile * 32u, row0, count);
    if (fast && owns) {
        uint32_t words[kTileWords], packed_words[kPackedWordsMax];

        to_words(page, words, kTileValues);
        ft_kv_bf_pack_tile_words(format, words, packed_words);
        words_to_bytes(packed_words, cache + tile * bytes, bytes);
    } else if (owns) {
        ft_kv_bf_pack_tile_rows(format, page, cache + tile * bytes, 0u, 32u);
    } else {
        ft_kv_bf_pack_tile_rows(format, page, cache + tile * bytes, row0, count);
    }
}

static void check_appends(void)
{
    /* Each sequence is a list of appends; a host append is split into per-tile descriptors. */
    static const append_case seq_a[] = { { 0, 32, false }, { 32, 32, false }, { 64, 32, false },
                                         { 96, 17, false }, { 113, 1, false }, { 114, 1, false } };
    static const append_case seq_b[] = { { 0, 1, false },  { 1, 1, false },   { 2, 30, false },
                                         { 32, 5, false }, { 37, 27, false }, { 64, 64, false },
                                         { 128, 9, true } };
    static const append_case seq_c[] = { { 0, 20, false }, { 20, 44, false }, { 64, 96, false },
                                         { 160, 32, false } };
    static const append_case seq_d[] = { { 0, 32, false }, { 32, 32, true },  { 64, 7, true },
                                         { 71, 25, false }, { 96, 1, false }, { 97, 63, false },
                                         { 160, 31, false }, { 191, 1, false } };
    static const struct { const append_case *steps; uint32_t count; } seqs[] = {
        { seq_a, sizeof(seq_a) / sizeof(seq_a[0]) }, { seq_b, sizeof(seq_b) / sizeof(seq_b[0]) },
        { seq_c, sizeof(seq_c) / sizeof(seq_c[0]) }, { seq_d, sizeof(seq_d) / sizeof(seq_d[0]) } };
    static uint16_t rows[kCachePositionTiles * 32][32];
    uint32_t pieces_fast = 0u, pieces_rows = 0u;

    for (uint32_t s = 0; s < sizeof(seqs) / sizeof(seqs[0]); s++) {
        for (uint32_t trial = 0; trial < 50u; trial++) {
            /* K Bfp8_b and V Bfp4_b caches fed by the same appends (different source rows). */
            uint8_t fast_cache[2][kCachePositionTiles * kKvBfBfp8TileBytes];
            uint8_t row_cache[2][kCachePositionTiles * kKvBfBfp8TileBytes];

            memset(fast_cache, 0, sizeof(fast_cache));
            memset(row_cache, 0, sizeof(row_cache));
            for (uint32_t f = 0; f < 2u; f++) {
                for (uint32_t step = 0; step < seqs[s].count; step++) {
                    const append_case *a = &seqs[s].steps[step];
                    uint32_t p = a->first;

                    source_rows(rows, a->first, a->count, trial * 7u + f);
                    while (p < a->first + a->count) {
                        const uint32_t tile = p / 32u, row0 = p % 32u;
                        const uint32_t end = a->first + a->count;
                        const uint32_t count = end - p < 32u - row0 ? end - p : 32u - row0;
                        const bool owns = a->whole_tiles || (row0 == 0u && count == 32u);

                        apply_piece(k_formats[f], fast_cache[f], rows, tile, row0, count,
                                    a->whole_tiles, true);
                        apply_piece(k_formats[f], row_cache[f], rows, tile, row0, count,
                                    a->whole_tiles, false);
                        if (owns)
                            pieces_fast++;
                        else
                            pieces_rows++;
                        p += count;
                    }
                }
            }
            CHECK(memcmp(fast_cache[0], row_cache[0], sizeof(fast_cache[0])) == 0,
                  "sequence %u trial %u: K (bfp8) cache differs between the paths", s, trial);
            CHECK(memcmp(fast_cache[1], row_cache[1], sizeof(fast_cache[1])) == 0,
                  "sequence %u trial %u: V (bfp4) cache differs between the paths", s, trial);
        }
    }
    CHECK(pieces_fast > 0u && pieces_rows > 0u, "append sequences used one path only");
    printf("  appends: 4 sequences x 50 trials, K bfp8 + V bfp4; %u whole-tile pieces, %u "
           "per-row pieces\n", pieces_fast, pieces_rows);
}

int main(int argc, char **argv)
{
    printf("test_kv_block_float_tile\n");
    check_rule_exhaustive();
    check_rows();
    check_tiles(argc > 1 ? argv[1] : NULL);
    check_appends();
    printf("%d checks, %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
