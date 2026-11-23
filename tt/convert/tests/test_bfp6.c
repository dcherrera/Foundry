/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_bfp6.c
 * @brief Off-device gates for BFP6_B and BFP2_B (qwen36 plan 1.2 / 1.3): the packer against
 *   a definition-level block reference, both datum layouts, and the exact BFP6_B -> BFP8_B
 *   expansion over every datum value and every exponent.
 *
 * WHAT IS INDEPENDENT. `reference_round_trip_block()` is written from the format's
 * definition (evals/expert_quant/bfp_block_round_trip.c, the eval's codec): shared exponent =
 * the block's largest exponent field, truncating align, round to nearest even, saturate,
 * decode = magnitude * 2^(E - 127 - (magnitude_bits - 1)). It never calls ft_bfp.c. Blocks are
 * 16 consecutive OUTPUT features at a fixed input — a tile's face rows once the weight is
 * K x N — so the comparison walks the untilized tile row by row.
 *
 * EVERY GATE HERE HAS A RED THAT MUST FAIL, run in the same binary:
 *   - the round trip compared against the reference blocked along IN (columns) DIFFERS;
 *   - the expansion compared against a datum << 1 expansion DIFFERS;
 *   - a layout rewrite compared against the other layout's bytes DIFFERS.
 * A comparator that cannot see those cannot be trusted green (coding-best-practices §9).
 *
 * @status New 2026-10-28 (qwen36 Phase 1).
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_bfp.h"

#define TILE_SIDE 32u
#define TILE_VALUES (TILE_SIDE * TILE_SIDE)
#define BLOCK 16u
#define TILE_COUNT 64u
#define BFP6_TILE 832u
#define BFP8_TILE 1088u

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (!condition)
        checks_failed++;
    printf("  %s  %s\n", condition ? "ok  " : "FAIL", what);
}

static void check_count(uint64_t got, uint64_t want, const char *what)
{
    checks_run++;
    if (got != want)
        checks_failed++;
    printf("  %s  %s = %llu (want %llu)\n", got == want ? "ok  " : "FAIL", what, (unsigned long long)got,
           (unsigned long long)want);
}

static uint32_t bits_of(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/** xorshift64*: deterministic, so a failure reproduces. */
static uint64_t random_state = 0x9E3779B97F4A7C15ull;

static uint32_t random_u32(void)
{
    random_state ^= random_state >> 12;
    random_state ^= random_state << 25;
    random_state ^= random_state >> 27;
    return (uint32_t)((random_state * 0x2545F4914F6CDD1Dull) >> 32);
}

/** Weight-like values: random sign and mantissa over a 2^-12 .. 2^2 range, some zeros. */
static float random_weight(void)
{
    uint32_t r = random_u32();
    float magnitude;

    if (r % 97u == 0u)
        return 0.0f;
    magnitude = ldexpf(1.0f + (float)(random_u32() & 0xffffffu) / 16777216.0f, (int)(r % 15u) - 12);
    return (r & 0x80000000u) ? -magnitude : magnitude;
}

/* ------------------------------------------------------------------------- */
/* The definition-level reference                                            */
/* ------------------------------------------------------------------------- */

static uint32_t round_off_ties_even(uint32_t aligned, uint32_t shift)
{
    uint32_t kept = aligned >> shift;
    uint32_t remainder = aligned & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1u);

    if (remainder > half || (remainder == half && (kept & 1u)))
        return kept + 1u;
    return kept;
}

/** Quantize then decode 16 values (stride `step` apart in `in`) at `magnitude_bits`. */
static void reference_round_trip_block(const float *in, size_t step, float *out, uint32_t magnitude_bits)
{
    uint32_t shared = 0, max_magnitude = (1u << magnitude_bits) - 1u;

    for (uint32_t i = 0; i < BLOCK; i++) {
        uint32_t exponent = (bits_of(in[i * step]) >> 23) & 0xffu;

        if (exponent > shared)
            shared = exponent;
    }
    for (uint32_t i = 0; i < BLOCK; i++) {
        uint32_t bits = bits_of(in[i * step]), exponent = (bits >> 23) & 0xffu, magnitude = 0;

        if (exponent != 0) {
            uint32_t alignment = shared - exponent;
            uint32_t significand = (bits & 0x7fffffu) | 0x800000u;

            magnitude = round_off_ties_even(alignment >= 32u ? 0u : significand >> alignment, 24u - magnitude_bits);
            if (magnitude > max_magnitude)
                magnitude = max_magnitude;
        }
        out[i * step] = magnitude == 0 ? 0.0f
                                       : ((bits >> 31) ? -1.0f : 1.0f) *
                                             ldexpf((float)magnitude, (int)shared - 127 - (int)(magnitude_bits - 1u));
    }
}

/** The reference over a whole row-major tile, blocked along rows (OUT) or columns (IN). */
static void reference_round_trip_tile(const float *tile, float *out, uint32_t magnitude_bits, int along_rows)
{
    for (uint32_t a = 0; a < TILE_SIDE; a++) {
        for (uint32_t half = 0; half < 2u; half++) {
            size_t origin = along_rows ? (size_t)a * TILE_SIDE + half * BLOCK : (size_t)half * BLOCK * TILE_SIDE + a;

            reference_round_trip_block(tile + origin, along_rows ? 1u : TILE_SIDE, out + origin, magnitude_bits);
        }
    }
}

/** Bit mismatches between ft_bfp's round trip of `tiles` and the reference. */
static uint64_t round_trip_mismatches(ft_bfp_format format, uint32_t magnitude_bits, const float *tiles,
                                      int reference_along_rows)
{
    unsigned char *packed = calloc(TILE_COUNT, BFP8_TILE);
    float *restored = calloc(TILE_COUNT * TILE_VALUES, sizeof(float));
    float *expected = calloc(TILE_COUNT * TILE_VALUES, sizeof(float));
    uint64_t mismatches = 0;

    if (!packed || !restored || !expected)
        abort();
    if (ft_bfp_pack_tiles(format, tiles, TILE_COUNT, packed) != FT_BFP_OK ||
        ft_bfp_unpack_tiles(format, packed, TILE_COUNT, restored) != FT_BFP_OK)
        mismatches = UINT64_MAX;
    for (uint32_t t = 0; t < TILE_COUNT && mismatches != UINT64_MAX; t++) {
        reference_round_trip_tile(tiles + (size_t)t * TILE_VALUES, expected + (size_t)t * TILE_VALUES, magnitude_bits,
                                  reference_along_rows);
        for (uint32_t i = 0; i < TILE_VALUES; i++)
            mismatches += bits_of(restored[(size_t)t * TILE_VALUES + i]) != bits_of(expected[(size_t)t * TILE_VALUES + i]);
    }
    free(packed);
    free(restored);
    free(expected);
    return mismatches;
}

static void test_round_trip_against_reference(const float *tiles)
{
    printf("\nround trip vs the definition-level reference (%u tiles)\n", TILE_COUNT);
    check_count(ft_bfp_tile_bytes(FT_BFP_BFP6_B), BFP6_TILE, "BFP6_B tile bytes");
    check_count(ft_bfp_tile_bytes(FT_BFP_BFP2_B), 320u, "BFP2_B tile bytes");
    check_count(ft_bfp_mantissa_bits(FT_BFP_BFP6_B), 5u, "BFP6_B magnitude bits");
    check_count(round_trip_mismatches(FT_BFP_BFP6_B, 5u, tiles, 1), 0, "BFP6_B mismatches, blocked along OUT");
    check_count(round_trip_mismatches(FT_BFP_BFP2_B, 1u, tiles, 1), 0, "BFP2_B mismatches, blocked along OUT");
    check_count(round_trip_mismatches(FT_BFP_BFP8_B, 7u, tiles, 1), 0, "BFP8_B mismatches (control: same reference)");
    check(round_trip_mismatches(FT_BFP_BFP6_B, 5u, tiles, 0) > 10000u,
          "RED: the same comparison blocked along IN DIFFERS (> 10000 values)");
}

/* ------------------------------------------------------------------------- */
/* Layouts and the expansion                                                 */
/* ------------------------------------------------------------------------- */

/** A BFP6_B tile set holding every datum value (64) under every exponent byte (256). */
static void fill_exhaustive_bfp6(unsigned char *stream_tiles, uint32_t tiles)
{
    for (uint32_t t = 0; t < tiles; t++) {
        unsigned char *tile = stream_tiles + (size_t)t * BFP6_TILE;

        for (uint32_t b = 0; b < 64u; b++)
            tile[b] = (unsigned char)((t * 64u + b) % 256u);
        memset(tile + 64, 0, BFP6_TILE - 64u);
        for (uint32_t i = 0; i < TILE_VALUES; i++) {
            uint32_t bit = i * 6u, datum = (i + t) % 64u;

            tile[64 + bit / 8u] = (unsigned char)(tile[64 + bit / 8u] | (datum << (bit % 8u)));
            if (bit % 8u > 2u)
                tile[64 + bit / 8u + 1u] = (unsigned char)(tile[64 + bit / 8u + 1u] | (datum >> (8u - bit % 8u)));
        }
    }
}

/** Decoded-value mismatches between a BFP6_B tile set and BFP8_B tiles claimed equal. */
static uint64_t expansion_value_mismatches(const unsigned char *bfp6, const unsigned char *bfp8, uint32_t tiles)
{
    float a[TILE_VALUES], b[TILE_VALUES];
    uint64_t mismatches = 0;

    for (uint32_t t = 0; t < tiles; t++) {
        ft_bfp_unpack_tiles(FT_BFP_BFP6_B, bfp6 + (size_t)t * BFP6_TILE, 1, a);
        ft_bfp_unpack_tiles(FT_BFP_BFP8_B, bfp8 + (size_t)t * BFP8_TILE, 1, b);
        for (uint32_t i = 0; i < TILE_VALUES; i++)
            mismatches += bits_of(a[i]) != bits_of(b[i]);
    }
    return mismatches;
}

static void test_expansion_exhaustive(void)
{
    const uint32_t tiles = 64u; /* 64 x 64 exponent bytes = every exponent 16 times */
    unsigned char *stream = calloc(tiles, BFP6_TILE), *planes = calloc(tiles, BFP6_TILE);
    unsigned char *back = calloc(tiles, BFP6_TILE), *bfp8 = calloc(tiles, BFP8_TILE);
    unsigned char *bfp8_planes = calloc(tiles, BFP8_TILE);
    uint64_t seen[64] = {0}, wrong_shift = 0, mismatches;

    if (!stream || !planes || !back || !bfp8 || !bfp8_planes)
        abort();
    printf("\nBFP6_B -> BFP8_B expansion, every datum value x every exponent byte\n");
    fill_exhaustive_bfp6(stream, tiles);
    for (uint32_t t = 0; t < tiles; t++)
        for (uint32_t i = 0; i < TILE_VALUES; i++)
            seen[(i + t) % 64u]++;
    check(seen[0] > 0 && seen[63] > 0 && seen[31] > 0 && seen[32] > 0, "all 64 datum values present");

    check(ft_bfp6_expand_tiles(FT_BFP6_LAYOUT_STREAM, stream, tiles, bfp8) == FT_BFP_OK, "expand (stream) runs");
    mismatches = expansion_value_mismatches(stream, bfp8, tiles);
    check_count(mismatches, 0, "datums (of 65,536) whose BFP8_B value differs from the BFP6_B value");
    {
        uint64_t exponent_tiles_differing = 0;

        for (uint32_t t = 0; t < tiles; t++)
            exponent_tiles_differing += memcmp(bfp8 + (size_t)t * BFP8_TILE, stream + (size_t)t * BFP6_TILE, 64) != 0;
        check_count(exponent_tiles_differing, 0, "tiles whose exponent bytes changed");
    }
    {
        unsigned char *mutant = calloc(tiles, BFP8_TILE);

        if (!mutant)
            abort();
        memcpy(mutant, bfp8, (size_t)tiles * BFP8_TILE);
        for (uint32_t t = 0; t < tiles; t++)
            for (uint32_t i = 0; i < TILE_VALUES; i++)
                mutant[(size_t)t * BFP8_TILE + 64 + i] = (unsigned char)(mutant[(size_t)t * BFP8_TILE + 64 + i] >> 1);
        wrong_shift = expansion_value_mismatches(stream, mutant, tiles);
        free(mutant);
    }
    check(wrong_shift > 30000u, "RED: a datum << 1 expansion DIFFERS (> 30000 values)");

    check(ft_bfp6_relayout_tiles(FT_BFP6_LAYOUT_STREAM, FT_BFP6_LAYOUT_PLANES, stream, tiles, planes) == FT_BFP_OK,
          "relayout stream -> planes runs");
    check(memcmp(planes, stream, (size_t)tiles * BFP6_TILE) != 0, "RED: the planes bytes DIFFER from the stream bytes");
    check(ft_bfp6_relayout_tiles(FT_BFP6_LAYOUT_PLANES, FT_BFP6_LAYOUT_STREAM, planes, tiles, back) == FT_BFP_OK &&
              memcmp(back, stream, (size_t)tiles * BFP6_TILE) == 0,
          "planes -> stream restores every byte");
    check(ft_bfp6_expand_tiles(FT_BFP6_LAYOUT_PLANES, planes, tiles, bfp8_planes) == FT_BFP_OK &&
              memcmp(bfp8_planes, bfp8, (size_t)tiles * BFP8_TILE) == 0,
          "expansion from planes == expansion from stream, byte for byte");
    check(ft_bfp6_expand_tiles((ft_bfp6_layout)7, stream, 1, bfp8) == FT_BFP_EINVAL, "unknown layout refused");
    free(stream);
    free(planes);
    free(back);
    free(bfp8);
    free(bfp8_planes);
}

/** The card's FOLD expander (tt/kernels/bfp6_expand_tile.h), word for word, little-endian:
 *  the reference the host checks the layout against before any card run. `shift` 2 ships,
 *  anything else is the red build. */
static void card_fold_expand(const unsigned char *in, unsigned char *out, uint32_t tiles, uint32_t shift)
{
    const uint32_t m = 0x03030303u, keep = 0xFCFCFCFCu;

    for (uint32_t t = 0; t < tiles; t++) {
        const unsigned char *src = in + (size_t)t * BFP6_TILE;
        unsigned char *dst = out + (size_t)t * BFP8_TILE;

        memcpy(dst, src, 64);
        for (uint32_t g = 0; g < 64u; g++) {
            uint32_t w[3], o[4];

            memcpy(w, src + 64 + g * 12u, 12);
            o[0] = w[0] & keep;
            o[1] = w[1] & keep;
            o[2] = w[2] & keep;
            o[3] = ((w[0] & m) << (shift + 4u)) | ((w[1] & m) << (shift + 2u)) | ((w[2] & m) << shift);
            if (shift != 2u)
                o[0] = (w[0] & keep) >> 1;
            memcpy(dst + 64 + g * 16u, o, 16);
        }
    }
}

/** FOLD layout (plan S3): relayout round trips, the fast converter == the datum-level one, and the
 *  card's eleven-op expander reproduces ft_bfp6_expand_tiles() byte for byte; reds differ. */
static void test_fold_layout(void)
{
    const uint32_t tiles = 64u;
    unsigned char *stream = calloc(tiles, BFP6_TILE), *fold = calloc(tiles, BFP6_TILE);
    unsigned char *fast = calloc(tiles, BFP6_TILE), *back = calloc(tiles, BFP6_TILE);
    unsigned char *ref8 = calloc(tiles, BFP8_TILE), *card8 = calloc(tiles, BFP8_TILE);

    if (!stream || !fold || !fast || !back || !ref8 || !card8)
        abort();
    printf("\nBFP6_B FOLD layout (plan S3): every datum value x every exponent byte\n");
    fill_exhaustive_bfp6(stream, tiles);
    check(ft_bfp6_relayout_tiles(FT_BFP6_LAYOUT_STREAM, FT_BFP6_LAYOUT_FOLD, stream, tiles, fold) == FT_BFP_OK,
          "relayout stream -> fold runs");
    check(memcmp(fold, stream, (size_t)tiles * BFP6_TILE) != 0, "RED: the fold bytes DIFFER from the stream bytes");
    check(ft_bfp6_relayout_tiles(FT_BFP6_LAYOUT_FOLD, FT_BFP6_LAYOUT_STREAM, fold, tiles, back) == FT_BFP_OK &&
              memcmp(back, stream, (size_t)tiles * BFP6_TILE) == 0,
          "fold -> stream restores every byte");
    check(ft_bfp6_fold_from_stream(stream, tiles, fast) == FT_BFP_OK &&
              memcmp(fast, fold, (size_t)tiles * BFP6_TILE) == 0,
          "ft_bfp6_fold_from_stream == the datum-level relayout, byte for byte");
    ft_bfp6_expand_tiles(FT_BFP6_LAYOUT_STREAM, stream, tiles, ref8);
    memset(card8, 0, (size_t)tiles * BFP8_TILE);
    check(ft_bfp6_expand_tiles(FT_BFP6_LAYOUT_FOLD, fold, tiles, card8) == FT_BFP_OK &&
              memcmp(card8, ref8, (size_t)tiles * BFP8_TILE) == 0,
          "expansion from fold == expansion from stream, byte for byte");
    card_fold_expand(fold, card8, tiles, 2u);
    check(memcmp(card8, ref8, (size_t)tiles * BFP8_TILE) == 0,
          "the card's fold expander (eleven ops a group) == ft_bfp6_expand_tiles(stream)");
    card_fold_expand(fold, card8, tiles, 1u);
    check(memcmp(card8, ref8, (size_t)tiles * BFP8_TILE) != 0, "RED: the card's red fold expander (shift 1) DIFFERS");
    card_fold_expand(stream, card8, tiles, 2u);
    check(memcmp(card8, ref8, (size_t)tiles * BFP8_TILE) != 0, "RED: the fold expander over STREAM bytes DIFFERS");
    {
        unsigned char *mutant = calloc(tiles, BFP6_TILE);

        if (!mutant)
            abort();
        memcpy(mutant, fold, (size_t)tiles * BFP6_TILE);
        mutant[64 + 8] ^= 0x01u; /* one low bit of word 2 = one bit of a word-3 datum */
        check(memcmp(mutant, fast, (size_t)tiles * BFP6_TILE) != 0 &&
                  ft_bfp6_relayout_tiles(FT_BFP6_LAYOUT_FOLD, FT_BFP6_LAYOUT_STREAM, mutant, tiles, back) == FT_BFP_OK &&
                  memcmp(back, stream, (size_t)tiles * BFP6_TILE) != 0,
              "RED: one flipped fold bit changes the stream it decodes to");
        free(mutant);
    }
    free(stream);
    free(fold);
    free(fast);
    free(back);
    free(ref8);
    free(card8);
}

/** Packing weight-like values to BFP6_B and expanding keeps every value. */
static void test_pack_expand_values(const float *tiles)
{
    unsigned char *bfp6 = calloc(TILE_COUNT, BFP6_TILE), *bfp8 = calloc(TILE_COUNT, BFP8_TILE);

    if (!bfp6 || !bfp8)
        abort();
    printf("\npack BFP6_B then expand: the BFP8_B tiles carry the BFP6_B values exactly\n");
    ft_bfp_pack_tiles(FT_BFP_BFP6_B, tiles, TILE_COUNT, bfp6);
    ft_bfp6_expand_tiles(FT_BFP6_LAYOUT_STREAM, bfp6, TILE_COUNT, bfp8);
    check_count(expansion_value_mismatches(bfp6, bfp8, TILE_COUNT), 0, "value mismatches on weight-like tiles");
    free(bfp6);
    free(bfp8);
}

int main(void)
{
    float *tiles = malloc((size_t)TILE_COUNT * TILE_VALUES * sizeof(float));

    if (!tiles)
        return 1;
    printf("test_bfp6 — BFP6_B / BFP2_B codec and the BFP6_B -> BFP8_B expansion\n");
    for (size_t i = 0; i < (size_t)TILE_COUNT * TILE_VALUES; i++)
        tiles[i] = random_weight();
    test_round_trip_against_reference(tiles);
    test_expansion_exhaustive();
    test_fold_layout();
    test_pack_expand_values(tiles);
    free(tiles);
    printf("\n%d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}
