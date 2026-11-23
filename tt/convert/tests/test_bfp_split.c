/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_bfp_split.c
 * @brief Off-device gates for the BFP4_B + BFP2_B split (qwen36 plan 1.4b, ft_bfp_split.c):
 *   the tile packer agrees with the block codec the eval uses, the high tile is exactly what
 *   the BFP4_B packer writes, every value lands on the documented representable set, and the
 *   error against BFP6_B is reported.
 *
 * REDS, in the same binary: a residual read under the wrong exponent (E - 5) DIFFERS from the
 * packed values; a high-only decode (the second pass dropped) DIFFERS.
 *
 * @status New 2026-10-28 (qwen36 Phase 1.4b).
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_bfp.h"
#include "ft_bfp_split.h"

#define TILE_SIDE 32u
#define TILE_VALUES (TILE_SIDE * TILE_SIDE)
#define TILE_COUNT 64u
#define VALUES ((size_t)TILE_COUNT * TILE_VALUES)

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (!condition)
        checks_failed++;
    printf("  %s  %s\n", condition ? "ok  " : "FAIL", what);
}

static uint32_t bits_of(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint64_t random_state = 0x2545F4914F6CDD1Dull;

static uint32_t random_u32(void)
{
    random_state ^= random_state >> 12;
    random_state ^= random_state << 25;
    random_state ^= random_state >> 27;
    return (uint32_t)((random_state * 0x2545F4914F6CDD1Dull) >> 32);
}

/** Gaussian-ish weights (sum of 4 uniforms), like an expert matrix: most blocks span 2-3 binades. */
static float random_weight(void)
{
    float sum = 0.0f;

    for (int i = 0; i < 4; i++)
        sum += (float)(random_u32() & 0xffffffu) / 16777216.0f - 0.5f;
    return sum * 0.05f;
}

/** Untilized row-major tile -> its face-row blocks in packed block order, contiguous. */
static void tile_blocks_contiguous(const float *tile, float *blocks)
{
    for (uint32_t b = 0; b < 64u; b++) {
        uint32_t face = b / 16u, row = b % 16u;
        uint32_t origin = ((face / 2u) * 16u + row) * TILE_SIDE + (face % 2u) * 16u;

        memcpy(blocks + b * 16u, tile + origin, 16u * sizeof(float));
    }
}

static double rms_error(const float *a, const float *b, size_t n)
{
    double sum = 0.0, ref = 0.0;

    for (size_t i = 0; i < n; i++) {
        sum += ((double)a[i] - (double)b[i]) * ((double)a[i] - (double)b[i]);
        ref += (double)a[i] * (double)a[i];
    }
    return sqrt(sum / ref);
}

static void test_mode(ft_bfp_split_residual mode, const char *name, const float *values, double bfp6_rms)
{
    unsigned char *high = calloc(TILE_COUNT, FT_BFP_SPLIT_HIGH_TILE_BYTES);
    unsigned char *low = calloc(TILE_COUNT, FT_BFP_SPLIT_LOW_TILE_BYTES);
    unsigned char *bfp4 = calloc(TILE_COUNT, FT_BFP_SPLIT_HIGH_TILE_BYTES);
    float *decoded = calloc(VALUES, sizeof(float)), *blocks = calloc(VALUES, sizeof(float));
    float *block_out = calloc(VALUES, sizeof(float)), *red = calloc(VALUES, sizeof(float));
    uint64_t mismatches = 0, off_set = 0, red_differ = 0;
    char line[160];

    if (!high || !low || !bfp4 || !decoded || !blocks || !block_out || !red)
        abort();
    printf("\nresidual mode %s\n", name);
    check(ft_bfp_split_pack_tiles(mode, values, TILE_COUNT, high, low) == FT_BFP_OK, "pack runs");
    check(ft_bfp_split_unpack_tiles(high, low, TILE_COUNT, decoded) == FT_BFP_OK, "unpack runs");
    ft_bfp_pack_tiles(FT_BFP_BFP4_B, values, TILE_COUNT, bfp4);
    check(memcmp(high, bfp4, (size_t)TILE_COUNT * FT_BFP_SPLIT_HIGH_TILE_BYTES) == 0,
          "high tiles are byte-identical to the BFP4_B packer's");

    for (uint32_t t = 0; t < TILE_COUNT; t++) {
        float expected[TILE_VALUES], restored[TILE_VALUES];

        tile_blocks_contiguous(values + (size_t)t * TILE_VALUES, blocks);
        ft_bfp_split_round_trip_blocks(mode, blocks, block_out, 64u);
        tile_blocks_contiguous(decoded + (size_t)t * TILE_VALUES, restored);
        memcpy(expected, block_out, sizeof(expected));
        for (uint32_t i = 0; i < TILE_VALUES; i++)
            mismatches += bits_of(expected[i]) != bits_of(restored[i]);
    }
    snprintf(line, sizeof(line), "tile codec == block codec (eval layout): %llu mismatches", (unsigned long long)mismatches);
    check(mismatches == 0, line);

    /* Every decoded value is high + low with |low| one step of its own exponent byte. */
    for (uint32_t t = 0; t < TILE_COUNT; t++) {
        for (uint32_t b = 0; b < 64u; b++) {
            uint32_t e_high = high[(size_t)t * FT_BFP_SPLIT_HIGH_TILE_BYTES + b];
            uint32_t e_low = low[(size_t)t * FT_BFP_SPLIT_LOW_TILE_BYTES + b];

            if (e_high >= 4u && !(e_low == e_high - 4u || (mode == FT_BFP_SPLIT_RESIDUAL_BEST && e_low == e_high - 3u)))
                off_set++;
        }
    }
    snprintf(line, sizeof(line), "blocks whose residual exponent is not the mode's: %llu", (unsigned long long)off_set);
    check(off_set == 0, line);

    /* RED 1: read every residual one exponent lower. */
    for (uint32_t t = 0; t < TILE_COUNT; t++)
        for (uint32_t b = 0; b < 64u; b++)
            if (low[(size_t)t * FT_BFP_SPLIT_LOW_TILE_BYTES + b] > 0u)
                low[(size_t)t * FT_BFP_SPLIT_LOW_TILE_BYTES + b]--;
    ft_bfp_split_unpack_tiles(high, low, TILE_COUNT, red);
    for (size_t i = 0; i < VALUES; i++)
        red_differ += bits_of(red[i]) != bits_of(decoded[i]);
    snprintf(line, sizeof(line), "RED: residual under E - 5 DIFFERS on %llu values", (unsigned long long)red_differ);
    check(red_differ > VALUES / 4u, line);

    /* RED 2: the second pass dropped. */
    ft_bfp_unpack_tiles(FT_BFP_BFP4_B, high, TILE_COUNT, red);
    printf("  info  relative RMS error: split %.5f, high only %.5f, BFP6_B %.5f (split / BFP6_B = %.2fx)\n",
           rms_error(values, decoded, VALUES), rms_error(values, red, VALUES), bfp6_rms,
           rms_error(values, decoded, VALUES) / bfp6_rms);
    check(rms_error(values, red, VALUES) > 1.5 * rms_error(values, decoded, VALUES),
          "RED: high-only decode is > 1.5x the split's error");
    free(high);
    free(low);
    free(bfp4);
    free(decoded);
    free(blocks);
    free(block_out);
    free(red);
}

int main(void)
{
    float *values = malloc(VALUES * sizeof(float)), *bfp6_values = malloc(VALUES * sizeof(float));
    unsigned char *bfp6 = malloc((size_t)TILE_COUNT * 832u);
    double bfp6_rms;

    if (!values || !bfp6_values || !bfp6)
        return 1;
    printf("test_bfp_split — BFP4_B high + BFP2_B residual (qwen36 plan 1.4b)\n");
    for (size_t i = 0; i < VALUES; i++)
        values[i] = random_weight();
    ft_bfp_pack_tiles(FT_BFP_BFP6_B, values, TILE_COUNT, bfp6);
    ft_bfp_unpack_tiles(FT_BFP_BFP6_B, bfp6, TILE_COUNT, bfp6_values);
    bfp6_rms = rms_error(values, bfp6_values, VALUES);
    test_mode(FT_BFP_SPLIT_RESIDUAL_E4, "E - 4", values, bfp6_rms);
    test_mode(FT_BFP_SPLIT_RESIDUAL_BEST, "best of E - 3 / E - 4", values, bfp6_rms);
    check(ft_bfp_split_pack_tiles((ft_bfp_split_residual)9, values, 1, bfp6, bfp6) == FT_BFP_EINVAL,
          "unknown residual mode refused");
    free(values);
    free(bfp6_values);
    free(bfp6);
    printf("\n%d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}
