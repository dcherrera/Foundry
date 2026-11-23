/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_bfp.c
 * @brief Off-device test: ft_bfp.c is bit-exact against the committed block-float
 *   reference vectors, for BFP8 and BFP4, exp_a and exp_b, pack and unpack.
 *
 * NO DEVICE IS TOUCHED. The comparison is against
 * reference/ft_bfp_reference_vectors.h, written by
 * reference/generate_bfp_reference.c only after its independent
 * definition-level reference agreed with ft_bfp.c on every byte and word.
 *
 * ---------------------------------------------------------------------------
 * THE VECTORS ARE THE TEST, SO THE VECTORS ARE THEMSELVES CHECKED
 * ---------------------------------------------------------------------------
 * "Bit-exact on random floats" is a weak claim: random floats never land on a
 * round-to-nearest-even tie, never carry a denormal, and rarely span more than
 * a couple of binades inside one 16-element block. So this file first asserts
 * what the input vectors actually CONTAIN — ties of both parities, exact
 * zeros, signed zeros, denormals, infinities, NaN, mantissa saturation, and
 * exponent differences past the 31-bit shift boundary — and only then asserts
 * that the packed bytes match. A coverage check that fails means the vectors
 * stopped testing something, even if every byte still matches.
 *
 * @status Green on macOS/arm64 under
 *   `cc -std=c99 -Wall -Wextra -Werror -pedantic -Wshadow -Wconversion`.
 *   Mutation-proven: see docs/tt_bringup_log.md §12.
 * @issues None
 * @todo None
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ft_bfp.h"
#include "ft_tt_tilize.h"
#include "ft_bfp_reference_vectors.h"

/* ------------------------------------------------------------------------- */
/* Tiny harness                                                              */
/* ------------------------------------------------------------------------- */

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (condition) {
        printf("  ok    %s\n", what);
    } else {
        checks_failed++;
        printf("  FAIL  %s\n", what);
    }
}

static void check_u64(uint64_t got, uint64_t want, const char *what)
{
    checks_run++;
    if (got == want) {
        printf("  ok    %s = %" PRIu64 "\n", what, got);
    } else {
        checks_failed++;
        printf("  FAIL  %s = %" PRIu64 ", expected %" PRIu64 "\n", what, got, want);
    }
}

static void section(const char *name)
{
    printf("\n== %s ==\n", name);
}

/* ------------------------------------------------------------------------- */
/* Vector coverage — what is actually in the inputs                          */
/* ------------------------------------------------------------------------- */

#define TOTAL_VALUES (FT_BFP_REF_TILE_COUNT * FT_BFP_REF_TILE_VALUES)

typedef struct {
    uint32_t zeros;
    uint32_t negative_zeros;
    uint32_t denormals;
    uint32_t infinities;
    uint32_t nans;
    uint32_t max_mantissa;      /**< fraction 0x7fffff — the saturation feeder */
    uint32_t blocks;
    uint32_t blocks_shift_over_31; /**< blocks with an exponent spread > 31 */
    uint32_t blocks_uniform_exp;
} vector_coverage;

static void survey_vectors(vector_coverage *cov)
{
    memset(cov, 0, sizeof(*cov));

    for (uint32_t i = 0; i < TOTAL_VALUES; i++) {
        uint32_t bits = ft_bfp_ref_input_bits[i];
        uint32_t exp = (bits >> 23) & 0xffu;
        uint32_t frac = bits & 0x7fffffu;

        if (exp == 0 && frac == 0) {
            cov->zeros++;
            if (bits >> 31)
                cov->negative_zeros++;
        } else if (exp == 0) {
            cov->denormals++;
        } else if (exp == 0xffu && frac == 0) {
            cov->infinities++;
        } else if (exp == 0xffu) {
            cov->nans++;
        }
        if (frac == 0x7fffffu)
            cov->max_mantissa++;
    }

    /* A block is 16 contiguous values of a face row; in the row-major tile
     * that is a contiguous run of 16, so blocks tile the input linearly. */
    for (uint32_t b = 0; b < TOTAL_VALUES / FT_BFP_BLOCK_VALUES; b++) {
        const uint32_t *block = &ft_bfp_ref_input_bits[b * FT_BFP_BLOCK_VALUES];
        uint32_t min_exp = 0xffffffffu;
        uint32_t max_exp = 0;

        cov->blocks++;
        for (uint32_t j = 0; j < FT_BFP_BLOCK_VALUES; j++) {
            uint32_t exp = (block[j] >> 23) & 0xffu;

            if (exp == 0)
                continue; /* zero and denormal do not participate */
            if (exp < min_exp)
                min_exp = exp;
            if (exp > max_exp)
                max_exp = exp;
        }
        if (min_exp == 0xffffffffu)
            continue;
        if (max_exp - min_exp > 31)
            cov->blocks_shift_over_31++;
        if (max_exp == min_exp)
            cov->blocks_uniform_exp++;
    }
}

static void test_vector_coverage(void)
{
    vector_coverage cov;

    section("reference vectors cover the interesting paths");
    survey_vectors(&cov);

    check(cov.zeros > 0, "vectors contain exact zeros");
    check(cov.negative_zeros > 0, "vectors contain negative zeros");
    check(cov.denormals > 0, "vectors contain denormals");
    check(cov.infinities > 0, "vectors contain infinities");
    check(cov.nans > 0, "vectors contain NaN");
    check(cov.max_mantissa > 0, "vectors contain full mantissas (saturation feeder)");
    check(cov.blocks_shift_over_31 > 0, "vectors contain blocks whose exponent spread exceeds 31");
    check(cov.blocks_uniform_exp > 0, "vectors contain blocks with a single exponent");

    printf("        zeros=%u (-0: %u) denormals=%u inf=%u nan=%u full-mantissa=%u\n", cov.zeros, cov.negative_zeros,
           cov.denormals, cov.infinities, cov.nans, cov.max_mantissa);
    printf("        blocks=%u  spread>31=%u  uniform-exp=%u\n", cov.blocks, cov.blocks_shift_over_31,
           cov.blocks_uniform_exp);
}

/* ------------------------------------------------------------------------- */
/* Ties — proven reached, not assumed                                        */
/* ------------------------------------------------------------------------- */

static void test_tie_coverage(void)
{
    static float values[TOTAL_VALUES];
    static unsigned char packed[TOTAL_VALUES * 2];
    uint64_t seen = 0, up = 0;

    section("round-to-nearest-even ties are actually exercised");

    for (uint32_t i = 0; i < TOTAL_VALUES; i++)
        memcpy(&values[i], &ft_bfp_ref_input_bits[i], sizeof(float));

    ft_bfp_tie_counters_reset();
    check(ft_bfp_pack_tiles(FT_BFP_BFP8_B, values, FT_BFP_REF_TILE_COUNT, packed) == FT_BFP_OK, "BFP8 pack succeeded");
    ft_bfp_tie_counters(&seen, &up);
    check(seen > 0, "BFP8 pack hit the tie path");
    check(up > 0, "BFP8 ties rounded UP at least once (odd guard bit)");
    check(seen > up, "BFP8 ties rounded DOWN at least once (even guard bit)");
    printf("        BFP8 ties: %" PRIu64 " (%" PRIu64 " up, %" PRIu64 " down)\n", seen, up, seen - up);

    ft_bfp_tie_counters_reset();
    check(ft_bfp_pack_tiles(FT_BFP_BFP4_B, values, FT_BFP_REF_TILE_COUNT, packed) == FT_BFP_OK, "BFP4 pack succeeded");
    ft_bfp_tie_counters(&seen, &up);
    check(seen > 0, "BFP4 pack hit the tie path");
    check(up > 0, "BFP4 ties rounded UP at least once");
    check(seen > up, "BFP4 ties rounded DOWN at least once");
    printf("        BFP4 ties: %" PRIu64 " (%" PRIu64 " up, %" PRIu64 " down)\n", seen, up, seen - up);

    ft_bfp_tie_counters_reset();
}

/* ------------------------------------------------------------------------- */
/* Bit-exactness against the reference vectors                               */
/* ------------------------------------------------------------------------- */

static void compare_packed(ft_bfp_format format, const char *name, const uint32_t *expected, size_t expected_words)
{
    static float values[TOTAL_VALUES];
    static uint32_t packed[TOTAL_VALUES]; /* >= 272 words per tile, comfortably */
    size_t words = ft_bfp_tile_bytes(format) / 4u * FT_BFP_REF_TILE_COUNT;
    size_t mismatches = 0;
    size_t first_mismatch = 0;

    for (uint32_t i = 0; i < TOTAL_VALUES; i++)
        memcpy(&values[i], &ft_bfp_ref_input_bits[i], sizeof(float));

    memset(packed, 0xa5, sizeof(packed));
    if (ft_bfp_pack_tiles(format, values, FT_BFP_REF_TILE_COUNT, packed) != FT_BFP_OK) {
        check(0, name);
        return;
    }

    if (words != expected_words) {
        checks_run++;
        checks_failed++;
        printf("  FAIL  %s: produced %zu words, reference has %zu\n", name, words, expected_words);
        return;
    }

    for (size_t i = 0; i < words; i++) {
        if (packed[i] != expected[i]) {
            if (mismatches == 0)
                first_mismatch = i;
            mismatches++;
        }
    }

    checks_run++;
    if (mismatches == 0) {
        printf("  ok    %s: %zu/%zu words identical to the reference\n", name, words, words);
    } else {
        checks_failed++;
        printf("  FAIL  %s: %zu/%zu words differ; first at word %zu (got 0x%08x, reference 0x%08x)\n", name, mismatches,
               words, first_mismatch, packed[first_mismatch], expected[first_mismatch]);
    }
}

/** Pack the reference tiles one call per tile; 1 if the words match the reference. */
static int per_tile_packing_matches_reference(void)
{
    static float values[TOTAL_VALUES];
    static uint32_t packed[TOTAL_VALUES];
    size_t tile_words = ft_bfp_tile_bytes(FT_BFP_BFP8_B) / 4u;

    for (uint32_t i = 0; i < TOTAL_VALUES; i++)
        memcpy(&values[i], &ft_bfp_ref_input_bits[i], sizeof(float));
    for (uint32_t t = 0; t < FT_BFP_REF_TILE_COUNT; t++)
        if (ft_bfp_pack_tiles(FT_BFP_BFP8_B, values + t * FT_BFP_REF_TILE_VALUES, 1, packed + t * tile_words) != FT_BFP_OK)
            return 0;
    return memcmp(packed, ft_bfp_ref_bfp8b, sizeof(ft_bfp_ref_bfp8b)) == 0;
}

static void test_pack_bit_exact(void)
{
    section("pack is bit-exact against the reference vectors");

    check_u64(ft_bfp_tile_bytes(FT_BFP_BFP8_B), 1088, "BFP8 tile bytes");
    check_u64(ft_bfp_tile_bytes(FT_BFP_BFP4_B), 576, "BFP4 tile bytes");
    check_u64(ft_bfp_tile_bytes(FT_BFP_BFP8_A), 1088, "BFP8 exp_a tile bytes");
    check_u64(ft_bfp_tile_bytes(FT_BFP_BFP4_A), 576, "BFP4 exp_a tile bytes");

    compare_packed(FT_BFP_BFP8_B, "BFP8_b", ft_bfp_ref_bfp8b,
                   sizeof(ft_bfp_ref_bfp8b) / sizeof(ft_bfp_ref_bfp8b[0]));
    compare_packed(FT_BFP_BFP4_B, "BFP4_b", ft_bfp_ref_bfp4b,
                   sizeof(ft_bfp_ref_bfp4b) / sizeof(ft_bfp_ref_bfp4b[0]));
    compare_packed(FT_BFP_BFP8_A, "BFP8 exp_a", ft_bfp_ref_bfp8_expa,
                   sizeof(ft_bfp_ref_bfp8_expa) / sizeof(ft_bfp_ref_bfp8_expa[0]));
    compare_packed(FT_BFP_BFP4_A, "BFP4 exp_a", ft_bfp_ref_bfp4_expa,
                   sizeof(ft_bfp_ref_bfp4_expa) / sizeof(ft_bfp_ref_bfp4_expa[0]));

    /* compare_packed packs all four tiles in one call; pin that packing them one
     * call per tile gives the same words. */
    check(per_tile_packing_matches_reference(), "per-tile and single-call packing agree (BFP8_b)");
}

static void test_unpack_bit_exact(void)
{
    size_t index = 0;
    size_t mismatches = 0;

    section("unpack is bit-exact against the reference vectors");

    for (uint32_t e = 0; e < FT_BFP_REF_UNPACK_EXP_B_COUNT; e++) {
        uint32_t shared = ft_bfp_ref_unpack_shared_exp_b[e];

        for (uint32_t d = 0; d < 256; d++) {
            if (ft_bfp_convert_datum(FT_BFP_BFP8_B, (uint8_t)d, shared) != ft_bfp_ref_unpack_bfp8b[index])
                mismatches++;
            index++;
        }
    }
    check_u64(mismatches, 0, "BFP8_b unpack mismatches over 256 data x every shared exponent");

    index = 0;
    mismatches = 0;
    for (uint32_t e = 0; e < FT_BFP_REF_UNPACK_EXP_B_COUNT; e++) {
        uint32_t shared = ft_bfp_ref_unpack_shared_exp_b[e];

        for (uint32_t d = 0; d < 16; d++) {
            if (ft_bfp_convert_datum(FT_BFP_BFP4_B, (uint8_t)d, shared) != ft_bfp_ref_unpack_bfp4b[index])
                mismatches++;
            index++;
        }
    }
    check_u64(mismatches, 0, "BFP4_b unpack mismatches");

    index = 0;
    mismatches = 0;
    for (uint32_t e = 0; e < FT_BFP_REF_UNPACK_EXP_A_COUNT; e++) {
        uint32_t shared = ft_bfp_ref_unpack_shared_exp_a[e];

        for (uint32_t d = 0; d < 256; d++) {
            if (ft_bfp_convert_datum(FT_BFP_BFP8_A, (uint8_t)d, shared) != ft_bfp_ref_unpack_bfp8_expa[index])
                mismatches++;
            index++;
        }
    }
    check_u64(mismatches, 0, "BFP8 exp_a unpack mismatches");

    index = 0;
    mismatches = 0;
    for (uint32_t e = 0; e < FT_BFP_REF_UNPACK_EXP_A_COUNT; e++) {
        uint32_t shared = ft_bfp_ref_unpack_shared_exp_a[e];

        for (uint32_t d = 0; d < 16; d++) {
            if (ft_bfp_convert_datum(FT_BFP_BFP4_A, (uint8_t)d, shared) != ft_bfp_ref_unpack_bfp4_expa[index])
                mismatches++;
            index++;
        }
    }
    check_u64(mismatches, 0, "BFP4 exp_a unpack mismatches");
}

/* ------------------------------------------------------------------------- */
/* Behaviour the header promises                                             */
/* ------------------------------------------------------------------------- */

static float float_from_bits(uint32_t bits)
{
    float f;

    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint32_t bits_from_float(float f)
{
    uint32_t bits;

    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static void test_documented_behaviour(void)
{
    static float tile[FT_TT_TILE_VALUES];
    static unsigned char packed[1088];
    static float restored[FT_TT_TILE_VALUES];

    section("documented edge behaviour");

    check_u64(ft_bfp_convert_float_bits(FT_BFP_BFP8_B, 0x00000000u, 127), 0, "+0 packs to datum 0");
    check_u64(ft_bfp_convert_float_bits(FT_BFP_BFP8_B, 0x80000000u, 127), 0, "-0 packs to datum 0 (sign dropped)");
    check_u64(ft_bfp_convert_float_bits(FT_BFP_BFP8_B, 0x00000001u, 127), 0, "smallest denormal packs to datum 0");
    check_u64(ft_bfp_convert_float_bits(FT_BFP_BFP8_B, 0x007fffffu, 127), 0, "largest denormal packs to datum 0");

    /* 1.99999988 under its own exponent: mantissa 0xffffff >> 17 = 0x7f, and
     * the residue rounds up, which must saturate rather than carry. */
    check_u64(ft_bfp_convert_float_bits(FT_BFP_BFP8_B, 0x3f7fffffu, 126), 0x7f, "rounding overflow saturates at 0x7f");
    check_u64(ft_bfp_convert_float_bits(FT_BFP_BFP4_B, 0x3f7fffffu, 126), 0x7, "BFP4 rounding overflow saturates");

    /* A value 2^40 below the block's shared exponent shifts away entirely. */
    check_u64(ft_bfp_convert_float_bits(FT_BFP_BFP8_B, 0xbf800000u, 127 + 40), 0,
              "value far below the shared exponent packs to +0, not -0");

    /* Exactly representable: 1.0 with shared exponent 127 is 0x40, and it must
     * come back as exactly 1.0. */
    check_u64(ft_bfp_convert_float_bits(FT_BFP_BFP8_B, 0x3f800000u, 127), 0x40, "1.0 packs to 0x40");
    check_u64(ft_bfp_convert_datum(FT_BFP_BFP8_B, 0x40, 127), 0x3f800000u, "0x40 unpacks to exactly 1.0");

    /* Whole-tile round trip on values that BFP8 represents exactly: a power of
     * two per block plus multiples of 2^-7 of it. */
    for (uint32_t row = 0; row < FT_TT_TILE_HEIGHT; row++) {
        for (uint32_t col = 0; col < FT_TT_TILE_WIDTH; col++)
            tile[row * FT_TT_TILE_WIDTH + col] = float_from_bits(0x3f800000u) * (float)(64 + (col % 64)) / 64.0f;
    }
    check(ft_bfp_pack_tiles(FT_BFP_BFP8_B, tile, 1, packed) == FT_BFP_OK, "pack of an exactly-representable tile");
    check(ft_bfp_unpack_tiles(FT_BFP_BFP8_B, packed, 1, restored) == FT_BFP_OK, "unpack of that tile");
    {
        size_t differing = 0;

        for (uint32_t i = 0; i < FT_TT_TILE_VALUES; i++) {
            if (bits_from_float(restored[i]) != bits_from_float(tile[i]))
                differing++;
        }
        check_u64(differing, 0, "exactly-representable tile survives pack+unpack unchanged");
    }

    section("argument validation");
    check(ft_bfp_pack_tiles((ft_bfp_format)99, tile, 1, packed) == FT_BFP_EINVAL, "unknown format is rejected");
    check(ft_bfp_pack_tiles(FT_BFP_BFP8_B, NULL, 1, packed) == FT_BFP_EINVAL, "NULL input is rejected");
    check(ft_bfp_pack_tiles(FT_BFP_BFP8_B, tile, 1, NULL) == FT_BFP_EINVAL, "NULL output is rejected");
    check(ft_bfp_pack_tiles(FT_BFP_BFP8_B, NULL, 0, NULL) == FT_BFP_OK, "zero tiles is a no-op, not an error");
    check_u64(ft_bfp_tile_bytes((ft_bfp_format)99), 0, "unknown format has no tile size");
}

int main(void)
{
    printf("test_bfp — ft_bfp.c against the block-float reference vectors\n");

    test_vector_coverage();
    test_tie_coverage();
    test_pack_bit_exact();
    test_unpack_bit_exact();
    test_documented_behaviour();

    printf("\n%d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}
