/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_fidelity_sweep.c
 * @brief Explains the docs/tt_bringup_log.md §16.3 inversion (LoFi beats HiFi4 on every
 *   long-K MiniCPM5-1B matmul shape, reverses on the one short-K shape) instead of leaving
 *   it as the INFERRED "probably 16-bit DST accumulation" note it was left at.
 *
 * ---------------------------------------------------------------------------
 * THE TWO DISCRIMINATING MEASUREMENTS
 * ---------------------------------------------------------------------------
 * 1. **K sweep, fidelity fixed at each of HiFi4/LoFi, DST_ACCUM_MODE fixed at 0 (the
 *    shipped setting).** If DST accumulator rounding dominates and scales with the number
 *    of K-tile write-backs, the HiFi4-vs-LoFi gap should move systematically with Kt
 *    rather than jump straight from "HiFi4 better" to "LoFi better" at some arbitrary
 *    shape. Every Kt point in the sweep re-reads DST's actual state after exactly that
 *    many accumulation steps — `matmul_tiles()` accumulates into DST across the K loop
 *    (matmul_compute.c) and `pack_tile()` reads DST out after the loop ends, so running
 *    the SAME op with tile_inner = Kt for increasing Kt on a fixed operand PREFIX is a
 *    direct, non-invasive read of the accumulator's trajectory — no debug hook needed.
 *
 * 2. **DST_ACCUM_MODE toggled, at each of HiFi4/LoFi, K fixed.** dst_accum_mode is a
 *    compile-time cbtable setting (tools/emit_tensix_cb_formats.py) that reaches the
 *    kernel as the constexpr `DST_ACCUM_MODE` ft_tensix_cb_formats.h defines, and controls
 *    whether the destination register holds Float16_b (16-bit) or Float32 values end to
 *    end (tech_reports/matrix_engine/matrix_engine.md "Fp32 Dest Acc"). This is the ONLY
 *    other lever that changes accumulation width without touching product precision, so a
 *    2x2 design over {HiFi4, LoFi} x {dst16, dst32} is what tells "the accumulator is the
 *    dominant term" apart from "fidelity phases are doing something unexpected": if the
 *    accumulator dominates, both dst32 arms should converge near each other regardless of
 *    fidelity, and the inversion should shrink or vanish at dst32.
 *
 * Four cbtables drive this, all in kernels/cb_tables/, all diagnostic-only (never named by
 * Makefile.ops or any shipped program directory):
 *   matmul_float16b.cbtable              HiFi4 dst16  (the shipped configuration)
 *   matmul_float16b_lofi.cbtable         LoFi  dst16
 *   matmul_float16b_fp32acc.cbtable      HiFi4 dst32
 *   matmul_float16b_lofi_fp32acc.cbtable LoFi  dst32
 * All four are built from the UNCHANGED matmul_reader.c / matmul_compute.c /
 * vector_add_writer.c sources — nothing about the kernel changes, only the compile-time
 * descriptor set, so a difference between arms cannot be a kernel-logic difference.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS DOES NOT SETTLE, AND WHERE THAT IS CHECKED INSTEAD
 * ---------------------------------------------------------------------------
 * Whether the CPU fp32 reference's own summation order could be biasing the comparison —
 * a classic false signal for exactly this shape of result — was checked OFF-HARDWARE (host
 * only, no card, on the Mac per the "CPU work on the Mac" rule) with a standalone scratch
 * program, not committed here since it is a property of the reference FUNCTION alone and
 * needs no device: same M=32/K=4608/N=32 shape, same bf16-rounded random operands,
 * reference_matmul's actual ascending-K double-accumulator order vs a descending order vs
 * pairwise (tree) summation, all in double. RESULT: bit-identical (max|difference| = 0.0)
 * across all three orders and 6 seeds. The reference's term count and dynamic range are
 * nowhere near double's ~52-bit mantissa budget, so order cannot be a contributor at any
 * magnitude, let alone the ~0.1-1 magnitude gaps this file measures on the card. MEASURED,
 * not assumed — see this agent's final report for the numbers.
 *
 * ---------------------------------------------------------------------------
 * ANTI-VACUITY GUARDS (dev notes §3 — a green result here has to be able to go red)
 * ---------------------------------------------------------------------------
 * Same discipline as test_ops.c: output poisoned to 0xA5 before every launch and checked
 * clear after; full-mantissa random operands, never small integers (a bf16-exact array is
 * bit-identical to a wrong tiling of itself); non-finite counts reported explicitly rather
 * than let a `-nan` PCC silently mean "identical to garbage". Unlike test_ops.c this file
 * does not gate on PCC >= 0.99 — the point is to READ max-abs error and its trend, not to
 * pass/fail a threshold — but every row still asserts the output is finite and unpoisoned
 * before its numbers are trusted.
 *
 * Environment:
 *   FT_TT_TENSIX_INDEX   which enabled Tensix to drive (default 101 — inside this file's
 *                        owner's assigned 100-110 range; 100 is left free of historical
 *                        runs from other sessions)
 *   FT_TT_PROGRAM_DIR    parent of the four arm directories (see ARM_SUBDIR below)
 *
 * @status Run on Tensix 101, 2026-08-12, 0 checks failed. RESULT: at DST_ACCUM_MODE=0 (the
 *   shipped setting) HiFi4's max-abs error grows ~159x over the K=32->4608 sweep while
 *   LoFi's grows only ~38x, so the two curves cross between Kt=32 and Kt=48 — HiFi4 ahead
 *   at every Kt <= 32, LoFi ahead at every Kt >= 48, matching the shipped-config finding's
 *   short-K/long-K split. At DST_ACCUM_MODE=1, HiFi4's error stays flat and near the
 *   sqrt(K) floor (0.0039 -> 0.052, ~13x) and beats LoFi at EVERY Kt with no crossover.
 *   PCC goes from visibly K-dependent (0.999992 -> 0.998962 for HiFi4 dst16) to flat at
 *   0.999999 for HiFi4 dst32. The off-hardware reference-order check (see file header)
 *   found summation order contributes exactly 0.0 difference. Full numbers, MEASURED vs
 *   INFERRED marked, are in this agent's final report — not written into
 *   docs/tt_bringup_log.md by this agent (shared file; another agent integrates it).
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_matmul.h"
#include "../ft_tt_tilize.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define TILE_DIM      FT_TT_OP_TILE_DIM
#define TILE_ELEMENTS (TILE_DIM * TILE_DIM)
#define POISON_BYTE   0xA5u
#define LAUNCH_TIMEOUT 20000u
#define M_ROWS        32u /**< one M-tile: the accumulation trajectory is per output
                            *   element and does not depend on M or N, so the smallest
                            *   legal shape is the cheapest full probe. */
#define N_COLS        32u
#define K_MAX_TILES   144u /**< ffn_down's real Kt — the longest K MiniCPM5-1B uses. */
#define K_MAX_ELEMENTS (K_MAX_TILES * TILE_DIM)

/** DRAM banks 6/7, deliberately away from test_ops.c's 2-5, so a concurrent run of that
 *  suite on the same card cannot collide with this one's buffers. */
#define DRAM_BANK_ACTIVATION 6u
#define DRAM_BANK_WEIGHT     6u
#define DRAM_BANK_OUTPUT     7u
#define ACTIVATION_BASE 0x00100000u
#define WEIGHT_BASE     0x00300000u
#define OUTPUT_BASE     0x00100000u

typedef struct op_context {
    ft_tt_hardware_session session;
    uint16_t core_x;
    uint16_t core_y;
    ft_tt_noc_coord dram[8];
} op_context;

/* ================================================================
 * Deterministic full-mantissa operands (dev notes §3: never small integers)
 * ================================================================ */

static uint64_t random_state = 0x9E3779B97F4A7C15ull;

static void random_seed(uint64_t seed) { random_state = seed ? seed : 1u; }

static float random_value(void)
{
    random_state ^= random_state >> 12;
    random_state ^= random_state << 25;
    random_state ^= random_state >> 27;
    return (float)((double)(random_state * 0x2545F4914F6CDD1Dull >> 11) /
                       (double)(1ull << 53) * 2.0 -
                   1.0);
}

static void round_to_bf16(float *matrix, size_t count)
{
    size_t index;

    for (index = 0; index < count; index++)
        matrix[index] = ft_float_from_bf16(ft_bf16_from_float(matrix[index]));
}

/* ================================================================
 * Scoring — duplicated from test_ops.c rather than shared, because these are file-local
 * statics there and this file must not edit test_ops.c to export them.
 * ================================================================ */

static double pearson(const float *a, const float *b, size_t count)
{
    double mean_a = 0.0, mean_b = 0.0, cov = 0.0, var_a = 0.0, var_b = 0.0;
    size_t index;

    for (index = 0; index < count; index++) {
        mean_a += (double)a[index];
        mean_b += (double)b[index];
    }
    mean_a /= (double)count;
    mean_b /= (double)count;
    for (index = 0; index < count; index++) {
        double da = (double)a[index] - mean_a;
        double db = (double)b[index] - mean_b;

        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if (var_a <= 0.0 || var_b <= 0.0)
        return 0.0 / 0.0;
    return cov / sqrt(var_a * var_b);
}

static double max_abs_difference(const float *a, const float *b, size_t count)
{
    double worst = 0.0;
    size_t index;

    for (index = 0; index < count; index++) {
        double difference = fabs((double)a[index] - (double)b[index]);

        if (difference > worst)
            worst = difference;
    }
    return worst;
}

static size_t count_non_finite(const float *values, size_t count)
{
    size_t bad = 0;
    size_t index;

    for (index = 0; index < count; index++)
        if (!isfinite((double)values[index]))
            bad++;
    return bad;
}

/**
 * C = A x B in fp32, accumulated in double, ascending K order — bit-identical to
 * test_ops.c's reference_matmul (same loop nest, same accumulation order). Kept as a
 * separate copy so this file has no dependency on test_ops.c's internals; the order
 * itself, and whether it could be biasing the comparison, was checked independently
 * off-hardware and found not to matter at all — see the file header.
 */
static void reference_matmul(const float *a, const float *b, uint32_t m, uint32_t k,
                             uint32_t n, float *out)
{
    double *accumulator = malloc((size_t)n * sizeof(double));
    uint32_t row;

    if (!accumulator)
        return;
    for (row = 0; row < m; row++) {
        uint32_t column, inner;

        for (column = 0; column < n; column++)
            accumulator[column] = 0.0;
        for (inner = 0; inner < k; inner++) {
            double scale = (double)a[(size_t)row * k + inner];
            const float *b_row = b + (size_t)inner * n;

            for (column = 0; column < n; column++)
                accumulator[column] += scale * (double)b_row[column];
        }
        for (column = 0; column < n; column++)
            out[(size_t)row * n + column] = (float)accumulator[column];
    }
    free(accumulator);
}

/* ================================================================
 * Device plumbing — poison/write/read, same shape as test_ops.c's
 * ================================================================ */

static int poison_buffer(const op_context *context, ft_tt_noc_coord where, uint32_t address,
                         uint32_t bytes)
{
    unsigned char *pattern = malloc(bytes);
    int rc;

    if (!pattern)
        return FT_TT_ENOMEM;
    memset(pattern, (int)POISON_BYTE, bytes);
    rc = context->session.noc.write(context->session.noc.user, (uint16_t)where.x,
                                    (uint16_t)where.y, (uint64_t)address, pattern, bytes);
    free(pattern);
    return rc;
}

static size_t poison_remaining(const unsigned char *data, size_t count)
{
    size_t remaining = 0;
    size_t index;

    for (index = 0; index < count; index++)
        if (data[index] == POISON_BYTE)
            remaining++;
    return remaining;
}

/* Chunked at 64 KiB, matching test_ops.c's device_write/device_read exactly — a proven
 * pattern on this hardware rather than a new untested one. */
static int device_write(const op_context *context, ft_tt_noc_coord where, uint32_t address,
                        const void *source, size_t bytes)
{
    const unsigned char *cursor = source;
    size_t offset = 0;

    while (offset < bytes) {
        size_t chunk = bytes - offset;
        int rc;

        if (chunk > 0x10000u)
            chunk = 0x10000u;
        rc = context->session.noc.write(context->session.noc.user, (uint16_t)where.x,
                                        (uint16_t)where.y, (uint64_t)address + offset,
                                        cursor + offset, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

static int device_read(const op_context *context, ft_tt_noc_coord where, uint32_t address,
                       void *destination, size_t bytes)
{
    unsigned char *cursor = destination;
    size_t offset = 0;

    while (offset < bytes) {
        size_t chunk = bytes - offset;
        int rc;

        if (chunk > 0x10000u)
            chunk = 0x10000u;
        rc = context->session.noc.read(context->session.noc.user, (uint16_t)where.x,
                                       (uint16_t)where.y, (uint64_t)address + offset,
                                       cursor + offset, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

/* ================================================================
 * One K point: run tile_inner = kt on a fixed operand PREFIX, i.e. read DST's actual
 * state after exactly kt accumulation steps.
 * ================================================================ */

static void run_k_point(op_context *context, const ft_tt_kernel_program *program,
                        const char *arm_label, const float *base_a, const float *base_b,
                        uint32_t kt, double *out_pcc, double *out_max_abs)
{
    uint32_t k = kt * TILE_DIM;
    float *expected = malloc((size_t)M_ROWS * N_COLS * sizeof(float));
    float *actual = malloc((size_t)M_ROWS * N_COLS * sizeof(float));
    uint16_t *a_tiles = malloc((size_t)kt * TILE_ELEMENTS * sizeof(uint16_t));
    uint16_t *b_tiles = malloc((size_t)kt * TILE_ELEMENTS * sizeof(uint16_t));
    uint16_t *out_tiles = malloc((size_t)FT_TT_OP_TILE_BYTES_BF16);
    size_t out_bytes = FT_TT_OP_TILE_BYTES_BF16;
    ft_tt_op_matmul_request request;
    int rc;

    *out_pcc = 0.0 / 0.0;
    *out_max_abs = -1.0;
    if (!expected || !actual || !a_tiles || !b_tiles || !out_tiles) {
        ft_tt_test_check_fmt("fidelity sweep allocation", false, "%s kt=%u: out of memory",
                             arm_label, kt);
        goto cleanup;
    }

    /* base_a is M_ROWS x K_MAX_ELEMENTS row-major fp32; the first `k` columns of every
     * row are the prefix this kt point uses, so it is first squeezed into a DENSE M_ROWS
     * x k copy — reference_matmul indexes its `a` argument with a row stride of `k`, and
     * base_a's real row stride is K_MAX_ELEMENTS. base_b is K_MAX_ELEMENTS x N_COLS with
     * N_COLS elements per row, so its first `k` rows are ALREADY a dense, contiguous
     * prefix and need no such copy. Both prefixes are the SAME terms every larger kt in
     * the sweep will also use — that is what makes the sweep a trajectory instead of a
     * series of unrelated draws. ft_tt_tilize_strip only rearranges bytes, it does not
     * convert format, so both prefixes are rendered to bf16 separately for the tile
     * upload — exactly what the card will read (base_a/base_b are pre-rounded to bf16
     * already, so this second rounding is exact, matching test_ops.c's convention). */
    {
        float *a_dense = malloc((size_t)M_ROWS * k * sizeof(float));
        uint16_t *a_bf16 = malloc((size_t)M_ROWS * k * sizeof(uint16_t));
        uint16_t *b_bf16 = malloc((size_t)k * N_COLS * sizeof(uint16_t));
        uint32_t row, inner;

        if (!a_dense || !a_bf16 || !b_bf16) {
            ft_tt_test_check_fmt("fidelity sweep allocation", false,
                                 "%s kt=%u: out of memory (staging)", arm_label, kt);
            free(a_dense);
            free(a_bf16);
            free(b_bf16);
            goto cleanup;
        }
        for (row = 0; row < M_ROWS; row++)
            memcpy(a_dense + (size_t)row * k, base_a + (size_t)row * K_MAX_ELEMENTS,
                  (size_t)k * sizeof(float));
        reference_matmul(a_dense, base_b, M_ROWS, k, N_COLS, expected);

        for (row = 0; row < M_ROWS; row++) {
            uint32_t column;

            for (column = 0; column < k; column++)
                a_bf16[(size_t)row * k + column] = ft_bf16_from_float(a_dense[(size_t)row * k +
                                                                              column]);
        }
        ft_tt_tilize_strip(a_bf16, sizeof(uint16_t), M_ROWS, k, k * sizeof(uint16_t), a_tiles);

        for (row = 0; row < k; row++) {
            uint32_t column;

            for (column = 0; column < N_COLS; column++)
                b_bf16[(size_t)row * N_COLS + column] =
                    ft_bf16_from_float(base_b[(size_t)row * N_COLS + column]);
        }
        for (inner = 0; inner < kt; inner++)
            ft_tt_tilize_strip(b_bf16 + (size_t)inner * TILE_DIM * N_COLS, sizeof(uint16_t),
                               TILE_DIM, N_COLS, N_COLS * sizeof(uint16_t),
                               b_tiles + (size_t)inner * TILE_ELEMENTS);
        free(a_dense);
        free(a_bf16);
        free(b_bf16);
    }

    rc = device_write(context, context->dram[DRAM_BANK_ACTIVATION], ACTIVATION_BASE, a_tiles,
                      (size_t)kt * FT_TT_OP_TILE_BYTES_BF16);
    rc |= device_write(context, context->dram[DRAM_BANK_WEIGHT], WEIGHT_BASE, b_tiles,
                       (size_t)kt * FT_TT_OP_TILE_BYTES_BF16);
    rc |= poison_buffer(context, context->dram[DRAM_BANK_OUTPUT], OUTPUT_BASE,
                        (uint32_t)out_bytes);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("fidelity sweep upload", false, "%s kt=%u: %s", arm_label, kt,
                             ft_tt_error_string(rc));
        goto cleanup;
    }

    memset(&request, 0, sizeof(request));
    request.tile_rows = 1u;
    request.tile_inner = kt;
    request.tile_cols = 1u;
    request.prefetch_tiles = 1u;
    request.activation.x = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].x;
    request.activation.y = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].y;
    request.activation.address = ACTIVATION_BASE;
    request.activation.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    request.weight = request.activation;
    request.weight.x = (uint16_t)context->dram[DRAM_BANK_WEIGHT].x;
    request.weight.y = (uint16_t)context->dram[DRAM_BANK_WEIGHT].y;
    request.weight.address = WEIGHT_BASE;
    request.output = request.activation;
    request.output.x = (uint16_t)context->dram[DRAM_BANK_OUTPUT].x;
    request.output.y = (uint16_t)context->dram[DRAM_BANK_OUTPUT].y;
    request.output.address = OUTPUT_BASE;

    rc = ft_tt_op_matmul_run(&context->session.noc, context->core_x, context->core_y, program,
                             &request, LAUNCH_TIMEOUT);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("fidelity sweep run", false, "%s kt=%u: %s", arm_label, kt,
                             ft_tt_error_string(rc));
        goto cleanup;
    }

    rc = device_read(context, context->dram[DRAM_BANK_OUTPUT], OUTPUT_BASE, out_tiles,
                     out_bytes);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("fidelity sweep read back", false, "%s kt=%u: %s", arm_label, kt,
                             ft_tt_error_string(rc));
        goto cleanup;
    }
    {
        size_t poison_bytes = poison_remaining((const unsigned char *)out_tiles, out_bytes);
        uint32_t element;

        for (element = 0; element < M_ROWS * N_COLS; element++)
            actual[element] = ft_float_from_bf16(
                out_tiles[ft_tt_tile_face_index(element / N_COLS, element % N_COLS)]);

        {
            size_t bad_expected = count_non_finite(expected, M_ROWS * N_COLS);
            size_t bad_actual = count_non_finite(actual, M_ROWS * N_COLS);
            /* A handful of the 2048 output BYTES legitimately equal the 0xA5 poison byte
             * by chance in real bf16 data (~1/256 of bytes, so ~8 of 2048 expected) — this
             * is a byte-level scan, not a word-level one, so "still poisoned" is a RATE,
             * not a count, exactly test_ops.c's poison_bytes*100 < out_bytes convention.
             * The first cut of this file used `poison_bytes == 0u`, which failed on
             * essentially every real run for this reason alone — caught by inspecting the
             * very first sweep's log rather than trusting a wall of FAILs. */
            int poison_cleared = poison_bytes * 100u < out_bytes;

            ft_tt_test_check_fmt("fidelity sweep output is real",
                                 poison_cleared && bad_expected == 0u && bad_actual == 0u,
                                 "%s kt=%u: %zu/%zu poison bytes left, %zu/%u non-finite "
                                 "expected, %zu/%u non-finite actual", arm_label, kt,
                                 poison_bytes, out_bytes, bad_expected, M_ROWS * N_COLS,
                                 bad_actual, M_ROWS * N_COLS);
            if (!poison_cleared || bad_expected != 0u || bad_actual != 0u)
                goto cleanup;
        }

        *out_pcc = pearson(expected, actual, M_ROWS * N_COLS);
        *out_max_abs = max_abs_difference(expected, actual, M_ROWS * N_COLS);
        ft_tt_test_step("%s kt=%3u (K=%4u): PCC %.9f  max-abs %.6g", arm_label, kt, k,
                        *out_pcc, *out_max_abs);
    }

cleanup:
    free(expected);
    free(actual);
    free(a_tiles);
    free(b_tiles);
    free(out_tiles);
}

/* ================================================================
 * One arm: boot ITS OWN firmware+kernel directory (dst_accum_mode changes what the
 * firmware's math/pack startup programs, so firmware and kernel must come from the same
 * build — ft_tt_program_directory.h), run every Kt point, park, release.
 * ================================================================ */

static const uint32_t KT_POINTS[] = {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u, 24u,
                                     32u, 48u, 64u, 96u, 128u, 144u};
#define KT_POINT_COUNT (sizeof(KT_POINTS) / sizeof(KT_POINTS[0]))

typedef struct fidelity_arm {
    const char *label;
    const char *subdirectory; /**< under FT_TT_PROGRAM_DIR */
} fidelity_arm;

static const fidelity_arm ARMS[] = {
    {"HiFi4 dst16 (shipped)", "matmul_fidelity_hifi4_dst16"},
    {"LoFi  dst16",           "matmul_fidelity_lofi_dst16"},
    {"HiFi4 dst32",           "matmul_fidelity_hifi4_dst32"},
    {"LoFi  dst32",           "matmul_fidelity_lofi_dst32"},
};
#define ARM_COUNT (sizeof(ARMS) / sizeof(ARMS[0]))

static double MAX_ABS[ARM_COUNT][KT_POINT_COUNT];
static double PCC[ARM_COUNT][KT_POINT_COUNT];

static void run_arm(op_context *context, size_t arm_index, const char *program_root,
                    const float *base_a, const float *base_b)
{
    char directory[512];
    ft_tt_firmware_set firmware; /* local, not context->firmware — each arm boots its OWN
                                  * firmware+kernel pair from its OWN directory
                                  * (ft_tt_program_directory.h: mixing is a wedged core,
                                  * not a configuration), so nothing carries over between
                                  * arms in the loop that calls this. */
    ft_tt_program_directory loaded;
    int rc;
    size_t point;

    memset(&firmware, 0, sizeof(firmware));
    snprintf(directory, sizeof(directory), "%s/%s", program_root, ARMS[arm_index].subdirectory);

    rc = ft_tt_firmware_set_load(directory, &firmware);
    ft_tt_test_check_fmt("fidelity sweep firmware loads", rc == FT_TT_OK, "%s: %s from %s",
                         ARMS[arm_index].label, ft_tt_error_string(rc), directory);
    if (rc != FT_TT_OK)
        return;

    rc = ft_tt_core_boot_firmware(&context->session.noc, context->core_x, context->core_y,
                                  &firmware, NULL);
    ft_tt_test_check_fmt("fidelity sweep firmware boots", rc == FT_TT_OK, "%s: %s",
                         ARMS[arm_index].label, ft_tt_error_string(rc));
    if (rc != FT_TT_OK) {
        ft_tt_firmware_set_release(&firmware);
        return;
    }

    rc = ft_tt_program_directory_load(directory, 0x1Fu, &loaded);
    ft_tt_test_check_fmt("fidelity sweep program loads", rc == FT_TT_OK, "%s: %s",
                         ARMS[arm_index].label, ft_tt_error_string(rc));
    if (rc == FT_TT_OK) {
        for (point = 0; point < KT_POINT_COUNT; point++)
            run_k_point(context, &loaded.program, ARMS[arm_index].label, base_a, base_b,
                       KT_POINTS[point], &PCC[arm_index][point], &MAX_ABS[arm_index][point]);
        ft_tt_program_directory_release(&loaded);
    }

    ft_tt_core_park(&context->session.noc, context->core_x, context->core_y);
    ft_tt_firmware_set_release(&firmware);
}

/**
 * The summary table this whole file exists to print: every arm's max-abs at every Kt,
 * side by side, so the K-scaling question and the 2x2 DST_ACCUM_MODE question can both be
 * read off one table instead of grepped out of interleaved per-run lines.
 */
static void print_summary_table(void)
{
    size_t point, arm;

    printf("\n=== max-abs error, all arms, all Kt (K = Kt*32) ===\n");
    printf("%6s", "Kt");
    for (arm = 0; arm < ARM_COUNT; arm++)
        printf("  %-22s", ARMS[arm].label);
    printf("\n");
    for (point = 0; point < KT_POINT_COUNT; point++) {
        printf("%6u", KT_POINTS[point]);
        for (arm = 0; arm < ARM_COUNT; arm++)
            printf("  %-22.6g", MAX_ABS[arm][point]);
        printf("\n");
    }
    printf("\n=== PCC, all arms, all Kt ===\n");
    printf("%6s", "Kt");
    for (arm = 0; arm < ARM_COUNT; arm++)
        printf("  %-22s", ARMS[arm].label);
    printf("\n");
    for (point = 0; point < KT_POINT_COUNT; point++) {
        printf("%6u", KT_POINTS[point]);
        for (arm = 0; arm < ARM_COUNT; arm++)
            printf("  %-22.9f", PCC[arm][point]);
        printf("\n");
    }
}

/** FT_TT_PROGRAM_DIR, else tt/programs (where build_tensix_program.sh trees go) relative to
 *  tt/tests, the directory these tests are run from. */
static const char *program_root(void)
{
    const char *value = getenv("FT_TT_PROGRAM_DIR");

    return value ? value : "../programs";
}

int main(void)
{
    op_context context;
    float *base_a = malloc((size_t)M_ROWS * K_MAX_ELEMENTS * sizeof(float));
    float *base_b = malloc((size_t)K_MAX_ELEMENTS * N_COLS * sizeof(float));
    uint32_t tensix_index = 101u;
    uint32_t bank;
    size_t index;
    int rc;

    memset(&context, 0, sizeof(context));
    if (!base_a || !base_b) {
        ft_tt_test_check_fmt("fidelity sweep allocation", false, "out of memory");
        return ft_tt_test_finish();
    }

    ft_tt_test_step("Fidelity/DST_ACCUM_MODE explanatory sweep — %zu arms x %zu Kt points",
                    ARM_COUNT, KT_POINT_COUNT);

    /* ONE draw of operands, shared by every arm and every Kt point (the prefix of it).
     * Scaled *0.5 and bf16-rounded, same convention as test_ops.c, so the values fed to
     * the card are exactly what the reference is scored against. */
    random_seed(0xF1DE71Au);
    for (index = 0; index < (size_t)M_ROWS * K_MAX_ELEMENTS; index++)
        base_a[index] = random_value() * 0.5f;
    for (index = 0; index < (size_t)K_MAX_ELEMENTS * N_COLS; index++)
        base_b[index] = random_value() * 0.5f;
    round_to_bf16(base_a, (size_t)M_ROWS * K_MAX_ELEMENTS);
    round_to_bf16(base_b, (size_t)K_MAX_ELEMENTS * N_COLS);

    rc = ft_tt_hardware_session_open(&context.session);
    ft_tt_test_check_fmt("card opens", rc == FT_TT_OK, "%s", ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    (void)ft_tt_tensix_index_from_environment(&tensix_index);
    rc = ft_tt_hardware_session_tensix(&context.session, tensix_index, &context.core_x,
                                       &context.core_y);
    ft_tt_test_check_fmt("Tensix resolves from live telemetry", rc == FT_TT_OK,
                         "index %u -> translated (%u, %u): %s", tensix_index,
                         (unsigned)context.core_x, (unsigned)context.core_y,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto close;

    /* A harness that names its core owns it — per-core, so this sweep runs
     * alongside any other suite on a different Tensix. */
    rc = ft_tt_hardware_session_claim_cores(&context.session, &tensix_index, 1u,
                                            "fidelity sweep", NULL, NULL);
    ft_tt_test_check_fmt("the Tensix is claimed for this run", rc == FT_TT_OK, "index %u: %s",
                         tensix_index, ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto close;

    for (bank = 0; bank < 8u; bank++) {
        rc = ft_tt_coord_dram_bank(&context.session.grid, bank, FT_TT_COORD_TRANSLATED,
                                   &context.dram[bank]);
        if (rc != FT_TT_OK)
            break;
    }
    ft_tt_test_check_fmt("DRAM bank coordinates derive", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto close;

    for (index = 0; index < ARM_COUNT; index++)
        run_arm(&context, index, program_root(), base_a, base_b);

    print_summary_table();

close:
    ft_tt_hardware_session_close(&context.session);
done:
    free(base_a);
    free(base_b);
    return ft_tt_test_finish();
}
