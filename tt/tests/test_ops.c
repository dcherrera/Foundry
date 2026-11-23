/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_ops.c
 * @brief Phase 4's acceptance: RMS norm, matmul and a BFP8 matmul run on the card and are
 *   scored against a CPU fp32 reference. PCC >= 0.99 per op, on real MiniCPM5-1B shapes.
 *
 * ---------------------------------------------------------------------------
 * WHAT MAKES THE REFERENCE INDEPENDENT
 * ---------------------------------------------------------------------------
 * dev notes §3: "any metric computed on a round trip through your own code is measuring
 * your code's self-consistency, not its correctness." So:
 *
 *  - the reference is a plain nested-loop fp32 computation in this file, written from the
 *    op's definition, not from the kernel;
 *  - it consumes the values the CARD will actually see. For a bf16 operand that is the
 *    bf16-rounded value, and for a BFP8 weight it is ft_bfp_unpack_tiles()'s output —
 *    otherwise the score would be dominated by quantization error the kernel is not
 *    responsible for. Quantization is Phase 3's measurement and it already passed;
 *    this one is about the kernel;
 *  - the comparison runs on UNTILIZED data, so a face-ordering mistake cannot cancel. PCC
 *    is blind to a consistent permutation (dev notes §3), and both this and Phase 3's
 *    gate would miss one; what catches it here is that the reference never goes through
 *    ft_tt_tilize at all — it indexes a plain row-major array.
 *
 * ---------------------------------------------------------------------------
 * WHAT WOULD MAKE THIS VACUOUS, AND WHAT STOPS IT
 * ---------------------------------------------------------------------------
 *  - **A destination that already holds the right answer.** Every output buffer is
 *    poisoned to 0xA5 before the launch and the poison pattern is checked for at the end,
 *    so a kernel that never ran scores ~0, not 1.
 *  - **Data that cannot tell layouts apart.** Inputs are random with full mantissas, never
 *    small integers: a bf16-exact array is bit-identical to its own bf16 rendering with
 *    zeros interleaved, which made two deliberately-different builds byte-equal once
 *    (bring-up log §11.2).
 *  - **A shape that hides an index bug.** Matmul runs every MiniCPM5-1B shape including
 *    the two non-square ones the model actually uses (attn_q is 2048x1536 and attn_output
 *    1536x2048), plus Nt=8, which is where an Mt/Nt transposition stops being invisible.
 *
 * ---------------------------------------------------------------------------
 * WHY THERE IS A TIMING COLUMN AT ALL
 * ---------------------------------------------------------------------------
 * The §8.1 checkpoint is a performance question answered with a correctness harness, so
 * the two are measured in the same run and against the same PCC gate — a fidelity or
 * prefetch change that buys time and loses accuracy has to be visible as a loss in the
 * same table, not in a separate benchmark nobody scores.
 *
 * Two numbers are reported per shape. **ns/pair** is total wall time over Mt*Nt*Kt
 * tile-pairs, which is how docs/tt_bringup_log.md §14.5 computed its 849 ns and is
 * therefore the comparable one. **ns/pair-net** subtracts the fixed per-launch cost
 * measured by the `smoke` shape in the SAME run — five kernel images uploaded over an
 * x1 Gen2 link, the CB config, five runtime-arg blocks and the go/done handshake — which
 * is what Phase 5 removes and what would otherwise flatter or flatten a reader change as
 * the reader gets faster.
 *
 * Environment:
 *   FT_TT_TENSIX_INDEX      which enabled Tensix to drive (default 2 — indices 0 and 1
 *                           belong to the Phase 2 suites)
 *   FT_TT_PROGRAM_DIR       where the program directories are
 *   FT_TT_FTTW              the FTTW artifact; skipped, with a report line, if absent
 *   FT_TT_OPS               comma-separated subset of rms_norm, matmul, matmul_bfp8,
 *                           matmul_bfp4, mt_sweep, reduce_scaler, reduce_dot. Unset runs the first
 *                           four; the Mt sweep is opt-in because it is long, and
 *                           reduce_scaler is opt-in because it is a MEASUREMENT of the
 *                           reduce primitive's scaler semantics, not an acceptance
 *   FT_TT_MATMUL_PREFETCH   K-tiles per NoC barrier in the reader (default 1, the
 *                           per-pair barrier Phase 4 measured)
 *   FT_TT_OP_REPEATS        run each matmul shape this many times and keep the FASTEST
 *                           (default 1). The host is two 1.8 GHz cores also driving eight
 *                           K80s, so a single sample carries scheduler noise; the minimum
 *                           is the sample least contaminated by it
 *   FT_TT_B_NMAJOR          1 stores the weight's TILE GRID column-major (tile index
 *                           nt*Kt + kt) and tells the reader, so a prefetch block's B
 *                           tiles are one transaction instead of `depth`. Both halves
 *                           move together: the flag without the layout reads the right
 *                           bytes from the wrong addresses, and nothing errors
 *   FT_TT_MT_LIST           token blocks the Mt sweep walks, in tiles (default
 *                           1,2,4,8,16,32). Mt=1 is the 32 rows every matmul measured
 *                           before 2026-08-12 ran at
 *   FT_TT_DUMP_DIR          write each shape's raw output tiles here, byte for byte as
 *                           they came off the card. Two runs differing only in a LAYOUT
 *                           must produce identical files — PCC cannot make that claim,
 *                           because it is blind to a consistent permutation (dev notes §3)
 *
 * @status Results in docs/tt_bringup_log.md §14 and §16.
 * @issues None
 * @todo None
 */

/* clock_gettime(CLOCK_MONOTONIC) is POSIX.1-2008, not C99. */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../convert/ft_bfp.h"
#include "../convert/ft_float_convert.h"
#include "../ft_tt_fttw.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_matmul.h"
#include "../ft_tt_op_rms_norm.h"
#include "../ft_tt_tilize.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define TILE_DIM       FT_TT_OP_TILE_DIM
#define TILE_ELEMENTS  (TILE_DIM * TILE_DIM)
#define PCC_ACCEPT     0.99
#define POISON_BYTE    0xA5
#define LAUNCH_TIMEOUT 20000u

/** DRAM banks 2..7 are this worker's; 0 and 1 belong to the Phase 2 follow-ups. */
#define DRAM_BANK_ACTIVATION 2u
#define DRAM_BANK_WEIGHT     3u
#define DRAM_BANK_OUTPUT     4u
#define DRAM_BANK_CONSTANT   5u

/** Where in a bank each buffer starts. Generous strides, one buffer per bank. */
#define DRAM_BUFFER_BASE 0x00100000u

/* ================================================================
 * Run configuration, read once from the environment
 * ================================================================ */

/** How many Mt values one sweep may name. */
#define MT_SWEEP_MAX 12

/** The knobs §16's sweeps turn. Defaults reproduce the Phase 4 run exactly. */
typedef struct op_run_settings {
    uint32_t prefetch_tiles; /**< FT_TT_MATMUL_PREFETCH */
    uint32_t repeats;        /**< FT_TT_OP_REPEATS */
    /** FT_TT_B_NMAJOR — store the weight's tile grid column-major and tell the reader,
     *  so a prefetch block's B tiles are one transaction instead of `depth`. This is a
     *  property of the BYTES IN DRAM, so the harness has to lay them out that way as
     *  well as pass the flag; setting only one of the two is a wrong answer. */
    int b_n_major;
    /** FT_TT_MT_LIST — the token-block sizes the Mt sweep walks, in tiles. */
    uint32_t mt_list[MT_SWEEP_MAX];
    uint32_t mt_count;
    /** FT_TT_DUMP_DIR — where to write each shape's raw output tiles, exactly as they
     *  came off the card. Two runs that differ only in a LAYOUT must produce
     *  byte-identical dumps; PCC cannot see a consistent permutation (dev notes §3) and
     *  "still >= 0.99" is not a proof of neutrality. Unset writes nothing. */
    const char *dump_directory;
    int run_rms_norm;
    /** FT_TT_OPS=reduce_scaler — the SCALER-SEMANTICS experiment (plan Phase 7.9 item 2).
     *  Off by default even when FT_TT_OPS is unset: it is a MEASUREMENT, not an acceptance,
     *  and it deliberately feeds the rms_norm program a scaler tile no product path ever
     *  builds. */
    int run_reduce_scaler;
    /** FT_TT_OPS=reduce_dot — the reduce's ACCUMULATION PRECISION at the GDN's own 128-term
     *  length. Opt-in for the same reason: a measurement, not an acceptance. */
    int run_reduce_dot;
    int run_matmul;
    int run_matmul_bfp8;
    int run_matmul_bfp4;
    int run_mt_sweep;
} op_run_settings;

static op_run_settings settings = {1u, 1u, 0, {1u, 2u, 4u, 8u, 16u, 32u, 0, 0, 0, 0, 0, 0}, 6u,
                                   NULL, 1, 0, 0, 1, 1, 1, 0};

/** Milliseconds of fixed per-launch cost, measured by the smoke shape in this same run.
 *  Zero until it has been measured, and the smoke row itself reports without it. */
static double launch_overhead_ms;

/** An unsigned environment value, or `fallback` when unset, empty or unparseable. */
static uint32_t environment_count(const char *name, uint32_t fallback)
{
    const char *value = getenv(name);
    unsigned long parsed;
    char *end = NULL;

    if (!value || value[0] == '\0')
        return fallback;
    parsed = strtoul(value, &end, 10);
    if (end == value || parsed == 0ul || parsed > 0xFFFFFFFFul)
        return fallback;
    return (uint32_t)parsed;
}

/** Whether `name` appears as a whole comma-separated token of `list`. Whole-token
 *  matching matters here: "matmul_bfp8" contains "matmul" as a substring, and a
 *  substring match would silently run an op the caller did not ask for. */
static int list_contains(const char *list, const char *name)
{
    size_t length = strlen(name);
    const char *cursor = list;

    while (*cursor != '\0') {
        const char *comma = strchr(cursor, ',');
        size_t span = comma ? (size_t)(comma - cursor) : strlen(cursor);

        if (span == length && memcmp(cursor, name, length) == 0)
            return 1;
        if (!comma)
            break;
        cursor = comma + 1;
    }
    return 0;
}

/** Parse "1,2,4,8" into the Mt sweep list. Anything unparseable leaves the default. */
static void read_mt_list(const char *text)
{
    uint32_t parsed[MT_SWEEP_MAX];
    uint32_t count = 0;
    const char *cursor = text;

    while (*cursor != '\0' && count < MT_SWEEP_MAX) {
        char *end = NULL;
        unsigned long value = strtoul(cursor, &end, 10);

        if (end == cursor || value == 0ul || value > 4096ul)
            return;
        parsed[count++] = (uint32_t)value;
        if (*end == '\0')
            break;
        if (*end != ',')
            return;
        cursor = end + 1;
    }
    if (count == 0u)
        return;
    memcpy(settings.mt_list, parsed, count * sizeof(parsed[0]));
    settings.mt_count = count;
}

/** FT_TT_OPS is a comma-separated allow-list; unset runs everything but the Mt sweep. */
static void read_settings(void)
{
    const char *selection = getenv("FT_TT_OPS");
    const char *mt_list = getenv("FT_TT_MT_LIST");
    const char *dump = getenv("FT_TT_DUMP_DIR");

    settings.prefetch_tiles = environment_count("FT_TT_MATMUL_PREFETCH", 1u);
    settings.repeats = environment_count("FT_TT_OP_REPEATS", 1u);
    settings.b_n_major = environment_count("FT_TT_B_NMAJOR", 0u) != 0u;
    if (mt_list && mt_list[0] != '\0')
        read_mt_list(mt_list);
    if (dump && dump[0] != '\0')
        settings.dump_directory = dump;
    if (selection && selection[0] != '\0') {
        settings.run_rms_norm = list_contains(selection, "rms_norm");
        settings.run_reduce_scaler = list_contains(selection, "reduce_scaler");
        settings.run_reduce_dot = list_contains(selection, "reduce_dot");
        settings.run_matmul = list_contains(selection, "matmul");
        settings.run_matmul_bfp8 = list_contains(selection, "matmul_bfp8");
        settings.run_matmul_bfp4 = list_contains(selection, "matmul_bfp4");
        settings.run_mt_sweep = list_contains(selection, "mt_sweep");
    }
}

/* ================================================================
 * A deterministic random source, so a failure is reproducible
 * ================================================================ */

static uint64_t random_state = 0x9E3779B97F4A7C15ull;

static void random_seed(uint64_t seed)
{
    random_state = seed ? seed : 1u;
}

/** xorshift64*, and then a value in [-1, 1) with a full mantissa — NOT a small integer,
 *  for the reason the file header gives. */
static float random_value(void)
{
    random_state ^= random_state >> 12;
    random_state ^= random_state << 25;
    random_state ^= random_state >> 27;
    return (float)((double)(random_state * 0x2545F4914F6CDD1Dull >> 11) /
                       (double)(1ull << 53) * 2.0 -
                   1.0);
}

/* ================================================================
 * Scoring
 * ================================================================ */

/** Pearson correlation of two arrays. The Phase 4 acceptance metric. */
static double pearson(const float *a, const float *b, size_t count)
{
    double mean_a = 0.0;
    double mean_b = 0.0;
    double cov = 0.0;
    double var_a = 0.0;
    double var_b = 0.0;
    size_t index;

    if (count == 0u)
        return 0.0;
    for (index = 0; index < count; index++) {
        mean_a += a[index];
        mean_b += b[index];
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
        return 0.0;
    return cov / sqrt(var_a * var_b);
}

/** Largest absolute difference, reported alongside PCC because a correlation is
 *  insensitive to a systematic scale error (bring-up log §12.5 makes the same point). */
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

/**
 * A one-line post-mortem for a comparison that produced NaN.
 *
 * PCC and max-abs both collapse to NaN/inf if a SINGLE element on either side is not
 * finite, so "PCC -nan, max abs inf" is the same output whether one element is wrong or
 * all of them are — which is exactly the ambiguity that costs a session. This says how
 * many elements on each side are not finite and prints the first offending pair.
 *
 * @param out  buffer for the description; always NUL-terminated
 * @return     `out`, or "" when both sides are entirely finite
 */
static const char *describe_non_finite(const float *expected, const float *actual, size_t count,
                                       char *out, size_t out_bytes)
{
    size_t expected_bad = 0;
    size_t actual_bad = 0;
    size_t first = count;
    size_t index;

    out[0] = '\0';
    for (index = 0; index < count; index++) {
        int bad_e = !isfinite(expected[index]);
        int bad_a = !isfinite(actual[index]);

        expected_bad += (size_t)bad_e;
        actual_bad += (size_t)bad_a;
        if ((bad_e || bad_a) && first == count)
            first = index;
    }
    if (first == count)
        return out;
    snprintf(out, out_bytes,
             ", NON-FINITE: %zu/%zu expected, %zu/%zu actual, first at %zu "
             "(expected %g, actual %g)",
             expected_bad, count, actual_bad, count, first, (double)expected[first],
             (double)actual[first]);
    return out;
}

/**
 * Where the non-finite words sit in TILE space, which is the only space the device works
 * in. The untilized index is a fair place to score and a useless place to diagnose: a
 * fixed offset inside a tile shows up at a different untilized index in every shape.
 *
 * @param tiles  the face-ordered bf16 buffer exactly as it came off the card
 */
static void report_tile_positions(const char *label, const uint16_t *tiles, size_t tile_count)
{
    char list[160];
    size_t reported = 0;
    size_t total = 0;
    size_t tile;
    int used = 0;

    list[0] = '\0';
    for (tile = 0; tile < tile_count; tile++) {
        size_t inside;

        for (inside = 0; inside < TILE_ELEMENTS; inside++) {
            uint16_t word = tiles[tile * TILE_ELEMENTS + inside];

            if ((word & 0x7F80u) != 0x7F80u)
                continue;
            total++;
            if (reported < 10u && used >= 0 && (size_t)used < sizeof(list)) {
                used += snprintf(list + used, sizeof(list) - (size_t)used, " %zu:%zu=0x%04X", tile,
                                 inside, (unsigned)word);
                reported++;
            }
        }
    }
    if (total == 0u)
        return;
    ft_tt_test_step("%s: %zu non-finite words in %zu tiles, first at tile:offset%s", label, total,
                    tile_count, list);
}

/* ================================================================
 * bf16 tile helpers
 * ================================================================ */

/** fp32 row-major matrix -> face-ordered bf16 tiles. Returns tile count. */
static uint32_t to_bf16_tiles(const float *matrix, uint32_t rows, uint32_t cols, uint16_t *out)
{
    uint16_t *strip = malloc((size_t)cols * TILE_DIM * sizeof(uint16_t));
    uint32_t tiles_across = ft_tt_tile_count_for(cols);
    uint32_t block;
    uint32_t blocks = rows / TILE_DIM;

    if (!strip)
        return 0u;
    for (block = 0; block < blocks; block++) {
        uint32_t row;
        uint32_t column;

        for (row = 0; row < TILE_DIM; row++)
            for (column = 0; column < cols; column++)
                strip[row * cols + column] =
                    ft_bf16_from_float(matrix[(block * TILE_DIM + row) * cols + column]);
        ft_tt_tilize_strip(strip, sizeof(uint16_t), TILE_DIM, cols, cols * sizeof(uint16_t),
                           out + (size_t)block * tiles_across * TILE_ELEMENTS);
    }
    free(strip);
    return blocks * tiles_across;
}

/** Face-ordered bf16 tiles -> fp32 row-major matrix. */
static void from_bf16_tiles(const uint16_t *tiles, uint32_t rows, uint32_t cols, float *out)
{
    uint16_t *strip = malloc((size_t)cols * TILE_DIM * sizeof(uint16_t));
    uint32_t tiles_across = ft_tt_tile_count_for(cols);
    uint32_t blocks = rows / TILE_DIM;
    uint32_t block;

    if (!strip)
        return;
    for (block = 0; block < blocks; block++) {
        uint32_t row;
        uint32_t column;

        ft_tt_untilize_strip(tiles + (size_t)block * tiles_across * TILE_ELEMENTS,
                             sizeof(uint16_t), TILE_DIM, cols, cols * sizeof(uint16_t), strip);
        for (row = 0; row < TILE_DIM; row++)
            for (column = 0; column < cols; column++)
                out[(block * TILE_DIM + row) * cols + column] =
                    ft_float_from_bf16(strip[row * cols + column]);
    }
    free(strip);
}

/** Round a matrix through bf16 in place: what the card will actually see. */
static void round_to_bf16(float *matrix, size_t count)
{
    size_t index;

    for (index = 0; index < count; index++)
        matrix[index] = ft_float_from_bf16(ft_bf16_from_float(matrix[index]));
}

/* ================================================================
 * Weight tile-grid layout
 * ================================================================ */

/**
 * Reorder a Kt x Nt tile grid from K-major (index kt*Nt + nt) to N-major (nt*Kt + kt).
 *
 * This is a PERMUTATION OF WHOLE TILES and nothing else — no element inside a tile moves,
 * no value changes, and it works on any format because the unit is `tile_bytes`. That is
 * the point: a reader given the matching flag then sees a prefetch block's B tiles as one
 * contiguous run, and the answer it computes must be bit-identical to the K-major arm.
 * If it is not, the layout is not what either side thinks it is.
 *
 * @param source      Kt*Nt tiles in K-major order
 * @param destination Kt*Nt tiles in N-major order; must not alias `source`
 */
static void permute_tiles_to_n_major(const void *source, void *destination, uint32_t tile_rows,
                                     uint32_t tile_cols, size_t tile_bytes)
{
    const unsigned char *from = source;
    unsigned char *to = destination;
    uint32_t kt;

    for (kt = 0; kt < tile_rows; kt++) {
        uint32_t nt;

        for (nt = 0; nt < tile_cols; nt++)
            memcpy(to + ((size_t)nt * tile_rows + kt) * tile_bytes,
                   from + ((size_t)kt * tile_cols + nt) * tile_bytes, tile_bytes);
    }
}

/**
 * Write one shape's raw output tiles, exactly as they came off the card.
 *
 * The bit-neutrality claim for a layout change is "these two runs produced the same
 * BYTES", not "both scored above the gate". PCC is blind to a consistent permutation
 * (dev notes §3: 0.992 correct order vs 0.031 face-permuted, and only reading the written
 * artifact back could tell them apart), so a comparison of scores cannot make the claim
 * and a comparison of dumps can.
 */
static void dump_raw_tiles(const char *label, const void *tiles, size_t bytes)
{
    char path[640];
    char safe[128];
    size_t index;
    FILE *file;

    if (!settings.dump_directory)
        return;
    for (index = 0; index + 1 < sizeof(safe) && label[index] != '\0'; index++)
        safe[index] = (label[index] == '/' || label[index] == ' ' || label[index] == '(' ||
                       label[index] == ')') ? '_' : label[index];
    safe[index] = '\0';
    snprintf(path, sizeof(path), "%s/%s.bin", settings.dump_directory, safe);
    file = fopen(path, "wb");
    if (!file) {
        ft_tt_test_step("dump: cannot open %s", path);
        return;
    }
    if (fwrite(tiles, 1, bytes, file) != bytes)
        ft_tt_test_step("dump: short write to %s", path);
    fclose(file);
}

/* ================================================================
 * CPU references — written from the op's definition, not the kernel
 * ================================================================ */

static void reference_rms_norm(const float *x, const float *gamma, uint32_t rows, uint32_t cols,
                               float eps, float *out)
{
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * cols;
        double sum = 0.0;
        double scale;
        uint32_t column;

        for (column = 0; column < cols; column++)
            sum += (double)source[column] * (double)source[column];
        scale = 1.0 / sqrt(sum / (double)cols + (double)eps);
        for (column = 0; column < cols; column++)
            out[(size_t)row * cols + column] = (float)((double)source[column] * scale) *
                                               gamma[column];
    }
}

/**
 * C = A x B in fp32, accumulated in double, written from the definition of a matmul.
 *
 * The loop nest is row / inner / column, NOT the textbook row / column / inner, and the
 * swap is exact rather than an approximation: with a per-column double accumulator, each
 * output element still sums its K terms in ascending `inner` order, which is the same
 * sequence of double additions the scalar version performed. So the result is
 * BIT-IDENTICAL, and the check that it is stays cheap — every PCC in
 * docs/tt_bringup_log.md §16.3-16.4 must reproduce to the digit, and they do.
 *
 * The reason to bother: `b` is now walked contiguously instead of with an n-element
 * stride, which is what makes the Mt sweep affordable at all. At Mt=32 on ffn_gate/up the
 * reference is 7.2 G multiply-adds on a 1.8 GHz no-AVX core, and the strided order spends
 * most of it missing cache.
 */
static void reference_matmul(const float *a, const float *b, uint32_t m, uint32_t k, uint32_t n,
                             float *out)
{
    double *accumulator = malloc((size_t)n * sizeof(double));
    uint32_t row;

    if (!accumulator)
        return;
    for (row = 0; row < m; row++) {
        uint32_t column;
        uint32_t inner;

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
 * Device plumbing
 * ================================================================ */

typedef struct op_context {
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    uint16_t core_x;
    uint16_t core_y;
    ft_tt_noc_coord dram[8];
} op_context;

/** Fill a device buffer with the poison pattern, so a kernel that never ran is visible. */
static int poison(const op_context *context, ft_tt_noc_coord where, uint32_t address,
                  uint32_t bytes)
{
    unsigned char *pattern = malloc(bytes);
    int rc;

    if (!pattern)
        return FT_TT_ENOMEM;
    memset(pattern, POISON_BYTE, bytes);
    rc = context->session.noc.write(context->session.noc.user, (uint16_t)where.x,
                                    (uint16_t)where.y,
                                    (uint64_t)address, pattern, bytes);
    free(pattern);
    return rc;
}

/** How many of `count` bytes still hold the poison pattern. */
static size_t poison_remaining(const unsigned char *data, size_t count)
{
    size_t remaining = 0;
    size_t index;

    for (index = 0; index < count; index++)
        if (data[index] == POISON_BYTE)
            remaining++;
    return remaining;
}

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
                                    (uint16_t)where.y,
                                        (uint64_t)address + offset, cursor + offset, chunk);
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
                                    (uint16_t)where.y,
                                       (uint64_t)address + offset, cursor + offset, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

/**
 * Prove the host↔DRAM path before scoring anything computed through it.
 *
 * Every op's score is `write operands -> run -> read output`, so a corrupt transfer and a
 * corrupt kernel produce the same failing PCC. This writes the same kind of payload the
 * ops use — full-mantissa bf16, never small integers (bring-up log §11.2) — to the output
 * bank at the address the ops use, reads it straight back and counts the bytes that
 * changed. A failure here says the ops were never the question.
 */
static void check_dram_round_trip(op_context *context)
{
    const size_t bytes = 64u * FT_TT_OP_TILE_BYTES_BF16; /* 64 tiles: attn_q's output */
    uint16_t *written = malloc(bytes);
    uint16_t *read_back = malloc(bytes);
    size_t words = bytes / sizeof(uint16_t);
    size_t mismatches = 0;
    size_t first = words;
    size_t index;
    int rc;

    if (!written || !read_back) {
        ft_tt_test_check("dram round trip allocation", false, "out of memory");
        goto cleanup;
    }
    random_seed(0xD4A11u);
    for (index = 0; index < words; index++)
        written[index] = ft_bf16_from_float(random_value());

    rc = device_write(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE, written, bytes);
    memset(read_back, 0, bytes);
    rc |= device_read(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE, read_back, bytes);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("dram round trip transfers", false, "%s", ft_tt_error_string(rc));
        goto cleanup;
    }

    for (index = 0; index < words; index++) {
        if (read_back[index] != written[index]) {
            mismatches++;
            if (first == words)
                first = index;
        }
    }
    ft_tt_test_check_fmt("host->DRAM->host round trip is byte-exact", mismatches == 0u,
                         "%zu of %zu bf16 words differ%s", mismatches, words,
                         mismatches ? "" : " — the transfer path is not the suspect");
    if (mismatches)
        ft_tt_test_check_fmt("dram round trip first mismatch", false,
                             "word %zu: wrote 0x%04X, read 0x%04X", first,
                             (unsigned)written[first], (unsigned)read_back[first]);

cleanup:
    free(written);
    free(read_back);
}

static double elapsed_ms(struct timespec start, struct timespec end)
{
    return (double)(end.tv_sec - start.tv_sec) * 1000.0 +
           (double)(end.tv_nsec - start.tv_nsec) / 1000000.0;
}

/* ================================================================
 * The three op tests
 * ================================================================ */

static void test_rms_norm(op_context *context, const char *program_directory,
                          const ft_tt_fttw *artifact)
{
    const uint32_t rows = 64u;
    const uint32_t cols = 1536u; /* MiniCPM5-1B's hidden size */
    const uint32_t tiles_across = cols / TILE_DIM;
    const uint32_t tile_count = (rows / TILE_DIM) * tiles_across;
    const float eps = 1e-5f;
    ft_tt_program_directory loaded;
    ft_tt_op_rms_norm_request request;
    float *x = malloc((size_t)rows * cols * sizeof(float));
    float *gamma = malloc((size_t)cols * sizeof(float));
    float *expected = malloc((size_t)rows * cols * sizeof(float));
    float *actual = malloc((size_t)rows * cols * sizeof(float));
    uint16_t *tiles = malloc((size_t)tile_count * TILE_ELEMENTS * sizeof(uint16_t));
    uint16_t *gamma_tiles = malloc((size_t)tiles_across * TILE_ELEMENTS * sizeof(uint16_t));
    float *gamma_row = malloc((size_t)tiles_across * TILE_DIM * sizeof(float));
    unsigned char constants[2u * FT_TT_OP_TILE_BYTES_BF16];
    struct timespec start;
    struct timespec end;
    size_t element_count = (size_t)rows * cols;
    uint32_t index;
    int rc;

    ft_tt_test_step("rms_norm: %u x %u, eps %g, %u tiles", rows, cols, (double)eps, tile_count);
    if (!x || !gamma || !expected || !actual || !tiles || !gamma_tiles || !gamma_row) {
        ft_tt_test_check("rms_norm host allocation", false, "out of memory");
        goto cleanup;
    }

    rc = ft_tt_program_directory_load(program_directory, 0x1Fu, &loaded);
    ft_tt_test_check_fmt("rms_norm program loads from disk", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto cleanup;

    random_seed(20260812u);
    for (index = 0; index < element_count; index++)
        x[index] = random_value() * 4.0f;
    for (index = 0; index < cols; index++)
        gamma[index] = 0.5f + random_value() * 0.25f;

    /* A real norm weight when the artifact is available: it is the tensor this op will
     * actually be given, and it exercises the loader's broadcast expansion end to end. */
    if (artifact) {
        const ft_tt_artifact_entry *entry = ft_tt_fttw_find(artifact, "blk.0.attn_norm.weight");

        if (entry && entry->n_dims == 1u && entry->dims[0] == cols) {
            const float *source =
                (const float *)(const void *)(artifact->base + entry->data_offset);

            memcpy(gamma, source, (size_t)cols * sizeof(float));
            ft_tt_test_check("rms_norm gamma is blk.0.attn_norm.weight from the artifact", true,
                             NULL);
        }
    }

    /* The reference consumes what the card sees: bf16 activations and a bf16 gamma. */
    round_to_bf16(x, element_count);
    for (index = 0; index < cols; index++)
        gamma_row[index] = ft_float_from_bf16(ft_bf16_from_float(gamma[index]));
    reference_rms_norm(x, gamma_row, rows, cols, eps, expected);

    to_bf16_tiles(x, rows, cols, tiles);
    for (index = 0; index < tiles_across; index++) {
        uint16_t plain[TILE_ELEMENTS];
        uint32_t row;
        uint32_t column;

        for (column = 0; column < TILE_DIM; column++) {
            uint16_t value = ft_bf16_from_float(gamma_row[index * TILE_DIM + column]);

            for (row = 0; row < TILE_DIM; row++)
                plain[row * TILE_DIM + column] = value;
        }
        ft_tt_tilize_strip(plain, sizeof(uint16_t), TILE_DIM, TILE_DIM,
                           TILE_DIM * sizeof(uint16_t), gamma_tiles + index * TILE_ELEMENTS);
    }

    rc = ft_tt_op_rms_norm_build_constants(cols, eps, constants, sizeof(constants));
    ft_tt_test_check_fmt("rms_norm constant tiles build", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto release;

    rc = device_write(context, context->dram[DRAM_BANK_ACTIVATION], DRAM_BUFFER_BASE, tiles,
                      (size_t)tile_count * FT_TT_OP_TILE_BYTES_BF16);
    rc |= device_write(context, context->dram[DRAM_BANK_WEIGHT], DRAM_BUFFER_BASE, gamma_tiles,
                       (size_t)tiles_across * FT_TT_OP_TILE_BYTES_BF16);
    rc |= device_write(context, context->dram[DRAM_BANK_CONSTANT], DRAM_BUFFER_BASE, constants,
                       sizeof(constants));
    rc |= poison(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE,
                 tile_count * FT_TT_OP_TILE_BYTES_BF16);
    ft_tt_test_check_fmt("rms_norm operands uploaded and output poisoned", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto release;

    memset(&request, 0, sizeof(request));
    request.row_blocks = rows / TILE_DIM;
    request.tile_cols = tiles_across;
    request.activation.x = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].x;
    request.activation.y = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].y;
    request.activation.address = DRAM_BUFFER_BASE;
    request.activation.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    request.gamma = request.activation;
    request.gamma.x = (uint16_t)context->dram[DRAM_BANK_WEIGHT].x;
    request.gamma.y = (uint16_t)context->dram[DRAM_BANK_WEIGHT].y;
    request.constants = request.activation;
    request.constants.x = (uint16_t)context->dram[DRAM_BANK_CONSTANT].x;
    request.constants.y = (uint16_t)context->dram[DRAM_BANK_CONSTANT].y;
    request.output = request.activation;
    request.output.x = (uint16_t)context->dram[DRAM_BANK_OUTPUT].x;
    request.output.y = (uint16_t)context->dram[DRAM_BANK_OUTPUT].y;

    clock_gettime(CLOCK_MONOTONIC, &start);
    rc = ft_tt_op_rms_norm_run(&context->session.noc, context->core_x, context->core_y,
                               &loaded.program, &request, LAUNCH_TIMEOUT);
    clock_gettime(CLOCK_MONOTONIC, &end);
    ft_tt_test_check_fmt("rms_norm completes", rc == FT_TT_OK, "%s", ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto release;

    rc = device_read(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE, tiles,
                     (size_t)tile_count * FT_TT_OP_TILE_BYTES_BF16);
    ft_tt_test_check_fmt("rms_norm output read back", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto release;

    {
        size_t poison_bytes = poison_remaining((const unsigned char *)tiles,
                                               (size_t)tile_count * FT_TT_OP_TILE_BYTES_BF16);
        ft_tt_test_check_fmt("rms_norm wrote over the poison", poison_bytes * 100u <
                                                                   (size_t)tile_count *
                                                                       FT_TT_OP_TILE_BYTES_BF16,
                             "%zu bytes still 0xA5", poison_bytes);
    }

    from_bf16_tiles(tiles, rows, cols, actual);
    {
        double pcc = pearson(expected, actual, element_count);
        double worst = max_abs_difference(expected, actual, element_count);
        char non_finite[192];

        ft_tt_test_check_fmt("rms_norm PCC >= 0.99 vs the CPU fp32 reference", pcc >= PCC_ACCEPT,
                             "PCC %.9f, max abs error %.6g, %.3f ms%s", pcc, worst,
                             elapsed_ms(start, end),
                             describe_non_finite(expected, actual, element_count, non_finite,
                                                 sizeof(non_finite)));
    }

release:
    ft_tt_program_directory_release(&loaded);
cleanup:
    free(x);
    free(gamma);
    free(expected);
    free(actual);
    free(tiles);
    free(gamma_tiles);
    free(gamma_row);
}

/* ================================================================
 * The SCALER-SEMANTICS experiment (plan Phase 7.9, item 2)
 * ================================================================
 *
 * WHAT IS BEING ASKED. `ft_tensix_unpack_ab_reduce_row(data_cb, scaler_cb, data_tile,
 * scaler_tile)` puts the SCALER on unpacker A and the DATA on unpacker B, and sets
 * Haloize on the scaler's unpacker only, so that the row reduce is a matrix multiply
 * "against a scaler column" (ft_tensix_compute_reduce.h:567, :716). Two readings fit that
 * sentence and they are not the same op:
 *
 *   A -- PER-POSITION. The scaler contributes a 32-long vector indexed by the reduce
 *        position, so reduce_row(data, scaler) computes  sum_c data[r][c] * s[c].
 *   B -- BROADCAST. The scaler contributes ONE constant, so it computes
 *        s0 * sum_c data[r][c].
 *
 * Nothing in the product can tell them apart, because every scaler this backend has ever
 * built is UNIFORM: rms_norm's is 1/W in every position (ft_tt_op_rms_norm.c:49-51), and a
 * uniform scaler makes A and B the same number. That is why the experiment has to build a
 * scaler no product path builds.
 *
 * WHY IT MATTERS, in one line: if A holds, the Gated-Delta-Net's two dot-product passes
 * (`pred_j = sum_i S[j][i] k_i` and `y_j = sum_i S[j][i] q_i`) are four calls to a shipped
 * primitive with `k`/`q` in the scaler CB and no new kernel code at all.
 *
 * THE PROBE, and why this shape. Data is the 32x32 IDENTITY and gamma is all-ones, so the
 * reduce over row `r` sees exactly ONE non-zero, at column `r`. The rms_norm program then
 * writes `out[r][c] = x[r][c] * rsqrt(R[r])`, whose only non-zero is the diagonal, so
 *
 *     R[r] = 1 / out[r][r]^2
 *
 * recovers the reduce result for every row from one descriptor -- and R is then the
 * EFFECTIVE SCALER VECTOR the hardware used, element by element. Under A it equals the
 * s[] we wrote (and a wrong index order would show as a permutation of it rather than as a
 * bare mismatch); under B it is the constant s[0] in all 32 rows. eps is 0, so the extra
 * accumulated eps reduce contributes zero under either reading.
 *
 * THE SEED IS pid+clock ON PURPOSE. A uniform -- or merely a fixed -- pattern is the
 * vacuous version of this experiment: it cannot distinguish the two readings, and a pass
 * could be inherited from a previous run's DRAM. Two independent seeds are run, and the
 * verdict has to be the same one twice.
 */

/** One 32x32 bf16 tile whose value at [row][col] is `column_values[col]`, tilized. */
static int build_column_scaler_tile(const float *column_values, unsigned char *out)
{
    uint16_t plain[TILE_ELEMENTS];
    uint32_t row;
    uint32_t column;

    for (column = 0; column < TILE_DIM; column++) {
        uint16_t value = ft_bf16_from_float(column_values[column]);

        for (row = 0; row < TILE_DIM; row++)
            plain[row * TILE_DIM + column] = value;
    }
    return ft_tt_tilize_strip(plain, sizeof(uint16_t), TILE_DIM, TILE_DIM,
                              TILE_DIM * sizeof(uint16_t), out);
}

/** Run ONE rms_norm descriptor at 32x32 over the identity, with `scaler` as the reduce
 *  scaler tile and a zero eps tile, and recover R[r] = 1/out[r][r]^2 into `recovered`.
 *  Returns FT_TT_OK when the descriptor ran and the diagonal came back finite. */
static int measure_effective_scaler(op_context *context, const ft_tt_kernel_program *program,
                                    const float *scaler_columns, float *recovered)
{
    unsigned char constants[2u * FT_TT_OP_TILE_BYTES_BF16];
    uint16_t identity_tile[TILE_ELEMENTS];
    uint16_t gamma_tile[TILE_ELEMENTS];
    uint16_t out_tile[TILE_ELEMENTS];
    float identity[TILE_DIM * TILE_DIM];
    float ones[TILE_DIM];
    float actual[TILE_DIM * TILE_DIM];
    ft_tt_op_rms_norm_request request;
    uint32_t index;
    int rc;

    for (index = 0; index < TILE_DIM * TILE_DIM; index++)
        identity[index] = 0.0f;
    for (index = 0; index < TILE_DIM; index++) {
        identity[index * TILE_DIM + index] = 1.0f;
        ones[index] = 1.0f;
    }
    to_bf16_tiles(identity, TILE_DIM, TILE_DIM, identity_tile);
    if (build_column_scaler_tile(ones, (unsigned char *)gamma_tile) != 0)
        return FT_TT_EINVAL;

    if (build_column_scaler_tile(scaler_columns, constants) != 0)
        return FT_TT_EINVAL;
    memset(constants + FT_TT_OP_TILE_BYTES_BF16, 0, FT_TT_OP_TILE_BYTES_BF16);

    rc = device_write(context, context->dram[DRAM_BANK_ACTIVATION], DRAM_BUFFER_BASE,
                      identity_tile, FT_TT_OP_TILE_BYTES_BF16);
    rc |= device_write(context, context->dram[DRAM_BANK_WEIGHT], DRAM_BUFFER_BASE, gamma_tile,
                       FT_TT_OP_TILE_BYTES_BF16);
    rc |= device_write(context, context->dram[DRAM_BANK_CONSTANT], DRAM_BUFFER_BASE, constants,
                       sizeof(constants));
    rc |= poison(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE,
                 FT_TT_OP_TILE_BYTES_BF16);
    if (rc != FT_TT_OK)
        return rc;

    memset(&request, 0, sizeof(request));
    request.row_blocks = 1u;
    request.tile_cols = 1u;
    request.activation.x = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].x;
    request.activation.y = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].y;
    request.activation.address = DRAM_BUFFER_BASE;
    request.activation.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    request.gamma = request.activation;
    request.gamma.x = (uint16_t)context->dram[DRAM_BANK_WEIGHT].x;
    request.gamma.y = (uint16_t)context->dram[DRAM_BANK_WEIGHT].y;
    request.constants = request.activation;
    request.constants.x = (uint16_t)context->dram[DRAM_BANK_CONSTANT].x;
    request.constants.y = (uint16_t)context->dram[DRAM_BANK_CONSTANT].y;
    request.output = request.activation;
    request.output.x = (uint16_t)context->dram[DRAM_BANK_OUTPUT].x;
    request.output.y = (uint16_t)context->dram[DRAM_BANK_OUTPUT].y;

    rc = ft_tt_op_rms_norm_run(&context->session.noc, context->core_x, context->core_y, program,
                               &request, LAUNCH_TIMEOUT);
    if (rc != FT_TT_OK)
        return rc;

    rc = device_read(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE, out_tile,
                     FT_TT_OP_TILE_BYTES_BF16);
    if (rc != FT_TT_OK)
        return rc;

    from_bf16_tiles(out_tile, TILE_DIM, TILE_DIM, actual);
    for (index = 0; index < TILE_DIM; index++) {
        float diagonal = actual[index * TILE_DIM + index];

        if (!(diagonal > 0.0f) || !(diagonal < 1.0e30f))
            return FT_TT_EIO;
        recovered[index] = 1.0f / (diagonal * diagonal);
    }
    return FT_TT_OK;
}

/** Worst relative error of `measured` against `expected`, over 32 rows. */
static double worst_relative(const float *measured, const float *expected)
{
    double worst = 0.0;
    uint32_t index;

    for (index = 0; index < TILE_DIM; index++) {
        double reference = fabs((double)expected[index]);
        double error = fabs((double)measured[index] - (double)expected[index]);

        if (reference > 0.0)
            error /= reference;
        if (error > worst)
            worst = error;
    }
    return worst;
}

static void describe_vector(const char *label, const float *values, char *out, size_t out_bytes)
{
    int used = snprintf(out, out_bytes, "%s", label);
    uint32_t index;

    for (index = 0; index < 8u && used > 0 && (size_t)used < out_bytes; index++)
        used += snprintf(out + used, out_bytes - (size_t)used, " %.4f", (double)values[index]);
    if (used > 0 && (size_t)used < out_bytes)
        snprintf(out + used, out_bytes - (size_t)used, " ...");
}

static void test_reduce_scaler(op_context *context, const char *program_directory)
{
    ft_tt_program_directory loaded;
    float scaler[TILE_DIM];
    float uniform[TILE_DIM];
    float recovered[TILE_DIM];
    float per_position[TILE_DIM];
    float broadcast[TILE_DIM];
    char line[256];
    uint64_t seed;
    uint32_t attempt;
    uint32_t index;
    int rc;

    ft_tt_test_step("reduce scaler semantics: is the reduce scaler PER-POSITION or a "
                    "BROADCAST constant?");

    rc = ft_tt_program_directory_load(program_directory, 0x1Fu, &loaded);
    ft_tt_test_check_fmt("reduce_scaler: rms_norm program loads from disk", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    /* CONTROL, and it is not optional: a uniform 1/32 scaler is what the product builds, so
     * this arm is the one whose answer is known independently of the question. Both readings
     * predict rsqrt(1/32) = 5.656854 on the diagonal, i.e. R[r] = 1/32 for every row. If this
     * fails, the 32x32 single-tile shape or the identity probe is wrong and NOTHING that
     * follows means anything. */
    for (index = 0; index < TILE_DIM; index++)
        uniform[index] = 1.0f / (float)TILE_DIM;
    rc = measure_effective_scaler(context, &loaded.program, uniform, recovered);
    ft_tt_test_check_fmt("reduce_scaler: the uniform-scaler control runs", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto release;
    describe_vector("recovered R[0..7] =", recovered, line, sizeof(line));
    ft_tt_test_check_fmt("reduce_scaler: control recovers R[r] = 1/32 on every row",
                         worst_relative(recovered, uniform) < 0.02, "%s (worst rel %.5f)", line,
                         worst_relative(recovered, uniform));

    /* THE DISCRIMINATOR, twice, with independent pid+clock seeds. */
    for (attempt = 0; attempt < 2u; attempt++) {
        struct timespec now;
        int per_position_matches;
        int broadcast_matches;
        double per_position_error;
        double broadcast_error;

        clock_gettime(CLOCK_MONOTONIC, &now);
        {
            const uint64_t golden = UINT64_C(0x9E3779B97F4A7C15);

            seed = (uint64_t)(unsigned)getpid() * golden ^ (uint64_t)now.tv_nsec ^
                   ((uint64_t)attempt << 43);
        }
        random_seed(seed);
        for (index = 0; index < TILE_DIM; index++) {
            /* [0.25, 4), rounded to bf16 so the value the card sees is the value we
             * predict against, and spread over four octaves so a wrong element is not
             * within bf16 noise of a right one. */
            float value = 0.25f + (random_value() + 1.0f) * 1.875f;

            scaler[index] = ft_float_from_bf16(ft_bf16_from_float(value));
        }

        rc = measure_effective_scaler(context, &loaded.program, scaler, recovered);
        ft_tt_test_check_fmt("reduce_scaler: the non-uniform arm runs", rc == FT_TT_OK,
                             "arm %u, seed 0x%016llX: %s", attempt, (unsigned long long)seed,
                             ft_tt_error_string(rc));
        if (rc != FT_TT_OK)
            goto release;

        for (index = 0; index < TILE_DIM; index++) {
            per_position[index] = scaler[index];
            broadcast[index] = scaler[0];
        }
        per_position_error = worst_relative(recovered, per_position);
        broadcast_error = worst_relative(recovered, broadcast);
        per_position_matches = per_position_error < 0.02;
        broadcast_matches = broadcast_error < 0.02;

        describe_vector("wrote s[0..7]  =", scaler, line, sizeof(line));
        ft_tt_test_step("%s", line);
        describe_vector("recovered R[0..7] =", recovered, line, sizeof(line));
        ft_tt_test_step("%s", line);

        ft_tt_test_check_fmt(
            "reduce_scaler: EXACTLY ONE reading matches",
            per_position_matches != broadcast_matches,
            "arm %u: PER-POSITION worst rel %.5f (%s), BROADCAST worst rel %.5f (%s)", attempt,
            per_position_error, per_position_matches ? "MATCH" : "no",
            broadcast_error, broadcast_matches ? "MATCH" : "no");
        ft_tt_test_step("VERDICT arm %u: the reduce scaler is %s", attempt,
                        per_position_matches
                            ? "PER-POSITION -- reduce_row(S, k) computes sum_i S[j][i]*k[i]"
                            : (broadcast_matches ? "a BROADCAST CONSTANT -- s0 * sum_i S[j][i]"
                                                 : "NEITHER -- see the vectors above"));
    }

release:
    ft_tt_program_directory_release(&loaded);
}

/* ================================================================
 * THE REDUCE'S ACCUMULATION PRECISION, at the Gated-Delta-Net's own length
 * ================================================================
 *
 * The scaler experiment above settles what the reduce COMPUTES. This settles whether the
 * answer is accurate enough to carry a RECURRENCE, which is a different question and the
 * one that decides the GDN's data formats.
 *
 * WHY IT IS NOT OPTIONAL. The GDN's two dot passes are 128-long — `pred_j = sum_i
 * S[j][i]*k[i]` over `state_size = 128` — and their result is written BACK into the state
 * every token. Pantheon's CPU oracle accumulates exactly these dots in DOUBLE on purpose;
 * its own comment names a token-60-90 divergence onset for doing them in float32. A Tensix
 * arm cannot do double, so the question is not "is it exact" but "does a 128-term Tensix
 * dot stay inside the bar the CUDA arm is held to, 1e-3 relative".
 *
 * WHAT IS MEASURED, and how the parts are separated. The probe runs the same rms_norm
 * descriptor at TWO widths, 32 columns (one tile) and 128 (four tiles, the GDN's own
 * length), and recovers the reduce result per row. Reported per width:
 *
 *   - card vs a DOUBLE-accumulated CPU reference   — the bar
 *   - a FLOAT32-accumulated CPU reference vs the same double one — what the accumulator
 *     width alone costs, computed on the host over identical inputs
 *
 * If the card's error is flat from 32 to 128 columns, the accumulation is not the term that
 * grows and the reduce is carrying its partial sums wider than its operands. If it grows
 * with the term count, the GDN needs its state CBs in Float32 rather than Float16_b.
 *
 * THE REFERENCE CONSUMES WHAT THE CARD SEES. rms_norm's first pass squares x into DST and
 * PACKS that product back to the intermediate CB before reducing it, so the reference
 * squares in float and rounds to bf16 too. Skipping that would charge the reduce for a
 * rounding that happens before it.
 *
 * WHAT THIS ARM CANNOT SEE, stated so the number is not over-read: it runs the SHIPPED
 * rms_norm program, which was built with the emitter's defaults — `math_fidelity 4` and
 * `dst_accum_mode 0`. dst_accum_mode 0 is the LESS accurate destination mode; the rope
 * cbtable turns it on and records 13-35x accuracy for zero wall clock. So this is a
 * PESSIMISTIC bound on what a GDN program built with `dst_accum_mode 1` would do, not an
 * estimate of it.
 */

/** Recover the reduce result per row from an rms_norm run at `tile_cols` tiles wide. */
static int measure_reduce_dot(op_context *context, const ft_tt_kernel_program *program,
                              uint32_t tile_cols, const float *x, const float *scaler_columns,
                              float *recovered)
{
    const uint32_t cols = tile_cols * TILE_DIM;
    unsigned char *constants = malloc((size_t)(tile_cols + 1u) * FT_TT_OP_TILE_BYTES_BF16);
    uint16_t *tiles = malloc((size_t)tile_cols * TILE_ELEMENTS * sizeof(uint16_t));
    uint16_t *gamma_tiles = malloc((size_t)tile_cols * TILE_ELEMENTS * sizeof(uint16_t));
    float *actual = malloc((size_t)TILE_DIM * cols * sizeof(float));
    float ones[TILE_DIM];
    ft_tt_op_rms_norm_request request;
    uint32_t index;
    uint32_t row;
    int rc = FT_TT_ENOMEM;

    if (!constants || !tiles || !gamma_tiles || !actual)
        goto done;
    for (index = 0; index < TILE_DIM; index++)
        ones[index] = 1.0f;

    /* The scaler tile for tile-column `wt` carries s[wt*32 .. wt*32+31] across its columns —
     * the per-position reading the arm above established. rms_norm passes scaler tile 0
     * every iteration, so all tile-columns must share ONE scaler tile; the probe therefore
     * repeats the same 32-long pattern in every tile-column of the DATA's scaler, and the
     * reference below uses s[c % 32] to match. */
    if (build_column_scaler_tile(scaler_columns, constants) != 0)
        goto done;
    memset(constants + FT_TT_OP_TILE_BYTES_BF16, 0, FT_TT_OP_TILE_BYTES_BF16);

    to_bf16_tiles(x, TILE_DIM, cols, tiles);
    for (index = 0; index < tile_cols; index++) {
        if (build_column_scaler_tile(ones, (unsigned char *)(gamma_tiles +
                                                             index * TILE_ELEMENTS)) != 0)
            goto done;
    }

    rc = device_write(context, context->dram[DRAM_BANK_ACTIVATION], DRAM_BUFFER_BASE, tiles,
                      (size_t)tile_cols * FT_TT_OP_TILE_BYTES_BF16);
    rc |= device_write(context, context->dram[DRAM_BANK_WEIGHT], DRAM_BUFFER_BASE, gamma_tiles,
                       (size_t)tile_cols * FT_TT_OP_TILE_BYTES_BF16);
    rc |= device_write(context, context->dram[DRAM_BANK_CONSTANT], DRAM_BUFFER_BASE, constants,
                       2u * FT_TT_OP_TILE_BYTES_BF16);
    rc |= poison(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE,
                 tile_cols * FT_TT_OP_TILE_BYTES_BF16);
    if (rc != FT_TT_OK)
        goto done;

    memset(&request, 0, sizeof(request));
    request.row_blocks = 1u;
    request.tile_cols = tile_cols;
    request.activation.x = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].x;
    request.activation.y = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].y;
    request.activation.address = DRAM_BUFFER_BASE;
    request.activation.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    request.gamma = request.activation;
    request.gamma.x = (uint16_t)context->dram[DRAM_BANK_WEIGHT].x;
    request.gamma.y = (uint16_t)context->dram[DRAM_BANK_WEIGHT].y;
    request.constants = request.activation;
    request.constants.x = (uint16_t)context->dram[DRAM_BANK_CONSTANT].x;
    request.constants.y = (uint16_t)context->dram[DRAM_BANK_CONSTANT].y;
    request.output = request.activation;
    request.output.x = (uint16_t)context->dram[DRAM_BANK_OUTPUT].x;
    request.output.y = (uint16_t)context->dram[DRAM_BANK_OUTPUT].y;

    rc = ft_tt_op_rms_norm_run(&context->session.noc, context->core_x, context->core_y, program,
                               &request, LAUNCH_TIMEOUT);
    if (rc != FT_TT_OK)
        goto done;
    rc = device_read(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE, tiles,
                     (size_t)tile_cols * FT_TT_OP_TILE_BYTES_BF16);
    if (rc != FT_TT_OK)
        goto done;

    from_bf16_tiles(tiles, TILE_DIM, cols, actual);
    /* out[r][c] = x[r][c] * rsqrt(R[r]). Recover from the LARGEST |x| in the row: every
     * column carries the same R, and the biggest operand is the one whose bf16 quotient
     * carries the most significant bits. */
    for (row = 0; row < TILE_DIM; row++) {
        double best = 0.0;
        double ratio = 0.0;
        uint32_t column;

        for (column = 0; column < cols; column++) {
            double magnitude = fabs((double)x[row * cols + column]);
            double out_value = (double)actual[row * cols + column];

            if (magnitude > best && out_value != 0.0) {
                best = magnitude;
                ratio = (double)x[row * cols + column] / out_value;
            }
        }
        if (best == 0.0) {
            rc = FT_TT_EIO;
            goto done;
        }
        recovered[row] = (float)(ratio * ratio);
    }
    rc = FT_TT_OK;

done:
    free(constants);
    free(tiles);
    free(gamma_tiles);
    free(actual);
    return rc;
}

static void test_reduce_dot(op_context *context, const char *program_directory)
{
    const uint32_t widths[2] = {1u, 4u};
    ft_tt_program_directory loaded;
    float scaler[TILE_DIM];
    float *x = malloc((size_t)TILE_DIM * 4u * TILE_DIM * sizeof(float));
    float recovered[TILE_DIM];
    float reference_double[TILE_DIM];
    float reference_float[TILE_DIM];
    double observed[2] = {0.0, 0.0};
    uint32_t which;
    uint32_t index;
    int rc;

    ft_tt_test_step("reduce accumulation precision: does a 128-term dot hold the 1e-3 bar "
                    "the CUDA arm is held to?");
    if (!x) {
        ft_tt_test_check("reduce_dot host allocation", false, "out of memory");
        return;
    }
    rc = ft_tt_program_directory_load(program_directory, 0x1Fu, &loaded);
    ft_tt_test_check_fmt("reduce_dot: rms_norm program loads from disk", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK) {
        free(x);
        return;
    }

    random_seed(0xD0D0CAFEull);
    for (index = 0; index < TILE_DIM; index++)
        scaler[index] = ft_float_from_bf16(ft_bf16_from_float(0.5f + (random_value() + 1.0f) *
                                                                        0.75f));
    for (index = 0; index < TILE_DIM * 4u * TILE_DIM; index++)
        x[index] = ft_float_from_bf16(ft_bf16_from_float(random_value() * 2.0f));

    for (which = 0; which < 2u; which++) {
        const uint32_t tile_cols = widths[which];
        const uint32_t cols = tile_cols * TILE_DIM;
        uint32_t row;
        double card_error;
        double float_error;

        rc = measure_reduce_dot(context, &loaded.program, tile_cols, x, scaler, recovered);
        ft_tt_test_check_fmt("reduce_dot: the descriptor runs", rc == FT_TT_OK, "%u columns: %s",
                             cols, ft_tt_error_string(rc));
        if (rc != FT_TT_OK)
            break;

        for (row = 0; row < TILE_DIM; row++) {
            double accumulate_double = 0.0;
            float accumulate_float = 0.0f;
            uint32_t column;

            for (column = 0; column < cols; column++) {
                /* The card squares into DST and PACKS to bf16 before reducing. */
                float value = x[row * cols + column];
                float square = ft_float_from_bf16(ft_bf16_from_float(value * value));
                float weight = scaler[column % TILE_DIM];

                accumulate_double += (double)square * (double)weight;
                accumulate_float += square * weight;
            }
            reference_double[row] = (float)accumulate_double;
            reference_float[row] = accumulate_float;
        }
        card_error = worst_relative(recovered, reference_double);
        float_error = worst_relative(reference_float, reference_double);
        observed[which] = card_error;

        ft_tt_test_step("%u-term dot: card worst rel %.6f against a DOUBLE accumulation; a "
                        "HOST float32 accumulation of the same terms is %.9f off the same "
                        "double",
                        cols, card_error, float_error);
    }

    /* THE CHECK IS ABOUT GROWTH, NOT THE ABSOLUTE, and that is why two widths are run.
     *
     * The absolute figure CANNOT be attributed to the reduce. What the probe recovers is
     * `R = (x/out)^2` out of an rms_norm output, so it carries the SFPU's rsqrt
     * approximation, the bf16 pack of `out`, and -- because the shipped rms_norm program is
     * built with the emitter's defaults -- a `dst_accum_mode 0` destination, the LESS
     * accurate of the two modes. Every one of those is a per-ROW multiplicative term and
     * NONE of them depends on how many terms the dot has.
     *
     * The accumulation is the only term that scales with the term count. So a 4x longer dot
     * that costs no more error says the reduce carries its partial sums wider than its
     * operands, and 128 is not a length this primitive struggles at. That is the question
     * the Gated-Delta-Net actually needs answered: its dots are 128 long and their result is
     * written back into the carried state every token, so an error that compounded with
     * length would compound with the sequence too.
     *
     * WHAT THIS DOES NOT LICENSE, stated so the number is not over-read: it is NOT a claim
     * that a GDN arm meets 1e-3. It bounds the composite through a `dst_accum_mode 0`
     * program. A GDN CB table would set `dst_accum_mode 1`, which the rope table records as
     * 13-35x accuracy for zero wall clock -- and that has to be MEASURED on the GDN's own
     * program rather than extrapolated from here.
     */
    if (rc == FT_TT_OK) {
        double growth = observed[0] > 0.0 ? observed[1] / observed[0] : 0.0;

        ft_tt_test_check_fmt("reduce_dot: 4x the terms does not cost more error — the "
                             "ACCUMULATION is not the term that grows",
                             growth > 0.0 && growth < 1.25,
                             "the 128-term error is %.3fx the 32-term error (%.6f vs %.6f); "
                             "the shared floor is rsqrt + the bf16 output pack + "
                             "dst_accum_mode 0, none of which scales with the term count",
                             growth, observed[1], observed[0]);
    }

    ft_tt_program_directory_release(&loaded);
    free(x);
}

/**
 * One matmul shape, in the decided orientation: A is [Mt*32 x Kt*32] and B is K x N.
 *
 * @param is_overhead_probe  record this shape's time as the run's fixed per-launch cost.
 *   The `smoke` shape is the probe: at 4 tile-pairs its own arithmetic is ~1% of its wall
 *   time, so what it measures is the five image uploads, the CB config, the runtime-arg
 *   blocks and the go/done handshake. It is an over-estimate of the overhead by that ~1%,
 *   which biases ns/pair-net DOWN — stated because a bias that flatters the result has to
 *   be named.
 */
static void run_matmul_shape(op_context *context, const ft_tt_kernel_program *program,
                             const char *label, uint32_t m, uint32_t k, uint32_t n,
                             int is_overhead_probe)
{
    ft_tt_op_matmul_request request;
    float *a = malloc((size_t)m * k * sizeof(float));
    float *b = malloc((size_t)k * n * sizeof(float));
    float *expected = malloc((size_t)m * n * sizeof(float));
    float *actual = malloc((size_t)m * n * sizeof(float));
    uint16_t *a_tiles = malloc((size_t)(m / TILE_DIM) * (k / TILE_DIM) * TILE_ELEMENTS *
                               sizeof(uint16_t));
    uint16_t *b_tiles = malloc((size_t)(k / TILE_DIM) * (n / TILE_DIM) * TILE_ELEMENTS *
                               sizeof(uint16_t));
    uint16_t *out_tiles = malloc((size_t)(m / TILE_DIM) * (n / TILE_DIM) * TILE_ELEMENTS *
                                 sizeof(uint16_t));
    size_t out_bytes = (size_t)(m / TILE_DIM) * (n / TILE_DIM) * FT_TT_OP_TILE_BYTES_BF16;
    struct timespec start;
    struct timespec end;
    double fastest_ms = 0.0;
    uint32_t index;
    int rc;

    if (!a || !b || !expected || !actual || !a_tiles || !b_tiles || !out_tiles) {
        ft_tt_test_check_fmt("matmul host allocation", false, "%s: out of memory", label);
        goto cleanup;
    }

    /* Scaled down so a Kt-long fp32 accumulation stays well inside bf16's range at the
     * output; the kernel accumulates in DST, not in fp32, and an overflow here would be
     * measuring the test's arithmetic rather than the kernel's. */
    random_seed(0x5EEDu + m + k + n);
    for (index = 0; index < m * k; index++)
        a[index] = random_value() * 0.5f;
    for (index = 0; index < k * n; index++)
        b[index] = random_value() * 0.5f;
    round_to_bf16(a, (size_t)m * k);
    round_to_bf16(b, (size_t)k * n);
    reference_matmul(a, b, m, k, n, expected);

    to_bf16_tiles(a, m, k, a_tiles);
    to_bf16_tiles(b, k, n, b_tiles);

    /* The N-major arm reorders the WEIGHT TILES IN DRAM and tells the reader so. Both
     * halves or neither: the flag alone reads the right bytes from the wrong places. */
    if (settings.b_n_major) {
        size_t b_bytes = (size_t)(k / TILE_DIM) * (n / TILE_DIM) * FT_TT_OP_TILE_BYTES_BF16;
        uint16_t *reordered = malloc(b_bytes);

        if (!reordered) {
            ft_tt_test_check_fmt("matmul N-major reorder", false, "%s: out of memory", label);
            goto cleanup;
        }
        permute_tiles_to_n_major(b_tiles, reordered, k / TILE_DIM, n / TILE_DIM,
                                 FT_TT_OP_TILE_BYTES_BF16);
        memcpy(b_tiles, reordered, b_bytes);
        free(reordered);
    }

    rc = device_write(context, context->dram[DRAM_BANK_ACTIVATION], DRAM_BUFFER_BASE, a_tiles,
                      (size_t)(m / TILE_DIM) * (k / TILE_DIM) * FT_TT_OP_TILE_BYTES_BF16);
    rc |= device_write(context, context->dram[DRAM_BANK_WEIGHT], DRAM_BUFFER_BASE, b_tiles,
                       (size_t)(k / TILE_DIM) * (n / TILE_DIM) * FT_TT_OP_TILE_BYTES_BF16);
    rc |= poison(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE, (uint32_t)out_bytes);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("matmul upload", false, "%s: %s", label, ft_tt_error_string(rc));
        goto cleanup;
    }

    memset(&request, 0, sizeof(request));
    request.tile_rows = m / TILE_DIM;
    request.tile_inner = k / TILE_DIM;
    request.tile_cols = n / TILE_DIM;
    request.prefetch_tiles = settings.prefetch_tiles;
    request.weight_n_major = (uint32_t)settings.b_n_major;
    request.activation.x = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].x;
    request.activation.y = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].y;
    request.activation.address = DRAM_BUFFER_BASE;
    request.activation.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    request.weight = request.activation;
    request.weight.x = (uint16_t)context->dram[DRAM_BANK_WEIGHT].x;
    request.weight.y = (uint16_t)context->dram[DRAM_BANK_WEIGHT].y;
    request.output = request.activation;
    request.output.x = (uint16_t)context->dram[DRAM_BANK_OUTPUT].x;
    request.output.y = (uint16_t)context->dram[DRAM_BANK_OUTPUT].y;

    /* Only the LAST repeat's output is scored, and the output buffer keeps the poison from
     * before the first launch: every repeat writes the same answer to the same place, so a
     * repeat that silently did nothing still leaves the previous repeat's result. That is
     * why the poison is not refreshed between repeats — refreshing it would let the last
     * repeat alone carry the whole check, and the check is that the op is repeatable. */
    for (index = 0; index < settings.repeats; index++) {
        double milliseconds;

        clock_gettime(CLOCK_MONOTONIC, &start);
        rc = ft_tt_op_matmul_run(&context->session.noc, context->core_x, context->core_y, program,
                                 &request, LAUNCH_TIMEOUT);
        clock_gettime(CLOCK_MONOTONIC, &end);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("matmul completes", false, "%s: %s", label,
                                 ft_tt_error_string(rc));
            goto cleanup;
        }
        milliseconds = elapsed_ms(start, end);
        if (index == 0u || milliseconds < fastest_ms)
            fastest_ms = milliseconds;
    }

    rc = device_read(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE, out_tiles,
                     out_bytes);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("matmul read back", false, "%s: %s", label, ft_tt_error_string(rc));
        goto cleanup;
    }
    from_bf16_tiles(out_tiles, m, n, actual);

    {
        double pcc = pearson(expected, actual, (size_t)m * n);
        double worst = max_abs_difference(expected, actual, (size_t)m * n);
        size_t poison_bytes = poison_remaining((const unsigned char *)out_tiles, out_bytes);
        double flops = 2.0 * (double)m * (double)k * (double)n;
        double pairs = (double)request.tile_rows * request.tile_cols * request.tile_inner;
        double net_ms = fastest_ms - launch_overhead_ms;
        uint32_t depth = ft_tt_op_matmul_prefetch_depth(request.tile_inner,
                                                        settings.prefetch_tiles);
        char non_finite[192];

        ft_tt_test_check_fmt("matmul PCC >= 0.99", pcc >= PCC_ACCEPT && poison_bytes * 100u < out_bytes,
                             "%s [%ux%u]x[%ux%u] Mt %u depth %u B%s: PCC %.9f, max abs %.6g, "
                             "%.3f ms, %.2f GFLOP/s, %.0f ns/pair, %.0f ns/pair-net, "
                             "%zu poison bytes left%s",
                             label, m, k, k, n, (unsigned)request.tile_rows, (unsigned)depth,
                             settings.b_n_major ? "N-major" : "K-major", pcc, worst, fastest_ms,
                             flops / (fastest_ms * 1e6), fastest_ms * 1e6 / pairs,
                             (net_ms > 0.0 ? net_ms : 0.0) * 1e6 / pairs, poison_bytes,
                             describe_non_finite(expected, actual, (size_t)m * n, non_finite,
                                                 sizeof(non_finite)));
        report_tile_positions(label, out_tiles, out_bytes / FT_TT_OP_TILE_BYTES_BF16);
        dump_raw_tiles(label, out_tiles, out_bytes);
        if (is_overhead_probe)
            launch_overhead_ms = fastest_ms;
    }

cleanup:
    free(a);
    free(b);
    free(expected);
    free(actual);
    free(a_tiles);
    free(b_tiles);
    free(out_tiles);
}

static void test_matmul(op_context *context, const char *program_directory)
{
    ft_tt_program_directory loaded;
    int rc = ft_tt_program_directory_load(program_directory, 0x1Fu, &loaded);

    ft_tt_test_step("matmul: every MiniCPM5-1B shape, 32 tokens, prefetch %u, %u repeat(s)",
                    settings.prefetch_tiles, settings.repeats);
    ft_tt_test_check_fmt("matmul program loads from disk", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    /* The shapes the model uses, as K x N after the orientation decision. The square-ish
     * ones would not catch an Mt/Nt transposition; attn_k/v at N=256 and attn_output at
     * K=2048 do. `smoke` runs first because it doubles as the fixed-cost probe every later
     * row's ns/pair-net subtracts. */
    run_matmul_shape(context, &loaded.program, "smoke", 32u, 64u, 64u, 1);
    run_matmul_shape(context, &loaded.program, "attn_q", 32u, 1536u, 2048u, 0);
    run_matmul_shape(context, &loaded.program, "attn_k/v", 32u, 1536u, 256u, 0);
    run_matmul_shape(context, &loaded.program, "attn_output", 32u, 2048u, 1536u, 0);
    run_matmul_shape(context, &loaded.program, "ffn_gate/up", 32u, 1536u, 4608u, 0);
    run_matmul_shape(context, &loaded.program, "ffn_down", 32u, 4608u, 1536u, 0);
    run_matmul_shape(context, &loaded.program, "decode (1 token padded)", 32u, 1536u, 2048u, 0);

    ft_tt_program_directory_release(&loaded);
}

/**
 * Every matmul ever measured here ran Mt = 1 — 32 rows, one tile of tokens. Prefill will
 * not be 32 rows, and ops 4-40 are written against whatever this op does at a real token
 * block, so the shape of the Mt curve is a Phase 6 input rather than a curiosity.
 *
 * Two shapes, because a sweep on one shape is a configuration: `ffn_gate/up` (Nt = 144,
 * where the B side dominates a block) and `attn_k/v` (Nt = 8, where it does not).
 *
 * **Mt does not touch L1 in this kernel and the sweep is what makes that a measurement.**
 * The two input CBs are sized `2 * depth` pages and the output CB two pages, none of
 * which mentions Mt — so the L1 ceiling this op has is on `depth`, and the ceiling on Mt
 * is DRAM and host memory. What grows with Mt is TRAFFIC: every B tile is re-read Mt
 * times, because the reader has no reuse. Whether that is free or linear is exactly what
 * the GFLOP/s column says.
 */
static void test_mt_sweep(op_context *context, const char *program_directory)
{
    static const struct {
        const char *label;
        uint32_t k;
        uint32_t n;
    } shapes[] = {
        {"mt ffn_gate/up", 1536u, 4608u},
        {"mt attn_k/v", 1536u, 256u},
    };
    ft_tt_program_directory loaded;
    int rc = ft_tt_program_directory_load(program_directory, 0x1Fu, &loaded);
    size_t shape;

    ft_tt_test_step("mt_sweep: token blocks from Mt=%u to Mt=%u, prefetch %u, %u repeat(s)",
                    settings.mt_list[0], settings.mt_list[settings.mt_count - 1u],
                    settings.prefetch_tiles, settings.repeats);
    ft_tt_test_check_fmt("mt_sweep program loads from disk", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    /* The same fixed-cost probe every matmul row subtracts; without it ns/pair-net for
     * the sweep would be measured against whatever the previous test happened to leave. */
    run_matmul_shape(context, &loaded.program, "smoke", 32u, 64u, 64u, 1);

    for (shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
        uint32_t index;

        for (index = 0; index < settings.mt_count; index++) {
            char label[96];
            uint32_t rows = settings.mt_list[index] * TILE_DIM;

            snprintf(label, sizeof(label), "%s Mt=%u", shapes[shape].label,
                     settings.mt_list[index]);
            run_matmul_shape(context, &loaded.program, label, rows, shapes[shape].k,
                             shapes[shape].n, 0);
        }
    }

    ft_tt_program_directory_release(&loaded);
}

/* ================================================================
 * Block-float matmul — Bfp8_b and Bfp4_b, weights consumed natively
 * ================================================================ */

/** One block-float shape: which artifact tensor supplies the values, and its K x N. */
typedef struct block_float_shape {
    const char *label;
    const char *tensor_name; /**< NULL runs on generated operands only */
    uint32_t k;
    uint32_t n;
} block_float_shape;

/**
 * Read a K x N window of an artifact tensor in the decided [in, out] orientation,
 * whatever block-float format the tensor is stored in, from EITHER artifact layout.
 *
 * Block float is stored swizzled — row-major inside each tile — because tt-metal's
 * packer does the face reordering itself. Where the tile lives depends on `layout`:
 *
 *   FT_TT_LAYOUT_SOURCE        GGUF order, rows = out, cols = in, row-major tile grid,
 *                              so element (out, in) is at tile (out/32)*tile_cols +
 *                              (in/32), offset (out%32)*32 + (in%32). This one has to be
 *                              transposed while it is read.
 *   FT_TT_LAYOUT_MATMUL_NMAJOR already [in x out] with a column-major tile grid, so
 *                              element (in, out) is at tile (out/32)*tile_rows + (in/32),
 *                              offset (in%32)*32 + (out%32). Nothing to transpose.
 *
 * The two branches producing the SAME numbers off two artifacts is what proves the
 * converter's transpose is the one the kernel wants, rather than a transpose that merely
 * round-trips through itself (dev notes §3: a metric computed on a round trip through
 * your own code measures self-consistency).
 *
 * @return 1 when the tensor was found, unpacked and read; 0 to keep generated operands
 */
static int read_artifact_weight_transposed(const ft_tt_fttw *artifact, const char *tensor_name,
                                           uint32_t k, uint32_t n, float *out)
{
    const ft_tt_artifact_entry *entry;
    const unsigned char *packed;
    uint32_t source_tiles;
    float *unpacked;
    ft_bfp_format format;
    uint32_t row;
    int ok = 0;

    if (!artifact || !tensor_name)
        return 0;
    entry = ft_tt_fttw_find(artifact, tensor_name);
    if (!entry)
        return 0;
    if (entry->format == FT_TT_FORMAT_BFP8_B)
        format = FT_BFP_BFP8_B;
    else if (entry->format == FT_TT_FORMAT_BFP4_B)
        format = FT_BFP_BFP4_B;
    else
        return 0;
    /* dims stays logical and in GGML order in both layouts: dims[0] = in_features,
     * dims[1] = out_features. */
    if (entry->dims[0] < k || entry->dims[1] < n)
        return 0;

    packed = artifact->base + entry->data_offset;
    source_tiles = entry->tile_rows * entry->tile_cols;
    unpacked = malloc((size_t)source_tiles * TILE_ELEMENTS * sizeof(float));
    if (!unpacked)
        return 0;
    if (ft_bfp_unpack_tiles(format, packed, source_tiles, unpacked) != FT_BFP_OK) {
        free(unpacked);
        return 0;
    }

    for (row = 0; row < k; row++) {
        uint32_t column;

        for (column = 0; column < n; column++) {
            uint32_t tile;
            uint32_t inside;

            if (entry->layout == FT_TT_LAYOUT_MATMUL_NMAJOR) {
                tile = (column / TILE_DIM) * entry->tile_rows + (row / TILE_DIM);
                inside = (row % TILE_DIM) * TILE_DIM + (column % TILE_DIM);
            } else {
                tile = (column / TILE_DIM) * entry->tile_cols + (row / TILE_DIM);
                inside = (column % TILE_DIM) * TILE_DIM + (row % TILE_DIM);
            }
            out[(size_t)row * n + column] = unpacked[(size_t)tile * TILE_ELEMENTS + inside];
        }
    }
    ok = 1;
    free(unpacked);
    return ok;
}

/**
 * One block-float matmul, scored against TWO references.
 *
 *  - **vs the values the FPU sees** (the weight unpacked back out of its own packed
 *    bytes). This is the KERNEL's score and the one the 0.99 gate applies to. Scoring
 *    against anything else would fold Phase 3's quantization error into a Phase 4
 *    measurement, which §14.3 already refused to do.
 *  - **vs the values before packing.** This is the END-TO-END score, i.e. what the
 *    format itself costs on this shape, and it is the number that would trigger Route B's
 *    fallback — which is PRECISION, not format. BFP4 has
 *    never been run, so the column has to exist before anyone can say it passed.
 *
 * Both are reported with max-abs beside them: dev notes §3 — PCC is the coarse column and
 * max-abs the sensitive one, and a format change is exactly the smooth per-element
 * perturbation PCC is worst at seeing.
 */
static void run_block_float_shape(op_context *context, const ft_tt_kernel_program *program,
                                  const ft_tt_fttw *artifact, ft_tt_tensor_format format,
                                  const block_float_shape *shape)
{
    const uint32_t m = 32u;
    const uint32_t k = shape->k;
    const uint32_t n = shape->n;
    const ft_bfp_format bfp = (format == FT_TT_FORMAT_BFP4_B) ? FT_BFP_BFP4_B : FT_BFP_BFP8_B;
    const uint32_t weight_tile_bytes = (format == FT_TT_FORMAT_BFP4_B)
                                           ? FT_TT_OP_TILE_BYTES_BFP4_B
                                           : FT_TT_OP_TILE_BYTES_BFP8_B;
    const char *format_name = (format == FT_TT_FORMAT_BFP4_B) ? "Bfp4_b" : "Bfp8_b";
    const uint32_t a_tiles_count = (m / TILE_DIM) * (k / TILE_DIM);
    const uint32_t b_tiles_count = (k / TILE_DIM) * (n / TILE_DIM);
    const uint32_t out_tiles_count = (m / TILE_DIM) * (n / TILE_DIM);
    ft_tt_op_matmul_request request;
    float *a = malloc((size_t)m * k * sizeof(float));
    float *b = malloc((size_t)k * n * sizeof(float));
    float *b_pre = malloc((size_t)k * n * sizeof(float));
    float *b_swizzled = malloc((size_t)b_tiles_count * TILE_ELEMENTS * sizeof(float));
    float *b_seen = malloc((size_t)b_tiles_count * TILE_ELEMENTS * sizeof(float));
    float *expected = malloc((size_t)m * n * sizeof(float));
    float *expected_pre = malloc((size_t)m * n * sizeof(float));
    float *actual = malloc((size_t)m * n * sizeof(float));
    uint16_t *a_tiles = malloc((size_t)a_tiles_count * TILE_ELEMENTS * sizeof(uint16_t));
    unsigned char *b_packed = malloc((size_t)b_tiles_count * weight_tile_bytes);
    uint16_t *out_tiles = malloc((size_t)out_tiles_count * TILE_ELEMENTS * sizeof(uint16_t));
    size_t out_bytes = (size_t)out_tiles_count * FT_TT_OP_TILE_BYTES_BF16;
    struct timespec start;
    struct timespec end;
    double fastest_ms = 0.0;
    uint32_t index;
    int from_artifact = 0;
    int rc;

    ft_tt_test_step("%s %s: %ux%u x %ux%u, weight consumed as %s natively", format_name,
                    shape->label, m, k, k, n, format_name);
    if (!a || !b || !b_pre || !b_swizzled || !b_seen || !expected || !expected_pre || !actual ||
        !a_tiles || !b_packed || !out_tiles) {
        ft_tt_test_check_fmt("block-float host allocation", false, "%s: out of memory",
                             shape->label);
        goto cleanup;
    }

    /* Full-mantissa random, never small integers: a bf16-exact array is bit-identical to
     * its own bf16 rendering with zeros interleaved, so integer data cannot tell two
     * different descriptor sets apart (dev notes §3, bring-up log §11.2). That matters
     * more here than anywhere — a 4-bit mantissa with a shared exponent is exactly the
     * format an integer operand would flatter. */
    random_seed(0xB1F0u + (uint64_t)format * 977u + k + n);
    for (index = 0; index < m * k; index++)
        a[index] = random_value() * 0.5f;
    for (index = 0; index < k * n; index++)
        b[index] = random_value() * 0.5f;

    from_artifact = read_artifact_weight_transposed(artifact, shape->tensor_name, k, n, b);
    if (shape->tensor_name)
        ft_tt_test_check_fmt("weight is a real tensor from the artifact", from_artifact != 0,
                             "%s, transposed into [in, out]%s", shape->tensor_name,
                             from_artifact ? ""
                                           : " — NOT FOUND, running on generated operands");

    round_to_bf16(a, (size_t)m * k);
    to_bf16_tiles(a, m, k, a_tiles);
    memcpy(b_pre, b, (size_t)k * n * sizeof(float));

    /* Pack the weight the way the artifact does — swizzled tiles into ft_bfp — then unpack
     * it again to get the values the FPU will see. */
    for (index = 0; index < (k / TILE_DIM); index++)
        ft_tt_tile_swizzle_strip(b + (size_t)index * TILE_DIM * n, sizeof(float), TILE_DIM, n,
                                 n * sizeof(float),
                                 b_swizzled + (size_t)index * (n / TILE_DIM) * TILE_ELEMENTS);
    rc = ft_bfp_pack_tiles(bfp, b_swizzled, b_tiles_count, b_packed);
    rc |= ft_bfp_unpack_tiles(bfp, b_packed, b_tiles_count, b_seen);
    ft_tt_test_check_fmt("weight packs and unpacks", rc == FT_BFP_OK, "%s %s: rc %d", format_name,
                         shape->label, rc);
    if (rc != FT_BFP_OK)
        goto cleanup;

    for (index = 0; index < b_tiles_count; index++) {
        uint32_t tile_row = index / (n / TILE_DIM);
        uint32_t tile_column = index % (n / TILE_DIM);
        uint32_t inside;

        for (inside = 0; inside < TILE_ELEMENTS; inside++)
            b[(size_t)(tile_row * TILE_DIM + inside / TILE_DIM) * n + tile_column * TILE_DIM +
              inside % TILE_DIM] = b_seen[(size_t)index * TILE_ELEMENTS + inside];
    }
    reference_matmul(a, b, m, k, n, expected);
    reference_matmul(a, b_pre, m, k, n, expected_pre);

    if (settings.b_n_major) {
        unsigned char *reordered = malloc((size_t)b_tiles_count * weight_tile_bytes);

        if (!reordered) {
            ft_tt_test_check("block-float N-major reorder", false, "out of memory");
            goto cleanup;
        }
        permute_tiles_to_n_major(b_packed, reordered, k / TILE_DIM, n / TILE_DIM,
                                 weight_tile_bytes);
        memcpy(b_packed, reordered, (size_t)b_tiles_count * weight_tile_bytes);
        free(reordered);
    }

    rc = device_write(context, context->dram[DRAM_BANK_ACTIVATION], DRAM_BUFFER_BASE, a_tiles,
                      (size_t)a_tiles_count * FT_TT_OP_TILE_BYTES_BF16);
    rc |= device_write(context, context->dram[DRAM_BANK_WEIGHT], DRAM_BUFFER_BASE, b_packed,
                       (size_t)b_tiles_count * weight_tile_bytes);
    rc |= poison(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE, (uint32_t)out_bytes);
    ft_tt_test_check_fmt("operands uploaded and output poisoned", rc == FT_TT_OK, "%s %s: %s",
                         format_name, shape->label, ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto cleanup;

    memset(&request, 0, sizeof(request));
    request.tile_rows = m / TILE_DIM;
    request.tile_inner = k / TILE_DIM;
    request.tile_cols = n / TILE_DIM;
    request.prefetch_tiles = settings.prefetch_tiles;
    request.weight_n_major = (uint32_t)settings.b_n_major;
    request.activation.x = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].x;
    request.activation.y = (uint16_t)context->dram[DRAM_BANK_ACTIVATION].y;
    request.activation.address = DRAM_BUFFER_BASE;
    request.activation.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    request.weight = request.activation;
    request.weight.x = (uint16_t)context->dram[DRAM_BANK_WEIGHT].x;
    request.weight.y = (uint16_t)context->dram[DRAM_BANK_WEIGHT].y;
    request.weight.tile_bytes = weight_tile_bytes;
    request.output = request.activation;
    request.output.x = (uint16_t)context->dram[DRAM_BANK_OUTPUT].x;
    request.output.y = (uint16_t)context->dram[DRAM_BANK_OUTPUT].y;

    for (index = 0; index < settings.repeats; index++) {
        double milliseconds;

        clock_gettime(CLOCK_MONOTONIC, &start);
        rc = ft_tt_op_matmul_run(&context->session.noc, context->core_x, context->core_y, program,
                                 &request, LAUNCH_TIMEOUT);
        clock_gettime(CLOCK_MONOTONIC, &end);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("block-float matmul completes", false, "%s %s: %s", format_name,
                                 shape->label, ft_tt_error_string(rc));
            goto cleanup;
        }
        milliseconds = elapsed_ms(start, end);
        if (index == 0u || milliseconds < fastest_ms)
            fastest_ms = milliseconds;
    }

    rc = device_read(context, context->dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE, out_tiles,
                     out_bytes);
    ft_tt_test_check_fmt("block-float output read back", rc == FT_TT_OK, "%s %s: %s", format_name,
                         shape->label, ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto cleanup;
    from_bf16_tiles(out_tiles, m, n, actual);

    {
        double pcc = pearson(expected, actual, (size_t)m * n);
        double worst = max_abs_difference(expected, actual, (size_t)m * n);
        double pcc_pre = pearson(expected_pre, actual, (size_t)m * n);
        double worst_pre = max_abs_difference(expected_pre, actual, (size_t)m * n);
        size_t poison_bytes = poison_remaining((const unsigned char *)out_tiles, out_bytes);
        double pairs = (double)request.tile_rows * request.tile_cols * request.tile_inner;
        double flops = 2.0 * (double)m * (double)k * (double)n;
        double net_ms = fastest_ms - launch_overhead_ms;
        uint32_t depth = ft_tt_op_matmul_prefetch_depth(request.tile_inner,
                                                        settings.prefetch_tiles);
        char label[160];
        char non_finite[192];

        snprintf(label, sizeof(label), "%s %s PCC >= 0.99 vs the CPU fp32 reference", format_name,
                 shape->label);
        ft_tt_test_check_fmt(label, pcc >= PCC_ACCEPT && poison_bytes * 100u < out_bytes,
                             "[%ux%u]x[%ux%u] depth %u B%s %s: KERNEL PCC %.9f max abs %.6g | "
                             "END-TO-END vs pre-pack PCC %.9f max abs %.6g | %.3f ms, "
                             "%.2f GFLOP/s, %.0f ns/pair, %.0f ns/pair-net, %zu poison left%s",
                             m, k, k, n, (unsigned)depth,
                             settings.b_n_major ? "N-major" : "K-major",
                             from_artifact ? "artifact" : "generated", pcc, worst, pcc_pre,
                             worst_pre, fastest_ms, flops / (fastest_ms * 1e6),
                             fastest_ms * 1e6 / pairs,
                             (net_ms > 0.0 ? net_ms : 0.0) * 1e6 / pairs, poison_bytes,
                             describe_non_finite(expected, actual, (size_t)m * n, non_finite,
                                                 sizeof(non_finite)));
        report_tile_positions(shape->label, out_tiles, out_bytes / FT_TT_OP_TILE_BYTES_BF16);
        dump_raw_tiles(shape->label, out_tiles, out_bytes);
    }

cleanup:
    free(a);
    free(b);
    free(b_pre);
    free(b_swizzled);
    free(b_seen);
    free(expected);
    free(expected_pre);
    free(actual);
    free(a_tiles);
    free(b_packed);
    free(out_tiles);
}

/**
 * The Bfp8_b acceptance: §14's shape, kept identical so its PCC stays comparable, plus
 * `ffn_down`, which is the other tensor class the ledger puts at BFP8.
 */
static void test_matmul_bfp8(op_context *context, const char *program_directory,
                             const ft_tt_fttw *artifact)
{
    static const block_float_shape shapes[] = {
        {"attn_q window", "blk.0.attn_q.weight", 1536u, 512u},
        {"ffn_down", "blk.0.ffn_down.weight", 4608u, 1536u},
    };
    ft_tt_program_directory loaded;
    int rc = ft_tt_program_directory_load(program_directory, 0x1Fu, &loaded);
    size_t index;

    ft_tt_test_check_fmt("matmul_bfp8 program loads from disk", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;
    for (index = 0; index < sizeof(shapes) / sizeof(shapes[0]); index++)
        run_block_float_shape(context, &loaded.program, artifact, FT_TT_FORMAT_BFP8_B,
                              &shapes[index]);
    ft_tt_program_directory_release(&loaded);
}

/**
 * **The Bfp4_b acceptance — the format that had never been run.**
 *
 * The shapes are the ones the production ledger ACTUALLY assigns to BFP4, not a
 * convenient small one: `FT_TT_ROLE_FFN_UP` is the only role that gets
 * FT_TT_FORMAT_BFP4_B (ft_tt_precision_ledger.c), and on MiniCPM5-1B that is
 * `ffn_gate` and `ffn_up`, both 1536 x 4608 in the decided [in, out] orientation. Both
 * are run, on their own real tensors out of the artifact rather than one of them twice,
 * because two draws of the same shape are what tell a shape result from a data result.
 * A narrow window is run as well: Nt = 16 is where an Mt/Nt confusion stops hiding.
 */
static void test_matmul_bfp4(op_context *context, const char *program_directory,
                             const ft_tt_fttw *artifact)
{
    static const block_float_shape shapes[] = {
        {"ffn_gate", "blk.0.ffn_gate.weight", 1536u, 4608u},
        {"ffn_up", "blk.0.ffn_up.weight", 1536u, 4608u},
        {"ffn_gate narrow", "blk.0.ffn_gate.weight", 1536u, 512u},
    };
    ft_tt_program_directory loaded;
    int rc = ft_tt_program_directory_load(program_directory, 0x1Fu, &loaded);
    size_t index;

    ft_tt_test_check_fmt("matmul_bfp4 program loads from disk", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;
    for (index = 0; index < sizeof(shapes) / sizeof(shapes[0]); index++)
        run_block_float_shape(context, &loaded.program, artifact, FT_TT_FORMAT_BFP4_B,
                              &shapes[index]);
    ft_tt_program_directory_release(&loaded);
}

/* ================================================================
 * main
 * ================================================================ */

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
    ft_tt_fttw artifact;
    ft_tt_fttw *artifact_pointer = NULL;
    const char *artifact_path = getenv("FT_TT_FTTW");
    const char *firmware_directory = NULL;
    char directory[512];
    uint32_t tensix_index = 2u;
    uint32_t bank;
    int rc;

    memset(&context, 0, sizeof(context));
    memset(&artifact, 0, sizeof(artifact));
    read_settings();

    ft_tt_test_step("Phase 4 op acceptance — PCC >= %.2f against a CPU fp32 reference "
                    "(prefetch %u, %u repeat(s))",
                    PCC_ACCEPT, settings.prefetch_tiles, settings.repeats);

    rc = ft_tt_hardware_session_open(&context.session);
    ft_tt_test_check_fmt("card opens", rc == FT_TT_OK, "%s", ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return ft_tt_test_finish();

    /* Indices 0 and 1 belong to the Phase 2 suites; this worker drives 2 and up. The
     * coordinate itself still comes from live telemetry, never a guess (spec §7.3). */
    (void)ft_tt_tensix_index_from_environment(&tensix_index);
    rc = ft_tt_hardware_session_tensix(&context.session, tensix_index, &context.core_x,
                                       &context.core_y);
    ft_tt_test_check_fmt("Tensix resolves from live telemetry", rc == FT_TT_OK,
                         "index %u -> translated (%u, %u): %s", tensix_index,
                         (unsigned)context.core_x, (unsigned)context.core_y,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto close;

    /* A harness that names its core owns it. This is a PER-CORE claim, not the
     * whole-card lock, so it runs alongside any other suite on a different Tensix
     * — and refuses, naming the holder's pid, if someone already has this one. */
    rc = ft_tt_hardware_session_claim_cores(&context.session, &tensix_index, 1u,
                                            "test_ops Phase 4 acceptance", NULL, NULL);
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

    if (artifact_path && ft_tt_fttw_open(artifact_path, &artifact) == FT_TT_OK) {
        artifact_pointer = &artifact;
        ft_tt_test_check_fmt("FTTW artifact opens", true, "%u tensors from %s",
                             artifact.header.tensor_count, artifact_path);
    } else {
        ft_tt_test_step("no FTTW artifact (set FT_TT_FTTW) — running on generated operands");
    }

    firmware_directory = ft_tt_firmware_directory();
    rc = ft_tt_firmware_set_load(firmware_directory, &context.firmware);
    ft_tt_test_check_fmt("firmware images load", rc == FT_TT_OK, "%s from %s",
                         ft_tt_error_string(rc),
                         firmware_directory ? firmware_directory : "(FT_TT_FIRMWARE_DIR unset)");
    if (rc != FT_TT_OK)
        goto close;

    rc = ft_tt_core_boot_firmware(&context.session.noc, context.core_x, context.core_y,
                                  &context.firmware, NULL);
    ft_tt_test_check_fmt("firmware boots on the core", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto park;

    check_dram_round_trip(&context);

    if (settings.run_rms_norm) {
        snprintf(directory, sizeof(directory), "%s/rms_norm", program_root());
        test_rms_norm(&context, directory, artifact_pointer);
    }
    if (settings.run_reduce_scaler) {
        snprintf(directory, sizeof(directory), "%s/rms_norm", program_root());
        test_reduce_scaler(&context, directory);
    }
    if (settings.run_reduce_dot) {
        snprintf(directory, sizeof(directory), "%s/rms_norm", program_root());
        test_reduce_dot(&context, directory);
    }
    if (settings.run_matmul) {
        snprintf(directory, sizeof(directory), "%s/matmul", program_root());
        test_matmul(&context, directory);
    }
    if (settings.run_matmul_bfp8) {
        snprintf(directory, sizeof(directory), "%s/matmul_bfp8", program_root());
        test_matmul_bfp8(&context, directory, artifact_pointer);
    }
    if (settings.run_matmul_bfp4) {
        snprintf(directory, sizeof(directory), "%s/matmul_bfp4", program_root());
        test_matmul_bfp4(&context, directory, artifact_pointer);
    }
    if (settings.run_mt_sweep) {
        snprintf(directory, sizeof(directory), "%s/matmul", program_root());
        test_mt_sweep(&context, directory);
    }

park:
    /* On every exit path, including failures: a core left running can still issue NoC
     * transactions (ft_tt_hardware_session.h). */
    ft_tt_core_park(&context.session.noc, context.core_x, context.core_y);
close:
    ft_tt_firmware_set_release(&context.firmware);
    ft_tt_fttw_close(&artifact);
    ft_tt_hardware_session_close(&context.session);
    return ft_tt_test_finish();
}
