/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_fidelity_swap.c
 * @brief P-swap probe: is a block-float-weight
 *   matmul at HiFi2 with the WEIGHT in SrcB bit-identical to today's HiFi4 with the weight
 *   in SrcA? And is HiFi3 bit-identical to HiFi4 when SrcA holds Bfp4 (the attention PV case)?
 *
 * The FPU computes SrcB x SrcA; in0 goes to SrcB, in1 to SrcA (ft_tensix_compute_matmul.h).
 * Its multiplier takes 5 mantissa bits of SrcA and 7 of SrcB per fidelity phase. Today the
 * weight is in1 (SrcA): Bfp8 (7 significant bits) needs 2 SrcA chunks and the bf16
 * activation (8) 2 SrcB chunks, so all 4 phases carry products. Swapped (Cᵀ = Wᵀ·Aᵀ) the
 * weight fits one SrcB chunk, so phases 3-4 should multiply zero and HiFi2 computes every
 * product bit. Whether the fp32 Dst sums come out bit-identical is what this measures.
 *
 * Wᵀ cannot be stored as Bfp8: its shared exponents run along the other axis, so the values
 * would change. The swapped arms store Wᵀ as bf16, which holds every Bfp8_b/Bfp4_b value
 * exactly (≤ 7 significant bits, same 8-bit exponent); the harness checks that per value.
 * A control arm runs today's orientation with the weight as bf16 and must equal the
 * block-float arm, which shows the storage format itself changes nothing.
 *
 * Every case is compared BIT FOR BIT (packed bf16 output) against its reference case, and
 * scored against an fp64 reference for information. Output poisoned before every launch.
 *
 * Environment:
 *   FT_TT_TENSIX_INDEX   which Tensix (default 84)
 *   FT_TT_PROGRAM_DIR    parent of the program directories named in PROGRAMS below
 *
 * @status Probe; read-only arithmetic question, no shipped program changes.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../convert/ft_bfp.h"
#include "../convert/ft_float_convert.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_matmul.h"
#include "../ft_tt_tilize.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define TILE_DIM       FT_TT_OP_TILE_DIM
#define TILE_ELEMENTS  (TILE_DIM * TILE_DIM)
#define POISON_BYTE    0xA5u
#define LAUNCH_TIMEOUT 60000u
#define M_TILES        2u  /**< prefill rows per probe: Mt 2 */
#define N_TILES        8u  /**< output columns: 256, so 16k outputs a case */
#define PREFETCH_TILES 32u /**< divides both Kt below */
#define SEED_COUNT     3u

static const uint32_t KT_POINTS[] = {160u, 544u}; /**< Qwen3.8 hidden / FFN width in tiles */
#define KT_POINT_COUNT (sizeof(KT_POINTS) / sizeof(KT_POINTS[0]))

/* DRAM: one bank per operand, far above anything a harness at low addresses touches. */
#define DRAM_BANK_IN0 5u
#define DRAM_BANK_IN1 6u
#define DRAM_BANK_OUT 7u
#define DRAM_BASE     0x20000000u

typedef enum { WEIGHT_BFP8, WEIGHT_BFP4 } weight_kind;

/** One program directory and the cases run on it. */
typedef struct swap_program {
    const char *directory; /**< under FT_TT_PROGRAM_DIR */
    uint32_t in1_tile_bytes;
} swap_program;

typedef enum { PROGRAM_BFP8, PROGRAM_BFP4, PROGRAM_BF16_HIFI4, PROGRAM_BF16_HIFI2,
               PROGRAM_BF16_HIFI3, PROGRAM_BFP4_HIFI3, PROGRAM_COUNT } program_id;

static const swap_program PROGRAMS[PROGRAM_COUNT] = {
    {"matmul_bfp8", 1088u},       /* shipped: HiFi4, fp32 dest, weight Bfp8 in SrcA */
    {"matmul_bfp4", 576u},        /* shipped: HiFi4, fp32 dest, weight Bfp4 in SrcA */
    {"fswap_bf16_hifi4", 2048u},  /* matmul_float16b_fp32acc */
    {"fswap_bf16_hifi2", 2048u},  /* matmul_float16b_fp32acc_hifi2 */
    {"fswap_bf16_hifi3", 2048u},  /* matmul_float16b_fp32acc_hifi3 */
    {"fswap_bfp4_hifi3", 576u},   /* matmul_bfp4_b_hifi3 */
};

/** A case: which program, which weight, which orientation, and what it must equal. */
typedef struct swap_case {
    const char *label;
    program_id program;
    weight_kind weight;
    int swapped;   /**< 1: in0 = Wᵀ (bf16), in1 = Aᵀ; output Cᵀ */
    int bf16_in1;  /**< unswapped with the weight stored as bf16 */
    int reference; /**< index of the case this must equal; -1 = it is a reference */
} swap_case;

static const swap_case CASES[] = {
    {"R8   bfp8 HiFi4 (today)",          PROGRAM_BFP8,       WEIGHT_BFP8, 0, 0, -1},
    {"C8   bf16-stored HiFi4",           PROGRAM_BF16_HIFI4, WEIGHT_BFP8, 0, 1, 0},
    {"S8.4 swapped HiFi4",               PROGRAM_BF16_HIFI4, WEIGHT_BFP8, 1, 0, 0},
    {"S8.2 swapped HiFi2 (L5)",          PROGRAM_BF16_HIFI2, WEIGHT_BFP8, 1, 0, 0},
    {"S8.3 swapped HiFi3",               PROGRAM_BF16_HIFI3, WEIGHT_BFP8, 1, 0, 0},
    {"R4   bfp4 HiFi4 (today)",          PROGRAM_BFP4,       WEIGHT_BFP4, 0, 0, -1},
    {"P4.3 bfp4 HiFi3 in SrcA (L6)",     PROGRAM_BFP4_HIFI3, WEIGHT_BFP4, 0, 0, 5},
    {"S4.2 swapped bfp4 HiFi2",          PROGRAM_BF16_HIFI2, WEIGHT_BFP4, 1, 0, 5},
    /* RED: weight in SrcA at HiFi2 drops products, so it must DIFFER from R8. */
    {"X8.2 red: unswapped HiFi2",        PROGRAM_BF16_HIFI2, WEIGHT_BFP8, 0, 1, 0},
};
#define CASE_COUNT (sizeof(CASES) / sizeof(CASES[0]))

typedef struct op_context {
    ft_tt_hardware_session session;
    uint16_t core_x;
    uint16_t core_y;
    ft_tt_noc_coord dram[8];
} op_context;

/** One operand draw: A (M x K, bf16 values) and W (K x N, block-float values). */
typedef struct operand_set {
    uint32_t kt;
    float *a;        /**< M x K row-major */
    float *w;        /**< K x N row-major, the values the FPU sees */
    unsigned char *w_packed; /**< K-major block-float tiles */
    double *exact;   /**< M x N fp64 reference */
} operand_set;

/** Per case and Kt point: bf16 outputs, row-major M x N (swapped cases transposed back). */
static uint16_t *g_outputs[CASE_COUNT][KT_POINT_COUNT][SEED_COUNT];

/* ================================================================
 * Operands
 * ================================================================ */

static uint64_t random_state = 1u;

static double random_uniform(void)
{
    random_state ^= random_state >> 12;
    random_state ^= random_state << 25;
    random_state ^= random_state >> 27;
    return (double)(random_state * 0x2545F4914F6CDD1Dull >> 11) / (double)(1ull << 53);
}

/** Roughly normal with a 1% tail at 16x: activations after a norm carry outliers. */
static float random_activation(void)
{
    double sum = 0.0;
    int index;

    for (index = 0; index < 4; index++)
        sum += random_uniform() - 0.5;
    if (random_uniform() < 0.01)
        sum *= 16.0;
    return ft_float_from_bf16(ft_bf16_from_float((float)sum));
}

static float random_weight(void)
{
    double sum = 0.0;
    int index;

    for (index = 0; index < 4; index++)
        sum += random_uniform() - 0.5;
    return (float)(sum * 0.04);
}

static void operand_set_release(operand_set *set)
{
    free(set->a);
    free(set->w);
    free(set->w_packed);
    free(set->exact);
    memset(set, 0, sizeof(*set));
}

/** Draw A and W, quantize W through the artifact's packer, compute the fp64 reference. */
static int operand_set_draw(operand_set *set, uint32_t kt, weight_kind kind, uint64_t seed)
{
    const ft_bfp_format format = kind == WEIGHT_BFP8 ? FT_BFP_BFP8_B : FT_BFP_BFP4_B;
    const uint32_t m = M_TILES * TILE_DIM, k = kt * TILE_DIM, n = N_TILES * TILE_DIM;
    const size_t tiles = (size_t)kt * N_TILES;
    float *swizzled = malloc(tiles * TILE_ELEMENTS * sizeof(float));
    float *seen = malloc(tiles * TILE_ELEMENTS * sizeof(float));
    size_t index;
    uint32_t row, column, inner;
    int rc = -1;

    memset(set, 0, sizeof(*set));
    set->kt = kt;
    set->a = malloc((size_t)m * k * sizeof(float));
    set->w = malloc((size_t)k * n * sizeof(float));
    set->w_packed = malloc(tiles * ft_bfp_tile_bytes(format));
    set->exact = malloc((size_t)m * n * sizeof(double));
    if (!swizzled || !seen || !set->a || !set->w || !set->w_packed || !set->exact)
        goto done;

    random_state = seed ? seed : 1u;
    for (index = 0; index < (size_t)m * k; index++)
        set->a[index] = random_activation();
    for (index = 0; index < (size_t)k * n; index++)
        set->w[index] = random_weight();

    for (inner = 0; inner < kt; inner++)
        ft_tt_tile_swizzle_strip(set->w + (size_t)inner * TILE_DIM * n, sizeof(float), TILE_DIM, n,
                                 n * sizeof(float), swizzled + (size_t)inner * N_TILES * TILE_ELEMENTS);
    if (ft_bfp_pack_tiles(format, swizzled, tiles, set->w_packed) != FT_BFP_OK ||
        ft_bfp_unpack_tiles(format, set->w_packed, tiles, seen) != FT_BFP_OK)
        goto done;
    for (index = 0; index < tiles; index++) {
        uint32_t tile_row = (uint32_t)(index / N_TILES), tile_column = (uint32_t)(index % N_TILES);
        uint32_t inside;

        for (inside = 0; inside < TILE_ELEMENTS; inside++)
            set->w[(size_t)(tile_row * TILE_DIM + inside / TILE_DIM) * n + tile_column * TILE_DIM +
                   inside % TILE_DIM] = seen[index * TILE_ELEMENTS + inside];
    }

    for (row = 0; row < m; row++)
        for (column = 0; column < n; column++) {
            double sum = 0.0;

            for (inner = 0; inner < k; inner++)
                sum += (double)set->a[(size_t)row * k + inner] * (double)set->w[(size_t)inner * n + column];
            set->exact[(size_t)row * n + column] = sum;
        }
    rc = 0;
done:
    free(swizzled);
    free(seen);
    if (rc != 0)
        operand_set_release(set);
    return rc;
}

/** Values of W that bf16 does not hold exactly (expected 0: the swapped arms rely on it). */
static size_t count_bf16_inexact(const float *values, size_t count)
{
    size_t bad = 0, index;

    for (index = 0; index < count; index++)
        if (ft_float_from_bf16(ft_bf16_from_float(values[index])) != values[index])
            bad++;
    return bad;
}

/** bf16 tiles of a rows x cols row-major matrix (rows, cols multiples of 32), row-major tile order. */
static uint16_t *bf16_tiles(const float *matrix, uint32_t rows, uint32_t cols, int transpose)
{
    uint16_t *dense = malloc((size_t)rows * cols * sizeof(uint16_t));
    uint16_t *tiles = malloc((size_t)rows * cols * sizeof(uint16_t));
    uint32_t row, column, strip;

    if (!dense || !tiles) {
        free(dense);
        free(tiles);
        return NULL;
    }
    /* transpose: `matrix` is cols x rows and the tiles are of its transpose. */
    for (row = 0; row < rows; row++)
        for (column = 0; column < cols; column++)
            dense[(size_t)row * cols + column] = ft_bf16_from_float(
                transpose ? matrix[(size_t)column * rows + row] : matrix[(size_t)row * cols + column]);
    for (strip = 0; strip < rows / TILE_DIM; strip++)
        ft_tt_tilize_strip(dense + (size_t)strip * TILE_DIM * cols, sizeof(uint16_t), TILE_DIM, cols,
                           cols * sizeof(uint16_t), tiles + (size_t)strip * TILE_DIM * cols);
    free(dense);
    return tiles;
}

/* ================================================================
 * Device plumbing (test_fidelity_sweep.c's 64 KiB chunked pattern)
 * ================================================================ */

static int device_copy(const op_context *context, ft_tt_noc_coord where, uint32_t address, void *data,
                       size_t bytes, int write)
{
    unsigned char *cursor = data;
    size_t offset = 0;

    while (offset < bytes) {
        size_t chunk = bytes - offset > 0x10000u ? 0x10000u : bytes - offset;
        int rc = write ? context->session.noc.write(context->session.noc.user, (uint16_t)where.x,
                                                    (uint16_t)where.y, (uint64_t)address + offset,
                                                    cursor + offset, chunk)
                       : context->session.noc.read(context->session.noc.user, (uint16_t)where.x,
                                                   (uint16_t)where.y, (uint64_t)address + offset,
                                                   cursor + offset, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

static ft_tt_op_operand operand_at(const op_context *context, uint32_t bank, uint32_t tile_bytes)
{
    ft_tt_op_operand operand;

    memset(&operand, 0, sizeof(operand));
    operand.x = (uint16_t)context->dram[bank].x;
    operand.y = (uint16_t)context->dram[bank].y;
    operand.address = DRAM_BASE;
    operand.tile_bytes = tile_bytes;
    return operand;
}

/* ================================================================
 * One case on one operand set
 * ================================================================ */

/** Upload, run, read back; returns M x N row-major bf16 (transposed back when swapped). */
static uint16_t *run_case(op_context *context, const ft_tt_kernel_program *program, const swap_case *c,
                          const operand_set *set, double *milliseconds)
{
    const uint32_t m = M_TILES * TILE_DIM, k = set->kt * TILE_DIM, n = N_TILES * TILE_DIM;
    const uint32_t out_rows = c->swapped ? n : m, out_cols = c->swapped ? m : n;
    const size_t out_bytes = (size_t)out_rows * out_cols * sizeof(uint16_t);
    uint16_t *in0 = NULL, *in1_bf16 = NULL, *out_tiles = malloc(out_bytes), *result = NULL;
    const void *in1;
    size_t in0_bytes = (size_t)(c->swapped ? n : m) * k * sizeof(uint16_t), in1_bytes;
    unsigned char *poison = malloc(out_bytes);
    ft_tt_op_matmul_request request;
    struct timespec start, end;
    uint32_t element;
    int rc;

    if (!out_tiles || !poison)
        goto done;
    if (c->swapped) {
        in0 = bf16_tiles(set->w, n, k, 1);      /* Wᵀ: N x K */
        in1_bf16 = bf16_tiles(set->a, k, m, 1); /* Aᵀ: K x M */
    } else {
        in0 = bf16_tiles(set->a, m, k, 0);
        if (c->bf16_in1)
            in1_bf16 = bf16_tiles(set->w, k, n, 0);
    }
    if (!in0 || ((c->swapped || c->bf16_in1) && !in1_bf16))
        goto done;
    if (in1_bf16) {
        in1 = in1_bf16;
        in1_bytes = (size_t)k * (c->swapped ? m : n) * sizeof(uint16_t);
    } else {
        in1 = set->w_packed;
        in1_bytes = (size_t)set->kt * N_TILES * PROGRAMS[c->program].in1_tile_bytes;
    }

    memset(poison, (int)POISON_BYTE, out_bytes);
    rc = device_copy(context, context->dram[DRAM_BANK_IN0], DRAM_BASE, in0, in0_bytes, 1);
    rc |= device_copy(context, context->dram[DRAM_BANK_IN1], DRAM_BASE, (void *)(uintptr_t)in1, in1_bytes, 1);
    rc |= device_copy(context, context->dram[DRAM_BANK_OUT], DRAM_BASE, poison, out_bytes, 1);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("swap probe upload", false, "%s: %s", c->label, ft_tt_error_string(rc));
        goto done;
    }

    memset(&request, 0, sizeof(request));
    request.tile_rows = out_rows / TILE_DIM;
    request.tile_inner = set->kt;
    request.tile_cols = out_cols / TILE_DIM;
    request.prefetch_tiles = PREFETCH_TILES;
    request.activation = operand_at(context, DRAM_BANK_IN0, FT_TT_OP_TILE_BYTES_BF16);
    request.weight = operand_at(context, DRAM_BANK_IN1, PROGRAMS[c->program].in1_tile_bytes);
    request.output = operand_at(context, DRAM_BANK_OUT, FT_TT_OP_TILE_BYTES_BF16);

    clock_gettime(CLOCK_MONOTONIC, &start);
    rc = ft_tt_op_matmul_run(&context->session.noc, context->core_x, context->core_y, program, &request,
                             LAUNCH_TIMEOUT);
    clock_gettime(CLOCK_MONOTONIC, &end);
    *milliseconds = (double)(end.tv_sec - start.tv_sec) * 1e3 + (double)(end.tv_nsec - start.tv_nsec) / 1e6;
    if (rc == FT_TT_OK)
        rc = device_copy(context, context->dram[DRAM_BANK_OUT], DRAM_BASE, out_tiles, out_bytes, 0);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("swap probe run", false, "%s kt=%u: %s", c->label, set->kt,
                             ft_tt_error_string(rc));
        goto done;
    }

    result = malloc((size_t)m * n * sizeof(uint16_t));
    if (!result)
        goto done;
    for (element = 0; element < out_rows * out_cols; element++) {
        uint32_t row = element / out_cols, column = element % out_cols;
        uint32_t tile = (row / TILE_DIM) * (out_cols / TILE_DIM) + column / TILE_DIM;
        uint16_t value = out_tiles[(size_t)tile * TILE_ELEMENTS +
                                   ft_tt_tile_face_index(row % TILE_DIM, column % TILE_DIM)];

        if (c->swapped)
            result[(size_t)column * n + row] = value;
        else
            result[(size_t)row * n + column] = value;
    }
done:
    free(in0);
    free(in1_bf16);
    free(out_tiles);
    free(poison);
    return result;
}

/* ================================================================
 * Scoring
 * ================================================================ */

/** Bit differences of two bf16 outputs; worst distance in bf16 ulps (same-sign values). */
static size_t count_bit_differences(const uint16_t *a, const uint16_t *b, size_t count, uint32_t *worst_ulps)
{
    size_t differing = 0, index;

    *worst_ulps = 0;
    for (index = 0; index < count; index++) {
        if (a[index] != b[index]) {
            uint32_t distance = (a[index] & 0x8000u) == (b[index] & 0x8000u)
                                    ? (uint32_t)abs((int)(a[index] & 0x7FFFu) - (int)(b[index] & 0x7FFFu))
                                    : 0xFFFFu;

            differing++;
            if (distance > *worst_ulps)
                *worst_ulps = distance;
        }
    }
    return differing;
}

/** Mean absolute error against the fp64 reference, and the count of non-finite outputs. */
static double mean_abs_error(const uint16_t *out, const double *exact, size_t count, size_t *non_finite)
{
    double total = 0.0;
    size_t index;

    *non_finite = 0;
    for (index = 0; index < count; index++) {
        double value = (double)ft_float_from_bf16(out[index]);

        if (!isfinite(value))
            (*non_finite)++;
        else
            total += fabs(value - exact[index]);
    }
    return total / (double)count;
}

/* ================================================================
 * Driver
 * ================================================================ */

static const char *program_root(void)
{
    const char *value = getenv("FT_TT_PROGRAM_DIR");

    return value ? value : "../programs";
}

/** Boot one program directory and run every case that names it over every operand set. */
static void run_program(op_context *context, program_id id, operand_set sets[2][KT_POINT_COUNT][SEED_COUNT])
{
    char directory[512];
    ft_tt_firmware_set firmware;
    ft_tt_program_directory loaded;
    size_t case_index, point, seed;
    int rc;

    memset(&firmware, 0, sizeof(firmware));
    snprintf(directory, sizeof(directory), "%s/%s", program_root(), PROGRAMS[id].directory);
    rc = ft_tt_firmware_set_load(directory, &firmware);
    if (rc == FT_TT_OK)
        rc = ft_tt_core_boot_firmware(&context->session.noc, context->core_x, context->core_y, &firmware, NULL);
    ft_tt_test_check_fmt("swap probe firmware boots", rc == FT_TT_OK, "%s: %s", directory,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK) {
        ft_tt_firmware_set_release(&firmware);
        return;
    }
    rc = ft_tt_program_directory_load(directory, 0x1Fu, &loaded);
    ft_tt_test_check_fmt("swap probe program loads", rc == FT_TT_OK, "%s: %s", directory,
                         ft_tt_error_string(rc));
    if (rc == FT_TT_OK) {
        for (case_index = 0; case_index < CASE_COUNT; case_index++) {
            const swap_case *c = &CASES[case_index];

            if (c->program != id)
                continue;
            for (point = 0; point < KT_POINT_COUNT; point++)
                for (seed = 0; seed < SEED_COUNT; seed++) {
                    double milliseconds = 0.0;

                    g_outputs[case_index][point][seed] =
                        run_case(context, &loaded.program, c, &sets[c->weight][point][seed], &milliseconds);
                    ft_tt_test_step("%-30s kt=%3u seed %zu: %s, %.2f ms", c->label, KT_POINTS[point], seed,
                                    g_outputs[case_index][point][seed] ? "ok" : "FAILED", milliseconds);
                }
        }
        ft_tt_program_directory_release(&loaded);
    }
    ft_tt_core_park(&context->session.noc, context->core_x, context->core_y);
    ft_tt_firmware_set_release(&firmware);
}

/** The verdict table: every case against its reference, bit for bit, per Kt. */
static void print_verdicts(operand_set sets[2][KT_POINT_COUNT][SEED_COUNT])
{
    const size_t outputs = (size_t)M_TILES * TILE_DIM * N_TILES * TILE_DIM;
    size_t case_index, point, seed;

    printf("\n=== P-swap verdicts (bf16 outputs, %zu a case per Kt per seed, %u seeds) ===\n", outputs,
           SEED_COUNT);
    for (case_index = 0; case_index < CASE_COUNT; case_index++) {
        const swap_case *c = &CASES[case_index];

        for (point = 0; point < KT_POINT_COUNT; point++) {
            size_t differing = 0, non_finite_total = 0;
            uint32_t worst = 0;
            double error = 0.0;
            int complete = 1;

            for (seed = 0; seed < SEED_COUNT; seed++) {
                const uint16_t *mine = g_outputs[case_index][point][seed];
                const uint16_t *theirs = c->reference >= 0 ? g_outputs[c->reference][point][seed] : NULL;
                size_t non_finite;
                uint32_t ulps;

                if (!mine || (c->reference >= 0 && !theirs)) {
                    complete = 0;
                    continue;
                }
                if (theirs) {
                    differing += count_bit_differences(mine, theirs, outputs, &ulps);
                    if (ulps > worst)
                        worst = ulps;
                }
                error += mean_abs_error(mine, sets[c->weight][point][seed].exact, outputs, &non_finite) /
                         SEED_COUNT;
                non_finite_total += non_finite;
            }
            if (!complete) {
                ft_tt_test_check_fmt("swap probe case complete", false, "%s kt=%u", c->label, KT_POINTS[point]);
                continue;
            }
            ft_tt_test_check_fmt("swap probe output finite", non_finite_total == 0u, "%s kt=%u: %zu non-finite",
                                 c->label, KT_POINTS[point], non_finite_total);
            if (c->reference < 0)
                printf("VERDICT %-30s kt=%3u  reference          mean|err| %.4g\n", c->label, KT_POINTS[point],
                       error);
            else
                printf("VERDICT %-30s kt=%3u  vs %-6.6s %s (%zu/%zu differ, worst %u ulp)  mean|err| %.4g\n",
                       c->label, KT_POINTS[point], CASES[c->reference].label,
                       differing == 0u ? "IDENTICAL" : "DIFFERS  ", differing, outputs * SEED_COUNT, worst, error);
        }
    }
}

static int open_core(op_context *context)
{
    uint32_t tensix_index = 84u, bank;
    int rc = ft_tt_hardware_session_open(&context->session);

    ft_tt_test_check_fmt("card opens", rc == FT_TT_OK, "%s", ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return rc;
    (void)ft_tt_tensix_index_from_environment(&tensix_index);
    rc = ft_tt_hardware_session_tensix(&context->session, tensix_index, &context->core_x, &context->core_y);
    if (rc == FT_TT_OK)
        rc = ft_tt_hardware_session_claim_cores(&context->session, &tensix_index, 1u, "fidelity swap probe",
                                                NULL, NULL);
    ft_tt_test_check_fmt("the Tensix resolves and is claimed", rc == FT_TT_OK, "index %u: %s", tensix_index,
                         ft_tt_error_string(rc));
    for (bank = 0; rc == FT_TT_OK && bank < 8u; bank++)
        rc = ft_tt_coord_dram_bank(&context->session.grid, bank, FT_TT_COORD_TRANSLATED, &context->dram[bank]);
    return rc;
}

int main(void)
{
    static operand_set sets[2][KT_POINT_COUNT][SEED_COUNT];
    op_context context;
    size_t point, seed, inexact = 0;
    int kind, id, rc;

    memset(&context, 0, sizeof(context));
    ft_tt_test_step("P-swap probe: %zu cases x %zu Kt x %u seeds, Mt %u Nt %u", CASE_COUNT, KT_POINT_COUNT,
                    SEED_COUNT, M_TILES, N_TILES);
    for (kind = 0; kind < 2; kind++)
        for (point = 0; point < KT_POINT_COUNT; point++)
            for (seed = 0; seed < SEED_COUNT; seed++) {
                operand_set *set = &sets[kind][point][seed];

                rc = operand_set_draw(set, KT_POINTS[point], (weight_kind)kind,
                                      0x5EED0000u + (uint64_t)(point * 16u + seed));
                ft_tt_test_check_fmt("operands draw", rc == 0, "kind %d kt %u seed %zu", kind, KT_POINTS[point],
                                     seed);
                if (rc != 0)
                    return ft_tt_test_finish();
                inexact += count_bf16_inexact(set->w, (size_t)set->kt * TILE_DIM * N_TILES * TILE_DIM);
            }
    ft_tt_test_check_fmt("every block-float weight value is exact in bf16", inexact == 0u,
                         "%zu values are not", inexact);

    rc = open_core(&context);
    if (rc == FT_TT_OK) {
        for (id = 0; id < PROGRAM_COUNT; id++)
            run_program(&context, (program_id)id, sets);
        print_verdicts(sets);
    }
    ft_tt_hardware_session_close(&context.session);
    for (kind = 0; kind < 2; kind++)
        for (point = 0; point < KT_POINT_COUNT; point++)
            for (seed = 0; seed < SEED_COUNT; seed++)
                operand_set_release(&sets[kind][point][seed]);
    return ft_tt_test_finish();
}
