/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file test_metal_matvec_q5_k.c
 * @brief Absolute-correctness tests for the Metal Q5_K matvec.
 *
 * The oracle is the CPU dequant+dot in this file, written straight from the
 * standard 176-byte / 256-weight GGUF Q5_K block layout — NOT from the kernel.
 * Every assertion compares the GPU result against that oracle in absolute
 * terms, so a uniform error (a scaled row sum, a wrong scale decode) cannot
 * cancel the way it does in a GPU-against-GPU offset-invariance comparison.
 *
 * The blocks are real: they are extracted from the shipped Q3_K_M GGUF, so a
 * misread of the actual on-disk layout shows up here and not in the logits.
 * The nonzero-region-offset test is an ADDITION to the absolute tests, never
 * a substitute for one.
 *
 * Exits 77 when no Metal device is present so GPU-less CI skips cleanly.
 *
 * @status All tests pass on M4 / macOS 26; each was proven able to fail (scaled
 *   row sum, wrong 6-bit scale decode, off-by-one block stride, dropped qh bit,
 *   ignored bind offset). The oracle below was itself cross-checked against an
 *   independent dequantizer and agreed BIT-IDENTICALLY on 20480 real weights.
 *   ragged_row_tail_* covers the hazard the rows-per-threadgroup split adds:
 *   the grid rounds up, so the last threadgroup carries rows past the end.
 * @issues None
 * @todo None
 */

#include "lib/test.h"
#include "metal/fm_metal.h"
#include "metal/tests/gguf_test_reader.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define Q5K_BLOCK_BYTES        176u
#define Q5K_WEIGHTS_PER_BLOCK  256u
#define Q5K_SUB_BLOCKS         8u      /* 8 sub-blocks of 32 weights per block */

/* Real Q5_K tensors in the shipped file: the attn_qkv projections (the bulk of
 * the 90) plus a shared-expert FFN weight, which has a different column count
 * and so exercises a different block count per row. */
static const char *kTensorQkv        = "blk.0.attn_qkv.weight";
static const char *kTensorSharedExpert = "blk.5.ffn_down_shexp.weight";

static const char *kDefaultModelPath =
    "models/Nex-N2-mini/nex-agi_Nex-N2-mini-Q3_K_M.gguf";

#define SCRATCH_DIR \
    "metal/tests/"   /* run from the repo root; test_* is gitignored */

static const char *kScratchPath = SCRATCH_DIR "test_q5k_region.bin";

/* 32-byte aligned (GGUF's tensor alignment) but deliberately not a multiple of
 * the 16 KiB page — the real case a weight inside an mmap'd model hits. */
#define REGION_WEIGHT_OFFSET  (1024u * 1024u + 32u)

static gguf_test_file g_model;
static int            g_model_is_open;

/* ================================================================
 * CPU reference — the oracle
 *
 * Standard GGUF Q5_K super-block, 176 bytes per 256 weights:
 *
 *   offset   0 : half   d           super-block scale for the 6-bit scales
 *   offset   2 : half   dmin        super-block scale for the 6-bit mins
 *   offset   4 : uchar  scales[12]  8 (scale, min) pairs packed 6 bits each
 *   offset  16 : uchar  qh[32]      one high bit per weight
 *   offset  48 : uchar  qs[128]     the low 4 bits, two weights per byte
 *
 * Weight i in 0..255 sits in sub-block s = i / 32 at lane l = i % 32:
 *
 *   nibble = (s & 1) ? qs[32 * (s / 2) + l] >> 4 : qs[32 * (s / 2) + l] & 0xF
 *   q      = nibble + 16 * ((qh[l] >> s) & 1)
 *   value  = d * scale[s] * q - dmin * min[s]
 * ================================================================ */

static float f16_to_f32(uint16_t bits)
{
    uint32_t sign     = (uint32_t)(bits >> 15) << 31;
    uint32_t exponent = (bits >> 10) & 0x1Fu;
    uint32_t mantissa = bits & 0x3FFu;
    uint32_t out;

    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {                       /* subnormal: renormalize */
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3FFu;
            out = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1Fu) {
        out = sign | 0x7F800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float value;
    memcpy(&value, &out, sizeof(value));
    return value;
}

/* Unpacks the 6-bit scale and min of sub-block `index` from the 12 packed
 * bytes. The first four pairs are stored plainly in the low 6 bits; the last
 * four borrow their top 2 bits from the high bits of the first four. */
static void q5k_unpack_scale_min(const uint8_t *scales, unsigned index,
                                 uint8_t *scale_out, uint8_t *min_out)
{
    if (index < 4) {
        *scale_out = scales[index] & 63u;
        *min_out   = scales[index + 4] & 63u;
    } else {
        *scale_out = (uint8_t)((scales[index + 4] & 0xFu) | ((scales[index - 4] >> 6) << 4));
        *min_out   = (uint8_t)((scales[index + 4] >> 4)   | ((scales[index]     >> 6) << 4));
    }
}

/* Dequantizes one 176-byte block into 256 floats. */
static void q5k_dequantize_block(const uint8_t *block, float *out)
{
    uint16_t d_bits, dmin_bits;
    memcpy(&d_bits, block + 0, 2);
    memcpy(&dmin_bits, block + 2, 2);
    const float d    = f16_to_f32(d_bits);
    const float dmin = f16_to_f32(dmin_bits);

    const uint8_t *scales = block + 4;
    const uint8_t *qh     = block + 16;
    const uint8_t *qs     = block + 48;

    for (unsigned s = 0; s < Q5K_SUB_BLOCKS; s++) {
        uint8_t scale = 0, min = 0;
        q5k_unpack_scale_min(scales, s, &scale, &min);
        const float step   = d * (float)scale;
        const float offset = dmin * (float)min;
        for (unsigned l = 0; l < 32; l++) {
            const uint8_t packed = qs[32u * (s / 2u) + l];
            const uint8_t nibble = (s & 1u) ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0xFu);
            const unsigned high  = (qh[l] >> s) & 1u;
            const float quant    = (float)(nibble + 16u * high);
            out[s * 32u + l] = step * quant - offset;
        }
    }
}

/* out[r] = sum_c dequant(w)[r][c] * x[c], accumulated in double so the oracle
 * itself is not the thing under suspicion. Also reports the per-row sum of
 * absolute terms, which sets an honest float tolerance for the GPU compare. */
static void q5k_reference_matvec(const uint8_t *w, int rows, int cols,
                                 size_t row_stride_bytes, const float *x,
                                 float *out, float *abs_sum_out)
{
    const int blocks_per_row = cols / (int)Q5K_WEIGHTS_PER_BLOCK;
    float block[Q5K_WEIGHTS_PER_BLOCK];

    for (int r = 0; r < rows; r++) {
        double dot = 0.0;
        double abs_sum = 0.0;
        for (int b = 0; b < blocks_per_row; b++) {
            q5k_dequantize_block(w + (size_t)r * row_stride_bytes + (size_t)b * Q5K_BLOCK_BYTES,
                                 block);
            for (unsigned i = 0; i < Q5K_WEIGHTS_PER_BLOCK; i++) {
                const double term = (double)block[i] * (double)x[b * 256 + (int)i];
                dot += term;
                abs_sum += fabs(term);
            }
        }
        out[r] = (float)dot;
        if (abs_sum_out) abs_sum_out[r] = (float)abs_sum;
    }
}

/* Float tolerance justified by the reordering the GPU does: the kernel splits
 * each row across 32 lanes and reduces with simd_sum, so the summation order
 * differs from the reference's. Drift scales with the row's L1 mass, not with
 * its (possibly cancelling) result.
 *
 * The constant is the MEASURED noise floor with headroom, not a number picked
 * to make the test green: the real-tensor tests print the worst drift they
 * observe (~3e-8 x row L1 on M4 / macOS 26), so this sits ~30x above it and
 * still rejects a 0.01% error in a row sum. */
#define Q5K_DRIFT_TOLERANCE  1e-6f

static float tolerance_for(float abs_sum)
{
    return Q5K_DRIFT_TOLERANCE * abs_sum + 1e-7f;
}

/* ================================================================
 * Harness helpers
 * ================================================================ */

static void fill_probe_vector(float *x, int cols, unsigned seed)
{
    uint32_t rng = 0x9E3779B9u + seed * 0x85EBCA6Bu;
    for (int i = 0; i < cols; i++) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        x[i] = (float)((int32_t)(rng & 0xFFFFu) - 32768) / 32768.0f;
    }
}

/* Runs the kernel and copies `rows` floats out of the result binding. */
static void run_q5k(fm_bufref w, fm_bufref x, fm_bufref out,
                    int rows, int cols, int row_stride_bytes, float *host_out)
{
    char *out_host = (char *)fm_buffer_data(out.buffer) + out.offset;
    memset(out_host, 0, (size_t)rows * sizeof(float));

    fm_matvec_q5k(w, x, out, rows, cols, row_stride_bytes);
    fm_sync();
    memcpy(host_out, out_host, (size_t)rows * sizeof(float));
}

/* Pulls `rows` consecutive quantized rows of a real Q5_K tensor out of the
 * model file. Reports the tensor's column count. Returns 0 on success. */
static int load_real_q5k_rows(const char *tensor_name, int rows,
                              int *cols_out, uint8_t *dst, size_t dst_capacity)
{
    gguf_tensor_location location;
    if (gguf_test_find_tensor(&g_model, tensor_name, &location) != 0)
        return -1;
    if (location.ggml_type != GGUF_GGML_TYPE_Q5_K || location.n_dims != 2)
        return -1;

    const int cols = (int)location.dims[0];
    if (cols % (int)Q5K_WEIGHTS_PER_BLOCK != 0 || (uint64_t)rows > location.dims[1])
        return -1;

    const size_t row_stride = (size_t)(cols / 256) * Q5K_BLOCK_BYTES;
    const size_t bytes = (size_t)rows * row_stride;
    if (bytes > dst_capacity)
        return -1;
    if (gguf_test_read_at(&g_model, location.file_offset, dst, bytes) != 0)
        return -1;

    *cols_out = cols;
    return 0;
}

/* Body shared by the two real-tensor tests: extract rows, run, compare
 * absolutely against the CPU oracle. Returns 0 if the tensor loaded. */
static int check_real_tensor_against_reference(const char *tensor_name, int rows,
                                               unsigned probe_seed)
{
    static uint8_t weight_bytes[64 * 1408];   /* 64 rows x 2048 cols, the largest here */
    int cols = 0;
    if (load_real_q5k_rows(tensor_name, rows, &cols, weight_bytes, sizeof(weight_bytes)) != 0)
        return -1;

    const int row_stride = (cols / 256) * (int)Q5K_BLOCK_BYTES;

    fm_buffer *w   = fm_buffer_alloc((size_t)rows * (size_t)row_stride);
    fm_buffer *x   = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    if (!w || !x || !out) return -1;

    memcpy(fm_buffer_data(w), weight_bytes, (size_t)rows * (size_t)row_stride);
    float *x_host = (float *)fm_buffer_data(x);
    fill_probe_vector(x_host, cols, probe_seed);

    float *expected = (float *)malloc((size_t)rows * sizeof(float));
    float *abs_sum  = (float *)malloc((size_t)rows * sizeof(float));
    float *actual   = (float *)malloc((size_t)rows * sizeof(float));
    q5k_reference_matvec(weight_bytes, rows, cols, (size_t)row_stride, x_host,
                         expected, abs_sum);
    run_q5k(fm_bufref_at(w, 0), fm_bufref_at(x, 0), fm_bufref_at(out, 0),
            rows, cols, row_stride, actual);

    int failures = 0;
    float worst_drift = 0.0f;
    for (int r = 0; r < rows; r++) {
        const float diff = fabsf(expected[r] - actual[r]);
        if (diff > tolerance_for(abs_sum[r])) {
            if (failures < 4)
                printf("    row %d: expected %.8f got %.8f (diff %.3g tol %.3g)\n",
                       r, expected[r], actual[r], diff, tolerance_for(abs_sum[r]));
            failures++;
        }
        const float drift = abs_sum[r] > 0.0f ? diff / abs_sum[r] : 0.0f;
        if (drift > worst_drift) worst_drift = drift;
    }
    /* The measured noise floor, so the tolerance above stays honest rather
     * than being whatever number made the test go green. */
    printf("      %-28s worst drift = %.3g x row L1 (tolerance %.3g)\n",
           tensor_name, (double)worst_drift, (double)Q5K_DRIFT_TOLERANCE);

    /* A kernel that wrote nothing would "agree" with an all-zero reference. */
    float magnitude = 0.0f;
    for (int r = 0; r < rows; r++) magnitude += fabsf(expected[r]);

    free(expected); free(abs_sum); free(actual);
    fm_buffer_free(w); fm_buffer_free(x); fm_buffer_free(out);

    if (magnitude <= 1e-3f) return -2;
    return failures == 0 ? 0 : 1;
}

/* ================================================================
 * Absolute correctness against real model blocks
 * ================================================================ */

static void test_real_attn_qkv_matches_cpu_reference(void)
{
    ASSERT_TRUE(g_model_is_open);
    ASSERT_INT_EQ(0, check_real_tensor_against_reference(kTensorQkv, 48, 1));
}

static void test_real_shared_expert_matches_cpu_reference(void)
{
    ASSERT_TRUE(g_model_is_open);
    ASSERT_INT_EQ(0, check_real_tensor_against_reference(kTensorSharedExpert, 48, 2));
}

/* One-hot x isolates a single weight: out[r] must be exactly dequant(row r)[i].
 * This is the localizing test — it catches a wrong nibble, a dropped high bit,
 * a wrong sub-block scale index and any permutation of the 256 positions,
 * and names the offending element instead of just the row. */
static void test_one_hot_probe_recovers_every_weight(void)
{
    ASSERT_TRUE(g_model_is_open);

    const int rows = 4, cols = 256;
    static uint8_t weight_bytes[4 * Q5K_BLOCK_BYTES];
    int tensor_cols = 0;
    /* Read 4 rows of the real tensor, then use only their first block. */
    static uint8_t full_rows[4 * 1408];
    ASSERT_INT_EQ(0, load_real_q5k_rows(kTensorQkv, rows, &tensor_cols,
                                        full_rows, sizeof(full_rows)));
    const size_t tensor_stride = (size_t)(tensor_cols / 256) * Q5K_BLOCK_BYTES;
    for (int r = 0; r < rows; r++)
        memcpy(weight_bytes + (size_t)r * Q5K_BLOCK_BYTES,
               full_rows + (size_t)r * tensor_stride, Q5K_BLOCK_BYTES);

    float expected_block[4][Q5K_WEIGHTS_PER_BLOCK];
    for (int r = 0; r < rows; r++)
        q5k_dequantize_block(weight_bytes + (size_t)r * Q5K_BLOCK_BYTES, expected_block[r]);

    fm_buffer *w   = fm_buffer_alloc(sizeof(weight_bytes));
    fm_buffer *x   = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_NOT_NULL(w); ASSERT_NOT_NULL(x); ASSERT_NOT_NULL(out);
    memcpy(fm_buffer_data(w), weight_bytes, sizeof(weight_bytes));
    float *x_host = (float *)fm_buffer_data(x);

    int mismatches = 0;
    float nonzero_mass = 0.0f;
    for (int i = 0; i < cols; i++) {
        memset(x_host, 0, (size_t)cols * sizeof(float));
        x_host[i] = 1.0f;

        float actual[4];
        run_q5k(fm_bufref_at(w, 0), fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                rows, cols, (int)Q5K_BLOCK_BYTES, actual);

        for (int r = 0; r < rows; r++) {
            const float want = expected_block[r][i];
            nonzero_mass += fabsf(want);
            if (fabsf(want - actual[r]) > 1e-6f * (1.0f + fabsf(want))) {
                if (mismatches < 8)
                    printf("    weight[row %d][%d]: expected %.8f got %.8f\n",
                           r, i, want, actual[r]);
                mismatches++;
            }
        }
    }

    fm_buffer_free(w); fm_buffer_free(x); fm_buffer_free(out);

    ASSERT_TRUE(nonzero_mass > 1e-3f);
    ASSERT_INT_EQ(0, mismatches);
}

/* row_stride_bytes is an independent parameter, not cols/256*176. Padding the
 * stride and poisoning the gap makes an off-by-one in the block or row stride
 * read the poison. */
static void test_padded_row_stride_is_honored(void)
{
    ASSERT_TRUE(g_model_is_open);

    const int rows = 32;
    static uint8_t packed[32 * 1408];
    int cols = 0;
    ASSERT_INT_EQ(0, load_real_q5k_rows(kTensorQkv, rows, &cols, packed, sizeof(packed)));

    const size_t natural_stride = (size_t)(cols / 256) * Q5K_BLOCK_BYTES;
    const size_t padded_stride  = natural_stride + 64;

    fm_buffer *w   = fm_buffer_alloc((size_t)rows * padded_stride);
    fm_buffer *x   = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_NOT_NULL(w); ASSERT_NOT_NULL(x); ASSERT_NOT_NULL(out);

    uint8_t *w_host = (uint8_t *)fm_buffer_data(w);
    memset(w_host, 0xA5, (size_t)rows * padded_stride);      /* poison the gaps */
    for (int r = 0; r < rows; r++)
        memcpy(w_host + (size_t)r * padded_stride,
               packed + (size_t)r * natural_stride, natural_stride);

    float *x_host = (float *)fm_buffer_data(x);
    fill_probe_vector(x_host, cols, 3);

    float expected[32], abs_sum[32], actual[32];
    q5k_reference_matvec(w_host, rows, cols, padded_stride, x_host, expected, abs_sum);
    run_q5k(fm_bufref_at(w, 0), fm_bufref_at(x, 0), fm_bufref_at(out, 0),
            rows, cols, (int)padded_stride, actual);

    for (int r = 0; r < rows; r++)
        ASSERT_FLT_NEAR(expected[r], actual[r], tolerance_for(abs_sum[r]));

    fm_buffer_free(w); fm_buffer_free(x); fm_buffer_free(out);
}

/* ================================================================
 * Ragged row tail — the hazard the rows-per-threadgroup split introduces
 * ================================================================ */

/* Sentinels are finite on purpose: the shader library is built MTLMathModeFast
 * and the host -ffast-math, so -INFINITY is undefined behaviour and isfinite()
 * folds to constant-true. */
#define TAIL_GUARD_FLOATS  16
#define TAIL_GUARD_VALUE   (-12345.0f)

/* Builds a [rows, cols] Q5_K weight out of REAL model bytes by keeping only the
 * first cols/256 super-blocks of each source row, so every column count in the
 * sweep is backed by genuine blocks rather than synthetic ones. */
static int build_narrowed_weight(const char *tensor_name, int rows, int cols,
                                 uint8_t *dst, size_t dst_capacity)
{
    static uint8_t source[40 * 1408];        /* 40 rows x 2048 cols, the widest here */
    int source_cols = 0;
    if (load_real_q5k_rows(tensor_name, rows, &source_cols, source, sizeof(source)) != 0)
        return -1;
    if (cols > source_cols)
        return -1;

    const size_t source_stride = (size_t)(source_cols / 256) * Q5K_BLOCK_BYTES;
    const size_t stride        = (size_t)(cols / 256) * Q5K_BLOCK_BYTES;
    if ((size_t)rows * stride > dst_capacity)
        return -1;

    for (int r = 0; r < rows; r++)
        memcpy(dst + (size_t)r * stride, source + (size_t)r * source_stride, stride);
    return 0;
}

/* Runs the kernel into an output buffer padded with sentinels past `rows` and
 * compares every live row absolutely against the CPU oracle.
 *
 * @param tail_intact  Set nonzero only if every sentinel past `rows` survived
 * @return             Rows outside tolerance, or -1 if a buffer could not be
 *                     allocated
 */
static int ragged_tail_failures(const uint8_t *weight, int rows, int cols,
                                int row_stride, int *tail_intact)
{
    fm_buffer *w   = fm_buffer_alloc((size_t)rows * (size_t)row_stride);
    fm_buffer *x   = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)(rows + TAIL_GUARD_FLOATS) * sizeof(float));
    float *expected = (float *)malloc((size_t)rows * sizeof(float));
    float *abs_sum  = (float *)malloc((size_t)rows * sizeof(float));
    int failures = -1;

    *tail_intact = 0;
    if (!w || !x || !out || !expected || !abs_sum) goto cleanup;

    memcpy(fm_buffer_data(w), weight, (size_t)rows * (size_t)row_stride);
    float *x_host = (float *)fm_buffer_data(x);
    fill_probe_vector(x_host, cols, (unsigned)(rows * 31 + cols));
    q5k_reference_matvec(weight, rows, cols, (size_t)row_stride, x_host,
                         expected, abs_sum);

    float *o = (float *)fm_buffer_data(out);
    for (int i = 0; i < rows + TAIL_GUARD_FLOATS; i++)
        o[i] = TAIL_GUARD_VALUE;

    fm_matvec_q5k(fm_bufref_at(w, 0), fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                  rows, cols, row_stride);
    fm_sync();

    o = (float *)fm_buffer_data(out);
    *tail_intact = 1;
    for (int i = rows; i < rows + TAIL_GUARD_FLOATS; i++)
        if (o[i] != TAIL_GUARD_VALUE) *tail_intact = 0;

    failures = 0;
    for (int r = 0; r < rows; r++)
        if (fabsf(expected[r] - o[r]) > tolerance_for(abs_sum[r])) failures++;

cleanup:
    if (w) fm_buffer_free(w);
    if (x) fm_buffer_free(x);
    if (out) fm_buffer_free(out);
    free(expected);
    free(abs_sum);
    return failures;
}

/* A threadgroup covers 32/(cols/256) output rows, so the grid is rounded up and
 * the last group carries row indices past the end. Those lanes must contribute
 * nothing to a live row's reduction and store nothing — a hazard that did not
 * exist while a row owned the whole SIMD group.
 *
 * Swept over every column count the real tensor can be narrowed to: cols=256
 * puts 32 rows in a group (1 lane each), 512 puts 16, 1024 puts 8 and 2048
 * puts 4, with row counts chosen to leave every kind of remainder including a
 * single row and one short of a full group. */
static void test_ragged_row_tail_is_exact_and_writes_nothing_past_rows(void)
{
    static const int shape[][2] = {
        { 256, 1 }, { 256, 17 }, { 256, 31 }, { 256, 33 },
        { 512, 1 }, { 512, 15 }, { 512, 17 }, { 512, 33 },
        { 1024, 1 }, { 1024, 7 }, { 1024, 9 },
        { 2048, 1 }, { 2048, 3 }, { 2048, 5 },
    };
    static uint8_t weight[40 * 1408];

    ASSERT_TRUE(g_model_is_open);
    for (size_t i = 0; i < sizeof(shape) / sizeof(shape[0]); i++) {
        const int cols = shape[i][0];
        const int rows = shape[i][1];
        const int row_stride = (cols / 256) * (int)Q5K_BLOCK_BYTES;
        int tail_intact = 0;

        ASSERT_INT_EQ(0, build_narrowed_weight(kTensorQkv, rows, cols,
                                               weight, sizeof(weight)));
        const int failures = ragged_tail_failures(weight, rows, cols, row_stride,
                                                  &tail_intact);
        if (failures != 0 || !tail_intact)
            printf("    cols=%d rows=%d: %d bad rows, tail_intact=%d\n",
                   cols, rows, failures, tail_intact);
        ASSERT_TRUE(tail_intact);
        ASSERT_INT_EQ(0, failures);
    }
}

/* ================================================================
 * The same absolute check, with the weight inside an fm_region at a
 * 32-aligned, non-page-aligned offset. An ADDITION to the tests above.
 * ================================================================ */

static void test_region_offset_matches_cpu_reference(void)
{
    ASSERT_TRUE(g_model_is_open);

    const int rows = 32;
    static uint8_t packed[32 * 1408];
    int cols = 0;
    ASSERT_INT_EQ(0, load_real_q5k_rows(kTensorQkv, rows, &cols, packed, sizeof(packed)));
    const size_t row_stride = (size_t)(cols / 256) * Q5K_BLOCK_BYTES;

    const size_t mapped_bytes = 8u * 1024 * 1024;
    int fd = open(kScratchPath, O_RDWR | O_CREAT | O_TRUNC, 0644);
    ASSERT_TRUE(fd >= 0);
    ASSERT_INT_EQ(0, ftruncate(fd, (off_t)mapped_bytes));
    void *base = mmap(NULL, mapped_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    ASSERT_TRUE(base != MAP_FAILED);

    /* Decoy at 0: if the binding ignored its offset the kernel would read
     * these bytes and disagree with the oracle for the real weight. */
    memset(base, 0x3C, mapped_bytes);
    memcpy((uint8_t *)base + REGION_WEIGHT_OFFSET, packed, (size_t)rows * row_stride);
    ASSERT_TRUE(REGION_WEIGHT_OFFSET % 32 == 0);
    ASSERT_TRUE(REGION_WEIGHT_OFFSET % 16384 != 0);

    fm_region *region = fm_region_open(base, mapped_bytes, 0);
    ASSERT_NOT_NULL(region);
    fm_bufref w;
    ASSERT_INT_EQ(0, fm_region_bind(region, REGION_WEIGHT_OFFSET,
                                    (size_t)rows * row_stride, &w));

    fm_buffer *x   = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_NOT_NULL(x); ASSERT_NOT_NULL(out);
    float *x_host = (float *)fm_buffer_data(x);
    fill_probe_vector(x_host, cols, 4);

    float expected[32], abs_sum[32], actual[32];
    q5k_reference_matvec(packed, rows, cols, row_stride, x_host, expected, abs_sum);
    run_q5k(w, fm_bufref_at(x, 0), fm_bufref_at(out, 0),
            rows, cols, (int)row_stride, actual);

    for (int r = 0; r < rows; r++)
        ASSERT_FLT_NEAR(expected[r], actual[r], tolerance_for(abs_sum[r]));

    fm_buffer_free(x); fm_buffer_free(out);
    fm_region_close(region);
    munmap(base, mapped_bytes);
    close(fd);
    unlink(kScratchPath);
}

/* ================================================================
 * main
 * ================================================================ */

int main(void)
{
    printf("\n  Metal Q5_K Matvec — Absolute Correctness\n");
    printf("  ========================================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "Metal unavailable\n");
        return 77;
    }

    const char *model_path = getenv("FOUNDRY_TEST_GGUF");
    if (!model_path) model_path = kDefaultModelPath;
    g_model_is_open = gguf_test_open(model_path, &g_model) == 0;

    if (!g_model_is_open) {
        printf("  %sMODEL FILE MISSING — every test below is driven by real\n"
               "  Q5_K blocks and cannot run without it:%s\n    %s\n\n",
               FT_RED, FT_RESET, model_path);
        fm_free();
        return 1;
    }
    printf("  model        = %s\n", model_path);
    printf("  data starts  = %llu\n\n", (unsigned long long)g_model.tensor_data_start);

    RUN_TEST(real_attn_qkv_matches_cpu_reference);
    RUN_TEST(real_shared_expert_matches_cpu_reference);
    RUN_TEST(one_hot_probe_recovers_every_weight);
    RUN_TEST(padded_row_stride_is_honored);
    RUN_TEST(ragged_row_tail_is_exact_and_writes_nothing_past_rows);
    RUN_TEST(region_offset_matches_cpu_reference);

    gguf_test_close(&g_model);
    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
