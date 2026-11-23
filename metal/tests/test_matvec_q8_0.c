/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_matvec_q8_0.c
 * @brief Absolute-correctness tests for fm_matvec_q8_0.
 *
 *   Two independent anchors, no GPU-against-GPU comparison as a primary
 *   assertion:
 *
 *   1. Hand-computed blocks. Scales and quants are chosen so the exact
 *      answer is a small integer written into the test, so the assertions
 *      depend on nothing but the Q8_0 block spec.
 *   2. Real blocks from the model file, bound at their real (32-byte
 *      aligned, non-page-aligned) offset inside an fm_region over the whole
 *      15 GiB mapping, against a CPU dequant+dot written here from the
 *      34-byte block spec.
 *
 *   Offset-invariance checks are additional, never a substitute: a uniform
 *   error cancels when both arms are the same kernel.
 *
 *   Exits 77 when there is no Metal device or the model file is absent, so
 *   a machine without either skips cleanly.
 *
 * @status All tests pass on M4 / macOS 26; each was proven able to fail.
 * @issues None
 * @todo None
 */

#include "lib/test.h"
#include "metal/fm_metal.h"
#include "metal/tests/gguf_test_reader.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MODEL_PATH \
    "models/Nex-N2-mini/nex-agi_Nex-N2-mini-Q3_K_M.gguf"

/* ggml type id for Q8_0, as it appears in a tensor info record. */
#define GGUF_GGML_TYPE_Q8_0 8

/* Q8_0 tensors in this file. dims[0] is cols (fastest-varying), dims[1] rows. */
#define TENSOR_WIDE   "blk.0.ffn_gate_shexp.weight"   /* [2048, 512]  */
#define TENSOR_NARROW "blk.0.ffn_down_shexp.weight"   /* [512, 2048]  */

/* Rows of a real tensor a test drives. Enough that a per-row bug cannot
 * hide, small enough that the double-precision CPU reference is instant. */
#define REAL_ROWS 256

/* The container directory comes from the shared reader; the mapping is
 * this test's own, because binding a weight where it really lives needs
 * the file as one page-aligned region, not a pread into a buffer. */
static gguf_test_file g_model;
static void          *g_map = NULL;
static size_t         g_map_bytes = 0;
static int            g_map_fd = -1;

/* Maps the model read/write-private: nothing is written, but Metal's
 * no-copy buffers are created read/write so the pages must be writable. */
static int map_model(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return -1; }
    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); return -1; }
    g_map = base;
    g_map_bytes = (size_t)st.st_size;
    g_map_fd = fd;
    return 0;
}

static void unmap_model(void)
{
    if (g_map) munmap(g_map, g_map_bytes);
    if (g_map_fd >= 0) close(g_map_fd);
    g_map = NULL;
    g_map_fd = -1;
}

/* IEEE half -> float, for the F16 block scale. Subnormals matter: real
 * Q8_0 scales do reach small exponents. */
static float f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            bits = sign | (exp << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* A Q8_0 block is 34 bytes, so its scale is only ever 2-byte aligned —
 * read it from bytes rather than through a uint16_t*. */
static float f16_at(const uint8_t *p)
{
    return f16_to_f32((uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)));
}

/* ================================================================
 * CPU reference — written from the GGUF Q8_0 block spec
 *
 * A block is 34 bytes: an F16 scale d followed by 32 int8 quants, and
 * weight[i] = d * q[i]. A row of `cols` weights is cols/32 such blocks
 * laid end to end.
 * ================================================================ */

#define Q8_0_BLOCK_BYTES   34
#define Q8_0_BLOCK_WEIGHTS 32

/* Accumulates in double so the reference is not itself a source of the
 * error the assertions are measuring. */
static double q8_0_reference_row(const uint8_t *row, const float *x, int cols)
{
    double acc = 0.0;
    int n_blocks = cols / Q8_0_BLOCK_WEIGHTS;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row + (size_t)b * Q8_0_BLOCK_BYTES;
        double d = (double)f16_at(blk);
        const int8_t *q = (const int8_t *)(blk + 2);
        for (int i = 0; i < Q8_0_BLOCK_WEIGHTS; i++)
            acc += d * (double)q[i] * (double)x[b * Q8_0_BLOCK_WEIGHTS + i];
    }
    return acc;
}

/* Sum of |term| over the row: the conditioning of the dot product, and so
 * the scale a float32 rounding error is allowed to live on. */
static double q8_0_row_magnitude(const uint8_t *row, const float *x, int cols)
{
    double acc = 0.0;
    int n_blocks = cols / Q8_0_BLOCK_WEIGHTS;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *blk = row + (size_t)b * Q8_0_BLOCK_BYTES;
        double d = (double)f16_at(blk);
        const int8_t *q = (const int8_t *)(blk + 2);
        for (int i = 0; i < Q8_0_BLOCK_WEIGHTS; i++)
            acc += fabs(d * (double)q[i] * (double)x[b * Q8_0_BLOCK_WEIGHTS + i]);
    }
    return acc;
}

/* ================================================================
 * Small helpers
 * ================================================================ */

static uint16_t f32_to_f16_exact(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Writes one Q8_0 block: scale `d` then the 32 quants in `q`. */
static void write_q8_0_block(uint8_t *blk, float d, const int8_t *q)
{
    uint16_t h = f32_to_f16_exact(d);
    blk[0] = (uint8_t)(h & 0xFFu);
    blk[1] = (uint8_t)(h >> 8);
    memcpy(blk + 2, q, Q8_0_BLOCK_WEIGHTS);
}

static float pseudo_random_activation(int i)
{
    /* Deterministic, spans both signs, never zero — a zero activation
     * would hide an indexing bug on that column. */
    uint32_t h = (uint32_t)i * 2654435761u;
    h ^= h >> 15;
    return 0.75f - (float)(h % 1493u) * (1.5f / 1492.0f);
}

/* ================================================================
 * Anchor 1: hand-computed blocks
 * ================================================================ */

/* Three rows of one block each, built so the exact answer is written into
 * the test rather than produced by any reference code:
 *
 *   row 0: d = 0.5,  q[i] = i - 16,          x = all 1  -> 0.5 * -16 = -8
 *   row 1: d = 2.0,  q[i] = 1,               x = all 1  -> 2.0 *  32 =  64
 *   row 2: d = 1.0,  q[5] = 3, rest 0,       x[i] = i   -> 1.0 * 3 * 5 = 15
 *
 * Row 2 is the index probe: any transposition of the quant or activation
 * indexing inside a block moves that 3 onto a different x and changes the
 * answer. All three scales are exactly representable in F16, so the
 * comparison is exact. */
static void test_hand_computed_blocks_are_exact(void)
{
    const int rows = 3, cols = 32;
    const size_t row_stride = Q8_0_BLOCK_BYTES;

    /* Placed at a nonzero 32-byte offset so even this smallest case goes
     * through the offset-bearing binding. */
    const size_t weight_offset = 32;
    fm_buffer *w = fm_buffer_alloc(weight_offset + (size_t)rows * row_stride);
    fm_buffer *x = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_NOT_NULL(w);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);

    uint8_t *wb = (uint8_t *)fm_buffer_data(w) + weight_offset;
    int8_t q[Q8_0_BLOCK_WEIGHTS];

    for (int i = 0; i < cols; i++) q[i] = (int8_t)(i - 16);
    write_q8_0_block(wb + 0 * row_stride, 0.5f, q);

    for (int i = 0; i < cols; i++) q[i] = 1;
    write_q8_0_block(wb + 1 * row_stride, 2.0f, q);

    memset(q, 0, sizeof(q));
    q[5] = 3;
    write_q8_0_block(wb + 2 * row_stride, 1.0f, q);

    float *xh = (float *)fm_buffer_data(x);

    /* Rows 0 and 1 with x = all ones. */
    for (int i = 0; i < cols; i++) xh[i] = 1.0f;
    memset(fm_buffer_data(out), 0, (size_t)rows * sizeof(float));
    fm_matvec_q8_0(fm_bufref_at(w, weight_offset), fm_bufref_at(x, 0),
                   fm_bufref_at(out, 0), rows, cols, (int)row_stride);
    fm_sync();
    const float *o = (const float *)fm_buffer_data(out);
    ASSERT_FLT_NEAR(-8.0f, o[0], 0.0f);
    ASSERT_FLT_NEAR(64.0f, o[1], 0.0f);
    ASSERT_FLT_NEAR(3.0f,  o[2], 0.0f);   /* q[5]=3 against x[5]=1 */

    /* Row 2 with x[i] = i: the index probe. */
    for (int i = 0; i < cols; i++) xh[i] = (float)i;
    memset(fm_buffer_data(out), 0, (size_t)rows * sizeof(float));
    fm_matvec_q8_0(fm_bufref_at(w, weight_offset), fm_bufref_at(x, 0),
                   fm_bufref_at(out, 0), rows, cols, (int)row_stride);
    fm_sync();
    o = (const float *)fm_buffer_data(out);
    ASSERT_FLT_NEAR(15.0f, o[2], 0.0f);

    /* sum_i (i-16)*i = sum i^2 - 16 sum i = 10416 - 7936 = 2480, times d = 0.5. */
    ASSERT_FLT_NEAR(1240.0f, o[0], 0.0f);
    /* sum_i i = 496, times d = 2.0. */
    ASSERT_FLT_NEAR(992.0f, o[1], 0.0f);

    fm_buffer_free(w);
    fm_buffer_free(x);
    fm_buffer_free(out);
}

/* A block spanning several F16 exponents and both signs of quant, still
 * hand-checkable: two blocks in one row, x = all ones, so the answer is
 * d0 * sum(q0) + d1 * sum(q1). */
static void test_multi_block_row_is_exact(void)
{
    const int rows = 1, cols = 64;
    const size_t row_stride = 2 * Q8_0_BLOCK_BYTES;

    fm_buffer *w = fm_buffer_alloc(row_stride);
    fm_buffer *x = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc(sizeof(float));
    ASSERT_NOT_NULL(w);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);

    uint8_t *wb = (uint8_t *)fm_buffer_data(w);
    int8_t q[Q8_0_BLOCK_WEIGHTS];

    /* Block 0: d = 0.25, quants sum to 32 * 4 = 128  ->  32 */
    for (int i = 0; i < Q8_0_BLOCK_WEIGHTS; i++) q[i] = 4;
    write_q8_0_block(wb, 0.25f, q);

    /* Block 1: d = 16.0, quants sum to -128         ->  -2048 */
    for (int i = 0; i < Q8_0_BLOCK_WEIGHTS; i++) q[i] = -4;
    write_q8_0_block(wb + Q8_0_BLOCK_BYTES, 16.0f, q);

    float *xh = (float *)fm_buffer_data(x);
    for (int i = 0; i < cols; i++) xh[i] = 1.0f;

    memset(fm_buffer_data(out), 0, sizeof(float));
    fm_matvec_q8_0(fm_bufref_at(w, 0), fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                   rows, cols, (int)row_stride);
    fm_sync();

    /* A kernel that reused block 0's scale for block 1 would return
     * 32 + 0.25 * -128 = 0 instead. */
    ASSERT_FLT_NEAR(-2016.0f, ((const float *)fm_buffer_data(out))[0], 0.0f);

    fm_buffer_free(w);
    fm_buffer_free(x);
    fm_buffer_free(out);
}

/* ================================================================
 * Anchor 2: real blocks from the model file
 * ================================================================ */

/* Runs the kernel on `rows` rows of the named Q8_0 tensor, bound at its
 * real offset inside a region over the whole file, and asserts every row
 * against the CPU reference over the same bytes.
 *
 * `row_skip` shifts the bound range by whole rows; a nonzero value must
 * still agree with the reference for those later rows, which is what a
 * dropped bind offset would break. */
static void check_real_tensor(const char *tensor_name, int rows, int row_skip)
{
    gguf_tensor_location t;
    ASSERT_INT_EQ(0, gguf_test_find_tensor(&g_model, tensor_name, &t));
    ASSERT_INT_EQ(GGUF_GGML_TYPE_Q8_0, (int)t.ggml_type);

    int cols = (int)t.dims[0];
    ASSERT_INT_EQ(0, cols % Q8_0_BLOCK_WEIGHTS);
    ASSERT_TRUE((int)t.dims[1] >= rows + row_skip);

    size_t row_stride = (size_t)(cols / Q8_0_BLOCK_WEIGHTS) * Q8_0_BLOCK_BYTES;
    size_t weight_offset = (size_t)t.file_offset + (size_t)row_skip * row_stride;
    size_t weight_bytes = (size_t)rows * row_stride;

    /* The real thing this design exists for: a 32-byte aligned offset that
     * is not a page boundary, inside a mapping bigger than one buffer. */
    ASSERT_INT_EQ(0, (int)(weight_offset % 32));

    fm_region *region = fm_region_open(g_map, g_map_bytes, 16u * 1024 * 1024);
    ASSERT_NOT_NULL(region);
    if (!region) return;

    fm_bufref w;
    ASSERT_INT_EQ(0, fm_region_bind(region, weight_offset, weight_bytes, &w));

    fm_buffer *x = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    float *xh = (float *)fm_buffer_data(x);
    for (int i = 0; i < cols; i++) xh[i] = pseudo_random_activation(i);

    memset(fm_buffer_data(out), 0, (size_t)rows * sizeof(float));
    fm_matvec_q8_0(w, fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                   rows, cols, (int)row_stride);
    fm_sync();

    const uint8_t *host = (const uint8_t *)g_map + weight_offset;
    const float *gpu = (const float *)fm_buffer_data(out);

    double worst_ratio = 0.0, sum_sq = 0.0, max_abs_err = 0.0;
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = host + (size_t)r * row_stride;
        double ref = q8_0_reference_row(row, xh, cols);
        double mag = q8_0_row_magnitude(row, xh, cols);
        double err = fabs((double)gpu[r] - ref);

        /* Float32 summation error lives on the sum of |terms|, not on the
         * (possibly cancelling) result, so that is what bounds it. The
         * measured worst case is ~1e-7 of it; 1e-5 leaves two decades. */
        ASSERT_TRUE(err <= 1e-5 * mag);
        if (mag > 0.0 && err / mag > worst_ratio) worst_ratio = err / mag;
        if (err > max_abs_err) max_abs_err = err;
        sum_sq += ref * ref;
    }

    /* Scale-invariant check on top: a uniformly scaled row sum shifts this
     * far past 1e-5 even where per-row cancellation is severe. */
    double rms = sqrt(sum_sq / (double)rows);
    ASSERT_TRUE(rms > 1e-6);
    ASSERT_TRUE(max_abs_err <= 1e-5 * rms);

    printf("      %s rows=%d cols=%d skip=%d: worst err/|terms| = %.3g, "
           "max err/RMS = %.3g\n",
           tensor_name, rows, cols, row_skip, worst_ratio, max_abs_err / rms);

    fm_buffer_free(x);
    fm_buffer_free(out);
    fm_region_close(region);
}

static void test_real_wide_tensor_matches_cpu_reference(void)
{
    check_real_tensor(TENSOR_WIDE, REAL_ROWS, 0);
}

static void test_real_narrow_tensor_matches_cpu_reference(void)
{
    /* 512 cols = 16 blocks per row: the shape where a block-per-lane split
     * would leave half the SIMD group idle. */
    check_real_tensor(TENSOR_NARROW, REAL_ROWS, 0);
}

static void test_real_tensor_bound_past_its_first_rows(void)
{
    /* The same tensor bound one row later. The reference moves with it, so
     * a binding that silently dropped its offset would now disagree. */
    check_real_tensor(TENSOR_WIDE, REAL_ROWS, 1);
}

/* Extra, not a substitute for the absolute checks: the same weight bytes
 * read through two different offsets must give the same numbers, and a
 * different weight must not. */
static void test_offset_binding_selects_the_right_rows(void)
{
    gguf_tensor_location t;
    ASSERT_INT_EQ(0, gguf_test_find_tensor(&g_model, TENSOR_WIDE, &t));

    int cols = (int)t.dims[0], rows = 32;
    size_t row_stride = (size_t)(cols / Q8_0_BLOCK_WEIGHTS) * Q8_0_BLOCK_BYTES;
    size_t base_offset = (size_t)t.file_offset;

    fm_region *region = fm_region_open(g_map, g_map_bytes, 16u * 1024 * 1024);
    ASSERT_NOT_NULL(region);
    if (!region) return;

    fm_buffer *x = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    float *xh = (float *)fm_buffer_data(x);
    for (int i = 0; i < cols; i++) xh[i] = pseudo_random_activation(i + 7);

    float first[32], shifted[32];
    fm_bufref w;

    ASSERT_INT_EQ(0, fm_region_bind(region, base_offset, (size_t)rows * row_stride, &w));
    fm_matvec_q8_0(w, fm_bufref_at(x, 0), fm_bufref_at(out, 0), rows, cols, (int)row_stride);
    fm_sync();
    memcpy(first, fm_buffer_data(out), sizeof(first));

    ASSERT_INT_EQ(0, fm_region_bind(region, base_offset + row_stride,
                                    (size_t)rows * row_stride, &w));
    fm_matvec_q8_0(w, fm_bufref_at(x, 0), fm_bufref_at(out, 0), rows, cols, (int)row_stride);
    fm_sync();
    memcpy(shifted, fm_buffer_data(out), sizeof(shifted));

    /* Shifting by one row must slide the answers by one row, exactly. */
    for (int r = 0; r + 1 < rows; r++)
        ASSERT_FLT_NEAR(first[r + 1], shifted[r], 0.0f);
    ASSERT_TRUE(fabsf(first[0] - shifted[0]) > 1e-6f);

    fm_buffer_free(x);
    fm_buffer_free(out);
    fm_region_close(region);
}

/* ================================================================
 * main
 * ================================================================ */

int main(void)
{
    printf("\n  Metal Q8_0 Matvec Tests\n");
    printf("  =======================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "Metal unavailable\n");
        return 77;
    }
    if (gguf_test_open(MODEL_PATH, &g_model) != 0 || map_model(MODEL_PATH) != 0) {
        fprintf(stderr, "model file unavailable: %s\n", MODEL_PATH);
        gguf_test_close(&g_model);
        fm_free();
        return 77;
    }
    printf("  model                    = %s\n", MODEL_PATH);
    printf("  tensors                  = %llu, data starts at %llu\n\n",
           (unsigned long long)g_model.tensor_count,
           (unsigned long long)g_model.tensor_data_start);

    RUN_TEST(hand_computed_blocks_are_exact);
    RUN_TEST(multi_block_row_is_exact);

    RUN_TEST(real_wide_tensor_matches_cpu_reference);
    RUN_TEST(real_narrow_tensor_matches_cpu_reference);
    RUN_TEST(real_tensor_bound_past_its_first_rows);
    RUN_TEST(offset_binding_selects_the_right_rows);

    unmap_model();
    gguf_test_close(&g_model);
    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
