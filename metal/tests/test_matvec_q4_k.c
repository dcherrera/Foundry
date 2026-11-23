/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file test_matvec_q4_k.c
 * @brief Absolute-correctness tests for the Q4_K family: fm_matvec_q4_k_at,
 *   fm_matvec_q4_k_nx, fm_sgemm_q4_k_at and fm_gemm_q4_k_f16_at.
 *
 *   THE GAP THIS CLOSES: matvec_q4_k had no absolute test in this repo. The
 *   region tests compare the kernel against ITSELF at a different binding
 *   offset to prove offset-invariance, so any error that is uniform across
 *   both arms — a scaled row sum, most obviously — cancels and nothing goes
 *   red. test_gemm_q4_k exercises a different kernel. So the matvec was
 *   only ever proven self-consistent, never right.
 *
 *   Two independent anchors, neither of them a GPU-against-GPU comparison:
 *
 *   1. Hand-built super-blocks whose exact answer is written into the test:
 *      the quant nibble, the six-bit scale (including the values above 15
 *      that use the packed high bits) and the six-bit min are probed one at
 *      a time, so a wrong decode of any of the three is a wrong number.
 *   2. Real super-blocks from the model file, bound at their real 32-byte
 *      aligned, non-page-aligned offset inside an fm_region over the whole
 *      15 GiB mapping, against a CPU dequant+dot written here from the
 *      144-byte block spec (per sub-block, not per qs byte-pair like the
 *      kernel).
 *
 *   Offset-invariance and nx-vs-single-column checks are additional.
 *
 *   Exits 77 when there is no Metal device or the model file is absent.
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
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MODEL_PATH \
    "models/Nex-N2-mini/nex-agi_Nex-N2-mini-Q3_K_M.gguf"

/* ggml type id for Q4_K, as it appears in a tensor info record. */
#define GGUF_GGML_TYPE_Q4_K 12

/* Q4_K tensors in this file. dims[0] is cols (fastest-varying). */
#define TENSOR_WIDE   "blk.3.attn_output.weight"     /* [4096, 2048]      */
#define TENSOR_NARROW "blk.0.ffn_down_exps.weight"   /* [512, 2048, 256]  */

#define Q4_K_BLOCK_BYTES   144
#define Q4_K_BLOCK_WEIGHTS 256
#define Q4_K_SUB_BLOCKS    8
#define Q4_K_SUB_WEIGHTS   32

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

/* IEEE half -> float, for the block's d and dmin. */
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

static float f16_at(const uint8_t *p)
{
    return f16_to_f32((uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)));
}

/* ================================================================
 * CPU reference — written from the GGUF Q4_K block spec
 *
 * A 144-byte super-block holds 256 weights:
 *   f16 d; f16 dmin; uint8 scales[12]; uint8 qs[128];
 * The 256 weights are 8 sub-blocks of 32. Sub-block j carries a six-bit
 * scale sc[j] and a six-bit min mn[j], packed into scales[12], and
 *   weight = d * sc[j] * q - dmin * mn[j].
 * Sub-block j reads its quants from qs[(j/2)*32 .. +32), the low nibble
 * when j is even and the high nibble when j is odd.
 * ================================================================ */

/* Unpacks the six-bit scale and min of sub-block j. The first four pairs
 * sit whole in scales[0..7]; the last four keep their low nibble in
 * scales[8..11] and borrow their top two bits from the spare high bits of
 * scales[j-4] and scales[j]. */
static void q4_k_scale_min(const uint8_t *scales, int j, int *sc, int *mn)
{
    if (j < 4) {
        *sc = scales[j] & 63;
        *mn = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *mn = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4);
    }
}

/* The inverse, for building hand-computed blocks. */
static void q4_k_pack_scales(const int *sc, const int *mn, uint8_t *scales)
{
    memset(scales, 0, 12);
    for (int j = 0; j < 4; j++) {
        scales[j]     = (uint8_t)(sc[j] & 63);
        scales[j + 4] = (uint8_t)(mn[j] & 63);
    }
    for (int j = 4; j < 8; j++) {
        scales[j + 4]  = (uint8_t)((sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4));
        scales[j - 4] |= (uint8_t)(((sc[j] >> 4) & 0x03) << 6);
        scales[j]     |= (uint8_t)(((mn[j] >> 4) & 0x03) << 6);
    }
}

/* Dequantizes one 144-byte super-block into 256 doubles. */
static void q4_k_dequant_block(const uint8_t *blk, double *out)
{
    double d    = (double)f16_at(blk);
    double dmin = (double)f16_at(blk + 2);
    const uint8_t *scales = blk + 4;
    const uint8_t *qs     = blk + 16;

    for (int j = 0; j < Q4_K_SUB_BLOCKS; j++) {
        int sc = 0, mn = 0;
        q4_k_scale_min(scales, j, &sc, &mn);
        const uint8_t *q = qs + (j / 2) * Q4_K_SUB_WEIGHTS;
        for (int l = 0; l < Q4_K_SUB_WEIGHTS; l++) {
            int nibble = (j % 2 == 0) ? (q[l] & 0x0F) : (q[l] >> 4);
            out[j * Q4_K_SUB_WEIGHTS + l] = d * sc * nibble - dmin * mn;
        }
    }
}

/* out[0] = dot product, out[1] = sum of |term| (the conditioning, and so
 * the scale a float32 rounding error is allowed to live on). */
static void q4_k_reference_row(const uint8_t *row, const float *x, int cols,
                               double *dot_out, double *magnitude_out)
{
    double dot = 0.0, magnitude = 0.0;
    double w[Q4_K_BLOCK_WEIGHTS];
    int n_blocks = cols / Q4_K_BLOCK_WEIGHTS;

    for (int b = 0; b < n_blocks; b++) {
        q4_k_dequant_block(row + (size_t)b * Q4_K_BLOCK_BYTES, w);
        const float *xb = x + b * Q4_K_BLOCK_WEIGHTS;
        for (int l = 0; l < Q4_K_BLOCK_WEIGHTS; l++) {
            double term = w[l] * (double)xb[l];
            dot += term;
            magnitude += fabs(term);
        }
    }
    *dot_out = dot;
    if (magnitude_out) *magnitude_out = magnitude;
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

/* Writes a super-block from exact parts: scales d/dmin, the eight six-bit
 * scale/min pairs, and 128 packed quant bytes. */
static void write_q4_k_block(uint8_t *blk, float d, float dmin,
                             const int *sc, const int *mn, const uint8_t *qs)
{
    uint16_t hd = f32_to_f16_exact(d), hm = f32_to_f16_exact(dmin);
    blk[0] = (uint8_t)(hd & 0xFFu); blk[1] = (uint8_t)(hd >> 8);
    blk[2] = (uint8_t)(hm & 0xFFu); blk[3] = (uint8_t)(hm >> 8);
    q4_k_pack_scales(sc, mn, blk + 4);
    memcpy(blk + 16, qs, 128);
}

/* Packs weight index `w` of a super-block (0..255) into qs so that its
 * nibble reads back as `value`. Sub-block j = w/32, position l = w%32. */
static void set_q4_k_quant(uint8_t *qs, int w, int value)
{
    int j = w / Q4_K_SUB_WEIGHTS, l = w % Q4_K_SUB_WEIGHTS;
    uint8_t *byte = &qs[(j / 2) * Q4_K_SUB_WEIGHTS + l];
    if (j % 2 == 0) *byte = (uint8_t)((*byte & 0xF0) | (value & 0x0F));
    else            *byte = (uint8_t)((*byte & 0x0F) | ((value & 0x0F) << 4));
}

static float pseudo_random_activation(int i)
{
    uint32_t h = (uint32_t)i * 2654435761u;
    h ^= h >> 15;
    return 0.75f - (float)(h % 1493u) * (1.5f / 1492.0f);
}

static void fill_activations(float *x, int n, int seed)
{
    for (int i = 0; i < n; i++) x[i] = pseudo_random_activation(i + seed);
}

/* ================================================================
 * Anchor 1: hand-computed super-blocks
 * ================================================================ */

/* Four rows of one super-block each, each isolating one part of the
 * decode, with the exact answer written into the test:
 *
 *   row 0  quant + position: every nibble 0 except weight 100 = 5,
 *          d = 1, sc = 1, dmin = 0, x[i] = i        ->  5 * 100 = 500
 *   row 1  the min term:     dmin = 0.25, mn[j] = 6*(j+1), sc = 0,
 *          nibbles 0, x = 1  ->  -0.25 * 32 * sum(mn) = -1728
 *   row 2  the six-bit scale: d = 0.5, sc[j] = 7*(j+1) (four of them
 *          above 15, so the packed high bits must be read), nibbles all
 *          1, dmin = 0, x = 1 ->  0.5 * 32 * sum(sc) = 4032
 *   row 3  both together:    d = 0.5, sc[j] = 2, dmin = 0.25, mn[j] = 4,
 *          nibbles all 3, x = 1
 *          -> 256 * (0.5*2*3 - 0.25*4) = 256 * 2 = 512
 */
static void build_hand_blocks(uint8_t *w, size_t row_stride)
{
    int sc[8], mn[8];
    uint8_t qs[128];

    memset(qs, 0, sizeof(qs));
    for (int j = 0; j < 8; j++) { sc[j] = 1; mn[j] = 0; }
    set_q4_k_quant(qs, 100, 5);
    write_q4_k_block(w + 0 * row_stride, 1.0f, 0.0f, sc, mn, qs);

    memset(qs, 0, sizeof(qs));
    for (int j = 0; j < 8; j++) { sc[j] = 0; mn[j] = 6 * (j + 1); }
    write_q4_k_block(w + 1 * row_stride, 1.0f, 0.25f, sc, mn, qs);

    memset(qs, 0x11, sizeof(qs));            /* every nibble = 1 */
    for (int j = 0; j < 8; j++) { sc[j] = 7 * (j + 1); mn[j] = 0; }
    write_q4_k_block(w + 2 * row_stride, 0.5f, 0.0f, sc, mn, qs);

    memset(qs, 0x33, sizeof(qs));            /* every nibble = 3 */
    for (int j = 0; j < 8; j++) { sc[j] = 2; mn[j] = 4; }
    write_q4_k_block(w + 3 * row_stride, 0.5f, 0.25f, sc, mn, qs);
}

static void test_hand_computed_blocks_are_exact(void)
{
    const int rows = 4, cols = Q4_K_BLOCK_WEIGHTS;
    const size_t row_stride = Q4_K_BLOCK_BYTES;
    const size_t weight_offset = 32;   /* exercise the offset binding */

    fm_buffer *w = fm_buffer_alloc(weight_offset + (size_t)rows * row_stride);
    fm_buffer *x = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_NOT_NULL(w);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);

    build_hand_blocks((uint8_t *)fm_buffer_data(w) + weight_offset, row_stride);
    float *xh = (float *)fm_buffer_data(x);

    for (int i = 0; i < cols; i++) xh[i] = 1.0f;
    memset(fm_buffer_data(out), 0, (size_t)rows * sizeof(float));
    fm_matvec_q4_k_at(fm_bufref_at(w, weight_offset), fm_bufref_at(x, 0),
                      fm_bufref_at(out, 0), rows, cols, (int)row_stride);
    fm_sync();
    const float *o = (const float *)fm_buffer_data(out);
    ASSERT_FLT_NEAR(5.0f,     o[0], 0.0f);     /* one nibble, x = 1 */
    ASSERT_FLT_NEAR(-1728.0f, o[1], 0.0f);
    ASSERT_FLT_NEAR(4032.0f,  o[2], 0.0f);
    ASSERT_FLT_NEAR(512.0f,   o[3], 0.0f);

    for (int i = 0; i < cols; i++) xh[i] = (float)i;
    memset(fm_buffer_data(out), 0, (size_t)rows * sizeof(float));
    fm_matvec_q4_k_at(fm_bufref_at(w, weight_offset), fm_bufref_at(x, 0),
                      fm_bufref_at(out, 0), rows, cols, (int)row_stride);
    fm_sync();
    o = (const float *)fm_buffer_data(out);
    ASSERT_FLT_NEAR(500.0f, o[0], 0.0f);       /* the position probe */
    /* With x[i] = i, sub-block j contributes sum_l (32j + l) = 1024j + 496,
     * and sum_j (j+1)(1024j + 496) = 189888. Row 1 scales that by
     * -dmin * 6 = -1.5, row 2 by d * 7 = 3.5. Both exact in f32. */
    ASSERT_FLT_NEAR(-284832.0f, o[1], 0.0f);
    ASSERT_FLT_NEAR(664608.0f,  o[2], 0.0f);

    fm_buffer_free(w);
    fm_buffer_free(x);
    fm_buffer_free(out);
}

/* ================================================================
 * Anchor 2: real super-blocks from the model file
 * ================================================================ */

typedef struct real_weight {
    fm_region *region;
    fm_bufref  w;
    const uint8_t *host;
    int    rows, cols;
    size_t row_stride;
} real_weight;

/* Binds `rows` rows of a named Q4_K tensor, `row_skip` rows in, at its real
 * offset inside a region over the whole model mapping. Returns 0 on
 * success; the caller releases it with release_real_weight. */
static int open_real_weight(const char *tensor_name, int rows, int row_skip,
                            real_weight *out)
{
    memset(out, 0, sizeof(*out));
    gguf_tensor_location t;
    if (gguf_test_find_tensor(&g_model, tensor_name, &t) != 0) return -1;
    if (t.ggml_type != GGUF_GGML_TYPE_Q4_K) return -1;

    int cols = (int)t.dims[0];
    if (cols % Q4_K_BLOCK_WEIGHTS) return -1;

    size_t row_stride = (size_t)(cols / Q4_K_BLOCK_WEIGHTS) * Q4_K_BLOCK_BYTES;
    size_t offset = (size_t)t.file_offset + (size_t)row_skip * row_stride;
    size_t bytes = (size_t)rows * row_stride;

    out->region = fm_region_open(g_map, g_map_bytes, 64u * 1024 * 1024);
    if (!out->region) return -1;
    if (fm_region_bind(out->region, offset, bytes, &out->w) != 0) {
        fm_region_close(out->region);
        out->region = NULL;
        return -1;
    }
    out->host       = (const uint8_t *)g_map + offset;
    out->rows       = rows;
    out->cols       = cols;
    out->row_stride = row_stride;
    return 0;
}

static void release_real_weight(real_weight *rw)
{
    if (rw->region) fm_region_close(rw->region);
    rw->region = NULL;
}

/* Asserts every entry of `gpu` (rows values, one per weight row) against
 * the CPU reference over the same bytes, and reports the margins. */
static void assert_rows_match_reference(const real_weight *rw, const float *x,
                                        const float *gpu, double per_row_tol,
                                        double rms_tol, const char *label)
{
    double worst_ratio = 0.0, sum_sq = 0.0, max_abs_err = 0.0;
    for (int r = 0; r < rw->rows; r++) {
        double ref = 0.0, magnitude = 0.0;
        q4_k_reference_row(rw->host + (size_t)r * rw->row_stride, x, rw->cols,
                           &ref, &magnitude);
        double err = fabs((double)gpu[r] - ref);
        ASSERT_TRUE(err <= per_row_tol * magnitude);
        if (magnitude > 0.0 && err / magnitude > worst_ratio)
            worst_ratio = err / magnitude;
        if (err > max_abs_err) max_abs_err = err;
        sum_sq += ref * ref;
    }
    double rms = sqrt(sum_sq / (double)rw->rows);
    ASSERT_TRUE(rms > 1e-6);
    ASSERT_TRUE(max_abs_err <= rms_tol * rms);
    printf("      %s rows=%d cols=%d: worst err/|terms| = %.3g, "
           "max err/RMS = %.3g\n",
           label, rw->rows, rw->cols, worst_ratio, max_abs_err / rms);
}

static void check_real_matvec(const char *tensor_name, int rows, int row_skip)
{
    real_weight rw;
    ASSERT_INT_EQ(0, open_real_weight(tensor_name, rows, row_skip, &rw));
    if (!rw.region) return;
    ASSERT_INT_EQ(0, (int)((size_t)(rw.host - (const uint8_t *)g_map) % 32));

    fm_buffer *x = fm_buffer_alloc((size_t)rw.cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    float *xh = (float *)fm_buffer_data(x);
    fill_activations(xh, rw.cols, 0);

    memset(fm_buffer_data(out), 0, (size_t)rows * sizeof(float));
    fm_matvec_q4_k_at(rw.w, fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                      rows, rw.cols, (int)rw.row_stride);
    fm_sync();

    assert_rows_match_reference(&rw, xh, (const float *)fm_buffer_data(out),
                                1e-5, 1e-5, tensor_name);

    fm_buffer_free(x);
    fm_buffer_free(out);
    release_real_weight(&rw);
}

static void test_real_wide_tensor_matches_cpu_reference(void)
{
    check_real_matvec(TENSOR_WIDE, REAL_ROWS, 0);
}

static void test_real_narrow_tensor_matches_cpu_reference(void)
{
    check_real_matvec(TENSOR_NARROW, REAL_ROWS, 0);
}

static void test_real_tensor_bound_past_its_first_rows(void)
{
    check_real_matvec(TENSOR_WIDE, REAL_ROWS, 3);
}

/* ================================================================
 * The ragged row tail
 * ================================================================ */

/* Floats of sentinel written past the output rows, to catch a store from a
 * lane whose row index ran off the end of the grid's last threadgroup. */
#define TAIL_GUARD_FLOATS  16
#define TAIL_GUARD_VALUE   (-12345.0f)

/*
 * Dispatches both matvec forms over `rows` rows of `tensor_name` into an output
 * buffer padded with a sentinel, and asserts the valid rows against the CPU
 * reference and the sentinel unchanged.
 *
 * @param tensor_name  Q4_K tensor to bind, at its real offset in the model
 * @param rows         Rows to dispatch — the point is a count that is NOT a
 *                     multiple of the threadgroup's row count
 * @param nx_columns   Activation columns for the _nx arm (>= 1)
 */
static void check_ragged_row_tail(const char *tensor_name, int rows, int nx_columns)
{
    real_weight rw;
    ASSERT_INT_EQ(0, open_real_weight(tensor_name, rows, 0, &rw));
    if (!rw.region) return;

    fm_buffer *x = fm_buffer_alloc((size_t)nx_columns * rw.cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc(((size_t)nx_columns * rows + TAIL_GUARD_FLOATS)
                                     * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    float *xh = (float *)fm_buffer_data(x);
    for (int t = 0; t < nx_columns; t++)
        fill_activations(xh + (size_t)t * rw.cols, rw.cols, 17 * (t + 1));

    /* Single-column arm: only rows values are written, so everything from
     * `rows` on must still hold the sentinel. */
    float *o = (float *)fm_buffer_data(out);
    for (size_t i = 0; i < (size_t)nx_columns * rows + TAIL_GUARD_FLOATS; i++)
        o[i] = TAIL_GUARD_VALUE;
    fm_matvec_q4_k_at(rw.w, fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                      rows, rw.cols, (int)rw.row_stride);
    fm_sync();

    o = (float *)fm_buffer_data(out);
    assert_rows_match_reference(&rw, xh, o, 1e-5, 1e-5, tensor_name);
    for (int i = 0; i < TAIL_GUARD_FLOATS; i++)
        ASSERT_FLT_NEAR(TAIL_GUARD_VALUE, o[rows + i], 0.0f);

    /* Batched arm: writes nx_columns * rows values, so only the pad past them
     * must survive.  _nx shares the lane split, so it shares the hazard. */
    for (size_t i = 0; i < (size_t)nx_columns * rows + TAIL_GUARD_FLOATS; i++)
        o[i] = TAIL_GUARD_VALUE;
    fm_matvec_q4_k_nx(rw.w, fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                      rows, rw.cols, (int)rw.row_stride, nx_columns);
    fm_sync();

    o = (float *)fm_buffer_data(out);
    for (int t = 0; t < nx_columns; t++) {
        char label[64];
        snprintf(label, sizeof(label), "%s nx col %d", tensor_name, t);
        assert_rows_match_reference(&rw, xh + (size_t)t * rw.cols,
                                    o + (size_t)t * rows, 1e-5, 1e-5, label);
    }
    for (int i = 0; i < TAIL_GUARD_FLOATS; i++)
        ASSERT_FLT_NEAR(TAIL_GUARD_VALUE, o[(size_t)nx_columns * rows + i], 0.0f);

    fm_buffer_free(x);
    fm_buffer_free(out);
    release_real_weight(&rw);
}

/*
 * A threadgroup covers 32/(cols/256) output rows, so the grid is rounded up and
 * the last group carries row indices past the end.  Those lanes must contribute
 * nothing to a live row's reduction and store nothing — a hazard that did not
 * exist while every threadgroup owned exactly one row.
 *
 * The narrow tensor is the MoE expert shape: cols=512 gives a row 2 lanes and
 * puts 16 rows in a group, the widest split in the model, so a row count of 33
 * leaves a group holding one live row and fifteen dead ones.  The wide tensor
 * (cols=4096) splits a group only 2 ways and checks the near-degenerate end.
 */
static void test_ragged_row_tail_is_exact_and_writes_nothing_past_rows(void)
{
    check_ragged_row_tail(TENSOR_NARROW, 1, 3);
    check_ragged_row_tail(TENSOR_NARROW, 33, 3);
    check_ragged_row_tail(TENSOR_NARROW, 47, 13);
    check_ragged_row_tail(TENSOR_WIDE, 15, 3);
}

/* ================================================================
 * The batched (_nx) form
 * ================================================================ */

#define NX_COLUMNS 13   /* 8 + 5: one full tile plus a narrower remainder tile */

static void test_nx_matches_cpu_reference(void)
{
    real_weight rw;
    ASSERT_INT_EQ(0, open_real_weight(TENSOR_WIDE, 128, 0, &rw));
    if (!rw.region) return;

    fm_buffer *x = fm_buffer_alloc((size_t)NX_COLUMNS * rw.cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)NX_COLUMNS * rw.rows * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    float *xh = (float *)fm_buffer_data(x);
    for (int t = 0; t < NX_COLUMNS; t++)
        fill_activations(xh + (size_t)t * rw.cols, rw.cols, 31 * (t + 1));

    memset(fm_buffer_data(out), 0, (size_t)NX_COLUMNS * rw.rows * sizeof(float));
    fm_matvec_q4_k_nx(rw.w, fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                      rw.rows, rw.cols, (int)rw.row_stride, NX_COLUMNS);
    fm_sync();

    const float *gpu = (const float *)fm_buffer_data(out);
    for (int t = 0; t < NX_COLUMNS; t++) {
        char label[64];
        snprintf(label, sizeof(label), "nx col %d", t);
        assert_rows_match_reference(&rw, xh + (size_t)t * rw.cols,
                                    gpu + (size_t)t * rw.rows, 1e-5, 1e-5, label);
    }

    fm_buffer_free(x);
    fm_buffer_free(out);
    release_real_weight(&rw);
}

/* Additional, on top of the absolute check: the documented contract is that
 * a batched column is BIT-identical to the single-column kernel on that
 * column, which is stronger than "close". */
static void test_nx_is_bit_identical_to_single_column(void)
{
    real_weight rw;
    ASSERT_INT_EQ(0, open_real_weight(TENSOR_WIDE, 128, 0, &rw));
    if (!rw.region) return;

    fm_buffer *x = fm_buffer_alloc((size_t)NX_COLUMNS * rw.cols * sizeof(float));
    fm_buffer *nx_out = fm_buffer_alloc((size_t)NX_COLUMNS * rw.rows * sizeof(float));
    fm_buffer *one_out = fm_buffer_alloc((size_t)rw.rows * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(nx_out);
    ASSERT_NOT_NULL(one_out);
    float *xh = (float *)fm_buffer_data(x);
    for (int t = 0; t < NX_COLUMNS; t++)
        fill_activations(xh + (size_t)t * rw.cols, rw.cols, 97 * (t + 1));

    fm_matvec_q4_k_nx(rw.w, fm_bufref_at(x, 0), fm_bufref_at(nx_out, 0),
                      rw.rows, rw.cols, (int)rw.row_stride, NX_COLUMNS);
    fm_sync();

    float *batched = malloc((size_t)NX_COLUMNS * rw.rows * sizeof(float));
    ASSERT_NOT_NULL(batched);
    memcpy(batched, fm_buffer_data(nx_out),
           (size_t)NX_COLUMNS * rw.rows * sizeof(float));

    for (int t = 0; t < NX_COLUMNS; t++) {
        fm_matvec_q4_k_at(rw.w, fm_bufref_at(x, (size_t)t * rw.cols * sizeof(float)),
                          fm_bufref_at(one_out, 0), rw.rows, rw.cols,
                          (int)rw.row_stride);
        fm_sync();
        const float *single = (const float *)fm_buffer_data(one_out);
        for (int r = 0; r < rw.rows; r++)
            ASSERT_FLT_NEAR(single[r], batched[(size_t)t * rw.rows + r], 0.0f);
    }

    free(batched);
    fm_buffer_free(x);
    fm_buffer_free(nx_out);
    fm_buffer_free(one_out);
    release_real_weight(&rw);
}

/* ================================================================
 * The GEMM entry points, now that they carry offsets
 * ================================================================ */

#define GEMM_BATCH 8

/* Both batched entry points write the same layout — fm_sgemm_q4_k_at's
 * out[B, rows] and fm_gemm_q4_k_f16_at's out[M, N] with N = weight rows are
 * the same index, b * rows + r. */
static void assert_batched_matches_reference(const real_weight *rw,
                                             const float *x, const float *gpu,
                                             int batch, double rms_tol,
                                             const char *label)
{
    double sum_sq = 0.0, max_abs_err = 0.0;
    for (int r = 0; r < rw->rows; r++) {
        for (int b = 0; b < batch; b++) {
            double ref = 0.0;
            q4_k_reference_row(rw->host + (size_t)r * rw->row_stride,
                               x + (size_t)b * rw->cols, rw->cols, &ref, NULL);
            double err = fabs((double)gpu[(size_t)b * rw->rows + r] - ref);
            if (err > max_abs_err) max_abs_err = err;
            sum_sq += ref * ref;
        }
    }
    double rms = sqrt(sum_sq / ((double)rw->rows * batch));
    ASSERT_TRUE(rms > 1e-6);
    ASSERT_TRUE(max_abs_err <= rms_tol * rms);
    printf("      %s rows=%d cols=%d B=%d: max err/RMS = %.3g\n",
           label, rw->rows, rw->cols, batch, max_abs_err / rms);
}

static void test_sgemm_at_reads_a_region_resident_weight(void)
{
    real_weight rw;
    ASSERT_INT_EQ(0, open_real_weight(TENSOR_WIDE, 128, 0, &rw));
    if (!rw.region) return;

    fm_buffer *x = fm_buffer_alloc((size_t)GEMM_BATCH * rw.cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)GEMM_BATCH * rw.rows * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    float *xh = (float *)fm_buffer_data(x);
    for (int b = 0; b < GEMM_BATCH; b++)
        fill_activations(xh + (size_t)b * rw.cols, rw.cols, 11 * (b + 1));

    memset(fm_buffer_data(out), 0, (size_t)GEMM_BATCH * rw.rows * sizeof(float));
    fm_sgemm_q4_k_at(rw.w, fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                     rw.rows, rw.cols, (int)rw.row_stride, GEMM_BATCH);
    fm_sync();

    /* F32 simdgroup accumulation, so this stays near the matvec's margin. */
    assert_batched_matches_reference(&rw, xh, (const float *)fm_buffer_data(out),
                                     GEMM_BATCH, 1e-4, "sgemm_q4_k_at");

    fm_buffer_free(x);
    fm_buffer_free(out);
    release_real_weight(&rw);
}

static void test_gemm_f16_at_reads_a_region_resident_weight(void)
{
    real_weight rw;
    ASSERT_INT_EQ(0, open_real_weight(TENSOR_WIDE, 128, 0, &rw));
    if (!rw.region) return;

    int M = GEMM_BATCH, N = rw.rows, K = rw.cols;
    fm_buffer *x = fm_buffer_alloc((size_t)M * K * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)M * N * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    float *xh = (float *)fm_buffer_data(x);
    for (int m = 0; m < M; m++)
        fill_activations(xh + (size_t)m * K, K, 23 * (m + 1));

    memset(fm_buffer_data(out), 0, (size_t)M * N * sizeof(float));
    fm_gemm_q4_k_f16_at(rw.w, fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                        M, N, K, (int)rw.row_stride);
    fm_sync();

    /* F16 staging on both operands: two decades looser than the F32 paths,
     * matching the tolerance test_gemm_q4_k_f16 already uses. */
    assert_batched_matches_reference(&rw, xh, (const float *)fm_buffer_data(out),
                                     M, 2e-2, "gemm_q4_k_f16_at");

    fm_buffer_free(x);
    fm_buffer_free(out);
    release_real_weight(&rw);
}

/* ================================================================
 * main
 * ================================================================ */

int main(void)
{
    printf("\n  Metal Q4_K Matvec / GEMM Absolute-Correctness Tests\n");
    printf("  ===================================================\n\n");

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

    RUN_TEST(real_wide_tensor_matches_cpu_reference);
    RUN_TEST(real_narrow_tensor_matches_cpu_reference);
    RUN_TEST(real_tensor_bound_past_its_first_rows);
    RUN_TEST(ragged_row_tail_is_exact_and_writes_nothing_past_rows);

    RUN_TEST(nx_matches_cpu_reference);
    RUN_TEST(nx_is_bit_identical_to_single_column);

    RUN_TEST(sgemm_at_reads_a_region_resident_weight);
    RUN_TEST(gemm_f16_at_reads_a_region_resident_weight);

    unmap_model();
    gguf_test_close(&g_model);
    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
