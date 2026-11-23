/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_metal_norm.c — RMS normalization kernels (fm_rms_norm,
 * fm_rms_norm_nx, fm_qk_rms_norm)
 *
 * fm_qk_rms_norm is checked against the CPU-reference golden set in
 * metal/tests/golden (provenance in fm_golden_vectors.h). The batched and
 * single-vector RMS forms are checked against a CPU reference computed here
 * — never against each other, and never against another GPU result.
 *
 * Two properties get their own tests because the obvious test would not
 * notice them:
 *   - the QK-norm weight is SHARED by every head, so a kernel that indexes
 *     it per head is wrong; a uniform weight vector hides that completely,
 *     so the trap test uses a weight buffer whose per-head-indexed region
 *     holds deliberately different values,
 *   - fm_rms_norm_nx's row stride is n, which only shows up when the rows
 *     hold different data.
 *
 * Helper functions are named case_* so lib/test.h's failure line, which
 * prints __func__ + 5, still reads as the case that failed.
 *
 * Exits 77 when no Metal device is present so GPU-less CI skips cleanly.
 */

#include "lib/test.h"
#include "metal/fm_metal.h"
#include "metal/tests/fm_golden_vectors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* qwen35moe attention shape the golden set was generated for. */
#define QKNORM_HEAD_DIM  256
#define QKNORM_HEADS      16
#define NORM_EPS        1e-6f

/* The GPU sums squares in float32 where the reference uses double, so some
 * drift is expected; measured on M4 / macOS 26 it stays sub-ULP at these
 * widths, well inside this band and two orders tighter than the 1e-4..1e-3
 * the golden set allows for Kepler. */
#define NORM_TOL_ABS    1e-6f
#define NORM_TOL_REL    1e-6f

static uint32_t norm_rng = 0x9e3779b9u;

static float next_random_float(void) {
    norm_rng ^= norm_rng << 13;
    norm_rng ^= norm_rng >> 17;
    norm_rng ^= norm_rng << 5;
    return (float)((int32_t)norm_rng) / 2147483648.0f;
}

/* CPU reference: out = x * rsqrt(mean(x^2) + eps) * weight, sum of squares
 * accumulated in double exactly as pn_vrms_norm_f32 does. weight NULL means
 * no gain. */
static void reference_rms_norm(const float *x, const float *weight, float *out,
                               int n, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++)
        sum_sq += (double)x[i] * (double)x[i];
    float inv_rms = (float)(1.0 / sqrt(sum_sq / (double)n + (double)eps));
    for (int i = 0; i < n; i++)
        out[i] = x[i] * inv_rms * (weight ? weight[i] : 1.0f);
}

static fm_buffer *upload_floats(const float *src, size_t n) {
    fm_buffer *buf = fm_buffer_alloc(n * sizeof(float));
    if (buf) memcpy(fm_buffer_data(buf), src, n * sizeof(float));
    return buf;
}

/* Diffs against the reference. `label` is printed just before the assertion
 * fires, because lib/test.h names the failing function and every case here
 * shares this one — without it a failure would not say which. */
static void case_compare(const char *label, const float *expected,
                         const float *got, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float tol = NORM_TOL_ABS + NORM_TOL_REL * fabsf(expected[i]);
        if (fabsf(expected[i] - got[i]) > tol)
            printf("    %s: element %zu of %zu\n", label, i, n);
        ASSERT_FLT_NEAR(expected[i], got[i], tol);
    }
}

/* ================================================================
 * fm_qk_rms_norm
 * ================================================================ */

static void test_qk_rms_norm_matches_golden(void) {
    size_t count = (size_t)QKNORM_HEADS * QKNORM_HEAD_DIM;
    float *input = fm_golden_load_f32("qknorm_input.bin", count);
    float *weight = fm_golden_load_f32("qknorm_weight.bin", QKNORM_HEAD_DIM);
    float *golden = fm_golden_load_f32("qknorm_output.bin", count);
    ASSERT_NOT_NULL(input);
    ASSERT_NOT_NULL(weight);
    ASSERT_NOT_NULL(golden);

    fm_buffer *bx = upload_floats(input, count);
    fm_buffer *bw = upload_floats(weight, QKNORM_HEAD_DIM);
    fm_buffer *bo = fm_buffer_alloc(count * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bw);
    ASSERT_NOT_NULL(bo);

    fm_qk_rms_norm(bx, bw, bo, QKNORM_HEADS, QKNORM_HEAD_DIM, NORM_EPS);
    fm_sync();

    case_compare("qk_rms_norm vs qknorm_output.bin",
                 golden, (const float *)fm_buffer_data(bo), count);

    fm_buffer_free(bx);
    fm_buffer_free(bw);
    fm_buffer_free(bo);
    free(input);
    free(weight);
    free(golden);
}

/* THE SHARED-WEIGHT TRAP. The weight buffer is n_heads * head_dim long: the
 * first head_dim floats are the real shared weight, and the rest — exactly
 * the region a per-head-indexing kernel would read for heads 1..3 — hold
 * very different values. So `weight[head * head_dim + i]` produces a
 * specific, deterministic wrong answer instead of reading past the buffer
 * and maybe getting away with it. A uniform weight would make both
 * indexings agree and the test would prove nothing. */
static void test_qk_rms_norm_weight_is_shared_across_heads(void) {
    const int heads = 4, head_dim = 64;
    const size_t count = (size_t)heads * head_dim;
    float input[4 * 64], weight[4 * 64], expected[4 * 64];

    for (size_t i = 0; i < count; i++)
        input[i] = next_random_float() * 3.0f;
    for (int i = 0; i < head_dim; i++)
        weight[i] = 0.5f + 0.01f * (float)i;          /* the shared weight */
    for (size_t i = head_dim; i < count; i++)
        weight[i] = -4.0f - 0.02f * (float)i;         /* never to be read */

    for (int h = 0; h < heads; h++)
        reference_rms_norm(input + (size_t)h * head_dim, weight,
                           expected + (size_t)h * head_dim, head_dim, NORM_EPS);

    fm_buffer *bx = upload_floats(input, count);
    fm_buffer *bw = upload_floats(weight, count);
    fm_buffer *bo = fm_buffer_alloc(count * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bw);
    ASSERT_NOT_NULL(bo);

    fm_qk_rms_norm(bx, bw, bo, heads, head_dim, NORM_EPS);
    fm_sync();

    case_compare("qk_rms_norm with a shared non-uniform weight",
                 expected, (const float *)fm_buffer_data(bo), count);

    fm_buffer_free(bx);
    fm_buffer_free(bw);
    fm_buffer_free(bo);
}

static void test_qk_rms_norm_honors_null_weight(void) {
    const int heads = 3, head_dim = 128;
    const size_t count = (size_t)heads * head_dim;
    float input[3 * 128], expected[3 * 128];

    for (size_t i = 0; i < count; i++)
        input[i] = next_random_float() * 2.0f;
    for (int h = 0; h < heads; h++)
        reference_rms_norm(input + (size_t)h * head_dim, NULL,
                           expected + (size_t)h * head_dim, head_dim, NORM_EPS);

    fm_buffer *bx = upload_floats(input, count);
    fm_buffer *bo = fm_buffer_alloc(count * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bo);

    fm_qk_rms_norm(bx, NULL, bo, heads, head_dim, NORM_EPS);
    fm_sync();

    case_compare("qk_rms_norm with weight == NULL",
                 expected, (const float *)fm_buffer_data(bo), count);

    fm_buffer_free(bx);
    fm_buffer_free(bo);
}

/* ================================================================
 * fm_rms_norm_nx / fm_rms_norm
 * ================================================================ */

/* Rows hold independent random data and n is neither a multiple of 4 nor of
 * the 32-lane SIMD width, so a wrong row stride, a dropped tail element or a
 * reduction that spans rows all change the result. */
static void test_rms_norm_nx_matches_cpu_reference(void) {
    const int rows = 5, n = 300;
    const size_t count = (size_t)rows * n;
    float input[5 * 300], weight[300], expected[5 * 300];

    for (size_t i = 0; i < count; i++)
        input[i] = next_random_float() * (1.0f + (float)(i % 7));
    for (int i = 0; i < n; i++)
        weight[i] = 0.75f + 0.002f * (float)i;

    for (int r = 0; r < rows; r++)
        reference_rms_norm(input + (size_t)r * n, weight,
                           expected + (size_t)r * n, n, NORM_EPS);

    fm_buffer *bx = upload_floats(input, count);
    fm_buffer *bw = upload_floats(weight, n);
    fm_buffer *bo = fm_buffer_alloc(count * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bw);
    ASSERT_NOT_NULL(bo);

    fm_rms_norm_nx(bx, bw, bo, rows, n, NORM_EPS);
    fm_sync();

    case_compare("rms_norm_nx rows=5 n=300",
                 expected, (const float *)fm_buffer_data(bo), count);

    fm_buffer_free(bx);
    fm_buffer_free(bw);
    fm_buffer_free(bo);
}

static void test_rms_norm_nx_honors_null_weight(void) {
    const int rows = 3, n = 129;
    const size_t count = (size_t)rows * n;
    float input[3 * 129], expected[3 * 129];

    for (size_t i = 0; i < count; i++)
        input[i] = next_random_float() * 5.0f;
    for (int r = 0; r < rows; r++)
        reference_rms_norm(input + (size_t)r * n, NULL,
                           expected + (size_t)r * n, n, NORM_EPS);

    fm_buffer *bx = upload_floats(input, count);
    fm_buffer *bo = fm_buffer_alloc(count * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bo);

    fm_rms_norm_nx(bx, NULL, bo, rows, n, NORM_EPS);
    fm_sync();

    case_compare("rms_norm_nx with weight == NULL",
                 expected, (const float *)fm_buffer_data(bo), count);

    fm_buffer_free(bx);
    fm_buffer_free(bo);
}

/* The single-vector entry point is the rows == 1 case of the same kernel;
 * this pins its behaviour to the CPU reference independently. */
static void test_rms_norm_single_vector_matches_cpu_reference(void) {
    const int n = 2048;
    float *input = (float *)malloc(n * sizeof(float));
    float *weight = (float *)malloc(n * sizeof(float));
    float *expected = (float *)malloc(n * sizeof(float));
    ASSERT_NOT_NULL(input);
    ASSERT_NOT_NULL(weight);
    ASSERT_NOT_NULL(expected);

    for (int i = 0; i < n; i++) {
        input[i] = next_random_float() * 4.0f;
        weight[i] = 1.0f + 0.1f * next_random_float();
    }
    reference_rms_norm(input, weight, expected, n, NORM_EPS);

    fm_buffer *bx = upload_floats(input, n);
    fm_buffer *bw = upload_floats(weight, n);
    fm_buffer *bo = fm_buffer_alloc(n * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bw);
    ASSERT_NOT_NULL(bo);

    fm_rms_norm(bx, bw, bo, n, NORM_EPS);
    fm_sync();

    case_compare("rms_norm single vector n=2048",
                 expected, (const float *)fm_buffer_data(bo), (size_t)n);

    fm_buffer_free(bx);
    fm_buffer_free(bw);
    fm_buffer_free(bo);
    free(input);
    free(weight);
    free(expected);
}

int main(int argc, char **argv) {
    (void)argc;
    fm_golden_locate(argv[0]);

    printf("\n  Metal Normalization Kernel Tests\n");
    printf("  ================================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "  Metal unavailable — skipping.\n\n");
        return 77;
    }

    RUN_TEST(qk_rms_norm_matches_golden);
    RUN_TEST(qk_rms_norm_weight_is_shared_across_heads);
    RUN_TEST(qk_rms_norm_honors_null_weight);
    RUN_TEST(rms_norm_nx_matches_cpu_reference);
    RUN_TEST(rms_norm_nx_honors_null_weight);
    RUN_TEST(rms_norm_single_vector_matches_cpu_reference);

    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
