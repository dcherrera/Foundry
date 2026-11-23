/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_metal_gemm_batched.c
 * @brief Tests fm_sgemm_batched against a plain CPU reference GEMM: several
 *   batch counts, non-square shapes, padded (non-packed) batch strides,
 *   alpha/beta, and rejection of invalid strides.
 *
 * Exits 77 when no Metal device is present so GPU-less CI skips cleanly.
 *
 * @status All tests pass on M4 / macOS 26; each was proven able to fail.
 * @issues None
 * @todo None
 */

#include "lib/test.h"
#include "metal/fm_metal.h"

#include <math.h>
#include <stdio.h>

/* Deterministic, well-conditioned filler — distinct per (matrix, element)
 * so a batch element read at the wrong stride cannot coincidentally match. */
static float sample_value(int seed, int i)
{
    return (float)(((seed * 31 + i * 17) % 23) - 11) * 0.25f;
}

/* Plain row-major reference: C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C. */
static void cpu_gemm(const float *a, const float *b, float *c,
                     int M, int N, int K, float alpha, float beta)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += a[m * K + k] * b[k * N + n];
            c[m * N + n] = alpha * sum + beta * c[m * N + n];
        }
    }
}

/* One batched-GEMM case: fills A and B, runs the GPU batch and the CPU
 * reference over every batch element, and compares.  `pad_matrices`
 * inserts one matrix worth of untouched padding between batch elements,
 * exercising a non-packed stride. */
static void test_batched_case_against_cpu(int batch, int M, int N, int K,
                                          float alpha, float beta, bool pad_matrices)
{
    size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;
    int span = pad_matrices ? 2 : 1;
    size_t a_stride = a_elems * span * sizeof(float);
    size_t b_stride = b_elems * span * sizeof(float);
    size_t c_stride = c_elems * span * sizeof(float);

    fm_buffer *ba = fm_buffer_alloc(a_stride * (size_t)batch);
    fm_buffer *bb = fm_buffer_alloc(b_stride * (size_t)batch);
    fm_buffer *bc = fm_buffer_alloc(c_stride * (size_t)batch);
    ASSERT_NOT_NULL(ba);
    ASSERT_NOT_NULL(bb);
    ASSERT_NOT_NULL(bc);

    float *a = (float *)fm_buffer_data(ba);
    float *b = (float *)fm_buffer_data(bb);
    float *c = (float *)fm_buffer_data(bc);

    for (int e = 0; e < batch; e++) {
        float *ae = a + (size_t)e * a_elems * span;
        float *be = b + (size_t)e * b_elems * span;
        float *ce = c + (size_t)e * c_elems * span;
        for (size_t i = 0; i < a_elems; i++) ae[i] = sample_value(e + 1, (int)i);
        for (size_t i = 0; i < b_elems; i++) be[i] = sample_value(e + 101, (int)i);
        for (size_t i = 0; i < c_elems; i++) ce[i] = sample_value(e + 201, (int)i);
    }

    /* CPU reference over a private copy of C, before the GPU touches it. */
    float *expected = (float *)malloc(c_elems * (size_t)batch * sizeof(float));
    ASSERT_NOT_NULL(expected);
    for (int e = 0; e < batch; e++) {
        float *ce = expected + (size_t)e * c_elems;
        memcpy(ce, c + (size_t)e * c_elems * span, c_elems * sizeof(float));
        cpu_gemm(a + (size_t)e * a_elems * span,
                 b + (size_t)e * b_elems * span,
                 ce, M, N, K, alpha, beta);
    }

    fm_sgemm_batched(ba, bb, bc, batch, M, N, K,
                     pad_matrices ? a_stride : 0,
                     pad_matrices ? b_stride : 0,
                     pad_matrices ? c_stride : 0,
                     alpha, beta);
    fm_sync();

    float *result = (float *)fm_buffer_data(bc);
    for (int e = 0; e < batch; e++) {
        const float *got  = result + (size_t)e * c_elems * span;
        const float *want = expected + (size_t)e * c_elems;
        for (size_t i = 0; i < c_elems; i++) {
            float tol = fabsf(want[i]) * 1e-4f + 1e-4f;
            if (fabsf(got[i] - want[i]) > tol)
                printf("\n    batch %d elem %zu: got %.6f want %.6f\n",
                       e, i, got[i], want[i]);
            ASSERT_TRUE(fabsf(got[i] - want[i]) <= tol);
        }
    }

    free(expected);
    fm_buffer_free(ba);
    fm_buffer_free(bb);
    fm_buffer_free(bc);
}

/* ================================================================
 * Correctness vs the CPU reference
 * ================================================================ */

static void test_batch_of_one_matches_reference(void)
{
    test_batched_case_against_cpu(1, 3, 5, 4, 1.0f, 0.0f, false);
}

static void test_batch_of_four_non_square(void)
{
    /* M != N != K, so a transposed or shape-confused encode cannot pass. */
    test_batched_case_against_cpu(4, 3, 5, 4, 1.0f, 0.0f, false);
}

static void test_batch_of_seventeen_non_square(void)
{
    test_batched_case_against_cpu(17, 6, 2, 9, 1.0f, 0.0f, false);
}

static void test_batch_honours_alpha_and_beta(void)
{
    test_batched_case_against_cpu(3, 4, 3, 5, 2.0f, -0.5f, false);
}

static void test_batch_with_padded_strides(void)
{
    /* Twice-packed strides: each batch element is followed by a matrix
     * worth of padding that must be skipped, not consumed. */
    test_batched_case_against_cpu(5, 3, 5, 4, 1.0f, 0.0f, true);
}

static void test_larger_batch_matches_reference(void)
{
    test_batched_case_against_cpu(8, 32, 24, 40, 1.0f, 0.0f, false);
}

/* ================================================================
 * Agreement with the single-matrix path
 * ================================================================ */

static void test_batch_element_matches_fm_sgemm(void)
{
    int batch = 3, M = 4, N = 6, K = 5;
    size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;

    fm_buffer *ba = fm_buffer_alloc(a_elems * (size_t)batch * sizeof(float));
    fm_buffer *bb = fm_buffer_alloc(b_elems * (size_t)batch * sizeof(float));
    fm_buffer *bc = fm_buffer_alloc(c_elems * (size_t)batch * sizeof(float));
    ASSERT_NOT_NULL(ba);
    ASSERT_NOT_NULL(bb);
    ASSERT_NOT_NULL(bc);

    float *a = (float *)fm_buffer_data(ba);
    float *b = (float *)fm_buffer_data(bb);
    for (int e = 0; e < batch; e++) {
        for (size_t i = 0; i < a_elems; i++) a[(size_t)e * a_elems + i] = sample_value(e + 3, (int)i);
        for (size_t i = 0; i < b_elems; i++) b[(size_t)e * b_elems + i] = sample_value(e + 53, (int)i);
    }

    fm_sgemm_batched(ba, bb, bc, batch, M, N, K, 0, 0, 0, 1.0f, 0.0f);
    fm_sync();

    /* Re-run the last batch element on its own through the single-matrix
     * path and require the same numbers. */
    fm_buffer *single_a = fm_buffer_alloc(a_elems * sizeof(float));
    fm_buffer *single_b = fm_buffer_alloc(b_elems * sizeof(float));
    fm_buffer *single_c = fm_buffer_alloc(c_elems * sizeof(float));
    ASSERT_NOT_NULL(single_a);
    ASSERT_NOT_NULL(single_b);
    ASSERT_NOT_NULL(single_c);

    int last = batch - 1;
    memcpy(fm_buffer_data(single_a), a + (size_t)last * a_elems, a_elems * sizeof(float));
    memcpy(fm_buffer_data(single_b), b + (size_t)last * b_elems, b_elems * sizeof(float));
    fm_sgemm(single_a, single_b, single_c, M, N, K, 1.0f, 0.0f);
    fm_sync();

    const float *batched = (const float *)fm_buffer_data(bc) + (size_t)last * c_elems;
    const float *single  = (const float *)fm_buffer_data(single_c);
    for (size_t i = 0; i < c_elems; i++)
        ASSERT_FLT_NEAR(single[i], batched[i], 1e-4f);

    fm_buffer_free(ba);
    fm_buffer_free(bb);
    fm_buffer_free(bc);
    fm_buffer_free(single_a);
    fm_buffer_free(single_b);
    fm_buffer_free(single_c);
}

/* ================================================================
 * Argument validation
 * ================================================================ */

static void test_invalid_arguments_leave_output_untouched(void)
{
    int batch = 2, M = 2, N = 2, K = 2;
    size_t c_elems = (size_t)M * N;

    fm_buffer *ba = fm_buffer_alloc((size_t)M * K * batch * sizeof(float));
    fm_buffer *bb = fm_buffer_alloc((size_t)K * N * batch * sizeof(float));
    fm_buffer *bc = fm_buffer_alloc(c_elems * batch * sizeof(float));
    ASSERT_NOT_NULL(ba);
    ASSERT_NOT_NULL(bb);
    ASSERT_NOT_NULL(bc);

    float *c = (float *)fm_buffer_data(bc);
    for (size_t i = 0; i < c_elems * (size_t)batch; i++) c[i] = -7.0f;

    /* A stride below the packed size, a stride that is not a multiple of
     * the row bytes, a zero batch, and a NULL operand must all be
     * rejected before any dispatch. */
    fm_sgemm_batched(ba, bb, bc, batch, M, N, K, sizeof(float), 0, 0, 1.0f, 0.0f);
    fm_sgemm_batched(ba, bb, bc, batch, M, N, K, 0, 0,
                     c_elems * sizeof(float) + 1, 1.0f, 0.0f);
    fm_sgemm_batched(ba, bb, bc, 0, M, N, K, 0, 0, 0, 1.0f, 0.0f);
    fm_sgemm_batched(NULL, bb, bc, batch, M, N, K, 0, 0, 0, 1.0f, 0.0f);
    fm_sync();

    const float *result = (const float *)fm_buffer_data(bc);
    for (size_t i = 0; i < c_elems * (size_t)batch; i++)
        ASSERT_FLT_NEAR(-7.0f, result[i], 1e-6f);

    fm_buffer_free(ba);
    fm_buffer_free(bb);
    fm_buffer_free(bc);
}

/* ================================================================
 * main
 * ================================================================ */

int main(void)
{
    printf("\n  Metal Batched GEMM Tests\n");
    printf("  ========================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "Metal unavailable\n");
        return 77;
    }

    RUN_TEST(batch_of_one_matches_reference);
    RUN_TEST(batch_of_four_non_square);
    RUN_TEST(batch_of_seventeen_non_square);
    RUN_TEST(batch_honours_alpha_and_beta);
    RUN_TEST(batch_with_padded_strides);
    RUN_TEST(larger_batch_matches_reference);
    RUN_TEST(batch_element_matches_fm_sgemm);
    RUN_TEST(invalid_arguments_leave_output_untouched);

    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
