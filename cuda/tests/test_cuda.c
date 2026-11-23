/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_cuda.c — Unit tests for CUDA GPU backend
 *
 * Tests run on the GPU server with CUDA available.
 * If CUDA is unavailable, tests gracefully skip.
 *
 * Build: make USE_CUDA=1 test-cuda
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Lifecycle tests
 * ================================================================ */

static void test_init_shutdown(void) {
    int ret = fc_init();
    if (ret != 0) {
        printf("    [SKIP] CUDA not available\n");
        return;
    }
    ASSERT_TRUE(fc_available());
    fc_free();
    ASSERT_FALSE(fc_available());

    /* Re-init should work (idempotent pattern) */
    ret = fc_init();
    ASSERT_TRUE(ret == 0);
    ASSERT_TRUE(fc_available());
    fc_free();
}

static void test_double_init(void) {
    int ret = fc_init();
    if (ret != 0) { printf("    [SKIP] CUDA not available\n"); return; }

    /* Second init should be idempotent */
    int ret2 = fc_init();
    ASSERT_TRUE(ret2 == 0);
    ASSERT_TRUE(fc_available());

    fc_free();
}

/* ================================================================
 * Buffer tests
 * ================================================================ */

static void test_buffer_roundtrip(void) {
    if (fc_init() != 0) { printf("    [SKIP]\n"); return; }

    float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    size_t bytes = sizeof(data);

    fc_buffer *buf = fc_buffer_from_ptr(data, bytes);
    ASSERT_NOT_NULL(buf);
    ASSERT_TRUE(fc_buffer_size(buf) == bytes);

    /* Read back */
    float result[5] = {0};
    fc_buffer_to_cpu(buf, result);

    for (int i = 0; i < 5; i++) {
        ASSERT_TRUE(fabsf(result[i] - data[i]) < 1e-6f);
    }

    fc_buffer_free(buf);
    fc_free();
}

static void test_buffer_alloc_and_write(void) {
    if (fc_init() != 0) { printf("    [SKIP]\n"); return; }

    fc_buffer *buf = fc_buffer_alloc(1024 * sizeof(float));
    ASSERT_NOT_NULL(buf);
    ASSERT_TRUE(fc_buffer_size(buf) == 1024 * sizeof(float));

    /* Write data to it */
    float *data = (float*)malloc(1024 * sizeof(float));
    for (int i = 0; i < 1024; i++) data[i] = (float)i;
    fc_buffer_from_cpu(buf, data);

    /* Read back */
    float *result = (float*)malloc(1024 * sizeof(float));
    fc_buffer_to_cpu(buf, result);

    for (int i = 0; i < 1024; i++) {
        ASSERT_TRUE(fabsf(result[i] - (float)i) < 1e-6f);
    }

    free(data);
    free(result);
    fc_buffer_free(buf);
    fc_free();
}

static void test_buffer_large(void) {
    if (fc_init() != 0) { printf("    [SKIP]\n"); return; }

    /* 1 MB buffer */
    size_t n = 256 * 1024;
    float *data = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) data[i] = (float)(i % 1000) * 0.001f;

    fc_buffer *buf = fc_buffer_from_ptr(data, n * sizeof(float));
    ASSERT_NOT_NULL(buf);

    float *result = (float*)malloc(n * sizeof(float));
    fc_buffer_to_cpu(buf, result);

    /* Spot check */
    ASSERT_TRUE(fabsf(result[0] - data[0]) < 1e-5f);
    ASSERT_TRUE(fabsf(result[1000] - data[1000]) < 1e-5f);
    ASSERT_TRUE(fabsf(result[n - 1] - data[n - 1]) < 1e-5f);

    free(data);
    free(result);
    fc_buffer_free(buf);
    fc_free();
}

/* ================================================================
 * cuBLAS SGEMM tests
 * ================================================================ */

static void test_sgemm_identity(void) {
    if (fc_init() != 0) { printf("    [SKIP]\n"); return; }

    /* C[2,2] = A[2,3] @ B[3,2] */
    float A[] = { 1, 2, 3,
                  4, 5, 6 };
    float B[] = { 7, 8,
                  9, 10,
                  11, 12 };
    float C[4] = {0};

    /* Expected: C = [[58, 64], [139, 154]] */
    float expected[] = { 58, 64, 139, 154 };

    fc_buffer *ba = fc_buffer_from_ptr(A, sizeof(A));
    fc_buffer *bb = fc_buffer_from_ptr(B, sizeof(B));
    fc_buffer *bc = fc_buffer_alloc(sizeof(C));
    ASSERT_NOT_NULL(ba);
    ASSERT_NOT_NULL(bb);
    ASSERT_NOT_NULL(bc);

    fc_sgemm(ba, bb, bc, 2, 2, 3, 1.0f, 0.0f);
    fc_sync();
    fc_buffer_to_cpu(bc, C);

    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(fabsf(C[i] - expected[i]) < 1e-3f);
    }

    fc_buffer_free(ba);
    fc_buffer_free(bb);
    fc_buffer_free(bc);
    fc_free();
}

static void test_sgemm_larger(void) {
    if (fc_init() != 0) { printf("    [SKIP]\n"); return; }

    /* 64x64 matmul */
    int M = 64, K = 64, N = 64;
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C_gpu = (float*)calloc(M * N, sizeof(float));
    float *C_cpu = (float*)calloc(M * N, sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 11) * 0.1f;

    /* CPU reference */
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += A[m * K + k] * B[k * N + n];
            C_cpu[m * N + n] = sum;
        }

    /* GPU */
    fc_buffer *ba = fc_buffer_from_ptr(A, M * K * sizeof(float));
    fc_buffer *bb = fc_buffer_from_ptr(B, K * N * sizeof(float));
    fc_buffer *bc = fc_buffer_alloc(M * N * sizeof(float));
    fc_sgemm(ba, bb, bc, M, N, K, 1.0f, 0.0f);
    fc_sync();
    fc_buffer_to_cpu(bc, C_gpu);

    /* Compare */
    float max_err = 0;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(C_gpu[i] - C_cpu[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err < 0.01f);

    free(A); free(B); free(C_gpu); free(C_cpu);
    fc_buffer_free(ba);
    fc_buffer_free(bb);
    fc_buffer_free(bc);
    fc_free();
}

/* ================================================================
 * cuBLAS SGEMV tests
 * ================================================================ */

static void test_sgemv(void) {
    if (fc_init() != 0) { printf("    [SKIP]\n"); return; }

    /* y[3] = A[3,4] @ x[4] */
    float A[] = { 1, 2, 3, 4,
                  5, 6, 7, 8,
                  9, 10, 11, 12 };
    float x[] = { 1, 2, 3, 4 };
    float y[3] = {0};

    /* Expected: [30, 70, 110] */
    float expected[] = { 30.0f, 70.0f, 110.0f };

    fc_buffer *ba = fc_buffer_from_ptr(A, sizeof(A));
    fc_buffer *bx = fc_buffer_from_ptr(x, sizeof(x));
    fc_buffer *by = fc_buffer_alloc(sizeof(y));

    fc_sgemv(ba, bx, by, 3, 4, 1.0f, 0.0f);
    fc_sync();
    fc_buffer_to_cpu(by, y);

    for (int i = 0; i < 3; i++) {
        ASSERT_TRUE(fabsf(y[i] - expected[i]) < 1e-3f);
    }

    fc_buffer_free(ba);
    fc_buffer_free(bx);
    fc_buffer_free(by);
    fc_free();
}

/* ================================================================
 * Custom kernel tests
 * ================================================================ */

static void test_add(void) {
    if (fc_init() != 0) { printf("    [SKIP]\n"); return; }

    int n = 1024;
    float *a = (float*)malloc(n * sizeof(float));
    float *b = (float*)malloc(n * sizeof(float));
    float *out = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)(n - i);
    }

    fc_buffer *ba = fc_buffer_from_ptr(a, n * sizeof(float));
    fc_buffer *bb = fc_buffer_from_ptr(b, n * sizeof(float));
    fc_buffer *bo = fc_buffer_alloc(n * sizeof(float));

    fc_add(ba, bb, bo, n);
    fc_sync();
    fc_buffer_to_cpu(bo, out);

    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(fabsf(out[i] - (float)n) < 1e-5f);
    }

    free(a); free(b); free(out);
    fc_buffer_free(ba);
    fc_buffer_free(bb);
    fc_buffer_free(bo);
    fc_free();
}

static void test_silu(void) {
    if (fc_init() != 0) { printf("    [SKIP]\n"); return; }

    int n = 512;
    float *x = (float*)malloc(n * sizeof(float));
    float *out_gpu = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        x[i] = (float)(i - n / 2) * 0.01f;
    }

    fc_buffer *bx = fc_buffer_from_ptr(x, n * sizeof(float));
    fc_buffer *bo = fc_buffer_alloc(n * sizeof(float));

    fc_silu(bx, bo, n);
    fc_sync();
    fc_buffer_to_cpu(bo, out_gpu);

    /* Compare with CPU reference */
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float expected = v / (1.0f + expf(-v));
        float err = fabsf(out_gpu[i] - expected);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err < 1e-5f);

    free(x); free(out_gpu);
    fc_buffer_free(bx);
    fc_buffer_free(bo);
    fc_free();
}

static void test_rms_norm(void) {
    if (fc_init() != 0) { printf("    [SKIP]\n"); return; }

    int n = 768;  /* typical hidden dim */
    float *x = (float*)malloc(n * sizeof(float));
    float *w = (float*)malloc(n * sizeof(float));
    float *out_gpu = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        x[i] = (float)(i % 17) * 0.1f - 0.8f;
        w[i] = 1.0f + (float)(i % 5) * 0.01f;
    }

    fc_buffer *bx = fc_buffer_from_ptr(x, n * sizeof(float));
    fc_buffer *bw = fc_buffer_from_ptr(w, n * sizeof(float));
    fc_buffer *bo = fc_buffer_alloc(n * sizeof(float));

    float eps = 1e-5f;
    fc_rms_norm(bx, bw, bo, n, eps);
    fc_sync();
    fc_buffer_to_cpu(bo, out_gpu);

    /* CPU reference */
    float sum_sq = 0;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float rms = sqrtf(sum_sq / n + eps);

    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float expected = (x[i] / rms) * w[i];
        float err = fabsf(out_gpu[i] - expected);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err < 1e-4f);

    free(x); free(w); free(out_gpu);
    fc_buffer_free(bx);
    fc_buffer_free(bw);
    fc_buffer_free(bo);
    fc_free();
}

static void test_softmax(void) {
    if (fc_init() != 0) { printf("    [SKIP]\n"); return; }

    int rows = 4, cols = 256;
    float *x = (float*)malloc(rows * cols * sizeof(float));
    float *out_gpu = (float*)malloc(rows * cols * sizeof(float));

    for (int i = 0; i < rows * cols; i++) {
        x[i] = (float)(i % 31) * 0.1f - 1.5f;
    }

    fc_buffer *bx = fc_buffer_from_ptr(x, rows * cols * sizeof(float));
    fc_buffer *bo = fc_buffer_alloc(rows * cols * sizeof(float));

    fc_softmax(bx, bo, rows, cols);
    fc_sync();
    fc_buffer_to_cpu(bo, out_gpu);

    /* Check each row sums to 1 and all values are in [0, 1] */
    for (int r = 0; r < rows; r++) {
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            float v = out_gpu[r * cols + c];
            ASSERT_TRUE(v >= 0.0f && v <= 1.0f);
            sum += v;
        }
        ASSERT_TRUE(fabsf(sum - 1.0f) < 1e-4f);
    }

    free(x); free(out_gpu);
    fc_buffer_free(bx);
    fc_buffer_free(bo);
    fc_free();
}

static void test_matvec_q8(void) {
    if (fc_init() != 0) { printf("    [SKIP]\n"); return; }

    int rows = 64, cols = 128;

    /* Create quantized weight matrix (int8) and scales */
    signed char *w = (signed char*)malloc(rows * cols);
    float *w_scales = (float*)malloc(rows * sizeof(float));
    for (int i = 0; i < rows; i++) {
        w_scales[i] = 0.01f * (1.0f + (float)(i % 3));
        for (int j = 0; j < cols; j++) {
            w[i * cols + j] = (signed char)((i + j) % 127);
        }
    }

    /* Create quantized input vector */
    signed char *x_q = (signed char*)malloc(cols);
    float x_scale = 0.02f;
    for (int j = 0; j < cols; j++) {
        x_q[j] = (signed char)(j % 64 - 32);
    }

    float *out_gpu = (float*)malloc(rows * sizeof(float));

    fc_buffer *bxq = fc_buffer_from_ptr(x_q, cols);
    fc_buffer *bw = fc_buffer_from_ptr(w, rows * cols);
    fc_buffer *bws = fc_buffer_from_ptr(w_scales, rows * sizeof(float));
    fc_buffer *bo = fc_buffer_alloc(rows * sizeof(float));

    fc_matvec_q8(bxq, bw, bws, bo, x_scale, rows, cols);
    fc_sync();
    fc_buffer_to_cpu(bo, out_gpu);

    /* CPU reference */
    float max_err = 0;
    for (int i = 0; i < rows; i++) {
        int dot = 0;
        for (int j = 0; j < cols; j++) {
            dot += (int)x_q[j] * (int)w[i * cols + j];
        }
        float expected = x_scale * w_scales[i] * (float)dot;
        float err = fabsf(out_gpu[i] - expected);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err < 0.01f);

    free(w); free(w_scales); free(x_q); free(out_gpu);
    fc_buffer_free(bxq);
    fc_buffer_free(bw);
    fc_buffer_free(bws);
    fc_buffer_free(bo);
    fc_free();
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("CUDA Backend Tests\n");
    printf("==================\n");

    /* Lifecycle */
    RUN_TEST(init_shutdown);
    RUN_TEST(double_init);

    /* Buffers */
    RUN_TEST(buffer_roundtrip);
    RUN_TEST(buffer_alloc_and_write);
    RUN_TEST(buffer_large);

    /* cuBLAS */
    RUN_TEST(sgemm_identity);
    RUN_TEST(sgemm_larger);
    RUN_TEST(sgemv);

    /* Custom kernels */
    RUN_TEST(add);
    RUN_TEST(silu);
    RUN_TEST(rms_norm);
    RUN_TEST(softmax);
    RUN_TEST(matvec_q8);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
