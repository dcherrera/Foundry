/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_metal.c — Tests for the Metal GPU compute backend
 *
 * Tests lifecycle, buffers, MPS GEMM/GEMV, and custom compute kernels.
 */

#include "lib/test.h"
#include "metal/fm_metal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Phase 1: Infrastructure + MPS GEMM
 * ================================================================ */

static void test_init_and_available(void) {
    int ret = fm_init();
    ASSERT_INT_EQ(0, ret);
    ASSERT_TRUE(fm_available());
}

static void test_double_init_is_safe(void) {
    int ret = fm_init();
    ASSERT_INT_EQ(0, ret);
    /* Second init should succeed (no-op) */
    ret = fm_init();
    ASSERT_INT_EQ(0, ret);
    ASSERT_TRUE(fm_available());
}

static void test_buffer_alloc(void) {
    fm_buffer *buf = fm_buffer_alloc(1024);
    ASSERT_NOT_NULL(buf);
    ASSERT_NOT_NULL(fm_buffer_data(buf));
    ASSERT_INT_EQ(1024, (int)fm_buffer_size(buf));
    fm_buffer_free(buf);
}

static void test_buffer_from_ptr(void) {
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    fm_buffer *buf = fm_buffer_from_ptr(data, sizeof(data));
    ASSERT_NOT_NULL(buf);
    ASSERT_INT_EQ(sizeof(data), (int)fm_buffer_size(buf));
    fm_buffer_free(buf);
}

static void test_buffer_alloc_write_read(void) {
    fm_buffer *buf = fm_buffer_alloc(4 * sizeof(float));
    ASSERT_NOT_NULL(buf);

    float *ptr = (float*)fm_buffer_data(buf);
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;
    ptr[3] = 4.0f;

    /* Read back */
    ASSERT_FLT_NEAR(1.0f, ptr[0], 1e-6f);
    ASSERT_FLT_NEAR(2.0f, ptr[1], 1e-6f);
    ASSERT_FLT_NEAR(3.0f, ptr[2], 1e-6f);
    ASSERT_FLT_NEAR(4.0f, ptr[3], 1e-6f);

    fm_buffer_free(buf);
}

static void test_sgemm_2x2(void) {
    /* A = [[1,2],[3,4]], B = [[5,6],[7,8]] => C = [[19,22],[43,50]] */
    float a[4] = {1, 2, 3, 4};
    float b[4] = {5, 6, 7, 8};
    float c[4] = {0, 0, 0, 0};

    fm_buffer *ba = fm_buffer_from_ptr(a, sizeof(a));
    fm_buffer *bb = fm_buffer_from_ptr(b, sizeof(b));
    fm_buffer *bc = fm_buffer_from_ptr(c, sizeof(c));

    fm_sgemm(ba, bb, bc, 2, 2, 2, 1.0f, 0.0f);

    /* Read result from Metal buffer */
    float *result = (float*)fm_buffer_data(bc);
    ASSERT_FLT_NEAR(19.0f, result[0], 0.01f);
    ASSERT_FLT_NEAR(22.0f, result[1], 0.01f);
    ASSERT_FLT_NEAR(43.0f, result[2], 0.01f);
    ASSERT_FLT_NEAR(50.0f, result[3], 0.01f);

    fm_buffer_free(ba);
    fm_buffer_free(bb);
    fm_buffer_free(bc);
}

static void test_sgemm_alpha_beta(void) {
    /* C = 2*A@B + 3*C with A=B=I(2), C=[[1,0],[0,1]] => C = [[5,0],[0,5]] */
    float a[4] = {1, 0, 0, 1};
    float b[4] = {1, 0, 0, 1};
    float c[4] = {1, 0, 0, 1};

    fm_buffer *ba = fm_buffer_from_ptr(a, sizeof(a));
    fm_buffer *bb = fm_buffer_from_ptr(b, sizeof(b));
    fm_buffer *bc = fm_buffer_from_ptr(c, sizeof(c));

    fm_sgemm(ba, bb, bc, 2, 2, 2, 2.0f, 3.0f);

    float *result = (float*)fm_buffer_data(bc);
    ASSERT_FLT_NEAR(5.0f, result[0], 0.01f);
    ASSERT_FLT_NEAR(0.0f, result[1], 0.01f);
    ASSERT_FLT_NEAR(0.0f, result[2], 0.01f);
    ASSERT_FLT_NEAR(5.0f, result[3], 0.01f);

    fm_buffer_free(ba);
    fm_buffer_free(bb);
    fm_buffer_free(bc);
}

static void test_sgemm_768x768(void) {
    /* Test larger matrix matches CPU result */
    int M = 768, N = 768, K = 768;
    size_t sz = M * N * sizeof(float);

    float *a = (float*)malloc(sz);
    float *b = (float*)malloc(sz);
    float *c_cpu = (float*)calloc(M * N, sizeof(float));
    float *c_gpu = (float*)calloc(M * N, sizeof(float));

    /* Fill with deterministic values */
    for (int i = 0; i < M * K; i++) a[i] = (float)(i % 17) * 0.01f;
    for (int i = 0; i < K * N; i++) b[i] = (float)(i % 13) * 0.01f;

    /* CPU reference (naive) */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += a[m * K + k] * b[k * N + n];
            }
            c_cpu[m * N + n] = sum;
        }
    }

    /* GPU */
    fm_buffer *ba = fm_buffer_from_ptr(a, M * K * sizeof(float));
    fm_buffer *bb = fm_buffer_from_ptr(b, K * N * sizeof(float));
    fm_buffer *bc = fm_buffer_from_ptr(c_gpu, M * N * sizeof(float));
    fm_sgemm(ba, bb, bc, M, N, K, 1.0f, 0.0f);

    /* Copy back if buffer was non-zero-copy */
    float *result = (float*)fm_buffer_data(bc);

    /* Compare with relaxed tolerance for large matmul */
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(result[i] - c_cpu[i]);
        if (diff > max_diff) max_diff = diff;
        /* Relative tolerance for large values */
        float tol = fabsf(c_cpu[i]) * 1e-4f + 1e-4f;
        if (diff > tol) mismatches++;
    }

    if (mismatches > 0) {
        printf("\n    768x768 GEMM: %d mismatches, max_diff=%.6f\n",
               mismatches, max_diff);
    }
    ASSERT_TRUE(mismatches == 0);

    fm_buffer_free(ba);
    fm_buffer_free(bb);
    fm_buffer_free(bc);
    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);
}

static void test_sgemv_basic(void) {
    /* A = [[1,2],[3,4]], x = [1,1] => y = [3, 7] */
    float a[4] = {1, 2, 3, 4};
    float x[2] = {1, 1};
    float y[2] = {0, 0};

    fm_buffer *ba = fm_buffer_from_ptr(a, sizeof(a));
    fm_buffer *bx = fm_buffer_from_ptr(x, sizeof(x));
    fm_buffer *by = fm_buffer_from_ptr(y, sizeof(y));

    fm_sgemv(ba, bx, by, 2, 2, 1.0f, 0.0f);

    float *result = (float*)fm_buffer_data(by);
    ASSERT_FLT_NEAR(3.0f, result[0], 0.01f);
    ASSERT_FLT_NEAR(7.0f, result[1], 0.01f);

    fm_buffer_free(ba);
    fm_buffer_free(bx);
    fm_buffer_free(by);
}

/* ================================================================
 * Phase 2: Custom compute kernels
 * ================================================================ */

static void test_silu_gpu(void) {
    int n = 8;
    float x[8] = {-2, -1, -0.5f, 0, 0.5f, 1, 2, 3};
    float out[8] = {0};
    float expected[8];

    for (int i = 0; i < n; i++)
        expected[i] = x[i] / (1.0f + expf(-x[i]));

    fm_buffer *bx = fm_buffer_from_ptr(x, sizeof(x));
    fm_buffer *bo = fm_buffer_from_ptr(out, sizeof(out));
    fm_silu(bx, bo, n);

    float *result = (float*)fm_buffer_data(bo);
    for (int i = 0; i < n; i++) {
        ASSERT_FLT_NEAR(expected[i], result[i], 0.01f);
    }

    fm_buffer_free(bx);
    fm_buffer_free(bo);
}

static void test_elementwise_add(void) {
    float a[4] = {1, 2, 3, 4};
    float b[4] = {5, 6, 7, 8};
    float out[4] = {0};

    fm_buffer *ba = fm_buffer_from_ptr(a, sizeof(a));
    fm_buffer *bb = fm_buffer_from_ptr(b, sizeof(b));
    fm_buffer *bo = fm_buffer_from_ptr(out, sizeof(out));
    fm_add(ba, bb, bo, 4);

    float *result = (float*)fm_buffer_data(bo);
    ASSERT_FLT_NEAR(6.0f,  result[0], 1e-6f);
    ASSERT_FLT_NEAR(8.0f,  result[1], 1e-6f);
    ASSERT_FLT_NEAR(10.0f, result[2], 1e-6f);
    ASSERT_FLT_NEAR(12.0f, result[3], 1e-6f);

    fm_buffer_free(ba);
    fm_buffer_free(bb);
    fm_buffer_free(bo);
}

static void test_elementwise_mul(void) {
    float a[4] = {1, 2, 3, 4};
    float b[4] = {5, 6, 7, 8};
    float out[4] = {0};

    fm_buffer *ba = fm_buffer_from_ptr(a, sizeof(a));
    fm_buffer *bb = fm_buffer_from_ptr(b, sizeof(b));
    fm_buffer *bo = fm_buffer_from_ptr(out, sizeof(out));
    fm_mul(ba, bb, bo, 4);

    float *result = (float*)fm_buffer_data(bo);
    ASSERT_FLT_NEAR(5.0f,  result[0], 1e-6f);
    ASSERT_FLT_NEAR(12.0f, result[1], 1e-6f);
    ASSERT_FLT_NEAR(21.0f, result[2], 1e-6f);
    ASSERT_FLT_NEAR(32.0f, result[3], 1e-6f);

    fm_buffer_free(ba);
    fm_buffer_free(bb);
    fm_buffer_free(bo);
}

static void test_softmax_gpu(void) {
    /* 2 rows x 4 cols */
    float x[8] = {1, 2, 3, 4, 1, 1, 1, 1};
    float out[8] = {0};

    fm_buffer *bx = fm_buffer_from_ptr(x, sizeof(x));
    fm_buffer *bo = fm_buffer_from_ptr(out, sizeof(out));
    fm_softmax(bx, bo, 2, 4);

    float *result = (float*)fm_buffer_data(bo);

    /* CPU reference for row 0: softmax([1,2,3,4]) */
    float cpu_out[8];
    for (int r = 0; r < 2; r++) {
        float mx = x[r*4];
        for (int c = 1; c < 4; c++)
            if (x[r*4+c] > mx) mx = x[r*4+c];
        float sum = 0;
        for (int c = 0; c < 4; c++) {
            cpu_out[r*4+c] = expf(x[r*4+c] - mx);
            sum += cpu_out[r*4+c];
        }
        for (int c = 0; c < 4; c++)
            cpu_out[r*4+c] /= sum;
    }

    for (int i = 0; i < 8; i++) {
        ASSERT_FLT_NEAR(cpu_out[i], result[i], 0.001f);
    }

    /* Row 1 should be uniform: [0.25, 0.25, 0.25, 0.25] */
    ASSERT_FLT_NEAR(0.25f, result[4], 0.001f);
    ASSERT_FLT_NEAR(0.25f, result[5], 0.001f);

    fm_buffer_free(bx);
    fm_buffer_free(bo);
}

static void test_rms_norm_gpu(void) {
    float x[4] = {1, 2, 3, 4};
    float w[4] = {1, 1, 1, 1};
    float out[4] = {0};

    /* CPU reference */
    float sum_sq = 0;
    for (int i = 0; i < 4; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / 4.0f + 1e-5f);
    float expected[4];
    for (int i = 0; i < 4; i++) expected[i] = x[i] * inv_rms * w[i];

    fm_buffer *bx = fm_buffer_from_ptr(x, sizeof(x));
    fm_buffer *bw = fm_buffer_from_ptr(w, sizeof(w));
    fm_buffer *bo = fm_buffer_from_ptr(out, sizeof(out));
    fm_rms_norm(bx, bw, bo, 4, 1e-5f);

    float *result = (float*)fm_buffer_data(bo);
    for (int i = 0; i < 4; i++) {
        ASSERT_FLT_NEAR(expected[i], result[i], 0.001f);
    }

    fm_buffer_free(bx);
    fm_buffer_free(bw);
    fm_buffer_free(bo);
}

static void test_wkv7_one_gpu(void) {
    /* Test WKV-7 with 2 heads, head_size=4.
     * Compare GPU output against CPU reference. */
    int n_head = 2, head_size = 4;
    int C = n_head * head_size;  /* 8 */
    int state_size = n_head * head_size * head_size;  /* 32 */

    float r[8], w[8], k[8], v[8], nkk[8], kka[8];
    float state_cpu[32], state_gpu[32];
    float y_cpu[8], y_gpu[8];

    /* Fill with deterministic values */
    for (int i = 0; i < C; i++) {
        r[i]   = 0.1f * (i + 1);
        w[i]   = 0.9f - 0.02f * i;
        k[i]   = 0.05f * (i % 5 + 1);
        v[i]   = 0.1f * (C - i);
        nkk[i] = -0.03f * (i + 1);
        kka[i] = 0.02f * (i + 2);
    }
    for (int i = 0; i < state_size; i++) {
        state_cpu[i] = 0.01f * (i % 7 + 1);
        state_gpu[i] = state_cpu[i];
    }

    /* CPU reference */
    for (int h = 0; h < n_head; h++) {
        int off = h * head_size;
        for (int i = 0; i < head_size; i++) {
            int sbase = (h * head_size + i) * head_size;
            float sa = 0;
            for (int j = 0; j < head_size; j++)
                sa += nkk[off + j] * state_cpu[sbase + j];
            float vv = v[off + i];
            float yy = 0;
            for (int j = 0; j < head_size; j++) {
                state_cpu[sbase + j] = state_cpu[sbase + j] * w[off + j]
                                     + k[off + j] * vv
                                     + sa * kka[off + j];
                yy += state_cpu[sbase + j] * r[off + j];
            }
            y_cpu[off + i] = yy;
        }
    }

    /* GPU */
    fm_buffer *br   = fm_buffer_from_ptr(r,   sizeof(r));
    fm_buffer *bw   = fm_buffer_from_ptr(w,   sizeof(w));
    fm_buffer *bk   = fm_buffer_from_ptr(k,   sizeof(k));
    fm_buffer *bv   = fm_buffer_from_ptr(v,   sizeof(v));
    fm_buffer *bnkk = fm_buffer_from_ptr(nkk, sizeof(nkk));
    fm_buffer *bkka = fm_buffer_from_ptr(kka, sizeof(kka));
    fm_buffer *bst  = fm_buffer_from_ptr(state_gpu, sizeof(state_gpu));
    fm_buffer *by   = fm_buffer_from_ptr(y_gpu, sizeof(y_gpu));

    fm_wkv7_one(br, bw, bk, bv, bnkk, bkka, bst, by, n_head, head_size);

    /* Compare output */
    float *gy = (float*)fm_buffer_data(by);
    for (int i = 0; i < C; i++) {
        ASSERT_FLT_NEAR(y_cpu[i], gy[i], 0.01f);
    }

    /* Compare state */
    float *gs = (float*)fm_buffer_data(bst);
    for (int i = 0; i < state_size; i++) {
        ASSERT_FLT_NEAR(state_cpu[i], gs[i], 0.01f);
    }

    fm_buffer_free(br);
    fm_buffer_free(bw);
    fm_buffer_free(bk);
    fm_buffer_free(bv);
    fm_buffer_free(bnkk);
    fm_buffer_free(bkka);
    fm_buffer_free(bst);
    fm_buffer_free(by);
}

static void test_matvec_q8_gpu(void) {
    /* 4 rows x 8 cols INT8 matvec */
    int rows = 4, cols = 8;

    int8_t x_q[8] = {10, -20, 30, -40, 50, -60, 70, -80};
    float x_scale = 0.01f;

    int8_t w_data[32];
    float w_scales[4] = {0.1f, 0.2f, 0.05f, 0.15f};
    for (int i = 0; i < 32; i++)
        w_data[i] = (int8_t)((i * 7 - 50) % 127);

    float out_cpu[4] = {0}, out_gpu[4] = {0};

    /* CPU reference */
    for (int r = 0; r < rows; r++) {
        int32_t dot = 0;
        for (int c = 0; c < cols; c++) {
            dot += (int32_t)x_q[c] * (int32_t)w_data[r * cols + c];
        }
        out_cpu[r] = (float)dot * w_scales[r] * x_scale;
    }

    fm_buffer *bxq = fm_buffer_from_ptr(x_q, sizeof(x_q));
    fm_buffer *bw  = fm_buffer_from_ptr(w_data, sizeof(w_data));
    fm_buffer *bws = fm_buffer_from_ptr(w_scales, sizeof(w_scales));
    fm_buffer *bo  = fm_buffer_from_ptr(out_gpu, sizeof(out_gpu));

    fm_matvec_q8(bxq, bw, bws, bo, x_scale, rows, cols);

    float *result = (float*)fm_buffer_data(bo);
    for (int r = 0; r < rows; r++) {
        ASSERT_FLT_NEAR(out_cpu[r], result[r], 0.01f);
    }

    fm_buffer_free(bxq);
    fm_buffer_free(bw);
    fm_buffer_free(bws);
    fm_buffer_free(bo);
}

/* ================================================================
 * Phase 4: Command batching
 * ================================================================ */

static void test_batch_basic(void) {
    ASSERT_FALSE(fm_batch_active());

    int ret = fm_batch_begin();
    ASSERT_INT_EQ(0, ret);
    ASSERT_TRUE(fm_batch_active());

    /* Do a couple ops within the batch */
    float a[4] = {1, 2, 3, 4};
    float b[4] = {5, 6, 7, 8};
    float out1[4] = {0};
    float out2[4] = {0};

    fm_buffer *ba = fm_buffer_from_ptr(a, sizeof(a));
    fm_buffer *bb = fm_buffer_from_ptr(b, sizeof(b));
    fm_buffer *bo1 = fm_buffer_from_ptr(out1, sizeof(out1));
    fm_buffer *bo2 = fm_buffer_from_ptr(out2, sizeof(out2));

    fm_add(ba, bb, bo1, 4);
    fm_mul(ba, bb, bo2, 4);

    /* Results shouldn't be ready until batch_end */
    fm_batch_end();
    ASSERT_FALSE(fm_batch_active());

    float *r1 = (float*)fm_buffer_data(bo1);
    float *r2 = (float*)fm_buffer_data(bo2);
    ASSERT_FLT_NEAR(6.0f,  r1[0], 1e-6f);
    ASSERT_FLT_NEAR(5.0f,  r2[0], 1e-6f);
    ASSERT_FLT_NEAR(32.0f, r2[3], 1e-6f);

    fm_buffer_free(ba);
    fm_buffer_free(bb);
    fm_buffer_free(bo1);
    fm_buffer_free(bo2);
}

/* ================================================================
 * main
 * ================================================================ */

int main(void) {
    printf("\n  Metal GPU Backend Tests\n");
    printf("  =======================\n\n");

    /* Initialize Metal once for all tests */
    if (fm_init() != 0) {
        printf("  Metal not available — skipping all tests.\n\n");
        return 0;
    }

    /* Phase 1: Infrastructure */
    RUN_TEST(init_and_available);
    RUN_TEST(double_init_is_safe);
    RUN_TEST(buffer_alloc);
    RUN_TEST(buffer_from_ptr);
    RUN_TEST(buffer_alloc_write_read);

    /* Phase 1: MPS GEMM/GEMV */
    RUN_TEST(sgemm_2x2);
    RUN_TEST(sgemm_alpha_beta);
    RUN_TEST(sgemm_768x768);
    RUN_TEST(sgemv_basic);

    /* Phase 2: Custom kernels */
    RUN_TEST(silu_gpu);
    RUN_TEST(elementwise_add);
    RUN_TEST(elementwise_mul);
    RUN_TEST(softmax_gpu);
    RUN_TEST(rms_norm_gpu);
    RUN_TEST(wkv7_one_gpu);

    /* Phase 3: INT8 quantized */
    RUN_TEST(matvec_q8_gpu);

    /* Phase 4: Batching */
    RUN_TEST(batch_basic);

    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
