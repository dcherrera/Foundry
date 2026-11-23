/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../quantization.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================= */
/* Utility Tests                                                 */
/* ============================================================= */

static void test_dtype_names(void) {
    ASSERT_STR_EQ("INT8", fq_dtype_name(FQ_INT8));
    ASSERT_STR_EQ("INT4", fq_dtype_name(FQ_INT4));
    ASSERT_STR_EQ("UINT8", fq_dtype_name(FQ_UINT8));
}

static void test_dtype_bytes(void) {
    ASSERT_FLT_NEAR(1.0f, fq_dtype_bytes(FQ_INT8), 0.001f);
    ASSERT_FLT_NEAR(0.5f, fq_dtype_bytes(FQ_INT4), 0.001f);
    ASSERT_FLT_NEAR(1.0f, fq_dtype_bytes(FQ_UINT8), 0.001f);
}

static void test_compute_scale_symmetric(void) {
    float data[] = {-1.0f, 0.5f, 1.0f, -0.5f};
    float scale;
    int32_t zero_point;

    fq_compute_scale(data, 4, FQ_SYMMETRIC, &scale, &zero_point);

    ASSERT_FLT_NEAR(1.0f / 127.0f, scale, 0.001f);
    ASSERT_INT_EQ(0, zero_point);
}

/* ============================================================= */
/* INT8 Quantization Tests                                       */
/* ============================================================= */

static void test_int8_quantize_basic(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int64_t shape[] = {4};

    fq_tensor* qt = fq_quantize_int8(data, shape, 1);
    ASSERT_NOT_NULL(qt);
    ASSERT_INT_EQ(FQ_INT8, qt->dtype);
    ASSERT_INT_EQ(FQ_SYMMETRIC, qt->mode);
    ASSERT_INT_EQ(FQ_PER_TENSOR, qt->granularity);
    ASSERT_INT_EQ(4, (int)qt->numel);
    ASSERT_INT_EQ(1, qt->num_scales);

    fq_free(qt);
}

static void test_int8_roundtrip(void) {
    float data[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    int64_t shape[] = {5};

    fq_tensor* qt = fq_quantize_int8(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    float* dequant = fq_dequantize(qt);
    ASSERT_NOT_NULL(dequant);

    /* Should be close to original */
    for (int i = 0; i < 5; i++) {
        ASSERT_FLT_NEAR(data[i], dequant[i], 0.02f);  /* ~1% error allowed */
    }

    free(dequant);
    fq_free(qt);
}

static void test_int8_zeros(void) {
    float data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int64_t shape[] = {4};

    fq_tensor* qt = fq_quantize_int8(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    float* dequant = fq_dequantize(qt);
    ASSERT_NOT_NULL(dequant);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLT_NEAR(0.0f, dequant[i], 0.001f);
    }

    free(dequant);
    fq_free(qt);
}

static void test_int8_large_values(void) {
    float data[] = {-100.0f, -50.0f, 0.0f, 50.0f, 100.0f};
    int64_t shape[] = {5};

    fq_tensor* qt = fq_quantize_int8(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    float* dequant = fq_dequantize(qt);
    ASSERT_NOT_NULL(dequant);

    /* Values should be approximately preserved */
    ASSERT_FLT_NEAR(-100.0f, dequant[0], 2.0f);
    ASSERT_FLT_NEAR(0.0f, dequant[2], 2.0f);
    ASSERT_FLT_NEAR(100.0f, dequant[4], 2.0f);

    free(dequant);
    fq_free(qt);
}

static void test_int8_2d_tensor(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int64_t shape[] = {2, 3};

    fq_tensor* qt = fq_quantize_int8(data, shape, 2);
    ASSERT_NOT_NULL(qt);
    ASSERT_INT_EQ(2, qt->ndim);
    ASSERT_INT_EQ(2, (int)qt->shape[0]);
    ASSERT_INT_EQ(3, (int)qt->shape[1]);
    ASSERT_INT_EQ(6, (int)qt->numel);

    fq_free(qt);
}

/* ============================================================= */
/* INT8 Per-Channel Tests                                        */
/* ============================================================= */

static void test_int8_per_channel(void) {
    /* 2 channels, 4 elements each */
    float data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,    /* channel 0: max=4 */
        10.0f, 20.0f, 30.0f, 40.0f /* channel 1: max=40 */
    };
    int64_t shape[] = {2, 4};

    fq_tensor* qt = fq_quantize_int8_per_channel(data, shape, 2, 0);
    ASSERT_NOT_NULL(qt);
    ASSERT_INT_EQ(FQ_PER_CHANNEL, qt->granularity);
    ASSERT_INT_EQ(2, qt->num_scales);

    /* Scales should be different for each channel */
    ASSERT_TRUE(qt->scales[0] != qt->scales[1]);

    float* dequant = fq_dequantize(qt);
    ASSERT_NOT_NULL(dequant);

    /* Both channels should be well-preserved */
    ASSERT_FLT_NEAR(4.0f, dequant[3], 0.1f);
    ASSERT_FLT_NEAR(40.0f, dequant[7], 1.0f);

    free(dequant);
    fq_free(qt);
}

/* ============================================================= */
/* INT4 Quantization Tests                                       */
/* ============================================================= */

static void test_int4_quantize_basic(void) {
    float data[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    int64_t shape[] = {8};

    fq_tensor* qt = fq_quantize_int4(data, shape, 1, 8);
    ASSERT_NOT_NULL(qt);
    ASSERT_INT_EQ(FQ_INT4, qt->dtype);
    ASSERT_INT_EQ(FQ_PER_GROUP, qt->granularity);
    ASSERT_INT_EQ(8, (int)qt->numel);

    fq_free(qt);
}

static void test_int4_roundtrip(void) {
    float data[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    int64_t shape[] = {4};

    fq_tensor* qt = fq_quantize_int4(data, shape, 1, 4);
    ASSERT_NOT_NULL(qt);

    float* dequant = fq_dequantize(qt);
    ASSERT_NOT_NULL(dequant);

    /* INT4 has more quantization error, allow ~15% */
    for (int i = 0; i < 4; i++) {
        float expected = data[i];
        float tolerance = fabs(expected) * 0.2f + 0.3f;
        ASSERT_FLT_NEAR(expected, dequant[i], tolerance);
    }

    free(dequant);
    fq_free(qt);
}

static void test_int4_packing(void) {
    /* Test that INT4 uses ~half the memory of INT8 */
    float data[128];
    for (int i = 0; i < 128; i++) {
        data[i] = (float)(i - 64) / 10.0f;
    }
    int64_t shape[] = {128};

    fq_tensor* qt8 = fq_quantize_int8(data, shape, 1);
    fq_tensor* qt4 = fq_quantize_int4(data, shape, 1, 32);

    ASSERT_NOT_NULL(qt8);
    ASSERT_NOT_NULL(qt4);

    size_t mem8 = fq_memory_usage(qt8);
    size_t mem4 = fq_memory_usage(qt4);

    /* INT4 should use roughly half the memory (plus scale overhead) */
    ASSERT_TRUE(mem4 < mem8);

    fq_free(qt8);
    fq_free(qt4);
}

/* ============================================================= */
/* Quantized Matmul Tests                                        */
/* ============================================================= */

static void test_quantized_matmul(void) {
    /* A: 2x3, B: 3x2 -> C: 2x2 */
    float A[] = {1.0f, 2.0f, 3.0f,
                 4.0f, 5.0f, 6.0f};

    float B[] = {1.0f, 2.0f,
                 3.0f, 4.0f,
                 5.0f, 6.0f};
    int64_t B_shape[] = {3, 2};

    /* Expected: A @ B
     * [1,2,3] @ [[1,2],[3,4],[5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
     * [4,5,6] @ [[1,2],[3,4],[5,6]] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
     */
    float expected[] = {22.0f, 28.0f, 49.0f, 64.0f};

    fq_tensor* B_q = fq_quantize_int8(B, B_shape, 2);
    ASSERT_NOT_NULL(B_q);

    float* C = fq_matmul(A, B_q, 2, 3, 2);
    ASSERT_NOT_NULL(C);

    /* Allow some quantization error */
    for (int i = 0; i < 4; i++) {
        ASSERT_FLT_NEAR(expected[i], C[i], 1.0f);
    }

    free(C);
    fq_free(B_q);
}

/* ============================================================= */
/* Memory Usage Tests                                            */
/* ============================================================= */

static void test_memory_usage(void) {
    float data[1024];
    for (int i = 0; i < 1024; i++) {
        data[i] = (float)i / 1024.0f;
    }
    int64_t shape[] = {1024};

    fq_tensor* qt = fq_quantize_int8(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    size_t mem = fq_memory_usage(qt);
    /* Should be ~1024 bytes for data + scale */
    ASSERT_TRUE(mem >= 1024);
    ASSERT_TRUE(mem < 2048);

    /* Compression ratio should be ~4x (float32 -> int8) */
    float compression = (1024.0f * 4.0f) / mem;
    ASSERT_TRUE(compression > 3.0f);

    fq_free(qt);
}

/* ============================================================= */
/* Quantization Error Tests                                      */
/* ============================================================= */

static void test_quantization_error(void) {
    float data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    int64_t shape[] = {5};

    fq_tensor* qt = fq_quantize_int8(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    float mse = fq_quantization_error(data, qt);
    ASSERT_TRUE(mse >= 0.0f);
    ASSERT_TRUE(mse < 0.01f);  /* Very low error expected */

    fq_free(qt);
}

/* ============================================================= */
/* Ternary Export Variant Tests                                  */
/* ============================================================= */

static void test_ternary_absmean_threshold(void) {
    /* All values same magnitude → threshold = 1.0 */
    float data[] = {1.0f, -1.0f, 1.0f, -1.0f};
    int64_t shape[] = {4};

    fq_tensor* qt = fq_quantize_ternary(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    /* absmean of {1,1,1,1} = 1.0 */
    ASSERT_FLT_NEAR(1.0f, qt->scales[0], 0.001f);

    fq_free(qt);
}

static void test_ternary_manual_threshold(void) {
    float data[] = {3.0f, -3.0f, 0.5f, -0.5f};
    int64_t shape[] = {4};

    fq_tensor* qt = fq_quantize_ternary_threshold(data, shape, 1, 1.0f);
    ASSERT_NOT_NULL(qt);

    ASSERT_FLT_NEAR(1.0f, qt->scales[0], 0.001f);

    float* dequant = fq_dequantize(qt);
    ASSERT_NOT_NULL(dequant);

    /* 3.0 > 1.0 → +1*1.0 = 1.0 */
    ASSERT_FLT_NEAR(1.0f, dequant[0], 0.001f);
    /* -3.0 < -1.0 → -1*1.0 = -1.0 */
    ASSERT_FLT_NEAR(-1.0f, dequant[1], 0.001f);
    /* 0.5 in [-1.0, 1.0] → 0 */
    ASSERT_FLT_NEAR(0.0f, dequant[2], 0.001f);
    /* -0.5 in [-1.0, 1.0] → 0 */
    ASSERT_FLT_NEAR(0.0f, dequant[3], 0.001f);

    free(dequant);
    fq_free(qt);
}

static void test_ternary_packing_edge_cases(void) {
    /* Test with numel not multiple of 4 */
    for (int n = 1; n <= 7; n++) {
        float* data = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            data[i] = (i % 3 == 0) ? 2.0f : (i % 3 == 1) ? -2.0f : 0.0f;
        }
        int64_t shape[] = {n};

        fq_tensor* qt = fq_quantize_ternary(data, shape, 1);
        ASSERT_NOT_NULL(qt);
        ASSERT_INT_EQ(n, (int)qt->numel);

        float* dequant = fq_dequantize(qt);
        ASSERT_NOT_NULL(dequant);

        /* Verify all values are valid ternary outputs */
        float scale = qt->scales[0];
        for (int i = 0; i < n; i++) {
            float abs_val = fabsf(dequant[i]);
            ASSERT_TRUE(abs_val < 0.001f || fabsf(abs_val - scale) < 0.001f);
        }

        free(dequant);
        free(data);
        fq_free(qt);
    }
}

static void test_ternary_error(void) {
    float data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, -0.1f, -0.3f, -0.5f};
    int64_t shape[] = {8};

    fq_tensor* qt = fq_quantize_ternary(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    float mse = fq_quantization_error(data, qt);
    ASSERT_TRUE(mse >= 0.0f);
    /* Ternary has higher error than INT8, but should be reasonable */
    ASSERT_TRUE(mse < 1.0f);

    fq_free(qt);
}

static void test_ternary_memory_usage(void) {
    float data[1024];
    for (int i = 0; i < 1024; i++) data[i] = (float)(i - 512) / 100.0f;
    int64_t shape[] = {1024};

    fq_tensor* qt = fq_quantize_ternary(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    size_t mem = fq_memory_usage(qt);
    /* 1024 values at 2 bits each = 256 bytes + 4 bytes scale = 260 */
    ASSERT_TRUE(mem >= 256);
    ASSERT_TRUE(mem < 512);

    /* Compression should be ~16x (f32 → ternary) */
    float compression = (1024.0f * 4.0f) / mem;
    ASSERT_TRUE(compression > 12.0f);

    fq_free(qt);
}

static void test_ternary_per_channel(void) {
    /* 2 channels with different scales */
    float data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,      /* channel 0: absmean = 2.5 */
        10.0f, 20.0f, 30.0f, 40.0f    /* channel 1: absmean = 25.0 */
    };
    int64_t shape[] = {2, 4};

    fq_tensor* qt = fq_quantize_ternary_per_channel(data, shape, 2, 0);
    ASSERT_NOT_NULL(qt);
    ASSERT_INT_EQ(FQ_PER_CHANNEL, qt->granularity);
    ASSERT_INT_EQ(2, qt->num_scales);
    ASSERT_TRUE(qt->scales[0] != qt->scales[1]);

    fq_free(qt);
}

static void test_ternary_generic_dispatch(void) {
    float data[] = {1.0f, -1.0f, 0.5f, -0.5f};
    int64_t shape[] = {4};

    fq_tensor* qt = fq_quantize(data, shape, 1, FQ_TERNARY, FQ_SYMMETRIC, FQ_PER_TENSOR, 0, 0);
    ASSERT_NOT_NULL(qt);
    ASSERT_INT_EQ(FQ_TERNARY, qt->dtype);

    fq_free(qt);
}

/* ============================================================= */
/* Edge Cases                                                    */
/* ============================================================= */

static void test_null_safety(void) {
    ASSERT_NULL(fq_quantize_int8(NULL, NULL, 0));
    ASSERT_NULL(fq_dequantize(NULL));
    ASSERT_NULL(fq_matmul(NULL, NULL, 0, 0, 0));
    ASSERT_INT_EQ(0, (int)fq_memory_usage(NULL));

    fq_free(NULL);  /* Should not crash */

    ASSERT_TRUE(1);
}

static void test_single_element(void) {
    float data[] = {42.0f};
    int64_t shape[] = {1};

    fq_tensor* qt = fq_quantize_int8(data, shape, 1);
    ASSERT_NOT_NULL(qt);
    ASSERT_INT_EQ(1, (int)qt->numel);

    float* dequant = fq_dequantize(qt);
    ASSERT_FLT_NEAR(42.0f, dequant[0], 0.5f);

    free(dequant);
    fq_free(qt);
}

static void test_dequantize_range(void) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int64_t shape[] = {5};

    fq_tensor* qt = fq_quantize_int8(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    float* range = fq_dequantize_range(qt, 1, 4);
    ASSERT_NOT_NULL(range);

    /* Should get elements 1, 2, 3 */
    ASSERT_FLT_NEAR(2.0f, range[0], 0.1f);
    ASSERT_FLT_NEAR(3.0f, range[1], 0.1f);
    ASSERT_FLT_NEAR(4.0f, range[2], 0.1f);

    free(range);
    fq_free(qt);
}

/* ============================================================= */
/* Ternary Quantization Tests                                    */
/* ============================================================= */

static void test_ternary_dtype(void) {
    ASSERT_STR_EQ("TERNARY", fq_dtype_name(FQ_TERNARY));
    ASSERT_FLT_NEAR(0.25f, fq_dtype_bytes(FQ_TERNARY), 0.001f);
}

static void test_ternary_quantize_basic(void) {
    float data[] = {1.0f, -1.0f, 0.5f, -0.5f, 0.0f, 2.0f};
    int64_t shape[] = {6};

    fq_tensor* qt = fq_quantize_ternary(data, shape, 1);
    ASSERT_NOT_NULL(qt);
    ASSERT_INT_EQ(FQ_TERNARY, qt->dtype);
    ASSERT_INT_EQ(FQ_SYMMETRIC, qt->mode);
    ASSERT_INT_EQ(FQ_PER_TENSOR, qt->granularity);
    ASSERT_INT_EQ(6, (int)qt->numel);
    ASSERT_INT_EQ(1, qt->ndim);
    ASSERT_INT_EQ(6, (int)qt->shape[0]);

    fq_free(qt);
}

static void test_ternary_values(void) {
    /* All values clearly map to -1, 0, or +1 */
    float data[] = {3.0f, -3.0f, 0.0f, 3.0f};
    int64_t shape[] = {4};
    /* absmean = (3+3+0+3)/4 = 2.25 */
    /* q[0] = round(3/2.25) = round(1.33) = 1 -> +1 */
    /* q[1] = round(-3/2.25) = round(-1.33) = -1 -> -1 */
    /* q[2] = round(0/2.25) = 0 -> 0 */
    /* q[3] = round(3/2.25) = round(1.33) = 1 -> +1 */

    fq_tensor* qt = fq_quantize_ternary(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    float* dequant = fq_dequantize(qt);
    ASSERT_NOT_NULL(dequant);

    float scale = qt->scales[0];
    /* Dequantized values should be exactly {scale, -scale, 0, scale} */
    ASSERT_FLT_NEAR(scale, dequant[0], 0.001f);
    ASSERT_FLT_NEAR(-scale, dequant[1], 0.001f);
    ASSERT_FLT_NEAR(0.0f, dequant[2], 0.001f);
    ASSERT_FLT_NEAR(scale, dequant[3], 0.001f);

    free(dequant);
    fq_free(qt);
}

static void test_ternary_absmean(void) {
    float data[] = {2.0f, -4.0f, 6.0f, -8.0f};
    int64_t shape[] = {4};
    /* absmean = (2+4+6+8)/4 = 5.0 */

    fq_tensor* qt = fq_quantize_ternary(data, shape, 1);
    ASSERT_NOT_NULL(qt);
    ASSERT_FLT_NEAR(5.0f, qt->scales[0], 0.001f);

    fq_free(qt);
}

static void test_ternary_roundtrip(void) {
    float data[] = {1.0f, -1.0f, 0.5f, -0.5f, 0.0f, 0.8f, -0.8f, 0.3f};
    int64_t shape[] = {8};

    fq_tensor* qt = fq_quantize_ternary(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    float* dequant = fq_dequantize(qt);
    ASSERT_NOT_NULL(dequant);

    /* Ternary is lossy — each dequantized value should be one of {-scale, 0, +scale} */
    float scale = qt->scales[0];
    for (int i = 0; i < 8; i++) {
        float v = dequant[i];
        ASSERT_TRUE(fabsf(v - scale) < 0.001f ||
                    fabsf(v + scale) < 0.001f ||
                    fabsf(v) < 0.001f);
    }

    free(dequant);
    fq_free(qt);
}

static void test_ternary_packing(void) {
    /* Ternary should use ~4x less memory than INT8 for data */
    float data[128];
    for (int i = 0; i < 128; i++) {
        data[i] = (float)(i - 64) / 10.0f;
    }
    int64_t shape[] = {128};

    fq_tensor* qt8 = fq_quantize_int8(data, shape, 1);
    fq_tensor* qtT = fq_quantize_ternary(data, shape, 1);

    ASSERT_NOT_NULL(qt8);
    ASSERT_NOT_NULL(qtT);

    size_t mem8 = fq_memory_usage(qt8);
    size_t memT = fq_memory_usage(qtT);

    /* Ternary should use significantly less memory */
    ASSERT_TRUE(memT < mem8);
    /* 128 elements: INT8 = 128 bytes data + 4 scale = 132
     * Ternary = 32 bytes data + 4 scale = 36
     * So memT should be roughly 1/4 of mem8 */
    ASSERT_TRUE(memT < mem8 / 2);

    fq_free(qt8);
    fq_free(qtT);
}

static void test_ternary_zeros(void) {
    float data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int64_t shape[] = {4};

    fq_tensor* qt = fq_quantize_ternary(data, shape, 1);
    ASSERT_NOT_NULL(qt);

    float* dequant = fq_dequantize(qt);
    ASSERT_NOT_NULL(dequant);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLT_NEAR(0.0f, dequant[i], 0.001f);
    }

    free(dequant);
    fq_free(qt);
}

static void test_ternary_matmul(void) {
    /* A: 2×3, B: 3×2 -> C: 2×2 */
    /* Use values that quantize cleanly to ternary */
    float A[] = {1.0f, 2.0f, 3.0f,
                 4.0f, 5.0f, 6.0f};

    /* B values designed so ternary quantization gives known weights */
    /* B = {1, -1, 0, 1, -1, 0} with scale = absmean = (1+1+0+1+1+0)/6 = 0.667 */
    /* After quantize: round(val/0.667) clamped to {-1,0,1} -> {1,-1,0,1,-1,0} */
    /* Wait — let's just use round numbers that are exact. */
    /* If B = {1,1,1,1,1,1}, absmean = 1.0, all round to 1 -> all +1 */
    float B[] = {1.0f, 1.0f,
                 1.0f, 1.0f,
                 1.0f, 1.0f};
    int64_t B_shape[] = {3, 2};

    fq_tensor* B_q = fq_quantize_ternary(B, B_shape, 2);
    ASSERT_NOT_NULL(B_q);

    /* All B values are +1, scale = 1.0 */
    /* C = A @ B_ternary = A @ ones(3,2) * scale
     * Row 0: (1+2+3)*1 = 6, (1+2+3)*1 = 6
     * Row 1: (4+5+6)*1 = 15, (4+5+6)*1 = 15
     */
    float* C = fq_matmul_ternary(A, B_q, 2, 3, 2);
    ASSERT_NOT_NULL(C);

    ASSERT_FLT_NEAR(6.0f, C[0], 0.01f);
    ASSERT_FLT_NEAR(6.0f, C[1], 0.01f);
    ASSERT_FLT_NEAR(15.0f, C[2], 0.01f);
    ASSERT_FLT_NEAR(15.0f, C[3], 0.01f);

    free(C);
    fq_free(B_q);
}

static void test_ternary_matmul_identity(void) {
    /* Test with mixed {-1, 0, +1} weights */
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};  /* 1×4 */

    /* B: 4×1, values = {1, -1, 1, -1}, absmean = 1.0, all exact */
    float B[] = {1.0f, -1.0f, 1.0f, -1.0f};
    int64_t B_shape[] = {4, 1};

    fq_tensor* B_q = fq_quantize_ternary(B, B_shape, 2);
    ASSERT_NOT_NULL(B_q);
    ASSERT_FLT_NEAR(1.0f, B_q->scales[0], 0.001f);

    /* C = [1,2,3,4] @ [1,-1,1,-1]^T * 1.0 = 1-2+3-4 = -2 */
    float* C = fq_matmul_ternary(A, B_q, 1, 4, 1);
    ASSERT_NOT_NULL(C);
    ASSERT_FLT_NEAR(-2.0f, C[0], 0.01f);

    free(C);
    fq_free(B_q);
}

static void test_ternary_null_safety(void) {
    ASSERT_NULL(fq_quantize_ternary(NULL, NULL, 0));
    ASSERT_NULL(fq_matmul_ternary(NULL, NULL, 0, 0, 0));

    /* fq_matmul_ternary_acc with NULL should not crash */
    fq_matmul_ternary_acc(NULL, NULL, NULL, 0, 0, 0);
    ASSERT_TRUE(1);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running quantization tests...\n\n");

    /* Utility tests */
    RUN_TEST(dtype_names);
    RUN_TEST(dtype_bytes);
    RUN_TEST(compute_scale_symmetric);

    /* INT8 tests */
    RUN_TEST(int8_quantize_basic);
    RUN_TEST(int8_roundtrip);
    RUN_TEST(int8_zeros);
    RUN_TEST(int8_large_values);
    RUN_TEST(int8_2d_tensor);

    /* INT8 per-channel tests */
    RUN_TEST(int8_per_channel);

    /* INT4 tests */
    RUN_TEST(int4_quantize_basic);
    RUN_TEST(int4_roundtrip);
    RUN_TEST(int4_packing);

    /* Quantized operations */
    RUN_TEST(quantized_matmul);

    /* Memory usage */
    RUN_TEST(memory_usage);

    /* Quantization error */
    RUN_TEST(quantization_error);

    /* Ternary export variant tests */
    RUN_TEST(ternary_absmean_threshold);
    RUN_TEST(ternary_manual_threshold);
    RUN_TEST(ternary_packing_edge_cases);
    RUN_TEST(ternary_error);
    RUN_TEST(ternary_memory_usage);
    RUN_TEST(ternary_per_channel);
    RUN_TEST(ternary_generic_dispatch);

    /* Edge cases */
    RUN_TEST(null_safety);
    RUN_TEST(single_element);
    RUN_TEST(dequantize_range);

    /* Ternary tests */
    RUN_TEST(ternary_dtype);
    RUN_TEST(ternary_quantize_basic);
    RUN_TEST(ternary_values);
    RUN_TEST(ternary_absmean);
    RUN_TEST(ternary_roundtrip);
    RUN_TEST(ternary_packing);
    RUN_TEST(ternary_zeros);
    RUN_TEST(ternary_matmul);
    RUN_TEST(ternary_matmul_identity);
    RUN_TEST(ternary_null_safety);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
