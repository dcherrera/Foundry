/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../tensor.h"

#include <math.h>

/* ============================================================= */
/* Creation Tests                                                */
/* ============================================================= */

static void test_zeros_creates_tensor(void) {
    int64_t shape[] = {2, 3};
    tm_tensor* t = tm_zeros(shape, 2, TM_F32);

    ASSERT_NOT_NULL(t);
    ASSERT_INT_EQ(2, tm_ndim(t));
    ASSERT_INT_EQ(2, tm_size(t, 0));
    ASSERT_INT_EQ(3, tm_size(t, 1));
    ASSERT_INT_EQ(6, tm_numel(t));

    /* Check all zeros */
    for (int64_t i = 0; i < 6; i++) {
        ASSERT_FLT_EQ(0.0f, tm_get_f32(t, i));
    }

    tm_free(t);
}

static void test_ones_creates_tensor(void) {
    int64_t shape[] = {3};
    tm_tensor* t = tm_ones(shape, 1, TM_F32);

    ASSERT_NOT_NULL(t);
    ASSERT_INT_EQ(1, tm_ndim(t));
    ASSERT_INT_EQ(3, tm_numel(t));

    for (int64_t i = 0; i < 3; i++) {
        ASSERT_FLT_EQ(1.0f, tm_get_f32(t, i));
    }

    tm_free(t);
}

static void test_full_creates_tensor(void) {
    int64_t shape[] = {2, 2};
    tm_tensor* t = tm_full(shape, 2, TM_F32, 3.14);

    ASSERT_NOT_NULL(t);
    ASSERT_INT_EQ(4, tm_numel(t));

    for (int64_t i = 0; i < 4; i++) {
        ASSERT_FLT_NEAR(3.14f, tm_get_f32(t, i), 0.001f);
    }

    tm_free(t);
}

static void test_randn_creates_tensor(void) {
    tm_manual_seed(42);
    int64_t shape[] = {100};
    tm_tensor* t = tm_randn(shape, 1, TM_F32);

    ASSERT_NOT_NULL(t);
    ASSERT_INT_EQ(100, tm_numel(t));

    /* Check approximate normal distribution */
    float sum = 0;
    for (int64_t i = 0; i < 100; i++) {
        sum += tm_get_f32(t, i);
    }
    float mean = sum / 100.0f;
    /* Mean should be close to 0 for normal distribution */
    ASSERT_FLT_NEAR(0.0f, mean, 0.5f);

    tm_free(t);
}

static void test_clone_creates_copy(void) {
    int64_t shape[] = {2, 2};
    tm_tensor* t = tm_full(shape, 2, TM_F32, 5.0);
    tm_tensor* c = tm_clone(t);

    ASSERT_NOT_NULL(c);
    ASSERT_INT_EQ(tm_numel(t), tm_numel(c));

    /* Modify original, clone should be unchanged */
    tm_set_f32(t, 0, 99.0f);
    ASSERT_FLT_EQ(5.0f, tm_get_f32(c, 0));

    tm_free(t);
    tm_free(c);
}

/* ============================================================= */
/* Accessor Tests                                                */
/* ============================================================= */

static void test_get_set_f32(void) {
    int64_t shape[] = {4};
    tm_tensor* t = tm_zeros(shape, 1, TM_F32);

    tm_set_f32(t, 2, 42.0f);
    ASSERT_FLT_EQ(42.0f, tm_get_f32(t, 2));
    ASSERT_FLT_EQ(0.0f, tm_get_f32(t, 0));

    tm_free(t);
}

static void test_data_pointer(void) {
    int64_t shape[] = {3};
    tm_tensor* t = tm_ones(shape, 1, TM_F32);

    float* data = tm_data_f32(t);
    ASSERT_NOT_NULL(data);
    ASSERT_FLT_EQ(1.0f, data[0]);

    data[1] = 7.0f;
    ASSERT_FLT_EQ(7.0f, tm_get_f32(t, 1));

    tm_free(t);
}

/* ============================================================= */
/* Shape Tests                                                   */
/* ============================================================= */

static void test_reshape(void) {
    int64_t shape[] = {2, 3};
    tm_tensor* t = tm_ones(shape, 2, TM_F32);

    int64_t new_shape[] = {6};
    tm_tensor* r = tm_reshape(t, new_shape, 1);

    ASSERT_NOT_NULL(r);
    ASSERT_INT_EQ(1, tm_ndim(r));
    ASSERT_INT_EQ(6, tm_size(r, 0));
    ASSERT_INT_EQ(6, tm_numel(r));

    tm_free(t);
    tm_free(r);
}

static void test_transpose(void) {
    int64_t shape[] = {2, 3};
    tm_tensor* t = tm_zeros(shape, 2, TM_F32);

    /* Set t[0, 1] = 5.0 */
    float* data = tm_data_f32(t);
    data[0 * 3 + 1] = 5.0f;

    tm_tensor* tr = tm_transpose(t, 0, 1);

    ASSERT_NOT_NULL(tr);
    ASSERT_INT_EQ(3, tm_size(tr, 0));
    ASSERT_INT_EQ(2, tm_size(tr, 1));

    tm_free(t);
    tm_free(tr);
}

static void test_is_contiguous(void) {
    int64_t shape[] = {2, 3};
    tm_tensor* t = tm_zeros(shape, 2, TM_F32);

    ASSERT_TRUE(tm_is_contiguous(t));

    /* 2D transpose now returns a contiguous copy for matmul compatibility */
    tm_tensor* tr = tm_transpose(t, 0, 1);
    ASSERT_TRUE(tm_is_contiguous(tr));  /* Now contiguous */
    ASSERT_INT_EQ(tr->shape[0], 3);     /* Shape is transposed */
    ASSERT_INT_EQ(tr->shape[1], 2);

    tm_free(t);
    tm_free(tr);
}

/* ============================================================= */
/* Element-wise Op Tests                                         */
/* ============================================================= */

static void test_add(void) {
    int64_t shape[] = {3};
    tm_tensor* a = tm_full(shape, 1, TM_F32, 2.0);
    tm_tensor* b = tm_full(shape, 1, TM_F32, 3.0);

    tm_tensor* c = tm_add(a, b);

    ASSERT_NOT_NULL(c);
    for (int64_t i = 0; i < 3; i++) {
        ASSERT_FLT_EQ(5.0f, tm_get_f32(c, i));
    }

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_sub(void) {
    int64_t shape[] = {3};
    tm_tensor* a = tm_full(shape, 1, TM_F32, 5.0);
    tm_tensor* b = tm_full(shape, 1, TM_F32, 3.0);

    tm_tensor* c = tm_sub(a, b);

    ASSERT_NOT_NULL(c);
    for (int64_t i = 0; i < 3; i++) {
        ASSERT_FLT_EQ(2.0f, tm_get_f32(c, i));
    }

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_mul(void) {
    int64_t shape[] = {3};
    tm_tensor* a = tm_full(shape, 1, TM_F32, 2.0);
    tm_tensor* b = tm_full(shape, 1, TM_F32, 4.0);

    tm_tensor* c = tm_mul(a, b);

    ASSERT_NOT_NULL(c);
    for (int64_t i = 0; i < 3; i++) {
        ASSERT_FLT_EQ(8.0f, tm_get_f32(c, i));
    }

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_mul_scalar(void) {
    int64_t shape[] = {3};
    tm_tensor* a = tm_full(shape, 1, TM_F32, 3.0);

    tm_tensor* c = tm_mul_scalar(a, 2.0);

    ASSERT_NOT_NULL(c);
    for (int64_t i = 0; i < 3; i++) {
        ASSERT_FLT_EQ(6.0f, tm_get_f32(c, i));
    }

    tm_free(a);
    tm_free(c);
}

static void test_exp(void) {
    int64_t shape[] = {2};
    tm_tensor* t = tm_zeros(shape, 1, TM_F32);
    tm_set_f32(t, 0, 0.0f);
    tm_set_f32(t, 1, 1.0f);

    tm_tensor* e = tm_exp(t);

    ASSERT_FLT_NEAR(1.0f, tm_get_f32(e, 0), 0.001f);       /* exp(0) = 1 */
    ASSERT_FLT_NEAR(2.718f, tm_get_f32(e, 1), 0.01f);     /* exp(1) ≈ 2.718 */

    tm_free(t);
    tm_free(e);
}

static void test_relu(void) {
    int64_t shape[] = {4};
    tm_tensor* t = tm_zeros(shape, 1, TM_F32);
    tm_set_f32(t, 0, -2.0f);
    tm_set_f32(t, 1, -0.5f);
    tm_set_f32(t, 2, 0.0f);
    tm_set_f32(t, 3, 3.0f);

    tm_tensor* r = tm_relu(t);

    ASSERT_FLT_EQ(0.0f, tm_get_f32(r, 0));
    ASSERT_FLT_EQ(0.0f, tm_get_f32(r, 1));
    ASSERT_FLT_EQ(0.0f, tm_get_f32(r, 2));
    ASSERT_FLT_EQ(3.0f, tm_get_f32(r, 3));

    tm_free(t);
    tm_free(r);
}

/* ============================================================= */
/* Matrix Op Tests                                               */
/* ============================================================= */

static void test_matmul_2d(void) {
    /* A = [[1, 2], [3, 4]] (2x2) */
    /* B = [[5, 6], [7, 8]] (2x2) */
    /* C = A @ B = [[19, 22], [43, 50]] */

    int64_t shape[] = {2, 2};
    tm_tensor* a = tm_zeros(shape, 2, TM_F32);
    tm_tensor* b = tm_zeros(shape, 2, TM_F32);

    float* ad = tm_data_f32(a);
    ad[0] = 1; ad[1] = 2; ad[2] = 3; ad[3] = 4;

    float* bd = tm_data_f32(b);
    bd[0] = 5; bd[1] = 6; bd[2] = 7; bd[3] = 8;

    tm_tensor* c = tm_matmul(a, b);

    ASSERT_NOT_NULL(c);
    ASSERT_INT_EQ(2, tm_ndim(c));
    ASSERT_INT_EQ(2, tm_size(c, 0));
    ASSERT_INT_EQ(2, tm_size(c, 1));

    float* cd = tm_data_f32(c);
    ASSERT_FLT_EQ(19.0f, cd[0]);
    ASSERT_FLT_EQ(22.0f, cd[1]);
    ASSERT_FLT_EQ(43.0f, cd[2]);
    ASSERT_FLT_EQ(50.0f, cd[3]);

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_matmul_different_shapes(void) {
    /* A = (3, 4), B = (4, 2) -> C = (3, 2) */
    int64_t shape_a[] = {3, 4};
    int64_t shape_b[] = {4, 2};

    tm_tensor* a = tm_ones(shape_a, 2, TM_F32);
    tm_tensor* b = tm_ones(shape_b, 2, TM_F32);

    tm_tensor* c = tm_matmul(a, b);

    ASSERT_NOT_NULL(c);
    ASSERT_INT_EQ(3, tm_size(c, 0));
    ASSERT_INT_EQ(2, tm_size(c, 1));

    /* Each element should be 4 (sum of 4 ones) */
    for (int64_t i = 0; i < 6; i++) {
        ASSERT_FLT_EQ(4.0f, tm_get_f32(c, i));
    }

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

/* ============================================================= */
/* Reduction Tests                                               */
/* ============================================================= */

static void test_sum_all(void) {
    int64_t shape[] = {2, 3};
    tm_tensor* t = tm_ones(shape, 2, TM_F32);

    tm_tensor* s = tm_sum(t, -1, false);

    ASSERT_NOT_NULL(s);
    ASSERT_INT_EQ(1, tm_numel(s));
    ASSERT_FLT_EQ(6.0f, tm_get_f32(s, 0));

    tm_free(t);
    tm_free(s);
}

static void test_mean_all(void) {
    int64_t shape[] = {4};
    tm_tensor* t = tm_zeros(shape, 1, TM_F32);
    tm_set_f32(t, 0, 1.0f);
    tm_set_f32(t, 1, 2.0f);
    tm_set_f32(t, 2, 3.0f);
    tm_set_f32(t, 3, 4.0f);

    tm_tensor* m = tm_mean(t, -1, false);

    ASSERT_NOT_NULL(m);
    ASSERT_FLT_EQ(2.5f, tm_get_f32(m, 0));

    tm_free(t);
    tm_free(m);
}

/* ============================================================= */
/* NN Op Tests                                                   */
/* ============================================================= */

static void test_softmax(void) {
    int64_t shape[] = {1, 3};
    tm_tensor* t = tm_zeros(shape, 2, TM_F32);
    tm_set_f32(t, 0, 1.0f);
    tm_set_f32(t, 1, 2.0f);
    tm_set_f32(t, 2, 3.0f);

    tm_tensor* s = tm_softmax(t, 1);

    ASSERT_NOT_NULL(s);

    /* Softmax values should sum to 1 */
    float sum = tm_get_f32(s, 0) + tm_get_f32(s, 1) + tm_get_f32(s, 2);
    ASSERT_FLT_NEAR(1.0f, sum, 0.001f);

    /* Values should be in increasing order */
    ASSERT_TRUE(tm_get_f32(s, 0) < tm_get_f32(s, 1));
    ASSERT_TRUE(tm_get_f32(s, 1) < tm_get_f32(s, 2));

    tm_free(t);
    tm_free(s);
}

static void test_layer_norm(void) {
    int64_t shape[] = {2, 4};
    tm_tensor* t = tm_zeros(shape, 2, TM_F32);
    float* data = tm_data_f32(t);
    /* Row 1: [1, 2, 3, 4] */
    data[0] = 1; data[1] = 2; data[2] = 3; data[3] = 4;
    /* Row 2: [4, 4, 4, 4] */
    data[4] = 4; data[5] = 4; data[6] = 4; data[7] = 4;

    tm_tensor* ln = tm_layer_norm(t, NULL, NULL, 1e-5f);

    ASSERT_NOT_NULL(ln);

    /* Normalized values should have mean ~0 and std ~1 per row */
    float* lnd = tm_data_f32(ln);

    /* Check row 1 mean */
    float mean = (lnd[0] + lnd[1] + lnd[2] + lnd[3]) / 4.0f;
    ASSERT_FLT_NEAR(0.0f, mean, 0.01f);

    /* Row 2 should be all zeros (constant input) */
    ASSERT_FLT_NEAR(0.0f, lnd[4], 0.01f);
    ASSERT_FLT_NEAR(0.0f, lnd[5], 0.01f);

    tm_free(t);
    tm_free(ln);
}

/* ============================================================= */
/* Autograd Tests                                                */
/* ============================================================= */

static void test_autograd_add(void) {
    int64_t shape[] = {3};
    tm_tensor* a = tm_ones(shape, 1, TM_F32);
    tm_tensor* b = tm_full(shape, 1, TM_F32, 2.0);

    tm_set_requires_grad(a, true);
    tm_set_requires_grad(b, true);

    tm_tensor* c = tm_add(a, b);
    tm_tensor* loss = tm_sum(c, -1, false);

    ASSERT_TRUE(loss->requires_grad);

    tm_backward(loss);

    /* Gradient of sum(a + b) w.r.t a and b is all ones */
    ASSERT_NOT_NULL(a->grad);
    ASSERT_NOT_NULL(b->grad);

    for (int64_t i = 0; i < 3; i++) {
        ASSERT_FLT_EQ(1.0f, tm_get_f32(a->grad, i));
        ASSERT_FLT_EQ(1.0f, tm_get_f32(b->grad, i));
    }

    tm_free(a);
    tm_free(b);
    tm_free(c);
    tm_free(loss);
}

static void test_autograd_mul(void) {
    int64_t shape[] = {2};
    tm_tensor* a = tm_full(shape, 1, TM_F32, 3.0);
    tm_tensor* b = tm_full(shape, 1, TM_F32, 4.0);

    tm_set_requires_grad(a, true);
    tm_set_requires_grad(b, true);

    tm_tensor* c = tm_mul(a, b);
    tm_tensor* loss = tm_sum(c, -1, false);

    tm_backward(loss);

    /* d(sum(a*b))/da = b, d(sum(a*b))/db = a */
    for (int64_t i = 0; i < 2; i++) {
        ASSERT_FLT_EQ(4.0f, tm_get_f32(a->grad, i));  /* grad = b */
        ASSERT_FLT_EQ(3.0f, tm_get_f32(b->grad, i));  /* grad = a */
    }

    tm_free(a);
    tm_free(b);
    tm_free(c);
    tm_free(loss);
}

static void test_no_grad_context(void) {
    int64_t shape[] = {2};
    tm_tensor* a = tm_ones(shape, 1, TM_F32);
    tm_tensor* b = tm_ones(shape, 1, TM_F32);
    tm_set_requires_grad(a, true);
    tm_set_requires_grad(b, true);

    bool prev = tm_no_grad_begin();

    tm_tensor* c = tm_add(a, b);

    /* In no_grad context, output should not require grad */
    ASSERT_FALSE(c->requires_grad);

    tm_no_grad_end(prev);

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

/* ============================================================= */
/* Edge Case Tests                                               */
/* ============================================================= */

static void test_null_safety(void) {
    ASSERT_NULL(tm_add(NULL, NULL));
    ASSERT_NULL(tm_matmul(NULL, NULL));
    ASSERT_NULL(tm_reshape(NULL, NULL, 0));
    ASSERT_INT_EQ(0, tm_numel(NULL));
    ASSERT_INT_EQ(0, tm_ndim(NULL));

    tm_free(NULL);  /* Should not crash */
    ASSERT_TRUE(1);
}

static void test_scalar_tensor(void) {
    int64_t shape[] = {1};
    tm_tensor* t = tm_full(shape, 1, TM_F32, 42.0);

    ASSERT_INT_EQ(1, tm_numel(t));
    ASSERT_FLT_EQ(42.0f, tm_get_f32(t, 0));

    tm_free(t);
}

/* ============================================================= */
/* Broadcasting Tests                                            */
/* ============================================================= */

static void test_broadcast_add_1d_to_2d(void) {
    /* Test [4, 3] + [3] -> [4, 3] */
    int64_t shape_a[] = {4, 3};
    int64_t shape_b[] = {3};

    tm_tensor* a = tm_ones(shape_a, 2, TM_F32);
    tm_tensor* b = tm_full(shape_b, 1, TM_F32, 2.0);

    tm_tensor* c = tm_add(a, b);

    ASSERT_NOT_NULL(c);
    ASSERT_INT_EQ(2, tm_ndim(c));
    ASSERT_INT_EQ(4, tm_size(c, 0));
    ASSERT_INT_EQ(3, tm_size(c, 1));

    /* All elements should be 3.0 (1.0 + 2.0) */
    for (int64_t i = 0; i < 12; i++) {
        ASSERT_FLT_NEAR(3.0f, tm_get_f32(c, i), 0.0001f);
    }

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_broadcast_mul_scalar_like(void) {
    /* Test [4, 3] * [1] -> [4, 3] */
    int64_t shape_a[] = {4, 3};
    int64_t shape_b[] = {1};

    tm_tensor* a = tm_full(shape_a, 2, TM_F32, 3.0);
    tm_tensor* b = tm_full(shape_b, 1, TM_F32, 2.0);

    tm_tensor* c = tm_mul(a, b);

    ASSERT_NOT_NULL(c);
    ASSERT_INT_EQ(12, tm_numel(c));

    for (int64_t i = 0; i < 12; i++) {
        ASSERT_FLT_NEAR(6.0f, tm_get_f32(c, i), 0.0001f);
    }

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_broadcast_sub(void) {
    /* Test [2, 3] - [3] -> [2, 3] */
    int64_t shape_a[] = {2, 3};
    int64_t shape_b[] = {3};

    tm_tensor* a = tm_full(shape_a, 2, TM_F32, 5.0);
    tm_tensor* b = tm_zeros(shape_b, 1, TM_F32);
    float* bd = tm_data_f32(b);
    bd[0] = 1.0f; bd[1] = 2.0f; bd[2] = 3.0f;

    tm_tensor* c = tm_sub(a, b);

    ASSERT_NOT_NULL(c);
    float* cd = tm_data_f32(c);
    /* Row 0: 5-1=4, 5-2=3, 5-3=2 */
    ASSERT_FLT_NEAR(4.0f, cd[0], 0.0001f);
    ASSERT_FLT_NEAR(3.0f, cd[1], 0.0001f);
    ASSERT_FLT_NEAR(2.0f, cd[2], 0.0001f);
    /* Row 1: same pattern */
    ASSERT_FLT_NEAR(4.0f, cd[3], 0.0001f);
    ASSERT_FLT_NEAR(3.0f, cd[4], 0.0001f);
    ASSERT_FLT_NEAR(2.0f, cd[5], 0.0001f);

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_broadcast_div(void) {
    /* Test [2, 2] / [2] -> [2, 2] */
    int64_t shape_a[] = {2, 2};
    int64_t shape_b[] = {2};

    tm_tensor* a = tm_full(shape_a, 2, TM_F32, 6.0);
    tm_tensor* b = tm_zeros(shape_b, 1, TM_F32);
    float* bd = tm_data_f32(b);
    bd[0] = 2.0f; bd[1] = 3.0f;

    tm_tensor* c = tm_div(a, b);

    ASSERT_NOT_NULL(c);
    float* cd = tm_data_f32(c);
    /* 6/2=3, 6/3=2 for each row */
    ASSERT_FLT_NEAR(3.0f, cd[0], 0.0001f);
    ASSERT_FLT_NEAR(2.0f, cd[1], 0.0001f);
    ASSERT_FLT_NEAR(3.0f, cd[2], 0.0001f);
    ASSERT_FLT_NEAR(2.0f, cd[3], 0.0001f);

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_broadcast_backward(void) {
    /* Test that gradients are summed correctly for broadcast */
    int64_t shape_a[] = {2, 3};
    int64_t shape_b[] = {3};

    tm_tensor* a = tm_ones(shape_a, 2, TM_F32);
    tm_tensor* b = tm_full(shape_b, 1, TM_F32, 2.0);
    tm_set_requires_grad(a, true);
    tm_set_requires_grad(b, true);

    tm_tensor* c = tm_add(a, b);

    /* Create gradient for c (all ones) */
    c->grad = tm_ones(c->shape, c->ndim, TM_F32);

    /* Call backward */
    tm_backward(c);

    /* a's gradient should be all 1s (same shape as a) */
    ASSERT_NOT_NULL(a->grad);
    for (int64_t i = 0; i < 6; i++) {
        ASSERT_FLT_NEAR(1.0f, tm_get_f32(a->grad, i), 0.0001f);
    }

    /* b's gradient should be summed over the broadcast dim (2 rows) */
    ASSERT_NOT_NULL(b->grad);
    for (int64_t i = 0; i < 3; i++) {
        ASSERT_FLT_NEAR(2.0f, tm_get_f32(b->grad, i), 0.0001f);
    }

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

/* ============================================================= */
/* P0 Feature Tests                                              */
/* ============================================================= */

static void test_arange(void) {
    tm_tensor* t = tm_arange(0, 5, 1, TM_F32);
    ASSERT_NOT_NULL(t);
    ASSERT_INT_EQ(5, tm_numel(t));
    ASSERT_FLT_EQ(0.0f, tm_get_f32(t, 0));
    ASSERT_FLT_EQ(4.0f, tm_get_f32(t, 4));
    tm_free(t);

    /* Negative step */
    tm_tensor* t2 = tm_arange(5, 0, -1, TM_F32);
    ASSERT_NOT_NULL(t2);
    ASSERT_INT_EQ(5, tm_numel(t2));
    ASSERT_FLT_EQ(5.0f, tm_get_f32(t2, 0));
    ASSERT_FLT_EQ(1.0f, tm_get_f32(t2, 4));
    tm_free(t2);
}

static void test_cat(void) {
    int64_t shape[] = {2, 3};
    tm_tensor* a = tm_ones(shape, 2, TM_F32);
    tm_tensor* b = tm_full(shape, 2, TM_F32, 2.0);

    tm_tensor* tensors[] = {a, b};
    tm_tensor* c = tm_cat(tensors, 2, 0);

    ASSERT_NOT_NULL(c);
    ASSERT_INT_EQ(2, tm_ndim(c));
    ASSERT_INT_EQ(4, tm_size(c, 0));  /* 2 + 2 */
    ASSERT_INT_EQ(3, tm_size(c, 1));

    /* First half should be 1s, second half 2s */
    ASSERT_FLT_EQ(1.0f, tm_get_f32(c, 0));
    ASSERT_FLT_EQ(2.0f, tm_get_f32(c, 6));

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_stack(void) {
    int64_t shape[] = {2, 3};
    tm_tensor* a = tm_ones(shape, 2, TM_F32);
    tm_tensor* b = tm_full(shape, 2, TM_F32, 2.0);

    tm_tensor* tensors[] = {a, b};
    tm_tensor* c = tm_stack(tensors, 2, 0);

    ASSERT_NOT_NULL(c);
    ASSERT_INT_EQ(3, tm_ndim(c));
    ASSERT_INT_EQ(2, tm_size(c, 0));  /* Stack dimension */
    ASSERT_INT_EQ(2, tm_size(c, 1));
    ASSERT_INT_EQ(3, tm_size(c, 2));

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_split(void) {
    int64_t shape[] = {6, 3};
    tm_tensor* t = tm_arange(0, 18, 1, TM_F32);
    t = tm_reshape(t, shape, 2);

    int num_chunks;
    tm_tensor** chunks = tm_split(t, 2, 0, &num_chunks);

    ASSERT_NOT_NULL(chunks);
    ASSERT_INT_EQ(3, num_chunks);

    for (int i = 0; i < num_chunks; i++) {
        ASSERT_NOT_NULL(chunks[i]);
        ASSERT_INT_EQ(2, tm_size(chunks[i], 0));
        ASSERT_INT_EQ(3, tm_size(chunks[i], 1));
        tm_free(chunks[i]);
    }
    free(chunks);
    tm_free(t);
}

static void test_permute(void) {
    int64_t shape[] = {2, 3, 4};
    tm_tensor* t = tm_randn(shape, 3, TM_F32);

    int dims[] = {1, 2, 0};
    tm_tensor* p = tm_permute(t, dims, 3);

    ASSERT_NOT_NULL(p);
    ASSERT_INT_EQ(3, tm_size(p, 0));
    ASSERT_INT_EQ(4, tm_size(p, 1));
    ASSERT_INT_EQ(2, tm_size(p, 2));

    tm_free(t);
    tm_free(p);
}

static void test_flatten(void) {
    int64_t shape[] = {2, 3, 4};
    tm_tensor* t = tm_randn(shape, 3, TM_F32);

    tm_tensor* f = tm_flatten(t, 0, -1);
    ASSERT_NOT_NULL(f);
    ASSERT_INT_EQ(1, tm_ndim(f));
    ASSERT_INT_EQ(24, tm_numel(f));

    tm_free(t);
    tm_free(f);
}

static void test_select(void) {
    int64_t shape[] = {3, 4};
    tm_tensor* t = tm_arange(0, 12, 1, TM_F32);
    t = tm_reshape(t, shape, 2);

    tm_tensor* s = tm_select(t, 0, 1);  /* Select row 1 */
    ASSERT_NOT_NULL(s);
    ASSERT_INT_EQ(1, tm_ndim(s));
    ASSERT_INT_EQ(4, tm_numel(s));
    ASSERT_FLT_EQ(4.0f, tm_get_f32(s, 0));
    ASSERT_FLT_EQ(5.0f, tm_get_f32(s, 1));

    tm_free(t);
    tm_free(s);
}

static void test_silu(void) {
    int64_t shape[] = {4};
    float data[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* out = tm_silu(t);
    ASSERT_NOT_NULL(out);

    /* SiLU(0) = 0 */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(out, 1), 0.0001f);
    /* SiLU(x) = x * sigmoid(x) */
    float sig1 = 1.0f / (1.0f + expf(-1.0f));
    ASSERT_FLT_NEAR(1.0f * sig1, tm_get_f32(out, 2), 0.0001f);

    tm_free(t);
    tm_free(out);
}

static void test_rms_norm(void) {
    int64_t shape[] = {2, 4};
    tm_tensor* t = tm_ones(shape, 2, TM_F32);

    /* Without weight */
    tm_tensor* out = tm_rms_norm(t, NULL, 1e-5f);
    ASSERT_NOT_NULL(out);
    /* RMS of all 1s is 1, so normalized should be ~1 */
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(out, 0), 0.01f);

    tm_free(t);
    tm_free(out);
}

static void test_linear(void) {
    /* x: [2, 3], weight: [4, 3], bias: [4] -> out: [2, 4] */
    int64_t x_shape[] = {2, 3};
    int64_t w_shape[] = {4, 3};
    int64_t b_shape[] = {4};

    tm_tensor* x = tm_ones(x_shape, 2, TM_F32);
    tm_tensor* w = tm_ones(w_shape, 2, TM_F32);
    tm_tensor* b = tm_full(b_shape, 1, TM_F32, 1.0);

    tm_tensor* out = tm_linear(x, w, b);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(2, tm_size(out, 0));
    ASSERT_INT_EQ(4, tm_size(out, 1));
    /* Each output = sum(1*1 * 3) + 1 = 4 */
    ASSERT_FLT_NEAR(4.0f, tm_get_f32(out, 0), 0.0001f);

    tm_free(x);
    tm_free(w);
    tm_free(b);
    tm_free(out);
}

static void test_triu(void) {
    int64_t shape[] = {3, 3};
    tm_tensor* t = tm_ones(shape, 2, TM_F32);

    tm_tensor* upper = tm_triu(t, 0);
    ASSERT_NOT_NULL(upper);

    /* Lower triangle should be 0 */
    float* d = tm_data_f32(upper);
    ASSERT_FLT_EQ(0.0f, d[3]);  /* (1,0) */
    ASSERT_FLT_EQ(0.0f, d[6]);  /* (2,0) */
    ASSERT_FLT_EQ(0.0f, d[7]);  /* (2,1) */
    /* Upper triangle should be 1 */
    ASSERT_FLT_EQ(1.0f, d[0]);  /* (0,0) */
    ASSERT_FLT_EQ(1.0f, d[2]);  /* (0,2) */

    tm_free(t);
    tm_free(upper);
}

static void test_sin_cos(void) {
    int64_t shape[] = {4};
    float data[] = {0.0f, 3.14159f/2, 3.14159f, 3.14159f*2};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* s = tm_sin(t);
    tm_tensor* c = tm_cos(t);
    ASSERT_NOT_NULL(s);
    ASSERT_NOT_NULL(c);

    ASSERT_FLT_NEAR(0.0f, tm_get_f32(s, 0), 0.001f);
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(s, 1), 0.001f);
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(c, 0), 0.001f);
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(c, 1), 0.001f);

    tm_free(t);
    tm_free(s);
    tm_free(c);
}

static void test_clamp(void) {
    int64_t shape[] = {5};
    float data[] = {-2.0f, -0.5f, 0.5f, 1.5f, 3.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* out = tm_clamp(t, 0.0, 1.0);
    ASSERT_NOT_NULL(out);

    ASSERT_FLT_EQ(0.0f, tm_get_f32(out, 0));
    ASSERT_FLT_EQ(0.0f, tm_get_f32(out, 1));
    ASSERT_FLT_EQ(0.5f, tm_get_f32(out, 2));
    ASSERT_FLT_EQ(1.0f, tm_get_f32(out, 3));
    ASSERT_FLT_EQ(1.0f, tm_get_f32(out, 4));

    tm_free(t);
    tm_free(out);
}

static void test_argmax(void) {
    int64_t shape[] = {2, 3};
    float data[] = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f, 6.0f};
    tm_tensor* t = tm_from_data(data, shape, 2, TM_F32);
    t->owns_data = false;

    tm_tensor* am = tm_argmax(t, -1, false);
    ASSERT_NOT_NULL(am);
    ASSERT_INT_EQ(1, tm_numel(am));

    /* Global argmax is at index 5 (value 6.0) */
    int64_t* idx = (int64_t*)am->data;
    ASSERT_INT_EQ(5, idx[0]);

    tm_free(t);
    tm_free(am);
}

static void test_logsumexp(void) {
    int64_t shape[] = {3};
    float data[] = {1.0f, 2.0f, 3.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* lse = tm_logsumexp(t, -1, false);
    ASSERT_NOT_NULL(lse);

    /* logsumexp([1,2,3]) = log(e^1 + e^2 + e^3) ≈ 3.407 */
    float expected = logf(expf(1.0f) + expf(2.0f) + expf(3.0f));
    ASSERT_FLT_NEAR(expected, tm_get_f32(lse, 0), 0.001f);

    tm_free(t);
    tm_free(lse);
}

static void test_abs(void) {
    int64_t shape[] = {4};
    float data[] = {-1.0f, 0.0f, 1.0f, -2.5f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* out = tm_abs(t);
    ASSERT_NOT_NULL(out);

    ASSERT_FLT_EQ(1.0f, tm_get_f32(out, 0));
    ASSERT_FLT_EQ(0.0f, tm_get_f32(out, 1));
    ASSERT_FLT_EQ(1.0f, tm_get_f32(out, 2));
    ASSERT_FLT_EQ(2.5f, tm_get_f32(out, 3));

    tm_free(t);
    tm_free(out);
}

static void test_pow(void) {
    int64_t shape[] = {3};
    float data[] = {2.0f, 3.0f, 4.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* out = tm_pow(t, 2.0);
    ASSERT_NOT_NULL(out);

    ASSERT_FLT_EQ(4.0f, tm_get_f32(out, 0));
    ASSERT_FLT_EQ(9.0f, tm_get_f32(out, 1));
    ASSERT_FLT_EQ(16.0f, tm_get_f32(out, 2));

    tm_free(t);
    tm_free(out);
}

static void test_where(void) {
    int64_t shape[] = {4};
    float cond_data[] = {1.0f, 0.0f, 1.0f, 0.0f};
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};

    tm_tensor* cond = tm_from_data(cond_data, shape, 1, TM_F32);
    tm_tensor* a = tm_from_data(a_data, shape, 1, TM_F32);
    tm_tensor* b = tm_from_data(b_data, shape, 1, TM_F32);
    cond->owns_data = false;
    a->owns_data = false;
    b->owns_data = false;

    tm_tensor* out = tm_where(cond, a, b);
    ASSERT_NOT_NULL(out);

    ASSERT_FLT_EQ(1.0f, tm_get_f32(out, 0));   /* cond=true, pick a */
    ASSERT_FLT_EQ(20.0f, tm_get_f32(out, 1));  /* cond=false, pick b */
    ASSERT_FLT_EQ(3.0f, tm_get_f32(out, 2));
    ASSERT_FLT_EQ(40.0f, tm_get_f32(out, 3));

    tm_free(cond);
    tm_free(a);
    tm_free(b);
    tm_free(out);
}

static void test_masked_fill(void) {
    int64_t shape[] = {4};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float mask_data[] = {0.0f, 1.0f, 0.0f, 1.0f};

    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    tm_tensor* mask = tm_from_data(mask_data, shape, 1, TM_F32);
    t->owns_data = false;
    mask->owns_data = false;

    tm_tensor* out = tm_masked_fill(t, mask, -99.0f);
    ASSERT_NOT_NULL(out);

    ASSERT_FLT_EQ(1.0f, tm_get_f32(out, 0));
    ASSERT_FLT_EQ(-99.0f, tm_get_f32(out, 1));
    ASSERT_FLT_EQ(3.0f, tm_get_f32(out, 2));
    ASSERT_FLT_EQ(-99.0f, tm_get_f32(out, 3));

    tm_free(t);
    tm_free(mask);
    tm_free(out);
}

static void test_eye(void) {
    tm_tensor* t = tm_eye(3, TM_F32);
    ASSERT_NOT_NULL(t);
    ASSERT_INT_EQ(2, tm_ndim(t));
    ASSERT_INT_EQ(3, tm_size(t, 0));
    ASSERT_INT_EQ(3, tm_size(t, 1));

    float* d = tm_data_f32(t);
    ASSERT_FLT_EQ(1.0f, d[0]);
    ASSERT_FLT_EQ(0.0f, d[1]);
    ASSERT_FLT_EQ(1.0f, d[4]);
    ASSERT_FLT_EQ(1.0f, d[8]);

    tm_free(t);
}

static void test_rsqrt(void) {
    int64_t shape[] = {3};
    float data[] = {1.0f, 4.0f, 9.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* out = tm_rsqrt(t);
    ASSERT_NOT_NULL(out);

    ASSERT_FLT_NEAR(1.0f, tm_get_f32(out, 0), 0.0001f);
    ASSERT_FLT_NEAR(0.5f, tm_get_f32(out, 1), 0.0001f);
    ASSERT_FLT_NEAR(1.0f/3.0f, tm_get_f32(out, 2), 0.0001f);

    tm_free(t);
    tm_free(out);
}

static void test_sub_div_scalar(void) {
    int64_t shape[] = {3};
    float data[] = {10.0f, 20.0f, 30.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* sub_out = tm_sub_scalar(t, 5.0);
    ASSERT_NOT_NULL(sub_out);
    ASSERT_FLT_EQ(5.0f, tm_get_f32(sub_out, 0));
    ASSERT_FLT_EQ(15.0f, tm_get_f32(sub_out, 1));

    tm_tensor* div_out = tm_div_scalar(t, 10.0);
    ASSERT_NOT_NULL(div_out);
    ASSERT_FLT_EQ(1.0f, tm_get_f32(div_out, 0));
    ASSERT_FLT_EQ(2.0f, tm_get_f32(div_out, 1));

    tm_free(t);
    tm_free(sub_out);
    tm_free(div_out);
}

static void test_inplace_ops(void) {
    int64_t shape[] = {4};
    tm_tensor* t = tm_full(shape, 1, TM_F32, 5.0);

    /* Test zero_ */
    tm_zero_(t);
    ASSERT_FLT_EQ(0.0f, tm_get_f32(t, 0));

    /* Test clamp_ */
    tm_fill_(t, 10.0f);
    tm_clamp_(t, 0.0f, 5.0f);
    ASSERT_FLT_EQ(5.0f, tm_get_f32(t, 0));

    /* Test copy_ */
    tm_tensor* src = tm_full(shape, 1, TM_F32, 42.0);
    tm_copy_(t, src);
    ASSERT_FLT_EQ(42.0f, tm_get_f32(t, 0));

    tm_free(t);
    tm_free(src);
}

/* ============================================================= */
/* P1/P2 Feature Tests                                           */
/* ============================================================= */

static void test_comparison_ops(void) {
    int64_t shape[] = {4};
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {1.0f, 3.0f, 2.0f, 4.0f};

    tm_tensor* a = tm_from_data(a_data, shape, 1, TM_F32);
    tm_tensor* b = tm_from_data(b_data, shape, 1, TM_F32);
    a->owns_data = false;
    b->owns_data = false;

    /* Test eq */
    tm_tensor* eq = tm_eq(a, b);
    ASSERT_NOT_NULL(eq);
    int32_t* eq_data = (int32_t*)eq->data;
    ASSERT_INT_EQ(1, eq_data[0]);  /* 1 == 1 */
    ASSERT_INT_EQ(0, eq_data[1]);  /* 2 != 3 */
    ASSERT_INT_EQ(1, eq_data[3]);  /* 4 == 4 */

    /* Test lt */
    tm_tensor* lt = tm_lt(a, b);
    int32_t* lt_data = (int32_t*)lt->data;
    ASSERT_INT_EQ(0, lt_data[0]);  /* 1 < 1 = false */
    ASSERT_INT_EQ(1, lt_data[1]);  /* 2 < 3 = true */
    ASSERT_INT_EQ(0, lt_data[2]);  /* 3 < 2 = false */

    /* Test gt */
    tm_tensor* gt = tm_gt(a, b);
    int32_t* gt_data = (int32_t*)gt->data;
    ASSERT_INT_EQ(0, gt_data[1]);  /* 2 > 3 = false */
    ASSERT_INT_EQ(1, gt_data[2]);  /* 3 > 2 = true */

    tm_free(a);
    tm_free(b);
    tm_free(eq);
    tm_free(lt);
    tm_free(gt);
}

static void test_maximum_minimum(void) {
    int64_t shape[] = {4};
    float a_data[] = {1.0f, 5.0f, 3.0f, 2.0f};
    float b_data[] = {2.0f, 3.0f, 4.0f, 1.0f};

    tm_tensor* a = tm_from_data(a_data, shape, 1, TM_F32);
    tm_tensor* b = tm_from_data(b_data, shape, 1, TM_F32);
    a->owns_data = false;
    b->owns_data = false;

    tm_tensor* max_t = tm_maximum(a, b);
    ASSERT_NOT_NULL(max_t);
    ASSERT_FLT_EQ(2.0f, tm_get_f32(max_t, 0));
    ASSERT_FLT_EQ(5.0f, tm_get_f32(max_t, 1));
    ASSERT_FLT_EQ(4.0f, tm_get_f32(max_t, 2));

    tm_tensor* min_t = tm_minimum(a, b);
    ASSERT_NOT_NULL(min_t);
    ASSERT_FLT_EQ(1.0f, tm_get_f32(min_t, 0));
    ASSERT_FLT_EQ(3.0f, tm_get_f32(min_t, 1));
    ASSERT_FLT_EQ(3.0f, tm_get_f32(min_t, 2));

    tm_free(a);
    tm_free(b);
    tm_free(max_t);
    tm_free(min_t);
}

static void test_leaky_relu(void) {
    int64_t shape[] = {4};
    float data[] = {-2.0f, -1.0f, 1.0f, 2.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* out = tm_leaky_relu(t, 0.1f);
    ASSERT_NOT_NULL(out);
    ASSERT_FLT_NEAR(-0.2f, tm_get_f32(out, 0), 0.0001f);
    ASSERT_FLT_NEAR(-0.1f, tm_get_f32(out, 1), 0.0001f);
    ASSERT_FLT_EQ(1.0f, tm_get_f32(out, 2));
    ASSERT_FLT_EQ(2.0f, tm_get_f32(out, 3));

    tm_free(t);
    tm_free(out);
}

static void test_gather(void) {
    int64_t shape[] = {3, 4};
    tm_tensor* t = tm_arange(0, 12, 1, TM_F32);
    t = tm_reshape(t, shape, 2);

    /* Index tensor for gathering along dim 1 */
    int64_t idx_shape[] = {3, 2};
    int32_t idx_data[] = {0, 2, 1, 3, 0, 1};
    tm_tensor* idx = tm_from_data(idx_data, idx_shape, 2, TM_I32);
    idx->owns_data = false;

    tm_tensor* out = tm_gather(t, 1, idx);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(3, tm_size(out, 0));
    ASSERT_INT_EQ(2, tm_size(out, 1));

    /* Row 0: indices [0,2] -> values [0, 2] */
    ASSERT_FLT_EQ(0.0f, tm_get_f32(out, 0));
    ASSERT_FLT_EQ(2.0f, tm_get_f32(out, 1));

    tm_free(t);
    tm_free(idx);
    tm_free(out);
}

static void test_index_select(void) {
    int64_t shape[] = {4, 3};
    tm_tensor* t = tm_arange(0, 12, 1, TM_F32);
    t = tm_reshape(t, shape, 2);

    int64_t idx_shape[] = {2};
    int32_t idx_data[] = {0, 2};
    tm_tensor* idx = tm_from_data(idx_data, idx_shape, 1, TM_I32);
    idx->owns_data = false;

    tm_tensor* out = tm_index_select(t, 0, idx);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(2, tm_size(out, 0));
    ASSERT_INT_EQ(3, tm_size(out, 1));

    /* Should get rows 0 and 2 */
    ASSERT_FLT_EQ(0.0f, tm_get_f32(out, 0));  /* row 0, col 0 */
    ASSERT_FLT_EQ(6.0f, tm_get_f32(out, 3));  /* row 2, col 0 */

    tm_free(t);
    tm_free(idx);
    tm_free(out);
}

static void test_narrow(void) {
    int64_t shape[] = {4, 5};
    tm_tensor* t = tm_arange(0, 20, 1, TM_F32);
    t = tm_reshape(t, shape, 2);

    tm_tensor* out = tm_narrow(t, 0, 1, 2);  /* Rows 1-2 */
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(2, tm_size(out, 0));
    ASSERT_INT_EQ(5, tm_size(out, 1));
    ASSERT_FLT_EQ(5.0f, tm_get_f32(out, 0));  /* First element of row 1 */

    tm_free(t);
    tm_free(out);
}

static void test_expand(void) {
    int64_t shape[] = {1, 3};
    float data[] = {1.0f, 2.0f, 3.0f};
    tm_tensor* t = tm_from_data(data, shape, 2, TM_F32);
    t->owns_data = false;

    int64_t new_shape[] = {4, 3};
    tm_tensor* out = tm_expand(t, new_shape, 2);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(4, tm_size(out, 0));
    ASSERT_INT_EQ(3, tm_size(out, 1));

    /* All rows should have the same values */
    ASSERT_FLT_EQ(1.0f, tm_get_f32(out, 0));
    ASSERT_FLT_EQ(1.0f, tm_get_f32(out, 3));
    ASSERT_FLT_EQ(2.0f, tm_get_f32(out, 1));
    ASSERT_FLT_EQ(2.0f, tm_get_f32(out, 4));

    tm_free(t);
    tm_free(out);
}

static void test_repeat(void) {
    int64_t shape[] = {2, 3};
    tm_tensor* t = tm_arange(0, 6, 1, TM_F32);
    t = tm_reshape(t, shape, 2);

    int64_t repeats[] = {2, 1};
    tm_tensor* out = tm_repeat(t, repeats, 2);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(4, tm_size(out, 0));
    ASSERT_INT_EQ(3, tm_size(out, 1));

    /* Row 0 and row 2 should be the same */
    ASSERT_FLT_EQ(0.0f, tm_get_f32(out, 0));
    ASSERT_FLT_EQ(0.0f, tm_get_f32(out, 6));

    tm_free(t);
    tm_free(out);
}

static void test_cumsum(void) {
    int64_t shape[] = {3};
    float data[] = {1.0f, 2.0f, 3.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* out = tm_cumsum(t, 0);
    ASSERT_NOT_NULL(out);
    ASSERT_FLT_EQ(1.0f, tm_get_f32(out, 0));
    ASSERT_FLT_EQ(3.0f, tm_get_f32(out, 1));
    ASSERT_FLT_EQ(6.0f, tm_get_f32(out, 2));

    tm_free(t);
    tm_free(out);
}

static void test_var_std(void) {
    int64_t shape[] = {4};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};  /* mean = 2.5 */
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    /* Variance (unbiased): sum((x - mean)^2) / (n-1) */
    /* (1.5^2 + 0.5^2 + 0.5^2 + 1.5^2) / 3 = 5/3 ≈ 1.667 */
    tm_tensor* var = tm_var(t, -1, false, true);
    ASSERT_NOT_NULL(var);
    ASSERT_FLT_NEAR(1.6667f, tm_get_f32(var, 0), 0.01f);

    tm_tensor* std = tm_std(t, -1, false, true);
    ASSERT_NOT_NULL(std);
    ASSERT_FLT_NEAR(sqrtf(5.0f/3.0f), tm_get_f32(std, 0), 0.01f);

    tm_free(t);
    tm_free(var);
    tm_free(std);
}

static void test_prod(void) {
    int64_t shape[] = {4};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* out = tm_prod(t, -1, false);
    ASSERT_NOT_NULL(out);
    ASSERT_FLT_EQ(24.0f, tm_get_f32(out, 0));  /* 1*2*3*4 = 24 */

    tm_free(t);
    tm_free(out);
}

static void test_type_conversion(void) {
    int64_t shape[] = {3};
    float data[] = {1.5f, 2.7f, 3.9f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    /* Convert to int (truncates) */
    tm_tensor* int_t = tm_int(t);
    ASSERT_NOT_NULL(int_t);
    ASSERT_INT_EQ(TM_I32, int_t->dtype);
    int32_t* int_data = (int32_t*)int_t->data;
    ASSERT_INT_EQ(1, int_data[0]);
    ASSERT_INT_EQ(2, int_data[1]);
    ASSERT_INT_EQ(3, int_data[2]);

    /* Convert to double */
    tm_tensor* dbl_t = tm_double(t);
    ASSERT_NOT_NULL(dbl_t);
    ASSERT_INT_EQ(TM_F64, dbl_t->dtype);
    double* dbl_data = (double*)dbl_t->data;
    ASSERT_TRUE(fabs(dbl_data[0] - 1.5) < 0.001);

    tm_free(t);
    tm_free(int_t);
    tm_free(dbl_t);
}

static void test_chunk(void) {
    int64_t shape[] = {6, 3};
    tm_tensor* t = tm_arange(0, 18, 1, TM_F32);
    t = tm_reshape(t, shape, 2);

    int actual_chunks;
    tm_tensor** chunks = tm_chunk(t, 3, 0, &actual_chunks);
    ASSERT_NOT_NULL(chunks);
    ASSERT_INT_EQ(3, actual_chunks);

    for (int i = 0; i < actual_chunks; i++) {
        ASSERT_NOT_NULL(chunks[i]);
        ASSERT_INT_EQ(2, tm_size(chunks[i], 0));
        ASSERT_INT_EQ(3, tm_size(chunks[i], 1));
        tm_free(chunks[i]);
    }
    free(chunks);
    tm_free(t);
}

static void test_isnan_isinf(void) {
    int64_t shape[] = {4};
    float data[] = {1.0f, NAN, INFINITY, -INFINITY};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* nan_mask = tm_isnan(t);
    ASSERT_NOT_NULL(nan_mask);
    int32_t* nan_data = (int32_t*)nan_mask->data;
    ASSERT_INT_EQ(0, nan_data[0]);
    ASSERT_INT_EQ(1, nan_data[1]);
    ASSERT_INT_EQ(0, nan_data[2]);

    tm_tensor* inf_mask = tm_isinf(t);
    ASSERT_NOT_NULL(inf_mask);
    int32_t* inf_data = (int32_t*)inf_mask->data;
    ASSERT_INT_EQ(0, inf_data[0]);
    ASSERT_INT_EQ(0, inf_data[1]);
    ASSERT_INT_EQ(1, inf_data[2]);
    ASSERT_INT_EQ(1, inf_data[3]);

    tm_free(t);
    tm_free(nan_mask);
    tm_free(inf_mask);
}

/* Full parity tests */

static void test_mm(void) {
    int64_t a_shape[] = {2, 3};
    int64_t b_shape[] = {3, 4};
    tm_tensor* a = tm_ones(a_shape, 2, TM_F32);
    tm_tensor* b = tm_full(b_shape, 2, TM_F32, 2.0);

    tm_tensor* c = tm_mm(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_INT_EQ(2, c->ndim);
    ASSERT_INT_EQ(2, tm_size(c, 0));
    ASSERT_INT_EQ(4, tm_size(c, 1));
    ASSERT_FLT_EQ(6.0f, tm_get_f32(c, 0));  /* 3 * 2.0 = 6.0 */

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_dot(void) {
    int64_t shape[] = {4};
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
    tm_tensor* a = tm_from_data(a_data, shape, 1, TM_F32);
    tm_tensor* b = tm_from_data(b_data, shape, 1, TM_F32);
    a->owns_data = false;
    b->owns_data = false;

    tm_tensor* c = tm_dot(a, b);
    ASSERT_NOT_NULL(c);
    /* dot = 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40 */
    ASSERT_FLT_EQ(40.0f, tm_get_f32(c, 0));

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_mv(void) {
    int64_t mat_shape[] = {2, 3};
    int64_t vec_shape[] = {3};
    tm_tensor* mat = tm_ones(mat_shape, 2, TM_F32);
    float vec_data[] = {1.0f, 2.0f, 3.0f};
    tm_tensor* vec = tm_from_data(vec_data, vec_shape, 1, TM_F32);
    vec->owns_data = false;

    tm_tensor* result = tm_mv(mat, vec);
    ASSERT_NOT_NULL(result);
    ASSERT_INT_EQ(1, result->ndim);
    ASSERT_INT_EQ(2, result->numel);
    ASSERT_FLT_EQ(6.0f, tm_get_f32(result, 0));  /* 1+2+3 = 6 */
    ASSERT_FLT_EQ(6.0f, tm_get_f32(result, 1));

    tm_free(mat);
    tm_free(vec);
    tm_free(result);
}

static void test_outer(void) {
    int64_t shape[] = {3};
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f};
    int64_t b_shape[] = {2};
    tm_tensor* a = tm_from_data(a_data, shape, 1, TM_F32);
    tm_tensor* b = tm_from_data(b_data, b_shape, 1, TM_F32);
    a->owns_data = false;
    b->owns_data = false;

    tm_tensor* c = tm_outer(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_INT_EQ(2, c->ndim);
    ASSERT_INT_EQ(3, tm_size(c, 0));
    ASSERT_INT_EQ(2, tm_size(c, 1));
    ASSERT_FLT_EQ(4.0f, tm_get_f32(c, 0));   /* 1*4 */
    ASSERT_FLT_EQ(10.0f, tm_get_f32(c, 3));  /* 2*5 */

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_floor_ceil_round(void) {
    int64_t shape[] = {4};
    float data[] = {1.2f, 2.7f, -1.5f, 3.5f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* f = tm_floor(t);
    ASSERT_FLT_EQ(1.0f, tm_get_f32(f, 0));
    ASSERT_FLT_EQ(2.0f, tm_get_f32(f, 1));
    ASSERT_FLT_EQ(-2.0f, tm_get_f32(f, 2));

    tm_tensor* c = tm_ceil(t);
    ASSERT_FLT_EQ(2.0f, tm_get_f32(c, 0));
    ASSERT_FLT_EQ(3.0f, tm_get_f32(c, 1));
    ASSERT_FLT_EQ(-1.0f, tm_get_f32(c, 2));

    tm_tensor* r = tm_round(t);
    ASSERT_FLT_EQ(1.0f, tm_get_f32(r, 0));
    ASSERT_FLT_EQ(3.0f, tm_get_f32(r, 1));

    tm_free(t);
    tm_free(f);
    tm_free(c);
    tm_free(r);
}

static void test_logical_ops(void) {
    int64_t shape[] = {4};
    float a_data[] = {1.0f, 0.0f, 1.0f, 0.0f};
    float b_data[] = {1.0f, 1.0f, 0.0f, 0.0f};
    tm_tensor* a = tm_from_data(a_data, shape, 1, TM_F32);
    tm_tensor* b = tm_from_data(b_data, shape, 1, TM_F32);
    a->owns_data = false;
    b->owns_data = false;

    tm_tensor* and_result = tm_logical_and(a, b);
    ASSERT_NOT_NULL(and_result);
    int32_t* and_data = (int32_t*)and_result->data;
    ASSERT_INT_EQ(1, and_data[0]);  /* 1 && 1 */
    ASSERT_INT_EQ(0, and_data[1]);  /* 0 && 1 */
    ASSERT_INT_EQ(0, and_data[2]);  /* 1 && 0 */
    ASSERT_INT_EQ(0, and_data[3]);  /* 0 && 0 */

    tm_tensor* or_result = tm_logical_or(a, b);
    int32_t* or_data = (int32_t*)or_result->data;
    ASSERT_INT_EQ(1, or_data[0]);
    ASSERT_INT_EQ(1, or_data[1]);
    ASSERT_INT_EQ(1, or_data[2]);
    ASSERT_INT_EQ(0, or_data[3]);

    tm_tensor* not_result = tm_logical_not(a);
    int32_t* not_data = (int32_t*)not_result->data;
    ASSERT_INT_EQ(0, not_data[0]);
    ASSERT_INT_EQ(1, not_data[1]);

    tm_free(a);
    tm_free(b);
    tm_free(and_result);
    tm_free(or_result);
    tm_free(not_result);
}

static void test_flip_roll(void) {
    int64_t shape[] = {4};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    int flip_dims[] = {0};
    tm_tensor* flipped = tm_flip(t, flip_dims, 1);
    ASSERT_NOT_NULL(flipped);
    ASSERT_FLT_EQ(4.0f, tm_get_f32(flipped, 0));
    ASSERT_FLT_EQ(1.0f, tm_get_f32(flipped, 3));

    tm_tensor* rolled = tm_roll(t, 2, 0);
    ASSERT_NOT_NULL(rolled);
    ASSERT_FLT_EQ(3.0f, tm_get_f32(rolled, 0));
    ASSERT_FLT_EQ(4.0f, tm_get_f32(rolled, 1));
    ASSERT_FLT_EQ(1.0f, tm_get_f32(rolled, 2));

    tm_free(t);
    tm_free(flipped);
    tm_free(rolled);
}

static void test_extra_inplace_ops(void) {
    int64_t shape[] = {4};
    tm_tensor* a = tm_full(shape, 1, TM_F32, 4.0);
    tm_tensor* b = tm_full(shape, 1, TM_F32, 2.0);

    tm_sub_(a, b);
    ASSERT_FLT_EQ(2.0f, tm_get_f32(a, 0));

    tm_mul_(a, b);
    ASSERT_FLT_EQ(4.0f, tm_get_f32(a, 0));

    tm_div_(a, b);
    ASSERT_FLT_EQ(2.0f, tm_get_f32(a, 0));

    tm_exp_(a);
    ASSERT_FLT_NEAR(expf(2.0f), tm_get_f32(a, 0), 0.001f);

    tm_free(a);
    a = tm_full(shape, 1, TM_F32, 2.0);
    tm_log_(a);
    ASSERT_FLT_NEAR(logf(2.0f), tm_get_f32(a, 0), 0.001f);

    tm_free(a);
    a = tm_full(shape, 1, TM_F32, 4.0);
    tm_sqrt_(a);
    ASSERT_FLT_EQ(2.0f, tm_get_f32(a, 0));

    tm_free(a);
    a = tm_full(shape, 1, TM_F32, -3.0);
    tm_abs_(a);
    ASSERT_FLT_EQ(3.0f, tm_get_f32(a, 0));

    tm_neg_(a);
    ASSERT_FLT_EQ(-3.0f, tm_get_f32(a, 0));

    tm_free(a);
    a = tm_full(shape, 1, TM_F32, -2.0);
    tm_relu_(a);
    ASSERT_FLT_EQ(0.0f, tm_get_f32(a, 0));

    tm_free(a);
    a = tm_full(shape, 1, TM_F32, 0.0);
    tm_sigmoid_(a);
    ASSERT_FLT_EQ(0.5f, tm_get_f32(a, 0));

    tm_free(a);
    tm_free(b);
}

static void test_equal_allclose(void) {
    int64_t shape[] = {3};
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f};
    float c_data[] = {1.0f, 2.0f, 4.0f};
    tm_tensor* a = tm_from_data(a_data, shape, 1, TM_F32);
    tm_tensor* b = tm_from_data(b_data, shape, 1, TM_F32);
    tm_tensor* c = tm_from_data(c_data, shape, 1, TM_F32);
    a->owns_data = false;
    b->owns_data = false;
    c->owns_data = false;

    ASSERT_TRUE(tm_equal(a, b));
    ASSERT_TRUE(!tm_equal(a, c));
    ASSERT_TRUE(tm_allclose(a, b, 1e-5, 1e-8));

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_like_functions(void) {
    int64_t shape[] = {2, 3};
    tm_tensor* t = tm_randn(shape, 2, TM_F32);

    tm_tensor* f = tm_full_like(t, 5.0);
    ASSERT_NOT_NULL(f);
    ASSERT_INT_EQ(2, f->ndim);
    ASSERT_INT_EQ(2, tm_size(f, 0));
    ASSERT_INT_EQ(3, tm_size(f, 1));
    ASSERT_FLT_EQ(5.0f, tm_get_f32(f, 0));

    tm_tensor* e = tm_empty_like(t);
    ASSERT_NOT_NULL(e);
    ASSERT_INT_EQ(2, e->ndim);

    tm_tensor* r = tm_rand_like(t);
    ASSERT_NOT_NULL(r);
    ASSERT_INT_EQ(2, r->ndim);

    tm_tensor* rn = tm_randn_like(t);
    ASSERT_NOT_NULL(rn);
    ASSERT_INT_EQ(2, rn->ndim);

    tm_free(t);
    tm_free(f);
    tm_free(e);
    tm_free(r);
    tm_free(rn);
}

static void test_cumprod(void) {
    int64_t shape[] = {4};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* cp = tm_cumprod(t, 0);
    ASSERT_NOT_NULL(cp);
    ASSERT_FLT_EQ(1.0f, tm_get_f32(cp, 0));
    ASSERT_FLT_EQ(2.0f, tm_get_f32(cp, 1));
    ASSERT_FLT_EQ(6.0f, tm_get_f32(cp, 2));
    ASSERT_FLT_EQ(24.0f, tm_get_f32(cp, 3));

    tm_free(t);
    tm_free(cp);
}

static void test_any_all(void) {
    int64_t shape[] = {4};
    float data1[] = {0.0f, 0.0f, 1.0f, 0.0f};
    float data2[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float data3[] = {0.0f, 0.0f, 0.0f, 0.0f};
    tm_tensor* t1 = tm_from_data(data1, shape, 1, TM_F32);
    tm_tensor* t2 = tm_from_data(data2, shape, 1, TM_F32);
    tm_tensor* t3 = tm_from_data(data3, shape, 1, TM_F32);
    t1->owns_data = false;
    t2->owns_data = false;
    t3->owns_data = false;

    tm_tensor* any1 = tm_any(t1, -1, false);
    ASSERT_NOT_NULL(any1);
    ASSERT_INT_EQ(1, ((int32_t*)any1->data)[0]);

    tm_tensor* any3 = tm_any(t3, -1, false);
    ASSERT_INT_EQ(0, ((int32_t*)any3->data)[0]);

    tm_tensor* all2 = tm_all(t2, -1, false);
    ASSERT_INT_EQ(1, ((int32_t*)all2->data)[0]);

    tm_tensor* all1 = tm_all(t1, -1, false);
    ASSERT_INT_EQ(0, ((int32_t*)all1->data)[0]);

    tm_free(t1);
    tm_free(t2);
    tm_free(t3);
    tm_free(any1);
    tm_free(any3);
    tm_free(all2);
    tm_free(all1);
}

static void test_norm(void) {
    int64_t shape[] = {3};
    float data[] = {3.0f, 4.0f, 0.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    /* L2 norm: sqrt(9 + 16 + 0) = 5 */
    tm_tensor* n2 = tm_norm(t, 2.0, -1, false);
    ASSERT_NOT_NULL(n2);
    ASSERT_FLT_EQ(5.0f, tm_get_f32(n2, 0));

    /* L1 norm: 3 + 4 + 0 = 7 */
    tm_tensor* n1 = tm_norm(t, 1.0, -1, false);
    ASSERT_FLT_EQ(7.0f, tm_get_f32(n1, 0));

    tm_free(t);
    tm_free(n2);
    tm_free(n1);
}

static void test_elu(void) {
    int64_t shape[] = {4};
    float data[] = {1.0f, 0.0f, -1.0f, -2.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* e = tm_elu(t, 1.0f);
    ASSERT_NOT_NULL(e);
    ASSERT_FLT_EQ(1.0f, tm_get_f32(e, 0));
    ASSERT_FLT_EQ(0.0f, tm_get_f32(e, 1));
    /* For x < 0: alpha * (exp(x) - 1) */
    ASSERT_FLT_NEAR(expf(-1.0f) - 1.0f, tm_get_f32(e, 2), 0.001f);

    tm_free(t);
    tm_free(e);
}

static void test_trig_functions(void) {
    int64_t shape[] = {1};
    float data[] = {0.5f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* tan_t = tm_tan(t);
    ASSERT_FLT_NEAR(tanf(0.5f), tm_get_f32(tan_t, 0), 0.001f);

    tm_tensor* asin_t = tm_asin(t);
    ASSERT_FLT_NEAR(asinf(0.5f), tm_get_f32(asin_t, 0), 0.001f);

    tm_tensor* acos_t = tm_acos(t);
    ASSERT_FLT_NEAR(acosf(0.5f), tm_get_f32(acos_t, 0), 0.001f);

    tm_tensor* atan_t = tm_atan(t);
    ASSERT_FLT_NEAR(atanf(0.5f), tm_get_f32(atan_t, 0), 0.001f);

    tm_tensor* sinh_t = tm_sinh(t);
    ASSERT_FLT_NEAR(sinhf(0.5f), tm_get_f32(sinh_t, 0), 0.001f);

    tm_tensor* cosh_t = tm_cosh(t);
    ASSERT_FLT_NEAR(coshf(0.5f), tm_get_f32(cosh_t, 0), 0.001f);

    tm_free(t);
    tm_free(tan_t);
    tm_free(asin_t);
    tm_free(acos_t);
    tm_free(atan_t);
    tm_free(sinh_t);
    tm_free(cosh_t);
}

/* NN Ops tests */

static void test_hardswish(void) {
    int64_t shape[] = {5};
    float data[] = {-4.0f, -3.0f, 0.0f, 3.0f, 4.0f};
    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* out = tm_hardswish(t);
    ASSERT_NOT_NULL(out);
    /* hardswish(-4) = -4 * 0 / 6 = 0 */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(out, 0), 0.01f);
    /* hardswish(0) = 0 * 3/6 = 0 */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(out, 2), 0.01f);
    /* hardswish(3) = 3 * 6/6 = 3 */
    ASSERT_FLT_NEAR(3.0f, tm_get_f32(out, 3), 0.01f);
    /* hardswish(4) = 4 * 6/6 = 4 */
    ASSERT_FLT_NEAR(4.0f, tm_get_f32(out, 4), 0.01f);

    tm_free(t);
    tm_free(out);
}

static void test_batch_norm(void) {
    /* [batch=2, channels=2, length=3] */
    int64_t shape[] = {2, 2, 3};
    tm_tensor* t = tm_randn(shape, 3, TM_F32);
    tm_manual_seed(42);

    /* No weight/bias (identity transform) */
    tm_tensor* out = tm_batch_norm(t, NULL, NULL, NULL, NULL, true, 0.1f, 1e-5f);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(3, out->ndim);
    ASSERT_INT_EQ(2, tm_size(out, 0));
    ASSERT_INT_EQ(2, tm_size(out, 1));
    ASSERT_INT_EQ(3, tm_size(out, 2));

    tm_free(t);
    tm_free(out);
}

static void test_group_norm(void) {
    /* [batch=1, channels=4, length=2] */
    int64_t shape[] = {1, 4, 2};
    tm_tensor* t = tm_randn(shape, 3, TM_F32);

    /* 2 groups */
    tm_tensor* out = tm_group_norm(t, 2, NULL, NULL, 1e-5f);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(3, out->ndim);

    tm_free(t);
    tm_free(out);
}

static void test_instance_norm(void) {
    /* [batch=2, channels=3, length=4] */
    int64_t shape[] = {2, 3, 4};
    tm_tensor* t = tm_randn(shape, 3, TM_F32);

    tm_tensor* out = tm_instance_norm(t, NULL, NULL, 1e-5f);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(3, out->ndim);

    tm_free(t);
    tm_free(out);
}

static void test_scaled_dot_product_attention(void) {
    /* Q, K, V: [batch=1, heads=2, seq=3, head_dim=4] */
    int64_t shape[] = {1, 2, 3, 4};
    tm_tensor* Q = tm_randn(shape, 4, TM_F32);
    tm_tensor* K = tm_randn(shape, 4, TM_F32);
    tm_tensor* V = tm_randn(shape, 4, TM_F32);

    tm_tensor* out = tm_scaled_dot_product_attention(Q, K, V, NULL, 0.0f, false);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(4, out->ndim);
    ASSERT_INT_EQ(1, tm_size(out, 0));
    ASSERT_INT_EQ(2, tm_size(out, 1));
    ASSERT_INT_EQ(3, tm_size(out, 2));
    ASSERT_INT_EQ(4, tm_size(out, 3));

    /* Test with causal mask */
    tm_tensor* out_causal = tm_scaled_dot_product_attention(Q, K, V, NULL, 0.0f, true);
    ASSERT_NOT_NULL(out_causal);

    tm_free(Q);
    tm_free(K);
    tm_free(V);
    tm_free(out);
    tm_free(out_causal);
}

static void test_split_merge_heads(void) {
    /* x: [batch=2, seq=4, embed_dim=8] */
    int64_t shape[] = {2, 4, 8};
    tm_tensor* x = tm_randn(shape, 3, TM_F32);

    /* Split into 2 heads */
    tm_tensor* split = tm_split_heads(x, 2);
    ASSERT_NOT_NULL(split);
    ASSERT_INT_EQ(4, split->ndim);
    ASSERT_INT_EQ(2, tm_size(split, 0));  /* batch */
    ASSERT_INT_EQ(2, tm_size(split, 1));  /* heads */
    ASSERT_INT_EQ(4, tm_size(split, 2));  /* seq */
    ASSERT_INT_EQ(4, tm_size(split, 3));  /* head_dim */

    /* Merge back */
    tm_tensor* merged = tm_merge_heads(split);
    ASSERT_NOT_NULL(merged);
    ASSERT_INT_EQ(3, merged->ndim);
    ASSERT_INT_EQ(2, tm_size(merged, 0));  /* batch */
    ASSERT_INT_EQ(4, tm_size(merged, 1));  /* seq */
    ASSERT_INT_EQ(8, tm_size(merged, 2));  /* embed_dim */

    tm_free(x);
    tm_free(split);
    tm_free(merged);
}

static void test_conv1d(void) {
    /* input: [batch=1, channels=2, length=5] */
    int64_t in_shape[] = {1, 2, 5};
    tm_tensor* input = tm_ones(in_shape, 3, TM_F32);

    /* weight: [out_channels=3, in_channels=2, kernel=3] */
    int64_t w_shape[] = {3, 2, 3};
    tm_tensor* weight = tm_ones(w_shape, 3, TM_F32);

    tm_tensor* out = tm_conv1d(input, weight, NULL, 1, 0, 1, 1);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(3, out->ndim);
    ASSERT_INT_EQ(1, tm_size(out, 0));
    ASSERT_INT_EQ(3, tm_size(out, 1));  /* out_channels */
    ASSERT_INT_EQ(3, tm_size(out, 2));  /* (5 - 3)/1 + 1 = 3 */

    /* Each output is sum of 2*3 ones = 6 */
    ASSERT_FLT_NEAR(6.0f, tm_get_f32(out, 0), 0.01f);

    tm_free(input);
    tm_free(weight);
    tm_free(out);
}

static void test_conv2d(void) {
    /* input: [batch=1, channels=1, H=4, W=4] */
    int64_t in_shape[] = {1, 1, 4, 4};
    tm_tensor* input = tm_ones(in_shape, 4, TM_F32);

    /* weight: [out_channels=1, in_channels=1, kH=2, kW=2] */
    int64_t w_shape[] = {1, 1, 2, 2};
    tm_tensor* weight = tm_ones(w_shape, 4, TM_F32);

    tm_tensor* out = tm_conv2d(input, weight, NULL, 1, 1, 0, 0, 1, 1, 1);
    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(4, out->ndim);
    ASSERT_INT_EQ(1, tm_size(out, 0));
    ASSERT_INT_EQ(1, tm_size(out, 1));
    ASSERT_INT_EQ(3, tm_size(out, 2));  /* (4 - 2)/1 + 1 = 3 */
    ASSERT_INT_EQ(3, tm_size(out, 3));

    /* Each output is sum of 2*2 ones = 4 */
    ASSERT_FLT_NEAR(4.0f, tm_get_f32(out, 0), 0.01f);

    tm_free(input);
    tm_free(weight);
    tm_free(out);
}

static void test_pooling(void) {
    /* 1D pooling: [batch=1, channels=1, length=4] */
    int64_t shape1[] = {1, 1, 4};
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tm_tensor* t1 = tm_from_data(data1, shape1, 3, TM_F32);
    t1->owns_data = false;

    tm_tensor* avg1 = tm_avg_pool1d(t1, 2, 2, 0);
    ASSERT_NOT_NULL(avg1);
    ASSERT_INT_EQ(2, tm_size(avg1, 2));  /* (4-2)/2 + 1 = 2 */
    ASSERT_FLT_NEAR(1.5f, tm_get_f32(avg1, 0), 0.01f);  /* avg(1,2) */
    ASSERT_FLT_NEAR(3.5f, tm_get_f32(avg1, 1), 0.01f);  /* avg(3,4) */

    tm_tensor* max1 = tm_max_pool1d(t1, 2, 2, 0);
    ASSERT_NOT_NULL(max1);
    ASSERT_FLT_NEAR(2.0f, tm_get_f32(max1, 0), 0.01f);  /* max(1,2) */
    ASSERT_FLT_NEAR(4.0f, tm_get_f32(max1, 1), 0.01f);  /* max(3,4) */

    /* 2D pooling: [batch=1, channels=1, H=2, W=2] */
    int64_t shape2[] = {1, 1, 2, 2};
    float data2[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tm_tensor* t2 = tm_from_data(data2, shape2, 4, TM_F32);
    t2->owns_data = false;

    tm_tensor* avg2 = tm_avg_pool2d(t2, 2, 2, 1, 1, 0, 0);
    ASSERT_NOT_NULL(avg2);
    ASSERT_INT_EQ(1, tm_size(avg2, 2));
    ASSERT_FLT_NEAR(2.5f, tm_get_f32(avg2, 0), 0.01f);  /* avg(1,2,3,4) */

    tm_free(t1);
    tm_free(avg1);
    tm_free(max1);
    tm_free(t2);
    tm_free(avg2);
}

static void test_adaptive_pooling(void) {
    /* 1D: [batch=1, channels=1, length=6] -> output_size=2 */
    int64_t shape1[] = {1, 1, 6};
    tm_tensor* t1 = tm_arange(1, 7, 1, TM_F32);
    t1 = tm_reshape(t1, shape1, 3);

    tm_tensor* out1 = tm_adaptive_avg_pool1d(t1, 2);
    ASSERT_NOT_NULL(out1);
    ASSERT_INT_EQ(2, tm_size(out1, 2));
    /* First 3 elements: avg(1,2,3) = 2, last 3: avg(4,5,6) = 5 */
    ASSERT_FLT_NEAR(2.0f, tm_get_f32(out1, 0), 0.01f);
    ASSERT_FLT_NEAR(5.0f, tm_get_f32(out1, 1), 0.01f);

    tm_free(t1);
    tm_free(out1);
}

/* ============================================================= */
/* Loss Function Tests                                           */
/* ============================================================= */

static void test_l1_loss(void) {
    /* L1 loss = mean(|pred - target|) */
    int64_t shape[] = {4};
    float pred_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float target_data[] = {1.5f, 1.5f, 3.5f, 3.5f};

    tm_tensor* pred = tm_from_data(pred_data, shape, 1, TM_F32);
    pred->owns_data = false;
    tm_tensor* target = tm_from_data(target_data, shape, 1, TM_F32);
    target->owns_data = false;

    tm_tensor* loss = tm_l1_loss(pred, target);
    ASSERT_NOT_NULL(loss);
    /* |1-1.5| + |2-1.5| + |3-3.5| + |4-3.5| = 0.5+0.5+0.5+0.5 = 2.0, avg = 0.5 */
    ASSERT_FLT_NEAR(0.5f, tm_get_f32(loss, 0), 0.001f);

    tm_free(pred);
    tm_free(target);
    tm_free(loss);
}

static void test_smooth_l1_loss(void) {
    /* Smooth L1 with beta=1.0 */
    int64_t shape[] = {4};
    float pred_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float target_data[] = {0.5f, 1.0f, 1.5f, 2.0f};

    tm_tensor* pred = tm_from_data(pred_data, shape, 1, TM_F32);
    pred->owns_data = false;
    tm_tensor* target = tm_from_data(target_data, shape, 1, TM_F32);
    target->owns_data = false;

    tm_tensor* loss = tm_smooth_l1_loss(pred, target, 1.0f);
    ASSERT_NOT_NULL(loss);
    /* For |x| < 1: 0.5*x^2, else |x| - 0.5
     * 0.5: 0.5*0.25 = 0.125
     * 1.0: 1.0 - 0.5 = 0.5
     * 1.5: 1.5 - 0.5 = 1.0
     * 2.0: 2.0 - 0.5 = 1.5
     * Total = 3.125, avg = 0.78125 */
    ASSERT_FLT_NEAR(0.78125f, tm_get_f32(loss, 0), 0.01f);

    tm_free(pred);
    tm_free(target);
    tm_free(loss);
}

static void test_nll_loss(void) {
    /* NLL loss with log probabilities */
    int64_t logp_shape[] = {2, 3};  /* batch=2, classes=3 */
    /* log_softmax values (approximately) */
    float logp_data[] = {
        -0.4f, -1.2f, -2.0f,  /* batch 0: class 0 is most likely */
        -2.0f, -0.4f, -1.2f   /* batch 1: class 1 is most likely */
    };

    int64_t target_shape[] = {2};
    int32_t target_data[] = {0, 1};  /* correct classes */

    tm_tensor* log_probs = tm_from_data(logp_data, logp_shape, 2, TM_F32);
    log_probs->owns_data = false;
    tm_tensor* targets = tm_from_data(target_data, target_shape, 1, TM_I32);
    targets->owns_data = false;

    tm_tensor* loss = tm_nll_loss(log_probs, targets);
    ASSERT_NOT_NULL(loss);
    /* -(-0.4 + -0.4) / 2 = 0.4 */
    ASSERT_FLT_NEAR(0.4f, tm_get_f32(loss, 0), 0.01f);

    tm_free(log_probs);
    tm_free(targets);
    tm_free(loss);
}

static void test_kl_div(void) {
    /* KL divergence test: KL(target || exp(input))
     * Uses very divergent distributions to ensure positive result */
    int64_t shape[] = {1, 3};
    /* Log probabilities (input) - very low prob for class 0 */
    float input_data[] = {-5.0f, -0.5f, -0.5f};
    /* Target probabilities - concentrated on class 0 */
    float target_data[] = {0.9f, 0.05f, 0.05f};

    tm_tensor* input = tm_from_data(input_data, shape, 2, TM_F32);
    input->owns_data = false;
    tm_tensor* target = tm_from_data(target_data, shape, 2, TM_F32);
    target->owns_data = false;

    tm_tensor* loss = tm_kl_div(input, target);
    ASSERT_NOT_NULL(loss);
    /* KL divergence should be non-negative */
    ASSERT_TRUE(tm_get_f32(loss, 0) >= 0);

    tm_free(input);
    tm_free(target);
    tm_free(loss);
}

static void test_binary_cross_entropy(void) {
    /* BCE loss */
    int64_t shape[] = {4};
    float pred_data[] = {0.9f, 0.1f, 0.8f, 0.2f};
    float target_data[] = {1.0f, 0.0f, 1.0f, 0.0f};

    tm_tensor* pred = tm_from_data(pred_data, shape, 1, TM_F32);
    pred->owns_data = false;
    tm_tensor* target = tm_from_data(target_data, shape, 1, TM_F32);
    target->owns_data = false;

    tm_tensor* loss = tm_binary_cross_entropy(pred, target);
    ASSERT_NOT_NULL(loss);
    /* All predictions are fairly correct, loss should be small */
    ASSERT_TRUE(tm_get_f32(loss, 0) < 0.5f);

    tm_free(pred);
    tm_free(target);
    tm_free(loss);
}

static void test_half_conversion(void) {
    /* Float16 conversion test */
    int64_t shape[] = {4};
    float data[] = {1.0f, 2.5f, -3.0f, 0.0f};

    tm_tensor* t = tm_from_data(data, shape, 1, TM_F32);
    t->owns_data = false;

    tm_tensor* h = tm_half(t);
    ASSERT_NOT_NULL(h);
    ASSERT_INT_EQ(TM_F16, h->dtype);

    /* Convert back to float and check values */
    tm_tensor* f = tm_float(h);
    ASSERT_NOT_NULL(f);
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(f, 0), 0.01f);
    ASSERT_FLT_NEAR(2.5f, tm_get_f32(f, 1), 0.01f);
    ASSERT_FLT_NEAR(-3.0f, tm_get_f32(f, 2), 0.01f);
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(f, 3), 0.01f);

    tm_free(t);
    tm_free(h);
    tm_free(f);
}

/* ============================================================= */
/* Additional Loss Function Tests                                */
/* ============================================================= */

static void test_cosine_embedding_loss(void) {
    /* Test cosine embedding loss */
    int64_t x_shape[] = {2, 3};
    float x1_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float x2_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};  /* First same, second orthogonal */
    float y_data[] = {1.0f, -1.0f};  /* First should be similar, second dissimilar */

    tm_tensor* x1 = tm_from_data(x1_data, x_shape, 2, TM_F32);
    x1->owns_data = false;
    tm_tensor* x2 = tm_from_data(x2_data, x_shape, 2, TM_F32);
    x2->owns_data = false;

    int64_t y_shape[] = {2};
    tm_tensor* y = tm_from_data(y_data, y_shape, 1, TM_F32);
    y->owns_data = false;

    tm_tensor* loss = tm_cosine_embedding_loss(x1, x2, y, 0.0f);
    ASSERT_NOT_NULL(loss);
    /* First pair: cos_sim=1, y=1, loss=0
     * Second pair: cos_sim=0, y=-1, loss=max(0, 0-0)=0
     * Total loss = 0 */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(loss, 0), 0.01f);

    tm_free(x1);
    tm_free(x2);
    tm_free(y);
    tm_free(loss);
}

static void test_triplet_margin_loss(void) {
    /* Test triplet margin loss */
    int64_t shape[] = {2, 3};
    float anchor_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float positive_data[] = {0.9f, 0.1f, 0.0f, 0.1f, 0.9f, 0.0f};  /* Close to anchor */
    float negative_data[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};  /* Far from anchor */

    tm_tensor* anchor = tm_from_data(anchor_data, shape, 2, TM_F32);
    anchor->owns_data = false;
    tm_tensor* positive = tm_from_data(positive_data, shape, 2, TM_F32);
    positive->owns_data = false;
    tm_tensor* negative = tm_from_data(negative_data, shape, 2, TM_F32);
    negative->owns_data = false;

    tm_tensor* loss = tm_triplet_margin_loss(anchor, positive, negative, 1.0f, 2.0f);
    ASSERT_NOT_NULL(loss);
    /* With positive closer than negative, and margin=1, loss should be manageable */
    ASSERT_TRUE(tm_get_f32(loss, 0) >= 0.0f);

    tm_free(anchor);
    tm_free(positive);
    tm_free(negative);
    tm_free(loss);
}

static void test_label_smoothing(void) {
    /* Test cross entropy with label smoothing */
    int64_t logits_shape[] = {2, 3};
    float logits_data[] = {2.0f, 1.0f, 0.0f, 0.0f, 2.0f, 1.0f};

    int64_t targets_shape[] = {2};
    int32_t targets_data[] = {0, 1};

    tm_tensor* logits = tm_from_data(logits_data, logits_shape, 2, TM_F32);
    logits->owns_data = false;
    tm_tensor* targets = tm_from_data(targets_data, targets_shape, 1, TM_I32);
    targets->owns_data = false;

    /* With smoothing=0, should equal regular cross entropy */
    tm_tensor* loss_no_smooth = tm_cross_entropy_label_smooth(logits, targets, 0.0f);
    tm_tensor* loss_regular = tm_cross_entropy(logits, targets);
    ASSERT_NOT_NULL(loss_no_smooth);
    ASSERT_NOT_NULL(loss_regular);
    ASSERT_FLT_NEAR(tm_get_f32(loss_regular, 0), tm_get_f32(loss_no_smooth, 0), 0.01f);

    /* With smoothing > 0, loss should be higher */
    tm_tensor* loss_smooth = tm_cross_entropy_label_smooth(logits, targets, 0.1f);
    ASSERT_NOT_NULL(loss_smooth);
    ASSERT_TRUE(tm_get_f32(loss_smooth, 0) > tm_get_f32(loss_no_smooth, 0));

    tm_free(logits);
    tm_free(targets);
    tm_free(loss_no_smooth);
    tm_free(loss_regular);
    tm_free(loss_smooth);
}

/* ============================================================= */
/* Linear Algebra Tests                                          */
/* ============================================================= */

static void test_inverse(void) {
    /* Test matrix inverse: A * A^-1 = I */
    int64_t shape[] = {3, 3};
    float data[] = {
        1.0f, 2.0f, 3.0f,
        0.0f, 1.0f, 4.0f,
        5.0f, 6.0f, 0.0f
    };

    tm_tensor* A = tm_from_data(data, shape, 2, TM_F32);
    A->owns_data = false;

    tm_tensor* A_inv = tm_inverse(A);
    ASSERT_NOT_NULL(A_inv);

    /* Verify A * A_inv ≈ I */
    tm_tensor* I_approx = tm_matmul(A, A_inv);
    ASSERT_NOT_NULL(I_approx);

    /* Check diagonal is 1, off-diagonal is 0 */
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(I_approx, 0), 0.01f);
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(I_approx, 4), 0.01f);
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(I_approx, 8), 0.01f);
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(I_approx, 1), 0.01f);

    tm_free(A);
    tm_free(A_inv);
    tm_free(I_approx);
}

static void test_det(void) {
    /* Test determinant */
    int64_t shape[] = {2, 2};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};

    tm_tensor* A = tm_from_data(data, shape, 2, TM_F32);
    A->owns_data = false;

    tm_tensor* det = tm_det(A);
    ASSERT_NOT_NULL(det);
    /* det([[1,2],[3,4]]) = 1*4 - 2*3 = -2 */
    ASSERT_FLT_NEAR(-2.0f, tm_get_f32(det, 0), 0.01f);

    tm_free(A);
    tm_free(det);
}

static void test_svd(void) {
    /* Test SVD: A = U * S * V^T */
    int64_t shape[] = {3, 2};
    float data[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };

    tm_tensor* A = tm_from_data(data, shape, 2, TM_F32);
    A->owns_data = false;

    int num_outputs;
    tm_tensor** usv = tm_svd(A, &num_outputs);
    ASSERT_NOT_NULL(usv);
    ASSERT_INT_EQ(3, num_outputs);

    tm_tensor* U = usv[0];
    tm_tensor* S = usv[1];
    tm_tensor* V = usv[2];

    ASSERT_NOT_NULL(U);
    ASSERT_NOT_NULL(S);
    ASSERT_NOT_NULL(V);

    /* U should be [3, 2], S should be [2], V should be [2, 2] */
    ASSERT_INT_EQ(3, tm_size(U, 0));
    ASSERT_INT_EQ(2, tm_size(U, 1));
    ASSERT_INT_EQ(2, tm_numel(S));

    /* Singular values should be positive and decreasing */
    ASSERT_TRUE(tm_get_f32(S, 0) > 0);
    ASSERT_TRUE(tm_get_f32(S, 0) >= tm_get_f32(S, 1));

    tm_free(A);
    tm_free(U);
    tm_free(S);
    tm_free(V);
    free(usv);
}

static void test_eig(void) {
    /* Test eigenvalues for symmetric matrix */
    int64_t shape[] = {2, 2};
    float data[] = {2.0f, 1.0f, 1.0f, 2.0f};  /* Symmetric matrix */

    tm_tensor* A = tm_from_data(data, shape, 2, TM_F32);
    A->owns_data = false;

    tm_tensor* eig = tm_eig(A);
    ASSERT_NOT_NULL(eig);
    ASSERT_INT_EQ(2, tm_numel(eig));

    /* Eigenvalues of [[2,1],[1,2]] are 3 and 1 */
    float e1 = tm_get_f32(eig, 0);
    float e2 = tm_get_f32(eig, 1);
    /* Check that eigenvalues are approximately 3 and 1 (order may vary) */
    float sum = e1 + e2;
    float prod = e1 * e2;
    ASSERT_FLT_NEAR(4.0f, sum, 0.1f);   /* trace = 2 + 2 = 4 */
    ASSERT_FLT_NEAR(3.0f, prod, 0.1f);  /* det = 4 - 1 = 3 */

    tm_free(A);
    tm_free(eig);
}

/* ============================================================= */
/* Autograd Extension Tests                                      */
/* ============================================================= */

static void test_retain_grad(void) {
    /* Test retain_grad - should allow non-leaf to keep gradient */
    int64_t shape[] = {2, 2};
    tm_tensor* a = tm_ones(shape, 2, TM_F32);
    tm_set_requires_grad(a, true);

    tm_tensor* b = tm_mul_scalar(a, 2.0f);
    tm_retain_grad(b);

    tm_tensor* c = tm_sum(b, -1, false);
    tm_backward(c);

    /* After backward, b should have gradient */
    ASSERT_NOT_NULL(b->grad);

    tm_free(a);
    tm_free(b);
    tm_free(c);
}

static void test_explicit_grad(void) {
    /* Test tm_grad for explicit gradient computation */
    int64_t shape[] = {2};
    tm_tensor* x = tm_full(shape, 1, TM_F32, 3.0f);
    tm_set_requires_grad(x, true);

    /* y = x^2 (via x * x) */
    tm_tensor* y = tm_mul(x, x);

    tm_tensor* outputs[] = {y};
    tm_tensor* inputs[] = {x};

    tm_tensor** grads = tm_grad(outputs, 1, inputs, 1, NULL);
    ASSERT_NOT_NULL(grads);

    /* dy/dx = 2x = 6 */
    if (grads[0]) {
        ASSERT_FLT_NEAR(6.0f, tm_get_f32(grads[0], 0), 0.1f);
        tm_free(grads[0]);
    }

    free(grads);
    tm_free(x);
    tm_free(y);
}

/* Loss function for Hessian test: L = sum(x^2) = x1^2 + x2^2 */
static tm_tensor* quadratic_loss(tm_tensor** params, int num_params, void* ctx) {
    (void)ctx;
    if (num_params != 1 || !params[0]) return NULL;

    /* L = sum(x^2) */
    tm_tensor* sq = tm_mul(params[0], params[0]);
    tm_tensor* loss = tm_sum(sq, -1, false);
    tm_free(sq);
    return loss;
}

static void test_hessian(void) {
    /* Test Hessian computation for L = x1^2 + x2^2
     * Hessian should be 2*I (diagonal matrix with 2s) */
    int64_t shape[] = {2};
    tm_tensor* x = tm_full(shape, 1, TM_F32, 1.0f);
    tm_set_requires_grad(x, true);

    tm_tensor* params[] = {x};
    tm_tensor* H = tm_hessian(params, 1, quadratic_loss, NULL);
    ASSERT_NOT_NULL(H);

    /* H should be [[2, 0], [0, 2]] */
    ASSERT_INT_EQ(2, H->ndim);
    ASSERT_INT_EQ(2, H->shape[0]);
    ASSERT_INT_EQ(2, H->shape[1]);

    float* h_data = (float*)H->data;
    /* Diagonal should be ~2 */
    ASSERT_FLT_NEAR(2.0f, h_data[0], 0.1f);  /* H[0,0] */
    ASSERT_FLT_NEAR(2.0f, h_data[3], 0.1f);  /* H[1,1] */
    /* Off-diagonal should be ~0 */
    ASSERT_FLT_NEAR(0.0f, h_data[1], 0.1f);  /* H[0,1] */
    ASSERT_FLT_NEAR(0.0f, h_data[2], 0.1f);  /* H[1,0] */

    tm_free(H);
    tm_free(x);
}

static void test_hvp(void) {
    /* Test Hessian-vector product for L = x1^2 + x2^2
     * H = [[2, 0], [0, 2]], v = [1, 2]
     * Hv = [2, 4] */
    int64_t shape[] = {2};
    tm_tensor* x = tm_full(shape, 1, TM_F32, 1.0f);
    tm_set_requires_grad(x, true);

    float v_data[] = {1.0f, 2.0f};
    tm_tensor* v = tm_from_data(v_data, shape, 1, TM_F32);

    tm_tensor* params[] = {x};
    tm_tensor* vecs[] = {v};

    tm_tensor* hvp = tm_hvp(params, 1, vecs, 1, quadratic_loss, NULL);
    ASSERT_NOT_NULL(hvp);

    /* Hv should be [2, 4] */
    ASSERT_INT_EQ(1, hvp->ndim);
    ASSERT_INT_EQ(2, hvp->shape[0]);

    float* hvp_data = (float*)hvp->data;
    ASSERT_FLT_NEAR(2.0f, hvp_data[0], 0.2f);
    ASSERT_FLT_NEAR(4.0f, hvp_data[1], 0.2f);

    tm_free(hvp);
    tm_free(v);
    tm_free(x);
}

/* Custom autograd function: squared function with custom backward */
static tm_tensor* square_forward(tm_tensor** inputs, int num_inputs, tm_custom_ctx* ctx) {
    if (num_inputs != 1 || !inputs[0]) return NULL;

    /* Save input for backward */
    tm_save_for_backward(ctx, inputs[0]);

    /* Return x^2 */
    return tm_mul(inputs[0], inputs[0]);
}

static tm_tensor** square_backward(tm_tensor* grad_output, tm_custom_ctx* ctx, int* num_grads) {
    int num_saved;
    tm_tensor** saved = tm_get_saved_tensors(ctx, &num_saved);
    if (num_saved != 1 || !saved[0]) {
        *num_grads = 0;
        return NULL;
    }

    /* d(x^2)/dx = 2x */
    tm_tensor* two_x = tm_mul_scalar(saved[0], 2.0f);
    tm_tensor* grad_input = tm_mul(grad_output, two_x);
    tm_free(two_x);

    tm_tensor** grads = (tm_tensor**)malloc(sizeof(tm_tensor*));
    grads[0] = grad_input;
    *num_grads = 1;

    return grads;
}

static void test_custom_autograd(void) {
    /* Register and test a custom autograd function */
    int fn_id = tm_register_autograd_function("square", square_forward, square_backward);
    ASSERT_TRUE(fn_id >= 0);

    /* Test forward pass */
    int64_t shape[] = {2};
    float x_data[] = {2.0f, 3.0f};
    tm_tensor* x = tm_from_data(x_data, shape, 1, TM_F32);
    tm_set_requires_grad(x, true);

    tm_tensor* inputs[] = {x};
    tm_tensor* y = tm_apply_custom_function("square", inputs, 1);
    ASSERT_NOT_NULL(y);

    /* y should be [4, 9] */
    float* y_data = (float*)y->data;
    ASSERT_FLT_NEAR(4.0f, y_data[0], 0.01f);
    ASSERT_FLT_NEAR(9.0f, y_data[1], 0.01f);

    /* Test backward pass */
    tm_tensor* loss = tm_sum(y, -1, false);
    tm_backward(loss);

    /* grad should be 2*x = [4, 6] */
    ASSERT_NOT_NULL(x->grad);
    float* grad = (float*)x->grad->data;
    ASSERT_FLT_NEAR(4.0f, grad[0], 0.01f);
    ASSERT_FLT_NEAR(6.0f, grad[1], 0.01f);

    tm_free(loss);
    tm_free(y);
    tm_free(x);

    /* Cleanup registry */
    tm_cleanup_autograd_registry();
}

static void test_inplace_add_grad(void) {
    /* Test tm_add_ with autograd: a += b */
    int64_t shape[] = {2};
    tm_tensor* a = tm_full(shape, 1, TM_F32, 2.0f);
    tm_tensor* b = tm_full(shape, 1, TM_F32, 3.0f);
    tm_set_requires_grad(a, true);
    tm_set_requires_grad(b, true);

    /* a = [2, 2], b = [3, 3] */
    tm_add_(a, b);  /* a becomes [5, 5] */

    ASSERT_FLT_NEAR(5.0f, tm_get_f32(a, 0), 0.01f);

    /* Compute loss = sum(a) = 10, backward */
    tm_tensor* loss = tm_sum(a, -1, false);
    tm_backward(loss);

    /* grad_b should be [1, 1] (gradient flows through) */
    ASSERT_NOT_NULL(b->grad);
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(b->grad, 0), 0.01f);

    tm_free(loss);
    tm_free(a);
    tm_free(b);
}

static void test_inplace_mul_grad(void) {
    /* Test tm_mul_ with autograd: a *= b */
    int64_t shape[] = {2};
    tm_tensor* a = tm_full(shape, 1, TM_F32, 2.0f);
    tm_tensor* b = tm_full(shape, 1, TM_F32, 3.0f);
    tm_set_requires_grad(a, true);
    tm_set_requires_grad(b, true);

    /* a = [2, 2], b = [3, 3] */
    tm_mul_(a, b);  /* a becomes [6, 6] */

    ASSERT_FLT_NEAR(6.0f, tm_get_f32(a, 0), 0.01f);

    /* Compute loss = sum(a) = 12, backward */
    tm_tensor* loss = tm_sum(a, -1, false);
    tm_backward(loss);

    /* grad_b should be [2, 2] (original a values) */
    ASSERT_NOT_NULL(b->grad);
    ASSERT_FLT_NEAR(2.0f, tm_get_f32(b->grad, 0), 0.1f);

    tm_free(loss);
    tm_free(a);
    tm_free(b);
}

static void test_inplace_exp_grad(void) {
    /* Test tm_exp_ with autograd */
    int64_t shape[] = {2};
    tm_tensor* a = tm_full(shape, 1, TM_F32, 1.0f);
    tm_set_requires_grad(a, true);

    /* a = [1, 1] -> exp -> [e, e] */
    tm_exp_(a);

    float e = expf(1.0f);
    ASSERT_FLT_NEAR(e, tm_get_f32(a, 0), 0.01f);

    /* Compute loss = sum(a), backward */
    tm_tensor* loss = tm_sum(a, -1, false);
    tm_backward(loss);

    /* grad should be [e, e] (d/dx exp(x) = exp(x)) */
    ASSERT_NOT_NULL(a->grad);
    ASSERT_FLT_NEAR(e, tm_get_f32(a->grad, 0), 0.01f);

    tm_free(loss);
    tm_free(a);
}

static void test_inplace_relu_grad(void) {
    /* Test tm_relu_ with autograd */
    int64_t shape[] = {4};
    float data[] = {-2.0f, -1.0f, 1.0f, 2.0f};
    tm_tensor* a = tm_from_data(data, shape, 1, TM_F32);
    tm_set_requires_grad(a, true);

    tm_relu_(a);

    /* a should be [0, 0, 1, 2] */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(a, 0), 0.01f);
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(a, 1), 0.01f);
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(a, 2), 0.01f);
    ASSERT_FLT_NEAR(2.0f, tm_get_f32(a, 3), 0.01f);

    /* Compute loss = sum(a), backward */
    tm_tensor* loss = tm_sum(a, -1, false);
    tm_backward(loss);

    /* grad should be [0, 0, 1, 1] */
    ASSERT_NOT_NULL(a->grad);
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(a->grad, 0), 0.01f);
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(a->grad, 1), 0.01f);
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(a->grad, 2), 0.01f);
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(a->grad, 3), 0.01f);

    tm_free(loss);
    tm_free(a);
}

static void test_inplace_sigmoid_grad(void) {
    /* Test tm_sigmoid_ with autograd */
    int64_t shape[] = {2};
    tm_tensor* a = tm_full(shape, 1, TM_F32, 0.0f);
    tm_set_requires_grad(a, true);

    /* sigmoid(0) = 0.5 */
    tm_sigmoid_(a);

    ASSERT_FLT_NEAR(0.5f, tm_get_f32(a, 0), 0.01f);

    /* Compute loss = sum(a), backward */
    tm_tensor* loss = tm_sum(a, -1, false);
    tm_backward(loss);

    /* grad should be 0.5 * (1 - 0.5) = 0.25 */
    ASSERT_NOT_NULL(a->grad);
    ASSERT_FLT_NEAR(0.25f, tm_get_f32(a->grad, 0), 0.01f);

    tm_free(loss);
    tm_free(a);
}

/* ============================================================= */
/* Additional NN Activation Tests                                */
/* ============================================================= */

static void test_mish(void) {
    int64_t shape[] = {4};
    tm_tensor* t = tm_full(shape, 1, TM_F32, 1.0f);
    tm_tensor* out = tm_mish(t);

    ASSERT_NOT_NULL(out);
    /* mish(1) = 1 * tanh(softplus(1)) = 1 * tanh(ln(1+e)) ≈ 0.865 */
    ASSERT_FLT_NEAR(0.865f, tm_get_f32(out, 0), 0.01f);

    tm_free(out);
    tm_free(t);
}

static void test_selu(void) {
    tm_tensor* t = tm_arange(-2.0, 2.0, 1.0, TM_F32);
    tm_tensor* out = tm_selu(t);

    ASSERT_NOT_NULL(out);
    /* selu(1) = scale * 1 ≈ 1.0507 */
    ASSERT_FLT_NEAR(1.0507f, tm_get_f32(out, 3), 0.01f);
    /* selu(-1) = scale * alpha * (exp(-1) - 1) ≈ -1.111 */
    ASSERT_TRUE(tm_get_f32(out, 1) < 0.0f);

    tm_free(out);
    tm_free(t);
}

static void test_celu(void) {
    tm_tensor* t = tm_arange(-2.0, 2.0, 1.0, TM_F32);
    tm_tensor* out = tm_celu(t, 1.0f);

    ASSERT_NOT_NULL(out);
    /* celu(1, 1) = 1 */
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(out, 3), 0.01f);
    /* celu(-1, 1) = 1*(exp(-1) - 1) ≈ -0.632 */
    ASSERT_FLT_NEAR(-0.632f, tm_get_f32(out, 1), 0.01f);

    tm_free(out);
    tm_free(t);
}

static void test_relu6(void) {
    /* tm_arange(-2, 8, 2.5) -> [-2, 0.5, 3, 5.5] (4 elements) */
    tm_tensor* t = tm_arange(-2.0, 10.0, 3.0, TM_F32);  /* [-2, 1, 4, 7] */
    tm_tensor* out = tm_relu6(t);

    ASSERT_NOT_NULL(out);
    /* relu6(-2) = 0 */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(out, 0), 0.01f);
    /* relu6(1) = 1 */
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(out, 1), 0.01f);
    /* relu6(4) = 4 */
    ASSERT_FLT_NEAR(4.0f, tm_get_f32(out, 2), 0.01f);
    /* relu6(7) = 6 (capped) */
    ASSERT_FLT_NEAR(6.0f, tm_get_f32(out, 3), 0.01f);

    tm_free(out);
    tm_free(t);
}

static void test_glu(void) {
    /* Create tensor [1, 2, 3, 4] and split along last dim */
    tm_tensor* t = tm_arange(1.0, 5.0, 1.0, TM_F32);
    tm_tensor* out = tm_glu(t, 0);

    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(2, (int)out->numel);
    /* a = [1, 2], b = [3, 4] -> a * sigmoid(b) */
    /* out[0] = 1 * sigmoid(3) ≈ 0.952 */
    ASSERT_FLT_NEAR(0.952f, tm_get_f32(out, 0), 0.01f);

    tm_free(out);
    tm_free(t);
}

static void test_softsign(void) {
    tm_tensor* t = tm_arange(-1.0, 2.0, 1.0, TM_F32);
    tm_tensor* out = tm_softsign(t);

    ASSERT_NOT_NULL(out);
    /* softsign(-1) = -1/(1+1) = -0.5 */
    ASSERT_FLT_NEAR(-0.5f, tm_get_f32(out, 0), 0.01f);
    /* softsign(0) = 0 */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(out, 1), 0.01f);
    /* softsign(1) = 1/(1+1) = 0.5 */
    ASSERT_FLT_NEAR(0.5f, tm_get_f32(out, 2), 0.01f);

    tm_free(out);
    tm_free(t);
}

static void test_tanhshrink(void) {
    tm_tensor* t = tm_arange(-1.0, 2.0, 1.0, TM_F32);
    tm_tensor* out = tm_tanhshrink(t);

    ASSERT_NOT_NULL(out);
    /* tanhshrink(0) = 0 - tanh(0) = 0 */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(out, 1), 0.01f);
    /* tanhshrink(1) = 1 - tanh(1) ≈ 0.238 */
    ASSERT_FLT_NEAR(0.238f, tm_get_f32(out, 2), 0.01f);

    tm_free(out);
    tm_free(t);
}

static void test_hardsigmoid(void) {
    tm_tensor* t = tm_arange(-4.0, 5.0, 2.0, TM_F32);  /* -4, -2, 0, 2, 4 */
    tm_tensor* out = tm_hardsigmoid(t);

    ASSERT_NOT_NULL(out);
    /* hardsigmoid(-4) = 0 (clamped) */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(out, 0), 0.01f);
    /* hardsigmoid(0) = 0/6 + 0.5 = 0.5 */
    ASSERT_FLT_NEAR(0.5f, tm_get_f32(out, 2), 0.01f);
    /* hardsigmoid(4) = 1 (clamped) */
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(out, 4), 0.01f);

    tm_free(out);
    tm_free(t);
}

static void test_threshold_fn(void) {
    tm_tensor* t = tm_arange(0.0, 4.0, 1.0, TM_F32);
    tm_tensor* out = tm_threshold(t, 2.0f, -1.0f);

    ASSERT_NOT_NULL(out);
    /* threshold(0, 2, -1) = -1 (0 <= 2) */
    ASSERT_FLT_NEAR(-1.0f, tm_get_f32(out, 0), 0.01f);
    /* threshold(2, 2, -1) = -1 (2 <= 2) */
    ASSERT_FLT_NEAR(-1.0f, tm_get_f32(out, 2), 0.01f);
    /* threshold(3, 2, -1) = 3 (3 > 2) */
    ASSERT_FLT_NEAR(3.0f, tm_get_f32(out, 3), 0.01f);

    tm_free(out);
    tm_free(t);
}

static void test_pad_fn(void) {
    /* Test 1D padding */
    tm_tensor* t1 = tm_arange(1.0, 4.0, 1.0, TM_F32);  /* [1, 2, 3] */
    int64_t padding1[] = {1, 2};  /* pad left=1, right=2 */
    tm_tensor* out1 = tm_pad(t1, padding1, 2, 0.0f);

    ASSERT_NOT_NULL(out1);
    ASSERT_INT_EQ(6, (int)out1->numel);  /* 3 + 1 + 2 = 6 */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(out1, 0), 0.01f);  /* pad */
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(out1, 1), 0.01f);  /* original */
    ASSERT_FLT_NEAR(3.0f, tm_get_f32(out1, 3), 0.01f);  /* original */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(out1, 5), 0.01f);  /* pad */

    tm_free(out1);
    tm_free(t1);
}

static void test_normalize_fn(void) {
    /* Test with 2D tensor, normalize along dim 1 */
    int64_t shape[] = {2, 3};
    tm_tensor* t = tm_full(shape, 2, TM_F32, 3.0f);  /* [[3,3,3], [3,3,3]] */
    tm_tensor* out = tm_normalize(t, 2.0f, 1, 1e-12f);

    ASSERT_NOT_NULL(out);
    /* L2 norm of [3,3,3] = sqrt(27) ≈ 5.196 */
    /* normalized = 3 / 5.196 ≈ 0.577 */
    ASSERT_FLT_NEAR(0.577f, tm_get_f32(out, 0), 0.01f);

    tm_free(out);
    tm_free(t);
}

static void test_interpolate_fn(void) {
    /* Test 4D tensor [1, 1, 2, 2] -> [1, 1, 4, 4] */
    int64_t shape[] = {1, 1, 2, 2};
    tm_tensor* t = tm_arange(1.0, 5.0, 1.0, TM_F32);
    t = tm_reshape(t, shape, 4);

    int64_t size[] = {4, 4};
    tm_tensor* out = tm_interpolate(t, size, 2, "nearest");

    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(4, (int)out->shape[2]);
    ASSERT_INT_EQ(4, (int)out->shape[3]);
    /* Nearest neighbor: should replicate values */
    ASSERT_FLT_NEAR(1.0f, tm_get_f32(out, 0), 0.01f);  /* top-left */

    tm_free(out);
    tm_free(t);
}

/* ============================================================= */
/* Sparse Tensor Tests                                           */
/* ============================================================= */

static void test_sparse_coo_creation(void) {
    /* Create a 3x3 sparse matrix with 3 non-zeros */
    int64_t idx_shape[] = {2, 3};
    tm_tensor* indices = tm_empty(idx_shape, 2, TM_I64);
    int64_t* idx_data = (int64_t*)indices->data;
    /* Row indices */
    idx_data[0] = 0; idx_data[1] = 1; idx_data[2] = 2;
    /* Col indices */
    idx_data[3] = 0; idx_data[4] = 1; idx_data[5] = 2;

    int64_t val_shape[] = {3};
    tm_tensor* values = tm_full(val_shape, 1, TM_F32, 1.0f);

    int64_t sparse_shape[] = {3, 3};
    tm_tensor* sparse = tm_sparse_coo(indices, values, sparse_shape, 2);

    ASSERT_NOT_NULL(sparse);
    ASSERT_TRUE(tm_is_sparse(sparse));
    ASSERT_INT_EQ(3, (int)tm_sparse_nnz(sparse));
    ASSERT_INT_EQ(TM_SPARSE_COO, tm_get_sparse_format(sparse));

    tm_free(sparse);
    tm_free(indices);
    tm_free(values);
}

static void test_sparse_csr_creation(void) {
    /* Create CSR representation of 3x3 identity matrix */
    int64_t rp_shape[] = {4};
    tm_tensor* row_ptr = tm_empty(rp_shape, 1, TM_I64);
    int64_t* rp_data = (int64_t*)row_ptr->data;
    rp_data[0] = 0; rp_data[1] = 1; rp_data[2] = 2; rp_data[3] = 3;

    int64_t ci_shape[] = {3};
    tm_tensor* col_ind = tm_empty(ci_shape, 1, TM_I64);
    int64_t* ci_data = (int64_t*)col_ind->data;
    ci_data[0] = 0; ci_data[1] = 1; ci_data[2] = 2;

    int64_t val_shape[] = {3};
    tm_tensor* values = tm_full(val_shape, 1, TM_F32, 1.0f);

    int64_t sparse_shape[] = {3, 3};
    tm_tensor* sparse = tm_sparse_csr(row_ptr, col_ind, values, sparse_shape, 2);

    ASSERT_NOT_NULL(sparse);
    ASSERT_TRUE(tm_is_sparse(sparse));
    ASSERT_INT_EQ(3, (int)tm_sparse_nnz(sparse));
    ASSERT_INT_EQ(TM_SPARSE_CSR, tm_get_sparse_format(sparse));

    tm_free(sparse);
    tm_free(row_ptr);
    tm_free(col_ind);
    tm_free(values);
}

static void test_dense_to_sparse_coo(void) {
    /* Create dense identity matrix */
    tm_tensor* dense = tm_eye(3, TM_F32);

    /* Convert to sparse */
    tm_tensor* sparse = tm_to_sparse(dense);

    ASSERT_NOT_NULL(sparse);
    ASSERT_TRUE(tm_is_sparse(sparse));
    ASSERT_INT_EQ(3, (int)tm_sparse_nnz(sparse));

    tm_free(sparse);
    tm_free(dense);
}

static void test_dense_to_sparse_csr(void) {
    /* Create dense identity matrix */
    tm_tensor* dense = tm_eye(3, TM_F32);

    /* Convert to sparse CSR */
    tm_tensor* sparse = tm_to_sparse_csr(dense);

    ASSERT_NOT_NULL(sparse);
    ASSERT_TRUE(tm_is_sparse(sparse));
    ASSERT_INT_EQ(3, (int)tm_sparse_nnz(sparse));
    ASSERT_INT_EQ(TM_SPARSE_CSR, tm_get_sparse_format(sparse));

    tm_free(sparse);
    tm_free(dense);
}

static void test_sparse_to_dense(void) {
    /* Create dense identity matrix */
    tm_tensor* dense = tm_eye(3, TM_F32);

    /* Convert to sparse and back */
    tm_tensor* sparse = tm_to_sparse(dense);
    tm_tensor* restored = tm_to_dense(sparse);

    ASSERT_NOT_NULL(restored);
    ASSERT_FALSE(tm_is_sparse(restored));

    /* Check values match */
    for (int64_t i = 0; i < 9; i++) {
        ASSERT_FLT_NEAR(tm_get_f32(dense, i), tm_get_f32(restored, i), 0.0001f);
    }

    tm_free(restored);
    tm_free(sparse);
    tm_free(dense);
}

static void test_sparse_spmm(void) {
    /* Create sparse identity matrix */
    tm_tensor* dense_eye = tm_eye(3, TM_F32);
    tm_tensor* sparse = tm_to_sparse(dense_eye);

    /* Create dense matrix to multiply */
    int64_t shape[] = {3, 2};
    tm_tensor* dense = tm_randn(shape, 2, TM_F32);

    /* SpMM: I @ dense = dense */
    tm_tensor* result = tm_spmm(sparse, dense);

    ASSERT_NOT_NULL(result);
    ASSERT_INT_EQ(3, (int)result->shape[0]);
    ASSERT_INT_EQ(2, (int)result->shape[1]);

    /* Result should equal dense (identity times dense) */
    for (int64_t i = 0; i < 6; i++) {
        ASSERT_FLT_NEAR(tm_get_f32(dense, i), tm_get_f32(result, i), 0.0001f);
    }

    tm_free(result);
    tm_free(dense);
    tm_free(sparse);
    tm_free(dense_eye);
}

static void test_sparse_spmv(void) {
    /* Create sparse identity matrix */
    tm_tensor* dense_eye = tm_eye(3, TM_F32);
    tm_tensor* sparse = tm_to_sparse(dense_eye);

    /* Create vector */
    int64_t shape[] = {3};
    tm_tensor* vec = tm_full(shape, 1, TM_F32, 2.0f);

    /* SpMV: I @ vec = vec */
    tm_tensor* result = tm_spmv(sparse, vec);

    ASSERT_NOT_NULL(result);
    ASSERT_INT_EQ(3, (int)result->numel);

    for (int64_t i = 0; i < 3; i++) {
        ASSERT_FLT_NEAR(2.0f, tm_get_f32(result, i), 0.0001f);
    }

    tm_free(result);
    tm_free(vec);
    tm_free(sparse);
    tm_free(dense_eye);
}

static void test_sparse_mul_scalar(void) {
    /* Create sparse matrix */
    tm_tensor* dense = tm_eye(3, TM_F32);
    tm_tensor* sparse = tm_to_sparse(dense);

    /* Multiply by scalar */
    tm_tensor* scaled = tm_sparse_mul_scalar(sparse, 2.0);

    ASSERT_NOT_NULL(scaled);
    ASSERT_TRUE(tm_is_sparse(scaled));

    /* Convert back and check values */
    tm_tensor* result = tm_to_dense(scaled);
    ASSERT_FLT_NEAR(2.0f, tm_get_f32(result, 0), 0.0001f);  /* [0,0] = 2 */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(result, 1), 0.0001f);  /* [0,1] = 0 */
    ASSERT_FLT_NEAR(2.0f, tm_get_f32(result, 4), 0.0001f);  /* [1,1] = 2 */

    tm_free(result);
    tm_free(scaled);
    tm_free(sparse);
    tm_free(dense);
}

static void test_sparse_format_conversion(void) {
    /* Create COO sparse matrix */
    tm_tensor* dense = tm_eye(3, TM_F32);
    tm_tensor* coo = tm_to_sparse(dense);

    /* Convert to CSR */
    tm_tensor* csr = tm_sparse_coo_to_csr(coo);
    ASSERT_NOT_NULL(csr);
    ASSERT_INT_EQ(TM_SPARSE_CSR, tm_get_sparse_format(csr));

    /* Convert back to COO */
    tm_tensor* coo2 = tm_sparse_csr_to_coo(csr);
    ASSERT_NOT_NULL(coo2);
    ASSERT_INT_EQ(TM_SPARSE_COO, tm_get_sparse_format(coo2));

    /* Check values are preserved */
    tm_tensor* result = tm_to_dense(coo2);
    for (int64_t i = 0; i < 9; i++) {
        ASSERT_FLT_NEAR(tm_get_f32(dense, i), tm_get_f32(result, i), 0.0001f);
    }

    tm_free(result);
    tm_free(coo2);
    tm_free(csr);
    tm_free(coo);
    tm_free(dense);
}

static void test_sparse_add(void) {
    /* Create two sparse matrices */
    tm_tensor* a_dense = tm_eye(3, TM_F32);
    tm_tensor* b_dense = tm_eye(3, TM_F32);

    tm_tensor* a = tm_to_sparse(a_dense);
    tm_tensor* b = tm_to_sparse(b_dense);

    /* Add them: I + I = 2*I */
    tm_tensor* sum = tm_sparse_add(a, b);
    ASSERT_NOT_NULL(sum);
    ASSERT_TRUE(tm_is_sparse(sum));

    /* Convert to dense and check */
    tm_tensor* result = tm_to_dense(sum);
    ASSERT_FLT_NEAR(2.0f, tm_get_f32(result, 0), 0.0001f);  /* [0,0] = 2 */
    ASSERT_FLT_NEAR(0.0f, tm_get_f32(result, 1), 0.0001f);  /* [0,1] = 0 */
    ASSERT_FLT_NEAR(2.0f, tm_get_f32(result, 4), 0.0001f);  /* [1,1] = 2 */

    tm_free(result);
    tm_free(sum);
    tm_free(b);
    tm_free(a);
    tm_free(b_dense);
    tm_free(a_dense);
}

static void test_sparse_values_indices(void) {
    /* Create sparse matrix */
    tm_tensor* dense = tm_eye(3, TM_F32);
    tm_tensor* sparse = tm_to_sparse(dense);

    /* Get values */
    tm_tensor* values = tm_sparse_values(sparse);
    ASSERT_NOT_NULL(values);
    ASSERT_INT_EQ(3, (int)values->numel);

    /* All values should be 1.0 */
    for (int64_t i = 0; i < 3; i++) {
        ASSERT_FLT_NEAR(1.0f, tm_get_f32(values, i), 0.0001f);
    }

    /* Get indices */
    tm_tensor* indices = tm_sparse_indices(sparse);
    ASSERT_NOT_NULL(indices);
    ASSERT_INT_EQ(2, indices->ndim);
    ASSERT_INT_EQ(2, (int)indices->shape[0]);
    ASSERT_INT_EQ(3, (int)indices->shape[1]);

    tm_free(indices);
    tm_free(values);
    tm_free(sparse);
    tm_free(dense);
}

static void test_sparse_coalesce(void) {
    /* Create sparse matrix and coalesce (should just work) */
    tm_tensor* dense = tm_eye(3, TM_F32);
    tm_tensor* sparse = tm_to_sparse(dense);

    tm_tensor* coalesced = tm_sparse_coalesce(sparse);
    ASSERT_NOT_NULL(coalesced);
    ASSERT_TRUE(tm_is_sparse(coalesced));
    ASSERT_INT_EQ(3, (int)tm_sparse_nnz(coalesced));

    tm_free(coalesced);
    tm_free(sparse);
    tm_free(dense);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running tensor tests...\n\n");

    /* Creation tests */
    RUN_TEST(zeros_creates_tensor);
    RUN_TEST(ones_creates_tensor);
    RUN_TEST(full_creates_tensor);
    RUN_TEST(randn_creates_tensor);
    RUN_TEST(clone_creates_copy);

    /* Accessor tests */
    RUN_TEST(get_set_f32);
    RUN_TEST(data_pointer);

    /* Shape tests */
    RUN_TEST(reshape);
    RUN_TEST(transpose);
    RUN_TEST(is_contiguous);

    /* Element-wise tests */
    RUN_TEST(add);
    RUN_TEST(sub);
    RUN_TEST(mul);
    RUN_TEST(mul_scalar);
    RUN_TEST(exp);
    RUN_TEST(relu);

    /* Matrix tests */
    RUN_TEST(matmul_2d);
    RUN_TEST(matmul_different_shapes);

    /* Reduction tests */
    RUN_TEST(sum_all);
    RUN_TEST(mean_all);

    /* NN op tests */
    RUN_TEST(softmax);
    RUN_TEST(layer_norm);

    /* Autograd tests */
    RUN_TEST(autograd_add);
    RUN_TEST(autograd_mul);
    RUN_TEST(no_grad_context);

    /* Edge case tests */
    RUN_TEST(null_safety);
    RUN_TEST(scalar_tensor);

    /* Broadcasting tests */
    RUN_TEST(broadcast_add_1d_to_2d);
    RUN_TEST(broadcast_mul_scalar_like);
    RUN_TEST(broadcast_sub);
    RUN_TEST(broadcast_div);
    RUN_TEST(broadcast_backward);

    /* P0 Feature tests */
    RUN_TEST(arange);
    RUN_TEST(cat);
    RUN_TEST(stack);
    RUN_TEST(split);
    RUN_TEST(permute);
    RUN_TEST(flatten);
    RUN_TEST(select);
    RUN_TEST(silu);
    RUN_TEST(rms_norm);
    RUN_TEST(linear);
    RUN_TEST(triu);
    RUN_TEST(sin_cos);
    RUN_TEST(clamp);
    RUN_TEST(argmax);
    RUN_TEST(logsumexp);
    RUN_TEST(abs);
    RUN_TEST(pow);
    RUN_TEST(where);
    RUN_TEST(masked_fill);
    RUN_TEST(eye);
    RUN_TEST(rsqrt);
    RUN_TEST(sub_div_scalar);
    RUN_TEST(inplace_ops);

    /* P1/P2 Feature tests */
    RUN_TEST(comparison_ops);
    RUN_TEST(maximum_minimum);
    RUN_TEST(leaky_relu);
    RUN_TEST(gather);
    RUN_TEST(index_select);
    RUN_TEST(narrow);
    RUN_TEST(expand);
    RUN_TEST(repeat);
    RUN_TEST(cumsum);
    RUN_TEST(var_std);
    RUN_TEST(prod);
    RUN_TEST(type_conversion);
    RUN_TEST(chunk);
    RUN_TEST(isnan_isinf);

    /* Full parity tests */
    RUN_TEST(mm);
    RUN_TEST(dot);
    RUN_TEST(mv);
    RUN_TEST(outer);
    RUN_TEST(floor_ceil_round);
    RUN_TEST(logical_ops);
    RUN_TEST(flip_roll);
    RUN_TEST(extra_inplace_ops);
    RUN_TEST(equal_allclose);
    RUN_TEST(like_functions);
    RUN_TEST(cumprod);
    RUN_TEST(any_all);
    RUN_TEST(norm);
    RUN_TEST(elu);
    RUN_TEST(trig_functions);

    /* Additional NN ops tests */
    RUN_TEST(hardswish);
    RUN_TEST(batch_norm);
    RUN_TEST(group_norm);
    RUN_TEST(instance_norm);
    RUN_TEST(scaled_dot_product_attention);
    RUN_TEST(split_merge_heads);
    RUN_TEST(conv1d);
    RUN_TEST(conv2d);
    RUN_TEST(pooling);
    RUN_TEST(adaptive_pooling);

    /* Loss function tests */
    RUN_TEST(l1_loss);
    RUN_TEST(smooth_l1_loss);
    RUN_TEST(nll_loss);
    RUN_TEST(kl_div);
    RUN_TEST(binary_cross_entropy);
    RUN_TEST(half_conversion);

    /* Additional loss function tests */
    RUN_TEST(cosine_embedding_loss);
    RUN_TEST(triplet_margin_loss);
    RUN_TEST(label_smoothing);

    /* Linear algebra tests */
    RUN_TEST(inverse);
    RUN_TEST(det);
    RUN_TEST(svd);
    RUN_TEST(eig);

    /* Autograd extension tests */
    RUN_TEST(retain_grad);
    RUN_TEST(explicit_grad);

    /* Higher-order gradient tests */
    RUN_TEST(hessian);
    RUN_TEST(hvp);

    /* Custom autograd tests */
    RUN_TEST(custom_autograd);

    /* In-place gradient tests */
    RUN_TEST(inplace_add_grad);
    RUN_TEST(inplace_mul_grad);
    RUN_TEST(inplace_exp_grad);
    RUN_TEST(inplace_relu_grad);
    RUN_TEST(inplace_sigmoid_grad);

    /* Additional NN activation tests */
    RUN_TEST(mish);
    RUN_TEST(selu);
    RUN_TEST(celu);
    RUN_TEST(relu6);
    RUN_TEST(glu);
    RUN_TEST(softsign);
    RUN_TEST(tanhshrink);
    RUN_TEST(hardsigmoid);
    RUN_TEST(threshold_fn);
    RUN_TEST(pad_fn);
    RUN_TEST(normalize_fn);
    RUN_TEST(interpolate_fn);

    /* Sparse tensor tests */
    RUN_TEST(sparse_coo_creation);
    RUN_TEST(sparse_csr_creation);
    RUN_TEST(dense_to_sparse_coo);
    RUN_TEST(dense_to_sparse_csr);
    RUN_TEST(sparse_to_dense);
    RUN_TEST(sparse_spmm);
    RUN_TEST(sparse_spmv);
    RUN_TEST(sparse_mul_scalar);
    RUN_TEST(sparse_format_conversion);
    RUN_TEST(sparse_add);
    RUN_TEST(sparse_values_indices);
    RUN_TEST(sparse_coalesce);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
