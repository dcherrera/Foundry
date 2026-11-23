/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../attention.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================= */
/* Config Tests                                                  */
/* ============================================================= */

static void test_config_default(void) {
    fa_config cfg = fa_config_default(8, 64);

    ASSERT_INT_EQ(8, cfg.num_heads);
    ASSERT_INT_EQ(64, cfg.head_dim);
    ASSERT_FALSE(cfg.causal);
    ASSERT_FLT_NEAR(1.0f / sqrtf(64.0f), cfg.scale, 0.001f);
}

/* ============================================================= */
/* Causal Mask Tests                                             */
/* ============================================================= */

/* Takes a pointer, not a float: under -ffast-math clang marks float parameters
 * nofpclass(inf nan) and may fold any test on them. */
static int is_negative_infinity(const float *value) {
    uint32_t bits;
    memcpy(&bits, value, sizeof(bits));
    return bits == 0xFF800000u;
}

static void test_causal_mask(void) {
    tm_tensor* mask = fa_causal_mask(4);

    ASSERT_NOT_NULL(mask);
    ASSERT_INT_EQ(4, mask->ndim);
    ASSERT_INT_EQ(1, (int)mask->shape[0]);
    ASSERT_INT_EQ(4, (int)mask->shape[2]);
    ASSERT_INT_EQ(4, (int)mask->shape[3]);

    float* data = tm_data_f32(mask);

    /* Diagonal and below should be 0 */
    ASSERT_FLT_NEAR(0.0f, data[0], 0.001f);  /* (0,0) */
    ASSERT_FLT_NEAR(0.0f, data[4], 0.001f);  /* (1,0) */
    ASSERT_FLT_NEAR(0.0f, data[5], 0.001f);  /* (1,1) */

    /* Above diagonal should be -inf. By bit pattern: this file builds with -ffast-math,
     * and GCC folds isinf() to 0 under -ffinite-math-only. */
    ASSERT_TRUE(is_negative_infinity(&data[1]));  /* (0,1) */
    ASSERT_TRUE(is_negative_infinity(&data[2]));  /* (0,2) */

    tm_release(mask);
}

/* ============================================================= */
/* Basic Attention Tests                                         */
/* ============================================================= */

static void test_attention_2d(void) {
    /* Simple 2D attention: (seq_len=4, dim=8) */
    int64_t shape[] = {4, 8};

    tm_tensor* Q = tm_ones(shape, 2, TM_F32);
    tm_tensor* K = tm_ones(shape, 2, TM_F32);
    tm_tensor* V = tm_ones(shape, 2, TM_F32);

    fa_config cfg = fa_config_default(1, 8);

    tm_tensor* out = fa_attention(Q, K, V, &cfg);

    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(2, out->ndim);
    ASSERT_INT_EQ(4, (int)out->shape[0]);
    ASSERT_INT_EQ(8, (int)out->shape[1]);

    /* With all ones, attention weights are uniform, output should be ones */
    float* data = tm_data_f32(out);
    ASSERT_FLT_NEAR(1.0f, data[0], 0.001f);

    tm_release(Q);
    tm_release(K);
    tm_release(V);
    tm_release(out);
}

static void test_attention_3d(void) {
    /* Batched attention: (batch=2, seq=4, dim=8) */
    int64_t shape[] = {2, 4, 8};

    tm_tensor* Q = tm_ones(shape, 3, TM_F32);
    tm_tensor* K = tm_ones(shape, 3, TM_F32);
    tm_tensor* V = tm_ones(shape, 3, TM_F32);

    fa_config cfg = fa_config_default(1, 8);

    tm_tensor* out = fa_attention(Q, K, V, &cfg);

    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(3, out->ndim);
    ASSERT_INT_EQ(2, (int)out->shape[0]);
    ASSERT_INT_EQ(4, (int)out->shape[1]);
    ASSERT_INT_EQ(8, (int)out->shape[2]);

    tm_release(Q);
    tm_release(K);
    tm_release(V);
    tm_release(out);
}

static void test_attention_causal(void) {
    int64_t shape[] = {4, 8};

    tm_tensor* Q = tm_ones(shape, 2, TM_F32);
    tm_tensor* K = tm_ones(shape, 2, TM_F32);
    tm_tensor* V = tm_randn(shape, 2, TM_F32);

    fa_config cfg = fa_config_default(1, 8);
    cfg.causal = true;

    tm_tensor* out = fa_attention(Q, K, V, &cfg);

    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(4, (int)out->shape[0]);

    /* First position should only see itself */
    /* Last position should see all positions */

    tm_release(Q);
    tm_release(K);
    tm_release(V);
    tm_release(out);
}

/* ============================================================= */
/* Multi-Head Attention Tests                                    */
/* ============================================================= */

static void test_multi_head(void) {
    /* Input: (seq=4, embed=32) */
    /* Projections: (32, 32) each */
    int64_t x_shape[] = {4, 32};
    int64_t w_shape[] = {32, 32};

    tm_tensor* x = tm_randn(x_shape, 2, TM_F32);
    tm_tensor* Wq = tm_randn(w_shape, 2, TM_F32);
    tm_tensor* Wk = tm_randn(w_shape, 2, TM_F32);
    tm_tensor* Wv = tm_randn(w_shape, 2, TM_F32);
    tm_tensor* Wo = tm_randn(w_shape, 2, TM_F32);

    fa_config cfg = fa_config_default(4, 8);  /* 4 heads, 8 dim each = 32 */

    tm_tensor* out = fa_multi_head(x, Wq, Wk, Wv, Wo, &cfg);

    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(2, out->ndim);
    ASSERT_INT_EQ(4, (int)out->shape[0]);

    tm_release(x);
    tm_release(Wq);
    tm_release(Wk);
    tm_release(Wv);
    tm_release(Wo);
    tm_release(out);
}

/* ============================================================= */
/* KV Cache Tests                                                */
/* ============================================================= */

static void test_context_create(void) {
    fa_context* ctx = fa_context_new(1, 128, 8, 64, 4);

    ASSERT_NOT_NULL(ctx);
    ASSERT_INT_EQ(0, fa_context_seq_len(ctx));

    fa_context_free(ctx);
}

static void test_context_cached_attention(void) {
    fa_context* ctx = fa_context_new(1, 64, 4, 8, 1);
    ASSERT_NOT_NULL(ctx);

    int64_t shape[] = {4, 32};  /* 4 tokens, dim=32 (4 heads * 8) */

    tm_tensor* Q = tm_ones(shape, 2, TM_F32);
    tm_tensor* K = tm_ones(shape, 2, TM_F32);
    tm_tensor* V = tm_ones(shape, 2, TM_F32);

    fa_config cfg = fa_config_default(4, 8);

    /* First forward - cache empty */
    tm_tensor* out1 = fa_attention_cached(Q, K, V, ctx, 0, 0, &cfg);
    ASSERT_NOT_NULL(out1);
    ASSERT_INT_EQ(4, fa_context_seq_len(ctx));

    tm_release(Q);
    tm_release(K);
    tm_release(V);
    tm_release(out1);
    fa_context_free(ctx);
}

static void test_context_clear(void) {
    fa_context* ctx = fa_context_new(1, 64, 4, 8, 1);
    ASSERT_NOT_NULL(ctx);

    int64_t shape[] = {4, 32};
    tm_tensor* Q = tm_ones(shape, 2, TM_F32);
    tm_tensor* K = tm_ones(shape, 2, TM_F32);
    tm_tensor* V = tm_ones(shape, 2, TM_F32);

    fa_config cfg = fa_config_default(4, 8);
    tm_tensor* out = fa_attention_cached(Q, K, V, ctx, 0, 0, &cfg);

    ASSERT_INT_EQ(4, fa_context_seq_len(ctx));

    fa_context_clear(ctx);
    ASSERT_INT_EQ(0, fa_context_seq_len(ctx));

    tm_release(Q);
    tm_release(K);
    tm_release(V);
    tm_release(out);
    fa_context_free(ctx);
}

/* ============================================================= */
/* RoPE Tests                                                    */
/* ============================================================= */

static void test_rope_freqs(void) {
    tm_tensor* freqs = fa_rope_freqs(128, 64);

    ASSERT_NOT_NULL(freqs);
    ASSERT_INT_EQ(2, freqs->ndim);
    ASSERT_INT_EQ(128, (int)freqs->shape[0]);
    ASSERT_INT_EQ(32, (int)freqs->shape[1]);  /* head_dim / 2 */

    /* First position should be all zeros */
    float* data = tm_data_f32(freqs);
    ASSERT_FLT_NEAR(0.0f, data[0], 0.001f);

    tm_release(freqs);
}

static void test_rope_apply(void) {
    int64_t shape[] = {4, 64};

    tm_tensor* Q = tm_ones(shape, 2, TM_F32);
    tm_tensor* K = tm_ones(shape, 2, TM_F32);

    /* Store original values */
    float q_orig = tm_data_f32(Q)[0];

    fa_rope(Q, K, 0, 64);

    /* Values should be modified */
    float q_new = tm_data_f32(Q)[0];

    /* At position 0, rotation is 0, so values should be unchanged */
    ASSERT_FLT_NEAR(q_orig, q_new, 0.001f);

    tm_release(Q);
    tm_release(K);
}

/* ============================================================= */
/* Edge Cases                                                    */
/* ============================================================= */

static void test_null_safety(void) {
    ASSERT_NULL(fa_attention(NULL, NULL, NULL, NULL));
    ASSERT_NULL(fa_multi_head(NULL, NULL, NULL, NULL, NULL, NULL));
    ASSERT_NULL(fa_attention_cached(NULL, NULL, NULL, NULL, 0, 0, NULL));
    ASSERT_INT_EQ(0, fa_context_seq_len(NULL));

    fa_context_free(NULL);  /* Should not crash */
    fa_context_clear(NULL); /* Should not crash */
    fa_rope(NULL, NULL, 0, 64); /* Should not crash */

    ASSERT_TRUE(1);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running attention tests...\n\n");

    /* Config tests */
    RUN_TEST(config_default);

    /* Causal mask tests */
    RUN_TEST(causal_mask);

    /* Basic attention tests */
    RUN_TEST(attention_2d);
    RUN_TEST(attention_3d);
    RUN_TEST(attention_causal);

    /* Multi-head tests */
    RUN_TEST(multi_head);

    /* KV cache tests */
    RUN_TEST(context_create);
    RUN_TEST(context_cached_attention);
    RUN_TEST(context_clear);

    /* RoPE tests */
    RUN_TEST(rope_freqs);
    RUN_TEST(rope_apply);

    /* Edge cases */
    RUN_TEST(null_safety);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
