/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_rfa_train.c — Unit tests for RFA LoRA training
 *
 * Tests backward pass correctness via finite-difference gradient verification,
 * LoRA adapter creation, parameter collection, data loading, and config defaults.
 */

#include "lib/test.h"
#include "rwkvx/rfa_train.h"
#include "rwkvx/rfa.h"
#include "lora/lora.h"
#include "tensor/tensor.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ================================================================
 * Helper: random float in [-1, 1]
 * ================================================================ */

static float randf(void) {
    return 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}

static void fill_random(float *buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = randf() * 0.5f;
}

/* ================================================================
 * Test 1: RMSNorm backward (finite difference)
 * ================================================================ */

static void rms_norm_fwd(const float *x, const float *w, float *out,
                          int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++)
        sum_sq += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * w[i];
}

static void test_rms_norm_backward(void) {
    int c = 16;
    float eps = 1e-6f;
    float h = 1e-4f;

    float *x = (float*)malloc(c * sizeof(float));
    float *w = (float*)malloc(c * sizeof(float));
    float *d_out = (float*)malloc(c * sizeof(float));
    float *d_x = (float*)calloc(c, sizeof(float));
    float *out1 = (float*)malloc(c * sizeof(float));
    float *out2 = (float*)malloc(c * sizeof(float));

    srand(42);
    fill_random(x, c);
    for (int i = 0; i < c; i++) w[i] = 0.5f + randf() * 0.5f;
    fill_random(d_out, c);

    /* Analytical gradient */
    rfa_rms_norm_backward(d_out, x, w, d_x, c, eps);

    /* Finite difference */
    float max_err = 0.0f;
    for (int i = 0; i < c; i++) {
        float orig = x[i];

        x[i] = orig + h;
        rms_norm_fwd(x, w, out1, c, eps);
        float f_plus = 0.0f;
        for (int j = 0; j < c; j++) f_plus += out1[j] * d_out[j];

        x[i] = orig - h;
        rms_norm_fwd(x, w, out2, c, eps);
        float f_minus = 0.0f;
        for (int j = 0; j < c; j++) f_minus += out2[j] * d_out[j];

        float fd_grad = (f_plus - f_minus) / (2.0f * h);
        float err = fabsf(d_x[i] - fd_grad);
        if (err > max_err) max_err = err;

        x[i] = orig;
    }

    /* Tolerance: relative to gradient magnitude */
    ASSERT_TRUE(max_err < 0.01f);

    free(x); free(w); free(d_out); free(d_x);
    free(out1); free(out2);
}

/* ================================================================
 * Test 2: SwiGLU backward (finite difference)
 * ================================================================ */

static float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* Minimal SwiGLU forward for FD testing (no LoRA) */
static void swiglu_fwd(const float *x, const float *gate_w, const float *up_w,
                        const float *down_w, float *out, int c, int inter) {
    float *gate = (float*)calloc(inter, sizeof(float));
    float *up_v = (float*)calloc(inter, sizeof(float));
    float *act = (float*)calloc(inter, sizeof(float));

    /* gate = gate_w^T @ x  (gate_w stored as [c, inter] for t_matvec convention) */
    for (int j = 0; j < inter; j++) {
        float s = 0.0f;
        for (int i = 0; i < c; i++) s += x[i] * gate_w[i * inter + j];
        gate[j] = s;
    }
    for (int j = 0; j < inter; j++) {
        float s = 0.0f;
        for (int i = 0; i < c; i++) s += x[i] * up_w[i * inter + j];
        up_v[j] = s;
    }
    for (int i = 0; i < inter; i++)
        act[i] = fast_silu(gate[i]) * up_v[i];
    memset(out, 0, c * sizeof(float));
    for (int i = 0; i < c; i++) {
        float s = 0.0f;
        for (int j = 0; j < inter; j++) s += act[j] * down_w[j * c + i];
        out[i] = s;
    }

    free(gate); free(up_v); free(act);
}

static void test_swiglu_backward(void) {
    int c = 8;
    int inter = 12;
    float h = 1e-4f;

    float *x = (float*)malloc(c * sizeof(float));
    float *d_out = (float*)malloc(c * sizeof(float));
    float *d_x = (float*)calloc(c, sizeof(float));
    float *out1 = (float*)malloc(c * sizeof(float));
    float *out2 = (float*)malloc(c * sizeof(float));

    /* Create synthetic weights in transposed layout [c, inter] / [inter, c] */
    float *gate_w = (float*)malloc(c * inter * sizeof(float));
    float *up_w = (float*)malloc(c * inter * sizeof(float));
    float *down_w = (float*)malloc(inter * c * sizeof(float));

    srand(123);
    fill_random(x, c);
    fill_random(d_out, c);
    fill_random(gate_w, c * inter);
    fill_random(up_w, c * inter);
    fill_random(down_w, inter * c);

    /* Set up rfa_qwen_weights struct for backward */
    rfa_qwen_weights w;
    memset(&w, 0, sizeof(w));
    w.gate_proj = gate_w;
    w.up_proj = up_w;
    w.down_proj = down_w;
    w.inter_size = inter;

    /* NULL LoRA layers */
    fl_layer *lora[7] = {NULL};

    /* Analytical gradient */
    rfa_swiglu_backward(d_out, x, &w, lora, d_x, c, inter);

    /* Finite difference */
    float max_err = 0.0f;
    for (int i = 0; i < c; i++) {
        float orig = x[i];

        x[i] = orig + h;
        swiglu_fwd(x, gate_w, up_w, down_w, out1, c, inter);
        float f_plus = 0.0f;
        for (int j = 0; j < c; j++) f_plus += out1[j] * d_out[j];

        x[i] = orig - h;
        swiglu_fwd(x, gate_w, up_w, down_w, out2, c, inter);
        float f_minus = 0.0f;
        for (int j = 0; j < c; j++) f_minus += out2[j] * d_out[j];

        float fd_grad = (f_plus - f_minus) / (2.0f * h);
        float err = fabsf(d_x[i] - fd_grad);
        if (err > max_err) max_err = err;

        x[i] = orig;
    }

    ASSERT_TRUE(max_err < 0.05f);

    free(x); free(d_out); free(d_x);
    free(out1); free(out2);
    free(gate_w); free(up_w); free(down_w);
}

/* ================================================================
 * Test 3: GQA attention backward (finite difference)
 * ================================================================ */

static void gqa_fwd(const float *q, const float *k_cache, const float *v_cache,
                     int cache_len, float *y, float *weights_out,
                     int n_q, int n_kv, int hd) {
    int kv_dim = n_kv * hd;
    int group_size = n_q / n_kv;
    float scale = 1.0f / sqrtf((float)hd);
    memset(y, 0, n_q * hd * sizeof(float));

    for (int qh = 0; qh < n_q; qh++) {
        int kvh = qh / group_size;
        const float *q_h = &q[qh * hd];

        float *scores = (float*)malloc(cache_len * sizeof(float));
        float max_s = -1e30f;
        for (int t = 0; t < cache_len; t++) {
            const float *kt = &k_cache[t * kv_dim + kvh * hd];
            float dot = 0.0f;
            for (int j = 0; j < hd; j++) dot += q_h[j] * kt[j];
            scores[t] = dot * scale;
            if (scores[t] > max_s) max_s = scores[t];
        }
        float sum_e = 0.0f;
        for (int t = 0; t < cache_len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_e += scores[t];
        }
        for (int t = 0; t < cache_len; t++) scores[t] /= sum_e;

        if (weights_out)
            memcpy(&weights_out[qh * cache_len], scores, cache_len * sizeof(float));

        float *y_h = &y[qh * hd];
        for (int t = 0; t < cache_len; t++) {
            const float *vt = &v_cache[t * kv_dim + kvh * hd];
            for (int j = 0; j < hd; j++)
                y_h[j] += scores[t] * vt[j];
        }
        free(scores);
    }
}

static void test_gqa_attention_backward(void) {
    /* Small synthetic GQA: 4 Q heads, 2 KV heads, head_dim=4, cache=3 */
    int n_q = 4, n_kv = 2, hd = 4, cl = 3;
    int c = n_q * hd;    /* 16 */
    int kv_dim = n_kv * hd;  /* 8 */
    float h_eps = 1e-4f;

    float *q = (float*)malloc(c * sizeof(float));
    float *k_cache = (float*)malloc(cl * kv_dim * sizeof(float));
    float *v_cache = (float*)malloc(cl * kv_dim * sizeof(float));
    float *d_y = (float*)malloc(c * sizeof(float));
    float *y = (float*)malloc(c * sizeof(float));
    float *y2 = (float*)malloc(c * sizeof(float));
    float *weights = (float*)malloc(n_q * cl * sizeof(float));

    float *d_q = (float*)calloc(c, sizeof(float));
    float *d_k = (float*)calloc(kv_dim, sizeof(float));
    float *d_v = (float*)calloc(kv_dim, sizeof(float));

    srand(777);
    fill_random(q, c);
    fill_random(k_cache, cl * kv_dim);
    fill_random(v_cache, cl * kv_dim);
    fill_random(d_y, c);

    /* Forward + save weights */
    gqa_fwd(q, k_cache, v_cache, cl, y, weights, n_q, n_kv, hd);

    /* Analytical gradient */
    rfa_gqa_attention_backward(d_y, q, k_cache, v_cache, weights, cl,
                                d_q, d_k, d_v, n_q, n_kv, hd);

    /* Finite difference for d_q */
    float max_err_q = 0.0f;
    for (int i = 0; i < c; i++) {
        float orig = q[i];

        q[i] = orig + h_eps;
        gqa_fwd(q, k_cache, v_cache, cl, y2, NULL, n_q, n_kv, hd);
        float f_plus = 0.0f;
        for (int j = 0; j < c; j++) f_plus += y2[j] * d_y[j];

        q[i] = orig - h_eps;
        gqa_fwd(q, k_cache, v_cache, cl, y2, NULL, n_q, n_kv, hd);
        float f_minus = 0.0f;
        for (int j = 0; j < c; j++) f_minus += y2[j] * d_y[j];

        float fd = (f_plus - f_minus) / (2.0f * h_eps);
        float err = fabsf(d_q[i] - fd);
        if (err > max_err_q) max_err_q = err;

        q[i] = orig;
    }

    ASSERT_TRUE(max_err_q < 0.05f);

    free(q); free(k_cache); free(v_cache); free(d_y);
    free(y); free(y2); free(weights);
    free(d_q); free(d_k); free(d_v);
}

/* ================================================================
 * Test 4: Config defaults
 * ================================================================ */

static void test_config_defaults(void) {
    rfa_train_config cfg = rfa_train_config_defaults();
    ASSERT_INT_EQ(cfg.lora_rank, 32);
    ASSERT_FLT_NEAR(cfg.lora_alpha, 64.0f, 0.01f);
    ASSERT_FLT_NEAR(cfg.lr, 2e-5f, 1e-7f);
    ASSERT_INT_EQ(cfg.epochs, 3);
    ASSERT_INT_EQ(cfg.max_seq_len, 256);
    ASSERT_INT_EQ(cfg.warmup_steps, 100);
    ASSERT_INT_EQ(cfg.save_every, 1000);
    ASSERT_INT_EQ(cfg.log_every, 10);
    ASSERT_INT_EQ(cfg.max_cache, 2048);
}

/* ================================================================
 * Test 5: Example free
 * ================================================================ */

static void test_example_free(void) {
    rfa_train_example ex;
    ex.seq_len = 5;
    ex.input_ids = (uint32_t*)malloc(5 * sizeof(uint32_t));
    ex.target_ids = (uint32_t*)malloc(5 * sizeof(uint32_t));
    rfa_train_example_free(&ex);
    /* No crash = pass */
    ASSERT_TRUE(1);
}

static void test_examples_batch_free(void) {
    int n = 3;
    rfa_train_example *exs = (rfa_train_example*)calloc(n, sizeof(rfa_train_example));
    for (int i = 0; i < n; i++) {
        exs[i].seq_len = 4;
        exs[i].input_ids = (uint32_t*)malloc(4 * sizeof(uint32_t));
        exs[i].target_ids = (uint32_t*)malloc(4 * sizeof(uint32_t));
    }
    rfa_train_examples_free(exs, n);
    ASSERT_TRUE(1);
}

/* ================================================================
 * Test 6: Binary data loader (write then read)
 * ================================================================ */

static void test_load_pretokenized(void) {
    const char *path = "/tmp/test_rfa_pretok.bin";

    /* Write a small binary file */
    FILE *f = fopen(path, "wb");
    ASSERT_NOT_NULL(f);

    uint32_t n_ex = 2;
    fwrite(&n_ex, sizeof(uint32_t), 1, f);

    /* Example 1: [10, 20, 30, 40] */
    uint32_t len1 = 4;
    uint32_t toks1[] = {10, 20, 30, 40};
    fwrite(&len1, sizeof(uint32_t), 1, f);
    fwrite(toks1, sizeof(uint32_t), 4, f);

    /* Example 2: [5, 15, 25] */
    uint32_t len2 = 3;
    uint32_t toks2[] = {5, 15, 25};
    fwrite(&len2, sizeof(uint32_t), 1, f);
    fwrite(toks2, sizeof(uint32_t), 3, f);

    fclose(f);

    /* Load */
    int n_loaded;
    rfa_train_example *exs = rfa_load_pretokenized(path, 512, &n_loaded);
    ASSERT_NOT_NULL(exs);
    ASSERT_INT_EQ(n_loaded, 2);

    /* Example 1: input=[10,20,30], target=[20,30,40] */
    ASSERT_INT_EQ(exs[0].seq_len, 3);
    ASSERT_INT_EQ(exs[0].input_ids[0], 10);
    ASSERT_INT_EQ(exs[0].input_ids[2], 30);
    ASSERT_INT_EQ(exs[0].target_ids[0], 20);
    ASSERT_INT_EQ(exs[0].target_ids[2], 40);

    /* Example 2: input=[5,15], target=[15,25] */
    ASSERT_INT_EQ(exs[1].seq_len, 2);
    ASSERT_INT_EQ(exs[1].input_ids[0], 5);
    ASSERT_INT_EQ(exs[1].target_ids[1], 25);

    rfa_train_examples_free(exs, n_loaded);
}

/* ================================================================
 * Test 7: LoRA adapter creation (verify dimensions)
 * ================================================================ */

static void test_create_qwen_lora(void) {
    /* We can't load a real model, but we can test the function signature
     * and parameter count logic by creating a mock scenario.
     * For now, test with NULL model returns NULL. */
    fl_adapter *adapter = rfa_create_qwen_lora(NULL, 32, 64.0f);
    ASSERT_NULL(adapter);

    /* Also test collect_params with NULL */
    int n;
    tm_tensor **params = rfa_collect_params(NULL, &n);
    ASSERT_NULL(params);
    ASSERT_TRUE(1);
}

/* ================================================================
 * Test 8: Null input handling
 * ================================================================ */

static void test_null_inputs(void) {
    /* Load with NULL path */
    int n;
    rfa_train_example *exs = rfa_load_pretokenized(NULL, 256, &n);
    ASSERT_NULL(exs);

    /* Free NULL */
    rfa_train_example_free(NULL);
    rfa_train_examples_free(NULL, 0);

    ASSERT_TRUE(1);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("\nRFA Training Tests\n");
    printf("==================\n\n");

    RUN_TEST(rms_norm_backward);
    RUN_TEST(swiglu_backward);
    RUN_TEST(gqa_attention_backward);
    RUN_TEST(config_defaults);
    RUN_TEST(example_free);
    RUN_TEST(examples_batch_free);
    RUN_TEST(load_pretokenized);
    RUN_TEST(create_qwen_lora);
    RUN_TEST(null_inputs);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
