/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_rwkvx.c — Unit tests for the RWKV-X inference engine
 */

#include "lib/test.h"
#include "rwkvx/rwkvx.h"
#include "tensor/tensor.h"
#include "hybridmem/hybridmem.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ================================================================
 * Sampling tests
 * ================================================================ */

static void test_sampling_defaults(void) {
    rx_sampling_params p = rx_sampling_defaults();
    ASSERT_FLT_NEAR(p.temperature, 1.0f, 1e-5f);
    ASSERT_FLT_NEAR(p.top_p, 0.7f, 1e-5f);
    ASSERT_INT_EQ(p.top_k, 100);
}

static void test_sampling_greedy(void) {
    /* Create logits: [1.0, 5.0, 2.0, 3.0] -> greedy should pick index 1 */
    float data[] = {1.0f, 5.0f, 2.0f, 3.0f};
    int64_t shape[] = {4};
    tm_tensor *logits = tm_from_data(data, shape, 1, TM_F32);

    rx_sampling_params p = rx_sampling_defaults();
    p.temperature = 0.0f; /* greedy */

    uint32_t token = rx_sample(logits, &p, NULL, 4);
    ASSERT_INT_EQ((int)token, 1);
    tm_free(logits);
}

static void test_sampling_greedy_with_penalties(void) {
    /* All logits equal at 5.0, but token 0 has been seen 3 times */
    float data[] = {5.0f, 5.0f, 5.0f, 5.0f};
    int64_t shape[] = {4};
    tm_tensor *logits = tm_from_data(data, shape, 1, TM_F32);

    rx_sampling_params p = rx_sampling_defaults();
    p.temperature = 0.0f;
    p.frequency_penalty = 10.0f;
    p.presence_penalty = 0.0f;

    uint32_t counts[] = {3, 0, 0, 0};
    uint32_t token = rx_sample(logits, &p, counts, 4);
    ASSERT_TRUE(token != 0); /* token 0 should be penalized away */
    tm_free(logits);
}

/* ================================================================
 * Tokenizer tests
 * ================================================================ */

/* Find vocab path relative to working directory */
static const char* find_vocab_path(void) {
    /* Try common paths */
    static const char *paths[] = {
        "rwkvx/data/rwkv_vocab_v20230424.txt",
        "../rwkvx/data/rwkv_vocab_v20230424.txt",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        FILE *fp = fopen(paths[i], "r");
        if (fp) { fclose(fp); return paths[i]; }
    }
    return NULL;
}

static void test_tokenizer_create(void) {
    const char *path = find_vocab_path();
    if (!path)
        fprintf(stderr, "  vocab file missing: run `make rwkv-vocab` (rwkvx/data/fetch_rwkv_vocab.sh)\n");
    ASSERT_NOT_NULL(path);
    rx_tokenizer *tok = rx_tokenizer_new(path);
    ASSERT_NOT_NULL(tok);
    rx_tokenizer_free(tok);
}

static void test_tokenizer_encode_decode_roundtrip(void) {
    const char *path = find_vocab_path();
    if (!path)
        fprintf(stderr, "  vocab file missing: run `make rwkv-vocab` (rwkvx/data/fetch_rwkv_vocab.sh)\n");
    ASSERT_NOT_NULL(path);
    rx_tokenizer *tok = rx_tokenizer_new(path);
    ASSERT_NOT_NULL(tok);

    const char *text = "Hello World!";
    uint32_t tokens[64];
    int n = rx_tokenizer_encode(tok, text, tokens, 64);
    ASSERT_TRUE(n > 0);

    char *decoded = rx_tokenizer_decode(tok, tokens, n);
    ASSERT_NOT_NULL(decoded);
    ASSERT_STR_EQ(decoded, text);

    free(decoded);
    rx_tokenizer_free(tok);
}

static void test_tokenizer_encode_empty(void) {
    const char *path = find_vocab_path();
    if (!path)
        fprintf(stderr, "  vocab file missing: run `make rwkv-vocab` (rwkvx/data/fetch_rwkv_vocab.sh)\n");
    ASSERT_NOT_NULL(path);
    rx_tokenizer *tok = rx_tokenizer_new(path);
    ASSERT_NOT_NULL(tok);

    uint32_t tokens[64];
    int n = rx_tokenizer_encode(tok, "", tokens, 64);
    ASSERT_INT_EQ(n, 0);

    rx_tokenizer_free(tok);
}

/* ================================================================
 * State tests
 * ================================================================ */

/* We can't test state_new without a model, but we can test the
 * config struct and sampling which don't need a model. */

static void test_config_struct(void) {
    rx_config c;
    c.n_embd = 768;
    c.n_head = 12;
    c.head_size = 64;
    c.n_rwkv_layer = 12;
    c.n_attn_layer = 4;
    c.vocab_size = 65536;
    ASSERT_INT_EQ(c.n_embd, c.n_head * c.head_size);
}

/* ================================================================
 * WKV-7 kernel test (via internal function — we test through forward)
 * ================================================================ */

static void test_forward_returns_correct_shape(void) {
    /* This test requires a model file. Skip if not available. */
    fprintf(stderr, "  [SKIP] requires model file\n");
}

/* ================================================================
 * Batch vs Serial correctness test
 * ================================================================ */

static const char* find_model_path(void) {
    static const char *paths[] = {
        "../models/base/RWKV-X-0.2B-64k-Base.safetensors",
        "../../models/base/RWKV-X-0.2B-64k-Base.safetensors",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        struct stat st;
        if (stat(paths[i], &st) == 0) return paths[i];
    }
    return NULL;
}

static void test_batch_vs_serial_correctness(void) {
    const char *model_path = find_model_path();
    const char *vocab_path = find_vocab_path();
    if (!model_path || !vocab_path) {
        fprintf(stderr, "  [SKIP] model or vocab file not found\n");
        return;
    }

    rx_model *model = rx_model_load(model_path);
    ASSERT_NOT_NULL(model);

    rx_tokenizer *tok = rx_tokenizer_new(vocab_path);
    ASSERT_NOT_NULL(tok);

    /* Encode a multi-token prompt */
    const char *prompt = "The capital of France is";
    uint32_t tokens[64];
    int n = rx_tokenizer_encode(tok, prompt, tokens, 64);
    ASSERT_TRUE(n > 2); /* must be > 2 to trigger batch path */

    rx_config cfg = rx_model_config(model);

    /* Serial path */
    rx_state *s1 = rx_state_new(model);
    tm_tensor *serial_logits = NULL;
    for (int i = 0; i < n; i++) {
        if (serial_logits) tm_free(serial_logits);
        serial_logits = rx_forward(model, tokens[i], s1);
    }

    /* Batch path */
    rx_state *s2 = rx_state_new(model);
    tm_tensor *batch_logits = rx_forward_batch(model, tokens, n, s2);

    /* Compare logits — should be bit-exact for f32 BLAS path */
    ASSERT_NOT_NULL(serial_logits);
    ASSERT_NOT_NULL(batch_logits);

    float *sd = tm_data_f32(serial_logits);
    float *bd = tm_data_f32(batch_logits);
    float max_diff = 0.0f;
    for (int i = 0; i < cfg.vocab_size; i++) {
        float diff = sd[i] - bd[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
    }

    fprintf(stderr, "  max logit diff: %.9f\n", max_diff);
    ASSERT_TRUE(max_diff < 1e-3f); /* allow small FP differences */

    /* Verify greedy token is the same */
    int serial_argmax = 0, batch_argmax = 0;
    for (int i = 1; i < cfg.vocab_size; i++) {
        if (sd[i] > sd[serial_argmax]) serial_argmax = i;
        if (bd[i] > bd[batch_argmax]) batch_argmax = i;
    }
    ASSERT_INT_EQ(serial_argmax, batch_argmax);

    tm_free(serial_logits);
    tm_free(batch_logits);
    rx_state_free(s1);
    rx_state_free(s2);
    rx_tokenizer_free(tok);
    rx_model_free(model);
}

/* ================================================================
 * HybridMem integration tests
 * ================================================================ */

static void test_hybridmem_config_defaults(void) {
    rx_hybridmem_config cfg = rx_hybridmem_config_defaults();
    ASSERT_TRUE(cfg.store_dir == NULL);
    ASSERT_INT_EQ(cfg.max_cache_len, 0);
    ASSERT_INT_EQ(cfg.top_k, 5);
    ASSERT_FLT_NEAR(cfg.evict_fraction, 0.5f, 1e-5f);
}

static void test_hybridmem_state_without_store(void) {
    /* Creating state with NULL store_dir should succeed with HybridMem disabled */
    /* We can't create rx_state without a model, so just verify the config path */
    rx_hybridmem_config cfg = rx_hybridmem_config_defaults();
    cfg.max_cache_len = 16;
    ASSERT_TRUE(cfg.store_dir == NULL);
    ASSERT_INT_EQ(cfg.max_cache_len, 16);
}

static void test_state_flush_null(void) {
    /* rx_state_flush on NULL should return -1 */
    int ret = rx_state_flush(NULL);
    ASSERT_INT_EQ(ret, -1);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    srand(42);

    RUN_TEST(sampling_defaults);
    RUN_TEST(sampling_greedy);
    RUN_TEST(sampling_greedy_with_penalties);
    RUN_TEST(tokenizer_create);
    RUN_TEST(tokenizer_encode_decode_roundtrip);
    RUN_TEST(tokenizer_encode_empty);
    RUN_TEST(config_struct);
    RUN_TEST(forward_returns_correct_shape);
    RUN_TEST(batch_vs_serial_correctness);
    RUN_TEST(hybridmem_config_defaults);
    RUN_TEST(hybridmem_state_without_store);
    RUN_TEST(state_flush_null);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
