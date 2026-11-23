/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../lora.h"

#include <string.h>
#include <stdlib.h>
#include <unistd.h>

/* ============================================================= */
/* Config Tests                                                  */
/* ============================================================= */

static void test_config_default(void) {
    fl_config cfg = fl_config_default();

    ASSERT_INT_EQ(8, cfg.rank);
    ASSERT_FLT_NEAR(16.0f, cfg.alpha, 0.001f);
    ASSERT_FLT_NEAR(0.0f, cfg.dropout, 0.001f);
    ASSERT_FALSE(cfg.fan_in_fan_out);
}

/* ============================================================= */
/* Layer Creation Tests                                          */
/* ============================================================= */

static void test_layer_create(void) {
    fl_config cfg = fl_config_default();
    fl_layer* layer = fl_layer_new("test.layer", 64, 128, &cfg);

    ASSERT_NOT_NULL(layer);
    ASSERT_STR_EQ("test.layer", layer->name);
    ASSERT_INT_EQ(64, layer->d_in);
    ASSERT_INT_EQ(128, layer->d_out);
    ASSERT_INT_EQ(8, layer->rank);
    ASSERT_FLT_NEAR(2.0f, layer->scale, 0.001f);  /* 16/8 = 2 */

    /* Check tensor shapes */
    ASSERT_NOT_NULL(layer->lora_A);
    ASSERT_NOT_NULL(layer->lora_B);

    ASSERT_INT_EQ(2, layer->lora_A->ndim);
    ASSERT_INT_EQ(8, (int)layer->lora_A->shape[0]);   /* rank */
    ASSERT_INT_EQ(64, (int)layer->lora_A->shape[1]);  /* d_in */

    ASSERT_INT_EQ(2, layer->lora_B->ndim);
    ASSERT_INT_EQ(128, (int)layer->lora_B->shape[0]); /* d_out */
    ASSERT_INT_EQ(8, (int)layer->lora_B->shape[1]);   /* rank */

    fl_layer_free(layer);
}

static void test_layer_custom_rank(void) {
    fl_config cfg = {
        .rank = 16,
        .alpha = 32.0f,
        .dropout = 0.0f,
        .fan_in_fan_out = false
    };

    fl_layer* layer = fl_layer_new("custom", 256, 512, &cfg);

    ASSERT_NOT_NULL(layer);
    ASSERT_INT_EQ(16, layer->rank);
    ASSERT_FLT_NEAR(2.0f, layer->scale, 0.001f);  /* 32/16 = 2 */

    ASSERT_INT_EQ(16, (int)layer->lora_A->shape[0]);
    ASSERT_INT_EQ(256, (int)layer->lora_A->shape[1]);
    ASSERT_INT_EQ(512, (int)layer->lora_B->shape[0]);
    ASSERT_INT_EQ(16, (int)layer->lora_B->shape[1]);

    fl_layer_free(layer);
}

static void test_layer_from_tensors(void) {
    /* Create A and B matrices */
    int64_t shape_A[] = {4, 32};  /* rank=4, d_in=32 */
    int64_t shape_B[] = {64, 4};  /* d_out=64, rank=4 */

    tm_tensor* A = tm_ones(shape_A, 2, TM_F32);
    tm_tensor* B = tm_ones(shape_B, 2, TM_F32);

    fl_layer* layer = fl_layer_from_tensors("from_tensors", A, B, 8.0f);

    ASSERT_NOT_NULL(layer);
    ASSERT_INT_EQ(32, layer->d_in);
    ASSERT_INT_EQ(64, layer->d_out);
    ASSERT_INT_EQ(4, layer->rank);
    ASSERT_FLT_NEAR(2.0f, layer->scale, 0.001f);  /* 8/4 = 2 */

    tm_release(A);
    tm_release(B);
    fl_layer_free(layer);
}

/* ============================================================= */
/* Forward Pass Tests                                            */
/* ============================================================= */

static void test_layer_forward_shape(void) {
    fl_config cfg = fl_config_default();
    fl_layer* layer = fl_layer_new("test", 32, 64, &cfg);

    /* Create input: (batch_size=4, d_in=32) */
    int64_t shape[] = {4, 32};
    tm_tensor* x = tm_ones(shape, 2, TM_F32);

    tm_tensor* out = fl_layer_forward(layer, x);

    ASSERT_NOT_NULL(out);
    ASSERT_INT_EQ(2, out->ndim);
    ASSERT_INT_EQ(4, (int)out->shape[0]);   /* batch_size */
    ASSERT_INT_EQ(64, (int)out->shape[1]);  /* d_out */

    tm_release(x);
    tm_release(out);
    fl_layer_free(layer);
}

static void test_layer_forward_values(void) {
    /* Create layer with known values */
    int64_t shape_A[] = {2, 4};   /* rank=2, d_in=4 */
    int64_t shape_B[] = {3, 2};   /* d_out=3, rank=2 */

    tm_tensor* A = tm_ones(shape_A, 2, TM_F32);
    tm_tensor* B = tm_ones(shape_B, 2, TM_F32);

    fl_layer* layer = fl_layer_from_tensors("known", A, B, 2.0f);  /* scale = 2/2 = 1 */

    /* Input of all ones: (1, 4) */
    int64_t shape_x[] = {1, 4};
    tm_tensor* x = tm_ones(shape_x, 2, TM_F32);

    tm_tensor* out = fl_layer_forward(layer, x);

    ASSERT_NOT_NULL(out);
    /* x @ A^T = [1,1,1,1] @ [[1,1,1,1],[1,1,1,1]]^T = [4, 4]
     * [4,4] @ B^T = [4,4] @ [[1,1],[1,1],[1,1]]^T = [8, 8, 8]
     * scale = 1, so output = [8, 8, 8]
     */
    float* data = tm_data_f32(out);
    ASSERT_FLT_NEAR(8.0f, data[0], 0.001f);
    ASSERT_FLT_NEAR(8.0f, data[1], 0.001f);
    ASSERT_FLT_NEAR(8.0f, data[2], 0.001f);

    tm_release(A);
    tm_release(B);
    tm_release(x);
    tm_release(out);
    fl_layer_free(layer);
}

/* ============================================================= */
/* Merge Tests                                                   */
/* ============================================================= */

static void test_layer_merge(void) {
    /* Create layer with known values */
    int64_t shape_A[] = {2, 4};   /* rank=2, d_in=4 */
    int64_t shape_B[] = {3, 2};   /* d_out=3, rank=2 */

    tm_tensor* A = tm_ones(shape_A, 2, TM_F32);
    tm_tensor* B = tm_ones(shape_B, 2, TM_F32);

    fl_layer* layer = fl_layer_from_tensors("merge", A, B, 2.0f);  /* scale = 1 */

    /* Base weight matrix: (d_out=3, d_in=4) all zeros */
    int64_t shape_W[] = {3, 4};
    tm_tensor* W = tm_zeros(shape_W, 2, TM_F32);

    tm_tensor* merged = fl_layer_get_merged(layer, W);

    ASSERT_NOT_NULL(merged);
    ASSERT_INT_EQ(3, (int)merged->shape[0]);
    ASSERT_INT_EQ(4, (int)merged->shape[1]);

    /* B @ A = [[1,1],[1,1],[1,1]] @ [[1,1,1,1],[1,1,1,1]] = [[2,2,2,2],[2,2,2,2],[2,2,2,2]] */
    /* scale = 1, W = 0, so merged = scale * B @ A = [[2,2,2,2], ...] */
    float* data = tm_data_f32(merged);
    for (int i = 0; i < 12; i++) {
        ASSERT_FLT_NEAR(2.0f, data[i], 0.001f);
    }

    tm_release(A);
    tm_release(B);
    tm_release(W);
    tm_release(merged);
    fl_layer_free(layer);
}

static void test_layer_merge_adds_to_weight(void) {
    int64_t shape_A[] = {1, 2};   /* rank=1, d_in=2 */
    int64_t shape_B[] = {2, 1};   /* d_out=2, rank=1 */

    tm_tensor* A = tm_ones(shape_A, 2, TM_F32);
    tm_tensor* B = tm_ones(shape_B, 2, TM_F32);

    fl_layer* layer = fl_layer_from_tensors("add", A, B, 1.0f);  /* scale = 1 */

    /* Base weight: all 5s */
    int64_t shape_W[] = {2, 2};
    tm_tensor* W = tm_full(shape_W, 2, TM_F32, 5.0);

    tm_tensor* merged = fl_layer_get_merged(layer, W);

    /* B @ A = [[1]] @ [[1,1]] = [[1,1],[1,1]]
     * scale = 1, W = 5, merged = W + scale * BA = 5 + 1 = 6
     */
    float* data = tm_data_f32(merged);
    ASSERT_FLT_NEAR(6.0f, data[0], 0.001f);
    ASSERT_FLT_NEAR(6.0f, data[1], 0.001f);
    ASSERT_FLT_NEAR(6.0f, data[2], 0.001f);
    ASSERT_FLT_NEAR(6.0f, data[3], 0.001f);

    tm_release(A);
    tm_release(B);
    tm_release(W);
    tm_release(merged);
    fl_layer_free(layer);
}

/* ============================================================= */
/* Adapter Tests                                                 */
/* ============================================================= */

static void test_adapter_create(void) {
    fl_adapter* adapter = fl_adapter_new();

    ASSERT_NOT_NULL(adapter);
    ASSERT_INT_EQ(0, fl_adapter_size(adapter));

    fl_adapter_free(adapter);
}

static void test_adapter_add_layers(void) {
    fl_adapter* adapter = fl_adapter_new();
    fl_config cfg = fl_config_default();

    fl_layer* layer1 = fl_layer_new("layer1", 32, 64, &cfg);
    fl_layer* layer2 = fl_layer_new("layer2", 64, 128, &cfg);

    fl_adapter_add(adapter, layer1);
    fl_adapter_add(adapter, layer2);

    ASSERT_INT_EQ(2, fl_adapter_size(adapter));

    fl_layer* found = fl_adapter_get(adapter, "layer1");
    ASSERT_NOT_NULL(found);
    ASSERT_STR_EQ("layer1", found->name);

    found = fl_adapter_get(adapter, "layer2");
    ASSERT_NOT_NULL(found);
    ASSERT_STR_EQ("layer2", found->name);

    ASSERT_NULL(fl_adapter_get(adapter, "nonexistent"));

    fl_adapter_free(adapter);  /* Also frees layers */
}

static void test_adapter_get_by_index(void) {
    fl_adapter* adapter = fl_adapter_new();
    fl_config cfg = fl_config_default();

    fl_layer* layer = fl_layer_new("indexed", 32, 64, &cfg);
    fl_adapter_add(adapter, layer);

    fl_layer* found = fl_adapter_get_idx(adapter, 0);
    ASSERT_NOT_NULL(found);
    ASSERT_STR_EQ("indexed", found->name);

    ASSERT_NULL(fl_adapter_get_idx(adapter, 1));
    ASSERT_NULL(fl_adapter_get_idx(adapter, -1));

    fl_adapter_free(adapter);
}

static void test_adapter_num_params(void) {
    fl_adapter* adapter = fl_adapter_new();

    fl_config cfg = {.rank = 4, .alpha = 8.0f, .dropout = 0.0f, .fan_in_fan_out = false};

    /* Layer 1: A is (4 x 32) = 128, B is (64 x 4) = 256, total = 384 */
    fl_layer* layer1 = fl_layer_new("l1", 32, 64, &cfg);
    fl_adapter_add(adapter, layer1);

    /* Layer 2: A is (4 x 64) = 256, B is (128 x 4) = 512, total = 768 */
    fl_layer* layer2 = fl_layer_new("l2", 64, 128, &cfg);
    fl_adapter_add(adapter, layer2);

    /* Total: 384 + 768 = 1152 */
    ASSERT_INT_EQ(1152, (int)fl_adapter_num_params(adapter));

    fl_adapter_free(adapter);
}

/* ============================================================= */
/* Adapter I/O Tests                                             */
/* ============================================================= */

static void test_adapter_save_and_load(void) {
    const char* path = "/tmp/test_foundry_lora.safetensors";

    /* Create adapter with known values */
    fl_adapter* adapter = fl_adapter_new();
    fl_config cfg = {.rank = 2, .alpha = 4.0f, .dropout = 0.0f, .fan_in_fan_out = false};

    fl_layer* layer = fl_layer_new("test.proj", 8, 16, &cfg);
    fl_adapter_add(adapter, layer);

    /* Save */
    bool saved = fl_adapter_save(adapter, path);
    ASSERT_TRUE(saved);

    fl_adapter_free(adapter);

    /* Load */
    fl_adapter* loaded = fl_adapter_load(path, &cfg);
    ASSERT_NOT_NULL(loaded);
    ASSERT_INT_EQ(1, fl_adapter_size(loaded));

    fl_layer* loaded_layer = fl_adapter_get(loaded, "test.proj");
    ASSERT_NOT_NULL(loaded_layer);
    ASSERT_INT_EQ(8, loaded_layer->d_in);
    ASSERT_INT_EQ(16, loaded_layer->d_out);
    ASSERT_INT_EQ(2, loaded_layer->rank);

    fl_adapter_free(loaded);
    unlink(path);
}

/* ============================================================= */
/* Edge Cases                                                    */
/* ============================================================= */

static void test_null_safety(void) {
    ASSERT_NULL(fl_layer_new(NULL, 32, 64, NULL));
    ASSERT_NULL(fl_layer_new("test", 0, 64, NULL));
    ASSERT_NULL(fl_layer_new("test", 32, 0, NULL));
    ASSERT_NULL(fl_layer_from_tensors(NULL, NULL, NULL, 1.0f));
    ASSERT_NULL(fl_layer_forward(NULL, NULL));

    ASSERT_INT_EQ(0, fl_adapter_size(NULL));
    ASSERT_NULL(fl_adapter_get(NULL, "test"));
    ASSERT_NULL(fl_adapter_get_idx(NULL, 0));
    ASSERT_INT_EQ(0, (int)fl_adapter_num_params(NULL));

    fl_layer_free(NULL);   /* Should not crash */
    fl_adapter_free(NULL); /* Should not crash */

    ASSERT_TRUE(1);
}

static void test_layer_b_initialized_to_zero(void) {
    fl_config cfg = fl_config_default();
    fl_layer* layer = fl_layer_new("test", 16, 32, &cfg);

    /* B should be zeros initially (standard LoRA init) */
    float* data = tm_data_f32(layer->lora_B);
    for (int64_t i = 0; i < layer->lora_B->numel; i++) {
        ASSERT_FLT_NEAR(0.0f, data[i], 0.0001f);
    }

    fl_layer_free(layer);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running lora tests...\n\n");

    /* Config tests */
    RUN_TEST(config_default);

    /* Layer creation tests */
    RUN_TEST(layer_create);
    RUN_TEST(layer_custom_rank);
    RUN_TEST(layer_from_tensors);

    /* Forward pass tests */
    RUN_TEST(layer_forward_shape);
    RUN_TEST(layer_forward_values);

    /* Merge tests */
    RUN_TEST(layer_merge);
    RUN_TEST(layer_merge_adds_to_weight);

    /* Adapter tests */
    RUN_TEST(adapter_create);
    RUN_TEST(adapter_add_layers);
    RUN_TEST(adapter_get_by_index);
    RUN_TEST(adapter_num_params);

    /* Adapter I/O tests */
    RUN_TEST(adapter_save_and_load);

    /* Edge cases */
    RUN_TEST(null_safety);
    RUN_TEST(layer_b_initialized_to_zero);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
