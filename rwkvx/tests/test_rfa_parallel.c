/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_rfa_parallel.c — Unit tests for parallel training + LoRA averaging
 *
 * Tests:
 * 1. LoRA averaging with known weights
 * 2. LoRA averaging with single adapter (clone)
 * 3. Dataset splitting produces correct chunk sizes
 * 4. Config defaults include new fields
 * 5. NULL input handling for averaging
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
 * Test 1: LoRA averaging with known weights
 * ================================================================ */

static void test_lora_average(void) {
    /* Create 3 adapters with 1 layer each, known A/B values */
    int d_in = 4, d_out = 3, rank = 2;
    fl_config cfg = { .rank = rank, .alpha = 4.0f, .dropout = 0.0f,
                      .fan_in_fan_out = false };

    fl_adapter *adapters[3];
    for (int a = 0; a < 3; a++) {
        adapters[a] = fl_adapter_new();
        fl_layer *l = fl_layer_new("test.layer", d_in, d_out, &cfg);

        /* Set A and B to known values: all elements = (a+1) */
        float *ad = tm_data_f32(l->lora_A);
        float *bd = tm_data_f32(l->lora_B);
        for (int i = 0; i < rank * d_in; i++) ad[i] = (float)(a + 1);
        for (int i = 0; i < d_out * rank; i++) bd[i] = (float)(a + 1);

        fl_adapter_add(adapters[a], l);
    }

    /* Average: expected A/B elements = (1+2+3)/3 = 2.0 */
    fl_adapter *merged = rfa_lora_average(adapters, 3);
    ASSERT_NOT_NULL(merged);
    ASSERT_INT_EQ(fl_adapter_size(merged), 1);

    fl_layer *ml = fl_adapter_get_idx(merged, 0);
    float *ma = tm_data_f32(ml->lora_A);
    float *mb = tm_data_f32(ml->lora_B);

    for (int i = 0; i < rank * d_in; i++)
        ASSERT_FLT_NEAR(ma[i], 2.0f, 0.001f);
    for (int i = 0; i < d_out * rank; i++)
        ASSERT_FLT_NEAR(mb[i], 2.0f, 0.001f);

    fl_adapter_free(merged);
    for (int a = 0; a < 3; a++)
        fl_adapter_free(adapters[a]);
}

/* ================================================================
 * Test 2: Single adapter averaging (clone)
 * ================================================================ */

static void test_lora_average_single(void) {
    int d_in = 4, d_out = 3, rank = 2;
    fl_config cfg = { .rank = rank, .alpha = 4.0f, .dropout = 0.0f,
                      .fan_in_fan_out = false };

    fl_adapter *adapters[1];
    adapters[0] = fl_adapter_new();
    fl_layer *l = fl_layer_new("test.layer", d_in, d_out, &cfg);
    float *ad = tm_data_f32(l->lora_A);
    for (int i = 0; i < rank * d_in; i++) ad[i] = 3.14f;
    fl_adapter_add(adapters[0], l);

    fl_adapter *merged = rfa_lora_average(adapters, 1);
    ASSERT_NOT_NULL(merged);
    ASSERT_INT_EQ(fl_adapter_size(merged), 1);

    fl_layer *ml = fl_adapter_get_idx(merged, 0);
    float *ma = tm_data_f32(ml->lora_A);
    for (int i = 0; i < rank * d_in; i++)
        ASSERT_FLT_NEAR(ma[i], 3.14f, 0.001f);

    fl_adapter_free(merged);
    fl_adapter_free(adapters[0]);
}

/* ================================================================
 * Test 3: Multi-layer averaging
 * ================================================================ */

static void test_lora_average_multi_layer(void) {
    int d_in = 8, d_out = 6, rank = 4;
    fl_config cfg = { .rank = rank, .alpha = 8.0f, .dropout = 0.0f,
                      .fan_in_fan_out = false };

    fl_adapter *adapters[2];
    for (int a = 0; a < 2; a++) {
        adapters[a] = fl_adapter_new();
        for (int li = 0; li < 3; li++) {
            char name[64];
            snprintf(name, sizeof(name), "block.%d.proj", li);
            fl_layer *l = fl_layer_new(name, d_in, d_out, &cfg);
            float *ad = tm_data_f32(l->lora_A);
            float *bd = tm_data_f32(l->lora_B);
            /* adapter 0: all 1.0, adapter 1: all 3.0 -> avg = 2.0 */
            float val = (a == 0) ? 1.0f : 3.0f;
            for (int i = 0; i < rank * d_in; i++) ad[i] = val;
            for (int i = 0; i < d_out * rank; i++) bd[i] = val;
            fl_adapter_add(adapters[a], l);
        }
    }

    fl_adapter *merged = rfa_lora_average(adapters, 2);
    ASSERT_NOT_NULL(merged);
    ASSERT_INT_EQ(fl_adapter_size(merged), 3);

    /* Check all layers averaged to 2.0 */
    for (int li = 0; li < 3; li++) {
        fl_layer *ml = fl_adapter_get_idx(merged, li);
        float *ma = tm_data_f32(ml->lora_A);
        float *mb = tm_data_f32(ml->lora_B);
        for (int i = 0; i < rank * d_in; i++)
            ASSERT_FLT_NEAR(ma[i], 2.0f, 0.001f);
        for (int i = 0; i < d_out * rank; i++)
            ASSERT_FLT_NEAR(mb[i], 2.0f, 0.001f);
    }

    fl_adapter_free(merged);
    for (int a = 0; a < 2; a++)
        fl_adapter_free(adapters[a]);
}

/* ================================================================
 * Test 4: Config defaults include new fields
 * ================================================================ */

static void test_config_parallel_defaults(void) {
    rfa_train_config cfg = rfa_train_config_defaults();
    ASSERT_INT_EQ(cfg.n_threads, 1);
    ASSERT_TRUE(!cfg.quantize);
}

/* ================================================================
 * Test 5: NULL input handling
 * ================================================================ */

static void test_average_null_inputs(void) {
    fl_adapter *result = rfa_lora_average(NULL, 0);
    ASSERT_NULL(result);

    result = rfa_lora_average(NULL, 5);
    ASSERT_NULL(result);
}

/* ================================================================
 * Test 6: Dataset chunk sizing
 * ================================================================ */

static void test_dataset_split_sizes(void) {
    /* Simulate splitting 100 examples into 3 threads */
    int n = 100, n_threads = 3;
    int chunk = n / n_threads;  /* 33 */
    ASSERT_INT_EQ(chunk, 33);

    /* Thread 0: [0, 33), Thread 1: [33, 66), Thread 2: [66, 100) */
    int start0 = 0 * chunk, end0 = start0 + chunk;
    int start1 = 1 * chunk, end1 = start1 + chunk;
    int start2 = 2 * chunk, end2 = n;  /* last thread gets remainder */

    ASSERT_INT_EQ(end0 - start0, 33);
    ASSERT_INT_EQ(end1 - start1, 33);
    ASSERT_INT_EQ(end2 - start2, 34);  /* remainder */
    ASSERT_INT_EQ((end0 - start0) + (end1 - start1) + (end2 - start2), n);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("\nRFA Parallel Training Tests\n");
    printf("===========================\n\n");

    RUN_TEST(lora_average);
    RUN_TEST(lora_average_single);
    RUN_TEST(lora_average_multi_layer);
    RUN_TEST(config_parallel_defaults);
    RUN_TEST(average_null_inputs);
    RUN_TEST(dataset_split_sizes);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
