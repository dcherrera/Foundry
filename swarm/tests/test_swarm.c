/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Swarm Module Tests
 *
 * Tests for protocol, config, compression, and basic functionality.
 * Note: Full integration tests require running coordinator and workers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../protocol.h"
#include "../config.h"
#include "../network.h"
#include "../coordinator.h"
#include "../worker.h"
#include "../../tensor/tensor.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN_TEST(name) do { \
    printf("  Testing %s... ", #name); \
    fflush(stdout); \
    test_##name(); \
    tests_passed++; \
    printf("PASSED\n"); \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAILED\n"); \
        printf("    Assertion failed: %s\n", #cond); \
        printf("    At %s:%d\n", __FILE__, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_NEAR(a, b, eps) ASSERT(fabs((a) - (b)) < (eps))
#define ASSERT_STR_EQ(a, b) ASSERT(strcmp((a), (b)) == 0)

/* ============================================================= */
/* Protocol Tests                                                */
/* ============================================================= */

TEST(fp16_conversion) {
    /* Test FP16 round-trip conversion */
    float values[] = {0.0f, 1.0f, -1.0f, 0.5f, 100.0f, 0.001f};
    int n = sizeof(values) / sizeof(values[0]);

    uint16_t* fp16 = malloc(n * sizeof(uint16_t));
    float* result = malloc(n * sizeof(float));

    sw_fp32_to_fp16(values, fp16, n);
    sw_fp16_to_fp32(fp16, result, n);

    for (int i = 0; i < n; i++) {
        /* FP16 has limited precision */
        float eps = fabs(values[i]) * 0.01f + 0.001f;
        ASSERT_NEAR(values[i], result[i], eps);
    }

    free(fp16);
    free(result);
}

TEST(topk_compression) {
    /* Test TopK compression */
    float values[] = {0.1f, 5.0f, 0.2f, -3.0f, 0.3f, 2.0f, 0.0f, -1.0f};
    int n = 8;
    int k = 3;

    uint32_t* indices = malloc(k * sizeof(uint32_t));
    float* topk_vals = malloc(k * sizeof(float));

    sw_topk_compress(values, n, k, indices, topk_vals);

    /* Should find the top 3 by absolute value: 5.0 (idx 1), -3.0 (idx 3), 2.0 (idx 5) */
    /* Values should be sorted by magnitude */
    ASSERT_NEAR(topk_vals[0], 5.0f, 0.001f);
    ASSERT_EQ(indices[0], 1);

    ASSERT_NEAR(topk_vals[1], -3.0f, 0.001f);
    ASSERT_EQ(indices[1], 3);

    ASSERT_NEAR(topk_vals[2], 2.0f, 0.001f);
    ASSERT_EQ(indices[2], 5);

    /* Test decompression */
    float* decompressed = calloc(n, sizeof(float));
    sw_topk_decompress(indices, topk_vals, k, decompressed, n);

    ASSERT_NEAR(decompressed[1], 5.0f, 0.001f);
    ASSERT_NEAR(decompressed[3], -3.0f, 0.001f);
    ASSERT_NEAR(decompressed[5], 2.0f, 0.001f);
    ASSERT_NEAR(decompressed[0], 0.0f, 0.001f);  /* Not in top-k */

    free(indices);
    free(topk_vals);
    free(decompressed);
}

TEST(message_alloc_free) {
    /* Test message allocation and validation */
    uint8_t* msg = sw_msg_alloc(SW_MSG_REGISTER, 100);
    ASSERT_NE(msg, NULL);

    sw_msg_header* hdr = sw_msg_get_header(msg);
    ASSERT_EQ(hdr->type, SW_MSG_REGISTER);
    ASSERT_EQ(hdr->length, 100);
    ASSERT(sw_msg_validate(msg, SW_HEADER_SIZE + 100));

    sw_msg_free(msg);
}

TEST(hardware_detection) {
    /* Test hardware profile detection */
    sw_hardware_profile profile = sw_detect_profile();

    /* Should return a valid profile */
    ASSERT(profile >= SW_PROFILE_RPI3 && profile <= SW_PROFILE_UNKNOWN);

    /* Test batch size recommendation */
    uint32_t batch = sw_profile_batch_size(profile);
    ASSERT(batch >= 2 && batch <= 256);
}

/* ============================================================= */
/* Config Tests                                                  */
/* ============================================================= */

TEST(config_parse) {
    /* Create a test config file */
    const char* config_path = "/tmp/test_swarm_config.toml";
    FILE* f = fopen(config_path, "w");
    ASSERT_NE(f, NULL);

    fprintf(f, "# Test config\n");
    fprintf(f, "[network]\n");
    fprintf(f, "port = 7100\n");
    fprintf(f, "host = \"localhost\"\n");
    fprintf(f, "\n");
    fprintf(f, "[training]\n");
    fprintf(f, "learning_rate = 1e-4\n");
    fprintf(f, "batch_size = 32\n");
    fprintf(f, "enabled = true\n");
    fclose(f);

    /* Load and test */
    sw_config* cfg = sw_config_load(config_path);
    ASSERT_NE(cfg, NULL);

    ASSERT_EQ(sw_config_get_int(cfg, "network", "port", 0), 7100);
    ASSERT_STR_EQ(sw_config_get_str(cfg, "network", "host", ""), "localhost");
    ASSERT_NEAR(sw_config_get_float(cfg, "training", "learning_rate", 0), 1e-4f, 1e-6f);
    ASSERT_EQ(sw_config_get_int(cfg, "training", "batch_size", 0), 32);
    ASSERT_EQ(sw_config_get_bool(cfg, "training", "enabled", false), true);

    /* Test defaults */
    ASSERT_EQ(sw_config_get_int(cfg, "network", "nonexistent", 42), 42);
    ASSERT_EQ(sw_config_has_key(cfg, "network", "port"), true);
    ASSERT_EQ(sw_config_has_key(cfg, "network", "nonexistent"), false);

    sw_config_free(cfg);
    remove(config_path);
}

/* ============================================================= */
/* Coordinator Tests                                             */
/* ============================================================= */

TEST(coordinator_default_config) {
    sw_coordinator_config config = sw_coordinator_default_config();

    ASSERT_EQ(config.port, SW_DEFAULT_PORT);
    ASSERT(config.learning_rate > 0);
    ASSERT(config.beta1 > 0 && config.beta1 < 1);
    ASSERT(config.beta2 > 0 && config.beta2 < 1);
    ASSERT(config.min_workers > 0);
}

TEST(coordinator_create_free) {
    /* Create dummy weights */
    int64_t shape[] = {64, 64};
    tm_tensor* w1 = tm_randn(shape, 2, TM_F32);
    tm_tensor* weights[] = {w1};

    sw_coordinator_config config = sw_coordinator_default_config();
    config.min_workers = 1;

    sw_coordinator* coord = sw_coordinator_new(weights, 1, &config);
    ASSERT_NE(coord, NULL);

    ASSERT_EQ(sw_coordinator_get_step(coord), 0);
    ASSERT_EQ(sw_coordinator_num_workers(coord), 0);

    sw_coordinator_free(coord);
    tm_free(w1);
}

/* ============================================================= */
/* Worker Tests                                                  */
/* ============================================================= */

TEST(worker_default_config) {
    sw_worker_config config = sw_worker_default_config();

    ASSERT_EQ(config.coordinator_port, SW_DEFAULT_PORT);
    ASSERT_STR_EQ(config.coordinator_host, "localhost");
    ASSERT(config.reconnect_attempts > 0);
}

TEST(worker_create_free) {
    sw_worker_config config = sw_worker_default_config();
    strcpy(config.worker_id, "test-worker");

    sw_worker* worker = sw_worker_new(&config);
    ASSERT_NE(worker, NULL);

    ASSERT_STR_EQ(sw_worker_get_id(worker), "test-worker");
    ASSERT_EQ((int)sw_worker_get_state(worker), (int)SW_WSTATE_DISCONNECTED);
    ASSERT_EQ(sw_worker_is_connected(worker), false);

    sw_worker_free(worker);
}

TEST(worker_auto_id) {
    sw_worker_config config = sw_worker_default_config();
    /* Leave worker_id empty */

    sw_worker* worker = sw_worker_new(&config);
    ASSERT_NE(worker, NULL);

    /* Should auto-generate an ID */
    const char* id = sw_worker_get_id(worker);
    ASSERT_NE(id, NULL);
    ASSERT(strlen(id) > 0);

    sw_worker_free(worker);
}

/* ============================================================= */
/* Tensor In-place Operations Tests                              */
/* ============================================================= */

TEST(tensor_zeros_like) {
    int64_t shape[] = {3, 4};
    tm_tensor* t = tm_randn(shape, 2, TM_F32);
    tm_tensor* z = tm_zeros_like(t);

    ASSERT_NE(z, NULL);
    ASSERT_EQ(z->shape[0], 3);
    ASSERT_EQ(z->shape[1], 4);
    ASSERT_EQ(tm_numel(z), 12);

    /* All elements should be zero */
    float* data = tm_data_f32(z);
    for (int i = 0; i < 12; i++) {
        ASSERT_NEAR(data[i], 0.0f, 1e-6f);
    }

    tm_free(t);
    tm_free(z);
}

TEST(tensor_fill_inplace) {
    int64_t shape[] = {2, 3};
    tm_tensor* t = tm_zeros(shape, 2, TM_F32);

    tm_fill_(t, 3.14f);

    float* data = tm_data_f32(t);
    for (int i = 0; i < 6; i++) {
        ASSERT_NEAR(data[i], 3.14f, 1e-6f);
    }

    tm_free(t);
}

TEST(tensor_add_inplace) {
    int64_t shape[] = {4};
    tm_tensor* a = tm_full(shape, 1, TM_F32, 1.0);
    tm_tensor* b = tm_full(shape, 1, TM_F32, 2.0);

    tm_add_(a, b);

    float* data = tm_data_f32(a);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(data[i], 3.0f, 1e-6f);
    }

    tm_free(a);
    tm_free(b);
}

TEST(tensor_scale_inplace) {
    int64_t shape[] = {3};
    tm_tensor* t = tm_full(shape, 1, TM_F32, 2.0);

    tm_scale_(t, 0.5f);

    float* data = tm_data_f32(t);
    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(data[i], 1.0f, 1e-6f);
    }

    tm_free(t);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n");
    printf("===========================================\n");
    printf("  Foundry Swarm Tests\n");
    printf("===========================================\n\n");

    printf("Protocol Tests:\n");
    RUN_TEST(fp16_conversion);
    RUN_TEST(topk_compression);
    RUN_TEST(message_alloc_free);
    RUN_TEST(hardware_detection);
    printf("\n");

    printf("Config Tests:\n");
    RUN_TEST(config_parse);
    printf("\n");

    printf("Coordinator Tests:\n");
    RUN_TEST(coordinator_default_config);
    RUN_TEST(coordinator_create_free);
    printf("\n");

    printf("Worker Tests:\n");
    RUN_TEST(worker_default_config);
    RUN_TEST(worker_create_free);
    RUN_TEST(worker_auto_id);
    printf("\n");

    printf("Tensor In-place Tests:\n");
    RUN_TEST(tensor_zeros_like);
    RUN_TEST(tensor_fill_inplace);
    RUN_TEST(tensor_add_inplace);
    RUN_TEST(tensor_scale_inplace);
    printf("\n");

    printf("===========================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("===========================================\n\n");

    return tests_failed > 0 ? 1 : 0;
}
