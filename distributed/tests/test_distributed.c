/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../distributed.h"

#include <string.h>
#include <stdlib.h>

/* ============================================================= */
/* Init Tests                                                    */
/* ============================================================= */

static void test_init_single_process(void) {
    fd_config cfg = fd_init(1, 0);

    ASSERT_INT_EQ(1, cfg.world_size);
    ASSERT_INT_EQ(0, cfg.rank);
    ASSERT_TRUE(cfg.is_main);
    ASSERT_TRUE(fd_is_initialized());

    fd_cleanup();
    ASSERT_FALSE(fd_is_initialized());
}

static void test_init_multi_process(void) {
    fd_config cfg = fd_init(4, 2);

    ASSERT_INT_EQ(4, cfg.world_size);
    ASSERT_INT_EQ(2, cfg.rank);
    ASSERT_FALSE(cfg.is_main);

    fd_cleanup();
}

static void test_get_config(void) {
    fd_init(2, 1);
    fd_config cfg = fd_get_config();

    ASSERT_INT_EQ(2, cfg.world_size);
    ASSERT_INT_EQ(1, cfg.rank);

    fd_cleanup();
}

/* ============================================================= */
/* Collective Ops Tests                                          */
/* ============================================================= */

static void test_all_reduce_single(void) {
    fd_init(1, 0);

    int64_t shape[] = {4};
    tm_tensor* t = tm_ones(shape, 1, TM_F32);

    /* Single process: should be no-op */
    fd_all_reduce(t);

    float* data = tm_data_f32(t);
    ASSERT_FLT_NEAR(1.0f, data[0], 0.001f);

    tm_release(t);
    fd_cleanup();
}

static void test_all_reduce_mean(void) {
    fd_init(4, 0);

    int64_t shape[] = {4};
    tm_tensor* t = tm_full(shape, 1, TM_F32, 4.0);

    fd_all_reduce_mean(t);

    /* world_size=4, so values should be divided by 4 */
    float* data = tm_data_f32(t);
    ASSERT_FLT_NEAR(1.0f, data[0], 0.001f);

    tm_release(t);
    fd_cleanup();
}

static void test_broadcast(void) {
    fd_init(1, 0);

    int64_t shape[] = {4};
    tm_tensor* t = tm_ones(shape, 1, TM_F32);

    /* Single process: should be no-op */
    fd_broadcast(t, 0);

    ASSERT_FLT_NEAR(1.0f, tm_data_f32(t)[0], 0.001f);

    tm_release(t);
    fd_cleanup();
}

static void test_all_gather(void) {
    fd_init(2, 0);

    int64_t shape[] = {3};
    tm_tensor* t = tm_ones(shape, 1, TM_F32);

    tm_tensor** gathered = fd_all_gather(t);

    ASSERT_NOT_NULL(gathered);
    ASSERT_NOT_NULL(gathered[0]);
    ASSERT_NOT_NULL(gathered[1]);

    /* Both should have same values */
    ASSERT_FLT_NEAR(1.0f, tm_data_f32(gathered[0])[0], 0.001f);
    ASSERT_FLT_NEAR(1.0f, tm_data_f32(gathered[1])[0], 0.001f);

    tm_release(gathered[0]);
    tm_release(gathered[1]);
    free(gathered);
    tm_release(t);
    fd_cleanup();
}

static void test_barrier(void) {
    fd_init(1, 0);

    /* Should not hang or crash */
    fd_barrier();

    ASSERT_TRUE(1);
    fd_cleanup();
}

/* ============================================================= */
/* Gradient Accumulation Tests                                   */
/* ============================================================= */

static void test_grad_accum_create(void) {
    int64_t shape[] = {4};
    tm_tensor* params[2];
    params[0] = tm_zeros(shape, 1, TM_F32);
    params[1] = tm_zeros(shape, 1, TM_F32);

    fd_grad_accum* accum = fd_grad_accum_new(params, 2, 4);

    ASSERT_NOT_NULL(accum);
    ASSERT_FALSE(fd_grad_accum_should_step(accum));

    fd_grad_accum_free(accum);
    tm_release(params[0]);
    tm_release(params[1]);
}

static void test_grad_accum_step(void) {
    int64_t shape[] = {2};
    tm_tensor* params[1];
    params[0] = tm_zeros(shape, 1, TM_F32);
    tm_set_requires_grad(params[0], true);
    params[0]->grad = tm_full(shape, 1, TM_F32, 1.0);

    fd_grad_accum* accum = fd_grad_accum_new(params, 1, 2);

    /* First step */
    fd_grad_accum_step(accum, params);
    ASSERT_FALSE(fd_grad_accum_should_step(accum));

    /* Second step */
    fd_grad_accum_step(accum, params);
    ASSERT_TRUE(fd_grad_accum_should_step(accum));

    /* Finish and check average */
    fd_grad_accum_finish(accum, params);

    /* After 2 steps with grad=1, average should be 1 */
    float* grad = tm_data_f32(params[0]->grad);
    ASSERT_FLT_NEAR(1.0f, grad[0], 0.001f);

    fd_grad_accum_free(accum);
    /* tm_release handles grad automatically */
    tm_release(params[0]);
}

static void test_grad_accum_reset(void) {
    int64_t shape[] = {2};
    tm_tensor* params[1];
    params[0] = tm_zeros(shape, 1, TM_F32);
    tm_set_requires_grad(params[0], true);
    params[0]->grad = tm_full(shape, 1, TM_F32, 1.0);

    fd_grad_accum* accum = fd_grad_accum_new(params, 1, 2);

    fd_grad_accum_step(accum, params);
    ASSERT_FALSE(fd_grad_accum_should_step(accum));

    fd_grad_accum_reset(accum);

    /* After reset, should need full accumulation_steps again */
    fd_grad_accum_step(accum, params);
    ASSERT_FALSE(fd_grad_accum_should_step(accum));

    fd_grad_accum_free(accum);
    /* tm_release handles grad automatically */
    tm_release(params[0]);
}

/* ============================================================= */
/* Data Parallel Tests                                           */
/* ============================================================= */

static void test_shard_indices_single(void) {
    fd_init(1, 0);

    int shard_size;
    int* indices = fd_shard_indices(10, &shard_size);

    ASSERT_NOT_NULL(indices);
    ASSERT_INT_EQ(10, shard_size);

    /* All indices should be present */
    for (int i = 0; i < 10; i++) {
        ASSERT_INT_EQ(i, indices[i]);
    }

    free(indices);
    fd_cleanup();
}

static void test_shard_indices_multi(void) {
    /* Rank 0 of 4 */
    fd_init(4, 0);

    int shard_size;
    int* indices = fd_shard_indices(10, &shard_size);

    ASSERT_NOT_NULL(indices);
    /* 10/4 = 2 base, 2 remainder -> rank 0 gets 3 */
    ASSERT_INT_EQ(3, shard_size);
    ASSERT_INT_EQ(0, indices[0]);
    ASSERT_INT_EQ(1, indices[1]);
    ASSERT_INT_EQ(2, indices[2]);

    free(indices);
    fd_cleanup();

    /* Rank 3 of 4 */
    fd_init(4, 3);
    indices = fd_shard_indices(10, &shard_size);

    /* rank 3 gets 2 elements (no extra from remainder) */
    ASSERT_INT_EQ(2, shard_size);

    free(indices);
    fd_cleanup();
}

static void test_shard_batch(void) {
    fd_init(1, 0);

    int64_t shape[] = {8, 4};
    tm_tensor* batch = tm_ones(shape, 2, TM_F32);

    tm_tensor* shard = fd_shard_batch(batch, 0);

    ASSERT_NOT_NULL(shard);
    /* Single process: should return same batch */
    ASSERT_INT_EQ(8, (int)shard->shape[0]);

    tm_release(shard);
    tm_release(batch);
    fd_cleanup();
}

/* ============================================================= */
/* Checkpointing Tests                                           */
/* ============================================================= */

static void test_should_save_main(void) {
    fd_init(2, 0);  /* Main rank */
    ASSERT_TRUE(fd_should_save());
    fd_cleanup();
}

static void test_should_save_non_main(void) {
    fd_init(2, 1);  /* Non-main rank */
    ASSERT_FALSE(fd_should_save());
    fd_cleanup();
}

/* ============================================================= */
/* Utility Tests                                                 */
/* ============================================================= */

static void test_get_seed(void) {
    fd_init(4, 2);

    uint64_t seed = fd_get_seed(42);
    ASSERT_TRUE(seed == 42 + 2 * 1000);

    fd_cleanup();
}

/* ============================================================= */
/* Edge Cases                                                    */
/* ============================================================= */

static void test_null_safety(void) {
    fd_all_reduce(NULL);
    fd_all_reduce_mean(NULL);
    fd_broadcast(NULL, 0);
    fd_reduce(NULL, 0);
    ASSERT_NULL(fd_all_gather(NULL));

    fd_grad_accum_free(NULL);
    fd_grad_accum_step(NULL, NULL);
    fd_grad_accum_reset(NULL);

    ASSERT_TRUE(1);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running distributed tests...\n\n");

    /* Init tests */
    RUN_TEST(init_single_process);
    RUN_TEST(init_multi_process);
    RUN_TEST(get_config);

    /* Collective ops tests */
    RUN_TEST(all_reduce_single);
    RUN_TEST(all_reduce_mean);
    RUN_TEST(broadcast);
    RUN_TEST(all_gather);
    RUN_TEST(barrier);

    /* Gradient accumulation tests */
    RUN_TEST(grad_accum_create);
    RUN_TEST(grad_accum_step);
    RUN_TEST(grad_accum_reset);

    /* Data parallel tests */
    RUN_TEST(shard_indices_single);
    RUN_TEST(shard_indices_multi);
    RUN_TEST(shard_batch);

    /* Checkpointing tests */
    RUN_TEST(should_save_main);
    RUN_TEST(should_save_non_main);

    /* Utility tests */
    RUN_TEST(get_seed);

    /* Edge cases */
    RUN_TEST(null_safety);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
