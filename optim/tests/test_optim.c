/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../optim.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================= */
/* SGD Tests                                                     */
/* ============================================================= */

static void test_sgd_default_config(void) {
    fo_sgd_config cfg = fo_sgd_default(0.01f);

    ASSERT_FLT_NEAR(0.01f, cfg.lr, 0.0001f);
    ASSERT_FLT_NEAR(0.0f, cfg.momentum, 0.0001f);
    ASSERT_FLT_NEAR(0.0f, cfg.weight_decay, 0.0001f);
    ASSERT_FALSE(cfg.nesterov);
}

static void test_sgd_create(void) {
    int64_t shape[] = {4};
    tm_tensor* params[2];
    params[0] = tm_zeros(shape, 1, TM_F32);
    params[1] = tm_zeros(shape, 1, TM_F32);
    tm_set_requires_grad(params[0], true);
    tm_set_requires_grad(params[1], true);

    fo_sgd_config cfg = fo_sgd_default(0.1f);
    fo_optimizer* opt = fo_sgd_new(params, 2, &cfg);

    ASSERT_NOT_NULL(opt);
    ASSERT_FLT_NEAR(0.1f, fo_get_lr(opt), 0.0001f);
    ASSERT_INT_EQ(0, fo_get_step(opt));

    fo_free(opt);
    tm_release(params[0]);
    tm_release(params[1]);
}

static void test_sgd_step(void) {
    int64_t shape[] = {2};
    tm_tensor* params[1];
    params[0] = tm_full(shape, 1, TM_F32, 1.0);  /* Initial value = 1 */
    tm_set_requires_grad(params[0], true);
    params[0]->grad = tm_full(shape, 1, TM_F32, 0.5);  /* Gradient = 0.5 */

    fo_sgd_config cfg = fo_sgd_default(0.1f);
    fo_optimizer* opt = fo_sgd_new(params, 1, &cfg);

    fo_step(opt);

    /* After step: param = 1.0 - 0.1 * 0.5 = 0.95 */
    float* data = tm_data_f32(params[0]);
    ASSERT_FLT_NEAR(0.95f, data[0], 0.0001f);
    ASSERT_INT_EQ(1, fo_get_step(opt));

    fo_free(opt);
    tm_release(params[0]);
}

static void test_sgd_momentum(void) {
    int64_t shape[] = {1};
    tm_tensor* params[1];
    params[0] = tm_full(shape, 1, TM_F32, 1.0);
    tm_set_requires_grad(params[0], true);
    params[0]->grad = tm_full(shape, 1, TM_F32, 1.0);

    fo_sgd_config cfg = fo_sgd_default(0.1f);
    cfg.momentum = 0.9f;
    fo_optimizer* opt = fo_sgd_new(params, 1, &cfg);

    /* First step */
    fo_step(opt);
    float* data = tm_data_f32(params[0]);
    ASSERT_FLT_NEAR(0.9f, data[0], 0.0001f);  /* 1.0 - 0.1 * 1.0 = 0.9 */

    /* Second step with same gradient */
    tm_data_f32(params[0]->grad)[0] = 1.0f;
    fo_step(opt);
    /* momentum_buf = 0.9 * 1.0 + 1.0 = 1.9 */
    /* param = 0.9 - 0.1 * 1.9 = 0.71 */
    ASSERT_FLT_NEAR(0.71f, data[0], 0.0001f);

    fo_free(opt);
    tm_release(params[0]);
}

/* ============================================================= */
/* Adam Tests                                                    */
/* ============================================================= */

static void test_adam_default_config(void) {
    fo_adam_config cfg = fo_adam_default(0.001f);

    ASSERT_FLT_NEAR(0.001f, cfg.lr, 0.0001f);
    ASSERT_FLT_NEAR(0.9f, cfg.beta1, 0.0001f);
    ASSERT_FLT_NEAR(0.999f, cfg.beta2, 0.0001f);
    ASSERT_FLT_NEAR(1e-8f, cfg.eps, 1e-10f);
    ASSERT_FALSE(cfg.amsgrad);
}

static void test_adam_create(void) {
    int64_t shape[] = {4};
    tm_tensor* params[1];
    params[0] = tm_zeros(shape, 1, TM_F32);
    tm_set_requires_grad(params[0], true);

    fo_adam_config cfg = fo_adam_default(0.001f);
    fo_optimizer* opt = fo_adam_new(params, 1, &cfg);

    ASSERT_NOT_NULL(opt);
    ASSERT_FLT_NEAR(0.001f, fo_get_lr(opt), 0.0001f);

    fo_free(opt);
    tm_release(params[0]);
}

static void test_adam_step(void) {
    int64_t shape[] = {1};
    tm_tensor* params[1];
    params[0] = tm_full(shape, 1, TM_F32, 0.0);
    tm_set_requires_grad(params[0], true);
    params[0]->grad = tm_full(shape, 1, TM_F32, 1.0);

    fo_adam_config cfg = fo_adam_default(0.1f);
    fo_optimizer* opt = fo_adam_new(params, 1, &cfg);

    fo_step(opt);

    /* After one step, parameter should have changed */
    float* data = tm_data_f32(params[0]);
    ASSERT_TRUE(data[0] != 0.0f);
    ASSERT_TRUE(data[0] < 0.0f);  /* Should be negative since grad was positive */

    fo_free(opt);
    tm_release(params[0]);
}

static void test_adamw_create(void) {
    int64_t shape[] = {4};
    tm_tensor* params[1];
    params[0] = tm_zeros(shape, 1, TM_F32);
    tm_set_requires_grad(params[0], true);

    fo_adam_config cfg = fo_adam_default(0.001f);
    cfg.weight_decay = 0.01f;
    fo_optimizer* opt = fo_adamw_new(params, 1, &cfg);

    ASSERT_NOT_NULL(opt);

    fo_free(opt);
    tm_release(params[0]);
}

/* ============================================================= */
/* Zero Grad Tests                                               */
/* ============================================================= */

static void test_zero_grad(void) {
    int64_t shape[] = {4};
    tm_tensor* params[1];
    params[0] = tm_zeros(shape, 1, TM_F32);
    tm_set_requires_grad(params[0], true);
    params[0]->grad = tm_full(shape, 1, TM_F32, 5.0);

    fo_sgd_config cfg = fo_sgd_default(0.1f);
    fo_optimizer* opt = fo_sgd_new(params, 1, &cfg);

    /* Verify grad is not zero */
    float* grad = tm_data_f32(params[0]->grad);
    ASSERT_FLT_NEAR(5.0f, grad[0], 0.0001f);

    fo_zero_grad(opt);

    /* Verify grad is now zero */
    ASSERT_FLT_NEAR(0.0f, grad[0], 0.0001f);

    fo_free(opt);
    tm_release(params[0]);
}

/* ============================================================= */
/* Learning Rate Tests                                           */
/* ============================================================= */

static void test_set_lr(void) {
    int64_t shape[] = {4};
    tm_tensor* params[1];
    params[0] = tm_zeros(shape, 1, TM_F32);
    tm_set_requires_grad(params[0], true);

    fo_sgd_config cfg = fo_sgd_default(0.1f);
    fo_optimizer* opt = fo_sgd_new(params, 1, &cfg);

    ASSERT_FLT_NEAR(0.1f, fo_get_lr(opt), 0.0001f);

    fo_set_lr(opt, 0.01f);
    ASSERT_FLT_NEAR(0.01f, fo_get_lr(opt), 0.0001f);

    fo_free(opt);
    tm_release(params[0]);
}

/* ============================================================= */
/* Scheduler Tests                                               */
/* ============================================================= */

static void test_scheduler_cosine(void) {
    int64_t shape[] = {4};
    tm_tensor* params[1];
    params[0] = tm_zeros(shape, 1, TM_F32);
    tm_set_requires_grad(params[0], true);

    fo_sgd_config cfg = fo_sgd_default(1.0f);
    fo_optimizer* opt = fo_sgd_new(params, 1, &cfg);

    fo_scheduler_config sched = fo_scheduler_default();
    sched.type = FO_SCHED_COSINE;
    sched.total_steps = 100;
    sched.min_lr = 0.0f;
    fo_set_scheduler(opt, &sched);

    /* At step 0, should be base_lr */
    float lr0 = fo_get_scheduled_lr(opt, 0);
    ASSERT_FLT_NEAR(1.0f, lr0, 0.01f);

    /* At step 50 (halfway), should be ~0.5 */
    float lr50 = fo_get_scheduled_lr(opt, 50);
    ASSERT_FLT_NEAR(0.5f, lr50, 0.1f);

    /* At step 100, should be min_lr */
    float lr100 = fo_get_scheduled_lr(opt, 100);
    ASSERT_FLT_NEAR(0.0f, lr100, 0.01f);

    fo_free(opt);
    tm_release(params[0]);
}

static void test_scheduler_warmup_cosine(void) {
    int64_t shape[] = {4};
    tm_tensor* params[1];
    params[0] = tm_zeros(shape, 1, TM_F32);
    tm_set_requires_grad(params[0], true);

    fo_sgd_config cfg = fo_sgd_default(1.0f);
    fo_optimizer* opt = fo_sgd_new(params, 1, &cfg);

    fo_scheduler_config sched = fo_scheduler_default();
    sched.type = FO_SCHED_WARMUP_COSINE;
    sched.warmup_steps = 10;
    sched.total_steps = 100;
    sched.min_lr = 0.0f;
    fo_set_scheduler(opt, &sched);

    /* During warmup, should increase linearly */
    float lr5 = fo_get_scheduled_lr(opt, 4);  /* Step 5 out of 10 warmup */
    ASSERT_TRUE(lr5 > 0.0f && lr5 < 1.0f);

    /* At end of warmup, should be at base_lr */
    float lr10 = fo_get_scheduled_lr(opt, 10);
    ASSERT_FLT_NEAR(1.0f, lr10, 0.1f);

    fo_free(opt);
    tm_release(params[0]);
}

/* ============================================================= */
/* Gradient Clipping Tests                                       */
/* ============================================================= */

static void test_clip_grad_norm(void) {
    int64_t shape[] = {4};
    tm_tensor* params[1];
    params[0] = tm_zeros(shape, 1, TM_F32);
    tm_set_requires_grad(params[0], true);
    params[0]->grad = tm_full(shape, 1, TM_F32, 2.0);  /* Norm = 4 */

    fo_sgd_config cfg = fo_sgd_default(0.1f);
    fo_optimizer* opt = fo_sgd_new(params, 1, &cfg);

    float norm = fo_clip_grad_norm(opt, 1.0f);

    /* Original norm should be 4 (sqrt(4 * 4)) = 4 */
    ASSERT_FLT_NEAR(4.0f, norm, 0.01f);

    /* After clipping to max_norm=1, gradients should be scaled down */
    float* grad = tm_data_f32(params[0]->grad);
    float new_norm = 0.0f;
    for (int i = 0; i < 4; i++) {
        new_norm += grad[i] * grad[i];
    }
    new_norm = sqrtf(new_norm);
    ASSERT_FLT_NEAR(1.0f, new_norm, 0.01f);

    fo_free(opt);
    tm_release(params[0]);
}

static void test_clip_grad_value(void) {
    int64_t shape[] = {4};
    tm_tensor* params[1];
    params[0] = tm_zeros(shape, 1, TM_F32);
    tm_set_requires_grad(params[0], true);
    params[0]->grad = tm_zeros(shape, 1, TM_F32);
    float* grad = tm_data_f32(params[0]->grad);
    grad[0] = 5.0f;
    grad[1] = -5.0f;
    grad[2] = 0.5f;
    grad[3] = -0.5f;

    fo_sgd_config cfg = fo_sgd_default(0.1f);
    fo_optimizer* opt = fo_sgd_new(params, 1, &cfg);

    fo_clip_grad_value(opt, 1.0f);

    ASSERT_FLT_NEAR(1.0f, grad[0], 0.0001f);
    ASSERT_FLT_NEAR(-1.0f, grad[1], 0.0001f);
    ASSERT_FLT_NEAR(0.5f, grad[2], 0.0001f);
    ASSERT_FLT_NEAR(-0.5f, grad[3], 0.0001f);

    fo_free(opt);
    tm_release(params[0]);
}

/* ============================================================= */
/* Null Safety Tests                                             */
/* ============================================================= */

static void test_null_safety(void) {
    fo_step(NULL);
    fo_zero_grad(NULL);
    fo_free(NULL);
    fo_set_lr(NULL, 0.1f);
    fo_scheduler_step(NULL);
    fo_clip_grad_norm(NULL, 1.0f);
    fo_clip_grad_value(NULL, 1.0f);

    ASSERT_FLT_NEAR(0.0f, fo_get_lr(NULL), 0.0001f);
    ASSERT_INT_EQ(0, fo_get_step(NULL));

    ASSERT_TRUE(1);  /* Didn't crash */
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running optim tests...\n\n");

    /* SGD tests */
    RUN_TEST(sgd_default_config);
    RUN_TEST(sgd_create);
    RUN_TEST(sgd_step);
    RUN_TEST(sgd_momentum);

    /* Adam tests */
    RUN_TEST(adam_default_config);
    RUN_TEST(adam_create);
    RUN_TEST(adam_step);
    RUN_TEST(adamw_create);

    /* Common ops tests */
    RUN_TEST(zero_grad);
    RUN_TEST(set_lr);

    /* Scheduler tests */
    RUN_TEST(scheduler_cosine);
    RUN_TEST(scheduler_warmup_cosine);

    /* Gradient clipping tests */
    RUN_TEST(clip_grad_norm);
    RUN_TEST(clip_grad_value);

    /* Null safety */
    RUN_TEST(null_safety);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
