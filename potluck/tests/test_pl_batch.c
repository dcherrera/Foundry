/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_pl_batch.c
 * @brief Tests for the token-budget batch planner and token-weighted gradient
 *   accumulation (card p1-batch, b1-b4).
 */
#include "../../lib/test.h"
#include "../pl_batch.h"

#include <stdlib.h>

#define TOY_DIM 3

/* ============================================================= */
/* b1 / b2: hand-worked plans                                     */
/* ============================================================= */

static const int plan_len[6] = {5, 3, 8, 2, 12, 4};
static const int plan_nv[6]  = {4, 0, 7, 1, 10, 3};

/* Plans plan_len over `order` at budget 10 and checks start/count/tokens of all 4 batches. */
static int check_hand_plan(const int *order, const int exp[4][3])
{
    pl_batch *b = NULL;
    int nb = 0;

    if (pl_batch_plan(plan_len, plan_nv, order, 6, 10, &b, &nb) != 0 || nb != 4) {
        free(b);
        return 0;
    }
    int ok = 1;
    for (int i = 0; i < 4; i++)
        ok &= b[i].start == exp[i][0] && b[i].count == exp[i][1] && b[i].tokens == exp[i][2];
    free(b);
    return ok;
}

static void test_b1_identity_order(void)
{
    const int order[6] = {0, 1, 2, 3, 4, 5};
    const int exp[4][3] = {{0, 2, 8}, {2, 2, 10}, {4, 1, 12}, {5, 1, 4}};
    ASSERT_TRUE(check_hand_plan(order, exp));
}

static void test_b1_reversed_order(void)
{
    const int order[6] = {5, 4, 3, 2, 1, 0};
    const int exp[4][3] = {{0, 1, 4}, {1, 1, 12}, {2, 2, 10}, {4, 2, 8}};
    ASSERT_TRUE(check_hand_plan(order, exp));
}

static void test_b2_n_valid(void)
{
    const int order[6] = {0, 1, 2, 3, 4, 5};
    const int exp_nv[4] = {4, 8, 10, 3};
    pl_batch *b = NULL;
    int nb = 0;

    ASSERT_INT_EQ(pl_batch_plan(plan_len, plan_nv, order, 6, 10, &b, &nb), 0);
    ASSERT_INT_EQ(nb, 4);
    int ok = 1;
    for (int i = 0; i < 4; i++)
        ok &= b[i].n_valid == exp_nv[i];
    free(b);
    ASSERT_TRUE(ok);
}

/* ============================================================= */
/* b3: accumulation identity on a linear least-squares toy model  */
/* ============================================================= */

typedef struct {
    double x[TOY_DIM];
    double y;
    int    valid;
} toy_token;

/* 3 examples (lengths 2, 5, 2; valid 2, 5, 1). Example 2's first token is
 * masked out and has a large residual, so a mask bug moves the gradient. */
static const toy_token toy_tokens[9] = {
    {{1.0, 0.0, 2.0}, 0.5, 1},  {{0.0, 1.0, -1.0}, 1.0, 1},
    {{1.0, 1.0, 1.0}, 0.0, 1},  {{2.0, 0.0, 0.5}, 1.5, 1},  {{0.0, -1.0, 1.0}, 0.25, 1},
    {{0.5, 0.5, 0.0}, -1.0, 1}, {{-1.0, 2.0, 1.0}, 0.0, 1},
    {{9.0, 9.0, 9.0}, -50.0, 0}, {{1.0, -2.0, 0.5}, 2.0, 1},
};
static const int toy_start[3] = {0, 2, 7};
static const int toy_len[3]   = {2, 5, 2};
static const int toy_nv[3]    = {2, 5, 1};
static const double toy_theta[TOY_DIM] = {0.5, -0.25, 1.0};

/* Adds Σ_valid (θ·x_t − y_t)·x_t over tokens [first, first+count) into sum. */
static void toy_residual_sum(int first, int count, double sum[TOY_DIM])
{
    for (int t = first; t < first + count; t++) {
        const toy_token *tok = &toy_tokens[t];
        if (!tok->valid)
            continue;
        double r = -tok->y;
        for (int j = 0; j < TOY_DIM; j++)
            r += toy_theta[j] * tok->x[j];
        for (int j = 0; j < TOY_DIM; j++)
            sum[j] += r * tok->x[j];
    }
}

static void test_b3_accumulation(void)
{
    const int order[3] = {0, 1, 2};
    pl_batch *b = NULL;
    int nb = 0;

    ASSERT_INT_EQ(pl_batch_plan(toy_len, toy_nv, order, 3, 9, &b, &nb), 0);
    int nv_batch = b[0].n_valid;
    free(b);
    ASSERT_INT_EQ(nb, 1);
    ASSERT_INT_EQ(nv_batch, 8);

    /* Per-example mean gradients g_i = (2/nv_i)·Σ_valid r·x, accumulated token-weighted. */
    float acc[TOY_DIM] = {0};
    for (int i = 0; i < 3; i++) {
        double sum[TOY_DIM] = {0};
        float g[TOY_DIM];
        toy_residual_sum(toy_start[i], toy_len[i], sum);
        for (int j = 0; j < TOY_DIM; j++)
            g[j] = (float)(2.0 / toy_nv[i] * sum[j]);
        pl_grad_axpy(acc, g, pl_batch_weight(toy_nv[i], nv_batch), TOY_DIM);
    }

    /* Batch loss gradient directly: (2/nv_batch)·Σ over all valid tokens of the batch. */
    double sum_all[TOY_DIM] = {0};
    toy_residual_sum(0, 9, sum_all);
    for (int j = 0; j < TOY_DIM; j++)
        ASSERT_FLT_NEAR(acc[j], 2.0 / nv_batch * sum_all[j], 1e-6f);
}

/* ============================================================= */
/* b4: bad args                                                   */
/* ============================================================= */

static void test_b4_bad_args(void)
{
    const int len[3] = {2, 3, 4};
    const int zero_len[3] = {2, 0, 4};
    const int nv[3] = {1, 1, 1};
    const int order[3] = {0, 1, 2};
    pl_batch *b = NULL;
    int nb = 0;

    ASSERT_INT_EQ(pl_batch_plan(len, nv, order, 3, 0, &b, &nb), -1);
    ASSERT_INT_EQ(pl_batch_plan(zero_len, nv, order, 3, 10, &b, &nb), -1);
    ASSERT_INT_EQ(pl_batch_plan(len, nv, order, 0, 10, &b, &nb), -1);
    ASSERT_INT_EQ(pl_batch_plan(len, NULL, order, 3, 10, &b, &nb), -1);
    ASSERT_TRUE(b == NULL);
    ASSERT_FLT_EQ(pl_batch_weight(3, 0), 0.0f);
}

int main(void)
{
    RUN_TEST(b1_identity_order);
    RUN_TEST(b1_reversed_order);
    RUN_TEST(b2_n_valid);
    RUN_TEST(b3_accumulation);
    RUN_TEST(b4_bad_args);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
