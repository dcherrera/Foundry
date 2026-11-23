/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../progress.h"

#include <unistd.h>

/* ============================================================= */
/* fp_new tests                                                  */
/* ============================================================= */

static void test_creates_progress_bar(void) {
    fp_progress* p = fp_new(100, "Test");
    ASSERT_NOT_NULL(p);
    fp_free(p);
}

static void test_null_description_ok(void) {
    fp_progress* p = fp_new(100, NULL);
    ASSERT_NOT_NULL(p);
    fp_free(p);
}

static void test_zero_total_ok(void) {
    fp_progress* p = fp_new(0, "Empty");
    ASSERT_NOT_NULL(p);
    fp_free(p);
}

/* ============================================================= */
/* fp_total / fp_current tests                                   */
/* ============================================================= */

static void test_total_returns_initial_value(void) {
    fp_progress* p = fp_new(500, "Test");
    ASSERT_INT_EQ(500, fp_total(p));
    fp_free(p);
}

static void test_current_starts_at_zero(void) {
    fp_progress* p = fp_new(100, "Test");
    ASSERT_INT_EQ(0, fp_current(p));
    fp_free(p);
}

static void test_percent_starts_at_zero(void) {
    fp_progress* p = fp_new(100, "Test");
    ASSERT_FLT_NEAR(0.0f, fp_percent(p), 0.01f);
    fp_free(p);
}

/* ============================================================= */
/* fp_update tests                                               */
/* ============================================================= */

static void test_update_increments_current(void) {
    fp_progress* p = fp_new(100, "Test");
    fp_disable(p);

    fp_update(p, 10);
    ASSERT_INT_EQ(10, fp_current(p));

    fp_update(p, 5);
    ASSERT_INT_EQ(15, fp_current(p));

    fp_free(p);
}

static void test_update_clamps_to_total(void) {
    fp_progress* p = fp_new(100, "Test");
    fp_disable(p);

    fp_update(p, 150);
    ASSERT_INT_EQ(100, fp_current(p));

    fp_free(p);
}

static void test_percent_updates_correctly(void) {
    fp_progress* p = fp_new(100, "Test");
    fp_disable(p);

    fp_update(p, 50);
    ASSERT_FLT_NEAR(50.0f, fp_percent(p), 0.01f);

    fp_update(p, 50);
    ASSERT_FLT_NEAR(100.0f, fp_percent(p), 0.01f);

    fp_free(p);
}

/* ============================================================= */
/* fp_set tests                                                  */
/* ============================================================= */

static void test_set_absolute_position(void) {
    fp_progress* p = fp_new(100, "Test");
    fp_disable(p);

    fp_set(p, 75);
    ASSERT_INT_EQ(75, fp_current(p));

    fp_free(p);
}

/* ============================================================= */
/* fp_rate / fp_eta tests                                        */
/* ============================================================= */

static void test_elapsed_increases(void) {
    fp_progress* p = fp_new(100, "Test");
    fp_disable(p);

    double t1 = fp_elapsed(p);
    usleep(10000); /* 10ms */
    double t2 = fp_elapsed(p);

    ASSERT_DBL_GT(t2, t1);
    fp_free(p);
}

static void test_rate_is_positive_after_update(void) {
    fp_progress* p = fp_new(100, "Test");
    fp_disable(p);

    fp_update(p, 10);
    usleep(10000); /* 10ms */

    double rate = fp_rate(p);
    ASSERT_DBL_GT(rate, 0.0);

    fp_free(p);
}

static void test_eta_decreases_as_progress_increases(void) {
    fp_progress* p = fp_new(100, "Test");
    fp_disable(p);

    fp_update(p, 10);
    usleep(10000);
    double eta1 = fp_eta(p);

    fp_update(p, 40);
    usleep(10000);
    double eta2 = fp_eta(p);

    ASSERT_DBL_LT(eta2, eta1);

    fp_free(p);
}

/* ============================================================= */
/* fp_set_postfix tests                                          */
/* ============================================================= */

static void test_postfix_does_not_crash(void) {
    fp_progress* p = fp_new(100, "Test");
    fp_disable(p);

    fp_set_postfix(p, "loss=%.4f, lr=%.2e", 0.1234, 1e-5);
    fp_update(p, 10);

    ASSERT_TRUE(1);

    fp_free(p);
}

/* ============================================================= */
/* fp_train_state tests                                          */
/* ============================================================= */

static void test_log_step_does_not_crash(void) {
    fp_train_state state = {
        .epoch = 1,
        .step = 100,
        .loss = 0.5f,
        .lr = 1e-4f,
        .throughput = 1000.0f
    };

    fp_log_step(&state);
    ASSERT_TRUE(1);
}

static void test_log_epoch_does_not_crash(void) {
    fp_train_state state = {
        .epoch = 5,
        .step = 5000,
        .loss = 0.1f,
        .lr = 1e-5f,
        .throughput = 1200.0f
    };

    fp_log_epoch(&state);
    ASSERT_TRUE(1);
}

/* ============================================================= */
/* Edge cases                                                    */
/* ============================================================= */

static void test_null_progress_is_safe(void) {
    fp_update(NULL, 10);
    fp_set(NULL, 10);
    fp_close(NULL);
    fp_free(NULL);

    ASSERT_INT_EQ(0, fp_total(NULL));
    ASSERT_INT_EQ(0, fp_current(NULL));
    ASSERT_DBL_EQ(0.0, fp_elapsed(NULL));
    ASSERT_DBL_EQ(0.0, fp_rate(NULL));
    ASSERT_DBL_EQ(0.0, fp_eta(NULL));
    ASSERT_FLT_EQ(0.0f, fp_percent(NULL));
}

static void test_close_is_idempotent(void) {
    fp_progress* p = fp_new(100, "Test");
    fp_disable(p);

    fp_close(p);
    fp_close(p);
    fp_close(p);

    ASSERT_TRUE(1);
    fp_free(p);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running progress tests...\n\n");

    /* fp_new tests */
    RUN_TEST(creates_progress_bar);
    RUN_TEST(null_description_ok);
    RUN_TEST(zero_total_ok);

    /* Accessor tests */
    RUN_TEST(total_returns_initial_value);
    RUN_TEST(current_starts_at_zero);
    RUN_TEST(percent_starts_at_zero);

    /* fp_update tests */
    RUN_TEST(update_increments_current);
    RUN_TEST(update_clamps_to_total);
    RUN_TEST(percent_updates_correctly);

    /* fp_set tests */
    RUN_TEST(set_absolute_position);

    /* Timing tests */
    RUN_TEST(elapsed_increases);
    RUN_TEST(rate_is_positive_after_update);
    RUN_TEST(eta_decreases_as_progress_increases);

    /* Postfix tests */
    RUN_TEST(postfix_does_not_crash);

    /* Training log tests */
    RUN_TEST(log_step_does_not_crash);
    RUN_TEST(log_epoch_does_not_crash);

    /* Edge cases */
    RUN_TEST(null_progress_is_safe);
    RUN_TEST(close_is_idempotent);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
