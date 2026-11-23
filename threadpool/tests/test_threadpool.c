/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_threadpool.c — Unit tests for fw_ thread pool
 */

#include "lib/test.h"
#include "threadpool/threadpool.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* ---- Test: create and destroy pool ---- */
static void test_create_destroy(void) {
    fw_pool *pool = fw_pool_new(4);
    ASSERT_NOT_NULL(pool);
    fw_pool_free(pool);
}

/* ---- Test: single-thread pool works ---- */
static void sum_fn(int start, int end, void *arg) {
    atomic_int *sum = (atomic_int *)arg;
    int local = 0;
    for (int i = start; i < end; i++)
        local += i;
    atomic_fetch_add(sum, local);
}

static void test_single_thread(void) {
    fw_pool *pool = fw_pool_new(1);
    ASSERT_NOT_NULL(pool);

    atomic_int sum;
    atomic_init(&sum, 0);
    fw_for(pool, 100, sum_fn, &sum);
    /* sum(0..99) = 4950 */
    ASSERT_INT_EQ(atomic_load(&sum), 4950);

    fw_pool_free(pool);
}

/* ---- Test: parallel sum with 4 threads ---- */
static void test_parallel_sum(void) {
    fw_pool *pool = fw_pool_new(4);
    ASSERT_NOT_NULL(pool);

    atomic_int sum;
    atomic_init(&sum, 0);
    fw_for(pool, 1000, sum_fn, &sum);
    /* sum(0..999) = 499500 */
    ASSERT_INT_EQ(atomic_load(&sum), 499500);

    fw_pool_free(pool);
}

/* ---- Test: every index [0, N) is covered exactly once ---- */
typedef struct {
    atomic_int *flags;
    int count;
} range_ctx;

static void range_fn(int start, int end, void *arg) {
    range_ctx *ctx = (range_ctx *)arg;
    for (int i = start; i < end; i++)
        atomic_fetch_add(&ctx->flags[i], 1);
}

static void test_parallel_for_range(void) {
    int n = 1024;
    fw_pool *pool = fw_pool_new(4);
    ASSERT_NOT_NULL(pool);

    atomic_int *flags = (atomic_int *)calloc(n, sizeof(atomic_int));
    for (int i = 0; i < n; i++)
        atomic_init(&flags[i], 0);

    range_ctx ctx = { .flags = flags, .count = n };
    fw_for(pool, n, range_fn, &ctx);

    /* Every index must be visited exactly once */
    for (int i = 0; i < n; i++)
        ASSERT_INT_EQ(atomic_load(&flags[i]), 1);

    free(flags);
    fw_pool_free(pool);
}

/* ---- Test: zero count returns immediately ---- */
static void never_called(int start, int end, void *arg) {
    (void)start; (void)end; (void)arg;
    /* Should never be called */
    ASSERT_TRUE(0);
}

static void test_zero_count(void) {
    fw_pool *pool = fw_pool_new(4);
    ASSERT_NOT_NULL(pool);
    fw_for(pool, 0, never_called, NULL);
    fw_pool_free(pool);
}

/* ---- Test: NULL pool runs serially ---- */
static void test_null_pool_fallback(void) {
    atomic_int sum;
    atomic_init(&sum, 0);
    fw_for(NULL, 100, sum_fn, &sum);
    ASSERT_INT_EQ(atomic_load(&sum), 4950);
}

/* ---- Test: multiple fw_for calls reuse pool ---- */
static void test_multiple_dispatches(void) {
    fw_pool *pool = fw_pool_new(4);
    ASSERT_NOT_NULL(pool);

    for (int round = 0; round < 10; round++) {
        atomic_int sum;
        atomic_init(&sum, 0);
        fw_for(pool, 500, sum_fn, &sum);
        /* sum(0..499) = 124750 */
        ASSERT_INT_EQ(atomic_load(&sum), 124750);
    }

    fw_pool_free(pool);
}

int main(void) {
    printf("\n  threadpool tests\n\n");
    RUN_TEST(create_destroy);
    RUN_TEST(single_thread);
    RUN_TEST(parallel_sum);
    RUN_TEST(parallel_for_range);
    RUN_TEST(zero_count);
    RUN_TEST(null_pool_fallback);
    RUN_TEST(multiple_dispatches);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
