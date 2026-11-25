/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_host_pinned_wait.c
 * @brief Pinned host memory (fc_host_alloc) moves bytes exactly, and fc_sync_ctx /
 *        fc_event_sync wait for the device under every FC_CUDA_WAIT mode.
 * @status complete
 * @issues none
 * @todo none
 *
 * Gates:
 *   - pinned: fc_host_alloc gives memory; 1 MiB H2D then D2H through it (sync and async)
 *     round-trips byte-exact; fc_host_free(NULL) is a no-op;
 *   - reuse: an async H2D from pinned memory, an event recorded after it and synced, then
 *     the host rewrites the memory: the device holds the first contents (the event is the
 *     point after which the source may be reused);
 *   - wait: 64 copy-kernel launches of 16 MiB are still running right after issue (proves
 *     the probe can fail), and complete as soon as fc_sync_ctx / fc_event_sync return;
 *     the result bytes are the last launch's.
 *
 * Run once per wait mode (the mode is read once per process):
 *   make USE_CUDA=1 test-host-pinned-wait
 *   (runs FC_CUDA_WAIT = driver, spin, yield, hybrid:50, sleep:20)
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PINNED_TEST_BYTES ((size_t)1 << 20)
#define WAIT_TEST_BYTES ((size_t)16 << 20)

static fc_cuda_ctx *ctx;

/** Fills `bytes` of p with a pattern keyed by `seed`. */
static void fill_pattern(uint8_t *p, size_t bytes, unsigned seed)
{
    for (size_t i = 0; i < bytes; i++)
        p[i] = (uint8_t)((i * 131u + seed * 17u + (i >> 9)) & 0xffu);
}

/** Count of bytes of p that differ from the pattern keyed by `seed`. */
static size_t pattern_mismatches(const uint8_t *p, size_t bytes, unsigned seed)
{
    size_t bad = 0;
    for (size_t i = 0; i < bytes; i++)
        bad += p[i] != (uint8_t)((i * 131u + seed * 17u + (i >> 9)) & 0xffu);
    return bad;
}

static void test_pinned(void)
{
    uint8_t *pin = fc_host_alloc(ctx, PINNED_TEST_BYTES);
    uint8_t *back = malloc(PINNED_TEST_BYTES);
    fc_buffer *d = fc_buffer_alloc_ctx(ctx, PINNED_TEST_BYTES);
    ASSERT_NOT_NULL(pin);
    ASSERT_NOT_NULL(back);
    ASSERT_NOT_NULL(d);
    fill_pattern(pin, PINNED_TEST_BYTES, 1);
    fc_buffer_from_cpu_n(d, pin, PINNED_TEST_BYTES);
    memset(back, 0, PINNED_TEST_BYTES);
    fc_buffer_to_cpu_n(d, back, PINNED_TEST_BYTES);
    ASSERT_INT_EQ(pattern_mismatches(back, PINNED_TEST_BYTES, 1), 0);

    fill_pattern(back, PINNED_TEST_BYTES, 2);
    fc_buffer_from_cpu_n(d, back, PINNED_TEST_BYTES);
    memset(pin, 0, PINNED_TEST_BYTES);
    ASSERT_INT_EQ(fc_buffer_to_cpu_n_async(d, pin, PINNED_TEST_BYTES), 0);
    fc_sync_ctx(ctx);
    ASSERT_INT_EQ(pattern_mismatches(pin, PINNED_TEST_BYTES, 2), 0);

    fc_host_free(ctx, NULL);
    fc_host_free(ctx, pin);
    fc_buffer_free(d);
    free(back);
}

static void test_reuse(void)
{
    uint8_t *pin = fc_host_alloc(ctx, PINNED_TEST_BYTES);
    uint8_t *back = malloc(PINNED_TEST_BYTES);
    fc_buffer *d = fc_buffer_alloc_ctx(ctx, PINNED_TEST_BYTES);
    fc_event *done = fc_event_create_ctx(ctx);
    ASSERT_NOT_NULL(pin);
    ASSERT_NOT_NULL(back);
    ASSERT_NOT_NULL(d);
    ASSERT_NOT_NULL(done);
    fill_pattern(pin, PINNED_TEST_BYTES, 3);
    ASSERT_INT_EQ(fc_buffer_from_cpu_n_async(d, pin, PINNED_TEST_BYTES), 0);
    ASSERT_INT_EQ(fc_event_record_ctx(ctx, done), 0);
    ASSERT_INT_EQ(fc_event_sync(done), 0);
    fill_pattern(pin, PINNED_TEST_BYTES, 4);
    fc_buffer_to_cpu_n(d, back, PINNED_TEST_BYTES);
    ASSERT_INT_EQ(pattern_mismatches(back, PINNED_TEST_BYTES, 3), 0);
    fc_event_free(done);
    fc_host_free(ctx, pin);
    fc_buffer_free(d);
    free(back);
}

/** Issues 64 copy-kernel launches of 16 MiB (x→y, last one y←z) and records `ev`. */
static int issue_long_work(fc_buffer *x, fc_buffer *y, fc_buffer *z, fc_event *ev)
{
    for (int i = 0; i < 63; i++)
        if (fc_buffer_copy_2d_kernel_async(y, 0, 16384, x, 0, 16384, 16384, 1024, ctx) != 0)
            return -1;
    if (fc_buffer_copy_2d_kernel_async(y, 0, 16384, z, 0, 16384, 16384, 1024, ctx) != 0)
        return -1;
    return fc_event_record_ctx(ctx, ev);
}

static void test_wait(void)
{
    const char *mode = getenv("FC_CUDA_WAIT");
    uint8_t *host = malloc(WAIT_TEST_BYTES);
    fc_buffer *x = fc_buffer_alloc_ctx(ctx, WAIT_TEST_BYTES);
    fc_buffer *y = fc_buffer_alloc_ctx(ctx, WAIT_TEST_BYTES);
    fc_buffer *z = fc_buffer_alloc_ctx(ctx, WAIT_TEST_BYTES);
    fc_event *ev = fc_event_create_ctx(ctx);
    ASSERT_NOT_NULL(host);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(y);
    ASSERT_NOT_NULL(z);
    ASSERT_NOT_NULL(ev);
    printf("    FC_CUDA_WAIT=%s\n", mode ? mode : "(unset)");
    fill_pattern(host, WAIT_TEST_BYTES, 5);
    fc_buffer_from_cpu_n(x, host, WAIT_TEST_BYTES);
    fill_pattern(host, WAIT_TEST_BYTES, 6);
    fc_buffer_from_cpu_n(z, host, WAIT_TEST_BYTES);

    for (int round = 0; round < 3; round++) {
        ASSERT_INT_EQ(issue_long_work(x, y, z, ev), 0);
        ASSERT_INT_EQ(fc_event_query(ev), 1);    /* still running: the probe can fail */
        fc_sync_ctx(ctx);
        ASSERT_INT_EQ(fc_event_query(ev), 0);
        ASSERT_INT_EQ(issue_long_work(x, y, z, ev), 0);
        ASSERT_INT_EQ(fc_event_query(ev), 1);
        ASSERT_INT_EQ(fc_event_sync(ev), 0);
        ASSERT_INT_EQ(fc_event_query(ev), 0);
    }
    memset(host, 0, WAIT_TEST_BYTES);
    ASSERT_INT_EQ(fc_buffer_to_cpu_n_async(y, host, WAIT_TEST_BYTES), 0);
    fc_sync_ctx(ctx);
    ASSERT_INT_EQ(pattern_mismatches(host, WAIT_TEST_BYTES, 6), 0);
    fc_event_free(ev);
    fc_buffer_free(x);
    fc_buffer_free(y);
    fc_buffer_free(z);
    free(host);
}

int main(void)
{
    printf("Pinned Host Memory + Wait Mode Tests\n");
    printf("====================================\n");
    ctx = fc_cuda_init_device(0);
    if (!ctx) {
        printf("  [SKIP] no CUDA device\n");
        TEST_REPORT();
        return 0;
    }
    RUN_TEST(pinned);
    RUN_TEST(reuse);
    RUN_TEST(wait);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
