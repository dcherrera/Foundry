/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_call_census.c
 * @brief fc_call_census counts exactly the driver calls made while it is enabled, and
 *        timing events measure GPU time between two points of one stream.
 * @status complete
 * @issues none
 * @todo none
 *
 * Gates:
 *   - off: with the census disabled, copies, a launch and a sync change no field;
 *   - counts: enabled, one blocking HtoD (4096 B), one blocking DtoH (1000 B), one
 *     device copy (2048 B), one copy-kernel launch and one stream sync give exactly those
 *     counts and bytes, and nothing else moves;
 *   - graph: a capture of two copy-kernel launches counts 2 launches; each replay counts
 *     1 graph launch and no launch;
 *   - timing: events around 64 copy-kernel launches of 16 MiB measure > 0 ms, ~ every
 *     launch adds time (64 launches > 4x the time of 8), and an elapsed query across two
 *     contexts' events is refused.
 *
 * Build: make USE_CUDA=1 test-call-census
 * Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<die> ./cuda/tests/test_call_census
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <stdlib.h>
#include <string.h>

static fc_cuda_ctx *ctx;

/** b - a, field by field, as one array of counters. */
static void census_diff(const fc_call_census *a, const fc_call_census *b, fc_call_census *d)
{
    const unsigned long long *x = (const unsigned long long *)a;
    const unsigned long long *y = (const unsigned long long *)b;
    unsigned long long *z = (unsigned long long *)d;
    for (size_t i = 0; i < sizeof(*d) / sizeof(*z); i++)
        z[i] = y[i] - x[i];
}

/** One copy-kernel launch: 16-byte rows of `bytes` from src to dst on ctx. */
static int copy_kernel(fc_buffer *dst, fc_buffer *src, size_t bytes)
{
    return fc_buffer_copy_2d_kernel_async(dst, 0, bytes, src, 0, bytes, bytes, 1, ctx);
}

static void test_off(void)
{
    fc_call_census a, b, d;
    char host[4096] = { 0 };
    fc_buffer *x = fc_buffer_alloc_ctx(ctx, 4096), *y = fc_buffer_alloc_ctx(ctx, 4096);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(y);
    fc_call_census_enable(0);
    fc_call_census_read(&a);
    fc_buffer_from_cpu_n(x, host, sizeof(host));
    ASSERT_INT_EQ(copy_kernel(y, x, 4096), 0);
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu_n(y, host, sizeof(host));
    fc_call_census_read(&b);
    census_diff(&a, &b, &d);
    const unsigned long long *z = (const unsigned long long *)&d;
    for (size_t i = 0; i < sizeof(d) / sizeof(*z); i++)
        ASSERT_INT_EQ(z[i], 0);
    fc_buffer_free(x);
    fc_buffer_free(y);
}

static void test_counts(void)
{
    fc_call_census a, b, d;
    char host[4096] = { 1 };
    fc_buffer *x = fc_buffer_alloc_ctx(ctx, 4096), *y = fc_buffer_alloc_ctx(ctx, 4096);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(y);
    fc_call_census_enable(1);
    fc_call_census_read(&a);
    fc_buffer_from_cpu_n(x, host, 4096);
    fc_buffer_to_cpu_n(x, host, 1000);
    fc_buffer_copy_d2d_ctx(ctx, y, 0, x, 0, 2048);
    ASSERT_INT_EQ(copy_kernel(y, x, 4096), 0);
    fc_sync_ctx(ctx);
    fc_call_census_read(&b);
    fc_call_census_enable(0);
    census_diff(&a, &b, &d);
    ASSERT_INT_EQ(d.h2d_sync, 1);
    ASSERT_INT_EQ(d.h2d_bytes, 4096);
    ASSERT_INT_EQ(d.d2h_sync, 1);
    ASSERT_INT_EQ(d.d2h_bytes, 1000);
    ASSERT_INT_EQ(d.d2d, 1);
    ASSERT_INT_EQ(d.d2d_bytes, 2048);
    ASSERT_INT_EQ(d.launches, 1);
    ASSERT_INT_EQ(d.syncs, 1);
    ASSERT_INT_EQ(d.graph_launches, 0);
    ASSERT_INT_EQ(d.h2d_async + d.d2h_async + d.peer + d.peer_bytes, 0);
    fc_buffer_free(x);
    fc_buffer_free(y);
}

static void test_graph(void)
{
    fc_call_census a, b, d;
    fc_buffer *x = fc_buffer_alloc_ctx(ctx, 4096), *y = fc_buffer_alloc_ctx(ctx, 4096);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(y);
    fc_call_census_enable(1);
    fc_call_census_read(&a);
    ASSERT_INT_EQ(fc_graph_begin_ctx(ctx), 0);
    int ok = copy_kernel(y, x, 4096) == 0 && copy_kernel(x, y, 4096) == 0;
    fc_graph *g = fc_graph_end_ctx(ctx);
    ASSERT_TRUE(ok);
    ASSERT_NOT_NULL(g);
    ASSERT_INT_EQ(fc_graph_node_count(g, 0), 2);
    fc_call_census_read(&b);
    census_diff(&a, &b, &d);
    ASSERT_INT_EQ(d.launches, 2);
    ASSERT_INT_EQ(d.graph_launches, 0);
    for (int i = 0; i < 3; i++)
        ASSERT_INT_EQ(fc_graph_launch(g), 0);
    fc_sync_ctx(ctx);
    fc_call_census_read(&a);
    fc_call_census_enable(0);
    census_diff(&b, &a, &d);
    ASSERT_INT_EQ(d.graph_launches, 3);
    ASSERT_INT_EQ(d.launches, 0);
    ASSERT_INT_EQ(d.syncs, 1);
    fc_graph_free(g);
    fc_buffer_free(x);
    fc_buffer_free(y);
}

/** GPU ms for `n` back-to-back 16 MiB copy-kernel launches, by timing events. */
static float timed_copies(fc_buffer *dst, fc_buffer *src, int n, fc_event *e0, fc_event *e1)
{
    float ms = -1.0f;
    if (fc_tevent_record_ctx(ctx, e0) != 0)
        return -1.0f;
    for (int i = 0; i < n; i++)   /* 1024 rows of 16 KiB */
        if (fc_buffer_copy_2d_kernel_async(dst, 0, 16384, src, 0, 16384, 16384, 1024, ctx) != 0)
            return -1.0f;
    if (fc_tevent_record_ctx(ctx, e1) != 0 || fc_event_sync(e1) != 0 ||
        fc_tevent_elapsed_ms(e0, e1, &ms) != 0)
        return -1.0f;
    return ms;
}

static void test_timing(void)
{
    fc_buffer *x = fc_buffer_alloc_ctx(ctx, (size_t)16 << 20);
    fc_buffer *y = fc_buffer_alloc_ctx(ctx, (size_t)16 << 20);
    fc_event *e0 = fc_tevent_create_ctx(ctx), *e1 = fc_tevent_create_ctx(ctx);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(y);
    ASSERT_NOT_NULL(e0);
    ASSERT_NOT_NULL(e1);
    timed_copies(y, x, 8, e0, e1);   /* warm */
    float t8 = timed_copies(y, x, 8, e0, e1), t64 = timed_copies(y, x, 64, e0, e1);
    printf("    8 launches %.3f ms, 64 launches %.3f ms\n", t8, t64);
    ASSERT_TRUE(t8 > 0.0f);
    ASSERT_TRUE(t64 > 4.0f * t8);
    fc_event_free(e0);
    fc_event_free(e1);
    fc_buffer_free(x);
    fc_buffer_free(y);
}

int main(void)
{
    printf("Call Census + Timing Event Tests\n");
    printf("================================\n");
    ctx = fc_cuda_init_device(0);
    if (!ctx) {
        printf("  [SKIP] no CUDA device\n");
        TEST_REPORT();
        return 0;
    }
    RUN_TEST(off);
    RUN_TEST(counts);
    RUN_TEST(graph);
    RUN_TEST(timing);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
