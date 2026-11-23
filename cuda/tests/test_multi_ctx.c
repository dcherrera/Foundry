/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_multi_ctx.c — Per-device context isolation tests for the CUDA backend
 *
 * Verifies the fc_cuda_ctx per-die refactor: two contexts bound to two
 * different device ids must own distinct context/stream/cuBLAS/module
 * handles and must not bleed into each other. Each context owns its own
 * buffers and a round-trip through one context must not disturb the other.
 *
 * Build: make USE_CUDA=1 test-cuda-multi  (run on a K80 host, >=2 dies)
 *
 * Degrades gracefully off-GPU: when the CUDA driver cannot be dlopen'd
 * (e.g. on the Mac dev box) or fewer than 2 dies are present, every test
 * prints [SKIP] and passes — so this compiles and runs everywhere, and
 * actually exercises isolation only on a multi-die host.
 *
 * @status Compiles on Mac (clang); device assertions pending K80 hardware.
 * @issues None
 * @todo None
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <stdlib.h>
#include <string.h>

#define FC_TEST_DEVICE_A 0
#define FC_TEST_DEVICE_B 1

/* Try to bring up two contexts on two distinct dies. On success, returns
 * 0 and fills *ctx_a / *ctx_b. On any failure (no driver, <2 dies) prints
 * a SKIP reason, leaves both NULL, and returns -1 so the caller bails. */
static int try_init_two_devices(fc_cuda_ctx **ctx_a, fc_cuda_ctx **ctx_b)
{
    *ctx_a = NULL;
    *ctx_b = NULL;

    if (fc_driver_load() != 0) {
        printf("    [SKIP] CUDA driver unavailable (off-GPU host)\n");
        return -1;
    }

    fc_cuda_ctx *a = fc_cuda_init_device(FC_TEST_DEVICE_A);
    if (!a) {
        printf("    [SKIP] device %d unavailable\n", FC_TEST_DEVICE_A);
        return -1;
    }

    fc_cuda_ctx *b = fc_cuda_init_device(FC_TEST_DEVICE_B);
    if (!b) {
        printf("    [SKIP] device %d unavailable (need >=2 dies)\n",
               FC_TEST_DEVICE_B);
        fc_cuda_ctx_free(a);
        return -1;
    }

    *ctx_a = a;
    *ctx_b = b;
    return 0;
}

/* ================================================================
 * Context identity / isolation
 * ================================================================ */

static void test_two_contexts_distinct(void)
{
    fc_cuda_ctx *ctx_a = NULL;
    fc_cuda_ctx *ctx_b = NULL;
    if (try_init_two_devices(&ctx_a, &ctx_b) != 0)
        return;

    ASSERT_TRUE(fc_cuda_ctx_available(ctx_a));
    ASSERT_TRUE(fc_cuda_ctx_available(ctx_b));

    /* Different handles entirely. */
    ASSERT_TRUE(ctx_a != ctx_b);

    /* Bound to the device ids we asked for. */
    ASSERT_TRUE(fc_cuda_ctx_device_id(ctx_a) == FC_TEST_DEVICE_A);
    ASSERT_TRUE(fc_cuda_ctx_device_id(ctx_b) == FC_TEST_DEVICE_B);
    ASSERT_TRUE(fc_cuda_ctx_device_id(ctx_a) != fc_cuda_ctx_device_id(ctx_b));

    fc_cuda_ctx_free(ctx_a);
    fc_cuda_ctx_free(ctx_b);
}

/* ================================================================
 * Buffers are owned by their context and don't cross over
 * ================================================================ */

static void test_buffers_per_context(void)
{
    fc_cuda_ctx *ctx_a = NULL;
    fc_cuda_ctx *ctx_b = NULL;
    if (try_init_two_devices(&ctx_a, &ctx_b) != 0)
        return;

    const int n = 256;
    float *src_a = malloc(n * sizeof(float));
    float *src_b = malloc(n * sizeof(float));
    float *dst_a = malloc(n * sizeof(float));
    float *dst_b = malloc(n * sizeof(float));
    ASSERT_NOT_NULL(src_a);
    ASSERT_NOT_NULL(src_b);
    ASSERT_NOT_NULL(dst_a);
    ASSERT_NOT_NULL(dst_b);

    for (int i = 0; i < n; i++) {
        src_a[i] = (float)i;            /* distinct payload per device */
        src_b[i] = (float)(1000 + i);
    }

    fc_buffer *buf_a = fc_buffer_from_ptr_ctx(ctx_a, src_a, n * sizeof(float));
    fc_buffer *buf_b = fc_buffer_from_ptr_ctx(ctx_b, src_b, n * sizeof(float));
    ASSERT_NOT_NULL(buf_a);
    ASSERT_NOT_NULL(buf_b);

    /* Distinct device allocations. */
    ASSERT_TRUE(buf_a != buf_b);

    /* Each round-trips its own data — no cross-context contamination. */
    memset(dst_a, 0, n * sizeof(float));
    memset(dst_b, 0, n * sizeof(float));
    fc_buffer_to_cpu(buf_a, dst_a);
    fc_buffer_to_cpu(buf_b, dst_b);
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(dst_a[i] == src_a[i]);
        ASSERT_TRUE(dst_b[i] == src_b[i]);
    }

    fc_buffer_free(buf_a);
    fc_buffer_free(buf_b);
    free(src_a); free(src_b); free(dst_a); free(dst_b);

    fc_cuda_ctx_free(ctx_a);
    fc_cuda_ctx_free(ctx_b);
}

/* ================================================================
 * A kernel on one context leaves the other context's data untouched
 * ================================================================ */

static void test_kernel_isolation(void)
{
    fc_cuda_ctx *ctx_a = NULL;
    fc_cuda_ctx *ctx_b = NULL;
    if (try_init_two_devices(&ctx_a, &ctx_b) != 0)
        return;

    const int n = 512;
    float *ones = malloc(n * sizeof(float));
    float *check_b = malloc(n * sizeof(float));
    float *out_a = malloc(n * sizeof(float));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(check_b);
    ASSERT_NOT_NULL(out_a);
    for (int i = 0; i < n; i++)
        ones[i] = 1.0f;

    /* Reference data parked on device B. */
    fc_buffer *buf_b = fc_buffer_from_ptr_ctx(ctx_b, ones, n * sizeof(float));
    ASSERT_NOT_NULL(buf_b);

    /* Run add on device A (1 + 1 = 2 everywhere). */
    fc_buffer *a0 = fc_buffer_from_ptr_ctx(ctx_a, ones, n * sizeof(float));
    fc_buffer *a1 = fc_buffer_from_ptr_ctx(ctx_a, ones, n * sizeof(float));
    fc_buffer *ao = fc_buffer_alloc_ctx(ctx_a, n * sizeof(float));
    ASSERT_NOT_NULL(a0);
    ASSERT_NOT_NULL(a1);
    ASSERT_NOT_NULL(ao);

    fc_add_ctx(ctx_a, a0, a1, ao, n);
    fc_sync_ctx(ctx_a);
    fc_buffer_to_cpu(ao, out_a);

    /* Device B's buffer must still read back as ones — A's work didn't bleed. */
    fc_buffer_to_cpu(buf_b, check_b);
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(out_a[i] == 2.0f);
        ASSERT_TRUE(check_b[i] == 1.0f);
    }

    fc_buffer_free(a0);
    fc_buffer_free(a1);
    fc_buffer_free(ao);
    fc_buffer_free(buf_b);
    free(ones); free(check_b); free(out_a);

    fc_cuda_ctx_free(ctx_a);
    fc_cuda_ctx_free(ctx_b);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void)
{
    printf("CUDA Multi-Context Isolation Tests\n");
    printf("==================================\n");

    RUN_TEST(two_contexts_distinct);
    RUN_TEST(buffers_per_context);
    RUN_TEST(kernel_isolation);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
