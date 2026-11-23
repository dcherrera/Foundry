/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_barswap.c — Unit tests for BAR Swap userspace library v2
 *
 * Tests gracefully skip (pass) if /dev/bar_swap is not available,
 * since we can't test the kernel module on the development machine.
 *
 * Phase 1 tests (6): init/free, null safety, bounds, num_dies, num_slots, mapped
 * Phase 2 tests (8): acquire/release, active_die tracking, implicit release, VRAM
 */

#include "../../lib/test.h"
#include "../barswap.h"

#include <stdio.h>

/* Track whether device is available for conditional tests */
static bs_context *g_ctx = NULL;
static int g_device_available = 0;

static void setup(void)
{
    g_ctx = bs_init();
    g_device_available = (g_ctx != NULL);
    if (!g_device_available) {
        printf("    (note: /dev/bar_swap not available — testing offline mode)\n");
    }
}

static void teardown(void)
{
    if (g_ctx) {
        bs_free(g_ctx);
        g_ctx = NULL;
    }
}

/* ---- Phase 1 tests ---- */

/* Test: init/free lifecycle — should not crash even without device */
static void test_init_free(void)
{
    bs_context *ctx = bs_init();
    /* May be NULL if no device — that's fine */
    bs_free(ctx);
    /* Also safe to free NULL */
    bs_free(NULL);
    ASSERT_TRUE(1);
}

/* Test: NULL context safety */
static void test_null_safety(void)
{
    ASSERT_INT_EQ(bs_num_dies(NULL), 0);
    ASSERT_INT_EQ(bs_num_slots(NULL), 0);
    ASSERT_INT_EQ(bs_is_mapped(NULL, 0), 0);
    ASSERT_INT_EQ(bs_is_mapped(NULL, -1), 0);
    ASSERT_INT_EQ(bs_active_die(NULL), -1);
}

/* Test: out-of-range die_id returns 0 */
static void test_is_mapped_bounds(void)
{
    if (!g_device_available) {
        ASSERT_INT_EQ(bs_is_mapped(g_ctx, 0), 0);
        ASSERT_INT_EQ(bs_is_mapped(g_ctx, 100), 0);
        ASSERT_INT_EQ(bs_is_mapped(g_ctx, -1), 0);
        return;
    }
    ASSERT_INT_EQ(bs_is_mapped(g_ctx, -1), 0);
    ASSERT_INT_EQ(bs_is_mapped(g_ctx, BS_MAX_DIES), 0);
    ASSERT_INT_EQ(bs_is_mapped(g_ctx, 9999), 0);
}

/* Test: num_dies returns a reasonable value when device is present */
static void test_num_dies(void)
{
    if (!g_device_available) {
        ASSERT_INT_EQ(bs_num_dies(g_ctx), 0);
        return;
    }
    uint32_t n = bs_num_dies(g_ctx);
    ASSERT_TRUE(n >= 2);
    ASSERT_TRUE(n <= BS_MAX_DIES);
}

/* Test: num_slots returns 1 (single MMIO slot in v2) */
static void test_num_slots(void)
{
    if (!g_device_available) {
        ASSERT_INT_EQ(bs_num_slots(g_ctx), 0);
        return;
    }
    ASSERT_INT_EQ(bs_num_slots(g_ctx), 1);
}

/* Test: active_die reports -1 when slot is empty */
static void test_slot_starts_empty(void)
{
    if (!g_device_available) {
        ASSERT_TRUE(1);  /* skip */
        return;
    }
    /* If a previous test run left a die acquired, release it first */
    int cur = bs_active_die(g_ctx);
    if (cur >= 0) {
        bs_gpu *stale = bs_acquire(g_ctx, cur);  /* re-acquire to get handle */
        bs_release(g_ctx, stale);
    }
    bs_refresh(g_ctx);
    /* Now slot should be empty */
    ASSERT_INT_EQ(bs_active_die(g_ctx), -1);
    uint32_t n = bs_num_dies(g_ctx);
    uint32_t i;
    for (i = 0; i < n; i++)
        ASSERT_INT_EQ(bs_is_mapped(g_ctx, (int)i), 0);
}

/* ---- Phase 2 tests ---- */

/* Test: acquire with NULL context returns NULL */
static void test_acquire_null_ctx(void)
{
    bs_gpu *gpu = bs_acquire(NULL, 0);
    ASSERT_TRUE(gpu == NULL);
}

/* Test: acquire invalid die_id returns NULL */
static void test_acquire_invalid_die(void)
{
    if (!g_device_available) {
        ASSERT_TRUE(1);  /* skip */
        return;
    }
    bs_gpu *gpu;

    gpu = bs_acquire(g_ctx, -1);
    ASSERT_TRUE(gpu == NULL);

    gpu = bs_acquire(g_ctx, 99);
    ASSERT_TRUE(gpu == NULL);
}

/* Test: release with NULL is safe no-op */
static void test_release_null(void)
{
    bs_release(NULL, NULL);
    if (g_device_available)
        bs_release(g_ctx, NULL);
    ASSERT_TRUE(1);
}

/* Test: acquire die 0, verify addresses, release */
static void test_acquire_basic(void)
{
    if (!g_device_available) {
        ASSERT_TRUE(1);  /* skip */
        return;
    }
    bs_gpu *gpu = bs_acquire(g_ctx, 0);
    ASSERT_NOT_NULL(gpu);
    ASSERT_INT_EQ(gpu->die_id, 0);
    ASSERT_TRUE(gpu->bar1_addr != 0);
    ASSERT_TRUE(gpu->bar1_size != 0);

    /* active_die should be 0 */
    ASSERT_INT_EQ(bs_active_die(g_ctx), 0);
    ASSERT_INT_EQ(bs_is_mapped(g_ctx, 0), 1);

    bs_release(g_ctx, gpu);

    /* After release, slot should be empty */
    ASSERT_INT_EQ(bs_active_die(g_ctx), -1);
    ASSERT_INT_EQ(bs_is_mapped(g_ctx, 0), 0);
}

/* Test: acquire/release each die in sequence */
static void test_acquire_all_dies(void)
{
    if (!g_device_available) {
        ASSERT_TRUE(1);  /* skip */
        return;
    }
    uint32_t n = bs_num_dies(g_ctx);
    uint32_t i;

    for (i = 0; i < n; i++) {
        bs_gpu *gpu = bs_acquire(g_ctx, (int)i);
        ASSERT_NOT_NULL(gpu);
        ASSERT_INT_EQ(gpu->die_id, (int)i);
        ASSERT_INT_EQ(bs_active_die(g_ctx), (int)i);
        bs_release(g_ctx, gpu);
        ASSERT_INT_EQ(bs_active_die(g_ctx), -1);
    }
}

/* Test: active_die tracking changes correctly */
static void test_active_die_tracking(void)
{
    if (!g_device_available) {
        /* Without device, active_die should be -1 */
        ASSERT_INT_EQ(bs_active_die(g_ctx), -1);
        return;
    }

    /* Start empty */
    ASSERT_INT_EQ(bs_active_die(g_ctx), -1);

    /* Acquire die 0 */
    bs_gpu *gpu0 = bs_acquire(g_ctx, 0);
    ASSERT_NOT_NULL(gpu0);
    ASSERT_INT_EQ(bs_active_die(g_ctx), 0);

    /* Release */
    bs_release(g_ctx, gpu0);
    ASSERT_INT_EQ(bs_active_die(g_ctx), -1);

    /* Acquire die 1 if available */
    if (bs_num_dies(g_ctx) >= 2) {
        bs_gpu *gpu1 = bs_acquire(g_ctx, 1);
        ASSERT_NOT_NULL(gpu1);
        ASSERT_INT_EQ(bs_active_die(g_ctx), 1);
        bs_release(g_ctx, gpu1);
        ASSERT_INT_EQ(bs_active_die(g_ctx), -1);
    }
}

/* Test: acquiring die B while die A is active (implicit release) */
static void test_implicit_release(void)
{
    if (!g_device_available || bs_num_dies(g_ctx) < 2) {
        ASSERT_TRUE(1);  /* skip */
        return;
    }

    /* Acquire die 0 */
    bs_gpu *gpu0 = bs_acquire(g_ctx, 0);
    ASSERT_NOT_NULL(gpu0);
    ASSERT_INT_EQ(bs_active_die(g_ctx), 0);

    /* Acquire die 1 — should implicitly release die 0 in kernel */
    bs_gpu *gpu1 = bs_acquire(g_ctx, 1);
    ASSERT_NOT_NULL(gpu1);
    ASSERT_INT_EQ(bs_active_die(g_ctx), 1);

    /* Die 0 is no longer mapped, die 1 is */
    bs_refresh(g_ctx);
    ASSERT_INT_EQ(bs_is_mapped(g_ctx, 0), 0);
    ASSERT_INT_EQ(bs_is_mapped(g_ctx, 1), 1);

    /* Clean up — free gpu0 handle (kernel already released it),
     * then release gpu1 properly */
    free(gpu0);  /* just free the userspace handle */
    bs_release(g_ctx, gpu1);
    ASSERT_INT_EQ(bs_active_die(g_ctx), -1);
}

/* Test: write pattern to VRAM, read back (requires server + /dev/mem) */
static void test_vram_roundtrip(void)
{
    if (!g_device_available) {
        ASSERT_TRUE(1);  /* skip */
        return;
    }

    bs_gpu *gpu = bs_acquire(g_ctx, 0);
    ASSERT_NOT_NULL(gpu);

    if (!gpu->vram) {
        /* /dev/mem not accessible — strict devmem or no root.
         * Verify the handle is still valid with addresses populated. */
        printf("    (note: VRAM not mmap'd — CONFIG_STRICT_DEVMEM or no root)\n");
        ASSERT_TRUE(gpu->bar1_addr != 0);
        ASSERT_TRUE(gpu->bar1_size != 0);
        bs_release(g_ctx, gpu);
        return;
    }

    /* Probe VRAM — without nvidia driver, GPU memory controller is
     * uninitialized and reads return 0xFFFFFFFF (PCIe timeout). */
    volatile uint32_t *vram = (volatile uint32_t *)gpu->vram;
    uint32_t probe = *vram;
    if (probe == 0xFFFFFFFF) {
        printf("    (note: VRAM reads 0xFFFFFFFF — GPU not initialized by nvidia)\n");
        ASSERT_TRUE(gpu->bar1_addr != 0);
        ASSERT_TRUE(gpu->bar1_size != 0);
        bs_release(g_ctx, gpu);
        return;
    }

    /* GPU is initialized — write a test pattern and read back */
    *vram = 0xDEADBEEF;
    uint32_t val = *vram;
    ASSERT_TRUE(val == 0xDEADBEEF);

    bs_release(g_ctx, gpu);
}

int main(void)
{
    printf("\n  barswap tests\n\n");

    setup();

    /* Phase 1 */
    RUN_TEST(init_free);
    RUN_TEST(null_safety);
    RUN_TEST(is_mapped_bounds);
    RUN_TEST(num_dies);
    RUN_TEST(num_slots);
    RUN_TEST(slot_starts_empty);

    /* Phase 2 */
    RUN_TEST(acquire_null_ctx);
    RUN_TEST(acquire_invalid_die);
    RUN_TEST(release_null);
    RUN_TEST(acquire_basic);
    RUN_TEST(acquire_all_dies);
    RUN_TEST(active_die_tracking);
    RUN_TEST(implicit_release);
    RUN_TEST(vram_roundtrip);

    teardown();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
