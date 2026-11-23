/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * vram_bar1_persist.c — Test VRAM persistence via direct BAR1 mmap
 *
 * This is the critical experiment for multi-die model persistence.
 * Tests whether K80 VRAM retains data across bar_swap release/acquire
 * cycles using direct BAR1 access (no CUDA, no nvidia driver).
 *
 * Three experiments:
 *
 *   Experiment 1: Raw BAR1 persistence
 *     1. Acquire die → write 0xDEADBEEF via BAR1 mmap → release
 *     2. Re-acquire same die → read BAR1 → check pattern
 *     Tests: Does physical VRAM retain data?
 *
 *   Experiment 2: nvidia init impact (--with-nvidia)
 *     1. Write pattern via BAR1
 *     2. Load nvidia driver
 *     3. Read BAR1 → does nvidia wipe VRAM?
 *
 *   Experiment 3: CUDA visibility (--with-cuda)
 *     1. Pre-write known pattern to BAR1 at offset 0
 *     2. Load nvidia, create CUDA context
 *     3. Allocate CUDA buffer, read its contents
 *     4. Check if CUDA sees the pre-written data
 *
 * Usage:
 *   sudo ./cuda/tests/vram_bar1_persist [die_id]                  # Experiment 1 only
 *   sudo ./cuda/tests/vram_bar1_persist [die_id] --with-nvidia    # Experiments 1+2
 *   sudo ./cuda/tests/vram_bar1_persist [die_id] --with-cuda      # Experiments 1+2+3
 *
 * Must be run as root (for /dev/mem and /dev/bar_swap).
 * nvidia must NOT be loaded when starting (experiment loads it itself).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "barswap/barswap.h"

#define TEST_SIZE   (1024 * 1024)   /* 1 MB */
#define PATTERN     0xDEADBEEF
#define VRAM_OFFSET 0               /* Write at start of BAR1 */

/* Second offset to test — well past nvidia's internal bookkeeping area.
 * nvidia typically uses the first ~256MB for page tables, channels, etc.
 * We test at 512MB to see if data outside that zone survives. */
#define VRAM_OFFSET_HIGH (512ULL * 1024 * 1024)

static int run_cmd(const char *cmd) {
    printf("  > %s\n", cmd);
    return system(cmd);
}

static int check_pattern(volatile uint32_t *base, int n_words, uint32_t pattern,
                         const char *label) {
    int match = 0;
    int first_mismatch = -1;
    uint32_t first_bad = 0;

    for (int i = 0; i < n_words; i++) {
        if (base[i] == pattern) {
            match++;
        } else if (first_mismatch < 0) {
            first_mismatch = i;
            first_bad = base[i];
        }
    }

    float pct = 100.0f * match / n_words;
    if (match == n_words) {
        printf("  [%s] ALL %d words match 0x%08X (100%%)\n", label, n_words, pattern);
        return 1;
    } else {
        printf("  [%s] %d/%d words match (%.1f%%). First mismatch at word %d: 0x%08X\n",
               label, match, n_words, pct, first_mismatch, first_bad);
        return 0;
    }
}

static void write_pattern(volatile uint32_t *base, int n_words, uint32_t pattern) {
    for (int i = 0; i < n_words; i++) {
        base[i] = pattern;
    }
    /* Memory barrier to ensure writes are flushed to device */
    __sync_synchronize();
}

int main(int argc, char *argv[]) {
    int die_id = 0;
    int with_nvidia = 0;
    int with_cuda = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--with-nvidia") == 0) {
            with_nvidia = 1;
        } else if (strcmp(argv[i], "--with-cuda") == 0) {
            with_nvidia = 1;
            with_cuda = 1;
        } else {
            die_id = atoi(argv[i]);
        }
    }

    printf("=== VRAM BAR1 Persistence Test ===\n");
    printf("Die: %d\n", die_id);
    printf("Mode: raw BAR1%s%s\n\n",
           with_nvidia ? " + nvidia" : "",
           with_cuda ? " + CUDA" : "");

    int n_words = TEST_SIZE / sizeof(uint32_t);

    /* ================================================================
     * Experiment 1: Raw BAR1 persistence
     * ================================================================ */
    printf("--- Experiment 1: Raw BAR1 VRAM Persistence ---\n\n");

    bs_context *ctx = bs_init();
    if (!ctx) {
        fprintf(stderr, "ERROR: bs_init failed (is bar_swap loaded?)\n");
        return 1;
    }

    printf("Phase 1a: Acquire die %d, write pattern via BAR1\n", die_id);
    bs_gpu *gpu = bs_acquire(ctx, die_id);
    if (!gpu) {
        fprintf(stderr, "ERROR: bs_acquire(%d) failed\n", die_id);
        bs_free(ctx);
        return 1;
    }

    if (!gpu->vram) {
        fprintf(stderr, "ERROR: VRAM mmap failed (need root for /dev/mem)\n");
        bs_release(ctx, gpu);
        bs_free(ctx);
        return 1;
    }

    printf("  BAR1 addr: 0x%lx, size: 0x%lx (%lu GB)\n",
           (unsigned long)gpu->bar1_addr,
           (unsigned long)gpu->bar1_size,
           (unsigned long)gpu->bar1_size / (1024*1024*1024));
    printf("  VRAM mmap: %p\n", gpu->vram);

    /* Write at offset 0 */
    volatile uint32_t *vram_low = (volatile uint32_t *)((uint8_t *)gpu->vram + VRAM_OFFSET);
    printf("  Writing %d words of 0x%08X at offset 0x%X...\n",
           n_words, PATTERN, VRAM_OFFSET);
    write_pattern(vram_low, n_words, PATTERN);

    /* Verify write */
    printf("  Verifying write...\n");
    if (!check_pattern(vram_low, n_words, PATTERN, "write-verify")) {
        fprintf(stderr, "ERROR: Write verification failed!\n");
        bs_release(ctx, gpu);
        bs_free(ctx);
        return 1;
    }

    /* Also write at high offset (past nvidia bookkeeping area) */
    volatile uint32_t *vram_high = NULL;
    if (VRAM_OFFSET_HIGH + TEST_SIZE <= gpu->bar1_size) {
        vram_high = (volatile uint32_t *)((uint8_t *)gpu->vram + VRAM_OFFSET_HIGH);
        printf("  Writing %d words of 0x%08X at offset 0x%llX (512MB)...\n",
               n_words, PATTERN, (unsigned long long)VRAM_OFFSET_HIGH);
        write_pattern(vram_high, n_words, PATTERN);
        check_pattern(vram_high, n_words, PATTERN, "high-write-verify");
    }

    printf("\nPhase 1b: Release die %d\n", die_id);
    bs_release(ctx, gpu);
    gpu = NULL;
    printf("  Released.\n\n");

    /* Small delay to let hardware settle */
    usleep(100000);  /* 100ms */

    printf("Phase 1c: Re-acquire die %d, read back\n", die_id);
    gpu = bs_acquire(ctx, die_id);
    if (!gpu || !gpu->vram) {
        fprintf(stderr, "ERROR: Re-acquire failed\n");
        bs_free(ctx);
        return 1;
    }

    vram_low = (volatile uint32_t *)((uint8_t *)gpu->vram + VRAM_OFFSET);
    printf("  Reading back %d words at offset 0x%X...\n", n_words, VRAM_OFFSET);
    int exp1_low = check_pattern(vram_low, n_words, PATTERN, "reacquire-low");

    int exp1_high = 0;
    if (VRAM_OFFSET_HIGH + TEST_SIZE <= gpu->bar1_size) {
        vram_high = (volatile uint32_t *)((uint8_t *)gpu->vram + VRAM_OFFSET_HIGH);
        printf("  Reading back %d words at offset 0x%llX (512MB)...\n",
               n_words, (unsigned long long)VRAM_OFFSET_HIGH);
        exp1_high = check_pattern(vram_high, n_words, PATTERN, "reacquire-high");
    }

    printf("\n=== Experiment 1 Result: %s ===\n",
           (exp1_low && exp1_high) ? "VRAM PERSISTS" :
           (exp1_low || exp1_high) ? "PARTIAL PERSISTENCE" : "VRAM WIPED");

    if (!exp1_low && !exp1_high) {
        printf("  Hardware wipes VRAM on BAR remap. Cannot proceed.\n");
        bs_release(ctx, gpu);
        bs_free(ctx);
        return 1;
    }

    if (!with_nvidia) {
        printf("\nDone. Run with --with-nvidia to test nvidia driver impact.\n");
        bs_release(ctx, gpu);
        bs_free(ctx);
        return (exp1_low && exp1_high) ? 0 : 1;
    }

    /* ================================================================
     * Experiment 2: Does nvidia init wipe VRAM?
     * ================================================================ */
    printf("\n--- Experiment 2: nvidia Driver Impact ---\n\n");

    /* Re-write fresh pattern (in case exp 1 left partial data) */
    printf("Phase 2a: Writing fresh pattern to both offsets...\n");
    vram_low = (volatile uint32_t *)((uint8_t *)gpu->vram + VRAM_OFFSET);
    write_pattern(vram_low, n_words, PATTERN);
    check_pattern(vram_low, n_words, PATTERN, "pre-nvidia-low");

    if (VRAM_OFFSET_HIGH + TEST_SIZE <= gpu->bar1_size) {
        vram_high = (volatile uint32_t *)((uint8_t *)gpu->vram + VRAM_OFFSET_HIGH);
        write_pattern(vram_high, n_words, PATTERN);
        check_pattern(vram_high, n_words, PATTERN, "pre-nvidia-high");
    }

    printf("\nPhase 2b: Loading nvidia driver...\n");
    run_cmd("modprobe --ignore-install nvidia");
    run_cmd("modprobe --ignore-install nvidia_uvm");
    sleep(1);

    printf("\nPhase 2c: Reading BAR1 after nvidia init...\n");

    /* Re-mmap because nvidia might have changed BAR config.
     * Actually, we still have the same mmap — let's just read. */
    vram_low = (volatile uint32_t *)((uint8_t *)gpu->vram + VRAM_OFFSET);
    int exp2_low = check_pattern(vram_low, n_words, PATTERN, "post-nvidia-low");

    int exp2_high = 0;
    if (VRAM_OFFSET_HIGH + TEST_SIZE <= gpu->bar1_size) {
        vram_high = (volatile uint32_t *)((uint8_t *)gpu->vram + VRAM_OFFSET_HIGH);
        exp2_high = check_pattern(vram_high, n_words, PATTERN, "post-nvidia-high");
    }

    printf("\n=== Experiment 2 Result: %s ===\n",
           (exp2_low && exp2_high) ? "SURVIVES nvidia init" :
           exp2_high ? "Low wiped, HIGH SURVIVES (nvidia uses low VRAM only)" :
           exp2_low ? "High wiped, low survives (unexpected)" :
           "BOTH WIPED by nvidia init");

    if (exp2_high && !exp2_low) {
        printf("  nvidia driver uses first ~256MB for internal bookkeeping.\n");
        printf("  We can pre-load weights at offset 512MB+ and they survive!\n");
    }

    if (!with_cuda) {
        printf("\nUnloading nvidia...\n");
        run_cmd("rmmod nvidia_uvm 2>/dev/null");
        run_cmd("rmmod nvidia_drm 2>/dev/null");
        run_cmd("rmmod nvidia_modeset 2>/dev/null");
        run_cmd("rmmod nvidia 2>/dev/null");
        printf("\nDone. Run with --with-cuda to test CUDA visibility.\n");
        bs_release(ctx, gpu);
        bs_free(ctx);
        return (exp2_low || exp2_high) ? 0 : 1;
    }

    /* ================================================================
     * Experiment 3: Can CUDA see pre-written BAR1 data?
     * ================================================================ */
    printf("\n--- Experiment 3: CUDA Visibility ---\n\n");

    /* nvidia is already loaded from exp 2. Now we need to test if
     * data written via BAR1 is visible from CUDA.
     *
     * Strategy: We know the BAR1 physical address. CUDA's GPU virtual
     * addresses map to physical VRAM. If we can figure out the mapping,
     * we can pre-write at the right physical offset.
     *
     * For now, we just allocate a CUDA buffer and dump its contents
     * to see if any of the pattern survived at the allocation offset. */

    printf("Phase 3a: Attempting CUDA init via fc_cuda...\n");

    /* We dlopen fc_cuda functions manually since we may not have
     * linked with fc_cuda.c. Instead, use the CUDA driver API directly
     * via the same dlopen approach. */
    #include <dlfcn.h>

    typedef int CUresult;
    typedef void* CUcontext;
    typedef unsigned long long CUdeviceptr;

    void *libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!libcuda) {
        printf("  Cannot load libcuda.so.1: %s\n", dlerror());
        printf("  Skipping CUDA experiment.\n");
        goto cleanup;
    }

    CUresult (*cuInit)(unsigned int) = dlsym(libcuda, "cuInit");
    CUresult (*cuCtxCreate)(CUcontext*, unsigned int, int) =
        dlsym(libcuda, "cuCtxCreate_v2");
    CUresult (*cuCtxDestroy)(CUcontext) =
        dlsym(libcuda, "cuCtxDestroy_v2");
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t) =
        dlsym(libcuda, "cuMemAlloc_v2");
    CUresult (*cuMemFree)(CUdeviceptr) =
        dlsym(libcuda, "cuMemFree_v2");
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) =
        dlsym(libcuda, "cuMemcpyDtoH_v2");
    CUresult (*cuDeviceGet)(int*, int) =
        dlsym(libcuda, "cuDeviceGet");

    if (!cuInit || !cuCtxCreate || !cuMemAlloc || !cuMemcpyDtoH) {
        printf("  Failed to load CUDA symbols\n");
        goto cleanup;
    }

    CUresult err = cuInit(0);
    if (err != 0) {
        printf("  cuInit failed: %d\n", err);
        goto cleanup;
    }

    int dev;
    cuDeviceGet(&dev, 0);
    CUcontext cuda_ctx;
    err = cuCtxCreate(&cuda_ctx, 0, dev);
    if (err != 0) {
        printf("  cuCtxCreate failed: %d\n", err);
        goto cleanup;
    }
    printf("  CUDA context created.\n");

    /* Allocate several buffers and check their contents.
     * CUDA may or may not zero them. If any contain our pattern,
     * we found a way to persist data. */
    printf("\nPhase 3b: Allocating CUDA buffers and checking contents...\n");

    uint32_t *host_buf = (uint32_t *)malloc(TEST_SIZE);
    int found_pattern = 0;

    for (int trial = 0; trial < 8; trial++) {
        CUdeviceptr dptr;
        err = cuMemAlloc(&dptr, TEST_SIZE);
        if (err != 0) {
            printf("  cuMemAlloc trial %d failed: %d\n", trial, err);
            break;
        }

        /* Read the GPU memory WITHOUT writing to it first */
        memset(host_buf, 0, TEST_SIZE);
        cuMemcpyDtoH(host_buf, dptr, TEST_SIZE);

        /* Check for our pattern */
        int matches = 0;
        for (int i = 0; i < n_words; i++) {
            if (host_buf[i] == PATTERN) matches++;
        }

        printf("  Trial %d: dptr=0x%llx, pattern matches=%d/%d (%.1f%%)",
               trial, (unsigned long long)dptr, matches, n_words,
               100.0f * matches / n_words);
        if (matches > 0) {
            printf(" *** FOUND PATTERN ***");
            found_pattern = 1;
        }

        /* Also check if it's all zeros (CUDA zeroed it) or random */
        int zeros = 0;
        for (int i = 0; i < n_words; i++) {
            if (host_buf[i] == 0) zeros++;
        }
        printf(", zeros=%d/%d\n", zeros, n_words);

        cuMemFree(dptr);
    }

    printf("\n=== Experiment 3 Result: %s ===\n",
           found_pattern ? "PATTERN VISIBLE FROM CUDA" : "CUDA cannot see BAR1 data");

    if (found_pattern) {
        printf("  Pre-written BAR1 data IS visible from CUDA allocations!\n");
        printf("  Architecture A is viable: pre-load weights via BAR1, use from CUDA.\n");
    } else {
        printf("  CUDA allocator zeros memory or maps to different physical VRAM.\n");
        printf("  Need to investigate GPU virtual → physical mapping.\n");
    }

    cuCtxDestroy(cuda_ctx);

cleanup:
    printf("\nCleaning up...\n");
    run_cmd("rmmod nvidia_uvm 2>/dev/null");
    run_cmd("rmmod nvidia_drm 2>/dev/null");
    run_cmd("rmmod nvidia_modeset 2>/dev/null");
    run_cmd("rmmod nvidia 2>/dev/null");

    bs_release(ctx, gpu);
    bs_free(ctx);
    if (libcuda) dlclose(libcuda);

    printf("\n=== Summary ===\n");
    printf("  Exp 1 (raw BAR1):     %s\n",
           (exp1_low && exp1_high) ? "PERSIST" : "WIPED");
    printf("  Exp 2 (nvidia init):  %s\n",
           with_nvidia ?
           ((exp2_low && exp2_high) ? "SURVIVES" :
            exp2_high ? "HIGH SURVIVES (low wiped)" :
            "WIPED") : "skipped");
    printf("  Exp 3 (CUDA):         %s\n",
           with_cuda ? (found_pattern ? "VISIBLE" : "NOT VISIBLE") : "skipped");

    printf("\nDone.\n");
    return 0;
}
