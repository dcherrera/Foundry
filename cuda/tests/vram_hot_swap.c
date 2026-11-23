/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * vram_hot_swap.c — Test CUDA context survival across bar_swap cycles
 *
 * The simplest approach to VRAM persistence: keep the CUDA context alive,
 * swap the die underneath via bar_swap, swap it back, and check if CUDA
 * can still read its data.
 *
 * Flow:
 *   1. Acquire die 0, load nvidia, CUDA alloc + write 0xDEADBEEF
 *   2. bar_swap release die 0 (nvidia stays loaded, CUDA context alive)
 *   3. bar_swap acquire die 0 again
 *   4. CUDA read back — does the data survive?
 *
 * Usage:
 *   sudo LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64 ./cuda/tests/vram_hot_swap
 *
 * Requires: bar_swap loaded, nvidia loaded, die 0 acquired, root.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <dlfcn.h>
#include <time.h>
#include "barswap/barswap.h"

#define TEST_SIZE    (1024 * 1024)    /* 1 MB */
#define PATTERN      0xDEADBEEF

typedef int CUresult;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;
#define CUDA_SUCCESS 0

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    printf("=== VRAM Hot Swap Test ===\n\n");

    int n_words = TEST_SIZE / sizeof(uint32_t);
    uint32_t *host_buf = malloc(TEST_SIZE);

    /* Load CUDA symbols */
    void *libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!libcuda) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    CUresult (*cuInit)(unsigned int) = dlsym(libcuda, "cuInit");
    CUresult (*cuDeviceGet)(int *, int) = dlsym(libcuda, "cuDeviceGet");
    CUresult (*cuCtxCreate)(CUcontext *, unsigned int, int) =
        dlsym(libcuda, "cuCtxCreate_v2");
    CUresult (*cuCtxDestroy)(CUcontext) = dlsym(libcuda, "cuCtxDestroy_v2");
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t) = dlsym(libcuda, "cuMemAlloc_v2");
    CUresult (*cuMemFree)(CUdeviceptr) = dlsym(libcuda, "cuMemFree_v2");
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t) =
        dlsym(libcuda, "cuMemcpyHtoD_v2");
    CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t) =
        dlsym(libcuda, "cuMemcpyDtoH_v2");
    CUresult (*cuCtxSynchronize)(void) = dlsym(libcuda, "cuCtxSynchronize");

    /* ================================================================
     * Step 1: CUDA alloc + write pattern (die 0 must already be acquired)
     * ================================================================ */
    printf("--- Step 1: CUDA alloc + write ---\n");

    CUresult err = cuInit(0);
    if (err) { fprintf(stderr, "cuInit: %d\n", err); return 1; }

    int dev;
    cuDeviceGet(&dev, 0);
    CUcontext cuda_ctx;
    err = cuCtxCreate(&cuda_ctx, 0, dev);
    if (err) { fprintf(stderr, "cuCtxCreate: %d\n", err); return 1; }
    printf("CUDA context created.\n");

    CUdeviceptr dptr;
    err = cuMemAlloc(&dptr, TEST_SIZE);
    if (err) { fprintf(stderr, "cuMemAlloc: %d\n", err); return 1; }
    printf("Allocated 1 MB at dptr=0x%llx\n", (unsigned long long)dptr);

    for (int i = 0; i < n_words; i++) host_buf[i] = PATTERN;
    cuMemcpyHtoD(dptr, host_buf, TEST_SIZE);
    cuCtxSynchronize();

    /* Verify */
    memset(host_buf, 0, TEST_SIZE);
    cuMemcpyDtoH(host_buf, dptr, TEST_SIZE);
    int ok = 1;
    for (int i = 0; i < n_words; i++) if (host_buf[i] != PATTERN) { ok = 0; break; }
    printf("Write verify: %s\n\n", ok ? "OK" : "FAIL");
    if (!ok) { fprintf(stderr, "Write failed, aborting.\n"); return 1; }

    /* ================================================================
     * Step 2: bar_swap release (CUDA context stays alive!)
     * ================================================================ */
    printf("--- Step 2: bar_swap release (CUDA stays alive) ---\n");

    bs_context *ctx = bs_init();
    if (!ctx) { fprintf(stderr, "bs_init failed\n"); return 1; }

    /* Release die 0 — this unmaps BAR0/BAR1 from the PCI slot.
     * nvidia driver still loaded, CUDA context still exists,
     * but the hardware is no longer accessible. */
    double t0 = now_ms();

    /* We can't use bs_release without a gpu handle from bs_acquire.
     * Use a raw ioctl to release die 0. */
    /* Actually, let's just use bs_acquire to get a handle first,
     * then release it. But bs_acquire will try to mmap BAR1 again...
     * Let's use the direct approach: release via acquire of a different
     * die (implicit release) then release that die. Or just release
     * via raw ioctl. */

    /* Simple approach: use the kernel module's implicit release.
     * Actually, the cleanest way is to do a raw RELEASE ioctl.
     * Let's include the kernel header. */
    /* Manually define the release ioctl to avoid header dep issues */
    struct release_req { uint32_t die_id; uint32_t flags; };
    #define MY_RELEASE _IOW('B', 5, struct release_req)

    int bs_fd = open("/dev/bar_swap", O_RDWR);
    if (bs_fd < 0) { perror("open /dev/bar_swap"); return 1; }

    struct release_req rel = { .die_id = 0, .flags = 0 };
    int ret = ioctl(bs_fd, MY_RELEASE, &rel);
    double t1 = now_ms();
    printf("Released die 0: %s (%.1fms)\n", ret == 0 ? "OK" : "FAIL", t1 - t0);
    if (ret != 0) { perror("release ioctl"); }

    printf("Die 0 released. CUDA context still alive. nvidia still loaded.\n");
    printf("Sleeping 500ms...\n\n");
    usleep(500000);

    /* ================================================================
     * Step 3: bar_swap acquire die 0 again
     * ================================================================ */
    printf("--- Step 3: bar_swap re-acquire die 0 ---\n");

    struct acquire_req {
        uint32_t die_id;
        uint32_t flags;
        uint64_t bar0_addr;
        uint64_t bar1_addr;
        uint64_t bar1_size;
        uint64_t bar3_addr;
    };
    #define MY_ACQUIRE _IOWR('B', 4, struct acquire_req)

    struct acquire_req acq = { .die_id = 0, .flags = 0 };
    t0 = now_ms();
    ret = ioctl(bs_fd, MY_ACQUIRE, &acq);
    t1 = now_ms();
    printf("Re-acquired die 0: %s (%.1fms)\n", ret == 0 ? "OK" : "FAIL", t1 - t0);
    if (ret == 0) {
        printf("  BAR0=0x%llx BAR1=0x%llx (size=0x%llx)\n",
               (unsigned long long)acq.bar0_addr,
               (unsigned long long)acq.bar1_addr,
               (unsigned long long)acq.bar1_size);
    } else {
        perror("acquire ioctl");
        close(bs_fd);
        return 1;
    }

    close(bs_fd);
    printf("Sleeping 200ms for hardware settle...\n\n");
    usleep(200000);

    /* ================================================================
     * Step 4: CUDA read back — did the data survive?
     * ================================================================ */
    printf("--- Step 4: CUDA read back ---\n");

    memset(host_buf, 0xAA, TEST_SIZE);  /* Fill with sentinel */
    err = cuMemcpyDtoH(host_buf, dptr, TEST_SIZE);
    printf("cuMemcpyDtoH: %s (err=%d)\n", err == CUDA_SUCCESS ? "OK" : "FAIL", err);

    if (err == CUDA_SUCCESS) {
        int match = 0, zeros = 0;
        uint32_t first_bad = 0;
        int first_bad_idx = -1;
        for (int i = 0; i < n_words; i++) {
            if (host_buf[i] == PATTERN) match++;
            else if (host_buf[i] == 0) zeros++;
            else if (first_bad_idx < 0) { first_bad_idx = i; first_bad = host_buf[i]; }
        }

        printf("Results: %d/%d match (%.1f%%), %d zeros",
               match, n_words, 100.0f * match / n_words, zeros);
        if (first_bad_idx >= 0)
            printf(", first bad at [%d]=0x%08X", first_bad_idx, first_bad);
        printf("\n");

        if (match == n_words) {
            printf("\n  >>> VRAM SURVIVES HOT SWAP! <<<\n");
            printf("  CUDA context + data persists across bar_swap release/acquire.\n");
            printf("  Architecture A is VIABLE: pre-load models, swap freely.\n");
        } else if (match > n_words / 2) {
            printf("\n  Partial persistence (%d/%d).\n", match, n_words);
        } else {
            printf("\n  Data did not survive the swap.\n");
        }
    } else {
        printf("CUDA DtoH copy failed — nvidia may have lost the GPU.\n");
        printf("Trying cuCtxSynchronize...\n");
        err = cuCtxSynchronize();
        printf("cuCtxSynchronize: %d\n", err);
    }

    /* ================================================================
     * Cleanup
     * ================================================================ */
    printf("\nCleaning up...\n");
    cuMemFree(dptr);
    cuCtxDestroy(cuda_ctx);

    bs_free(ctx);
    free(host_buf);
    dlclose(libcuda);

    printf("Done.\n");
    return 0;
}
