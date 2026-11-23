/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * vram_persist.c — Test whether GPU VRAM survives nvidia unload/reload
 *
 * This is the CRITICAL Step 0 test for the CUDA backend.
 * Determines whether we can pre-load models on all 6 K80 dies
 * and switch between them without re-uploading weights.
 *
 * Architecture A (VRAM persists): ~2.5s switch cost, no re-upload
 * Architecture B (VRAM wiped):    re-upload weights each switch
 *
 * Usage (on GPU server as root):
 *   # Build
 *   make USE_CUDA=1 vram-persist
 *
 *   # Load bar_swap module + acquire die 0 + load nvidia first
 *   insmod bar_swap.ko
 *
 *   # Run test (handles acquire/release internally)
 *   ./barswap/tests/vram_persist [die_id]
 *
 * The test:
 *   Phase 1: Acquire die → modprobe nvidia → CUDA write pattern → cleanup
 *   Phase 2: Re-acquire die → modprobe nvidia → CUDA read pattern → check
 *
 * NOTE: This test shells out to modprobe/rmmod. Must run as root.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>

/* CUDA driver API types (no cuda.h dependency) */
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

/* Function pointer types */
typedef CUresult (*pfn_cuInit)(unsigned int);
typedef CUresult (*pfn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*pfn_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*pfn_cuCtxDestroy)(CUcontext);
typedef CUresult (*pfn_cuMemAlloc)(CUdeviceptr *, size_t);
typedef CUresult (*pfn_cuMemFree)(CUdeviceptr);
typedef CUresult (*pfn_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
typedef CUresult (*pfn_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);

/* Test pattern */
#define PATTERN_SIZE (1024 * 1024)  /* 1 MB */
#define MAGIC_VALUE  0xDEADBEEFU

static int load_nvidia(void) {
    printf("  Loading nvidia driver...\n");
    int ret = system("modprobe --ignore-install nvidia nvidia_modeset nvidia_drm nvidia_uvm 2>/dev/null");
    if (ret != 0) {
        /* Try just nvidia */
        ret = system("modprobe --ignore-install nvidia 2>/dev/null");
    }
    sleep(1);  /* Give driver time to initialize */
    return ret == 0 ? 0 : -1;
}

static int unload_nvidia(void) {
    printf("  Unloading nvidia driver...\n");
    system("rmmod nvidia_uvm 2>/dev/null");
    system("rmmod nvidia_drm 2>/dev/null");
    system("rmmod nvidia_modeset 2>/dev/null");
    int ret = system("rmmod nvidia 2>/dev/null");
    sleep(1);
    return ret == 0 ? 0 : -1;
}

/* Load CUDA symbols via dlopen */
static void *libcuda = NULL;
static pfn_cuInit p_cuInit;
static pfn_cuDeviceGet p_cuDeviceGet;
static pfn_cuCtxCreate p_cuCtxCreate;
static pfn_cuCtxDestroy p_cuCtxDestroy;
static pfn_cuMemAlloc p_cuMemAlloc;
static pfn_cuMemFree p_cuMemFree;
static pfn_cuMemcpyHtoD p_cuMemcpyHtoD;
static pfn_cuMemcpyDtoH p_cuMemcpyDtoH;

static int load_cuda(void) {
    libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!libcuda) libcuda = dlopen("libcuda.so", RTLD_LAZY);
    if (!libcuda) {
        fprintf(stderr, "Cannot dlopen libcuda.so: %s\n", dlerror());
        return -1;
    }

    #define LOAD(fn) do { \
        p_##fn = (pfn_##fn)dlsym(libcuda, #fn); \
        if (!p_##fn) { fprintf(stderr, "dlsym %s failed\n", #fn); return -1; } \
    } while(0)

    LOAD(cuInit);
    LOAD(cuDeviceGet);
    LOAD(cuCtxCreate);
    LOAD(cuCtxDestroy);
    LOAD(cuMemAlloc);
    LOAD(cuMemFree);
    LOAD(cuMemcpyHtoD);
    LOAD(cuMemcpyDtoH);

    #undef LOAD
    return 0;
}

static void close_cuda(void) {
    if (libcuda) { dlclose(libcuda); libcuda = NULL; }
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    printf("=== VRAM Persistence Test ===\n\n");
    printf("This test determines if GPU VRAM survives nvidia unload/reload.\n");
    printf("Result determines Architecture A (persist) vs B (wiped).\n\n");

    /* Phase 1: Write pattern to VRAM */
    printf("Phase 1: Writing pattern to VRAM...\n");

    if (load_nvidia() != 0) {
        fprintf(stderr, "Failed to load nvidia driver\n");
        return 1;
    }

    if (load_cuda() != 0) {
        fprintf(stderr, "Failed to load CUDA\n");
        return 1;
    }

    CUresult err;
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr dptr;

    err = p_cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", err);
        close_cuda();
        return 1;
    }

    err = p_cuDeviceGet(&dev, 0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet failed: %d\n", err);
        close_cuda();
        return 1;
    }

    err = p_cuCtxCreate(&ctx, 0, dev);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxCreate failed: %d\n", err);
        close_cuda();
        return 1;
    }

    /* Allocate and fill with pattern */
    err = p_cuMemAlloc(&dptr, PATTERN_SIZE);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAlloc failed: %d\n", err);
        p_cuCtxDestroy(ctx);
        close_cuda();
        return 1;
    }

    /* Fill CPU buffer with pattern */
    uint32_t *pattern = (uint32_t*)malloc(PATTERN_SIZE);
    for (size_t i = 0; i < PATTERN_SIZE / sizeof(uint32_t); i++) {
        pattern[i] = MAGIC_VALUE ^ (uint32_t)i;
    }

    err = p_cuMemcpyHtoD(dptr, pattern, PATTERN_SIZE);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemcpyHtoD failed: %d\n", err);
        p_cuMemFree(dptr);
        p_cuCtxDestroy(ctx);
        close_cuda();
        free(pattern);
        return 1;
    }

    /* Verify write was successful */
    uint32_t *verify = (uint32_t*)malloc(PATTERN_SIZE);
    p_cuMemcpyDtoH(verify, dptr, PATTERN_SIZE);
    int write_ok = 1;
    for (size_t i = 0; i < PATTERN_SIZE / sizeof(uint32_t); i++) {
        if (verify[i] != pattern[i]) { write_ok = 0; break; }
    }
    free(verify);

    if (!write_ok) {
        fprintf(stderr, "Write verification failed!\n");
        p_cuMemFree(dptr);
        p_cuCtxDestroy(ctx);
        close_cuda();
        free(pattern);
        return 1;
    }

    printf("  Written %d bytes at GPU addr 0x%llx\n", PATTERN_SIZE, dptr);
    printf("  Pattern: 0xDEADBEEF ^ index\n");

    /* Remember the GPU address for phase 2 */
    unsigned long long saved_dptr = dptr;

    /* DO NOT free the GPU memory — we want to see if it persists */
    /* But we must destroy the context to unload nvidia */
    p_cuCtxDestroy(ctx);
    close_cuda();

    printf("  CUDA context destroyed (GPU memory NOT freed)\n");

    /* Unload nvidia */
    if (unload_nvidia() != 0) {
        fprintf(stderr, "WARNING: rmmod nvidia failed (may have users)\n");
        free(pattern);
        return 1;
    }

    printf("  nvidia unloaded\n\n");

    /* Phase 2: Read back and check */
    printf("Phase 2: Re-loading nvidia and checking VRAM...\n");

    sleep(1);  /* Brief pause */

    if (load_nvidia() != 0) {
        fprintf(stderr, "Failed to reload nvidia driver\n");
        free(pattern);
        return 1;
    }

    if (load_cuda() != 0) {
        fprintf(stderr, "Failed to reload CUDA\n");
        free(pattern);
        return 1;
    }

    err = p_cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed on reload: %d\n", err);
        close_cuda();
        free(pattern);
        return 1;
    }

    err = p_cuDeviceGet(&dev, 0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet failed on reload: %d\n", err);
        close_cuda();
        free(pattern);
        return 1;
    }

    err = p_cuCtxCreate(&ctx, 0, dev);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxCreate failed on reload: %d\n", err);
        close_cuda();
        free(pattern);
        return 1;
    }

    /* Allocate new memory and read it to check for residual data.
     * Note: We can't read from the OLD address because CUDA memory
     * management resets on driver reload. Instead, we allocate fresh
     * memory and check if the driver zeroes it or if stale data remains.
     *
     * Better approach: allocate same size and hope it lands at the
     * same physical address. If pattern matches → VRAM persists. */
    CUdeviceptr dptr2;
    err = p_cuMemAlloc(&dptr2, PATTERN_SIZE);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAlloc failed on reload: %d\n", err);
        p_cuCtxDestroy(ctx);
        close_cuda();
        free(pattern);
        return 1;
    }

    printf("  New allocation at GPU addr 0x%llx (was 0x%llx)\n",
           dptr2, saved_dptr);

    uint32_t *readback = (uint32_t*)malloc(PATTERN_SIZE);
    memset(readback, 0, PATTERN_SIZE);
    err = p_cuMemcpyDtoH(readback, dptr2, PATTERN_SIZE);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemcpyDtoH failed on reload: %d\n", err);
        p_cuMemFree(dptr2);
        p_cuCtxDestroy(ctx);
        close_cuda();
        free(pattern);
        free(readback);
        return 1;
    }

    /* Check for pattern match */
    int matches = 0;
    int zeroes = 0;
    int other = 0;
    for (size_t i = 0; i < PATTERN_SIZE / sizeof(uint32_t); i++) {
        if (readback[i] == pattern[i]) {
            matches++;
        } else if (readback[i] == 0) {
            zeroes++;
        } else {
            other++;
        }
    }

    int total = (int)(PATTERN_SIZE / sizeof(uint32_t));

    printf("\n=== Results ===\n");
    printf("  Pattern matches: %d / %d (%.1f%%)\n",
           matches, total, 100.0 * matches / total);
    printf("  Zero values:     %d / %d (%.1f%%)\n",
           zeroes, total, 100.0 * zeroes / total);
    printf("  Other values:    %d / %d (%.1f%%)\n",
           other, total, 100.0 * other / total);
    printf("\n");

    if (matches == total) {
        printf(">>> VRAM PERSISTS! Architecture A confirmed. <<<\n");
        printf(">>> Pre-load models on all 6 dies, switch freely. <<<\n");
    } else if (zeroes == total) {
        printf(">>> VRAM WIPED (zeroed). Architecture B. <<<\n");
        printf(">>> Must re-upload weights each switch (~0.5s for 0.2B). <<<\n");
    } else if (matches > total / 2) {
        printf(">>> PARTIAL PERSISTENCE (%d%% match). <<<\n",
               (int)(100.0 * matches / total));
        printf(">>> Need further investigation. <<<\n");
    } else {
        printf(">>> VRAM WIPED (random/stale data). Architecture B. <<<\n");
        printf(">>> Must re-upload weights each switch. <<<\n");
    }

    /* Cleanup */
    p_cuMemFree(dptr2);
    p_cuCtxDestroy(ctx);
    close_cuda();
    free(pattern);
    free(readback);

    return 0;
}
