/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * compute_persist.c — Test whether GPU COMPUTE survives bar_swap release
 *
 * VRAM persistence (data at rest) is already confirmed (Architecture A).
 * This test checks whether IN-FLIGHT GPU computation continues executing
 * on the die after bar_swap release unmaps the BARs.
 *
 * If PASS:  True parallel MoE — launch kernels on die A, release,
 *           launch on die B, both compute simultaneously.
 * If FAIL:  Must sync before release — experts run sequentially.
 *
 * Usage (on GPU server as root):
 *   make compute-persist
 *   ./barswap/tests/compute_persist [die_id]
 *
 * Prerequisites:
 *   - bar_swap.ko loaded
 *   - Run as root (needs modprobe for nvidia)
 *   - Die available and idle
 *
 * What it does:
 *   1. Acquire die, create CUDA context, upload data
 *   2. Calibration: run kernel sync'd to measure execution time
 *   3. Launch kernel again (async), immediately release die
 *   4. Wait 3x kernel time
 *   5. Re-acquire die, sync, read results
 *   6. If results correct → compute persisted through release
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <time.h>

#include "barswap/barswap.h"

/* ================================================================
 * CUDA driver API types (no cuda.h dependency)
 * ================================================================ */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS         0
#define CUDA_ERROR_NOT_READY 600

/* ================================================================
 * Function pointer typedefs
 * ================================================================ */

typedef CUresult (*pfn_cuInit)(unsigned int);
typedef CUresult (*pfn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*pfn_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*pfn_cuCtxDestroy)(CUcontext);
typedef CUresult (*pfn_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*pfn_cuCtxSynchronize)(void);
typedef CUresult (*pfn_cuMemAlloc)(CUdeviceptr *, size_t);
typedef CUresult (*pfn_cuMemFree)(CUdeviceptr);
typedef CUresult (*pfn_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
typedef CUresult (*pfn_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
typedef CUresult (*pfn_cuModuleLoadData)(CUmodule *, const void *);
typedef CUresult (*pfn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult (*pfn_cuLaunchKernel)(CUfunction,
    unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, CUstream, void **, void **);

/* ================================================================
 * Globals
 * ================================================================ */

static void *libcuda = NULL;

static pfn_cuInit              p_cuInit;
static pfn_cuDeviceGet         p_cuDeviceGet;
static pfn_cuCtxCreate         p_cuCtxCreate;
static pfn_cuCtxDestroy        p_cuCtxDestroy;
static pfn_cuCtxSetCurrent     p_cuCtxSetCurrent;
static pfn_cuCtxSynchronize    p_cuCtxSynchronize;
static pfn_cuMemAlloc          p_cuMemAlloc;
static pfn_cuMemFree           p_cuMemFree;
static pfn_cuMemcpyHtoD        p_cuMemcpyHtoD;
static pfn_cuMemcpyDtoH        p_cuMemcpyDtoH;
static pfn_cuModuleLoadData    p_cuModuleLoadData;
static pfn_cuModuleGetFunction p_cuModuleGetFunction;
static pfn_cuLaunchKernel      p_cuLaunchKernel;

/* ================================================================
 * Embedded PTX kernel
 *
 * slow_add: each thread loads input[tid], adds 1.0 in a loop
 * `iters` times, writes to output[tid].
 *
 * Expected: output[i] = input[i] + iters
 * With input = 1.0 and iters = 10M: output = 10000001.0
 * (exact in float32 since 10000001 < 2^24 = 16777216)
 * ================================================================ */

static const char *slow_add_ptx =
    ".version 5.0\n"
    ".target sm_37\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry slow_add(\n"
    "    .param .u64 p_out,\n"
    "    .param .u64 p_in,\n"
    "    .param .u32 p_n,\n"
    "    .param .u32 p_iters\n"
    ") {\n"
    "    .reg .u32 %r<10>;\n"
    "    .reg .u64 %rd<10>;\n"
    "    .reg .f32 %f<4>;\n"
    "    .reg .pred %p1;\n"
    "\n"
    "    mov.u32 %r0, %tid.x;\n"
    "    mov.u32 %r1, %ctaid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mad.lo.u32 %r0, %r1, %r2, %r0;\n"
    "\n"
    "    ld.param.u32 %r3, [p_n];\n"
    "    setp.ge.u32 %p1, %r0, %r3;\n"
    "    @%p1 bra $L_done;\n"
    "\n"
    "    ld.param.u64 %rd0, [p_in];\n"
    "    cvt.u64.u32 %rd1, %r0;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.u64 %rd0, %rd0, %rd1;\n"
    "    ld.global.f32 %f0, [%rd0];\n"
    "\n"
    "    ld.param.u32 %r4, [p_iters];\n"
    "    mov.u32 %r5, 0;\n"
    "    mov.f32 %f1, 0f3F800000;\n"   /* 1.0f */
    "\n"
    "$L_loop:\n"
    "    add.f32 %f0, %f0, %f1;\n"
    "    add.u32 %r5, %r5, 1;\n"
    "    setp.lt.u32 %p1, %r5, %r4;\n"
    "    @%p1 bra $L_loop;\n"
    "\n"
    "    ld.param.u64 %rd2, [p_out];\n"
    "    cvt.u64.u32 %rd3, %r0;\n"
    "    shl.b64 %rd3, %rd3, 2;\n"
    "    add.u64 %rd2, %rd2, %rd3;\n"
    "    st.global.f32 [%rd2], %f0;\n"
    "\n"
    "$L_done:\n"
    "    ret;\n"
    "}\n";

/* ================================================================
 * Test parameters
 * ================================================================ */

#define N_ELEMENTS  (1024 * 1024)     /* 1M elements = 4 MB per buffer  */
#define N_ITERS     10000000          /* 10M adds per element            */
#define EXPECTED    (1.0f + N_ITERS)  /* 10000001.0                      */
#define BLOCK_SIZE  256

/* ================================================================
 * Helpers
 * ================================================================ */

static int load_nvidia(void) {
    printf("  Loading nvidia driver...\n");
    /* Each module must be loaded separately with --ignore-install
     * because "install nvidia /bin/false" in modprobe.d blocks
     * dependency resolution for submodules. */
    int ret = system("modprobe --ignore-install nvidia 2>/dev/null");
    if (ret != 0) {
        fprintf(stderr, "  modprobe nvidia failed\n");
        return -1;
    }
    sleep(1);
    /* nvidia_uvm is required for cuInit to succeed */
    ret = system("modprobe --ignore-install nvidia_uvm 2>/dev/null");
    if (ret != 0) {
        fprintf(stderr, "  modprobe nvidia_uvm failed\n");
        return -1;
    }
    sleep(1);
    return 0;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int load_cuda_symbols(void) {
    libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!libcuda) libcuda = dlopen("libcuda.so", RTLD_LAZY);
    if (!libcuda) {
        fprintf(stderr, "  Cannot dlopen libcuda: %s\n", dlerror());
        return -1;
    }

    /* Use _v2 symbols where needed (legacy v1 creates contexts
     * incompatible with modern CUDA operations) */
    #define LOAD(name, sym) do { \
        p_##name = (pfn_##name)dlsym(libcuda, sym); \
        if (!p_##name) { \
            fprintf(stderr, "  dlsym(%s) failed: %s\n", sym, dlerror()); \
            return -1; \
        } \
    } while(0)

    LOAD(cuInit,              "cuInit");
    LOAD(cuDeviceGet,         "cuDeviceGet");
    LOAD(cuCtxCreate,         "cuCtxCreate_v2");
    LOAD(cuCtxDestroy,        "cuCtxDestroy_v2");
    LOAD(cuCtxSetCurrent,     "cuCtxSetCurrent");
    LOAD(cuCtxSynchronize,    "cuCtxSynchronize");
    LOAD(cuMemAlloc,          "cuMemAlloc_v2");
    LOAD(cuMemFree,           "cuMemFree_v2");
    LOAD(cuMemcpyHtoD,        "cuMemcpyHtoD_v2");
    LOAD(cuMemcpyDtoH,        "cuMemcpyDtoH_v2");
    LOAD(cuModuleLoadData,    "cuModuleLoadData");
    LOAD(cuModuleGetFunction, "cuModuleGetFunction");
    LOAD(cuLaunchKernel,      "cuLaunchKernel");

    #undef LOAD
    return 0;
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char *argv[]) {
    int die_id = (argc > 1) ? atoi(argv[1]) : 0;

    printf("=== Compute Persistence Test ===\n\n");
    printf("Tests whether GPU computation continues after bar_swap release.\n");
    printf("Die: %d   Elements: %dM   Iters: %dM\n\n",
           die_id, N_ELEMENTS / 1000000, N_ITERS / 1000000);

    /* ---- Phase 1: Setup ---- */
    printf("Phase 1: Setup\n");

    bs_context *bs = bs_init();
    if (!bs) {
        fprintf(stderr, "  bs_init() failed — is bar_swap.ko loaded?\n");
        return 1;
    }
    printf("  bar_swap: %d dies available\n", bs_num_dies(bs));

    printf("  Acquiring die %d...\n", die_id);
    bs_gpu *gpu = bs_acquire(bs, die_id);
    if (!gpu) {
        fprintf(stderr, "  bs_acquire(%d) failed\n", die_id);
        bs_free(bs);
        return 1;
    }
    printf("  Die %d acquired (BARs programmed)\n", die_id);

    /* Load nvidia now that BARs are mapped — nvidia needs to see a GPU */
    if (load_nvidia() != 0) {
        fprintf(stderr, "  Failed to load nvidia driver\n");
        bs_release(bs, gpu);
        bs_free(bs);
        return 1;
    }
    printf("  nvidia loaded\n");

    if (load_cuda_symbols() != 0) {
        bs_release(bs, gpu);
        bs_free(bs);
        return 1;
    }
    printf("  CUDA symbols loaded\n");

    CUresult err;
    CUdevice dev;
    CUcontext ctx;

    err = p_cuInit(0);
    if (err) { fprintf(stderr, "  cuInit failed: %d\n", err); return 1; }

    err = p_cuDeviceGet(&dev, 0);
    if (err) { fprintf(stderr, "  cuDeviceGet failed: %d\n", err); return 1; }

    err = p_cuCtxCreate(&ctx, 0, dev);
    if (err) { fprintf(stderr, "  cuCtxCreate failed: %d\n", err); return 1; }
    printf("  CUDA context created\n");

    /* Allocate GPU buffers */
    size_t buf_bytes = (size_t)N_ELEMENTS * sizeof(float);
    CUdeviceptr d_in, d_out;

    err = p_cuMemAlloc(&d_in, buf_bytes);
    if (err) { fprintf(stderr, "  cuMemAlloc(in) failed: %d\n", err); return 1; }
    err = p_cuMemAlloc(&d_out, buf_bytes);
    if (err) { fprintf(stderr, "  cuMemAlloc(out) failed: %d\n", err); return 1; }
    printf("  GPU memory: 2x %.1f MB allocated\n", buf_bytes / (1024.0 * 1024.0));

    /* Upload input: all 1.0f */
    float *h_in = (float *)malloc(buf_bytes);
    for (int i = 0; i < N_ELEMENTS; i++) h_in[i] = 1.0f;
    err = p_cuMemcpyHtoD(d_in, h_in, buf_bytes);
    if (err) { fprintf(stderr, "  cuMemcpyHtoD failed: %d\n", err); return 1; }
    printf("  Input uploaded (%d elements, all 1.0)\n", N_ELEMENTS);

    /* Load PTX module */
    CUmodule mod;
    CUfunction kern;
    err = p_cuModuleLoadData(&mod, slow_add_ptx);
    if (err) { fprintf(stderr, "  cuModuleLoadData failed: %d\n", err); return 1; }
    err = p_cuModuleGetFunction(&kern, mod, "slow_add");
    if (err) { fprintf(stderr, "  cuModuleGetFunction failed: %d\n", err); return 1; }
    printf("  PTX kernel 'slow_add' loaded\n");

    /* ---- Phase 2: Calibration ---- */
    printf("\nPhase 2: Calibration (sync'd kernel run)\n");

    unsigned int n = N_ELEMENTS;
    unsigned int iters = N_ITERS;
    unsigned int grid = (N_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *args[] = { &d_out, &d_in, &n, &iters };

    double t0 = now_ms();
    err = p_cuLaunchKernel(kern, grid, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args, NULL);
    if (err) { fprintf(stderr, "  cuLaunchKernel failed: %d\n", err); return 1; }
    err = p_cuCtxSynchronize();
    if (err) { fprintf(stderr, "  cuCtxSynchronize failed: %d\n", err); return 1; }
    double kernel_ms = now_ms() - t0;
    printf("  Kernel time: %.0f ms\n", kernel_ms);

    /* Verify calibration output */
    float *h_cal = (float *)malloc(buf_bytes);
    p_cuMemcpyDtoH(h_cal, d_out, buf_bytes);

    int cal_ok = 1;
    for (int i = 0; i < N_ELEMENTS; i++) {
        if (h_cal[i] != EXPECTED) { cal_ok = 0; break; }
    }
    free(h_cal);

    if (!cal_ok) {
        fprintf(stderr, "  Calibration FAILED — kernel produced wrong output!\n");
        return 1;
    }
    printf("  Calibration PASSED — all %d elements = %.0f\n", N_ELEMENTS, EXPECTED);

    if (kernel_ms < 200) {
        printf("  WARNING: kernel fast (%.0f ms). May complete before release.\n", kernel_ms);
        printf("  Consider increasing N_ITERS if test gives false positive.\n");
    }

    /* Reset output buffer to zero for the real test */
    float *h_zeros = (float *)calloc(N_ELEMENTS, sizeof(float));
    p_cuMemcpyHtoD(d_out, h_zeros, buf_bytes);
    free(h_zeros);
    printf("  Output buffer reset to zeros\n");

    /* ---- Phase 3: Launch & Release ---- */
    printf("\nPhase 3: Launch kernel, then RELEASE die\n");

    t0 = now_ms();
    err = p_cuLaunchKernel(kern, grid, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args, NULL);
    if (err) { fprintf(stderr, "  cuLaunchKernel failed: %d\n", err); return 1; }
    double launch_ms = now_ms() - t0;
    printf("  Kernel launched (%.2f ms to enqueue)\n", launch_ms);
    printf("  NOT syncing — releasing die immediately...\n");

    t0 = now_ms();
    bs_release(bs, gpu);
    gpu = NULL;
    double release_ms = now_ms() - t0;
    printf("  Die %d released (%.2f ms)\n", die_id, release_ms);
    printf("  GPU is now unmapped from MMIO slot\n");

    /* Wait for well beyond the kernel's execution time */
    double wait_s = (kernel_ms / 1000.0) * 3.0;
    if (wait_s < 3.0) wait_s = 3.0;
    if (wait_s > 60.0) wait_s = 60.0;
    printf("  Waiting %.1f s (3x kernel time = %.0f ms)...\n",
           wait_s, kernel_ms * 3.0);

    struct timespec ts_wait = {
        .tv_sec  = (time_t)wait_s,
        .tv_nsec = (long)((wait_s - (time_t)wait_s) * 1e9)
    };
    nanosleep(&ts_wait, NULL);

    /* ---- Phase 4: Re-acquire & Check ---- */
    printf("\nPhase 4: Re-acquire die and check results\n");

    t0 = now_ms();
    gpu = bs_acquire(bs, die_id);
    double acquire_ms = now_ms() - t0;
    if (!gpu) {
        fprintf(stderr, "  bs_acquire(%d) FAILED on re-acquire!\n", die_id);
        bs_free(bs);
        return 1;
    }
    printf("  Die %d re-acquired (%.2f ms)\n", die_id, acquire_ms);

    /* Re-activate CUDA context */
    printf("  Re-activating CUDA context...\n");
    err = p_cuCtxSetCurrent(ctx);
    if (err) {
        fprintf(stderr, "  cuCtxSetCurrent FAILED: %d\n", err);
        fprintf(stderr, "  >>> CUDA context did not survive bar_swap release <<<\n");
        bs_release(bs, gpu);
        bs_free(bs);
        return 1;
    }
    printf("  CUDA context active\n");

    /* Sync — if compute persisted, this returns immediately.
     * If compute died, this might error or hang. */
    printf("  Synchronizing (cuCtxSynchronize)...\n");
    t0 = now_ms();
    err = p_cuCtxSynchronize();
    double sync_ms = now_ms() - t0;
    if (err) {
        fprintf(stderr, "  cuCtxSynchronize FAILED: %d (%.2f ms)\n", err, sync_ms);
        fprintf(stderr, "  >>> GPU compute context broken after release <<<\n");
        bs_release(bs, gpu);
        bs_free(bs);
        return 1;
    }
    printf("  Sync OK (%.2f ms) — %s\n", sync_ms,
           sync_ms < 10.0 ? "returned immediately (work already done)"
                          : "waited for GPU (work still in progress?)");

    /* Read back results */
    printf("  Reading results...\n");
    float *h_out = (float *)malloc(buf_bytes);
    memset(h_out, 0xAA, buf_bytes);  /* fill with sentinel */
    err = p_cuMemcpyDtoH(h_out, d_out, buf_bytes);
    if (err) {
        fprintf(stderr, "  cuMemcpyDtoH FAILED: %d\n", err);
        bs_release(bs, gpu);
        bs_free(bs);
        return 1;
    }

    /* ---- Phase 5: Analyze ---- */
    int correct = 0, zeros = 0, other = 0;
    for (int i = 0; i < N_ELEMENTS; i++) {
        if (h_out[i] == EXPECTED) correct++;
        else if (h_out[i] == 0.0f) zeros++;
        else other++;
    }

    printf("\n========================================\n");
    printf("  RESULTS\n");
    printf("========================================\n");
    printf("  Expected: %.0f  (1.0 + %d)\n", EXPECTED, N_ITERS);
    printf("  Correct:  %7d / %d  (%5.1f%%)\n",
           correct, N_ELEMENTS, 100.0 * correct / N_ELEMENTS);
    printf("  Zeros:    %7d / %d  (%5.1f%%)\n",
           zeros, N_ELEMENTS, 100.0 * zeros / N_ELEMENTS);
    printf("  Other:    %7d / %d  (%5.1f%%)\n",
           other, N_ELEMENTS, 100.0 * other / N_ELEMENTS);

    if (other > 0) {
        printf("  Sample 'other' values: ");
        int shown = 0;
        for (int i = 0; i < N_ELEMENTS && shown < 5; i++) {
            if (h_out[i] != EXPECTED && h_out[i] != 0.0f) {
                printf("[%d]=%.1f ", i, h_out[i]);
                shown++;
            }
        }
        printf("\n");
    }

    printf("========================================\n\n");

    if (correct == N_ELEMENTS) {
        printf(">>> COMPUTE PERSISTS THROUGH BAR_SWAP RELEASE! <<<\n");
        printf("\n");
        printf("GPU continued executing kernels after BARs were unmapped.\n");
        printf("This enables TRUE PARALLEL MoE:\n");
        printf("  1. Acquire die A → launch expert A kernels → release\n");
        printf("  2. Acquire die B → launch expert B kernels → release\n");
        printf("  3. Both dies compute simultaneously\n");
        printf("  4. Re-acquire A → collect results → re-acquire B → collect\n");
        printf("  Switch overhead: ~0.2ms per die (launch only)\n");
        printf("  Top-2 cost = top-1 cost (parallel, not sequential)\n");
    } else if (zeros == N_ELEMENTS) {
        printf(">>> COMPUTE DID NOT PERSIST <<<\n");
        printf("\n");
        printf("Output is all zeros — kernel did not write results.\n");
        printf("GPU halted (or results lost) when BARs were unmapped.\n");
        printf("MoE must sync before release — experts run sequentially.\n");
    } else if (correct > 0) {
        printf(">>> PARTIAL COMPUTE (%d%% done) <<<\n",
               (int)(100.0 * correct / N_ELEMENTS));
        printf("\n");
        printf("Some threads completed, others didn't.\n");
        printf("GPU may have stopped mid-execution when BARs unmapped.\n");
    } else {
        printf(">>> UNEXPECTED RESULTS <<<\n");
        printf("No correct outputs, no zeros. Investigate.\n");
    }

    printf("\nTiming summary:\n");
    printf("  Calibrated kernel time: %.0f ms\n", kernel_ms);
    printf("  Wait after release:     %.0f ms\n", wait_s * 1000);
    printf("  Sync after re-acquire:  %.2f ms\n", sync_ms);

    /* Cleanup */
    p_cuMemFree(d_in);
    p_cuMemFree(d_out);
    p_cuCtxDestroy(ctx);
    if (libcuda) dlclose(libcuda);
    free(h_in);
    free(h_out);
    bs_release(bs, gpu);
    bs_free(bs);

    return (correct == N_ELEMENTS) ? 0 : 1;
}
