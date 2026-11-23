/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#define _GNU_SOURCE

/*
 * worker_switch.c — Does Pantheon's SETTLED GPU-owner shape work: a privileged
 *                   PARENT that forks a WORKER PROCESS per active die, switching
 *                   dies by KILLING the worker (fd close drops usage_count) then
 *                   unbind/bind?  And does VRAM survive the worker's process exit
 *                   so the next worker re-reads weights WITHOUT re-uploading?
 *
 * WHY THIS EXACT SHAPE (settled from NVIDIA's own source):
 *
 *   The nvidia device usage_count decrements ONLY when the process that opened
 *   /dev/nvidiaN closes that fd. cuCtxDestroy and cuDevicePrimaryCtxReset do NOT
 *   drop it — so any IN-PROCESS attempt to unbind busy-waits forever on a
 *   non-zero usage count (this is exactly what gpu_switch_probe.c risks).
 *   ⇒ The owner must be a PARENT that forks a WORKER child per active die.
 *     A die switch is: KILL the worker (its fd close drives usage_count to 0)
 *     -> unbind the die -> bind the next die -> fork the next worker.
 *   Here the worker exits on its own when done, which closes the fd identically
 *   to a kill — so the parent's post-worker unbind should return FAST/CLEAN.
 *
 * THE ONE PRODUCT-CRITICAL GATE this test validates:
 *
 *   Does VRAM physically PERSIST across a worker's PROCESS EXIT, so the next
 *   worker that re-allocs the same buffers (same size, same order) re-reads the
 *   weights WITHOUT re-uploading?  Re-upload is ~40 s/16 GB at the rig's
 *   0.4 GB/s host link — a non-starter.  So this gate MUST pass.
 *
 *   This extends vram_persist.c's trick (same size/order alloc -> same physical
 *   VRAM -> read WITHOUT writing -> compare) across a PROCESS BOUNDARY instead
 *   of an rmmod: the WRITE worker uploads + computes a seed and EXITS WITHOUT
 *   cuMemFree (so the bytes are only logically freed); a later READVERIFY worker
 *   re-allocs the same order and reads back WITHOUT uploading.
 *
 * PARENT/WORKER SPLIT:
 *   - PARENT (owner): owns barswap, driver_override/block_all_except, and the
 *     nvidia bind/unbind. It never opens /dev/nvidiaN itself — it forks a worker
 *     that does. It times every kill->unbind->bind and prints the verdict.
 *   - WORKER (run_worker): one die, attaches to the already-bound nvidia
 *     (cuInit -> cuDeviceGet(0) -> cuCtxCreate), does WRITE or READVERIFY, and
 *     reports via exit code (0 = pass) + stdout.  Workers _Exit() too.
 *
 * SEQUENCE (loud per-step prints, unbuffered stdout, timed unbinds):
 *   1. bs_init (NULL -> [SKIP] no barswap, exit 0); discover_gk210s; load nvidia
 *      once if needed.
 *   2. Switch-to(A, WRITE): acquire A -> block others -> bind A -> fork worker;
 *      worker uploads SEED_A, runs slow_add, prints its device pointers, exits
 *      WITHOUT freeing.  Parent waitpid's.
 *   3. Switch to B: worker already exited (== kill/reap done) -> TIME the unbind
 *      of A ("[unbind A] returned in <ms> ms" — should be fast since no process
 *      holds the fd; this is the make-or-break for per-worker) -> acquire B ->
 *      bind B -> fork WRITE worker for B -> waitpid.
 *   4. Switch BACK to A (READVERIFY) — the cross-process VRAM gate: unbind B ->
 *      acquire A -> bind A -> fork READVERIFY worker.  It re-allocs the same
 *      order, reads back WITHOUT uploading, and reports whether SEED_A's pattern
 *      + slow_add result survived A's worker exit and the B detour.
 *   5. Verdict (parent): [PASS] per-worker switch clean AND cross-proc VRAM
 *      persists, or [FAIL <which: unbind-hang | vram-lost | compute>].  _Exit().
 *
 * Reuses (verbatim where noted):
 *   - test_swap.c / gpu_switch_probe.c : discover_gk210s, read_bar0,
 *     find_acquired_bdf, set_driver_override, block_all_except, sysfs_write
 *   - compute_persist.c / gpu_switch_probe.c : the dlopen CUDA table + LOAD
 *     macro, load_nvidia, the slow_add PTX kernel
 *
 * On a host with no barswap device (this Mac) bs_init() returns NULL -> prints
 * [SKIP] and exits 0, so it compiles, links and runs everywhere. Shells out to
 * modprobe ONCE and to sysfs bind/unbind — must run as root on the K80 host. Both
 * parent and workers call _Exit() because the rig's CUDA atexit teardown hangs
 * after manual swaps.
 *
 * Usage (on a K80 host, as root):
 *   make worker-switch
 *   sudo ./barswap/tests/worker_switch <dieA> <dieB>   (default 0 2 — cross-card)
 *
 * @status Builds clean on Mac (SKIP path). Not yet run on hardware.
 * @issues None
 * @todo None
 */

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "barswap/barswap.h"

#define COLOR_GREEN  "\033[92m"
#define COLOR_RED    "\033[31m"
#define COLOR_YELLOW "\033[93m"
#define COLOR_RESET  "\033[0m"

#define MAX_DIES BS_MAX_DIES

/* ================================================================
 * GK210 enumeration + driver_override protocol (from test_swap.c /
 * gpu_switch_probe.c — verbatim ports).
 * ================================================================ */

/* BDF strings for all GK210 devices found in sysfs */
static char gk210_bdf[MAX_DIES][16];
static int n_gk210;

/*
 * Discover GK210 device BDFs from sysfs (vendor 0x10de, device 0x102d).
 * Verbatim port of test_swap.c:discover_gk210s().
 */
static int discover_gk210s(void)
{
    FILE *fp;
    char path[512], buf[16];
    DIR *dir;
    struct dirent *ent;
    char *bdfs[MAX_DIES];
    int count = 0;

    dir = opendir("/sys/bus/pci/devices");
    if (!dir)
        return 0;

    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.')
            continue;

        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", ent->d_name);
        fp = fopen(path, "r");
        if (!fp)
            continue;
        if (!fgets(buf, sizeof(buf), fp)) { fclose(fp); continue; }
        fclose(fp);
        if (strncmp(buf, "0x10de", 6) != 0)
            continue;

        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/device", ent->d_name);
        fp = fopen(path, "r");
        if (!fp)
            continue;
        if (!fgets(buf, sizeof(buf), fp)) { fclose(fp); continue; }
        fclose(fp);
        if (strncmp(buf, "0x102d", 6) != 0)
            continue;

        if (count < MAX_DIES) {
            bdfs[count] = strdup(ent->d_name);
            count++;
        }
    }
    closedir(dir);

    /* Sort by BDF name */
    for (int i = 0; i < count - 1; i++)
        for (int j = i + 1; j < count; j++)
            if (strcmp(bdfs[i], bdfs[j]) > 0) {
                char *swap_tmp = bdfs[i]; bdfs[i] = bdfs[j]; bdfs[j] = swap_tmp;
            }

    for (int i = 0; i < count; i++) {
        snprintf(gk210_bdf[i], sizeof(gk210_bdf[i]), "%s", bdfs[i]);
        free(bdfs[i]);
    }
    n_gk210 = count;
    return count;
}

/*
 * Read BAR 0 value from sysfs for a PCI device.
 * Returns the resource start address, or 0 if unreadable/unmapped.
 * Verbatim port of test_swap.c:read_bar0().
 */
static uint64_t read_bar0(const char *bdf)
{
    char path[512], buf[256];
    uint64_t start = 0;

    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource", bdf);
    FILE *fp = fopen(path, "r");
    if (!fp)
        return 0;

    /* First line of 'resource' file is BAR 0: start end flags */
    if (fgets(buf, sizeof(buf), fp))
        start = strtoull(buf, NULL, 16);
    fclose(fp);
    return start;
}

/*
 * Find which GK210 BDF currently has BARs programmed into the MMIO slot.
 * Only the acquired die has a non-zero BAR 0 (the rest are zeroed on release).
 * Mirrors gpu_switch_probe.c:find_acquired_bdf().
 * Returns the index into gk210_bdf[], or -1 if none found.
 */
static int find_acquired_bdf(void)
{
    for (int i = 0; i < n_gk210; i++) {
        if (read_bar0(gk210_bdf[i]) != 0)
            return i;
    }
    return -1;
}

/*
 * Write a NUL-terminated string to a sysfs attribute (O_WRONLY, write, close).
 * Returns 0 on success, -1 on error. The trailing newline expected by the PCI
 * bind/unbind attributes must be included by the caller.
 * Verbatim port of gpu_switch_probe.c:sysfs_write().
 */
static int sysfs_write(const char *path, const char *str)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) {
        fprintf(stderr, "  sysfs_write: open(%s) failed\n", path);
        return -1;
    }
    ssize_t want = (ssize_t)strlen(str);
    ssize_t got = write(fd, str, (size_t)want);
    close(fd);
    if (got != want) {
        fprintf(stderr, "  sysfs_write: write(%s, \"%s\") failed (%zd/%zd)\n",
                path, str, got, want);
        return -1;
    }
    return 0;
}

static void set_driver_override(const char *bdf, const char *value)
{
    char path[512];
    int fd;
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/driver_override", bdf);
    fd = open(path, O_WRONLY);
    if (fd < 0)
        return;
    if (*value == '\0') {
        /* Write newline to clear override — 0-byte write is ignored by VFS */
        if (write(fd, "\n", 1) < 0) { /* best-effort */ }
    } else {
        if (write(fd, value, strlen(value)) < 0) { /* best-effort */ }
    }
    close(fd);
}

/*
 * Block all GK210 devices except the one at bdf_index, so nvidia only binds the
 * acquired die. Verbatim port of test_swap.c:block_all_except().
 */
static void block_all_except(int bdf_index)
{
    for (int i = 0; i < n_gk210; i++) {
        if (i == bdf_index)
            set_driver_override(gk210_bdf[i], "");      /* clear — allow nvidia */
        else
            set_driver_override(gk210_bdf[i], "none");  /* block */
    }
}

static void unblock_all(void)
{
    for (int i = 0; i < n_gk210; i++)
        set_driver_override(gk210_bdf[i], "");
}

/*
 * Load nvidia ONCE for the whole run. Mirrors compute_persist.c:load_nvidia().
 * After this returns the module stays resident — it is never rmmod'd; switching
 * is done purely by PCI unbind/bind of the single active function.
 */
static int load_nvidia_once(void)
{
    int ret;
    /* Each module loaded separately with --ignore-install: "install nvidia
     * /bin/false" in modprobe.d otherwise blocks dependency resolution. */
    ret = system("modprobe --ignore-install nvidia 2>/dev/null");
    if (ret != 0)
        return -1;
    sleep(1);
    /* nvidia_uvm is required for cuInit to succeed */
    ret = system("modprobe --ignore-install nvidia_uvm 2>/dev/null");
    if (ret != 0)
        return -1;
    sleep(1);
    return 0;
}

/* ================================================================
 * CUDA driver API via dlopen (from compute_persist.c / gpu_switch_probe.c).
 * ================================================================ */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

typedef CUresult (*pfn_cuInit)(unsigned int);
typedef CUresult (*pfn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*pfn_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*pfn_cuCtxSynchronize)(void);
typedef CUresult (*pfn_cuMemAlloc)(CUdeviceptr *, size_t);
typedef CUresult (*pfn_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
typedef CUresult (*pfn_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
typedef CUresult (*pfn_cuModuleLoadData)(CUmodule *, const void *);
typedef CUresult (*pfn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult (*pfn_cuLaunchKernel)(CUfunction,
    unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, CUstream, void **, void **);

static void *libcuda = NULL;
static pfn_cuInit              p_cuInit;
static pfn_cuDeviceGet         p_cuDeviceGet;
static pfn_cuCtxCreate         p_cuCtxCreate;
static pfn_cuCtxSynchronize    p_cuCtxSynchronize;
static pfn_cuMemAlloc          p_cuMemAlloc;
static pfn_cuMemcpyHtoD        p_cuMemcpyHtoD;
static pfn_cuMemcpyDtoH        p_cuMemcpyDtoH;
static pfn_cuModuleLoadData    p_cuModuleLoadData;
static pfn_cuModuleGetFunction p_cuModuleGetFunction;
static pfn_cuLaunchKernel      p_cuLaunchKernel;

/*
 * Embedded PTX kernel — adds 1.0 to each element `iters` times.
 * Verbatim port of compute_persist.c:slow_add_ptx (entry "slow_add").
 * out[i] = in[i] + iters, exact in float32 while values stay under 2^24.
 */
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

static int load_cuda_symbols(void)
{
    libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!libcuda)
        libcuda = dlopen("libcuda.so", RTLD_LAZY);
    if (!libcuda) {
        fprintf(stderr, "  Cannot dlopen libcuda: %s\n", dlerror());
        return -1;
    }

    /* _v2 symbols where needed (legacy v1 ctx is incompatible). Mirrors
     * compute_persist.c:load_cuda_symbols(). */
    #define LOAD(name, sym) do { \
        p_##name = (pfn_##name)dlsym(libcuda, sym); \
        if (!p_##name) { \
            fprintf(stderr, "  dlsym(%s) failed: %s\n", sym, dlerror()); \
            return -1; \
        } \
    } while (0)

    LOAD(cuInit,              "cuInit");
    LOAD(cuDeviceGet,         "cuDeviceGet");
    LOAD(cuCtxCreate,         "cuCtxCreate_v2");
    LOAD(cuCtxSynchronize,    "cuCtxSynchronize");
    LOAD(cuMemAlloc,          "cuMemAlloc_v2");
    LOAD(cuMemcpyHtoD,        "cuMemcpyHtoD_v2");
    LOAD(cuMemcpyDtoH,        "cuMemcpyDtoH_v2");
    LOAD(cuModuleLoadData,    "cuModuleLoadData");
    LOAD(cuModuleGetFunction, "cuModuleGetFunction");
    LOAD(cuLaunchKernel,      "cuLaunchKernel");

    #undef LOAD
    return 0;
}

/* ================================================================
 * Test payload + timing
 * ================================================================ */

#define N_ELEMENTS  (1024 * 1024)            /* 1M floats = 4 MB per buffer */
#define N_ITERS     1000                     /* 1000 adds per element       */
#define BLOCK_SIZE  256

/* Worker modes. */
#define MODE_WRITE      0
#define MODE_READVERIFY 1

/* Per-die seeds so a stale read of the wrong die's VRAM cannot pass. */
#define SEED_A  0xA5A5A5A5u
#define SEED_B  0x5B5B5B5Bu

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Distinct per-die input pattern; values 0..4095 keep out=in+N_ITERS exact. */
static void fill_die_pattern(float *host_buf, uint32_t seed)
{
    for (int i = 0; i < N_ELEMENTS; i++) {
        uint32_t v = (seed + (uint32_t)i * 2654435761u) & 0xFFFu;
        host_buf[i] = (float)v;
    }
}

/* Host-side expected compute result for element i: in[i] + N_ITERS. */
static float expected_compute_at(uint32_t seed, int i)
{
    uint32_t v = (seed + (uint32_t)i * 2654435761u) & 0xFFFu;
    return (float)v + (float)N_ITERS;
}

/* ================================================================
 * Worker child — one die, attaches to the already-bound nvidia, then
 * WRITE or READVERIFY, then _Exit() (closing its /dev/nvidiaN fd, which
 * is what drives the device usage_count down — the whole reason for the
 * per-process design).
 * ================================================================ */

/*
 * worker_attach — bring up CUDA in the worker on the slot die. The parent has
 * already acquired/bound the die, so the worker only attaches: cuInit ->
 * cuDeviceGet(0) -> cuCtxCreate. Returns 0 and fills *ctx_out on success.
 */
static int worker_attach(CUcontext *ctx_out)
{
    CUresult err;
    CUdevice dev;
    CUcontext ctx = NULL;

    if (load_cuda_symbols() != 0)
        return -1;
    if ((err = p_cuInit(0)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [worker] cuInit failed: %d\n", err);
        return -1;
    }
    if ((err = p_cuDeviceGet(&dev, 0)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [worker] cuDeviceGet failed: %d\n", err);
        return -1;
    }
    if ((err = p_cuCtxCreate(&ctx, 0, dev)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [worker] cuCtxCreate failed: %d\n", err);
        return -1;
    }
    *ctx_out = ctx;
    return 0;
}

/*
 * worker_write — alloc two buffers in the FIXED canonical order (in, then out),
 * print the returned device pointers, upload the die's seed pattern, run
 * slow_add, sync, and self-verify out=in+N_ITERS. Returns 0 on success.
 *
 * Deliberately does NOT cuMemFree: we WANT to test that the physical VRAM bytes
 * survive this worker's process exit, even though exit frees them logically.
 */
static int worker_write(uint32_t seed, char tag)
{
    CUresult err;
    CUdeviceptr d_in = 0, d_out = 0;
    CUmodule mod;
    CUfunction kern;
    float *h_in = NULL, *h_check = NULL;
    int rc = -1;

    size_t buf_bytes = (size_t)N_ELEMENTS * sizeof(float);
    /* Canonical order: d_in first, then d_out — the READVERIFY worker must
     * cuMemAlloc in this exact order to land on the same physical VRAM. */
    if ((err = p_cuMemAlloc(&d_in, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c WRITE] cuMemAlloc(in) failed: %d\n", tag, err); goto done;
    }
    if ((err = p_cuMemAlloc(&d_out, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c WRITE] cuMemAlloc(out) failed: %d\n", tag, err); goto done;
    }
    printf("  [%c WRITE] device ptrs: d_in=0x%llx d_out=0x%llx\n",
           tag, (unsigned long long)d_in, (unsigned long long)d_out);

    h_in = (float *)malloc(buf_bytes);
    fill_die_pattern(h_in, seed);
    if ((err = p_cuMemcpyHtoD(d_in, h_in, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c WRITE] cuMemcpyHtoD failed: %d\n", tag, err); goto done;
    }

    if ((err = p_cuModuleLoadData(&mod, slow_add_ptx)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c WRITE] cuModuleLoadData failed: %d\n", tag, err); goto done;
    }
    if ((err = p_cuModuleGetFunction(&kern, mod, "slow_add")) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c WRITE] cuModuleGetFunction failed: %d\n", tag, err); goto done;
    }

    unsigned int n = N_ELEMENTS, iters = N_ITERS;
    unsigned int grid = (N_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *args[] = { &d_out, &d_in, &n, &iters };
    if ((err = p_cuLaunchKernel(kern, grid, 1, 1, BLOCK_SIZE, 1, 1,
                                0, NULL, args, NULL)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c WRITE] cuLaunchKernel failed: %d\n", tag, err); goto done;
    }
    if ((err = p_cuCtxSynchronize()) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c WRITE] cuCtxSynchronize failed: %d\n", tag, err); goto done;
    }

    h_check = (float *)malloc(buf_bytes);
    p_cuMemcpyDtoH(h_check, d_out, buf_bytes);
    int ok = 1;
    for (int i = 0; i < N_ELEMENTS; i++)
        if (h_check[i] != expected_compute_at(seed, i)) { ok = 0; break; }
    if (!ok) {
        fprintf(stderr, "  [%c WRITE] compute self-check FAILED\n", tag);
        goto done;
    }
    printf("  [%c WRITE] compute OK (seed 0x%08x, out=in+%d)\n", tag, seed, N_ITERS);
    rc = 0;

done:
    /* NOTE: deliberately NOT calling cuMemFree(d_in/d_out) — process exit frees
     * them logically; we are testing that the physical bytes survive that. */
    free(h_in);
    free(h_check);
    return rc;
}

/*
 * worker_readverify — the cross-process VRAM gate. Re-alloc the SAME two buffers
 * in the SAME canonical order (in, then out) and print the pointers (should
 * match the WRITE worker's), then read back WITHOUT writing/uploading and check
 * the seed pattern and slow_add result are still present. Returns 0 if the seed
 * survived this die's WRITE worker exit + the detour to the other die.
 */
static int worker_readverify(uint32_t seed, char tag)
{
    CUresult err;
    CUdeviceptr d_in = 0, d_out = 0;
    float *r_in = NULL, *r_out = NULL;
    int rc = -1;

    size_t buf_bytes = (size_t)N_ELEMENTS * sizeof(float);
    /* Same size + same order as worker_write so these land on the same physical
     * VRAM the WRITE worker used. */
    if ((err = p_cuMemAlloc(&d_in, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c READVERIFY] cuMemAlloc(in) failed: %d\n", tag, err); goto done;
    }
    if ((err = p_cuMemAlloc(&d_out, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c READVERIFY] cuMemAlloc(out) failed: %d\n", tag, err); goto done;
    }
    printf("  [%c READVERIFY] device ptrs: d_in=0x%llx d_out=0x%llx (should match WRITE)\n",
           tag, (unsigned long long)d_in, (unsigned long long)d_out);

    /* Read back WITHOUT any upload — d_in should still hold the seed pattern,
     * d_out should still hold the slow_add result from the WRITE worker. */
    r_in = (float *)malloc(buf_bytes);
    r_out = (float *)malloc(buf_bytes);
    memset(r_in, 0, buf_bytes);
    memset(r_out, 0, buf_bytes);
    p_cuMemcpyDtoH(r_in, d_in, buf_bytes);
    p_cuMemcpyDtoH(r_out, d_out, buf_bytes);

    int in_match = 0, out_match = 0;
    for (int i = 0; i < N_ELEMENTS; i++) {
        uint32_t v = (seed + (uint32_t)i * 2654435761u) & 0xFFFu;
        if (r_in[i] == (float)v)
            in_match++;
        if (r_out[i] == expected_compute_at(seed, i))
            out_match++;
    }

    int persisted = (in_match == N_ELEMENTS && out_match == N_ELEMENTS);
    printf("  [%c READVERIFY] seed pattern: %d/%d   slow_add result: %d/%d\n",
           tag, in_match, N_ELEMENTS, out_match, N_ELEMENTS);
    printf("  [%c READVERIFY] sample: d_in[0]=%.1f (want %.1f)  d_out[0]=%.1f (want %.1f)\n",
           tag, r_in[0], (float)((seed + 0u) & 0xFFFu),
           r_out[0], expected_compute_at(seed, 0));
    printf("  [xproc-vram] persisted across process exit: %s\n",
           persisted ? "YES" : "NO");
    rc = persisted ? 0 : -1;

done:
    /* No cuMemFree — exit handles it; keep the read-only contract simple. */
    free(r_in);
    free(r_out);
    return rc;
}

/*
 * run_worker — worker child entry. The parent has already bound nvidia to `die`
 * (`bdf` is for logging only). Attach CUDA, run the requested mode, and _Exit()
 * with 0 on pass / 1 on fail. The _Exit() (not return) both skips the CUDA
 * atexit teardown that hangs on this rig AND closes /dev/nvidiaN, dropping the
 * device usage_count — the event the parent relies on for a clean later unbind.
 */
static void run_worker(int die, const char *bdf, int mode, uint32_t seed, char tag)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const char *mode_name = (mode == MODE_WRITE) ? "WRITE" : "READVERIFY";
    printf("  [%c %s] worker pid %d attaching to die %d (%s)\n",
           tag, mode_name, (int)getpid(), die, bdf ? bdf : "?");

    CUcontext ctx = NULL;
    if (worker_attach(&ctx) != 0)
        _Exit(1);

    int rc = (mode == MODE_WRITE) ? worker_write(seed, tag)
                                  : worker_readverify(seed, tag);
    /* Intentionally no cuCtxDestroy: process exit closes the fd, which is the
     * usage_count drop we are validating. */
    _Exit(rc == 0 ? 0 : 1);
}

/* ================================================================
 * Parent (owner) — barswap + driver_override + bind/unbind + fork.
 * ================================================================ */

static bs_context *g_bs;
static int nvidia_loaded;          /* set after the single modprobe */

/*
 * owner_acquire_and_bind — map `die` into the MMIO slot, block the other dies,
 * and bind nvidia to it (modprobe-once on the first call, sysfs bind after).
 * The parent does NOT touch CUDA — that is the worker's job. Returns 0 and fills
 * *gpu_out / *bdf_idx_out.
 */
static int owner_acquire_and_bind(int die, bs_gpu **gpu_out, int *bdf_idx_out,
                                  char tag)
{
    bs_gpu *gpu = bs_acquire(g_bs, die);
    if (!gpu) {
        fprintf(stderr, "  [%c] bs_acquire(%d) failed\n", tag, die);
        return -1;
    }
    int idx = find_acquired_bdf();
    if (idx < 0) {
        fprintf(stderr, "  [%c] could not find acquired BDF for die %d\n", tag, die);
        bs_release(g_bs, gpu);
        return -1;
    }
    printf("  [%c] die %d -> GK210 %s in MMIO slot\n", tag, die, gk210_bdf[idx]);
    block_all_except(idx);

    if (!nvidia_loaded) {
        printf("  [%c] loading nvidia (first time only — probes the allowed die)\n", tag);
        if (load_nvidia_once() != 0) {
            fprintf(stderr, "  [%c] nvidia load failed\n", tag);
            bs_release(g_bs, gpu);
            return -1;
        }
        nvidia_loaded = 1;
    } else {
        char val[32];
        snprintf(val, sizeof(val), "%s\n", gk210_bdf[idx]);
        printf("  [%c] binding device %s\n", tag, gk210_bdf[idx]);
        if (sysfs_write("/sys/bus/pci/drivers/nvidia/bind", val) != 0)
            fprintf(stderr, "  [%c] warning: bind of %s failed\n", tag, gk210_bdf[idx]);
    }

    *gpu_out = gpu;
    *bdf_idx_out = idx;
    return 0;
}

/*
 * owner_unbind_observed — the critical observable. With the worker already
 * exited (its /dev/nvidiaN fd closed, so usage_count should be 0), timestamp,
 * write "<bdf>\n" to .../nvidia/unbind, and report elapsed time. If the unbind
 * hangs (the failure mode the per-worker design is meant to AVOID), the
 * "[unbind X] returned ..." line never prints and this becomes the last visible
 * output. Returns sysfs_write's result.
 */
static int owner_unbind_observed(int bdf_idx, char tag)
{
    char val[32];
    snprintf(val, sizeof(val), "%s\n", gk210_bdf[bdf_idx]);

    printf("  [unbind %c] worker already exited (fd closed) -> writing %s to "
           ".../nvidia/unbind (should be fast now)...\n", tag, gk210_bdf[bdf_idx]);
    double t0 = now_ms();
    int rc = sysfs_write("/sys/bus/pci/drivers/nvidia/unbind", val);
    double elapsed = now_ms() - t0;
    printf("  [unbind %c] returned in %.1f ms (rc=%d)\n", tag, elapsed, rc);
    return rc;
}

/*
 * owner_fork_worker — fork a worker child for the bound die, wait for it, and
 * return its result (0 = pass). The worker's fd-close on exit is the usage_count
 * drop the next unbind depends on.
 */
static int owner_fork_worker(int die, int bdf_idx, int mode, uint32_t seed,
                             char tag)
{
    const char *bdf = (bdf_idx >= 0) ? gk210_bdf[bdf_idx] : NULL;
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "  [%c] fork failed\n", tag);
        return -1;
    }
    if (pid == 0)
        run_worker(die, bdf, mode, seed, tag);   /* never returns */

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        fprintf(stderr, "  [%c] waitpid failed\n", tag);
        return -1;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "  [%c] worker pid %d did not exit cleanly (status 0x%x)\n",
                tag, (int)pid, status);
        return -1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    /* Unbuffered so a hanging unbind still shows its "writing..." line. */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int die_a = (argc > 1) ? atoi(argv[1]) : 0;
    int die_b = (argc > 2) ? atoi(argv[2]) : 2;

    printf("=== Worker Switch — per-die worker processes (the SETTLED owner shape) ===\n\n");
    printf("Validates: (1) killing/exiting the per-die WORKER drops the nvidia\n"
           "usage_count so the parent's unbind returns FAST/CLEAN, and (2) VRAM\n"
           "PHYSICALLY PERSISTS across a worker's process exit so the next worker\n"
           "re-reads weights WITHOUT re-uploading (re-upload = ~40 s/16 GB).\n\n");
    printf("die A = %d, die B = %d (cross-card by default)\n\n", die_a, die_b);

    /* Gate on barswap availability so this compiles + runs everywhere. */
    g_bs = bs_init();
    if (!g_bs) {
        printf("[SKIP] no barswap (bs_init failed — not on a K80 host)\n");
        return 0;
    }

    if (getuid() != 0) {
        fprintf(stderr, "error: must run as root (sudo) — needs modprobe + sysfs\n");
        bs_free(g_bs);
        return 1;
    }

    discover_gk210s();
    printf("barswap: %u dies available, %d GK210s in sysfs\n\n",
           bs_num_dies(g_bs), n_gk210);

    const char *fail_where = NULL;
    int unbind_a_ok = 0, b_write_ok = 0, a_persisted = 0;

    bs_gpu *gpu_a = NULL, *gpu_b = NULL;
    int bdf_a = -1, bdf_b = -1;

    /* ---- Step 2: Switch-to(A, WRITE) ---- */
    printf(COLOR_YELLOW "[Step 2] Switch to die A (%d) — fork WRITE worker"
           COLOR_RESET "\n", die_a);
    if (owner_acquire_and_bind(die_a, &gpu_a, &bdf_a, 'A') != 0) {
        fail_where = "acquire/bind A"; goto verdict;
    }
    if (owner_fork_worker(die_a, bdf_a, MODE_WRITE, SEED_A, 'A') != 0) {
        fail_where = "worker A WRITE (compute)"; goto verdict;
    }
    printf(COLOR_GREEN "  [A] WRITE worker exited clean" COLOR_RESET "\n\n");

    /* ---- Step 3: Switch to die B — TIME the post-worker unbind of A ---- */
    printf(COLOR_YELLOW "[Step 3] Switch to die B (%d) — unbind A (make-or-break), "
           "then fork B WRITE worker" COLOR_RESET "\n", die_b);
    unbind_a_ok = (owner_unbind_observed(bdf_a, 'A') == 0);
    bs_release(g_bs, gpu_a);
    gpu_a = NULL;
    if (!unbind_a_ok) {
        fail_where = "unbind A (HUNG — usage_count not dropped)"; goto verdict;
    }
    if (owner_acquire_and_bind(die_b, &gpu_b, &bdf_b, 'B') != 0) {
        fail_where = "acquire/bind B"; goto verdict;
    }
    if (owner_fork_worker(die_b, bdf_b, MODE_WRITE, SEED_B, 'B') != 0) {
        fail_where = "worker B WRITE (compute)"; goto verdict;
    }
    b_write_ok = 1;
    printf(COLOR_GREEN "  [B] WRITE worker exited clean" COLOR_RESET "\n\n");

    /* ---- Step 4: Switch BACK to die A (READVERIFY) — the cross-proc gate ---- */
    printf(COLOR_YELLOW "[Step 4] Switch back to die A — fork READVERIFY worker "
           "(cross-process VRAM gate)" COLOR_RESET "\n");
    if (owner_unbind_observed(bdf_b, 'B') != 0)
        fprintf(stderr, "  warning: unbind B did not return cleanly\n");
    bs_release(g_bs, gpu_b);
    gpu_b = NULL;
    if (owner_acquire_and_bind(die_a, &gpu_a, &bdf_a, 'A') != 0) {
        fail_where = "re-acquire/bind A"; goto verdict;
    }
    a_persisted = (owner_fork_worker(die_a, bdf_a, MODE_READVERIFY, SEED_A, 'A') == 0);
    if (!a_persisted) {
        fail_where = "vram-lost (A's seed did not survive process exit + detour)";
        goto verdict;
    }
    printf(COLOR_GREEN "  [A] READVERIFY worker confirmed VRAM persisted"
           COLOR_RESET "\n\n");

verdict:
    /* Best-effort teardown (nvidia stays resident; parent never held a CUDA fd). */
    if (gpu_a) bs_release(g_bs, gpu_a);
    if (gpu_b) bs_release(g_bs, gpu_b);
    unblock_all();

    printf("=== Verdict ===\n");
    if (!fail_where && unbind_a_ok && b_write_ok && a_persisted) {
        printf(COLOR_GREEN
               "[PASS] per-worker switch clean (unbinds fast) AND cross-proc "
               "VRAM persists" COLOR_RESET "\n");
        bs_free(g_bs);
        _Exit(0);
    }

    /* Classify the failure: unbind-hang | vram-lost | compute. */
    const char *which;
    if (!unbind_a_ok)
        which = "unbind-hang";
    else if (!b_write_ok)
        which = "compute";
    else if (!a_persisted)
        which = "vram-lost";
    else
        which = "compute";
    printf(COLOR_RED "[FAIL] %s — %s" COLOR_RESET "\n",
           which, fail_where ? fail_where : "unknown");
    printf("  unbind A fast/clean: %s\n", unbind_a_ok ? "yes" : "no (HUNG or errored)");
    printf("  B WRITE worker:      %s\n", b_write_ok ? "yes" : "no");
    printf("  A VRAM persisted:    %s\n", a_persisted ? "yes" : "no");
    bs_free(g_bs);
    _Exit(1);
}
