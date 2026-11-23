/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#define _GNU_SOURCE

/*
 * gpu_switch_probe.c — Does a FULL in-process CUDA release let a PCI
 *                      unbind/rebind switch between K80 dies succeed cleanly?
 *
 * THE ONE QUESTION this probe answers (it gates Pantheon's GPU-owner design):
 *
 *   When we switch nvidia from die A to die B by PCI unbind/bind on the barswap
 *   rig, the unbind HANGS in an nvidia_uvm ioctl with a "non-zero usage count"
 *   because libcuda / nvidia_uvm still hold /dev/nvidia0 open. cuCtxDestroy
 *   ALONE is not enough — roundrobin_persist.c did exactly that (destroy ctx,
 *   then unbind) and left buffers resident and skipped the device reset.
 *
 *   The UNTESTED lever: do a COMPLETE release before the unbind —
 *       free all device memory  -> cuMemFree(d_in), cuMemFree(d_out)
 *       drain the device        -> cuCtxSynchronize
 *       destroy the context     -> cuCtxDestroy(ctx)
 *       drop the primary context-> cuDevicePrimaryCtxReset(dev)   <-- the lever
 *   ...and only THEN write the BDF to .../nvidia/unbind. The hypothesis is that
 *   cuDevicePrimaryCtxReset drives the device usage count to zero so the unbind
 *   returns cleanly instead of hanging.
 *
 * The whole point is OBSERVABILITY around the unbind: every step prints loudly
 * with timing on UNBUFFERED stdout, so the operator can see exactly where it
 * hangs — and, critically, whether "[unbind A] returned in <ms> ms" ever prints.
 * If the unbind hangs, that line will be the last thing on screen; if it
 * returns, we know the lever works and we proceed to bind + compute on B.
 *
 * Sequence:
 *   1. bs_init + discover_gk210s
 *   2. ACQUIRE A: acquire -> find bdf -> block others -> load nvidia (once) ->
 *      cuInit -> cuCtxCreate -> alloc + upload pattern -> slow_add -> sync ->
 *      verify  => "[A] compute OK"
 *   3. FULL RELEASE of A (the lever): cuMemFree x2 -> cuCtxSynchronize ->
 *      cuCtxDestroy -> cuDevicePrimaryCtxReset, printing each CUresult
 *   4. UNBIND A (the critical observable): timestamp, write BDF to unbind,
 *      print elapsed => "[unbind A] returned in <ms> ms" (or it hangs here)
 *   5. ACQUIRE + BIND B: acquire -> find bdf -> block others -> bind ->
 *      cuCtxCreate -> alloc + upload + slow_add + sync + verify => "[B] compute OK"
 *   6. (bonus) SWITCH BACK TO A: full-release B -> unbind B -> acquire A ->
 *      bind A -> re-alloc SAME size/order -> read back WITHOUT writing ->
 *      check A's original pattern persisted => "[A re-read] vram persisted: yes/no"
 *   7. Verdict: [PASS] / [FAIL <where>], then _Exit().
 *
 * Reuses (verbatim where noted):
 *   - test_swap.c / roundrobin_persist.c : discover_gk210s, read_bar0,
 *     find_acquired_bdf, set_driver_override, block_all_except, sysfs_write
 *   - compute_persist.c : the dlopen CUDA scaffolding + the slow_add PTX kernel
 *
 * On a host with no barswap device (this Mac) bs_init() returns NULL -> prints
 * [SKIP] and exits 0, so it compiles, links and runs everywhere. Shells out to
 * modprobe ONCE and to sysfs bind/unbind — must run as root on the K80 host. Calls
 * _Exit() at the end because the rig's CUDA atexit teardown hangs after manual
 * swaps.
 *
 * Usage (on a K80 host, as root):
 *   make gpu-switch-probe
 *   sudo ./barswap/tests/gpu_switch_probe <dieA> <dieB>   (default 0 2)
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
 * roundrobin_persist.c — verbatim ports).
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
                char *tmp = bdfs[i]; bdfs[i] = bdfs[j]; bdfs[j] = tmp;
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
 * Mirrors roundrobin_persist.c:find_acquired_bdf() (keyed on "non-zero" since
 * the barswap lib's bs_gpu does not expose bar0_addr).
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
 * Verbatim port of roundrobin_persist.c:sysfs_write().
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
 * CUDA driver API via dlopen (from compute_persist.c).
 * cuDevicePrimaryCtxReset is the lever this probe adds to the table.
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
typedef CUresult (*pfn_cuCtxDestroy)(CUcontext);
typedef CUresult (*pfn_cuCtxSynchronize)(void);
typedef CUresult (*pfn_cuDevicePrimaryCtxReset)(CUdevice);
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

static void *libcuda = NULL;
static pfn_cuInit                  p_cuInit;
static pfn_cuDeviceGet             p_cuDeviceGet;
static pfn_cuCtxCreate            p_cuCtxCreate;
static pfn_cuCtxDestroy           p_cuCtxDestroy;
static pfn_cuCtxSynchronize       p_cuCtxSynchronize;
static pfn_cuDevicePrimaryCtxReset p_cuDevicePrimaryCtxReset;
static pfn_cuMemAlloc             p_cuMemAlloc;
static pfn_cuMemFree              p_cuMemFree;
static pfn_cuMemcpyHtoD          p_cuMemcpyHtoD;
static pfn_cuMemcpyDtoH          p_cuMemcpyDtoH;
static pfn_cuModuleLoadData       p_cuModuleLoadData;
static pfn_cuModuleGetFunction    p_cuModuleGetFunction;
static pfn_cuLaunchKernel         p_cuLaunchKernel;

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

    LOAD(cuInit,                  "cuInit");
    LOAD(cuDeviceGet,             "cuDeviceGet");
    LOAD(cuCtxCreate,             "cuCtxCreate_v2");
    LOAD(cuCtxDestroy,            "cuCtxDestroy_v2");
    LOAD(cuCtxSynchronize,        "cuCtxSynchronize");
    LOAD(cuDevicePrimaryCtxReset, "cuDevicePrimaryCtxReset_v2");
    LOAD(cuMemAlloc,              "cuMemAlloc_v2");
    LOAD(cuMemFree,               "cuMemFree_v2");
    LOAD(cuMemcpyHtoD,            "cuMemcpyHtoD_v2");
    LOAD(cuMemcpyDtoH,            "cuMemcpyDtoH_v2");
    LOAD(cuModuleLoadData,        "cuModuleLoadData");
    LOAD(cuModuleGetFunction,     "cuModuleGetFunction");
    LOAD(cuLaunchKernel,          "cuLaunchKernel");

    #undef LOAD
    return 0;
}

/* ================================================================
 * Test payload + timing
 * ================================================================ */

#define N_ELEMENTS  (1024 * 1024)            /* 1M floats = 4 MB per buffer */
#define N_ITERS     1000                     /* 1000 adds per element       */
#define BLOCK_SIZE  256

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
 * Shared probe state
 * ================================================================ */

static bs_context *g_bs;
static int nvidia_loaded;          /* set after the single modprobe */

/* Per-die seeds so a stale read of the wrong die's VRAM cannot pass. */
#define SEED_A  0xA5A5A5A5u
#define SEED_B  0x5B5B5B5Bu

/*
 * acquire_and_bind — map `die` into the MMIO slot, point nvidia at it, and bring
 * up a fresh CUDA context on device 0 (== the slot die).
 *
 * On the first call nvidia is modprobe'd once (which probes + binds the only
 * allowed device); on later calls the device is PCI-bound via sysfs. Returns 0
 * on success and fills *gpu_out, *ctx_out, *dev_out, *bdf_idx_out.
 */
static int acquire_and_bind(int die, bs_gpu **gpu_out, CUcontext *ctx_out,
                            CUdevice *dev_out, int *bdf_idx_out, char tag)
{
    CUresult err;
    CUdevice dev;
    CUcontext ctx = NULL;

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

    if ((err = p_cuInit(0)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c] cuInit failed: %d\n", tag, err);
        bs_release(g_bs, gpu);
        return -1;
    }
    if ((err = p_cuDeviceGet(&dev, 0)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c] cuDeviceGet failed: %d\n", tag, err);
        bs_release(g_bs, gpu);
        return -1;
    }
    if ((err = p_cuCtxCreate(&ctx, 0, dev)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c] cuCtxCreate failed: %d\n", tag, err);
        bs_release(g_bs, gpu);
        return -1;
    }

    *gpu_out = gpu;
    *ctx_out = ctx;
    *dev_out = dev;
    *bdf_idx_out = idx;
    return 0;
}

/*
 * run_compute — alloc d_in/d_out, upload the die's pattern, launch slow_add,
 * sync, and self-verify out=in+N_ITERS. Leaves the two device pointers in
 * *d_in_out / *d_out_out for the caller's full-release step. Returns 0 if the
 * compute was correct.
 */
static int run_compute(uint32_t seed, CUdeviceptr *d_in_out, CUdeviceptr *d_out_out,
                       char tag)
{
    CUresult err;
    CUdeviceptr d_in = 0, d_out = 0;
    CUmodule mod;
    CUfunction kern;
    float *h_in = NULL, *h_check = NULL;
    int rc = -1;

    size_t buf_bytes = (size_t)N_ELEMENTS * sizeof(float);
    if ((err = p_cuMemAlloc(&d_in, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c] cuMemAlloc(in) failed: %d\n", tag, err); goto done;
    }
    if ((err = p_cuMemAlloc(&d_out, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c] cuMemAlloc(out) failed: %d\n", tag, err); goto done;
    }

    h_in = (float *)malloc(buf_bytes);
    fill_die_pattern(h_in, seed);
    if ((err = p_cuMemcpyHtoD(d_in, h_in, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c] cuMemcpyHtoD failed: %d\n", tag, err); goto done;
    }

    if ((err = p_cuModuleLoadData(&mod, slow_add_ptx)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c] cuModuleLoadData failed: %d\n", tag, err); goto done;
    }
    if ((err = p_cuModuleGetFunction(&kern, mod, "slow_add")) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c] cuModuleGetFunction failed: %d\n", tag, err); goto done;
    }

    unsigned int n = N_ELEMENTS, iters = N_ITERS;
    unsigned int grid = (N_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *args[] = { &d_out, &d_in, &n, &iters };
    if ((err = p_cuLaunchKernel(kern, grid, 1, 1, BLOCK_SIZE, 1, 1,
                                0, NULL, args, NULL)) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c] cuLaunchKernel failed: %d\n", tag, err); goto done;
    }
    if ((err = p_cuCtxSynchronize()) != CUDA_SUCCESS) {
        fprintf(stderr, "  [%c] cuCtxSynchronize failed: %d\n", tag, err); goto done;
    }

    h_check = (float *)malloc(buf_bytes);
    p_cuMemcpyDtoH(h_check, d_out, buf_bytes);
    int ok = 1;
    for (int i = 0; i < N_ELEMENTS; i++)
        if (h_check[i] != expected_compute_at(seed, i)) { ok = 0; break; }
    if (!ok) {
        fprintf(stderr, "  [%c] compute self-check FAILED\n", tag);
        goto done;
    }
    rc = 0;

done:
    free(h_in);
    free(h_check);
    *d_in_out = d_in;
    *d_out_out = d_out;
    return rc;
}

/*
 * full_release — THE LEVER. Free all device memory, drain the device, destroy
 * the context, then drop the primary context with cuDevicePrimaryCtxReset.
 * Prints the CUresult of every call. This is the complete in-process CUDA
 * release we believe is required to drive the device usage count to zero so a
 * subsequent PCI unbind returns instead of hanging.
 */
static void full_release(CUcontext ctx, CUdevice dev, CUdeviceptr d_in,
                         CUdeviceptr d_out, char tag)
{
    CUresult err;

    printf("  [%c] FULL RELEASE (the lever):\n", tag);
    err = p_cuMemFree(d_in);
    printf("      cuMemFree(d_in)            -> %d\n", err);
    err = p_cuMemFree(d_out);
    printf("      cuMemFree(d_out)           -> %d\n", err);
    err = p_cuCtxSynchronize();
    printf("      cuCtxSynchronize           -> %d\n", err);
    err = p_cuCtxDestroy(ctx);
    printf("      cuCtxDestroy               -> %d\n", err);
    err = p_cuDevicePrimaryCtxReset(dev);
    printf("      cuDevicePrimaryCtxReset    -> %d\n", err);
}

/*
 * unbind_observed — the critical observable. Timestamp, write "<bdf>\n" to
 * .../nvidia/unbind, and report the elapsed time. If the unbind hangs (the
 * failure mode we are probing), the "[unbind X] returned ..." line never prints
 * and this becomes the last visible output. Returns sysfs_write's result.
 */
static int unbind_observed(int bdf_idx, char tag)
{
    char val[32];
    snprintf(val, sizeof(val), "%s\n", gk210_bdf[bdf_idx]);

    printf("  [unbind %c] writing %s to .../nvidia/unbind (may hang here)...\n",
           tag, gk210_bdf[bdf_idx]);
    double t0 = now_ms();
    int rc = sysfs_write("/sys/bus/pci/drivers/nvidia/unbind", val);
    double elapsed = now_ms() - t0;
    printf("  [unbind %c] returned in %.1f ms (rc=%d)\n", tag, elapsed, rc);
    return rc;
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char *argv[])
{
    /* Unbuffered so the [unbind ...] line is on screen even if it never returns. */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int die_a = (argc > 1) ? atoi(argv[1]) : 0;
    int die_b = (argc > 2) ? atoi(argv[2]) : 2;

    printf("=== GPU Switch Probe — full-release unbind/rebind ===\n\n");
    printf("Question: does a FULL in-process CUDA release (ending in\n"
           "cuDevicePrimaryCtxReset) let the nvidia PCI unbind return cleanly\n"
           "so we can switch dies — instead of hanging on a non-zero usage count?\n\n");
    printf("die A = %d, die B = %d\n\n", die_a, die_b);

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
    if (load_cuda_symbols() != 0) {
        bs_free(g_bs);
        return 1;
    }
    printf("barswap: %u dies available, %d GK210s in sysfs\n\n",
           bs_num_dies(g_bs), n_gk210);

    const char *fail_where = NULL;
    int unbind_a_ok = 0, b_compute_ok = 0, a_persisted = 0;

    bs_gpu *gpu_a = NULL, *gpu_b = NULL;
    CUcontext ctx_a = NULL, ctx_b = NULL;
    CUdevice dev_a = 0, dev_b = 0;
    int bdf_a = -1, bdf_b = -1;
    CUdeviceptr a_in = 0, a_out = 0, b_in = 0, b_out = 0;

    /* ---- Step 2: ACQUIRE A + compute ---- */
    printf(COLOR_YELLOW "[Step 2] Acquire die A (%d) and compute" COLOR_RESET "\n", die_a);
    if (acquire_and_bind(die_a, &gpu_a, &ctx_a, &dev_a, &bdf_a, 'A') != 0) {
        fail_where = "acquire A"; goto verdict;
    }
    if (run_compute(SEED_A, &a_in, &a_out, 'A') != 0) {
        fail_where = "compute A"; goto verdict;
    }
    printf(COLOR_GREEN "  [A] compute OK" COLOR_RESET "\n\n");

    /* ---- Step 3: FULL RELEASE of A (the lever) ---- */
    printf(COLOR_YELLOW "[Step 3] Full release of die A" COLOR_RESET "\n");
    full_release(ctx_a, dev_a, a_in, a_out, 'A');
    ctx_a = NULL;
    printf("\n");

    /* ---- Step 4: UNBIND A — the critical observable ---- */
    printf(COLOR_YELLOW "[Step 4] Unbind die A (the make-or-break)" COLOR_RESET "\n");
    unbind_a_ok = (unbind_observed(bdf_a, 'A') == 0);
    /* Drop the slot mapping now that A's PCI function is (hopefully) unbound. */
    bs_release(g_bs, gpu_a);
    gpu_a = NULL;
    if (!unbind_a_ok) {
        fail_where = "unbind A"; goto verdict;
    }
    printf("\n");

    /* ---- Step 5: ACQUIRE + BIND B + compute ---- */
    printf(COLOR_YELLOW "[Step 5] Acquire + bind die B (%d) and compute" COLOR_RESET "\n",
           die_b);
    if (acquire_and_bind(die_b, &gpu_b, &ctx_b, &dev_b, &bdf_b, 'B') != 0) {
        fail_where = "acquire/bind B"; goto verdict;
    }
    if (run_compute(SEED_B, &b_in, &b_out, 'B') != 0) {
        fail_where = "compute B"; goto verdict;
    }
    b_compute_ok = 1;
    printf(COLOR_GREEN "  [B] compute OK" COLOR_RESET "\n\n");

    /* ---- Step 6 (bonus): SWITCH BACK TO A, check VRAM persisted ---- */
    printf(COLOR_YELLOW "[Step 6] Switch back to die A — check A's VRAM persisted"
           COLOR_RESET "\n");
    full_release(ctx_b, dev_b, b_in, b_out, 'B');
    ctx_b = NULL;
    if (unbind_observed(bdf_b, 'B') != 0)
        fprintf(stderr, "  warning: unbind B did not return cleanly\n");
    bs_release(g_bs, gpu_b);
    gpu_b = NULL;

    if (acquire_and_bind(die_a, &gpu_a, &ctx_a, &dev_a, &bdf_a, 'A') != 0) {
        fail_where = "re-acquire A"; goto verdict;
    }

    /* Re-alloc SAME size/order as Step 2 so buffers land on the same physical
     * VRAM, then read back WITHOUT writing and check A's original pattern. */
    size_t buf_bytes = (size_t)N_ELEMENTS * sizeof(float);
    CUdeviceptr ra_in = 0, ra_out = 0;
    p_cuMemAlloc(&ra_in, buf_bytes);
    p_cuMemAlloc(&ra_out, buf_bytes);
    float *r_out = (float *)malloc(buf_bytes);
    memset(r_out, 0, buf_bytes);
    p_cuMemcpyDtoH(r_out, ra_out, buf_bytes);
    int out_match = 0;
    for (int i = 0; i < N_ELEMENTS; i++)
        if (r_out[i] == expected_compute_at(SEED_A, i))
            out_match++;
    free(r_out);
    p_cuMemFree(ra_in);
    p_cuMemFree(ra_out);
    a_persisted = (out_match == N_ELEMENTS);
    printf("  [A re-read] %d/%d elements matched A's original result\n",
           out_match, N_ELEMENTS);
    printf("  [A re-read] vram persisted: %s\n", a_persisted ? "yes" : "no");
    printf("\n");

verdict:
    /* Best-effort teardown of whatever is still live (nvidia stays resident). */
    if (ctx_a) p_cuCtxDestroy(ctx_a);
    if (ctx_b) p_cuCtxDestroy(ctx_b);
    if (gpu_a) bs_release(g_bs, gpu_a);
    if (gpu_b) bs_release(g_bs, gpu_b);
    unblock_all();

    printf("=== Verdict ===\n");
    if (!fail_where && unbind_a_ok && b_compute_ok) {
        printf(COLOR_GREEN
               "[PASS] in-process clean switch works "
               "(unbind A returned, B computed, A persisted: %s)"
               COLOR_RESET "\n", a_persisted ? "yes" : "no");
        bs_free(g_bs);
        _Exit(0);
    }

    printf(COLOR_RED "[FAIL] %s" COLOR_RESET "\n",
           fail_where ? fail_where : "unknown");
    printf("  unbind A returned: %s\n", unbind_a_ok ? "yes" : "no (HUNG or errored)");
    printf("  B computed:        %s\n", b_compute_ok ? "yes" : "no");
    bs_free(g_bs);
    _Exit(1);
}
