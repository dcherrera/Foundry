/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#define _GNU_SOURCE

/*
 * xcard_persist.c — Cross-card VRAM + compute persistence test for a multi-K80 host
 *
 * Confirms that VRAM and a compute result written to one die survive while
 * OTHER dies (on OTHER physical K80 cards) are each loaded and unloaded in
 * between. This is the persistence property the barswap MoE design relies on:
 * write expert weights to each die in turn, then come back and use them — even
 * though the nvidia driver was torn down and rebuilt around other dies in the
 * meantime.
 *
 * This uses the PROVEN swap protocol — one die mapped into the single MMIO
 * slot at a time, nvidia bound to only that die — NOT persistent simultaneous
 * contexts (which has been shown not to work on this rig).
 *
 * Per-die protocol mirrors test_swap.c + compute_persist.c:
 *   bs_acquire(die)  → program BARs into the MMIO slot (barswap lib)
 *   block_all_except → driver_override "none" on every other GK210, clear the
 *                      acquired one, so nvidia binds only the acquired die
 *   load_nvidia      → modprobe --ignore-install nvidia{,_uvm,...}
 *   CUDA work        → cuInit/ctx/alloc/HtoD/launch/sync
 *   cuCtxDestroy + unload_nvidia + unblock_all + bs_release
 *
 * Persistence check mirrors vram_persist.c (lines 270-315): after the driver
 * has been reloaded, we allocate a fresh buffer of the SAME size and read it
 * back. The CUDA allocator resets across a driver reload, so we cannot read the
 * old CUdeviceptr; instead we rely on the fresh allocation landing on the same
 * physical VRAM and check whether our pattern (and the compute result) is still
 * there. Full pattern match → VRAM persisted; all zero → wiped.
 *
 * Usage (on a K80 host, as root):
 *   make xcard-persist
 *   sudo ./barswap/tests/xcard_persist <die0> <die1> [die2 ...]
 *   e.g. sudo ./barswap/tests/xcard_persist 0 2 4   (one die per card A/B/C)
 *
 * On a host with no barswap device (e.g. the Mac) it prints [SKIP] and exits 0,
 * so it compiles, links, and runs everywhere.
 *
 * NOTE: shells out to modprobe/rmmod — must run as root. Calls _Exit() at the
 * end because the rig's CUDA atexit teardown hangs after manual swaps.
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
#include <unistd.h>

#include "barswap/barswap.h"

#define COLOR_GREEN  "\033[92m"
#define COLOR_RED    "\033[31m"
#define COLOR_YELLOW "\033[93m"
#define COLOR_RESET  "\033[0m"

#define MAX_DIES BS_MAX_DIES

/* ================================================================
 * GK210 enumeration + driver_override protocol (from test_swap.c)
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
 * Mirrors test_swap.c:find_acquired_bdf(), but keyed on "non-zero" rather than
 * a fixed expected address, since the barswap lib's bs_gpu does not expose
 * bar0_addr — only the acquired die has its slot BAR programmed.
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
 * Block all GK210 devices except the one at bdf_index, so nvidia only binds
 * the acquired die. Verbatim port of test_swap.c:block_all_except().
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

static int load_nvidia(void)
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

static int unload_nvidia(void)
{
    system("rmmod nvidia_uvm 2>/dev/null");
    system("rmmod nvidia_drm 2>/dev/null");
    system("rmmod nvidia_modeset 2>/dev/null");
    int ret = system("rmmod nvidia 2>/dev/null");
    sleep(1);
    return ret == 0 ? 0 : -1;
}

/* ================================================================
 * CUDA driver API via dlopen (from vram_persist.c + compute_persist.c)
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
static pfn_cuInit              p_cuInit;
static pfn_cuDeviceGet         p_cuDeviceGet;
static pfn_cuCtxCreate         p_cuCtxCreate;
static pfn_cuCtxDestroy        p_cuCtxDestroy;
static pfn_cuCtxSynchronize    p_cuCtxSynchronize;
static pfn_cuMemAlloc          p_cuMemAlloc;
static pfn_cuMemFree           p_cuMemFree;
static pfn_cuMemcpyHtoD        p_cuMemcpyHtoD;
static pfn_cuMemcpyDtoH        p_cuMemcpyDtoH;
static pfn_cuModuleLoadData    p_cuModuleLoadData;
static pfn_cuModuleGetFunction p_cuModuleGetFunction;
static pfn_cuLaunchKernel      p_cuLaunchKernel;

/*
 * Embedded PTX kernel — adds 1.0 to each element `iters` times.
 * Mirrors compute_persist.c:slow_add_ptx. With input = base pattern and
 * iters small, output is deterministic and exact in float32.
 */
static const char *add_iters_ptx =
    ".version 5.0\n"
    ".target sm_37\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry add_iters(\n"
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
    LOAD(cuCtxDestroy,        "cuCtxDestroy_v2");
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

static void close_cuda(void)
{
    if (libcuda) { dlclose(libcuda); libcuda = NULL; }
}

/* ================================================================
 * Per-die test state + parameters
 * ================================================================ */

#define N_ELEMENTS  (1024 * 1024)            /* 1M floats = 4 MB per buffer */
#define N_ITERS     1000                     /* 1000 adds per element       */
#define BLOCK_SIZE  256

/*
 * Per-die expectation, computed in Phase 1 and re-checked in Phase 2.
 * Each die gets a DISTINCT base pattern so a stale read of the wrong die's
 * VRAM cannot masquerade as a pass.
 */
typedef struct die_expectation {
    int die_id;
    float base_value;        /* distinct per-die input fill */
    float expected_compute;  /* base_value + N_ITERS (exact in f32)         */
    int phase1_ok;           /* Phase 1 wrote + computed successfully       */
    int vram_pass;           /* Phase 2: input pattern survived             */
    int compute_pass;        /* Phase 2: compute result survived            */
} die_expectation;

/*
 * Build a distinct, die-specific input pattern in a host buffer.
 * Pattern is base_value in every element — small integers chosen so
 * base_value + N_ITERS stays exact in float32 (well under 2^24).
 */
static void fill_die_pattern(float *host_buf, float base_value)
{
    for (int i = 0; i < N_ELEMENTS; i++)
        host_buf[i] = base_value;
}

/*
 * Acquire one die via the proven protocol and load nvidia onto it only.
 * Returns the acquired bs_gpu (and sets *out_bdf_idx), or NULL on failure.
 * On failure the caller does not need to clean up; this unwinds itself.
 */
static bs_gpu *acquire_and_load(bs_context *bs, int die_id, int *out_bdf_idx)
{
    bs_gpu *gpu = bs_acquire(bs, die_id);
    if (!gpu) {
        fprintf(stderr, "  bs_acquire(%d) failed\n", die_id);
        return NULL;
    }

    int bdf_idx = find_acquired_bdf();
    if (bdf_idx < 0) {
        fprintf(stderr, "  could not find acquired BDF in sysfs for die %d\n", die_id);
        bs_release(bs, gpu);
        return NULL;
    }
    printf("  die %d acquired -> GK210 %s, BAR1=0x%llx\n",
           die_id, gk210_bdf[bdf_idx], (unsigned long long)gpu->bar1_addr);

    block_all_except(bdf_idx);

    if (load_nvidia() != 0) {
        fprintf(stderr, "  nvidia load failed for die %d\n", die_id);
        unload_nvidia();
        unblock_all();
        bs_release(bs, gpu);
        return NULL;
    }

    *out_bdf_idx = bdf_idx;
    return gpu;
}

/*
 * Tear down a die: destroy CUDA ctx, unload nvidia, unblock overrides, release.
 * ctx may be NULL (nothing to destroy).
 */
static void teardown_die(bs_context *bs, bs_gpu *gpu, CUcontext ctx)
{
    if (ctx)
        p_cuCtxDestroy(ctx);
    close_cuda();
    unload_nvidia();
    unblock_all();
    bs_release(bs, gpu);
}

/*
 * Phase 1 for one die: acquire+load, CUDA write distinct pattern, run the
 * compute kernel, sync, verify the write+compute landed, then tear down.
 * Records what Phase 2 must re-verify. Returns 0 on success, -1 on failure.
 */
static int phase1_write_and_compute(bs_context *bs, die_expectation *exp)
{
    int bdf_idx = -1;
    int rc = -1;
    CUcontext ctx = NULL;
    CUdeviceptr d_in = 0, d_out = 0;
    float *h_in = NULL, *h_check = NULL;

    printf(COLOR_YELLOW "\n[Phase 1] die %d — write pattern + compute" COLOR_RESET "\n",
           exp->die_id);

    bs_gpu *gpu = acquire_and_load(bs, exp->die_id, &bdf_idx);
    if (!gpu)
        return -1;

    if (load_cuda_symbols() != 0)
        goto cleanup;

    CUresult err;
    CUdevice dev;
    if ((err = p_cuInit(0)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuInit failed: %d\n", err); goto cleanup;
    }
    if ((err = p_cuDeviceGet(&dev, 0)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuDeviceGet failed: %d\n", err); goto cleanup;
    }
    if ((err = p_cuCtxCreate(&ctx, 0, dev)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuCtxCreate failed: %d\n", err); ctx = NULL; goto cleanup;
    }

    size_t buf_bytes = (size_t)N_ELEMENTS * sizeof(float);
    if ((err = p_cuMemAlloc(&d_in, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuMemAlloc(in) failed: %d\n", err); goto cleanup;
    }
    if ((err = p_cuMemAlloc(&d_out, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuMemAlloc(out) failed: %d\n", err); goto cleanup;
    }

    /* Upload this die's distinct input pattern */
    h_in = (float *)malloc(buf_bytes);
    fill_die_pattern(h_in, exp->base_value);
    if ((err = p_cuMemcpyHtoD(d_in, h_in, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuMemcpyHtoD failed: %d\n", err); goto cleanup;
    }

    /* Compute: out[i] = in[i] + N_ITERS */
    CUmodule mod;
    CUfunction kern;
    if ((err = p_cuModuleLoadData(&mod, add_iters_ptx)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuModuleLoadData failed: %d\n", err); goto cleanup;
    }
    if ((err = p_cuModuleGetFunction(&kern, mod, "add_iters")) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuModuleGetFunction failed: %d\n", err); goto cleanup;
    }

    unsigned int n = N_ELEMENTS, iters = N_ITERS;
    unsigned int grid = (N_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *args[] = { &d_out, &d_in, &n, &iters };
    if ((err = p_cuLaunchKernel(kern, grid, 1, 1, BLOCK_SIZE, 1, 1,
                                0, NULL, args, NULL)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuLaunchKernel failed: %d\n", err); goto cleanup;
    }
    if ((err = p_cuCtxSynchronize()) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuCtxSynchronize failed: %d\n", err); goto cleanup;
    }

    /* Verify the write+compute landed before we tear the driver down */
    h_check = (float *)malloc(buf_bytes);
    p_cuMemcpyDtoH(h_check, d_out, buf_bytes);
    int ok = 1;
    for (int i = 0; i < N_ELEMENTS; i++)
        if (h_check[i] != exp->expected_compute) { ok = 0; break; }
    if (!ok) {
        fprintf(stderr, "  Phase 1 self-check failed for die %d\n", exp->die_id);
        goto cleanup;
    }

    printf("  die %d: input=%.0f, compute out=%.0f (=input+%d), in-die verify OK\n",
           exp->die_id, exp->base_value, exp->expected_compute, N_ITERS);
    exp->phase1_ok = 1;
    rc = 0;

cleanup:
    /* Free host buffers; leave GPU memory allocated so VRAM contents persist.
     * Mirrors vram_persist.c — do NOT cuMemFree before reload. */
    free(h_in);
    free(h_check);
    teardown_die(bs, gpu, ctx);
    return rc;
}

/*
 * Phase 2 for one die: re-acquire+load, then verify the input pattern AND the
 * compute result are still resident in VRAM after the OTHER dies were each
 * loaded/unloaded between Phase 1 and now.
 *
 * Verification mirrors vram_persist.c lines 270-315: the CUDA allocator resets
 * on driver reload, so the old CUdeviceptr is gone. We allocate fresh buffers
 * of the SAME size and read them back, relying on the fresh allocation landing
 * on the same physical VRAM. Full match → persisted; all zero → wiped.
 * Returns 0 on success (test ran), -1 if it could not run.
 */
static int phase2_verify(bs_context *bs, die_expectation *exp)
{
    int bdf_idx = -1;
    int rc = -1;
    CUcontext ctx = NULL;
    CUdeviceptr d_in = 0, d_out = 0;
    float *r_in = NULL, *r_out = NULL;

    printf(COLOR_YELLOW "\n[Phase 2] die %d — re-verify VRAM + compute" COLOR_RESET "\n",
           exp->die_id);

    bs_gpu *gpu = acquire_and_load(bs, exp->die_id, &bdf_idx);
    if (!gpu)
        return -1;

    if (load_cuda_symbols() != 0)
        goto cleanup;

    CUresult err;
    CUdevice dev;
    if ((err = p_cuInit(0)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuInit failed: %d\n", err); goto cleanup;
    }
    if ((err = p_cuDeviceGet(&dev, 0)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuDeviceGet failed: %d\n", err); goto cleanup;
    }
    if ((err = p_cuCtxCreate(&ctx, 0, dev)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuCtxCreate failed: %d\n", err); ctx = NULL; goto cleanup;
    }

    size_t buf_bytes = (size_t)N_ELEMENTS * sizeof(float);

    /* Fresh same-size allocations — same order/size as Phase 1 (in then out),
     * to land on the same physical VRAM. (vram_persist.c:278) */
    if ((err = p_cuMemAlloc(&d_in, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuMemAlloc(in) failed: %d\n", err); goto cleanup;
    }
    if ((err = p_cuMemAlloc(&d_out, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuMemAlloc(out) failed: %d\n", err); goto cleanup;
    }

    /* Read back without writing anything — checking for residual data.
     * (vram_persist.c:290-292) */
    r_in = (float *)malloc(buf_bytes);
    r_out = (float *)malloc(buf_bytes);
    memset(r_in, 0, buf_bytes);
    memset(r_out, 0, buf_bytes);
    p_cuMemcpyDtoH(r_in, d_in, buf_bytes);
    p_cuMemcpyDtoH(r_out, d_out, buf_bytes);

    /* Tally input-pattern match (vram_persist.c:304-315) */
    int in_match = 0;
    for (int i = 0; i < N_ELEMENTS; i++)
        if (r_in[i] == exp->base_value)
            in_match++;
    /* Tally compute-result match */
    int out_match = 0;
    for (int i = 0; i < N_ELEMENTS; i++)
        if (r_out[i] == exp->expected_compute)
            out_match++;

    exp->vram_pass = (in_match == N_ELEMENTS);
    exp->compute_pass = (out_match == N_ELEMENTS);

    printf("  die %d: vram pattern %d/%d, compute result %d/%d\n",
           exp->die_id, in_match, N_ELEMENTS, out_match, N_ELEMENTS);
    rc = 0;

cleanup:
    if (d_in)  p_cuMemFree(d_in);
    if (d_out) p_cuMemFree(d_out);
    free(r_in);
    free(r_out);
    teardown_die(bs, gpu, ctx);
    return rc;
}

int main(int argc, char *argv[])
{
    printf("=== Cross-Card VRAM + Compute Persistence Test ===\n\n");

    /* Gate on barswap availability so this compiles+runs everywhere. */
    bs_context *bs = bs_init();
    if (!bs) {
        printf("[SKIP] no barswap (bs_init failed — not on a K80 host)\n");
        return 0;
    }

    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <die0> <die1> [die2 ...]   (>= 2 dies, cross-card)\n"
                "  e.g. %s 0 2 4\n", argv[0], argv[0]);
        bs_free(bs);
        return 1;
    }

    if (getuid() != 0) {
        fprintf(stderr, "error: must run as root (sudo) — needs modprobe\n");
        bs_free(bs);
        return 1;
    }

    int n_dies = argc - 1;
    if (n_dies > MAX_DIES)
        n_dies = MAX_DIES;

    die_expectation dies[MAX_DIES] = {0};
    for (int i = 0; i < n_dies; i++) {
        dies[i].die_id = atoi(argv[i + 1]);
        /* Distinct per-die base value; +N_ITERS stays exact in f32. */
        dies[i].base_value = (float)(1000 * (i + 1));
        dies[i].expected_compute = dies[i].base_value + (float)N_ITERS;
    }

    discover_gk210s();
    unload_nvidia();  /* start from a clean driver state */

    printf("barswap: %u dies available, %d GK210s in sysfs\n",
           bs_num_dies(bs), n_gk210);
    printf("testing %d dies:", n_dies);
    for (int i = 0; i < n_dies; i++)
        printf(" %d", dies[i].die_id);
    printf("  (each gets a distinct pattern)\n");

    /* Phase 1: write+compute on each die in turn. Between any two dies the
     * driver is fully torn down and rebuilt around the next die — that is the
     * whole point of the cross-card test. */
    int phase1_failures = 0;
    for (int i = 0; i < n_dies; i++)
        if (phase1_write_and_compute(bs, &dies[i]) != 0)
            phase1_failures++;

    /* Phase 2: re-verify each die retained its VRAM + compute result. */
    int phase2_failures = 0;
    for (int i = 0; i < n_dies; i++)
        if (phase2_verify(bs, &dies[i]) != 0)
            phase2_failures++;

    /* Report */
    printf("\n=== Results ===\n");
    int all_pass = 1;
    for (int i = 0; i < n_dies; i++) {
        die_expectation *d = &dies[i];
        int pass = d->phase1_ok && d->vram_pass && d->compute_pass;
        if (!pass)
            all_pass = 0;
        printf("%s die %d:" COLOR_RESET " vram=%s compute=%s\n",
               pass ? COLOR_GREEN "[PASS]" : COLOR_RED "[FAIL]",
               d->die_id,
               d->vram_pass ? "OK" : "LOST",
               d->compute_pass ? "OK" : "LOST");
    }

    if (phase1_failures || phase2_failures)
        all_pass = 0;

    printf("\n%s cross-card persistence (%d dies)" COLOR_RESET "\n",
           all_pass ? COLOR_GREEN "[PASS]" : COLOR_RED "[FAIL]", n_dies);

    bs_free(bs);

    /* The rig's CUDA atexit teardown hangs after manual swaps — hard-exit. */
    _Exit(all_pass ? 0 : 1);
}
