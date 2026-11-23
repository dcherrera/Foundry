/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#define _GNU_SOURCE

/*
 * roundrobin_persist.c — No-teardown round-robin multi-die persistence test
 *
 * Proves the production switch mechanism for the K80 barswap rig: nvidia is
 * loaded EXACTLY ONCE and never rmmod'd. Weights stay resident in each die's
 * VRAM, and we move between dies purely by:
 *
 *   1. cuCtxDestroy        — drop the current die's CUDA context (driver stays)
 *   2. PCI unbind          — write the old device's BDF to .../nvidia/unbind
 *   3. bs_acquire(die)     — barswap maps the next die into the single MMIO slot
 *   4. driver_override     — clear it on the target, "none" on every other GK210
 *   5. PCI bind            — write the new device's BDF to .../nvidia/bind
 *                            (or, on the very first switch only, modprobe once)
 *   6. cuCtxCreate         — fresh context on device 0 (== the slot die)
 *
 * This is the no-teardown analogue of xcard_persist.c. xcard_persist proves
 * persistence across a full driver unload/reload between dies; this proves the
 * SAME persistence while the nvidia module never leaves the kernel — only the
 * single active PCI function is rebound. That is the mechanism we actually ship.
 *
 * Per-die payload is a distinct pattern seeded from a real SmolLM2 weight chunk
 * (per-die file offset) so a stale read of the wrong die's VRAM cannot pass.
 *
 * Persistence is verified the vram_persist.c way (lines 270-315): after
 * re-establishing a context on the die, allocate same-size buffers in the SAME
 * order as Phase 1 so they land on the same physical VRAM, read back WITHOUT
 * writing, and compare to the recorded expected pattern + compute result.
 *
 * Reuses (verbatim where noted):
 *   - test_swap.c   : discover_gk210s, read_bar0, find_acquired_bdf,
 *                     set_driver_override, block_all_except, unblock_all
 *   - compute_persist.c : the dlopen CUDA scaffolding + the slow_add PTX kernel
 *   - vram_persist.c    : the same-size-realloc persistence-verify trick
 *
 * Usage (on a K80 host, as root):
 *   make roundrobin-persist
 *   sudo ./barswap/tests/roundrobin_persist <die0> <die1> ...
 *   e.g. sudo ./barswap/tests/roundrobin_persist 0 2 4 6 8 10 12 14
 *
 * SmolLM2 payload path: argv has priority via none (it's all dies); the path
 * comes from $SMOL_GGUF, else the SMOL_GGUF_DEFAULT #define. Missing file ->
 * synthetic per-die ramp (noted in output).
 *
 * On a host with no barswap device (e.g. the Mac) it prints [SKIP] and exits 0,
 * so it compiles, links, and runs everywhere. Shells out to modprobe ONCE and
 * to sysfs bind/unbind — must run as root. Calls _Exit() at the end because the
 * rig's CUDA atexit teardown hangs after manual swaps.
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

/* Default SmolLM2 weights path; override with $SMOL_GGUF. */
#define SMOL_GGUF_DEFAULT "/root/n2-test/models/SmolLM2-135M-Instruct-Q8_0.gguf"

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
 * Mirrors test_swap.c:find_acquired_bdf(), keyed on "non-zero" since the
 * barswap lib's bs_gpu does not expose bar0_addr.
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
 * Returns 0 on success, -1 on error (open or write failure), reporting via
 * stderr. The trailing newline expected by the PCI bind/unbind attributes must
 * be included by the caller.
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

/*
 * Load nvidia ONCE for the whole run. Mirrors compute_persist.c:load_nvidia().
 * After this returns the module stays resident — it is never unloaded.
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

/*
 * Confirm the nvidia module is still resident in the kernel. Reads
 * /proc/modules and looks for the "nvidia " entry. Returns 1 if present.
 * Used at the end to assert the module never left.
 */
static int nvidia_module_present(void)
{
    FILE *fp = fopen("/proc/modules", "r");
    if (!fp)
        return 0;
    char line[256];
    int found = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "nvidia ", 7) == 0) { found = 1; break; }
    }
    fclose(fp);
    return found;
}

/* ================================================================
 * CUDA driver API via dlopen (from compute_persist.c + vram_persist.c)
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

/* ================================================================
 * SmolLM2 per-die payload
 * ================================================================ */

#define N_ELEMENTS  (1024 * 1024)            /* 1M floats = 4 MB per buffer */
#define N_ITERS     1000                     /* 1000 adds per element       */
#define BLOCK_SIZE  256

/* Per-die file stride so each die reads a different SmolLM2 region. */
#define SMOL_DIE_STRIDE_BYTES  (1024 * 1024)  /* 1 MiB apart per die index */

static const char *smol_path;   /* resolved once in main() */
static int smol_available;      /* 1 if the file opened, 0 -> synthetic fallback */

/*
 * Per-die expectation, recorded in Phase 1 and re-checked in Phase 2.
 * Each die gets a DISTINCT input pattern so a stale read of the wrong die's
 * VRAM cannot masquerade as a pass.
 */
typedef struct die_expectation {
    int die_id;
    int list_index;          /* position in the argv die list (drives offset)  */
    uint32_t seed;           /* distinct per-die seed (from SmolLM2 or synthetic)*/
    int bdf_idx;             /* gk210_bdf index this die mapped to (-1 if none) */
    int phase1_ok;           /* Phase 1 wrote + computed successfully           */
    int vram_pass;           /* Phase 2: input pattern survived                 */
    int compute_pass;        /* Phase 2: compute result survived                */
} die_expectation;

/*
 * Derive a distinct 32-bit seed for a die by reading a SmolLM2 weight chunk at
 * a per-die file offset and folding its bytes. Falls back to a synthetic seed
 * if the file can't be read at that offset. The seed makes the resident pattern
 * unmistakably die-specific.
 */
static uint32_t derive_die_seed(int list_index)
{
    uint32_t synthetic = 0x9E3779B9u * (uint32_t)(list_index + 1);

    if (!smol_available)
        return synthetic;

    FILE *fp = fopen(smol_path, "rb");
    if (!fp)
        return synthetic;

    off_t offset = (off_t)list_index * SMOL_DIE_STRIDE_BYTES;
    unsigned char chunk[4096];
    if (fseeko(fp, offset, SEEK_SET) != 0) {
        fclose(fp);
        return synthetic;
    }
    size_t got = fread(chunk, 1, sizeof(chunk), fp);
    fclose(fp);
    if (got == 0)
        return synthetic;

    /* FNV-1a fold of the chunk -> distinct seed tied to real weight bytes */
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < got; i++) {
        h ^= chunk[i];
        h *= 16777619u;
    }
    return h ? h : synthetic;
}

/*
 * Fill a host buffer with a distinct, deterministic per-die pattern derived
 * from the die's seed. Values are kept small (0..4095) so that out = in + N_ITERS
 * stays exact in float32 (well under 2^24), making the compute check reliable
 * regardless of the raw SmolLM2 byte magnitudes.
 */
static void fill_die_pattern(float *host_buf, uint32_t seed)
{
    for (int i = 0; i < N_ELEMENTS; i++) {
        uint32_t v = (seed + (uint32_t)i * 2654435761u) & 0xFFFu;  /* 0..4095 */
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
 * The switch primitive — THE point of this test
 * ================================================================ */

static bs_context *g_bs;
static int active_bdf_idx = -1;     /* gk210_bdf index currently nvidia-bound  */
static int nvidia_loaded;           /* set after the single modprobe           */
static bs_gpu *current_gpu;         /* die currently in the MMIO slot          */
static CUcontext current_ctx;       /* live ctx on the slot die (resident)     */

/*
 * Unbind a single PCI device from the nvidia driver via sysfs.
 * Writes "<bdf>\n" to /sys/bus/pci/drivers/nvidia/unbind.
 */
static int pci_unbind(const char *bdf)
{
    char val[32];
    snprintf(val, sizeof(val), "%s\n", bdf);
    return sysfs_write("/sys/bus/pci/drivers/nvidia/unbind", val);
}

/*
 * Bind a single PCI device to the nvidia driver via sysfs (re-probe just it).
 * Writes "<bdf>\n" to /sys/bus/pci/drivers/nvidia/bind.
 */
static int pci_bind(const char *bdf)
{
    char val[32];
    snprintf(val, sizeof(val), "%s\n", bdf);
    return sysfs_write("/sys/bus/pci/drivers/nvidia/bind", val);
}

/*
 * switch_to — move the active GPU to `die`, NO driver teardown.
 *
 * Drops the current context and PCI-unbinds the old device, acquires the new
 * die into the MMIO slot, points driver_override at it, and either loads nvidia
 * (first switch ONLY) or PCI-binds just this device. Leaves a fresh CUDA context
 * live in current_ctx and the handle in current_gpu. Returns 0 on success.
 *
 * The nvidia module is loaded exactly once across the whole run; every later
 * switch is cuCtxDestroy -> unbind -> acquire -> bind. nvidia is NEVER rmmod'd.
 */
static int switch_to(int die)
{
    CUresult err;
    CUdevice dev;

    /* 1. Drop the previous die's context + unbind its PCI function (driver
     *    stays resident). */
    if (active_bdf_idx >= 0) {
        if (current_ctx) {
            p_cuCtxDestroy(current_ctx);
            current_ctx = NULL;
        }
        printf("  switch: unbinding old device %s\n", gk210_bdf[active_bdf_idx]);
        if (pci_unbind(gk210_bdf[active_bdf_idx]) != 0)
            fprintf(stderr, "  warning: unbind of %s failed\n",
                    gk210_bdf[active_bdf_idx]);
    }

    /* Release the old slot mapping so the next acquire is clean. */
    if (current_gpu) {
        bs_release(g_bs, current_gpu);
        current_gpu = NULL;
    }

    /* 2. Map the target die into the single MMIO slot. */
    current_gpu = bs_acquire(g_bs, die);
    if (!current_gpu) {
        fprintf(stderr, "  switch: bs_acquire(%d) failed\n", die);
        return -1;
    }

    /* 3. Identify which GK210 now occupies the slot. */
    int idx = find_acquired_bdf();
    if (idx < 0) {
        fprintf(stderr, "  switch: could not find acquired BDF for die %d\n", die);
        return -1;
    }
    active_bdf_idx = idx;
    printf("  switch: target die %d -> acquired GK210 %s, active BDF %s\n",
           die, gk210_bdf[idx], gk210_bdf[idx]);

    /* 4. Allow nvidia to bind only the acquired die. */
    block_all_except(idx);

    /* 5. Load nvidia ONCE, ever; afterwards just re-probe this one device. */
    if (!nvidia_loaded) {
        printf("  switch: loading nvidia (first switch only)\n");
        if (load_nvidia_once() != 0) {
            fprintf(stderr, "  switch: nvidia load failed\n");
            return -1;
        }
        nvidia_loaded = 1;
        /* First modprobe probes the only allowed device, binding it for us. */
    } else {
        printf("  switch: binding device %s\n", gk210_bdf[idx]);
        if (pci_bind(gk210_bdf[idx]) != 0)
            fprintf(stderr, "  warning: bind of %s failed\n", gk210_bdf[idx]);
    }

    /* 6. Fresh context on device 0 (== the slot die). */
    if ((err = p_cuInit(0)) != CUDA_SUCCESS) {
        fprintf(stderr, "  switch: cuInit failed: %d\n", err);
        return -1;
    }
    if ((err = p_cuDeviceGet(&dev, 0)) != CUDA_SUCCESS) {
        fprintf(stderr, "  switch: cuDeviceGet failed: %d\n", err);
        return -1;
    }
    if ((err = p_cuCtxCreate(&current_ctx, 0, dev)) != CUDA_SUCCESS) {
        fprintf(stderr, "  switch: cuCtxCreate failed: %d\n", err);
        current_ctx = NULL;
        return -1;
    }
    return 0;
}

/* ================================================================
 * Phases
 * ================================================================ */

/*
 * Phase 1 for one die: switch_to (NO teardown), allocate d_in/d_out, upload the
 * die's distinct pattern, launch the kernel (out = in + N_ITERS), sync, and
 * self-verify. Buffers and the context are LEFT resident — the next switch_to
 * handles cuCtxDestroy + PCI unbind. Returns 0 on success.
 */
static int phase1_load_resident(die_expectation *exp)
{
    CUresult err;
    CUdeviceptr d_in = 0, d_out = 0;
    CUmodule mod;
    CUfunction kern;
    float *h_in = NULL, *h_check = NULL;
    int rc = -1;

    printf(COLOR_YELLOW "\n[Phase 1] die %d — load resident (no teardown)" COLOR_RESET "\n",
           exp->die_id);

    if (switch_to(exp->die_id) != 0)
        return -1;
    exp->bdf_idx = active_bdf_idx;

    size_t buf_bytes = (size_t)N_ELEMENTS * sizeof(float);
    if ((err = p_cuMemAlloc(&d_in, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuMemAlloc(in) failed: %d\n", err); goto done;
    }
    if ((err = p_cuMemAlloc(&d_out, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuMemAlloc(out) failed: %d\n", err); goto done;
    }

    h_in = (float *)malloc(buf_bytes);
    fill_die_pattern(h_in, exp->seed);
    if ((err = p_cuMemcpyHtoD(d_in, h_in, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuMemcpyHtoD failed: %d\n", err); goto done;
    }

    if ((err = p_cuModuleLoadData(&mod, slow_add_ptx)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuModuleLoadData failed: %d\n", err); goto done;
    }
    if ((err = p_cuModuleGetFunction(&kern, mod, "slow_add")) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuModuleGetFunction failed: %d\n", err); goto done;
    }

    unsigned int n = N_ELEMENTS, iters = N_ITERS;
    unsigned int grid = (N_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *args[] = { &d_out, &d_in, &n, &iters };
    if ((err = p_cuLaunchKernel(kern, grid, 1, 1, BLOCK_SIZE, 1, 1,
                                0, NULL, args, NULL)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuLaunchKernel failed: %d\n", err); goto done;
    }
    if ((err = p_cuCtxSynchronize()) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuCtxSynchronize failed: %d\n", err); goto done;
    }

    /* In-die self-check before we ever leave this die. */
    h_check = (float *)malloc(buf_bytes);
    p_cuMemcpyDtoH(h_check, d_out, buf_bytes);
    int ok = 1;
    for (int i = 0; i < N_ELEMENTS; i++)
        if (h_check[i] != expected_compute_at(exp->seed, i)) { ok = 0; break; }
    if (!ok) {
        fprintf(stderr, "  Phase 1 self-check failed for die %d\n", exp->die_id);
        goto done;
    }

    printf("  die %d: seed=0x%08x, pattern uploaded + compute (out=in+%d), verify OK\n",
           exp->die_id, exp->seed, N_ITERS);
    printf("  die %d: buffers LEFT resident, ctx LEFT live (next switch tears down)\n",
           exp->die_id);
    exp->phase1_ok = 1;
    rc = 0;

done:
    /* Free host buffers only. GPU buffers stay allocated; the context stays
     * live — both are intentionally left resident across the switch. */
    free(h_in);
    free(h_check);
    return rc;
}

/*
 * Phase 2 for one die: switch_to, re-allocate the SAME two buffers in the SAME
 * order (in then out) so they land on the same physical VRAM, read back WITHOUT
 * writing, and compare to the recorded pattern + compute result.
 * Verification mirrors vram_persist.c lines 278 / 290-292 / 304-315.
 * Returns 0 if the check ran, -1 if it could not run.
 */
static int phase2_verify(die_expectation *exp)
{
    CUresult err;
    CUdeviceptr d_in = 0, d_out = 0;
    float *r_in = NULL, *r_out = NULL;
    int rc = -1;

    printf(COLOR_YELLOW "\n[Phase 2] die %d — verify resident (no teardown)" COLOR_RESET "\n",
           exp->die_id);

    if (switch_to(exp->die_id) != 0)
        return -1;

    size_t buf_bytes = (size_t)N_ELEMENTS * sizeof(float);

    /* Same-size, same-order allocations as Phase 1 -> same physical VRAM.
     * (vram_persist.c:278) */
    if ((err = p_cuMemAlloc(&d_in, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuMemAlloc(in) failed: %d\n", err); goto done;
    }
    if ((err = p_cuMemAlloc(&d_out, buf_bytes)) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuMemAlloc(out) failed: %d\n", err); goto done;
    }

    /* Read back without writing — checking for residual data.
     * (vram_persist.c:290-292) */
    r_in = (float *)malloc(buf_bytes);
    r_out = (float *)malloc(buf_bytes);
    memset(r_in, 0, buf_bytes);
    memset(r_out, 0, buf_bytes);
    p_cuMemcpyDtoH(r_in, d_in, buf_bytes);
    p_cuMemcpyDtoH(r_out, d_out, buf_bytes);

    /* Tally per-element matches (vram_persist.c:304-315). */
    int in_match = 0, out_match = 0;
    for (int i = 0; i < N_ELEMENTS; i++) {
        uint32_t v = (exp->seed + (uint32_t)i * 2654435761u) & 0xFFFu;
        if (r_in[i] == (float)v)
            in_match++;
        if (r_out[i] == (float)v + (float)N_ITERS)
            out_match++;
    }

    exp->vram_pass = (in_match == N_ELEMENTS);
    exp->compute_pass = (out_match == N_ELEMENTS);

    printf("  die %d: vram pattern %d/%d, compute result %d/%d\n",
           exp->die_id, in_match, N_ELEMENTS, out_match, N_ELEMENTS);
    rc = 0;

done:
    if (d_in)  p_cuMemFree(d_in);
    if (d_out) p_cuMemFree(d_out);
    free(r_in);
    free(r_out);
    return rc;
}

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("=== Round-Robin No-Teardown Persistence Test ===\n\n");

    /* Gate on barswap availability so this compiles+runs everywhere. */
    g_bs = bs_init();
    if (!g_bs) {
        printf("[SKIP] no barswap (bs_init failed — not on a K80 host)\n");
        return 0;
    }

    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <die0> <die1> ...   (e.g. %s 0 2 4 6 8 10 12 14)\n",
                argv[0], argv[0]);
        bs_free(g_bs);
        return 1;
    }

    if (getuid() != 0) {
        fprintf(stderr, "error: must run as root (sudo) — needs modprobe + sysfs\n");
        bs_free(g_bs);
        return 1;
    }

    /* Resolve SmolLM2 payload source: $SMOL_GGUF, else the default path. */
    smol_path = getenv("SMOL_GGUF");
    if (!smol_path || !*smol_path)
        smol_path = SMOL_GGUF_DEFAULT;
    FILE *probe = fopen(smol_path, "rb");
    smol_available = (probe != NULL);
    if (probe)
        fclose(probe);

    int n_dies = argc - 1;
    if (n_dies > MAX_DIES)
        n_dies = MAX_DIES;

    die_expectation dies[MAX_DIES] = {0};
    for (int i = 0; i < n_dies; i++) {
        dies[i].die_id = atoi(argv[i + 1]);
        dies[i].list_index = i;
        dies[i].seed = derive_die_seed(i);
        dies[i].bdf_idx = -1;
    }

    discover_gk210s();

    /* Load CUDA symbols once; the nvidia module itself is loaded lazily by the
     * first switch_to and then never unloaded. */
    if (load_cuda_symbols() != 0) {
        bs_free(g_bs);
        return 1;
    }

    printf("barswap: %u dies available, %d GK210s in sysfs\n",
           bs_num_dies(g_bs), n_gk210);
    printf("SmolLM2 payload: %s (%s)\n", smol_path,
           smol_available ? "found — per-die file offsets" : "MISSING — synthetic ramp");
    printf("testing %d dies (no driver teardown between dies):", n_dies);
    for (int i = 0; i < n_dies; i++)
        printf(" %d", dies[i].die_id);
    printf("\n");

    /* Phase 1: load each die's resident pattern, leaving ctx + buffers live.
     * Between dies there is NO rmmod — only cuCtxDestroy + PCI unbind/bind. */
    int phase1_failures = 0;
    for (int i = 0; i < n_dies; i++)
        if (phase1_load_resident(&dies[i]) != 0)
            phase1_failures++;

    /* Phase 2: re-verify each die retained VRAM + compute result. */
    int phase2_failures = 0;
    for (int i = 0; i < n_dies; i++)
        if (phase2_verify(&dies[i]) != 0)
            phase2_failures++;

    /* Final cleanup of the last resident context + slot mapping. nvidia stays. */
    if (current_ctx) {
        p_cuCtxDestroy(current_ctx);
        current_ctx = NULL;
    }
    if (current_gpu) {
        bs_release(g_bs, current_gpu);
        current_gpu = NULL;
    }
    unblock_all();

    /* Report */
    printf("\n=== Results ===\n");
    int all_pass = 1;
    for (int i = 0; i < n_dies; i++) {
        die_expectation *d = &dies[i];
        int pass = d->phase1_ok && d->vram_pass && d->compute_pass;
        if (!pass)
            all_pass = 0;
        printf("%s die %d (%s):" COLOR_RESET " vram=%s compute=%s\n",
               pass ? COLOR_GREEN "[PASS]" : COLOR_RED "[FAIL]",
               d->die_id,
               d->bdf_idx >= 0 ? gk210_bdf[d->bdf_idx] : "bdf n/a",
               d->vram_pass ? "ok" : "LOST",
               d->compute_pass ? "ok" : "LOST");
    }

    if (phase1_failures || phase2_failures)
        all_pass = 0;

    int still_loaded = nvidia_module_present();
    if (!still_loaded)
        all_pass = 0;
    printf("\nnvidia stayed loaded the whole run: %s\n",
           still_loaded ? COLOR_GREEN "YES" COLOR_RESET
                        : COLOR_RED "NO (module left the kernel!)" COLOR_RESET);

    printf("\n%s round-robin no-teardown persistence (%d dies)" COLOR_RESET "\n",
           all_pass ? COLOR_GREEN "[PASS]" : COLOR_RED "[FAIL]", n_dies);

    bs_free(g_bs);

    /* The rig's CUDA atexit teardown hangs after manual swaps — hard-exit. */
    _Exit(all_pass ? 0 : 1);
}
