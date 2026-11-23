/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gpu_owner_util.c
 * @brief Implementation of the shared multi-die GPU resource-owner helpers:
 *   GK210 sysfs enumeration + driver_override blocking, nvidia load-once, the
 *   dlopen'd CUDA driver-API table, and the slow_add die-pattern helpers.
 *
 * Helpers were extracted verbatim (renamed + de-static'd) from
 * barswap/tests/worker_switch.c and barswap/tests/compute_persist.c so the
 * owner and its tests share one copy. The gk210_bdf[]/n_gk210 cache stays
 * file-private here and is exposed only through gou_gk210_bdf()/gou_num_gk210().
 *
 * @status None
 * @issues None
 * @todo None
 */

/* 1. This module's own header */
#include "barswap/gpu_owner_util.h"

/* 2. Standard library */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 3. System / platform headers */
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#define CUDA_SUCCESS 0

/* ================================================================
 * GK210 enumeration + driver_override protocol (ported from
 * worker_switch.c — verbatim logic, renamed + de-static'd).
 * ================================================================ */

/* BDF strings for all GK210 devices found in sysfs (file-private cache). */
static char gk210_bdf[GOU_MAX_DIES][16];
static int n_gk210;

int gou_discover_gk210s(void)
{
    FILE *fp;
    char path[512], buf[16];
    DIR *dir;
    struct dirent *ent;
    char *bdfs[GOU_MAX_DIES];
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

        if (count < GOU_MAX_DIES) {
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

int gou_num_gk210(void)
{
    return n_gk210;
}

const char *gou_gk210_bdf(int index)
{
    if (index < 0 || index >= n_gk210)
        return NULL;
    return gk210_bdf[index];
}

uint64_t gou_read_bar0(const char *bdf)
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

int gou_find_acquired_bdf(void)
{
    for (int i = 0; i < n_gk210; i++) {
        if (gou_read_bar0(gk210_bdf[i]) != 0)
            return i;
    }
    return -1;
}

int gou_sysfs_write(const char *path, const char *str)
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

void gou_set_driver_override(const char *bdf, const char *value)
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

void gou_block_all_except(int bdf_index)
{
    for (int i = 0; i < n_gk210; i++) {
        if (i == bdf_index)
            gou_set_driver_override(gk210_bdf[i], "");      /* clear — allow nvidia */
        else
            gou_set_driver_override(gk210_bdf[i], "none");  /* block */
    }
}

void gou_unblock_all(void)
{
    for (int i = 0; i < n_gk210; i++)
        gou_set_driver_override(gk210_bdf[i], "");
}

int gou_load_nvidia_once(void)
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
 * CUDA driver API via dlopen (ported from compute_persist.c — the loose
 * p_cu* globals reshaped into a single gou_cuda table).
 * ================================================================ */

static void *libcuda = NULL;
static gou_cuda cuda;
static int cuda_loaded;

const gou_cuda *gou_cuda_load_symbols(void)
{
    if (cuda_loaded)
        return &cuda;

    libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!libcuda)
        libcuda = dlopen("libcuda.so", RTLD_LAZY);
    if (!libcuda)
        return NULL;   /* SKIP path: not on a K80 host (e.g. the Mac) */

    /* _v2 symbols where the originals do (legacy v1 ctx is ABI-incompatible). */
    #define LOAD(member, sym, type) do { \
        cuda.member = (type)dlsym(libcuda, sym); \
        if (!cuda.member) { \
            fprintf(stderr, "  dlsym(%s) failed: %s\n", sym, dlerror()); \
            dlclose(libcuda); \
            libcuda = NULL; \
            return NULL; \
        } \
    } while (0)

    LOAD(cuInit,              "cuInit",              CUresult (*)(unsigned));
    LOAD(cuDeviceGetCount,    "cuDeviceGetCount",    CUresult (*)(int*));
    LOAD(cuDeviceGet,         "cuDeviceGet",         CUresult (*)(CUdevice*,int));
    LOAD(cuCtxCreate,         "cuCtxCreate_v2",      CUresult (*)(CUcontext*,unsigned,CUdevice));
    LOAD(cuCtxDestroy,        "cuCtxDestroy_v2",     CUresult (*)(CUcontext));
    LOAD(cuCtxSetCurrent,     "cuCtxSetCurrent",     CUresult (*)(CUcontext));
    LOAD(cuCtxGetCurrent,     "cuCtxGetCurrent",     CUresult (*)(CUcontext*));
    LOAD(cuCtxSynchronize,    "cuCtxSynchronize",    CUresult (*)(void));
    LOAD(cuMemAlloc,          "cuMemAlloc_v2",       CUresult (*)(CUdeviceptr*,size_t));
    LOAD(cuMemFree,           "cuMemFree_v2",        CUresult (*)(CUdeviceptr));
    LOAD(cuMemcpyHtoD,        "cuMemcpyHtoD_v2",     CUresult (*)(CUdeviceptr,const void*,size_t));
    LOAD(cuMemcpyDtoH,        "cuMemcpyDtoH_v2",     CUresult (*)(void*,CUdeviceptr,size_t));
    LOAD(cuMemGetInfo,        "cuMemGetInfo_v2",     CUresult (*)(size_t*,size_t*));
    LOAD(cuModuleLoadData,    "cuModuleLoadData",    CUresult (*)(CUmodule*,const void*));
    LOAD(cuModuleGetFunction, "cuModuleGetFunction", CUresult (*)(CUfunction*,CUmodule,const char*));
    LOAD(cuLaunchKernel,      "cuLaunchKernel",
         CUresult (*)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**));

    #undef LOAD

    cuda_loaded = 1;
    return &cuda;
}

/* ================================================================
 * Compute payload — slow_add PTX + die-pattern helpers (ported from
 * worker_switch.c: slow_add_ptx, fill_die_pattern, expected_compute_at).
 * ================================================================ */

/*
 * Embedded PTX kernel — adds 1.0 to each element `iters` times.
 * out[i] = in[i] + iters, exact in float32 while values stay under 2^24.
 */
const char *const gou_slow_add_ptx =
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

void gou_fill_die_pattern(float *buf, int n, uint32_t seed)
{
    for (int i = 0; i < n; i++) {
        uint32_t v = (seed + (uint32_t)i * 2654435761u) & 0xFFFu;
        buf[i] = (float)v;
    }
}

int gou_verify_die_pattern(const float *buf, int n, uint32_t seed, int iters)
{
    for (int i = 0; i < n; i++) {
        uint32_t v = (seed + (uint32_t)i * 2654435761u) & 0xFFFu;
        float expected = (float)v + (float)iters;
        if (buf[i] != expected)
            return 0;
    }
    return 1;
}
