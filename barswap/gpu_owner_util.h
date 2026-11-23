/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gpu_owner_util.h
 * @brief Shared helpers for the multi-die GPU resource owner: GK210 sysfs
 *   enumeration + driver_override blocking, nvidia load-once, a dlopen'd CUDA
 *   driver-API table, and the slow_add die-pattern compute helpers.
 *
 * These are de-static'd, collected copies of helpers that previously lived as
 * file-private statics inside barswap/tests/worker_switch.c and
 * barswap/tests/compute_persist.c. On a host with no nvidia/libcuda (e.g. the
 * Mac), gou_cuda_load_symbols() returns NULL — callers treat that as the SKIP
 * path ("not on a K80 host").
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_GPU_OWNER_UTIL_H
#define FOUNDRY_GPU_OWNER_UTIL_H

#include <stddef.h>
#include <stdint.h>

/* Max GK210 dies addressable on an 8-card K80 host (8 cards x 2 dies). */
#define GOU_MAX_DIES 16

/* ================================================================
 * CUDA driver API types (no cuda.h dependency) + dlopen'd symbol table.
 * ================================================================ */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;

/*
 * Resolved CUDA driver-API entry points. Function-pointer members mirror the
 * subset of libcuda the owner needs; _v2 ABI variants are loaded for the calls
 * that have them (see gou_cuda_load_symbols).
 */
typedef struct gou_cuda {
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGetCount)(int*);
    CUresult (*cuDeviceGet)(CUdevice*,int);
    CUresult (*cuCtxCreate)(CUcontext*,unsigned,CUdevice);
    CUresult (*cuCtxDestroy)(CUcontext);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuCtxGetCurrent)(CUcontext*);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuMemAlloc)(CUdeviceptr*,size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr,const void*,size_t);
    CUresult (*cuMemcpyDtoH)(void*,CUdeviceptr,size_t);
    CUresult (*cuMemGetInfo)(size_t*,size_t*);
    CUresult (*cuModuleLoadData)(CUmodule*,const void*);
    CUresult (*cuModuleGetFunction)(CUfunction*,CUmodule,const char*);
    CUresult (*cuLaunchKernel)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**);
} gou_cuda;

/**
 * Resolve the CUDA driver API by dlopen'ing libcuda and dlsym'ing every entry
 * point into a file-static gou_cuda table (resolved once, idempotent).
 *
 * Tries libcuda.so.1 then libcuda.so. If neither opens (e.g. on the Mac) the
 * function returns NULL — the SKIP path callers treat as "not on a K80 host".
 * If any required symbol fails to resolve it also returns NULL.
 *
 * @return  Pointer to the resolved table, or NULL if libcuda is unavailable
 *          or a required symbol is missing. The table is owned by this module.
 *
 * Thread-safe: no (lazy init of a shared static)
 * Reentrant: no
 * Signal-safe: no
 */
const gou_cuda *gou_cuda_load_symbols(void);

/* ================================================================
 * GK210 enumeration + driver_override protocol (sysfs).
 * ================================================================ */

/**
 * Discover GK210 device BDFs from sysfs (vendor 0x10de, device 0x102d), sort
 * them by BDF string, and cache them file-statically.
 *
 * @return  Number of GK210 devices found (also queryable via gou_num_gk210).
 */
int gou_discover_gk210s(void);

/**
 * @return  Number of GK210 devices cached by the last gou_discover_gk210s().
 */
int gou_num_gk210(void);

/**
 * @param index  Index into the cached GK210 BDF table.
 * @return       The BDF string at index, or NULL if index is out of range.
 */
const char *gou_gk210_bdf(int index);

/**
 * Read BAR 0 start address from a PCI device's sysfs `resource` file.
 *
 * @param bdf  PCI BDF string (e.g. "0000:03:00.0").
 * @return     BAR 0 start address, or 0 if unreadable/unmapped.
 */
uint64_t gou_read_bar0(const char *bdf);

/**
 * Find which cached GK210 currently has BARs programmed into the MMIO slot.
 * Only the acquired die has a non-zero BAR 0 (the rest are zeroed on release).
 *
 * @return  Index into the cached BDF table, or -1 if none found.
 */
int gou_find_acquired_bdf(void);

/**
 * Write a NUL-terminated string to a sysfs attribute (O_WRONLY, write, close).
 * The caller must include any trailing newline the attribute expects.
 *
 * @param path  Absolute sysfs attribute path.
 * @param str   NUL-terminated string to write.
 * @return      0 on success, -1 on error.
 */
int gou_sysfs_write(const char *path, const char *str);

/**
 * Set (or clear) a device's driver_override. Pass "" to clear the override
 * (allow nvidia to bind), or "none" to block binding.
 *
 * @param bdf    PCI BDF string of the device.
 * @param value  Override value; "" clears it.
 */
void gou_set_driver_override(const char *bdf, const char *value);

/**
 * Block every cached GK210 except the one at bdf_index, so nvidia only binds
 * the acquired die.
 *
 * @param bdf_index  Index of the die to leave bindable.
 */
void gou_block_all_except(int bdf_index);

/**
 * Clear the driver_override on every cached GK210.
 */
void gou_unblock_all(void);

/**
 * Load the nvidia and nvidia_uvm modules ONCE for the whole run (via modprobe
 * --ignore-install). After this returns the modules stay resident.
 *
 * @return  0 on success, -1 if a modprobe fails.
 */
int gou_load_nvidia_once(void);

/* ================================================================
 * Compute payload — slow_add PTX + die-pattern helpers.
 * ================================================================ */

/*
 * Embedded PTX kernel (entry "slow_add", .target sm_37): out[i] = in[i] + iters,
 * exact in float32 while values stay under 2^24.
 */
extern const char *const gou_slow_add_ptx;

/**
 * Fill a host buffer with a distinct per-die input pattern. Values are kept in
 * 0..4095 so out = in + iters stays exact in float32.
 *
 * @param buf   Destination buffer (n floats).
 * @param n     Element count.
 * @param seed  Per-die seed selecting the pattern.
 */
void gou_fill_die_pattern(float *buf, int n, uint32_t seed);

/**
 * Verify a buffer holds the slow_add result for a seed: every element must
 * equal gou_fill_die_pattern(seed)'s value + iters.
 *
 * @param buf    Buffer to check (n floats).
 * @param n      Element count.
 * @param seed   Per-die seed the pattern was generated from.
 * @param iters  Number of +1.0 adds the kernel applied.
 * @return       1 if every element matches in[i] + iters, else 0.
 */
int gou_verify_die_pattern(const float *buf, int n, uint32_t seed, int iters);

#endif /* FOUNDRY_GPU_OWNER_UTIL_H */
