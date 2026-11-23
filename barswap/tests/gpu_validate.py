#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
GPU Validation Test for K80 BAR Swap

Tests the GPU using the CUDA driver API directly via ctypes.
No CUDA toolkit or Python packages required — just libcuda.so from the nvidia driver.

Tests:
  1. CUDA initialization + device detection
  2. Device properties (name, memory, compute capability)
  3. GPU memory allocation (1 GB)
  4. Host-to-device memcpy (1 MB pattern)
  5. Device-to-host memcpy + verification
  6. Large transfer bandwidth (256 MB)
"""

import ctypes
import ctypes.util
import struct
import time
import sys

# CUDA driver API constants
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 8
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
CU_MEMHOSTALLOC_PORTABLE = 0x01

def load_cuda():
    """Load libcuda.so and set up function signatures."""
    path = ctypes.util.find_library("cuda")
    if not path:
        # Try common paths
        for p in ["/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                  "/usr/lib/x86_64-linux-gnu/libcuda.so",
                  "/usr/lib/libcuda.so.1"]:
            try:
                return ctypes.CDLL(p)
            except OSError:
                continue
        raise RuntimeError("Cannot find libcuda.so — is the nvidia driver installed?")
    return ctypes.CDLL(path)

def cuda_check(cuda, result, msg):
    """Check CUDA return code and print error if failed."""
    if result != CUDA_SUCCESS:
        err_str = ctypes.c_char_p()
        cuda.cuGetErrorString(result, ctypes.byref(err_str))
        err = err_str.value.decode() if err_str.value else f"error {result}"
        print(f"  FAIL: {msg} — {err}")
        return False
    return True

def main():
    print("=== K80 GPU Validation Test ===\n")

    # Load CUDA driver API
    try:
        cuda = load_cuda()
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 1

    # Test 1: Initialize CUDA
    print("[1] Initializing CUDA driver API...")
    ret = cuda.cuInit(0)
    if not cuda_check(cuda, ret, "cuInit"):
        return 1

    device_count = ctypes.c_int(0)
    ret = cuda.cuDeviceGetCount(ctypes.byref(device_count))
    if not cuda_check(cuda, ret, "cuDeviceGetCount"):
        return 1

    print(f"    Devices found: {device_count.value}")
    if device_count.value == 0:
        print("FAIL: No CUDA devices")
        return 1
    print("[1] PASS\n")

    # Test 2: Device properties
    print("[2] Querying device properties...")
    device = ctypes.c_int(0)
    ret = cuda.cuDeviceGet(ctypes.byref(device), 0)
    if not cuda_check(cuda, ret, "cuDeviceGet"):
        return 1

    name = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name, 256, device)
    print(f"    Name: {name.value.decode()}")

    total_mem = ctypes.c_size_t(0)
    cuda.cuDeviceTotalMem_v2(ctypes.byref(total_mem), device)
    print(f"    Memory: {total_mem.value // (1024*1024)} MB")

    def get_attr(attr_id):
        val = ctypes.c_int(0)
        cuda.cuDeviceGetAttribute(ctypes.byref(val), attr_id, device)
        return val.value

    cc_major = get_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    cc_minor = get_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    sm_count = get_attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    clock_mhz = get_attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE) // 1000
    mem_clock = get_attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) // 1000
    bus_width = get_attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)

    print(f"    Compute capability: {cc_major}.{cc_minor}")
    print(f"    SMs: {sm_count}, Core clock: {clock_mhz} MHz")
    print(f"    Memory clock: {mem_clock} MHz, Bus width: {bus_width}-bit")
    peak_bw = 2.0 * mem_clock * 1e6 * (bus_width / 8) / 1e9
    print(f"    Peak memory bandwidth: {peak_bw:.1f} GB/s")

    if total_mem.value < 10 * 1024 * 1024 * 1024:  # < 10 GB
        print(f"  WARNING: Expected ~12 GB, got {total_mem.value // (1024*1024)} MB")
    print("[2] PASS\n")

    # Create context
    context = ctypes.c_void_p(0)
    ret = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
    if not cuda_check(cuda, ret, "cuCtxCreate"):
        return 1

    # Test 3: GPU memory allocation (1 GB)
    print("[3] Allocating 1 GB GPU memory...")
    alloc_size = 1024 * 1024 * 1024  # 1 GB
    d_ptr = ctypes.c_uint64(0)
    ret = cuda.cuMemAlloc_v2(ctypes.byref(d_ptr), alloc_size)
    if not cuda_check(cuda, ret, f"cuMemAlloc({alloc_size} bytes)"):
        return 1
    print(f"    Allocated at device address: 0x{d_ptr.value:016x}")
    print("[3] PASS\n")

    # Test 4: Host-to-device memcpy (1 MB test pattern)
    test_size = 1024 * 1024  # 1 MB
    n_ints = test_size // 4

    print(f"[4] Host-to-device memcpy ({test_size // 1024} KB)...")
    # Create test pattern: 0, 1, 2, 3, ... (32-bit integers)
    h_data = (ctypes.c_uint32 * n_ints)()
    for i in range(n_ints):
        h_data[i] = i & 0xFFFFFFFF

    ret = cuda.cuMemcpyHtoD_v2(d_ptr, h_data, test_size)
    if not cuda_check(cuda, ret, "cuMemcpyHtoD"):
        return 1
    print("[4] PASS\n")

    # Test 5: Device-to-host + verify
    print("[5] Device-to-host memcpy + verification...")
    h_result = (ctypes.c_uint32 * n_ints)()
    ret = cuda.cuMemcpyDtoH_v2(h_result, d_ptr, test_size)
    if not cuda_check(cuda, ret, "cuMemcpyDtoH"):
        return 1

    errors = 0
    for i in range(n_ints):
        if h_result[i] != (i & 0xFFFFFFFF):
            if errors < 5:
                print(f"    MISMATCH at [{i}]: got 0x{h_result[i]:08x}, expected 0x{i & 0xFFFFFFFF:08x}")
            errors += 1

    if errors == 0:
        print(f"    All {n_ints} values verified correct")
        print("[5] PASS\n")
    else:
        print(f"    {errors} mismatches out of {n_ints}")
        print("[5] FAIL\n")
        return 1

    # Test 6: Bandwidth measurement (256 MB transfers)
    bw_size = 256 * 1024 * 1024  # 256 MB
    print(f"[6] Bandwidth test ({bw_size // (1024*1024)} MB transfers)...")

    d_bw = ctypes.c_uint64(0)
    ret = cuda.cuMemAlloc_v2(ctypes.byref(d_bw), bw_size)
    if not cuda_check(cuda, ret, "cuMemAlloc for bandwidth"):
        return 1

    h_bw = (ctypes.c_char * bw_size)()

    # H2D bandwidth
    t0 = time.monotonic()
    ret = cuda.cuMemcpyHtoD_v2(d_bw, h_bw, bw_size)
    cuda.cuCtxSynchronize()
    t1 = time.monotonic()
    if not cuda_check(cuda, ret, "cuMemcpyHtoD bandwidth"):
        return 1
    h2d_bw = bw_size / (t1 - t0) / 1e9
    print(f"    Host-to-Device: {h2d_bw:.2f} GB/s")

    # D2H bandwidth
    t0 = time.monotonic()
    ret = cuda.cuMemcpyDtoH_v2(h_bw, d_bw, bw_size)
    cuda.cuCtxSynchronize()
    t1 = time.monotonic()
    if not cuda_check(cuda, ret, "cuMemcpyDtoH bandwidth"):
        return 1
    d2h_bw = bw_size / (t1 - t0) / 1e9
    print(f"    Device-to-Host: {d2h_bw:.2f} GB/s")

    # D2D bandwidth (copy within GPU)
    d_bw2 = ctypes.c_uint64(0)
    ret = cuda.cuMemAlloc_v2(ctypes.byref(d_bw2), bw_size)
    if not cuda_check(cuda, ret, "cuMemAlloc for D2D"):
        return 1

    t0 = time.monotonic()
    ret = cuda.cuMemcpyDtoD_v2(d_bw2, d_bw, bw_size)
    cuda.cuCtxSynchronize()
    t1 = time.monotonic()
    if not cuda_check(cuda, ret, "cuMemcpyDtoD bandwidth"):
        return 1
    d2d_bw = bw_size / (t1 - t0) / 1e9
    print(f"    Device-to-Device: {d2d_bw:.2f} GB/s")

    cuda.cuMemFree_v2(d_bw)
    cuda.cuMemFree_v2(d_bw2)
    print("[6] PASS\n")

    # Cleanup
    cuda.cuMemFree_v2(d_ptr)
    cuda.cuCtxDestroy_v2(context)

    print("=== ALL 6 TESTS PASSED ===")
    print("GPU is fully functional — VRAM accessible, data integrity verified.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
