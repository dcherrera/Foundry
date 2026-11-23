#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# run_on_gpu.sh — Acquire K80 die, load nvidia, run CUDA tests, cleanup
# Usage: sudo ./cuda/tests/run_on_gpu.sh [die_id]
# Must run as root. bar_swap.ko must be loaded.
set -e

DIE=${1:-0}
FOUNDRY_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== CUDA Test on K80 die $DIE ==="

# Check bar_swap is loaded
if [ ! -c /dev/bar_swap ]; then
    echo "ERROR: /dev/bar_swap not found. Load bar_swap.ko first."
    exit 1
fi

# Build the acquire helper if needed
ACQUIRE="$FOUNDRY_DIR/cuda/tests/acquire_helper"
if [ ! -f "$ACQUIRE" ]; then
    echo "Building acquire helper..."
    cc -Wall -O2 -I"$FOUNDRY_DIR" -o "$ACQUIRE" \
        "$FOUNDRY_DIR/cuda/tests/acquire_helper.c" \
        "$FOUNDRY_DIR/barswap/barswap.c"
fi

# Acquire die
echo "Acquiring die $DIE..."
"$ACQUIRE" acquire "$DIE"

# Load nvidia
echo "Loading nvidia driver..."
modprobe --ignore-install nvidia 2>/dev/null || true
modprobe --ignore-install nvidia_modeset 2>/dev/null || true
modprobe --ignore-install nvidia_drm 2>/dev/null || true
modprobe --ignore-install nvidia_uvm 2>/dev/null || true
sleep 1

# Verify GPU visible
if [ -d /proc/driver/nvidia/gpus/ ]; then
    echo "GPU visible:"
    cat /proc/driver/nvidia/gpus/*/information 2>/dev/null | head -5
else
    echo "WARNING: nvidia loaded but no GPUs in /proc"
fi

# Build CUDA tests
echo ""
echo "Building CUDA tests..."
cd "$FOUNDRY_DIR"
make USE_CUDA=1 USE_BLAS=1 test-cuda-build 2>&1 || \
    make USE_CUDA=1 USE_BLAS=1 cuda/tests/test_cuda 2>&1

# Run CUDA tests with cuBLAS library path
echo ""
echo "Running CUDA tests..."
LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH \
    ./cuda/tests/test_cuda

# Cleanup
echo ""
echo "Cleaning up..."
rmmod nvidia_uvm 2>/dev/null || true
rmmod nvidia_drm 2>/dev/null || true
rmmod nvidia_modeset 2>/dev/null || true
rmmod nvidia 2>/dev/null || true
"$ACQUIRE" release "$DIE"
echo "Done."
