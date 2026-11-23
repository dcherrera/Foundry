/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Portions modelled on ggml (MIT License, Copyright (c) 2023-2026 The ggml
 * authors, https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file fm_kernel_matvec_q8_0.m
 * @brief Host dispatch and MSL source for the GGUF Q8_0 matvec kernel.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_matvec_q8_0.h"

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

/* Lanes per output row.  One SIMD group, and exactly one Q8_0 block's worth
 * of weights, so a lane owns element `lane` of whichever block the row is
 * currently on and the simd_sum at the end is a single reduction. */
#define FM_Q8_0_LANES_PER_ROW FM_Q8_0_BLOCK_WEIGHTS

void fm_matvec_q8_0(fm_bufref w, fm_bufref x, fm_bufref out,
                    int rows, int cols, int row_stride_bytes) {
    if (!fm_metal_is_initialized() || !w.buffer || !x.buffer || !out.buffer) return;
    if (rows <= 0 || cols <= 0 || (cols % FM_Q8_0_BLOCK_WEIGHTS) != 0) return;

    fm_bufref refs[] = { w, x, out };
    uint32_t consts[] = { (uint32_t)rows, (uint32_t)cols,
                          (uint32_t)row_stride_bytes };
    /* One threadgroup per output row.  The weight offset is applied by the
     * binding, so the kernel's row indexing is identical whether the tensor
     * sits at the start of its own buffer or deep inside a shared mmap
     * region. */
    fm_dispatch_threadgroups_refs(@"matvec_q8_0", NULL, refs, 3, consts, 3,
                                  rows, FM_Q8_0_LANES_PER_ROW);
}

NSString *fm_kernel_msl_matvec_q8_0(void)
{
    return @
    "/* ============================================================\n"
    " * Q8_0 fused dequant + matvec (no scratch roundtrip)\n"
    " *\n"
    " * One threadgroup (1 SIMD group = 32 lanes) per output row.  The\n"
    " * weight stays GPU-resident as standard GGUF Q8_0 blocks: 34 bytes\n"
    " * per 32 weights, an F16 scale d followed by 32 int8 quants, and\n"
    " * value[i] = d * q[i].  Nothing is repacked — these are the bytes\n"
    " * the loader mmaps.\n"
    " *\n"
    " * Lane l owns element l of the block the row is currently on, so\n"
    " * every lane is busy for any cols that is a multiple of 32 (a\n"
    " * block-per-lane split would idle half the SIMD group on the 512-col\n"
    " * shared-expert weights) and both the quant and the activation reads\n"
    " * are contiguous across the group.  The 32 lanes read the same scale\n"
    " * bytes, which the cache broadcasts.\n"
    " *\n"
    " * The scale is assembled from two uchar loads rather than read\n"
    " * through a `device const half*`: a 34-byte block stride means a\n"
    " * block's address carries no alignment stronger than 2, and paying\n"
    " * two byte loads per 32 weights costs nothing next to the quants.\n"
    " * ============================================================ */\n"
    "\n"
    "kernel void matvec_q8_0(\n"
    "    device const uchar* W        [[buffer(0)]],\n"
    "    device const float* x        [[buffer(1)]],\n"
    "    device float*       out      [[buffer(2)]],\n"
    "    constant uint& rows          [[buffer(3)]],\n"
    "    constant uint& cols          [[buffer(4)]],\n"
    "    constant uint& row_stride    [[buffer(5)]],\n"
    "    uint row  [[threadgroup_position_in_grid]],\n"
    "    uint lane [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    if (row >= rows) return;\n"
    "    uint n_blocks = cols / 32u;\n"
    "    device const uchar* base = W + (ulong)row * row_stride;\n"
    "\n"
    "    float acc = 0.0f;\n"
    "    for (uint b = 0; b < n_blocks; b++) {\n"
    "        device const uchar* blk = base + (ulong)b * 34u;\n"
    "        ushort dbits = (ushort)blk[0] | ((ushort)blk[1] << 8);\n"
    "        float  d     = float(as_type<half>(dbits));\n"
    "        device const char* q = (device const char*)(blk + 2);\n"
    "        acc += d * float(q[lane]) * x[b * 32u + lane];\n"
    "    }\n"
    "\n"
    "    float row_sum = simd_sum(acc);\n"
    "    if (lane == 0) out[row] = row_sum;\n"
    "}\n"
    "\n"
    ;
}
