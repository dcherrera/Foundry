/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Portions modelled on ggml (MIT License, Copyright (c) 2023-2026 The ggml
 * authors, https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file fm_kernel_matvec_q6_k.m
 * @brief Host dispatch and MSL source for the GGUF Q6_K matvec kernel.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_matvec_q6_k.h"

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_matvec_lane_split.h"
#include "metal/kernels/fm_kernel_registry.h"

void fm_matvec_q6k(fm_bufref w, fm_bufref x, fm_bufref out,
                   int rows, int cols, int row_stride_bytes)
{
    if (!fm_metal_is_initialized() || !w.buffer || !x.buffer || !out.buffer) return;
    if (rows <= 0 || cols <= 0 || cols % FM_Q6K_WEIGHTS_PER_BLOCK != 0) return;
    if (row_stride_bytes <= 0 || row_stride_bytes % 2 != 0) return;

    fm_bufref refs[] = { w, x, out };
    uint32_t consts[] = { (uint32_t)rows, (uint32_t)cols,
                          (uint32_t)row_stride_bytes };

    /* The lane unit is a whole 256-weight SUPER-BLOCK, so the block's `d` and
     * its 16 signed scales are read exactly once per super-block.  Splitting a
     * row into 32-weight quarters instead made eight lanes re-read one header,
     * which is what pinned this kernel at ~52 GB/s against a ~90 GB/s roof.
     *  The lanes that decomposition
     * was buying come from OTHER ROWS now: a row gets as many lanes as it has
     * super-blocks and 32/lanes rows share the SIMD group.
     *
     * Specializing the per-row super-block count makes the kernel's outer loop
     * compile-time bounded, so it unrolls against a fixed lane stride instead
     * of testing a runtime bound every iteration.  The lane count is
     * specialized for the same reason, and because 32/lanes, the lane mask and
     * the reduction butterfly all fold once it is known.  One pipeline is
     * cached per distinct column count, and the model has only a handful. */
    int blocks_per_row = cols / FM_Q6K_WEIGHTS_PER_BLOCK;
    int lanes = fm_matvec_lanes_per_row(blocks_per_row);
    fm_fc_values fc;
    fm_fc_reset(&fc);
    fm_fc_set_int(&fc, FM_FC_INDEX_Q6K_BLOCKS_PER_ROW, blocks_per_row);
    if (fm_fc_set_int(&fc, FM_FC_INDEX_Q6K_LANES_PER_ROW, lanes) != 0) return;

    fm_dispatch_threadgroups_refs(@"matvec_q6_k", &fc, refs, 3, consts, 3,
                                  fm_matvec_threadgroup_count(rows, lanes),
                                  FM_Q6K_THREADS_PER_ROW);
}

NSString *fm_kernel_msl_matvec_q6_k(void)
{
    return @
    "/* ============================================================\n"
    " * Q6_K fused dequant + matvec (no scratch roundtrip)\n"
    " *\n"
    " * Standard GGUF Q6_K super-block, 210 bytes / 256 weights:\n"
    " *   +0    uchar ql[128]     the low 4 bits, two weights per byte\n"
    " *   +128  uchar qh[64]      the high 2 bits, four weights per byte\n"
    " *   +192  char  scales[16]  signed 8-bit scale per 16 weights\n"
    " *   +208  half  d           super-block scale\n"
    " *\n"
    " * Weight i splits as half h = i/128, r = i%128, quarter u = r/32,\n"
    " * lane l = r%32:\n"
    " *   packed = ql[64*h + 32*(u&1) + l]\n"
    " *   nibble = (u<2) ? packed & 0xF : packed >> 4\n"
    " *   high   = (qh[32*h + l] >> (2*u)) & 3\n"
    " *   q      = int(nibble | (high << 4)) - 32      // signed, [-32, 31]\n"
    " *   value  = d * scales[8*h + 2*u + l/16] * q\n"
    " *\n"
    " * One threadgroup is one SIMD group (32 lanes) and covers\n"
    " * 32/fmq6k_lanes output rows: a row gets as many lanes as it has\n"
    " * 256-weight SUPER-BLOCKS, capped at 32.  A lane owns a whole super-block,\n"
    " * so `d` and the 16 signed scales are read ONCE per super-block — giving a\n"
    " * row more lanes than it has blocks is what makes eight lanes re-read one\n"
    " * header, and that is the cost this avoids.  A shuffle butterfly reduces\n"
    " * each row's partials.\n"
    " *\n"
    " * The (h, j, l) walk pairs quarters j and j+2 because they share the same\n"
    " * 32 ql bytes, and hoists qh[l] because all four quarters of a half share\n"
    " * it: one ql byte plus one qh byte yields two weights.\n"
    " * ============================================================ */\n"
    "\n"
    "constant int fmq6k_blocks_per_row_in [[function_constant(60)]];\n"
    "constant int fmq6k_blocks_per_row =\n"
    "    is_function_constant_defined(fmq6k_blocks_per_row_in)\n"
    "        ? fmq6k_blocks_per_row_in : 0;\n"
    "\n"
    "/* Lanes one output row gets; 32 (one row per SIMD group) unspecialized. */\n"
    "constant int fmq6k_lanes_in [[function_constant(61)]];\n"
    "constant uint fmq6k_lanes =\n"
    "    is_function_constant_defined(fmq6k_lanes_in) ? uint(fmq6k_lanes_in) : 32u;\n"
    "\n"
    "kernel void matvec_q6_k(\n"
    "    device const uchar* W        [[buffer(0)]],\n"
    "    device const float* x        [[buffer(1)]],\n"
    "    device float*       out      [[buffer(2)]],\n"
    "    constant uint& rows          [[buffer(3)]],\n"
    "    constant uint& cols          [[buffer(4)]],\n"
    "    constant uint& row_stride    [[buffer(5)]],\n"
    "    uint group [[threadgroup_position_in_grid]],\n"
    "    uint tid   [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    // Compile-time when specialized, so the loop below unrolls.\n"
    "    const uint n_blocks = fmq6k_blocks_per_row > 0\n"
    "        ? uint(fmq6k_blocks_per_row) : cols / 256u;\n"
    "    const uint lanes = fmq6k_lanes;\n"
    "    uint lane_in_row = tid & (lanes - 1u);\n"
    "    uint row = group * (32u / lanes) + tid / lanes;\n"
    "\n"
    "    // Lanes owning a row past the end accumulate nothing but still enter the\n"
    "    // reduction, so all 32 lanes are active at the shuffles.  The butterfly\n"
    "    // never leaves a row's own lanes: the run is based at a multiple of a\n"
    "    // power-of-two width, so lane ^ offset stays inside it.\n"
    "    float acc = 0.0f;\n"
    "    if (row < rows) {\n"
    "        device const uchar* base = W + (ulong)row * row_stride;\n"
    "        for (uint b = lane_in_row; b < n_blocks; b += lanes) {\n"
    "            device const uchar* blk = base + (ulong)b * 210u;\n"
    "            device const half*  dh  = (device const half*)(blk + 208u);\n"
    "            device const char*  sc  = (device const char*)(blk + 192u);\n"
    "            float d = float(dh[0]);\n"
    "            device const float* xb = x + (ulong)b * 256u;\n"
    "\n"
    "            for (uint h = 0; h < 2u; h++) {   // 128 weights per half\n"
    "                device const uchar* ql = blk + 64u * h;\n"
    "                device const uchar* qh = blk + 128u + 32u * h;\n"
    "                device const float* xh = xb + 128u * h;\n"
    "                // step[2*quarter + (l >= 16)] — one signed scale per 16 weights.\n"
    "                float step[8];\n"
    "                for (uint k = 0; k < 8u; k++) step[k] = d * float(sc[8u*h + k]);\n"
    "\n"
    "                for (uint j = 0; j < 2u; j++) {  // quarters j, j+2 share 32 ql bytes\n"
    "                    device const uchar* qlj = ql + 32u * j;\n"
    "                    uint k_lo = 2u * j;          // quarter j\n"
    "                    uint k_hi = 2u * (j + 2u);   // quarter j+2\n"
    "                    for (uint l = 0; l < 32u; l++) {\n"
    "                        uint packed = uint(qlj[l]);\n"
    "                        uint hb     = uint(qh[l]);\n"
    "                        uint s      = l >> 4;\n"
    "                        int q_lo = int((packed & 0xFu) | (((hb >> k_lo) & 3u) << 4)) - 32;\n"
    "                        int q_hi = int((packed >> 4)   | (((hb >> k_hi) & 3u) << 4)) - 32;\n"
    "                        acc += step[k_lo + s] * float(q_lo) * xh[32u * j + l];\n"
    "                        acc += step[k_hi + s] * float(q_hi) * xh[32u * (j + 2u) + l];\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    float row_sum = acc;\n"
    "    if (lanes == 32u) {\n"
    "        row_sum = simd_sum(acc);\n"
    "    } else {\n"
    "        for (uint off = lanes >> 1; off > 0u; off >>= 1)\n"
    "            row_sum += simd_shuffle_xor(row_sum, ushort(off));\n"
    "    }\n"
    "    if (lane_in_row == 0u && row < rows) out[row] = row_sum;\n"
    "}\n"
    "\n"
    ;
}
