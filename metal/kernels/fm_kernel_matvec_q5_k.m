/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Portions modelled on ggml (MIT License, Copyright (c) 2023-2026 The ggml
 * authors, https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file fm_kernel_matvec_q5_k.m
 * @brief Host dispatch and MSL source for the GGUF Q5_K matvec kernel.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_matvec_q5_k.h"

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_matvec_lane_split.h"
#include "metal/kernels/fm_kernel_registry.h"

void fm_matvec_q5k(fm_bufref w, fm_bufref x, fm_bufref out,
                   int rows, int cols, int row_stride_bytes)
{
    if (!fm_metal_is_initialized() || !w.buffer || !x.buffer || !out.buffer) return;
    if (rows <= 0 || cols <= 0 || cols % FM_Q5K_WEIGHTS_PER_BLOCK != 0) return;
    if (row_stride_bytes <= 0 || row_stride_bytes % 2 != 0) return;

    fm_bufref refs[] = { w, x, out };
    uint32_t consts[] = { (uint32_t)rows, (uint32_t)cols,
                          (uint32_t)row_stride_bytes };

    /* The lane unit is a whole 256-weight SUPER-BLOCK, so the block's `d`,
     * `dmin` and packed 6-bit scale/min pairs are decoded exactly once per
     * super-block.  Splitting a row into 32-weight sub-blocks instead made
     * eight lanes re-decode one header, which is what pinned this kernel at
     * ~38 GB/s against a ~90 GB/s roof.
     * The lanes that decomposition was buying come from OTHER ROWS now: a row
     * gets as many lanes as it has super-blocks and 32/lanes rows share the
     * SIMD group.
     *
     * Specializing the per-row super-block count makes the kernel's outer loop
     * compile-time bounded, so it unrolls against a fixed lane stride instead
     * of testing a runtime bound every iteration.  The lane count is
     * specialized for the same reason, and because 32/lanes, the lane mask and
     * the reduction butterfly all fold once it is known.  One pipeline is
     * cached per distinct column count, and the model has only a handful. */
    int blocks_per_row = cols / FM_Q5K_WEIGHTS_PER_BLOCK;
    int lanes = fm_matvec_lanes_per_row(blocks_per_row);
    fm_fc_values fc;
    fm_fc_reset(&fc);
    fm_fc_set_int(&fc, FM_FC_INDEX_Q5K_BLOCKS_PER_ROW, blocks_per_row);
    if (fm_fc_set_int(&fc, FM_FC_INDEX_Q5K_LANES_PER_ROW, lanes) != 0) return;

    fm_dispatch_threadgroups_refs(@"matvec_q5_k", &fc, refs, 3, consts, 3,
                                  fm_matvec_threadgroup_count(rows, lanes),
                                  FM_Q5K_THREADS_PER_ROW);
}

NSString *fm_kernel_msl_matvec_q5_k(void)
{
    return @
    "/* ============================================================\n"
    " * Q5_K fused dequant + matvec (no scratch roundtrip)\n"
    " *\n"
    " * Standard GGUF Q5_K super-block, 176 bytes / 256 weights:\n"
    " *   +0   half  d           super-block scale for the 6-bit scales\n"
    " *   +2   half  dmin        super-block scale for the 6-bit mins\n"
    " *   +4   uchar scales[12]  8 (scale, min) pairs packed 6 bits each\n"
    " *   +16  uchar qh[32]      one high bit per weight\n"
    " *   +48  uchar qs[128]     the low 4 bits, two weights per byte\n"
    " *\n"
    " * Weight i sits in sub-block s = i/32 at lane l = i%32:\n"
    " *   nibble = (s&1) ? qs[32*(s/2)+l] >> 4 : qs[32*(s/2)+l] & 0xF\n"
    " *   q      = nibble + 16*((qh[l] >> s) & 1)\n"
    " *   value  = d*scale[s]*q - dmin*min[s]\n"
    " *\n"
    " * One threadgroup is one SIMD group (32 lanes) and covers\n"
    " * 32/fmq5k_lanes output rows: a row gets as many lanes as it has\n"
    " * 256-weight SUPER-BLOCKS, capped at 32.  A lane owns a whole super-block,\n"
    " * so d/dmin and the 8 packed 6-bit scale/min pairs are decoded ONCE per\n"
    " * super-block — giving a row more lanes than it has blocks is what makes\n"
    " * eight lanes re-decode one header, and that is the cost this avoids.\n"
    " * A shuffle butterfly reduces each row's partials.\n"
    " *\n"
    " * The (p, l) walk pairs sub-blocks 2p and 2p+1 because they share the same\n"
    " * 32 qs bytes and the same 32 qh bytes: one byte pair yields two weights.\n"
    " * ============================================================ */\n"
    "\n"
    "constant int fmq5k_blocks_per_row_in [[function_constant(50)]];\n"
    "constant int fmq5k_blocks_per_row =\n"
    "    is_function_constant_defined(fmq5k_blocks_per_row_in)\n"
    "        ? fmq5k_blocks_per_row_in : 0;\n"
    "\n"
    "/* Lanes one output row gets; 32 (one row per SIMD group) unspecialized. */\n"
    "constant int fmq5k_lanes_in [[function_constant(51)]];\n"
    "constant uint fmq5k_lanes =\n"
    "    is_function_constant_defined(fmq5k_lanes_in) ? uint(fmq5k_lanes_in) : 32u;\n"
    "\n"
    "kernel void matvec_q5_k(\n"
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
    "    const uint n_blocks = fmq5k_blocks_per_row > 0\n"
    "        ? uint(fmq5k_blocks_per_row) : cols / 256u;\n"
    "    const uint lanes = fmq5k_lanes;\n"
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
    "            device const uchar* blk = base + (ulong)b * 176u;\n"
    "            device const half*  dh  = (device const half*)blk;\n"
    "            float d    = float(dh[0]);\n"
    "            float dmin = float(dh[1]);\n"
    "            device const uchar* sr = blk + 4;       // scales[12]\n"
    "            device const uchar* qh = blk + 16;      // qh[32]\n"
    "            device const uchar* qs = blk + 48;      // qs[128]\n"
    "\n"
    "            // 6-bit scale/min: the first four pairs are stored plainly, the\n"
    "            // last four borrow their top 2 bits from the first four.\n"
    "            uchar sc[8]; uchar mn[8];\n"
    "            for (uint j = 0; j < 4; j++) { sc[j] = sr[j] & 63; mn[j] = sr[j+4] & 63; }\n"
    "            for (uint j = 4; j < 8; j++) {\n"
    "                sc[j] = (sr[j+4] & 0xF) | ((sr[j-4] >> 6) << 4);\n"
    "                mn[j] = (sr[j+4] >> 4)  | ((sr[j]   >> 6) << 4);\n"
    "            }\n"
    "\n"
    "            device const float* xb = x + (ulong)b * 256u;\n"
    "            for (uint p = 0; p < 4; p++) {  // sub-blocks 2p, 2p+1 share 32 qs bytes\n"
    "                float step_lo = d * float(sc[2*p]);\n"
    "                float off_lo  = dmin * float(mn[2*p]);\n"
    "                float step_hi = d * float(sc[2*p+1]);\n"
    "                float off_hi  = dmin * float(mn[2*p+1]);\n"
    "                device const uchar* q = qs + p * 32u;\n"
    "                uint o = p * 64u;\n"
    "                for (uint l = 0; l < 32; l++) {\n"
    "                    uint qb = uint(q[l]);\n"
    "                    uint hb = uint(qh[l]);\n"
    "                    float q_lo = float((qb & 0xFu) + (((hb >> (2u*p))      & 1u) << 4));\n"
    "                    float q_hi = float((qb >> 4)   + (((hb >> (2u*p + 1u)) & 1u) << 4));\n"
    "                    acc += (step_lo * q_lo - off_lo) * xb[o + l];\n"
    "                    acc += (step_hi * q_hi - off_hi) * xb[o + 32u + l];\n"
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
