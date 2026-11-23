/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   fm_kernel_matvec_q3_k.m
 * @brief  Host dispatch and MSL source for the Q3_K matvec kernels.
 *
 * Written from a behaviour specification; no ggml or
 * llama.cpp source was consulted.
 *
 * @status None
 * @issues _nx leaves most lanes idle at narrow cols (one row per group, lanes stride
 *         blocks by 32).
 * @todo   None
 */

#include "metal/kernels/fm_kernel_matvec_q3_k.h"

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_matvec_lane_split.h"
#include "metal/kernels/fm_kernel_registry.h"

/* Shape checks shared by both entry points. The stride must hold a whole row of
 * blocks; anything wider is padding the kernel skips. */
static bool fm_q3k_shape_ok(int rows, int cols, int row_stride_bytes) {
    if (rows <= 0 || cols <= 0 || (cols % FM_Q3_K_BLOCK_WEIGHTS) != 0)
        return false;
    return row_stride_bytes >= (cols / FM_Q3_K_BLOCK_WEIGHTS) * FM_Q3_K_BLOCK_BYTES;
}

void fm_matvec_q3k_at(fm_bufref w, fm_bufref x, fm_bufref out,
                      int rows, int cols, int row_stride_bytes) {
    if (!fm_metal_is_initialized() || !w.buffer || !x.buffer || !out.buffer)
        return;
    if (!fm_q3k_shape_ok(rows, cols, row_stride_bytes))
        return;

    /* Lanes per row is a function constant so the lane mask, rows per group and
     * the butterfly fold at pipeline-build time. Set even when it is 32. */
    int lanes = fm_matvec_lanes_per_row(cols / FM_Q3_K_BLOCK_WEIGHTS);
    fm_fc_values fc;
    fm_fc_reset(&fc);
    if (fm_fc_set_int(&fc, FM_Q3_K_FC_LANES_PER_ROW, lanes) != 0)
        return;

    fm_bufref refs[] = { w, x, out };
    uint32_t consts[] = { (uint32_t)rows, (uint32_t)cols, (uint32_t)row_stride_bytes };
    fm_dispatch_threadgroups_refs(@"matvec_q3k", &fc, refs, 3, consts, 3,
                                  fm_matvec_threadgroup_count(rows, lanes),
                                  FM_MATVEC_SIMD_LANES);
}

void fm_matvec_q3k(fm_buffer *w, fm_buffer *x, fm_buffer *out,
                   int rows, int cols, int row_stride_bytes) {
    if (!w || !x || !out)
        return;
    fm_matvec_q3k_at(fm_bufref_at(w, 0), fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                     rows, cols, row_stride_bytes);
}

/* One _nx dispatch: `tiles` tiles of `tile_cols` columns starting at column col0,
 * one threadgroup per (row, tile). */
static void fm_q3k_nx_dispatch(const fm_bufref *refs, int rows, int cols,
                               int row_stride_bytes, int n, int tile_cols,
                               int col0, int tiles) {
    fm_fc_values fc;
    fm_fc_reset(&fc);
    if (fm_fc_set_int(&fc, FM_Q3_K_FC_NX_COLS, tile_cols) != 0)
        return;

    uint32_t consts[] = { (uint32_t)rows, (uint32_t)cols, (uint32_t)row_stride_bytes,
                          (uint32_t)n, (uint32_t)col0, (uint32_t)tiles };
    fm_dispatch_threadgroups_refs(@"matvec_q3k_nx", &fc, refs, 3, consts, 6,
                                  rows * tiles, FM_MATVEC_SIMD_LANES);
}

void fm_matvec_q3k_nx_at(fm_bufref w, fm_bufref x, fm_bufref out,
                         int rows, int cols, int row_stride_bytes, int n) {
    if (!fm_metal_is_initialized() || !w.buffer || !x.buffer || !out.buffer)
        return;
    if (!fm_q3k_shape_ok(rows, cols, row_stride_bytes) || n <= 0)
        return;

    fm_bufref refs[] = { w, x, out };
    int full_tiles = n / FM_Q3_K_NX_MAX_COLS;
    int remainder = n % FM_Q3_K_NX_MAX_COLS;

    if (full_tiles > 0)
        fm_q3k_nx_dispatch(refs, rows, cols, row_stride_bytes, n,
                           FM_Q3_K_NX_MAX_COLS, 0, full_tiles);
    if (remainder > 0)
        fm_q3k_nx_dispatch(refs, rows, cols, row_stride_bytes, n,
                           remainder, full_tiles * FM_Q3_K_NX_MAX_COLS, 1);
}

void fm_matvec_q3k_nx(fm_buffer *w, fm_buffer *x, fm_buffer *out,
                      int rows, int cols, int row_stride_bytes, int n) {
    if (!w || !x || !out)
        return;
    fm_matvec_q3k_nx_at(fm_bufref_at(w, 0), fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                        rows, cols, row_stride_bytes, n);
}

NSString *fm_kernel_msl_matvec_q3_k(void)
{
    return @
    "/* ============================================================\n"
    " * Q3_K fused dequant + matvec\n"
    " *\n"
    " * Block (110 bytes, 256 weights): hmask[32] | qs[64] | scales[12] | half d.\n"
    " * Sub-block s (16 weights) = half u, level v, group g with s = 8u + 2v + g.\n"
    " * Weight l of s: low two bits from qs[32u + 16g + l] at shift 2v, minus 4\n"
    " * unless hmask[16g + l] bit (4u + v) is set; multiplier d * (S_s - 32).\n"
    " *\n"
    " * Every group is one SIMD group; no threadgroup memory. Fast math makes\n"
    " * the bits depend on lane ownership, visit order and the reduction\n"
    " * primitive, so those are fixed: simd_sum at 32 lanes per row, an xor\n"
    " * butterfly below it.\n"
    " * ============================================================ */\n"
    "\n"
    "constant int fm_q3k_nx_cols_fc [[function_constant(30)]];\n"
    "constant int fm_q3k_nx_cols = is_function_constant_defined(fm_q3k_nx_cols_fc)\n"
    "                              ? fm_q3k_nx_cols_fc : 8;\n"
    "constant int fm_q3k_lanes_fc [[function_constant(31)]];\n"
    "constant int fm_q3k_lanes = is_function_constant_defined(fm_q3k_lanes_fc)\n"
    "                            ? fm_q3k_lanes_fc : 32;\n"
    "\n"
    "/* Six-bit scale of sub-block s, with i = s & 3 and quarter q = s >> 2: low\n"
    " * four bits from scales[i + 4(q & 1)] (low nibble for q < 2, high nibble\n"
    " * otherwise), top two bits from scales[8 + i] at shift 2q. */\n"
    "static inline int fm_q3k_six_bit_scale(device const uchar* sc, uint s) {\n"
    "    uint i = s & 3u;\n"
    "    uint q = s >> 2;\n"
    "    int lo_byte = int(sc[i + 4u * (q & 1u)]);\n"
    "    int lo = (q < 2u) ? (lo_byte & 15) : (lo_byte >> 4);\n"
    "    int hi = (int(sc[8u + i]) >> (2u * q)) & 3;\n"
    "    return lo | (hi << 4);\n"
    "}\n"
    "\n"
    "/* The 16 sub-block multipliers d * (S_s - 32) of one block. */\n"
    "static inline void fm_q3k_block_multipliers(device const uchar* blk, thread float* dl) {\n"
    "    float d = float(*(device const half*)(blk + 108));\n"
    "    for (uint s = 0; s < 16u; s++)\n"
    "        dl[s] = d * float(fm_q3k_six_bit_scale(blk + 96, s) - 32);\n"
    "}\n"
    "\n"
    "/* The 16 dequantized weights of sub-block s. */\n"
    "static inline void fm_q3k_sub_block_weights(device const uchar* blk, uint s, float dl,\n"
    "                                            thread float* w) {\n"
    "    uint u = s >> 3;\n"
    "    uint v = (s & 7u) >> 1;\n"
    "    uint g = s & 1u;\n"
    "    device const uchar* qs = blk + 32u + 32u * u + 16u * g;\n"
    "    device const uchar* hm = blk + 16u * g;\n"
    "    uint shift = 2u * v;\n"
    "    uint mask = 1u << (4u * u + v);\n"
    "    for (uint l = 0; l < 16u; l++) {\n"
    "        int q2 = int((uint(qs[l]) >> shift) & 3u);\n"
    "        int quant = (uint(hm[l]) & mask) ? q2 : q2 - 4;\n"
    "        w[l] = dl * float(quant);\n"
    "    }\n"
    "}\n"
    "\n"
    "/* One block against 256 activations into a running chain: per sub-block a\n"
    " * fresh 16-term partial, then added to acc. */\n"
    "static inline float fm_q3k_block_dot(device const uchar* blk, device const float* xb,\n"
    "                                     float acc) {\n"
    "    float dl[16];\n"
    "    fm_q3k_block_multipliers(blk, dl);\n"
    "    for (uint s = 0; s < 16u; s++) {\n"
    "        float w[16];\n"
    "        fm_q3k_sub_block_weights(blk, s, dl[s], w);\n"
    "        float part = 0.0f;\n"
    "        for (uint l = 0; l < 16u; l++)\n"
    "            part += w[l] * xb[16u * s + l];\n"
    "        acc += part;\n"
    "    }\n"
    "    return acc;\n"
    "}\n"
    "\n"
    "/* L lanes per row, 32 / L rows per SIMD group; lane k of a row takes blocks\n"
    " * k, k + L, ... Rows past the end accumulate 0 but still join the reduction. */\n"
    "kernel void matvec_q3k(\n"
    "    device const uchar* W     [[buffer(0)]],\n"
    "    device const float* x     [[buffer(1)]],\n"
    "    device float*       out   [[buffer(2)]],\n"
    "    constant uint& rows       [[buffer(3)]],\n"
    "    constant uint& cols       [[buffer(4)]],\n"
    "    constant uint& row_stride [[buffer(5)]],\n"
    "    uint tg  [[threadgroup_position_in_grid]],\n"
    "    uint tid [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    const uint L = uint(fm_q3k_lanes);\n"
    "    uint lane_in_row = tid & (L - 1u);\n"
    "    uint row = tg * (32u / L) + tid / L;\n"
    "    uint nb = cols / 256u;\n"
    "\n"
    "    float acc = 0.0f;\n"
    "    if (row < rows) {\n"
    "        device const uchar* base = W + (ulong)row * row_stride;\n"
    "        for (uint b = lane_in_row; b < nb; b += L)\n"
    "            acc = fm_q3k_block_dot(base + (ulong)b * 110u, x + (ulong)b * 256u, acc);\n"
    "    }\n"
    "\n"
    "    float row_sum;\n"
    "    if (L == 32u) {\n"
    "        row_sum = simd_sum(acc);\n"
    "    } else {\n"
    "        row_sum = acc;\n"
    "        for (uint off = L / 2u; off > 0u; off >>= 1)\n"
    "            row_sum += simd_shuffle_xor(row_sum, (ushort)off);\n"
    "    }\n"
    "    if (lane_in_row == 0u && row < rows) out[row] = row_sum;\n"
    "}\n"
    "\n"
    "/* One SIMD group per (row, tile of NC token columns). Each block is decoded\n"
    " * once per sub-block and applied to every live column of the tile. */\n"
    "kernel void matvec_q3k_nx(\n"
    "    device const uchar* W     [[buffer(0)]],\n"
    "    device const float* X     [[buffer(1)]],\n"
    "    device float*       out   [[buffer(2)]],\n"
    "    constant uint& rows       [[buffer(3)]],\n"
    "    constant uint& cols       [[buffer(4)]],\n"
    "    constant uint& row_stride [[buffer(5)]],\n"
    "    constant uint& n          [[buffer(6)]],\n"
    "    constant uint& col0       [[buffer(7)]],\n"
    "    constant uint& tiles      [[buffer(8)]],\n"
    "    uint tg  [[threadgroup_position_in_grid]],\n"
    "    uint tid [[thread_position_in_threadgroup]],\n"
    "    uint T   [[threads_per_threadgroup]]\n"
    ") {\n"
    "    const uint NC = uint(fm_q3k_nx_cols);\n"
    "    uint row = tg / tiles;\n"
    "    if (row >= rows) return;\n"
    "    uint c0 = col0 + (tg - row * tiles) * NC;\n"
    "    uint nb = cols / 256u;\n"
    "\n"
    "    float acc[8];\n"
    "    for (uint c = 0; c < NC; c++) acc[c] = 0.0f;\n"
    "\n"
    "    device const uchar* base = W + (ulong)row * row_stride;\n"
    "    for (uint b = tid; b < nb; b += T) {\n"
    "        device const uchar* blk = base + (ulong)b * 110u;\n"
    "        float dl[16];\n"
    "        fm_q3k_block_multipliers(blk, dl);\n"
    "        for (uint s = 0; s < 16u; s++) {\n"
    "            float w[16];\n"
    "            fm_q3k_sub_block_weights(blk, s, dl[s], w);\n"
    "            for (uint c = 0; c < NC; c++) {\n"
    "                if (c0 + c >= n) continue;\n"
    "                device const float* xs = X + (ulong)(c0 + c) * cols + (ulong)b * 256u\n"
    "                                         + 16u * s;\n"
    "                float part = 0.0f;\n"
    "                for (uint l = 0; l < 16u; l++)\n"
    "                    part += w[l] * xs[l];\n"
    "                acc[c] += part;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "\n"
    "    for (uint c = 0; c < NC; c++) {\n"
    "        float col_sum = simd_sum(acc[c]);\n"
    "        if (tid == 0u && c0 + c < n) out[(ulong)(c0 + c) * rows + row] = col_sum;\n"
    "    }\n"
    "}\n"
    "\n"
    ;
}
