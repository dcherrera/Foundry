/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_matvec_q8.m
 * @brief Host dispatch and MSL source for the single-scale INT8 matvec and
 *   batched GEMM kernels.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_matvec_q8.h"

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

void fm_matvec_q8(fm_buffer *x_q, fm_buffer *w, fm_buffer *w_scales,
                  fm_buffer *out, float x_scale, int rows, int cols) {
    if (!fm_metal_is_initialized()) return;

    fm_buffer *bufs[] = { x_q, w, w_scales, out };
    uint32_t consts[] = { (uint32_t)rows, (uint32_t)cols };
    /* One threadgroup per output row, 256 threads for partial dot products */
    int tpg = cols < 256 ? cols : 256;
    fm_dispatch_with_float_constant(@"matvec_q8", bufs, 4, consts, 2,
                                    x_scale, rows, tpg);
}

void fm_sgemm_q8(fm_buffer *X, fm_buffer *W_q8, fm_buffer *scales,
                 fm_buffer *Out, int M, int N, int K) {
    if (!fm_metal_is_initialized() || !X || !W_q8 || !scales || !Out) return;

    id<MTLComputePipelineState> ps = fm_get_pipeline(@"gemm_q8");
    if (!ps) return;

    id<MTLCommandBuffer> cmdbuf = fm_get_command_buffer();
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

    [enc setComputePipelineState:ps];
    [enc setBuffer:X->mtl offset:0 atIndex:0];
    [enc setBuffer:W_q8->mtl offset:0 atIndex:1];
    [enc setBuffer:scales->mtl offset:0 atIndex:2];
    [enc setBuffer:Out->mtl offset:0 atIndex:3];
    uint32_t uM = (uint32_t)M, uN = (uint32_t)N, uK = (uint32_t)K;
    [enc setBytes:&uM length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&uN length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&uK length:sizeof(uint32_t) atIndex:6];

    /* Dispatch: 256 threads (linear), grid covers BM=64 x BN=64 tiles */
    MTLSize tpg = MTLSizeMake(256, 1, 1);
    MTLSize groups = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:tpg];
    [enc endEncoding];

    fm_submit_if_not_batching();
}

NSString *fm_kernel_msl_matvec_q8(void)
{
    return @
    "/* ============================================================\n"
    " * INT8 Quantized Matrix-Vector Product\n"
    " *\n"
    " * One threadgroup per output row, threads cooperate on dot product.\n"
    " * Accumulates int32, scales at end.\n"
    " * ============================================================ */\n"
    "\n"
    "kernel void matvec_q8(\n"
    "    device const char* x_q       [[buffer(0)]],\n"
    "    device const char* w         [[buffer(1)]],\n"
    "    device const float* w_scales [[buffer(2)]],\n"
    "    device float* out            [[buffer(3)]],\n"
    "    constant uint& rows          [[buffer(4)]],\n"
    "    constant uint& cols          [[buffer(5)]],\n"
    "    constant float& x_scale      [[buffer(6)]],\n"
    "    uint row   [[threadgroup_position_in_grid]],\n"
    "    uint tid   [[thread_position_in_threadgroup]],\n"
    "    uint tsize [[threads_per_threadgroup]]\n"
    ") {\n"
    "    if (row >= rows) return;\n"
    "    uint base = row * cols;\n"
    "\n"
    "    // Partial dot product per thread\n"
    "    int local_dot = 0;\n"
    "    for (uint j = tid; j < cols; j += tsize) {\n"
    "        local_dot += int(x_q[j]) * int(w[base + j]);\n"
    "    }\n"
    "\n"
    "    // SIMD reduction\n"
    "    threadgroup int shared_dots[32];\n"
    "    int sdot = simd_sum(local_dot);\n"
    "    if (simd_is_first()) shared_dots[tid / 32] = sdot;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (tid < 32) {\n"
    "        uint n_simd = (tsize + 31) / 32;\n"
    "        sdot = (tid < n_simd) ? shared_dots[tid] : 0;\n"
    "        sdot = simd_sum(sdot);\n"
    "        if (tid == 0) {\n"
    "            out[row] = float(sdot) * w_scales[row] * x_scale;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============================================================\n"
    " * Batched INT8 GEMM with on-the-fly dequantization\n"
    " *\n"
    " * Out[M,N] = X[M,K] @ dequant(W[N,K], scales[N])^T\n"
    " * W is stored as [N, K] int8 (transposed: rows = output dim).\n"
    " * scales[N] is per-output-row.\n"
    " *\n"
    " * Each threadgroup computes a BM x BN tile of output.\n"
    " * Register-tiled INT8 GEMM with threadgroup shared memory.\n"
    " * Each thread computes a TM x TN (4x4) block of outputs.\n"
    " * 256 threads (16x16) per threadgroup, effective tile 64x64.\n"
    " * BK=32, shared mem: X_tile[64][32]=8KB + W_tile[32][64]=8KB = 16KB.\n"
    " * ============================================================ */\n"
    "\n"
    "#define BM 64\n"
    "#define BN 64\n"
    "#define BK 32\n"
    "#define TM 4\n"
    "#define TN 4\n"
    "#define BLOCK_THREADS 256\n"
    "\n"
    "kernel void gemm_q8(\n"
    "    device const float* X        [[buffer(0)]],\n"
    "    device const char*  W        [[buffer(1)]],\n"
    "    device const float* scales   [[buffer(2)]],\n"
    "    device float*       Out      [[buffer(3)]],\n"
    "    constant uint& M             [[buffer(4)]],\n"
    "    constant uint& N             [[buffer(5)]],\n"
    "    constant uint& K             [[buffer(6)]],\n"
    "    uint2 gid  [[threadgroup_position_in_grid]],\n"
    "    uint  lid  [[thread_index_in_threadgroup]]\n"
    ") {\n"
    "    // Map linear thread ID to 2D position within the tile\n"
    "    // 16 threads along N (each handles TN=4 cols), 16 along M (each handles TM=4 rows)\n"
    "    uint ty = lid / 16;   // 0..15 -> M dimension\n"
    "    uint tx = lid % 16;   // 0..15 -> N dimension\n"
    "\n"
    "    // Shared memory tiles\n"
    "    threadgroup float X_tile[BM][BK];   // 64x32x4 = 8KB\n"
    "    threadgroup float W_tile[BK][BN];   // 32x64x4 = 8KB\n"
    "\n"
    "    // Per-thread accumulators: TM x TN = 4x4 = 16 outputs\n"
    "    float acc[TM][TN] = {{0}};\n"
    "\n"
    "    // Global base position for this thread's output block\n"
    "    uint base_row = gid.y * BM + ty * TM;\n"
    "    uint base_col = gid.x * BN + tx * TN;\n"
    "\n"
    "    // Loop over K dimension in tiles of BK\n"
    "    for (uint kb = 0; kb < K; kb += BK) {\n"
    "        // Cooperative load: 256 threads load 64x32 = 2048 elements each tile\n"
    "        // Each thread loads 2048/256 = 8 elements per tile\n"
    "        // X_tile: iterate with stride of BLOCK_THREADS\n"
    "        for (uint i = lid; i < BM * BK; i += BLOCK_THREADS) {\n"
    "            uint tile_row = i / BK;\n"
    "            uint tile_col = i % BK;\n"
    "            uint g_row = gid.y * BM + tile_row;\n"
    "            uint g_col = kb + tile_col;\n"
    "            X_tile[tile_row][tile_col] = (g_row < M && g_col < K) ? X[g_row * K + g_col] : 0.0f;\n"
    "        }\n"
    "\n"
    "        // W_tile: load int8, dequantize to float\n"
    "        // W is [N, K] row-major. Load 4 bytes at once with char4 vectorized reads.\n"
    "        // Each thread loads 4 consecutive K elements per iteration.\n"
    "        for (uint i = lid; i < (BK / 4) * BN; i += BLOCK_THREADS) {\n"
    "            uint tile_k_base = (i / BN) * 4;  // K offset (groups of 4)\n"
    "            uint tile_n = i % BN;              // N offset\n"
    "            uint g_n = gid.x * BN + tile_n;\n"
    "            uint g_k = kb + tile_k_base;\n"
    "            if (g_n < N && g_k + 3 < K) {\n"
    "                // Vectorized load: 4 consecutive int8 values\n"
    "                device const char4* w4 = (device const char4*)(W + g_n * K + g_k);\n"
    "                char4 v = *w4;\n"
    "                W_tile[tile_k_base    ][tile_n] = float(v.x);\n"
    "                W_tile[tile_k_base + 1][tile_n] = float(v.y);\n"
    "                W_tile[tile_k_base + 2][tile_n] = float(v.z);\n"
    "                W_tile[tile_k_base + 3][tile_n] = float(v.w);\n"
    "            } else {\n"
    "                for (uint d = 0; d < 4; d++) {\n"
    "                    uint gk = g_k + d;\n"
    "                    W_tile[tile_k_base + d][tile_n] = (g_n < N && gk < K) ? float(W[g_n * K + gk]) : 0.0f;\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "        // Accumulate: each thread computes TM x TN outputs\n"
    "        // Load TM X values and TN W values per K step, compute outer product\n"
    "        for (uint kk = 0; kk < BK; ++kk) {\n"
    "            // Load TM values from X_tile into registers\n"
    "            float xv[TM];\n"
    "            for (uint i = 0; i < TM; i++)\n"
    "                xv[i] = X_tile[ty * TM + i][kk];\n"
    "            // Load TN values from W_tile into registers\n"
    "            float wv[TN];\n"
    "            for (uint j = 0; j < TN; j++)\n"
    "                wv[j] = W_tile[kk][tx * TN + j];\n"
    "            // Outer product accumulate\n"
    "            for (uint i = 0; i < TM; i++)\n"
    "                for (uint j = 0; j < TN; j++)\n"
    "                    acc[i][j] += xv[i] * wv[j];\n"
    "        }\n"
    "\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "\n"
    "    // Write outputs with per-row scale\n"
    "    for (uint i = 0; i < TM; i++) {\n"
    "        uint out_row = base_row + i;\n"
    "        if (out_row >= M) continue;\n"
    "        for (uint j = 0; j < TN; j++) {\n"
    "            uint out_col = base_col + j;\n"
    "            if (out_col < N)\n"
    "                Out[out_row * N + out_col] = acc[i][j] * scales[out_col];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "/* The fragments share one MSL translation unit, so these would stay\n"
    " * defined for every family concatenated after this one. */\n"
    "#undef BM\n"
    "#undef BN\n"
    "#undef BK\n"
    "#undef TM\n"
    "#undef TN\n"
    "#undef BLOCK_THREADS\n"
    "\n"
    ;
}
