/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Portions modelled on ggml (MIT License, Copyright (c) 2023-2026 The ggml
 * authors, https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file fm_kernel_matvec_q4_k.h
 * @brief GGUF Q4_K family: fused dequant + matvec, a tiled simdgroup_matrix
 *   batched-decode GEMM, and a dequantize-to-F16 prefill GEMM. Weights stay
 *   GPU-resident as raw 144-byte / 256-weight super-blocks.
 *
 * @status Every entry point has an offset-bearing (_at) form, so any of them
 *   can read a weight that lives inside an fm_region.  The plain fm_buffer
 *   forms are kept as offset-0 wrappers for callers that own their weights.
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_METAL_KERNEL_MATVEC_Q4_K_H
#define FOUNDRY_METAL_KERNEL_MATVEC_Q4_K_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Q4_K quantized matrix-vector product (fused dequant + dot, no scratch).
 * w: raw Q4_K super-block bytes [rows, cols] (144 bytes / 256 cols per block),
 *    row_stride_bytes = (cols/256)*144.
 * x: [cols] float32 input, out: [rows] float32 output.
 * One SIMD group covers several output rows when a row has too few
 * super-blocks to fill it (fm_kernel_matvec_lane_split.h); weights stay
 * GPU-resident and quantized so only ~0.5625 bytes/weight are read (Q4_K),
 * not F16/F32. */
void fm_matvec_q4_k(fm_buffer *w, fm_buffer *x, fm_buffer *out,
                    int rows, int cols, int row_stride_bytes);

/**
 * Q4_K matvec reading each operand at a byte offset — the form a weight
 * inside a shared mmap region needs.
 *
 * `w` normally comes from fm_region_bind and points at the tensor's
 * first super-block inside the one no-copy region buffer; the kernel
 * addresses row r at w.offset + r * row_stride_bytes.  `x` and `out` may
 * likewise be slices of a staging buffer.  fm_matvec_q4_k is exactly
 * this with all three offsets zero.
 *
 * @param w                 Q4_K super-block bytes, [rows, cols]
 * @param x                 [cols] float32 input
 * @param out               [rows] float32 output
 * @param rows, cols        Weight shape; cols must be a multiple of 256
 * @param row_stride_bytes  Bytes per weight row, (cols/256)*144
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_matvec_q4_k_at(fm_bufref w, fm_bufref x, fm_bufref out,
                       int rows, int cols, int row_stride_bytes);

/* Bytes in one GGUF Q4_K super-block: d, dmin, scales[12], qs[128]. */
#define FM_Q4_K_BLOCK_BYTES   144

/* Weights per GGUF Q4_K super-block. */
#define FM_Q4_K_BLOCK_WEIGHTS 256

/* Function-constant indices this family reserves in the shared MSL
 * translation unit: the column-tile width of matvec_q4_k_nx, and the lanes
 * one output row gets in both matvec kernels.
 *
 * Indices are GLOBAL across the assembled library — every fragment is
 * concatenated into one translation unit, so two families declaring the
 * same index is a whole-library compile failure that breaks every kernel
 * in the backend, not just the two that collided.  Allocation:
 *   0..2  probe   30..31  Q3_K   50..51  Q5_K   60..61  Q6_K   70  Q8_0
 *   80..81  Q4_K */
#define FM_Q4_K_NX_TILE_CONSTANT_INDEX 80
#define FM_Q4_K_FC_LANES_PER_ROW       81

/* Widest column tile matvec_q4_k_nx will specialize for.  The kernel keeps
 * one accumulator per column of a tile in registers, so this bounds the
 * register cost, not the batch size. */
#define FM_Q4_K_NX_MAX_TILE 8

/**
 * Q4_K matvec batched over n activation columns — the decode-batch /
 * prefill form of fm_matvec_q4_k_at.
 *
 * out[t][rows] = dequant(w) . x[t] for t in [0, n).  x is [n, cols] and
 * out is [n, rows], both token-major (token t is contiguous).  Note the
 * batch axis is the ACTIVATION COLUMNS, not the weight rows — rows are
 * already the kernel's parallel axis.
 *
 * Each super-block is decoded once per column tile and reused across that
 * tile's columns, so the dequant work amortizes over the batch instead of
 * being repeated per column.  The tile width is a function constant, which
 * is what keeps the per-column accumulators in registers: a runtime bound
 * would leave the array in thread-local memory and cost more than the
 * batching saves.  n is split into full FM_Q4_K_NX_MAX_TILE tiles plus at
 * most one narrower remainder tile, so a call is at most two dispatches.
 *
 * Bit-exact per column: column t of the output is bit-identical to
 * fm_matvec_q4_k_at called on x[t] alone.  The per-lane block assignment,
 * the accumulation order inside a block and the SIMD reduction are the
 * same, so batching changes only how many accumulators a lane carries.
 *
 * Offsets, as everywhere in this family: w normally comes from
 * fm_region_bind; x and out may be slices of a staging buffer.
 *
 * @param w                 Q4_K super-block bytes, [rows, cols]
 * @param x                 [n, cols] float32 activations
 * @param out               [n, rows] float32 output
 * @param rows, cols        Weight shape; cols must be a multiple of 256
 * @param row_stride_bytes  Bytes per weight row, (cols/256)*144
 * @param n                 Activation columns; must be >= 1
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_matvec_q4_k_nx(fm_bufref w, fm_bufref x, fm_bufref out,
                       int rows, int cols, int row_stride_bytes, int n);

/* Q4_K x B GEMM (batched decode). out[B, rows] = x[B, cols] @ dequant(w)^T.
 * w: raw Q4_K super-block bytes [rows, cols], row_stride_bytes = (cols/256)*144.
 * x: [B, cols] float32, out: [B, rows] float32. Tiled simdgroup_matrix GEMM:
 * one threadgroup per 64-row tile (256 threads / 8 SIMD groups). Each 64-column
 * slice stages the batch activations into threadgroup memory once (coalesced)
 * and cooperatively dequantizes the Q4_K weights into a transposed tile, both
 * reused across every batch and row sub-tile; the inner product runs on
 * simdgroup_float8x8 with F32 accumulation. The Q4_K dequant is amortized
 * across the batch (the GPU analog of a CPU AMX panel). B is padded to a
 * multiple of 8 internally (padded rows never stored); tested up to B = 32.
 * ~8x over a naive one-row-per-SIMD-group kernel, and every tested shape clears
 * the CPU AMX per-weight budget in isolation. */
void fm_sgemm_q4_k(fm_buffer *w, fm_buffer *x, fm_buffer *out,
                   int rows, int cols, int row_stride_bytes, int B);

/**
 * fm_sgemm_q4_k reading each operand at a byte offset.
 *
 * Same kernel and same contract; the only difference is that the weight
 * may live inside an fm_region instead of owning its buffer.
 * fm_sgemm_q4_k is exactly this with all three offsets zero.
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_sgemm_q4_k_at(fm_bufref w, fm_bufref x, fm_bufref out,
                      int rows, int cols, int row_stride_bytes, int B);

/* Large-M Q4_K GEMM for PREFILL: out[M, N] = x[M, K] @ dequant(w)[N, K]^T.
 * Computed on the GPU by dequantizing the Q4_K weight to F16 and running an
 * F16 MPS matmul (F16 has ~2x the GPU throughput of F32, and prefill's post-
 * RMSNorm activations are bounded, so F16 is safe here). w is GPU-resident raw
 * Q4_K bytes [N, K] (row_stride_bytes = (K/256)*144); x is F32 [M, K]; out is
 * F32 [M, N] (all fm_buffers). Intended for large M (hundreds-thousands of
 * tokens) where MPS runs well above the CPU AMX GEMM. */
void fm_gemm_q4_k_f16(fm_buffer *w, fm_buffer *x_f32, fm_buffer *out_f32,
                      int M, int N, int K, int row_stride_bytes);

/**
 * fm_gemm_q4_k_f16 reading each operand at a byte offset.
 *
 * Same staging and same F16 MPS matmul; only the three caller-supplied
 * operands gain offsets, so the Q4_K weight may live inside an fm_region.
 * fm_gemm_q4_k_f16 is exactly this with all three offsets zero.
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_gemm_q4_k_f16_at(fm_bufref w, fm_bufref x_f32, fm_bufref out_f32,
                         int M, int N, int K, int row_stride_bytes);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_MATVEC_Q4_K_H */
