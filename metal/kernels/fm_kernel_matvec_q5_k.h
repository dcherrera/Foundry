/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Portions modelled on ggml (MIT License, Copyright (c) 2023-2026 The ggml
 * authors, https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file fm_kernel_matvec_q5_k.h
 * @brief GGUF Q5_K family: fused dequant + matvec against the standard 176-byte /
 *   256-weight Q5_K block. Weights stay GPU-resident as raw super-blocks, so
 *   only ~0.6875 bytes/weight are read.
 *
 *   Decoded from the STANDARD GGUF block layout, not from the CUDA kernels'
 *   152/184/216-byte repack — that repack exists to dodge Kepler's integer-op
 *   penalty and has no analog on Apple GPUs.
 *   Keeping the device bytes identical to what the loader mmaps is what makes
 *   the zero-copy fm_region upload possible at all.
 *
 * @status fm_matvec_q5k is complete and takes fm_bufref operands, so it reads a
 *   weight living at a nonzero offset inside a shared mmap region.
 * @issues None
 * @todo
 *   - [ ] fm_matvec_q5k_nx — batch the matvec over N activation columns (x[n,cols] -> out[n,rows]).
 */

#ifndef FOUNDRY_METAL_KERNEL_MATVEC_Q5_K_H
#define FOUNDRY_METAL_KERNEL_MATVEC_Q5_K_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bytes and weights in one standard GGUF Q5_K super-block. */
#define FM_Q5K_BLOCK_BYTES        176
#define FM_Q5K_WEIGHTS_PER_BLOCK  256

/* Threads per threadgroup. Exactly one SIMD group, because the row reduction
 * uses SIMD-group primitives.  A threadgroup covers several output rows when a
 * row has too few super-blocks to fill it — see fm_kernel_matvec_lane_split.h. */
#define FM_Q5K_THREADS_PER_ROW    32

/* Function-constant indices this family specializes on: the super-blocks per
 * weight row, and the lanes one output row gets.  Indices live in ONE namespace
 * shared by every kernel family in the library, so a new family must claim an
 * unused one: 0-2 belong to the probe kernel, 30-31 to Q3_K, 50-51 to Q5_K,
 * 60-61 to Q6_K, 70 to Q8_0, 80-81 to Q4_K. */
#define FM_FC_INDEX_Q5K_BLOCKS_PER_ROW  50
#define FM_FC_INDEX_Q5K_LANES_PER_ROW   51

/**
 * Q5_K quantized matrix-vector product (fused dequant + dot, no scratch
 * roundtrip), reading each operand at a byte offset.
 *
 * `w` normally comes from fm_region_bind and points at the tensor's first
 * super-block inside the one no-copy region buffer; the kernel addresses row r
 * at w.offset + r * row_stride_bytes. `x` and `out` may likewise be slices of a
 * staging buffer. Pass fm_bufref_at(buffer, 0) for a whole-buffer operand.
 *
 * One threadgroup of FM_Q5K_THREADS_PER_ROW lanes, shared by
 * fm_matvec_rows_per_threadgroup() output rows: a row gets as many lanes as it
 * has 256-weight super-blocks (capped at 32), each lane dequantizes whole
 * super-blocks strided by that lane count and FMA-accumulates against x, then a
 * shuffle butterfly reduces the row's partials.  Keeping the super-block whole
 * is what makes its header a once-per-block cost.  At cols >= 8192 a row owns
 * the whole group and this is the original one-row-per-threadgroup geometry.
 *
 * @param w                 Q5_K super-block bytes, [rows, cols]
 * @param x                 [cols] float32 input
 * @param out               [rows] float32 output
 * @param rows              Output row count
 * @param cols              Weight column count; must be a multiple of 256
 * @param row_stride_bytes  Bytes per weight row, normally (cols/256)*176.
 *                          Must be even — the block's leading `d`/`dmin` are
 *                          halves, and an odd stride would misalign them.
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_matvec_q5k(fm_bufref w, fm_bufref x, fm_bufref out,
                   int rows, int cols, int row_stride_bytes);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_MATVEC_Q5_K_H */
