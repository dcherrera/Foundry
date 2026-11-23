/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   fm_kernel_matvec_q3_k.h
 * @brief  GGUF Q3_K family: fused dequant + matvec against 110-byte / 256-weight blocks.
 *
 * Written from a behaviour specification; no ggml or
 * llama.cpp source was consulted.
 *
 * @status None
 * @issues _nx gives one row a whole SIMD group and strides blocks by 32, so narrow rows
 *         (cols = 2048 is 8 blocks) leave most lanes idle.
 * @todo   None
 */

#ifndef FOUNDRY_METAL_KERNEL_MATVEC_Q3_K_H
#define FOUNDRY_METAL_KERNEL_MATVEC_Q3_K_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bytes in one Q3_K super-block: hmask[32] + qs[64] + scales[12] + F16 d. */
#define FM_Q3_K_BLOCK_BYTES   110

/* Weights in one Q3_K super-block. */
#define FM_Q3_K_BLOCK_WEIGHTS 256

/* Widest activation-column tile one _nx dispatch applies a decoded block to. */
#define FM_Q3_K_NX_MAX_COLS   8

/* Function-constant index (MSL int): the _nx tile width, 1..FM_Q3_K_NX_MAX_COLS. */
#define FM_Q3_K_FC_NX_COLS       30

/* Function-constant index (MSL int): lanes one output row gets in matvec_q3k. */
#define FM_Q3_K_FC_LANES_PER_ROW 31

/**
 * Q3_K matrix-vector product: out[r] = dequant(W row r) . x.
 *
 * The weight stays as the raw GGUF blocks (no repack). Row r begins at byte
 * w.offset + r * row_stride_bytes, so a tensor inside a shared no-copy region is
 * bound by offset. A row gets min(32, pow2_floor(cols / 256)) lanes and a SIMD
 * group carries 32 / lanes rows. Does nothing (outputs untouched) when the backend
 * is not initialized, an operand buffer is NULL, rows or cols is not positive,
 * cols is not a multiple of 256, or row_stride_bytes < (cols / 256) * 110.
 *
 * @param w                 Q3_K blocks, rows x (cols / 256)
 * @param x                 [cols] float32 input
 * @param out               [rows] float32 output; nothing past rows is written
 * @param rows              Output rows
 * @param cols              Input length, multiple of 256
 * @param row_stride_bytes  Bytes between row starts (padding is skipped)
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_matvec_q3k_at(fm_bufref w, fm_bufref x, fm_bufref out,
                      int rows, int cols, int row_stride_bytes);

/**
 * fm_matvec_q3k_at with every operand bound at offset 0. Does nothing if any
 * buffer is NULL.
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_matvec_q3k(fm_buffer *w, fm_buffer *x, fm_buffer *out,
                   int rows, int cols, int row_stride_bytes);

/**
 * Q3_K matrix times n token-major vectors: out[t * rows + r] = dequant(W row r) .
 * X[t * cols ..] for t < n. Each decoded block is applied to a tile of up to 8
 * columns; n is covered by n / 8 full tiles in one dispatch, then one dispatch for
 * the n % 8 remainder. Rejects as fm_matvec_q3k_at, plus n <= 0.
 *
 * @param w                 Q3_K blocks, rows x (cols / 256)
 * @param x                 [n][cols] float32 input
 * @param out               [n][rows] float32 output
 * @param rows              Output rows
 * @param cols              Input length, multiple of 256
 * @param row_stride_bytes  Bytes between row starts
 * @param n                 Number of input vectors
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_matvec_q3k_nx_at(fm_bufref w, fm_bufref x, fm_bufref out,
                         int rows, int cols, int row_stride_bytes, int n);

/**
 * fm_matvec_q3k_nx_at with every operand bound at offset 0. Does nothing if any
 * buffer is NULL.
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_matvec_q3k_nx(fm_buffer *w, fm_buffer *x, fm_buffer *out,
                      int rows, int cols, int row_stride_bytes, int n);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_MATVEC_Q3_K_H */
