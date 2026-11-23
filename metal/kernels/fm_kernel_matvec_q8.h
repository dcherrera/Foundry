/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_matvec_q8.h
 * @brief Single-scale INT8 family: matrix-vector and batched GEMM against
 *   int8 weights carrying ONE float scale per output row.
 *
 *   This is NOT the GGUF Q8_0 block layout (32 weights per block, one F16
 *   scale per block). A Q8_0 weight decoded through these kernels reads
 *   garbage — it gets its own unit, fm_kernel_matvec_q8_0.
 *
 * @status Both kernels bind whole buffers at offset 0.
 * @issues Cannot read a weight that lives inside an fm_region — needs the
 *   fm_bufref form before it can serve a region-resident model.
 * @todo
 *   - [ ] Convert fm_matvec_q8 and fm_sgemm_q8 to fm_bufref operands.
 */

#ifndef FOUNDRY_METAL_KERNEL_MATVEC_Q8_H
#define FOUNDRY_METAL_KERNEL_MATVEC_Q8_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* INT8 quantized matrix-vector product.
 * x_q: [cols] int8, w: [rows, cols] int8, w_scales: [rows] float32,
 * out: [rows] float32. */
void fm_matvec_q8(fm_buffer *x_q, fm_buffer *w, fm_buffer *w_scales,
                  fm_buffer *out, float x_scale, int rows, int cols);

/* Batched INT8 quantized matrix multiply (GEMM with on-the-fly dequantization).
 * Computes: Out[M, N] = X[M, K] @ dequant(W_q8[N, K], scales[N])^T
 * X: [M, K] float32 input activations.
 * W_q8: [N, K] int8 weights (row-major, rows = output dim, transposed layout).
 * scales: [N] float32 per-row scales.
 * Out: [M, N] float32 output. */
void fm_sgemm_q8(fm_buffer *X, fm_buffer *W_q8, fm_buffer *scales,
                 fm_buffer *Out, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_MATVEC_Q8_H */
