/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_gemm_f32.h
 * @brief Dense float32 GEMM / GEMV via MPS (MetalPerformanceShaders), single
 *   and batched. No custom shader — MPS supplies the kernels, this unit owns
 *   the descriptors, the stride validation and the encode.
 *
 *   All matrices are row-major float32.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_METAL_KERNEL_GEMM_F32_H
#define FOUNDRY_METAL_KERNEL_GEMM_F32_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* C = alpha * A @ B + beta * C
 * A: [M, K], B: [K, N], C: [M, N] — all row-major float32. */
void fm_sgemm(fm_buffer *A, fm_buffer *B, fm_buffer *C,
              int M, int N, int K, float alpha, float beta);

/* y = alpha * A @ x + beta * y
 * A: [M, K] row-major, x: [K], y: [M] — all float32. */
void fm_sgemv(fm_buffer *A, fm_buffer *x, fm_buffer *y,
              int M, int K, float alpha, float beta);

/**
 * Batch of independent GEMMs in a single MPS submission:
 *
 *     C[b] = alpha * A[b] @ B[b] + beta * C[b],   for b in [0, batch)
 *
 * All `batch` products share the same M, N, K — this is the uniform-shape
 * batch the chunked delta-net prefill needs (chunk x head), not a ragged one.
 * Encoding one batched kernel instead of `batch` separate ones is the point:
 * it collapses the per-call MPS object construction and dispatch overhead,
 * which dominates when the individual matrices are small.
 *
 * Memory layout (row-major float32, all three buffers):
 *
 *   A[b] occupies rows M x cols K starting at byte  b * a_matrix_stride_bytes
 *   B[b] occupies rows K x cols N starting at byte  b * b_matrix_stride_bytes
 *   C[b] occupies rows M x cols N starting at byte  b * c_matrix_stride_bytes
 *
 * Within a matrix, element (i, j) is at row_bytes * i + sizeof(float) * j,
 * with row_bytes = K, N and N floats respectively — i.e. each matrix is
 * densely packed, only the spacing *between* batch elements is adjustable.
 * A stride of 0 means "packed": M*K, K*N and M*N floats respectively.
 * A non-zero stride must be a multiple of that matrix's row_bytes and at
 * least as large as the packed size; anything else is rejected with a log
 * message and no dispatch.  Strides let a caller batch over slices of a
 * larger tensor without copying, and let A and B advance at different rates.
 *
 * Keep hot paths packed.  A fully packed batch is encoded as one batched
 * MPS kernel; a padded one falls back to one kernel per batch element in
 * the same command buffer (still a single submission, same results, but
 * without the batched kernel).  The fallback exists because MPS sizes a
 * batch from the packed extent and silently drops the tail of a padded
 * one — see the comment in fm_kernel_gemm_f32.m.
 *
 * @param A                     Buffer holding the batch of [M, K] left matrices
 * @param B                     Buffer holding the batch of [K, N] right matrices
 * @param C                     Buffer holding the batch of [M, N] result matrices
 * @param batch                 Number of products, >= 1
 * @param M, N, K               Shape shared by every product, all >= 1
 * @param a_matrix_stride_bytes Byte distance between A[b] and A[b+1], 0 = packed
 * @param b_matrix_stride_bytes Byte distance between B[b] and B[b+1], 0 = packed
 * @param c_matrix_stride_bytes Byte distance between C[b] and C[b+1], 0 = packed
 * @param alpha, beta           Scalars applied as in the formula above
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_sgemm_batched(fm_buffer *A, fm_buffer *B, fm_buffer *C,
                      int batch, int M, int N, int K,
                      size_t a_matrix_stride_bytes,
                      size_t b_matrix_stride_bytes,
                      size_t c_matrix_stride_bytes,
                      float alpha, float beta);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_GEMM_F32_H */
