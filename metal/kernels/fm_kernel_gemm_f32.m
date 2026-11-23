/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_gemm_f32.m
 * @brief Dense float32 GEMM/GEMV via MPS.
 *
 *   MPS uses row-major addressing once the MPSMatrixDescriptor carries the
 *   correct row bytes (columns * sizeof(float)).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_gemm_f32.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal/fm_metal_internal.h"

void fm_sgemm(fm_buffer *A, fm_buffer *B, fm_buffer *C,
              int M, int N, int K, float alpha, float beta) {
    if (!fm_metal_is_initialized() || !A || !B || !C) return;

    /* No @autoreleasepool here: in auto-batch mode, MPS objects must survive
     * until the command buffer commits (fm_sync). ARC handles release. */

    /* Create MPS matrix descriptors (row-major) */
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
                         columns:K
                        rowBytes:K * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:K
                         columns:N
                        rowBytes:N * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
                         columns:N
                        rowBytes:N * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:A->mtl descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:B->mtl descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:C->mtl descriptor:descC];

    MPSMatrixMultiplication *mmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:fm_metal_device()
         transposeLeft:NO
        transposeRight:NO
            resultRows:M
         resultColumns:N
       interiorColumns:K
                 alpha:alpha
                  beta:beta];

    id<MTLCommandBuffer> cmdbuf = fm_get_command_buffer();
    [mmul encodeToCommandBuffer:cmdbuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    fm_submit_if_not_batching();
}

void fm_sgemv(fm_buffer *A, fm_buffer *x, fm_buffer *y,
              int M, int K, float alpha, float beta) {
    if (!fm_metal_is_initialized() || !A || !x || !y) return;

    /* Treat as GEMM with N=1:  y[M,1] = A[M,K] @ x[K,1] */
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
                         columns:K
                        rowBytes:K * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor *descX = [MPSMatrixDescriptor
        matrixDescriptorWithRows:K
                         columns:1
                        rowBytes:1 * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor *descY = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
                         columns:1
                        rowBytes:1 * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:A->mtl descriptor:descA];
    MPSMatrix *matX = [[MPSMatrix alloc] initWithBuffer:x->mtl descriptor:descX];
    MPSMatrix *matY = [[MPSMatrix alloc] initWithBuffer:y->mtl descriptor:descY];

    MPSMatrixMultiplication *mmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:fm_metal_device()
         transposeLeft:NO
        transposeRight:NO
            resultRows:M
         resultColumns:1
       interiorColumns:K
                 alpha:alpha
                  beta:beta];

    id<MTLCommandBuffer> cmdbuf = fm_get_command_buffer();
    [mmul encodeToCommandBuffer:cmdbuf leftMatrix:matA rightMatrix:matX resultMatrix:matY];
    fm_submit_if_not_batching();
}

/* Resolves and validates the batch stride of one batched-GEMM operand.
 * A requested stride of 0 means "densely packed".  Returns the stride in
 * bytes, or 0 (never a legal stride) after logging why it was rejected. */
static size_t fm_batched_matrix_stride(const char *label, fm_buffer *buf,
                                       int batch, int rows, int cols,
                                       size_t requested_stride) {
    size_t row_bytes = (size_t)cols * sizeof(float);
    size_t packed    = (size_t)rows * row_bytes;
    size_t stride    = requested_stride ? requested_stride : packed;

    /* MPS requires matrixBytes to be a multiple of rowBytes and to cover
     * a whole matrix; anything else silently reads the wrong elements. */
    if (stride < packed || (stride % row_bytes) != 0) {
        NSLog(@"[fm] sgemm_batched: %s stride %zu invalid — must be a multiple of "
              @"%zu and at least %zu", label, stride, row_bytes, packed);
        return 0;
    }

    size_t needed = (size_t)(batch - 1) * stride + packed;
    if (fm_buffer_size(buf) < needed) {
        NSLog(@"[fm] sgemm_batched: %s buffer is %zu bytes, batch needs %zu",
              label, fm_buffer_size(buf), needed);
        return 0;
    }
    return stride;
}

/* Row-major float32 descriptor for `count` matrices spaced `stride` bytes
 * apart.  `stride` of 0 means densely packed. */
static MPSMatrixDescriptor *fm_matrix_descriptor(int rows, int cols,
                                                 int count, size_t stride) {
    size_t row_bytes = (size_t)cols * sizeof(float);
    return [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows
                                                 columns:(NSUInteger)cols
                                                matrices:(NSUInteger)count
                                                rowBytes:row_bytes
                                             matrixBytes:stride ? stride : (size_t)rows * row_bytes
                                                dataType:MPSDataTypeFloat32];
}

static MPSMatrixMultiplication *fm_new_matmul(int M, int N, int K,
                                              int batch, float alpha, float beta) {
    MPSMatrixMultiplication *mmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:fm_metal_device()
         transposeLeft:NO
        transposeRight:NO
            resultRows:(NSUInteger)M
         resultColumns:(NSUInteger)N
       interiorColumns:(NSUInteger)K
                 alpha:alpha
                  beta:beta];
    /* Stated for the contract's sake, but measured inert on macOS 26:
     * MPS takes the batch it processes from the descriptors' `matrices`
     * count, not from batchSize (pinning batchSize to 1 still runs the
     * whole batch).  Do not rely on it to limit work. */
    mmul.batchStart = 0;
    mmul.batchSize  = (NSUInteger)batch;
    return mmul;
}

/* Encodes one C = alpha*A@B + beta*C at explicit byte offsets into each
 * buffer.  Used for the strided fallback described in fm_sgemm_batched. */
static void fm_encode_gemm_at_offsets(id<MTLCommandBuffer> cmdbuf,
                                      fm_buffer *A, size_t a_offset,
                                      fm_buffer *B, size_t b_offset,
                                      fm_buffer *C, size_t c_offset,
                                      int M, int N, int K,
                                      float alpha, float beta) {
    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:A->mtl offset:a_offset
                                             descriptor:fm_matrix_descriptor(M, K, 1, 0)];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:B->mtl offset:b_offset
                                             descriptor:fm_matrix_descriptor(K, N, 1, 0)];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:C->mtl offset:c_offset
                                             descriptor:fm_matrix_descriptor(M, N, 1, 0)];

    MPSMatrixMultiplication *mmul = fm_new_matmul(M, N, K, 1, alpha, beta);
    [mmul encodeToCommandBuffer:cmdbuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
}

void fm_sgemm_batched(fm_buffer *A, fm_buffer *B, fm_buffer *C,
                      int batch, int M, int N, int K,
                      size_t a_matrix_stride_bytes,
                      size_t b_matrix_stride_bytes,
                      size_t c_matrix_stride_bytes,
                      float alpha, float beta) {
    if (!fm_metal_is_initialized() || !A || !B || !C) return;

    if (batch < 1 || M < 1 || N < 1 || K < 1) {
        NSLog(@"[fm] sgemm_batched: bad shape batch=%d M=%d N=%d K=%d",
              batch, M, N, K);
        return;
    }

    size_t stride_a = fm_batched_matrix_stride("A", A, batch, M, K, a_matrix_stride_bytes);
    size_t stride_b = fm_batched_matrix_stride("B", B, batch, K, N, b_matrix_stride_bytes);
    size_t stride_c = fm_batched_matrix_stride("C", C, batch, M, N, c_matrix_stride_bytes);
    if (!stride_a || !stride_b || !stride_c) return;

    bool packed = stride_a == (size_t)M * K * sizeof(float)
               && stride_b == (size_t)K * N * sizeof(float)
               && stride_c == (size_t)M * N * sizeof(float);

    /* No @autoreleasepool here, for the same reason as fm_sgemm: in
     * auto-batch mode the MPS objects must survive until the command
     * buffer commits in fm_sync(). */
    id<MTLCommandBuffer> cmdbuf = fm_get_command_buffer();

    if (packed) {
        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:A->mtl
                                                 descriptor:fm_matrix_descriptor(M, K, batch, stride_a)];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:B->mtl
                                                 descriptor:fm_matrix_descriptor(K, N, batch, stride_b)];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:C->mtl
                                                 descriptor:fm_matrix_descriptor(M, N, batch, stride_c)];

        MPSMatrixMultiplication *mmul = fm_new_matmul(M, N, K, batch, alpha, beta);
        [mmul encodeToCommandBuffer:cmdbuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    } else {
        /* MPS sizes a batch from the *packed* extent: it processes only as
         * many matrices as fit, at matrixBytes spacing, inside
         * matrices * rows * rowBytes bytes — so with padding it silently
         * drops the tail (measured macOS 26 / M4: a 5-element batch at
         * twice the packed stride computes 3 and leaves 2 untouched).
         * Padded layouts therefore get one encode per element.  Same
         * command buffer, same submission, same results — just without
         * the batched kernel, so keep hot paths packed. */
        for (int e = 0; e < batch; e++) {
            fm_encode_gemm_at_offsets(cmdbuf,
                                      A, (size_t)e * stride_a,
                                      B, (size_t)e * stride_b,
                                      C, (size_t)e * stride_c,
                                      M, N, K, alpha, beta);
        }
    }

    fm_submit_if_not_batching();
}
