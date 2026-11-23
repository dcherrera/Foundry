/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * matvec_q8.cu — INT8 quantized matrix-vector product for Foundry
 *
 * Target: sm_37 (K80, Kepler)
 * Compile: nvcc -ptx -arch=sm_37 matvec_q8.cu -o matvec_q8.ptx
 *
 * Computes: out[row] = x_scale * w_scales[row] * dot(x_q[cols], w[row, cols])
 * where x_q and w are int8, out is float32.
 *
 * One block per output row. Threads cooperatively compute the dot product
 * then reduce via shared memory.
 */

extern "C" {

/*
 * INT8 quantized matvec.
 *
 * x_q:      [cols] int8 — quantized input vector
 * w:        [rows, cols] int8 — quantized weight matrix (row-major)
 * w_scales: [rows] float32 — per-row weight scales
 * out:      [rows] float32 — output vector
 * x_scale:  float32 — input vector scale
 * rows:     number of output elements
 * cols:     inner dimension (input size)
 *
 * Launched with rows blocks, blockDim.x threads.
 * Shared memory: blockDim.x * sizeof(float)
 */
__global__ void fc_matvec_q8(const signed char *x_q, const signed char *w,
                              const float *w_scales, float *out,
                              float x_scale, int rows, int cols) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int threads = blockDim.x;

    const signed char *w_row = w + row * cols;

    /* Each thread computes a partial dot product over its stride */
    int partial_sum = 0;
    for (int i = tid; i < cols; i += threads) {
        partial_sum += (int)x_q[i] * (int)w_row[i];
    }

    sdata[tid] = (float)partial_sum;
    __syncthreads();

    /* Shared memory reduction */
    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Thread 0 writes the final result */
    if (tid == 0) {
        out[row] = x_scale * w_scales[row] * sdata[0];
    }
}

} /* extern "C" */
