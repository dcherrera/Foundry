/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * normalization.cu — CUDA normalization kernels for Foundry
 *
 * Target: sm_37 (K80, Kepler)
 * Compile: nvcc -ptx -arch=sm_37 normalization.cu -o normalization.ptx
 *
 * RMS norm:     single-block reduction with shared memory
 * Softmax:      one block per row, two-pass (max + exp-sum) with shared memory
 * QK-RMSNorm:   one block per head, per-head RMS over head_dim, shared weight
 *
 * PTX symbols emitted (for embedding into ../cuda/fc_cuda_ptx.h):
 *   FC_PTX_NORMALIZATION -> kernels  fc_rms_norm, fc_softmax, fc_qk_rms_norm
 */

extern "C" {

/*
 * RMS normalization: out[i] = (x[i] / rms) * weight[i]
 * where rms = sqrt(mean(x^2) + eps)
 *
 * Launched with 1 block, blockDim.x threads.
 * Shared memory: blockDim.x * sizeof(float)
 *
 * weight may be NULL (passed as 0 device pointer) — if so, skip scaling.
 */
__global__ void fc_rms_norm(const float *x, const float *weight, float *out,
                            int n, float eps) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int threads = blockDim.x;

    /* Phase 1: compute partial sum of squares */
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += threads) {
        float v = x[i];
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    /* Shared memory reduction */
    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Phase 2: compute RMS and normalize */
    float rms = sqrtf(sdata[0] / (float)n + eps);
    float inv_rms = 1.0f / rms;

    for (int i = tid; i < n; i += threads) {
        float v = x[i] * inv_rms;
        if (weight) {
            v *= weight[i];
        }
        out[i] = v;
    }
}

/*
 * Softmax over last dimension.
 * x: [rows, cols], out: [rows, cols]
 *
 * Launched with rows blocks, blockDim.x threads per block.
 * Shared memory: blockDim.x * sizeof(float) * 2
 *   First half for max reduction, second half for sum reduction.
 */
__global__ void fc_softmax(const float *x, float *out, int rows, int cols) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int threads = blockDim.x;

    const float *x_row = x + row * cols;
    float *o_row = out + row * cols;

    float *smax = sdata;              /* [threads] for max reduction */
    float *ssum = sdata + threads;    /* [threads] for sum reduction */

    /* Pass 1: find max */
    float local_max = -1e30f;
    for (int i = tid; i < cols; i += threads) {
        float v = x_row[i];
        if (v > local_max) local_max = v;
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax[tid + s] > smax[tid])
                smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }
    float row_max = smax[0];

    /* Pass 2: compute exp(x - max) and sum */
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += threads) {
        float v = expf(x_row[i] - row_max);
        o_row[i] = v;  /* store intermediate exp values */
        local_sum += v;
    }
    ssum[tid] = local_sum;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / ssum[0];

    /* Pass 3: normalize */
    for (int i = tid; i < cols; i += threads) {
        o_row[i] *= inv_sum;
    }
}

/*
 * Per-head QK-RMSNorm, in place.
 * data: [n_heads * head_dim] fp32; each head is RMS-normalized over its own
 * head_dim and scaled by the shared `weight` [head_dim]. The norm weight is
 * shared across all heads — the GGUF stores it pre-baked (incl. any (1+w)
 * offset), so this is just one RMS-norm per head.
 *
 * Mirrors the CPU reference Pantheon/src/models/qwen35moe_attn.c
 * qwen35moe_qk_norm_heads (lines 39-46), which calls pn_vrms_norm_f32 once per
 * head:  out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i].
 *
 * Launched with `n_heads` blocks, blockDim.x threads (use 256).
 * Shared memory: blockDim.x * sizeof(float) (sum-of-squares reduction).
 *
 * weight may be NULL (passed as 0 device pointer) — if so, skip scaling.
 */
__global__ void fc_qk_rms_norm(float *data, const float *weight,
                               int n_heads, int head_dim, float eps) {
    extern __shared__ float sdata[];

    int head = blockIdx.x;
    if (head >= n_heads) return;

    int tid = threadIdx.x;
    int threads = blockDim.x;
    float *h = data + (size_t)head * head_dim;

    /* Phase 1: partial sum of squares over this head's dims */
    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += threads) {
        float v = h[i];
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Phase 2: normalize and scale by the shared weight */
    float rms = sqrtf(sdata[0] / (float)head_dim + eps);
    float inv_rms = 1.0f / rms;

    for (int i = tid; i < head_dim; i += threads) {
        float v = h[i] * inv_rms;
        if (weight) {
            v *= weight[i];
        }
        h[i] = v;
    }
}

/*
 * Batched (prefill) RMS norm: one block per ROW of x [rows][n], each row
 * normalized independently with the exact fc_rms_norm op order (same
 * blockDim -> same reduction tree -> bit-exact per row).
 *
 * Launched with `rows` blocks, blockDim.x threads (the single-row
 * launcher's choice: min(256, n)). Shared: blockDim.x * sizeof(float).
 */
__global__ void fc_rms_norm_nx(const float *x, const float *weight, float *out,
                               int n, float eps, int rows) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int threads = blockDim.x;
    const float *xr = x + (size_t)row * n;
    float *outr = out + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += threads) {
        float v = xr[i];
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)n + eps);
    float inv_rms = 1.0f / rms;

    for (int i = tid; i < n; i += threads) {
        float v = xr[i] * inv_rms;
        if (weight) {
            v *= weight[i];
        }
        outr[i] = v;
    }
}

} /* extern "C" */
