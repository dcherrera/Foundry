/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * elementwise.cu — CUDA elementwise kernels for Foundry
 *
 * Target: sm_37 (K80, Kepler)
 * Compile: nvcc -ptx -arch=sm_37 elementwise.cu -o elementwise.ptx
 *
 * Constraints:
 *   - No FP16 math (sm_37 has no native half support)
 *   - Use __shfl_down() not __shfl_down_sync() (sync is sm_70+)
 *   - Max 1024 threads per block, 48 KB shared memory per SM
 */

extern "C" {

/* out[i] = a[i] + b[i] */
__global__ void fc_add(const float *a, const float *b, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

/* out[i] = a[i] * b[i] */
__global__ void fc_mul(const float *a, const float *b, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

/* out[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i])) */
__global__ void fc_silu(const float *x, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = v / (1.0f + expf(-v));
    }
}

/* Strided 2-D copy in 16-byte units: row r's width16 units from src + r * src_pitch16
 * to dst + r * dst_pitch16 (grid.y = rows). Runs on the die that owns dst; src may be
 * a peer die's memory (unified addressing, peer access enabled), so a cross-die copy
 * is a PULL by the reader's SMs. A CUDA graph captures it as a kernel node that stays
 * on this die, unlike a memcpy node, which the graph issues from its launching die. */
__global__ void fc_copy_2d(const uint4 *src, uint4 *dst, int width16, int src_pitch16,
                           int dst_pitch16) {
    const uint4 *s = src + (size_t)blockIdx.y * src_pitch16;
    uint4 *d = dst + (size_t)blockIdx.y * dst_pitch16;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width16;
         i += gridDim.x * blockDim.x)
        d[i] = s[i];
}

/* One die's half of a two-die exchange (tensor-parallel gather), one launch per die and
 * exchange, ordered by device counters instead of cross-die events. sync = this die's
 * {sig, cnt, ticket} words for the exchange slot (its own memory); peer_sig = the
 * partner's sig (peer memory). Every block takes target = cnt + 1 and publishes
 * sig = target at entry (stream order: the source is complete; every block writes the
 * same value, so no block's scheduling can hold the partner back). blockIdx.z 0
 * copies the own rows (own -> dst + own_off16); z 1 waits until the partner's sig
 * reaches target, then pulls its rows (peer -> dst + peer_off16). The last block to
 * finish sets cnt = target. Both dies launch once per exchange, so the counters advance
 * in lockstep and a captured graph replays without baked values. The partner publishes
 * its next sig for this slot only after this launch's pull of its source (its stream
 * order), so a source may be rewritten once the next exchange's wait has passed. A wait
 * longer than ~2^31 clocks traps (the launch fails; nothing is silently wrong). */
__global__ void fc_xchg_2d(const uint4 *own, const uint4 *peer, uint4 *dst, int width16,
                           int src_pitch16, int dst_pitch16, int own_off16, int peer_off16,
                           unsigned int *sync, const unsigned int *peer_sig) {
    volatile unsigned int *vs = sync;
    const unsigned int target = vs[1] + 1u;
    const int pull = blockIdx.z;
    if (threadIdx.x == 0) {
        vs[0] = target;
        __threadfence_system();
    }
    if (pull) {
        if (threadIdx.x == 0) {
            const volatile unsigned int *ps = peer_sig;
            long long t0 = clock64();
            while ((int)(*ps - target) < 0)
                if (clock64() - t0 > (1LL << 31))
                    asm volatile("trap;");
            __threadfence();
        }
        __syncthreads();
    }
    const uint4 *s = (pull ? peer : own) + (size_t)blockIdx.y * src_pitch16;
    uint4 *d = dst + (pull ? peer_off16 : own_off16) + (size_t)blockIdx.y * dst_pitch16;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width16;
         i += gridDim.x * blockDim.x)
        d[i] = s[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned int nb = gridDim.x * gridDim.y * gridDim.z;
        if (atomicAdd(&sync[2], 1u) == nb - 1u) {
            vs[2] = 0u;
            vs[1] = target;
        }
    }
}

} /* extern "C" */
