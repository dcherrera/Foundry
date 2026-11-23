/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rope.cu — partial NEOX rotary position embedding for Foundry
 *
 * Target: sm_37 (K80, Kepler)
 * Compile: nvcc -ptx -arch=sm_37 rope.cu -o rope.ptx
 *
 * Applies partial NEOX RoPE in place to a packed [n_heads * head_dim] tensor:
 * only the first rope_dim (=64) dims of each head_dim (=256) head are rotated;
 * dims [rope_dim, head_dim) pass through untouched.
 *
 * Mirrors, value-for-value, the CPU reference:
 *   Pantheon/src/models/qwen35moe_attn.c
 *   qwen35moe_rope_partial (lines 57-76)
 *
 * NEOX pairing: dim j pairs with dim j+half (half = rope_dim/2), angle =
 *   pos * theta_base^(-2j/rope_dim). This is the single-section collapse the
 *   CPU path documents: for this text-only GGUF every section shares the one
 *   linear token position, so multi-section / interleaved head logic does NOT
 *   apply here. (RoPE pairs dims within a head; it does no GQA head broadcast,
 *   so the interleaved-vs-blocked head-tiling trap is not in play in this file.)
 *
 * PTX symbol emitted (for embedding into ../cuda/fc_cuda_ptx.h):
 *   FC_PTX_ROPE  -> kernel  fc_rope_neox_partial
 *
 * Grid/block layout:
 *   grid  = n_heads        (one block per head)
 *   block = half threads, capped at 256 (rope_dim/2 = 32 for this model, so
 *           one warp's worth — well within Kepler limits). One thread per
 *           rotation pair; idle threads (tid >= half) early-out.
 *
 * Kepler notes:
 *   - Pure fp32. Uses cosf/sinf/powf — same libm intrinsics the CPU path uses,
 *     so the only divergence source is fp32 vs the CPU's fp32 (identical type).
 *   - No reductions, no shared memory, no __shfl — embarrassingly parallel.
 */

extern "C" {

/*
 * Partial NEOX RoPE, in place.
 *
 * data:       [n_heads * head_dim] fp32, rotated in place
 * n_heads:    number of heads in this tensor (Q: 16, K: 2)
 * head_dim:   per-head dimension (256)
 * rope_dim:   number of leading dims to rotate (64); rest pass through
 * pos:        token position (0-based)
 * theta_base: rope.freq_base (e.g. 1e7)
 *
 * Launched with `n_heads` blocks; blockDim.x >= rope_dim/2 (use 32 or 256).
 * No shared memory.
 *
 * Matches CPU qwen35moe_rope_partial: rot = (rope_dim in (0,head_dim)) ?
 * rope_dim : head_dim, half = rot/2; freq = 1/theta_base^(2j/rot).
 */
__global__ void fc_rope_neox_partial(float *data, int n_heads, int head_dim,
                                     int rope_dim, int pos, float theta_base) {
    int head = blockIdx.x;
    if (head >= n_heads) return;

    int rot = (rope_dim > 0 && rope_dim < head_dim) ? rope_dim : head_dim;
    int half = rot / 2;

    int j = threadIdx.x;
    if (j >= half) return;          /* one thread per rotation pair */

    float *h = data + (size_t)head * head_dim;

    /* freq = theta_base^(-2j/rot); angle = pos * freq. Computed per-thread to
     * stay bit-faithful to the CPU per-j powf/cosf/sinf sequence. */
    float freq = 1.0f / powf(theta_base, (float)(2 * j) / (float)rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = h[j];
    float x1 = h[j + half];
    h[j]        = x0 * cos_a - x1 * sin_a;
    h[j + half] = x0 * sin_a + x1 * cos_a;
}

/*
 * Batched (prefill) variant: n consecutive positions in one launch.
 * data: [n][n_heads * head_dim], position of row t is pos0 + t.
 * Grid (n_heads, n); per (head, position) the math is identical to
 * fc_rope_neox_partial at pos = pos0 + t -> bit-exact.
 */
__global__ void fc_rope_neox_partial_nx(float *data, int n_heads, int head_dim,
                                        int rope_dim, int pos0,
                                        float theta_base) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    if (head >= n_heads) return;

    int rot = (rope_dim > 0 && rope_dim < head_dim) ? rope_dim : head_dim;
    int half = rot / 2;

    int j = threadIdx.x;
    if (j >= half) return;

    float *h = data + ((size_t)t * n_heads + head) * head_dim;
    int pos = pos0 + t;

    float freq = 1.0f / powf(theta_base, (float)(2 * j) / (float)rot);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = h[j];
    float x1 = h[j + half];
    h[j]        = x0 * cos_a - x1 * sin_a;
    h[j + half] = x0 * sin_a + x1 * cos_a;
}

/*
 * Multi-user decode variant: n rows, each at its OWN arbitrary position.
 * data: [n][n_heads * head_dim], pos: [n] int32 device array — row t is
 * rotated at position pos[t]. Grid (n_heads, n); per (head, row) the math
 * is identical to fc_rope_neox_partial at pos = pos[t] -> bit-exact per
 * user vs decoding alone.
 */
__global__ void fc_rope_neox_partial_pos(float *data, int n_heads,
                                         int head_dim, int rope_dim,
                                         const int *pos, float theta_base) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    if (head >= n_heads) return;

    int rot = (rope_dim > 0 && rope_dim < head_dim) ? rope_dim : head_dim;
    int half = rot / 2;

    int j = threadIdx.x;
    if (j >= half) return;

    float *h = data + ((size_t)t * n_heads + head) * head_dim;
    int p = pos[t];

    float freq = 1.0f / powf(theta_base, (float)(2 * j) / (float)rot);
    float angle = (float)p * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = h[j];
    float x1 = h[j + half];
    h[j]        = x0 * cos_a - x1 * sin_a;
    h[j + half] = x0 * sin_a + x1 * cos_a;
}

} /* extern "C" */
