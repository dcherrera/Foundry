/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   attention.cu
 * @brief  On-die causal GQA flash attention for Foundry (fp32 and Q4_0 KV cache).
 *
 * Written from a behaviour specification; no ggml or llama.cpp source was consulted.
 * Scope of that rewrite: the Q4_0 KV-cache format defines, fc_kvq4_row_bytes,
 * fc_kvq4_quant_block, fc_kvq4_dequant_elem, fc_kvq4_dequant_lane_chunk and
 * fc_attn_flash_prefill_q4_kvs; the rest of the file is unchanged original work.
 *
 * Target: sm_37 (K80, Kepler)
 * Compile: nvcc -ptx -arch=sm_37 attention.cu -o attention.ptx
 *
 * Replaces the HOST-side causal GQA softmax in the qwen35moe batched
 * prefill (the O(N^2) 2-core bottleneck: 87.5% of an 8k prefill) with a
 * flash-attention-style kernel: tiled QK^T -> ONLINE softmax (running
 * max + running sum, the N x N score matrix is never materialized) ->
 * PV accumulate, entirely on the die against a GPU-resident KV cache.
 *
 * Numerics: same math as the host reference (qwen35moe_forward_cuda.c
 * host softmax) but reduced in tile order with running-max rescaling, so
 * results drift ~1e-6 relative vs the host's global-max single pass —
 * NOT bit-exact by design (gate: argmax-exact + logits within ~1e-3).
 *
 * Sliding-window attention (QWEN35_ATTN_WINDOW, StreamingLLM-style): each
 * flash kernel takes attn_sink/attn_win. When attn_win > 0 the query at
 * absolute position cur_pos attends ONLY the key set
 *   [0, attn_sink)  U  [cur_pos - attn_win + 1, cur_pos]
 * (attention sinks + recent window); tiles that intersect neither are
 * skipped BEFORE any staging/sync, turning prefill O(N^2) -> O(N*W). The
 * online-softmax (m, l, acc) simply never sees the excluded keys, so the
 * result is a correct softmax over the windowed set. When attn_win <= 0
 * every tile is processed — byte-identical to full attention (the OFF /
 * A-baseline path). A query always sees its own position (cur_pos is in
 * the recent window for attn_win >= 1); an overlapping sink/window tile is
 * processed once (the range predicate is an OR).
 *
 * PTX symbols emitted (for embedding into ../cuda/fc_cuda_ptx.h):
 *   FC_PTX_ATTENTION -> kernels fc_attn_qz_split, fc_attn_kv_append,
 *                               fc_attn_flash_prefill (+ _q4 variants),
 *                               fc_attn_flash_decode(_q4) + _combine,
 *                               the *_dev device-position variants for
 *                               CUDA-graph capture (C1), and the *_slots
 *                               multi-user decode variants (L7)
 *
 * Kepler notes:
 *   - Pure fp32 FMA (no tensor cores); expf matches the host libm family.
 *   - 48 KB shared per block: the flash kernel's working set (q group +
 *     one padded K tile + softmax state) is 26 720 B for kv_groups=8/
 *     head_dim=256, so 4 blocks per SMX (114 688 B/SMX; shared -- not
 *     registers -- is what caps residency here). V is streamed through
 *     __ldg (read-only cache), never staged in shared.
 *   - __shfl_down_sync with width 16 for the dot-product lane reduction
 *     (sm_30+; the mask is ignored pre-Volta).
 *
 * @status None
 * @issues None
 * @todo   Split into prefill and decode files (see the TODO in the
 *         NARROW-STAGING comment for the SASS-identity gate it needs).
 */

/* Flash-kernel tiling: keys staged per tile, lanes cooperating on one
 * q.k dot, and the register-accumulator cap on GQA group size. */
#define FC_ATTN_TILE_KEYS  16
#define FC_ATTN_DOT_LANES  16
#define FC_ATTN_MAX_GROUPS 8

/* K-tile row stride, in floats. The tile is read as k_s[ky*stride + lane +
 * i*16] with ky = tid/16 and lane = tid%16, so the two half-warps of a warp
 * carry IDENTICAL lanes and ADJACENT ky. At the shipped head_dim = 256 an
 * unpadded row stride is exactly 8 x 32 banks, which puts both half-warps on
 * the same banks at different addresses: a 2-way conflict on every k-operand
 * LDS. Padding the row by half a bank cycle breaks it for any head_dim that
 * is a multiple of 16. Costs TILE_KEYS*16 floats (1 KB) of dynamic shared and
 * does not change occupancy at the shipped shape (26 720 B x 4 blocks of
 * 114 688 B/SMX). MUST match fc_attn_flash_shared_bytes() in cuda/fc_cuda.c. */
#define FC_ATTN_K_PAD         16
#define FC_ATTN_K_STRIDE(hd)  ((hd) + FC_ATTN_K_PAD)

/* ============================================================
 * 4-bit KV cache (Q4_0 blocks)
 *
 * A K or V row of head_dim floats is stored as head_dim/32 blocks of 18 bytes:
 *   bytes 0-1   scale d, binary16 bit pattern, little-endian
 *   bytes 2-17  16 code bytes; byte 2+j holds element j in its low nibble and
 *               element j+16 in its high nibble
 * A code c in 0..15 decodes to (c - 8) * d. The host (Pantheon pn_kvq4_*)
 * reads and writes the same bytes, so the layout is an interface.
 * ============================================================ */
#include <cuda_fp16.h>

#define FC_KVQ4_BLK       32
#define FC_KVQ4_BLK_BYTES 18

/* fp16 sink + recent-window exemption: the attention sink (first `sink`
 * positions) and the last `win` tokens are kept at fp16 in small side buffers
 * instead of q4, so greedy argmax stops flipping on near-ties. The q4 body
 * still holds every position; recent rows are read from the fp16 window ring
 * while recent and from q4 once they age out (no rolling). Buffers are sized to
 * these maxima (must match host PN_KVQ4_WMAX / PN_KVQ4_SINKMAX).
 *
 * THE RING LAPS AT FC_KVQ4_WMAX, NOT AT `win`. The reader's window is only
 * `win` wide, but a BATCHED prefill call appends ALL n rows before ANY query
 * row reads, so a ring lapped at `win` lets row t's own window keys be
 * overwritten by rows t+1..n-1 of the same call — row t then attends to K/V
 * from positions AFTER it, a causality violation (see fc_attn_flash_prefill_q4
 * for the guard and Pantheon results/q4_win_ring_hazard.md for the history).
 * Lapping at WMAX instead keeps live every key a batch of up to
 * (WMAX - win) rows can still need, and costs nothing: the buffers were
 * always WMAX rows per head — only `win` of them were ever being used. */
#define FC_KVQ4_WMAX    256
#define FC_KVQ4_SINKMAX 8

/* Predicate: key position s read from the fp16 shadow (sink or recent window)
 * for a query at cur_pos. Matches host pn_kvq4_is_fp16 exactly. */
__device__ __forceinline__ int fc_kvq4_is_fp16(int s, int cur_pos, int sink,
                                               int win) {
    return (s < sink) || (win > 0 && s > cur_pos - win);
}

/* Bytes in one Q4_0 row of head_dim elements (host mirror: FC_ATTN_KVQ4_ROW_BYTES). */
__device__ __forceinline__ int fc_kvq4_row_bytes(int head_dim) {
    return (head_dim / FC_KVQ4_BLK) * FC_KVQ4_BLK_BYTES;
}

/* Max head_dim the append kernels stage in shared memory (CUDA's block-size
 * ceiling; every dispatcher launches block = head_dim threads). */
#define FC_KVQ4_MAX_HEAD_DIM 1024

/* Quantize one 32-element block x32 (shared memory, already synced) into the
 * 18 bytes at blk_ptr. Called by all 32 lanes of a warp with the same block;
 * lane 0 writes the scale, lanes 0..15 one code byte each, so every byte of the
 * block is written exactly once.
 *
 * Each lane folds the block maximum itself instead of reducing across the warp:
 * on sm_37 a shuffle from an inactive source lane returns 0, which corrupted the
 * high nibbles in a cooperative version. The serial fold also matches the host
 * quantizer's order bit for bit. The scale is stored negative (d = amax / -8).
 * The order convert -> +8 -> clamp is part of the output for out-of-range
 * products (NaN scale, reciprocal overflow); keep it. */
__device__ __forceinline__ void fc_kvq4_quant_block(const float *x32,
                                                    unsigned char *blk_ptr,
                                                    int lane) {
    float amax = 0.0f;
    for (int j = 0; j < FC_KVQ4_BLK; j++)
        amax = fmaxf(amax, fabsf(x32[j]));

    const float d = amax / -8.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    if (lane == 0) {
        /* One 16-bit store. Every block starts at an even offset (18-byte
         * stride), so the scale is 2-byte aligned. Do not split it into two
         * byte stores: ptxas 11.4 (sm_37) compiles st.u8 of the f16 register
         * as a value conversion (F2I.U32.F16.TRUNC) and writes 0 as the low
         * byte of every (negative) scale. */
        *(__half *)blk_ptr = __float2half(d);
    }
    if (lane < FC_KVQ4_BLK / 2) {
        int q_lo = (int)rintf(x32[lane] * id) + 8;
        int q_hi = (int)rintf(x32[lane + FC_KVQ4_BLK / 2] * id) + 8;
        if (q_lo < 0)
            q_lo = 0;
        else if (q_lo > 15)
            q_lo = 15;
        if (q_hi < 0)
            q_hi = 0;
        else if (q_hi > 15)
            q_hi = 15;
        blk_ptr[2 + lane] = (unsigned char)(q_lo | (q_hi << 4));
    }
}

/* Decode element e of a Q4_0 row. The scale is read as two byte loads. */
__device__ __forceinline__ float fc_kvq4_dequant_elem(const unsigned char *row, int e) {
    const unsigned char *blk = row + (e / FC_KVQ4_BLK) * FC_KVQ4_BLK_BYTES;
    const unsigned short d_bits = (unsigned short)(blk[0] | (blk[1] << 8));
    const float d = __half2float(__ushort_as_half(d_bits));
    const int local = e & (FC_KVQ4_BLK - 1);
    const int half_blk = FC_KVQ4_BLK / 2;
    const int code_off = local < half_blk ? local : local - half_blk;
    const int shift = local < half_blk ? 0 : 4;
    const int code = (blk[2 + code_off] >> shift) & 0xF;
    return (float)(code - 8) * d;
}

/* Sliding-window tile predicate: is the key tile [j0, j0+tk) attended by the
 * query at absolute position cur_pos? True when attn_win <= 0 (full attention),
 * OR the tile intersects the sink region [0, attn_sink), OR (attn_win > 0) it
 * intersects the recent window [cur_pos - attn_win + 1, cur_pos]. The test is
 * uniform across the block (depends only on j0/tk/cur_pos), so branching on it
 * before a __syncthreads() is safe. */
__device__ __forceinline__ int fc_attn_tile_in_window(int j0, int tk,
                                                      int cur_pos,
                                                      int attn_sink,
                                                      int attn_win) {
    if (attn_win <= 0)
        return 1;
    if (j0 < attn_sink)
        return 1;
    int win_lo = cur_pos - attn_win + 1;
    if (win_lo < 0)
        win_lo = 0;
    /* intersects [win_lo, cur_pos]: tile end > win_lo AND tile start <= cur_pos */
    return (j0 + tk > win_lo) && (j0 <= cur_pos);
}

extern "C" {

/*
 * Split the joint Q+gate projection into contiguous q and z tensors.
 *
 * qg:       [n][n_head][2*head_dim] fp32 — per head: [q head_dim | z head_dim]
 * q, z:     [n][n_head][head_dim] fp32 (written)
 * head_dim: per-head dimension
 * total:    n * n_head * head_dim (elements of each output)
 *
 * Replaces the host-side split loop (a_qg D2H + memcpy + a_q H2D).
 * 1D launch, one thread per output element.
 */
__global__ void fc_attn_qz_split(const float *qg, float *q, float *z,
                                 int head_dim, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int dd = i % head_dim;
    int th = i / head_dim;          /* flattened (position, head) */
    size_t src = (size_t)th * 2 * head_dim + dd;
    q[i] = qg[src];
    z[i] = qg[src + head_dim];
}

/*
 * Append n roped K/V rows to the GPU-resident ring KV cache.
 *
 * k_new, v_new:     [n][n_head_kv][head_dim] fp32 (post QK-norm + RoPE)
 * k_cache, v_cache: [n_head_kv][max_seq][head_dim] fp32 (written) — same
 *                   layout as the host cache so host<->GPU syncs are
 *                   contiguous per-head row ranges
 * pos0:             global position of row 0 (row t lands at
 *                   (pos0 + t) % max_seq, the host ring rule)
 *
 * Grid (n, n_head_kv), block head_dim threads.
 */
__global__ void fc_attn_kv_append(const float *k_new, const float *v_new,
                                  float *k_cache, float *v_cache,
                                  int n_head_kv, int head_dim, int max_seq,
                                  int pos0) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int dd = threadIdx.x;
    if (dd >= head_dim) return;

    int cache_pos = (pos0 + t) % max_seq;
    size_t src = ((size_t)t * n_head_kv + h) * head_dim + dd;
    size_t dst = ((size_t)h * max_seq + cache_pos) * head_dim + dd;
    k_cache[dst] = k_new[src];
    v_cache[dst] = v_new[src];
}

/*
 * Causal GQA flash attention over the GPU KV cache (prefill mode).
 *
 * q:        [n][n_head][head_dim] fp32 (post QK-norm + RoPE);
 *           n_head = gridDim.y * kv_groups, group g of kv head h is
 *           global head h*kv_groups + g (the host GQA map qh/kv_groups)
 * k_cache:  [n_head_kv][max_seq][head_dim] fp32 (rows [0, pos0+n) live)
 * v_cache:  same layout
 * out:      [n][n_head][head_dim] fp32 (written; softmax(qK^T*scale)V)
 * kv_groups: q heads per kv head (<= FC_ATTN_MAX_GROUPS)
 * pos0:     global position of query row 0 — query t attends cache rows
 *           [0, pos0+t] (causal); caller guarantees pos0+n <= max_seq
 *           (no ring wrap inside the window)
 * scale:    1/sqrt(head_dim)
 *
 * Grid (n, n_head_kv), block head_dim threads (multiple of 16), dynamic
 * shared = (kv_groups*head_dim + TILE*FC_ATTN_K_STRIDE(head_dim) +
 * 2*kv_groups*TILE + 3*kv_groups) floats.
 *
 * Per K tile: the block's kv_groups q heads score the tile's keys with
 * 16-lane cooperative dots (each KV row is read ONCE per tile and reused
 * across the whole q group — the GQA lever), one thread per head folds
 * the tile into the running (max, sum) state, then every thread
 * accumulates its output dim across the tile's V rows with the standard
 * online-softmax rescale. Fully-masked tiles are never visited (the key
 * loop stops at the query's causal bound).
 */
__global__ void fc_attn_flash_prefill(const float *q, const float *k_cache,
                                      const float *v_cache, float *out,
                                      int kv_groups, int head_dim,
                                      int max_seq, int pos0, int attn_sink,
                                      int attn_win, float scale) {
    const int t = blockIdx.x;             /* query row in the batch */
    const int kvh = blockIdx.y;           /* kv head */
    const int tid = threadIdx.x;          /* == output dim owned */
    const int G = kv_groups;
    const int seq_len = pos0 + t + 1;     /* causal window */
    const int cur_pos = seq_len - 1;      /* this query's absolute position */

    const int kstride = FC_ATTN_K_STRIDE(head_dim);
    extern __shared__ float sh[];
    float *q_s = sh;                                   /* [G][head_dim] */
    float *k_s = q_s + (size_t)G * head_dim;           /* [TILE][kstride] */
    float *score_s = k_s + (size_t)FC_ATTN_TILE_KEYS * kstride;  /* [G][TILE] */
    float *p_s = score_s + G * FC_ATTN_TILE_KEYS;      /* [TILE][G] */
    float *m_s = p_s + G * FC_ATTN_TILE_KEYS;          /* [G] running max */
    float *l_s = m_s + G;                              /* [G] running sum */
    float *corr_s = l_s + G;                           /* [G] tile rescale */

    /* Stage this block's q group; init the online-softmax state. */
    const size_t n_head = (size_t)gridDim.y * G;
    const float *q_base = q + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;
    for (int g = 0; g < G; g++)
        q_s[(size_t)g * head_dim + tid] = q_base[(size_t)g * head_dim + tid];
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000); /* -inf */
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    /* Compile-time width, so ptxas keeps acc[] in REGISTERS. Every loop that
     * indexes acc[] is unrolled to FC_ATTN_MAX_GROUPS with an `if (g < G)`
     * guard; a loop bounded by the RUNTIME kv_groups cannot be unrolled, the
     * array lands in LOCAL memory, and the PV inner loop then pays an
     * LDL.128/STL.128 round trip per key. ptxas reports that as a 32-byte
     * "stack frame" and prints "0 bytes spill stores" right beside it --
     * a never-promoted local array is not counted as a spill. */
    float acc[FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        acc[g] = 0.0f;

    const float *k_head = k_cache + (size_t)kvh * max_seq * head_dim;
    const float *v_head = v_cache + (size_t)kvh * max_seq * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;   /* key-in-tile this thread scores */
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = head_dim / FC_ATTN_DOT_LANES; /* dims per lane */

    for (int j0 = 0; j0 < seq_len; j0 += FC_ATTN_TILE_KEYS) {
        int tk = seq_len - j0;
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;

        /* Sliding window: skip tiles outside sink U recent-window before any
         * staging/sync (uniform across the block — no __syncthreads() split). */
        if (!fc_attn_tile_in_window(j0, tk, cur_pos, attn_sink, attn_win))
            continue;

        /* Stage the K tile (coalesced; read once, reused by all G heads).
         * Key-outer so the padded row stride costs no address arithmetic. */
        for (int key = 0; key < tk; key++)
            for (int e = tid; e < head_dim; e += blockDim.x)
                k_s[(size_t)key * kstride + e] =
                    k_head[(size_t)(j0 + key) * head_dim + e];
        __syncthreads();

        /* QK^T: 16 lanes cooperate on each (head, key) dot.
         *
         * i-outer / g-inner: each thread reads its chunk of the staged K
         * row ONCE per tile into a register and fans it across the whole GQA
         * group, instead of re-reading k_s once per group (G=8 -> 1 LDS per
         * element). BIT-EXACT: every part[g] still accumulates over e in the
         * same ascending order the g-outer form used, so this is pure operand
         * motion. part[] is indexed only by the unrolled compile-time g, for
         * the same reason acc[] is -- a runtime-bounded loop would push it to
         * LOCAL memory and cost an LDL/STL round trip per element. */
        {
            float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                part[g] = 0.0f;
            if (ky < tk) {
                const float *kv = k_s + (size_t)ky * kstride;
                for (int i = 0; i < chunk; i++) {
                    int e = lane + i * FC_ATTN_DOT_LANES;
                    float kve = kv[e];
#pragma unroll
                    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                        if (g < G)
                            part[g] += q_s[(size_t)g * head_dim + e] * kve;
                }
            }
            /* G is uniform across the block, so the guarded unroll keeps the
             * shuffle warp-uniform exactly as the runtime-bounded loop did. */
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    float p = part[g];
                    for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,
                                              FC_ATTN_DOT_LANES);
                    if (lane == 0 && ky < tk)
                        score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;
                }
            }
        }
        __syncthreads();

        /* Online-softmax fold, one thread per head: new running max, the
         * rescale factor for the accumulators, this tile's probabilities. */
        if (tid < G) {
            const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
            float m_new = m_s[tid];
            for (int j = 0; j < tk; j++)
                m_new = fmaxf(m_new, sc[j]);
            float corr = expf(m_s[tid] - m_new);
            float sum = 0.0f;
            for (int j = 0; j < tk; j++) {
                float pj = expf(sc[j] - m_new);
                p_s[(size_t)j * G + tid] = pj;   /* [TILE][G]: PV reads a run */
                sum += pj;
            }
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = m_new;
            corr_s[tid] = corr;
        }
        __syncthreads();

        /* PV accumulate: thread tid owns out dim tid; V rows stream through
         * the read-only cache (coalesced across the block). */
        for (int j = 0; j < tk; j++) {
            float vv = __ldg(&v_head[(size_t)(j0 + j) * head_dim + tid]);
            const float *pj_s = p_s + (size_t)j * G;
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    if (j == 0)
                        acc[g] *= corr_s[g];
                    acc[g] += pj_s[g] * vv;
                }
            }
        }
        __syncthreads(); /* next tile overwrites k_s/score_s/p_s */
    }

    float *out_base = out + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        if (g < G)
            out_base[(size_t)g * head_dim + tid] = acc[g] / l_s[g];
}

/*
 * Append n roped K/V rows to the GPU-resident Q4 ring KV cache (quantizing
 * each row to Q4_0 as it lands). Same ring rule as fc_attn_kv_append.
 *
 * k_new, v_new:     [n][n_head_kv][head_dim] fp32 (post QK-norm + RoPE)
 * k_cache, v_cache: [n_head_kv][max_seq][row_bytes] Q4_0 bytes (written)
 * pos0:             global position of row 0 (caller guarantees pos0+n<=max_seq
 *                   so cache_pos == pos0+t, no ring wrap in the window)
 *
 * Grid (n, n_head_kv), block head_dim threads (a multiple of 32; each 32-lane
 * warp owns one Q4_0 block of the row).
 */
__global__ void fc_attn_kv_append_q4(const float *k_new, const float *v_new,
                                     unsigned char *k_cache,
                                     unsigned char *v_cache,
                                     __half *k_sink, __half *v_sink,
                                     __half *k_win, __half *v_win,
                                     int n_head_kv, int head_dim, int max_seq,
                                     int pos0, int sink, int win) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    if (tid >= head_dim) return;

    int cache_pos = pos0 + t;
    int row_bytes = fc_kvq4_row_bytes(head_dim);
    size_t src = ((size_t)t * n_head_kv + h) * head_dim + tid;
    size_t dst_row = ((size_t)h * max_seq + cache_pos) * row_bytes;
    int blk = tid / FC_KVQ4_BLK;
    int lane = tid & (FC_KVQ4_BLK - 1);

    /* Stage the row in shared memory so the (shuffle-free, Kepler-safe)
     * block quantizer can read all 32 elements of its block. */
    __shared__ float s_k[FC_KVQ4_MAX_HEAD_DIM];
    __shared__ float s_v[FC_KVQ4_MAX_HEAD_DIM];
    float kf = k_new[src];
    float vf = v_new[src];
    s_k[tid] = kf;
    s_v[tid] = vf;
    __syncthreads();
    fc_kvq4_quant_block(s_k + (size_t)blk * FC_KVQ4_BLK,
                        k_cache + dst_row + (size_t)blk * FC_KVQ4_BLK_BYTES,
                        lane);
    fc_kvq4_quant_block(s_v + (size_t)blk * FC_KVQ4_BLK,
                        v_cache + dst_row + (size_t)blk * FC_KVQ4_BLK_BYTES,
                        lane);

    /* Mirror the row into the fp16 sink / recent-window shadow (element tid). */
    if (win > 0) {
        size_t wi = ((size_t)h * FC_KVQ4_WMAX + (cache_pos % FC_KVQ4_WMAX)) * head_dim + tid;
        k_win[wi] = __float2half(kf);
        v_win[wi] = __float2half(vf);
    }
    if (cache_pos < sink) {
        size_t si = ((size_t)h * FC_KVQ4_SINKMAX + cache_pos) * head_dim + tid;
        k_sink[si] = __float2half(kf);
        v_sink[si] = __float2half(vf);
    }
}

/*
 * Causal GQA flash attention over the Q4 GPU KV cache (prefill mode). Identical
 * online-softmax math and tiling to fc_attn_flash_prefill; the only difference
 * is K/V are read from the packed Q4_0 cache and dequantized in-kernel (the K
 * tile into shared, V per element on read) so no fp32 KV is ever materialized.
 *
 * k_cache/v_cache: [n_head_kv][max_seq][row_bytes] Q4_0 bytes.
 * Grid/shared/other args as fc_attn_flash_prefill.
 */
__global__ void fc_attn_flash_prefill_q4(const float *q,
                                         const unsigned char *k_cache,
                                         const unsigned char *v_cache,
                                         const __half *k_sink,
                                         const __half *v_sink,
                                         const __half *k_win,
                                         const __half *v_win,
                                         float *out, int kv_groups,
                                         int head_dim, int max_seq, int pos0,
                                         int sink, int win, int attn_sink,
                                         int attn_win, float scale) {
    const int t = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    const int G = kv_groups;
    const int seq_len = pos0 + t + 1;
    const int cur_pos = seq_len - 1;   /* this query row's absolute position */
    const int row_bytes = fc_kvq4_row_bytes(head_dim);
    /* Liveness floor for the fp16 window ring. Every row of this call was
     * appended BEFORE any row reads, so slot (p % WMAX) holds position p only
     * while p + WMAX > last_pos, where last_pos is the highest position this
     * call appended (row gridDim.x-1). Keys below that floor were lapped by a
     * LATER row of the same batch; reading them would attend to a future
     * token. They are still exact in the q4 body, so fall back to it. With the
     * host capping a batch at (WMAX - win) rows this never fires; it is the
     * backstop that keeps an oversized batch merely lossier, never acausal. */
    const int win_live_lo = (cur_pos - t + (int)gridDim.x - 1) - FC_KVQ4_WMAX;

    const int kstride = FC_ATTN_K_STRIDE(head_dim);
    extern __shared__ float sh[];
    float *q_s = sh;
    float *k_s = q_s + (size_t)G * head_dim;
    float *score_s = k_s + (size_t)FC_ATTN_TILE_KEYS * kstride;
    float *p_s = score_s + G * FC_ATTN_TILE_KEYS;
    float *m_s = p_s + G * FC_ATTN_TILE_KEYS;
    float *l_s = m_s + G;
    float *corr_s = l_s + G;

    const size_t n_head = (size_t)gridDim.y * G;
    const float *q_base = q + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;
    for (int g = 0; g < G; g++)
        q_s[(size_t)g * head_dim + tid] = q_base[(size_t)g * head_dim + tid];
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    /* Compile-time width, so ptxas keeps acc[] in REGISTERS. Every loop that
     * indexes acc[] is unrolled to FC_ATTN_MAX_GROUPS with an `if (g < G)`
     * guard; a loop bounded by the RUNTIME kv_groups cannot be unrolled, the
     * array lands in LOCAL memory, and the PV inner loop then pays an
     * LDL.128/STL.128 round trip per key. ptxas reports that as a 32-byte
     * "stack frame" and prints "0 bytes spill stores" right beside it --
     * a never-promoted local array is not counted as a spill. */
    float acc[FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        acc[g] = 0.0f;

    const unsigned char *k_head = k_cache + (size_t)kvh * max_seq * row_bytes;
    const unsigned char *v_head = v_cache + (size_t)kvh * max_seq * row_bytes;
    const __half *k_win_head = k_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *v_win_head = v_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *k_sink_head = k_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const __half *v_sink_head = v_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = head_dim / FC_ATTN_DOT_LANES;

    for (int j0 = 0; j0 < seq_len; j0 += FC_ATTN_TILE_KEYS) {
        int tk = seq_len - j0;
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;

        /* Sliding window: skip tiles outside sink U recent-window (attn_sink/
         * attn_win are the ATTENDED set, orthogonal to sink/win which pick the
         * fp16-shadow vs q4-dequant read path). Uniform across the block. */
        if (!fc_attn_tile_in_window(j0, tk, cur_pos, attn_sink, attn_win))
            continue;

        /* Stage the K tile (read once, reused by all G heads): fp16 shadow for
         * sink/recent keys, in-kernel q4 dequant for the aged-out middle. */
        for (int key = 0; key < tk; key++) {
            int kp = j0 + key;
            float *krow = k_s + (size_t)key * kstride;
            if (kp < sink) {
                const __half *s = k_sink_head + (size_t)kp * head_dim;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    krow[e] = __half2float(s[e]);
            } else if (win > 0 && kp > cur_pos - win && kp > win_live_lo) {
                const __half *s = k_win_head +
                    (size_t)(kp % FC_KVQ4_WMAX) * head_dim;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    krow[e] = __half2float(s[e]);
            } else {
                const unsigned char *s = k_head + (size_t)kp * row_bytes;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    krow[e] = fc_kvq4_dequant_elem(s, e);
            }
        }
        __syncthreads();

        /* QK^T, i-outer / g-inner: each thread reads its chunk of the staged K
         * row ONCE per tile into a register and fans it across the whole GQA
         * group, instead of re-reading k_s once per group (G=8 -> 1 LDS per
         * element). BIT-EXACT: every part[g] still accumulates over e in the
         * same ascending order the g-outer form used, so this is pure operand
         * motion. part[] is indexed only by the unrolled compile-time g, for
         * the same reason acc[] is -- a runtime-bounded loop would push it to
         * LOCAL memory and cost an LDL/STL round trip per element. */
        {
            float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                part[g] = 0.0f;
            if (ky < tk) {
                const float *kv = k_s + (size_t)ky * kstride;
                for (int i = 0; i < chunk; i++) {
                    int e = lane + i * FC_ATTN_DOT_LANES;
                    float kve = kv[e];
#pragma unroll
                    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                        if (g < G)
                            part[g] += q_s[(size_t)g * head_dim + e] * kve;
                }
            }
            /* G is uniform across the block, so the guarded unroll keeps the
             * shuffle warp-uniform exactly as the runtime-bounded loop did. */
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    float p = part[g];
                    for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,
                                              FC_ATTN_DOT_LANES);
                    if (lane == 0 && ky < tk)
                        score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;
                }
            }
        }
        __syncthreads();

        if (tid < G) {
            const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
            float m_new = m_s[tid];
            for (int j = 0; j < tk; j++)
                m_new = fmaxf(m_new, sc[j]);
            float corr = expf(m_s[tid] - m_new);
            float sum = 0.0f;
            for (int j = 0; j < tk; j++) {
                float pj = expf(sc[j] - m_new);
                p_s[(size_t)j * G + tid] = pj;   /* [TILE][G]: PV reads a run */
                sum += pj;
            }
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = m_new;
            corr_s[tid] = corr;
        }
        __syncthreads();

        /* PV accumulate: thread tid owns out dim tid; V rows dequantized on
         * read (element tid of each V row). */
        for (int j = 0; j < tk; j++) {
            int vp = j0 + j;
            float vv;
            if (vp < sink)
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);
            else if (win > 0 && vp > cur_pos - win && vp > win_live_lo)
                vv = __half2float(v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]);
            else
                vv = fc_kvq4_dequant_elem(v_head + (size_t)vp * row_bytes, tid);
            const float *pj_s = p_s + (size_t)j * G;
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    if (j == 0)
                        acc[g] *= corr_s[g];
                    acc[g] += pj_s[g] * vv;
                }
            }
        }
        __syncthreads();
    }

    float *out_base = out + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        if (g < G)
            out_base[(size_t)g * head_dim + tid] = acc[g] / l_s[g];
}

/* ============================================================
 * REGISTER-K prefill variant (FC_ATTN_KTILE_REG) — BIT-EXACT, DEFAULT OFF
 *
 * Same tiling, same online softmax, same q4 KV cache and the same accumulation
 * ORDER as fc_attn_flash_prefill_q4. One thing changes: the staged K tile is
 * DELETED and K lives in per-thread registers instead.
 *
 * Why that is free. In the QK dot, thread (ky, lane) reads exactly the
 * head_dim/16 elements k[ky][lane + 16*i] of key ky and nothing else — the
 * shared tile is written by one thread and read back by that same thread. It
 * exists only because the STAGING loop assigns elements element-outer
 * (`for e = tid; e < head_dim; e += blockDim.x`) while the CONSUMING loop reads
 * them lane-strided. Re-assigning the staging to match the consumer removes the
 * STS/LDS round trip and the tile with it:
 *
 *   shared  26 720 -> 9 312 B      residency  4 -> 7 blocks/SMX
 *   registers 72 (unchanged)       stack frame 0, no LDL/STL
 *
 * Measured on the standalone microbenchmark (cuda_kernels/tests/exp_attn_qtile.cu
 * arm `xq_q1_kr`, driven by cuda/tests/bench_attn_qtile.c): 1.51x on the kernel
 * at the shipped shape, depth-flat 1 k -> 32 k, at both 666 and 875 MHz, and
 * BIT-EXACT — 0 of 524 288 output floats differ from fc_attn_flash_prefill_q4.
 * The factor decomposes as 1.194x structural (the round trip) x 1.269x from the
 * residency gain. Q-tiling the same body was REFUTED there (Q_TILE=2 costs
 * shared and comes out at 0.886x); this variant is Q_TILE=1 only.
 *
 * TWO THINGS ARE LOAD-BEARING AND MUST NOT BE "SIMPLIFIED":
 *
 *  1. The HOISTED dequant (fc_kvq4_dequant_lane_chunk) is the enabler, not a
 *     detail. Calling fc_kvq4_dequant_elem once per element instead costs
 *     105 REGISTERS, which caps residency back at 4 blocks/SMX and collapses
 *     the measured gain from 1.51x to 1.068x (microbench arm `xq_q1_krn`).
 *  2. The online-softmax rescale stays FUSED in the PV loop at j == 0, exactly
 *     where fc_attn_flash_prefill_q4 puts it, because ptxas contracts
 *     `acc *= corr; acc += p0*v0` into a single `acc = fma(acc, corr, p0*v0)` —
 *     one rounding instead of two. Hoisting it out of the j loop is
 *     algebraically identical and moved 148 099 of 524 288 output floats by up
 *     to 4.0e-7 relative (1-ulp class), measured. Keep it at j == 0.
 *
 * CONTRACT (enforced host-side in fc_attn_flash_prefill_q4_ctx):
 *   - head_dim <= FC_ATTN_KREG_MAX_CHUNK * FC_ATTN_DOT_LANES (= 256), so the
 *     kreg[] array is bounded by a COMPILE-TIME constant. An array bounded by
 *     the runtime `chunk` is never promoted to registers: it lands in local
 *     memory and ptxas prints "0 bytes spill" beside a nonzero stack frame.
 *   - FC_ATTN_DOT_LANES == 16 (the hoisted dequant's nibble mapping).
 *   - attn_win <= 0 (full attention). The tile predicate here is the shipped
 *     one and a windowed prefill should behave identically, but that is
 *     UNMEASURED, so the host declines attn_win > 0 and uses the shipped
 *     kernel.
 * ============================================================ */

#if FC_ATTN_DOT_LANES != 16
#error "fc_kvq4_dequant_lane_chunk assumes FC_ATTN_DOT_LANES == 16"
#endif

/* Compile-time bound on head_dim / FC_ATTN_DOT_LANES; 16 covers head_dim 256. */
#define FC_ATTN_KREG_MAX_CHUNK 16

/* Decode the elements a QK-dot lane owns, lane + 16*i for i < chunk, into
 * kreg[i]. Element lane + 16*i sits in block i/2 at code byte 2 + lane: the low
 * nibble for even i, the high nibble for odd i. So one scale and one code byte
 * per block serve two kreg entries. kreg[chunk..] is left untouched.
 *
 * Unrolled over the compile-time bound so every kreg index is a constant; a
 * runtime index would demote kreg to local memory. Requires lane < 16 and
 * chunk <= FC_ATTN_KREG_MAX_CHUNK. Bit-equal to fc_kvq4_dequant_elem. */
__device__ __forceinline__ void fc_kvq4_dequant_lane_chunk(
        const unsigned char *row, int lane, int chunk, float *kreg) {
#pragma unroll
    for (int b = 0; b < FC_ATTN_KREG_MAX_CHUNK / 2; b++) {
        if (2 * b < chunk) {
            const unsigned char *bp = row + (size_t)b * FC_KVQ4_BLK_BYTES;
            const unsigned short d_bits = (unsigned short)(bp[0] | (bp[1] << 8));
            const float d = __half2float(__ushort_as_half(d_bits));
            const unsigned int codes = bp[2 + lane];
            kreg[2 * b] = (float)((int)(codes & 0xF) - 8) * d;
            if (2 * b + 1 < chunk)
                kreg[2 * b + 1] = (float)((int)((codes >> 4) & 0xF) - 8) * d;
        }
    }
}

/*
 * Causal GQA flash attention over the Q4 GPU KV cache (prefill mode), with the
 * K tile held in REGISTERS instead of shared memory. Bit-exact against
 * fc_attn_flash_prefill_q4; see the block comment above for the contract, the
 * measured factor and the two things that must not be changed.
 *
 * Args, grid and block are identical to fc_attn_flash_prefill_q4 — only the
 * dynamic-shared request differs (fc_attn_flash_kreg_shared_bytes in
 * cuda/fc_cuda.c).
 */
__global__ void fc_attn_flash_prefill_q4_kr(const float *q,
                                            const unsigned char *k_cache,
                                            const unsigned char *v_cache,
                                            const __half *k_sink,
                                            const __half *v_sink,
                                            const __half *k_win,
                                            const __half *v_win,
                                            float *out, int kv_groups,
                                            int head_dim, int max_seq, int pos0,
                                            int sink, int win, int attn_sink,
                                            int attn_win, float scale) {
    const int t = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    const int G = kv_groups;
    const int seq_len = pos0 + t + 1;
    const int cur_pos = seq_len - 1;
    const int row_bytes = fc_kvq4_row_bytes(head_dim);
    const int win_live_lo = (cur_pos - t + (int)gridDim.x - 1) - FC_KVQ4_WMAX;

    /* No K tile: q group + scores + probs + m/l/corr only. MUST match
     * fc_attn_flash_kreg_shared_bytes() in cuda/fc_cuda.c. */
    extern __shared__ float shk[];
    float *q_s = shk;
    float *score_s = q_s + (size_t)G * head_dim;
    float *p_s = score_s + G * FC_ATTN_TILE_KEYS;
    float *m_s = p_s + G * FC_ATTN_TILE_KEYS;
    float *l_s = m_s + G;
    float *corr_s = l_s + G;

    const size_t n_head = (size_t)gridDim.y * G;
    const float *q_base = q + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;
    for (int g = 0; g < G; g++)
        q_s[(size_t)g * head_dim + tid] = q_base[(size_t)g * head_dim + tid];
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    float acc[FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        acc[g] = 0.0f;

    const unsigned char *k_head = k_cache + (size_t)kvh * max_seq * row_bytes;
    const unsigned char *v_head = v_cache + (size_t)kvh * max_seq * row_bytes;
    const __half *k_win_head = k_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *v_win_head = v_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *k_sink_head = k_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const __half *v_sink_head = v_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;   /* key-in-tile this thread scores */
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = head_dim / FC_ATTN_DOT_LANES;

    for (int j0 = 0; j0 < seq_len; j0 += FC_ATTN_TILE_KEYS) {
        int tk = seq_len - j0;
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;

        if (!fc_attn_tile_in_window(j0, tk, cur_pos, attn_sink, attn_win))
            continue;

        /* Load key ky's own 16 elements straight into registers — the same read
         * path fc_attn_flash_prefill_q4's staging loop uses (fp16 shadow for
         * sink/recent keys, q4 dequant for the aged-out middle), just assigned
         * to the thread that consumes it. No shared write, so no sync. */
        float kreg[FC_ATTN_KREG_MAX_CHUNK];
        if (ky < tk) {
            const int kp = j0 + ky;
            if (kp < sink) {
                const __half *s = k_sink_head + (size_t)kp * head_dim;
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++)
                    if (i < chunk)
                        kreg[i] = __half2float(s[lane + i * FC_ATTN_DOT_LANES]);
            } else if (win > 0 && kp > cur_pos - win && kp > win_live_lo) {
                const __half *s = k_win_head +
                    (size_t)(kp % FC_KVQ4_WMAX) * head_dim;
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++)
                    if (i < chunk)
                        kreg[i] = __half2float(s[lane + i * FC_ATTN_DOT_LANES]);
            } else {
                fc_kvq4_dequant_lane_chunk(k_head + (size_t)kp * row_bytes,
                                           lane, chunk, kreg);
            }
        }

        /* QK^T, i-outer / g-inner. Same accumulation order over e as
         * fc_attn_flash_prefill_q4 — the K operand simply comes from a register
         * instead of from k_s. */
        {
            float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                part[g] = 0.0f;
            if (ky < tk) {
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++) {
                    if (i < chunk) {
                        const int e = lane + i * FC_ATTN_DOT_LANES;
                        const float kve = kreg[i];
#pragma unroll
                        for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                            if (g < G)
                                part[g] += q_s[(size_t)g * head_dim + e] * kve;
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    float p = part[g];
                    for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,
                                              FC_ATTN_DOT_LANES);
                    if (lane == 0 && ky < tk)
                        score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;
                }
            }
        }
        __syncthreads();

        if (tid < G) {
            const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
            float m_new = m_s[tid];
            for (int j = 0; j < tk; j++)
                m_new = fmaxf(m_new, sc[j]);
            float corr = expf(m_s[tid] - m_new);
            float sum = 0.0f;
            for (int j = 0; j < tk; j++) {
                float pj = expf(sc[j] - m_new);
                p_s[(size_t)j * G + tid] = pj;
                sum += pj;
            }
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = m_new;
            corr_s[tid] = corr;
        }
        __syncthreads();

        /* PV accumulate — byte-for-byte the shipped loop, INCLUDING the rescale
         * fused at j == 0 (see point 2 of the block comment). */
        for (int j = 0; j < tk; j++) {
            int vp = j0 + j;
            float vv;
            if (vp < sink)
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);
            else if (win > 0 && vp > cur_pos - win && vp > win_live_lo)
                vv = __half2float(v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]);
            else
                vv = fc_kvq4_dequant_elem(v_head + (size_t)vp * row_bytes, tid);
            const float *pj_s = p_s + (size_t)j * G;
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    if (j == 0)
                        acc[g] *= corr_s[g];
                    acc[g] += pj_s[g] * vv;
                }
            }
        }
        __syncthreads();
    }

    float *out_base = out + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        if (g < G)
            out_base[(size_t)g * head_dim + tid] = acc[g] / l_s[g];
}

/* ============================================================
 * SHARED-V prefill variant (FC_ATTN_VSTAGE) — BIT-EXACT, DEFAULT OFF
 *
 * fc_attn_flash_prefill_q4_kr with one thing changed: the key tile's V rows are
 * staged into shared memory AS RAW Q4 BYTES once per tile, and the per-key
 * dequant reads shared instead of global.
 *
 * WHY, and it is the mirror image of the register-K finding. Register-K deleted
 * the shared K tile because its reuse was 1:1 — one writer, and the only reader
 * was the same thread. V's reuse is 32:1 and it had NO tile: a V row is
 * row_bytes (144 at the shipped shape), and the block issued 256 threads x 3
 * byte loads = 768 load instructions to read it, because all 32 threads of a
 * Q4_0 block load the same 2-byte fp16 scale and each nibble byte serves 2
 * threads. The staging pattern (none) did not match the consuming pattern
 * (block-wide). So the same question that said "delete the tile" for K says
 * "add one" for V.
 *
 * MEASURED, die 6 pinned 666 MHz, noise floor 0.301 % over 10 processes, three
 * interleaved reps per arm, all bit-EXACT on 524 288 outputs:
 *
 *   VIND  (the address-arithmetic half alone, V still in global)   1.0418x
 *   THIS  (VIND + the shared V tile)                              1.1497x
 *   the shared-V half on its own                                  1.0959x
 *
 * DEPTH-FLAT: 1.1453 / 1.1455 / 1.1479 at 8 192 / 14 080 / 32 000 tokens.
 *
 * It is FREE in residency. Shared goes 9 312 -> 11 616 B and 7 blocks/SMX is
 * unchanged because 7 is the REGISTER cap at 72 registers x 256 threads
 * (floor(131072/(72*256)) = 7; 80 registers would give 6). Registers stay at
 * 72 — COUNTED, and it was this lever's whole risk. Staging is placed in the
 * SOFTMAX SHADOW: the online softmax is `if (tid < G)`, so 248 of 256 threads
 * are idle there, and the copy is covered by the __syncthreads() that already
 * separates the softmax from the PV loop. No barrier is added.
 *
 * WHY IT PAYS MORE THAN ITS INSTRUCTION COUNT: the per-key 64-bit global address
 * chain (29 of the PV loop's 97 executed instructions, 15 of them loop-invariant
 * and recomputed all 16 times) collapses to 3 instructions of 32-bit shared
 * addressing, and the three LD.E.U8 become ONE LDS.U16 plus one LDS.U8 — in
 * shared ptxas CAN prove the 2-byte alignment of the fp16 scale that it cannot
 * prove for v_head + vp*row_bytes with a runtime row_bytes, so the PRMT that
 * reassembled it disappears. But the counted instruction cut is only 15.7 % and
 * the measured gain is 15.0 %, i.e. instruction reduction converts at ~27 % on
 * this kernel: the rest is LATENCY. The old form exposed 16 serial global round
 * trips per warp per tile at 3 loads in flight; this one exposes 3 at tile entry
 * and reads shared thereafter.
 *
 * THREE THINGS ARE LOAD-BEARING AND MUST NOT BE "SIMPLIFIED":
 *
 *  1. Everything fc_attn_flash_prefill_q4_kr's own block comment says is still
 *     load-bearing here, because this kernel is that kernel: the HOISTED K
 *     dequant keeps registers at 72, and the online-softmax rescale stays FUSED
 *     in the PV loop at j == 0 where ptxas contracts it to
 *     acc = fma(acc, corr, p0*v0). Hoisting it is algebraically identical and
 *     moves 148 099 of 524 288 outputs by up to 4.0e-7 relative.
 *  2. The three staging rounds are written OUT rather than looped under a
 *     #pragma unroll. A rolled runtime-bounded copy is the exact pathology the
 *     matvec staging gate priced at 1.426x for 1 -> 8 loads in flight, and a
 *     dropped unroll would compile it silently — nvcc emitted "this pragma must
 *     immediately precede a statement" for the looped version.
 *  3. The dequant is INLINE, not a call to fc_kvq4_dequant_elem. Passing a
 *     shared pointer into a function taking a generic pointer lets ptxas fall
 *     back to generic LD instead of LDS. Inline, the SASS is LDS.U16 + LDS.U8 —
 *     verified in cuobjdump, which is the only reason the latency win exists.
 *
 * CONTRACT (enforced host-side in fc_attn_flash_prefill_q4_ctx):
 *   - everything fc_attn_flash_prefill_q4_kr requires, plus
 *   - row_bytes % 4 == 0, so the tile copies as 32-bit words. True for every
 *     head_dim that is a multiple of 64; head_dim is pinned to exactly 256 by
 *     the kr contract, giving row_bytes 144.
 *   - the staged tile must fit: shared <= 48 KB, checked by the host.
 * ============================================================ */

/* Words in the staged V tile at the contracted shape, and the number of
 * blockDim-strided rounds needed to copy it. head_dim is pinned to
 * FC_ATTN_TILE_KEYS * FC_ATTN_DOT_LANES by the kr contract, and a block has
 * head_dim threads, so both are compile-time constants. */
#define FC_ATTN_VS_ROW_BYTES  ((FC_ATTN_TILE_KEYS * FC_ATTN_DOT_LANES /        \
                                FC_KVQ4_BLK) * FC_KVQ4_BLK_BYTES)
#define FC_ATTN_VS_TILE_WORDS ((FC_ATTN_TILE_KEYS * FC_ATTN_VS_ROW_BYTES) / 4)
#define FC_ATTN_VS_THREADS    (FC_ATTN_TILE_KEYS * FC_ATTN_DOT_LANES)
#define FC_ATTN_VS_ROUNDS     ((FC_ATTN_VS_TILE_WORDS + FC_ATTN_VS_THREADS - 1) \
                               / FC_ATTN_VS_THREADS)
#if FC_ATTN_VS_ROUNDS != 3
#error "fc_attn_flash_prefill_q4_kvs stages the V tile in exactly 3 rounds"
#endif
#if (FC_ATTN_TILE_KEYS * FC_ATTN_VS_ROW_BYTES) % 4 != 0
#error "the staged V tile must be a whole number of 32-bit words"
#endif

/* Falsification hook for the in-model digest, default OFF. See the use site. */
#ifndef FC_ATTN_VSTAGE_POISON
#define FC_ATTN_VSTAGE_POISON 0
#endif

/*
 * Causal GQA flash attention over the Q4 GPU KV cache (prefill mode):
 * fc_attn_flash_prefill_q4_kr with each key tile's V rows staged in shared
 * memory as raw Q4 bytes and decoded from there. Bit-exact against
 * fc_attn_flash_prefill_q4 and _kr; see the block comment above for why and for
 * the contract.
 *
 * Args, grid and block are identical to fc_attn_flash_prefill_q4; dynamic
 * shared is fc_attn_flash_kvs_shared_bytes in cuda/fc_cuda.c.
 */
__global__ void fc_attn_flash_prefill_q4_kvs(const float *q,
                                             const unsigned char *k_cache,
                                             const unsigned char *v_cache,
                                             const __half *k_sink,
                                             const __half *v_sink,
                                             const __half *k_win,
                                             const __half *v_win,
                                             float *out, int kv_groups,
                                             int head_dim, int max_seq, int pos0,
                                             int sink, int win, int attn_sink,
                                             int attn_win, float scale) {
    const int t = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    const int G = kv_groups;
    const int seq_len = pos0 + t + 1;
    const int cur_pos = seq_len - 1;
    const int row_bytes = fc_kvq4_row_bytes(head_dim);
    /* Ring slots below the batch's highest appended position - WMAX are lapped. */
    const int win_live_lo = (cur_pos - t + (int)gridDim.x - 1) - FC_KVQ4_WMAX;

    /* kr's layout plus the staged V tile after corr_s. MUST match
     * fc_attn_flash_kvs_shared_bytes() in cuda/fc_cuda.c. */
    extern __shared__ float shv[];
    float *q_s = shv;
    float *score_s = q_s + (size_t)G * head_dim;
    float *p_s = score_s + G * FC_ATTN_TILE_KEYS;
    float *m_s = p_s + G * FC_ATTN_TILE_KEYS;
    float *l_s = m_s + G;
    float *corr_s = l_s + G;
    unsigned int *v_sh = (unsigned int *)(corr_s + G);

    const size_t n_head = (size_t)gridDim.y * G;
    const float *q_base = q + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;
    for (int g = 0; g < G; g++)
        q_s[(size_t)g * head_dim + tid] = q_base[(size_t)g * head_dim + tid];
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    float acc[FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        acc[g] = 0.0f;

    const unsigned char *k_head = k_cache + (size_t)kvh * max_seq * row_bytes;
    const unsigned char *v_head = v_cache + (size_t)kvh * max_seq * row_bytes;
    const __half *k_win_head = k_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *v_win_head = v_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *k_sink_head = k_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const __half *v_sink_head = v_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;   /* key-in-tile this thread scores */
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = head_dim / FC_ATTN_DOT_LANES;

    /* Where element tid sits inside a staged V row; fixed per thread. */
    const int vlocal = tid & (FC_KVQ4_BLK - 1);
    const int vhalf = FC_KVQ4_BLK / 2;
    const int vbyte = 2 + (vlocal < vhalf ? vlocal : vlocal - vhalf);
    const int vshift = vlocal < vhalf ? 0 : 4;
    const int vqoff = (tid / FC_KVQ4_BLK) * FC_KVQ4_BLK_BYTES;

    for (int j0 = 0; j0 < seq_len; j0 += FC_ATTN_TILE_KEYS) {
        int tk = seq_len - j0;
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;

        if (!fc_attn_tile_in_window(j0, tk, cur_pos, attn_sink, attn_win))
            continue;

        /* K for key ky straight into registers, as in _kr. */
        float kreg[FC_ATTN_KREG_MAX_CHUNK];
        if (ky < tk) {
            const int kp = j0 + ky;
            if (kp < sink) {
                const __half *s = k_sink_head + (size_t)kp * head_dim;
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++)
                    if (i < chunk)
                        kreg[i] = __half2float(s[lane + i * FC_ATTN_DOT_LANES]);
            } else if (win > 0 && kp > cur_pos - win && kp > win_live_lo) {
                const __half *s = k_win_head +
                    (size_t)(kp % FC_KVQ4_WMAX) * head_dim;
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++)
                    if (i < chunk)
                        kreg[i] = __half2float(s[lane + i * FC_ATTN_DOT_LANES]);
            } else {
                fc_kvq4_dequant_lane_chunk(k_head + (size_t)kp * row_bytes,
                                           lane, chunk, kreg);
            }
        }

        /* QK^T, i-outer / g-inner; every thread joins the shuffles. */
        {
            float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                part[g] = 0.0f;
            if (ky < tk) {
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++) {
                    if (i < chunk) {
                        const int e = lane + i * FC_ATTN_DOT_LANES;
                        const float kve = kreg[i];
#pragma unroll
                        for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                            if (g < G)
                                part[g] += q_s[(size_t)g * head_dim + e] * kve;
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    float p = part[g];
                    for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,
                                              FC_ATTN_DOT_LANES);
                    if (lane == 0 && ky < tk)
                        score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;
                }
            }
        }
        __syncthreads();

        if (tid < G) {
            const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
            float m_new = m_s[tid];
            for (int j = 0; j < tk; j++)
                m_new = fmaxf(m_new, sc[j]);
            float corr = expf(m_s[tid] - m_new);
            float sum = 0.0f;
            for (int j = 0; j < tk; j++) {
                float pj = expf(sc[j] - m_new);
                p_s[(size_t)j * G + tid] = pj;
                sum += pj;
            }
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = m_new;
            corr_s[tid] = corr;
        }

        /* Stage the tile's V rows as words while the softmax threads work; the
         * barrier below covers both. Three rounds written out so all three
         * loads are in flight before the first store. Every load is guarded: a
         * partial last tile is shorter than one round and must not read past
         * the end of v_cache. */
        {
            const unsigned int *vsrc =
                (const unsigned int *)(v_head + (size_t)j0 * row_bytes);
            const int vwords = (tk * row_bytes) >> 2;
            const int w0_idx = tid;
            const int w1_idx = tid + FC_ATTN_VS_THREADS;
            const int w2_idx = tid + 2 * FC_ATTN_VS_THREADS;
            const unsigned int w0 = w0_idx < vwords ? vsrc[w0_idx] : 0u;
            const unsigned int w1 = w1_idx < vwords ? vsrc[w1_idx] : 0u;
            const unsigned int w2 = w2_idx < vwords ? vsrc[w2_idx] : 0u;
            if (w0_idx < vwords)
                v_sh[w0_idx] = w0;
            if (w1_idx < vwords)
                v_sh[w1_idx] = w1;
            if (w2_idx < vwords)
                v_sh[w2_idx] = w2;
        }
        __syncthreads();

        /* PV accumulate with the rescale fused at j == 0, as in _kr. */
        for (int j = 0; j < tk; j++) {
            int vp = j0 + j;
            float vv;
            if (vp < sink) {
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);
            } else if (win > 0 && vp > cur_pos - win && vp > win_live_lo) {
                vv = __half2float(v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]);
            } else {
                /* Inline on purpose: through a generic-pointer helper ptxas
                 * loses the shared address space and emits generic loads. */
                const unsigned char *bp =
                    (const unsigned char *)v_sh + j * row_bytes + vqoff;
                const unsigned short d_bits = (unsigned short)(bp[0] | (bp[1] << 8));
                const float d = __half2float(__ushort_as_half(d_bits));
                const int code = (bp[vbyte] >> vshift) & 0xF;
                vv = (float)(code - 8) * d;
#if FC_ATTN_VSTAGE_POISON
                vv = __fmul_rn(vv, 1.00000011920928955078125f);
#endif
            }
            const float *pj_s = p_s + (size_t)j * G;
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    if (j == 0)
                        acc[g] *= corr_s[g];
                    acc[g] += pj_s[g] * vv;
                }
            }
        }
        __syncthreads();
    }

    float *out_base = out + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        if (g < G)
            out_base[(size_t)g * head_dim + tid] = acc[g] / l_s[g];
}

/* ============================================================
 * NARROW-STAGING prefill variants (FC_ATTN_KTILE_FP16) — LOSSY, DEFAULT OFF
 *
 * Same tiling, same online softmax and the same q4 KV cache as
 * fc_attn_flash_prefill_q4. ONE thing changes now:
 *
 *  1. The i-outer QK dot with eight REGISTER partials — which used to be a
 *     difference here — is on the DEFAULT path since the lossless landing, so
 *     all six shipped flash kernels carry it. It is what makes (2) affordable
 *     (a narrow tile costs one widening per K LOAD, and this cuts loads 8x),
 *     but it is no longer part of what this flag turns on.
 *  2. The staged K tile (and, at level 2, the staged q group) is __half
 *     instead of float. That is the whole point: the K tile is 17 408 B of
 *     the 26 720 B footprint, and residency on this kernel is capped by
 *     SHARED MEMORY, never by registers (72 regs would allow 7 blocks/SMX).
 *     26 720 -> 18 016 B lifts residency 4 -> 6 blocks/SMX; also narrowing q
 *     gives 13 920 B and 7 blocks/SMX.
 *
 * NUMERICS — this is NOT bit-exact, which is why it is behind a flag:
 *   - sink and recent-window keys are ALREADY fp16 in memory (k_sink/k_win);
 *     staging them as __half is bit-exact for those keys and today's widening
 *     to float is pure waste.
 *   - aged-out keys are q4-sourced (fp16 scale x a 4-bit level), so rounding
 *     them to fp16 costs <= 2^-11 relative — orders below the q4 error the
 *     cache already carries.
 *   - level 2 additionally rounds the q group, a live activation. Strictly
 *     lossier and gated separately.
 * Measured max relative output error at the shipped shape: 1.3e-5 (level 1),
 * 1.7e-5 (level 2). See Pantheon results/prefill_attention_ktile_fp16.md.
 *
 * TODO: attention.cu is past the 1 000-line guidance and now carries THREE
 * prefill families (exact, narrow-staged, register-K) plus the whole decode
 * family. Splitting it into a prefill file and a decode file is still the right
 * move, and it needs a SASS-identity check as its gate: extracting code out of a
 * flash body is NOT free here even behind __forceinline__. Pulling the
 * register-K path's three-way K load into one helper was tried and measured to
 * CHANGE that kernel's SASS (same 72 registers, same 0-byte stack frame,
 * different instruction stream), so it was reverted rather than shipped on top
 * of a measurement it would have invalidated. Same reason the register-K body is
 * a copy of fc_attn_flash_prefill_q4's rather than a shared template.
 * See Pantheon results/prefill_attention_register_ktile.md §8 gap 14.
 * ============================================================ */

/* K-tile row stride in ELEMENTS is FC_ATTN_K_STRIDE(head_dim) for both the
 * float and the __half tile: two __halfs inside one 32-bit word are
 * conflict-free on sm_37 (the sub-word rule), and the +16-element pad still
 * puts the two half-warps of a warp on disjoint bank ranges.
 *
 * The shared body is a TEMPLATE, which cannot carry C linkage, so the file's
 * enclosing `extern "C"` block closes here and reopens for the two entry
 * points — whose PTX names must stay unmangled, since cuModuleGetFunction
 * looks them up by string. */
} /* extern "C" */

#include "attention_prefill_g8.cuh"

/* FC_ATTN_G8 (default on): fc_attn_flash_prefill_q4_kvs specialised for
 * kv_groups 8 / head_dim 256: vector q loads, the tile softmax on 128 threads
 * and the next K tile staged in shared (forms QVEC + PSOFT + KSH of
 * attention_prefill_g8.cuh; microbench 0 diffs, 1.60-1.70x over kvs at depth
 * 8k-32k). Same args and grid as kvs; kv_groups / head_dim are checked
 * host-side and unused here.
 * The launch bound holds it at kvs's 72 registers (7 blocks/SMX): with the
 * shape constant ptxas otherwise hoists loads up to 126 registers (4 blocks). */
extern "C" __global__ void __launch_bounds__(256, 7) fc_attn_flash_prefill_q4_g8(
        const float *q, const unsigned char *k_cache,
        const unsigned char *v_cache, const __half *k_sink,
        const __half *v_sink, const __half *k_win, const __half *v_win,
        float *out, int kv_groups, int head_dim, int max_seq, int pos0,
        int sink, int win, int attn_sink, int attn_win, float scale)
{
    (void)kv_groups;
    (void)head_dim;
    fc_attn_flash_prefill_q4_g8_body<true, true, false, true, false>(
        q, k_cache, v_cache, k_sink, v_sink, k_win, v_win, out, max_seq, pos0,
        sink, win, attn_sink, attn_win, scale);
}

template <bool Q_FP16>
__device__ __forceinline__ void fc_attn_flash_prefill_q4_narrow(
        const float *q, const unsigned char *k_cache,
        const unsigned char *v_cache, const __half *k_sink,
        const __half *v_sink, const __half *k_win, const __half *v_win,
        float *out, int kv_groups, int head_dim, int max_seq, int pos0,
        int sink, int win, int attn_sink, int attn_win, float scale)
{
    const int t = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    const int G = kv_groups;
    const int seq_len = pos0 + t + 1;
    const int cur_pos = seq_len - 1;
    const int row_bytes = fc_kvq4_row_bytes(head_dim);
    const int win_live_lo = (cur_pos - t + (int)gridDim.x - 1) - FC_KVQ4_WMAX;

    /* Byte-addressed shared layout so the fp32 softmax state stays 4-byte
     * aligned whichever tiles are narrowed. MUST match
     * fc_attn_flash_narrow_shared_bytes() in cuda/fc_cuda.c. */
    const int kstride = FC_ATTN_K_STRIDE(head_dim);
    extern __shared__ char shb[];
    const int q_bytes = (int)(G * head_dim *
                              (Q_FP16 ? sizeof(__half) : sizeof(float)));
    const int k_bytes = (int)(FC_ATTN_TILE_KEYS * kstride * sizeof(__half));
    float *q_f = (float *)shb;
    __half *q_h = (__half *)shb;
    __half *k_h = (__half *)(shb + q_bytes);
    float *score_s = (float *)(shb + q_bytes + k_bytes);
    float *p_s = score_s + G * FC_ATTN_TILE_KEYS;
    float *m_s = p_s + G * FC_ATTN_TILE_KEYS;
    float *l_s = m_s + G;
    float *corr_s = l_s + G;

    const size_t n_head = (size_t)gridDim.y * G;
    const float *q_base = q + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;
    for (int g = 0; g < G; g++) {
        float qv = q_base[(size_t)g * head_dim + tid];
        if (Q_FP16) q_h[(size_t)g * head_dim + tid] = __float2half(qv);
        else        q_f[(size_t)g * head_dim + tid] = qv;
    }
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    float acc[FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        acc[g] = 0.0f;

    const unsigned char *k_head = k_cache + (size_t)kvh * max_seq * row_bytes;
    const unsigned char *v_head = v_cache + (size_t)kvh * max_seq * row_bytes;
    const __half *k_win_head = k_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *v_win_head = v_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *k_sink_head = k_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const __half *v_sink_head = v_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = head_dim / FC_ATTN_DOT_LANES;

    for (int j0 = 0; j0 < seq_len; j0 += FC_ATTN_TILE_KEYS) {
        int tk = seq_len - j0;
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;
        if (!fc_attn_tile_in_window(j0, tk, cur_pos, attn_sink, attn_win))
            continue;

        /* Stage the K tile as __half. The fp16 shadow rows copy across with no
         * conversion at all (they were fp16 to begin with); only the q4 body
         * rounds. */
        for (int key = 0; key < tk; key++) {
            int kp = j0 + key;
            const size_t base = (size_t)key * kstride;
            if (kp < sink) {
                const __half *s = k_sink_head + (size_t)kp * head_dim;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    k_h[base + e] = s[e];
            } else if (win > 0 && kp > cur_pos - win && kp > win_live_lo) {
                const __half *s = k_win_head +
                    (size_t)(kp % FC_KVQ4_WMAX) * head_dim;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    k_h[base + e] = s[e];
            } else {
                const unsigned char *s = k_head + (size_t)kp * row_bytes;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    k_h[base + e] = __float2half(fc_kvq4_dequant_elem(s, e));
            }
        }
        __syncthreads();

        /* QK^T, i-outer: one K load (and one widening) per element per tile,
         * eight register partials across the GQA group. */
        {
            float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                part[g] = 0.0f;
            if (ky < tk) {
                const size_t kb = (size_t)ky * kstride;
                for (int i = 0; i < chunk; i++) {
                    int e = lane + i * FC_ATTN_DOT_LANES;
                    float kv = __half2float(k_h[kb + e]);
#pragma unroll
                    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                        if (g < G) {
                            size_t qi = (size_t)g * head_dim + e;
                            float qv = Q_FP16 ? __half2float(q_h[qi]) : q_f[qi];
                            part[g] += qv * kv;
                        }
                }
            }
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    float p = part[g];
                    for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,
                                              FC_ATTN_DOT_LANES);
                    if (lane == 0 && ky < tk)
                        score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;
                }
            }
        }
        __syncthreads();

        if (tid < G) {
            const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
            float m_new = m_s[tid];
            for (int j = 0; j < tk; j++)
                m_new = fmaxf(m_new, sc[j]);
            float corr = expf(m_s[tid] - m_new);
            float sum = 0.0f;
            for (int j = 0; j < tk; j++) {
                float pj = expf(sc[j] - m_new);
                p_s[(size_t)j * G + tid] = pj;
                sum += pj;
            }
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = m_new;
            corr_s[tid] = corr;
        }
        __syncthreads();

        /* PV accumulate — unchanged from fc_attn_flash_prefill_q4: V streams
         * through __ldg and is never staged, so it is untouched by all this. */
        for (int j = 0; j < tk; j++) {
            int vp = j0 + j;
            float vv;
            if (vp < sink)
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);
            else if (win > 0 && vp > cur_pos - win && vp > win_live_lo)
                vv = __half2float(v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]);
            else
                vv = fc_kvq4_dequant_elem(v_head + (size_t)vp * row_bytes, tid);
            const float *pj_s = p_s + (size_t)j * G;
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    if (j == 0)
                        acc[g] *= corr_s[g];
                    acc[g] += pj_s[g] * vv;
                }
            }
        }
        __syncthreads();
    }

    float *out_base = out + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        if (g < G)
            out_base[(size_t)g * head_dim + tid] = acc[g] / l_s[g];
}

extern "C" {

/* FC_ATTN_KTILE_FP16=1: __half K tile, fp32 q group. 18 016 B at the shipped
 * shape -> 6 blocks/SMX. */
__global__ void fc_attn_flash_prefill_q4_kh(
        const float *q, const unsigned char *k_cache,
        const unsigned char *v_cache, const __half *k_sink,
        const __half *v_sink, const __half *k_win, const __half *v_win,
        float *out, int kv_groups, int head_dim, int max_seq, int pos0,
        int sink, int win, int attn_sink, int attn_win, float scale)
{
    fc_attn_flash_prefill_q4_narrow<false>(
        q, k_cache, v_cache, k_sink, v_sink, k_win, v_win, out, kv_groups,
        head_dim, max_seq, pos0, sink, win, attn_sink, attn_win, scale);
}

/* FC_ATTN_KTILE_FP16=2: __half K tile AND __half q group. 13 920 B ->
 * 7 blocks/SMX. Strictly lossier than level 1 — q is an activation. */
__global__ void fc_attn_flash_prefill_q4_kqh(
        const float *q, const unsigned char *k_cache,
        const unsigned char *v_cache, const __half *k_sink,
        const __half *v_sink, const __half *k_win, const __half *v_win,
        float *out, int kv_groups, int head_dim, int max_seq, int pos0,
        int sink, int win, int attn_sink, int attn_win, float scale)
{
    fc_attn_flash_prefill_q4_narrow<true>(
        q, k_cache, v_cache, k_sink, v_sink, k_win, v_win, out, kv_groups,
        head_dim, max_seq, pos0, sink, win, attn_sink, attn_win, scale);
}

/* ============================================================
 * Decode-mode flash attention (QWEN35_FLASH_DECODE)
 *
 * A single query (the new token at absolute position seq_len-1) attends the
 * full cached context [0, seq_len). Reusing fc_attn_flash_prefill with npos=1
 * launches only n_head_kv blocks (=2 for this model) — the K80's ~13 SMX sit
 * idle. So the KV sequence is SPLIT across blocks: grid (n_splits, n_head_kv),
 * each block runs the same tiled online-softmax over its key slice for the
 * whole GQA group and writes a PARTIAL {running max m, running sum l,
 * UNnormalized accumulator acc} to scratch. fc_attn_flash_decode_combine then
 * merges the n_splits partials per head with the standard max-rescale, which
 * is algebraically identical to a single-pass softmax (only ~1e-6 tile drift,
 * same class as prefill; gate: argmax-exact vs the host decode path).
 *
 * Partial scratch layout, indexed by GLOBAL head qh = kvh*G + g:
 *   m_out[qh*n_splits + split], l_out[qh*n_splits + split]
 *   acc_out[(qh*n_splits + split)*head_dim + dim]
 * ============================================================ */

/* Split plan from the context length: ~kps keys per split, capped at `cap`
 * splits, split_len rounded up to the tile width. MUST stay formula-identical
 * to the host plan in qwen35moe_forward_cuda.c gpu_attn_block_decode_flash —
 * the *_dev kernels below recompute the plan on-device (from a device-resident
 * position) so a captured CUDA graph replays correctly at every position, and
 * the recomputed plan must equal the host plan for the same seq_len or the
 * graphed and direct paths stop being bit-identical. */
__device__ __forceinline__ int fc_attn_dec_plan(int seq_len, int kps, int cap,
                                                int *split_len_out) {
    int n_splits = (seq_len + kps - 1) / kps;
    if (n_splits < 1)
        n_splits = 1;
    if (n_splits > cap)
        n_splits = cap;
    int split_len = (seq_len + n_splits - 1) / n_splits;
    split_len = (split_len + 15) & ~15;
    *split_len_out = split_len;
    return (seq_len + split_len - 1) / split_len;
}

/* Merge the n_splits per-head partials into the final attention output.
 * One block per head; thread tid owns output dim tid. Shared verbatim by the
 * host-arg and device-pos combine kernels. */
__device__ __forceinline__ void fc_attn_flash_decode_combine_body(
        const float *m_in, const float *l_in, const float *acc_in, float *out,
        int head_dim, int n_splits) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= head_dim) return;

    const float *m_h = m_in + (size_t)head * n_splits;
    const float *l_h = l_in + (size_t)head * n_splits;
    const float *acc_h = acc_in + (size_t)head * n_splits * head_dim;

    float M = -__int_as_float(0x7f800000);
    for (int s = 0; s < n_splits; s++)
        M = fmaxf(M, m_h[s]);
    float o = 0.0f, L = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        float w = expf(m_h[s] - M);     /* empty split: m=-inf -> w=0 */
        L += l_h[s] * w;
        o += acc_h[(size_t)s * head_dim + tid] * w;
    }
    out[(size_t)head * head_dim + tid] = o / L;
}

/* Merge the n_splits per-head partials into the final attention output.
 * Grid (n_head), block head_dim threads (thread tid owns output dim tid). */
__global__ void fc_attn_flash_decode_combine(const float *m_in,
                                             const float *l_in,
                                             const float *acc_in, float *out,
                                             int head_dim, int n_splits) {
    fc_attn_flash_decode_combine_body(m_in, l_in, acc_in, out, head_dim,
                                      n_splits);
}

/* Combine + sigmoid output gate in one launch (decode slots path): out =
 * merged * sigmoid(z), replacing fc_attn_flash_decode_combine + fc_sigmoid(z)
 * + fc_mul. The merge is fc_attn_flash_decode_combine_body's loop verbatim;
 * the gate is fc_sigmoid's expression and the product is pinned with
 * __fmul_rn, so every intermediate is rounded exactly where the three kernels
 * stored it. z is read, not written. BIT-EXACT. Grid (n_head), block head_dim. */
__global__ void fc_attn_flash_decode_combine_gated(const float *m_in,
                                                   const float *l_in,
                                                   const float *acc_in,
                                                   const float *z, float *out,
                                                   int head_dim, int n_splits) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= head_dim) return;

    const float *m_h = m_in + (size_t)head * n_splits;
    const float *l_h = l_in + (size_t)head * n_splits;
    const float *acc_h = acc_in + (size_t)head * n_splits * head_dim;

    float M = -__int_as_float(0x7f800000);
    for (int s = 0; s < n_splits; s++)
        M = fmaxf(M, m_h[s]);
    float o = 0.0f, L = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        float w = expf(m_h[s] - M);
        L += l_h[s] * w;
        o += acc_h[(size_t)s * head_dim + tid] * w;
    }
    const size_t idx = (size_t)head * head_dim + tid;
    const float merged = o / L;
    const float gate = 1.0f / (1.0f + expf(-z[idx]));
    out[idx] = __fmul_rn(merged, gate);
}

/* Device-position combine (CUDA-graph capture): seq_len = pos[0] + 1 is read
 * from device memory and the split plan recomputed with fc_attn_dec_plan, so
 * the launch itself is position-invariant. Bit-exact to
 * fc_attn_flash_decode_combine at the same position. */
__global__ void fc_attn_flash_decode_combine_dev(const float *m_in,
                                                 const float *l_in,
                                                 const float *acc_in,
                                                 float *out, int head_dim,
                                                 const int *pos, int kps,
                                                 int splits_cap) {
    int seq_len = pos[0] + 1;
    int split_len;
    int n_splits = fc_attn_dec_plan(seq_len, kps, splits_cap, &split_len);
    fc_attn_flash_decode_combine_body(m_in, l_in, acc_in, out, head_dim,
                                      n_splits);
}

/*
 * Decode split-attention over the fp32 GPU KV cache. Same online-softmax math
 * and tiling as fc_attn_flash_prefill, but the query is the single decode
 * position and the key loop is restricted to this block's split slice
 * [split*split_len, min(seq_len, (split+1)*split_len)). Writes the split's
 * partial {m, l, acc} for every head in the group (see the section header).
 *
 * q:        [n_head][head_dim] fp32 (post QK-norm + RoPE), single position.
 * Grid (n_splits, n_head_kv), block head_dim threads (multiple of 16),
 * dynamic shared identical to fc_attn_flash_prefill.
 */
__device__ __forceinline__ void fc_attn_flash_decode_body(
        const float *q, const float *k_cache, const float *v_cache,
        float *m_out, float *l_out, float *acc_out, int kv_groups,
        int head_dim, int max_seq, int seq_len, int n_splits, int split_len,
        int attn_sink, int attn_win, float scale) {
    const int split = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    const int G = kv_groups;
    const int cur_pos = seq_len - 1;      /* decode query's absolute position */
    int k_lo = split * split_len;
    int k_hi = k_lo + split_len;
    if (k_hi > seq_len) k_hi = seq_len;

    const int kstride = FC_ATTN_K_STRIDE(head_dim);
    extern __shared__ float sh[];
    float *q_s = sh;
    float *k_s = q_s + (size_t)G * head_dim;
    float *score_s = k_s + (size_t)FC_ATTN_TILE_KEYS * kstride;
    float *p_s = score_s + G * FC_ATTN_TILE_KEYS;
    float *m_s = p_s + G * FC_ATTN_TILE_KEYS;
    float *l_s = m_s + G;
    float *corr_s = l_s + G;

    const float *q_base = q + (size_t)kvh * G * head_dim;
    for (int g = 0; g < G; g++)
        q_s[(size_t)g * head_dim + tid] = q_base[(size_t)g * head_dim + tid];
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    /* Compile-time width, so ptxas keeps acc[] in REGISTERS. Every loop that
     * indexes acc[] is unrolled to FC_ATTN_MAX_GROUPS with an `if (g < G)`
     * guard; a loop bounded by the RUNTIME kv_groups cannot be unrolled, the
     * array lands in LOCAL memory, and the PV inner loop then pays an
     * LDL.128/STL.128 round trip per key. ptxas reports that as a 32-byte
     * "stack frame" and prints "0 bytes spill stores" right beside it --
     * a never-promoted local array is not counted as a spill. */
    float acc[FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        acc[g] = 0.0f;

    const float *k_head = k_cache + (size_t)kvh * max_seq * head_dim;
    const float *v_head = v_cache + (size_t)kvh * max_seq * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = head_dim / FC_ATTN_DOT_LANES;

    for (int j0 = k_lo; j0 < k_hi; j0 += FC_ATTN_TILE_KEYS) {
        int tk = k_hi - j0;
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;

        /* Sliding window: skip tiles outside sink U recent-window. */
        if (!fc_attn_tile_in_window(j0, tk, cur_pos, attn_sink, attn_win))
            continue;

        for (int key = 0; key < tk; key++)
            for (int e = tid; e < head_dim; e += blockDim.x)
                k_s[(size_t)key * kstride + e] =
                    k_head[(size_t)(j0 + key) * head_dim + e];
        __syncthreads();

        /* QK^T, i-outer / g-inner: each thread reads its chunk of the staged K
         * row ONCE per tile into a register and fans it across the whole GQA
         * group, instead of re-reading k_s once per group (G=8 -> 1 LDS per
         * element). BIT-EXACT: every part[g] still accumulates over e in the
         * same ascending order the g-outer form used, so this is pure operand
         * motion. part[] is indexed only by the unrolled compile-time g, for
         * the same reason acc[] is -- a runtime-bounded loop would push it to
         * LOCAL memory and cost an LDL/STL round trip per element. */
        {
            float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                part[g] = 0.0f;
            if (ky < tk) {
                const float *kv = k_s + (size_t)ky * kstride;
                for (int i = 0; i < chunk; i++) {
                    int e = lane + i * FC_ATTN_DOT_LANES;
                    float kve = kv[e];
#pragma unroll
                    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                        if (g < G)
                            part[g] += q_s[(size_t)g * head_dim + e] * kve;
                }
            }
            /* G is uniform across the block, so the guarded unroll keeps the
             * shuffle warp-uniform exactly as the runtime-bounded loop did. */
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    float p = part[g];
                    for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,
                                              FC_ATTN_DOT_LANES);
                    if (lane == 0 && ky < tk)
                        score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;
                }
            }
        }
        __syncthreads();

        if (tid < G) {
            const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
            float m_new = m_s[tid];
            for (int j = 0; j < tk; j++)
                m_new = fmaxf(m_new, sc[j]);
            float corr = expf(m_s[tid] - m_new);
            float sum = 0.0f;
            for (int j = 0; j < tk; j++) {
                float pj = expf(sc[j] - m_new);
                p_s[(size_t)j * G + tid] = pj;   /* [TILE][G]: PV reads a run */
                sum += pj;
            }
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = m_new;
            corr_s[tid] = corr;
        }
        __syncthreads();

        for (int j = 0; j < tk; j++) {
            float vv = __ldg(&v_head[(size_t)(j0 + j) * head_dim + tid]);
            const float *pj_s = p_s + (size_t)j * G;
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    if (j == 0)
                        acc[g] *= corr_s[g];
                    acc[g] += pj_s[g] * vv;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
        if (g < G) {
            size_t po = ((size_t)kvh * G + g) * n_splits + split;
            acc_out[po * head_dim + tid] = acc[g];
            if (tid == 0) {
                m_out[po] = m_s[g];
                l_out[po] = l_s[g];
            }
        }
    }
}

__global__ void fc_attn_flash_decode(const float *q, const float *k_cache,
                                     const float *v_cache, float *m_out,
                                     float *l_out, float *acc_out,
                                     int kv_groups, int head_dim, int max_seq,
                                     int seq_len, int n_splits, int split_len,
                                     int attn_sink, int attn_win, float scale) {
    fc_attn_flash_decode_body(q, k_cache, v_cache, m_out, l_out, acc_out,
                              kv_groups, head_dim, max_seq, seq_len, n_splits,
                              split_len, attn_sink, attn_win, scale);
}

/*
 * Device-position decode split-attention (CUDA-graph capture): the query's
 * position is read from device memory (pos[0], seq_len = pos[0] + 1) and the
 * split plan recomputed on-device, so every launch argument and the grid are
 * position-invariant — capture once, replay every token. Launched with a
 * FIXED grid (splits_cap, n_head_kv); blocks past the live plan exit before
 * touching scratch (the partial layout is strided by the LIVE n_splits, same
 * as the host-arg kernel, so combine sees an identical layout). Bit-exact to
 * fc_attn_flash_decode at the same position. Caller guarantees no ring wrap
 * (pos + 1 <= max_seq).
 */
__global__ void fc_attn_flash_decode_dev(const float *q, const float *k_cache,
                                         const float *v_cache, float *m_out,
                                         float *l_out, float *acc_out,
                                         int kv_groups, int head_dim,
                                         int max_seq, const int *pos, int kps,
                                         int splits_cap, int attn_sink,
                                         int attn_win, float scale) {
    int seq_len = pos[0] + 1;
    int split_len;
    int n_splits = fc_attn_dec_plan(seq_len, kps, splits_cap, &split_len);
    if ((int)blockIdx.x >= n_splits)
        return;
    fc_attn_flash_decode_body(q, k_cache, v_cache, m_out, l_out, acc_out,
                              kv_groups, head_dim, max_seq, seq_len, n_splits,
                              split_len, attn_sink, attn_win, scale);
}

/*
 * Decode split-attention over the Q4 GPU KV cache. Identical to
 * fc_attn_flash_decode but K/V are read from the packed Q4_0 cache (with the
 * fp16 sink / recent-window shadow) and dequantized in-kernel, matching
 * fc_attn_flash_prefill_q4. Writes the same partial layout; the combine
 * kernel is shared.
 */
__device__ __forceinline__ void fc_attn_flash_decode_q4_body(
        const float *q, const unsigned char *k_cache,
        const unsigned char *v_cache, const __half *k_sink,
        const __half *v_sink, const __half *k_win, const __half *v_win,
        float *m_out, float *l_out, float *acc_out, int kv_groups,
        int head_dim, int max_seq, int seq_len, int n_splits, int split_len,
        int sink, int win, int attn_sink, int attn_win, float scale) {
    const int split = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    const int G = kv_groups;
    const int cur_pos = seq_len - 1;
    const int row_bytes = fc_kvq4_row_bytes(head_dim);
    int k_lo = split * split_len;
    int k_hi = k_lo + split_len;
    if (k_hi > seq_len) k_hi = seq_len;

    const int kstride = FC_ATTN_K_STRIDE(head_dim);
    extern __shared__ float sh[];
    float *q_s = sh;
    float *k_s = q_s + (size_t)G * head_dim;
    float *score_s = k_s + (size_t)FC_ATTN_TILE_KEYS * kstride;
    float *p_s = score_s + G * FC_ATTN_TILE_KEYS;
    float *m_s = p_s + G * FC_ATTN_TILE_KEYS;
    float *l_s = m_s + G;
    float *corr_s = l_s + G;

    const float *q_base = q + (size_t)kvh * G * head_dim;
    for (int g = 0; g < G; g++)
        q_s[(size_t)g * head_dim + tid] = q_base[(size_t)g * head_dim + tid];
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    /* Compile-time width, so ptxas keeps acc[] in REGISTERS. Every loop that
     * indexes acc[] is unrolled to FC_ATTN_MAX_GROUPS with an `if (g < G)`
     * guard; a loop bounded by the RUNTIME kv_groups cannot be unrolled, the
     * array lands in LOCAL memory, and the PV inner loop then pays an
     * LDL.128/STL.128 round trip per key. ptxas reports that as a 32-byte
     * "stack frame" and prints "0 bytes spill stores" right beside it --
     * a never-promoted local array is not counted as a spill. */
    float acc[FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        acc[g] = 0.0f;

    const unsigned char *k_head = k_cache + (size_t)kvh * max_seq * row_bytes;
    const unsigned char *v_head = v_cache + (size_t)kvh * max_seq * row_bytes;
    const __half *k_win_head = k_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *v_win_head = v_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *k_sink_head = k_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const __half *v_sink_head = v_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = head_dim / FC_ATTN_DOT_LANES;

    for (int j0 = k_lo; j0 < k_hi; j0 += FC_ATTN_TILE_KEYS) {
        int tk = k_hi - j0;
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;

        /* Sliding window: skip tiles outside sink U recent-window. */
        if (!fc_attn_tile_in_window(j0, tk, cur_pos, attn_sink, attn_win))
            continue;

        for (int key = 0; key < tk; key++) {
            int kp = j0 + key;
            float *krow = k_s + (size_t)key * kstride;
            if (kp < sink) {
                const __half *s = k_sink_head + (size_t)kp * head_dim;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    krow[e] = __half2float(s[e]);
            } else if (win > 0 && kp > cur_pos - win) {
                const __half *s = k_win_head +
                    (size_t)(kp % FC_KVQ4_WMAX) * head_dim;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    krow[e] = __half2float(s[e]);
            } else {
                const unsigned char *s = k_head + (size_t)kp * row_bytes;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    krow[e] = fc_kvq4_dequant_elem(s, e);
            }
        }
        __syncthreads();

        /* QK^T, i-outer / g-inner: each thread reads its chunk of the staged K
         * row ONCE per tile into a register and fans it across the whole GQA
         * group, instead of re-reading k_s once per group (G=8 -> 1 LDS per
         * element). BIT-EXACT: every part[g] still accumulates over e in the
         * same ascending order the g-outer form used, so this is pure operand
         * motion. part[] is indexed only by the unrolled compile-time g, for
         * the same reason acc[] is -- a runtime-bounded loop would push it to
         * LOCAL memory and cost an LDL/STL round trip per element. */
        {
            float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                part[g] = 0.0f;
            if (ky < tk) {
                const float *kv = k_s + (size_t)ky * kstride;
                for (int i = 0; i < chunk; i++) {
                    int e = lane + i * FC_ATTN_DOT_LANES;
                    float kve = kv[e];
#pragma unroll
                    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                        if (g < G)
                            part[g] += q_s[(size_t)g * head_dim + e] * kve;
                }
            }
            /* G is uniform across the block, so the guarded unroll keeps the
             * shuffle warp-uniform exactly as the runtime-bounded loop did. */
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    float p = part[g];
                    for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,
                                              FC_ATTN_DOT_LANES);
                    if (lane == 0 && ky < tk)
                        score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;
                }
            }
        }
        __syncthreads();

        if (tid < G) {
            const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
            float m_new = m_s[tid];
            for (int j = 0; j < tk; j++)
                m_new = fmaxf(m_new, sc[j]);
            float corr = expf(m_s[tid] - m_new);
            float sum = 0.0f;
            for (int j = 0; j < tk; j++) {
                float pj = expf(sc[j] - m_new);
                p_s[(size_t)j * G + tid] = pj;   /* [TILE][G]: PV reads a run */
                sum += pj;
            }
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = m_new;
            corr_s[tid] = corr;
        }
        __syncthreads();

        for (int j = 0; j < tk; j++) {
            int vp = j0 + j;
            float vv;
            if (vp < sink)
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);
            else if (win > 0 && vp > cur_pos - win)
                vv = __half2float(v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]);
            else
                vv = fc_kvq4_dequant_elem(v_head + (size_t)vp * row_bytes, tid);
            const float *pj_s = p_s + (size_t)j * G;
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    if (j == 0)
                        acc[g] *= corr_s[g];
                    acc[g] += pj_s[g] * vv;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
        if (g < G) {
            size_t po = ((size_t)kvh * G + g) * n_splits + split;
            acc_out[po * head_dim + tid] = acc[g];
            if (tid == 0) {
                m_out[po] = m_s[g];
                l_out[po] = l_s[g];
            }
        }
    }
}

__global__ void fc_attn_flash_decode_q4(const float *q,
                                        const unsigned char *k_cache,
                                        const unsigned char *v_cache,
                                        const __half *k_sink,
                                        const __half *v_sink,
                                        const __half *k_win,
                                        const __half *v_win, float *m_out,
                                        float *l_out, float *acc_out,
                                        int kv_groups, int head_dim,
                                        int max_seq, int seq_len, int n_splits,
                                        int split_len, int sink, int win,
                                        int attn_sink, int attn_win,
                                        float scale) {
    fc_attn_flash_decode_q4_body(q, k_cache, v_cache, k_sink, v_sink, k_win,
                                 v_win, m_out, l_out, acc_out, kv_groups,
                                 head_dim, max_seq, seq_len, n_splits,
                                 split_len, sink, win, attn_sink, attn_win,
                                 scale);
}

/* Device-position Q4 decode split-attention (CUDA-graph capture). Same
 * contract as fc_attn_flash_decode_dev; bit-exact to fc_attn_flash_decode_q4
 * at the same position. */
__global__ void fc_attn_flash_decode_q4_dev(const float *q,
                                            const unsigned char *k_cache,
                                            const unsigned char *v_cache,
                                            const __half *k_sink,
                                            const __half *v_sink,
                                            const __half *k_win,
                                            const __half *v_win, float *m_out,
                                            float *l_out, float *acc_out,
                                            int kv_groups, int head_dim,
                                            int max_seq, const int *pos,
                                            int kps, int splits_cap, int sink,
                                            int win, int attn_sink,
                                            int attn_win, float scale) {
    int seq_len = pos[0] + 1;
    int split_len;
    int n_splits = fc_attn_dec_plan(seq_len, kps, splits_cap, &split_len);
    if ((int)blockIdx.x >= n_splits)
        return;
    fc_attn_flash_decode_q4_body(q, k_cache, v_cache, k_sink, v_sink, k_win,
                                 v_win, m_out, l_out, acc_out, kv_groups,
                                 head_dim, max_seq, seq_len, n_splits,
                                 split_len, sink, win, attn_sink, attn_win,
                                 scale);
}

/*
 * Device-position single-row KV append (CUDA-graph capture): the decode
 * token's ring slot is computed from device-resident pos[0] instead of a host
 * scalar, so the recorded launch replays correctly at every position. Grid
 * (1, n_head_kv) — decode appends exactly one row. Bit-exact to
 * fc_attn_kv_append with pos0 = pos[0], npos = 1.
 */
__global__ void fc_attn_kv_append_dev(const float *k_new, const float *v_new,
                                      float *k_cache, float *v_cache,
                                      int n_head_kv, int head_dim, int max_seq,
                                      const int *pos) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int dd = threadIdx.x;
    if (dd >= head_dim) return;

    int cache_pos = (pos[0] + t) % max_seq;
    size_t src = ((size_t)t * n_head_kv + h) * head_dim + dd;
    size_t dst = ((size_t)h * max_seq + cache_pos) * head_dim + dd;
    k_cache[dst] = k_new[src];
    v_cache[dst] = v_new[src];
}

/* ============================================================
 * Multi-user decode attention (QWEN35_FLASH_SLOTS)
 *
 * The batched-decode "slots" path: K independent users, each decoding ONE
 * token at its OWN position against its OWN KV ring. The slot kernels below
 * move the whole per-step attention on-die — replacing the host per-user
 * softmax loop and its 2 syncs + PCIe round-trips per layer (the K=8
 * steady-state ceiling, multi-tenant-speed-opt.md §3.2).
 *
 * KV layout: one ring PER SLOT, each slice byte-identical to the single-user
 * ring — [slot][n_head_kv][max_seq][row] (row = head_dim fp32 or Q4_0
 * bytes + per-slot fp16 sink/window shadows).
 *
 * Split plan: every user recomputes its OWN split plan from its position
 * (fc_attn_dec_plan(seq_len_u), the exact single-user formula), so each
 * user's per-split (m, l, acc) partials are bit-identical to the
 * single-token decode kernel at the same position. The launch is uniform
 * (grid.x = the max plan across users; partial stride = gridDim.x); blocks
 * past a user's own plan write EMPTY partials (m=-inf, l=0, acc=0), which
 * the combine's max-rescale weights to exactly zero — merging them adds
 * only +0.0f terms, so the merged output is bit-identical to combining the
 * live splits alone. Combine reuses fc_attn_flash_decode_combine with
 * gridDim.x = nslots * n_head (partials/out are (slot, head)-major).
 * ============================================================ */

/*
 * Append each slot's single roped K/V row to its own fp32 ring at its own
 * position (pos[slot], device int32 array — the RoPE position buffer).
 *
 * k_new, v_new:     [nslots][n_head_kv][head_dim] fp32
 * k_cache, v_cache: [nslots][n_head_kv][max_seq][head_dim] fp32 (written)
 *
 * Grid (nslots, n_head_kv), block head_dim threads. Row lands at
 * pos[slot] % max_seq (the host ring rule).
 */
__global__ void fc_attn_kv_append_slots(const float *k_new, const float *v_new,
                                        float *k_cache, float *v_cache,
                                        int n_head_kv, int head_dim,
                                        int max_seq, const int *pos,
                                        const int *rows) {
    int u = blockIdx.x;                   /* slot */
    int h = blockIdx.y;
    int dd = threadIdx.x;
    if (dd >= head_dim) return;

    int r = rows ? rows[u] : u;           /* the slot's ring row */
    int cache_pos = pos[u] % max_seq;
    size_t src = ((size_t)u * n_head_kv + h) * head_dim + dd;
    size_t dst = (((size_t)r * n_head_kv + h) * max_seq + cache_pos) *
                 head_dim + dd;
    k_cache[dst] = k_new[src];
    v_cache[dst] = v_new[src];
}

/* Per-slot Q4 ring append (quantizing in-kernel) + per-slot fp16 sink /
 * recent-window shadow mirror. Same contract as fc_attn_kv_append_q4 per
 * slot; caller guarantees no ring wrap (pos[u] + 1 <= max_seq).
 *
 * k_cache/v_cache: [nslots][n_head_kv][max_seq][row_bytes] Q4_0 bytes
 * k_sink/v_sink:   [nslots][n_head_kv][FC_KVQ4_SINKMAX][head_dim] fp16
 * k_win/v_win:     [nslots][n_head_kv][FC_KVQ4_WMAX][head_dim] fp16
 *
 * Grid (nslots, n_head_kv), block head_dim threads (multiple of 32). */
__global__ void fc_attn_kv_append_slots_q4(const float *k_new,
                                           const float *v_new,
                                           unsigned char *k_cache,
                                           unsigned char *v_cache,
                                           __half *k_sink, __half *v_sink,
                                           __half *k_win, __half *v_win,
                                           int n_head_kv, int head_dim,
                                           int max_seq, const int *pos,
                                           int sink, int win,
                                           const int *rows) {
    int u = blockIdx.x;                   /* slot */
    int h = blockIdx.y;
    int tid = threadIdx.x;
    if (tid >= head_dim) return;

    int r = rows ? rows[u] : u;           /* the slot's ring row */
    int cache_pos = pos[u];
    int row_bytes = fc_kvq4_row_bytes(head_dim);
    size_t src = ((size_t)u * n_head_kv + h) * head_dim + tid;
    size_t dst_row = (((size_t)r * n_head_kv + h) * max_seq + cache_pos) *
                     row_bytes;
    int blk = tid / FC_KVQ4_BLK;
    int lane = tid & (FC_KVQ4_BLK - 1);

    /* Shared staging for the shuffle-free block quantizer (Kepler-safe). */
    __shared__ float s_k[FC_KVQ4_MAX_HEAD_DIM];
    __shared__ float s_v[FC_KVQ4_MAX_HEAD_DIM];
    float kf = k_new[src];
    float vf = v_new[src];
    s_k[tid] = kf;
    s_v[tid] = vf;
    __syncthreads();
    fc_kvq4_quant_block(s_k + (size_t)blk * FC_KVQ4_BLK,
                        k_cache + dst_row + (size_t)blk * FC_KVQ4_BLK_BYTES,
                        lane);
    fc_kvq4_quant_block(s_v + (size_t)blk * FC_KVQ4_BLK,
                        v_cache + dst_row + (size_t)blk * FC_KVQ4_BLK_BYTES,
                        lane);

    /* Mirror the row into this slot's fp16 sink / recent-window shadow. */
    if (win > 0) {
        size_t wi = (((size_t)r * n_head_kv + h) * FC_KVQ4_WMAX +
                     (cache_pos % FC_KVQ4_WMAX)) * head_dim + tid;
        k_win[wi] = __float2half(kf);
        v_win[wi] = __float2half(vf);
    }
    if (cache_pos < sink) {
        size_t si = (((size_t)r * n_head_kv + h) * FC_KVQ4_SINKMAX +
                     cache_pos) * head_dim + tid;
        k_sink[si] = __float2half(kf);
        v_sink[si] = __float2half(vf);
    }
}

/*
 * Multi-user decode split-attention over the fp32 per-slot KV rings. Each
 * block runs one (split, kv-head, slot) slice of one user's decode attention
 * with the user's OWN split plan recomputed on-device from pos[slot] — the
 * per-split math is bit-identical to fc_attn_flash_decode at that position.
 *
 * q:   [nslots][n_head][head_dim] fp32 (post QK-norm + RoPE), one row/slot.
 * Partials indexed by GLOBAL head gh = slot*n_head + kvh*G + g, strided by
 * the LAUNCH width gridDim.x:
 *   m_out[gh*gridDim.x + split], l_out[...], acc_out[(...)*head_dim + dim]
 * Off-plan blocks (split >= this user's n_splits) write EMPTY partials.
 *
 * Grid (launch_splits, n_head_kv, nslots), block head_dim threads, dynamic
 * shared identical to fc_attn_flash_prefill. Caller guarantees no ring wrap
 * (pos[u] + 1 <= max_seq for every slot).
 */
__global__ void fc_attn_flash_decode_slots(const float *q,
                                           const float *k_cache,
                                           const float *v_cache, float *m_out,
                                           float *l_out, float *acc_out,
                                           int kv_groups, int head_dim,
                                           int max_seq, const int *pos,
                                           int kps, int splits_cap,
                                           int attn_sink, int attn_win,
                                           float scale, const int *rows) {
    const int split = blockIdx.x;
    const int kvh = blockIdx.y;
    const int slot = blockIdx.z;
    const int row = rows ? rows[slot] : slot;   /* the slot's ring row */
    const int tid = threadIdx.x;
    const int G = kv_groups;
    const int n_head_kv = gridDim.y;
    const int stride = gridDim.x;         /* uniform partial stride */
    const int seq_len = pos[slot] + 1;
    const int cur_pos = seq_len - 1;

    /* This user's own split plan (the single-user formula). */
    int split_len;
    int n_splits = fc_attn_dec_plan(seq_len, kps, splits_cap, &split_len);

    /* Partial base: global head (slot, kvh, g=0), strided by the launch. */
    const size_t head0 = ((size_t)slot * n_head_kv + kvh) * G;

    if (split >= n_splits) {
        /* Off-plan for this (shorter-context) user: EMPTY partial. The
         * combine weights m=-inf splits to exactly 0 (see section header). */
        for (int g = 0; g < G; g++) {
            size_t po = (head0 + g) * stride + split;
            acc_out[po * head_dim + tid] = 0.0f;
            if (tid == 0) {
                m_out[po] = -__int_as_float(0x7f800000);
                l_out[po] = 0.0f;
            }
        }
        return;
    }

    int k_lo = split * split_len;
    int k_hi = k_lo + split_len;
    if (k_hi > seq_len) k_hi = seq_len;

    const int kstride = FC_ATTN_K_STRIDE(head_dim);
    extern __shared__ float sh[];
    float *q_s = sh;
    float *k_s = q_s + (size_t)G * head_dim;
    float *score_s = k_s + (size_t)FC_ATTN_TILE_KEYS * kstride;
    float *p_s = score_s + G * FC_ATTN_TILE_KEYS;
    float *m_s = p_s + G * FC_ATTN_TILE_KEYS;
    float *l_s = m_s + G;
    float *corr_s = l_s + G;

    const float *q_base = q + head0 * head_dim;
    for (int g = 0; g < G; g++)
        q_s[(size_t)g * head_dim + tid] = q_base[(size_t)g * head_dim + tid];
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    /* Compile-time width, so ptxas keeps acc[] in REGISTERS. Every loop that
     * indexes acc[] is unrolled to FC_ATTN_MAX_GROUPS with an `if (g < G)`
     * guard; a loop bounded by the RUNTIME kv_groups cannot be unrolled, the
     * array lands in LOCAL memory, and the PV inner loop then pays an
     * LDL.128/STL.128 round trip per key. ptxas reports that as a 32-byte
     * "stack frame" and prints "0 bytes spill stores" right beside it --
     * a never-promoted local array is not counted as a spill. */
    float acc[FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        acc[g] = 0.0f;

    const float *k_head = k_cache +
        ((size_t)row * n_head_kv + kvh) * max_seq * head_dim;
    const float *v_head = v_cache +
        ((size_t)row * n_head_kv + kvh) * max_seq * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = head_dim / FC_ATTN_DOT_LANES;

    for (int j0 = k_lo; j0 < k_hi; j0 += FC_ATTN_TILE_KEYS) {
        int tk = k_hi - j0;
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;

        /* Sliding window: skip tiles outside sink U recent-window. */
        if (!fc_attn_tile_in_window(j0, tk, cur_pos, attn_sink, attn_win))
            continue;

        for (int key = 0; key < tk; key++)
            for (int e = tid; e < head_dim; e += blockDim.x)
                k_s[(size_t)key * kstride + e] =
                    k_head[(size_t)(j0 + key) * head_dim + e];
        __syncthreads();

        /* QK^T, i-outer / g-inner: each thread reads its chunk of the staged K
         * row ONCE per tile into a register and fans it across the whole GQA
         * group, instead of re-reading k_s once per group (G=8 -> 1 LDS per
         * element). BIT-EXACT: every part[g] still accumulates over e in the
         * same ascending order the g-outer form used, so this is pure operand
         * motion. part[] is indexed only by the unrolled compile-time g, for
         * the same reason acc[] is -- a runtime-bounded loop would push it to
         * LOCAL memory and cost an LDL/STL round trip per element. */
        {
            float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                part[g] = 0.0f;
            if (ky < tk) {
                const float *kv = k_s + (size_t)ky * kstride;
                for (int i = 0; i < chunk; i++) {
                    int e = lane + i * FC_ATTN_DOT_LANES;
                    float kve = kv[e];
#pragma unroll
                    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                        if (g < G)
                            part[g] += q_s[(size_t)g * head_dim + e] * kve;
                }
            }
            /* G is uniform across the block, so the guarded unroll keeps the
             * shuffle warp-uniform exactly as the runtime-bounded loop did. */
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    float p = part[g];
                    for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,
                                              FC_ATTN_DOT_LANES);
                    if (lane == 0 && ky < tk)
                        score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;
                }
            }
        }
        __syncthreads();

        if (tid < G) {
            const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
            float m_new = m_s[tid];
            for (int j = 0; j < tk; j++)
                m_new = fmaxf(m_new, sc[j]);
            float corr = expf(m_s[tid] - m_new);
            float sum = 0.0f;
            for (int j = 0; j < tk; j++) {
                float pj = expf(sc[j] - m_new);
                p_s[(size_t)j * G + tid] = pj;   /* [TILE][G]: PV reads a run */
                sum += pj;
            }
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = m_new;
            corr_s[tid] = corr;
        }
        __syncthreads();

        for (int j = 0; j < tk; j++) {
            float vv = __ldg(&v_head[(size_t)(j0 + j) * head_dim + tid]);
            const float *pj_s = p_s + (size_t)j * G;
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    if (j == 0)
                        acc[g] *= corr_s[g];
                    acc[g] += pj_s[g] * vv;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
        if (g < G) {
            size_t po = (head0 + g) * stride + split;
            acc_out[po * head_dim + tid] = acc[g];
            if (tid == 0) {
                m_out[po] = m_s[g];
                l_out[po] = l_s[g];
            }
        }
    }
}

/*
 * Multi-user decode split-attention over the per-slot Q4 KV rings (with the
 * per-slot fp16 sink / recent-window shadows). Identical plan/partial scheme
 * to fc_attn_flash_decode_slots; K/V dequantized in-kernel like
 * fc_attn_flash_decode_q4.
 */
__global__ void fc_attn_flash_decode_slots_q4(const float *q,
                                              const unsigned char *k_cache,
                                              const unsigned char *v_cache,
                                              const __half *k_sink,
                                              const __half *v_sink,
                                              const __half *k_win,
                                              const __half *v_win,
                                              float *m_out, float *l_out,
                                              float *acc_out, int kv_groups,
                                              int head_dim, int max_seq,
                                              const int *pos, int kps,
                                              int splits_cap, int sink,
                                              int win, int attn_sink,
                                              int attn_win, float scale,
                                              const int *rows) {
    const int split = blockIdx.x;
    const int kvh = blockIdx.y;
    const int slot = blockIdx.z;
    const int row = rows ? rows[slot] : slot;   /* the slot's ring row */
    const int tid = threadIdx.x;
    const int G = kv_groups;
    const int n_head_kv = gridDim.y;
    const int stride = gridDim.x;
    const int seq_len = pos[slot] + 1;
    const int cur_pos = seq_len - 1;
    const int row_bytes = fc_kvq4_row_bytes(head_dim);

    int split_len;
    int n_splits = fc_attn_dec_plan(seq_len, kps, splits_cap, &split_len);

    const size_t head0 = ((size_t)slot * n_head_kv + kvh) * G;

    if (split >= n_splits) {
        for (int g = 0; g < G; g++) {
            size_t po = (head0 + g) * stride + split;
            acc_out[po * head_dim + tid] = 0.0f;
            if (tid == 0) {
                m_out[po] = -__int_as_float(0x7f800000);
                l_out[po] = 0.0f;
            }
        }
        return;
    }

    int k_lo = split * split_len;
    int k_hi = k_lo + split_len;
    if (k_hi > seq_len) k_hi = seq_len;

    const int kstride = FC_ATTN_K_STRIDE(head_dim);
    extern __shared__ float sh[];
    float *q_s = sh;
    float *k_s = q_s + (size_t)G * head_dim;
    float *score_s = k_s + (size_t)FC_ATTN_TILE_KEYS * kstride;
    float *p_s = score_s + G * FC_ATTN_TILE_KEYS;
    float *m_s = p_s + G * FC_ATTN_TILE_KEYS;
    float *l_s = m_s + G;
    float *corr_s = l_s + G;

    const float *q_base = q + head0 * head_dim;
    for (int g = 0; g < G; g++)
        q_s[(size_t)g * head_dim + tid] = q_base[(size_t)g * head_dim + tid];
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    /* Compile-time width, so ptxas keeps acc[] in REGISTERS. Every loop that
     * indexes acc[] is unrolled to FC_ATTN_MAX_GROUPS with an `if (g < G)`
     * guard; a loop bounded by the RUNTIME kv_groups cannot be unrolled, the
     * array lands in LOCAL memory, and the PV inner loop then pays an
     * LDL.128/STL.128 round trip per key. ptxas reports that as a 32-byte
     * "stack frame" and prints "0 bytes spill stores" right beside it --
     * a never-promoted local array is not counted as a spill. */
    float acc[FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        acc[g] = 0.0f;

    const unsigned char *k_head = k_cache +
        ((size_t)row * n_head_kv + kvh) * max_seq * row_bytes;
    const unsigned char *v_head = v_cache +
        ((size_t)row * n_head_kv + kvh) * max_seq * row_bytes;
    const __half *k_win_head = k_win +
        ((size_t)row * n_head_kv + kvh) * FC_KVQ4_WMAX * head_dim;
    const __half *v_win_head = v_win +
        ((size_t)row * n_head_kv + kvh) * FC_KVQ4_WMAX * head_dim;
    const __half *k_sink_head = k_sink +
        ((size_t)row * n_head_kv + kvh) * FC_KVQ4_SINKMAX * head_dim;
    const __half *v_sink_head = v_sink +
        ((size_t)row * n_head_kv + kvh) * FC_KVQ4_SINKMAX * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = head_dim / FC_ATTN_DOT_LANES;

    for (int j0 = k_lo; j0 < k_hi; j0 += FC_ATTN_TILE_KEYS) {
        int tk = k_hi - j0;
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;

        /* Sliding window: skip tiles outside sink U recent-window. */
        if (!fc_attn_tile_in_window(j0, tk, cur_pos, attn_sink, attn_win))
            continue;

        for (int key = 0; key < tk; key++) {
            int kp = j0 + key;
            float *krow = k_s + (size_t)key * kstride;
            if (kp < sink) {
                const __half *s = k_sink_head + (size_t)kp * head_dim;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    krow[e] = __half2float(s[e]);
            } else if (win > 0 && kp > cur_pos - win) {
                const __half *s = k_win_head +
                    (size_t)(kp % FC_KVQ4_WMAX) * head_dim;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    krow[e] = __half2float(s[e]);
            } else {
                const unsigned char *s = k_head + (size_t)kp * row_bytes;
                for (int e = tid; e < head_dim; e += blockDim.x)
                    krow[e] = fc_kvq4_dequant_elem(s, e);
            }
        }
        __syncthreads();

        /* QK^T, i-outer / g-inner: each thread reads its chunk of the staged K
         * row ONCE per tile into a register and fans it across the whole GQA
         * group, instead of re-reading k_s once per group (G=8 -> 1 LDS per
         * element). BIT-EXACT: every part[g] still accumulates over e in the
         * same ascending order the g-outer form used, so this is pure operand
         * motion. part[] is indexed only by the unrolled compile-time g, for
         * the same reason acc[] is -- a runtime-bounded loop would push it to
         * LOCAL memory and cost an LDL/STL round trip per element. */
        {
            float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                part[g] = 0.0f;
            if (ky < tk) {
                const float *kv = k_s + (size_t)ky * kstride;
                for (int i = 0; i < chunk; i++) {
                    int e = lane + i * FC_ATTN_DOT_LANES;
                    float kve = kv[e];
#pragma unroll
                    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                        if (g < G)
                            part[g] += q_s[(size_t)g * head_dim + e] * kve;
                }
            }
            /* G is uniform across the block, so the guarded unroll keeps the
             * shuffle warp-uniform exactly as the runtime-bounded loop did. */
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    float p = part[g];
                    for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,
                                              FC_ATTN_DOT_LANES);
                    if (lane == 0 && ky < tk)
                        score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;
                }
            }
        }
        __syncthreads();

        if (tid < G) {
            const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
            float m_new = m_s[tid];
            for (int j = 0; j < tk; j++)
                m_new = fmaxf(m_new, sc[j]);
            float corr = expf(m_s[tid] - m_new);
            float sum = 0.0f;
            for (int j = 0; j < tk; j++) {
                float pj = expf(sc[j] - m_new);
                p_s[(size_t)j * G + tid] = pj;   /* [TILE][G]: PV reads a run */
                sum += pj;
            }
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = m_new;
            corr_s[tid] = corr;
        }
        __syncthreads();

        for (int j = 0; j < tk; j++) {
            int vp = j0 + j;
            float vv;
            if (vp < sink)
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);
            else if (win > 0 && vp > cur_pos - win)
                vv = __half2float(v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]);
            else
                vv = fc_kvq4_dequant_elem(v_head + (size_t)vp * row_bytes, tid);
            const float *pj_s = p_s + (size_t)j * G;
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    if (j == 0)
                        acc[g] *= corr_s[g];
                    acc[g] += pj_s[g] * vv;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
        if (g < G) {
            size_t po = (head0 + g) * stride + split;
            acc_out[po * head_dim + tid] = acc[g];
            if (tid == 0) {
                m_out[po] = m_s[g];
                l_out[po] = l_s[g];
            }
        }
    }
}

/*
 * Register-K + shared-V form of fc_attn_flash_decode_slots_q4 — BIT-EXACT.
 *
 * The prefill kernel family went through two bit-exact restructures that the
 * decode split kernel never got (fc_attn_flash_prefill_q4_kr, then _kvs):
 *   - K for the key this thread scores is dequantized straight into registers
 *     (fc_kvq4_dequant_lane_chunk) instead of being staged as an fp32 tile in
 *     shared by a 16-iteration all-threads loop and re-read;
 *   - the tile's V rows are copied into shared AS RAW Q4 BYTES in three word
 *     rounds (all loads in flight first) during the softmax shadow, so the PV
 *     loop reads shared instead of paying 16 serial global round trips.
 * Everything that sets the fp result is unchanged: the same dequant formula,
 * the same i-outer q.k accumulation order and shuffle tree, the same online
 * softmax by thread tid < G, and the PV rescale fused at j == 0. Same split
 * plan, same partial layout, so the combine kernel is shared.
 *
 * At decode depth this kernel is the step's growth term: +1.33 ms/step per 1k
 * keys measured on the shipped kernel, ~4 GB/s of KV read against a ~150 GB/s
 * die, i.e. latency-bound, which is exactly what both restructures removed on
 * the prefill side.
 *
 * CONTRACT (host-enforced in fc_attn_flash_decode_slots_q4_ctx): head_dim ==
 * FC_ATTN_TILE_KEYS * FC_ATTN_DOT_LANES (= 256: a block of head_dim threads
 * scores a full key tile and stages V in exactly FC_ATTN_VS_ROUNDS rounds),
 * row_bytes % 4 == 0. Dynamic shared = fc_attn_flash_kvs_shared_bytes().
 */
__global__ void fc_attn_flash_decode_slots_q4_kvs(const float *q,
                                                  const unsigned char *k_cache,
                                                  const unsigned char *v_cache,
                                                  const __half *k_sink,
                                                  const __half *v_sink,
                                                  const __half *k_win,
                                                  const __half *v_win,
                                                  float *m_out, float *l_out,
                                                  float *acc_out, int kv_groups,
                                                  int head_dim, int max_seq,
                                                  const int *pos, int kps,
                                                  int splits_cap, int sink,
                                                  int win, int attn_sink,
                                                  int attn_win, float scale,
                                                  const int *rows) {
    const int split = blockIdx.x;
    const int kvh = blockIdx.y;
    const int slot = blockIdx.z;
    const int row = rows ? rows[slot] : slot;   /* the slot's ring row */
    const int tid = threadIdx.x;
    const int G = kv_groups;
    const int n_head_kv = gridDim.y;
    const int stride = gridDim.x;
    const int seq_len = pos[slot] + 1;
    const int cur_pos = seq_len - 1;
    const int row_bytes = fc_kvq4_row_bytes(head_dim);

    int split_len;
    int n_splits = fc_attn_dec_plan(seq_len, kps, splits_cap, &split_len);

    const size_t head0 = ((size_t)slot * n_head_kv + kvh) * G;

    if (split >= n_splits) {
        for (int g = 0; g < G; g++) {
            size_t po = (head0 + g) * stride + split;
            acc_out[po * head_dim + tid] = 0.0f;
            if (tid == 0) {
                m_out[po] = -__int_as_float(0x7f800000);
                l_out[po] = 0.0f;
            }
        }
        return;
    }

    int k_lo = split * split_len;
    int k_hi = k_lo + split_len;
    if (k_hi > seq_len) k_hi = seq_len;

    /* MUST match fc_attn_flash_kvs_shared_bytes() in cuda/fc_cuda.c. */
    extern __shared__ float shv[];
    float *q_s = shv;
    float *score_s = q_s + (size_t)G * head_dim;
    float *p_s = score_s + G * FC_ATTN_TILE_KEYS;
    float *m_s = p_s + G * FC_ATTN_TILE_KEYS;
    float *l_s = m_s + G;
    float *corr_s = l_s + G;
    unsigned int *v_sh = (unsigned int *)(corr_s + G);

    const float *q_base = q + head0 * head_dim;
    for (int g = 0; g < G; g++)
        q_s[(size_t)g * head_dim + tid] = q_base[(size_t)g * head_dim + tid];
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    float acc[FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
        acc[g] = 0.0f;

    const size_t hoff = (size_t)row * n_head_kv + kvh;
    const unsigned char *k_head = k_cache + hoff * max_seq * row_bytes;
    const unsigned char *v_head = v_cache + hoff * max_seq * row_bytes;
    const __half *k_win_head = k_win + hoff * FC_KVQ4_WMAX * head_dim;
    const __half *v_win_head = v_win + hoff * FC_KVQ4_WMAX * head_dim;
    const __half *k_sink_head = k_sink + hoff * FC_KVQ4_SINKMAX * head_dim;
    const __half *v_sink_head = v_sink + hoff * FC_KVQ4_SINKMAX * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = head_dim / FC_ATTN_DOT_LANES;

    const int vlocal = tid & (FC_KVQ4_BLK - 1);
    const int vhalf = FC_KVQ4_BLK / 2;
    const int vbyte = 2 + (vlocal < vhalf ? vlocal : vlocal - vhalf);
    const int vshift = vlocal < vhalf ? 0 : 4;
    const int vqoff = (tid / FC_KVQ4_BLK) * FC_KVQ4_BLK_BYTES;

    for (int j0 = k_lo; j0 < k_hi; j0 += FC_ATTN_TILE_KEYS) {
        int tk = k_hi - j0;
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;

        if (!fc_attn_tile_in_window(j0, tk, cur_pos, attn_sink, attn_win))
            continue;

        float kreg[FC_ATTN_KREG_MAX_CHUNK];
        if (ky < tk) {
            const int kp = j0 + ky;
            if (kp < sink) {
                const __half *s = k_sink_head + (size_t)kp * head_dim;
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++)
                    if (i < chunk)
                        kreg[i] = __half2float(s[lane + i * FC_ATTN_DOT_LANES]);
            } else if (win > 0 && kp > cur_pos - win) {
                const __half *s = k_win_head +
                    (size_t)(kp % FC_KVQ4_WMAX) * head_dim;
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++)
                    if (i < chunk)
                        kreg[i] = __half2float(s[lane + i * FC_ATTN_DOT_LANES]);
            } else {
                fc_kvq4_dequant_lane_chunk(k_head + (size_t)kp * row_bytes,
                                           lane, chunk, kreg);
            }
        }

        {
            float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                part[g] = 0.0f;
            if (ky < tk) {
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++) {
                    if (i < chunk) {
                        const int e = lane + i * FC_ATTN_DOT_LANES;
                        const float kve = kreg[i];
#pragma unroll
                        for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                            if (g < G)
                                part[g] += q_s[(size_t)g * head_dim + e] * kve;
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    float p = part[g];
                    for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)
                        p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,
                                              FC_ATTN_DOT_LANES);
                    if (lane == 0 && ky < tk)
                        score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;
                }
            }
        }
        __syncthreads();

        if (tid < G) {
            const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
            float m_new = m_s[tid];
            for (int j = 0; j < tk; j++)
                m_new = fmaxf(m_new, sc[j]);
            float corr = expf(m_s[tid] - m_new);
            float sum = 0.0f;
            for (int j = 0; j < tk; j++) {
                float pj = expf(sc[j] - m_new);
                p_s[(size_t)j * G + tid] = pj;
                sum += pj;
            }
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = m_new;
            corr_s[tid] = corr;
        }

        /* V tile as raw q4 words, in the softmax shadow; written out so all
         * three loads are in flight before the first store. */
        {
            const unsigned int *vsrc =
                (const unsigned int *)(v_head + (size_t)j0 * row_bytes);
            const int vwords = (tk * row_bytes) >> 2;
            const int w0_idx = tid;
            const int w1_idx = tid + FC_ATTN_VS_THREADS;
            const int w2_idx = tid + 2 * FC_ATTN_VS_THREADS;
            const unsigned int w0 = w0_idx < vwords ? vsrc[w0_idx] : 0u;
            const unsigned int w1 = w1_idx < vwords ? vsrc[w1_idx] : 0u;
            const unsigned int w2 = w2_idx < vwords ? vsrc[w2_idx] : 0u;
            if (w0_idx < vwords)
                v_sh[w0_idx] = w0;
            if (w1_idx < vwords)
                v_sh[w1_idx] = w1;
            if (w2_idx < vwords)
                v_sh[w2_idx] = w2;
        }
        __syncthreads();

        for (int j = 0; j < tk; j++) {
            int vp = j0 + j;
            float vv;
            if (vp < sink) {
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);
            } else if (win > 0 && vp > cur_pos - win) {
                vv = __half2float(v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]);
            } else {
                /* Inline: through a generic-pointer helper ptxas loses the
                 * shared address space and emits generic loads. */
                const unsigned char *bp =
                    (const unsigned char *)v_sh + j * row_bytes + vqoff;
                const unsigned short d_bits = (unsigned short)(bp[0] | (bp[1] << 8));
                const float d = __half2float(__ushort_as_half(d_bits));
                const int code = (bp[vbyte] >> vshift) & 0xF;
                vv = (float)(code - 8) * d;
            }
            const float *pj_s = p_s + (size_t)j * G;
#pragma unroll
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                if (g < G) {
                    if (j == 0)
                        acc[g] *= corr_s[g];
                    acc[g] += pj_s[g] * vv;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
        if (g < G) {
            size_t po = (head0 + g) * stride + split;
            acc_out[po * head_dim + tid] = acc[g];
            if (tid == 0) {
                m_out[po] = m_s[g];
                l_out[po] = l_s[g];
            }
        }
    }
}

/* Device-position single-row Q4 KV append (CUDA-graph capture). Bit-exact to
 * fc_attn_kv_append_q4 with pos0 = pos[0], npos = 1 (caller guarantees no
 * ring wrap, matching the host-arg kernel's contract). */
__global__ void fc_attn_kv_append_q4_dev(const float *k_new,
                                         const float *v_new,
                                         unsigned char *k_cache,
                                         unsigned char *v_cache,
                                         __half *k_sink, __half *v_sink,
                                         __half *k_win, __half *v_win,
                                         int n_head_kv, int head_dim,
                                         int max_seq, const int *pos, int sink,
                                         int win) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    if (tid >= head_dim) return;

    int cache_pos = pos[0] + t;
    int row_bytes = fc_kvq4_row_bytes(head_dim);
    size_t src = ((size_t)t * n_head_kv + h) * head_dim + tid;
    size_t dst_row = ((size_t)h * max_seq + cache_pos) * row_bytes;
    int blk = tid / FC_KVQ4_BLK;
    int lane = tid & (FC_KVQ4_BLK - 1);

    /* Shared staging for the shuffle-free block quantizer (Kepler-safe). */
    __shared__ float s_k[FC_KVQ4_MAX_HEAD_DIM];
    __shared__ float s_v[FC_KVQ4_MAX_HEAD_DIM];
    float kf = k_new[src];
    float vf = v_new[src];
    s_k[tid] = kf;
    s_v[tid] = vf;
    __syncthreads();
    fc_kvq4_quant_block(s_k + (size_t)blk * FC_KVQ4_BLK,
                        k_cache + dst_row + (size_t)blk * FC_KVQ4_BLK_BYTES,
                        lane);
    fc_kvq4_quant_block(s_v + (size_t)blk * FC_KVQ4_BLK,
                        v_cache + dst_row + (size_t)blk * FC_KVQ4_BLK_BYTES,
                        lane);

    /* Mirror the row into the fp16 sink / recent-window shadow (element tid). */
    if (win > 0) {
        size_t wi = ((size_t)h * FC_KVQ4_WMAX + (cache_pos % FC_KVQ4_WMAX)) * head_dim + tid;
        k_win[wi] = __float2half(kf);
        v_win[wi] = __float2half(vf);
    }
    if (cache_pos < sink) {
        size_t si = ((size_t)h * FC_KVQ4_SINKMAX + cache_pos) * head_dim + tid;
        k_sink[si] = __float2half(kf);
        v_sink[si] = __float2half(vf);
    }
}

} /* extern "C" */

/* R-row speculative-verify decode kernel (fc_attn_flash_decode_rows_q4_kvs). */
#include "attention_decode_rows.cuh"
#include "attention_decode_fast.cuh"
