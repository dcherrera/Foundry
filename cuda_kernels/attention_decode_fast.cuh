/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   attention_decode_fast.cuh
 * @brief  Fewer-instruction form of the decode slots split flash attention
 *         (fc_attn_flash_decode_slots_q4_kvf) — BIT-EXACT against
 *         fc_attn_flash_decode_slots_q4_kvs.
 *
 * Included at the end of attention.cu (same PTX module, same helpers).
 *
 * At long context the decode attention is most of a multi-user step (Qwen3.6
 * Q8 on 4 dies, 4 users at 64k: ~0.68 ms per 1k keys per user per step, ~60%
 * of the step). nvprof on the K80 says the kvs kernel is INSTRUCTION-bound,
 * not memory-bound: IPC 2.6 with 55% of issue slots used, 8 GB/s of KV, ~880
 * warp instructions per (key, kv head) of which only ~1 in 4 is an FMA. So
 * this kernel keeps every arithmetic operation of kvs, in the same order, and
 * removes the bookkeeping around them:
 *   - q is staged INTERLEAVED ([dim][group], kv_groups == 8), so the q.k dot
 *     reads the 8 groups' q for one dim with two 128-bit shared loads instead
 *     of eight 32-bit ones;
 *   - the PV loop reads the 8 groups' probabilities of a key with two 128-bit
 *     loads, and a tile with no fp16-shadow key (no sink, out of the recent
 *     window: almost every tile at depth) runs a branch-free unrolled loop;
 *   - every q4 block scale is one 16-bit load (they sit at even offsets), in
 *     the K dequant (global) and the V dequant (shared).
 * The values each FMA consumes, and the FMA order per accumulator, are the kvs
 * kernel's; so are the split plan, the tiles, the online softmax, the partial
 * layout and the combine. CONTRACT (host-enforced): everything the kvs kernel
 * requires, plus kv_groups == FC_ATTN_KVF_GROUPS.
 *
 * @status Default OFF (FC_ATTN_DEC_KVF=1), gated by cuda/tests/test_attn_decode_kvf.
 * @issues None
 * @todo   None
 */

/* Falsification hook for the exactness gate, default OFF: build with
 * NVCCFLAGS_EXTRA=-DFC_ATTN_KVF_MUTATE=1 and test_attn_decode_kvf must FAIL. */
#ifndef FC_ATTN_KVF_MUTATE
#define FC_ATTN_KVF_MUTATE 0
#endif

#define FC_ATTN_KVF_GROUPS 8

/* Minimum resident blocks per SMX asked of ptxas (0 = no bound). Unbounded,
 * kvf takes 128 registers (4 blocks of 256) and gains only 1.14x at one user.
 * Measured on the K80 (bench_attn_decode_kvf, 8k-128k, 1/2/4 users): 5 blocks
 * (96 registers, no spill) 1.50-2.04x, 6 blocks (80) 1.53-1.85x, none
 * 1.14-1.85x. Bit-exact at every setting (the bound changes registers, not
 * arithmetic). Override with NVCCFLAGS_EXTRA=-DFC_ATTN_KVF_MINBLOCKS=<n>. */
#ifndef FC_ATTN_KVF_MINBLOCKS
#define FC_ATTN_KVF_MINBLOCKS 5
#endif
#if FC_ATTN_KVF_MINBLOCKS > 0
#define FC_ATTN_KVF_BOUNDS __launch_bounds__(256, FC_ATTN_KVF_MINBLOCKS)
#else
#define FC_ATTN_KVF_BOUNDS
#endif

/* q4 code c (0..15) to (c - 8) * d without a conversion: c is inserted under exponent 0x4B
 * (the float 2^23 + c) and 2^23 + 8 is subtracted, exactly; the value and the +0 at c = 8
 * are those of (float)(c - 8) * d for every code, f16 scale and shift: 3 instructions for a
 * low nibble, 4 for a high one or V's shifted nibble, against 4-5 with I2F.
 * FC_ATTN_KVF_RED is a gate-only red build (the codes offset by 7). */
#ifdef FC_ATTN_KVF_RED
#define FC_KVF_MINUS_BIAS (-8388615.0f)
#else
#define FC_KVF_MINUS_BIAS (-8388616.0f)   /* -(2^23 + 8) */
#endif

/* base with its low 4 bits replaced by those of src. */
__device__ __forceinline__ unsigned fc_kvf_bfi4(unsigned src, unsigned base) {
    unsigned r;
    asm("bfi.b32 %0, %1, %2, 0, 4;" : "=r"(r) : "r"(src), "r"(base));
    return r;
}

__device__ __forceinline__ float fc_kvf_q4_scaled(unsigned magic, float d) {
    return __fmul_rn(__fadd_rn(__uint_as_float(magic), FC_KVF_MINUS_BIAS), d);
}

__device__ __forceinline__ float fc_kvf_dec_k_lo(unsigned codes, float d) {
    return fc_kvf_q4_scaled(fc_kvf_bfi4(codes, 0x4b000000u), d);
}

__device__ __forceinline__ float fc_kvf_dec_k_hi(unsigned codes, float d) {
    unsigned hi;
    asm("bfe.u32 %0, %1, 4, 4;" : "=r"(hi) : "r"(codes));
    return fc_kvf_q4_scaled(hi | 0x4b000000u, d);
}

/* V: the nibble at the thread's runtime shift of its byte. */
__device__ __forceinline__ float fc_kvf_dec_v(unsigned byte, int shift, float d) {
    return fc_kvf_q4_scaled(fc_kvf_bfi4((unsigned)((int)byte >> shift), 0x4b000000u), d);
}

/* One q4 block's scale: the fp16 at the block's (even) byte offset. */
__device__ __forceinline__ float fc_attn_kvf_scale_g(const unsigned char *bp) {
    return __half2float(__ushort_as_half(__ldg((const unsigned short *)bp)));
}

/* This lane's 16 K elements of one q4 row (fc_kvq4_dequant_lane_chunk's exact
 * expressions, head_dim 256: chunk 16 = 8 blocks x {low, high} nibble). */
__device__ __forceinline__ void fc_attn_kvf_dequant_k(const unsigned char *row,
                                                      int lane, float *kreg) {
#pragma unroll
    for (int b = 0; b < 8; b++) {
        const unsigned char *bp = row + b * FC_KVQ4_BLK_BYTES;
        const float d = fc_attn_kvf_scale_g(bp);
        const unsigned int codes = __ldg(bp + 2 + lane);
        kreg[2 * b] = fc_kvf_dec_k_lo(codes, d);
        kreg[2 * b + 1] = fc_kvf_dec_k_hi(codes, d);
    }
}

/* acc[g] (+)= p[g] * vv for the 8 groups, with the rounding nvcc gives the kvs
 * source (read from its PTX, so it is spelled out with intrinsics here rather
 * than left to contraction): at a tile's first key
 *   acc = fma(acc, corr, round(p * vv))
 * and at every other key acc = fma(p, vv, acc). p is the key's [G] row. */
__device__ __forceinline__ void fc_attn_kvf_pv(float *acc, const float *pj,
                                               float vv, int first,
                                               const float *corr) {
    const float4 pa = *(const float4 *)pj;
    const float4 pb = *(const float4 *)(pj + 4);
    const float p[FC_ATTN_KVF_GROUPS] = { pa.x, pa.y, pa.z, pa.w,
                                          pb.x, pb.y, pb.z, pb.w };
#pragma unroll
    for (int g = 0; g < FC_ATTN_KVF_GROUPS; g++)
        acc[g] = first ? __fmaf_rn(acc[g], corr[g], __fmul_rn(vv, p[g]))
                       : __fmaf_rn(p[g], vv, acc[g]);
}

extern "C" __global__ void FC_ATTN_KVF_BOUNDS fc_attn_flash_decode_slots_q4_kvf(
        const float *q, const unsigned char *k_cache, const unsigned char *v_cache,
        const __half *k_sink, const __half *v_sink, const __half *k_win,
        const __half *v_win, float *m_out, float *l_out, float *acc_out,
        int kv_groups, int head_dim, int max_seq, const int *pos, int kps,
        int splits_cap, int sink, int win, int attn_sink, int attn_win,
        float scale, const int *rows) {
    const int G = FC_ATTN_KVF_GROUPS;
    const int split = blockIdx.x;
    const int kvh = blockIdx.y;
    const int slot = blockIdx.z;
    const int row = rows ? rows[slot] : slot;
    const int tid = threadIdx.x;
    const int n_head_kv = gridDim.y;
    const int stride = gridDim.x;
    const int seq_len = pos[slot] + 1;
    const int cur_pos = seq_len - 1;
    const int row_bytes = fc_kvq4_row_bytes(head_dim);
    (void)kv_groups;

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
    const int k_lo = split * split_len;
    int k_hi = k_lo + split_len;
    if (k_hi > seq_len) k_hi = seq_len;

    /* MUST match fc_attn_flash_kvf_shared_bytes() in cuda/fc_cuda.c. Same as
     * kvs, except q is [head_dim][G] (interleaved). */
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
        q_s[(size_t)tid * G + g] = q_base[(size_t)g * head_dim + tid];
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    float acc[FC_ATTN_KVF_GROUPS];
#pragma unroll
    for (int g = 0; g < G; g++)
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
                    kreg[i] = __half2float(s[lane + i * FC_ATTN_DOT_LANES]);
            } else if (win > 0 && kp > cur_pos - win) {
                const __half *s = k_win_head + (size_t)(kp % FC_KVQ4_WMAX) * head_dim;
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++)
                    kreg[i] = __half2float(s[lane + i * FC_ATTN_DOT_LANES]);
            } else {
                fc_attn_kvf_dequant_k(k_head + (size_t)kp * row_bytes, lane, kreg);
            }
        }

        {
            float part[FC_ATTN_KVF_GROUPS];
#pragma unroll
            for (int g = 0; g < G; g++)
                part[g] = 0.0f;
            if (ky < tk) {
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++) {
                    const int e = lane + i * FC_ATTN_DOT_LANES;
                    const float kve = kreg[i];
                    const float4 qa = *(const float4 *)(q_s + (size_t)e * G);
                    const float4 qb = *(const float4 *)(q_s + (size_t)e * G + 4);
                    part[0] += qa.x * kve;
                    part[1] += qa.y * kve;
                    part[2] += qa.z * kve;
                    part[3] += qa.w * kve;
                    part[4] += qb.x * kve;
                    part[5] += qb.y * kve;
                    part[6] += qb.z * kve;
                    part[7] += qb.w * kve;
                }
            }
#pragma unroll
            for (int g = 0; g < G; g++) {
                float p = part[g];
                for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)
                    p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,
                                          FC_ATTN_DOT_LANES);
                if (lane == 0 && ky < tk)
                    score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;
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
#if FC_ATTN_KVF_MUTATE
            for (int j = tk - 1; j >= 0; j--) {    /* falsifier: reversed sum */
#else
            for (int j = 0; j < tk; j++) {
#endif
                float pj = expf(sc[j] - m_new);
                p_s[(size_t)j * G + tid] = pj;
                sum += pj;
            }
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = m_new;
            corr_s[tid] = corr;
        }

        /* V tile as raw q4 words, in the softmax shadow (kvs). */
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

        const unsigned char *vb = (const unsigned char *)v_sh + vqoff;
        const int all_q4 = j0 >= sink && (win <= 0 || j0 + tk - 1 <= cur_pos - win);
        if (all_q4 && tk == FC_ATTN_TILE_KEYS) {
#pragma unroll
            for (int j = 0; j < FC_ATTN_TILE_KEYS; j++) {
                const unsigned char *bp = vb + j * row_bytes;
                const float d = __half2float(__ushort_as_half(*(const unsigned short *)bp));
                fc_attn_kvf_pv(acc, p_s + (size_t)j * G, fc_kvf_dec_v(bp[vbyte], vshift, d),
                               j == 0, corr_s);
            }
        } else {
            for (int j = 0; j < tk; j++) {
                const int vp = j0 + j;
                float vv;
                if (vp < sink) {
                    vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);
                } else if (win > 0 && vp > cur_pos - win) {
                    vv = __half2float(v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]);
                } else {
                    const unsigned char *bp = vb + j * row_bytes;
                    const float d = __half2float(__ushort_as_half(*(const unsigned short *)bp));
                    vv = fc_kvf_dec_v(bp[vbyte], vshift, d);
                }
                fc_attn_kvf_pv(acc, p_s + (size_t)j * G, vv, j == 0, corr_s);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < G; g++) {
        size_t po = (head0 + g) * stride + split;
        acc_out[po * head_dim + tid] = acc[g];
        if (tid == 0) {
            m_out[po] = m_s[g];
            l_out[po] = l_s[g];
        }
    }
}
