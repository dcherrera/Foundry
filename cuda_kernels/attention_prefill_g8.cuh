/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file attention_prefill_g8.cuh
 * @brief Causal GQA prefill flash attention over the Q4 KV cache, specialised
 *   at compile time for kv_groups = 8 and head_dim = 256 (Qwen3.6/3.8 full
 *   attention, one TP2 rank or the whole model). Bit-exact against
 *   fc_attn_flash_prefill_q4_kvs: same per-output arithmetic, different thread
 *   mapping and operand motion only.
 *
 * Included by attention.cu INSIDE no extern "C" block (the body is a template);
 * attention.cu instantiates the shipped form as fc_attn_flash_prefill_q4_g8 and
 * cuda_kernels/tests/bench_attn_prefill_g8.cu instantiates every form.
 *
 * WHAT STAYS IDENTICAL per output element (the exactness contract; anything not
 * listed is free to move):
 *   1. score = 16-lane partials, lane l: sum_i q[l+16i]*k[l+16i] as FFMA in i
 *      order; the width-16 shuffle-down tree 8,4,2,1; then * scale.
 *   2. per 16-key tile from key 0: m_new = fmaxf fold over m_old and the tile's
 *      scores (fmaxf is exact and order-free); corr = expf(m_old - m_new);
 *      p_j = expf(s_j - m_new); sum = p_0 + p_1 + ... in j order;
 *      l = l * corr + sum.
 *   3. acc: j ascending, `if (j == 0) acc *= corr; acc += p_j * v_j` written
 *      exactly as kvs writes it (ptxas contracts the j == 0 pair to one FFMA;
 *      hoisting it moves outputs by 1 ulp).
 *   4. K/V dequant and the fp16 sink/window path select incl. win_live_lo.
 *   5. out = acc / l.
 *
 * THE FORMS (template arguments; each is a single-variable change on kvs):
 *   QVEC   q staged as [i][g/4][lane][g%4], so the QK loop reads a lane's 8 q
 *          values for one i as two LDS.128 instead of eight scalar LDS
 *          (128 -> 32 q loads a thread a tile). G = 8 at compile time also
 *          drops every `if (g < G)` predicate.
 *   PSOFT  the tile's 16 x 8 expf run on 128 threads instead of serially on 8;
 *          the sequential sum and the l / m update move into the PV phase on 8
 *          threads, so the barrier count stays 3 a tile.
 *   PVEC   keys j >= 1 read their 8 p values as two LDS.128; key 0 keeps kvs's
 *          exact source form. (Vectorising key 0 too changed ptxas's contraction
 *          of the fused rescale: 72k of 262k outputs moved.)
 *   KSH    the next tile's K rows are staged as q4 words in shared during
 *          the softmax phase and dequantised from there (same bytes, same
 *          decode) instead of 24 global byte loads a thread a tile.
 *   POISON the bench's red arm: one ulp on every q4 V value (must differ).
 *
 * Launch: grid (npos, n_head_kv), block 256, dynamic shared
 * fc_attn_g8_shared_bytes() = 8*256*4 + (16*8 [score] + 16*8 [p] + 4*8) * 4
 * + 16*144 [V tile] = 11 648 B (+ 2 304 B K tile for KSH).
 *
 * @status Microbenchmark gate first (bench_attn_prefill_g8.cu): 0 differing
 *   floats against kvs at depth 8k-32k, red arm differs.
 * @issues Registers are the residency gate (kvs: 72 regs = 7 blocks/SMX). The
 *   OFFLINE ptxas count in the bench is not the shipping JIT's.
 * @todo None
 */

#ifndef FC_ATTENTION_PREFILL_G8_CUH
#define FC_ATTENTION_PREFILL_G8_CUH

#define FC_ATTN_G8_G     8
#define FC_ATTN_G8_HD    256
#define FC_ATTN_G8_CHUNK (FC_ATTN_G8_HD / FC_ATTN_DOT_LANES)   /* 16 */

/* Shared layout, in floats from the base: q (G*HD), score [G][16], p [16][G],
 * m [G], l [G], corr [G], m_new [G], then the V tile as words. */
#define FC_ATTN_G8_Q_FLOATS     (FC_ATTN_G8_G * FC_ATTN_G8_HD)
#define FC_ATTN_G8_SCORE_OFF    FC_ATTN_G8_Q_FLOATS
#define FC_ATTN_G8_P_OFF        (FC_ATTN_G8_SCORE_OFF + FC_ATTN_G8_G * FC_ATTN_TILE_KEYS)
#define FC_ATTN_G8_M_OFF        (FC_ATTN_G8_P_OFF + FC_ATTN_TILE_KEYS * FC_ATTN_G8_G)
#define FC_ATTN_G8_L_OFF        (FC_ATTN_G8_M_OFF + FC_ATTN_G8_G)
#define FC_ATTN_G8_CORR_OFF     (FC_ATTN_G8_L_OFF + FC_ATTN_G8_G)
#define FC_ATTN_G8_MNEW_OFF     (FC_ATTN_G8_CORR_OFF + FC_ATTN_G8_G)
#define FC_ATTN_G8_V_OFF        (FC_ATTN_G8_MNEW_OFF + FC_ATTN_G8_G)
/* KSH form only: the next tile's K rows as q4 words, after the V tile. */
#define FC_ATTN_G8_TILE_WORDS   (FC_ATTN_TILE_KEYS * (FC_ATTN_G8_HD / FC_KVQ4_BLK) * \
                                 FC_KVQ4_BLK_BYTES / 4)
#define FC_ATTN_G8_K_OFF        (FC_ATTN_G8_V_OFF + FC_ATTN_G8_TILE_WORDS)

/* Stage tk rows of q4 bytes from src as words into dst (all loads guarded; a
 * partial last tile must not read past the cache). 3 words a thread. */
__device__ __forceinline__ void fc_attn_g8_stage_words(unsigned int *dst,
                                                       const unsigned char *src,
                                                       int tk, int row_bytes,
                                                       int tid) {
    const unsigned int *w = (const unsigned int *)src;
    const int nw = (tk * row_bytes) >> 2;
    const int i0 = tid, i1 = tid + FC_ATTN_G8_HD, i2 = tid + 2 * FC_ATTN_G8_HD;
    const unsigned int w0 = i0 < nw ? w[i0] : 0u;
    const unsigned int w1 = i1 < nw ? w[i1] : 0u;
    const unsigned int w2 = i2 < nw ? w[i2] : 0u;
    if (i0 < nw)
        dst[i0] = w0;
    if (i1 < nw)
        dst[i1] = w1;
    if (i2 < nw)
        dst[i2] = w2;
}

/* Index of q[g][e] in the QVEC layout: e = lane + 16 i, g = 4 gh + gq. */
#define FC_ATTN_G8_QIDX(g, e)                                                  \
    (((((e) / FC_ATTN_DOT_LANES) * 2 + (g) / 4) * FC_ATTN_DOT_LANES +           \
      (e) % FC_ATTN_DOT_LANES) * 4 + (g) % 4)

template <bool QVEC, bool PSOFT, bool PVEC, bool KSH, bool POISON>
__device__ __forceinline__ void fc_attn_flash_prefill_q4_g8_body(
        const float *q, const unsigned char *k_cache,
        const unsigned char *v_cache, const __half *k_sink,
        const __half *v_sink, const __half *k_win, const __half *v_win,
        float *out, int max_seq, int pos0, int sink, int win, int attn_sink,
        int attn_win, float scale)
{
    const int G = FC_ATTN_G8_G;
    const int head_dim = FC_ATTN_G8_HD;
    const int t = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    const int seq_len = pos0 + t + 1;
    const int cur_pos = seq_len - 1;
    const int row_bytes = (FC_ATTN_G8_HD / FC_KVQ4_BLK) * FC_KVQ4_BLK_BYTES;
    const int win_live_lo = (cur_pos - t + (int)gridDim.x - 1) - FC_KVQ4_WMAX;

    extern __shared__ float4 g8_sh4[];
    float *sh = (float *)g8_sh4;
    float *q_s = sh;
    float *score_s = sh + FC_ATTN_G8_SCORE_OFF;
    float *p_s = sh + FC_ATTN_G8_P_OFF;
    float *m_s = sh + FC_ATTN_G8_M_OFF;
    float *l_s = sh + FC_ATTN_G8_L_OFF;
    float *corr_s = sh + FC_ATTN_G8_CORR_OFF;
    float *mnew_s = sh + FC_ATTN_G8_MNEW_OFF;
    unsigned int *v_sh = (unsigned int *)(sh + FC_ATTN_G8_V_OFF);
    unsigned int *k_sh = (unsigned int *)(sh + FC_ATTN_G8_K_OFF);
    /* The K prefetch assumes no tile is skipped; a windowed call (excluded by
     * the host contract anyway) reads K from global as kvs does. */
    const bool ksh = KSH && attn_win <= 0;

    const size_t n_head = (size_t)gridDim.y * G;
    const float *q_base = q + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;
#pragma unroll
    for (int g = 0; g < G; g++) {
        const float qv = q_base[(size_t)g * head_dim + tid];
        if (QVEC)
            q_s[FC_ATTN_G8_QIDX(g, tid)] = qv;
        else
            q_s[(size_t)g * head_dim + tid] = qv;
    }
    if (tid < G) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    if (ksh) {
        const unsigned char *k0 = k_cache + (size_t)kvh * max_seq * row_bytes;
        fc_attn_g8_stage_words(k_sh, k0, seq_len < FC_ATTN_TILE_KEYS ? seq_len
                                                                      : FC_ATTN_TILE_KEYS,
                               row_bytes, tid);
    }
    __syncthreads();

    float acc[FC_ATTN_G8_G];
#pragma unroll
    for (int g = 0; g < G; g++)
        acc[g] = 0.0f;

    const unsigned char *k_head = k_cache + (size_t)kvh * max_seq * row_bytes;
    const unsigned char *v_head = v_cache + (size_t)kvh * max_seq * row_bytes;
    const __half *k_win_head = k_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *v_win_head = v_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *k_sink_head = k_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const __half *v_sink_head = v_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = FC_ATTN_G8_CHUNK;

    const int vlocal = tid & (FC_KVQ4_BLK - 1);
    const int vhalf = FC_KVQ4_BLK / 2;
    const int vbyte = 2 + (vlocal < vhalf ? vlocal : vlocal - vhalf);
    const int vshift = vlocal < vhalf ? 0 : 4;
    const int vqoff = (tid / FC_KVQ4_BLK) * FC_KVQ4_BLK_BYTES;

    /* PSOFT: thread tid < 128 owns score (sg, sj) of the tile. */
    const int sg = tid / FC_ATTN_TILE_KEYS;
    const int sj = tid % FC_ATTN_TILE_KEYS;

    for (int j0 = 0; j0 < seq_len; j0 += FC_ATTN_TILE_KEYS) {
        int tk = seq_len - j0;
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
            } else if (win > 0 && kp > cur_pos - win && kp > win_live_lo) {
                const __half *s = k_win_head +
                    (size_t)(kp % FC_KVQ4_WMAX) * head_dim;
#pragma unroll
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++)
                    if (i < chunk)
                        kreg[i] = __half2float(s[lane + i * FC_ATTN_DOT_LANES]);
            } else if (ksh) {
                /* fc_kvq4_dequant_lane_chunk on the staged row, inline so the
                 * loads stay in the shared address space. */
                const unsigned char *row =
                    (const unsigned char *)k_sh + ky * row_bytes;
#pragma unroll
                for (int b = 0; b < FC_ATTN_KREG_MAX_CHUNK / 2; b++) {
                    const unsigned char *bp = row + b * FC_KVQ4_BLK_BYTES;
                    const unsigned short d_bits = (unsigned short)(bp[0] | (bp[1] << 8));
                    const float d = __half2float(__ushort_as_half(d_bits));
                    const unsigned int codes = bp[2 + lane];
                    kreg[2 * b] = (float)((int)(codes & 0xF) - 8) * d;
                    kreg[2 * b + 1] = (float)((int)((codes >> 4) & 0xF) - 8) * d;
                }
            } else {
                fc_kvq4_dequant_lane_chunk(k_head + (size_t)kp * row_bytes,
                                           lane, chunk, kreg);
            }
        }

        /* QK^T: per g, FFMA over i ascending, then the 16-lane tree. */
        {
            float part[FC_ATTN_G8_G];
#pragma unroll
            for (int g = 0; g < G; g++)
                part[g] = 0.0f;
            if (ky < tk) {
#pragma unroll
                for (int i = 0; i < FC_ATTN_G8_CHUNK; i++) {
                    const int e = lane + i * FC_ATTN_DOT_LANES;
                    const float kve = kreg[i];
                    if (QVEC) {
                        const float4 *q4 = (const float4 *)q_s;
                        const float4 qa = q4[(i * 2 + 0) * FC_ATTN_DOT_LANES + lane];
                        const float4 qb = q4[(i * 2 + 1) * FC_ATTN_DOT_LANES + lane];
                        part[0] += qa.x * kve;
                        part[1] += qa.y * kve;
                        part[2] += qa.z * kve;
                        part[3] += qa.w * kve;
                        part[4] += qb.x * kve;
                        part[5] += qb.y * kve;
                        part[6] += qb.z * kve;
                        part[7] += qb.w * kve;
                    } else {
#pragma unroll
                        for (int g = 0; g < G; g++)
                            part[g] += q_s[(size_t)g * head_dim + e] * kve;
                    }
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

        if (PSOFT) {
            /* Every score thread folds its head's max itself (exact in any
             * order), takes its own expf; key 0's thread also takes corr. */
            if (tid < G * FC_ATTN_TILE_KEYS && sj < tk) {
                const float *sc = score_s + sg * FC_ATTN_TILE_KEYS;
                float m_new = m_s[sg];
                for (int j = 0; j < tk; j++)
                    m_new = fmaxf(m_new, sc[j]);
                p_s[(size_t)sj * G + sg] = expf(sc[sj] - m_new);
                if (sj == 0) {
                    corr_s[sg] = expf(m_s[sg] - m_new);
                    mnew_s[sg] = m_new;
                }
            }
        } else if (tid < G) {
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

        /* KSH: the NEXT tile's K rows (QK of this tile is done, so the one
         * buffer is free; the next QK reads it after the end-of-PV barrier). */
        if (ksh && j0 + FC_ATTN_TILE_KEYS < seq_len) {
            const int nk = seq_len - (j0 + FC_ATTN_TILE_KEYS);
            fc_attn_g8_stage_words(k_sh, k_head + (size_t)(j0 + FC_ATTN_TILE_KEYS) * row_bytes,
                                   nk < FC_ATTN_TILE_KEYS ? nk : FC_ATTN_TILE_KEYS,
                                   row_bytes, tid);
        }

        /* V tile staged as words, as kvs does (all loads guarded). */
        {
            const unsigned int *vsrc =
                (const unsigned int *)(v_head + (size_t)j0 * row_bytes);
            const int vwords = (tk * row_bytes) >> 2;
            const int w0_idx = tid;
            const int w1_idx = tid + FC_ATTN_G8_HD;
            const int w2_idx = tid + 2 * FC_ATTN_G8_HD;
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

        /* PSOFT: the sequential sum and the l / m update, off the barrier
         * path (nothing else reads l_s / m_s until after the next barrier). */
        if (PSOFT && tid < G) {
            float sum = 0.0f;
            for (int j = 0; j < tk; j++)
                sum += p_s[(size_t)j * G + tid];
            const float corr = corr_s[tid];
            l_s[tid] = l_s[tid] * corr + sum;
            m_s[tid] = mnew_s[tid];
        }

        for (int j = 0; j < tk; j++) {
            int vp = j0 + j;
            float vv;
            if (vp < sink) {
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);
            } else if (win > 0 && vp > cur_pos - win && vp > win_live_lo) {
                vv = __half2float(v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]);
            } else {
                const unsigned char *bp =
                    (const unsigned char *)v_sh + j * row_bytes + vqoff;
                const unsigned short d_bits = (unsigned short)(bp[0] | (bp[1] << 8));
                const float d = __half2float(__ushort_as_half(d_bits));
                const int code = (bp[vbyte] >> vshift) & 0xF;
                vv = (float)(code - 8) * d;
                if (POISON)   /* red arm only: one ulp on every q4 V */
                    vv = __fmul_rn(vv, 1.00000011920928955078125f);
            }
            if (PVEC && j > 0) {
                /* j > 0 has no rescale: acc += p * v is one FFMA either way. */
                const float4 *p4 = (const float4 *)(p_s + (size_t)j * G);
                const float4 pa = p4[0];
                const float4 pb = p4[1];
                const float pj[FC_ATTN_G8_G] = { pa.x, pa.y, pa.z, pa.w,
                                                 pb.x, pb.y, pb.z, pb.w };
#pragma unroll
                for (int g = 0; g < G; g++)
                    acc[g] += pj[g] * vv;
            } else {
                const float *pj_s = p_s + (size_t)j * G;
#pragma unroll
                for (int g = 0; g < G; g++) {
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
    for (int g = 0; g < G; g++)
        out_base[(size_t)g * head_dim + tid] = acc[g] / l_s[g];
}

#endif /* FC_ATTENTION_PREFILL_G8_CUH */
