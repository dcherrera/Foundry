/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file exp_attn_qtile.cu
 * @brief Q-dimension-tiled prefill flash-attention variants for the K80/GK210
 *   Q_TILE x split-KV microbenchmark. Includes attention.cu VERBATIM so the
 *   shipped kernel is a byte-identical control symbol inside the same module.
 *
 * Hypothesis under test: fc_attn_flash_prefill_q4 launches one block per QUERY
 * row (grid = npos x n_head_kv), so its staged K tile amortises across the 8
 * GQA heads but NOT across the 128 query positions -- the same KV is re-staged
 * 128x per chunk. Giving one block Q_TILE query rows multiplies arithmetic
 * intensity per staged K byte by Q_TILE.
 *
 * The structural cost is shared memory: q_s is Q_TILE * kv_groups * head_dim
 * floats, i.e. 1024 B per unit of arithmetic intensity at the shipped shape.
 * Q_TILE=2 with the fp32 K tile still in shared is 36 032 B -> 3 blocks/SMX,
 * below the shipped kernel's 4. KREG=1 therefore moves the K tile OUT of shared
 * and into per-thread REGISTERS: thread (ky, lane) already reads exactly the 16
 * elements k[ky][lane + 16*i], so the shared round trip (STS + LDS) is pure
 * overhead once the staging assignment is changed to match the QK access
 * pattern. That is what buys the shared-memory budget back.
 *
 * BIT-EXACTNESS. Every arm with BREAK=0 is intended to be bit-exact against
 * fc_attn_flash_prefill_q4: the accumulation order over e (QK) and over j (PV)
 * is unchanged, only the operand's home changes. The one place Q-tiling can
 * change VALUES is the fp16 sink/recent-window predicate, which depends on the
 * query row's cur_pos: for a Q-tile of rows [t0, t0+QT) the key positions in
 * (P0 - win, P0 + QT - 1 - win] read from the fp16 shadow for some rows and
 * from the q4 body for others. Such a tile is handled in QT separate passes
 * (one query row each), so the per-row read path is reproduced exactly. That
 * costs QT passes on at most one tile of the whole sequence.
 *
 * @status Microbenchmark only. Nothing here is reachable from Pantheon or from
 *   fc_cuda.c; no shipped kernel or default is modified.
 * @issues
 *   - KREG=1 requires head_dim <= 256 (kreg[] must be compile-time bounded, see
 *     XQ_MAX_CHUNK) and HOIST=1 additionally assumes FC_ATTN_DOT_LANES == 16.
 *   - attn_win > 0 (sliding-window attention) is not covered by the Q-tiled
 *     path: the tile predicate is evaluated as an OR across the Q-tile's rows,
 *     which is correct only for attn_win <= 0 (the shipped configuration).
 * @todo None
 */

#include "../attention.cu"

/* Compile-time bound on head_dim / FC_ATTN_DOT_LANES. A kreg[] sized by the
 * RUNTIME chunk count is never promoted to registers -- it lands in LOCAL
 * memory and ptxas prints "0 bytes spill" beside a nonzero stack frame (see
 * attention.cu's acc[] comment). 16 covers head_dim <= 256. */
#define XQ_MAX_CHUNK 16

/* Dequantise the 16 Q4_0 elements a (ky, lane) thread owns -- lane + 16*i for
 * i in [0, chunk) -- with the fp16 scale hoisted per 32-element block.
 *
 * Valid only for FC_ATTN_DOT_LANES == 16, so lane < 16 and element lane + 16*i
 * lies in block i/2 at byte 2 + lane, low nibble for even i and high nibble for
 * odd i. That collapses 3 byte loads per element (48) to 3 per block (24) and
 * removes the integer division in fc_kvq4_dequant_elem. Bit-identical to
 * calling fc_kvq4_dequant_elem per element.
 *
 * @param row    first block header of the key's Q4_0 row
 * @param lane   threadIdx.x % FC_ATTN_DOT_LANES, must be < 16
 * @param chunk  head_dim / FC_ATTN_DOT_LANES, must be <= XQ_MAX_CHUNK
 * @param kreg   [XQ_MAX_CHUNK] output, entries [0, chunk) written
 */
__device__ __forceinline__ void xq_dequant_lane_chunk(const unsigned char *row,
                                                      int lane, int chunk,
                                                      float *kreg)
{
#pragma unroll
    for (int b = 0; b < XQ_MAX_CHUNK / 2; b++) {
        if (2 * b < chunk) {
            const unsigned char *bp = row + (size_t)b * FC_KVQ4_BLK_BYTES;
            unsigned short raw = (unsigned short)(bp[0] | (bp[1] << 8));
            float d = __half2float(*reinterpret_cast<const __half *>(&raw));
            unsigned char byte = bp[2 + lane];
            kreg[2 * b] = (float)((int)(byte & 0xF) - 8) * d;
            if (2 * b + 1 < chunk)
                kreg[2 * b + 1] = (float)((int)((byte >> 4) & 0xF) - 8) * d;
        }
    }
}

/* Q-tiled causal GQA flash-attention prefill over the Q4 KV cache.
 *
 * QT       query rows per block (grid.x = npos / QT)
 * KREG     0 = stage the K tile in shared as the shipped kernel does;
 *          1 = hold it in per-thread registers, no shared K tile at all
 * SPLITKV  1 = one block owns the whole KV range; >1 = grid.z slices the range
 *          and each slice writes its own PARTIAL output (NO combine pass, so
 *          the result is not the attention output -- a timing upper bound only)
 * HOIST    1 = use xq_dequant_lane_chunk for the register K load (KREG=1 only)
 * BREAK    1 = deliberately corrupt the K read path (harness falsification arm)
 */
template <int QT, int KREG, int SPLITKV, int HOIST, int BREAK>
__device__ __forceinline__ void xq_prefill_q4_body(
    const float *q, const unsigned char *k_cache, const unsigned char *v_cache,
    const __half *k_sink, const __half *v_sink, const __half *k_win,
    const __half *v_win, float *out, int kv_groups, int head_dim, int max_seq,
    int pos0, int npos, int sink, int win, int attn_sink, int attn_win,
    float scale)
{
    const int t0 = (int)blockIdx.x * QT;
    const int kvh = (int)blockIdx.y;
    const int zs = SPLITKV > 1 ? (int)blockIdx.z : 0;
    const int tid = (int)threadIdx.x;
    const int G = kv_groups;
    const int QG = QT * G;
    const int P0 = pos0 + t0;               /* absolute position of row t0 */
    const int seq_max = P0 + QT;            /* keys the deepest row attends */
    const int row_bytes = fc_kvq4_row_bytes(head_dim);
    /* Same liveness floor as the shipped kernel, but from npos rather than
     * gridDim.x -- Q-tiling shrinks the grid without shrinking the batch. */
    const int win_live_lo = (pos0 + npos - 1) - FC_KVQ4_WMAX;
    const int kstride = FC_ATTN_K_STRIDE(head_dim);

    /* Renamed from attention.cu's `sh` only to keep the two declarations of the
     * dynamic-shared window from colliding in one translation unit. */
    extern __shared__ float xq_sh[];
    float *q_s = xq_sh;                                       /* [QG][head_dim] */
    float *k_s = q_s + (size_t)QG * head_dim;              /* KREG == 0 only */
    float *score_s = k_s + (KREG ? (size_t)0
                                 : (size_t)FC_ATTN_TILE_KEYS * kstride);
    float *p_s = score_s + (size_t)QG * FC_ATTN_TILE_KEYS;  /* [TILE][QG] */
    float *m_s = p_s + (size_t)QG * FC_ATTN_TILE_KEYS;
    float *l_s = m_s + QG;
    float *corr_s = l_s + QG;

    const size_t n_head = (size_t)gridDim.y * G;
#pragma unroll
    for (int qt = 0; qt < QT; qt++) {
        const float *qb =
            q + ((size_t)(t0 + qt) * n_head + (size_t)kvh * G) * head_dim;
        for (int g = 0; g < G; g++)
            q_s[(size_t)(qt * G + g) * head_dim + tid] =
                qb[(size_t)g * head_dim + tid];
    }
    if (tid < QG) {
        m_s[tid] = -__int_as_float(0x7f800000);
        l_s[tid] = 0.0f;
    }
    __syncthreads();

    /* acc[] is indexed only by compile-time (qt, g) -- the QT stride is the
     * compile-time FC_ATTN_MAX_GROUPS, never the runtime G, so every index is
     * a constant after unrolling and the array stays in registers. */
    float acc[QT * FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int a = 0; a < QT * FC_ATTN_MAX_GROUPS; a++)
        acc[a] = 0.0f;

    const unsigned char *k_head = k_cache + (size_t)kvh * max_seq * row_bytes;
    const unsigned char *v_head = v_cache + (size_t)kvh * max_seq * row_bytes;
    const __half *k_win_head = k_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *v_win_head = v_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;
    const __half *k_sink_head = k_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const __half *v_sink_head = v_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;
    const int ky = tid / FC_ATTN_DOT_LANES;
    const int lane = tid % FC_ATTN_DOT_LANES;
    const int chunk = head_dim / FC_ATTN_DOT_LANES;

    int ntile = (seq_max + FC_ATTN_TILE_KEYS - 1) / FC_ATTN_TILE_KEYS;
    int jt_lo = 0, jt_hi = ntile;
    if (SPLITKV > 1) {
        int per = (ntile + SPLITKV - 1) / SPLITKV;
        jt_lo = zs * per;
        jt_hi = jt_lo + per;
        if (jt_hi > ntile) jt_hi = ntile;
        if (jt_lo > ntile) jt_lo = ntile;
    }

    for (int jt = jt_lo; jt < jt_hi; jt++) {
        const int j0 = jt * FC_ATTN_TILE_KEYS;
        int tk = seq_max - j0;
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;

        /* Sliding window: keep the tile if ANY row of the Q-tile attends it. */
        int in_any = 0;
#pragma unroll
        for (int qt = 0; qt < QT; qt++)
            if (fc_attn_tile_in_window(j0, tk, P0 + qt, attn_sink, attn_win))
                in_any = 1;
        if (!in_any) continue;

        /* Does the fp16-shadow predicate differ across the Q-tile's rows? Only
         * key positions in (P0 - win, P0 + QT - 1 - win] can, so at most QT - 1
         * positions of the whole sequence, i.e. at most two tiles. */
        const int mlo = P0 - win + 1;
        const int mhi = P0 + QT - 1 - win;
        const int mixed = (QT > 1) && (win > 0) &&
                          (j0 <= mhi) && (j0 + tk - 1 >= mlo);
        const int npass = mixed ? QT : 1;

        for (int pass = 0; pass < npass; pass++) {
            const int qlo = mixed ? pass : 0;
            const int qhi = mixed ? pass + 1 : QT;
            const int ppos = P0 + (mixed ? pass : 0);

            /* Per-row causal key count for this tile. */
            int tkq[QT];
#pragma unroll
            for (int qt = 0; qt < QT; qt++) {
                int c = (P0 + qt) - j0 + 1;
                if (c > tk) c = tk;
                if (c < 0) c = 0;
                tkq[qt] = c;
            }

            float kreg[XQ_MAX_CHUNK];
            if (KREG) {
                if (ky < tk) {
                    const int kp = j0 + ky;
                    const int use_win = BREAK ? (win < 0)
                                              : (win > 0 && kp > ppos - win &&
                                                 kp > win_live_lo);
                    if (kp < sink) {
                        const __half *s = k_sink_head + (size_t)kp * head_dim;
#pragma unroll
                        for (int i = 0; i < XQ_MAX_CHUNK; i++)
                            if (i < chunk)
                                kreg[i] = __half2float(
                                    s[lane + i * FC_ATTN_DOT_LANES]);
                    } else if (use_win) {
                        const __half *s = k_win_head +
                            (size_t)(kp % FC_KVQ4_WMAX) * head_dim;
#pragma unroll
                        for (int i = 0; i < XQ_MAX_CHUNK; i++)
                            if (i < chunk)
                                kreg[i] = __half2float(
                                    s[lane + i * FC_ATTN_DOT_LANES]);
                    } else {
                        const unsigned char *s = k_head + (size_t)kp * row_bytes;
                        if (HOIST) {
                            xq_dequant_lane_chunk(s, lane, chunk, kreg);
                        } else {
#pragma unroll
                            for (int i = 0; i < XQ_MAX_CHUNK; i++)
                                if (i < chunk)
                                    kreg[i] = fc_kvq4_dequant_elem(
                                        s, lane + i * FC_ATTN_DOT_LANES);
                        }
                    }
                }
            } else {
                for (int key = 0; key < tk; key++) {
                    const int kp = j0 + key;
                    float *krow = k_s + (size_t)key * kstride;
                    if (kp < sink) {
                        const __half *s = k_sink_head + (size_t)kp * head_dim;
                        for (int e = tid; e < head_dim; e += blockDim.x)
                            krow[e] = __half2float(s[e]);
                    } else if (win > 0 && kp > ppos - win && kp > win_live_lo) {
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
            }

            /* QK^T, i-outer / g-inner, one register partial per GQA group. The
             * K operand is read once per element and fanned across the group;
             * with KREG it never touches shared memory at all. */
#pragma unroll
            for (int qt = 0; qt < QT; qt++) {
                if (qt < qlo || qt >= qhi) continue;
                float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
                for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                    part[g] = 0.0f;
                if (ky < tk) {
                    if (KREG) {
#pragma unroll
                        for (int i = 0; i < XQ_MAX_CHUNK; i++) {
                            if (i < chunk) {
                                const int e = lane + i * FC_ATTN_DOT_LANES;
                                const float kve = kreg[i];
#pragma unroll
                                for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                                    if (g < G)
                                        part[g] += q_s[(size_t)(qt * G + g) *
                                                       head_dim + e] * kve;
                            }
                        }
                    } else {
                        const float *kv = k_s + (size_t)ky * kstride;
                        for (int i = 0; i < chunk; i++) {
                            const int e = lane + i * FC_ATTN_DOT_LANES;
                            const float kve = kv[e];
#pragma unroll
                            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                                if (g < G)
                                    part[g] += q_s[(size_t)(qt * G + g) *
                                                   head_dim + e] * kve;
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
                            score_s[(size_t)(qt * G + g) * FC_ATTN_TILE_KEYS +
                                    ky] = p * scale;
                    }
                }
            }
            __syncthreads();

            /* Online-softmax fold: one thread per (query row, GQA group). */
            {
                const int nq = qhi - qlo;
                if (tid < nq * G) {
                    const int qt = qlo + tid / G;
                    const int g = tid % G;
                    const int idx = qt * G + g;
                    int c = (P0 + qt) - j0 + 1;
                    if (c > tk) c = tk;
                    if (c > 0) {
                        const float *sc = score_s +
                            (size_t)idx * FC_ATTN_TILE_KEYS;
                        float m_new = m_s[idx];
                        for (int j = 0; j < c; j++)
                            m_new = fmaxf(m_new, sc[j]);
                        const float corr = expf(m_s[idx] - m_new);
                        float sum = 0.0f;
                        for (int j = 0; j < c; j++) {
                            const float pj = expf(sc[j] - m_new);
                            p_s[(size_t)j * QG + idx] = pj;
                            sum += pj;
                        }
                        l_s[idx] = l_s[idx] * corr + sum;
                        m_s[idx] = m_new;
                        corr_s[idx] = corr;
                    } else {
                        corr_s[idx] = 1.0f;
                    }
                }
            }
            __syncthreads();

            /* PV accumulate. The online-softmax rescale MUST stay fused inside
             * the j loop at j == 0, exactly where the shipped kernel puts it:
             * hoisting it into its own loop is algebraically identical but
             * costs ptxas the `acc = fma(acc, corr, p0*v0)` contraction, which
             * moved 148 099 of 524 288 output floats by up to 4.0e-7 relative
             * -- a 1-ulp-class difference, measured, on the template control
             * that changes nothing else. For every row with tkq > 0, j == 0 is
             * that row's first accumulated key, so the fusion point is the same
             * one the shipped kernel uses. */
            for (int j = 0; j < tk; j++) {
                const int vp = j0 + j;
                float vv;
                if (vp < sink)
                    vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);
                else if (win > 0 && vp > ppos - win && vp > win_live_lo)
                    vv = __half2float(v_win_head[
                        (size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]);
                else
                    vv = fc_kvq4_dequant_elem(v_head + (size_t)vp * row_bytes,
                                              tid);
#pragma unroll
                for (int qt = 0; qt < QT; qt++) {
                    if (qt < qlo || qt >= qhi) continue;
                    if (j < tkq[qt]) {
                        const float *pj_s = p_s + (size_t)j * QG + qt * G;
#pragma unroll
                        for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                            if (g < G) {
                                if (j == 0)
                                    acc[qt * FC_ATTN_MAX_GROUPS + g] *=
                                        corr_s[qt * G + g];
                                acc[qt * FC_ATTN_MAX_GROUPS + g] +=
                                    pj_s[g] * vv;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    const size_t slice = SPLITKV > 1
        ? (size_t)zs * (size_t)npos * n_head * head_dim : (size_t)0;
#pragma unroll
    for (int qt = 0; qt < QT; qt++) {
        float *ob = out + slice +
            ((size_t)(t0 + qt) * n_head + (size_t)kvh * G) * head_dim;
#pragma unroll
        for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
            if (g < G)
                ob[(size_t)g * head_dim + tid] =
                    acc[qt * FC_ATTN_MAX_GROUPS + g] / l_s[qt * G + g];
    }
}

#define XQ_ENTRY(NAME, QT, KREG, SPLITKV, HOIST, BREAK)                        \
extern "C" __global__ void NAME(                                               \
    const float *q, const unsigned char *k_cache,                              \
    const unsigned char *v_cache, const __half *k_sink, const __half *v_sink,  \
    const __half *k_win, const __half *v_win, float *out, int kv_groups,       \
    int head_dim, int max_seq, int pos0, int npos, int sink, int win,           \
    int attn_sink, int attn_win, float scale)                                   \
{                                                                              \
    xq_prefill_q4_body<QT, KREG, SPLITKV, HOIST, BREAK>(                        \
        q, k_cache, v_cache, k_sink, v_sink, k_win, v_win, out, kv_groups,      \
        head_dim, max_seq, pos0, npos, sink, win, attn_sink, attn_win, scale);  \
}

/*                name          QT KREG SPLIT HOIST BREAK */
XQ_ENTRY(xq_q1_sh,               1,  0,   1,    0,   0)   /* template control */
XQ_ENTRY(xq_q1_krn,              1,  1,   1,    0,   0)   /* K in regs, naive dequant */
XQ_ENTRY(xq_q1_kr,               1,  1,   1,    1,   0)   /* K in regs, hoisted */
XQ_ENTRY(xq_q2_sh,               2,  0,   1,    0,   0)   /* Q_TILE=2, shared K */
XQ_ENTRY(xq_q2_krn,              2,  1,   1,    0,   0)
XQ_ENTRY(xq_q2_kr,               2,  1,   1,    1,   0)
XQ_ENTRY(xq_q4_kr,               4,  1,   1,    1,   0)
XQ_ENTRY(xq_q1_kr_s2,            1,  1,   2,    1,   0)   /* split-KV upper bound */
XQ_ENTRY(xq_q1_kr_s4,            1,  1,   4,    1,   0)
XQ_ENTRY(xq_q2_kr_s2,            2,  1,   2,    1,   0)
XQ_ENTRY(xq_q4_kr_s2,            4,  1,   2,    1,   0)
XQ_ENTRY(xq_q4_kr_s4,            4,  1,   4,    1,   0)
XQ_ENTRY(xq_broken,              1,  1,   1,    1,   1)   /* falsification arm */
