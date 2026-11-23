/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   attention_decode_rows.cuh
 * @brief  R-row decode split flash attention over ONE session's Q4 KV ring —
 *         the speculative-verify kernel (fc_attn_flash_decode_rows_q4_kvs).
 *
 * Included at the end of attention.cu (same PTX module, same helpers).
 *
 * A speculative verify runs R consecutive positions pos0..pos0+R-1 of one
 * session. Every row attends the SAME ring, row r over keys [0, pos0+r]. The
 * prefill flash kernel launches one block per (row, kv head) that walks the
 * whole context serially (2-10 blocks on a 13-SMX die: ~6x the decode step's
 * per-position cost). This kernel is the decode split kernel
 * (fc_attn_flash_decode_slots_q4_kvs) with the rows folded INTO the block: a
 * block owns (split, kv head, row chunk), dequantizes each K key into registers
 * once and stages each V tile once, and scores / accumulates every row of the
 * chunk against them. The causal mask inside the draft rows falls out of the
 * per-row key range. Measured on the K80 the kernel is latency-bound, so the
 * host defaults to ONE row per block (5x the blocks beats one K/V read for 5
 * rows); each width is its own template instance so a narrow block keeps the
 * slots kernel's register budget (72 registers at 1 row, 120 at 5).
 *
 * BIT-EXACT PER ROW against the decode slots kernel at that row's position
 * (= one plain decode step), under this contract:
 *   - every row of a launch has the same split_len (the host groups rows by
 *     plan and launches each group separately; the grid width is the max
 *     n_splits over all rows so the partial layout and the combine are the
 *     slots ones);
 *   - per row, the key range of split s is [s*split_len, min(.., seq_len_r))
 *     exactly as the decode plan gives it, the q.k accumulation order, the
 *     16-lane shuffle tree, the online softmax, the PV order (rescale fused at
 *     j == 0) and every expression are the slots kernel's;
 *   - the fp16 sink / recent-window shadow is chosen PER ROW (a key can sit in
 *     row 0's window and outside row R-1's), so a tile that straddles the
 *     window edge of the chunk reloads K per row;
 *   - a row with no key in a split writes the empty partial (m=-inf, l=0,
 *     acc=0) the slots kernel writes for split >= n_splits.
 * A tile past row r's end is skipped for row r (the slots kernel never sees
 * it), so the trailing rows of the chunk cost nothing extra to the earlier ones.
 *
 * CONTRACT (host-enforced in fc_attn_flash_decode_rows_q4_ctx): head_dim ==
 * FC_ATTN_TILE_KEYS * FC_ATTN_DOT_LANES (256), row_bytes % 4 == 0, rows per
 * block <= FC_ATTN_ROWS_MAX, full attention (attn_win <= 0), no ring wrap.
 * Dynamic shared = fc_attn_flash_rows_shared_bytes() in cuda/fc_cuda.c.
 *
 * @status Speculative-verify path behind QWEN35_BATCH_VERIFY (Pantheon).
 * @issues None
 * @todo None
 */

/* Rows one block carries. 5 keeps the q set (5 x 8 heads x 256 fp32 = 40 KB)
 * inside sm_37's 48 KB per-block shared. MUST match fc_cuda.h. */
#define FC_ATTN_ROWS_MAX 5

/*
 * q:        [rows][n_head][head_dim] fp32, row r at absolute position
 *           pos_lo + (r - row_lo)
 * k_cache, v_cache, k_sink, v_sink, k_win, v_win: ONE session's ring (the
 *           single-user layout of fc_attn_flash_decode_q4)
 * m_out, l_out, acc_out: split partials, slots layout with slot = absolute row
 * pos_lo:   position of row row_lo
 * row_lo:   first absolute row of this launch
 * nrows:    rows in this launch
 * RPB:      rows per block, compile time (blockIdx.z selects rows
 *           [z*RPB, z*RPB + RPB)); one instance per width keeps the register
 *           set (acc[RPB][G]) and occupancy of a narrow verify at the slots
 *           kernel's, instead of always paying for FC_ATTN_ROWS_MAX rows
 * split_len: the common split length of these rows
 * Grid (launch_splits, n_head_kv, ceil(nrows / RPB)), block head_dim threads.
 */
template <int RPB, int PSM = 0>
__device__ __forceinline__ void fc_attn_rows_body(const float *q,
                                                 const unsigned char *k_cache,
                                                 const unsigned char *v_cache,
                                                 const __half *k_sink,
                                                 const __half *v_sink,
                                                 const __half *k_win,
                                                 const __half *v_win,
                                                 float *m_out, float *l_out,
                                                 float *acc_out, int kv_groups,
                                                 int head_dim, int max_seq,
                                                 int pos_lo, int row_lo,
                                                 int nrows,
                                                 int split_len, int sink,
                                                 int win, float scale) {
    const int split = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    const int G = kv_groups;
    const int n_head_kv = gridDim.y;
    const int stride = gridDim.x;
    const int row_bytes = fc_kvq4_row_bytes(head_dim);

    /* This block's rows: [r0, r0 + nr) of the launch. */
    const int r0 = blockIdx.z * RPB;
    int nr = nrows - r0;
    if (nr > RPB) nr = RPB;
    const int RG = nr * G;
    const int pos0 = pos_lo + r0;           /* cur_pos of the block's row 0 */
    const int k_lo = split * split_len;

    /* Row r's key end in this split, min(k_lo + split_len, pos0 + r + 1):
     * computed, not stored (an indexed array would land in local memory). A
     * row with k_lo >= its seq_len has no key here. The block's end is the
     * last row's. */
#define FC_ROWS_KHI(r) min(k_lo + split_len, pos0 + (r) + 1)
    const int k_hi = max(k_lo, FC_ROWS_KHI(nr - 1));

    extern __shared__ float shr[];
    float *q_s = shr;
    float *score_s = q_s + (size_t)RG * head_dim;
    float *p_s = score_s + RG * FC_ATTN_TILE_KEYS;
    float *m_s = p_s + RG * FC_ATTN_TILE_KEYS;
    float *l_s = m_s + RG;
    float *corr_s = l_s + RG;
    unsigned int *v_sh = (unsigned int *)(corr_s + RG);

    const size_t head_base = ((size_t)(row_lo + r0) * n_head_kv + kvh) * G;
    float acc[RPB][FC_ATTN_MAX_GROUPS];
#pragma unroll
    for (int r = 0; r < RPB; r++)
#pragma unroll
        for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
            acc[r][g] = 0.0f;

    if (k_hi > k_lo) {
        for (int rg = 0; rg < RG; rg++) {
            const int r = rg / G, g = rg % G;
            const size_t hq = ((size_t)(row_lo + r0 + r) * n_head_kv + kvh) * G + g;
            q_s[(size_t)rg * head_dim + tid] = q[hq * head_dim + tid];
        }
        if (tid < RG) {
            m_s[tid] = -__int_as_float(0x7f800000);
            l_s[tid] = 0.0f;
        }
        __syncthreads();

        const size_t hoff = (size_t)kvh;
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

        /* Keys whose shadow choice differs across the block's rows: fp16
         * window for row 0, q4 body for the last row. */
        int mix_lo = pos0 - win + 1;
        if (mix_lo < sink) mix_lo = sink;
        const int mix_hi = pos0 + nr - 1 - win;

        for (int j0 = k_lo; j0 < k_hi; j0 += FC_ATTN_TILE_KEYS) {
            int tk = k_hi - j0;
            if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;
            const int mixed = win > 0 && nr > 1 && mix_hi >= mix_lo &&
                              j0 + tk - 1 >= mix_lo && j0 <= mix_hi;

            float kreg[FC_ATTN_KREG_MAX_CHUNK];
#pragma unroll
            for (int r = 0; r < RPB; r++) {
                if (r >= nr) break;
                int tk_r = FC_ROWS_KHI(r) - j0;
                if (tk_r > FC_ATTN_TILE_KEYS) tk_r = FC_ATTN_TILE_KEYS;
                const int cur_pos = pos0 + r;
                if ((r == 0 || mixed) && ky < tk) {
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

                float part[FC_ATTN_MAX_GROUPS];
#pragma unroll
                for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                    part[g] = 0.0f;
                if (ky < tk_r) {
#pragma unroll
                    for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++) {
                        if (i < chunk) {
                            const int e = lane + i * FC_ATTN_DOT_LANES;
                            const float kve = kreg[i];
#pragma unroll
                            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)
                                if (g < G)
                                    part[g] += q_s[(size_t)(r * G + g) * head_dim + e] * kve;
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
                        if (lane == 0 && ky < tk_r)
                            score_s[(r * G + g) * FC_ATTN_TILE_KEYS + ky] = p * scale;
                    }
                }
            }
            __syncthreads();

            /* Online softmax: thread tid owns (row tid / G, group tid % G). PSM: the
             * tile's expf calls run on (rg, key) threads in parallel and the owner then
             * sums them in key order — the same values, the same additions. */
            if (PSM) {
                /* [RG] this tile's new running max, past the V tile words */
                float *mn_s = (float *)(v_sh + FC_ATTN_VS_TILE_WORDS);
                if (tid < RG) {
                    int tk_r = FC_ROWS_KHI(tid / G) - j0;
                    if (tk_r > FC_ATTN_TILE_KEYS) tk_r = FC_ATTN_TILE_KEYS;
                    if (tk_r > 0) {
                        const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
                        float m_new = m_s[tid];
                        for (int j = 0; j < tk_r; j++)
                            m_new = fmaxf(m_new, sc[j]);
                        mn_s[tid] = m_new;
                        corr_s[tid] = expf(m_s[tid] - m_new);
                    }
                }
                __syncthreads();
                for (int idx = tid; idx < RG * FC_ATTN_TILE_KEYS; idx += blockDim.x) {
                    const int rg = idx / FC_ATTN_TILE_KEYS, j = idx % FC_ATTN_TILE_KEYS;
                    int tk_r = FC_ROWS_KHI(rg / G) - j0;
                    if (tk_r > FC_ATTN_TILE_KEYS) tk_r = FC_ATTN_TILE_KEYS;
                    if (j < tk_r)
                        p_s[(size_t)j * RG + rg] =
                            expf(score_s[rg * FC_ATTN_TILE_KEYS + j] - mn_s[rg]);
                }
                __syncthreads();
                if (tid < RG) {
                    int tk_r = FC_ROWS_KHI(tid / G) - j0;
                    if (tk_r > FC_ATTN_TILE_KEYS) tk_r = FC_ATTN_TILE_KEYS;
                    if (tk_r > 0) {
                        float sum = 0.0f;
                        for (int j = 0; j < tk_r; j++)
                            sum += p_s[(size_t)j * RG + tid];
                        const float corr = corr_s[tid];
                        l_s[tid] = l_s[tid] * corr + sum;
                        m_s[tid] = mn_s[tid];
                    }
                }
            } else if (tid < RG) {
                int tk_r = FC_ROWS_KHI(tid / G) - j0;
                if (tk_r > FC_ATTN_TILE_KEYS) tk_r = FC_ATTN_TILE_KEYS;
                if (tk_r > 0) {
                    const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;
                    float m_new = m_s[tid];
                    for (int j = 0; j < tk_r; j++)
                        m_new = fmaxf(m_new, sc[j]);
                    float corr = expf(m_s[tid] - m_new);
                    float sum = 0.0f;
                    for (int j = 0; j < tk_r; j++) {
                        float pj = expf(sc[j] - m_new);
                        p_s[(size_t)j * RG + tid] = pj;
                        sum += pj;
                    }
                    l_s[tid] = l_s[tid] * corr + sum;
                    m_s[tid] = m_new;
                    corr_s[tid] = corr;
                }
            }

            /* V tile as raw q4 words, once for every row of the block. */
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
                const int vp = j0 + j;
                float vv_body;
                if (vp < sink) {
                    vv_body = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);
                } else {
                    const unsigned char *bp =
                        (const unsigned char *)v_sh + j * row_bytes + vqoff;
                    const unsigned short d_bits = (unsigned short)(bp[0] | (bp[1] << 8));
                    const float d = __half2float(__ushort_as_half(d_bits));
                    const int code = (bp[vbyte] >> vshift) & 0xF;
                    vv_body = (float)(code - 8) * d;
                }
                /* Row 0 has the lowest window edge: if no row reads this key
                 * from the window, row 0 does not either. */
                float vv_win = 0.0f;
                if (vp >= sink && win > 0 && vp > pos0 - win)
                    vv_win = __half2float(v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]);
#pragma unroll
                for (int r = 0; r < RPB; r++) {
                    if (r >= nr) break;
                    if (j >= FC_ROWS_KHI(r) - j0) continue;
                    const int cur_pos = pos0 + r;
                    const float vv = (vp >= sink && win > 0 && vp > cur_pos - win)
                                         ? vv_win : vv_body;
                    const float *pj_s = p_s + (size_t)j * RG + r * G;
                    const float *cr_s = corr_s + r * G;
#pragma unroll
                    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
                        if (g < G) {
                            if (j == 0)
                                acc[r][g] *= cr_s[g];
                            acc[r][g] += pj_s[g] * vv;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int r = 0; r < RPB; r++) {
        if (r >= nr) break;
        const int active = FC_ROWS_KHI(r) > k_lo;
#pragma unroll
        for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {
            if (g < G) {
                size_t po = (head_base + (size_t)r * n_head_kv * G + g) * stride + split;
                acc_out[po * head_dim + tid] = active ? acc[r][g] : 0.0f;
                if (tid == 0) {
                    m_out[po] = active ? m_s[r * G + g] : -__int_as_float(0x7f800000);
                    l_out[po] = active ? l_s[r * G + g] : 0.0f;
                }
            }
        }
    }
}

#undef FC_ROWS_KHI

extern "C" {

/* One kernel per rows-per-block width; arguments as fc_attn_rows_body. */
#define FC_ATTN_ROWS_KERNEL(N)                                                \
__global__ void fc_attn_flash_decode_rows_q4_kvs_r##N(                        \
        const float *q, const unsigned char *k_cache,                         \
        const unsigned char *v_cache, const __half *k_sink,                   \
        const __half *v_sink, const __half *k_win, const __half *v_win,       \
        float *m_out, float *l_out, float *acc_out, int kv_groups,            \
        int head_dim, int max_seq, int pos_lo, int row_lo, int nrows,         \
        int split_len, int sink, int win, float scale) {                      \
    fc_attn_rows_body<N>(q, k_cache, v_cache, k_sink, v_sink, k_win, v_win,   \
                         m_out, l_out, acc_out, kv_groups, head_dim, max_seq, \
                         pos_lo, row_lo, nrows, split_len, sink, win, scale); \
}
FC_ATTN_ROWS_KERNEL(1)
FC_ATTN_ROWS_KERNEL(2)
FC_ATTN_ROWS_KERNEL(3)
FC_ATTN_ROWS_KERNEL(4)
FC_ATTN_ROWS_KERNEL(5)
#undef FC_ATTN_ROWS_KERNEL

/* Device-position form (CUDA-graph capture): one row per block, row z at
 * pos[0] + z, each block derives ITS row's split plan from the device
 * position with the host plan's formula (fc_attn_rows_plan). Grid
 * (launch_splits, n_head_kv, nrows); a split past the row's n_splits writes
 * the empty partial, exactly as the slots kernel's dead blocks do. Same body,
 * same bytes as fc_attn_flash_decode_rows_q4_kvs_r1. */
#define FC_ATTN_ROWS_DEV_KERNEL(NAME, PSM)                                   \
__global__ void NAME(                                                         \
        const float *q, const unsigned char *k_cache,                         \
        const unsigned char *v_cache, const __half *k_sink,                   \
        const __half *v_sink, const __half *k_win, const __half *v_win,       \
        float *m_out, float *l_out, float *acc_out, int kv_groups,            \
        int head_dim, int max_seq, const int *pos, int kps, int splits_cap,   \
        int sink, int win, float scale) {                                     \
    const int pos_lo = pos[0];                                                \
    const int seq = pos_lo + (int)blockIdx.z + 1;                             \
    int ns = (seq + kps - 1) / kps;                                           \
    if (ns < 1) ns = 1;                                                       \
    if (ns > splits_cap) ns = splits_cap;                                     \
    int sl = (seq + ns - 1) / ns;                                             \
    sl = (sl + 15) & ~15;                                                     \
    fc_attn_rows_body<1, PSM>(q, k_cache, v_cache, k_sink, v_sink, k_win,     \
                              v_win, m_out, l_out, acc_out, kv_groups,        \
                              head_dim, max_seq, pos_lo, 0, (int)gridDim.z,   \
                              sl, sink, win, scale);                          \
}
FC_ATTN_ROWS_DEV_KERNEL(fc_attn_flash_decode_rows_q4_dev, 0)
/* The same with the parallel-expf softmax (one extra RG floats of shared). */
FC_ATTN_ROWS_DEV_KERNEL(fc_attn_flash_decode_rows_q4_dev_ps, 1)
#undef FC_ATTN_ROWS_DEV_KERNEL

} /* extern "C" */
