/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * topk_logits.cu — device-side exact top-k over the LM-head logits
 *
 * Target: sm_37 (K80, Kepler)
 * Compile: nvcc -ptx -arch=sm_37 topk_logits.cu -o topk_logits.ptx
 *
 * Why: on a small-BAR K80 host the host link runs at ~0.4 GB/s, so copying
 * the full 248k-vocab logits (1 MB) D2H costs ~4 ms per decoded token — the
 * single largest non-compute slice of the token budget. Greedy decode only
 * needs the argmax and the serving sampler (top_k=40) only needs the top 40
 * (id, logit) pairs, so we reduce on the die and copy back k pairs instead.
 *
 * Exactness: the result is the first k elements of the strict total order
 *     (val_a > val_b) || (val_a == val_b && idx_a < idx_b)
 * over the values above -inf (NaN and -inf never win), i.e. exactly what k
 * rounds of the host's serial strict-'>' argmax scan return (lowest index
 * wins ties) with previous winners removed. Rounds past the last value above
 * -inf report (-inf, FC_TOPK_NO_WINNER). No float arithmetic is performed on
 * the logits — only compares — so the values are the bit-exact LM-head
 * outputs, and the logits are only read, never written.
 *
 * Fast path (driven by fc_topk_logits_slots_ctx in fc_cuda.c), no rounds:
 *   1. fc_topk_blockmax — each block takes one chunk of a row and emits the
 *      chunk's best (val, idx).
 *   2. fc_topk_filter — every block ranks the row's chunk maxima and takes
 *      the k-th best as a threshold T (at least k elements are >= T, so the
 *      row's top-k all are), then appends each element >= T to a per-row
 *      candidate list (warp-aggregated atomics, any order).
 *   3. fc_topk_rank_select — one block per row ranks the candidates (rank =
 *      how many candidates beat it; distinct under the total order) and
 *      writes rank r < k to slot r. A row whose candidates overflow
 *      FC_TOPK_CAND_CAP (mass ties at T) reports status 1.
 * Fallback for an overflowed row (or more than FC_TOPK_FAST_MAXBLK chunks):
 * a tree of fc_topk_select launches — each block emits its chunk's top-k by
 * k rounds of block argmax, later levels merge until one block per row is
 * left. Every element of a row's top-k is in its own chunk's top-k.
 *
 * (Was k rounds of grid-wide argmax with -inf clobber/restore: 2k+1 launches
 * that each re-read the whole row, 1.84 ms for k=40 over 248320 on a K80.)
 *
 * PTX symbols emitted (for embedding into ../cuda/fc_cuda_ptx.h):
 *   FC_PTX_TOPK_LOGITS -> kernels
 *     fc_topk_blockmax, fc_topk_filter, fc_topk_rank_select, fc_topk_select
 *
 * Kepler notes: pure fp32, __shfl_down (the non-.sync variant) warp
 * reductions, two __syncthreads per selection round.
 */

extern "C" {

#define FC_TOPK_NEG_INF __int_as_float(0xff800000)

/* Index a round reports when no value beats -inf: every remaining value is
 * NaN or -inf (NaN never wins a compare). It is not a position; the host sees
 * it in out_idx and rejects the row. */
#define FC_TOPK_NO_WINNER 0x7fffffff

/* Elements per thread, compile-time max (registers; the host picks e <= this,
 * fc_cuda.c FC_TOPK_SEL_E). */
#define FC_TOPK_SEL_E 16

/* Fast-path limits (mirrored in fc_cuda.c): chunk maxima ranked per row in
 * shared memory, and candidates per row. */
#define FC_TOPK_FAST_MAXBLK 1024
#define FC_TOPK_CAND_CAP    2048

/* Comparator: does candidate (av, ai) beat incumbent (bv, bi)?
 * Strictly greater value wins; equal value -> lower index wins. */
__device__ __forceinline__ int fc_topk_beats(float av, int ai,
                                             float bv, int bi)
{
    return (av > bv) || (av == bv && ai < bi);
}

/* Warp argmax under fc_topk_beats; lane 0 ends with the warp winner. */
__device__ __forceinline__ void fc_topk_warp_best(float *v, int *i)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down(*v, off);
        int   oi = __shfl_down(*i, off);
        if (fc_topk_beats(ov, oi, *v, *i)) { *v = ov; *i = oi; }
    }
}

/* Load element p of a row as a candidate: NaN / -inf become (-inf, NO_WINNER),
 * which never wins. */
__device__ __forceinline__ void fc_topk_load(const float *row, int p, int m,
                                             float *v, int *i)
{
    float a = FC_TOPK_NEG_INF;
    int ai = FC_TOPK_NO_WINNER;
    if (p < m) {
        a = row[p];
        ai = p;
        if (!(a > FC_TOPK_NEG_INF)) { a = FC_TOPK_NEG_INF; ai = FC_TOPK_NO_WINNER; }
    }
    *v = a;
    *i = ai;
}

/*
 * Fast path 1: best (val, idx) of each chunk.
 *
 * x:      [nslots][n] fp32 logits
 * e:      elements per thread; chunk = blockDim.x * e
 * bv/bi:  [nslots][gridDim.x] chunk maxima (output)
 * count:  [nslots] candidate counters, zeroed here for stage 2
 *
 * Grid (ceil(n / chunk), nslots); blockDim.x a multiple of 32, <= 1024.
 */
__global__ void fc_topk_blockmax(const float *x, int n, int e,
                                 float *bv, int *bi, int *count)
{
    __shared__ float s_v[32];
    __shared__ int   s_i[32];
    const float *row = x + (size_t)blockIdx.y * n;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int base = blockIdx.x * blockDim.x * e + tid;

    float wv = FC_TOPK_NEG_INF;
    int wi = FC_TOPK_NO_WINNER;
    for (int j = 0; j < e; j++) {
        float a;
        int ai;
        fc_topk_load(row, base + j * blockDim.x, n, &a, &ai);
        if (fc_topk_beats(a, ai, wv, wi)) { wv = a; wi = ai; }
    }
    fc_topk_warp_best(&wv, &wi);
    if (lane == 0) { s_v[warp] = wv; s_i[warp] = wi; }
    __syncthreads();
    if (warp == 0) {
        int nwarp = blockDim.x >> 5;
        wv = lane < nwarp ? s_v[lane] : FC_TOPK_NEG_INF;
        wi = lane < nwarp ? s_i[lane] : FC_TOPK_NO_WINNER;
        fc_topk_warp_best(&wv, &wi);
        if (lane == 0) {
            bv[(size_t)blockIdx.y * gridDim.x + blockIdx.x] = wv;
            bi[(size_t)blockIdx.y * gridDim.x + blockIdx.x] = wi;
            if (blockIdx.x == 0)
                count[blockIdx.y] = 0;
        }
    }
}

/*
 * Fast path 2: threshold T = k-th best chunk maximum, then append every
 * element of the chunk that T does not beat to the row's candidate list.
 * If fewer than k chunks have a value above -inf, T is (-inf, NO_WINNER) and
 * every value above -inf is appended (stage 3 then overflows or pads).
 *
 * x, n, e:  as fc_topk_blockmax (same grid, same chunks)
 * bv/bi:    [nslots][gridDim.x] chunk maxima from stage 1
 * cv/ci:    [nslots][FC_TOPK_CAND_CAP] candidates (output, any order)
 * count:    [nslots] appended count (may exceed the cap: overflow)
 *
 * Shared memory: gridDim.x <= FC_TOPK_FAST_MAXBLK maxima.
 */
__global__ void fc_topk_filter(const float *x, int n, int e, const float *bv,
                               const int *bi, int k, float *cv, int *ci,
                               int *count)
{
    __shared__ float s_bv[FC_TOPK_FAST_MAXBLK];
    __shared__ int   s_bi[FC_TOPK_FAST_MAXBLK];
    __shared__ float s_tv;
    __shared__ int   s_ti;
    const int u = blockIdx.y, nblk = gridDim.x;
    const float *row = x + (size_t)u * n;
    int tid = threadIdx.x, lane = tid & 31;

    if (tid == 0) { s_tv = FC_TOPK_NEG_INF; s_ti = FC_TOPK_NO_WINNER; }
    for (int t = tid; t < nblk; t += blockDim.x) {
        s_bv[t] = bv[(size_t)u * nblk + t];
        s_bi[t] = bi[(size_t)u * nblk + t];
    }
    __syncthreads();
    for (int t = tid; t < nblk; t += blockDim.x) {
        float mv = s_bv[t];
        int mi = s_bi[t];
        if (mi == FC_TOPK_NO_WINNER)
            continue;
        int rank = 0;
        for (int j = 0; j < nblk; j++)
            rank += fc_topk_beats(s_bv[j], s_bi[j], mv, mi);
        if (rank == k - 1) { s_tv = mv; s_ti = mi; }
    }
    __syncthreads();
    const float tv = s_tv;
    const int ti = s_ti;

    int base = blockIdx.x * blockDim.x * e + tid;
    float *cvr = cv + (size_t)u * FC_TOPK_CAND_CAP;
    int *cir = ci + (size_t)u * FC_TOPK_CAND_CAP;
    for (int j = 0; j < e; j++) {
        float a;
        int ai;
        fc_topk_load(row, base + j * blockDim.x, n, &a, &ai);
        int keep = ai != FC_TOPK_NO_WINNER && !fc_topk_beats(tv, ti, a, ai);
        unsigned int mask = __ballot(keep);
        if (mask == 0)
            continue;
        int pos = 0;
        if (lane == __ffs(mask) - 1)
            pos = atomicAdd(&count[u], __popc(mask));
        pos = __shfl(pos, __ffs(mask) - 1) + __popc(mask & ((1u << lane) - 1u));
        if (keep && pos < FC_TOPK_CAND_CAP) { cvr[pos] = a; cir[pos] = ai; }
    }
}

/*
 * Fast path 3: rank-select the row's candidates. Candidate c goes to slot
 * rank(c) = #candidates that beat it, if that is < k; slots past the
 * candidate count get (-inf, NO_WINNER). status[u] = 1 (nothing written) if
 * the list overflowed FC_TOPK_CAND_CAP, else 0.
 *
 * Grid (1, nslots); any blockDim.x. Shared memory: the candidates.
 */
__global__ void fc_topk_rank_select(const float *cv, const int *ci,
                                    const int *count, int k, float *oval,
                                    int *oidx, int *status)
{
    __shared__ float s_v[FC_TOPK_CAND_CAP];
    __shared__ int   s_i[FC_TOPK_CAND_CAP];
    const int u = blockIdx.y;
    const int c = count[u];
    int tid = threadIdx.x;
    if (c > FC_TOPK_CAND_CAP) {
        if (tid == 0)
            status[u] = 1;
        return;
    }
    for (int j = tid; j < c; j += blockDim.x) {
        s_v[j] = cv[(size_t)u * FC_TOPK_CAND_CAP + j];
        s_i[j] = ci[(size_t)u * FC_TOPK_CAND_CAP + j];
    }
    __syncthreads();
    float *ov = oval + (size_t)u * k;
    int *oi = oidx + (size_t)u * k;
    for (int j = tid; j < c; j += blockDim.x) {
        float a = s_v[j];
        int ai = s_i[j];
        int rank = 0;
        for (int q = 0; q < c; q++)
            rank += fc_topk_beats(s_v[q], s_i[q], a, ai);
        if (rank < k) { ov[rank] = a; oi[rank] = ai; }
    }
    for (int r = c + tid; r < k; r += blockDim.x) {
        ov[r] = FC_TOPK_NEG_INF;
        oi[r] = FC_TOPK_NO_WINNER;
    }
    if (tid == 0)
        status[u] = 0;
}

/*
 * One level of the selection tree: block (b, u) emits the top-k of row u's
 * chunk b, in order, padded with (-inf, NO_WINNER).
 *
 * val:  [nslots][m] fp32 — logits (level 0) or candidate values
 * idx:  [nslots][m] int32 — candidate indices, or NULL (level 0: index = p)
 * e:    elements per thread (<= FC_TOPK_SEL_E); chunk = blockDim.x * e
 * oval: [nslots][gridDim.x][k] fp32 — winners per chunk (output)
 * oidx: [nslots][gridDim.x][k] int32
 *
 * Grid (ceil(m / chunk), nslots); blockDim.x a multiple of 32, <= 1024.
 * Each thread holds its e elements in registers; a round is a warp argmax,
 * a block argmax over the warp winners, and the owner thread dropping the
 * winner and rescanning its own e elements.
 */
__global__ void fc_topk_select(const float *val, const int *idx, int m,
                               int e, int k, float *oval, int *oidx)
{
    __shared__ float s_v[32];
    __shared__ int   s_i[32];
    __shared__ float s_wv;
    __shared__ int   s_wi;

    const float *vr = val + (size_t)blockIdx.y * m;
    const int   *ir = idx ? idx + (size_t)blockIdx.y * m : 0;
    size_t ob = ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * k;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int nwarp = blockDim.x >> 5;
    int base = blockIdx.x * blockDim.x * e + tid;

    float v[FC_TOPK_SEL_E];
    int   id[FC_TOPK_SEL_E];
    float bv = FC_TOPK_NEG_INF;
    int   bi = FC_TOPK_NO_WINNER;
#pragma unroll
    for (int j = 0; j < FC_TOPK_SEL_E; j++) {
        int p = base + j * blockDim.x;
        float a = FC_TOPK_NEG_INF;
        int ai = FC_TOPK_NO_WINNER;
        if (j < e && p < m) {
            a = vr[p];
            ai = ir ? ir[p] : p;
            if (!(a > FC_TOPK_NEG_INF)) {   /* NaN / -inf never win */
                a = FC_TOPK_NEG_INF;
                ai = FC_TOPK_NO_WINNER;
            }
        }
        v[j] = a;
        id[j] = ai;
        if (fc_topk_beats(a, ai, bv, bi)) { bv = a; bi = ai; }
    }

    int r = 0;
    for (; r < k; r++) {
        float wv = bv;
        int wi = bi;
        fc_topk_warp_best(&wv, &wi);
        if (lane == 0) { s_v[warp] = wv; s_i[warp] = wi; }
        __syncthreads();
        if (warp == 0) {
            wv = lane < nwarp ? s_v[lane] : FC_TOPK_NEG_INF;
            wi = lane < nwarp ? s_i[lane] : FC_TOPK_NO_WINNER;
            fc_topk_warp_best(&wv, &wi);
            if (lane == 0) { s_wv = wv; s_wi = wi; }
        }
        __syncthreads();
        wv = s_wv;
        wi = s_wi;
        if (tid == 0) { oval[ob + r] = wv; oidx[ob + r] = wi; }
        if (wi == FC_TOPK_NO_WINNER)
            break;                          /* uniform: wi came from shared */
        if (bi == wi) {                     /* indices are unique: one owner */
            bv = FC_TOPK_NEG_INF;
            bi = FC_TOPK_NO_WINNER;
#pragma unroll
            for (int j = 0; j < FC_TOPK_SEL_E; j++) {
                if (id[j] == wi) { v[j] = FC_TOPK_NEG_INF; id[j] = FC_TOPK_NO_WINNER; }
                if (fc_topk_beats(v[j], id[j], bv, bi)) { bv = v[j]; bi = id[j]; }
            }
        }
    }
    for (int rr = r + tid; rr < k; rr += blockDim.x) {
        oval[ob + rr] = FC_TOPK_NEG_INF;
        oidx[ob + rr] = FC_TOPK_NO_WINNER;
    }
}

} /* extern "C" */
