/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * moe_routing.cu — Qwen3.5-MoE routing kernels for Foundry
 *
 * Target: sm_37 (K80, Kepler)
 * Compile: nvcc -ptx -arch=sm_37 moe_routing.cu -o moe_routing.ptx
 *
 * The MoE FFN splits into three GPU-side pieces. The heavy per-expert SwiGLU
 * matmuls are NOT here — they are the existing quantized matvec kernels
 * (fc_matvec_q3k + fc_silu + elementwise mul), launched per selected expert by
 * the host. This file owns the cheap glue that surrounds them:
 *
 *   1. fc_moe_router_topk   — softmax(router_logits) over n_expert, select
 *                             top-k by probability, norm_topk_prob renormalize,
 *                             apply expert_weights_scale. Emits top_idx[k] and
 *                             top_wt[k] for the host to drive expert dispatch.
 *   2. fc_moe_weighted_add  — accumulate one expert's output into moe_out with
 *                             its routed weight (the scatter/sum step).
 *   3. fc_moe_shared_gate   — sigmoid scalar gate applied to the shared-expert
 *                             output (always-on branch).
 *
 * Mirrors, value-for-value, the CPU reference:
 *   Pantheon/src/models/qwen35moe_moe.c
 *     qwen35moe_router_softmax   (lines 47-61)
 *     qwen35moe_select_topk      (lines 65-89)
 *     norm_topk_prob + scale     (lines 131-136)
 *     weighted expert sum        (lines 138-147)
 *     shared-expert sigmoid gate (lines 179-185)
 *
 * Config: n_expert 256, n_expert_used 8, n_embd 2048, expert_ff 512.
 *
 * Head-broadcast trap (qwen35moe.md watch-list): MoE routing does NO GQA head
 * broadcasting, so the interleaved-vs-blocked tiling bug does not apply in this
 * file. Noted here so the integration step doesn't have to re-derive it.
 *
 * PTX symbols emitted (for embedding into ../cuda/fc_cuda_ptx.h):
 *   FC_PTX_MOE_ROUTING -> kernels
 *     fc_moe_router_topk, fc_moe_router_topk_nx, fc_moe_router_topk_rank,
 *     fc_moe_weighted_add, fc_moe_shared_gate
 *
 * Kepler notes: pure fp32; single-block reductions via shared memory (no
 * __shfl). Top-k selection is a serial selection sort on thread 0 — k=8 and
 * n_expert=256, so it is cheap and exactly reproduces the CPU tie-break order.
 */

extern "C" {

/* Hard cap on top-k — sized for the on-block selection scratch. Matches the
 * CPU PN_QWEN35MOE_MAX_TOPK so behaviour is identical for any configured k. */
#define FC_MOE_MAX_TOPK 32

/*
 * Router: softmax over n_expert logits, top-k select, norm_topk_prob, scale.
 *
 * logits:  [n_expert] fp32 router logits (raw ffn_gate_inp . x). Overwritten
 *          in place with the softmax probabilities.
 * top_idx: [k] int32 — selected expert indices (output)
 * top_wt:  [k] fp32 — final routed weights (output), already renormalized and
 *          multiplied by expert_weights_scale.
 * n_expert:             number of experts (256)
 * k:                    n_expert_used (8); must be <= FC_MOE_MAX_TOPK
 * expert_weights_scale: cfg->expert_weights_scale (1.0 unless GGUF overrides)
 *
 * Launched with 1 block, blockDim.x threads (use 256).
 * Shared memory: blockDim.x * sizeof(float) * 2 (max reduction | sum reduction).
 *
 * Softmax + selection exactly follow the CPU path: selection sort picks the
 * highest probability each round, skipping already-chosen experts, so ties
 * resolve to the lowest index — identical to qwen35moe_select_topk.
 */
__global__ void fc_moe_router_topk(float *logits, int *top_idx, float *top_wt,
                                   int n_expert, int k,
                                   float expert_weights_scale) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int threads = blockDim.x;
    float *smax = sdata;
    float *ssum = sdata + threads;

    /* Softmax pass 1: max */
    float local_max = -1e30f;
    for (int e = tid; e < n_expert; e += threads) {
        float v = logits[e];
        if (v > local_max) local_max = v;
    }
    smax[tid] = local_max;
    __syncthreads();
    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid])
            smax[tid] = smax[tid + s];
        __syncthreads();
    }
    float row_max = smax[0];

    /* Softmax pass 2: exp(x - max) in place + sum */
    float local_sum = 0.0f;
    for (int e = tid; e < n_expert; e += threads) {
        float v = expf(logits[e] - row_max);
        logits[e] = v;
        local_sum += v;
    }
    ssum[tid] = local_sum;
    __syncthreads();
    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s)
            ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / ssum[0];

    /* Softmax pass 3: normalize to probabilities in place */
    for (int e = tid; e < n_expert; e += threads)
        logits[e] *= inv_sum;
    __syncthreads();

    /* Top-k via k rounds of PARALLEL argmax (tree reduction, ties -> lowest
     * index, identical to the CPU first-seen-highest tie-break). The original
     * serial selection sort on thread 0 measured 3.3 ms per launch (~134 ms
     * per token across 40 layers) — one GPU thread scanning k*n_expert*k; the
     * parallel version is microseconds. Selected entries are knocked out of
     * logits[] in place with -1 (logits is per-token scratch; nothing reads
     * the probabilities after this kernel). */
    __shared__ float rv[256];
    __shared__ int ri[256];
    __shared__ float swt_sum;
    if (tid == 0)
        swt_sum = 0.0f;

    for (int sel = 0; sel < k; sel++) {
        /* Strided local argmax (ascending e + strict > => lowest index wins
         * within a thread), then tree argmax preferring the lower index on
         * equal values. */
        float best_val = -1.0f;
        int best = -1;
        for (int e = tid; e < n_expert; e += threads) {
            float v = logits[e];
            if (v > best_val) {
                best_val = v;
                best = e;
            }
        }
        rv[tid] = best_val;
        ri[tid] = best;
        __syncthreads();
        for (int s = threads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (rv[tid + s] > rv[tid] ||
                    (rv[tid + s] == rv[tid] && ri[tid + s] >= 0 &&
                     (ri[tid] < 0 || ri[tid + s] < ri[tid]))) {
                    rv[tid] = rv[tid + s];
                    ri[tid] = ri[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            top_idx[sel] = ri[0];
            top_wt[sel] = rv[0];
            swt_sum += rv[0];
            if (ri[0] >= 0)
                logits[ri[0]] = -1.0f;   /* knock out for the next round */
        }
        __syncthreads();
    }

    /* norm_topk_prob: renormalize selected weights to sum to 1, then apply
     * expert_weights_scale. */
    if (tid == 0) {
        float wt_sum = swt_sum;
        float inv_wt = (wt_sum > 0.0f) ? (1.0f / wt_sum) : 0.0f;
        float scale = expert_weights_scale * inv_wt;
        for (int s = 0; s < k; s++)
            top_wt[s] *= scale;
    }
}

/*
 * Batched (prefill) router: one token per BLOCK (blockIdx.x = t), each
 * block running the exact fc_moe_router_topk body on its token's slice:
 * logits [n_tok][n_expert] (softmax'd/knocked-out in place),
 * top_idx [n_tok][k], top_wt [n_tok][k]. Same blockDim as the single-token
 * launch -> same reduction trees -> bit-exact per token.
 */
__global__ void fc_moe_router_topk_nx(float *logits, int *top_idx,
                                      float *top_wt, int n_expert, int k,
                                      float expert_weights_scale) {
    extern __shared__ float sdata[];

    int tok = blockIdx.x;
    float *lg = logits + (size_t)tok * n_expert;
    int *ti = top_idx + (size_t)tok * k;
    float *tw = top_wt + (size_t)tok * k;

    int tid = threadIdx.x;
    int threads = blockDim.x;
    float *smax = sdata;
    float *ssum = sdata + threads;

    /* Softmax pass 1: max */
    float local_max = -1e30f;
    for (int e = tid; e < n_expert; e += threads) {
        float v = lg[e];
        if (v > local_max) local_max = v;
    }
    smax[tid] = local_max;
    __syncthreads();
    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid])
            smax[tid] = smax[tid + s];
        __syncthreads();
    }
    float row_max = smax[0];

    /* Softmax pass 2: exp(x - max) in place + sum */
    float local_sum = 0.0f;
    for (int e = tid; e < n_expert; e += threads) {
        float v = expf(lg[e] - row_max);
        lg[e] = v;
        local_sum += v;
    }
    ssum[tid] = local_sum;
    __syncthreads();
    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s)
            ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / ssum[0];

    /* Softmax pass 3: normalize to probabilities in place */
    for (int e = tid; e < n_expert; e += threads)
        lg[e] *= inv_sum;
    __syncthreads();

    /* Top-k via k rounds of parallel argmax (ties -> lowest index). */
    __shared__ float rv[256];
    __shared__ int ri[256];
    __shared__ float swt_sum;
    if (tid == 0)
        swt_sum = 0.0f;

    for (int sel = 0; sel < k; sel++) {
        float best_val = -1.0f;
        int best = -1;
        for (int e = tid; e < n_expert; e += threads) {
            float v = lg[e];
            if (v > best_val) {
                best_val = v;
                best = e;
            }
        }
        rv[tid] = best_val;
        ri[tid] = best;
        __syncthreads();
        for (int s = threads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (rv[tid + s] > rv[tid] ||
                    (rv[tid + s] == rv[tid] && ri[tid + s] >= 0 &&
                     (ri[tid] < 0 || ri[tid + s] < ri[tid]))) {
                    rv[tid] = rv[tid + s];
                    ri[tid] = ri[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            ti[sel] = ri[0];
            tw[sel] = rv[0];
            swt_sum += rv[0];
            if (ri[0] >= 0)
                lg[ri[0]] = -1.0f;
        }
        __syncthreads();
    }

    /* norm_topk_prob + expert_weights_scale. */
    if (tid == 0) {
        float wt_sum = swt_sum;
        float inv_wt = (wt_sum > 0.0f) ? (1.0f / wt_sum) : 0.0f;
        float scale = expert_weights_scale * inv_wt;
        for (int s = 0; s < k; s++)
            tw[s] *= scale;
    }
}

/*
 * Rank router: fc_moe_router_topk_nx with no selection rounds. One block per
 * token (blockIdx.x), blockDim.x = max(32, n_expert), thread t = expert t.
 *
 * Bit-exact to fc_moe_router_topk(_nx) launched as the host launches them
 * (blockDim = n_expert) for n_expert a power of two in [1, 256]
 * (FC_MOE_RANK_MAX_EXPERT); the host only picks it then.
 *   - max: order-free (strict '>' never adopts NaN), same -1e30f floor.
 *   - sum: the same tree, ssum[t] + ssum[t + s] for s = n/2 .. 1: shared
 *     memory for s >= 32, the same __shfl_down steps below; threads past
 *     n_expert (n < 32) add +0.0f, which is exact on exp() outputs.
 *   - selection: the block kernel's k rounds of argmax return the experts in
 *     the strict total order (prob desc, index asc), NaN never chosen. Here
 *     the k-th best warp maximum bounds the top-k from below; each expert at
 *     or above it is ranked (how many beat it) among those, rank r < k fills
 *     slot r, and slots past the non-NaN count get (-1.0f, -1) as a no-winner
 *     round.
 *   - weights: summed on one thread in slot order, as the block kernel does.
 *   - logits are left as the block kernel leaves them: probs, -1 at the
 *     chosen experts.
 * Was 53 us per launch on a K80 (8 rounds of 9-step __syncthreads trees).
 */
#define FC_MOE_RANK_MAX_EXPERT 256

/* The block router's tree comparator: does (av, ai) beat (bv, bi)? Index -1
 * is a no-winner candidate (value -1.0f). A strict total order on the
 * candidates it sees (NaN is never one). */
__device__ __forceinline__ int fc_moe_router_beats(float av, int ai,
                                                   float bv, int bi) {
    return av > bv || (av == bv && ai >= 0 && (bi < 0 || ai < bi));
}

__global__ void fc_moe_router_topk_rank(float *logits, int *top_idx,
                                        float *top_wt, int n_expert, int k,
                                        float expert_weights_scale) {
    __shared__ float sp[FC_MOE_RANK_MAX_EXPERT];
    __shared__ int ssi[FC_MOE_RANK_MAX_EXPERT];
    __shared__ float swv[8];
    __shared__ int swi[8];
    __shared__ float st_v;
    __shared__ int st_i;
    __shared__ int s_nsurv;
    __shared__ float selv[FC_MOE_MAX_TOPK];
    __shared__ int seli[FC_MOE_MAX_TOPK];
    __shared__ float sbc[2];

    int tok = blockIdx.x;
    float *lg = logits + (size_t)tok * n_expert;
    int *ti = top_idx + (size_t)tok * k;
    float *tw = top_wt + (size_t)tok * k;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int nwarp = blockDim.x >> 5;
    int live = tid < n_expert;

    float x = live ? lg[tid] : 0.0f;
    if (tid < k) { selv[tid] = -1.0f; seli[tid] = -1; }

    /* Softmax pass 1: max */
    float mx = live ? (x > -1e30f ? x : -1e30f) : -1e30f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_xor(mx, off);
        if (o > mx) mx = o;
    }
    if (lane == 0) swv[warp] = mx;
    __syncthreads();
    mx = swv[0];
    for (int w = 1; w < nwarp; w++)
        if (swv[w] > mx) mx = swv[w];

    /* Softmax pass 2: exp(x - max), the block kernel's sum tree */
    float ev = live ? expf(x - mx) : 0.0f;
    sp[tid] = ev;
    for (int st = n_expert / 2; st >= 32; st >>= 1) {
        __syncthreads();
        if (tid < st)
            sp[tid] = sp[tid] + sp[tid + st];
    }
    __syncthreads();
    if (warp == 0) {
        float sum = sp[lane];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down(sum, off);
        if (lane == 0) sbc[0] = 1.0f / sum;
    }
    __syncthreads();
    float p = ev * sbc[0];

    /* Selection by rank among survivors. Candidate (p, t) for non-NaN p,
     * else the no-winner (-1, -1). T = k-th best warp maximum, when there are
     * k warps: k experts are >= T, so the top-k all are, and only experts T
     * does not beat ("survivors") can be chosen. A survivor's rank among the
     * survivors is its rank among all experts. */
    float cv = (live && p == p) ? p : -1.0f;
    int ci = (live && p == p) ? tid : -1;
    float wv = cv;
    int wi = ci;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor(wv, off);
        int oi = __shfl_xor(wi, off);
        if (fc_moe_router_beats(ov, oi, wv, wi)) { wv = ov; wi = oi; }
    }
    if (lane == 0) { swv[warp] = wv; swi[warp] = wi; }
    if (tid == 0) { st_v = -1.0f; st_i = -1; s_nsurv = 0; }
    __syncthreads();
    if (tid < nwarp && k <= nwarp) {
        int rank = 0;
        for (int w = 0; w < nwarp; w++)
            rank += fc_moe_router_beats(swv[w], swi[w], swv[tid], swi[tid]);
        if (rank == k - 1) { st_v = swv[tid]; st_i = swi[tid]; }
    }
    __syncthreads();
    int keep = ci >= 0 && !fc_moe_router_beats(st_v, st_i, cv, ci);
    if (keep) {
        int pos = atomicAdd(&s_nsurv, 1);
        sp[pos] = cv;
        ssi[pos] = ci;
    }
    __syncthreads();
    int chosen = 0;
    if (keep) {
        int ns = s_nsurv, rank = 0;
        for (int j = 0; j < ns; j++)
            rank += fc_moe_router_beats(sp[j], ssi[j], cv, ci);
        if (rank < k) { selv[rank] = cv; seli[rank] = ci; chosen = 1; }
    }
    if (live)
        lg[tid] = chosen ? -1.0f : p;
    __syncthreads();

    /* norm_topk_prob + expert_weights_scale. */
    if (tid == 0) {
        float wt_sum = 0.0f;
        for (int s = 0; s < k; s++)
            wt_sum += selv[s];
        float inv_wt = (wt_sum > 0.0f) ? (1.0f / wt_sum) : 0.0f;
        sbc[1] = expert_weights_scale * inv_wt;
    }
    __syncthreads();
    if (tid < k) {
        ti[tid] = seli[tid];
        tw[tid] = selv[tid] * sbc[1];
    }
}

/*
 * Weighted accumulate: moe_out[i] += weight * expert_out[i].
 * The scatter/sum step of the routed-expert loop (qwen35moe_moe.c lines 144-146).
 *
 * expert_out: [n_embd] fp32 — one expert's SwiGLU output
 * moe_out:    [n_embd] fp32 — running accumulator (host zeroes it before s=0)
 * weight:     the routed weight top_wt[s] for this expert
 *
 * Launched with ceil(n_embd / blockDim.x) blocks, blockDim.x threads (use 256).
 * No shared memory.
 */
__global__ void fc_moe_weighted_add(const float *expert_out, float *moe_out,
                                    float weight, int n_embd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd)
        moe_out[i] += weight * expert_out[i];
}

/*
 * Shared-expert sigmoid gate: shexp_out[i] *= sigmoid(gate_logit).
 * The scalar gate applied ONLY to the shared-expert branch
 * (qwen35moe_moe.c lines 183-185). gate_logit = (ffn_gate_inp_shexp . x) is a
 * single dot product computed by the host (the shared gate is a 1-D vector, so
 * the CPU path computes it directly rather than via the matvec kernel).
 *
 * shexp_out:  [n_embd] fp32 — shared-expert SwiGLU output, gated in place
 * gate_logit: scalar pre-sigmoid logit
 *
 * Launched with ceil(n_embd / blockDim.x) blocks, blockDim.x threads (use 256).
 * No shared memory.
 */
__global__ void fc_moe_shared_gate(float *shexp_out, float gate_logit,
                                   int n_embd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        float sgate = 1.0f / (1.0f + expf(-gate_logit));
        shexp_out[i] *= sgate;
    }
}

} /* extern "C" */
