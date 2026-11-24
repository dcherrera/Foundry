/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * ssm_gated_delta.cu — Qwen3.5-MoE Gated-Delta-Net SSM kernels for Foundry
 *
 * Target: sm_37 (K80, Kepler)
 * Compile: nvcc -ptx -arch=sm_37 ssm_gated_delta.cu -o ssm_gated_delta.ptx
 *
 * The GPU halves of the single-token (decode) Gated-Delta-Net SSM block. Each
 * kernel mirrors, line-for-line, a static helper in the CPU reference
 *   Pantheon/src/models/qwen35moe_ssm.c
 * and is validated against the golden vectors in Pantheon/tests/golden/.
 *
 * Kernels emitted (PTX symbols for ../cuda/fc_cuda_ptx.h, FC_PTX_SSM_GATED_DELTA).
 * ★ ADDING A KERNEL HERE IS A THREE-STEP CHANGE, and skipping either of the last
 * two makes the kernel INERT while the serve prints a misleading load failure:
 *   1. define it here, then
 *   2. `cd cuda_kernels && make` ON A HOST WITH nvcc 11.4 to regenerate ../cuda/fc_cuda_ptx.h
 *      (a checked-in artifact — the kernel does not exist until it is in there), and
 *   3. add its name to ssm_names[] in fc_cuda.c AND bump the count argument to
 *      fc_load_ptx_module. The loader assigns names in order and stops at the
 *      first one the module lacks, so a name with no symbol behind it makes
 *      fc_load_ptx_module return nonzero and print "SSM gated-delta PTX load
 *      failed (kernels unavailable)" even though every earlier kernel loaded fine.
 * The oldest three, for orientation (the rest are documented at their definitions):
 *   fc_ssm_recurrence    — per-token delta-rule recurrence (THE centerpiece)
 *                          mirrors qwen35moe_compute_recur_heads (ssm.c:163-196)
 *   fc_ssm_conv1d_silu   — causal depthwise conv1d (k=4) + SiLU, rolls conv_state
 *                          mirrors qwen35moe_conv1d_silu (ssm.c:110-135)
 *   fc_ssm_gated_norm    — gated RMSNorm: RMSNorm(y,w) * silu(z), per V-head
 *                          mirrors qwen35moe_gated_norm (ssm.c:206-223)
 *
 * Config (verified Nex-N2-mini): n_v_heads=32, n_k_heads=16, state_size S=128,
 * scale=1/sqrt(128), conv_dim=8192, conv_kernel=4. Transposed state layout:
 *   s[h*S*S + j*S + i] == S_h[i][j]   (row j contiguous over i).
 * The q/k arriving here are ALREADY L2-normed and GQA-repeated to n_v_heads on
 * the host (interleaved tiling vh = r*n_k_heads + kh); this kernel never sees
 * the 16->32 repeat, only the post-repeat [n_v_heads*S] streams.
 *
 * Kepler notes: pure fp32 throughout; plain shared-memory tree reductions
 * (no __shfl / _sync portability question); expf/__expf via the standard libm
 * device intrinsics, matching the CPU expf bit-tolerance band (~1e-4..1e-3 rel).
 */

extern "C" {

/* SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
 * Mirrors qwen35moe_silu (ssm.c:60-63). */
__device__ __forceinline__ float fc_ssm_silu(float x) {
    return x / (1.0f + expf(-x));
}

/*
 * ============================================================
 * Kernel 1: per-token delta-rule recurrence (THE centerpiece)
 * ============================================================
 *
 * Mirrors qwen35moe_compute_recur_heads (ssm.c:163-196), one (head h, row j)
 * per CUDA block. The CPU walks all S rows of all heads serially; the GPU maps
 * the 4096 independent (h,j) pairs onto the grid. Within a block the S=128
 * threads cooperate on the two length-S dot reductions.
 *
 * State is transposed: s[h*S*S + j*S + i] == S_h[i][j]; row j is contiguous
 * over i, so each block owns one unit-stride row `row[i] = s[h*S*S + j*S + i]`.
 * For one token (CPU lines 180-193):
 *   (a) row[i] *= exp(g[h])                       scalar decay of the whole row
 *   (b) pred = sum_i row[i]*k[h][i]               (shared-mem tree reduction)
 *       delta = (v[h][j] - pred) * beta[h]
 *   (c) row[i] += k[h][i] * delta                 rank-1 write, in place
 *   (d) y[h][j] = scale * sum_i row[i]*q[h][i]    read-AFTER-write (2nd reduce)
 * The decayed+updated row is written back to s in place (state carries forward).
 *
 * Inputs:
 *   s      [n_v_heads * S * S] transposed state, updated in place
 *   q,k,v  [n_v_heads * S]     q/k L2-normed + GQA-repeated on host; v straight
 *   g      [n_v_heads]         scalar decay (PRE-exp; we exp here, CPU line 173)
 *   beta   [n_v_heads]         sigmoid'd on host (CPU applies sigmoid earlier)
 *   y      [n_v_heads * S]     output, one scalar per (h,j)
 *   state_size  S (must equal blockDim.x)
 *   scale  1/sqrt(S)
 *
 * Grid:  (state_size, n_v_heads, 1)  — blockIdx.x = j (row), blockIdx.y = h.
 * Block: state_size threads (= S = 128), one thread per i.
 * Shared mem: blockDim.x * sizeof(float) (reduction scratch).
 */
__global__ void fc_ssm_recurrence(float *s, const float *q, const float *k,
                                  const float *v, const float *g,
                                  const float *beta, float *y,
                                  int state_size, float scale) {
    extern __shared__ float sdata[];

    const int S = state_size;
    const int j = blockIdx.x;    /* row j of the transposed state */
    const int h = blockIdx.y;    /* V-head */
    const int i = threadIdx.x;   /* state coordinate within the row */
    if (i >= S)
        return;

    float *row = s + ((size_t)h * S + j) * S;   /* s[h*S*S + j*S + i] over i */
    const float *kh = k + (size_t)h * S;
    const float *qh = q + (size_t)h * S;

    const float decay = expf(g[h]);
    const float beta_h = beta[h];

    /* (a) scalar decay of the whole row (CPU lines 180-182). */
    float ri = row[i] * decay;

    /* (b) pred = sum_i row'[i]*k[i] via shared-mem tree reduction. */
    sdata[i] = ri * kh[i];
    __syncthreads();
    for (int stride = S / 2; stride > 0; stride >>= 1) {
        if (i < stride)
            sdata[i] += sdata[i + stride];
        __syncthreads();
    }
    float pred = sdata[0];
    /* delta = (v[h][j] - pred) * beta  (CPU line 186; v indexed by j). */
    float delta = (v[(size_t)h * S + j] - pred) * beta_h;
    __syncthreads();   /* all threads consumed sdata[0] before the next reduce */

    /* (c) rank-1 write: row[i] += k[i]*delta (CPU lines 188-190), in place. */
    ri += kh[i] * delta;
    row[i] = ri;

    /* (d) read-after-write: y[h][j] = scale * sum_i row[i]*q[i] (CPU line 193). */
    sdata[i] = ri * qh[i];
    __syncthreads();
    for (int stride = S / 2; stride > 0; stride >>= 1) {
        if (i < stride)
            sdata[i] += sdata[i + stride];
        __syncthreads();
    }
    if (i == 0)
        y[(size_t)h * S + j] = scale * sdata[0];
}

/*
 * ============================================================
 * Kernel 2: causal conv1d (k=4) + SiLU, rolls conv_state
 * ============================================================
 *
 * Mirrors qwen35moe_conv1d_silu (ssm.c:110-135). One thread per channel c over
 * the full conv_dim stream (depthwise; channels are independent). Each channel's
 * sliding window is [conv_state(c,0) .. conv_state(c,k-2), qkv(c)] (length k):
 *   out[c] = SiLU( sum_{t<k-1} conv_state[c][t]*w[c][t] + qkv[c]*w[c][k-1] )
 * then the per-channel conv_state is rolled (drop oldest, append raw qkv[c]).
 *
 * conv_state layout: [conv_dim][k-1], oldest..newest per channel (CPU comment
 * lines 102-108). conv weight is row-major [conv_dim, k]: w[c][tap]=weight[c*k+tap].
 * conv_state is updated in place; out holds the SiLU'd convolved stream.
 *
 * Inputs:
 *   qkv         [conv_dim]        raw projected stream (this token)
 *   weight      [conv_dim * k]    depthwise conv weights, row-major
 *   conv_state  [conv_dim*(k-1)]  per-channel history, updated in place
 *   out         [conv_dim]        SiLU'd output
 *   conv_dim, conv_kernel (k)
 *
 * Grid:  ceil(conv_dim / blockDim.x) blocks.
 * Block: 256 threads. No shared memory.
 */
__global__ void fc_ssm_conv1d_silu(const float *qkv, const float *weight,
                                   float *conv_state, float *out,
                                   int conv_dim, int conv_kernel) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim)
        return;

    const int k = conv_kernel;
    const int km1 = k - 1;
    const float *cw = weight + (size_t)c * k;
    float *cs = conv_state + (size_t)c * km1;

    /* out[c] = SiLU( history . w[0..k-2] + qkv[c]*w[k-1] )  (CPU lines 125-128). */
    float acc = qkv[c] * cw[km1];
    for (int t = 0; t < km1; t++)
        acc += cs[t] * cw[t];
    out[c] = fc_ssm_silu(acc);

    /* Roll the per-channel state: drop oldest, append raw qkv (CPU lines 131-133). */
    for (int t = 0; t < km1 - 1; t++)
        cs[t] = cs[t + 1];
    cs[km1 - 1] = qkv[c];
}

/*
 * ============================================================
 * Kernel 3: gated RMSNorm — RMSNorm(y, w) * silu(z), per V-head
 * ============================================================
 *
 * Mirrors qwen35moe_gated_norm (ssm.c:206-223), minus the trailing ssm_out
 * projection (that reuses fc_matvec_q3k on the host). One block per V-head; the
 * S=128 threads do a shared-mem sum-of-squares reduction, then each writes
 *   gated[h*S+d] = (y[h*S+d] / sqrt(mean(y^2)+eps)) * w[d] * silu(z[h*S+d]).
 * The norm weight w is shared across heads (length S), exactly like the CPU
 * pn_vrms_norm_f32 call per head (CPU line 219) followed by the silu(z) multiply
 * (CPU lines 220-221).
 *
 * Inputs:
 *   y       [n_v_heads * S]   recurrence output
 *   weight  [S]               ssm_norm weight (shared across heads)
 *   z       [n_v_heads * S]   attn_gate projection (pre-silu)
 *   gated   [n_v_heads * S]   output
 *   state_size S, eps
 *
 * Grid:  (n_v_heads, 1, 1)  one block per head.
 * Block: 256 threads (or S if smaller). Shared mem: blockDim.x * sizeof(float).
 */
__global__ void fc_ssm_gated_norm(const float *y, const float *weight,
                                  const float *z, float *gated,
                                  int state_size, float eps) {
    extern __shared__ float sdata[];

    const int S = state_size;
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    const float *yh = y + (size_t)h * S;
    const float *zh = z + (size_t)h * S;
    float *gh = gated + (size_t)h * S;

    /* Phase 1: partial sum of squares of y over this head's S dims. */
    float sum_sq = 0.0f;
    for (int d = tid; d < S; d += threads) {
        float v = yh[d];
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    /* RMSNorm divisor matches pn_vrms_norm_f32: sqrt(mean(y^2) + eps). */
    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)S + eps);

    /* Phase 2: gated[d] = RMSNorm(y)[d] * w[d] * silu(z[d]). */
    for (int d = tid; d < S; d += threads)
        gh[d] = yh[d] * inv_rms * weight[d] * fc_ssm_silu(zh[d]);
}

/* ================================================================
 * SSM "glue" kernels — replace the host round-trips in the qwen35moe GPU
 * forward (each was a D2H -> CPU -> H2D per SSM layer, ~270 sync'd PCIe
 * transfers/token = the tok/s floor). Small element-wise ops, pure fp32.
 * ================================================================ */

/* Per-element sigmoid, in place: x = 1/(1+exp(-x)). One thread per element.
 * Replaces the host-side beta = sigmoid(beta). */
__global__ void fc_sigmoid(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = 1.0f / (1.0f + expf(-x[i]));
}

/* GDA decay gate: g[h] = ssm_a[h] * softplus(alpha[h] + ssm_dt[h]), per V-head.
 * softplus(v) = v>20 ? v : log1p(exp(v)); ssm_a = -exp(A_log) (pre-baked host
 * side). One thread per head. Replaces the 3x D2H + CPU + H2D decay-gate glue. */
__global__ void fc_ssm_decay_gate(const float *alpha, const float *ssm_dt,
                                  const float *ssm_a, float *g, int n) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n)
        return;
    float v = alpha[h] + ssm_dt[h];
    float sp = (v > 20.0f) ? v : log1pf(expf(v));
    g[h] = ssm_a[h] * sp;
}

/* Per-K-head L2-norm of q and k over their S state dims, then GQA-repeat
 * interleaved to n_v_heads (vh = r*n_k_heads + kh). conv_out layout is
 * [q key_dim | k key_dim | v val_dim]; this handles q,k only (v is a straight
 * copy done by the launcher). One block per K-head, blockDim.x == S, shared mem
 * [2*S]. L2-norm matches gpu_l2_norm: x / max(sqrt(sum x^2), eps). */
__global__ void fc_ssm_qknorm_gqa(const float *conv_out, float *qg, float *kg,
                                  int n_k_heads, int kv_groups, int S,
                                  int key_dim, float eps) {
    int kh = blockIdx.x;
    if (kh >= n_k_heads)
        return;
    int i = threadIdx.x;                 /* one thread per state dim (== S) */
    extern __shared__ float sh[];        /* [2*blockDim.x]: q | k sumsq partials */
    const float *q_src = conv_out;
    const float *k_src = conv_out + key_dim;

    float qv = q_src[(size_t)kh * S + i];
    float kv = k_src[(size_t)kh * S + i];
    sh[i] = qv * qv;
    sh[blockDim.x + i] = kv * kv;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (i < stride) {
            sh[i] += sh[i + stride];
            sh[blockDim.x + i] += sh[blockDim.x + i + stride];
        }
        __syncthreads();
    }
    float inv_q = 1.0f / fmaxf(sqrtf(sh[0]), eps);
    float inv_k = 1.0f / fmaxf(sqrtf(sh[blockDim.x]), eps);
    float qn = qv * inv_q;
    float kn = kv * inv_k;
    for (int r = 0; r < kv_groups; r++) {
        int vh = r * n_k_heads + kh;     /* interleaved GQA tiling */
        qg[(size_t)vh * S + i] = qn;
        kg[(size_t)vh * S + i] = kn;
    }
}

/* ================================================================
 * BATCHED (prefill) variants — n positions of ONE sequence per launch.
 *
 * The two stateful kernels (conv1d, recurrence) are a SCAN over the
 * positions: each block loops t = 0..n-1 internally, carrying its slice of
 * the state forward exactly as n single-token launches would — the per-
 * position fp op order is identical to the single-token kernels above, so
 * every position's output and the final carried state are bit-exact. The
 * stateless glue (decay gate, qknorm+GQA) just gains a position axis.
 *
 * Batched activations are token-major: qkv/out [n][conv_dim],
 * q/k/v/y [n][n_v_heads*S], g/beta [n][n_v_heads].
 * ================================================================ */

/* Causal conv1d + SiLU over n positions (scan). One thread per channel;
 * the (k-1)-deep history lives in registers across positions and is
 * written back once at the end. Per position the accumulation order
 * matches fc_ssm_conv1d_silu exactly. Requires conv_kernel <= 4.
 *
 * snaps (optional, NULL = off): per-position state snapshots for
 * speculative decoding — snaps[t] = the conv_state image AFTER position t
 * was consumed, [n][conv_dim][km1] (same channel-major layout as
 * conv_state), so a partial accept of M positions restores state-after-M
 * with one contiguous copy of snaps[M-1].
 *
 * LOCAL-MEMORY NOTE (why the k==4 case below is hand-scalarised):
 * the generic body keeps the history in `float hist[3]` indexed by loops
 * bounded by the RUNTIME parameter km1, and stores hist[km1-1] at a runtime
 * index. nvcc cannot promote an array under a runtime-indexed store, so the
 * whole array went to local memory — the committed PTX carries
 *   .local .align 4 .b8 __local_depot6[12];
 * with ~3 ld.local + 3 st.local per position per thread plus two
 * unroll-remainder branch ladders, while `ptxas -v` still prints
 * "0 bytes spill" (read the STACK FRAME, not the spill counters).
 * conv_kernel is 4 for every model that reaches here and the launcher
 * already rejects conv_kernel > 4, so k==4 is specialised into three named
 * scalars: no runtime index, no local traffic. BIT-EXACT — the accumulation is
 * the same `mul` then three `fma`s in the same u = 0,1,2 order the generic path
 * already emitted (verified against the committed PTX).
 *
 * MEASURED IN THE REGENERATED PTX (2026-08-10), and read this before chasing the
 * frame again: `__local_depot6[12]` is STILL DECLARED, because the generic k<4
 * fallback below lives in the same entry and still stores at a runtime index, so
 * nvcc must reserve the frame for it. What moved is the *executed* path — the
 * `setp.eq %r, 4` branch target carries **0 ld.local and 0 st.local**, against
 * **26 ld.local + 14 st.local** when k==4 had no specialisation and that was the
 * only path there was. Every model reaching here has conv_kernel == 4, so no
 * executed instruction touches local memory and the reserved frame costs nothing
 * but address space. Getting the frame to a literal zero needs the k==2 and k==3
 * cases hand-scalarised too (or the launcher narrowed to conv_kernel == 4 and the
 * generic body deleted); this kernel is 0.4 % of prefill, so that was not done. */
__global__ void fc_ssm_conv1d_silu_seq(const float *qkv, const float *weight,
                                       float *conv_state, float *out,
                                       int conv_dim, int conv_kernel, int n,
                                       float *snaps) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim)
        return;

    const int k = conv_kernel;
    const int km1 = k - 1;
    const float *cw = weight + (size_t)c * k;
    float *cs = conv_state + (size_t)c * km1;

    if (k == 4) {
        const float w0 = cw[0], w1 = cw[1], w2 = cw[2], w3 = cw[3];
        float h0 = cs[0], h1 = cs[1], h2 = cs[2];
        for (int t = 0; t < n; t++) {
            float xv = qkv[(size_t)t * conv_dim + c];
            float acc = __fmul_rn(xv, w3);
            acc = __fmaf_rn(h0, w0, acc);
            acc = __fmaf_rn(h1, w1, acc);
            acc = __fmaf_rn(h2, w2, acc);
            out[(size_t)t * conv_dim + c] = fc_ssm_silu(acc);
            h0 = h1;
            h1 = h2;
            h2 = xv;
            if (snaps) {
                float *sn = snaps + ((size_t)t * conv_dim + c) * 3;
                sn[0] = h0;
                sn[1] = h1;
                sn[2] = h2;
            }
        }
        cs[0] = h0;
        cs[1] = h1;
        cs[2] = h2;
        return;
    }

    float hist[3];                       /* km1 <= 3 (conv_kernel <= 4) */

    for (int t = 0; t < km1; t++)
        hist[t] = cs[t];

    for (int t = 0; t < n; t++) {
        float xv = qkv[(size_t)t * conv_dim + c];
        float acc = xv * cw[km1];
        for (int u = 0; u < km1; u++)
            acc += hist[u] * cw[u];
        out[(size_t)t * conv_dim + c] = fc_ssm_silu(acc);
        for (int u = 0; u < km1 - 1; u++)
            hist[u] = hist[u + 1];
        hist[km1 - 1] = xv;
        if (snaps) {
            float *sn = snaps + ((size_t)t * conv_dim + c) * km1;
            for (int u = 0; u < km1; u++)
                sn[u] = hist[u];
        }
    }

    for (int t = 0; t < km1; t++)
        cs[t] = hist[t];
}

/* Delta-rule recurrence over n positions (scan). Block (j, h) loops the
 * positions, carrying its state row in registers (ri) between them — the
 * row is read once, updated n times with the exact per-position op order
 * of fc_ssm_recurrence, and written back once. q/k/v/y advance by
 * n_v_heads*S per position, g/beta by n_v_heads.
 *
 * snaps (optional, NULL = off): per-position state snapshots for
 * speculative decoding — snaps[t] = the full recurrent state AFTER
 * position t's update, [n][n_v_heads*S*S] (same layout as s), so a
 * partial accept of M positions restores state-after-M with one
 * contiguous copy of snaps[M-1]. */
__global__ void fc_ssm_recurrence_seq(float *s, const float *q, const float *k,
                                      const float *v, const float *g,
                                      const float *beta, float *y,
                                      int state_size, int n_v_heads, int n,
                                      float scale, float *snaps) {
    extern __shared__ float sdata[];

    const int S = state_size;
    const int j = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;
    if (i >= S)
        return;

    const size_t row_off = ((size_t)h * S + j) * S;
    const size_t state_total = (size_t)n_v_heads * S * S;
    float *row = s + row_off;
    float ri = row[i];

    for (int t = 0; t < n; t++) {
        const float *kh = k + (size_t)t * n_v_heads * S + (size_t)h * S;
        const float *qh = q + (size_t)t * n_v_heads * S + (size_t)h * S;
        const float decay = expf(g[(size_t)t * n_v_heads + h]);
        const float beta_h = beta[(size_t)t * n_v_heads + h];

        ri *= decay;
        sdata[i] = ri * kh[i];
        __syncthreads();
        for (int stride = S / 2; stride > 0; stride >>= 1) {
            if (i < stride)
                sdata[i] += sdata[i + stride];
            __syncthreads();
        }
        float pred = sdata[0];
        float delta = (v[(size_t)t * n_v_heads * S + (size_t)h * S + j] - pred)
                      * beta_h;
        __syncthreads();

        ri += kh[i] * delta;
        if (snaps)
            snaps[(size_t)t * state_total + row_off + i] = ri;
        sdata[i] = ri * qh[i];
        __syncthreads();
        for (int stride = S / 2; stride > 0; stride >>= 1) {
            if (i < stride)
                sdata[i] += sdata[i + stride];
            __syncthreads();
        }
        if (i == 0)
            y[(size_t)t * n_v_heads * S + (size_t)h * S + j] = scale * sdata[0];
        __syncthreads();
    }

    row[i] = ri;
}

/* UNROLLED-REDUCTION prefill recurrence — the cheap half of the warp rewrite.
 *
 * WHY. fc_ssm_recurrence_seq's two tree reductions are written
 * `for (stride = S/2; stride > 0; stride >>= 1)` where S == state_size is a
 * RUNTIME kernel parameter, so ptxas cannot unroll the 7-step tree and every
 * step pays a predicate, a branch, a shift and a latch. From the committed PTX
 * that loop control alone is 336 of the kernel's ~740 warp-issue slots per
 * (block, position) — 45 %. Specialising S to 128 makes the bound a literal and
 * lets ptxas unroll both trees, deleting the loop control but keeping the
 * barrier chain and the shared-memory traffic: ~1.83x fewer issue slots, against
 * ~10x for fc_ssm_recurrence_seq_warp below.
 *
 * It exists as a separately priceable arm: it is ~15 lines and cannot regress
 * occupancy (identical shared and register footprint), so if the warp rewrite
 * fails its ptxas gate this is the fallback that still collects the loop-control
 * half of the win.
 *
 * BIT-EXACT, and trivially so: this is fc_ssm_recurrence_seq verbatim with the
 * loop bound written as a literal. Every floating-point operation is the same
 * expression in the same order with the same operands, and every intermediate
 * still round-trips through shared memory (so ptxas cannot contract a product
 * into an fma here either). Do NOT "tidy" the plain `*` and `+=` into
 * intrinsics: matching the baseline's codegen is the point.
 *
 * Guarded on state_size == FC_SSM_UNROLL_S; the guard is block-uniform so it can
 * never split a __syncthreads. The launcher keeps the baseline for anything else.
 *
 * Grid/block/shared: identical to fc_ssm_recurrence_seq — (S, n_v_heads),
 * S threads, S floats of shared. */

#define FC_SSM_UNROLL_S 128   /* state_size this variant is specialised for */

__global__ void fc_ssm_recurrence_seq_unroll128(float *s, const float *q,
                                                const float *k, const float *v,
                                                const float *g,
                                                const float *beta, float *y,
                                                int state_size, int n_v_heads,
                                                int n, float scale,
                                                float *snaps) {
    extern __shared__ float sdata[];

    /* Block-uniform: every thread returns or none does. */
    if (state_size != FC_SSM_UNROLL_S)
        return;

    const int S = FC_SSM_UNROLL_S;
    const int j = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;
    if (i >= S)
        return;

    const size_t row_off = ((size_t)h * S + j) * S;
    const size_t state_total = (size_t)n_v_heads * S * S;
    float *row = s + row_off;
    float ri = row[i];

    for (int t = 0; t < n; t++) {
        const float *kh = k + (size_t)t * n_v_heads * S + (size_t)h * S;
        const float *qh = q + (size_t)t * n_v_heads * S + (size_t)h * S;
        const float decay = expf(g[(size_t)t * n_v_heads + h]);
        const float beta_h = beta[(size_t)t * n_v_heads + h];

        ri *= decay;
        sdata[i] = ri * kh[i];
        __syncthreads();
#pragma unroll
        for (int stride = FC_SSM_UNROLL_S / 2; stride > 0; stride >>= 1) {
            if (i < stride)
                sdata[i] += sdata[i + stride];
            __syncthreads();
        }
        float pred = sdata[0];
        float delta = (v[(size_t)t * n_v_heads * S + (size_t)h * S + j] - pred)
                      * beta_h;
        __syncthreads();

        ri += kh[i] * delta;
        if (snaps)
            snaps[(size_t)t * state_total + row_off + i] = ri;
        sdata[i] = ri * qh[i];
        __syncthreads();
#pragma unroll
        for (int stride = FC_SSM_UNROLL_S / 2; stride > 0; stride >>= 1) {
            if (i < stride)
                sdata[i] += sdata[i + stride];
            __syncthreads();
        }
        if (i == 0)
            y[(size_t)t * n_v_heads * S + (size_t)h * S + j] = scale * sdata[0];
        __syncthreads();
    }

    row[i] = ri;
}

/* ================================================================
 * WARP-REDUCTION prefill recurrence — the tree-preserving rewrite.
 *
 * WHY. In fc_ssm_recurrence_seq above, one 128-thread block advances ONE
 * state column (h, j) and spends its whole life in two shared-memory tree
 * reductions. From the committed PTX for that kernel, per (block, position):
 *
 *   740 warp-instruction issue slots, of which
 *     488 (66 %)  the two tree reductions, and inside those
 *       336 (45 %)  setp/bra/shr/mov loop control — the 7-step loop is NOT
 *                   unrolled because its bound is `state_size`, a RUNTIME
 *                   kernel parameter, so every step pays a predicate, a
 *                   branch, a shift and a latch
 *        72 ( 10 %) bar.sync (18 barriers per position, not 14)
 *        48 ( 6 %)  the actual ld.shared/add.f32/st.shared
 *      76 (10 %)  expf(g[t][h]) — 19 PTX ops incl. ex2.approx.ftz, on a
 *                 value UNIFORM across the entire block, recomputed by all
 *                 128 threads at all 128 positions (16384x redundant)
 *    and only ~41 slots (5.5 %) doing useful floating-point work.
 * Occupancy is already 100 % (512 B shared, ~40 regs, 16 blocks/SMX), the
 * grid is 4096 blocks = 315 per SMX, and the kernel still sustains only
 * IPC ~2/SMX. It is not bandwidth-, compute-, launch- or occupancy-bound:
 * it is issue-bound on overhead, and 3 of its 4 warps idle at a barrier
 * through 5 of the 7 tree steps.
 *
 * WHAT. One WARP owns one column (h, j); lane L carries state elements
 * i = L, L+32, L+64, L+96 in registers. The length-128 dot then costs 4
 * products, 3 intra-lane adds and a 5-step __shfl_xor butterfly — no shared
 * memory, and __syncthreads() disappears from the position loop entirely
 * (one barrier per BLOCK, for the decay/beta staging). ~72 issue slots per
 * (column, position) against 740: a 10x instruction cut with 49 % of the
 * slots now doing useful FP work.
 *
 * BIT-EXACT, and this is the whole point of the structure:
 *  - fc_ssm_warp_tree_dot reproduces the baseline's reduction TREE, not just
 *    its value. The baseline pairs index i with i+64, then i with i+32, then
 *    runs strides 16..1. In lane-local terms 64 and 32 are e-strides 2 and 1
 *    over the 4 elements a lane holds, so p[0]+=p[2]; p[1]+=p[3]; p[0]+=p[1]
 *    lands the identical grouping, and strides 16..1 are exactly a 32-lane
 *    butterfly. __shfl_xor gives lane L the pair (v_L, v_{L^st}); for L < st
 *    that is the baseline's (v_L, v_{L+st}) and for L >= st it is the same
 *    two values with the operands swapped — fp add is commutative bit-for-bit,
 *    so every lane ends holding the baseline's sdata[0] exactly, which also
 *    removes the broadcast.
 *  - every arithmetic op uses __fmul_rn/__fadd_rn/__fsub_rn/__fmaf_rn. This
 *    is NOT decoration: the baseline's per-element product went through
 *    shared memory, so nvcc could not contract it, while here everything is
 *    in registers and nvcc WOULD fuse p[0] + p[2] into fma(r2, k2, p0) and
 *    silently change the result. The _rn intrinsics are contraction barriers.
 *  - expf is hoisted, not approximated: the same expf() on the same fp32
 *    input, stored fp32 in shared and reloaded, is the same bits.
 *
 * Specialised to state_size == 128 (every model that reaches this kernel);
 * the launcher keeps the baseline for anything else, and the guard below is
 * block-uniform so it can never split a __syncthreads.
 *
 * Grid:  (state_size / FC_SSM_WARP_COLS, n_v_heads)  = (32, 32) = 1024 blocks
 * Block: FC_SSM_WARP_COLS * 32 = 128 threads (4 warps = 4 columns)
 * Shared: 2 * n floats — decay[n] | beta[n], staged once per block.
 * ================================================================ */

#define FC_SSM_WARP_S     128   /* state_size this variant is specialised for */
#define FC_SSM_WARP_ELEMS 4     /* FC_SSM_WARP_S / 32 state elements per lane */
#define FC_SSM_WARP_COLS  4     /* warps per block == state columns per block */

#if FC_SSM_WARP_ELEMS != 4
#error "fc_ssm_warp_tree_dot's pairing is derived for S=128 (4 elems/lane)"
#endif

/* Length-FC_SSM_WARP_S dot of the lane-distributed vectors r and x, reducing
 * in the exact binary tree fc_ssm_recurrence_seq's shared-memory sweep uses.
 * Returns the identical fp32 bits in all 32 lanes.
 *
 * p[] is indexed only by loops with COMPILE-TIME trip counts, so it is
 * register-promoted. Verify that, do not assume it: `ptxas -v` must report a
 * ZERO stack frame for this kernel. On this project a small array under a
 * runtime-bounded loop lands in local memory while the spill counters still
 * read "0 bytes spill" (see fc_ssm_conv1d_silu_seq's __local_depot above). */
__device__ __forceinline__ float fc_ssm_warp_tree_dot(const float *r,
                                                      const float *x) {
    /* Level 0: the products the baseline writes into sdata[i]. */
    float p[FC_SSM_WARP_ELEMS];
#pragma unroll
    for (int e = 0; e < FC_SSM_WARP_ELEMS; e++)
        p[e] = __fmul_rn(r[e], x[e]);

    /* Baseline strides S/2 .. 32 — entirely inside one lane (e-strides). */
#pragma unroll
    for (int st = FC_SSM_WARP_ELEMS / 2; st > 0; st >>= 1)
#pragma unroll
        for (int e = 0; e < st; e++)
            p[e] = __fadd_rn(p[e], p[e + st]);

    /* Baseline strides 16 .. 1 over the 32 survivors — a warp butterfly. */
    float acc = p[0];
#pragma unroll
    for (int st = 16; st > 0; st >>= 1)
        acc = __fadd_rn(acc, __shfl_xor(acc, st));
    return acc;
}

__global__ void fc_ssm_recurrence_seq_warp(float *s, const float *q,
                                           const float *k, const float *v,
                                           const float *g, const float *beta,
                                           float *y, int state_size,
                                           int n_v_heads, int n, float scale,
                                           float *snaps) {
    extern __shared__ float sh[];        /* [2*n]: decay[n] | beta[n] */

    /* Block-uniform: every thread returns or none does. */
    if (state_size != FC_SSM_WARP_S)
        return;

    const int S = FC_SSM_WARP_S;
    const int lane = (int)(threadIdx.x & 31u);
    const int col  = (int)(threadIdx.x >> 5);
    const int j = blockIdx.x * FC_SSM_WARP_COLS + col;
    const int h = blockIdx.y;

    /* Stage the block-uniform decay/beta for all n positions once. */
    float *sh_decay = sh;
    float *sh_beta = sh + n;
    for (int t = (int)threadIdx.x; t < n; t += FC_SSM_WARP_COLS * 32) {
        sh_decay[t] = expf(g[(size_t)t * n_v_heads + h]);
        sh_beta[t] = beta[(size_t)t * n_v_heads + h];
    }
    __syncthreads();                     /* the only barrier in this kernel */

    if (j >= S)
        return;

    const size_t row_off = ((size_t)h * S + j) * S;
    const size_t state_total = (size_t)n_v_heads * S * S;
    const size_t pos_stride = (size_t)n_v_heads * S;
    float *row = s + row_off;

    float r[FC_SSM_WARP_ELEMS];
#pragma unroll
    for (int e = 0; e < FC_SSM_WARP_ELEMS; e++)
        r[e] = row[lane + e * 32];

    const float *kh = k + (size_t)h * S + lane;
    const float *qh = q + (size_t)h * S + lane;
    const float *vj = v + (size_t)h * S + j;
    float *yj = y + (size_t)h * S + j;

    for (int t = 0; t < n; t++) {
        const size_t toff = (size_t)t * pos_stride;
        const float decay = sh_decay[t];
        const float beta_h = sh_beta[t];

        float kv[FC_SSM_WARP_ELEMS];
#pragma unroll
        for (int e = 0; e < FC_SSM_WARP_ELEMS; e++) {
            kv[e] = kh[toff + e * 32];
            r[e] = __fmul_rn(r[e], decay);
        }
        const float pred = fc_ssm_warp_tree_dot(r, kv);
        const float delta = __fmul_rn(__fsub_rn(vj[toff], pred), beta_h);

        float qv[FC_SSM_WARP_ELEMS];
#pragma unroll
        for (int e = 0; e < FC_SSM_WARP_ELEMS; e++) {
            r[e] = __fmaf_rn(kv[e], delta, r[e]);
            qv[e] = qh[toff + e * 32];
        }
        if (snaps) {
            float *sn = snaps + (size_t)t * state_total + row_off;
#pragma unroll
            for (int e = 0; e < FC_SSM_WARP_ELEMS; e++)
                sn[lane + e * 32] = r[e];
        }
        const float out = fc_ssm_warp_tree_dot(r, qv);
        if (lane == 0)
            yj[toff] = __fmul_rn(out, scale);
    }

#pragma unroll
    for (int e = 0; e < FC_SSM_WARP_ELEMS; e++)
        row[lane + e * 32] = r[e];
}

/* ================================================================
 * MULTI-USER DECODE (slot) variants — K independent sequences, ONE token
 * each, per launch. Unlike the _seq scans above there is NO cross-position
 * dependency: slot u owns its OWN state image at offset u and advances it
 * exactly one step. Per slot the fp op order is identical to the
 * single-token kernels (fc_ssm_conv1d_silu / fc_ssm_recurrence), so each
 * user's output and carried state are bit-exact vs decoding alone.
 *
 * Slot-major activations match the batch layout: qkv/out [K][conv_dim],
 * q/k/v/y [K][n_v_heads*S], g/beta [K][n_v_heads]. States are slot-major:
 * conv_state [K][conv_dim*(k-1)], s [K][n_v_heads*S*S].
 * ================================================================ */

/* Causal conv1d + SiLU, one step per slot. Grid (ceil(conv_dim/block), K);
 * per (slot, channel) the math is fc_ssm_conv1d_silu verbatim on that
 * slot's state slice. */
__global__ void fc_ssm_conv1d_silu_slots(const float *qkv, const float *weight,
                                         float *conv_state, float *out,
                                         int conv_dim, int conv_kernel,
                                         const int *rows) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int u = blockIdx.y;                  /* slot (user) */
    if (c >= conv_dim)
        return;

    const int k = conv_kernel;
    const int km1 = k - 1;
    const int r = rows ? rows[u] : u;    /* the slot's state row */
    const float *cw = weight + (size_t)c * k;
    float *cs = conv_state + ((size_t)r * conv_dim + c) * km1;
    const float *x = qkv + (size_t)u * conv_dim;
    float *o = out + (size_t)u * conv_dim;

    float acc = x[c] * cw[km1];
    for (int t = 0; t < km1; t++)
        acc += cs[t] * cw[t];
    o[c] = fc_ssm_silu(acc);

    for (int t = 0; t < km1 - 1; t++)
        cs[t] = cs[t + 1];
    cs[km1 - 1] = x[c];
}

/* Delta-rule recurrence, one step per slot. Grid (S, n_v_heads, K); block
 * (j, h, u) updates row j of head h of SLOT u's state — the body is
 * fc_ssm_recurrence verbatim at that slot's offsets. */
__global__ void fc_ssm_recurrence_slots(float *s, const float *q,
                                        const float *k, const float *v,
                                        const float *g, const float *beta,
                                        float *y, int state_size,
                                        int n_v_heads, float scale,
                                        const int *rows) {
    extern __shared__ float sdata[];

    const int S = state_size;
    const int j = blockIdx.x;    /* row j of the transposed state */
    const int h = blockIdx.y;    /* V-head */
    const int u = blockIdx.z;    /* slot (user) */
    const int i = threadIdx.x;   /* state coordinate within the row */
    if (i >= S)
        return;

    const int sr = rows ? rows[u] : u;  /* the slot's state row */
    float *row = s + (size_t)sr * n_v_heads * S * S + ((size_t)h * S + j) * S;
    const float *kh = k + (size_t)u * n_v_heads * S + (size_t)h * S;
    const float *qh = q + (size_t)u * n_v_heads * S + (size_t)h * S;

    const float decay = expf(g[(size_t)u * n_v_heads + h]);
    const float beta_h = beta[(size_t)u * n_v_heads + h];

    float ri = row[i] * decay;

    sdata[i] = ri * kh[i];
    __syncthreads();
    for (int stride = S / 2; stride > 0; stride >>= 1) {
        if (i < stride)
            sdata[i] += sdata[i + stride];
        __syncthreads();
    }
    float pred = sdata[0];
    float delta = (v[(size_t)u * n_v_heads * S + (size_t)h * S + j] - pred)
                  * beta_h;
    __syncthreads();

    ri += kh[i] * delta;
    row[i] = ri;

    sdata[i] = ri * qh[i];
    __syncthreads();
    for (int stride = S / 2; stride > 0; stride >>= 1) {
        if (i < stride)
            sdata[i] += sdata[i + stride];
        __syncthreads();
    }
    if (i == 0)
        y[(size_t)u * n_v_heads * S + (size_t)h * S + j] = scale * sdata[0];
}

/* ================================================================
 * T4: 3-pass -> 2-pass state-access factorization (fable2 member T4).
 *
 * The baseline fc_ssm_recurrence_slots touches the recurrent state twice in
 * global memory (one read of row[i], one write), but it runs TWO shared-memory
 * tree reductions per (h,j) block separated by a read-after-write barrier:
 *   R1: pred = sum_i (decay*row_prev[i]) * k[i]     -> delta
 *   R2: y[j] = scale * sum_i row_new[i] * q[i]       (row_new needs delta)
 * R2 is data-dependent on R1 (it reads the just-written row), so on Kepler the
 * two length-S reductions serialize behind ~2*log2(S) __syncthreads barriers
 * plus a second LDS sweep — the dominant per-block cost.
 *
 * The T4 algebraic identity  S_t^T q = g*(S_{t-1}^T q) + (q^T k)*delta  lets the
 * OUTPUT ride the SAME read pass as retrieval: with row_dec[i] = decay*row_prev[i],
 *   y[j] = scale * sum_i (row_dec[i] + k[i]*delta) * q[i]
 *        = scale * ( sum_i row_dec[i]*q[i]  +  delta * sum_i k[i]*q[i] )
 *        = scale * ( oq[j]                  +  delta * qk )
 * where qk = sum_i k[i]*q[i] is a scalar shared across all rows j of the head.
 * All three partials (pred = sum ri*k, oq = sum ri*q, qk = sum k*q) are reduced
 * together in ONE fused pass ([3*S] shared), removing the second reduction and
 * its read-after-write barrier. State traffic is unchanged (still one read, one
 * write); the win is the eliminated LDS sweep + barriers.
 *
 * LOSSY-tiny (reassociation class): the baseline forms y as one fused dot over
 * row_new[i]; this forms it as two separate sums (oq + delta*qk). The result
 * differs only by FP associativity of  sum((a+b)*q)  vs  sum(a*q)+sum(b*q)  —
 * same magnitude as any tile/reduction-order change, oracle-diff gated. Guarded
 * by QWEN35_SSM_2PASS (default OFF); the baseline kernel stays the bit-exact
 * serve path. Grid/block identical to fc_ssm_recurrence_slots. */
__global__ void fc_ssm_recurrence_slots_2pass(float *s, const float *q,
                                              const float *k, const float *v,
                                              const float *g, const float *beta,
                                              float *y, int state_size,
                                              int n_v_heads, float scale,
                                              const int *rows) {
    extern __shared__ float sdata[];     /* [3*S]: pred | oq | qk partials */

    const int S = state_size;
    const int j = blockIdx.x;    /* row j of the transposed state */
    const int h = blockIdx.y;    /* V-head */
    const int u = blockIdx.z;    /* slot (user) */
    const int i = threadIdx.x;   /* state coordinate within the row */
    if (i >= S)
        return;

    const int sr = rows ? rows[u] : u;  /* the slot's state row */
    float *row = s + (size_t)sr * n_v_heads * S * S + ((size_t)h * S + j) * S;
    const float *kh = k + (size_t)u * n_v_heads * S + (size_t)h * S;
    const float *qh = q + (size_t)u * n_v_heads * S + (size_t)h * S;

    const float decay = expf(g[(size_t)u * n_v_heads + h]);
    const float beta_h = beta[(size_t)u * n_v_heads + h];

    /* Decay in place (per-element, same as baseline -> bit-identical row_dec). */
    const float ri = row[i] * decay;
    const float ki = kh[i];
    const float qi = qh[i];

    /* ONE fused reduction: pred=sum ri*k, oq=sum ri*q, qk=sum k*q. */
    float *p_pred = sdata;
    float *p_oq   = sdata + S;
    float *p_qk   = sdata + 2 * S;
    p_pred[i] = ri * ki;
    p_oq[i]   = ri * qi;
    p_qk[i]   = ki * qi;
    __syncthreads();
    for (int stride = S / 2; stride > 0; stride >>= 1) {
        if (i < stride) {
            p_pred[i] += p_pred[i + stride];
            p_oq[i]   += p_oq[i + stride];
            p_qk[i]   += p_qk[i + stride];
        }
        __syncthreads();
    }
    const float pred = p_pred[0];
    const float oq   = p_oq[0];
    const float qk   = p_qk[0];

    const float delta = (v[(size_t)u * n_v_heads * S + (size_t)h * S + j] - pred)
                        * beta_h;

    /* rank-1 write (state carries forward, byte-layout identical). */
    row[i] = ri + ki * delta;

    /* T4 output: y[j] = scale*(oq + qk*delta) — no second reduction. */
    if (i == 0)
        y[(size_t)u * n_v_heads * S + (size_t)h * S + j] = scale * (oq + qk * delta);
}

/* ================================================================
 * Fused single-step GDN for the decode SLOTS path: the beta sigmoid, the decay
 * gate and the delta-rule recurrence in ONE launch, warp-per-row.
 *
 * Replaces fc_sigmoid(beta) + fc_ssm_decay_gate_nx + fc_ssm_recurrence_slots.
 * The recurrence was 63% of ssm_slots on the decode probe (~160 us per layer
 * for a 64 KB/head state that a bandwidth floor puts near 1/6 of that): a
 * 128-thread block per (row, head) runs two shared-memory tree reductions
 * behind ~16 __syncthreads. Here a WARP owns a row: 4 state elements per lane,
 * both dots reduced by fc_ssm_warp_tree_dot, which reproduces the baseline's
 * binary tree pairing exactly (same proof as fc_ssm_recurrence_seq_warp, which
 * is bit-identical to fc_ssm_recurrence_seq on the prefill path).
 *
 * The folded prologue is the two glue kernels' expressions verbatim, evaluated
 * per block for its head instead of written to s_beta / s_g and re-read:
 *   beta  = 1 / (1 + expf(-beta_raw))                       (fc_sigmoid)
 *   g     = ssm_a * (v > 20 ? v : log1pf(expf(v))), v = alpha + ssm_dt
 *   decay = expf(g)                                (fc_ssm_recurrence_slots)
 * BIT-EXACT vs the three-launch chain, carried state included.
 *
 * Specialised to state_size == 128 (block-uniform guard; the launcher keeps
 * the old chain otherwise).
 * Grid:  (128 / FC_SSM_WARP_COLS, n_v_heads, K) = (32, 32, K)
 * Block: FC_SSM_WARP_COLS * 32 = 128 threads (one warp per state row j).
 * ================================================================ */
__global__ void fc_ssm_gdn_step_slots_warp(float *s, const float *q,
                                           const float *k, const float *v,
                                           const float *alpha,
                                           const float *ssm_dt,
                                           const float *ssm_a,
                                           const float *beta_raw, float *y,
                                           int state_size, int n_v_heads,
                                           float scale, const int *rows) {
    if (state_size != FC_SSM_WARP_S)
        return;

    const int S = FC_SSM_WARP_S;
    const int lane = (int)(threadIdx.x & 31u);
    const int j = blockIdx.x * FC_SSM_WARP_COLS + (int)(threadIdx.x >> 5);
    const int h = blockIdx.y;
    const int u = blockIdx.z;
    const size_t hu = (size_t)u * n_v_heads + h;

    const float braw = beta_raw[hu];
    const float beta_h = 1.0f / (1.0f + expf(-braw));
    const float gv = alpha[hu] + ssm_dt[h];
    const float sp = (gv > 20.0f) ? gv : log1pf(expf(gv));
    const float decay = expf(ssm_a[h] * sp);

    const int sr = rows ? rows[u] : u;  /* the slot's state row */
    float *row = s + (size_t)sr * n_v_heads * S * S + ((size_t)h * S + j) * S;
    const size_t vec = hu * S;
    const float *kh = k + vec + lane;
    const float *qh = q + vec + lane;

    float r[FC_SSM_WARP_ELEMS], kv[FC_SSM_WARP_ELEMS], qv[FC_SSM_WARP_ELEMS];
#pragma unroll
    for (int e = 0; e < FC_SSM_WARP_ELEMS; e++) {
        r[e] = __fmul_rn(row[lane + e * 32], decay);
        kv[e] = kh[e * 32];
        qv[e] = qh[e * 32];
    }
    const float pred = fc_ssm_warp_tree_dot(r, kv);
    const float delta = __fmul_rn(__fsub_rn(v[vec + j], pred), beta_h);
#pragma unroll
    for (int e = 0; e < FC_SSM_WARP_ELEMS; e++) {
        r[e] = __fmaf_rn(kv[e], delta, r[e]);
        row[lane + e * 32] = r[e];
    }
    const float out = fc_ssm_warp_tree_dot(r, qv);
    if (lane == 0)
        y[vec + j] = __fmul_rn(out, scale);
}

/* ================================================================
 * 8-LANE LAYOUT of the two warp kernels above (fc_ssm_recurrence_seq_warp8,
 * fc_ssm_gdn_step_slots_warp8). Measured on one K80 die against the 32-lane
 * kernels: seq 1.28x at n = 1, 1.37-1.42x at n = 2, 1.59-1.65x at n = 4;
 * slots 1.26-1.30x at K = 1-4 (16 and 32 heads).
 *
 * 8 lanes per state column, so a warp steps 4 columns and a 128-thread block
 * 16. Lane l (= lane & 7) of a column holds the 16 elements i = 4l + c + 32e
 * (c, e = 0..3) as 4 float4s, so state, k and q move as 128-bit accesses (the
 * host checks the pointers are 16-B aligned). The dot keeps the baseline tree
 * exactly: strides 64 and 32 are e pairs inside the lane, 16 / 8 / 4 the lane
 * bits (a __shfl_xor butterfly inside the column's 8 lanes), 2 and 1 the
 * float4 components. Same tree, same operand order, same contraction-barrier
 * intrinsics and expf, so the same bits as the 32-lane kernels.
 * FC_SSM_W8_RED is a gate-only red build: the state update rounded twice.
 * ================================================================ */
#define FC_SSM_W8_LANES 8   /* lanes per state column */
#define FC_SSM_W8_COLS  16  /* state columns per 128-thread block */

__device__ __forceinline__ float4 fc_ssm_f4_mul(float4 a, float4 b) {
    return make_float4(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y),
                       __fmul_rn(a.z, b.z), __fmul_rn(a.w, b.w));
}

__device__ __forceinline__ float4 fc_ssm_f4_add(float4 a, float4 b) {
    return make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y),
                       __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
}

__device__ __forceinline__ float4 fc_ssm_f4_scale(float4 a, float s) {
    return make_float4(__fmul_rn(a.x, s), __fmul_rn(a.y, s),
                       __fmul_rn(a.z, s), __fmul_rn(a.w, s));
}

/* r += k * delta, one rounding (the red build rounds the product first). */
__device__ __forceinline__ float4 fc_ssm_f4_update(float4 k, float delta,
                                                   float4 r) {
#ifdef FC_SSM_W8_RED
    return fc_ssm_f4_add(fc_ssm_f4_scale(k, delta), r);
#else
    return make_float4(__fmaf_rn(k.x, delta, r.x), __fmaf_rn(k.y, delta, r.y),
                       __fmaf_rn(k.z, delta, r.z), __fmaf_rn(k.w, delta, r.w));
#endif
}

/* fc_ssm_warp_tree_dot in the 8-lane layout: the same binary tree over
 * i = 4l + c + 32e. Returns the identical fp32 bits in the column's 8 lanes. */
__device__ __forceinline__ float fc_ssm_w8_tree_dot(const float4 (&r)[4],
                                                    const float4 (&x)[4]) {
    float4 p0 = fc_ssm_f4_mul(r[0], x[0]), p1 = fc_ssm_f4_mul(r[1], x[1]);
    const float4 p2 = fc_ssm_f4_mul(r[2], x[2]), p3 = fc_ssm_f4_mul(r[3], x[3]);
    p0 = fc_ssm_f4_add(p0, p2);          /* stride 64 */
    p1 = fc_ssm_f4_add(p1, p3);
    p0 = fc_ssm_f4_add(p0, p1);          /* stride 32 */
#pragma unroll
    for (int st = 4; st > 0; st >>= 1) { /* strides 16, 8, 4 */
        p0.x = __fadd_rn(p0.x, __shfl_xor(p0.x, st));
        p0.y = __fadd_rn(p0.y, __shfl_xor(p0.y, st));
        p0.z = __fadd_rn(p0.z, __shfl_xor(p0.z, st));
        p0.w = __fadd_rn(p0.w, __shfl_xor(p0.w, st));
    }
    return __fadd_rn(__fadd_rn(p0.x, p0.z), __fadd_rn(p0.y, p0.w)); /* 2, 1 */
}

__device__ __forceinline__ void fc_ssm_w8_load(float4 (&dst)[4], const float *p) {
#pragma unroll
    for (int e = 0; e < 4; e++)
        dst[e] = *(const float4 *)(p + 32 * e);
}

__device__ __forceinline__ void fc_ssm_w8_store(float *p, const float4 (&src)[4]) {
#pragma unroll
    for (int e = 0; e < 4; e++)
        *(float4 *)(p + 32 * e) = src[e];
}

/* fc_ssm_recurrence_seq_warp in the 8-lane layout. Same arguments and shared
 * staging. Grid (state_size / FC_SSM_W8_COLS, n_v_heads), block 128. */
__global__ void fc_ssm_recurrence_seq_warp8(float *s, const float *q,
                                            const float *k, const float *v,
                                            const float *g, const float *beta,
                                            float *y, int state_size,
                                            int n_v_heads, int n, float scale,
                                            float *snaps) {
    extern __shared__ float sh[];        /* [2*n]: decay[n] | beta[n] */

    if (state_size != FC_SSM_WARP_S)     /* block-uniform */
        return;

    const int S = FC_SSM_WARP_S;
    const int lane = (int)(threadIdx.x & 31u);
    const int l = lane & (FC_SSM_W8_LANES - 1);
    const int j = (blockIdx.x * 4 + (int)(threadIdx.x >> 5)) * 4 + (lane >> 3);
    const int h = blockIdx.y;

    float *sh_decay = sh;
    float *sh_beta = sh + n;
    for (int t = (int)threadIdx.x; t < n; t += 128) {
        sh_decay[t] = expf(g[(size_t)t * n_v_heads + h]);
        sh_beta[t] = beta[(size_t)t * n_v_heads + h];
    }
    __syncthreads();

    if (j >= S)
        return;

    const size_t row_off = ((size_t)h * S + j) * S;
    const size_t state_total = (size_t)n_v_heads * S * S;
    const size_t pos_stride = (size_t)n_v_heads * S;
    float *row = s + row_off + 4 * l;
    const float *kh = k + (size_t)h * S + 4 * l;
    const float *qh = q + (size_t)h * S + 4 * l;

    float4 r[4];
    fc_ssm_w8_load(r, row);
    for (int t = 0; t < n; t++) {
        const size_t toff = (size_t)t * pos_stride;
        const float decay = sh_decay[t];
        const float beta_h = sh_beta[t];
        const float vj = v[toff + (size_t)h * S + j];
        float4 kv[4], qv[4];
        fc_ssm_w8_load(kv, kh + toff);
        fc_ssm_w8_load(qv, qh + toff);
#pragma unroll
        for (int e = 0; e < 4; e++)
            r[e] = fc_ssm_f4_scale(r[e], decay);
        const float pred = fc_ssm_w8_tree_dot(r, kv);
        const float delta = __fmul_rn(__fsub_rn(vj, pred), beta_h);
#pragma unroll
        for (int e = 0; e < 4; e++)
            r[e] = fc_ssm_f4_update(kv[e], delta, r[e]);
        if (snaps)
            fc_ssm_w8_store(snaps + (size_t)t * state_total + row_off + 4 * l, r);
        const float out = fc_ssm_w8_tree_dot(r, qv);
        if (l == 0)
            y[toff + (size_t)h * S + j] = __fmul_rn(out, scale);
    }
    fc_ssm_w8_store(row, r);
}

/* fc_ssm_gdn_step_slots_warp in the 8-lane layout: the per-slot gates are
 * computed once per thread for its column instead of once per 32 lanes.
 * Grid (state_size / FC_SSM_W8_COLS, n_v_heads, nslots), block 128. */
__global__ void fc_ssm_gdn_step_slots_warp8(float *s, const float *q,
                                            const float *k, const float *v,
                                            const float *alpha,
                                            const float *ssm_dt,
                                            const float *ssm_a,
                                            const float *beta_raw, float *y,
                                            int state_size, int n_v_heads,
                                            float scale, const int *rows) {
    if (state_size != FC_SSM_WARP_S)
        return;

    const int S = FC_SSM_WARP_S;
    const int lane = (int)(threadIdx.x & 31u);
    const int l = lane & (FC_SSM_W8_LANES - 1);
    const int j = (blockIdx.x * 4 + (int)(threadIdx.x >> 5)) * 4 + (lane >> 3);
    const int h = blockIdx.y;
    const int u = blockIdx.z;
    const size_t hu = (size_t)u * n_v_heads + h;

    const float braw = beta_raw[hu];
    const float beta_h = 1.0f / (1.0f + expf(-braw));
    const float gv = alpha[hu] + ssm_dt[h];
    const float sp = (gv > 20.0f) ? gv : log1pf(expf(gv));
    const float decay = expf(ssm_a[h] * sp);

    const int sr = rows ? rows[u] : u;  /* the slot's state row */
    float *row = s + (size_t)sr * n_v_heads * S * S + ((size_t)h * S + j) * S
                 + 4 * l;
    const size_t vec = hu * S;

    float4 r[4], kv[4], qv[4];
    fc_ssm_w8_load(r, row);
    fc_ssm_w8_load(kv, k + vec + 4 * l);
    fc_ssm_w8_load(qv, q + vec + 4 * l);
#pragma unroll
    for (int e = 0; e < 4; e++)
        r[e] = fc_ssm_f4_scale(r[e], decay);
    const float pred = fc_ssm_w8_tree_dot(r, kv);
    const float delta = __fmul_rn(__fsub_rn(v[vec + j], pred), beta_h);
#pragma unroll
    for (int e = 0; e < 4; e++)
        r[e] = fc_ssm_f4_update(kv[e], delta, r[e]);
    fc_ssm_w8_store(row, r);
    const float out = fc_ssm_w8_tree_dot(r, qv);
    if (l == 0)
        y[vec + j] = __fmul_rn(out, scale);
}

/* Decay gate over n positions: g[t][h] = ssm_a[h] * softplus(alpha[t][h] +
 * ssm_dt[h]). One thread per (t, h); n_heads gives the per-position head
 * count for the broadcast vectors. total = n * n_heads. */
__global__ void fc_ssm_decay_gate_nx(const float *alpha, const float *ssm_dt,
                                     const float *ssm_a, float *g,
                                     int n_heads, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    int h = idx % n_heads;
    float v = alpha[idx] + ssm_dt[h];
    float sp = (v > 20.0f) ? v : log1pf(expf(v));
    g[idx] = ssm_a[h] * sp;
}

/* q/k L2-norm + GQA repeat over n positions (grid.y = position), with the v
 * passthrough folded in (each block copies its val_dim / n_k_heads slice of
 * v — a plain copy, no fp math, replacing the launcher's DtoD). */
__global__ void fc_ssm_qknorm_gqa_nx(const float *conv_out, float *qg,
                                     float *kg, float *vg, int n_k_heads,
                                     int kv_groups, int S, int key_dim,
                                     int val_dim, float eps) {
    int kh = blockIdx.x;
    int t = blockIdx.y;
    if (kh >= n_k_heads)
        return;
    int i = threadIdx.x;
    extern __shared__ float sh[];
    int conv_dim = 2 * key_dim + val_dim;
    const float *q_src = conv_out + (size_t)t * conv_dim;
    const float *k_src = q_src + key_dim;
    const float *v_src = q_src + 2 * key_dim;
    float *qg_t = qg + (size_t)t * val_dim;
    float *kg_t = kg + (size_t)t * val_dim;
    float *vg_t = vg + (size_t)t * val_dim;

    float qv = q_src[(size_t)kh * S + i];
    float kv = k_src[(size_t)kh * S + i];
    sh[i] = qv * qv;
    sh[blockDim.x + i] = kv * kv;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (i < stride) {
            sh[i] += sh[i + stride];
            sh[blockDim.x + i] += sh[blockDim.x + i + stride];
        }
        __syncthreads();
    }
    float inv_q = 1.0f / fmaxf(sqrtf(sh[0]), eps);
    float inv_k = 1.0f / fmaxf(sqrtf(sh[blockDim.x]), eps);
    float qn = qv * inv_q;
    float kn = kv * inv_k;
    for (int r = 0; r < kv_groups; r++) {
        int vh = r * n_k_heads + kh;
        qg_t[(size_t)vh * S + i] = qn;
        kg_t[(size_t)vh * S + i] = kn;
    }
    /* v passthrough: this block copies slice kh of val_dim. */
    int slice = val_dim / n_k_heads;
    for (int e = i; e < slice; e += blockDim.x)
        vg_t[(size_t)kh * slice + e] = v_src[(size_t)kh * slice + e];
}

} /* extern "C" */
