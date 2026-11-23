/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_ssm_recurrence_seq.cu
 * @brief Bit-exactness + timing harness for the prefill Gated-DeltaNet
 *   recurrence: fc_ssm_recurrence_seq (shipped baseline) vs
 *   fc_ssm_recurrence_seq_warp (warp-shuffle rewrite) vs
 *   fc_ssm_recurrence_seq_unroll128 (the cheap fallback: same barrier-chained
 *   sweep, literal trip count so ptxas unrolls it). Includes
 *   ssm_gated_delta.cu VERBATIM so every kernel is a control symbol compiled
 *   from the same source in the same module.
 *
 * Hypothesis under test: fc_ssm_recurrence_seq is ISSUE-BOUND ON OVERHEAD, not
 * bandwidth-, compute-, launch- or occupancy-bound. From its committed PTX,
 * per (block, position) it issues ~740 warp-instruction slots of which 66 % is
 * the two shared-memory tree reductions (45 pp of that being setp/bra/shr/mov
 * loop control, because the 7-step loop's bound is `state_size`, a RUNTIME
 * kernel parameter, so ptxas cannot unroll it), 10 % is bar.sync, 10 % is a
 * 16384x-redundant expf on a block-uniform value, and only ~5.5 % does useful
 * floating-point work. Occupancy is already 100 % (512 B shared, 128-thread
 * blocks, 16 blocks/SMX) and the grid is 4096 blocks, so barrier LATENCY is
 * already hidden -- what is not hidden is the instruction COUNT.
 *
 * fc_ssm_recurrence_seq_warp cuts that to ~72 slots per (column, position) and
 * removes __syncthreads from the position loop entirely. If the hypothesis is
 * right, the measured ratio tracks the instruction ratio far better than it
 * tracks any byte count (this project has repeatedly seen byte reduction NOT
 * convert while instruction reduction did).
 *
 * WHAT WOULD REFUTE IT: a ratio near 1.0x despite the 10x instruction cut. That
 * would mean the kernel is bound by something neither variant changes -- global
 * load latency on the k/q streams, or the L1/read-only cache request rate (both
 * variants issue the same 1 KB of q+k requests per column-position). In that
 * case report the refutation; do not tune the warp kernel to chase it.
 *
 * NEGATIVE CONTROLS (this harness must be able to FAIL before green means
 * anything -- see the repo's "harnesses here measure nothing" lesson):
 *   1. `--self` compares the baseline against a second run of the BASELINE.
 *      It must print EXACT for y and s, and a ratio inside the noise floor.
 *      If --self ever prints DIFFER, the comparison is broken (uninitialised
 *      buffer, wrong length, aliasing) and every other verdict is void.
 *   2. fc_ssm_recurrence_seq_warp_poison is the warp kernel with ONE deliberate
 *      defect: the intra-lane reduction accumulates sequentially
 *      (((p0+p1)+p2)+p3) instead of reproducing the baseline's tree
 *      ((p0+p2)+(p1+p3)). It is mathematically the same sum and numerically
 *      within 1e-7, so a tolerance-based check passes it. This harness MUST
 *      report it as DIFFER. If it reports EXACT, the bit-comparison is not
 *      actually comparing bits and the bit-exactness claim for the real kernel
 *      is unproven.
 *
 * Run on a K80 host (nvcc 11.4, gcc-11 host, sm_37), ONE die, no sibling load --
 * two busy dies on one K80 board drop to a 398 MHz power cap:
 *   nvcc -arch=sm_37 -ccbin gcc-11 -Wno-deprecated-gpu-targets \
 *        -isystem ../nvcc_shim -O3 -Xptxas -v \
 *        -o bench_ssm_recurrence_seq bench_ssm_recurrence_seq.cu
 *   CUDA_VISIBLE_DEVICES=<free die> ./bench_ssm_recurrence_seq --self
 *   CUDA_VISIBLE_DEVICES=<free die> ./bench_ssm_recurrence_seq
 *   CUDA_VISIBLE_DEVICES=<free die> ./bench_ssm_recurrence_seq --poison
 *   CUDA_VISIBLE_DEVICES=<free die> ./bench_ssm_recurrence_seq --unroll
 *   CUDA_VISIBLE_DEVICES=<free die> ./bench_ssm_recurrence_seq --snaps
 * --snaps additionally compares the per-position state SNAPSHOTS the shipped
 * launcher requests in speculative-decoding mode (spec ? d->ssm_snaps : NULL).
 * Without it that output of the warp kernel is never written, so the default
 * arms prove nothing about it; it costs 256 MiB per side, hence the flag.
 * Read the `-Xptxas -v` banner for BOTH kernels and record the STACK FRAME, not
 * the spill counters: on this project an array indexed by a runtime-bounded
 * loop is never promoted, and ptxas prints "0 bytes spill" right beside a live
 * stack frame. The warp kernel must show 0 bytes stack frame and <= 64
 * registers -- above 64 the register file caps residency below 16 blocks/SMX
 * and the projected multiplier does not hold.
 *
 * @status Microbenchmark only. Nothing here is reachable from Pantheon; the
 *   shipped launcher keeps fc_ssm_recurrence_seq unless QWEN35_SSM_WARP=1.
 *   RUN 2026-08-10, die 6 at 875 MHz, 0.10 % noise floor: --self EXACT,
 *   --poison DIFFER at 1.1e-08, warp 8.167x bit-exact, unroll128 1.189x
 *   bit-exact, 32 registers / 0 B stack frame for all three.
 * @issues
 *   - The warp arm requires state_size == 128 (FC_SSM_WARP_S) and stages
 *     2*npos floats of shared memory, so npos must stay small (128 is the
 *     shipped prefill chunk).
 *   - Timing here is a per-kernel ratio on synthetic operands. It does NOT
 *     predict whole-prefill tok/s: use it to accept or reject the binding
 *     hypothesis, then re-measure end to end with QWEN35_PROF=1
 *     QWEN35_PROF_SUB=1 and read the `recur_seq` span.
 * @todo None
 */

#include "../ssm_gated_delta.cu"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Shipped prefill shape for qwen35moe / nex-n2-mini / saber-orch. */
#define BENCH_STATE_SIZE 128
#define BENCH_N_V_HEADS  32
#define BENCH_NPOS       128
#define BENCH_WARP_COLS  4

/*
 * NEGATIVE CONTROL. fc_ssm_recurrence_seq_warp with exactly one defect: the
 * intra-lane reduction accumulates left-to-right instead of reproducing the
 * baseline's binary tree. Same value to ~1e-7, different bits. Any check that
 * calls this bit-exact is not comparing bits.
 */
__device__ __forceinline__ float fc_ssm_warp_seq_dot_poison(const float *r,
                                                            const float *x) {
    float acc = __fmul_rn(r[0], x[0]);
#pragma unroll
    for (int e = 1; e < FC_SSM_WARP_ELEMS; e++)
        acc = __fadd_rn(acc, __fmul_rn(r[e], x[e]));
#pragma unroll
    for (int st = 16; st > 0; st >>= 1)
        acc = __fadd_rn(acc, __shfl_xor(acc, st));
    return acc;
}

extern "C" __global__ void fc_ssm_recurrence_seq_warp_poison(
        float *s, const float *q, const float *k, const float *v,
        const float *g, const float *beta, float *y, int state_size,
        int n_v_heads, int n, float scale, float *snaps) {
    extern __shared__ float sh[];

    if (state_size != FC_SSM_WARP_S)
        return;

    const int S = FC_SSM_WARP_S;
    const int lane = (int)(threadIdx.x & 31u);
    const int col = (int)(threadIdx.x >> 5);
    const int j = blockIdx.x * FC_SSM_WARP_COLS + col;
    const int h = blockIdx.y;

    float *sh_decay = sh;
    float *sh_beta = sh + n;
    for (int t = (int)threadIdx.x; t < n; t += FC_SSM_WARP_COLS * 32) {
        sh_decay[t] = expf(g[(size_t)t * n_v_heads + h]);
        sh_beta[t] = beta[(size_t)t * n_v_heads + h];
    }
    __syncthreads();

    if (j >= S)
        return;

    const size_t row_off = ((size_t)h * S + j) * S;
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
    (void)snaps;

    for (int t = 0; t < n; t++) {
        const size_t toff = (size_t)t * pos_stride;
        const float decay = sh_decay[t];
        const float beta_h = sh_beta[t];
        float kv[FC_SSM_WARP_ELEMS], qv[FC_SSM_WARP_ELEMS];
#pragma unroll
        for (int e = 0; e < FC_SSM_WARP_ELEMS; e++) {
            kv[e] = kh[toff + e * 32];
            r[e] = __fmul_rn(r[e], decay);
        }
        const float pred = fc_ssm_warp_seq_dot_poison(r, kv);
        const float delta = __fmul_rn(__fsub_rn(vj[toff], pred), beta_h);
#pragma unroll
        for (int e = 0; e < FC_SSM_WARP_ELEMS; e++) {
            r[e] = __fmaf_rn(kv[e], delta, r[e]);
            qv[e] = qh[toff + e * 32];
        }
        const float out = fc_ssm_warp_seq_dot_poison(r, qv);
        if (lane == 0)
            yj[toff] = __fmul_rn(out, scale);
    }

#pragma unroll
    for (int e = 0; e < FC_SSM_WARP_ELEMS; e++)
        row[lane + e * 32] = r[e];
}

/* ---------------------------------------------------------------- host ---- */

#define CHECK(call)                                                           \
    do {                                                                      \
        cudaError_t rc_ = (call);                                             \
        if (rc_ != cudaSuccess) {                                             \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,    \
                    cudaGetErrorString(rc_));                                 \
            return 1;                                                        \
        }                                                                     \
    } while (0)

/* Deterministic operands: both arms must see byte-identical inputs or the
 * bit-comparison is meaningless. */
static void fill_deterministic(float *p, size_t n, unsigned seed, float lo,
                               float hi) {
    unsigned st = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)(st >> 8) / (float)(1u << 24);
        p[i] = lo + u * (hi - lo);
    }
}

/* L2-normalise each [n_v_heads][S] row of a [npos][n_v_heads*S] stream, as the
 * qk-norm kernel does upstream, so the recurrence sees realistic magnitudes. */
static void l2_norm_rows(float *p, int npos, int n_v_heads, int S) {
    for (int t = 0; t < npos; t++) {
        for (int h = 0; h < n_v_heads; h++) {
            float *row = p + ((size_t)t * n_v_heads + h) * S;
            float ss = 0.0f;
            for (int i = 0; i < S; i++)
                ss += row[i] * row[i];
            float inv = 1.0f / fmaxf(sqrtf(ss), 1e-6f);
            for (int i = 0; i < S; i++)
                row[i] *= inv;
        }
    }
}

enum bench_arm { ARM_BASE = 0, ARM_WARP, ARM_POISON, ARM_UNROLL };

int main(int argc, char **argv) {
    int self_mode = 0, poison_mode = 0, unroll_mode = 0, snaps_mode = 0, reps = 30;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--self"))
            self_mode = 1;
        else if (!strcmp(argv[i], "--poison"))
            poison_mode = 1;
        else if (!strcmp(argv[i], "--unroll"))
            unroll_mode = 1;
        else if (!strcmp(argv[i], "--snaps"))
            snaps_mode = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
    }

    /* Exactly two timed arms per run: the baseline, then whichever variant the
     * mode selects. Two arms means the comparison buffers cannot be cross-wired
     * between modes, which is how a negative control gets quietly disarmed. */
    const enum bench_arm cmp_arm = self_mode   ? ARM_BASE
                                 : poison_mode ? ARM_POISON
                                 : unroll_mode ? ARM_UNROLL
                                               : ARM_WARP;

    const int S = BENCH_STATE_SIZE, H = BENCH_N_V_HEADS, N = BENCH_NPOS;
    const float scale = 1.0f / sqrtf((float)S);
    const size_t n_state = (size_t)H * S * S;
    const size_t n_act = (size_t)N * H * S;
    const size_t n_head = (size_t)N * H;

    float *h_s0 = (float *)malloc(n_state * sizeof(float));
    float *h_q = (float *)malloc(n_act * sizeof(float));
    float *h_k = (float *)malloc(n_act * sizeof(float));
    float *h_v = (float *)malloc(n_act * sizeof(float));
    float *h_g = (float *)malloc(n_head * sizeof(float));
    float *h_b = (float *)malloc(n_head * sizeof(float));
    float *h_ya = (float *)malloc(n_act * sizeof(float));
    float *h_yb = (float *)malloc(n_act * sizeof(float));
    float *h_sa = (float *)malloc(n_state * sizeof(float));
    float *h_sb = (float *)malloc(n_state * sizeof(float));
    if (!h_s0 || !h_q || !h_k || !h_v || !h_g || !h_b || !h_ya || !h_yb ||
        !h_sa || !h_sb) {
        fprintf(stderr, "host alloc failed\n");
        return 1;
    }

    fill_deterministic(h_s0, n_state, 11u, -0.5f, 0.5f);
    fill_deterministic(h_q, n_act, 22u, -1.0f, 1.0f);
    fill_deterministic(h_k, n_act, 33u, -1.0f, 1.0f);
    fill_deterministic(h_v, n_act, 44u, -1.0f, 1.0f);
    l2_norm_rows(h_q, N, H, S);
    l2_norm_rows(h_k, N, H, S);
    /* g is pre-exp and negative (decay in (0,1]); beta is post-sigmoid. */
    fill_deterministic(h_g, n_head, 55u, -0.30f, -0.001f);
    fill_deterministic(h_b, n_head, 66u, 0.05f, 0.95f);

    /* --snaps exercises the speculative-decoding output the shipped launcher
     * passes when spec mode is on. It is 256 MiB per side at this shape and is
     * NOT part of a timed arm's workload, so it is allocated only on request. */
    const size_t n_snaps = (size_t)N * n_state;
    float *h_sna = NULL, *h_snb = NULL, *d_snaps = NULL;
    if (snaps_mode) {
        h_sna = (float *)malloc(n_snaps * sizeof(float));
        h_snb = (float *)malloc(n_snaps * sizeof(float));
        if (!h_sna || !h_snb) {
            fprintf(stderr, "host alloc failed (snaps)\n");
            return 1;
        }
    }

    float *d_s, *d_q, *d_k, *d_v, *d_g, *d_b, *d_y;
    CHECK(cudaMalloc(&d_s, n_state * sizeof(float)));
    CHECK(cudaMalloc(&d_q, n_act * sizeof(float)));
    CHECK(cudaMalloc(&d_k, n_act * sizeof(float)));
    CHECK(cudaMalloc(&d_v, n_act * sizeof(float)));
    CHECK(cudaMalloc(&d_g, n_head * sizeof(float)));
    CHECK(cudaMalloc(&d_b, n_head * sizeof(float)));
    CHECK(cudaMalloc(&d_y, n_act * sizeof(float)));
    if (snaps_mode)
        CHECK(cudaMalloc(&d_snaps, n_snaps * sizeof(float)));
    CHECK(cudaMemcpy(d_q, h_q, n_act * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_k, h_k, n_act * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, h_v, n_act * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_g, h_g, n_head * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, n_head * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    float ms[2] = { 0.0f, 0.0f };

    for (int a = 0; a < 2; a++) {
        const enum bench_arm arm = (a == 0) ? ARM_BASE : cmp_arm;

        /* Only the warp variants change the launch geometry: one warp per state
         * column instead of one block. The unrolled arm is the baseline's own
         * geometry. */
        int warp_geom = (arm == ARM_WARP || arm == ARM_POISON);
        unsigned gx = warp_geom ? (unsigned)(S / BENCH_WARP_COLS) : (unsigned)S;
        unsigned bx = warp_geom ? (unsigned)(BENCH_WARP_COLS * 32) : (unsigned)S;
        unsigned shb = warp_geom ? (unsigned)(2 * N * sizeof(float))
                                 : (unsigned)(S * sizeof(float));

#define LAUNCH_ARM()                                                          \
        do {                                                                  \
            if (arm == ARM_BASE)                                              \
                fc_ssm_recurrence_seq<<<dim3(gx, H), bx, shb>>>(              \
                    d_s, d_q, d_k, d_v, d_g, d_b, d_y, S, H, N, scale, d_snaps);  \
            else if (arm == ARM_WARP)                                         \
                fc_ssm_recurrence_seq_warp<<<dim3(gx, H), bx, shb>>>(         \
                    d_s, d_q, d_k, d_v, d_g, d_b, d_y, S, H, N, scale, d_snaps);  \
            else if (arm == ARM_UNROLL)                                       \
                fc_ssm_recurrence_seq_unroll128<<<dim3(gx, H), bx, shb>>>(    \
                    d_s, d_q, d_k, d_v, d_g, d_b, d_y, S, H, N, scale, d_snaps);  \
            else                                                              \
                fc_ssm_recurrence_seq_warp_poison<<<dim3(gx, H), bx, shb>>>(  \
                    d_s, d_q, d_k, d_v, d_g, d_b, d_y, S, H, N, scale, d_snaps);  \
        } while (0)

        /* Pass 1 -- CORRECTNESS. Fresh state, y poisoned to 0xCD so an
         * unwritten element cannot masquerade as a match, exactly one launch. */
        CHECK(cudaMemcpy(d_s, h_s0, n_state * sizeof(float),
                         cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_y, 0xCD, n_act * sizeof(float)));
        if (snaps_mode)
            CHECK(cudaMemset(d_snaps, 0xCD, n_snaps * sizeof(float)));
        LAUNCH_ARM();
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        float *y_dst = (a == 0) ? h_ya : h_yb;
        float *s_dst = (a == 0) ? h_sa : h_sb;
        CHECK(cudaMemcpy(y_dst, d_y, n_act * sizeof(float),
                         cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(s_dst, d_s, n_state * sizeof(float),
                         cudaMemcpyDeviceToHost));
        if (snaps_mode)
            CHECK(cudaMemcpy((a == 0) ? h_sna : h_snb, d_snaps,
                             n_snaps * sizeof(float), cudaMemcpyDeviceToHost));

        /* Pass 2 -- TIMING. Nothing but back-to-back launches inside the span:
         * a per-iteration state reset would put ~4 MiB of PCIe traffic and a
         * full device sync in the span and drag every ratio toward 1.00x. The
         * state is reset once before the loop and then allowed to advance; with
         * decay in (0,1) it stays bounded, and the instruction stream is
         * identical whatever the operand values are. */
        CHECK(cudaMemcpy(d_s, h_s0, n_state * sizeof(float),
                         cudaMemcpyHostToDevice));
        for (int w = 0; w < 3; w++)
            LAUNCH_ARM();
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(ev0));
        for (int it = 0; it < reps; it++)
            LAUNCH_ARM();
        CHECK(cudaEventRecord(ev1));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        float total = 0.0f;
        CHECK(cudaEventElapsedTime(&total, ev0, ev1));
        ms[a] = total / (float)reps;
#undef LAUNCH_ARM
    }

    const char *arm_name = self_mode ? "baseline(self)"
                         : poison_mode ? "warp_POISON"
                         : unroll_mode ? "unroll128" : "warp";

    size_t y_diff = 0, s_diff = 0;
    double y_max = 0.0, s_max = 0.0;
    for (size_t i = 0; i < n_act; i++) {
        if (memcmp(&h_ya[i], &h_yb[i], sizeof(float)) != 0) {
            y_diff++;
            double d = fabs((double)h_ya[i] - (double)h_yb[i]);
            if (d > y_max)
                y_max = d;
        }
    }
    for (size_t i = 0; i < n_state; i++) {
        if (memcmp(&h_sa[i], &h_sb[i], sizeof(float)) != 0) {
            s_diff++;
            double d = fabs((double)h_sa[i] - (double)h_sb[i]);
            if (d > s_max)
                s_max = d;
        }
    }

    printf("shape        S=%d n_v_heads=%d npos=%d  grid_base=(%d,%d)x%d "
           "grid_warp=(%d,%d)x%d\n", S, H, N, S, H, S, S / BENCH_WARP_COLS, H,
           BENCH_WARP_COLS * 32);
    printf("baseline     %8.4f ms/launch\n", ms[0]);
    printf("%-12s %8.4f ms/launch   ratio %.4fx\n", arm_name, ms[1],
           ms[1] > 0.0f ? ms[0] / ms[1] : 0.0f);
    printf("y  bits      %s  (%zu/%zu differ, max |d| %.3e)\n",
           y_diff ? "DIFFER" : "EXACT", y_diff, n_act, y_max);
    printf("s  bits      %s  (%zu/%zu differ, max |d| %.3e)\n",
           s_diff ? "DIFFER" : "EXACT", s_diff, n_state, s_max);

    size_t sn_diff = 0;
    if (snaps_mode) {
        double sn_max = 0.0;
        for (size_t i = 0; i < n_snaps; i++) {
            if (memcmp(&h_sna[i], &h_snb[i], sizeof(float)) != 0) {
                sn_diff++;
                double d = fabs((double)h_sna[i] - (double)h_snb[i]);
                if (d > sn_max)
                    sn_max = d;
            }
        }
        printf("snaps bits   %s  (%zu/%zu differ, max |d| %.3e)\n",
               sn_diff ? "DIFFER" : "EXACT", sn_diff, n_snaps, sn_max);
    }

    int verdict_ok;
    if (self_mode) {
        verdict_ok = (y_diff == 0 && s_diff == 0);
        printf("VERDICT      control-1 %s -- baseline vs baseline must be "
               "EXACT; ratio is the noise floor for every other run\n",
               verdict_ok ? "PASS" : "FAIL (comparison is broken; all other "
                                     "verdicts are void)");
    } else if (poison_mode) {
        verdict_ok = (y_diff != 0);
        printf("VERDICT      control-2 %s -- the poison arm MUST differ; "
               "EXACT here means the bit-check is not comparing bits\n",
               verdict_ok ? "PASS" : "FAIL");
    } else {
        verdict_ok = (y_diff == 0 && s_diff == 0 && sn_diff == 0);
        printf("VERDICT      %s -- the %s arm must be bit-exact in BOTH y and s; "
               "run --self and --poison before trusting this\n",
               verdict_ok ? "PASS" : "FAIL", arm_name);
    }

    free(h_s0); free(h_q); free(h_k); free(h_v); free(h_g); free(h_b);
    free(h_ya); free(h_yb); free(h_sa); free(h_sb);
    free(h_sna); free(h_snb); cudaFree(d_snaps);
    cudaFree(d_s); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_g); cudaFree(d_b); cudaFree(d_y);
    return verdict_ok ? 0 : 2;
}
