/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file bench_moe_grid_compaction.cu
 * @brief Bit-exactness + timing gate for the COMPACTED grid of the MoE dedup
 *   gate+up: the shipped fc_moe_gateup_q3k_dedup_shx at
 *   (ceil(ff/8), n_expert, ceil(maxr/RT)) = 262,144 blocks against
 *   fc_moe_gateup_q3k_dedup_shxc at (ceil(ff/8), wmax, 1) = 22,528, over a
 *   device-built (expert, request-tile) work list. Includes
 *   moe_expert_group.cu VERBATIM with -DFC_MOE_DEDUP_COMPACT=1, so the baseline
 *   is a control symbol compiled from the same source in the same module.
 *
 * THE DEFECT UNDER TEST — MEASURED, not estimated.
 * results/moe_dedup_occupancy_probe.md §5.2: the dedup gate+up's grid.z is sized
 * off maxr (== ntok == 128) because grid dims are host-side while the per-expert
 * request counts are device-side, so grid.y*grid.z is 4096 where the router
 * actually produces ~257 (expert, tile) pairs. Only 6.3 % of the blocks do any
 * work. Launched with every ecount zero — no weight bytes, no dequant, no FMA,
 * every block exiting on its first load — the kernel still cost 3.277 ms against
 * the full kernel's 17.578 ms: 18.6 %. That measurement was taken on the plain
 * fc_moe_gateup_q3k_dedup (95 registers, 5 blocks/SMX). PRODUCTION RUNS _shx (62
 * registers, 8 blocks/SMX) and the empty-block cost has NEVER been measured on
 * it — see --empty below, which is why that arm exists.
 *
 * ★ THE QUESTION THIS HARNESS EXISTS TO SETTLE, AND IT IS NOT "IS THE KERNEL
 * FASTER". It is whether the empty-block cost is ADDITIVE or HIDDEN.
 *   - ADDITIVE if what an empty block consumes is a BLOCK SLOT: an SMX holds 8
 *     resident blocks, an empty one holds one for the latency of a single
 *     dependent global load, and slot residency is the same resource the useful
 *     blocks need. Then removing 91.4 % of the blocks returns ~17 % of the
 *     kernel.
 *   - HIDDEN if the 3.277 ms is instead the GLOBAL block-dispatch engine's
 *     throughput (262,144 / 3.277 ms = 80 M blocks/s). The full kernel needs
 *     only 14.9 M blocks/s, well inside that, so dispatch would overlap with
 *     compute entirely and compaction returns ~nothing.
 * NOTHING MEASURED SO FAR DISTINGUISHES THESE. The probe's ledger credited
 * dispatch x1.23 and left a x1.22 residual unexplained; if dispatch is hidden
 * the residual is x1.50 and the ledger reads the same. --gridpad is the clean
 * single-variable dial that separates them: the SAME compacted kernel over the
 * SAME work list at grid.y = wmax, 2x, 4x and the full 4096, so registers,
 * occupancy, shared memory, arithmetic and memory traffic are all identical and
 * the ONLY variable is how many blocks exit immediately. It is the block-count
 * analogue of the dynamic-shared occupancy dial that killed the occupancy
 * hypothesis in the probe (§4.1), and it should be run BEFORE the in-model arm.
 *
 * BIT-EXACTNESS — by construction, and the argument is unusually strong here:
 *   - the SET of (expert, r0, rc) work items is identical. The uncompacted grid
 *     runs (e, z) for every e with z*RT < ecount[e]; the work list holds exactly
 *     those, same r0 = z*RT, same rc = min(ecount[e] - r0, RT).
 *   - the kernel body is the SAME TOKENS — both entry points expand
 *     FC_MOE_DEDUP_SHX_BODY — over the same ereq, x and weights.
 *   - every hid[f*ff + row] is written by exactly ONE work item with a plain
 *     store; a request f is in exactly one expert's list exactly once. No
 *     atomic, no accumulation, no read-modify-write across blocks, so the order
 *     blocks run in is not observable. This is stronger than "same accumulation
 *     order": there is no shared accumulation to order.
 *   - rc is a loop bound, never an operand.
 * Note the shipped path is ALREADY order-nondeterministic in a way this does not
 * add to: ereq is built with atomicAdd, so which tile a request lands in already
 * varies run to run, and the output is already invariant to it for the reason
 * above. The compacted work list is built by an exclusive scan, so it is MORE
 * deterministic than what ships, not less.
 *
 * NEGATIVE CONTROLS — this harness must be able to FAIL before green means
 * anything (the repo's "harnesses here measure nothing" lesson: six
 * green-but-vacuous gates in one day):
 *   1. `--self` compares the shipped baseline against a second run of the
 *      BASELINE. It must print EXACT, and its ratio IS the noise floor. If
 *      --self ever prints DIFFER the comparison is broken and every other
 *      verdict from this harness is void.
 *   2. `--poison` runs fc_moe_gateup_q3k_dedup_shxc_poison: the compacted
 *      kernel with ONE defect — every x element staged into shared scaled by
 *      (1 + 2^-23), one ulp — so each hid element lands ~1.2e-07 RELATIVE from
 *      correct. That is inside every tolerance in this tree (1e-6, 1e-5), so a
 *      tolerance-based check PASSES it; this harness must report DIFFER. The
 *      same calibration as the matvec staging gate, whose host replica predicted
 *      the GPU differ-count to 0.4 % (84.85 % of outputs move a bit).
 *      A single-element poison was rejected by that gate for good reason: on a
 *      2048-term sum it lands ~2e-10 relative, below fp32 epsilon, and can round
 *      away entirely. A negative control that sometimes passes is worse than none.
 *
 * WHAT IS *NOT* IN HERE. No in-model arm, no serve, no tok/s. This times one
 * kernel on one synthetic router. The microbenchmark's router is a model of the
 * router, not the router: the probe measured its own Zipf draw at 3.98x realised
 * reduction against the real 4.15x, and found the microbenchmark and the real
 * model disagreeing on the dedup-vs-_nx ratio (1.034x vs 1.097x). Treat it as a
 * DIAGNOSTIC, not a predictor; the in-model A/B through tests/dbg_qwen35_batch
 * is what produces a number worth quoting.
 *
 * Run on a K80 host (nvcc 11.4, gcc-11 host, sm_37), ONE die, sibling verified
 * idle in the trace — two busy dies on one K80 board drop to a 398 MHz power cap.
 * Pin the clock LOW so the power floor never binds; at 875 the control's spread
 * on this box is 12.4 % against 0.66 % pinned.
 *
 *   nvcc -arch=sm_37 -ccbin gcc-11 -Wno-deprecated-gpu-targets \
 *        -isystem ../nvcc_shim -I.. -O3 -Xptxas -v \
 *        -o bench_moe_grid_compaction bench_moe_grid_compaction.cu -lm
 *   sudo nvidia-smi -i <die> -ac 2505,666       # NEVER -rac: it CAPS at 562
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_moe_grid_compaction --self     # EXACT
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_moe_grid_compaction --poison   # DIFFER
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_moe_grid_compaction --empty    # the 18.6 %
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_moe_grid_compaction --gridpad  # THE dial
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_moe_grid_compaction            # the arm
 *   sudo nvidia-smi -i <die> -ac 2505,875       # RESTORE
 *
 * Default shape is nex-n2-mini / saber's prefill MoE: n_expert 256, k 8,
 * ff 512, cols 2048, ntok 128. ~305 MB of synthetic weights, so it needs a die
 * to itself but nothing like a model load.
 *
 * @status Written, compiled, NEVER RUN. Every decision rule is pre-registered in
 *   results/prefill_moe_grid_compaction.md §6 and must not be edited after an
 *   arm has been seen.
 * @issues Registers/occupancy printed here come from the OFFLINE ptxas in this
 *   binary. Foundry ships PTX and loads it with no JIT options, so the driver's
 *   ptxas is what allocates registers on the shipping path — the probe checked
 *   the two agree on this kernel, but `cuobjdump -res-usage` on a cubin built
 *   from moe_expert_group.cu is the authority, not this printout.
 * @todo None
 */

#define FC_MOE_DEDUP_COMPACT 1
#include "../moe_expert_group.cu"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* nex-n2-mini / saber's routed-expert MoE at the shipped prefill chunk. */
#define BENCH_N_EXPERT 256
#define BENCH_K        8
#define BENCH_FF       512
#define BENCH_COLS     2048
#define BENCH_NTOK     128
#define BENCH_THREADS  256
#define BENCH_SMX      13              /* GK210 */

/* 1 + 2^-23. Calibrated in tests/calib_nx_stage_poison.c: ~1.2e-07 relative on
 * the output, inside every tolerance here, and it moves 84.85 % of output bit
 * patterns (not 100 % — cancelled long sums round back). Do NOT weaken it. */
#define BENCH_POISON_SCALE 1.00000011920928955078125f

/*
 * POISON ARM. The compacted kernel with exactly one perturbation: the x slice
 * staged into shared is scaled by one ulp. Deliberately a copy of the compacted
 * entry point rather than a flag inside FC_MOE_DEDUP_SHX_BODY, so the shipped
 * macro cannot be reached by a poison path even by accident.
 */
extern "C" __global__ void fc_moe_gateup_q3k_dedup_shxc_poison(
        const unsigned char *gate_all, const unsigned char *up_all,
        const float *x, const int *ereq, const int2 *wlist, float *hid, int k,
        int ff, int cols, int row_stride, int maxr) {
    __shared__ float sh_x[FC_MOE_DEDUP_RT * 256];
    __shared__ int sh_fj[FC_MOE_DEDUP_RT];
    __shared__ int sh_tok[FC_MOE_DEDUP_RT];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int2 item = wlist[blockIdx.y];
    int e = item.x;
    if (e < 0)
        return;
    int r0 = FC_MOE_WL_R0(item.y);
    int rc = FC_MOE_WL_RC(item.y);
    int active = row < ff;

    if ((int)threadIdx.x < FC_MOE_DEDUP_RT) {
        int j = threadIdx.x;
        int f = (j < rc) ? ereq[(size_t)e * maxr + r0 + j] : -1;
        sh_fj[j] = f;
        sh_tok[j] = (f >= 0) ? f / k : 0;
    }
    __syncthreads();

    float pg[FC_MOE_DEDUP_RT], pu[FC_MOE_DEDUP_RT];
#pragma unroll
    for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {
        pg[j] = 0.0f;
        pu[j] = 0.0f;
    }

    size_t expert_stride = (size_t)ff * (size_t)row_stride;
    const unsigned char *gb = gate_all + (size_t)e * expert_stride +
                              (size_t)row * (size_t)row_stride;
    const unsigned char *ub = up_all + (size_t)e * expert_stride +
                              (size_t)row * (size_t)row_stride;

    int nsb = cols >> 8;
    for (int sb = 0; sb < nsb; sb++) {
        __syncthreads();
        for (int j = 0; j < rc; j++) {
            const float *src = x + (size_t)sh_tok[j] * cols + (size_t)sb * 256;
            /* THE ONE DEFECT. */
            sh_x[j * 256 + threadIdx.x] =
                __fmul_rn(__ldg(src + threadIdx.x), BENCH_POISON_SCALE);
        }
        __syncthreads();
        if (active) {
            float wg[8], wu[8];
            float dg = fc_q3k_decode_sb(gb, lane, wg);
            float du = fc_q3k_decode_sb(ub, lane, wu);
#pragma unroll
            for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {
                if (j < rc) {
                    const float *xb = sh_x + j * 256;
                    float ag = 0.0f, au = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        float xv = xb[32 * i + lane];
                        ag = __fmaf_rn(wg[i], xv, ag);
                        au = __fmaf_rn(wu[i], xv, au);
                    }
                    pg[j] += dg * ag;
                    pu[j] += du * au;
                }
            }
            gb += FC_Q3K_RBLOCK_BYTES;
            ub += FC_Q3K_RBLOCK_BYTES;
        }
    }

#pragma unroll
    for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {
        float g = fc_warp_sum_grp(pg[j]);
        float u = fc_warp_sum_grp(pu[j]);
        if (active && lane == 0 && j < rc)
            hid[(size_t)sh_fj[j] * ff + row] = (g / (1.0f + expf(-g))) * u;
    }
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

static unsigned bench_lcg(unsigned *st) {
    *st = *st * 1664525u + 1013904223u;
    return *st;
}

static void fill_deterministic(float *p, size_t n, unsigned seed, float lo,
                               float hi) {
    unsigned st = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        float u = (float)(bench_lcg(&st) >> 8) / (float)(1u << 24);
        p[i] = lo + u * (hi - lo);
    }
}

/*
 * Synthetic REPACKED Q3_K super-blocks, 152 B each: u32 qw[32] @0, u8 sc[16]
 * @128 raw 6-bit, f32 d @144, pad @148. Values are arbitrary — the instruction
 * stream does not depend on them — but d stays small so a 2048-term dot cannot
 * reach inf and let two arms "differ" for the wrong reason.
 */
static void fill_q3k_blocks(unsigned char *buf, size_t nblocks, unsigned seed) {
    unsigned st = seed | 1u;
    for (size_t b = 0; b < nblocks; b++) {
        unsigned char *blk = buf + b * FC_Q3K_RBLOCK_BYTES;
        unsigned int *qw = (unsigned int *)blk;
        for (int i = 0; i < 32; i++)
            qw[i] = bench_lcg(&st);
        for (int i = 0; i < 16; i++)
            blk[128 + i] = (unsigned char)(bench_lcg(&st) >> 26);   /* 0..63 */
        float d = 1.0e-4f + 1.0e-4f * (float)(bench_lcg(&st) >> 8) /
                            (float)(1u << 24);
        memcpy(blk + 144, &d, sizeof(float));
        memset(blk + 148, 0, 4);
    }
}

/*
 * Zipf(s) top-k WITHOUT replacement per token — the probe's router model, whose
 * realised decode reduction (3.98x) sat within 5 % of the real router's 4.15x.
 * It is a model of the router, not the router.
 */
static void fill_router_zipf(int *top_idx, int ntok, int k, int n_expert,
                             double s, unsigned seed) {
    double *cum = (double *)malloc((size_t)n_expert * sizeof(double));
    unsigned st = seed | 1u;
    double acc = 0.0;
    for (int e = 0; e < n_expert; e++) {
        acc += 1.0 / pow((double)(e + 1), s);
        cum[e] = acc;
    }
    for (int t = 0; t < ntok; t++) {
        int *slot = top_idx + (size_t)t * k;
        for (int j = 0; j < k; j++) {
            int e;
            int dup;
            do {
                double u = (double)(bench_lcg(&st) >> 8) /
                           (double)(1u << 24) * acc;
                e = 0;
                while (e < n_expert - 1 && cum[e] < u)
                    e++;
                dup = 0;
                for (int q = 0; q < j; q++)
                    if (slot[q] == e)
                        dup = 1;
            } while (dup);
            slot[j] = e;
        }
    }
    free(cum);
}

/*
 * Exact upper bound on the compacted work-item count == the compacted grid.y.
 * MUST MATCH fc_moe_dedup_wmax in ../../cuda/fc_cuda.c, which this cannot link
 * against (that is driver-API host code; this binary is runtime-API). The
 * derivation is in ../../cuda/fc_cuda.h. Drift is caught, not merely commented:
 * main() checks the measured tile count against this and aborts if it exceeds it.
 */
static int bench_wmax(int n_expert, int k, int ntok, int maxr) {
    if (n_expert <= 0 || k <= 0 || ntok <= 0 || maxr <= 0)
        return 0;
    if (n_expert > FC_MOE_WL_EXPERT_CAP || maxr > 65535)
        return 0;
    long long total = (long long)ntok * k;
    long long uniq = (total < (long long)n_expert) ? total : (long long)n_expert;
    long long b1 = (total + (FC_MOE_DEDUP_RT - 1) * uniq) / FC_MOE_DEDUP_RT;
    long long b2 = uniq * ((maxr + FC_MOE_DEDUP_RT - 1) / FC_MOE_DEDUP_RT);
    long long w = (b1 < b2) ? b1 : b2;
    if (total < w)
        w = total;
    return (int)w;
}

enum bench_arm { ARM_BASE = 0, ARM_COMPACT, ARM_POISON };

int main(int argc, char **argv) {
    int self_mode = 0, poison_mode = 0, empty_mode = 0, gridpad_mode = 0;
    int reps = 20;
    double zipf_s = 1.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--self"))
            self_mode = 1;
        else if (!strcmp(argv[i], "--poison"))
            poison_mode = 1;
        else if (!strcmp(argv[i], "--empty"))
            empty_mode = 1;
        else if (!strcmp(argv[i], "--gridpad"))
            gridpad_mode = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--zipf") && i + 1 < argc)
            zipf_s = atof(argv[++i]);
        else {
            fprintf(stderr,
                    "usage: %s [--self|--poison|--empty|--gridpad] "
                    "[--reps N] [--zipf S]\n", argv[0]);
            return 2;
        }
    }

    const int n_expert = BENCH_N_EXPERT, k = BENCH_K, ff = BENCH_FF;
    const int cols = BENCH_COLS, ntok = BENCH_NTOK, maxr = BENCH_NTOK;
    const int row_stride = (cols / 256) * FC_Q3K_RBLOCK_BYTES;
    const int wmax = bench_wmax(n_expert, k, ntok, maxr);
    const unsigned zt = (unsigned)((maxr + FC_MOE_DEDUP_RT - 1) /
                                   FC_MOE_DEDUP_RT);
    const unsigned gx = (unsigned)((ff + 7) / 8);
    if (wmax <= 0) {
        fprintf(stderr, "shape is not compactable (wmax=0)\n");
        return 2;
    }
    /* The work list is allocated at the UNCOMPACTED grid.y*grid.z so --gridpad
     * can pad grid.y all the way back to it. */
    const int wcap = n_expert * (int)zt;

    const size_t wbytes = (size_t)n_expert * ff * row_stride;
    const size_t n_x = (size_t)ntok * cols;
    const size_t n_hid = (size_t)ntok * k * ff;

    unsigned char *h_w = (unsigned char *)malloc(wbytes);
    float *h_x = (float *)malloc(n_x * sizeof(float));
    int *h_tidx = (int *)malloc((size_t)ntok * k * sizeof(int));
    float *h_oa = (float *)malloc(n_hid * sizeof(float));
    float *h_ob = (float *)malloc(n_hid * sizeof(float));
    int *h_ecount = (int *)malloc((size_t)n_expert * sizeof(int));
    if (!h_w || !h_x || !h_tidx || !h_oa || !h_ob || !h_ecount) {
        fprintf(stderr, "host alloc failed (%zu B of weights)\n", wbytes);
        return 1;
    }
    fill_q3k_blocks(h_w, wbytes / FC_Q3K_RBLOCK_BYTES, 0x51ce51ceu);
    fill_deterministic(h_x, n_x, 0xa5a5u, -1.0f, 1.0f);
    fill_router_zipf(h_tidx, ntok, k, n_expert, zipf_s, 0x1234u);

    unsigned char *d_gate, *d_up;
    float *d_x, *d_hid;
    int *d_tidx, *d_ecount, *d_ereq;
    int2 *d_wlist;
    CHECK(cudaMalloc(&d_gate, wbytes));
    CHECK(cudaMalloc(&d_up, wbytes));
    CHECK(cudaMalloc(&d_x, n_x * sizeof(float)));
    CHECK(cudaMalloc(&d_hid, n_hid * sizeof(float)));
    CHECK(cudaMalloc(&d_tidx, (size_t)ntok * k * sizeof(int)));
    CHECK(cudaMalloc(&d_ecount, (size_t)n_expert * sizeof(int)));
    CHECK(cudaMalloc(&d_ereq, (size_t)n_expert * maxr * sizeof(int)));
    CHECK(cudaMalloc(&d_wlist, (size_t)wcap * sizeof(int2)));
    CHECK(cudaMemcpy(d_gate, h_w, wbytes, cudaMemcpyHostToDevice));
    /* up gets a different seed so gate and up cannot cancel each other. */
    fill_q3k_blocks(h_w, wbytes / FC_Q3K_RBLOCK_BYTES, 0x7ea17ea1u);
    CHECK(cudaMemcpy(d_up, h_w, wbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, h_x, n_x * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_tidx, h_tidx, (size_t)ntok * k * sizeof(int),
                     cudaMemcpyHostToDevice));

    /* Request lists — the SHIPPED builder, unchanged, feeding both arms. */
    fc_moe_dedup_build<<<1, BENCH_THREADS>>>(d_tidx, d_ecount, d_ereq,
                                             n_expert, k, ntok, maxr);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    if (empty_mode)
        CHECK(cudaMemset(d_ecount, 0, (size_t)n_expert * sizeof(int)));
    CHECK(cudaMemcpy(h_ecount, d_ecount, (size_t)n_expert * sizeof(int),
                     cudaMemcpyDeviceToHost));

    int unique = 0, total = 0, tiles = 0, maxR = 0;
    for (int e = 0; e < n_expert; e++) {
        int r = h_ecount[e];
        if (r <= 0)
            continue;
        unique++;
        total += r;
        tiles += (r + FC_MOE_DEDUP_RT - 1) / FC_MOE_DEDUP_RT;
        if (r > maxR)
            maxR = r;
    }
    /* The bound is the whole basis of the compacted launch: if a real router
     * ever produced more tiles than wmax, the tail of the work list would be
     * silently dropped. Prove it on every run rather than trusting the algebra. */
    if (tiles > wmax) {
        fprintf(stderr, "BOUND VIOLATED: tiles=%d > wmax=%d — the compacted "
                        "launch would drop work; fix fc_moe_dedup_wmax\n",
                tiles, wmax);
        return 1;
    }

    cudaFuncAttributes at_base, at_comp;
    CHECK(cudaFuncGetAttributes(&at_base, fc_moe_gateup_q3k_dedup_shx));
    CHECK(cudaFuncGetAttributes(&at_comp, fc_moe_gateup_q3k_dedup_shxc));
    int occ_base = 0, occ_comp = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_base, fc_moe_gateup_q3k_dedup_shx, BENCH_THREADS, 0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_comp, fc_moe_gateup_q3k_dedup_shxc, BENCH_THREADS, 0));

    printf("shape        n_expert=%d k=%d ff=%d cols=%d ntok=%d maxr=%d "
           "row_stride=%d\n", n_expert, k, ff, cols, ntok, maxr, row_stride);
    printf("router       apps=%d unique=%d/%d maxR=%d meanR=%.2f tiles=%d "
           "realised=%.2fx (zipf s=%.2f)\n", total, unique, n_expert, maxR,
           unique ? (double)total / unique : 0.0, tiles,
           tiles ? (double)total / tiles : 0.0, zipf_s);
    printf("grid base    (%u,%u,%u) = %u blocks, %d useful (%.1f %%)\n",
           gx, (unsigned)n_expert, zt, gx * (unsigned)n_expert * zt,
           (int)gx * tiles,
           100.0 * (double)((int)gx * tiles) /
               (double)(gx * (unsigned)n_expert * zt));
    printf("grid compact (%u,%u,1) = %u blocks, %d useful (%.1f %%)  "
           "wmax=%d  block cut %.2fx\n",
           gx, (unsigned)wmax, gx * (unsigned)wmax, (int)gx * tiles,
           100.0 * (double)((int)gx * tiles) / (double)(gx * (unsigned)wmax),
           wmax, (double)(n_expert * (int)zt) / (double)wmax);
    printf("waves        base %.1f, compact %.1f  (%d SMX x %d resident "
           "blocks)\n",
           (double)(gx * (unsigned)n_expert * zt) /
               (double)(BENCH_SMX * (occ_base ? occ_base : 1)),
           (double)(gx * (unsigned)wmax) /
               (double)(BENCH_SMX * (occ_comp ? occ_comp : 1)),
           BENCH_SMX, occ_comp);
    printf("resources    base REG~%d shared %zu B frame %zu B blocks/SMX %d | "
           "compact REG~%d shared %zu B frame %zu B blocks/SMX %d  "
           "(OFFLINE ptxas — the shipping path JITs from PTX)\n",
           at_base.numRegs, at_base.sharedSizeBytes,
           (size_t)at_base.localSizeBytes, occ_base,
           at_comp.numRegs, at_comp.sharedSizeBytes,
           (size_t)at_comp.localSizeBytes, occ_comp);

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    /*
     * ★ --gridpad: the single-variable block-count dial. Same kernel, same work
     * list, same work; grid.y padded with sentinel entries. A linear fit of
     * ms against grid.y is the marginal cost of ONE empty y-slice (gx blocks),
     * and its extrapolation to grid.y = wcap is what compaction can return.
     */
    if (gridpad_mode) {
        const int pads[] = { wmax, 2 * wmax, 4 * wmax, wcap };
        printf("\n--gridpad    grid.y      blocks     ms/launch   vs wmax\n");
        double base_ms = 0.0;
        for (size_t p = 0; p < sizeof(pads) / sizeof(pads[0]); p++) {
            int W = pads[p] > wcap ? wcap : pads[p];
            fc_moe_dedup_build_wl<<<1, BENCH_THREADS>>>(d_ecount, d_wlist,
                                                        n_expert, W);
            CHECK(cudaDeviceSynchronize());
            for (int w = 0; w < 3; w++)
                fc_moe_gateup_q3k_dedup_shxc<<<dim3(gx, (unsigned)W),
                                               BENCH_THREADS>>>(
                    d_gate, d_up, d_x, d_ereq, d_wlist, d_hid, k, ff, cols,
                    row_stride, maxr);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < reps; it++)
                fc_moe_gateup_q3k_dedup_shxc<<<dim3(gx, (unsigned)W),
                                               BENCH_THREADS>>>(
                    d_gate, d_up, d_x, d_ereq, d_wlist, d_hid, k, ff, cols,
                    row_stride, maxr);
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            float total_ms = 0.0f;
            CHECK(cudaEventElapsedTime(&total_ms, ev0, ev1));
            double ms = (double)total_ms / (double)reps;
            if (p == 0)
                base_ms = ms;
            printf("             %6d  %10u   %9.4f   %6.4fx\n", W,
                   gx * (unsigned)W, ms, base_ms > 0.0 ? ms / base_ms : 0.0);
        }
        printf("             a FLAT column means the empty-block cost is "
               "HIDDEN behind compute and lever B is dead; a LINEAR one means "
               "it is additive and compaction returns its slope x %d\n",
               wcap - wmax);
        return 0;
    }

    /* Work list for the compacted arms, at the real launch width. */
    fc_moe_dedup_build_wl<<<1, BENCH_THREADS>>>(d_ecount, d_wlist, n_expert,
                                                wmax);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    const enum bench_arm cmp_arm = self_mode   ? ARM_BASE
                                 : poison_mode ? ARM_POISON : ARM_COMPACT;
    float ms[2] = { 0.0f, 0.0f };

    for (int a = 0; a < 2; a++) {
        const enum bench_arm arm = (a == 0) ? ARM_BASE : cmp_arm;

#define LAUNCH_ARM()                                                          \
        do {                                                                  \
            if (arm == ARM_BASE)                                              \
                fc_moe_gateup_q3k_dedup_shx<<<dim3(gx, (unsigned)n_expert,    \
                                                   zt), BENCH_THREADS>>>(     \
                    d_gate, d_up, d_x, d_ecount, d_ereq, d_hid, k, ff, cols,  \
                    row_stride, maxr);                                        \
            else if (arm == ARM_COMPACT)                                      \
                fc_moe_gateup_q3k_dedup_shxc<<<dim3(gx, (unsigned)wmax),      \
                                               BENCH_THREADS>>>(              \
                    d_gate, d_up, d_x, d_ereq, d_wlist, d_hid, k, ff, cols,   \
                    row_stride, maxr);                                        \
            else                                                              \
                fc_moe_gateup_q3k_dedup_shxc_poison<<<                        \
                    dim3(gx, (unsigned)wmax), BENCH_THREADS>>>(               \
                    d_gate, d_up, d_x, d_ereq, d_wlist, d_hid, k, ff, cols,   \
                    row_stride, maxr);                                        \
        } while (0)

        /* Pass 1 — CORRECTNESS. hid poisoned to 0xCD first, so an element no
         * arm writes cannot masquerade as a match. Exactly one launch. */
        CHECK(cudaMemset(d_hid, 0xCD, n_hid * sizeof(float)));
        LAUNCH_ARM();
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy((a == 0) ? h_oa : h_ob, d_hid,
                         n_hid * sizeof(float), cudaMemcpyDeviceToHost));

        /* Pass 2 — TIMING. Back-to-back launches only; the kernel is idempotent
         * so no reset belongs inside the timed span. */
        for (int w = 0; w < 3; w++)
            LAUNCH_ARM();
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(ev0));
        for (int it = 0; it < reps; it++)
            LAUNCH_ARM();
        CHECK(cudaEventRecord(ev1));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        float total_ms = 0.0f;
        CHECK(cudaEventElapsedTime(&total_ms, ev0, ev1));
        ms[a] = total_ms / (float)reps;
#undef LAUNCH_ARM
    }

    const char *arm_name = self_mode   ? "shx(self)"
                         : poison_mode ? "shxc_POISON" : "shxc_compact";

    size_t diff = 0;
    double dmax = 0.0;
    for (size_t i = 0; i < n_hid; i++) {
        if (memcmp(&h_oa[i], &h_ob[i], sizeof(float)) != 0) {
            diff++;
            double d = fabs((double)h_oa[i] - (double)h_ob[i]);
            if (d > dmax)
                dmax = d;
        }
    }

    printf("baseline     %8.4f ms/launch   (fc_moe_gateup_q3k_dedup_shx, the "
           "shipped kernel)\n", ms[0]);
    printf("%-12s %8.4f ms/launch   ratio %.4fx\n", arm_name, ms[1],
           ms[1] > 0.0f ? ms[0] / ms[1] : 0.0f);
    printf("hid bits     %s  (%zu/%zu differ, max |d| %.3e)\n",
           diff ? "DIFFER" : "EXACT", diff, n_hid, dmax);
    if (empty_mode)
        printf("NOTE         --empty: every ecount is 0, so NO block does any "
               "work in either arm. The baseline column is the empty-block "
               "cost of the SHIPPED kernel, which has never been measured on "
               "_shx (the 18.6 %% figure is the plain dedup's).\n");
    if (self_mode && diff)
        printf("VERDICT      BROKEN — --self must be EXACT; every other "
               "verdict from this harness is void\n");
    else if (poison_mode && !diff)
        printf("VERDICT      BROKEN — --poison must be DIFFER; the bit "
               "comparison is not comparing bits\n");
    else if (self_mode)
        printf("VERDICT      control OK; the ratio above is the noise floor\n");
    else if (poison_mode)
        printf("VERDICT      negative control fires\n");
    else
        printf("VERDICT      %s vs shipped fc_moe_gateup_q3k_dedup_shx: %s\n",
               arm_name,
               diff ? "NOT bit-exact — investigate before believing the ratio"
                    : "bit-exact");

    CHECK(cudaFree(d_gate));
    CHECK(cudaFree(d_up));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_hid));
    CHECK(cudaFree(d_tidx));
    CHECK(cudaFree(d_ecount));
    CHECK(cudaFree(d_ereq));
    CHECK(cudaFree(d_wlist));
    free(h_w);
    free(h_x);
    free(h_tidx);
    free(h_oa);
    free(h_ob);
    free(h_ecount);
    return 0;
}
