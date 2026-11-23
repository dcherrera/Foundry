/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file bench_moe_shx_staging.cu
 * @brief Bit-exactness + timing gate for the x-staging forms of the MoE dedup
 *   gate+up (moe_dedup_shx_staging.cuh). Baseline is the SHIPPED kernel
 *   fc_moe_gateup_q3k_dedup_shx compiled from ../moe_expert_group.cu at the
 *   default flag, so it is a control symbol from the same source in the same
 *   translation unit as every arm.
 *
 * THE DEFECT UNDER TEST — COUNTED, not estimated.
 * The shipped staging loop is bounded by the RUNTIME request count `rc`:
 *
 *     for (int j = 0; j < rc; j++)
 *         sh_x[j*256 + tid] = __ldg(x + sh_tok[j]*cols + sb*256 + tid);
 *
 * COUNTED in SASS: the whole staging
 * region carries **1 global load in flight** — every LDG is followed by its own
 * TEXDEPBAR 0x0 — because the address of load j comes from an ld.shared of
 * sh_tok[j] that sits after the st.shared of sh_x[j-1], and ptxas will not hoist
 * a shared load above a shared store it cannot disambiguate. The four LDG of the
 * unrolled body are otherwise mutually independent, so 1 is an artifact of that
 * one alias edge, not a real dependency. The matvec staging gate measured
 * **1.426x** on the kernel for taking exactly this count from 1 -> 8
 * and established that on this silicon
 * loads-in-flight, NOT executed instructions and NOT occupancy, is the predictor
 * that tracks wall clock. Three of the five candidates here were built and
 * counted BEFORE any of them was run; two of them cut instructions hard and do
 * not move loads-in-flight at all, which is why they are in the harness.
 *
 * ★ THE RESOURCE CONSTRAINT THAT MAKES THIS KERNEL DIFFERENT FROM THE MATVECS.
 * _shx runs at 62 registers, and Kepler's 8-register allocation granularity puts
 * the 8-blocks/SMX cliff at exactly 64: 131072 / (64 * 256) = 8, and 72 gives 7.
 * So a form has TWO registers of headroom. HOIST8 spends nine and drops to 7
 * blocks/SMX (a 1.14x residency loss) to buy its 8 loads in flight; SPLIT8 buys
 * the same 8 for one register. Every arm prints its own REG and blocks/SMX and
 * they MUST match the standalone-build numbers in the results doc — a mismatch
 * means this binary compiled a different form than the shipped source would, and
 * the arm is mislabelled. That is not hypothetical: the matvec gate had exactly
 * this failure (_Pragma dropped inside a macro argument) and it took two
 * instruments disagreeing to catch it.
 *
 * BIT-EXACTNESS — by construction, argued in full in moe_dedup_shx_staging.cuh.
 * In one line: all forms read the same global addresses and write the same shared
 * addresses, each exactly once, between the same two __syncthreads(), with ZERO
 * floating-point operations anywhere in the staging code — so there is nothing to
 * reassociate and no contraction to break. The VEC4 forms additionally
 * re-partition (request, element) across threads, which the trailing barrier
 * makes unobservable.
 *
 * NEGATIVE CONTROLS — this harness must be able to FAIL before green means
 * anything (the repo's "harnesses here measure nothing" lesson):
 *   1. `--self` compares the shipped baseline against a second run of the
 *      BASELINE. It must print EXACT, and its ratio spread across separate
 *      processes IS the noise floor. A DIFFER voids every other verdict.
 *   2. `--poison` runs the SHIPPED FLAT staging plus exactly one perturbation:
 *      every staged x element scaled by (1 + 2^-23). That is ~1.2e-07 RELATIVE on
 *      the output — inside every tolerance in this tree (1e-6, 1e-5), so a
 *      tolerance-based check PASSES it and this harness must report DIFFER. Same
 *      calibration as the matvec gate, whose host replica predicted the GPU
 *      differ-count to 0.4 % (84.85 % predicted, 84.51 % measured). The poison
 *      arm expands the SAME form macro as the baseline with a different XF, so it
 *      cannot drift from what ships. A single-element poison was rejected there
 *      for good reason: on a 2048-term sum it lands ~2e-10 relative and can round
 *      away, and a negative control that sometimes passes is worse than none.
 *
 * WHAT IS *NOT* IN HERE. No in-model arm, no serve, no tok/s. This times one
 * kernel on one synthetic router, and the microbenchmark's router is a model of
 * the router, not the router (the dedup probe measured 3.98x realised reduction
 * against the real 4.15x, and microbenchmark and model disagreeing 1.034x vs
 * 1.097x on the dedup ratio). Treat it as a DIAGNOSTIC; the in-model A/B through
 * tests/dbg_qwen35_batch is what produces a number worth quoting.
 *
 * Run on a K80 host (nvcc 11.4, gcc-11 host, sm_37), ONE die, sibling verified idle
 * in the trace — two busy dies on one K80 board drop to a 398 MHz power cap.
 *
 *   nvcc -arch=sm_37 -ccbin gcc-11 -Wno-deprecated-gpu-targets \
 *        -isystem ../nvcc_shim -I.. -O3 -Xptxas -v \
 *        -o bench_moe_shx_staging bench_moe_shx_staging.cu -lm
 *   sudo nvidia-smi -i <die> -ac 2505,666       # NEVER -rac: it CAPS at 562
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_moe_shx_staging --self      # EXACT
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_moe_shx_staging --poison    # DIFFER
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_moe_shx_staging --sweep     # all 5 forms
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_moe_shx_staging --form 4    # the arm
 *   sudo nvidia-smi -i <die> -ac 2505,875       # RESTORE
 *
 * Default shape is nex-n2-mini / saber's prefill MoE: n_expert 256, k 8, ff 512,
 * cols 2048, ntok 128. ~305 MB of synthetic weights.
 *
 * @status Written, compiled, NEVER RUN. Decision rules are pre-registered in
 *   results/prefill_moe_shx_staging.md §6 and must not be edited after an arm has
 *   been seen.
 * @issues Registers/occupancy printed here come from the OFFLINE ptxas in this
 *   binary. Foundry ships PTX and the driver's ptxas allocates on the shipping
 *   path; `cuobjdump -res-usage` on a cubin built from moe_expert_group.cu is the
 *   authority, and cuFuncGetAttribute on the loaded module is the only authority
 *   for what actually ships.
 * @todo None
 */

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
 * the output, inside every tolerance here, and it moves ~84.9 % of output bit
 * patterns (not 100 % — cancelled long sums round back). Do NOT weaken it. */
#define BENCH_POISON_SCALE 1.00000011920928955078125f
#define BENCH_POISON(v) __fmul_rn((v), BENCH_POISON_SCALE)

/* Per-arm staging macros: the form with its XF baked in, so each can be passed
 * to FC_MOE_DEDUP_SHX_BODY_S as a two-argument STAGE. */
#define BENCH_STAGE_FLAT_KEEP(RC, SB)                                          \
    FC_MOE_SHX_STAGE_FLAT_FORM(RC, SB, FC_MOE_SHX_KEEP)
#define BENCH_STAGE_FLAT_POISON(RC, SB)                                        \
    FC_MOE_SHX_STAGE_FLAT_FORM(RC, SB, BENCH_POISON)
#define BENCH_STAGE_UNROLL8(RC, SB)                                            \
    FC_MOE_SHX_STAGE_UNROLL8_FORM(RC, SB, FC_MOE_SHX_KEEP)
#define BENCH_STAGE_VEC4(RC, SB)                                               \
    FC_MOE_SHX_STAGE_VEC4_FORM(RC, SB, FC_MOE_SHX_KEEP)
#define BENCH_STAGE_HOIST8(RC, SB)                                             \
    FC_MOE_SHX_STAGE_HOIST8_FORM(RC, SB, FC_MOE_SHX_KEEP)
#define BENCH_STAGE_SPLIT8(RC, SB)                                             \
    FC_MOE_SHX_STAGE_SPLIT8_FORM(RC, SB, FC_MOE_SHX_KEEP)
#define BENCH_STAGE_SPLITV4(RC, SB)                                            \
    FC_MOE_SHX_STAGE_SPLITV4_FORM(RC, SB, FC_MOE_SHX_KEEP)

/*
 * One arm = fc_moe_gateup_q3k_dedup_shx's prologue verbatim plus the shared body
 * macro with a different (PRE, STAGE) pair. Generated rather than copied so no
 * arm can drift from the shipped prologue, and so the only textual difference
 * between the baseline and an arm is the staging form under test.
 */
#define BENCH_DEF_SHX_ARM(NAME, PRE, STAGE, ALIGN)                             \
extern "C" __global__ void NAME(const unsigned char *gate_all,                 \
        const unsigned char *up_all, const float *x, const int *ecount,        \
        const int *ereq, float *hid, int k, int ff, int cols, int row_stride,   \
        int maxr) {                                                            \
    __shared__ ALIGN float sh_x[FC_MOE_DEDUP_RT * 256];                        \
    __shared__ int sh_fj[FC_MOE_DEDUP_RT];                                     \
    __shared__ int sh_tok[FC_MOE_DEDUP_RT];                                    \
    int warp = threadIdx.x >> 5;                                               \
    int lane = threadIdx.x & 31;                                               \
    int row = blockIdx.x * 8 + warp;                                           \
    int e = blockIdx.y;                                                        \
    int R = ecount[e];                                                         \
    int r0 = blockIdx.z * FC_MOE_DEDUP_RT;                                     \
    if (r0 >= R)                                                               \
        return;                                                                \
    int rc = R - r0;                                                           \
    if (rc > FC_MOE_DEDUP_RT)                                                  \
        rc = FC_MOE_DEDUP_RT;                                                  \
    int active = row < ff;                                                     \
    FC_MOE_DEDUP_SHX_BODY_S(PRE, STAGE, e, r0, rc);                            \
}

BENCH_DEF_SHX_ARM(bench_shx_poison, FC_MOE_SHX_STAGE_NO_PRE,
                  BENCH_STAGE_FLAT_POISON, )
BENCH_DEF_SHX_ARM(bench_shx_unroll8, FC_MOE_SHX_STAGE_NO_PRE,
                  BENCH_STAGE_UNROLL8, )
BENCH_DEF_SHX_ARM(bench_shx_vec4, FC_MOE_SHX_STAGE_NO_PRE, BENCH_STAGE_VEC4,
                  __align__(16))
BENCH_DEF_SHX_ARM(bench_shx_hoist8, FC_MOE_SHX_STAGE_HOIST8_PRE,
                  BENCH_STAGE_HOIST8, )
BENCH_DEF_SHX_ARM(bench_shx_split8, FC_MOE_SHX_STAGE_NO_PRE,
                  BENCH_STAGE_SPLIT8, )
BENCH_DEF_SHX_ARM(bench_shx_splitv4, FC_MOE_SHX_STAGE_NO_PRE,
                  BENCH_STAGE_SPLITV4, __align__(16))

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

typedef void (*shx_kernel_fn)(const unsigned char *, const unsigned char *,
                              const float *, const int *, const int *, float *,
                              int, int, int, int, int);

struct shx_arm {
    const char *name;
    shx_kernel_fn fn;
    int reg_expected;                  /* from the standalone cubin, §4 */
    int blocks_expected;
};

/* REG / blocks-per-SMX COUNTED from cuobjdump -res-usage on cubins built from
 * ../moe_expert_group.cu at each FC_MOE_SHX_STAGE value. If this binary reports
 * different numbers it did not compile the form it claims to. */
static const struct shx_arm bench_arms[] = {
    { "flat(self)",  fc_moe_gateup_q3k_dedup_shx, 62, 8 },
    { "FLAT_POISON", bench_shx_poison,            62, 8 },
    { "UNROLL8",     bench_shx_unroll8,           62, 8 },
    { "VEC4",        bench_shx_vec4,              63, 8 },
    { "HOIST8",      bench_shx_hoist8,            71, 7 },
    { "SPLIT8",      bench_shx_split8,            63, 8 },
    { "SPLITV4",     bench_shx_splitv4,           64, 8 },
};
#define BENCH_N_ARMS ((int)(sizeof(bench_arms) / sizeof(bench_arms[0])))
/* Index of each arm in bench_arms, matching --form N (0 = self, 1 = poison). */
#define BENCH_ARM_SELF   0
#define BENCH_ARM_POISON 1

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
 * @128 raw 6-bit, f32 d @144, pad @148. d stays small so a 2048-term dot cannot
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

/* Zipf(s) top-k without replacement per token — the dedup probe's router model,
 * whose realised reduction sat within 5 % of the real router's. */
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

int main(int argc, char **argv) {
    int form = BENCH_ARM_SELF, sweep = 0, reps = 20;
    double zipf_s = 1.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--self"))
            form = BENCH_ARM_SELF;
        else if (!strcmp(argv[i], "--poison"))
            form = BENCH_ARM_POISON;
        else if (!strcmp(argv[i], "--sweep"))
            sweep = 1;
        else if (!strcmp(argv[i], "--form") && i + 1 < argc)
            form = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--zipf") && i + 1 < argc)
            zipf_s = atof(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--self|--poison|--form N|--sweep] "
                            "[--reps N] [--zipf S]\n", argv[0]);
            return 2;
        }
    }
    if (form < 0 || form >= BENCH_N_ARMS) {
        fprintf(stderr, "--form must be 0..%d\n", BENCH_N_ARMS - 1);
        return 2;
    }

    const int n_expert = BENCH_N_EXPERT, k = BENCH_K, ff = BENCH_FF;
    const int cols = BENCH_COLS, ntok = BENCH_NTOK, maxr = BENCH_NTOK;
    const int row_stride = (cols / 256) * FC_Q3K_RBLOCK_BYTES;
    const unsigned zt = (unsigned)((maxr + FC_MOE_DEDUP_RT - 1) /
                                   FC_MOE_DEDUP_RT);
    const unsigned gx = (unsigned)((ff + 7) / 8);

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
    CHECK(cudaMalloc(&d_gate, wbytes));
    CHECK(cudaMalloc(&d_up, wbytes));
    CHECK(cudaMalloc(&d_x, n_x * sizeof(float)));
    CHECK(cudaMalloc(&d_hid, n_hid * sizeof(float)));
    CHECK(cudaMalloc(&d_tidx, (size_t)ntok * k * sizeof(int)));
    CHECK(cudaMalloc(&d_ecount, (size_t)n_expert * sizeof(int)));
    CHECK(cudaMalloc(&d_ereq, (size_t)n_expert * maxr * sizeof(int)));
    CHECK(cudaMemcpy(d_gate, h_w, wbytes, cudaMemcpyHostToDevice));
    /* up gets a different seed so gate and up cannot cancel each other. */
    fill_q3k_blocks(h_w, wbytes / FC_Q3K_RBLOCK_BYTES, 0x7ea17ea1u);
    CHECK(cudaMemcpy(d_up, h_w, wbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, h_x, n_x * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_tidx, h_tidx, (size_t)ntok * k * sizeof(int),
                     cudaMemcpyHostToDevice));

    /* Request lists — the SHIPPED builder, unchanged, feeding every arm. */
    fc_moe_dedup_build<<<1, BENCH_THREADS>>>(d_tidx, d_ecount, d_ereq,
                                             n_expert, k, ntok, maxr);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_ecount, d_ecount, (size_t)n_expert * sizeof(int),
                     cudaMemcpyDeviceToHost));

    /* The rc HISTOGRAM is the shape variable this lever is sensitive to: FLAT's
     * dead unroll ladder is entered on rc/4 and rc%4, and every candidate stages
     * all RT slots predicated. Print it so a ratio can be attributed. */
    int unique = 0, total = 0, tiles = 0, rc_hist[FC_MOE_DEDUP_RT + 1];
    memset(rc_hist, 0, sizeof(rc_hist));
    for (int e = 0; e < n_expert; e++) {
        int r = h_ecount[e];
        if (r <= 0)
            continue;
        unique++;
        total += r;
        for (int r0 = 0; r0 < r; r0 += FC_MOE_DEDUP_RT) {
            int rc = r - r0;
            if (rc > FC_MOE_DEDUP_RT)
                rc = FC_MOE_DEDUP_RT;
            rc_hist[rc]++;
            tiles++;
        }
    }

    printf("shape        n_expert=%d k=%d ff=%d cols=%d ntok=%d maxr=%d "
           "row_stride=%d nsb=%d\n", n_expert, k, ff, cols, ntok, maxr,
           row_stride, cols >> 8);
    printf("router       apps=%d unique=%d/%d meanR=%.2f tiles=%d "
           "(zipf s=%.2f)\n", total, unique, n_expert,
           unique ? (double)total / unique : 0.0, tiles, zipf_s);
    printf("rc histogram ");
    for (int r = 1; r <= FC_MOE_DEDUP_RT; r++)
        printf("%d:%d ", r, rc_hist[r]);
    printf("\n");
    printf("grid         (%u,%u,%u) = %u blocks\n", gx, (unsigned)n_expert, zt,
           gx * (unsigned)n_expert * zt);

    printf("resources    (OFFLINE ptxas in this binary — the shipping path JITs "
           "from PTX; cuobjdump on a moe_expert_group.cu cubin is the authority)"
           "\n");
    int mismatch = 0;
    for (int a = 0; a < BENCH_N_ARMS; a++) {
        cudaFuncAttributes at;
        int occ = 0;
        CHECK(cudaFuncGetAttributes(&at, (const void *)bench_arms[a].fn));
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, (const void *)bench_arms[a].fn, BENCH_THREADS, 0));
        int bad = (at.numRegs != bench_arms[a].reg_expected ||
                   occ != bench_arms[a].blocks_expected);
        mismatch |= bad;
        printf("  %-12s REG %3d (expect %3d)  frame %4zu B  shared %5zu B  "
               "blocks/SMX %d (expect %d)%s\n",
               bench_arms[a].name, at.numRegs, bench_arms[a].reg_expected,
               (size_t)at.localSizeBytes, at.sharedSizeBytes, occ,
               bench_arms[a].blocks_expected, bad ? "   <== MISMATCH" : "");
    }
    if (mismatch)
        printf("WARNING      an arm's REG/occupancy does not match the "
               "standalone build. This binary may have compiled a different "
               "form than the shipped source would (a _Pragma dropped inside a "
               "macro argument does exactly this). Do NOT trust a ratio until "
               "the arm is re-identified from SASS.\n");

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    const int first = sweep ? BENCH_ARM_POISON + 1 : form;
    const int last = sweep ? BENCH_N_ARMS - 1 : form;
    printf("\n%-12s %10s %10s %10s %8s   %s\n", "arm", "A1 ms", "B ms", "A2 ms",
           "ratio", "hid bits");

    for (int a = first; a <= last; a++) {
        /* A...B...A so the per-process warming bias (measured at -0.49 % on the
         * matvec gate's control) cancels instead of being corrected afterwards. */
        double t[3] = { 0.0, 0.0, 0.0 };
        size_t diff = 0;
        double dmax = 0.0;
        for (int pass = 0; pass < 3; pass++) {
            shx_kernel_fn fn = (pass == 1) ? bench_arms[a].fn
                                           : fc_moe_gateup_q3k_dedup_shx;
            /* Pass 1 — CORRECTNESS. hid poisoned to 0xCD first so an element no
             * arm writes cannot masquerade as a match. Exactly one launch. */
            CHECK(cudaMemset(d_hid, 0xCD, n_hid * sizeof(float)));
            fn<<<dim3(gx, (unsigned)n_expert, zt), BENCH_THREADS>>>(
                d_gate, d_up, d_x, d_ecount, d_ereq, d_hid, k, ff, cols,
                row_stride, maxr);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpy((pass == 1) ? h_ob : h_oa, d_hid,
                             n_hid * sizeof(float), cudaMemcpyDeviceToHost));

            for (int w = 0; w < 3; w++)
                fn<<<dim3(gx, (unsigned)n_expert, zt), BENCH_THREADS>>>(
                    d_gate, d_up, d_x, d_ecount, d_ereq, d_hid, k, ff, cols,
                    row_stride, maxr);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < reps; it++)
                fn<<<dim3(gx, (unsigned)n_expert, zt), BENCH_THREADS>>>(
                    d_gate, d_up, d_x, d_ecount, d_ereq, d_hid, k, ff, cols,
                    row_stride, maxr);
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            t[pass] = (double)ms / (double)reps;
        }
        for (size_t i = 0; i < n_hid; i++) {
            if (memcmp(&h_oa[i], &h_ob[i], sizeof(float)) != 0) {
                diff++;
                double d = fabs((double)h_oa[i] - (double)h_ob[i]);
                if (d > dmax)
                    dmax = d;
            }
        }
        double abase = 0.5 * (t[0] + t[2]);
        printf("%-12s %10.4f %10.4f %10.4f %8.4fx  %s (%zu/%zu differ, "
               "max |d| %.3e)   A-drift %.3f %%\n",
               bench_arms[a].name, t[0], t[1], t[2],
               t[1] > 0.0 ? abase / t[1] : 0.0,
               diff ? "DIFFER" : "EXACT", diff, n_hid, dmax,
               t[0] > 0.0 ? 100.0 * (t[2] - t[0]) / t[0] : 0.0);

        if (a == BENCH_ARM_SELF && diff)
            printf("VERDICT      BROKEN — --self must be EXACT; every other "
                   "verdict from this harness is void\n");
        else if (a == BENCH_ARM_POISON && !diff)
            printf("VERDICT      BROKEN — --poison must be DIFFER; the bit "
                   "comparison is not comparing bits\n");
        else if (a == BENCH_ARM_SELF)
            printf("VERDICT      control OK; the ratio above is this process's "
                   "noise floor (run 10 processes for the real one)\n");
        else if (a == BENCH_ARM_POISON)
            printf("VERDICT      negative control fires (~84.9 %% of %zu "
                   "expected)\n", n_hid);
        else
            printf("VERDICT      %s vs shipped fc_moe_gateup_q3k_dedup_shx: "
                   "%s\n", bench_arms[a].name,
                   diff ? "NOT bit-exact — STOP, the construction argument is "
                          "wrong and there is a real bug"
                        : "bit-exact");
    }

    CHECK(cudaFree(d_gate));
    CHECK(cudaFree(d_up));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_hid));
    CHECK(cudaFree(d_tidx));
    CHECK(cudaFree(d_ecount));
    CHECK(cudaFree(d_ereq));
    free(h_w);
    free(h_x);
    free(h_tidx);
    free(h_oa);
    free(h_ob);
    free(h_ecount);
    return 0;
}
