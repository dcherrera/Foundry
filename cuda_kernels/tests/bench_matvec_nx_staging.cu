/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file bench_matvec_nx_staging.cu
 * @brief Bit-exactness + timing harness for the x-STAGING loop of the batched
 *   K-quant matvecs: fc_matvec_q3k_nx8 (shipped baseline, the kernel every
 *   n>=5 dense projection lands on) vs two staging rewrites that remove the
 *   runtime-divisor integer division from it. Includes matvec_q3k.cu VERBATIM
 *   so the baseline is a control symbol compiled from the same source in the
 *   same module.
 *
 * THE DEFECT UNDER TEST — counted from the committed PTX, not estimated.
 * fc_matvec_q3k_nx_body stages NC*cend activation floats per chunk with
 *
 *     for (int i = threadIdx.x; i < NC * cend; i += 256) {
 *         int c = i / cend, e = i - c * cend;      // <-- cend is RUNTIME
 *         sh_x[c * FC_NX_CHUNK + e] = (c < nc) ? __ldg(...) : 0.0f;
 *     }
 *
 * `cend = min(cols - cbase, FC_NX_CHUNK)` is a runtime value, so nvcc emits a
 * real `div.s32` per staged element. In cuda/fc_cuda_ptx.h that is SEVEN
 * `div.s32` sites per kernel (three in the 0-3 element prologue peel, four in
 * the x4-unrolled main body) and it is present in ALL TWELVE batched K-quant
 * matvecs: q3k/q4k/q5k/q6k/q80 x nx/nx2/nx8. At the shipped prefill shape
 * (NC=8, cend=512, 256 threads) every thread executes NC*cend/256 = 16 of them
 * per chunk. Counting PTX minus the movs ptxas folds, the staging loop is
 * ~256 instructions + 16 divisions against an inner loop of 2 x 347, so the
 * division's share of the kernel's issue slots is 16*D/(950+16*D) for a SASS
 * expansion of D — 17 % at D=12, 25 % at D=20, 35 % at D=32.
 *
 * D IS THE UNKNOWN AND IT IS NOT MEASURED HERE. Settle it FIRST, compile-only,
 * no die required (see the ptxas/cuobjdump recipe at the bottom): count the
 * SASS instructions between the staging labels of the baseline and of
 * _stage_nested. Only then is the projected multiplier worth anything.
 *
 * Hypothesis: the batched matvecs are ISSUE-bound enough for an instruction cut
 * to convert. The prior on this hardware is sharp and two-sided — a 4.15x cut
 * in BYTES and dequant instructions returned 1.097x (moe_expert_dedup_prefill),
 * while a 10.3x cut in issue slots returned 8.26x (prefill_ssm_recurrence_warp,
 * 79 % conversion). Derived IPC for this kernel from the one measured absolute
 * (attn_qkv 8192x2048, n=128, 8-wide, 10.547 ms at 875 MHz, single die —
 * minlift_vocabsplit_ship_gate.md) is ~3.0 warp-instructions/cycle/SMX against
 * a Kepler dual-issue ceiling of 8, so it is NOT at the issue wall and the
 * conversion could be well under 79 %.
 *
 * WHAT WOULD REFUTE IT: _stage_shift within noise of the baseline. That means
 * the division is fully hidden behind warp-level parallelism and the staging
 * loop is not on the critical path — report the refutation and close the lever;
 * do NOT then go hunting for a bigger staging rewrite. A second, more
 * interesting refutation: _stage_shift wins but _stage_nested does not, which
 * would mean the nested form's own loop structure costs back what the division
 * cost. Both are cheap to distinguish because both arms are in this one module.
 *
 * BIT-EXACTNESS. Both rewrites are bit-exact BY CONSTRUCTION, which is the
 * strongest class available here: they write the same float values, loaded by
 * the same __ldg, to the same shared addresses, before the same
 * __syncthreads(), and they do not touch one floating-point operation of the
 * dot. There is no reassociation to gate. The harness proves it anyway.
 *
 * NEGATIVE CONTROLS (this harness must be able to FAIL before green means
 * anything — see the repo's "harnesses here measure nothing" lesson):
 *   1. `--self` compares the baseline against a second run of the BASELINE. It
 *      must print EXACT and a ratio inside the noise floor. If --self ever
 *      prints DIFFER the comparison is broken and every other verdict is void.
 *   2. `--poison` runs fc_matvec_q3k_nx8_stage_poison: the nested staging with
 *      ONE defect — every staged x element scaled by (1 + 2^-23), one ulp, so
 *      the dot lands 1.2e-07 RELATIVE from correct. That is inside every
 *      tolerance this tree uses, so ANY tolerance-based check passes it. This
 *      harness MUST report DIFFER. If it reports EXACT the bit comparison is
 *      not comparing bits and the exactness claim above is unproven.
 *
 * Run on a K80 host (nvcc 11.4, gcc-11 host, sm_37), ONE die, sibling verified
 * idle — two busy dies on one K80 board drop to a 398 MHz power cap. Pin the
 * clock LOW (`-ac 2505,666`) so the power floor never binds; at 875 the
 * control's spread on this box has been measured at 12.4 % against 0.66 %
 * pinned.
 *
 *   nvcc -arch=sm_37 -ccbin gcc-11 -Wno-deprecated-gpu-targets \
 *        -isystem ../nvcc_shim -I.. -O3 -Xptxas -v \
 *        -o bench_matvec_nx_staging bench_matvec_nx_staging.cu -lm
 *   sudo nvidia-smi -i <die> -ac 2505,666      # NEVER -rac: it CAPS at 562
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_matvec_nx_staging --self      # EXACT
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_matvec_nx_staging --poison    # DIFFER
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_matvec_nx_staging --shift
 *   CUDA_VISIBLE_DEVICES=<die> ./bench_matvec_nx_staging             # nested
 *   sudo nvidia-smi -i <die> -ac 2505,875      # RESTORE
 *
 * Default shape is saber's attn_qkv, 8192 x 2048 at n=128 — the largest dense
 * projection in the model and the one with a measured absolute to anchor
 * against. `--shape ROWS COLS [N]` runs any other; ssm_out is 2048 4096 and
 * ffn_down_shexp is 2048 512.
 *
 * Read the `-Xptxas -v` banner for every kernel and record the STACK FRAME, not
 * the spill counters: on this project an array indexed by a runtime-bounded
 * loop is never promoted and ptxas prints "0 bytes spill" beside a live frame.
 * A staging rewrite must not move the register count — if _stage_nested costs
 * registers it can lose residency and the projection is void.
 *
 * @status Step 1 (compile-only SASS count) is RUN: D = 14.25, MUFU.RCP 3 -> 0,
 *   registers 95 -> 80, 5 -> 6 blocks/SMX. Steps 2-4 (noise floor, --self,
 *   --poison, in-model A/B) have NOT run — no timed arm of any kind exists yet.
 *   Nothing here is reachable from Pantheon or foundry: tests/ is not in
 *   cuda_kernels/Makefile's KERNELS list, so these symbols are not in
 *   fc_cuda_ptx.h and no launcher can select them. The nested and poison arms
 *   now expand the SHIPPED macro from ../matvec_nx_staging.cuh
 *   (FC_NX_STAGE_X_NESTED), so the arm and the deployable code cannot drift.
 *   Motivated by an SASS instruction audit of the prefill MoE path; this harness
 *   answers the gate that audit pre-registered.
 * @issues
 *   - The timing is a per-kernel ratio on synthetic weights. It does NOT
 *     predict whole-prefill tok/s. Accept or reject the binding hypothesis
 *     here, then re-measure end to end with QWEN35_PROF=1 QWEN35_PROF_SUB=1 and
 *     read the `proj`, `out`, `attn_mv` and `router_shexp` spans together —
 *     the four of them are ~99 % dense matvec and ~34.6 % of prefill at 14 080.
 *   - _stage_shift asserts cend == FC_NX_CHUNK and returns without writing if
 *     not, so it is only valid where cols is a multiple of 512. Every saber
 *     dense shape is (2048, 4096, 512). It exists to isolate the division, not
 *     to ship.
 *   - Synthetic weights, not saber's. The instruction stream does not depend on
 *     the operand values, but the L2 hit pattern does; a real weight buffer of
 *     the same size behaves the same here because both arms stream the same
 *     bytes in the same order.
 * @todo None
 */

#include "../matvec_q3k.cu"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * This harness IS the A/B, so it must be built with the shipped default OFF.
 * Built with FC_NX_STAGE_NESTED=1 the baseline symbol fc_matvec_q3k_nx8 would
 * ITSELF be the nested form: every ratio would be 1.00x and every bit
 * comparison would be EXACT by construction, i.e. a green run that measures
 * nothing. Refuse at compile time rather than print it.
 */
static_assert(FC_NX_STAGE_NESTED == FC_NX_STAGE_FLAT_FORM,
              "the shipped default is now EOUTER, so this bench must be built "
              "with -DFC_NX_STAGE_NESTED=0 to keep fc_matvec_q3k_nx8 as the FLAT "
              "reference; otherwise the baseline arm IS the arm under test and "
              "every ratio is 1.00x by construction");

/* Saber's largest dense projection, at the shipped prefill chunk width. */
#define BENCH_ROWS 8192
#define BENCH_COLS 2048
#define BENCH_N    128
#define BENCH_NC   8
#define BENCH_ROWS_PER_BLOCK 16
#define BENCH_THREADS 256

/*
 * ---------------------------------------------------------------------------
 * ARM 1 — _stage_nested: the division removed by splitting the flat staging
 * index into its two real loops. Same values, same shared addresses, same
 * order of global loads within a column, same __syncthreads placement. `nc`
 * becomes loop-invariant per column, so the zero-fill separates from the copy
 * without a per-element branch.
 *
 * THIS ARM IS NOT A LOOKALIKE — it expands FC_NX_STAGE_X_NESTED from
 * ../matvec_nx_staging.cuh, the same macro fc_matvec_q3k_nx_body and the three
 * matvec_kquants.cu bodies expand when built with FC_NX_STAGE_NESTED=1. What is
 * measured here is textually what ships.
 * ---------------------------------------------------------------------------
 */
template <int NC>
__device__ __forceinline__ void bench_nx_body_nested(const unsigned char *w,
                                                     const float *x, float *out,
                                                     int rows, int cols,
                                                     int row_stride, int n) {
    extern __shared__ float sh_x[];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = (blockIdx.x * 8 + warp) * 2;
    int col0 = blockIdx.y * NC;
    int nc = n - col0;
    if (nc > NC)
        nc = NC;
    int active0 = row0 < rows;
    int active1 = row0 + 1 < rows;

    const unsigned char *b0 = w + (size_t)row0 * (size_t)row_stride;
    const unsigned char *b1 = active1 ? b0 + (size_t)row_stride : b0;

    float p0[NC], p1[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        p0[c] = 0.0f;
        p1[c] = 0.0f;
    }
    for (int cbase = 0; cbase < cols; cbase += FC_NX_CHUNK) {
        int cend = cols - cbase < FC_NX_CHUNK ? cols - cbase : FC_NX_CHUNK;
        __syncthreads();
        FC_NX_STAGE_X_NESTED(sh_x, x, col0, cols, cbase, cend, NC, nc,
                             FC_NX_STAGE_KEEP);
        __syncthreads();
        if (active0) {
            int nsb = cend >> 8;
            const float *xb = sh_x;
            for (int sb = 0; sb < nsb; sb++) {
                fc_q3k_sb_lane_dot2nx<NC>(b0, b1, xb, lane, p0, p1);
                b0 += FC_Q3K_RBLOCK_BYTES;
                b1 += FC_Q3K_RBLOCK_BYTES;
                xb += 256;
            }
        }
    }

#pragma unroll
    for (int c = 0; c < NC; c++) {
        float s0 = fc_q3k_warp_sum(p0[c]);
        float s1 = fc_q3k_warp_sum(p1[c]);
        if (c < nc && lane == 0) {
            if (active0)
                out[(size_t)(col0 + c) * rows + row0] = s0;
            if (active1)
                out[(size_t)(col0 + c) * rows + row0 + 1] = s1;
        }
    }
}

/*
 * ---------------------------------------------------------------------------
 * ARM 2 — _stage_shift: the shipped staging loop with ONE token changed,
 * `i / cend` -> `i >> 9`, valid only because cend == FC_NX_CHUNK == 512 for
 * every shape saber runs. Everything else — the flat index, the per-element
 * `c < nc` branch, the address arithmetic — is byte-for-byte the shipped form.
 * It exists to attribute: if _stage_shift captures the whole win then the
 * division alone was the defect; if only _stage_nested wins then the flat
 * index's addressing was carrying part of it.
 * ---------------------------------------------------------------------------
 */
/* The `i >> 9` below hard-codes log2(FC_NX_CHUNK). If the chunk width ever
 * changes, this arm must be updated or it silently stages the wrong columns and
 * reports the resulting garbage as a speedup. */
static_assert(FC_NX_CHUNK == 512, "bench_nx_body_shift hard-codes i >> 9");

template <int NC>
__device__ __forceinline__ void bench_nx_body_shift(const unsigned char *w,
                                                    const float *x, float *out,
                                                    int rows, int cols,
                                                    int row_stride, int n) {
    extern __shared__ float sh_x[];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = (blockIdx.x * 8 + warp) * 2;
    int col0 = blockIdx.y * NC;
    int nc = n - col0;
    if (nc > NC)
        nc = NC;
    int active0 = row0 < rows;
    int active1 = row0 + 1 < rows;

    /* Guard, not a fallback: this arm is a measurement instrument, and a
     * silently-wrong staging would be reported as a speedup. */
    if (cols % FC_NX_CHUNK != 0)
        return;

    const unsigned char *b0 = w + (size_t)row0 * (size_t)row_stride;
    const unsigned char *b1 = active1 ? b0 + (size_t)row_stride : b0;

    float p0[NC], p1[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        p0[c] = 0.0f;
        p1[c] = 0.0f;
    }
    for (int cbase = 0; cbase < cols; cbase += FC_NX_CHUNK) {
        const int cend = FC_NX_CHUNK;
        __syncthreads();
        for (int i = threadIdx.x; i < NC * cend; i += 256) {
            int c = i >> 9, e = i - (c << 9);
            sh_x[c * FC_NX_CHUNK + e] = (c < nc)
                ? __ldg(x + (size_t)(col0 + c) * cols + cbase + e) : 0.0f;
        }
        __syncthreads();
        if (active0) {
            int nsb = cend >> 8;
            const float *xb = sh_x;
            for (int sb = 0; sb < nsb; sb++) {
                fc_q3k_sb_lane_dot2nx<NC>(b0, b1, xb, lane, p0, p1);
                b0 += FC_Q3K_RBLOCK_BYTES;
                b1 += FC_Q3K_RBLOCK_BYTES;
                xb += 256;
            }
        }
    }

#pragma unroll
    for (int c = 0; c < NC; c++) {
        float s0 = fc_q3k_warp_sum(p0[c]);
        float s1 = fc_q3k_warp_sum(p1[c]);
        if (c < nc && lane == 0) {
            if (active0)
                out[(size_t)(col0 + c) * rows + row0] = s0;
            if (active1)
                out[(size_t)(col0 + c) * rows + row0 + 1] = s1;
        }
    }
}

/*
 * ---------------------------------------------------------------------------
 * NEGATIVE CONTROL — _stage_poison: _stage_nested with exactly one defect.
 * Every staged element is scaled by (1 + 2^-23) — one ulp — so the whole dot
 * comes out 1.2e-07 RELATIVE from correct. That is inside every tolerance this
 * tree uses (1e-6, 1e-5), so a tolerance-based check passes it, and it is large
 * enough to move the last bit of essentially every output, so a real bit
 * comparison cannot miss it. Calibrated deliberately: perturbing a SINGLE
 * element instead would land ~2e-10 relative, below fp32 epsilon for a
 * 2048-term sum, and could round away entirely — a negative control that
 * sometimes passes is worse than none.
 *
 * It is the SHIPPED nested staging macro with one transform substituted, so the
 * only difference between this arm and ARM 1 is a single __fmul_rn per staged
 * element. Nothing else can drift between the control and the poison.
 *
 * WHY IT CANNOT ROUND AWAY, per element: for x = m * 2^E with m in [1, 2),
 * ulp(x) = 2^-23 * 2^E and x * 2^-23 = m * ulp(x) with m in [1, 2). So
 * x * (1 + 2^-23) rounds to x + 1 ulp for m < 1.5 and x + 2 ulp for m >= 1.5 —
 * never back to x, for EVERY finite normal x. Every staged element changes, and
 * the dot of 2048 changed terms lands 1.2e-07 to 2.4e-07 relative away, which is
 * 1-2 ulps of the result. A per-element no-op is therefore impossible; only a
 * comparison that is not looking at bits can report EXACT.
 * ---------------------------------------------------------------------------
 */
#define BENCH_POISON_SCALE 1.00000011920928955078125f   /* 1 + 2^-23 */
#define BENCH_POISON(v) __fmul_rn((v), BENCH_POISON_SCALE)

template <int NC>
__device__ __forceinline__ void bench_nx_body_poison(const unsigned char *w,
                                                     const float *x, float *out,
                                                     int rows, int cols,
                                                     int row_stride, int n) {
    extern __shared__ float sh_x[];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = (blockIdx.x * 8 + warp) * 2;
    int col0 = blockIdx.y * NC;
    int nc = n - col0;
    if (nc > NC)
        nc = NC;
    int active0 = row0 < rows;
    int active1 = row0 + 1 < rows;

    const unsigned char *b0 = w + (size_t)row0 * (size_t)row_stride;
    const unsigned char *b1 = active1 ? b0 + (size_t)row_stride : b0;

    float p0[NC], p1[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        p0[c] = 0.0f;
        p1[c] = 0.0f;
    }
    for (int cbase = 0; cbase < cols; cbase += FC_NX_CHUNK) {
        int cend = cols - cbase < FC_NX_CHUNK ? cols - cbase : FC_NX_CHUNK;
        __syncthreads();
        FC_NX_STAGE_X_NESTED(sh_x, x, col0, cols, cbase, cend, NC, nc,
                             BENCH_POISON);
        __syncthreads();
        if (active0) {
            int nsb = cend >> 8;
            const float *xb = sh_x;
            for (int sb = 0; sb < nsb; sb++) {
                fc_q3k_sb_lane_dot2nx<NC>(b0, b1, xb, lane, p0, p1);
                b0 += FC_Q3K_RBLOCK_BYTES;
                b1 += FC_Q3K_RBLOCK_BYTES;
                xb += 256;
            }
        }
    }

#pragma unroll
    for (int c = 0; c < NC; c++) {
        float s0 = fc_q3k_warp_sum(p0[c]);
        float s1 = fc_q3k_warp_sum(p1[c]);
        if (c < nc && lane == 0) {
            if (active0)
                out[(size_t)(col0 + c) * rows + row0] = s0;
            if (active1)
                out[(size_t)(col0 + c) * rows + row0 + 1] = s1;
        }
    }
}

extern "C" __global__ void fc_matvec_q3k_nx8_stage_nested(
        const unsigned char *w, const float *x, float *out, int rows, int cols,
        int row_stride, int n) {
    bench_nx_body_nested<BENCH_NC>(w, x, out, rows, cols, row_stride, n);
}

extern "C" __global__ void fc_matvec_q3k_nx8_stage_shift(
        const unsigned char *w, const float *x, float *out, int rows, int cols,
        int row_stride, int n) {
    bench_nx_body_shift<BENCH_NC>(w, x, out, rows, cols, row_stride, n);
}

extern "C" __global__ void fc_matvec_q3k_nx8_stage_poison(
        const unsigned char *w, const float *x, float *out, int rows, int cols,
        int row_stride, int n) {
    bench_nx_body_poison<BENCH_NC>(w, x, out, rows, cols, row_stride, n);
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

/*
 * Synthetic REPACKED Q3_K super-blocks, 152 B each (the layout fc_repack_weight
 * produces and matvec_q3k.cu documents): u32 qw[32] @0, u8 sc[16] @128 raw
 * 6-bit, f32 d @144, pad @148. Values are arbitrary — the instruction stream
 * does not depend on them — but d is kept small so a 2048-term dot stays well
 * inside fp32 range and no arm can differ merely by producing inf.
 */
static void fill_q3k_blocks(unsigned char *buf, size_t nblocks, unsigned seed) {
    unsigned st = seed | 1u;
    for (size_t b = 0; b < nblocks; b++) {
        unsigned char *blk = buf + b * FC_Q3K_RBLOCK_BYTES;
        unsigned int *qw = (unsigned int *)blk;
        for (int i = 0; i < 32; i++) {
            st = st * 1664525u + 1013904223u;
            qw[i] = st;
        }
        for (int i = 0; i < 16; i++) {
            st = st * 1664525u + 1013904223u;
            blk[128 + i] = (unsigned char)(st >> 26);      /* 0..63 */
        }
        st = st * 1664525u + 1013904223u;
        float d = 1.0e-4f + 1.0e-4f * (float)(st >> 8) / (float)(1u << 24);
        memcpy(blk + 144, &d, sizeof(float));
        memset(blk + 148, 0, 4);
    }
}

enum bench_arm { ARM_BASE = 0, ARM_NESTED, ARM_SHIFT, ARM_POISON };

int main(int argc, char **argv) {
    int self_mode = 0, shift_mode = 0, poison_mode = 0, reps = 20;
    int rows = BENCH_ROWS, cols = BENCH_COLS, n = BENCH_N;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--self"))
            self_mode = 1;
        else if (!strcmp(argv[i], "--shift"))
            shift_mode = 1;
        else if (!strcmp(argv[i], "--poison"))
            poison_mode = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shape") && i + 2 < argc) {
            rows = atoi(argv[++i]);
            cols = atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-')
                n = atoi(argv[++i]);
        }
    }
    if (cols % 256 != 0) {
        fprintf(stderr, "cols must be a multiple of 256 (Q3_K super-block)\n");
        return 1;
    }

    /* Exactly two timed arms per run: the baseline, then whichever variant the
     * mode selects. Two arms means the comparison buffers cannot be cross-wired
     * between modes, which is how a negative control gets quietly disarmed. */
    const enum bench_arm cmp_arm = self_mode   ? ARM_BASE
                                 : poison_mode ? ARM_POISON
                                 : shift_mode  ? ARM_SHIFT
                                               : ARM_NESTED;

    const int row_stride = (cols / 256) * FC_Q3K_RBLOCK_BYTES;
    const size_t nblocks = (size_t)rows * (size_t)(cols / 256);
    const size_t wbytes = nblocks * FC_Q3K_RBLOCK_BYTES;
    const size_t n_x = (size_t)n * (size_t)cols;
    const size_t n_out = (size_t)n * (size_t)rows;

    unsigned char *h_w = (unsigned char *)malloc(wbytes);
    float *h_x = (float *)malloc(n_x * sizeof(float));
    float *h_oa = (float *)malloc(n_out * sizeof(float));
    float *h_ob = (float *)malloc(n_out * sizeof(float));
    if (!h_w || !h_x || !h_oa || !h_ob) {
        fprintf(stderr, "host alloc failed\n");
        return 1;
    }
    fill_q3k_blocks(h_w, nblocks, 77u);
    fill_deterministic(h_x, n_x, 88u, -1.0f, 1.0f);

    unsigned char *d_w;
    float *d_x, *d_out;
    CHECK(cudaMalloc(&d_w, wbytes));
    CHECK(cudaMalloc(&d_x, n_x * sizeof(float)));
    CHECK(cudaMalloc(&d_out, n_out * sizeof(float)));
    CHECK(cudaMemcpy(d_w, h_w, wbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, h_x, n_x * sizeof(float), cudaMemcpyHostToDevice));

    const unsigned gx = (unsigned)((rows + BENCH_ROWS_PER_BLOCK - 1) /
                                   BENCH_ROWS_PER_BLOCK);
    const unsigned gy = (unsigned)((n + BENCH_NC - 1) / BENCH_NC);
    const int sh_floats = cols < FC_NX_CHUNK ? cols : FC_NX_CHUNK;
    const unsigned shb = (unsigned)(BENCH_NC * sh_floats * sizeof(float));

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    float ms[2] = { 0.0f, 0.0f };

    for (int a = 0; a < 2; a++) {
        const enum bench_arm arm = (a == 0) ? ARM_BASE : cmp_arm;

#define LAUNCH_ARM()                                                          \
        do {                                                                  \
            if (arm == ARM_BASE)                                              \
                fc_matvec_q3k_nx8<<<dim3(gx, gy), BENCH_THREADS, shb>>>(      \
                    d_w, d_x, d_out, rows, cols, row_stride, n);              \
            else if (arm == ARM_NESTED)                                       \
                fc_matvec_q3k_nx8_stage_nested<<<dim3(gx, gy),                \
                    BENCH_THREADS, shb>>>(                                    \
                    d_w, d_x, d_out, rows, cols, row_stride, n);              \
            else if (arm == ARM_SHIFT)                                        \
                fc_matvec_q3k_nx8_stage_shift<<<dim3(gx, gy),                 \
                    BENCH_THREADS, shb>>>(                                    \
                    d_w, d_x, d_out, rows, cols, row_stride, n);              \
            else                                                              \
                fc_matvec_q3k_nx8_stage_poison<<<dim3(gx, gy),                \
                    BENCH_THREADS, shb>>>(                                    \
                    d_w, d_x, d_out, rows, cols, row_stride, n);              \
        } while (0)

        /* Pass 1 — CORRECTNESS. out poisoned to 0xCD first so an unwritten
         * element cannot masquerade as a match. Exactly one launch. */
        CHECK(cudaMemset(d_out, 0xCD, n_out * sizeof(float)));
        LAUNCH_ARM();
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy((a == 0) ? h_oa : h_ob, d_out,
                         n_out * sizeof(float), cudaMemcpyDeviceToHost));

        /* Pass 2 — TIMING. Back-to-back launches only; this kernel is
         * idempotent so no reset belongs inside the span. */
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

    const char *arm_name = self_mode   ? "baseline(self)"
                         : poison_mode ? "nested_POISON"
                         : shift_mode  ? "stage_shift" : "stage_nested";

    size_t diff = 0;
    double dmax = 0.0;
    for (size_t i = 0; i < n_out; i++) {
        if (memcmp(&h_oa[i], &h_ob[i], sizeof(float)) != 0) {
            diff++;
            double d = fabs((double)h_oa[i] - (double)h_ob[i]);
            if (d > dmax)
                dmax = d;
        }
    }

    /* The counted quantities the audit's projection rests on, printed so a
     * reader can check them against the PTX rather than trust the doc. */
    const int staged_per_thread = BENCH_NC * sh_floats / BENCH_THREADS;
    const int nsb_per_chunk = sh_floats >> 8;
    printf("shape        rows=%d cols=%d n=%d NC=%d row_stride=%d\n",
           rows, cols, n, BENCH_NC, row_stride);
    printf("launch       grid=(%u,%u) block=%d shared=%u B  chunks=%d\n",
           gx, gy, BENCH_THREADS, shb, (cols + FC_NX_CHUNK - 1) / FC_NX_CHUNK);
    printf("staging      %d elements/thread/chunk = %d div.s32/thread/chunk "
           "in the baseline; inner loop %d super-blocks/chunk\n",
           staged_per_thread, staged_per_thread, nsb_per_chunk);
    printf("build        FC_NX_STAGE_NESTED=%d (baseline arm is the %s form)  "
           "poison scale 1+2^-23 = %.17g\n",
           FC_NX_STAGE_NESTED, FC_NX_STAGE_NESTED ? "NESTED" : "flat",
           (double)BENCH_POISON_SCALE);
    printf("baseline     %8.4f ms/launch\n", ms[0]);
    printf("%-12s %8.4f ms/launch   ratio %.4fx\n", arm_name, ms[1],
           ms[1] > 0.0f ? ms[0] / ms[1] : 0.0f);
    printf("out bits     %s  (%zu/%zu differ, max |d| %.3e)\n",
           diff ? "DIFFER" : "EXACT", diff, n_out, dmax);
    if (self_mode && diff)
        printf("VERDICT      BROKEN — --self must be EXACT; every other "
               "verdict from this harness is void\n");
    else if (poison_mode && !diff)
        printf("VERDICT      BROKEN — --poison must be DIFFER; the bit "
               "comparison is not comparing bits\n");
    else if (self_mode)
        printf("VERDICT      control OK; ratio above is the noise floor\n");
    else if (poison_mode)
        printf("VERDICT      negative control fires\n");
    else
        printf("VERDICT      %s vs shipped fc_matvec_q3k_nx8: %s\n", arm_name,
               diff ? "NOT bit-exact — investigate before believing the ratio"
                    : "bit-exact");

    CHECK(cudaFree(d_w));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_out));
    free(h_w);
    free(h_x);
    free(h_oa);
    free(h_ob);
    return 0;
}
