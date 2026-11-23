/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file probe_nx_stage_mlp.cu
 * @brief FOLLOW-UP PROBE to bench_matvec_nx_staging.cu, written after that gate
 *   FALSIFIED its own prediction #2. Question: can a DIVISION-FREE staging form
 *   keep the flat loop's memory-level parallelism?
 *
 * WHAT THE GATE FOUND (die 6, pinned 666 MHz, noise floor 0.19 %):
 *   _stage_shift  1.2627x    (division replaced by SHR; needs a cols % 512 guard)
 *   _stage_nested 1.1511x    (division structurally absent; no guard needed)
 * despite the nested arm having FEWER counted instructions (968 vs 1034 executed
 * per thread per chunk) and BETTER occupancy (6 vs 5 blocks/SMX). Counted
 * instructions predicted the wrong winner.
 *
 * THE MECHANISM, COUNTED FROM SASS: the maximum `TEXDEPBAR` argument in the
 * staging region is the number of global loads outstanding before the first wait.
 *   fc_matvec_q3k_nx8        max TEXDEPBAR 0x3 -> 4 loads in flight
 *   _stage_shift             max TEXDEPBAR 0x3 -> 4 loads in flight
 *   _stage_nested            max TEXDEPBAR 0x0 -> 1 load in flight
 * The `#pragma unroll 1` that keeps the nested form's register count at 80 (and
 * buys the 6th resident block) also SERIALISES its loads. This staging loop is
 * latency-bound, so 4x the memory-level parallelism beat 1.27x fewer
 * instructions. That is the same lesson this hardware keeps teaching in the other
 * direction: instruction counts do not predict wall clock on their own.
 *
 * THE ARMS. All are division-free and all write the same values to the same
 * shared addresses in the same set, so all are bit-exact by the same argument as
 * FC_NX_STAGE_X_NESTED (tests/test_nx_staging_equivalence.c proves the indexing
 * for the shipped one; each arm here is compared bitwise against the flat kernel
 * at run time, which is the stronger check anyway).
 *
 *   nested   the landed form: column outer rolled, element inner rolled (1 in flight)
 *   eouter8  element outer rolled, ALL NC columns inner and fully unrolled
 *            (NC loads in flight; the form rejected earlier at 96 registers)
 *   eouter4  element outer rolled, columns in groups of 4 (4 loads in flight,
 *            matching the flat form's MLP at a lower register cost)
 *
 * POSITION BIAS. --self measured the second-timed arm as 0.49 % +/- 0.07 % slower
 * than the first across 10 processes: the die warms inside the process. This probe
 * therefore times every arm TWICE, once in forward order and once reversed, and
 * reports the mean of the two — which cancels the bias to first order instead of
 * correcting for it afterwards.
 *
 * Run on a K80 host, ONE die, sibling idle, `-ac 2505,666`, entry temperature
 * recorded:
 *   nvcc -arch=sm_37 -ccbin gcc-11 -Wno-deprecated-gpu-targets \
 *        -isystem ../nvcc_shim -I.. -O3 -o /tmp/probe_mlp probe_nx_stage_mlp.cu -lm
 *   CUDA_VISIBLE_DEVICES=6 /tmp/probe_mlp --reps 100
 *
 * @status Written 2026-08-10 in response to the gate's falsified prediction.
 * @issues Timing only, on synthetic weights; the in-model A/B is what decides.
 * @todo None
 */

#include "../matvec_q3k.cu"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static_assert(FC_NX_STAGE_NESTED == FC_NX_STAGE_FLAT_FORM,
              "the shipped default is now EOUTER, so this probe must be built with "
              "-DFC_NX_STAGE_NESTED=0 to keep fc_matvec_q3k_nx8 as the FLAT "
              "reference every arm is compared against");

#define PROBE_ROWS 8192
#define PROBE_COLS 2048
#define PROBE_N    128
#define PROBE_NC   8
#define PROBE_THREADS 256
#define PROBE_ROWS_PER_BLOCK 16

/* ARM: element outer, all NC columns inner and fully unrolled -> NC loads in
 * flight. `nc` is block-uniform so the ternary is a predicate, exactly as in the
 * shipped flat loop. */
#define PROBE_STAGE_EOUTER(SH_X, X, COL0, COLS, CBASE, CEND, NC, NC_LIVE)       \
    do {                                                                       \
        FC_NX_STAGE_ROLLED                                                     \
        for (int e = (int)threadIdx.x; e < (CEND); e += 256) {                  \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < (NC); c++)                                     \
                (SH_X)[c * FC_NX_CHUNK + e] = (c < (NC_LIVE))                   \
                    ? __ldg((X) + (size_t)((COL0) + c) * (COLS) + (CBASE) + e)  \
                    : 0.0f;                                                    \
        }                                                                      \
    } while (0)

/* ARM: same, but columns in groups of GRP so only GRP loads are outstanding —
 * matches the flat form's 4-wide MLP without holding NC values live. */
#define PROBE_STAGE_EOUTER_GRP(SH_X, X, COL0, COLS, CBASE, CEND, NC, NC_LIVE, GRP) \
    do {                                                                       \
        for (int c0 = 0; c0 < (NC); c0 += (GRP)) {                             \
            FC_NX_STAGE_ROLLED                                                 \
            for (int e = (int)threadIdx.x; e < (CEND); e += 256) {              \
                _Pragma("unroll")                                              \
                for (int g = 0; g < (GRP); g++)                                \
                    (SH_X)[(c0 + g) * FC_NX_CHUNK + e] = (c0 + g < (NC_LIVE))   \
                        ? __ldg((X) + (size_t)((COL0) + c0 + g) * (COLS)        \
                                + (CBASE) + e)                                 \
                        : 0.0f;                                                \
            }                                                                  \
        }                                                                      \
    } while (0)

#define PROBE_BODY(NAME, STAGE_STMT)                                           \
template <int NC>                                                              \
__device__ __forceinline__ void NAME(const unsigned char *w, const float *x,    \
        float *out, int rows, int cols, int row_stride, int n) {                \
    extern __shared__ float sh_x[];                                            \
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;                       \
    int row0 = (blockIdx.x * 8 + warp) * 2, col0 = blockIdx.y * NC;             \
    int nc = n - col0;                                                         \
    if (nc > NC)                                                               \
        nc = NC;                                                               \
    int active0 = row0 < rows, active1 = row0 + 1 < rows;                       \
    const unsigned char *b0 = w + (size_t)row0 * (size_t)row_stride;            \
    const unsigned char *b1 = active1 ? b0 + (size_t)row_stride : b0;           \
    float p0[NC], p1[NC];                                                      \
    _Pragma("unroll")                                                          \
    for (int c = 0; c < NC; c++) { p0[c] = 0.0f; p1[c] = 0.0f; }                \
    for (int cbase = 0; cbase < cols; cbase += FC_NX_CHUNK) {                   \
        int cend = cols - cbase < FC_NX_CHUNK ? cols - cbase : FC_NX_CHUNK;     \
        __syncthreads();                                                       \
        STAGE_STMT;                                                            \
        __syncthreads();                                                       \
        if (active0) {                                                         \
            int nsb = cend >> 8;                                               \
            const float *xb = sh_x;                                            \
            for (int sb = 0; sb < nsb; sb++) {                                 \
                fc_q3k_sb_lane_dot2nx<NC>(b0, b1, xb, lane, p0, p1);            \
                b0 += FC_Q3K_RBLOCK_BYTES;                                     \
                b1 += FC_Q3K_RBLOCK_BYTES;                                     \
                xb += 256;                                                     \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int c = 0; c < NC; c++) {                                             \
        float s0 = fc_q3k_warp_sum(p0[c]), s1 = fc_q3k_warp_sum(p1[c]);         \
        if (c < nc && lane == 0) {                                             \
            if (active0) out[(size_t)(col0 + c) * rows + row0] = s0;            \
            if (active1) out[(size_t)(col0 + c) * rows + row0 + 1] = s1;        \
        }                                                                      \
    }                                                                          \
}

PROBE_BODY(probe_body_nested,
    FC_NX_STAGE_X_NESTED(sh_x, x, col0, cols, cbase, cend, NC, nc,
                         FC_NX_STAGE_KEEP))
PROBE_BODY(probe_body_eouter8,
    PROBE_STAGE_EOUTER(sh_x, x, col0, cols, cbase, cend, NC, nc))
PROBE_BODY(probe_body_eouter4,
    PROBE_STAGE_EOUTER_GRP(sh_x, x, col0, cols, cbase, cend, NC, nc, 4))

extern "C" __global__ void probe_nx8_nested(const unsigned char *w,
        const float *x, float *out, int rows, int cols, int row_stride, int n) {
    probe_body_nested<PROBE_NC>(w, x, out, rows, cols, row_stride, n);
}
extern "C" __global__ void probe_nx8_eouter8(const unsigned char *w,
        const float *x, float *out, int rows, int cols, int row_stride, int n) {
    probe_body_eouter8<PROBE_NC>(w, x, out, rows, cols, row_stride, n);
}
extern "C" __global__ void probe_nx8_eouter4(const unsigned char *w,
        const float *x, float *out, int rows, int cols, int row_stride, int n) {
    probe_body_eouter4<PROBE_NC>(w, x, out, rows, cols, row_stride, n);
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

static void fill_deterministic(float *p, size_t n, unsigned seed, float lo,
                               float hi) {
    unsigned st = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)(st >> 8) / (float)(1u << 24);
        p[i] = lo + u * (hi - lo);
    }
}

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
            blk[128 + i] = (unsigned char)(st >> 26);
        }
        st = st * 1664525u + 1013904223u;
        float d = 1.0e-4f + 1.0e-4f * (float)(st >> 8) / (float)(1u << 24);
        memcpy(blk + 144, &d, sizeof(float));
        memset(blk + 148, 0, 4);
    }
}

#define NARMS 4
static const char *arm_name[NARMS] = { "flat (shipped)", "nested (landed)",
                                       "eouter8", "eouter4" };

int main(int argc, char **argv) {
    int reps = 100, rows = PROBE_ROWS, cols = PROBE_COLS, n = PROBE_N;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shape") && i + 2 < argc) {
            rows = atoi(argv[++i]);
            cols = atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-')
                n = atoi(argv[++i]);
        }
    }

    const int row_stride = (cols / 256) * FC_Q3K_RBLOCK_BYTES;
    const size_t nblocks = (size_t)rows * (size_t)(cols / 256);
    const size_t wbytes = nblocks * FC_Q3K_RBLOCK_BYTES;
    const size_t n_x = (size_t)n * (size_t)cols;
    const size_t n_out = (size_t)n * (size_t)rows;

    unsigned char *h_w = (unsigned char *)malloc(wbytes);
    float *h_x = (float *)malloc(n_x * sizeof(float));
    float *h_ref = (float *)malloc(n_out * sizeof(float));
    float *h_arm = (float *)malloc(n_out * sizeof(float));
    if (!h_w || !h_x || !h_ref || !h_arm) {
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

    const unsigned gx = (unsigned)((rows + PROBE_ROWS_PER_BLOCK - 1) /
                                   PROBE_ROWS_PER_BLOCK);
    const unsigned gy = (unsigned)((n + PROBE_NC - 1) / PROBE_NC);
    const int sh_floats = cols < FC_NX_CHUNK ? cols : FC_NX_CHUNK;
    const unsigned shb = (unsigned)(PROBE_NC * sh_floats * sizeof(float));

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

#define LAUNCH(a)                                                             \
    do {                                                                      \
        if ((a) == 0)                                                         \
            fc_matvec_q3k_nx8<<<dim3(gx, gy), PROBE_THREADS, shb>>>(          \
                d_w, d_x, d_out, rows, cols, row_stride, n);                   \
        else if ((a) == 1)                                                    \
            probe_nx8_nested<<<dim3(gx, gy), PROBE_THREADS, shb>>>(           \
                d_w, d_x, d_out, rows, cols, row_stride, n);                   \
        else if ((a) == 2)                                                    \
            probe_nx8_eouter8<<<dim3(gx, gy), PROBE_THREADS, shb>>>(          \
                d_w, d_x, d_out, rows, cols, row_stride, n);                   \
        else                                                                  \
            probe_nx8_eouter4<<<dim3(gx, gy), PROBE_THREADS, shb>>>(          \
                d_w, d_x, d_out, rows, cols, row_stride, n);                   \
    } while (0)

    /* Correctness: every arm bitwise against the flat kernel. */
    size_t diff[NARMS] = { 0, 0, 0, 0 };
    for (int a = 0; a < NARMS; a++) {
        CHECK(cudaMemset(d_out, 0xCD, n_out * sizeof(float)));
        LAUNCH(a);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(a == 0 ? h_ref : h_arm, d_out,
                         n_out * sizeof(float), cudaMemcpyDeviceToHost));
        if (a)
            for (size_t i = 0; i < n_out; i++)
                if (memcmp(&h_ref[i], &h_arm[i], sizeof(float)) != 0)
                    diff[a]++;
    }

    /* Timing: forward pass then reversed pass, so the in-process warming bias
     * (measured at 0.49 % against the second-timed arm) cancels to first order
     * instead of being corrected for afterwards. */
    float ms[NARMS][2];
    for (int pass = 0; pass < 2; pass++) {
        for (int k = 0; k < NARMS; k++) {
            int a = pass ? (NARMS - 1 - k) : k;
            for (int w = 0; w < 3; w++)
                LAUNCH(a);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < reps; it++)
                LAUNCH(a);
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            float total = 0.0f;
            CHECK(cudaEventElapsedTime(&total, ev0, ev1));
            ms[a][pass] = total / (float)reps;
        }
    }

    printf("shape        rows=%d cols=%d n=%d NC=%d reps=%d\n", rows, cols, n,
           PROBE_NC, reps);
    printf("%-16s %10s %10s %10s   %8s   %s\n", "arm", "fwd ms", "rev ms",
           "mean ms", "ratio", "bits vs flat");
    double base = 0.5 * ((double)ms[0][0] + (double)ms[0][1]);
    for (int a = 0; a < NARMS; a++) {
        double m = 0.5 * ((double)ms[a][0] + (double)ms[a][1]);
        printf("%-16s %10.4f %10.4f %10.4f   %8.4fx  %s", arm_name[a],
               ms[a][0], ms[a][1], m, base / m,
               a == 0 ? "reference" : (diff[a] ? "DIFFER" : "EXACT"));
        if (a && diff[a])
            printf(" (%zu/%zu)", diff[a], n_out);
        printf("\n");
    }
    printf("fwd/rev spread: ");
    for (int a = 0; a < NARMS; a++)
        printf("%s %.2f%%  ", arm_name[a],
               100.0 * fabs((double)ms[a][0] - (double)ms[a][1]) /
               (0.5 * ((double)ms[a][0] + (double)ms[a][1])));
    printf("\n");
#undef LAUNCH

    CHECK(cudaFree(d_w));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_out));
    free(h_w);
    free(h_x);
    free(h_ref);
    free(h_arm);
    return 0;
}
