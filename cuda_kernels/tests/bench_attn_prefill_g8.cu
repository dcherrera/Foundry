/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_attn_prefill_g8.cu
 * @brief Microbench + bit-exact gate for the G=8 prefill attention forms
 *   (attention_prefill_g8.cuh) against the shipped fc_attn_flash_prefill_q4_kvs,
 *   at the Qwen3.6 TP2 rank shape: head_dim 256, kv_groups 8, ONE kv head
 *   (--nkv 2 for the whole model), npos 128 (the wide-prefill chunk), q4 KV with
 *   the default fp16 sink 4 / window 128, depth 8k-32k.
 *
 * Includes attention.cu verbatim, so kvs and the shipped
 * fc_attn_flash_prefill_q4_g8 are the exact symbols the module carries.
 *
 * Each arm: one launch per kernel for the bit compare (out pre-poisoned to 0xCD),
 * then A (kvs) / B (arm) / A timing, ratio against the mean of the two A passes.
 *
 * Gate, decided before running:
 *   - kvs(self) EXACT, ratio within the A-drift.
 *   - POISON DIFFER on a large fraction (else the compare is not comparing).
 *   - every other arm EXACT on all outputs at every depth, or it is a bug.
 *   - ship the fastest EXACT form; ratios should be close to flat in depth.
 *
 * Build + run (bank, one die of dies 8-11; ~30 MB VRAM):
 *   cd foundry/cuda_kernels/tests
 *   /usr/local/cuda-11.4/bin/nvcc -arch=sm_37 -ccbin gcc-11 \
 *       -Wno-deprecated-gpu-targets -isystem ../nvcc_shim -I.. -O3 -Xptxas -v \
 *       -o build/bench_attn_prefill_g8 bench_attn_prefill_g8.cu -lm
 *   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 \
 *       build/bench_attn_prefill_g8 --sweep --depths 8192,16384,32768
 *
 * @status Compiled and run on the bank only (no nvcc on the Mac).
 * @issues Register counts are the offline ptxas allocation; the shipping path
 *   JITs from PTX.
 * @todo None
 */

#include "../attention.cu"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BG_HEAD_DIM  256
#define BG_KV_GROUPS 8
#define BG_NPOS      128
#define BG_SINK      4
#define BG_WIN       128
#define BG_ROW_BYTES ((BG_HEAD_DIM / FC_KVQ4_BLK) * FC_KVQ4_BLK_BYTES)   /* 144 */

/* kvs: kreg layout (q + score + p + m/l/corr) plus the 16-row V tile. */
#define BG_KVS_SHARED ((BG_KV_GROUPS * BG_HEAD_DIM + 2 * BG_KV_GROUPS * FC_ATTN_TILE_KEYS + \
                        3 * BG_KV_GROUPS) * 4 + FC_ATTN_TILE_KEYS * BG_ROW_BYTES)
/* G8: q + score + p + m/l/corr/m_new plus the V tile (and the K tile for KSH). */
#define BG_G8_SHARED  (FC_ATTN_G8_V_OFF * 4 + FC_ATTN_TILE_KEYS * BG_ROW_BYTES)
#define BG_KSH_SHARED (BG_G8_SHARED + FC_ATTN_TILE_KEYS * BG_ROW_BYTES)

/* Every form carries the shipped entry's launch bound (72 registers). */
#define BG_DEFINE_FORM(NAME, QV, PS, PV, KS, PO)                               \
    extern "C" __global__ void __launch_bounds__(256, 7) NAME(                 \
            const float *q, const unsigned char *k_cache,                      \
            const unsigned char *v_cache, const __half *k_sink,                \
            const __half *v_sink, const __half *k_win, const __half *v_win,    \
            float *out, int kv_groups, int head_dim, int max_seq, int pos0,    \
            int sink, int win, int attn_sink, int attn_win, float scale) {     \
        (void)kv_groups;                                                       \
        (void)head_dim;                                                        \
        fc_attn_flash_prefill_q4_g8_body<QV, PS, PV, KS, PO>(                  \
            q, k_cache, v_cache, k_sink, v_sink, k_win, v_win, out, max_seq,   \
            pos0, sink, win, attn_sink, attn_win, scale);                      \
    }

BG_DEFINE_FORM(bg_poison, true, true, false, false, true)
BG_DEFINE_FORM(bg_g8, false, false, false, false, false)
BG_DEFINE_FORM(bg_qvec, true, false, false, false, false)
BG_DEFINE_FORM(bg_psoft, false, true, false, false, false)
BG_DEFINE_FORM(bg_pvec, false, false, true, false, false)
BG_DEFINE_FORM(bg_qvec_pvec, true, false, true, false, false)
BG_DEFINE_FORM(bg_qvec_psoft, true, true, false, false, false)
BG_DEFINE_FORM(bg_ksh, true, true, false, true, false)
BG_DEFINE_FORM(bg_ksh_pvec, true, true, true, true, false)

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t rc_ = (call);                                              \
        if (rc_ != cudaSuccess) {                                              \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,     \
                    cudaGetErrorString(rc_));                                  \
            return 1;                                                          \
        }                                                                      \
    } while (0)

typedef void (*bg_kernel_fn)(const float *, const unsigned char *,
                             const unsigned char *, const __half *,
                             const __half *, const __half *, const __half *,
                             float *, int, int, int, int, int, int, int, int,
                             float);

struct bg_arm {
    const char *name;
    bg_kernel_fn fn;
    unsigned shared;
};

static const struct bg_arm bg_arms[] = {
    { "kvs(self)",  fc_attn_flash_prefill_q4_kvs, BG_KVS_SHARED },
    { "POISON",     bg_poison,                    BG_G8_SHARED },
    { "G8",         bg_g8,                        BG_G8_SHARED },
    { "QVEC",       bg_qvec,                      BG_G8_SHARED },
    { "PSOFT",      bg_psoft,                     BG_G8_SHARED },
    { "PVEC",       bg_pvec,                      BG_G8_SHARED },
    { "QVEC+PVEC",  bg_qvec_pvec,                 BG_G8_SHARED },
    { "QVEC+PSOFT", bg_qvec_psoft,                BG_G8_SHARED },
    { "QV+PS+KSH",  bg_ksh,                       BG_KSH_SHARED },
    { "+KSH+PVEC",  bg_ksh_pvec,                  BG_KSH_SHARED },
    { "ALL(g8)",    fc_attn_flash_prefill_q4_g8,  BG_G8_SHARED },
};
#define BG_N_ARMS ((int)(sizeof(bg_arms) / sizeof(bg_arms[0])))

static unsigned bg_lcg(unsigned *st) {
    *st = *st * 1664525u + 1013904223u;
    return *st;
}

static void bg_fill_f32(float *p, size_t n, unsigned seed) {
    unsigned st = seed | 1u;
    for (size_t i = 0; i < n; i++)
        p[i] = -1.0f + 2.0f * (float)(bg_lcg(&st) >> 8) / (float)(1u << 24);
}

/* Finite fp16 magnitudes ~0.25..2 (raw halves; no host fp16 needed). */
static void bg_fill_f16(unsigned short *p, size_t n, unsigned seed) {
    unsigned st = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        unsigned e = 0x0eu + (bg_lcg(&st) & 3u);
        unsigned m = bg_lcg(&st) & 0x3ffu;
        unsigned s = (bg_lcg(&st) >> 3) & 1u;
        p[i] = (unsigned short)((s << 15) | (e << 10) | m);
    }
}

/* Q4_0 rows: fp16 scale then 16 nibble-pair bytes per 32-element block. */
static void bg_fill_q4(unsigned char *buf, size_t nrows, unsigned seed) {
    unsigned st = seed | 1u;
    for (size_t r = 0; r < nrows; r++)
        for (int b = 0; b < BG_ROW_BYTES / FC_KVQ4_BLK_BYTES; b++) {
            unsigned char *blk = buf + r * BG_ROW_BYTES + b * FC_KVQ4_BLK_BYTES;
            unsigned e = 0x0cu + (bg_lcg(&st) & 3u);
            unsigned short d = (unsigned short)((1u << 15) | (e << 10) |
                                                (bg_lcg(&st) & 0x3ffu));
            blk[0] = (unsigned char)(d & 0xff);
            blk[1] = (unsigned char)(d >> 8);
            for (int i = 0; i < 16; i++)
                blk[2 + i] = (unsigned char)(bg_lcg(&st) >> 24);
        }
}

struct bg_bufs {
    float *q, *out;
    unsigned char *k, *v;
    __half *ks, *vs, *kw, *vw;
};

static void bg_launch(const struct bg_arm *a, const struct bg_bufs *b, int nkv,
                      int depth, float scale) {
    a->fn<<<dim3(BG_NPOS, nkv), BG_HEAD_DIM, a->shared>>>(
        b->q, b->k, b->v, b->ks, b->vs, b->kw, b->vw, b->out, BG_KV_GROUPS,
        BG_HEAD_DIM, depth, depth - BG_NPOS, BG_SINK, BG_WIN, 0, 0, scale);
}

/* Time `reps` launches of arm a; ms per launch. */
static double bg_time(const struct bg_arm *a, const struct bg_bufs *b, int nkv,
                      int depth, float scale, int reps, cudaEvent_t e0,
                      cudaEvent_t e1) {
    bg_launch(a, b, nkv, depth, scale);
    cudaEventRecord(e0);
    for (int i = 0; i < reps; i++)
        bg_launch(a, b, nkv, depth, scale);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    return (double)ms / reps;
}

/* One launch of arm a into host out (out pre-poisoned). */
static int bg_run_once(const struct bg_arm *a, const struct bg_bufs *b, int nkv,
                       int depth, float scale, float *h_out, size_t n_out) {
    CHECK(cudaMemset(b->out, 0xCD, n_out * sizeof(float)));
    bg_launch(a, b, nkv, depth, scale);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_out, b->out, n_out * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

static int bg_depth(int depth, int nkv, int first, int last, int reps) {
    const float scale = 1.0f / sqrtf((float)BG_HEAD_DIM);
    const size_t n_out = (size_t)BG_NPOS * nkv * BG_KV_GROUPS * BG_HEAD_DIM;
    const size_t cache = (size_t)nkv * depth * BG_ROW_BYTES;
    const size_t n_win = (size_t)nkv * FC_KVQ4_WMAX * BG_HEAD_DIM;
    const size_t n_snk = (size_t)nkv * FC_KVQ4_SINKMAX * BG_HEAD_DIM;
    float *h_q = (float *)malloc(n_out * sizeof(float));
    float *h_a = (float *)malloc(n_out * sizeof(float));
    float *h_b = (float *)malloc(n_out * sizeof(float));
    unsigned char *h_kv = (unsigned char *)malloc(cache);
    unsigned short *h_16 = (unsigned short *)malloc(n_win * 2);
    if (!h_q || !h_a || !h_b || !h_kv || !h_16)
        return 1;
    struct bg_bufs b;
    CHECK(cudaMalloc(&b.q, n_out * sizeof(float)));
    CHECK(cudaMalloc(&b.out, n_out * sizeof(float)));
    CHECK(cudaMalloc(&b.k, cache));
    CHECK(cudaMalloc(&b.v, cache));
    CHECK(cudaMalloc(&b.kw, n_win * 2));
    CHECK(cudaMalloc(&b.vw, n_win * 2));
    CHECK(cudaMalloc(&b.ks, n_snk * 2));
    CHECK(cudaMalloc(&b.vs, n_snk * 2));
    bg_fill_f32(h_q, n_out, 0x51ceu + depth);
    CHECK(cudaMemcpy(b.q, h_q, n_out * sizeof(float), cudaMemcpyHostToDevice));
    bg_fill_q4(h_kv, cache / BG_ROW_BYTES, 0x1111u);
    CHECK(cudaMemcpy(b.k, h_kv, cache, cudaMemcpyHostToDevice));
    bg_fill_q4(h_kv, cache / BG_ROW_BYTES, 0x2222u);
    CHECK(cudaMemcpy(b.v, h_kv, cache, cudaMemcpyHostToDevice));
    bg_fill_f16(h_16, n_win, 0x3333u);
    CHECK(cudaMemcpy(b.kw, h_16, n_win * 2, cudaMemcpyHostToDevice));
    bg_fill_f16(h_16, n_win, 0x4444u);
    CHECK(cudaMemcpy(b.vw, h_16, n_win * 2, cudaMemcpyHostToDevice));
    bg_fill_f16(h_16, n_snk, 0x5555u);
    CHECK(cudaMemcpy(b.ks, h_16, n_snk * 2, cudaMemcpyHostToDevice));
    bg_fill_f16(h_16, n_snk, 0x6666u);
    CHECK(cudaMemcpy(b.vs, h_16, n_snk * 2, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    if (bg_run_once(&bg_arms[0], &b, nkv, depth, scale, h_a, n_out))
        return 1;
    printf("\ndepth %d nkv %d npos %d (%d outputs)\n", depth, nkv, BG_NPOS, (int)n_out);
    printf("%-10s %9s %9s %9s %8s  %s\n", "arm", "A1 ms", "B ms", "A2 ms", "ratio",
           "bits vs kvs");
    for (int a = first; a <= last; a++) {
        if (bg_run_once(&bg_arms[a], &b, nkv, depth, scale, h_b, n_out))
            return 1;
        size_t diff = 0;
        for (size_t i = 0; i < n_out; i++)
            diff += memcmp(&h_a[i], &h_b[i], sizeof(float)) != 0;
        double ta1 = bg_time(&bg_arms[0], &b, nkv, depth, scale, reps, e0, e1);
        double tb = bg_time(&bg_arms[a], &b, nkv, depth, scale, reps, e0, e1);
        double ta2 = bg_time(&bg_arms[0], &b, nkv, depth, scale, reps, e0, e1);
        CHECK(cudaGetLastError());
        printf("%-10s %9.3f %9.3f %9.3f %7.4fx  %s %zu/%zu\n", bg_arms[a].name, ta1,
               tb, ta2, 0.5 * (ta1 + ta2) / tb, diff ? "DIFFER" : "EXACT", diff,
               n_out);
    }
    cudaFree(b.q); cudaFree(b.out); cudaFree(b.k); cudaFree(b.v);
    cudaFree(b.kw); cudaFree(b.vw); cudaFree(b.ks); cudaFree(b.vs);
    free(h_q); free(h_a); free(h_b); free(h_kv); free(h_16);
    return 0;
}

int main(int argc, char **argv) {
    int nkv = 1, reps = 3, sweep = 0, form = 0;
    const char *depths = "8192,16384,32768";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--sweep"))
            sweep = 1;
        else if (!strcmp(argv[i], "--form") && i + 1 < argc)
            form = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nkv") && i + 1 < argc)
            nkv = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depths") && i + 1 < argc)
            depths = argv[++i];
        else {
            fprintf(stderr, "usage: %s [--sweep|--form N] [--nkv 1|2] [--reps N] "
                            "[--depths a,b,c]\n", argv[0]);
            return 2;
        }
    }
    if (form < 0 || form >= BG_N_ARMS || nkv < 1 || nkv > 2)
        return 2;
    printf("arms:");
    for (int a = 0; a < BG_N_ARMS; a++) {
        cudaFuncAttributes at;
        int occ = 0;
        CHECK(cudaFuncGetAttributes(&at, (const void *)bg_arms[a].fn));
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, (const void *)bg_arms[a].fn, BG_HEAD_DIM, bg_arms[a].shared));
        printf("\n  %d %-10s REG %3d frame %4zu B shared %5u B blocks/SMX %d", a,
               bg_arms[a].name, at.numRegs, (size_t)at.localSizeBytes,
               bg_arms[a].shared, occ);
    }
    printf("\n");
    const int first = sweep ? 0 : form, last = sweep ? BG_N_ARMS - 1 : form;
    for (const char *p = depths; p && *p;) {
        int depth = atoi(p);
        if (depth > BG_NPOS && bg_depth(depth, nkv, first, last, reps))
            return 1;
        p = strchr(p, ',');
        if (p)
            p++;
    }
    return 0;
}
