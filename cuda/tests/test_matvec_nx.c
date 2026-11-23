/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_matvec_nx.c — bit-exact gate + micro-bench for the BATCHED (multi-
 * column) matvec kernels fc_matvec_*_nx against their single-column
 * counterparts on one K80 die.
 *
 * The batched-prefill correctness contract is per-COLUMN bit-exactness:
 * column t of fc_matvec_*_nx(x[n][cols]) must be bit-identical (memcmp) to
 * fc_matvec_*(x[t]) — the single-column kernels are already proven against
 * the CPU replica by bench_matvec.c, so equality here chains the proof to
 * the batched path. Weights are random quant bytes with fp16 block scales
 * clamped to a sane exponent (as in bench_matvec.c), repacked via
 * fc_repack_weight exactly like a model upload.
 *
 * Also times nx-vs-looped-single for the model's hot shapes to report the
 * batching win (the single-token kernels are instruction-issue bound on
 * sm_37, so decoding each weight block once per 4 columns should approach
 * a 2-3x effective speedup per pass).
 *
 * Build (from foundry/):
 *   gcc -O2 -I. -o cuda/tests/test_matvec_nx cuda/tests/test_matvec_nx.c \
 *       cuda/fc_cuda.c -lm -ldl
 * Run: CUDA_VISIBLE_DEVICES=<die> ./cuda/tests/test_matvec_nx
 */

#include "cuda/fc_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Batch-tile width the launcher picks for n columns of quant type t (L6
 * dual-width dispatch: n <= 4 -> the 4-wide tile, n >= 5 -> the 8-wide tile;
 * Q4_K instead crosses over at FC_Q4K_NX8_MIN_N because its min-lifted 8-wide
 * kernel is register-heavy). Only used for the effective-GB/s report; keep in
 * sync with fc_cuda.c. */
#define Q4K_NX8_MIN_N_DEFAULT 32

static int nx_tile(int t, int n)
{
    const char *e = getenv("FC_NX1_DIRECT");
    if (!(e && e[0] == '0')) {
        if (n == 1)
            return 1;                   /* L10: n==1 dispatches single-column */
        if (n == 2)
            return 2;                   /* L10: n==2 uses the 2-wide tile */
    }
    if (t == 12) {
        const char *m = getenv("FC_Q4K_NX8_MIN_N");
        int min_n = m ? atoi(m) : Q4K_NX8_MIN_N_DEFAULT;
        return (min_n > 0 && n >= min_n) ? 8 : 4;
    }
    return n >= 5 ? 8 : 4;
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static size_t raw_block_bytes(int t)
{
    switch (t) {
    case 8:  return 34;
    case 11: return 110;
    case 12: return 144;
    case 13: return 176;
    case 14: return 210;
    default: return 0;
    }
}

/* Clamp fp16 block scales to exponent 2^-4..2^-3 (random halves can be
 * inf/nan, which breaks memcmp compares). Mirrors bench_matvec.c. */
static void sanitize_quant(int t, unsigned char *p, size_t bytes)
{
    size_t bs = raw_block_bytes(t);
    size_t off;
    for (off = 0; bs && off + bs <= bytes; off += bs) {
        unsigned char *b = p + off;
        unsigned short h = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        unsigned short h2 = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        switch (t) {
        case 11: b[108] = h & 0xFF; b[109] = h >> 8; break;
        case 14: b[208] = h & 0xFF; b[209] = h >> 8; break;
        case 12: case 13:
            b[0] = h & 0xFF;  b[1] = h >> 8;
            b[2] = h2 & 0xFF; b[3] = h2 >> 8;
            break;
        case 8:  b[0] = h & 0xFF;  b[1] = h >> 8; break;
        }
    }
}

static unsigned char *rand_bytes(size_t bytes)
{
    unsigned char *p = malloc(bytes);
    size_t i;
    for (i = 0; i < bytes; i++)
        p[i] = (unsigned char)(rand() & 0xFF);
    return p;
}

static float *rand_floats(size_t n)
{
    float *p = malloc(n * sizeof(float));
    size_t i;
    for (i = 0; i < n; i++)
        p[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    return p;
}

/* Random raw quant weight -> repacked device buffer (as a model upload). */
static fc_buffer *make_weight(int t, long rows, int cols, int *dev_stride)
{
    size_t nb = (size_t)((t == 8) ? cols / 32 : cols / 256);
    size_t raw_stride = raw_block_bytes(t) * nb;
    unsigned char *raw = rand_bytes(raw_stride * (size_t)rows);
    fc_buffer *b = NULL;

    sanitize_quant(t, raw, raw_stride * (size_t)rows);
    if (t == 0) {                        /* F32 rows: no repack */
        free(raw);
        return NULL;
    }
    {
        size_t rs = fc_repack_row_bytes(t, cols);
        unsigned char *tmp = malloc(rs * (size_t)rows);
        if (tmp && fc_repack_weight(t, raw, tmp, rows, cols) == 0)
            b = fc_buffer_from_ptr(tmp, rs * (size_t)rows);
        free(tmp);
        *dev_stride = (int)rs;
    }
    free(raw);
    return b;
}

typedef void (*mv_fn)(fc_cuda_ctx *, fc_buffer *, fc_buffer *, fc_buffer *,
                      int, int, int);
typedef void (*mv_nx_fn)(fc_cuda_ctx *, fc_buffer *, fc_buffer *, fc_buffer *,
                         int, int, int, int);

/* Verify + bench one type: n-column nx output vs n single-column calls.
 * Single-column results are computed one x column at a time into a per-
 * column device buffer, exactly like the single-token forward would.
 * Returns 0 on bit-exact PASS. */
static int check_nx(const char *name, int t, mv_fn single, mv_nx_fn nx,
                    int rows, int cols, int n, int iters)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    int stride = 0;
    fc_buffer *w = NULL;
    int fail = 0, tk, i;

    if (t == 0) {                        /* F32 weight */
        float *wh = rand_floats((size_t)rows * cols);
        w = fc_buffer_from_ptr(wh, (size_t)rows * cols * sizeof(float));
        free(wh);
        stride = cols * (int)sizeof(float);
    } else {
        w = make_weight(t, rows, cols, &stride);
    }
    if (!w) {
        printf("%-14s SKIP (weight alloc failed)\n", name);
        return 1;
    }

    float *xh = rand_floats((size_t)n * cols);
    fc_buffer *xall = fc_buffer_from_ptr(xh, (size_t)n * cols * sizeof(float));
    fc_buffer *x1 = fc_buffer_alloc((size_t)cols * sizeof(float));
    fc_buffer *y1 = fc_buffer_alloc((size_t)rows * sizeof(float));
    fc_buffer *yn = fc_buffer_alloc((size_t)n * rows * sizeof(float));
    float *ref = malloc((size_t)n * rows * sizeof(float));
    float *got = malloc((size_t)n * rows * sizeof(float));

    /* Reference: n single-column launches. */
    for (tk = 0; tk < n; tk++) {
        fc_buffer_from_cpu(x1, xh + (size_t)tk * cols);
        single(ctx, w, x1, y1, rows, cols, stride);
        fc_sync();
        fc_buffer_to_cpu(y1, ref + (size_t)tk * rows);
    }

    /* Batched: one nx launch. */
    nx(ctx, w, xall, yn, rows, cols, stride, n);
    fc_sync();
    fc_buffer_to_cpu(yn, got);

    for (tk = 0; tk < n && !fail; tk++) {
        if (memcmp(ref + (size_t)tk * rows, got + (size_t)tk * rows,
                   (size_t)rows * sizeof(float)) != 0) {
            int r;
            for (r = 0; r < rows; r++) {
                float a = ref[(size_t)tk * rows + r];
                float b = got[(size_t)tk * rows + r];
                if (memcmp(&a, &b, 4) != 0) {
                    printf("%-14s FAIL col %d row %d: single=%.9g nx=%.9g\n",
                           name, tk, r, a, b);
                    break;
                }
            }
            fail = 1;
        }
    }

    if (!fail) {
        /* Bench: nx pass vs n sequential single passes (device-resident x). */
        double t0, dt_nx, dt_1;
        nx(ctx, w, xall, yn, rows, cols, stride, n);
        fc_sync();
        t0 = now_s();
        for (i = 0; i < iters; i++)
            nx(ctx, w, xall, yn, rows, cols, stride, n);
        fc_sync();
        dt_nx = (now_s() - t0) / iters;

        t0 = now_s();
        for (i = 0; i < iters; i++)
            for (tk = 0; tk < n; tk++)
                single(ctx, w, xall, y1, rows, cols, stride);
        fc_sync();
        dt_1 = (now_s() - t0) / iters;

        printf("%-14s PASS  %6dx%-5d n=%-3d  nx %8.3f ms  %dx1 %8.3f ms  "
               "speedup %.2fx  (%6.2f GB/s weight eff.)\n",
               name, rows, cols, n, dt_nx * 1e3, n, dt_1 * 1e3, dt_1 / dt_nx,
               (double)rows * stride * ((n + nx_tile(t, n) - 1) / nx_tile(t, n))
                   / dt_nx / 1e9);
    }

    free(xh); free(ref); free(got);
    fc_buffer_free(w); fc_buffer_free(xall);
    fc_buffer_free(x1); fc_buffer_free(y1); fc_buffer_free(yn);
    return fail;
}

int main(void)
{
    int fails = 0;

    if (fc_init() != 0) {
        fprintf(stderr, "no CUDA\n");
        return 1;
    }
    srand(42);

    printf("== fc_matvec_*_nx vs single-column (bit-exact gate) ==\n");
    /* Model hot shapes; the L6 dual-width dispatch means n <= 4 exercises
     * the 4-wide tile and n >= 5 the 8-wide one — cover both, plus partial
     * tiles (n=2/6/12), one 8-wide tile exactly (n=8 — the K=8
     * batched-decode shape L6 targets), and multi-tile batches. */
    /* L10 n=1: the _nx entry points dispatch the dedicated single-column
     * kernel (FC_NX1_DIRECT, default on) because a 4-wide tile fed one live
     * column runs 4x the FMA chains. Run the suite a SECOND time with
     * FC_NX1_DIRECT=0 to gate the invariant the dispatch rests on — the
     * tiled kernel at n=1 must equal the single-column kernel bit for bit.
     * The n=1 speedup column prints what the dispatch buys. */
    fails += check_nx("q3k n=1", 11, fc_matvec_q3k_ctx, fc_matvec_q3k_nx_ctx,
                      4096, 2048, 1, 40);
    fails += check_nx("q4k n=1", 12, fc_matvec_q4k_ctx, fc_matvec_q4k_nx_ctx,
                      2048, 512, 1, 40);
    fails += check_nx("q5k n=1", 13, fc_matvec_q5k_ctx, fc_matvec_q5k_nx_ctx,
                      8192, 2048, 1, 20);
    fails += check_nx("q6k head n=1", 14, fc_matvec_q6k_ctx,
                      fc_matvec_q6k_nx_ctx, 32768, 2048, 1, 10);
    fails += check_nx("q80 n=1", 8, fc_matvec_q80_ctx, fc_matvec_q80_nx_ctx,
                      4096, 2048, 1, 40);
    fails += check_nx("f32 n=1", 0, fc_matvec_f32_ctx, fc_matvec_f32_nx_ctx,
                      256, 2048, 1, 40);
    fails += check_nx("q3k n=2", 11, fc_matvec_q3k_ctx, fc_matvec_q3k_nx_ctx,
                      4096, 2048, 2, 20);
    fails += check_nx("q6k head n=2", 14, fc_matvec_q6k_ctx,
                      fc_matvec_q6k_nx_ctx, 32768, 2048, 2, 10);
    fails += check_nx("q4k n=2", 12, fc_matvec_q4k_ctx, fc_matvec_q4k_nx_ctx,
                      2048, 512, 2, 20);
    fails += check_nx("q3k n=4", 11, fc_matvec_q3k_ctx, fc_matvec_q3k_nx_ctx,
                      4096, 2048, 4, 20);
    fails += check_nx("q3k n=8", 11, fc_matvec_q3k_ctx, fc_matvec_q3k_nx_ctx,
                      4096, 2048, 8, 20);
    fails += check_nx("q6k n=4", 14, fc_matvec_q6k_ctx, fc_matvec_q6k_nx_ctx,
                      4096, 2048, 4, 10);
    fails += check_nx("q6k n=8", 14, fc_matvec_q6k_ctx, fc_matvec_q6k_nx_ctx,
                      4096, 2048, 8, 10);
    fails += check_nx("q3k n=12", 11, fc_matvec_q3k_ctx, fc_matvec_q3k_nx_ctx,
                      4096, 2048, 12, 20);
    fails += check_nx("q3k n=6", 11, fc_matvec_q3k_ctx, fc_matvec_q3k_nx_ctx,
                      512, 2048, 6, 20);
    fails += check_nx("q3k down n=32", 11, fc_matvec_q3k_ctx,
                      fc_matvec_q3k_nx_ctx, 2048, 512, 32, 20);
    fails += check_nx("q4k n=12", 12, fc_matvec_q4k_ctx, fc_matvec_q4k_nx_ctx,
                      2048, 512, 12, 20);
    fails += check_nx("q5k n=12", 13, fc_matvec_q5k_ctx, fc_matvec_q5k_nx_ctx,
                      8192, 2048, 12, 10);
    fails += check_nx("q6k n=12", 14, fc_matvec_q6k_ctx, fc_matvec_q6k_nx_ctx,
                      4096, 2048, 12, 10);
    fails += check_nx("q80 n=12", 8, fc_matvec_q80_ctx, fc_matvec_q80_nx_ctx,
                      4096, 2048, 12, 20);
    fails += check_nx("f32 n=12", 0, fc_matvec_f32_ctx, fc_matvec_f32_nx_ctx,
                      256, 2048, 12, 20);
    fails += check_nx("q5k n=128", 13, fc_matvec_q5k_ctx, fc_matvec_q5k_nx_ctx,
                      8192, 2048, 128, 4);

    /* PREFILL WIDTHS — the shapes and the n that decide FC_Q4K_NX8_MIN_N.
     * QWEN35_BS_PREFILL_CHUNK is 128 tokens, so n=128 is what every dense
     * Q4_K projection sees during a prefill; the n=12 measurement that
     * disabled fc_matvec_q4k_nx8 outright is two orders of magnitude away
     * from it. Shapes are saber's own Q4_K dense tensors (GGUF census:
     * attn_qkv x28, ffn_down_shexp x38, attn_output x10, attn_v x10 per
     * forward; ffn_down_exps goes through the grouped-MoE path instead).
     * Run the whole suite a SECOND time with FC_Q4K_NX8_MIN_N=0 to gate the
     * 4-wide fallback: both tiles must be bit-exact per column. */
    fails += check_nx("q4k qkv n=128", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 8192, 2048, 128, 10);
    fails += check_nx("q4k dshexp n=128", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 2048, 512, 128, 10);
    fails += check_nx("q4k aout n=128", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 2048, 4096, 128, 10);
    fails += check_nx("q4k av n=128", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 512, 2048, 128, 10);
    fails += check_nx("q4k qkv n=256", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 8192, 2048, 256, 6);
    /* The crossover itself: FC_Q4K_NX8_MIN_N has to hold for EVERY width a
     * prefill can present, because the last chunk is total_tokens % 128. It is
     * not a pure function of n either -- 8-wide loses 22% at n=12 on the narrow
     * 2048x512 shexp shape and wins ~7% at the same-ish n on the wide
     * 8192x2048 qkv shape -- so both shapes are swept across the boundary. */
    fails += check_nx("q4k qkv n=16", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 8192, 2048, 16, 20);
    fails += check_nx("q4k qkv n=24", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 8192, 2048, 24, 20);
    fails += check_nx("q4k qkv n=32", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 8192, 2048, 32, 20);
    fails += check_nx("q4k qkv n=12", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 8192, 2048, 12, 20);
    fails += check_nx("q4k dshexp n=16", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 2048, 512, 16, 40);
    fails += check_nx("q4k dshexp n=32", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 2048, 512, 32, 40);
    fails += check_nx("q4k dshexp n=64", 12, fc_matvec_q4k_ctx,
                      fc_matvec_q4k_nx_ctx, 2048, 512, 64, 20);

    /* Narrow rows (cols < FC_NX_CHUNK = 512): the tiles stage column c at
     * c * FC_NX_CHUNK floats, so the launch must reserve a full chunk per column
     * (it once reserved min(cols, 512) and the 2-8 wide tiles wrote past it). */
    fails += check_nx("q80 narrow n=2", 8, fc_matvec_q80_ctx, fc_matvec_q80_nx_ctx,
                      300, 256, 2, 20);
    fails += check_nx("q80 narrow n=4", 8, fc_matvec_q80_ctx, fc_matvec_q80_nx_ctx,
                      300, 256, 4, 20);
    fails += check_nx("q80 narrow n=8", 8, fc_matvec_q80_ctx, fc_matvec_q80_nx_ctx,
                      300, 256, 8, 20);
    fails += check_nx("q4k narrow n=4", 12, fc_matvec_q4k_ctx, fc_matvec_q4k_nx_ctx,
                      300, 256, 4, 20);

    /* Q8_0 multi-user widths on the Qwen3.6 TP2 per-die shapes: the row-tile grid rule
     * (fc_cuda.c) sends the small grids to _nx and keeps the tiles (r4c4 at 3 blocks/SMX)
     * on 4096 rows, so both routes are checked against the single-column kernel. */
    static const int q80_tp2[][2] = {
        { 4096, 2048 }, { 2048, 2048 }, { 1024, 4096 }, { 256, 2048 }, { 1024, 512 },
    };
    for (int i = 0; i < (int)(sizeof q80_tp2 / sizeof q80_tp2[0]); i++) {
        for (int n = 3; n <= 5; n++) {
            char name[48];
            snprintf(name, sizeof name, "q80 tp2 %dx%d n=%d", q80_tp2[i][0], q80_tp2[i][1], n);
            fails += check_nx(name, 8, fc_matvec_q80_ctx, fc_matvec_q80_nx_ctx,
                              q80_tp2[i][0], q80_tp2[i][1], n, 10);
        }
    }

    printf(fails ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    fc_free();
    return fails ? 1 : 0;
}
