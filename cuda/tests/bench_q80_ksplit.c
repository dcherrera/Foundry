/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_q80_ksplit.c
 * @brief Q8_0 parity K-split tiles vs the full kernel at half the columns (ONE K80 die).
 *
 * At the Qwen3.8 FFN down shape (5120 x 17408) for each n: the full kernel on the whole
 * matrix, the full kernel on a 5120 x 8704 matrix (what an ideal column half costs), and
 * every parity tile on rank 0's slab (8704 compact inputs, 80 B per 128 full columns).
 * Median of TB_REPS timings of TB_ITERS launches; GB/s counts the weight bytes read.
 * Exactness is test_matvec_q80_ksplit's job; this only times.
 *
 * The shipped PTX holds the seven picked tiles; a cuda_kernels build with
 * NVCCFLAGS_EXTRA=-DFC_Q80P_SWEEP adds the other five.
 * Build: make USE_CUDA=1 cuda/tests/bench_q80_ksplit
 * Run:   CUDA_VISIBLE_DEVICES=<die> ./cuda/tests/bench_q80_ksplit [n,n,..]
 *        (default 1,2,3,4,5,8,128; env TB_REPS 7, TB_ITERS 40)
 *
 * @status Working.
 * @issues None
 * @todo None
 */

#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const int tiles[][2] = {
    { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 }, { 2, 3 }, { 2, 4 }, { 2, 5 },
    { 4, 3 }, { 4, 4 }, { 4, 5 }, { 4, 8 }, { 2, 8 },
};
#define NTILES ((int)(sizeof(tiles) / sizeof(tiles[0])))

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static uint32_t rng_state = 2463534242u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static uint8_t *q80_raw(int rows, int cols)
{
    size_t n = (size_t)rows * (cols / 32);
    uint8_t *raw = malloc(n * 34);
    for (size_t i = 0; raw && i < n; i++) {
        uint16_t d = (uint16_t)(0x0400 + rng() % 0x1000);
        memcpy(raw + 34 * i, &d, 2);
        for (int j = 0; j < 32; j++)
            raw[34 * i + 2 + j] = (uint8_t)rng();
    }
    return raw;
}

typedef struct {
    fc_cuda_ctx *ctx;
    fc_buffer *w, *x, *y;
    int rows, cols, stride, n, tr, tnc;   /* tr == 0: full kernel */
} TbRun;

static void tb_launch(const TbRun *r)
{
    if (r->tr)
        fc_matvec_q80_par_tile_ctx(r->ctx, r->w, r->x, r->y, r->rows, r->cols, r->stride,
                                   r->n, r->tr, r->tnc);
    else
        fc_matvec_q80_nx_ctx(r->ctx, r->w, r->x, r->y, r->rows, r->cols, r->stride, r->n);
}

static double tb_time_ms(const TbRun *r, int reps, int iters)
{
    double t[64];
    if (reps > 64)
        reps = 64;
    for (int i = 0; i < 5; i++)
        tb_launch(r);
    fc_sync_ctx(r->ctx);
    for (int k = 0; k < reps; k++) {
        double t0 = now_s();
        for (int i = 0; i < iters; i++)
            tb_launch(r);
        fc_sync_ctx(r->ctx);
        t[k] = (now_s() - t0) * 1e3 / iters;
    }
    qsort(t, (size_t)reps, sizeof(double), cmp_double);
    return t[reps / 2];
}

int main(int argc, char **argv)
{
    const int rows = 5120, cols = 17408, kh = cols / 2, nmax = 128;
    const char *e = getenv("TB_REPS");
    const int reps = e ? atoi(e) : 7;
    e = getenv("TB_ITERS");
    const int iters = e ? atoi(e) : 40;
    int ns[16], nn = 0;
    const char *list = argc > 1 ? argv[1] : "1,2,3,4,5,8,128";
    for (const char *p = list; *p && nn < 16;) {
        ns[nn++] = atoi(p);
        while (*p && *p != ',')
            p++;
        if (*p == ',')
            p++;
    }
    if (fc_init() != 0) {
        fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    fc_cuda_ctx *ctx = fc_default_ctx();
    uint8_t *raw = q80_raw(rows, cols), *rawh = q80_raw(rows, kh);
    size_t rbf = fc_repack_row_bytes(8, cols), rbh = fc_repack_row_bytes(8, kh);
    size_t rbp = fc_repack_q80_parity_row_bytes(cols);
    uint8_t *rf = malloc(rows * rbf), *rh = malloc(rows * rbh), *rp = malloc(rows * rbp);
    float *x = malloc((size_t)nmax * cols * sizeof(float));
    if (!raw || !rawh || !rf || !rh || !rp || !x ||
        fc_repack_weight(8, raw, rf, rows, cols) || fc_repack_weight(8, rawh, rh, rows, kh) ||
        fc_repack_weight_q80_parity(raw, rp, rows, cols, 0)) {
        fprintf(stderr, "setup failed\n");
        return 1;
    }
    for (size_t i = 0; i < (size_t)nmax * cols; i++)
        x[i] = ((float)(int)(rng() % 20001) - 10000.0f) * 1e-4f;
    fc_buffer *wf = fc_buffer_from_ptr_ctx(ctx, rf, rows * rbf);
    fc_buffer *wh = fc_buffer_from_ptr_ctx(ctx, rh, rows * rbh);
    fc_buffer *wp = fc_buffer_from_ptr_ctx(ctx, rp, rows * rbp);
    fc_buffer *bx = fc_buffer_from_ptr_ctx(ctx, x, (size_t)nmax * cols * sizeof(float));
    fc_buffer *by = fc_buffer_alloc_ctx(ctx, (size_t)nmax * rows * sizeof(float));
    printf("ffn_down %dx%d: full %zu B/row, half %zu, parity %zu; reps %d x %d\n", rows, cols,
           rbf, rbh, rbp, reps, iters);
    for (int i = 0; i < nn; i++) {
        const int n = ns[i];
        TbRun full = { ctx, wf, bx, by, rows, cols, (int)rbf, n, 0, 0 };
        TbRun half = { ctx, wh, bx, by, rows, kh, (int)rbh, n, 0, 0 };
        const double tf = tb_time_ms(&full, reps, iters), th = tb_time_ms(&half, reps, iters);
        printf("n=%-3d full %.3f ms  cols/2 %.3f ms (%.0f GB/s)\n", n, tf, th,
               rows * (double)rbh / (th * 1e6));
        double best = 1e30;
        int bi = -1;
        for (int t = 0; t < NTILES; t++) {
            TbRun pr = { ctx, wp, bx, by, rows, kh, (int)rbp, n, tiles[t][0], tiles[t][1] };
            if (fc_matvec_q80_par_tile_ctx(ctx, wp, bx, by, rows, kh, (int)rbp, n, tiles[t][0],
                                           tiles[t][1]) != 0)
                continue;   /* sweep-only tile: build the PTX with -DFC_Q80P_SWEEP */
            const double tp = tb_time_ms(&pr, reps, iters);
            printf("      r%dc%d %.3f ms  %.2fx cols/2  (%.0f GB/s)\n", tiles[t][0],
                   tiles[t][1], tp, tp / th, rows * (double)rbp / (tp * 1e6));
            if (tp < best) {
                best = tp;
                bi = t;
            }
        }
        printf("  BEST n=%d r%dc%d %.3f ms = %.2fx cols/2\n", n, tiles[bi][0], tiles[bi][1],
               best, best / th);
    }
    fc_buffer_free(wf); fc_buffer_free(wh); fc_buffer_free(wp);
    fc_buffer_free(bx); fc_buffer_free(by);
    free(raw); free(rawh); free(rf); free(rh); free(rp); free(x);
    fc_free();
    return 0;
}
