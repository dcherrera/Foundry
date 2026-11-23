/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_moe_q80_bw.c
 * @brief Grouped MoE Q8_0 gate+up and down at the Qwen3.6 expert shape must stream at a floor (ONE K80 die).
 * @status complete
 * @issues none
 * @todo none
 *
 * Qwen3.6-35B-A3B decode runs fc_moe_gateup_ctx / fc_moe_down_accum_ctx once per
 * layer on 8 of 256 experts (cols 2048, ff 512): ~19 MB and ~9 MB of expert
 * weights per call, the bulk of a token's weight traffic. Plain weight loads
 * ran ~118 GB/s (gate+up) and ~115 (down); __ldg loads and, for down, both
 * super-blocks of two slots' experts in flight reach ~159 and ~132. The floors
 * sit between. Bit-exactness: test_moe_expert_quants.c (q80 arms).
 *
 * The batched _nx forms serve slot decode (a few tokens) and prefill (512-token
 * chunks, tokens sharing experts). __ldg wins the first and loses the second,
 * so the host switches to plain loads above fc_moe_q80_ldg_max_tok() tokens.
 * nx_decode / nx_prefill gate both sides on 256 experts with random routing,
 * gate+up and down per call under ceilings at 4 and at 512 tokens. Die 8 sweep:
 * 4 tok gate+up 437-494 us with __ldg, 588 plain (foundry before __ldg) to 639
 * (the _nxp kernels); 512 tok gate+up 80 ms plain,
 * 159 ms __ldg, down 44 ms plain, 81 ms __ldg. Down at 4 tokens does not separate
 * the two paths (269-360 vs 344 us); its ceiling only catches a regression.
 * `test_moe_q80_bw --sweep` prints both load paths over 1..512 tokens.
 *
 * Build: make USE_CUDA=1 test-moe-q80-bw
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_EXPERT 16
#define NX_EXPERT 256   /* Qwen3.6's expert count, for the batched gates */
#define K 8
#define FF 512
#define COLS 2048

#define GATEUP_MIN_GBS 138.0   /* plain loads ~118, __ldg ~159 */
#define DOWN_MIN_GBS 123.0     /* plain loads ~115, __ldg + 2 slots in flight ~132 */
/* per batched call, us */
#define NX4_GU_MAX_US 520.0
#define NX4_DN_MAX_US 450.0
#define NX512_GU_MAX_US 100000.0
#define NX512_DN_MAX_US 55000.0

static uint32_t rng_state = 2463534242u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* n_expert * rows GGUF Q8_0 rows of cols, uploaded in the expert layout. */
static fc_buffer *q80_experts(int n_expert, int rows, int cols, int *stride)
{
    long n_rows = (long)n_expert * rows;
    size_t nb = (size_t)n_rows * (cols / 32);
    uint8_t *raw = malloc(nb * 34);
    size_t rb = fc_moe_repack_row_bytes(8, cols);
    uint8_t *rep = malloc((size_t)n_rows * rb);
    fc_buffer *w = NULL;
    for (size_t i = 0; raw && i < nb; i++) {
        uint16_t d = (uint16_t)(0x0400 + rng() % 0x1000);
        memcpy(raw + 34 * i, &d, 2);
        for (int j = 0; j < 32; j++)
            raw[34 * i + 2 + j] = (uint8_t)rng();
    }
    if (raw && rep && rb && fc_moe_repack_weight(8, raw, rep, n_rows, cols) == 0)
        w = fc_buffer_from_ptr(rep, (size_t)n_rows * rb);
    *stride = (int)rb;
    free(raw);
    free(rep);
    return w;
}

static fc_buffer *f32_vec(size_t n)
{
    float *x = malloc(n * sizeof(float));
    fc_buffer *b = NULL;
    if (x) {
        for (size_t i = 0; i < n; i++)
            x[i] = ((float)(int)(rng() % 20001) - 10000.0f) * 1e-4f;
        b = fc_buffer_from_ptr(x, n * sizeof(float));
    }
    free(x);
    return b;
}

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

/* Distinct experts 0, 2, 4, ...: every selected expert's bytes come from DRAM. */
static fc_buffer *top_idx(void)
{
    int32_t idx[K];
    for (int i = 0; i < K; i++)
        idx[i] = 2 * i;
    return fc_buffer_from_ptr(idx, sizeof(idx));
}

#define BEST_OF(us, CALL)                                                 \
    do {                                                                  \
        (us) = 1e30;                                                      \
        for (int rep = 0; rep < 5; rep++) {                               \
            for (int i = 0; i < 10; i++)                                  \
                CALL;                                                     \
            fc_sync_ctx(ctx);                                             \
            double t0 = now_us();                                         \
            for (int i = 0; i < 300; i++)                                 \
                CALL;                                                     \
            fc_sync_ctx(ctx);                                             \
            double t = (now_us() - t0) / 300;                             \
            if (t < (us))                                                 \
                (us) = t;                                                 \
        }                                                                 \
    } while (0)

static void test_gateup_bandwidth(void)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    int stride = 0, stride_u = 0;
    fc_buffer *g = q80_experts(N_EXPERT, FF, COLS, &stride);
    fc_buffer *u = q80_experts(N_EXPERT, FF, COLS, &stride_u);
    fc_buffer *x = f32_vec(COLS);
    fc_buffer *idx = top_idx();
    fc_buffer *hid = fc_buffer_alloc((size_t)K * FF * sizeof(float));
    ASSERT_TRUE(g && u && x && idx && hid);
    if (g && u && x && idx && hid) {
        double us;
        BEST_OF(us, fc_moe_gateup_ctx(ctx, 8, g, u, x, idx, hid, K, FF, COLS, stride));
        double gbs = 2.0 * K * FF * stride / (us * 1e3);
        printf("    gate+up 8x512x2048 Q8_0: %.1f us, %.1f GB/s (min %.0f)\n", us, gbs,
               GATEUP_MIN_GBS);
        ASSERT_TRUE(gbs >= GATEUP_MIN_GBS);
    }
    fc_buffer_free(g);
    fc_buffer_free(u);
    fc_buffer_free(x);
    fc_buffer_free(idx);
    fc_buffer_free(hid);
}

static void test_down_bandwidth(void)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    int stride = 0;
    fc_buffer *d = q80_experts(N_EXPERT, COLS, FF, &stride);
    fc_buffer *hid = f32_vec((size_t)K * FF);
    fc_buffer *idx = top_idx();
    fc_buffer *wt = f32_vec(K);
    fc_buffer *out = fc_buffer_alloc((size_t)COLS * sizeof(float));
    ASSERT_TRUE(d && hid && idx && wt && out);
    if (d && hid && idx && wt && out) {
        double us;
        BEST_OF(us, fc_moe_down_accum_ctx(ctx, 8, d, hid, idx, wt, out, K, COLS, FF, stride));
        double gbs = (double)K * COLS * stride / (us * 1e3);
        printf("    down 8x2048x512 Q8_0: %.1f us, %.1f GB/s (min %.0f)\n", us, gbs,
               DOWN_MIN_GBS);
        ASSERT_TRUE(gbs >= DOWN_MIN_GBS);
    }
    fc_buffer_free(d);
    fc_buffer_free(hid);
    fc_buffer_free(idx);
    fc_buffer_free(wt);
    fc_buffer_free(out);
}

/* The batched fixture: 256 experts, ntok tokens each routed to 8 distinct
 * random experts, x / weights random. */
typedef struct {
    int stride_gu, stride_d, ntok;
    fc_buffer *g, *u, *d, *x, *idx, *wt, *hid, *out;
} nx_fix;

static int nx_init(nx_fix *f, int max_tok)
{
    memset(f, 0, sizeof(*f));
    int32_t *idx = malloc((size_t)max_tok * K * sizeof(int32_t));
    for (int t = 0; idx && t < max_tok; t++)
        for (int i = 0; i < K; i++) {
            int e, dup;
            do {
                e = (int)(rng() % NX_EXPERT);
                dup = 0;
                for (int j = 0; j < i; j++)
                    dup |= idx[t * K + j] == e;
            } while (dup);
            idx[t * K + i] = e;
        }
    f->g = q80_experts(NX_EXPERT, FF, COLS, &f->stride_gu);
    f->u = q80_experts(NX_EXPERT, FF, COLS, &f->stride_gu);
    f->d = q80_experts(NX_EXPERT, COLS, FF, &f->stride_d);
    f->x = f32_vec((size_t)max_tok * COLS);
    f->idx = idx ? fc_buffer_from_ptr(idx, (size_t)max_tok * K * sizeof(int32_t)) : NULL;
    f->wt = f32_vec((size_t)max_tok * K);
    f->hid = fc_buffer_alloc((size_t)max_tok * K * FF * sizeof(float));
    f->out = fc_buffer_alloc((size_t)max_tok * COLS * sizeof(float));
    free(idx);
    return f->g && f->u && f->d && f->x && f->idx && f->wt && f->hid && f->out ? 0 : -1;
}

static void nx_free(nx_fix *f)
{
    fc_buffer_free(f->g); fc_buffer_free(f->u); fc_buffer_free(f->d);
    fc_buffer_free(f->x); fc_buffer_free(f->idx); fc_buffer_free(f->wt);
    fc_buffer_free(f->hid); fc_buffer_free(f->out);
}

/* One batched MoE call, gate+up then down, best of 5 x 300 (x 30 above 64 tokens). */
static double nx_us(nx_fix *f, int ntok, double *gu_us)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    int iters = ntok > 64 ? 30 : 300;
    double best = 1e30, best_gu = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        fc_moe_gateup_nx_ctx(ctx, 8, f->g, f->u, f->x, f->idx, f->hid, K, FF, COLS,
                             f->stride_gu, ntok);
        fc_sync_ctx(ctx);
        double t0 = now_us();
        for (int i = 0; i < iters; i++)
            fc_moe_gateup_nx_ctx(ctx, 8, f->g, f->u, f->x, f->idx, f->hid, K, FF, COLS,
                                 f->stride_gu, ntok);
        fc_sync_ctx(ctx);
        double t1 = now_us();
        for (int i = 0; i < iters; i++)
            fc_moe_down_accum_nx_ctx(ctx, 8, f->d, f->hid, f->idx, f->wt, f->out, K, COLS,
                                     FF, f->stride_d, ntok);
        fc_sync_ctx(ctx);
        double t2 = now_us();
        if ((t1 - t0) / iters < best_gu)
            best_gu = (t1 - t0) / iters;
        if ((t2 - t1) / iters < best)
            best = (t2 - t1) / iters;
    }
    if (gu_us)
        *gu_us = best_gu;
    return best_gu + best;
}

/* gate+up and down us per call at ntok tokens; -1 if the fixture failed. The
 * asserts stay in the test_ functions (lib/test.h names a failure by __func__). */
static int nx_time(int ntok, double gu_max, double dn_max, double *gu, double *dn)
{
    nx_fix f;
    int rc = -1;
    if (nx_init(&f, ntok) == 0) {
        double us = nx_us(&f, ntok, gu);
        *dn = us - *gu;
        printf("    nx %d tok (ldg <= %d): gate+up %.1f us (max %.0f), down %.1f us (max %.0f)\n",
               ntok, fc_moe_q80_ldg_max_tok(), *gu, gu_max, *dn, dn_max);
        rc = 0;
    }
    nx_free(&f);
    return rc;
}

static void test_nx_decode(void)
{
    double gu = 1e30, dn = 1e30;
    ASSERT_TRUE(nx_time(4, NX4_GU_MAX_US, NX4_DN_MAX_US, &gu, &dn) == 0);
    ASSERT_TRUE(gu <= NX4_GU_MAX_US);
    ASSERT_TRUE(dn <= NX4_DN_MAX_US);
}

static void test_nx_prefill(void)
{
    double gu = 1e30, dn = 1e30;
    ASSERT_TRUE(nx_time(512, NX512_GU_MAX_US, NX512_DN_MAX_US, &gu, &dn) == 0);
    ASSERT_TRUE(gu <= NX512_GU_MAX_US);
    ASSERT_TRUE(dn <= NX512_DN_MAX_US);
}

/* --sweep: both load paths, gate+up / down per call, 1..512 tokens. */
static void sweep(void)
{
    nx_fix f;
    if (nx_init(&f, 512) != 0)
        return;
    printf("  ntok   ldg gu    ldg dn   plain gu  plain dn  (us per call)\n");
    for (int n = 1; n <= 512; n *= 2) {
        double gl, gp;
        setenv("FC_MOE_Q80_LDG_MAX_TOK", "1000000", 1);
        double l = nx_us(&f, n, &gl);
        setenv("FC_MOE_Q80_LDG_MAX_TOK", "0", 1);
        double p = nx_us(&f, n, &gp);
        printf("  %4d %9.1f %9.1f %9.1f %9.1f\n", n, gl, l - gl, gp, p - gp);
    }
    unsetenv("FC_MOE_Q80_LDG_MAX_TOK");
    nx_free(&f);
}

int main(int argc, char **argv)
{
    printf("Grouped MoE Q8_0 Bandwidth Tests\n");
    printf("================================\n");
    if (fc_init() != 0) {
        printf("  [SKIP] no CUDA device\n");
        TEST_REPORT();
        return __ft_failed > 0 ? 1 : 0;
    }
    if (argc > 1 && strcmp(argv[1], "--sweep") == 0) {
        sweep();
        fc_free();
        return 0;
    }
    RUN_TEST(gateup_bandwidth);
    RUN_TEST(down_bandwidth);
    RUN_TEST(nx_decode);
    RUN_TEST(nx_prefill);
    fc_free();
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
