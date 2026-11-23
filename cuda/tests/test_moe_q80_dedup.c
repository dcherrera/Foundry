/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_moe_q80_dedup.c
 * @brief The Q8_0 expert-grouped gate+up (fc_moe_gateup_q80_dedup) is bit-exact with
 *   fc_moe_gateup_nx_ctx (Q8_0) on every request.
 * @status complete
 * @issues none
 * @todo none
 *
 * Prefill shapes: ntok 16, 64, 128, 256; top-8 distinct experts per token (as the router
 * gives); uniform routing and a skewed one (half the slots on 4 hot experts, so lists span
 * many request tiles). Shapes: the Qwen3.6 TP2 rank (ff 256 of 512, cols 2048) and the
 * whole expert (ff 512). Gate: hid memcmp-identical. SENSITIVITY: the grouped kernel on an
 * x with ONE element moved by one ulp must differ from the nx output on the original x.
 * Also prints nx vs grouped ms at the real TP2-rank shape (256 experts, ff 256, n 128).
 *
 * Build: make USE_CUDA=1 test-moe-q80-dedup
 * Run (one die): CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 ./cuda/tests/test_moe_q80_dedup
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GGUF_Q80 8
#define TOPK     8
#define NMAX     256

static uint32_t rng_state = 88172645u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* Random GGUF Q8_0 rows repacked to the expert layout on ctx. */
static fc_buffer *q80_experts(fc_cuda_ctx *ctx, int rows, int cols, int *stride)
{
    size_t nblk = (size_t)rows * (cols / 32);
    size_t rb = fc_moe_repack_row_bytes(GGUF_Q80, cols);
    uint8_t *raw = malloc(nblk * 34), *rep = malloc((size_t)rows * rb);
    fc_buffer *w = NULL;
    for (size_t i = 0; raw && i < nblk; i++) {
        uint16_t d = (uint16_t)(0x0400 + rng() % 0x1000);
        memcpy(raw + 34 * i, &d, 2);
        for (int j = 0; j < 32; j++)
            raw[34 * i + 2 + j] = (uint8_t)rng();
    }
    if (raw && rep && rb && fc_moe_repack_weight(GGUF_Q80, raw, rep, rows, cols) == 0)
        w = fc_buffer_from_ptr_ctx(ctx, rep, (size_t)rows * rb);
    *stride = (int)rb;
    free(raw);
    free(rep);
    return w;
}

/* Top-k distinct experts per token; skew puts half the slots on experts 0..3. */
static void route(int *tidx, int n, int ne, int skew)
{
    for (int t = 0; t < n; t++)
        for (int s = 0; s < TOPK; s++) {
            int e, dup;
            do {
                e = (skew && s < TOPK / 2) ? (int)(rng() % 4u) + 4 * (s / 4)
                                            : (int)(rng() % (uint32_t)ne);
                dup = 0;
                for (int p = 0; p < s; p++)
                    dup |= tidx[t * TOPK + p] == e;
            } while (dup);
            tidx[t * TOPK + s] = e;
        }
}

static int count_diff(const float *a, const float *b, size_t n)
{
    int d = 0;
    for (size_t i = 0; i < n; i++)
        d += memcmp(&a[i], &b[i], 4) != 0;
    return d;
}

/* Work-list buffer shared by every call (sized for NMAX tokens, 256 experts). */
static fc_buffer *g_wl;

/* Grouped gate+up for n tokens into hd (lists + work list built on the device). */
static int run_dedup(fc_cuda_ctx *ctx, fc_buffer *g, fc_buffer *u, fc_buffer *bx,
                     fc_buffer *bi, fc_buffer *ec, fc_buffer *er, fc_buffer *hd, int ne,
                     int ff, int cols, int stride, int n)
{
    int wmax = fc_moe_dedup_wmax(ne, TOPK, n, n);
    if (!g_wl)
        g_wl = fc_buffer_alloc_ctx(ctx, (size_t)fc_moe_dedup_wmax(256, TOPK, NMAX, NMAX) *
                                            2 * sizeof(int));
    if (!g_wl || wmax <= 0)
        return -1;
    fc_moe_dedup_build_ctx(ctx, bi, ec, er, ne, TOPK, n, n);
    fc_moe_dedup_build_wl_ctx(ctx, ec, g_wl, ne, wmax);
    return fc_moe_gateup_q80_dedup_ctx(ctx, g, u, bx, er, g_wl, hd, TOPK, ff, cols, stride,
                                       wmax, n);
}

/* One shape: returns mismatching cases (-1 on setup failure). */
static int shape_exact(fc_cuda_ctx *ctx, int ne, int ff, int cols, int *cases)
{
    static const int ns[] = { 16, 64, 128, 256 };
    int stride = 0, bad = 0;
    fc_buffer *g = q80_experts(ctx, ne * ff, cols, &stride);
    fc_buffer *u = q80_experts(ctx, ne * ff, cols, &stride);
    float *x = malloc((size_t)NMAX * cols * sizeof(float));
    int *tidx = malloc((size_t)NMAX * TOPK * sizeof(int));
    size_t hn = (size_t)NMAX * TOPK * ff;
    float *want = malloc(hn * sizeof(float)), *got = malloc(hn * sizeof(float));
    fc_buffer *bx = NULL, *bi = fc_buffer_alloc_ctx(ctx, (size_t)NMAX * TOPK * sizeof(int));
    fc_buffer *ec = fc_buffer_alloc_ctx(ctx, (size_t)ne * sizeof(int));
    fc_buffer *er = fc_buffer_alloc_ctx(ctx, (size_t)ne * NMAX * sizeof(int));
    fc_buffer *hn_b = fc_buffer_alloc_ctx(ctx, hn * sizeof(float));
    fc_buffer *hd_b = fc_buffer_alloc_ctx(ctx, hn * sizeof(float));
    for (size_t i = 0; x && i < (size_t)NMAX * cols; i++)
        x[i] = ((float)(int)(rng() % 20001) - 10000.0f) * 1e-4f;
    if (x)
        bx = fc_buffer_from_ptr_ctx(ctx, x, (size_t)NMAX * cols * sizeof(float));
    if (!g || !u || !x || !tidx || !want || !got || !bx || !bi || !ec || !er || !hn_b ||
        !hd_b) {
        bad = -1;
        goto out;
    }
    for (int skew = 0; skew < 2; skew++)
        for (int ni = 0; ni < 4; ni++) {
            const int n = ns[ni];
            route(tidx, n, ne, skew);
            fc_buffer_from_cpu_n(bi, tidx, (size_t)n * TOPK * sizeof(int));
            if (fc_moe_gateup_nx_ctx(ctx, GGUF_Q80, g, u, bx, bi, hn_b, TOPK, ff, cols,
                                     stride, n) != 0 ||
                run_dedup(ctx, g, u, bx, bi, ec, er, hd_b, ne, ff, cols, stride, n) != 0) {
                bad++;
                continue;
            }
            fc_sync_ctx(ctx);
            fc_buffer_to_cpu_n(hn_b, want, (size_t)n * TOPK * ff * sizeof(float));
            fc_buffer_to_cpu_n(hd_b, got, (size_t)n * TOPK * ff * sizeof(float));
            (*cases)++;
            int d = count_diff(got, want, (size_t)n * TOPK * ff);
            if (d) {
                printf("    ne=%d ff=%d n=%d skew=%d: %d/%d hid differ\n", ne, ff, n, skew,
                       d, n * TOPK * ff);
                bad++;
            }
        }

    /* SENSITIVITY: one x element one ulp up must change the grouped output. */
    {
        const int n = 128;
        route(tidx, n, ne, 0);
        fc_buffer_from_cpu_n(bi, tidx, (size_t)n * TOPK * sizeof(int));
        fc_moe_gateup_nx_ctx(ctx, GGUF_Q80, g, u, bx, bi, hn_b, TOPK, ff, cols, stride, n);
        uint32_t bits;
        memcpy(&bits, &x[5 * cols + 300], 4);
        bits++;
        memcpy(&x[5 * cols + 300], &bits, 4);
        fc_sync_ctx(ctx);
        fc_buffer_from_cpu_n(bx, x, (size_t)NMAX * cols * sizeof(float));
        run_dedup(ctx, g, u, bx, bi, ec, er, hd_b, ne, ff, cols, stride, n);
        fc_sync_ctx(ctx);
        fc_buffer_to_cpu_n(hn_b, want, (size_t)n * TOPK * ff * sizeof(float));
        fc_buffer_to_cpu_n(hd_b, got, (size_t)n * TOPK * ff * sizeof(float));
        int d = count_diff(got, want, (size_t)n * TOPK * ff);
        printf("    sensitivity ne=%d ff=%d: one-ulp x -> %d hid differ\n", ne, ff, d);
        if (d == 0)
            bad++;
    }
out:
    fc_buffer_free(g); fc_buffer_free(u); fc_buffer_free(bx); fc_buffer_free(bi);
    fc_buffer_free(ec); fc_buffer_free(er); fc_buffer_free(hn_b); fc_buffer_free(hd_b);
    free(x); free(tidx); free(want); free(got);
    return bad;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* Timing only (no assert): nx vs grouped gate+up, 256 experts, ff 256, cols 2048. */
static void test_q80_dedup_speed(void)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    const int ne = 256, ff = 256, cols = 2048, n = 128, reps = 20;
    int stride = 0;
    fc_buffer *g = q80_experts(ctx, ne * ff, cols, &stride);
    fc_buffer *u = q80_experts(ctx, ne * ff, cols, &stride);
    float *x = malloc((size_t)n * cols * sizeof(float));
    int *tidx = malloc((size_t)n * TOPK * sizeof(int));
    for (size_t i = 0; x && i < (size_t)n * cols; i++)
        x[i] = ((float)(int)(rng() % 20001) - 10000.0f) * 1e-4f;
    fc_buffer *bx = x ? fc_buffer_from_ptr_ctx(ctx, x, (size_t)n * cols * sizeof(float)) : NULL;
    fc_buffer *bi = fc_buffer_alloc_ctx(ctx, (size_t)n * TOPK * sizeof(int));
    fc_buffer *ec = fc_buffer_alloc_ctx(ctx, (size_t)ne * sizeof(int));
    fc_buffer *er = fc_buffer_alloc_ctx(ctx, (size_t)ne * n * sizeof(int));
    fc_buffer *h = fc_buffer_alloc_ctx(ctx, (size_t)n * TOPK * ff * sizeof(float));
    ASSERT_TRUE(g && u && x && tidx && bx && bi && ec && er && h);
    if (g && u && x && tidx && bx && bi && ec && er && h) {
        route(tidx, n, ne, 0);
        fc_buffer_from_cpu_n(bi, tidx, (size_t)n * TOPK * sizeof(int));
        double t[2];
        for (int arm = 0; arm < 2; arm++) {
            for (int it = -2; it < reps; it++) {
                if (it == 0) {
                    fc_sync_ctx(ctx);
                    t[arm] = now_ms();
                }
                if (arm == 0)
                    fc_moe_gateup_nx_ctx(ctx, GGUF_Q80, g, u, bx, bi, h, TOPK, ff, cols, stride, n);
                else
                    run_dedup(ctx, g, u, bx, bi, ec, er, h, ne, ff, cols, stride, n);
            }
            fc_sync_ctx(ctx);
            t[arm] = (now_ms() - t[arm]) / reps;
        }
        printf("    speed ne=%d ff=%d n=%d: nx %.3f ms, grouped (incl. list build) %.3f ms, %.2fx\n",
               ne, ff, n, t[0], t[1], t[0] / t[1]);
    }
    fc_buffer_free(g); fc_buffer_free(u); fc_buffer_free(bx); fc_buffer_free(bi);
    fc_buffer_free(ec); fc_buffer_free(er); fc_buffer_free(h);
    free(x); free(tidx);
}

static void test_q80_dedup_exact(void)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    int cases = 0;
    ASSERT_TRUE(shape_exact(ctx, 64, 256, 2048, &cases) == 0);
    ASSERT_TRUE(shape_exact(ctx, 32, 512, 2048, &cases) == 0);
    printf("    %d (shape, ntok, routing) cases compared\n", cases);
}

int main(void)
{
    printf("Grouped MoE Q8_0 Dedup Gate+Up Tests\n");
    printf("====================================\n");
    if (fc_init() != 0) {
        printf("  [SKIP] no CUDA device\n");
        TEST_REPORT();
        return __ft_failed > 0 ? 1 : 0;
    }
    if (!fc_moe_gateup_quant_available_ctx(fc_default_ctx(), GGUF_Q80) ||
        !fc_moe_dedup_q80_available_ctx(fc_default_ctx())) {
        printf("  [SKIP] no Q8_0 expert / dedup kernels in the PTX\n");
        fc_free();
        TEST_REPORT();
        return __ft_failed > 0 ? 1 : 0;
    }
    RUN_TEST(q80_dedup_exact);
    RUN_TEST(q80_dedup_speed);
    fc_buffer_free(g_wl);
    fc_free();
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
