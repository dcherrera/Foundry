/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_matvec_q80_ksplit.c
 * @brief Q8_0 parity K-split (tensor-parallel FFN down) is bit-exact with the full kernel.
 * @status complete
 * @issues none
 * @todo none
 *
 * Rank p of a two-die split holds fc_repack_weight_q80_parity(..., p) rows and the
 * compact input x_p[i] = x[2i + p]; fc_matvec_q80_par_nx_ctx gives each row's
 * V_p + V_(p+2). The test gates, for every parity tile in the PTX (the shipped seven,
 * or all twelve from a -DFC_Q80P_SWEEP build), the default pick, and n = 1..8, 16, 128
 * on three shapes (the Qwen3.8 FFN down 5120 x 17408 among them):
 *   - out_0 + out_1 and out_1 + out_0 (fc_add_ctx) are memcmp-identical to
 *     fc_matvec_q80_nx_ctx on the full G4 weight;
 *   - the slab layout puts every scale and quant byte where the kernel reads it;
 *   - SENSITIVITY: the contiguous (Megatron) K-split of the same matrix, summed the same
 *     way, differs from the full kernel, so the memcmp can see a reassociation;
 *   - fc_buffer_copy_2d_async moves a strided block byte-exactly on one die and, when a
 *     second die and its peer link exist, die to die.
 * The red build (cuda_kernels: make NVCCFLAGS_EXTRA=-DFC_Q80P_RED, the lane tree in the
 * wrong order) must fail the exactness test.
 *
 * Build: make USE_CUDA=1 test-matvec-q80-ksplit
 * Run (two dies of one board for the peer arm): CUDA_VISIBLE_DEVICES=8,9 ...
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng_state = 2463534242u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* Random GGUF Q8_0 rows: fp16 d in [2^-14, 2^-10), random qs. */
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

/* Columns [c0, c0 + nc) of raw GGUF rows (whole 32-blocks). */
static uint8_t *q80_cols(const uint8_t *raw, int rows, int cols, int c0, int nc)
{
    uint8_t *out = malloc((size_t)rows * (nc / 32) * 34);
    for (int r = 0; out && r < rows; r++)
        memcpy(out + (size_t)r * (nc / 32) * 34,
               raw + ((size_t)r * (cols / 32) + c0 / 32) * 34, (size_t)(nc / 32) * 34);
    return out;
}

static fc_buffer *full_weight(fc_cuda_ctx *ctx, const uint8_t *raw, int rows, int cols,
                              int *stride)
{
    size_t rb = fc_repack_row_bytes(8, cols);
    uint8_t *rep = malloc((size_t)rows * rb);
    fc_buffer *w = NULL;
    if (rep && fc_repack_weight(8, raw, rep, rows, cols) == 0)
        w = fc_buffer_from_ptr_ctx(ctx, rep, (size_t)rows * rb);
    *stride = (int)rb;
    free(rep);
    return w;
}

static fc_buffer *parity_weight(fc_cuda_ctx *ctx, const uint8_t *raw, int rows, int cols,
                                int p, int *stride)
{
    size_t rb = fc_repack_q80_parity_row_bytes(cols);
    uint8_t *rep = malloc((size_t)rows * rb);
    fc_buffer *w = NULL;
    if (rep && fc_repack_weight_q80_parity(raw, rep, rows, cols, p) == 0)
        w = fc_buffer_from_ptr_ctx(ctx, rep, (size_t)rows * rb);
    *stride = (int)rb;
    free(rep);
    return w;
}

static float *rand_x(size_t n)
{
    float *x = malloc(n * sizeof(float));
    for (size_t i = 0; x && i < n; i++)
        x[i] = ((float)(int)(rng() % 20001) - 10000.0f) * 1e-4f;
    return x;
}

static int layout_ok(int rows, int cols, int p)
{
    uint8_t *raw = q80_raw(rows, cols);
    size_t rb = fc_repack_q80_parity_row_bytes(cols);
    uint8_t *rep = malloc((size_t)rows * rb);
    int ok = raw && rep && rb == (size_t)(cols / 128) * 80 &&
             fc_repack_weight_q80_parity(raw, rep, rows, cols, p) == 0;
    for (int r = 0; ok && r < rows; r++)
        for (int b = 0; ok && b < cols / 32; b++) {
            const uint8_t *src = raw + ((size_t)r * (cols / 32) + b) * 34;
            const uint8_t *grp = rep + (size_t)r * rb + (size_t)(b / 4) * 80;
            uint16_t h;
            float d;
            memcpy(&h, src, 2);
            memcpy(&d, grp + 4 * (b % 4), 4);
            float want = ldexpf(1.0f + (float)(h & 1023) / 1024.0f, ((h >> 10) & 31) - 15);
            if (d != want)
                ok = 0;
            for (int s = 0; ok && s < 8; s++)
                for (int m = 0; m < 2; m++)
                    if (grp[16 + 8 * s + 2 * (b % 4) + m] != src[2 + 4 * s + p + 2 * m])
                        ok = 0;
            if (!ok)
                printf("    cols=%d p=%d row %d block %d misplaced\n", cols, p, r, b);
        }
    free(raw);
    free(rep);
    return ok;
}

static void test_parity_layout(void)
{
    ASSERT_TRUE(layout_ok(3, 128, 0));
    ASSERT_TRUE(layout_ok(3, 128, 1));
    ASSERT_TRUE(layout_ok(2, 2176, 0));
    ASSERT_TRUE(layout_ok(2, 2176, 1));
    ASSERT_TRUE(fc_repack_q80_parity_row_bytes(96) == 0);
}

/* Shipped tiles (fc_q80p_tiles in fc_cuda.c). */
static const int tiles[][2] = {
    { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 }, { 2, 3 }, { 2, 4 }, { 2, 5 },
    { 4, 3 }, { 4, 4 }, { 4, 5 }, { 4, 8 }, { 2, 8 },
};
#define NTILES ((int)(sizeof(tiles) / sizeof(tiles[0])))
static const int ns[] = { 1, 2, 3, 4, 5, 6, 7, 8, 16, 128 };
#define NNS ((int)(sizeof(ns) / sizeof(ns[0])))

/* Every tile and n on one shape; returns mismatching (tile, n, order) cases. */
static int shape_exact(fc_cuda_ctx *ctx, int rows, int cols, int *cases)
{
    const int kh = cols / 2, nmax = 128;
    uint8_t *raw = q80_raw(rows, cols);
    float *x = rand_x((size_t)nmax * cols);
    float *xp[2] = { malloc((size_t)nmax * kh * sizeof(float)),
                     malloc((size_t)nmax * kh * sizeof(float)) };
    float *want = malloc((size_t)nmax * rows * sizeof(float));
    float *got = malloc((size_t)nmax * rows * sizeof(float));
    int sf = 0, s0 = 0, s1 = 0, bad = 0;
    for (int c = 0; xp[0] && xp[1] && x && c < nmax; c++)
        for (int i = 0; i < kh; i++)
            for (int p = 0; p < 2; p++)
                xp[p][(size_t)c * kh + i] = x[(size_t)c * cols + 2 * i + p];
    fc_buffer *wf = full_weight(ctx, raw, rows, cols, &sf);
    fc_buffer *w0 = parity_weight(ctx, raw, rows, cols, 0, &s0);
    fc_buffer *w1 = parity_weight(ctx, raw, rows, cols, 1, &s1);
    fc_buffer *bx = fc_buffer_from_ptr_ctx(ctx, x, (size_t)nmax * cols * sizeof(float));
    fc_buffer *bx0 = fc_buffer_from_ptr_ctx(ctx, xp[0], (size_t)nmax * kh * sizeof(float));
    fc_buffer *bx1 = fc_buffer_from_ptr_ctx(ctx, xp[1], (size_t)nmax * kh * sizeof(float));
    fc_buffer *yf = fc_buffer_alloc_ctx(ctx, (size_t)nmax * rows * sizeof(float));
    fc_buffer *y0 = fc_buffer_alloc_ctx(ctx, (size_t)nmax * rows * sizeof(float));
    fc_buffer *y1 = fc_buffer_alloc_ctx(ctx, (size_t)nmax * rows * sizeof(float));
    fc_buffer *ys = fc_buffer_alloc_ctx(ctx, (size_t)nmax * rows * sizeof(float));
    if (!raw || !want || !got || !wf || !w0 || !w1 || !bx || !bx0 || !bx1 || !yf || !y0 ||
        !y1 || !ys) {
        bad = -1;
        goto out;
    }
    for (int ni = 0; ni < NNS; ni++) {
        const int n = ns[ni];
        fc_matvec_q80_nx_ctx(ctx, wf, bx, yf, rows, cols, sf, n);
        fc_sync_ctx(ctx);
        fc_buffer_to_cpu_n(yf, want, (size_t)n * rows * sizeof(float));
        for (int t = 0; t < NTILES; t++) {
            if (fc_matvec_q80_par_tile_ctx(ctx, w0, bx0, y0, rows, kh, s0, n, tiles[t][0],
                                           tiles[t][1]) != 0 ||
                fc_matvec_q80_par_tile_ctx(ctx, w1, bx1, y1, rows, kh, s1, n, tiles[t][0],
                                           tiles[t][1]) != 0) {
                continue;   /* a sweep-only tile: not in the shipped PTX */
            }
            for (int order = 0; order < 2; order++) {
                fc_add_ctx(ctx, order ? y1 : y0, order ? y0 : y1, ys, n * rows);
                fc_sync_ctx(ctx);
                fc_buffer_to_cpu_n(ys, got, (size_t)n * rows * sizeof(float));
                (*cases)++;
                if (memcmp(got, want, (size_t)n * rows * sizeof(float)) != 0) {
                    int diff = 0;
                    for (int i = 0; i < n * rows; i++)
                        diff += memcmp(&got[i], &want[i], 4) != 0;
                    if (bad < 8)
                        printf("    %dx%d tile r%dc%d n=%d order %d: %d/%d outputs differ\n",
                               rows, cols, tiles[t][0], tiles[t][1], n, order, diff,
                               n * rows);
                    bad++;
                }
            }
        }
        /* The default per-n pick, once. */
        fc_matvec_q80_par_nx_ctx(ctx, w0, bx0, y0, rows, kh, s0, n);
        fc_matvec_q80_par_nx_ctx(ctx, w1, bx1, y1, rows, kh, s1, n);
        fc_add_ctx(ctx, y0, y1, ys, n * rows);
        fc_sync_ctx(ctx);
        fc_buffer_to_cpu_n(ys, got, (size_t)n * rows * sizeof(float));
        (*cases)++;
        if (memcmp(got, want, (size_t)n * rows * sizeof(float)) != 0) {
            printf("    %dx%d default pick n=%d differs\n", rows, cols, n);
            bad++;
        }
    }
out:
    fc_buffer_free(wf); fc_buffer_free(w0); fc_buffer_free(w1);
    fc_buffer_free(bx); fc_buffer_free(bx0); fc_buffer_free(bx1);
    fc_buffer_free(yf); fc_buffer_free(y0); fc_buffer_free(y1); fc_buffer_free(ys);
    free(raw); free(x); free(xp[0]); free(xp[1]); free(want); free(got);
    return bad;
}

static void test_ksplit_exact(void)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    int cases = 0;
    ASSERT_TRUE(shape_exact(ctx, 5120, 17408, &cases) == 0);
    ASSERT_TRUE(shape_exact(ctx, 300, 1152, &cases) == 0);
    /* cols >= 512: the full _nx tiles stride shared x by 512 floats per column. */
    ASSERT_TRUE(shape_exact(ctx, 97, 640, &cases) == 0);
    printf("    %d (shape, tile, n, order) cases compared\n", cases);
}

/* The contiguous K-split (each half through the full kernel, then added) must differ:
 * proves a memcmp at these shapes sees a reassociated sum. */
static void test_megatron_split_differs(void)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    const int rows = 5120, cols = 17408, kh = cols / 2, n = 4;
    uint8_t *raw = q80_raw(rows, cols);
    uint8_t *lo = raw ? q80_cols(raw, rows, cols, 0, kh) : NULL;
    uint8_t *hi = raw ? q80_cols(raw, rows, cols, kh, kh) : NULL;
    float *x = rand_x((size_t)n * cols);
    float *xl = malloc((size_t)n * kh * sizeof(float)), *xh = malloc((size_t)n * kh * sizeof(float));
    float *want = malloc((size_t)n * rows * sizeof(float)), *got = malloc((size_t)n * rows * sizeof(float));
    int sf = 0, sh = 0, diff = 0;
    ASSERT_TRUE(raw && lo && hi && x && xl && xh && want && got);
    if (!raw || !lo || !hi || !x || !xl || !xh || !want || !got)
        goto out;
    for (int c = 0; c < n; c++) {
        memcpy(xl + (size_t)c * kh, x + (size_t)c * cols, (size_t)kh * sizeof(float));
        memcpy(xh + (size_t)c * kh, x + (size_t)c * cols + kh, (size_t)kh * sizeof(float));
    }
    fc_buffer *wf = full_weight(ctx, raw, rows, cols, &sf);
    fc_buffer *wl = full_weight(ctx, lo, rows, kh, &sh);
    fc_buffer *wh = full_weight(ctx, hi, rows, kh, &sh);
    fc_buffer *bx = fc_buffer_from_ptr_ctx(ctx, x, (size_t)n * cols * sizeof(float));
    fc_buffer *bl = fc_buffer_from_ptr_ctx(ctx, xl, (size_t)n * kh * sizeof(float));
    fc_buffer *bh = fc_buffer_from_ptr_ctx(ctx, xh, (size_t)n * kh * sizeof(float));
    fc_buffer *yf = fc_buffer_alloc_ctx(ctx, (size_t)n * rows * sizeof(float));
    fc_buffer *yl = fc_buffer_alloc_ctx(ctx, (size_t)n * rows * sizeof(float));
    fc_buffer *yh = fc_buffer_alloc_ctx(ctx, (size_t)n * rows * sizeof(float));
    fc_matvec_q80_nx_ctx(ctx, wf, bx, yf, rows, cols, sf, n);
    fc_matvec_q80_nx_ctx(ctx, wl, bl, yl, rows, kh, sh, n);
    fc_matvec_q80_nx_ctx(ctx, wh, bh, yh, rows, kh, sh, n);
    fc_add_ctx(ctx, yl, yh, yl, n * rows);
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu_n(yf, want, (size_t)n * rows * sizeof(float));
    fc_buffer_to_cpu_n(yl, got, (size_t)n * rows * sizeof(float));
    for (int i = 0; i < n * rows; i++)
        diff += memcmp(&got[i], &want[i], 4) != 0;
    printf("    contiguous K-split: %d/%d outputs differ from the full kernel\n", diff,
           n * rows);
    ASSERT_TRUE(diff > 0);
    fc_buffer_free(wf); fc_buffer_free(wl); fc_buffer_free(wh);
    fc_buffer_free(bx); fc_buffer_free(bl); fc_buffer_free(bh);
    fc_buffer_free(yf); fc_buffer_free(yl); fc_buffer_free(yh);
out:
    free(raw); free(lo); free(hi); free(x); free(xl); free(xh); free(want); free(got);
}

/* 3 rows of 3 x 1 KiB chunks scattered into rows of 8 KiB at +4 KiB: the GDN core
 * gather's shape. */
/* The copy is queued on dst_ctx's stream (the reader pulls) and only that stream is
 * synced before the read. */
static int copy2d_ok(fc_cuda_ctx *src_ctx, fc_cuda_ctx *dst_ctx)
{
    const size_t w = 4096, h = 9, sp = 4096, dp = 8192, off = 4096;
    uint8_t *s = malloc(w * h), *d0 = malloc(dp * h), *got = malloc(dp * h);
    int ok = s && d0 && got;
    for (size_t i = 0; ok && i < w * h; i++)
        s[i] = (uint8_t)rng();
    for (size_t i = 0; ok && i < dp * h; i++)
        d0[i] = (uint8_t)rng();
    fc_buffer *bs = ok ? fc_buffer_from_ptr_ctx(src_ctx, s, w * h) : NULL;
    fc_buffer *bd = ok ? fc_buffer_from_ptr_ctx(dst_ctx, d0, dp * h) : NULL;
    ok = bs && bd && fc_buffer_copy_2d_async(bd, off, dp, bs, 0, sp, w, h, dst_ctx) == 0;
    /* Overflow refused, nothing queued. */
    ok = ok && fc_buffer_copy_2d_async(bd, off + 1, dp, bs, 0, sp, w, h, dst_ctx) == -1;
    if (ok) {
        fc_sync_ctx(dst_ctx);
        fc_buffer_to_cpu(bd, got);
        for (size_t r = 0; r < h; r++)
            for (size_t i = 0; i < dp; i++) {
                const uint8_t want = (i >= off && i < off + w) ? s[r * sp + i - off]
                                                                : d0[r * dp + i];
                if (got[r * dp + i] != want)
                    ok = 0;
            }
        if (!ok) {
            size_t moved = 0, kept = 0;
            for (size_t r = 0; r < h; r++)
                for (size_t i = off; i < off + w; i++) {
                    moved += got[r * dp + i] == s[r * sp + i - off];
                    kept += got[r * dp + i] == d0[r * dp + i];
                }
            printf("    2-D copy wrong: window bytes moved %zu, untouched %zu of %zu\n", moved,
                   kept, w * h);
        }
    }
    fc_buffer_free(bs);
    fc_buffer_free(bd);
    free(s); free(d0); free(got);
    return ok;
}

static void test_copy_2d(void)
{
    fc_cuda_ctx *a = fc_default_ctx();
    ASSERT_TRUE(copy2d_ok(a, a));
    fc_cuda_ctx *b = fc_cuda_init_device(1);
    if (!b) {
        printf("    one die visible: peer arm skipped\n");
        return;
    }
    if (fc_cuda_enable_peer(a, b) != 0) {
        printf("    dies 0/1 not peer-capable: peer arm skipped\n");
    } else {
        ASSERT_TRUE(copy2d_ok(a, b));
        ASSERT_TRUE(copy2d_ok(b, a));
        printf("    die-to-die 2-D copy verified both ways\n");
    }
    fc_cuda_ctx_free(b);
}

int main(void)
{
    printf("Q8_0 Parity K-split Tests\n");
    printf("=========================\n");
    RUN_TEST(parity_layout);
    if (fc_init() != 0) {
        printf("  [SKIP] no CUDA device\n");
        TEST_REPORT();
        return __ft_failed > 0 ? 1 : 0;
    }
    RUN_TEST(ksplit_exact);
    RUN_TEST(megatron_split_differs);
    RUN_TEST(copy_2d);
    fc_free();
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
