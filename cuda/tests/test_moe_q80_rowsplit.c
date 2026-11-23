/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_moe_q80_rowsplit.c
 * @brief Grouped Q8_0 MoE kernels split by OUTPUT ROWS are bit-exact with the full launch.
 * @status complete
 * @issues none
 * @todo none
 *
 * Tensor-parallel MoE (two dies of a board) gives rank h every expert's rows
 * [h*ff/2, +ff/2) of gate/up and rows [h*rows/2, +rows/2) of down, as merged slabs in
 * expert order (fc_moe_repack_weight on the gathered raw rows). The grouped kernels
 * compute every output row whole (gate/up: one warp per (row, slot); down: the slot walk,
 * the fma chain over the routing weights and both trees per row), so for ntok = 1..8,
 * 16, 128 (the __ldg and plain-load variants) the test gates:
 *   - fc_moe_gateup_nx_ctx on the two half slabs (ff/2), reassembled per (token, slot),
 *     is memcmp-identical to the full launch;
 *   - fc_moe_down_accum_nx_ctx on the two half slabs (rows/2) over the full hid,
 *     concatenated per token, is memcmp-identical to the full launch;
 *   - SENSITIVITY: down split over ff instead (each half of the inputs through the full
 *     kernel, outputs added) differs, so the memcmp can see a reassociation.
 * Shapes: the Qwen3.6-35B-A3B expert (ff 512, n_embd 2048) with 16 experts, and a small
 * one (ff 256, rows 1024, cols 512).
 *
 * Build: make USE_CUDA=1 test-moe-q80-rowsplit
 * Run (one die): CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 ./cuda/tests/test_moe_q80_rowsplit
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_Q80 8
#define TOPK     8
#define NMAX     128

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

static float *rand_f(size_t n, float scale)
{
    float *x = malloc(n * sizeof(float));
    for (size_t i = 0; x && i < n; i++)
        x[i] = ((float)(int)(rng() % 20001) - 10000.0f) * scale;
    return x;
}

/* Rows [h*per/2, +per/2) of every expert of a merged raw [ne*per][cols] tensor, in
 * expert order (h < 0: all rows), repacked to the expert layout on ctx. */
static fc_buffer *expert_slab(fc_cuda_ctx *ctx, const uint8_t *raw, int ne, int per, int cols,
                              int h, int *stride)
{
    const size_t rb_raw = (size_t)(cols / 32) * 34;
    const int keep = h < 0 ? per : per / 2, r0 = h < 0 ? 0 : h * per / 2;
    const int rows = ne * keep;
    size_t rb = fc_moe_repack_row_bytes(GGUF_Q80, cols);
    uint8_t *sub = malloc((size_t)rows * rb_raw), *rep = malloc((size_t)rows * rb);
    fc_buffer *w = NULL;
    for (int e = 0; sub && e < ne; e++)
        memcpy(sub + (size_t)e * keep * rb_raw, raw + ((size_t)e * per + r0) * rb_raw,
               (size_t)keep * rb_raw);
    if (sub && rep && rb && fc_moe_repack_weight(GGUF_Q80, sub, rep, rows, cols) == 0)
        w = fc_buffer_from_ptr_ctx(ctx, rep, (size_t)rows * rb);
    *stride = (int)rb;
    free(sub);
    free(rep);
    return w;
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

static int count_diff(const float *a, const float *b, size_t n)
{
    int d = 0;
    for (size_t i = 0; i < n; i++)
        d += memcmp(&a[i], &b[i], 4) != 0;
    return d;
}

static const int ns[] = { 1, 2, 3, 4, 5, 6, 7, 8, 16, 128 };
#define NNS ((int)(sizeof(ns) / sizeof(ns[0])))

/* One expert shape: ne experts, gate/up [ne*ff][cols], down [ne*rows][ff]. Returns the
 * mismatching (kernel, ntok) cases; *cases counts the comparisons. */
static int shape_exact(fc_cuda_ctx *ctx, int ne, int ff, int rows, int cols, int *cases)
{
    const int k = TOPK, fh = ff / 2, rh = rows / 2;
    uint8_t *graw = q80_raw(ne * ff, cols), *uraw = q80_raw(ne * ff, cols);
    uint8_t *draw = q80_raw(ne * rows, ff);
    float *x = rand_f((size_t)NMAX * cols, 1e-4f);
    float *twt = rand_f((size_t)NMAX * k, 1e-4f);
    int *tidx = malloc((size_t)NMAX * k * sizeof(int));
    float *want = malloc((size_t)NMAX * k * ff * sizeof(float));
    float *got = malloc((size_t)NMAX * k * ff * sizeof(float));
    float *half = malloc((size_t)NMAX * k * ff * sizeof(float));
    int bad = 0, sg = 0, sgh = 0, sd = 0, sdh = 0;
    /* Ids repeat across tokens (prefill shares experts) and may repeat within one. */
    for (int i = 0; tidx && i < NMAX * k; i++)
        tidx[i] = (int)(rng() % (uint32_t)ne);
    fc_buffer *gf = graw ? expert_slab(ctx, graw, ne, ff, cols, -1, &sg) : NULL;
    fc_buffer *uf = uraw ? expert_slab(ctx, uraw, ne, ff, cols, -1, &sg) : NULL;
    fc_buffer *df = draw ? expert_slab(ctx, draw, ne, rows, ff, -1, &sd) : NULL;
    fc_buffer *gh[2], *uh[2], *dh[2];
    for (int h = 0; h < 2; h++) {
        gh[h] = graw ? expert_slab(ctx, graw, ne, ff, cols, h, &sgh) : NULL;
        uh[h] = uraw ? expert_slab(ctx, uraw, ne, ff, cols, h, &sgh) : NULL;
        dh[h] = draw ? expert_slab(ctx, draw, ne, rows, ff, h, &sdh) : NULL;
    }
    fc_buffer *bx = x ? fc_buffer_from_ptr_ctx(ctx, x, (size_t)NMAX * cols * sizeof(float))
                      : NULL;
    fc_buffer *bi = tidx ? fc_buffer_from_ptr_ctx(ctx, tidx, (size_t)NMAX * k * sizeof(int))
                         : NULL;
    fc_buffer *bw = twt ? fc_buffer_from_ptr_ctx(ctx, twt, (size_t)NMAX * k * sizeof(float))
                        : NULL;
    fc_buffer *hf = fc_buffer_alloc_ctx(ctx, (size_t)NMAX * k * ff * sizeof(float));
    fc_buffer *hh = fc_buffer_alloc_ctx(ctx, (size_t)NMAX * k * fh * sizeof(float));
    fc_buffer *of = fc_buffer_alloc_ctx(ctx, (size_t)NMAX * rows * sizeof(float));
    fc_buffer *oh = fc_buffer_alloc_ctx(ctx, (size_t)NMAX * rh * sizeof(float));
    if (!want || !got || !half || !gf || !uf || !df || !gh[0] || !gh[1] || !uh[0] ||
        !uh[1] || !dh[0] || !dh[1] || !bx || !bi || !bw || !hf || !hh || !of || !oh) {
        bad = -1;
        goto out;
    }
    for (int ni = 0; ni < NNS; ni++) {
        const int n = ns[ni];
        /* gate/up: halves reassembled into [n][k][ff]. */
        if (fc_moe_gateup_nx_ctx(ctx, GGUF_Q80, gf, uf, bx, bi, hf, k, ff, cols, sg, n) != 0) {
            bad++;
            continue;
        }
        fc_sync_ctx(ctx);
        fc_buffer_to_cpu_n(hf, want, (size_t)n * k * ff * sizeof(float));
        for (int h = 0; h < 2; h++) {
            if (fc_moe_gateup_nx_ctx(ctx, GGUF_Q80, gh[h], uh[h], bx, bi, hh, k, fh, cols, sgh,
                                     n) != 0) {
                bad++;
                continue;
            }
            fc_sync_ctx(ctx);
            fc_buffer_to_cpu_n(hh, half, (size_t)n * k * fh * sizeof(float));
            for (int ts = 0; ts < n * k; ts++)
                memcpy(got + (size_t)ts * ff + h * fh, half + (size_t)ts * fh,
                       (size_t)fh * sizeof(float));
        }
        (*cases)++;
        if (memcmp(got, want, (size_t)n * k * ff * sizeof(float)) != 0) {
            printf("    gateup ff=%d n=%d: %d/%d hid differ\n", ff, n,
                   count_diff(got, want, (size_t)n * k * ff), n * k * ff);
            bad++;
        }

        /* down over the full hid: halves concatenated per token into [n][rows]. */
        fc_moe_down_accum_nx_ctx(ctx, GGUF_Q80, df, hf, bi, bw, of, k, rows, ff, sd, n);
        fc_sync_ctx(ctx);
        fc_buffer_to_cpu_n(of, want, (size_t)n * rows * sizeof(float));
        for (int h = 0; h < 2; h++) {
            fc_moe_down_accum_nx_ctx(ctx, GGUF_Q80, dh[h], hf, bi, bw, oh, k, rh, ff, sdh, n);
            fc_sync_ctx(ctx);
            fc_buffer_to_cpu_n(oh, half, (size_t)n * rh * sizeof(float));
            for (int t = 0; t < n; t++)
                memcpy(got + (size_t)t * rows + h * rh, half + (size_t)t * rh,
                       (size_t)rh * sizeof(float));
        }
        (*cases)++;
        if (memcmp(got, want, (size_t)n * rows * sizeof(float)) != 0) {
            printf("    down rows=%d n=%d: %d/%d outputs differ\n", rows, n,
                   count_diff(got, want, (size_t)n * rows), n * rows);
            bad++;
        }
    }
out:
    fc_buffer_free(gf); fc_buffer_free(uf); fc_buffer_free(df);
    for (int h = 0; h < 2; h++) {
        fc_buffer_free(gh[h]); fc_buffer_free(uh[h]); fc_buffer_free(dh[h]);
    }
    fc_buffer_free(bx); fc_buffer_free(bi); fc_buffer_free(bw);
    fc_buffer_free(hf); fc_buffer_free(hh); fc_buffer_free(of); fc_buffer_free(oh);
    free(graw); free(uraw); free(draw); free(x); free(twt); free(tidx);
    free(want); free(got); free(half);
    return bad;
}

static void test_rowsplit_exact(void)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    int cases = 0;
    ASSERT_TRUE(shape_exact(ctx, 16, 512, 2048, 2048, &cases) == 0);
    ASSERT_TRUE(shape_exact(ctx, 8, 256, 1024, 512, &cases) == 0);
    printf("    %d (shape, kernel, ntok) cases compared\n", cases);
}

/* Down split over ff (the K dim): each half of the expert inputs through the full
 * kernel, outputs added. Must differ from the full launch. */
static void test_ff_split_differs(void)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    const int ne = 16, ff = 512, rows = 2048, k = TOPK, n = 4, fh = ff / 2;
    uint8_t *draw = q80_raw(ne * rows, ff);
    uint8_t *lo = draw ? q80_cols(draw, ne * rows, ff, 0, fh) : NULL;
    uint8_t *hi = draw ? q80_cols(draw, ne * rows, ff, fh, fh) : NULL;
    float *hid = rand_f((size_t)n * k * ff, 1e-3f);
    float *hl = malloc((size_t)n * k * fh * sizeof(float));
    float *hhi = malloc((size_t)n * k * fh * sizeof(float));
    float *twt = rand_f((size_t)n * k, 1e-4f);
    int *tidx = malloc((size_t)n * k * sizeof(int));
    float *want = malloc((size_t)n * rows * sizeof(float));
    float *got = malloc((size_t)n * rows * sizeof(float));
    int sd = 0, sh = 0, diff = 0;
    ASSERT_TRUE(draw && lo && hi && hid && hl && hhi && twt && tidx && want && got);
    if (!draw || !lo || !hi || !hid || !hl || !hhi || !twt || !tidx || !want || !got)
        goto out;
    for (int i = 0; i < n * k; i++) {
        tidx[i] = (int)(rng() % (uint32_t)ne);
        memcpy(hl + (size_t)i * fh, hid + (size_t)i * ff, (size_t)fh * sizeof(float));
        memcpy(hhi + (size_t)i * fh, hid + (size_t)i * ff + fh, (size_t)fh * sizeof(float));
    }
    fc_buffer *df = expert_slab(ctx, draw, ne, rows, ff, -1, &sd);
    fc_buffer *dl = expert_slab(ctx, lo, ne, rows, fh, -1, &sh);
    fc_buffer *dhi = expert_slab(ctx, hi, ne, rows, fh, -1, &sh);
    fc_buffer *bh = fc_buffer_from_ptr_ctx(ctx, hid, (size_t)n * k * ff * sizeof(float));
    fc_buffer *bl = fc_buffer_from_ptr_ctx(ctx, hl, (size_t)n * k * fh * sizeof(float));
    fc_buffer *bhi = fc_buffer_from_ptr_ctx(ctx, hhi, (size_t)n * k * fh * sizeof(float));
    fc_buffer *bi = fc_buffer_from_ptr_ctx(ctx, tidx, (size_t)n * k * sizeof(int));
    fc_buffer *bw = fc_buffer_from_ptr_ctx(ctx, twt, (size_t)n * k * sizeof(float));
    fc_buffer *of = fc_buffer_alloc_ctx(ctx, (size_t)n * rows * sizeof(float));
    fc_buffer *ol = fc_buffer_alloc_ctx(ctx, (size_t)n * rows * sizeof(float));
    fc_buffer *oh = fc_buffer_alloc_ctx(ctx, (size_t)n * rows * sizeof(float));
    ASSERT_TRUE(df && dl && dhi && bh && bl && bhi && bi && bw && of && ol && oh);
    if (df && dl && dhi && bh && bl && bhi && bi && bw && of && ol && oh) {
        fc_moe_down_accum_nx_ctx(ctx, GGUF_Q80, df, bh, bi, bw, of, k, rows, ff, sd, n);
        fc_moe_down_accum_nx_ctx(ctx, GGUF_Q80, dl, bl, bi, bw, ol, k, rows, fh, sh, n);
        fc_moe_down_accum_nx_ctx(ctx, GGUF_Q80, dhi, bhi, bi, bw, oh, k, rows, fh, sh, n);
        fc_add_ctx(ctx, ol, oh, ol, n * rows);
        fc_sync_ctx(ctx);
        fc_buffer_to_cpu_n(of, want, (size_t)n * rows * sizeof(float));
        fc_buffer_to_cpu_n(ol, got, (size_t)n * rows * sizeof(float));
        diff = count_diff(got, want, (size_t)n * rows);
        printf("    down split over ff: %d/%d outputs differ from the full launch\n", diff,
               n * rows);
        ASSERT_TRUE(diff > 0);
    }
    fc_buffer_free(df); fc_buffer_free(dl); fc_buffer_free(dhi);
    fc_buffer_free(bh); fc_buffer_free(bl); fc_buffer_free(bhi);
    fc_buffer_free(bi); fc_buffer_free(bw);
    fc_buffer_free(of); fc_buffer_free(ol); fc_buffer_free(oh);
out:
    free(draw); free(lo); free(hi); free(hid); free(hl); free(hhi); free(twt); free(tidx);
    free(want); free(got);
}

int main(void)
{
    printf("Grouped MoE Q8_0 Row-split Tests\n");
    printf("================================\n");
    if (fc_init() != 0) {
        printf("  [SKIP] no CUDA device\n");
        TEST_REPORT();
        return __ft_failed > 0 ? 1 : 0;
    }
    if (!fc_moe_gateup_quant_available_ctx(fc_default_ctx(), GGUF_Q80) ||
        !fc_moe_down_quant_available_ctx(fc_default_ctx(), GGUF_Q80)) {
        printf("  [SKIP] no Q8_0 expert kernels in the PTX\n");
        fc_free();
        TEST_REPORT();
        return __ft_failed > 0 ? 1 : 0;
    }
    RUN_TEST(rowsplit_exact);
    RUN_TEST(ff_split_differs);
    fc_free();
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
