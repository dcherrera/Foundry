/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_fused_q80.c
 * @brief fc_fused_multi Q8_0 segments: bit-exact vs fc_matvec_q80, and as fast (ONE K80 die).
 * @status complete
 * @issues none
 * @todo none
 *
 * Every Q8_0 segment of one fc_fused_multi launch (kind 0, and the kind 2
 * gate+up pair) must equal the stand-alone fc_matvec_q80 (+ fc_moe_silu_mul
 * for the pair) bit for bit, over row counts that are not multiples of 32 and
 * column counts with a block tail. Speed: a fused launch holding only the
 * Qwen3.6 SSM qkv projection (8192 x 2048) must cost no more than
 * FUSED_RATIO_MAX times the stand-alone matvec; the lane-per-element fused
 * body took ~1.36x.
 *
 * Build: make USE_CUDA=1 test-fused-q80
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t rng_state = 2463534242u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* Random GGUF Q8_0 rows (fp16 d in [2^-14, 2^-10), random qs), repacked. */
static fc_buffer *q80_weight(int rows, int cols, int *stride)
{
    int nb = cols / 32;
    size_t raw_row = (size_t)nb * 34;
    uint8_t *raw = malloc((size_t)rows * raw_row);
    size_t rb = fc_repack_row_bytes(8, cols);
    uint8_t *rep = rb ? malloc((size_t)rows * rb) : NULL;
    fc_buffer *w = NULL;
    if (raw && rep) {
        for (size_t i = 0; i < (size_t)rows * nb; i++) {
            uint8_t *b = raw + 34 * i;
            uint16_t d = (uint16_t)(0x0400 + rng() % 0x1000);
            memcpy(b, &d, 2);
            for (int j = 0; j < 32; j++)
                b[2 + j] = (uint8_t)rng();
        }
        if (fc_repack_weight(8, raw, rep, rows, cols) == 0)
            w = fc_buffer_from_ptr(rep, (size_t)rows * rb);
    }
    *stride = (int)rb;
    free(raw);
    free(rep);
    return w;
}

static fc_buffer *f32_vec(int n)
{
    float *x = malloc((size_t)n * sizeof(float));
    fc_buffer *b = NULL;
    if (x) {
        for (int i = 0; i < n; i++)
            x[i] = ((float)(int)(rng() % 20001) - 10000.0f) * 1e-4f;
        b = fc_buffer_from_ptr(x, (size_t)n * sizeof(float));
    }
    free(x);
    return b;
}

static int same_bits(fc_buffer *a, fc_buffer *b, int n, const char *what)
{
    float *ha = malloc((size_t)n * sizeof(float));
    float *hb = malloc((size_t)n * sizeof(float));
    int ok = ha && hb;
    if (ok) {
        fc_buffer_to_cpu(a, ha);
        fc_buffer_to_cpu(b, hb);
        for (int i = 0; ok && i < n; i++)
            if (memcmp(&ha[i], &hb[i], sizeof(float))) {
                printf("    %s row %d: fused %a vs stand-alone %a\n", what, i,
                       ha[i], hb[i]);
                ok = 0;
            }
    }
    free(ha);
    free(hb);
    return ok;
}

/* One launch: kind-0 segments of the given rows plus one kind-2 pair. */
static int fused_matches(int cols, const int *rows, int nrows, int pair_rows)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    fc_fused_seg segs[FC_FUSED_MAX_SEG];
    fc_buffer *ref[FC_FUSED_MAX_SEG] = { 0 }, *up_ref = NULL;
    int nseg = 0, ok = 1, blocks = 0, sh = 0;
    fc_buffer *x = f32_vec(cols);
    memset(segs, 0, sizeof(segs));
    for (int i = 0; ok && i < nrows + 1; i++) {
        fc_fused_seg *s = &segs[nseg];
        int r = i < nrows ? rows[i] : pair_rows;
        s->w = q80_weight(r, cols, &s->row_stride);
        s->out = fc_buffer_alloc((size_t)r * sizeof(float));
        ref[nseg] = fc_buffer_alloc((size_t)r * sizeof(float));
        s->rows = r;
        s->cols = cols;
        s->qtype = 8;
        s->kind = i < nrows ? 0 : 2;
        if (s->kind == 2) {
            s->w2 = q80_weight(r, cols, &s->row_stride);
            up_ref = fc_buffer_alloc((size_t)r * sizeof(float));
        }
        ok = x && s->w && s->out && ref[nseg] && (s->kind == 0 || (s->w2 && up_ref));
        nseg++;
    }
    fc_buffer *desc = ok ? fc_fused_build_ctx(ctx, segs, nseg, &blocks, &sh) : NULL;
    ok = ok && desc;
    if (ok) {
        fc_fused_multi_ctx(ctx, desc, nseg, blocks, sh, x);
        for (int i = 0; i < nseg; i++) {
            fc_fused_seg *s = &segs[i];
            fc_matvec_q80_ctx(ctx, s->w, x, ref[i], s->rows, cols, s->row_stride);
            if (s->kind == 2) {
                fc_matvec_q80_ctx(ctx, s->w2, x, up_ref, s->rows, cols, s->row_stride);
                fc_moe_silu_mul_ctx(ctx, ref[i], up_ref, ref[i], s->rows);
            }
        }
        fc_sync_ctx(ctx);
        for (int i = 0; i < nseg; i++)
            if (!same_bits(segs[i].out, ref[i], segs[i].rows,
                           segs[i].kind == 2 ? "pair" : "seg"))
                ok = 0;
    }
    if (!ok)
        printf("    cols=%d: fused Q8_0 differs or failed to build\n", cols);
    fc_buffer_free(desc);
    for (int i = 0; i < nseg; i++) {
        fc_buffer_free(segs[i].w);
        fc_buffer_free(segs[i].w2);
        fc_buffer_free(segs[i].out);
        fc_buffer_free(ref[i]);
    }
    fc_buffer_free(up_ref);
    fc_buffer_free(x);
    return ok;
}

static void test_fused_q80_bit_exact(void)
{
    static const int ssm[] = { 8192, 4096, 32, 32 };     /* Qwen3.6 proj4 */
    static const int odd[] = { 1, 31, 33, 100, 257 };
    ASSERT_TRUE(fused_matches(2048, ssm, 4, 512));
    ASSERT_TRUE(fused_matches(2048, odd, 5, 77));
    ASSERT_TRUE(fused_matches(5120, odd, 5, 40));        /* 160 blocks */
    ASSERT_TRUE(fused_matches(96, odd, 3, 9));           /* 3 blocks */
    ASSERT_TRUE(fused_matches(4128, odd, 4, 31));        /* 129 blocks */
}

#define FUSED_RATIO_MAX 1.12   /* lane-per-element body: ~1.36 */

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

static void test_fused_qkv_as_fast_as_matvec(void)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    const int rows = 8192, cols = 2048, iters = 300;
    fc_fused_seg s;
    int blocks = 0, sh = 0;
    memset(&s, 0, sizeof(s));
    s.w = q80_weight(rows, cols, &s.row_stride);
    s.out = fc_buffer_alloc((size_t)rows * sizeof(float));
    s.rows = rows;
    s.cols = cols;
    s.qtype = 8;
    fc_buffer *x = f32_vec(cols);
    fc_buffer *desc = (s.w && s.out && x) ? fc_fused_build_ctx(ctx, &s, 1, &blocks, &sh) : NULL;
    ASSERT_TRUE(desc != NULL);
    if (desc) {
        double best_f = 1e30, best_m = 1e30;
        for (int rep = 0; rep < 5; rep++) {   /* interleave: clocks drift */
            for (int i = 0; i < 10; i++)
                fc_fused_multi_ctx(ctx, desc, 1, blocks, sh, x);
            fc_sync_ctx(ctx);
            double t0 = now_us();
            for (int i = 0; i < iters; i++)
                fc_fused_multi_ctx(ctx, desc, 1, blocks, sh, x);
            fc_sync_ctx(ctx);
            double f = (now_us() - t0) / iters;
            for (int i = 0; i < 10; i++)
                fc_matvec_q80_ctx(ctx, s.w, x, s.out, rows, cols, s.row_stride);
            fc_sync_ctx(ctx);
            t0 = now_us();
            for (int i = 0; i < iters; i++)
                fc_matvec_q80_ctx(ctx, s.w, x, s.out, rows, cols, s.row_stride);
            fc_sync_ctx(ctx);
            double m = (now_us() - t0) / iters;
            if (f < best_f) best_f = f;
            if (m < best_m) best_m = m;
        }
        printf("    qkv 8192x2048 Q8_0: fused %.1f us, matvec %.1f us (ratio %.2f, max %.2f)\n",
               best_f, best_m, best_f / best_m, FUSED_RATIO_MAX);
        ASSERT_TRUE(best_f < FUSED_RATIO_MAX * best_m);
    }
    fc_buffer_free(desc);
    fc_buffer_free(s.w);
    fc_buffer_free(s.out);
    fc_buffer_free(x);
}

int main(void)
{
    printf("Fused Q8_0 Segment Tests\n");
    printf("========================\n");
    if (fc_init() != 0) {
        printf("  [SKIP] no CUDA device\n");
        return 0;
    }
    RUN_TEST(fused_q80_bit_exact);
    RUN_TEST(fused_qkv_as_fast_as_matvec);
    fc_free();
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
