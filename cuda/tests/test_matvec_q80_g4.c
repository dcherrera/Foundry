/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_matvec_q80_g4.c
 * @brief Q8_0 G4 repack layout, and the dense Q8_0 matvec at the bandwidth it buys (ONE K80 die).
 * @status complete
 * @issues none
 * @todo none
 *
 * fc_repack_weight(8, ...) must write the G4 layout when cols % 128 == 0 (per 4
 * blocks: f32 d[4], then for lane s bytes 4s..4s+3 of blocks 0..3) and the plain
 * 36-B blocks otherwise, with unchanged row bytes. Bit-exactness of the kernels
 * on it: test_matvec_q80_parity.c (CPU replica), test_matvec_nx (batched vs
 * single), test_fused_q80 (fused vs stand-alone). Speed: fc_matvec_q80 at the Qwen3.6 SSM qkv shape
 * (8192 x 2048) must stream at least Q80_MIN_GBS; the 36-B layout reached ~127.
 *
 * Build: make USE_CUDA=1 test-matvec-q80-g4
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

/* Byte offset of quant byte e (0..31) of block b in a repacked row. */
static size_t q_off(int cols, int b, int e)
{
    if (cols % 128)
        return (size_t)36 * b + 4 + e;
    return (size_t)144 * (b / 4) + 16 + 16 * (e / 4) + 4 * (b % 4) + (e % 4);
}

static size_t d_off(int cols, int b)
{
    return cols % 128 ? (size_t)36 * b : (size_t)144 * (b / 4) + 4 * (b % 4);
}

static int layout_ok(int rows, int cols)
{
    uint8_t *raw = q80_raw(rows, cols);
    size_t rb = fc_repack_row_bytes(8, cols);
    uint8_t *rep = malloc((size_t)rows * rb);
    int ok = raw && rep && rb == (size_t)(cols / 32) * 36 &&
             fc_repack_weight(8, raw, rep, rows, cols) == 0;
    for (int r = 0; ok && r < rows; r++)
        for (int b = 0; ok && b < cols / 32; b++) {
            const uint8_t *src = raw + ((size_t)r * (cols / 32) + b) * 34;
            const uint8_t *row = rep + (size_t)r * rb;
            uint16_t h;
            float d;
            memcpy(&h, src, 2);
            memcpy(&d, row + d_off(cols, b), 4);
            /* fp16 normal -> f32 is exact: compare against the decoded half */
            float want = ldexpf(1.0f + (float)(h & 1023) / 1024.0f, ((h >> 10) & 31) - 15);
            if (d != want)
                ok = 0;
            for (int e = 0; ok && e < 32; e++)
                if (row[q_off(cols, b, e)] != src[2 + e])
                    ok = 0;
            if (!ok)
                printf("    cols=%d row %d block %d misplaced\n", cols, r, b);
        }
    free(raw);
    free(rep);
    return ok;
}

static void test_g4_layout(void)
{
    ASSERT_TRUE(layout_ok(3, 128));
    ASSERT_TRUE(layout_ok(5, 2048));
    ASSERT_TRUE(layout_ok(2, 2176));
    ASSERT_TRUE(layout_ok(3, 96));     /* plain 36-B blocks */
    ASSERT_TRUE(layout_ok(2, 2080));
}

static fc_buffer *q80_weight(int rows, int cols, int *stride)
{
    uint8_t *raw = q80_raw(rows, cols);
    size_t rb = fc_repack_row_bytes(8, cols);
    uint8_t *rep = malloc((size_t)rows * rb);
    fc_buffer *w = NULL;
    if (raw && rep && fc_repack_weight(8, raw, rep, rows, cols) == 0)
        w = fc_buffer_from_ptr(rep, (size_t)rows * rb);
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

#define Q80_MIN_GBS 136.0   /* G4 ~146 GB/s, 36-B layout ~127 at this shape */

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

static void test_qkv_bandwidth(void)
{
    fc_cuda_ctx *ctx = fc_default_ctx();
    const int rows = 8192, cols = 2048, iters = 300;
    int stride = 0;
    fc_buffer *w = q80_weight(rows, cols, &stride);
    fc_buffer *x = f32_vec(cols);
    fc_buffer *y = fc_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_TRUE(w && x && y);
    if (w && x && y) {
        double best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            for (int i = 0; i < 10; i++)
                fc_matvec_q80_ctx(ctx, w, x, y, rows, cols, stride);
            fc_sync_ctx(ctx);
            double t0 = now_us();
            for (int i = 0; i < iters; i++)
                fc_matvec_q80_ctx(ctx, w, x, y, rows, cols, stride);
            fc_sync_ctx(ctx);
            double us = (now_us() - t0) / iters;
            if (us < best)
                best = us;
        }
        double gbs = (double)rows * stride / (best * 1e3);
        printf("    qkv 8192x2048 Q8_0: %.1f us, %.1f GB/s (min %.0f)\n", best, gbs,
               Q80_MIN_GBS);
        ASSERT_TRUE(gbs >= Q80_MIN_GBS);
    }
    fc_buffer_free(w);
    fc_buffer_free(x);
    fc_buffer_free(y);
}

int main(void)
{
    printf("Q8_0 G4 Layout Tests\n");
    printf("====================\n");
    RUN_TEST(g4_layout);
    if (fc_init() != 0) {
        printf("  [SKIP] no CUDA device\n");
        TEST_REPORT();
        return __ft_failed > 0 ? 1 : 0;
    }
    RUN_TEST(qkv_bandwidth);
    fc_free();
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
