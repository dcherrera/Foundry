/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_pl_q8.c
 * @brief  Tests for the Q8_0 frozen-base linear: block layout, round trip, forward, backward dx.
 * @status None
 * @issues None
 * @todo   None
 */

#include "lib/test.h"
#include "potluck/pl_q8.h"
#include "tt/convert/ft_float_convert.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Local xorshift32 -> uniform [-1, 1). */
static uint32_t rng_state;
static float urand(void)
{
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return (float)(x >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

static void fill(float *a, size_t n, uint32_t seed)
{
    rng_state = seed;
    for (size_t i = 0; i < n; i++) a[i] = urand();
}

static float block_scale(const pl_q8 *Q, int o, int g)
{
    const uint8_t *b = Q->blocks + ((size_t)o * Q->nb + g) * 34;
    return ft_float_from_fp16((uint16_t)(b[0] | (b[1] << 8)));
}

enum { T_ = 5, IN_ = 96, OUT_ = 70 };

/* q1: hand-worked block from the card. */
static void test_q1(void)
{
    float W[32];
    for (int k = 0; k < 32; k++) W[k] = (k - 16) * 0.05f;
    pl_q8 Q;
    ASSERT_TRUE(pl_q8_from_f32(W, 32, 1, &Q) == 0);
    ASSERT_TRUE(Q.in == 32 && Q.out == 1 && Q.nb == 1);
    const uint8_t *b = Q.blocks;
    ASSERT_TRUE((uint16_t)(b[0] | (b[1] << 8)) == 0x1e73);
    ASSERT_TRUE((int8_t)b[2 + 0] == -127);
    ASSERT_TRUE((int8_t)b[2 + 1] == -119);
    ASSERT_TRUE((int8_t)b[2 + 16] == 0);
    ASSERT_TRUE((int8_t)b[2 + 31] == 119);
    float v[32];
    pl_q8_dequant_rows(&Q, 0, 1, v);
    ASSERT_TRUE(fabsf(v[0] - (-0.79985428f)) <= 1e-7f);
    ASSERT_TRUE(fabsf(v[31] - 0.74946976f) <= 1e-7f);
    pl_q8_free(&Q);
}

/* q2: round-trip bound per element. q = round(x / d32) with the fp32 scale (ggml), stored scale
 * d_b = fp16(d32), so |W - q*d_b| <= d32/2 + |q|*|d32 - d_b| <= d_b/2 + 127.5*|d32 - d_b|. */
static void test_q2(void)
{
    float *W = malloc(sizeof(float) * IN_ * OUT_), *Wt = malloc(sizeof(float) * IN_ * OUT_);
    ASSERT_NOT_NULL(W); ASSERT_NOT_NULL(Wt);
    fill(W, (size_t)IN_ * OUT_, 0x1234567u);
    pl_q8 Q;
    ASSERT_TRUE(pl_q8_from_f32(W, IN_, OUT_, &Q) == 0);
    ASSERT_TRUE(Q.nb == IN_ / 32);
    pl_q8_dequant_rows(&Q, 0, OUT_, Wt);
    int bad = 0;
    for (int o = 0; o < OUT_; o++)
        for (int g = 0; g < Q.nb; g++) {
            float amax = 0.0f;
            for (int k = 0; k < 32; k++) amax = fmaxf(amax, fabsf(W[(g * 32 + k) * OUT_ + o]));
            double d32 = amax / 127.0f, db = block_scale(&Q, o, g);
            double tol = db / 2 + 127.5 * fabs(d32 - db) + 1e-6;
            for (int k = 0; k < 32; k++) {
                int i = g * 32 + k;
                if (fabs((double)W[i * OUT_ + o] - Wt[o * IN_ + i]) > tol) bad++;
            }
        }
    ASSERT_TRUE(bad == 0);
    pl_q8_free(&Q); free(W); free(Wt);
}

/* q3: forward y = x * W' against a double loop over the dequantised W'. */
static void test_q3(void)
{
    float *W = malloc(sizeof(float) * IN_ * OUT_), *Wt = malloc(sizeof(float) * IN_ * OUT_);
    float x[T_ * IN_], y[T_ * OUT_];
    ASSERT_NOT_NULL(W); ASSERT_NOT_NULL(Wt);
    fill(W, (size_t)IN_ * OUT_, 0xabcdefu);
    fill(x, T_ * IN_, 0x2468aceu);
    pl_q8 Q;
    ASSERT_TRUE(pl_q8_from_f32(W, IN_, OUT_, &Q) == 0);
    pl_q8_dequant_rows(&Q, 0, OUT_, Wt);
    for (int i = 0; i < T_ * OUT_; i++) y[i] = 1e30f;   /* must be overwritten */
    pl_q8_forward(&Q, x, y, T_);
    double maxd = 0, maxr = 0;
    for (int t = 0; t < T_; t++)
        for (int o = 0; o < OUT_; o++) {
            double r = 0;
            for (int i = 0; i < IN_; i++) r += (double)x[t * IN_ + i] * Wt[o * IN_ + i];
            maxd = fmax(maxd, fabs(y[t * OUT_ + o] - r));
            maxr = fmax(maxr, fabs(r));
        }
    ASSERT_TRUE(maxr > 0 && maxd / maxr <= 1e-5);
    pl_q8_free(&Q); free(W); free(Wt);
}

/* q4: backward dx += dy * W'^T with dx preset to 1.0. */
static void test_q4(void)
{
    float *W = malloc(sizeof(float) * IN_ * OUT_), *Wt = malloc(sizeof(float) * IN_ * OUT_);
    float dy[T_ * OUT_], dx[T_ * IN_];
    ASSERT_NOT_NULL(W); ASSERT_NOT_NULL(Wt);
    fill(W, (size_t)IN_ * OUT_, 0x13579bdu);
    fill(dy, T_ * OUT_, 0x0fedcbau);
    pl_q8 Q;
    ASSERT_TRUE(pl_q8_from_f32(W, IN_, OUT_, &Q) == 0);
    pl_q8_dequant_rows(&Q, 0, OUT_, Wt);
    for (int i = 0; i < T_ * IN_; i++) dx[i] = 1.0f;
    pl_q8_backward_dx(&Q, dy, dx, T_);
    double maxd = 0, maxr = 0;
    for (int t = 0; t < T_; t++)
        for (int i = 0; i < IN_; i++) {
            double r = 1.0;
            for (int o = 0; o < OUT_; o++) r += (double)dy[t * OUT_ + o] * Wt[o * IN_ + i];
            maxd = fmax(maxd, fabs(dx[t * IN_ + i] - r));
            maxr = fmax(maxr, fabs(r));
        }
    ASSERT_TRUE(maxr > 0 && maxd / maxr <= 1e-5);
    pl_q8_free(&Q); free(W); free(Wt);
}

/* q5: in not a multiple of 32 is rejected. */
static void test_q5(void)
{
    float W[33 * 2] = {0};
    pl_q8 Q;
    memset(&Q, 0, sizeof Q);
    ASSERT_TRUE(pl_q8_from_f32(W, 33, 2, &Q) == -1);
    ASSERT_TRUE(Q.blocks == NULL);
}

int main(void)
{
    printf("=== test_pl_q8 ===\n");
    RUN_TEST(q1);
    RUN_TEST(q2);
    RUN_TEST(q3);
    RUN_TEST(q4);
    RUN_TEST(q5);
    TEST_REPORT();
    return __ft_failed ? 1 : 0;
}
