/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_codec.c
 * @brief  Plain and delta payload round trips on synthetic BF16 tensors, plus red tests: a
 *         corrupted block or table entry must fail the decode or change the output.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lib/test.h"
#include "ingot/codec/ingot_delta.h"
#include "ingot/codec/ingot_plain.h"
#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_error.h"

/* xorshift64* so the tests are deterministic. */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_next(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 2685821657736338717ull;
}

static double rng_gauss(void) {
    double u = ((double)(rng_next() >> 11) + 0.5) / 9007199254740992.0;
    double v = ((double)(rng_next() >> 11) + 0.5) / 9007199254740992.0;
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

static uint16_t *gauss_bf16(size_t n, double sigma) {
    uint16_t *w = malloc(n * sizeof(*w));
    for (size_t i = 0; w && i < n; i++)
        w[i] = ingot_f32_to_bf16((float)(sigma * rng_gauss()));
    return w;
}

/* Encode to memory with the plain coder. */
static ingot_sink plain_encode_mem(const uint16_t *w, size_t n, uint32_t groups, int threads) {
    ingot_sink s = {.fd = -1};
    ingot_plain_plan *plan = NULL;
    if (ingot_plain_plan_create(w, n, groups, threads, &plan) == 0)
        ingot_plain_encode(plan, threads, &s);
    ingot_plain_plan_free(plan);
    return s;
}

static int plain_roundtrip_ok(const uint16_t *w, size_t n, uint32_t groups, int threads) {
    ingot_sink s = plain_encode_mem(w, n, groups, threads);
    uint16_t *out = calloc(n, sizeof(*out));
    int rc = ingot_plain_decode(s.mem, s.pos, n, threads, out);
    int ok = rc == 0 && memcmp(out, w, n * sizeof(*w)) == 0;
    free(out);
    free(s.mem);
    return ok;
}

static void test_plain_roundtrip_gauss_multiblock(void) {
    size_t n = (1u << 18) * 3 + 12345; /* three full blocks and a short one */
    uint16_t *w = gauss_bf16(n, 0.02);
    ASSERT_TRUE(plain_roundtrip_ok(w, n, 1, 4));
    free(w);
}

static void test_plain_roundtrip_every_bit_pattern(void) {
    size_t n = 65536 * 4;
    uint16_t *w = malloc(n * sizeof(*w));
    for (size_t i = 0; i < n; i++)
        w[i] = (uint16_t)(i * 40503u); /* every pattern, shuffled, 4 times */
    ASSERT_TRUE(plain_roundtrip_ok(w, n, 1, 2));
    free(w);
}

static void test_plain_roundtrip_groups_and_tiny(void) {
    size_t n = 8 * 70001;
    uint16_t *w = gauss_bf16(n, 0.05);
    ASSERT_TRUE(plain_roundtrip_ok(w, n, 8, 3));
    ASSERT_TRUE(plain_roundtrip_ok(w, 1, 1, 1));
    ASSERT_TRUE(plain_roundtrip_ok(w, 31, 1, 1));
    uint16_t same[1000];
    for (int i = 0; i < 1000; i++)
        same[i] = 0x3C00; /* one symbol: freq = 2^16 */
    ASSERT_TRUE(plain_roundtrip_ok(same, 1000, 1, 1));
    free(w);
}

static void test_plain_size_near_entropy(void) {
    size_t n = 1u << 20;
    uint16_t *w = gauss_bf16(n, 0.02);
    ingot_plain_plan *plan = NULL;
    ASSERT_INT_EQ(0, ingot_plain_plan_create(w, n, 1, 2, &plan));
    ingot_sink s = {.fd = -1};
    ASSERT_INT_EQ(0, ingot_plain_encode(plan, 2, &s));
    double bpw = 8.0 * (double)s.pos / (double)n;
    double est = 8.0 * (double)ingot_plain_plan_bytes(plan) / (double)n;
    /* Gaussian BF16 order-0 floor is 10.4638 (Phase 0, exact enumeration). */
    ASSERT_DBL_LT(bpw, 10.52);
    ASSERT_DBL_GT(bpw, 10.40);
    ASSERT_DBL_LT(fabs(bpw - est), 0.005);
    ingot_plain_plan_free(plan);
    free(s.mem);
    free(w);
}

/* Red: flipping one byte inside a block's rANS words must be caught. */
static void test_plain_corrupt_block_fails(void) {
    size_t n = (1u << 18) + 999;
    uint16_t *w = gauss_bf16(n, 0.02);
    ingot_sink s = plain_encode_mem(w, n, 1, 1);
    uint16_t *out = calloc(n, sizeof(*out));
    s.mem[s.pos - 64] ^= 0x10; /* inside the last block's word stream */
    int rc = ingot_plain_decode(s.mem, s.pos, n, 1, out);
    ASSERT_TRUE(rc != 0 || memcmp(out, w, n * sizeof(*w)) != 0);
    ASSERT_INT_EQ(-INGOT_ECORRUPT, rc);
    free(out);
    free(s.mem);
    free(w);
}

/* Move one unit of frequency from the largest entry of a serialized table to its first entry
 * (layout: u16 used, then used x (u16 symbol, u16 freq - 1)). */
static void corrupt_table_entry(uint8_t *t) {
    uint32_t used = t[0] | (t[1] << 8), big = 0;
    for (uint32_t e = 1; e < used; e++)
        if ((t[4 + 4 * e] | (t[5 + 4 * e] << 8)) > (t[4 + 4 * big] | (t[5 + 4 * big] << 8)))
            big = e;
    uint32_t fb = (uint32_t)(t[4 + 4 * big] | (t[5 + 4 * big] << 8)) - 1u;
    uint32_t f0 = (uint32_t)(t[4] | (t[5] << 8)) + 1u;
    t[4 + 4 * big] = (uint8_t)fb;
    t[5 + 4 * big] = (uint8_t)(fb >> 8);
    t[4] = (uint8_t)f0;
    t[5] = (uint8_t)(f0 >> 8);
}

/* Red: moving one unit of frequency between two table entries keeps the table valid but must
 * break the decode. */
static void test_plain_corrupt_table_fails(void) {
    size_t n = 200000;
    uint16_t *w = gauss_bf16(n, 0.02);
    ingot_sink s = plain_encode_mem(w, n, 1, 1);
    corrupt_table_entry(s.mem + 24); /* the only table */
    uint16_t *out = calloc(n, sizeof(*out));
    int rc = ingot_plain_decode(s.mem, s.pos, n, 1, out);
    ASSERT_TRUE(rc != 0 || memcmp(out, w, n * sizeof(*w)) != 0);
    free(out);
    free(s.mem);
    free(w);
}

/* Fine-tune-like target: base + relative noise `rel` on most rows; every 4th row identical. */
static uint16_t *finetune_of(const uint16_t *base, size_t rows, size_t cols, double rel) {
    uint16_t *t = malloc(rows * cols * sizeof(*t));
    for (size_t r = 0; t && r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            size_t i = r * cols + c;
            float b = ingot_bf16_to_f32(base[i]);
            t[i] = (r % 4 == 0) ? base[i] : ingot_f32_to_bf16(b + (float)(rel * 0.02 * rng_gauss()));
        }
    }
    return t;
}

static ingot_sink delta_encode_mem(const uint16_t *t, const uint16_t *b, size_t rows, size_t cols,
                                   int threads, uint64_t *estimate) {
    ingot_sink s = {.fd = -1};
    ingot_delta_plan *plan = NULL;
    if (ingot_delta_plan_create(t, b, rows, cols, threads, &plan) == 0) {
        ingot_delta_encode(plan, threads, &s);
        if (estimate)
            *estimate = ingot_delta_plan_bytes(plan);
    }
    ingot_delta_plan_free(plan);
    return s;
}

static int delta_roundtrip_ok(const uint16_t *t, const uint16_t *b, size_t rows, size_t cols,
                              int threads, double *bpw) {
    uint64_t est = 0;
    ingot_sink s = delta_encode_mem(t, b, rows, cols, threads, &est);
    size_t n = rows * cols;
    uint16_t *out = calloc(n, sizeof(*out));
    int rc = ingot_delta_decode(s.mem, s.pos, b, n, threads, out);
    int ok = rc == 0 && memcmp(out, t, n * sizeof(*t)) == 0 &&
             fabs((double)est - (double)s.pos) < 64.0 + 0.001 * (double)s.pos;
    if (bpw)
        *bpw = 8.0 * (double)s.pos / (double)n;
    free(out);
    free(s.mem);
    return ok;
}

static void test_delta_roundtrip_finetune(void) {
    size_t rows = 600, cols = 1500; /* 900k weights: 4 blocks, rows straddle blocks */
    uint16_t *b = gauss_bf16(rows * cols, 0.02);
    uint16_t *t = finetune_of(b, rows, cols, 0.05);
    double bpw = 0.0;
    ASSERT_TRUE(delta_roundtrip_ok(t, b, rows, cols, 4, &bpw));
    /* rel 0.05 -> log2(1/0.05) ~ 4.3 bits saved on 3/4 of the rows vs ~10.5 alone */
    ASSERT_DBL_LT(bpw, 6.0);
    free(t);
    free(b);
}

static void test_delta_roundtrip_identical_and_unrelated(void) {
    size_t rows = 300, cols = 1000;
    uint16_t *b = gauss_bf16(rows * cols, 0.02);
    double bpw = 1.0;
    ASSERT_TRUE(delta_roundtrip_ok(b, b, rows, cols, 2, &bpw));
    ASSERT_DBL_LT(bpw, 0.02); /* row flags + framing only */
    uint16_t *u = gauss_bf16(rows * cols, 0.02);
    ASSERT_TRUE(delta_roundtrip_ok(u, b, rows, cols, 2, &bpw));
    ASSERT_DBL_LT(bpw, 12.5); /* unrelated: worse than plain, still exact */
    free(u);
    free(b);
}

static void test_delta_roundtrip_extreme_patterns(void) {
    size_t rows = 256, cols = 256;
    uint16_t *b = malloc(rows * cols * sizeof(*b)), *t = malloc(rows * cols * sizeof(*t));
    for (size_t i = 0; i < rows * cols; i++) {
        b[i] = (uint16_t)(i * 40503u);         /* every pattern incl. inf/nan/subnormal */
        t[i] = (uint16_t)(i * 40503u + 7919u); /* far-apart ordered values, both signs */
    }
    ASSERT_TRUE(delta_roundtrip_ok(t, b, rows, cols, 1, NULL));
    free(t);
    free(b);
}

static void test_delta_corrupt_block_and_table_fail(void) {
    size_t rows = 512, cols = 1024;
    uint16_t *b = gauss_bf16(rows * cols, 0.02);
    uint16_t *t = finetune_of(b, rows, cols, 0.1);
    uint16_t *out = calloc(rows * cols, sizeof(*out));
    ingot_sink s = delta_encode_mem(t, b, rows, cols, 1, NULL);
    s.mem[s.pos - 64] ^= 0x01;
    int rc = ingot_delta_decode(s.mem, s.pos, b, rows * cols, 1, out);
    ASSERT_TRUE(rc != 0 || memcmp(out, t, rows * cols * sizeof(*t)) != 0);
    s.mem[s.pos - 64] ^= 0x01;
    ASSERT_INT_EQ(0, ingot_delta_decode(s.mem, s.pos, b, rows * cols, 1, out));
    /* First non-empty table after the row scales (32-byte header + 512 rows, 8-aligned). */
    uint8_t *tab = s.mem + 32 + 512;
    while (tab[0] == 0 && tab[1] == 0)
        tab += 2;
    corrupt_table_entry(tab);
    rc = ingot_delta_decode(s.mem, s.pos, b, rows * cols, 1, out);
    ASSERT_TRUE(rc != 0 || memcmp(out, t, rows * cols * sizeof(*t)) != 0);
    free(s.mem);
    free(out);
    free(t);
    free(b);
}

int main(void) {
    RUN_TEST(delta_roundtrip_finetune);
    RUN_TEST(delta_roundtrip_identical_and_unrelated);
    RUN_TEST(delta_roundtrip_extreme_patterns);
    RUN_TEST(delta_corrupt_block_and_table_fail);
    RUN_TEST(plain_roundtrip_gauss_multiblock);
    RUN_TEST(plain_roundtrip_every_bit_pattern);
    RUN_TEST(plain_roundtrip_groups_and_tiny);
    RUN_TEST(plain_size_near_entropy);
    RUN_TEST(plain_corrupt_block_fails);
    RUN_TEST(plain_corrupt_table_fails);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
