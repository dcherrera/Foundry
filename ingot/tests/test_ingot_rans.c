/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_rans.c
 * @brief  rANS table quantization, serialization and the interleaved block coder.
 */
#include <stdlib.h>
#include <string.h>

#include "lib/test.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_rans.h"

static uint64_t rng = 88172645463325252ull;
static uint64_t next_u64(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}

static void test_table_sums_to_total_and_keeps_rare_symbols(void) {
    uint64_t counts[1024] = {0};
    for (int s = 0; s < 1024; s += 3)
        counts[s] = (s % 7 == 0) ? 1 : (uint64_t)s * s * 1000;
    ingot_rans_table t;
    ASSERT_INT_EQ(0, ingot_rans_table_build(&t, counts, 1024));
    uint64_t sum = 0;
    int rare_ok = 1;
    for (int s = 0; s < 1024; s++) {
        sum += t.freq[s];
        if (counts[s] && !t.freq[s])
            rare_ok = 0;
        if (!counts[s] && t.freq[s])
            rare_ok = 0;
    }
    ASSERT_TRUE(sum == INGOT_RANS_TOTAL);
    ASSERT_TRUE(rare_ok);
}

static void test_table_serialize_roundtrip_and_reject(void) {
    uint64_t counts[119] = {0};
    for (int s = 0; s < 119; s++)
        counts[s] = next_u64() % 5000;
    ingot_rans_table t, u;
    ASSERT_INT_EQ(0, ingot_rans_table_build(&t, counts, 119));
    uint8_t buf[2 + 4 * 119];
    size_t n = ingot_rans_table_write(&t, buf), used = 0;
    ASSERT_INT_EQ((int)ingot_rans_table_size(&t), (int)n);
    ASSERT_INT_EQ(0, ingot_rans_table_read(&u, 119, buf, n, &used));
    ASSERT_TRUE(used == n && !memcmp(t.freq, u.freq, sizeof(t.freq)));
    buf[4] ^= 1; /* frequency no longer sums to 2^16 */
    ASSERT_INT_EQ(-INGOT_EFORMAT, ingot_rans_table_read(&u, 119, buf, n, &used));
    buf[4] ^= 1;
    ASSERT_INT_EQ(-INGOT_EFORMAT, ingot_rans_table_read(&u, 100, buf, n, &used));
    ASSERT_INT_EQ(-INGOT_EFORMAT, ingot_rans_table_read(&u, 119, buf, n - 1, &used));
}

/* Encode n symbols drawn from two tables (and some empty positions), decode, compare. */
static int block_roundtrip(size_t n, int skewed) {
    uint64_t c0[1024] = {0}, c1[1024] = {0};
    for (int s = 0; s < 1024; s++)
        c0[s] = skewed ? (s == 5 ? 1000000 : (s < 40 ? 1 : 0)) : 1 + next_u64() % 100;
    c1[17] = 1; /* single-symbol table: freq = 2^16 */
    ingot_rans_table t[2];
    ingot_rans_table_build(&t[0], c0, 1024);
    ingot_rans_table_build(&t[1], c1, 1024);
    ingot_rans_symbol *syms = calloc(n, sizeof(*syms));
    uint16_t *words = calloc(n + 1, sizeof(*words));
    uint16_t *slots = malloc(2 * INGOT_RANS_TOTAL * sizeof(*slots));
    for (size_t i = 0; i < n; i++) {
        uint64_t r = next_u64();
        int k = (int)(r % 5);
        if (k == 4)
            continue; /* no symbol at this position */
        const ingot_rans_table *tt = &t[k == 3];
        uint32_t s = 0;
        do {
            s = (uint32_t)((r >> 8) % 1024);
            r = next_u64();
        } while (!tt->freq[s]);
        syms[i] = (ingot_rans_symbol){tt, s};
    }
    uint32_t states[INGOT_RANS_LANES];
    size_t wc = 0;
    int ok = ingot_rans_encode_block(syms, n, states, words, &wc) == 0;
    ingot_rans_slots_build(&t[0], slots);
    ingot_rans_slots_build(&t[1], slots + INGOT_RANS_TOTAL);
    size_t wp = 0;
    for (size_t i = 0; ok && i < n; i++) {
        if (!syms[i].table)
            continue;
        int k = syms[i].table == &t[1];
        int32_t s = ingot_rans_decode(&states[i % INGOT_RANS_LANES], syms[i].table,
                                      slots + (size_t)k * INGOT_RANS_TOTAL, words, wc, &wp);
        ok = s == (int32_t)syms[i].symbol;
    }
    for (int l = 0; ok && l < INGOT_RANS_LANES; l++)
        ok = states[l] == INGOT_RANS_LOW;
    ok = ok && wp == wc;
    free(syms);
    free(words);
    free(slots);
    return ok;
}

static void test_block_roundtrip_mixed_tables(void) {
    ASSERT_TRUE(block_roundtrip(100000, 0));
    ASSERT_TRUE(block_roundtrip(100000, 1));
    ASSERT_TRUE(block_roundtrip(31, 0));
    ASSERT_TRUE(block_roundtrip(1, 0));
}

static void test_encode_rejects_zero_frequency_symbol(void) {
    uint64_t c[4] = {5, 0, 5, 5};
    ingot_rans_table t;
    ingot_rans_table_build(&t, c, 4);
    ingot_rans_symbol s = {&t, 1};
    uint32_t states[INGOT_RANS_LANES];
    uint16_t words[2];
    size_t wc = 0;
    ASSERT_INT_EQ(-INGOT_EINVAL, ingot_rans_encode_block(&s, 1, states, words, &wc));
}

int main(void) {
    RUN_TEST(table_sums_to_total_and_keeps_rare_symbols);
    RUN_TEST(table_serialize_roundtrip_and_reject);
    RUN_TEST(block_roundtrip_mixed_tables);
    RUN_TEST(encode_rejects_zero_frequency_symbol);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
