/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_rans.c
 * @brief  rANS tables and the interleaved block encoder (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_rans.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ingot_error.h"

static void table_fill_starts(ingot_rans_table *t) {
    uint32_t acc = 0;
    for (uint32_t s = 0; s < t->alphabet; s++) {
        t->start[s] = acc;
        acc += t->freq[s];
    }
}

typedef struct {
    double key;
    uint32_t symbol;
} sort_entry;

/* Descending key, ties by ascending symbol, so the table is deterministic. */
static int sort_entry_desc(const void *a, const void *b) {
    const sort_entry *x = a, *y = b;
    if (x->key != y->key)
        return x->key < y->key ? 1 : -1;
    return x->symbol < y->symbol ? -1 : (x->symbol > y->symbol);
}

/* Move the assigned total to exactly INGOT_RANS_TOTAL. Short: +1 to the largest fractional parts
 * (the deficit is below the used count). Over (rare symbols bumped to 1): -1 round-robin over the
 * largest frequencies above 1. */
static void table_settle(ingot_rans_table *t, const uint64_t *counts, const double *frac,
                         uint32_t assigned) {
    sort_entry order[INGOT_RANS_MAX_ALPHABET];
    uint32_t n = 0;
    int short_total = assigned < INGOT_RANS_TOTAL;
    for (uint32_t s = 0; s < t->alphabet; s++)
        if (counts[s])
            order[n++] = (sort_entry){short_total ? frac[s] : (double)t->freq[s], s};
    qsort(order, n, sizeof(order[0]), sort_entry_desc);
    if (short_total) {
        for (uint32_t i = 0; assigned < INGOT_RANS_TOTAL; i = (i + 1) % n, assigned++)
            t->freq[order[i].symbol]++;
        return;
    }
    while (assigned > INGOT_RANS_TOTAL) {
        for (uint32_t i = 0; i < n && assigned > INGOT_RANS_TOTAL; i++) {
            if (t->freq[order[i].symbol] > 1) {
                t->freq[order[i].symbol]--;
                assigned--;
            }
        }
    }
}

int ingot_rans_table_build(ingot_rans_table *t, const uint64_t *counts, uint32_t alphabet) {
    if (!t || !counts || !alphabet || alphabet > INGOT_RANS_MAX_ALPHABET)
        return -INGOT_EINVAL;
    memset(t, 0, sizeof(*t));
    t->alphabet = alphabet;
    uint64_t total = 0;
    for (uint32_t s = 0; s < alphabet; s++) {
        total += counts[s];
        t->used += counts[s] > 0;
    }
    if (!total)
        return 0;
    if (t->used > INGOT_RANS_TOTAL)
        return -INGOT_EINVAL;
    double frac[INGOT_RANS_MAX_ALPHABET] = {0};
    uint32_t assigned = 0;
    for (uint32_t s = 0; s < alphabet; s++) {
        if (!counts[s])
            continue;
        double ideal = (double)counts[s] * (double)INGOT_RANS_TOTAL / (double)total;
        t->freq[s] = ideal < 1.0 ? 1u : (uint32_t)ideal;
        frac[s] = ideal - (double)t->freq[s];
        assigned += t->freq[s];
    }
    table_settle(t, counts, frac, assigned);
    table_fill_starts(t);
    return 0;
}

double ingot_rans_table_coded_bits(const ingot_rans_table *t, const uint64_t *counts) {
    double bits = 0.0;
    for (uint32_t s = 0; s < t->alphabet; s++)
        if (counts[s])
            bits += (double)counts[s] * ((double)INGOT_RANS_PROB_BITS - log2((double)t->freq[s]));
    return bits;
}

size_t ingot_rans_table_size(const ingot_rans_table *t) { return 2 + 4 * (size_t)t->used; }

static void put_u16(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

static uint32_t get_u16(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }

size_t ingot_rans_table_write(const ingot_rans_table *t, uint8_t *dst) {
    put_u16(dst, t->used);
    size_t at = 2;
    for (uint32_t s = 0; s < t->alphabet; s++) {
        if (!t->freq[s])
            continue;
        put_u16(dst + at, s);
        put_u16(dst + at + 2, t->freq[s] - 1u);
        at += 4;
    }
    return at;
}

int ingot_rans_table_read(ingot_rans_table *t, uint32_t alphabet, const uint8_t *src,
                          size_t avail, size_t *consumed) {
    if (!t || !src || !consumed || !alphabet || alphabet > INGOT_RANS_MAX_ALPHABET || avail < 2)
        return -INGOT_EFORMAT;
    memset(t, 0, sizeof(*t));
    t->alphabet = alphabet;
    uint32_t used = get_u16(src);
    if (avail < 2 + 4 * (size_t)used)
        return -INGOT_EFORMAT;
    uint64_t sum = 0;
    int64_t prev = -1;
    for (uint32_t e = 0; e < used; e++) {
        uint32_t s = get_u16(src + 2 + 4 * (size_t)e);
        uint32_t f = get_u16(src + 4 + 4 * (size_t)e) + 1u;
        if ((int64_t)s <= prev || s >= alphabet)
            return -INGOT_EFORMAT;
        t->freq[s] = f;
        sum += f;
        prev = s;
    }
    if (used && sum != INGOT_RANS_TOTAL)
        return -INGOT_EFORMAT;
    t->used = used;
    table_fill_starts(t);
    *consumed = 2 + 4 * (size_t)used;
    return 0;
}

void ingot_rans_slots_build(const ingot_rans_table *t, uint16_t *slots) {
    for (uint32_t s = 0; s < t->alphabet; s++)
        for (uint32_t k = 0; k < t->freq[s]; k++)
            slots[t->start[s] + k] = (uint16_t)s;
}

int ingot_rans_encode_block(const ingot_rans_symbol *syms, size_t n, uint32_t *states,
                            uint16_t *words, size_t *word_count) {
    uint32_t x[INGOT_RANS_LANES];
    for (int l = 0; l < INGOT_RANS_LANES; l++)
        x[l] = INGOT_RANS_LOW;
    size_t emitted = 0;
    for (size_t i = n; i-- > 0;) {
        const ingot_rans_table *t = syms[i].table;
        if (!t)
            continue;
        uint32_t s = syms[i].symbol;
        uint32_t freq = s < t->alphabet ? t->freq[s] : 0;
        if (!freq)
            return -INGOT_EINVAL;
        uint32_t *lane = &x[i % INGOT_RANS_LANES];
        if ((uint64_t)*lane >= ((uint64_t)freq << 16)) {
            words[emitted++] = (uint16_t)(*lane & 0xFFFFu);
            *lane >>= 16;
        }
        *lane = ((*lane / freq) << INGOT_RANS_PROB_BITS) + (*lane % freq) + t->start[s];
    }
    for (size_t a = 0, b = emitted; a + 1 < b; a++, b--) {
        uint16_t tmp = words[a];
        words[a] = words[b - 1];
        words[b - 1] = tmp;
    }
    memcpy(states, x, sizeof(x));
    *word_count = emitted;
    return 0;
}
