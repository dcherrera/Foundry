/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_fcoder.c
 * @brief  Field coder tables, block encode/decode framing (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_fcoder.h"

#include <stdlib.h>
#include <string.h>

struct ingot_fc_tables {
    uint32_t contexts;
    ingot_rans_table *tables;
    uint8_t **slots8;      /* per context: byte slot map (alphabet <= 256), or NULL */
    uint16_t **slots16;    /* per context: slot map (larger alphabets), or NULL (search) */
};

int ingot_fc_model_add(ingot_fc_model *m, uint32_t count, uint32_t alphabet, uint32_t *first) {
    if (!m || !alphabet || alphabet > INGOT_RANS_MAX_ALPHABET ||
        count > INGOT_FC_MAX_CONTEXTS - m->contexts)
        return -INGOT_EINVAL;
    if (first)
        *first = m->contexts;
    for (uint32_t i = 0; i < count; i++) {
        m->alphabet[m->contexts] = alphabet;
        m->cell[m->contexts + 1] = m->cell[m->contexts] + alphabet;
        m->contexts++;
    }
    return 0;
}

static int tables_alloc(uint32_t contexts, ingot_fc_tables **out) {
    ingot_fc_tables *t = calloc(1, sizeof(*t));
    if (!t)
        return -INGOT_ENOMEM;
    t->contexts = contexts;
    t->tables = calloc(contexts ? contexts : 1, sizeof(*t->tables));
    t->slots8 = calloc(contexts ? contexts : 1, sizeof(*t->slots8));
    t->slots16 = calloc(contexts ? contexts : 1, sizeof(*t->slots16));
    if (!t->tables || !t->slots8 || !t->slots16) {
        ingot_fc_tables_free(t);
        return -INGOT_ENOMEM;
    }
    *out = t;
    return 0;
}

int ingot_fc_tables_build(const ingot_fc_model *m, const uint64_t *counts, ingot_fc_tables **out,
                          double *coded_bits) {
    if (!m || !counts || !out)
        return -INGOT_EINVAL;
    ingot_fc_tables *t = NULL;
    int rc = tables_alloc(m->contexts, &t);
    double bits = 0.0;
    for (uint32_t cx = 0; !rc && cx < m->contexts; cx++) {
        rc = ingot_rans_table_build(&t->tables[cx], counts + m->cell[cx], m->alphabet[cx]);
        if (!rc)
            bits += ingot_rans_table_coded_bits(&t->tables[cx], counts + m->cell[cx]);
    }
    if (rc) {
        ingot_fc_tables_free(t);
        return rc;
    }
    if (coded_bits)
        *coded_bits = bits;
    *out = t;
    return 0;
}

size_t ingot_fc_tables_size(const ingot_fc_tables *t) {
    size_t bytes = 0;
    for (uint32_t cx = 0; cx < t->contexts; cx++)
        bytes += ingot_rans_table_size(&t->tables[cx]);
    return bytes;
}

size_t ingot_fc_tables_write(const ingot_fc_tables *t, uint8_t *out) {
    size_t at = 0;
    for (uint32_t cx = 0; cx < t->contexts; cx++)
        at += ingot_rans_table_write(&t->tables[cx], out + at);
    return at;
}

/* Slot map of a non-empty table: bytes for alphabets up to 256 (half the cache footprint). */
static int build_slots(const ingot_rans_table *t, uint8_t **slots8, uint16_t **slots16) {
    uint16_t *wide = malloc(INGOT_RANS_TOTAL * sizeof(*wide));
    if (!wide)
        return -INGOT_ENOMEM;
    ingot_rans_slots_build(t, wide);
    if (t->alphabet > 256) {
        *slots16 = wide;
        return 0;
    }
    *slots8 = malloc(INGOT_RANS_TOTAL);
    for (uint32_t i = 0; *slots8 && i < INGOT_RANS_TOTAL; i++)
        (*slots8)[i] = (uint8_t)wide[i];
    free(wide);
    return *slots8 ? 0 : -INGOT_ENOMEM;
}

int ingot_fc_tables_read(const ingot_fc_model *m, const uint8_t *src, uint64_t avail,
                         ingot_fc_tables **out, uint64_t *consumed) {
    if (!m || !src || !out || !consumed)
        return -INGOT_EFORMAT;
    ingot_fc_tables *t = NULL;
    int rc = tables_alloc(m->contexts, &t);
    uint64_t at = 0;
    for (uint32_t cx = 0; !rc && cx < m->contexts; cx++) {
        size_t used = 0;
        rc = ingot_rans_table_read(&t->tables[cx], m->alphabet[cx], src + at, avail - at, &used);
        at += used;
        if (rc || !t->tables[cx].used || m->alphabet[cx] <= INGOT_FC_SEARCH_MAX)
            continue;
        rc = build_slots(&t->tables[cx], &t->slots8[cx], &t->slots16[cx]);
    }
    if (rc) {
        ingot_fc_tables_free(t);
        return rc;
    }
    *consumed = at;
    *out = t;
    return 0;
}

void ingot_fc_tables_free(ingot_fc_tables *t) {
    if (!t)
        return;
    for (uint32_t cx = 0; t->slots8 && t->slots16 && cx < t->contexts; cx++) {
        free(t->slots8[cx]);
        free(t->slots16[cx]);
    }
    free(t->slots8);
    free(t->slots16);
    free(t->tables);
    free(t);
}

void ingot_fc_begin_count(ingot_fcoder *c, const ingot_fc_model *m, uint64_t *counts) {
    *c = (ingot_fcoder){.mode = INGOT_FC_COUNT, .counts = counts, .cell = m->cell};
}

void ingot_fc_begin_encode(ingot_fcoder *c, const ingot_fc_tables *t) {
    *c = (ingot_fcoder){.mode = INGOT_FC_ENCODE, .tables = t->tables};
}

void ingot_fc_grow_syms(ingot_fcoder *c) {
    size_t cap = c->sym_capacity ? c->sym_capacity * 2 : 1 << 16;
    ingot_rans_symbol *s = realloc(c->syms, cap * sizeof(*s));
    if (!s) {
        c->err = -INGOT_ENOMEM;
        return;
    }
    c->syms = s;
    c->sym_capacity = cap;
}

void ingot_fc_grow_raw(ingot_fcoder *c) {
    size_t cap = c->raw_out.capacity ? c->raw_out.capacity * 2 : 1 << 14;
    uint8_t *b = realloc(c->raw_out.buf, cap);
    if (!b) {
        c->err = -INGOT_ENOMEM;
        return;
    }
    memset(b + c->raw_out.capacity, 0, cap - c->raw_out.capacity);
    c->raw_out.buf = b;
    c->raw_out.capacity = cap;
}

void ingot_fc_release(ingot_fcoder *c) {
    if (c->mode != INGOT_FC_ENCODE)
        return;
    free(c->syms);
    free(c->raw_out.buf);
    c->syms = NULL;
    c->raw_out.buf = NULL;
    c->sym_capacity = 0;
    c->raw_out.capacity = 0;
}

int ingot_fc_finish_encode(ingot_fcoder *c, ingot_block_buf *out) {
    int rc = c->err;
    uint16_t *words = NULL;
    if (!rc && c->n > UINT32_MAX)
        rc = -INGOT_EINVAL;
    if (!rc && !(words = malloc((c->n ? c->n : 1) * sizeof(*words))))
        rc = -INGOT_ENOMEM;
    uint32_t states[INGOT_RANS_LANES];
    size_t word_count = 0;
    if (!rc)
        rc = ingot_rans_encode_block(c->syms, c->n, states, words, &word_count);
    if (!rc)
        rc = ingot_block_serialize(states, c->raw_out.buf, ingot_bits_bytes(&c->raw_out), words,
                                   word_count, out);
    free(words);
    ingot_fc_release(c);
    return rc;
}

void ingot_fc_begin_decode(ingot_fcoder *c, const ingot_fc_tables *t, const ingot_block_view *v) {
    *c = (ingot_fcoder){.mode = INGOT_FC_DECODE,
                        .tables = t->tables,
                        .slots8 = (const uint8_t *const *)t->slots8,
                        .slots16 = (const uint16_t *const *)t->slots16,
                        .words = v->words,
                        .word_count = v->word_count,
                        .raw_in = {v->raw, v->raw_bytes, 0}};
    memcpy(c->x, v->states, sizeof(c->x));
}

int ingot_fc_finish_decode(const ingot_fcoder *c) {
    if (c->err)
        return c->err;
    if ((c->raw_in.bitpos + 7) / 8 != c->raw_in.size || c->wp != c->word_count)
        return -INGOT_ECORRUPT;
    for (int l = 0; l < INGOT_RANS_LANES; l++)
        if (c->x[l] != INGOT_RANS_LOW)
            return -INGOT_ECORRUPT;
    return 0;
}
