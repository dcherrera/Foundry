/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_fcoder.h
 * @brief  Field coder: one code path that counts, encodes or decodes a stream of context-coded
 *         symbols and raw bits, so a quantized-type coder is written once and its encoder and
 *         decoder cannot drift apart.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * A type coder calls `v = ingot_fc_sym(c, ctx, v)` for each symbol and
 * `v = ingot_fc_raw(c, nbits, v)` for raw bits. Counting and encoding record v and return it;
 * decoding ignores the argument and returns the decoded value. Symbols use static 16-bit rANS
 * tables (one per context) and the 32-lane block layout of ingot_block.h: symbol n of a block
 * uses lane n % 32. Raw bits go LSB-first to the block's raw stream. Errors are sticky in c->err
 * (decoding returns 0 after the first one).
 *
 * Decoding looks a symbol up in a 64 K slot map (bytes for alphabets up to 256) for alphabets
 * above INGOT_FC_SEARCH_MAX, else by binary search over the table's starts. Slot maps cost 64 KB
 * per used context (a K-quant tensor: ~5 MB) and decode 1.6-2x faster than the search.
 */
#ifndef FOUNDRY_INGOT_FCODER_H
#define FOUNDRY_INGOT_FCODER_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/codec/ingot_block.h"
#include "ingot/codec/ingot_step.h"
#include "ingot/core/ingot_bits.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_rans.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { INGOT_FC_MAX_CONTEXTS = 1024, INGOT_FC_SEARCH_MAX = 4 };

/* The per-symbol functions must inline into the type coders: a call per weight halves decode
 * speed. */
#if defined(__GNUC__) || defined(__clang__)
#define INGOT_FC_INLINE static inline __attribute__((always_inline))
#else
#define INGOT_FC_INLINE static inline
#endif

/** Contexts and their alphabets; cell[] are histogram offsets. */
typedef struct {
    uint32_t contexts;
    uint32_t alphabet[INGOT_FC_MAX_CONTEXTS];
    uint32_t cell[INGOT_FC_MAX_CONTEXTS + 1];
} ingot_fc_model;

/**
 * Append `count` contexts of one alphabet; *first = the first new context id.
 * @return 0 or -INGOT_EINVAL (too many contexts, or alphabet outside 1..1024)
 *
 * Thread-safe: yes (caller-owned model)   Reentrant: yes   Signal-safe: yes
 */
int ingot_fc_model_add(ingot_fc_model *m, uint32_t count, uint32_t alphabet, uint32_t *first);

/** Per-context tables (and decode lookup structures). Opaque. */
typedef struct ingot_fc_tables ingot_fc_tables;

/**
 * Build tables from histograms (m->cell[m->contexts] cells).
 *
 * @param m           Model
 * @param counts      Histograms, laid out by m->cell
 * @param out         Out: tables, freed with ingot_fc_tables_free
 * @param coded_bits  Out (optional): exact coded size of the counts under the tables
 * @return            0, -INGOT_EINVAL or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_fc_tables_build(const ingot_fc_model *m, const uint64_t *counts, ingot_fc_tables **out,
                          double *coded_bits);

/**
 * Serialized size, and serialization (one ingot_rans table per context, in order).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_fc_tables_size(const ingot_fc_tables *t);
size_t ingot_fc_tables_write(const ingot_fc_tables *t, uint8_t *out);

/**
 * Parse serialized tables for model m and build the decode lookups.
 *
 * @param consumed  Out: bytes used
 * @return          0, -INGOT_EFORMAT or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_fc_tables_read(const ingot_fc_model *m, const uint8_t *src, uint64_t avail,
                         ingot_fc_tables **out, uint64_t *consumed);

/**
 * Free (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_fc_tables_free(ingot_fc_tables *t);

typedef enum { INGOT_FC_COUNT = 0, INGOT_FC_ENCODE = 1, INGOT_FC_DECODE = 2 } ingot_fc_mode;

/** Coder state for one block. Fields are internal; use the functions below. */
typedef struct {
    ingot_fc_mode mode;
    int err;
    size_t n;                          /* symbols so far */
    /* count */
    uint64_t *counts;
    const uint32_t *cell;
    uint64_t raw_bits;
    /* encode */
    const ingot_rans_table *tables;
    ingot_rans_symbol *syms;
    size_t sym_capacity;
    ingot_bit_writer raw_out;
    /* decode */
    const uint8_t *const *slots8;     /* per context: byte slot map (alphabet <= 256) */
    const uint16_t *const *slots16;   /* per context: slot map (larger alphabets) */
    uint32_t x[INGOT_RANS_LANES];
    const uint16_t *words;
    size_t word_count;
    size_t wp;
    ingot_bit_reader raw_in;
} ingot_fcoder;

/**
 * Start counting into counts (laid out by m->cell; accumulates).
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: yes
 */
void ingot_fc_begin_count(ingot_fcoder *c, const ingot_fc_model *m, uint64_t *counts);

/**
 * Start encoding one block with tables t.
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: yes
 */
void ingot_fc_begin_encode(ingot_fcoder *c, const ingot_fc_tables *t);

/**
 * Finish encoding: rANS-code the symbols and serialize the block into out (free(out->data)).
 * Releases the coder's buffers either way.
 *
 * @return 0, c->err, -INGOT_EINVAL (a symbol outside its table) or -INGOT_ENOMEM
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: no
 */
int ingot_fc_finish_encode(ingot_fcoder *c, ingot_block_buf *out);

/**
 * Start decoding block view v with tables t.
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: yes
 */
void ingot_fc_begin_decode(ingot_fcoder *c, const ingot_fc_tables *t, const ingot_block_view *v);

/**
 * Finish decoding: every lane back at its start state, every word and raw byte used.
 *
 * @return 0, c->err or -INGOT_ECORRUPT
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: yes
 */
int ingot_fc_finish_decode(const ingot_fcoder *c);

/**
 * Release buffers of an unfinished encode (no-op otherwise).
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: no
 */
void ingot_fc_release(ingot_fcoder *c);

/** Internal: grow the symbol buffer or the raw buffer; sets c->err on failure. */
void ingot_fc_grow_syms(ingot_fcoder *c);
void ingot_fc_grow_raw(ingot_fcoder *c);

/** 1 when the coder reads values from the caller (count, encode), 0 when decoding. */
static inline int ingot_fc_encoding(const ingot_fcoder *c) { return c->mode != INGOT_FC_DECODE; }

/* Largest symbol whose start is <= slot (the one whose interval holds it). */
INGOT_FC_INLINE uint32_t ingot_fc_search(const ingot_rans_table *t, uint32_t slot) {
    uint32_t lo = 0, hi = t->alphabet;
    while (hi - lo > 1) {
        uint32_t mid = (lo + hi) / 2;
        if (t->start[mid] <= slot)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

INGOT_FC_INLINE uint32_t ingot_fc_decode_sym(ingot_fcoder *c, uint32_t ctx) {
    const ingot_rans_table *t = &c->tables[ctx];
    uint32_t *x = &c->x[c->n % INGOT_RANS_LANES];
    uint32_t slot = *x & (INGOT_RANS_TOTAL - 1u);
    if (!t->used) {
        c->err = -INGOT_ECORRUPT;
        return 0;
    }
    uint32_t s = c->slots8[ctx]    ? c->slots8[ctx][slot]
                 : c->slots16[ctx] ? c->slots16[ctx][slot]
                                   : ingot_fc_search(t, slot);
    uint32_t v = t->freq[s] * (*x >> INGOT_RANS_PROB_BITS) + slot - t->start[s];
    if (v < INGOT_RANS_LOW) {
        if (c->wp >= c->word_count) {
            c->err = -INGOT_ECORRUPT;
            return 0;
        }
        v = (v << 16) | c->words[c->wp++];
    }
    *x = v;
    c->n++;
    return s;
}

/**
 * Code one symbol of context ctx (see the file comment).
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: yes
 */
INGOT_FC_INLINE uint32_t ingot_fc_sym(ingot_fcoder *c, uint32_t ctx, uint32_t value) {
    if (c->err)
        return 0;
    switch (c->mode) {
    case INGOT_FC_COUNT:
        c->counts[c->cell[ctx] + value]++;
        break;
    case INGOT_FC_ENCODE:
        if (c->n == c->sym_capacity)
            ingot_fc_grow_syms(c);
        if (c->err)
            return 0;
        c->syms[c->n] = (ingot_rans_symbol){&c->tables[ctx], value};
        break;
    case INGOT_FC_DECODE:
        return ingot_fc_decode_sym(c, ctx);
    }
    c->n++;
    return value;
}

/**
 * Code nbits (0..25) raw bits.
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: yes
 */
INGOT_FC_INLINE uint32_t ingot_fc_raw(ingot_fcoder *c, unsigned nbits, uint32_t value) {
    if (c->err || !nbits)
        return 0;
    switch (c->mode) {
    case INGOT_FC_COUNT:
        c->raw_bits += nbits;
        break;
    case INGOT_FC_ENCODE:
        if (ingot_bits_put(&c->raw_out, value, nbits)) {
            ingot_fc_grow_raw(c);
            if (c->err || ingot_bits_put(&c->raw_out, value, nbits))
                c->err = c->err ? c->err : -INGOT_ENOMEM;
        }
        break;
    case INGOT_FC_DECODE:
        if (ingot_bits_get(&c->raw_in, nbits, &value)) {
            c->err = -INGOT_ECORRUPT;
            return 0;
        }
        break;
    }
    return value;
}

/**
 * Code n symbols of one context: values[i] in (count, encode) or out (decode). The same stream as
 * n ingot_fc_sym calls; decoding keeps the coder state in locals (the per-symbol path re-reads
 * it from memory after every store), which is most of the decode time of a quantized tensor.
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: yes
 */
INGOT_FC_INLINE void ingot_fc_syms(ingot_fcoder *c, uint32_t ctx, uint32_t *restrict values,
                                   size_t n) {
    if (c->mode != INGOT_FC_DECODE || c->err) {
        for (size_t i = 0; i < n; i++)
            values[i] = ingot_fc_sym(c, ctx, values[i]);
        return;
    }
    const ingot_rans_table *t = &c->tables[ctx];
    const uint8_t *s8 = c->slots8[ctx];
    const uint16_t *s16 = c->slots16[ctx];
    const uint16_t *words = c->words;
    size_t wp = c->wp, wc = c->word_count, pos = c->n;
    int ok = t->used != 0;
    for (size_t i = 0; ok && i < n; i++, pos++) {
        uint32_t x = c->x[pos % INGOT_RANS_LANES], slot = x & (INGOT_RANS_TOTAL - 1u);
        uint32_t s = s8 ? s8[slot] : s16 ? s16[slot] : ingot_fc_search(t, slot);
        x = t->freq[s] * (x >> INGOT_RANS_PROB_BITS) + slot - t->start[s];
        if (x < INGOT_RANS_LOW) {
            ok = wp < wc;
            x = ok ? (x << 16) | words[wp++] : x;
        }
        c->x[pos % INGOT_RANS_LANES] = x;
        values[i] = s;
    }
    c->wp = wp;
    c->n = pos;
    if (!ok)
        c->err = -INGOT_ECORRUPT;
}

/**
 * Code a signed difference d (|d| < 2^16) as a step code in context ctx (alphabet
 * INGOT_STEP_ALPHABET); returns d.
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: yes
 */
INGOT_FC_INLINE int32_t ingot_fc_step(ingot_fcoder *c, uint32_t ctx, int32_t d) {
    ingot_step_code s = ingot_step_encode(ingot_fc_encoding(c) ? d : 0);
    uint32_t sym = ingot_fc_sym(c, ctx, s.symbol);
    uint32_t raw = ingot_fc_raw(c, ingot_step_raw_bits(sym), s.raw);
    return ingot_step_join(sym, raw);
}

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_FCODER_H */
