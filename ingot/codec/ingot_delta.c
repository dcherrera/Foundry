/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_delta.c
 * @brief  Delta-mode BF16 tensor coder (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_delta.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/codec/ingot_step.h"
#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_bits.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_parallel.h"

enum {
    CTX_LO = -16,
    CTX_HI = 12,
    STEP_CONTEXTS = CTX_HI - CTX_LO + 1,
    CTX_FALLBACK = STEP_CONTEXTS,
    CONTEXTS = STEP_CONTEXTS + 1,
    CTX_NONE = 255,
    STEP_ALPHABET = 119,
    FALLBACK_ALPHABET = 1024,
    HIST_CELLS = STEP_CONTEXTS * STEP_ALPHABET + FALLBACK_ALPHABET,
    ROW_IDENTICAL = -128,
    HEADER_BYTES = 32,
    BLOCK = 1 << INGOT_BLOCK_LOG2,
    MAX_RAW_BITS = 13,
};

static uint32_t context_alphabet(unsigned ctx) {
    return ctx == CTX_FALLBACK ? FALLBACK_ALPHABET : STEP_ALPHABET;
}

static size_t hist_offset(unsigned ctx) { return (size_t)ctx * STEP_ALPHABET; }

/* Context of a weight in a non-identical row (shared by encoder and decoder). */
static inline unsigned delta_context(int row_scale, uint16_t base) {
    int e = (int)ingot_bf16_exp(base);
    int k = row_scale - 2 * ((e ? e : 1) - 134);
    if (k > CTX_HI)
        return CTX_FALLBACK;
    return (unsigned)((k < CTX_LO ? CTX_LO : k) - CTX_LO);
}

typedef ingot_step_code delta_code;

/* Inverse of the step code: bits of |d| below the symbol come from the raw stream. */
static inline int step_decode(uint32_t sym, ingot_bit_reader *raw, int32_t *d) {
    uint32_t low = 0;
    if (ingot_bits_get(raw, ingot_step_raw_bits(sym), &low))
        return -1;
    *d = ingot_step_join(sym, low);
    return 0;
}

struct ingot_delta_plan {
    const uint16_t *target;
    const uint16_t *base;
    uint64_t rows;
    uint64_t cols;
    size_t block_count;
    int8_t *row_scale;
    ingot_rans_table *tables;
    double coded_bits;
    uint64_t raw_bits;
};

static int row_scale_item(void *ctx, size_t r) {
    ingot_delta_plan *p = ctx;
    const uint16_t *t = p->target + r * p->cols, *b = p->base + r * p->cols;
    double abs_err = 0.0;
    int identical = 1;
    for (uint64_t j = 0; j < p->cols; j++) {
        identical &= t[j] == b[j];
        abs_err += fabs((double)ingot_bf16_to_f32(t[j]) - (double)ingot_bf16_to_f32(b[j]));
    }
    double s = identical ? 0.0 : 2.0 * log2(abs_err / (double)p->cols);
    if (!identical && !(s < 127.0))
        s = 127.0; /* inf/nan weights or huge errors: any scale decodes, this one is valid */
    if (s < -127.0)
        s = -127.0;
    p->row_scale[r] = identical ? (int8_t)ROW_IDENTICAL : (int8_t)lround(s);
    return 0;
}

/* Code of the weight at flat index i; returns its context or CTX_NONE (identical row). */
static inline unsigned weight_code(const ingot_delta_plan *p, uint64_t i, int row_scale,
                                   delta_code *c) {
    if (row_scale == ROW_IDENTICAL)
        return CTX_NONE;
    uint16_t t = p->target[i], b = p->base[i];
    unsigned ctx = delta_context(row_scale, b);
    if (ctx == CTX_FALLBACK)
        *c = (delta_code){ingot_bf16_symbol(t), ingot_bf16_raw6(t), 6};
    else
        *c = ingot_step_encode(ingot_bf16_ordered(t) - ingot_bf16_ordered(b));
    return ctx;
}

typedef struct {
    const ingot_delta_plan *plan;
    uint32_t *block_counts;
    uint64_t *block_raw_bits;
} histogram_job;

static int histogram_block(void *ctx, size_t blk) {
    histogram_job *job = ctx;
    const ingot_delta_plan *p = job->plan;
    uint64_t lo = (uint64_t)blk * BLOCK, total = p->rows * p->cols;
    uint64_t hi = lo + BLOCK < total ? lo + BLOCK : total;
    uint32_t *counts = job->block_counts + blk * HIST_CELLS;
    uint64_t raw_bits = 0;
    for (uint64_t i = lo; i < hi; i++) {
        delta_code c = {0, 0, 0};
        unsigned cx = weight_code(p, i, p->row_scale[i / p->cols], &c);
        if (cx == CTX_NONE)
            continue;
        counts[hist_offset(cx) + c.symbol]++;
        raw_bits += c.raw_bits;
    }
    job->block_raw_bits[blk] = raw_bits;
    return 0;
}

static int plan_tables(ingot_delta_plan *p, const uint32_t *block_counts,
                       const uint64_t *block_raw_bits) {
    uint64_t *counts = calloc(HIST_CELLS, sizeof(*counts));
    if (!counts)
        return -INGOT_ENOMEM;
    for (size_t b = 0; b < p->block_count; b++) {
        for (size_t k = 0; k < HIST_CELLS; k++)
            counts[k] += block_counts[b * HIST_CELLS + k];
        p->raw_bits += block_raw_bits[b];
    }
    int rc = 0;
    for (unsigned cx = 0; !rc && cx < CONTEXTS; cx++) {
        rc = ingot_rans_table_build(&p->tables[cx], counts + hist_offset(cx),
                                    context_alphabet(cx));
        if (!rc)
            p->coded_bits += ingot_rans_table_coded_bits(&p->tables[cx], counts + hist_offset(cx));
    }
    free(counts);
    return rc;
}

static int plan_histograms(ingot_delta_plan *p, int threads) {
    uint32_t *block_counts = calloc(p->block_count * HIST_CELLS, sizeof(*block_counts));
    uint64_t *block_raw_bits = calloc(p->block_count, sizeof(*block_raw_bits));
    int rc = (block_counts && block_raw_bits) ? 0 : -INGOT_ENOMEM;
    if (!rc) {
        histogram_job job = {p, block_counts, block_raw_bits};
        rc = ingot_parallel_for(p->block_count, threads, histogram_block, &job);
    }
    if (!rc)
        rc = plan_tables(p, block_counts, block_raw_bits);
    free(block_counts);
    free(block_raw_bits);
    return rc;
}

int ingot_delta_plan_create(const uint16_t *target, const uint16_t *base, uint64_t rows,
                            uint64_t cols, int threads, ingot_delta_plan **out) {
    if (!target || !base || !out || !rows || !cols)
        return -INGOT_EINVAL;
    ingot_delta_plan *p = calloc(1, sizeof(*p));
    if (!p)
        return -INGOT_ENOMEM;
    *p = (ingot_delta_plan){.target = target, .base = base, .rows = rows, .cols = cols};
    p->block_count = (size_t)((rows * cols + BLOCK - 1) / BLOCK);
    p->row_scale = calloc(rows, sizeof(*p->row_scale));
    p->tables = calloc(CONTEXTS, sizeof(*p->tables));
    int rc = (p->row_scale && p->tables) ? 0 : -INGOT_ENOMEM;
    if (!rc)
        rc = ingot_parallel_for((size_t)rows, threads, row_scale_item, p);
    if (!rc)
        rc = plan_histograms(p, threads);
    if (rc) {
        ingot_delta_plan_free(p);
        return rc;
    }
    *out = p;
    return 0;
}

uint64_t ingot_delta_plan_prefix_bytes(const ingot_delta_plan *p) {
    uint64_t bytes = (HEADER_BYTES + p->rows + 7) & ~(uint64_t)7;
    for (unsigned cx = 0; cx < CONTEXTS; cx++)
        bytes += ingot_rans_table_size(&p->tables[cx]);
    return (bytes + 7) & ~(uint64_t)7;
}

uint64_t ingot_delta_plan_bytes(const ingot_delta_plan *p) {
    uint64_t framing = (uint64_t)p->block_count * (INGOT_BLOCK_HEADER_BYTES + 8 + 8) + 8;
    return ingot_delta_plan_prefix_bytes(p) + framing + (p->raw_bits + 7) / 8 +
           (uint64_t)(p->coded_bits / 8.0);
}

static int encode_block(void *ctx, size_t blk, ingot_block_buf *out) {
    const ingot_delta_plan *p = ctx;
    uint64_t lo = (uint64_t)blk * BLOCK, total = p->rows * p->cols;
    size_t n = (size_t)((lo + BLOCK < total ? lo + BLOCK : total) - lo);
    size_t raw_cap = (n * MAX_RAW_BITS + 7) / 8 + 1;
    ingot_rans_symbol *syms = malloc(n * sizeof(*syms));
    uint16_t *words = malloc(n * sizeof(*words));
    ingot_bit_writer raw = {calloc(1, raw_cap), raw_cap, 0};
    int rc = (syms && words && raw.buf) ? 0 : -INGOT_ENOMEM;
    for (size_t i = 0; !rc && i < n; i++) {
        delta_code c = {0, 0, 0};
        unsigned cx = weight_code(p, lo + i, p->row_scale[(lo + i) / p->cols], &c);
        syms[i] = (ingot_rans_symbol){cx == CTX_NONE ? NULL : &p->tables[cx], c.symbol};
        if (cx != CTX_NONE)
            rc = ingot_bits_put(&raw, c.raw, c.raw_bits) ? -INGOT_ENOMEM : 0;
    }
    uint32_t states[INGOT_RANS_LANES];
    size_t word_count = 0;
    if (!rc)
        rc = ingot_rans_encode_block(syms, n, states, words, &word_count);
    if (!rc)
        rc = ingot_block_serialize(states, raw.buf, ingot_bits_bytes(&raw), words, word_count,
                                   out);
    free(syms);
    free(words);
    free(raw.buf);
    return rc;
}

int ingot_delta_encode(const ingot_delta_plan *p, int threads, ingot_sink *s) {
    if (!p || !s || s->pos % 8)
        return -INGOT_EINVAL;
    uint8_t header[HEADER_BYTES];
    ingot_store_u32(header, CONTEXTS);
    ingot_store_u32(header + 4, INGOT_BLOCK_LOG2);
    ingot_store_u64(header + 8, p->rows);
    ingot_store_u64(header + 16, p->cols);
    ingot_store_u64(header + 24, p->block_count);
    int rc = ingot_sink_write(s, header, sizeof(header));
    if (!rc)
        rc = ingot_sink_write(s, p->row_scale, (size_t)p->rows);
    if (!rc)
        rc = ingot_sink_pad(s, 8);
    uint8_t table[2 + 4 * FALLBACK_ALPHABET];
    for (unsigned cx = 0; !rc && cx < CONTEXTS; cx++)
        rc = ingot_sink_write(s, table, ingot_rans_table_write(&p->tables[cx], table));
    if (!rc)
        rc = ingot_sink_pad(s, 8);
    if (!rc)
        rc = ingot_block_area_write(s, p->block_count, threads, encode_block, (void *)p);
    return rc;
}

void ingot_delta_plan_free(ingot_delta_plan *p) {
    if (!p)
        return;
    free(p->row_scale);
    free(p->tables);
    free(p);
}

size_t ingot_delta_plan_blocks(const ingot_delta_plan *p) { return p->block_count; }

struct ingot_delta_decoder {
    int8_t *row_scale;
    uint64_t rows;
    uint64_t cols;
    size_t block_count;
    ingot_rans_table *tables;
    uint16_t *slots;
};

/* Decode one weight of a non-identical row; returns 0 or -1 on corrupt input. */
static inline int decode_weight(const ingot_delta_decoder *d, int row_scale, uint16_t b,
                                uint32_t *x, const ingot_block_view *v, size_t *wp,
                                ingot_bit_reader *raw, uint16_t *out) {
    unsigned cx = delta_context(row_scale, b);
    const ingot_rans_table *t = &d->tables[cx];
    if (!t->used)
        return -1;
    int32_t s = ingot_rans_decode(x, t, d->slots + (size_t)cx * INGOT_RANS_TOTAL, v->words,
                                  v->word_count, wp);
    if (s < 0)
        return -1;
    if (cx == CTX_FALLBACK) {
        uint32_t r6 = 0;
        if (ingot_bits_get(raw, 6, &r6))
            return -1;
        *out = ingot_bf16_join((uint32_t)s, r6);
        return 0;
    }
    int32_t step = 0;
    if (step_decode((uint32_t)s, raw, &step))
        return -1;
    int32_t ord = ingot_bf16_ordered(b) + step;
    if (ord < -32768 || ord > 32767)
        return -1;
    *out = ingot_bf16_from_ordered(ord);
    return 0;
}

static int parse_prefix(const uint8_t *prefix, uint64_t avail, ingot_delta_decoder *d,
                        uint64_t *at) {
    *at = (HEADER_BYTES + d->rows + 7) & ~(uint64_t)7;
    if (*at > avail)
        return -INGOT_EFORMAT;
    memcpy(d->row_scale, prefix + HEADER_BYTES, (size_t)d->rows);
    for (unsigned cx = 0; cx < CONTEXTS; cx++) {
        size_t used = 0;
        int rc = ingot_rans_table_read(&d->tables[cx], context_alphabet(cx), prefix + *at,
                                       avail - *at, &used);
        if (rc)
            return rc;
        *at += used;
    }
    *at = (*at + 7) & ~(uint64_t)7;
    return *at <= avail ? 0 : -INGOT_EFORMAT;
}

int ingot_delta_decoder_open(const uint8_t *prefix, uint64_t avail, uint64_t count,
                             ingot_delta_decoder **out, uint64_t *prefix_bytes) {
    if (!prefix || !out || !prefix_bytes || avail < HEADER_BYTES)
        return -INGOT_EFORMAT;
    ingot_delta_decoder *d = calloc(1, sizeof(*d));
    if (!d)
        return -INGOT_ENOMEM;
    d->rows = ingot_load_u64(prefix + 8);
    d->cols = ingot_load_u64(prefix + 16);
    uint64_t block_count = ingot_load_u64(prefix + 24);
    int rc = 0;
    if (ingot_load_u32(prefix) != CONTEXTS || ingot_load_u32(prefix + 4) != INGOT_BLOCK_LOG2 ||
        !d->rows || !d->cols || d->rows > count || d->rows * d->cols != count ||
        block_count != (count + BLOCK - 1) / BLOCK)
        rc = -INGOT_EFORMAT;
    if (!rc) {
        d->block_count = (size_t)block_count;
        d->row_scale = malloc((size_t)d->rows);
        d->tables = calloc(CONTEXTS, sizeof(*d->tables));
        d->slots = malloc((size_t)CONTEXTS * INGOT_RANS_TOTAL * sizeof(*d->slots));
        rc = (d->row_scale && d->tables && d->slots) ? 0 : -INGOT_ENOMEM;
    }
    uint64_t at = 0;
    if (!rc)
        rc = parse_prefix(prefix, avail, d, &at);
    for (unsigned cx = 0; !rc && cx < CONTEXTS; cx++)
        if (d->tables[cx].used)
            ingot_rans_slots_build(&d->tables[cx], d->slots + (size_t)cx * INGOT_RANS_TOTAL);
    if (rc) {
        ingot_delta_decoder_close(d);
        return rc;
    }
    *prefix_bytes = at;
    *out = d;
    return 0;
}

size_t ingot_delta_decoder_blocks(const ingot_delta_decoder *d) { return d->block_count; }

void ingot_delta_decoder_range(const ingot_delta_decoder *d, size_t b, uint64_t *lo,
                               uint64_t *hi) {
    uint64_t total = d->rows * d->cols;
    *lo = (uint64_t)b * BLOCK;
    *hi = *lo + BLOCK < total ? *lo + BLOCK : total;
}

int ingot_delta_decoder_block(const ingot_delta_decoder *d, size_t b, const ingot_block_view *v,
                              const uint16_t *base, uint16_t *out) {
    uint64_t lo = 0, hi = 0;
    ingot_delta_decoder_range(d, b, &lo, &hi);
    size_t n = (size_t)(hi - lo);
    uint32_t x[INGOT_RANS_LANES];
    memcpy(x, v->states, sizeof(x));
    ingot_bit_reader raw = {v->raw, v->raw_bytes, 0};
    size_t wp = 0;
    uint64_t row = lo / d->cols, col = lo % d->cols;
    for (size_t i = 0; i < n; i++) {
        int rs = d->row_scale[row];
        if (rs == ROW_IDENTICAL)
            out[i] = base[i];
        else if (decode_weight(d, rs, base[i], &x[i % INGOT_RANS_LANES], v, &wp, &raw, &out[i]))
            return -INGOT_ECORRUPT;
        if (++col == d->cols) {
            col = 0;
            row++;
        }
    }
    if ((raw.bitpos + 7) / 8 != v->raw_bytes)
        return -INGOT_ECORRUPT;
    return ingot_block_check_end(x, wp, v);
}

void ingot_delta_decoder_close(ingot_delta_decoder *d) {
    if (!d)
        return;
    free(d->row_scale);
    free(d->tables);
    free(d->slots);
    free(d);
}

typedef struct {
    const ingot_delta_decoder *dec;
    ingot_block_area area;
    const uint16_t *base;
    uint16_t *out;
} decode_job;

static int decode_block(void *ctx, size_t b) {
    decode_job *job = ctx;
    ingot_block_view v;
    uint64_t lo = 0, hi = 0;
    int rc = ingot_block_view_get(&job->area, b, &v);
    ingot_delta_decoder_range(job->dec, b, &lo, &hi);
    return rc ? rc : ingot_delta_decoder_block(job->dec, b, &v, job->base + lo, job->out + lo);
}

int ingot_delta_decode(const uint8_t *payload, uint64_t size, const uint16_t *base,
                       uint64_t count, int threads, uint16_t *out) {
    if (!payload || !base || !out)
        return -INGOT_EFORMAT;
    ingot_delta_decoder *d = NULL;
    uint64_t at = 0;
    int rc = ingot_delta_decoder_open(payload, size, count, &d, &at);
    decode_job job = {.dec = d, .base = base, .out = out};
    if (!rc)
        rc = ingot_block_area_open(payload + at, size - at, d->block_count, &job.area);
    if (!rc)
        rc = ingot_parallel_for(d->block_count, threads, decode_block, &job);
    ingot_delta_decoder_close(d);
    return rc;
}
