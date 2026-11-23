/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_plain.c
 * @brief  Plain-mode BF16 tensor coder (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_plain.h"

#include <stdlib.h>
#include <string.h>

#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_bits.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_parallel.h"

enum { ALPHABET = 1024, HEADER_BYTES = 24, BLOCK = 1 << INGOT_BLOCK_LOG2 };

struct ingot_plain_plan {
    const uint16_t *weights;
    uint64_t count;
    uint64_t group_weights;
    uint32_t group_count;
    size_t blocks_per_group;
    size_t block_count;
    ingot_rans_table *tables;
    double coded_bits;
};

/* Weight range [lo, hi) and group of block b. */
static void block_range(uint64_t group_weights, size_t blocks_per_group, size_t b, uint64_t *lo,
                        uint64_t *hi, size_t *group) {
    size_t g = b / blocks_per_group, local = b % blocks_per_group;
    *lo = (uint64_t)g * group_weights + (uint64_t)local * BLOCK;
    uint64_t end = (uint64_t)(g + 1) * group_weights;
    *hi = *lo + BLOCK < end ? *lo + BLOCK : end;
    *group = g;
}

typedef struct {
    const ingot_plain_plan *plan;
    uint32_t *block_counts;
} histogram_job;

static int histogram_block(void *ctx, size_t b) {
    histogram_job *job = ctx;
    uint64_t lo = 0, hi = 0;
    size_t g = 0;
    block_range(job->plan->group_weights, job->plan->blocks_per_group, b, &lo, &hi, &g);
    uint32_t *c = job->block_counts + b * ALPHABET;
    for (uint64_t i = lo; i < hi; i++)
        c[ingot_bf16_symbol(job->plan->weights[i])]++;
    return 0;
}

static int plan_tables(ingot_plain_plan *p, const uint32_t *block_counts) {
    uint64_t counts[ALPHABET];
    for (uint32_t g = 0; g < p->group_count; g++) {
        memset(counts, 0, sizeof(counts));
        for (size_t b = g * p->blocks_per_group; b < (g + 1) * p->blocks_per_group; b++)
            for (int s = 0; s < ALPHABET; s++)
                counts[s] += block_counts[b * ALPHABET + s];
        int rc = ingot_rans_table_build(&p->tables[g], counts, ALPHABET);
        if (rc)
            return rc;
        p->coded_bits += ingot_rans_table_coded_bits(&p->tables[g], counts);
    }
    return 0;
}

int ingot_plain_plan_create(const uint16_t *weights, uint64_t count, uint32_t group_count,
                            int threads, ingot_plain_plan **out) {
    if (!weights || !out || !count || !group_count || count % group_count)
        return -INGOT_EINVAL;
    ingot_plain_plan *p = calloc(1, sizeof(*p));
    if (!p)
        return -INGOT_ENOMEM;
    p->weights = weights;
    p->count = count;
    p->group_count = group_count;
    p->group_weights = count / group_count;
    p->blocks_per_group = (size_t)((p->group_weights + BLOCK - 1) / BLOCK);
    p->block_count = p->blocks_per_group * group_count;
    p->tables = calloc(group_count, sizeof(*p->tables));
    uint32_t *block_counts = calloc(p->block_count * ALPHABET, sizeof(*block_counts));
    int rc = (p->tables && block_counts) ? 0 : -INGOT_ENOMEM;
    if (!rc) {
        histogram_job job = {p, block_counts};
        rc = ingot_parallel_for(p->block_count, threads, histogram_block, &job);
    }
    if (!rc)
        rc = plan_tables(p, block_counts);
    free(block_counts);
    if (rc) {
        ingot_plain_plan_free(p);
        return rc;
    }
    *out = p;
    return 0;
}

uint64_t ingot_plain_plan_prefix_bytes(const ingot_plain_plan *p) {
    uint64_t bytes = HEADER_BYTES;
    for (uint32_t g = 0; g < p->group_count; g++)
        bytes += ingot_rans_table_size(&p->tables[g]);
    return (bytes + 7) & ~(uint64_t)7;
}

uint64_t ingot_plain_plan_bytes(const ingot_plain_plan *p) {
    uint64_t raw = (p->count * 6 + 7) / 8;
    uint64_t framing = (uint64_t)p->block_count * (INGOT_BLOCK_HEADER_BYTES + 8 + 8) + 8;
    return ingot_plain_plan_prefix_bytes(p) + framing + raw + (uint64_t)(p->coded_bits / 8.0);
}

static int encode_block(void *ctx, size_t b, ingot_block_buf *out) {
    const ingot_plain_plan *p = ctx;
    uint64_t lo = 0, hi = 0;
    size_t g = 0;
    block_range(p->group_weights, p->blocks_per_group, b, &lo, &hi, &g);
    size_t n = (size_t)(hi - lo);
    ingot_rans_symbol *syms = malloc(n * sizeof(*syms));
    uint16_t *words = malloc(n * sizeof(*words));
    ingot_bit_writer raw = {calloc(1, (n * 6 + 7) / 8 + 1), (n * 6 + 7) / 8 + 1, 0};
    int rc = (syms && words && raw.buf) ? 0 : -INGOT_ENOMEM;
    for (size_t i = 0; !rc && i < n; i++) {
        uint16_t w = p->weights[lo + i];
        syms[i] = (ingot_rans_symbol){&p->tables[g], ingot_bf16_symbol(w)};
        rc = ingot_bits_put(&raw, ingot_bf16_raw6(w), 6) ? -INGOT_ENOMEM : 0;
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

int ingot_plain_encode(const ingot_plain_plan *p, int threads, ingot_sink *s) {
    if (!p || !s || s->pos % 8)
        return -INGOT_EINVAL;
    uint8_t header[HEADER_BYTES];
    ingot_store_u32(header, p->group_count);
    ingot_store_u32(header + 4, INGOT_BLOCK_LOG2);
    ingot_store_u64(header + 8, p->group_weights);
    ingot_store_u64(header + 16, p->block_count);
    int rc = ingot_sink_write(s, header, sizeof(header));
    uint8_t table[2 + 4 * ALPHABET];
    for (uint32_t g = 0; !rc && g < p->group_count; g++) {
        size_t n = ingot_rans_table_write(&p->tables[g], table);
        rc = ingot_sink_write(s, table, n);
    }
    if (!rc)
        rc = ingot_sink_pad(s, 8);
    if (!rc)
        rc = ingot_block_area_write(s, p->block_count, threads, encode_block, (void *)p);
    return rc;
}

void ingot_plain_plan_free(ingot_plain_plan *p) {
    if (!p)
        return;
    free(p->tables);
    free(p);
}

struct ingot_plain_decoder {
    uint64_t count;
    uint64_t group_weights;
    size_t blocks_per_group;
    size_t block_count;
    uint32_t group_count;
    ingot_rans_table *tables;
    uint16_t *slots;
};

static int parse_tables(const uint8_t *prefix, uint64_t avail, ingot_plain_decoder *d,
                        uint64_t *at) {
    for (uint32_t g = 0; g < d->group_count; g++) {
        size_t used = 0;
        int rc = ingot_rans_table_read(&d->tables[g], ALPHABET, prefix + *at, avail - *at, &used);
        if (rc)
            return rc;
        *at += used;
    }
    *at = (*at + 7) & ~(uint64_t)7;
    return *at <= avail ? 0 : -INGOT_EFORMAT;
}

int ingot_plain_decoder_open(const uint8_t *prefix, uint64_t avail, uint64_t count,
                             ingot_plain_decoder **out, uint64_t *prefix_bytes) {
    if (!prefix || !out || !prefix_bytes || avail < HEADER_BYTES)
        return -INGOT_EFORMAT;
    ingot_plain_decoder *d = calloc(1, sizeof(*d));
    if (!d)
        return -INGOT_ENOMEM;
    d->count = count;
    d->group_count = ingot_load_u32(prefix);
    d->group_weights = ingot_load_u64(prefix + 8);
    uint64_t block_count = ingot_load_u64(prefix + 16);
    int rc = 0;
    if (ingot_load_u32(prefix + 4) != INGOT_BLOCK_LOG2 || !d->group_count || !d->group_weights ||
        d->group_weights * d->group_count != count)
        rc = -INGOT_EFORMAT;
    if (!rc) {
        d->blocks_per_group = (size_t)((d->group_weights + BLOCK - 1) / BLOCK);
        d->block_count = d->blocks_per_group * d->group_count;
        rc = block_count == d->block_count ? 0 : -INGOT_EFORMAT;
    }
    if (!rc) {
        d->tables = calloc(d->group_count, sizeof(*d->tables));
        d->slots = malloc((size_t)d->group_count * INGOT_RANS_TOTAL * sizeof(*d->slots));
        rc = (d->tables && d->slots) ? 0 : -INGOT_ENOMEM;
    }
    uint64_t at = HEADER_BYTES;
    if (!rc)
        rc = parse_tables(prefix, avail, d, &at);
    for (uint32_t g = 0; !rc && g < d->group_count; g++)
        if (d->tables[g].used)
            ingot_rans_slots_build(&d->tables[g], d->slots + (size_t)g * INGOT_RANS_TOTAL);
    if (rc) {
        ingot_plain_decoder_close(d);
        return rc;
    }
    *prefix_bytes = at;
    *out = d;
    return 0;
}

size_t ingot_plain_decoder_blocks(const ingot_plain_decoder *d) { return d->block_count; }

void ingot_plain_decoder_range(const ingot_plain_decoder *d, size_t b, uint64_t *lo,
                               uint64_t *hi) {
    size_t g = 0;
    block_range(d->group_weights, d->blocks_per_group, b, lo, hi, &g);
}

int ingot_plain_decoder_block(const ingot_plain_decoder *d, size_t b, const ingot_block_view *v,
                              uint16_t *out) {
    uint64_t lo = 0, hi = 0;
    size_t g = 0;
    block_range(d->group_weights, d->blocks_per_group, b, &lo, &hi, &g);
    size_t n = (size_t)(hi - lo);
    if (v->raw_bytes != (n * 6 + 7) / 8 || !d->tables[g].used)
        return -INGOT_EFORMAT;
    const ingot_rans_table *t = &d->tables[g];
    const uint16_t *slots = d->slots + g * INGOT_RANS_TOTAL;
    uint32_t x[INGOT_RANS_LANES];
    memcpy(x, v->states, sizeof(x));
    ingot_bit_reader raw = {v->raw, v->raw_bytes, 0};
    size_t wp = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t s = ingot_rans_decode(&x[i % INGOT_RANS_LANES], t, slots, v->words,
                                      v->word_count, &wp);
        uint32_t r6 = 0;
        if (s < 0 || ingot_bits_get(&raw, 6, &r6))
            return -INGOT_ECORRUPT;
        out[i] = ingot_bf16_join((uint32_t)s, r6);
    }
    return ingot_block_check_end(x, wp, v);
}

void ingot_plain_decoder_close(ingot_plain_decoder *d) {
    if (!d)
        return;
    free(d->tables);
    free(d->slots);
    free(d);
}

typedef struct {
    const ingot_plain_decoder *dec;
    ingot_block_area area;
    uint16_t *out;
} decode_job;

static int decode_block(void *ctx, size_t b) {
    decode_job *job = ctx;
    ingot_block_view v;
    uint64_t lo = 0, hi = 0;
    int rc = ingot_block_view_get(&job->area, b, &v);
    ingot_plain_decoder_range(job->dec, b, &lo, &hi);
    return rc ? rc : ingot_plain_decoder_block(job->dec, b, &v, job->out + lo);
}

int ingot_plain_decode(const uint8_t *payload, uint64_t size, uint64_t count, int threads,
                       uint16_t *out) {
    if (!payload || !out)
        return -INGOT_EFORMAT;
    ingot_plain_decoder *d = NULL;
    uint64_t at = 0;
    int rc = ingot_plain_decoder_open(payload, size, count, &d, &at);
    decode_job job = {.dec = d, .out = out};
    if (!rc)
        rc = ingot_block_area_open(payload + at, size - at, d->block_count, &job.area);
    if (!rc)
        rc = ingot_parallel_for(d->block_count, threads, decode_block, &job);
    ingot_plain_decoder_close(d);
    return rc;
}

size_t ingot_plain_plan_blocks(const ingot_plain_plan *p) { return p->block_count; }
