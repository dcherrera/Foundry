/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_tensor_codec.c
 * @brief  Block decoder dispatch over the coded modes (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_tensor_codec.h"

#include <stdlib.h>
#include <string.h>

#include "ingot/codec/ingot_delta.h"
#include "ingot/codec/ingot_match.h"
#include "ingot/codec/ingot_plain.h"
#include "ingot/codec/ingot_predict.h"
#include "ingot/codec/ingot_qtensor.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_parallel.h"

struct ingot_tensor_decoder {
    ingot_mode mode;
    ingot_plain_decoder *plain;
    ingot_delta_decoder *delta;
    ingot_qtensor_decoder *quant;
    ingot_predict_decoder *predict;
    ingot_match_decoder *match;
};

void ingot_entry_rows_cols(const ingot_entry *e, uint64_t *rows, uint64_t *cols) {
    uint64_t n = 1;
    for (int d = 0; d < e->ndim; d++)
        n *= e->shape[d];
    *cols = e->ndim ? e->shape[e->ndim - 1] : 1;
    *rows = *cols ? n / *cols : 0;
}

/* A predict payload: 2-D BF16, naming the same partners as its directory field. */
static int open_predict(const ingot_entry *e, const uint8_t *prefix, uint64_t avail,
                        ingot_predict_decoder **out, uint64_t *prefix_bytes) {
    uint32_t listed[INGOT_PREDICT_MAX_PARTNERS] = {0};
    uint32_t n = ingot_partner_list(e->base_hash, listed);
    if (e->ndim != 2 || e->dtype != INGOT_DTYPE_BF16 || !n)
        return -INGOT_EFORMAT;
    int rc = ingot_predict_decoder_open(e->shape[0], e->shape[1], prefix, avail, out,
                                        prefix_bytes);
    if (rc)
        return rc;
    const ingot_predict_geometry *g = ingot_predict_decoder_geometry(*out);
    rc = g->partner_count == n ? 0 : -INGOT_EFORMAT;
    for (uint32_t j = 0; !rc && j < n; j++)
        rc = g->partner[j].entry == listed[j] ? 0 : -INGOT_EFORMAT;
    if (rc) {
        ingot_predict_decoder_close(*out);
        *out = NULL;
    }
    return rc;
}

/* A match payload: 2-D BF16, naming the same sources as its directory list (by hash). */
static int open_match(const ingot_entry *e, const uint8_t *prefix, uint64_t avail,
                      ingot_match_decoder **out, uint64_t *prefix_bytes) {
    if (e->ndim != 2 || e->dtype != INGOT_DTYPE_BF16)
        return -INGOT_EFORMAT;
    int rc = ingot_match_decoder_open(e->shape[0], e->shape[1], prefix, avail, out, prefix_bytes);
    if (rc)
        return rc;
    const ingot_match_params *p = ingot_match_decoder_params(*out);
    uint32_t refs[INGOT_MATCH_MAX_SOURCES];
    for (uint32_t s = 0; s < p->source_count; s++)
        refs[s] = p->source[s].ref;
    if (ingot_match_ref_hash(refs, p->source_count) != e->base_hash) {
        ingot_match_decoder_close(*out);
        *out = NULL;
        return -INGOT_EFORMAT;
    }
    return 0;
}

int ingot_tensor_decoder_open(const ingot_entry *e, const uint8_t *prefix, uint64_t avail,
                              ingot_tensor_decoder **out, uint64_t *prefix_bytes) {
    if (!e || !prefix || !out || !prefix_bytes)
        return -INGOT_EINVAL;
    ingot_tensor_decoder *d = calloc(1, sizeof(*d));
    if (!d)
        return -INGOT_ENOMEM;
    d->mode = e->mode;
    uint64_t rows = 0, cols = 0;
    ingot_entry_rows_cols(e, &rows, &cols);
    int rc = -INGOT_EFORMAT;
    if (e->mode == INGOT_MODE_PLAIN)
        rc = ingot_plain_decoder_open(prefix, avail, e->source_size / 2, &d->plain, prefix_bytes);
    else if (e->mode == INGOT_MODE_DELTA)
        rc = ingot_delta_decoder_open(prefix, avail, e->source_size / 2, &d->delta, prefix_bytes);
    else if (e->mode == INGOT_MODE_PREDICT)
        rc = open_predict(e, prefix, avail, &d->predict, prefix_bytes);
    else if (e->mode == INGOT_MODE_MATCH)
        rc = open_match(e, prefix, avail, &d->match, prefix_bytes);
    else if (e->mode != INGOT_MODE_RAW && e->mode < INGOT_MODE_COUNT)
        rc = ingot_qtensor_decoder_open(e->mode, e->dtype, rows, cols, prefix, avail, &d->quant,
                                        prefix_bytes);
    if (rc) {
        ingot_tensor_decoder_close(d);
        return rc;
    }
    *out = d;
    return 0;
}

size_t ingot_tensor_decoder_blocks(const ingot_tensor_decoder *d) {
    if (d->plain)
        return ingot_plain_decoder_blocks(d->plain);
    if (d->delta)
        return ingot_delta_decoder_blocks(d->delta);
    if (d->predict)
        return ingot_predict_decoder_blocks(d->predict);
    if (d->match)
        return ingot_match_decoder_blocks(d->match);
    return ingot_qtensor_decoder_blocks(d->quant);
}

void ingot_tensor_decoder_range(const ingot_tensor_decoder *d, size_t b, uint64_t *lo,
                                uint64_t *hi) {
    if (d->quant) {
        ingot_qtensor_decoder_range(d->quant, b, lo, hi);
        return;
    }
    if (d->plain)
        ingot_plain_decoder_range(d->plain, b, lo, hi);
    else if (d->predict)
        ingot_predict_decoder_range(d->predict, b, lo, hi);
    else if (d->match)
        ingot_match_decoder_range(d->match, b, lo, hi);
    else
        ingot_delta_decoder_range(d->delta, b, lo, hi);
    *lo *= 2; /* weights -> bytes */
    *hi *= 2;
}

int ingot_tensor_decoder_serial(const ingot_tensor_decoder *d) {
    return d->match ? ingot_match_decoder_serial(d->match) : 0;
}

int ingot_tensor_decoder_needs_base(const ingot_tensor_decoder *d) {
    return d->mode == INGOT_MODE_DELTA || d->mode == INGOT_MODE_QUANT_DELTA ||
           d->mode == INGOT_MODE_BLOCK_DELTA;
}

uint32_t ingot_tensor_decoder_partners(const ingot_tensor_decoder *d, uint32_t *entries) {
    if (d->match) {
        const ingot_match_params *p = ingot_match_decoder_params(d->match);
        for (uint32_t s = 0; entries && s < p->source_count; s++)
            entries[s] = p->source[s].ref;
        return p->source_count;
    }
    if (!d->predict)
        return 0;
    const ingot_predict_geometry *g = ingot_predict_decoder_geometry(d->predict);
    for (uint32_t j = 0; entries && j < g->partner_count; j++)
        entries[j] = g->partner[j].entry;
    return g->partner_count;
}

int ingot_tensor_decoder_bind(ingot_tensor_decoder *d, uint32_t j, const ingot_entry *pe,
                              const uint8_t *data) {
    if (d->match) {
        const ingot_match_params *p = ingot_match_decoder_params(d->match);
        if (j < p->source_count && p->source[j].self)
            return 0; /* blocks read the tensor being decoded */
        if (!pe || !data || pe->dtype != INGOT_DTYPE_BF16 || pe->ndim != 2 ||
            pe->shape[0] * pe->shape[1] * 2 != pe->source_size)
            return -INGOT_EFORMAT;
        return ingot_match_decoder_bind(d->match, j, pe->shape[0], pe->shape[1],
                                        (const uint16_t *)(const void *)data);
    }
    if (!d->predict || !pe || !data || pe->dtype != INGOT_DTYPE_BF16 || pe->ndim != 2 ||
        pe->shape[0] * pe->shape[1] * 2 != pe->source_size)
        return -INGOT_EFORMAT;
    return ingot_predict_decoder_bind(d->predict, j, pe->shape[0], pe->shape[1],
                                      (const uint16_t *)(const void *)data);
}

int ingot_tensor_decoder_block(const ingot_tensor_decoder *d, size_t b,
                               const ingot_block_view *v, const uint8_t *base, uint8_t *out) {
    if (ingot_tensor_decoder_needs_base(d) && !base)
        return -INGOT_EBASE;
    if (d->plain)
        return ingot_plain_decoder_block(d->plain, b, v, (uint16_t *)(void *)out);
    if (d->predict)
        return ingot_predict_decoder_block(d->predict, b, v, (uint16_t *)(void *)out);
    if (d->match)
        return ingot_match_decoder_block(d->match, b, v, (uint16_t *)(void *)out);
    if (d->delta)
        return ingot_delta_decoder_block(d->delta, b, v, (const uint16_t *)(const void *)base,
                                         (uint16_t *)(void *)out);
    return ingot_qtensor_decoder_block(d->quant, b, v, base, out);
}

void ingot_tensor_decoder_close(ingot_tensor_decoder *d) {
    if (!d)
        return;
    ingot_plain_decoder_close(d->plain);
    ingot_delta_decoder_close(d->delta);
    ingot_qtensor_decoder_close(d->quant);
    ingot_predict_decoder_close(d->predict);
    ingot_match_decoder_close(d->match);
    free(d);
}

typedef struct {
    const ingot_tensor_decoder *dec;
    ingot_block_area area;
    const uint8_t *base;
    uint8_t *out;
    uint64_t limit; /* bytes of out: a block past it decodes aside and is cut */
} payload_job;

static int payload_block(void *ctx, size_t b) {
    payload_job *job = ctx;
    ingot_block_view v;
    uint64_t lo = 0, hi = 0;
    int rc = ingot_block_view_get(&job->area, b, &v);
    ingot_tensor_decoder_range(job->dec, b, &lo, &hi);
    const uint8_t *base = job->base ? job->base + lo : NULL;
    if (rc || hi <= job->limit)
        return rc ? rc : ingot_tensor_decoder_block(job->dec, b, &v, base, job->out + lo);
    uint8_t *aside = malloc((size_t)(hi - lo));
    rc = aside ? ingot_tensor_decoder_block(job->dec, b, &v, base, aside) : -INGOT_ENOMEM;
    if (!rc)
        memcpy(job->out + lo, aside, (size_t)(job->limit - lo));
    free(aside);
    return rc;
}

/* Blocks whose range starts below limit (ranges ascend from 0). */
static size_t blocks_below(const ingot_tensor_decoder *d, uint64_t limit) {
    size_t n = ingot_tensor_decoder_blocks(d), b = 0;
    uint64_t lo = 0, hi = 0;
    while (b < n && (ingot_tensor_decoder_range(d, b, &lo, &hi), lo < limit))
        b++;
    return b;
}

/* Bind every partner a predict payload names. */
static int bind_partners(ingot_tensor_decoder *d, const ingot_partner_data *partners) {
    uint32_t n = ingot_tensor_decoder_partners(d, NULL);
    if (n && !partners)
        return -INGOT_EINVAL;
    int rc = 0;
    for (uint32_t j = 0; !rc && j < n; j++)
        rc = ingot_tensor_decoder_bind(d, j, partners[j].entry, partners[j].data);
    return rc;
}

/* Decode bytes [0, limit) of a payload (limit = source_size: all of it). prefix_only: refuse a
 * payload that needs a base or partners. */
static int decode_payload(const ingot_entry *e, const uint8_t *payload, uint64_t size, int framed,
                          const uint8_t *base, const ingot_partner_data *partners, int threads,
                          uint64_t limit, int prefix_only, uint8_t *out) {
    uint64_t frame_prefix = 0, frame_blocks = 0, skip = 0;
    if (framed) {
        if (size < INGOT_PAYLOAD_FRAME_BYTES)
            return -INGOT_EFORMAT;
        frame_prefix = ingot_load_u64(payload);
        frame_blocks = ingot_load_u64(payload + 8);
        skip = INGOT_PAYLOAD_FRAME_BYTES;
    }
    ingot_tensor_decoder *d = NULL;
    uint64_t at = 0;
    int rc = ingot_tensor_decoder_open(e, payload + skip, size - skip, &d, &at);
    if (!rc && framed && (frame_prefix != skip + at ||
                          frame_blocks != ingot_tensor_decoder_blocks(d)))
        rc = -INGOT_EFORMAT;
    if (!rc && prefix_only &&
        (ingot_tensor_decoder_needs_base(d) || ingot_tensor_decoder_partners(d, NULL)))
        rc = -INGOT_EUNSUPPORTED;
    if (!rc && !prefix_only)
        rc = bind_partners(d, partners);
    payload_job job = {.dec = d, .base = base, .out = out, .limit = limit};
    if (!rc)
        rc = ingot_block_area_open(payload + skip + at, size - skip - at,
                                   ingot_tensor_decoder_blocks(d), &job.area);
    if (!rc)
        rc = ingot_parallel_for(blocks_below(d, limit),
                                ingot_tensor_decoder_serial(d) ? 1 : threads, payload_block, &job);
    ingot_tensor_decoder_close(d);
    return rc;
}

int ingot_tensor_decode_payload(const ingot_entry *e, const uint8_t *payload, uint64_t size,
                                int framed, const uint8_t *base,
                                const ingot_partner_data *partners, int threads, uint8_t *out) {
    if (!e || !payload || !out)
        return -INGOT_EINVAL;
    return decode_payload(e, payload, size, framed, base, partners, threads, e->source_size, 0,
                          out);
}

int ingot_tensor_decode_payload_prefix(const ingot_entry *e, const uint8_t *payload,
                                       uint64_t size, int framed, uint64_t n, int threads,
                                       uint8_t *out) {
    if (!e || !payload || !out || n > e->source_size)
        return -INGOT_EINVAL;
    return n ? decode_payload(e, payload, size, framed, NULL, NULL, threads, n, 1, out) : 0;
}
