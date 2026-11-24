/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_writer_tensor.c
 * @brief  Mode choice and payload coding of one buffered tensor (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_writer_tensor.h"

#include <stdlib.h>
#include <string.h>

#include "ingot/codec/ingot_delta.h"
#include "ingot/codec/ingot_plain.h"
#include "ingot/codec/ingot_qtensor.h"
#include "ingot/codec/ingot_tensor_codec.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"

/* A 3-D BF16 tensor gets one table per leading-dim slice (MoE experts) only if each slice spans
 * at least one block; below that the tables would cost more than they save. Plain and predict
 * estimates closer than 1 / CLOSE_CALL are both encoded and the exact sizes decide. */
enum { MIN_GROUP_WEIGHTS = 1 << INGOT_BLOCK_LOG2, CLOSE_CALL = 100 };

static uint32_t group_count_for(const ingot_entry *e, uint64_t count) {
    if (e->ndim < 3 || !e->shape[0] || count % e->shape[0])
        return 1;
    return count / e->shape[0] >= MIN_GROUP_WEIGHTS ? (uint32_t)e->shape[0] : 1;
}

/* The candidate plans of one tensor; the chosen one is encoded. */
typedef struct {
    ingot_plain_plan *plain;
    ingot_delta_plan *delta;
    ingot_qtensor_plan *quant;
    ingot_predict_plan *predict;
    ingot_match_plan *match;
    uint64_t plain_bytes;
    uint64_t predict_bytes; /* the predict or match candidate */
    ingot_mode v2_mode;  /* best choice without predict */
    uint64_t v2_bytes;   /* its estimate */
    uint64_t v2_exact;   /* its exact payload when it was encoded, else 0 */
    ingot_mode mode;
    uint64_t bytes;
    const uint8_t *base; /* the base tensor (owned by the base set), or NULL */
} coding_choice;

static void choice_free(coding_choice *c) {
    ingot_plain_plan_free(c->plain);
    ingot_delta_plan_free(c->delta);
    ingot_qtensor_plan_free(c->quant);
    ingot_predict_plan_free(c->predict);
    ingot_match_plan_free(c->match);
    *c = (coding_choice){0};
}

/* Keep candidate (mode, bytes) if it beats the current choice (or is forced). */
static void consider(coding_choice *c, ingot_mode mode, uint64_t bytes, int forced) {
    if (!forced && bytes >= c->bytes)
        return;
    c->mode = mode;
    c->bytes = bytes;
}

/* The matching base tensor (same dtype and size), loaded whole; NULL if none. */
static int base_tensor(const ingot_encode_options *opt, ingot_entry *e, const uint8_t **base) {
    ingot_base_tensor t;
    *base = NULL;
    if (!opt->base || opt->policy == INGOT_POLICY_PLAIN ||
        ingot_base_set_find(opt->base, e->name, &t) || t.dtype != e->dtype ||
        t.size != e->source_size)
        return 0;
    int rc = ingot_base_set_load(opt->base, &t, base);
    if (!rc)
        e->base_hash = ingot_hash64(*base, (size_t)e->source_size);
    return rc;
}

static int choose_predict(const ingot_encode_options *opt, const uint8_t *data,
                          const ingot_writer_partners *p, coding_choice *c) {
    const uint16_t *partners[INGOT_PREDICT_MAX_PARTNERS] = {0};
    for (uint32_t j = 0; j < p->geom.partner_count; j++)
        partners[j] = (const uint16_t *)(const void *)p->data[j];
    int rc = ingot_predict_plan_create(&p->geom, (const uint16_t *)(const void *)data, partners,
                                       opt->threads, &c->predict);
    if (rc)
        return rc;
    c->predict_bytes = INGOT_PAYLOAD_FRAME_BYTES + ingot_predict_plan_bytes(c->predict);
    consider(c, INGOT_MODE_PREDICT, c->predict_bytes, 0);
    return 0;
}

static int choose_match(const ingot_encode_options *opt, const uint8_t *data,
                        const ingot_writer_match *m, coding_choice *c) {
    int rc = ingot_match_plan_create(m->axis, m->units, m->unit_len,
                                     (const uint16_t *)(const void *)data, m->unit_major, m->src,
                                     m->source_count, &m->cand, opt->threads, &c->match);
    if (rc)
        return rc;
    c->predict_bytes = INGOT_PAYLOAD_FRAME_BYTES + ingot_match_plan_bytes(c->match);
    consider(c, INGOT_MODE_MATCH, c->predict_bytes, 0);
    return 0;
}

static int choose_bf16(const ingot_encode_options *opt, ingot_entry *e, const uint8_t *data,
                       const uint8_t *base, const ingot_writer_partners *partners,
                       coding_choice *c) {
    uint64_t count = e->source_size / 2, rows = 0, cols = 0;
    const uint16_t *w = (const uint16_t *)(const void *)data;
    int rc = ingot_plain_plan_create(w, count, group_count_for(e, count), opt->threads,
                                     &c->plain);
    if (!rc) {
        c->plain_bytes = INGOT_PAYLOAD_FRAME_BYTES + ingot_plain_plan_bytes(c->plain);
        consider(c, INGOT_MODE_PLAIN, c->plain_bytes, 0);
    }
    ingot_entry_rows_cols(e, &rows, &cols);
    if (!rc && base)
        rc = ingot_delta_plan_create(w, (const uint16_t *)(const void *)base, rows, cols,
                                     opt->threads, &c->delta);
    if (!rc && base)
        consider(c, INGOT_MODE_DELTA, INGOT_PAYLOAD_FRAME_BYTES + ingot_delta_plan_bytes(c->delta),
                 opt->policy == INGOT_POLICY_DELTA);
    c->v2_mode = c->mode;
    c->v2_bytes = c->bytes;
    if (!rc && partners && partners->match && c->mode != INGOT_MODE_DELTA)
        rc = choose_match(opt, data, partners->match, c);
    else if (!rc && partners && c->mode != INGOT_MODE_DELTA)
        rc = choose_predict(opt, data, partners, c);
    return rc;
}

static int choose_quant(const ingot_encode_options *opt, ingot_entry *e, const uint8_t *data,
                        const uint8_t *base, coding_choice *c) {
    uint64_t rows = 0, cols = 0;
    ingot_entry_rows_cols(e, &rows, &cols);
    ingot_mode mode = INGOT_MODE_QUANT;
    if (base)
        mode = ingot_qtensor_fits(INGOT_MODE_QUANT_DELTA, e->dtype, cols) ? INGOT_MODE_QUANT_DELTA
                                                                          : INGOT_MODE_BLOCK_DELTA;
    if (!ingot_qtensor_fits(mode, e->dtype, cols))
        return 0;
    int rc = ingot_qtensor_plan_create(mode, e->dtype, data, base, rows, cols, opt->threads,
                                       &c->quant);
    if (!rc)
        consider(c, mode, INGOT_PAYLOAD_FRAME_BYTES + ingot_qtensor_plan_bytes(c->quant),
                 base && opt->policy == INGOT_POLICY_DELTA);
    return rc;
}

/* Plan every candidate mode of a buffered tensor and keep the smallest (raw if nothing wins). */
static int choose_mode(const ingot_encode_options *opt, ingot_entry *e, const uint8_t *data,
                       const ingot_writer_partners *partners, coding_choice *c) {
    *c = (coding_choice){.mode = INGOT_MODE_RAW, .bytes = e->source_size,
                         .v2_mode = INGOT_MODE_RAW, .v2_bytes = e->source_size};
    const uint8_t *base = NULL;
    int rc = base_tensor(opt, e, &base);
    c->base = base;
    if (!rc && e->dtype == INGOT_DTYPE_BF16 && e->source_size >= 2 && e->source_size % 2 == 0)
        rc = choose_bf16(opt, e, data, base, partners, c);
    else if (!rc && e->source_size && (base || ingot_qtensor_supports(e->dtype)))
        rc = choose_quant(opt, e, data, base, c);
    return rc;
}

static int write_frame(ingot_sink *s, uint64_t prefix, size_t blocks) {
    uint8_t frame[INGOT_PAYLOAD_FRAME_BYTES];
    ingot_store_u64(frame, INGOT_PAYLOAD_FRAME_BYTES + prefix);
    ingot_store_u64(frame + 8, blocks);
    return ingot_sink_write(s, frame, sizeof(frame));
}

static int encode_mode(const ingot_encode_options *opt, const coding_choice *c, ingot_mode mode,
                       const uint8_t *data, uint64_t size, ingot_sink *s) {
    int rc = 0;
    switch (mode) {
    case INGOT_MODE_PLAIN:
        rc = write_frame(s, ingot_plain_plan_prefix_bytes(c->plain),
                         ingot_plain_plan_blocks(c->plain));
        return rc ? rc : ingot_plain_encode(c->plain, opt->threads, s);
    case INGOT_MODE_DELTA:
        rc = write_frame(s, ingot_delta_plan_prefix_bytes(c->delta),
                         ingot_delta_plan_blocks(c->delta));
        return rc ? rc : ingot_delta_encode(c->delta, opt->threads, s);
    case INGOT_MODE_PREDICT:
        rc = write_frame(s, ingot_predict_plan_prefix_bytes(c->predict),
                         ingot_predict_plan_blocks(c->predict));
        return rc ? rc : ingot_predict_encode(c->predict, opt->threads, s);
    case INGOT_MODE_MATCH:
        rc = write_frame(s, ingot_match_plan_prefix_bytes(c->match),
                         ingot_match_plan_blocks(c->match));
        return rc ? rc : ingot_match_encode(c->match, opt->threads, s);
    case INGOT_MODE_RAW:
        return ingot_sink_write(s, data, (size_t)size);
    default:
        rc = write_frame(s, ingot_qtensor_plan_prefix_bytes(c->quant),
                         ingot_qtensor_plan_blocks(c->quant));
        return rc ? rc : ingot_qtensor_encode(c->quant, opt->threads, s);
    }
}

/* The candidate that predicts (predict or match) and its mode, if any. */
static ingot_mode predicted_mode(const coding_choice *c) {
    return c->match ? INGOT_MODE_MATCH : c->predict ? INGOT_MODE_PREDICT : INGOT_MODE_RAW;
}

/* Plain vs predict / match within the estimate's error: encode the other one too, keep the
 * smaller. */
static int settle_close_call(const ingot_encode_options *opt, coding_choice *c,
                             const uint8_t *data, uint64_t size, ingot_sink *payload) {
    ingot_mode pm = predicted_mode(c);
    if (!c->plain || pm == INGOT_MODE_RAW || (c->mode != INGOT_MODE_PLAIN && c->mode != pm))
        return 0;
    uint64_t hi = c->plain_bytes > c->predict_bytes ? c->plain_bytes : c->predict_bytes;
    uint64_t lo = c->plain_bytes > c->predict_bytes ? c->predict_bytes : c->plain_bytes;
    if (hi - lo > hi / CLOSE_CALL)
        return 0;
    ingot_mode other = c->mode == INGOT_MODE_PLAIN ? pm : INGOT_MODE_PLAIN;
    ingot_sink alt = {.fd = INGOT_SINK_MEMORY};
    int rc = encode_mode(opt, c, other, data, size, &alt);
    if (!rc && alt.pos < payload->pos) {
        ingot_sink kept = *payload;
        *payload = alt;
        alt = kept;
        c->mode = other;
    }
    if (!rc && c->mode == pm && c->v2_mode == INGOT_MODE_PLAIN)
        c->v2_exact = alt.pos; /* the plain payload, encoded either way */
    free(alt.mem);
    return rc;
}

/* Decode the payload (in memory) and compare it with the source tensor. */
static int self_check(const ingot_encode_options *opt, const ingot_sink *payload,
                      const ingot_entry *e, const uint8_t *source, const uint8_t *base,
                      const ingot_writer_partners *p, const ingot_match_plan *match) {
    ingot_partner_data partners[INGOT_MATCH_MAX_SOURCES] = {{0}};
    for (uint32_t j = 0; p && !match && j < p->geom.partner_count; j++)
        partners[j] = (ingot_partner_data){p->entry[j], p->data[j]};
    if (match) { /* the plan's sources (unused ones pruned), found by ref in the candidate */
        const ingot_match_params *q = ingot_match_plan_params(match);
        for (uint32_t s = 0; s < q->source_count; s++)
            for (uint32_t k = 0; k < p->match->source_count; k++)
                if (p->match->src[k].ref == q->source[s].ref)
                    partners[s] = (ingot_partner_data){p->match->src_entry[k],
                                                       (const uint8_t *)(const void *)
                                                           p->match->src[k].data};
    }
    uint8_t *out = malloc((size_t)e->source_size);
    int rc = out ? 0 : -INGOT_ENOMEM;
    if (!rc)
        rc = ingot_tensor_decode_payload(e, payload->mem, payload->pos, 1, base,
                                         p ? partners : NULL, opt->threads, out);
    if (!rc && memcmp(out, source, (size_t)e->source_size))
        rc = -INGOT_ECORRUPT;
    free(out);
    return rc;
}

static uint64_t base_field(const coding_choice *c, const ingot_entry *e,
                           const ingot_writer_partners *p, ingot_writer_coded *coded) {
    if (c->mode == INGOT_MODE_DELTA || c->mode == INGOT_MODE_QUANT_DELTA ||
        c->mode == INGOT_MODE_BLOCK_DELTA)
        return e->base_hash;
    if (c->mode == INGOT_MODE_MATCH) {
        const ingot_match_params *q = ingot_match_plan_params(c->match);
        for (uint32_t s = 0; s < q->source_count; s++)
            coded->match_refs[s] = q->source[s].ref;
        coded->match_ref_count = q->source_count;
        return ingot_match_ref_hash(coded->match_refs, q->source_count);
    }
    if (c->mode != INGOT_MODE_PREDICT)
        return 0;
    uint32_t index[INGOT_PREDICT_MAX_PARTNERS];
    for (uint32_t j = 0; j < p->geom.partner_count; j++)
        index[j] = p->geom.partner[j].entry;
    return ingot_partner_field(index, p->geom.partner_count);
}

/* A predicted tensor's best other choice: exact (encoded into nothing) or the estimate. */
static int v2_size(const ingot_encode_options *opt, const coding_choice *c, const uint8_t *data,
                   uint64_t size, ingot_writer_coded *coded) {
    coded->prefix_bytes = c->match ? ingot_match_plan_prefix_bytes(c->match)
                                   : ingot_predict_plan_prefix_bytes(c->predict);
    coded->v2_bytes = c->v2_exact ? c->v2_exact : c->v2_bytes;
    if (c->v2_exact || !opt->exact_v2)
        return 0;
    ingot_sink none = {.fd = INGOT_SINK_DISCARD};
    int rc = encode_mode(opt, c, c->v2_mode, data, size, &none);
    coded->v2_bytes = none.pos;
    return rc;
}

int ingot_writer_code_tensor(const ingot_encode_options *opt, ingot_entry *e,
                             const uint8_t *data, const ingot_writer_partners *partners,
                             ingot_sink *payload, ingot_writer_coded *coded) {
    coding_choice c = {0};
    *coded = (ingot_writer_coded){0};
    int rc = choose_mode(opt, e, data, partners, &c);
    if (!rc)
        rc = encode_mode(opt, &c, c.mode, data, e->source_size, payload);
    if (!rc)
        rc = settle_close_call(opt, &c, data, e->source_size, payload);
    e->mode = c.mode;
    e->base_hash = base_field(&c, e, partners, coded);
    int predicted = e->mode == INGOT_MODE_PREDICT || e->mode == INGOT_MODE_MATCH;
    if (!rc && opt->self_check && e->mode != INGOT_MODE_RAW) {
        e->payload_size = payload->pos;
        rc = self_check(opt, payload, e, data, c.base, predicted ? partners : NULL,
                        e->mode == INGOT_MODE_MATCH ? c.match : NULL);
    }
    coded->v2_bytes = payload->pos;
    if (!rc && predicted)
        rc = v2_size(opt, &c, data, e->source_size, coded);
    choice_free(&c);
    return rc;
}
