/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_match.c
 * @brief  Match mode: parameters, the integer predictor, contexts, the side stream, the prefix
 *         and the block decoder (see header). The encoder side is in ingot_match_plan.c.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_match.h"

#include <stdlib.h>
#include <string.h>

#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"

enum {
    THRESHOLDS = INGOT_PREDICT_BUCKETS - 1,
    SYMBOL_ALPHABET = 1024,
    BLOCK = 1 << INGOT_BLOCK_LOG2,
    NEG_BIT = 0x80,
    HEADER = 88,
    SOURCE_RECORD = 24,
    CTX_SCALE = 0,
    CTX_K = 1,
    CTX_LINK = 2,
    CTX_COEF = 3,
};

int ingot_match_params_alloc(ingot_match_params *p) {
    size_t u = (size_t)p->units, slots = u * INGOT_MATCH_MAX_K;
    p->scale = calloc(u ? u : 1, 1);
    p->k = calloc(u ? u : 1, 1);
    p->link = calloc(slots ? slots : 1, 1);
    p->index = calloc(slots ? slots : 1, sizeof(*p->index));
    p->coef = calloc(slots ? slots : 1, sizeof(*p->coef));
    if (p->scale && p->k && p->link && p->index && p->coef)
        return 0;
    ingot_match_params_release(p);
    return -INGOT_ENOMEM;
}

void ingot_match_params_release(ingot_match_params *p) {
    free(p->scale);
    free(p->k);
    free(p->link);
    free(p->index);
    free(p->coef);
    p->scale = NULL;
    p->k = NULL;
    p->link = NULL;
    p->index = NULL;
    p->coef = NULL;
}

/* Raw index bits of a source with n units: ceil(log2 n). */
static unsigned index_bits(uint64_t n) {
    unsigned b = 0;
    while (b < 64 && ((uint64_t)1 << b) < n)
        b++;
    return b;
}

/* Strips: at most the row count, and a strip of at most 2^30 weights. */
static int strip_check(const ingot_match_params *p) {
    uint64_t rows = p->axis == INGOT_PREDICT_ROWS ? p->units : p->unit_len;
    uint64_t cols = p->axis == INGOT_PREDICT_ROWS ? p->unit_len : p->units;
    return p->strip_rows > rows || (p->strip_rows && p->strip_rows > ((uint64_t)1 << 30) / cols) ||
           (p->waves && !p->strip_rows);
}

static int geometry_check(const ingot_match_params *p) {
    if (!p->units || !p->unit_len || p->units > INGOT_PREDICT_LIMIT ||
        p->unit_len > INGOT_PREDICT_LIMIT || p->source_count < 1 ||
        p->source_count > INGOT_MATCH_MAX_SOURCES ||
        (p->axis != INGOT_PREDICT_ROWS && p->axis != INGOT_PREDICT_COLS) || p->classes < 1 ||
        p->classes > INGOT_PREDICT_MAX_CLASSES || p->class_width < 1 ||
        p->class_width > INGOT_PREDICT_MAX_CLASS_WIDTH ||
        p->scale_c < -INGOT_PREDICT_EXP_LIMIT || p->scale_c > INGOT_PREDICT_EXP_LIMIT ||
        p->class_ref < -256 || p->class_ref > 256 || strip_check(p))
        return -INGOT_EINVAL;
    for (uint32_t s = 0; s < p->source_count; s++) {
        const ingot_match_source *src = &p->source[s];
        if (!src->units || index_bits(src->units) > INGOT_MATCH_INDEX_BITS_MAX ||
            (src->axis != INGOT_PREDICT_ROWS && src->axis != INGOT_PREDICT_COLS) ||
            src->coef_exp < -INGOT_PREDICT_EXP_LIMIT || src->coef_exp > INGOT_PREDICT_EXP_LIMIT ||
            (src->self && (src->axis != INGOT_PREDICT_ROWS || p->axis != INGOT_PREDICT_ROWS ||
                           src->units != p->units)))
            return -INGOT_EINVAL;
    }
    return 0;
}

int ingot_match_params_check(const ingot_match_params *p) {
    if (!p || !p->scale || !p->k || !p->link || !p->index || !p->coef || geometry_check(p))
        return -INGOT_EINVAL;
    for (uint64_t u = 0; u < p->units; u++) {
        if (p->k[u] > INGOT_MATCH_MAX_K)
            return -INGOT_EINVAL;
        for (uint32_t k = 0; k < p->k[u]; k++) {
            size_t at = (size_t)u * INGOT_MATCH_MAX_K + k;
            const ingot_match_source *src =
                p->link[at] < p->source_count ? &p->source[p->link[at]] : NULL;
            if (!src || p->index[at] >= src->units || (src->self && p->index[at] >= u) ||
                (src->self && p->strip_rows && !p->waves && p->index[at] < u - u % p->strip_rows) ||
                p->coef[at] < -INGOT_MATCH_COEF_MAX)
                return -INGOT_EINVAL;
        }
    }
    return 0;
}

uint64_t ingot_match_param_hash(const ingot_match_params *p) {
    uint64_t partners = 0;
    for (uint64_t u = 0; u < p->units; u++)
        partners += p->k[u];
    ingot_hash64_state h;
    ingot_hash64_begin(&h, 2 * p->units + 11 * partners);
    for (uint64_t u = 0; u < p->units; u++) {
        uint8_t head[2] = {(uint8_t)p->scale[u], p->k[u]};
        ingot_hash64_update(&h, head, 2);
        for (uint32_t k = 0; k < p->k[u]; k++) {
            size_t at = (size_t)u * INGOT_MATCH_MAX_K + k;
            uint8_t rec[11];
            rec[0] = p->link[at];
            ingot_store_u64(rec + 1, p->index[at]);
            rec[9] = (uint8_t)((uint16_t)p->coef[at] & 0xFFu);
            rec[10] = (uint8_t)((uint16_t)p->coef[at] >> 8);
            ingot_hash64_update(&h, rec, sizeof(rec));
        }
    }
    return ingot_hash64_end(&h);
}

uint64_t ingot_match_ref_hash(const uint32_t *refs, uint32_t n) {
    ingot_hash64_state h;
    ingot_hash64_begin(&h, 4 * (uint64_t)n);
    for (uint32_t i = 0; i < n; i++) {
        uint8_t b[4];
        ingot_store_u32(b, refs[i]);
        ingot_hash64_update(&h, b, 4);
    }
    return ingot_hash64_end(&h);
}

/* ---- Model and contexts ---- */

struct ingot_match_model {
    const ingot_match_params *p;
    int64_t *fixed[INGOT_MATCH_MAX_SOURCES]; /* [65536]; sources with one shift share a table */
    int owns[INGOT_MATCH_MAX_SOURCES];
    int64_t *thresholds;                     /* [U][THRESHOLDS] */
    uint8_t *zbase;                          /* [U] */
    uint32_t zcount;
    ingot_fc_model fc;
};

static int build_fixed(ingot_match_model *m) {
    const ingot_match_params *p = m->p;
    for (uint32_t s = 0; s < p->source_count; s++) {
        int32_t shift = p->scale_c - p->source[s].coef_exp;
        for (uint32_t t = 0; t < s && !m->fixed[s]; t++)
            if (p->scale_c - p->source[t].coef_exp == shift)
                m->fixed[s] = m->fixed[t];
        if (m->fixed[s])
            continue;
        m->fixed[s] = malloc(65536 * sizeof(int64_t));
        if (!m->fixed[s])
            return -INGOT_ENOMEM;
        m->owns[s] = 1;
        for (uint32_t b = 0; b < 65536; b++)
            m->fixed[s][b] = ingot_predict_fixed((uint16_t)b, shift);
    }
    return 0;
}

int ingot_match_model_create(const ingot_match_params *p, ingot_match_model **out) {
    if (!p || !out || geometry_check(p) || !p->scale)
        return -INGOT_EINVAL;
    ingot_match_model *m = calloc(1, sizeof(*m));
    if (!m)
        return -INGOT_ENOMEM;
    m->p = p;
    m->zcount = p->classes * INGOT_PREDICT_BUCKETS;
    m->thresholds = malloc((size_t)p->units * THRESHOLDS * sizeof(*m->thresholds));
    m->zbase = malloc((size_t)p->units);
    int rc = (m->thresholds && m->zbase) ? build_fixed(m) : -INGOT_ENOMEM;
    for (uint64_t u = 0; !rc && u < p->units; u++) {
        ingot_predict_thresholds(p->scale[u], p->scale_c, m->thresholds + u * THRESHOLDS);
        m->zbase[u] = (uint8_t)(INGOT_PREDICT_BUCKETS *
                                ingot_predict_class(p->scale[u], p->classes, p->class_width,
                                                    p->class_ref));
    }
    if (!rc)
        rc = ingot_fc_model_add(&m->fc, m->zcount, 2, NULL);
    if (!rc)
        rc = ingot_fc_model_add(&m->fc, 2 * m->zcount, SYMBOL_ALPHABET, NULL);
    if (rc) {
        ingot_match_model_free(m);
        return rc;
    }
    *out = m;
    return 0;
}

void ingot_match_model_free(ingot_match_model *m) {
    if (!m)
        return;
    for (uint32_t s = 0; s < INGOT_MATCH_MAX_SOURCES; s++)
        if (m->owns[s])
            free(m->fixed[s]);
    free(m->thresholds);
    free(m->zbase);
    free(m);
}

const ingot_fc_model *ingot_match_fc_model(const ingot_match_model *m) { return &m->fc; }

/* Unit u's partners: fixed-point table, first value and stride, coefficient. */
typedef struct {
    const int64_t *fixed;
    const uint16_t *x;
    int64_t q;
} partner_view;

static uint32_t partner_views(const ingot_match_model *m, const uint16_t *const *sources,
                              uint64_t u, partner_view *pv) {
    const ingot_match_params *p = m->p;
    uint32_t k = p->k[u];
    for (uint32_t j = 0; j < k; j++) {
        size_t at = (size_t)u * INGOT_MATCH_MAX_K + j;
        uint64_t i = p->index[at];
        pv[j].fixed = m->fixed[p->link[at]];
        pv[j].q = p->coef[at];
        pv[j].x = sources[p->link[at]] + i * p->unit_len; /* unit-major, either axis */
    }
    return k;
}

static inline uint8_t context_at(const ingot_match_model *m, uint64_t u, const partner_view *pv,
                                 uint32_t k, uint64_t h) {
    int64_t p = 0;
    for (uint32_t j = 0; j < k; j++)
        p += pv[j].q * pv[j].fixed[pv[j].x[h]];
    int64_t mag = p < 0 ? -p : p; /* |p| < 2^58 */
    const int64_t *t = m->thresholds + u * THRESHOLDS;
    uint32_t b = 0;
    for (int j = 0; j < THRESHOLDS; j++)
        b += mag >= t[j];
    return (uint8_t)((p < 0 ? NEG_BIT : 0) | (m->zbase[u] + b));
}

void ingot_match_contexts(const ingot_match_model *m, const uint16_t *const *sources,
                          uint64_t lo, uint64_t hi, uint8_t *ctx) {
    const ingot_match_params *p = m->p;
    uint64_t units = p->units, len = p->unit_len;
    partner_view pv[INGOT_MATCH_MAX_K];
    if (lo >= hi)
        return;
    if (p->axis == INGOT_PREDICT_ROWS) {
        for (uint64_t u = lo / len; u * len < hi; u++) {
            uint64_t h0 = u * len < lo ? lo - u * len : 0;
            uint64_t h1 = hi - u * len < len ? hi - u * len : len;
            uint32_t k = partner_views(m, sources, u, pv);
            for (uint64_t h = h0; h < h1; h++)
                ctx[u * len + h - lo] = context_at(m, u, pv, k, h);
        }
        return;
    }
    uint64_t r0 = lo / units, r1 = (hi - 1) / units;
    for (uint64_t u = 0; u < units; u++) {
        uint64_t h0 = r0 + (u < lo % units), h_end = r1 + 1 - (u > (hi - 1) % units);
        uint32_t k = partner_views(m, sources, u, pv);
        for (uint64_t h = h0; h < h_end; h++)
            ctx[h * units + u - lo] = context_at(m, u, pv, k, h);
    }
}

/* ---- Side stream ---- */

int ingot_match_side_model(uint32_t source_count, ingot_fc_model *out) {
    if (source_count < 1 || source_count > INGOT_MATCH_MAX_SOURCES)
        return -INGOT_EINVAL;
    *out = (ingot_fc_model){0};
    int rc = ingot_fc_model_add(out, 1, 256, NULL);
    if (!rc)
        rc = ingot_fc_model_add(out, 1, INGOT_MATCH_MAX_K + 1, NULL);
    if (!rc)
        rc = ingot_fc_model_add(out, 1, 2 * source_count, NULL);
    if (!rc)
        rc = ingot_fc_model_add(out, source_count, INGOT_STEP_ALPHABET, NULL);
    return rc;
}

/* One partner of unit u: link symbol, index bits, coefficient. Decoding validates each value
 * (out-of-range values set the coder's error and decode as 0). */
static void side_partner(ingot_fcoder *c, ingot_match_params *p, uint64_t u, size_t at) {
    uint32_t ns = p->source_count;
    uint32_t src = p->link[at];
    int same = ingot_fc_encoding(c) && p->index[at] == u && !p->source[src].self;
    uint32_t link = ingot_fc_sym(c, CTX_LINK, 2 * src + (uint32_t)same);
    src = link / 2;
    same = (int)(link & 1);
    if (src >= ns) {
        c->err = c->err ? c->err : -INGOT_ECORRUPT;
        return;
    }
    const ingot_match_source *s = &p->source[src];
    uint64_t index = u;
    if (!same)
        index = ingot_fc_raw(c, index_bits(s->units), (uint32_t)p->index[at]);
    int32_t q = ingot_fc_step(c, CTX_COEF + src, p->coef[at]);
    if (index >= s->units || (s->self && index >= u) || q < -INGOT_MATCH_COEF_MAX ||
        q > INGOT_MATCH_COEF_MAX || (same && s->self)) {
        c->err = c->err ? c->err : -INGOT_ECORRUPT;
        return;
    }
    p->link[at] = (uint8_t)src;
    p->index[at] = index;
    p->coef[at] = (int16_t)q;
}

void ingot_match_side_code(ingot_fcoder *c, ingot_match_params *p) {
    for (uint64_t u = 0; u < p->units && !c->err; u++) {
        p->scale[u] = (int8_t)((int32_t)ingot_fc_sym(c, CTX_SCALE, (uint8_t)(p->scale[u] + 128)) -
                               128);
        uint32_t k = ingot_fc_sym(c, CTX_K, p->k[u]);
        if (k > INGOT_MATCH_MAX_K) {
            c->err = -INGOT_ECORRUPT;
            return;
        }
        p->k[u] = (uint8_t)k;
        for (uint32_t j = 0; j < k && !c->err; j++)
            side_partner(c, p, u, (size_t)u * INGOT_MATCH_MAX_K + j);
    }
}

/* ---- Prefix ---- */

uint64_t ingot_match_header_bytes(uint32_t source_count) {
    return HEADER + (uint64_t)SOURCE_RECORD * source_count;
}

static uint64_t pad8(uint64_t n) { return (n + 7) & ~(uint64_t)7; }

uint64_t ingot_match_prefix_bytes(uint32_t source_count, const ingot_fc_tables *side_tables,
                                  size_t side_block, const ingot_fc_tables *tables) {
    return pad8(pad8(pad8(ingot_match_header_bytes(source_count) +
                          ingot_fc_tables_size(side_tables)) +
                     side_block) +
                ingot_fc_tables_size(tables));
}

void ingot_match_prefix_write(const ingot_match_params *p, uint64_t block_count,
                              uint32_t contexts, const ingot_fc_tables *side_tables,
                              const ingot_block_buf *side_block, const ingot_fc_tables *tables,
                              uint8_t *out) {
    uint32_t flags = 0;
    for (uint32_t s = 0; s < p->source_count; s++)
        flags |= p->source[s].self ? INGOT_MATCH_FLAG_SELF : 0;
    flags |= p->strip_rows ? INGOT_MATCH_FLAG_STRIPS : 0;
    flags |= p->waves ? INGOT_MATCH_FLAG_WAVES : 0;
    ingot_store_u32(out, p->source_count);
    ingot_store_u32(out + 4, (uint32_t)p->axis);
    ingot_store_u64(out + 8, p->units);
    ingot_store_u64(out + 16, p->unit_len);
    ingot_store_u32(out + 24, (uint32_t)p->scale_c);
    ingot_store_u32(out + 28, INGOT_BLOCK_LOG2);
    ingot_store_u64(out + 32, block_count);
    ingot_store_u32(out + 40, p->classes);
    ingot_store_u32(out + 44, p->class_width);
    ingot_store_u32(out + 48, (uint32_t)p->class_ref);
    ingot_store_u32(out + 52, contexts);
    ingot_store_u64(out + 56, ingot_match_param_hash(p));
    ingot_store_u32(out + 64, flags);
    ingot_store_u32(out + 68, (uint32_t)p->strip_rows);
    ingot_store_u64(out + 72, ingot_fc_tables_size(side_tables));
    ingot_store_u64(out + 80, side_block->size);
    for (uint32_t s = 0; s < p->source_count; s++) {
        uint8_t *r = out + HEADER + SOURCE_RECORD * s;
        ingot_store_u32(r, p->source[s].ref);
        ingot_store_u32(r + 4, (uint32_t)p->source[s].axis);
        ingot_store_u64(r + 8, p->source[s].units);
        ingot_store_u32(r + 16, (uint32_t)p->source[s].coef_exp);
    }
    uint64_t at = ingot_match_header_bytes(p->source_count);
    at = pad8(at + ingot_fc_tables_write(side_tables, out + at));
    memcpy(out + at, side_block->data, side_block->size);
    at = pad8(at + side_block->size);
    ingot_fc_tables_write(tables, out + at);
}

/* Header fields and source records into p (per-unit arrays not touched). */
static int parse_header(const uint8_t *in, uint64_t avail, ingot_match_params *p, uint64_t *block_count, uint32_t *contexts,
                        uint64_t *side_tables, uint64_t *side_block) {
    if (avail < HEADER)
        return -INGOT_EFORMAT;
    p->source_count = ingot_load_u32(in);
    uint32_t axis = ingot_load_u32(in + 4), flags = ingot_load_u32(in + 64), self = 0;
    p->units = ingot_load_u64(in + 8);
    p->unit_len = ingot_load_u64(in + 16);
    p->scale_c = (int32_t)ingot_load_u32(in + 24);
    *block_count = ingot_load_u64(in + 32);
    p->classes = ingot_load_u32(in + 40);
    p->class_width = ingot_load_u32(in + 44);
    p->class_ref = (int32_t)ingot_load_u32(in + 48);
    *contexts = ingot_load_u32(in + 52);
    *side_tables = ingot_load_u64(in + 72);
    *side_block = ingot_load_u64(in + 80);
    p->strip_rows = ingot_load_u32(in + 68);
    p->waves = (flags & INGOT_MATCH_FLAG_WAVES) != 0;
    if (p->source_count < 1 || p->source_count > INGOT_MATCH_MAX_SOURCES || axis > 1 ||
        ingot_load_u32(in + 28) != INGOT_BLOCK_LOG2 ||
        (flags & ~(uint32_t)(INGOT_MATCH_FLAG_SELF | INGOT_MATCH_FLAG_STRIPS |
                             INGOT_MATCH_FLAG_WAVES)) ||
        !(flags & INGOT_MATCH_FLAG_STRIPS) != !p->strip_rows ||
        avail < ingot_match_header_bytes(p->source_count))
        return -INGOT_EFORMAT;
    p->axis = (ingot_predict_axis)axis;
    for (uint32_t s = 0; s < p->source_count; s++) {
        const uint8_t *r = in + HEADER + SOURCE_RECORD * s;
        ingot_match_source *src = &p->source[s];
        uint32_t sa = ingot_load_u32(r + 4);
        src->ref = ingot_load_u32(r);
        src->units = ingot_load_u64(r + 8);
        src->coef_exp = (int32_t)ingot_load_u32(r + 16);
        src->self = src->ref == INGOT_MATCH_SELF_REF;
        self |= (uint32_t)src->self;
        if (sa > 1 || ingot_load_u32(r + 20))
            return -INGOT_EFORMAT;
        src->axis = (ingot_predict_axis)sa;
    }
    if (self != (flags & INGOT_MATCH_FLAG_SELF))
        return -INGOT_EFORMAT;
    return geometry_check(p) ? -INGOT_EFORMAT : 0;
}

/* Decode the side stream (one block) into p's per-unit arrays. */
static int decode_side(const uint8_t *in, uint64_t avail, uint64_t table_bytes,
                       uint64_t block_bytes, ingot_match_params *p, uint64_t *at) {
    ingot_fc_model sm;
    ingot_fc_tables *t = NULL;
    uint64_t used = 0;
    int rc = ingot_match_side_model(p->source_count, &sm) ? -INGOT_EFORMAT : 0;
    if (!rc && (table_bytes > avail - *at))
        rc = -INGOT_EFORMAT;
    if (!rc)
        rc = ingot_fc_tables_read(&sm, in + *at, table_bytes, &t, &used);
    if (!rc && used != table_bytes)
        rc = -INGOT_EFORMAT;
    *at = pad8(*at + table_bytes);
    if (!rc && (*at > avail || block_bytes > avail - *at))
        rc = -INGOT_EFORMAT;
    uint8_t offsets[16] = {0};
    ingot_store_u64(offsets + 8, block_bytes);
    ingot_block_area area = {in + *at, block_bytes, 1, offsets};
    ingot_block_view v;
    if (!rc)
        rc = ingot_block_view_get(&area, 0, &v);
    if (!rc)
        rc = ingot_match_params_alloc(p);
    if (!rc) {
        ingot_fcoder c;
        ingot_fc_begin_decode(&c, t, &v);
        ingot_match_side_code(&c, p);
        rc = c.err ? -INGOT_ECORRUPT : ingot_fc_finish_decode(&c);
    }
    *at = pad8(*at + block_bytes);
    ingot_fc_tables_free(t);
    return rc;
}

void ingot_match_transpose(const uint16_t *in, uint64_t rows, uint64_t cols, uint16_t *out) {
    enum { TILE = 64 };
    for (uint64_t r0 = 0; r0 < rows; r0 += TILE)
        for (uint64_t c0 = 0; c0 < cols; c0 += TILE)
            for (uint64_t i = r0; i < rows && i < r0 + TILE; i++)
                for (uint64_t j = c0; j < cols && j < c0 + TILE; j++)
                    out[j * rows + i] = in[i * cols + j];
}

/* ---- Decoder ---- */

struct ingot_match_decoder {
    ingot_match_params params; /* owns the per-unit arrays */
    ingot_match_model *model;
    ingot_fc_tables *tables;
    const uint16_t *sources[INGOT_MATCH_MAX_SOURCES]; /* unit-major */
    uint16_t *owned[INGOT_MATCH_MAX_SOURCES];         /* transposed column sources */
    uint64_t count;
    size_t block_count;
    int serial;  /* blocks in order: a self source without strips */
    int rowwise; /* contexts row by row inside a block: any self source */
};

int ingot_match_decoder_open(uint64_t rows, uint64_t cols, const uint8_t *prefix, uint64_t avail,
                             ingot_match_decoder **out, uint64_t *prefix_bytes) {
    if (!prefix || !out || !prefix_bytes)
        return -INGOT_EFORMAT;
    ingot_match_decoder *d = calloc(1, sizeof(*d));
    if (!d)
        return -INGOT_ENOMEM;
    ingot_match_params *p = &d->params;
    uint64_t block_count = 0, side_tables = 0, side_block = 0, at = 0;
    uint32_t contexts = 0;
    int rc = parse_header(prefix, avail, p, &block_count, &contexts, &side_tables,
                          &side_block);
    if (!rc && (p->axis == INGOT_PREDICT_ROWS ? (rows != p->units || cols != p->unit_len)
                                              : (rows != p->unit_len || cols != p->units)))
        rc = -INGOT_EFORMAT;
    d->count = p->units * p->unit_len;
    d->block_count = rc ? 0 : ingot_match_block_count(p, NULL);
    if (!rc && block_count != d->block_count)
        rc = -INGOT_EFORMAT;
    at = ingot_match_header_bytes(p->source_count);
    if (!rc)
        rc = decode_side(prefix, avail, side_tables, side_block, p, &at);
    if (!rc && (ingot_match_params_check(p) ||
                ingot_match_param_hash(p) != ingot_load_u64(prefix + 56)))
        rc = -INGOT_ECORRUPT;
    if (!rc)
        rc = ingot_match_model_create(p, &d->model) ? -INGOT_EFORMAT : 0;
    if (!rc && contexts != ingot_match_fc_model(d->model)->contexts)
        rc = -INGOT_EFORMAT;
    uint64_t used = 0;
    if (!rc && at > avail)
        rc = -INGOT_EFORMAT;
    if (!rc)
        rc = ingot_fc_tables_read(ingot_match_fc_model(d->model), prefix + at, avail - at,
                                  &d->tables, &used);
    at = pad8(at + used);
    if (!rc && at > avail)
        rc = -INGOT_EFORMAT;
    for (uint32_t s = 0; s < p->source_count; s++)
        d->rowwise |= p->source[s].self;
    d->serial = d->rowwise && (!p->strip_rows || p->waves);
    if (rc) {
        ingot_match_decoder_close(d);
        return rc;
    }
    *prefix_bytes = at;
    *out = d;
    return 0;
}

const ingot_match_params *ingot_match_decoder_params(const ingot_match_decoder *d) {
    return &d->params;
}

int ingot_match_decoder_bind(ingot_match_decoder *d, uint32_t s, uint64_t rows, uint64_t cols,
                             const uint16_t *values) {
    if (!d || !values || s >= d->params.source_count || d->params.source[s].self)
        return -INGOT_EFORMAT;
    const ingot_match_source *src = &d->params.source[s];
    uint64_t want_rows = src->axis == INGOT_PREDICT_ROWS ? src->units : d->params.unit_len;
    uint64_t want_cols = src->axis == INGOT_PREDICT_ROWS ? d->params.unit_len : src->units;
    if (rows != want_rows || cols != want_cols)
        return -INGOT_EFORMAT;
    if (src->axis == INGOT_PREDICT_ROWS) {
        d->sources[s] = values;
        return 0;
    }
    free(d->owned[s]); /* column units: one transposed copy, read unit by unit */
    d->owned[s] = malloc((size_t)(rows * cols) * sizeof(uint16_t));
    if (!d->owned[s])
        return -INGOT_ENOMEM;
    ingot_match_transpose(values, rows, cols, d->owned[s]);
    d->sources[s] = d->owned[s];
    return 0;
}

int ingot_match_decoder_serial(const ingot_match_decoder *d) { return d->serial; }

size_t ingot_match_decoder_blocks(const ingot_match_decoder *d) { return d->block_count; }

void ingot_match_decoder_range(const ingot_match_decoder *d, size_t b, uint64_t *lo,
                               uint64_t *hi) {
    ingot_match_block_range(&d->params, b, lo, hi);
}

size_t ingot_match_block_count(const ingot_match_params *p, uint64_t *max_weights) {
    uint64_t count = p->units * p->unit_len;
    uint64_t cols = p->axis == INGOT_PREDICT_ROWS ? p->unit_len : p->units;
    uint64_t block = p->strip_rows ? p->strip_rows * cols : (uint64_t)BLOCK;
    if (max_weights)
        *max_weights = block < count ? block : count;
    return (size_t)((count + block - 1) / block);
}

void ingot_match_block_range(const ingot_match_params *p, size_t b, uint64_t *lo, uint64_t *hi) {
    uint64_t count = p->units * p->unit_len;
    uint64_t cols = p->axis == INGOT_PREDICT_ROWS ? p->unit_len : p->units;
    uint64_t block = p->strip_rows ? p->strip_rows * cols : (uint64_t)BLOCK;
    *lo = (uint64_t)b * block;
    *hi = *lo + block < count ? *lo + block : count;
}

/* Row-wise block (a self source): each row's contexts after the rows before it are decoded. */
static void decode_rows(const ingot_match_decoder *d, ingot_fcoder *c, const uint16_t **sources,
                        uint64_t lo, uint64_t hi, uint16_t *out, uint8_t *ctx) {
    uint64_t len = d->params.unit_len;
    for (uint64_t at = lo; at < hi && !c->err;) {
        uint64_t end = (at / len + 1) * len;
        end = end < hi ? end : hi;
        ingot_match_contexts(d->model, sources, at, end, ctx);
        ingot_predict_code_z(c, d->params.classes * INGOT_PREDICT_BUCKETS, ctx, NULL,
                             out + (at - lo), (size_t)(end - at));
        at = end;
    }
}

int ingot_match_decoder_block(const ingot_match_decoder *d, size_t b, const ingot_block_view *v,
                              uint16_t *out) {
    const uint16_t *sources[INGOT_MATCH_MAX_SOURCES];
    uint64_t lo = 0, hi = 0;
    ingot_match_decoder_range(d, b, &lo, &hi);
    for (uint32_t s = 0; s < d->params.source_count; s++) {
        sources[s] = d->params.source[s].self ? out - lo : d->sources[s];
        if (!sources[s])
            return -INGOT_EINVAL;
    }
    uint8_t *ctx = malloc((size_t)(hi - lo));
    if (!ctx)
        return -INGOT_ENOMEM;
    ingot_fcoder c;
    ingot_fc_begin_decode(&c, d->tables, v);
    if (d->rowwise) {
        decode_rows(d, &c, sources, lo, hi, out, ctx);
    } else {
        ingot_match_contexts(d->model, sources, lo, hi, ctx);
        ingot_predict_code_z(&c, d->params.classes * INGOT_PREDICT_BUCKETS, ctx, NULL, out,
                             (size_t)(hi - lo));
    }
    free(ctx);
    return ingot_fc_finish_decode(&c);
}

void ingot_match_decoder_close(ingot_match_decoder *d) {
    if (!d)
        return;
    ingot_fc_tables_free(d->tables);
    ingot_match_model_free(d->model);
    ingot_match_params_release(&d->params);
    for (uint32_t s = 0; s < INGOT_MATCH_MAX_SOURCES; s++)
        free(d->owned[s]);
    free(d);
}
