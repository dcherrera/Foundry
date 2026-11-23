/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_predict.c
 * @brief  Predict mode: the integer predictor, the context model, the mirrored weight coder and
 *         the block decoder (see header). The encoder side is in ingot_predict_plan.c.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_predict.h"

#include <stdlib.h>
#include <string.h>

#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"

enum {
    THRESHOLDS = INGOT_PREDICT_BUCKETS - 1,
    SYMBOL_ALPHABET = 1024,
    BLOCK = 1 << INGOT_BLOCK_LOG2,
    LOW_BITS = 5,
    NEG_BIT = 0x80,
};

/* Thresholds t_k in 1/256 units, and 2^(r/4) in 1/65536 units (r = 0..3). */
static const int64_t THRESHOLD_Q8[THRESHOLDS] = {16, 32, 64, 96, 128, 192, 256, 320, 384, 512, 768};
static const int64_t ROOT4_Q16[4] = {65536, 77936, 92682, 110218};

uint64_t ingot_predict_map(const ingot_predict_partner *p, uint64_t u) {
    return p->row_offset + ((u / p->head_dim) / p->group) * p->head_dim + u % p->head_dim;
}

/* Largest partner row any unit maps to (the map is monotone within a head, not across). */
static uint64_t max_row(const ingot_predict_partner *p, uint64_t units) {
    uint64_t best = 0;
    for (uint64_t u = 0; u < units; u++) {
        uint64_t r = ingot_predict_map(p, u);
        best = r > best ? r : best;
    }
    return best;
}

int ingot_predict_geometry_check(const ingot_predict_geometry *g) {
    if (!g || g->partner_count < 1 || g->partner_count > INGOT_PREDICT_MAX_PARTNERS ||
        !g->units || !g->unit_len || g->units > INGOT_PREDICT_LIMIT ||
        g->unit_len > INGOT_PREDICT_LIMIT ||
        (g->axis != INGOT_PREDICT_ROWS && g->axis != INGOT_PREDICT_COLS))
        return -INGOT_EINVAL;
    for (uint32_t j = 0; j < g->partner_count; j++) {
        const ingot_predict_partner *p = &g->partner[j];
        if (!p->head_dim || !p->group || p->head_dim > INGOT_PREDICT_LIMIT ||
            p->group > INGOT_PREDICT_LIMIT || p->row_offset > INGOT_PREDICT_LIMIT ||
            max_row(p, g->units) >= p->rows)
            return -INGOT_EINVAL;
    }
    return 0;
}

int64_t ingot_predict_fixed(uint16_t bf16, int32_t shift) {
    const int64_t sat = (int64_t)1 << INGOT_PREDICT_FIX_BITS;
    uint32_t exp = (bf16 >> 7) & 0xFFu, mant = bf16 & 0x7Fu;
    int64_t m = exp ? (int64_t)(128u | mant) : (int64_t)mant;
    int64_t e = (int64_t)(exp ? exp : 1u) - 134 + shift; /* value = m * 2^(e - shift) */
    int64_t v = 0; /* m < 2^8, so m << e fits int64 for e < 40 */
    if (m && e > -8)
        v = e >= INGOT_PREDICT_FIX_BITS ? sat : e >= 0 ? m << e : m >> -e;
    v = v > sat || exp == 0xFFu ? sat : v; /* inf and NaN clamp too */
    return (bf16 & 0x8000u) ? -v : v;
}

void ingot_predict_thresholds(int8_t scale, int32_t c, int64_t out[INGOT_PREDICT_BUCKETS - 1]) {
    int32_t biased = (int32_t)scale + 128; /* 0..255: floor and remainder without signed shifts */
    int64_t shift = (int64_t)(biased >> 2) - 32 + (int64_t)c - 24;
    int64_t root = ROOT4_Q16[biased & 3];
    for (int k = 0; k < THRESHOLDS; k++) {
        int64_t base = THRESHOLD_Q8[k] * root; /* 2^20 .. 2^27 */
        if (shift >= 0) /* base >= 2^20: past shift 42 it is above the clamp anyway */
            out[k] = shift > 42 || base > (INGOT_PREDICT_THRESHOLD_MAX >> shift)
                         ? INGOT_PREDICT_THRESHOLD_MAX
                         : base << shift;
        else
            out[k] = shift <= -28 ? 0 : base >> -shift;
    }
}

struct ingot_predict_model {
    const ingot_predict_params *p;
    int32_t shift[INGOT_PREDICT_MAX_PARTNERS]; /* F_j = C - E_j */
    int64_t *fixed[INGOT_PREDICT_MAX_PARTNERS]; /* [65536]: ingot_predict_fixed(bits, F_j) */
    int64_t *thresholds;                       /* [U][THRESHOLDS] */
    uint8_t *zbase;                            /* [U]: class * BUCKETS */
    uint32_t zcount;                           /* classes * BUCKETS */
    ingot_fc_model fc;
};

uint32_t ingot_predict_class(int32_t scale, uint32_t classes, uint32_t width, int32_t ref) {
    int32_t a = scale - ref, w = (int32_t)width;
    int32_t q = a / w - (a % w != 0 && a < 0); /* floor(a / w), w > 0 */
    int32_t cls = q + (int32_t)classes / 2;
    return (uint32_t)(cls < 0 ? 0 : cls >= (int32_t)classes ? (int32_t)classes - 1 : cls);
}

static int params_check(const ingot_predict_params *p) {
    const ingot_predict_geometry *g = &p->geom;
    if (!g->units || !g->unit_len || g->partner_count < 1 ||
        g->partner_count > INGOT_PREDICT_MAX_PARTNERS || !p->scale || !p->coef ||
        p->classes < 1 || p->classes > INGOT_PREDICT_MAX_CLASSES || p->class_width < 1 ||
        p->class_width > INGOT_PREDICT_MAX_CLASS_WIDTH || p->scale_c < -INGOT_PREDICT_EXP_LIMIT ||
        p->scale_c > INGOT_PREDICT_EXP_LIMIT || p->class_ref < -256 || p->class_ref > 256)
        return -INGOT_EINVAL;
    for (uint32_t j = 0; j < g->partner_count; j++)
        if (p->coef_exp[j] < -INGOT_PREDICT_EXP_LIMIT || p->coef_exp[j] > INGOT_PREDICT_EXP_LIMIT)
            return -INGOT_EINVAL;
    return 0;
}

int ingot_predict_model_create(const ingot_predict_params *p, ingot_predict_model **out) {
    if (!p || !out || params_check(p))
        return -INGOT_EINVAL;
    ingot_predict_model *m = calloc(1, sizeof(*m));
    if (!m)
        return -INGOT_ENOMEM;
    uint64_t units = p->geom.units;
    m->p = p;
    m->zcount = p->classes * INGOT_PREDICT_BUCKETS;
    m->thresholds = malloc((size_t)units * THRESHOLDS * sizeof(*m->thresholds));
    m->zbase = malloc((size_t)units);
    int rc = (m->thresholds && m->zbase) ? 0 : -INGOT_ENOMEM;
    for (uint32_t j = 0; !rc && j < p->geom.partner_count; j++) {
        m->shift[j] = p->scale_c - p->coef_exp[j];
        m->fixed[j] = malloc(65536 * sizeof(*m->fixed[j]));
        rc = m->fixed[j] ? 0 : -INGOT_ENOMEM;
        for (uint32_t b = 0; !rc && b < 65536; b++)
            m->fixed[j][b] = ingot_predict_fixed((uint16_t)b, m->shift[j]);
    }
    for (uint64_t u = 0; !rc && u < units; u++) {
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
        ingot_predict_model_free(m);
        return rc;
    }
    *out = m;
    return 0;
}

void ingot_predict_model_free(ingot_predict_model *m) {
    if (!m)
        return;
    free(m->thresholds);
    free(m->zbase);
    free(m->fixed[0]);
    free(m->fixed[1]);
    free(m);
}

const ingot_fc_model *ingot_predict_fc_model(const ingot_predict_model *m) { return &m->fc; }

/* Context byte of unit u given its partner values. */
static inline uint8_t context_of(const ingot_predict_model *m, uint64_t u, const int16_t *a,
                                 const uint16_t *const *x, uint64_t h, uint32_t partners) {
    int64_t p = (int64_t)a[0] * m->fixed[0][x[0][h]];
    if (partners > 1)
        p += (int64_t)a[1] * m->fixed[1][x[1][h]];
    int64_t mag = p < 0 ? -p : p; /* |p| < 2^57 */
    const int64_t *t = m->thresholds + u * THRESHOLDS;
    uint32_t k = 0;
    for (int j = 0; j < THRESHOLDS; j++)
        k += mag >= t[j];
    return (uint8_t)((p < 0 ? NEG_BIT : 0) | (m->zbase[u] + k));
}

/* Unit u's partner rows. */
static void partner_rows(const ingot_predict_model *m, const uint16_t *const *partners,
                         uint64_t u, const uint16_t **rows) {
    const ingot_predict_geometry *g = &m->p->geom;
    for (uint32_t j = 0; j < g->partner_count; j++)
        rows[j] = partners[j] + ingot_predict_map(&g->partner[j], u) * g->unit_len;
}

void ingot_predict_contexts(const ingot_predict_model *m, const uint16_t *const *partners,
                            uint64_t lo, uint64_t hi, uint8_t *ctx) {
    const ingot_predict_geometry *g = &m->p->geom;
    uint64_t units = g->units, len = g->unit_len;
    uint32_t np = g->partner_count;
    const uint16_t *rows[INGOT_PREDICT_MAX_PARTNERS] = {0};
    if (lo >= hi)
        return;
    if (g->axis == INGOT_PREDICT_ROWS) { /* D [U, H]: unit u is row u, contiguous */
        for (uint64_t u = lo / len; u * len < hi; u++) {
            uint64_t h0 = u * len < lo ? lo - u * len : 0;
            uint64_t h1 = hi - u * len < len ? hi - u * len : len;
            partner_rows(m, partners, u, rows);
            const int16_t *a = m->p->coef + u * np;
            for (uint64_t h = h0; h < h1; h++)
                ctx[u * len + h - lo] = context_of(m, u, a, rows, h, np);
        }
        return;
    }
    /* D [H, U]: unit u is column u. Unit-major, so each partner row is read in order. */
    uint64_t r0 = lo / units, r1 = (hi - 1) / units; /* rows of D in [lo, hi) */
    for (uint64_t u = 0; u < units; u++) {
        uint64_t h0 = r0 + (u < lo % units), h_end = r1 + 1 - (u > (hi - 1) % units);
        partner_rows(m, partners, u, rows);
        const int16_t *a = m->p->coef + u * np;
        for (uint64_t h = h0; h < h_end; h++)
            ctx[h * units + u - lo] = context_of(m, u, a, rows, h, np);
    }
}

/* Decoder state in locals (the per-symbol path re-reads the coder from memory after every
 * store; this loop is ~2x faster). The same stream as the generic loop below: the round-trip
 * tests compare the two directions on every predict payload. */
typedef struct {
    const ingot_rans_table *tables;
    const uint8_t *const *slots8;
    const uint16_t *const *slots16;
    const uint16_t *words;
    size_t wp, word_count, pos;
    uint32_t x[INGOT_RANS_LANES];
} local_decoder;

static inline int local_symbol(local_decoder *d, uint32_t cx, uint32_t *out) {
    const ingot_rans_table *t = &d->tables[cx];
    uint32_t *x = &d->x[d->pos % INGOT_RANS_LANES];
    uint32_t slot = *x & (INGOT_RANS_TOTAL - 1u);
    if (!t->used)
        return -1;
    uint32_t s = d->slots8[cx]    ? d->slots8[cx][slot]
                 : d->slots16[cx] ? d->slots16[cx][slot]
                                  : ingot_fc_search(t, slot);
    uint32_t v = t->freq[s] * (*x >> INGOT_RANS_PROB_BITS) + slot - t->start[s];
    if (v < INGOT_RANS_LOW) {
        if (d->wp >= d->word_count)
            return -1;
        v = (v << 16) | d->words[d->wp++];
    }
    *x = v;
    d->pos++;
    *out = s;
    return 0;
}

static void decode_local(ingot_fcoder *c, uint32_t zcount, const uint8_t *ctx, uint16_t *dst,
                         size_t n) {
    local_decoder d = {c->tables, c->slots8, c->slots16, c->words, c->wp, c->word_count, c->n,
                       {0}};
    memcpy(d.x, c->x, sizeof(d.x));
    ingot_bit_reader raw = c->raw_in;
    int bad = 0;
    for (size_t i = 0; !bad && i < n; i++) {
        uint32_t neg = ctx[i] >> 7, z = ctx[i] & 0x7Fu, flip = 0, sym = 0, low = 0;
        bad = local_symbol(&d, z, &flip) || local_symbol(&d, zcount + 2 * z + flip, &sym) ||
              ingot_bits_get(&raw, LOW_BITS, &low);
        dst[i] = (uint16_t)(((flip ^ neg) << 15) | (sym << 5) | low);
    }
    memcpy(c->x, d.x, sizeof(d.x));
    c->wp = d.wp;
    c->n = d.pos;
    c->raw_in = raw;
    if (bad)
        c->err = -INGOT_ECORRUPT;
}

void ingot_predict_code(ingot_fcoder *c, const ingot_predict_model *m, const uint8_t *ctx,
                        const uint16_t *src, uint16_t *dst, size_t n) {
    uint32_t zcount = m->zcount;
    if (c->mode == INGOT_FC_DECODE && !c->err && dst) {
        decode_local(c, zcount, ctx, dst, n);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        uint32_t neg = ctx[i] >> 7, z = ctx[i] & 0x7Fu;
        uint16_t w = src ? src[i] : 0;
        uint32_t flip = ingot_fc_sym(c, z, (uint32_t)(w >> 15) ^ neg);
        uint32_t sym = ingot_fc_sym(c, zcount + 2 * z + flip, ingot_bf16_symbol(w));
        uint32_t low = ingot_fc_raw(c, LOW_BITS, w & 0x1Fu);
        if (dst)
            dst[i] = (uint16_t)(((flip ^ neg) << 15) | (sym << 5) | low);
    }
}

static int prefix_parse(const uint8_t *in, uint64_t avail, ingot_predict_params *p,
                        uint64_t *block_count, uint32_t *contexts, uint64_t *at);

struct ingot_predict_decoder {
    ingot_predict_params params; /* owns scale and coef */
    ingot_predict_model *model;
    ingot_fc_tables *tables;
    const uint16_t *partners[INGOT_PREDICT_MAX_PARTNERS];
    uint64_t max_row[INGOT_PREDICT_MAX_PARTNERS];
    uint64_t count;
    size_t block_count;
};

int ingot_predict_decoder_open(uint64_t rows, uint64_t cols, const uint8_t *prefix,
                               uint64_t avail, ingot_predict_decoder **out,
                               uint64_t *prefix_bytes) {
    if (!prefix || !out || !prefix_bytes)
        return -INGOT_EFORMAT;
    ingot_predict_decoder *d = calloc(1, sizeof(*d));
    if (!d)
        return -INGOT_ENOMEM;
    uint64_t block_count = 0, at = 0;
    uint32_t contexts = 0;
    int rc = prefix_parse(prefix, avail, &d->params, &block_count, &contexts, &at);
    const ingot_predict_geometry *g = &d->params.geom;
    d->count = g->units * g->unit_len;
    if (!rc && (g->axis == INGOT_PREDICT_ROWS ? (rows != g->units || cols != g->unit_len)
                                              : (rows != g->unit_len || cols != g->units)))
        rc = -INGOT_EFORMAT;
    d->block_count = (size_t)((d->count + BLOCK - 1) / BLOCK);
    if (!rc && block_count != d->block_count)
        rc = -INGOT_EFORMAT;
    if (!rc)
        rc = ingot_predict_model_create(&d->params, &d->model) ? -INGOT_EFORMAT : 0;
    if (!rc && contexts != ingot_predict_fc_model(d->model)->contexts)
        rc = -INGOT_EFORMAT;
    uint64_t used = 0;
    if (!rc)
        rc = ingot_fc_tables_read(ingot_predict_fc_model(d->model), prefix + at, avail - at,
                                  &d->tables, &used);
    at = (at + used + 7) & ~(uint64_t)7;
    if (!rc && at > avail)
        rc = -INGOT_EFORMAT;
    for (uint32_t j = 0; !rc && j < g->partner_count; j++)
        d->max_row[j] = max_row(&g->partner[j], g->units);
    if (rc) {
        ingot_predict_decoder_close(d);
        return rc;
    }
    *prefix_bytes = at;
    *out = d;
    return 0;
}

const ingot_predict_geometry *ingot_predict_decoder_geometry(const ingot_predict_decoder *d) {
    return &d->params.geom;
}

int ingot_predict_decoder_bind(ingot_predict_decoder *d, uint32_t j, uint64_t rows,
                               uint64_t cols, const uint16_t *values) {
    if (!d || !values || j >= d->params.geom.partner_count || cols != d->params.geom.unit_len ||
        rows <= d->max_row[j])
        return -INGOT_EFORMAT;
    d->partners[j] = values;
    d->params.geom.partner[j].rows = rows;
    return 0;
}

size_t ingot_predict_decoder_blocks(const ingot_predict_decoder *d) { return d->block_count; }

void ingot_predict_decoder_range(const ingot_predict_decoder *d, size_t b, uint64_t *lo,
                                 uint64_t *hi) {
    *lo = (uint64_t)b * BLOCK;
    *hi = *lo + BLOCK < d->count ? *lo + BLOCK : d->count;
}

int ingot_predict_decoder_block(const ingot_predict_decoder *d, size_t b,
                                const ingot_block_view *v, uint16_t *out) {
    for (uint32_t j = 0; j < d->params.geom.partner_count; j++)
        if (!d->partners[j])
            return -INGOT_EINVAL;
    uint64_t lo = 0, hi = 0;
    ingot_predict_decoder_range(d, b, &lo, &hi);
    uint8_t *ctx = malloc((size_t)(hi - lo));
    if (!ctx)
        return -INGOT_ENOMEM;
    ingot_predict_contexts(d->model, d->partners, lo, hi, ctx);
    ingot_fcoder c;
    ingot_fc_begin_decode(&c, d->tables, v);
    ingot_predict_code(&c, d->model, ctx, NULL, out, (size_t)(hi - lo));
    free(ctx);
    return ingot_fc_finish_decode(&c);
}

void ingot_predict_decoder_close(ingot_predict_decoder *d) {
    if (!d)
        return;
    ingot_fc_tables_free(d->tables);
    ingot_predict_model_free(d->model);
    free(d->params.scale);
    free(d->params.coef);
    free(d);
}

/* Prefix layout (see header): fixed header, partner records, scales, coefficients. */
enum { PREFIX_HEADER = 64, PARTNER_RECORD = 40 };

static uint64_t scale_at(uint32_t partners) { return PREFIX_HEADER + PARTNER_RECORD * partners; }

static uint64_t coef_at(const ingot_predict_geometry *g) {
    return (scale_at(g->partner_count) + g->units + 1) & ~(uint64_t)1;
}

uint64_t ingot_predict_params_bytes(const ingot_predict_geometry *g) {
    return (coef_at(g) + 2 * g->units * g->partner_count + 7) & ~(uint64_t)7;
}

static uint64_t param_hash(const uint8_t *prefix, const ingot_predict_geometry *g) {
    uint64_t coef_bytes = 2 * g->units * g->partner_count;
    ingot_hash64_state h;
    ingot_hash64_begin(&h, g->units + coef_bytes);
    ingot_hash64_update(&h, prefix + scale_at(g->partner_count), (size_t)g->units);
    ingot_hash64_update(&h, prefix + coef_at(g), (size_t)coef_bytes);
    return ingot_hash64_end(&h);
}

void ingot_predict_prefix_write(const ingot_predict_params *p, uint64_t block_count,
                                uint32_t contexts, const ingot_fc_tables *t, uint8_t *out) {
    const ingot_predict_geometry *g = &p->geom;
    ingot_store_u32(out, g->partner_count);
    ingot_store_u32(out + 4, (uint32_t)g->axis);
    ingot_store_u64(out + 8, g->units);
    ingot_store_u64(out + 16, g->unit_len);
    ingot_store_u32(out + 24, (uint32_t)p->scale_c);
    ingot_store_u32(out + 28, INGOT_BLOCK_LOG2);
    ingot_store_u64(out + 32, block_count);
    ingot_store_u32(out + 40, p->classes);
    ingot_store_u32(out + 44, p->class_width);
    ingot_store_u32(out + 48, (uint32_t)p->class_ref);
    ingot_store_u32(out + 52, contexts);
    for (uint32_t j = 0; j < g->partner_count; j++) {
        uint8_t *r = out + PREFIX_HEADER + PARTNER_RECORD * j;
        ingot_store_u32(r, g->partner[j].entry);
        ingot_store_u64(r + 8, g->partner[j].row_offset);
        ingot_store_u64(r + 16, g->partner[j].head_dim);
        ingot_store_u64(r + 24, g->partner[j].group);
        ingot_store_u32(r + 32, (uint32_t)p->coef_exp[j]);
    }
    memcpy(out + scale_at(g->partner_count), p->scale, (size_t)g->units);
    uint8_t *c = out + coef_at(g);
    for (uint64_t i = 0; i < g->units * g->partner_count; i++) {
        c[2 * i] = (uint8_t)((uint16_t)p->coef[i] & 0xFFu);
        c[2 * i + 1] = (uint8_t)((uint16_t)p->coef[i] >> 8);
    }
    ingot_store_u64(out + 56, param_hash(out, g));
    ingot_fc_tables_write(t, out + ingot_predict_params_bytes(g));
}

/* Header and partner records; the geometry's partner rows stay 0 until bound. */
static int parse_header(const uint8_t *in, uint64_t avail, ingot_predict_params *p,
                        uint64_t *block_count, uint32_t *contexts) {
    ingot_predict_geometry *g = &p->geom;
    if (avail < PREFIX_HEADER)
        return -INGOT_EFORMAT;
    g->partner_count = ingot_load_u32(in);
    uint32_t axis = ingot_load_u32(in + 4);
    g->units = ingot_load_u64(in + 8);
    g->unit_len = ingot_load_u64(in + 16);
    p->scale_c = (int32_t)ingot_load_u32(in + 24);
    *block_count = ingot_load_u64(in + 32);
    p->classes = ingot_load_u32(in + 40);
    p->class_width = ingot_load_u32(in + 44);
    p->class_ref = (int32_t)ingot_load_u32(in + 48);
    *contexts = ingot_load_u32(in + 52);
    if (g->partner_count < 1 || g->partner_count > INGOT_PREDICT_MAX_PARTNERS || axis > 1 ||
        !g->units || !g->unit_len || g->units > INGOT_PREDICT_LIMIT ||
        g->unit_len > INGOT_PREDICT_LIMIT || ingot_load_u32(in + 28) != INGOT_BLOCK_LOG2 ||
        avail < ingot_predict_params_bytes(g))
        return -INGOT_EFORMAT;
    g->axis = (ingot_predict_axis)axis;
    for (uint32_t j = 0; j < g->partner_count; j++) {
        const uint8_t *r = in + PREFIX_HEADER + PARTNER_RECORD * j;
        ingot_predict_partner *q = &g->partner[j];
        q->entry = ingot_load_u32(r);
        q->row_offset = ingot_load_u64(r + 8);
        q->head_dim = ingot_load_u64(r + 16);
        q->group = ingot_load_u64(r + 24);
        p->coef_exp[j] = (int32_t)ingot_load_u32(r + 32);
        if (ingot_load_u32(r + 4) || ingot_load_u32(r + 36) || !q->head_dim || !q->group ||
            q->head_dim > INGOT_PREDICT_LIMIT || q->group > INGOT_PREDICT_LIMIT ||
            q->row_offset > INGOT_PREDICT_LIMIT)
            return -INGOT_EFORMAT;
    }
    return 0;
}

/* Parse the prefix up to the tables into p (allocating scale and coef); *at = table offset. */
static int prefix_parse(const uint8_t *in, uint64_t avail, ingot_predict_params *p,
                        uint64_t *block_count, uint32_t *contexts, uint64_t *at) {
    int rc = parse_header(in, avail, p, block_count, contexts);
    if (rc)
        return rc;
    const ingot_predict_geometry *g = &p->geom;
    if (param_hash(in, g) != ingot_load_u64(in + 56))
        return -INGOT_ECORRUPT;
    uint64_t n = g->units * g->partner_count;
    p->scale = malloc((size_t)g->units);
    p->coef = malloc((size_t)n * sizeof(*p->coef));
    if (!p->scale || !p->coef)
        return -INGOT_ENOMEM;
    memcpy(p->scale, in + scale_at(g->partner_count), (size_t)g->units);
    const uint8_t *c = in + coef_at(g);
    for (uint64_t i = 0; i < n; i++)
        p->coef[i] = (int16_t)(uint16_t)(c[2 * i] | (c[2 * i + 1] << 8));
    *at = ingot_predict_params_bytes(g);
    return 0;
}
