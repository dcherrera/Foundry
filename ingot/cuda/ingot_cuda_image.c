/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cuda_image.c
 * @brief  Device image builder for BF16 payloads (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_cuda_image.h"

#include <stdlib.h>
#include <string.h>

#include "ingot/codec/ingot_block.h"
#include "ingot/codec/ingot_match.h"
#include "ingot/codec/ingot_predict.h"
#include "ingot/codec/ingot_tensor_codec.h"
#include "ingot/core/ingot_error.h"

enum {
    SECTION_ALIGN = 256,
    AREA_SLACK = 16,          /* raw reads may run 8 bytes past a block's last raw word */
    FRAME = INGOT_PAYLOAD_FRAME_BYTES,
    PLAIN_HEADER = 24,
    DELTA_HEADER = 32,
    DELTA_CONTEXTS = 30,
    DELTA_FALLBACK = 29,
    PREDICT_HEADER = 64,
    PREDICT_PARTNER = 40,
    MATCH_SIDE_TABLES_AT = 72,
    MATCH_SIDE_BLOCK_AT = 80,
    SYMBOL_ALPHABET = 1024,
    STEP_ALPHABET = 119,
    Q8_HEADER = 48,
    DTYPE_Q8_0 = INGOT_DTYPE_GGML + 8,
};

static uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

size_t ingot_cuda_table_image_bytes(const ingot_rans_table *t) {
    return INGOT_CUDA_TABLE_HEADER + 8u * t->used;
}

void ingot_cuda_table_image_write(const ingot_rans_table *t, uint8_t *dst) {
    uint16_t *coarse = (uint16_t *)dst;
    uint64_t *entry = (uint64_t *)(dst + INGOT_CUDA_TABLE_HEADER);
    uint32_t rank = 0, cell = 0;
    for (uint32_t s = 0; s < t->alphabet; s++) {
        if (!t->freq[s])
            continue;
        entry[rank] = (uint64_t)t->start[s] | (uint64_t)(t->freq[s] - 1u) << 16 |
                      (uint64_t)s << 32;
        /* every cell whose first slot falls in this symbol's interval starts at this rank */
        while (cell < INGOT_CUDA_COARSE_CELLS &&
               ((uint32_t)cell << INGOT_CUDA_COARSE_SHIFT) < t->start[s] + t->freq[s])
            coarse[cell++] = (uint16_t)rank;
        rank++;
    }
}

/* ---- Tables --------------------------------------------------------------------------------- */

/** Parsed rANS tables of one payload. */
typedef struct {
    uint32_t count;
    ingot_rans_table *t;
} table_list;

/* Read `count` consecutive tables at src; alphabet_of(i) gives each context's alphabet. */
static int tables_read(table_list *tl, uint32_t count, const uint32_t *alphabets,
                       const uint8_t *src, uint64_t avail) {
    tl->count = count;
    tl->t = calloc(count ? count : 1, sizeof(*tl->t));
    if (!tl->t)
        return -INGOT_ENOMEM;
    uint64_t at = 0;
    for (uint32_t i = 0; i < count; i++) {
        size_t used = 0;
        int rc = ingot_rans_table_read(&tl->t[i], alphabets[i], src + at, avail - at, &used);
        if (rc)
            return rc;
        at += used;
    }
    return 0;
}

/* Largest table image (the plain kernel copies this many bytes from any table's start). */
static uint64_t tables_max_bytes(const table_list *tl) {
    uint64_t m = 0;
    for (uint32_t i = 0; i < tl->count; i++) {
        uint64_t n = tl->t[i].used ? ingot_cuda_table_image_bytes(&tl->t[i]) : 0;
        m = n > m ? n : m;
    }
    return m;
}

/* Pool bytes, with the largest image again as slack after the last table. */
static uint64_t tables_image_bytes(const table_list *tl) {
    uint64_t n = 4u * tl->count;
    n = align_up(n, 8);
    for (uint32_t i = 0; i < tl->count; i++)
        n += tl->t[i].used ? ingot_cuda_table_image_bytes(&tl->t[i]) : 0;
    return n + tables_max_bytes(tl);
}

/* Offsets (u32 per context, relative to the pool) then the lookup images. */
static void tables_image_write(const table_list *tl, uint8_t *offsets, uint8_t *pool) {
    uint64_t at = 0;
    for (uint32_t i = 0; i < tl->count; i++) {
        uint32_t off = INGOT_CUDA_TABLE_EMPTY;
        if (tl->t[i].used) {
            off = (uint32_t)at;
            ingot_cuda_table_image_write(&tl->t[i], pool + at);
            at += ingot_cuda_table_image_bytes(&tl->t[i]);
        }
        memcpy(offsets + 4u * i, &off, 4);
    }
}

/* ---- Image assembly ------------------------------------------------------------------------- */

/** Sizes of the sections before allocation. */
typedef struct {
    uint64_t area, offsets, pool, kind_a, kind_b, kind_c;
} section_sizes;

/* Lay out the sections: area, table offsets, table pool, then up to three kind sections. */
static int image_alloc(ingot_cuda_image *img, const section_sizes *s, uint64_t *at) {
    const uint64_t sizes[6] = {s->area + AREA_SLACK, s->offsets, s->pool,
                               s->kind_a, s->kind_b, s->kind_c};
    uint64_t total = 0;
    for (int i = 0; i < 6; i++) {
        at[i] = total;
        total = align_up(total + sizes[i], SECTION_ALIGN);
    }
    img->image = calloc(1, (size_t)total);
    img->image_bytes = (size_t)total;
    return img->image ? 0 : -INGOT_ENOMEM;
}

/* Common part: copy the block area, write the tables, set the shared params. */
static int image_common(ingot_cuda_image *img, const uint8_t *payload, uint64_t size,
                        const table_list *tl, const section_sizes *s, uint64_t *at) {
    uint64_t prefix = ingot_load_u64(payload);
    int rc = image_alloc(img, s, at);
    if (rc)
        return rc;
    memcpy(img->image + at[0], payload + prefix, (size_t)(size - prefix));
    tables_image_write(tl, img->image + at[1], img->image + at[2]);
    img->params.area = at[0];
    img->params.table_offsets = at[1];
    img->params.tables = at[2];
    img->params.block_count = ingot_load_u64(payload + 8);
    img->params.launch_blocks = img->params.block_count;
    img->params.block_weights = (uint64_t)1 << INGOT_BLOCK_LOG2;
    img->params.table_slot_bytes = (uint32_t)tables_max_bytes(tl);
    img->table_bytes = s->offsets + s->pool;
    return 0;
}

static void tables_free(table_list *tl) {
    free(tl->t);
    tl->t = NULL;
}

/* ---- Plain (mode 1) ------------------------------------------------------------------------- */

static int build_plain(ingot_cuda_image *img, const uint8_t *payload, uint64_t size) {
    const uint8_t *p = payload + FRAME;
    uint32_t groups = ingot_load_u32(p);
    uint32_t *alpha = malloc(sizeof(uint32_t) * (groups ? groups : 1));
    if (!alpha)
        return -INGOT_ENOMEM;
    for (uint32_t g = 0; g < groups; g++)
        alpha[g] = SYMBOL_ALPHABET;
    table_list tl = {0};
    int rc = tables_read(&tl, groups, alpha, p + PLAIN_HEADER, size - FRAME - PLAIN_HEADER);
    free(alpha);
    section_sizes s = {size - ingot_load_u64(payload), 4u * groups, tables_image_bytes(&tl), 0, 0,
                       0};
    uint64_t at[6];
    if (!rc)
        rc = image_common(img, payload, size, &tl, &s, at);
    tables_free(&tl);
    if (rc)
        return rc;
    img->params.kind = INGOT_CUDA_KIND_PLAIN;
    img->params.group_weights = ingot_load_u64(p + 8);
    img->params.blocks_per_group = img->params.block_count / groups;
    return 0;
}

/* ---- Delta (mode 2) ------------------------------------------------------------------------- */

static int build_delta(ingot_cuda_image *img, const uint8_t *payload, uint64_t size) {
    const uint8_t *p = payload + FRAME;
    uint64_t rows = ingot_load_u64(p + 8);
    uint64_t tables_at = align_up(DELTA_HEADER + rows, 8);
    uint32_t alpha[DELTA_CONTEXTS];
    for (int c = 0; c < DELTA_CONTEXTS; c++)
        alpha[c] = c == DELTA_FALLBACK ? SYMBOL_ALPHABET : STEP_ALPHABET;
    table_list tl = {0};
    int rc = tables_read(&tl, DELTA_CONTEXTS, alpha, p + tables_at, size - FRAME - tables_at);
    section_sizes s = {size - ingot_load_u64(payload), 4u * DELTA_CONTEXTS,
                       tables_image_bytes(&tl), rows, 0, 0};
    uint64_t at[6];
    if (!rc)
        rc = image_common(img, payload, size, &tl, &s, at);
    tables_free(&tl);
    if (rc)
        return rc;
    memcpy(img->image + at[3], p + DELTA_HEADER, (size_t)rows);
    img->params.kind = INGOT_CUDA_KIND_DELTA;
    img->params.row_scale = at[3];
    img->params.group_weights = img->params.rows * img->params.cols;
    img->params.blocks_per_group = img->params.block_count;
    return 0;
}

/* ---- Q8_0 (mode 3) -------------------------------------------------------------------------- */

static int build_q8(ingot_cuda_image *img, const uint8_t *payload, uint64_t size) {
    const uint8_t *p = payload + FRAME;
    static const uint32_t alpha[INGOT_CUDA_Q8_CONTEXTS] = {
        STEP_ALPHABET + 1, STEP_ALPHABET + 1, STEP_ALPHABET, INGOT_CUDA_Q8_BLOCK + 1,
        256, 256, 256, 256, 256, 256, 256, 256};
    if (ingot_load_u32(p + 4) || ingot_load_u32(p + 40) != INGOT_CUDA_Q8_CONTEXTS)
        return -INGOT_EUNSUPPORTED; /* a base (mode 3 with flags) is not decoded on the device */
    table_list tl = {0};
    int rc = tables_read(&tl, INGOT_CUDA_Q8_CONTEXTS, alpha, p + Q8_HEADER,
                         size - FRAME - Q8_HEADER);
    section_sizes s = {size - ingot_load_u64(payload), 4u * INGOT_CUDA_Q8_CONTEXTS,
                       tables_image_bytes(&tl), 0, 0, 0};
    uint64_t at[6];
    if (!rc)
        rc = image_common(img, payload, size, &tl, &s, at);
    tables_free(&tl);
    if (rc)
        return rc;
    img->params.kind = INGOT_CUDA_KIND_Q8;
    img->params.block_weights = ingot_load_u64(p + 24) * img->params.cols;
    img->params.group_weights = img->params.rows * img->params.cols;
    img->params.blocks_per_group = img->params.block_count;
    return 0;
}

/* ---- Predictor layouts (modes 6 and 7) ------------------------------------------------------ */

/** Per-unit predictor values gathered from either mode before the image is laid out. */
typedef struct {
    uint64_t units, unit_len;
    uint32_t axis, classes, class_width, source_count;
    int32_t class_ref, scale_c;
    const int8_t *scale;      /* [U] */
    uint32_t *k;              /* [U] */
    uint64_t total;           /* partners */
    uint64_t strip_rows;      /* mode 7 strips: R-row blocks (0: 2^18-weight blocks) */
    uint64_t *partner;        /* [U * 8]: index | (u16)q << 32 | source << 48 (slot k of u) */
} unit_table;

/* Unit records, partner offsets and packed partners into the three kind sections. */
static void units_write(const unit_table *ut, uint8_t *rec_out, uint8_t *off_out,
                        uint8_t *part_out) {
    uint64_t at = 0;
    for (uint64_t u = 0; u < ut->units; u++) {
        int32_t s = ut->scale[u];
        uint32_t zbase = 12u * ingot_predict_class(s, ut->classes, ut->class_width, ut->class_ref);
        uint32_t rec = ingot_cuda_unit_record(zbase, ingot_cuda_unit_rsel(s),
                                              ingot_cuda_unit_shift(s, ut->scale_c), ut->k[u]);
        uint32_t off = (uint32_t)at;
        memcpy(rec_out + 4 * u, &rec, 4);
        memcpy(off_out + 4 * u, &off, 4);
        memcpy(part_out + 8 * at, ut->partner + u * INGOT_CUDA_MAX_K, 8u * ut->k[u]);
        at += ut->k[u];
    }
    uint32_t end = (uint32_t)at;
    memcpy(off_out + 4 * ut->units, &end, 4);
}

static uint64_t pack_partner(uint64_t index, int16_t q, uint32_t source) {
    return index | (uint64_t)(uint16_t)q << 32 | (uint64_t)source << 48;
}

/* Shared tail of modes 6 and 7: tables at `tables`, then the unit sections. */
static int build_units(ingot_cuda_image *img, const uint8_t *payload, uint64_t size,
                       const unit_table *ut, const uint8_t *tables, uint64_t tables_avail) {
    uint32_t zc = ut->classes * 12u, contexts = 3 * zc;
    uint32_t alpha[3 * 12 * INGOT_PREDICT_MAX_CLASSES];
    for (uint32_t c = 0; c < contexts; c++)
        alpha[c] = c < zc ? 2u : SYMBOL_ALPHABET;
    table_list tl = {0};
    int rc = tables_read(&tl, contexts, alpha, tables, tables_avail);
    section_sizes s = {size - ingot_load_u64(payload), 4u * contexts, tables_image_bytes(&tl),
                       4 * ut->units, 4 * (ut->units + 1), 8 * (ut->total ? ut->total : 1)};
    uint64_t at[6];
    if (!rc)
        rc = image_common(img, payload, size, &tl, &s, at);
    tables_free(&tl);
    if (rc)
        return rc;
    units_write(ut, img->image + at[3], img->image + at[4], img->image + at[5]);
    ingot_cuda_params *p = &img->params;
    p->kind = INGOT_CUDA_KIND_MATCH;
    p->units = ut->units;
    p->unit_len = ut->unit_len;
    p->axis = ut->axis;
    p->zcount = zc;
    p->unit_records = at[3];
    p->partner_offsets = at[4];
    p->partners = at[5];
    p->source_count = ut->source_count;
    p->group_weights = p->rows * p->cols;
    if (ut->strip_rows)
        p->block_weights = ut->strip_rows * p->cols;
    p->blocks_per_group = p->block_count;
    img->side_bytes = s.kind_a + s.kind_b + s.kind_c;
    return 0;
}

/* Mode 6: header, partner records, scales and coefficients (layout in ingot_predict.h). */
static int build_predict(ingot_cuda_image *img, const uint8_t *payload, uint64_t size) {
    const uint8_t *p = payload + FRAME;
    uint32_t J = ingot_load_u32(p);
    unit_table ut = {.units = ingot_load_u64(p + 8), .unit_len = ingot_load_u64(p + 16),
                     .axis = ingot_load_u32(p + 4), .scale_c = (int32_t)ingot_load_u32(p + 24),
                     .classes = ingot_load_u32(p + 40), .class_width = ingot_load_u32(p + 44),
                     .class_ref = (int32_t)ingot_load_u32(p + 48), .source_count = J};
    uint64_t scale_at = PREDICT_HEADER + (uint64_t)PREDICT_PARTNER * J;
    uint64_t coef_at = align_up(scale_at + ut.units, 2);
    uint64_t tables_at = align_up(coef_at + 2 * ut.units * J, 8);
    ut.scale = (const int8_t *)(p + scale_at);
    ut.k = malloc(sizeof(uint32_t) * ut.units);
    ut.partner = malloc(sizeof(uint64_t) * ut.units * INGOT_CUDA_MAX_K);
    if (!ut.k || !ut.partner) {
        free(ut.k);
        free(ut.partner);
        return -INGOT_ENOMEM;
    }
    ingot_predict_partner part[INGOT_PREDICT_MAX_PARTNERS];
    for (uint32_t j = 0; j < J; j++) {
        const uint8_t *r = p + PREDICT_HEADER + PREDICT_PARTNER * j;
        part[j] = (ingot_predict_partner){ingot_load_u32(r), 0, ingot_load_u64(r + 8),
                                          ingot_load_u64(r + 16), ingot_load_u64(r + 24)};
        int32_t E = (int32_t)ingot_load_u32(r + 32);
        img->params.source[j].shift = ut.scale_c - E;
        img->shape[j] = (ingot_cuda_source_shape){part[j].entry, 0, 0, 0, 0};
    }
    for (uint64_t u = 0; u < ut.units; u++) {
        ut.k[u] = J;
        for (uint32_t j = 0; j < J; j++) {
            uint64_t row = ingot_predict_map(&part[j], u);
            const uint8_t *c = p + coef_at + 2 * (u * J + j);
            int16_t q = (int16_t)(uint16_t)(c[0] | c[1] << 8);
            ut.partner[u * INGOT_CUDA_MAX_K + j] = pack_partner(row, q, j);
            img->shape[j].min_rows = row + 1 > img->shape[j].min_rows ? row + 1
                                                                      : img->shape[j].min_rows;
        }
    }
    ut.total = ut.units * J;
    int rc = build_units(img, payload, size, &ut, p + tables_at, size - FRAME - tables_at);
    free(ut.k);
    free(ut.partner);
    return rc;
}

/* Mode 7: the codec's decoder decodes and checks the side stream; its params give the units. */
static int build_match(ingot_cuda_image *img, const uint8_t *payload, uint64_t size) {
    const uint8_t *p = payload + FRAME;
    ingot_match_decoder *d = NULL;
    uint64_t prefix_bytes = 0;
    int rc = ingot_match_decoder_open(img->params.rows, img->params.cols, p, size - FRAME, &d,
                                      &prefix_bytes);
    if (rc)
        return rc;
    const ingot_match_params *mp = ingot_match_decoder_params(d);
    unit_table ut = {.units = mp->units, .unit_len = mp->unit_len, .axis = mp->axis,
                     .classes = mp->classes, .class_width = mp->class_width,
                     .class_ref = mp->class_ref, .scale_c = mp->scale_c,
                     .source_count = mp->source_count, .scale = mp->scale,
                     .strip_rows = mp->strip_rows};
    ut.k = malloc(sizeof(uint32_t) * ut.units);
    ut.partner = malloc(sizeof(uint64_t) * ut.units * INGOT_CUDA_MAX_K);
    rc = ut.k && ut.partner ? 0 : -INGOT_ENOMEM;
    for (uint64_t u = 0; !rc && u < ut.units; u++) {
        ut.k[u] = mp->k[u];
        ut.total += mp->k[u];
        for (uint32_t j = 0; j < mp->k[u]; j++) {
            size_t at = (size_t)u * INGOT_MATCH_MAX_K + j;
            ut.partner[u * INGOT_CUDA_MAX_K + j] =
                pack_partner(mp->index[at], mp->coef[at], mp->link[at]);
        }
    }
    for (uint32_t s = 0; !rc && s < mp->source_count; s++) {
        const ingot_match_source *src = &mp->source[s];
        img->params.source[s].shift = mp->scale_c - src->coef_exp;
        img->params.source[s].self = (uint32_t)src->self;
        img->shape[s] = (ingot_cuda_source_shape){src->ref, (uint32_t)src->axis, src->units, 0,
                                                  src->self};
        img->serial |= src->self && !mp->strip_rows; /* strips: blocks independent */
        img->self_source |= src->self;
    }
    uint64_t at = ingot_match_header_bytes(mp->source_count);
    at = align_up(at + ingot_load_u64(p + MATCH_SIDE_TABLES_AT), 8);
    at = align_up(at + ingot_load_u64(p + MATCH_SIDE_BLOCK_AT), 8);
    if (!rc)
        rc = build_units(img, payload, size, &ut, p + at, size - FRAME - at);
    free(ut.k);
    free(ut.partner);
    ingot_match_decoder_close(d);
    return rc;
}

/* ---- Public --------------------------------------------------------------------------------- */

/* The codec's own decoder must accept the payload (format, frame and block area checks). */
static int validate_with_codec(const ingot_entry *e, const uint8_t *payload, uint64_t size) {
    ingot_tensor_decoder *d = NULL;
    uint64_t at = 0;
    if (size < FRAME)
        return -INGOT_EFORMAT;
    int rc = ingot_tensor_decoder_open(e, payload + FRAME, size - FRAME, &d, &at);
    if (!rc && (ingot_load_u64(payload) != FRAME + at ||
                ingot_load_u64(payload + 8) != ingot_tensor_decoder_blocks(d)))
        rc = -INGOT_EFORMAT;
    ingot_block_area area;
    if (!rc)
        rc = ingot_block_area_open(payload + FRAME + at, size - FRAME - at,
                                   ingot_tensor_decoder_blocks(d), &area);
    for (size_t b = 0; !rc && b < area.block_count; b++) { /* the kernels trust these sizes */
        ingot_block_view v;
        rc = ingot_block_view_get(&area, b, &v);
    }
    ingot_tensor_decoder_close(d);
    return rc;
}

int ingot_cuda_image_build(const ingot_entry *e, const uint8_t *payload, uint64_t size,
                           ingot_cuda_image *out) {
    if (!e || !payload || !out)
        return -INGOT_EINVAL;
    memset(out, 0, sizeof(*out));
    uint64_t rows = 0, cols = 0;
    ingot_entry_rows_cols(e, &rows, &cols);
    const int bf16 = e->dtype == INGOT_DTYPE_BF16 &&
                     (e->mode == INGOT_MODE_PLAIN || e->mode == INGOT_MODE_DELTA ||
                      e->mode == INGOT_MODE_PREDICT || e->mode == INGOT_MODE_MATCH);
    const int q8 = e->dtype == DTYPE_Q8_0 && e->mode == INGOT_MODE_QUANT;
    if ((!bf16 && !q8) || rows * cols >= ((uint64_t)1 << 32))
        return -INGOT_EUNSUPPORTED;
    int rc = validate_with_codec(e, payload, size);
    if (rc)
        return rc;
    out->params.rows = rows;
    out->params.cols = cols;
    switch (e->mode) {
    case INGOT_MODE_PLAIN:
        rc = build_plain(out, payload, size);
        break;
    case INGOT_MODE_DELTA:
        rc = build_delta(out, payload, size);
        break;
    case INGOT_MODE_PREDICT:
        rc = build_predict(out, payload, size);
        break;
    case INGOT_MODE_QUANT:
        rc = build_q8(out, payload, size);
        break;
    default:
        rc = build_match(out, payload, size);
        break;
    }
    if (rc)
        ingot_cuda_image_free(out);
    return rc;
}

void ingot_cuda_image_free(ingot_cuda_image *img) {
    if (!img)
        return;
    free(img->image);
    memset(img, 0, sizeof(*img));
}

void ingot_cuda_image_params(const ingot_cuda_image *img, uint64_t base, ingot_cuda_params *out) {
    *out = img->params;
    out->area += base;
    out->tables += base;
    out->table_offsets += base;
    if (out->kind == INGOT_CUDA_KIND_DELTA)
        out->row_scale += base;
    if (out->kind == INGOT_CUDA_KIND_MATCH) {
        out->unit_records += base;
        out->partner_offsets += base;
        out->partners += base;
    }
}

int ingot_cuda_bind_source(const ingot_cuda_image *img, ingot_cuda_params *p, uint32_t s,
                           uint64_t address, uint64_t rows, uint64_t cols) {
    if (p->kind != INGOT_CUDA_KIND_MATCH || s >= p->source_count || img->shape[s].self)
        return -INGOT_EFORMAT;
    const ingot_cuda_source_shape *sh = &img->shape[s];
    ingot_cuda_source *src = &p->source[s];
    if (!sh->units) { /* predict partner: [rows, H] covering every mapped row */
        if (cols != p->unit_len || rows < sh->min_rows)
            return -INGOT_EFORMAT;
        src->unit_stride = (uint32_t)cols;
        src->pos_stride = 1;
    } else if (sh->axis == 0) {
        if (rows != sh->units || cols != p->unit_len)
            return -INGOT_EFORMAT;
        src->unit_stride = (uint32_t)cols;
        src->pos_stride = 1;
    } else {
        if (rows != p->unit_len || cols != sh->units)
            return -INGOT_EFORMAT;
        src->unit_stride = 1;
        src->pos_stride = (uint32_t)cols;
    }
    if (rows * cols >= ((uint64_t)1 << 32))
        return -INGOT_EFORMAT;
    src->ptr = address;
    return 0;
}

void ingot_cuda_bind_self(ingot_cuda_params *p, uint64_t out_address) {
    for (uint32_t s = 0; p->kind == INGOT_CUDA_KIND_MATCH && s < p->source_count; s++)
        if (p->source[s].self) {
            p->source[s].ptr = out_address;
            p->source[s].unit_stride = (uint32_t)p->unit_len;
            p->source[s].pos_stride = 1;
        }
}
