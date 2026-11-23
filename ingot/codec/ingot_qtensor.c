/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_qtensor.c
 * @brief  Quantized-tensor payload driver (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_qtensor.h"

#include <stdlib.h>
#include <string.h>

#include "ingot/codec/ingot_qtype.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_ggml.h"
#include "ingot/core/ingot_parallel.h"

enum {
    HEADER_BYTES = 48,
    TARGET_WEIGHTS = 1 << INGOT_BLOCK_LOG2,
    FLAG_BASE = 1,
    UNIT_WEIGHTS = 32, /* block delta: flag units of at least this many weights */
};

/* Everything both sides derive from (mode, dtype, cols). */
typedef struct {
    const ingot_qtype *type;
    uint32_t ggml_type;
    uint32_t block_weights; /* coder unit geometry */
    uint32_t block_bytes;
    int with_base;
    uint64_t rows;
    uint64_t cols;
    uint64_t row_bytes;
    uint64_t rows_per_block;
    size_t block_count;
    ingot_fc_model model;
} qtensor_shape;

static const ingot_qtype *quant_type(uint32_t ggml_type) {
    if (ggml_type == INGOT_GGML_Q8_0)
        return ingot_q8_qtype();
    return ingot_kquant_qtype(ggml_type);
}

int ingot_qtensor_supports(ingot_dtype dtype) {
    uint32_t type = 0;
    return ingot_dtype_ggml_type(dtype, &type) == 0 && quant_type(type) != NULL;
}

/* Coder unit: the ggml block, or for block delta a run of small blocks of >= 32 weights. */
static int shape_unit(ingot_mode mode, uint32_t type, uint64_t cols, qtensor_shape *s) {
    uint32_t bw = 0, bb = 0;
    if (ingot_ggml_geometry(type, &bw, &bb) || !cols || cols % bw)
        return -INGOT_EINVAL;
    s->block_weights = bw;
    s->block_bytes = bb;
    if (mode == INGOT_MODE_BLOCK_DELTA) {
        s->type = ingot_blockdelta_qtype();
        if (bw < UNIT_WEIGHTS && UNIT_WEIGHTS % bw == 0 && cols % UNIT_WEIGHTS == 0) {
            s->block_bytes = bb * (UNIT_WEIGHTS / bw);
            s->block_weights = UNIT_WEIGHTS;
        }
        return 0;
    }
    if (mode != INGOT_MODE_QUANT && mode != INGOT_MODE_QUANT_DELTA)
        return -INGOT_EINVAL;
    s->type = quant_type(type);
    return s->type ? 0 : -INGOT_EINVAL;
}

static int shape_init(ingot_mode mode, ingot_dtype dtype, uint64_t rows, uint64_t cols,
                      qtensor_shape *s) {
    *s = (qtensor_shape){0};
    if (ingot_dtype_ggml_type(dtype, &s->ggml_type) || !rows)
        return -INGOT_EINVAL;
    int rc = shape_unit(mode, s->ggml_type, cols, s);
    if (rc)
        return rc;
    s->with_base = mode != INGOT_MODE_QUANT;
    s->rows = rows;
    s->cols = cols;
    s->row_bytes = cols / s->block_weights * s->block_bytes;
    s->rows_per_block = cols >= TARGET_WEIGHTS ? 1 : TARGET_WEIGHTS / cols;
    s->block_count = (size_t)((rows + s->rows_per_block - 1) / s->rows_per_block);
    return s->type->model(&s->model, s->with_base);
}

int ingot_qtensor_fits(ingot_mode mode, ingot_dtype dtype, uint64_t cols) {
    qtensor_shape s;
    return shape_init(mode, dtype, 1, cols, &s) == 0;
}

static void block_rows(const qtensor_shape *s, size_t b, uint64_t *r0, uint64_t *r1) {
    *r0 = (uint64_t)b * s->rows_per_block;
    *r1 = *r0 + s->rows_per_block < s->rows ? *r0 + s->rows_per_block : s->rows;
}

static ingot_qrows rows_of(const qtensor_shape *s, size_t b, void *scratch) {
    uint64_t r0 = 0, r1 = 0;
    block_rows(s, b, &r0, &r1);
    return (ingot_qrows){.rows = r1 - r0, .cols = s->cols, .block_weights = s->block_weights,
                         .block_bytes = s->block_bytes, .scratch = scratch};
}

struct ingot_qtensor_plan {
    qtensor_shape shape;
    const uint8_t *data;
    const uint8_t *base;
    ingot_fc_tables *tables;
    double coded_bits;
    uint64_t raw_bits;
};

typedef struct {
    const ingot_qtensor_plan *plan;
    size_t chunks;
    uint64_t *counts;   /* chunks x cells */
    uint64_t *raw_bits; /* per chunk */
} count_job;

static int count_chunk(void *ctx, size_t k) {
    count_job *job = ctx;
    const qtensor_shape *s = &job->plan->shape;
    size_t first = k * s->block_count / job->chunks, end = (k + 1) * s->block_count / job->chunks;
    void *scratch = malloc(s->type->scratch_bytes(s->cols));
    if (!scratch)
        return -INGOT_ENOMEM;
    ingot_fcoder c;
    ingot_fc_begin_count(&c, &s->model, job->counts + k * s->model.cell[s->model.contexts]);
    for (size_t b = first; b < end; b++) {
        ingot_qrows q = rows_of(s, b, scratch);
        uint64_t at = (uint64_t)b * s->rows_per_block * s->row_bytes;
        q.src = job->plan->data + at;
        q.base = job->plan->base ? job->plan->base + at : NULL;
        s->type->code(&c, &q);
    }
    free(scratch);
    job->raw_bits[k] = c.raw_bits;
    return c.err;
}

static int plan_count(ingot_qtensor_plan *p, int threads) {
    const qtensor_shape *s = &p->shape;
    size_t cells = s->model.cell[s->model.contexts];
    size_t chunks = (size_t)(threads > 1 ? threads : 1);
    chunks = chunks < s->block_count ? chunks : s->block_count;
    count_job job = {p, chunks, calloc(chunks * cells, sizeof(uint64_t)),
                     calloc(chunks, sizeof(uint64_t))};
    int rc = (job.counts && job.raw_bits) ? 0 : -INGOT_ENOMEM;
    if (!rc)
        rc = ingot_parallel_for(chunks, threads, count_chunk, &job);
    for (size_t k = 1; !rc && k < chunks; k++)
        for (size_t i = 0; i < cells; i++)
            job.counts[i] += job.counts[k * cells + i];
    for (size_t k = 0; !rc && k < chunks; k++)
        p->raw_bits += job.raw_bits[k];
    if (!rc)
        rc = ingot_fc_tables_build(&s->model, job.counts, &p->tables, &p->coded_bits);
    free(job.counts);
    free(job.raw_bits);
    return rc;
}

int ingot_qtensor_plan_create(ingot_mode mode, ingot_dtype dtype, const uint8_t *data,
                              const uint8_t *base, uint64_t rows, uint64_t cols, int threads,
                              ingot_qtensor_plan **out) {
    if (!data || !out || (mode != INGOT_MODE_QUANT && !base))
        return -INGOT_EINVAL;
    ingot_qtensor_plan *p = calloc(1, sizeof(*p));
    if (!p)
        return -INGOT_ENOMEM;
    p->data = data;
    p->base = mode == INGOT_MODE_QUANT ? NULL : base;
    int rc = shape_init(mode, dtype, rows, cols, &p->shape);
    if (!rc)
        rc = plan_count(p, threads);
    if (rc) {
        ingot_qtensor_plan_free(p);
        return rc;
    }
    *out = p;
    return 0;
}

uint64_t ingot_qtensor_plan_prefix_bytes(const ingot_qtensor_plan *p) {
    return (HEADER_BYTES + ingot_fc_tables_size(p->tables) + 7) & ~(uint64_t)7;
}

size_t ingot_qtensor_plan_blocks(const ingot_qtensor_plan *p) { return p->shape.block_count; }

uint64_t ingot_qtensor_plan_bytes(const ingot_qtensor_plan *p) {
    uint64_t framing = (uint64_t)p->shape.block_count * (INGOT_BLOCK_HEADER_BYTES + 8 + 8) + 8;
    return ingot_qtensor_plan_prefix_bytes(p) + framing + (p->raw_bits + 7) / 8 +
           (uint64_t)(p->coded_bits / 8.0);
}

static int encode_block(void *ctx, size_t b, ingot_block_buf *out) {
    const ingot_qtensor_plan *p = ctx;
    const qtensor_shape *s = &p->shape;
    void *scratch = malloc(s->type->scratch_bytes(s->cols));
    if (!scratch)
        return -INGOT_ENOMEM;
    ingot_qrows q = rows_of(s, b, scratch);
    uint64_t at = (uint64_t)b * s->rows_per_block * s->row_bytes;
    q.src = p->data + at;
    q.base = p->base ? p->base + at : NULL;
    ingot_fcoder c;
    ingot_fc_begin_encode(&c, p->tables);
    s->type->code(&c, &q);
    free(scratch);
    return ingot_fc_finish_encode(&c, out);
}

int ingot_qtensor_encode(const ingot_qtensor_plan *p, int threads, ingot_sink *sink) {
    if (!p || !sink || sink->pos % 8)
        return -INGOT_EINVAL;
    const qtensor_shape *s = &p->shape;
    uint64_t prefix = ingot_qtensor_plan_prefix_bytes(p);
    uint8_t *buf = calloc(1, (size_t)prefix);
    if (!buf)
        return -INGOT_ENOMEM;
    ingot_store_u32(buf, s->ggml_type);
    ingot_store_u32(buf + 4, s->with_base ? FLAG_BASE : 0);
    ingot_store_u64(buf + 8, s->rows);
    ingot_store_u64(buf + 16, s->cols);
    ingot_store_u64(buf + 24, s->rows_per_block);
    ingot_store_u64(buf + 32, s->block_count);
    ingot_store_u32(buf + 40, s->model.contexts);
    ingot_fc_tables_write(p->tables, buf + HEADER_BYTES);
    int rc = ingot_sink_write(sink, buf, (size_t)prefix);
    free(buf);
    if (!rc)
        rc = ingot_block_area_write(sink, s->block_count, threads, encode_block, (void *)p);
    return rc;
}

void ingot_qtensor_plan_free(ingot_qtensor_plan *p) {
    if (!p)
        return;
    ingot_fc_tables_free(p->tables);
    free(p);
}

struct ingot_qtensor_decoder {
    qtensor_shape shape;
    ingot_fc_tables *tables;
};

int ingot_qtensor_decoder_open(ingot_mode mode, ingot_dtype dtype, uint64_t rows, uint64_t cols,
                               const uint8_t *prefix, uint64_t avail,
                               ingot_qtensor_decoder **out, uint64_t *prefix_bytes) {
    if (!prefix || !out || !prefix_bytes || avail < HEADER_BYTES)
        return -INGOT_EFORMAT;
    ingot_qtensor_decoder *d = calloc(1, sizeof(*d));
    if (!d)
        return -INGOT_ENOMEM;
    qtensor_shape *s = &d->shape;
    int rc = shape_init(mode, dtype, rows, cols, s) ? -INGOT_EFORMAT : 0;
    if (!rc && (ingot_load_u32(prefix) != s->ggml_type ||
                ingot_load_u32(prefix + 4) != (s->with_base ? (uint32_t)FLAG_BASE : 0u) ||
                ingot_load_u64(prefix + 8) != rows || ingot_load_u64(prefix + 16) != cols ||
                ingot_load_u64(prefix + 24) != s->rows_per_block ||
                ingot_load_u64(prefix + 32) != s->block_count ||
                ingot_load_u32(prefix + 40) != s->model.contexts || ingot_load_u32(prefix + 44)))
        rc = -INGOT_EFORMAT;
    uint64_t used = 0;
    if (!rc)
        rc = ingot_fc_tables_read(&s->model, prefix + HEADER_BYTES, avail - HEADER_BYTES,
                                  &d->tables, &used);
    uint64_t at = (HEADER_BYTES + used + 7) & ~(uint64_t)7;
    if (!rc && at > avail)
        rc = -INGOT_EFORMAT;
    if (rc) {
        ingot_qtensor_decoder_close(d);
        return rc;
    }
    *prefix_bytes = at;
    *out = d;
    return 0;
}

size_t ingot_qtensor_decoder_blocks(const ingot_qtensor_decoder *d) {
    return d->shape.block_count;
}

void ingot_qtensor_decoder_range(const ingot_qtensor_decoder *d, size_t b, uint64_t *lo,
                                 uint64_t *hi) {
    uint64_t r0 = 0, r1 = 0;
    block_rows(&d->shape, b, &r0, &r1);
    *lo = r0 * d->shape.row_bytes;
    *hi = r1 * d->shape.row_bytes;
}

int ingot_qtensor_decoder_block(const ingot_qtensor_decoder *d, size_t b,
                                const ingot_block_view *v, const uint8_t *base, uint8_t *out) {
    const qtensor_shape *s = &d->shape;
    if (s->with_base && !base)
        return -INGOT_EBASE;
    void *scratch = malloc(s->type->scratch_bytes(s->cols));
    if (!scratch)
        return -INGOT_ENOMEM;
    ingot_qrows q = rows_of(s, b, scratch);
    q.dst = out;
    q.base = s->with_base ? base : NULL;
    ingot_fcoder c;
    ingot_fc_begin_decode(&c, d->tables, v);
    s->type->code(&c, &q);
    free(scratch);
    return ingot_fc_finish_decode(&c);
}

void ingot_qtensor_decoder_close(ingot_qtensor_decoder *d) {
    if (!d)
        return;
    ingot_fc_tables_free(d->tables);
    free(d);
}
