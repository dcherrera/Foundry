/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_predict_plan.c
 * @brief  Predict mode encoder: per-unit least-squares fit, coefficient quantization, symbol
 *         counts, tables and payload encode (see ingot_predict.h).
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * The fit runs in double precision and may differ across platforms; that only changes which
 * coefficients get stored. Everything the decoder recomputes goes through the integer code in
 * ingot_predict.c, so the stored payload decodes the same everywhere.
 */
#include "ingot_predict.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_parallel.h"

enum {
    TILE_UNITS = 64,     /* column units fitted together (rows of D read once per tile) */
    BLOCK = 1 << INGOT_BLOCK_LOG2,
    COEF_MAX = 32767,
    COEF_QUANTILE = 999, /* per mille */
    FIX_HEADROOM = 1,    /* partner fixed point stays below 2^(FIX_BITS - 1) */
};

/* Normal equations of one unit: x_j . x_k, x_j . y, y . y. */
typedef struct {
    double xx[3]; /* 11, 12, 22 */
    double xy[2];
    double yy;
} unit_sums;

struct ingot_predict_plan {
    ingot_predict_params params; /* owns scale and coef */
    ingot_predict_model *model;
    const uint16_t *data;
    const uint16_t *partners[INGOT_PREDICT_MAX_PARTNERS];
    uint64_t count;
    size_t block_count;
    ingot_fc_tables *tables;
    double coded_bits;
    uint64_t raw_bits;
};

static double finite_value(uint16_t w) {
    return (w & 0x7F80u) == 0x7F80u ? 0.0 : (double)ingot_bf16_to_f32(w);
}

typedef struct {
    const ingot_predict_plan *plan;
    unit_sums *sums;       /* [U] */
    uint16_t *max_mag;     /* [tiles][J]: largest finite partner magnitude (BF16 bits) */
} fit_job;

static void accumulate(unit_sums *s, const double *x, uint32_t np, double y) {
    s->xx[0] += x[0] * x[0];
    s->xy[0] += x[0] * y;
    if (np > 1) {
        s->xx[1] += x[0] * x[1];
        s->xx[2] += x[1] * x[1];
        s->xy[1] += x[1] * y;
    }
    s->yy += y * y;
}

/* Sums of units [u0, u1). COLS reads D row by row across the tile; ROWS reads unit rows. */
static int fit_tile(void *ctx, size_t tile) {
    fit_job *job = ctx;
    const ingot_predict_plan *p = job->plan;
    const ingot_predict_geometry *g = &p->params.geom;
    uint64_t u0 = (uint64_t)tile * TILE_UNITS;
    uint64_t u1 = u0 + TILE_UNITS < g->units ? u0 + TILE_UNITS : g->units;
    uint32_t np = g->partner_count;
    uint16_t *mag = job->max_mag + tile * INGOT_PREDICT_MAX_PARTNERS;
    for (uint64_t h = 0; h < g->unit_len; h++) {
        for (uint64_t u = u0; u < u1; u++) {
            double x[INGOT_PREDICT_MAX_PARTNERS] = {0};
            for (uint32_t j = 0; j < np; j++) {
                uint16_t v = p->partners[j][ingot_predict_map(&g->partner[j], u) * g->unit_len + h];
                x[j] = finite_value(v);
                if ((v & 0x7F80u) != 0x7F80u && (v & 0x7FFFu) > mag[j])
                    mag[j] = v & 0x7FFFu;
            }
            uint64_t at = g->axis == INGOT_PREDICT_ROWS ? u * g->unit_len + h : h * g->units + u;
            accumulate(&job->sums[u], x, np, finite_value(p->data[at]));
        }
    }
    return 0;
}

/* Least-squares coefficients of one unit (tiny ridge keeps a singular pair finite). */
static void solve(const unit_sums *s, uint32_t np, double *a) {
    if (np == 1) {
        a[0] = s->xx[0] > 0 ? s->xy[0] / s->xx[0] : 0.0;
        return;
    }
    double ridge = 1e-9 * (s->xx[0] + s->xx[2]);
    double s11 = s->xx[0] + ridge, s22 = s->xx[2] + ridge, s12 = s->xx[1];
    double det = s11 * s22 - s12 * s12;
    a[0] = det > 0 ? (s22 * s->xy[0] - s12 * s->xy[1]) / det : 0.0;
    a[1] = det > 0 ? (s11 * s->xy[1] - s12 * s->xy[0]) / det : 0.0;
    for (int j = 0; j < 2; j++)
        a[j] = isfinite(a[j]) ? a[j] : 0.0;
}

static int8_t unit_scale(double yy, uint64_t len) {
    double rms = sqrt(yy / (double)len);
    if (!(rms > 0))
        return -128;
    double s = nearbyint(4.0 * log2(rms));
    return (int8_t)(s < -128 ? -128 : s > 127 ? 127 : s);
}

static int32_t clamp_exp(double e) {
    return e < -INGOT_PREDICT_EXP_LIMIT ? -INGOT_PREDICT_EXP_LIMIT
           : e > INGOT_PREDICT_EXP_LIMIT ? INGOT_PREDICT_EXP_LIMIT : (int32_t)e;
}

/* Exponents, int16 coefficients and C from the real coefficients and partner magnitudes. */
static int compare_double(const void *x, const void *y) {
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}

/* The coefficient magnitude partner j's exponent is set for: a high quantile of |a|, so one
 * outlier unit does not coarsen every other unit (the few above it clamp to +-COEF_MAX). */
static int coef_range(const double *a, uint64_t units, uint32_t np, uint32_t j, double *range) {
    double *mag = malloc((size_t)units * sizeof(*mag));
    if (!mag)
        return -INGOT_ENOMEM;
    for (uint64_t u = 0; u < units; u++)
        mag[u] = fabs(a[u * np + j]);
    qsort(mag, (size_t)units, sizeof(*mag), compare_double);
    *range = mag[(units - 1) * COEF_QUANTILE / 1000];
    *range = *range > 0 ? *range : mag[units - 1];
    free(mag);
    return 0;
}

static int quantize(ingot_predict_params *q, const double *a, const uint16_t *max_mag) {
    const ingot_predict_geometry *g = &q->geom;
    uint32_t np = g->partner_count;
    int32_t c = INGOT_PREDICT_EXP_LIMIT;
    for (uint32_t j = 0; j < np; j++) {
        double amax = 0;
        int rc = coef_range(a, g->units, np, j, &amax);
        if (rc)
            return rc;
        q->coef_exp[j] = amax > 0 ? clamp_exp(floor(log2((double)COEF_MAX / amax))) : 0;
        /* |x| < 2^(exp - 126) for a BF16 magnitude with exponent field exp */
        int32_t top = (int32_t)(max_mag[j] >> 7) - 126;
        int32_t cj = q->coef_exp[j] + INGOT_PREDICT_FIX_BITS - FIX_HEADROOM - top;
        c = cj < c ? cj : c;
    }
    q->scale_c = c < -INGOT_PREDICT_EXP_LIMIT ? -INGOT_PREDICT_EXP_LIMIT : c;
    for (uint64_t i = 0; i < g->units * np; i++) {
        double v = nearbyint(ldexp(a[i], q->coef_exp[i % np]));
        q->coef[i] = (int16_t)(v > COEF_MAX ? COEF_MAX : v < -COEF_MAX ? -COEF_MAX : v);
    }
    return 0;
}

/* Fit every unit, then fill p->params (scales, coefficients, exponents, classes). */
static int plan_fit(ingot_predict_plan *p, int threads) {
    const ingot_predict_geometry *g = &p->params.geom;
    size_t tiles = (size_t)((g->units + TILE_UNITS - 1) / TILE_UNITS);
    fit_job job = {p, calloc((size_t)g->units, sizeof(unit_sums)),
                   calloc(tiles * INGOT_PREDICT_MAX_PARTNERS, sizeof(uint16_t))};
    double *a = calloc((size_t)(g->units * g->partner_count), sizeof(double));
    int rc = (job.sums && job.max_mag && a) ? 0 : -INGOT_ENOMEM;
    if (!rc)
        rc = ingot_parallel_for(tiles, threads, fit_tile, &job);
    uint16_t max_mag[INGOT_PREDICT_MAX_PARTNERS] = {0};
    for (size_t t = 0; !rc && t < tiles; t++)
        for (int j = 0; j < INGOT_PREDICT_MAX_PARTNERS; j++)
            if (job.max_mag[t * INGOT_PREDICT_MAX_PARTNERS + j] > max_mag[j])
                max_mag[j] = job.max_mag[t * INGOT_PREDICT_MAX_PARTNERS + j];
    for (uint64_t u = 0; !rc && u < g->units; u++) {
        solve(&job.sums[u], g->partner_count, a + u * g->partner_count);
        p->params.scale[u] = unit_scale(job.sums[u].yy, g->unit_len);
    }
    if (!rc)
        rc = quantize(&p->params, a, max_mag);
    free(job.sums);
    free(job.max_mag);
    free(a);
    return rc;
}

static void block_range(const ingot_predict_plan *p, size_t b, uint64_t *lo, uint64_t *hi) {
    *lo = (uint64_t)b * BLOCK;
    *hi = *lo + BLOCK < p->count ? *lo + BLOCK : p->count;
}

/* Counts per (scale bin, bucket, sign flip, symbol): every class layout the search tries is a
 * merge of these. Bin = scale - class_ref + FINE_HALF, clamped: with width <= 4 and up to 8
 * classes, every scale past +-FINE_HALF lands in an edge class, as its edge bin does. */
enum {
    FINE_HALF = 16,
    FINE_BINS = 2 * FINE_HALF + 1,
    FINE_CELLS = FINE_BINS * INGOT_PREDICT_BUCKETS * 2,
    SYMBOLS = 1024,
    FINE_COUNTS = FINE_CELLS * (1 + SYMBOLS),
};

typedef struct {
    const ingot_predict_plan *plan;
    const ingot_predict_model *flat; /* one class: context byte = bucket */
    uint8_t *bin;                    /* [U] */
    size_t chunks;
    uint64_t *counts;                /* chunks x FINE_COUNTS: flips, then symbols */
} count_job;

static void count_block(const count_job *job, uint64_t lo, uint64_t hi, const uint8_t *ctx,
                        uint64_t *counts) {
    const ingot_predict_plan *p = job->plan;
    const ingot_predict_geometry *g = &p->params.geom;
    uint64_t *symbols = counts + FINE_CELLS;
    for (uint64_t i = lo; i < hi; i++) {
        uint64_t u = g->axis == INGOT_PREDICT_ROWS ? i / g->unit_len : i % g->units;
        uint8_t c = ctx[i - lo];
        uint16_t w = p->data[i];
        uint32_t cell = (job->bin[u] * INGOT_PREDICT_BUCKETS + (c & 0x7Fu)) * 2 +
                        (((uint32_t)w >> 15) ^ (uint32_t)(c >> 7));
        counts[cell]++;
        symbols[(size_t)cell * SYMBOLS + ingot_bf16_symbol(w)]++;
    }
}

static int count_chunk(void *ctx, size_t k) {
    count_job *job = ctx;
    const ingot_predict_plan *p = job->plan;
    uint8_t *ctxbuf = malloc(BLOCK);
    if (!ctxbuf)
        return -INGOT_ENOMEM;
    for (size_t b = k * p->block_count / job->chunks;
         b < (k + 1) * p->block_count / job->chunks; b++) {
        uint64_t lo = 0, hi = 0;
        block_range(p, b, &lo, &hi);
        ingot_predict_contexts(job->flat, p->partners, lo, hi, ctxbuf);
        count_block(job, lo, hi, ctxbuf, job->counts + k * (size_t)FINE_COUNTS);
    }
    free(ctxbuf);
    return 0;
}

/* Median unit scale (the class layouts are centred on it). */
static int32_t median_scale(const ingot_predict_params *q) {
    uint64_t hist[256] = {0}, seen = 0;
    for (uint64_t u = 0; u < q->geom.units; u++)
        hist[q->scale[u] + 128]++;
    for (int s = 0; s < 256; s++) {
        seen += hist[s];
        if (2 * seen >= q->geom.units)
            return s - 128;
    }
    return 0;
}

/* Fine counts of the whole tensor (summed over chunks into counts[0..FINE_COUNTS)). */
static int count_fine(ingot_predict_plan *p, int threads, uint64_t **out) {
    ingot_predict_params flat = p->params;
    flat.classes = 1;
    ingot_predict_model *m = NULL;
    size_t chunks = (size_t)(threads > 1 ? threads : 1);
    chunks = chunks < p->block_count ? chunks : p->block_count;
    count_job job = {p, NULL, malloc((size_t)p->params.geom.units), chunks,
                     calloc(chunks * (size_t)FINE_COUNTS, sizeof(uint64_t))};
    int rc = (job.bin && job.counts) ? ingot_predict_model_create(&flat, &m) : -INGOT_ENOMEM;
    job.flat = m;
    for (uint64_t u = 0; !rc && u < p->params.geom.units; u++) {
        int32_t b = p->params.scale[u] - p->params.class_ref + FINE_HALF;
        job.bin[u] = (uint8_t)(b < 0 ? 0 : b >= FINE_BINS ? FINE_BINS - 1 : b);
    }
    if (!rc)
        rc = ingot_parallel_for(chunks, threads, count_chunk, &job);
    for (size_t k = 1; !rc && k < chunks; k++)
        for (size_t i = 0; i < FINE_COUNTS; i++)
            job.counts[i] += job.counts[k * (size_t)FINE_COUNTS + i];
    ingot_predict_model_free(m);
    free(job.bin);
    if (rc)
        free(job.counts);
    else
        *out = job.counts;
    return rc;
}

/* Tables of one class layout from the fine counts; *bytes = coded symbols + tables. */
static int try_layout(const ingot_predict_plan *p, const uint64_t *fine, uint32_t classes,
                      uint32_t width, ingot_fc_tables **tables, double *bits, double *bytes) {
    ingot_predict_params q = p->params;
    q.classes = classes;
    q.class_width = width;
    ingot_predict_model *m = NULL;
    int rc = ingot_predict_model_create(&q, &m);
    const ingot_fc_model *fm = rc ? NULL : ingot_predict_fc_model(m);
    uint64_t *counts = rc ? NULL : calloc(fm->cell[fm->contexts], sizeof(uint64_t));
    rc = rc ? rc : counts ? 0 : -INGOT_ENOMEM;
    uint32_t zcount = classes * INGOT_PREDICT_BUCKETS;
    for (uint32_t b = 0; !rc && b < FINE_BINS; b++) {
        uint32_t cls = ingot_predict_class((int32_t)b - FINE_HALF + q.class_ref, classes, width,
                                           q.class_ref);
        for (uint32_t cell = 0; cell < INGOT_PREDICT_BUCKETS * 2; cell++) {
            uint32_t z = cls * INGOT_PREDICT_BUCKETS + cell / 2, flip = cell % 2;
            size_t from = (size_t)b * INGOT_PREDICT_BUCKETS * 2 + cell;
            counts[fm->cell[z] + flip] += fine[from];
            const uint64_t *src = fine + FINE_CELLS + from * SYMBOLS;
            uint64_t *dst = counts + fm->cell[zcount + 2 * z + flip];
            for (uint32_t s = 0; s < SYMBOLS; s++)
                dst[s] += src[s];
        }
    }
    if (!rc)
        rc = ingot_fc_tables_build(fm, counts, tables, bits);
    if (!rc)
        *bytes = *bits / 8.0 + (double)ingot_fc_tables_size(*tables);
    free(counts);
    ingot_predict_model_free(m);
    return rc;
}

/* The scale-class layout (classes x width) with the fewest coded bytes, tables included. */
static int plan_count(ingot_predict_plan *p, int threads) {
    static const uint32_t LAYOUTS[][2] = {{1, 1}, {2, 1}, {2, 2}, {2, 4}, {4, 1},
                                          {4, 2}, {4, 4}, {8, 1}, {8, 2}, {8, 4}};
    uint64_t *fine = NULL;
    p->params.class_ref = median_scale(&p->params);
    int rc = count_fine(p, threads, &fine);
    double best = 0;
    for (size_t k = 0; !rc && k < sizeof(LAYOUTS) / sizeof(LAYOUTS[0]); k++) {
        ingot_fc_tables *t = NULL;
        double bits = 0, bytes = 0;
        rc = try_layout(p, fine, LAYOUTS[k][0], LAYOUTS[k][1], &t, &bits, &bytes);
        if (!rc && (!p->tables || bytes < best)) {
            ingot_fc_tables_free(p->tables);
            p->tables = t;
            p->coded_bits = bits;
            p->params.classes = LAYOUTS[k][0];
            p->params.class_width = LAYOUTS[k][1];
            best = bytes;
        } else {
            ingot_fc_tables_free(t);
        }
    }
    p->raw_bits = 5 * p->count;
    free(fine);
    return rc;
}

int ingot_predict_plan_create(const ingot_predict_geometry *g, const uint16_t *data,
                              const uint16_t *const *partners, int threads,
                              ingot_predict_plan **out) {
    if (!g || !data || !partners || !out || ingot_predict_geometry_check(g))
        return -INGOT_EINVAL;
    for (uint32_t j = 0; j < g->partner_count; j++)
        if (!partners[j])
            return -INGOT_EINVAL;
    ingot_predict_plan *p = calloc(1, sizeof(*p));
    if (!p)
        return -INGOT_ENOMEM;
    p->params.geom = *g;
    p->params.classes = 1;
    p->params.class_width = 1;
    p->data = data;
    for (uint32_t j = 0; j < g->partner_count; j++)
        p->partners[j] = partners[j];
    p->count = g->units * g->unit_len;
    p->block_count = (size_t)((p->count + BLOCK - 1) / BLOCK);
    p->params.scale = malloc((size_t)g->units);
    p->params.coef = calloc((size_t)(g->units * g->partner_count), sizeof(*p->params.coef));
    int rc = (p->params.scale && p->params.coef) ? 0 : -INGOT_ENOMEM;
    if (!rc)
        rc = plan_fit(p, threads);
    if (!rc)
        rc = plan_count(p, threads);
    if (!rc)
        rc = ingot_predict_model_create(&p->params, &p->model);
    if (rc) {
        ingot_predict_plan_free(p);
        return rc;
    }
    *out = p;
    return 0;
}

uint64_t ingot_predict_plan_prefix_bytes(const ingot_predict_plan *p) {
    return (ingot_predict_params_bytes(&p->params.geom) + ingot_fc_tables_size(p->tables) + 7) &
           ~(uint64_t)7;
}

size_t ingot_predict_plan_blocks(const ingot_predict_plan *p) { return p->block_count; }

uint64_t ingot_predict_plan_bytes(const ingot_predict_plan *p) {
    uint64_t framing = (uint64_t)p->block_count * (INGOT_BLOCK_HEADER_BYTES + 8 + 8) + 8;
    return ingot_predict_plan_prefix_bytes(p) + framing + (p->raw_bits + 7) / 8 +
           (uint64_t)(p->coded_bits / 8.0);
}

static int encode_block(void *ctx, size_t b, ingot_block_buf *out) {
    const ingot_predict_plan *p = ctx;
    uint64_t lo = 0, hi = 0;
    block_range(p, b, &lo, &hi);
    uint8_t *ctxbuf = malloc((size_t)(hi - lo));
    if (!ctxbuf)
        return -INGOT_ENOMEM;
    ingot_predict_contexts(p->model, p->partners, lo, hi, ctxbuf);
    ingot_fcoder c;
    ingot_fc_begin_encode(&c, p->tables);
    ingot_predict_code(&c, p->model, ctxbuf, p->data + lo, NULL, (size_t)(hi - lo));
    free(ctxbuf);
    return ingot_fc_finish_encode(&c, out);
}

int ingot_predict_encode(const ingot_predict_plan *p, int threads, ingot_sink *sink) {
    if (!p || !sink || sink->pos % 8)
        return -INGOT_EINVAL;
    uint64_t prefix = ingot_predict_plan_prefix_bytes(p);
    uint8_t *buf = calloc(1, (size_t)prefix);
    if (!buf)
        return -INGOT_ENOMEM;
    ingot_predict_prefix_write(&p->params, p->block_count,
                               ingot_predict_fc_model(p->model)->contexts, p->tables, buf);
    int rc = ingot_sink_write(sink, buf, (size_t)prefix);
    free(buf);
    if (!rc)
        rc = ingot_block_area_write(sink, p->block_count, threads, encode_block, (void *)p);
    return rc;
}

void ingot_predict_plan_free(ingot_predict_plan *p) {
    if (!p)
        return;
    ingot_fc_tables_free(p->tables);
    ingot_predict_model_free(p->model);
    free(p->params.scale);
    free(p->params.coef);
    free(p);
}
