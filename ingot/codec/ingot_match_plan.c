/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_match_plan.c
 * @brief  Match mode encoder: per-unit partner choice (exact re-rank of the search candidates,
 *         least squares, cost-based K), coefficient quantization, the side stream, symbol counts,
 *         tables and payload encode (see ingot_match.h).
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Fits run in double precision and may differ across platforms; that only changes which
 * partners and coefficients get stored. Everything the decoder recomputes goes through the
 * integer code in ingot_match.c.
 */
#include "ingot_match.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_parallel.h"

enum {
    K_MAX = INGOT_MATCH_MAX_K,
    TILE_UNITS = 64,
    BLOCK = 1 << INGOT_BLOCK_LOG2,
    NO_SOURCE = 0xFF,
    LINK_BITS = 3,       /* side cost estimates (bits) */
    COEF_BITS = 10,
    K_BITS = 2,
    FIX_HEADROOM = 1,
    EXP_SPAN = 14,       /* coefficient exponents tried below the full-range one */
};

/* Fit result of one unit: up to 8 partners with real coefficients. */
typedef struct {
    double rss;                 /* residual with the real coefficients */
    double gkk[K_MAX];          /* partner squared norms */
    double a[K_MAX];
    uint16_t top[K_MAX];        /* largest finite partner magnitude (BF16 bits, sign cleared) */
} unit_fit;

struct ingot_match_plan {
    ingot_match_params params; /* owns the per-unit arrays */
    ingot_match_model *model;
    const uint16_t *data;
    const uint16_t *sources[INGOT_MATCH_MAX_SOURCES];
    uint64_t count;
    size_t block_count;
    ingot_fc_tables *tables;
    ingot_fc_tables *side_tables;
    ingot_block_buf side_block;
    double coded_bits;
    uint64_t raw_bits;
};

static double finite_value(uint16_t w) {
    return (w & 0x7F80u) == 0x7F80u ? 0.0 : (double)ingot_bf16_to_f32(w);
}

static double dot_bf16(const uint16_t *x, const uint16_t *y, uint64_t n) {
    double s = 0;
    for (uint64_t h = 0; h < n; h++)
        s += finite_value(x[h]) * finite_value(y[h]);
    return s;
}

/* Solve G a = b (k x k, row stride K_MAX) by Cholesky with a tiny ridge; returns RSS = yy - b.a,
 * or yy when singular. */
static double solve_ls(const double *g, const double *b, uint32_t k, double yy, double *a) {
    double l[K_MAX * K_MAX] = {0}, z[K_MAX];
    double ridge = 0;
    for (uint32_t i = 0; i < k; i++)
        ridge += g[i * K_MAX + i];
    ridge *= 1e-9;
    for (uint32_t i = 0; i < k; i++) {
        for (uint32_t j = 0; j <= i; j++) {
            double s = g[i * K_MAX + j] + (i == j ? ridge : 0);
            for (uint32_t t = 0; t < j; t++)
                s -= l[i * K_MAX + t] * l[j * K_MAX + t];
            if (i == j) {
                if (!(s > 0))
                    return yy;
                l[i * K_MAX + i] = sqrt(s);
            } else {
                l[i * K_MAX + j] = s / l[j * K_MAX + j];
            }
        }
    }
    for (uint32_t i = 0; i < k; i++) {
        double s = b[i];
        for (uint32_t t = 0; t < i; t++)
            s -= l[i * K_MAX + t] * z[t];
        z[i] = s / l[i * K_MAX + i];
    }
    for (uint32_t i = k; i-- > 0;) {
        double s = z[i];
        for (uint32_t t = i + 1; t < k; t++)
            s -= l[t * K_MAX + i] * a[t];
        a[i] = s / l[i * K_MAX + i];
    }
    double fit = 0;
    for (uint32_t i = 0; i < k; i++)
        fit += a[i] * b[i];
    return yy - fit > 0 ? yy - fit : 0;
}

typedef struct {
    ingot_match_plan *plan;
    const uint16_t *target;        /* unit-major [U][H] */
    const ingot_match_input *src;
    const ingot_match_candidates *cand;
    unit_fit *fits;
} select_job;

static unsigned bits_for(uint64_t n) {
    unsigned b = 0;
    while (b < 64 && ((uint64_t)1 << b) < n)
        b++;
    return b;
}

static uint16_t max_magnitude(const uint16_t *x, uint64_t n) {
    uint16_t m = 0;
    for (uint64_t h = 0; h < n; h++) {
        uint16_t v = x[h] & 0x7FFFu;
        if ((v & 0x7F80u) != 0x7F80u && v > m)
            m = v;
    }
    return m;
}

/* Unit u: rank candidates by exact |cos|, then choose K by estimated code length. */
static void select_unit(select_job *job, uint64_t u) {
    ingot_match_params *p = &job->plan->params;
    uint64_t len = p->unit_len;
    const uint16_t *y = job->target + u * len;
    uint32_t m = job->cand->per_unit;
    double yy = dot_bf16(y, y, len);
    uint8_t cs[K_MAX];
    uint64_t ci[K_MAX];
    const uint16_t *cx[K_MAX];
    double cos2[K_MAX] = {0}, cb[K_MAX], cg[K_MAX];
    uint32_t have = 0;
    p->k[u] = 0;
    double rms = sqrt(yy / (double)len), sc = rms > 0 ? nearbyint(4.0 * log2(rms)) : -128;
    p->scale[u] = (int8_t)(sc < -128 ? -128 : sc > 127 ? 127 : sc);
    for (uint32_t j = 0; yy > 0 && j < m; j++) {
        uint8_t s = job->cand->source[u * m + j];
        uint64_t i = job->cand->index[u * m + j];
        if (s == NO_SOURCE || s >= p->source_count || i >= job->src[s].units ||
            (job->src[s].self && i >= u))
            continue;
        int dup = 0;
        for (uint32_t t = 0; t < have; t++)
            dup |= cs[t] == s && ci[t] == i;
        const uint16_t *x = job->src[s].unit_major + i * len;
        double g = dup ? 0 : dot_bf16(x, x, len);
        if (!(g > 0))
            continue;
        double b = dot_bf16(x, y, len), c2 = b * b / (g * yy);
        uint32_t at = have < K_MAX ? have++ : K_MAX;
        if (at == K_MAX && c2 <= cos2[K_MAX - 1])
            continue;
        at = at == K_MAX ? K_MAX - 1 : at;
        while (at > 0 && cos2[at - 1] < c2) {
            cs[at] = cs[at - 1], ci[at] = ci[at - 1], cx[at] = cx[at - 1];
            cos2[at] = cos2[at - 1], cb[at] = cb[at - 1], cg[at] = cg[at - 1];
            at--;
        }
        cs[at] = s, ci[at] = i, cx[at] = x, cos2[at] = c2, cb[at] = b, cg[at] = g;
    }
    double gram[K_MAX * K_MAX];
    for (uint32_t i = 0; i < have; i++) {
        gram[i * K_MAX + i] = cg[i];
        for (uint32_t j = 0; j < i; j++)
            gram[i * K_MAX + j] = gram[j * K_MAX + i] = dot_bf16(cx[i], cx[j], len);
    }
    double best_cost = 0, a[K_MAX], best_a[K_MAX], side = K_BITS, best_rss = yy;
    uint32_t best_k = 0;
    for (uint32_t k = 1; k <= have; k++) {
        int same = !job->src[cs[k - 1]].self && ci[k - 1] == u;
        side += LINK_BITS + COEF_BITS + (same ? 0 : bits_for(job->src[cs[k - 1]].units));
        double rss = solve_ls(gram, cb, k, yy, a);
        double cost = 0.5 * (double)len * log2((rss > 0 ? rss : yy * 1e-12) / yy) + side;
        if (cost < best_cost) {
            best_cost = cost;
            best_k = k;
            best_rss = rss;
            memcpy(best_a, a, sizeof(a));
        }
    }
    unit_fit *f = &job->fits[u];
    f->rss = best_rss > 0 ? best_rss : yy * 1e-12;
    for (uint32_t k = 0; k < best_k; k++) {
        size_t at = (size_t)u * K_MAX + k;
        p->link[at] = cs[k];
        p->index[at] = ci[k];
        f->a[k] = best_a[k];
        f->gkk[k] = cg[k];
        f->top[k] = max_magnitude(cx[k], len);
    }
    p->k[u] = (uint8_t)best_k;
}

static int select_tile(void *ctx, size_t tile) {
    select_job *job = ctx;
    uint64_t u0 = (uint64_t)tile * TILE_UNITS, units = job->plan->params.units;
    uint64_t u1 = u0 + TILE_UNITS < units ? u0 + TILE_UNITS : units;
    for (uint64_t u = u0; u < u1; u++)
        select_unit(job, u);
    return 0;
}

/* Estimated bits of a step-coded coefficient. */
static double coef_bits(int64_t q) {
    uint64_t mag = (uint64_t)(q < 0 ? -q : q);
    return (double)(bits_for(mag + 1)) + 2.0;
}

/* Exponent of source s: the E minimizing coefficient bits + quantization loss over its partners
 * (loss per coefficient ~ H / (2 ln 2) * step^2 / 12 * |x|^2 / RSS). */
static int32_t choose_exponent(const ingot_match_params *p, const unit_fit *fits, uint32_t s) {
    double amax = 0;
    for (uint64_t u = 0; u < p->units; u++)
        for (uint32_t k = 0; k < p->k[u]; k++)
            if (p->link[(size_t)u * K_MAX + k] == s && fabs(fits[u].a[k]) > amax)
                amax = fabs(fits[u].a[k]);
    if (!(amax > 0))
        return 0;
    double ehi = floor(log2((double)INGOT_MATCH_COEF_MAX / amax));
    ehi = ehi > INGOT_PREDICT_EXP_LIMIT ? INGOT_PREDICT_EXP_LIMIT : ehi;
    double best = 0;
    int32_t best_e = (int32_t)ehi;
    for (int32_t e = (int32_t)ehi; e >= (int32_t)ehi - EXP_SPAN && e > -INGOT_PREDICT_EXP_LIMIT;
         e--) {
        double step2 = ldexp(1.0, -2 * e), cost = 0;
        for (uint64_t u = 0; u < p->units; u++)
            for (uint32_t k = 0; k < p->k[u]; k++) {
                if (p->link[(size_t)u * K_MAX + k] != s)
                    continue;
                const unit_fit *f = &fits[u];
                cost += coef_bits((int64_t)nearbyint(ldexp(f->a[k], e))) +
                        (double)p->unit_len / (2.0 * M_LN2) * step2 / 12.0 * f->gkk[k] / f->rss;
            }
        if (e == (int32_t)ehi || cost < best) {
            best = cost;
            best_e = e;
        }
    }
    return best_e;
}

/* Exponents, int16 coefficients (zeros dropped) and C. */
static void quantize(ingot_match_params *p, const unit_fit *fits) {
    uint16_t top[INGOT_MATCH_MAX_SOURCES] = {0};
    for (uint32_t s = 0; s < p->source_count; s++)
        p->source[s].coef_exp = choose_exponent(p, fits, s);
    for (uint64_t u = 0; u < p->units; u++) {
        uint32_t kept = 0;
        for (uint32_t k = 0; k < p->k[u]; k++) {
            size_t at = (size_t)u * K_MAX + k;
            uint8_t s = p->link[at];
            double v = nearbyint(ldexp(fits[u].a[k], p->source[s].coef_exp));
            v = v > INGOT_MATCH_COEF_MAX ? INGOT_MATCH_COEF_MAX
                : v < -INGOT_MATCH_COEF_MAX ? -INGOT_MATCH_COEF_MAX : v;
            if (v == 0)
                continue;
            size_t to = (size_t)u * K_MAX + kept++;
            p->link[to] = s;
            p->index[to] = p->index[at];
            p->coef[to] = (int16_t)v;
            top[s] = fits[u].top[k] > top[s] ? fits[u].top[k] : top[s];
        }
        p->k[u] = (uint8_t)kept;
    }
    int32_t c = INGOT_PREDICT_EXP_LIMIT;
    for (uint32_t s = 0; s < p->source_count; s++) {
        int32_t hi = (int32_t)(top[s] >> 7) - 126; /* |x| < 2^hi */
        int32_t cs = p->source[s].coef_exp + INGOT_PREDICT_FIX_BITS - FIX_HEADROOM - hi;
        c = top[s] && cs < c ? cs : c;
    }
    p->scale_c = c < -INGOT_PREDICT_EXP_LIMIT ? -INGOT_PREDICT_EXP_LIMIT : c;
}

/* ---- Counts and class layout (as mode 6: fine counts per scale bin, merged per layout) ---- */

enum {
    FINE_HALF = 16,
    FINE_BINS = 2 * FINE_HALF + 1,
    FINE_CELLS = FINE_BINS * INGOT_PREDICT_BUCKETS * 2,
    SYMBOLS = 1024,
    FINE_COUNTS = FINE_CELLS * (1 + SYMBOLS),
};

typedef struct {
    const ingot_match_plan *plan;
    const ingot_match_model *flat;
    const uint8_t *bin;
    size_t chunks;
    uint64_t *counts;
} count_job;

static void block_range(const ingot_match_plan *p, size_t b, uint64_t *lo, uint64_t *hi) {
    *lo = (uint64_t)b * BLOCK;
    *hi = *lo + BLOCK < p->count ? *lo + BLOCK : p->count;
}

static int count_chunk(void *ctx, size_t k) {
    count_job *job = ctx;
    const ingot_match_plan *p = job->plan;
    uint64_t *counts = job->counts + k * (size_t)FINE_COUNTS, *symbols = counts + FINE_CELLS;
    uint8_t *cb = malloc(BLOCK);
    if (!cb)
        return -INGOT_ENOMEM;
    for (size_t b = k * p->block_count / job->chunks;
         b < (k + 1) * p->block_count / job->chunks; b++) {
        uint64_t lo = 0, hi = 0;
        block_range(p, b, &lo, &hi);
        ingot_match_contexts(job->flat, p->sources, lo, hi, cb);
        for (uint64_t i = lo; i < hi; i++) {
            uint64_t u = p->params.axis == INGOT_PREDICT_ROWS ? i / p->params.unit_len
                                                              : i % p->params.units;
            uint8_t c = cb[i - lo];
            uint16_t w = p->data[i];
            uint32_t cell = (job->bin[u] * INGOT_PREDICT_BUCKETS + (c & 0x7Fu)) * 2 +
                            (((uint32_t)w >> 15) ^ (uint32_t)(c >> 7));
            counts[cell]++;
            symbols[(size_t)cell * SYMBOLS + ingot_bf16_symbol(w)]++;
        }
    }
    free(cb);
    return 0;
}

static int32_t median_scale(const ingot_match_params *q) {
    uint64_t hist[256] = {0}, seen = 0;
    for (uint64_t u = 0; u < q->units; u++)
        hist[q->scale[u] + 128]++;
    for (int s = 0; s < 256; s++) {
        seen += hist[s];
        if (2 * seen >= q->units)
            return s - 128;
    }
    return 0;
}

static int count_fine(ingot_match_plan *p, int threads, uint64_t **out) {
    ingot_match_params flat = p->params;
    flat.classes = 1;
    flat.class_width = 1;
    ingot_match_model *m = NULL;
    size_t chunks = (size_t)(threads > 1 ? threads : 1);
    chunks = chunks < p->block_count ? chunks : p->block_count;
    uint8_t *bin = malloc((size_t)p->params.units);
    count_job job = {p, NULL, bin, chunks, calloc(chunks * (size_t)FINE_COUNTS, sizeof(uint64_t))};
    int rc = (bin && job.counts) ? ingot_match_model_create(&flat, &m) : -INGOT_ENOMEM;
    job.flat = m;
    for (uint64_t u = 0; !rc && u < p->params.units; u++) {
        int32_t b = p->params.scale[u] - p->params.class_ref + FINE_HALF;
        bin[u] = (uint8_t)(b < 0 ? 0 : b >= FINE_BINS ? FINE_BINS - 1 : b);
    }
    if (!rc)
        rc = ingot_parallel_for(chunks, threads, count_chunk, &job);
    for (size_t k = 1; !rc && k < chunks; k++)
        for (size_t i = 0; i < FINE_COUNTS; i++)
            job.counts[i] += job.counts[k * (size_t)FINE_COUNTS + i];
    ingot_match_model_free(m);
    free(bin);
    if (rc)
        free(job.counts);
    else
        *out = job.counts;
    return rc;
}

static int try_layout(const ingot_match_plan *p, const uint64_t *fine, uint32_t classes,
                      uint32_t width, ingot_fc_tables **tables, double *bits, double *bytes) {
    ingot_match_params q = p->params;
    q.classes = classes;
    q.class_width = width;
    ingot_match_model *m = NULL;
    int rc = ingot_match_model_create(&q, &m);
    const ingot_fc_model *fm = rc ? NULL : ingot_match_fc_model(m);
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
    ingot_match_model_free(m);
    return rc;
}

static int plan_count(ingot_match_plan *p, int threads) {
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

/* Side stream: count, build its tables, encode its one block. */
static int plan_side(ingot_match_plan *p) {
    ingot_fc_model sm;
    int rc = ingot_match_side_model(p->params.source_count, &sm);
    uint64_t *counts = rc ? NULL : calloc(sm.cell[sm.contexts], sizeof(uint64_t));
    rc = rc ? rc : counts ? 0 : -INGOT_ENOMEM;
    ingot_fcoder c;
    if (!rc) {
        ingot_fc_begin_count(&c, &sm, counts);
        ingot_match_side_code(&c, &p->params);
        rc = c.err;
    }
    if (!rc)
        rc = ingot_fc_tables_build(&sm, counts, &p->side_tables, NULL);
    free(counts);
    if (!rc) {
        ingot_fc_begin_encode(&c, p->side_tables);
        ingot_match_side_code(&c, &p->params);
        rc = ingot_fc_finish_encode(&c, &p->side_block);
    }
    return rc;
}

/* Drop sources no unit uses (keeping at least one), renumbering the links. */
static void prune_sources(ingot_match_plan *p) {
    ingot_match_params *q = &p->params;
    uint8_t used[INGOT_MATCH_MAX_SOURCES] = {0}, map[INGOT_MATCH_MAX_SOURCES] = {0};
    for (uint64_t u = 0; u < q->units; u++)
        for (uint32_t k = 0; k < q->k[u]; k++)
            used[q->link[(size_t)u * K_MAX + k]] = 1;
    uint32_t kept = 0;
    for (uint32_t s = 0; s < q->source_count; s++) {
        if (!used[s] && !(s + 1 == q->source_count && kept == 0))
            continue;
        map[s] = (uint8_t)kept;
        q->source[kept] = q->source[s];
        p->sources[kept++] = p->sources[s];
    }
    q->source_count = kept;
    for (uint64_t u = 0; u < q->units; u++)
        for (uint32_t k = 0; k < q->k[u]; k++)
            q->link[(size_t)u * K_MAX + k] = map[q->link[(size_t)u * K_MAX + k]];
}

static int plan_fit(ingot_match_plan *p, const uint16_t *unit_major, const ingot_match_input *src,
                    const ingot_match_candidates *cand, int threads) {
    unit_fit *fits = calloc((size_t)p->params.units, sizeof(*fits));
    if (!fits)
        return -INGOT_ENOMEM;
    select_job job = {p, unit_major, src, cand, fits};
    int rc = ingot_parallel_for((size_t)((p->params.units + TILE_UNITS - 1) / TILE_UNITS),
                                threads, select_tile, &job);
    if (!rc) {
        quantize(&p->params, fits);
        prune_sources(p);
    }
    free(fits);
    return rc;
}

int ingot_match_plan_create(ingot_predict_axis axis, uint64_t units, uint64_t unit_len,
                            const uint16_t *data, const uint16_t *unit_major,
                            const ingot_match_input *src, uint32_t source_count,
                            const ingot_match_candidates *cand, int threads,
                            ingot_match_plan **out) {
    if (!data || !unit_major || !src || !cand || !out || !source_count ||
        source_count > INGOT_MATCH_MAX_SOURCES)
        return -INGOT_EINVAL;
    ingot_match_plan *p = calloc(1, sizeof(*p));
    if (!p)
        return -INGOT_ENOMEM;
    ingot_match_params *q = &p->params;
    *q = (ingot_match_params){.axis = axis, .units = units, .unit_len = unit_len,
                              .source_count = source_count, .classes = 1, .class_width = 1};
    for (uint32_t s = 0; s < source_count; s++) {
        q->source[s] = (ingot_match_source){.ref = src[s].ref, .axis = src[s].axis,
                                            .units = src[s].units, .self = src[s].self};
        p->sources[s] = src[s].self ? data : src[s].data;
    }
    p->data = data;
    p->count = units * unit_len;
    p->block_count = (size_t)((p->count + BLOCK - 1) / BLOCK);
    int rc = ingot_match_params_alloc(q);
    if (!rc)
        rc = plan_fit(p, unit_major, src, cand, threads);
    if (!rc && ingot_match_params_check(q))
        rc = -INGOT_EINVAL;
    if (!rc)
        rc = plan_count(p, threads);
    if (!rc)
        rc = plan_side(p);
    if (!rc)
        rc = ingot_match_model_create(q, &p->model);
    if (rc) {
        ingot_match_plan_free(p);
        return rc;
    }
    *out = p;
    return 0;
}

uint64_t ingot_match_plan_prefix_bytes(const ingot_match_plan *p) {
    return ingot_match_prefix_bytes(p->params.source_count, p->side_tables, p->side_block.size,
                                    p->tables);
}

size_t ingot_match_plan_blocks(const ingot_match_plan *p) { return p->block_count; }

const ingot_match_params *ingot_match_plan_params(const ingot_match_plan *p) {
    return &p->params;
}

uint64_t ingot_match_plan_bytes(const ingot_match_plan *p) {
    uint64_t framing = (uint64_t)p->block_count * (INGOT_BLOCK_HEADER_BYTES + 8 + 8) + 8;
    return ingot_match_plan_prefix_bytes(p) + framing + (p->raw_bits + 7) / 8 +
           (uint64_t)(p->coded_bits / 8.0);
}

static int encode_block(void *ctx, size_t b, ingot_block_buf *out) {
    const ingot_match_plan *p = ctx;
    uint64_t lo = 0, hi = 0;
    block_range(p, b, &lo, &hi);
    uint8_t *cb = malloc((size_t)(hi - lo));
    if (!cb)
        return -INGOT_ENOMEM;
    ingot_match_contexts(p->model, p->sources, lo, hi, cb);
    ingot_fcoder c;
    ingot_fc_begin_encode(&c, p->tables);
    ingot_predict_code_z(&c, p->params.classes * INGOT_PREDICT_BUCKETS, cb, p->data + lo, NULL,
                         (size_t)(hi - lo));
    free(cb);
    return ingot_fc_finish_encode(&c, out);
}

int ingot_match_encode(const ingot_match_plan *p, int threads, ingot_sink *sink) {
    if (!p || !sink || sink->pos % 8)
        return -INGOT_EINVAL;
    uint64_t prefix = ingot_match_plan_prefix_bytes(p);
    uint8_t *buf = calloc(1, (size_t)prefix);
    if (!buf)
        return -INGOT_ENOMEM;
    ingot_match_prefix_write(&p->params, p->block_count,
                             ingot_match_fc_model(p->model)->contexts, p->side_tables,
                             &p->side_block, p->tables, buf);
    int rc = ingot_sink_write(sink, buf, (size_t)prefix);
    free(buf);
    if (!rc)
        rc = ingot_block_area_write(sink, p->block_count, threads, encode_block, (void *)p);
    return rc;
}

void ingot_match_plan_free(ingot_match_plan *p) {
    if (!p)
        return;
    ingot_fc_tables_free(p->tables);
    ingot_fc_tables_free(p->side_tables);
    free(p->side_block.data);
    ingot_match_model_free(p->model);
    ingot_match_params_release(&p->params);
    free(p);
}
