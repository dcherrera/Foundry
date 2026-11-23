/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_match_search.c
 * @brief  Match mode candidate search (encoder only): random-projection sketches of unit vectors
 *         and, per target unit, the pool units with the largest |sketch dot| (see ingot_match.h).
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Nothing here is part of the format: the search only proposes candidates, and the plan keeps
 * what codes smallest. Floating point is fine; results may differ across machines.
 */
#include "ingot_match.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_parallel.h"

enum {
    DIM = INGOT_MATCH_SKETCH_DIM,
    VEC = 4,             /* floats per vector register */
    TARGET_TILE = 64,    /* target units per work item */
    POOL_TILE = 256,     /* pool units scored per pass over a target tile */
    NO_SOURCE = 0xFF,
};

typedef float v4f __attribute__((vector_size(16)));

/* Sign bits (dimensions 64 word .. 64 word + 63, channel h) of the fixed projection: the
 * splitmix64 finalizer of (h, word), so every bit is a well-mixed function of the channel (a
 * linear xorshift of h K left low bits periodic in h and the sketch blind to structured rows). */
static uint64_t sign_word(uint64_t h, uint32_t word) {
    uint64_t x = h * 0x9E3779B97F4A7C15ull + ((uint64_t)word + 1) * 0xD1B54A32D192ED03ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

typedef struct {
    const uint16_t *units;
    uint64_t len;
    float *out;
    const v4f *signs;      /* [len][DIM / VEC]: +-1 */
    uint64_t n;
} sketch_job;

static int sketch_chunk(void *ctx, size_t chunk) {
    sketch_job *job = ctx;
    uint64_t u0 = (uint64_t)chunk * TARGET_TILE;
    uint64_t u1 = u0 + TARGET_TILE < job->n ? u0 + TARGET_TILE : job->n;
    for (uint64_t u = u0; u < u1; u++) {
        const uint16_t *x = job->units + u * job->len;
        v4f acc[DIM / VEC] = {0};
        double norm = 0;
        for (uint64_t h = 0; h < job->len; h++) {
            uint16_t w = x[h];
            float v = (w & 0x7F80u) == 0x7F80u ? 0.0f : ingot_bf16_to_f32(w);
            if (v == 0.0f)
                continue;
            norm += (double)v * v;
            const v4f *s = job->signs + h * (DIM / VEC);
            v4f vv = {v, v, v, v};
            for (int i = 0; i < DIM / VEC; i++)
                acc[i] += vv * s[i];
        }
        float scale = norm > 0 ? (float)(1.0 / sqrt(norm * DIM)) : 0.0f;
        for (int d = 0; d < DIM; d++)
            job->out[u * DIM + d] = acc[d / VEC][d % VEC] * scale;
    }
    return 0;
}

int ingot_match_sketch(const uint16_t *unit_major, uint64_t n, uint64_t unit_len, int threads,
                       float *out) {
    v4f *signs = malloc((size_t)unit_len * (DIM / VEC) * sizeof(*signs));
    if (!signs)
        return -INGOT_ENOMEM;
    for (uint64_t h = 0; h < unit_len; h++)
        for (int d = 0; d < DIM; d++)
            signs[h * (DIM / VEC) + d / VEC][d % VEC] =
                ((sign_word(h, (uint32_t)(d / 64)) >> (d % 64)) & 1) ? 1.0f : -1.0f;
    sketch_job job = {unit_major, unit_len, out, signs, n};
    int rc = ingot_parallel_for((size_t)((n + TARGET_TILE - 1) / TARGET_TILE), threads,
                                sketch_chunk, &job);
    free(signs);
    return rc;
}

/* ---- Top-M search ---- */

typedef struct {
    const float *target;
    uint64_t units;
    const ingot_match_pool *pool;
    uint32_t pool_count;
    ingot_match_candidates *cand;
} search_job;

/* Insert (score, source, index) into unit u's list (sorted descending by score). */
static void list_insert(float *score, uint8_t *src, uint64_t *idx, uint32_t m, float s,
                        uint8_t source, uint64_t index) {
    uint32_t at = m - 1;
    while (at > 0 && score[at - 1] < s) {
        score[at] = score[at - 1];
        src[at] = src[at - 1];
        idx[at] = idx[at - 1];
        at--;
    }
    score[at] = s;
    src[at] = source;
    idx[at] = index;
}

static inline float dot(const v4f *a, const v4f *b) {
    v4f s0 = {0}, s1 = {0};
    for (int i = 0; i < DIM / VEC; i += 2) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
    }
    s0 += s1;
    return s0[0] + s0[1] + s0[2] + s0[3];
}

/* Four target units against one pool unit. */
static inline void dot4(const v4f *t0, const v4f *t1, const v4f *t2, const v4f *t3,
                        const v4f *p, float *out) {
    v4f s0 = {0}, s1 = {0}, s2 = {0}, s3 = {0};
    for (int i = 0; i < DIM / VEC; i++) {
        v4f x = p[i];
        s0 += t0[i] * x;
        s1 += t1[i] * x;
        s2 += t2[i] * x;
        s3 += t3[i] * x;
    }
    out[0] = s0[0] + s0[1] + s0[2] + s0[3];
    out[1] = s1[0] + s1[1] + s1[2] + s1[3];
    out[2] = s2[0] + s2[1] + s2[2] + s2[3];
    out[3] = s3[0] + s3[1] + s3[2] + s3[3];
}

static void score_tile(const search_job *job, uint64_t u0, uint64_t u1, float *best) {
    uint32_t m = job->cand->per_unit;
    for (uint32_t s = 0; s < job->pool_count; s++) {
        const ingot_match_pool *pl = &job->pool[s];
        uint64_t limit = pl->self ? (u1 < pl->units ? u1 : pl->units) : pl->units;
        for (uint64_t i = 0; i < limit; i++) {
            const v4f *pv = (const v4f *)(const void *)(pl->sketch + i * DIM);
            uint64_t u = u0;
            for (; u + 4 <= u1; u += 4) {
                const v4f *t = (const v4f *)(const void *)(job->target + u * DIM);
                float sc[4];
                dot4(t, t + DIM / VEC, t + 2 * DIM / VEC, t + 3 * DIM / VEC, pv, sc);
                for (int j = 0; j < 4; j++) {
                    float a = fabsf(sc[j]);
                    uint64_t uu = u + (uint64_t)j;
                    if (a > best[(uu - u0) * m + m - 1] && (!pl->self || i < uu))
                        list_insert(best + (uu - u0) * m, job->cand->source + uu * m,
                                    job->cand->index + uu * m, m, a, (uint8_t)s, i);
                }
            }
            for (; u < u1; u++) {
                float a = fabsf(dot((const v4f *)(const void *)(job->target + u * DIM), pv));
                if (a > best[(u - u0) * m + m - 1] && (!pl->self || i < u))
                    list_insert(best + (u - u0) * m, job->cand->source + u * m,
                                job->cand->index + u * m, m, a, (uint8_t)s, i);
            }
        }
    }
}

static int search_chunk(void *ctx, size_t chunk) {
    search_job *job = ctx;
    uint32_t m = job->cand->per_unit;
    uint64_t u0 = (uint64_t)chunk * TARGET_TILE;
    uint64_t u1 = u0 + TARGET_TILE < job->units ? u0 + TARGET_TILE : job->units;
    float *best = malloc((size_t)TARGET_TILE * m * sizeof(*best));
    if (!best)
        return -INGOT_ENOMEM;
    for (size_t i = 0; i < (size_t)TARGET_TILE * m; i++)
        best[i] = -1.0f;
    for (uint64_t u = u0; u < u1; u++)
        for (uint32_t j = 0; j < m; j++)
            job->cand->source[u * m + j] = NO_SOURCE;
    score_tile(job, u0, u1, best);
    free(best);
    return 0;
}

int ingot_match_search(const float *target_sketch, uint64_t units, const ingot_match_pool *pool,
                       uint32_t pool_count, int threads, ingot_match_candidates *cand) {
    if (!cand || !cand->per_unit || (pool_count && !pool) ||
        pool_count > INGOT_MATCH_MAX_SOURCES)
        return -INGOT_EINVAL;
    search_job job = {target_sketch, units, pool, pool_count, cand};
    return ingot_parallel_for((size_t)((units + TARGET_TILE - 1) / TARGET_TILE), threads,
                              search_chunk, &job);
}
