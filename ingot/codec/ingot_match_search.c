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
 * what codes smallest. Sketches are int16 so the dot products run as 16-bit multiply-adds
 * (SSE2 pmaddwd, NEON vmlal; scalar elsewhere); every path gives the same integer sums.
 */
#include "ingot_match.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_parallel.h"

enum {
    DIM = INGOT_MATCH_SKETCH_DIM,
    VEC = 4,             /* floats per vector register (projection) */
    TARGET_TILE = 64,    /* target units per work item */
    NO_SOURCE = 0xFF,
    QUANT_ONE = 256,     /* a unit-variance sketch coordinate -> 256 */
    QUANT_MAX = 2047,    /* clip at 8 sigma: |sum| < DIM * 2047^2 < 2^31 */
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
    int16_t *out;
    const v4f *signs; /* [len][DIM / VEC]: +-1 */
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
        /* coordinate d ~ N(0, 1) for a unit vector; dot / (256^2 DIM) estimates the cosine */
        float scale = norm > 0 ? (float)(QUANT_ONE / sqrt(norm)) : 0.0f;
        for (int d = 0; d < DIM; d++) {
            float q = nearbyintf(acc[d / VEC][d % VEC] * scale);
            q = q > QUANT_MAX ? QUANT_MAX : q < -QUANT_MAX ? -QUANT_MAX : q;
            job->out[u * DIM + d] = (int16_t)q;
        }
    }
    return 0;
}

int ingot_match_sketch(const uint16_t *unit_major, uint64_t n, uint64_t unit_len, int threads,
                       int16_t *out) {
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

/* ---- Dot products: four targets against one pool unit ---- */

#if defined(__SSE2__)
static inline int32_t hsum(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));
    return _mm_cvtsi128_si32(v);
}

static inline __m128i load8(const int16_t *q) {
    return _mm_loadu_si128((const __m128i *)(const void *)q);
}

static void dot4(const int16_t *t, const int16_t *p, int32_t *out) {
    __m128i s0 = _mm_setzero_si128(), s1 = s0, s2 = s0, s3 = s0;
    for (int i = 0; i < DIM; i += 8) {
        __m128i x = load8(p + i);
        s0 = _mm_add_epi32(s0, _mm_madd_epi16(x, load8(t + i)));
        s1 = _mm_add_epi32(s1, _mm_madd_epi16(x, load8(t + DIM + i)));
        s2 = _mm_add_epi32(s2, _mm_madd_epi16(x, load8(t + 2 * DIM + i)));
        s3 = _mm_add_epi32(s3, _mm_madd_epi16(x, load8(t + 3 * DIM + i)));
    }
    out[0] = hsum(s0), out[1] = hsum(s1), out[2] = hsum(s2), out[3] = hsum(s3);
}
#elif defined(__ARM_NEON)
static void dot4(const int16_t *t, const int16_t *p, int32_t *out) {
    int32x4_t s0 = vdupq_n_s32(0), s1 = s0, s2 = s0, s3 = s0;
    for (int i = 0; i < DIM; i += 8) {
        int16x8_t x = vld1q_s16(p + i);
        int16x8_t a = vld1q_s16(t + i), b = vld1q_s16(t + DIM + i);
        int16x8_t c = vld1q_s16(t + 2 * DIM + i), d = vld1q_s16(t + 3 * DIM + i);
        s0 = vmlal_s16(vmlal_s16(s0, vget_low_s16(x), vget_low_s16(a)), vget_high_s16(x),
                       vget_high_s16(a));
        s1 = vmlal_s16(vmlal_s16(s1, vget_low_s16(x), vget_low_s16(b)), vget_high_s16(x),
                       vget_high_s16(b));
        s2 = vmlal_s16(vmlal_s16(s2, vget_low_s16(x), vget_low_s16(c)), vget_high_s16(x),
                       vget_high_s16(c));
        s3 = vmlal_s16(vmlal_s16(s3, vget_low_s16(x), vget_low_s16(d)), vget_high_s16(x),
                       vget_high_s16(d));
    }
    out[0] = vaddvq_s32(s0), out[1] = vaddvq_s32(s1), out[2] = vaddvq_s32(s2);
    out[3] = vaddvq_s32(s3);
}
#else
static void dot4(const int16_t *t, const int16_t *p, int32_t *out) {
    int32_t s[4] = {0};
    for (int i = 0; i < DIM; i++)
        for (int k = 0; k < 4; k++)
            s[k] += (int32_t)t[k * DIM + i] * p[i];
    memcpy(out, s, sizeof(s));
}
#endif

/* ---- Top-M search ---- */

typedef struct {
    const int16_t *target;
    uint64_t units;
    const ingot_match_pool *pool;
    uint32_t pool_count;
    ingot_match_candidates *cand;
} search_job;

/* Insert (score, source, index) into a unit's list (sorted descending by score). */
static void list_insert(int32_t *score, uint8_t *src, uint64_t *idx, uint32_t m, int32_t s,
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

static void consider(const search_job *job, int32_t *best, uint64_t u0, uint64_t u,
                     const ingot_match_pool *pl, uint32_t s, uint64_t i, int32_t dot) {
    uint32_t m = job->cand->per_unit;
    int32_t a = dot < 0 ? -dot : dot;
    if (a > best[(u - u0) * m + m - 1] && (!pl->self || i < u))
        list_insert(best + (u - u0) * m, job->cand->source + u * m, job->cand->index + u * m, m,
                    a, (uint8_t)s, i);
}

/* Target units [u0, u1) (padded to a multiple of 4 with zero sketches) against every pool unit. */
static void score_tile(const search_job *job, const int16_t *tile, uint64_t u0, uint64_t u1,
                       int32_t *best) {
    uint64_t n = u1 - u0;
    for (uint32_t s = 0; s < job->pool_count; s++) {
        const ingot_match_pool *pl = &job->pool[s];
        uint64_t limit = pl->self ? (u1 < pl->units ? u1 : pl->units) : pl->units;
        for (uint64_t i = 0; i < limit; i++) {
            const int16_t *pv = pl->sketch + i * DIM;
            for (uint64_t k = 0; k < n; k += 4) {
                int32_t dot[4];
                dot4(tile + k * DIM, pv, dot);
                for (uint64_t j = 0; j < 4 && k + j < n; j++)
                    consider(job, best, u0, u0 + k + j, pl, s, i, dot[j]);
            }
        }
    }
}

static int search_chunk(void *ctx, size_t chunk) {
    search_job *job = ctx;
    uint32_t m = job->cand->per_unit;
    uint64_t u0 = (uint64_t)chunk * TARGET_TILE;
    uint64_t u1 = u0 + TARGET_TILE < job->units ? u0 + TARGET_TILE : job->units;
    int32_t *best = malloc((size_t)TARGET_TILE * m * sizeof(*best));
    int16_t *tile = calloc((size_t)TARGET_TILE * DIM, sizeof(*tile));
    if (!best || !tile) {
        free(best);
        free(tile);
        return -INGOT_ENOMEM;
    }
    memcpy(tile, job->target + u0 * DIM, (size_t)(u1 - u0) * DIM * sizeof(*tile));
    for (size_t i = 0; i < (size_t)TARGET_TILE * m; i++)
        best[i] = -1;
    for (uint64_t u = u0; u < u1; u++)
        for (uint32_t j = 0; j < m; j++)
            job->cand->source[u * m + j] = NO_SOURCE;
    score_tile(job, tile, u0, u1, best);
    free(best);
    free(tile);
    return 0;
}

int ingot_match_search(const int16_t *target_sketch, uint64_t units,
                       const ingot_match_pool *pool, uint32_t pool_count, int threads,
                       ingot_match_candidates *cand) {
    if (!cand || !cand->per_unit || (pool_count && !pool) ||
        pool_count > INGOT_MATCH_MAX_SOURCES)
        return -INGOT_EINVAL;
    search_job job = {target_sketch, units, pool, pool_count, cand};
    return ingot_parallel_for((size_t)((units + TARGET_TILE - 1) / TARGET_TILE), threads,
                              search_chunk, &job);
}
