/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_attention_splitk_model.c
 * @brief Split-K attention models and the float64 reference. See
 *   ft_tt_attention_splitk_model.h.
 *
 * Every `keep()` below is one place the compute kernel packs a value into a circular buffer
 * (resident_attention_compute.c); the function named beside it is the kernel step.
 *
 * @status Used by tests/test_attention_splitk.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_attention_splitk_model.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_op_attention.h"
#include "../kernels/ft_kv_block_float.h"

#define TILE_DIM      32u
#define MODEL_MAX_DIM 256u

/** One chunk's partial: un-normalised O, its max and its sum, all as stored. */
typedef struct splitk_partial {
    float o[MODEL_MAX_DIM];
    float m;
    float l;
} splitk_partial;

static float round_bf16(float value)
{
    return ft_float_from_bf16(ft_bf16_from_float(value));
}

/** The value as the kernel holds it after packing it into a CB. */
static float keep(float value, ft_tt_splitk_store store)
{
    return store == FT_TT_SPLITK_STORE_BF16 ? round_bf16(value) : value;
}

static bool problem_is_valid(const ft_tt_splitk_problem *problem)
{
    if (!problem || !problem->live || !problem->query || !problem->keys || !problem->values)
        return false;
    if (problem->rows == 0u || problem->head_dim == 0u || problem->head_dim > MODEL_MAX_DIM ||
        problem->head_dim % TILE_DIM != 0u)
        return false;
    for (uint32_t r = 0; r < problem->rows; r++)
        if (problem->live[r] == 0u || problem->live[r] > problem->capacity_positions)
            return false;
    return true;
}

static uint32_t longest_live(const ft_tt_splitk_problem *problem)
{
    uint32_t longest = 0u;

    for (uint32_t r = 0; r < problem->rows; r++)
        if (problem->live[r] > longest)
            longest = problem->live[r];
    return longest;
}

static uint32_t shortest_live(const ft_tt_splitk_problem *problem)
{
    uint32_t shortest = UINT32_MAX;

    for (uint32_t r = 0; r < problem->rows; r++)
        if (problem->live[r] < shortest)
            shortest = problem->live[r];
    return shortest;
}

/** scale_query: Qs = Q * bf16(1/sqrt(d)), packed to cb_qs (always bf16: the CB is). */
static void scaled_query(const ft_tt_splitk_problem *problem, uint32_t row, float *out)
{
    const float scale = round_bf16((float)(1.0 / sqrt((double)problem->head_dim)));

    for (uint32_t e = 0; e < problem->head_dim; e++)
        out[e] = round_bf16(problem->query[(size_t)row * problem->head_dim + e] * scale);
}

bool ft_tt_splitk_reference(const ft_tt_splitk_problem *problem, float *out)
{
    float qs[MODEL_MAX_DIM];

    if (!problem_is_valid(problem) || !out)
        return false;
    for (uint32_t r = 0; r < problem->rows; r++) {
        double maximum = -HUGE_VAL;
        double total = 0.0;
        double acc[MODEL_MAX_DIM];

        scaled_query(problem, r, qs);
        for (uint32_t pass = 0; pass < 2u; pass++) {
            for (uint32_t p = 0; p < problem->live[r]; p++) {
                const float *key = problem->keys + (size_t)p * problem->head_dim;
                const float *value = problem->values + (size_t)p * problem->head_dim;
                double score = 0.0;

                for (uint32_t e = 0; e < problem->head_dim; e++)
                    score += (double)qs[e] * (double)key[e];
                if (pass == 0u) {
                    if (score > maximum)
                        maximum = score;
                    continue;
                }
                if (p == 0u)
                    memset(acc, 0, sizeof(acc));
                score = exp(score - maximum);
                total += score;
                for (uint32_t e = 0; e < problem->head_dim; e++)
                    acc[e] += score * (double)value[e];
            }
        }
        for (uint32_t e = 0; e < problem->head_dim; e++)
            out[(size_t)r * problem->head_dim + e] = (float)(acc[e] / total);
    }
    return true;
}

/** score_block: one position's score, packed to cb_s; a dead one takes the additive mask
 *  first (the mask bounce through cb_scratch, then + mask, packed to cb_s). */
static float block_score(const ft_tt_splitk_problem *problem, const float *qs, uint32_t position,
                         uint32_t dead_from, ft_tt_splitk_store store)
{
    const float *key = problem->keys + (size_t)position * problem->head_dim;
    float score = 0.0f;

    for (uint32_t e = 0; e < problem->head_dim; e++)
        score += qs[e] * key[e];
    score = keep(score, store);
    if (position >= dead_from)
        score = keep(score + FT_TT_OP_ATTENTION_MASK_VALUE, store);
    return score;
}

/** merge_block: c = e^(m_old - m_new); l = c*l + s; O = c*O + pv, each packed. */
static void merge_into(splitk_partial *acc, float m_new, float s, const float *pv, uint32_t dim,
                       ft_tt_splitk_store store)
{
    const float c = keep(expf(acc->m - m_new), store);

    acc->l = keep(keep(c * acc->l, store) + s, store);
    for (uint32_t e = 0; e < dim; e++)
        acc->o[e] = keep(keep(c * acc->o[e], store) + pv[e], store);
    acc->m = m_new;
}

/** seed_accumulators (first == true) or merge_block. */
static void fold(splitk_partial *acc, bool first, float m_new, float s, const float *pv,
                 uint32_t dim, ft_tt_splitk_store store)
{
    if (first) {
        acc->m = m_new;
        acc->l = s;
        memcpy(acc->o, pv, dim * sizeof(float));
        return;
    }
    merge_into(acc, m_new, s, pv, dim, store);
}

/**
 * compute_one_attention's block loop for one row over chunk tiles [first, first + count):
 * the padded window rounds the chunk's live tiles up to whole double blocks, and a position
 * at or past `dead_from` (the row's limit inside this descriptor) is masked.
 */
static void chunk_partial(const ft_tt_splitk_problem *problem, uint32_t row, uint32_t first_tile,
                          uint32_t live_tiles, uint32_t dead_from, uint32_t block_tiles,
                          ft_tt_splitk_store store, splitk_partial *out)
{
    const uint32_t dim = problem->head_dim;
    const uint32_t padded = ft_tt_op_attention_padded_tiles(live_tiles * TILE_DIM, block_tiles);
    const uint32_t block_positions = block_tiles * TILE_DIM;
    float qs[MODEL_MAX_DIM];
    float scores[TILE_DIM * 64u];
    float pv[MODEL_MAX_DIM];

    scaled_query(problem, row, qs);
    for (uint32_t block = 0; block * block_tiles < padded; block++) {
        const uint32_t start = first_tile * TILE_DIM + block * block_positions;
        float m_new = -HUGE_VALF;
        float s = 0.0f;

        for (uint32_t i = 0; i < block_positions; i++) {
            scores[i] = block_score(problem, qs, start + i, dead_from, store);
            if (scores[i] > m_new)
                m_new = scores[i];
        }
        if (block != 0u && out->m > m_new) /* running_max: seeded with the previous max */
            m_new = out->m;
        m_new = keep(m_new, store);
        memset(pv, 0, sizeof(pv));
        for (uint32_t i = 0; i < block_positions; i++) {
            const float *value = problem->values + (size_t)(start + i) * dim;
            const float p = keep(expf(scores[i] - m_new), store); /* exponentiate_block */

            s += p;
            for (uint32_t e = 0; e < dim; e++)
                pv[e] += p * value[e];
        }
        s = keep(s, store); /* block_row_sum */
        for (uint32_t e = 0; e < dim; e++)
            pv[e] = keep(pv[e], store); /* multiply_by_values */
        fold(out, block == 0u, m_new, s, pv, dim, store);
    }
}

/** normalise_and_emit: 1/l packed, then O * (1/l) packed. */
static void normalise(const splitk_partial *acc, uint32_t dim, ft_tt_splitk_store store,
                      float *out)
{
    const float reciprocal = keep(1.0f / acc->l, store);

    for (uint32_t e = 0; e < dim; e++)
        out[e] = keep(acc->o[e] * reciprocal, store);
}

/** The padded windows read up to the live tiles rounded to double blocks; that must fit. */
static bool window_fits(const ft_tt_splitk_problem *problem, uint32_t first_tile,
                        uint32_t live_tiles, uint32_t block_tiles)
{
    const uint32_t padded = ft_tt_op_attention_padded_tiles(live_tiles * TILE_DIM, block_tiles);

    return block_tiles != 0u && block_tiles <= 64u &&
           (first_tile + padded) * TILE_DIM <= problem->capacity_positions;
}

bool ft_tt_splitk_lanes_model(const ft_tt_splitk_problem *problem, uint32_t block_tiles,
                              ft_tt_splitk_store store, float *out)
{
    splitk_partial acc;
    uint32_t live_tiles;

    if (!problem_is_valid(problem) || !out)
        return false;
    live_tiles = (longest_live(problem) + TILE_DIM - 1u) / TILE_DIM;
    if (!window_fits(problem, 0u, live_tiles, block_tiles))
        return false;
    for (uint32_t r = 0; r < problem->rows; r++) {
        chunk_partial(problem, r, 0u, live_tiles, problem->live[r], block_tiles, store, &acc);
        normalise(&acc, problem->head_dim, store, out + (size_t)r * problem->head_dim);
    }
    return true;
}

bool ft_tt_splitk_chunks_are_valid(const ft_tt_splitk_chunk *chunks, uint32_t chunk_count,
                                   uint32_t live_tiles, uint32_t last_tile_first)
{
    uint32_t next = 0u;

    if (!chunks || chunk_count == 0u || chunk_count > FT_TT_OP_ATTENTION_MAX_PARTIALS)
        return false;
    for (uint32_t c = 0; c < chunk_count; c++) {
        if (chunks[c].tile_count == 0u || chunks[c].first_tile != next)
            return false;
        next += chunks[c].tile_count;
    }
    return next == live_tiles && chunks[chunk_count - 1u].first_tile <= last_tile_first;
}

/** MERGE: partial i staged as one more block (c_i = e^(m_i - m'), l_i and O_i scaled and
 *  packed), then seed_accumulators / merge_block; out = O / l. */
static void merge_partials(const splitk_partial *partials, uint32_t count, uint32_t skip,
                           uint32_t dim, ft_tt_splitk_store store, float *out)
{
    splitk_partial acc;
    float pv[MODEL_MAX_DIM];
    bool first = true;

#ifdef FT_TT_SPLITK_RED_DROP
    /* RED (make -f Makefile.attention red-splitk): the last chunk's partial never merges. */
    if (skip == UINT32_MAX && count > 1u)
        skip = count - 1u;
#endif
    for (uint32_t i = 0; i < count; i++) {
        const splitk_partial *part = &partials[i];
        float m_new;
        float c;

        if (i == skip)
            continue;
        m_new = first || part->m > acc.m ? part->m : acc.m;
        c = keep(expf(part->m - m_new), store);
        for (uint32_t e = 0; e < dim; e++)
            pv[e] = keep(part->o[e] * c, store);
        fold(&acc, first, m_new, keep(part->l * c, store), pv, dim, store);
        first = false;
    }
    if (first) { /* every partial skipped: nothing merged */
        memset(out, 0, dim * sizeof(float));
        return;
    }
    normalise(&acc, dim, store, out);
}

bool ft_tt_splitk_model(const ft_tt_splitk_problem *problem, uint32_t block_tiles,
                        const ft_tt_splitk_chunk *chunks, uint32_t chunk_count,
                        ft_tt_splitk_store store, uint32_t drop_chunk, float *out)
{
    static splitk_partial partials[FT_TT_OP_ATTENTION_MAX_PARTIALS];
    uint32_t live_tiles;

    if (!problem_is_valid(problem) || !out)
        return false;
    live_tiles = (longest_live(problem) + TILE_DIM - 1u) / TILE_DIM;
    if (!ft_tt_splitk_chunks_are_valid(chunks, chunk_count, live_tiles,
                                       (shortest_live(problem) - 1u) / TILE_DIM))
        return false;
    for (uint32_t c = 0; c < chunk_count; c++)
        if (!window_fits(problem, chunks[c].first_tile, chunks[c].tile_count, block_tiles))
            return false;
    for (uint32_t r = 0; r < problem->rows; r++) {
        for (uint32_t c = 0; c < chunk_count; c++) {
            const uint32_t end = (chunks[c].first_tile + chunks[c].tile_count) * TILE_DIM;
            /* A non-final chunk is entirely live for every row (its descriptor carries no
             * mask past its own end); the final one holds every row's causal limit. */
            const uint32_t dead_from = problem->live[r] < end ? problem->live[r] : end;

            chunk_partial(problem, r, chunks[c].first_tile, chunks[c].tile_count, dead_from,
                          block_tiles, store, &partials[c]);
        }
        merge_partials(partials, chunk_count, drop_chunk, problem->head_dim, store,
                       out + (size_t)r * problem->head_dim);
    }
    return true;
}

void ft_tt_splitk_round_block_float(float *values, uint32_t count, bool bfp4)
{
    const uint32_t format = bfp4 ? (uint32_t)kKvFormatBfp4 : (uint32_t)kKvFormatBfp8;

    for (uint32_t row = 0; row + (uint32_t)kKvBfRowValues <= count; row += kKvBfRowValues) {
        uint16_t bf16[kKvBfRowValues];
        uint8_t datums[kKvBfRowValues];
        uint32_t shared;

        for (uint32_t i = 0; i < (uint32_t)kKvBfRowValues; i++)
            bf16[i] = ft_bf16_from_float(values[row + i]);
        shared = ft_kv_bf_pack_row(format, bf16, datums);
        for (uint32_t i = 0; i < (uint32_t)kKvBfRowValues; i++) {
            const uint32_t datum = bfp4 ? (uint32_t)(datums[i >> 1] >> ((i & 1u) * 4u)) & 0xFu
                                        : (uint32_t)datums[i];
            const uint32_t bits = ft_kv_bf_datum_to_fp32(format, datum, shared);
            float value;

            memcpy(&value, &bits, sizeof(value));
            values[row + i] = value;
        }
    }
}
