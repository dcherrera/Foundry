/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_attention_reference.c
 * @brief Implementation of the attention reference and the blocked online model. See
 *   ft_tt_attention_reference.h for why there are two of them.
 *
 * @status Used by both attention suites.
 * @issues None
 * @todo None
 */

#include "ft_tt_attention_reference.h"

#include <math.h>
#include <string.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_op_attention.h"

#define TILE_DIM 32u

/** The largest group and head dimension the model's fixed-size state serves. 32 rows is a
 *  tile and 256 dimensions is twice MiniCPM5-1B's head_dim. */
#define MODEL_MAX_ROWS 32u
#define MODEL_MAX_DIM  256u

static uint64_t g_random_state = 0x243F6A8885A308D3ull;

void ft_tt_attention_seed(uint64_t seed)
{
    g_random_state = seed | 1ull;
}

float ft_tt_attention_random(void)
{
    g_random_state = g_random_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((double)(g_random_state >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

/** bf16 round-trip: what the card stores after any pack. */
static float round_bf16(float value)
{
    return ft_float_from_bf16(ft_bf16_from_float(value));
}

/** The query scale, applied the way the kernel applies it: multiply by the bf16 scale, then
 *  round, because the kernel's multiply packs its result to bf16. */
static float scaled_query(float value, float scale)
{
    return round_bf16(value * scale);
}

static bool problem_is_valid(const ft_tt_attention_problem *problem)
{
    if (!problem || !problem->query || !problem->keys || !problem->values)
        return false;
    if (problem->q_rows == 0u || problem->q_rows > MODEL_MAX_ROWS)
        return false;
    if (problem->head_dim == 0u || problem->head_dim > MODEL_MAX_DIM)
        return false;
    if (problem->head_dim % TILE_DIM != 0u)
        return false;
    return problem->kv_positions != 0u;
}

bool ft_tt_attention_reference(const ft_tt_attention_problem *problem, float *out)
{
    const float scale = round_bf16((float)(1.0 / sqrt((double)(problem ? problem->head_dim : 1u))));
    uint32_t row;

    if (!problem_is_valid(problem) || !out)
        return false;

    for (row = 0; row < problem->q_rows; row++) {
        const float *query = problem->query + (size_t)row * problem->head_dim;
        double maximum = -HUGE_VAL;
        double total = 0.0;
        uint32_t position;
        uint32_t element;

        /* Two passes over the row, which is exactly what the card cannot afford to do and
         * what a reference is free to do. */
        for (position = 0; position < problem->kv_positions; position++) {
            const float *key = problem->keys + (size_t)position * problem->head_dim;
            double score = 0.0;

            for (element = 0; element < problem->head_dim; element++)
                score += (double)scaled_query(query[element], scale) * (double)key[element];
            if (score > maximum)
                maximum = score;
        }

        for (element = 0; element < problem->head_dim; element++)
            out[(size_t)row * problem->head_dim + element] = 0.0f;

        for (position = 0; position < problem->kv_positions; position++) {
            const float *key = problem->keys + (size_t)position * problem->head_dim;
            const float *value = problem->values + (size_t)position * problem->head_dim;
            double score = 0.0;
            double weight;

            for (element = 0; element < problem->head_dim; element++)
                score += (double)scaled_query(query[element], scale) * (double)key[element];
            weight = exp(score - maximum);
            total += weight;
            for (element = 0; element < problem->head_dim; element++)
                out[(size_t)row * problem->head_dim + element] +=
                    (float)(weight * (double)value[element]);
        }

        for (element = 0; element < problem->head_dim; element++)
            out[(size_t)row * problem->head_dim + element] =
                (float)((double)out[(size_t)row * problem->head_dim + element] / total);
    }
    return true;
}

bool ft_tt_attention_chunk_reference(const ft_tt_attention_problem *problem,
                                     uint32_t chunk_first, float *out)
{
    uint32_t row;

    if (!problem_is_valid(problem) || !out)
        return false;
    /* The chunk ENDS the live range: row r is position `chunk_first + r`, and the last row
     * attends to everything. Anything else is a caller that has not decided what its rows
     * mean. */
    if (problem->kv_positions != chunk_first + problem->q_rows)
        return false;

    /* ONE ROW AT A TIME THROUGH THE DEFINITIONAL REFERENCE, rather than a second copy of
     * the softmax with a variable bound. Causality is then expressed entirely as a LIVE
     * LENGTH — row r sees positions 0..chunk_first+r — which is the same sentence the
     * causal mask tile encodes, arrived at independently of it. */
    for (row = 0; row < problem->q_rows; row++) {
        ft_tt_attention_problem one = *problem;

        one.q_rows = 1u;
        one.query = problem->query + (size_t)row * problem->head_dim;
        one.kv_positions = chunk_first + row + 1u;
        if (!ft_tt_attention_reference(&one, out + (size_t)row * problem->head_dim))
            return false;
    }
    return true;
}

/** One position's score against one query row, in the padded window. Positions at or past
 *  the live length are DEAD and get the mask value, exactly as the kernel's additive mask
 *  drives them. */
static double model_score(const ft_tt_attention_problem *problem, const float *query,
                          uint32_t position, float scale)
{
    const float *key = problem->keys + (size_t)position * problem->head_dim;
    double score = 0.0;
    uint32_t element;

    if (position >= problem->kv_positions)
        return (double)FT_TT_OP_ATTENTION_MASK_VALUE;
    for (element = 0; element < problem->head_dim; element++)
        score += (double)scaled_query(query[element], scale) * (double)key[element];
    return score;
}

bool ft_tt_attention_online_model(const ft_tt_attention_problem *problem, uint32_t block_tiles,
                                  float *out)
{
    const float scale = round_bf16((float)(1.0 / sqrt((double)(problem ? problem->head_dim : 1u))));
    uint32_t padded;
    uint32_t row;

    if (!problem_is_valid(problem) || !out || block_tiles == 0u)
        return false;
    padded = ft_tt_op_attention_padded_tiles(problem->kv_positions, block_tiles);
    if (padded == 0u)
        return false;

    for (row = 0; row < problem->q_rows; row++) {
        const float *query = problem->query + (size_t)row * problem->head_dim;
        double accumulator[MODEL_MAX_DIM];
        double maximum = 0.0;
        double total = 0.0;
        bool first = true;
        uint32_t block;
        uint32_t element;

        for (element = 0; element < problem->head_dim; element++)
            accumulator[element] = 0.0;

        for (block = 0; block * block_tiles < padded; block++) {
            const uint32_t start = block * block_tiles * TILE_DIM;
            const uint32_t count = block_tiles * TILE_DIM;
            double block_max = -HUGE_VAL;
            double block_total = 0.0;
            double correction;
            uint32_t index;

            /* The running max, seeded with the previous one -- the step a block-local max
             * would get wrong, and which softmax's shift invariance hides from PCC. */
            for (index = 0; index < count; index++) {
                const double score = model_score(problem, query, start + index, scale);

                if (score > block_max)
                    block_max = score;
            }
            if (!first && maximum > block_max)
                block_max = maximum;
            correction = first ? 1.0 : exp(maximum - block_max);

            for (element = 0; element < problem->head_dim; element++)
                accumulator[element] *= correction;
            total *= correction;

            for (index = 0; index < count; index++) {
                const uint32_t position = start + index;
                const double score = model_score(problem, query, position, scale);
                const double weight = exp(score - block_max);

                /* A DEAD POSITION IS NOT SKIPPED HERE, AND THAT IS DELIBERATE. The kernel
                 * has no way to skip it: it computes every tile of every block and relies
                 * on the additive mask driving the weight to exactly zero. A `continue`
                 * would be a SECOND masking mechanism, and a model with two of them stays
                 * green when one is removed — which is precisely what happened the first
                 * time this was mutated (dev notes §3: a check with several sufficient
                 * grounds tests whichever is cheapest). One mechanism, so breaking it
                 * shows. */
                block_total += weight;
                for (element = 0; element < problem->head_dim; element++)
                    accumulator[element] +=
                        weight * (double)problem->values[(size_t)position * problem->head_dim +
                                                          element];
            }
            total += block_total;
            maximum = block_max;
            first = false;
        }

        for (element = 0; element < problem->head_dim; element++)
            out[(size_t)row * problem->head_dim + element] =
                (float)(accumulator[element] / total);
    }
    return true;
}

double ft_tt_attention_pcc(const float *expected, const float *actual, size_t count)
{
    double mean_e = 0.0;
    double mean_a = 0.0;
    double cov = 0.0;
    double var_e = 0.0;
    double var_a = 0.0;
    size_t index;

    if (!expected || !actual || count == 0u)
        return 0.0;
    for (index = 0; index < count; index++) {
        mean_e += (double)expected[index];
        mean_a += (double)actual[index];
    }
    mean_e /= (double)count;
    mean_a /= (double)count;
    for (index = 0; index < count; index++) {
        const double de = (double)expected[index] - mean_e;
        const double da = (double)actual[index] - mean_a;

        cov += de * da;
        var_e += de * de;
        var_a += da * da;
    }
    if (var_e <= 0.0 || var_a <= 0.0)
        return 0.0;
    return cov / sqrt(var_e * var_a);
}

double ft_tt_attention_max_abs(const float *expected, const float *actual, size_t count)
{
    double worst = 0.0;
    size_t index;

    if (!expected || !actual)
        return 0.0;
    for (index = 0; index < count; index++) {
        const double difference = fabs((double)expected[index] - (double)actual[index]);

        if (difference > worst)
            worst = difference;
    }
    return worst;
}

size_t ft_tt_attention_non_finite(const float *values, size_t count)
{
    size_t found = 0;
    size_t index;

    if (!values)
        return 0;
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index]))
            found++;
    }
    return found;
}
