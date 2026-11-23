/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_topk_reference.c
 * @brief Implementation of the CPU reference for the on-card top-k and sampler. See
 *   ft_tt_topk_reference.h for what it deliberately does NOT share with the kernel.
 *
 * @status Off-hardware only.
 * @issues None
 * @todo None
 */

#include "ft_tt_topk_reference.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_tilize.h"

/* The polynomial, transliterated from kernels/resident_topk_protocol.h. test_topk_ops.c
 * PARSES that header and compares these literals against it, so the two cannot drift. */
#define REFERENCE_LOG2E 1.44269504088896340736f
#define REFERENCE_EXP2_C1 0.69314718055994531f
#define REFERENCE_EXP2_C2 0.24022650695910072f
#define REFERENCE_EXP2_C3 0.05550410866482158f
#define REFERENCE_EXP2_C4 0.00961812910762848f
#define REFERENCE_EXP2_C5 0.00133335581467035f
#define REFERENCE_EXP2_C6 0.00015403530393381f

uint32_t ft_tt_topk_reference_key(uint16_t bf16_bits)
{
    if (bf16_bits & 0x8000u)
        return (uint32_t)((uint16_t)~bf16_bits);
    return (uint32_t)(bf16_bits | 0x8000u);
}

float ft_tt_topk_reference_exp(float x)
{
    float scaled = x * REFERENCE_LOG2E;
    int32_t power;
    float fraction;
    float poly;
    uint32_t exponent_bits;
    float scale;

    if (scaled < -126.0f)
        return 0.0f;
    power = (int32_t)(scaled - 0.5f);
    fraction = scaled - (float)power;
    poly = REFERENCE_EXP2_C6;
    poly = poly * fraction + REFERENCE_EXP2_C5;
    poly = poly * fraction + REFERENCE_EXP2_C4;
    poly = poly * fraction + REFERENCE_EXP2_C3;
    poly = poly * fraction + REFERENCE_EXP2_C2;
    poly = poly * fraction + REFERENCE_EXP2_C1;
    poly = poly * fraction + 1.0f;
    exponent_bits = (uint32_t)(power + 127) << 23;
    memcpy(&scale, &exponent_bits, sizeof(scale));
    return poly * scale;
}

uint32_t ft_tt_topk_reference_rng_next(uint32_t *state)
{
    uint32_t value = *state;

    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

float ft_tt_topk_reference_rng_unit(uint32_t *state)
{
    return (float)(ft_tt_topk_reference_rng_next(state) >> 8) * (1.0f / 16777216.0f);
}

int ft_tt_topk_reference_select(const uint16_t *row, uint32_t vocab, uint32_t select_count,
                                uint32_t *out_indices, float *out_values)
{
    uint32_t taken[FT_TT_OP_SAMPLE_MAX_SELECT];
    uint32_t round;

    if (!row || !out_indices || !out_values)
        return -1;
    if (select_count == 0u || select_count > FT_TT_OP_SAMPLE_MAX_SELECT || select_count > vocab)
        return -1;

    /* Iterative argmax, exactly the K80's round structure: the winner of each round is
     * removed and the next round scans what is left. Removal is a membership test rather
     * than the CUDA kernel's -inf clobber, so the input row is never mutated and a caller
     * cannot be handed a damaged buffer by a reference. */
    for (round = 0u; round < select_count; round++) {
        uint32_t best = vocab;
        float best_value = 0.0f;
        uint32_t index;

        for (index = 0u; index < vocab; index++) {
            float value;
            uint32_t earlier;
            int already = 0;

            for (earlier = 0u; earlier < round; earlier++)
                if (taken[earlier] == index)
                    already = 1;
            if (already)
                continue;
            value = ft_float_from_bf16(row[index]);
            /* Strict `>` scanned in ascending index order: the LOWEST index wins a tie,
             * which is the whole of the tie rule. */
            if (best == vocab || value > best_value) {
                best = index;
                best_value = value;
            }
        }
        taken[round] = best;
        out_indices[round] = best;
        out_values[round] = best_value;
    }
    return 0;
}

/** Descending order for the two threshold passes. Ties are interchangeable in a threshold,
 *  so the comparator does not need to be stable and the reference does not pretend it is. */
static int compare_descending(const void *left, const void *right)
{
    float a = *(const float *)left;
    float b = *(const float *)right;

    if (a < b)
        return 1;
    if (a > b)
        return -1;
    return 0;
}

/** Insertion sort, so the reference does not depend on qsort's tie behaviour either. */
static void sort_descending(float *values, uint32_t count)
{
    uint32_t entry;

    for (entry = 1u; entry < count; entry++) {
        float value = values[entry];
        uint32_t position = entry;

        while (position > 0u && compare_descending(&values[position - 1u], &value) > 0) {
            values[position] = values[position - 1u];
            position--;
        }
        values[position] = value;
    }
}

/** Gather the survivors into ASCENDING INDEX order — the order the vocabulary scan the
 *  cumulative draw imitates would have visited them in. */
static void gather_ascending(const uint32_t *indices, const float *values, uint32_t count,
                             uint32_t *out_ids, float *out_values)
{
    uint32_t entry;

    for (entry = 0u; entry < count; entry++) {
        uint32_t index = indices[entry];
        float value = values[entry];
        uint32_t position = entry;

        while (position > 0u && out_ids[position - 1u] > index) {
            out_ids[position] = out_ids[position - 1u];
            out_values[position] = out_values[position - 1u];
            position--;
        }
        out_ids[position] = index;
        out_values[position] = value;
    }
}

uint32_t ft_tt_topk_reference_sample(const uint32_t *indices, const float *values,
                                     uint32_t count,
                                     const ft_tt_op_sample_params *params,
                                     const uint32_t *recent, uint32_t recent_count,
                                     uint32_t *rng_state, int use_libm_exp)
{
    uint32_t ids[FT_TT_OP_SAMPLE_MAX_SELECT];
    float vals[FT_TT_OP_SAMPLE_MAX_SELECT];
    float probabilities[FT_TT_OP_SAMPLE_MAX_SELECT];
    float work[FT_TT_OP_SAMPLE_MAX_SELECT];
    float inverse_temperature;
    float maximum;
    float sum = 0.0f;
    float inverse_sum;
    float draw;
    float cumulative = 0.0f;
    uint32_t entry;
    uint32_t best = 0u;

    if (!indices || !values || !params || count == 0u || count > FT_TT_OP_SAMPLE_MAX_SELECT)
        return 0u;
    gather_ascending(indices, values, count, ids, vals);

    if (params->temperature <= 0.0f) {
        /* GREEDY: raw values, NO repetition penalty, strict `>` in ascending index order. */
        for (entry = 1u; entry < count; entry++)
            if (vals[entry] > vals[best])
                best = entry;
        return ids[best];
    }

    if (params->repeat_penalty != 1.0f && recent && recent_count > 0u) {
        uint32_t back;

        for (back = 0u; back < recent_count; back++) {
            uint32_t token = recent[back];

            for (entry = 0u; entry < count; entry++) {
                if (ids[entry] != token)
                    continue;
                if (vals[entry] > 0.0f)
                    vals[entry] = vals[entry] / params->repeat_penalty;
                else
                    vals[entry] = vals[entry] * params->repeat_penalty;
            }
        }
    }

    inverse_temperature = 1.0f / params->temperature;
    for (entry = 0u; entry < count; entry++)
        vals[entry] = vals[entry] * inverse_temperature;

    maximum = vals[0];
    for (entry = 1u; entry < count; entry++)
        if (vals[entry] > maximum)
            maximum = vals[entry];

    for (entry = 0u; entry < count; entry++) {
        probabilities[entry] = use_libm_exp ? expf(vals[entry] - maximum)
                                            : ft_tt_topk_reference_exp(vals[entry] - maximum);
        sum = sum + probabilities[entry];
    }
    inverse_sum = 1.0f / sum;
    for (entry = 0u; entry < count; entry++)
        probabilities[entry] = probabilities[entry] * inverse_sum;

    if (params->top_k > 0u && params->top_k < count) {
        float threshold;

        memcpy(work, probabilities, (size_t)count * sizeof(work[0]));
        sort_descending(work, count);
        threshold = work[params->top_k - 1u];
        for (entry = 0u; entry < count; entry++)
            if (probabilities[entry] < threshold)
                probabilities[entry] = 0.0f;
    }

    if (params->top_p < 1.0f) {
        uint32_t candidates = 0u;
        float mass = 0.0f;
        float threshold = 0.0f;

        for (entry = 0u; entry < count; entry++)
            if (probabilities[entry] > 0.0f)
                work[candidates++] = probabilities[entry];
        sort_descending(work, candidates);
        for (entry = 0u; entry < candidates; entry++) {
            if (!(mass < params->top_p))
                break;
            if (work[entry] <= 0.0f)
                break;
            mass = mass + work[entry];
            threshold = work[entry];
        }
        for (entry = 0u; entry < count; entry++)
            if (probabilities[entry] < threshold)
                probabilities[entry] = 0.0f;
    }

    sum = 0.0f;
    for (entry = 0u; entry < count; entry++)
        sum = sum + probabilities[entry];
    if (sum > 0.0f) {
        inverse_sum = 1.0f / sum;
        for (entry = 0u; entry < count; entry++)
            probabilities[entry] = probabilities[entry] * inverse_sum;
    }

    draw = ft_tt_topk_reference_rng_unit(rng_state);
    for (entry = 0u; entry < count; entry++) {
        cumulative = cumulative + probabilities[entry];
        if (draw < cumulative)
            return ids[entry];
    }
    for (entry = count; entry > 0u; entry--)
        if (probabilities[entry - 1u] > 0.0f)
            return ids[entry - 1u];
    /* The full-vocabulary scan falls off the end and returns token 0. Matching that is
     * deliberate: it is NOT "the lowest survivor". */
    return 0u;
}

int ft_tt_topk_reference_tilize_row(const uint16_t *row_values, uint32_t vocab, uint32_t row,
                                    uint16_t filler, void *out, size_t out_bytes)
{
    uint16_t *plain = NULL;
    uint8_t *destination = (uint8_t *)out;
    uint32_t tiles = vocab / 32u;
    uint32_t tile;
    uint32_t line;
    uint32_t column;
    int status = 0;

    if (!row_values || !out || vocab == 0u || (vocab % 32u) != 0u || row >= 32u)
        return -1;
    if (out_bytes < (size_t)tiles * 2048u)
        return -1;

    plain = (uint16_t *)calloc((size_t)FT_TT_TILE_HEIGHT * FT_TT_TILE_WIDTH, sizeof(*plain));
    if (!plain)
        return -1;

    for (tile = 0u; tile < tiles; tile++) {
        for (line = 0u; line < (uint32_t)FT_TT_TILE_HEIGHT; line++)
            for (column = 0u; column < (uint32_t)FT_TT_TILE_WIDTH; column++)
                plain[line * FT_TT_TILE_WIDTH + column] =
                    (line == row) ? row_values[tile * 32u + column] : filler;
        if (ft_tt_tilize_strip(plain, sizeof(uint16_t), FT_TT_TILE_HEIGHT, FT_TT_TILE_WIDTH,
                               FT_TT_TILE_WIDTH * sizeof(uint16_t),
                               destination + (size_t)tile * 2048u) != 0) {
            status = -1;
            break;
        }
    }
    free(plain);
    return status;
}
