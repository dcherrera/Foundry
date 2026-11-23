/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_moe_route.c
 * @brief Off-hardware gate for kernels/ft_moe_route.h (qwen36 Phase 5.3): the card's routing
 *   step against an independent double-precision reference written the CPU oracle's way
 *   (Pantheon qwen35moe_moe.c: softmax over all logits, selection sort for the strictly
 *   greatest, renormalise the selected). Selection must match EXACTLY (indices and rank order,
 *   ties to the lowest index); weights within 4 ulp-ish (1e-6 relative); the shared-expert gate
 *   within 1e-6. Each red bit must change the answer.
 *
 *   cc -std=c99 -O2 -I../kernels test_moe_route.c -lm -o build/test_moe_route && ./build/test_moe_route
 *
 * @status New 2026-10-28.
 * @issues None
 * @todo None
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "ft_moe_route.h"

#define N 256u
#define K 8u

static int checks, failures;

static void check(int ok, const char *what)
{
    checks++;
    failures += !ok;
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
}

static uint32_t bits_of(float f)
{
    uint32_t u;

    memcpy(&u, &f, 4);
    return u;
}

static float float_of(uint32_t u)
{
    float f;

    memcpy(&f, &u, 4);
    return f;
}

static uint32_t g_state = 12345u;

static float uniform(void)
{
    g_state = g_state * 1664525u + 1013904223u;
    return (float)(g_state >> 8) / (float)(1u << 24);
}

/** The oracle's way: softmax probabilities, then k rounds of "strictly greatest unselected". */
static void reference(const float *logits, uint32_t *index, double *weight)
{
    double probs[N], max = -INFINITY, sum = 0.0, wsum = 0.0;
    int taken[N] = { 0 };

    for (uint32_t i = 0; i < N; i++)
        if (logits[i] > max)
            max = logits[i];
    for (uint32_t i = 0; i < N; i++)
        sum += (probs[i] = exp((double)logits[i] - max));
    for (uint32_t i = 0; i < N; i++)
        probs[i] /= sum;
    for (uint32_t r = 0; r < K; r++) {
        double best = -1.0;
        uint32_t at = 0;

        for (uint32_t i = 0; i < N; i++)
            if (!taken[i] && probs[i] > best) {
                best = probs[i];
                at = i;
            }
        taken[at] = 1;
        index[r] = at;
        weight[r] = best;
        wsum += best;
    }
    for (uint32_t r = 0; r < K; r++)
        weight[r] /= wsum;
}

static void route(const float *logits, uint32_t red, uint32_t *index, uint32_t *weight)
{
    ftmr_top top;

    ftmr_init(&top, K);
    for (uint32_t i = 0; i < N; i++)
        ftmr_offer(&top, i, bits_of(logits[i]), red);
    ftmr_weights(&top, red, index, weight);
}

/** ROWS mode helpers (plan S1): the union list is ascending and exactly the chosen set; a row's
 *  weight for an unchosen expert is 0; a set tile row lands at the face-order positions. */
static void test_rows_helpers(void)
{
    ftmr_row rows[64];
    uint32_t bitmap[FTMR_EXPERTS_MAX / 32u] = { 0u }, list[FTMR_EXPERTS_MAX], n, chosen[256] = { 0u };
    uint32_t tile[512];
    int ascending = 1, complete = 1, zero_ok = 1, layout_ok = 1, other_rows_zero = 1;

    printf("== rows mode: union list, row weights, tile rows ==\n");
    for (uint32_t r = 0; r < 64u; r++) {
        float logits[N];
        uint32_t index[K], weight[K];

        for (uint32_t i = 0; i < N; i++)
            logits[i] = uniform() - 0.5f;
        route(logits, 0u, index, weight);
        memcpy(rows[r].index, index, sizeof(index));
        memcpy(rows[r].weight, weight, sizeof(weight));
        ftmr_union_add(bitmap, &rows[r], K);
        for (uint32_t k = 0; k < K; k++)
            chosen[index[k]] = 1u;
    }
    n = ftmr_union_list(bitmap, N, list);
    for (uint32_t s = 1; s < n; s++)
        ascending &= list[s] > list[s - 1u];
    for (uint32_t e = 0, s = 0; e < N; e++) {
        if (!chosen[e])
            continue;
        complete &= s < n && list[s] == e;
        s++;
    }
    {
        uint32_t count = 0;

        for (uint32_t e = 0; e < N; e++)
            count += chosen[e];
        complete &= count == n;
    }
    check(ascending && complete, "the union list is ascending and exactly the chosen experts");
    for (uint32_t e = 0; e < N; e++) {
        int in_row = 0;

        for (uint32_t k = 0; k < K; k++)
            in_row |= rows[0].index[k] == e;
        if (!in_row)
            zero_ok &= ftmr_row_weight(&rows[0], K, e) == 0u;
        else
            zero_ok &= ftmr_row_weight(&rows[0], K, e) != 0u;
    }
    check(zero_ok, "a row's weight is nonzero exactly for its own experts");
    memset(tile, 0, sizeof(tile));
    ftmr_tile_set_row(tile, 21u, 0x3F80u);
    for (uint32_t row = 0; row < 32u; row++)
        for (uint32_t col = 0; col < 32u; col++) {
            const uint32_t face = (row / 16u) * 2u + col / 16u;
            const uint32_t at = face * 256u + (row % 16u) * 16u + col % 16u; /* bf16 index */
            const uint32_t value = (tile[at / 2u] >> ((at % 2u) * 16u)) & 0xFFFFu;

            if (row == 21u)
                layout_ok &= value == 0x3F80u;
            else
                other_rows_zero &= value == 0u;
        }
    check(layout_ok && other_rows_zero, "ftmr_tile_set_row fills exactly row 21 (faces 2/3) with the value");
}

int main(void)
{
    float logits[N];
    uint32_t index[K], weight[K], ref_index[K];
    double ref_weight[K], worst = 0.0;
    int selection_ok = 1;

    printf("== random logits: selection exact, weights to 1e-6 ==\n");
    for (int trial = 0; trial < 2000; trial++) {
        for (uint32_t i = 0; i < N; i++)
            logits[i] = (uniform() - 0.5f) * (float)(1 + trial % 16);
        if (trial % 5 == 0) /* quantised logits: many exact ties */
            for (uint32_t i = 0; i < N; i++)
                logits[i] = floorf(logits[i] * 4.0f) / 4.0f;
        route(logits, 0u, index, weight);
        reference(logits, ref_index, ref_weight);
        for (uint32_t r = 0; r < K; r++) {
            double rel = fabs((double)float_of(weight[r]) - ref_weight[r]) / ref_weight[r];

            selection_ok &= index[r] == ref_index[r];
            if (rel > worst)
                worst = rel;
        }
    }
    check(selection_ok, "2000 trials (400 with ties): indices and rank order == the oracle's");
    printf("        worst weight relative error %.3g\n", worst);
    check(worst < 1e-6, "weights within 1e-6 relative of the double reference");

    test_rows_helpers();
    printf("== ties and NaN ==\n");
    for (uint32_t i = 0; i < N; i++)
        logits[i] = -1.0f;
    logits[9] = 3.0f;
    logits[5] = 3.0f;
    logits[200] = NAN;
    route(logits, 0u, index, weight);
    {
        static const uint32_t want[K] = { 5u, 9u, 0u, 1u, 2u, 3u, 4u, 6u };

        check(memcmp(index, want, sizeof(want)) == 0,
              "equal logits: lowest index first (5 before 9, then 0 1 2 3 4 6); NaN never selected");
    }
    route(logits, FTMR_RED_TIE_FLIP, index, weight);
    check(index[0] == 9u, "RED tie flip: 9 before 5 (must differ)");

    printf("== shared-expert gate ==\n");
    {
        double rel = fabs((double)float_of(ft_det_sigmoid(bits_of(0.75f))) - 1.0 / (1.0 + exp(-0.75))) /
                     (1.0 / (1.0 + exp(-0.75)));

        check(rel < 1e-6, "sigmoid(0.75) within 1e-6");
    }

    printf("== reds ==\n");
    for (uint32_t i = 0; i < N; i++)
        logits[i] = (uniform() - 0.5f) * 8.0f;
    route(logits, 0u, index, weight);
    {
        uint32_t i2[K], w2[K];

        route(logits, FTMR_RED_UNNORM, i2, w2);
        check(memcmp(w2, weight, sizeof(w2)) != 0 && memcmp(i2, index, sizeof(i2)) == 0,
              "RED unnorm: same experts, different weights");
        route(logits, FTMR_RED_RANK_PERMUTE, i2, w2);
        check(i2[0] == index[K - 1u] && i2[K - 1u] == index[0], "RED rank permute: ranks reversed");
    }
    printf("\n%d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}
