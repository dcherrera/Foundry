/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pl_batch.c
 * @brief Greedy token-budget batch planner and the weight/axpy helpers used to
 *   accumulate per-example gradients into a token-weighted batch gradient.
 */
#include "pl_batch.h"

#include <stdlib.h>

/* ============================================================= */
/* Token-budget greedy batching                                   */
/* ============================================================= */

int pl_batch_plan(const int *len, const int *n_valid_ex, const int *order,
                  int n, int token_budget,
                  pl_batch **out, int *n_batches)
{
    if (!len || !n_valid_ex || !order || !out || !n_batches || n <= 0 || token_budget <= 0)
        return -1;
    for (int k = 0; k < n; k++)
        if (order[k] < 0 || order[k] >= n || len[order[k]] < 1)
            return -1;

    /* At most n batches (one example each); shrink once the plan is known. */
    pl_batch *batches = calloc((size_t)n, sizeof(*batches));
    if (!batches)
        return -1;

    int nb = 0;
    for (int k = 0; k < n; k++) {
        int ex = order[k];
        int fits = nb > 0 && batches[nb - 1].tokens + len[ex] <= token_budget;

        if (!fits)
            batches[nb++].start = k;  /* empty batch: always takes the example */
        pl_batch *cur = &batches[nb - 1];
        cur->count++;
        cur->tokens += len[ex];
        cur->n_valid += n_valid_ex[ex];
    }

    pl_batch *shrunk = realloc(batches, (size_t)nb * sizeof(*batches));
    *out = shrunk ? shrunk : batches;
    *n_batches = nb;
    return 0;
}

/* ============================================================= */
/* Gradient accumulation helpers                                  */
/* ============================================================= */

float pl_batch_weight(int nv_i, int nv_batch)
{
    if (nv_batch <= 0)
        return 0.0f;
    return (float)nv_i / (float)nv_batch;
}

void pl_grad_axpy(float *acc, const float *g, float w, int n)
{
    for (int j = 0; j < n; j++)
        acc[j] += w * g[j];
}
