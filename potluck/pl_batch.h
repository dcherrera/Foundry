/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pl_batch.h
 * @brief Token-budget batching and token-weighted gradient accumulation for
 *   Potluck LoRA training.
 *
 * The planner groups examples (in the caller's `order`; it never shuffles)
 * into batches of at most `token_budget` tokens. Each example's gradient is a
 * per-example mean over its valid (loss-masked) targets, so scaling it by
 * nv_i / nv_batch before summing gives the gradient of the mean loss over
 * every valid token in the batch (spec §5.1.1 item 8).
 */
#ifndef FOUNDRY_PL_BATCH_H
#define FOUNDRY_PL_BATCH_H

#ifdef __cplusplus
extern "C" {
#endif

/* Batch segment: covers order[start .. start+count-1] */
typedef struct {
    int start;    /* index into `order` */
    int count;    /* number of examples in this batch */
    int tokens;   /* sum of len[order[k]] for k in batch */
    int n_valid;  /* sum of n_valid_ex[order[k]] for k in batch */
} pl_batch;

/* Greedy token-budget batching over `order`.
 *
 * Appends example order[k] (k = 0,1,...) to the current batch while
 *   batch.tokens + len[order[k]] <= token_budget,
 * or while the batch is empty (a single example longer than the budget
 * becomes its own batch).  tokens = Σ len, n_valid = Σ n_valid_ex.
 *
 * Returns 0 and sets *out / *n_batches on success (caller must free *out).
 * Returns -1 on bad args (n <= 0, token_budget <= 0, NULL pointers,
 *         an order entry outside [0, n), or any len < 1) or out of memory.
 */
int pl_batch_plan(const int *len, const int *n_valid_ex, const int *order,
                  int n, int token_budget,
                  pl_batch **out, int *n_batches);

/* Weight for one example's (per-example-mean) gradient inside its batch:
 *   nv_i / nv_batch    (0.0f if nv_batch == 0)
 */
float pl_batch_weight(int nv_i, int nv_batch);

/* acc[j] += w * g[j]   for j = 0 .. n-1 */
void pl_grad_axpy(float *acc, const float *g, float w, int n);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_PL_BATCH_H */
