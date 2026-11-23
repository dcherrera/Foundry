/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_pooling_loss.c — HDC Memory Viability Test 9
 *
 * Mean-Pooling Information Loss (synthetic, no model needed)
 *
 * Method:
 *   1. Generate random embeddings (dim=768)
 *   2. Group into batches of n_tokens (1, 4, 8, 16, 32)
 *   3. For each pooling strategy:
 *      a. Pool batch → single representative embedding
 *      b. Store via fm_put()
 *      c. Query with each original single embedding
 *      d. Measure recall@5 (does the batch appear in top-5?)
 *   4. Compare: mean-pool, max-pool, first-token, last-token
 *
 * Usage:
 *   ./examples/test_pooling_loss
 */

#include "hybridmem/hybridmem.h"
#include "hdc/hdc.h"
#include "tensor/tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple xorshift PRNG */
static uint32_t rng_state = 42;
static float randf(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float)(rng_state & 0x7FFFFFFF) / (float)0x7FFFFFFF * 2.0f - 1.0f;
}

/* Generate random embeddings: out[n * dim] */
static void gen_random_embeddings(float *out, int n, int dim) {
    for (int i = 0; i < n * dim; i++)
        out[i] = randf();
}

/* Mean-pool: out[dim] = mean of rows[n * dim] */
static void mean_pool(const float *rows, int n, int dim, float *out) {
    memset(out, 0, dim * sizeof(float));
    for (int t = 0; t < n; t++)
        for (int j = 0; j < dim; j++)
            out[j] += rows[t * dim + j];
    float inv_n = 1.0f / n;
    for (int j = 0; j < dim; j++)
        out[j] *= inv_n;
}

/* Max-pool: out[j] = max over rows */
static void max_pool(const float *rows, int n, int dim, float *out) {
    memcpy(out, rows, dim * sizeof(float));
    for (int t = 1; t < n; t++)
        for (int j = 0; j < dim; j++)
            if (rows[t * dim + j] > out[j])
                out[j] = rows[t * dim + j];
}

/* First-token: out = rows[0] */
static void first_token_pool(const float *rows, int n, int dim, float *out) {
    (void)n;
    memcpy(out, rows, dim * sizeof(float));
}

/* Last-token: out = rows[n-1] */
static void last_token_pool(const float *rows, int n, int dim, float *out) {
    memcpy(out, &rows[(n - 1) * dim], dim * sizeof(float));
}

typedef void (*pool_fn)(const float *rows, int n, int dim, float *out);

/* Run recall experiment for one pooling strategy at a given batch size.
 * Returns recall@5 (fraction of query embeddings that find their batch in top-5). */
static float run_recall_experiment(
    pool_fn pool, const char *pool_name,
    int dim, int batch_size, int n_batches, int top_k)
{
    /* Create HybridMem store */
    char store_dir[256];
    snprintf(store_dir, sizeof(store_dir), "/tmp/hm_pooling_test_%s_%d",
             pool_name, batch_size);
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", store_dir);
    system(cmd);

    fm_config cfg = fm_config_default(dim);
    cfg.store_dir = store_dir;
    fm_store *store = fm_store_new(&cfg);
    if (!store) {
        fprintf(stderr, "Failed to create store at %s\n", store_dir);
        return -1;
    }

    /* Generate embeddings and store pooled versions */
    int total_tokens = n_batches * batch_size;
    float *all_embeddings = (float*)malloc(total_tokens * dim * sizeof(float));
    gen_random_embeddings(all_embeddings, total_tokens, dim);

    float *pooled = (float*)malloc(dim * sizeof(float));

    for (int b = 0; b < n_batches; b++) {
        float *batch = &all_embeddings[b * batch_size * dim];
        pool(batch, batch_size, dim, pooled);

        /* Store: keys = pooled (1 token), values = pooled (dummy) */
        int64_t shape_k[] = {1, dim};
        tm_tensor *keys = tm_from_data(pooled, shape_k, 2, TM_F32);
        tm_tensor *keys_owned = tm_clone(keys);
        tm_tensor *vals_owned = tm_clone(keys);
        tm_free(keys);
        fm_put(store, keys_owned, vals_owned, 0, 1.0f);
        tm_free(keys_owned);
        tm_free(vals_owned);
    }

    fm_store_flush(store);

    /* Query with individual embeddings */
    int hits = 0;
    int queries = 0;
    int sample_step = batch_size > 1 ? batch_size / 2 : 1;  /* sample from each batch */

    for (int b = 0; b < n_batches; b++) {
        /* Query with one token from each batch */
        float *query_emb = &all_embeddings[(b * batch_size + sample_step - 1) * dim];
        int64_t shape_q[] = {dim};
        tm_tensor *query_t = tm_from_data(query_emb, shape_q, 1, TM_F32);

        fm_results *results = fm_query(store, query_t, 0, top_k);
        tm_free(query_t);

        if (results) {
            /* Check if any result matches batch b (by checking similarity) */
            float *batch_pooled = (float*)malloc(dim * sizeof(float));
            pool(&all_embeddings[b * batch_size * dim], batch_size, dim, batch_pooled);

            for (int r = 0; r < results->count; r++) {
                float *rk = tm_data_f32(results->items[r].keys);
                int n_ret = (int)tm_size(results->items[r].keys, 0);
                for (int t = 0; t < n_ret; t++) {
                    /* Check if this matches our pooled embedding */
                    float diff_sq = 0.0f;
                    for (int j = 0; j < dim; j++) {
                        float d = rk[t * dim + j] - batch_pooled[j];
                        diff_sq += d * d;
                    }
                    if (diff_sq < 1e-4f) { hits++; goto next_query; }
                }
            }
next_query:
            free(batch_pooled);
            fm_results_free(results);
        }
        queries++;
    }

    free(pooled);
    free(all_embeddings);
    fm_store_free(store);

    snprintf(cmd, sizeof(cmd), "rm -rf %s", store_dir);
    system(cmd);

    return queries > 0 ? (float)hits / queries : 0.0f;
}

int main(void) {
    int dim = 768;
    int top_k = 5;
    int n_batches = 100;

    int batch_sizes[] = {1, 4, 8, 16, 32};
    int n_sizes = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    const char *pool_names[] = {"mean", "max", "first", "last"};
    pool_fn pool_fns[] = {mean_pool, max_pool, first_token_pool, last_token_pool};
    int n_pools = 4;

    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "HDC Memory Viability — Test 9\n");
    fprintf(stderr, "Mean-Pooling Information Loss\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "Embedding dim:  %d\n", dim);
    fprintf(stderr, "Batches:        %d\n", n_batches);
    fprintf(stderr, "Top-k:          %d\n\n", top_k);

    /* Results table */
    fprintf(stderr, "%-8s", "batch");
    for (int p = 0; p < n_pools; p++)
        fprintf(stderr, "  %-10s", pool_names[p]);
    fprintf(stderr, "\n");
    for (int i = 0; i < 8 + n_pools * 12; i++) fprintf(stderr, "-");
    fprintf(stderr, "\n");

    float results[5][4]; /* [batch_size_idx][pool_idx] */

    for (int s = 0; s < n_sizes; s++) {
        fprintf(stderr, "%-8d", batch_sizes[s]);
        for (int p = 0; p < n_pools; p++) {
            rng_state = 42 + s * 100 + p * 10; /* reproducible per experiment */
            float recall = run_recall_experiment(
                pool_fns[p], pool_names[p],
                dim, batch_sizes[s], n_batches, top_k);
            results[s][p] = recall;
            fprintf(stderr, "  %-10.1f%%", recall * 100.0f);
        }
        fprintf(stderr, "\n");
    }

    /* Analysis */
    fprintf(stderr, "\n--- Analysis ---\n");
    fprintf(stderr, "batch=1 is the baseline (perfect recall expected for all methods).\n");
    fprintf(stderr, "Recall drop from batch=1 to batch=32 shows information loss from pooling.\n\n");

    /* Check if mean-pool maintains >50% recall at batch=8 */
    float mean_recall_8 = results[2][0]; /* batch_sizes[2] = 8, pool_fns[0] = mean */
    float mean_recall_32 = results[4][0];

    fprintf(stderr, "VERDICT:\n");
    if (mean_recall_8 >= 0.50f) {
        fprintf(stderr, "  PASS — Mean-pool recall@5 at batch=8: %.1f%% (>50%%)\n",
                mean_recall_8 * 100.0f);
    } else {
        fprintf(stderr, "  CONCERNING — Mean-pool recall@5 at batch=8: %.1f%% (<50%%)\n",
                mean_recall_8 * 100.0f);
        fprintf(stderr, "  Consider: store individual tokens or use multi-vector retrieval\n");
    }

    /* Find best pooling strategy at batch=8 */
    int best_pool = 0;
    float best_recall = results[2][0];
    for (int p = 1; p < n_pools; p++) {
        if (results[2][p] > best_recall) {
            best_recall = results[2][p];
            best_pool = p;
        }
    }
    fprintf(stderr, "  Best strategy at batch=8: %s (%.1f%% recall)\n",
            pool_names[best_pool], best_recall * 100.0f);
    fprintf(stderr, "  Mean-pool at batch=32: %.1f%%\n", mean_recall_32 * 100.0f);

    fprintf(stderr, "\n==========================================\n");
    return 0;
}
