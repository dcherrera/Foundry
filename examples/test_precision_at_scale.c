/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_precision_at_scale.c — HDC Memory Viability Test 4
 *
 * Retrieval Precision at Scale
 *
 * Method:
 *   1. Generate N random embeddings (dim=768), store all via fm_put()
 *   2. For 100 queries: brute-force cosine top-5 vs HDC top-5
 *   3. N = 100, 1K, 10K, 100K
 *   4. Report precision@5 at each N
 *
 * Usage:
 *   ./examples/test_precision_at_scale [--max-n 100000]
 */

#include "hybridmem/hybridmem.h"
#include "hdc/hdc.h"
#include "tensor/tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Simple xorshift PRNG */
static uint32_t rng_state = 42;
static float randf(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float)(rng_state & 0x7FFFFFFF) / (float)0x7FFFFFFF * 2.0f - 1.0f;
}

/* Cosine similarity */
static float cosine_sim(const float *a, const float *b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float d = sqrtf(na) * sqrtf(nb);
    return d > 1e-12f ? dot / d : 0;
}

typedef struct { int idx; float score; } scored_item;

static int cmp_scored_desc(const void *a, const void *b) {
    float diff = ((const scored_item*)b)->score - ((const scored_item*)a)->score;
    return diff > 0 ? 1 : (diff < 0 ? -1 : 0);
}

/* Brute-force cosine top-k. Returns indices. */
static void brute_force_topk(const float *query, const float *corpus,
                              int n, int dim, int k, int *out_indices) {
    scored_item *items = (scored_item*)malloc(n * sizeof(scored_item));
    for (int i = 0; i < n; i++) {
        items[i].idx = i;
        items[i].score = cosine_sim(query, &corpus[i * dim], dim);
    }
    qsort(items, n, sizeof(scored_item), cmp_scored_desc);
    for (int i = 0; i < k && i < n; i++)
        out_indices[i] = items[i].idx;
    free(items);
}

/* Run precision experiment at given N */
static float run_precision_experiment(int N, int dim, int n_queries, int top_k) {
    fprintf(stderr, "  N=%d: generating...", N);
    fflush(stderr);

    /* Generate embeddings */
    float *corpus = (float*)malloc((size_t)N * dim * sizeof(float));
    rng_state = 42;
    for (int i = 0; i < N * dim; i++)
        corpus[i] = randf();

    /* Create HybridMem store and insert all entries */
    char store_dir[256];
    snprintf(store_dir, sizeof(store_dir), "/tmp/hm_precision_%d", N);
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", store_dir);
    system(cmd);

    fm_config cfg = fm_config_default(dim);
    cfg.store_dir = store_dir;
    fm_store *store = fm_store_new(&cfg);
    if (!store) {
        fprintf(stderr, " ERROR: failed to create store\n");
        free(corpus);
        return -1;
    }

    fprintf(stderr, " storing...");
    fflush(stderr);

    /* Store entries individually (each is a 1-token entry) */
    for (int i = 0; i < N; i++) {
        int64_t shape_kv[] = {1, dim};
        tm_tensor *k = tm_from_data(&corpus[i * dim], shape_kv, 2, TM_F32);
        tm_tensor *k_owned = tm_clone(k);
        tm_tensor *v_owned = tm_clone(k);
        tm_free(k);
        fm_put(store, k_owned, v_owned, 0, 1.0f);
        tm_free(k_owned);
        tm_free(v_owned);
    }
    fm_store_flush(store);

    fprintf(stderr, " querying...");
    fflush(stderr);

    /* Run queries: pick random query embeddings and compare */
    int total_hits = 0;
    int total_possible = 0;

    uint32_t query_rng = 12345;
    for (int q = 0; q < n_queries; q++) {
        /* Pick a random embedding as query */
        query_rng ^= query_rng << 13;
        query_rng ^= query_rng >> 17;
        query_rng ^= query_rng << 5;
        int query_idx = query_rng % N;
        float *query = &corpus[query_idx * dim];

        /* Brute-force cosine top-k */
        int bf_indices[top_k];
        brute_force_topk(query, corpus, N, dim, top_k, bf_indices);

        /* HDC retrieval via fm_query */
        int64_t shape_q[] = {dim};
        tm_tensor *query_t = tm_from_data(query, shape_q, 1, TM_F32);
        fm_results *results = fm_query(store, query_t, 0, top_k);
        tm_free(query_t);

        if (results) {
            /* Collect retrieved indices by matching embedding content */
            for (int r = 0; r < results->count; r++) {
                float *rk = tm_data_f32(results->items[r].keys);
                int n_ret = (int)tm_size(results->items[r].keys, 0);
                for (int t = 0; t < n_ret; t++) {
                    /* Check against brute-force top-k */
                    for (int bi = 0; bi < top_k; bi++) {
                        float *bf_emb = &corpus[bf_indices[bi] * dim];
                        float diff_sq = 0;
                        for (int j = 0; j < dim; j++) {
                            float d = rk[t * dim + j] - bf_emb[j];
                            diff_sq += d * d;
                        }
                        if (diff_sq < 1e-4f) {
                            total_hits++;
                            break;
                        }
                    }
                }
            }
            fm_results_free(results);
        }

        total_possible += top_k;
    }

    fm_store_free(store);
    free(corpus);
    snprintf(cmd, sizeof(cmd), "rm -rf %s", store_dir);
    system(cmd);

    float precision = total_possible > 0 ? (float)total_hits / total_possible : 0;
    fprintf(stderr, " precision@%d = %.1f%%\n", top_k, precision * 100.0f);
    return precision;
}

int main(int argc, char **argv) {
    int max_n = 100000;
    int dim = 768;
    int n_queries = 100;
    int top_k = 5;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--max-n") == 0 && i + 1 < argc)
            max_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc)
            dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--queries") == 0 && i + 1 < argc)
            n_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc)
            top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: %s [options]\n"
                "  --max-n N     Maximum store size (default: 100000)\n"
                "  --dim N       Embedding dimension (default: 768)\n"
                "  --queries N   Number of queries (default: 100)\n"
                "  --top-k N     Top-k for precision (default: 5)\n",
                argv[0]);
            return 0;
        }
    }

    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "HDC Memory Viability — Test 4\n");
    fprintf(stderr, "Retrieval Precision at Scale\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "Embedding dim:  %d\n", dim);
    fprintf(stderr, "Queries:        %d\n", n_queries);
    fprintf(stderr, "Top-k:          %d\n", top_k);
    fprintf(stderr, "Max N:          %d\n\n", max_n);

    int test_sizes[] = {100, 1000, 10000, 100000};
    int n_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    float precisions[4];
    int n_run = 0;

    for (int s = 0; s < n_sizes; s++) {
        if (test_sizes[s] > max_n) break;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        precisions[s] = run_precision_experiment(test_sizes[s], dim, n_queries, top_k);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        fprintf(stderr, "    Time: %.1fs\n", secs);
        n_run++;
    }

    /* Summary */
    fprintf(stderr, "\n--- Summary ---\n");
    fprintf(stderr, "%-10s  %-15s\n", "N", "precision@5");
    for (int i = 0; i < 30; i++) fprintf(stderr, "-");
    fprintf(stderr, "\n");
    for (int s = 0; s < n_run; s++) {
        fprintf(stderr, "%-10d  %.1f%%\n", test_sizes[s], precisions[s] * 100.0f);
    }

    fprintf(stderr, "\nVERDICT:\n");
    /* Check precision at largest tested scale */
    float last_precision = precisions[n_run - 1];
    if (last_precision >= 0.80f) {
        fprintf(stderr, "  PASS — precision@5 = %.1f%% at N=%d\n",
                last_precision * 100.0f, test_sizes[n_run - 1]);
    } else if (last_precision >= 0.50f) {
        fprintf(stderr, "  MARGINAL — precision@5 = %.1f%% at N=%d\n",
                last_precision * 100.0f, test_sizes[n_run - 1]);
        fprintf(stderr, "  Consider: increase HDC dim, add re-ranking, or use multi-probe\n");
    } else {
        fprintf(stderr, "  FAIL — precision@5 = %.1f%% at N=%d\n",
                last_precision * 100.0f, test_sizes[n_run - 1]);
        fprintf(stderr, "  HDC index quality degrades too much at this scale\n");
    }

    fprintf(stderr, "\n==========================================\n");
    return 0;
}
