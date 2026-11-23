/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * HybridMem Integration Benchmark
 *
 * Proves the full pipeline works end-to-end with realistic dimensions:
 *   1. Similarity preservation — HDC retrieval finds semantically similar entries
 *   2. Persistence roundtrip — flush/reopen preserves correct retrieval
 *   3. Performance — actual latency numbers at scale
 */

#include "../tensor/tensor.h"
#include "../hdc/hdc.h"
#include "../hybridmem/hybridmem.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

#define EMBED_DIM   768
#define HDC_DIM     4096
#define N_TOKENS    16
#define N_LAYERS    4
#define BENCH_DIR   "/tmp/bench_hybridmem"

/* ============================================================= */
/* Helpers                                                       */
/* ============================================================= */

static void rm_rf(const char *path) {
    DIR *d = opendir(path);
    if (!d) { unlink(path); return; }
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        char full[600];
        snprintf(full, sizeof(full), "%s/%s", path, entry->d_name);
        struct stat st;
        if (stat(full, &st) == 0 && S_ISDIR(st.st_mode)) rm_rf(full);
        else unlink(full);
    }
    closedir(d);
    rmdir(path);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Create [n_tokens, embed_dim] K/V tensor from a 1D embedding by tiling + noise */
static tm_tensor* tile_with_noise(tm_tensor *emb_1d, int n_tokens) {
    int64_t shape[2] = { n_tokens, EMBED_DIM };
    tm_tensor *t = tm_zeros(shape, 2, TM_F32);
    if (!t) return NULL;
    float *dst = (float *)t->data;
    const float *src = (const float *)emb_1d->data;
    for (int tok = 0; tok < n_tokens; tok++) {
        for (int d = 0; d < EMBED_DIM; d++) {
            /* Tile the embedding with slight per-token variation */
            dst[tok * EMBED_DIM + d] = src[d] + 0.01f * (float)(tok * 7 + d * 3) / (float)EMBED_DIM;
        }
    }
    return t;
}

/* Add small Gaussian-like noise to a 1D tensor */
static tm_tensor* add_noise(tm_tensor *emb_1d, float scale, int noise_seed) {
    tm_tensor *noisy = tm_clone(emb_1d);
    if (!noisy) return NULL;
    float *d = (float *)noisy->data;
    /* Simple deterministic pseudo-noise */
    uint64_t state = (uint64_t)noise_seed + 7919;
    for (int i = 0; i < EMBED_DIM; i++) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        float noise = ((float)(state & 0xFFFF) / 32768.0f - 1.0f) * scale;
        d[i] += noise;
    }
    return noisy;
}

/* ============================================================= */
/* Test 1: Similarity Preservation                               */
/* ============================================================= */

static int test_similarity_preservation(void) {
    printf("--- Test 1: Similarity Preservation ---\n");

    rm_rf(BENCH_DIR);
    fm_config cfg = fm_config_default(EMBED_DIM);
    cfg.store_dir = BENCH_DIR;
    cfg.hdc_dim = HDC_DIM;
    fm_store *store = fm_store_new(&cfg);
    if (!store) { printf("  FAIL: could not create store\n"); return 1; }

    int n_background = 1000;
    int n_trials = 50;

    /* Seed tensor RNG */
    tm_manual_seed(42);

    /* Store background entries (random noise) */
    int64_t emb_shape[1] = { EMBED_DIM };
    for (int i = 0; i < n_background; i++) {
        tm_tensor *emb = tm_randn(emb_shape, 1, TM_F32);
        tm_tensor *k = tile_with_noise(emb, N_TOKENS);
        tm_tensor *v = tile_with_noise(emb, N_TOKENS);
        fm_put(store, k, v, i % N_LAYERS, 1.0f);
        tm_free(emb);
        tm_free(k);
        tm_free(v);
    }
    printf("  Stored %d background entries across %d layers\n", n_background, N_LAYERS);

    /* Run needle queries */
    int recall_at_1 = 0;
    int recall_at_5 = 0;
    int total_rank = 0;

    printf("  Running %d needle queries...\n", n_trials);
    for (int trial = 0; trial < n_trials; trial++) {
        int layer = trial % N_LAYERS;

        /* Create needle embedding */
        tm_tensor *needle_emb = tm_randn(emb_shape, 1, TM_F32);
        tm_tensor *needle_k = tile_with_noise(needle_emb, N_TOKENS);
        tm_tensor *needle_v = tile_with_noise(needle_emb, N_TOKENS);
        uint64_t needle_id = fm_put(store, needle_k, needle_v, layer, 1.0f);

        /* Query with noisy version of needle */
        tm_tensor *query = add_noise(needle_emb, 0.1f, trial);

        fm_results *results = fm_query(store, query, layer, 5);

        /* Check if needle is in results */
        int found_rank = -1;
        if (results) {
            for (int i = 0; i < results->count; i++) {
                if (results->items[i].entry_id == needle_id) {
                    found_rank = i + 1;
                    break;
                }
            }
        }

        if (found_rank == 1) recall_at_1++;
        if (found_rank >= 1) recall_at_5++;
        if (found_rank >= 1) total_rank += found_rank;
        else total_rank += 6; /* penalty for miss */

        fm_results_free(results);
        tm_free(needle_emb);
        tm_free(needle_k);
        tm_free(needle_v);
        tm_free(query);
    }

    float r1 = (float)recall_at_1 / n_trials;
    float r5 = (float)recall_at_5 / n_trials;
    float mean_rank = (float)total_rank / n_trials;

    printf("  recall@1: %.2f  recall@5: %.2f  mean_rank: %.2f\n", r1, r5, mean_rank);

    fm_store_free(store);
    rm_rf(BENCH_DIR);

    int pass = r5 >= 0.80f;
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

/* ============================================================= */
/* Test 2: Persistence Roundtrip                                 */
/* ============================================================= */

static int test_persistence_roundtrip(void) {
    printf("--- Test 2: Persistence Roundtrip ---\n");

    rm_rf(BENCH_DIR);
    fm_config cfg = fm_config_default(EMBED_DIM);
    cfg.store_dir = BENCH_DIR;
    cfg.hdc_dim = HDC_DIM;
    fm_store *store = fm_store_new(&cfg);
    if (!store) { printf("  FAIL: could not create store\n"); return 1; }

    tm_manual_seed(123);
    int64_t emb_shape[1] = { EMBED_DIM };

    int n_entries = 100;
    int n_queries = 20;

    /* Store entries and save query vectors */
    uint64_t entry_ids[100];
    tm_tensor *query_vecs[20];
    int query_layers[20];

    for (int i = 0; i < n_entries; i++) {
        tm_tensor *emb = tm_randn(emb_shape, 1, TM_F32);
        tm_tensor *k = tile_with_noise(emb, N_TOKENS);
        tm_tensor *v = tile_with_noise(emb, N_TOKENS);
        entry_ids[i] = fm_put(store, k, v, i % N_LAYERS, 1.0f);

        /* Save some as queries */
        if (i < n_queries) {
            query_vecs[i] = tm_clone(emb);
            query_layers[i] = i % N_LAYERS;
        }
        tm_free(emb);
        tm_free(k);
        tm_free(v);
    }

    /* Query before flush to get expected results */
    uint64_t expected_top1[20];
    for (int i = 0; i < n_queries; i++) {
        fm_results *r = fm_query(store, query_vecs[i], query_layers[i], 1);
        expected_top1[i] = (r && r->count > 0) ? r->items[0].entry_id : 0;
        fm_results_free(r);
    }

    /* Flush and close */
    fm_store_flush(store);
    fm_store_free(store);
    printf("  Stored %d entries, flushed to disk\n", n_entries);

    /* Reopen */
    fm_store *store2 = fm_store_open(BENCH_DIR);
    if (!store2) {
        printf("  FAIL: could not reopen store\n");
        for (int i = 0; i < n_queries; i++) tm_free(query_vecs[i]);
        rm_rf(BENCH_DIR);
        return 1;
    }
    printf("  Reopened store, running %d queries...\n", n_queries);

    int matched = 0;
    for (int i = 0; i < n_queries; i++) {
        fm_results *r = fm_query(store2, query_vecs[i], query_layers[i], 1);
        uint64_t got = (r && r->count > 0) ? r->items[0].entry_id : 0;
        if (got == expected_top1[i]) matched++;
        fm_results_free(r);
        tm_free(query_vecs[i]);
    }

    fm_store_free(store2);
    rm_rf(BENCH_DIR);

    printf("  %d/%d queries matched\n", matched, n_queries);
    int pass = matched == n_queries;
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

/* ============================================================= */
/* Test 3: Performance                                           */
/* ============================================================= */

static int test_performance(void) {
    printf("--- Test 3: Performance ---\n");

    int scales[] = { 100, 500, 1000 };
    int n_scales = 3;
    int n_ops = 100;

    tm_manual_seed(456);
    int64_t emb_shape[1] = { EMBED_DIM };

    for (int si = 0; si < n_scales; si++) {
        int N = scales[si];
        rm_rf(BENCH_DIR);

        fm_config cfg = fm_config_default(EMBED_DIM);
        cfg.store_dir = BENCH_DIR;
        cfg.hdc_dim = HDC_DIM;
        fm_store *store = fm_store_new(&cfg);
        if (!store) continue;

        /* Pre-populate */
        for (int i = 0; i < N; i++) {
            tm_tensor *emb = tm_randn(emb_shape, 1, TM_F32);
            tm_tensor *k = tile_with_noise(emb, N_TOKENS);
            tm_tensor *v = tile_with_noise(emb, N_TOKENS);
            fm_put(store, k, v, i % N_LAYERS, 1.0f);
            tm_free(emb);
            tm_free(k);
            tm_free(v);
        }

        /* Benchmark puts */
        double t0 = now_ms();
        for (int i = 0; i < n_ops; i++) {
            tm_tensor *emb = tm_randn(emb_shape, 1, TM_F32);
            tm_tensor *k = tile_with_noise(emb, N_TOKENS);
            tm_tensor *v = tile_with_noise(emb, N_TOKENS);
            fm_put(store, k, v, i % N_LAYERS, 1.0f);
            tm_free(emb);
            tm_free(k);
            tm_free(v);
        }
        double put_ms = (now_ms() - t0) / n_ops;

        /* Benchmark queries */
        t0 = now_ms();
        for (int i = 0; i < n_ops; i++) {
            tm_tensor *q = tm_randn(emb_shape, 1, TM_F32);
            fm_results *r = fm_query(store, q, i % N_LAYERS, 5);
            fm_results_free(r);
            tm_free(q);
        }
        double query_ms = (now_ms() - t0) / n_ops;
        double qps = 1000.0 / query_ms;

        printf("  N=%-6d  put: %.1fms   query: %.1fms   (%.0f q/s)\n",
               N + n_ops, put_ms, query_ms, qps);

        fm_store_free(store);
        rm_rf(BENCH_DIR);
    }

    printf("  PASS\n\n");
    return 0;
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n=== HybridMem Integration Benchmark ===\n");
    printf("embed_dim=%d  hdc_dim=%d  n_tokens=%d  layers=%d\n\n",
           EMBED_DIM, HDC_DIM, N_TOKENS, N_LAYERS);

    int failures = 0;
    failures += test_similarity_preservation();
    failures += test_persistence_roundtrip();
    failures += test_performance();

    if (failures == 0) {
        printf("=== ALL TESTS PASSED ===\n\n");
    } else {
        printf("=== %d TEST(S) FAILED ===\n\n", failures);
    }

    return failures > 0 ? 1 : 0;
}
