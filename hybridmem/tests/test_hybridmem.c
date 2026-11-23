/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../hybridmem.h"
#include "../../tensor/tensor.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>

#define TEST_EMBED_DIM 64
#define TEST_DIR "/tmp/fm_test_store"

/* Helper: recursively remove directory */
static void rm_rf(const char *path) {
    DIR *d = opendir(path);
    if (!d) { unlink(path); return; }
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        char full[600];
        snprintf(full, sizeof(full), "%s/%s", path, entry->d_name);
        struct stat st;
        if (stat(full, &st) == 0 && S_ISDIR(st.st_mode)) {
            rm_rf(full);
        } else {
            unlink(full);
        }
    }
    closedir(d);
    rmdir(path);
}

/* Helper: create random 2D tensor [n_tokens, embed_dim] */
static tm_tensor* make_kv(int n_tokens, int embed_dim, float base) {
    int64_t shape[2] = { n_tokens, embed_dim };
    tm_tensor *t = tm_zeros(shape, 2, TM_F32);
    if (!t) return NULL;
    float *d = (float *)t->data;
    for (int i = 0; i < n_tokens * embed_dim; i++) {
        d[i] = base + (float)(i % 17) * 0.1f;
    }
    return t;
}

/* Helper: create 1D query tensor [embed_dim] */
static tm_tensor* make_query(int embed_dim, float base) {
    int64_t shape[1] = { embed_dim };
    tm_tensor *t = tm_zeros(shape, 1, TM_F32);
    if (!t) return NULL;
    float *d = (float *)t->data;
    for (int i = 0; i < embed_dim; i++) {
        d[i] = base + (float)(i % 11) * 0.1f;
    }
    return t;
}

/* ============================================================= */
/* Lifecycle Tests                                               */
/* ============================================================= */

static void test_config_default(void) {
    fm_config c = fm_config_default(128);
    ASSERT_INT_EQ(128, c.embed_dim);
    ASSERT_INT_EQ(4096, c.hdc_dim);
    ASSERT_INT_EQ(42, (int)c.seed);
    ASSERT_TRUE(c.shard_max_bytes > 0);
    ASSERT_FALSE(c.quantize_stored);
}

static void test_store_create_free(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);
    ASSERT_INT_EQ(0, (int)fm_store_size(s));
    fm_store_free(s);
    rm_rf(TEST_DIR);

    /* NULL config */
    ASSERT_NULL(fm_store_new(NULL));

    /* Invalid embed_dim */
    fm_config bad = fm_config_default(0);
    ASSERT_NULL(fm_store_new(&bad));

    /* NULL-safe free */
    fm_store_free(NULL);
}

/* ============================================================= */
/* Put Tests                                                     */
/* ============================================================= */

static void test_put_single(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    tm_tensor *k = make_kv(4, TEST_EMBED_DIM, 1.0f);
    tm_tensor *v = make_kv(4, TEST_EMBED_DIM, 2.0f);
    ASSERT_NOT_NULL(k);
    ASSERT_NOT_NULL(v);

    uint64_t id = fm_put(s, k, v, 0, 1.0f);
    ASSERT_TRUE(id > 0);
    ASSERT_INT_EQ(1, (int)fm_store_size(s));

    tm_free(k);
    tm_free(v);
    fm_store_free(s);
    rm_rf(TEST_DIR);
}

static void test_put_multiple(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    for (int i = 0; i < 10; i++) {
        tm_tensor *k = make_kv(4, TEST_EMBED_DIM, (float)i);
        tm_tensor *v = make_kv(4, TEST_EMBED_DIM, (float)i + 10.0f);
        uint64_t id = fm_put(s, k, v, i % 3, (float)i * 0.5f);
        ASSERT_TRUE(id > 0);
        tm_free(k);
        tm_free(v);
    }
    ASSERT_INT_EQ(10, (int)fm_store_size(s));

    fm_store_free(s);
    rm_rf(TEST_DIR);
}

/* ============================================================= */
/* Query Tests                                                   */
/* ============================================================= */

static void test_query_empty(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    tm_tensor *q = make_query(TEST_EMBED_DIM, 1.0f);
    fm_results *r = fm_query(s, q, 0, 5);
    ASSERT_NOT_NULL(r);
    ASSERT_INT_EQ(0, r->count);

    fm_results_free(r);
    tm_free(q);
    fm_store_free(s);
    rm_rf(TEST_DIR);
}

static void test_query_exact(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    /* Store entry with known key pattern */
    tm_tensor *k = make_kv(4, TEST_EMBED_DIM, 5.0f);
    tm_tensor *v = make_kv(4, TEST_EMBED_DIM, 10.0f);
    uint64_t id = fm_put(s, k, v, 0, 1.0f);
    ASSERT_TRUE(id > 0);

    /* Query with same pattern as mean of keys */
    tm_tensor *q = make_query(TEST_EMBED_DIM, 5.0f);
    fm_results *r = fm_query(s, q, 0, 1);
    ASSERT_NOT_NULL(r);
    ASSERT_INT_EQ(1, r->count);
    ASSERT_TRUE(r->items[0].entry_id == id);
    ASSERT_NOT_NULL(r->items[0].keys);
    ASSERT_NOT_NULL(r->items[0].values);

    fm_results_free(r);
    tm_free(q);
    tm_free(k);
    tm_free(v);
    fm_store_free(s);
    rm_rf(TEST_DIR);
}

static void test_query_top_k_order(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    /* Store several entries on same layer */
    for (int i = 0; i < 10; i++) {
        tm_tensor *k = make_kv(2, TEST_EMBED_DIM, (float)i * 3.0f);
        tm_tensor *v = make_kv(2, TEST_EMBED_DIM, (float)i * 3.0f + 1.0f);
        fm_put(s, k, v, 0, 1.0f);
        tm_free(k);
        tm_free(v);
    }

    tm_tensor *q = make_query(TEST_EMBED_DIM, 0.0f);
    fm_results *r = fm_query(s, q, 0, 5);
    ASSERT_NOT_NULL(r);
    ASSERT_TRUE(r->count > 0);

    /* Results should be in decreasing score order */
    for (int i = 1; i < r->count; i++) {
        ASSERT_TRUE(r->items[i - 1].score >= r->items[i].score);
    }

    fm_results_free(r);
    tm_free(q);
    fm_store_free(s);
    rm_rf(TEST_DIR);
}

static void test_query_similar_higher(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    /* Create near entry: positive values, same direction as query */
    int64_t shape2[2] = { 2, TEST_EMBED_DIM };
    tm_tensor *k_near = tm_zeros(shape2, 2, TM_F32);
    tm_tensor *v_near = tm_zeros(shape2, 2, TM_F32);
    float *kn = (float *)k_near->data;
    for (int i = 0; i < 2 * TEST_EMBED_DIM; i++) kn[i] = 1.0f;
    float *vn = (float *)v_near->data;
    for (int i = 0; i < 2 * TEST_EMBED_DIM; i++) vn[i] = 1.0f;

    /* Create far entry: alternating sign, very different direction */
    tm_tensor *k_far = tm_zeros(shape2, 2, TM_F32);
    tm_tensor *v_far = tm_zeros(shape2, 2, TM_F32);
    float *kf = (float *)k_far->data;
    for (int i = 0; i < 2 * TEST_EMBED_DIM; i++) kf[i] = (i % 2 == 0) ? -5.0f : 5.0f;
    float *vf = (float *)v_far->data;
    for (int i = 0; i < 2 * TEST_EMBED_DIM; i++) vf[i] = (i % 2 == 0) ? -5.0f : 5.0f;

    uint64_t near_id = fm_put(s, k_near, v_near, 0, 1.0f);
    fm_put(s, k_far, v_far, 0, 1.0f);

    /* Query: all positive, same direction as near entry */
    int64_t shape1[1] = { TEST_EMBED_DIM };
    tm_tensor *q = tm_zeros(shape1, 1, TM_F32);
    float *qd = (float *)q->data;
    for (int i = 0; i < TEST_EMBED_DIM; i++) qd[i] = 1.0f;

    fm_results *r = fm_query(s, q, 0, 2);
    ASSERT_NOT_NULL(r);
    ASSERT_INT_EQ(2, r->count);

    /* Near entry should rank higher (same sign direction) */
    ASSERT_TRUE(r->items[0].entry_id == near_id);
    ASSERT_TRUE(r->items[0].score >= r->items[1].score);

    fm_results_free(r);
    tm_free(q);
    tm_free(k_near);
    tm_free(v_near);
    tm_free(k_far);
    tm_free(v_far);
    fm_store_free(s);
    rm_rf(TEST_DIR);
}

static void test_query_layer_filter(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    /* Store entries on different layers */
    tm_tensor *k = make_kv(2, TEST_EMBED_DIM, 1.0f);
    tm_tensor *v = make_kv(2, TEST_EMBED_DIM, 2.0f);
    fm_put(s, k, v, 0, 1.0f);
    fm_put(s, k, v, 1, 1.0f);
    fm_put(s, k, v, 2, 1.0f);
    fm_put(s, k, v, 0, 1.0f);
    fm_put(s, k, v, 1, 1.0f);

    /* Query layer 1 only */
    tm_tensor *q = make_query(TEST_EMBED_DIM, 1.0f);
    fm_results *r = fm_query(s, q, 1, 10);
    ASSERT_NOT_NULL(r);
    /* Should only get layer 1 entries (2 of them) */
    ASSERT_INT_EQ(2, r->count);
    for (int i = 0; i < r->count; i++) {
        ASSERT_INT_EQ(1, r->items[i].layer_idx);
    }

    fm_results_free(r);
    tm_free(q);
    tm_free(k);
    tm_free(v);
    fm_store_free(s);
    rm_rf(TEST_DIR);
}

static void test_result_shapes(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    int n_tokens = 8;
    tm_tensor *k = make_kv(n_tokens, TEST_EMBED_DIM, 1.0f);
    tm_tensor *v = make_kv(n_tokens, TEST_EMBED_DIM, 2.0f);
    fm_put(s, k, v, 0, 1.0f);

    tm_tensor *q = make_query(TEST_EMBED_DIM, 1.0f);
    fm_results *r = fm_query(s, q, 0, 1);
    ASSERT_NOT_NULL(r);
    ASSERT_INT_EQ(1, r->count);
    ASSERT_NOT_NULL(r->items[0].keys);
    ASSERT_NOT_NULL(r->items[0].values);

    /* Check shapes match original */
    ASSERT_INT_EQ(2, r->items[0].keys->ndim);
    ASSERT_INT_EQ(n_tokens, (int)r->items[0].keys->shape[0]);
    ASSERT_INT_EQ(TEST_EMBED_DIM, (int)r->items[0].keys->shape[1]);
    ASSERT_INT_EQ(2, r->items[0].values->ndim);
    ASSERT_INT_EQ(n_tokens, (int)r->items[0].values->shape[0]);
    ASSERT_INT_EQ(TEST_EMBED_DIM, (int)r->items[0].values->shape[1]);

    fm_results_free(r);
    tm_free(q);
    tm_free(k);
    tm_free(v);
    fm_store_free(s);
    rm_rf(TEST_DIR);
}

static void test_results_free(void) {
    /* Free NULL should not crash */
    fm_results_free(NULL);

    /* Free empty results */
    fm_results *r = calloc(1, sizeof(fm_results));
    fm_results_free(r);
}

static void test_null_safety(void) {
    /* All functions should handle NULL gracefully */
    fm_store_free(NULL);
    ASSERT_INT_EQ(0, (int)fm_store_size(NULL));
    ASSERT_INT_EQ(0, (int)fm_evict(NULL, 0));

    fm_stats s = fm_stats_get(NULL);
    ASSERT_INT_EQ(0, (int)s.num_entries);

    ASSERT_NULL(fm_query(NULL, NULL, 0, 5));
    ASSERT_INT_EQ(0, (int)fm_put(NULL, NULL, NULL, 0, 0.0f));
}

/* ============================================================= */
/* Stats Tests                                                   */
/* ============================================================= */

static void test_stats(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    fm_stats st = fm_stats_get(s);
    ASSERT_INT_EQ(0, (int)st.total_stores);
    ASSERT_INT_EQ(0, (int)st.total_queries);

    tm_tensor *k = make_kv(2, TEST_EMBED_DIM, 1.0f);
    tm_tensor *v = make_kv(2, TEST_EMBED_DIM, 2.0f);
    fm_put(s, k, v, 0, 1.0f);
    fm_put(s, k, v, 0, 1.0f);

    st = fm_stats_get(s);
    ASSERT_INT_EQ(2, (int)st.total_stores);
    ASSERT_INT_EQ(2, (int)st.num_entries);

    tm_tensor *q = make_query(TEST_EMBED_DIM, 1.0f);
    fm_results *r = fm_query(s, q, 0, 5);
    fm_results_free(r);

    st = fm_stats_get(s);
    ASSERT_INT_EQ(1, (int)st.total_queries);

    tm_free(q);
    tm_free(k);
    tm_free(v);
    fm_store_free(s);
    rm_rf(TEST_DIR);
}

/* ============================================================= */
/* Persistence Tests                                             */
/* ============================================================= */

static void test_flush_creates_files(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    tm_tensor *k = make_kv(4, TEST_EMBED_DIM, 1.0f);
    tm_tensor *v = make_kv(4, TEST_EMBED_DIM, 2.0f);
    fm_put(s, k, v, 0, 1.0f);

    int ret = fm_store_flush(s);
    ASSERT_INT_EQ(0, ret);

    /* Check files exist */
    char path[600];
    struct stat st;
    snprintf(path, sizeof(path), "%s/index.json", TEST_DIR);
    ASSERT_INT_EQ(0, stat(path, &st));
    snprintf(path, sizeof(path), "%s/entries.bin", TEST_DIR);
    ASSERT_INT_EQ(0, stat(path, &st));

    tm_free(k);
    tm_free(v);
    fm_store_free(s);
    rm_rf(TEST_DIR);
}

static void test_open_roundtrip(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    /* Store entries */
    tm_tensor *k = make_kv(4, TEST_EMBED_DIM, 3.0f);
    tm_tensor *v = make_kv(4, TEST_EMBED_DIM, 7.0f);
    uint64_t id1 = fm_put(s, k, v, 0, 1.0f);
    uint64_t id2 = fm_put(s, k, v, 1, 2.0f);
    ASSERT_TRUE(id1 > 0);
    ASSERT_TRUE(id2 > 0);

    fm_store_flush(s);
    fm_store_free(s);

    /* Reopen */
    fm_store *s2 = fm_store_open(TEST_DIR);
    ASSERT_NOT_NULL(s2);
    ASSERT_INT_EQ(2, (int)fm_store_size(s2));

    /* Query should still work */
    tm_tensor *q = make_query(TEST_EMBED_DIM, 3.0f);
    fm_results *r = fm_query(s2, q, 0, 5);
    ASSERT_NOT_NULL(r);
    ASSERT_TRUE(r->count > 0);
    ASSERT_NOT_NULL(r->items[0].keys);
    ASSERT_NOT_NULL(r->items[0].values);

    fm_results_free(r);
    tm_free(q);
    tm_free(k);
    tm_free(v);
    fm_store_free(s2);
    rm_rf(TEST_DIR);
}

/* ============================================================= */
/* Eviction Tests                                                */
/* ============================================================= */

static void test_evict_reduces_count(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    for (int i = 0; i < 10; i++) {
        tm_tensor *k = make_kv(2, TEST_EMBED_DIM, (float)i);
        tm_tensor *v = make_kv(2, TEST_EMBED_DIM, (float)i);
        fm_put(s, k, v, 0, (float)i * 0.1f);
        tm_free(k);
        tm_free(v);
    }
    ASSERT_INT_EQ(10, (int)fm_store_size(s));

    size_t evicted = fm_evict(s, 5);
    ASSERT_INT_EQ(5, (int)evicted);
    ASSERT_INT_EQ(5, (int)fm_store_size(s));

    fm_store_free(s);
    rm_rf(TEST_DIR);
}

static void test_evict_keeps_important(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    /* Low importance entries */
    for (int i = 0; i < 5; i++) {
        tm_tensor *k = make_kv(2, TEST_EMBED_DIM, (float)i);
        tm_tensor *v = make_kv(2, TEST_EMBED_DIM, (float)i);
        fm_put(s, k, v, 0, 0.1f);
        tm_free(k);
        tm_free(v);
    }

    /* High importance entry */
    tm_tensor *k_imp = make_kv(2, TEST_EMBED_DIM, 99.0f);
    tm_tensor *v_imp = make_kv(2, TEST_EMBED_DIM, 99.0f);
    uint64_t imp_id = fm_put(s, k_imp, v_imp, 0, 100.0f);
    tm_free(k_imp);
    tm_free(v_imp);

    /* Evict down to 1 */
    fm_evict(s, 1);
    ASSERT_INT_EQ(1, (int)fm_store_size(s));

    /* Query for the surviving entry */
    tm_tensor *q = make_query(TEST_EMBED_DIM, 99.0f);
    fm_results *r = fm_query(s, q, 0, 1);
    ASSERT_NOT_NULL(r);
    ASSERT_INT_EQ(1, r->count);
    ASSERT_TRUE(r->items[0].entry_id == imp_id);

    fm_results_free(r);
    tm_free(q);
    fm_store_free(s);
    rm_rf(TEST_DIR);
}

static void test_compact_reduces_size(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    /* Store 10 entries */
    for (int i = 0; i < 10; i++) {
        tm_tensor *k = make_kv(4, TEST_EMBED_DIM, (float)i);
        tm_tensor *v = make_kv(4, TEST_EMBED_DIM, (float)i);
        fm_put(s, k, v, 0, (float)i);
        tm_free(k);
        tm_free(v);
    }

    fm_stats before = fm_stats_get(s);
    ASSERT_TRUE(before.disk_bytes_used > 0);

    /* Evict half */
    fm_evict(s, 5);
    ASSERT_INT_EQ(5, (int)fm_store_size(s));

    /* Compact */
    int ret = fm_compact(s);
    ASSERT_INT_EQ(0, ret);

    fm_stats after = fm_stats_get(s);
    ASSERT_TRUE(after.disk_bytes_used < before.disk_bytes_used);

    fm_store_free(s);
    rm_rf(TEST_DIR);
}

/* ============================================================= */
/* Integration Bridge Tests                                      */
/* ============================================================= */

static void test_augmented_kv_shapes(void) {
    rm_rf(TEST_DIR);
    fm_config c = fm_config_default(TEST_EMBED_DIM);
    c.store_dir = TEST_DIR;
    fm_store *s = fm_store_new(&c);
    ASSERT_NOT_NULL(s);

    /* Store an entry */
    int stored_tokens = 4;
    tm_tensor *k_store = make_kv(stored_tokens, TEST_EMBED_DIM, 1.0f);
    tm_tensor *v_store = make_kv(stored_tokens, TEST_EMBED_DIM, 2.0f);
    fm_put(s, k_store, v_store, 0, 1.0f);

    /* Create cache */
    int cache_tokens = 8;
    tm_tensor *cache_k = make_kv(cache_tokens, TEST_EMBED_DIM, 1.0f);
    tm_tensor *cache_v = make_kv(cache_tokens, TEST_EMBED_DIM, 2.0f);
    tm_tensor *q = make_query(TEST_EMBED_DIM, 1.0f);

    tm_tensor *mk = NULL, *mv = NULL;
    int ret = fm_augmented_kv(q, cache_k, cache_v, s, 0, 1, &mk, &mv);
    ASSERT_INT_EQ(0, ret);
    ASSERT_NOT_NULL(mk);
    ASSERT_NOT_NULL(mv);

    /* Merged shape should be [cache_tokens + stored_tokens, embed_dim] */
    ASSERT_INT_EQ(2, mk->ndim);
    ASSERT_INT_EQ(cache_tokens + stored_tokens, (int)mk->shape[0]);
    ASSERT_INT_EQ(TEST_EMBED_DIM, (int)mk->shape[1]);
    ASSERT_INT_EQ(2, mv->ndim);
    ASSERT_INT_EQ(cache_tokens + stored_tokens, (int)mv->shape[0]);
    ASSERT_INT_EQ(TEST_EMBED_DIM, (int)mv->shape[1]);

    tm_free(mk);
    tm_free(mv);
    tm_free(q);
    tm_free(cache_k);
    tm_free(cache_v);
    tm_free(k_store);
    tm_free(v_store);
    fm_store_free(s);
    rm_rf(TEST_DIR);
}

static void test_augmented_kv_null_store(void) {
    int cache_tokens = 8;
    tm_tensor *cache_k = make_kv(cache_tokens, TEST_EMBED_DIM, 1.0f);
    tm_tensor *cache_v = make_kv(cache_tokens, TEST_EMBED_DIM, 2.0f);
    tm_tensor *q = make_query(TEST_EMBED_DIM, 1.0f);

    tm_tensor *mk = NULL, *mv = NULL;
    int ret = fm_augmented_kv(q, cache_k, cache_v, NULL, 0, 5, &mk, &mv);
    ASSERT_INT_EQ(0, ret);
    ASSERT_NOT_NULL(mk);
    ASSERT_NOT_NULL(mv);

    /* Should be cache-only (same shape as cache) */
    ASSERT_INT_EQ(cache_tokens, (int)mk->shape[0]);
    ASSERT_INT_EQ(cache_tokens, (int)mv->shape[0]);

    tm_free(mk);
    tm_free(mv);
    tm_free(q);
    tm_free(cache_k);
    tm_free(cache_v);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  HybridMem Module Tests\n\n");

    RUN_TEST(config_default);
    RUN_TEST(store_create_free);
    RUN_TEST(put_single);
    RUN_TEST(put_multiple);
    RUN_TEST(query_empty);
    RUN_TEST(query_exact);
    RUN_TEST(query_top_k_order);
    RUN_TEST(query_similar_higher);
    RUN_TEST(query_layer_filter);
    RUN_TEST(result_shapes);
    RUN_TEST(results_free);
    RUN_TEST(null_safety);
    RUN_TEST(stats);
    RUN_TEST(flush_creates_files);
    RUN_TEST(open_roundtrip);
    RUN_TEST(evict_reduces_count);
    RUN_TEST(evict_keeps_important);
    RUN_TEST(compact_reduces_size);
    RUN_TEST(augmented_kv_shapes);
    RUN_TEST(augmented_kv_null_store);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
