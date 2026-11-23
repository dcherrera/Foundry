/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "hybridmem.h"
#include "hdc/hdc.h"
#include "tensor/tensor.h"
#include "safetensors/safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>

/* ============================================================= */
/* Internal types                                                */
/* ============================================================= */

/* Per-entry metadata stored in memory */
typedef struct {
    uint64_t entry_id;
    int layer_idx;
    int n_tokens;
    float importance;
    uint64_t timestamp;
    uint32_t access_count;
    int shard_idx;
    bool evicted;
    fv_vector *signature;       /* HDC signature for search */
} fm_entry;

/* Shard tracking */
typedef struct {
    char path[512];
    size_t byte_size;
    int num_entries;
} fm_shard;

/* LRU shard cache — keeps up to max_open_shards fs_file handles open */
typedef struct {
    int shard_idx;          /* which shard this slot caches (-1 = empty) */
    fs_file *file;          /* open file handle (mmap'd) */
    uint64_t last_access;   /* monotonic counter for LRU eviction */
} fm__shard_cache_slot;

/* Write buffer entry — holds K/V data in memory until flushed to disk */
typedef struct {
    uint64_t entry_id;
    float *k_data;          /* owned copy: [n_tokens * embed_dim] */
    float *v_data;          /* owned copy: [n_tokens * embed_dim] */
    int n_tokens;
    int shard_idx;          /* target shard (assigned at put time) */
} fm__write_buf_entry;

/* ---- LSH index for sub-linear query ---- */

#define FM_LSH_NUM_TABLES  8    /* number of hash tables */
#define FM_LSH_BITS        14   /* bits per hash key → 16384 buckets */
#define FM_LSH_NUM_BUCKETS (1 << FM_LSH_BITS)

/* Each hash table selects FM_LSH_BITS random bit positions from the HDC signature */
typedef struct {
    int bit_positions[FM_LSH_BITS];  /* which bits to extract */
} fm__lsh_table_config;

/* Dynamic bucket: array of entry indices */
typedef struct {
    size_t *indices;
    int count;
    int cap;
} fm__lsh_bucket;

/* Full LSH index */
typedef struct {
    fm__lsh_table_config tables[FM_LSH_NUM_TABLES];
    fm__lsh_bucket *buckets;   /* [FM_LSH_NUM_TABLES * FM_LSH_NUM_BUCKETS] */
    bool built;
} fm__lsh_index;

/* Store internal state */
struct fm_store {
    fm_config config;
    char store_dir[512];

    /* Projection */
    fv_projector *projector;

    /* Entries */
    fm_entry *entries;
    size_t num_entries;
    size_t entries_capacity;

    /* Shards */
    fm_shard *shards;
    int num_shards;
    int shards_capacity;

    /* Shard cache (LRU pool of open file handles) */
    fm__shard_cache_slot *shard_cache;
    int shard_cache_cap;
    uint64_t shard_cache_clock;

    /* Write buffer — batches pending writes to avoid per-entry shard rewrites */
    fm__write_buf_entry *write_buf;
    int write_buf_count;
    int write_buf_cap;

    /* LSH index for sub-linear query */
    fm__lsh_index *lsh;

    /* ID counter */
    uint64_t next_id;

    /* Stats */
    size_t total_stores;
    size_t total_queries;
    size_t total_hits;
    size_t disk_bytes_used;

    /* Timestamp counter (monotonic) */
    uint64_t timestamp;
};

/* ============================================================= */
/* Internal helpers                                              */
/* ============================================================= */

static int fm__ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) return 0;
    return mkdir(path, 0755);
}

static void fm__shard_path(fm_store *store, int shard_idx, char *buf, size_t bufsz) {
    snprintf(buf, bufsz, "%s/shard_%04d.safetensors", store->store_dir, shard_idx);
}

/* ---- Shard cache helpers ---- */

/* Get a cached fs_file* for the given shard, opening on miss.
 * Returns NULL if the shard file cannot be opened. The returned
 * pointer is owned by the cache — caller must NOT fs_close() it. */
static fs_file* fm__shard_cache_get(fm_store *store, int shard_idx) {
    if (!store->shard_cache || shard_idx < 0) return NULL;

    /* Check for cache hit */
    for (int i = 0; i < store->shard_cache_cap; i++) {
        if (store->shard_cache[i].shard_idx == shard_idx) {
            store->shard_cache[i].last_access = ++store->shard_cache_clock;
            return store->shard_cache[i].file;
        }
    }

    /* Cache miss — find a slot (empty or LRU) */
    int victim = 0;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < store->shard_cache_cap; i++) {
        if (store->shard_cache[i].shard_idx < 0) {
            victim = i;
            break;
        }
        if (store->shard_cache[i].last_access < oldest) {
            oldest = store->shard_cache[i].last_access;
            victim = i;
        }
    }

    /* Evict victim if occupied */
    if (store->shard_cache[victim].file) {
        fs_close(store->shard_cache[victim].file);
        store->shard_cache[victim].file = NULL;
        store->shard_cache[victim].shard_idx = -1;
    }

    /* Open shard file */
    fm_shard *shard = &store->shards[shard_idx];
    fs_file *f = fs_open(shard->path);
    if (!f) return NULL;

    store->shard_cache[victim].shard_idx = shard_idx;
    store->shard_cache[victim].file = f;
    store->shard_cache[victim].last_access = ++store->shard_cache_clock;
    return f;
}

/* Invalidate a shard's cache entry (e.g., after rewriting the file) */
static void fm__shard_cache_invalidate(fm_store *store, int shard_idx) {
    if (!store->shard_cache) return;
    for (int i = 0; i < store->shard_cache_cap; i++) {
        if (store->shard_cache[i].shard_idx == shard_idx) {
            if (store->shard_cache[i].file) {
                fs_close(store->shard_cache[i].file);
            }
            store->shard_cache[i].file = NULL;
            store->shard_cache[i].shard_idx = -1;
            return;
        }
    }
}

/* Close all cached shard handles */
static void fm__shard_cache_free(fm_store *store) {
    if (!store->shard_cache) return;
    for (int i = 0; i < store->shard_cache_cap; i++) {
        if (store->shard_cache[i].file) {
            fs_close(store->shard_cache[i].file);
        }
    }
    free(store->shard_cache);
    store->shard_cache = NULL;
}

/* ---- Write buffer helpers ---- */

/* Add K/V data to write buffer. Copies the tensor data. */
static int fm__write_buf_add(fm_store *store, uint64_t entry_id,
                              const tm_tensor *keys, const tm_tensor *values,
                              int shard_idx) {
    if (store->write_buf_count >= store->write_buf_cap) {
        int new_cap = store->write_buf_cap == 0 ? 64 : store->write_buf_cap * 2;
        fm__write_buf_entry *nb = realloc(store->write_buf,
                                           new_cap * sizeof(fm__write_buf_entry));
        if (!nb) return -1;
        store->write_buf = nb;
        store->write_buf_cap = new_cap;
    }
    int n_tokens = (int)keys->shape[0];
    int embed_dim = (int)keys->shape[1];
    size_t data_bytes = (size_t)n_tokens * embed_dim * sizeof(float);

    fm__write_buf_entry *wb = &store->write_buf[store->write_buf_count];
    wb->entry_id = entry_id;
    wb->n_tokens = n_tokens;
    wb->shard_idx = shard_idx;
    wb->k_data = malloc(data_bytes);
    wb->v_data = malloc(data_bytes);
    if (!wb->k_data || !wb->v_data) {
        free(wb->k_data); free(wb->v_data);
        return -1;
    }
    memcpy(wb->k_data, keys->data, data_bytes);
    memcpy(wb->v_data, values->data, data_bytes);
    store->write_buf_count++;
    return 0;
}

/* Load K/V from write buffer for a given entry_id.
 * Returns 0 on success (found in buffer), -1 if not found. */
static int fm__write_buf_load(fm_store *store, uint64_t entry_id,
                               tm_tensor **keys_out, tm_tensor **values_out) {
    int embed_dim = store->config.embed_dim;
    for (int i = 0; i < store->write_buf_count; i++) {
        fm__write_buf_entry *wb = &store->write_buf[i];
        if (wb->entry_id != entry_id) continue;

        int64_t shape[] = {wb->n_tokens, embed_dim};
        size_t data_bytes = (size_t)wb->n_tokens * embed_dim * sizeof(float);

        tm_tensor *k = tm_zeros(shape, 2, TM_F32);
        tm_tensor *v = tm_zeros(shape, 2, TM_F32);
        if (!k || !v) { tm_free(k); tm_free(v); return -1; }

        memcpy(k->data, wb->k_data, data_bytes);
        memcpy(v->data, wb->v_data, data_bytes);

        *keys_out = k;
        *values_out = v;
        return 0;
    }
    return -1;
}

/* Flush write buffer to disk — one shard write per unique shard_idx.
 * Returns 0 on success, -1 on error. */
static int fm__write_buf_flush(fm_store *store) {
    if (store->write_buf_count == 0) return 0;
    if (!store->store_dir[0]) {
        /* No disk — just discard buffer */
        for (int i = 0; i < store->write_buf_count; i++) {
            free(store->write_buf[i].k_data);
            free(store->write_buf[i].v_data);
        }
        store->write_buf_count = 0;
        return 0;
    }

    fm__ensure_dir(store->store_dir);

    /* Group by shard_idx and write each shard once */
    /* Find unique shard indices */
    int max_shard = -1;
    for (int i = 0; i < store->write_buf_count; i++) {
        if (store->write_buf[i].shard_idx > max_shard)
            max_shard = store->write_buf[i].shard_idx;
    }

    for (int si = 0; si <= max_shard; si++) {
        /* Count entries for this shard */
        int count = 0;
        for (int i = 0; i < store->write_buf_count; i++) {
            if (store->write_buf[i].shard_idx == si) count++;
        }
        if (count == 0) continue;

        /* Invalidate shard cache before rewriting */
        fm__shard_cache_invalidate(store, si);

        fm_shard *shard = &store->shards[si];
        fs_writer *w = fs_writer_new();
        if (!w) return -1;

        /* Copy existing shard contents */
        fs_file *existing = fs_open(shard->path);
        if (existing) {
            int nc = 0;
            const char **names = fs_tensor_names(existing, &nc);
            if (names) {
                for (int i = 0; i < nc; i++) {
                    fs_tensor_info *info = fs_get_info(existing, names[i]);
                    if (info) {
                        const void *data = fs_get_data(existing, names[i]);
                        if (data) {
                            fs_writer_add_tensor(w, names[i], info->dtype,
                                                 info->shape, info->ndim, data);
                        }
                        fs_free_info(info);
                    }
                }
                free(names);
            }
            fs_close(existing);
        }

        /* Add all buffered entries for this shard */
        int embed_dim = store->config.embed_dim;
        for (int i = 0; i < store->write_buf_count; i++) {
            fm__write_buf_entry *wb = &store->write_buf[i];
            if (wb->shard_idx != si) continue;

            char key_name[64], val_name[64];
            snprintf(key_name, sizeof(key_name), "k_%llu", (unsigned long long)wb->entry_id);
            snprintf(val_name, sizeof(val_name), "v_%llu", (unsigned long long)wb->entry_id);

            int64_t shape[] = {wb->n_tokens, embed_dim};
            fs_writer_add_f32(w, key_name, shape, 2, wb->k_data);
            fs_writer_add_f32(w, val_name, shape, 2, wb->v_data);
        }

        bool ok = fs_writer_save(w, shard->path);
        fs_writer_free(w);
        if (!ok) return -1;

        /* Update shard size */
        struct stat st;
        if (stat(shard->path, &st) == 0) {
            shard->byte_size = st.st_size;
        }
        shard->num_entries += count;
    }

    /* Free buffer entries */
    for (int i = 0; i < store->write_buf_count; i++) {
        free(store->write_buf[i].k_data);
        free(store->write_buf[i].v_data);
    }
    store->write_buf_count = 0;

    /* Recalculate disk usage */
    store->disk_bytes_used = 0;
    for (int i = 0; i < store->num_shards; i++) {
        store->disk_bytes_used += store->shards[i].byte_size;
    }

    return 0;
}

/* Free write buffer without flushing */
static void fm__write_buf_free(fm_store *store) {
    for (int i = 0; i < store->write_buf_count; i++) {
        free(store->write_buf[i].k_data);
        free(store->write_buf[i].v_data);
    }
    free(store->write_buf);
    store->write_buf = NULL;
    store->write_buf_count = 0;
    store->write_buf_cap = 0;
}

/* ---- LSH index helpers ---- */

/* Simple xorshift64 PRNG for selecting random bit positions */
static uint64_t fm__lsh_rand(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* Compute hash key for a signature in table t */
static uint32_t fm__lsh_hash(const fm__lsh_table_config *tc, const fv_vector *sig) {
    uint32_t key = 0;
    for (int b = 0; b < FM_LSH_BITS; b++) {
        int pos = tc->bit_positions[b];
        int word = pos / 64;
        int bit = pos % 64;
        if (sig->bits[word] & ((uint64_t)1 << bit))
            key |= ((uint32_t)1 << b);
    }
    return key;
}

/* Add an entry index to a bucket */
static void fm__lsh_bucket_add(fm__lsh_bucket *bucket, size_t idx) {
    if (bucket->count >= bucket->cap) {
        int new_cap = bucket->cap == 0 ? 4 : bucket->cap * 2;
        size_t *ni = realloc(bucket->indices, new_cap * sizeof(size_t));
        if (!ni) return;
        bucket->indices = ni;
        bucket->cap = new_cap;
    }
    bucket->indices[bucket->count++] = idx;
}

/* Create and initialize LSH index */
static fm__lsh_index* fm__lsh_new(int hdc_dim, uint64_t seed) {
    fm__lsh_index *lsh = calloc(1, sizeof(fm__lsh_index));
    if (!lsh) return NULL;

    size_t total_buckets = (size_t)FM_LSH_NUM_TABLES * FM_LSH_NUM_BUCKETS;
    lsh->buckets = calloc(total_buckets, sizeof(fm__lsh_bucket));
    if (!lsh->buckets) { free(lsh); return NULL; }

    /* Select random bit positions for each table */
    uint64_t rng = seed ^ 0xDEADBEEFCAFEULL;
    for (int t = 0; t < FM_LSH_NUM_TABLES; t++) {
        for (int b = 0; b < FM_LSH_BITS; b++) {
            lsh->tables[t].bit_positions[b] = (int)(fm__lsh_rand(&rng) % (uint64_t)hdc_dim);
        }
    }

    lsh->built = true;
    return lsh;
}

/* Insert an entry into the LSH index */
static void fm__lsh_insert(fm__lsh_index *lsh, size_t entry_idx, const fv_vector *sig) {
    if (!lsh || !lsh->built || !sig) return;
    for (int t = 0; t < FM_LSH_NUM_TABLES; t++) {
        uint32_t key = fm__lsh_hash(&lsh->tables[t], sig);
        fm__lsh_bucket *bucket = &lsh->buckets[(size_t)t * FM_LSH_NUM_BUCKETS + key];
        fm__lsh_bucket_add(bucket, entry_idx);
    }
}

/* Query the LSH index: returns candidate entry indices.
 * Caller must free(*out_indices). Returns count of candidates. */
static int fm__lsh_query(fm__lsh_index *lsh, const fv_vector *query_hv,
                          size_t **out_indices) {
    if (!lsh || !lsh->built || !query_hv) { *out_indices = NULL; return 0; }

    /* Collect candidates from all tables into a flat array */
    size_t *candidates = NULL;
    int n_candidates = 0;
    int candidates_cap = 0;

    for (int t = 0; t < FM_LSH_NUM_TABLES; t++) {
        uint32_t key = fm__lsh_hash(&lsh->tables[t], query_hv);
        fm__lsh_bucket *bucket = &lsh->buckets[(size_t)t * FM_LSH_NUM_BUCKETS + key];
        for (int i = 0; i < bucket->count; i++) {
            if (n_candidates >= candidates_cap) {
                int new_cap = candidates_cap == 0 ? 64 : candidates_cap * 2;
                size_t *nc = realloc(candidates, new_cap * sizeof(size_t));
                if (!nc) break;
                candidates = nc;
                candidates_cap = new_cap;
            }
            candidates[n_candidates++] = bucket->indices[i];
        }
    }

    if (n_candidates == 0) { *out_indices = NULL; return 0; }

    /* Deduplicate by sorting then removing consecutive duplicates */
    /* Simple insertion sort — n_candidates is typically small (<200) */
    for (int i = 1; i < n_candidates; i++) {
        size_t val = candidates[i];
        int j = i - 1;
        while (j >= 0 && candidates[j] > val) {
            candidates[j + 1] = candidates[j];
            j--;
        }
        candidates[j + 1] = val;
    }
    int unique = 0;
    for (int i = 0; i < n_candidates; i++) {
        if (i == 0 || candidates[i] != candidates[i - 1]) {
            candidates[unique++] = candidates[i];
        }
    }

    *out_indices = candidates;
    return unique;
}

/* Free the LSH index */
static void fm__lsh_free(fm__lsh_index *lsh) {
    if (!lsh) return;
    if (lsh->buckets) {
        size_t total = (size_t)FM_LSH_NUM_TABLES * FM_LSH_NUM_BUCKETS;
        for (size_t i = 0; i < total; i++) {
            free(lsh->buckets[i].indices);
        }
        free(lsh->buckets);
    }
    free(lsh);
}

/* Mean-pool a 2D tensor [n_tokens, embed_dim] -> [embed_dim] */
static tm_tensor* fm__mean_pool(const tm_tensor *t) {
    if (!t || t->ndim != 2 || t->dtype != TM_F32) return NULL;
    int64_t n_tokens = t->shape[0];
    int64_t embed_dim = t->shape[1];
    int64_t shape[1] = { embed_dim };
    tm_tensor *out = tm_zeros(shape, 1, TM_F32);
    if (!out) return NULL;
    float *o = (float *)out->data;
    const float *d = (const float *)t->data;
    for (int64_t tok = 0; tok < n_tokens; tok++) {
        for (int64_t dim = 0; dim < embed_dim; dim++) {
            o[dim] += d[tok * embed_dim + dim];
        }
    }
    float inv_n = 1.0f / (float)n_tokens;
    for (int64_t dim = 0; dim < embed_dim; dim++) {
        o[dim] *= inv_n;
    }
    return out;
}

/* Grow entries array if needed */
static int fm__grow_entries(fm_store *store) {
    if (store->num_entries < store->entries_capacity) return 0;
    size_t new_cap = store->entries_capacity == 0 ? 64 : store->entries_capacity * 2;
    fm_entry *new_entries = realloc(store->entries, new_cap * sizeof(fm_entry));
    if (!new_entries) return -1;
    store->entries = new_entries;
    store->entries_capacity = new_cap;
    return 0;
}

/* Grow shards array if needed */
static int fm__grow_shards(fm_store *store) {
    if (store->num_shards < store->shards_capacity) return 0;
    int new_cap = store->shards_capacity == 0 ? 8 : store->shards_capacity * 2;
    fm_shard *new_shards = realloc(store->shards, new_cap * sizeof(fm_shard));
    if (!new_shards) return -1;
    store->shards = new_shards;
    store->shards_capacity = new_cap;
    return 0;
}

/* Get or create current shard for writing */
static int fm__current_shard(fm_store *store) {
    if (store->num_shards > 0) {
        fm_shard *last = &store->shards[store->num_shards - 1];
        if (last->byte_size < store->config.shard_max_bytes) {
            return store->num_shards - 1;
        }
    }
    /* Create new shard */
    if (fm__grow_shards(store) != 0) return -1;
    int idx = store->num_shards++;
    fm_shard *s = &store->shards[idx];
    memset(s, 0, sizeof(fm_shard));
    fm__shard_path(store, idx, s->path, sizeof(s->path));
    s->byte_size = 0;
    s->num_entries = 0;
    return idx;
}

/* Load K/V from shard for a given entry_id.
 * Uses shard cache to avoid repeated open/mmap/close syscalls. */
static int fm__load_from_shard(fm_store *store, int shard_idx, uint64_t entry_id,
                               tm_tensor **keys_out, tm_tensor **values_out) {
    if (shard_idx < 0 || shard_idx >= store->num_shards) return -1;

    /* Try cached handle first, fall back to direct open */
    fs_file *f = fm__shard_cache_get(store, shard_idx);
    bool cached = (f != NULL);
    if (!f) {
        fm_shard *shard = &store->shards[shard_idx];
        f = fs_open(shard->path);
        if (!f) return -1;
    }

    char key_name[64], val_name[64];
    snprintf(key_name, sizeof(key_name), "k_%llu", (unsigned long long)entry_id);
    snprintf(val_name, sizeof(val_name), "v_%llu", (unsigned long long)entry_id);

    /* Get tensor info for shapes */
    fs_tensor_info *k_info = fs_get_info(f, key_name);
    fs_tensor_info *v_info = fs_get_info(f, val_name);
    if (!k_info || !v_info) {
        if (k_info) fs_free_info(k_info);
        if (v_info) fs_free_info(v_info);
        if (!cached) fs_close(f);
        return -1;
    }

    /* Read data (from mmap — zero-copy read) */
    const float *k_data = fs_get_f32(f, key_name);
    const float *v_data = fs_get_f32(f, val_name);
    if (!k_data || !v_data) {
        fs_free_info(k_info);
        fs_free_info(v_info);
        if (!cached) fs_close(f);
        return -1;
    }

    /* Create tensors (must copy — the mmap may be evicted from cache later) */
    tm_tensor *k_tensor = tm_zeros(k_info->shape, k_info->ndim, TM_F32);
    tm_tensor *v_tensor = tm_zeros(v_info->shape, v_info->ndim, TM_F32);
    if (!k_tensor || !v_tensor) {
        if (k_tensor) tm_free(k_tensor);
        if (v_tensor) tm_free(v_tensor);
        fs_free_info(k_info);
        fs_free_info(v_info);
        if (!cached) fs_close(f);
        return -1;
    }

    memcpy(k_tensor->data, k_data, k_info->data_size);
    memcpy(v_tensor->data, v_data, v_info->data_size);

    *keys_out = k_tensor;
    *values_out = v_tensor;

    fs_free_info(k_info);
    fs_free_info(v_info);
    if (!cached) fs_close(f);
    return 0;
}

/* ============================================================= */
/* Config                                                        */
/* ============================================================= */

fm_config fm_config_default(int embed_dim) {
    fm_config c;
    memset(&c, 0, sizeof(fm_config));
    c.store_dir = NULL;
    c.embed_dim = embed_dim;
    c.hdc_dim = 4096;
    c.seed = 42;
    c.max_entries = 0;          /* unlimited */
    c.max_disk_bytes = 0;       /* unlimited */
    c.shard_max_bytes = 256 * 1024 * 1024;  /* 256 MB */
    c.max_open_shards = 4;
    c.quantize_stored = false;
    return c;
}

/* ============================================================= */
/* Lifecycle                                                     */
/* ============================================================= */

fm_store* fm_store_new(fm_config *config) {
    if (!config || config->embed_dim <= 0) return NULL;

    fm_store *store = calloc(1, sizeof(fm_store));
    if (!store) return NULL;

    store->config = *config;
    store->next_id = 1;
    store->timestamp = 0;

    /* Set up store directory */
    if (config->store_dir) {
        snprintf(store->store_dir, sizeof(store->store_dir), "%s", config->store_dir);
        fm__ensure_dir(store->store_dir);
    }

    /* Create projector for HDC encoding */
    store->projector = fv_projector_new(config->embed_dim, config->hdc_dim, config->seed);
    if (!store->projector) {
        free(store);
        return NULL;
    }

    /* Initialize shard cache */
    int cache_cap = config->max_open_shards > 0 ? config->max_open_shards : 4;
    store->shard_cache = calloc(cache_cap, sizeof(fm__shard_cache_slot));
    if (store->shard_cache) {
        store->shard_cache_cap = cache_cap;
        for (int i = 0; i < cache_cap; i++)
            store->shard_cache[i].shard_idx = -1;
    }
    store->shard_cache_clock = 0;

    /* Initialize LSH index */
    store->lsh = fm__lsh_new(config->hdc_dim, config->seed);

    return store;
}

fm_store* fm_store_open(const char *store_dir) {
    if (!store_dir) return NULL;

    /* Read index.json for config */
    char index_path[600];
    snprintf(index_path, sizeof(index_path), "%s/index.json", store_dir);

    FILE *f = fopen(index_path, "r");
    if (!f) return NULL;

    /* Parse minimal JSON: embed_dim, hdc_dim, seed, next_id, num_shards */
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);

    int embed_dim = 0, hdc_dim = 4096, num_shards = 0;
    uint64_t seed = 42, next_id = 1;

    /* Simple JSON parsing (no external deps) */
    char *p;
    if ((p = strstr(buf, "\"embed_dim\""))) sscanf(p, "\"embed_dim\": %d", &embed_dim);
    if ((p = strstr(buf, "\"hdc_dim\""))) sscanf(p, "\"hdc_dim\": %d", &hdc_dim);
    if ((p = strstr(buf, "\"seed\""))) sscanf(p, "\"seed\": %llu", (unsigned long long *)&seed);
    if ((p = strstr(buf, "\"next_id\""))) sscanf(p, "\"next_id\": %llu", (unsigned long long *)&next_id);
    if ((p = strstr(buf, "\"num_shards\""))) sscanf(p, "\"num_shards\": %d", &num_shards);

    if (embed_dim <= 0) return NULL;

    fm_config config = fm_config_default(embed_dim);
    config.store_dir = store_dir;
    config.hdc_dim = hdc_dim;
    config.seed = seed;

    fm_store *store = fm_store_new(&config);
    if (!store) return NULL;

    store->next_id = next_id;

    /* Load entries from entries.bin */
    char entries_path[600];
    snprintf(entries_path, sizeof(entries_path), "%s/entries.bin", store_dir);

    FILE *ef = fopen(entries_path, "rb");
    if (ef) {
        /* Read entries */
        uint32_t entry_count;
        if (fread(&entry_count, sizeof(uint32_t), 1, ef) == 1) {
            for (uint32_t i = 0; i < entry_count; i++) {
                if (fm__grow_entries(store) != 0) break;

                fm_entry *e = &store->entries[store->num_entries];
                memset(e, 0, sizeof(fm_entry));

                /* Read fixed fields */
                if (fread(&e->entry_id, sizeof(uint64_t), 1, ef) != 1) break;
                uint32_t layer32;
                if (fread(&layer32, sizeof(uint32_t), 1, ef) != 1) break;
                e->layer_idx = (int)layer32;
                uint32_t ntok;
                if (fread(&ntok, sizeof(uint32_t), 1, ef) != 1) break;
                e->n_tokens = (int)ntok;
                if (fread(&e->importance, sizeof(float), 1, ef) != 1) break;
                if (fread(&e->timestamp, sizeof(uint64_t), 1, ef) != 1) break;
                if (fread(&e->access_count, sizeof(uint32_t), 1, ef) != 1) break;
                uint32_t shard32;
                if (fread(&shard32, sizeof(uint32_t), 1, ef) != 1) break;
                e->shard_idx = (int)shard32;
                uint32_t pad;
                if (fread(&pad, sizeof(uint32_t), 1, ef) != 1) break;

                /* Read HDC signature */
                e->signature = fv_alloc(hdc_dim);
                if (!e->signature) break;
                int words = (hdc_dim + 63) / 64;
                if (fread(e->signature->bits, sizeof(uint64_t), words, ef) != (size_t)words) {
                    fv_free(e->signature);
                    e->signature = NULL;
                    break;
                }

                e->evicted = false;
                store->num_entries++;
            }
        }
        fclose(ef);
    }

    /* Reconstruct shard tracking */
    for (int i = 0; i < num_shards; i++) {
        if (fm__grow_shards(store) != 0) break;
        fm_shard *s = &store->shards[store->num_shards];
        memset(s, 0, sizeof(fm_shard));
        fm__shard_path(store, i, s->path, sizeof(s->path));
        struct stat st;
        if (stat(s->path, &st) == 0) {
            s->byte_size = st.st_size;
            store->disk_bytes_used += st.st_size;
        }
        store->num_shards++;
    }

    /* Populate LSH index with loaded entries */
    if (store->lsh) {
        for (size_t i = 0; i < store->num_entries; i++) {
            fm_entry *e = &store->entries[i];
            if (!e->evicted && e->signature) {
                fm__lsh_insert(store->lsh, i, e->signature);
            }
        }
    }

    return store;
}

void fm_store_free(fm_store *store) {
    if (!store) return;
    fm__lsh_free(store->lsh);
    fm__write_buf_free(store);
    fm__shard_cache_free(store);
    for (size_t i = 0; i < store->num_entries; i++) {
        fv_free(store->entries[i].signature);
    }
    free(store->entries);
    free(store->shards);
    fv_projector_free(store->projector);
    free(store);
}

int fm_store_flush(fm_store *store) {
    if (!store || !store->store_dir[0]) return -1;

    fm__ensure_dir(store->store_dir);

    /* Flush write buffer to disk first */
    if (fm__write_buf_flush(store) != 0) return -1;

    /* Write index.json */
    char index_path[600];
    snprintf(index_path, sizeof(index_path), "%s/index.json", store->store_dir);
    FILE *f = fopen(index_path, "w");
    if (!f) return -1;
    fprintf(f, "{\n");
    fprintf(f, "  \"embed_dim\": %d,\n", store->config.embed_dim);
    fprintf(f, "  \"hdc_dim\": %d,\n", store->config.hdc_dim);
    fprintf(f, "  \"seed\": %llu,\n", (unsigned long long)store->config.seed);
    fprintf(f, "  \"next_id\": %llu,\n", (unsigned long long)store->next_id);
    fprintf(f, "  \"num_entries\": %zu,\n", store->num_entries);
    fprintf(f, "  \"num_shards\": %d\n", store->num_shards);
    fprintf(f, "}\n");
    fclose(f);

    /* Write entries.bin */
    char entries_path[600];
    snprintf(entries_path, sizeof(entries_path), "%s/entries.bin", store->store_dir);
    FILE *ef = fopen(entries_path, "wb");
    if (!ef) return -1;

    /* Count non-evicted entries */
    uint32_t live_count = 0;
    for (size_t i = 0; i < store->num_entries; i++) {
        if (!store->entries[i].evicted) live_count++;
    }
    fwrite(&live_count, sizeof(uint32_t), 1, ef);

    int hdc_dim = store->config.hdc_dim;
    int words = (hdc_dim + 63) / 64;

    for (size_t i = 0; i < store->num_entries; i++) {
        fm_entry *e = &store->entries[i];
        if (e->evicted) continue;

        fwrite(&e->entry_id, sizeof(uint64_t), 1, ef);
        uint32_t layer32 = (uint32_t)e->layer_idx;
        fwrite(&layer32, sizeof(uint32_t), 1, ef);
        uint32_t ntok = (uint32_t)e->n_tokens;
        fwrite(&ntok, sizeof(uint32_t), 1, ef);
        fwrite(&e->importance, sizeof(float), 1, ef);
        fwrite(&e->timestamp, sizeof(uint64_t), 1, ef);
        fwrite(&e->access_count, sizeof(uint32_t), 1, ef);
        uint32_t shard32 = (uint32_t)e->shard_idx;
        fwrite(&shard32, sizeof(uint32_t), 1, ef);
        uint32_t pad = 0;
        fwrite(&pad, sizeof(uint32_t), 1, ef);

        /* Write signature bits */
        if (e->signature) {
            fwrite(e->signature->bits, sizeof(uint64_t), words, ef);
        } else {
            /* Write zeros if no signature */
            for (int w = 0; w < words; w++) {
                uint64_t z = 0;
                fwrite(&z, sizeof(uint64_t), 1, ef);
            }
        }
    }

    fclose(ef);
    return 0;
}

/* ============================================================= */
/* Store                                                         */
/* ============================================================= */

uint64_t fm_put(fm_store *store, tm_tensor *keys, tm_tensor *values,
                int layer_idx, float importance) {
    if (!store || !keys || !values) return 0;
    if (keys->ndim != 2 || values->ndim != 2) return 0;
    if (keys->dtype != TM_F32 || values->dtype != TM_F32) return 0;
    if (keys->shape[0] != values->shape[0]) return 0;
    if (keys->shape[1] != store->config.embed_dim) return 0;
    if (values->shape[1] != store->config.embed_dim) return 0;

    /* Auto-evict if at capacity */
    if (store->config.max_entries > 0 && store->num_entries >= store->config.max_entries) {
        fm_evict(store, store->config.max_entries - 1);
    }

    /* Mean-pool keys for HDC signature */
    tm_tensor *pooled = fm__mean_pool(keys);
    if (!pooled) return 0;

    fv_vector *sig = fv_project(store->projector, pooled);
    tm_free(pooled);
    if (!sig) return 0;

    /* Get shard for storage */
    int shard_idx = fm__current_shard(store);
    if (shard_idx < 0) { fv_free(sig); return 0; }

    uint64_t id = store->next_id++;

    /* Buffer K/V data in memory (deferred write — flushed on fm_store_flush) */
    if (store->store_dir[0]) {
        fm__ensure_dir(store->store_dir);
        if (fm__write_buf_add(store, id, keys, values, shard_idx) != 0) {
            fv_free(sig);
            return 0;
        }
    }

    /* Add entry to index */
    if (fm__grow_entries(store) != 0) { fv_free(sig); return 0; }

    fm_entry *e = &store->entries[store->num_entries++];
    memset(e, 0, sizeof(fm_entry));
    e->entry_id = id;
    e->layer_idx = layer_idx;
    e->n_tokens = (int)keys->shape[0];
    e->importance = importance;
    e->timestamp = store->timestamp++;
    e->access_count = 0;
    e->shard_idx = shard_idx;
    e->evicted = false;
    e->signature = sig;

    /* Insert into LSH index */
    fm__lsh_insert(store->lsh, store->num_entries - 1, sig);

    store->total_stores++;

    return id;
}

uint64_t fm_put_presig(fm_store *store, tm_tensor *keys, tm_tensor *values,
                       int layer_idx, float importance, fv_vector *presig) {
    if (!store || !keys || !values || !presig) return 0;
    if (keys->ndim != 2 || values->ndim != 2) return 0;
    if (keys->dtype != TM_F32 || values->dtype != TM_F32) return 0;
    if (keys->shape[0] != values->shape[0]) return 0;
    if (keys->shape[1] != store->config.embed_dim) return 0;
    if (values->shape[1] != store->config.embed_dim) return 0;

    /* Auto-evict if at capacity */
    if (store->config.max_entries > 0 && store->num_entries >= store->config.max_entries) {
        fm_evict(store, store->config.max_entries - 1);
    }

    /* Get shard for storage */
    int shard_idx = fm__current_shard(store);
    if (shard_idx < 0) { fv_free(presig); return 0; }

    uint64_t id = store->next_id++;

    /* Buffer K/V data in memory */
    if (store->store_dir[0]) {
        fm__ensure_dir(store->store_dir);
        if (fm__write_buf_add(store, id, keys, values, shard_idx) != 0) {
            fv_free(presig);
            return 0;
        }
    }

    /* Add entry to index */
    if (fm__grow_entries(store) != 0) { fv_free(presig); return 0; }

    fm_entry *e = &store->entries[store->num_entries++];
    memset(e, 0, sizeof(fm_entry));
    e->entry_id = id;
    e->layer_idx = layer_idx;
    e->n_tokens = (int)keys->shape[0];
    e->importance = importance;
    e->timestamp = store->timestamp++;
    e->access_count = 0;
    e->shard_idx = shard_idx;
    e->evicted = false;
    e->signature = presig;  /* takes ownership */

    /* Insert into LSH index */
    fm__lsh_insert(store->lsh, store->num_entries - 1, presig);

    store->total_stores++;

    return id;
}

fv_projector* fm_store_projector(fm_store *store) {
    return store ? store->projector : NULL;
}

/* ============================================================= */
/* Query                                                         */
/* ============================================================= */

/* Min-heap for top-k selection: O(N log k) instead of O(N²) sort.
 * Maintains a heap of size k where the root is the SMALLEST score.
 * When a new candidate has score > root, replace root and sift down. */

typedef struct { size_t idx; float score; } fm__heap_item;

static void fm__heap_sift_down(fm__heap_item *heap, int n, int i) {
    while (1) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left < n && heap[left].score < heap[smallest].score)
            smallest = left;
        if (right < n && heap[right].score < heap[smallest].score)
            smallest = right;
        if (smallest == i) break;
        fm__heap_item tmp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = tmp;
        i = smallest;
    }
}

static void fm__heap_sift_up(fm__heap_item *heap, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap[i].score >= heap[parent].score) break;
        fm__heap_item tmp = heap[i];
        heap[i] = heap[parent];
        heap[parent] = tmp;
        i = parent;
    }
}

fm_results* fm_query(fm_store *store, tm_tensor *query,
                     int layer_idx, int top_k) {
    if (!store || !query) return NULL;
    if (query->ndim != 1 || query->dtype != TM_F32) return NULL;
    if (query->shape[0] != store->config.embed_dim) return NULL;
    if (top_k <= 0) return NULL;

    store->total_queries++;

    /* Project query to HDC */
    fv_vector *query_hv = fv_project(store->projector, query);
    if (!query_hv) return NULL;

    /* Min-heap of size top_k: root = smallest score in heap.
     * Single allocation, zero reallocations. */
    fm__heap_item *heap = malloc(top_k * sizeof(fm__heap_item));
    if (!heap) { fv_free(query_hv); return NULL; }
    int heap_size = 0;

    /* Use LSH index if available and store is large enough to benefit.
     * For small stores (<64 entries), linear scan is faster. */
    bool used_lsh = false;
    if (store->lsh && store->lsh->built && store->num_entries >= 64) {
        size_t *candidates = NULL;
        int n_cand = fm__lsh_query(store->lsh, query_hv, &candidates);
        if (n_cand > 0) {
            used_lsh = true;
            for (int ci = 0; ci < n_cand; ci++) {
                size_t i = candidates[ci];
                if (i >= store->num_entries) continue;
                fm_entry *e = &store->entries[i];
                if (e->evicted) continue;
                if (e->layer_idx != layer_idx) continue;

                float score = fv_similarity(query_hv, e->signature);

                if (heap_size < top_k) {
                    heap[heap_size].idx = i;
                    heap[heap_size].score = score;
                    heap_size++;
                    fm__heap_sift_up(heap, heap_size - 1);
                } else if (score > heap[0].score) {
                    heap[0].idx = i;
                    heap[0].score = score;
                    fm__heap_sift_down(heap, heap_size, 0);
                }
            }
            free(candidates);
        }
    }

    /* Fallback: linear scan (used when LSH not available or store is small) */
    if (!used_lsh) {
        for (size_t i = 0; i < store->num_entries; i++) {
            fm_entry *e = &store->entries[i];
            if (e->evicted) continue;
            if (e->layer_idx != layer_idx) continue;

            float score = fv_similarity(query_hv, e->signature);

            if (heap_size < top_k) {
                heap[heap_size].idx = i;
                heap[heap_size].score = score;
                heap_size++;
                fm__heap_sift_up(heap, heap_size - 1);
            } else if (score > heap[0].score) {
                heap[0].idx = i;
                heap[0].score = score;
                fm__heap_sift_down(heap, heap_size, 0);
            }
        }
    }

    fv_free(query_hv);

    if (heap_size == 0) {
        free(heap);
        fm_results *r = calloc(1, sizeof(fm_results));
        return r;
    }

    /* Extract heap items sorted by score descending.
     * Pop from heap (extract-min) into array, then reverse. */
    fm__heap_item *sorted = malloc(heap_size * sizeof(fm__heap_item));
    if (!sorted) { free(heap); return NULL; }

    int n_sorted = heap_size;
    for (int i = heap_size - 1; i >= 0; i--) {
        sorted[i] = heap[0];
        heap[0] = heap[heap_size - 1];
        heap_size--;
        if (heap_size > 0) fm__heap_sift_down(heap, heap_size, 0);
    }
    free(heap);

    /* Build results */
    fm_results *results = calloc(1, sizeof(fm_results));
    if (!results) { free(sorted); return NULL; }

    results->items = calloc(n_sorted, sizeof(fm_result));
    if (!results->items) { free(sorted); free(results); return NULL; }
    results->count = 0;

    for (int i = 0; i < n_sorted; i++) {
        size_t eidx = sorted[i].idx;
        fm_entry *e = &store->entries[eidx];

        fm_result *r = &results->items[results->count];
        r->entry_id = e->entry_id;
        r->layer_idx = e->layer_idx;
        r->score = sorted[i].score;
        r->keys = NULL;
        r->values = NULL;

        /* Load K/V — check write buffer first, then disk */
        if (store->store_dir[0]) {
            if (fm__write_buf_load(store, e->entry_id, &r->keys, &r->values) != 0) {
                if (fm__load_from_shard(store, e->shard_idx, e->entry_id,
                                        &r->keys, &r->values) != 0) {
                    continue; /* skip this result if load fails */
                }
            }
        }

        e->access_count++;
        results->count++;
        store->total_hits++;
    }

    free(sorted);
    return results;
}

void fm_results_free(fm_results *results) {
    if (!results) return;
    for (int i = 0; i < results->count; i++) {
        tm_free(results->items[i].keys);
        tm_free(results->items[i].values);
    }
    free(results->items);
    free(results);
}

/* ============================================================= */
/* Management                                                    */
/* ============================================================= */

fm_stats fm_stats_get(fm_store *store) {
    fm_stats s;
    memset(&s, 0, sizeof(fm_stats));
    if (!store) return s;
    s.total_stores = store->total_stores;
    s.total_queries = store->total_queries;
    s.total_hits = store->total_hits;
    s.disk_bytes_used = store->disk_bytes_used;
    /* Include buffered (not yet flushed) data in the size estimate */
    for (int i = 0; i < store->write_buf_count; i++) {
        s.disk_bytes_used += (size_t)store->write_buf[i].n_tokens
                             * store->config.embed_dim * sizeof(float) * 2;
    }
    /* Count live entries */
    s.num_entries = 0;
    for (size_t i = 0; i < store->num_entries; i++) {
        if (!store->entries[i].evicted) s.num_entries++;
    }
    return s;
}

size_t fm_store_size(fm_store *store) {
    if (!store) return 0;
    size_t count = 0;
    for (size_t i = 0; i < store->num_entries; i++) {
        if (!store->entries[i].evicted) count++;
    }
    return count;
}

size_t fm_evict(fm_store *store, size_t target_entries) {
    if (!store) return 0;
    size_t live = fm_store_size(store);
    if (live <= target_entries) return 0;
    size_t to_evict = live - target_entries;
    size_t evicted = 0;

    /* Score entries: lower score = evict first
     * Score = importance * (1 + log(1 + access_count)) */
    typedef struct { size_t idx; float score; } scored;
    scored *ranking = malloc(live * sizeof(scored));
    if (!ranking) return 0;

    int ri = 0;
    for (size_t i = 0; i < store->num_entries; i++) {
        if (store->entries[i].evicted) continue;
        fm_entry *e = &store->entries[i];
        ranking[ri].idx = i;
        ranking[ri].score = e->importance * (1.0f + logf(1.0f + (float)e->access_count));
        ri++;
    }

    /* Quickselect: partition to find the `to_evict` lowest-scored entries.
     * O(N) average instead of O(N²) selection sort. We don't need them
     * sorted — just need the bottom-k separated from the rest. */
    {
        int k = (int)to_evict;
        if (k > ri) k = ri;
        int lo = 0, hi = ri - 1;
        while (lo < hi) {
            /* Median-of-three pivot selection */
            int mid = lo + (hi - lo) / 2;
            if (ranking[mid].score < ranking[lo].score) {
                scored tmp = ranking[lo]; ranking[lo] = ranking[mid]; ranking[mid] = tmp;
            }
            if (ranking[hi].score < ranking[lo].score) {
                scored tmp = ranking[lo]; ranking[lo] = ranking[hi]; ranking[hi] = tmp;
            }
            if (ranking[mid].score < ranking[hi].score) {
                scored tmp = ranking[mid]; ranking[mid] = ranking[hi]; ranking[hi] = tmp;
            }
            float pivot = ranking[hi].score;
            int i = lo, j = hi - 1;
            for (;;) {
                while (i <= j && ranking[i].score <= pivot) i++;
                while (j >= i && ranking[j].score > pivot) j--;
                if (i >= j) break;
                scored tmp = ranking[i]; ranking[i] = ranking[j]; ranking[j] = tmp;
            }
            scored tmp = ranking[i]; ranking[i] = ranking[hi]; ranking[hi] = tmp;
            if (i == k - 1) break;
            else if (i < k - 1) lo = i + 1;
            else hi = i - 1;
        }
    }

    /* Evict lowest-scored entries (now in ranking[0..to_evict-1]) */
    for (size_t i = 0; i < to_evict && i < (size_t)ri; i++) {
        fm_entry *e = &store->entries[ranking[i].idx];
        e->evicted = true;
        fv_free(e->signature);
        e->signature = NULL;
        evicted++;
    }

    free(ranking);
    return evicted;
}

int fm_compact(fm_store *store) {
    if (!store || !store->store_dir[0]) return -1;

    /* Flush buffered writes to disk before compacting */
    fm__write_buf_flush(store);

    /* Rewrite all shards, excluding evicted entries */
    for (int si = 0; si < store->num_shards; si++) {
        fm_shard *shard = &store->shards[si];
        fm__shard_cache_invalidate(store, si);
        fs_file *old_file = fs_open(shard->path);
        if (!old_file) continue;

        fs_writer *w = fs_writer_new();
        if (!w) { fs_close(old_file); continue; }

        bool has_entries = false;
        for (size_t i = 0; i < store->num_entries; i++) {
            fm_entry *e = &store->entries[i];
            if (e->evicted || e->shard_idx != si) continue;

            char key_name[64], val_name[64];
            snprintf(key_name, sizeof(key_name), "k_%llu", (unsigned long long)e->entry_id);
            snprintf(val_name, sizeof(val_name), "v_%llu", (unsigned long long)e->entry_id);

            fs_tensor_info *k_info = fs_get_info(old_file, key_name);
            fs_tensor_info *v_info = fs_get_info(old_file, val_name);
            if (k_info && v_info) {
                const void *k_data = fs_get_data(old_file, key_name);
                const void *v_data = fs_get_data(old_file, val_name);
                if (k_data && v_data) {
                    fs_writer_add_tensor(w, key_name, k_info->dtype,
                                         k_info->shape, k_info->ndim, k_data);
                    fs_writer_add_tensor(w, val_name, v_info->dtype,
                                         v_info->shape, v_info->ndim, v_data);
                    has_entries = true;
                }
            }
            if (k_info) fs_free_info(k_info);
            if (v_info) fs_free_info(v_info);
        }

        fs_close(old_file);

        if (has_entries) {
            fs_writer_save(w, shard->path);
            struct stat st;
            if (stat(shard->path, &st) == 0) {
                shard->byte_size = st.st_size;
            }
        } else {
            /* No live entries — remove shard file */
            unlink(shard->path);
            shard->byte_size = 0;
        }
        fs_writer_free(w);
    }

    /* Recalculate disk usage */
    store->disk_bytes_used = 0;
    for (int i = 0; i < store->num_shards; i++) {
        store->disk_bytes_used += store->shards[i].byte_size;
    }

    return 0;
}

/* ============================================================= */
/* Integration Bridge                                            */
/* ============================================================= */

int fm_augmented_kv(tm_tensor *query, tm_tensor *cache_k, tm_tensor *cache_v,
                    fm_store *store, int layer_idx, int top_k,
                    tm_tensor **merged_k_out, tm_tensor **merged_v_out) {
    if (!cache_k || !cache_v || !merged_k_out || !merged_v_out) return -1;
    if (cache_k->ndim != 2 || cache_v->ndim != 2) return -1;
    if (cache_k->dtype != TM_F32 || cache_v->dtype != TM_F32) return -1;

    int64_t cache_len = cache_k->shape[0];
    int64_t embed_dim = cache_k->shape[1];

    /* If no store or no query, just clone cache */
    if (!store || !query || top_k <= 0) {
        *merged_k_out = tm_clone(cache_k);
        *merged_v_out = tm_clone(cache_v);
        return (*merged_k_out && *merged_v_out) ? 0 : -1;
    }

    /* Query store */
    fm_results *results = fm_query(store, query, layer_idx, top_k);
    if (!results || results->count == 0) {
        fm_results_free(results);
        *merged_k_out = tm_clone(cache_k);
        *merged_v_out = tm_clone(cache_v);
        return (*merged_k_out && *merged_v_out) ? 0 : -1;
    }

    /* Calculate total tokens from retrieved entries */
    int64_t retrieved_tokens = 0;
    for (int i = 0; i < results->count; i++) {
        if (results->items[i].keys) {
            retrieved_tokens += results->items[i].keys->shape[0];
        }
    }

    /* Allocate merged tensors: [cache_len + retrieved_tokens, embed_dim] */
    int64_t total_len = cache_len + retrieved_tokens;
    int64_t merged_shape[2] = { total_len, embed_dim };
    tm_tensor *mk = tm_zeros(merged_shape, 2, TM_F32);
    tm_tensor *mv = tm_zeros(merged_shape, 2, TM_F32);
    if (!mk || !mv) {
        tm_free(mk);
        tm_free(mv);
        fm_results_free(results);
        return -1;
    }

    float *mk_data = (float *)mk->data;
    float *mv_data = (float *)mv->data;

    /* Copy cache first */
    memcpy(mk_data, cache_k->data, cache_len * embed_dim * sizeof(float));
    memcpy(mv_data, cache_v->data, cache_len * embed_dim * sizeof(float));

    /* Append retrieved entries */
    int64_t offset = cache_len;
    for (int i = 0; i < results->count; i++) {
        fm_result *r = &results->items[i];
        if (!r->keys || !r->values) continue;
        int64_t n = r->keys->shape[0];
        memcpy(mk_data + offset * embed_dim, r->keys->data, n * embed_dim * sizeof(float));
        memcpy(mv_data + offset * embed_dim, r->values->data, n * embed_dim * sizeof(float));
        offset += n;
    }

    fm_results_free(results);
    *merged_k_out = mk;
    *merged_v_out = mv;
    return 0;
}
