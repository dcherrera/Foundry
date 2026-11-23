/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_HYBRIDMEM_H
#define FOUNDRY_HYBRIDMEM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * HybridMem: Persistent disk-backed memory tier (Tier 3) for RWKV-X.
 * Uses HDC for fast similarity search, safetensors for exact K/V data.
 *
 * Three-tier memory architecture:
 *   Tier 1: RWKV-7 recurrent state      (constant size, lossy)
 *   Tier 2: Sparse attention KV cache    (bounded, in-memory)
 *   Tier 3: HybridMem                    (disk, unlimited, precise)
 */

/* Forward declarations */
typedef struct tm_tensor tm_tensor;

#include "hdc/hdc.h"

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* Opaque store handle */
typedef struct fm_store fm_store;

/* Configuration */
typedef struct {
    const char *store_dir;       /* Directory for persistent storage */
    int embed_dim;               /* Model embedding dimension (e.g. 3584) */
    int hdc_dim;                 /* HDC dimension for index (default: 4096) */
    uint64_t seed;               /* Seed for projection matrix */
    size_t max_entries;          /* Max entries before eviction (0 = unlimited) */
    size_t max_disk_bytes;       /* Max disk usage (0 = unlimited) */
    size_t shard_max_bytes;      /* Max per shard file (default: 256 MB) */
    int max_open_shards;         /* Max concurrent shard readers (default: 4) */
    bool quantize_stored;        /* INT8 quantize stored K/V (saves ~4x disk) */
} fm_config;

/* Single query result */
typedef struct {
    uint64_t entry_id;
    int layer_idx;
    float score;                 /* Similarity score */
    tm_tensor *keys;             /* [n_tokens, embed_dim] */
    tm_tensor *values;           /* [n_tokens, embed_dim] */
} fm_result;

/* Query result set */
typedef struct {
    fm_result *items;
    int count;
} fm_results;

/* Store statistics */
typedef struct {
    size_t total_stores;
    size_t total_queries;
    size_t total_hits;
    size_t disk_bytes_used;
    size_t num_entries;
} fm_stats;

/* ============================================================= */
/* Lifecycle                                                     */
/* ============================================================= */

/* Get default config with sensible defaults */
fm_config fm_config_default(int embed_dim);

/* Create new store from config */
fm_store* fm_store_new(fm_config *config);

/* Reopen existing store from disk */
fm_store* fm_store_open(const char *store_dir);

/* Free store (does NOT flush — call fm_store_flush first) */
void fm_store_free(fm_store *store);

/* Flush index and metadata to disk */
int fm_store_flush(fm_store *store);

/* ============================================================= */
/* Store (eviction path)                                         */
/* ============================================================= */

/* Store K/V entry. Returns entry_id (0 on error). */
uint64_t fm_put(fm_store *store, tm_tensor *keys, tm_tensor *values,
                int layer_idx, float importance);

/* Store K/V entry with pre-computed HDC signature (skips projection).
 * Takes ownership of presig (caller must NOT free it). */
uint64_t fm_put_presig(fm_store *store, tm_tensor *keys, tm_tensor *values,
                       int layer_idx, float importance, fv_vector *presig);

/* Get the projection matrix used by the store (for incremental projection).
 * Returns NULL if store is NULL. Caller must NOT free the returned pointer. */
fv_projector* fm_store_projector(fm_store *store);

/* ============================================================= */
/* Query (retrieval path)                                        */
/* ============================================================= */

/* Find top-k similar entries for query embedding at given layer.
 * query: [embed_dim] float tensor (mean-pooled from K/V).
 * Returns NULL on error. Caller must free with fm_results_free(). */
fm_results* fm_query(fm_store *store, tm_tensor *query,
                     int layer_idx, int top_k);

/* Free query results (including contained tensors) */
void fm_results_free(fm_results *results);

/* ============================================================= */
/* Management                                                    */
/* ============================================================= */

/* Get store statistics */
fm_stats fm_stats_get(fm_store *store);

/* Get number of entries */
size_t fm_store_size(fm_store *store);

/* Evict least-important entries until at most target_entries remain.
 * Returns number of entries evicted. */
size_t fm_evict(fm_store *store, size_t target_entries);

/* Compact shards: rewrite to reclaim space from evicted entries.
 * Returns 0 on success. */
int fm_compact(fm_store *store);

/* ============================================================= */
/* Integration Bridge                                            */
/* ============================================================= */

/* Merge cache K/V with retrieved disk entries.
 * cache_k, cache_v: existing in-memory cache [seq_len, embed_dim]
 * store: HybridMem store (may be NULL for cache-only mode)
 * query: [embed_dim] query for retrieval
 * merged_k_out, merged_v_out: output merged tensors (caller must free)
 * Returns 0 on success. */
int fm_augmented_kv(tm_tensor *query, tm_tensor *cache_k, tm_tensor *cache_v,
                    fm_store *store, int layer_idx, int top_k,
                    tm_tensor **merged_k_out, tm_tensor **merged_v_out);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_HYBRIDMEM_H */
