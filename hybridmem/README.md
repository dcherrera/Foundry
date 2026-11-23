# HybridMem Module (`fm_`)

> Persistent disk-backed memory tier (Tier 3) for RWKV-X inference.

---

## Overview

HybridMem extends RWKV-X's context window to disk, providing unlimited persistent memory with precise K/V retrieval. It uses Hyperdimensional Computing (HDC) for fast similarity search and safetensors for zero-copy K/V data storage.

### Three-Tier Memory Architecture

```
Tier 1: RWKV-7 Recurrent State     ~50 MB     constant     lossy
Tier 2: Sparse Attention Cache      ~200 MB    bounded      exact
Tier 3: HybridMem (this module)     disk       unlimited    exact
```

## API Reference

### Types

```c
typedef struct fm_store fm_store;   /* Opaque store handle */

typedef struct {
    const char *store_dir;       /* Storage directory */
    int embed_dim;               /* Model embedding dimension */
    int hdc_dim;                 /* HDC index dimension (default: 4096) */
    uint64_t seed;               /* Projection seed */
    size_t max_entries;          /* Max entries (0 = unlimited) */
    size_t max_disk_bytes;       /* Max disk (0 = unlimited) */
    size_t shard_max_bytes;      /* Per-shard limit (default: 256 MB) */
    int max_open_shards;         /* Concurrent readers (default: 4) */
    bool quantize_stored;        /* INT8 compress K/V */
} fm_config;

typedef struct {
    uint64_t entry_id;
    int layer_idx;
    float score;
    tm_tensor *keys;             /* [n_tokens, embed_dim] */
    tm_tensor *values;           /* [n_tokens, embed_dim] */
} fm_result;

typedef struct {
    fm_result *items;
    int count;
} fm_results;
```

### Lifecycle

```c
fm_config  fm_config_default(int embed_dim);
fm_store*  fm_store_new(fm_config *config);
fm_store*  fm_store_open(const char *store_dir);   /* Reopen from disk */
void       fm_store_free(fm_store *store);
int        fm_store_flush(fm_store *store);         /* Persist to disk */
```

### Store & Query

```c
/* Store K/V entry with importance score. Returns entry_id. */
uint64_t fm_put(fm_store *store, tm_tensor *keys, tm_tensor *values,
                int layer_idx, float importance);

/* Retrieve top-k entries matching query at layer. */
fm_results* fm_query(fm_store *store, tm_tensor *query,
                     int layer_idx, int top_k);
void fm_results_free(fm_results *results);
```

### Management

```c
fm_stats fm_stats_get(fm_store *store);
size_t   fm_store_size(fm_store *store);
size_t   fm_evict(fm_store *store, size_t target_entries);
int      fm_compact(fm_store *store);
```

### Integration Bridge

```c
/* Merge cache K/V with disk-retrieved entries for sparse attention */
int fm_augmented_kv(tm_tensor *query, tm_tensor *cache_k, tm_tensor *cache_v,
                    fm_store *store, int layer_idx, int top_k,
                    tm_tensor **merged_k_out, tm_tensor **merged_v_out);
```

## Storage Format

```
<store_dir>/
├── index.json              # Config + metadata
├── entries.bin              # Entry metadata + HDC signatures
├── shard_0000.safetensors  # K/V data (mmap'd via safetensors)
├── shard_0001.safetensors
└── ...
```

### entries.bin (per entry)

| Field | Type | Bytes |
|-------|------|-------|
| entry_id | uint64 | 8 |
| layer_idx | uint32 | 4 |
| n_tokens | uint32 | 4 |
| importance | float32 | 4 |
| timestamp | uint64 | 8 |
| access_count | uint32 | 4 |
| shard_idx | uint32 | 4 |
| padding | uint32 | 4 |
| signature | hdc_dim/8 bytes | 512 (for 4096-dim) |

### Shard files

Standard safetensors format with tensors named `k_{entry_id}` and `v_{entry_id}`, shape `[n_tokens, embed_dim]`, dtype F32.

## Query Flow

```
1. Mean-pool query tokens → [embed_dim] vector
2. Project to HDC:        query_hv = fv_project(projector, pooled)
3. Search HDC index:      fv_similarity(query_hv, signature) for each entry
4. Filter by layer_idx
5. Sort by score, take top-k
6. Load K/V from shards:  fs_open(shard) → fs_get_f32("k_{id}")
7. Return results
```

## Usage Example

```c
#include "hybridmem/hybridmem.h"

/* Create store */
fm_config cfg = fm_config_default(3584);  /* RWKV-X 3.6B embed_dim */
cfg.store_dir = "/data/hybridmem";
cfg.max_entries = 100000;
fm_store *store = fm_store_new(&cfg);

/* Store evicted K/V from sparse attention */
uint64_t id = fm_put(store, keys_tensor, values_tensor, layer_idx, importance);

/* Query during inference */
fm_results *results = fm_query(store, query_embedding, layer_idx, 5);
for (int i = 0; i < results->count; i++) {
    /* results->items[i].keys and .values are ready to use */
}
fm_results_free(results);

/* Or use the integration bridge directly */
tm_tensor *merged_k, *merged_v;
fm_augmented_kv(query, cache_k, cache_v, store, layer, 5, &merged_k, &merged_v);

/* Persist */
fm_store_flush(store);
fm_store_free(store);
```

## Eviction Policy

Entries are scored by: `importance * (1 + log(1 + access_count))`

Lowest-scored entries are evicted first. This favors entries that are both important (high initial score) and frequently accessed (high access count).

## Build

```bash
cd foundry
make hybridmem           # Build module
make test-hybridmem      # Run 20 tests
```

## Dependencies

- `hdc/` — HDC similarity search index
- `tensor/` — Tensor data types
- `safetensors/` — K/V data storage and mmap
