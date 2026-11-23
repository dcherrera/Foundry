# Retrieval

ASIC-accelerated hash-based document retrieval for LLM memory augmentation.

## Why Hash-Based Retrieval?

Traditional retrieval uses dense embeddings + cosine similarity:
- Compute 768+ dim embedding for query
- Compare against millions of document embeddings
- GPU-heavy, memory-bound

Hash-based retrieval uses locality-sensitive hashing:
- Compute hash signature for query (fast)
- Find documents with similar hash signatures (bit operations)
- CPU/ASIC-friendly, embarrassingly parallel

**The ASIC advantage**: SHA-256 miners compute billions of hashes per second at ~1W.
If retrieval reduces to hashing, you get 1000x better cost/watt than GPU.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Foundry LLM                                 │
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────────────┐ │
│  │ Tokenize │───▶│ Attention │───▶│ Generate                 │ │
│  └──────────┘    └─────┬─────┘    └──────────────────────────┘ │
│                        │                                        │
│                        │ retrieve context                       │
│                        ▼                                        │
│              ┌─────────────────┐                                │
│              │  fr_retrieval   │ ◀── This module               │
│              └────────┬────────┘                                │
│                       │                                         │
└───────────────────────┼─────────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
   ┌─────────────┐            ┌─────────────┐
   │ CPU Backend │            │ASIC Backend │
   │ (fallback)  │            │ (external)  │
   └─────────────┘            └─────────────┘
                                     │
                              ┌──────┴──────┐
                              │ USB/UART    │
                              │ ASIC Miner  │
                              └─────────────┘
```

## Hash Techniques

| Technique | Use Case | How It Works |
|-----------|----------|--------------|
| SimHash | Text similarity | Hash of weighted token features |
| MinHash | Set similarity (Jaccard) | Min of many hash permutations |
| LSH | Approximate nearest neighbor | Hash buckets group similar items |

### SimHash for Text

```
Document: "the quick brown fox"

1. Extract features: {"the": 1, "quick": 1, "brown": 1, "fox": 1}
2. For each feature, compute SHA256(feature) → 256 bits
3. For each bit position:
   - If bit is 1, add feature weight (+1)
   - If bit is 0, subtract feature weight (-1)
4. Final signature: sign(sum) for each position → 256 bits

Similar documents have similar bit patterns.
Hamming distance ≈ cosine distance.
```

### LSH Index Structure

```
┌─────────────────────────────────────────┐
│              LSH Index                  │
├─────────────────────────────────────────┤
│ Band 0: hash(sig[0:32])   → bucket_id   │
│ Band 1: hash(sig[32:64])  → bucket_id   │
│ Band 2: hash(sig[64:96])  → bucket_id   │
│ ...                                     │
│ Band 7: hash(sig[224:256]) → bucket_id  │
└─────────────────────────────────────────┘

Query: Check all bands, candidates = union of matching buckets
       Then rank candidates by full signature similarity
```

## ASIC Device Interface

The ASIC backend expects an external device abstraction (not in this tree):

```c
// expected interface:
typedef struct asic_device asic_device;
asic_device* asic_open_mock(void);
asic_device* asic_open_lv03(const char *port);
int asic_hash_batch(asic_device *dev, uint8_t **inputs, size_t n, uint8_t **outputs);
```

This module builds on top:

```c
// Compute SimHash signature using ASIC
fr_signature* fr_simhash_asic(asic_device *dev, const char *text);

// Batch compute signatures (saturates ASIC throughput)
int fr_simhash_batch_asic(asic_device *dev, const char **texts, size_t n,
                          fr_signature **sigs);
```

## API Overview

### Types

```c
fr_signature     // 256-bit hash signature
fr_index         // LSH index structure
fr_result        // Single retrieval result (doc_id, score)
fr_results       // List of retrieval results
fr_config        // Retrieval configuration
```

### Core Functions

```c
// Signatures
fr_signature* fr_simhash(const char *text);
fr_signature* fr_minhash(const char **tokens, size_t n);
int fr_hamming(fr_signature *a, fr_signature *b);

// Indexing
fr_index* fr_index_new(fr_config *config);
int fr_index_add(fr_index *idx, uint64_t doc_id, fr_signature *sig);
int fr_index_add_text(fr_index *idx, uint64_t doc_id, const char *text);

// Retrieval
fr_results* fr_search(fr_index *idx, fr_signature *query, int top_k);
fr_results* fr_search_text(fr_index *idx, const char *query, int top_k);

// ASIC acceleration
void fr_set_backend(fr_index *idx, asic_device *dev);
```

## Usage Example

### Basic RAG Pipeline

```c
#include "retrieval/retrieval.h"
#include "asic.h"  // external ASIC interface, not in this tree

// Setup
fr_config config = fr_config_default();
fr_index *idx = fr_index_new(&config);

// Optional: Enable ASIC acceleration
asic_device *asic = asic_open_lv03("/dev/tty.usbserial-0001");
if (asic) {
    fr_set_backend(idx, asic);
}

// Index documents
fr_index_add_text(idx, 0, "The capital of France is Paris.");
fr_index_add_text(idx, 1, "Python is a programming language.");
fr_index_add_text(idx, 2, "Paris is known for the Eiffel Tower.");
// ... millions more

// Query
fr_results *results = fr_search_text(idx, "What is the capital of France?", 5);

// Use results to augment LLM context
for (int i = 0; i < results->count; i++) {
    printf("Doc %llu (score: %d): %s\n",
           results->items[i].doc_id,
           results->items[i].score,
           get_document_text(results->items[i].doc_id));
}

// Cleanup
fr_results_free(results);
fr_index_free(idx);
asic_close(asic);
```

### Tight Integration (Retro-style)

```c
// During transformer forward pass
tm_tensor* transformer_layer(tm_tensor *x, int layer_idx, fr_index *memory) {
    // Standard attention
    tm_tensor *attn = fa_self_attention(x, weights->qkv, weights->o, &config);

    // Retrieve relevant context based on hidden state
    char *query_text = extract_query_from_hidden(x);  // Implementation-specific
    fr_results *retrieved = fr_search_text(memory, query_text, 4);

    // Cross-attention with retrieved context
    tm_tensor *context = embed_retrieved_docs(retrieved);
    tm_tensor *cross_attn = fa_cross_attention(attn, context, weights->cross, &config);

    // FFN
    tm_tensor *out = ffn_forward(cross_attn, weights->ffn);

    fr_results_free(retrieved);
    return out;
}
```

## Performance Targets

| Metric | CPU Baseline | With ASIC |
|--------|--------------|-----------|
| Index 1M docs | ~60 sec | ~2 sec |
| Query latency | ~5 ms | ~0.5 ms |
| Throughput | ~10K qps | ~100K qps |
| Power | ~50W | ~5W |

## File Format

### Index Format (`.fri`)

```
[Header: 16 bytes]
  magic: "FRI1" (4 bytes)
  version: uint16
  num_docs: uint64
  config_flags: uint16

[Config: variable]
  num_bands: uint8
  band_size: uint8
  signature_bits: uint16

[Signatures: num_docs * 32 bytes]
  For each doc:
    signature: uint8[32]  // 256 bits

[LSH Tables: num_bands * variable]
  For each band:
    num_buckets: uint32
    For each bucket:
      bucket_hash: uint32
      num_docs: uint32
      doc_ids: uint64[num_docs]
```

## Dependencies

- `../tensor/` - For embedding integration (optional)
- `../tokenizer/` - For text tokenization
- An external ASIC device abstraction (not in this tree)
