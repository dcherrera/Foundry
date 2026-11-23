/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_RETRIEVAL_H
#define FOUNDRY_RETRIEVAL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Hash-based document retrieval with ASIC acceleration.
 * Replaces dense embedding retrieval for RAG pipelines.
 *
 * Uses SimHash/MinHash/LSH for approximate similarity search.
 * Optional ASIC backend accelerates hash computation 100-1000x.
 */

/* ============================================================= */
/* Forward Declarations                                          */
/* ============================================================= */

/* ASIC device handle (opaque; the ASIC backend is not in this tree) */
typedef struct asic_device asic_device;

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* 256-bit hash signature (SimHash or MinHash output) */
typedef struct {
    uint64_t bits[4];  /* 256 bits = 4 x 64-bit words */
} fr_signature;

/* Hash algorithm selection */
typedef enum {
    FR_HASH_SIMHASH,   /* SimHash: weighted feature hashing */
    FR_HASH_MINHASH,   /* MinHash: Jaccard similarity */
} fr_hash_type;

/* Backend selection */
typedef enum {
    FR_BACKEND_CPU,    /* Software SHA-256 (fallback) */
    FR_BACKEND_ASIC,   /* Hardware ASIC acceleration */
} fr_backend_type;

/* Retrieval configuration */
typedef struct {
    fr_hash_type hash_type;       /* SimHash or MinHash */
    int num_bands;                /* LSH bands (default: 8) */
    int band_size;                /* Bits per band (default: 32) */
    int num_permutations;         /* MinHash permutations (default: 128) */
    bool normalize_text;          /* Lowercase, strip punctuation */
    int ngram_size;               /* N-gram size for features (default: 3) */
} fr_config;

/* Single retrieval result */
typedef struct {
    uint64_t doc_id;     /* Document identifier */
    int score;           /* Similarity score (256 - hamming distance) */
    fr_signature sig;    /* Document signature (for re-ranking) */
} fr_result;

/* List of retrieval results */
typedef struct {
    fr_result *items;    /* Array of results */
    size_t count;        /* Number of results */
    size_t capacity;     /* Allocated capacity */
} fr_results;

/* LSH index (opaque) */
typedef struct fr_index fr_index;

/* Batch operation context (for async ASIC ops) */
typedef struct fr_batch fr_batch;

/* ============================================================= */
/* Configuration                                                 */
/* ============================================================= */

/* Default configuration */
fr_config fr_config_default(void);

/* Configuration for specific use cases */
fr_config fr_config_text(void);      /* Optimized for text documents */
fr_config fr_config_code(void);      /* Optimized for source code */
fr_config fr_config_fast(void);      /* Faster, less accurate */
fr_config fr_config_accurate(void);  /* Slower, more accurate */

/* ============================================================= */
/* Signatures                                                    */
/* ============================================================= */

/* Compute SimHash signature from text
 * Returns NULL on error
 */
fr_signature* fr_simhash(const char *text, fr_config *config);

/* Compute SimHash with explicit features and weights
 * features: array of feature strings
 * weights: array of feature weights (NULL = all 1.0)
 * n: number of features
 */
fr_signature* fr_simhash_weighted(const char **features, const float *weights,
                                   size_t n, fr_config *config);

/* Compute MinHash signature from token set
 * tokens: array of tokens (treated as set)
 * n: number of tokens
 */
fr_signature* fr_minhash(const char **tokens, size_t n, fr_config *config);

/* Compute Hamming distance between signatures
 * Returns number of differing bits (0-256)
 */
int fr_hamming(const fr_signature *a, const fr_signature *b);

/* Compute similarity score (256 - hamming distance)
 * Returns 0-256, higher is more similar
 */
int fr_similarity(const fr_signature *a, const fr_signature *b);

/* Free signature */
void fr_signature_free(fr_signature *sig);

/* ============================================================= */
/* ASIC-Accelerated Signatures                                   */
/* ============================================================= */

/* Compute SimHash using ASIC backend
 * Falls back to CPU if dev is NULL
 */
fr_signature* fr_simhash_asic(asic_device *dev, const char *text,
                               fr_config *config);

/* Batch compute signatures (saturates ASIC throughput)
 * texts: array of text strings
 * n: number of texts
 * sigs: output array (must be pre-allocated, n elements)
 * Returns 0 on success, -1 on error
 */
int fr_simhash_batch(asic_device *dev, const char **texts, size_t n,
                     fr_signature **sigs, fr_config *config);

/* Async batch operations */
fr_batch* fr_batch_start(asic_device *dev, size_t capacity);
int fr_batch_add(fr_batch *batch, const char *text);
int fr_batch_execute(fr_batch *batch);  /* Blocks until complete */
fr_signature* fr_batch_get(fr_batch *batch, size_t idx);
void fr_batch_free(fr_batch *batch);

/* ============================================================= */
/* Index Management                                              */
/* ============================================================= */

/* Create new LSH index */
fr_index* fr_index_new(fr_config *config);

/* Free index and all signatures */
void fr_index_free(fr_index *idx);

/* Set ASIC backend for index operations
 * Pass NULL to use CPU fallback
 */
void fr_index_set_backend(fr_index *idx, asic_device *dev);

/* Get current backend type */
fr_backend_type fr_index_get_backend(fr_index *idx);

/* Get number of indexed documents */
size_t fr_index_size(fr_index *idx);

/* ============================================================= */
/* Indexing                                                      */
/* ============================================================= */

/* Add document with pre-computed signature
 * Takes ownership of sig
 * Returns 0 on success, -1 on error
 */
int fr_index_add(fr_index *idx, uint64_t doc_id, fr_signature *sig);

/* Add document from text (computes signature)
 * Returns 0 on success, -1 on error
 */
int fr_index_add_text(fr_index *idx, uint64_t doc_id, const char *text);

/* Batch add documents (efficient for ASIC backend)
 * Returns number of documents added, -1 on error
 */
int fr_index_add_batch(fr_index *idx, const uint64_t *doc_ids,
                       const char **texts, size_t n);

/* Remove document from index
 * Returns 0 on success, -1 if not found
 */
int fr_index_remove(fr_index *idx, uint64_t doc_id);

/* Rebuild LSH tables after bulk operations
 * Call after adding many documents for better performance
 */
int fr_index_rebuild(fr_index *idx);

/* ============================================================= */
/* Retrieval                                                     */
/* ============================================================= */

/* Search with pre-computed signature
 * Returns top_k most similar documents
 * Caller owns returned results (use fr_results_free)
 */
fr_results* fr_search(fr_index *idx, const fr_signature *query, int top_k);

/* Search with text query (computes signature)
 * Returns top_k most similar documents
 */
fr_results* fr_search_text(fr_index *idx, const char *query, int top_k);

/* Search with minimum similarity threshold
 * Returns all documents with similarity >= min_score
 */
fr_results* fr_search_threshold(fr_index *idx, const fr_signature *query,
                                 int min_score);

/* Batch search (multiple queries)
 * queries: array of signatures
 * n: number of queries
 * top_k: results per query
 * Returns array of n result sets
 */
fr_results** fr_search_batch(fr_index *idx, const fr_signature **queries,
                              size_t n, int top_k);

/* Free results */
void fr_results_free(fr_results *results);

/* Free batch results */
void fr_results_batch_free(fr_results **results, size_t n);

/* ============================================================= */
/* Persistence                                                   */
/* ============================================================= */

/* Save index to file
 * Format: .fri (Foundry Retrieval Index)
 * Returns 0 on success, -1 on error
 */
int fr_index_save(fr_index *idx, const char *path);

/* Load index from file
 * Returns NULL on error
 */
fr_index* fr_index_load(const char *path);

/* Memory-map index for read-only access (large indices)
 * Returns NULL on error
 */
fr_index* fr_index_mmap(const char *path);

/* ============================================================= */
/* Utilities                                                     */
/* ============================================================= */

/* Print signature as hex string */
void fr_signature_print(const fr_signature *sig);

/* Print index statistics */
void fr_index_stats(fr_index *idx);

/* Estimate memory usage in bytes */
size_t fr_index_memory(fr_index *idx);

/* Benchmark hash throughput
 * Returns hashes per second
 */
double fr_benchmark_hash(asic_device *dev, int num_hashes);

/* ============================================================= */
/* Integration with Foundry Tokenizer                            */
/* ============================================================= */

/* Forward declaration */
typedef struct ft_tokenizer ft_tokenizer;

/* Compute signature using tokenizer for feature extraction
 * Useful for consistent tokenization with LLM
 */
fr_signature* fr_simhash_tokenized(ft_tokenizer *tok, const char *text,
                                    fr_config *config);

/* ============================================================= */
/* Integration with Foundry Tensor (optional)                    */
/* ============================================================= */

/* Forward declaration */
typedef struct tm_tensor tm_tensor;

/* Convert signature to dense tensor (for hybrid approaches)
 * Returns (256,) float tensor with -1.0/+1.0 values
 */
tm_tensor* fr_signature_to_tensor(const fr_signature *sig);

/* Convert dense embedding to signature (quantization)
 * Expects normalized embedding tensor
 */
fr_signature* fr_tensor_to_signature(const tm_tensor *embedding);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_RETRIEVAL_H */
