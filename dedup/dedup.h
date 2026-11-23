/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_DEDUP_H
#define FOUNDRY_DEDUP_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Training Data Deduplication
 *
 * ASIC-accelerated MinHash/LSH for deduplicating LLM training data.
 * Designed for Common Crawl scale (billions of documents).
 *
 * The bottleneck in deduplication is computing MinHash signatures,
 * which requires millions of hash operations per document. ASIC
 * miners can compute these 100x faster than CPUs.
 */

/* ============================================================= */
/* Forward Declarations                                          */
/* ============================================================= */

/* ASIC device handle (opaque; the ASIC backend is not in this tree) */
typedef struct asic_device asic_device;

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* Single shingle (n-gram) */
typedef struct {
    uint64_t hash;      /* Precomputed hash of the n-gram */
} fd_shingle;

/* MinHash signature */
typedef struct {
    uint32_t *values;   /* Array of minimum hash values */
    size_t size;        /* Number of permutations (dimensions) */
} fd_signature;

/* Candidate pair for verification */
typedef struct {
    uint64_t doc_a;
    uint64_t doc_b;
    float estimated_jaccard;  /* Estimated from signature overlap */
} fd_candidate;

/* List of candidate pairs */
typedef struct {
    fd_candidate *pairs;
    size_t count;
    size_t capacity;
} fd_candidates;

/* Output mode */
typedef enum {
    FD_OUTPUT_ALL,          /* Output all docs with duplicate annotations */
    FD_OUTPUT_UNIQUE_ONLY,  /* Only output unique documents */
    FD_OUTPUT_DUPLICATES,   /* Only output duplicate pairs */
} fd_output_mode;

/* Configuration */
typedef struct {
    /* Shingling */
    int shingle_size;           /* N-gram size (default: 13) */
    bool char_shingles;         /* Character n-grams (true) or word n-grams (false) */

    /* MinHash */
    int num_permutations;       /* Signature dimensions (default: 128) */
    uint64_t seed;              /* Random seed for permutations */

    /* LSH */
    int num_bands;              /* LSH bands (default: 8) */
    float similarity_threshold; /* Jaccard threshold (default: 0.8) */

    /* Input */
    const char *text_field;     /* JSON field containing text (default: "text") */
    const char *id_field;       /* JSON field for document ID (default: "id") */

    /* Output */
    fd_output_mode output_mode;

    /* Streaming/batching */
    bool streaming;             /* Stream large files instead of loading */
    size_t batch_size;          /* Documents per batch (default: 10000) */
    const char *index_path;     /* Path for memory-mapped index (streaming mode) */

    /* Performance */
    int num_threads;            /* Worker threads (default: auto) */
} fd_config;

/* Deduplication results */
typedef struct {
    size_t total_docs;          /* Total documents processed */
    size_t unique_docs;         /* Unique documents */
    size_t duplicates;          /* Duplicate documents found */
    size_t duplicate_pairs;     /* Duplicate pairs (for cluster analysis) */
    size_t hashes_computed;     /* Total hash operations */
    double elapsed_sec;         /* Total time */
    double docs_per_sec;        /* Throughput */
    double hashes_per_sec;      /* Hash throughput */
} fd_result;

/* LSH index (opaque) */
typedef struct fd_index fd_index;

/* Dedup pipeline (opaque) */
typedef struct fd_pipeline fd_pipeline;

/* Progress callback */
typedef void (*fd_progress_fn)(size_t docs_processed, size_t total_docs,
                                size_t duplicates_found, void *user_data);

/* ============================================================= */
/* Configuration                                                 */
/* ============================================================= */

/**
 * Get default configuration.
 * Good defaults for general text deduplication.
 */
fd_config fd_config_default(void);

/**
 * Configuration optimized for exact duplicate detection.
 * Faster but only catches identical documents.
 */
fd_config fd_config_exact(void);

/**
 * Configuration optimized for aggressive deduplication.
 * Lower threshold, catches more near-duplicates.
 */
fd_config fd_config_aggressive(void);

/* ============================================================= */
/* Shingling                                                     */
/* ============================================================= */

/**
 * Extract shingles (n-grams) from text.
 *
 * @param text Input text
 * @param n Shingle size (characters or words based on config)
 * @param count Output: number of shingles
 * @return Array of shingles (caller must free with fd_shingles_free)
 */
fd_shingle* fd_shingle_text(const char *text, int n, size_t *count);

/**
 * Extract word-level shingles.
 */
fd_shingle* fd_shingle_words(const char *text, int n, size_t *count);

/**
 * Free shingle array.
 */
void fd_shingles_free(fd_shingle *shingles);

/* ============================================================= */
/* MinHash Signatures                                            */
/* ============================================================= */

/**
 * Compute MinHash signature from shingles (CPU).
 *
 * @param shingles Array of shingles
 * @param n Number of shingles
 * @param config Configuration
 * @return Signature (caller must free with fd_signature_free)
 */
fd_signature* fd_minhash(const fd_shingle *shingles, size_t n, fd_config *config);

/**
 * Compute MinHash signature using ASIC backend.
 * Falls back to CPU if dev is NULL.
 *
 * @param dev ASIC device (or NULL for CPU)
 * @param shingles Array of shingles
 * @param n Number of shingles
 * @param config Configuration
 * @return Signature
 */
fd_signature* fd_minhash_asic(asic_device *dev, const fd_shingle *shingles,
                              size_t n, fd_config *config);

/**
 * Compute MinHash signatures for multiple documents (batch).
 * More efficient for ASIC backend.
 *
 * @param dev ASIC device (or NULL for CPU)
 * @param shingle_arrays Array of shingle arrays
 * @param counts Array of shingle counts
 * @param num_docs Number of documents
 * @param config Configuration
 * @param signatures Output: array of signatures (pre-allocated)
 * @return 0 on success, -1 on error
 */
int fd_minhash_batch(asic_device *dev, const fd_shingle **shingle_arrays,
                     const size_t *counts, size_t num_docs,
                     fd_config *config, fd_signature **signatures);

/**
 * Estimate Jaccard similarity from signatures.
 *
 * @param a First signature
 * @param b Second signature
 * @return Estimated Jaccard similarity (0.0 - 1.0)
 */
float fd_jaccard_estimate(const fd_signature *a, const fd_signature *b);

/**
 * Free signature.
 */
void fd_signature_free(fd_signature *sig);

/* ============================================================= */
/* LSH Index                                                     */
/* ============================================================= */

/**
 * Create new LSH index.
 */
fd_index* fd_index_new(fd_config *config);

/**
 * Free index.
 */
void fd_index_free(fd_index *idx);

/**
 * Add document signature to index.
 *
 * @param idx Index
 * @param doc_id Document identifier
 * @param sig Signature (index takes ownership)
 * @return 0 on success, -1 on error
 */
int fd_index_add(fd_index *idx, uint64_t doc_id, fd_signature *sig);

/**
 * Query for candidate duplicates.
 *
 * @param idx Index
 * @param sig Query signature
 * @return Candidates (caller must free with fd_candidates_free)
 */
fd_candidates* fd_index_query(fd_index *idx, const fd_signature *sig);

/**
 * Get index size (number of documents).
 */
size_t fd_index_size(fd_index *idx);

/**
 * Save index to file.
 */
int fd_index_save(fd_index *idx, const char *path);

/**
 * Load index from file.
 */
fd_index* fd_index_load(const char *path);

/**
 * Memory-map index for read-only access.
 */
fd_index* fd_index_mmap(const char *path);

/**
 * Free candidates.
 */
void fd_candidates_free(fd_candidates *candidates);

/* ============================================================= */
/* Pipeline                                                      */
/* ============================================================= */

/**
 * Create new dedup pipeline.
 */
fd_pipeline* fd_pipeline_new(fd_config *config);

/**
 * Free pipeline.
 */
void fd_pipeline_free(fd_pipeline *pipe);

/**
 * Set ASIC backend for pipeline.
 * Pass NULL to use CPU fallback.
 */
void fd_pipeline_set_backend(fd_pipeline *pipe, asic_device *dev);

/**
 * Add input file to pipeline.
 * Supports: .jsonl, .json, .txt (one doc per line)
 *
 * @return 0 on success, -1 on error
 */
int fd_pipeline_add_file(fd_pipeline *pipe, const char *path);

/**
 * Add input directory (all .jsonl files).
 */
int fd_pipeline_add_dir(fd_pipeline *pipe, const char *path);

/**
 * Set progress callback.
 */
void fd_pipeline_on_progress(fd_pipeline *pipe, fd_progress_fn callback,
                              void *user_data);

/**
 * Run deduplication pipeline.
 *
 * @param pipe Pipeline
 * @return Results (caller must free with fd_result_free)
 */
fd_result* fd_pipeline_run(fd_pipeline *pipe);

/**
 * Write deduplicated output.
 *
 * @param pipe Pipeline (must have been run)
 * @param path Output file path
 * @return 0 on success, -1 on error
 */
int fd_pipeline_write_output(fd_pipeline *pipe, const char *path);

/**
 * Get duplicate pairs (for analysis).
 */
fd_candidates* fd_pipeline_get_duplicates(fd_pipeline *pipe);

/**
 * Free result.
 */
void fd_result_free(fd_result *result);

/* ============================================================= */
/* Utilities                                                     */
/* ============================================================= */

/**
 * Compute exact Jaccard similarity (for verification).
 * Slower than estimate, but accurate.
 */
float fd_jaccard_exact(const fd_shingle *a, size_t na,
                       const fd_shingle *b, size_t nb);

/**
 * Print configuration.
 */
void fd_config_print(fd_config *config);

/**
 * Print result summary.
 */
void fd_result_print(fd_result *result);

/**
 * Benchmark hash throughput.
 *
 * @param dev ASIC device (or NULL for CPU)
 * @param num_hashes Number of hashes to compute
 * @return Hashes per second
 */
double fd_benchmark_hash(asic_device *dev, size_t num_hashes);

/**
 * Estimate resources needed for dataset.
 *
 * @param num_docs Estimated number of documents
 * @param avg_doc_size Average document size in bytes
 * @param config Configuration
 * @param memory_gb Output: estimated memory usage (GB)
 * @param hashes Output: estimated total hashes
 */
void fd_estimate_resources(size_t num_docs, size_t avg_doc_size,
                           fd_config *config,
                           double *memory_gb, size_t *hashes);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_DEDUP_H */
