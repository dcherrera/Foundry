/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_HDC_H
#define FOUNDRY_HDC_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Hyperdimensional Computing (HDC) operations.
 * General-purpose library for binary hypervector manipulation.
 *
 * HDC encodes information as high-dimensional binary vectors and performs
 * computation via simple bitwise operations (XOR for binding, majority vote
 * for bundling, cyclic shift for permutation).
 *
 * Uses random projection (sign of Gaussian matrix-vector product) to encode
 * dense float vectors into binary hypervectors while preserving similarity
 * relationships (Johnson-Lindenstrauss lemma).
 */

/* Forward declarations */
typedef struct tm_tensor tm_tensor;

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* Hypervector: bit-packed binary vector */
typedef struct {
    uint64_t *bits;     /* Bit-packed storage, ceil(dim/64) elements */
    int dim;            /* Dimensionality (e.g. 2048, 4096, 10000) */
} fv_vector;

/* Random projection matrix for float -> binary encoding (opaque) */
typedef struct fv_projector fv_projector;

/* Configuration */
typedef struct {
    int dim;            /* Hypervector dimension (default: 4096) */
    uint64_t seed;      /* RNG seed for deterministic projection */
} fv_config;

/* ============================================================= */
/* Config                                                        */
/* ============================================================= */

/* Get default config (dim=4096, seed=42) */
fv_config fv_config_default(void);

/* ============================================================= */
/* Vector Lifecycle                                              */
/* ============================================================= */

/* Allocate zero-initialized vector of given dimension */
fv_vector* fv_alloc(int dim);

/* Create deterministic random vector from seed */
fv_vector* fv_random(int dim, uint64_t seed);

/* Create zero vector */
fv_vector* fv_zeros(int dim);

/* Deep copy */
fv_vector* fv_clone(const fv_vector *v);

/* Free vector */
void fv_free(fv_vector *v);

/* ============================================================= */
/* Core HDC Operations                                           */
/* ============================================================= */

/* Bind: XOR of two vectors (associative, commutative, self-inverse) */
fv_vector* fv_xor(const fv_vector *a, const fv_vector *b);

/* Permute: cyclic bit shift (positive = left shift, negative = right) */
fv_vector* fv_permute(const fv_vector *v, int shift);

/* Bundle: majority vote across vectors (threshold at n/2) */
fv_vector* fv_bundle(fv_vector **vectors, int count);

/* Weighted bundle: weighted majority vote */
fv_vector* fv_bundle_weighted(fv_vector **vectors, float *weights, int count);

/* Hamming distance between two vectors */
int fv_hamming(const fv_vector *a, const fv_vector *b);

/* Cosine-like similarity: 1.0 - hamming/dim (range [0, 1]) */
float fv_similarity(const fv_vector *a, const fv_vector *b);

/* ============================================================= */
/* Bit Operations                                                */
/* ============================================================= */

/* Get bit at index (0 or 1) */
int fv_get_bit(const fv_vector *v, int idx);

/* Set bit at index (val: 0 or 1) */
void fv_set_bit(fv_vector *v, int idx, int val);

/* Count number of set bits (population count) */
int fv_popcount(const fv_vector *v);

/* ============================================================= */
/* Random Projection (float -> binary)                           */
/* ============================================================= */

/* Create projector: maps input_dim floats to hdc_dim binary vector */
fv_projector* fv_projector_new(int input_dim, int hdc_dim, uint64_t seed);

/* Free projector */
void fv_projector_free(fv_projector *proj);

/* Project float embedding to binary hypervector: sign(R @ x) */
fv_vector* fv_project(const fv_projector *proj, const tm_tensor *embedding);

/* Incremental projection: accumulate R @ x into float buffer (no thresholding).
 * accum must be zeroed before first call and have size >= hdc_dim floats.
 * Multiple calls accumulate: accum += R @ x_i.
 * Finalize with fv_project_threshold(). */
void fv_project_accum(const fv_projector *proj, const float *x, float *accum);

/* Threshold accumulated projection to binary: sign(accum).
 * Returns a new fv_vector. Caller must free with fv_free(). */
fv_vector* fv_project_threshold(const fv_projector *proj, const float *accum);

/* Get HDC dimension of projector */
int fv_projector_hdc_dim(const fv_projector *proj);

/* ============================================================= */
/* Tensor Conversion                                             */
/* ============================================================= */

/* Binary HV -> float tensor [dim] with values 0.0/1.0 */
tm_tensor* fv_to_tensor(const fv_vector *v);

/* Float tensor -> binary HV (threshold at 0: positive -> 1, else -> 0) */
fv_vector* fv_from_tensor(const tm_tensor *t);

/* ============================================================= */
/* Serialization                                                 */
/* ============================================================= */

/* Save vector to binary file (8 bytes dim + bit data) */
int fv_save(const fv_vector *v, const char *path);

/* Load vector from binary file */
fv_vector* fv_load(const char *path);

/* ============================================================= */
/* Batch Search                                                  */
/* ============================================================= */

/* Find top-k most similar vectors in corpus.
 * Returns number of results found (min of top_k, corpus_size).
 * indices_out and scores_out must be pre-allocated with top_k elements. */
int fv_search(fv_vector **corpus, int corpus_size,
              const fv_vector *query, int top_k,
              int *indices_out, float *scores_out);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_HDC_H */
