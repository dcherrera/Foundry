/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_TURBOQUANT_H
#define FOUNDRY_TURBOQUANT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * TurboQuant KV cache compression (foundry/attention, prefix fa_tq_).
 *
 * Based on: "TurboQuant: Online Vector Quantization with Near-optimal
 * Distortion Rate" (Zandieh et al., Google Research / DeepMind, 2025,
 * arXiv:2504.19874). See attention/spec.md for the full design.
 *
 * Pipeline per K or V vector:
 *   1. Rotate by a randomized Hadamard transform  D * H * x / sqrt(d)
 *      (O(d log d), D = random +-1 diagonal from a seed, H = Walsh-Hadamard).
 *   2. Map rotated coords into [0,1] via per-head min/max and scalar-quantize
 *      each coord with precomputed Lloyd-Max levels for Beta(d/2, d/2).
 *   3. Pack the level indices into bytes (2/3/4 bits per coord).
 * Dequant reverses: unpack -> centroid lookup -> denormalize -> inverse rotate.
 *
 * This module is deliberately free of any tensor-library dependency: it
 * operates on raw float* / uint8_t* buffers so it can be called from the
 * foundry attention module as well as the hand-rolled attention loops in
 * RWKV-X and Pantheon. Depends only on <math.h>, <string.h>, <stdint.h>.
 */

/* ============================================================= */
/* Codebook                                                      */
/* ============================================================= */

/* Precomputed Lloyd-Max scalar quantizer for Beta(d/2, d/2) on [0,1].
 * boundaries[] holds (levels - 1) decision thresholds (ascending);
 * centroids[] holds (levels) reconstruction values. */
typedef struct {
    int   bits;              /* 2, 3, or 4 */
    int   levels;            /* 1 << bits (<= 16) */
    float boundaries[15];    /* levels - 1 decision boundaries */
    float centroids[16];     /* levels reconstruction centroids */
} fa_tq_codebook;

/* ============================================================= */
/* Config / State                                                */
/* ============================================================= */

typedef struct {
    int key_bits;            /* 2, 3, or 4 (default: 4) */
    int value_bits;          /* 2, 3, or 4 (default: 2) */
} fa_tq_config;

/* Opaque quantizer state: rotation sign vector + K/V codebooks. */
typedef struct fa_tq_state fa_tq_state;

/* Opaque compressed KV cache. */
typedef struct fa_tq_cache fa_tq_cache;

/* ============================================================= */
/* Lifecycle                                                     */
/* ============================================================= */

/* Build a quantizer for a given head_dim and bit config.  seed drives the
 * random Hadamard sign vector (deterministic / reproducible).  head_dim need
 * not be a power of two; it is rounded up internally for the transform.
 * Returns NULL on invalid arguments or allocation failure. */
fa_tq_state* fa_tq_init(int head_dim, fa_tq_config config, uint64_t seed);
void         fa_tq_free(fa_tq_state* state);

/* Default config: K4 / V2. */
fa_tq_config fa_tq_config_default(void);

/* Introspection helpers. */
int  fa_tq_head_dim(const fa_tq_state* state);   /* logical head_dim */
int  fa_tq_rot_dim(const fa_tq_state* state);    /* padded transform dim (pow2) */
const fa_tq_codebook* fa_tq_key_codebook(const fa_tq_state* state);
const fa_tq_codebook* fa_tq_value_codebook(const fa_tq_state* state);

/* Packed bytes per head for a key / value vector (Step 3 packing). */
size_t fa_tq_key_bytes_per_head(const fa_tq_state* state);
size_t fa_tq_value_bytes_per_head(const fa_tq_state* state);

/* ============================================================= */
/* Codebook generation (Lloyd-Max for Beta(d/2, d/2))            */
/* ============================================================= */

/* Fill cb with the Lloyd-Max optimal quantizer for Beta(d/2, d/2) on [0,1]
 * at the given bit width.  Uses a precomputed static table when (head_dim,
 * bits) matches a shipped codebook, otherwise runs Lloyd's algorithm with
 * numerical quadrature.  Returns 0 on success, -1 on bad arguments. */
int fa_tq_build_codebook(fa_tq_codebook* cb, int head_dim, int bits);

/* ============================================================= */
/* Rotation (randomized Hadamard)                                */
/* ============================================================= */

/* Forward rotation  y = D * H * x / sqrt(dpow)  in place.  buf has length
 * fa_tq_rot_dim(state); the caller zero-pads x beyond head_dim.  Inverse is
 * the same transform (H and D are self-inverse up to the 1/sqrt scaling). */
void fa_tq_rotate(const fa_tq_state* state, float* buf);
void fa_tq_rotate_inverse(const fa_tq_state* state, float* buf);

/* ============================================================= */
/* Low-level single-head quantize / dequantize                   */
/* ============================================================= */

/* Quantize one head vector (head_dim floats) into out_packed
 * (fa_tq_*_bytes_per_head bytes) plus a 2-float [min,max] range.  is_key
 * selects the key vs value codebook / bit width. */
void fa_tq_quantize_head(const fa_tq_state* state, const float* vec,
                         bool is_key, uint8_t* out_packed, float* out_range);

/* Reconstruct one head vector (head_dim floats) from packed bytes + range. */
void fa_tq_dequantize_head(const fa_tq_state* state, const uint8_t* packed,
                           const float* range, bool is_key, float* out_vec);

/* ============================================================= */
/* Cache                                                         */
/* ============================================================= */

fa_tq_cache* fa_tq_cache_new(fa_tq_state* state, int num_layers,
                             int num_heads, int max_seq_len);
void         fa_tq_cache_free(fa_tq_cache* cache);
void         fa_tq_cache_clear(fa_tq_cache* cache);
int          fa_tq_cache_seq_len(const fa_tq_cache* cache);

/* Total compressed bytes held by the cache (all layers, both K and V,
 * including range metadata). */
size_t       fa_tq_cache_bytes(const fa_tq_cache* cache);

/* Quantize a full multi-head vector [num_heads * head_dim] into the cache
 * at (layer, pos).  is_key selects K vs V storage. */
void fa_tq_quantize(fa_tq_state* state, fa_tq_cache* cache,
                    int layer, int pos, const float* vec, bool is_key);

/* Dequantize count consecutive positions starting at start into out
 * [count * num_heads * head_dim].  Returns out. */
float* fa_tq_dequantize(fa_tq_state* state, fa_tq_cache* cache,
                        int layer, int start, int count,
                        bool is_key, float* out);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_TURBOQUANT_H */
