/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_attention_reference.h
 * @brief The CPU reference flash-decode attention is scored against, and a BLOCKED model
 *   of what the kernel actually does — two different things, on purpose.
 *
 * ---------------------------------------------------------------------------
 * WHY THERE ARE TWO
 * ---------------------------------------------------------------------------
 * `ft_tt_attention_reference()` is attention from its DEFINITION: scores, one softmax over
 * the whole live row, one weighted sum. It knows nothing about blocks, tiles, masks or
 * running maxima, which is what makes it an independent answer rather than a second copy of
 * the kernel.
 *
 * `ft_tt_attention_online_model()` is the ALGORITHM the kernel runs: the same KV blocking,
 * the same padded window, the same additive mask, the same running max / row sum / output
 * rescale. It exists so the online softmax can be proven correct — and proven WRONG when it
 * is broken — without spending a card run, and because the vector family measured that an
 * off-hardware tile-space model predicts this silicon to about six decimals.
 *
 * The two must agree to fp32 rounding on every shape. Where they disagree, the model is the
 * one describing the kernel, so a disagreement is a bug in the ALGORITHM rather than in the
 * kernel's execution of it — a distinction that is expensive to make on the card.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE REFERENCE CONSUMES
 * ---------------------------------------------------------------------------
 * bf16-ROUNDED inputs, because the card sees bf16 and a reference fed fp32 measures a
 * quantization the kernel is not responsible for (dev notes §3). It also applies the query
 * scale the way the kernel does — multiply, then round — so the comparison is about
 * attention and not about where the 1/sqrt(d) landed.
 *
 * @status Used by tests/test_attention_ops.c (off-hardware) and
 *   tests/test_phase6_attention.c (on-card).
 * @issues None
 * @todo None
 */

#ifndef FT_TT_ATTENTION_REFERENCE_H
#define FT_TT_ATTENTION_REFERENCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** One reference problem. Every pointer is fp32 already rounded through bf16. */
typedef struct ft_tt_attention_problem {
    uint32_t q_rows;       /**< Q heads in the group */
    uint32_t head_dim;     /**< elements, a multiple of 32 */
    uint32_t kv_positions; /**< LIVE positions */
    const float *query;    /**< q_rows * head_dim, row-major */
    const float *keys;     /**< kv_capacity * head_dim, row-major */
    const float *values;   /**< kv_capacity * head_dim, row-major */
} ft_tt_attention_problem;

/**
 * Attention from its definition: out = softmax(Q Kᵀ / sqrt(d)) V over the live positions.
 *
 * @param out  q_rows * head_dim, row-major
 * @return     true on success, false for a shape it cannot serve
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool ft_tt_attention_reference(const ft_tt_attention_problem *problem, float *out);

/**
 * CHUNK attention from its definition: `q_rows` CONSECUTIVE POSITIONS of one head, row r
 * being position `chunk_first + r`, each attending to positions `0 .. chunk_first + r`.
 *
 * `problem->kv_positions` must be `chunk_first + q_rows` — the chunk ends the live range.
 * Implemented as `q_rows` calls to ft_tt_attention_reference() at increasing live lengths,
 * so causality is a LIVE LENGTH here and a MASK TILE on the card: two spellings of the same
 * sentence, reached differently, which is what makes the comparison mean something.
 *
 * @param out  q_rows * head_dim, row-major
 * @return     true on success
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool ft_tt_attention_chunk_reference(const ft_tt_attention_problem *problem,
                                     uint32_t chunk_first, float *out);

/**
 * The kernel's algorithm: the same blocking, padding, masking and online rescale.
 *
 * `keys` and `values` must be readable for the PADDED window — `block_tiles * 2` rounded up
 * — because that is what the kernel reads, and feeding this model only the live rows would
 * hide exactly the bug the mask exists to prevent.
 *
 * @param block_tiles  position tiles per KV block
 * @param out          q_rows * head_dim, row-major
 * @return             true on success
 */
bool ft_tt_attention_online_model(const ft_tt_attention_problem *problem, uint32_t block_tiles,
                                  float *out);

/** Pearson correlation between two arrays. */
double ft_tt_attention_pcc(const float *expected, const float *actual, size_t count);

/** Largest absolute difference. */
double ft_tt_attention_max_abs(const float *expected, const float *actual, size_t count);

/** Non-finite elements in `values`. Counted per SIDE, because `PCC -nan` is a refusal to
 *  answer and does not say which side failed (dev notes §3). */
size_t ft_tt_attention_non_finite(const float *values, size_t count);

/** Deterministic full-mantissa random in [-1, 1). Seeded per call site so a failure
 *  reproduces. */
void ft_tt_attention_seed(uint64_t seed);
float ft_tt_attention_random(void);

#endif /* FT_TT_ATTENTION_REFERENCE_H */
