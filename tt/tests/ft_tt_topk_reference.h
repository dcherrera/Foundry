/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_topk_reference.h
 * @brief The CPU reference the on-card top-k and sampler are gated against: an iterative
 *   argmax over an untilized bf16 logit row, and Pantheon's sparse-survivor sampler.
 *
 * ---------------------------------------------------------------------------
 * WHY PCC IS THE WRONG GATE FOR THIS OP, ENTIRELY
 * ---------------------------------------------------------------------------
 * Every other op in this backend is accepted at PCC ≥ 0.99 against a CPU fp32 reference.
 * **That gate cannot see this op's failure mode.** Top-k returns INDICES; a result whose
 * k values are all correct and whose k-th index is a tie resolved the other way correlates
 * at 1.0 with the truth and is a wrong answer — the model emits a different token. So the
 * acceptance here is EXACT INDEX AGREEMENT, pair for pair, and the sampled token id
 * compared as an integer.
 *
 * The same reasoning one level down: bf16 has EIGHT mantissa bits and MiniCPM5-1B's
 * vocabulary is 130,560, so a logit row has on average two entries per representable value
 * before the distribution is even considered. **Ties are the main event.** The inputs the
 * suite builds therefore plant exact ties and adjacent-representable near-ties on purpose,
 * at and around the selection boundary, because that is where a reference and an
 * implementation disagree silently.
 *
 * ---------------------------------------------------------------------------
 * INDEPENDENCE — WHAT THIS FILE DELIBERATELY DOES NOT SHARE WITH THE KERNEL
 * ---------------------------------------------------------------------------
 * The selection here is the plain O(N·V) iterative argmax on **fp32 values**, with the
 * comparator written as a float comparison — NOT the kernel's order-preserving integer key
 * (`ftk_key_from_bf16`) and NOT its accept-interval pruning. A reference that shared the
 * kernel's representation could not disagree with it, and a check that cannot disagree is
 * not a check (dev notes §3, "any metric computed on a round trip through your own code is
 * measuring your code's self-consistency").
 *
 * The ONE thing that is shared by construction is `ft_tt_topk_reference_exp()`, because an
 * exact sampled-token gate needs both sides to compute the same exponential and the device
 * has no FPU and no libm. It is a transliteration, and `test_topk_ops.c` PARSES the
 * coefficients out of `kernels/resident_topk_protocol.h` and compares them against these —
 * a retyped constant is exactly the drift dev notes §4 catalogues. `expf` is kept beside it
 * so the approximation itself has a measured error rather than an assumed one.
 *
 * @status Off-hardware only; no device access anywhere in this file. Drives both the
 *   off-hardware gate and the on-card acceptance.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_TOPK_REFERENCE_H
#define FT_TT_TOPK_REFERENCE_H

#include <stddef.h>
#include <stdint.h>

#include "../ft_tt_op_sample.h"

/**
 * Exact top-`select_count` of an untilized bf16 logit row.
 *
 * Iterative argmax, `select_count` rounds, each taking the global maximum of what is left
 * with the comparator `(v > best) || (v == best && index < best_index)` — value-for-value
 * and tie-for-tie what Pantheon's `fc_topk_logits` reproduces from the host's serial scan.
 * Output is DESCENDING by value, ties by ascending index, which is the order the card
 * writes and the order `fc_topk_logits` returns.
 *
 * @param row           `vocab` bf16 patterns, untilized, in vocabulary index order
 * @param vocab         its length
 * @param select_count  pairs to produce; <= FT_TT_OP_SAMPLE_MAX_SELECT and <= vocab
 * @param out_indices   receives `select_count` vocabulary indices
 * @param out_values    receives `select_count` fp32 values (exact widenings of the bf16)
 * @return              0 on success, -1 on a bad argument
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_topk_reference_select(const uint16_t *row, uint32_t vocab, uint32_t select_count,
                                uint32_t *out_indices, float *out_values);

/** The order-preserving integer key, for the exhaustive 65,536-pattern order sweep. This
 *  IS the kernel's map, transliterated, and the sweep is what proves it agrees with float
 *  order everywhere except the documented ±0 case. */
uint32_t ft_tt_topk_reference_key(uint16_t bf16_bits);

/** The kernel's software exp, transliterated. Only x <= 0 is defined. */
float ft_tt_topk_reference_exp(float x);

/** xorshift32, and the [0,1) draw built on it. Both must match the kernel exactly or the
 *  sampled-token gate is statistical rather than exact. */
uint32_t ft_tt_topk_reference_rng_next(uint32_t *state);
float ft_tt_topk_reference_rng_unit(uint32_t *state);

/**
 * The sampler, transliterated from Pantheon `src/core/engine.c` `sparse_sample()` and
 * `sparse_argmax()` — the path the K80 ALREADY takes whenever `fc_topk_logits` reduced the
 * row, so this is the deployed semantics and not a new approximation.
 *
 * THE ORDER IS THE CONTRACT AND IT IS ASSERTED SEPARATELY:
 *   temperature <= 0  ->  argmax on the RAW values, NO repetition penalty, lowest id wins
 *   temperature  > 0  ->  penalty -> temperature -> softmax -> top_k -> top_p ->
 *                         renormalize -> cumulative draw
 * Applying the penalty before the greedy argmax is a real bug that shipped once and lost an
 * answer on a near tie; the two paths are separate functions here for the same reason they
 * are separate in the kernel.
 *
 * @param indices       survivors, DESCENDING by value as the card writes them
 * @param values        their fp32 values
 * @param count         how many
 * @param params        the knobs; `rng_seed` is ignored, `rng_state` carries the stream
 * @param recent        the repetition window, OLDEST FIRST; may be NULL
 * @param recent_count  its length
 * @param rng_state     in/out; advanced once per stochastic draw, untouched when greedy
 * @param use_libm_exp  non-zero uses `expf` instead of the transliterated polynomial — the
 *                      arm that measures the approximation rather than assuming it
 * @return              the sampled vocabulary index
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_topk_reference_sample(const uint32_t *indices, const float *values,
                                     uint32_t count,
                                     const ft_tt_op_sample_params *params,
                                     const uint32_t *recent, uint32_t recent_count,
                                     uint32_t *rng_state, int use_libm_exp);

/**
 * Tilize a bf16 logit row into the face-ordered [32 x vocab] tile strip the card reads,
 * writing `row` with the logits and every other row with `filler`.
 *
 * The filler is not decoration. The LM head runs at Mt = 1, so 32 token rows exist and one
 * is being selected over; filling the other 31 with values that would WIN if the kernel
 * read the wrong row is what turns a row-indexing bug into a red test instead of a
 * plausible answer.
 *
 * @param row_values   `vocab` bf16 patterns
 * @param vocab        must be a multiple of 32
 * @param row          0..31, the token row
 * @param filler       bf16 pattern written to the other 31 rows
 * @param out          receives `vocab / 32 * 2048` bytes
 * @param out_bytes    its size
 * @return             0 on success, -1 on a bad argument
 */
int ft_tt_topk_reference_tilize_row(const uint16_t *row_values, uint32_t vocab, uint32_t row,
                                    uint16_t filler, void *out, size_t out_bytes);

#endif /* FT_TT_TOPK_REFERENCE_H */
