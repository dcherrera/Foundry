/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_vector_reference.h
 * @brief CPU fp32 references for the vector op family, a tile-space MODEL of what the
 *   kernels do to them, and the two scoring metrics — shared by the off-hardware suite
 *   (test_vector_ops.c) and the on-card acceptance (test_phase6_vector.c).
 *
 * ---------------------------------------------------------------------------
 * WHY THERE ARE TWO LEVELS AND NOT ONE
 * ---------------------------------------------------------------------------
 * dev notes §3: "any metric computed on a round trip through your own code is measuring
 * your code's self-consistency, not its correctness." So the REFERENCE is a plain
 * nested-loop fp32 computation written from each op's definition on UNTILIZED data, and it
 * is what the card is scored against.
 *
 * The MODEL is a second, separate thing and it exists because the card is not always
 * available. It walks the same tiles in the same order the kernel does and applies the same
 * rounding at the same places — bf16 into a circular buffer at every pack, fp32 inside the
 * destination register because `dst_accum_mode 1` — so it answers a question the reference
 * cannot: *does this arrangement of passes clear a PCC ≥ 0.99 gate at all, before anyone
 * spends a core finding out?*
 *
 * **What the model can and cannot prove.** It can prove the pass structure is right (a
 * softmax that reduces over one tile instead of Wt fails it), and it can predict the score.
 * It CANNOT prove the kernel implements the model — that is the hardware pass, and nothing
 * off-hardware substitutes for it. Read a green model run as "the plan is sound", never as
 * "the op works".
 *
 * ---------------------------------------------------------------------------
 * WHAT THE MODEL DELIBERATELY DOES NOT MODEL
 * ---------------------------------------------------------------------------
 * Face order. A per-tile elementwise map never touches it, and
 * a row reduce stays inside a tile row, so there is no face-space decision in either op for
 * a model to get wrong or right. The face-ordering risk in these ops lives entirely in
 * ft_tt_tilize, which has its own suite, and in the device's own packer.
 *
 * The SFPU's approximations. exp, sigmoid and reciprocal are hardware polynomial
 * evaluations; the model uses libm. So a model score is an UPPER BOUND on the card's, and
 * a card score slightly below it is expected rather than alarming.
 *
 * @status Green. See test_vector_ops.c; the model's PCC predictions matched the card's to
 *   ~6 decimals for all four reduce members as well as for the elementwise set.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_VECTOR_REFERENCE_H
#define FT_TT_VECTOR_REFERENCE_H

#include <stddef.h>
#include <stdint.h>

#include "../ft_tt_op_vector.h"

/* ================================================================
 * The fp32 references — each written from the op's definition
 * ================================================================ */

/**
 * y = f(a, b) elementwise, in fp32, over `count` values.
 *
 * `b` may be NULL for a unary function. Nothing here is tiled, blocked or rounded: this is
 * the definition of the op and nothing else.
 */
void ft_tt_vector_reference_eltwise(ft_tt_op_vector_function function, const float *a,
                                    const float *b, float *out, size_t count);

/**
 * Row-wise softmax of a [rows x cols] row-major matrix, in fp32, max-subtracted.
 *
 * The max subtraction is in the REFERENCE as well as in the kernel, and that is not
 * redundant: without it in both, the two would differ by the exact numerical effect the
 * kernel's extra pass exists to have, and the score would be measuring the shift rather
 * than the op.
 */
void ft_tt_vector_reference_softmax(const float *x, float *out, uint32_t rows, uint32_t cols);

/**
 * y = x * gamma * rsqrt(mean(x^2) + eps), row-wise. `gamma` is `cols` values.
 *
 * Written from the definition and accumulated in double, and it is the SAME computation as
 * test_ops.c's Phase 4 `reference_rms_norm()` — deliberately, because the merged kernel's
 * acceptance is that it matches what the Phase 4 one-shot scored on the same input, and two
 * references that differ would make that comparison meaningless.
 */
void ft_tt_vector_reference_rms_norm(const float *x, const float *gamma, float eps, float *out,
                                     uint32_t rows, uint32_t cols);

/** y = (x - mean) * gamma * rsqrt(mean((x - mean)^2) + eps) + beta, row-wise. The variance
 *  is the two-pass one — centre, then square — not E[x^2] - mean^2, which cancels
 *  catastrophically on a row whose mean is large beside its spread. */
void ft_tt_vector_reference_layer_norm(const float *x, const float *gamma, const float *beta,
                                       float eps, float *out, uint32_t rows, uint32_t cols);

/** y = x / sum(x), row-wise — the MoE router's renormalise over the selected gates. */
void ft_tt_vector_reference_router_normalise(const float *x, float *out, uint32_t rows,
                                             uint32_t cols);

/* ================================================================
 * The tile-space model of what the kernels do
 * ================================================================ */

/**
 * What the elementwise triad produces for `count` values, tile by tile, with bf16 at every
 * circular-buffer boundary.
 *
 * One tile at a time, one DST section per tile, so there is nothing to model about the
 * blocking — which is itself the finding: an elementwise op has no numerical structure the
 * tile decomposition can disturb. The rounding is real though: input is bf16 (it came off
 * DRAM), DST is fp32, the pack back to a CB is bf16. FN_SILU_MUL rounds TWICE, because
 * silu(a) goes through a scratch CB page before the multiply — which is the numerical price
 * of the fusion and is worth being able to see.
 *
 * @param count  must be a multiple of the 1024 elements in a tile
 */
void ft_tt_vector_model_eltwise(ft_tt_op_vector_function function, const float *a,
                                const float *b, float *out, size_t count);

/**
 * What the softmax triad produces for a [rows x cols] matrix, pass by pass.
 *
 * Models all four passes and the rounding between them:
 *   1. row max accumulated in fp32 across all `cols`/32 tiles, packed to bf16 (cb_max)
 *   2. exp(x - max) in fp32 from bf16 inputs, packed to bf16 (cb_exps)
 *   3. row sum of the BF16 exponentials accumulated in fp32, reciprocal, packed to bf16
 *   4. exps * recip in fp32 from bf16 operands, packed to bf16 (cb_out)
 *
 * Step 3 consuming the ROUNDED exponentials rather than the fp32 ones is the whole reason
 * this is a model and not a second reference: it is where a real softmax loses its accuracy
 * on this hardware, and it is invisible to anything that stays in fp32 end to end.
 *
 * @param rows  must be a multiple of 32; `cols` must be a multiple of 32
 */
void ft_tt_vector_model_softmax(const float *x, float *out, uint32_t rows, uint32_t cols);

/**
 * What the merged reduce triad produces for RMS norm, pass by pass.
 *
 * One pass and a two-step chain: x*x packed to bf16 in cb_mid, the row sum of THOSE bf16
 * squares accumulated in fp32 against a bf16 1/W scaler with the eps tile as one more
 * accumulated term, an rsqrt packed to bf16, then x*gamma packed to bf16 and multiplied by
 * the scale and packed again.
 */
void ft_tt_vector_model_rms_norm(const float *x, const float *gamma, float eps, float *out,
                                 uint32_t rows, uint32_t cols);

/** What the merged reduce triad produces for layer norm: two passes and a FOUR-step chain,
 *  so four bf16 packs per output tile — the most rounding of any member, and worth being
 *  able to see before spending a core on it. */
void ft_tt_vector_model_layer_norm(const float *x, const float *gamma, const float *beta,
                                   float eps, float *out, uint32_t rows, uint32_t cols);

/** What the merged reduce triad produces for the router normalise: one SUM pass with a
 *  reciprocal, one chain step. */
void ft_tt_vector_model_router_normalise(const float *x, float *out, uint32_t rows,
                                         uint32_t cols);

/**
 * The merged reduce triad driven by the PROGRAM WORDS themselves, with the scaler and the
 * eps tile supplied rather than derived.
 *
 * The four models above each bake in "the scaler is 1/cols", which is exactly the
 * assumption a PADDED descriptor breaks: the reduction runs over `cols` lanes while the
 * divisor belongs to a narrower, real W. This one takes both separately, so a caller can
 * hand it the padded activation and the scaler tile READ OUT OF the constants buffer the
 * host built — and then the model is not re-deriving the padding it is supposed to be
 * scoring (dev notes §3: never instrument a mechanism with itself).
 *
 * It is also the only model that walks the descriptor, so it doubles as a check that the
 * program words encode the op: `ft_tt_op_reduce_program()`'s output goes in, and the four
 * members' existing models are what it is validated against at an exact W.
 *
 * @param pass0_word  from ft_tt_op_reduce_program()
 * @param pass1_word  from ft_tt_op_reduce_program(); read only at pass count 2
 * @param final_word  pass count, final-chain source, chain step flags
 * @param x           [rows x cols] row-major, ALREADY padded — cols must be a multiple of 32
 * @param gamma       `cols` values, read only when the chain multiplies by gamma
 * @param beta        `cols` values, read only when the chain adds beta
 * @param scaler      the value in the scaler tile, NOT 1/cols
 * @param eps_tile    the value in the eps tile
 * @param out         [rows x cols]; filled with NaN if the program is one this cannot run,
 *                    so a refusal is loud in the non-finite census rather than silent
 */
void ft_tt_vector_model_reduce_program(uint32_t pass0_word, uint32_t pass1_word,
                                       uint32_t final_word, const float *x, const float *gamma,
                                       const float *beta, float scaler, float eps_tile,
                                       float *out, uint32_t rows, uint32_t cols);

/**
 * The same program placed at `reduce_float32.cbtable` instead of `reduce_float16b.cbtable`.
 *
 * IDENTICAL ARGUMENTS AND IDENTICAL KERNEL SOURCE — the CB table is the only difference, and
 * modelling it takes exactly two substitutions:
 *
 *   * a value stored to L1 (or to the DRAM strip) is kept EXACTLY instead of bf16-rounded;
 *   * a value read back into the source registers is TRUNCATED TO Tf32 — 11 mantissa bits,
 *     toward zero, ~4.9e-4 an operand and biased low.
 *
 * The second is why "all Float32" is not "fp32 arithmetic", and it is the floor this program
 * cannot go below however the table is written. Use it to price the move BEFORE spending card
 * time: `tests/test_vector_ops.c` scores both formats on the GDN head shape (W = 128) against
 * the same fp32 oracle, which is the comparison `PN_TT_GDN_NORMCHECK` makes on silicon.
 *
 * @param x  fp32 values — NOT pre-rounded. A Float32 strip stores them as given, and
 *           pre-rounding the input would price the operand's format into the reduce's score.
 */
void ft_tt_vector_model_reduce_program_float32(uint32_t pass0_word, uint32_t pass1_word,
                                               uint32_t final_word, const float *x,
                                               const float *gamma, const float *beta,
                                               float scaler, float eps_tile, float *out,
                                               uint32_t rows, uint32_t cols);

/**
 * THE COUNTERFACTUAL: the same program with an fp32 unpack the hardware does not have.
 *
 * It is not a CB table anyone can write and it is not a target. It exists to ATTRIBUTE — run
 * it beside `..._float32()` and the difference between the two residues is Tf32 truncation
 * and nothing else, which is what turns "the Float32 program still misses the gate" into
 * "and here is the term that is left, and no table can remove it".
 */
void ft_tt_vector_model_reduce_program_exact(uint32_t pass0_word, uint32_t pass1_word,
                                             uint32_t final_word, const float *x,
                                             const float *gamma, const float *beta,
                                             float scaler, float eps_tile, float *out,
                                             uint32_t rows, uint32_t cols);

/* ================================================================
 * THE TWO GDN HEAD NORMS AT Float32, WITH EVERY TRUNCATION STAGE UNDER A SWITCH
 *
 * `..._float32()` above answers "does Tf32 clear 1e-3". It cannot answer "WHICH stage spends
 * the budget", and that is the question left once the answer is no: the L2 member's chain is
 * already ONE step and it still misses the gate on silicon, so "shorten the chain" is a
 * hypothesis about the RMS member's EXCESS and not about the floor either member sits on.
 *
 * This models the SAME program — one SUM pass over x*x with eps and an rsqrt, then a chain of
 * [gamma] and scale — with each Tf32 unpack independently removable and the SFPU's own rsqrt
 * substitutable for libm's. It REFUSES any other program (NaN out) rather than quietly
 * modelling something else, and at `arms = 0` it reproduces `..._float32()` BIT-FOR-BIT, which
 * is the control that licenses every arm run through it.
 *
 * The four shipped members' model is deliberately untouched — this is a second function, not
 * a flag threaded through theirs.
 * ================================================================ */

/** The SFPU's own rsqrt substituted for libm's. */
#define FT_TT_REDUCE_ARM_SFPU_RSQRT 0x0001u
/** RED ARM: rsqrt off by 1e-3 relative — what the SFPU polynomial would have to be to account
 *  for the card's 2x over this model. Proves the score can SEE an rsqrt defect. */
#define FT_TT_REDUCE_ARM_RSQRT_PERTURB 0x0002u
/** The compute kernel rounds DST to the NEAREST Tf32 before every pack, so the unpacker's
 *  truncation becomes a no-op and what is left is unbiased. An SFPU pass, i.e. kernel source. */
#define FT_TT_REDUCE_ARM_ROUND_NEAREST 0x0004u
/** The HOST pre-rounds the fp32 strip and the constants buffer the same way — free, since it
 *  is a render it already does, and it covers the operands the compute kernel cannot reach. */
#define FT_TT_REDUCE_ARM_HOST_ROUND 0x0008u
/** The chain reordered: build `gamma * column` first, then ONE multiply by x. Same two FPU
 *  multiplies; the value path is one unpack shorter and the constants' is one longer. */
#define FT_TT_REDUCE_ARM_PREMULTIPLY 0x0010u
/** gamma folded into a downstream weight, so the RMS member runs the L2 member's ONE-step
 *  chain. The caller passes gamma = NULL and scores against a gamma-free oracle. */
#define FT_TT_REDUCE_ARM_NO_GAMMA 0x0020u

/** Each Tf32 unpack, removable on its own. NOT reachable on hardware — these ATTRIBUTE. */
#define FT_TT_REDUCE_ARM_EXACT_MAP 0x0040u     /**< x, unpacked by the squaring map */
#define FT_TT_REDUCE_ARM_EXACT_MID 0x0080u     /**< x*x, unpacked by the sum reduce */
#define FT_TT_REDUCE_ARM_EXACT_CHAIN_X 0x0100u /**< x, unpacked AGAIN by the chain */
#define FT_TT_REDUCE_ARM_EXACT_COLUMN 0x0200u  /**< the rsqrt column, unpacked by the chain */
#define FT_TT_REDUCE_ARM_EXACT_GAMMA 0x0400u   /**< gamma, unpacked by the chain */
#define FT_TT_REDUCE_ARM_EXACT_STAGE 0x0800u   /**< the chain's scratch-CB round trip */

/**
 * THE UNPACK KEEPS NINE STORED MANTISSA BITS, NOT TEN.
 *
 * "E8M10" is written down two ways and the difference is exactly one bit: ten stored plus a
 * hidden one, or ten of significand of which nine are stored. The ten-bit reading models the
 * card 2x optimistic on BOTH members; the nine-bit reading reproduces the q/k L2 member's
 * silicon figure — 1.737e-03 — to four significant figures on uniform data. It is a
 * HYPOTHESIS with one strong prediction behind it, and the arm that settles it is
 * `PN_TT_RED=head_norm_synthetic` on the card.
 */
#define FT_TT_REDUCE_ARM_TF32_NINE_BITS 0x1000u

/**
 * `ft_sfpu_reciprocal_sqrt()` (kernels/ft_tensix_sfpu_math.h) in scalar fp32 — the magic-seed
 * Kokosinski/Gepner/Moroz iteration the card actually runs, not libm's.
 *
 * Every operation of it is an SFPU float op and DST accumulates in fp32 under this family's
 * tables (`dst_accum_mode 1`), so the kernel's bf16-rounding `#if` is not taken and a scalar
 * fp32 evaluation IS the card's arithmetic.
 */
float ft_tt_vector_sfpu_rsqrt(float x);

/**
 * One Float32 head norm per row, with `arms` naming what differs from the shipped program.
 *
 * @param gamma  `cols` values, or NULL — which is both the L2 member and
 *               `FT_TT_REDUCE_ARM_NO_GAMMA`
 * @param arms   any combination of the flags above; 0 is `..._float32()` exactly
 * @param out    [rows x cols], or all-NaN if the program is not this shape
 */
void ft_tt_vector_model_head_norm_float32(uint32_t pass0_word, uint32_t pass1_word,
                                          uint32_t final_word, const float *x,
                                          const float *gamma, float scaler, float eps_tile,
                                          float *out, uint32_t rows, uint32_t cols,
                                          uint32_t arms);

/* ================================================================
 * Scoring — both metrics, always
 * ================================================================ */

/**
 * Pearson correlation of two arrays: the acceptance metric, and the COARSE one.
 *
 * A per-column corruption moved this only 0.99998 -> 0.929 while max-abs moved 0.024 ->
 * 0.841 (dev notes §3), so it is never reported alone.
 *
 * @return the correlation, or -2.0 if either side has a non-finite element — which is a
 *         refusal to answer and not a score, and the caller must say so
 */
double ft_tt_vector_pearson(const float *a, const float *b, size_t count);

/** Largest absolute difference. The SENSITIVE metric; report it beside every PCC. */
double ft_tt_vector_max_abs_difference(const float *a, const float *b, size_t count);

/**
 * Count non-finite elements on EACH SIDE separately, with the first offending pair.
 *
 * `PCC -nan, max abs inf` collapses under a single bad element out of 65,536 and does not
 * even say which side was bad; a session went into "the kernel regressed" when the truth
 * was a damaged core writing 0x7F80 into 1.1% of the output (bringup log §16.0). This is
 * the instrument that turns that round in one run.
 *
 * @param out_detail  receives a human-readable census; never truncated silently
 * @return            total non-finite elements across both sides
 */
size_t ft_tt_vector_count_non_finite(const float *expected, const float *actual, size_t count,
                                     char *out_detail, size_t detail_bytes);

/* ================================================================
 * Test data
 * ================================================================ */

/** Deterministic xorshift64*, seeded per test so a failure is reproducible. */
void ft_tt_vector_random_seed(uint64_t seed);

/**
 * A value in [-1, 1) with a FULL MANTISSA.
 *
 * Never small integers: a bf16-exact array is bit-identical to its own bf16 rendering with
 * zeros interleaved, so wrong CB descriptors reproduce the right answer and two
 * deliberately-different builds come out byte-equal (dev notes §3). Small integers are the
 * right choice for testing arithmetic and the wrong one for testing a format or a layout.
 */
float ft_tt_vector_random_value(void);

/** A value in [-`scale`, `scale`) with a full mantissa. For the softmax range test, where
 *  the whole question is what happens when the row is wide in magnitude. */
float ft_tt_vector_random_value_scaled(float scale);

#endif /* FT_TT_VECTOR_REFERENCE_H */
