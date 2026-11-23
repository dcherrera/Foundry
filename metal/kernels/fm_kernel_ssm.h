/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_ssm.h
 * @brief Gated-delta SSM family: decay gate, causal conv1d+SiLU, gated RMSNorm,
 *   and the recurrence (register-resident decode form plus a sequential
 *   reference scan kept permanently for chunked-path parity).
 *
 * @status Complete: fm_ssm_decay_gate, fm_ssm_conv1d_silu(_seq),
 *   fm_ssm_gated_norm, fm_ssm_recurrence and fm_ssm_recurrence_seq, each with
 *   an fm_bufref form and a whole-buffer convenience wrapper. Validated
 *   against the qwen35moe golden set in metal/tests/golden.
 * @issues None
 * @todo None
 *
 * THREAD MAPPING AND LANE OCCUPANCY
 *   fm_ssm_recurrence / _seq: ONE SIMD group (32 lanes) owns one (head, state
 *     row) pair, and each lane holds state_size/32 row elements in REGISTERS
 *     across the whole token loop — 4 at the model's state_size = 128. Lane L
 *     owns elements L, 32+L, 64+L, 96+L, so every device access to the row is
 *     a fully coalesced 128-byte transaction and no lane is idle. Eight SIMD
 *     groups share a 256-thread threadgroup, so the model's 32 heads x 128
 *     rows = 4096 pairs dispatch as 512 threadgroups. The two per-token
 *     reductions are simd_sum: no threadgroup barrier, no shared-memory state
 *     array, and the state reaches device memory exactly once per dispatch.
 *     Lanes only idle if state_size is not a multiple of 32, which no shape in
 *     this model has.
 *   fm_ssm_gated_norm: one SIMD group per V-head, head_dim/32 = 4 elements per
 *     lane, one simd_sum — same shape as fm_qk_rms_norm.
 *   fm_ssm_conv1d_silu(_seq): one thread per channel over conv_dim = 8192,
 *     depthwise so channels never interact; the (conv_kernel - 1)-deep history
 *     lives in registers.
 *   fm_ssm_decay_gate: one thread per (row, head).
 *
 *   Measured on M4 (10 GPU cores) / macOS 26 at the model shape, with the
 *   command-buffer round trip amortized across 50 encoded dispatches:
 *     recurrence 0.0303 ms · conv1d 0.0086 ms · gated_norm 0.0078 ms.
 *   The recurrence moves 4 MiB of state (2 MiB in, 2 MiB out) in that time =
 *   138 GB/s, i.e. it saturates the memory path — the same above-DRAM-roof
 *   reading fm_rms_norm_nx shows, for the same reason: a 2 MiB working set
 *   stays in cache across the repeats. The other three are too small to be
 *   bandwidth-bound; fm_ssm_decay_gate is 32 threads and is pure dispatch
 *   cost, so it should ride in the same command buffer as its neighbours
 *   rather than be measured or optimized on its own.
 *
 * FUNCTION-CONSTANT INDICES OWNED BY THIS FAMILY: 200-219 (200-202 used).
 *   Function constants share ONE index space across the whole assembled
 *   shader library, so each family reserves a block; two families colliding
 *   on an index fails the compile of every kernel in the backend, not just
 *   theirs. Allocated so far: 0-2 probe / 30 Q3_K / 50 Q5_K / 60 Q6_K / 70 Q8_0 / 80 Q4_K /
 *   100-109 rope / 110-119 norm / 120-129 elementwise / 200-219 ssm /
 *   220-239 ssm_chunked / 240-259 moe / 260-279 attn.
 */

#ifndef FOUNDRY_METAL_KERNEL_SSM_H
#define FOUNDRY_METAL_KERNEL_SSM_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Shape limits
 *
 * These bound the thread-local arrays the shaders declare. The arrays are
 * indexed only by loops whose trip count is a function constant, so after
 * unrolling every element is a named register and the unused tail costs
 * nothing — but the declared bound must still be a literal, because a
 * function constant is not a valid array size in MSL.
 * ================================================================ */

/* Largest state_size fm_ssm_recurrence accepts: 32 lanes x 8 registers. */
#define FM_SSM_MAX_STATE_SIZE 256

/* Largest conv_kernel fm_ssm_conv1d_silu accepts (history is k-1 deep). */
#define FM_SSM_MAX_CONV_KERNEL 8

/* ================================================================
 * Function-constant indices
 *
 * The MSL fragment spells the same numbers literally in its
 * [[function_constant(N)]] attributes — change one and change both.
 * ================================================================ */

#define FM_SSM_FC_STATE_SIZE   200   /* recurrence state_size */
#define FM_SSM_FC_CONV_KERNEL  201   /* conv1d tap count */
#define FM_SSM_FC_HEAD_DIM     202   /* gated-norm per-head width */

/* ================================================================
 * Decay gate
 * ================================================================ */

/**
 * Gated-delta decay gate: g[r][h] = a[h] * softplus(alpha[r][h] + dt[h]).
 *
 * softplus is guarded at 20 — above it softplus(v) is v to well within
 * float32, and evaluating log1p(exp(v)) there would overflow exp to
 * infinity and poison the whole layer. `a` is the pre-baked -exp(A_log),
 * so g is negative and exp(g) lands in (0, 1).
 *
 * `g` is PRE-exp: fm_ssm_recurrence exponentiates it. `dt` and `a` are
 * per-head and broadcast across rows; only `alpha` and `g` carry a row axis,
 * which is why a single entry point covers both decode (rows == 1) and
 * prefill instead of two kernels that could drift apart.
 *
 * @param alpha    [rows, n_heads] float32 projection output
 * @param dt       [n_heads] float32 per-head bias
 * @param a        [n_heads] float32 per-head -exp(A_log)
 * @param g        [rows, n_heads] float32 output, pre-exp
 * @param n_heads  Heads per row (n_v_heads)
 * @param rows     Token count; 1 for decode
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_ssm_decay_gate_at(fm_bufref alpha, fm_bufref dt, fm_bufref a,
                          fm_bufref g, int n_heads, int rows);

/* fm_ssm_decay_gate_at with every operand bound at offset 0. */
void fm_ssm_decay_gate(fm_buffer *alpha, fm_buffer *dt, fm_buffer *a,
                       fm_buffer *g, int n_heads, int rows);

/* ================================================================
 * Causal depthwise conv1d + SiLU
 * ================================================================ */

/**
 * One step of the depthwise causal conv1d over conv_dim channels, followed
 * by SiLU, rolling conv_state in place.
 *
 * Channel c's window is [conv_state[c][0] .. conv_state[c][k-2], qkv[c]]:
 *   out[c] = silu(qkv[c] * w[c][k-1] + sum_t conv_state[c][t] * w[c][t])
 * and conv_state[c] then drops its oldest tap and appends the RAW qkv[c],
 * not the convolved value.
 *
 * @param qkv          [conv_dim] float32, this token's projected stream
 * @param weight       [conv_dim, conv_kernel] float32, row-major
 * @param conv_state   [conv_dim, conv_kernel-1] float32, oldest-first,
 *                     updated in place
 * @param out          [conv_dim] float32
 * @param conv_dim     Channel count
 * @param conv_kernel  Taps per channel, 2 .. FM_SSM_MAX_CONV_KERNEL
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
void fm_ssm_conv1d_silu_at(fm_bufref qkv, fm_bufref weight, fm_bufref conv_state,
                           fm_bufref out, int conv_dim, int conv_kernel);

/* fm_ssm_conv1d_silu_at with every operand bound at offset 0. */
void fm_ssm_conv1d_silu(fm_buffer *qkv, fm_buffer *weight, fm_buffer *conv_state,
                        fm_buffer *out, int conv_dim, int conv_kernel);

/**
 * The conv1d scan over n_tokens consecutive positions of ONE sequence.
 *
 * Each thread reads its channel's history once into registers, advances it
 * n_tokens times and writes it back once. The per-position arithmetic is
 * the single-step kernel's, term for term, so running this over n tokens and
 * running the single-step kernel n times produce bit-identical output and
 * bit-identical carried state.
 *
 * @param qkv          [n_tokens, conv_dim] float32, token-major
 * @param weight       [conv_dim, conv_kernel] float32, row-major
 * @param conv_state   [conv_dim, conv_kernel-1] float32, updated in place
 * @param out          [n_tokens, conv_dim] float32, token-major
 * @param conv_dim     Channel count
 * @param conv_kernel  Taps per channel, 2 .. FM_SSM_MAX_CONV_KERNEL
 * @param n_tokens     Positions to scan (>= 1)
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
void fm_ssm_conv1d_silu_seq_at(fm_bufref qkv, fm_bufref weight,
                               fm_bufref conv_state, fm_bufref out,
                               int conv_dim, int conv_kernel, int n_tokens);

/* fm_ssm_conv1d_silu_seq_at with every operand bound at offset 0. */
void fm_ssm_conv1d_silu_seq(fm_buffer *qkv, fm_buffer *weight,
                            fm_buffer *conv_state, fm_buffer *out,
                            int conv_dim, int conv_kernel, int n_tokens);

/* ================================================================
 * Gated RMSNorm
 * ================================================================ */

/**
 * Fused per-head gated RMSNorm: gated = RMSNorm(y) * weight * silu(z).
 *
 * Each V-head is normalized over its own head_dim values; the weight is
 * SHARED across heads (head_dim long, not n_heads * head_dim) exactly as
 * fm_qk_rms_norm's is, because GGUF stores one pre-baked norm weight per
 * tensor. The multiply order is ((y * inv_rms) * weight) * silu(z), matching
 * the CPU path's pn_vrms_norm_f32 followed by a silu multiply.
 *
 * @param y         [n_heads, head_dim] float32, the recurrence output
 * @param weight    [head_dim] float32 shared gain
 * @param z         [n_heads, head_dim] float32 gate projection, pre-silu
 * @param gated     [n_heads, head_dim] float32 output (may alias y)
 * @param n_heads   V-head count
 * @param head_dim  Elements per head; a multiple of 32
 * @param eps       Added to mean(y^2) before the rsqrt
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
void fm_ssm_gated_norm_at(fm_bufref y, fm_bufref weight, fm_bufref z,
                          fm_bufref gated, int n_heads, int head_dim, float eps);

/* fm_ssm_gated_norm_at with every operand bound at offset 0. */
void fm_ssm_gated_norm(fm_buffer *y, fm_buffer *weight, fm_buffer *z,
                       fm_buffer *gated, int n_heads, int head_dim, float eps);

/* ================================================================
 * Delta-rule recurrence
 * ================================================================ */

/**
 * One decode step of the gated-delta recurrence, updating the recurrent
 * state in place and emitting this token's readout.
 *
 * The state is TRANSPOSED: state[h*S*S + j*S + i] is S_h[i][j], so row j of
 * head h is contiguous over i and one SIMD group owns it. Per (h, j):
 *   row  *= exp(g[h])                     decay
 *   pred  = sum_i row[i] * k[h][i]        first simd_sum
 *   delta = (v[h][j] - pred) * beta[h]
 *   row  += k[h] * delta                  rank-1 update
 *   y[h][j] = scale * sum_i row[i] * q[h][i]   second simd_sum, read-after-write
 *
 * q and k must arrive already L2-normed and GQA-repeated to n_v_heads, and
 * beta already sigmoid'd; g is pre-exp. That is the same input contract the
 * CUDA kernel and the CPU reference take.
 *
 * @param state       [n_v_heads, S, S] float32, transposed, updated in place
 * @param q, k, v     [n_v_heads, S] float32
 * @param g           [n_v_heads] float32 pre-exp decay
 * @param beta        [n_v_heads] float32, already sigmoid'd
 * @param y           [n_v_heads, S] float32 output
 * @param n_v_heads   V-head count
 * @param state_size  S; a multiple of 32, at most FM_SSM_MAX_STATE_SIZE
 * @param scale       Readout scale, 1/sqrt(S) in the model
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
void fm_ssm_recurrence_at(fm_bufref state, fm_bufref q, fm_bufref k, fm_bufref v,
                          fm_bufref g, fm_bufref beta, fm_bufref y,
                          int n_v_heads, int state_size, float scale);

/* fm_ssm_recurrence_at with every operand bound at offset 0. */
void fm_ssm_recurrence(fm_buffer *state, fm_buffer *q, fm_buffer *k, fm_buffer *v,
                       fm_buffer *g, fm_buffer *beta, fm_buffer *y,
                       int n_v_heads, int state_size, float scale);

/**
 * The naive sequential scan of the recurrence over n_tokens positions of one
 * sequence — the PERMANENT correctness reference the chunked prefill path is
 * validated against, not a placeholder to delete once that path lands.
 *
 * Each SIMD group carries its state row in registers across all n_tokens
 * positions and writes it back once. The per-position arithmetic is
 * fm_ssm_recurrence's, term for term and reduction for reduction, so the
 * scan and n single-step dispatches agree bit for bit — which is what makes
 * this usable as an oracle rather than merely a close second opinion.
 *
 * @param state       [n_v_heads, S, S] float32, transposed, updated in place
 * @param q, k, v     [n_tokens, n_v_heads, S] float32, token-major
 * @param g           [n_tokens, n_v_heads] float32 pre-exp decay
 * @param beta        [n_tokens, n_v_heads] float32, already sigmoid'd
 * @param y           [n_tokens, n_v_heads, S] float32 output, token-major
 * @param n_v_heads   V-head count
 * @param state_size  S; a multiple of 32, at most FM_SSM_MAX_STATE_SIZE
 * @param n_tokens    Positions to scan (>= 1)
 * @param scale       Readout scale, 1/sqrt(S) in the model
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
void fm_ssm_recurrence_seq_at(fm_bufref state, fm_bufref q, fm_bufref k,
                              fm_bufref v, fm_bufref g, fm_bufref beta,
                              fm_bufref y, int n_v_heads, int state_size,
                              int n_tokens, float scale);

/* fm_ssm_recurrence_seq_at with every operand bound at offset 0. */
void fm_ssm_recurrence_seq(fm_buffer *state, fm_buffer *q, fm_buffer *k,
                           fm_buffer *v, fm_buffer *g, fm_buffer *beta,
                           fm_buffer *y, int n_v_heads, int state_size,
                           int n_tokens, float scale);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_SSM_H */
