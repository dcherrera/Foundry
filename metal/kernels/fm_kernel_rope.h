/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_rope.h
 * @brief Rotary position embedding family (NEOX pairing, partial rotation).
 *   Rotates the first rope_dim of each head_dim; the remaining dims pass
 *   through unchanged. No shared memory, no shuffles — one thread per pair.
 *
 * @status Complete: single-position and per-row-position forms, both in
 *   place.
 * @issues None
 * @todo None
 *
 * FUNCTION-CONSTANT INDICES OWNED BY THIS FAMILY: 100-109 (100, 101 used).
 *   Function constants share ONE index space across the whole assembled
 *   shader library, so each family reserves a block and two families
 *   colliding on an index fails the compile of every kernel in the backend,
 *   not just theirs. Allocated so far: 0-2 probe · 30 Q3_K · 50 Q5_K ·
 *   60 Q6_K · 70/80 Q8_0+Q4_K · 100-109 rope · 110-119 norm ·
 *   120-129 elementwise (claimed, none used).
 */

#ifndef FOUNDRY_METAL_KERNEL_ROPE_H
#define FOUNDRY_METAL_KERNEL_ROPE_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Partial NEOX rotary embedding over one token's heads, in place.
 *
 * Only the first `rope_dim` dims of each head are rotated; dims
 * [rope_dim, head_dim) are left byte-for-byte alone — the kernel never
 * writes them. NEOX pairing rotates dim j against dim j + rope_dim/2:
 *
 *     angle    = pos / theta_base^(2j / rope_dim)
 *     out[j]   = x[j] * cos(angle) - x[j + half] * sin(angle)
 *     out[j+h] = x[j] * sin(angle) + x[j + half] * cos(angle)
 *
 * `rope_dim` outside (0, head_dim) means "rotate the whole head", matching
 * the CPU reference.
 *
 * @param data        [n_heads, head_dim] float32, rotated in place
 * @param n_heads     Number of heads in `data`
 * @param head_dim    Elements per head
 * @param rope_dim    Leading dims to rotate (64 for qwen35moe)
 * @param pos         Token position; 0 is an exact identity
 * @param theta_base  Frequency base (1e7 for qwen35moe)
 *
 * Thread-safe: no (encodes into the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_rope_neox_partial(fm_buffer *data, int n_heads, int head_dim,
                          int rope_dim, int pos, float theta_base);

/**
 * fm_rope_neox_partial over a batch of rows, each at its own position —
 * the prefill form, where one dispatch rotates a whole chunk of tokens.
 *
 * @param data        [n_rows, n_heads, head_dim] float32, rotated in place
 * @param positions   int32[n_rows], the token position of each row
 * @param n_rows      Number of token rows
 * @param n_heads     Number of heads per row
 * @param head_dim    Elements per head
 * @param rope_dim    Leading dims to rotate
 * @param theta_base  Frequency base
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
void fm_rope_neox_partial_pos(fm_buffer *data, fm_buffer *positions,
                              int n_rows, int n_heads, int head_dim,
                              int rope_dim, float theta_base);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_ROPE_H */
