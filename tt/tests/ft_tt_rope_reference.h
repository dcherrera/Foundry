/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_rope_reference.h
 * @brief The two CPU sides of the RoPE gate, kept apart on purpose: an INDEPENDENT
 *   reference written from the op's definition, and a DEVICE MODEL that emulates the card's
 *   arithmetic exactly.
 *
 * They answer different questions and a suite that has only one of them cannot tell a wrong
 * op from a rounding:
 *
 *   REFERENCE  transliterated from `foundry/cuda_kernels/rope.cu` (`fc_rope_neox_partial`)
 *              and `Pantheon/src/models/llama_forward_cuda.c` (`llama_rope_neox`) — fp32,
 *              `powf`/`cosf`/`sinf`, per-j, in place. It knows nothing about tiles, faces,
 *              tables or circular buffers, and it is what the card is SCORED against.
 *   MODEL      the card's own sequence: bf16 operands, two products each rounded to bf16 at
 *              their pack, then a sum rounded to bf16 at its pack. It takes the sin/cos as
 *              ARRAYS rather than computing them, so a test can feed it a synthetic table of
 *              exactly-representable values and demand BYTE equality — which is how the
 *              structure (pairing, partner order, pass-through, face map) gets a gate with no
 *              rounding tolerance to hide in.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FT_TT_ROPE_REFERENCE_H
#define FT_TT_ROPE_REFERENCE_H

#include <stddef.h>
#include <stdint.h>

/**
 * Partial NeoX RoPE on ONE token row of `heads * head_dim` fp32 features, in place.
 *
 * Verbatim in structure from `fc_rope_neox_partial`: `rot = (rope_dim > 0 && rope_dim <=
 * head_dim) ? rope_dim : head_dim`, `half = rot/2`, `freq = 1/powf(theta, 2j/rot)`,
 * `angle = pos * freq`, and the pair `(j, j+half)` rotated by it. Features `[rot, head_dim)`
 * are not touched.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
void ft_tt_rope_reference_row(float *row, uint32_t heads, uint32_t head_dim,
                              uint32_t rope_dim, uint32_t position, float theta_base);

/**
 * `ft_tt_rope_reference_row()` over `rows` token rows of a [rows x heads*head_dim] matrix.
 *
 * Row `r` is rotated at position `block_first_position + r`, which is the ROW-PHASE CONTRACT
 * stated as arithmetic: the caller passes the position of ROW 0 of the table block, i.e.
 * `first_position - (first_position % 32)`.
 */
void ft_tt_rope_reference_apply(const float *in, float *out, uint32_t rows, uint32_t heads,
                                uint32_t head_dim, uint32_t rope_dim,
                                uint32_t block_first_position, float theta_base);

/**
 * The DEVICE MODEL: what the card computes, rounding where the card rounds.
 *
 * For each rotated column `c` of each head, with `partner(c) = (c + rot/2) mod rot`:
 *
 *     out[c] = bf16( bf16(in[c] * cos[c]) + bf16(in[partner(c)] * sin[c]) )
 *
 * and `out[c] = in[c]` for `c >= rot`. Every array is PLAIN (untilized) bf16, laid out
 * [rows x heads*head_dim] for the activations and [rows x rot] for each half of the table.
 * `sin` is the SIGNED table — the sign is not applied here, because it is not applied on the
 * card either.
 *
 * @param cos_table  [rows x rot] bf16
 * @param sin_table  [rows x rot] bf16, already signed
 */
void ft_tt_rope_model_apply(const uint16_t *in, const uint16_t *cos_table,
                            const uint16_t *sin_table, uint16_t *out, uint32_t rows,
                            uint32_t heads, uint32_t head_dim, uint32_t rot);

/**
 * Fill a [32 x rot] pair of PLAIN bf16 table halves for `block_first_position`, the same
 * values `ft_tt_op_rope_build_table()` tilizes.
 *
 * Exposed so a test can compare the TILIZED artifact against the plain values through
 * `ft_tt_tile_face_index()` — which is the only check that can see a table tilized with the
 * wrong face map, because such a table is a wrong ANGLE and not a permuted answer.
 */
void ft_tt_rope_reference_table(uint16_t *cos_table, uint16_t *sin_table,
                                uint32_t block_first_position, uint32_t rot,
                                double theta_base);

#endif /* FT_TT_ROPE_REFERENCE_H */
