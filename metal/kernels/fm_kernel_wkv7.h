/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_wkv7.h
 * @brief WKV-7 (Generalized Delta Rule) recurrence kernels — single timestep
 *   and whole-sequence. One threadgroup per head, head_size threads per group.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_METAL_KERNEL_WKV7_H
#define FOUNDRY_METAL_KERNEL_WKV7_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* WKV-7 single timestep (Generalized Delta Rule).
 * All inputs/outputs are flat float32 arrays.
 * state is [n_head, head_size, head_size], modified in place. */
void fm_wkv7_one(fm_buffer *r, fm_buffer *w, fm_buffer *k, fm_buffer *v,
                 fm_buffer *neg_kk, fm_buffer *kk_a,
                 fm_buffer *state, fm_buffer *y_out,
                 int n_head, int head_size);

/* WKV-7 sequence of timesteps.
 * Inputs are [T, n_head * head_size] except state. */
void fm_wkv7_seq(fm_buffer *r, fm_buffer *w, fm_buffer *k, fm_buffer *v,
                 fm_buffer *neg_kk, fm_buffer *kk_a,
                 fm_buffer *state, fm_buffer *y_out,
                 int n_head, int head_size, int t_len);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_WKV7_H */
