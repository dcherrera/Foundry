/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   pl_q8.h
 * @brief  Q8_0 frozen-base quantization and dequantization for linear layers.
 *
 * Static fp16 conversion helpers + row-wise Q8_0 quantize / dequant functions.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef POTLUCK_PL_Q8_H
#define POTLUCK_PL_Q8_H

#include <stdint.h>

typedef struct {
    int in, out;          /* W is [in,out]; stored transposed: one Q8_0 row per output o, along in */
    int nb;               /* in / 32 (in must be a multiple of 32, else pl_q8_from_f32 returns -1) */
    uint8_t *blocks;      /* [out][nb][34] */
} pl_q8;

int  pl_q8_from_f32(const float *W_inout, int in, int out, pl_q8 *Q);   /* row o = column o of W */
void pl_q8_free(pl_q8 *Q);
void pl_q8_dequant_rows(const pl_q8 *Q, int o0, int n, float *dst);     /* dst [n][in] = Wᵀ rows o0..o0+n-1 */
/* y[T][out]  = x[T][in] · W          (overwrites y)  */
void pl_q8_forward(const pl_q8 *Q, const float *x, float *y, int T);
/* dx[T][in] += dy[T][out] · Wᵀ       (accumulates)   */
void pl_q8_backward_dx(const pl_q8 *Q, const float *dy, float *dx, int T);

#endif /* POTLUCK_PL_Q8_H */