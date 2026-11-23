/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pl_lora.h
 * @brief LoRA adapter forward/backward over T tokens + fp64 FD check (code, Mac).
 *
 * Spec §5.1.1 item 3 — low-rank adaptation of a linear layer.
 *
 * @status Experimental
 * @issues None
 * @todo None
 */
#ifndef PL_LORA_H
#define PL_LORA_H

#include <stdint.h>

/* LoRA adapter: low-rank update W0 + s·B·A on top of a base linear. */
typedef struct {
    int din, dout, rank;
    float scale;              /* alpha / rank */
    float *A, *B;             /* [rank*din], [dout*rank] */
    float *dA, *dB;           /* grads, same shapes; accumulated (+=) by backward */
} pl_lora;

/* Initialize LoRA adapter. A from pl_lora_init_a(seed, layer_index),
 * B = 0, dA = dB = 0. Returns 0 on success or -1 on bad args/alloc.
 */
int  pl_lora_init(pl_lora *L, int din, int dout, int rank,
                  float alpha, uint64_t seed, int layer_index);

/* Free all LoRA adapter memory. */
void pl_lora_free(pl_lora *L);

/* Zero out gradient buffers dA and dB. */
void pl_lora_zero_grad(pl_lora *L);

/* Forward: y[t][o] += s·Σ_r B[o][r]·Σ_j A[r][j]·x[t][j],  t = 0..T-1.
 * x is [T][din], y is [T][dout], row-major.
 */
void pl_lora_forward(const pl_lora *L, const float *x, float *y, int T);

/* Backward: accumulates dA, dB; optionally writes dx.
 * x is [T][din], dy is [T][dout], dx is [T][din] (may be NULL).
 * h and dh are [T][rank] scratch, malloc'd per call.
 * Equations: recompute h; dB += s·Σ_t dy·h; dh = s·Σ_o B·dy;
 *   dA += Σ_t dh·x; dx += Σ_r A·dh (skip if dx == NULL).
 */
void pl_lora_backward(const pl_lora *L, const float *x, const float *dy,
                      float *dx, int T);

#endif /* PL_LORA_H */