/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pl_lora.c
 * @brief LoRA adapter forward/backward over T tokens + fp64 FD check (code, Mac).
 *
 * Spec §5.1.1 item 3 — low-rank adaptation of a linear layer.
 *
 * @status Experimental
 * @issues None
 * @todo None
 */
#include "pl_lora.h"
#include "pl_rng.h"

#include <stdlib.h>
#include <string.h>

int pl_lora_init(pl_lora *L, int din, int dout, int rank,
                 float alpha, uint64_t seed, int layer_index)
{
    if (!L || din <= 0 || dout <= 0 || rank <= 0 || alpha <= 0.f)
        return -1;

    L->din    = din;
    L->dout   = dout;
    L->rank   = rank;
    L->scale  = alpha / (float)rank;

    size_t nA = (size_t)rank * (size_t)din;
    size_t nB = (size_t)dout * (size_t)rank;

    L->A = (float *)malloc(nA * sizeof(float));
    L->B = (float *)calloc(nB, sizeof(float));   /* B = 0 */
    L->dA = (float *)calloc(nA, sizeof(float));
    L->dB = (float *)calloc(nB, sizeof(float));

    if (!L->A || !L->B || !L->dA || !L->dB) {
        free(L->A); free(L->B); free(L->dA); free(L->dB);
        L->A = L->B = L->dA = L->dB = NULL;
        return -1;
    }

    /* A from pl_lora_init_a(seed, layer_index) */
    pl_lora_init_a(seed, layer_index, L->A, rank, din);
    return 0;
}

void pl_lora_free(pl_lora *L)
{
    if (!L) return;
    free(L->A); free(L->B); free(L->dA); free(L->dB);
    memset(L, 0, sizeof(*L));
}

void pl_lora_zero_grad(pl_lora *L)
{
    if (!L) return;
    memset(L->dA, 0, (size_t)L->rank * (size_t)L->din * sizeof(float));
    memset(L->dB, 0, (size_t)L->dout * (size_t)L->rank * sizeof(float));
}

/* Forward: y[t][o] += s·Σ_r B[o][r]·Σ_j A[r][j]·x[t][j],  t = 0..T-1.
 * x is [T][din], y is [T][dout], row-major.
 * h is [T][rank] scratch, malloc'd per call.
 */
void pl_lora_forward(const pl_lora *L, const float *x, float *y, int T)
{
    if (!L || !x || !y || T <= 0) return;

    float *h = (float *)calloc((size_t)T * (size_t)L->rank, sizeof(float));
    if (!h) return;

    const int rank = L->rank;
    const int din  = L->din;
    const int dout = L->dout;
    const float s  = L->scale;

    /* Step 1: compute h[t][r] = Σ_j A[r][j]·x[t][j] */
    for (int t = 0; t < T; t++) {
        for (int r = 0; r < rank; r++) {
            float sum = 0.f;
            for (int j = 0; j < din; j++) {
                sum += L->A[r * din + j] * x[t * din + j];
            }
            h[t * rank + r] = sum;
        }
    }

    /* Step 2: y[t][o] += s·Σ_r B[o][r]·h[t][r] */
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < dout; o++) {
            float sum = 0.f;
            for (int r = 0; r < rank; r++) {
                sum += L->B[o * rank + r] * h[t * rank + r];
            }
            y[t * dout + o] += s * sum;
        }
    }

    free(h);
}

/* Backward: accumulates dA, dB; optionally writes dx.
 * x is [T][din], dy is [T][dout], dx is [T][din] (may be NULL).
 * h and dh are [T][rank] scratch, malloc'd per call.
 * Equations: recompute h; dB += s·Σ_t dy·h; dh = s·Σ_o B·dy;
 *   dA += Σ_t dh·x; dx += Σ_r A·dh (skip if dx == NULL).
 */
void pl_lora_backward(const pl_lora *L, const float *x, const float *dy,
                      float *dx, int T)
{
    if (!L || !x || !dy || T <= 0) return;

    float *h  = (float *)calloc((size_t)T * (size_t)L->rank, sizeof(float));
    float *dh = (float *)calloc((size_t)T * (size_t)L->rank, sizeof(float));
    if (!h || !dh) { free(h); free(dh); return; }

    const int rank = L->rank;
    const int din  = L->din;
    const int dout = L->dout;
    const float s  = L->scale;

    /* Recompute h[t][r] = Σ_j A[r][j]·x[t][j] */
    for (int t = 0; t < T; t++) {
        for (int r = 0; r < rank; r++) {
            float sum = 0.f;
            for (int j = 0; j < din; j++) {
                sum += L->A[r * din + j] * x[t * din + j];
            }
            h[t * rank + r] = sum;
        }
    }

    /* dB[o][r] += s·Σ_t dy[t][o]·h[t][r] */
    for (int o = 0; o < dout; o++) {
        for (int r = 0; r < rank; r++) {
            float sum = 0.f;
            for (int t = 0; t < T; t++) {
                sum += dy[t * dout + o] * h[t * rank + r];
            }
            L->dB[o * rank + r] += s * sum;
        }
    }

    /* dh[t][r] = s·Σ_o B[o][r]·dy[t][o] */
    for (int t = 0; t < T; t++) {
        for (int r = 0; r < rank; r++) {
            float sum = 0.f;
            for (int o = 0; o < dout; o++) {
                sum += L->B[o * rank + r] * dy[t * dout + o];
            }
            dh[t * rank + r] = s * sum;
        }
    }

    /* dA[r][j] += Σ_t dh[t][r]·x[t][j] */
    for (int r = 0; r < rank; r++) {
        for (int j = 0; j < din; j++) {
            float sum = 0.f;
            for (int t = 0; t < T; t++) {
                sum += dh[t * rank + r] * x[t * din + j];
            }
            L->dA[r * din + j] += sum;
        }
    }

    /* dx[t][j] += Σ_r A[r][j]·dh[t][r] (skip if dx == NULL) */
    if (dx) {
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < din; j++) {
                float sum = 0.f;
                for (int r = 0; r < rank; r++) {
                    sum += L->A[r * din + j] * dh[t * rank + r];
                }
                dx[t * din + j] += sum;
            }
        }
    }

    free(h);
    free(dh);
}