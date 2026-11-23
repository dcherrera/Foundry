/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   pl_q8.c
 * @brief  Q8_0 frozen-base quantization and dequantization for linear layers.
 * @status None
 * @issues None
 * @todo   None
 */

#include "pl_q8.h"
#include "tt/convert/ft_float_convert.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_BLAS
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#define TILE 64

static uint16_t float_to_fp16(float v)
{
    uint32_t bits; memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t frac = (bits >> 13) & 0x3ffu;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00u;
    uint32_t dropped = bits & 0x1ffffu;
    uint32_t rb = (dropped >> 12) & 1u;
    uint32_t sticky = (dropped & 0xfffu) != 0;
    uint32_t tie = rb && !sticky;
    frac += rb + (tie & (frac & 1u));
    if (frac >= 0x400u) { frac = 0; exp++; if (exp >= 31) return sign | 0x7c00u; }
    return sign | (exp << 10) | (uint16_t)frac;
}

int pl_q8_from_f32(const float *W_inout, int in, int out, pl_q8 *Q)
{
    if (in % 32 != 0) return -1;
    Q->in = in; Q->out = out; Q->nb = in / 32;
    Q->blocks = malloc((size_t)out * (size_t)Q->nb * 34);
    if (!Q->blocks) return -1;
    for (int o = 0; o < out; o++) {
        uint8_t *brow = Q->blocks + (size_t)o * (size_t)Q->nb * 34;
        for (int g = 0; g < Q->nb; g++) {
            float amax = 0.0f;
            for (int k = 0; k < 32; k++) {
                int idx = (g * 32 + k) * out + o;
                float a = fabsf(W_inout[idx]);
                if (a > amax) amax = a;
            }
            float d = amax / 127.0f, id = (d > 0.0f) ? (1.0f / d) : 0.0f;   /* ggml: q from fp32 d */
            uint16_t d_fp16 = float_to_fp16(d);
            uint8_t *blk = brow + g * 34;
            blk[0] = (uint8_t)(d_fp16 & 0xff); blk[1] = (uint8_t)((d_fp16 >> 8) & 0xff);
            for (int k = 0; k < 32; k++) {
                int idx = (g * 32 + k) * out + o;
                int q = (int)roundf(W_inout[idx] * id);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                blk[2 + k] = (uint8_t)q;
            }
        }
    }
    return 0;
}

void pl_q8_free(pl_q8 *Q) { if (Q && Q->blocks) { free(Q->blocks); Q->blocks = NULL; } }

void pl_q8_dequant_rows(const pl_q8 *Q, int o0, int n, float *dst);

#ifndef USE_BLAS
/* C[T][ldc] (n columns) = A[T][in] * B[n][in]^T, row-major, plain C */
static void gemm_fwd(const float *A, const float *B, float *C, int ldc, int T, int in, int n)
{
    for (int t = 0; t < T; t++) {
        for (int p = 0; p < n; p++) {
            float s = 0.0f;
            for (int i = 0; i < in; i++)
                s += A[t * in + i] * B[p * in + i];
            C[(size_t)t * ldc + p] = s;
        }
    }
}

/* C[T][in] += A[T][lda] (n columns) * B[n][in], row-major, plain C */
static void gemm_bwd(const float *A, int lda, const float *B, float *C, int T, int n, int in)
{
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < in; i++) {
            float s = 0.0f;
            for (int p = 0; p < n; p++)
                s += A[(size_t)t * lda + p] * B[p * in + i];
            C[(size_t)t * in + i] += s;
        }
    }
}
#endif

void pl_q8_forward(const pl_q8 *Q, const float *x, float *y, int T)
{
    int n_tile;
    float *scratch = malloc((size_t)TILE * (size_t)Q->in * sizeof(float));
    if (!scratch) return;

    for (int o0 = 0; o0 < Q->out; o0 += TILE) {
        n_tile = Q->out - o0;
        if (n_tile > TILE) n_tile = TILE;
        pl_q8_dequant_rows(Q, o0, n_tile, scratch);

#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, n_tile, Q->in,
                    1.0f, x, Q->in,
                    scratch, Q->in,
                    0.0f, y + (size_t)o0, Q->out);
#else
        gemm_fwd(x, scratch, y + (size_t)o0, Q->out, T, Q->in, n_tile);
#endif
    }
    free(scratch);
}

void pl_q8_backward_dx(const pl_q8 *Q, const float *dy, float *dx, int T)
{
    int n_tile;
    float *scratch = malloc((size_t)TILE * (size_t)Q->in * sizeof(float));
    if (!scratch) return;

    for (int o0 = 0; o0 < Q->out; o0 += TILE) {
        n_tile = Q->out - o0;
        if (n_tile > TILE) n_tile = TILE;
        pl_q8_dequant_rows(Q, o0, n_tile, scratch);

#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, Q->in, n_tile,
                    1.0f, dy + (size_t)o0, Q->out,
                    scratch, Q->in,
                    1.0f, dx, Q->in);
#else
        gemm_bwd(dy + (size_t)o0, Q->out, scratch, dx, T, n_tile, Q->in);
#endif
    }
    free(scratch);
}

void pl_q8_dequant_rows(const pl_q8 *Q, int o0, int n, float *dst)
{
    for (int p = 0; p < n; p++) {
        int o = o0 + p;
        const uint8_t *row = Q->blocks + (size_t)o * (size_t)Q->nb * 34;
        for (int g = 0; g < Q->nb; g++) {
            const uint8_t *blk = row + g * 34;
            uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            float d = ft_float_from_fp16(d_bits);
            for (int k = 0; k < 32; k++) {
                int idx = p * Q->in + g * 32 + k;
                dst[idx] = (int8_t)blk[2 + k] * d;
            }
        }
    }
}