/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Phase 2 quality gate for TurboQuant KV-cache compression.
 *
 * The unit tests (test_turboquant.c) prove the pipeline is correct and that
 * per-vector round-trip error is small.  The real question for OCR-grade
 * decoding is different: does compressing K and V change the ATTENTION OUTPUT?
 *
 * This standalone program:
 *   1. Generates realistic Q/K/V (head_dim=128, a few hundred tokens) under
 *      two distributions:
 *        - "gauss": standard normal per coordinate.
 *        - "heavy": heavy-tailed with a handful of high-variance outlier
 *          channels, to mimic real LLM activations / KV outliers.
 *   2. Computes causal scaled-dot-product attention with full-precision K/V.
 *   3. Computes the same attention with TurboQuant-compressed K/V (K4/V2 and
 *      K3/V2), decompressing each cached vector before use.
 *   4. Reports, aggregated over all query positions:
 *        - cosine similarity of the two attention outputs (mean, min),
 *        - max abs diff of the two outputs,
 *        - relative L2 error of the two outputs,
 *      plus the per-vector K/V round-trip MSE and relative error.
 *
 * Not wired into the Makefile / does not touch other files.  Build:
 *   clang -Wall -Wextra -std=c11 -O3 -march=native -ffast-math -I. \
 *     -o /tmp/validate_tq attention/tests/validate_tq_attention.c \
 *     attention/turboquant.c -lm
 */

#include "../turboquant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define HEAD_DIM   128
#define N_TOKENS   320

/* Deterministic RNG (splitmix64 -> uniform) so runs are reproducible. */
static uint64_t g_rng = 0x9e3779b97f4a7c15ULL;
static double urand(void) {
    g_rng += 0x9e3779b97f4a7c15ULL;
    uint64_t z = g_rng;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);   /* [0,1) */
}
static double gauss(void) {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* mode: 0 = gauss i.i.d.; 1 = heavy-tailed w/ outlier channels;
 *       2 = correlated low-rank (mimics real LLM activations: tokens live near
 *           a low-dim affine subspace, so the attention-weighted average has a
 *           strong coherent component and quantization noise is comparatively
 *           independent -> averaging can attenuate it). */
static void gen_matrix(float* m, int n, int d, int mode) {
    float chan_scale[HEAD_DIM];
    for (int c = 0; c < d; c++) chan_scale[c] = 1.0f;
    if (mode == 1) {
        int outliers[6] = { 3, 17, 41, 68, 95, 120 };
        float scales[6]  = { 8.0f, 6.0f, 10.0f, 5.0f, 7.0f, 9.0f };
        for (int i = 0; i < 6; i++)
            if (outliers[i] < d) chan_scale[outliers[i]] = scales[i];
    }

    if (mode == 2) {
        /* Rank-R shared subspace + shared mean; tokens are strongly correlated. */
        const int R = 8;
        float* basis = malloc(sizeof(float) * R * d);
        float* mu = malloc(sizeof(float) * d);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < d; c++) basis[r * d + c] = (float)gauss();
        for (int c = 0; c < d; c++) mu[c] = (float)(1.5 * gauss());
        for (int i = 0; i < n; i++) {
            float a[8];
            for (int r = 0; r < R; r++) a[r] = (float)gauss();
            for (int c = 0; c < d; c++) {
                double v = mu[c];
                for (int r = 0; r < R; r++) v += a[r] * basis[r * d + c] / sqrt((double)R);
                v += 0.25 * gauss();               /* small idiosyncratic noise */
                m[(size_t)i * d + c] = (float)v;
            }
        }
        free(basis); free(mu);
        return;
    }

    for (int i = 0; i < n; i++) {
        for (int c = 0; c < d; c++) {
            double x = gauss();
            if (mode == 1) {
                double g = 0.0;
                for (int k = 0; k < 3; k++) { double z = gauss(); g += z * z; }
                x *= sqrt(3.0 / (g + 1e-6));   /* Student-t-ish, df ~ 3 */
            }
            m[(size_t)i * d + c] = (float)(x * chan_scale[c]);
        }
    }
}

static double dotf(const float* a, const float* b, int d) {
    double s = 0.0;
    for (int i = 0; i < d; i++) s += (double)a[i] * b[i];
    return s;
}

/* Causal single-head attention. out[i*d + :] = sum_j softmax(q_i.k_j/sqrt d) v_j
 * over j <= i.  K and V are provided as callbacks so we can feed either full
 * precision or dequantized vectors. */
static void attention(const float* Q, const float* K, const float* V,
                      int n, int d, float* out) {
    double scale = 1.0 / sqrt((double)d);
    double* scores = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        double mx = -1e300;
        for (int j = 0; j <= i; j++) {
            scores[j] = dotf(Q + (size_t)i * d, K + (size_t)j * d, d) * scale;
            if (scores[j] > mx) mx = scores[j];
        }
        double sum = 0.0;
        for (int j = 0; j <= i; j++) { scores[j] = exp(scores[j] - mx); sum += scores[j]; }
        double inv = 1.0 / sum;
        float* o = out + (size_t)i * d;
        for (int c = 0; c < d; c++) o[c] = 0.0f;
        for (int j = 0; j <= i; j++) {
            double w = scores[j] * inv;
            const float* v = V + (size_t)j * d;
            for (int c = 0; c < d; c++) o[c] += (float)(w * v[c]);
        }
    }
    free(scores);
}

static double cosine(const float* a, const float* b, int d) {
    double na = sqrt(dotf(a, a, d)), nb = sqrt(dotf(b, b, d));
    if (na < 1e-20 || nb < 1e-20) return 1.0;
    return dotf(a, b, d) / (na * nb);
}

/* Round-trip a full K/V matrix through TurboQuant, filling dq[n*d].
 * Accumulates per-vector MSE and relative L2 error. */
static void roundtrip(fa_tq_state* st, const float* src, int n, int d,
                      int is_key, float* dq,
                      double* out_mse, double* out_rel) {
    size_t bph = is_key ? fa_tq_key_bytes_per_head(st)
                        : fa_tq_value_bytes_per_head(st);
    uint8_t* packed = malloc(bph);
    float range[2];
    double mse = 0.0, rel = 0.0;
    for (int i = 0; i < n; i++) {
        const float* s = src + (size_t)i * d;
        float* o = dq + (size_t)i * d;
        fa_tq_quantize_head(st, s, is_key ? true : false, packed, range);
        fa_tq_dequantize_head(st, packed, range, is_key ? true : false, o);
        double e2 = 0.0, n2 = 0.0;
        for (int c = 0; c < d; c++) {
            double e = (double)o[c] - s[c];
            e2 += e * e; n2 += (double)s[c] * s[c];
        }
        mse += e2 / d;
        rel += (n2 > 1e-20) ? sqrt(e2 / n2) : 0.0;
    }
    *out_mse = mse / n;
    *out_rel = rel / n;
    free(packed);
}

typedef struct {
    double kmse, krel, vmse, vrel;
    double cos_mean, cos_min, maxabs, relL2;   /* over all positions i>=1 */
    double cos_tail, relL2_tail;               /* last 32 positions (long ctx) */
} result;

static result run_preset(const char* dist, int mode,
                         int key_bits, int value_bits) {
    int n = N_TOKENS, d = HEAD_DIM;
    float* Q = malloc(sizeof(float) * n * d);
    float* K = malloc(sizeof(float) * n * d);
    float* V = malloc(sizeof(float) * n * d);
    float* Kdq = malloc(sizeof(float) * n * d);
    float* Vdq = malloc(sizeof(float) * n * d);
    float* out_fp = malloc(sizeof(float) * n * d);
    float* out_q  = malloc(sizeof(float) * n * d);

    /* Reset RNG per preset so Q/K/V are identical across bit configs. */
    g_rng = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)mode * 0xabcdefULL + 0x123456ULL);
    gen_matrix(Q, n, d, mode);
    gen_matrix(K, n, d, mode);
    gen_matrix(V, n, d, mode);

    fa_tq_config cfg = { key_bits, value_bits };
    fa_tq_state* st = fa_tq_init(d, cfg, 0xC0FFEEULL);

    result r;
    roundtrip(st, K, n, d, 1, Kdq, &r.kmse, &r.krel);
    roundtrip(st, V, n, d, 0, Vdq, &r.vmse, &r.vrel);

    attention(Q, K,   V,   n, d, out_fp);
    attention(Q, Kdq, Vdq, n, d, out_q);

    /* Aggregate over query positions (skip i=0: trivially exact).  Also track
     * the long-context regime (last 32 positions), which is the regime that
     * actually matters for a KV cache: you compress *because* context is long,
     * and those queries average over many V vectors. */
    int tail_start = n - 32; if (tail_start < 1) tail_start = 1;
    double cs = 0.0, cmin = 1.0, maxabs = 0.0, num = 0.0, den = 0.0;
    double cs_t = 0.0, num_t = 0.0, den_t = 0.0;
    int cnt = 0, cnt_t = 0;
    for (int i = 1; i < n; i++) {
        double c = cosine(out_fp + (size_t)i * d, out_q + (size_t)i * d, d);
        cs += c; if (c < cmin) cmin = c; cnt++;
        double dnum = 0.0, dden = 0.0;
        for (int k = 0; k < d; k++) {
            double diff = fabs((double)out_fp[(size_t)i*d+k] - out_q[(size_t)i*d+k]);
            if (diff > maxabs) maxabs = diff;
            dnum += diff * diff;
            dden += (double)out_fp[(size_t)i*d+k] * out_fp[(size_t)i*d+k];
        }
        num += dnum; den += dden;
        if (i >= tail_start) { cs_t += c; num_t += dnum; den_t += dden; cnt_t++; }
    }
    r.cos_mean = cs / cnt;
    r.cos_min = cmin;
    r.maxabs = maxabs;
    r.relL2 = sqrt(num / den);
    r.cos_tail = cs_t / cnt_t;
    r.relL2_tail = sqrt(num_t / den_t);

    printf("  [%-5s K%d/V%d]  Kmse=%.3e Krel=%.4f  Vmse=%.3e Vrel=%.4f\n",
           dist, key_bits, value_bits, r.kmse, r.krel, r.vmse, r.vrel);
    printf("             attn(all)  cos mean=%.6f min=%.6f  maxabs=%.4e  relL2=%.4e\n",
           r.cos_mean, r.cos_min, r.maxabs, r.relL2);
    printf("             attn(tail) cos=%.6f  relL2=%.4e   (last 32 positions, long-ctx regime)\n",
           r.cos_tail, r.relL2_tail);

    fa_tq_free(st);
    free(Q); free(K); free(V); free(Kdq); free(Vdq); free(out_fp); free(out_q);
    return r;
}

int main(void) {
    printf("=== TurboQuant attention-quality validation (head_dim=%d, tokens=%d, causal) ===\n\n",
           HEAD_DIM, N_TOKENS);

    printf("Gaussian K/V:\n");
    result g42 = run_preset("gauss", 0, 4, 2);
    result g32 = run_preset("gauss", 0, 3, 2);
    printf("\nHeavy-tailed K/V (outlier channels, Student-t-ish tails):\n");
    result h42 = run_preset("heavy", 1, 4, 2);
    result h32 = run_preset("heavy", 1, 3, 2);

    printf("\nCorrelated low-rank K/V (mimics real LLM activation structure):\n");
    result c42 = run_preset("corr", 2, 4, 2);
    result c32 = run_preset("corr", 2, 3, 2);

    printf("\nDiagnostic: value-bit sweep on Gaussian (shows V bits are the driver):\n");
    result g44 = run_preset("gauss", 0, 4, 4);
    result g43 = run_preset("gauss", 0, 4, 3);
    (void)g44; (void)g43;

    printf("\n--- MACHINE ---\n");
    printf("gauss_K4V2 kmse=%.6e krel=%.6f vmse=%.6e vrel=%.6f cos=%.6f cosmin=%.6f maxabs=%.6e relL2=%.6e\n",
           g42.kmse, g42.krel, g42.vmse, g42.vrel, g42.cos_mean, g42.cos_min, g42.maxabs, g42.relL2);
    printf("gauss_K3V2 kmse=%.6e krel=%.6f vmse=%.6e vrel=%.6f cos=%.6f cosmin=%.6f maxabs=%.6e relL2=%.6e\n",
           g32.kmse, g32.krel, g32.vmse, g32.vrel, g32.cos_mean, g32.cos_min, g32.maxabs, g32.relL2);
    printf("heavy_K4V2 kmse=%.6e krel=%.6f vmse=%.6e vrel=%.6f cos=%.6f cosmin=%.6f maxabs=%.6e relL2=%.6e\n",
           h42.kmse, h42.krel, h42.vmse, h42.vrel, h42.cos_mean, h42.cos_min, h42.maxabs, h42.relL2);
    printf("heavy_K3V2 kmse=%.6e krel=%.6f vmse=%.6e vrel=%.6f cos=%.6f cosmin=%.6f maxabs=%.6e relL2=%.6e\n",
           h32.kmse, h32.krel, h32.vmse, h32.vrel, h32.cos_mean, h32.cos_min, h32.maxabs, h32.relL2);
    printf("corr_K4V2  kmse=%.6e krel=%.6f vmse=%.6e vrel=%.6f cos=%.6f cosmin=%.6f maxabs=%.6e relL2=%.6e\n",
           c42.kmse, c42.krel, c42.vmse, c42.vrel, c42.cos_mean, c42.cos_min, c42.maxabs, c42.relL2);
    printf("corr_K3V2  kmse=%.6e krel=%.6f vmse=%.6e vrel=%.6f cos=%.6f cosmin=%.6f maxabs=%.6e relL2=%.6e\n",
           c32.kmse, c32.krel, c32.vmse, c32.vrel, c32.cos_mean, c32.cos_min, c32.maxabs, c32.relL2);
    return 0;
}
