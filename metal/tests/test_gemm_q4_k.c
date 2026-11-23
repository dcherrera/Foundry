/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/* Validate + benchmark the Q4_K x B GEMM (fm_sgemm_q4_k) against a CPU
 * reference that dequantizes the same super-block bytes. Since both sides read
 * identical Q4_K bytes, the weight need not be "meaningfully" quantized — only
 * structurally valid — to check the GPU kernel matches the dequant math.
 *
 * Usage: test_gemm_q4_k [rows] [cols] [B]
 */

#include "metal/fm_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* Minimal fp32 -> fp16 (round to nearest even, no NaN/Inf handling needed for
 * the small magnitudes we synthesize). */
static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t out;
    if (exp == 0) { if (mant == 0) { out = sign; } else {
        exp = 127 - 15 + 1; while (!(mant & 0x400)) { mant <<= 1; exp--; }
        mant &= 0x3FF; out = sign | (exp << 23) | (mant << 13); } }
    else if (exp == 31) out = sign | 0x7F800000u | (mant << 13);
    else out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f; memcpy(&f, &out, 4); return f;
}

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static uint32_t rng = 0x1234567u;
static uint32_t urand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float frand(void) { return (float)(urand() % 2000) / 1000.0f - 1.0f; }

/* Q4_K super-block = 144 bytes: half d, half dmin, uchar scales[12], uchar qs[128]. */
#define QK_BYTES 144u

/* Dequantize block b of row into out[256], mirroring the kernel. */
static void deq_block(const uint8_t *blk, float *out) {
    float d    = f16_to_f32(((const uint16_t *)blk)[0]);
    float dmin = f16_to_f32(((const uint16_t *)blk)[1]);
    const uint8_t *sr = blk + 4, *qs = blk + 16;
    uint8_t sc[8], mn[8];
    for (int j = 0; j < 4; j++) { sc[j] = sr[j] & 63; mn[j] = sr[j+4] & 63; }
    for (int j = 4; j < 8; j++) {
        sc[j] = (sr[j+4] & 0xF) | ((sr[j-4] >> 6) << 4);
        mn[j] = (sr[j+4] >> 4)  | ((sr[j]   >> 6) << 4);
    }
    for (int p = 0; p < 4; p++) {
        float d1 = d * sc[2*p], m1 = dmin * mn[2*p];
        float d2 = d * sc[2*p+1], m2 = dmin * mn[2*p+1];
        const uint8_t *q = qs + p * 32;
        int o = p * 64;
        for (int l = 0; l < 32; l++) {
            out[o + l]      = d1 * (q[l] & 0xF) - m1;
            out[o + 32 + l] = d2 * (q[l] >> 4)  - m2;
        }
    }
}

int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 2048;
    int cols = argc > 2 ? atoi(argv[2]) : 1024;
    int B    = argc > 3 ? atoi(argv[3]) : 16;
    if (cols % 256) { fprintf(stderr, "cols must be a multiple of 256\n"); return 1; }

    if (fm_init() != 0 || !fm_available()) { fprintf(stderr, "Metal unavailable\n"); return 77; }

    int nblk = cols / 256;
    size_t row_stride = (size_t)nblk * QK_BYTES;
    size_t wbytes = (size_t)rows * row_stride;
    uint8_t *W = malloc(wbytes);
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < nblk; b++) {
            uint8_t *blk = W + (size_t)r * row_stride + (size_t)b * QK_BYTES;
            ((uint16_t *)blk)[0] = f32_to_f16(0.02f + 0.01f * (float)(urand() % 8));
            ((uint16_t *)blk)[1] = f32_to_f16(0.01f * (float)(urand() % 8));
            for (int i = 4; i < 144; i++) blk[i] = (uint8_t)urand();
        }
    }
    float *X = malloc((size_t)B * cols * sizeof(float));
    for (int i = 0; i < B * cols; i++) X[i] = frand();

    /* CPU reference: out_ref[b, r] = sum_c dequant(W[r]) * X[b]. */
    float *ref = malloc((size_t)B * rows * sizeof(float));
    double c0 = now_ms();
    float blk[256];
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < nblk; b++) {
            deq_block(W + (size_t)r * row_stride + (size_t)b * QK_BYTES, blk);
            for (int s = 0; s < B; s++) {
                float acc = (b == 0) ? 0.0f : ref[(size_t)s * rows + r];
                const float *xb = X + (size_t)s * cols + b * 256;
                for (int l = 0; l < 256; l++) acc += blk[l] * xb[l];
                ref[(size_t)s * rows + r] = acc;
            }
        }
    }
    double cpu_ms = now_ms() - c0;

    /* GPU. */
    fm_buffer *bW = fm_buffer_from_ptr(W, wbytes);
    fm_buffer *bX = fm_buffer_from_ptr(X, (size_t)B * cols * sizeof(float));
    fm_buffer *bO = fm_buffer_alloc((size_t)B * rows * sizeof(float));
    /* warm up (compile pipeline + page in) */
    fm_sgemm_q4_k(bW, bX, bO, rows, cols, (int)row_stride, B); fm_sync();

    int iters = 50;
    double g0 = now_ms();
    for (int i = 0; i < iters; i++) { fm_sgemm_q4_k(bW, bX, bO, rows, cols, (int)row_stride, B); }
    fm_sync();
    double gpu_ms = (now_ms() - g0) / iters;

    float *out = fm_buffer_data(bO);
    double maxabs = 0, sumsq = 0;
    for (int i = 0; i < B * rows; i++) {
        double e = fabs((double)out[i] - ref[i]);
        if (e > maxabs) maxabs = e;
        sumsq += (double)ref[i] * ref[i];
    }
    double rms = sqrt(sumsq / (B * rows));
    double rel_to_scale = maxabs / (rms + 1e-12);   /* err vs typical magnitude */

    printf("gemm_q4_k  rows=%d cols=%d B=%d\n", rows, cols, B);
    printf("  correctness: max abs err = %.4g, output RMS = %.4g, err/RMS = %.4g -> %s\n",
           maxabs, rms, rel_to_scale, (rel_to_scale < 1e-3) ? "PASS" : "FAIL");
    printf("  GPU: %.3f ms/call   (CPU ref build: %.1f ms, unoptimized)\n", gpu_ms, cpu_ms);

    fm_buffer_free(bW); fm_buffer_free(bX); fm_buffer_free(bO);
    free(W); free(X); free(ref);
    return (rel_to_scale < 1e-3) ? 0 : 1;
}
