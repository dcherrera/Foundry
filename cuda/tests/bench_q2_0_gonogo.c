/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/*
 * bench_q2_0_gonogo.c — Phase 1 GO/NO-GO for the Bonsai native-ternary
 * (GGUF type 42, Q2_0_g128) sm_37 matvec.
 *
 * Two questions, decisively:
 *   1. BIT-EXACT: fc_matvec_q2_0 (GPU, repacked) == a CPU replica that mirrors
 *      the kernel's exact fp op order (per-lane 4-wide (code-1)*x chain, one
 *      d*group scale-FMA per 128-block, then the 5-round __shfl_down tree).
 *   2. SPEED A/B: fc_matvec_q2_0 GB/s (effective, device weight bytes) + us vs
 *      fc_matvec_q3k on the SAME Bonsai shapes @ 875 MHz, >= REPS times each.
 *
 * The Kepler thesis: Q3_K runs ~30 GB/s effective
 * because the dequant int-ops (BFE/shift/cvt) starve the memory pipe. Ternary
 * add/sub with no dequant should run nearer the ~150 GB/s GEMV ceiling. GO if
 * bit-exact AND ternary clears Q3_K by a clear margin; NO-GO if bit-exact but
 * it does NOT beat Q3_K (the card is also launch/issue-bound).
 *
 * Build (from foundry/, on a K80 host):
 *   gcc -O2 -I. -o cuda/tests/bench_q2_0_gonogo cuda/tests/bench_q2_0_gonogo.c \
 *       cuda/fc_cuda.c -lm -ldl
 * Run (FREE die only): CUDA_VISIBLE_DEVICES=0 ./cuda/tests/bench_q2_0_gonogo
 */

#include "cuda/fc_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REPS 3

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float half2f(unsigned short h)
{
    unsigned int sign = ((unsigned int)(h & 0x8000)) << 16;
    int exp = (h >> 10) & 0x1F;
    unsigned int mant = h & 0x3FF;
    unsigned int bits;
    float out;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            exp++; mant &= ~0x400u;
            bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13);
    }
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float tree32(float v[32])
{
    float t[32];
    int off, l;
    for (off = 16; off >= 1; off >>= 1) {
        for (l = 0; l < 32; l++)
            t[l] = v[l] + ((l + off < 32) ? v[l + off] : v[l]);
        memcpy(v, t, sizeof(t));
    }
    return v[0];
}

/* -------- byte generators (raw GGUF Q2_0 = 34B block: [fp16 d][32 code B]) -- */
static size_t q2_raw_stride(int cols) { return (size_t)(cols / 128) * 34; }
static size_t q3_raw_stride(int cols) { return (size_t)(cols / 256) * 110; }

/* Random ternary codes {0,1,2} + a sane fp16 group scale (exp 2^-4..2^-3). */
static unsigned char *q2_rand_row(long rows, int cols)
{
    size_t stride = q2_raw_stride(cols);
    unsigned char *p = malloc(stride * (size_t)rows);
    long r; int b, nb = cols / 128, i, k;
    for (r = 0; r < rows; r++) {
        for (b = 0; b < nb; b++) {
            unsigned char *blk = p + ((size_t)r * nb + b) * 34;
            unsigned short h = (unsigned short)(0x2C00 | (rand() & 0x3FF));
            blk[0] = h & 0xFF; blk[1] = h >> 8;
            for (i = 0; i < 32; i++) {
                unsigned char byte = 0;
                for (k = 0; k < 4; k++) byte |= (unsigned char)((rand() % 3) << (k * 2));
                blk[2 + i] = byte;
            }
        }
    }
    return p;
}

/* Q3_K random bytes with sanitized fp16 super-block scale (byte 108/109). */
static unsigned char *q3_rand_row(long rows, int cols)
{
    size_t stride = q3_raw_stride(cols);
    unsigned char *p = malloc(stride * (size_t)rows);
    size_t i;
    for (i = 0; i < stride * (size_t)rows; i++) p[i] = (unsigned char)(rand() & 0xFF);
    /* sanitize each 110B block's fp16 d */
    { size_t off; for (off = 0; off + 110 <= stride * (size_t)rows; off += 110) {
        unsigned short h = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        p[off + 108] = h & 0xFF; p[off + 109] = h >> 8;
    } }
    return p;
}

static float *rand_floats(size_t n)
{
    float *p = malloc(n * sizeof(float));
    size_t i;
    for (i = 0; i < n; i++) p[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    return p;
}

/* CPU replica of ONE fc_matvec_q2_0 output row in the kernel's exact op order:
 * per lane l, for each 128-block b: read code byte l (elements 4l..4l+3),
 * g = sum_{k=0..3} (code_k - 1) * x[b*128 + 4l + k]  (ascending, plain +),
 * partial_l += d_b * g; then tree32 over the 32 lanes. */
/* fp16 round-trip of a positive scale (matches the GPU fc_fp16_roundtrip_pos
 * and the CPU pn_float_to_half/pn_half_to_float). */
static float f2half_rt_pos(float f)
{
    unsigned int x; memcpy(&x, &f, 4);
    int exp = (int)((x >> 23) & 0xFF) - 127 + 15;
    unsigned int mant = x & 0x7FFFFFu;
    unsigned short h;
    if (((x >> 23) & 0xFF) == 0xFF) { h = 0x7C00u | (mant ? 0x0200u : 0); }
    else if (exp >= 0x1F) { h = 0x7C00u; }
    else if (exp <= 0) {
        if (exp < -10) { h = 0; }
        else { mant |= 0x800000u; int sh = 14 - exp;
               unsigned int hm = mant >> sh, rem = mant & ((1u<<sh)-1), half = 1u<<(sh-1);
               if (rem > half || (rem == half && (hm & 1))) hm++; h = (unsigned short)hm; }
    } else {
        unsigned int hm = mant >> 13, rem = mant & 0x1FFF;
        if (rem > 0x1000u || (rem == 0x1000u && (hm & 1))) { hm++;
            if (hm == 0x400u) { hm = 0; exp++; if (exp >= 0x1F) return 1.0f/0.0f; } }
        h = (unsigned short)(((unsigned int)exp << 10) | hm);
    }
    return half2f(h);
}

/* CPU replica of ONE fc_matvec_q2_0 output row in the FIXED kernel's op order:
 * activation Q8_0-quantized per 32-elem sub-block (sub-block s = lane/8, amax
 * over its 8 lanes), q = roundf(x/d_pre), d_act = fp16(d_pre); per lane
 * partial += d_w * d_act * sum_k (code_k-1)*q_k; then tree32 over 32 lanes. */
static float ref_row_q2_0(const unsigned char *row, const float *x, int cols)
{
    float p[32] = { 0 };
    int nb = cols / 128, b, lane, k;
    for (b = 0; b < nb; b++) {
        const unsigned char *blk = row + (size_t)b * 34;
        float d_w = half2f((unsigned short)(blk[0] | (blk[1] << 8)));
        const unsigned char *qs = blk + 2;
        const float *xb = x + (size_t)b * 128;
        /* per-sub-block (32 elems = 8 lanes) amax + d_act */
        float d_act[4], inv[4];
        for (int sb = 0; sb < 4; sb++) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) { float a = (float)fabs((double)xb[sb*32 + j]); if (a > amax) amax = a; }
            float d_pre = amax / 127.0f;
            inv[sb]   = d_pre != 0.0f ? 1.0f / d_pre : 0.0f;
            d_act[sb] = f2half_rt_pos(d_pre);
        }
        for (lane = 0; lane < 32; lane++) {
            int sb = lane / 8;
            unsigned char cbyte = qs[lane];
            int is = 0;
            for (k = 0; k < 4; k++) {
                int code = (cbyte >> (k * 2)) & 3;
                int q = (int)roundf(xb[lane * 4 + k] * inv[sb]);
                is += (code - 1) * q;
            }
            p[lane] += d_w * d_act[sb] * (float)is;
        }
    }
    return tree32(p);
}

static int ulp_diff(float a, float b)
{
    int32_t ia, ib; int64_t d;
    memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
    d = (int64_t)ia - ib; if (d < 0) d = -d;
    return (d > 1000000) ? 1000000 : (int)d;
}

/* Upload + repack a Q2_0 raw row; return device buffer, write device stride. */
static fc_buffer *q2_upload(const unsigned char *raw, long rows, int cols, int *ds)
{
    size_t rs = fc_repack_row_bytes(42, cols);
    unsigned char *tmp = malloc(rs * (size_t)rows);
    fc_buffer *b;
    if (!tmp || fc_repack_weight(42, raw, tmp, rows, cols) != 0) { free(tmp); return NULL; }
    b = fc_buffer_from_ptr(tmp, rs * (size_t)rows);
    free(tmp); *ds = (int)rs; return b;
}
static fc_buffer *q3_upload(const unsigned char *raw, long rows, int cols, int *ds)
{
    size_t rs = fc_repack_row_bytes(11, cols);
    unsigned char *tmp = malloc(rs * (size_t)rows);
    fc_buffer *b;
    if (!tmp || fc_repack_weight(11, raw, tmp, rows, cols) != 0) { free(tmp); return NULL; }
    b = fc_buffer_from_ptr(tmp, rs * (size_t)rows);
    free(tmp); *ds = (int)rs; return b;
}

static int verify_q2(fc_cuda_ctx *ctx, int rows, int cols)
{
    unsigned char *raw = q2_rand_row(rows, cols);
    float *xh = rand_floats((size_t)cols);
    float *yh = malloc((size_t)rows * sizeof(float));
    int ds = 0, r, nmis = 0, maxulp = 0;
    fc_buffer *w = q2_upload(raw, rows, cols, &ds);
    fc_buffer *x = fc_buffer_from_ptr(xh, (size_t)cols * sizeof(float));
    fc_buffer *y = fc_buffer_alloc((size_t)rows * sizeof(float));
    size_t raw_stride = q2_raw_stride(cols);
    fc_matvec_q2_0_ctx(ctx, w, x, y, rows, cols, ds);
    fc_sync();
    fc_buffer_to_cpu(y, yh);
    for (r = 0; r < rows; r++) {
        float want = ref_row_q2_0(raw + (size_t)r * raw_stride, xh, cols);
        if (memcmp(&want, &yh[r], 4) != 0) {
            int u = ulp_diff(want, yh[r]);
            if (u > maxulp) maxulp = u;
            if (nmis < 3)
                printf("    row %d: ref %.9g (%08x) gpu %.9g (%08x)\n",
                       r, (double)want, *(unsigned *)&want,
                       (double)yh[r], *(unsigned *)&yh[r]);
            nmis++;
        }
    }
    printf("verify q2_0  %6dx%-6d  %s (mismatches %d, max ulp %d)\n",
           rows, cols, nmis ? "FAIL" : "BIT-EXACT", nmis, maxulp);
    fc_buffer_free(w); fc_buffer_free(x); fc_buffer_free(y);
    free(raw); free(xh); free(yh);
    return nmis ? -1 : 0;
}

/* Time one kernel; returns GB/s of device weight bytes (min over REPS -> peak). */
static void bench_pair(fc_cuda_ctx *ctx, const char *tag, int rows, int cols, int iters)
{
    unsigned char *q2raw = q2_rand_row(rows, cols);
    unsigned char *q3raw = q3_rand_row(rows, cols);
    int ds2 = 0, ds3 = 0, rep, i;
    fc_buffer *w2 = q2_upload(q2raw, rows, cols, &ds2);
    fc_buffer *w3 = q3_upload(q3raw, rows, cols, &ds3);
    float *xh = rand_floats((size_t)cols);
    fc_buffer *x = fc_buffer_from_ptr(xh, (size_t)cols * sizeof(float));
    fc_buffer *y = fc_buffer_alloc((size_t)rows * sizeof(float));
    double best2 = 0, best3 = 0;
    free(q2raw); free(q3raw); free(xh);

    for (rep = 0; rep < REPS; rep++) {
        double t0, dt2, dt3, gb2, gb3;
        /* q2_0 */
        fc_matvec_q2_0_ctx(ctx, w2, x, y, rows, cols, ds2); fc_sync();
        t0 = now_s();
        for (i = 0; i < iters; i++) fc_matvec_q2_0_ctx(ctx, w2, x, y, rows, cols, ds2);
        fc_sync(); dt2 = (now_s() - t0) / iters;
        gb2 = (double)rows * ds2 / dt2 / 1e9;
        if (gb2 > best2) best2 = gb2;
        /* q3k */
        fc_matvec_q3k_ctx(ctx, w3, x, y, rows, cols, ds3); fc_sync();
        t0 = now_s();
        for (i = 0; i < iters; i++) fc_matvec_q3k_ctx(ctx, w3, x, y, rows, cols, ds3);
        fc_sync(); dt3 = (now_s() - t0) / iters;
        gb3 = (double)rows * ds3 / dt3 / 1e9;
        if (gb3 > best3) best3 = gb3;
        printf("  %-18s rep %d  q2_0 %8.3f us %7.2f GB/s | q3k %8.3f us %7.2f GB/s\n",
               tag, rep, dt2 * 1e6, gb2, dt3 * 1e6, gb3);
    }
    printf("  %-18s PEAK  q2_0 %7.2f GB/s | q3k %7.2f GB/s | ratio %.2fx  (%dx%d)\n",
           tag, best2, best3, best3 > 0 ? best2 / best3 : 0.0, rows, cols);
    fc_buffer_free(w2); fc_buffer_free(w3);
    fc_buffer_free(x); fc_buffer_free(y);
}

int main(void)
{
    fc_cuda_ctx *ctx;
    int it = 100, fails = 0;

    if (fc_init() != 0) { fprintf(stderr, "no CUDA\n"); return 1; }
    ctx = fc_default_ctx();
    srand(42);

    printf("== BIT-EXACT: fc_matvec_q2_0 (GPU) vs CPU replica ==\n");
    fails += verify_q2(ctx, 512, 5120);
    fails += verify_q2(ctx, 512, 17408);
    fails += verify_q2(ctx, 5120, 5120);
    fails += verify_q2(ctx, 640, 6144);
    if (fails) { printf("BIT-EXACT FAILED -> NO-GO on correctness. Stop.\n"); fc_free(); return 1; }
    printf("BIT-EXACT: PASS\n\n");

    printf("== SPEED A/B: q2_0 vs q3k, same shapes, %d reps, %d iters ==\n", REPS, it);
    /* Representative Bonsai matmul shapes (rows x cols). */
    bench_pair(ctx, "attn proj",    5120, 5120,  it);
    bench_pair(ctx, "ffn gate/up", 17408, 5120,  it);
    bench_pair(ctx, "ffn down",     5120, 17408, it);
    bench_pair(ctx, "lm head",     32768, 5120,  10);

    fc_free();
    return 0;
}
