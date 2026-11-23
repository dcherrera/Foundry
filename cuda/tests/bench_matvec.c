/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/*
 * bench_matvec.c — micro-benchmark + bit-exact verifier for the Foundry quant
 * matvec + grouped MoE kernels on one K80 die. No model load: weights are
 * random bytes (fp16 block scales clamped to a sane exponent so sums stay
 * finite), shapes mirror the Nex-N2-mini hot paths. Reports ms/launch and
 * effective GB/s of weight-bytes so kernel variants can be compared in
 * seconds instead of a 100 s model load per iteration.
 *
 * VERIFY: for each quant type a CPU replica recomputes the kernel output with
 * the exact device op order (per-lane fmaf accumulation chains + the 5-round
 * __shfl_down tree, mirroring the PTX: q3k/q6k partial = fma(acc, d, partial);
 * q5k term = (d*sc)*nib - dmin*mn, partial += acc; q80 partial =
 * fma(x, d*q, partial)). The kernel result must match BIT-EXACTLY — this is
 * how the upload-time repack (fc_repack_weight) is proven numerically
 * identical to the original GGUF-layout path.
 *
 * fc_matvec_q4k is the exception: it is MIN-LIFTED (see matvec_kquants.cu),
 * which reassociates the sum, so it cannot be bit-identical to the previous
 * term order. verify_mv_reassoc gates it BIT-EXACTLY against a replica of
 * its new order (ref_row_q4k_minlift) and additionally bounds its fp32
 * relative error against an fp64 reference, both absolutely and relative to
 * the old order's error. The other quants keep the zero-tolerance memcmp.
 *
 * Build (from foundry/):
 *   gcc -O2 -I. -o cuda/tests/bench_matvec cuda/tests/bench_matvec.c \
 *       cuda/fc_cuda.c -lm -ldl
 *   -DUSE_REPACK=0 benches/verifies the raw-GGUF-layout kernels (the
 *   pre-repack ones); default 1 repacks via fc_repack_weight first.
 * Run: CUDA_VISIBLE_DEVICES=0 ./cuda/tests/bench_matvec
 */

#include "cuda/fc_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef USE_REPACK
#define USE_REPACK 1
#endif

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------
 * Random weight/activation generation
 * ------------------------------------------------------------------ */

static size_t raw_block_bytes(int t)
{
    switch (t) {
    case 8:  return 34;
    case 11: return 110;
    case 12: return 144;
    case 13: return 176;
    case 14: return 210;
    default: return 0;
    }
}

/* Clamp every fp16 block scale (d, and dmin for Q4_K/Q5_K) to exponent
 * 2^-4..2^-3 with a random mantissa: random halves can be inf/nan which
 * breaks the bit-exact compare (nan != nan). Positions per gguf_dequant.h. */
static void sanitize_quant(int t, unsigned char *p, size_t bytes)
{
    size_t bs = raw_block_bytes(t);
    size_t off;
    for (off = 0; bs && off + bs <= bytes; off += bs) {
        unsigned char *b = p + off;
        unsigned short h = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        unsigned short h2 = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        switch (t) {
        case 11: b[108] = h & 0xFF; b[109] = h >> 8; break;
        case 14: b[208] = h & 0xFF; b[209] = h >> 8; break;
        case 12: case 13:
            b[0] = h & 0xFF;  b[1] = h >> 8;
            b[2] = h2 & 0xFF; b[3] = h2 >> 8;
            break;
        case 8:  b[0] = h & 0xFF;  b[1] = h >> 8; break;
        }
    }
}

static unsigned char *rand_bytes(size_t bytes)
{
    unsigned char *p = malloc(bytes);
    size_t i;
    for (i = 0; i < bytes; i++)
        p[i] = (unsigned char)(rand() & 0xFF);
    return p;
}

static float *rand_floats(size_t n)
{
    float *p = malloc(n * sizeof(float));
    size_t i;
    for (i = 0; i < n; i++)
        p[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    return p;
}

/* Upload a raw quant weight for the kernels: repacked (default) or as-is.
 * Returns the device buffer and writes the device row stride. */
static fc_buffer *upload_weight(int t, const unsigned char *raw, long rows,
                                int cols, int *dev_stride)
{
    size_t raw_stride = raw_block_bytes(t) * (size_t)((t == 8) ? cols / 32
                                                               : cols / 256);
#if USE_REPACK
    size_t rs = fc_repack_row_bytes(t, cols);
    unsigned char *tmp = malloc(rs * (size_t)rows);
    fc_buffer *b;
    if (!tmp || fc_repack_weight(t, raw, tmp, rows, cols) != 0) {
        free(tmp);
        return NULL;
    }
    b = fc_buffer_from_ptr(tmp, rs * (size_t)rows);
    free(tmp);
    *dev_stride = (int)rs;
    return b;
#else
    *dev_stride = (int)raw_stride;
    return fc_buffer_from_ptr(raw, raw_stride * (size_t)rows);
#endif
}

/* ------------------------------------------------------------------
 * CPU replica — decodes the ORIGINAL GGUF layout (source of truth for
 * both kernel generations) with the device's exact fp op order.
 * ------------------------------------------------------------------ */

static float half2f(unsigned short h)
{
    unsigned int sign = ((unsigned int)(h & 0x8000)) << 16;
    int exp = (h >> 10) & 0x1F;
    unsigned int mant = h & 0x3FF;
    unsigned int bits;
    float out;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            exp++;
            mant &= ~0x400u;
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

/* 5-round __shfl_down(16,8,4,2,1) + add tree over the 32 lane partials
 * (out-of-range shfl returns the caller's own value). Result = lane 0. */
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

static void kq_scale_min(const unsigned char *sr, int s, float *sc, float *mn)
{
    if (s < 4) {
        *sc = (float)(sr[s] & 63);
        *mn = (float)(sr[s + 4] & 63);
    } else {
        *sc = (float)((sr[s + 4] & 0xF) | ((sr[s - 4] >> 6) << 4));
        *mn = (float)((sr[s + 4] >> 4)  | ((sr[s]     >> 6) << 4));
    }
}

/* One output row of each kernel, replicated per-lane. The _lanes variants
 * accumulate the 32 per-lane fmaf chains into p[] (caller zeroes) without
 * the final shfl tree — the grouped-MoE replicas below need the raw lane
 * partials to reproduce the down kernels' per-slot weighted chains. */
static void ref_lanes_q3k(const unsigned char *row, const float *x, int cols,
                          float p[32])
{
    int n_sb = cols >> 8, sb, lane, n, j;
    for (sb = 0; sb < n_sb; sb++) {
        const unsigned char *blk = row + (size_t)sb * 110;
        const unsigned char *hm = blk, *qs = blk + 32, *sc = blk + 96;
        float d = half2f((unsigned short)(blk[108] | (blk[109] << 8)));
        const float *xb = x + (sb << 8);
        for (lane = 0; lane < 32; lane++) {
            int lhalf = lane >> 4;
            float acc = 0.0f;
            for (n = 0; n < 2; n++) {
                unsigned char qb = qs[32 * n + lane];
                for (j = 0; j < 4; j++) {
                    int s = 8 * n + 2 * j + lhalf;
                    int i = s & 3, a = s >> 2;
                    unsigned char lowb = (a & 1) ? sc[i + 4] : sc[i];
                    int low = (a & 2) ? (lowb >> 4) : (lowb & 0xF);
                    int high = (sc[8 + i] >> (2 * a)) & 3;
                    int scale = (low | (high << 4)) - 32;
                    int qv = (qb >> (2 * j)) & 3;
                    int hb = (hm[lane] & (1u << (4 * n + j))) ? 0 : 4;
                    acc = fmaf(xb[128 * n + 32 * j + lane],
                               (float)(scale * (qv - hb)), acc);
                }
            }
            p[lane] = fmaf(acc, d, p[lane]);
        }
    }
}

static float ref_row_q3k(const unsigned char *row, const float *x, int cols)
{
    float p[32] = { 0 };
    ref_lanes_q3k(row, x, cols, p);
    return tree32(p);
}

static void ref_lanes_q4k(const unsigned char *row, const float *x, int cols,
                          float p[32])
{
    int n_sb = cols >> 8, sb, lane, j, h;
    for (sb = 0; sb < n_sb; sb++) {
        const unsigned char *blk = row + (size_t)sb * 144;
        float d = half2f((unsigned short)(blk[0] | (blk[1] << 8)));
        float dmin = half2f((unsigned short)(blk[2] | (blk[3] << 8)));
        const float *xb = x + (sb << 8);
        for (lane = 0; lane < 32; lane++) {
            float acc = 0.0f;
            for (j = 0; j < 4; j++) {
                unsigned char qb = blk[16 + 32 * j + lane];
                for (h = 0; h < 2; h++) {
                    float sc, mn;
                    int nib = h ? (qb >> 4) : (qb & 0xF);
                    kq_scale_min(blk + 4, 2 * j + h, &sc, &mn);
                    acc = fmaf(xb[64 * j + 32 * h + lane],
                               (d * sc) * (float)nib - dmin * mn, acc);
                }
            }
            p[lane] += acc;
        }
    }
}

static float ref_row_q4k(const unsigned char *row, const float *x, int cols)
{
    float p[32] = { 0 };
    ref_lanes_q4k(row, x, cols, p);
    return tree32(p);
}

/* Replica of the kernel's sub-block activation sum: four threads own each
 * 32-element group, thread `quarter` summing the 8 elements
 * 8*quarter + (k ^ (group & 7)) in k order (the XOR is the kernel's LDS
 * bank-conflict swizzle), then __shfl_down(2) + __shfl_down(1) folds the
 * quarters as (q0 + q2) + (q1 + q3). */
static float ref_xsum_group(const float *xg, int group)
{
    float q[4];
    int quarter, k;
    for (quarter = 0; quarter < 4; quarter++) {
        float s = 0.0f;
        for (k = 0; k < 8; k++)
            s += xg[8 * quarter + (k ^ (group & 7))];
        q[quarter] = s;
    }
    return (q[0] + q[2]) + (q[1] + q[3]);
}

/* Replica of the MIN-LIFTED fc_matvec_q4k (matvec_kquants.cu), in the
 * kernel's exact op order:
 *   - the SCALE half accumulates sc*nib*x per lane and applies d ONCE per
 *     super-block (fmaf(d, acc, p));
 *   - the MIN half accumulates dmin*mn*xsum at one (super-block, sub-block)
 *     pair per lane per pass, where xsum is the block's __shfl_down tree sum
 *     of the 32 staged activations of that sub-block;
 *   - the two are subtracted per lane, then ONE tree32.
 * The 2048-element staging chunk is load-bearing: it fixes both the xsum
 * grouping and the pair -> lane assignment, so it is replicated here. */
static float ref_row_q4k_minlift(const unsigned char *row, const float *x,
                                 int cols)
{
    float p[32] = { 0 }, m[32] = { 0 }, f[32];
    int cbase, lane, sb, j, h, g, pi;
    for (cbase = 0; cbase < cols; cbase += 2048) {
        int cend = cols - cbase < 2048 ? cols - cbase : 2048;
        int nsb = cend >> 8;
        float xsum[64];
        for (g = 0; g < (cend >> 5); g++)
            xsum[g] = ref_xsum_group(x + cbase + (g << 5), g);
        for (sb = 0; sb < nsb; sb++) {
            const unsigned char *blk = row + (size_t)((cbase >> 8) + sb) * 144;
            float d = half2f((unsigned short)(blk[0] | (blk[1] << 8)));
            const float *xb = x + cbase + (sb << 8);
            for (lane = 0; lane < 32; lane++) {
                float a = 0.0f;
                for (j = 0; j < 4; j++) {
                    unsigned char qb = blk[16 + 32 * j + lane];
                    for (h = 0; h < 2; h++) {
                        float sc, mn;
                        int nib = h ? (qb >> 4) : (qb & 0xF);
                        kq_scale_min(blk + 4, 2 * j + h, &sc, &mn);
                        a = fmaf(sc * (float)nib,
                                 xb[64 * j + 32 * h + lane], a);
                    }
                }
                p[lane] = fmaf(d, a, p[lane]);
            }
        }
        for (lane = 0; lane < 32; lane++) {
            for (pi = lane; pi < (nsb << 3); pi += 32) {
                const unsigned char *blk = row +
                    (size_t)((cbase >> 8) + (pi >> 3)) * 144;
                float dmin = half2f((unsigned short)(blk[2] | (blk[3] << 8)));
                float sc, mn;
                kq_scale_min(blk + 4, pi & 7, &sc, &mn);
                m[lane] = fmaf(dmin * mn, xsum[pi], m[lane]);
            }
        }
    }
    for (lane = 0; lane < 32; lane++)
        f[lane] = p[lane] - m[lane];
    return tree32(f);
}

/* fp64 reference for the same dot — the yardstick both fp32 associations are
 * measured against, so "the reassociation is numerically fine" is a number
 * and not an assertion. Association-independent to within fp64 rounding. */
static double ref_row_q4k_f64(const unsigned char *row, const float *x,
                              int cols)
{
    double acc = 0.0;
    int n_sb = cols >> 8, sb, lane, j, h;
    for (sb = 0; sb < n_sb; sb++) {
        const unsigned char *blk = row + (size_t)sb * 144;
        double d = half2f((unsigned short)(blk[0] | (blk[1] << 8)));
        double dmin = half2f((unsigned short)(blk[2] | (blk[3] << 8)));
        const float *xb = x + (sb << 8);
        for (lane = 0; lane < 32; lane++)
            for (j = 0; j < 4; j++) {
                unsigned char qb = blk[16 + 32 * j + lane];
                for (h = 0; h < 2; h++) {
                    float sc, mn;
                    int nib = h ? (qb >> 4) : (qb & 0xF);
                    kq_scale_min(blk + 4, 2 * j + h, &sc, &mn);
                    acc += ((d * sc) * nib - dmin * mn) *
                           (double)xb[64 * j + 32 * h + lane];
                }
            }
    }
    return acc;
}

static float ref_row_q5k(const unsigned char *row, const float *x, int cols)
{
    float p[32] = { 0 };
    int n_sb = cols >> 8, sb, lane, j, h;
    for (sb = 0; sb < n_sb; sb++) {
        const unsigned char *blk = row + (size_t)sb * 176;
        float d = half2f((unsigned short)(blk[0] | (blk[1] << 8)));
        float dmin = half2f((unsigned short)(blk[2] | (blk[3] << 8)));
        const float *xb = x + (sb << 8);
        for (lane = 0; lane < 32; lane++) {
            unsigned char qh = blk[16 + lane];
            float acc = 0.0f;
            for (j = 0; j < 4; j++) {
                unsigned char qb = blk[48 + 32 * j + lane];
                for (h = 0; h < 2; h++) {
                    float sc, mn;
                    int nib = h ? (qb >> 4) : (qb & 0xF);
                    int hi = (qh & (1u << (2 * j + h))) ? 16 : 0;
                    kq_scale_min(blk + 4, 2 * j + h, &sc, &mn);
                    acc = fmaf(xb[64 * j + 32 * h + lane],
                               (d * sc) * (float)(nib + hi) - dmin * mn, acc);
                }
            }
            p[lane] += acc;
        }
    }
    return tree32(p);
}

static float ref_row_q6k(const unsigned char *row, const float *x, int cols)
{
    float p[32] = { 0 };
    int n_sb = cols >> 8, sb, lane, half, quad;
    for (sb = 0; sb < n_sb; sb++) {
        const unsigned char *blk = row + (size_t)sb * 210;
        float d = half2f((unsigned short)(blk[208] | (blk[209] << 8)));
        const float *xb = x + (sb << 8);
        for (lane = 0; lane < 32; lane++) {
            int lhalf = lane >> 4;
            float acc = 0.0f;
            for (half = 0; half < 2; half++) {
                unsigned char qh = blk[128 + 32 * half + lane];
                unsigned char ql0 = blk[64 * half + lane];
                unsigned char ql1 = blk[64 * half + 32 + lane];
                for (quad = 0; quad < 4; quad++) {
                    unsigned char qlb = (quad & 1) ? ql1 : ql0;
                    int nib = (quad & 2) ? (qlb >> 4) : (qlb & 0xF);
                    int hb = (qh >> (2 * quad)) & 3;
                    int q = (nib | (hb << 4)) - 32;
                    signed char sc = (signed char)blk[192 + half * 8 +
                                                      quad * 2 + lhalf];
                    acc = fmaf(xb[128 * half + 32 * quad + lane],
                               (float)(sc * q), acc);
                }
            }
            p[lane] = fmaf(acc, d, p[lane]);
        }
    }
    return tree32(p);
}

static float ref_row_q80(const unsigned char *row, const float *x, int cols)
{
    float p[32] = { 0 };
    int n_b = cols >> 5, b, lane;
    for (b = 0; b < n_b; b++) {
        const unsigned char *blk = row + (size_t)b * 34;
        float d = half2f((unsigned short)(blk[0] | (blk[1] << 8)));
        for (lane = 0; lane < 32; lane++)
            p[lane] = fmaf(x[(b << 5) + lane],
                           d * (float)((signed char)blk[2 + lane]), p[lane]);
    }
    return tree32(p);
}

typedef float (*ref_row_fn)(const unsigned char *, const float *, int);
typedef void (*mv_fn)(fc_buffer *, fc_buffer *, fc_buffer *, int, int, int);

/* Only fc_matvec_q3k has a default-ctx wrapper; shim the rest. */
#define MV_SHIM(name)                                                         \
    static void name##_dflt(fc_buffer *w, fc_buffer *x, fc_buffer *y,         \
                            int r, int c, int rs)                             \
    { name##_ctx(fc_default_ctx(), w, x, y, r, c, rs); }
MV_SHIM(fc_matvec_q4k)
MV_SHIM(fc_matvec_q5k)
MV_SHIM(fc_matvec_q6k)
MV_SHIM(fc_matvec_q80)
MV_SHIM(fc_matvec_f32)

static int ulp_diff(float a, float b)
{
    int32_t ia, ib;
    int64_t d;
    memcpy(&ia, &a, 4);
    memcpy(&ib, &b, 4);
    d = (int64_t)ia - ib;
    if (d < 0) d = -d;
    return (d > 1000000) ? 1000000 : (int)d;
}

/* Verify one quant matvec kernel vs the CPU replica, bit-exact. */
static int verify_mv(const char *name, int t, mv_fn fn, ref_row_fn ref,
                     int rows, int cols)
{
    size_t raw_stride = raw_block_bytes(t) * (size_t)((t == 8) ? cols / 32
                                                               : cols / 256);
    unsigned char *raw = rand_bytes(raw_stride * (size_t)rows);
    float *xh = rand_floats((size_t)cols);
    float *yh = malloc((size_t)rows * sizeof(float));
    int dev_stride = 0, r, nmis = 0, maxulp = 0, ret;
    fc_buffer *w, *x, *y;

    sanitize_quant(t, raw, raw_stride * (size_t)rows);
    w = upload_weight(t, raw, rows, cols, &dev_stride);
    x = fc_buffer_from_ptr(xh, (size_t)cols * sizeof(float));
    y = fc_buffer_alloc((size_t)rows * sizeof(float));
    fn(w, x, y, rows, cols, dev_stride);
    fc_sync();
    fc_buffer_to_cpu(y, yh);

    for (r = 0; r < rows; r++) {
        float want = ref(raw + (size_t)r * raw_stride, xh, cols);
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
    printf("verify %-6s %6dx%-5d  %s (mismatches %d, max ulp %d)\n",
           name, rows, cols, nmis ? "FAIL" : "bit-exact", nmis, maxulp);
    ret = nmis ? -1 : 0;
    fc_buffer_free(w); fc_buffer_free(x); fc_buffer_free(y);
    free(raw); free(xh); free(yh);
    return ret;
}

/* Verify a REASSOCIATED kernel — one whose op order was deliberately changed
 * for speed, so it cannot be bit-identical to the previous order.
 *
 * Two gates, both hard:
 *   1. BIT-EXACT against a replica of the kernel's NEW op order. This is the
 *      real proof: it says the restructured algebra is exactly what the
 *      kernel computes, not merely "close". A dropped or misindexed
 *      correction term fails here immediately.
 *   2. The fp32 relative error against an fp64 reference must stay under
 *      tol_rel AND within REASSOC_ERR_FACTOR of the OLD order's error. The
 *      second half is what catches a restructure that is algebraically right
 *      but numerically worse; an absolute cap alone would not.
 * The old-vs-new ULP spread is reported for context (it is the amount real
 * model outputs will move), not gated — it is expected to be nonzero. */
#define REASSOC_ERR_FACTOR 4.0

static int verify_mv_reassoc(const char *name, int t, mv_fn fn,
                             ref_row_fn ref_new, ref_row_fn ref_old,
                             double (*ref_f64)(const unsigned char *,
                                               const float *, int),
                             int rows, int cols, double tol_rel)
{
    size_t raw_stride = raw_block_bytes(t) * (size_t)(cols / 256);
    unsigned char *raw = rand_bytes(raw_stride * (size_t)rows);
    float *xh = rand_floats((size_t)cols);
    float *yh = malloc((size_t)rows * sizeof(float));
    int dev_stride = 0, r, nmis = 0, maxulp = 0, drift_ulp = 0, ret;
    double rel_new = 0.0, rel_old = 0.0;
    fc_buffer *w, *x, *y;

    sanitize_quant(t, raw, raw_stride * (size_t)rows);
    w = upload_weight(t, raw, rows, cols, &dev_stride);
    x = fc_buffer_from_ptr(xh, (size_t)cols * sizeof(float));
    y = fc_buffer_alloc((size_t)rows * sizeof(float));
    fn(w, x, y, rows, cols, dev_stride);
    fc_sync();
    fc_buffer_to_cpu(y, yh);

    for (r = 0; r < rows; r++) {
        const unsigned char *rp = raw + (size_t)r * raw_stride;
        float want = ref_new(rp, xh, cols);
        float prev = ref_old(rp, xh, cols);
        double exact = ref_f64(rp, xh, cols);
        double scale = fabs(exact) > 1e-6 ? fabs(exact) : 1e-6;
        double en = fabs((double)want - exact) / scale;
        double eo = fabs((double)prev - exact) / scale;
        int d = ulp_diff(prev, want);
        if (en > rel_new) rel_new = en;
        if (eo > rel_old) rel_old = eo;
        if (d > drift_ulp) drift_ulp = d;
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
    ret = (nmis || rel_new > tol_rel ||
           rel_new > REASSOC_ERR_FACTOR * rel_old) ? -1 : 0;
    printf("verify %-6s %6dx%-5d  %s (mismatches %d, max ulp %d)\n",
           name, rows, cols, nmis ? "FAIL" : "bit-exact vs new order",
           nmis, maxulp);
    printf("    reassoc: max |new-old| %d ulp; rel err vs fp64 "
           "new %.3g old %.3g (tol %.3g, factor cap %.1fx) %s\n",
           drift_ulp, rel_new, rel_old, tol_rel, REASSOC_ERR_FACTOR,
           ret ? "FAIL" : "ok");
    fc_buffer_free(w); fc_buffer_free(x); fc_buffer_free(y);
    free(raw); free(xh); free(yh);
    return ret;
}

/* ------------------------------------------------------------------
 * Grouped-MoE replicas (moe_expert_group.cu)
 * ------------------------------------------------------------------ */

/* The down kernels' wpp launch arg — must mirror fc_moe_down_accum_ctx's
 * env/default resolution so the replica reproduces the launched shape. */
static int moe_down_wpp(void)
{
    const char *e = getenv("FC_MOE_DOWN_WPP");
    if (e) {
        int v = atoi(e);
        if (v == 1 || v == 2 || v == 4 || v == 8)
            return v;
    }
    return FC_MOE_DOWN_WPP_DEFAULT;
}

/* Replica of one fc_moe_gateup_q3k output: silu(gate.row . x) * (up.row . x).
 * glibc expf and the device expf differ by a couple of ulp, so the caller
 * compares with a small ulp tolerance instead of bit-exact. */
static float ref_moe_gateup(const unsigned char *gate_raw,
                            const unsigned char *up_raw, const float *x,
                            int cols)
{
    float g = ref_row_q3k(gate_raw, x, cols);
    float u = ref_row_q3k(up_raw, x, cols);
    return (g / (1.0f + expf(-g))) * u;
}

/* Replica of one fc_moe_down_* output row in the kernel's exact op order:
 * warp wk chains t = fmaf(top_wt[s], dot_s, t) per lane over slots wk,
 * wk+wpp, ..., tree-reduces its lanes, and the wpp per-warp partials are
 * added in ascending wk order. Bit-exact vs the kernel. hid should be the
 * DEVICE-computed hid (downloaded) so this isolates the down kernel. */
static float ref_moe_down(int t, const unsigned char *down_raw,
                          size_t raw_row_stride, size_t raw_expert_stride,
                          const float *hid, const int32_t *idx,
                          const float *wt, int k, int row, int ff, int wpp)
{
    float total = 0.0f;
    int wk, s, l;
    for (wk = 0; wk < wpp; wk++) {
        float tl[32] = { 0 };
        for (s = wk; s < k; s += wpp) {
            const unsigned char *rp = down_raw +
                (size_t)idx[s] * raw_expert_stride +
                (size_t)row * raw_row_stride;
            float p[32] = { 0 };
            if (t == 11)
                ref_lanes_q3k(rp, hid + (size_t)s * ff, ff, p);
            else
                ref_lanes_q4k(rp, hid + (size_t)s * ff, ff, p);
            for (l = 0; l < 32; l++)
                tl[l] = fmaf(wt[s], p[l], tl[l]);
        }
        total += tree32(tl);
    }
    return total;
}

/* ------------------------------------------------------------------
 * Timing
 * ------------------------------------------------------------------ */

static double bench_mv(const char *name, int t, mv_fn fn, int rows, int cols,
                       int iters)
{
    size_t raw_stride = raw_block_bytes(t) * (size_t)((t == 8) ? cols / 32
                                                               : cols / 256);
    unsigned char *raw = rand_bytes(raw_stride * (size_t)rows);
    int dev_stride = 0, i;
    fc_buffer *w, *x, *y;
    double t0, dt, gb;

    sanitize_quant(t, raw, raw_stride * (size_t)rows);
    w = upload_weight(t, raw, rows, cols, &dev_stride);
    free(raw);
    {
        float *xh = rand_floats((size_t)cols);
        x = fc_buffer_from_ptr(xh, (size_t)cols * sizeof(float));
        free(xh);
    }
    y = fc_buffer_alloc((size_t)rows * sizeof(float));

    fn(w, x, y, rows, cols, dev_stride);              /* warm up */
    fc_sync();
    t0 = now_s();
    for (i = 0; i < iters; i++)
        fn(w, x, y, rows, cols, dev_stride);
    fc_sync();
    dt = (now_s() - t0) / iters;
    gb = (double)rows * dev_stride / dt / 1e9;
    printf("%-24s %6dx%-5d  %8.3f ms  %7.2f GB/s\n", name, rows, cols,
           dt * 1e3, gb);
    fc_buffer_free(w); fc_buffer_free(x); fc_buffer_free(y);
    return dt * 1e3;
}

/* Matched-shape per-quant sweep — the apples-to-apples question the per-type
 * benches above cannot answer, because each of them uses a different shape.
 * Same rows x cols for every type, so the ms column IS the cost of the quant
 * CHOICE for a tensor of that shape. Device bytes per 256 weights after
 * fc_repack_weight: q3k 152, q4k 152 (identical!), q5k 184, q6k 216, q80 288.
 * Normalised to q4k so "what does staying on q6k cost me" reads directly. */
static void sweep_quants(int rows, int cols, int iters)
{
    static const char *nm[5] = { "q3k", "q4k", "q5k", "q6k", "q80" };
    static const int ty[5] = { 11, 12, 13, 14, 8 };
    static const double bpb[5] = { 152, 152, 184, 216, 288 }; /* B / 256 wts */
    mv_fn fn[5];
    double ms[5];
    int i;

    fn[0] = fc_matvec_q3k;     fn[1] = fc_matvec_q4k_dflt;
    fn[2] = fc_matvec_q5k_dflt; fn[3] = fc_matvec_q6k_dflt;
    fn[4] = fc_matvec_q80_dflt;

    printf("-- matched shape %d x %d, %d iters --\n", rows, cols, iters);
    for (i = 0; i < 5; i++)
        ms[i] = bench_mv(nm[i], ty[i], fn[i], rows, cols, iters);

    printf("   %-6s %9s %9s %9s %9s\n",
           "type", "ms", "vs q4k", "dev B/wt", "ms/B ratio");
    for (i = 0; i < 5; i++)
        printf("   %-6s %9.3f %8.3fx %9.4f %9.3f\n", nm[i], ms[i],
               ms[i] / ms[1], bpb[i] / 256.0,
               (ms[i] / ms[1]) / (bpb[i] / bpb[1]));
    printf("   (ms/B ratio > 1 = that quant's DECODE is more expensive per\n"
           "    byte than q4k's; = 1 = pure bandwidth/issue scaling)\n");
}

/* F32 matvec has no repack; bench it raw. */
static void bench_f32(int rows, int cols, int iters)
{
    float *wh = rand_floats((size_t)rows * cols);
    float *xh = rand_floats((size_t)cols);
    fc_buffer *w = fc_buffer_from_ptr(wh, (size_t)rows * cols * sizeof(float));
    fc_buffer *x = fc_buffer_from_ptr(xh, (size_t)cols * sizeof(float));
    fc_buffer *y = fc_buffer_alloc((size_t)rows * sizeof(float));
    int stride = cols * 4, i;
    double t0, dt;
    free(wh); free(xh);
    fc_matvec_f32_dflt(w, x, y, rows, cols, stride);
    fc_sync();
    t0 = now_s();
    for (i = 0; i < iters; i++)
        fc_matvec_f32_dflt(w, x, y, rows, cols, stride);
    fc_sync();
    dt = (now_s() - t0) / iters;
    printf("%-24s %6dx%-5d  %8.3f ms  %7.2f GB/s\n", "f32 router", rows, cols,
           dt * 1e3, (double)rows * stride / dt / 1e9);
    fc_buffer_free(w); fc_buffer_free(x); fc_buffer_free(y);
}

int main(void)
{
    fc_cuda_ctx *ctx;
    int it = 50, fails = 0;

    if (fc_init() != 0) {
        fprintf(stderr, "no CUDA\n");
        return 1;
    }
    ctx = fc_default_ctx();
    srand(42);

    printf("== verify vs CPU replica (repack=%d) ==\n", USE_REPACK);
    fails += verify_mv("q3k", 11, fc_matvec_q3k, ref_row_q3k, 512, 2048);
    /* q4k is MIN-LIFTED and therefore reassociated — bit-exact against its
     * own order's replica, plus an fp64 error gate. tol 1e-4: a 2048-term
     * fp32 dot of random signed values cancels hard, so the ratio of the
     * fp32 rounding error to the (small) result is orders of magnitude
     * larger than eps; both orders sit far below this, and the factor cap
     * is what actually holds the new order to the old one's accuracy. */
    fails += verify_mv_reassoc("q4k", 12, fc_matvec_q4k_dflt,
                               ref_row_q4k_minlift, ref_row_q4k,
                               ref_row_q4k_f64, 512, 2048, 1e-4);
    /* 512 cols: 2 super-blocks per chunk, so only lanes 0-15 run the min
     * correction pass. 4096 cols: TWO staging chunks, so the xsum table is
     * rebuilt and the pair -> lane assignment restarts mid-row. Both shapes
     * exercise indexing the 2048 case cannot. */
    fails += verify_mv_reassoc("q4k512", 12, fc_matvec_q4k_dflt,
                               ref_row_q4k_minlift, ref_row_q4k,
                               ref_row_q4k_f64, 2048, 512, 1e-4);
    fails += verify_mv_reassoc("q4k4k", 12, fc_matvec_q4k_dflt,
                               ref_row_q4k_minlift, ref_row_q4k,
                               ref_row_q4k_f64, 512, 4096, 1e-4);
    fails += verify_mv("q5k", 13, fc_matvec_q5k_dflt, ref_row_q5k, 512, 2048);
    fails += verify_mv("q6k", 14, fc_matvec_q6k_dflt, ref_row_q6k, 512, 2048);
    fails += verify_mv("q80", 8, fc_matvec_q80_dflt, ref_row_q80, 512, 2048);
    fails += verify_mv("q3k512", 11, fc_matvec_q3k, ref_row_q3k, 2048, 512);
    if (fails) {
        printf("VERIFY FAILED — skipping timing\n");
        fc_free();
        return 1;
    }

    printf("== plain matvecs (rows x cols, device-weight GB/s) ==\n");
    bench_mv("q3k gate/up (expert)", 11, fc_matvec_q3k, 512, 2048, it);
    bench_mv("q3k down (expert)", 11, fc_matvec_q3k, 2048, 512, it);
    bench_mv("q3k qkv-ish", 11, fc_matvec_q3k, 8192, 2048, it);
    bench_mv("q4k 2048x512", 12, fc_matvec_q4k_dflt, 2048, 512, it);
    bench_mv("q5k qkv 8192x2048", 13, fc_matvec_q5k_dflt, 8192, 2048, it);
    bench_mv("q6k head 248320x2048", 14, fc_matvec_q6k_dflt, 248320, 2048, 10);
    bench_mv("q80 4096x2048", 8, fc_matvec_q80_dflt, 4096, 2048, it);
    bench_f32(256, 2048, it);

    printf("== per-quant MATCHED-SHAPE sweep (what the quant choice costs) ==\n");
    sweep_quants(2048, 2048, it);      /* generic dense / ssm projection */
    sweep_quants(8192, 2048, it);      /* attn qkv                        */
    sweep_quants(248320, 2048, 10);    /* LM head (saber/ornith vocab)    */

    printf("== grouped MoE (per-layer shapes, k=8) ==\n");
    {
        int n_expert = 256, ff = 512, cols = 2048, k = 8;
        int gu_stride = 0, dn3_stride = 0, dn4_stride = 0;
        size_t graw = raw_block_bytes(11) * (cols / 256);
        size_t d3raw = raw_block_bytes(11) * (ff / 256);
        size_t d4raw = raw_block_bytes(12) * (ff / 256);
        /* Only k experts are touched per launch; allocating all 256 experts
         * matches the real memory layout / cache behavior. */
        unsigned char *gr = rand_bytes((size_t)n_expert * ff * graw);
        unsigned char *ur = rand_bytes((size_t)n_expert * ff * graw);
        unsigned char *d3r = rand_bytes((size_t)n_expert * cols * d3raw);
        unsigned char *d4r = rand_bytes((size_t)n_expert * cols * d4raw);
        int32_t idx[8] = { 3, 17, 42, 77, 101, 155, 201, 250 };
        float wt[8] = { .2f, .2f, .15f, .15f, .1f, .1f, .05f, .05f };
        fc_buffer *gate, *up, *dn3, *dn4, *tidx, *twt, *x, *hid, *out;
        float *xh = rand_floats((size_t)cols);
        double t0, dt, gb;
        int i;

        sanitize_quant(11, gr, (size_t)n_expert * ff * graw);
        sanitize_quant(11, ur, (size_t)n_expert * ff * graw);
        sanitize_quant(11, d3r, (size_t)n_expert * cols * d3raw);
        sanitize_quant(12, d4r, (size_t)n_expert * cols * d4raw);
        gate = upload_weight(11, gr, (long)n_expert * ff, cols, &gu_stride);
        up = upload_weight(11, ur, (long)n_expert * ff, cols, &gu_stride);
        dn3 = upload_weight(11, d3r, (long)n_expert * cols, ff, &dn3_stride);
        dn4 = upload_weight(12, d4r, (long)n_expert * cols, ff, &dn4_stride);
        tidx = fc_buffer_from_ptr(idx, sizeof(idx));
        twt = fc_buffer_from_ptr(wt, sizeof(wt));
        x = fc_buffer_from_ptr(xh, (size_t)cols * sizeof(float));
        hid = fc_buffer_alloc((size_t)k * ff * sizeof(float));
        out = fc_buffer_alloc((size_t)cols * sizeof(float));

        /* -- verify vs the CPU replicas before timing: gateup with a small
         * ulp tolerance (expf), the cooperative down kernels bit-exact
         * against the downloaded device hid -- */
        {
            float *hid_h = malloc((size_t)k * ff * sizeof(float));
            float *out_h = malloc((size_t)cols * sizeof(float));
            int wpp = moe_down_wpp();
            int s, r, q, moe_fails = 0;

            fc_moe_gateup_q3k_ctx(ctx, gate, up, x, tidx, hid, k, ff, cols,
                                  gu_stride);
            fc_sync();
            fc_buffer_to_cpu(hid, hid_h);
            {
                int maxulp = 0, nmis = 0;
                for (s = 0; s < k; s++)
                    for (r = 0; r < ff; r++) {
                        const unsigned char *grow =
                            gr + ((size_t)idx[s] * ff + r) * graw;
                        const unsigned char *urow =
                            ur + ((size_t)idx[s] * ff + r) * graw;
                        float want = ref_moe_gateup(grow, urow, xh, cols);
                        int u = ulp_diff(want, hid_h[(size_t)s * ff + r]);
                        if (u > maxulp) maxulp = u;
                        if (u > 8) nmis++;
                    }
                printf("verify moe gateup q3k    %s (mismatches %d, "
                       "max ulp %d, tol 8)\n",
                       nmis ? "FAIL" : "ok", nmis, maxulp);
                if (nmis) moe_fails++;
            }
            for (q = 0; q < 2; q++) {
                int t = q ? 12 : 11;
                size_t rstride = q ? d4raw : d3raw;
                const unsigned char *draw = q ? d4r : d3r;
                int maxulp = 0, nmis = 0;
                fc_moe_down_accum_ctx(ctx, t, q ? dn4 : dn3, hid, tidx, twt,
                                      out, k, cols, ff,
                                      q ? dn4_stride : dn3_stride);
                fc_sync();
                fc_buffer_to_cpu(out, out_h);
                for (r = 0; r < cols; r++) {
                    float want = ref_moe_down(t, draw, rstride,
                                              (size_t)cols * rstride, hid_h,
                                              idx, wt, k, r, ff, wpp);
                    if (memcmp(&want, &out_h[r], 4) != 0) {
                        int u = ulp_diff(want, out_h[r]);
                        if (u > maxulp) maxulp = u;
                        if (nmis < 3)
                            printf("    row %d: ref %.9g (%08x) gpu %.9g "
                                   "(%08x)\n", r, (double)want,
                                   *(unsigned *)&want, (double)out_h[r],
                                   *(unsigned *)&out_h[r]);
                        nmis++;
                    }
                }
                printf("verify moe down %s (wpp=%d) %s (mismatches %d, "
                       "max ulp %d)\n", q ? "q4k" : "q3k", wpp,
                       nmis ? "FAIL" : "bit-exact", nmis, maxulp);
                if (nmis) moe_fails++;
            }
            free(hid_h); free(out_h);
            if (moe_fails) {
                printf("MOE VERIFY FAILED — skipping timing\n");
                fc_free();
                return 1;
            }
        }
        free(gr); free(ur); free(d3r); free(d4r);
        free(xh);

        fc_moe_gateup_q3k_ctx(ctx, gate, up, x, tidx, hid, k, ff, cols,
                              gu_stride);
        fc_sync();
        t0 = now_s();
        for (i = 0; i < it; i++)
            fc_moe_gateup_q3k_ctx(ctx, gate, up, x, tidx, hid, k, ff, cols,
                                  gu_stride);
        fc_sync();
        dt = (now_s() - t0) / it;
        gb = 2.0 * k * ff * gu_stride / dt / 1e9;
        printf("%-24s %6s       %8.3f ms  %7.2f GB/s\n", "moe gateup q3k",
               "8x512x2048", dt * 1e3, gb);

        fc_moe_down_accum_ctx(ctx, 11, dn3, hid, tidx, twt, out, k, cols, ff,
                              dn3_stride);
        fc_sync();
        t0 = now_s();
        for (i = 0; i < it; i++)
            fc_moe_down_accum_ctx(ctx, 11, dn3, hid, tidx, twt, out, k, cols,
                                  ff, dn3_stride);
        fc_sync();
        dt = (now_s() - t0) / it;
        gb = (double)k * cols * dn3_stride / dt / 1e9;
        printf("%-24s %6s       %8.3f ms  %7.2f GB/s\n", "moe down q3k",
               "8x2048x512", dt * 1e3, gb);

        fc_moe_down_accum_ctx(ctx, 12, dn4, hid, tidx, twt, out, k, cols, ff,
                              dn4_stride);
        fc_sync();
        t0 = now_s();
        for (i = 0; i < it; i++)
            fc_moe_down_accum_ctx(ctx, 12, dn4, hid, tidx, twt, out, k, cols,
                                  ff, dn4_stride);
        fc_sync();
        dt = (now_s() - t0) / it;
        gb = (double)k * cols * dn4_stride / dt / 1e9;
        printf("%-24s %6s       %8.3f ms  %7.2f GB/s\n", "moe down q4k",
               "8x2048x512", dt * 1e3, gb);

        fc_buffer_free(gate); fc_buffer_free(up);
        fc_buffer_free(dn3); fc_buffer_free(dn4);
        fc_buffer_free(tidx); fc_buffer_free(twt);
        fc_buffer_free(x); fc_buffer_free(hid); fc_buffer_free(out);
    }

    fc_free();
    return 0;
}
