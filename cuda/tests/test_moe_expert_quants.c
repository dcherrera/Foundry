/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/*
 * test_moe_expert_quants.c — parity + dispatch gate for the grouped MoE
 * expert kernels (moe_expert_group.cu) on one K80 die.
 *
 * WHY. moe_expert_group.cu used to cover exactly the Q3_K_M mini's mixture:
 * a Q3_K gate+up with NO type dispatch at all, and a down that dispatched
 * Q3_K|Q4_K. A Qwen3.5-MoE quantized Q4_K_M is a different mixture — Q4_K
 * gate/up throughout, and a merged ffn_down_exps that is Q4_K on half the
 * layers and Q6_K on the other half. So the host fed Q4_K bytes to a Q3_K
 * gate+up decode, and Q6_K down layers lost the grouped path entirely. This
 * test gates both fixes: the new fc_moe_gateup_q4k{,_nx} and
 * fc_moe_down_q6k{,_nx}, and the fail-closed dispatch in front of them.
 *
 * METHOD — the same one cuda/tests/bench_matvec.c uses for every other kernel
 * in this tree, and the replica helpers below are that file's, unchanged: a
 * CPU replica decodes the ORIGINAL GGUF block layout (the source of truth on
 * both sides of the upload-time fc_repack_weight) in the DEVICE's exact fp op
 * order — per-lane fmaf chains, then the 5-round __shfl_down(16,8,4,2,1) tree
 * — so a correct kernel matches bit-exactly. The dot products here ARE
 * compared bit-exactly; only the final silu carries a tolerance, because
 * glibc expf and the device expf differ by a couple of ulp (bench_matvec's
 * grouped-MoE arm allows 8 ulp there for the same reason).
 *
 * The 2-byte formats (gguf type 1 F16, type 30 BF16) are the same test one rung
 * up, and it is the rung that matters: F32 storage does not fit on the reference 8-card host
 * (35B x 4 B = 140 GB against 97.6 GB of VRAM), so 2 bytes per weight IS the
 * ceiling. Neither has blocks or scales, so their arms exercise something the
 * quant arms cannot — a 512 B step instead of 152/216, no fc_repack_weight pass
 * at all, and the sm_37 2-byte load + in-register widen (there is no fp16
 * ARITHMETIC on this hardware). BOTH are here because they are not
 * interchangeable: the checkpoints ship BF16, so bf16 is the control that
 * carries the published weight values, and F16 is the variant that trades
 * exponent range for 3 more stored mantissa bits. Converting one to the other
 * re-rounds every weight, so a run in the wrong format is a run on a perturbed
 * model.
 *
 * Both are bit-exact against the replica for the same reason the quants are.
 * For F16 the replica's half2f reproduces the IEEE widening including the
 * subnormal renormalisation; for BF16 there is no rounding to reproduce at all,
 * because a bf16 value IS the top 16 bits of the fp32 pattern and the widen is
 * a shift. Both fixtures deliberately leave subnormals in the input. bf16's
 * coarser mantissa does not soften the compare — the comparison is on the fp32
 * result and the widen is lossless — it only makes the summed VALUES coarser.
 *
 * A HOST hazard this file learned the hard way, see host_denormals_on(): the
 * tree's -ffast-math sets MXCSR FTZ/DAZ, so by default the replica flushes fp32
 * subnormals and the device does not. Only BF16 can reach that regime, and it
 * cost a full round of 12 red arms that were entirely the harness's fault. The
 * flush is now cleared at startup and arm 0 gates that it stayed cleared.
 *
 * ARMS
 *   0. host denormals  — the replica's own precondition (see above).
 *   0b. bf16/f16 gateup/down — the ceiling, scalar loads. Same comparisons as
 *                        1-5, one rung up, plus the same set again against the
 *                        uint2 kernels and the map red-proof that keeps those
 *                        from being vacuous (arm_vec_map_red).
 *   1. q4k gateup      — fc_moe_gateup_ctx(quant 12) vs the replica. THE GATE.
 *   2. q3k gateup      — fc_moe_gateup_ctx(quant 11) vs the replica. Proves
 *                        the shared FC_MOE_GATEUP_BODY macro did not disturb
 *                        the shipping kernel.
 *   3. q4k gateup 4096 — cols = 4096 is TWO 2048-float staging chunks, so the
 *                        shared-x loop restarts mid-row. 2048 never exercises
 *                        that.
 *   4. q4k _nx         — the batched kernel over ntok tokens, each token
 *                        against the same replica, plus bit-exactness against
 *                        the single-token kernel's hid.
 *   5. down q6k/q4k    — fc_moe_down_accum_ctx and its _nx twin against a
 *                        replica of the COOPERATIVE kernel's exact order
 *                        (warp wk chains slots wk, wk+wpp, ...; per-warp
 *                        shfl tree; wpp partials added in ascending wk).
 *                        Bit-exact, zero tolerance — no expf here.
 *   5b. q80 gateup/down — Q8_0 on the 288-B expert layout (fc_moe_repack_weight)
 *                        vs a replica on the ORIGINAL 34-B blocks: gateup
 *                        2048/4096/nx (tol 8, silu only), down + down nx
 *                        at k 8 and k 7 (lone last slot); the batched arms
 *                        again on the plain-load _nxp kernels
 *                        bit-exact, plus arm_q80_order_red (the dense q80 op
 *                        order must NOT match). MOE_TEST_RED_Q80=1 reads the
 *                        replica's quants unsigned: every q80 arm must FAIL.
 *   6. fail-closed     — an unsupported quant must return -1 / launch NOTHING
 *                        (canary output intact) on BOTH gate+up and down, not
 *                        fall through to another quant's decode.
 *
 * Build (from foundry/):  make USE_CUDA=1 cuda/tests/test_moe_expert_quants
 * Run:                    make USE_CUDA=1 test-moe-expert-quants
 * K80 host only — needs a K80 die and the CUDA driver.
 */

#include "cuda/fc_cuda.h"

#include <math.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__SSE2__)
#include <xmmintrin.h>                     /* _MM_SET_FLUSH_ZERO_MODE */
#endif
#if defined(__SSE3__)
#include <pmmintrin.h>                     /* _MM_SET_DENORMALS_ZERO_MODE */
#endif

#define GG_F16 1                   /* the ceiling: 2-byte storage, fp32 math */
#define GG_Q80 8                   /* the rung below: 34 B per 32 weights */
#define GG_Q3K 11
#define GG_Q4K 12
#define GG_Q5K 13                  /* the unsupported quant the gate rejects */
#define GG_Q6K 14
#define GG_BF16 30                 /* the ceiling AS SHIPPED — see the header */

/* FC_MOE_VEC_LOADS drives BOTH the kernel choice (in fc_cuda.c) and which
 * lane->element map the replica below must model. Setting them from two places
 * would turn a real mismatch into a green run, so this is the only writer, and
 * it reads the flag back through the library's own accessor rather than
 * assuming the setenv took. */
static int g_vec_map;

static void set_vec_mode(int on)
{
    setenv("FC_MOE_VEC_LOADS", on ? "1" : "0", 1);
    g_vec_map = fc_moe_vec_loads_enabled();
}

/*
 * Turn the HOST's subnormal flush off — without this the replica is not a
 * replica at the bottom of the fp32 range.
 *
 * This tree builds with -ffast-math, which on x86 links crtfastmath.o and sets
 * MXCSR FTZ + DAZ at startup: every fp32 subnormal OPERAND then reads as zero
 * and every subnormal RESULT is flushed to zero. sm_37 does neither — Kepler
 * has full fp32 denormal support and nvcc defaults to -ftz=false — so under the
 * stock flags the two sides are running different arithmetic.
 *
 * It stayed invisible through every K-quant and through F16 because none of
 * them can put a subnormal fp32 into the dot: an F16 subnormal WIDENS to a
 * normal fp32 near 1e-8, and the K-quants multiply integers by normal scales.
 * BF16 is the first format that can, because a bf16 subnormal widens to an fp32
 * subnormal near 1e-38, and the flush then makes the host's term exactly zero
 * where the device's is not.
 *
 * The size of that disagreement is ~1e-38 against sums of order 1-20, i.e. some
 * 30 orders of magnitude below their ulp, and it would round away in every case
 * but one: a round-to-even TIE. A tie tipped by 1e-38 moves the result a full
 * ulp, and one tie per few thousand fmaf chains is exactly the 1-2 mismatching
 * rows in 2048 this produced before the flush was turned off. Measured, not
 * assumed: rebuilding the identical sources with -fno-fast-math turned all 12
 * failing arms green with no change to any kernel.
 */
static void host_denormals_on(void)
{
#if defined(__SSE2__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
#endif
#if defined(__SSE3__)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
#endif
}

/* ...and prove it took, so that a future flag change cannot silently put the
 * flush back. volatile keeps the product out of the constant folder, which
 * would compute it with denormals regardless of the runtime MXCSR and hide the
 * very thing being checked. */
static int arm_host_denormals(void)
{
    volatile float tiny = 1e-30f, small = 1e-10f;
    float p = tiny * small;                /* ~1e-40: subnormal, or 0 if FTZ */
    int ok = p != 0.0f;
    printf("  %-28s %s (1e-30 * 1e-10 = %g)\n", "host denormals live",
           ok ? "ok" : "FAIL", (double)p);
    return ok ? 0 : 1;
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------
 * Random weights (from bench_matvec.c)
 * ------------------------------------------------------------------ */

static size_t raw_block_bytes(int t)
{
    if (t == GG_Q3K) return 110;
    if (t == GG_Q6K) return 210;
    /* F16/BF16 have no block structure — 256 halves is 512 B, which is exactly
     * what the fixtures' raw_block_bytes(t) * (cols/256) then resolves to
     * cols*2, the GGUF row stride. Nothing is repacked. */
    if (t == GG_F16 || t == GG_BF16) return 512;
    if (t == GG_Q80) return 272;           /* 8 x 34-B Q8_0 blocks */
    return 144;                            /* q4k */
}

/* The two 2-byte formats share every layout rule and differ only in the decode
 * — the one predicate the fixtures, the upload and the replica all branch on. */
static int is_raw16(int t)
{
    return t == GG_F16 || t == GG_BF16;
}

/* Clamp every fp16 block scale (d, and dmin for Q4_K) to exponent 2^-4..2^-3
 * with a random mantissa: random halves can be inf/nan, which breaks a
 * bit-exact compare (nan != nan). Positions per gguf_dequant.h. */
/* F16 has no scales — EVERY element is the fp16 that must not be inf/nan, so
 * the clamp is per-element rather than per-block. Exponents 0x12..0x1F are
 * folded down by 0x0E, which keeps |w| < 8, leaves exponent 0 (SUBNORMALS, ~1
 * value in 32) reachable on purpose — the device cvt.f32.f16 and the host
 * half2f must agree there too — and makes 0x1F (inf/nan) unreachable. */
static void sanitize_f16(unsigned char *p, size_t bytes)
{
    size_t off;
    for (off = 0; off + 2 <= bytes; off += 2) {
        unsigned short h = (unsigned short)(p[off] | (p[off + 1] << 8));
        int e = (h >> 10) & 0x1F;
        if (e > 0x11)
            e -= 0x0E;
        h = (unsigned short)((h & 0x83FFu) | ((unsigned)e << 10));
        p[off] = (unsigned char)(h & 0xFF);
        p[off + 1] = (unsigned char)(h >> 8);
    }
}

/* bf16 is 1|8|7, so its exponent field is 8 bits wide against F16's 5 and the
 * fold above does not transfer. Instead take 5 random bits, fold them exactly
 * as sanitize_f16 does, then re-bias 15 -> 127: the two fixtures then span the
 * SAME unbiased exponents (-15..+2, |w| < 8), which is what makes an f16-vs-bf16
 * timing comparison a comparison of the DECODE and not of the data. e == 0
 * (subnormal, 1 in 32) stays reachable on purpose — a bf16 subnormal widens to
 * an fp32 subnormal by the same shift, with no renormalisation step, and that
 * is the case worth pinning. 0xFF (inf/nan) is unreachable, as a bit-exact
 * compare needs. The 7 mantissa bits and the sign stay random. */
static void sanitize_bf16(unsigned char *p, size_t bytes)
{
    size_t off;
    for (off = 0; off + 2 <= bytes; off += 2) {
        unsigned short h = (unsigned short)(p[off] | (p[off + 1] << 8));
        int e5 = (h >> 8) & 0x1F;
        int e8;
        if (e5 > 0x11)
            e5 -= 0x0E;
        e8 = e5 ? e5 + 112 : 0;            /* (e5 - 15) == (e8 - 127) */
        h = (unsigned short)((h & 0x807Fu) | ((unsigned)e8 << 7));
        p[off] = (unsigned char)(h & 0xFF);
        p[off + 1] = (unsigned char)(h >> 8);
    }
}

static void sanitize_quant(int t, unsigned char *p, size_t bytes)
{
    size_t bs = raw_block_bytes(t), off;
    if (t == GG_F16) {
        sanitize_f16(p, bytes);
        return;
    }
    if (t == GG_BF16) {
        sanitize_bf16(p, bytes);
        return;
    }
    if (t == GG_Q80) {                     /* fp16 d at the head of each block */
        for (off = 0; off + 34 <= bytes; off += 34) {
            unsigned short h = (unsigned short)(0x2C00 | (rand() & 0x3FF));
            p[off] = h & 0xFF;
            p[off + 1] = h >> 8;
        }
        return;
    }
    for (off = 0; off + bs <= bytes; off += bs) {
        unsigned char *b = p + off;
        unsigned short h = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        unsigned short h2 = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        if (t == GG_Q3K) {
            b[108] = h & 0xFF; b[109] = h >> 8;
        } else if (t == GG_Q6K) {
            b[208] = h & 0xFF; b[209] = h >> 8;
        } else {
            b[0] = h & 0xFF;  b[1] = h >> 8;
            b[2] = h2 & 0xFF; b[3] = h2 >> 8;
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

static fc_buffer *upload_weight(int t, const unsigned char *raw, long rows,
                                int cols, int *dev_stride)
{
    /* The EXPERT layout, as Pantheon's grouped upload uses it: identical to
     * fc_repack_* for every type but Q8_0 (288-B super-blocks). */
    size_t rs = fc_moe_repack_row_bytes(t, cols);
    unsigned char *tmp;
    fc_buffer *b;
    /* F16/BF16 are not repacked: the GGUF bytes ARE the device layout. This
     * mirrors gpu_weight_upload's raw branch exactly (fc_repack_row_bytes
     * returns 0 for types 1 and 30, so the repack branch is skipped and the
     * file stride survives). */
    if (is_raw16(t)) {
        *dev_stride = cols * 2;
        return fc_buffer_from_ptr((void *)raw,
                                  (size_t)cols * 2 * (size_t)rows);
    }
    tmp = malloc(rs * (size_t)rows);
    if (!tmp || fc_moe_repack_weight(t, raw, tmp, rows, cols) != 0) {
        free(tmp);
        return NULL;
    }
    b = fc_buffer_from_ptr(tmp, rs * (size_t)rows);
    free(tmp);
    *dev_stride = (int)rs;
    return b;
}

/* ------------------------------------------------------------------
 * CPU replica (from bench_matvec.c) — original GGUF layout, device op order
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

static void ref_lanes_q6k(const unsigned char *row, const float *x, int cols,
                          float p[32])
{
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
}

/* F16: no scale, no repack — the 256-element step is 512 raw bytes and the
 * per-lane order is the kernel's, elements 32*i+lane for i = 0..7, chained from
 * 0.0f and added into p[lane] (fc_f16_sb_lane_grp2's `*p0 += a0`). half2f above
 * is the same IEEE widening cvt.f32.f16 performs, subnormals included. */
static void ref_lanes_f16(const unsigned char *row, const float *x, int cols,
                          float p[32])
{
    int n_sb = cols >> 8, sb, lane, i;
    for (sb = 0; sb < n_sb; sb++) {
        const unsigned char *blk = row + (size_t)sb * 512;
        const float *xb = x + (sb << 8);
        for (lane = 0; lane < 32; lane++) {
            float acc = 0.0f;
            for (i = 0; i < 8; i++) {
                int el = 32 * i + lane;
                unsigned short h = (unsigned short)(blk[2 * el] |
                                                    (blk[2 * el + 1] << 8));
                acc = fmaf(half2f(h), xb[el], acc);
            }
            p[lane] += acc;
        }
    }
}

/* BF16: identical structure to ref_lanes_f16, and the widen is the whole
 * difference — the value IS the top 16 bits of the fp32 bit pattern, so the
 * replica's conversion is a shift and a memcpy, exactly what the device's
 * __int_as_float(b << 16) does. Nothing rounds on either side, subnormals
 * included, so the dot is bit-exact for the same reason the F16 dot is; bf16's
 * coarser mantissa changes the VALUES being summed, not the exactness of the
 * comparison. */
static float bf162f(unsigned short b)
{
    unsigned int bits = (unsigned int)b << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void ref_lanes_bf16(const unsigned char *row, const float *x, int cols,
                           float p[32])
{
    int n_sb = cols >> 8, sb, lane, i;
    for (sb = 0; sb < n_sb; sb++) {
        const unsigned char *blk = row + (size_t)sb * 512;
        const float *xb = x + (sb << 8);
        for (lane = 0; lane < 32; lane++) {
            float acc = 0.0f;
            for (i = 0; i < 8; i++) {
                int el = 32 * i + lane;
                unsigned short b = (unsigned short)(blk[2 * el] |
                                                    (blk[2 * el + 1] << 8));
                acc = fmaf(bf162f(b), xb[el], acc);
            }
            p[lane] += acc;
        }
    }
}

/* The uint2 kernels' lane map, and the reason they need a replica of their own:
 * lane l owns the CONTIGUOUS runs {4l..4l+3} and {128+4l..128+4l+3} rather than
 * the scalar map's stride-32 set {32i+l}. Same 256 products per super-block,
 * different partition across lanes, so the per-lane fmaf chains and the shfl
 * tree associate differently and the two kernels' outputs are not bit-equal.
 * Comparing a vec kernel against the scalar replica is therefore a REAL failure
 * this test must be able to see, not a tolerance question — which is what
 * arm_vec_map_red below gates. */
static void ref_lanes_raw16_vec(int t, const unsigned char *row,
                                const float *x, int cols, float p[32])
{
    int n_sb = cols >> 8, sb, lane, h, j;
    for (sb = 0; sb < n_sb; sb++) {
        const unsigned char *blk = row + (size_t)sb * 512;
        const float *xb = x + (sb << 8);
        for (lane = 0; lane < 32; lane++) {
            float acc = 0.0f;
            for (h = 0; h < 2; h++)
                for (j = 0; j < 4; j++) {
                    int el = 128 * h + 4 * lane + j;
                    unsigned short w = (unsigned short)(blk[2 * el] |
                                                        (blk[2 * el + 1] << 8));
                    acc = fmaf(t == GG_F16 ? half2f(w) : bf162f(w), xb[el],
                               acc);
                }
            p[lane] += acc;
        }
    }
}

/* Q8_0 over the expert super-block (fc_moe_repack_weight), replicated from the
 * ORIGINAL 34-B blocks: lane l owns {4l..4l+3} (block l>>3) and
 * {128+4l..128+4l+3} (block 4+(l>>3)); each run is a 4-fmaf chain from 0 and
 * is then folded in as p = fmaf(d, run, p). g_q80_order selects a DIFFERENT
 * order for the red arm (1: dense q80's t = d*q, p = fmaf(t, x, p)) and the
 * MOE_TEST_RED_Q80=1 manual red (2: quants read unsigned). */
static int g_q80_order;

static void ref_lanes_q80(const unsigned char *row, const float *x, int cols,
                          float p[32])
{
    int n_sb = cols >> 8, sb, lane, h, j;
    for (sb = 0; sb < n_sb; sb++) {
        const unsigned char *raw = row + (size_t)sb * 272;
        const float *xb = x + (sb << 8);
        for (lane = 0; lane < 32; lane++)
            for (h = 0; h < 2; h++) {
                int el0 = 128 * h + 4 * lane;
                const unsigned char *blk = raw + 34 * (el0 >> 5);
                float d = half2f((unsigned short)(blk[0] | (blk[1] << 8)));
                float a = 0.0f;
                for (j = 0; j < 4; j++) {
                    int el = el0 + j;
                    unsigned char qb = blk[2 + (el & 31)];
                    float q = (g_q80_order == 2) ? (float)qb
                                                 : (float)(signed char)qb;
                    if (g_q80_order == 1)
                        p[lane] = fmaf(d * q, xb[el], p[lane]);
                    else
                        a = fmaf(q, xb[el], a);
                }
                if (g_q80_order != 1)
                    p[lane] = fmaf(d, a, p[lane]);
            }
    }
}

static void ref_lanes(int t, const unsigned char *row, const float *x,
                      int cols, float p[32])
{
    if (t == GG_Q80)
        ref_lanes_q80(row, x, cols, p);
    else if (is_raw16(t) && g_vec_map)
        ref_lanes_raw16_vec(t, row, x, cols, p);
    else if (t == GG_Q3K)  ref_lanes_q3k(row, x, cols, p);
    else if (t == GG_Q6K)  ref_lanes_q6k(row, x, cols, p);
    else if (t == GG_F16)  ref_lanes_f16(row, x, cols, p);
    else if (t == GG_BF16) ref_lanes_bf16(row, x, cols, p);
    else                   ref_lanes_q4k(row, x, cols, p);
}

static float ref_row(int t, const unsigned char *row, const float *x, int cols)
{
    float p[32] = { 0 };
    ref_lanes(t, row, x, cols, p);
    return tree32(p);
}

/* One fc_moe_gateup_* output: silu(gate.row . x) * (up.row . x). */
static float ref_gateup(int t, const unsigned char *grow,
                        const unsigned char *urow, const float *x, int cols)
{
    float g = ref_row(t, grow, x, cols);
    float u = ref_row(t, urow, x, cols);
    return (g / (1.0f + expf(-g))) * u;
}

static int ulp_diff(float a, float b)
{
    int32_t ia, ib;
    memcpy(&ia, &a, 4);
    memcpy(&ib, &b, 4);
    if (ia < 0) ia = (int32_t)0x80000000 - ia;
    if (ib < 0) ib = (int32_t)0x80000000 - ib;
    return abs(ia - ib);
}

/* ------------------------------------------------------------------
 * The arms
 * ------------------------------------------------------------------ */

#define ULP_TOL 8                  /* silu's expf only; the dots are exact */

/* Shared per-arm fixture: k selected experts out of n_expert, ff rows, cols
 * inner dim, ntok tokens of x (token 0 is the single-token arm's x). */
typedef struct {
    int t, n_expert, ff, cols, k, ntok, stride;
    unsigned char *gr, *ur;        /* raw GGUF bytes, [n_expert*ff][raw] */
    size_t raw_row;
    int32_t *idx;                  /* [ntok*k] */
    float *xh;                     /* [ntok*cols] */
    fc_buffer *gate, *up, *x, *tidx, *hid;
} fixture;

static int fixture_init(fixture *f, int t, int n_expert, int ff, int cols,
                        int k, int ntok)
{
    int i;
    memset(f, 0, sizeof(*f));
    f->t = t; f->n_expert = n_expert; f->ff = ff; f->cols = cols;
    f->k = k; f->ntok = ntok;
    f->raw_row = raw_block_bytes(t) * (size_t)(cols / 256);
    f->gr = rand_bytes((size_t)n_expert * ff * f->raw_row);
    f->ur = rand_bytes((size_t)n_expert * ff * f->raw_row);
    sanitize_quant(t, f->gr, (size_t)n_expert * ff * f->raw_row);
    sanitize_quant(t, f->ur, (size_t)n_expert * ff * f->raw_row);
    f->idx = malloc((size_t)ntok * k * sizeof(int32_t));
    for (i = 0; i < ntok * k; i++)
        f->idx[i] = (int32_t)(rand() % n_expert);
    f->xh = rand_floats((size_t)ntok * cols);

    f->gate = upload_weight(t, f->gr, (long)n_expert * ff, cols, &f->stride);
    f->up = upload_weight(t, f->ur, (long)n_expert * ff, cols, &f->stride);
    f->x = fc_buffer_from_ptr(f->xh, (size_t)ntok * cols * sizeof(float));
    f->tidx = fc_buffer_from_ptr(f->idx, (size_t)ntok * k * sizeof(int32_t));
    f->hid = fc_buffer_alloc((size_t)ntok * k * ff * sizeof(float));
    return (f->gate && f->up && f->x && f->tidx && f->hid) ? 0 : -1;
}

static void fixture_free(fixture *f)
{
    fc_buffer_free(f->gate); fc_buffer_free(f->up); fc_buffer_free(f->x);
    fc_buffer_free(f->tidx); fc_buffer_free(f->hid);
    free(f->gr); free(f->ur); free(f->idx); free(f->xh);
}

/* Compare hid[tok] against the CPU replica for every (slot, row). */
static int check_token(const fixture *f, const float *hid_tok, int tok,
                       const char *what)
{
    int s, r, nmis = 0, maxulp = 0;
    for (s = 0; s < f->k; s++) {
        int e = f->idx[tok * f->k + s];
        for (r = 0; r < f->ff; r++) {
            const unsigned char *grow =
                f->gr + ((size_t)e * f->ff + r) * f->raw_row;
            const unsigned char *urow =
                f->ur + ((size_t)e * f->ff + r) * f->raw_row;
            float want = ref_gateup(f->t, grow, urow,
                                    f->xh + (size_t)tok * f->cols, f->cols);
            float got = hid_tok[(size_t)s * f->ff + r];
            int u = ulp_diff(want, got);
            if (u > maxulp) maxulp = u;
            if (u > ULP_TOL) {
                if (nmis < 3)
                    printf("    %s tok %d slot %d row %d: ref %.9g (%08x) "
                           "gpu %.9g (%08x)\n", what, tok, s, r, (double)want,
                           *(const unsigned *)&want, (double)got,
                           *(const unsigned *)&got);
                nmis++;
            }
        }
    }
    printf("  %-28s %s (mismatches %d, max ulp %d, tol %d)\n", what,
           nmis ? "FAIL" : "ok", nmis, maxulp, ULP_TOL);
    return nmis ? 1 : 0;
}

/* Arms 1-3: single-token gate+up for one quant/shape. */
static int arm_gateup(fc_cuda_ctx *ctx, int t, int cols, const char *what)
{
    fixture f;
    float *hid_h;
    int rc, bad;

    if (fixture_init(&f, t, 16, 512, cols, 8, 1) != 0) {
        printf("  %-28s FAIL (fixture)\n", what);
        return 1;
    }
    rc = fc_moe_gateup_ctx(ctx, t, f.gate, f.up, f.x, f.tidx, f.hid, f.k, f.ff,
                           f.cols, f.stride);
    fc_sync();
    if (rc != 0) {
        printf("  %-28s FAIL (dispatch returned %d)\n", what, rc);
        fixture_free(&f);
        return 1;
    }
    hid_h = malloc((size_t)f.k * f.ff * sizeof(float));
    fc_buffer_to_cpu(f.hid, hid_h);
    bad = check_token(&f, hid_h, 0, what);
    free(hid_h);
    fixture_free(&f);
    return bad;
}

/* Arm 4: the batched kernel — every token against the replica, and against
 * the single-token kernel run on that token's slices (bit-exact). */
static int arm_gateup_nx(fc_cuda_ctx *ctx, int t, const char *what)
{
    fixture f;
    float *hid_h, *one_h;
    int rc, bad = 0, tok, i, n;

    if (fixture_init(&f, t, 16, 512, 2048, 8, 3) != 0) {
        printf("  %-28s FAIL (fixture)\n", what);
        return 1;
    }
    n = f.k * f.ff;
    rc = fc_moe_gateup_nx_ctx(ctx, t, f.gate, f.up, f.x, f.tidx, f.hid, f.k,
                              f.ff, f.cols, f.stride, f.ntok);
    fc_sync();
    if (rc != 0) {
        printf("  %-28s FAIL (dispatch returned %d)\n", what, rc);
        fixture_free(&f);
        return 1;
    }
    hid_h = malloc((size_t)f.ntok * n * sizeof(float));
    one_h = malloc((size_t)n * sizeof(float));
    fc_buffer_to_cpu(f.hid, hid_h);
    for (tok = 0; tok < f.ntok; tok++)
        bad += check_token(&f, hid_h + (size_t)tok * n, tok, what);

    /* Per-token bit-exactness vs the single-token kernel on the same slices. */
    for (tok = 0; tok < f.ntok && !bad; tok++) {
        fc_buffer *x1 = fc_buffer_from_ptr(f.xh + (size_t)tok * f.cols,
                                           (size_t)f.cols * sizeof(float));
        fc_buffer *i1 = fc_buffer_from_ptr(f.idx + (size_t)tok * f.k,
                                           (size_t)f.k * sizeof(int32_t));
        fc_buffer *h1 = fc_buffer_alloc((size_t)n * sizeof(float));
        int nmis = 0;
        fc_moe_gateup_ctx(ctx, t, f.gate, f.up, x1, i1, h1, f.k, f.ff, f.cols,
                          f.stride);
        fc_sync();
        fc_buffer_to_cpu(h1, one_h);
        for (i = 0; i < n; i++)
            if (memcmp(&one_h[i], &hid_h[(size_t)tok * n + i], 4) != 0)
                nmis++;
        printf("  %-28s %s (tok %d, mismatches %d)\n", "  nx vs 1-token",
               nmis ? "FAIL" : "bit-exact", tok, nmis);
        bad += nmis ? 1 : 0;
        fc_buffer_free(x1); fc_buffer_free(i1); fc_buffer_free(h1);
    }
    free(hid_h); free(one_h);
    fixture_free(&f);
    return bad;
}

/* Arm 5: an unsupported quant must be REFUSED, not decoded as something else.
 * hid is pre-filled with a canary; a launch of any kind would overwrite it. */
static int arm_fail_closed(fc_cuda_ctx *ctx)
{
    fixture f;
    float *canary, *back;
    int n, i, bad = 0, rc, rcn;

    if (fixture_init(&f, GG_Q4K, 16, 512, 2048, 8, 1) != 0) {
        printf("  %-28s FAIL (fixture)\n", "fail-closed");
        return 1;
    }
    n = f.k * f.ff;
    canary = malloc((size_t)n * sizeof(float));
    back = malloc((size_t)n * sizeof(float));
    for (i = 0; i < n; i++)
        canary[i] = -12345.0f;
    fc_buffer_from_cpu(f.hid, canary);
    fc_sync();

    printf("  (one '[fc_cuda] moe_gateup: unsupported quant' line each is "
           "EXPECTED below)\n");
    rc = fc_moe_gateup_ctx(ctx, GG_Q5K, f.gate, f.up, f.x, f.tidx, f.hid, f.k,
                           f.ff, f.cols, f.stride);
    rcn = fc_moe_gateup_nx_ctx(ctx, GG_Q5K, f.gate, f.up, f.x, f.tidx, f.hid,
                               f.k, f.ff, f.cols, f.stride, 1);
    fc_sync();
    fc_buffer_to_cpu(f.hid, back);
    for (i = 0; i < n; i++)
        if (memcmp(&back[i], &canary[i], 4) != 0)
            bad++;
    printf("  %-28s %s (rc %d / nx rc %d, hid writes %d)\n",
           "fail-closed q5k refused",
           (rc == -1 && rcn == -1 && bad == 0) ? "ok" : "FAIL", rc, rcn, bad);

    /* And the availability probe must agree with the dispatch. */
    {
        int a3 = fc_moe_gateup_quant_available_ctx(ctx, GG_Q3K);
        int a4 = fc_moe_gateup_quant_available_ctx(ctx, GG_Q4K);
        int a5 = fc_moe_gateup_quant_available_ctx(ctx, GG_Q5K);
        int af = fc_moe_gateup_quant_available_ctx(ctx, GG_F16);
        int ab = fc_moe_gateup_quant_available_ctx(ctx, GG_BF16);
        int a8 = fc_moe_gateup_quant_available_ctx(ctx, GG_Q80);
        int ok = a3 && a4 && af && ab && a8 && !a5;
        printf("  %-28s %s (q3k %d q4k %d f16 %d bf16 %d q80 %d q5k %d)\n",
               "availability probe", ok ? "ok" : "FAIL", a3, a4, af, ab, a8,
               a5);
        if (!ok) bad++;
    }

    free(canary); free(back);
    fixture_free(&f);
    return (rc == -1 && rcn == -1 && bad == 0) ? 0 : 1;
}

/* ------------------------------------------------------------------
 * Arm 5: grouped down + weighted accumulate
 * ------------------------------------------------------------------ */

/* The down kernels' wpp launch arg — mirrors fc_moe_down_accum_ctx's env/
 * default resolution so the replica reproduces the launched shape. */
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

/* Replica of one fc_moe_down_* output row in the kernel's exact op order
 * (from bench_matvec.c's ref_moe_down). hid must be the DEVICE-computed hid so
 * this isolates the down kernel. */
static float ref_down_row(int t, const unsigned char *draw, size_t raw_row,
                          size_t raw_expert, const float *hid,
                          const int32_t *idx, const float *wt, int k, int row,
                          int ff, int wpp)
{
    float total = 0.0f;
    int wk, s, l;
    for (wk = 0; wk < wpp; wk++) {
        float tl[32] = { 0 };
        for (s = wk; s < k; s += wpp) {
            const unsigned char *rp = draw + (size_t)idx[s] * raw_expert +
                                      (size_t)row * raw_row;
            float p[32] = { 0 };
            ref_lanes(t, rp, hid + (size_t)s * ff, ff, p);
            for (l = 0; l < 32; l++)
                tl[l] = fmaf(wt[s], p[l], tl[l]);
        }
        total += tree32(tl);
    }
    return total;
}

/* rows = n_embd (2048), ff = 512 — the real per-layer down shape.
 *
 * Returns the count of output rows that differ from the replica bit-for-bit.
 * The CALLER decides which way that reads: every parity arm wants zero, and
 * arm_vec_map_red wants a non-zero, because it deliberately scores the vec
 * kernel against the scalar replica.
 *
 * replica_vec picks the replica's lane map independently of the kernel's, which
 * is the only reason this knob exists — everywhere except that red arm it is
 * g_vec_map, i.e. the same source of truth the dispatch used. */
/* Selected experts per token in the down arms. 8 (Qwen3.6) gives every warp an
 * even slot count at wpp 2; 7 leaves one warp a lone last slot, the q80 down
 * kernel's plain-loop remainder after its two-slots-at-a-time walk. */
static int g_down_k = 8;

static int down_mismatches(fc_cuda_ctx *ctx, int t, int nx, const char *what,
                           int replica_vec)
{
    const int n_expert = 16, rows = 2048, ff = 512, k = g_down_k;
    const int ntok = nx ? 3 : 1;
    size_t raw_row = raw_block_bytes(t) * (size_t)(ff / 256);
    size_t raw_expert = (size_t)rows * raw_row;
    unsigned char *dr = rand_bytes((size_t)n_expert * raw_expert);
    int32_t *idx = malloc((size_t)ntok * k * sizeof(int32_t));
    float *wt = malloc((size_t)ntok * k * sizeof(float));
    float *hid_h = malloc((size_t)ntok * k * ff * sizeof(float));
    float *out_h = malloc((size_t)ntok * rows * sizeof(float));
    float *rnd;
    fc_buffer *dn, *hid, *tidx, *twt, *out;
    int stride = 0, i, tok, r, nmis = 0, maxulp = 0, wpp = moe_down_wpp();
    int saved_map = g_vec_map;

    sanitize_quant(t, dr, (size_t)n_expert * raw_expert);
    for (i = 0; i < ntok * k; i++) {
        idx[i] = (int32_t)(rand() % n_expert);
        wt[i] = (float)rand() / (float)RAND_MAX;
    }
    rnd = rand_floats((size_t)ntok * k * ff);
    memcpy(hid_h, rnd, (size_t)ntok * k * ff * sizeof(float));
    free(rnd);

    dn = upload_weight(t, dr, (long)n_expert * rows, ff, &stride);
    hid = fc_buffer_from_ptr(hid_h, (size_t)ntok * k * ff * sizeof(float));
    tidx = fc_buffer_from_ptr(idx, (size_t)ntok * k * sizeof(int32_t));
    twt = fc_buffer_from_ptr(wt, (size_t)ntok * k * sizeof(float));
    out = fc_buffer_alloc((size_t)ntok * rows * sizeof(float));
    if (!dn || !hid || !tidx || !twt || !out) {
        /* -1, not a mismatch count: arm_vec_map_red PASSES on a non-zero count,
         * so a fixture failure reported as "1 mismatch" would read as a pass. */
        printf("  %-28s FAIL (fixture)\n", what);
        return -1;
    }

    if (nx)
        fc_moe_down_accum_nx_ctx(ctx, t, dn, hid, tidx, twt, out, k, rows, ff,
                                 stride, ntok);
    else
        fc_moe_down_accum_ctx(ctx, t, dn, hid, tidx, twt, out, k, rows, ff,
                              stride);
    fc_sync();
    fc_buffer_to_cpu(out, out_h);

    g_vec_map = replica_vec;           /* replica only — the launch is done */
    for (tok = 0; tok < ntok; tok++)
        for (r = 0; r < rows; r++) {
            float want = ref_down_row(t, dr, raw_row, raw_expert,
                                      hid_h + (size_t)tok * k * ff,
                                      idx + (size_t)tok * k,
                                      wt + (size_t)tok * k, k, r, ff, wpp);
            float got = out_h[(size_t)tok * rows + r];
            if (memcmp(&want, &got, 4) != 0) {
                int u = ulp_diff(want, got);
                if (u > maxulp) maxulp = u;
                if (nmis < 3)
                    printf("    %s tok %d row %d: ref %.9g (%08x) gpu %.9g "
                           "(%08x)\n", what, tok, r, (double)want,
                           *(const unsigned *)&want, (double)got,
                           *(const unsigned *)&got);
                nmis++;
            }
        }
    g_vec_map = saved_map;
    printf("  %-28s wpp %d, mismatches %d, max ulp %d\n", what, wpp, nmis,
           maxulp);

    fc_buffer_free(dn); fc_buffer_free(hid); fc_buffer_free(tidx);
    fc_buffer_free(twt); fc_buffer_free(out);
    free(dr); free(idx); free(wt); free(hid_h); free(out_h);
    return nmis;
}

/* Parity: bit-exact, zero tolerance. No expf anywhere in the down path, and
 * every fp op on both sides is an fp32 fmaf in the same order, so any tolerance
 * at all would only hide a real difference. */
static int arm_down(fc_cuda_ctx *ctx, int t, int nx, const char *what)
{
    int nmis = down_mismatches(ctx, t, nx, what, g_vec_map);
    if (nmis < 0)
        return 1;                          /* fixture failure, already named */
    printf("  %-28s %s\n", what, nmis ? "FAIL" : "bit-exact");
    return nmis ? 1 : 0;
}

/* The red proof the vec arms need, run as a normal arm so it can never rot.
 *
 * Every vec arm compares a vec kernel against the vec replica and expects
 * bit-equality; that only means something if the vec replica is DISCRIMINATING
 * — if scoring the same kernel output against the SCALAR replica goes red. It
 * has to be checked rather than assumed, because the two maps sum the same 256
 * products and a reassociation can land back on the same fp32 value. (A
 * previous pass on these kernels tried exactly one such permutation as a red
 * proof and it stayed green for that reason; this arm is that mistake turned
 * into a gate.) Passes when the mismatch count is NON-ZERO. */
static int arm_vec_map_red(fc_cuda_ctx *ctx, int t, const char *what)
{
    int nmis = down_mismatches(ctx, t, 0, what, 0);
    if (nmis < 0)
        return 1;                          /* fixture failure, already named */
    printf("  %-28s %s (expects mismatches, got %d)\n", what,
           nmis ? "discriminates" : "FAIL", nmis);
    return nmis ? 0 : 1;
}

/* Red proof for the Q8_0 replica's op order, kept as a normal arm: the same
 * kernel output scored against dense q80's per-element order (t = d*q first)
 * must MISMATCH, or the bit-exact q80 arms would not be discriminating.
 * Passes when the mismatch count is non-zero. */
static int arm_q80_order_red(fc_cuda_ctx *ctx)
{
    int saved = g_q80_order, nmis;
    g_q80_order = 1;
    nmis = down_mismatches(ctx, GG_Q80, 0, "q80 order red (dense order)",
                           g_vec_map);
    g_q80_order = saved;
    if (nmis < 0)
        return 1;
    printf("  %-28s %s (expects mismatches, got %d)\n", "q80 order is not dense",
           nmis ? "discriminates" : "FAIL", nmis);
    return nmis ? 0 : 1;
}

/* The down dispatch has no return code (it predates one), so its fail-closed
 * behaviour is proved the only way it is observable: a canary output that a
 * launch of any kind would overwrite. */
static int arm_down_fail_closed(fc_cuda_ctx *ctx)
{
    const int n_expert = 4, rows = 256, ff = 512, k = 4;
    size_t raw_row = raw_block_bytes(GG_Q6K) * (size_t)(ff / 256);
    unsigned char *dr = rand_bytes((size_t)n_expert * rows * raw_row);
    int32_t idx[4] = { 0, 1, 2, 3 };
    float wt[4] = { .25f, .25f, .25f, .25f };
    float *hid_h = rand_floats((size_t)k * ff);
    float *canary = malloc((size_t)rows * sizeof(float));
    float *back = malloc((size_t)rows * sizeof(float));
    fc_buffer *dn, *hid, *tidx, *twt, *out;
    int stride = 0, i, bad = 0, a3, a4, a6, a5, af, ab, a8, ok;

    sanitize_quant(GG_Q6K, dr, (size_t)n_expert * rows * raw_row);
    dn = upload_weight(GG_Q6K, dr, (long)n_expert * rows, ff, &stride);
    hid = fc_buffer_from_ptr(hid_h, (size_t)k * ff * sizeof(float));
    tidx = fc_buffer_from_ptr(idx, sizeof(idx));
    twt = fc_buffer_from_ptr(wt, sizeof(wt));
    out = fc_buffer_alloc((size_t)rows * sizeof(float));
    for (i = 0; i < rows; i++)
        canary[i] = -98765.0f;
    fc_buffer_from_cpu(out, canary);
    fc_sync();
    fc_moe_down_accum_ctx(ctx, GG_Q5K, dn, hid, tidx, twt, out, k, rows, ff,
                          stride);
    fc_moe_down_accum_nx_ctx(ctx, GG_Q5K, dn, hid, tidx, twt, out, k, rows, ff,
                             stride, 1);
    fc_sync();
    fc_buffer_to_cpu(out, back);
    for (i = 0; i < rows; i++)
        if (memcmp(&back[i], &canary[i], 4) != 0)
            bad++;
    printf("  %-28s %s (out writes %d)\n", "down fail-closed q5k",
           bad ? "FAIL" : "ok", bad);

    a3 = fc_moe_down_quant_available_ctx(ctx, GG_Q3K);
    a4 = fc_moe_down_quant_available_ctx(ctx, GG_Q4K);
    a6 = fc_moe_down_quant_available_ctx(ctx, GG_Q6K);
    a5 = fc_moe_down_quant_available_ctx(ctx, GG_Q5K);
    af = fc_moe_down_quant_available_ctx(ctx, GG_F16);
    ab = fc_moe_down_quant_available_ctx(ctx, GG_BF16);
    a8 = fc_moe_down_quant_available_ctx(ctx, GG_Q80);
    ok = a3 && a4 && a6 && af && ab && a8 && !a5;
    printf("  %-28s %s (q3k %d q4k %d q6k %d f16 %d bf16 %d q80 %d q5k %d)\n",
           "down avail probe", ok ? "ok" : "FAIL", a3, a4, a6, af, ab, a8, a5);

    fc_buffer_free(dn); fc_buffer_free(hid); fc_buffer_free(tidx);
    fc_buffer_free(twt); fc_buffer_free(out);
    free(dr); free(hid_h); free(canary); free(back);
    return (bad == 0 && ok) ? 0 : 1;
}

/* --bench: ms/launch for each quant at the real per-layer shape, so the cost
 * of the quant CHOICE reads directly. Not part of the gate — Q4_K is EXPECTED
 * to be slower than Q3_K here (more bytes, and this box is instruction-bound);
 * the point of the K-quant kernels is capability, not throughput.
 *
 * The 16-bit arms are the ones where the number is an ANSWER rather than a
 * datum, and they only compare cleanly because the fixture is the same shape
 * for all of them: 8 slots x 512 rows x 4096 B of gate + the same of up =
 * 33.55 MB touched per launch, against a 1.5 MB L2, so every arm streams from
 * DRAM. Reading them in pairs isolates one variable each: f16 vs bf16 is the
 * DECODE at identical traffic and identical map, and scalar vs vec is the LOAD
 * WIDTH at identical traffic and identical decode.
 *
 * The benches force the k selected experts DISTINCT, which the parity fixtures
 * deliberately do not. Drawing 8 of 16 with replacement repeats ~1.5 slots, and
 * a repeated expert's second reader can be served from L2 — harmless to a
 * scalar-vs-vec comparison (both arms pay it) but it makes the ABSOLUTE GB/s
 * ambiguous by up to 19%, and the absolute number is the one that says whether
 * a kernel is anywhere near the die's 240 GB/s. Distinct experts make nominal
 * bytes and DRAM bytes the same number. */
static void bench_distinct_experts(int32_t *idx, int k, int n_expert)
{
    int i, j;
    for (i = 0; i < k && i < n_expert; i++) {
        int e, dup;
        do {
            e = rand() % n_expert;
            for (dup = 0, j = 0; j < i; j++)
                if (idx[j] == e)
                    dup = 1;
        } while (dup);
        idx[i] = (int32_t)e;
    }
}

static void bench_gateup(fc_cuda_ctx *ctx, int t, const char *nm, int iters)
{
    fixture f;
    fc_buffer *tidx_d;
    double t0, dt;
    int i;
    if (fixture_init(&f, t, 16, 512, 2048, 8, 1) != 0)
        return;
    bench_distinct_experts(f.idx, f.k, f.n_expert);
    tidx_d = fc_buffer_from_ptr(f.idx, (size_t)f.k * sizeof(int32_t));
    fc_buffer_free(f.tidx);
    f.tidx = tidx_d;
    fc_moe_gateup_ctx(ctx, t, f.gate, f.up, f.x, f.tidx, f.hid, f.k, f.ff,
                      f.cols, f.stride);
    fc_sync();
    t0 = now_s();
    for (i = 0; i < iters; i++)
        fc_moe_gateup_ctx(ctx, t, f.gate, f.up, f.x, f.tidx, f.hid, f.k, f.ff,
                          f.cols, f.stride);
    fc_sync();
    dt = (now_s() - t0) / iters;
    printf("  gateup %-10s 8x512x2048  %8.4f ms  %7.2f GB/s (dev bytes)\n", nm,
           dt * 1e3, 2.0 * f.k * f.ff * f.stride / dt / 1e9);
    fixture_free(&f);
}

/* The same measurement for the down kernel: it reads its weights from global
 * with __ldg and its x from global too, where gateup stages x in shared. If the
 * load width matters at all it should show in both, and if it shows in only one
 * of them that localises the effect. */
static void bench_down(fc_cuda_ctx *ctx, int t, const char *nm, int iters)
{
    const int n_expert = 16, rows = 2048, ff = 512, k = 8;
    size_t raw_row = raw_block_bytes(t) * (size_t)(ff / 256);
    size_t raw_expert = (size_t)rows * raw_row;
    unsigned char *dr = rand_bytes((size_t)n_expert * raw_expert);
    int32_t *idx = malloc((size_t)k * sizeof(int32_t));
    float *wt = malloc((size_t)k * sizeof(float));
    float *hid_h = rand_floats((size_t)k * ff);
    fc_buffer *dn, *hid, *tidx, *twt, *out;
    int stride = 0, i;
    double t0, dt;

    sanitize_quant(t, dr, (size_t)n_expert * raw_expert);
    bench_distinct_experts(idx, k, n_expert);
    for (i = 0; i < k; i++)
        wt[i] = (float)rand() / (float)RAND_MAX;
    dn = upload_weight(t, dr, (long)n_expert * rows, ff, &stride);
    hid = fc_buffer_from_ptr(hid_h, (size_t)k * ff * sizeof(float));
    tidx = fc_buffer_from_ptr(idx, (size_t)k * sizeof(int32_t));
    twt = fc_buffer_from_ptr(wt, (size_t)k * sizeof(float));
    out = fc_buffer_alloc((size_t)rows * sizeof(float));
    if (dn && hid && tidx && twt && out) {
        fc_moe_down_accum_ctx(ctx, t, dn, hid, tidx, twt, out, k, rows, ff,
                              stride);
        fc_sync();
        t0 = now_s();
        for (i = 0; i < iters; i++)
            fc_moe_down_accum_ctx(ctx, t, dn, hid, tidx, twt, out, k, rows, ff,
                                  stride);
        fc_sync();
        dt = (now_s() - t0) / iters;
        printf("  down   %-10s 8x2048x512  %8.4f ms  %7.2f GB/s (dev bytes)\n",
               nm, dt * 1e3, (double)k * rows * stride / dt / 1e9);
    }
    fc_buffer_free(dn); fc_buffer_free(hid); fc_buffer_free(tidx);
    fc_buffer_free(twt); fc_buffer_free(out);
    free(dr); free(idx); free(wt); free(hid_h);
}

/* Both 16-bit formats, both load widths, gate+up and down. The vec arms run
 * under set_vec_mode(1) because that is the only thing that selects them. */
static void bench_raw16_matrix(fc_cuda_ctx *ctx, int iters)
{
    set_vec_mode(0);
    bench_gateup(ctx, GG_F16, "f16", iters);
    bench_gateup(ctx, GG_BF16, "bf16", iters);
    set_vec_mode(1);
    bench_gateup(ctx, GG_F16, "f16-vec", iters);
    bench_gateup(ctx, GG_BF16, "bf16-vec", iters);
    set_vec_mode(0);
    bench_down(ctx, GG_F16, "f16", iters);
    bench_down(ctx, GG_BF16, "bf16", iters);
    set_vec_mode(1);
    bench_down(ctx, GG_F16, "f16-vec", iters);
    bench_down(ctx, GG_BF16, "bf16-vec", iters);
    set_vec_mode(0);
}

int main(int argc, char **argv)
{
    fc_cuda_ctx *ctx;
    int fails = 0;
    int bench = (argc > 1 && strcmp(argv[1], "--bench") == 0);

    if (fc_init() != 0) {
        fprintf(stderr, "no CUDA\n");
        return 1;
    }
    ctx = fc_default_ctx();
    srand(1234);
    host_denormals_on();

    printf("== grouped MoE experts: quant dispatch + CPU-replica parity ==\n");
    fails += arm_host_denormals();   /* every arm below depends on this one */
    /* The 2-byte formats first — they are the top of the ladder and the reason
     * these arms exist. BF16 leads because it is the CONTROL: the checkpoints
     * ship bf16, so it is the only format carrying unrounded weights. cols 2048
     * is one staging chunk, 4096 is two, and the _nx arms additionally pin each
     * batched kernel bit-exact against its single-token twin. */
    set_vec_mode(0);
    fails += arm_gateup(ctx, GG_BF16, 2048, "bf16 gateup 2048");
    fails += arm_gateup(ctx, GG_BF16, 4096, "bf16 gateup 4096 (2 chunks)");
    fails += arm_gateup_nx(ctx, GG_BF16, "bf16 gateup nx");
    fails += arm_down(ctx, GG_BF16, 0, "bf16 down");
    fails += arm_down(ctx, GG_BF16, 1, "bf16 down nx");
    fails += arm_gateup(ctx, GG_F16, 2048, "f16 gateup 2048");
    fails += arm_gateup(ctx, GG_F16, 4096, "f16 gateup 4096 (2 chunks)");
    fails += arm_gateup_nx(ctx, GG_F16, "f16 gateup nx");
    fails += arm_down(ctx, GG_F16, 0, "f16 down");
    fails += arm_down(ctx, GG_F16, 1, "f16 down nx");

    /* Same arms against the uint2 kernels, each scored on the vec replica —
     * then arm_vec_map_red proves that replica can actually reject the scalar
     * grouping, without which the four arms above it would be vacuous. */
    set_vec_mode(1);
    fails += arm_gateup(ctx, GG_BF16, 2048, "bf16-vec gateup 2048");
    fails += arm_gateup_nx(ctx, GG_BF16, "bf16-vec gateup nx");
    fails += arm_down(ctx, GG_BF16, 0, "bf16-vec down");
    fails += arm_gateup(ctx, GG_F16, 2048, "f16-vec gateup 2048");
    fails += arm_gateup_nx(ctx, GG_F16, "f16-vec gateup nx");
    fails += arm_down(ctx, GG_F16, 0, "f16-vec down");
    fails += arm_vec_map_red(ctx, GG_BF16, "bf16-vec map is not scalar");
    fails += arm_vec_map_red(ctx, GG_F16, "f16-vec map is not scalar");
    set_vec_mode(0);

    /* Q8_0 on the 288-B expert layout. MOE_TEST_RED_Q80=1 reads the replica's
     * quants unsigned: every q80 arm below must then FAIL (manual red). */
    {
        const char *red = getenv("MOE_TEST_RED_Q80");
        g_q80_order = (red && atoi(red) == 1) ? 2 : 0;
    }
    fails += arm_gateup(ctx, GG_Q80, 2048, "q80 gateup 2048");
    fails += arm_gateup(ctx, GG_Q80, 4096, "q80 gateup 4096 (2 chunks)");
    fails += arm_gateup_nx(ctx, GG_Q80, "q80 gateup nx");
    fails += arm_down(ctx, GG_Q80, 0, "q80 down");
    fails += arm_down(ctx, GG_Q80, 1, "q80 down nx");
    g_down_k = 7;
    fails += arm_down(ctx, GG_Q80, 0, "q80 down k=7");
    fails += arm_down(ctx, GG_Q80, 1, "q80 down nx k=7");
    g_down_k = 8;
    /* The same batched arms on the plain-load _nxp kernels the host picks for
     * prefill-sized calls (FC_MOE_Q80_LDG_MAX_TOK=0: every ntok is "large"). */
    setenv("FC_MOE_Q80_LDG_MAX_TOK", "0", 1);
    fails += arm_gateup_nx(ctx, GG_Q80, "q80 gateup nx plain");
    fails += arm_down(ctx, GG_Q80, 1, "q80 down nx plain");
    g_down_k = 7;
    fails += arm_down(ctx, GG_Q80, 1, "q80 down nx plain k=7");
    g_down_k = 8;
    unsetenv("FC_MOE_Q80_LDG_MAX_TOK");
    fails += arm_q80_order_red(ctx);

    fails += arm_gateup(ctx, GG_Q4K, 2048, "q4k gateup 2048");
    fails += arm_gateup(ctx, GG_Q3K, 2048, "q3k gateup 2048 (regress)");
    fails += arm_gateup(ctx, GG_Q4K, 4096, "q4k gateup 4096 (2 chunks)");
    fails += arm_gateup_nx(ctx, GG_Q4K, "q4k gateup nx");
    fails += arm_down(ctx, GG_Q6K, 0, "q6k down");
    fails += arm_down(ctx, GG_Q6K, 1, "q6k down nx");
    fails += arm_down(ctx, GG_Q4K, 0, "q4k down (regress)");
    fails += arm_down(ctx, GG_Q3K, 0, "q3k down (regress)");
    fails += arm_fail_closed(ctx);
    fails += arm_down_fail_closed(ctx);

    if (bench && !fails) {
        printf("== speed (capability, not a gate) ==\n");
        bench_gateup(ctx, GG_Q3K, "q3k", 50);
        bench_gateup(ctx, GG_Q4K, "q4k", 50);
        bench_gateup(ctx, GG_Q80, "q80", 50);
        bench_down(ctx, GG_Q3K, "q3k", 50);
        bench_down(ctx, GG_Q4K, "q4k", 50);
        bench_down(ctx, GG_Q6K, "q6k", 50);
        bench_down(ctx, GG_Q80, "q80", 50);
        bench_raw16_matrix(ctx, 50);
    }
    printf("%s (%d failing arm%s)\n", fails ? "FAIL" : "PASS", fails,
           fails == 1 ? "" : "s");
    fc_free();
    return fails ? 1 : 0;
}
