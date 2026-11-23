/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_attn_fast_exp.c
 * @brief The attention fast exp (kernels/ft_tensix_sfpu_exp_attention.h, -DATTN_FAST_EXP),
 *   emulated instruction by instruction in fp32 on the host, off hardware.
 *
 *   Each SFPU instruction the kernel's sequence compiles to (12 an element, read off the
 *   SFPI 7.68.0 image) is a host function written from its BlackholeA0 ISA page: SFPSWAP
 *   min/max (sign-magnitude compare, VC gets the max), SFPMAD / SFPADD / SFPMUL (denormal
 *   inputs read as zero, denormal results flushed to signed zero, one rounding), SFPSHFT
 *   (logical left), SFPIADD (two's complement). The constants come from the kernel header
 *   itself (FT_SFPU_EXPA_CONSTANTS_ONLY), so the test runs exactly the numbers the card does.
 *
 *   SFPMAD is only partially fused on Blackhole (the product keeps four extra bits and is not
 *   rounded; SFPMAD.md). Rather than model that, every sweep runs TWICE: fully fused (fmaf)
 *   and fully split (product rounded, then the add), and both must meet the bound. The two
 *   differ by an fp32 ulp, 2^-24 against a 2^-9 bound.
 *
 *   1. EMULATOR CONTROL. The accurate body (ft_sfpu_exp_fp32_accurate, SFPSTOCHRND / SFPCAST /
 *      SFPABS / SFPSETSGN / SFPEXEXP / SFPSETEXP as well) through the same primitives: its
 *      error must be fp32-grade (< 2^-20), or the primitives are wrong and the rest proves
 *      nothing.
 *   2. THE BOUND. The fast exp over a dense sweep of [-88, 0] (2^24 points, every ln2 period
 *      sampled ~35,000 times): max relative error <= 2^-9 wherever exp(a) >= FLT_MIN, and
 *      exactly +0 at and below -87.68 (j = -127) and for the mask value, -FLT_MAX and -inf;
 *      between -87.68 and ln(FLT_MIN) it is 0 or within the bound.
 *   3. exp(0) = c0, and positive inputs up to 88 within the bound (not used, documented).
 *
 *   RED BUILDS, each must FAIL:
 *     make -f Makefile.attnfastexp RED=drop_term run   the emulation drops c2 (degree 1)
 *     make -f Makefile.attnfastexp RED=no_clamp run    no max(a, -88): the mask value wraps
 *
 * @status Added with ATTN_FAST_EXP (Qwen3.8 prefill attention fast-math, card 0112).
 * @issues The emulation is of the sequence as written in the header; that the compiler emits
 *   exactly it is checked by reading the disassembly (header comment), not here.
 * @todo None
 */

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ft_tt_test_report.h"

#define FT_SFPU_EXPA_CONSTANTS_ONLY
#include "kernels/ft_tensix_sfpu_exp_attention.h"

/* The accurate body's constants (kernels/ft_tensix_sfpu_math.h), for the emulator control;
 * that header is SFPU-only and cannot be included here. */
#define ACCURATE_LOG2_E 1.442695f
#define ACCURATE_LN2_HIGH -6.93145752e-1f
#define ACCURATE_LN2_LOW -1.42860677e-6f
#define ACCURATE_POLY_5 1.37805939e-3f
#define ACCURATE_POLY_4 8.37312452e-3f
#define ACCURATE_POLY_3 4.16695364e-2f
#define ACCURATE_POLY_2 1.66664720e-1f
#define ACCURATE_POLY_1 4.99999851e-1f

#define SWEEP_POINTS (1u << 24)
#define SWEEP_LOW -88.0
/* j = round(a * log2e) is -127 at and below this: the scale is exactly 0. */
#define EXACT_ZERO_BELOW -87.68
#define BOUND 0x1p-9
#define CONTROL_BOUND 0x1p-20

typedef enum { kMadFused, kMadSplit } mad_mode;

/* ===================================================================================
 * SFPU primitives, from the ISA pages
 * =================================================================================== */

static uint32_t bits_of(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float float_of(uint32_t bits)
{
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

/** Denormal -> signed zero (SFPMAD reads denormal inputs as zero, flushes denormal results). */
static float flush_denormal(float value)
{
    const uint32_t bits = bits_of(value);

    return (bits & 0x7F800000u) == 0u ? float_of(bits & 0x80000000u) : value;
}

/** SFPMAD: a * b + c, one rounding (fused) or two (split); see the file comment. */
static float sfpu_mad(float a, float b, float c, mad_mode mode)
{
    volatile float product; /* keep the split product a rounded fp32 */

    a = flush_denormal(a);
    b = flush_denormal(b);
    c = flush_denormal(c);
    if (mode == kMadFused)
        return flush_denormal(fmaf(a, b, c));
    product = flush_denormal(a * b);
    return flush_denormal(product + c);
}

/** Sign-magnitude ordering of two 32-bit patterns (SFPSWAP's SignMagIsSmaller). */
static bool sign_magnitude_smaller(uint32_t a, uint32_t b)
{
    const int64_t va = (a & 0x80000000u) ? -(int64_t)(a & 0x7FFFFFFFu) : (int64_t)a;
    const int64_t vb = (b & 0x80000000u) ? -(int64_t)(b & 0x7FFFFFFFu) : (int64_t)b;

    return va < vb;
}

/** SFPSWAP VEC_MIN_MAX with `vd` = the floor and `vc` = a; returns VC (the max). */
static float sfpu_swap_max(float vd, float vc)
{
    const uint32_t c = bits_of(vc);
    const uint32_t d = bits_of(vd);
    /* VD gets the min: swap when VC < VD, or equal and negative. */
    const bool swap = sign_magnitude_smaller(c, d) || (c == d && (c & 0x80000000u));

    return swap ? vd : vc;
}

/** SFPSTOCHRND FP32 -> sign-magnitude int16, round to nearest (ties away), saturating. */
static uint32_t sfpu_round_smag16(float value)
{
    const uint32_t c = bits_of(value);
    uint32_t sign = c & 0x80000000u;
    const int32_t exponent = (int32_t)((c >> 23) & 0xFFu) - 127;
    uint64_t magnitude;

    if (exponent < -1)
        return 0u;
    if (exponent >= 16) {
        magnitude = 32767u;
    } else {
        magnitude = 0x800000u | (c & 0x7FFFFFu);
        magnitude = exponent >= 0 ? magnitude << exponent : magnitude >> -exponent;
        magnitude = (magnitude >> 23) + ((magnitude & 0x7FFFFFu) >= 0x400000u ? 1u : 0u);
        if (magnitude > 32767u)
            magnitude = 32767u;
        if (magnitude == 0u)
            sign = 0u;
    }
    return sign | (uint32_t)magnitude;
}

/** SFPCAST sign-magnitude int32 -> fp32, round to nearest even (exact for |v| <= 32767). */
static float sfpu_cast_smag(uint32_t value)
{
    const float magnitude = (float)(value & 0x7FFFFFFFu);

    return (value & 0x80000000u) ? -magnitude : magnitude;
}

/** SFPABS integer: two's complement negate when the sign bit is set. */
static uint32_t sfpu_int_abs(uint32_t value)
{
    return (value & 0x80000000u) ? 0u - value : value;
}

/** SFPSETSGN from a register: `value` with `sign_source`'s sign bit. */
static uint32_t sfpu_copy_sign(uint32_t value, uint32_t sign_source)
{
    return (value & 0x7FFFFFFFu) | (sign_source & 0x80000000u);
}

/* ===================================================================================
 * The two bodies
 * =================================================================================== */

/** ft_sfpu_exp_fp32_accurate, step for step (the control). */
static float accurate_exp(float a, mad_mode mode)
{
    const float j0 = sfpu_mad(a, ACCURATE_LOG2_E, 0.0f, mode);
    const uint32_t signed_magnitude = sfpu_round_smag16(j0);
    const float j = sfpu_cast_smag(signed_magnitude);
    float f = sfpu_mad(j, ACCURATE_LN2_HIGH, a, mode);
    float r = ACCURATE_POLY_5;
    float y;
    uint32_t i;
    int32_t e;

    f = sfpu_mad(j, ACCURATE_LN2_LOW, f, mode);
    r = sfpu_mad(r, f, ACCURATE_POLY_4, mode);
    r = sfpu_mad(r, f, ACCURATE_POLY_3, mode);
    r = sfpu_mad(r, f, ACCURATE_POLY_2, mode);
    r = sfpu_mad(r, f, ACCURATE_POLY_1, mode);
    i = sfpu_int_abs(signed_magnitude);
    y = sfpu_mad(r, f, 1.0f, mode);
    i = sfpu_copy_sign(i, bits_of(j));
    r = sfpu_mad(y, f, 1.0f, mode);
    y = sfpu_mad(y, (float)INFINITY, 0.0f, mode);
    e = (int32_t)((bits_of(r) >> 23) & 0xFFu) + (int32_t)i;
    if (e - 255 < 0) {
        y = float_of((bits_of(r) & 0x807FFFFFu) | (((uint32_t)e & 0xFFu) << 23));
        if (e - 255 < -254)
            y = 0.0f;
    }
    return y;
}

/** ft_sfpu_exp_attention, instruction for instruction (header comment, THE SEQUENCE). */
static float fast_exp(float a, mad_mode mode)
{
#ifdef ATTN_FAST_EXP_TEST_RED_DROP_TERM
    const float c2 = 0.0f;
#else
    const float c2 = FT_SFPU_EXPA_C2;
#endif
#ifdef ATTN_FAST_EXP_TEST_RED_NO_CLAMP
    const float clamped = sfpu_swap_max(-(float)INFINITY, a); /* a floor that clamps nothing */
#else
    const float clamped = sfpu_swap_max(FT_SFPU_EXPA_FLOOR, a);
#endif
    const float t = sfpu_mad(clamped, FT_SFPU_EXPA_LOG2E, FT_SFPU_EXPA_ROUND_MAGIC, mode);
    const float j = sfpu_mad(-1.0f, FT_SFPU_EXPA_ROUND_MAGIC, t, mode); /* SFPADD, negated */
    const float f = sfpu_mad(j, FT_SFPU_EXPA_NEG_LN2, clamped, mode);
    const float q = sfpu_mad(f, c2, FT_SFPU_EXPA_C1, mode);
    const float p = sfpu_mad(q, f, FT_SFPU_EXPA_C0, mode);
    const uint32_t scale = (bits_of(t) << FT_SFPU_EXPA_EXPONENT_SHIFT) + FT_SFPU_EXPA_ONE_BITS;

    return sfpu_mad(p, float_of(scale), 0.0f, mode);
}

/* ===================================================================================
 * Sweeps
 * =================================================================================== */

typedef struct sweep_result {
    double max_rel;
    double worst_a;
    uint32_t nonzero_below; /* a <= EXACT_ZERO_BELOW but y != +0 */
    uint32_t bad_flush;     /* exp(a) < FLT_MIN, y neither 0 nor within the bound */
} sweep_result;

static double relative_error(float y, double reference)
{
    return fabs((double)y - reference) / reference;
}

static sweep_result sweep(float (*body)(float, mad_mode), mad_mode mode, double low, double high)
{
    sweep_result result = { 0.0, 0.0, 0u, 0u };
    const double flt_min_log = log((double)FLT_MIN);
    uint32_t k;

    for (k = 0; k < SWEEP_POINTS; k++) {
        const float a = (float)(low + (high - low) * (double)k / (double)(SWEEP_POINTS - 1u));
        const float y = body(a, mode);
        const double reference = exp((double)a);

        if ((double)a <= EXACT_ZERO_BELOW) {
            if (bits_of(y) != 0u)
                result.nonzero_below++;
        } else if ((double)a < flt_min_log) {
            if (bits_of(y) != 0u && relative_error(y, reference) > BOUND)
                result.bad_flush++;
        } else {
            const double rel = relative_error(y, reference);

            if (!(rel <= result.max_rel)) { /* NaN counts as worse */
                result.max_rel = isnan(rel) ? INFINITY : rel;
                result.worst_a = (double)a;
            }
        }
    }
    return result;
}

static const char *mode_name(mad_mode mode)
{
    return mode == kMadFused ? "fused" : "split";
}

static void test_control(void)
{
    mad_mode mode;

    ft_tt_test_step("1. emulator control: the accurate body through the same primitives");
    for (mode = kMadFused; mode <= kMadSplit; mode++) {
        const sweep_result r = sweep(accurate_exp, mode, SWEEP_LOW, 0.0);

        ft_tt_test_check_fmt("accurate exp is fp32-grade (< 2^-20)", r.max_rel < CONTROL_BOUND,
                             "%s: max rel %.3e (2^%.2f) at a=%.6f", mode_name(mode), r.max_rel,
                             log2(r.max_rel), r.worst_a);
    }
}

static void test_bound(void)
{
    const float masked[] = { -1.0e30f, -FLT_MAX, -(float)INFINITY, -88.0f, -100.0f };
    mad_mode mode;
    size_t m;

    ft_tt_test_step("2. the fast exp over [-88, 0], %u points, bound 2^-9", SWEEP_POINTS);
    for (mode = kMadFused; mode <= kMadSplit; mode++) {
        const sweep_result r = sweep(fast_exp, mode, SWEEP_LOW, 0.0);

        ft_tt_test_check_fmt("max relative error <= 2^-9 where exp(a) >= FLT_MIN",
                             r.max_rel <= BOUND, "%s: %.4e (2^%.2f) at a=%.6f", mode_name(mode),
                             r.max_rel, log2(r.max_rel), r.worst_a);
        ft_tt_test_check_fmt("exactly +0 at and below -87.68", r.nonzero_below == 0u,
                             "%s: %u nonzero", mode_name(mode), r.nonzero_below);
        ft_tt_test_check_fmt("0 or within the bound down to ln(FLT_MIN)", r.bad_flush == 0u,
                             "%s: %u bad", mode_name(mode), r.bad_flush);
        for (m = 0; m < sizeof(masked) / sizeof(masked[0]); m++) {
            const float y = fast_exp(masked[m], mode);

            ft_tt_test_check_fmt("a masked score's exp is exactly +0", bits_of(y) == 0u,
                                 "%s: exp(%g) = %g (0x%08X)", mode_name(mode), (double)masked[m],
                                 (double)y, bits_of(y));
        }
    }
}

static void test_edges(void)
{
    mad_mode mode;

    ft_tt_test_step("3. exp(0) = c0; positive inputs (unused by attention) up to 88");
    for (mode = kMadFused; mode <= kMadSplit; mode++) {
        const float one = fast_exp(0.0f, mode);
        const sweep_result r = sweep(fast_exp, mode, 0.0, 88.0);

        ft_tt_test_check_fmt("exp(0) == c0", one == FT_SFPU_EXPA_C0, "%s: %.9g", mode_name(mode),
                             (double)one);
        ft_tt_test_check_fmt("positive a up to 88 within the bound", r.max_rel <= BOUND,
                             "%s: %.4e at a=%.4f", mode_name(mode), r.max_rel, r.worst_a);
    }
}

int main(void)
{
    test_control();
    test_bound();
    test_edges();
    return ft_tt_test_finish();
}
