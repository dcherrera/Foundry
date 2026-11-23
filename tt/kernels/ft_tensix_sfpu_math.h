/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_sfpu_math.h
 * @brief Vector Unit (SFPU) math bodies behind the eltwise-SFPU wrapper: reciprocal square
 *   root, exp, reciprocal (two forms), sigmoid, silu, square and a binary max, plus the
 *   constant-register inits they need.
 *
 * Each ft_sfpu_calculate_* processes one face (8 SFPU-view Dst registers) in place; the
 * wrapper (ft_tensix_math_eltwise_sfpu.h) calls it once per face. Numerics:
 * - rsqrt: magic-constant seed, one quadratic refinement, one Newton step, inf/zero/negative
 *   guards.
 * - exp: Cody-Waite range reduction by ln2 (split high/low), degree-5 polynomial, exponent
 *   rebuild with overflow to +inf and underflow to 0.
 * - reciprocal: hardware estimate (SFPARECIP, BlackholeA0 SFPARECIP.md) plus Newton steps;
 *   the "compat" form normalises the exponent and runs Newton from a constant guess.
 * - binary max: a load-macro program (BlackholeA0/TensixTile/TensixCoprocessor/
 *   SFPLOADMACRO.md, SFPCONFIG.md, SFPSWAP.md) that swaps the larger value into the output.
 *
 * Reads FT_CB_DST_ACCUM_FP32 from the generated CB-format header (included by the kernel
 * first): when Dst is not fp32, each result is rounded to bf16 before it is stored.
 *
 * @status Clean-room rewrite (plan Phase 4, step 4.5b).
 * @issues The bf16-Dst flavour of sigmoid/silu keeps the fp32 exp and two Newton steps; only
 *   the final rounding differs between accumulate modes.
 * @todo None
 */
#ifndef FT_TENSIX_SFPU_MATH_H
#define FT_TENSIX_SFPU_MATH_H

#include <stdint.h>

#include "ft_tensix_sfpu.h"

#ifndef FT_SFPU_DST_ACCUMULATE_FP32
#ifdef FT_CB_DST_ACCUM_FP32
#define FT_SFPU_DST_ACCUMULATE_FP32 FT_CB_DST_ACCUM_FP32
#else
#define FT_SFPU_DST_ACCUMULATE_FP32 0
#endif
#endif

/* ===== Constants ========================================================================= */

/* rsqrt seed (integer bit trick) and quadratic refinement coefficients. */
#define FT_SFPU_SQRT_SEED_MAGIC 0x5f1110a0u
#define FT_SFPU_SQRT_COEFFICIENT_1 2.2825186f
#define FT_SFPU_SQRT_COEFFICIENT_2 2.2533049f

/* exp: log2(e), ln2 split into high and low parts, polynomial coefficients. */
#define FT_SFPU_EXP_LOG2_E 1.442695f
#define FT_SFPU_EXP_LN2_HIGH -6.93145752e-1f
#define FT_SFPU_EXP_LN2_LOW -1.42860677e-6f
#define FT_SFPU_EXP_POLY_5 1.37805939e-3f
#define FT_SFPU_EXP_POLY_4 8.37312452e-3f
#define FT_SFPU_EXP_POLY_3 4.16695364e-2f
#define FT_SFPU_EXP_POLY_2 1.66664720e-1f
#define FT_SFPU_EXP_POLY_1 4.99999851e-1f

/* Starting guess of the compat reciprocal (input normalised to [0.5, 1)). */
#define FT_SFPU_RECIPROCAL_COMPAT_GUESS 1.442695f

/* Binary max load-macro program: SFPLOADI halves, SFPSWAP "dest gets max", SFPSHFT2 "shift by
 * immediate", default load/store format, and the two LREGs it uses. */
#define FT_SFPU_LOADI_MOD0_UPPER 8u
#define FT_SFPU_LOADI_MOD0_LOWER 10u
#define FT_SFPU_SWAP_MOD1_VEC_MAX_MIN 9u
#define FT_SFPU_SHFT2_MOD1_SHIFT_IMMEDIATE 6u
#define FT_SFPU_LOAD_MOD0_DEFAULT 0u
#define FT_SFPU_LREG2 2u
#define FT_SFPU_LREG3 3u

/* Dst address of tile `index` for the binary-max loads (64 address units per tile). */
#define FT_SFPU_BINARY_DST_OFFSET(index) (((index) * 32u) << 1)

/* ===== Constant-register inits =========================================================== */

static FT_BH_INLINE void ft_sfpu_sqrt_init_constants(void)
{
    ft_sfpu_write_constant_register(FT_SFPU_CONST_REG_PROGRAMMABLE_0,
                                    ft_sfpu_uint_constant(FT_SFPU_SQRT_SEED_MAGIC));
    ft_sfpu_write_constant_register(FT_SFPU_CONST_REG_PROGRAMMABLE_1,
                                    ft_sfpu_float_constant(FT_SFPU_SQRT_COEFFICIENT_1));
    ft_sfpu_write_constant_register(FT_SFPU_CONST_REG_PROGRAMMABLE_2,
                                    ft_sfpu_float_constant(FT_SFPU_SQRT_COEFFICIENT_2));
}

/** P0 = 2.0, the Newton constant of ft_sfpu_reciprocal. */
static FT_BH_INLINE void ft_sfpu_reciprocal_init_constants(void)
{
    ft_sfpu_write_constant_register(FT_SFPU_CONST_REG_PROGRAMMABLE_0,
                                    ft_sfpu_float_constant(2.0f));
}

/* ===== Reciprocal square root ============================================================ */

/** 1/sqrt(x); +inf for 0, 0 for +inf, NaN for x < 0. Needs ft_sfpu_sqrt_init_constants. */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_reciprocal_sqrt(ft_sfpu_vector x)
{
    ft_sfpu_vector seed_shift = FT_SFPU_SHIFT_RIGHT_LOGICAL(x, 1);
    ft_sfpu_vector y =
        ft_sfpu_int_sub(ft_sfpu_read_constant_register(FT_SFPU_CONST_REG_PROGRAMMABLE_0),
                        seed_shift);
    ft_sfpu_vector xy = ft_sfpu_float_mul(x, y);
    ft_sfpu_vector negative_y = ft_sfpu_float_neg(y);
    ft_sfpu_vector c = ft_sfpu_float_mul(negative_y, xy);
    ft_sfpu_vector infinity_bits = ft_sfpu_float16b_constant(__builtin_inff());

    FT_SFPU_ASSIGN(
        y, ft_sfpu_float_mul(
               y, ft_sfpu_float_add(
                      ft_sfpu_read_constant_register(FT_SFPU_CONST_REG_PROGRAMMABLE_1),
                      ft_sfpu_float_mul(
                          c, ft_sfpu_float_add(ft_sfpu_read_constant_register(
                                                   FT_SFPU_CONST_REG_PROGRAMMABLE_2),
                                               c)))));
    FT_SFPU_ASSIGN(xy, ft_sfpu_float_mul(x, y));
    FT_SFPU_ASSIGN(negative_y, ft_sfpu_float_neg(y));

    ft_sfpu_vector one_minus_xyy =
        ft_sfpu_float_add(ft_sfpu_float_mul(negative_y, xy), ft_sfpu_float_constant(1.0f));
    ft_sfpu_vector half_y = FT_SFPU_ADD_EXPONENT(y, -1);
    ft_sfpu_vector infinity_minus_x_bits = ft_sfpu_int_sub(infinity_bits, x);

    FT_SFPU_IF(ft_sfpu_and(
        FT_SFPU_COMPARE_INT_IMMEDIATE(infinity_minus_x_bits, FT_SFPU_COMPARE_NE, 0),
        FT_SFPU_COMPARE_INT_IMMEDIATE(x, FT_SFPU_COMPARE_NE, 0)))
    {
        FT_SFPU_ASSIGN(y, ft_sfpu_float_add(ft_sfpu_float_mul(one_minus_xyy, half_y), y));
    }
    FT_SFPU_ELSE
    {
        FT_SFPU_ASSIGN(y, infinity_minus_x_bits);
    }
    FT_SFPU_ENDIF

    FT_SFPU_IF(FT_SFPU_COMPARE_FLOAT_IMMEDIATE(x, FT_SFPU_COMPARE_LT, 0.0f))
    {
        FT_SFPU_ASSIGN(y, ft_sfpu_float_constant(__builtin_nanf("")));
    }
    FT_SFPU_ENDIF

    return y;
}

static FT_BH_INLINE void ft_sfpu_calculate_rsqrt(void)
{
    int iteration;
#pragma GCC unroll 8
    for (iteration = 0; iteration < 8; iteration++) {
        ft_sfpu_vector result = ft_sfpu_reciprocal_sqrt(FT_SFPU_DST_LOAD_FLOAT(0));
#if !FT_SFPU_DST_ACCUMULATE_FP32
        FT_SFPU_ASSIGN(result, ft_sfpu_round_to_float16b(result));
#endif
        FT_SFPU_DST_STORE_FLOAT(0, result);
        ft_sfpu_dst_advance(1);
    }
}

/* ===== Exp =============================================================================== */

/** exp(a) to fp32 accuracy; +inf on overflow, 0 on underflow. */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_exp_fp32_accurate(ft_sfpu_vector a)
{
    ft_sfpu_vector i = FT_SFPU_UNINITIALISED();
    ft_sfpu_vector e = FT_SFPU_UNINITIALISED();
    ft_sfpu_vector f = FT_SFPU_UNINITIALISED();
    ft_sfpu_vector r = FT_SFPU_UNINITIALISED();
    ft_sfpu_vector j = FT_SFPU_UNINITIALISED();
    ft_sfpu_vector y = FT_SFPU_UNINITIALISED();
    ft_sfpu_vector signed_magnitude = FT_SFPU_UNINITIALISED();

    /* a = j*ln2 + f with j = round(a*log2(e)), |f| <= ln2/2. */
    FT_SFPU_ASSIGN(j, ft_sfpu_float_mul(a, ft_sfpu_float_constant(FT_SFPU_EXP_LOG2_E)));
    FT_SFPU_ASSIGN(r, ft_sfpu_float_constant(FT_SFPU_EXP_POLY_5));
    FT_SFPU_ASSIGN(signed_magnitude, ft_sfpu_round_to_signed_magnitude16(j));
    FT_SFPU_ASSIGN(j, ft_sfpu_convert_signed_magnitude_to_float(signed_magnitude));
    FT_SFPU_ASSIGN(f, ft_sfpu_float_add(
                          ft_sfpu_float_mul(j, ft_sfpu_float_constant(FT_SFPU_EXP_LN2_HIGH)), a));
    FT_SFPU_ASSIGN(f, ft_sfpu_float_add(
                          ft_sfpu_float_mul(j, ft_sfpu_float_constant(FT_SFPU_EXP_LN2_LOW)), f));

    /* exp(f) by Horner. */
    FT_SFPU_ASSIGN(r, ft_sfpu_float_add(ft_sfpu_float_mul(r, f),
                                        ft_sfpu_float_constant(FT_SFPU_EXP_POLY_4)));
    FT_SFPU_ASSIGN(r, ft_sfpu_float_add(ft_sfpu_float_mul(r, f),
                                        ft_sfpu_float_constant(FT_SFPU_EXP_POLY_3)));
    FT_SFPU_ASSIGN(r, ft_sfpu_float_add(ft_sfpu_float_mul(r, f),
                                        ft_sfpu_float_constant(FT_SFPU_EXP_POLY_2)));
    FT_SFPU_ASSIGN(r, ft_sfpu_float_add(ft_sfpu_float_mul(r, f),
                                        ft_sfpu_float_constant(FT_SFPU_EXP_POLY_1)));
    FT_SFPU_ASSIGN(i, ft_sfpu_int_abs(signed_magnitude));
    FT_SFPU_ASSIGN(y, ft_sfpu_float_add(ft_sfpu_float_mul(r, f), ft_sfpu_float_constant(1.0f)));
    FT_SFPU_ASSIGN(i, ft_sfpu_copy_sign(i, j));
    FT_SFPU_ASSIGN(r, ft_sfpu_float_add(ft_sfpu_float_mul(y, f), ft_sfpu_float_constant(1.0f)));
    FT_SFPU_ASSIGN(y, ft_sfpu_float_mul(y, ft_sfpu_float_constant(__builtin_inff())));

    /* Scale by 2^j through the exponent field; out-of-range lanes keep +inf or become 0. */
    FT_SFPU_ASSIGN(e, ft_sfpu_int_add(ft_sfpu_extract_exponent_biased(r), i));
    FT_SFPU_BLOCK
    {
        ft_sfpu_vector exponent_below_255 = FT_SFPU_INT_ADD_IMMEDIATE_SET_CC_NEGATIVE(e, -255);
        FT_SFPU_ASSIGN(y, ft_sfpu_set_exponent(r, e));
        FT_SFPU_IF(
            FT_SFPU_COMPARE_INT_IMMEDIATE(exponent_below_255, FT_SFPU_COMPARE_LT, -254))
        {
            FT_SFPU_ASSIGN(y, ft_sfpu_float_constant(0.0f));
        }
        FT_SFPU_ENDIF
    }
    FT_SFPU_ENDBLOCK

    return y;
}

static FT_BH_INLINE void ft_sfpu_calculate_exponential(void)
{
    int iteration;
    for (iteration = 0; iteration < 8; iteration++) {
        ft_sfpu_vector value = FT_SFPU_DST_LOAD_FLOAT(0);
        FT_SFPU_DST_STORE_FLOAT(0, ft_sfpu_exp_fp32_accurate(value));
        ft_sfpu_dst_advance(1);
    }
}

/* ===== Reciprocal, sigmoid, silu ========================================================= */

/** 1/x: hardware estimate refined by Newton steps. Needs P0 = 2.0. Lanes whose first residual
 *  is not negative keep the raw estimate. */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_reciprocal(ft_sfpu_vector x)
{
    ft_sfpu_vector y = ft_sfpu_approximate_reciprocal(x);
    /* t = x*y - 2 (sign inverted on purpose; -t is the Newton factor). */
    ft_sfpu_vector t = ft_sfpu_float_sub(
        ft_sfpu_float_mul(x, y), ft_sfpu_read_constant_register(FT_SFPU_CONST_REG_PROGRAMMABLE_0));
    ft_sfpu_vector y1 = ft_sfpu_float_sub(ft_sfpu_float_mul(y, ft_sfpu_float_neg(t)),
                                          ft_sfpu_float_constant(0.0f));

    FT_SFPU_IF(FT_SFPU_COMPARE_FLOAT_IMMEDIATE(t, FT_SFPU_COMPARE_LT, 0.0f))
    {
        FT_SFPU_ASSIGN(t, ft_sfpu_float_sub(ft_sfpu_float_mul(x, y1),
                                            ft_sfpu_read_constant_register(
                                                FT_SFPU_CONST_REG_PROGRAMMABLE_0)));
        FT_SFPU_ASSIGN(y, ft_sfpu_float_sub(ft_sfpu_float_mul(y1, ft_sfpu_float_neg(t)),
                                            ft_sfpu_float_constant(0.0f)));
    }
    FT_SFPU_ENDIF

    return y;
}

/** 1 / (1 + exp(-x)). */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_sigmoid(ft_sfpu_vector x)
{
    ft_sfpu_vector exp_negative_x = ft_sfpu_exp_fp32_accurate(ft_sfpu_float_neg(x));
    ft_sfpu_vector denominator = ft_sfpu_float_add(exp_negative_x, ft_sfpu_float_constant(1.0f));
    return ft_sfpu_reciprocal(denominator);
}

static FT_BH_INLINE void ft_sfpu_calculate_sigmoid(void)
{
    int iteration;
#pragma GCC unroll 8
    for (iteration = 0; iteration < 8; iteration++) {
        ft_sfpu_vector result = ft_sfpu_sigmoid(FT_SFPU_DST_LOAD_FLOAT(0));
#if !FT_SFPU_DST_ACCUMULATE_FP32
        FT_SFPU_ASSIGN(result, ft_sfpu_round_to_float16b(result));
#endif
        FT_SFPU_DST_STORE_FLOAT(0, result);
        ft_sfpu_dst_advance(1);
    }
}

/** Sigmoid of the first two SFPU-view registers of the face only. */
static FT_BH_INLINE void ft_sfpu_calculate_sigmoid_row0(void)
{
    int iteration;
#pragma GCC unroll 2
    for (iteration = 0; iteration < 2; iteration++) {
        ft_sfpu_vector result = ft_sfpu_sigmoid(FT_SFPU_DST_LOAD_FLOAT(0));
#if !FT_SFPU_DST_ACCUMULATE_FP32
        FT_SFPU_ASSIGN(result, ft_sfpu_round_to_float16b(result));
#endif
        FT_SFPU_DST_STORE_FLOAT(0, result);
        ft_sfpu_dst_advance(1);
    }
}

/** x * sigmoid(x). */
static FT_BH_INLINE void ft_sfpu_calculate_silu(void)
{
    int iteration;
#pragma GCC unroll 8
    for (iteration = 0; iteration < 8; iteration++) {
        ft_sfpu_vector x = FT_SFPU_DST_LOAD_FLOAT(0);
        ft_sfpu_vector result = ft_sfpu_float_mul(x, ft_sfpu_sigmoid(x));
#if !FT_SFPU_DST_ACCUMULATE_FP32
        FT_SFPU_ASSIGN(result, ft_sfpu_round_to_float16b(result));
#endif
        FT_SFPU_DST_STORE_FLOAT(0, result);
        ft_sfpu_dst_advance(1);
    }
}

static FT_BH_INLINE void ft_sfpu_calculate_silu_row0(void)
{
    int iteration;
#pragma GCC unroll 2
    for (iteration = 0; iteration < 2; iteration++) {
        ft_sfpu_vector x = FT_SFPU_DST_LOAD_FLOAT(0);
        ft_sfpu_vector result = ft_sfpu_float_mul(x, ft_sfpu_sigmoid(x));
#if !FT_SFPU_DST_ACCUMULATE_FP32
        FT_SFPU_ASSIGN(result, ft_sfpu_round_to_float16b(result));
#endif
        FT_SFPU_DST_STORE_FLOAT(0, result);
        ft_sfpu_dst_advance(1);
    }
}

/* ===== Reciprocal, compat form =========================================================== */

/** 1/|in|: normalise |in| to [0.5, 1), three Newton steps from a constant guess, then restore
 *  the exponent (underflow to 0). The caller applies the sign. */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_reciprocal_compat(ft_sfpu_vector in)
{
    ft_sfpu_vector val = FT_SFPU_SET_SIGN_IMMEDIATE(in, 1);
    ft_sfpu_vector guess;
    ft_sfpu_vector two;
    ft_sfpu_vector result;
    ft_sfpu_vector original_exponent;
    ft_sfpu_vector new_exponent;
    int iteration;

    FT_SFPU_ASSIGN(val, FT_SFPU_SET_EXPONENT_IMMEDIATE(val, 126));
    guess = ft_sfpu_float_constant(FT_SFPU_RECIPROCAL_COMPAT_GUESS);
    two = ft_sfpu_float_constant(2.0f);
    result = ft_sfpu_float_mul(guess, ft_sfpu_float_add(ft_sfpu_float_mul(val, guess), two));
    for (iteration = 0; iteration < 2; iteration++) {
        FT_SFPU_ASSIGN(result, ft_sfpu_float_mul(
                                   result, ft_sfpu_float_add(ft_sfpu_float_mul(val, result), two)));
    }

    original_exponent = ft_sfpu_extract_exponent(in);
    new_exponent = ft_sfpu_extract_exponent(result);
    FT_SFPU_ASSIGN(new_exponent, ft_sfpu_int_sub(new_exponent, original_exponent));
    FT_SFPU_ASSIGN(new_exponent, FT_SFPU_INT_ADD_IMMEDIATE(new_exponent, 126));
    FT_SFPU_IF(FT_SFPU_COMPARE_INT_IMMEDIATE(new_exponent, FT_SFPU_COMPARE_LT, 0))
    {
        FT_SFPU_ASSIGN(result, ft_sfpu_float_constant(0.0f));
        FT_SFPU_ASSIGN(new_exponent, ft_sfpu_int_constant(0));
    }
    FT_SFPU_ENDIF

    return ft_sfpu_set_exponent(result, new_exponent);
}

static FT_BH_INLINE void ft_sfpu_calculate_reciprocal(void)
{
    int iteration;
#pragma GCC unroll 8
    for (iteration = 0; iteration < 8; iteration++) {
        ft_sfpu_vector in = FT_SFPU_DST_LOAD_FLOAT(0);
        ft_sfpu_vector out = ft_sfpu_reciprocal_compat(in);
        FT_SFPU_IF(FT_SFPU_COMPARE_FLOAT_IMMEDIATE(in, FT_SFPU_COMPARE_LT, 0.0f))
        {
            FT_SFPU_ASSIGN(out, ft_sfpu_float_neg(out));
        }
        FT_SFPU_ENDIF
#if !FT_SFPU_DST_ACCUMULATE_FP32
        FT_SFPU_ASSIGN(out, ft_sfpu_round_to_float16b(out));
#endif
        FT_SFPU_DST_STORE_FLOAT(0, out);
        ft_sfpu_dst_advance(1);
    }
}

/* ===== Square ============================================================================ */

static FT_BH_INLINE void ft_sfpu_calculate_square(void)
{
    int iteration;
#pragma GCC unroll 0
    for (iteration = 0; iteration < 8; iteration++) {
        ft_sfpu_vector value = FT_SFPU_DST_LOAD_FLOAT(0);
        FT_SFPU_DST_STORE_FLOAT(0, ft_sfpu_float_mul(value, value));
        ft_sfpu_dst_advance(1);
    }
}

/* ===== Binary max ======================================================================== */

/** Program the load-macro sequences of the binary max: macro 0 swaps the loaded value against
 *  LREG2 keeping the max, macro 1 stores it. Instruction templates go in through LREG12/13,
 *  macro descriptors through SFPLOADI + SFPCONFIG, then the misc configuration word. */
static FT_BH_INLINE void ft_sfpu_binary_max_init_macros(void)
{
    FT_BH_TTINSN_SFPSWAP(0, FT_SFPU_LREG2, 12, FT_SFPU_SWAP_MOD1_VEC_MAX_MIN);
    FT_BH_TTINSN_SFPSHFT2(0, 0, 13, FT_SFPU_SHFT2_MOD1_SHIFT_IMMEDIATE);
    FT_BH_TTINSN_SFPLOADI(0, FT_SFPU_LOADI_MOD0_LOWER, 0x008C);
    FT_BH_TTINSN_SFPLOADI(0, FT_SFPU_LOADI_MOD0_UPPER, 0x00DD);
    FT_BH_TTINSN_SFPCONFIG(0, 4, 0);
    FT_BH_TTINSN_SFPLOADI(0, FT_SFPU_LOADI_MOD0_LOWER, 0x0000);
    FT_BH_TTINSN_SFPLOADI(0, FT_SFPU_LOADI_MOD0_UPPER, 0x5300);
    FT_BH_TTINSN_SFPCONFIG(0, 5, 0);
    FT_BH_TTINSN_SFPCONFIG(0x330, 8, 1);
}

/** Dst tile out = max(Dst tile in0, Dst tile in1), one face per call. */
static FT_BH_INLINE void ft_sfpu_calculate_binary_max(uint32_t dst_index_in0,
                                                      uint32_t dst_index_in1,
                                                      uint32_t dst_index_out)
{
    uint32_t offset0 = FT_SFPU_BINARY_DST_OFFSET(dst_index_in0);
    uint32_t offset1 = FT_SFPU_BINARY_DST_OFFSET(dst_index_in1);
    uint32_t offset2 = FT_SFPU_BINARY_DST_OFFSET(dst_index_out);
    int iteration;
#pragma GCC unroll 8
    for (iteration = 0; iteration < 8; iteration++) {
        uint32_t a = (uint32_t)iteration & 1u;
        FT_BH_PUSH_SFPLOADMACRO((0u << 2) | (a & 3u), FT_SFPU_LOAD_MOD0_DEFAULT, 7,
                                offset0 | (a >> 2));
        FT_BH_PUSH_SFPLOAD(FT_SFPU_LREG2, FT_SFPU_LOAD_MOD0_DEFAULT, 7, offset1);
        FT_BH_PUSH_SFPLOADMACRO((1u << 2) | (FT_SFPU_LREG3 & 3u), FT_SFPU_LOAD_MOD0_DEFAULT, 6,
                                offset2 | (FT_SFPU_LREG3 >> 2));
    }
    FT_BH_TTINSN_SFPNOP;
    FT_BH_TTINSN_SFPNOP;
    FT_BH_TTINSN_SFPNOP;
}

#endif /* FT_TENSIX_SFPU_MATH_H */
