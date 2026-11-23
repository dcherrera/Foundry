/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_sfpu.h
 * @brief C substrate for Vector Unit (SFPU) code on the math thread: the 32-lane vector type,
 *   Dst load/store/advance, immediates and constant registers, float/integer/exponent
 *   primitives, comparisons and the lane-predication (condition-code) stack.
 *
 * Every primitive is one SFPI compiler intrinsic (`__builtin_rvtt_*`) with fixed modifier
 * arguments. Those intrinsics are known to the SFPI GCC (-mcpu=tt-bh-tensix) without any
 * declaration; declaring one by hand turns it into a real call, so none is declared here.
 * Intrinsics that take an instruction-buffer argument take it first; it only orders and
 * aliases the issue and is never referenced by the emitted code.
 *
 * Modifier values follow the Vector Unit instruction pages (BlackholeA0/TensixTile/
 * TensixCoprocessor/SFPLOAD.md, SFPSTORE.md, SFPLOADI.md, SFPMOV.md, SFPIADD.md, SFPSHFT.md,
 * SFPEXEXP.md, SFPDIVP2.md, SFPSTOCHRND.md, SFPCAST.md, SFPABS.md, SFPARECIP.md, SFPPUSHC.md,
 * SFPPOPC.md and the WormholeB0 pages they defer to). Only math-thread code includes this
 * header, through ft_tensix_sfpu_math.h.
 *
 * @status Clean-room rewrite (plan Phase 4, step 4.5b).
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_SFPU_H
#define FT_TENSIX_SFPU_H

#include <stdint.h>

#include "ft_tensix_compute.h"
#include "../fw/ft_bh_isa.h"

/* ===== Vector type ======================================================================= */

/* One SFPU register: 32 lanes of 32 bits. This exact GCC vector type is the one the SFPI
 * intrinsics accept and return. */
typedef unsigned __attribute__((vector_size(sizeof(unsigned) * 32))) ft_sfpu_vector;

/* Ordering token passed first to intrinsics that issue through the instruction buffer. */
#define FT_SFPU_INSTRUCTION_BUFFER ((volatile void *)FT_BH_INSTRN_BUF)

/* ===== Modifier fields =================================================================== */

/* SFPLOAD / SFPSTORE: format "as stored in Dst" and the Blackhole "no increment" address mode
 * (slot 7, which Foundry programs to all-zero increments; Wormhole uses slot 3). One Dst
 * "register" of the SFPU view is two address units. */
#define FT_SFPU_LOAD_FMT_SRCB 0u
#define FT_SFPU_STORE_FMT_SRCB 0u
#define FT_SFPU_LOAD_ADDR_MODE_NOINC 7u
#define FT_SFPU_STORE_ADDR_MODE_NOINC 7u
#define FT_SFPU_DST_REGISTER_STRIDE 2u

/* SFPLOADI: immediate is a bf16 (upper half of an fp32). */
#define FT_SFPU_LOADI_FMT_FLOAT16B 0u

/* SFPMOV: copy with the sign bit complemented. */
#define FT_SFPU_MOV_COMPLEMENT_SIGN 1u

/* SFPIADD: condition-code handling and operand negation. */
#define FT_SFPU_IADD_CC_NONE 4u
#define FT_SFPU_IADD_CC_LESS_THAN_ZERO 0u
#define FT_SFPU_IADD_ARG_2SCOMP_LREG_DST 2u

/* SFPSHFT: logical shift. */
#define FT_SFPU_SHIFT_LOGICAL 0u

/* SFPEXEXP: exponent with or without the bias removed. */
#define FT_SFPU_EXEXP_DEBIAS 0u
#define FT_SFPU_EXEXP_NODEBIAS 1u

/* SFPDIVP2: add the immediate to the exponent. */
#define FT_SFPU_DIVP2_ADD 1u

/* SFPSTOCHRND: rounding mode and conversion selectors. */
#define FT_SFPU_STOCHRND_ROUND_EVEN 0u
#define FT_SFPU_STOCHRND_FP32_TO_FP16B 1u
#define FT_SFPU_STOCHRND_FP32_TO_SMAG16 7u

/* SFPCAST: sign-magnitude int32 to fp32, round to nearest even. */
#define FT_SFPU_CAST_SMAG32_TO_FP32_ROUND_EVEN 0u

/* SFPABS: integer absolute value. */
#define FT_SFPU_ABS_INT 0u

/* SFPARECIP (Blackhole only): approximate reciprocal. */
#define FT_SFPU_ARECIP_RECIPROCAL 0u

/* SFPPUSHC / SFPPOPC: plain push and pop of the condition-code stack. */
#define FT_SFPU_PUSHC_PUSH 0u
#define FT_SFPU_POPC_POP 0u

/* Comparison codes and flags of the compiler's compare intrinsics. */
#define FT_SFPU_COMPARE_LT 1u
#define FT_SFPU_COMPARE_NE 4u
#define FT_SFPU_COMPARE_IMMEDIATE_IS_FLOAT 32u
#define FT_SFPU_COMPARE_INT_SIGNED 8u

/* Boolean combiner code for two condition handles. */
#define FT_SFPU_BOOL_AND 2u

/* Programmable constant registers (LREG indices 12..14). */
#define FT_SFPU_CONST_REG_PROGRAMMABLE_0 12u
#define FT_SFPU_CONST_REG_PROGRAMMABLE_1 13u
#define FT_SFPU_CONST_REG_PROGRAMMABLE_2 14u

/* ===== Bits and constants ================================================================ */

/** IEEE-754 bits of `value` (folds to a constant for literals). */
static FT_BH_INLINE uint32_t ft_sfpu_float_bits(float value)
{
    union {
        float as_float;
        uint32_t as_bits;
    } pun;
    pun.as_float = value;
    return pun.as_bits;
}

/** Full 32-bit float immediate, as an opaque bit pattern. */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_float_constant(float value)
{
    return __builtin_rvtt_sfpxloadi(FT_SFPU_INSTRUCTION_BUFFER, ft_sfpu_float_bits(value), 0, 0,
                                    -32);
}

/** Full 32-bit unsigned immediate, as an opaque bit pattern. */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_uint_constant(uint32_t value)
{
    return __builtin_rvtt_sfpxloadi(FT_SFPU_INSTRUCTION_BUFFER, value, 0, 0, -32);
}

/** Signed integer immediate; the compiler may narrow it to a shorter load. */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_int_constant(int32_t value)
{
    return __builtin_rvtt_sfpxloadi(FT_SFPU_INSTRUCTION_BUFFER, (uint32_t)value, 0, 0, 31);
}

/** bf16 immediate: the float with its low 16 mantissa bits dropped (one SFPLOADI). */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_float16b_constant(float value)
{
    return __builtin_rvtt_sfploadi(FT_SFPU_INSTRUCTION_BUFFER, ft_sfpu_float_bits(value) >> 16, 0,
                                   0, FT_SFPU_LOADI_FMT_FLOAT16B);
}

/** Read constant register `index`. */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_read_constant_register(unsigned index)
{
    return __builtin_rvtt_sfpreadlreg(index);
}

/** Program constant register `index` with lane values `value`. */
static FT_BH_INLINE void ft_sfpu_write_constant_register(unsigned index, ft_sfpu_vector value)
{
    __builtin_rvtt_sfpwriteconfig_v(value, index);
}

/* ===== Dst register file ================================================================= */

/* Load / store SFPU-view Dst register `index` (compile-time constant) in its stored format. */
#define FT_SFPU_DST_LOAD_FLOAT(index)                                                          \
    __builtin_rvtt_sfpload(FT_SFPU_INSTRUCTION_BUFFER, (index) * FT_SFPU_DST_REGISTER_STRIDE, 0, \
                           0, FT_SFPU_LOAD_FMT_SRCB, FT_SFPU_LOAD_ADDR_MODE_NOINC)
#define FT_SFPU_DST_STORE_FLOAT(index, value)                                                  \
    __builtin_rvtt_sfpstore(FT_SFPU_INSTRUCTION_BUFFER, (value),                               \
                            (index) * FT_SFPU_DST_REGISTER_STRIDE, 0, 0, FT_SFPU_STORE_FMT_SRCB, \
                            FT_SFPU_STORE_ADDR_MODE_NOINC)

/** Advance the Dst read/write counter by `registers` SFPU-view registers. */
static FT_BH_INLINE void ft_sfpu_dst_advance(int registers)
{
    __builtin_rvtt_ttincrwc(0, (int)FT_SFPU_DST_REGISTER_STRIDE * registers, 0, 0);
}

/* ===== Assignment ======================================================================== */

/* Re-assign an already-live vector variable (a fresh declaration uses a plain initialiser). */
#define FT_SFPU_ASSIGN(destination, value)                                                     \
    ((destination) = __builtin_rvtt_sfpassign_lv((destination), (value)))

/* A declared vector with no value yet. */
#define FT_SFPU_UNINITIALISED() __builtin_rvtt_sfpnovalue()

/* ===== Float arithmetic ================================================================== */

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_float_mul(ft_sfpu_vector a, ft_sfpu_vector b)
{
    return __builtin_rvtt_sfpmul(a, b, 0);
}

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_float_add(ft_sfpu_vector a, ft_sfpu_vector b)
{
    return __builtin_rvtt_sfpadd(a, b, 0);
}

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_float_neg(ft_sfpu_vector a)
{
    return __builtin_rvtt_sfpmov(a, FT_SFPU_MOV_COMPLEMENT_SIGN);
}

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_float_sub(ft_sfpu_vector a, ft_sfpu_vector b)
{
    return __builtin_rvtt_sfpadd(a, ft_sfpu_float_neg(b), 0);
}

/* ===== Integer arithmetic ================================================================ */

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_int_add(ft_sfpu_vector a, ft_sfpu_vector b)
{
    return __builtin_rvtt_sfpiadd_v(a, b, FT_SFPU_IADD_CC_NONE);
}

/** a - b: the intrinsic negates its first operand, so the operands go in reversed. */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_int_sub(ft_sfpu_vector a, ft_sfpu_vector b)
{
    return __builtin_rvtt_sfpiadd_v(b, a, FT_SFPU_IADD_CC_NONE | FT_SFPU_IADD_ARG_2SCOMP_LREG_DST);
}

#define FT_SFPU_INT_ADD_IMMEDIATE(a, immediate)                                                \
    __builtin_rvtt_sfpiadd_i(FT_SFPU_INSTRUCTION_BUFFER, (a), (unsigned)(immediate), 0, 0,      \
                             FT_SFPU_IADD_CC_NONE)

/* As above, and the lane condition code is set where the sum is negative. */
#define FT_SFPU_INT_ADD_IMMEDIATE_SET_CC_NEGATIVE(a, immediate)                                \
    __builtin_rvtt_sfpiadd_i(FT_SFPU_INSTRUCTION_BUFFER, (a), (unsigned)(immediate), 0, 0,      \
                             FT_SFPU_IADD_CC_LESS_THAN_ZERO)

/* A right shift is a shift by the negated amount. */
#define FT_SFPU_SHIFT_RIGHT_LOGICAL(a, amount)                                                 \
    __builtin_rvtt_sfpshft_i(FT_SFPU_INSTRUCTION_BUFFER, (a), (unsigned)(-(amount)), 0, 0,      \
                             FT_SFPU_SHIFT_LOGICAL)

/* ===== Exponent, sign and conversion ===================================================== */

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_extract_exponent(ft_sfpu_vector v)
{
    return __builtin_rvtt_sfpexexp(v, FT_SFPU_EXEXP_DEBIAS);
}

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_extract_exponent_biased(ft_sfpu_vector v)
{
    return __builtin_rvtt_sfpexexp(v, FT_SFPU_EXEXP_NODEBIAS);
}

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_set_exponent(ft_sfpu_vector v, ft_sfpu_vector e)
{
    return __builtin_rvtt_sfpsetexp_v(v, e, 0);
}

#define FT_SFPU_SET_EXPONENT_IMMEDIATE(value, exponent)                                        \
    __builtin_rvtt_sfpsetexp_i(FT_SFPU_INSTRUCTION_BUFFER, (value), (unsigned)(exponent), 0, 0, 0)

#define FT_SFPU_SET_SIGN_IMMEDIATE(value, sign)                                                \
    __builtin_rvtt_sfpsetsgn_i(FT_SFPU_INSTRUCTION_BUFFER, (value), (unsigned)(sign), 0, 0, 0)

/* Multiply by 2^exponent by adding to the exponent field. */
#define FT_SFPU_ADD_EXPONENT(value, exponent)                                                  \
    __builtin_rvtt_sfpdivp2(FT_SFPU_INSTRUCTION_BUFFER, (value), (unsigned)(exponent), 0, 0,    \
                            FT_SFPU_DIVP2_ADD)

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_round_to_float16b(ft_sfpu_vector v)
{
    return __builtin_rvtt_sfpstochrnd_i(FT_SFPU_INSTRUCTION_BUFFER, v, 0, 0, 0,
                                        FT_SFPU_STOCHRND_FP32_TO_FP16B,
                                        FT_SFPU_STOCHRND_ROUND_EVEN);
}

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_round_to_signed_magnitude16(ft_sfpu_vector v)
{
    return __builtin_rvtt_sfpstochrnd_i(FT_SFPU_INSTRUCTION_BUFFER, v, 0, 0, 0,
                                        FT_SFPU_STOCHRND_FP32_TO_SMAG16,
                                        FT_SFPU_STOCHRND_ROUND_EVEN);
}

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_convert_signed_magnitude_to_float(ft_sfpu_vector v)
{
    return __builtin_rvtt_sfpcast(v, FT_SFPU_CAST_SMAG32_TO_FP32_ROUND_EVEN);
}

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_int_abs(ft_sfpu_vector v)
{
    return __builtin_rvtt_sfpabs(v, FT_SFPU_ABS_INT);
}

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_copy_sign(ft_sfpu_vector value,
                                                     ft_sfpu_vector sign_source)
{
    return __builtin_rvtt_sfpsetsgn_v(value, sign_source, 0);
}

static FT_BH_INLINE ft_sfpu_vector ft_sfpu_approximate_reciprocal(ft_sfpu_vector v)
{
    return __builtin_rvtt_sfparecip(v, FT_SFPU_ARECIP_RECIPROCAL);
}

/* ===== Comparisons ======================================================================= */
/* Each yields an opaque condition handle for FT_SFPU_IF or ft_sfpu_and. */

#define FT_SFPU_COMPARE_FLOAT_IMMEDIATE(a, comparison, literal)                                \
    __builtin_rvtt_sfpxfcmps(FT_SFPU_INSTRUCTION_BUFFER, (a), ft_sfpu_float_bits(literal), 0, 0, \
                             (comparison) | FT_SFPU_COMPARE_IMMEDIATE_IS_FLOAT)

#define FT_SFPU_COMPARE_INT_IMMEDIATE(a, comparison, literal)                                  \
    __builtin_rvtt_sfpxicmps(FT_SFPU_INSTRUCTION_BUFFER, (a), (unsigned)(literal), 0, 0,        \
                             (comparison) | FT_SFPU_COMPARE_INT_SIGNED)

/** Both conditions. A function so each handle is evaluated exactly once. */
static FT_BH_INLINE unsigned ft_sfpu_and(unsigned a, unsigned b)
{
    return __builtin_rvtt_sfpxbool(FT_SFPU_BOOL_AND, a, b);
}

/* ===== Predication ======================================================================= */
/*
 * Usage: FT_SFPU_IF(c) { ... } FT_SFPU_ELSE { ... } FT_SFPU_ENDIF, and
 * FT_SFPU_BLOCK { ... } FT_SFPU_ENDBLOCK. Each opener pushes the condition-code stack once and
 * records it in a block-scoped counter; the closer pops that many times (the loop folds away).
 * A nested use shadows the outer counter on purpose.
 */

/* Narrow the active lanes by `expression`. The three statements must stay separate and in this
 * order: the compare has to be emitted after the dependency marker. */
#define FT_SFPU_CONDITION(expression)                                                          \
    do {                                                                                       \
        unsigned ft_sfpu_cc_dependency = __builtin_rvtt_sfpxvif();                             \
        unsigned ft_sfpu_cc_result = (expression);                                             \
        __builtin_rvtt_sfpxcondb(ft_sfpu_cc_result, ft_sfpu_cc_dependency);                    \
    } while (0)

#define FT_SFPU_IF(expression)                                                                 \
    {                                                                                          \
        unsigned ft_sfpu_cc_depth = 1u;                                                        \
        __builtin_rvtt_sfppushc(FT_SFPU_PUSHC_PUSH);                                           \
        FT_SFPU_CONDITION(expression);                                                         \
        {

#define FT_SFPU_ELSE                                                                           \
    }                                                                                          \
    __builtin_rvtt_sfpcompc();                                                                 \
    {

#define FT_SFPU_ENDIF                                                                          \
    }                                                                                          \
    for (unsigned ft_sfpu_cc_pop = ft_sfpu_cc_depth; ft_sfpu_cc_pop--;)                        \
        __builtin_rvtt_sfppopc(FT_SFPU_POPC_POP);                                              \
    }

#define FT_SFPU_BLOCK                                                                          \
    {                                                                                          \
        unsigned ft_sfpu_cc_depth = 1u;                                                        \
        __builtin_rvtt_sfppushc(FT_SFPU_PUSHC_PUSH);

#define FT_SFPU_ENDBLOCK                                                                       \
    for (unsigned ft_sfpu_cc_pop = ft_sfpu_cc_depth; ft_sfpu_cc_pop--;)                        \
        __builtin_rvtt_sfppopc(FT_SFPU_POPC_POP);                                              \
    }

#endif /* FT_TENSIX_SFPU_H */
