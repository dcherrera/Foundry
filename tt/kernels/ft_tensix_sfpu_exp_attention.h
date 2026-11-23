/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_sfpu_exp_attention.h
 * @brief The attention FAST EXP: a bf16-grade exp(a) for a <= 0 on the Vector Unit (SFPU), in
 *   place of ft_sfpu_exp_fp32_accurate, for the attention compute kernel's -DATTN_FAST_EXP
 *   builds only (programs attention_k8v4{,_split}_{fexp,fast}). No shipped program includes it.
 *
 * WHAT ATTENTION FEEDS IT. Every exp in resident_attention_compute.c takes a NON-POSITIVE
 * argument: S - rowmax(S) and m_old - m_new (the max includes the value subtracted), and a
 * masked score drives it to about -1e30 (FT_TT_OP_ATTENTION_MASK_VALUE). The result is packed
 * to a Float16_b CB (cb_s, cb_c) and feeds a bf16 matmul, so fp32 accuracy is spent for
 * nothing. So, against the accurate body:
 *   - no overflow path (y * inf): positive inputs are still right up to a ~ 88, not needed;
 *   - no LN2_LOW correction: f = a - j*ln2 with one fp32 ln2 is off by <= 127 * 1.9e-9;
 *   - a degree-2 minimax polynomial on |f| <= ln2/2 instead of degree 6;
 *   - the underflow-to-0 clamp KEPT, as a clamp of the input: a = max(a, -88) gives j >= -127,
 *     and j = -127 makes the scale's bits exactly 0, so y = p * 0 = +0 for every a <= -87.68
 *     (the mask value included); between there and ln(FLT_MIN) = -87.34 SFPMAD flushes the
 *     denormal product to 0 (SFPMAD.md).
 *
 * THE SEQUENCE, per element (the accurate body's steps in brackets):
 *     a  = max(a, -88)                   SFPSWAP min/max, -88 from one bf16 SFPLOADI
 *     t  = a * log2(e) + 1.5*2^23        SFPMAD: j = round(a*log2e) lands in t's low bits
 *     j  = t - 1.5*2^23                  SFPADD (exact: Sterbenz)        [stochrnd + cast]
 *     f  = j * -ln2 + a                  SFPMAD                          [two SFPMADs]
 *     p  = (c2*f + c1)*f + c0            two SFPMADs                     [six SFPMADs]
 *     s  = bits(1.0) + (bits(t) << 23)   SFPSHFT + SFPIADD: 2^j, its bits 0 at j = -127
 *                                                          [abs, setsgn, exexp, iadd, setexp]
 *     y  = p * s                         SFPMUL
 *   (bits(t) << 23 == j << 23 mod 2^32 because bits(1.5*2^23) has nine zero low bits.)
 *
 * SFPU INSTRUCTION COUNT, per element, SFPLOAD to SFPSTORE inclusive, read off the SFPI
 * 7.68.0 build of attention_k8v4_split's trisc1.elf (riscv-tt-elf-objdump -d):
 *     ft_sfpu_exp_fp32_accurate   39  (14 of them SFPLOADIs re-loading constants, plus the
 *                                      overflow/underflow predication)
 *     ft_sfpu_exp_attention       12  (load, loadi, swap, 6 MAD-class, shft, iadd, store)
 *   The three fp32 constants live in the programmable constant registers (P0..P2, written by
 *   the tile init: 3 x {2 SFPLOADI + SFPCONFIG}); the three short ones (1.5*2^23 bf16, c1 and
 *   c2 fp16) are loaded once per tile, before the face loop, and stay in LREGs. Only the -88
 *   is re-loaded per element: SFPSWAP writes the min into its register. Stalls: SFPSWAP and
 *   each dependent SFPMAD cost one cycle more (SFPSWAP.md, SFPMAD.md), about 4 an element.
 *
 * ACCURACY: max relative error 1.80e-3 = 2^-9.12 for a in [ln(FLT_MIN), 0], exact 0 below
 * -87.68 (tests/test_attn_fast_exp.c emulates this sequence in fp32 over a dense sweep of
 * [-88, 0] and carries a RED). The bf16 rounding of the packed result adds up to 2^-9.
 * exp(0) = c0 = 1.00032, not 1: every exp carries that factor, so it cancels in O / l.
 *
 * THE POLYNOMIAL (tests/fit_attn_exp_poly.py): relative minimax of e^f on |f| <= ln2/2 with
 * c1, c2 fp16-representable (one short SFPLOADI each) and c0 fp32. c0 = 1 pinned costs 2^-8.96,
 * over the bound; degree 3 would be 1e-4 for one more SFPMAD.
 *
 * CONSTANT REGISTERS. The init overwrites P0..P2 (the accurate exp's init writes P0 = 2.0, the
 * reciprocal's Newton constant). Nothing in the attention kernel reads them between an exp and
 * the next SFPU op's own init: recip_tile is the compat form (no P0), binary max programs its
 * own macros. SFPSWAP mode VEC_MAX_MIN (9) is refused by the compiler's modifier mask, so the
 * swap is VEC_MIN_MAX with -88 as VD (it gets the min) and a as VC (the max).
 *
 * Host use: define FT_SFPU_EXPA_CONSTANTS_ONLY before including for the constants alone.
 *
 * @status Added with ATTN_FAST_EXP (Qwen3.8 prefill attention fast-math, card 0112). Host
 *   emulation green; not yet measured on the card.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_SFPU_EXP_ATTENTION_H
#define FT_TENSIX_SFPU_EXP_ATTENTION_H

/* ===== Constants ========================================================================= */

/* fp32, in the programmable constant registers. */
#define FT_SFPU_EXPA_LOG2E 1.44269504f      /* 0x3FB8AA3B */
#define FT_SFPU_EXPA_NEG_LN2 -0.693147182f  /* 0xBF317218 */
#define FT_SFPU_EXPA_C0 1.00032079f         /* 0x3F800A83 */
/* Short immediates (one SFPLOADI each). */
#define FT_SFPU_EXPA_C1 1.0146484375f        /* fp16 0x3C0F */
#define FT_SFPU_EXPA_C2 0.4970703125f        /* fp16 0x37F4 */
#define FT_SFPU_EXPA_ROUND_MAGIC 12582912.0f /* 1.5 * 2^23, bf16 0x4B40 */
#define FT_SFPU_EXPA_FLOOR -88.0f            /* bf16 0xC2B0: j = round(-88 log2e) = -127 */
/* bits(1.0), SFPU constant register 10: the scale's exponent bias. */
#define FT_SFPU_EXPA_ONE_BITS 0x3F800000u
/* bits(t) << 23 moves j into the exponent field. */
#define FT_SFPU_EXPA_EXPONENT_SHIFT 23u

#ifndef FT_SFPU_EXPA_CONSTANTS_ONLY

#include <stdint.h>

#include "ft_tensix_math_eltwise_sfpu.h"

#define FT_SFPU_EXPA_REG_LOG2E FT_SFPU_CONST_REG_PROGRAMMABLE_0
#define FT_SFPU_EXPA_REG_NEG_LN2 FT_SFPU_CONST_REG_PROGRAMMABLE_1
#define FT_SFPU_EXPA_REG_C0 FT_SFPU_CONST_REG_PROGRAMMABLE_2
/* LReg 10: 1.0 in every lane (LReg.md). */
#define FT_SFPU_EXPA_REG_ONE 10u
/* SFPSWAP: VD gets the min, VC the max (SFPSWAP.md). */
#define FT_SFPU_EXPA_SWAP_VEC_MIN_MAX 1u

/** P0 = log2(e), P1 = -ln2, P2 = c0. */
static FT_BH_INLINE void ft_sfpu_exp_attention_init_constants(void)
{
    ft_sfpu_write_constant_register(FT_SFPU_EXPA_REG_LOG2E,
                                    ft_sfpu_float_constant(FT_SFPU_EXPA_LOG2E));
    ft_sfpu_write_constant_register(FT_SFPU_EXPA_REG_NEG_LN2,
                                    ft_sfpu_float_constant(FT_SFPU_EXPA_NEG_LN2));
    ft_sfpu_write_constant_register(FT_SFPU_EXPA_REG_C0, ft_sfpu_float_constant(FT_SFPU_EXPA_C0));
}

/** exp(a) for a <= 0 (right up to a ~ 88), bf16-grade; 0 for a <= -87.68. Needs
 *  ft_sfpu_exp_attention_init_constants; `magic`, `c1`, `c2` are the caller's hoisted loads. */
static FT_BH_INLINE ft_sfpu_vector ft_sfpu_exp_attention(ft_sfpu_vector a, ft_sfpu_vector magic,
                                                         ft_sfpu_vector c1, ft_sfpu_vector c2)
{
    ft_sfpu_vector floor = ft_sfpu_float16b_constant(FT_SFPU_EXPA_FLOOR);
    ft_sfpu_vector clamped = __builtin_rvtt_sfpselect2(
        __builtin_rvtt_sfpswap(floor, a, FT_SFPU_EXPA_SWAP_VEC_MIN_MAX), 1);
    /* An explicit SFPMAD: written as mul + add, the compiler turns the add of a short constant
     * into SFPADDI and the pair costs two instructions. */
    ft_sfpu_vector t = __builtin_rvtt_sfpmad(
        clamped, ft_sfpu_read_constant_register(FT_SFPU_EXPA_REG_LOG2E), magic, 0);
    ft_sfpu_vector j = ft_sfpu_float_sub(t, magic);
    ft_sfpu_vector f = ft_sfpu_float_add(
        ft_sfpu_float_mul(j, ft_sfpu_read_constant_register(FT_SFPU_EXPA_REG_NEG_LN2)), clamped);
    ft_sfpu_vector p = ft_sfpu_float_add(ft_sfpu_float_mul(f, c2), c1);
    ft_sfpu_vector scale;

    FT_SFPU_ASSIGN(p, ft_sfpu_float_add(ft_sfpu_float_mul(p, f),
                                        ft_sfpu_read_constant_register(FT_SFPU_EXPA_REG_C0)));
    scale = ft_sfpu_int_add(__builtin_rvtt_sfpshft_i(FT_SFPU_INSTRUCTION_BUFFER, t,
                                                     FT_SFPU_EXPA_EXPONENT_SHIFT, 0, 0,
                                                     FT_SFPU_SHIFT_LOGICAL),
                            ft_sfpu_read_constant_register(FT_SFPU_EXPA_REG_ONE));
    return ft_sfpu_float_mul(p, scale);
}

/** One face. Dst is fp32 in every attention build (dst_accum_mode 1), so no bf16 rounding. */
static FT_BH_INLINE void ft_sfpu_calculate_exponential_attention(ft_sfpu_vector magic,
                                                                 ft_sfpu_vector c1,
                                                                 ft_sfpu_vector c2)
{
    int iteration;
    for (iteration = 0; iteration < 8; iteration++) {
        ft_sfpu_vector value = FT_SFPU_DST_LOAD_FLOAT(0);
        FT_SFPU_DST_STORE_FLOAT(0, ft_sfpu_exp_attention(value, magic, c1, c2));
        ft_sfpu_dst_advance(1);
    }
}

#ifdef FT_TRISC_MATH
static FT_BH_INLINE void ft_tensix_math_exp_attention_tile_init(void)
{
    ft_tensix_math_sfpu_generic_init();
    ft_sfpu_exp_attention_init_constants();
}

/** The tile: the short constants loaded once, before the four faces. */
static FT_BH_INLINE void ft_tensix_math_exp_attention_tile(uint32_t dst_index)
{
    ft_sfpu_vector magic = ft_sfpu_float_constant(FT_SFPU_EXPA_ROUND_MAGIC);
    ft_sfpu_vector c1 = ft_sfpu_float_constant(FT_SFPU_EXPA_C1);
    ft_sfpu_vector c2 = ft_sfpu_float_constant(FT_SFPU_EXPA_C2);

    FT_TENSIX_MATH_SFPU_UNARY_TILE_RC(dst_index,
                                      ft_sfpu_calculate_exponential_attention(magic, c1, c2));
}
#endif /* FT_TRISC_MATH */

#endif /* FT_SFPU_EXPA_CONSTANTS_ONLY */
#endif /* FT_TENSIX_SFPU_EXP_ATTENTION_H */
