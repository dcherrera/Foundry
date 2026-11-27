/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_sfpu_tf32_round.h
 * @brief The Tf32 round bias as a math-thread SFPU tile op: every fp32 word of a Dst tile
 *   gets 0x1000 added as a 32-bit integer, so the FPU's truncating Tf32 unpack of the packed
 *   value rounds to the nearer Tf32, ties away from zero (the GDN state, d1).
 *
 * The words move through the SFPU as Int32 (SFPLOAD / SFPSTORE mode 4): the Fp32 store would
 * flush denormals, and an integer add on float bits needs the bits untouched
 * (BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md, SFPSTORE.md, SFPIADD.md).
 *
 * @status Added 2026-11-27 (d1).
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_SFPU_TF32_ROUND_H
#define FT_TENSIX_SFPU_TF32_ROUND_H

#include <stdint.h>

#include "ft_tensix_math_eltwise_sfpu.h"

#ifdef FT_TRISC_MATH

/* Half a Tf32 ulp in fp32 bits: Tf32 keeps 10 of fp32's 23 mantissa bits. A red build
 * overrides it (EXTRA_KERNEL_DEFINES) to prove a card check can tell the rule apart. */
#ifndef FT_SFPU_TF32_ROUND_BIAS
#define FT_SFPU_TF32_ROUND_BIAS 0x1000u
#endif

/* SFPLOAD / SFPSTORE "Int32": the 32 Dst bits as they are. */
#define FT_SFPU_LOAD_FMT_INT32 4u
#define FT_SFPU_STORE_FMT_INT32 4u

#define FT_SFPU_DST_LOAD_INT32(index)                                                          \
    __builtin_rvtt_sfpload(FT_SFPU_INSTRUCTION_BUFFER, (index) * FT_SFPU_DST_REGISTER_STRIDE, 0, \
                           0, FT_SFPU_LOAD_FMT_INT32, FT_SFPU_LOAD_ADDR_MODE_NOINC)
#define FT_SFPU_DST_STORE_INT32(index, value)                                                  \
    __builtin_rvtt_sfpstore(FT_SFPU_INSTRUCTION_BUFFER, (value),                               \
                            (index) * FT_SFPU_DST_REGISTER_STRIDE, 0, 0, FT_SFPU_STORE_FMT_INT32, \
                            FT_SFPU_STORE_ADDR_MODE_NOINC)

/** fp32 bits += FT_SFPU_TF32_ROUND_BIAS as integers, one face. */
static FT_BH_INLINE void ft_sfpu_calculate_tf32_round_bias(void)
{
    const ft_sfpu_vector bias = ft_sfpu_int_constant((int32_t)FT_SFPU_TF32_ROUND_BIAS);
    int iteration;
#pragma GCC unroll 0
    for (iteration = 0; iteration < 8; iteration++) {
        ft_sfpu_vector value = FT_SFPU_DST_LOAD_INT32(0);
        FT_SFPU_DST_STORE_INT32(0, ft_sfpu_int_add(value, bias));
        ft_sfpu_dst_advance(1);
    }
}

static FT_BH_INLINE void ft_tensix_math_tf32_round_bias_tile_init(void)
{
    ft_tensix_math_sfpu_generic_init();
}

/** Dst tile `dst_index` (fp32 Dst): every word's bits += the bias, before it is packed. */
static FT_BH_INLINE void ft_tensix_math_tf32_round_bias_tile(uint32_t dst_index)
{
    FT_TENSIX_MATH_SFPU_UNARY_TILE_RC(dst_index, ft_sfpu_calculate_tf32_round_bias());
}

#endif /* FT_TRISC_MATH */

#endif /* FT_TENSIX_SFPU_TF32_ROUND_H */
