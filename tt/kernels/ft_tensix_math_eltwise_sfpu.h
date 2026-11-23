/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_math_eltwise_sfpu.h
 * @brief Math-thread wrapper that runs the SFPU bodies of ft_tensix_sfpu_math.h over a Dst
 *   tile: per-op init (SFPU config reset, address modifiers, counter resets, constant
 *   registers), the start/done bracket and the four-face loop.
 *
 * start points the math Dst base at the tile and stalls the SFPU until the FPU is done
 * (BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md); each body then covers one face and
 * two Dst counter steps of 8 rows move to the next face (SETRWC.md). The SFPU uses address
 * modifier slot 7 (all zero) for its loads and stores, and slot 6 (Dst + 2) where a body
 * steps Dst itself. No FPU/SFPU semaphore is taken: the stall is the only hand-off.
 *
 * @status Clean-room rewrite (plan Phase 4, step 4.5b).
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_MATH_ELTWISE_SFPU_H
#define FT_TENSIX_MATH_ELTWISE_SFPU_H

#include <stdint.h>

#include "ft_tensix_compute.h"
#include "ft_tensix_sfpu_math.h"
#include "../fw/ft_bh_isa.h"

#ifdef FT_TRISC_MATH

#include "ft_tensix_compute_math.h"

/* ===== Building blocks =================================================================== */

/** Reset the whole SFPU configuration (SFPCONFIG.md: all destinations, reset form). */
static FT_BH_INLINE void ft_tensix_math_sfpu_init_config_register(void)
{
    FT_BH_TTINSN_SFPCONFIG(0, 0xF, 1);
}

/** Reset the SrcA, SrcB, Dst and fidelity counters. */
static FT_BH_INLINE void ft_tensix_math_reset_counters_abd_f(void)
{
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

/** Slot 7: no increments (SFPU loads and stores). */
static FT_BH_INLINE void ft_tensix_math_sfpu_configure_addrmod(void)
{
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_7, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
}

/** Slot 6: Dst + 2 per instruction. */
static FT_BH_INLINE void ft_tensix_math_sfpu_configure_addrmod_dest_step_2(void)
{
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_6, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(2, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
}

static FT_BH_INLINE void ft_tensix_math_sfpu_generic_init(void)
{
    ft_tensix_math_sfpu_init_config_register();
    ft_tensix_math_sfpu_configure_addrmod();
    ft_tensix_math_reset_counters_abd_f();
}

static FT_BH_INLINE void ft_tensix_math_sfpu_generic_init_dest_step_2(void)
{
    ft_tensix_math_sfpu_init_config_register();
    ft_tensix_math_sfpu_configure_addrmod();
    ft_tensix_math_sfpu_configure_addrmod_dest_step_2();
    ft_tensix_math_reset_counters_abd_f();
}

/** Point the math Dst base at tile `dst_index` (in the active Dst half) and wait for the FPU. */
static FT_BH_INLINE void ft_tensix_math_sfpu_start(uint32_t dst_index)
{
    uint32_t dst_offset = (dst_index << 6) + ((ft_tensix_dest_offset_id != 0)
                                                  ? FT_TENSIX_DEST_REGISTER_HALF_SIZE
                                                  : 0u);
    FT_BH_PUSH_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD, dst_offset);
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_SFPU, FT_BH_STALLWAIT_COND_MATH);
}

/** Move the Dst counter one face (16 rows) on. */
static FT_BH_INLINE void ft_tensix_math_sfpu_increment_dst_face_address(void)
{
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, FT_BH_SETRWC_CR_D, 8, 0, 0, FT_BH_SETRWC_SET_D);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, FT_BH_SETRWC_CR_D, 8, 0, 0, FT_BH_SETRWC_SET_D);
}

/** Reset the Dst counter. */
static FT_BH_INLINE void ft_tensix_math_sfpu_done(void)
{
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_D);
}

/* Run `sfpu_body` on all four faces. The body is inlined per use; the loop must not unroll. */
#define FT_TENSIX_MATH_SFPU_APPLY_VECTOR_MODE_RC(sfpu_body)                                    \
    do {                                                                                       \
        int ft_tensix_sfpu_face;                                                               \
        _Pragma("GCC unroll 0") for (ft_tensix_sfpu_face = 0; ft_tensix_sfpu_face < 4;         \
                                     ft_tensix_sfpu_face++)                                    \
        {                                                                                      \
            sfpu_body;                                                                         \
            ft_tensix_math_sfpu_increment_dst_face_address();                                  \
        }                                                                                      \
    } while (0)

/* Run `sfpu_body` on the two top faces, then skip the bottom two. */
#define FT_TENSIX_MATH_SFPU_APPLY_VECTOR_MODE_R(sfpu_body)                                     \
    do {                                                                                       \
        int ft_tensix_sfpu_face;                                                               \
        _Pragma("GCC unroll 0") for (ft_tensix_sfpu_face = 0; ft_tensix_sfpu_face < 2;         \
                                     ft_tensix_sfpu_face++)                                    \
        {                                                                                      \
            sfpu_body;                                                                         \
            ft_tensix_math_sfpu_increment_dst_face_address();                                  \
        }                                                                                      \
        ft_tensix_math_sfpu_increment_dst_face_address();                                      \
        ft_tensix_math_sfpu_increment_dst_face_address();                                      \
    } while (0)

#define FT_TENSIX_MATH_SFPU_UNARY_TILE_RC(dst_index, sfpu_body)                                \
    do {                                                                                       \
        ft_tensix_math_sfpu_start(dst_index);                                                  \
        FT_TENSIX_MATH_SFPU_APPLY_VECTOR_MODE_RC(sfpu_body);                                   \
        ft_tensix_math_sfpu_done();                                                            \
    } while (0)

/* The binary body addresses its tiles by Dst offset itself, so the base is tile 0. */
#define FT_TENSIX_MATH_SFPU_BINARY_TILE_RC(sfpu_body)                                          \
    do {                                                                                       \
        ft_tensix_math_sfpu_start(0);                                                          \
        FT_TENSIX_MATH_SFPU_APPLY_VECTOR_MODE_RC(sfpu_body);                                   \
        ft_tensix_math_sfpu_done();                                                            \
    } while (0)

/* ===== Public ops ======================================================================== */

static FT_BH_INLINE void ft_tensix_math_rsqrt_tile_init(void)
{
    ft_tensix_math_sfpu_generic_init();
    ft_tensix_math_reset_counters_abd_f();
    ft_tensix_math_reset_counters_abd_f();
    ft_sfpu_sqrt_init_constants();
}

static FT_BH_INLINE void ft_tensix_math_rsqrt_tile(uint32_t dst_index)
{
    FT_TENSIX_MATH_SFPU_UNARY_TILE_RC(dst_index, ft_sfpu_calculate_rsqrt());
}

static FT_BH_INLINE void ft_tensix_math_sigmoid_tile_init(void)
{
    ft_tensix_math_sfpu_generic_init();
    ft_tensix_math_reset_counters_abd_f();
    ft_sfpu_reciprocal_init_constants();
}

static FT_BH_INLINE void ft_tensix_math_sigmoid_tile(uint32_t dst_index)
{
    FT_TENSIX_MATH_SFPU_UNARY_TILE_RC(dst_index, ft_sfpu_calculate_sigmoid());
}

/** Sigmoid of the first two SFPU-view registers of the two top faces (a row-vector tile). */
static FT_BH_INLINE void ft_tensix_math_sigmoid_tile_row0(uint32_t dst_index)
{
    ft_tensix_math_sfpu_start(dst_index);
    FT_TENSIX_MATH_SFPU_APPLY_VECTOR_MODE_R(ft_sfpu_calculate_sigmoid_row0());
    ft_tensix_math_sfpu_done();
}

static FT_BH_INLINE void ft_tensix_math_silu_tile_init(void)
{
    ft_tensix_math_sfpu_generic_init();
    ft_tensix_math_reset_counters_abd_f();
    ft_tensix_math_reset_counters_abd_f();
    ft_sfpu_reciprocal_init_constants();
}

static FT_BH_INLINE void ft_tensix_math_silu_tile(uint32_t dst_index)
{
    FT_TENSIX_MATH_SFPU_UNARY_TILE_RC(dst_index, ft_sfpu_calculate_silu());
}

static FT_BH_INLINE void ft_tensix_math_silu_tile_row0(uint32_t dst_index)
{
    ft_tensix_math_sfpu_start(dst_index);
    FT_TENSIX_MATH_SFPU_APPLY_VECTOR_MODE_R(ft_sfpu_calculate_silu_row0());
    ft_tensix_math_sfpu_done();
}

static FT_BH_INLINE void ft_tensix_math_exp_tile_init(void)
{
    ft_tensix_math_sfpu_generic_init();
    ft_tensix_math_sfpu_generic_init();
#if FT_SFPU_DST_ACCUMULATE_FP32
    ft_sfpu_reciprocal_init_constants();
#endif
}

static FT_BH_INLINE void ft_tensix_math_exp_tile(uint32_t dst_index)
{
    FT_TENSIX_MATH_SFPU_UNARY_TILE_RC(dst_index, ft_sfpu_calculate_exponential());
}

static FT_BH_INLINE void ft_tensix_math_recip_tile_init(void)
{
    ft_tensix_math_sfpu_generic_init_dest_step_2();
    ft_tensix_math_sfpu_generic_init_dest_step_2();
}

static FT_BH_INLINE void ft_tensix_math_recip_tile(uint32_t dst_index)
{
    FT_TENSIX_MATH_SFPU_UNARY_TILE_RC(dst_index, ft_sfpu_calculate_reciprocal());
}

static FT_BH_INLINE void ft_tensix_math_square_tile_init(void)
{
    ft_tensix_math_sfpu_generic_init();
}

static FT_BH_INLINE void ft_tensix_math_square_tile(uint32_t dst_index)
{
    FT_TENSIX_MATH_SFPU_UNARY_TILE_RC(dst_index, ft_sfpu_calculate_square());
}

static FT_BH_INLINE void ft_tensix_math_binary_max_tile_init(void)
{
    ft_tensix_math_sfpu_generic_init_dest_step_2();
    ft_sfpu_binary_max_init_macros();
}

/** Dst tile dst_index_out = max(dst_index_in0, dst_index_in1), elementwise. */
static FT_BH_INLINE void ft_tensix_math_binary_max_tile(uint32_t dst_index_in0,
                                                        uint32_t dst_index_in1,
                                                        uint32_t dst_index_out)
{
    FT_TENSIX_MATH_SFPU_BINARY_TILE_RC(
        ft_sfpu_calculate_binary_max(dst_index_in0, dst_index_in1, dst_index_out));
}

#endif /* FT_TRISC_MATH */

#endif /* FT_TENSIX_MATH_ELTWISE_SFPU_H */
