/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_compute_math.h
 * @brief MATH (TRISC1) side of the compute triad: FPU/ALU configuration for two operand
 *   formats, the Src zero-flag state, the math<->pack dest handshake (two dest halves) and
 *   the no-broadcast eltwise ADD/SUB sequence.
 *
 * Acquiring dest does not zero it: the PACK thread zeroes each half with ZEROACC through
 * ADDR_MOD slot 1, which this thread programs, so slot 1 is programmed although the math
 * sequence never uses it.
 *
 * @status Clean-room rewrite, plan Phase 4 step 4.5a.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_COMPUTE_MATH_H
#define FT_TENSIX_COMPUTE_MATH_H

#include <stdint.h>

#include "ft_tensix_compute.h"

/* ===== Src zero-flag state (Foundry convention) ========================================== */

#define FT_TENSIX_SRC_ZERO_FLAG_UNCONFIGURED 0
#define FT_TENSIX_SRC_ZERO_FLAG_DEFAULT 1
#define FT_TENSIX_SRC_ZERO_FLAG_MOV_OPS 3

static uint8_t ft_tensix_src_zero_flag_state = FT_TENSIX_SRC_ZERO_FLAG_UNCONFIGURED;
static uint32_t ft_tensix_src_zero_flag_srca_format = 0xff;
static uint32_t ft_tensix_src_zero_flag_srcb_format = 0xff;

/** UInt16 operands need the Src zero flags off. */
static FT_BH_INLINE int ft_tensix_requires_disabled_src_zero_flag(uint32_t srca_format,
                                                                  uint32_t srcb_format)
{
    return (srca_format == FT_BH_DATA_FORMAT_UINT16) || (srcb_format == FT_BH_DATA_FORMAT_UINT16);
}

static FT_BH_INLINE int ft_tensix_math_is_int8_or_int32(uint32_t format)
{
    return (FT_BH_DATA_FORMAT_CONFIG_BITS(format) == FT_BH_DATA_FORMAT_INT8)
           || (format == FT_BH_DATA_FORMAT_INT32);
}

static FT_BH_INLINE void ft_tensix_math_configure_src_zero_flag(int disable)
{
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG,
                           FT_BH_STALLWAIT_COND_MATH | FT_BH_STALLWAIT_COND_SFPU);
    FT_TENSIX_CFG_RMW_FIELD(SRC_ZERO_FLAG_DISABLE, disable ? 1u : 0u);
}

static FT_BH_INLINE void ft_tensix_math_configure_default_zero_flag_state(uint32_t srca_format,
                                                                         uint32_t srcb_format)
{
    if (ft_tensix_src_zero_flag_state == FT_TENSIX_SRC_ZERO_FLAG_DEFAULT
        && ft_tensix_src_zero_flag_srca_format == srca_format
        && ft_tensix_src_zero_flag_srcb_format == srcb_format) {
        return;
    }
    ft_tensix_src_zero_flag_srca_format = srca_format;
    ft_tensix_src_zero_flag_srcb_format = srcb_format;
    ft_tensix_src_zero_flag_state = FT_TENSIX_SRC_ZERO_FLAG_DEFAULT;
    ft_tensix_math_configure_src_zero_flag(
        ft_tensix_requires_disabled_src_zero_flag(srca_format, srcb_format));
}

/* ===== Configuration ===================================================================== */

/** Wait for the pack side to drain, then reset the math<->pack semaphore and dest half 0. */
static FT_BH_INLINE void ft_tensix_math_pack_sync_init(void)
{
    ft_tensix_sync();
    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_MATH_PACK) > 0) {
    }
    FT_BH_TTINSN_SEMINIT(2, 0, FT_BH_SEMAPHORE_1);
    ft_tensix_dest_offset_id = 0;
    FT_BH_TTINSN_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD, 0);
}

static FT_BH_INLINE void ft_tensix_math_hw_configure(uint32_t srca_format, uint32_t srcb_format,
                                                     uint32_t is_fp32_dest_acc_en)
{
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG,
                           FT_BH_STALLWAIT_COND_MATH | FT_BH_STALLWAIT_COND_SFPU);
    FT_TENSIX_CFG_RMW_FIELD(ZEROACC_ABSOLUTE_TILE_MODE, 0u);
    FT_TENSIX_CFG_RMW_FIELD(INT8_MATH_MODE, ft_tensix_math_is_int8_or_int32(srca_format)
                                                || ft_tensix_math_is_int8_or_int32(srcb_format));
    FT_TENSIX_CFG_RMW_FIELD(DEST_FP32_MODE, is_fp32_dest_acc_en);
    FT_TENSIX_CFG_RMW_FIELD(SFPU_DEST_FP32_MODE, is_fp32_dest_acc_en);
    /* Clear the SrcA/SrcB format overrides; the Dst override lanes of this word are PACK's. */
    FT_TENSIX_CFG_RMW(FT_BH_CFG_MATH_SRCA_FORMAT_OVERRIDE_VALUE_WORD, 0,
                      FT_BH_CFG_MATH_SRCA_FORMAT_OVERRIDE_VALUE_MASK
                          | FT_BH_CFG_MATH_SRCA_FORMAT_OVERRIDE_ENABLE_MASK
                          | FT_BH_CFG_MATH_SRCB_FORMAT_OVERRIDE_VALUE_MASK
                          | FT_BH_CFG_MATH_SRCB_FORMAT_OVERRIDE_ENABLE_MASK,
                      0u);
    ft_tensix_math_configure_default_zero_flag_state(srca_format, srcb_format);
}

/* ===== Dest <-> pack handshake =========================================================== */

/** Block until the pack side has freed a dest half. */
static FT_BH_INLINE void ft_tensix_math_wait_for_dest_available(void)
{
    FT_BH_TTINSN_SEMWAIT(FT_BH_STALLWAIT_BLOCK_MATH | FT_BH_STALLWAIT_BLOCK_SFPU
                             | FT_BH_STALLWAIT_BLOCK_SYNC,
                         FT_TENSIX_T6_SEM(FT_TENSIX_SEM_MATH_PACK), FT_BH_SEMWAIT_COND_MAX);
}

/** Hand the current dest half to the pack side and switch to the other half. */
static FT_BH_INLINE void ft_tensix_math_dest_section_done(void)
{
    uint32_t base;
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_SYNC,
                           FT_BH_STALLWAIT_COND_MATH | FT_BH_STALLWAIT_COND_SFPU);
    FT_BH_TTINSN_SEMPOST(FT_BH_SEMAPHORE_1);
    math_sync_tile_dst_index = 0;
    ft_tensix_dest_offset_id = 1 - ft_tensix_dest_offset_id;
    base = (ft_tensix_dest_offset_id != 0) ? FT_TENSIX_DEST_REGISTER_HALF_SIZE : 0u;
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG,
                           FT_BH_STALLWAIT_COND_MATH | FT_BH_STALLWAIT_COND_SFPU1);
    FT_BH_PUSH_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD, base);
}

/* ===== Eltwise binary, no broadcast ====================================================== */

/** ADDR_MOD slots 0..3 for eltwise binary. Slot 1 is all-zero (used by PACK's ZEROACC). */
static FT_BH_INLINE void ft_tensix_math_eltwise_binary_configure_addrmod(
    uint32_t fidelity_increment)
{
    FT_TENSIX_ADDR_MOD_SET(0, FT_TENSIX_ADDR_MOD_SRC_VAL(FT_TENSIX_MAX_FPU_ROWS, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(FT_TENSIX_MAX_FPU_ROWS, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(FT_TENSIX_MAX_FPU_ROWS, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(1, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(2, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 1, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 1, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 1, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(fidelity_increment, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(3, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 1, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 1, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(FT_TENSIX_MAX_FPU_ROWS, 0, 0, 1),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 1),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
}

static FT_BH_INLINE void ft_tensix_math_eltwise_binary_configure_mop_addsub(
    uint32_t binary_op_word, uint32_t num_faces, uint32_t face_r_dim)
{
    uint32_t inner = (face_r_dim > FT_TENSIX_MAX_FPU_ROWS)
                         ? (face_r_dim >> FT_TENSIX_MAX_FPU_ROWS_LOG2)
                         : 1u;
    ft_tensix_mop_template t = ft_tensix_mop_template_1(num_faces, inner, binary_op_word);
    if (face_r_dim <= FT_TENSIX_MAX_FPU_ROWS) {
        t.loop_op1 = FT_BH_ENC_INCRWC(0, FT_TENSIX_MAX_FPU_ROWS, FT_TENSIX_MAX_FPU_ROWS,
                                      FT_TENSIX_MAX_FPU_ROWS);
    }
    t.end_op0 = FT_BH_ENC_SETRWC(FT_BH_SETRWC_CLR_AB, FT_BH_SETRWC_CR_AB, 0, 0, 0,
                                 FT_BH_SETRWC_SET_AB);
    t.end_op1 = FT_BH_ENC_NOP;
    ft_tensix_mop_template_program(&t);
}

/** `binary_op_word` is a caller-built ELWADD/ELWSUB word. */
static FT_BH_INLINE void ft_tensix_math_eltwise_binary_addsub_init(uint32_t binary_op_word,
                                                                   uint32_t num_faces,
                                                                   uint32_t face_r_dim)
{
    ft_tensix_math_eltwise_binary_configure_addrmod(0u);
    ft_tensix_math_eltwise_binary_configure_mop_addsub(binary_op_word, num_faces, face_r_dim);
    FT_BH_TTINSN_SETC16(FT_BH_CFG_SRCA_DVALID_CLEAR_DISABLE_WORD, 0);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

/** Run the eltwise MOP into dest tile `dst_index` of the current half. */
static FT_BH_INLINE void ft_tensix_math_eltwise_binary_addsub(uint32_t dst_index)
{
    uint32_t dst_offset = (dst_index << 6)
                          + ((ft_tensix_dest_offset_id != 0) ? FT_TENSIX_DEST_REGISTER_HALF_SIZE
                                                             : 0u);
    FT_BH_PUSH_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD, dst_offset);
    ft_tensix_mop_template_run();
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_D);
}

#endif /* FT_TENSIX_COMPUTE_MATH_H */
