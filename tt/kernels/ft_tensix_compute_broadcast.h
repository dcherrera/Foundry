/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_compute_broadcast.h
 * @brief Column-broadcast eltwise binary: multiply or subtract a whole tile by one column of
 *   a second tile (SrcB column broadcast). UNPACK-side MOP/init and MATH-side MOP/init/run;
 *   PACK is not involved.
 *
 * Against the no-broadcast path: the unpack MOP loops only the SrcA unpack, unpacks SrcB once
 * as the start op and re-arms SrcB's Z counter as the end op; the face close clears SrcA only
 * (the SrcB column is reused by the next face row); the math run loops per face row with a
 * SrcB clear between rows. ELWSUB is LoFi only; ELWMUL loops over the build's fidelity
 * phases. ADDR_MOD slots 0..3 are the no-broadcast table (ft_tensix_compute_math.h).
 *
 * @status Clean-room rewrite, plan Phase 4 step 4.5a.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_COMPUTE_BROADCAST_H
#define FT_TENSIX_COMPUTE_BROADCAST_H

#include <stdint.h>

#include "ft_tensix_compute.h"
#ifdef FT_TRISC_MATH
#include "ft_tensix_compute_math.h"
#endif
#ifdef FT_TRISC_UNPACK
#include "ft_tensix_compute_unpack.h"
#endif

/* Math fidelity codes (Foundry convention, the FT_CB_MATH_FIDELITY encoding). Other headers
 * define the same names with the same tokens. */
#ifndef FT_TENSIX_MATH_FIDELITY_LOFI
#define FT_TENSIX_MATH_FIDELITY_LOFI 0u
#endif

/* ===== UNPACK ============================================================================ */
#ifdef FT_TRISC_UNPACK

/* Same UNPACR words as the two-operand path. */
#define FT_TENSIX_BCAST_UNPACR(which)                                                           \
    FT_BH_ENC_UNPACR((which), 1, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0, 0, 0, 0, 1)

static FT_BH_INLINE void ft_tensix_unpack_ab_broadcast_col_mop_config(uint32_t num_faces_r_dim,
                                                                      uint32_t num_faces_c_dim)
{
    ft_tensix_mop_template t = ft_tensix_mop_template_1(
        num_faces_r_dim, num_faces_c_dim, FT_TENSIX_BCAST_UNPACR(FT_BH_UNPACR_SRC_A));
    t.start_op0 = FT_TENSIX_BCAST_UNPACR(FT_BH_UNPACR_SRC_B);
    t.end_op0 = FT_BH_ENC_SETADCZW(FT_BH_SETADC_UNP_B, 0, 0, 0, num_faces_c_dim, 0x1);
    ft_tensix_mop_template_program(&t);
}

/** Face height comes from operand A for both unpackers. */
static FT_BH_INLINE void ft_tensix_unpack_ab_broadcast_col_init(uint32_t face_r_dim,
                                                                uint32_t num_faces_r_dim,
                                                                uint32_t num_faces_c_dim)
{
    FT_TENSIX_CFG_RMW_FIELD(UNPACKER0_HALOIZE_MODE, 0u);
    switch (face_r_dim) {
    case 1:
        FT_BH_TTINSN_SETADCXX(FT_BH_SETADC_UNP_AB, 1 * FT_TENSIX_FACE_C_DIM - 1, 0);
        break;
    case 2:
        FT_BH_TTINSN_SETADCXX(FT_BH_SETADC_UNP_AB, 2 * FT_TENSIX_FACE_C_DIM - 1, 0);
        break;
    case 4:
        FT_BH_TTINSN_SETADCXX(FT_BH_SETADC_UNP_AB, 4 * FT_TENSIX_FACE_C_DIM - 1, 0);
        break;
    case 8:
        FT_BH_TTINSN_SETADCXX(FT_BH_SETADC_UNP_AB, 8 * FT_TENSIX_FACE_C_DIM - 1, 0);
        break;
    default:
        FT_BH_TTINSN_SETADCXX(FT_BH_SETADC_UNP_AB, 16 * FT_TENSIX_FACE_C_DIM - 1, 0);
        break;
    }
    ft_tensix_unpack_ab_broadcast_col_mop_config(num_faces_r_dim, num_faces_c_dim);
}

#endif /* FT_TRISC_UNPACK */

/* ===== MATH ============================================================================== */
#ifdef FT_TRISC_MATH

#define FT_TENSIX_BCAST_INNER(face_r_dim)                                                       \
    (((face_r_dim) > FT_TENSIX_MAX_FPU_ROWS) ? ((face_r_dim) >> FT_TENSIX_MAX_FPU_ROWS_LOG2) : 1u)
#define FT_TENSIX_BCAST_DST_OFFSET(dst_index)                                                   \
    (((dst_index) << 6)                                                                         \
     + ((ft_tensix_dest_offset_id != 0) ? FT_TENSIX_DEST_REGISTER_HALF_SIZE : 0u))

/* --- ELWMUL --- */

/** Outer loop = the fidelity code = the number of fidelity phases. */
static FT_BH_INLINE void ft_tensix_math_eltwise_binary_configure_mop_mul_broadcast_col(
    uint32_t math_fidelity, uint32_t face_r_dim)
{
    ft_tensix_mop_template t = ft_tensix_mop_template_1(
        math_fidelity, FT_TENSIX_BCAST_INNER(face_r_dim),
        FT_BH_ENC_ELWMUL(0, 0, FT_BH_ELWADD_SRCB_BCAST_COL, FT_BH_ADDR_MOD_0, 0));
    t.loop1_last_instr = FT_BH_ENC_ELWMUL(0, 0, FT_BH_ELWADD_SRCB_BCAST_COL, FT_BH_ADDR_MOD_2, 0);
    t.loop0_last_instr = FT_BH_ENC_ELWMUL(FT_BH_SETRWC_CLR_A, 0, FT_BH_ELWADD_SRCB_BCAST_COL,
                                          FT_BH_ADDR_MOD_3, 0);
    ft_tensix_mop_template_program(&t);
}

static FT_BH_INLINE void ft_tensix_math_eltwise_binary_mul_broadcast_col_init(
    uint32_t math_fidelity, uint32_t face_r_dim)
{
    ft_tensix_math_eltwise_binary_configure_addrmod(
        math_fidelity != FT_TENSIX_MATH_FIDELITY_LOFI ? 1u : 0u);
    ft_tensix_math_eltwise_binary_configure_mop_mul_broadcast_col(math_fidelity, face_r_dim);
    FT_BH_TTINSN_SETC16(FT_BH_CFG_SRCA_DVALID_CLEAR_DISABLE_WORD, 0);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

static FT_BH_INLINE void ft_tensix_math_eltwise_binary_mul_broadcast_col(
    uint32_t dst_index, uint32_t math_fidelity, uint32_t num_faces_r_dim,
    uint32_t num_faces_c_dim, uint32_t face_r_dim)
{
    uint32_t off = FT_TENSIX_BCAST_DST_OFFSET(dst_index);
    uint32_t fidelity_loop =
        (math_fidelity != FT_TENSIX_MATH_FIDELITY_LOFI) ? num_faces_c_dim : 1u;
    uint32_t face_row;
    uint32_t phase;

    FT_BH_PUSH_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD, off);
    for (face_row = 0; face_row < num_faces_r_dim; face_row++) {
        for (phase = 0; phase < fidelity_loop; phase++) {
            ft_tensix_mop_template_run();
            if (math_fidelity != FT_TENSIX_MATH_FIDELITY_LOFI
                && face_r_dim <= FT_TENSIX_MAX_FPU_ROWS) {
                FT_BH_TTINSN_INCRWC(FT_BH_SETRWC_CR_D, FT_TENSIX_MAX_FPU_ROWS, 0, 0);
            }
        }
        FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_B, 0, 0, 0, 0, 0);
    }
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_D);
}

/* --- ELWSUB (LoFi only) --- */

static FT_BH_INLINE void ft_tensix_math_eltwise_binary_configure_mop_sub_broadcast_col(
    uint32_t num_faces_c_dim, uint32_t face_r_dim)
{
    ft_tensix_mop_template t = ft_tensix_mop_template_1(
        num_faces_c_dim, FT_TENSIX_BCAST_INNER(face_r_dim),
        FT_BH_ENC_ELWSUB(0, 0, FT_BH_ELWADD_SRCB_BCAST_COL, FT_BH_ADDR_MOD_0, 0));
    if (face_r_dim <= FT_TENSIX_MAX_FPU_ROWS) {
        t.loop_op1 = FT_BH_ENC_INCRWC(0, FT_TENSIX_MAX_FPU_ROWS, FT_TENSIX_MAX_FPU_ROWS,
                                      FT_TENSIX_MAX_FPU_ROWS);
    }
    t.end_op0 = FT_BH_ENC_SETRWC(FT_BH_SETRWC_CLR_A, FT_BH_SETRWC_CR_AB, 0, 0, 0,
                                 FT_BH_SETRWC_SET_AB);
    t.end_op1 = FT_BH_ENC_NOP;
    ft_tensix_mop_template_program(&t);
}

static FT_BH_INLINE void ft_tensix_math_eltwise_binary_sub_broadcast_col_init(
    uint32_t num_faces_c_dim, uint32_t face_r_dim)
{
    ft_tensix_math_eltwise_binary_configure_addrmod(0u);
    ft_tensix_math_eltwise_binary_configure_mop_sub_broadcast_col(num_faces_c_dim, face_r_dim);
    FT_BH_TTINSN_SETC16(FT_BH_CFG_SRCA_DVALID_CLEAR_DISABLE_WORD, 0);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

static FT_BH_INLINE void ft_tensix_math_eltwise_binary_sub_broadcast_col(uint32_t dst_index,
                                                                         uint32_t num_faces_r_dim)
{
    uint32_t face_row;
    FT_BH_PUSH_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD, FT_TENSIX_BCAST_DST_OFFSET(dst_index));
    for (face_row = 0; face_row < num_faces_r_dim; face_row++) {
        ft_tensix_mop_template_run();
        FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_B, 0, 0, 0, 0, 0);
    }
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_D);
}

#endif /* FT_TRISC_MATH */

#endif /* FT_TENSIX_COMPUTE_BROADCAST_H */
