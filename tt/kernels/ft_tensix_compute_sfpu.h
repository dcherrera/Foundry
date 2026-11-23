/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_compute_sfpu.h
 * @brief FPU-side sequences used around SFPU work: single-operand unpack (UNPACK thread), the
 *   SrcA-to-Dst datacopy and the eltwise multiply (MATH thread).
 *
 * The FPU always consumes both source registers, so a single-operand unpack fills SrcA with
 * one UNPACR at MOP start and, per face, issues an UNPACR_NOP that zeroes SrcB and marks it
 * valid (WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP.md). Datacopy is then an ELWADD of
 * SrcA with the zeroed SrcB (ELWADD.md). The eltwise multiply runs ELWMUL with the fidelity
 * phase count as the MOP outer loop (MOPExpander.md) and never accumulates into Dst.
 *
 * Kernels include this header once per thread section of a translation unit, so it is guarded.
 *
 * @status Clean-room rewrite (plan Phase 4, step 4.5b).
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_COMPUTE_SFPU_H
#define FT_TENSIX_COMPUTE_SFPU_H

#include <stdint.h>

#include "ft_tensix_circular_buffer.h"
#include "ft_tensix_compute.h"
#include "../fw/ft_bh_isa.h"

/* UNPACR_NOP field values (UNPACR_NOP.md): sub-op "zero the source" and "set data valid". */
#define FT_TENSIX_UNPACR_NOP_UNP_ZEROSRC 0x1u
#define FT_TENSIX_UNPACR_NOP_SET_DVALID 0x1u

/* Math fidelity (Foundry convention): the high-fidelity MOP outer loop count is this value.
 * ft_tensix_compute_broadcast.h defines the same macro under the same guard. */
#ifndef FT_TENSIX_MATH_FIDELITY_LOFI
#define FT_TENSIX_MATH_FIDELITY_LOFI 0u
#endif

/* ===== Unpack thread ===================================================================== */
#ifdef FT_TRISC_UNPACK

#include "ft_tensix_compute_unpack.h"

/** MOP: SrcA unpack at start, then one "zero SrcB, set valid" per face. */
static FT_BH_INLINE void ft_tensix_unpack_a_mop_config(uint32_t num_faces)
{
    uint32_t unpack_srca =
        FT_BH_ENC_UNPACR(FT_BH_UNPACR_SRC_A, 0x1, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0,
                         0, 0, 0, 1);
    uint32_t unpack_srcb_set_dvalid =
        FT_BH_ENC_UNPACR_NOP(FT_BH_UNPACR_SRC_B, 0, 0, FT_TENSIX_UNPACR_NOP_SET_DVALID, 0, 0, 0, 0,
                             FT_TENSIX_UNPACR_NOP_UNP_ZEROSRC);
    ft_tensix_mop_template t = ft_tensix_mop_template_1(num_faces, 1u, unpack_srcb_set_dvalid);
    t.start_op0 = unpack_srca;
    ft_tensix_mop_template_program(&t);
}

/** Configure the single-operand unpack: no in-face transpose, unpacker A row length of
 *  `face_r_dim` rows of 16 datums, `num_faces` faces per tile. */
static FT_BH_INLINE void ft_tensix_unpack_a_init(uint32_t face_r_dim, uint32_t num_faces)
{
    FT_TENSIX_CFG_RMW_FIELD(UNPACKER0_HALOIZE_MODE, 0u);
    switch (face_r_dim) {
    case 1:
        FT_BH_TTINSN_SETADCXX(FT_BH_SETADC_UNP_A, 1u * FT_TENSIX_FACE_C_DIM - 1u, 0x0);
        break;
    case 2:
        FT_BH_TTINSN_SETADCXX(FT_BH_SETADC_UNP_A, 2u * FT_TENSIX_FACE_C_DIM - 1u, 0x0);
        break;
    case 4:
        FT_BH_TTINSN_SETADCXX(FT_BH_SETADC_UNP_A, 4u * FT_TENSIX_FACE_C_DIM - 1u, 0x0);
        break;
    case 8:
        FT_BH_TTINSN_SETADCXX(FT_BH_SETADC_UNP_A, 8u * FT_TENSIX_FACE_C_DIM - 1u, 0x0);
        break;
    default:
        FT_BH_TTINSN_SETADCXX(FT_BH_SETADC_UNP_A, FT_TENSIX_FACE_R_DIM * FT_TENSIX_FACE_C_DIM - 1u,
                              0x0);
        break;
    }
    ft_tensix_unpack_a_mop_config(num_faces);
}

/** Unpack tile `tile_index` of CB `operand` into SrcA through the current config context,
 *  then flip the context. */
static FT_BH_INLINE void ft_tensix_unpack_a(uint32_t operand, uint32_t tile_index)
{
    uint32_t address = (ft_tensix_cb(operand)->fifo_read_pointer - 1u) +
                       ft_tensix_cb(operand)->fifo_page_size * tile_index;
    volatile uint32_t FT_BH_REG_PTR *cfg;

    FT_BH_TTINSN_SETADCZW(FT_BH_SETADC_UNP_AB, 0, 0, 0, 0, 0xF);
    cfg = ft_tensix_cfg_pointer();

    /* Wait for a free config context (two exist). */
    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_UNPACK_SYNC) >= 2u) {
    }
    if (unp_cfg_context == 0) {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_WORD] = address;
    } else {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_WORD] = address;
    }
    ft_tensix_semaphore_post(FT_TENSIX_SEM_UNPACK_SYNC);

    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_UNPACK, FT_BH_STALLWAIT_COND_TRISC_CFG);
    ft_tensix_mop_template_run();
    FT_BH_TTINSN_SEMGET(FT_TENSIX_T6_SEM(FT_TENSIX_SEM_UNPACK_SYNC));

    unp_cfg_context = 1u - unp_cfg_context;
    if (unp_cfg_context == 0) {
        FT_BH_TTINSN_SETC16(FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_WORD, 0x0000);
    } else {
        FT_BH_TTINSN_SETC16(FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_WORD, 0x0101);
    }
}

#endif /* FT_TRISC_UNPACK */

/* ===== Math thread ======================================================================= */
#ifdef FT_TRISC_MATH

#include "ft_tensix_compute_math.h"

/** Slots 3 (still), 0 (one row), 2 (one FPU block of 8 rows); slot 1 is left alone. */
static FT_BH_INLINE void ft_tensix_math_datacopy_configure_addrmod(void)
{
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_3, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_0, FT_TENSIX_ADDR_MOD_SRC_VAL(1, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(1, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_2,
                           FT_TENSIX_ADDR_MOD_SRC_VAL(FT_TENSIX_MAX_FPU_ROWS, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(FT_TENSIX_MAX_FPU_ROWS, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
}

/** MOP: per face, num_rows/8 ELWADDs of SrcA + zero SrcB, then release both sources. */
static FT_BH_INLINE void ft_tensix_math_datacopy_configure_mop(uint32_t num_faces,
                                                               uint32_t num_rows)
{
    ft_tensix_mop_template t = ft_tensix_mop_template_1(
        num_faces, num_rows >> FT_TENSIX_MAX_FPU_ROWS_LOG2,
        FT_BH_ENC_ELWADD(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_2, 0));
    t.end_op0 = FT_BH_ENC_SETRWC(FT_BH_SETRWC_CLR_AB, 0, 0, 0, 0, FT_BH_SETRWC_SET_AB);
    ft_tensix_mop_template_program(&t);
}

static FT_BH_INLINE void ft_tensix_math_datacopy_init(uint32_t num_faces)
{
    ft_tensix_math_datacopy_configure_addrmod();
    ft_tensix_math_datacopy_configure_mop(num_faces, FT_TENSIX_FACE_R_DIM);
    FT_BH_TTINSN_SETC16(FT_BH_CFG_SRCA_DVALID_CLEAR_DISABLE_WORD, 0);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

/** Copy the unpacked SrcA tile into Dst tile `dst_index`. */
static FT_BH_INLINE void ft_tensix_math_datacopy(uint32_t dst_index)
{
    uint32_t dst_offset = (dst_index << 6) + ((ft_tensix_dest_offset_id != 0)
                                                  ? FT_TENSIX_DEST_REGISTER_HALF_SIZE
                                                  : 0u);
    FT_BH_PUSH_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD, dst_offset);
    ft_tensix_mop_template_run();
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_D);
}

/** MOP for ELWMUL: outer loop = fidelity phases, inner = 8-row blocks of one face; the last
 *  instruction of the inner loop moves on (slot 2), the last of all releases SrcA/SrcB. */
static FT_BH_INLINE void ft_tensix_math_eltwise_binary_configure_mop_mul(uint32_t math_fidelity,
                                                                         uint32_t face_r_dim)
{
    uint32_t inner_loop = (face_r_dim > 8u) ? (face_r_dim >> 3) : 1u;
    ft_tensix_mop_template t = ft_tensix_mop_template_1(
        math_fidelity, inner_loop,
        FT_BH_ENC_ELWMUL(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_0, 0));
    t.loop1_last_instr = FT_BH_ENC_ELWMUL(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_2, 0);
    t.loop0_last_instr =
        FT_BH_ENC_ELWMUL(FT_BH_SETRWC_CLR_AB, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_3, 0);
    ft_tensix_mop_template_program(&t);
}

static FT_BH_INLINE void ft_tensix_math_eltwise_binary_mul_init(uint32_t math_fidelity,
                                                                uint32_t face_r_dim)
{
    ft_tensix_math_eltwise_binary_configure_addrmod(
        math_fidelity != FT_TENSIX_MATH_FIDELITY_LOFI ? 1u : 0u);
    ft_tensix_math_eltwise_binary_configure_mop_mul(math_fidelity, face_r_dim);
    FT_BH_TTINSN_SETC16(FT_BH_CFG_SRCA_DVALID_CLEAR_DISABLE_WORD, 0);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

/** Dst tile `dst_index` = SrcA * SrcB, face by face. */
static FT_BH_INLINE void ft_tensix_math_eltwise_binary_mul(uint32_t dst_index, uint32_t num_faces,
                                                           uint32_t face_r_dim)
{
    uint32_t dst_offset = (dst_index << 6) + ((ft_tensix_dest_offset_id != 0)
                                                  ? FT_TENSIX_DEST_REGISTER_HALF_SIZE
                                                  : 0u);
    uint32_t face;

    FT_BH_PUSH_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD, dst_offset);
    for (face = 0; face < num_faces; face++) {
        ft_tensix_mop_template_run();
        if (face_r_dim <= 8u) {
            FT_BH_TTINSN_INCRWC(FT_BH_SETRWC_CR_D, 8, 0, 0);
        }
    }
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_D);
}

#endif /* FT_TRISC_MATH */

#endif /* FT_TENSIX_COMPUTE_SFPU_H */
