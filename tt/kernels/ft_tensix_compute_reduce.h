/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_compute_reduce.h
 * @brief Row-wise reduce of one 32x32 tile into Dst column 0, sum and max arms, for all three
 *   TRISC threads.
 *
 * SUM: the row sum is a matmul against a scaler tile. MATH records 8 MVMULs into the replay
 * buffer and replays them from the MOP (WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md,
 * REPLAY.md); UNPACK sends the scaler to SrcA and the data to SrcB.
 * MAX: MATH pools each face with GMPOOL (GMPOOL.md), then turns the pooled row into a column
 * through SrcB (MOVD2B, TRNSPSRCB, ZEROSRC, ELWADD) with the Src zero flags disabled for the
 * moves (ft_tensix_compute_math.h owns that state). UNPACK sends data to SrcA, scaler to SrcB,
 * one face per MOP iteration.
 * PACK (both arms): an edge mask that keeps datum 0 of every row (WormholeB0/TensixTile/
 * TensixCoprocessor/Packers/EdgeMasking.md); ft_tensix_pack_reduce_mask_clear undoes it, and
 * tools/check_tensix_reduce_exit_state.py checks every kernel calling the config also calls
 * the clear.
 *
 * Replay slots [16, 32) are the MATH thread's (Foundry convention, as in
 * ft_tensix_compute_matmul.h); UNPACK records from slot 0.
 *
 * @status Clean-room rewrite (plan Phase 4, step 4.5b).
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_COMPUTE_REDUCE_H
#define FT_TENSIX_COMPUTE_REDUCE_H

#include <stdint.h>

#include "ft_tensix_circular_buffer.h"
#include "ft_tensix_compute.h"
#include "../fw/ft_bh_isa.h"

/* The SFPI compiler's replay-record intrinsic; same prototype as ft_tensix_compute_matmul.h. */
void __builtin_rvtt_ttreplay(volatile void *, unsigned, unsigned, unsigned, unsigned, unsigned,
                             unsigned);

/* REPLAY word that plays slots [start, start + length). */
#define FT_TENSIX_REPLAY_WORD(start, length) FT_BH_ENC_REPLAY((start), (length), 0, 0)

/* First MATH-thread FPU replay slot (Foundry convention). */
#define FT_TENSIX_REPLAY_MATH_OFFSET 16u

/** Record the next `length` Tensix instructions into replay slots [start, start + length). */
static FT_BH_INLINE void ft_tensix_replay_record(uint32_t start, uint32_t length)
{
    __builtin_rvtt_ttreplay(FT_BH_INSTRN_BUF, length, 0, 0, start, 0, 1);
}

/* SETRWC field values beyond ft_bh_isa.h (SETRWC.md: set-mask bits SrcA 0, SrcB 1, Dst 2;
 * counter-restore bit SrcB 1). */
#ifndef FT_TENSIX_SETRWC_SET_ABD
#define FT_TENSIX_SETRWC_SET_ABD 0x7u
#endif
#ifndef FT_TENSIX_SETRWC_SET_BD
#define FT_TENSIX_SETRWC_SET_BD 0x6u
#endif
#ifndef FT_TENSIX_SETRWC_CR_B
#define FT_TENSIX_SETRWC_CR_B 0x2u
#endif

/* GAPOOL / GMPOOL mode "pool a 16x16 face" and arg-max index off (GMPOOL.md). */
#ifndef FT_TENSIX_GPOOL_DIM_16X16
#define FT_TENSIX_GPOOL_DIM_16X16 0x1u
#endif
#ifndef FT_TENSIX_GPOOL_INDEX_DIS
#define FT_TENSIX_GPOOL_INDEX_DIS 0x0u
#endif

/* MOVD2B source row 16 and "move one row" mode (MOVD2B.md). */
#ifndef FT_TENSIX_MOVD2B_SRC_ROW16_OFFSET
#define FT_TENSIX_MOVD2B_SRC_ROW16_OFFSET 0x10u
#endif
#ifndef FT_TENSIX_MOVD2B_MOV_1_ROW
#define FT_TENSIX_MOVD2B_MOV_1_ROW 0x0u
#endif

/* Pack-thread scratch GPR, low half only (Foundry convention). */
#ifndef FT_TENSIX_GPR_PACK_TMP_LO
#define FT_TENSIX_GPR_PACK_TMP_LO 30u
#endif

/* ===== Math thread ======================================================================= */
#ifdef FT_TRISC_MATH

#include "ft_tensix_compute_math.h"

#ifndef FT_TENSIX_MATH_HAVE_MOV_OPS_ZERO_FLAG
#define FT_TENSIX_MATH_HAVE_MOV_OPS_ZERO_FLAG
/** Disable the Src zero flags for data moves, once per change of state. */
static FT_BH_INLINE void ft_tensix_math_configure_mov_ops_zero_flag_state(void)
{
    if (ft_tensix_src_zero_flag_state == FT_TENSIX_SRC_ZERO_FLAG_MOV_OPS) {
        return;
    }
    ft_tensix_src_zero_flag_state = FT_TENSIX_SRC_ZERO_FLAG_MOV_OPS;
    ft_tensix_math_configure_src_zero_flag(1);
}
#endif

/** Dst write offset of tile `dst_index` in the active Dst half. */
static FT_BH_INLINE uint32_t ft_tensix_math_dst_write_offset(uint32_t dst_index)
{
    return (dst_index << 6) +
           ((ft_tensix_dest_offset_id != 0) ? FT_TENSIX_DEST_REGISTER_HALF_SIZE : 0u);
}

/* ----- Sum arm ----- */

static FT_BH_INLINE void ft_tensix_math_reduce_row_configure_addrmod(uint32_t high_fidelity)
{
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_0, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(8, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(8, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_1, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(24, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(24, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_2, FT_TENSIX_ADDR_MOD_SRC_VAL(16, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(16, 0, 1),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 1, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_3, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 1, 1),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 1, 1),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 1, 1, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL((high_fidelity ? 1u : 0u), 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
}

/** Record the 8 MVMULs of one phase; the MOP replays them once per fidelity phase. */
static FT_BH_INLINE void ft_tensix_math_reduce_row_configure_mop(uint32_t high_fidelity,
                                                                 uint32_t fidelity_phases)
{
    const uint32_t replay_start = FT_TENSIX_REPLAY_MATH_OFFSET;
    const uint32_t replay_length = 8u;
    ft_tensix_mop_template t;

    ft_tensix_replay_record(replay_start, replay_length);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_1, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_2, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_1, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    if (high_fidelity) {
        FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_3, 0);
    } else {
        FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_AB, 0, FT_BH_ADDR_MOD_3, 0);
    }

    t = ft_tensix_mop_template_1(1u, high_fidelity ? fidelity_phases : 1u,
                                 FT_TENSIX_REPLAY_WORD(replay_start, replay_length));
    if (high_fidelity) {
        t.end_op0 = FT_BH_ENC_SETRWC(FT_BH_SETRWC_CLR_AB, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
    }
    ft_tensix_mop_template_program(&t);
}

static FT_BH_INLINE void ft_tensix_math_reduce_row_init(uint32_t high_fidelity,
                                                        uint32_t fidelity_phases)
{
    ft_tensix_math_reduce_row_configure_addrmod(high_fidelity);
    ft_tensix_math_reduce_row_configure_mop(high_fidelity, fidelity_phases);
    FT_BH_TTINSN_SETC16(FT_BH_CFG_SRCA_DVALID_CLEAR_DISABLE_WORD, 0);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

/** Accumulate the row sums of the unpacked tile into Dst tile `dst_index`. */
static FT_BH_INLINE void ft_tensix_math_reduce_row(uint32_t dst_index)
{
    FT_BH_PUSH_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD,
                      ft_tensix_math_dst_write_offset(dst_index));
    ft_tensix_mop_template_run();
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_TENSIX_SETRWC_SET_ABD);
}

/* ----- Max arm ----- */

/** Slot 0 only clears the fidelity phase; slot 1 steps SrcB and Dst by one face row block. */
static FT_BH_INLINE void ft_tensix_math_reduce_row_max_configure_addrmod(void)
{
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_0, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 1),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_1, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(8, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(8, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
}

/** A GAPOOL MOP. The max path never runs it, but programs it at high fidelity. */
static FT_BH_INLINE void ft_tensix_math_reduce_max_configure_mop(uint32_t fidelity_phases)
{
    uint32_t loop_op = FT_BH_ENC_GAPOOL(FT_BH_SETRWC_CLR_NONE, FT_TENSIX_GPOOL_DIM_16X16,
                                        FT_BH_ADDR_MOD_1, FT_TENSIX_GPOOL_INDEX_DIS, 0);
    uint32_t last_op = FT_BH_ENC_GAPOOL(FT_BH_SETRWC_CLR_NONE, FT_TENSIX_GPOOL_DIM_16X16,
                                        FT_BH_ADDR_MOD_0, FT_TENSIX_GPOOL_INDEX_DIS, 0);
    ft_tensix_mop_template t = ft_tensix_mop_template_1(1u, fidelity_phases, loop_op);
    t.loop1_last_instr = last_op;
    t.loop0_last_instr = last_op;
    ft_tensix_mop_template_program(&t);
}

static FT_BH_INLINE void ft_tensix_math_reduce_row_max_init(uint32_t high_fidelity,
                                                            uint32_t fidelity_phases)
{
    ft_tensix_math_reduce_row_max_configure_addrmod();
    if (high_fidelity) {
        ft_tensix_math_reduce_max_configure_mop(fidelity_phases);
    }
    FT_BH_TTINSN_SETC16(FT_BH_CFG_SRCA_DVALID_CLEAR_DISABLE_WORD, 0);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

/** Max-pool the faces of one face row into Dst; only the last pool keeps SrcA/SrcB. */
static FT_BH_INLINE void ft_tensix_math_reduce_row_max_pool_faces(uint32_t num_faces_c_dim)
{
    for (uint32_t face = 0; face + 1u < num_faces_c_dim; face++) {
        FT_BH_TTINSN_GMPOOL(FT_BH_SETRWC_CLR_AB, FT_TENSIX_GPOOL_DIM_16X16, FT_BH_ADDR_MOD_0,
                            FT_TENSIX_GPOOL_INDEX_DIS, 0);
    }
    FT_BH_TTINSN_GMPOOL(FT_BH_SETRWC_CLR_NONE, FT_TENSIX_GPOOL_DIM_16X16, FT_BH_ADDR_MOD_0,
                        FT_TENSIX_GPOOL_INDEX_DIS, 0);
}

/** Turn the pooled row into a column: Dst row -> SrcB, transpose SrcB, add back into Dst
 *  against a zeroed SrcA, with the zero flags disabled for the moves. */
static FT_BH_INLINE void ft_tensix_math_reduce_row_max_transpose(void)
{
    ft_tensix_math_configure_mov_ops_zero_flag_state();
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, FT_BH_SETRWC_CR_D, 0, 0, 0, FT_BH_SETRWC_SET_AB);
    FT_BH_TTINSN_MOVD2B(0, FT_TENSIX_MOVD2B_SRC_ROW16_OFFSET, FT_BH_ADDR_MOD_0,
                        FT_TENSIX_MOVD2B_MOV_1_ROW, 0);
    FT_BH_TTINSN_TRNSPSRCB;
    FT_BH_TTINSN_MOVD2B(0, FT_TENSIX_MOVD2B_SRC_ROW16_OFFSET, FT_BH_ADDR_MOD_0,
                        FT_TENSIX_MOVD2B_MOV_1_ROW, 0);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, FT_TENSIX_SETRWC_CR_B, 0, 8, 0,
                        FT_BH_SETRWC_SET_B);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, FT_TENSIX_SETRWC_CR_B, 0, 8, 0,
                        FT_BH_SETRWC_SET_B);
    FT_BH_TTINSN_ZEROSRC(0, 1, 0, 1);
    FT_BH_TTINSN_ELWADD(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_1, 0);
    FT_BH_TTINSN_ELWADD(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_1, 0);
    ft_tensix_math_configure_default_zero_flag_state(ft_tensix_src_zero_flag_srca_format,
                                                     ft_tensix_src_zero_flag_srcb_format);
}

/** Step Dst to the second face row (16 rows for a narrow tile, 32 otherwise). */
static FT_BH_INLINE void ft_tensix_math_reduce_row_max_advance_dest(uint32_t is_narrow_tile)
{
    if (!is_narrow_tile) {
        FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, FT_BH_SETRWC_CR_D, 8, 0, 0,
                            FT_BH_SETRWC_SET_D);
        FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, FT_BH_SETRWC_CR_D, 8, 0, 0,
                            FT_BH_SETRWC_SET_D);
    }
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, FT_BH_SETRWC_CR_D, 8, 0, 0, FT_BH_SETRWC_SET_D);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_AB, FT_BH_SETRWC_CR_D, 8, 0, 0, FT_TENSIX_SETRWC_SET_BD);
}

/** Row maxima of the unpacked tile into column 0 of Dst tile `dst_index`. */
static FT_BH_INLINE void ft_tensix_math_reduce_row_max(uint32_t dst_index,
                                                       uint32_t num_faces_r_dim,
                                                       uint32_t num_faces_c_dim)
{
    const uint32_t is_narrow_tile = (num_faces_c_dim < num_faces_r_dim) ? 1u : 0u;

    FT_BH_PUSH_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD,
                      ft_tensix_math_dst_write_offset(dst_index));
    ft_tensix_math_reduce_row_max_pool_faces(num_faces_c_dim);
    ft_tensix_math_reduce_row_max_transpose();
    if (num_faces_r_dim > 1u) {
        ft_tensix_math_reduce_row_max_advance_dest(is_narrow_tile);
        ft_tensix_math_reduce_row_max_pool_faces(num_faces_c_dim);
        ft_tensix_math_reduce_row_max_transpose();
    }
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_AB, 0, 0, 0, 0, FT_TENSIX_SETRWC_SET_BD);
}

/** Nothing to undo on the math thread. */
static FT_BH_INLINE void ft_tensix_math_reduce_uninit(void)
{
}

#endif /* FT_TRISC_MATH */

/* ===== Unpack thread ===================================================================== */
#ifdef FT_TRISC_UNPACK

/* ----- Sum arm ----- */

/** Replay program: SrcA then SrcB, whole tile each; the MOP plays it once. */
static FT_BH_INLINE void ft_tensix_unpack_ab_reduce_mop_config(void)
{
    ft_tensix_mop_template t;

    ft_tensix_replay_record(0u, 2u);
    FT_BH_TTINSN_UNPACR(FT_BH_UNPACR_SRC_A, 0, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0, 0,
                        0, 0, 1);
    FT_BH_TTINSN_UNPACR(FT_BH_UNPACR_SRC_B, 0, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0, 0,
                        0, 0, 1);
    t = ft_tensix_mop_template_1(1u, 1u, FT_TENSIX_REPLAY_WORD(0u, 2u));
    ft_tensix_mop_template_program(&t);
}

static FT_BH_INLINE void ft_tensix_unpack_ab_reduce_row_init(uint32_t num_faces)
{
    uint32_t x_end = num_faces * FT_TENSIX_FACE_R_DIM * FT_TENSIX_FACE_C_DIM - 1u;

    FT_TENSIX_CFG_RMW_FIELD(UNPACKER0_HALOIZE_MODE, 1u);
    FT_BH_PUSH_SETADCXX(FT_BH_SETADC_UNP_A, x_end, 0x0);
    FT_BH_PUSH_SETADCXX(FT_BH_SETADC_UNP_B, x_end, 0x0);
    ft_tensix_unpack_ab_reduce_mop_config();
}

/** Unpack the scaler tile to SrcA and the data tile to SrcB, then flip the config context. */
static FT_BH_INLINE void ft_tensix_unpack_ab_reduce_row(uint32_t data_cb, uint32_t scaler_cb,
                                                        uint32_t data_tile_index,
                                                        uint32_t scaler_tile_index)
{
    uint32_t address_data = (ft_tensix_cb(data_cb)->fifo_read_pointer - 1u) +
                            ft_tensix_cb(data_cb)->fifo_page_size * data_tile_index;
    uint32_t address_scaler = (ft_tensix_cb(scaler_cb)->fifo_read_pointer - 1u) +
                              ft_tensix_cb(scaler_cb)->fifo_page_size * scaler_tile_index;
    volatile uint32_t FT_BH_REG_PTR *cfg;

    FT_BH_TTINSN_SETADCZW(FT_BH_SETADC_UNP_AB, 0, 0, 0, 0, 0xF);
    cfg = ft_tensix_cfg_pointer();

    /* Wait for a free config context (two exist). */
    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_UNPACK_SYNC) >= 2u) {
    }
    if (unp_cfg_context == 0) {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_WORD] = address_scaler;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT0_WORD] = address_data;
    } else {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_WORD] = address_scaler;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT1_WORD] = address_data;
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

/* ----- Max arm ----- */

/* Row length of unpacker `unp_sel`: `face_r_dim` rows of 16 datums. A switch of immediates
 * (unp_sel is an instruction field), not arithmetic, so each arm stays a .ttinsn. */
#define FT_TENSIX_UNPACK_CONFIG_X_END(unp_sel, face_r_dim)                                     \
    do {                                                                                       \
        switch (face_r_dim) {                                                                  \
        case 1:                                                                                \
            FT_BH_TTINSN_SETADCXX((unp_sel), 1u * FT_TENSIX_FACE_C_DIM - 1u, 0x0);             \
            break;                                                                             \
        case 2:                                                                                \
            FT_BH_TTINSN_SETADCXX((unp_sel), 2u * FT_TENSIX_FACE_C_DIM - 1u, 0x0);             \
            break;                                                                             \
        case 4:                                                                                \
            FT_BH_TTINSN_SETADCXX((unp_sel), 4u * FT_TENSIX_FACE_C_DIM - 1u, 0x0);             \
            break;                                                                             \
        case 8:                                                                                \
            FT_BH_TTINSN_SETADCXX((unp_sel), 8u * FT_TENSIX_FACE_C_DIM - 1u, 0x0);             \
            break;                                                                             \
        default:                                                                               \
            FT_BH_TTINSN_SETADCXX((unp_sel), FT_TENSIX_FACE_R_DIM * FT_TENSIX_FACE_C_DIM - 1u, \
                                  0x0);                                                        \
            break;                                                                             \
        }                                                                                      \
    } while (0)

/** Replay program: SrcA then SrcB, one face each; the MOP plays it once per face. */
static FT_BH_INLINE void ft_tensix_unpack_ab_reduce_max_mop_config(uint32_t num_faces)
{
    ft_tensix_mop_template t;

    ft_tensix_replay_record(0u, 2u);
    FT_BH_TTINSN_UNPACR(FT_BH_UNPACR_SRC_A, 0x1, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0,
                        0, 0, 0, 1);
    FT_BH_TTINSN_UNPACR(FT_BH_UNPACR_SRC_B, 0x1, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0,
                        0, 0, 0, 1);
    t = ft_tensix_mop_template_1(1u, num_faces, FT_TENSIX_REPLAY_WORD(0u, 2u));
    ft_tensix_mop_template_program(&t);
}

static FT_BH_INLINE void ft_tensix_unpack_ab_reduce_row_max_init(uint32_t num_faces,
                                                                 uint32_t face_r_dim)
{
    FT_TENSIX_CFG_RMW_FIELD(UNPACKER0_HALOIZE_MODE, 1u);
    FT_TENSIX_UNPACK_CONFIG_X_END(FT_BH_SETADC_UNP_A, face_r_dim);
    FT_TENSIX_UNPACK_CONFIG_X_END(FT_BH_SETADC_UNP_B, 1u);
    ft_tensix_unpack_ab_reduce_max_mop_config(num_faces);
}

/** Unpack the data tile to SrcA and the scaler tile to SrcB, then flip the config context. */
static FT_BH_INLINE void ft_tensix_unpack_ab_reduce_row_max(uint32_t data_cb, uint32_t scaler_cb,
                                                            uint32_t data_tile_index,
                                                            uint32_t scaler_tile_index)
{
    uint32_t address_data = (ft_tensix_cb(data_cb)->fifo_read_pointer - 1u) +
                            ft_tensix_cb(data_cb)->fifo_page_size * data_tile_index;
    uint32_t address_scaler = (ft_tensix_cb(scaler_cb)->fifo_read_pointer - 1u) +
                              ft_tensix_cb(scaler_cb)->fifo_page_size * scaler_tile_index;
    volatile uint32_t FT_BH_REG_PTR *cfg;

    FT_BH_TTINSN_SETADCZW(FT_BH_SETADC_UNP_AB, 0, 0, 0, 0, 0xF);
    cfg = ft_tensix_cfg_pointer();

    /* Wait for a free config context (two exist). */
    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_UNPACK_SYNC) >= 2u) {
    }
    if (unp_cfg_context == 0) {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_WORD] = address_data;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT0_WORD] = address_scaler;
    } else {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_WORD] = address_data;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT1_WORD] = address_scaler;
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

/* ===== Pack thread ======================================================================= */
#ifdef FT_TRISC_PACK

/** Keep only datum 0 of every row: edge-mask set 0 masks everything, set 1 keeps datum 0, and
 *  every row of the tile selects set 1. Undo with ft_tensix_pack_reduce_mask_clear. */
static FT_BH_INLINE void ft_tensix_pack_reduce_row_mask_config(uint32_t face_r_dim)
{
    /* Edge-offset word: mask 0, mode 0, the four per-packer row-set select bits set. */
    const uint32_t pack_edge_offset = (1u << 17) | (1u << 19) | (1u << 21) | (1u << 23);
    const uint32_t edge_offset_sec1_mask = 0x0001u;
    const uint32_t row_set_mapping_1 = 0x55555555u;

    FT_BH_TTINSN_SETDMAREG(0, FT_TENSIX_LOWER_HALFWORD(pack_edge_offset), 0,
                           FT_TENSIX_LO_16(FT_TENSIX_GPR_PACK_TMP0));
    FT_BH_TTINSN_SETDMAREG(0, FT_TENSIX_UPPER_HALFWORD(pack_edge_offset), 0,
                           FT_TENSIX_HI_16(FT_TENSIX_GPR_PACK_TMP0));
    FT_BH_TTINSN_SETDMAREG(0, FT_TENSIX_LOWER_HALFWORD(edge_offset_sec1_mask), 0,
                           FT_TENSIX_LO_16(FT_TENSIX_GPR_PACK_TMP_LO));
    FT_BH_TTINSN_SETDMAREG(0, FT_TENSIX_LOWER_HALFWORD(row_set_mapping_1), 0,
                           FT_TENSIX_LO_16(FT_TENSIX_GPR_PACK_TMP1));
    FT_BH_TTINSN_SETDMAREG(0, FT_TENSIX_UPPER_HALFWORD(row_set_mapping_1), 0,
                           FT_TENSIX_HI_16(FT_TENSIX_GPR_PACK_TMP1));
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG, FT_BH_STALLWAIT_COND_PACK);
    FT_TENSIX_CFG_RMW_FIELD(PACKER0_READS_PER_XY_PLANE, face_r_dim);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_PACK_TMP0, FT_BH_WRCFG_32B, FT_BH_CFG_PACKER_EDGE_MASK0_WORD);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_PACK_TMP_LO, FT_BH_WRCFG_32B,
                       FT_BH_CFG_PACKER_EDGE_MASK1_WORD);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_PACK_TMP1, FT_BH_WRCFG_32B,
                       FT_BH_CFG_PACKER_TILE_ROW_MAP1_WORD);
    FT_BH_TTINSN_NOP;
    FT_BH_TTINSN_NOP;
}

/** Restore the full-row edge masks, the default row mapping and one read per XY plane. */
static FT_BH_INLINE void ft_tensix_pack_reduce_mask_clear(void)
{
    const uint32_t pack_edge_offset = 0x0000ffffu;

    FT_BH_TTINSN_SETDMAREG(0, FT_TENSIX_LOWER_HALFWORD(pack_edge_offset), 0,
                           FT_TENSIX_LO_16(FT_TENSIX_GPR_PACK_TMP0));
    FT_BH_TTINSN_SETDMAREG(0, FT_TENSIX_UPPER_HALFWORD(pack_edge_offset), 0,
                           FT_TENSIX_HI_16(FT_TENSIX_GPR_PACK_TMP0));
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG, FT_BH_STALLWAIT_COND_PACK);
    FT_TENSIX_CFG_RMW_FIELD(PACKER0_READS_PER_XY_PLANE, 1u);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_PACK_TMP0, FT_BH_WRCFG_32B, FT_BH_CFG_PACKER_EDGE_MASK0_WORD);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_PACK_TMP0, FT_BH_WRCFG_32B, FT_BH_CFG_PACKER_EDGE_MASK1_WORD);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_ZERO, FT_BH_WRCFG_32B, FT_BH_CFG_PACKER_TILE_ROW_MAP0_WORD);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_ZERO, FT_BH_WRCFG_32B, FT_BH_CFG_PACKER_TILE_ROW_MAP1_WORD);
    FT_BH_TTINSN_NOP;
    FT_BH_TTINSN_NOP;
}

#endif /* FT_TRISC_PACK */

#endif /* FT_TENSIX_COMPUTE_REDUCE_H */
