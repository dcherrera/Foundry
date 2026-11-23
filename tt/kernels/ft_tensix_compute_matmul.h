/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_compute_matmul.h
 * @brief Tile matrix multiply for the MATH and UNPACK threads: address-modifier programming,
 *   replay-buffer recording, MOP templates and per-tile issue.
 *
 * MATH: a 32x32 by 32x32 tile product is 16 MVMULs (WormholeB0/TensixTile/TensixCoprocessor/
 * MVMUL.md), each consuming 8 rows; their SrcA/SrcB/Dst walk is encoded in address-modifier
 * slots 0, 1, 2, 4 and 5 (RWCs.md, slot 5 also advances the fidelity phase). The 16 MVMULs
 * are recorded once into the replay buffer (REPLAY.md) and the MOP loop instruction replays
 * them (MOPExpander.md). A 16-row in0 variant uses 8 MVMULs.
 *
 * UNPACK: a 12-instruction program is recorded into the replay buffer, two 6-instruction
 * halves, one per unpacker config context. Each half unpacks SrcA, then advances unpacker 0's
 * L1 base address by the A tile size (GPR 36) so the MOP can walk the K dimension. The
 * operands are deliberately crossed: in0 goes to SrcB (unpacker 1) and in1 to SrcA
 * (unpacker 0), since the FPU computes SrcB x SrcA.
 *
 * Replay slots: [0, 16) belong to the UNPACK program and the SFPU, [16, 32) to the MATH
 * thread's FPU sequences (Foundry convention). GPR numbers are Foundry conventions.
 *
 * @status Clean-room rewrite (plan Phase 4, step 4.5b).
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_COMPUTE_MATMUL_H
#define FT_TENSIX_COMPUTE_MATMUL_H

#include <stdint.h>

#include "ft_tensix_circular_buffer.h"
#include "ft_tensix_compute.h"
#include "../fw/ft_bh_isa.h"

/* The SFPI compiler's replay-record intrinsic. Its usual declaration cannot be included from C,
 * so it is declared here (ft_tensix_compute_reduce.h declares the same prototype). With
 * record = 1 the next `length` Tensix instructions of this thread are stored into replay slots
 * [start, start + length) instead of executed. */
void __builtin_rvtt_ttreplay(volatile void *instruction_buffer, unsigned length, unsigned unused0,
                             unsigned unused1, unsigned start, unsigned execute_while_loading,
                             unsigned record);

#define FT_TENSIX_REPLAY_RECORD(start, length)                                                 \
    __builtin_rvtt_ttreplay((volatile void *)__instrn_buffer, (length), 0, 0, (start), 0, 1)

/* REPLAY word that plays slots [start, start + length) (a data value for the MOP). */
#define FT_TENSIX_REPLAY_INSN(start, length) FT_BH_ENC_REPLAY((start), (length), 0, 0)

/* Replay layout (Foundry convention). */
#define FT_TENSIX_MATH_REPLAY_BUF_OFFSET 16u
#define FT_TENSIX_MATH_MATMUL_REPLAY_LEN 16u
#define FT_TENSIX_MATH_MATMUL_HALF_REPLAY_LEN 8u
#define FT_TENSIX_UNPACK_MATMUL_REPLAY_PROG_LEN 12u
#define FT_TENSIX_UNPACK_MATMUL_REPLAY_RUN_LEN 6u

/* Unpack-thread GPRs (Foundry convention). */
#ifndef FT_TENSIX_GPR_UNPACK_TMP0
#define FT_TENSIX_GPR_UNPACK_TMP0 12u
#endif
#ifndef FT_TENSIX_GPR_UNPACK_KT_DIM
#define FT_TENSIX_GPR_UNPACK_KT_DIM 38u
#endif

/* ===== Math thread ======================================================================= */
#ifdef FT_TRISC_MATH

/** Address modifiers of the full-tile walk. `transpose` swaps the SrcA steps of slots 1, 4. */
static FT_BH_INLINE void ft_tensix_math_matmul_configure_addrmod(uint32_t transpose,
                                                                 uint32_t fidelity_increment)
{
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_0, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(8, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(8, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_5, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 1, 1),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 1, 1),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 1, 1, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(fidelity_increment, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    if (transpose != 0) {
        FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_1, FT_TENSIX_ADDR_MOD_SRC_VAL(32, 0, 0),
                               FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 1),
                               FT_TENSIX_ADDR_MOD_DEST_VAL(8, 0, 0, 0),
                               FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                               FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    } else {
        FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_1, FT_TENSIX_ADDR_MOD_SRC_VAL(16, 0, 0),
                               FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 1),
                               FT_TENSIX_ADDR_MOD_DEST_VAL(8, 0, 0, 0),
                               FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                               FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    }
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_2, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 1),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(32, 0, 1),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(8, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    if (transpose != 0) {
        FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_4, FT_TENSIX_ADDR_MOD_SRC_VAL(16, 0, 1),
                               FT_TENSIX_ADDR_MOD_SRC_VAL(48, 0, 1),
                               FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 1, 0),
                               FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                               FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    } else {
        FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_4, FT_TENSIX_ADDR_MOD_SRC_VAL(32, 0, 1),
                               FT_TENSIX_ADDR_MOD_SRC_VAL(48, 0, 1),
                               FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 1, 0),
                               FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                               FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    }
}

/** Record the 16 MVMULs of one fidelity phase and program a MOP that replays them
 *  `inner_loops` times (once per fidelity phase at high fidelity). At low fidelity the last
 *  MVMUL releases SrcA; at high fidelity the MOP end op does, after all phases. */
static FT_BH_INLINE void ft_tensix_math_matmul_configure_mop(uint32_t inner_loops,
                                                             int high_fidelity)
{
    ft_tensix_mop_template t;

    FT_TENSIX_REPLAY_RECORD(FT_TENSIX_MATH_REPLAY_BUF_OFFSET, FT_TENSIX_MATH_MATMUL_REPLAY_LEN);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_1, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_2, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_1, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_4, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_1, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_2, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_1, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    if (high_fidelity) {
        FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_5, 0);
    } else {
        FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_A, 0, FT_BH_ADDR_MOD_5, 0);
    }

    t = ft_tensix_mop_template_1(1u, inner_loops,
                                 FT_TENSIX_REPLAY_INSN(FT_TENSIX_MATH_REPLAY_BUF_OFFSET,
                                                       FT_TENSIX_MATH_MATMUL_REPLAY_LEN));
    if (high_fidelity) {
        t.end_op0 = FT_BH_ENC_SETRWC(FT_BH_SETRWC_CLR_A, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
        t.end_op1 = FT_BH_ENC_NOP;
    }
    ft_tensix_mop_template_program(&t);
}

static FT_BH_INLINE void ft_tensix_math_matmul_init(uint32_t transpose,
                                                    uint32_t fidelity_increment,
                                                    uint32_t inner_loops, int high_fidelity)
{
    ft_tensix_math_matmul_configure_addrmod(transpose, fidelity_increment);
    ft_tensix_math_matmul_configure_mop(inner_loops, high_fidelity);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

/** Accumulate SrcB x SrcA into Dst tile `dst_index` (Dst is not cleared here). */
static FT_BH_INLINE void ft_tensix_math_matmul(uint32_t dst_index)
{
    uint32_t dst_offset = (dst_index << 6) + ((ft_tensix_dest_offset_id != 0)
                                                  ? FT_TENSIX_DEST_REGISTER_HALF_SIZE
                                                  : 0u);
    FT_BH_PUSH_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD, dst_offset);
    ft_tensix_mop_template_run();
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_B, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

/** One row of `ct_dim` output tiles from Dst tile `dst_index` on, then release SrcB. */
static FT_BH_INLINE void ft_tensix_math_matmul_block(uint32_t dst_index, uint32_t ct_dim)
{
    uint32_t column;
    for (column = 0u; column < ct_dim; column++) {
        uint32_t dst_offset = ((dst_index + column) << 6) +
                              ((ft_tensix_dest_offset_id != 0) ? FT_TENSIX_DEST_REGISTER_HALF_SIZE
                                                               : 0u);
        FT_BH_PUSH_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD, dst_offset);
        ft_tensix_mop_template_run();
    }
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_B, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

/** Address modifiers of the 16-row in0 walk. */
static FT_BH_INLINE void ft_tensix_math_matmul_configure_addrmod_in0_16row(
    uint32_t fidelity_increment)
{
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_0, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(8, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(8, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_5, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 1, 1),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 1, 1),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 1, 1, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(fidelity_increment, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_1, FT_TENSIX_ADDR_MOD_SRC_VAL(16, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 1),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(8, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_2, FT_TENSIX_ADDR_MOD_SRC_VAL(16, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 1),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(8, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_4, FT_TENSIX_ADDR_MOD_SRC_VAL(16, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(16, 0, 1),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 1, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0),
                           FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
}

/** As ft_tensix_math_matmul_configure_mop with 8 MVMULs per phase. */
static FT_BH_INLINE void ft_tensix_math_matmul_configure_mop_in0_16row(uint32_t inner_loops,
                                                                       int high_fidelity)
{
    ft_tensix_mop_template t;

    FT_TENSIX_REPLAY_RECORD(FT_TENSIX_MATH_REPLAY_BUF_OFFSET,
                            FT_TENSIX_MATH_MATMUL_HALF_REPLAY_LEN);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_2, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_4, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_1, 0);
    FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_0, 0);
    if (high_fidelity) {
        FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_NONE, 0, FT_BH_ADDR_MOD_5, 0);
    } else {
        FT_BH_TTINSN_MVMUL(FT_BH_SETRWC_CLR_A, 0, FT_BH_ADDR_MOD_5, 0);
    }

    t = ft_tensix_mop_template_1(1u, inner_loops,
                                 FT_TENSIX_REPLAY_INSN(FT_TENSIX_MATH_REPLAY_BUF_OFFSET,
                                                       FT_TENSIX_MATH_MATMUL_HALF_REPLAY_LEN));
    if (high_fidelity) {
        t.end_op0 = FT_BH_ENC_SETRWC(FT_BH_SETRWC_CLR_A, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
        t.end_op1 = FT_BH_ENC_NOP;
    }
    ft_tensix_mop_template_program(&t);
}

static FT_BH_INLINE void ft_tensix_math_matmul_init_in0_16row(uint32_t fidelity_increment,
                                                              uint32_t inner_loops,
                                                              int high_fidelity)
{
    ft_tensix_math_matmul_configure_addrmod_in0_16row(fidelity_increment);
    ft_tensix_math_matmul_configure_mop_in0_16row(inner_loops, high_fidelity);
    FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F);
}

#endif /* FT_TRISC_MATH */

/* ===== Unpack thread ===================================================================== */
#ifdef FT_TRISC_UNPACK

/** Record the two-context unpack program and program the single-loop unpack MOP to replay
 *  the half of the current context (MOPExpander.md, unpack template registers 1..8). */
static FT_BH_INLINE void ft_tensix_unpack_ab_matmul_mop_config(void)
{
    volatile uint32_t *mop_cfg = FT_TENSIX_MOP_CFG;

    FT_TENSIX_REPLAY_RECORD(0u, FT_TENSIX_UNPACK_MATMUL_REPLAY_PROG_LEN);

    /* Context 0: unpack SrcA, then unpacker 0 base address += A tile size. */
    FT_BH_TTINSN_UNPACR(FT_BH_UNPACR_SRC_A, 0, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0, 0,
                        0, 0, 1);
    FT_BH_TTINSN_RDCFG(FT_TENSIX_GPR_UNPACK_TMP0, FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_WORD);
    FT_BH_TTINSN_ADDDMAREG(0, FT_TENSIX_GPR_UNPACK_TMP0, FT_TENSIX_GPR_UNPACK_TMP0,
                           FT_TENSIX_GPR_UNPACK_TILE_SIZE_A);
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG, FT_BH_STALLWAIT_COND_THCON);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_UNPACK_TMP0, FT_BH_WRCFG_32B,
                       FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_WORD);
    FT_BH_TTINSN_NOP;

    /* Context 1: the same on the context-1 base address. */
    FT_BH_TTINSN_UNPACR(FT_BH_UNPACR_SRC_A, 0, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0, 0,
                        0, 0, 1);
    FT_BH_TTINSN_RDCFG(FT_TENSIX_GPR_UNPACK_TMP0, FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_WORD);
    FT_BH_TTINSN_ADDDMAREG(0, FT_TENSIX_GPR_UNPACK_TMP0, FT_TENSIX_GPR_UNPACK_TMP0,
                           FT_TENSIX_GPR_UNPACK_TILE_SIZE_A);
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG, FT_BH_STALLWAIT_COND_THCON);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_UNPACK_TMP0, FT_BH_WRCFG_32B,
                       FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_WORD);
    FT_BH_TTINSN_NOP;

    ft_tensix_mop_sync();
    /* Unpack template: [1] flags, [2] B instruction, [3] A instruction, [4..6] A1..A3,
     * [7] skip-A instruction (zmask set: context 1), [8] skip-B instruction. */
    mop_cfg[1] = 0u;
    mop_cfg[2] = 0u;
    mop_cfg[3] = FT_TENSIX_REPLAY_INSN(0u, FT_TENSIX_UNPACK_MATMUL_REPLAY_RUN_LEN);
    mop_cfg[4] = 0u;
    mop_cfg[5] = 0u;
    mop_cfg[6] = 0u;
    mop_cfg[7] = FT_TENSIX_REPLAY_INSN(FT_TENSIX_UNPACK_MATMUL_REPLAY_RUN_LEN,
                                       FT_TENSIX_UNPACK_MATMUL_REPLAY_RUN_LEN);
    mop_cfg[8] = 0u;
}

/** Configure the matmul unpack: in-face transpose of SrcA, row lengths of both unpackers,
 *  K tile count (GPR 38). */
static FT_BH_INLINE void ft_tensix_unpack_ab_matmul_init(uint32_t transpose, uint32_t unpa_x_end,
                                                         uint32_t unpb_x_end, uint32_t kt_dim)
{
    FT_TENSIX_CFG_RMW_FIELD(UNPACKER0_HALOIZE_MODE, transpose);
    FT_BH_TTINSN_SETADCZW(FT_BH_SETADC_UNP_AB, 0, 0, 0, 0, 0xF);
    FT_BH_PUSH_SETADCXX(FT_BH_SETADC_UNP_A, unpa_x_end, 0x0);
    FT_BH_PUSH_SETADCXX(FT_BH_SETADC_UNP_B, unpb_x_end, 0x0);
    FT_BH_PUSH_SETDMAREG(0, FT_TENSIX_LOWER_HALFWORD(kt_dim), 0,
                         FT_TENSIX_LO_16(FT_TENSIX_GPR_UNPACK_KT_DIM));
    ft_tensix_unpack_ab_matmul_mop_config();
}

/** Set the SrcA step between K tiles (GPR 36, low half). */
static FT_BH_INLINE void ft_tensix_unpack_matmul_set_srca_tile_stride(uint32_t tile_stride)
{
    FT_BH_PUSH_SETDMAREG(0, FT_TENSIX_LOWER_HALFWORD(tile_stride), 0,
                         FT_TENSIX_LO_16(FT_TENSIX_GPR_UNPACK_TILE_SIZE_A));
}

/** Unpack in0 tile `tile_index_a` of CB `operand_a` to SrcB and `ct_dim` in1 tiles from
 *  `tile_index_b` of CB `operand_b` to SrcA, then flip the config context. */
static FT_BH_INLINE void ft_tensix_unpack_ab_matmul(uint32_t operand_a, uint32_t operand_b,
                                                    uint32_t tile_index_a, uint32_t tile_index_b,
                                                    uint32_t ct_dim)
{
    uint32_t base_address_a = ft_tensix_cb(operand_a)->fifo_read_pointer - 1u;
    uint32_t base_address_b = ft_tensix_cb(operand_b)->fifo_read_pointer - 1u;
    uint32_t tile_size_a = ft_tensix_cb(operand_a)->fifo_page_size;
    uint32_t tile_size_b = ft_tensix_cb(operand_b)->fifo_page_size;
    volatile uint32_t FT_BH_REG_PTR *cfg = ft_tensix_cfg_pointer();
    uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    uint32_t address_b = base_address_b + tile_size_b * tile_index_b;
    uint32_t context;

    /* Wait for a free config context (two exist). */
    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_UNPACK_SYNC) >= 2u) {
    }
    context = unp_cfg_context;
    if (context != 0) {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_WORD] = address_b;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT1_WORD] = address_a;
    } else {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_WORD] = address_b;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT0_WORD] = address_a;
    }
    ft_tensix_semaphore_post(FT_TENSIX_SEM_UNPACK_SYNC);

    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_UNPACK, FT_BH_STALLWAIT_COND_TRISC_CFG);
    FT_BH_TTINSN_UNPACR(FT_BH_UNPACR_SRC_B, 0, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0, 0,
                        0, 0, 1);
    FT_BH_PUSH_MOP(0, ct_dim - 1u, (context == 0) ? 0u : 0xffu);
    FT_BH_TTINSN_SEMGET(FT_TENSIX_T6_SEM(FT_TENSIX_SEM_UNPACK_SYNC));

    unp_cfg_context = 1u - context;
    if (context != 0) {
        FT_BH_TTINSN_SETC16(FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_WORD, 0x0000);
    } else {
        FT_BH_TTINSN_SETC16(FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_WORD, 0x0101);
    }
}

#if defined(FT_RTRIAD_MIXW) && FT_RTRIAD_MIXW
/**
 * ft_tensix_unpack_ab_matmul() with operand B (the weight, to SrcA) addressed at an explicit
 * `tile_size_b` instead of its CB's page size: the mixed-weight resident matmul lands Bfp4_b
 * tiles at a 576 B stride inside 1,088 B pages (resident_matmul_reader.c). A SIBLING compiled
 * only into that build, not a tile-size argument on the entry above: the one-body form (old
 * entry passing `fifo_page_size`) moved the UNPACK image of 8 shipped programs, rtriad_bfp8's
 * included (tools/check_mixw_program_gates.sh, 2026-11-13).
 *
 * @param tile_size_b  B's tile stride in the compute cores' CB unit (16-byte words)
 */
static FT_BH_INLINE void ft_tensix_unpack_ab_matmul_tile_b(uint32_t operand_a, uint32_t operand_b,
                                                           uint32_t tile_index_a,
                                                           uint32_t tile_index_b, uint32_t ct_dim,
                                                           uint32_t tile_size_b)
{
    uint32_t base_address_a = ft_tensix_cb(operand_a)->fifo_read_pointer - 1u;
    uint32_t base_address_b = ft_tensix_cb(operand_b)->fifo_read_pointer - 1u;
    uint32_t tile_size_a = ft_tensix_cb(operand_a)->fifo_page_size;
    volatile uint32_t FT_BH_REG_PTR *cfg = ft_tensix_cfg_pointer();
    uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    uint32_t address_b = base_address_b + tile_size_b * tile_index_b;
    uint32_t context;

    /* Wait for a free config context (two exist). */
    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_UNPACK_SYNC) >= 2u) {
    }
    context = unp_cfg_context;
    if (context != 0) {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_WORD] = address_b;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT1_WORD] = address_a;
    } else {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_WORD] = address_b;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT0_WORD] = address_a;
    }
    ft_tensix_semaphore_post(FT_TENSIX_SEM_UNPACK_SYNC);

    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_UNPACK, FT_BH_STALLWAIT_COND_TRISC_CFG);
    FT_BH_TTINSN_UNPACR(FT_BH_UNPACR_SRC_B, 0, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0, 0,
                        0, 0, 1);
    FT_BH_PUSH_MOP(0, ct_dim - 1u, (context == 0) ? 0u : 0xffu);
    FT_BH_TTINSN_SEMGET(FT_TENSIX_T6_SEM(FT_TENSIX_SEM_UNPACK_SYNC));

    unp_cfg_context = 1u - context;
    if (context != 0) {
        FT_BH_TTINSN_SETC16(FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_WORD, 0x0000);
    } else {
        FT_BH_TTINSN_SETC16(FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_WORD, 0x0101);
    }
}
#endif /* FT_RTRIAD_MIXW */

#endif /* FT_TRISC_UNPACK */

#endif /* FT_TENSIX_COMPUTE_MATMUL_H */
