/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_compute_pack.h
 * @brief PACK (TRISC2) side of the compute triad: packer configuration, pack ADDR_MOD slots
 *   and MOP, the dest-half handshake from the packer side (including the ZEROACC that zeroes
 *   the half just packed) and the per-tile pack into the next in-order slot of an output CB.
 *
 * ft_tensix_pack_hw_configure() also rewrites the packer edge mask and the tile row-set
 * mapping to their baseline: a reduce leaves them masked, and the state outlives the program
 * (tt/tools/check_tensix_reduce_exit_state.py checks those two stores by source text).
 * Packer programming: WormholeB0/TensixTile/TensixCoprocessor/Packers/ and PACR.md (BH page).
 *
 * @status Clean-room rewrite, plan Phase 4 step 4.5a.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_COMPUTE_PACK_H
#define FT_TENSIX_COMPUTE_PACK_H

#include <stdint.h>

#include "ft_tensix_circular_buffer.h"
#include "ft_tensix_compute.h"

/* Blackhole has one packer. */
#define FT_TENSIX_NUM_PACKERS 1u

/* ===== Internal helpers ================================================================== */

/** Bytes per datum of a packer input format: Float32 4, Float16 2, else 1. */
static FT_BH_INLINE uint32_t ft_tensix_pack_datum_size_in_bytes(uint32_t format)
{
    uint32_t f = format & 0x3u;
    return (f == 0) ? 4u : ((f == 1) ? 2u : 1u);
}

/** Packer input Y/Z/W strides, staged in two GPRs and written with WRCFG. The X stride slot
 *  receives the Y value's low half (the 32-bit write covers both). */
static FT_BH_INLINE void ft_tensix_pack_set_strides(uint32_t pack_src_format)
{
    uint32_t x = ft_tensix_pack_datum_size_in_bytes(pack_src_format);
    uint32_t y = FT_TENSIX_FACE_C_DIM * x;
    uint32_t w = FT_TENSIX_TILE_NUM_FACES * FT_TENSIX_FACE_C_DIM * FT_TENSIX_FACE_R_DIM * x;
    uint32_t z = FT_TENSIX_FACE_R_DIM * y;

    FT_BH_PUSH_SETDMAREG(0,
                         FT_TENSIX_LOWER_HALFWORD(y << FT_BH_CFG_PACKER0_INPUT_Y_STRIDE_SHIFT),
                         0, FT_TENSIX_LO_16(FT_TENSIX_GPR_PACK_TMP0));
    FT_BH_PUSH_SETDMAREG(0,
                         FT_TENSIX_UPPER_HALFWORD(y << FT_BH_CFG_PACKER0_INPUT_Y_STRIDE_SHIFT),
                         0, FT_TENSIX_HI_16(FT_TENSIX_GPR_PACK_TMP0));
    FT_BH_PUSH_SETDMAREG(0,
                         FT_TENSIX_LOWER_HALFWORD(z << FT_BH_CFG_PACKER0_INPUT_Z_STRIDE_SHIFT),
                         0, FT_TENSIX_LO_16(FT_TENSIX_GPR_PACK_TMP1));
    FT_BH_PUSH_SETDMAREG(0,
                         FT_TENSIX_UPPER_HALFWORD(w << FT_BH_CFG_PACKER0_INPUT_W_STRIDE_SHIFT),
                         0, FT_TENSIX_HI_16(FT_TENSIX_GPR_PACK_TMP1));
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG, FT_BH_STALLWAIT_COND_THCON);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_PACK_TMP0, FT_BH_WRCFG_32B,
                       FT_BH_CFG_PACKER0_INPUT_X_STRIDE_WORD);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_PACK_TMP1, FT_BH_WRCFG_32B,
                       FT_BH_CFG_PACKER0_INPUT_Z_STRIDE_WORD);
    FT_BH_TTINSN_NOP;
    FT_BH_TTINSN_NOP;
}

static FT_BH_INLINE void ft_tensix_pack_reconfigure_l1_acc(uint32_t enable)
{
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG, FT_BH_STALLWAIT_COND_PACK);
    FT_TENSIX_CFG_RMW_FIELD(PACKER0_ZERO_FLAGS_DISABLE, enable);
    FT_TENSIX_CFG_RMW_FIELD(PACKER0_L1_ACCUMULATE, enable);
}

static FT_BH_INLINE void ft_tensix_pack_set_config(uint32_t pack_src_format,
                                                   uint32_t pack_dst_format, uint32_t num_faces,
                                                   int partial_face, uint32_t is_fp32_dest_acc_en)
{
    volatile uint32_t FT_BH_REG_PTR *cfg = ft_tensix_cfg_pointer();
    uint32_t src_m = FT_BH_DATA_FORMAT_CONFIG_BITS(pack_src_format);
    uint32_t dst_m = FT_BH_DATA_FORMAT_CONFIG_BITS(pack_dst_format);
    uint32_t hw_src = ((pack_dst_format & 0x1F) == FT_BH_DATA_FORMAT_FP8_E4M3)
                          ? FT_BH_DATA_FORMAT_FLOAT16
                          : src_m;
    uint32_t dest_read_control = 0;
    uint32_t exp_section_size =
        (dst_m == FT_BH_DATA_FORMAT_LF8 || dst_m == FT_BH_DATA_FORMAT_INT8)
            ? 0u
            : (partial_face ? 1u : num_faces);
    uint32_t word0 = exp_section_size << 16;
    uint32_t word2 = 1u | (dst_m << 4) | (hw_src << 8);
    int is32;
    int is8;

    FT_TENSIX_CFG_RMW(FT_BH_CFG_PACKER0_EXPONENT_THRESHOLD_WORD, 0,
                      FT_BH_CFG_PACKER0_EXPONENT_THRESHOLD_ENABLE_MASK
                          | FT_BH_CFG_PACKER0_EXPONENT_THRESHOLD_MASK,
                      0u);
    cfg[FT_BH_CFG_PACKER0_ROW_START_SECTION_SIZE_WORD + 0] = word0;
    cfg[FT_BH_CFG_PACKER0_ROW_START_SECTION_SIZE_WORD + 2] = word2;
    ft_tensix_pack_reconfigure_l1_acc(0);

    /* Dest read control: bit 0 read 32-bit data, bit 1 read int8, bit 2 read unsigned,
     * bit 3 round to a 10-bit mantissa. */
    is32 = (pack_src_format == FT_BH_DATA_FORMAT_INT32)
           || (pack_src_format == FT_BH_DATA_FORMAT_UINT32)
           || (pack_src_format == FT_BH_DATA_FORMAT_FLOAT32);
    is8 = (pack_src_format == FT_BH_DATA_FORMAT_INT8)
          || (pack_src_format == FT_BH_DATA_FORMAT_UINT8);
    if (is32 || is_fp32_dest_acc_en) {
        dest_read_control |= 1u << 0;
    }
    if (!(is_fp32_dest_acc_en || is32) && is8) {
        dest_read_control |= 1u << 1;
    }
    if (pack_dst_format == FT_BH_DATA_FORMAT_UINT8) {
        dest_read_control |= 1u << 2;
    }
    if ((is_fp32_dest_acc_en && pack_src_format == FT_BH_DATA_FORMAT_FLOAT16)
        || (pack_dst_format & 0x1F) == FT_BH_DATA_FORMAT_FP8_E4M3) {
        dest_read_control |= 1u << 3;
    }
    cfg[FT_BH_CFG_PACKER_DEST_READ_32BIT_WORD] = dest_read_control;

    FT_TENSIX_REGFILE[FT_TENSIX_GPR_PACK_EXP0_SEC_SIZE_BFP] =
        (partial_face ? 1u : num_faces) << FT_BH_CFG_PACKER0_EXPONENT_SECTION_SIZE_SHIFT;
    ft_tensix_sync_regfile_write(FT_TENSIX_GPR_PACK_EXP0_SEC_SIZE_BFP);
}

static FT_BH_INLINE void ft_tensix_pack_configure_addrmod(void)
{
    FT_TENSIX_ADDR_MOD_PACK_SET(
        0, FT_TENSIX_ADDR_MOD_PACK_VAL(FT_TENSIX_ADDR_MOD_PACK_VALS(4, 0, 0),
                                       FT_TENSIX_ADDR_MOD_PACK_VALS(4, 0, 0),
                                       FT_TENSIX_ADDR_MOD_PACK_REDUCED(0, 0),
                                       FT_TENSIX_ADDR_MOD_PACK_REDUCED(0, 0)));
    FT_TENSIX_ADDR_MOD_PACK_SET(
        1, FT_TENSIX_ADDR_MOD_PACK_VAL(FT_TENSIX_ADDR_MOD_PACK_VALS(0, 1, 0),
                                       FT_TENSIX_ADDR_MOD_PACK_VALS(0, 1, 0),
                                       FT_TENSIX_ADDR_MOD_PACK_REDUCED(0, 1),
                                       FT_TENSIX_ADDR_MOD_PACK_REDUCED(0, 0)));
    FT_TENSIX_ADDR_MOD_PACK_SET(
        2, FT_TENSIX_ADDR_MOD_PACK_VAL(FT_TENSIX_ADDR_MOD_PACK_VALS(0, 1, 0),
                                       FT_TENSIX_ADDR_MOD_PACK_VALS(4, 0, 0),
                                       FT_TENSIX_ADDR_MOD_PACK_REDUCED(1, 0),
                                       FT_TENSIX_ADDR_MOD_PACK_REDUCED(0, 0)));
}

/* PACR with address modifier `am` and Last bit `last`, reading interfaces `intf`. */
#define FT_TENSIX_PACK_PACR(am, intf, last)                                                     \
    FT_BH_ENC_PACR(FT_BH_PACR_CFG_CTXT_0, FT_BH_PACR_NO_ROW_PAD_ZERO,                          \
                   FT_BH_PACR_DST_ACCESS_NORMAL_MODE, (am), FT_BH_PACR_ADDR_CNT_CTXT_0,        \
                   FT_BH_PACR_ZERO_OUTPUT_DISABLED, (intf), 0, 0, FT_BH_PACR_NO_CTXT_CTRL, 0,  \
                   (last))

static FT_BH_INLINE void ft_tensix_pack_mop_config(uint32_t face_r_dim, uint32_t num_faces,
                                                   uint32_t num_tiles)
{
    uint32_t intf = (face_r_dim == 1)   ? FT_BH_PACR_SINGLE_INTF_ACTIVE
                    : (face_r_dim == 2) ? FT_BH_PACR_TWO_INTFS_ACTIVE
                                        : FT_BH_PACR_ALL_INTF_ACTIVE;
    uint32_t inner = (face_r_dim < 4) ? 1u : (face_r_dim >> 2);
    uint32_t outer = num_faces * num_tiles;
    ft_tensix_mop_template t =
        ft_tensix_mop_template_1(outer, inner, FT_TENSIX_PACK_PACR(FT_BH_ADDR_MOD_0, intf, 0));
    t.loop1_last_instr = FT_TENSIX_PACK_PACR(FT_BH_ADDR_MOD_2, intf, 0);
    t.loop0_last_instr = FT_TENSIX_PACK_PACR(FT_BH_ADDR_MOD_1, intf, 1);
    ft_tensix_mop_template_program(&t);
}

/** Point the packer at the dest half named by ft_tensix_dest_offset_id. */
static FT_BH_INLINE void ft_tensix_pack_select_dest_registers(void)
{
    FT_BH_PUSH_WRCFG(ft_tensix_dest_offset_id ? FT_TENSIX_GPR_PACK_DEST_OFFSET_HI
                                              : FT_TENSIX_GPR_PACK_DEST_OFFSET_LO,
                     FT_BH_WRCFG_128B, FT_BH_CFG_PACKER0_DEST_BASE_OFFSET_WORD);
    FT_BH_TTINSN_DMANOP;
    FT_BH_TTINSN_DMANOP;
}

static FT_BH_INLINE void ft_tensix_pack_init_dest_offset_registers(void)
{
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_TDMA | FT_BH_STALLWAIT_BLOCK_THCON,
                           FT_BH_STALLWAIT_COND_PACK);
    FT_BH_TTINSN_SETDMAREG(0, 0x00, 0, FT_TENSIX_LO_16(FT_TENSIX_GPR_PACK_DEST_OFFSET_LO + 0));
    FT_BH_TTINSN_SETDMAREG(0, FT_TENSIX_DEST_REGISTER_HALF_SIZE + 0x00, 0,
                           FT_TENSIX_LO_16(FT_TENSIX_GPR_PACK_DEST_OFFSET_HI + 0));
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG, FT_BH_STALLWAIT_COND_THCON);
    ft_tensix_pack_select_dest_registers();
}

/** Set the packer's L1 destination. Bit 31 is set only for the WRCFG write; the GPR's high
 *  half is restored afterwards. */
static FT_BH_INLINE void ft_tensix_pack_program_destination(uint32_t address)
{
    uint32_t hi_addr = (1u << 31) | address;
    FT_BH_PUSH_SETDMAREG(0, address & 0xFFFF, 0, FT_TENSIX_LO_16(FT_TENSIX_GPR_PACK_OUTPUT_ADDR));
    FT_BH_PUSH_SETDMAREG(0, hi_addr >> 16, 0, FT_TENSIX_HI_16(FT_TENSIX_GPR_PACK_OUTPUT_ADDR));
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG, FT_BH_STALLWAIT_COND_THCON);
    FT_BH_TTINSN_WRCFG(FT_TENSIX_GPR_PACK_OUTPUT_ADDR, FT_BH_WRCFG_32B,
                       FT_BH_CFG_PACKER0_L1_DEST_ADDRESS_WORD);
    FT_BH_PUSH_SETDMAREG(0, address >> 16, 0, FT_TENSIX_HI_16(FT_TENSIX_GPR_PACK_OUTPUT_ADDR));
    FT_BH_TTINSN_DMANOP;
}

/* ===== External API ====================================================================== */

static FT_BH_INLINE void ft_tensix_pack_hw_configure(uint32_t pack_src_format,
                                                     uint32_t pack_dst_format, uint32_t tile_size,
                                                     uint32_t num_faces, int partial_face,
                                                     uint32_t relu_config,
                                                     uint32_t is_fp32_dest_acc_en)
{
    volatile uint32_t FT_BH_REG_PTR *cfg = ft_tensix_cfg_pointer();
    uint32_t src_m = FT_BH_DATA_FORMAT_CONFIG_BITS(pack_src_format);
    int is_fp8 = ((pack_dst_format & 0x1F) == FT_BH_DATA_FORMAT_FP8_E4M3);
    uint32_t index;

    ft_tensix_pack_set_strides(pack_src_format);

    /* These config words are shared with other threads: hold the mutex. */
    FT_BH_TTINSN_ATGETM(FT_TENSIX_MUTEX_REG_RMW);
    FT_TENSIX_CFG_RMW_FIELD(PACKER0_LF8_4BIT_EXPONENT, is_fp8 ? 1u : 0u);
    FT_TENSIX_CFG_RMW_FIELD(DEST_ACCUMULATOR_FORMAT, is_fp8 ? FT_BH_DATA_FORMAT_FLOAT16 : src_m);
    FT_TENSIX_CFG_RMW(FT_BH_CFG_DEST_FORMAT_OVERRIDE_VALUE_WORD, 0,
                      FT_BH_CFG_DEST_FORMAT_OVERRIDE_VALUE_MASK
                          | FT_BH_CFG_DEST_FORMAT_OVERRIDE_ENABLE_MASK,
                      0u);
    FT_TENSIX_CFG_RMW(FT_BH_CFG_PACKER_RELU_MODE_WORD, 0,
                      FT_BH_CFG_PACKER_RELU_MODE_MASK | FT_BH_CFG_PACKER_RELU_THRESHOLD_MASK,
                      (relu_config & 0xffff)
                          | (((relu_config >> 16) & 0xffff)
                             << FT_BH_CFG_PACKER_RELU_THRESHOLD_SHIFT));
    FT_BH_TTINSN_ATRELM(FT_TENSIX_MUTEX_REG_RMW);

    ft_tensix_pack_set_config(pack_src_format, pack_dst_format, num_faces, partial_face,
                              is_fp32_dest_acc_en);

    for (index = 0; index < FT_TENSIX_NUM_PACKERS; index++) {
        cfg[FT_BH_CFG_PACKER0_DATUMS_PER_XY_PLANE_WORD + index] =
            1u << FT_BH_CFG_PACKER0_READS_PER_XY_PLANE_SHIFT;
    }

    /* Packer baseline: edge mask passes every datum, every row maps to row set 0. */
    cfg[FT_BH_CFG_PACKER_EDGE_MASK0_WORD] = 0xffff;
    cfg[FT_BH_CFG_PACKER_TILE_ROW_MAP0_WORD] = 0x0;

    FT_TENSIX_REGFILE[FT_TENSIX_GPR_PACK_TILE_HEADER + 0] = tile_size;
    FT_TENSIX_REGFILE[FT_TENSIX_GPR_PACK_TILE_HEADER + 1] = 0;
    FT_TENSIX_REGFILE[FT_TENSIX_GPR_PACK_TILE_HEADER + 2] = 0;
    FT_TENSIX_REGFILE[FT_TENSIX_GPR_PACK_TILE_HEADER + 3] = 0;
    ft_tensix_sync_regfile_write(FT_TENSIX_GPR_PACK_TILE_HEADER + 3);
}

static FT_BH_INLINE void ft_tensix_pack_init(uint32_t pack_src_format, uint32_t face_r_dim,
                                             uint32_t num_faces, uint32_t num_tiles)
{
    ft_tensix_pack_configure_addrmod();
    ft_tensix_pack_mop_config(face_r_dim, num_faces, num_tiles);
    ft_tensix_pack_set_strides(pack_src_format);
    FT_BH_TTINSN_SETADCXX(FT_BH_SETADC_PAC, FT_TENSIX_FACE_C_DIM - 1, 0);
}

static FT_BH_INLINE void ft_tensix_pack_dest_init(void)
{
    ft_tensix_sync();
    ft_tensix_dest_offset_id = 0;
    ft_tensix_pack_init_dest_offset_registers();
    FT_BH_TTINSN_SETADCXY(FT_BH_SETADC_PAC, 0, 0, 0, 0, 0xB);
    FT_BH_TTINSN_SETADCZW(FT_BH_SETADC_PAC, 0, 0, 0, 0, 0xF);
    pack_sync_tile_dst_ptr = 0;
}

/** Block until the math side has handed over a dest half. */
static FT_BH_INLINE void ft_tensix_packer_wait_for_math_done(void)
{
    FT_BH_TTINSN_SEMWAIT(FT_BH_STALLWAIT_BLOCK_TDMA, FT_TENSIX_T6_SEM(FT_TENSIX_SEM_MATH_PACK),
                         FT_BH_SEMWAIT_COND_ZERO);
}

/** Zero the dest half just packed, release it to math and switch halves. */
static FT_BH_INLINE void ft_tensix_pack_dest_section_done(uint32_t is_fp32_dest_acc_en)
{
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_MATH, FT_BH_STALLWAIT_COND_PACK);
    FT_BH_PUSH_ZEROACC(FT_BH_ZEROACC_CLR_HALF, is_fp32_dest_acc_en, 0, FT_BH_ADDR_MOD_1,
                       ft_tensix_dest_offset_id % 2u);
    FT_BH_TTINSN_SEMGET(FT_BH_SEMAPHORE_1);
    ft_tensix_dest_offset_id = 1u - ft_tensix_dest_offset_id;
    ft_tensix_pack_select_dest_registers();
}

/** Pack dest tile `tile_index` into the next in-order slot of CB `output_cb`. */
static FT_BH_INLINE void ft_tensix_pack_tile(uint32_t tile_index, uint32_t output_cb)
{
    ft_tensix_cb_interface *cb = ft_tensix_cb(output_cb);
    uint32_t addr = cb->fifo_write_pointer + cb->fifo_write_tile_pointer - 1u;
    cb->fifo_write_tile_pointer += cb->fifo_page_size;
    FT_BH_PUSH_SETADC(FT_BH_SETADC_PAC, FT_BH_SETADC_CH_0, FT_BH_SETADC_SET_W, tile_index);
    ft_tensix_pack_program_destination(addr);
    ft_tensix_mop_template_run();
    FT_BH_TTINSN_SETADCZW(FT_BH_SETADC_PAC, 0, 0, 0, 0, 0x5);
}

#endif /* FT_TENSIX_COMPUTE_PACK_H */
