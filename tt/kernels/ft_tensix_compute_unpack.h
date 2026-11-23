/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_compute_unpack.h
 * @brief UNPACK (TRISC0) side for two-operand ops: unpacker configuration, the A/B MOP, and
 *   the per-tile unpack of one A and one B tile from L1 into SrcA/SrcB with double-buffered
 *   config contexts.
 *
 * Config-context handshake: the RISC side posts the unpack-sync semaphore with a store; the
 * coprocessor releases it with SEMGET after the MOP. The two are not interchangeable.
 * Unpacker programming: WormholeB0/TensixTile/TensixCoprocessor/UNPACR_Regular.md.
 *
 * @status Clean-room rewrite, plan Phase 4 step 4.5a.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_COMPUTE_UNPACK_H
#define FT_TENSIX_COMPUTE_UNPACK_H

#include <stdint.h>

#include "ft_tensix_circular_buffer.h"
#include "ft_tensix_compute.h"

/* ===== Strides =========================================================================== */

/** Bytes per datum of an unpacker output format: Float32 4, Float16 2, else 1. */
static FT_BH_INLINE uint32_t ft_tensix_unpack_x_stride(uint32_t unpack_dst_format)
{
    uint32_t f = unpack_dst_format & 0x3u;
    return (f == 0) ? 4u : ((f == 1) ? 2u : 1u);
}

#define FT_TENSIX_UNPACK_Y_STRIDE(fmt) (FT_TENSIX_FACE_R_DIM * ft_tensix_unpack_x_stride(fmt))
#define FT_TENSIX_UNPACK_Z_STRIDE(fmt)                                                          \
    (FT_TENSIX_FACE_C_DIM * FT_TENSIX_FACE_R_DIM * ft_tensix_unpack_x_stride(fmt))

/* ===== Hardware configuration ============================================================ */

static FT_BH_INLINE void ft_tensix_unpack_hw_configure_ab(
    uint32_t unpa_src_format, uint32_t unpb_src_format, uint32_t unpa_dst_format,
    uint32_t unpb_dst_format, uint32_t unpa_face_r_dim, uint32_t unpb_face_r_dim,
    uint32_t unpa_num_faces, uint32_t unpb_num_faces, uint32_t unpa_tile_size,
    uint32_t unpb_tile_size)
{
    uint32_t a_src_m = FT_BH_DATA_FORMAT_CONFIG_BITS(unpa_src_format);
    uint32_t b_src_m = FT_BH_DATA_FORMAT_CONFIG_BITS(unpb_src_format);
    uint32_t a_dst_m = FT_BH_DATA_FORMAT_CONFIG_BITS(unpa_dst_format);
    uint32_t b_dst_m = FT_BH_DATA_FORMAT_CONFIG_BITS(unpb_dst_format);
    volatile uint32_t FT_BH_REG_PTR *cfg;
    uint32_t payload;
    uint32_t d;
    uint32_t fd;
    uint32_t v;

    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_UNPACK_SYNC) > 0) {
    }
    FT_BH_TTINSN_SETADCXY(FT_BH_SETADC_UNP_AB, 0, 0, 0, 0, 0xB);
    FT_BH_TTINSN_SETADCZW(FT_BH_SETADC_UNP_AB, 0, 0, 0, 0, 0xF);

    cfg = ft_tensix_cfg_pointer();
    cfg[FT_BH_CFG_UNPACKER0_OUTPUT_Z_STRIDE_WORD] =
        (0u << FT_BH_CFG_UNPACKER0_OUTPUT_W_STRIDE_SHIFT)
        | (FT_TENSIX_UNPACK_Z_STRIDE(a_dst_m) << FT_BH_CFG_UNPACKER0_OUTPUT_Z_STRIDE_SHIFT);
    cfg[FT_BH_CFG_UNPACKER1_OUTPUT_Z_STRIDE_WORD] =
        (0u << FT_BH_CFG_UNPACKER1_OUTPUT_W_STRIDE_SHIFT)
        | (FT_TENSIX_UNPACK_Z_STRIDE(b_dst_m) << FT_BH_CFG_UNPACKER1_OUTPUT_Z_STRIDE_SHIFT);
    /* Word and mask of the output Y stride, shift of the input Y stride (Foundry convention,
     * both 16). */
    FT_TENSIX_CFG_RMW(FT_BH_CFG_UNPACKER0_OUTPUT_Y_STRIDE_WORD,
                      FT_BH_CFG_UNPACKER0_INPUT_Y_STRIDE_SHIFT,
                      FT_BH_CFG_UNPACKER0_OUTPUT_Y_STRIDE_MASK, FT_TENSIX_UNPACK_Y_STRIDE(a_dst_m));

    /* The ALU format word is shared with MATH and PACK (other byte lanes): hold the mutex. */
    FT_BH_TTINSN_ATGETM(FT_TENSIX_MUTEX_REG_RMW);
    payload = 0;
    if (unpa_src_format == FT_BH_DATA_FORMAT_UINT8) {
        payload |= FT_BH_CFG_MATH_SRCA_UNSIGNED_MASK;
    }
    if (unpb_src_format == FT_BH_DATA_FORMAT_UINT8) {
        payload |= FT_BH_CFG_MATH_SRCB_UNSIGNED_MASK;
    }
    FT_TENSIX_CFG_RMW(FT_BH_CFG_MATH_SRCA_FORMAT_WORD, 0,
                      FT_BH_CFG_MATH_SRCA_UNSIGNED_MASK | FT_BH_CFG_MATH_SRCB_UNSIGNED_MASK
                          | FT_BH_CFG_FPU_STOCHASTIC_ROUNDING_MASK
                          | FT_BH_CFG_GASKET_STOCHASTIC_ROUNDING_MASK
                          | FT_BH_CFG_PACKER_STOCHASTIC_ROUNDING_MASK,
                      payload);
    FT_TENSIX_CFG_RMW_FIELD(UNPACKER0_LF8_4BIT_EXPONENT,
                            ((unpa_src_format & 0x1F) == FT_BH_DATA_FORMAT_FP8_E4M3) ? 1u : 0u);
    FT_TENSIX_CFG_RMW_FIELD(UNPACKER1_LF8_4BIT_EXPONENT,
                            ((unpb_src_format & 0x1F) == FT_BH_DATA_FORMAT_FP8_E4M3) ? 1u : 0u);
    FT_BH_TTINSN_ATRELM(FT_TENSIX_MUTEX_REG_RMW);

    /* Tile descriptors; A's x-dim stays 0, the per-context x-dim below overrides it. */
    cfg[FT_BH_CFG_UNPACKER0_TILE_DESCRIPTOR_WORD + 0] = a_src_m | (1u << 4);
    cfg[FT_BH_CFG_UNPACKER0_TILE_DESCRIPTOR_WORD + 1] = 1u | (unpa_num_faces << 16);
    cfg[FT_BH_CFG_UNPACKER1_TILE_DESCRIPTOR_WORD + 0] =
        b_src_m | (1u << 4) | ((unpb_face_r_dim * FT_TENSIX_FACE_C_DIM) << 16);
    cfg[FT_BH_CFG_UNPACKER1_TILE_DESCRIPTOR_WORD + 1] = 1u | (unpb_num_faces << 16);

    /* Output format with throttle mode 2. */
    cfg[FT_BH_CFG_UNPACKER0_OUTPUT_FORMAT_WORD + 0] = a_dst_m | (2u << 4);
    cfg[FT_BH_CFG_UNPACKER0_OUTPUT_FORMAT_WORD + 1] = 0xFu | (0xFu << 16);
    cfg[FT_BH_CFG_UNPACKER1_OUTPUT_FORMAT_WORD + 0] = b_dst_m | (2u << 4);
    cfg[FT_BH_CFG_UNPACKER1_OUTPUT_FORMAT_WORD + 1] = 0xFu | (0xFu << 16);

    d = (unpa_face_r_dim == 0) ? (22u * 16u) : (4u * 16u);
    cfg[FT_BH_CFG_UNPACKER0_DEST_CONTEXT0_ADDRESS_WORD] = d | (d << 16);
    fd = unpa_face_r_dim * FT_TENSIX_FACE_C_DIM;
    cfg[FT_BH_CFG_UNPACKER0_CONTEXT0_TILE_X_DIM_WORD] = fd | (fd << 16);

    v = FT_TENSIX_FACE_R_DIM * FT_TENSIX_FACE_C_DIM;
    FT_TENSIX_REGFILE[FT_TENSIX_GPR_UNPACK_FACE_DIM_16x16] = (v / 1) | ((v / 1) << 16);
    FT_TENSIX_REGFILE[FT_TENSIX_GPR_UNPACK_FACE_DIM_8x16] = (v / 2) | ((v / 2) << 16);
    FT_TENSIX_REGFILE[FT_TENSIX_GPR_UNPACK_FACE_DIM_4x16] = (v / 4) | ((v / 4) << 16);
    FT_TENSIX_REGFILE[FT_TENSIX_GPR_UNPACK_FACE_DIM_2x16] = (v / 8) | ((v / 8) << 16);
    FT_TENSIX_REGFILE[FT_TENSIX_GPR_UNPACK_FACE_DIM_1x16] = (v / 16) | ((v / 16) << 16);
    ft_tensix_sync_regfile_write(FT_TENSIX_GPR_UNPACK_FACE_DIM_1x16);

    FT_BH_TTINSN_SETC16(FT_BH_CFG_SRCA_WRITE_BASE_WORD, 0x4);
    cfg[FT_BH_CFG_UNPACKER0_ADD_DEST_ADDRESS_COUNTER_WORD] =
        0x1u << FT_BH_CFG_UNPACKER0_ADD_DEST_ADDRESS_COUNTER_SHIFT;
    unp_cfg_context = 0;
    FT_BH_TTINSN_SETC16(FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_WORD, 0x0000);
    FT_BH_PUSH_SETDMAREG(0, FT_TENSIX_LOWER_HALFWORD(unpa_tile_size), 0,
                         FT_TENSIX_LO_16(FT_TENSIX_GPR_UNPACK_TILE_SIZE_A));
    FT_BH_PUSH_SETDMAREG(0, FT_TENSIX_LOWER_HALFWORD(unpb_tile_size), 0,
                         FT_TENSIX_LO_16(FT_TENSIX_GPR_UNPACK_TILE_SIZE_B));
}

/**
 * Switch unpacker 0 (SrcA) to another block-float input format BETWEEN tiles, without
 * re-running ft_tensix_unpack_hw_configure_ab() (re-running that mid-kernel garbled the M5
 * row-0 builds). Both config contexts drained, unpacker 0 idle, then the tile descriptor's and
 * the output format's format nibbles rewritten in the Tensix stream, and the tile-size GPR.
 * Only for formats of one exponent family with the same tile geometry (BFP4_B <-> BFP2_B).
 */
static FT_BH_INLINE void ft_tensix_unpack_reconfigure_srca_format(uint32_t src_format, uint32_t dst_format,
                                                                  uint32_t tile_size)
{
    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_UNPACK_SYNC) > 0) {
    }
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG, FT_BH_STALLWAIT_COND_UNPACK0);
    FT_TENSIX_CFG_RMW(FT_BH_CFG_UNPACKER0_TILE_DESCRIPTOR_WORD, 0, FT_BH_DATA_FORMAT_CONFIG_MASK,
                      FT_BH_DATA_FORMAT_CONFIG_BITS(src_format));
    FT_TENSIX_CFG_RMW(FT_BH_CFG_UNPACKER0_OUTPUT_FORMAT_WORD, 0, FT_BH_DATA_FORMAT_CONFIG_MASK,
                      FT_BH_DATA_FORMAT_CONFIG_BITS(dst_format));
    FT_BH_PUSH_SETDMAREG(0, FT_TENSIX_LOWER_HALFWORD(tile_size), 0,
                         FT_TENSIX_LO_16(FT_TENSIX_GPR_UNPACK_TILE_SIZE_A));
}

/* ===== MOP + init ======================================================================== */

/* UNPACR into SrcA (which = 0) or SrcB (which = 1): address modifier 1, override thread id,
 * set data valid, last. */
#define FT_TENSIX_UNPACK_AB_UNPACR(which)                                                       \
    FT_BH_ENC_UNPACR((which), 1, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0, 0, 0, 0, 1)

static FT_BH_INLINE void ft_tensix_unpack_ab_mop_config(uint32_t num_faces_r_dim,
                                                        uint32_t num_faces_c_dim)
{
    ft_tensix_mop_template t =
        ft_tensix_mop_template_2(num_faces_r_dim, num_faces_c_dim,
                                 FT_TENSIX_UNPACK_AB_UNPACR(FT_BH_UNPACR_SRC_A),
                                 FT_TENSIX_UNPACK_AB_UNPACR(FT_BH_UNPACR_SRC_B));
    ft_tensix_mop_template_program(&t);
}

static FT_BH_INLINE void ft_tensix_unpack_ab_init(uint32_t face_r_dim, uint32_t num_faces_r_dim,
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
    ft_tensix_unpack_ab_mop_config(num_faces_r_dim, num_faces_c_dim);
}

/* ===== Per-tile unpack =================================================================== */

/** Unpack tile `tile_index_a` of CB `operand_a` into SrcA and `tile_index_b` of `operand_b`
 *  into SrcB, through the free config context. */
static FT_BH_INLINE void ft_tensix_unpack_ab(uint32_t operand_a, uint32_t operand_b,
                                             uint32_t tile_index_a, uint32_t tile_index_b)
{
    /* The unpacker base address carries a -1 bias (Foundry CB convention). */
    uint32_t address_a = (ft_tensix_cb(operand_a)->fifo_read_pointer - 1u)
                         + ft_tensix_cb(operand_a)->fifo_page_size * tile_index_a;
    uint32_t address_b = (ft_tensix_cb(operand_b)->fifo_read_pointer - 1u)
                         + ft_tensix_cb(operand_b)->fifo_page_size * tile_index_b;
    volatile uint32_t FT_BH_REG_PTR *cfg;

    FT_BH_TTINSN_SETADCZW(FT_BH_SETADC_UNP_AB, 0, 0, 0, 0, 0xF);
    cfg = ft_tensix_cfg_pointer();

    /* Wait for a free config context (two contexts). */
    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_UNPACK_SYNC) >= 2u) {
    }
    if (unp_cfg_context == 0) {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_WORD] = address_a;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT0_WORD] = address_b;
    } else {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_WORD] = address_a;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT1_WORD] = address_b;
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

#endif /* FT_TENSIX_COMPUTE_UNPACK_H */
