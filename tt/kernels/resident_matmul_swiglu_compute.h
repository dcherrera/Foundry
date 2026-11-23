/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_matmul_swiglu_compute.h
 * @brief The SILU_MUL half of a fused SwiGLU descriptor, for resident_matmul_compute.c built
 *   with -DFT_RTRIAD_SWIGLU=1 (program `rtriad_bfp8_swiglu`). Kernel-only; the wire contract is
 *   resident_matmul_swiglu_protocol.h.
 *
 * THE SEQUENCE IS ELTWISE'S, CALL FOR CALL. ftsw_silu_gated_multiply() is
 *   resident_eltwise_compute.c's apply_silu_gated_multiply() with three substitutions and no
 *   other change: the gate and up tiles come from the PANEL CB at an index (eltwise reads the
 *   front of cb_in0 / cb_in1), silu(gate) is staged in the STAGE CB (eltwise's cb_scratch), and
 *   a DST commit goes through resident_matmul_dst_sync.h (this program is SyncFull; the values
 *   do not depend on which DST section holds them). Same verbs, same order, same formats
 *   (every CB bf16), same fidelity (FT_CB_MATH_FIDELITY, HiFi4 in both tables), same fp32 DST,
 *   same SFPU silu (ft_tensix_math_silu_tile / _row0) -- so ffn_swiglu's bytes are eltwise's.
 *   tests/test_swiglu_fusion.c compares the two sequences from the two sources.
 *
 * THE ONE THING ELTWISE NEVER DOES: SrcA CHANGES FORMAT. The matmul feeds the Bfp8_b weight
 *   to SrcA; datacopy and the multiply feed it bf16. ftsw_unpack_srca_format() is
 *   resident_attention_compute.c's attention_unpack_reconfig_srca() (the card-proven
 *   block-float KV switch: source and destination format, the output Z/Y strides that follow
 *   the datum width, the tile-size GPR), plus the drain of both unpack contexts that
 *   ft_tensix_unpack_reconfigure_srca_format() takes. Switched to bf16 once per column block
 *   and back once; the matmul MOPs are re-initialised after (ftsw_matmul_restore lives in the
 *   compute .c beside matmul_select_in0_shape, which it mirrors).
 *
 * @status Added 2026-11-17 (Nex-n2 qwen38_tt_chain_fusion_plan.md 3.3, S3b).
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_MATMUL_SWIGLU_COMPUTE_H
#define RESIDENT_MATMUL_SWIGLU_COMPUTE_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_cb_formats.h"
#include "ft_tensix_compute.h"
#include "ft_tensix_compute_cb.h"
#ifdef FT_TRISC_MATH
#include "ft_tensix_compute_math.h"
#include "ft_tensix_compute_sfpu.h"
#include "ft_tensix_math_eltwise_sfpu.h"
#endif
#ifdef FT_TRISC_UNPACK
#include "ft_tensix_compute_sfpu.h"
#include "ft_tensix_compute_unpack.h"
#endif
#ifdef FT_TRISC_PACK
#include "ft_tensix_compute_pack.h"
#endif

#include "resident_matmul_dst_sync.h"
#include "resident_matmul_swiglu_protocol.h"
#include "resident_queue_protocol.h"

#define FTSW_DST_REGISTER 0u

/* ===== SrcA format ======================================================================= */

#ifdef FT_TRISC_UNPACK
/**
 * Point unpacker A (SrcA) at another format between tiles: wait until neither unpack context
 * holds work, stall the config writes on unpacker 0, then the A half of
 * ft_tensix_unpack_hw_configure_ab that depends on the format (output Z and Y strides follow
 * the destination datum width: 1 B Bfp8_b, 2 B Float16_b), both format nibbles and the
 * tile-size GPR. The shared ALU word is left as configured (neither format is UInt8/FP8).
 */
static FT_BH_INLINE void ftsw_unpack_srca_format(uint32_t src_format, uint32_t dst_format,
                                                 uint32_t tile_size)
{
    const uint32_t dst_m = FT_BH_DATA_FORMAT_CONFIG_BITS(dst_format);

    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_UNPACK_SYNC) > 0) {
    }
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG, FT_BH_STALLWAIT_COND_UNPACK0);
    FT_TENSIX_CFG_RMW(FT_BH_CFG_UNPACKER0_OUTPUT_Z_STRIDE_WORD,
                      FT_BH_CFG_UNPACKER0_OUTPUT_Z_STRIDE_SHIFT,
                      FT_BH_CFG_UNPACKER0_OUTPUT_Z_STRIDE_MASK, FT_TENSIX_UNPACK_Z_STRIDE(dst_m));
    FT_TENSIX_CFG_RMW(FT_BH_CFG_UNPACKER0_OUTPUT_Y_STRIDE_WORD,
                      FT_BH_CFG_UNPACKER0_OUTPUT_Y_STRIDE_SHIFT,
                      FT_BH_CFG_UNPACKER0_OUTPUT_Y_STRIDE_MASK, FT_TENSIX_UNPACK_Y_STRIDE(dst_m));
    FT_TENSIX_CFG_RMW(FT_BH_CFG_UNPACKER0_TILE_DESCRIPTOR_WORD, 0, 0xFu,
                      FT_BH_DATA_FORMAT_CONFIG_BITS(src_format));
    FT_TENSIX_CFG_RMW(FT_BH_CFG_UNPACKER0_OUTPUT_FORMAT_WORD, 0, 0xFu,
                      FT_BH_DATA_FORMAT_CONFIG_BITS(dst_format));
    FT_BH_PUSH_SETDMAREG(0, FT_TENSIX_LOWER_HALFWORD(tile_size), 0,
                         FT_TENSIX_LO_16(FT_TENSIX_GPR_UNPACK_TILE_SIZE_A));
}
#endif

/* ===== Eltwise verbs (resident_eltwise_compute.c's, with a tile index) =================== */

/** Copy init: UNPACK first, then MATH (eltwise_copy_init). */
static FT_BH_INLINE void ftsw_copy_init(uint32_t cb)
{
    (void)cb;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a_init(
        ft_cb_unpack_tile_face_rows[cb],
        (uint32_t)ft_cb_unpack_num_face_rows[cb] * ft_cb_unpack_num_face_cols[cb]));
    FT_TENSIX_MATH(ft_tensix_math_datacopy_init((uint32_t)ft_cb_unpack_num_face_rows[cb] *
                                                ft_cb_unpack_num_face_cols[cb]));
}

/** Copy tile `tile` of `cb` into DST tile `dst` (eltwise_copy_apply at tile 0, DST 0). */
static FT_BH_INLINE void ftsw_copy_apply(uint32_t cb, uint32_t tile, uint32_t dst)
{
    (void)cb;
    (void)tile;
    (void)dst;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a(cb, tile));
    FT_TENSIX_MATH(ft_tensix_math_datacopy(dst));
}

/** Multiply init: MATH first, then UNPACK; geometry from `a` only (eltwise_mul_init). */
static FT_BH_INLINE void ftsw_mul_init(uint32_t a, uint32_t b)
{
    (void)a;
    (void)b;
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_init(FT_CB_MATH_FIDELITY,
                                                          ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[a],
                                              ft_cb_unpack_num_face_rows[a],
                                              ft_cb_unpack_num_face_cols[a]));
}

/** Multiply tile `tile_a` of `a` by tile `tile_b` of `b` into DST 0 (eltwise_mul_apply). */
static FT_BH_INLINE void ftsw_mul_apply(uint32_t a, uint32_t b, uint32_t tile_a, uint32_t tile_b)
{
    (void)a;
    (void)b;
    (void)tile_a;
    (void)tile_b;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, tile_a, tile_b));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul(
        FTSW_DST_REGISTER,
        (uint32_t)ft_cb_unpack_num_face_rows[a] * ft_cb_unpack_num_face_cols[a],
        ft_cb_unpack_tile_face_rows[a]));
}

/** Hand DST to PACK and pack DST 0 as one page of `cb` (eltwise commit_dst_to, through this
 *  program's DST sync). */
static void ftsw_commit_dst_to(ftq_context *ctx, uint32_t cb, uint32_t wait_mode)
{
    FT_TENSIX_MATH(ftmm_math_dest_section_done());
    (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, 1, wait_mode);
    FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
    FT_TENSIX_PACK(ft_tensix_pack_tile(FTSW_DST_REGISTER, cb));
    FT_TENSIX_PACK(ftmm_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, 1));
}

/* ===== One output tile =================================================================== */

/**
 * y = silu(gate) * up for one tile: apply_silu_gated_multiply(), the gate at panel index
 * `gate_tile`, the up at `up_tile`, silu staged bf16 through FT_TT_MATMUL_SWIGLU_CB_STAGE. The
 * silu init is redone every tile because the multiply init in between reconfigures the unit.
 */
static void ftsw_silu_gated_multiply(ftq_context *ctx, uint32_t gate_tile, uint32_t up_tile,
                                     uint32_t cb_out, bool row0, uint32_t wait_mode)
{
    const uint32_t cb_panel = FT_TT_MATMUL_SWIGLU_CB_PANEL;
    const uint32_t cb_stage = FT_TT_MATMUL_SWIGLU_CB_STAGE;

    (void)row0;
    ftsw_copy_init(cb_panel);
    FT_TENSIX_MATH(ft_tensix_math_silu_tile_init());
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    ftsw_copy_apply(cb_panel, gate_tile, FTSW_DST_REGISTER);
    if (row0) {
        FT_TENSIX_MATH(ft_tensix_math_silu_tile_row0(FTSW_DST_REGISTER));
    } else {
        FT_TENSIX_MATH(ft_tensix_math_silu_tile(FTSW_DST_REGISTER));
    }
    ftsw_commit_dst_to(ctx, cb_stage, wait_mode);

    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_stage, 1, wait_mode);
    ftsw_mul_init(cb_stage, cb_panel);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    ftsw_mul_apply(cb_stage, cb_panel, 0u, up_tile);
    ftsw_commit_dst_to(ctx, cb_out, wait_mode);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_stage, 1));
}

/* ===== RED: no bf16 staging ============================================================== */

#ifdef FT_TRISC_MATH
/** SFPU view registers from one DST tile to the next (FT_SFPU_BINARY_DST_OFFSET(1)). */
#define FTSW_SFPU_NEXT_TILE (FT_SFPU_BINARY_DST_OFFSET(1) / FT_SFPU_DST_REGISTER_STRIDE)

/** RED body: DST tile 0 <- silu(tile 0) * tile 1, all fp32 (ft_sfpu_calculate_silu's math,
 *  with the multiply by up done on the SFPU instead of through a bf16 stage). */
static FT_BH_INLINE void ftsw_sfpu_silu_times_next_tile(void)
{
    int iteration;
#pragma GCC unroll 8
    for (iteration = 0; iteration < 8; iteration++) {
        ft_sfpu_vector x = FT_SFPU_DST_LOAD_FLOAT(0);
        ft_sfpu_vector up = FT_SFPU_DST_LOAD_FLOAT(FTSW_SFPU_NEXT_TILE);
        ft_sfpu_vector result = ft_sfpu_float_mul(ft_sfpu_float_mul(x, ft_sfpu_sigmoid(x)), up);

        FT_SFPU_DST_STORE_FLOAT(0, result);
        ft_sfpu_dst_advance(1);
    }
}
#endif

/**
 * RED `fuse_swiglu_no_stage` (word 15 bit 26): up copied to DST 1, gate to DST 0, then
 * silu(gate) * up on the SFPU in fp32 -- the staging rounding point is skipped, everything
 * else is kept. The answer must DIFFER by bytes; it is never a production path.
 */
static void ftsw_silu_multiply_unstaged(ftq_context *ctx, uint32_t gate_tile, uint32_t up_tile,
                                        uint32_t cb_out, uint32_t wait_mode)
{
    const uint32_t cb_panel = FT_TT_MATMUL_SWIGLU_CB_PANEL;

    ftsw_copy_init(cb_panel);
    FT_TENSIX_MATH(ft_tensix_math_silu_tile_init());
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    ftsw_copy_apply(cb_panel, up_tile, 1u);
    ftsw_copy_apply(cb_panel, gate_tile, FTSW_DST_REGISTER);
    FT_TENSIX_MATH(FT_TENSIX_MATH_SFPU_UNARY_TILE_RC(FTSW_DST_REGISTER,
                                                     ftsw_sfpu_silu_times_next_tile()));
    ftsw_commit_dst_to(ctx, cb_out, wait_mode);
}

#endif /* RESIDENT_MATMUL_SWIGLU_COMPUTE_H */
