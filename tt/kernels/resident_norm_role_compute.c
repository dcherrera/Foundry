/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_norm_role_compute.c
 * @brief Compute third of the PERSISTENT NORM ROLE (resident_norm_role_protocol.h): per step,
 *        h (+)= input on this member's slice, x^2 of the slice, the row reduce over ALL T
 *        tiles of every member's x^2 in tile order, then gamma and scale on the slice.
 *
 * EXACT BY CONSTRUCTION — every tile sees today's verbs, operands, order and pack points:
 *   - ADD is the eltwise program's (resident_eltwise_compute.c eltwise_addsub_init/_apply,
 *     residual as operand A, the projection as B, whole tile, packed bf16): pn_tt_add(hidden,
 *     project). LOAD copies the embedding row through DST (bf16 -> DST -> bf16 is the
 *     identity), so h holds today's `hidden` tile bit for bit, all 32 rows.
 *   - x^2, the reduce, eps, rsqrt and the gamma/scale chain are resident_reduce_compute.c's
 *     map_row / reduce_row_sum / chain_step_* for the norm program (one pass: map x*x, SUM,
 *     add eps, rsqrt; chain gamma then scale; half tile), call for call. The reduce reads the
 *     SAME T tiles in the SAME order: each SqAll page is the member's packed x^2 tile with
 *     faces 2-3 zero — what today's cb_mid holds.
 *   - Only scheduling moves: the reduce waits slice by slice instead of for the whole row
 *     (P3's streamed map does the same), and the chain runs in one run of S tiles (P3 chunk).
 *   tests/test_norm_role.c compares the verbs against both sources, init and per tile.
 *
 * h IS RESIDENT: CB H holds the current h in its front S pages across steps. An ADD step
 * packs the new h behind the old and then pops the old; a LOAD step pops the old (a token's
 * first step) and copies the input in.
 *
 * ONE hardware configuration for the whole program: every CB is Float16_b 32x32, so the
 * eltwise and reduce programs' startups configure the units identically.
 *
 * Trip counts come from the descriptor (kNrParamKind / TileCols / SliceTiles): the MATH
 * thread has no CB records, so it cannot read the schedule table the reader checks them
 * against.
 *
 * Compile-time args: 0 wait mode (1 stoppable, ships). CB indices are the protocol's.
 * Runtime args: the ten of resident_queue_protocol.h.
 *
 * @status P0 (Nex-n2 qwen38_tt_persistent_plan.md P0.2). Card gate pending.
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_cb_formats.h"
#include "ft_tensix_compute.h"
#include "ft_tensix_compute_broadcast.h"
#include "ft_tensix_compute_cb.h"
#include "ft_tensix_compute_reduce.h"
#include "ft_tensix_kernel_args.h"
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

#include "resident_norm_role_protocol.h"
#include "resident_queue_protocol.h"

#define DST_REGISTER 0u
#define NR_WAIT_MODE ft_tensix_compile_time_arg(0)

#define NR_CB_IN ((uint32_t)kNrCbIn)
#define NR_CB_SCALER ((uint32_t)kNrCbScaler)
#define NR_CB_COL ((uint32_t)kNrCbCol)
#define NR_CB_H ((uint32_t)kNrCbH)
#define NR_CB_SQ ((uint32_t)kNrCbSq)
#define NR_CB_EPS ((uint32_t)kNrCbEps)
#define NR_CB_GAMMA ((uint32_t)kNrCbGamma)
#define NR_CB_SQALL ((uint32_t)kNrCbSqAll)
#define NR_CB_STAGE ((uint32_t)kNrCbStage)
#define NR_CB_OUT ((uint32_t)kNrCbOut)

/* The eltwise program's ADD word, verbatim (resident_eltwise_compute.c ELTWISE_ADD_WORD). */
#define NR_ELTWISE_ADD_WORD FT_BH_ENC_ELWADD(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_0, 0)

/* The half tile today's decode norm runs (resident_reduce_compute.c, word-3 bit 23): faces
 * 0-1 for the map and the chain. */
#define NR_HALF_FACE_ROWS 1u
#define NR_HALF_FACES 2u

/* ===== Shared ============================================================================ */

/** DST -> `cb` (resident_reduce_compute.c commit_dst_to, = the eltwise one). */
static void nr_commit_dst_to(ftq_context *ctx, uint32_t cb)
{
    FT_TENSIX_MATH(ft_tensix_math_dest_section_done());
    (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, 1, NR_WAIT_MODE);
    FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
    FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER, cb));
    FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, 1));
}

/* ===== The residual (eltwise program verbs) ============================================== */

/** eltwise_addsub_init(ELTWISE_ADD_WORD, a). */
static void nr_add_init(uint32_t a)
{
    (void)a;
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub_init(
        NR_ELTWISE_ADD_WORD, (uint32_t)ft_cb_unpack_tile_num_faces[a],
        (uint32_t)ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[a],
                                              ft_cb_unpack_num_face_rows[a],
                                              ft_cb_unpack_num_face_cols[a]));
}

/** eltwise_addsub_apply(a, b) on tile `a_index` of a and the FRONT tile of b. */
static void nr_add_apply(uint32_t a, uint32_t b, uint32_t a_index)
{
    (void)a;
    (void)b;
    (void)a_index;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, a_index, 0));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub(DST_REGISTER));
}

/** eltwise_copy_init(cb). */
static void nr_copy_init(uint32_t cb)
{
    (void)cb;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a_init(
        ft_cb_unpack_tile_face_rows[cb],
        (uint32_t)ft_cb_unpack_num_face_rows[cb] * ft_cb_unpack_num_face_cols[cb]));
    FT_TENSIX_MATH(ft_tensix_math_datacopy_init((uint32_t)ft_cb_unpack_num_face_rows[cb] *
                                                ft_cb_unpack_num_face_cols[cb]));
}

/** eltwise_copy_apply(cb): the front tile of `cb` into DST. */
static void nr_copy_apply(uint32_t cb)
{
    (void)cb;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a(cb, 0));
    FT_TENSIX_MATH(ft_tensix_math_datacopy(DST_REGISTER));
}

/**
 * The new h: ADD packs h_old[i] + input[i] behind the old h, then pops the old; LOAD pops
 * the old h (if the role has one) and copies the input in. Per tile, today's order: acquire,
 * apply, commit, pop the input.
 */
static void nr_update_residual(ftq_context *ctx, bool add, bool h_live, uint32_t slice)
{
    uint32_t i;

    if (add) {
        (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)NR_CB_H, (int32_t)slice, NR_WAIT_MODE);
        nr_add_init(NR_CB_H);
    } else {
        if (h_live) {
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(NR_CB_H, slice));
        }
        nr_copy_init(NR_CB_IN);
    }
    for (i = 0; i < slice; i++) {
        (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)NR_CB_IN, 1, NR_WAIT_MODE);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        if (add)
            nr_add_apply(NR_CB_H, NR_CB_IN, i);
        else
            nr_copy_apply(NR_CB_IN);
        nr_commit_dst_to(ctx, NR_CB_H);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(NR_CB_IN, 1));
    }
    if (add) {
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(NR_CB_H, slice));
    }
    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)NR_CB_H, (int32_t)slice, NR_WAIT_MODE);
}

/* ===== The norm (reduce program verbs, norm program, half tile) ========================== */

/** map_row(kRedMapSquareSelf) over the slice of h, into Sq. */
static void nr_map_square(ftq_context *ctx, uint32_t slice)
{
    uint32_t wt;

    for (wt = 0; wt < slice; wt++) {
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_init(
            FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[NR_CB_H]));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[NR_CB_H],
                                                  NR_HALF_FACE_ROWS,
                                                  ft_cb_unpack_num_face_cols[NR_CB_H]));
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab(NR_CB_H, NR_CB_H, wt, wt));
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul(
            DST_REGISTER, NR_HALF_FACES, (uint32_t)ft_cb_unpack_tile_face_rows[NR_CB_H]));
        nr_commit_dst_to(ctx, NR_CB_SQ);
    }
}

/** reduce_row_sum over all T SqAll tiles, + eps, rsqrt, into Col. Waits slice by slice. */
static void nr_reduce_sum(ftq_context *ctx, uint32_t tile_cols, uint32_t slice)
{
    uint32_t wt;

    FT_TENSIX_UNPACK(
        ft_tensix_unpack_ab_reduce_row_init((uint32_t)ft_cb_unpack_tile_num_faces[NR_CB_SQALL]));
    FT_TENSIX_MATH(ft_tensix_math_reduce_row_init(FT_CB_MATH_FIDELITY != 0 ? 1u : 0u,
                                                  FT_CB_MATH_FIDELITY));
    FT_TENSIX_PACK(
        ft_tensix_pack_reduce_row_mask_config((uint32_t)ft_cb_pack_tile_face_rows[NR_CB_COL]));
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    for (wt = 0; wt < tile_cols; wt++) {
        if ((wt % slice) == 0u)
            (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)NR_CB_SQALL, (int32_t)(wt + slice),
                                              NR_WAIT_MODE);
        FT_TENSIX_MATH(ft_tensix_math_reduce_row(DST_REGISTER));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_reduce_row(NR_CB_SQALL, NR_CB_SCALER, wt, 0));
    }
    FT_TENSIX_MATH(ft_tensix_math_reduce_row(DST_REGISTER));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_reduce_row(NR_CB_EPS, NR_CB_SCALER, 0, 0));
    FT_TENSIX_MATH(ft_tensix_math_rsqrt_tile_init());
    FT_TENSIX_MATH(ft_tensix_math_rsqrt_tile(DST_REGISTER));
    nr_commit_dst_to(ctx, NR_CB_COL);
    FT_TENSIX_MATH(ft_tensix_math_reduce_uninit());
    FT_TENSIX_PACK(ft_tensix_pack_reduce_mask_clear());
}

/** chain_step_init(kRedChainGamma, H) then the gamma tiles into Stage; chain_step_init(
 *  kRedChainScale, Stage) then the scale tiles into Out — final_row with one run of S. */
static void nr_chain(ftq_context *ctx, uint32_t slice)
{
    uint32_t t;

    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_init(
        FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[NR_CB_H]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[NR_CB_H],
                                              NR_HALF_FACE_ROWS,
                                              ft_cb_unpack_num_face_cols[NR_CB_H]));
    for (t = 0; t < slice; t++) {
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab(NR_CB_H, NR_CB_GAMMA, t, t));
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul(
            DST_REGISTER, NR_HALF_FACES, (uint32_t)ft_cb_unpack_tile_face_rows[NR_CB_H]));
        nr_commit_dst_to(ctx, NR_CB_STAGE);
    }
    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)NR_CB_STAGE, (int32_t)slice, NR_WAIT_MODE);

    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_broadcast_col_init(
        FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[NR_CB_STAGE]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_broadcast_col_init(
        ft_cb_unpack_tile_face_rows[NR_CB_STAGE], NR_HALF_FACE_ROWS,
        ft_cb_unpack_num_face_cols[NR_CB_STAGE]));
    for (t = 0; t < slice; t++) {
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab(NR_CB_STAGE, NR_CB_COL, 0, 0));
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_broadcast_col(
            DST_REGISTER, FT_CB_MATH_FIDELITY, NR_HALF_FACE_ROWS,
            (uint32_t)ft_cb_unpack_num_face_cols[NR_CB_STAGE],
            (uint32_t)ft_cb_unpack_tile_face_rows[NR_CB_STAGE]));
        nr_commit_dst_to(ctx, NR_CB_OUT);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(NR_CB_STAGE, 1));
    }
}

/* ===== One step ========================================================================== */

/** The compute half of one role step. Trip counts come from the table alone (rule 3). */
static void nr_compute_step(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                            bool *h_live)
{
    const uint32_t kind = parameters[kNrParamKind];
    const uint32_t tile_cols = parameters[kNrParamTileCols];
    const uint32_t slice = parameters[kNrParamSliceTiles];
    const bool add = kind == (uint32_t)kNrKindAdd;

    /* The reader refuses a descriptor that disagrees with the table; this only keeps the
     * trip counts finite on every thread. */
    if (parameters[kNrParamMagic] != (uint32_t)kNrMagic || slice == 0u || tile_cols == 0u ||
        tile_cols % slice != 0u || (kind != (uint32_t)kNrKindAdd && kind != (uint32_t)kNrKindLoad)) {
        ftq_abort(ctx, kErrBadParameters);
        return;
    }

    nr_update_residual(ctx, add, *h_live, slice);
    *h_live = true;
    nr_map_square(ctx, slice);

    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)NR_CB_SCALER, 1, NR_WAIT_MODE);
    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)NR_CB_EPS, 1, NR_WAIT_MODE);
    nr_reduce_sum(ctx, tile_cols, slice);
    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)NR_CB_COL, 1, NR_WAIT_MODE);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(NR_CB_SQALL, tile_cols));

    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)NR_CB_GAMMA, (int32_t)slice, NR_WAIT_MODE);
    nr_chain(ctx, slice);

    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(NR_CB_COL, 1));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(NR_CB_SCALER, 1));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(NR_CB_EPS, 1));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(NR_CB_GAMMA, slice));
}

void kernel_main(void)
{
    ftq_context ctx;
    bool running = true;
    bool h_live = false;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);
    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    /* compute_kernel_hw_startup(in, scaler, out): the reduce program's, which every CB of
     * this table (one format, one page size) makes the eltwise program's as well. */
    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[NR_CB_IN], ft_cb_unpack_src_format[NR_CB_SCALER],
        ft_cb_unpack_dst_format[NR_CB_IN], ft_cb_unpack_dst_format[NR_CB_SCALER],
        ft_cb_unpack_tile_face_rows[NR_CB_IN], ft_cb_unpack_tile_face_rows[NR_CB_SCALER],
        ft_cb_unpack_tile_num_faces[NR_CB_IN], ft_cb_unpack_tile_num_faces[NR_CB_SCALER],
        ft_tensix_cb(NR_CB_IN)->fifo_page_size, ft_tensix_cb(NR_CB_SCALER)->fifo_page_size));
    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[NR_CB_IN],
                                               ft_cb_unpack_dst_format[NR_CB_SCALER],
                                               FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(
        ft_cb_pack_src_format[NR_CB_OUT], ft_cb_pack_dst_format[NR_CB_OUT],
        ft_tensix_cb(NR_CB_OUT)->fifo_page_size, ft_cb_pack_tile_num_faces[NR_CB_OUT],
        ft_cb_pack_partial_face[NR_CB_OUT], 0 /*relu_config*/, FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[NR_CB_OUT],
                                       ft_cb_pack_tile_face_rows[NR_CB_OUT],
                                       ft_cb_pack_tile_num_faces[NR_CB_OUT], 1 /*num_tiles*/));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());

    ctx.cursor = ctx.control[kFieldCompletedCount];
    while (running) {
        volatile FTQ_L1_PTR uint32_t *descriptor;
        uint32_t opcode;

        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;
        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);
        descriptor = ftq_descriptor(&ctx);
        opcode = descriptor[kDescOpcode];
        ftq_status_opcode(&ctx, opcode);
        if (opcode == kOpReduce) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            nr_compute_step(&ctx, ftq_parameters(descriptor), &h_live);
            ftq_status_set(&ctx, kStateExec);
            if (ctx.aborting)
                running = false;
        } else if (opcode == kOpStop) {
            running = false;
        }
        if (!ctx.aborting)
            ctx.cursor++;
    }
    ftq_status_set(&ctx, kStateExit);
}
