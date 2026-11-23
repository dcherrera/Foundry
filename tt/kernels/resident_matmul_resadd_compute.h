/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_matmul_resadd_compute.h
 * @brief The ADD half of a RESADD descriptor (fusion S3a), for resident_matmul_compute.c built
 *   with -DFT_RTRIAD_SWIGLU=1 -DFT_RTRIAD_RESADD=1. Kernel-only; the wire contract is
 *   resident_matmul_resadd_protocol.h. Uses the fused SwiGLU's SrcA switch and DST commit
 *   (resident_matmul_swiglu_compute.h), which is why the RESADD build needs the SwiGLU one.
 *
 * THE SEQUENCE IS ELTWISE ADD'S, CALL FOR CALL. resident_eltwise_compute.c runs an ADD as
 *   configure_for_function(kVecFnAdd) = eltwise_addsub_init(ELTWISE_ADD_WORD, cb_in0), then per
 *   tile: wait_for_dest_available, eltwise_addsub_apply(cb_in0, cb_in1) (unpack_ab of the two
 *   front tiles, ELWADD into DST 0), commit_dst_to(cb_out). Here: the same init with the
 *   residual CB as `a`, then per tile the same calls with the residual tile j as A and the
 *   panel tile j (bf16(A . B)) as B, the commit through this program's DST sync. Operand order
 *   is eltwise's: pn_tt_add(hidden, project) puts hidden in cb_in0, i.e. SrcA.
 *   tests/test_resadd_fusion.c compares the two sequences from the two sources.
 *
 * @status Added with S3a (q38dec-p5-s3).
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_MATMUL_RESADD_COMPUTE_H
#define RESIDENT_MATMUL_RESADD_COMPUTE_H

#include <stdbool.h>
#include <stdint.h>

#include "resident_matmul_resadd_protocol.h"
#include "resident_matmul_swiglu_compute.h"

/** resident_eltwise_compute.c's ELTWISE_ADD_WORD, verbatim. */
#define FTRA_ELTWISE_ADD_WORD                                                                  \
    FT_BH_ENC_ELWADD(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_0, 0)

/** Add/sub init (always lowest fidelity): MATH first, then UNPACK (eltwise_addsub_init). */
static FT_BH_INLINE void ftra_addsub_init(uint32_t binary_op_word, uint32_t a)
{
    (void)binary_op_word;
    (void)a;
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub_init(
        binary_op_word, (uint32_t)ft_cb_unpack_tile_num_faces[a],
        (uint32_t)ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[a],
                                              ft_cb_unpack_num_face_rows[a],
                                              ft_cb_unpack_num_face_cols[a]));
}

/** Add tile `tile_a` of `a` and tile `tile_b` of `b` into DST 0 (eltwise_addsub_apply). */
static FT_BH_INLINE void ftra_addsub_apply(uint32_t a, uint32_t b, uint32_t tile_a,
                                           uint32_t tile_b)
{
    (void)a;
    (void)b;
    (void)tile_a;
    (void)tile_b;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, tile_a, tile_b));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub(FTSW_DST_REGISTER));
}

/**
 * The ADD half of one column block: `columns` output tiles, residual tile j + panel tile j,
 * packed bf16 to `cb_out`. SrcA must already be bf16. RED (`red_skip`): the panel tile is
 * copied out unchanged, so C = bf16(A . B) and the answer must DIFFER.
 */
static void ftra_add_block(ftq_context *ctx, uint32_t columns, uint32_t cb_out, bool red_skip,
                           uint32_t wait_mode)
{
    const uint32_t cb_panel = FT_TT_MATMUL_SWIGLU_CB_PANEL;
    const uint32_t cb_residual = FT_TT_MATMUL_RESADD_CB_RESIDUAL;
    uint32_t j;

    if (red_skip)
        ftsw_copy_init(cb_panel);
    else
        ftra_addsub_init(FTRA_ELTWISE_ADD_WORD, cb_residual);
    for (j = 0; j < columns; j++) {
        /* The acquire does not zero DST; PACK zeroes it in its section-done. */
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        if (red_skip)
            ftsw_copy_apply(cb_panel, j, FTSW_DST_REGISTER);
        else
            ftra_addsub_apply(cb_residual, cb_panel, j, j);
        ftsw_commit_dst_to(ctx, cb_out, wait_mode);
    }
}

#endif /* RESIDENT_MATMUL_RESADD_COMPUTE_H */
