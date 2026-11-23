/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file rms_norm_compute.c
 * @brief Compute third of Foundry's RMS-norm triad -- the `fc_rms_norm` equivalent
 *   (spec 5.1). For each row r of a [rows x W] activation:
 *
 *       y[r][c] = x[r][c] * gamma[c] * rsqrt( mean_c(x[r][c]^2) + eps )
 *
 *   at 32x32 tile granularity, so one "row block" is 32 rows wide by Wt tiles.
 *
 * THREE THINGS HERE ARE WORTH READING BEFORE EDITING, and they are the original C++
 * kernel's, unchanged by the port:
 *
 *   1. eps COSTS NO CODE. The row-sum is produced by a row-wise SUM reduce against a scaler
 *      tile holding 1/W, and the reduce ACCUMULATES into DST across calls. So one extra
 *      reduce against a constant CB whose every element is eps*W/32 adds exactly eps to each
 *      row's mean square -- no broadcast add, no extra pack, no extra CB round trip. (Each
 *      row of that tile has 32 columns; 32 * (eps*W/32) * (1/W) = eps.) The host builds that
 *      tile; see ft_tt_op_rms_norm.c.
 *
 *   2. gamma ARRIVES AS FULL 32x32 TILES, not a broadcast row. The norm weight is a 1-D
 *      tensor and the FTTW artifact stores 1-D tensors untiled (bringup log 12.7). The
 *      LOADER expands it to Wt tiles with the row replicated 32 times, which turns the gamma
 *      multiply into a plain tile multiply instead of a broadcast op and costs 96 KiB of L1
 *      for a 1536-wide model.
 *
 *   3. THE SCALER AND EPS CBs ARE PUSHED ONCE AND NEVER POPPED. A CB wait is idempotent as
 *      long as nothing pops, so they stay resident for every row block.
 *
 * WHAT THE PORT ADDS, AND THE ONE HAZARD IT MAKES VISIBLE. Three things in this kernel are
 * only correct because of statement ORDER, and none of them is enforced by a type:
 *
 *   * `reduce_uninit` IS THE ONLY THING THAT CLEARS THE PACKER EDGE MASK. Between
 *     `ft_tensix_pack_reduce_row_mask_config()` and `ft_tensix_pack_reduce_mask_clear()`
 *     EVERY packed tile carries one datum per row and zeros elsewhere. MEASURED 2026-08-13:
 *     delete the clear at the bottom of the reduce block below and this kernel goes from
 *     PCC 0.999978735 to **PCC 0.195355069**, because `x*gamma` and the final product are
 *     then packed through the reduce mask. It is one line and nothing else moves -- the other
 *     four images stay byte-identical.
 *     The state IS per-core and DOES outlive the program, but the "so a kernel that exits in
 *     between poisons the next image on that core" this used to say is RETRACTED: the next
 *     image's `ft_tensix_pack_hw_configure()` rewrites `PCK_EDGE_OFFSET_SEC0 = 0xffff` with
 *     its four row-set selectors zeroed and `TILE_ROW_SET_MAPPING_0 = 0`, which makes the
 *     stale cfg 25 / cfg 21 unreachable. Proven both ways on cores 60-63 -- a matmul run on
 *     the same core in the same process after this kernel leaked scored 0.999998638, and only
 *     went to 0.25 once those two lines were deleted from its own startup. The live hazard is
 *     the rest of THIS kernel, and a RESIDENT triad, which runs startup once and then serves
 *     descriptors with nothing between them. `tools/check_tensix_reduce_exit_state.py`.
 *   * After the reduce, MATH still holds reduce's ADDR_MOD_0..3, its MOP template and its
 *     eight MVMULs in replay slots 16-23, and UNPACK still has the haloize bit SET. The
 *     multiply init that follows is what reprograms them, and nothing may unpack in between.
 *   * `ft_tensix_math_wait_for_dest_available()` (`tile_regs_acquire`) does NOT zero DST.
 *     The reduce is an accumulation into DST, so what makes each row block start from zero
 *     is PACK's per-tile `TT_ZEROACC` -- cross-thread, and invisible in this file.
 *
 * AND THE COLUMN-0 CONTRACT IS NOT "GARBAGE ELSEWHERE". the reduce kernel API states that while
 * the reduce packer mask is programmed, every non-column-0 datum is packed as ZERO. So the
 * `cb_scale` tile is a genuine column vector, and the column broadcast that consumes it
 * reads column 0 only. Two comments in this tree used to claim the other lanes held
 * leftovers; they were wrong and are corrected here.
 *
 * Compile-time args (positional):
 *   0: cb_in       activations, Wt pages
 *   1: cb_gamma    expanded norm weight, Wt pages, pushed once
 *   2: cb_scaler   one tile of 1/W, pushed once
 *   3: cb_eps      one tile of eps*W/32, pushed once
 *   4: cb_xx       intermediate x*x, Wt pages, produced and consumed here
 *   5: cb_scale    intermediate rsqrt result, 1 page, produced and consumed here
 *   6: cb_xg       intermediate x*gamma, 1 page, produced and consumed here
 *   7: cb_out      output, >= 2 pages
 * Runtime args:
 *   0: number of row blocks (rows / 32)
 *   1: Wt (W / 32)
 *
 * Build:
 *   CTARGS=0,1,2,3,4,5,6,16 bash foundry/tt/build_tensix_kernel.sh \
 *       foundry/tt/kernels/rms_norm_compute.c /tmp/ttk_rms_norm
 *
 * IT BUILDS `--uniform-default`, i.e. DST_ACCUM_MODE 0 -- no `CB_TABLE` on that command line.
 * That is not a detail: it selects rsqrt's bf16-ROUNDING arm (a `sfpstochrnd` before every
 * DST store) rather than the fp32-dest arm. Both are compiled into
 * `ft_tensix_sfpu_math.h`; changing the build to a table with `dst_accum_mode 1` changes
 * which one is emitted and nothing in this file.
 *
 * @status CONVERTED, 2026-08-13 (plan Phase 6.6), first build and first diff. `rms_norm_
 *   compute.cpp` is retained only as the diff baseline. Measured against it with
 *   tools/diff_tensix_kernel_instructions.sh, per image:
 *     trisc2 (PACK)   Tensix stream BYTE-IDENTICAL, 91 instructions, and the SAME 2548 B
 *                     `.text`. 32 RISC-V lines differ, all of them one register swapped
 *                     (a0<->a1) and two stack slots swapped (104<->108) -- the same
 *                     spill-slot pattern vector_add's PACK and the MAX probe pair showed.
 *     trisc1 (MATH)   identical opcode histogram on all 25 mnemonics EXCEPT `ttmop`, 8 vs
 *                     14; 2504 vs 2500 B. The six extra are C UNROLLING the two `mul_tiles`
 *                     face loops (num_faces = 4, a compile-time constant in both arms -- the
 *                     C++ image proves it with `li a5,4`) where C++ keeps a four-trip loop,
 *                     and the loop bookkeeping C drops pays for them almost exactly. Every
 *                     one of the 232 SFPU instructions, all 8 MVMULs, the replay record and
 *                     all 59 SETC16 are in the same order. After normalising registers and
 *                     branch targets the ONLY remaining delta is that unrolling: 6 added
 *                     `ttmop`, 2 removed `li 4`, 3 removed `addi -1`.
 *     trisc0 (UNPACK) identical opcode histogram except STALLWAIT/MOP/SEMGET at 10 vs 5 --
 *                     C++ TAIL-DUPLICATES the `unp_cfg_context` branch and carries a second
 *                     static copy of those three per unpack. There are exactly 5 unpack call
 *                     sites in this kernel and C emits exactly 5 of each; the C image is
 *                     60 B SMALLER (2448 -> 2388), which is 5 x 3 instructions to the byte.
 *                     Same systemic C-vs-C++ layout gap the plan already records for
 *                     `vector_add_compute` and the MAX probe pair.
 *   This kernel is also the FIRST CALLER of `ft_tensix_compute_reduce.h`'s SUM arm, which
 *   until now had never been diffed against a C++ build; it PASSES, per image and per
 *   function (isolation unit 6).
 * @issues rsqrt runs over the whole DST tile, but only column 0 of the reduce result is
 *   meaningful; the other columns are ZERO (see the packer contract above), and rsqrt(0) is
 *   +inf. The column broadcast reads column 0 only, so those lanes are never consumed. Do
 *   not "fix" this by zeroing them -- it would cost a pass.
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

/* One tile at a time, so only destination register 0 is ever live. */
#define DST_REGISTER 0u

void kernel_main(void)
{
    const uint32_t cb_in = ft_tensix_compile_time_arg(0);
    const uint32_t cb_gamma = ft_tensix_compile_time_arg(1);
    const uint32_t cb_scaler = ft_tensix_compile_time_arg(2);
    const uint32_t cb_eps = ft_tensix_compile_time_arg(3);
    const uint32_t cb_xx = ft_tensix_compile_time_arg(4);
    const uint32_t cb_scale = ft_tensix_compile_time_arg(5);
    const uint32_t cb_xg = ft_tensix_compile_time_arg(6);
    const uint32_t cb_out = ft_tensix_compile_time_arg(7);

    const uint32_t row_blocks = ft_tensix_runtime_arg(0);
    const uint32_t tile_cols = ft_tensix_runtime_arg(1);

    uint32_t block;
    uint32_t wt;

    /* compute_kernel_hw_startup(cb_in, cb_in, cb_xx) -- both source registers take cb_in. */
    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[cb_in], ft_cb_unpack_src_format[cb_in], ft_cb_unpack_dst_format[cb_in],
        ft_cb_unpack_dst_format[cb_in], ft_cb_unpack_tile_face_rows[cb_in], ft_cb_unpack_tile_face_rows[cb_in],
        ft_cb_unpack_tile_num_faces[cb_in], ft_cb_unpack_tile_num_faces[cb_in],
        ft_tensix_cb(cb_in)->fifo_page_size, ft_tensix_cb(cb_in)->fifo_page_size));

    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[cb_in],
                                               ft_cb_unpack_dst_format[cb_in], FT_CB_DST_ACCUM_FP32));

    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(
        ft_cb_pack_src_format[cb_xx], ft_cb_pack_dst_format[cb_xx], ft_tensix_cb(cb_xx)->fifo_page_size,
        ft_cb_pack_tile_num_faces[cb_xx], ft_cb_pack_partial_face[cb_xx], 0 /*relu_config*/,
        FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[cb_xx], ft_cb_pack_tile_face_rows[cb_xx],
                                       ft_cb_pack_tile_num_faces[cb_xx], 1 /*num_tiles*/));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());

    /* The constants the reader pushes once. Waiting for them here rather than inside the
     * block loop is what makes "pushed once, never popped" work. */
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_scaler, 1));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_eps, 1));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_gamma, tile_cols));

    for (block = 0; block < row_blocks; block++) {
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_in, tile_cols));

        /* --- x*x into cb_xx ------------------------------------------------- */
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_init(
            FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[cb_in]));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[cb_in],
                                                  ft_cb_unpack_num_face_rows[cb_in],
                                                  ft_cb_unpack_num_face_cols[cb_in]));

        for (wt = 0; wt < tile_cols; wt++) {
            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
            FT_TENSIX_UNPACK(ft_tensix_unpack_ab(cb_in, cb_in, wt, wt));
            FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul(
                DST_REGISTER, (uint32_t)ft_cb_unpack_tile_num_faces[cb_in],
                (uint32_t)ft_cb_unpack_tile_face_rows[cb_in]));
            FT_TENSIX_MATH(ft_tensix_math_dest_section_done());

            FT_TENSIX_PACK(ft_tensix_compute_cb_reserve_back(cb_xx, 1));
            FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
            FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER, cb_xx));
            FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
            FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_xx, 1));
        }

        /* --- mean square + eps, then rsqrt, into cb_scale -------------------- */
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_xx, tile_cols));

        /* reduce_init<SUM, REDUCE_ROW>(cb_xx, cb_scaler, cb_scale). The SUM arm SWAPS the
         * operands -- the scaler goes to SrcA and the data to SrcB -- which is what makes the
         * row reduce a matrix multiply. */
        FT_TENSIX_UNPACK(
            ft_tensix_unpack_ab_reduce_row_init((uint32_t)ft_cb_unpack_tile_num_faces[cb_xx]));
        FT_TENSIX_MATH(ft_tensix_math_reduce_row_init(
            FT_CB_MATH_FIDELITY != 0 ? 1u : 0u, FT_CB_MATH_FIDELITY));
        FT_TENSIX_PACK(
            ft_tensix_pack_reduce_row_mask_config((uint32_t)ft_cb_pack_tile_face_rows[cb_scale]));

        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());

        for (wt = 0; wt < tile_cols; wt++) {
            FT_TENSIX_MATH(ft_tensix_math_reduce_row(DST_REGISTER));
            FT_TENSIX_UNPACK(ft_tensix_unpack_ab_reduce_row(cb_xx, cb_scaler, wt, 0));
        }

        /* eps, as one more accumulated row-sum. See note 1 in the file header. */
        FT_TENSIX_MATH(ft_tensix_math_reduce_row(DST_REGISTER));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_reduce_row(cb_eps, cb_scaler, 0, 0));

        FT_TENSIX_MATH(ft_tensix_math_rsqrt_tile_init());
        FT_TENSIX_MATH(ft_tensix_math_rsqrt_tile(DST_REGISTER));

        FT_TENSIX_MATH(ft_tensix_math_dest_section_done());

        FT_TENSIX_PACK(ft_tensix_compute_cb_reserve_back(cb_scale, 1));
        FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
        FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER, cb_scale));
        FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
        FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_scale, 1));

        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_xx, tile_cols));

        /* reduce_uninit(). MATH restores NOTHING (the body is empty upstream); PACK's edge
         * mask restore is the whole of it, and it is not optional -- see the file header. */
        FT_TENSIX_MATH(ft_tensix_math_reduce_uninit());
        FT_TENSIX_PACK(ft_tensix_pack_reduce_mask_clear());

        /* --- y = (x * gamma) * scale ---------------------------------------- */
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_scale, 1));

        for (wt = 0; wt < tile_cols; wt++) {
            FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_init(
                FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[cb_in]));
            FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[cb_in],
                                                      ft_cb_unpack_num_face_rows[cb_in],
                                                      ft_cb_unpack_num_face_cols[cb_in]));

            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
            FT_TENSIX_UNPACK(ft_tensix_unpack_ab(cb_in, cb_gamma, wt, wt));
            FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul(
                DST_REGISTER, (uint32_t)ft_cb_unpack_tile_num_faces[cb_in],
                (uint32_t)ft_cb_unpack_tile_face_rows[cb_in]));
            FT_TENSIX_MATH(ft_tensix_math_dest_section_done());

            FT_TENSIX_PACK(ft_tensix_compute_cb_reserve_back(cb_xg, 1));
            FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
            FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER, cb_xg));
            FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
            FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_xg, 1));

            FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_xg, 1));

            /* mul_bcast_cols_init_short(cb_xg, cb_scale) -- "short" means it reprograms the
             * per-op ADDR_MOD and MOP state only; the hardware configure and the packer init
             * from startup still stand. */
            FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_broadcast_col_init(
                FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[cb_xg]));
            FT_TENSIX_UNPACK(ft_tensix_unpack_ab_broadcast_col_init(
                ft_cb_unpack_tile_face_rows[cb_xg], ft_cb_unpack_num_face_rows[cb_xg],
                ft_cb_unpack_num_face_cols[cb_xg]));

            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
            FT_TENSIX_UNPACK(ft_tensix_unpack_ab(cb_xg, cb_scale, 0, 0));
            FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_broadcast_col(
                DST_REGISTER, FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_num_face_rows[cb_xg],
                (uint32_t)ft_cb_unpack_num_face_cols[cb_xg],
                (uint32_t)ft_cb_unpack_tile_face_rows[cb_xg]));
            FT_TENSIX_MATH(ft_tensix_math_dest_section_done());

            FT_TENSIX_PACK(ft_tensix_compute_cb_reserve_back(cb_out, 1));
            FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
            FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER, cb_out));
            FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
            FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, 1));

            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_xg, 1));
        }

        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_scale, 1));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in, tile_cols));
    }
}
