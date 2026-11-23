/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file vector_add_compute.c
 * @brief Compute half of Foundry's vector-add triad, and the acceptance kernel for plan
 *   Phase 2 ("our C runs our binary on a Tensix core and we can tell that it finished").
 *   Adds tiles from two input circular buffers into one output CB, tile at a time, at 32x32
 *   granularity -- the tile model in docs/tt_kernel_model.md.
 *
 * This ONE source is compiled THREE times, once per TRISC, and every call below is tagged
 * with the thread that executes it: cb_wait_front/cb_pop_front and the unpack survive only on
 * TRISC0, the add only on TRISC1, cb_reserve_back/cb_push_back and the pack only on TRISC2
 * (docs/tt_kernel_model.md 4). As a C kernel the gating is on `FT_TRISC_*` from the command
 * line rather than the `TRISC_*` the C++ prolog defines, because a separate translation unit
 * never sees the prolog.
 *
 * Compile-time args, positional only:
 *   0: cb_in0 index
 *   1: cb_in1 index
 *   2: cb_out index
 * Runtime args:
 *   0: number of tiles to add
 *
 * Build (or build_tensix_program.sh for the whole triad at once):
 *   CTARGS=0,1,16 bash foundry/tt/build_tensix_kernel.sh \
 *       foundry/tt/kernels/vector_add_compute.c /tmp/ttk_vector_add
 *
 * @status CONVERTED, 2026-08-13 (plan Phase 6.6). vector_add_compute.c is retained only as
 *   the diff baseline. Measured against it with tools/diff_tensix_kernel_instructions.sh:
 *     trisc1 (MATH)   .text BYTE-IDENTICAL, 780 B
 *     trisc2 (PACK)   .text differs in THREE places, all provably empty: two stack slots for
 *                     sync_regfile_write's temporary are swapped between its two call sites,
 *                     and two commutative operands are swapped (`beq a4,a5` / `add a5,a6,a3`).
 *                     Same 1440 B, same 28 Tensix instructions in the same order.
 *     trisc0 (UNPACK) SAME Tensix instruction stream, same values, same order, and the same
 *                     config-register and cb_interface stores; 1256 B against 1292 B because
 *                     C++ TAIL-DUPLICATES the unp_cfg_context != 0 arm (carrying a second
 *                     STALLWAIT/MOP/SEMGET and both SETC16 41,{0,257} arms) and hoists six
 *                     read-only CB fields plus the cfg-state-id select out of the tile loop,
 *                     where this build rematerializes them. Register allocation and block
 *                     layout, not a different store -- and the two images agree on ALL inputs,
 *                     including the unreachable unp_cfg_context > 1. Backed on silicon: the
 *                     full C triad ran byte-exact 3/3 on Tensix 20-23 with the C++ triad as
 *                     the control on the same cores (2026-08-13).
 * @issues The compile-time args are CB indices only, so TRISC1's text does not change when
 *   they change (the math thread never sees a CB index). That makes "vary a compile-time arg,
 *   watch the size move" useless as a non-vacuity probe for TRISC1 specifically; the
 *   weakened-vs-un-weakened link contrast is the probe that works for all five processors.
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_cb_formats.h"
#include "ft_tensix_compute.h"
#include "ft_tensix_compute_cb.h"
#include "ft_tensix_kernel_args.h"
#ifdef FT_TRISC_MATH
#include "ft_tensix_compute_math.h"
#endif
#ifdef FT_TRISC_UNPACK
#include "ft_tensix_compute_unpack.h"
#endif
#ifdef FT_TRISC_PACK
#include "ft_tensix_compute_pack.h"
#endif

/* One tile at a time, so only destination register 0 is ever live. DST holds 16 tiles;
 * batching them is a Phase 4 optimisation, not a Phase 2 correctness concern. */
#define DST_REGISTER 0u

void kernel_main(void)
{
    const uint32_t cb_in0 = ft_tensix_compile_time_arg(0);
    const uint32_t cb_in1 = ft_tensix_compile_time_arg(1);
    const uint32_t cb_out = ft_tensix_compile_time_arg(2);
    const uint32_t tile_count = ft_tensix_runtime_arg(0);

    /* binary_op_init_common(cb_in0, cb_in1, cb_out). Eltwise ops use the default source
     * order; only matmul reverses it (docs/tt_kernel_model.md 5) -- getting that wrong yields
     * wrong numbers, not an error, so it is worth stating even when it is the default. */
    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[cb_in0], ft_cb_unpack_src_format[cb_in1], ft_cb_unpack_dst_format[cb_in0],
        ft_cb_unpack_dst_format[cb_in1], ft_cb_unpack_tile_face_rows[cb_in0], ft_cb_unpack_tile_face_rows[cb_in1],
        ft_cb_unpack_tile_num_faces[cb_in0], ft_cb_unpack_tile_num_faces[cb_in1],
        ft_tensix_cb(cb_in0)->fifo_page_size, ft_tensix_cb(cb_in1)->fifo_page_size));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[cb_in0],
                                              ft_cb_unpack_num_face_rows[cb_in0],
                                              ft_cb_unpack_num_face_cols[cb_in0]));

    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[cb_in0],
                                               ft_cb_unpack_dst_format[cb_in1], FT_CB_DST_ACCUM_FP32));

    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(
        ft_cb_pack_src_format[cb_out], ft_cb_pack_dst_format[cb_out], ft_tensix_cb(cb_out)->fifo_page_size,
        ft_cb_pack_tile_num_faces[cb_out], ft_cb_pack_partial_face[cb_out], 0 /*relu_config*/,
        FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[cb_out], ft_cb_pack_tile_face_rows[cb_out],
                                       ft_cb_pack_tile_num_faces[cb_out], 1 /*num_tiles*/));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());

    /* add_tiles_init(cb_in0, cb_in1) -- MATH programs the ADDR_MODs and the MOP, UNPACK
     * re-runs its AB init. */
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub_init(
        FT_BH_ENC_ELWADD(0 /*clr_src*/, 0 /*acc_to_dest*/, FT_BH_ELWADD_SRCB_NO_BCAST,
                        FT_BH_ADDR_MOD_0, 0 /*dst*/),
        (uint32_t)ft_cb_unpack_tile_num_faces[cb_in0], (uint32_t)ft_cb_unpack_tile_face_rows[cb_in0]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[cb_in0],
                                              ft_cb_unpack_num_face_rows[cb_in0],
                                              ft_cb_unpack_num_face_cols[cb_in0]));

    for (uint32_t tile = 0; tile < tile_count; tile++) {
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_in0, 1));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_in1, 1));

        /* tile_regs_acquire(): MATH blocks until DST is free. It does NOT zero DST -- PACK
         * zeroes per tile with TT_ZEROACC, which names ADDR_MOD_1, a slot the MATH thread
         * programs, so the zeroing is cross-thread and invisible here. Whatever writes DST
         * first must not assume a zero start. */
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());

        FT_TENSIX_UNPACK(ft_tensix_unpack_ab(cb_in0, cb_in1, 0, 0));
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub(DST_REGISTER));

        FT_TENSIX_MATH(ft_tensix_math_dest_section_done()); /* tile_regs_commit() */

        /* Reserving output space between commit and wait overlaps the wait for room with the
         * math drain (docs/tt_kernel_model.md 3). */
        FT_TENSIX_PACK(ft_tensix_compute_cb_reserve_back(cb_out, 1));
        FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done()); /* tile_regs_wait() */
        FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER, cb_out));
        FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));

        FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, 1));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in0, 1));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in1, 1));
    }
}
