/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file matmul_compute.c
 * @brief Compute third of Foundry's matmul triad -- the load-bearing op of the whole backend
 *   (spec 5.1 fc_sgemm/fc_sgemv). Blocked outer product at 32x32 tile granularity:
 *   C[mt][nt] = sum over kt of A[mt][kt] * B[kt][nt], accumulated in one destination register
 *   across the K loop because a tile matmul accumulates into DST rather than overwriting it
 *   (docs/tt_kernel_model.md 1).
 *
 * SHAPES ARE RUNTIME ARGS, NOT COMPILE-TIME ARGS. Upstream's matmul_single_core takes Mt/Kt/Nt
 * as compile-time args; one binary then serves exactly one shape, and a rebuild here is ~4
 * minutes on the card host's two cores. MiniCPM5-1B alone needs six distinct shapes, so this kernel
 * takes them at run time and the compile-time args are CB indices only.
 *
 * ORIENTATION: in0 is the ACTIVATION, [tokens x in_features], and in1 is the WEIGHT in
 * [in_features x out_features] -- i.e. K x N, the transpose of the [out, in] order GGUF stores.
 * See docs/tt_kernel_model.md and the plan's Phase 4 orientation note.
 *
 * MATMUL IS THE ONE OP WITH REVERSED SOURCE ORDER: in0 goes to SrcB and in1 to SrcA. That is
 * why every hardware-configuration call below is passed (cb_in1, cb_in0) and not (cb_in0,
 * cb_in1) -- it is `compute_kernel_hw_startup<SrcOrder::Reverse>` written out. Getting it
 * wrong produces wrong numbers, not an error.
 *
 * Compile-time args (positional):
 *   0: cb_in0 index (activation)   1: cb_in1 index (weight)   2: cb_out index
 * Runtime args:
 *   0: Mt (activation tile rows)  1: Kt (shared dim in tiles)  2: Nt (output tile cols)
 *
 * The reader must push tiles in exactly the order consumed here: for each (mt, nt), Kt pairs of
 * (A tile, B tile). Two pages per input CB is enough -- the CB is a pipeline, not a cache.
 *
 * Build:
 *   CTARGS=0,1,16 CB_TABLE=kernels/cb_tables/matmul_float16b.cbtable \
 *       bash foundry/tt/build_tensix_kernel.sh foundry/tt/kernels/matmul_compute.c /tmp/ttk_mm
 *
 * @status CONVERSION MEASURED, 2026-08-13 (plan Phase 6.6). NOT YET THE SHIPPED KERNEL --
 *   matmul_compute.c is retained as the baseline until all three images are byte-identical,
 *   because there is no partial credit. Against the C++ build with
 *   tools/diff_tensix_kernel_instructions.sh:
 *     trisc1 (MATH)   .text BYTE-IDENTICAL, 900 B
 *     trisc0 (UNPACK) Tensix instruction stream identical 31 of 31 (same words, same order)
 *                     and every computed instruction-word immediate identical; C is 0x600 vs
 *                     0x620 because C++ hoists the CB fields and the cfg pointer out of the
 *                     tile loop and spills one more callee-saved register
 *     trisc2 (PACK)   Tensix stream identical 37 of 37 and the same 0x5d0; differs only in two
 *                     stack slots for sync_regfile_write's temporary and two commutative
 *                     operand orders
 *   See docs/tt_bringup_log.md 14 for measured text sizes and on-card results.
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_cb_formats.h"
#include "ft_tensix_compute.h"
#include "ft_tensix_compute_cb.h"
#include "ft_tensix_compute_matmul.h"
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

/* One output tile at a time, so only destination register 0 is ever live. */
#define DST_REGISTER 0u

#ifdef FT_TRISC_MATH
/* Fidelity is a MULTIPLIER on math cycles, not a mode: the MOP's inner loop count IS the
 * fidelity value, so a tile is 16 MVMULs at LoFi and 64 at HiFi4. It comes from the CB table,
 * which is where the choice was actually made (kernels/cb_tables/matmul_float16b.cbtable). */
#define MATMUL_HIGH_FIDELITY (FT_CB_MATH_FIDELITY != 0)
#define MATMUL_INNER_LOOPS (MATMUL_HIGH_FIDELITY ? (uint32_t)FT_CB_MATH_FIDELITY : 1u)
#define MATMUL_FIDELITY_INCREMENT (MATMUL_HIGH_FIDELITY ? 1u : 0u)
#endif

void kernel_main(void)
{
    const uint32_t cb_in0 = ft_tensix_compile_time_arg(0);
    const uint32_t cb_in1 = ft_tensix_compile_time_arg(1);
    const uint32_t cb_out = ft_tensix_compile_time_arg(2);

    const uint32_t tile_rows = ft_tensix_runtime_arg(0);
    const uint32_t tile_inner = ft_tensix_runtime_arg(1);
    const uint32_t tile_cols = ft_tensix_runtime_arg(2);

    uint32_t mt;
    uint32_t nt;
    uint32_t kt;

    /* compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out): the source
     * registers are programmed with the operands SWAPPED, so src_a is cb_in1 and src_b is
     * cb_in0 in every call below. */
    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[cb_in1], ft_cb_unpack_src_format[cb_in0], ft_cb_unpack_dst_format[cb_in1],
        ft_cb_unpack_dst_format[cb_in0], ft_cb_unpack_tile_face_rows[cb_in1],
        ft_cb_unpack_tile_face_rows[cb_in0], ft_cb_unpack_tile_num_faces[cb_in1],
        ft_cb_unpack_tile_num_faces[cb_in0], ft_tensix_cb(cb_in1)->fifo_page_size,
        ft_tensix_cb(cb_in0)->fifo_page_size));

    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[cb_in1],
                                               ft_cb_unpack_dst_format[cb_in0], FT_CB_DST_ACCUM_FP32));

    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(
        ft_cb_pack_src_format[cb_out], ft_cb_pack_dst_format[cb_out], ft_tensix_cb(cb_out)->fifo_page_size,
        ft_cb_pack_tile_num_faces[cb_out], ft_cb_pack_partial_face[cb_out], 0 /*relu_config*/,
        FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[cb_out], ft_cb_pack_tile_face_rows[cb_out],
                                       ft_cb_pack_tile_num_faces[cb_out], 1 /*num_tiles*/));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());

    /* matmul_init(cb_in0, cb_in1) with transpose 0. The unpacker's A operand is cb_in1. */
    FT_TENSIX_MATH(ft_tensix_math_matmul_init(0 /*transpose*/, MATMUL_FIDELITY_INCREMENT,
                                              MATMUL_INNER_LOOPS, MATMUL_HIGH_FIDELITY));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul_init(
        0 /*transpose*/,
        (uint32_t)ft_cb_unpack_tile_num_faces[cb_in1] * ft_cb_unpack_tile_face_rows[cb_in1] *
                FT_TENSIX_FACE_C_DIM - 1u,
        (uint32_t)ft_cb_unpack_tile_num_faces[cb_in0] * ft_cb_unpack_tile_face_rows[cb_in0] *
                FT_TENSIX_FACE_C_DIM - 1u,
        1u /*kt_dim*/));

    for (mt = 0; mt < tile_rows; mt++) {
        for (nt = 0; nt < tile_cols; nt++) {
            /* tile_regs_acquire(): MATH blocks until DST is free. It does NOT zero DST -- PACK
             * zeroes per tile with TT_ZEROACC, which names ADDR_MOD_1, a slot the MATH thread
             * programs, so the zeroing is cross-thread and invisible here. */
            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());

            for (kt = 0; kt < tile_inner; kt++) {
                FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_in0, 1));
                FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_in1, 1));

                FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul(cb_in0, cb_in1, 0, 0, 1 /*ct_dim*/));
                FT_TENSIX_MATH(ft_tensix_math_matmul(DST_REGISTER));

                FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in0, 1));
                FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in1, 1));
            }

            FT_TENSIX_MATH(ft_tensix_math_dest_section_done()); /* tile_regs_commit() */

            FT_TENSIX_PACK(ft_tensix_compute_cb_reserve_back(cb_out, 1));
            FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done()); /* tile_regs_wait() */
            FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER, cb_out));
            FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
            FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, 1));
        }
    }
}
