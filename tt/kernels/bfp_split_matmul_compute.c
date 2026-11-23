/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bfp_split_matmul_compute.c
 * @brief Compute of the qwen36 plan 1.4b price bench: matmul_compute.c's blocked outer
 *   product with TWO weight passes into ONE Dst tile. The high pass unpacks SrcA from cb_in1
 *   (BFP4_B), then the unpacker's SrcA format is switched to cb_in2's (BFP2_B) and the low
 *   pass accumulates on top, then it is switched back. No expansion step: both formats are
 *   ones the unpacker reads natively.
 *
 * THE SWITCH (ft_tensix_unpack_reconfigure_srca_format): both unpack config contexts drained,
 * STALLWAIT until unpacker 0 is idle, then the tile-descriptor and output format nibbles
 * rewritten through RMWCIB (ordered in the Tensix stream). It does NOT re-run
 * hw_configure_ab — re-running that mid-kernel is what broke the M5 row-0 builds
 * (Ornith spec "M5").
 *
 * FIDELITY PHASES (compile-time arg 4). 4 = HiFi4 (phases 0,1,2,3). 2 = phases 0 and 2 only
 * (fidelity increment 2): the SrcA-RESIDUAL phases 1 and 3 multiply SrcA's mantissa bits past
 * implicit+4 (MatrixUnit.md SrcAFidelityBits), which are ZERO for any weight with <= 5
 * significant bits — BFP4_B, BFP2_B and an expanded BFP6_B tile — so those phases add exact
 * zeros and skipping them must be BIT-IDENTICAL to HiFi4 for such weights. The bench checks
 * exactly that, and its red (a 7-bit BFP8_B weight) must DIFFER.
 *
 * Compile-time args: 0 cb_in0 (activation)  1 cb_in1 (high)  2 cb_in2 (low)  3 cb_out
 *                    4 fidelity phases (4 or 2)
 * Runtime args: 0 Mt  1 Kt  2 Nt  3 mode (1 high only, 2 low only, 3 both)
 *
 * @status New 2026-10-28 (qwen36 Phase 1.4b price bench).
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

#define DST_REGISTER 0u

#ifdef FT_TRISC_UNPACK
/** Point unpacker 0 (SrcA, the weight) at a new block-float format between tiles. */
static void unpack_switch_weight_format(uint32_t cb)
{
    ft_tensix_unpack_reconfigure_srca_format(ft_cb_unpack_src_format[cb], ft_cb_unpack_dst_format[cb],
                                             ft_tensix_cb(cb)->fifo_page_size);
}
#endif

/** One pass: Kt (activation, weight) tile pairs from cb_weight accumulated into Dst. */
static void matmul_pass(uint32_t cb_in0, uint32_t cb_weight, uint32_t tile_inner)
{
    for (uint32_t kt = 0; kt < tile_inner; kt++) {
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_in0, 1));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_weight, 1));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul(cb_in0, cb_weight, 0, 0, 1));
        FT_TENSIX_MATH(ft_tensix_math_matmul(DST_REGISTER));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in0, 1));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_weight, 1));
    }
}

void kernel_main(void)
{
    const uint32_t cb_in0 = ft_tensix_compile_time_arg(0);
    const uint32_t cb_in1 = ft_tensix_compile_time_arg(1);
    const uint32_t cb_in2 = ft_tensix_compile_time_arg(2);
    const uint32_t cb_out = ft_tensix_compile_time_arg(3);
    const uint32_t phases = ft_tensix_compile_time_arg(4);
    const uint32_t tile_rows = ft_tensix_runtime_arg(0);
    const uint32_t tile_inner = ft_tensix_runtime_arg(1);
    const uint32_t tile_cols = ft_tensix_runtime_arg(2);
    const uint32_t mode = ft_tensix_runtime_arg(3);
    const uint32_t first_weight_cb = (mode & 1u) ? cb_in1 : cb_in2;

    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[first_weight_cb], ft_cb_unpack_src_format[cb_in0],
        ft_cb_unpack_dst_format[first_weight_cb], ft_cb_unpack_dst_format[cb_in0],
        ft_cb_unpack_tile_face_rows[first_weight_cb], ft_cb_unpack_tile_face_rows[cb_in0],
        ft_cb_unpack_tile_num_faces[first_weight_cb], ft_cb_unpack_tile_num_faces[cb_in0],
        ft_tensix_cb(first_weight_cb)->fifo_page_size, ft_tensix_cb(cb_in0)->fifo_page_size));
    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[first_weight_cb],
                                               ft_cb_unpack_dst_format[cb_in0], FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(ft_cb_pack_src_format[cb_out], ft_cb_pack_dst_format[cb_out],
                                               ft_tensix_cb(cb_out)->fifo_page_size,
                                               ft_cb_pack_tile_num_faces[cb_out], ft_cb_pack_partial_face[cb_out],
                                               0, FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[cb_out], ft_cb_pack_tile_face_rows[cb_out],
                                       ft_cb_pack_tile_num_faces[cb_out], 1));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());
    FT_TENSIX_MATH(ft_tensix_math_matmul_init(0, phases == 2u ? 2u : 1u, phases == 2u ? 2u : 4u, 1));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul_init(
        0, (uint32_t)ft_cb_unpack_tile_num_faces[cb_in1] * ft_cb_unpack_tile_face_rows[cb_in1] *
                   FT_TENSIX_FACE_C_DIM - 1u,
        (uint32_t)ft_cb_unpack_tile_num_faces[cb_in0] * ft_cb_unpack_tile_face_rows[cb_in0] *
            FT_TENSIX_FACE_C_DIM - 1u,
        1u));

    for (uint32_t mt = 0; mt < tile_rows; mt++) {
        for (uint32_t nt = 0; nt < tile_cols; nt++) {
            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
            if (mode & 1u)
                matmul_pass(cb_in0, cb_in1, tile_inner);
            if (mode == 3u) {
                FT_TENSIX_UNPACK(unpack_switch_weight_format(cb_in2));
            }
            if (mode & 2u)
                matmul_pass(cb_in0, cb_in2, tile_inner);
            if (mode == 3u) {
                FT_TENSIX_UNPACK(unpack_switch_weight_format(cb_in1));
            }
            FT_TENSIX_MATH(ft_tensix_math_dest_section_done());
            FT_TENSIX_PACK(ft_tensix_compute_cb_reserve_back(cb_out, 1));
            FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
            FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER, cb_out));
            FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
            FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, 1));
        }
    }
}
