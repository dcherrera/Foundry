/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bfp6_sfpu_floor_compute.c
 * @brief The FLOOR of route 2 of qwen36 spec §4.1b (SFPU decode in the compute kernel): every
 *   piece of that route except the decode arithmetic itself. Per tile: unpack the BFP6_B
 *   tile's high plane — which in the planes layout IS a valid BFP4_B tile (sign + 3 high
 *   magnitude bits, the same exponents) — to SrcA, copy it to Dst, pack it as BFP8_B into the
 *   output CB the matmul would read. The real route adds a second unpack (the low plane) and
 *   the SFPU instructions that merge the two, so its cost per tile can only be HIGHER than
 *   this. If the floor already busts the gate, the route does.
 *
 * Gate: the packed BFP8_B tile decodes to exactly the BFP4_B values (<= 4 significant bits
 * survive any packer rounding), checked by the bench.
 *
 * Compile-time args: 0 cb_in (BFP4_B)  1 cb_out (BFP8_B)
 * Runtime args: 0 tiles
 *
 * @status New 2026-10-28 (qwen36 Phase 1.4 price bench).
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_cb_formats.h"
#include "ft_tensix_compute.h"
#include "ft_tensix_compute_cb.h"
#include "ft_tensix_compute_sfpu.h"
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

void kernel_main(void)
{
    const uint32_t cb_in = ft_tensix_compile_time_arg(0);
    const uint32_t cb_out = ft_tensix_compile_time_arg(1);
    const uint32_t tiles = ft_tensix_runtime_arg(0);

    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[cb_in], ft_cb_unpack_src_format[cb_in], ft_cb_unpack_dst_format[cb_in],
        ft_cb_unpack_dst_format[cb_in], ft_cb_unpack_tile_face_rows[cb_in], ft_cb_unpack_tile_face_rows[cb_in],
        ft_cb_unpack_tile_num_faces[cb_in], ft_cb_unpack_tile_num_faces[cb_in], ft_tensix_cb(cb_in)->fifo_page_size,
        ft_tensix_cb(cb_in)->fifo_page_size));
    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[cb_in], ft_cb_unpack_dst_format[cb_in],
                                               FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(ft_cb_pack_src_format[cb_out], ft_cb_pack_dst_format[cb_out],
                                               ft_tensix_cb(cb_out)->fifo_page_size,
                                               ft_cb_pack_tile_num_faces[cb_out], ft_cb_pack_partial_face[cb_out],
                                               0, FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[cb_out], ft_cb_pack_tile_face_rows[cb_out],
                                       ft_cb_pack_tile_num_faces[cb_out], 1));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());
    FT_TENSIX_UNPACK(ft_tensix_unpack_a_init(ft_cb_unpack_tile_face_rows[cb_in],
                                             (uint32_t)ft_cb_unpack_num_face_rows[cb_in] *
                                                 ft_cb_unpack_num_face_cols[cb_in]));
    FT_TENSIX_MATH(ft_tensix_math_datacopy_init((uint32_t)ft_cb_unpack_num_face_rows[cb_in] *
                                                ft_cb_unpack_num_face_cols[cb_in]));

    for (uint32_t t = 0; t < tiles; t++) {
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_in, 1));
        FT_TENSIX_UNPACK(ft_tensix_unpack_a(cb_in, 0));
        FT_TENSIX_MATH(ft_tensix_math_datacopy(0));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in, 1));
        FT_TENSIX_MATH(ft_tensix_math_dest_section_done());
        FT_TENSIX_PACK(ft_tensix_compute_cb_reserve_back(cb_out, 1));
        FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
        FT_TENSIX_PACK(ft_tensix_pack_tile(0, cb_out));
        FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
        FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, 1));
    }
}
