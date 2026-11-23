/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_rope_compute.c
 * @brief Compute third of the resident ROPE triad: rotary position embedding with no element
 *   indexing. The reader delivers four aligned tile streams -- activation (in), partner (rot),
 *   cos, and a sin table with the sign already folded in -- and per output tile:
 *
 *     rotated column:      out = in * cos + rot * sin
 *     pass-through column: out = in
 *
 *   The FPU multiply cannot accumulate into DST here (a second multiply into the same DST tile
 *   overwrites it), so each product is packed to its own one-page scratch CB and the two are
 *   then added: three DST sections per rotated tile.
 *
 *   Compiled three times (FT_TRISC_UNPACK / MATH / PACK); every hardware statement sits in one
 *   FT_TENSIX_<thread>(...) wrapper. Loop bounds come only from the descriptor, and after an
 *   abort every pop / acquire / commit / push still runs so the threads stay balanced.
 *
 * Compile-time args: 0 cb_in, 1 cb_rot, 2 cb_cos, 3 cb_sin, 4 cb_prod_cos, 5 cb_prod_sin,
 *   6 cb_out, 7 wait_mode (0 upstream, 1 stoppable, 2 bound-only).
 *
 * Build:
 *   CTARGS=0,1,2,3,4,5,16,1 CB_TABLE=kernels/cb_tables/rope_float16b.cbtable \
 *       bash foundry/tt/build_tensix_kernel.sh \
 *           foundry/tt/kernels/resident_rope_compute.c out/ttk_rope
 *
 * @status Clean-room rewrite (plan Phase 4.7).
 * @issues None known.
 * @todo None
 */
#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_cb_formats.h"
#include "ft_tensix_compute.h"
#include "ft_tensix_compute_cb.h"
#include "ft_tensix_kernel_args.h"
#include "ft_tensix_compute_sfpu.h"
#ifdef FT_TRISC_MATH
#include "ft_tensix_compute_math.h"
#endif
#ifdef FT_TRISC_UNPACK
#include "ft_tensix_compute_unpack.h"
#endif
#ifdef FT_TRISC_PACK
#include "ft_tensix_compute_pack.h"
#endif

#include "resident_queue_protocol.h"
#include "resident_rope_protocol.h"

#define cb_in ft_tensix_compile_time_arg(0)
#define cb_rot ft_tensix_compile_time_arg(1)
#define cb_cos ft_tensix_compile_time_arg(2)
#define cb_sin ft_tensix_compile_time_arg(3)
#define cb_prod_cos ft_tensix_compile_time_arg(4)
#define cb_prod_sin ft_tensix_compile_time_arg(5)
#define cb_out ft_tensix_compile_time_arg(6)
#define wait_mode ft_tensix_compile_time_arg(7)

enum { dst_register = 0 };

/* ===== Compute verbs (thread order inside each is fixed) ================================ */

/** One-time unit setup: A -> SrcA, B -> SrcB, packer configured for `out`. */
static FT_BH_INLINE void rope_hw_startup(uint32_t a, uint32_t b, uint32_t out)
{
    (void)a;
    (void)b;
    (void)out;
    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[a], ft_cb_unpack_src_format[b], ft_cb_unpack_dst_format[a],
        ft_cb_unpack_dst_format[b], ft_cb_unpack_tile_face_rows[a], ft_cb_unpack_tile_face_rows[b],
        ft_cb_unpack_tile_num_faces[a], ft_cb_unpack_tile_num_faces[b],
        ft_tensix_cb(a)->fifo_page_size, ft_tensix_cb(b)->fifo_page_size));
    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[a],
                                               ft_cb_unpack_dst_format[b], FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(
        ft_cb_pack_src_format[out], ft_cb_pack_dst_format[out], ft_tensix_cb(out)->fifo_page_size,
        ft_cb_pack_tile_num_faces[out], ft_cb_pack_partial_face[out], 0 /* relu */,
        FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[out], ft_cb_pack_tile_face_rows[out],
                                       ft_cb_pack_tile_num_faces[out], 1));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());
}

/** Multiply init: MATH first, then UNPACK; geometry from `a` only. */
static FT_BH_INLINE void rope_mul_init(uint32_t a, uint32_t b)
{
    (void)a;
    (void)b;
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_init(FT_CB_MATH_FIDELITY,
                                                          ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[a],
                                              ft_cb_unpack_num_face_rows[a],
                                              ft_cb_unpack_num_face_cols[a]));
}

/** Multiply the front tiles of `a` and `b` into DST. */
static FT_BH_INLINE void rope_mul_apply(uint32_t a, uint32_t b)
{
    (void)a;
    (void)b;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, 0, 0));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul(
        dst_register, (uint32_t)ft_cb_unpack_num_face_rows[a] * ft_cb_unpack_num_face_cols[a],
        ft_cb_unpack_tile_face_rows[a]));
}

/** Add init (lowest fidelity): MATH first, then UNPACK; geometry from `a` only. */
static FT_BH_INLINE void rope_add_init(uint32_t a, uint32_t b)
{
    (void)a;
    (void)b;
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub_init(
        FT_BH_ENC_ELWADD(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_0, 0),
        (uint32_t)ft_cb_unpack_tile_num_faces[a], (uint32_t)ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[a],
                                              ft_cb_unpack_num_face_rows[a],
                                              ft_cb_unpack_num_face_cols[a]));
}

/** Add the front tiles of `a` and `b` into DST. */
static FT_BH_INLINE void rope_add_apply(uint32_t a, uint32_t b)
{
    (void)a;
    (void)b;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, 0, 0));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub(dst_register));
}

/** Copy init: UNPACK first, then MATH. */
static FT_BH_INLINE void rope_copy_init(uint32_t cb)
{
    (void)cb;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a_init(
        ft_cb_unpack_tile_face_rows[cb],
        (uint32_t)ft_cb_unpack_num_face_rows[cb] * ft_cb_unpack_num_face_cols[cb]));
    FT_TENSIX_MATH(ft_tensix_math_datacopy_init((uint32_t)ft_cb_unpack_num_face_rows[cb] *
                                                ft_cb_unpack_num_face_cols[cb]));
}

/** Copy one tile of `cb` into DST. */
static FT_BH_INLINE void rope_copy_apply(uint32_t cb)
{
    (void)cb;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a(cb, 0));
    FT_TENSIX_MATH(ft_tensix_math_datacopy(dst_register));
}

/* ===== Steps ============================================================================ */

/** Hand DST to PACK and pack it as one page of `cb`. */
static void commit_dst_to(ftq_context *ctx, uint32_t cb)
{
    FT_TENSIX_MATH(ft_tensix_math_dest_section_done());
    (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, 1, wait_mode);
    FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
    FT_TENSIX_PACK(ft_tensix_pack_tile(dst_register, cb));
    FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, 1));
}

/** destination <- a * b, one tile. */
static void multiply_into(ftq_context *ctx, uint32_t a, uint32_t b, uint32_t destination)
{
    rope_mul_init(a, b);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    rope_mul_apply(a, b);
    commit_dst_to(ctx, destination);
}

/** out <- in * cos + rot * sin, one tile, through the two product CBs. */
static void rotate_one_tile(ftq_context *ctx)
{
    multiply_into(ctx, cb_in, cb_cos, cb_prod_cos);
    multiply_into(ctx, cb_rot, cb_sin, cb_prod_sin);
    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_prod_cos, 1, wait_mode);
    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_prod_sin, 1, wait_mode);
    rope_add_init(cb_prod_cos, cb_prod_sin);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    rope_add_apply(cb_prod_cos, cb_prod_sin);
    commit_dst_to(ctx, cb_out);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_prod_cos, 1));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_prod_sin, 1));
}

/** out <- in, one tile. */
static void copy_one_tile(ftq_context *ctx)
{
    rope_copy_init(cb_in);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    rope_copy_apply(cb_in);
    commit_dst_to(ctx, cb_out);
}

/** Run one ROPE descriptor; an unsupported shape is refused before any CB traffic. */
static void compute_one_rope(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t tile_count = parameters[kRopeTileCount];
    const uint32_t head_tiles = parameters[kRopeHeadTiles];
    const uint32_t rot_tiles = parameters[kRopeRotTiles];
    uint32_t tile;

    if (!ftr_shape_is_supported(head_tiles, rot_tiles) || tile_count == 0u ||
        tile_count % head_tiles != 0u) {
        ftq_abort(ctx, kErrBadParameters);
        return;
    }

    for (tile = 0; tile < tile_count; tile++) {
        const uint32_t column = tile % head_tiles;
        const bool rotated = (column < rot_tiles);

        (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in, 1, wait_mode);
        if (rotated) {
            (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_rot, 1, wait_mode);
            (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_cos, 1, wait_mode);
            (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_sin, 1, wait_mode);
            rotate_one_tile(ctx);
        } else {
            copy_one_tile(ctx);
        }
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in, 1));
        if (rotated) {
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_rot, 1));
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_cos, 1));
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_sin, 1));
        }
    }
}

void kernel_main(void)
{
    ftq_context ctx;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);
    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    /* Once: cb_in and cb_cos are the A/B pair the unpacker is configured for. */
    rope_hw_startup(cb_in, cb_cos, cb_out);

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
        if (opcode == kOpRope) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            compute_one_rope(&ctx, ftq_parameters(descriptor));
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
