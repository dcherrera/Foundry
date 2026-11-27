/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_gdn_compute.c
 * @brief Compute third (TRISC0/1/2) of the resident gated-delta-net DECODE triad: one decode
 * token of Qwen3.5's linear-attention layer, in one of two phases, with tile-level verbs over
 * the circular buffers resident_gdn_reader.c fills and resident_gdn_writer.c drains.
 *
 * PHASE CONV (kGdnPhaseConv): depthwise causal conv1d + SiLU over the conv_dim stream, then
 * re-emits the rolled window.
 *     out_c = SiLU( qkv_c * w[taps-1] + sum_{t<taps-1} window[t]_c * w[t] )
 *     new window[t] = old window[t+1] (t < taps-2); new window[taps-2] = qkv
 * Pass A emits every conv-out tile, pass B every window tile; the writer splits CB_OUT by
 * position.
 *
 * PHASE DELTA (kGdnPhaseDelta): per v-head, per state tile-row j:
 *     (a) S *= decay  (b) pred = S.k  (c) delta = (v - pred) * beta  (d) S += k (x) delta
 *     (e) y = scale * S.q on the UPDATED S.
 * Under kGdnFlagStateRound the two state packs, (a) and (d), first add 0x1000 to each fp32
 * word as an integer, so every later Tf32 unpack of that state rounds instead of truncating.
 * The state is stored transposed so every per-head scalar is a per-row scalar (a column-0
 * broadcast multiply). CB_OUT per tile-row: the state_tiles updated state tiles, then one y.
 *
 * The eltwise surface has no fused multiply-add (a second multiply into the same DST tile
 * overwrites), so every product is packed to a staging CB and then added. Every loop bound
 * comes from the descriptor or the compile-time args, never from a wait result; a failed wait
 * shortens nothing, so the three threads stay in lockstep. This thread never publishes
 * completion (the writer does) and advances its cursor only for descriptors it kept.
 *
 * Compile-time args:
 *   0 cb_in      CONV qkv stream (the reader is its only producer)
 *   1 cb_state   CONV rolling window / DELTA one state tile-row
 *   2 cb_scaler  DELTA this head's k replicas then q replicas
 *   3 cb_vector  CONV weight taps / DELTA v columns then decay, beta, scale columns
 *   4 cb_pred    DELTA reduce results (b) and (e)
 *   5 cb_delta   CONV running tap sum / DELTA (v - pred) * beta
 *   6 cb_prod    CONV one tap's product / DELTA k (x) delta
 *   7 cb_out
 *   8 wait mode  0 upstream, 1 stoppable, 2 bound-only
 *   9 conv_depth CONV tile columns per block (== reader arg, == FT_TT_OP_GDN_CONV_DEPTH)
 *  10 cb_work    DELTA private working buffer; must differ from cb_in
 *  11 dst_batch_max  DELTA DST batch width before ftg_dst_batch()
 *  12 delta_extras   bit 0 fuse the state emit into (d); bit 1 batch pushes on cb_work and
 *                    cb_prod
 *
 * Build (the whole program; this file is the TRISC third):
 *   RELOCATE_KERNEL_TEXT=1 \
 *     READER_CTARGS=0,1,2,3,4,4096,1,2,8,7,4,1 COMPUTE_CTARGS=0,1,2,3,5,6,7,16,1,2,8,4,1 \
 *     WRITER_CTARGS=16,4096,1,0 CB_TABLE=kernels/cb_tables/gdn_float32.cbtable \
 *     bash foundry/tt/build_tensix_program.sh \
 *       kernels/resident_gdn_reader.c kernels/resident_gdn_compute.c \
 *       kernels/resident_gdn_writer.c programs/gdn
 *
 * @status Clean-room rewrite (Phase 4); byte gate against the shipped gdn images.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
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
#include "ft_tensix_sfpu_tf32_round.h"
#endif
#ifdef FT_TRISC_UNPACK
#include "ft_tensix_compute_sfpu.h"
#include "ft_tensix_compute_unpack.h"
#endif
#ifdef FT_TRISC_PACK
#include "ft_tensix_compute_pack.h"
#endif

#include "resident_gdn_protocol.h"
#include "resident_queue_protocol.h"

/* The copy verb below implements only the non-32-bit unpack path, and pass B picks its copy
 * source at run time, so a 32-bit unpack table must stop the build. */
FTQ_STATIC_ASSERT(FT_CB_ANY_32BIT_UNPACK == 0,
                  "resident_gdn_compute.c supports no 32-bit unpack source");

#define CB_IN ft_tensix_compile_time_arg(0)
#define CB_STATE ft_tensix_compile_time_arg(1)
#define CB_SCALER ft_tensix_compile_time_arg(2)
#define CB_VECTOR ft_tensix_compile_time_arg(3)
#define CB_PRED ft_tensix_compile_time_arg(4)
#define CB_DELTA ft_tensix_compile_time_arg(5)
#define CB_PROD ft_tensix_compile_time_arg(6)
#define CB_OUT ft_tensix_compile_time_arg(7)
#define WAIT_MODE ft_tensix_compile_time_arg(8)
#define CONV_DEPTH ft_tensix_compile_time_arg(9)
#define CB_WORK ft_tensix_compile_time_arg(10)
#define DST_BATCH_MAX ft_tensix_compile_time_arg(11)
#define DELTA_EXTRAS ft_tensix_compile_time_arg(12)

/* delta_extras bits. */
#define EXTRA_FUSE_STATE_EMIT 1u
#define EXTRA_BATCH_PUSHES 2u

/* Single-tile ops use DST register 0. */
#define DST0 0u

/* Control words (Foundry convention): 62 = unpack-side CB-wait cycles over a CONV descriptor,
 * 63 = the part of them spent waiting on reader-fed inputs. */
#define CONTROL_CONV_WAIT 62
#define CONTROL_CONV_INPUT_WAIT 63

/* ELWADD / ELWSUB words: no clear, no accumulate, no SrcB broadcast, address modifier 0,
 * DST row 0 (ELWADD.md / ELWSUB.md). */
#define ADD_WORD FT_BH_ENC_ELWADD(0u, 0u, FT_BH_ELWADD_SRCB_NO_BCAST, 0u, 0u)
#define SUB_WORD FT_BH_ENC_ELWSUB(0u, 0u, FT_BH_ELWADD_SRCB_NO_BCAST, 0u, 0u)

/* ---- Compute verbs ---------------------------------------------------------------------- */

/** Configure unpack, math and pack once: a -> SrcA, b -> SrcB, packing to `out`. */
static FT_BH_INLINE void gdn_hw_startup(uint32_t a, uint32_t b, uint32_t out)
{
    (void)a;
    (void)b;
    (void)out;
    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[a], ft_cb_unpack_src_format[b], ft_cb_unpack_dst_format[a],
        ft_cb_unpack_dst_format[b], ft_cb_unpack_tile_face_rows[a],
        ft_cb_unpack_tile_face_rows[b], ft_cb_unpack_tile_num_faces[a],
        ft_cb_unpack_tile_num_faces[b], ft_tensix_cb(a)->fifo_page_size,
        ft_tensix_cb(b)->fifo_page_size));
    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[a],
                                               ft_cb_unpack_dst_format[b],
                                               FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(
        ft_cb_pack_src_format[out], ft_cb_pack_dst_format[out], ft_tensix_cb(out)->fifo_page_size,
        ft_cb_pack_tile_num_faces[out], ft_cb_pack_partial_face[out], 0, FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[out], ft_cb_pack_tile_face_rows[out],
                                       ft_cb_pack_tile_num_faces[out], 1));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());
}

static FT_BH_INLINE void gdn_mul_tiles_init(uint32_t a)
{
    (void)a;
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_init(FT_CB_MATH_FIDELITY,
                                                          ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[a],
                                              ft_cb_unpack_num_face_rows[a],
                                              ft_cb_unpack_num_face_cols[a]));
}

static FT_BH_INLINE void gdn_mul_tiles(uint32_t a, uint32_t b, uint32_t ia, uint32_t ib,
                                       uint32_t dst)
{
    (void)a;
    (void)b;
    (void)ia;
    (void)ib;
    (void)dst;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, ia, ib));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul(
        dst, (uint32_t)ft_cb_unpack_tile_num_faces[a], ft_cb_unpack_tile_face_rows[a]));
}

static FT_BH_INLINE void gdn_addsub_tiles_init(uint32_t op_word, uint32_t a)
{
    (void)op_word;
    (void)a;
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub_init(
        op_word, (uint32_t)ft_cb_unpack_tile_num_faces[a],
        (uint32_t)ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[a],
                                              ft_cb_unpack_num_face_rows[a],
                                              ft_cb_unpack_num_face_cols[a]));
}

/** a + b or a - b, per the op word the last init was given. */
static FT_BH_INLINE void gdn_addsub_tiles(uint32_t a, uint32_t b, uint32_t ia, uint32_t ib,
                                          uint32_t dst)
{
    (void)a;
    (void)b;
    (void)ia;
    (void)ib;
    (void)dst;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, ia, ib));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub(dst));
}

static FT_BH_INLINE void gdn_mul_bcast_cols_init(uint32_t a)
{
    (void)a;
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_broadcast_col_init(
        FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_broadcast_col_init(ft_cb_unpack_tile_face_rows[a],
                                                            ft_cb_unpack_num_face_rows[a],
                                                            ft_cb_unpack_num_face_cols[a]));
}

/** Column 0 of b scales each row of a. */
static FT_BH_INLINE void gdn_mul_bcast_cols(uint32_t a, uint32_t b, uint32_t ia, uint32_t ib,
                                            uint32_t dst)
{
    (void)a;
    (void)b;
    (void)ia;
    (void)ib;
    (void)dst;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, ia, ib));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_broadcast_col(
        dst, FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_num_face_rows[a],
        (uint32_t)ft_cb_unpack_num_face_cols[a], (uint32_t)ft_cb_unpack_tile_face_rows[a]));
}

static FT_BH_INLINE void gdn_copy_tile_init(uint32_t cb)
{
    (void)cb;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a_init(
        ft_cb_unpack_tile_face_rows[cb],
        (uint32_t)ft_cb_unpack_num_face_rows[cb] * ft_cb_unpack_num_face_cols[cb]));
    FT_TENSIX_MATH(ft_tensix_math_datacopy_init((uint32_t)ft_cb_unpack_num_face_rows[cb] *
                                                ft_cb_unpack_num_face_cols[cb]));
}

static FT_BH_INLINE void gdn_copy_tile(uint32_t cb, uint32_t index, uint32_t dst)
{
    (void)cb;
    (void)index;
    (void)dst;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a(cb, index));
    FT_TENSIX_MATH(ft_tensix_math_datacopy(dst));
}

/* ---- Waits (out of line: spin loops with no Tensix instruction) ------------------------- */

static __attribute__((noinline)) bool gdn_wait_front(ftq_context *ctx, uint32_t cb,
                                                     uint32_t pages)
{
    return ftq_cb_wait_front_stoppable(ctx, (int32_t)cb, (int32_t)pages, WAIT_MODE);
}

static __attribute__((noinline)) bool gdn_reserve_back(ftq_context *ctx, uint32_t cb,
                                                       uint32_t pages)
{
    return ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, (int32_t)pages, WAIT_MODE);
}

/* ---- Commit helpers (the pack side of one DST section) ---------------------------------- */

/** DST tiles [0, tiles) -> `tiles` pages at the back of `cb`. */
static void commit_dst_block_to(ftq_context *ctx, uint32_t cb, uint32_t tiles,
                                bool block_aligned)
{
    FT_TENSIX_MATH(ft_tensix_math_dest_section_done());
    if (block_aligned && (DELTA_EXTRAS & EXTRA_BATCH_PUSHES) != 0u && tiles > 1u) {
        (void)gdn_reserve_back(ctx, cb, tiles);
        FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
        for (uint32_t index = 0; index < tiles; index++)
            FT_TENSIX_PACK(ft_tensix_pack_tile(index, cb));
        FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
        FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, tiles));
        return;
    }
    (void)gdn_reserve_back(ctx, cb, 1u);
    FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
    FT_TENSIX_PACK(ft_tensix_pack_tile(0u, cb));
    for (uint32_t index = 1; index < tiles; index++) {
        FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, 1));
        (void)gdn_reserve_back(ctx, cb, 1u);
        FT_TENSIX_PACK(ft_tensix_pack_tile(index, cb));
    }
    FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, 1));
}

/**
 * DST tiles [0, tiles) replace the FRONT `tiles` pages of `cb` (pop front, push back), each
 * optionally also packed to `also` and to the carry CB. The pop must precede the reserve: the
 * buffer is full, so a reserve first could never be satisfied. `also` is cb_out and is always
 * pushed one page at a time.
 */
static void commit_dst_block_in_place_and_out(ftq_context *ctx, uint32_t cb, uint32_t tiles,
                                              uint32_t also, bool have_also, bool carry)
{
    const bool batch_pushes = (DELTA_EXTRAS & EXTRA_BATCH_PUSHES) != 0u && tiles > 1u;

    (void)also;
    FT_TENSIX_MATH(ft_tensix_math_dest_section_done());
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb, tiles));
    (void)gdn_reserve_back(ctx, cb, batch_pushes ? tiles : 1u);
    FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
    for (uint32_t index = 0; index < tiles; index++) {
        if (index != 0u && !batch_pushes) {
            FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, 1));
            (void)gdn_reserve_back(ctx, cb, 1u);
        }
        FT_TENSIX_PACK(ft_tensix_pack_tile(index, cb));
        if (have_also) {
            (void)gdn_reserve_back(ctx, also, 1u);
            FT_TENSIX_PACK(ft_tensix_pack_tile(index, also));
            FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(also, 1));
        }
        if (carry) {
            (void)gdn_reserve_back(ctx, (uint32_t)kGdnCbCarry, 1u);
            FT_TENSIX_PACK(ft_tensix_pack_tile(index, (uint32_t)kGdnCbCarry));
            FT_TENSIX_PACK(ft_tensix_compute_cb_push_back((uint32_t)kGdnCbCarry, 1));
        }
    }
    FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, batch_pushes ? tiles : 1u));
}

static void commit_dst_block_in_place(ftq_context *ctx, uint32_t cb, uint32_t tiles)
{
    commit_dst_block_in_place_and_out(ctx, cb, tiles, 0u, false, false);
}

static void commit_dst_to(ftq_context *ctx, uint32_t cb)
{
    commit_dst_block_to(ctx, cb, 1u, false);
}

static void commit_dst_in_place(ftq_context *ctx, uint32_t cb)
{
    commit_dst_block_in_place(ctx, cb, 1u);
}

/**
 * Row SUM reduce of `tiles` tiles of `source` against cb_scaler tiles scaler_first + i, into
 * column 0 of one tile pushed to `destination`. The mask clear after the pack is the only
 * restore of the packer edge mask, which outlives the program.
 */
static void reduce_row_sum(ftq_context *ctx, uint32_t source, uint32_t scaler_first,
                           uint32_t tiles, uint32_t destination)
{
    (void)source;
    (void)scaler_first;
    FT_TENSIX_UNPACK(
        ft_tensix_unpack_ab_reduce_row_init((uint32_t)ft_cb_unpack_tile_num_faces[source]));
    FT_TENSIX_MATH(ft_tensix_math_reduce_row_init(FT_CB_MATH_FIDELITY != 0 ? 1u : 0u,
                                                  FT_CB_MATH_FIDELITY));
    FT_TENSIX_PACK(
        ft_tensix_pack_reduce_row_mask_config((uint32_t)ft_cb_pack_tile_face_rows[destination]));
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    for (uint32_t index = 0; index < tiles; index++) {
        FT_TENSIX_MATH(ft_tensix_math_reduce_row(0));
        FT_TENSIX_UNPACK(
            ft_tensix_unpack_ab_reduce_row(source, CB_SCALER, index, scaler_first + index));
    }
    commit_dst_to(ctx, destination);
    FT_TENSIX_MATH(ft_tensix_math_reduce_uninit());
    FT_TENSIX_PACK(ft_tensix_pack_reduce_mask_clear());
}

/* ---- PHASE DELTA ------------------------------------------------------------------------- */

/** kGdnFlagStateRound: DST tiles [0, tiles) += 0x1000 as integers, before they are packed. */
static void delta_round_bias_dst_block(uint32_t tiles)
{
    FT_TENSIX_MATH(ft_tensix_math_tf32_round_bias_tile_init());
    for (uint32_t index = 0; index < tiles; index++)
        FT_TENSIX_MATH(ft_tensix_math_tf32_round_bias_tile(index));
}

/** (a) S *= decay, into cb_work. `source` is indexed absolutely and popped once at the end. */
static void delta_decay_tile_row(ftq_context *ctx, uint32_t source, uint32_t tiles,
                                 uint32_t decay_page, uint32_t batch, bool round)
{
    for (uint32_t first = 0; first < tiles; first += batch) {
        gdn_mul_bcast_cols_init(source);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        for (uint32_t index = 0; index < batch; index++)
            gdn_mul_bcast_cols(source, CB_VECTOR, first + index, decay_page, index);
        if (round)
            delta_round_bias_dst_block(batch);
        commit_dst_block_to(ctx, CB_WORK, batch, true);
    }
    (void)gdn_wait_front(ctx, CB_WORK, tiles);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(source, tiles));
}

/** (c) delta = (v - pred) * beta, left as the one page of cb_delta. */
static void delta_residual(ftq_context *ctx, uint32_t v_index, uint32_t beta_page)
{
    gdn_addsub_tiles_init(SUB_WORD, CB_VECTOR);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    gdn_addsub_tiles(CB_VECTOR, CB_PRED, v_index, 0u, DST0);
    commit_dst_to(ctx, CB_DELTA);
    (void)gdn_wait_front(ctx, CB_DELTA, 1u);

    gdn_mul_bcast_cols_init(CB_DELTA);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    gdn_mul_bcast_cols(CB_DELTA, CB_VECTOR, 0u, beta_page, DST0);
    commit_dst_in_place(ctx, CB_DELTA);
    (void)gdn_wait_front(ctx, CB_DELTA, 1u);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_PRED, 1));
}

/**
 * (d) S += k (x) delta. cb_work is indexed from zero, not from `first`: each block's rotation
 * moves its front by `batch`, so the tiles to update are always at the front.
 */
static void delta_update_tile_row(ftq_context *ctx, uint32_t tiles, uint32_t batch, bool carry,
                                  bool round)
{
    for (uint32_t first = 0; first < tiles; first += batch) {
        gdn_mul_bcast_cols_init(CB_SCALER);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        for (uint32_t index = 0; index < batch; index++)
            gdn_mul_bcast_cols(CB_SCALER, CB_DELTA, first + index, 0u, index);
        commit_dst_block_to(ctx, CB_PROD, batch, true);
        (void)gdn_wait_front(ctx, CB_PROD, batch);

        gdn_addsub_tiles_init(ADD_WORD, CB_WORK);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        for (uint32_t index = 0; index < batch; index++)
            gdn_addsub_tiles(CB_WORK, CB_PROD, index, index, index);
        if (round)
            delta_round_bias_dst_block(batch);
        commit_dst_block_in_place_and_out(ctx, CB_WORK, batch, CB_OUT,
                                          (DELTA_EXTRAS & EXTRA_FUSE_STATE_EMIT) != 0u, carry);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_PROD, batch));
    }
    (void)gdn_wait_front(ctx, CB_WORK, tiles);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_DELTA, 1));
}

/** Emit the updated state row (unless fused into (d)), then y = scale * (S.q). */
static void delta_emit_tile_row(ftq_context *ctx, uint32_t tiles, uint32_t scale_page,
                                uint32_t batch)
{
    if ((DELTA_EXTRAS & EXTRA_FUSE_STATE_EMIT) == 0u) {
        for (uint32_t first = 0; first < tiles; first += batch) {
            gdn_copy_tile_init(CB_WORK);
            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
            for (uint32_t index = 0; index < batch; index++)
                gdn_copy_tile(CB_WORK, first + index, index);
            commit_dst_block_to(ctx, CB_OUT, batch, false);
        }
    }
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_WORK, tiles));

    gdn_mul_bcast_cols_init(CB_PRED);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    gdn_mul_bcast_cols(CB_PRED, CB_VECTOR, 0u, scale_page, DST0);
    commit_dst_to(ctx, CB_OUT);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_PRED, 1));
}

/** Steps (a)-(e) for one state tile-row. */
static void delta_tile_row(ftq_context *ctx, uint32_t source, uint32_t tiles, uint32_t tile_row,
                           uint32_t batch, bool carry, bool round)
{
    (void)gdn_wait_front(ctx, source, tiles);
    delta_decay_tile_row(ctx, source, tiles, tiles + (uint32_t)kGdnVectorDecay, batch, round);
    reduce_row_sum(ctx, CB_WORK, 0u, tiles, CB_PRED);
    (void)gdn_wait_front(ctx, CB_PRED, 1u);
    delta_residual(ctx, tile_row, tiles + (uint32_t)kGdnVectorBeta);
    delta_update_tile_row(ctx, tiles, batch, carry, round);
    reduce_row_sum(ctx, CB_WORK, tiles, tiles, CB_PRED);
    (void)gdn_wait_front(ctx, CB_PRED, 1u);
    delta_emit_tile_row(ctx, tiles, tiles + (uint32_t)kGdnVectorScale, batch);
}

/**
 * One head for one row. cb_vector is popped one page at a time: `constants` does not divide
 * its depth, and a pop chunk that does not divide it steps over the wrap limit.
 */
static void delta_one_head_row(ftq_context *ctx, uint32_t source, uint32_t tiles, uint32_t batch,
                               bool carry, bool round)
{
    const uint32_t constants = tiles + (uint32_t)kGdnVectorConstants;

    (void)gdn_wait_front(ctx, CB_SCALER, 2u * tiles);
    (void)gdn_wait_front(ctx, CB_VECTOR, constants);
    for (uint32_t index = 0; index < tiles; index++)
        delta_tile_row(ctx, source, tiles, index, batch, carry, round);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_SCALER, 2u * tiles));
    for (uint32_t index = 0; index < constants; index++)
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_VECTOR, 1));
}

/** Row 0 decays the reader's state; later verify rows decay the carry the previous row packed. */
static void delta_one_head(ftq_context *ctx, uint32_t tiles, uint32_t batch, uint32_t rows,
                           bool round)
{
    for (uint32_t row = 0; row < rows; row++)
        delta_one_head_row(ctx, row == 0u ? CB_STATE : (uint32_t)kGdnCbCarry, tiles, batch,
                           row + 1u < rows, round);
}

/* ---- PHASE CONV -------------------------------------------------------------------------- */

/** Tile columns per block: CONV_DEPTH clamped to a divisor of tile_cols; 0 = refuse. */
static uint32_t conv_block_depth(uint32_t tile_cols, uint32_t taps)
{
    uint32_t depth = CONV_DEPTH;

    if (taps < 2u || tile_cols == 0u || depth == 0u)
        return 0u;
    if (depth > tile_cols)
        depth = tile_cols;
    while (depth > 0u && (tile_cols % depth) != 0u)
        depth--;
    return depth;
}

/** One output column: the tap sum through cb_delta, SiLU on the last tap, out to cb_out. */
static void conv_one_column(ftq_context *ctx, uint32_t taps, uint32_t depth, uint32_t column)
{
    const uint32_t dst_register = DST0;

    (void)dst_register;
    gdn_mul_tiles_init(CB_IN);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    gdn_mul_tiles(CB_IN, CB_VECTOR, column, (taps - 1u) * depth + column, dst_register);
    commit_dst_to(ctx, CB_DELTA);
    (void)gdn_wait_front(ctx, CB_DELTA, 1u);

    for (uint32_t tap = 0; tap + 1u < taps; tap++) {
        gdn_mul_tiles_init(CB_STATE);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        gdn_mul_tiles(CB_STATE, CB_VECTOR, tap * depth + column, tap * depth + column,
                      dst_register);
        commit_dst_to(ctx, CB_PROD);
        (void)gdn_wait_front(ctx, CB_PROD, 1u);

        gdn_addsub_tiles_init(ADD_WORD, CB_DELTA);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        gdn_addsub_tiles(CB_DELTA, CB_PROD, 0u, 0u, dst_register);
        if (tap + 2u < taps) {
            commit_dst_in_place(ctx, CB_DELTA);
            (void)gdn_wait_front(ctx, CB_DELTA, 1u);
        } else {
            FT_TENSIX_MATH(ft_tensix_math_silu_tile_init());
            FT_TENSIX_MATH(ft_tensix_math_silu_tile(dst_register));
            commit_dst_to(ctx, CB_OUT);
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_DELTA, 1));
        }
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_PROD, 1));
    }
}

/** A wait on a reader-fed buffer, its unpack-side cycles also added to control word 63. */
static void gdn_conv_input_wait(ftq_context *ctx, uint32_t cb, uint32_t pages)
{
    const uint32_t waited = ctx->status ? ctx->status[kStatusPayload] : 0u;

    (void)waited;
    (void)gdn_wait_front(ctx, cb, pages);
    FT_TENSIX_UNPACK(if (ctx->status) ctx->control[CONTROL_CONV_INPUT_WAIT] +=
                     ctx->status[kStatusPayload] - waited);
}

/**
 * Pass A: every conv-out tile. cb_vector holds the block's weights tap-major then column
 * (tile tap * depth + column); cb_state the window planes likewise. cb_state is popped per tap
 * because taps - 1 does not divide its depth.
 */
static void conv_convolve(ftq_context *ctx, const ftg_shape *shape, uint32_t depth)
{
    const uint32_t taps = shape->conv_taps;

    for (uint32_t first_col = 0; first_col < shape->col_count; first_col += depth) {
        gdn_conv_input_wait(ctx, CB_IN, depth);
        gdn_conv_input_wait(ctx, CB_VECTOR, depth * taps);
        gdn_conv_input_wait(ctx, CB_STATE, depth * (taps - 1u));
        for (uint32_t column = 0; column < depth; column++)
            conv_one_column(ctx, taps, depth, column);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_IN, depth));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_VECTOR, depth * taps));
        for (uint32_t tap = 0; tap + 1u < taps; tap++)
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_STATE, depth));
    }
}

/** Pass B: the rolled window, tap-major; the last plane is the new qkv. */
static void conv_roll_window(ftq_context *ctx, const ftg_shape *shape, uint32_t depth)
{
    const uint32_t taps = shape->conv_taps;

    for (uint32_t tap = 0; tap + 1u < taps; tap++) {
        const uint32_t source = (tap + 2u < taps) ? CB_STATE : CB_IN;

        for (uint32_t first_col = 0; first_col < shape->col_count; first_col += depth) {
            gdn_conv_input_wait(ctx, source, depth);
            for (uint32_t column = 0; column < depth; column++) {
                gdn_copy_tile_init(source);
                FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
                gdn_copy_tile(source, column, DST0);
                commit_dst_to(ctx, CB_OUT);
            }
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(source, depth));
        }
    }
}

/* ---- Descriptor dispatch ----------------------------------------------------------------- */

/**
 * Return the indexed buffers to their base at the top of a descriptor: tile addressing is
 * base + index * page with no wrap, and a descriptor's traffic is not a whole number of
 * buffers. Safe without reservation, since everything is drained by then. cb_pred, cb_delta
 * and cb_out are only ever accessed at tile 0; cb_in's write side is not this thread's.
 */
static void gdn_rebase_indexed_buffers(void)
{
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_IN));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_STATE));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_SCALER));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_VECTOR));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_WORK));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_PROD));
    FT_TENSIX_PACK(ft_tensix_cb_rebase_write_pointer(CB_WORK));
    FT_TENSIX_PACK(ft_tensix_cb_rebase_write_pointer(CB_PROD));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer((uint32_t)kGdnCbCarry));
    FT_TENSIX_PACK(ft_tensix_cb_rebase_write_pointer((uint32_t)kGdnCbCarry));
}

/**
 * One kOpGdn descriptor. Refusals happen before any CB traffic, on the same words the reader
 * and writer test, so nobody waits for pages another processor decided not to send. The head
 * loop never names a head: the reader feeds them in order.
 */
static void compute_one_gdn(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const ftg_shape shape = ftg_decode(parameters);

    if (!ftg_shape_is_supported(&shape)) {
        ftq_abort(ctx, kErrBadParameters);
        return;
    }
    gdn_rebase_indexed_buffers();

    if (shape.phase == (uint32_t)kGdnPhaseDelta) {
        if (shape.tile_count < shape.head_count * ftg_state_tiles_per_head(&shape)) {
            ftq_abort(ctx, kErrBadParameters);
            return;
        }
        const uint32_t batch = ftg_dst_batch(shape.state_tiles, DST_BATCH_MAX);
        const bool round = (parameters[kGdnFlags] & (uint32_t)kGdnFlagStateRound) != 0u;

        for (uint32_t head = 0; head < shape.head_count; head++)
            delta_one_head(ctx, shape.state_tiles, batch, shape.row_count, round);
        return;
    }

    const uint32_t depth = conv_block_depth(shape.col_count, shape.conv_taps);

    if (depth == 0u || !ftg_conv_depth_agrees(shape.declared_conv_depth, (uint32_t)CONV_DEPTH) ||
        shape.tile_count < shape.conv_tile_cols * (shape.conv_taps - 1u)) {
        ftq_abort(ctx, kErrBadParameters);
        return;
    }
    const uint32_t waited = ctx->status ? ctx->status[kStatusPayload] : 0u;

    (void)waited;
    for (uint32_t row = 0; row < shape.row_count; row++) {
        if (row > 0u)
            gdn_rebase_indexed_buffers();
        conv_convolve(ctx, &shape, depth);
        conv_roll_window(ctx, &shape, depth);
    }
    FT_TENSIX_UNPACK(if (ctx->status) ctx->control[CONTROL_CONV_WAIT] +=
                     ctx->status[kStatusPayload] - waited);
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

    gdn_hw_startup(CB_IN, CB_VECTOR, CB_OUT);
    ctx.cursor = ctx.control[kFieldCompletedCount];
    FT_TENSIX_UNPACK(ctx.control[CONTROL_CONV_WAIT] = 0u; ctx.control[CONTROL_CONV_INPUT_WAIT] = 0u);

    while (running) {
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);

        volatile FTQ_L1_PTR uint32_t *descriptor = ftq_descriptor(&ctx);
        const uint32_t opcode = descriptor[kDescOpcode];

        ftq_status_opcode(&ctx, opcode);
        if (opcode == kOpGdn) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            compute_one_gdn(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
            if (ctx.aborting)
                running = false;
        } else if (opcode == kOpStop) {
            running = false;
        }
        /* Other opcodes belong to the reader; nothing to do here. */

        if (!ctx.aborting)
            ctx.cursor++;
    }

    ftq_status_set(&ctx, kStateExit);
}
