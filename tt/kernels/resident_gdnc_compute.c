/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_gdnc_compute.c
 * @brief Compute third (TRISC0/1/2) of the resident CHUNKWISE gated-delta-net triad: a chunk
 * of C = 32 positions (one tile's rows) at once, in one of two phases.
 *
 * PHASE CONV (kGdncPhaseConv): depthwise conv1d + SiLU over the chunk as a banded matmul. Per
 * tile column c, with X_c the chunk input, Win_c the assembled window tile, W_tau the row-
 * replicated weight tiles and the 2*taps constant 0/1 band tiles (S_tau, C_tau) the host builds:
 *     out_c = SiLU( sum_tau [ S_tau . (X_c (x) W_tau) + C_tau . (Win_c (x) W_tau) ] )
 * (x) is an eltwise multiply packed to CB_PROD (there is no eltwise FMA); . is a matmul
 * accumulating in ONE DST section. The raw X_c tile is then copied out behind conv_out (the
 * writer takes the window roll from it). CB_OUT per column: conv_out, then the X_c copy.
 *
 * PHASE DELTA (kGdncPhaseDelta): the chunkwise delta rule of each v-head as a UT transform
 * (derived in resident_gdnc_protocol.h): two Gram matrices, a causal decay mask, an exact
 * 4-step Newton inverse of the unit-lower-triangular A, then U, Y and the carried state
 * N_C = G N_0 + U^T K_bar, the [S,S] state stored transposed. CB_OUT per head: T y tiles, then
 * T*T state tiles (T = state_tiles = S/32). Under kGdncDeltaFlagStateRound each state tile
 * first gets 0x1000 added to its fp32 words as integers, so the next Tf32 unpack of the state
 * (the next chunk here, or decode `gdn`) rounds instead of truncating.
 *
 * Matmul dominates, so the hardware is configured once at boot in REVERSED source order
 * (in0 -> SrcB, in1 -> SrcA) and the eltwise/broadcast verbs run on that same configuration;
 * that is legal because every CB carries one format, so no data-format reconfig is issued.
 * Each verb family's init is re-issued before each group of that family. Loop bounds come from
 * the descriptor, an abort drains, and an abandoned descriptor is not completed; completion is
 * the writer's.
 *
 * Compile-time args:
 *   0 cb_in      CONV input tile column / DELTA one head's q|k|v
 *   1 cb_state   CONV assembled window / DELTA one head's [S,S] state
 *   2 cb_band    CONV the 2*taps band tiles / DELTA the four constants
 *   3 cb_vector  CONV this block's taps weights / DELTA the five gate columns
 *   4 cb_prod    CONV the 2*taps products (unused in DELTA)
 *   5 cb_out
 *   6 wait mode  0 upstream, 1 stoppable, 2 bound-only
 *   7 cb_work    DELTA nine [C,S] working groups
 *   8 cb_decay   DELTA this head's [C,C] decay-ratio tile
 *   9 cb_cc      DELTA the [C,C] working tiles
 *  10 conv_depth CONV tile columns per block (== the reader's depth arg)
 *
 * Build (the whole program; this file is the TRISC third):
 *   RELOCATE_KERNEL_TEXT=1 \
 *     READER_CTARGS=0,1,2,3,4,4096,1,5,7,8,6,1 COMPUTE_CTARGS=0,1,2,3,4,16,1,5,7,8,1 \
 *     WRITER_CTARGS=16,4096,1 CB_TABLE=kernels/cb_tables/gdnc_float32.cbtable \
 *     bash foundry/tt/build_tensix_program.sh \
 *       kernels/resident_gdnc_reader.c kernels/resident_gdnc_compute.c \
 *       kernels/resident_gdnc_writer.c programs/gdnc
 *
 * @status Clean-room rewrite (Phase 4); byte gate against the shipped gdnc images.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_cb_formats.h"
#include "ft_tensix_compute.h"
#include "ft_tensix_compute_cb.h"
#include "ft_tensix_compute_broadcast.h"
#include "ft_tensix_compute_matmul.h"
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

#include "resident_gdnc_protocol.h"
#include "resident_queue_protocol.h"

/* The copy verb implements only the non-32-bit unpack path. */
FTQ_STATIC_ASSERT(FT_CB_ANY_32BIT_UNPACK == 0,
                  "resident_gdnc_compute.c supports no 32-bit unpack source");

#define CB_IN ft_tensix_compile_time_arg(0)
#define CB_STATE ft_tensix_compile_time_arg(1)
#define CB_BAND ft_tensix_compile_time_arg(2)
#define CB_VECTOR ft_tensix_compile_time_arg(3)
#define CB_PROD ft_tensix_compile_time_arg(4)
#define CB_OUT ft_tensix_compile_time_arg(5)
#define WAIT_MODE ft_tensix_compile_time_arg(6)
#define CB_WORK ft_tensix_compile_time_arg(7)
#define CB_DECAY ft_tensix_compile_time_arg(8)
#define CB_CC ft_tensix_compile_time_arg(9)
#define CONV_DEPTH ft_tensix_compile_time_arg(10)

/* Matmul fidelity; only the math thread evaluates these (FT_CB_MATH_FIDELITY exists there). */
#define MATMUL_HIGH_FIDELITY (FT_CB_MATH_FIDELITY != 0)
#define MATMUL_INNER_LOOPS (MATMUL_HIGH_FIDELITY ? (uint32_t)FT_CB_MATH_FIDELITY : 1u)
#define MATMUL_FIDELITY_INCREMENT (MATMUL_HIGH_FIDELITY ? 1u : 0u)

/* ELWADD / ELWSUB words: no clear, no accumulate, no SrcB broadcast, address modifier 0,
 * DST row 0 (ELWADD.md / ELWSUB.md). */
#define ADD_WORD FT_BH_ENC_ELWADD(0u, 0u, FT_BH_ELWADD_SRCB_NO_BCAST, 0u, 0u)
#define SUB_WORD FT_BH_ENC_ELWSUB(0u, 0u, FT_BH_ELWADD_SRCB_NO_BCAST, 0u, 0u)

/* DST ceiling with half-sync and fp32 accumulation (hardware fact). */
#define PRODUCT_BATCH_MAX 4u

/* Newton steps: exact at C = 32, since A_off is nilpotent. */
#define NEWTON_STEPS 4u

/* CB_CC positions within one head (Foundry convention). */
enum {
    kCcKKt = 0,     /* K K^T */
    kCcQKt = 1,     /* Q K^T */
    kCcPm = 2,      /* D o QK^T */
    kCcPms = 3,     /* scale column * Pm */
    kCcDs = 4,      /* D o StrictLower */
    kCcDsb = 5,     /* beta * Ds */
    kCcAoff = 6,    /* KK^T o Dsb */
    kCcA = 7,       /* I + Aoff */
    kCcX0 = 8,      /* I - Aoff; Newton step s: X_s 8+3s, A.X_s 9+3s, 2I - A.X_s 10+3s */
    kCcX4 = 20,     /* X_4 */
    kCcBdiag = 21,  /* I o_bcast beta */
    kCcXb = 22,     /* X_4 . Bdiag */
    kCcGi = 23      /* I o_bcast G */
};

/* CB_WORK groups, each T tiles (tile i of group g at g*T + i). */
enum {
    kWorkKh = 0, /* P (x)rows k */
    kWorkKn = 1, /* Kh . N0^T */
    kWorkZ = 2,  /* V - Kn */
    kWorkU = 3,  /* Xb . Z */
    kWorkQh = 4, /* PS (x)rows q */
    kWorkQn = 5, /* Qh . N0^T */
    kWorkPu = 6, /* Pms . U */
    kWorkUt = 7, /* U^T */
    kWorkKb = 8  /* F (x)rows k */
};

FTQ_STATIC_ASSERT(kCcGi + 1 == kGdncDeltaCcSpan, "CB_CC map must fill kGdncDeltaCcSpan");
FTQ_STATIC_ASSERT(kWorkKb + 1 == kGdncDeltaWorkPerRow, "CB_WORK map must fill the row span");

/* ---- Compute verbs ---------------------------------------------------------------------- */

static FT_BH_INLINE void gdnc_mul_tiles_init(uint32_t a)
{
    (void)a;
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_init(FT_CB_MATH_FIDELITY,
                                                          ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[a],
                                              ft_cb_unpack_num_face_rows[a],
                                              ft_cb_unpack_num_face_cols[a]));
}

static FT_BH_INLINE void gdnc_mul_tiles(uint32_t a, uint32_t b, uint32_t ia, uint32_t ib,
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

static FT_BH_INLINE void gdnc_addsub_tiles_init(uint32_t op_word, uint32_t a)
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
static FT_BH_INLINE void gdnc_addsub_tiles(uint32_t a, uint32_t b, uint32_t ia, uint32_t ib,
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

static FT_BH_INLINE void gdnc_mul_bcast_cols_init(uint32_t a)
{
    (void)a;
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_broadcast_col_init(
        FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_broadcast_col_init(ft_cb_unpack_tile_face_rows[a],
                                                            ft_cb_unpack_num_face_rows[a],
                                                            ft_cb_unpack_num_face_cols[a]));
}

/** Column 0 of b scales each row of a. */
static FT_BH_INLINE void gdnc_mul_bcast_cols(uint32_t a, uint32_t b, uint32_t ia, uint32_t ib,
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

static FT_BH_INLINE void gdnc_copy_tile_init(uint32_t cb)
{
    (void)cb;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a_init(
        ft_cb_unpack_tile_face_rows[cb],
        (uint32_t)ft_cb_unpack_num_face_rows[cb] * ft_cb_unpack_num_face_cols[cb]));
    FT_TENSIX_MATH(ft_tensix_math_datacopy_init((uint32_t)ft_cb_unpack_num_face_rows[cb] *
                                                ft_cb_unpack_num_face_cols[cb]));
}

static FT_BH_INLINE void gdnc_copy_tile(uint32_t cb, uint32_t index, uint32_t dst)
{
    (void)cb;
    (void)index;
    (void)dst;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a(cb, index));
    FT_TENSIX_MATH(ft_tensix_math_datacopy(dst));
}

/**
 * Matmul init for in0 . in1 (or in0 . in1^T with transpose = 1, in1 transposed as a whole
 * tile). The unpacker takes in1's extent first: the reversed source order.
 */
static FT_BH_INLINE void gdnc_matmul_init(uint32_t in0, uint32_t in1, uint32_t transpose)
{
    (void)in0;
    (void)in1;
    (void)transpose;
    FT_TENSIX_MATH(ft_tensix_math_matmul_init(transpose, MATMUL_FIDELITY_INCREMENT,
                                              MATMUL_INNER_LOOPS, MATMUL_HIGH_FIDELITY));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul_init(
        transpose,
        (uint32_t)ft_cb_unpack_tile_num_faces[in1] * ft_cb_unpack_tile_face_rows[in1] *
                FT_TENSIX_FACE_C_DIM - 1u,
        (uint32_t)ft_cb_unpack_tile_num_faces[in0] * ft_cb_unpack_tile_face_rows[in0] *
                FT_TENSIX_FACE_C_DIM - 1u,
        1u /*kt_dim*/));
}

/** Accumulate in0[i0] . in1[i1] into DST 0. */
static FT_BH_INLINE void gdnc_matmul_tiles(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1)
{
    (void)in0;
    (void)in1;
    (void)i0;
    (void)i1;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul(in0, in1, i0, i1, 1 /*ct_dim*/));
    FT_TENSIX_MATH(ft_tensix_math_matmul(0));
}

/* ---- Waits (out of line: spin loops with no Tensix instruction) ------------------------- */

static __attribute__((noinline)) bool gdnc_wait_front(ftq_context *ctx, uint32_t cb,
                                                      uint32_t pages)
{
    return ftq_cb_wait_front_stoppable(ctx, (int32_t)cb, (int32_t)pages, WAIT_MODE);
}

static __attribute__((noinline)) bool gdnc_reserve_back(ftq_context *ctx, uint32_t cb,
                                                        uint32_t pages)
{
    return ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, (int32_t)pages, WAIT_MODE);
}

/** DST tiles [0, tiles) -> `tiles` pages at the back of `cb`, one page at a time. */
static void commit_dst_block_to(ftq_context *ctx, uint32_t cb, uint32_t tiles)
{
    FT_TENSIX_MATH(ft_tensix_math_dest_section_done());
    (void)gdnc_reserve_back(ctx, cb, 1u);
    FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
    FT_TENSIX_PACK(ft_tensix_pack_tile(0u, cb));
    for (uint32_t index = 1; index < tiles; index++) {
        FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, 1));
        (void)gdnc_reserve_back(ctx, cb, 1u);
        FT_TENSIX_PACK(ft_tensix_pack_tile(index, cb));
    }
    FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, 1));
}

/** Tiles per DST section for the products: the largest divisor of `products` up to 4. */
static uint32_t product_batch(uint32_t products)
{
    uint32_t batch = PRODUCT_BATCH_MAX;

    if (products == 0u)
        return 1u;
    if (batch > products)
        batch = products;
    while (batch > 1u && products % batch != 0u)
        batch--;
    return batch;
}

/* ---- PHASE CONV -------------------------------------------------------------------------- */

/** Tile columns per block: CONV_DEPTH clamped to a divisor of col_count; 0 = refuse. */
static uint32_t conv_block_depth(const ftgc_shape *shape)
{
    uint32_t depth = CONV_DEPTH;

    if (shape->conv_taps < 2u || shape->col_count == 0u || depth == 0u)
        return 0u;
    if (depth > shape->col_count)
        depth = shape->col_count;
    while (depth > 0u && (shape->col_count % depth) != 0u)
        depth--;
    return depth;
}

/**
 * The 2*taps products of one column, in band order: product 2*tau = X (x) W_tau, 2*tau + 1 =
 * Win (x) W_tau, so the matmul walks both operands with one index. cb_vector is tap-major then
 * column within the block.
 */
static void conv_products(ftq_context *ctx, uint32_t taps, uint32_t batch, uint32_t depth,
                          uint32_t slot)
{
    const uint32_t products = 2u * taps;

    for (uint32_t first = 0; first < products; first += batch) {
        gdnc_mul_tiles_init(CB_IN);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        for (uint32_t index = 0; index < batch; index++) {
            const uint32_t product = first + index;
            const uint32_t tap = product / 2u;
            const uint32_t source = ((product & 1u) == 0u) ? CB_IN : CB_STATE;

            gdnc_mul_tiles(source, CB_VECTOR, slot, tap * depth + slot, index);
        }
        commit_dst_block_to(ctx, CB_PROD, batch);
    }
}

/** Band matmul + SiLU -> conv_out, then the raw input tile copied out behind it. */
static void conv_emit_column(ftq_context *ctx, uint32_t taps, uint32_t slot)
{
    const uint32_t products = 2u * taps;

    (void)gdnc_wait_front(ctx, CB_PROD, products);
    gdnc_matmul_init(CB_BAND, CB_PROD, 0u);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    for (uint32_t index = 0; index < products; index++)
        gdnc_matmul_tiles(CB_BAND, CB_PROD, index, index);
    FT_TENSIX_MATH(ft_tensix_math_silu_tile_init());
    FT_TENSIX_MATH(ft_tensix_math_silu_tile(0));
    commit_dst_block_to(ctx, CB_OUT, 1u);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_PROD, products));

    gdnc_copy_tile_init(CB_IN);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    gdnc_copy_tile(CB_IN, slot, 0);
    commit_dst_block_to(ctx, CB_OUT, 1u);
}

static void conv_one_block(ftq_context *ctx, uint32_t taps, uint32_t batch, uint32_t depth)
{
    (void)gdnc_wait_front(ctx, CB_IN, depth);
    (void)gdnc_wait_front(ctx, CB_STATE, depth);
    (void)gdnc_wait_front(ctx, CB_VECTOR, depth * taps);
    for (uint32_t slot = 0; slot < depth; slot++) {
        conv_products(ctx, taps, batch, depth, slot);
        conv_emit_column(ctx, taps, slot);
    }
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_IN, depth));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_STATE, depth));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_VECTOR, depth * taps));
}

/* ---- PHASE DELTA ------------------------------------------------------------------------- */
/* Every push is followed by a wait on the same buffer: a tile this thread packed is readable
 * by its unpack side only after the pack-side push is seen. The waits are cumulative, since
 * nothing pops inside a head. */

/** Tile `index` of CB_WORK group `group`. */
static uint32_t wk(uint32_t group, uint32_t tiles, uint32_t index)
{
    return group * tiles + index;
}

/** CB_CC position of the Newton iterate X_step. */
static uint32_t cc_newton_x(uint32_t step)
{
    return 8u + 3u * step;
}

/** CB_CC position of A . X_step. */
static uint32_t cc_newton_ax(uint32_t step)
{
    return 9u + 3u * step;
}

/** CB_CC position of 2I - A . X_step. */
static uint32_t cc_newton_t(uint32_t step)
{
    return 9u + 3u * step + 1u;
}

/** Each row of `tiles` source tiles scaled by gate column `column`, into CB_WORK. */
static void delta_scale_rows(ftq_context *ctx, uint32_t source, uint32_t source_first,
                             uint32_t column, uint32_t tiles)
{
    gdnc_mul_bcast_cols_init(source);
    for (uint32_t index = 0; index < tiles; index++) {
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        gdnc_mul_bcast_cols(source, CB_VECTOR, source_first + index, column, 0);
        commit_dst_block_to(ctx, CB_WORK, 1u);
    }
}

/** One tile to CB_CC, then wait until `pushed` CB_CC pages are visible. */
static void delta_cc_commit(ftq_context *ctx, uint32_t pushed)
{
    commit_dst_block_to(ctx, CB_CC, 1u);
    (void)gdnc_wait_front(ctx, CB_CC, pushed);
}

/** Gram tile sum_i in[a_first + i] . in[b_first + i]^T -> CB_CC. */
static void delta_gram(ftq_context *ctx, uint32_t a_first, uint32_t b_first, uint32_t tiles,
                       uint32_t pushed)
{
    gdnc_matmul_init(CB_IN, CB_IN, 1u);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    for (uint32_t index = 0; index < tiles; index++)
        gdnc_matmul_tiles(CB_IN, CB_IN, a_first + index, b_first + index);
    delta_cc_commit(ctx, pushed);
}

static void delta_cc_mul(ftq_context *ctx, uint32_t a, uint32_t b, uint32_t ia, uint32_t ib,
                         uint32_t pushed)
{
    gdnc_mul_tiles_init(a);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    gdnc_mul_tiles(a, b, ia, ib, 0);
    delta_cc_commit(ctx, pushed);
}

static void delta_cc_bcast(ftq_context *ctx, uint32_t a, uint32_t b, uint32_t ia, uint32_t ib,
                           uint32_t pushed)
{
    gdnc_mul_bcast_cols_init(a);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    gdnc_mul_bcast_cols(a, b, ia, ib, 0);
    delta_cc_commit(ctx, pushed);
}

static void delta_cc_addsub(ftq_context *ctx, uint32_t op_word, uint32_t a, uint32_t b,
                            uint32_t ia, uint32_t ib, uint32_t pushed)
{
    gdnc_addsub_tiles_init(op_word, a);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    gdnc_addsub_tiles(a, b, ia, ib, 0);
    delta_cc_commit(ctx, pushed);
}

static void delta_cc_matmul(ftq_context *ctx, uint32_t a, uint32_t b, uint32_t ia, uint32_t ib,
                            uint32_t pushed)
{
    gdnc_matmul_init(a, b, 0u);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    gdnc_matmul_tiles(a, b, ia, ib);
    delta_cc_commit(ctx, pushed);
}

/** [C,S] = X . N0^T for CB_WORK group `group`, into group `out_group`. */
static void delta_state_project(ftq_context *ctx, uint32_t group, uint32_t out_group,
                                uint32_t tiles)
{
    gdnc_matmul_init(CB_WORK, CB_STATE, 1u);
    for (uint32_t row_block = 0; row_block < tiles; row_block++) {
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        for (uint32_t index = 0; index < tiles; index++)
            gdnc_matmul_tiles(CB_WORK, CB_STATE, wk(group, tiles, index),
                              row_block * tiles + index);
        commit_dst_block_to(ctx, CB_WORK, 1u);
    }
    (void)gdnc_wait_front(ctx, CB_WORK, (out_group + 1u) * tiles);
}

/** The [C,C] tiles of one head, ending with Xb = X_4 . Bdiag and Gi. */
static void delta_solve(ftq_context *ctx, uint32_t tiles)
{
    const uint32_t k_first = tiles;

    delta_gram(ctx, k_first, k_first, tiles, (uint32_t)kCcKKt + 1u);
    delta_gram(ctx, 0u, k_first, tiles, (uint32_t)kCcQKt + 1u);
    delta_cc_mul(ctx, CB_CC, CB_DECAY, (uint32_t)kCcQKt, 0u, (uint32_t)kCcPm + 1u);
    delta_cc_bcast(ctx, CB_CC, CB_BAND, (uint32_t)kCcPm, (uint32_t)kGdncDeltaConstScale,
                   (uint32_t)kCcPms + 1u);
    delta_cc_mul(ctx, CB_DECAY, CB_BAND, 0u, (uint32_t)kGdncDeltaConstStrict,
                 (uint32_t)kCcDs + 1u);
    delta_cc_bcast(ctx, CB_CC, CB_VECTOR, (uint32_t)kCcDs, (uint32_t)kGdncGateB,
                   (uint32_t)kCcDsb + 1u);
    delta_cc_mul(ctx, CB_CC, CB_CC, (uint32_t)kCcKKt, (uint32_t)kCcDsb, (uint32_t)kCcAoff + 1u);
    delta_cc_addsub(ctx, ADD_WORD, CB_BAND, CB_CC, (uint32_t)kGdncDeltaConstIdentity,
                    (uint32_t)kCcAoff, (uint32_t)kCcA + 1u);
    delta_cc_addsub(ctx, SUB_WORD, CB_BAND, CB_CC, (uint32_t)kGdncDeltaConstIdentity,
                    (uint32_t)kCcAoff, (uint32_t)kCcX0 + 1u);
    for (uint32_t step = 0; step < NEWTON_STEPS; step++) {
        delta_cc_matmul(ctx, CB_CC, CB_CC, (uint32_t)kCcA, cc_newton_x(step),
                        cc_newton_ax(step) + 1u);
        delta_cc_addsub(ctx, SUB_WORD, CB_BAND, CB_CC, (uint32_t)kGdncDeltaConstTwoI,
                        cc_newton_ax(step), cc_newton_t(step) + 1u);
        delta_cc_matmul(ctx, CB_CC, CB_CC, cc_newton_x(step), cc_newton_t(step),
                        cc_newton_x(step + 1u) + 1u);
    }
    delta_cc_bcast(ctx, CB_BAND, CB_VECTOR, (uint32_t)kGdncDeltaConstIdentity,
                   (uint32_t)kGdncGateB, (uint32_t)kCcBdiag + 1u);
    delta_cc_matmul(ctx, CB_CC, CB_CC, cc_newton_x(NEWTON_STEPS), (uint32_t)kCcBdiag,
                    (uint32_t)kCcXb + 1u);
    delta_cc_bcast(ctx, CB_BAND, CB_VECTOR, (uint32_t)kGdncDeltaConstIdentity,
                   (uint32_t)kGdncGateG, (uint32_t)kCcGi + 1u);
}

/**
 * One v-head: y tiles to CB_OUT, then the new state. The last loop issues two matmuls in one
 * DST section, the second on a different operand pair than the init named, with no re-init;
 * the adds likewise reuse the reversed boot configuration.
 */
static void delta_one_head(ftq_context *ctx, uint32_t tiles, bool round)
{
    const uint32_t k_first = tiles;
    const uint32_t v_first = 2u * tiles;

    (void)gdnc_wait_front(ctx, CB_IN, 3u * tiles);
    (void)gdnc_wait_front(ctx, CB_STATE, tiles * tiles);
    (void)gdnc_wait_front(ctx, CB_DECAY, 1u);
    (void)gdnc_wait_front(ctx, CB_VECTOR, (uint32_t)kGdncDeltaVectorSpan);
    delta_scale_rows(ctx, CB_IN, k_first, (uint32_t)kGdncGateP, tiles);
    (void)gdnc_wait_front(ctx, CB_WORK, ((uint32_t)kWorkKh + 1u) * tiles);
    delta_solve(ctx, tiles);
    delta_state_project(ctx, (uint32_t)kWorkKh, (uint32_t)kWorkKn, tiles);

    gdnc_addsub_tiles_init(SUB_WORD, CB_IN);
    for (uint32_t index = 0; index < tiles; index++) {
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        gdnc_addsub_tiles(CB_IN, CB_WORK, v_first + index, wk(kWorkKn, tiles, index), 0);
        commit_dst_block_to(ctx, CB_WORK, 1u);
    }
    (void)gdnc_wait_front(ctx, CB_WORK, ((uint32_t)kWorkZ + 1u) * tiles);

    gdnc_matmul_init(CB_CC, CB_WORK, 0u);
    for (uint32_t index = 0; index < tiles; index++) {
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        gdnc_matmul_tiles(CB_CC, CB_WORK, (uint32_t)kCcXb, wk(kWorkZ, tiles, index));
        commit_dst_block_to(ctx, CB_WORK, 1u);
    }
    (void)gdnc_wait_front(ctx, CB_WORK, ((uint32_t)kWorkU + 1u) * tiles);

    delta_scale_rows(ctx, CB_IN, 0u, (uint32_t)kGdncGatePS, tiles);
    (void)gdnc_wait_front(ctx, CB_WORK, ((uint32_t)kWorkQh + 1u) * tiles);
    delta_state_project(ctx, (uint32_t)kWorkQh, (uint32_t)kWorkQn, tiles);

    gdnc_matmul_init(CB_CC, CB_WORK, 0u);
    for (uint32_t index = 0; index < tiles; index++) {
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        gdnc_matmul_tiles(CB_CC, CB_WORK, (uint32_t)kCcPms, wk(kWorkU, tiles, index));
        commit_dst_block_to(ctx, CB_WORK, 1u);
    }
    (void)gdnc_wait_front(ctx, CB_WORK, ((uint32_t)kWorkPu + 1u) * tiles);

    gdnc_addsub_tiles_init(ADD_WORD, CB_WORK);
    for (uint32_t index = 0; index < tiles; index++) {
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        gdnc_addsub_tiles(CB_WORK, CB_WORK, wk(kWorkQn, tiles, index), wk(kWorkPu, tiles, index),
                          0);
        commit_dst_block_to(ctx, CB_OUT, 1u);
    }

    gdnc_matmul_init(CB_BAND, CB_WORK, 1u);
    for (uint32_t index = 0; index < tiles; index++) {
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        gdnc_matmul_tiles(CB_BAND, CB_WORK, (uint32_t)kGdncDeltaConstIdentity,
                          wk(kWorkU, tiles, index));
        commit_dst_block_to(ctx, CB_WORK, 1u);
    }
    (void)gdnc_wait_front(ctx, CB_WORK, ((uint32_t)kWorkUt + 1u) * tiles);

    delta_scale_rows(ctx, CB_IN, k_first, (uint32_t)kGdncGateF, tiles);
    (void)gdnc_wait_front(ctx, CB_WORK, ((uint32_t)kWorkKb + 1u) * tiles);

    gdnc_matmul_init(CB_CC, CB_STATE, 0u);
    for (uint32_t row_block = 0; row_block < tiles; row_block++) {
        for (uint32_t index = 0; index < tiles; index++) {
            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
            gdnc_matmul_tiles(CB_CC, CB_STATE, (uint32_t)kCcGi, row_block * tiles + index);
            gdnc_matmul_tiles(CB_WORK, CB_WORK, wk(kWorkUt, tiles, row_block),
                              wk(kWorkKb, tiles, index));
            if (round) {
                FT_TENSIX_MATH(ft_tensix_math_tf32_round_bias_tile_init());
                FT_TENSIX_MATH(ft_tensix_math_tf32_round_bias_tile(0u));
            }
            commit_dst_block_to(ctx, CB_OUT, 1u);
            /* The SFPU init reset the counters this loop's matmuls run on. */
            if (round)
                FT_TENSIX_MATH(ft_tensix_math_matmul_init(0u, MATMUL_FIDELITY_INCREMENT,
                                                          MATMUL_INNER_LOOPS,
                                                          MATMUL_HIGH_FIDELITY));
        }
    }

    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_IN, 3u * tiles));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_STATE, tiles * tiles));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_DECAY, 1));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_VECTOR, (uint32_t)kGdncDeltaVectorSpan));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_CC, (uint32_t)kGdncDeltaCcSpan));
    FT_TENSIX_UNPACK(
        ft_tensix_compute_cb_pop_front(CB_WORK, (uint32_t)kGdncDeltaWorkPerRow * tiles));
}

/* ---- Descriptor dispatch ----------------------------------------------------------------- */

/** Return the indexed buffers to their base at the top of a descriptor (no-wrap addressing).
 *  CB_OUT is always accessed at tile 0 and is never rebased. */
static void gdnc_rebase_indexed_buffers(void)
{
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_IN));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_STATE));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_BAND));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_VECTOR));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_PROD));
    FT_TENSIX_PACK(ft_tensix_cb_rebase_write_pointer(CB_PROD));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_DECAY));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_CC));
    FT_TENSIX_PACK(ft_tensix_cb_rebase_write_pointer(CB_CC));
    FT_TENSIX_UNPACK(ft_tensix_cb_rebase_read_pointer(CB_WORK));
    FT_TENSIX_PACK(ft_tensix_cb_rebase_write_pointer(CB_WORK));
}

/** One kOpGdnChunk descriptor. CB_BAND is pushed once per descriptor, indexed absolutely and
 *  popped at the end. */
static void compute_one_gdnc(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const ftgc_shape shape = ftgc_decode(parameters);
    uint32_t constants;
    uint32_t index;
    uint32_t batch;

    if (!ftgc_shape_is_supported(&shape)) {
        ftq_abort(ctx, kErrBadParameters);
        return;
    }
    if (shape.phase != (uint32_t)kGdncPhaseDelta &&
        (!ftgc_conv_depth_agrees(shape.declared_conv_depth, (uint32_t)CONV_DEPTH) ||
         conv_block_depth(&shape) == 0u)) {
        ftq_abort(ctx, kErrBadParameters);
        return;
    }
    gdnc_rebase_indexed_buffers();

    constants = (shape.phase == (uint32_t)kGdncPhaseDelta) ? (uint32_t)kGdncDeltaConstTiles
                                                           : ftgc_band_tiles(&shape);
    (void)gdnc_wait_front(ctx, CB_BAND, constants);
    if (shape.phase == (uint32_t)kGdncPhaseDelta) {
        const bool round =
            (parameters[kGdncFlags] & (uint32_t)kGdncDeltaFlagStateRound) != 0u;

        for (index = 0; index < shape.head_count; index++)
            delta_one_head(ctx, shape.state_tiles, round);
    } else {
        const uint32_t depth = conv_block_depth(&shape);

        batch = product_batch(constants);
        for (index = 0; index < shape.col_count; index += depth)
            conv_one_block(ctx, shape.conv_taps, batch, depth);
    }
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(CB_BAND, constants));
}

/** Hardware startup in REVERSED source order: a = CB_PROD, b = CB_BAND, packing to CB_OUT. */
static FT_BH_INLINE void gdnc_hw_startup(void)
{
    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[CB_PROD], ft_cb_unpack_src_format[CB_BAND],
        ft_cb_unpack_dst_format[CB_PROD], ft_cb_unpack_dst_format[CB_BAND],
        ft_cb_unpack_tile_face_rows[CB_PROD], ft_cb_unpack_tile_face_rows[CB_BAND],
        ft_cb_unpack_tile_num_faces[CB_PROD], ft_cb_unpack_tile_num_faces[CB_BAND],
        ft_tensix_cb(CB_PROD)->fifo_page_size, ft_tensix_cb(CB_BAND)->fifo_page_size));
    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[CB_PROD],
                                               ft_cb_unpack_dst_format[CB_BAND],
                                               FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(
        ft_cb_pack_src_format[CB_OUT], ft_cb_pack_dst_format[CB_OUT],
        ft_tensix_cb(CB_OUT)->fifo_page_size, ft_cb_pack_tile_num_faces[CB_OUT],
        ft_cb_pack_partial_face[CB_OUT], 0, FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[CB_OUT],
                                       ft_cb_pack_tile_face_rows[CB_OUT],
                                       ft_cb_pack_tile_num_faces[CB_OUT], 1));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());
    gdnc_matmul_init(CB_BAND, CB_PROD, 0u);
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

    gdnc_hw_startup();
    ctx.cursor = ctx.control[kFieldCompletedCount];

    while (running) {
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);

        volatile FTQ_L1_PTR uint32_t *descriptor = ftq_descriptor(&ctx);
        const uint32_t opcode = descriptor[kDescOpcode];

        ftq_status_opcode(&ctx, opcode);
        if (opcode == kOpGdnChunk) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            compute_one_gdnc(&ctx, ftq_parameters(descriptor));
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
