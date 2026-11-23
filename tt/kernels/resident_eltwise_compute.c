/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_eltwise_compute.c
 * @brief Compute third of the resident ELEMENTWISE triad: one binary serving silu, sigmoid,
 *   add, mul, sub and the silu-gated multiply y = silu(a) * b, the function chosen per
 *   descriptor by the parameter block's kVecFunction word (resident_vector_protocol.h).
 *
 *   Compiled three times (FT_TRISC_UNPACK / MATH / PACK). Every hardware statement sits in one
 *   FT_TENSIX_<thread>(...) wrapper, so each image keeps only its own thread's work, while the
 *   control flow and the queue protocol run identically on all three threads.
 *
 *   Queue-protocol rules that shape the code: loop bounds come only from the descriptor; after
 *   an abort every pop, DST acquire/commit and push still runs (on garbage) so the threads'
 *   handshakes stay balanced; an abandoned descriptor does not advance the cursor.
 *
 * Compile-time args: 0 cb_in0 (input A), 1 cb_in1 (input B), 2 cb_scratch (one-page staging for
 *   silu(a) in the gated function), 3 cb_out, 4 wait_mode (0 upstream, 1 stoppable, 2 bound-only).
 *
 * Build:
 *   CTARGS=0,1,2,16,1 CB_TABLE=kernels/cb_tables/eltwise_float16b.cbtable \
 *       bash foundry/tt/build_tensix_kernel.sh \
 *           foundry/tt/kernels/resident_eltwise_compute.c out/ttk_eltwise
 *
 * Also built with gate_narrow.cbtable, render_narrow.cbtable and render_widen.cbtable (narrow
 * tiles), so each per-CB geometry array below is chosen deliberately.
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
#include "ft_tensix_compute_sfpu.h"
#include "ft_tensix_kernel_args.h"
#ifdef FT_TRISC_MATH
#include "ft_tensix_compute_math.h"
#include "ft_tensix_math_eltwise_sfpu.h"
#endif
#ifdef FT_TRISC_UNPACK
#include "ft_tensix_compute_unpack.h"
#endif
#ifdef FT_TRISC_PACK
#include "ft_tensix_compute_pack.h"
#endif

#include "resident_queue_protocol.h"
#include "resident_vector_protocol.h"

#define cb_in0 ft_tensix_compile_time_arg(0)
#define cb_in1 ft_tensix_compile_time_arg(1)
#define cb_scratch ft_tensix_compile_time_arg(2)
#define cb_out ft_tensix_compile_time_arg(3)
#define wait_mode ft_tensix_compile_time_arg(4)

enum { dst_register = 0 };

/* Encoded FPU words handed to the add/sub init (not issued here). */
#define ELTWISE_ADD_WORD                                                                       \
    FT_BH_ENC_ELWADD(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_0, 0)
#define ELTWISE_SUB_WORD                                                                       \
    FT_BH_ENC_ELWSUB(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_0, 0)

/* ===== Compute verbs (thread order inside each is fixed) ================================ */

/** One-time unit setup: A -> SrcA, B -> SrcB, packer configured for `out`. */
static FT_BH_INLINE void eltwise_hw_startup(uint32_t a, uint32_t b, uint32_t out)
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
                                       ft_cb_pack_tile_num_faces[out], 1 /* tiles */));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());
}

/** Copy init: UNPACK first, then MATH. */
static FT_BH_INLINE void eltwise_copy_init(uint32_t cb)
{
    (void)cb;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a_init(
        ft_cb_unpack_tile_face_rows[cb],
        (uint32_t)ft_cb_unpack_num_face_rows[cb] * ft_cb_unpack_num_face_cols[cb]));
    FT_TENSIX_MATH(ft_tensix_math_datacopy_init((uint32_t)ft_cb_unpack_num_face_rows[cb] *
                                                ft_cb_unpack_num_face_cols[cb]));
}

/** Copy one tile of `cb` into DST. */
static FT_BH_INLINE void eltwise_copy_apply(uint32_t cb)
{
    (void)cb;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a(cb, 0));
    FT_TENSIX_MATH(ft_tensix_math_datacopy(dst_register));
}

/** Multiply init: MATH first, then UNPACK; geometry from `a` only. */
static FT_BH_INLINE void eltwise_mul_init(uint32_t a, uint32_t b)
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
static FT_BH_INLINE void eltwise_mul_apply(uint32_t a, uint32_t b)
{
    (void)a;
    (void)b;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, 0, 0));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul(
        dst_register, (uint32_t)ft_cb_unpack_num_face_rows[a] * ft_cb_unpack_num_face_cols[a],
        ft_cb_unpack_tile_face_rows[a]));
}

/** Add/sub init (always lowest fidelity): MATH first, then UNPACK. */
static FT_BH_INLINE void eltwise_addsub_init(uint32_t binary_op_word, uint32_t a)
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

/** Add or subtract the front tiles of `a` and `b` into DST (op chosen by the init). */
static FT_BH_INLINE void eltwise_addsub_apply(uint32_t a, uint32_t b)
{
    (void)a;
    (void)b;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, 0, 0));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub(dst_register));
}

/* ===== Per-descriptor steps ============================================================= */

/** Configure the units for `function` once per descriptor. */
static void configure_for_function(uint32_t function)
{
    switch (function) {
    case kVecFnSilu:
        eltwise_copy_init(cb_in0);
        FT_TENSIX_MATH(ft_tensix_math_silu_tile_init());
        break;
    case kVecFnSigmoid:
        eltwise_copy_init(cb_in0);
        FT_TENSIX_MATH(ft_tensix_math_sigmoid_tile_init());
        break;
    case kVecFnAdd:
        eltwise_addsub_init(ELTWISE_ADD_WORD, cb_in0);
        break;
    case kVecFnMul:
        eltwise_mul_init(cb_in0, cb_in1);
        break;
    case kVecFnSub:
        eltwise_addsub_init(ELTWISE_SUB_WORD, cb_in0);
        break;
    default:
        break;
    }
}

/** Compute one tile of `function` into DST (row0: only row 0 of the tile is live). */
static void apply_to_dst(uint32_t function, bool row0)
{
    (void)row0;
    switch (function) {
    case kVecFnSilu:
        eltwise_copy_apply(cb_in0);
        if (row0) {
            FT_TENSIX_MATH(ft_tensix_math_silu_tile_row0(dst_register));
        } else {
            FT_TENSIX_MATH(ft_tensix_math_silu_tile(dst_register));
        }
        break;
    case kVecFnSigmoid:
        eltwise_copy_apply(cb_in0);
        if (row0) {
            FT_TENSIX_MATH(ft_tensix_math_sigmoid_tile_row0(dst_register));
        } else {
            FT_TENSIX_MATH(ft_tensix_math_sigmoid_tile(dst_register));
        }
        break;
    case kVecFnAdd:
        eltwise_addsub_apply(cb_in0, cb_in1);
        break;
    case kVecFnMul:
        eltwise_mul_apply(cb_in0, cb_in1);
        break;
    case kVecFnSub:
        eltwise_addsub_apply(cb_in0, cb_in1);
        break;
    default:
        break;
    }
}

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

/** y = silu(a) * b for one tile, staging silu(a) through cb_scratch. The silu init is redone
 *  every tile because the multiply init in between reconfigures the unit. */
static void apply_silu_gated_multiply(ftq_context *ctx, bool row0)
{
    (void)row0;
    eltwise_copy_init(cb_in0);
    FT_TENSIX_MATH(ft_tensix_math_silu_tile_init());
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    eltwise_copy_apply(cb_in0);
    if (row0) {
        FT_TENSIX_MATH(ft_tensix_math_silu_tile_row0(dst_register));
    } else {
        FT_TENSIX_MATH(ft_tensix_math_silu_tile(dst_register));
    }
    commit_dst_to(ctx, cb_scratch);

    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_scratch, 1, wait_mode);
    eltwise_mul_init(cb_scratch, cb_in1);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    eltwise_mul_apply(cb_scratch, cb_in1);
    commit_dst_to(ctx, cb_out);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_scratch, 1));
}

/** Run one ELTWISE descriptor. */
static void compute_one_eltwise(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t function = parameters[kVecFunction];
    const uint32_t tile_count = parameters[kVecTileCount];
    const bool binary = ftv_function_is_binary(function);
    const bool gated = ftv_function_uses_scratch(function);
    const bool row0 = (parameters[kVecEltwiseFlags] & kVecEltwiseRow0) != 0u;
    const bool probe = (ftv_probe_bits(parameters[kVecDepth]) & kVecProbeNoCompute) != 0u;
    uint32_t tile;

    if (probe) {
        /* Timing arm: CB handshakes only, no unpack/math/pack. */
        for (tile = 0; tile < tile_count; tile++) {
            (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in0, 1, wait_mode);
            if (binary)
                (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in1, 1, wait_mode);
            (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb_out, 1, wait_mode);
            FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, 1));
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in0, 1));
            if (binary) {
                FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in1, 1));
            }
        }
        return;
    }

    configure_for_function(function);
    for (tile = 0; tile < tile_count; tile++) {
        (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in0, 1, wait_mode);
        if (binary)
            (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in1, 1, wait_mode);
        if (gated) {
            apply_silu_gated_multiply(ctx, row0);
        } else {
            /* The acquire does not zero DST; PACK zeroes it in its section-done. */
            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
            apply_to_dst(function, row0);
            commit_dst_to(ctx, cb_out);
        }
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in0, 1));
        if (binary) {
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in1, 1));
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

    /* Once, outside the loop: every CB of one table carries one format. */
    eltwise_hw_startup(cb_in0, cb_in1, cb_out);

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
        if (opcode == kOpEltwise) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            compute_one_eltwise(&ctx, ftq_parameters(descriptor));
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
