/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_attention_transpose_probe.c
 * @brief THE ONE EXPERIMENT flash-decode attention could not be written without, and it is
 *   deliberately the smallest program that can answer it. Despite the name it is a TRISC
 *   COMPUTE image, not a data-movement one.
 *
 * ft_tt_op_memory.h PART 1 1.3 records ONE inferred assumption behind the KV cache layout:
 * that `matmul_init(in0, in1, transpose)` -- documented as "perform transpose operation on B",
 * reaching the unpacker's matmul init -- actually transposes the in1 tile on its way to SrcA
 * ON BLACKHOLE. Nothing in this repo had executed it. The layout survives either answer (1.3
 * states the fallback: transpose Q, 4 tiles per head per token, instead of K), but ATTENTION'S
 * INNER LOOP DOES NOT -- the scores come out row-oriented or column-oriented depending on it,
 * and the softmax reduces along a different axis in each case.
 *
 * So this file is resident_matmul_compute.c's loop with ONE change: the transpose flag is read
 * from the descriptor's parameter block (word 14, which the matmul block leaves unused) and
 * passed to the matmul init. That makes the A/B a DESCRIPTOR difference and not a build
 * difference -- the same binary, the same core, the same operands, one word apart -- which is
 * the standard this repo already holds a layout A/B to (docs/tt_bringup_log.md 18.4).
 *
 * THE RUNTIME FLAG IS WHY THIS KERNEL'S ADDR_MOD BLOCK HAS BRANCHES. Slots 1 and 4 have
 * transpose-dependent SrcA increments, so both arms are emitted here where every other matmul
 * folds to one; slot 2 has no transpose branch at all in the full-tile case, which is
 * upstream's control flow and not an omission.
 *
 * It is driven by the UNMODIFIED resident_matmul_{reader,writer}.c, so the only thing in the
 * program that is not already green on silicon is the flag.
 *
 * Compile-time args (positional), identical to resident_matmul_compute.c:
 *   0: cb_in0 index   1: cb_in1 index   2: cb_out index
 *   3: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build:
 *   CTARGS=0,1,16,1 CB_TABLE=kernels/cb_tables/matmul_float16b.cbtable \
 *       bash foundry/tt/build_tensix_kernel.sh \
 *           foundry/tt/kernels/resident_attention_transpose_probe.c /tmp/ttk_probe
 *
 * @status ANSWERED on Tensix 70; C99 CONVERSION MEASURED 2026-08-13 (plan Phase 6.6), with
 *   the .cpp retained as the baseline until all three images are byte-identical. Against the
 *   C++ build all three Tensix instruction streams are IDENTICAL (trisc0 45/45, trisc1 89/89
 *   -- including BOTH arms of the transpose-branched ADDR_MOD block -- trisc2 37/37); trisc1
 *   and trisc2 hold the same 0x7c0 / 0xa00 with an identical immediate multiset and differ
 *   only in register allocation and commutative operand order, and trisc0 is 0xb30 vs 0xbd0.
 *   The original finding follows. THE TRANSPOSE FLAG
 *   WORKS ON BLACKHOLE. transpose=1 reproduces A*B^T (PCC 0.999998668, 942/1024 elements
 *   bf16-exact) and transpose=0 reproduces A*B (PCC 0.999998688, 932/1024) against a
 *   deliberately non-symmetric operand pair, with each arm scoring ~0 (PCC -0.046) against the
 *   other candidate. So ft_tt_op_memory.h 1.3's inferred assumption is discharged and its
 *   fallback -- transpose Q instead of K -- is not needed.
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

#include "resident_queue_protocol.h"

#define DST_REGISTER 0u

/** Word 14 of the matmul parameter block: 0 = plain C = A*B, 1 = C = A*B^T. The matmul family
 *  writes 0 into it and reads it nowhere, so borrowing it here cannot change what a matmul
 *  descriptor means to any other program. */
#define PROBE_TRANSPOSE_WORD 14u

#ifdef FT_TRISC_MATH
#define MATMUL_HIGH_FIDELITY (FT_CB_MATH_FIDELITY != 0)
#define MATMUL_INNER_LOOPS (MATMUL_HIGH_FIDELITY ? (uint32_t)FT_CB_MATH_FIDELITY : 1u)
#define MATMUL_FIDELITY_INCREMENT (MATMUL_HIGH_FIDELITY ? 1u : 0u)
#endif

/**
 * resident_matmul_compute.c's compute_one_matmul(), with the transpose flag applied.
 *
 * The matmul init is re-run INSIDE the descriptor loop here and not once at boot, because the
 * flag is a per-descriptor word -- which is the whole point of the probe. On the real
 * attention kernel it is re-run once per phase for the same reason (the QK matmul wants the
 * flag and the PV matmul does not).
 */
static void compute_one_matmul(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                               uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out,
                               uint32_t wait_mode)
{
    const uint32_t tile_rows = parameters[kMatmulTileRows];
    const uint32_t tile_inner = parameters[kMatmulTileInner];
    const uint32_t tile_cols = parameters[kMatmulTileCols];
    const uint32_t transpose = parameters[PROBE_TRANSPOSE_WORD];
    uint32_t mt;
    uint32_t nt;
    uint32_t kt;

    FT_TENSIX_MATH(ft_tensix_math_matmul_init(transpose, MATMUL_FIDELITY_INCREMENT,
                                              MATMUL_INNER_LOOPS, MATMUL_HIGH_FIDELITY));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul_init(
        transpose,
        (uint32_t)ft_cb_unpack_tile_num_faces[cb_in1] * ft_cb_unpack_tile_face_rows[cb_in1] *
                FT_TENSIX_FACE_C_DIM - 1u,
        (uint32_t)ft_cb_unpack_tile_num_faces[cb_in0] * ft_cb_unpack_tile_face_rows[cb_in0] *
                FT_TENSIX_FACE_C_DIM - 1u,
        1u /*kt_dim*/));

    for (mt = 0; mt < tile_rows; mt++) {
        for (nt = 0; nt < tile_cols; nt++) {
            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());

            for (kt = 0; kt < tile_inner; kt++) {
                (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in0, 1, wait_mode);
                (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in1, 1, wait_mode);

                /* Issued even while aborting: skipping the unpack stops SrcA/SrcB dvalid and
                 * MATH stalls in the BACKEND, where no flag reaches (resident_queue_protocol.h
                 * rule 3 -- it wedged a core until reboot). */
                FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul(cb_in0, cb_in1, 0, 0, 1 /*ct_dim*/));
                FT_TENSIX_MATH(ft_tensix_math_matmul(DST_REGISTER));

                FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in0, 1));
                FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in1, 1));
            }

            FT_TENSIX_MATH(ft_tensix_math_dest_section_done()); /* tile_regs_commit() */

            (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb_out, 1, wait_mode);

            FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done()); /* tile_regs_wait() */
            FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER, cb_out));
            FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
            FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, 1));
        }
    }
}

void kernel_main(void)
{
    const uint32_t cb_in0 = ft_tensix_compile_time_arg(0);
    const uint32_t cb_in1 = ft_tensix_compile_time_arg(1);
    const uint32_t cb_out = ft_tensix_compile_time_arg(2);
    const uint32_t wait_mode = ft_tensix_compile_time_arg(3);

    ftq_context ctx;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);

    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    /* compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out): in0 -> SrcB,
     * in1 -> SrcA, which is also why the transpose flag lands on in1. */
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

    FT_TENSIX_MATH(ft_tensix_math_matmul_init(0 /*transpose*/, MATMUL_FIDELITY_INCREMENT,
                                              MATMUL_INNER_LOOPS, MATMUL_HIGH_FIDELITY));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul_init(
        0 /*transpose*/,
        (uint32_t)ft_cb_unpack_tile_num_faces[cb_in1] * ft_cb_unpack_tile_face_rows[cb_in1] *
                FT_TENSIX_FACE_C_DIM - 1u,
        (uint32_t)ft_cb_unpack_tile_num_faces[cb_in0] * ft_cb_unpack_tile_face_rows[cb_in0] *
                FT_TENSIX_FACE_C_DIM - 1u,
        1u /*kt_dim*/));

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
        if (opcode == kOpMatmul) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            compute_one_matmul(&ctx, (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)descriptor[kDescA0],
                               cb_in0, cb_in1, cb_out, wait_mode);
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
