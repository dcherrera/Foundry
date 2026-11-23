/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_matmul_compute.c
 * @brief Compute third of the RESIDENT matmul triad. matmul_compute.c's blocked outer product,
 *   unchanged, inside a queue loop -- and the file where `cb_wait_front` finally becomes
 *   stoppable, which is the whole reason the resident triad was an open question
 *   (resident_queue_protocol.h, rules 2-4).
 *
 * THREE THINGS THAT ARE NOT OBVIOUS AND ARE LOAD-BEARING:
 *
 *   1. THE THREE TRISC THREADS RUN THE SAME TRIP COUNTS, ALWAYS. Mt/Kt/Nt come from the
 *      descriptor's parameter block, which all three read; no loop bound here depends on a
 *      wait. That is what makes rule 3's DRAIN safe: when UNPACK gives up on a
 *      `cb_wait_front`, it keeps issuing the remaining unpacks and pops, MATH keeps issuing
 *      its matmuls, and PACK keeps its commit/wait/release pairs -- so the hardware handshakes
 *      between them stay balanced and every thread reaches the end of the descriptor together.
 *
 *   2. RETURNING EARLY IS NOT AN OPTION HERE, IT IS THE FAILURE. The DST-available wait, the
 *      packer's wait-for-math and the unpacker's dvalid are conditions the RISC-V has ALREADY
 *      handed to the hardware, so no poll of the stop flag can reach them. If UNPACK returned
 *      from the middle of the K loop, MATH's next MVMUL would stall in the backend and the
 *      MATH RISC-V would then stall pushing into a full instruction FIFO -- a hang no host
 *      write can retire, which is exactly the class of wedge this phase exists to make
 *      impossible.
 *
 *   3. THE COMPUTE CURSOR FOLLOWS `head`, NOT `tail`. Compute must be free to start a
 *      descriptor while the reader is still filling it -- that is what the CB pipeline is for
 *      -- so it takes work as soon as the host publishes it and blocks in the CB, which is the
 *      right place to block. Only the WRITER's cursor follows `tail`.
 *
 *   4. `kMatmulColumnBlock` MAKES DST HOLD NB TILES INSTEAD OF ONE. The reader supplies NB
 *      output columns per pass so it fetches the A column Nt/NB times instead of Nt; this
 *      thread is what consumes that -- one in0 unpack feeds `ct_dim = NB` MVMUL groups
 *      writing NB DST tiles, and SrcB is cleared once at the end of the block rather than
 *      once per tile (ft_tensix_math_matmul_block). THE CEILING IS THE DST SECTION: SyncHalf
 *      with `dst_accum_mode 1` is 4 tiles of 32x32 (measured), so NB > 4
 *      would silently write outside the half. CB_OUT is untouched -- one reserve, one pack,
 *      one push per tile, so the writer sees the same order it always has.
 *
 *      THE SECTION SIZE IS `FT_TT_MATMUL_DST_SYNC`'s TO SET, and this file reaches it
 *      through exactly three calls -- `ftmm_math_pack_sync_init`,
 *      `ftmm_math_dest_section_done`, `ftmm_pack_dest_section_done`
 *      (resident_matmul_dst_sync.h). At the default SyncHalf all three forward to the shared
 *      `ft_tensix_*` entry points, so the shipped instruction stream does not move; at
 *      SyncFull the section is 8 tiles and the MATH/PACK double buffer is gone. No loop, no
 *      DST index and no MOP below is aware of which one it is.
 *
 *   5. THE ROW BLOCK IS THE STOCK `rt_dim`, AND IT NEEDS NO NEW MOP, REPLAY BUFFER OR
 *      ADDR_MOD. the stock matmul math picks `reuse_a` whenever `ct_dim >= rt_dim`, and its
 *      reuse_a arm is literally `for t in rt_dim { set_dst_write_addr(dst + ct_dim*t + rut)
 *      over rut in ct_dim; clear SrcB }` -- which is `ft_tensix_math_matmul_block(dst +
 *      t*ct_dim, ct_dim)` called `rt_dim` times. the stock matmul unpack is the same shape:
 *      its per-`t` body is exactly `ft_tensix_unpack_ab_matmul()`, with operand A's tile
 *      index advanced by `t * kt_dim` and operand B's held. Here the K extent in L1 is
 *      `depth`, so the A index is `row * depth + step`. So MB is a LOOP on this thread and
 *      an L1 layout on the reader, and the two families of Tensix state this file programs
 *      once at startup are untouched.
 *
 *      MB AND NB SHARE THE DST SECTION: `MB * NB <= 4`. The reader refuses a descriptor that
 *      asks for more, and resident_matmul_block_protocol.h carries the upstream assert the
 *      bound comes from. What MB buys is on the READER's side of the pipe -- the weight
 *      column is swept once for MB output rows -- and what it costs is here: the weight
 *      tiles are re-unpacked into SrcA once per row, which is the same per-output-tile unpack
 *      count the ct_dim == 1 shape always paid.
 *
 * Compile-time args (positional):
 *   0: cb_in0 index (activation)   1: cb_in1 index (weight)   2: cb_out index
 *   3: wait mode -- 0 upstream (the cost arm), 1 stoppable (ships), 2 bound-only (the red
 *      proof for the ask)
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 * Defines: `-DFT_RTRIAD_MIXW=1` builds `rtriad_mixw` (CB table matmul_mixw.cbtable): per
 *   descriptor, word 13 bit 30 selects the Bfp4_b weight format and tile size of the table's
 *   cb 2 entry for SrcA; unpacker 0 is switched only when the format changes
 *   (matmul_select_weight_format). Exclusive with FT_RTRIAD_MOE (#error).
 *
 * Build:
 *   CTARGS=0,1,16,1 CB_TABLE=kernels/cb_tables/matmul_float16b.cbtable \
 *       bash foundry/tt/build_tensix_kernel.sh \
 *           foundry/tt/kernels/resident_matmul_compute.c /tmp/ttk_rtriad
 *
 * @status CONVERSION MEASURED, 2026-08-13 (plan Phase 6.6). NOT YET THE SHIPPED KERNEL --
 *   resident_matmul_compute.c is retained as the baseline until all three images are
 *   byte-identical. Against the C++ build: ALL THREE Tensix instruction streams IDENTICAL
 *   (trisc0 31/31, trisc1 47/47, trisc2 37/37 -- same words, same order), and trisc1's
 *   immediate multiset is identical at the same 0x6d0 (it differs only in register allocation
 *   and four commutative operand orders). trisc0 is 0xa60 vs 0xae0 and trisc2 0x9f0 vs 0xa00,
 *   both from GCC hoisting less in C; no store value and no transaction differs anywhere.
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

#include "resident_matmul_block_protocol.h"
#include "resident_matmul_dst_sync.h"
#include "resident_queue_protocol.h"
/* The FUSED-SWIGLU build (program `rtriad_bfp8_swiglu`): resident_matmul_swiglu_protocol.h. Every
 * line it adds is under this switch, so every other build preprocesses to the same tokens. */
#if defined(FT_RTRIAD_SWIGLU) && FT_RTRIAD_SWIGLU
#if (defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE) || (defined(FT_RTRIAD_MIXW) && FT_RTRIAD_MIXW)
#error "resident_matmul_compute.c: FT_RTRIAD_SWIGLU is its own program; build it without MOE/MIXW"
#endif
#include "resident_matmul_swiglu_compute.h"
#endif
/* S3a, the RESIDUAL-ADD descriptor (resident_matmul_resadd_protocol.h): built only into the
 * fused program, beside the SwiGLU code whose SrcA switch and panel pass it reuses. */
#if defined(FT_RTRIAD_RESADD) && FT_RTRIAD_RESADD
#if !defined(FT_RTRIAD_SWIGLU) || !FT_RTRIAD_SWIGLU
#error "resident_matmul_compute.c: FT_RTRIAD_RESADD is built into the fused SwiGLU program only"
#endif
#include "resident_matmul_resadd_compute.h"
#endif
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
#include "resident_matmul_moe_helpers.h"
#if defined(FT_TRISC_UNPACK)
#define FTMOE_WORKER 2u
#elif defined(FT_TRISC_MATH)
#define FTMOE_WORKER 3u
#else
#define FTMOE_WORKER 4u
#endif
/* EXPANSION HELPER: every TRISC takes its share of block b before consuming it. */
#define FTMOE_HELP_BLOCK(ctx) ((void)ftmoe_help((ctx), FTMOE_WORKER))
#else
#define FTMOE_HELP_BLOCK(ctx) ((void)0)
#endif

/* The MIXED-WEIGHT build (program `rtriad_mixw`). Every line it adds is under this switch, so
 * every other build's preprocessed source is token-identical to the one without it. */
#if defined(FT_RTRIAD_MIXW) && FT_RTRIAD_MIXW
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
#error "resident_matmul_compute.c: FT_RTRIAD_MIXW and FT_RTRIAD_MOE are two programs (rtriad_mixw, rtriad_bfp6); build one"
#endif
/** The CB table entry carrying the Bfp4_b weight's format and tile size. Format and size
 *  only: the host never places it; the weight still arrives through cb_in1's pages. */
#define FTMM_MIXW_ALT_CB 2u

#ifdef FT_TRISC_UNPACK
/** 1 while unpacker 0 (SrcA, the weight) is configured for the Bfp4_b format. */
static uint32_t mm_weight_bfp4 = 0u;

/** The weight's tile size in the compute cores' CB unit (16-byte words, as fifo_page_size). */
static uint32_t matmul_weight_page(uint32_t weight_bfp4, uint32_t cb_in1)
{
    return weight_bfp4 ? (uint32_t)ft_cb_unpack_tile_size[FTMM_MIXW_ALT_CB] >> 4
                       : ft_tensix_cb(cb_in1)->fifo_page_size;
}

/**
 * Point SrcA at this descriptor's weight format, LAZILY: only a change costs the switch
 * (both unpack contexts drained, unpacker 0 idle, two CFG RMWs, TILE_SIZE_A), measured below
 * resolution at 32 switches a run (qwen36 bench 1.4b). MATH and PACK need nothing: Bfp8_b and
 * Bfp4_b are one exponent family, unpacked to the same SrcA format, at the same fidelity.
 */
static void matmul_select_weight_format(uint32_t weight_bfp4, uint32_t cb_in1)
{
    const uint32_t cb = weight_bfp4 ? FTMM_MIXW_ALT_CB : cb_in1;

    if (weight_bfp4 == mm_weight_bfp4)
        return;
    mm_weight_bfp4 = weight_bfp4;
    ft_tensix_unpack_reconfigure_srca_format(ft_cb_unpack_src_format[cb], ft_cb_unpack_dst_format[cb],
                                             matmul_weight_page(weight_bfp4, cb_in1));
}
#endif
#endif

#define DST_REGISTER 0u

#ifdef FT_TRISC_MATH
#define MATMUL_HIGH_FIDELITY (FT_CB_MATH_FIDELITY != 0)
#define MATMUL_INNER_LOOPS (MATMUL_HIGH_FIDELITY ? (uint32_t)FT_CB_MATH_FIDELITY : 1u)
#define MATMUL_FIDELITY_INCREMENT (MATMUL_HIGH_FIDELITY ? 1u : 0u)
#endif

/* M5: which in0 shape the unpacker and the MOP are configured for. 0 = the full tile the
 * startup configured; 1 = the 8x32 partial-face tile. Switched only at a descriptor boundary,
 * where no unpack or MVMUL of the previous descriptor is pending a reconfiguration.
 * 1 = in0 as a 16x32 half tile (faces 0-1), num_faces 2: one UNPACR of 512 datums and the
 * 8-MVMUL MOP (ft_tensix_math_matmul_init_in0_16row). */
static uint32_t mm_in0_half = 0u;

static void matmul_select_in0_shape(uint32_t want, uint32_t cb_in0, uint32_t cb_in1)
{
    (void)cb_in0;
    (void)cb_in1;
    if (want == mm_in0_half)
        return;
    mm_in0_half = want;
    /* NO hw_configure re-run: the half tile is one UNPACR whose length is x_end alone, so the
     * tile descriptor's face count is not consulted (run 3 re-ran it; ruled out here). */
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul_init(
        0 /*transpose*/,
        (uint32_t)ft_cb_unpack_tile_num_faces[cb_in1] * ft_cb_unpack_tile_face_rows[cb_in1] *
                FT_TENSIX_FACE_C_DIM - 1u,
        (want ? 2u : (uint32_t)ft_cb_unpack_tile_num_faces[cb_in0]) *
                (uint32_t)ft_cb_unpack_tile_face_rows[cb_in0] * FT_TENSIX_FACE_C_DIM - 1u,
        1u /*kt_dim*/));
    if (want) {
        FT_TENSIX_MATH(ft_tensix_math_matmul_init_in0_16row(
            MATMUL_FIDELITY_INCREMENT, MATMUL_INNER_LOOPS, MATMUL_HIGH_FIDELITY));
    } else {
        FT_TENSIX_MATH(ft_tensix_math_matmul_init(0 /*transpose*/, MATMUL_FIDELITY_INCREMENT,
                                                  MATMUL_INNER_LOOPS, MATMUL_HIGH_FIDELITY));
    }
}

#if defined(FT_RTRIAD_SWIGLU) && FT_RTRIAD_SWIGLU
/**
 * Put back what the SILU_MUL half changed, so the next matmul block runs on exactly the state
 * the descriptor's prologue left: SrcA at the weight's format (which rewrites TILE_SIZE_A),
 * the column-block SrcA stride over it, then both matmul MOPs for the current in0 shape --
 * matmul_select_in0_shape()'s calls, unconditionally, since mm_in0_half has not changed.
 */
static void ftsw_matmul_restore(uint32_t cb_in0, uint32_t cb_in1, uint32_t depth,
                                uint32_t column_block)
{
    (void)cb_in0;
    (void)cb_in1;
    (void)depth;
    (void)column_block;
    FT_TENSIX_UNPACK(ftsw_unpack_srca_format(ft_cb_unpack_src_format[cb_in1],
                                             ft_cb_unpack_dst_format[cb_in1],
                                             ft_tensix_cb(cb_in1)->fifo_page_size));
    FT_TENSIX_UNPACK(if (column_block > 1u) ft_tensix_unpack_matmul_set_srca_tile_stride(
                         depth * ft_tensix_cb(cb_in1)->fifo_page_size));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul_init(
        0 /*transpose*/,
        (uint32_t)ft_cb_unpack_tile_num_faces[cb_in1] * ft_cb_unpack_tile_face_rows[cb_in1] *
                FT_TENSIX_FACE_C_DIM - 1u,
        (mm_in0_half ? 2u : (uint32_t)ft_cb_unpack_tile_num_faces[cb_in0]) *
                (uint32_t)ft_cb_unpack_tile_face_rows[cb_in0] * FT_TENSIX_FACE_C_DIM - 1u,
        1u /*kt_dim*/));
    if (mm_in0_half) {
        FT_TENSIX_MATH(ft_tensix_math_matmul_init_in0_16row(
            MATMUL_FIDELITY_INCREMENT, MATMUL_INNER_LOOPS, MATMUL_HIGH_FIDELITY));
    } else {
        FT_TENSIX_MATH(ft_tensix_math_matmul_init(0 /*transpose*/, MATMUL_FIDELITY_INCREMENT,
                                                  MATMUL_INNER_LOOPS, MATMUL_HIGH_FIDELITY));
    }
}

/**
 * One panel pass of a fused descriptor: compute_one_matmul's K loop for ONE output row (Mt =
 * MB = 1) over `columns` columns, packed bf16 into the PANEL CB instead of CB_OUT. Kept as
 * its own copy rather than a shared helper so the shipped compute_one_matmul is untouched
 * (every other program's image stays LOAD-IDENTICAL); rule 3 (drain) holds the same way --
 * every pop, DST handshake and push sits outside the waits.
 */
static void ftsw_matmul_pass(ftq_context *ctx, uint32_t cb_in0, uint32_t cb_in1,
                             uint32_t tile_inner, uint32_t depth, uint32_t column_block,
                             uint32_t columns, uint32_t wait_mode)
{
    const uint32_t cb_panel = FT_TT_MATMUL_SWIGLU_CB_PANEL;
    const uint32_t chunk = depth * column_block;
    uint32_t kt;
    uint32_t step;
    uint32_t column;

    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    for (kt = 0; kt < tile_inner; kt += depth) {
        (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in0, (int32_t)depth, wait_mode);
        (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in1, (int32_t)chunk, wait_mode);
        for (step = 0; step < depth; step++) {
            FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul(cb_in0, cb_in1, step, step, columns));
            FT_TENSIX_MATH(ft_tensix_math_matmul_block(DST_REGISTER, columns));
        }
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in0, depth));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in1, chunk));
    }
    FT_TENSIX_MATH(ftmm_math_dest_section_done());
    FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
    for (column = 0; column < columns; column++) {
        (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb_panel, 1, wait_mode);
        FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER + column, cb_panel));
        FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_panel, 1));
    }
    FT_TENSIX_PACK(ftmm_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
    /* PAD PAGES: a short pass still occupies NB pages, so every block stays on the ring's
     * 2*NB grid and the indexed unpacks never run past the CB's end (protocol header). */
    for (column = columns; column < ftsw_panel_pass_pages(column_block); column++) {
        (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb_panel, 1, wait_mode);
        FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_panel, 1));
    }
}

/**
 * The compute half of one FUSED descriptor (resident_matmul_swiglu_protocol.h): per column
 * block of w columns, the gate pass and the up pass into the panel CB (NB pages each, pad pages
 * after a short pass: the reader's pass order), SrcA to bf16, eltwise SILU_MUL on each pair
 * (j, NB + j) into
 * CB_OUT, the panel popped, SrcA and the matmul MOPs put back. Trip counts come from the
 * descriptor alone (rule 1); a waits-abandoned descriptor still runs every handshake.
 *
 * DST: a pass holds w <= NB tiles (the reader's ftmm_block_fits_dst bound), the SILU_MUL half
 * one tile (the RED arm two), each committed before the next is acquired.
 */
static void compute_one_swiglu(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                               uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out,
                               uint32_t wait_mode, uint32_t depth, uint32_t in0_half)
{
    const uint32_t tile_inner = parameters[kMatmulTileInner];
    const uint32_t tile_cols = parameters[kMatmulTileCols];
    const uint32_t block_word = parameters[kMatmulColumnBlock];
    const uint32_t column_block = ftmm_column_block(block_word);
    const uint32_t passes = ftsw_passes(tile_cols, column_block);
    const bool row0 = (block_word & FT_TT_MATMUL_SWIGLU_SILU_ROW0) != 0u;
    const bool red_no_stage = (block_word & FT_TT_MATMUL_SWIGLU_RED_NO_STAGE) != 0u;
    const uint32_t cb_panel = FT_TT_MATMUL_SWIGLU_CB_PANEL;
    uint32_t pass;
    uint32_t j;

    (void)in0_half;
    for (pass = 0; pass < passes; pass += 2u) {
        const uint32_t columns = ftsw_pass_columns(pass, tile_cols, column_block);

        ftsw_matmul_pass(ctx, cb_in0, cb_in1, tile_inner, depth, column_block, columns,
                         wait_mode);
        ftsw_matmul_pass(ctx, cb_in0, cb_in1, tile_inner, depth, column_block, columns,
                         wait_mode);

        (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_panel,
                                          (int32_t)ftsw_panel_pages(column_block), wait_mode);
        FT_TENSIX_UNPACK(ftsw_unpack_srca_format(ft_cb_unpack_src_format[cb_panel],
                                                 ft_cb_unpack_dst_format[cb_panel],
                                                 ft_tensix_cb(cb_panel)->fifo_page_size));
        for (j = 0; j < columns; j++) {
            if (red_no_stage)
                ftsw_silu_multiply_unstaged(ctx, ftsw_panel_gate_index(j),
                                            ftsw_panel_up_index(j, column_block), cb_out,
                                            wait_mode);
            else
                ftsw_silu_gated_multiply(ctx, ftsw_panel_gate_index(j),
                                         ftsw_panel_up_index(j, column_block), cb_out, row0,
                                         wait_mode);
        }
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_panel, ftsw_panel_pages(column_block)));
        ftsw_matmul_restore(cb_in0, cb_in1, depth, column_block);
    }
}
#endif

#if defined(FT_RTRIAD_RESADD) && FT_RTRIAD_RESADD
/**
 * The compute half of one RESADD descriptor (resident_matmul_resadd_protocol.h): per column
 * block of w columns, the projection's pass into the panel CB (ftsw_matmul_pass: the same pack
 * of the same DST the unfused matmul writes to `project`) padded to the whole 2*NB block, the
 * reader's NB residual pages, SrcA to bf16, eltwise ADD on each (residual j, panel j) into
 * CB_OUT, both CBs popped, SrcA and the matmul MOPs put back. Trip counts come from the
 * descriptor alone (rule 1).
 */
static void compute_one_resadd(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                               uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out,
                               uint32_t wait_mode, uint32_t depth)
{
    const uint32_t tile_inner = parameters[kMatmulTileInner];
    const uint32_t tile_cols = parameters[kMatmulTileCols];
    const uint32_t block_word = parameters[kMatmulColumnBlock];
    const uint32_t column_block = ftmm_column_block(block_word);
    const uint32_t blocks = ftra_blocks(tile_cols, column_block);
    const bool red_skip = (block_word & FT_TT_MATMUL_RESADD_RED_SKIP) != 0u;
    const uint32_t cb_panel = FT_TT_MATMUL_SWIGLU_CB_PANEL;
    const uint32_t cb_residual = FT_TT_MATMUL_RESADD_CB_RESIDUAL;
    uint32_t block;
    uint32_t page;

    for (block = 0; block < blocks; block++) {
        const uint32_t columns = ftra_block_columns(block, tile_cols, column_block);

        ftsw_matmul_pass(ctx, cb_in0, cb_in1, tile_inner, depth, column_block, columns,
                         wait_mode);
        /* The block's second NB panel pages are pad: the SwiGLU's 2*NB grid (protocol). */
        for (page = ftsw_panel_pass_pages(column_block);
             page < ftra_panel_block_pages(column_block); page++) {
            (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb_panel, 1, wait_mode);
            FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_panel, 1));
        }
        (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_panel,
                                          (int32_t)ftra_panel_block_pages(column_block), wait_mode);
        (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_residual,
                                          (int32_t)ftra_residual_block_pages(column_block),
                                          wait_mode);
        FT_TENSIX_UNPACK(ftsw_unpack_srca_format(ft_cb_unpack_src_format[cb_panel],
                                                 ft_cb_unpack_dst_format[cb_panel],
                                                 ft_tensix_cb(cb_panel)->fifo_page_size));
        ftra_add_block(ctx, columns, cb_out, red_skip, wait_mode);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_panel,
                                                        ftra_panel_block_pages(column_block)));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_residual,
                                                        ftra_residual_block_pages(column_block)));
        ftsw_matmul_restore(cb_in0, cb_in1, depth, column_block);
    }
}
#endif

/**
 * The compute half of one matmul descriptor: matmul_compute.c's loop with stoppable CB waits
 * and the drain rule. Note that every pop, every DST handshake and every push below is OUTSIDE
 * the `if` that guards the waits -- rule 3 in code form.
 */
static void compute_one_matmul(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                               uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out,
                               uint32_t wait_mode)
{
    const uint32_t tile_rows = parameters[kMatmulTileRows];
    const uint32_t tile_inner = parameters[kMatmulTileInner];
    const uint32_t tile_cols = parameters[kMatmulTileCols];
    const uint32_t requested_depth = parameters[kMatmulDepth];
    const uint32_t block_word = parameters[kMatmulColumnBlock];
    /* THE SAME COLLAPSES THE READER MAKES, from the same words. Neither thread may ask
     * the other what it decided -- rule 1 is that all three TRISC trip counts come from the
     * descriptor -- so the arithmetic is repeated, not shared. */
    const uint32_t depth = (requested_depth > 0u && tile_inner % requested_depth == 0u)
                               ? requested_depth
                               : 1u;
    const uint32_t column_block = ftmm_column_block(block_word);
    const uint32_t row_block = ftmm_row_block(block_word);
    const uint32_t chunk = depth * column_block;
    const uint32_t a_chunk = depth * row_block;
    const bool probe_no_compute = (parameters[kMatmulBNMajor] & kMatmulProbeNoCompute) != 0u;
    /* Only the PIPELINED reader (bit 8) writes the compact page, so both bits or neither. */
    /* bit 17 = DIAGNOSTIC: the reader takes the row-0 path but the compute stays full. */
    const uint32_t in0_half = ((parameters[kMatmulBNMajor] & kMatmulRow0) != 0u &&
                               (parameters[kMatmulBNMajor] & (1u << 17)) == 0u &&
                               (parameters[kMatmulBNMajor] & (1u << 8)) != 0u &&
                               tile_rows == 1u && row_block == 1u)
                                  ? 1u
                                  : 0u;
    uint32_t mt;
    uint32_t nt;
    uint32_t kt;
    uint32_t step;
    uint32_t column;
    uint32_t row;

    /* The MOP walks SrcA one `TILE_SIZE_A` at a time between the block's columns, and the
     * reader lands them COLUMN-MAJOR -- column j at tile `j*depth` -- so the stride is
     * `depth` tiles. Set per descriptor because `depth` is per descriptor, and only when it
     * differs from the tile size hw_configure already left there. */
#if defined(FT_RTRIAD_MIXW) && FT_RTRIAD_MIXW
    /* The switch first: it rewrites TILE_SIZE_A, and the column-block stride overrides it. */
    FT_TENSIX_UNPACK(const uint32_t weight_bfp4 =
                         (parameters[kMatmulBNMajor] & (uint32_t)kMatmulWeightBfp4) != 0u ? 1u : 0u;
                     const uint32_t weight_page = matmul_weight_page(weight_bfp4, cb_in1);
                     matmul_select_weight_format(weight_bfp4, cb_in1));
    FT_TENSIX_UNPACK(if (column_block > 1u)
                         ft_tensix_unpack_matmul_set_srca_tile_stride(depth * weight_page));
#else
    FT_TENSIX_UNPACK(if (column_block > 1u) ft_tensix_unpack_matmul_set_srca_tile_stride(
                         depth * ft_tensix_cb(cb_in1)->fifo_page_size));
#endif

    matmul_select_in0_shape(in0_half, cb_in0, cb_in1);

#if defined(FT_RTRIAD_SWIGLU) && FT_RTRIAD_SWIGLU
    if ((block_word & FT_TT_MATMUL_SWIGLU_FLAG) != 0u) {
        compute_one_swiglu(ctx, parameters, cb_in0, cb_in1, cb_out, wait_mode, depth, in0_half);
        return;
    }
#endif
#if defined(FT_RTRIAD_RESADD) && FT_RTRIAD_RESADD
    if ((block_word & FT_TT_MATMUL_RESADD_FLAG) != 0u) {
        compute_one_resadd(ctx, parameters, cb_in0, cb_in1, cb_out, wait_mode, depth);
        return;
    }
#endif

    if (probe_no_compute) {
        /* M5 PROBE: the same CB traffic, no unpack, no math, no pack. Output pages carry
         * whatever L1 held. */
        for (mt = 0; mt < tile_rows; mt += row_block) {
            const uint32_t rows = (tile_rows - mt) < row_block ? (tile_rows - mt) : row_block;

            for (nt = 0; nt < tile_cols; nt += column_block) {
                const uint32_t columns =
                    (tile_cols - nt) < column_block ? (tile_cols - nt) : column_block;

                for (kt = 0; kt < tile_inner; kt += depth) {
                    FTMOE_HELP_BLOCK(ctx);
                    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in0, (int32_t)a_chunk,
                                                      wait_mode);
                    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in1, (int32_t)chunk,
                                                      wait_mode);
                    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in0, a_chunk));
                    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in1, chunk));
                }
                for (row = 0; row < rows * columns; row++) {
                    (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb_out, 1, wait_mode);
                    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, 1));
                }
            }
        }
        return;
    }

    for (mt = 0; mt < tile_rows; mt += row_block) {
        const uint32_t rows = (tile_rows - mt) < row_block ? (tile_rows - mt) : row_block;

        for (nt = 0; nt < tile_cols; nt += column_block) {
            const uint32_t columns =
                (tile_cols - nt) < column_block ? (tile_cols - nt) : column_block;

            /* tile_regs_acquire(): MATH blocks until DST is free. It does NOT zero DST -- PACK
             * zeroes per tile with TT_ZEROACC, which names ADDR_MOD_1, a slot the MATH thread
             * programs, so the zeroing is cross-thread and invisible here. */
            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());

            for (kt = 0; kt < tile_inner; kt += depth) {
                FTMOE_HELP_BLOCK(ctx);
                (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in0, (int32_t)a_chunk,
                                                  wait_mode);
                (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_in1, (int32_t)chunk, wait_mode);

                for (step = 0; step < depth; step++) {
                    /* THE UNPACK AND THE MVMUL ARE ISSUED EVEN WHILE ABORTING, and that is
                     * the whole of rule 3 in two lines. Skipping them on the abort path --
                     * which is the obvious, careful-looking thing to do, and is what the
                     * first version did -- stops the unpacker feeding SrcA/SrcB dvalid, so
                     * MATH's next MVMUL stalls IN THE BACKEND and the MATH RISC-V then
                     * stalls pushing into a full instruction FIFO. That is a hang no stop
                     * flag and no spin bound can reach, and it wedged a core until reboot:
                     * the ask stopped BRISC, NCRISC, TRISC0 and TRISC2, and left TRISC1
                     * stalled forever.
                     *
                     * Issuing them on garbage data is harmless -- the CB pages are real L1
                     * and the numbers are discarded, because the descriptor is not being
                     * retired.
                     *
                     * ONE in0 UNPACK FEEDS `columns` MVMUL GROUPS. in0 goes to SrcB and is
                     * the operand held across the block; the MOP unpacks `columns` SrcA
                     * tiles behind it, and the MATH side clears SrcB once at the end rather
                     * than once per tile. `columns` is short only on a ragged last block --
                     * the CB chunk stays `depth * NB` either way.
                     *
                     * AND ONE PASS OF THE INNER LOOP PER OUTPUT ROW, which is upstream's
                     * `rt_dim`: the in1 tiles are unpacked again for every row while the in0
                     * tile MOVES, and the DST base steps by `columns`.
                     * the stock math reuse_a arm IS this loop and
                     * the stock unpack `t` loop IS this call -- with operand A's
                     * tile index advanced by `t * kt_dim`, which in L1 is `row * depth`. At
                     * MB = 1 it runs once at index `step` and DST base 0, which is the
                     * pre-MB instruction stream. */
                    for (row = 0; row < rows; row++) {
#if defined(FT_RTRIAD_MIXW) && FT_RTRIAD_MIXW
                        /* B (the weight) addressed at this descriptor's tile size. */
                        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul_tile_b(
                            cb_in0, cb_in1, row * depth + step, step, columns, weight_page));
#else
                        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul(
                            cb_in0, cb_in1, row * depth + step, step, columns));
#endif
                        FT_TENSIX_MATH(
                            ft_tensix_math_matmul_block(DST_REGISTER + row * columns, columns));
                    }
                }

                FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in0, a_chunk));
                FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_in1, chunk));
            }

            FT_TENSIX_MATH(ftmm_math_dest_section_done()); /* tile_regs_commit() */

            /* ONE tile_regs_wait() for the SECTION, then the block's tiles one page at a
             * time. CB_OUT keeps its two pages and its one-page pushes.
             *
             * ROW-MAJOR WITHIN THE BLOCK, because that is the order the DST indices were
             * assigned in: `row * columns + column`. At MB = 1 the outer loop runs once and
             * the sequence is `0 .. columns-1`, exactly the order the writer has always
             * seen -- which is why a decode descriptor's CB_OUT stream does not move. */
            FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done()); /* tile_regs_wait() */
            for (row = 0; row < rows; row++) {
                for (column = 0; column < columns; column++) {
                    (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb_out, 1, wait_mode);
                    FT_TENSIX_PACK(
                        ft_tensix_pack_tile(DST_REGISTER + row * columns + column, cb_out));
                    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, 1));
                }
            }
            FT_TENSIX_PACK(ftmm_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
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
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
    ftmoe_boot();
#endif

    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    /* compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out): in0 goes to SrcB
     * and in1 to SrcA, so src_a is cb_in1 in every call below. Done ONCE, outside the queue
     * loop -- the hardware configuration is a property of the binary, not of a descriptor,
     * which is also why one resident binary serves one opcode set rather than one op
     * (docs/tt_bringup_log.md 17.8 note 2). */
    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[cb_in1], ft_cb_unpack_src_format[cb_in0], ft_cb_unpack_dst_format[cb_in1],
        ft_cb_unpack_dst_format[cb_in0], ft_cb_unpack_tile_face_rows[cb_in1],
        ft_cb_unpack_tile_face_rows[cb_in0], ft_cb_unpack_tile_num_faces[cb_in1],
        ft_cb_unpack_tile_num_faces[cb_in0], ft_tensix_cb(cb_in1)->fifo_page_size,
        ft_tensix_cb(cb_in0)->fifo_page_size));
#if defined(FT_RTRIAD_MIXW) && FT_RTRIAD_MIXW
    FT_TENSIX_UNPACK(mm_weight_bfp4 = 0u); /* hw_configure just set cb_in1's (Bfp8_b) format */
#endif

    FT_TENSIX_MATH(ftmm_math_pack_sync_init());
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

    /* All five resume from completed_count -- see resident_matmul_reader.c for why, and for
     * the refusal that enforces it. */
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
        /* Every other opcode is the reader's alone and costs this thread one cursor bump. */

        if (!ctx.aborting)
            ctx.cursor++;
    }

    ftq_status_set(&ctx, kStateExit);
}
