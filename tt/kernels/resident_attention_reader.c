/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_attention_reader.c
 * @brief Reader third of the RESIDENT FLASH-DECODE ATTENTION triad: the group's Q, packed into
 * `dim_tiles` tiles whose rows are Q heads; the four constant tiles this descriptor
 * selects; and then the KV cache, ONE BLOCK of K and one of V at a time, each block a
 * SINGLE contiguous `noc_async_read`.
 *
 * THE KV BLOCK IS ONE TRANSACTION AND THAT IS THE WHOLE POINT OF THE LAYOUT.
 * ft_tt_op_memory.h PART 1 §1.2: positions [32a, 32b) of one head are one contiguous byte
 * range, walked linearly in exactly the order the matmul reader already wants. So a block
 * of `block_tiles` position tiles is `block_tiles * dim_tiles` consecutive tiles, i.e.
 * ONE read and ONE barrier -- the shape that was worth 5.9x on the matmul
 * (docs/tt_kernel_model.md §12) rather than a strided walk.
 *
 * THE Q PACK IS THE ONE PIECE OF REAL WORK THE READER DOES.
 * Q arrives in the model's activation layout -- head h's dim tile d at tile column
 * `h*dim_tiles + d`, the decode token at row `position % 32` -- and the compute thread
 * wants ONE tile per dim tile whose ROW h is head `q_first + h`. The group's tiles are
 * consecutive, so the whole set is one read into a staging buffer, and the pack is
 * `q_rows * dim_tiles * 2` copies of a 32-byte face row. It reads 64 KiB to use 2 KiB of
 * it, and that is the right trade: it is ~0.4% of one layer's KV traffic at 8k context,
 * and the alternative -- a 32-byte strided DRAM read per face row -- is not legal at all
 * (64 B read alignment, ft_tt_noc_alignment()).
 *
 * THIS PROCESSOR OWNS `tail`, AND NOTHING ELSE IN THE CONTROL BLOCK.
 *
 * SPLIT-K (-DATTN_SPLITK, resident_attention_protocol.h §7): a PARTIAL reads exactly as a
 * decode / ROWS descriptor (the host folds the chunk into K_BASE / V_BASE); a MERGE reads the
 * group's partials instead of Q, constants and KV (read_partials).
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_Q index   1: CB_K index   2: CB_V index   3: CB_CONST index
 *   4: CB_QSTAGE index (reader-private staging; never pushed, never popped)
 *   5: tile size in bytes
 *   6: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0,1,2,3,13,2048,1 NOC=0 KERNEL_TEXT_BASE=0x15000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_attention_reader.c brisc /tmp/ttk_attn
 *
 * @status GREEN ON SILICON on Tensix 70/71/72/73, 0 checks failed.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_queue_protocol.h"
#include "resident_precondition.h"
#include "resident_attention_protocol.h"

#define CB_Q ft_tensix_compile_time_arg(0)
#define CB_K ft_tensix_compile_time_arg(1)
#define CB_V ft_tensix_compile_time_arg(2)
#define CB_CONST ft_tensix_compile_time_arg(3)
#define CB_QSTAGE ft_tensix_compile_time_arg(4)
#define TILE_BYTES ft_tensix_compile_time_arg(5)
#define WAIT_MODE ft_tensix_compile_time_arg(6)
/* The K/V CACHE tile of `cb` (CB_K or CB_V): block float in the attention_bfp8 / attention_bfp4
 * builds (KERNEL_DEFINES=-DATTN_KV_TILE_BYTES=1088 / 576, ft_tt_op_memory.h §1.6), bf16
 * otherwise. The MIXED builds (attention_k8v4, _k16v4, _k4v16, _k16v8) give K and V their own
 * tile: -DATTN_K_TILE_BYTES=<K> -DATTN_V_TILE_BYTES=<V>. */
#if defined(ATTN_K_TILE_BYTES) && defined(ATTN_V_TILE_BYTES)
#define KV_TILE_BYTES_OF(cb) \
    ((cb) == CB_V ? (uint32_t)ATTN_V_TILE_BYTES : (uint32_t)ATTN_K_TILE_BYTES)
#elif defined(ATTN_KV_TILE_BYTES)
#define KV_TILE_BYTES_OF(cb) ATTN_KV_TILE_BYTES
#else
#define KV_TILE_BYTES_OF(cb) TILE_BYTES
#endif
/**
 * Refuse a descriptor the circular buffers this residency was STAGED with cannot serve.
 *
 * The page counts are fixed when the resident program is staged; the shape is a
 * per-descriptor word. So the host CAN ask for a block bigger than the buffer -- it did, on
 * another op's first hardware run, and the result was ten seconds of blocked polling
 * reported as an idle timeout (resident_topk_reader.c). And the two block-prefetch
 * conditions are the §5 invariant: `pages % depth == 0` so every wrap lands exactly on the
 * limit, and `tile_count % pages == 0` so the descriptor leaves the write pointer where it
 * found it. Without the second, a later descriptor's block steps OVER the limit and walks
 * up L1 -- measured, on this family, landing on the queue control block with nothing
 * faulting.
 */
static bool descriptor_is_executable(ftq_context *ctx, const fta_shape *shape)
{
    const uint32_t depth = shape->block_tiles * shape->dim_tiles;
    const uint32_t k_pages = ft_tensix_cb(CB_K)->fifo_num_pages;
    const uint32_t v_pages = ft_tensix_cb(CB_V)->fifo_num_pages;
    const uint32_t q_pages = ft_tensix_cb(CB_Q)->fifo_num_pages;
    const uint32_t stage_pages = ft_tensix_cb(CB_QSTAGE)->fifo_num_pages;
    const uint32_t const_pages = ft_tensix_cb(CB_CONST)->fifo_num_pages;
    const uint32_t tile_count = shape->padded_tiles * shape->dim_tiles;

    /* CHUNK mode reads the Q tiles straight into CB_Q, so it never touches the staging
     * buffer and must not be measured against it: at C = 32 and dim_tiles = 8 the decode
     * residency's 32 staging pages would refuse every chunk descriptor for a buffer the
     * descriptor does not use. */
    const uint32_t stage_needed = fta_is_chunk(shape) ? 0u : shape->q_rows * shape->dim_tiles;

    if (!fta_shape_valid(shape) || TILE_BYTES == 0u || depth == 0u)
        ctx->error = kErrBadParameters;
    else if (k_pages == 0u || v_pages == 0u || q_pages < shape->dim_tiles ||
             const_pages < kAttnCbConstPages || stage_pages < stage_needed)
        ctx->error = kErrBadParameters;
    else if (depth > k_pages || (k_pages % depth) != 0u || (tile_count % k_pages) != 0u)
        ctx->error = kErrBadParameters;
    else if (depth > v_pages || (v_pages % depth) != 0u || (tile_count % v_pages) != 0u)
        ctx->error = kErrBadParameters;
    else
        return true;
    return false;
}

/**
 * Read `tiles` consecutive K or V tiles at `base + first*KV_TILE_BYTES_OF(cb)` into `cb` as ONE
 * block.
 *
 * @return false if a reservation was abandoned, or if the length came out zero -- which is
 *         a refusal and never a skipped transfer (§6)
 */
static bool read_block(ftq_context *ctx, uint32_t cb, uint32_t x, uint32_t y, uint32_t base,
                       uint32_t first, uint32_t tiles)
{
    const uint32_t bytes = tiles * KV_TILE_BYTES_OF(cb);

    if (!fta_length_is_issuable(bytes)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, (int32_t)tiles, WAIT_MODE))
        return false;

    // Rule 3: while draining, issue NOTHING. The push still happens so the page counters
    // stay balanced, but a transaction issued during a drain would still be in flight when
    // the kernel returns.
    if (!ctx->aborting) {
        ft_tensix_noc_read(ft_tensix_noc_address(x, y, base + first * KV_TILE_BYTES_OF(cb)),
                           ft_tensix_cb_write_pointer(cb), bytes);
        // ONE barrier for the whole block: it is per-core, not per-transaction.
        ft_tensix_noc_read_barrier();
    }
    ft_tensix_cb_push_back(cb, tiles);
    return true;
}

/**
 * The four constants this descriptor uses, in the order the compute thread indexes them:
 * the reduce scaler, the 1/sqrt(head_dim) scale, the tail mask and the all-dead mask.
 *
 * Two of the four are adjacent in the operand so they are one transaction; the two masks
 * are selected by `kv_positions % 32` and by nothing else, which is why the mask INDEX is
 * derived here rather than sent (a second field that must agree with the first is a second
 * field that can disagree).
 */
static bool read_constants(ftq_context *ctx, const fta_shape *shape, uint32_t x, uint32_t y,
                           uint32_t base, uint32_t rows_mask)
{
    /* The TAIL slot carries the causal tile in chunk mode and the tail mask otherwise --
     * one function, resident_attention_protocol.h, so this and the off-hardware suite
     * cannot spell the selection differently. */
    const uint32_t tail_mask = fta_tail_slot_source_tile(shape);
    const uint32_t bytes = kAttnCbConstPages * TILE_BYTES;

    if (!fta_length_is_issuable(bytes)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_CONST,
                                       (int32_t)kAttnCbConstPages, WAIT_MODE))
        return false;

    if (!ctx->aborting) {
        const uint32_t write = ft_tensix_cb_write_pointer(CB_CONST);

        /* Tiles 0 and 1 are adjacent in the operand: one transaction. */
        ft_tensix_noc_read(ft_tensix_noc_address(x, y, base + kAttnConstScalerTile * TILE_BYTES),
                           write,
                           2u * TILE_BYTES);
        ft_tensix_noc_read(ft_tensix_noc_address(x, y,
                                                 (shape->flags & (uint32_t)kAttnFlagRows) != 0u
                                                     ? rows_mask
                                                     : base + tail_mask * TILE_BYTES),
                           write + kAttnCbConstTailMask * TILE_BYTES, TILE_BYTES);
        ft_tensix_noc_read(ft_tensix_noc_address(x, y, base + kAttnConstMaskTile * TILE_BYTES),
                           write + kAttnCbConstDeadMask * TILE_BYTES, TILE_BYTES);
        ft_tensix_noc_read_barrier();
    }
    ft_tensix_cb_push_back(CB_CONST, kAttnCbConstPages);
    return true;
}

/**
 * Build the packed Q: `dim_tiles` tiles whose row h is Q head `q_first + h`.
 *
 * The group's `q_rows * dim_tiles` Q tiles are consecutive in the activation buffer, so
 * they arrive in ONE transaction; the pack is then a face-row scatter inside L1. Source
 * tile (h, dt) is at staging index `h * dim_tiles + dt` and contributes its row
 * `query_row` to row h of packed tile dt.
 *
 * IN CHUNK MODE THERE IS NO PACK AT ALL — see the branch below.
 */
static bool read_and_pack_q(ftq_context *ctx, const fta_shape *shape, uint32_t x, uint32_t y,
                            uint32_t base)
{
    /* Chunk mode reads ONE head's `dim_tiles` tiles; decode reads the whole group's
     * `q_rows * dim_tiles` into staging and packs one row out of each. */
    const uint32_t source_tiles =
        fta_is_chunk(shape) ? shape->dim_tiles : shape->q_rows * shape->dim_tiles;
    const uint32_t bytes = source_tiles * TILE_BYTES;
    const uint32_t chunk = fta_chunk_bytes(TILE_BYTES);
    uint32_t stage;
    uint32_t packed;

    if (!fta_length_is_issuable(bytes) || !fta_length_is_issuable(chunk)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_Q, (int32_t)shape->dim_tiles, WAIT_MODE))
        return false;

    /* CHUNK MODE: THERE IS NOTHING TO PACK. The activation tile already has position t in
     * row t, which IS the packed Q -- `dim_tiles` tiles whose rows are what the score
     * matmul's M axis should be. One contiguous read, no staging buffer, no face-row
     * copies: 16 transactions a layer instead of 64 KiB read to use 2 KiB of it. */
    if (fta_is_chunk(shape)) {
        if (!ctx->aborting) {
            ft_tensix_noc_read(ft_tensix_noc_address(x, y, base),
                               ft_tensix_cb_write_pointer(CB_Q), bytes);
            ft_tensix_noc_read_barrier();
        }
        ft_tensix_cb_push_back(CB_Q, shape->dim_tiles);
        return true;
    }

    stage = ft_tensix_cb_write_pointer(CB_QSTAGE);
    packed = ft_tensix_cb_write_pointer(CB_Q);
    if (!ctx->aborting) {
        ft_tensix_noc_read(ft_tensix_noc_address(x, y, base), stage, bytes);
        ft_tensix_noc_read_barrier();

        for (uint32_t dt = 0; dt < shape->dim_tiles; dt++) {
            for (uint32_t p = 0; p < shape->positions; p++) {
                for (uint32_t head = 0; head < shape->q_rows; head++) {
                    const uint32_t source_tile = (head * shape->dim_tiles + dt) * TILE_BYTES;

                    for (uint32_t half = 0; half < 2u; half++) {
                        const uint32_t from = stage + source_tile +
                                              fta_chunk_offset(shape->query_row + p, half,
                                                               TILE_BYTES);
                        const uint32_t to = packed + dt * TILE_BYTES +
                                            fta_chunk_offset(p * shape->q_rows + head, half,
                                                             TILE_BYTES);

                        fta_copy_words((uint32_t *)(uintptr_t)(to),
                                       (const uint32_t *)(uintptr_t)(from), chunk / 4u);
                    }
                }
            }
        }
    }
    ft_tensix_cb_push_back(CB_Q, shape->dim_tiles);
    return true;
}

#ifdef ATTN_SPLITK
/** Read `tiles` consecutive bf16 tiles at `address` into `cb` as one block. */
static bool read_tiles(ftq_context *ctx, uint32_t cb, uint32_t x, uint32_t y, uint32_t address,
                       uint32_t tiles)
{
    const uint32_t bytes = tiles * TILE_BYTES;

    if (!fta_length_is_issuable(bytes)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, (int32_t)tiles, WAIT_MODE))
        return false;
    if (!ctx->aborting) {
        ft_tensix_noc_read(ft_tensix_noc_address(x, y, address), ft_tensix_cb_write_pointer(cb),
                           bytes);
        ft_tensix_noc_read_barrier();
    }
    ft_tensix_cb_push_back(cb, tiles);
    return true;
}

/**
 * The reader half of a MERGE descriptor (resident_attention_protocol.h §7): per partial, in
 * order, its (m, l) pair and then its O tiles.
 *
 * ONLY CB_Q AND CB_CONST, because a CB must be pushed by one processor for the life of the
 * program (gdn_float32.cbtable: a compute push stores over the counter a dataflow push
 * increments) and these are the two bf16 CBs the reader already owns. O goes in as CB_Q's
 * own dim_tiles block. The pairs go in as 2-page pushes, and an odd count gets one spare pair
 * (the last one again; the compute thread drops it) so a descriptor pushes CB_CONST a whole
 * number of 4-page laps: the next decode descriptor's 4-page constants read then starts on
 * the grid, which is §5's condition.
 */
static bool read_partials(ftq_context *ctx, const fta_shape *shape, uint32_t x, uint32_t y,
                          uint32_t base)
{
    const uint32_t stride = fta_partial_tiles(shape->dim_tiles) * TILE_BYTES;
    const uint32_t stats = shape->dim_tiles * TILE_BYTES;
    const uint32_t partials = shape->kv_positions;

    for (uint32_t i = 0; i < partials; i++) {
        if (!read_tiles(ctx, CB_CONST, x, y, base + i * stride + stats, kAttnPartialStatTiles))
            return false;
        if (!read_tiles(ctx, CB_Q, x, y, base + i * stride, shape->dim_tiles))
            return false;
    }
    if ((partials % 2u) != 0u &&
        !read_tiles(ctx, CB_CONST, x, y, base + (partials - 1u) * stride + stats,
                    kAttnPartialStatTiles))
        return false;
    return !ctx->aborting;
}

/** A MERGE descriptor's buffers: CB_Q takes whole dim_tiles blocks, CB_CONST whole pairs. */
static bool merge_is_executable(ftq_context *ctx, const fta_shape *shape)
{
    const uint32_t q_pages = ft_tensix_cb(CB_Q)->fifo_num_pages;
    const uint32_t const_pages = ft_tensix_cb(CB_CONST)->fifo_num_pages;

    if (!fta_shape_valid(shape) || TILE_BYTES == 0u || q_pages < shape->dim_tiles ||
        (q_pages % shape->dim_tiles) != 0u || const_pages < kAttnPartialStatTiles ||
        (const_pages % kAttnPartialStatTiles) != 0u ||
        ((((shape->kv_positions + 1u) / 2u) * 2u * kAttnPartialStatTiles) % const_pages) != 0u) {
        ctx->error = kErrBadParameters;
        return false;
    }
    return true;
}
#endif

/**
 * The reader half of one ATTENTION descriptor.
 *
 * Every trip count comes from the parameter block and none from a wait, which is what makes
 * the drain safe (resident_queue_protocol.h rule 3).
 *
 * @return true if the whole descriptor was read; false if a wait was abandoned, in which
 *         case the loop still ran to completion and the caller must not retire it
 */
static bool read_one_attention(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const fta_shape shape = fta_decode(parameters);
    const uint32_t q_x = fta_coord_x(parameters[kAttnQCoord]);
    const uint32_t q_y = fta_coord_y(parameters[kAttnQCoord]);
    const uint32_t k_x = fta_coord_x(parameters[kAttnKCoord]);
    const uint32_t k_y = fta_coord_y(parameters[kAttnKCoord]);
    const uint32_t v_x = fta_coord_x(parameters[kAttnVCoord]);
    const uint32_t v_y = fta_coord_y(parameters[kAttnVCoord]);
    const uint32_t c_x = fta_coord_x(parameters[kAttnConstCoord]);
    const uint32_t c_y = fta_coord_y(parameters[kAttnConstCoord]);
    const uint32_t depth = shape.block_tiles * shape.dim_tiles;

#ifdef ATTN_SPLITK
    if (fta_is_merge(&shape)) {
        if (!merge_is_executable(ctx, &shape)) {
            ftq_abort(ctx, ctx->error);
            return false;
        }
        return read_partials(ctx, &shape, q_x, q_y, parameters[kAttnQBase]);
    }
#endif
    if (!descriptor_is_executable(ctx, &shape)) {
        ftq_abort(ctx, ctx->error);
        return false;
    }

    if (!read_and_pack_q(ctx, &shape, q_x, q_y, parameters[kAttnQBase]))
        return false;
    if (!read_constants(ctx, &shape, c_x, c_y, parameters[kAttnConstBase],
                        parameters[kAttnReserved]))
        return false;

    for (uint32_t block = 0; block < shape.blocks; block++) {
        const uint32_t first = block * depth;

        if (!read_block(ctx, CB_K, k_x, k_y, parameters[kAttnKBase], first, depth))
            return false;
        if (!read_block(ctx, CB_V, v_x, v_y, parameters[kAttnVBase], first, depth))
            return false;
    }
    return !ctx->aborting;
}

void kernel_main(void)
{
    ftq_context ctx;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);
    ctx.control[kFieldState] = kStateBoot;

    if (!ftq_check_control(&ctx)) {
        ctx.control[kFieldError] = ctx.error;
        ctx.control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    // RESUME rather than restart; resident_matmul_reader.c carries the argument and the
    // refusal that enforces it.
    if (ctx.control[kFieldTail] != ctx.control[kFieldCompletedCount]) {
        ctx.error = kErrBadParameters;
        ctx.control[kFieldError] = ctx.error;
        ctx.control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }
    ctx.cursor = ctx.control[kFieldCompletedCount];

    while (running) {
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);
        /* S2: the producer this descriptor's sidecar entry names has completed, before any
         * input is read (resident_precondition.h). Refused or stopped: leave, unconsumed. */
        if (!ftq_precondition_wait(&ctx)) {
            running = false;
            continue;
        }

        volatile FTQ_L1_PTR uint32_t *descriptor = ftq_descriptor(&ctx);
        const uint32_t opcode = descriptor[kDescOpcode];
        bool done;

        ctx.control[kFieldLastOpcode] = opcode;
        ftq_status_opcode(&ctx, opcode);

        if (opcode == kOpAttention) {
            ftq_status_set(&ctx, kStateCbReserveBack);
            done = read_one_attention(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
            if (!done)
                running = false;
        } else if (opcode == kOpNop) {
            done = true;
        } else if (opcode == kOpStop) {
            done = true;
            running = false;
        } else {
            // A binary is bound to one slab, so an opcode another program implements is
            // genuinely not available here. Refusing names that.
            ctx.error = kErrBadOpcode;
            done = false;
            running = false;
        }

        if (done) {
            ctx.cursor++;
            ctx.control[kFieldTail] = ctx.cursor;
        }
    }

    ft_tensix_noc_read_barrier();
    ft_tensix_noc_write_barrier();
    ctx.control[kFieldError] = ctx.error;
    ctx.control[kFieldState] = kStateExit;
    ftq_status_set(&ctx, kStateExit);
}
