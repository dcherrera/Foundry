/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_place_writer.c
 * @brief Writer half of the RESIDENT MEMORY triad — EMBEDDING and KV_APPEND — and the processor
 * that PUBLISHES COMPLETION, per core and per group.
 *
 * IT IS NOT resident_vector_writer.c, AND THE DIFFERENCE IS THE WHOLE OP. That writer
 * drains `tile_count` whole tiles to consecutive destinations. This one writes a PARTIAL
 * destination: an append into a live KV cache must land `row_count` rows of each tile and
 * leave the other rows exactly as they were, because those rows are earlier positions of
 * the same sequence. So it emits FACE-ROW RUNS (resident_memory_protocol.h,
 * ftm_destination_runs) rather than tiles.
 *
 * THE RUNS COLLAPSE, WHICH IS WHY ONE CODE PATH SERVES DECODE AND PREFILL:
 *   rows [r, r+n) of one face column-half are CONTIGUOUS, so a range is at most four
 *   runs per tile;
 *   rows [0, 32) is the whole tile, and a whole GROUP of tiles is contiguous, so a
 *   32-row append is ONE transaction of `group_tiles * TILE_BYTES`.
 * A one-token decode append is therefore `group_tiles * 2` writes of 32 bytes, and a
 * 32-token prefill append is one write of 8 KiB, out of the same loop.
 *
 * 32 BYTES IS A LEGAL DRAM WRITE and that is the load-bearing fact. The NoC's write
 * alignment is 16 B; only the read direction needs 64 (ft_tt_noc.c,
 * ft_tt_noc_alignment()). tests/test_phase6_memory.c measures it rather than trusting it,
 * and checks the NEIGHBOURS as well as the row — a partial write that quietly rounded up
 * to 64 bytes would pass a check that only looked at what it meant to write.
 *
 * WHY THE WRITER OWNS COMPLETION (resident_queue_protocol.h rule 1). An op is done when
 * the result has left L1. This processor owns `completed_count` and `completed_token`,
 * the reader owns `tail`, and nothing else writes either. It FETCHES on `head` so it
 * drains while the reader is still filling, and PUBLISHES only after `tail` has passed
 * its cursor — fetching on `tail` is a pipeline deadlock, not a precaution
 * (resident_matmul_writer.c records the run that cost).
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_OUT_EMBEDDING index    1: tile size in bytes
 *   2: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 *   3: CB_OUT_KV index -- KV_APPEND's own output CB, because A CB CARRIES ONE GROUP SIZE
 *      (ft_tt_op_memory.h has the arithmetic and the incident). It must name the same two
 *      buffers, in the same roles, as the reader's args 0 and 5: the two processors are
 *      the producer and the consumer of ONE buffer and nothing checks that they agree.
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 * Descriptor words: a3/a4 the S1 group pull on member 0 of a split descriptor
 *   (resident_group_pull.h); a5 the group sequence, stored in control word 56 before
 *   completion is published.
 *
 * Build (NCRISC, NoC 1):
 *   CTARGS=16,2048,1,17 NOC=1 KERNEL_TEXT_BASE=0x18000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_place_writer.c ncrisc /tmp/ttk_memory
 *
 * @status GREEN ON SILICON 2026-08-12 — Tensix 30 and Tensix 32, 17 checks and 0 failures
 *         each. The 32-byte DRAM write is MEASURED, neighbours included. 2656 B of text
 *         and 24 B of bss on NCRISC, against a 12 KiB slab. RE-GREEN on Tensix 40,
 *         2026-08-13, with the output CB split per group size (ft_tt_op_memory.h @status
 *         has the fix's two proofs).
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "resident_memory_protocol.h"
#include "ft_kv_block_float.h"
#include "ft_tensix_dataflow.h"
#include "resident_group_pull.h"
#include "resident_queue_protocol.h"

#define CB_OUT_EMBEDDING ft_tensix_compile_time_arg(0)
#define TILE_BYTES ft_tensix_compile_time_arg(1)
#define WAIT_MODE ft_tensix_compile_time_arg(2)
#define CB_OUT_KV ft_tensix_compile_time_arg(3)

/** The output CB an opcode's group size belongs to — the consumer half of the reader's
 *  output_circular_buffer(). ONE GROUP SIZE PER CB (ft_tt_op_memory.h). */
static uint32_t output_circular_buffer(uint32_t opcode)
{
    return (opcode == kOpKvAppend) ? CB_OUT_KV : CB_OUT_EMBEDDING;
}
/** Put one group's assembled tiles on the destination.
 *
 *  ONE barrier for the whole group, never one per run: a barrier drains this core's
 *  outstanding transactions and is per core, not per transaction. */
/** @return false if any length came out zero — the caller refuses the descriptor rather
 *          than issuing it (resident_memory_protocol.h, ftm_length_is_issuable). */
static bool write_one_group(uint32_t block, uint32_t destination_x, uint32_t destination_y,
                            uint32_t destination_base, uint32_t group_tiles, uint32_t row0,
                            uint32_t row_count, bool whole_tiles)
{
    ftm_run runs[kMemMaxRuns];
    uint32_t run_count;

    if (!ftm_length_is_issuable(group_tiles * TILE_BYTES))
        return false;
    if (whole_tiles || (row0 == 0u && row_count == kMemTileRows)) {
        // The descriptor OWNS the destination tile row — either it said so, or it covers
        // every row of it. Both ends are contiguous, so it is ONE transaction. The rows it
        // did not place are the zeros the reader put in the circular-buffer page.
        ft_tensix_noc_write(block, ft_tensix_noc_address(destination_x, destination_y,
                                                         destination_base),
                            group_tiles * TILE_BYTES);
        ft_tensix_noc_write_barrier();
        return true;
    }

    run_count = ftm_destination_runs(row0, row_count, TILE_BYTES, runs);
    if (run_count == 0u)
        return false;
    for (uint32_t index = 0; index < run_count; index++) {
        if (!ftm_length_is_issuable(runs[index].bytes))
            return false;
    }
    for (uint32_t tile = 0; tile < group_tiles; tile++) {
        const uint32_t tile_offset = tile * TILE_BYTES;

        for (uint32_t index = 0; index < run_count; index++) {
            const uint32_t offset = tile_offset + runs[index].offset;

            ft_tensix_noc_write(block + offset,
                                ft_tensix_noc_address(destination_x, destination_y,
                                                      destination_base + offset),
                                runs[index].bytes);
        }
    }
    ft_tensix_noc_write_barrier();
    return true;
}

/** Read packed destination tile `tile` into `slot` (only when the append keeps rows). */
static void read_packed_tile(uint32_t slot, uint32_t x, uint32_t y, uint32_t destination_base,
                             uint32_t tile, uint32_t packed_bytes)
{
    ft_tensix_noc_read(ft_tensix_noc_address(x, y, destination_base + tile * packed_bytes), slot,
                       packed_bytes);
}

/**
 * The BLOCK-FLOAT form of write_one_group (ft_tt_op_memory.h §1.6): pack the group's rows
 * from the bf16 CB page into Bfp8_b / Bfp4_b and write whole packed tiles.
 *
 * A packed face row (16 or 8 datum bytes, one exponent byte) is below the NoC's write granule,
 * so a partial append READS the destination tile, patches its rows and writes the tile back.
 * Two slots: tile t+1's destination is read while tile t is packed. Nothing else writes these
 * tiles concurrently (appends are ordered on this queue; K and V are separate regions), so the
 * kept rows are written back unchanged. A whole-tile append needs no read: the rows the reader
 * did not place are zeros in the page, exactly as on the bf16 path.
 */
static bool write_one_group_block_float(uint32_t block, uint32_t destination_x,
                                        uint32_t destination_y, uint32_t destination_base,
                                        uint32_t group_tiles, uint32_t row0, uint32_t row_count,
                                        bool whole_tiles, uint32_t format)
{
    const uint32_t packed_bytes = ft_kv_bf_tile_bytes(format);
    const uint32_t slots = ft_tensix_cb_write_pointer(kMemCbKvPack);
    const bool keeps_rows = !(whole_tiles || (row0 == 0u && row_count == kMemTileRows));

    if (packed_bytes == 0u || packed_bytes > (uint32_t)kKvBfBfp8TileBytes)
        return false;
    if (!keeps_rows) {
        /* WHOLE TILES (every full prefill band): ft_kv_bf_pack_tile_words, the same bytes as
         * the row loop below in word-wide code, and no barrier between pack and write. A slot
         * is refilled two tiles after its write, so the wait is for that write to have LEFT
         * this core (its source is reusable), not for its acknowledgement; the one
         * acknowledgement barrier is at the end. The cache is invalidated once up front, as the
         * row loop's first barrier did, so the CB page is read as it landed. */
        ft_bh_l1_cache_invalidate();
        for (uint32_t tile = 0; tile < group_tiles; tile++) {
            const uint32_t slot = slots + (tile & 1u) * (uint32_t)kKvBfBfp8TileBytes;

            if (tile >= 2u)
                ft_tensix_noc_writes_flushed_barrier();
            ft_kv_bf_pack_tile_words(format,
                                     (const uint32_t *)(uintptr_t)(block + tile * TILE_BYTES),
                                     (uint32_t *)(uintptr_t)slot);
            ft_tensix_noc_write(slot,
                                ft_tensix_noc_address(destination_x, destination_y,
                                                      destination_base + tile * packed_bytes),
                                packed_bytes);
        }
        ft_tensix_noc_write_barrier();
        return true;
    }
    /* PARTIAL tiles (decode, a band's first or last tile): read, patch the rows, write back. */
    read_packed_tile(slots, destination_x, destination_y, destination_base, 0u, packed_bytes);
    for (uint32_t tile = 0; tile < group_tiles; tile++) {
        const uint32_t slot = slots + (tile & 1u) * (uint32_t)kKvBfBfp8TileBytes;

        ft_tensix_noc_read_barrier();
        if (tile + 1u < group_tiles) {
            /* The other slot's write (tile - 1) must have left before it is refilled. */
            ft_tensix_noc_write_barrier();
            read_packed_tile(slots + ((tile + 1u) & 1u) * (uint32_t)kKvBfBfp8TileBytes,
                             destination_x, destination_y, destination_base, tile + 1u,
                             packed_bytes);
        }
        ft_kv_bf_pack_tile_rows(format,
                                (const uint16_t *)(uintptr_t)(block + tile * TILE_BYTES),
                                (uint8_t *)(uintptr_t)slot, row0, row_count);
        ft_tensix_noc_write(slot, ft_tensix_noc_address(destination_x, destination_y,
                                                        destination_base + tile * packed_bytes),
                            packed_bytes);
    }
    ft_tensix_noc_write_barrier();
    return true;
}

/**
 * Drain one descriptor: `group_count` groups of `group_tiles` tiles each.
 *
 * @return true if the whole descriptor was written; false if a wait was abandoned
 */
static bool write_one_memory_op(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                                uint32_t cb_out)
{
    const uint32_t flags = parameters[kMemFlags];
    const uint32_t row_count = parameters[kMemRowCount];
    const uint32_t group_count = parameters[kMemGroupCount];
    const uint32_t group_tiles = parameters[kMemGroupTiles];
    const uint32_t destination_x = parameters[kMemDstX];
    const uint32_t destination_y = parameters[kMemDstY];
    const uint32_t destination_base = parameters[kMemDstBase];
    const uint32_t destination_group_stride = parameters[kMemDstGroupStride];
    const uint32_t destination_row0 = parameters[kMemDstRow0];
    const uint32_t row_in_tile = destination_row0 % kMemTileRows;
    const uint32_t tile_row = destination_row0 / kMemTileRows;
    const uint32_t kv_format = (flags >> kMemKvFormatShift) & kMemKvFormatMask;
    const uint32_t destination_tile_bytes = kv_format ? ft_kv_bf_tile_bytes(kv_format) : TILE_BYTES;
    uint32_t group_bytes;
    uint32_t tile_row_offset;

    if (row_count == 0u || row_count > kMemMaxRows || group_count == 0u || group_tiles == 0u ||
        row_in_tile + row_count > kMemTileRows) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    /* THE DESTINATION WAS UNBOUNDED. `destination_row0` is an absolute KV position, so
     * `tile_row * group_tiles * TILE_BYTES` is a raw byte offset with nothing above it — a
     * position past the cache capacity wrote into whatever DRAM follows, at full length,
     * with nothing faulting. The bound is kMemDstGroupStride, the bytes each group OWNS;
     * see ftm_destination_span_fits(). */
    if (destination_tile_bytes == 0u ||
        !ft_tensix_product_fits(group_tiles, destination_tile_bytes, &group_bytes) ||
        !ft_tensix_product_fits(tile_row, group_bytes, &tile_row_offset)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    /* And the CB CYCLE this file did not have: the output CB is POPPED `group_tiles` at a
     * time and the read pointer wraps by the same exact-hit rule as the write pointer, so a
     * descriptor that does not close the cycle leaves every later one off the grid. Kept
     * unweakened after the one-CB-per-group-size fix, for the reason the reader states. */
    if (!ft_tensix_cb_pop_grid_is_safe(cb_out, group_tiles)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }

    for (uint32_t group = 0; group < group_count; group++) {
        const bool have_tiles =
            ftq_cb_wait_front_stoppable(ctx, (int32_t)cb_out, (int32_t)group_tiles, WAIT_MODE);

        if (have_tiles) {
            uint32_t group_offset;
            uint32_t x = destination_x, y = destination_y, slot = group;
            /* BANKED: this group's own bank, and its slot inside it (one base address). */
            const bool placed = (flags & kMemFlagDstBanked) == 0u ||
                                ftm_banked_destination(flags, destination_x, destination_y,
                                                       group, &x, &y, &slot);
            /* Inside the GROUP, not merely inside the region: the tighter statement, and the
             * one that also stops head h's append landing in head h+1. */
            const bool bounded =
                placed && ft_tensix_product_fits(slot, destination_group_stride, &group_offset) &&
                group_offset <= 0xFFFFFFFFu - tile_row_offset &&
                ftm_destination_span_fits(destination_group_stride, tile_row_offset,
                                          group_bytes);
            const uint32_t base = destination_base + group_offset + tile_row_offset;

            const bool whole = (flags & kMemFlagDstWholeTiles) != 0u;
            const bool written =
                bounded &&
                (kv_format ? write_one_group_block_float(ft_tensix_cb_read_pointer(cb_out), x, y,
                                                         base, group_tiles, row_in_tile,
                                                         row_count, whole, kv_format)
                           : write_one_group(ft_tensix_cb_read_pointer(cb_out), x, y, base,
                                             group_tiles, row_in_tile, row_count, whole));

            if (!written) {
                ftq_abort(ctx, kErrBadParameters);
                ft_tensix_cb_pop_front(cb_out, group_tiles);
                return false;
            }
        }
        ft_tensix_cb_pop_front(cb_out, group_tiles);
    }
    return !ctx->aborting;
}

void kernel_main(void)
{
    ftq_context ctx;
    uint32_t completed;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);

    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    completed = ctx.control[kFieldCompletedCount];
    ctx.cursor = completed;

    while (running) {
        // FETCH on `head`, so the writer drains the output CB while the reader is still
        // filling it. Publishing waits on `tail`, below.
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);

        volatile FTQ_L1_PTR uint32_t *descriptor = ftq_descriptor(&ctx);
        const uint32_t opcode = descriptor[kDescOpcode];
        const uint32_t token = descriptor[kDescToken];
        const uint32_t group_sequence = descriptor[kDescA5];
        const bool memory_op = opcode == kOpEmbedding || opcode == kOpKvAppend;
        bool done = true;

        ftq_status_opcode(&ctx, opcode);
        if (memory_op) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            done = write_one_memory_op(&ctx, ftq_parameters(descriptor),
                                       output_circular_buffer(opcode));
            ftq_status_set(&ctx, kStateExec);
        } else if (opcode == kOpStop) {
            running = false;
        }

        if (!done) {
            running = false;
            continue;
        }

        // THE PRECONDITION, and the only place it belongs: the reader must have finished
        // this descriptor before its completion is published.
        if (!ftq_wait_until_reaches(&ctx, kFieldTail, ctx.cursor + 1u)) {
            running = false;
            continue;
        }

        ctx.cursor++;
        /* S1 GROUP PULL (resident_group_pull.h): on member 0 of a split descriptor, a3 names
         * the members and a4 the sequence they must reach; it publishes only after every one
         * of them has. a3 = 0 (every unsplit descriptor) skips it. */
        {
            const uint32_t pull = descriptor[kDescA3];

            if (memory_op && (pull & ~kGroupPullFlags) > 1u &&
                (pull & kPullRedSkip) == 0u &&
                !ftq_group_pull(&ctx, pull & ~kGroupPullFlags, descriptor[kDescA4])) {
                running = false;
                continue;
            }
        }
        // The done-sequence a member 0 pull reads (control word 56), before the count.
        ctx.control[kFieldGroupDone] = group_sequence;
        // Token before count: a host that sees the count is entitled to trust the token it
        // reads next. Both are plain stores from ONE in-order RISC-V.
        ctx.control[kFieldCompletedToken] = token;
        ctx.control[kFieldCompletedCount] = ++completed;
        ftq_group_post(&ctx, group_sequence, token);
    }

    // The writer does NOT touch kFieldError or kFieldState: the reader owns those, and one
    // writer per word is the whole ordering argument.
    ftq_status_set(&ctx, kStateExit);
}
