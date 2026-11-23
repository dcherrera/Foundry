/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_gather_reader.c
 * @brief Reader half of the RESIDENT MEMORY triad — EMBEDDING and KV_APPEND. It fetches the
 * source rows and ASSEMBLES them into face-ordered destination tiles in the output
 * circular buffer; resident_place_writer.c puts those tiles on the destination.
 *
 * THIS IS THE DEGENERATE TRIAD: reader and writer, no compute thread at all. Neither op
 * has any arithmetic in it — they are placed copies — and the enables mask is DM0|DM1.
 * resident_memory_protocol.h has the reason that is safe (TRISC0 zeroes the CB page
 * counters only when BRISC ASKS, and a resident BRISC never returns to the firmware loop
 * that asks).
 *
 * WHY THE SCATTER IS A CPU LOOP AND NOT MORE NoC TRANSACTIONS. A tile row is not
 * contiguous: one row of one tile is two 32-byte face rows half a tile apart. So placing
 * a row means `group_tiles * 2` short moves. Done with the NoC that is 96 transactions
 * for one 1536-wide embedding row at ~50 ns of issue each; done with RISC-V loads and
 * stores into L1 — which is ordinary memory here, with its data cache disabled — it is
 * ~768 word copies. The read from DRAM stays ONE transaction because the source row is
 * contiguous, and a 32-byte DRAM READ would not be legal anyway (64 B alignment,
 * ft_tt_noc_alignment()), so there was never a version of this that gathered face rows
 * directly out of DRAM.
 *
 * THE FAST PATH IS THE PREFILL PATH. When the source is already tiled and the source row
 * and destination row agree inside their tiles, the destination tiles ARE the source
 * tiles: the read goes straight into the output CB and no byte is touched twice. That is
 * every 32-token-aligned KV append.
 *
 * THIS PROCESSOR OWNS `tail`, AND NOTHING ELSE IN THE CONTROL BLOCK
 * (resident_queue_protocol.h rule 1). An aborted descriptor advances nothing.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_OUT_EMBEDDING index    1: CB_STAGE index   2: CB_INDEX index
 *   3: tile size in bytes
 *   4: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 *   5: CB_OUT_KV index -- KV_APPEND's own output CB, because A CB CARRIES ONE GROUP SIZE
 *      (ft_tt_op_memory.h has the arithmetic and the incident). Passing arg 5 EQUAL to
 *      arg 0 rebuilds the defect deliberately and is how the guard is proved to refuse;
 *      ft_tensix_compile_time_arg() has no bounds check, so OMITTING it reads past the
 *      array and produces a CB index nobody chose -- which the same guard refuses, since
 *      an unconfigured CB has zero pages.
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=16,0,1,2048,1,17 NOC=0 KERNEL_TEXT_BASE=0x15000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_gather_reader.c brisc /tmp/ttk_memory
 *
 * @status GREEN ON SILICON 2026-08-12 — Tensix 30 and Tensix 32, 17 checks and 0 failures
 *         each, byte-exact. 2264 B of text on BRISC, no bss, against a 12 KiB slab.
 *         RE-GREEN on Tensix 40, 2026-08-13, with the output CB split per group size: 28
 *         descriptors on one image load, 0 checks failed. The aliased build (arg 5 = arg 0)
 *         on Tensix 41 refuses the embedding descriptor at exactly the point the defect
 *         predicts — see ft_tt_op_memory.h @status.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "resident_memory_protocol.h"
#include "ft_tensix_dataflow.h"
#include "resident_queue_protocol.h"
#include "resident_precondition.h"

#define CB_OUT_EMBEDDING ft_tensix_compile_time_arg(0)
#define CB_STAGE ft_tensix_compile_time_arg(1)
#define CB_INDEX ft_tensix_compile_time_arg(2)
#define TILE_BYTES ft_tensix_compile_time_arg(3)
#define WAIT_MODE ft_tensix_compile_time_arg(4)
#define CB_OUT_KV ft_tensix_compile_time_arg(5)

/** The output CB an opcode's group size belongs to. ONE GROUP SIZE PER CB — the whole
 *  reason there are two (ft_tt_op_memory.h). Both are compile-time constants, so this
 *  folds to one of them at -Os. */
static uint32_t output_circular_buffer(uint32_t opcode)
{
    return (opcode == kOpKvAppend) ? CB_OUT_KV : CB_OUT_EMBEDDING;
}
/** Everything one descriptor says, read once so the loops index registers. */
typedef struct memory_request {
    uint32_t flags;
    uint32_t row_count;
    uint32_t group_count;
    uint32_t group_tiles;
    uint32_t source_x;
    uint32_t source_y;
    uint32_t source_base;
    uint32_t source_row_stride;
    uint32_t source_row0;
    uint32_t source_row_limit; /**< rows the ROW-MAJOR table has; bounds every gather index */
    uint32_t source_group_stride;
    uint32_t destination_row; /**< within its tile: DST_ROW0 % 32 */
} memory_request;

static void load_request(memory_request *request, volatile FTQ_L1_PTR uint32_t *parameters)
{
    request->flags = parameters[kMemFlags];
    request->row_count = parameters[kMemRowCount];
    request->group_count = parameters[kMemGroupCount];
    request->group_tiles = parameters[kMemGroupTiles];
    request->source_x = parameters[kMemSrcX];
    request->source_y = parameters[kMemSrcY];
    request->source_base = parameters[kMemSrcBase];
    request->source_row_stride = parameters[kMemSrcRowStride];
    request->source_row0 = parameters[kMemSrcRow0];
    request->source_row_limit = parameters[kMemSrcRowLimit];
    request->source_group_stride = parameters[kMemSrcGroupStride];
    request->destination_row = parameters[kMemDstRow0] % kMemTileRows;
}

/** A descriptor the kernel refuses rather than half-executes. The host checks all of this
 *  too; the kernel repeats it because a parameter block is a POINTER in a descriptor and a
 *  wrong one turns into NoC transactions at addresses nobody chose (spec §7.3). */
static bool request_is_sane(const memory_request *request, uint32_t cb_out)
{
    if (request->row_count == 0u || request->row_count > kMemMaxRows)
        return false;
    if (request->group_count == 0u || request->group_tiles == 0u)
        return false;
    if (request->destination_row + request->row_count > kMemTileRows)
        return false;
    if ((request->flags & kMemFlagSrcTiled) != 0u &&
        request->source_row0 + request->row_count > kMemTileRows)
        return false;
    /* THE CB CYCLE, which this file did not have. The output CB takes `group_tiles` pages
     * per group and `group_count * group_tiles` in total; `cb_push_back` wraps only on an
     * EXACT hit of the limit, so a descriptor that leaves the write pointer part-way round
     * sends the NEXT descriptor's blocks over the limit and up into L1
     * (ft_tensix_cb_push_grid_is_safe(), and the measured incident it records).
     *
     * With ONE CB PER GROUP SIZE this can no longer fail for a well-built residency — the
     * pointer never leaves its own grid. It is kept, unweakened, because that is exactly
     * the statement being relied on, and because a build that aliases the two CB indices
     * onto one buffer must still be REFUSED rather than run. */
    if (!ft_tensix_cb_push_grid_is_safe(cb_out, request->group_tiles))
        return false;
    return true;
}

/** Assemble one group from a TILED source.
 *
 *  @return false if a length came out zero — see ftm_length_is_issuable(); the caller then
 *          aborts the descriptor rather than issuing it. */
static bool assemble_from_tiles(const memory_request *request, uint32_t block_address)
{
    const uint32_t group_bytes = request->group_tiles * TILE_BYTES;
    const bool rows_align = request->source_row0 == request->destination_row;
    const bool whole_tiles = (request->flags & kMemFlagDstWholeTiles) != 0u;
    const bool direct = rows_align && !(whole_tiles && request->row_count < kMemTileRows);
    const uint32_t landing = direct ? block_address : ft_tensix_cb_write_pointer(CB_STAGE);

    if (!ftm_length_is_issuable(group_bytes))
        return false;
    ft_tensix_noc_read(ft_tensix_noc_address(request->source_x, request->source_y,
                                             request->source_base), landing,
                       group_bytes);
    ft_tensix_noc_read_barrier();
    if (direct)
        return true;

    for (uint32_t row = 0; row < request->row_count; row++) {
        ftm_place_row_from_tiles((uint32_t *)(uintptr_t)(block_address),
                                 (const uint32_t *)(uintptr_t)(landing),
                                 request->source_row0 + row, request->destination_row + row,
                                 request->group_tiles, TILE_BYTES);
    }
    return true;
}

/** Assemble one group from a ROW-MAJOR source, gathering row indices when asked to.
 *
 *  Every row is issued before the ONE barrier: a barrier is per core, not per transaction
 *  (the dataflow API's barrier), and paying it per row is the mistake that cost 5.9x on the
 *  matmul reader (docs/tt_bringup_log.md §16.1). */
static bool assemble_from_rows(const memory_request *request, uint32_t block_address)
{
    const uint32_t row_bytes = ftm_group_row_bytes(request->group_tiles, TILE_BYTES);
    const uint32_t stage = ft_tensix_cb_write_pointer(CB_STAGE);
    volatile FTQ_L1_PTR uint32_t *indices =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(ft_tensix_cb_read_pointer(CB_INDEX));
    const bool gather = (request->flags & kMemFlagGather) != 0u;

    if (!ftm_length_is_issuable(row_bytes))
        return false;
    /* EVERY row index is checked BEFORE the first transaction is issued, not as it goes.
     * A loop that refused half way would leave the earlier reads in flight and the CB page
     * counters mid-descriptor; refusing up front means nothing was issued at all. The index
     * is a wire value — for a GATHER it is a word out of CB_INDEX — and it becomes a raw
     * DRAM address, so this is the bound between a wrong token id and an arbitrary read.
     * The same predicate also carries the 64-byte stride alignment the row read needs. */
    for (uint32_t row = 0; row < request->row_count; row++) {
        const uint32_t index = gather ? indices[row] : request->source_row0 + row;

        if (!ftm_row_major_source_is_readable(request->source_base, request->source_row_stride,
                                              request->source_row_limit, index, row_bytes))
            return false;
    }
    for (uint32_t row = 0; row < request->row_count; row++) {
        const uint32_t index = gather ? indices[row] : request->source_row0 + row;

        const uint32_t source = request->source_base + index * request->source_row_stride;

        ft_tensix_noc_read(ft_tensix_noc_address(request->source_x, request->source_y, source),
                           stage + row * row_bytes, row_bytes);
    }
    ft_tensix_noc_read_barrier();

    for (uint32_t row = 0; row < request->row_count; row++) {
        ftm_place_row_from_row_major((uint32_t *)(uintptr_t)(block_address),
                                     (const uint32_t *)(uintptr_t)(stage + row * row_bytes),
                                     request->destination_row + row, request->group_tiles,
                                     TILE_BYTES);
    }
    return true;
}

/**
 * The reader half of one EMBEDDING or KV_APPEND descriptor.
 *
 * @return true if the whole descriptor was assembled; false if a wait was abandoned, in
 *         which case the loop still ran to completion (rule 3) and the caller must not
 *         retire it
 */
static bool read_one_memory_op(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                               uint32_t cb_out)
{
    memory_request request;

    /* A zero tile size would make every transfer below zero-length, which is the shape
     * suspected of taking the whole card down (ftm_length_is_issuable). TILE_BYTES is a
     * compile-time argument, so this folds away entirely at -Os and costs nothing; it is a
     * refusal rather than a build-time assert only because a compile-time argument is not an
     * integer constant expression in C. */
    if (TILE_BYTES == 0u) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    load_request(&request, parameters);
    if (!request_is_sane(&request, cb_out)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }

    for (uint32_t group = 0; group < request.group_count; group++) {
        memory_request group_request = request;
        uint32_t block;

        group_request.source_base = request.source_base + group * request.source_group_stride;
        if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb_out,
                                           (int32_t)request.group_tiles, WAIT_MODE))
            return false;
        block = ft_tensix_cb_write_pointer(cb_out);

        // Rule 3: while draining, issue NOTHING. The push below still happens so the page
        // counters stay balanced, but a transaction issued during a drain would still be in
        // flight when the kernel returns.
        if (!ctx->aborting) {
            // A partial destination tile in WHOLE-TILES mode has to be zero where this
            // descriptor places nothing — otherwise it publishes whatever the CB page held
            // from an earlier descriptor, which is a plausible wrong answer.
            if ((request.flags & kMemFlagDstWholeTiles) != 0u &&
                request.row_count < kMemTileRows) {
                ftm_zero_words((uint32_t *)(uintptr_t)(block),
                               request.group_tiles * TILE_BYTES / 4u);
            }
            const bool assembled = ((request.flags & kMemFlagSrcTiled) != 0u)
                                       ? assemble_from_tiles(&group_request, block)
                                       : assemble_from_rows(&group_request, block);

            // A zero-length transfer is REFUSED, never skipped: skipping would advance the
            // descriptor and let the writer publish completion for work nobody did, and
            // issuing it is suspected of taking the card down (ftm_length_is_issuable).
            if (!assembled) {
                ftq_abort(ctx, kErrBadParameters);
                ft_tensix_cb_push_back(cb_out, request.group_tiles);
                return false;
            }
        }

        ft_tensix_cb_push_back(cb_out, request.group_tiles);
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

    // RESUME rather than restart, from completed_count — the ONE number that means "every
    // stage finished this descriptor". resident_matmul_reader.c carries the argument.
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

        if (opcode == kOpEmbedding || opcode == kOpKvAppend) {
            ftq_status_set(&ctx, kStateCbReserveBack);
            done = read_one_memory_op(&ctx, ftq_parameters(descriptor),
                                      output_circular_buffer(opcode));
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
            // genuinely not available here. Refusing names that; executing nothing would
            // retire work that never happened.
            ctx.error = kErrBadOpcode;
            done = false;
            running = false;
        }

        // Rule 5. An abandoned descriptor was handed to nobody, so it advances NOTHING and
        // the next launch re-attempts it.
        if (done) {
            ctx.cursor++;
            ctx.control[kFieldTail] = ctx.cursor;
        }
    }

    // Nothing is outstanding here: the drain path issues no transactions. The barrier is
    // the cheap statement of that invariant rather than a fix for a case that can happen.
    ft_tensix_noc_read_barrier();
    ft_tensix_noc_write_barrier();
    ctx.control[kFieldError] = ctx.error;
    ctx.control[kFieldState] = kStateExit;
    ftq_status_set(&ctx, kStateExit);
}
