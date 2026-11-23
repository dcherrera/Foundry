/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_rope_reader.c
 * @brief Reader third of the RESIDENT ROPE triad, and THE PROCESSOR THAT IMPLEMENTS THE PAIRING.
 *
 * Everything RoPE-specific about the tile layout is here, as ADDRESSING. The compute
 * thread never indexes an element and never knows which half a tile is in: it is handed
 * four aligned tile streams and multiplies them.
 *
 *   CB_IN    the activation, tile column c of a head, in order
 *   CB_ROT   the PARTNER tile, ftr_partner_tile(c) of the same head — the whole rotation
 *   CB_COS   cos(angle) for column c, from the table
 *   CB_SIN   SIGNED sin(angle) for column c: -sin below the half, +sin above it
 *
 * THE PARTNER STREAM IS TWO NoC READS PER BLOCK, NEVER MORE. Inside one head the partner
 * index is a rotation by half modulo rot_tiles, so a block of `depth` consecutive columns
 * maps to at most TWO contiguous runs — the run from (c0 + half) to the end, and the
 * wrapped remainder from 0. Both land in the contiguous CB block the reserve just gave us.
 *
 * AND IT IS PER-BLOCK INTERLEAVED, WHICH IS A DEADLOCK ARGUMENT AND NOT A STYLE CHOICE.
 * The compute thread consumes column c of all four streams together, so a reader that
 * pushed a whole head of CB_IN before starting CB_ROT would block in cb_reserve_back once
 * CB_IN was full while compute blocked in cb_wait_front on CB_ROT — three processors
 * waiting on each other, exactly the pipeline deadlock resident_matmul_writer.c records.
 * Every stream advances one block at a time, in the order the consumer wants them.
 *
 * ONE BARRIER PER BLOCK, for four streams and up to five transactions. It is a per-CORE
 * drain, not a per-transaction one, and paying it per tile is
 * what cost 5.9x on the matmul (docs/tt_bringup_log.md 16.1).
 *
 * THIS PROCESSOR OWNS `tail`, AND NOTHING ELSE IN THE CONTROL BLOCK
 * (resident_queue_protocol.h rule 1). An aborted descriptor advances nothing.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_IN index   1: CB_ROT index   2: CB_COS index   3: CB_SIN index
 *   4: tile size in bytes
 *   5: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0,1,2,3,2048,1 NOC=0 KERNEL_TEXT_BASE=0x15000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_rope_reader.c brisc /tmp/ttk_rope
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_queue_protocol.h"
#include "resident_precondition.h"
#include "resident_rope_protocol.h"

#define CB_IN ft_tensix_compile_time_arg(0)
#define CB_ROT ft_tensix_compile_time_arg(1)
#define CB_COS ft_tensix_compile_time_arg(2)
#define CB_SIN ft_tensix_compile_time_arg(3)
#define TILE_BYTES ft_tensix_compile_time_arg(4)
#define WAIT_MODE ft_tensix_compile_time_arg(5)
/** One descriptor's shape, read once so the block loop indexes registers. */
typedef struct rope_shape {
    uint32_t tile_count;
    uint32_t head_tiles;
    uint32_t rot_tiles;
    uint32_t heads;
    uint32_t depth;
    uint32_t a_x;
    uint32_t a_y;
    uint32_t a_base;
    uint32_t table_x;
    uint32_t table_y;
    uint32_t table_base;
} rope_shape;

static void load_shape(rope_shape *shape, volatile FTQ_L1_PTR uint32_t *parameters)
{
    shape->tile_count = parameters[kRopeTileCount];
    shape->head_tiles = parameters[kRopeHeadTiles];
    shape->rot_tiles = parameters[kRopeRotTiles];
    shape->heads = parameters[kRopeHeads];
    shape->a_x = parameters[kRopeAX];
    shape->a_y = parameters[kRopeAY];
    shape->a_base = parameters[kRopeABase];
    shape->table_x = parameters[kRopeTableX];
    shape->table_y = parameters[kRopeTableY];
    shape->table_base = parameters[kRopeTableBase];

    /* The depth must divide BOTH runs it blocks over: head_tiles for the activation and
     * rot_tiles for the three rotated streams. The host already shrinks it; this is the
     * kernel refusing to guess, which is the same belt the matmul reader wears. */
    const uint32_t requested = parameters[kRopeDepth];
    const uint32_t on_head = ftr_block_depth(shape->head_tiles, requested);
    shape->depth = ftr_block_depth(shape->rot_tiles, on_head);
}

/**
 * Refuse a descriptor this program cannot execute, INSTEAD of writing outside the four
 * circular buffers.
 *
 * `ftr_block_depth()` makes `depth` divide BOTH runs, which is the ONE-SHOT invariant and
 * only half of the resident one. The FIFO write pointer wraps on an EXACT hit of the limit
 * and nothing else, so each buffer also has to end a descriptor
 * where it started it: `pages % depth == 0` AND `pushes % pages == 0`.
 * `ft_tensix_cb_push_grid_is_safe()` carries the derivation and the measured incident — a read
 * that walked onto the queue control block at 0x170000 with nothing faulting.
 *
 * THE FOUR STREAMS PUSH DIFFERENT TOTALS, which is what makes this op the worst case in the
 * family: CB_IN takes `tile_count` pages while the three rotated streams take
 * `instances * rot_tiles`, and `rot_tiles` is HALF `head_tiles` in the shapes this op
 * supports. A single test against CB_IN would pass while the other three walked.
 */
static bool descriptor_is_executable(ftq_context *ctx, const rope_shape *shape)
{
    if (shape->depth == 0u || shape->head_tiles == 0u || shape->rot_tiles == 0u ||
        TILE_BYTES == 0u)
        ctx->error = kErrBadParameters;
    else if (!ft_tensix_cb_push_grid_is_safe(CB_IN, shape->depth) ||
             !ft_tensix_cb_push_grid_is_safe(CB_ROT, shape->depth) ||
             !ft_tensix_cb_push_grid_is_safe(CB_COS, shape->depth) ||
             !ft_tensix_cb_push_grid_is_safe(CB_SIN, shape->depth))
        ctx->error = kErrBadParameters;
    else
        return true;
    return false;
}

/**
 * Issue the partner block for columns [c0, c0 + depth) of the head based at `head_base`.
 *
 * At most TWO transactions by construction: the partner index is (c + half) mod rot_tiles,
 * which runs contiguously up to the wrap and contiguously after it.
 */
static void issue_partner_block(const rope_shape *shape, uint32_t head_base, uint32_t write_address,
                                uint32_t c0)
{
    const uint32_t start = ftr_partner_tile(c0, shape->rot_tiles);
    const uint32_t to_end = shape->rot_tiles - start;
    const uint32_t first = (shape->depth < to_end) ? shape->depth : to_end;

    ft_tensix_noc_read(ft_tensix_noc_address(shape->a_x, shape->a_y,
                                             head_base + start * TILE_BYTES),
                       write_address, first * TILE_BYTES);
    if (first < shape->depth) {
        ft_tensix_noc_read(ft_tensix_noc_address(shape->a_x, shape->a_y, head_base),
                           write_address + first * TILE_BYTES, (shape->depth - first) * TILE_BYTES);
    }
}

/**
 * The reader half of one ROPE descriptor.
 *
 * @return true if the whole descriptor was read; false if a wait was abandoned, in which
 *         case the loop still ran to completion (rule 3) and the caller must not retire it
 */
static bool read_one_rope(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    rope_shape shape;

    load_shape(&shape, parameters);
    if (!ftr_shape_is_supported(shape.head_tiles, shape.rot_tiles) || shape.heads == 0u ||
        shape.tile_count == 0u || (shape.tile_count % shape.head_tiles) != 0u ||
        !descriptor_is_executable(ctx, &shape)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }

    /* One "instance" is one head of one row block: head_tiles activation tiles. The TABLE
     * advances once per ROW BLOCK and is shared by every head in it, which is why `heads` is
     * a field — every head of a row block rotates by the same angles. */
    const uint32_t instances = shape.tile_count / shape.head_tiles;
    const uint32_t table_block_tiles = 2u * shape.rot_tiles;

    for (uint32_t instance = 0; instance < instances; instance++) {
        const uint32_t row_block = instance / shape.heads;
        const uint32_t head_base = shape.a_base + instance * shape.head_tiles * TILE_BYTES;
        const uint32_t table = shape.table_base + row_block * table_block_tiles * TILE_BYTES;

        for (uint32_t c0 = 0; c0 < shape.head_tiles; c0 += shape.depth) {
            const bool rotated = c0 < shape.rot_tiles;
            uint32_t in_write;
            uint32_t rot_write = 0u;
            uint32_t cos_write = 0u;
            uint32_t sin_write = 0u;

            if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN,
                                               (int32_t)shape.depth, WAIT_MODE))
                return false;
            in_write = ft_tensix_cb_write_pointer(CB_IN);
            if (rotated) {
                if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_ROT,
                                                   (int32_t)shape.depth, WAIT_MODE))
                    return false;
                rot_write = ft_tensix_cb_write_pointer(CB_ROT);
                if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_COS,
                                                   (int32_t)shape.depth, WAIT_MODE))
                    return false;
                cos_write = ft_tensix_cb_write_pointer(CB_COS);
                if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_SIN,
                                                   (int32_t)shape.depth, WAIT_MODE))
                    return false;
                sin_write = ft_tensix_cb_write_pointer(CB_SIN);
            }

            // Rule 3: while draining, issue NOTHING. The pushes below still happen so the
            // page counters stay balanced, but a transaction issued during a drain would
            // still be in flight when the kernel returns.
            if (!ctx->aborting) {
                ft_tensix_noc_read(ft_tensix_noc_address(shape.a_x, shape.a_y,
                                                         head_base + c0 * TILE_BYTES),
                                   in_write, shape.depth * TILE_BYTES);
                if (rotated) {
                    issue_partner_block(&shape, head_base, rot_write, c0);
                    const uint32_t sin_offset = (shape.rot_tiles + c0) * TILE_BYTES;

                    ft_tensix_noc_read(ft_tensix_noc_address(shape.table_x, shape.table_y,
                                                             table + c0 * TILE_BYTES),
                                       cos_write, shape.depth * TILE_BYTES);
                    ft_tensix_noc_read(ft_tensix_noc_address(shape.table_x, shape.table_y,
                                                             table + sin_offset),
                                       sin_write, shape.depth * TILE_BYTES);
                }
                // ONE barrier for the whole block, all four streams.
                ft_tensix_noc_read_barrier();
            }

            ft_tensix_cb_push_back(CB_IN, shape.depth);
            if (rotated) {
                ft_tensix_cb_push_back(CB_ROT, shape.depth);
                ft_tensix_cb_push_back(CB_COS, shape.depth);
                ft_tensix_cb_push_back(CB_SIN, shape.depth);
            }
        }
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

    // RESUME rather than restart. All five processors resume from completed_count, the ONE
    // number that means "every stage finished this descriptor" — resident_matmul_reader.c
    // carries the full argument and the refusal that enforces it.
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

        if (opcode == kOpRope) {
            ftq_status_set(&ctx, kStateCbReserveBack);
            done = read_one_rope(&ctx, ftq_parameters(descriptor));
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
