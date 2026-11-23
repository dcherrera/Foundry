/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_attention_writer.c
 * @brief Writer third of the RESIDENT FLASH-DECODE ATTENTION triad, and the processor that
 * PUBLISHES COMPLETION -- both per core (completed_count/completed_token) and per group.
 *
 * IT IS THE INVERSE OF THE READER'S Q PACK. The compute thread hands it `dim_tiles`
 * tiles whose ROW h is Q head `q_first + h`; the model wants head h's dim tile d at tile
 * column `h*dim_tiles + d` of the activation buffer, in the decode token's own row
 * `position % 32` (ft_tt_op_memory.h §1.5b, the row-phase invariant this op inherits).
 * So one output tile becomes `q_rows * 2` writes of a 32-byte face row.
 *
 * A 32-BYTE WRITE TO DRAM AT A 32-BYTE-ALIGNED ADDRESS LANDS, AND LEAVES ITS NEIGHBOURS
 * ALONE. That is measured, not assumed -- it is the same write KV_APPEND makes, checked
 * on silicon against a poisoned region with 0 bytes of an untouched neighbour disturbed
 * (ft_tt_op_memory.h @status). It is also why the activation layout works at all: the
 * NoC's minimum DRAM write granule is 16 B, so a face row is legal and a COLUMN would not
 * be.
 *
 * ONE BARRIER PER OUTPUT TILE, not one per write: a barrier drains this core's
 * outstanding transactions and is per core, not per transaction
 * (docs/tt_kernel_model.md §12).
 *
 * SPLIT-K (-DATTN_SPLITK, resident_attention_protocol.h §7): a PARTIAL's dim_tiles + 2 tiles
 * go out whole to OUT, unscattered; a MERGE is scattered as a decode / ROWS output.
 *
 * WHY THE WRITER OWNS COMPLETION, AND WHY IT FETCHES ON `head` BUT PUBLISHES ON `tail`:
 * resident_matmul_writer.c carries the argument, and the deadlock that made the
 * distinction necessary.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_OUT index   1: tile size in bytes
 *   2: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 * Descriptor words: a3/a4 the S1 group pull on member 0 of a split descriptor
 *   (resident_group_pull.h); a5 the group sequence, stored in control word 56 before
 *   completion is published.
 *
 * Build (NCRISC, NoC 1):
 *   CTARGS=16,2048,1 NOC=1 KERNEL_TEXT_BASE=0x18000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_attention_writer.c ncrisc /tmp/ttk_attn
 *
 * @status GREEN ON SILICON on Tensix 70/71/72/73, 0 checks failed.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_group_pull.h"
#include "resident_queue_protocol.h"
#include "resident_attention_protocol.h"

#define CB_OUT ft_tensix_compile_time_arg(0)
#define TILE_BYTES ft_tensix_compile_time_arg(1)
#define WAIT_MODE ft_tensix_compile_time_arg(2)
/**
 * Scatter one output tile: row h of the tile becomes the query row of destination tile
 * `h * dim_tiles + dim_tile`.
 *
 * @return false if a length came out zero -- a REFUSAL, never a skipped write
 *         (resident_attention_protocol.h §6)
 */
static bool scatter_one_tile(const fta_shape *shape, uint32_t source, uint32_t x, uint32_t y,
                             uint32_t base, uint32_t dim_tile)
{
    const uint32_t chunk = fta_chunk_bytes(TILE_BYTES);

    /* CHUNK MODE IS THE INVERSE OF THE READER'S: nothing to scatter. The compute thread's
     * output tile already has position t in row t, which is the activation layout the next
     * op wants, so the tile goes out whole -- ONE write of TILE_BYTES instead of q_rows*2
     * writes of 32 B. Rows C..31 are written too; they are the strip's padding rows, whose
     * value is discarded by every consumer and whose scores the causal tile keeps finite. */
    if (fta_is_chunk(shape)) {
        if (!fta_length_is_issuable(TILE_BYTES))
            return false;
        ft_tensix_noc_write(source,
                            ft_tensix_noc_address(x, y, base + dim_tile * TILE_BYTES),
                            TILE_BYTES);
        ft_tensix_noc_write_barrier();
        return true;
    }

    if (!fta_length_is_issuable(chunk))
        return false;

    for (uint32_t p = 0; p < shape->positions; p++) {
        for (uint32_t head = 0; head < shape->q_rows; head++) {
            const uint32_t destination_tile = (head * shape->dim_tiles + dim_tile) * TILE_BYTES;

            for (uint32_t half = 0; half < 2u; half++) {
                ft_tensix_noc_write(source + fta_chunk_offset(p * shape->q_rows + head, half,
                                                              TILE_BYTES),
                                    ft_tensix_noc_address(x, y,
                                                          base + destination_tile +
                                                          fta_chunk_offset(shape->query_row + p,
                                                                           half, TILE_BYTES)),
                                    chunk);
            }
        }
    }
    ft_tensix_noc_write_barrier();
    return true;
}

/**
 * The writer half of one ATTENTION descriptor: `dim_tiles` tiles, scattered.
 *
 * One tile is waited for at a time, so nothing here depends on several circular-buffer
 * pages being contiguous -- which is what lets CB_OUT be double-buffered without the
 * block-prefetch invariant applying to it (§5 is about the PRODUCER's block size).
 *
 * @return true if the whole descriptor was written; false if a wait was abandoned
 */
static bool write_one_attention(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const fta_shape shape = fta_decode(parameters);
    const uint32_t x = fta_coord_x(parameters[kAttnOutCoord]);
    const uint32_t y = fta_coord_y(parameters[kAttnOutCoord]);
    const uint32_t base = parameters[kAttnOutBase];

    if (!fta_shape_valid(&shape) || TILE_BYTES == 0u) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }

#ifdef ATTN_SPLITK
    /* A PARTIAL goes out as dim_tiles + 2 WHOLE tiles (O, m, l; protocol §7), unscattered. */
    if (fta_is_partial(&shape)) {
        for (uint32_t t = 0; t < fta_partial_tiles(shape.dim_tiles); t++) {
            if (ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE) &&
                !ctx->aborting) {
                ft_tensix_noc_write(ft_tensix_cb_read_pointer(CB_OUT),
                                    ft_tensix_noc_address(x, y, base + t * TILE_BYTES),
                                    TILE_BYTES);
                ft_tensix_noc_write_barrier();
            }
            ft_tensix_cb_pop_front(CB_OUT, 1);
        }
        return !ctx->aborting;
    }
#endif
    for (uint32_t dt = 0; dt < shape.dim_tiles; dt++) {
        const bool have_tile = ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE);

        if (have_tile) {
            if (!scatter_one_tile(&shape, ft_tensix_cb_read_pointer(CB_OUT), x, y, base, dt)) {
                ftq_abort(ctx, kErrBadParameters);
                ft_tensix_cb_pop_front(CB_OUT, 1);
                return false;
            }
        }
        ft_tensix_cb_pop_front(CB_OUT, 1);
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
        // filling the input ones. Publishing waits on `tail`, below.
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);

        volatile FTQ_L1_PTR uint32_t *descriptor = ftq_descriptor(&ctx);
        const uint32_t opcode = descriptor[kDescOpcode];
        const uint32_t token = descriptor[kDescToken];
        const uint32_t group_sequence = descriptor[kDescA5];
        bool done = true;

        ftq_status_opcode(&ctx, opcode);
        if (opcode == kOpAttention) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            done = write_one_attention(&ctx, ftq_parameters(descriptor));
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

            if (opcode == kOpAttention && (pull & ~kGroupPullFlags) > 1u &&
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
