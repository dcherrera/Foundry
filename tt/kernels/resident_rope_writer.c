/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_rope_writer.c
 * @brief Writer third of the RESIDENT ROPE triad, and the processor that PUBLISHES COMPLETION,
 * per core and per group.
 *
 * IT IS resident_vector_writer.c WITH THREE DIFFERENT WORD OFFSETS, and that is worth
 * saying plainly rather than hiding. A writer is "drain `tile_count` tiles from CB_OUT to
 * (x, y, base)" — it does not know what produced them or what shape they came from — so
 * this is the THIRD copy of one function (matmul, vector, rope), each differing only in
 * where its family's parameter block spells the same four fields. resident_vector_writer.c
 * already names the fix: an "output descriptor" sub-block every family shares, at fixed
 * words, so there is ONE writer for every op there will ever be. This op is the third
 * witness for it; it did not seem right to change two other families' wire formats from
 * inside a RoPE round.
 *
 * WHY THE WRITER OWNS COMPLETION (resident_queue_protocol.h rule 1). An op is not done
 * when the reader has fetched its operands or when the FPU has finished; it is done when
 * the result has left L1. So this processor owns `completed_count` and `completed_token`,
 * the reader owns `tail`, and nothing else writes either.
 *
 * AND WHY IT FETCHES ON `head` BUT PUBLISHES ON `tail`. Fetching on `tail` is a pipeline
 * deadlock, not a precaution — the output CB fills, compute blocks in cb_reserve_back, the
 * reader blocks because nothing drains the inputs, and the reader can never reach the end
 * of the descriptor to bump `tail`.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_OUT index   1: output tile size in bytes
 *   2: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build (NCRISC, NoC 1):
 *   CTARGS=16,2048,1 NOC=1 KERNEL_TEXT_BASE=0x18000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_rope_writer.c ncrisc /tmp/ttk_rope
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_queue_protocol.h"
#include "resident_rope_protocol.h"

#define CB_OUT ft_tensix_compile_time_arg(0)
#define OUT_TILE_BYTES ft_tensix_compile_time_arg(1)
#define WAIT_MODE ft_tensix_compile_time_arg(2)
/**
 * Drain one descriptor's output tiles.
 *
 * The barrier is per tile, as it is in resident_vector_writer.c and
 * resident_matmul_writer.c. That is a known cost and it is deliberately UNCHANGED here:
 * the reader's per-block barrier was worth 5.9x and the writer's has never been measured, so
 * batching it belongs in a round that measures it rather than in the round that introduces
 * the op.
 */
static bool write_one_rope(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t tile_count = parameters[kRopeTileCount];
    const uint32_t head_tiles = parameters[kRopeHeadTiles];
    const uint32_t rot_tiles = parameters[kRopeRotTiles];
    const uint32_t destination_x = parameters[kRopeCX];
    const uint32_t destination_y = parameters[kRopeCY];
    const uint32_t destination_base = parameters[kRopeCBase];

    /* The same refusal the other two threads make, on the same words, before any circular
     * buffer is touched — so a refused descriptor leaves nobody waiting. */
    if (!ftr_shape_is_supported(head_tiles, rot_tiles) || tile_count == 0u ||
        (tile_count % head_tiles) != 0u) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }

    for (uint32_t tile = 0; tile < tile_count; tile++) {
        const bool have_tile = ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE);

        if (have_tile) {
            ft_tensix_noc_write(ft_tensix_cb_read_pointer(CB_OUT),
                                ft_tensix_noc_address(destination_x, destination_y,
                                                      destination_base + tile * OUT_TILE_BYTES),
                                OUT_TILE_BYTES);
            ft_tensix_noc_write_barrier();
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
        if (opcode == kOpRope) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            done = write_one_rope(&ctx, ftq_parameters(descriptor));
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
        // Token before count: a host that sees the count is entitled to trust the token it
        // reads next. Both are plain stores from ONE in-order RISC-V, which is the only
        // ordering argument this protocol makes and the only one it needs — every other
        // control word has a different single writer.
        ctx.control[kFieldCompletedToken] = token;
        ctx.control[kFieldCompletedCount] = ++completed;
        ftq_group_post(&ctx, group_sequence, token);
    }

    // The writer does NOT touch kFieldError or kFieldState: the reader owns those, and one
    // writer per word is the whole ordering argument.
    ftq_status_set(&ctx, kStateExit);
}
