/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_vector_writer_blocked.c
 * @brief Vector-family writer, measurement arm "one write barrier per block of tiles".
 *
 * Same job and the same queue/completion behaviour as the shipped resident_vector_writer.c,
 * except it waits for, writes and pops CB_OUT in blocks of WRITE_BLOCK tiles and takes one
 * full NoC write barrier per block. A block is contiguous in the CB only when the CB holds a
 * whole number of blocks and every descriptor is a whole number of CB loads; both are checked
 * before any transfer and a failure is a kErrBadParameters refusal, never a guessed address.
 *
 * Not in the shipped program set. Built only for the writer A/B of tests/test_phase6_fusion.c
 * (program directory "eltwise_blocked"). It documents no build invocation, so
 * tools/check_tensix_kernel_compile_time_arg_counts.py allowlists it with a WARN.
 *
 * Compile-time args:
 *   0: CB_OUT index   1: output tile size in bytes
 *   2: wait mode -- 0 upstream, 1 stoppable, 2 bound-only
 *   3: WRITE_BLOCK, tiles per wait/pop/barrier
 *   4: OUT_PAGES, pages the host gave CB_OUT
 * Runtime args: the ten of resident_queue_protocol.h.
 *
 * @status Clean-room rewrite (Phase 4); byte gate against the eltwise_blocked reference image.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_queue_protocol.h"
#include "resident_vector_protocol.h"

#define CB_OUT ft_tensix_compile_time_arg(0)
#define OUT_TILE_BYTES ft_tensix_compile_time_arg(1)
#define WAIT_MODE ft_tensix_compile_time_arg(2)
#define WRITE_BLOCK ft_tensix_compile_time_arg(3)
#define OUT_PAGES ft_tensix_compile_time_arg(4)

/* Compile-time legality of the block shape (every term folds). */
#define BLOCK_SHAPE_LEGAL                                                                    \
    (WRITE_BLOCK > 0u && OUT_TILE_BYTES > 0u && OUT_PAGES > 0u && OUT_PAGES % WRITE_BLOCK == 0u)

/**
 * Drain one descriptor's output tiles in blocks. The pop is unconditional so the page
 * counters stay balanced on the aborting path; every issued write is barriered inside its
 * block, so there is no trailing barrier.
 */
static bool write_one_vector_op_blocked(ftq_context *ctx,
                                        volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t tile_count = parameters[kVecTileCount];
    const uint32_t destination_x = parameters[kVecCX];
    const uint32_t destination_y = parameters[kVecCY];
    const uint32_t destination_base = parameters[kVecCBase];

    /* Short-circuit keeps the modulo from running with a zero OUT_PAGES. */
    if (!BLOCK_SHAPE_LEGAL || (tile_count % OUT_PAGES) != 0u) {
        ctx->error = kErrBadParameters;
        return false;
    }

    for (uint32_t tile = 0; tile < tile_count; tile += WRITE_BLOCK) {
        const bool have_block =
            ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, (int32_t)WRITE_BLOCK, WAIT_MODE);

        if (have_block && !ctx->aborting) {
            const uint32_t read_base = ft_tensix_cb_read_pointer(CB_OUT);

            for (uint32_t step = 0; step < WRITE_BLOCK; step++) {
                const uint32_t offset = (tile + step) * OUT_TILE_BYTES;

                ft_tensix_noc_write(read_base + step * OUT_TILE_BYTES,
                                    ft_tensix_noc_address(destination_x, destination_y,
                                                          destination_base + offset),
                                    OUT_TILE_BYTES);
            }
            ft_tensix_noc_write_barrier();
        }
        ft_tensix_cb_pop_front(CB_OUT, WRITE_BLOCK);
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
        /* Fetch on head so the output CB drains while the reader is still filling; publish
         * only once tail shows the reader finished this descriptor. */
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
        if (opcode == kOpEltwise || opcode == kOpSoftmax || opcode == kOpReduce) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            done = write_one_vector_op_blocked(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
        } else if (opcode == kOpStop) {
            running = false;
        }

        if (!done) {
            running = false;
            continue;
        }

        if (!ftq_wait_until_reaches(&ctx, kFieldTail, ctx.cursor + 1u)) {
            running = false;
            continue;
        }

        ctx.cursor++;
        /* Token before count: a host that sees the count may trust the token. */
        ctx.control[kFieldCompletedToken] = token;
        ctx.control[kFieldCompletedCount] = ++completed;
        ftq_group_post(&ctx, group_sequence, token);
    }

    ftq_status_set(&ctx, kStateExit);
}
