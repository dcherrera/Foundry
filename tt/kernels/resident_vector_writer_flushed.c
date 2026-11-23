/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_vector_writer_flushed.c
 * @brief Vector-family writer, measurement arm "flush per tile, one write barrier per
 * descriptor".
 *
 * Same job as the shipped resident_vector_writer.c: drain a descriptor's output tiles from
 * CB_OUT to (x, y, base) over NoC 1 and publish completion for the triad. The one difference
 * is the barrier shape: after each tile's NoC write it waits only until the write request has
 * left L1 (writes-flushed), which is all the CB pop needs, and it takes a single full write
 * barrier after the descriptor's last tile, which is what publishing needs.
 *
 * Not in the shipped program set. Built only for the writer A/B of tests/test_phase6_fusion.c
 * (program directory "eltwise_flushed").
 *
 * Compile-time args:
 *   0: CB_OUT index   1: output tile size in bytes
 *   2: wait mode -- 0 upstream, 1 stoppable, 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h.
 *
 * Build (NCRISC, NoC 1):
 *   CTARGS=16,2048,1 NOC=1 KERNEL_TEXT_BASE=0x18000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_vector_writer_flushed.c ncrisc build/ttk_eltwise_flushed
 *
 * @status Clean-room rewrite (Phase 4); byte gate against the eltwise_flushed reference image.
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

/**
 * Drain one descriptor's output tiles. The pop is unconditional so the page counters stay
 * balanced on the aborting path; the trailing write barrier runs on every path except the
 * zero-tile-size refusal (which issues no write).
 */
static bool write_one_vector_op_flushed(ftq_context *ctx,
                                        volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t tile_count = parameters[kVecTileCount];
    const uint32_t destination_x = parameters[kVecCX];
    const uint32_t destination_y = parameters[kVecCY];
    const uint32_t destination_base = parameters[kVecCBase];

    if (OUT_TILE_BYTES == 0u) {
        ctx->error = kErrBadParameters;
        return false;
    }

    for (uint32_t tile = 0; tile < tile_count; tile++) {
        const bool have_tile = ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE);

        if (have_tile && !ctx->aborting) {
            ft_tensix_noc_write(ft_tensix_cb_read_pointer(CB_OUT),
                                ft_tensix_noc_address(destination_x, destination_y,
                                                      destination_base + tile * OUT_TILE_BYTES),
                                OUT_TILE_BYTES);
            ft_tensix_noc_writes_flushed_barrier();
        }
        ft_tensix_cb_pop_front(CB_OUT, 1);
    }
    ft_tensix_noc_write_barrier();
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
            done = write_one_vector_op_flushed(&ctx, ftq_parameters(descriptor));
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
