/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_gdngate_writer.c
 * @brief Writer half of the RESIDENT GDN GATE pair: puts each block the reader assembled at
 * its slot in DRAM and publishes the descriptor's completion. resident_gdngate_protocol.h has
 * the layout; resident_gdngate_reader.c the arithmetic.
 *
 * The writer walks the SAME (band, sub) sequence as the reader from the same parameter words
 * (`ftgg_sub_live_rows()`), so the n-th page it pops is the n-th block the reader pushed.
 *
 * THIS PROCESSOR OWNS `completed_count` AND `completed_token` (resident_queue_protocol.h
 * rule 1), and publishes only after the reader's `tail` passed the descriptor.
 *
 * Compile-time args:
 *   0: CB_OUT index
 *   1: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h
 *
 * Build (NCRISC, NoC 1):
 *   CTARGS=16,1 NOC=1 KERNEL_TEXT_BASE=0x18000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_gdngate_writer.c ncrisc <out>
 *
 * @status Written PM 7 (2026-10-08). Unexecuted.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_gdngate_protocol.h"
#include "resident_queue_protocol.h"

#define CB_OUT ft_tensix_compile_time_arg(0)
#define WAIT_MODE ft_tensix_compile_time_arg(1)

/**
 * The writer half of one GDN_GATE descriptor.
 * @return true if every block was written; false if refused or a wait was abandoned
 */
static bool write_one_gate_op(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t rows = parameters[kGgRows];
    const uint32_t band_count = parameters[kGgBandCount];
    const uint32_t sub_rows = parameters[kGgSubRows];
    const uint32_t heads = parameters[kGgHeads];
    const uint32_t dst_x = parameters[kGgDstXY] & 0xFFFFu;
    const uint32_t dst_y = parameters[kGgDstXY] >> 16;
    const uint32_t dst_base = parameters[kGgDstBase];
    const uint32_t block_stride = parameters[kGgBlockStride];
    const uint32_t block_bytes = ftgg_block_bytes(heads);

    const uint32_t flags = parameters[kGgFlags];

    // ITEM 6 FLAT: one page; this slice's two runs (decay, beta) of every row.
    if ((flags & kGgFlagFlat) != 0u) {
        const uint32_t first = parameters[kGgSubRows], total = parameters[kGgBandStride];
        const uint32_t run = heads * 4u;

        if (rows == 0u || band_count != 1u || heads == 0u || total > kGgMaxHeads ||
            first + heads > total || block_stride < 2u * total * 4u ||
            (block_stride % 16u) != 0u || (dst_base % 16u) != 0u ||
            ftgg_flat_bytes(rows, total, block_stride) > ftgg_block_bytes(kGgMaxHeads) ||
            !ft_tensix_cb_pop_grid_is_safe(CB_OUT, 1u)) {
            ftq_abort(ctx, kErrBadParameters);
            return false;
        }
        if (ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE) && !ctx->aborting) {
            const uint32_t page = ft_tensix_cb_read_pointer(CB_OUT);

            for (uint32_t row = 0; row < rows; row++) {
                const uint32_t decay = row * block_stride + first * 4u;
                const uint32_t beta = row * block_stride + (total + first) * 4u;

                ft_tensix_noc_write(page + decay, ft_tensix_noc_address(dst_x, dst_y,
                                                                         dst_base + decay), run);
                ft_tensix_noc_write(page + beta, ft_tensix_noc_address(dst_x, dst_y,
                                                                        dst_base + beta), run);
            }
            ft_tensix_noc_write_barrier();
        }
        ft_tensix_cb_pop_front(CB_OUT, 1u);
        return !ctx->aborting;
    }
    // The reader refuses the same shapes; the writer must not wait for pages it never gets.
    if (rows == 0u || band_count == 0u || band_count > kGgMaxBands ||
        (rows + kGgTileRows - 1u) / kGgTileRows != band_count ||
        (sub_rows != 16u && sub_rows != 32u) || heads == 0u || heads > kGgMaxHeads ||
        block_stride < block_bytes || (block_stride % 64u) != 0u || (dst_base % 64u) != 0u ||
        !ft_tensix_cb_pop_grid_is_safe(CB_OUT, 1u)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    for (uint32_t band = 0; band < band_count; band++) {
        const uint32_t done = band * kGgTileRows;
        const uint32_t band_rows = rows - done < kGgTileRows ? rows - done : kGgTileRows;

        for (uint32_t sub = 0; sub < 2u; sub++) {
            if (ftgg_sub_live_rows(band_rows, sub_rows, sub) == 0u)
                continue;
            if (ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE) &&
                !ctx->aborting) {
                const uint32_t slot = dst_base + (2u * band + sub) * block_stride;

                ft_tensix_noc_write(ft_tensix_cb_read_pointer(CB_OUT),
                                    ft_tensix_noc_address(dst_x, dst_y, slot), block_bytes);
                ft_tensix_noc_write_barrier();
            }
            ft_tensix_cb_pop_front(CB_OUT, 1u);
        }
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
        if (opcode == kOpGdnGate) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            done = write_one_gate_op(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
        } else if (opcode == kOpStop) {
            running = false;
        }
        if (!done) {
            running = false;
            continue;
        }
        // The reader must have finished this descriptor before its completion is published.
        if (!ftq_wait_until_reaches(&ctx, kFieldTail, ctx.cursor + 1u)) {
            running = false;
            continue;
        }
        ctx.cursor++;
        ctx.control[kFieldCompletedToken] = token;
        ctx.control[kFieldCompletedCount] = ++completed;
        ftq_group_post(&ctx, group_sequence, token);
    }
    ftq_status_set(&ctx, kStateExit);
}
