/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_topk_compute.c
 * @brief Compute third of the RESIDENT TOPK/SAMPLE program -- and it deliberately computes
 *   NOTHING. This file exists to say so in the one place a reader will look, and to keep the
 *   three TRISCs following the queue so the program is a normal five-image resident triad
 *   with the normal stop protocol rather than a special case.
 *
 * ===========================================================================
 * WHY THE SFPU IS IDLE FOR THIS OP
 * ===========================================================================
 * Top-k is a sorting network over a stream, not a tile map, and the three reasons are in
 * resident_topk_protocol.h 2. The short form: the answer is an INDEX and the FPU/SFPU
 * produce values; the row is 6.2x too big for L1 so any exact answer is a streaming
 * reduction whatever computes it; and the state is 64 scalar pairs, which is what a RISC-V
 * is for. Forcing it into reader/compute/writer would move data through DST to compute
 * something DST cannot express.
 *
 * Blackhole DOES ship a bitonic top-k in the SFPU -- `topk_local_sort`, `topk_merge`,
 * `topk_rebuild` and `max_reduce_with_indices` -- carrying index tiles beside the
 * data. It was READ and rejected for this op, not overlooked: it sorts 64 values spanning
 * TWO tiles along the COLUMN axis while our row lies along W, wants K in {4,8,16,32,64},
 * runs in APPROXIMATE mode, and needs the sorted run resident in DST. Wt here is 4080,
 * which is neither a power of two nor L1-resident.
 *
 * ===========================================================================
 * WHAT WOULD PUT IT BACK TO WORK, AND THE NUMBER THAT DECIDES
 * ===========================================================================
 * There IS a real two-stage shape and it is worth writing down rather than rediscovering:
 *
 *   stage 1 (here): `reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>` per tile gives the
 *                   maximum of that tile's 32 logits for free, as part of a dataflow the
 *                   op is paying for anyway;
 *   stage 2 (the selector): the N-th largest TILE maximum is a valid threshold, because
 *                   each of the top-N tile maxima is itself an element of the row -- so
 *                   {x : x >= T} contains the true top-N, and only tiles that clear T need
 *                   their 32 logits looked at.
 *
 * That cuts the selector's per-element work from 130,560 to ~N*32 + Wt.
 *
 * AND THE MEASUREMENT NOW SAYS TO BUILD IT. The question was only ever whether the scalar
 * scan exceeds the DRAM stream underneath it, because the reject path is already one
 * subtract and one branch per logit (protocol 3). On the card, on the real row: the whole
 * op is 2.590 ms a token, the drain-only arm -- same binary, same descriptors, selection
 * switched off by a runtime word -- is 0.295 ms, so the SELECTION IS 2.295 ms, i.e. 89% of
 * it, at ~14 cycles a logit against the ~4 the instruction sequence suggests. The prefilter
 * attacks exactly that term and the DRAM stream it would have to hide behind is 8x smaller.
 * Two things to check first, because the 14 cycles are not explained: the data-movement
 * kernels are built `-Os` under LTO (build_tensix_dataflow_kernel.sh), and an L1 load
 * latency the in-order core cannot hide would show up the same way.
 *
 * Compile-time args:
 *   0: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build:
 *   CTARGS=1 CB_TABLE=kernels/cb_tables/topk_float16b.cbtable \
 *       bash foundry/tt/build_tensix_kernel.sh \
 *           foundry/tt/kernels/resident_topk_compute.c /tmp/ttk_topk
 *
 * HELPERS (2026-10-21): with a helper area named, each TRISC scans a quarter of the tiles
 * (help_one_topk); plain scalar C, still no Tensix instruction.
 *
 * @status CONVERTED, MEASURED 2026-08-13. NOT YET THE SHIPPED KERNEL --
 *   resident_topk_compute.c is retained as the baseline. Against it with
 *   tools/diff_tensix_kernel_instructions.sh: all three images the SAME SIZE (784 / 788 /
 *   772 B) and the same instruction sequence except ONE instruction each --
 *   `and a5,t4,a4` against `and a5,a4,t4`, the power-of-two test in ftq_check_control. It is
 *   a commutative operand swap in the register allocation, and it is NOT source-controllable:
 *   writing the expression the other way round leaves the C++ image byte-identical and the C
 *   image unchanged, so GCC canonicalises it and the two frontends rank the operands
 *   differently. Zero Tensix instructions in any image, as before.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "resident_queue_protocol.h"
#include "resident_topk_protocol.h"

/* HELPERS (resident_topk_protocol.h "HELPERS"): this TRISC scans tiles t % 4 == index + 1. */
#if defined(FT_TRISC_UNPACK)
#define TOPK_HELPER_INDEX 0u
#elif defined(FT_TRISC_MATH)
#define TOPK_HELPER_INDEX 1u
#else
#define TOPK_HELPER_INDEX 2u
#endif

/** Wait until the L1 word reaches `want`, stoppable and bounded. */
static bool helper_wait_word(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *word, uint32_t want)
{
    uint32_t spins = 0u;

    while (*word != want) {
        if (ftq_stop_requested(ctx) || ++spins >= ctx->cb_spin_limit) {
            ftq_abort(ctx, ftq_stop_requested(ctx) ? kErrNone : kErrCbWaitTimeout);
            return false;
        }
    }
    return true;
}

/**
 * This TRISC's share of one TOPK descriptor with helpers: for every block NCRISC publishes,
 * offer this processor's tiles to its own selection per row (in the helper area, L1), post
 * the block's stamp; after the last block post the final stamp. The trip count is the
 * descriptor's, so a helper never waits for a block that will not come.
 */
static void help_one_topk(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t tile_count = parameters[kTopkTileCount];
    const uint32_t row = parameters[kTopkRow];
    const uint32_t rows = parameters[kTopkRows] == 0u ? 1u : parameters[kTopkRows];
    const uint32_t base = parameters[kTopkHelperArea];
    volatile FTQ_L1_PTR uint32_t *area = (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)base;
    ftk_select *selection = (ftk_select *)(uintptr_t)(base + kTopkHelperSelectOffset +
                                                      TOPK_HELPER_INDEX * kTopkMaxRows *
                                                          (uint32_t)sizeof(ftk_select));

    if (rows > (uint32_t)kTopkMaxRows || row + rows > 32u)
        return;
    for (uint32_t k = 0u; k < rows; k++)
        ftk_select_init(&selection[k], parameters[kTopkSelectCount]);
    for (uint32_t tile = 0u, block = 0u, depth = 1u; tile < tile_count; tile += depth, block++) {
        uint32_t page, tile_bytes;

        if (!helper_wait_word(ctx, area + kTopkHelperStamp, ftk_stamp(ctx->cursor, block + 1u)))
            return;
        /* NCRISC's own block depth (written before its first stamp): no division here. */
        depth = area[kTopkHelperDepth];
        page = area[kTopkHelperPage];
        tile_bytes = area[kTopkHelperTileBytes];
        if (depth == 0u)
            return;
        for (uint32_t within = 0u; within < depth; within++) {
            if (((tile + within) & 3u) != TOPK_HELPER_INDEX + 1u)
                continue;
            for (uint32_t k = 0u; k < rows; k++)
                ftk_offer_tile_row(&selection[k], page + within * tile_bytes, row + k,
                                   (tile + within) * 32u);
        }
        area[kTopkHelperDone + TOPK_HELPER_INDEX] = ftk_stamp(ctx->cursor, block + 1u);
    }
    __asm__ volatile("" ::: "memory");
    area[kTopkHelperDone + TOPK_HELPER_INDEX] = ftk_stamp(ctx->cursor, kTopkStampFinal);
}

void kernel_main(void)
{
    ftq_context ctx;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);

    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    ctx.cursor = ctx.control[kFieldCompletedCount];

    /* No compute_kernel_hw_startup(): this image issues no Tensix instruction at all, so
     * there is no unpacker source order to set and nothing for the backend to configure.
     * That is also what makes the drain rule (resident_queue_protocol.h rule 3) trivially
     * satisfied here -- there is no dvalid handshake and no MATH/PACK semaphore in flight,
     * because none was ever started. */
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
        if (opcode == kOpStop)
            running = false;
        else if (opcode == kOpTopk && ftk_helpers_wanted(ftq_parameters(descriptor))) {
            help_one_topk(&ctx, ftq_parameters(descriptor));
            if (ctx.aborting)
                running = false;
        }

        ctx.cursor++;
    }

    ftq_status_set(&ctx, kStateExit);
}
