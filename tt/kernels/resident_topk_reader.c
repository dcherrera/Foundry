/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_topk_reader.c
 * @brief Reader third of the RESIDENT TOPK/SAMPLE program: block-prefetch the vocabulary logit
 * row out of DRAM into CB_LOGITS, `depth` tiles per NoC barrier, and own `tail`.
 *
 * IT READS WHOLE TILES FOR ONE ROW, AND THAT IS A MEASURED CHOICE WITH A NAMED
 * ALTERNATIVE. The LM head runs at Mt = 1, so it writes 32 token rows and decode wants
 * one: of a tile's 2048 bytes exactly 64 are the row this op is selecting over
 * (two 32-byte face rows, docs/tt_kernel_model.md §9). A reader could fetch only those,
 * cutting DRAM traffic 32x -- at the cost of 2 transactions per tile, i.e. 8160 of them
 * for MiniCPM5-1B against ~85 block reads at depth 48. §12's rule is that transaction
 * COUNT is what binds on this part and byte count past ~1088 B a tile is not, so the
 * strided reader is very unlikely to win and is NOT implemented on that reasoning
 * (INFERRED, not measured -- the arithmetic is in the report and it is one runtime-arg
 * away if someone wants the measurement).
 *
 * THE ROW DOES NOT FIT L1 AND THAT IS WHY THIS IS A READER AT ALL. 4080 tiles x 2048 B is
 * 8,355,840 B against a 1,351,680 B arena, so there is no "load it and sort it" version
 * of this op at any tile size -- see resident_topk_protocol.h §2.
 *
 * THIS PROCESSOR OWNS `tail`, AND NOTHING ELSE IN THE CONTROL BLOCK.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_LOGITS index   1: tile size in bytes
 *   2: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0,2048,1 NOC=0 KERNEL_TEXT_BASE=0x15000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_topk_reader.c brisc /tmp/ttk_topk
 *
 * @status GREEN ON SILICON (Tensix 51, 57, 59; 40 checks, 0 failures). text 1616 B on
 *         BRISC. Five kernel mutations were run through the unmodified test binary and
 *         all five go red.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_queue_protocol.h"
#include "resident_precondition.h"
#include "resident_topk_protocol.h"

#define CB_LOGITS ft_tensix_compile_time_arg(0)
#define TILE_BYTES ft_tensix_compile_time_arg(1)
#define WAIT_MODE ft_tensix_compile_time_arg(2)
/**
 * The depth a block may actually use: it must DIVIDE the tile count, or the CB write
 * pointer leaves the block grid and every later block straddles the wrap. The kernel repeats the host's arithmetic rather than trusting
 * it, exactly as resident_matmul_reader.c does; falling back to 1 is always correct.
 */
static uint32_t block_depth(uint32_t run, uint32_t requested)
{
    return (requested > 0u && run > 0u && run % requested == 0u) ? requested : 1u;
}

/**
 * Refuse a descriptor this program cannot execute, INSTEAD of blocking on it or writing
 * outside the circular buffer. Four conditions, and three of the four were paid for.
 *
 * 1. **A ZERO-LENGTH NoC READ.** A sibling op's card-down on 2026-08-12 traced to a computed
 *    transfer length that came out zero, and the working hypothesis — INFERRED, and
 *    deliberately not re-confirmed because confirming it costs the card for everyone — is
 *    that a zero-length NoC read takes the chip down rather than returning nothing. This
 *    op's length is `depth * TILE_BYTES` with both factors already non-zero, so the guard is
 *    belt and braces; it is here anyway because "the arithmetic cannot reach zero" is exactly
 *    the sentence that was true of the other op until a branch made it false.
 *
 * 2. **A BLOCK BIGGER THAN THE CIRCULAR BUFFER.** `cb_reserve_back(depth)` on a CB holding
 *    fewer than `depth` pages is not slow, it is UNSATISFIABLE — and the page count is fixed
 *    when the RESIDENT program is staged while the depth is a per-descriptor word, so the
 *    host CAN ask for it. It did, on the first hardware run: six red checks, ten seconds of
 *    blocked polling and a dead residency, reported as an idle timeout, which it was not.
 *
 * 3 and 4. **THE BLOCK-PREFETCH INVARIANT IS NOT "DEPTH DIVIDES THE RUN" ON A RESIDENT OP,
 *    AND THE DIFFERENCE OVERWRITES L1.** docs/tt_kernel_model.md §12 states the rule for a
 *    ONE-SHOT kernel, where the FIFO write pointer starts at the base. A resident kernel
 *    runs many descriptors, and `cb_push_back` only wraps when the pointer lands EXACTLY on
 *    the limit — so a descriptor that leaves the pointer off the
 *    next descriptor's block grid makes the next block STEP OVER the limit instead of
 *    wrapping, and every block after it walks further up L1. MEASURED, on this op, on the
 *    card: 128 tiles at depth 32 through a 96-page buffer leaves the pointer 32 pages in;
 *    the next descriptor's 48-page blocks then marched past the arena and a
 *    `noc_async_read` landed on the QUEUE CONTROL BLOCK at 0x170000, which came back
 *    holding the test's own decoy value. Nothing faulted and nothing said so.
 *
 *    Two conditions make it impossible by induction, given the pointer starts at the base:
 *      - `pages % depth == 0`      — every block wrap lands exactly on the limit;
 *      - `tile_count % pages == 0` — the descriptor leaves the pointer where it found it.
 *
 *    This is a fact about the FAMILY, not about this op: any resident reader whose
 *    descriptors vary the depth, or whose tile count is not a whole number of buffer cycles,
 *    has the same hole.
 */
static bool descriptor_is_executable(ftq_context *ctx, uint32_t tile_count, uint32_t depth)
{
    const uint32_t pages = ft_tensix_cb(CB_LOGITS)->fifo_num_pages;

    if (tile_count == 0u || depth == 0u || TILE_BYTES == 0u || pages == 0u)
        ctx->error = kErrBadParameters;
    else if (depth > pages || (pages % depth) != 0u || (tile_count % pages) != 0u)
        ctx->error = kErrBadParameters;
    else
        return true;
    return false;
}

/**
 * The reader half of one TOPK descriptor.
 *
 * @return true if the whole row was read; false if a wait was abandoned, in which case the
 *         loop still ran to completion (rule 3) and the caller must not retire it
 */
static bool read_one_logit_row(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t tile_count = parameters[kTopkTileCount];
    const uint32_t logits_x = parameters[kTopkLogitsX];
    const uint32_t logits_y = parameters[kTopkLogitsY];
    const uint32_t logits_base = parameters[kTopkLogitsBase];
    const uint32_t depth = block_depth(tile_count, parameters[kTopkDepth]);
    const uint32_t block_bytes = depth * TILE_BYTES;

    if (!descriptor_is_executable(ctx, tile_count, depth))
        return false;

    for (uint32_t tile = 0; tile < tile_count; tile += depth) {
        uint32_t write_address;

        if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_LOGITS, (int32_t)depth, WAIT_MODE))
            return false;
        write_address = ft_tensix_cb_write_pointer(CB_LOGITS);

        // Rule 3: while draining, issue NOTHING. The push still happens so the page
        // counters stay balanced, but a transaction issued during a drain would still be in
        // flight when the kernel returns.
        // The zero-length guard is repeated AT THE ISSUE SITE and not only at the top,
        // because the only version of this check that is worth anything is the one the
        // transfer cannot get past.
        if (!ctx->aborting && block_bytes != 0u) {
            // Consecutive tile indices are consecutive DRAM bytes and land in a contiguous
            // CB block by the depth invariant, so the whole block is ONE read and ONE
            // barrier -- the pattern that was worth 5.9x (docs/tt_kernel_model.md §12).
            ft_tensix_noc_read(ft_tensix_noc_address(logits_x, logits_y,
                                                     logits_base + tile * TILE_BYTES),
                               write_address, block_bytes);
            ft_tensix_noc_read_barrier();
        }
        ft_tensix_cb_push_back(CB_LOGITS, depth);
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

        if (opcode == kOpTopk) {
            ftq_status_set(&ctx, kStateCbReserveBack);
            done = read_one_logit_row(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
            if (!done)
                running = false;
        } else if (opcode == kOpSample || opcode == kOpNop) {
            // SAMPLE moves no tiles: it reads a result block the previous TOPK left in L1.
            // The reader still walks the descriptor so `tail` keeps naming the same item
            // every other processor is on.
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
