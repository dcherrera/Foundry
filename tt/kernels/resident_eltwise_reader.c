/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_eltwise_reader.c
 * @brief Reader third of the RESIDENT ELEMENTWISE triad: stream `tile_count` tiles of A, and
 * of B when the function is binary, from DRAM into the two input circular buffers.
 *
 * IT IS THE BLOCK-PREFETCH READER, AND IT IS THE EASY CASE OF IT. matmul_reader.c had
 * to argue that a block's A tiles are consecutive and that its B tiles are only
 * consecutive under an N-major weight grid (docs/tt_kernel_model.md 12). Here BOTH
 * streams are consecutive tile indices unconditionally -- an elementwise op walks each
 * operand once, in order -- so a block of `depth` tiles is ONE noc_async_read on each
 * side and there is no layout question to get wrong. The barrier is per BLOCK, never per
 * tile: it is a per-core drain, not a per-transaction one, and
 * paying it per tile is what cost 5.9x on the matmul (bringup log 16.1).
 *
 * THE ONE INVARIANT: `depth` MUST DIVIDE `tile_count`. cb_reserve_back does not
 * guarantee a contiguous block in general -- the FIFO write pointer only wraps when it
 * lands exactly on the limit -- so a ragged final block moves the pointer off the block
 * grid and every later block straddles the wrap. The host shrinks the request to a
 * divisor (ft_tt_op_block_depth) and the kernel checks it again and falls back to 1,
 * because "always correct, sometimes slower" is the right failure for this.
 *
 * THIS PROCESSOR OWNS `tail`, AND NOTHING ELSE IN THE CONTROL BLOCK
 * (resident_queue_protocol.h rule 1). An aborted descriptor advances nothing.
 *
 *
 * THE TWO INPUT STREAMS CARRY THEIR OWN TILE SIZE, AND THAT IS WHAT LETS A FORMAT SEAM SIT
 * ON AN OP THAT WAS ALREADY HAPPENING. Until 2026-08-25 this kernel had ONE `TILE_BYTES` for
 * both streams, so a descriptor whose A and B operands are different formats needed a whole
 * separate program (`render_widen` / `render_narrow`, which read one strip and add zero to
 * it). `ft_tt_op_eltwise_place_cbs()` has always sized CB_IN1 from `input_b.tile_bytes`
 * independently, and `resident_eltwise_compute.c` reads its formats per CB out of
 * `ft_tensix_cb_formats.h`, so this reader was the ONLY thing holding the two together. With
 * arg 4 the SILU-gated multiply can take a Float32 B operand under a MIXED CB table and get
 * exactly the truncating fp32->bf16 unpack `render_narrow` performs today — see
 * kernels/cb_tables/gate_narrow.cbtable.
 *
 * ARG 4 IS REFUSED WHEN ZERO ON A BINARY DESCRIPTOR rather than defaulted to arg 2. A short
 * CTARGS line folds an out-of-range read to a literal `0` with no diagnostic (that is what
 * `make check-kernel-compile-time-args` exists for), and a B stream read at 0 bytes a tile
 * is a silent wrong answer; refusing makes it a device error. Every existing recipe passes
 * its own A size there, so their images are byte-identical to the pre-2026-08-25 build.
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_IN0 index   1: CB_IN1 index   2: A tile size in bytes
 *   3: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 *   4: B tile size in bytes -- equal to arg 2 for every same-format program
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0,1,2048,1,2048 NOC=0 KERNEL_TEXT_BASE=0x15000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_eltwise_reader.c brisc /tmp/ttk_eltwise
 *
 * @status BUILT ON THE CARD HOST 2026-08-12, NOT RUN -- text 1876 B on BRISC. See
 *         tests/test_phase6_vector.c for the on-card acceptance.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_queue_protocol.h"
#include "resident_precondition.h"
#include "resident_vector_protocol.h"

#define CB_IN0 ft_tensix_compile_time_arg(0)
#define CB_IN1 ft_tensix_compile_time_arg(1)
#define TILE_BYTES ft_tensix_compile_time_arg(2)
#define WAIT_MODE ft_tensix_compile_time_arg(3)
#define TILE_BYTES_B ft_tensix_compile_time_arg(4)
/**
 * Refuse a descriptor this program cannot execute, INSTEAD of writing outside the circular
 * buffers.
 *
 * The header above states the ONE-SHOT invariant — "`depth` must divide `tile_count`" — and
 * `ftv_block_depth()` enforces exactly that. IT IS NOT ENOUGH ON A RESIDENT KERNEL. The FIFO
 * write pointer wraps only on an EXACT hit of the limit, so a
 * descriptor also has to leave the pointer where it found it, which needs
 * `tile_count % pages == 0` as well as `pages % depth == 0`. Without both, descriptor N+1's
 * blocks step OVER the limit and walk up L1 — measured on the top-k reader, where a read
 * landed on the queue control block at 0x170000 and nothing faulted.
 *
 * `ft_tensix_cb_push_grid_is_safe()` carries the derivation. CB_IN1 is tested only when the
 * function is binary, because a unary descriptor never pushes into it.
 */
static bool descriptor_is_executable(ftq_context *ctx, uint32_t tile_count, uint32_t depth,
                                     bool binary)
{
    if (tile_count == 0u || depth == 0u || TILE_BYTES == 0u)
        ctx->error = kErrBadParameters;
    else if (binary && TILE_BYTES_B == 0u)
        ctx->error = kErrBadParameters;
    else if (!ft_tensix_cb_push_grid_is_safe(CB_IN0, depth))
        ctx->error = kErrBadParameters;
    else if (binary && !ft_tensix_cb_push_grid_is_safe(CB_IN1, depth))
        ctx->error = kErrBadParameters;
    else
        return true;
    return false;
}

/**
 * The reader half of one ELTWISE descriptor.
 *
 * @return true if the whole descriptor was read; false if a wait was abandoned, in which
 *         case the loop still ran to completion (rule 3) and the caller must not retire it
 */
static bool read_one_eltwise(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t function = parameters[kVecFunction];
    const uint32_t tile_count = parameters[kVecTileCount];
    const uint32_t a_x = parameters[kVecAX];
    const uint32_t a_y = parameters[kVecAY];
    const uint32_t a_base = parameters[kVecABase];
    const uint32_t b_x = parameters[kVecBX];
    const uint32_t b_y = parameters[kVecBY];
    const uint32_t b_base = parameters[kVecBBase];
    const uint32_t depth = ftv_block_depth(tile_count, ftv_depth_word(parameters[kVecDepth]));
    const bool probe_no_read = (ftv_probe_bits(parameters[kVecDepth]) & kVecProbeNoRead) != 0u;
    const bool binary = ftv_function_is_binary(function);

    if (!descriptor_is_executable(ctx, tile_count, depth, binary)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }

    for (uint32_t tile = 0; tile < tile_count; tile += depth) {
        uint32_t a_write;
        uint32_t b_write = 0u;

        if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN0, (int32_t)depth, WAIT_MODE))
            return false;
        a_write = ft_tensix_cb_write_pointer(CB_IN0);
        if (binary) {
            if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN1, (int32_t)depth, WAIT_MODE))
                return false;
            b_write = ft_tensix_cb_write_pointer(CB_IN1);
        }

        // Rule 3: while draining, issue NOTHING. The pushes below still happen so the page
        // counters stay balanced, but a transaction issued during a drain would still be in
        // flight when the kernel returns.
        if (!ctx->aborting && !probe_no_read) {
            ft_tensix_noc_read(ft_tensix_noc_address(a_x, a_y, a_base + tile * TILE_BYTES), a_write,
                               depth * TILE_BYTES);
            if (binary) {
                ft_tensix_noc_read(ft_tensix_noc_address(b_x, b_y, b_base + tile * TILE_BYTES_B),
                                   b_write,
                                   depth * TILE_BYTES_B);
            }
            // ONE barrier for the whole block: it is per-core, not per-transaction.
            ft_tensix_noc_read_barrier();
        }

        ft_tensix_cb_push_back(CB_IN0, depth);
        if (binary)
            ft_tensix_cb_push_back(CB_IN1, depth);
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

        if (opcode == kOpEltwise) {
            ftq_status_set(&ctx, kStateCbReserveBack);
            done = read_one_eltwise(&ctx, ftq_parameters(descriptor));
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
