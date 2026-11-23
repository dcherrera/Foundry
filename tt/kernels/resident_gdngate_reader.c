/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_gdngate_reader.c
 * @brief Reader half of the RESIDENT GDN GATE pair: reads each band's alpha and beta tiles and
 * the layer's constants, computes every position's decay and beta with ft_det_math.h, and
 * assembles each 16-row sub-chunk's COMPACT gate block in the output CB.
 * resident_gdngate_writer.c puts the blocks in DRAM. resident_gdngate_protocol.h has the
 * layout and the reasons.
 *
 * THE ARITHMETIC IS THE HOST'S, NOT A COPY OF IT: the host compiles the same ft_det_math.h,
 * which is integer-only, so the bits cannot depend on which compiler, flags or FPU ran it.
 * That is the whole claim this family rests on and tests/test_det_math.c + the card sweep
 * are its proof.
 *
 * THIS PROCESSOR OWNS `tail`, AND NOTHING ELSE IN THE CONTROL BLOCK (resident_queue_protocol.h
 * rule 1). An aborted descriptor advances nothing.
 *
 * Compile-time args:
 *   0: CB_STAGE index (addressed, never cycled: 2 tiles + the 256-byte constants block)
 *   1: CB_OUT index (one page = one block)
 *   2: tile size in bytes (2048, bf16)
 *   3: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on every processor
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0,16,2048,1 NOC=0 KERNEL_TEXT_BASE=0x15000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_gdngate_reader.c brisc <out>
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
#include "resident_precondition.h"

#define CB_STAGE ft_tensix_compile_time_arg(0)
#define CB_OUT ft_tensix_compile_time_arg(1)
#define TILE_BYTES ft_tensix_compile_time_arg(2)
#define WAIT_MODE ft_tensix_compile_time_arg(3)

/** Everything one descriptor says, read once. */
typedef struct gate_request {
    uint32_t flags, rows, band_count, sub_rows, heads;
    uint32_t alpha_x, alpha_y, alpha_base;
    uint32_t beta_x, beta_y, beta_base;
    uint32_t band_stride;
    uint32_t const_x, const_y, const_base;
    uint32_t dst_base, block_stride;
    uint32_t scale_bits;
} gate_request;

static void load_request(gate_request *request, volatile FTQ_L1_PTR uint32_t *parameters)
{
    request->flags = parameters[kGgFlags];
    request->rows = parameters[kGgRows];
    request->band_count = parameters[kGgBandCount];
    request->sub_rows = parameters[kGgSubRows];
    request->heads = parameters[kGgHeads];
    request->alpha_x = parameters[kGgAlphaXY] & 0xFFFFu;
    request->alpha_y = parameters[kGgAlphaXY] >> 16;
    request->alpha_base = parameters[kGgAlphaBase];
    request->beta_x = parameters[kGgBetaXY] & 0xFFFFu;
    request->beta_y = parameters[kGgBetaXY] >> 16;
    request->beta_base = parameters[kGgBetaBase];
    request->band_stride = parameters[kGgBandStride];
    request->const_x = parameters[kGgConstXY] & 0xFFFFu;
    request->const_y = parameters[kGgConstXY] >> 16;
    request->const_base = parameters[kGgConstBase];
    request->dst_base = parameters[kGgDstBase];
    request->block_stride = parameters[kGgBlockStride];
    request->scale_bits = parameters[kGgScaleBits];
}

/** Refuse rather than half-execute: a parameter block is a pointer in a descriptor, and a
 *  wrong one becomes NoC traffic at addresses nobody chose (spec §7.3). */
static bool request_is_sane(const gate_request *request)
{
    if (request->rows == 0u || request->band_count == 0u || request->band_count > kGgMaxBands)
        return false;
    if ((request->rows + kGgTileRows - 1u) / kGgTileRows != request->band_count)
        return false;
    if (request->heads == 0u || request->heads > kGgMaxHeads)
        return false;
    if (!ft_tensix_cb_push_grid_is_safe(CB_OUT, 1u))
        return false;
    // The writer's refusals too: a descriptor one half refuses and the other runs would
    // leave the reader filling a CB nobody drains.
    if ((request->flags & kGgFlagFlat) != 0u) {
        // ONE band, a head slice (first = sub_rows word) inside the total (band_stride word),
        // and the rows' span inside the one page.
        const uint32_t total = request->band_stride;

        return request->band_count == 1u && total <= kGgMaxHeads &&
               request->sub_rows + request->heads <= total &&
               request->block_stride >= 2u * total * 4u && (request->block_stride % 16u) == 0u &&
               (request->dst_base % 16u) == 0u &&
               ftgg_flat_bytes(request->rows, total, request->block_stride) <=
                   ftgg_block_bytes(kGgMaxHeads);
    }
    if (request->sub_rows != 16u && request->sub_rows != 32u)
        return false;
    if (request->band_stride < TILE_BYTES || (request->band_stride % 64u) != 0u)
        return false;
    return request->block_stride >= ftgg_block_bytes(request->heads) &&
           (request->block_stride % 64u) == 0u && (request->dst_base % 64u) == 0u;
}

/** Read band `band`'s two tiles into the stage. */
static void read_band(const gate_request *request, uint32_t band, uint32_t stage)
{
    const uint32_t offset = band * request->band_stride;

    ft_tensix_noc_read(ft_tensix_noc_address(request->alpha_x, request->alpha_y,
                                             request->alpha_base + offset),
                       stage, TILE_BYTES);
    ft_tensix_noc_read(ft_tensix_noc_address(request->beta_x, request->beta_y,
                                             request->beta_base + offset),
                       stage + TILE_BYTES, TILE_BYTES);
    ft_tensix_noc_read_barrier();
}

/**
 * The reader half of one GDN_GATE descriptor.
 * @return true if every block was assembled; false if a wait was abandoned (the loop still
 *         ran to completion, rule 3) or the descriptor was refused
 */
static bool read_one_gate_op(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t stage = ft_tensix_cb_write_pointer(CB_STAGE);
    const uint32_t constants = stage + 2u * TILE_BYTES;
    gate_request request;

    if (TILE_BYTES == 0u) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    load_request(&request, parameters);
    if (!request_is_sane(&request)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    ft_tensix_noc_read(ft_tensix_noc_address(request.const_x, request.const_y,
                                             request.const_base),
                       constants, kGgConstWords * 4u);
    ft_tensix_noc_read_barrier();

    if ((request.flags & kGgFlagFlat) != 0u) {
        if (!ctx->aborting)
            read_band(&request, 0u, stage);
        if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE))
            return false;
        if (!ctx->aborting)
            ftgg_build_flat((uint32_t *)(uintptr_t)ft_tensix_cb_write_pointer(CB_OUT),
                            request.sub_rows, request.heads, request.band_stride,
                            (const uint16_t *)(uintptr_t)stage,
                            (const uint16_t *)(uintptr_t)(stage + TILE_BYTES),
                            (const uint32_t *)(uintptr_t)constants, request.rows,
                            request.block_stride,
                            (request.flags & kGgFlagRedBetaAsDecay) != 0u);
        ft_tensix_cb_push_back(CB_OUT, 1u);
        return !ctx->aborting;
    }
    for (uint32_t band = 0; band < request.band_count; band++) {
        const uint32_t done = band * kGgTileRows;
        const uint32_t band_rows = request.rows - done < kGgTileRows ? request.rows - done
                                                                      : kGgTileRows;

        if (!ctx->aborting)
            read_band(&request, band, stage);
        for (uint32_t sub = 0; sub < 2u; sub++) {
            const uint32_t live = ftgg_sub_live_rows(band_rows, request.sub_rows, sub);

            if (live == 0u)
                continue;
            if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE))
                return false;
            // Rule 3: while draining, compute nothing; the push still balances the counters.
            if (!ctx->aborting)
                ftgg_build_block((uint32_t *)(uintptr_t)ft_tensix_cb_write_pointer(CB_OUT),
                                 request.heads, request.scale_bits,
                                 (const uint16_t *)(uintptr_t)stage,
                                 (const uint16_t *)(uintptr_t)(stage + TILE_BYTES),
                                 (const uint32_t *)(uintptr_t)constants, sub * request.sub_rows,
                                 live, (request.flags & kGgFlagRedBetaAsDecay) != 0u);
            ft_tensix_cb_push_back(CB_OUT, 1u);
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
    // RESUME rather than restart, from completed_count (resident_matmul_reader.c).
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

        if (opcode == kOpGdnGate) {
            ftq_status_set(&ctx, kStateCbReserveBack);
            done = read_one_gate_op(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
            if (!done)
                running = false;
        } else if (opcode == kOpNop) {
            done = true;
        } else if (opcode == kOpStop) {
            done = true;
            running = false;
        } else {
            ctx.error = kErrBadOpcode;
            done = false;
            running = false;
        }
        // Rule 5: an abandoned descriptor advances NOTHING.
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
