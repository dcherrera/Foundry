/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_reduce_reader.c
 * @brief Reader third of the RESIDENT REDUCE triad: the constants this descriptor's program
 * asks for, then Wt activation tiles per row block, block-prefetched.
 *
 * THE CONSTANTS ARE ONE OPERAND WITH A FIXED TILE LAYOUT:
 *
 *     tile 0            the reduce scaler (1.0, or 1/W for a mean)
 *     tile 1            eps * W / 32, pushed only when a pass adds eps
 *     tiles 2 .. 2+Wt   gamma, pushed only when the final chain multiplies by it
 *     tiles 2+Wt ..     beta,  pushed only when the final chain adds it
 *
 * The layout does NOT move with the flags — an unused eps tile still occupies slot 1.
 * A layout that shifted would be a wrong address for one descriptor in four, and the
 * host builds the buffer either way. rms_norm_reader.c already put its scaler and eps
 * tiles adjacent in one buffer "so they cost one address rather than two"; this is that,
 * generalised to gamma and beta, and it is what keeps the parameter block at sixteen
 * words with a single constants operand instead of three.
 *
 * WHAT IT PUSHES IS DECIDED BY ftr_program_uses_eps() AND THE CHAIN FLAGS, WHICH IS THE
 * SAME FUNCTION THE COMPUTE THREAD POPS FROM. Two processors agreeing about a CB by both
 * reading one predicate is the only arrangement in which they cannot disagree; an arity
 * field on the wire would be a second fact to keep in step (resident_vector_protocol.h,
 * THE ARITY RULE).
 *
 * THE CONSTANTS ARE FETCHED PER DESCRIPTOR, NOT ONCE AT BOOT. rms_norm_reader.c pushes
 * its constants once before its block loop and never pops them, which works for a kernel
 * that runs once. A RESIDENT kernel has no "before the loop" that knows an address: the
 * constants live wherever THIS descriptor's parameter block says. gamma is Wt tiles in one
 * noc_async_read, so the price is one transaction per stream per descriptor.
 *
 * BLOCK PREFETCH, WITH THE RUN BEING ONE ROW. `depth` divides Wt, not the total: a block
 * must not span two row blocks, because the compute thread consumes CB_IN a whole row at
 * a time. Within a row the tile indices are consecutive, so a block is ONE noc_async_read
 * and ONE barrier (docs/tt_kernel_model.md 12).
 *
 * CB_IN HOLDS Wt PAGES FOR A MODEL NORM, SO THE READER AND THE COMPUTE THREAD SERIALISE PER
 * ROW BLOCK — the reader fills the row, blocks in cb_reserve_back until compute pops it, and
 * only then starts the next. A head-norm residency places CB_IN at a whole number of rows
 * (P3: four), which lets this reader run that many heads ahead; every descriptor on it has
 * the one Wt, so a row still starts on the row grid and never straddles the wrap.
 *
 * P3 STREAMED READ (kRedSchedStream, resident_reduce_protocol.h). A decode norm is ONE row
 * block, so double-buffering buys nothing there; what costs is that nothing overlaps the
 * read. The streamed path keeps up to kRedStreamFlight blocks in flight on their own NoC
 * transaction ids (resident_matmul_reader.c's M4 pattern) and pushes each as it lands, so the
 * compute thread's x*x map starts on block 0 while block 3 is still on the NoC. gamma and
 * beta move behind row 0's input: the chain is the last thing that needs them. Half reads
 * (kRedSchedHalfRead) land faces 0-1 of each input and gamma tile — the only faces a
 * half-tile norm program ever unpacks — so the bytes halve as well. The ORDER of the data the
 * compute thread sees is unchanged, which is why none of this can move an output bit.
 *
 * THERE IS NO MASK TILE, AND THE CONTRACT THAT MAKES THAT SAFE IS NOT STATED ANYWHERE THIS
 * KERNEL CAN SEE. Upstream masks a ragged reduction width — softmax builds a `-inf` pad
 * tile — so that a W which is not a multiple of 32 reduces over the columns that exist and
 * not over whatever the pad holds. This op reduces over `tile_cols * 32` columns
 * unconditionally.
 *
 * It is safe TODAY for a reason the kernel cannot check and therefore has to record: a
 * ragged W HAS NO WIRE REPRESENTATION. `ft_tt_op_reduce_request` carries `tile_cols`, never
 * W, and the only entry point that takes W —
 * `ft_tt_op_reduce_build_constants()` — refuses `width % 32 != 0` outright
 * (ft_tt_op_vector.c). So the reduction width is `Wt * 32` by construction on both sides.
 *
 * What is left is a caller that PADS: builds constants at a legal W = 1504 for a tensor
 * that is really 1500 wide. That reduces over four pad columns and neither end can see it,
 * because neither end is told the logical width. **DELIBERATELY NOT FIXED HERE.** A mask
 * tile is a new circular buffer and a change to the chain in `resident_reduce_compute.c`,
 * which is not this round's to edit; and the cheaper half — carrying the logical W on the
 * wire so the host can refuse a padded request — is a protocol change that buys nothing
 * until a shape needs it. Every dimension MiniCPM5-1B reduces over is a multiple of 32.
 *
 * THIS PROCESSOR OWNS `tail`, AND NOTHING ELSE IN THE CONTROL BLOCK.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_IN index   1: CB_SCALER index   2: CB_EPS index   3: CB_GAMMA index
 *   4: CB_BETA index 5: tile size in bytes
 *   6: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0,1,5,6,7,2048,1 NOC=0 KERNEL_TEXT_BASE=0x15000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_reduce_reader.c brisc /tmp/ttk_reduce
 *
 * @status Green on silicon -- see the report for this round.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_queue_protocol.h"
#include "resident_precondition.h"
#include "resident_reduce_protocol.h"

#define CB_IN ft_tensix_compile_time_arg(0)
#define CB_SCALER ft_tensix_compile_time_arg(1)
#define CB_EPS ft_tensix_compile_time_arg(2)
#define CB_GAMMA ft_tensix_compile_time_arg(3)
#define CB_BETA ft_tensix_compile_time_arg(4)
#define TILE_BYTES ft_tensix_compile_time_arg(5)
#define WAIT_MODE ft_tensix_compile_time_arg(6)
/**
 * Refuse a descriptor this program cannot execute, INSTEAD of writing outside the circular
 * buffers.
 *
 * FIVE circular buffers, and every one of them is a resident cycle. `cb_push_back` wraps
 * only on an EXACT hit of the limit, so a descriptor that leaves
 * any of the five write pointers part-way through its buffer sends the NEXT descriptor's
 * blocks over the limit and up into L1 — the failure that put a read on the queue control
 * block at 0x170000 with nothing faulting. `ft_tensix_cb_push_grid_is_safe()` carries it, and it is a test on the POINTER rather
 * than on the total — see that function for why the total-based form was wrong.
 *
 * The header above says double-buffering CB_IN is "a host-side page-count change and no
 * kernel change at all". IT IS NOT: at 2*Wt pages an odd `row_blocks` leaves the pointer
 * half way round, and that sentence is exactly the invitation this refusal has to survive.
 *
 * THE CONSTANT STREAMS ARE CHECKED AT THEIR OWN BLOCK SIZE — one page for the scaler and
 * for eps, `tile_cols` for gamma and for beta — because each is reserved and pushed in one
 * go and a buffer that is not a whole multiple of that would straddle on its first wrap.
 *
 * P3: A FINAL-CHAIN CHUNK PAST THE SCRATCH CBs' PAGES IS REFUSED (ftr_chain_chunk()). Only
 * this processor can check it — the compute threads' firmware records no page counts.
 *
 * AND A RAGGED `tile_count` IS REFUSED RATHER THAN TRUNCATED. `row_blocks` was
 * `tile_count / tile_cols`, so a tile count that is not a whole number of rows silently
 * dropped its remainder — the reader read fewer tiles than the descriptor named while the
 * writer still drained `tile_count` of them.
 */
static bool descriptor_is_executable(ftq_context *ctx, const ftr_program *program,
                                     uint32_t tile_cols, uint32_t tile_count, uint32_t depth,
                                     uint32_t slice_count, uint32_t schedule)
{
    const uint32_t stage0 = ft_tensix_cb((uint32_t)kRedCbStage0)->fifo_num_pages;
    const uint32_t stage1 = ft_tensix_cb((uint32_t)kRedCbStage1)->fifo_num_pages;

    if (tile_cols == 0u || tile_count == 0u || depth == 0u || TILE_BYTES == 0u)
        ctx->error = kErrBadParameters;
    else if (ftr_chain_chunk(schedule, stage0 < stage1 ? stage0 : stage1) == 0u)
        ctx->error = kErrBadParameters; /* P3: a chunk the scratch CBs cannot hold wedges PACK */
    else if ((tile_count % tile_cols) != 0u)
        ctx->error = kErrBadParameters;
    else if (!ft_tensix_cb_push_grid_is_safe(CB_IN, depth))
        ctx->error = kErrBadParameters;
    else if (!ft_tensix_cb_push_grid_is_safe(CB_SCALER, 1u))
        ctx->error = kErrBadParameters;
    else if (ftr_program_uses_eps(program) && !ft_tensix_cb_push_grid_is_safe(CB_EPS, 1u))
        ctx->error = kErrBadParameters;
    else if ((program->chain & kRedChainGamma) != 0u &&
             !ft_tensix_cb_push_grid_is_safe(CB_GAMMA, slice_count))
        ctx->error = kErrBadParameters;
    else if ((program->chain & kRedChainBeta) != 0u &&
             !ft_tensix_cb_push_grid_is_safe(CB_BETA, slice_count))
        ctx->error = kErrBadParameters;
    else
        return true;
    return false;
}

/**
 * Issue the reads of `tiles` consecutive DRAM tiles from `tile` on into L1 at `write`.
 *
 * A whole-tile read is ONE transaction: the tiles are consecutive in DRAM and the destination
 * is contiguous by the block-grid invariant (docs/tt_kernel_model.md 12). A half read cannot
 * be one — the faces it wants are the first `tile_read` bytes of every tile — so it is one
 * transaction per tile, into the front of each page. No barrier here: the caller owns it.
 */
static void issue_tile_reads(uint32_t x, uint32_t y, uint32_t base, uint32_t tile,
                             uint32_t tiles, uint32_t write, uint32_t tile_read)
{
    if (tile_read == TILE_BYTES) {
        ft_tensix_noc_read(ft_tensix_noc_address(x, y, base + tile * TILE_BYTES), write,
                           tiles * TILE_BYTES);
        return;
    }
    for (uint32_t t = 0; t < tiles; t++)
        ft_tensix_noc_read(ft_tensix_noc_address(x, y, base + (tile + t) * TILE_BYTES),
                           write + t * TILE_BYTES, tile_read);
}

/**
 * Fetch `tiles` consecutive tiles starting at constants tile `first` into `cb`.
 *
 * The reservation is the whole stream, so the destination is contiguous; `tile_read` is
 * TILE_BYTES except for a half-read gamma (ftr_tile_read_bytes()).
 *
 * @return false if the reservation was abandoned — the caller must not retire the descriptor
 */
static bool read_constant_stream(ftq_context *ctx, uint32_t cb, uint32_t base, uint32_t x,
                                 uint32_t y, uint32_t first, uint32_t tiles, uint32_t tile_read)
{
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, (int32_t)tiles, WAIT_MODE))
        return false;
    if (!ctx->aborting) {
        issue_tile_reads(x, y, base, first, tiles, ft_tensix_cb_write_pointer(cb), tile_read);
        ft_tensix_noc_read_barrier();
    }
    ft_tensix_cb_push_back(cb, tiles);
    return true;
}

/** Pages free in CB_IN NOW, counted from the write pointer — reserve_back's test, once. */
static uint32_t input_free_pages(void)
{
    const uint32_t received = ft_tensix_cb_pages_received_pointer((int32_t)CB_IN)[0];
    uint16_t acked;

    ft_bh_l1_cache_invalidate();
    acked = (uint16_t)ft_bh_reg_read((uintptr_t)ft_tensix_cb_pages_acked_pointer((int32_t)CB_IN));
    return (uint16_t)(ft_tensix_cb(CB_IN)->fifo_num_pages - ((uint16_t)received - acked));
}

/** L1 address `pages` past CB_IN's write pointer, wrapped. A block never straddles the wrap:
 *  the buffer is a whole number of blocks and the pointer sits on the block grid. */
static uint32_t input_address_ahead(uint32_t pages)
{
    uint32_t address = ft_tensix_cb_write_pointer(CB_IN) + pages * TILE_BYTES;

    if (address >= ft_tensix_cb(CB_IN)->fifo_limit)
        address -= ft_tensix_cb(CB_IN)->fifo_size;
    return address;
}

/** The NoC transaction id of streamed block `block`: 1..kRedStreamFlight, never 0 (untagged). */
static uint32_t stream_trid(uint32_t block)
{
    return 1u + block % (uint32_t)kRedStreamFlight;
}

/**
 * P3 STREAMED READ: `blocks` blocks of `depth` input tiles from tile `first` on, up to
 * kRedStreamFlight in flight, each pushed as soon as its own id drains. ftr_stream_action()
 * decides every step and carries the deadlock argument.
 *
 * Rule 3 holds per block: while draining nothing is issued but every block is still pushed.
 * An abandoned wait leaves blocks issued and unpushed, so the barrier below runs on EVERY
 * exit — a read still landing after the kernel returns would write into the next image.
 *
 * @return false if a wait was abandoned — the caller must not retire the descriptor
 */
static bool stream_input_blocks(ftq_context *ctx, uint32_t x, uint32_t y, uint32_t base,
                                uint32_t first, uint32_t blocks, uint32_t depth,
                                uint32_t tile_read, bool no_read)
{
    uint32_t issued = 0u;
    uint32_t retired = 0u;
    bool ok = true;

    while (ok) {
        const uint32_t action =
            ftr_stream_action(issued, retired, blocks, input_free_pages(), depth);

        if (action == (uint32_t)kRedStreamDone)
            break;
        if (action == (uint32_t)kRedStreamIssue) {
            if (!ctx->aborting && !no_read) {
                ft_tensix_noc_read_set_trid(stream_trid(issued));
                issue_tile_reads(x, y, base, first + issued * depth, depth,
                                 input_address_ahead((issued - retired) * depth), tile_read);
            }
            issued++;
        } else if (action == (uint32_t)kRedStreamRetire) {
            if (!ctx->aborting && !no_read)
                ft_tensix_noc_read_barrier_trid(stream_trid(retired));
            ft_tensix_cb_push_back(CB_IN, (int32_t)depth);
            retired++;
        } else {
            ok = ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN, (int32_t)depth, WAIT_MODE);
        }
    }
    ft_tensix_noc_read_barrier();
    ft_tensix_noc_read_set_trid(0u);
    return ok;
}

/** The pre-P3 input loop: `rows` row blocks, one block at a time, one barrier each. */
static bool read_input_blocks(ftq_context *ctx, uint32_t x, uint32_t y, uint32_t base,
                              uint32_t rows, uint32_t tile_cols, uint32_t depth,
                              uint32_t tile_read, bool no_read)
{
    for (uint32_t block = 0; block < rows; block++) {
        for (uint32_t wt = 0; wt < tile_cols; wt += depth) {
            uint32_t write_address;

            if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN, (int32_t)depth, WAIT_MODE))
                return false;
            write_address = ft_tensix_cb_write_pointer(CB_IN);

            // Rule 3: while draining, issue NOTHING. The push below still happens so the
            // page counters stay balanced, but a transaction issued during a drain would
            // still be in flight when the kernel returns.
            if (!ctx->aborting && !no_read) {
                issue_tile_reads(x, y, base, block * tile_cols + wt, depth, write_address,
                                 tile_read);
                // ONE barrier for the whole block: it is per-core, not per-transaction.
                ft_tensix_noc_read_barrier();
            }
            ft_tensix_cb_push_back(CB_IN, depth);
        }
    }
    return true;
}

/**
 * The reader half of one REDUCE descriptor.
 *
 * @return true if the whole descriptor was read; false if a wait was abandoned, in which
 *         case the loop still ran to completion (rule 3) and the caller must not retire it
 */
static bool read_one_reduce(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const ftr_program program = ftr_decode(parameters);
    const uint32_t schedule = parameters[kRedSchedule];
    const uint32_t tile_cols = parameters[kVecTileCols];
    const uint32_t tile_count = parameters[kVecTileCount];
    const uint32_t x_x = parameters[kVecAX];
    const uint32_t x_y = parameters[kVecAY];
    const uint32_t x_base = parameters[kVecABase];
    const uint32_t const_x = parameters[kVecBX];
    const uint32_t const_y = parameters[kVecBY];
    const uint32_t const_base = parameters[kVecBBase];
    const uint32_t depth = ftv_block_depth(tile_cols, ftv_depth_word(parameters[kVecDepth]));
    const bool probe_no_read = (ftv_probe_bits(parameters[kVecDepth]) & kVecProbeNoRead) != 0u;
    const bool stream = (schedule & (uint32_t)kRedSchedStream) != 0u;
    const uint32_t tile_read =
        ftr_tile_read_bytes(TILE_BYTES, ftr_half_read(schedule, parameters[kVecDepth], &program),
                            schedule);
    const uint32_t row_blocks = (tile_cols == 0u) ? 0u : tile_count / tile_cols;
    uint32_t slice_first, slice_count;

    ftr_slice(parameters[kRedFinal], tile_cols, &slice_first, &slice_count);
    if (!ftr_program_valid(&program) || tile_cols == 0u || row_blocks == 0u ||
        !descriptor_is_executable(ctx, &program, tile_cols, tile_count, depth, slice_count,
                                  schedule)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }

    if (!read_constant_stream(ctx, CB_SCALER, const_base, const_x, const_y, kRedConstScalerTile,
                              1u, TILE_BYTES))
        return false;
    if (ftr_program_uses_eps(&program) &&
        !read_constant_stream(ctx, CB_EPS, const_base, const_x, const_y, kRedConstEpsTile, 1u,
                              TILE_BYTES))
        return false;
    /* P3: streamed, row 0 goes BEFORE gamma — the map and the reduce need it first and the
     * chain needs gamma last, so gamma's read overlaps row 0's compute instead of delaying
     * it. The compute thread waits for gamma just before its first chain step. */
    if (stream && !stream_input_blocks(ctx, x_x, x_y, x_base, 0u, tile_cols / depth, depth,
                                       tile_read, probe_no_read))
        return false;
    if ((program.chain & kRedChainGamma) != 0u &&
        !read_constant_stream(ctx, CB_GAMMA, const_base, const_x, const_y,
                              kRedConstStreamTile + slice_first, slice_count, tile_read))
        return false;
    if ((program.chain & kRedChainBeta) != 0u &&
        !read_constant_stream(ctx, CB_BETA, const_base, const_x, const_y,
                              kRedConstStreamTile + tile_cols + slice_first, slice_count,
                              TILE_BYTES))
        return false;

    if (stream) {
        if (!stream_input_blocks(ctx, x_x, x_y, x_base, tile_cols,
                                 (row_blocks - 1u) * (tile_cols / depth), depth, tile_read,
                                 probe_no_read))
            return false;
    } else if (!read_input_blocks(ctx, x_x, x_y, x_base, row_blocks, tile_cols, depth,
                                  tile_read, probe_no_read)) {
        return false;
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

        if (opcode == kOpReduce) {
            ftq_status_set(&ctx, kStateCbReserveBack);
            done = read_one_reduce(&ctx, ftq_parameters(descriptor));
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
