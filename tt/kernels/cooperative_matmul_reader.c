/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file cooperative_matmul_reader.c
 * @brief The reader third of a matmul that runs on N cores at once, with the shared A block
 *   MULTICAST once instead of read from DRAM by every core.
 *
 *   This is the first Foundry kernel that uses more than one Tensix on one op. Everything
 *   measured in this backend so far is one core of 140, and the blocker has been named since
 *   Phase 4: "every A tile re-read Nt times, every B tile Mt times" — no reuse across cores.
 *   The primitive that fixes it is kernels/ft_tensix_noc_multicast.h, measured 2026-08-13 to
 *   turn the A-side re-read from O(N) into O(1): the DRAM arm grows +1.93 us per extra core
 *   while multicast grows +0.15 us, crossing over at N=3 for a depth-48 A block and reaching
 *   1.39x at N=4.
 *
 *   **IT IS matmul_reader.c's LOOP NEST, UNCHANGED.** The blocked outer product, the
 *   one-transaction A block, the N-major B block and the depth-divides-Kt contract are exactly
 *   as that file has them and its header documents them. The only edits are that the A read
 *   may come from a peer instead of DRAM, that B is indexed with a COLUMN OFFSET so a core can
 *   own a slice, and that a status block is published at the end. Keeping the arithmetic
 *   identical is deliberate: the solo/cooperative A/B has to measure COOPERATION, not two
 *   readers.
 *
 *   The whole wire contract — roles, argument indices, status words, refusal codes — and the
 *   reasoning for splitting over N rather than K or M are in cooperative_matmul_protocol.h,
 *   which the host includes verbatim.
 *
 *   NOTHING ABOUT THE COMPUTE OR WRITER THIRDS CHANGES. The compute kernel waits on circular
 *   buffers regardless of who filled them, so it takes its own slice width as runtime arg 2
 *   and is otherwise the shipped matmul_compute.c; the writer streams `Mt * Nt_local` tiles
 *   to this core's own output panel and is the shipped vector_add_writer.c. A cooperating
 *   matmul is a reader-side and host-side change, and this file is the evidence for that.
 *
 *   **The SOLO role is also how the two CONTROLS are expressed**, which is why it takes a
 *   column offset it does not otherwise need: N cores in SOLO role over disjoint column
 *   slices is "more cores, no sharing", and N cores in SOLO role over the WHOLE width is "N
 *   independent matmuls", which is the concurrent-stream and DRAM-saturation question. All
 *   four arms are one binary and one runtime word apart.
 *
 * C99 on Foundry's own ft_tensix_dataflow.h — no tt-metal include (plan Phase 6.6).
 *
 * Compile-time args (identical to matmul_reader.c, so the two are interchangeable):
 *   0: cb_in0 index   1: cb_in1 index
 *   2: in0 tile size in bytes   3: in1 tile size in bytes
 * Runtime args: the 28 of cooperative_matmul_protocol.h.
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0,1,2048,2048 NOC=0 KERNEL_TEXT_BASE=0x15000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/cooperative_matmul_reader.c brisc /tmp/ttk_coop
 *
 * ---------------------------------------------------------------------------
 * WHAT IT MEASURED, AND THE CONDITION THE MULTICAST NEEDS
 * ---------------------------------------------------------------------------
 * MEASURED on the p150a card host 2026-08-13, MiniCPM5-1B shapes at 32 tokens, byte-identical to the
 * single-core result at every N. **The multicast pays only when the SHARED operand's DRAM bank
 * is the bottleneck, and with A and B in separate banks it never is.**
 *
 * The reader's traffic is Mt*Nt*Kt tiles of A and Mt*Nt*Kt of B — always EQUAL. Multicast
 * divides the A side by N and leaves B alone, so with one bank per operand the binding
 * constraint goes from max(X, X) to max(X/N, X): unchanged. Put both operands in ONE bank and
 * it goes from 2X to X(1 + 1/N), and the multicast wins:
 *
 *   ffn_gate/up, 4 cores      A and B in TWO banks      A and B in ONE bank
 *     no sharing (SPLIT)          0.386 ms                  0.677 ms
 *     multicast   (COOP)          0.559 ms  (0.69x)         0.638 ms  (1.06x, WINS)
 *
 * The crossover is at N=3 in the one-bank case — exactly where tests/test_multicast_bench.c
 * put it for a 98304-byte block, reproduced here inside a real op. The rendezvous costs ~6 us
 * a block at N=4, which is also that file's number.
 *
 * So this kernel is not the win on today's single-bank-per-operand layout; it is the
 * PREREQUISITE for the layout that scales. Once the weight is striped across DRAM banks — the
 * only way past the measured ~42 GB/s per-bank read wall — A becomes the contended operand and
 * this is what removes it.
 *
 * @status MEASURED on the card 2026-08-13 by tests/test_cooperative_matmul.c: byte-identical
 *   to the single-core arm on ffn_gate/up, ffn_down and attn_k/v at N = 2, 3, 4 and 8.
 * @issues A 2x4 group (8 receivers, 4 columns wide) left ONE of its cores unbootable with the
 *   recoverable `soft reset 0x00047000` signature — the same family as the 5-wide hazard in
 *   tests/test_multicast_bench.c, and a new data point that implicates RECEIVER COUNT rather
 *   than width. Groups of 4 and below ran clean throughout.
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "cooperative_matmul_protocol.h"
#include "ft_tensix_dataflow.h"
#include "ft_tensix_noc_multicast.h"

#define CB_IN0 ft_tensix_compile_time_arg(0)
#define CB_IN1 ft_tensix_compile_time_arg(1)
#define IN0_TILE_BYTES ft_tensix_compile_time_arg(2)
#define IN1_TILE_BYTES ft_tensix_compile_time_arg(3)

/** DRAM's read alignment, the stricter of the two ends of a DRAM->L1 transfer. A destination
 *  that is only 16- or 32-byte aligned reads back byte-exact until its LAST 64 bytes,
 *  silently (tests/ft_tt_phase2_l1_layout.h). */
#define FTCM_DRAM_READ_ALIGNMENT 64u

/** Everything the roles share, read once at entry so the block loop indexes a struct instead
 *  of re-reading L1 through the runtime-arg accessor on every use. */
typedef struct ftcm_context {
    uint32_t role;
    uint32_t tile_rows;
    uint32_t tile_inner;
    uint32_t tile_cols;
    uint32_t a_x;
    uint32_t a_y;
    uint32_t a_base;
    uint32_t b_x;
    uint32_t b_y;
    uint32_t b_base;
    uint32_t b_column_first;
    uint32_t b_full_cols;
    uint32_t depth;
    uint32_t b_n_major;
    uint32_t cb_in0_base;
    uint32_t rectangle_x_low;
    uint32_t rectangle_y_low;
    uint32_t rectangle_x_high;
    uint32_t rectangle_y_high;
    uint32_t destination_count;
    uint32_t leader_x;
    uint32_t leader_y;
    uint32_t ready_semaphore_address;
    uint32_t arrival_semaphore_address;
    uint32_t valid_word_address;
    volatile uint32_t *status;
    volatile const uint32_t *stop_flag;
    uint32_t spin_limit;

    /* Run state. `error` keeps the FIRST failure; `aborting` is what stops every issue while
     * the loop runs on to drain the circular buffers. */
    int error;
    int last_issue_code;
    bool aborting;
    uint32_t observed_write_pointer;
} ftcm_context;

static void ftcm_read_arguments(ftcm_context *context)
{
    context->role = ft_tensix_runtime_arg((int)FTCM_ARG_ROLE);
    context->tile_rows = ft_tensix_runtime_arg((int)FTCM_ARG_TILE_ROWS);
    context->tile_inner = ft_tensix_runtime_arg((int)FTCM_ARG_TILE_INNER);
    context->tile_cols = ft_tensix_runtime_arg((int)FTCM_ARG_TILE_COLS);
    context->a_x = ft_tensix_runtime_arg((int)FTCM_ARG_A_X);
    context->a_y = ft_tensix_runtime_arg((int)FTCM_ARG_A_Y);
    context->a_base = ft_tensix_runtime_arg((int)FTCM_ARG_A_BASE);
    context->b_x = ft_tensix_runtime_arg((int)FTCM_ARG_B_X);
    context->b_y = ft_tensix_runtime_arg((int)FTCM_ARG_B_Y);
    context->b_base = ft_tensix_runtime_arg((int)FTCM_ARG_B_BASE);
    context->b_column_first = ft_tensix_runtime_arg((int)FTCM_ARG_B_COLUMN_FIRST);
    context->b_full_cols = ft_tensix_runtime_arg((int)FTCM_ARG_B_FULL_COLS);
    context->depth = ft_tensix_runtime_arg((int)FTCM_ARG_DEPTH);
    context->b_n_major = ft_tensix_runtime_arg((int)FTCM_ARG_B_N_MAJOR);
    context->cb_in0_base = ft_tensix_runtime_arg((int)FTCM_ARG_CB_IN0_BASE);
    context->rectangle_x_low = ft_tensix_runtime_arg((int)FTCM_ARG_RECTANGLE_X_LOW);
    context->rectangle_y_low = ft_tensix_runtime_arg((int)FTCM_ARG_RECTANGLE_Y_LOW);
    context->rectangle_x_high = ft_tensix_runtime_arg((int)FTCM_ARG_RECTANGLE_X_HIGH);
    context->rectangle_y_high = ft_tensix_runtime_arg((int)FTCM_ARG_RECTANGLE_Y_HIGH);
    context->destination_count = ft_tensix_runtime_arg((int)FTCM_ARG_DESTINATION_COUNT);
    context->leader_x = ft_tensix_runtime_arg((int)FTCM_ARG_LEADER_X);
    context->leader_y = ft_tensix_runtime_arg((int)FTCM_ARG_LEADER_Y);
    context->ready_semaphore_address = ft_tensix_runtime_arg((int)FTCM_ARG_READY_SEMAPHORE);
    context->arrival_semaphore_address = ft_tensix_runtime_arg((int)FTCM_ARG_ARRIVAL_SEMAPHORE);
    context->valid_word_address = ft_tensix_runtime_arg((int)FTCM_ARG_VALID_WORD);
    context->status =
        (volatile uint32_t *)(unsigned long)ft_tensix_runtime_arg((int)FTCM_ARG_STATUS);
    context->stop_flag =
        (volatile const uint32_t *)(unsigned long)ft_tensix_runtime_arg((int)FTCM_ARG_STOP_FLAG);
    context->spin_limit = ft_tensix_runtime_arg((int)FTCM_ARG_SPIN_LIMIT);

    context->error = FTCM_OK;
    context->last_issue_code = FTCM_OK;
    context->aborting = false;
    context->observed_write_pointer = 0u;
}

/** Record the FIRST failure and switch the loop into its drain. It never returns early — see
 *  the protocol header's drain rule and DEV_NOTES.md §1 (core 62, where skipping the loop
 *  body during an abort stalled the matrix engine in a way no flag could reach). */
static void ftcm_abort(ftcm_context *context, int error)
{
    if (context->error == FTCM_OK)
        context->error = error;
    context->aborting = true;
}

static void ftcm_publish(const ftcm_context *context, uint32_t blocks_completed,
                         uint32_t blocks_expected, uint32_t clock_at_entry,
                         uint32_t clock_after_first_block, uint32_t clock_at_exit)
{
    /* Magic LAST: a host reading a torn block must see "never ran", not a half-filled result
     * (kernels/resident_queue_protocol.h rule 1). */
    context->status[FTCM_STATUS_ROLE] = context->role;
    context->status[FTCM_STATUS_BLOCKS_COMPLETED] = blocks_completed;
    context->status[FTCM_STATUS_BLOCKS_EXPECTED] = blocks_expected;
    context->status[FTCM_STATUS_ERROR] = (uint32_t)context->error;
    context->status[FTCM_STATUS_LAST_ISSUE_CODE] = (uint32_t)context->last_issue_code;
    context->status[FTCM_STATUS_CLOCK_AT_ENTRY] = clock_at_entry;
    context->status[FTCM_STATUS_CLOCK_AFTER_FIRST_BLOCK] = clock_after_first_block;
    context->status[FTCM_STATUS_CLOCK_AT_EXIT] = clock_at_exit;
    context->status[FTCM_STATUS_OBSERVED_WRITE_POINTER] = context->observed_write_pointer;
    context->status[FTCM_STATUS_MAGIC_WORD] = FTCM_STATUS_MAGIC;
}

/**
 * Refuse a shape this kernel cannot execute, BEFORE anything is issued.
 *
 * The depth rule is matmul_reader.c's contract restated as a REFUSAL rather than a silent
 * collapse to depth 1. On the cooperative path a silent collapse would be worse than a wrong
 * answer: it changes this core's block COUNT, and the rendezvous is only correct because every
 * member walks the same number of blocks. The host makes depth a divisor; a value that is not
 * one is a host bug and is refused.
 */
static bool ftcm_shape_is_executable(ftcm_context *context)
{
    if (context->tile_rows == 0u || context->tile_inner == 0u || context->tile_cols == 0u ||
        context->depth == 0u || IN0_TILE_BYTES == 0u || IN1_TILE_BYTES == 0u) {
        ftcm_abort(context, FTCM_ERROR_BAD_PARAMETERS);
        return false;
    }
    if ((context->tile_inner % context->depth) != 0u) {
        ftcm_abort(context, FTCM_ERROR_BAD_PARAMETERS);
        return false;
    }
    if (context->b_full_cols == 0u ||
        context->b_column_first + context->tile_cols > context->b_full_cols) {
        ftcm_abort(context, FTCM_ERROR_BAD_PARAMETERS);
        return false;
    }
    if (context->role != FTCM_ROLE_SOLO && context->role != FTCM_ROLE_LEADER &&
        context->role != FTCM_ROLE_RECEIVER) {
        ftcm_abort(context, FTCM_ERROR_BAD_ROLE);
        return false;
    }
    /* A block of `depth` A tiles is ONE DRAM read, so its length and its L1 destination must
     * satisfy DRAM's 64 bytes and not L1's 16. Checked HERE, in the kernel, because
     * ft_tt_noc_check_transfer() runs host-side and cannot see a kernel-issued transfer —
     * which is exactly how `noc_inline_dw_write` to DRAM cost cores 50 and 52 with a suite
     * reporting 0 failures (DEV_NOTES.md §1). */
    if (((context->depth * IN0_TILE_BYTES) % FTCM_DRAM_READ_ALIGNMENT) != 0u ||
        (context->cb_in0_base % FTCM_DRAM_READ_ALIGNMENT) != 0u) {
        ftcm_abort(context, FTCM_ERROR_ALIGNMENT);
        return false;
    }
    if (context->role != FTCM_ROLE_SOLO && context->destination_count == 0u) {
        /* The zero-length NoC transfer in multicast clothing. Refuse it; never "optimise" it
         * into a solo run, which would turn a mis-planned group into a wrong answer that looks
         * like a right one. */
        ftcm_abort(context, FTCM_ERROR_ZERO_DESTINATIONS);
        return false;
    }
    return true;
}

/**
 * Where block `block_index`'s A tiles MUST land, from the CB wrap rule alone.
 *
 * A 2*depth-page buffer pushed `depth` at a time puts the write pointer at the base or one
 * block up and nowhere else — the pointer resets by subtraction on an EXACT hit of the limit
 * and no other wrap is legal (ft_tensix_circular_buffer.h). So the address is a function of
 * the block index, identical on every member, which is what makes ONE multicast destination
 * correct for all of them. This is checked every block rather than assumed: a multicast writes
 * whatever address it is given into every node of the rectangle, and no hardware would refuse
 * the wrong one.
 */
static uint32_t ftcm_expected_write_pointer(const ftcm_context *context, uint32_t block_index)
{
    return context->cb_in0_base + (block_index & 1u) * context->depth * IN0_TILE_BYTES;
}

/** The shared A block: `depth` consecutive tiles of row `mt`, which are consecutive bytes in
 *  DRAM, so ONE transaction (matmul_reader.c's measured 27-37%). */
static void ftcm_read_a_block(const ftcm_context *context, uint32_t mt, uint32_t kt,
                              uint32_t a_write)
{
    const uint32_t a_offset = (mt * context->tile_inner + kt) * IN0_TILE_BYTES;

    ft_tensix_noc_read(ft_tensix_noc_address(context->a_x, context->a_y,
                                             context->a_base + a_offset),
                       a_write, context->depth * IN0_TILE_BYTES);
}

/** This core's own B block, which no other core reads. One transaction when the weight is
 *  stored N-major, `depth` of them when it is K-major — matmul_reader.c's two cases, with the
 *  column index shifted into the FULL grid so a core can own a slice of it. */
static void ftcm_read_b_block(const ftcm_context *context, uint32_t nt, uint32_t kt,
                              uint32_t b_write)
{
    const uint32_t column = context->b_column_first + nt;

    if (context->b_n_major) {
        const uint32_t b_offset = (column * context->tile_inner + kt) * IN1_TILE_BYTES;

        ft_tensix_noc_read(ft_tensix_noc_address(context->b_x, context->b_y,
                                                 context->b_base + b_offset),
                           b_write, context->depth * IN1_TILE_BYTES);
    } else {
        for (uint32_t step = 0u; step < context->depth; step++) {
            const uint32_t b_tile = (kt + step) * context->b_full_cols + column;

            ft_tensix_noc_read(ft_tensix_noc_address(context->b_x, context->b_y,
                                                     context->b_base +
                                                         b_tile * IN1_TILE_BYTES),
                               b_write + step * IN1_TILE_BYTES, IN1_TILE_BYTES);
        }
    }
}

/** SOLO: matmul_reader.c's block body, with B's column offset. No semaphores, no rectangle,
 *  no multicast — this is the arm every cooperative number is measured against. */
static void ftcm_solo_block(ftcm_context *context, uint32_t mt, uint32_t nt, uint32_t kt,
                            uint32_t a_write, uint32_t b_write)
{
    ftcm_read_a_block(context, mt, kt, a_write);
    ftcm_read_b_block(context, nt, kt, b_write);
    /* ONE barrier for the whole block: it is per-core, not per-transaction, which is exactly
     * why depth converts (matmul_reader.c). */
    ft_tensix_noc_read_barrier();
}

/**
 * LEADER: collect every receiver, fetch A and its own B, then hand A to the group.
 *
 * The order of the last four steps is the protocol and none of it is rearrangeable —
 * kernels/ft_tensix_noc_multicast.h's header block has the full account, including why the
 * flush between the two multicasts is mandatory on Blackhole and nowhere else.
 *
 * The payload's SOURCE is this core's own CB_IN0 block, so there is no staging buffer and none
 * is needed. Reusing that block next round is safe for exactly one reason: the writes-flushed
 * barrier below guarantees the payload's request has LEFT, which is what frees the source. A
 * full write barrier would additionally wait for acknowledgements this core has no use for
 * per block, and is taken once after the loop instead.
 */
static void ftcm_leader_block(ftcm_context *context, uint32_t mt, uint32_t nt, uint32_t kt,
                              uint32_t a_write, uint32_t b_write)
{
    volatile uint32_t *ready_semaphore =
        (volatile uint32_t *)(unsigned long)context->ready_semaphore_address;
    uint64_t payload_destination;
    uint64_t arrival_destination;
    int code;

    code = ft_tensix_wait_semaphore_at_least(ready_semaphore, context->destination_count,
                                             context->spin_limit, context->stop_flag);
    if (code != FTCM_OK) {
        context->last_issue_code = code;
        ftcm_abort(context, code);
        return;
    }
    /* Safe here and only here: a receiver cannot increment again until it has been told this
     * round landed, which is the last thing this function does. */
    ft_tensix_set_local_semaphore(ready_semaphore, 0u);

    ftcm_read_a_block(context, mt, kt, a_write);
    ftcm_read_b_block(context, nt, kt, b_write);
    ft_tensix_noc_read_barrier();

    /* The destination L1 address is this core's own write pointer, which the caller has
     * already proven equals every member's. */
    code = ft_tensix_multicast_address(context->rectangle_x_low, context->rectangle_y_low,
                                       context->rectangle_x_high, context->rectangle_y_high,
                                       a_write, &payload_destination);
    if (code == FTCM_OK)
        code = ft_tensix_noc_write_multicast(a_write, payload_destination,
                                             context->depth * IN0_TILE_BYTES,
                                             context->destination_count);
    context->last_issue_code = code;
    if (code != FTCM_OK) {
        ftcm_abort(context, code);
        return;
    }

    /* BLACKHOLE ONLY, and load-bearing: the payload multicast and the semaphore multicast go
     * into different command-buffer FIFOs and are not guaranteed to leave in issue order, so
     * without this a receiver can be told the block arrived before it did and read the
     * PREVIOUS block's bytes with nothing reporting an error. */
    ft_tensix_noc_writes_flushed_barrier();

    code = ft_tensix_multicast_address(context->rectangle_x_low, context->rectangle_y_low,
                                       context->rectangle_x_high, context->rectangle_y_high,
                                       context->arrival_semaphore_address, &arrival_destination);
    if (code == FTCM_OK)
        code = ft_tensix_noc_semaphore_set_multicast(context->valid_word_address,
                                                     arrival_destination,
                                                     context->destination_count);
    context->last_issue_code = code;
    if (code != FTCM_OK)
        ftcm_abort(context, code);
}

/**
 * RECEIVER: clear the arrival flag, start its own B read, announce readiness, wait for A.
 *
 * The local clear comes FIRST. The leader cannot set the flag until it has seen this core's
 * atomic, and this core issues that atomic after the clear — so a VALID can never be erased by
 * the clear that was meant to precede it.
 *
 * The B read is issued BEFORE the announcement so it overlaps the whole rendezvous: this
 * core's own DRAM traffic and the leader's multicast are then in flight together, which is the
 * only reason a receiver's block costs anything less than the sum of the two.
 *
 * The read barrier is taken whether or not the wait succeeded, because the B read was already
 * issued and a transaction left outstanding is how L1 the next kernel is about to use gets
 * written underneath it.
 */
static void ftcm_receiver_block(ftcm_context *context, uint32_t nt, uint32_t kt,
                                uint32_t b_write)
{
    volatile uint32_t *arrival_semaphore =
        (volatile uint32_t *)(unsigned long)context->arrival_semaphore_address;
    int code;

    ft_tensix_set_local_semaphore(arrival_semaphore, FT_TENSIX_SEMAPHORE_INVALID);
    ftcm_read_b_block(context, nt, kt, b_write);
    ft_tensix_noc_semaphore_increment(
        ft_tensix_noc_address(context->leader_x, context->leader_y,
                              context->ready_semaphore_address),
        1u);

    code = ft_tensix_wait_semaphore_equals(arrival_semaphore, FT_TENSIX_SEMAPHORE_VALID,
                                           context->spin_limit, context->stop_flag);
    ft_tensix_noc_read_barrier();
    if (code != FTCM_OK) {
        context->last_issue_code = code;
        ftcm_abort(context, code);
    }
}

void kernel_main(void)
{
    ftcm_context context;
    uint32_t block_index = 0u;
    uint32_t blocks_completed = 0u;
    uint32_t blocks_expected = 0u;
    uint32_t clock_at_entry;
    uint32_t clock_after_first_block = 0u;
    uint32_t clock_at_exit;

    ftcm_read_arguments(&context);
    clock_at_entry = ft_bh_wall_clock_low();

    if (ftcm_shape_is_executable(&context))
        blocks_expected = context.tile_rows * context.tile_cols *
                          (context.tile_inner / context.depth);

    for (uint32_t mt = 0u; mt < context.tile_rows && blocks_expected != 0u; mt++) {
        for (uint32_t nt = 0u; nt < context.tile_cols; nt++) {
            for (uint32_t kt = 0u; kt < context.tile_inner; kt += context.depth) {
                uint32_t a_write;
                uint32_t b_write;

                ft_tensix_cb_reserve_back((int32_t)CB_IN0, (int32_t)context.depth);
                ft_tensix_cb_reserve_back((int32_t)CB_IN1, (int32_t)context.depth);
                a_write = ft_tensix_cb_write_pointer(CB_IN0);
                b_write = ft_tensix_cb_write_pointer(CB_IN1);

                if (!context.aborting &&
                    a_write != ftcm_expected_write_pointer(&context, block_index)) {
                    context.observed_write_pointer = a_write;
                    ftcm_abort(&context, FTCM_ERROR_WRITE_POINTER);
                }

                /* Rule 3: while draining, issue NOTHING. The pushes below still happen so the
                 * page counters stay balanced and the compute and writer thirds run to
                 * completion — a reader that returned here would leave them in cb_wait_front,
                 * which is the shape that wedges a core (DEV_NOTES.md §1). */
                if (!context.aborting) {
                    if (context.role == FTCM_ROLE_SOLO)
                        ftcm_solo_block(&context, mt, nt, kt, a_write, b_write);
                    else if (context.role == FTCM_ROLE_LEADER)
                        ftcm_leader_block(&context, mt, nt, kt, a_write, b_write);
                    else
                        ftcm_receiver_block(&context, nt, kt, b_write);
                    if (!context.aborting)
                        blocks_completed++;
                }

                ft_tensix_cb_push_back(CB_IN0, (int32_t)context.depth);
                ft_tensix_cb_push_back(CB_IN1, (int32_t)context.depth);
                block_index++;
                if (block_index == 1u)
                    clock_after_first_block = ft_bh_wall_clock_low();
            }
        }
    }

    clock_at_exit = ft_bh_wall_clock_low();

    /* Outside the timed window and unconditional. An abandoned block may have left a read, a
     * write or an atomic in flight, and brisck.cc asserts the NoC counters are settled after
     * kernel_main() returns. */
    ft_tensix_noc_read_barrier();
    ft_tensix_noc_write_barrier();
    ft_tensix_noc_atomic_barrier();

    ftcm_publish(&context, blocks_completed, blocks_expected, clock_at_entry,
                 clock_after_first_block, clock_at_exit);
}
