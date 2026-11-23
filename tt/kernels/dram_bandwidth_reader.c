/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * dram_bandwidth_reader.c
 *   ONE BRISC kernel, C99, whose only job is to pull bytes out of one DRAM NoC port as
 *   fast as it can and say — on its own wall clock — how long that took.
 *
 *   It exists to settle a contradiction between two of this project's own measurements:
 *   the cooperative-matmul round put one DRAM bank at ~42 GB/s, while an earlier DRAM
 *   round recorded >= 537 GB/s across 14 cores (~67 GB/s a bank) and did not find the
 *   ceiling. The suspected difference is ACCESS PATTERN, and the variables are confounded
 *   in both. So every one of them is a RUNTIME ARGUMENT here and nothing else changes:
 *
 *     sharing        the host gives N cores the same base address, or disjoint ones
 *     working set    `wrap_blocks * block_stride`, from a few KiB to hundreds of MiB
 *     port and bank  `dram_x`/`dram_y`, so a bank's THREE NoC ports are separable from
 *                    the bank behind them (ft_tt_coord.h: three doors, one memory)
 *     request size   `block_bytes`
 *     concurrency    `blocks_per_pass` / `barrier_every`, i.e. how many reads are in
 *                    flight before anything is waited on
 *
 *   ONE BINARY FOR EVERY ARM IS THE METHOD, not a convenience — the same rule
 *   multicast_bench_reader.c follows. Two binaries would put every codegen difference
 *   between the arms inside the number.
 *
 * ===========================================================================
 * WHAT IT MEASURES, AND WHY IT IS TIMED THIS WAY
 * ===========================================================================
 * A PASS is `blocks_per_pass` reads of `block_bytes` from a walking address, with a read
 * barrier every `barrier_every` reads. The kernel runs ONE UNTIMED PASS and then
 * `iterations` timed ones, reading its own RISCV_DEBUG_REG_WALL_CLOCK_0 between.
 *
 * Timing is device-side because the host is two 1.8 GHz cores that also drive eight K80s
 * and drift enough over ten seconds to swamp the quantity (DEV_NOTES.md §8), and because a
 * launch is ~14.5 us. The untimed pass absorbs the instruction cache, the first row-buffer
 * miss and the launch skew of the other cores in the group.
 *
 * `barrier_every` IS THE HARNESS'S OWN FALSIFIER. At `barrier_every == 1` every read is
 * waited on before the next is issued, so the result is round-trip LATENCY-bound and must
 * come out far below the pipelined arms. A harness that reports a plausible bandwidth for
 * that arm is mis-instrumented, and the run that proves it is one runtime word away from
 * the run being believed.
 *
 * ===========================================================================
 * THE WALK, AND WHY IT IS A COMPARE RATHER THAN A MODULO
 * ===========================================================================
 * `address += block_stride; if (address >= wrap_end) address = base;` — sequential inside
 * a bounded region, which is what streaming a weight matrix looks like. A modulo would put
 * a RISC-V division in a loop body whose whole budget is a few tens of cycles; this is two
 * arithmetic ops and a predictable branch. It also makes the final state computable on the
 * host in closed form, which is what lets the READ-BACK CHECK below exist.
 *
 * ===========================================================================
 * HOW A GREEN RUN IS PROVEN TO HAVE MOVED THE BYTES
 * ===========================================================================
 * The destination is `l1_slot_count` slots, used round-robin. After the last pass, slot
 * `(blocks_per_pass * (iterations + 1) - k) % l1_slot_count` holds the block the walk was
 * pointing at k reads from the end — an address the host computes with the same closed
 * form and stages with distinct content beforehand. The host poisons every slot first, so
 * "the reads never happened" and "the reads happened" are different bit patterns rather
 * than the same zero (DEV_NOTES.md §3).
 *
 * `final_walk_offset` is published so the host's arithmetic is CHECKED against the
 * kernel's rather than assumed equal to it.
 *
 * ===========================================================================
 * RUNTIME ARGUMENTS
 * ===========================================================================
 *   0  iterations         timed passes, >= 1 (ignored when duration_cycles is non-zero)
 *   1  blocks_per_pass    reads per pass, >= 1
 *   2  barrier_every      reads between read barriers, >= 1
 *   3  block_bytes        multiple of 64, > 0
 *   4  dram_x             TRANSLATED NoC coordinate of the DRAM PORT
 *   5  dram_y
 *   6  dram_base          byte address in that bank, 64-aligned
 *   7  block_stride       bytes between consecutive blocks, 64-aligned, >= block_bytes
 *   8  wrap_blocks        blocks before the walk returns to dram_base, >= 1
 *   9  l1_destination     64-aligned local L1 address of slot 0
 *  10  l1_slot_count      >= 1
 *  11  l1_slot_stride     >= block_bytes, 64-aligned
 *  12  status_address     local L1, FTDB_STATUS_WORDS words
 *  13  duration_cycles    0: run `iterations` passes. Non-zero: run passes until this many
 *                         wall-clock cycles have elapsed, and publish how many blocks that
 *                         took. **THIS IS THE MODE THE SWEEPS USE, and the reason is the
 *                         TAIL.** With a fixed byte count and N cores at different rates —
 *                         and they DO differ, by ~20% across the grid — the fast cores
 *                         finish early and the last stretch of the run has fewer readers in
 *                         it. The sum of per-core rates then overstates the card (it assumes
 *                         full overlap) while bytes-over-the-union understates it (it counts
 *                         the tail). A common wall-clock window collapses the two into one
 *                         number. Iteration mode stays because the CALIBRATION needs a
 *                         device interval it did not choose, and the byte-proof needs a
 *                         deterministic final walk position
 *  14  direction          0: DRAM -> L1 reads. 1: L1 -> DRAM writes. The walk, the wrap and
 *                         the L1 slot round-robin are identical; only the transfer and its
 *                         barrier change. **The barrier is NOT interchangeable** — a read
 *                         barrier deadlocks on Blackhole while writes are outstanding
 *                         (ft_tensix_noc.h), so the direction picks the barrier at every
 *                         place one is taken, not only inside the loop
 *
 * Compile-time args: none. Everything about the CALL is a runtime arg.
 *
 *   Build (BRISC, NoC 0):
 *     CTARGS=0 NOC=0 bash foundry/tt/build_tensix_dataflow_kernel.sh \
 *         foundry/tt/kernels/dram_bandwidth_reader.c brisc /tmp/ttk_dramcap
 *
 * @status NEW 2026-08-14. Driven by tests/test_dram_capacity.c. Runtime arg 14 added the
 *   same day so the same binary measures WRITES — everything known about this card's DRAM
 *   was read bandwidth, and Phase 7 writes KV cache every token.
 * @issues None
 * @todo None
 */

#include "ft_tensix_dataflow.h"

/* ---------------------------------------------------------------------------
 * The wire contract. tests/test_dram_capacity.c carries the same values; change
 * both or neither.
 * --------------------------------------------------------------------------- */

/** "FTDB". A status block still reading the host's poison means the kernel never ran. */
#define FTDB_STATUS_MAGIC 0x46544442u

#define FTDB_STATUS_MAGIC_WORD 0u
#define FTDB_STATUS_PASSES_COMPLETED 1u /**< the untimed pass plus every timed one */
#define FTDB_STATUS_BLOCKS_ISSUED 2u    /**< TIMED blocks only — the numerator of the rate */
#define FTDB_STATUS_ELAPSED_CYCLES 3u
#define FTDB_STATUS_ERROR 4u
#define FTDB_STATUS_START_CLOCK 5u
#define FTDB_STATUS_END_CLOCK 6u
#define FTDB_STATUS_FINAL_WALK_OFFSET 7u
#define FTDB_STATUS_WORDS 8u

/** Refusals. Every one of them happens BEFORE any transaction is issued, so a refused run
 *  leaves nothing in flight and the core is as healthy as it was at entry. */
#define FTDB_OK 0
#define FTDB_ERROR_ZERO_LENGTH (-1)
#define FTDB_ERROR_ALIGNMENT (-2)
#define FTDB_ERROR_SHAPE (-3)
#define FTDB_ERROR_L1_BOUNDS (-4)

/** DRAM's read alignment — the stricter of the two ends of a DRAM->L1 transfer
 *  (ft_tt_noc.h, ft_tt_noc_alignment). A 16- or 32-byte-aligned address reads back
 *  byte-exact until its LAST 64 bytes, silently. */
#define FTDB_DRAM_READ_ALIGNMENT 64u

/* --------------------------------------------------------------------------- */

/** Everything read once at entry, so the timed loop indexes registers rather than
 *  re-reading the runtime-argument block out of L1. */
typedef struct ftdb_context {
    uint32_t iterations;
    uint32_t blocks_per_pass;
    uint32_t barrier_every;
    uint32_t block_bytes;
    uint32_t dram_x;
    uint32_t dram_y;
    uint32_t dram_base;
    uint32_t block_stride;
    uint32_t wrap_blocks;
    uint32_t l1_destination;
    uint32_t l1_slot_count;
    uint32_t l1_slot_stride;
    volatile uint32_t *status;
    uint32_t duration_cycles;
    uint32_t direction;
} ftdb_context;

#define FTDB_DIRECTION_READ 0u
#define FTDB_DIRECTION_WRITE 1u
/** M4: reads, PIPELINED by transaction id — each group of `barrier_every` reads carries trid
 *  1 or 2 alternately, and a group's barrier waits for the PREVIOUS group only, so two groups
 *  are in flight (a double-buffered reader). */
#define FTDB_DIRECTION_READ_PIPELINED 2u

static uint32_t ftdb_trid_current = 1u;
static uint32_t ftdb_trid_previous = 0u;

static void ftdb_read_arguments(ftdb_context *context)
{
    context->iterations = ft_tensix_runtime_arg((int)0u);
    context->blocks_per_pass = ft_tensix_runtime_arg((int)1u);
    context->barrier_every = ft_tensix_runtime_arg((int)2u);
    context->block_bytes = ft_tensix_runtime_arg((int)3u);
    context->dram_x = ft_tensix_runtime_arg((int)4u);
    context->dram_y = ft_tensix_runtime_arg((int)5u);
    context->dram_base = ft_tensix_runtime_arg((int)6u);
    context->block_stride = ft_tensix_runtime_arg((int)7u);
    context->wrap_blocks = ft_tensix_runtime_arg((int)8u);
    context->l1_destination = ft_tensix_runtime_arg((int)9u);
    context->l1_slot_count = ft_tensix_runtime_arg((int)10u);
    context->l1_slot_stride = ft_tensix_runtime_arg((int)11u);
    context->status = (volatile uint32_t *)(unsigned long)ft_tensix_runtime_arg((int)12u);
    context->duration_cycles = ft_tensix_runtime_arg((int)13u);
    context->direction = ft_tensix_runtime_arg((int)14u);
}

/** The barrier that matches the direction. Taking the wrong one is not a slow path, it is a
 *  deadlock: a read barrier on Blackhole spins forever while a write is outstanding. */
static void ftdb_barrier(const ftdb_context *context)
{
    if (context->direction == FTDB_DIRECTION_WRITE)
        ft_tensix_noc_write_barrier();
    else
        ft_tensix_noc_read_barrier();
}

static void ftdb_publish(const ftdb_context *context, uint32_t passes_completed,
                         uint32_t blocks_issued, uint32_t start_clock, uint32_t end_clock,
                         uint32_t final_walk_offset, int error)
{
    /* Magic LAST, so a host reading a torn block sees "never ran" rather than a half-filled
     * result — the token-before-count ordering rule (resident_queue_protocol.h rule 1). */
    context->status[FTDB_STATUS_PASSES_COMPLETED] = passes_completed;
    context->status[FTDB_STATUS_BLOCKS_ISSUED] = blocks_issued;
    context->status[FTDB_STATUS_START_CLOCK] = start_clock;
    context->status[FTDB_STATUS_END_CLOCK] = end_clock;
    context->status[FTDB_STATUS_ELAPSED_CYCLES] = end_clock - start_clock;
    context->status[FTDB_STATUS_FINAL_WALK_OFFSET] = final_walk_offset;
    context->status[FTDB_STATUS_ERROR] = (uint32_t)error;
    context->status[FTDB_STATUS_MAGIC_WORD] = FTDB_STATUS_MAGIC;
}

/**
 * Every refusal this kernel can make, all of them before a single transaction is issued.
 *
 * The host-side checker (ft_tt_noc_check_transfer) runs HOST-side and cannot see a
 * kernel-issued transfer, so the rules have to be enforced here, at the point of issue —
 * that is the generalisable half of the incident that cost cores 50 and 52
 * (DEV_NOTES.md §1). A zero-length NoC transfer is the shape that took this card to
 * `Mem- BusMaster-` on 2026-08-12.
 */
static int ftdb_validate(const ftdb_context *context)
{
    uint64_t l1_span;
    uint64_t dram_span;

    if (context->block_bytes == 0u || context->blocks_per_pass == 0u ||
        context->barrier_every == 0u || context->iterations == 0u ||
        context->wrap_blocks == 0u || context->l1_slot_count == 0u)
        return FTDB_ERROR_ZERO_LENGTH;

    if ((context->block_bytes % FTDB_DRAM_READ_ALIGNMENT) != 0u ||
        (context->dram_base % FTDB_DRAM_READ_ALIGNMENT) != 0u ||
        (context->block_stride % FTDB_DRAM_READ_ALIGNMENT) != 0u ||
        (context->l1_destination % FTDB_DRAM_READ_ALIGNMENT) != 0u ||
        (context->l1_slot_stride % FTDB_DRAM_READ_ALIGNMENT) != 0u)
        return FTDB_ERROR_ALIGNMENT;

    /* Overlapping blocks or slots would make the read-back check meaningless, and an L1
     * slot shorter than a block would have one read scribbling on the next slot. */
    if (context->block_stride < context->block_bytes ||
        context->l1_slot_stride < context->block_bytes ||
        context->direction > FTDB_DIRECTION_READ_PIPELINED)
        return FTDB_ERROR_SHAPE;

    /* The whole destination window must be inside L1, computed in 64 bits so the bound
     * cannot be defeated by the multiplication wrapping. */
    l1_span = (uint64_t)context->l1_destination +
              (uint64_t)context->l1_slot_count * (uint64_t)context->l1_slot_stride;
    if (l1_span > (uint64_t)FT_TENSIX_L1_SIZE_BYTES)
        return FTDB_ERROR_L1_BOUNDS;

    /* Same for the walk: it must stay inside the 32-bit address a runtime arg can express. */
    dram_span = (uint64_t)context->dram_base +
                (uint64_t)context->wrap_blocks * (uint64_t)context->block_stride;
    if (dram_span > 0x100000000uLL)
        return FTDB_ERROR_SHAPE;

    return FTDB_OK;
}

/**
 * One pass: `blocks_per_pass` reads, a read barrier every `barrier_every` of them, and a
 * final barrier so the pass is complete when it returns.
 *
 * `*walk_address` and `*slot_address` are carried across passes, which is what makes the
 * walk continuous rather than restarting — a restart would re-read the same few blocks and
 * turn a streaming test back into a cached one.
 */
static void ftdb_run_pass(const ftdb_context *context, uint32_t wrap_end, uint32_t slot_end,
                          uint32_t *walk_address, uint32_t *slot_address)
{
    uint32_t block;
    uint32_t since_barrier = 0u;

    for (block = 0u; block < context->blocks_per_pass; block++) {
        if (context->direction == FTDB_DIRECTION_READ_PIPELINED && since_barrier == 0u)
            ft_tensix_noc_read_set_trid(ftdb_trid_current);
        if (context->direction == FTDB_DIRECTION_WRITE)
            ft_tensix_noc_write(*slot_address,
                                ft_tensix_noc_address(context->dram_x, context->dram_y,
                                                      *walk_address),
                                context->block_bytes);
        else
            ft_tensix_noc_read(ft_tensix_noc_address(context->dram_x, context->dram_y,
                                                     *walk_address),
                               *slot_address, context->block_bytes);

        *walk_address += context->block_stride;
        if (*walk_address >= wrap_end)
            *walk_address = context->dram_base;
        *slot_address += context->l1_slot_stride;
        if (*slot_address >= slot_end)
            *slot_address = context->l1_destination;

        since_barrier++;
        if (since_barrier == context->barrier_every) {
            if (context->direction == FTDB_DIRECTION_READ_PIPELINED) {
                if (ftdb_trid_previous != 0u)
                    ft_tensix_noc_read_barrier_trid(ftdb_trid_previous);
                ftdb_trid_previous = ftdb_trid_current;
                ftdb_trid_current = ftdb_trid_current == 1u ? 2u : 1u;
            } else {
                ftdb_barrier(context);
            }
            since_barrier = 0u;
        }
    }
    if (since_barrier != 0u)
        ftdb_barrier(context);
}

void kernel_main(void)
{
    ftdb_context context;
    uint32_t wrap_end;
    uint32_t slot_end;
    uint32_t walk_address;
    uint32_t slot_address;
    uint32_t pass;
    uint32_t start_clock = 0u;
    uint32_t end_clock = 0u;
    int error;

    ftdb_read_arguments(&context);
    error = ftdb_validate(&context);
    if (error != FTDB_OK) {
        /* Nothing was issued, so there is nothing to drain. */
        ftdb_publish(&context, 0u, 0u, 0u, 0u, 0u, error);
        return;
    }

    wrap_end = context.dram_base + context.wrap_blocks * context.block_stride;
    slot_end = context.l1_destination + context.l1_slot_count * context.l1_slot_stride;
    walk_address = context.dram_base;
    slot_address = context.l1_destination;

    /* Pass 0 is untimed: it absorbs the instruction cache, the first row-buffer miss and
     * the launch skew of the other cores in the group. */
    ftdb_run_pass(&context, wrap_end, slot_end, &walk_address, &slot_address);
    start_clock = ft_bh_wall_clock_low();
    if (context.duration_cycles != 0u) {
        /* The clock is read once per PASS, not once per read, so the deadline costs one
         * MMIO read per `blocks_per_pass` reads and cannot be inside the measured rate at
         * any depth this suite uses. */
        for (pass = 0u; ft_bh_wall_clock_low() - start_clock < context.duration_cycles;
             pass++)
            ftdb_run_pass(&context, wrap_end, slot_end, &walk_address, &slot_address);
    } else {
        for (pass = 0u; pass < context.iterations; pass++)
            ftdb_run_pass(&context, wrap_end, slot_end, &walk_address, &slot_address);
    }
    end_clock = ft_bh_wall_clock_low();
    if (context.direction == FTDB_DIRECTION_READ_PIPELINED) {
        ft_tensix_noc_read_barrier();
        ft_tensix_noc_read_set_trid(0u);
    }

    /* Outside the timed window and unconditional: brisck.cc asserts both counters are
     * settled after kernel_main() returns, and a transaction left in flight writes L1 the
     * next kernel is about to use. No write or atomic is issued above, so these are cheap. */
    ft_tensix_noc_write_barrier();
    ft_tensix_noc_atomic_barrier();

    ftdb_publish(&context, pass + 1u, pass * context.blocks_per_pass, start_clock, end_clock,
                 walk_address - context.dram_base, FTDB_OK);
}
