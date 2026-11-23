/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * multicast_bench_reader.c
 *   ONE BRISC kernel, in C99, that plays every role the multicast measurement needs:
 *   the multicast sender, a multicast receiver, or a core reading the same block out of
 *   DRAM for itself. The role is a RUNTIME argument.
 *
 *   **One binary for all three roles is the measurement's method, not a convenience.**
 *   The whole question is "what does a multicast cost against N cores each reading DRAM",
 *   and if the two arms were two binaries then every difference between them -- inlining,
 *   text placement, which registers the loop got -- would be inside the number. Same
 *   binary, same L1 map, same issue layer (kernels/ft_tensix_noc_multicast.h), one runtime
 *   word apart. This is the same rule `weight_n_major` follows in matmul_reader.c for
 *   the same reason (docs/tt_kernel_model.md §14).
 *
 *   It is C because the dataflow half of a kernel does not need C++: everything here is a
 *   store to a fixed register address or a do/while over one. It includes NOTHING from
 *   tt-metal -- only kernels/ft_tensix_dataflow.h (Phase 6.6's C99 dataflow library) and
 *   kernels/ft_tensix_noc_multicast.h on top of it. build_tensix_dataflow_kernel.sh's `.c`
 *   path is how a C translation unit becomes a Tensix ELF.
 *
 * ===========================================================================
 * WHAT IT MEASURES, AND WHY IT IS TIMED THIS WAY
 * ===========================================================================
 * Both arms deliver the SAME thing: one block of `payload_bytes` into the SAME L1 address
 * on N cores, `iterations` times. The measured quantity is the steady-state period of
 * that, per core, in wall-clock cycles read off the core's own counter.
 *
 *   MULTICAST arm   1 sender + N receivers. Per round the sender reads the block from
 *                   DRAM once and multicasts it; each receiver announces readiness and
 *                   waits for it. A receiver's per-round time is "how long until the
 *                   block is in MY L1", which is what a cooperating matmul core actually
 *                   waits on.
 *   DRAM arm        N cores, no sender, no synchronisation: each reads the block from
 *                   DRAM into the same L1 address itself. This is the status quo the
 *                   plan calls out -- "every A tile re-read Nt times".
 *
 * **Timing is device-side on purpose.** The host is two 1.8 GHz cores that also drive
 * eight K80s and drift enough over ten seconds to swamp a microsecond (DEV_NOTES.md §8),
 * and a launch is ~14.5 us against a round that should be ~1 us. Each core reads
 * RISCV_DEBUG_REG_WALL_CLOCK_0 itself, so launch skew across N cores affects how much the
 * arms OVERLAP -- which is a real property of the arm -- and not the measurement.
 *
 * **Round 0 is not timed.** The clock is sampled AFTER round 0 and after round
 * `iterations`, so the window is exactly `iterations` steady-state rounds and excludes
 * the sender waiting for receivers to boot, the first DRAM row-buffer miss, and the
 * instruction cache warming.
 *
 * ===========================================================================
 * HOW THE RESULT IS PROVEN TO BE THE RESULT
 * ===========================================================================
 * A destination that matches the expected bytes could be stale L1 (DEV_NOTES.md §3), so
 * the payload ALTERNATES between two different blocks staged at two different DRAM
 * addresses, by round parity. After `iterations + 1` rounds every destination must hold
 * block `iterations & 1` and not the other one, on every core -- which shows the data
 * tracked the source and that the LAST round landed, not merely that something arrived.
 *
 * The alternation is by ADDRESS, never by mutating a buffer in place. A RISC-V store
 * followed by a NoC read of the same L1 word is not ordered on this part, and it is a race
 * you lose every time, not sometimes -- the group-completion post found that and an
 * `invalidate_l1_cache()` between them did not fix it
 * (kernels/resident_queue_protocol.h, ftq_group_post).
 *
 * ===========================================================================
 * RUNTIME ARGUMENTS -- one layout for every role
 * ===========================================================================
 * One layout rather than three, because three arg maps is three things to keep in step
 * with one host, and a runtime-arg block that did not say what its reader thought has
 * already stalled a core here (docs/tt_launch_notes.md §3).
 *
 *   0  role                       FTMB_ROLE_*
 *   1  iterations                 timed rounds; round 0 runs untimed on top
 *   2  payload bytes              multiple of 64 (DRAM read alignment is the strictest end)
 *   3  staging A L1 address       sender only: DRAM lands here on EVEN rounds
 *   4  staging B L1 address       sender only: odd rounds
 *   5  shared destination address the SAME L1 address on every receiver / DRAM reader.
 *                                 The hardware requires it for multicast; the DRAM arm
 *                                 uses it too so the two arms differ in nothing else
 *   6  rectangle x low            TRANSLATED NoC coordinates, from the host's ft_tt_coord
 *   7  rectangle y low
 *   8  rectangle x high
 *   9  rectangle y high
 *  10  destination count          receivers, EXCLUDING the sender
 *  11  sender NoC x
 *  12  sender NoC y
 *  13  ready semaphore address    on the SENDER; receivers atomically increment it
 *  14  arrival semaphore address  on EVERY receiver, same address
 *  15  valid word address         on the sender; the host stages FT_TENSIX_SEMAPHORE_VALID here
 *  16  DRAM NoC x
 *  17  DRAM NoC y
 *  18  DRAM address A
 *  19  DRAM address B
 *  20  status block address
 *  21  stop flag address
 *  22  spin limit                 every wait is bounded as well as askable
 *
 * Compile-time args: none. Everything about the CALL is a runtime arg
 * (docs/tt_kernel_model.md §14).
 *
 *   Build (BRISC, NoC 0):
 *     CTARGS=0 NOC=0 bash foundry/tt/build_tensix_dataflow_kernel.sh \
 *         foundry/tt/kernels/multicast_bench_reader.c brisc /tmp/ttk_multicast
 *
 * Status: NEW 2026-08-13. Built and run on the p150a card host by tests/test_multicast_bench.c.
 * Issues: None
 * Todo:   None
 */

#include "ft_tensix_dataflow.h"
#include "ft_tensix_noc_multicast.h"

/* ---------------------------------------------------------------------------
 * The wire contract. tests/test_multicast_bench.c carries the same values; change
 * both or neither.
 * --------------------------------------------------------------------------- */

#define FTMB_ROLE_MULTICAST_SENDER 1u
#define FTMB_ROLE_MULTICAST_RECEIVER 2u
#define FTMB_ROLE_DRAM_READER 3u

/** "FTMB". A status block still reading the host's poison means the kernel never ran. */
#define FTMB_STATUS_MAGIC 0x46544D42u

#define FTMB_STATUS_MAGIC_WORD 0u
#define FTMB_STATUS_ROLE 1u
#define FTMB_STATUS_ROUNDS_COMPLETED 2u
#define FTMB_STATUS_ELAPSED_CYCLES 3u
#define FTMB_STATUS_ERROR 4u
#define FTMB_STATUS_START_CLOCK 5u
#define FTMB_STATUS_END_CLOCK 6u
#define FTMB_STATUS_LAST_ISSUE_CODE 7u
#define FTMB_STATUS_WORDS 8u

/** DRAM's read alignment, the stricter of the two ends of a DRAM->L1 transfer
 *  (ft_tt_noc.h, ft_tt_noc_alignment). A derived address that is only 16- or 32-byte
 *  aligned reads back byte-exact until its LAST 64 bytes, silently
 *  (tests/ft_tt_phase2_l1_layout.h). The NoC index this kernel issues on is NOC_INDEX,
 *  baked in by the build and read by the multicast header -- which is where the rectangle's
 *  corner order depends on it. */
#define FTMB_DRAM_READ_ALIGNMENT 64u

/** L1's alignment, which is what a multicast destination takes. */
#define FTMB_L1_ALIGNMENT 16u

/* --------------------------------------------------------------------------- */

/** Everything the three roles share, read once at entry so the timed loops index
 *  registers instead of re-reading L1. */
typedef struct ftmb_context {
    uint32_t role;
    uint32_t iterations;
    uint32_t payload_bytes;
    uint32_t staging_address[2];
    uint32_t shared_destination_address;
    uint32_t rectangle_x_low;
    uint32_t rectangle_y_low;
    uint32_t rectangle_x_high;
    uint32_t rectangle_y_high;
    uint32_t destination_count;
    uint32_t sender_x;
    uint32_t sender_y;
    uint32_t ready_semaphore_address;
    uint32_t arrival_semaphore_address;
    uint32_t valid_word_address;
    uint32_t dram_x;
    uint32_t dram_y;
    uint32_t dram_address[2];
    volatile uint32_t *status;
    volatile const uint32_t *stop_flag;
    uint32_t spin_limit;
} ftmb_context;

static void ftmb_read_arguments(ftmb_context *context)
{
    context->role = ft_tensix_runtime_arg((int)0u);
    context->iterations = ft_tensix_runtime_arg((int)1u);
    context->payload_bytes = ft_tensix_runtime_arg((int)2u);
    context->staging_address[0] = ft_tensix_runtime_arg((int)3u);
    context->staging_address[1] = ft_tensix_runtime_arg((int)4u);
    context->shared_destination_address = ft_tensix_runtime_arg((int)5u);
    context->rectangle_x_low = ft_tensix_runtime_arg((int)6u);
    context->rectangle_y_low = ft_tensix_runtime_arg((int)7u);
    context->rectangle_x_high = ft_tensix_runtime_arg((int)8u);
    context->rectangle_y_high = ft_tensix_runtime_arg((int)9u);
    context->destination_count = ft_tensix_runtime_arg((int)10u);
    context->sender_x = ft_tensix_runtime_arg((int)11u);
    context->sender_y = ft_tensix_runtime_arg((int)12u);
    context->ready_semaphore_address = ft_tensix_runtime_arg((int)13u);
    context->arrival_semaphore_address = ft_tensix_runtime_arg((int)14u);
    context->valid_word_address = ft_tensix_runtime_arg((int)15u);
    context->dram_x = ft_tensix_runtime_arg((int)16u);
    context->dram_y = ft_tensix_runtime_arg((int)17u);
    context->dram_address[0] = ft_tensix_runtime_arg((int)18u);
    context->dram_address[1] = ft_tensix_runtime_arg((int)19u);
    context->status = (volatile uint32_t *)(unsigned long)ft_tensix_runtime_arg((int)20u);
    context->stop_flag = (volatile const uint32_t *)(unsigned long)ft_tensix_runtime_arg((int)21u);
    context->spin_limit = ft_tensix_runtime_arg((int)22u);
}

static void ftmb_publish(const ftmb_context *context, uint32_t rounds_completed,
                         uint32_t start_clock, uint32_t end_clock, int error,
                         int last_issue_code)
{
    /* Magic LAST, so a host reading a torn block sees "never ran" rather than a
     * half-filled result -- the same ordering rule the resident queue's
     * token-before-count uses (kernels/resident_queue_protocol.h rule 1). */
    context->status[FTMB_STATUS_ROLE] = context->role;
    context->status[FTMB_STATUS_ROUNDS_COMPLETED] = rounds_completed;
    context->status[FTMB_STATUS_START_CLOCK] = start_clock;
    context->status[FTMB_STATUS_END_CLOCK] = end_clock;
    context->status[FTMB_STATUS_ELAPSED_CYCLES] = end_clock - start_clock;
    context->status[FTMB_STATUS_ERROR] = (uint32_t)error;
    context->status[FTMB_STATUS_LAST_ISSUE_CODE] = (uint32_t)last_issue_code;
    context->status[FTMB_STATUS_MAGIC_WORD] = FTMB_STATUS_MAGIC;
}

/**
 * One block out of DRAM into local L1, with the guards the shared read path leaves to its
 * caller.
 *
 * `ft_tensix_noc_read()` deliberately does not check alignment, because the rule depends on
 * the SOURCE space -- 64 bytes reading DRAM, 16 reading L1 -- and a 64-bit NoC address does
 * not say which endpoint it names. This function knows the source is DRAM, so it applies
 * the rule and REFUSES rather than issuing. The zero-length guard is the one that matters:
 * a zero-length NoC transfer is what took this card from working to `Mem- BusMaster-` on
 * 2026-08-12 with no error anywhere until the next read returned 0xFFFFFFFF
 * (DEV_NOTES.md §1), and a host-side check cannot see a kernel-issued transfer.
 */
static int ftmb_read_from_dram(const ftmb_context *context, uint32_t dram_address,
                               uint32_t destination_l1_address)
{
    if (context->payload_bytes == 0u)
        return FT_TENSIX_MULTICAST_ERROR_ZERO_LENGTH;
    if ((context->payload_bytes % FTMB_DRAM_READ_ALIGNMENT) != 0u ||
        (dram_address % FTMB_DRAM_READ_ALIGNMENT) != 0u ||
        (destination_l1_address % FTMB_DRAM_READ_ALIGNMENT) != 0u)
        return FT_TENSIX_MULTICAST_ERROR_ALIGNMENT;

    ft_tensix_noc_read(ft_tensix_noc_address(context->dram_x, context->dram_y, dram_address),
                       destination_l1_address, context->payload_bytes);
    return FT_TENSIX_MULTICAST_OK;
}

/* ---------------------------------------------------------------------------
 * The three roles
 * --------------------------------------------------------------------------- */

/**
 * One sender round: collect every receiver, fetch the block from DRAM, multicast it, then
 * announce it.
 *
 * The order of the last three is the protocol and none of it is rearrangeable --
 * ft_tt_multicast_kernel.h's header block has the full account, including why the flush
 * between the two multicasts is mandatory on Blackhole and nowhere else.
 */
static int ftmb_sender_round(const ftmb_context *context, uint32_t round, int *out_issue_code)
{
    volatile uint32_t *ready_semaphore =
        (volatile uint32_t *)(unsigned long)context->ready_semaphore_address;
    const uint32_t parity = round & 1u;
    uint64_t payload_destination;
    uint64_t arrival_destination;
    int code;

    code = ft_tensix_wait_semaphore_at_least(ready_semaphore, context->destination_count,
                                        context->spin_limit, context->stop_flag);
    if (code != FT_TENSIX_MULTICAST_OK)
        return code;
    ft_tensix_set_local_semaphore(ready_semaphore, 0u);

    code = ftmb_read_from_dram(context, context->dram_address[parity],
                               context->staging_address[parity]);
    *out_issue_code = code;
    if (code != FT_TENSIX_MULTICAST_OK)
        return code;
    ft_tensix_noc_read_barrier();

    code = ft_tensix_multicast_address(context->rectangle_x_low, context->rectangle_y_low,
                                       context->rectangle_x_high, context->rectangle_y_high,
                                       context->shared_destination_address,
                                       &payload_destination);
    if (code != FT_TENSIX_MULTICAST_OK) {
        *out_issue_code = code;
        return code;
    }
    code = ft_tensix_noc_write_multicast(context->staging_address[parity], payload_destination,
                                         context->payload_bytes, context->destination_count);
    *out_issue_code = code;
    if (code != FT_TENSIX_MULTICAST_OK)
        return code;

    /* BLACKHOLE ONLY, and load-bearing: the data multicast and the semaphore multicast go
     * into different command-buffer FIFOs and are not guaranteed to leave in issue order,
     * so without this a receiver can be told the block arrived before it did and read the
     * PREVIOUS round's bytes with nothing reporting an error. */
    ft_tensix_noc_writes_flushed_barrier();

    code = ft_tensix_multicast_address(context->rectangle_x_low, context->rectangle_y_low,
                                       context->rectangle_x_high, context->rectangle_y_high,
                                       context->arrival_semaphore_address,
                                       &arrival_destination);
    if (code != FT_TENSIX_MULTICAST_OK) {
        *out_issue_code = code;
        return code;
    }
    code = ft_tensix_noc_semaphore_set_multicast(context->valid_word_address,
                                                 arrival_destination,
                                                 context->destination_count);
    *out_issue_code = code;
    return code;
}

/**
 * One receiver round: clear the arrival flag, announce readiness, wait for the block.
 *
 * The local clear comes FIRST. The sender cannot set the flag until it has seen this
 * core's atomic, and this core issues that atomic after the clear -- so a VALID can never
 * be erased by the clear that was meant to precede it.
 */
static int ftmb_receiver_round(const ftmb_context *context, int *out_issue_code)
{
    volatile uint32_t *arrival_semaphore =
        (volatile uint32_t *)(unsigned long)context->arrival_semaphore_address;
    int code;

    ft_tensix_set_local_semaphore(arrival_semaphore, FT_TENSIX_SEMAPHORE_INVALID);
    ft_tensix_noc_semaphore_increment(
        ft_tensix_noc_address(context->sender_x, context->sender_y,
                              context->ready_semaphore_address),
        1u);
    *out_issue_code = FT_TENSIX_MULTICAST_OK;
    return ft_tensix_wait_semaphore_equals(arrival_semaphore, FT_TENSIX_SEMAPHORE_VALID,
                                           context->spin_limit, context->stop_flag);
}

/** One DRAM-reader round: exactly the sender's fetch, with no sender and no rendezvous. */
static int ftmb_dram_reader_round(const ftmb_context *context, uint32_t round,
                                  int *out_issue_code)
{
    const uint32_t parity = round & 1u;
    int code = ftmb_read_from_dram(context, context->dram_address[parity],
                                   context->shared_destination_address);

    *out_issue_code = code;
    if (code != FT_TENSIX_MULTICAST_OK)
        return code;
    ft_tensix_noc_read_barrier();
    return FT_TENSIX_MULTICAST_OK;
}

static int ftmb_run_round(const ftmb_context *context, uint32_t round, int *out_issue_code)
{
    if (context->role == FTMB_ROLE_MULTICAST_SENDER)
        return ftmb_sender_round(context, round, out_issue_code);
    if (context->role == FTMB_ROLE_MULTICAST_RECEIVER)
        return ftmb_receiver_round(context, out_issue_code);
    if (context->role == FTMB_ROLE_DRAM_READER)
        return ftmb_dram_reader_round(context, round, out_issue_code);
    return FT_TENSIX_MULTICAST_ERROR_RECTANGLE; /* an unknown role is a refusal, never a no-op */
}

/* --------------------------------------------------------------------------- */

void kernel_main(void)
{
    ftmb_context context;
    uint32_t round;
    uint32_t rounds_completed = 0u;
    uint32_t start_clock = 0u;
    uint32_t end_clock = 0u;
    int issue_code = FT_TENSIX_MULTICAST_OK;
    int error = FT_TENSIX_MULTICAST_OK;

    ftmb_read_arguments(&context);

    /* Round 0 is untimed: it absorbs the sender waiting for receivers to boot, the first
     * DRAM row-buffer miss and the instruction cache. Then `iterations` timed rounds. */
    for (round = 0u; round <= context.iterations; round++) {
        error = ftmb_run_round(&context, round, &issue_code);
        if (error != FT_TENSIX_MULTICAST_OK)
            break;
        rounds_completed++;
        if (round == 0u)
            start_clock = ft_bh_wall_clock_low();
    }
    end_clock = ft_bh_wall_clock_low();

    /* Outside the timed window, and unconditional: an abandoned round may have left a
     * write or an atomic in flight, and brisck.cc asserts both counters are settled after
     * kernel_main() returns. Leaving a transaction outstanding is also how L1 the next
     * kernel is about to use gets written underneath it. */
    ft_tensix_noc_write_barrier();
    ft_tensix_noc_atomic_barrier();

    ftmb_publish(&context, rounds_completed, start_clock, end_clock, error, issue_code);
}
