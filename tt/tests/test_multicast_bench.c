/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_multicast_bench.c — what a NoC multicast COSTS, against N cores each reading the
 * same block out of DRAM for themselves.
 *
 * This is the measurement the whole multicast primitive exists to justify. The plan has
 * named the blocker since Phase 4 — "no reuse: every A tile re-read Nt times, every B tile
 * Mt times" — and the fix is to read a shared operand once and multicast it. Whether that
 * is worth building is a NUMBER, and this program is that number.
 *
 * ===========================================================================
 * THE THREE ARMS, AND WHY THEY ARE COMPARABLE
 * ===========================================================================
 * All three deliver the SAME thing: one block of `payload` bytes into the SAME L1 address on
 * N cores, `iterations` times, from the same DRAM bank at the same address.
 *
 *   MULTICAST  one extra core is the sender. Per round it reads the block from DRAM once
 *              and multicasts it to the N receivers, with the two-semaphore rendezvous.
 *              A receiver's per-round time is "how long until the block is in MY L1".
 *   DRAM       the SAME N cores, no sender, no synchronisation: each reads the block from
 *              DRAM itself, all of them through the bank's PORT 0 — which is what
 *              `ft_tt_coord_dram_bank()` returns and therefore what every reader in this
 *              backend used. N readers, ONE endpoint, ~45 GB/s between them.
 *   DRAM+PORTS the same again, except reader `r` is handed door `r % 3` of the SAME bank.
 *              No bytes move and the kernel is not touched — three ports of a bank are
 *              three doors into one memory (ft_tt_dram_stripe.h) — so the arm isolates the
 *              ADDRESSING and nothing else, and what it can buy is the bank's ~64 GB/s
 *              instead of one door's ~45.
 *
 * **DRAM+PORTS is the arm the multicast now has to beat.** The recorded crossover below was
 * measured when every reader was stuck on one door; a shared operand that N cores can pull
 * through three doors is a different opponent, and quoting the old DRAM column against a
 * rotated world would restate the conclusion instead of testing it.
 *
 * Three things make the comparison mean something:
 *  - **Same binary.** kernels/multicast_bench_reader.c plays all three roles and the role
 *    is one RUNTIME word. Two binaries would put every codegen difference inside the
 *    number.
 *  - **Same cores, same L1 addresses, same DRAM.** The arms differ in the role word and
 *    nothing else.
 *  - **Interleaved, and repeated.** MCAST, DRAM, MCAST, DRAM, … at each N, medians
 *    reported. This host is two 1.8 GHz cores also driving eight K80s and it drifts enough
 *    over ten seconds to swamp a microsecond — measuring one arm to completion and then
 *    the other once reported a NEGATIVE cost for CONFIGURE_TLB.
 *
 * **Timing is device-side.** Each core reads its own wall clock, so a launch (~14.5 µs)
 * and the host's scheduling are outside the window, and round 0 is excluded so the sender
 * waiting for receivers to boot and the first DRAM row-buffer miss are not in it.
 * Stage 1 CALIBRATES cycles to seconds from two run lengths rather than assuming the
 * measured 800 MHz busy clock, because "the clock did not change under load" and "we
 * sampled while idle" print the same number.
 *
 * ===========================================================================
 * WHAT MAKES A GREEN RUN EVIDENCE
 * ===========================================================================
 *  - The payload ALTERNATES between two different blocks at two different DRAM addresses,
 *    by round parity, so a destination that matches proves the data TRACKED the source and
 *    that the LAST round landed — not merely that something arrived.
 *  - Every destination is POISONED before the run and the poison must be gone.
 *  - Every receiver is checked against the OTHER block too and must NOT match it.
 *  - The status block is host-POISONED before launch, so "never ran" and "finished" are
 *    different bit patterns rather than the same zero.
 *  - Stage 2 is the RED HALF: two runs that must REFUSE — a zero-length payload and a
 *    rectangle with zero destinations. Both are the hazard that took this card to
 *    `Mem- BusMaster-` on 2026-08-12, and both are safe to run precisely
 *    because the refusal is in the kernel at the point of issue. If either came back
 *    FTMC_OK the guards would be decorative.
 *  - The suite ENDS by proving every core it touched still BOOTS, unconditionally, even
 *    when an earlier stage failed. A read cannot tell a wedged core from a healthy idle
 *    one. The COMPUTE half of that check is `ft_tt_core probe`, which the runner script
 *    runs immediately after this program and which is a DIFFERENT triad — see
 *    run_multicast_suite.sh.
 *
 * ===========================================================================
 * THE RESULT, MEASURED 2026-08-13 (device clock calibrated at 800.0 MHz)
 * ===========================================================================
 * **Multicast pays only when the shared block is big enough that the DRAM arm is
 * contended, and at one tile it never is.** Two sweeps, same binary, same cores:
 *
 *   2048 B (one bf16 tile), 20000 rounds — multicast LOSES at every N:
 *     N=1  2122 vs  661 ns   N=4  2269 vs 661   N=8  2272 vs  679   N=10 2323 vs 839
 *   The DRAM arm is FLAT to N=8 and only +27% at N=10, so there is nothing to save; the
 *   multicast's 2.1 us floor is the rendezvous, and it grows only ~50 ns per receiver.
 *
 *   98304 B (a whole depth-48 A block), 2000 rounds — multicast crosses over at N=3:
 *     N=1  5916 vs 3088 ns (1.92x worse)     N=3  6219 vs 6661 (0.93x, WINS)
 *     N=2  6057 vs 4442 ns (1.36x worse)     N=4  6370 vs 8881 (0.72x, 1.39x FASTER)
 *
 * The shape is the whole point and it is what a matmul needs: the DRAM arm grows LINEARLY
 * with N (+1.93 us per extra core) while multicast is FLAT (+0.15 us). At 96 KiB the A-side
 * really is contended, and multicast turns an O(N) cost into an O(1) one — which is exactly
 * the "no reuse: every A tile re-read Nt times" blocker the plan has named since Phase 4.
 * INFERRED from that linear fit, NOT measured (a 5-wide rectangle wedges cores, below):
 * 2.4x at 8 receivers, 2.8x at 10.
 *
 * ===========================================================================
 * A KNOWN, MEASURED HAZARD: A 5-WIDE RECTANGLE LEAVES SOME RECEIVERS UNBOOTABLE
 * ===========================================================================
 * MEASURED 2026-08-13, and found by this suite's own stage 4. After a run whose multicast
 * rectangle is FIVE COLUMNS WIDE, a minority of the receivers cannot complete their next
 * firmware boot: 1 of 5 on a 1x5, 3 of 10 on a 2x5. The signature is the recoverable
 * family — `soft reset 0x00047000`, BRISC out of reset, subordinates never released.
 *
 * What is known, all of it measured:
 *  - Narrower rectangles are clean. 1x1, 1x2, 1x3 and 1x4 were each run in isolation
 *    (FT_TT_MULTICAST_ARMS=multicast) and every core re-booted; the 1x2 case survived five
 *    independent runs.
 *  - It is not cumulative. Four consecutive 1x2 runs — about thirty launches and eight
 *    boots — stayed green, so it is not launch or boot count.
 *  - The affected cores look PERFECTLY HEALTHY until you boot them. Every run completes
 *    with the right payload, the right round count and no timeout, and stage 3b reads
 *    `soft reset 0x00000000` with every subordinate sync at DONE immediately before the
 *    re-boot that then fails. This is precisely why the BOOT check has to be a boot.
 *  - WIDTH and RECEIVER COUNT are CONFOUNDED here and this card cannot separate them with
 *    the cores this agent owns: five receivers only fits as 1x5, and ten only as 2x5, so
 *    "5 wide" and ">= 5 receivers" have the same evidence. A part with more rows available
 *    would settle it with a 5x1.
 *
 * UNTIL THAT IS UNDERSTOOD, run the sweep at FT_TT_MULTICAST_POOL_COLUMNS=4 or below unless
 * you are deliberately reproducing the hazard, and always through run_multicast_suite.sh so
 * the compute probe runs afterwards.
 *
 * ===========================================================================
 * TWO EXPERIMENTS THAT ASK WHETHER THE BROADCAST IS THE CAUSE AT ALL
 * ===========================================================================
 * A source review on 2026-08-16 found that our multicast ISSUE PATH is a bit-for-bit
 * transliteration of upstream's — same command word, register order, counter accounting,
 * VC, rectangle encoding, semaphore multicast, the Blackhole inter-multicast flush and the
 * two-semaphore rendezvous — and that upstream states in writing there is no destination
 * cap and ships two users that exceed four ON THIS SILICON (their multicore matmul at 13
 * and 9 destinations; their dispatcher multicasting the go signal to the whole worker grid
 * on every program launch). Nothing in either tree explains a threshold at exactly five.
 *
 * What upstream never does is our TEARDOWN. tt-metal boots worker firmware once per device
 * open and dispatches into a persistent loop; this program calls `ft_tt_core_park()`
 * (assert reset) and then `ft_tt_core_boot_firmware()` PER RUN (boot_core(), below). Every
 * wedge on record is the NEXT BOOT after a clean run, and DEV_NOTES.md §1 already records
 * two cores dying exactly that way with no multicast anywhere near them.
 *
 * So `FT_TT_MULTICAST_EXPERIMENT` selects one of two arms in place of the timed sweep.
 * NEITHER IS A MEASUREMENT — they are correctness arms and they print no ns/block.
 *
 *   relaunch  EXPERIMENT A. Run the multicast arm at the pool's shape, then, WITHOUT any
 *             park or re-boot, re-plant runtime args and re-launch the same still-booted
 *             program on the same cores (stage 3c). Stage 4's boot check then runs as
 *             usual, so one process produces both answers.
 *
 *               stage 3c CLEAN and stage 4 RED  =>  the broadcast is innocent and the
 *                   defect is in park/boot after a broadcast has landed in a core's L1.
 *               stage 3c RED                    =>  the broadcast itself damaged the core
 *                   with no re-boot involved, and the rule stands as written.
 *               stage 3c CLEAN and stage 4 GREEN => this shape did not reproduce the wedge
 *                   at all; the run says nothing either way and must not be quoted.
 *
 *             The probe is a DRAM-reader launch, not another broadcast, on purpose: a
 *             re-launched multicast would hang at the rendezvous if a receiver were
 *             damaged, which conflates "the core is dead" with "the group is dead". A DRAM
 *             reader needs no peer, so each core answers for itself.
 *
 *   split     EXPERIMENT B. At N=8, TWO senders each broadcast to their own disjoint row of
 *             four receivers, instead of one sender broadcasting to eight (stage 3d). Same
 *             cores, same rounds, same teardown, and no packet ever exceeds four
 *             destinations. Stage 4 then decides it:
 *
 *               stage 4 GREEN =>  the variable is DESTINATIONS PER PACKET, and the
 *                   `ceil(N/4)` workaround is sound.
 *               stage 4 RED   =>  the variable is "this core took part in a run with >= 5
 *                   receivers", which is not a property of a packet at all and points
 *                   straight back at experiment A.
 *
 *             It costs one confound worth stating: two senders, not one. Making it one
 *             sender issuing two broadcasts per round would need a kernel change, and a
 *             kernel change would alter what the experiment measures.
 *
 * Both arms end in the same unconditional health stages as everything else here, and both
 * are gated by tests/test_relaunch_gate.c off-hardware — the verdict is a pure function
 * precisely so that "the second launch was fine" can be proven to be a statement the
 * instrument was capable of contradicting.
 *
 * ===========================================================================
 * CORES
 * ===========================================================================
 * Live Tensix indices FT_TT_MULTICAST_FIRST_INDEX .. +11, which on this 14-column card is
 * one straight row and therefore a legal multicast rectangle. Consecutive live indices are
 * NOT a rectangle in general — they wrap the row — which is what
 * ft_tt_group_rectangle_coordinates() refuses and why this program asks for the block
 * through it rather than walking indices itself.
 *
 * Usage:
 *   test_multicast_bench <program-directory> [iterations] [payload-bytes]
 * Environment:
 *   FT_TT_MULTICAST_FIRST_INDEX   live index of the sender (default 100)
 *   FT_TT_MULTICAST_MAX_RECEIVERS largest N in the sweep (default 11)
 *   FT_TT_MULTICAST_EXPERIMENT    none (default) | relaunch | split — see above
 *   FT_TT_MULTICAST_SENDER_B_INDEX  live index of the SECOND sender, `split` only
 *   FT_TT_MULTICAST_RELAUNCH_CYCLES how many broadcast+probe cycles `relaunch` runs
 *   FT_TT_FIRMWARE_DIR            where the five firmware ELFs live (the same directory)
 *
 * Build:
 *   make -f Makefile.multicast PHASE1B=1 all
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_dram_stripe.h"
#include "ft_tt_group_shared_operand.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_multicast.h"
#include "ft_tt_noc.h"
#include "ft_tt_phase2_l1_layout.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_relaunch_gate.h"
#include "ft_tt_split_broadcast_geometry.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------------- */
/* The wire contract — kernels/multicast_bench_reader.c carries the same values */
/* ------------------------------------------------------------------------- */

#define ROLE_MULTICAST_SENDER 1u
#define ROLE_MULTICAST_RECEIVER 2u
#define ROLE_DRAM_READER 3u

#define STATUS_MAGIC 0x46544D42u /* "FTMB" */
#define STATUS_MAGIC_WORD 0u
#define STATUS_ROLE 1u
#define STATUS_ROUNDS_COMPLETED 2u
#define STATUS_ELAPSED_CYCLES 3u
#define STATUS_ERROR 4u
#define STATUS_START_CLOCK 5u
#define STATUS_END_CLOCK 6u
#define STATUS_LAST_ISSUE_CODE 7u
#define STATUS_WORDS 8u

/** Kernel-side refusal codes, from kernels/ft_tt_multicast_kernel.h. */
#define FTMC_OK 0
#define FTMC_ERROR_ZERO_LENGTH (-1)
#define FTMC_ERROR_ZERO_DESTINATIONS (-2)

#define KERNEL_ARG_COUNT 23u

/** The host poison for the status block. Different from every state the kernel writes and
 *  different from cleared L1, so "never ran" is its own bit pattern. */
#define STATUS_POISON 0x7B1B1B1Bu

/* ------------------------------------------------------------------------- */
/* Test configuration                                                        */
/* ------------------------------------------------------------------------- */

/**
 * The receiver POOL is a rows x columns block of live Tensix indices starting at
 * FIRST_INDEX, and the sender is a core OUTSIDE it.
 *
 * **Five columns, not twelve, and that is a hardware fact rather than a budget.** The
 * Tensix columns of a Blackhole are NoC x 1..7 and 10..16 — x = 8 and x = 9 carry the
 * ARC/L2CPU spine and the east DRAM column. A multicast writes EVERY node in its bounding
 * box, so a rectangle may not straddle that gap, and the widest usable run on the low side
 * is seven columns. MEASURED here on 2026-08-13: asking for one row of twelve consecutive
 * live indices produced x 3..7 and 10..16 and was refused by the fill check before
 * anything was issued. Growing N therefore means growing the rectangle in Y, which is why
 * the sweep is over SHAPES and not over a scalar N.
 */
#define DEFAULT_FIRST_INDEX 100u
#define DEFAULT_SENDER_INDEX 119u
/** The SECOND sender, used only by the `split` experiment. Its default sits next to the
 *  first so that the pair is outside any pool a caller is likely to pick, and the stage
 *  REFUSES rather than assumes: both senders are checked against both rectangles before
 *  anything is issued, because a multicast writes every node in its bounding box and a
 *  sender caught inside the other group's box would have its own staging buffer
 *  overwritten mid-round. */
#define DEFAULT_SENDER_B_INDEX 120u
#define POOL_ROWS 2u
#define POOL_COLUMNS 5u

/**
 * Each experiment's OWN default pool, because inheriting the sweep's 2x5 would be a
 * ten-destination broadcast nobody asked for.
 *
 * Experiment A defaults to 1x5 — the SMALLEST shape that reproduces the wedge (5 of 5 in
 * four of four arms). Going wider proves nothing it does not already prove and puts more
 * cores at risk; 1x4 is the control worth running first, and it is one environment
 * variable away. Experiment B defaults to 2x4, which is N=8 as two rows of four, and is
 * the only shape it accepts without being told otherwise.
 */
/** The standing rule: never more than four destinations per multicast. Named once here and
 *  referred to everywhere, so an arm that exceeds it says so out loud rather than doing it
 *  by arithmetic accident. */
#define SAFE_DESTINATION_LIMIT 4u

#define RELAUNCH_POOL_ROWS 1u
#define RELAUNCH_POOL_COLUMNS 5u
#define SPLIT_POOL_ROWS 2u
#define SPLIT_POOL_COLUMNS 4u
/** Pool + BOTH senders. The second slot is occupied only by the `split` experiment; every
 *  other mode leaves it unused and reports `member_count` one lower, so no stage ever
 *  boots, probes or re-boots a core the run did not use. */
#define MAX_CORES (POOL_ROWS * POOL_COLUMNS + 2u)

#define DEFAULT_ITERATIONS 20000u
#define DEFAULT_PAYLOAD_BYTES 2048u /* one bf16 32x32 tile */

/** Repetitions of each (N, arm) pair. Odd, so the median is a measured value. */
#define REPETITIONS 3u

/** Bounded, per resident-kernel rule 4: a wait that expires must report, never hang. At
 *  800 MHz a spin is a few cycles, so 200 million is well under a second and far longer
 *  than any legitimate wait here. */
#define SPIN_LIMIT 200000000u

#define RUN_TIMEOUT_MS 20000u

/** DRAM bank and the two block addresses the payload alternates between. 64-byte aligned
 *  (DRAM's read alignment, the stricter end of a DRAM->L1 transfer) and deliberately NOT a
 *  power-of-two apart: a power-of-two stride measured 10-14% on the one matmul shape that
 *  had one, INFERRED to be bank/channel aliasing. They are 768 KiB
 *  apart, which is far enough to hold the largest payload this program sweeps (a whole
 *  depth-48 A block, 96 KiB) without the two overlapping. */
#define DRAM_LOGICAL_BANK 3u
#define DRAM_BLOCK_A_ADDRESS 0x02000000uLL
#define DRAM_BLOCK_B_ADDRESS 0x020C0000uLL

#define DESTINATION_POISON_BYTE 0xA5u
#define STAGING_POISON_BYTE 0x5Au
/** What sits at the block addresses of the NEXT bank, so stage 2b's red arm reads something
 *  that is deterministically neither payload rather than uninitialised DRAM. */
#define WRONG_BANK_POISON_BYTE 0x3Cu
/** Rounds for the red arm. It is a correctness check, not a measurement, and every round is
 *  a full payload read by every receiver. */
#define RED_ITERATIONS 8u

/** Rounds for a re-launch probe (experiment A). Small because it is a correctness check —
 *  what is being asked is "does this still-booted core accept work at all", and the answer
 *  does not get truer with 20,000 rounds. It is EVEN so that the parity correction in
 *  relaunch_probe_rounds() has a defined starting point. */
#define RELAUNCH_PROBE_ROUNDS 64u

/** How many broadcast+probe cycles experiment A runs by default. More than one because a
 *  single clean probe cannot distinguish "the broadcast never damages a core" from "this
 *  one did not"; overridable with FT_TT_MULTICAST_RELAUNCH_CYCLES. */
#define DEFAULT_RELAUNCH_CYCLES 2u

/** Offsets from the derived kernel-config base for the three data buffers.
 *
 * NOT the Phase 2 layout's source0/source1/destination: those are 0x2000 apart, which is
 * fine for one 2048-byte tile and silently overlaps at any block bigger than 8 KiB. The
 * point of this program is to sweep the block size up to a whole depth-48 A block, so the
 * buffers get 256 KiB apiece. Everything here is above the layout's own regions and below
 * L1's 1536 KiB end.
 */
#define STAGING_A_OFFSET 0x040000u
#define STAGING_B_OFFSET 0x080000u
#define OPERAND_OFFSET 0x0C0000u
#define MAX_PAYLOAD_BYTES 0x040000u

/** Offsets inside the Phase 2 layout's probe region for the control words. 16 bytes apart
 *  so no two share an L1 line. */
#define STATUS_OFFSET 0x000u
#define READY_SEMAPHORE_OFFSET 0x040u
#define ARRIVAL_SEMAPHORE_OFFSET 0x050u
#define VALID_WORD_OFFSET 0x060u
#define STOP_FLAG_OFFSET 0x070u

/* ------------------------------------------------------------------------- */

/**
 * The DRAM arm exists twice, and the difference between the two is one COORDINATE.
 *
 * ARM_DRAM is what the backend has always done: every reader is handed the bank's own
 * (x, y), which `ft_tt_coord_dram_bank()` makes PORT 0 — so N concurrent readers queue at
 * ONE endpoint and share its ~45 GB/s. ARM_DRAM_PORTS hands reader `r` door `r % 3` of the
 * SAME bank at the SAME address. **No bytes move and no kernel changes**: three ports of a
 * bank are three doors into one memory (ft_tt_dram_stripe.h), so the pair of runtime args
 * the reader already carries is the whole difference, and what it buys is the bank's ~64
 * instead of one door's ~45.
 *
 * Both are kept rather than one replacing the other, because the recorded crossover
 * ("multicast wins from N=3 at 96 KiB") was measured against the port-0 arm and comparing a
 * new number against a retired baseline would silently restate the conclusion instead of
 * revising it.
 *
 * ARM_DRAM_WRONG_BANK is the RED arm and is never timed: same rotation, same addresses,
 * one bank over — where the payload was never written. It must FAIL the payload check.
 */
typedef enum bench_arm {
    ARM_MULTICAST = 0,
    ARM_DRAM = 1,
    ARM_DRAM_PORTS = 2,
    ARM_DRAM_WRONG_BANK = 3
} bench_arm;

/**
 * Which arms stage 3 runs. `both` is the measurement; the two single-arm settings exist as
 * CONTROLS, because "the suite left a core unbootable" is a claim about one of two arms and
 * a suite that always runs both cannot say which. FT_TT_MULTICAST_ARMS selects it.
 *
 * A control must run on cores the suspect has never touched — that is the
 * caller's job, via FT_TT_MULTICAST_FIRST_INDEX and FT_TT_MULTICAST_SENDER_INDEX.
 */
typedef enum bench_arm_selection {
    ARMS_BOTH = 0,
    ARMS_MULTICAST_ONLY = 1,
    ARMS_DRAM_ONLY = 2
} bench_arm_selection;

/**
 * Which of the three stage-3 bodies this process runs. ONE PER PROCESS, always: the two
 * experiments exist to attribute a wedge to a specific thing, and a process that ran the
 * timed sweep first would have done 1x1..2x5 broadcasts on the same cores before the
 * experiment started.
 */
typedef enum bench_experiment {
    EXPERIMENT_NONE = 0,      /**< the timed sweep — the measurement */
    EXPERIMENT_RELAUNCH = 1,  /**< A: broadcast, then re-launch with NO re-boot */
    EXPERIMENT_SPLIT = 2      /**< B: N=8 as two disjoint 4-destination broadcasts */
} bench_experiment;

typedef struct bench_fixture {
    ft_tt_hardware_session *session;
    ft_tt_noc_path *dram_path;
    const ft_tt_firmware_set *firmware;
    const ft_tt_program_directory *program;
    ft_tt_noc_coord dram;
    /** Every door of DRAM_LOGICAL_BANK, from live telemetry — what ARM_DRAM_PORTS rotates
     *  over. `port[0].coord` is `dram` above; the rest are the doors nothing in this backend
     *  has ever used. */
    ft_tt_dram_port_set dram_ports;
    /** Every door of the NEXT bank, for the red arm. A live endpoint holding the wrong
     *  bytes, so a reader aimed at it completes and returns a payload that must NOT match —
     *  a coordinate that is merely invalid would hang the chip instead. */
    ft_tt_dram_port_set wrong_bank_ports;
    /** members[0] is the sender; members[1..] are the receiver pool, row-major inside the
     *  rectangle. */
    ft_tt_noc_coord members[MAX_CORES];
    /** The LIVE TENSIX INDEX behind each member, parallel to `members`. Recovery is driven
     *  by live index (`ft_tt_core status/probe/recover <index>`) and the health stages
     *  report translated (x,y), so a wedged receiver otherwise has to be mapped back by
     *  hand — which is a transcription step in the middle of an incident. */
    uint32_t live_index[MAX_CORES];
    uint32_t member_count;
    uint32_t first_index;
    uint32_t sender_index;
    /** The `split` experiment's second sender, and the slot it occupies — the LAST slot,
     *  after the pool. Both are zero when the experiment is not armed, and `sender_b_slot`
     *  being zero is what every other stage reads as "there is no second sender". */
    uint32_t sender_b_index;
    uint32_t sender_b_slot;
    ft_tt_phase2_l1_layout layout;
    ft_tt_launch_config config[MAX_CORES];
    uint32_t iterations;
    uint32_t payload_bytes;
    /** Cycles per second, measured in stage 1. Zero until then. */
    double cycles_per_second;
    bench_arm_selection arms;
    bench_experiment experiment;
    /** How many broadcast+probe cycles experiment A runs. */
    uint32_t relaunch_cycles;
    /** The receiver pool's shape. Defaults to POOL_ROWS x POOL_COLUMNS; overridable so a
     *  control can be given a block of cores the suspect has never touched. */
    uint32_t pool_rows;
    uint32_t pool_columns;
} bench_fixture;

/** One core's status block, as read back. */
typedef struct core_status {
    uint32_t word[STATUS_WORDS];
} core_status;

static uint8_t g_block_a[MAX_PAYLOAD_BYTES];
static uint8_t g_block_b[MAX_PAYLOAD_BYTES];
static uint8_t g_readback[MAX_PAYLOAD_BYTES];

/* ------------------------------------------------------------------------- */
/* Small helpers                                                             */
/* ------------------------------------------------------------------------- */

static uint32_t environment_u32(const char *name, uint32_t fallback)
{
    const char *text = getenv(name);
    char *end = NULL;
    unsigned long value;

    if (text == NULL || *text == '\0')
        return fallback;
    value = strtoul(text, &end, 0);
    if (end == NULL || *end != '\0')
        return fallback;
    return (uint32_t)value;
}

static double monotonic_seconds(void)
{
    struct timespec now;

    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static int compare_uint32(const void *left, const void *right)
{
    uint32_t a = *(const uint32_t *)left;
    uint32_t b = *(const uint32_t *)right;

    return (a > b) - (a < b);
}

static uint32_t median_of(uint32_t *values, uint32_t count)
{
    qsort(values, count, sizeof(values[0]), compare_uint32);
    return values[count / 2u];
}

/** Fill a block so that no two bytes of the two blocks are confusable and the low mantissa
 *  bits are populated — small integers cannot tell two layouts apart. */
static void fill_block(uint8_t *block, uint32_t bytes, uint32_t seed)
{
    uint32_t state = seed * 2654435761u + 12345u;
    uint32_t index;

    for (index = 0u; index < bytes; index++) {
        state = state * 1103515245u + 12345u;
        block[index] = (uint8_t)(state >> 17);
    }
}

static bool blocks_match(const uint8_t *left, const uint8_t *right, uint32_t bytes,
                         uint32_t *out_index)
{
    uint32_t index;

    for (index = 0u; index < bytes; index++) {
        if (left[index] != right[index]) {
            *out_index = index;
            return false;
        }
    }
    *out_index = bytes;
    return true;
}

/* ------------------------------------------------------------------------- */
/* L1 addresses                                                              */
/* ------------------------------------------------------------------------- */

static uint32_t status_address(const bench_fixture *fixture)
{
    return fixture->layout.probe_address + STATUS_OFFSET;
}

static uint32_t staging_address(const bench_fixture *fixture, unsigned which)
{
    return fixture->layout.kernel_config_base +
           (which == 0u ? STAGING_A_OFFSET : STAGING_B_OFFSET);
}

static void fill_layout(const bench_fixture *fixture, ft_tt_group_shared_operand_layout *out)
{
    out->operand_address = fixture->layout.kernel_config_base + OPERAND_OFFSET;
    out->operand_bytes = fixture->payload_bytes;
    out->ready_semaphore_address = fixture->layout.probe_address + READY_SEMAPHORE_OFFSET;
    out->arrival_semaphore_address = fixture->layout.probe_address + ARRIVAL_SEMAPHORE_OFFSET;
    out->valid_word_address = fixture->layout.probe_address + VALID_WORD_OFFSET;
}

/* ------------------------------------------------------------------------- */
/* Boot and launch                                                           */
/* ------------------------------------------------------------------------- */

static void build_config(const ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                         const ft_tt_kernel_program *program,
                         const ft_tt_phase2_l1_layout *layout, ft_tt_launch_config *out_config)
{
    unsigned processor;

    memset(out_config, 0, sizeof(*out_config));
    out_config->kernel_config_base = layout->kernel_config_base;
    out_config->enables = ft_tt_kernel_program_enables(program);
    out_config->brisc_noc_id = 0u;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if (placements[processor].bytes_written == 0u)
            continue;
        out_config->kernel_text_address[processor] = placements[processor].entry_address;
        out_config->kernel_text_bytes[processor] = placements[processor].text_bytes;
        /* Every enabled processor gets a real block, not just DM0: a stale or sentinel
         * block resolves to an address the kernel dereferences and has stalled a core
         * here before. */
        out_config->runtime_args_address[processor] =
            ft_tt_phase2_runtime_args_address(layout, processor);
    }
}

/**
 * Bring one core from unknown to "firmware running, program loaded".
 *
 * SEQUENTIAL by construction — one process, one core at a time. At 12-way and above,
 * firmware boots from INDEPENDENT host processes start returning `device I/O failed` and
 * leave cores mid-handshake; it wedged cores 110 and 111 during the DRAM fan-out.
 * This program is one process and never overlaps a boot.
 */
static int boot_core(bench_fixture *fixture, uint32_t slot)
{
    ft_tt_image_placement firmware_placements[FT_TT_PROC_COUNT];
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    const ft_tt_noc_access *noc = &fixture->session->noc;
    const uint16_t x = (uint16_t)fixture->members[slot].x;
    const uint16_t y = (uint16_t)fixture->members[slot].y;
    int rc;

    rc = ft_tt_core_park(noc, x, y);
    if (rc == FT_TT_OK)
        rc = ft_tt_core_boot_firmware(noc, x, y, fixture->firmware, firmware_placements);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_phase2_l1_layout_derive(firmware_placements, FT_TT_PROC_COUNT, &fixture->layout);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_core_load_program(noc, x, y, &fixture->program->program, placements);
    if (rc != FT_TT_OK)
        return rc;
    build_config(placements, &fixture->program->program, &fixture->layout,
                 &fixture->config[slot]);
    return FT_TT_OK;
}

/**
 * Plant one core's runtime-arg block.
 *
 * @param dram  the DRAM DOOR this core reads through. It is a parameter rather than
 *              `fixture->dram` because the door is the only thing ARM_DRAM_PORTS changes,
 *              and passing it explicitly is what keeps "which arm" out of this function.
 *              Ignored by the multicast roles, which read no DRAM.
 * @param sender  the core a RECEIVER announces itself to, and the core a SENDER believes
 *              itself to be. A parameter for the same reason `dram` is: the `split`
 *              experiment runs two senders at once, and a hard-wired `members[0]` would
 *              have made the one-sender shape the only expressible one — which is exactly
 *              the mistake ft_tt_group_shared_operand_bind() takes its leader separately
 *              to avoid.
 */
static int plant_arguments(const bench_fixture *fixture, uint32_t slot, uint32_t role,
                           uint32_t iterations, uint32_t payload_bytes,
                           const ft_tt_multicast_rectangle *rectangle,
                           const ft_tt_group_shared_operand_layout *layout,
                           ft_tt_noc_coord dram, ft_tt_noc_coord sender)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    const uint16_t x = (uint16_t)fixture->members[slot].x;
    const uint16_t y = (uint16_t)fixture->members[slot].y;
    uint32_t arguments[KERNEL_ARG_COUNT];
    uint32_t zero = 0u;
    unsigned processor;
    int rc;

    arguments[0] = role;
    arguments[1] = iterations;
    arguments[2] = payload_bytes;
    arguments[3] = staging_address(fixture, 0u);
    arguments[4] = staging_address(fixture, 1u);
    arguments[5] = layout->operand_address;
    arguments[6] = rectangle->x_low;
    arguments[7] = rectangle->y_low;
    arguments[8] = rectangle->x_high;
    arguments[9] = rectangle->y_high;
    arguments[10] = rectangle->destination_count;
    arguments[11] = sender.x;
    arguments[12] = sender.y;
    arguments[13] = layout->ready_semaphore_address;
    arguments[14] = layout->arrival_semaphore_address;
    arguments[15] = layout->valid_word_address;
    arguments[16] = dram.x;
    arguments[17] = dram.y;
    arguments[18] = (uint32_t)DRAM_BLOCK_A_ADDRESS;
    arguments[19] = (uint32_t)DRAM_BLOCK_B_ADDRESS;
    arguments[20] = status_address(fixture);
    arguments[21] = fixture->layout.probe_address + STOP_FLAG_OFFSET;
    arguments[22] = SPIN_LIMIT;

    rc = ft_tt_launch_plant_runtime_args(
        noc, x, y, fixture->config[slot].runtime_args_address[FT_TT_PROC_DM0], arguments,
        KERNEL_ARG_COUNT);
    for (processor = FT_TT_PROC_MATH0; processor < FT_TT_PROC_COUNT && rc == FT_TT_OK;
         processor++) {
        if (fixture->config[slot].runtime_args_address[processor] == 0u)
            continue;
        rc = ft_tt_launch_plant_runtime_args(
            noc, x, y, fixture->config[slot].runtime_args_address[processor], &zero, 1u);
    }
    return rc;
}

/** Poison the status block and the stop flag, so "never ran" is its own bit pattern. */
static int poison_control(const bench_fixture *fixture, uint32_t slot)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    const uint16_t x = (uint16_t)fixture->members[slot].x;
    const uint16_t y = (uint16_t)fixture->members[slot].y;
    uint32_t poison[STATUS_WORDS];
    uint32_t zero = 0u;
    unsigned index;
    int rc;

    for (index = 0u; index < STATUS_WORDS; index++)
        poison[index] = STATUS_POISON;
    rc = noc->write(noc->user, x, y, status_address(fixture), poison, sizeof(poison));
    if (rc == FT_TT_OK)
        rc = noc->write(noc->user, x, y, fixture->layout.probe_address + STOP_FLAG_OFFSET,
                        &zero, sizeof(zero));
    return rc;
}

static int read_status(const bench_fixture *fixture, uint32_t slot, core_status *out)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;

    return noc->read(noc->user, (uint16_t)fixture->members[slot].x,
                     (uint16_t)fixture->members[slot].y, status_address(fixture), out->word,
                     sizeof(out->word));
}

/**
 * Launch `count` cores and wait for all of them.
 *
 * Receivers go first and the sender last, so the sender's round 0 does not spend its whole
 * bound waiting for cores that have not been given a go message yet. Correctness does not
 * depend on it — every wait is bounded and askable — but the untimed round 0 gets shorter.
 */
static int launch_and_wait(const bench_fixture *fixture, const uint32_t *slots, uint32_t count)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    uint32_t index;
    int rc = FT_TT_OK;

    for (index = 0u; index < count && rc == FT_TT_OK; index++) {
        uint32_t slot = slots[count - 1u - index];

        rc = ft_tt_launch_write_launch_message(noc, (uint16_t)fixture->members[slot].x,
                                               (uint16_t)fixture->members[slot].y,
                                               &fixture->config[slot]);
    }
    for (index = 0u; index < count && rc == FT_TT_OK; index++) {
        uint32_t slot = slots[count - 1u - index];

        rc = ft_tt_launch_send_go(noc, (uint16_t)fixture->members[slot].x,
                                  (uint16_t)fixture->members[slot].y);
    }
    for (index = 0u; index < count && rc == FT_TT_OK; index++) {
        rc = ft_tt_launch_wait_done(noc, (uint16_t)fixture->members[slots[index]].x,
                                    (uint16_t)fixture->members[slots[index]].y, RUN_TIMEOUT_MS,
                                    NULL);
    }
    return rc;
}

/* ------------------------------------------------------------------------- */
/* One arm of one run                                                        */
/* ------------------------------------------------------------------------- */

/**
 * Which DRAM door reader `index` of a concurrent group reads through, for this arm.
 *
 * Round-robin over the bank's doors rather than anything cleverer, because the measured rule
 * is BALANCE — the outstanding-read budget is per ENDPOINT, and an unbalanced map cost 30%
 * and was worse than either neighbouring balanced reader count (plan Phase 6.7). Modulo
 * differs by at most one reader between doors at every group size, which is why
 * ft_tt_dram_stripe.h states it as the rule.
 *
 * The NoC ROW rule does not bind here and is deliberately not consulted: every door belongs
 * to ONE bank, so all three sit in one NoC column, and two endpoints in one column do not
 * interact at all. Row collisions are a property of striping across BANKS, which this bench
 * does not do — its whole design is N readers against ONE block in ONE bank.
 */
static ft_tt_noc_coord reader_door(const bench_fixture *fixture, bench_arm arm, uint32_t index)
{
    const ft_tt_dram_endpoint *endpoint = NULL;

    if (arm == ARM_DRAM_PORTS)
        endpoint = ft_tt_dram_port_set_reader_endpoint(&fixture->dram_ports, index);
    else if (arm == ARM_DRAM_WRONG_BANK)
        endpoint = ft_tt_dram_port_set_reader_endpoint(&fixture->wrong_bank_ports, index);
    return endpoint != NULL ? endpoint->coord : fixture->dram;
}

/**
 * What one arm of one run produced.
 *
 * `first_error` folds three different failures into one word for the sweep's one-line
 * note, and the three fields below it UNFOLD them again. Both are wanted: the sweep is
 * read as a table and needs one column, while the re-launch gate needs to say WHICH ground
 * failed — a check with several sufficient grounds passes for the right reason and fails
 * for the wrong one otherwise.
 */
typedef struct arm_result {
    int rc;                       /**< host-side launch result */
    uint32_t worst_cycles;        /**< the slowest receiver's elapsed cycles */
    uint32_t sender_cycles;       /**< multicast arm only; 0 otherwise */
    uint32_t first_error;         /**< first non-zero kernel error code */
    bool payload_correct;         /**< every receiver holds the expected block */
    bool payload_rejected_other;  /**< and does NOT hold the other one */
    double host_seconds;          /**< wall time around launch+wait, for the calibration */
    bool every_status_magic_present; /**< no status block still held the host poison */
    bool rounds_complete;         /**< every core reported iterations + 1 rounds */
    uint32_t kernel_error;        /**< first non-zero STATUS_ERROR, unfolded */
} arm_result;

/**
 * Stage the L1 of every participating core, launch the arm, and read everything back.
 *
 * `receiver_count` is N: the number of cores that must end up holding the block. The
 * multicast arm uses slot 0 as the sender on top of those N; the DRAM arm does not use
 * slot 0 at all.
 */
static int run_arm(bench_fixture *fixture, bench_arm arm, const uint32_t *receiver_slots,
                   uint32_t receiver_count, uint32_t iterations, arm_result *out)
{
    ft_tt_noc_coord receivers[MAX_CORES] = {{0u, 0u}};
    ft_tt_group_shared_operand_layout layout;
    ft_tt_group_shared_operand operand;
    ft_tt_multicast_rectangle rectangle;
    const ft_tt_noc_access *noc = &fixture->session->noc;
    uint32_t slots[MAX_CORES];
    uint32_t slot_count = 0u;
    uint32_t index;
    uint32_t difference = 0u;
    const uint8_t *expected;
    const uint8_t *rejected;
    double started;
    int rc;

    memset(out, 0, sizeof(*out));
    out->payload_correct = true;
    out->payload_rejected_other = true;
    out->every_status_magic_present = true;
    out->rounds_complete = true;

    fill_layout(fixture, &layout);

    for (index = 0u; index < receiver_count; index++)
        receivers[index] = fixture->members[receiver_slots[index]];

    /* The rectangle is ALWAYS exactly the receivers — the same rectangle in both arms, so
     * the arms differ in the role word and nothing else. The sender is outside it, so
     * destination_count == N. */
    rc = ft_tt_multicast_rectangle_from_cores(receivers, receiver_count, fixture->members[0],
                                              &rectangle);
    if (rc != FT_TT_OK)
        return rc;

    /* Bind the receivers and the leader so the layout checks run against the real
     * addresses. The leader is OUTSIDE the rectangle here — a dedicated feeder — which is
     * what lets both arms run on exactly the same set of cores. The reset below is what
     * puts the barrier into its start state. */
    rc = ft_tt_group_shared_operand_bind(receivers, receiver_count, fixture->members[0],
                                         &layout, &operand);
    if (rc != FT_TT_OK)
        return rc;

    /* Poison every destination, and the sender's staging buffers, so a green result cannot
     * be stale L1 and "the DRAM read never happened" is visible. */
    memset(g_readback, DESTINATION_POISON_BYTE, fixture->payload_bytes);
    for (index = 0u; index < receiver_count; index++) {
        rc = noc->write(noc->user, (uint16_t)receivers[index].x, (uint16_t)receivers[index].y,
                        layout.operand_address, g_readback, fixture->payload_bytes);
        if (rc != FT_TT_OK)
            return rc;
    }
    memset(g_readback, STAGING_POISON_BYTE, fixture->payload_bytes);
    rc = noc->write(noc->user, (uint16_t)fixture->members[0].x,
                    (uint16_t)fixture->members[0].y, staging_address(fixture, 0u), g_readback,
                    fixture->payload_bytes);
    if (rc == FT_TT_OK)
        rc = noc->write(noc->user, (uint16_t)fixture->members[0].x,
                        (uint16_t)fixture->members[0].y, staging_address(fixture, 1u),
                        g_readback, fixture->payload_bytes);
    if (rc != FT_TT_OK)
        return rc;

    rc = ft_tt_group_shared_operand_reset(noc, &operand, receivers, receiver_count);
    if (rc != FT_TT_OK)
        return rc;

    /* Roles. Slot 0 participates only in the multicast arm. */
    if (arm == ARM_MULTICAST) {
        slots[slot_count++] = 0u;
        rc = poison_control(fixture, 0u);
        if (rc == FT_TT_OK)
            rc = plant_arguments(fixture, 0u, ROLE_MULTICAST_SENDER, iterations,
                                 fixture->payload_bytes, &rectangle, &layout, fixture->dram,
                                 fixture->members[0]);
        if (rc != FT_TT_OK)
            return rc;
    }
    for (index = 0u; index < receiver_count; index++) {
        slots[slot_count++] = receiver_slots[index];
        rc = poison_control(fixture, receiver_slots[index]);
        if (rc == FT_TT_OK)
            rc = plant_arguments(fixture, receiver_slots[index],
                                 arm == ARM_MULTICAST ? ROLE_MULTICAST_RECEIVER
                                                      : ROLE_DRAM_READER,
                                 iterations, fixture->payload_bytes, &rectangle, &layout,
                                 reader_door(fixture, arm, index), fixture->members[0]);
        if (rc != FT_TT_OK)
            return rc;
    }

    started = monotonic_seconds();
    out->rc = launch_and_wait(fixture, slots, slot_count);
    out->host_seconds = monotonic_seconds() - started;
    if (out->rc != FT_TT_OK)
        return out->rc;

    /* Results. A status block still reading the poison means the kernel never ran. */
    for (index = 0u; index < slot_count; index++) {
        core_status status;
        uint32_t slot = slots[index];

        rc = read_status(fixture, slot, &status);
        if (rc != FT_TT_OK)
            return rc;
        if (status.word[STATUS_MAGIC_WORD] != STATUS_MAGIC) {
            out->first_error = status.word[STATUS_MAGIC_WORD];
            out->every_status_magic_present = false;
            out->payload_correct = false;
            return FT_TT_ERROR;
        }
        if (status.word[STATUS_ERROR] != 0u && out->kernel_error == 0u)
            out->kernel_error = status.word[STATUS_ERROR];
        if (status.word[STATUS_ERROR] != 0u && out->first_error == 0u)
            out->first_error = status.word[STATUS_ERROR];
        if (status.word[STATUS_ROUNDS_COMPLETED] != iterations + 1u)
            out->rounds_complete = false;
        if (status.word[STATUS_ROUNDS_COMPLETED] != iterations + 1u && out->first_error == 0u)
            out->first_error = 0xDEAD0000u | status.word[STATUS_ROUNDS_COMPLETED];
        if (slot == 0u)
            out->sender_cycles = status.word[STATUS_ELAPSED_CYCLES];
        else if (status.word[STATUS_ELAPSED_CYCLES] > out->worst_cycles)
            out->worst_cycles = status.word[STATUS_ELAPSED_CYCLES];
    }

    /* The payload must be the block for the LAST round, and must not be the other one.
     * That is what shows the data tracked the source rather than merely arriving. */
    expected = ((iterations & 1u) != 0u) ? g_block_b : g_block_a;
    rejected = ((iterations & 1u) != 0u) ? g_block_a : g_block_b;
    for (index = 0u; index < receiver_count; index++) {
        rc = noc->read(noc->user, (uint16_t)receivers[index].x, (uint16_t)receivers[index].y,
                       layout.operand_address, g_readback, fixture->payload_bytes);
        if (rc != FT_TT_OK)
            return rc;
        if (!blocks_match(expected, g_readback, fixture->payload_bytes, &difference))
            out->payload_correct = false;
        if (blocks_match(rejected, g_readback, fixture->payload_bytes, &difference))
            out->payload_rejected_other = false;
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* Stages                                                                    */
/* ------------------------------------------------------------------------- */

/**
 * Stage 0 — the transfer path, before anything is scored through it.
 *
 * A host->DRAM->host round trip with the exact payload kind the arms use, so "the transfer
 * path is fine" is a line in the report instead of an assumption.
 */
static void run_stage_0_transfer_round_trip(bench_fixture *fixture)
{
    uint32_t difference = 0u;
    int rc;

    ft_tt_test_step("stage 0 — host->DRAM->host round trip on bank %u, %u bytes",
                    DRAM_LOGICAL_BANK, fixture->payload_bytes);

    fill_block(g_block_a, fixture->payload_bytes, 0x5EEDA11Fu);
    fill_block(g_block_b, fixture->payload_bytes, 0x0B10C1B2u);

    rc = ft_tt_write_dram(fixture->dram_path, fixture->dram, DRAM_BLOCK_A_ADDRESS, g_block_a,
                          fixture->payload_bytes);
    if (rc == FT_TT_OK)
        rc = ft_tt_write_dram(fixture->dram_path, fixture->dram, DRAM_BLOCK_B_ADDRESS,
                              g_block_b, fixture->payload_bytes);
    ft_tt_test_check_fmt("both blocks staged in DRAM", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    memset(g_readback, DESTINATION_POISON_BYTE, fixture->payload_bytes);
    rc = ft_tt_read_dram(fixture->dram_path, fixture->dram, DRAM_BLOCK_A_ADDRESS, g_readback,
                         fixture->payload_bytes);
    ft_tt_test_check_fmt("block A reads back byte for byte",
                         rc == FT_TT_OK &&
                             blocks_match(g_block_a, g_readback, fixture->payload_bytes,
                                          &difference),
                         "first difference at %u of %u", difference, fixture->payload_bytes);

    memset(g_readback, DESTINATION_POISON_BYTE, fixture->payload_bytes);
    rc = ft_tt_read_dram(fixture->dram_path, fixture->dram, DRAM_BLOCK_B_ADDRESS, g_readback,
                         fixture->payload_bytes);
    ft_tt_test_check_fmt("block B reads back byte for byte, and differs from A",
                         rc == FT_TT_OK &&
                             blocks_match(g_block_b, g_readback, fixture->payload_bytes,
                                          &difference) &&
                             !blocks_match(g_block_a, g_block_b, fixture->payload_bytes,
                                           &difference),
                         "first difference at %u of %u", difference, fixture->payload_bytes);
}

/**
 * Stage 1 — calibrate the device wall clock against the host's, from TWO run lengths.
 *
 * A single run cannot do it: the launch, the boot handshake and the untimed round 0 are all
 * inside the host's measurement and none of them is inside the device's. Two lengths cancel
 * every fixed cost on both sides, and the ratio of the two differences is the clock.
 *
 * This exists because "the card runs at 800 MHz" is a MEASURED busy AICLK, not a statement
 * about what RISCV_DEBUG_REG_WALL_CLOCK_0 counts, and reporting microseconds derived from
 * an assumed rate would make every number in this file inherit it.
 */
static void run_stage_1_clock_calibration(bench_fixture *fixture)
{
    const uint32_t one_receiver = 1u;
    const uint32_t short_iterations = fixture->iterations;
    const uint32_t long_iterations = fixture->iterations * 4u;
    arm_result shorter;
    arm_result longer;
    double device_cycles;
    double host_seconds;
    int rc;

    ft_tt_test_step("stage 1 — calibrate the device wall clock from %u vs %u rounds on ONE "
                    "DRAM reader", short_iterations, long_iterations);

    rc = run_arm(fixture, ARM_DRAM, &one_receiver, 1u, short_iterations, &shorter);
    if (rc == FT_TT_OK)
        rc = run_arm(fixture, ARM_DRAM, &one_receiver, 1u, long_iterations, &longer);
    ft_tt_test_check_fmt("both calibration runs completed", rc == FT_TT_OK &&
                             shorter.first_error == 0u && longer.first_error == 0u,
                         "rc=%d (%s), errors %u / %u", rc, ft_tt_error_string(rc),
                         shorter.first_error, longer.first_error);
    if (rc != FT_TT_OK || shorter.first_error != 0u || longer.first_error != 0u)
        return;

    device_cycles = (double)longer.worst_cycles - (double)shorter.worst_cycles;
    host_seconds = longer.host_seconds - shorter.host_seconds;
    ft_tt_test_check_fmt("the two lengths differ measurably on both clocks",
                         device_cycles > 0.0 && host_seconds > 0.0,
                         "%.0f cycles over %.6f s", device_cycles, host_seconds);
    if (device_cycles <= 0.0 || host_seconds <= 0.0)
        return;

    fixture->cycles_per_second = device_cycles / host_seconds;
    /* The band spans BOTH clocks on purpose. 800 MHz was never a property of the card —
     * tt-kmd initialises a legacy fd to request the MINIMUM AI clock unless it is opened
     * with O_APPEND, and foundry opened with flags 0, so we were asking for it. Max became
     * the default on 2026-08-15 and reads 1350; `FT_TT_AI_CLK=min` still reproduces 800, so
     * this harness has to accept either rather than failing the arm nobody selected. Pinned
     * at 640-960, it failed every arm at 1349 MHz on 2026-08-16, control included. */
    ft_tt_test_check_fmt("the wall clock runs within 20%% of one of this card's two AICLK "
                         "settings (800 min / 1350 max)",
                         (fixture->cycles_per_second > 640e6 &&
                          fixture->cycles_per_second < 960e6) ||
                             (fixture->cycles_per_second > 1080e6 &&
                              fixture->cycles_per_second < 1620e6),
                         "measured %.1f MHz (%.0f cycles / %.6f s)",
                         fixture->cycles_per_second / 1e6, device_cycles, host_seconds);
    /* Fall back to the measured busy clock if the calibration is out of band, so the
     * microsecond column is still printed and clearly labelled. */
    if (fixture->cycles_per_second <= 640e6 || fixture->cycles_per_second >= 960e6)
        fixture->cycles_per_second = 800e6;
}

/**
 * Stage 2 — the RED HALF. Two runs that must be REFUSED by the kernel at the point of
 * issue, and must leave the core healthy.
 *
 * Both are the zero-length NoC transfer, which is the shape that took this card to
 * `Mem- BusMaster-` on 2026-08-12 with no error anywhere until the next read returned
 * 0xFFFFFFFF. If either of these came back FTMC_OK the guards in
 * kernels/ft_tt_multicast_kernel.h would be decoration and the primitive would be one
 * arithmetic slip away from costing the card.
 *
 * They are safe to run for exactly one reason: the refusal is BEFORE the register writes,
 * so nothing is ever issued. A test that had to issue one to find out would not be run.
 */
static void run_stage_2_refusals(bench_fixture *fixture)
{
    ft_tt_group_shared_operand_layout layout;
    ft_tt_multicast_rectangle rectangle;
    core_status status;
    uint32_t slot = 0u;
    int rc;

    ft_tt_test_step("stage 2 — the two hazard refusals, on silicon");

    fill_layout(fixture, &layout);
    rc = ft_tt_multicast_rectangle_from_cores(&fixture->members[1], 1u, fixture->members[0],
                                              &rectangle);
    (void)rectangle;
    if (rc != FT_TT_OK) {
        ft_tt_test_check("a one-receiver rectangle built", false, ft_tt_error_string(rc));
        return;
    }

    /* (a) zero length, on a single DRAM reader — no rendezvous, so nothing waits. */
    rc = poison_control(fixture, slot);
    if (rc == FT_TT_OK)
        rc = plant_arguments(fixture, slot, ROLE_DRAM_READER, 1u, 0u, &rectangle, &layout,
                             fixture->dram, fixture->members[0]);
    if (rc == FT_TT_OK)
        rc = launch_and_wait(fixture, &slot, 1u);
    if (rc == FT_TT_OK)
        rc = read_status(fixture, slot, &status);
    ft_tt_test_check_fmt("a ZERO-LENGTH payload is refused in the kernel, not issued",
                         rc == FT_TT_OK && status.word[STATUS_MAGIC_WORD] == STATUS_MAGIC &&
                             (int32_t)status.word[STATUS_LAST_ISSUE_CODE] ==
                                 FTMC_ERROR_ZERO_LENGTH,
                         "rc=%d, magic 0x%08x, issue code %d (wanted %d), rounds %u",
                         rc, status.word[STATUS_MAGIC_WORD],
                         (int)(int32_t)status.word[STATUS_LAST_ISSUE_CODE],
                         FTMC_ERROR_ZERO_LENGTH, status.word[STATUS_ROUNDS_COMPLETED]);

    /* (b) zero destinations, on the sender alone. The rectangle is handed to the kernel
     * with destination_count forced to 0, which is what a rectangle that degenerated onto
     * the sender would produce. No receivers are launched and none are needed: the sender's
     * ready-semaphore wait is satisfied immediately by a target of zero, and the refusal
     * happens at the multicast. */
    {
        ft_tt_multicast_rectangle degenerate = rectangle;

        degenerate.destination_count = 0u;
        rc = poison_control(fixture, slot);
        if (rc == FT_TT_OK)
            rc = plant_arguments(fixture, slot, ROLE_MULTICAST_SENDER, 1u,
                                 fixture->payload_bytes, &degenerate, &layout, fixture->dram,
                                 fixture->members[0]);
        if (rc == FT_TT_OK)
            rc = launch_and_wait(fixture, &slot, 1u);
        if (rc == FT_TT_OK)
            rc = read_status(fixture, slot, &status);
        ft_tt_test_check_fmt("a ZERO-DESTINATION rectangle is refused in the kernel",
                             rc == FT_TT_OK &&
                                 status.word[STATUS_MAGIC_WORD] == STATUS_MAGIC &&
                                 (int32_t)status.word[STATUS_LAST_ISSUE_CODE] ==
                                     FTMC_ERROR_ZERO_DESTINATIONS,
                             "rc=%d, magic 0x%08x, issue code %d (wanted %d)", rc,
                             status.word[STATUS_MAGIC_WORD],
                             (int)(int32_t)status.word[STATUS_LAST_ISSUE_CODE],
                             FTMC_ERROR_ZERO_DESTINATIONS);
    }

    /* And the host side refuses the same two before they can ever be planted. */
    ft_tt_test_check("the HOST refuses a zero-length payload too",
                     ft_tt_multicast_check_payload(layout.operand_address, 0u) != FT_TT_OK,
                     "ft_tt_multicast_check_payload(addr, 0)");
}

/**
 * Stage 2b — the RED PROOF for the port rotation, on silicon.
 *
 * The rotation is an ADDRESSING change and its whole claim is that a door is a door into the
 * same memory — so the one way it can be wrong is by addressing the wrong MEMORY. This runs
 * the identical rotation one BANK over, where a deliberate poison pattern sits at the two
 * block addresses instead of the payload, and requires the payload check to FAIL.
 *
 * Without it the green arm proves nothing: a rotation that quietly fell back to port 0 on
 * every reader would also return the right bytes, also complete, and also be reported green.
 * The endpoints are live and derived from telemetry either way, so this costs a run and not
 * a core; a coordinate that was merely INVALID would hang the chip and is
 * never issued anywhere in this file.
 */
static void run_stage_2b_wrong_bank_red(bench_fixture *fixture)
{
    const uint32_t receiver_slots[2] = {1u, 2u};
    const uint32_t receivers = fixture->member_count > 2u ? 2u : 1u;
    arm_result red;
    int rc;

    ft_tt_test_step("stage 2b — the port rotation aimed one BANK over must NOT match");

    /* Deterministic rather than "whatever was in that bank": the block addresses in the
     * wrong bank are filled with a pattern that is not either payload, so a green result
     * here cannot be a coincidence of uninitialised DRAM. */
    memset(g_readback, WRONG_BANK_POISON_BYTE, fixture->payload_bytes);
    rc = ft_tt_write_dram(fixture->dram_path, fixture->wrong_bank_ports.port[0].coord,
                          DRAM_BLOCK_A_ADDRESS, g_readback, fixture->payload_bytes);
    if (rc == FT_TT_OK)
        rc = ft_tt_write_dram(fixture->dram_path, fixture->wrong_bank_ports.port[0].coord,
                              DRAM_BLOCK_B_ADDRESS, g_readback, fixture->payload_bytes);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("the wrong bank could be poisoned", false, ft_tt_error_string(rc));
        return;
    }

    memset(&red, 0, sizeof(red));
    rc = run_arm(fixture, ARM_DRAM_WRONG_BANK, receiver_slots, receivers, RED_ITERATIONS,
                 &red);
    ft_tt_test_check_fmt(
        "a reader given a live door on the WRONG BANK returns the wrong bytes",
        rc == FT_TT_OK && red.first_error == 0u && !red.payload_correct,
        "N=%u: rc=%d (%s), kernel error 0x%08x, payload %s — a green comparison here would "
        "mean the rotated arm's green cannot fail",
        receivers, rc, ft_tt_error_string(rc), red.first_error,
        red.payload_correct ? "MATCHED (the rotation is not being applied)" : "differs (correct)");
}

/**
 * Stage 3 — the sweep. This is the measurement.
 */
/** One rectangle shape in the sweep. Rows x columns of the receiver pool, so N = rows *
 *  columns and the same N can appear at two different SHAPES — which is itself a question
 *  worth answering, because a 2x2 rectangle and a 1x4 line reach the same four cores over
 *  different NoC paths. */
typedef struct sweep_shape {
    uint32_t rows;
    uint32_t columns;
} sweep_shape;

static void run_stage_3_sweep(bench_fixture *fixture)
{
    static const sweep_shape default_sweep[] = {{1u, 1u}, {1u, 2u}, {1u, 3u}, {1u, 4u},
                                                {1u, 5u}, {2u, 2u}, {2u, 3u}, {2u, 4u},
                                                {2u, 5u}};
    /* ONE SHAPE PER PROCESS, so a wedge is attributable to a SHAPE rather than to a ladder.
     * The default table runs every shape that fits the pool, so a 2x3 pool also exercises
     * 1x1..1x3 and 2x2 first — and "is it receiver COUNT or rectangle WIDTH" cannot be
     * answered by a run that touched five shapes on the same cores. FT_TT_MULTICAST_SHAPE
     * replaces the table with exactly one rectangle; set the pool to the same shape so the
     * cores booted are exactly the cores the shape uses. */
    sweep_shape selected[1];
    const sweep_shape *sweep = default_sweep;
    uint32_t sweep_count = (uint32_t)(sizeof(default_sweep) / sizeof(default_sweep[0]));
    const char *shape_text = getenv("FT_TT_MULTICAST_SHAPE");
    uint32_t entry;

    if (shape_text != NULL && shape_text[0] != '\0') {
        unsigned rows = 0u;
        unsigned columns = 0u;

        if (sscanf(shape_text, "%ux%u", &rows, &columns) != 2 || rows == 0u || columns == 0u) {
            ft_tt_test_check_fmt("FT_TT_MULTICAST_SHAPE parses as RxC", false, "got \"%s\"",
                                 shape_text);
            return;
        }
        selected[0].rows = (uint32_t)rows;
        selected[0].columns = (uint32_t)columns;
        sweep = selected;
        sweep_count = 1u;
    }

    ft_tt_test_step("stage 3 — %u rounds of a %u-byte block, %s, %u repetitions, medians "
                    "reported", fixture->iterations, fixture->payload_bytes,
                    fixture->arms == ARMS_BOTH
                        ? "multicast vs N cores reading DRAM, arms INTERLEAVED"
                        : (fixture->arms == ARMS_MULTICAST_ONLY
                               ? "MULTICAST ARM ONLY (a control, not the measurement)"
                               : "DRAM ARM ONLY (a control, not the measurement)"),
                    REPETITIONS);

    printf("  %3s %5s | %13s | %13s | %13s | %7s | %8s | %13s | %s\n", "N", "shape",
           "mcast ns/blk", "dram 1 door", "dram 3 doors", "doors x", "speedup",
           "sender ns/blk", "notes");
    printf("  ----------+---------------+---------------+---------------+---------+"
           "----------+---------------+------\n");

    for (entry = 0u; entry < sweep_count; entry++) {
        uint32_t receivers = sweep[entry].rows * sweep[entry].columns;
        uint32_t receiver_slots[MAX_CORES];
        uint32_t row;
        uint32_t column;
        uint32_t multicast_cycles[REPETITIONS];
        uint32_t dram_cycles[REPETITIONS];
        uint32_t dram_ports_cycles[REPETITIONS];
        uint32_t sender_cycles[REPETITIONS];
        uint32_t repetition;
        double multicast_ns;
        double dram_ns;
        double dram_ports_ns;
        double sender_ns;
        bool ok = true;
        char note[160];

        if (sweep[entry].rows > fixture->pool_rows ||
            sweep[entry].columns > fixture->pool_columns)
            continue;

        /* The receivers are the top-left rows x columns sub-block of the pool, which is a
         * rectangle in NoC space because the pool is one (the partitioner refused anything
         * that crossed the non-Tensix column gap). Slot 0 is the sender, so the pool starts
         * at slot 1. */
        {
            uint32_t next = 0u;

            for (row = 0u; row < sweep[entry].rows; row++)
                for (column = 0u; column < sweep[entry].columns; column++)
                    receiver_slots[next++] = 1u + row * fixture->pool_columns + column;
        }

        note[0] = '\0';
        for (repetition = 0u; repetition < REPETITIONS; repetition++) {
            arm_result multicast;
            arm_result dram;
            arm_result dram_ports;
            int rc;

            /* Interleaved: the arms of one repetition are adjacent in time, so a host that
             * drifts over the sweep drifts through ALL of them equally. */
            memset(&multicast, 0, sizeof(multicast));
            memset(&dram, 0, sizeof(dram));
            memset(&dram_ports, 0, sizeof(dram_ports));
            rc = FT_TT_OK;
            if (fixture->arms != ARMS_DRAM_ONLY)
                rc = run_arm(fixture, ARM_MULTICAST, receiver_slots, receivers,
                             fixture->iterations, &multicast);
            if (fixture->arms != ARMS_DRAM_ONLY &&
                (rc != FT_TT_OK || multicast.first_error != 0u || !multicast.payload_correct ||
                 !multicast.payload_rejected_other)) {
                (void)snprintf(note, sizeof(note),
                               "multicast arm failed: rc=%d (%s), kernel error 0x%08x, "
                               "payload %s, other-block %s",
                               rc, ft_tt_error_string(rc), multicast.first_error,
                               multicast.payload_correct ? "ok" : "WRONG",
                               multicast.payload_rejected_other ? "rejected" : "MATCHED");
                ok = false;
                break;
            }
            if (fixture->arms != ARMS_MULTICAST_ONLY)
                rc = run_arm(fixture, ARM_DRAM, receiver_slots, receivers, fixture->iterations,
                             &dram);
            if (fixture->arms != ARMS_MULTICAST_ONLY &&
                (rc != FT_TT_OK || dram.first_error != 0u || !dram.payload_correct ||
                 !dram.payload_rejected_other)) {
                (void)snprintf(note, sizeof(note),
                               "dram arm failed: rc=%d (%s), kernel error 0x%08x, payload %s",
                               rc, ft_tt_error_string(rc), dram.first_error,
                               dram.payload_correct ? "ok" : "WRONG");
                ok = false;
                break;
            }
            /* The same N cores, the same block, the same address — reader r through door
             * r % 3 of the same bank instead of all N through port 0. */
            if (fixture->arms != ARMS_MULTICAST_ONLY)
                rc = run_arm(fixture, ARM_DRAM_PORTS, receiver_slots, receivers,
                             fixture->iterations, &dram_ports);
            if (fixture->arms != ARMS_MULTICAST_ONLY &&
                (rc != FT_TT_OK || dram_ports.first_error != 0u ||
                 !dram_ports.payload_correct || !dram_ports.payload_rejected_other)) {
                (void)snprintf(note, sizeof(note),
                               "port-rotated dram arm failed: rc=%d (%s), kernel error "
                               "0x%08x, payload %s",
                               rc, ft_tt_error_string(rc), dram_ports.first_error,
                               dram_ports.payload_correct ? "ok" : "WRONG");
                ok = false;
                break;
            }
            multicast_cycles[repetition] = multicast.worst_cycles;
            sender_cycles[repetition] = multicast.sender_cycles;
            dram_cycles[repetition] = dram.worst_cycles;
            dram_ports_cycles[repetition] = dram_ports.worst_cycles;
        }

        ft_tt_test_check_fmt("every arm correct on every repetition", ok, "N = %u (%ux%u): %s",
                             receivers, sweep[entry].rows, sweep[entry].columns,
                             note[0] != '\0' ? note : "payload tracked the source, "
                                                      "poison gone, other block rejected");
        if (!ok)
            continue;

        multicast_ns = (double)median_of(multicast_cycles, REPETITIONS) /
                       (double)fixture->iterations / fixture->cycles_per_second * 1e9;
        dram_ns = (double)median_of(dram_cycles, REPETITIONS) / (double)fixture->iterations /
                  fixture->cycles_per_second * 1e9;
        dram_ports_ns = (double)median_of(dram_ports_cycles, REPETITIONS) /
                        (double)fixture->iterations / fixture->cycles_per_second * 1e9;
        sender_ns = (double)median_of(sender_cycles, REPETITIONS) /
                    (double)fixture->iterations / fixture->cycles_per_second * 1e9;

        printf("  %3u %2ux%-2u | %13.1f | %13.1f | %13.1f | %6.2fx | %7.2fx | %13.1f | %s\n",
               receivers, sweep[entry].rows, sweep[entry].columns, multicast_ns, dram_ns,
               dram_ports_ns, dram_ports_ns > 0.0 ? dram_ns / dram_ports_ns : 0.0,
               dram_ports_ns > 0.0 ? multicast_ns / dram_ports_ns : 0.0, sender_ns,
               multicast_ns < dram_ports_ns ? "multicast wins" : "DRAM wins");
        fflush(stdout);
    }
    printf("\n  `doors x` is the port rotation ALONE: the same N cores reading the same block\n"
           "  from the same bank at the same address, through three doors instead of one.\n"
           "  No bytes moved and no kernel changed - it is two runtime args.\n"
           "  `speedup` is multicast / the ROTATED DRAM arm, which is the honest comparison\n"
           "  now that the rotation is available to every reader: BELOW 1.00x means a\n"
           "  receiver got its block sooner by being multicast to than by reading for itself.\n"
           "  ns/block is per core per block, device-side, round 0 excluded.\n\n");
}

/* ------------------------------------------------------------------------- */
/* Experiment A — the re-launch probe                                        */
/* ------------------------------------------------------------------------- */

/**
 * How many rounds a re-launch probe runs, chosen so that the block it DEMANDS is not the
 * block the preceding multicast arm left in the destination.
 *
 * The kernel alternates blocks by round parity, so a run of `n` rounds ends holding block
 * `n & 1`. A probe of the same parity would be satisfied by bytes that were already there,
 * and the whole experiment turns on the probe being able to fail. This computes the
 * parity; ft_tt_relaunch_gate_judge() independently REFUSES a probe whose demanded block
 * equals the previous one, so an arithmetic slip here is reported as vacuous rather than
 * quietly turning into a green.
 */
static uint32_t relaunch_probe_rounds(const bench_fixture *fixture)
{
    return RELAUNCH_PROBE_ROUNDS + (((fixture->iterations & 1u) == 0u) ? 1u : 0u);
}

/** Poison every destination and status block in the probe, then plant DRAM-reader args. */
static int relaunch_probe_prepare(const bench_fixture *fixture, const uint32_t *slots,
                                  uint32_t count, bench_arm arm, uint32_t rounds,
                                  const ft_tt_multicast_rectangle *rectangle,
                                  const ft_tt_group_shared_operand_layout *layout)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    uint32_t index;
    int rc = FT_TT_OK;

    memset(g_readback, DESTINATION_POISON_BYTE, fixture->payload_bytes);
    for (index = 0u; index < count && rc == FT_TT_OK; index++) {
        uint32_t slot = slots[index];

        rc = noc->write(noc->user, (uint16_t)fixture->members[slot].x,
                        (uint16_t)fixture->members[slot].y, layout->operand_address,
                        g_readback, fixture->payload_bytes);
        if (rc == FT_TT_OK)
            rc = poison_control(fixture, slot);
        if (rc == FT_TT_OK)
            rc = plant_arguments(fixture, slot, ROLE_DRAM_READER, rounds,
                                 fixture->payload_bytes, rectangle, layout,
                                 reader_door(fixture, arm, index), fixture->members[0]);
    }
    return rc;
}

/**
 * RE-LAUNCH the program that is ALREADY BOOTED on `slots`, with no park and no boot, and
 * judge whether those cores accepted the work.
 *
 * This is the whole of experiment A's instrument. Three things make it able to say no:
 * every status block is host-poisoned first, so "never ran" is its own bit pattern
 * (RUN_MSG_DONE is 0 — DEV_NOTES.md §3); every destination is poisoned first, so a match
 * cannot be stale L1; and the round count is chosen for the OTHER block, so the payload
 * comparison tracks the source rather than merely finding something there. The verdict
 * itself is `ft_tt_relaunch_gate_judge()`, which is a pure function driven red on the Mac
 * by tests/test_relaunch_gate.c, one mutation per ground.
 *
 * The role is DRAM READER, never another broadcast: a re-launched multicast would hang at
 * the rendezvous if any receiver were damaged, and "the group is dead" is not the question.
 * Each core reads for itself and answers for itself.
 */
static ft_tt_relaunch_outcome run_relaunch_probe(bench_fixture *fixture, const uint32_t *slots,
                                                 uint32_t count, bench_arm arm,
                                                 uint32_t rounds, uint32_t previous_block,
                                                 const ft_tt_multicast_rectangle *rectangle,
                                                 const ft_tt_group_shared_operand_layout *layout,
                                                 char *detail, size_t detail_bytes)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    ft_tt_relaunch_evidence evidence;
    ft_tt_relaunch_outcome outcome;
    const uint8_t *expected = ((rounds & 1u) != 0u) ? g_block_b : g_block_a;
    const uint8_t *rejected = ((rounds & 1u) != 0u) ? g_block_a : g_block_b;
    uint32_t difference = 0u;
    /** The first slot that contributed anything negative. Recorded because recovery is
     *  driven by LIVE INDEX and an incident should not begin with a transcription step. */
    uint32_t bad_slot = MAX_CORES;
    uint32_t index;
    int rc;

    memset(&evidence, 0, sizeof(evidence));
    evidence.every_status_magic_present = true;
    evidence.rounds_complete = true;
    evidence.payload_matched_demanded = true;
    evidence.payload_rejected_other = true;
    evidence.previous_block_index = previous_block & 1u;
    evidence.demanded_block_index = rounds & 1u;

    rc = relaunch_probe_prepare(fixture, slots, count, arm, rounds, rectangle, layout);
    evidence.status_poisoned_before = rc == FT_TT_OK;
    evidence.destination_poisoned_before = rc == FT_TT_OK;
    evidence.launch_rc = rc == FT_TT_OK ? launch_and_wait(fixture, slots, count) : rc;

    for (index = 0u; index < count && evidence.launch_rc == FT_TT_OK; index++) {
        core_status status;

        /* A read that FAILS is a host-side I/O failure, not "the kernel never ran", and it
         * is recorded as one — a ground that names the wrong cause is how a diagnosis goes
         * the wrong way at 2am. */
        rc = read_status(fixture, slots[index], &status);
        if (rc != FT_TT_OK) {
            evidence.launch_rc = rc;
            bad_slot = slots[index];
            break;
        }
        if (status.word[STATUS_MAGIC_WORD] != STATUS_MAGIC)
            evidence.every_status_magic_present = false;
        else if (status.word[STATUS_ROUNDS_COMPLETED] != rounds + 1u)
            evidence.rounds_complete = false;
        if (status.word[STATUS_ERROR] != 0u && evidence.kernel_error == 0u)
            evidence.kernel_error = status.word[STATUS_ERROR];
        rc = noc->read(noc->user, (uint16_t)fixture->members[slots[index]].x,
                       (uint16_t)fixture->members[slots[index]].y, layout->operand_address,
                       g_readback, fixture->payload_bytes);
        if (rc != FT_TT_OK) {
            evidence.launch_rc = rc;
            bad_slot = slots[index];
            break;
        }
        if (!blocks_match(expected, g_readback, fixture->payload_bytes, &difference))
            evidence.payload_matched_demanded = false;
        if (blocks_match(rejected, g_readback, fixture->payload_bytes, &difference))
            evidence.payload_rejected_other = false;
        if (bad_slot == MAX_CORES &&
            (!evidence.every_status_magic_present || !evidence.rounds_complete ||
             !evidence.payload_matched_demanded || !evidence.payload_rejected_other ||
             evidence.kernel_error != 0u))
            bad_slot = slots[index];
    }

    outcome = ft_tt_relaunch_gate_judge(&evidence);
    (void)snprintf(detail, detail_bytes,
                   "%u core(s), %u rounds, demanded block %c (previous %c): %s "
                   "[launch rc=%d (%s), kernel error 0x%08x, first core in question: %s%u]",
                   count, rounds, (char)('A' + (int)evidence.demanded_block_index),
                   (char)('A' + (int)evidence.previous_block_index),
                   ft_tt_relaunch_outcome_text(outcome), evidence.launch_rc,
                   ft_tt_error_string(evidence.launch_rc), evidence.kernel_error,
                   bad_slot < MAX_CORES ? "live index " : "none, n/a ",
                   bad_slot < MAX_CORES ? fixture->live_index[bad_slot] : 0u);
    return outcome;
}

/**
 * Stage 3c — EXPERIMENT A. Broadcast, then re-launch the same still-booted program.
 *
 * Read the header block for what a green and a red mean. In one line: this stage is CLEAN
 * and stage 4 is RED means the broadcast is innocent and park/boot is the defect; this
 * stage RED means the broadcast damaged the core with no re-boot in the story at all; both
 * green means the shape did not reproduce the wedge and the run says nothing.
 *
 * The RED CONTROL comes first and is not optional. It runs the identical probe aimed one
 * DRAM bank over — where stage 2b left a poison pattern that is neither payload — and
 * requires it to come back PAYLOAD_WRONG. Without it, every green below could be a probe
 * that cannot fail on silicon, which is the exact failure this file's stage 2b exists to
 * prevent for the port rotation.
 */
static void run_stage_3c_relaunch_no_reboot(bench_fixture *fixture)
{
    ft_tt_group_shared_operand_layout layout;
    ft_tt_multicast_rectangle rectangle;
    uint32_t receiver_slots[MAX_CORES];
    const uint32_t receivers = fixture->pool_rows * fixture->pool_columns;
    const uint32_t sender_slot = 0u;
    const uint32_t rounds = relaunch_probe_rounds(fixture);
    const uint32_t previous_block = fixture->iterations & 1u;
    ft_tt_relaunch_outcome outcome;
    uint32_t cycle;
    uint32_t index;
    char detail[256];
    int rc;

    ft_tt_test_step("stage 3c — EXPERIMENT A: %u-destination broadcast, then a RE-LAUNCH "
                    "with NO park and NO re-boot, %u cycles", receivers,
                    fixture->relaunch_cycles);

    if (receivers > SAFE_DESTINATION_LIMIT)
        printf("  This arm DELIBERATELY exceeds the %u-destination rule at %u receivers — "
               "that is the\n  hypothesis under test, and a wedge here is an expected "
               "outcome rather than a fault.\n  Run it only when you mean to reproduce the "
               "wedge, and only through\n  run_multicast_suite.sh so the compute probe "
               "follows it. Every one of these has\n  recovered with `ft_tt_core recover "
               "<cores> --rung 3 --write --then-probe` plus a\n  link retrain — which "
               "resets the WHOLE card, so nothing else may be using it.\n\n",
               SAFE_DESTINATION_LIMIT, receivers);

    for (index = 0u; index < receivers; index++)
        receiver_slots[index] = 1u + index;
    fill_layout(fixture, &layout);
    rc = ft_tt_multicast_rectangle_from_cores(&fixture->members[1], receivers,
                                              fixture->members[0], &rectangle);
    ft_tt_test_check_fmt("the pool is a legal rectangle with the sender outside it",
                         rc == FT_TT_OK, "%ux%u pool: rc=%d (%s)", fixture->pool_rows,
                         fixture->pool_columns, rc, ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    outcome = run_relaunch_probe(fixture, receiver_slots, receivers, ARM_DRAM_WRONG_BANK,
                                 rounds, previous_block, &rectangle, &layout, detail,
                                 sizeof(detail));
    ft_tt_test_check_fmt("RED CONTROL: the same probe aimed one BANK over goes red, so a "
                         "green below is a probe that COULD have failed",
                         outcome == FT_TT_RELAUNCH_PAYLOAD_WRONG, "%s", detail);
    if (outcome != FT_TT_RELAUNCH_PAYLOAD_WRONG)
        return;

    for (cycle = 0u; cycle < fixture->relaunch_cycles; cycle++) {
        arm_result multicast;

        rc = run_arm(fixture, ARM_MULTICAST, receiver_slots, receivers, fixture->iterations,
                     &multicast);
        ft_tt_test_check_fmt("the broadcast arm itself completed", rc == FT_TT_OK &&
                                 multicast.first_error == 0u && multicast.payload_correct &&
                                 multicast.payload_rejected_other,
                             "cycle %u, N=%u: rc=%d (%s), kernel error 0x%08x, payload %s",
                             cycle, receivers, rc, ft_tt_error_string(rc),
                             multicast.first_error,
                             multicast.payload_correct ? "ok" : "WRONG");
        if (rc != FT_TT_OK || multicast.first_error != 0u || !multicast.payload_correct)
            return;

        outcome = run_relaunch_probe(fixture, receiver_slots, receivers, ARM_DRAM, rounds,
                                     previous_block, &rectangle, &layout, detail,
                                     sizeof(detail));
        ft_tt_test_check_fmt("every RECEIVER accepted a re-launch with no park and no boot",
                             ft_tt_relaunch_outcome_is_clean(outcome), "cycle %u: %s", cycle,
                             detail);

        outcome = run_relaunch_probe(fixture, &sender_slot, 1u, ARM_DRAM, rounds,
                                     previous_block, &rectangle, &layout, detail,
                                     sizeof(detail));
        ft_tt_test_check_fmt("and so did the SENDER, which issued every broadcast",
                             ft_tt_relaunch_outcome_is_clean(outcome), "cycle %u: %s", cycle,
                             detail);
    }

    printf("  READ THIS WITH STAGE 4, NOT ALONE:\n"
           "    3c clean + stage 4 RED   -> the broadcast is innocent; the defect is in\n"
           "                                ft_tt_core_park() / ft_tt_core_boot_firmware()\n"
           "                                after a broadcast has landed in a core's L1.\n"
           "    3c RED                   -> the broadcast itself damaged a core with no\n"
           "                                re-boot involved; the >4 rule stands as written.\n"
           "    3c clean + stage 4 GREEN -> this shape did not reproduce the wedge. The run\n"
           "                                says NOTHING either way and must not be quoted.\n\n");
}

/* ------------------------------------------------------------------------- */
/* Experiment B — two disjoint 4-destination broadcasts instead of one of 8   */
/* ------------------------------------------------------------------------- */

/** How many senders the split arm runs, and therefore how many groups. */
#define SPLIT_GROUP_COUNT 2u

/** Largest destination count a single broadcast in the split arm may carry. This is the
 *  standing rule the experiment is built to respect, not a tuning knob: the arm's whole
 *  claim is that no packet exceeds four destinations. */
#define SPLIT_MAX_DESTINATIONS SAFE_DESTINATION_LIMIT

/** One sender and the row of receivers it broadcasts to. */
typedef struct split_group {
    ft_tt_multicast_rectangle rectangle;
    uint32_t sender_slot;
    uint32_t receiver_slots[MAX_CORES];
    uint32_t receiver_count;
} split_group;

/**
 * Turn the pool into two groups, refusing every geometry that is not safe to issue.
 *
 * The refusals themselves live in ft_tt_split_broadcast_geometry.c — a rectangle over the
 * per-packet limit, a sender inside ANY rectangle, two rectangles that overlap — because
 * arithmetic that protects the card has to be provable WITHOUT the card, and
 * tests/test_split_broadcast_geometry.c drives every one of them red on the Mac. What is
 * left here is the slot bookkeeping, which is this program's business and nobody else's.
 */
static int build_split_groups(const bench_fixture *fixture, split_group *groups, char *why,
                              size_t why_bytes)
{
    ft_tt_noc_coord senders[SPLIT_GROUP_COUNT];
    ft_tt_split_broadcast_plan plan;
    uint32_t group;
    uint32_t index;
    int rc;

    if (fixture->pool_rows != SPLIT_GROUP_COUNT || fixture->sender_b_slot == 0u) {
        (void)snprintf(why, why_bytes,
                       "needs a %u-row pool and a second sender; got %u rows and sender B "
                       "slot %u", SPLIT_GROUP_COUNT, fixture->pool_rows,
                       fixture->sender_b_slot);
        return FT_TT_EINVAL;
    }

    for (group = 0u; group < SPLIT_GROUP_COUNT; group++) {
        groups[group].sender_slot = group == 0u ? 0u : fixture->sender_b_slot;
        groups[group].receiver_count = fixture->pool_columns;
        senders[group] = fixture->members[groups[group].sender_slot];
        for (index = 0u; index < fixture->pool_columns; index++)
            groups[group].receiver_slots[index] = 1u + group * fixture->pool_columns + index;
    }

    rc = ft_tt_split_broadcast_plan_build(&fixture->members[1], SPLIT_GROUP_COUNT,
                                          fixture->pool_columns, senders, SPLIT_GROUP_COUNT,
                                          SPLIT_MAX_DESTINATIONS, &plan, why, why_bytes);
    if (rc != FT_TT_OK)
        return rc;
    for (group = 0u; group < SPLIT_GROUP_COUNT; group++)
        groups[group].rectangle = plan.rectangle[group];
    return FT_TT_OK;
}

/** Poison one group's destinations and staging, and put its barrier in its start state. */
static int split_group_stage_l1(const bench_fixture *fixture, const split_group *group,
                                const ft_tt_group_shared_operand_layout *layout)
{
    ft_tt_noc_coord receivers[MAX_CORES];
    ft_tt_group_shared_operand operand;
    const ft_tt_noc_access *noc = &fixture->session->noc;
    const uint32_t sender_slot = group->sender_slot;
    uint32_t index;
    int rc;

    memset(g_readback, DESTINATION_POISON_BYTE, fixture->payload_bytes);
    for (index = 0u; index < group->receiver_count; index++) {
        uint32_t slot = group->receiver_slots[index];

        receivers[index] = fixture->members[slot];
        rc = noc->write(noc->user, (uint16_t)fixture->members[slot].x,
                        (uint16_t)fixture->members[slot].y, layout->operand_address,
                        g_readback, fixture->payload_bytes);
        if (rc != FT_TT_OK)
            return rc;
    }
    memset(g_readback, STAGING_POISON_BYTE, fixture->payload_bytes);
    rc = noc->write(noc->user, (uint16_t)fixture->members[sender_slot].x,
                    (uint16_t)fixture->members[sender_slot].y, staging_address(fixture, 0u),
                    g_readback, fixture->payload_bytes);
    if (rc == FT_TT_OK)
        rc = noc->write(noc->user, (uint16_t)fixture->members[sender_slot].x,
                        (uint16_t)fixture->members[sender_slot].y,
                        staging_address(fixture, 1u), g_readback, fixture->payload_bytes);
    if (rc == FT_TT_OK)
        rc = ft_tt_group_shared_operand_bind(receivers, group->receiver_count,
                                             fixture->members[sender_slot], layout, &operand);
    if (rc == FT_TT_OK)
        rc = ft_tt_group_shared_operand_reset(noc, &operand, receivers,
                                              group->receiver_count);
    return rc;
}

/** Plant one group's runtime args: its sender, then its receivers pointing at that sender. */
static int split_group_plant(const bench_fixture *fixture, const split_group *group,
                             uint32_t iterations,
                             const ft_tt_group_shared_operand_layout *layout)
{
    const ft_tt_noc_coord sender = fixture->members[group->sender_slot];
    uint32_t index;
    int rc;

    rc = poison_control(fixture, group->sender_slot);
    if (rc == FT_TT_OK)
        rc = plant_arguments(fixture, group->sender_slot, ROLE_MULTICAST_SENDER, iterations,
                             fixture->payload_bytes, &group->rectangle, layout,
                             fixture->dram, sender);
    for (index = 0u; index < group->receiver_count && rc == FT_TT_OK; index++) {
        rc = poison_control(fixture, group->receiver_slots[index]);
        if (rc == FT_TT_OK)
            rc = plant_arguments(fixture, group->receiver_slots[index],
                                 ROLE_MULTICAST_RECEIVER, iterations, fixture->payload_bytes,
                                 &group->rectangle, layout, fixture->dram, sender);
    }
    return rc;
}

/**
 * Run one split arm: both senders and all their receivers, one launch.
 *
 * The slot order matters and matches launch_and_wait()'s contract: it sends GO from the
 * LAST slot backwards, so the senders go at the front of the array and are launched last —
 * their untimed round 0 then spends less of its bound waiting for receivers that have not
 * been told to start.
 */
static int run_split_arm(bench_fixture *fixture, const split_group *groups,
                         uint32_t iterations, arm_result *out)
{
    ft_tt_group_shared_operand_layout layout;
    uint32_t slots[MAX_CORES];
    uint32_t slot_count = 0u;
    uint32_t group;
    uint32_t index;
    uint32_t difference = 0u;
    const uint8_t *expected = ((iterations & 1u) != 0u) ? g_block_b : g_block_a;
    const uint8_t *rejected = ((iterations & 1u) != 0u) ? g_block_a : g_block_b;
    const ft_tt_noc_access *noc = &fixture->session->noc;
    int rc = FT_TT_OK;

    memset(out, 0, sizeof(*out));
    out->payload_correct = true;
    out->payload_rejected_other = true;
    out->every_status_magic_present = true;
    out->rounds_complete = true;
    fill_layout(fixture, &layout);

    for (group = 0u; group < SPLIT_GROUP_COUNT && rc == FT_TT_OK; group++)
        rc = split_group_stage_l1(fixture, &groups[group], &layout);
    for (group = 0u; group < SPLIT_GROUP_COUNT && rc == FT_TT_OK; group++)
        rc = split_group_plant(fixture, &groups[group], iterations, &layout);
    if (rc != FT_TT_OK)
        return rc;

    for (group = 0u; group < SPLIT_GROUP_COUNT; group++)
        slots[slot_count++] = groups[group].sender_slot;
    for (group = 0u; group < SPLIT_GROUP_COUNT; group++)
        for (index = 0u; index < groups[group].receiver_count; index++)
            slots[slot_count++] = groups[group].receiver_slots[index];

    out->rc = launch_and_wait(fixture, slots, slot_count);
    if (out->rc != FT_TT_OK)
        return out->rc;

    for (index = 0u; index < slot_count; index++) {
        core_status status;

        rc = read_status(fixture, slots[index], &status);
        if (rc != FT_TT_OK)
            return rc;
        if (status.word[STATUS_MAGIC_WORD] != STATUS_MAGIC) {
            out->every_status_magic_present = false;
            out->first_error = status.word[STATUS_MAGIC_WORD];
            return FT_TT_ERROR;
        }
        if (status.word[STATUS_ERROR] != 0u && out->first_error == 0u)
            out->first_error = status.word[STATUS_ERROR];
        if (status.word[STATUS_ROUNDS_COMPLETED] != iterations + 1u) {
            out->rounds_complete = false;
            if (out->first_error == 0u)
                out->first_error = 0xDEAD0000u | status.word[STATUS_ROUNDS_COMPLETED];
        }
        if (index < SPLIT_GROUP_COUNT)
            out->sender_cycles = status.word[STATUS_ELAPSED_CYCLES];
        else if (status.word[STATUS_ELAPSED_CYCLES] > out->worst_cycles)
            out->worst_cycles = status.word[STATUS_ELAPSED_CYCLES];
    }

    for (index = SPLIT_GROUP_COUNT; index < slot_count; index++) {
        rc = noc->read(noc->user, (uint16_t)fixture->members[slots[index]].x,
                       (uint16_t)fixture->members[slots[index]].y, layout.operand_address,
                       g_readback, fixture->payload_bytes);
        if (rc != FT_TT_OK)
            return rc;
        if (!blocks_match(expected, g_readback, fixture->payload_bytes, &difference))
            out->payload_correct = false;
        if (blocks_match(rejected, g_readback, fixture->payload_bytes, &difference))
            out->payload_rejected_other = false;
    }
    return FT_TT_OK;
}

/**
 * Stage 3d — EXPERIMENT B. The same N receivers, reached by two disjoint broadcasts of at
 * most four destinations each instead of one broadcast of N.
 *
 * This stage does not decide anything by itself — every launch here is expected to
 * complete, exactly as the 8-destination one does. STAGE 4 decides it: if the cores still
 * boot afterwards, the variable is destinations-per-packet and one sender issuing
 * `ceil(N/4)` broadcasts is a sound workaround; if they do not, the variable is "took part
 * in a run with >= 5 receivers", which is not a property of a packet and points back at
 * experiment A.
 *
 * The confound worth stating rather than hiding: this arm runs TWO senders where the
 * 8-destination arm runs one. Making it one sender issuing two broadcasts per round needs
 * a kernel change, and a kernel change would alter what is being compared.
 */
static void run_stage_3d_split_broadcast(bench_fixture *fixture)
{
    split_group groups[SPLIT_GROUP_COUNT];
    uint32_t repetition;
    char why[192];
    int rc;

    ft_tt_test_step("stage 3d — EXPERIMENT B: N=%u as %u disjoint broadcasts of <= %u "
                    "destinations, %u repetitions",
                    fixture->pool_rows * fixture->pool_columns, SPLIT_GROUP_COUNT,
                    SPLIT_MAX_DESTINATIONS, REPETITIONS);

    rc = build_split_groups(fixture, groups, why, sizeof(why));
    ft_tt_test_check_fmt("the split geometry is legal and no broadcast exceeds four "
                         "destinations", rc == FT_TT_OK, "%s", why);
    if (rc != FT_TT_OK)
        return;

    for (repetition = 0u; repetition < REPETITIONS; repetition++) {
        arm_result split;

        rc = run_split_arm(fixture, groups, fixture->iterations, &split);
        ft_tt_test_check_fmt("both broadcasts landed and every payload tracked the source",
                             rc == FT_TT_OK && split.first_error == 0u &&
                                 split.payload_correct && split.payload_rejected_other,
                             "repetition %u: rc=%d (%s), kernel error 0x%08x, payload %s, "
                             "other-block %s", repetition, rc, ft_tt_error_string(rc),
                             split.first_error, split.payload_correct ? "ok" : "WRONG",
                             split.payload_rejected_other ? "rejected" : "MATCHED");
        if (rc != FT_TT_OK || split.first_error != 0u || !split.payload_correct)
            return;
    }

    printf("  STAGE 4 IS WHAT DECIDES THIS ARM:\n"
           "    stage 4 GREEN -> the variable is DESTINATIONS PER PACKET; ceil(N/4)\n"
           "                     broadcasts from one sender is a sound workaround.\n"
           "    stage 4 RED   -> the variable is \"took part in a run with >= 5\n"
           "                     receivers\", which is not a property of a packet at all,\n"
           "                     and the suspect is the teardown (experiment A).\n"
           "  Confound, stated: this arm has TWO senders where the 8-destination arm has\n"
           "  one. One sender issuing two broadcasts per round would need a kernel change.\n\n");
}

/**
 * Stage 3b — what state every core is in BEFORE anything re-boots it.
 *
 * This exists because "the suite left a core unbootable" has two very different causes and
 * a stage-4 failure alone cannot tell them apart: the WORK may have left the core bad, or
 * the RE-BOOT may be what breaks it (`ft_tt_core_park()` on a core that has just finished is
 * exactly the shape that broke core 65's next boot, DEV_NOTES.md §1). This stage issues
 * READS ONLY, so it is safe on a stuck core, and it runs before stage 4 touches anything.
 *
 * A healthy core here reads `soft reset 0x00047001` — BRISC held after the host's park, or
 * all-five-out-of-reset with the mailbox at DONE if it is idle in firmware. What it must NOT
 * read is the mid-handshake `0x00047000` with the subordinates still at 0x00, which is the
 * signature of a boot that never finished.
 */
static void run_stage_3b_state_before_reboot(bench_fixture *fixture)
{
    uint32_t slot;

    ft_tt_test_step("stage 3b — every core's run state BEFORE any re-boot (reads only)");

    for (slot = 0u; slot < fixture->member_count; slot++) {
        ft_tt_launch_state state;
        int rc = ft_tt_launch_read_state(&fixture->session->noc,
                                         (uint16_t)fixture->members[slot].x,
                                         (uint16_t)fixture->members[slot].y, &state);

        ft_tt_test_check_fmt("a core answers with a plausible run state", rc == FT_TT_OK,
                             "core slot %u = live index %u at translated (%u,%u): rc=%d "
                             "(%s), soft reset 0x%08x, subordinate sync %02x %02x %02x %02x",
                             slot, fixture->live_index[slot], fixture->members[slot].x,
                             fixture->members[slot].y, rc,
                             ft_tt_error_string(rc), state.reset_register,
                             state.subordinate_sync[0], state.subordinate_sync[1],
                             state.subordinate_sync[2], state.subordinate_sync[3]);
    }
}

/**
 * Stage 4 — every core this suite touched still BOOTS.
 *
 * UNCONDITIONAL, even when an earlier stage failed, because a read cannot tell a wedged
 * core from a healthy idle one and a multicast bug is exactly the kind that wedges a
 * receiver quietly. The COMPUTE half of the rule is `ft_tt_core probe`,
 * a DIFFERENT triad, run by run_multicast_suite.sh immediately after this program.
 */
static void run_stage_4_cores_still_boot(bench_fixture *fixture)
{
    uint32_t slot;

    ft_tt_test_step("stage 4 — every core touched still BOOTS (run unconditionally)");

    for (slot = 0u; slot < fixture->member_count; slot++) {
        int rc = boot_core(fixture, slot);

        ft_tt_test_check_fmt("a core this suite touched still boots", rc == FT_TT_OK,
                             "core slot %u = live index %u at translated (%u,%u): rc=%d (%s)",
                             slot, fixture->live_index[slot], fixture->members[slot].x,
                             fixture->members[slot].y, rc, ft_tt_error_string(rc));
    }
    printf("  The COMPUTE half of this check is `ft_tt_core probe`, which drives its own\n"
           "  vector_add triad and must be run next — see run_multicast_suite.sh.\n\n");
}

/* ------------------------------------------------------------------------- */

/** BRISC out of reset with every subordinate still held. This is the fingerprint every
 *  wedge on this card has carried — `soft reset 0x00047000`, boot handshake never finished
 *  — and it is deliberately an EQUALITY against that one value: a core
 *  freshly parked reads all five held, and a core idle in firmware reads none held, so
 *  neither can be mistaken for this. */
#define WEDGED_SOFT_RESET_FINGERPRINT \
    (FT_TT_SOFT_RESET_ALL_RISCS & ~(uint32_t)FT_TT_SOFT_RESET_BRISC)

/**
 * RECOVERY-ON-ACQUIRE for the core claim: INSPECT and REFUSE, never retire by parking.
 *
 * A released claim is not a retired residency — the kernel hands the lock back the instant
 * a holder dies and cannot hand the CORE back — so the next claimant has to decide what to
 * do with what it finds. The one thing it must not do is assert reset: parking a core
 * whose processors are still live is what breaks the NEXT boot, which is the very failure
 * this program's experiments are trying to attribute. So this reads, and refuses.
 *
 * What it can and cannot see is worth being honest about. It catches a core left in the
 * known wedge fingerprint, and it catches a core that cannot be read at all. It does NOT
 * catch a core happily running somebody else's resident kernel — that reads as all five
 * out of reset, exactly like an idle one — and there is no generic way to tell those apart
 * without asking the op family's own queue (ft_tt_tensix_claim.h). The claim itself is
 * what protects against a live holder; this is the belt to that braces.
 *
 * @return 0 to accept the core, non-zero to refuse it (which releases the whole set)
 */
static int refuse_a_core_that_shows_the_wedge_fingerprint(void *context, uint32_t core_index)
{
    const ft_tt_hardware_session *session = (const ft_tt_hardware_session *)context;
    ft_tt_launch_state state;
    ft_tt_noc_coord coord;
    int rc;

    rc = ft_tt_coord_tensix_by_index(&session->grid, core_index, FT_TT_COORD_TRANSLATED,
                                     &coord);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_read_state(&session->noc, (uint16_t)coord.x, (uint16_t)coord.y, &state);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "core %u at translated (%u,%u) could not be read: %s\n", core_index,
                coord.x, coord.y, ft_tt_error_string(rc));
        return rc;
    }
    if (state.reset_register == WEDGED_SOFT_RESET_FINGERPRINT) {
        fprintf(stderr, "core %u at translated (%u,%u) is already wedged: soft reset "
                        "0x%08x — recover it with `ft_tt_core recover %u --rung 3 --write "
                        "--then-probe` before running here\n",
                core_index, coord.x, coord.y, state.reset_register, core_index);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_program_directory program;
    ft_tt_noc_path dram_path;
    bench_fixture fixture;
    const char *directory;
    uint32_t slot;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <program-directory> [iterations] [payload-bytes]\n",
                argv[0]);
        return 2;
    }
    directory = argv[1];

    memset(&fixture, 0, sizeof(fixture));
    fixture.iterations = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 0) : DEFAULT_ITERATIONS;
    fixture.payload_bytes =
        argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 0) : DEFAULT_PAYLOAD_BYTES;
    fixture.cycles_per_second = 800e6; /* replaced by stage 1's measurement */
    fixture.first_index = environment_u32("FT_TT_MULTICAST_FIRST_INDEX", DEFAULT_FIRST_INDEX);
    fixture.sender_index = environment_u32("FT_TT_MULTICAST_SENDER_INDEX", DEFAULT_SENDER_INDEX);
    {
        const char *arms = getenv("FT_TT_MULTICAST_ARMS");

        if (arms != NULL && strcmp(arms, "multicast") == 0)
            fixture.arms = ARMS_MULTICAST_ONLY;
        else if (arms != NULL && strcmp(arms, "dram") == 0)
            fixture.arms = ARMS_DRAM_ONLY;
        else
            fixture.arms = ARMS_BOTH;
    }
    {
        const char *experiment = getenv("FT_TT_MULTICAST_EXPERIMENT");

        if (experiment != NULL && strcmp(experiment, "relaunch") == 0)
            fixture.experiment = EXPERIMENT_RELAUNCH;
        else if (experiment != NULL && strcmp(experiment, "split") == 0)
            fixture.experiment = EXPERIMENT_SPLIT;
        else if (experiment != NULL && experiment[0] != '\0' &&
                 strcmp(experiment, "none") != 0) {
            fprintf(stderr, "FT_TT_MULTICAST_EXPERIMENT must be none, relaunch or split\n");
            return 2;
        }
    }
    fixture.relaunch_cycles =
        environment_u32("FT_TT_MULTICAST_RELAUNCH_CYCLES", DEFAULT_RELAUNCH_CYCLES);
    {
        uint32_t default_rows = POOL_ROWS;
        uint32_t default_columns = POOL_COLUMNS;

        if (fixture.experiment == EXPERIMENT_RELAUNCH) {
            default_rows = RELAUNCH_POOL_ROWS;
            default_columns = RELAUNCH_POOL_COLUMNS;
        } else if (fixture.experiment == EXPERIMENT_SPLIT) {
            default_rows = SPLIT_POOL_ROWS;
            default_columns = SPLIT_POOL_COLUMNS;
        }
        fixture.pool_rows = environment_u32("FT_TT_MULTICAST_POOL_ROWS", default_rows);
        fixture.pool_columns =
            environment_u32("FT_TT_MULTICAST_POOL_COLUMNS", default_columns);
    }
    if (fixture.pool_rows == 0u || fixture.pool_columns == 0u ||
        fixture.pool_rows * fixture.pool_columns + 2u > MAX_CORES ||
        fixture.relaunch_cycles == 0u) {
        fprintf(stderr, "pool must be 1..%u cores and relaunch cycles at least 1\n",
                MAX_CORES - 2u);
        return 2;
    }
    fixture.member_count = fixture.pool_rows * fixture.pool_columns + 1u;
    /* The second sender exists only for the split experiment, and only then is its core
     * booted, probed and re-booted. Every other mode leaves the slot unoccupied. */
    if (fixture.experiment == EXPERIMENT_SPLIT) {
        fixture.sender_b_index =
            environment_u32("FT_TT_MULTICAST_SENDER_B_INDEX", DEFAULT_SENDER_B_INDEX);
        fixture.sender_b_slot = fixture.member_count;
        fixture.member_count++;
    }

    if (fixture.payload_bytes == 0u || fixture.payload_bytes > MAX_PAYLOAD_BYTES ||
        (fixture.payload_bytes % 64u) != 0u) {
        fprintf(stderr, "payload must be a multiple of 64 in 64..%u bytes\n",
                MAX_PAYLOAD_BYTES);
        return 2;
    }
    if (fixture.iterations == 0u) {
        fprintf(stderr, "iterations must be at least 1\n");
        return 2;
    }

    ft_tt_test_step("multicast vs N cores reading DRAM — %u iterations, %u-byte block",
                    fixture.iterations, fixture.payload_bytes);

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;
    fixture.session = &session;

    /* This arm opens its OWN write-combined window through the tt-kmd context, so it
     * is one of the harnesses that cannot run on the userspace driver: the udrv bulk
     * path is an ft_tt_noc_access (runtime.bulk_dram), not an ft_tt_noc_path, and this
     * fixture holds the latter. It names the transport it needs instead of
     * dereferencing a handle that is NULL there. */
    if (!ft_tt_hardware_session_kmd_context(&session)) {
        ft_tt_log_error("this suite opens a second WC window through the tt-kmd context; "
                        "run it with FT_TT_DRIVER=kmd");
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    rc = ft_tt_noc_path_open(ft_tt_hardware_session_kmd_context(&session), FT_TT_KMD_CACHE_WC,
                             &dram_path);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    fixture.dram_path = &dram_path;

    /* Port 0 of the bank, which is what `ft_tt_coord_dram_bank()` returns and what the
     * un-rotated arm and every host-side write below use — the historical baseline, kept
     * verbatim so the new arm is compared against it rather than replacing it. */
    rc = ft_tt_coord_dram_bank(&session.grid, DRAM_LOGICAL_BANK, FT_TT_COORD_TRANSLATED,
                               &fixture.dram);
    /* And ALL of that bank's doors, plus the next bank's for the red arm. Derived from live
     * telemetry, never probed: a wrong DRAM endpoint does not fail a read, it hangs the chip. */
    if (rc == FT_TT_OK)
        rc = ft_tt_dram_port_set_build(&session.grid, DRAM_LOGICAL_BANK,
                                       FT_TT_COORD_TRANSLATED, &fixture.dram_ports);
    if (rc == FT_TT_OK)
        rc = ft_tt_dram_port_set_build(&session.grid,
                                       (DRAM_LOGICAL_BANK + 1u) %
                                           ft_tt_coord_dram_bank_count(&session.grid),
                                       FT_TT_COORD_TRANSLATED, &fixture.wrong_bank_ports);
    if (rc == FT_TT_OK)
        rc = ft_tt_coord_tensix_by_index(&session.grid, fixture.sender_index,
                                         FT_TT_COORD_TRANSLATED, &fixture.members[0]);
    if (rc == FT_TT_OK)
        rc = ft_tt_group_rectangle_coordinates(&session.grid, fixture.first_index,
                                               fixture.pool_rows, fixture.pool_columns,
                                               &fixture.members[1], MAX_CORES - 1u);
    if (rc == FT_TT_OK && fixture.sender_b_slot != 0u)
        rc = ft_tt_coord_tensix_by_index(&session.grid, fixture.sender_b_index,
                                         FT_TT_COORD_TRANSLATED,
                                         &fixture.members[fixture.sender_b_slot]);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("cores and DRAM bank resolved from live telemetry", false,
                         ft_tt_error_string(rc));
        goto close_path;
    }

    /* The same arithmetic ft_tt_group_rectangle_coordinates() used, kept beside the members
     * it labels: pool member (row, column) is live index first + row * grid columns +
     * column, and slot 0 is the sender. Derived from the LIVE column count, never 14. */
    {
        uint32_t column_count = ft_tt_coord_tensix_column_count(&session.grid);
        uint32_t pool_count = fixture.pool_rows * fixture.pool_columns;
        uint32_t pool_slot;

        fixture.live_index[0] = fixture.sender_index;
        for (pool_slot = 0u; pool_slot < pool_count; pool_slot++)
            fixture.live_index[pool_slot + 1u] = fixture.first_index +
                                                 (pool_slot / fixture.pool_columns) *
                                                     column_count +
                                                 (pool_slot % fixture.pool_columns);
        if (fixture.sender_b_slot != 0u)
            fixture.live_index[fixture.sender_b_slot] = fixture.sender_b_index;
    }

    /* CLAIM every core this process is about to drive, BEFORE it drives any of them and
     * before the firmware is even read from disk. Per-core rather than the card lock
     * (ft_tt_tensix_claim.h): everything below writes one core's L1, mailbox and soft-reset
     * register at a time. The MULTICAST is the part a core claim does not fully cover — a
     * broadcast physically reaches every node in its bounding box — which is why the
     * rectangle is always exactly the claimed pool, and why that header lists this as an
     * unsettled blast radius rather than a solved one. */
    rc = ft_tt_hardware_session_claim_cores(&session, fixture.live_index,
                                            fixture.member_count, "multicast bench",
                                            refuse_a_core_that_shows_the_wedge_fingerprint,
                                            &session);
    ft_tt_test_check_fmt("every core this run drives is claimed exclusively", rc == FT_TT_OK,
                         "%u cores from live index %u plus sender(s): rc=%d (%s)",
                         fixture.member_count, fixture.first_index, rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto close_path;

    rc = ft_tt_firmware_set_load(directory, &firmware);
    if (rc != FT_TT_OK)
        goto close_path;
    fixture.firmware = &firmware;
    /* DM0 + the three TRISCs. NCRISC is not enabled: this kernel is BRISC-only, and a
     * processor with no image must not be in the enables mask. */
    rc = ft_tt_program_directory_load(directory, 0x1du, &program);
    if (rc != FT_TT_OK) {
        ft_tt_firmware_set_release(&firmware);
        goto close_path;
    }
    fixture.program = &program;

    printf("  sender: live index %u at translated (%u,%u)\n", fixture.sender_index,
           fixture.members[0].x, fixture.members[0].y);
    printf("  receiver pool: %ux%u rectangle from live index %u, translated x %u..%u, "
           "y %u..%u\n", fixture.pool_rows, fixture.pool_columns, fixture.first_index,
           fixture.members[1].x, fixture.members[fixture.pool_columns].x,
           fixture.members[1].y, fixture.members[fixture.member_count - 1u].y);

    ft_tt_test_step("boot — %u cores, SEQUENTIALLY (concurrent boots at 12-way and above "
                    "wedge cores, DEV_NOTES.md §1)", fixture.member_count);
    for (slot = 0u; slot < fixture.member_count; slot++) {
        rc = boot_core(&fixture, slot);
        ft_tt_test_check_fmt("a core booted with the program loaded", rc == FT_TT_OK,
                             "core slot %u at translated (%u,%u): rc=%d (%s)",
                             slot, fixture.members[slot].x,
                             fixture.members[slot].y, rc, ft_tt_error_string(rc));
        if (rc != FT_TT_OK)
            goto health;
    }

    run_stage_0_transfer_round_trip(&fixture);
    run_stage_1_clock_calibration(&fixture);
    run_stage_2_refusals(&fixture);
    run_stage_2b_wrong_bank_red(&fixture);
    /* ONE stage-3 body per process. The experiments exist to attribute a wedge to one
     * thing, and a process that ran the timed sweep first would have broadcast at every
     * shape from 1x1 to 2x5 on these cores before the experiment began. */
    if (fixture.experiment == EXPERIMENT_RELAUNCH)
        run_stage_3c_relaunch_no_reboot(&fixture);
    else if (fixture.experiment == EXPERIMENT_SPLIT)
        run_stage_3d_split_broadcast(&fixture);
    else
        run_stage_3_sweep(&fixture);

health:
    run_stage_3b_state_before_reboot(&fixture);
    run_stage_4_cores_still_boot(&fixture);

    ft_tt_program_directory_release(&program);
    ft_tt_firmware_set_release(&firmware);

close_path:
    (void)ft_tt_noc_path_close(&dram_path);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
