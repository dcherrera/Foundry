/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_dram_capacity.c — how much DRAM read bandwidth this card actually delivers, and
 * therefore how many Tensix cores can be kept fed.
 *
 * ===========================================================================
 * THE CONTRADICTION THIS EXISTS TO SETTLE
 * ===========================================================================
 * Two of this project's own measurements disagree and both look sound:
 *
 *   Phase 6.7 (cooperative matmul)  one DRAM bank tops out at ~42 GB/s, and one Tensix
 *                                   reader already runs at 39 — 92% of a bank. 8 x 42 =
 *                                   336 GB/s, which feeds ~8-9 cores of 140.
 *   the earlier DRAM round          >= 537 GB/s across 14 concurrent cores (~67 GB/s a
 *                                   bank) and NOT saturated — the ceiling was never found.
 *
 * The suspected difference is ACCESS PATTERN: the 537 test had all 14 cores reading a
 * SHARED ~17 MB working set, the friendliest possible row-buffer pattern, while the 42
 * GB/s test streams ~14 MB per bank, disjoint. Both are measurements. Until it is settled,
 * any statement about how many models the card can serve is a guess.
 *
 * FOUR VARIABLES ARE CONFOUNDED IN THOSE TWO RESULTS AND THIS PROGRAM SEPARATES THEM:
 *
 *   sharing        N cores on the same addresses vs N cores on disjoint ones
 *   working set    from small enough to sit in row buffers to far past any locality effect
 *   bank AND PORT  a Blackhole GDDR bank exposes THREE NoC ports — three doors into one
 *                  memory (ft_tt_coord.h). Everything measured here before used port 0
 *                  only, because ft_tt_coord_dram_bank() IS port 0. "Per bank" and "per
 *                  port" are therefore not the same claim, and neither old number says
 *                  which one it measured
 *   core count     1 .. past 14, since 14 is where 537 was recorded and was called
 *                  unsaturated
 *
 * ===========================================================================
 * WHAT MAKES A NUMBER FROM THIS PROGRAM EVIDENCE
 * ===========================================================================
 *  - **The harness is shown to produce a number nobody wanted.** Stage 4 runs a
 *    DELIBERATELY SERIALISED arm — `barrier_every = 1`, one read waited on before the next
 *    is issued — whose answer must be far BELOW the pipelined arms because it is
 *    round-trip-latency bound. It also reproduces the recorded single-core reader rate. A
 *    bandwidth harness that only ever prints plausible figures has not been tested.
 *  - **Device-side timing, host-calibrated.** Each core reads its own wall clock, so the
 *    ~14.5 us launch and this 2-core host's scheduling are outside the window. Stage 2
 *    CALIBRATES cycles to seconds from two run lengths rather than assuming the measured
 *    800 MHz busy AICLK — "the clock did not change under load" and "we sampled while idle"
 *    print the same number. The sysfs AICLK is sampled at both ends of
 *    every sweep anyway, so an assumption and a reading are both on the page.
 *  - **Arms are INTERLEAVED, never run one condition to completion and then the next.**
 *    The whole condition list is run REPETITIONS times round-robin; this host drifts enough
 *    over ten seconds to swamp the quantity.
 *  - **Best, median and worst of N are all reported.** A mean hides a bimodal result.
 *  - **Host load is recorded at both ends of every repetition.** `pantheon` serves on the
 *    K80s of this same 2-core box and must not be killed; a host-side stall inside a timed
 *    region is a real risk here, and a run that straddles a change in load is void.
 *  - **Stage 3 proves the bytes moved.** Every L1 destination slot is poisoned, and after
 *    the run must hold the DRAM contents at the address the walk was pointing at — an
 *    address the host recomputes in closed form and cross-checks against the kernel's own
 *    published `final_walk_offset`.
 *  - **Stage 5 is the red half**: three runs that must be REFUSED in the kernel before any
 *    transaction is issued (zero length, bad alignment, an L1 window that does not fit).
 *
 * ===========================================================================
 * STAGES, AND HOW TO RUN THEM
 * ===========================================================================
 *   usage: test_dram_capacity <program-directory>
 *
 *   FT_TT_DRAMCAP_STAGES     comma list, default "validate,onebank,workingset,aggregate"
 *                            also: blocksize, depth, peak, the four the cliff/NoC round
 *                            added — cliff, spread, distance, ports — plus `anomaly` (sweep P) and
 *                            `address` (sweep Q), which ask why a 64 KiB working set is
 *                            slower than a 512 MiB one, and `mmshape` (sweep M), the decode
 *                            matmul's read shape without its compute
 *   FT_TT_DRAMCAP_PERCORE    non-zero: print every reader's own rate, its grid position, the
 *                            endpoint it read and the hop counts between them
 *   FT_TT_DRAMCAP_FIRST_INDEX first live Tensix index (default 8; index 0 is the
 *                            bring-up core and is never used)
 *   FT_TT_DRAMCAP_CORES      how many cores to boot (default 48, max MAX_CORES)
 *   FT_TT_DRAMCAP_REPS       repetitions of the interleaved sweep (default 3)
 *   FT_TT_FIRMWARE_DIR       the same program directory
 *
 * Build:
 *   make -f Makefile.dramcap PHASE1B=1 all
 *
 * @status NEW 2026-08-14. Extended the same day with the placement/endpoint axes
 *   (core_offset, core_stride, bank_offset, port_offset, port_by_bank) and sweeps G-J, for
 *   the in-flight cliff, the DRAM-or-NoC question and the port-0 cap. 2026-08-15 added the
 *   BLOCK STRIDE axis and sweep P for the 64 KiB anomaly. Stage 3 runs three times — once
 *   plain, once through the placement/endpoint axes and once through the stride — so every
 *   axis a sweep depends on is falsifiable rather than assumed.
 *   2026-08-18: RUNS ON THE USERSPACE DRIVER. The host DRAM access moved off a
 *   self-opened `ft_tt_noc_path` (a tt-kmd type) onto the session's `runtime.bulk_dram`
 *   accessor, which is wired on both transports and is the same WC/BLOCK16 window. The
 *   suite previously refused with "run it with FT_TT_DRIVER=kmd" and so could not run at
 *   all on the board the card now lives in, where tt-kmd is never installed.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_hardware_session.h"
#include "ft_tt_noc.h"
#include "ft_tt_phase2_l1_layout.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------------- */
/* The wire contract — kernels/dram_bandwidth_reader.c carries the same values */
/* ------------------------------------------------------------------------- */

#define STATUS_MAGIC 0x46544442u /* "FTDB" */
#define STATUS_MAGIC_WORD 0u
#define STATUS_PASSES_COMPLETED 1u
#define STATUS_BLOCKS_ISSUED 2u
#define STATUS_ELAPSED_CYCLES 3u
#define STATUS_ERROR 4u
#define STATUS_START_CLOCK 5u
#define STATUS_END_CLOCK 6u
#define STATUS_FINAL_WALK_OFFSET 7u
#define STATUS_WORDS 8u

#define FTDB_OK 0
#define FTDB_ERROR_ZERO_LENGTH (-1)
#define FTDB_ERROR_ALIGNMENT (-2)
#define FTDB_ERROR_SHAPE (-3)
#define FTDB_ERROR_L1_BOUNDS (-4)

#define KERNEL_ARG_COUNT 15u

/** Different from every state the kernel writes and from cleared L1, so "never ran" is its
 *  own bit pattern rather than a zero that could also mean success. */
#define STATUS_POISON 0x7B1B1B1Bu
#define SLOT_POISON_BYTE 0xA5u

/* ------------------------------------------------------------------------- */
/* Fixture limits                                                            */
/* ------------------------------------------------------------------------- */

#define MAX_CORES 128u
#define MAX_BANKS 8u
#define MAX_PORTS 3u
#define MAX_REPETITIONS 5u
#define MAX_CONDITIONS 420u

#define RUN_TIMEOUT_MS 60000u

/** Where every reader's walk starts inside its bank. 128 MiB in, so nothing this suite
 *  touches can collide with the low addresses other suites stage into (the multicast bench
 *  uses 0x02000000, test_phase2_dram the allocator's first arena). */
#define DRAM_WALK_BASE 0x08000000u

/** Ceiling for base + regions. A Blackhole bank is MEM_DRAM_SIZE = 0xFF000000 and the
 *  kernel takes a 32-bit address, so this is the usable end with a margin. */
#define DRAM_WALK_LIMIT 0xF0000000u

/** L1 destination window, relative to the derived kernel-config base. The Phase 2 layout's
 *  own regions end at +0x10000; the multicast bench uses +0x40000..+0x100000 for the same
 *  reason. 1 MiB is far more than any condition here needs. */
#define L1_DESTINATION_OFFSET 0x040000u
#define L1_DESTINATION_BUDGET 0x100000u

/** Control-word offsets inside the Phase 2 probe region. */
#define STATUS_OFFSET 0x000u
/** Where a reader of ANOTHER core's L1 walks (stage `place`, L1-source arms): above the
 *  destination slots this program writes, inside the 1.5 MiB L1. Reads only. */
#define L1_SOURCE_OFFSET 0x0C0000u
#define L1_SOURCE_SPAN 0x080000uLL

/* ------------------------------------------------------------------------- */
/* One condition of the sweep                                                */
/* ------------------------------------------------------------------------- */

/**
 * One measurable configuration. Every axis the contradiction confounds is a field here,
 * and the sweep tables below differ in ONE of them at a time wherever that is possible.
 */
typedef struct sweep_condition {
    char label[48];
    uint32_t core_count;
    uint32_t bank_count;     /**< banks 0 .. bank_count-1 are used, round-robin */
    uint32_t ports_per_bank; /**< 1..3; port 0 only is what every earlier round used */
    uint32_t shared;         /**< 1: every core reads the SAME addresses; 0: disjoint */
    uint64_t working_set;    /**< bytes walked before wrapping, per core (or shared) */
    uint32_t block_bytes;    /**< one NoC read request */
    uint32_t depth;          /**< reads issued per pass */
    uint32_t barrier_every;  /**< reads between barriers; 1 is the serialised falsifier */
    uint64_t target_bytes;   /**< timed bytes per core, sets the iteration count */
    uint32_t duration_ms;    /**< non-zero: run a fixed WALL-CLOCK window instead, so every
                              *   core stops at the same moment and the tail of a run with
                              *   unequal per-core rates cannot bias the aggregate either
                              *   way. The sweeps use this; the calibration and the byte
                              *   proof cannot, and say why at their call sites. */
    /* --- PLACEMENT AND ENDPOINT SELECTION (added for the cliff/NoC round) -------------
     * Without these a condition can only ever use booted slots 0..core_count-1 against
     * banks 0..bank_count-1 and ports 0..ports_per_bank-1, which cannot express either
     * "the same reader against a different DRAM endpoint" (the NoC-distance question) or
     * "one endpoint per bank, but a DIFFERENT one per bank" (the port-0 question).
     * All four default to 0, so every pre-existing condition is unchanged. */
    uint32_t core_offset;  /**< first booted slot this condition uses */
    uint32_t core_stride;  /**< 0 or 1: contiguous booted slots; k: every k-th */
    uint32_t bank_offset;  /**< first bank used */
    uint32_t bank_stride;  /**< 0 or 1: consecutive banks; k: bank_offset + i*k. Banks 0-3
                            *   sit in one NoC column and 4-7 in the other, and port p of
                            *   bank b and of bank b+4 share a NoC ROW — neither of which a
                            *   consecutive bank range can select against */
    uint32_t port_offset;  /**< ports used are port_offset .. +ports_per_bank-1 */
    uint32_t direction;    /**< 0: DRAM->L1 reads. 1: L1->DRAM writes */
    uint32_t base_offset;  /**< added to DRAM_WALK_BASE, so a write sweep cannot land on the
                            *   blocks the read proofs stage */
    uint32_t port_by_bank; /**< 1: ONE port per bank but a different one per bank,
                            *   port = (bank + port_offset) % 3. `ports_per_bank` must be
                            *   1. This is what separates "port 0 is special" from "one
                            *   endpoint per bank is the limit" */
    /**
     * Bytes between consecutive blocks. 0 means `block_bytes` — a contiguous stream, which
     * is what every condition here was before this field existed.
     *
     * It exists to SEPARATE TWO THINGS `working_set` CONFOUNDS. `working_set / block_bytes`
     * is how many DISTINCT ADDRESSES the walk cycles through; `working_set` itself is how
     * many BYTES that cycle spans. At stride == block_bytes those move together, so the
     * 64 KiB anomaly could be either "only 4 distinct addresses" or "only 64 KiB of DRAM
     * touched". With a stride the block COUNT is held at 4 and the SPAN is moved, and the
     * two hypotheses give opposite answers.
     */
    uint32_t stride_bytes;
    /* --- EXPLICIT READER MAP (tt-place lane, stage `place`) ---------------------------
     * Non-NULL: reader i runs on booted slot `map_slot[i]` and reads bank `map_bank[i]`
     * through port `map_port[i]`, in its own disjoint region i. This is what expresses a
     * PRODUCTION placement (the rotation, nearest-bank, farthest-bank) reader by reader,
     * which the round-robin axes above cannot. With `source_is_core` the "bank" index is a
     * BOOTED SLOT whose L1 is read instead of DRAM (the L1-staging question). */
    const uint8_t *map_slot;
    const uint8_t *map_bank;
    const uint8_t *map_port;
    uint32_t source_is_core;
    const uint8_t *map_core; /**< per reader: 1 = `map_bank[i]` is a booted slot (its L1) */
} sweep_condition;

/** Does reader i of this condition read another core's L1 rather than a DRAM bank? */
static bool reader_reads_core(const sweep_condition *condition, uint32_t index)
{
    return condition->source_is_core || (condition->map_core && condition->map_core[index]);
}

/** What one core reported. */
typedef struct core_reading {
    uint32_t slot; /**< which BOOTED slot ran it — not the condition's own index */
    uint32_t bank;
    uint32_t port;
    uint32_t error;
    uint32_t passes;
    uint32_t start_clock;
    uint32_t end_clock;
    uint32_t elapsed_cycles;
    uint32_t final_walk_offset;
    uint32_t timed_blocks;
    double gigabytes_per_second;
} core_reading;

/** What one repetition of one condition produced. */
typedef struct run_result {
    int rc;
    uint32_t first_error;
    uint32_t core_count;
    core_reading core[MAX_CORES];
    double aggregate;                    /**< sum of the per-core rates */
    double window;                       /**< total bytes / the common overlap window */
    double bank[MAX_BANKS];
    double port[MAX_BANKS][MAX_PORTS];
    double slowest_core;
    double fastest_core;
    double host_seconds;
} run_result;

typedef struct capacity_fixture {
    ft_tt_hardware_session *session;
    /**
     * The host's DRAM data accessor — `runtime.bulk_dram`, WC/BLOCK16, borrowed from the
     * session rather than opened here.
     *
     * It used to be an `ft_tt_noc_path` this file opened for itself, which is a tt-kmd
     * type: on the userspace driver the bulk path is an `ft_tt_udrv_noc_path` inside the
     * opaque udrv runtime, so the whole suite refused with "run it with FT_TT_DRIVER=kmd".
     * That is unrunnable on a board where tt-kmd is never installed. The accessor vtable is
     * wired on BOTH transports and is the same WC/BLOCK16 window either way, so taking it
     * instead of the path makes the suite transport-agnostic and changes nothing on kmd.
     */
    const ft_tt_noc_access *bulk;
    const ft_tt_firmware_set *firmware;
    const ft_tt_program_directory *program;
    ft_tt_noc_coord core[MAX_CORES];
    ft_tt_noc_coord core_noc0[MAX_CORES];
    uint32_t core_index[MAX_CORES];
    ft_tt_launch_config config[MAX_CORES];
    uint32_t core_count;
    ft_tt_noc_coord dram[MAX_BANKS][MAX_PORTS];
    ft_tt_noc_coord dram_noc0[MAX_BANKS][MAX_PORTS];
    ft_tt_phase2_l1_layout layout;
    double cycles_per_second;
    uint32_t repetitions;
    uint32_t first_index; /**< live Tensix index of booted slot 0 — the distance sweep names
                           *   grid positions, and this is what turns one into a slot */
} capacity_fixture;

/* ------------------------------------------------------------------------- */
/* Small helpers                                                             */
/* ------------------------------------------------------------------------- */

/**
 * Host write into a DRAM bank through the session's bulk accessor.
 *
 * The replacement for ft_tt_write_dram() on a fixture that no longer owns an
 * ft_tt_noc_path. Same window, same BLOCK16 width, same 16 B alignment rules — those live
 * under the accessor on both transports, not in the helper this used to call.
 */
static int fixture_write_dram(const capacity_fixture *fixture, ft_tt_noc_coord bank_port,
                              uint64_t address, const void *src, size_t bytes)
{
    return fixture->bulk->write(fixture->bulk->user, (uint16_t)bank_port.x,
                                (uint16_t)bank_port.y, address, src, bytes);
}

/** Host read from a DRAM bank through the session's bulk accessor. See above. */
static int fixture_read_dram(const capacity_fixture *fixture, ft_tt_noc_coord bank_port,
                             uint64_t address, void *dst, size_t bytes)
{
    return fixture->bulk->read(fixture->bulk->user, (uint16_t)bank_port.x,
                               (uint16_t)bank_port.y, address, dst, bytes);
}

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

static int compare_double(const void *left, const void *right)
{
    double a = *(const double *)left;
    double b = *(const double *)right;

    return (a > b) - (a < b);
}

/** The board's AICLK, READ rather than assumed. `tt_aiclk` is 800 MHz measured busy on
 *  this card, and a run that straddles a change in it is void the same way a run that
 *  straddles a link-speed change is. */
static unsigned read_board_aiclk_mhz(void)
{
    FILE *file = fopen("/sys/class/tenstorrent/tenstorrent!0/tt_aiclk", "r");
    unsigned value = 0;

    if (file == NULL)
        return 0;
    if (fscanf(file, "%u", &value) != 1)
        value = 0;
    fclose(file);
    return value;
}

/** One-minute load average. `pantheon` serves on the K80s of this same 2-core box. */
static double read_host_load(void)
{
    FILE *file = fopen("/proc/loadavg", "r");
    double value = -1.0;

    if (file == NULL)
        return -1.0;
    if (fscanf(file, "%lf", &value) != 1)
        value = -1.0;
    fclose(file);
    return value;
}

/* ------------------------------------------------------------------------- */
/* Addressing                                                                */
/* ------------------------------------------------------------------------- */

/** Which bank and port a slot reads, and which disjoint region inside that bank. */
static void assign_core(const sweep_condition *condition, uint32_t slot, uint32_t *out_bank,
                        uint32_t *out_port, uint32_t *out_region)
{
    uint32_t within = slot / condition->bank_count;

    if (condition->map_bank) {
        *out_bank = condition->map_bank[slot];
        *out_port = condition->map_port ? condition->map_port[slot] : 0u;
        *out_region = slot;
        return;
    }
    uint32_t stride = condition->bank_stride == 0u ? 1u : condition->bank_stride;
    uint32_t bank = condition->bank_offset + (slot % condition->bank_count) * stride;

    *out_bank = bank;
    if (condition->port_by_bank)
        *out_port = (bank + condition->port_offset) % MAX_PORTS;
    else
        *out_port = condition->port_offset + within % condition->ports_per_bank;
    *out_region = within;
}

/** Which BOOTED core slot runs the condition's `index`-th reader. Default (offset 0,
 *  stride 0) is the contiguous behaviour every pre-existing condition had. */
static uint32_t fixture_slot_for(const sweep_condition *condition, uint32_t index)
{
    uint32_t stride = condition->core_stride == 0u ? 1u : condition->core_stride;

    if (condition->map_slot)
        return condition->map_slot[index];
    return condition->core_offset + index * stride;
}

/** Hop counts from a reader to its DRAM endpoint in NOC 0 space, on the torus this NoC
 *  is, reported per axis because a dimension-ordered router treats them separately.
 *  DERIVED from ft_tt_coord's own coordinates — never probed. */
static void noc0_hops(ft_tt_noc_coord from, ft_tt_noc_coord to, uint32_t size_x,
                      uint32_t size_y, uint32_t *out_x, uint32_t *out_y)
{
    uint32_t dx = (to.x + size_x - from.x) % size_x;
    uint32_t dy = (to.y + size_y - from.y) % size_y;

    *out_x = dx < size_x - dx ? dx : size_x - dx;
    *out_y = dy < size_y - dy ? dy : size_y - dy;
}

/** Bytes between consecutive blocks. `stride_bytes == 0` is the contiguous stream every
 *  condition used before the stride axis existed. */
static uint32_t condition_stride(const sweep_condition *condition)
{
    return condition->stride_bytes == 0u ? condition->block_bytes : condition->stride_bytes;
}

/** How many DISTINCT addresses the walk cycles through before returning to its base. */
static uint32_t condition_wrap_blocks(const sweep_condition *condition)
{
    return (uint32_t)(condition->working_set / condition->block_bytes);
}

/** How many BYTES that cycle spans — `wrap_blocks * stride`, which equals the working set
 *  only for a contiguous stream. This is the figure the address space has to hold. */
static uint64_t condition_span(const sweep_condition *condition)
{
    return (uint64_t)condition_wrap_blocks(condition) * condition_stride(condition);
}

/** Bytes reserved per disjoint region — the SPAN, rounded up to 1 MiB so two regions can
 *  never share a row even at a tiny working set. */
static uint64_t region_bytes(const sweep_condition *condition)
{
    uint64_t rounded = (condition_span(condition) + 0xFFFFFuLL) & ~0xFFFFFuLL;

    return rounded < 0x100000uLL ? 0x100000uLL : rounded;
}

/** Where one core's walk starts. Shared mode gives every core the same answer, which is
 *  the whole point of the axis. */
static uint64_t core_walk_base(const sweep_condition *condition, uint32_t region)
{
    uint64_t base = (uint64_t)DRAM_WALK_BASE + condition->base_offset;

    if (condition->shared)
        return base;
    return base + (uint64_t)region * region_bytes(condition);
}

/** Every address the condition will touch fits under the 32-bit runtime argument. */
static bool condition_fits(const sweep_condition *condition, uint32_t core_count)
{
    uint32_t regions = (core_count + condition->bank_count - 1u) / condition->bank_count;
    uint64_t highest;

    if (condition->source_is_core)
        return condition_span(condition) <= L1_SOURCE_SPAN;
    if (condition->map_core && condition_span(condition) > L1_SOURCE_SPAN)
        return false;
    if (condition->map_bank)
        regions = core_count;
    if (condition->shared)
        regions = 1u;
    highest = core_walk_base(condition, regions - 1u) + condition_span(condition);
    return highest <= (uint64_t)DRAM_WALK_LIMIT;
}

/**
 * Every core slot, bank and port this condition names exists.
 *
 * A condition that reaches past the booted core set, past bank 7 or past port 2 must be
 * SKIPPED rather than clamped: clamping would silently answer a different question, and
 * these axes exist precisely to ask which endpoint was read.
 */
static bool condition_is_expressible(const sweep_condition *condition, uint32_t booted_cores)
{
    if (condition->core_count == 0u || condition->bank_count == 0u ||
        condition->ports_per_bank == 0u)
        return false;
    if (condition->map_bank) {
        uint32_t i;

        for (i = 0u; i < condition->core_count; i++)
            if (fixture_slot_for(condition, i) >= booted_cores ||
                (reader_reads_core(condition, i) ? condition->map_bank[i] >= booted_cores
                                                 : condition->map_bank[i] >= MAX_BANKS) ||
                (condition->map_port && condition->map_port[i] >= MAX_PORTS))
                return false;
        return true;
    }
    if (fixture_slot_for(condition, condition->core_count - 1u) >= booted_cores)
        return false;
    if (condition->bank_offset +
            (condition->bank_count - 1u) * (condition->bank_stride == 0u ? 1u :
                                            condition->bank_stride) >= MAX_BANKS)
        return false;
    if (condition->port_by_bank)
        return condition->ports_per_bank == 1u;
    return condition->port_offset + condition->ports_per_bank <= MAX_PORTS;
}

static uint32_t slot_count_for(const sweep_condition *condition)
{
    uint32_t slots = L1_DESTINATION_BUDGET / condition->block_bytes;

    if (slots > condition->depth)
        slots = condition->depth;
    if (slots > 8u)
        slots = 8u;
    return slots == 0u ? 1u : slots;
}

static uint32_t iterations_for(const sweep_condition *condition)
{
    uint64_t per_pass = (uint64_t)condition->depth * (uint64_t)condition->block_bytes;
    uint64_t iterations = condition->target_bytes / per_pass;

    if (iterations < 1u)
        iterations = 1u;
    if (iterations > 4000000uLL)
        iterations = 4000000uLL;
    return (uint32_t)iterations;
}

/* ------------------------------------------------------------------------- */
/* Boot and launch                                                           */
/* ------------------------------------------------------------------------- */

static void build_config(const ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                         const ft_tt_kernel_program *program,
                         const ft_tt_phase2_l1_layout *layout, ft_tt_launch_config *out)
{
    unsigned processor;

    memset(out, 0, sizeof(*out));
    out->kernel_config_base = layout->kernel_config_base;
    out->enables = ft_tt_kernel_program_enables(program);
    out->brisc_noc_id = 0u;
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if (placements[processor].bytes_written == 0u)
            continue;
        out->kernel_text_address[processor] = placements[processor].entry_address;
        out->kernel_text_bytes[processor] = placements[processor].text_bytes;
        out->runtime_args_address[processor] =
            ft_tt_phase2_runtime_args_address(layout, processor);
    }
}

/**
 * Bring one core from unknown to "firmware running, program loaded".
 *
 * SEQUENTIAL by construction — one process, one core at a time. At 12-way and above,
 * firmware boots from INDEPENDENT host processes start returning `device I/O failed` and
 * leave cores mid-handshake; it wedged cores 110 and 111 during a DRAM fan-out.
 * This program is one process and never overlaps a boot.
 */
static int boot_core(capacity_fixture *fixture, uint32_t slot)
{
    ft_tt_image_placement firmware_placements[FT_TT_PROC_COUNT];
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    const ft_tt_noc_access *noc = &fixture->session->noc;
    const uint16_t x = (uint16_t)fixture->core[slot].x;
    const uint16_t y = (uint16_t)fixture->core[slot].y;
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

static uint32_t status_address(const capacity_fixture *fixture)
{
    return fixture->layout.probe_address + STATUS_OFFSET;
}

static uint32_t destination_address(const capacity_fixture *fixture)
{
    return fixture->layout.kernel_config_base + L1_DESTINATION_OFFSET;
}

static int plant_arguments(const capacity_fixture *fixture, uint32_t slot,
                           const sweep_condition *condition, uint32_t iterations,
                           ft_tt_noc_coord dram, uint64_t walk_base)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    const uint16_t x = (uint16_t)fixture->core[slot].x;
    const uint16_t y = (uint16_t)fixture->core[slot].y;
    uint32_t arguments[KERNEL_ARG_COUNT];
    uint32_t zero = 0u;
    unsigned processor;
    int rc;

    arguments[0] = iterations;
    arguments[1] = condition->depth;
    arguments[2] = condition->barrier_every;
    arguments[3] = condition->block_bytes;
    arguments[4] = dram.x;
    arguments[5] = dram.y;
    arguments[6] = (uint32_t)walk_base;
    arguments[7] = condition_stride(condition); /* == block_bytes unless the arm says else */
    arguments[8] = condition_wrap_blocks(condition);
    arguments[9] = destination_address(fixture);
    arguments[10] = slot_count_for(condition);
    arguments[11] = condition->block_bytes;
    arguments[12] = status_address(fixture);
    arguments[13] = (uint32_t)((double)condition->duration_ms * 1e-3 *
                               fixture->cycles_per_second);
    arguments[14] = condition->direction;

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

static int poison_status(const capacity_fixture *fixture, uint32_t slot)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    uint32_t poison[STATUS_WORDS];
    unsigned index;

    for (index = 0u; index < STATUS_WORDS; index++)
        poison[index] = STATUS_POISON;
    return noc->write(noc->user, (uint16_t)fixture->core[slot].x,
                      (uint16_t)fixture->core[slot].y, status_address(fixture), poison,
                      sizeof(poison));
}

static int read_status(const capacity_fixture *fixture, uint32_t slot, uint32_t *out)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;

    return noc->read(noc->user, (uint16_t)fixture->core[slot].x,
                     (uint16_t)fixture->core[slot].y, status_address(fixture), out,
                     STATUS_WORDS * sizeof(uint32_t));
}

/** Stage the whole group, kick the whole group, wait for all of it. The group is the
 *  condition's SELECTED slots, which are contiguous only by default. */
static int launch_and_wait(const capacity_fixture *fixture, const sweep_condition *condition)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    uint32_t index;
    uint32_t slot;
    int rc = FT_TT_OK;

    for (index = 0u; index < condition->core_count && rc == FT_TT_OK; index++) {
        slot = fixture_slot_for(condition, index);
        rc = ft_tt_launch_write_launch_message(noc, (uint16_t)fixture->core[slot].x,
                                               (uint16_t)fixture->core[slot].y,
                                               &fixture->config[slot]);
    }
    for (index = 0u; index < condition->core_count && rc == FT_TT_OK; index++) {
        slot = fixture_slot_for(condition, index);
        rc = ft_tt_launch_send_go(noc, (uint16_t)fixture->core[slot].x,
                                  (uint16_t)fixture->core[slot].y);
    }
    for (index = 0u; index < condition->core_count && rc == FT_TT_OK; index++) {
        slot = fixture_slot_for(condition, index);
        rc = ft_tt_launch_wait_done(noc, (uint16_t)fixture->core[slot].x,
                                    (uint16_t)fixture->core[slot].y, RUN_TIMEOUT_MS, NULL);
    }
    return rc;
}

/* ------------------------------------------------------------------------- */
/* One run of one condition                                                  */
/* ------------------------------------------------------------------------- */

/**
 * Stage, launch and score one condition once.
 *
 * The per-core rate is that core's own bytes over its own elapsed cycles, so launch skew
 * across the group changes how much the cores OVERLAP — a real property of the arm — and
 * not the measurement. `window` is the cross-check: total bytes over the union of every
 * core's timed interval, on the common device clock. The two agreeing is what says the
 * cores really were concurrent; they are both reported and the verdict uses the lower.
 */
static int run_condition(capacity_fixture *fixture, const sweep_condition *condition,
                         run_result *out)
{
    const uint32_t iterations = iterations_for(condition);
    const uint32_t core_count = condition->core_count;
    uint64_t total_bytes = 0u;
    uint32_t min_start = 0xFFFFFFFFu;
    uint32_t max_end = 0u;
    double started;
    uint32_t index;
    int rc;

    memset(out, 0, sizeof(*out));
    out->core_count = core_count;
    out->slowest_core = 1e30;

    for (index = 0u; index < core_count; index++) {
        uint32_t bank;
        uint32_t port;
        uint32_t region;
        uint32_t slot = fixture_slot_for(condition, index);

        assign_core(condition, index, &bank, &port, &region);
        out->core[index].slot = slot;
        out->core[index].bank = bank;
        out->core[index].port = port;
        rc = poison_status(fixture, slot);
        if (rc == FT_TT_OK && reader_reads_core(condition, index))
            rc = plant_arguments(fixture, slot, condition, iterations, fixture->core[bank],
                                 fixture->layout.kernel_config_base + L1_SOURCE_OFFSET);
        else if (rc == FT_TT_OK)
            rc = plant_arguments(fixture, slot, condition, iterations,
                                 fixture->dram[bank][port], core_walk_base(condition, region));
        if (rc != FT_TT_OK) {
            out->rc = rc;
            return rc;
        }
    }

    started = monotonic_seconds();
    out->rc = launch_and_wait(fixture, condition);
    out->host_seconds = monotonic_seconds() - started;
    if (out->rc != FT_TT_OK)
        return out->rc;

    for (index = 0u; index < core_count; index++) {
        uint32_t status[STATUS_WORDS];
        core_reading *reading = &out->core[index];
        double seconds;

        rc = read_status(fixture, reading->slot, status);
        if (rc != FT_TT_OK) {
            out->rc = rc;
            return rc;
        }
        if (status[STATUS_MAGIC_WORD] != STATUS_MAGIC) {
            out->first_error = status[STATUS_MAGIC_WORD];
            return FT_TT_ERROR;
        }
        reading->error = status[STATUS_ERROR];
        reading->passes = status[STATUS_PASSES_COMPLETED];
        reading->start_clock = status[STATUS_START_CLOCK];
        reading->end_clock = status[STATUS_END_CLOCK];
        reading->elapsed_cycles = status[STATUS_ELAPSED_CYCLES];
        reading->final_walk_offset = status[STATUS_FINAL_WALK_OFFSET];
        reading->timed_blocks = status[STATUS_BLOCKS_ISSUED];
        if (reading->error != 0u && out->first_error == 0u)
            out->first_error = reading->error;
        if (condition->duration_ms == 0u && reading->passes != iterations + 1u &&
            out->first_error == 0u)
            out->first_error = 0xDEAD0000u | (reading->passes & 0xFFFFu);
        if (reading->elapsed_cycles == 0u || reading->timed_blocks == 0u) {
            out->first_error = 0xC10C0000u;
            continue;
        }
        seconds = (double)reading->elapsed_cycles / fixture->cycles_per_second;
        reading->gigabytes_per_second =
            (double)reading->timed_blocks * condition->block_bytes / seconds / 1e9;
        total_bytes += (uint64_t)reading->timed_blocks * condition->block_bytes;
        out->aggregate += reading->gigabytes_per_second;
        if (!reader_reads_core(condition, index)) {
            out->bank[reading->bank] += reading->gigabytes_per_second;
            out->port[reading->bank][reading->port] += reading->gigabytes_per_second;
        }
        if (reading->gigabytes_per_second < out->slowest_core)
            out->slowest_core = reading->gigabytes_per_second;
        if (reading->gigabytes_per_second > out->fastest_core)
            out->fastest_core = reading->gigabytes_per_second;
        if (reading->start_clock < min_start)
            min_start = reading->start_clock;
        if (reading->end_clock > max_end)
            max_end = reading->end_clock;
    }

    if (max_end > min_start)
        out->window = (double)total_bytes /
                      ((double)(max_end - min_start) / fixture->cycles_per_second) / 1e9;
    return out->first_error == 0u ? FT_TT_OK : FT_TT_ERROR;
}

/* ------------------------------------------------------------------------- */
/* The sweep driver                                                          */
/* ------------------------------------------------------------------------- */

/** Aggregated over repetitions. `best` is the whole result of the fastest repetition, kept
 *  because the contradiction is stated PER BANK and per-bank numbers cannot be recovered
 *  from an aggregate. */
typedef struct condition_summary {
    double aggregate[MAX_REPETITIONS];
    double window[MAX_REPETITIONS];
    run_result best;
    uint32_t completed;
    uint32_t first_error;
    int rc;
} condition_summary;

/** Per-bank and per-port breakdown of the fastest repetition of every condition. The
 *  contradiction being settled is stated per bank, so it has to be answered per bank. */
static void print_per_bank_table(const sweep_condition *conditions,
                                 const condition_summary *summary, uint32_t count,
                                 uint32_t repetitions)
{
    uint32_t entry;
    uint32_t bank;

    printf("  per-bank GB/s, fastest repetition of each condition\n");
    printf("  %-38s %8s |", "condition", "total");
    for (bank = 0u; bank < MAX_BANKS; bank++)
        printf(" %6u", bank);
    printf("\n  %s\n", "-------------------------------------------------------------------"
                       "----------------------------------");
    for (entry = 0u; entry < count; entry++) {
        if (summary[entry].completed != repetitions)
            continue;
        printf("  %-38s %8.1f |", conditions[entry].label, summary[entry].best.aggregate);
        for (bank = 0u; bank < MAX_BANKS; bank++)
            printf(" %6.1f", summary[entry].best.bank[bank]);
        printf("\n");
    }
    printf("\n  per-PORT GB/s inside bank 0 (three NoC doors into one memory)\n");
    printf("  %-38s %8s | %6s %6s %6s\n", "condition", "bank0", "port0", "port1", "port2");
    printf("  %s\n", "-------------------------------------------------------------------"
                     "--");
    for (entry = 0u; entry < count; entry++) {
        if (summary[entry].completed != repetitions)
            continue;
        printf("  %-38s %8.1f | %6.1f %6.1f %6.1f\n", conditions[entry].label,
               summary[entry].best.bank[0], summary[entry].best.port[0][0],
               summary[entry].best.port[0][1], summary[entry].best.port[0][2]);
    }
    printf("\n");
    fflush(stdout);
}

/**
 * Every reader of one condition, with WHERE it sat and WHICH endpoint it read.
 *
 * The aggregate cannot distinguish "every core slowed down by the same fraction" from "most
 * cores kept their rate and a few starved", and those are different mechanisms — a shared
 * resource divided fairly against an allocation that some requesters lose. The spread is
 * the whole point, so this prints every core rather than a summary of them.
 */
static void print_per_core_table(const capacity_fixture *fixture,
                                 const sweep_condition *condition, const run_result *result)
{
    uint32_t size_x = 0u;
    uint32_t size_y = 0u;
    uint32_t index;

    if (condition->source_is_core || condition->map_core)
        return;
    (void)ft_tt_coord_noc_size(&fixture->session->grid, &size_x, &size_y);
    printf("    per-core: %s\n", condition->label);
    printf("    %5s %6s %10s %5s %5s %9s %5s %5s %8s\n", "index", "slot", "core noc0", "bank",
           "port", "dram noc0", "hop x", "hop y", "GB/s");
    for (index = 0u; index < result->core_count; index++) {
        const core_reading *reading = &result->core[index];
        ft_tt_noc_coord core = fixture->core_noc0[reading->slot];
        ft_tt_noc_coord dram = fixture->dram_noc0[reading->bank][reading->port];
        uint32_t hop_x = 0u;
        uint32_t hop_y = 0u;

        noc0_hops(core, dram, size_x, size_y, &hop_x, &hop_y);
        printf("    %5u %6u %5u,%-4u %5u %5u %4u,%-4u %5u %5u %8.2f\n",
               fixture->core_index[reading->slot], reading->slot, core.x, core.y,
               reading->bank, reading->port, dram.x, dram.y, hop_x, hop_y,
               reading->gigabytes_per_second);
    }
    printf("\n");
    fflush(stdout);
}

/**
 * Run `count` conditions `repetitions` times ROUND-ROBIN and report best/median/worst.
 *
 * Round-robin rather than condition-at-a-time is the method: this host is two 1.8 GHz cores
 * also driving eight K80s, and measuring one condition to completion and then the next once
 * reported a NEGATIVE cost for CONFIGURE_TLB. Every condition therefore
 * drifts through the same host weather.
 */
/** The median aggregate of every condition of the LAST run_sweep(), 0 when incomplete —
 *  what the placement stage builds its (core, bank) matrices from. */
static double g_sweep_median[MAX_CONDITIONS];
/** And the per-reader rates of the fastest repetition, for the per-reader map columns. */
static double g_sweep_reader_best[MAX_CONDITIONS][32];

static void run_sweep(capacity_fixture *fixture, const char *title,
                      const sweep_condition *conditions, uint32_t count)
{
    static condition_summary summary[MAX_CONDITIONS];
    uint32_t repetition;
    uint32_t entry;

    ft_tt_test_step("%s — %u conditions x %u repetitions, INTERLEAVED", title, count,
                    fixture->repetitions);
    memset(summary, 0, sizeof(summary));
    memset(g_sweep_median, 0, sizeof(g_sweep_median));
    memset(g_sweep_reader_best, 0, sizeof(g_sweep_reader_best));

    for (repetition = 0u; repetition < fixture->repetitions; repetition++) {
        double load_before = read_host_load();
        unsigned aiclk_before = read_board_aiclk_mhz();
        double load_after;
        unsigned aiclk_after;

        for (entry = 0u; entry < count; entry++) {
            const sweep_condition *condition = &conditions[entry];
            run_result result;
            int rc;

            if (!condition_is_expressible(condition, fixture->core_count) ||
                !condition_fits(condition, condition->core_count)) {
                summary[entry].rc = FT_TT_EINVAL;
                continue;
            }
            rc = run_condition(fixture, condition, &result);
            if (rc != FT_TT_OK) {
                summary[entry].rc = rc;
                if (summary[entry].first_error == 0u)
                    summary[entry].first_error = result.first_error;
                continue;
            }
            summary[entry].aggregate[repetition] = result.aggregate;
            summary[entry].window[repetition] = result.window;
            if (result.aggregate > summary[entry].best.aggregate)
                summary[entry].best = result;
            summary[entry].completed++;
        }
        load_after = read_host_load();
        aiclk_after = read_board_aiclk_mhz();
        printf("  repetition %u: host load %.2f -> %.2f, AICLK %u -> %u MHz\n", repetition,
               load_before, load_after, aiclk_before, aiclk_after);
        fflush(stdout);
    }

    printf("\n  %-38s %5s %5s %6s | %8s %8s %8s | %8s | %8s %8s\n", "condition", "cores",
           "banks", "ports", "best", "median", "worst", "window", "per-core", "slowest");
    printf("  %s\n", "---------------------------------------------------------------------"
                     "--------------------------------------------------");
    for (entry = 0u; entry < count; entry++) {
        const sweep_condition *condition = &conditions[entry];
        condition_summary *row = &summary[entry];
        double sorted[MAX_REPETITIONS];
        double windows[MAX_REPETITIONS];
        uint32_t n = fixture->repetitions;

        if (row->completed != n) {
            printf("  %-38s %5u %5u %6u | INCOMPLETE %u/%u, rc=%d, first error 0x%08x\n",
                   condition->label, condition->core_count, condition->bank_count,
                   condition->ports_per_bank, row->completed, n, row->rc, row->first_error);
            continue;
        }
        memcpy(sorted, row->aggregate, sizeof(double) * n);
        memcpy(windows, row->window, sizeof(double) * n);
        qsort(sorted, n, sizeof(double), compare_double);
        qsort(windows, n, sizeof(double), compare_double);
        g_sweep_median[entry] = sorted[n / 2u];
        {
            uint32_t r;

            for (r = 0u; r < row->best.core_count && r < 32u; r++)
                g_sweep_reader_best[entry][r] = row->best.core[r].gigabytes_per_second;
        }
        printf("  %-38s %5u %5u %6u | %8.1f %8.1f %8.1f | %8.1f | %8.2f %8.2f\n",
               condition->label, condition->core_count, condition->bank_count,
               condition->ports_per_bank, sorted[n - 1u], sorted[n / 2u], sorted[0],
               windows[n / 2u], sorted[n - 1u] / (double)condition->core_count,
               row->best.slowest_core);
    }
    printf("\n  Aggregate GB/s is the SUM of the per-core rates, each core timed on its own\n"
           "  wall clock. `window` is the same bytes over the union of every core's timed\n"
           "  interval and is the concurrency cross-check. `per-core` is best/cores and\n"
           "  `slowest` is the slowest single core of the fastest repetition — the spread\n"
           "  a mean would hide. best/median/worst are over %u repetitions.\n\n",
           fixture->repetitions);
    print_per_bank_table(conditions, summary, count, fixture->repetitions);

    if (environment_u32("FT_TT_DRAMCAP_PERCORE", 0u) != 0u)
        for (entry = 0u; entry < count; entry++)
            if (summary[entry].completed == fixture->repetitions)
                print_per_core_table(fixture, &conditions[entry], &summary[entry].best);
}

/* ------------------------------------------------------------------------- */
/* Condition builders                                                        */
/* ------------------------------------------------------------------------- */

static void set_condition(sweep_condition *condition, const char *label, uint32_t cores,
                          uint32_t banks, uint32_t ports, uint32_t shared,
                          uint64_t working_set, uint32_t block_bytes, uint32_t depth,
                          uint32_t barrier_every, uint64_t target_bytes)
{
    memset(condition, 0, sizeof(*condition));
    (void)snprintf(condition->label, sizeof(condition->label), "%s", label);
    condition->core_count = cores;
    condition->bank_count = banks;
    condition->ports_per_bank = ports;
    condition->shared = shared;
    condition->working_set = working_set;
    condition->block_bytes = block_bytes;
    condition->depth = depth;
    condition->barrier_every = barrier_every;
    condition->target_bytes = target_bytes;
}

#define MIB (1024uLL * 1024uLL)
#define KIB 1024uLL

/** The default streaming shape: 16 KiB requests (one NOC_MAX_BURST_SIZE, so the request
 *  count cannot be the wall), 32 in flight, 2 GiB of timed traffic per core. */
#define STREAM_BLOCK 16384u
#define STREAM_DEPTH 32u
#define STREAM_TARGET (2048uLL * MIB)

/* ------------------------------------------------------------------------- */
/* Stages                                                                    */
/* ------------------------------------------------------------------------- */

/**
 * Stage 1 — the transfer path, before anything is scored through it.
 *
 * A host->DRAM->host round trip at the walk base of every bank, so "the transfer path is
 * fine" is a line in the report rather than an assumption, and so a bank that answers
 * wrong is found before a bandwidth number is attributed to it.
 */
static void run_stage_transfer_round_trip(capacity_fixture *fixture)
{
    static uint8_t written[4096];
    static uint8_t readback[4096];
    uint32_t bank;
    uint32_t index;

    ft_tt_test_step("stage 1 — host->DRAM->host round trip at the walk base of all %u banks",
                    MAX_BANKS);
    for (index = 0u; index < sizeof(written); index++)
        written[index] = (uint8_t)(index * 7u + 13u);

    for (bank = 0u; bank < MAX_BANKS; bank++) {
        int rc;

        written[0] = (uint8_t)(0xB0u + bank); /* so a bank cannot pass on another's bytes */
        memset(readback, SLOT_POISON_BYTE, sizeof(readback));
        rc = fixture_write_dram(fixture, fixture->dram[bank][0], DRAM_WALK_BASE, written,
                                sizeof(written));
        if (rc == FT_TT_OK)
            rc = fixture_read_dram(fixture, fixture->dram[bank][0], DRAM_WALK_BASE, readback,
                                   sizeof(readback));
        ft_tt_test_check_fmt("a bank reads back what the host wrote",
                             rc == FT_TT_OK && memcmp(written, readback, sizeof(written)) == 0,
                             "bank %u port 0 at translated (%u,%u), rc=%d (%s)", bank,
                             fixture->dram[bank][0].x, fixture->dram[bank][0].y, rc,
                             ft_tt_error_string(rc));
    }
}

/**
 * Stage 2 — calibrate the device wall clock against the host's, from TWO run lengths.
 *
 * A single run cannot: the launch, the boot handshake and the untimed pass are inside the
 * host's measurement and none of them is inside the device's. Two lengths cancel every
 * fixed cost on both sides and the ratio of the differences is the clock.
 *
 * This exists because "the card runs at 800 MHz" is a MEASURED busy AICLK, not a statement
 * about what RISCV_DEBUG_REG_WALL_CLOCK_0 counts, and every GB/s below would inherit the
 * assumption.
 */
static void run_stage_clock_calibration(capacity_fixture *fixture)
{
    sweep_condition shorter;
    sweep_condition longer;
    run_result short_result;
    run_result long_result;
    double device_cycles;
    double host_seconds;
    unsigned aiclk = read_board_aiclk_mhz();
    int rc;

    ft_tt_test_step("stage 2 — calibrate the device wall clock from two run lengths on ONE "
                    "core (sysfs tt_aiclk reads %u MHz right now)", aiclk);

    set_condition(&shorter, "calibration short", 1u, 1u, 1u, 0u, 256uLL * MIB, STREAM_BLOCK,
                  STREAM_DEPTH, STREAM_DEPTH, 512uLL * MIB);
    longer = shorter;
    (void)snprintf(longer.label, sizeof(longer.label), "calibration long");
    longer.target_bytes = 2048uLL * MIB;

    rc = run_condition(fixture, &shorter, &short_result);
    if (rc == FT_TT_OK)
        rc = run_condition(fixture, &longer, &long_result);
    ft_tt_test_check_fmt("both calibration runs completed", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    device_cycles =
        (double)long_result.core[0].elapsed_cycles - (double)short_result.core[0].elapsed_cycles;
    host_seconds = long_result.host_seconds - short_result.host_seconds;
    ft_tt_test_check_fmt("the two lengths differ measurably on both clocks",
                         device_cycles > 0.0 && host_seconds > 0.0, "%.0f cycles over %.6f s",
                         device_cycles, host_seconds);
    if (device_cycles <= 0.0 || host_seconds <= 0.0)
        return;

    /**
     * THE MEASUREMENT IS KEPT EVEN WHEN THE CHECK GOES RED, AND THAT IS THE WHOLE POINT.
     *
     * This used to demand 640-960 MHz — 800 MHz +/- 20% — and, on a miss, ASSIGN 800e6 and
     * carry on. Both halves were wrong once the AI clock stopped being 800. foundry asked
     * tt-kmd for the MINIMUM AI clock by accident until 2026-08-15; max is now the default
     * and this card runs at **1350 MHz**. So the calibration measured
     * 1350, the band rejected it, the stale constant was substituted, and every GB/s the
     * suite printed came out at 800/1350 = 0.593x of the truth — while the only line saying
     * so was a check whose text still named 800 MHz as "this card's". A harness that
     * discards its own measurement in favour of the assumption the measurement exists to
     * replace is measuring nothing.
     *
     * The band is now a PLAUSIBILITY range spanning both documented clock states (800 min,
     * 1350 max, +/- 20%), and a miss is reported without being acted on: the two-length
     * difference cancels every fixed cost on both clocks and is the better estimator of the
     * two under any circumstance where they disagree.
     */
    fixture->cycles_per_second = device_cycles / host_seconds;
    ft_tt_test_check_fmt("the wall clock is within 20%% of a documented AI clock state "
                         "(800 MHz min / 1350 MHz max)",
                         fixture->cycles_per_second > 640e6 &&
                             fixture->cycles_per_second < 1620e6,
                         "measured %.1f MHz (%.0f cycles / %.6f s); sysfs says %u MHz",
                         fixture->cycles_per_second / 1e6, device_cycles, host_seconds, aiclk);
}

/**
 * Stage 3 — the bytes really moved, and they came from the address the walk names.
 *
 * Every L1 slot is poisoned; distinct content is staged at the DRAM blocks the last pass
 * will land in; after the run each slot must hold its block. The block index is recomputed
 * on the host in the same closed form the kernel walks, and CROSS-CHECKED against the
 * kernel's published `final_walk_offset` — so a host arithmetic slip shows up as a
 * disagreement rather than as a wrong "pass".
 */
static void run_stage_bytes_moved(capacity_fixture *fixture, uint32_t bank, uint32_t port,
                                  uint32_t core_offset, uint32_t stride_bytes)
{
    enum { PROOF_BLOCK = 1024u, PROOF_SLOTS = 4u, PROOF_CORES = 4u };
    static uint8_t staged[PROOF_SLOTS][PROOF_BLOCK];
    static uint8_t readback[PROOF_BLOCK];
    const ft_tt_noc_access *noc = &fixture->session->noc;
    sweep_condition condition;
    run_result result;
    uint32_t wrap_blocks;
    uint32_t stride;
    uint32_t iterations;
    uint32_t total_reads;
    uint32_t slot;
    uint32_t index;
    int rc;

    ft_tt_test_step("stage 3 — the reads carried DRAM bytes into L1, on %u cores from booted "
                    "slot %u, out of bank %u port %u, block stride %u B",
                    (unsigned)PROOF_CORES, core_offset, bank, port,
                    stride_bytes == 0u ? (unsigned)PROOF_BLOCK : stride_bytes);

    /* Small everything: this stage is a correctness proof, not a timing, and the read-back
     * direction is 4.7 MB/s. A whole working set of 4 blocks makes the closed form easy to
     * read as well as to compute. */
    set_condition(&condition, "bytes-moved proof", PROOF_CORES, 1u, 1u, 1u,
                  (uint64_t)PROOF_SLOTS * PROOF_BLOCK, PROOF_BLOCK, PROOF_SLOTS, PROOF_SLOTS,
                  (uint64_t)PROOF_SLOTS * PROOF_BLOCK * 16u);
    condition.bank_offset = bank;
    condition.port_offset = port;
    condition.core_offset = core_offset;
    condition.stride_bytes = stride_bytes;
    wrap_blocks = condition_wrap_blocks(&condition);
    stride = condition_stride(&condition);
    iterations = iterations_for(&condition);
    total_reads = (iterations + 1u) * condition.depth;

    /* The pattern carries the bank and the port, so a run that read the RIGHT address of
     * the WRONG endpoint fails here instead of passing on another bank's bytes. */
    for (slot = 0u; slot < PROOF_SLOTS; slot++) {
        for (index = 0u; index < PROOF_BLOCK; index++)
            staged[slot][index] =
                (uint8_t)(index * 31u + slot * 97u + bank * 13u + port * 53u + 5u);
        rc = fixture_write_dram(fixture, fixture->dram[bank][port],
                                DRAM_WALK_BASE + (uint64_t)slot * stride, staged[slot],
                                PROOF_BLOCK);
        if (rc != FT_TT_OK) {
            ft_tt_test_check("the proof blocks staged in DRAM", false, ft_tt_error_string(rc));
            return;
        }
    }

    /* Poison the L1 of every booted slot up to and including the ones this condition
     * selects: the slots BELOW core_offset must still be poisoned afterwards, which is what
     * makes the core-selection axis falsifiable rather than assumed. */
    memset(readback, SLOT_POISON_BYTE, sizeof(readback));
    for (slot = 0u; slot < core_offset + PROOF_CORES; slot++) {
        for (index = 0u; index < PROOF_SLOTS; index++) {
            rc = noc->write(noc->user, (uint16_t)fixture->core[slot].x,
                            (uint16_t)fixture->core[slot].y,
                            destination_address(fixture) + index * PROOF_BLOCK, readback,
                            PROOF_BLOCK);
            if (rc != FT_TT_OK) {
                ft_tt_test_check("every L1 slot poisoned", false, ft_tt_error_string(rc));
                return;
            }
        }
    }

    rc = run_condition(fixture, &condition, &result);
    ft_tt_test_check_fmt("the proof run completed", rc == FT_TT_OK, "rc=%d (%s), error 0x%08x",
                         rc, ft_tt_error_string(rc), result.first_error);
    if (rc != FT_TT_OK)
        return;

    total_reads = result.core[0].passes * condition.depth;
    ft_tt_test_check_fmt("the kernel's final walk offset is where the host's arithmetic puts "
                         "it",
                         result.core[0].final_walk_offset ==
                             (total_reads % wrap_blocks) * stride,
                         "kernel 0x%x, host 0x%x (%u reads, wrap %u, stride %u)",
                         result.core[0].final_walk_offset,
                         (total_reads % wrap_blocks) * stride, total_reads, wrap_blocks,
                         stride);

    for (slot = core_offset; slot < core_offset + PROOF_CORES; slot++) {
        uint32_t mismatches = 0u;
        uint32_t poisoned = 0u;

        for (index = 0u; index < PROOF_SLOTS; index++) {
            /* Read number (total_reads - PROOF_SLOTS + index) is the one that last landed
             * in L1 slot ((total_reads - PROOF_SLOTS + index) % PROOF_SLOTS), and it read
             * DRAM block ((that read number) % wrap_blocks). */
            uint32_t read_number = total_reads - PROOF_SLOTS + index;
            uint32_t l1_slot = read_number % PROOF_SLOTS;
            uint32_t dram_block = read_number % wrap_blocks;

            rc = noc->read(noc->user, (uint16_t)fixture->core[slot].x,
                           (uint16_t)fixture->core[slot].y,
                           destination_address(fixture) + l1_slot * PROOF_BLOCK, readback,
                           PROOF_BLOCK);
            if (rc != FT_TT_OK) {
                mismatches++;
                continue;
            }
            if (readback[0] == SLOT_POISON_BYTE && readback[1] == SLOT_POISON_BYTE)
                poisoned++;
            if (memcmp(readback, staged[dram_block], PROOF_BLOCK) != 0)
                mismatches++;
        }
        ft_tt_test_check_fmt("every L1 slot holds the DRAM block the walk names",
                             mismatches == 0u && poisoned == 0u,
                             "core index %u: %u of %u slots wrong, %u still poisoned",
                             fixture->core_index[slot], mismatches, (unsigned)PROOF_SLOTS,
                             poisoned);
    }

    /* And nobody else read anything. Without this the core-selection axis could be ignored
     * entirely — slots 0..PROOF_CORES-1 would do the work, and the check above would still
     * pass whenever core_offset happened to be 0. */
    for (slot = 0u; slot < core_offset; slot++) {
        rc = noc->read(noc->user, (uint16_t)fixture->core[slot].x,
                       (uint16_t)fixture->core[slot].y, destination_address(fixture), readback,
                       PROOF_BLOCK);
        ft_tt_test_check_fmt("a core this condition did NOT name stayed poisoned",
                             rc == FT_TT_OK && readback[0] == SLOT_POISON_BYTE &&
                                 readback[PROOF_BLOCK - 1u] == SLOT_POISON_BYTE,
                             "core index %u: rc=%d, first byte 0x%02x", fixture->core_index[slot],
                             rc, readback[0]);
    }
}

/** Where the write arms walk. Far from everything the read proofs stage, because a write
 *  sweep leaves garbage behind it by design. */
#define DRAM_WRITE_BASE_OFFSET 0x40000000u

/**
 * Stage 3W — the WRITES really moved, and they went to the address the walk names.
 *
 * The mirror of stage 3 and the only thing that makes a write number mean anything: DRAM is
 * poisoned, four L1 slots are staged with distinct content, and afterwards DRAM block i must
 * hold L1 slot i — the walk and the slot round-robin advance together from the same start,
 * so the mapping is the identity and is checkable in closed form.
 *
 * ONE core, deliberately: two cores writing one region would race and the check would be
 * scoring whichever won.
 */
static void run_stage_writes_moved(capacity_fixture *fixture)
{
    enum { PROOF_BLOCK = 1024u, PROOF_SLOTS = 4u };
    static uint8_t staged[PROOF_SLOTS][PROOF_BLOCK];
    static uint8_t readback[PROOF_BLOCK];
    const ft_tt_noc_access *noc = &fixture->session->noc;
    sweep_condition condition;
    run_result result;
    uint32_t slot;
    uint32_t index;
    uint32_t mismatches = 0u;
    uint32_t poisoned = 0u;
    int rc;

    ft_tt_test_step("stage 3W — the WRITES carried L1 bytes into DRAM");

    set_condition(&condition, "bytes-written proof", 1u, 1u, 1u, 0u,
                  (uint64_t)PROOF_SLOTS * PROOF_BLOCK, PROOF_BLOCK, PROOF_SLOTS, PROOF_SLOTS,
                  (uint64_t)PROOF_SLOTS * PROOF_BLOCK * 16u);
    condition.direction = 1u;
    condition.base_offset = DRAM_WRITE_BASE_OFFSET;

    memset(readback, SLOT_POISON_BYTE, sizeof(readback));
    for (index = 0u; index < PROOF_SLOTS; index++) {
        for (slot = 0u; slot < PROOF_BLOCK; slot++)
            staged[index][slot] = (uint8_t)(slot * 17u + index * 61u + 23u);
        rc = fixture_write_dram(fixture, fixture->dram[0][0],
                                (uint64_t)DRAM_WALK_BASE + DRAM_WRITE_BASE_OFFSET +
                                    (uint64_t)index * PROOF_BLOCK, readback, PROOF_BLOCK);
        if (rc == FT_TT_OK)
            rc = noc->write(noc->user, (uint16_t)fixture->core[0].x,
                            (uint16_t)fixture->core[0].y,
                            destination_address(fixture) + index * PROOF_BLOCK, staged[index],
                            PROOF_BLOCK);
        if (rc != FT_TT_OK) {
            ft_tt_test_check("DRAM poisoned and L1 staged", false, ft_tt_error_string(rc));
            return;
        }
    }

    rc = run_condition(fixture, &condition, &result);
    ft_tt_test_check_fmt("the write proof run completed", rc == FT_TT_OK,
                         "rc=%d (%s), error 0x%08x", rc, ft_tt_error_string(rc),
                         result.first_error);
    if (rc != FT_TT_OK)
        return;

    for (index = 0u; index < PROOF_SLOTS; index++) {
        rc = fixture_read_dram(fixture, fixture->dram[0][0],
                               (uint64_t)DRAM_WALK_BASE + DRAM_WRITE_BASE_OFFSET +
                                   (uint64_t)index * PROOF_BLOCK, readback, PROOF_BLOCK);
        if (rc != FT_TT_OK || memcmp(readback, staged[index], PROOF_BLOCK) != 0)
            mismatches++;
        if (readback[0] == SLOT_POISON_BYTE && readback[1] == SLOT_POISON_BYTE)
            poisoned++;
    }
    ft_tt_test_check_fmt("every DRAM block holds the L1 slot the walk names",
                         mismatches == 0u && poisoned == 0u,
                         "%u of %u blocks wrong, %u still poisoned", mismatches,
                         (unsigned)PROOF_SLOTS, poisoned);
}

/**
 * Stage 4 — THE FALSIFIER. Two runs whose answers are predicted, one of which must come
 * out LOW.
 *
 * (a) A single core streaming disjointly from one bank port must reproduce the recorded
 *     ~39 GB/s reader rate. A harness whose scale is wrong misses this.
 * (b) The SAME core, SAME bytes, SAME everything except `barrier_every = 1` — every read
 *     waited on before the next is issued — must come out far below (a), because it is
 *     round-trip-latency bound rather than bandwidth bound. If a deliberately serialised
 *     arm reports a plausible bandwidth, nothing else in this file is evidence.
 * (c) Half the timed bytes must give the same rate. A harness that mixes a fixed cost into
 *     the timed window fails this and only this.
 */
static void run_stage_falsifier(capacity_fixture *fixture)
{
    sweep_condition pipelined;
    sweep_condition serialised;
    sweep_condition halved;
    sweep_condition tile_shaped;
    run_result pipelined_result;
    run_result serialised_result;
    run_result halved_result;
    run_result tile_result;
    int rc;

    ft_tt_test_step("stage 4 — the harness must be able to produce a number nobody wanted");

    set_condition(&pipelined, "1 core, pipelined", 1u, 1u, 1u, 0u, 256uLL * MIB, STREAM_BLOCK,
                  STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
    serialised = pipelined;
    (void)snprintf(serialised.label, sizeof(serialised.label), "1 core, SERIALISED");
    serialised.barrier_every = 1u;
    serialised.target_bytes = 64uLL * MIB;
    halved = pipelined;
    (void)snprintf(halved.label, sizeof(halved.label), "1 core, half the bytes");
    halved.target_bytes = STREAM_TARGET / 2u;
    set_condition(&tile_shaped, "1 core, 2 KiB requests (matmul shape)", 1u, 1u, 1u, 0u,
                  256uLL * MIB, 2048u, 48u, 48u, 512uLL * MIB);

    /* Interleaved, and the serialised arm between the two pipelined ones, so host drift
     * cannot be what separates them. */
    rc = run_condition(fixture, &pipelined, &pipelined_result);
    if (rc == FT_TT_OK)
        rc = run_condition(fixture, &serialised, &serialised_result);
    if (rc == FT_TT_OK)
        rc = run_condition(fixture, &halved, &halved_result);
    if (rc == FT_TT_OK)
        rc = run_condition(fixture, &tile_shaped, &tile_result);
    ft_tt_test_check_fmt("all four falsifier runs completed", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    ft_tt_test_check_fmt("a single pipelined core lands in the recorded 30-50 GB/s band",
                         pipelined_result.aggregate > 30.0 && pipelined_result.aggregate < 50.0,
                         "%.2f GB/s at %u KiB requests, depth %u", pipelined_result.aggregate,
                         STREAM_BLOCK / 1024u, STREAM_DEPTH);
    ft_tt_test_check_fmt("the SERIALISED arm comes out far below it — the harness can fail",
                         serialised_result.aggregate < pipelined_result.aggregate * 0.5,
                         "%.2f GB/s serialised vs %.2f pipelined (%.2fx)",
                         serialised_result.aggregate, pipelined_result.aggregate,
                         pipelined_result.aggregate /
                             (serialised_result.aggregate > 0.0 ? serialised_result.aggregate
                                                                : 1e-9));
    ft_tt_test_check_fmt("half the timed bytes gives the same rate — no fixed cost is inside "
                         "the window",
                         halved_result.aggregate > pipelined_result.aggregate * 0.95 &&
                             halved_result.aggregate < pipelined_result.aggregate * 1.05,
                         "%.2f vs %.2f GB/s", halved_result.aggregate,
                         pipelined_result.aggregate);
    printf("  the matmul-shaped arm (2 KiB requests, depth 48) reads %.2f GB/s — the "
           "recorded reader is 39.4\n\n", tile_result.aggregate);
    fflush(stdout);
}

/**
 * Stage 5 — the red half: three shapes that must be REFUSED in the kernel before any
 * transaction is issued.
 *
 * A zero-length NoC transfer is the shape that took this card to `Mem- BusMaster-` on
 * 2026-08-12 with no error anywhere until the next read returned 0xFFFFFFFF, and the
 * host-side checker cannot see a kernel-issued transfer. These are safe
 * to run for exactly one reason: the refusal is before the register writes.
 */
static void run_stage_refusals(capacity_fixture *fixture)
{
    static const struct {
        const char *what;
        uint32_t argument;
        uint32_t value;
        int expected;
    } cases[] = {
        {"a ZERO-LENGTH request", 3u, 0u, FTDB_ERROR_ZERO_LENGTH},
        {"a 32-byte-ALIGNED DRAM base", 6u, DRAM_WALK_BASE + 32u, FTDB_ERROR_ALIGNMENT},
        {"an L1 window past the end of L1", 9u, 0x178000u, FTDB_ERROR_L1_BOUNDS},
    };
    const ft_tt_noc_access *noc = &fixture->session->noc;
    sweep_condition condition;
    uint32_t entry;

    ft_tt_test_step("stage 5 — the refusals, on silicon");
    set_condition(&condition, "refusal", 1u, 1u, 1u, 0u, 16uLL * MIB, STREAM_BLOCK, 4u, 4u,
                  16uLL * MIB);

    for (entry = 0u; entry < sizeof(cases) / sizeof(cases[0]); entry++) {
        uint32_t status[STATUS_WORDS];
        uint32_t argument_address;
        int rc;

        rc = poison_status(fixture, 0u);
        if (rc == FT_TT_OK)
            rc = plant_arguments(fixture, 0u, &condition, 4u, fixture->dram[0][0],
                                 DRAM_WALK_BASE);
        argument_address = fixture->config[0].runtime_args_address[FT_TT_PROC_DM0] +
                           cases[entry].argument * 4u;
        if (rc == FT_TT_OK)
            rc = noc->write(noc->user, (uint16_t)fixture->core[0].x,
                            (uint16_t)fixture->core[0].y, argument_address, &cases[entry].value,
                            sizeof(cases[entry].value));
        if (rc == FT_TT_OK)
            rc = launch_and_wait(fixture, &condition);
        if (rc == FT_TT_OK)
            rc = read_status(fixture, 0u, status);
        ft_tt_test_check_fmt("refused in the kernel, before anything was issued",
                             rc == FT_TT_OK && status[STATUS_MAGIC_WORD] == STATUS_MAGIC &&
                                 (int32_t)status[STATUS_ERROR] == cases[entry].expected &&
                                 status[STATUS_BLOCKS_ISSUED] == 0u,
                             "%s: rc=%d, magic 0x%08x, error %d (wanted %d), blocks issued %u",
                             cases[entry].what, rc, status[STATUS_MAGIC_WORD],
                             (int)(int32_t)status[STATUS_ERROR], cases[entry].expected,
                             status[STATUS_BLOCKS_ISSUED]);
    }
}

/* ------------------------------------------------------------------------- */
/* The sweeps                                                                */
/* ------------------------------------------------------------------------- */

/**
 * Sweep A — ONE BANK. This is where 42-vs-67 per bank is decided, and it separates the
 * port from the bank behind it for the first time.
 *
 * Four families over the same core counts: port 0 only vs all three ports, each disjoint
 * vs shared. If 42 GB/s is a PORT limit the port-0 family plateaus there while the
 * three-port family does not; if it is a BANK limit both plateau together.
 */
static uint32_t build_one_bank_sweep(sweep_condition *out)
{
    static const uint32_t cores[] = {1u, 2u, 3u, 4u, 6u, 8u, 12u};
    static const uint32_t blocks[] = {2048u, 16384u};
    uint32_t count = 0u;
    uint32_t index;
    uint32_t ports;
    uint32_t shared;
    uint32_t size;

    for (size = 0u; size < 2u; size++) {
        for (ports = 1u; ports <= 3u; ports += 2u) {
            for (shared = 0u; shared <= 1u; shared++) {
                for (index = 0u; index < sizeof(cores) / sizeof(cores[0]); index++) {
                    char label[48];

                    (void)snprintf(label, sizeof(label), "%2uK %s %s, %2u cores",
                                   blocks[size] / 1024u, ports == 1u ? "1port" : "3port",
                                   shared ? "SHARED  " : "disjoint", cores[index]);
                    set_condition(&out[count++], label, cores[index], 1u, ports, shared,
                                  shared ? 16uLL * MIB : 256uLL * MIB, blocks[size],
                                  STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
                }
            }
        }
    }
    return count;
}

/**
 * Sweep B — WORKING SET. Six cores on bank 0 across all three ports, disjoint and shared,
 * from a set small enough to sit in row buffers to one far past any locality effect.
 *
 * The 537 GB/s round's ~17 MB shared set and the cooperative round's ~14 MB disjoint one
 * are both inside this range, so if working set is the explanation it shows up here as a
 * knee rather than as a difference between two programs.
 */
static uint32_t build_working_set_sweep(sweep_condition *out)
{
    /* 512 MiB is the ceiling the disjoint arm can reach with six regions inside one bank's
     * 32-bit address space, and it is already 32,768 blocks with zero reuse — far past any
     * row-buffer effect, which is what the axis is for. */
    static const uint64_t sets[] = {64uLL * KIB, 512uLL * KIB, 4uLL * MIB,  16uLL * MIB,
                                    64uLL * MIB, 256uLL * MIB, 512uLL * MIB};
    uint32_t count = 0u;
    uint32_t index;
    uint32_t shared;

    for (shared = 0u; shared <= 1u; shared++) {
        for (index = 0u; index < sizeof(sets) / sizeof(sets[0]); index++) {
            char label[48];
            uint64_t set = sets[index];

            (void)snprintf(label, sizeof(label), "WS %5llu KiB, 6 cores %s",
                           (unsigned long long)(set / KIB), shared ? "SHARED" : "disjoint");
            set_condition(&out[count++], label, 6u, 1u, 3u, shared, set, STREAM_BLOCK,
                          STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
        }
    }
    return count;
}

/**
 * Sweep P — WHY IS A 64 KiB WORKING SET SLOWER THAN A 512 MiB ONE?
 *
 * Sweep B measured 41.4 GB/s at 64 KiB against 58.4 flat from 512 KiB to 512 MiB, on six
 * cores, one bank, three ports — the small set being the slow one, which is backwards from
 * every intuition about locality. Sharing made no difference at any size, so it is not
 * contention between the readers.
 *
 * THE ONE THING THAT DIFFERS between sweep B's 64 KiB arm and its 512 KiB arm is
 * `wrap_blocks`: 4 against 32. Same block size, same depth, same barrier, same base, same
 * region placement (both round to the 1 MiB minimum). And `depth` is 32 — so at 64 KiB the
 * 32 reads inside ONE barrier window aim at only 4 distinct addresses, EIGHT reads apiece,
 * while at 512 KiB each of the 32 has its own. The knee sits exactly at
 * `working_set == depth * block_bytes`, which is a coincidence worth destroying.
 *
 * Four families, and the three hypotheses give DIFFERENT answers across them:
 *
 *   1 KNEE      the original shape, finely, over wrap 1..256. Cliff or ramp, and where.
 *   2 DEPTH     the same wrap ladder at depth 8 and 64. If the mechanism is duplicate
 *               addresses INSIDE one in-flight window, the knee MOVES with depth — 128 KiB
 *               at depth 8 and 1 MiB at depth 64. If it is DRAM locality it does not move,
 *               because the DRAM cannot see how many reads the core has outstanding.
 *   3 SPAN      wrap held at 4 and 8 while the STRIDE moves the span from 64 KiB to 16 MiB.
 *               Same number of distinct addresses, same wrap frequency, same instruction
 *               stream — only how far apart the addresses sit. This is the one that
 *               separates "few addresses" from "few bytes".
 *   4 BLOCK     the same wrap ladder at 2 KiB requests. The duplicate-window hypothesis puts
 *               the knee at the same WRAP (32 blocks = 64 KiB here, 512 KiB there); a
 *               byte-span hypothesis puts it at the same NUMBER OF BYTES.
 *
 * The fourth candidate — that the WRAP ITSELF costs, a branch taken every 4th read instead
 * of every 32nd — is bounded off the page and tested anyway. At 9.7 GB/s per core a 16 KiB
 * read is ~1350 core cycles, and wrap 4 against wrap 32 is 7 extra taken branches per 32
 * reads: 0.02% where the effect is 29%. Family 3 tests it regardless, because a big stride
 * keeps the wrap frequency and changes only the addresses.
 *
 * Six cores, bank 0, three ports, disjoint — sweep B's own shape, so the numbers are
 * comparable to the record rather than to a new configuration.
 */
static uint32_t build_anomaly_sweep(sweep_condition *out)
{
    /* wrap_blocks, i.e. how many distinct addresses the walk cycles through. */
    static const uint32_t wraps[] = {1u,  2u,  3u,  4u,  6u,  8u,   12u,
                                     16u, 24u, 32u, 48u, 64u, 256u};
    /** The powers of two out of `wraps`, for the families that need a ladder rather than a
     *  resolution — a knee that MOVES is visible at octaves. */
    static const uint32_t coarse_wraps[] = {1u, 2u, 4u, 8u, 16u, 32u, 64u, 256u};
    static const uint32_t depths[] = {8u, 64u};
    /* Family 3: (wrap_blocks, stride) at 16 KiB blocks and depth 32. */
    static const struct {
        uint32_t wrap;
        uint32_t stride;
    } spans[] = {{4u, 16384u},   {4u, 65536u},   {4u, 262144u}, {4u, 4194304u},
                 {8u, 16384u},   {8u, 2097152u}, {32u, 16384u}, {32u, 524288u}};
    uint32_t count = 0u;
    uint32_t index;
    uint32_t which;

    for (index = 0u; index < sizeof(wraps) / sizeof(wraps[0]); index++) {
        char label[48];
        uint64_t set = (uint64_t)wraps[index] * STREAM_BLOCK;

        (void)snprintf(label, sizeof(label), "1 knee  wrap %3u d32  WS %5u KiB", wraps[index],
                       (unsigned)(set / KIB));
        set_condition(&out[count++], label, 6u, 1u, 3u, 0u, set, STREAM_BLOCK, STREAM_DEPTH,
                      STREAM_DEPTH, STREAM_TARGET);
    }
    for (which = 0u; which < sizeof(depths) / sizeof(depths[0]); which++) {
        for (index = 0u; index < sizeof(coarse_wraps) / sizeof(coarse_wraps[0]); index++) {
            char label[48];
            uint64_t set = (uint64_t)coarse_wraps[index] * STREAM_BLOCK;

            (void)snprintf(label, sizeof(label), "2 depth wrap %3u d%-2u  WS %5u KiB",
                           coarse_wraps[index], depths[which], (unsigned)(set / KIB));
            set_condition(&out[count++], label, 6u, 1u, 3u, 0u, set, STREAM_BLOCK, depths[which],
                          depths[which], STREAM_TARGET);
        }
    }
    for (index = 0u; index < sizeof(spans) / sizeof(spans[0]); index++) {
        char label[48];
        uint64_t set = (uint64_t)spans[index].wrap * STREAM_BLOCK;
        uint64_t span = (uint64_t)spans[index].wrap * spans[index].stride;

        (void)snprintf(label, sizeof(label), "3 span  wrap %3u d32  span %5u KiB",
                       spans[index].wrap, (unsigned)(span / KIB));
        set_condition(&out[count++], label, 6u, 1u, 3u, 0u, set, STREAM_BLOCK, STREAM_DEPTH,
                      STREAM_DEPTH, STREAM_TARGET);
        out[count - 1u].stride_bytes = spans[index].stride;
    }
    for (index = 0u; index < sizeof(coarse_wraps) / sizeof(coarse_wraps[0]); index++) {
        char label[48];
        uint64_t set = (uint64_t)coarse_wraps[index] * 2048u;

        (void)snprintf(label, sizeof(label), "4 block wrap %3u d32 2K WS %4u KiB",
                       coarse_wraps[index], (unsigned)(set / KIB));
        set_condition(&out[count++], label, 6u, 1u, 3u, 0u, set, 2048u, STREAM_DEPTH,
                      STREAM_DEPTH, STREAM_TARGET);
    }
    return count;
}

/**
 * Sweep Q — the same anomaly once sweep P has said what it is NOT.
 *
 * Sweep P killed three explanations and left one standing. It is not the wrap arithmetic
 * (wrap 4 reads 41.4 GB/s at a 16 KiB stride and 56.6 at a 64 KiB one — same instruction
 * stream, same wrap frequency, only the addresses move). It is not duplicate addresses
 * inside one in-flight window (the knee sits at 128 KiB at depth 8, 32 AND 64). It is not
 * the number of distinct blocks (2 KiB requests reproduce the same curve against BYTES with
 * eight times the block count — 32 KiB of working set reads 32.0 GB/s whether that is 2
 * blocks or 16). What is left is the ADDRESS STREAM itself, and sweep P's own stride family
 * says so directly: 16 KiB stride 58.4, 64 KiB 56.6, 256 KiB and above 31.5 — the same
 * 31.5 a 16 KiB working set gives. Both ends of that are "the addresses cover few of
 * whatever the memory system interleaves over".
 *
 * So this sweep measures the address stream at resolution, and adds the control sweep P
 * could not give: ONE reader. Every number in P is six readers on one bank, and a per-reader
 * mechanism and a six-way interaction are not distinguishable there.
 *
 *   5 stride   wrap 32, block 16 KiB, stride 16 KiB .. 2 MiB. If there is an interleave
 *              PERIOD the curve is periodic in the stride rather than monotone in it.
 *   6 span     2 KiB requests so the block count is never the small number, working set
 *              stepped in 16 KiB units through the dip. Sweep P saw 64 KiB and 192 KiB below
 *              BOTH their neighbours — 64 KiB slower than 48 KiB — and two points is not a
 *              shape.
 *   7 one core the same two ladders on a single reader. If the curve survives, the mechanism
 *              is per-stream and the six-way arms were never the point.
 *   8 base     a 64 KiB working set moved to four different base offsets. Size against
 *              ALIGNMENT: if only the size matters these are all 41.4.
 */
static uint32_t build_anomaly_address_sweep(sweep_condition *out)
{
    static const uint32_t strides_kib[] = {16u,  24u,  32u,  48u,  64u,   96u,  128u,
                                           192u, 256u, 384u, 512u, 1024u, 2048u};
    static const uint32_t spans_kib[] = {16u,  24u,  32u,  48u,  64u,  80u,  96u,  112u,
                                         128u, 160u, 192u, 224u, 256u, 320u, 384u, 512u};
    static const uint32_t solo_spans_kib[] = {16u, 32u, 64u, 96u, 128u, 192u, 256u, 512u, 4096u};
    static const uint32_t solo_strides_kib[] = {16u, 64u, 256u, 1024u};
    static const uint32_t base_offsets_kib[] = {0u, 16u, 32u, 64u, 512u};
    uint32_t count = 0u;
    uint32_t index;

    for (index = 0u; index < sizeof(strides_kib) / sizeof(strides_kib[0]); index++) {
        char label[48];

        (void)snprintf(label, sizeof(label), "5 stride 6c wrap 32 stride %4u KiB",
                       strides_kib[index]);
        set_condition(&out[count++], label, 6u, 1u, 3u, 0u, 32uLL * STREAM_BLOCK, STREAM_BLOCK,
                      STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
        out[count - 1u].stride_bytes = (uint32_t)(strides_kib[index] * KIB);
    }
    for (index = 0u; index < sizeof(spans_kib) / sizeof(spans_kib[0]); index++) {
        char label[48];

        (void)snprintf(label, sizeof(label), "6 span   6c 2K blocks WS %4u KiB", spans_kib[index]);
        set_condition(&out[count++], label, 6u, 1u, 3u, 0u, spans_kib[index] * KIB, 2048u,
                      STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
    }
    for (index = 0u; index < sizeof(solo_spans_kib) / sizeof(solo_spans_kib[0]); index++) {
        char label[48];

        (void)snprintf(label, sizeof(label), "7 solo   1c 16K blocks WS %5u KiB",
                       solo_spans_kib[index]);
        set_condition(&out[count++], label, 1u, 1u, 1u, 0u, solo_spans_kib[index] * KIB,
                      STREAM_BLOCK, STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
    }
    for (index = 0u; index < sizeof(solo_strides_kib) / sizeof(solo_strides_kib[0]); index++) {
        char label[48];

        (void)snprintf(label, sizeof(label), "7 solo   1c wrap 32 stride %4u KiB",
                       solo_strides_kib[index]);
        set_condition(&out[count++], label, 1u, 1u, 1u, 0u, 32uLL * STREAM_BLOCK, STREAM_BLOCK,
                      STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
        out[count - 1u].stride_bytes = (uint32_t)(solo_strides_kib[index] * KIB);
    }
    for (index = 0u; index < sizeof(base_offsets_kib) / sizeof(base_offsets_kib[0]); index++) {
        char label[48];

        (void)snprintf(label, sizeof(label), "8 base   6c WS 64 KiB base +%4u KiB",
                       base_offsets_kib[index]);
        set_condition(&out[count++], label, 6u, 1u, 3u, 0u, 64uLL * KIB, STREAM_BLOCK,
                      STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
        out[count - 1u].base_offset = (uint32_t)(base_offsets_kib[index] * KIB);
    }
    return count;
}

/**
 * Sweep C — THE CARD. Every bank, every port, disjoint and large: the realistic
 * weight-streaming pattern. This is the number Phase 7 plans against.
 *
 * The shared arms at 14 and 24 cores are there to reproduce the OLD condition — a shared
 * ~17 MB working set — against the new one on the same cores in the same process.
 */
static uint32_t build_aggregate_sweep(sweep_condition *out)
{
    static const uint32_t cores[] = {8u, 16u, 24u, 32u, 48u, 64u, 80u, 96u, 112u};
    static const uint32_t blocks[] = {2048u, 16384u};
    uint32_t count = 0u;
    uint32_t index;
    uint32_t size;

    for (size = 0u; size < 2u; size++) {
        for (index = 0u; index < sizeof(cores) / sizeof(cores[0]); index++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "%2uK 8bank 3port disjoint, %3u cores",
                           blocks[size] / 1024u, cores[index]);
            set_condition(&out[count++], label, cores[index], 8u, 3u, 0u, 256uLL * MIB,
                          blocks[size], STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
        }
    }
    /* Port 0 only is what every earlier round on this card used, so the whole difference
     * between "42 GB/s a bank" and the answer is this row against the one above it. */
    set_condition(&out[count++], " 2K 8bank 1port disjoint,  24 cores", 24u, 8u, 1u, 0u,
                  256uLL * MIB, 2048u, STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
    set_condition(&out[count++], " 2K 8bank 1port disjoint,  64 cores", 64u, 8u, 1u, 0u,
                  256uLL * MIB, 2048u, STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
    set_condition(&out[count++], " 2K 8bank 1port disjoint, 112 cores", 112u, 8u, 1u, 0u,
                  256uLL * MIB, 2048u, STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
    /* The OLD condition, reproduced on the same cores in the same process: a shared ~17 MB
     * working set, which is where >= 537 GB/s across 14 cores was recorded. */
    set_condition(&out[count++], " 2K SHARED 17MB,  14 cores", 14u, 8u, 3u, 1u, 17uLL * MIB,
                  2048u, STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
    set_condition(&out[count++], " 2K SHARED 17MB,  64 cores", 64u, 8u, 3u, 1u, 17uLL * MIB,
                  2048u, STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
    set_condition(&out[count++], " 2K SHARED 17MB, 112 cores", 112u, 8u, 3u, 1u, 17uLL * MIB,
                  2048u, STREAM_DEPTH, STREAM_DEPTH, STREAM_TARGET);
    return count;
}

/** Sweep D — request size, at a core count that is already at the ceiling. */
static uint32_t build_block_size_sweep(sweep_condition *out)
{
    static const uint32_t blocks[] = {512u, 1024u, 2048u, 4096u, 8192u, 16384u};
    static const uint32_t cores[] = {1u, 24u, 96u};
    uint32_t count = 0u;
    uint32_t index;
    uint32_t which;

    for (which = 0u; which < sizeof(cores) / sizeof(cores[0]); which++) {
        for (index = 0u; index < sizeof(blocks) / sizeof(blocks[0]); index++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "block %5u B, %2u cores disjoint",
                           blocks[index], cores[which]);
            set_condition(&out[count++], label, cores[which], 8u, 3u, 0u, 256uLL * MIB,
                          blocks[index], STREAM_DEPTH, STREAM_DEPTH, 512uLL * MIB);
        }
    }
    return count;
}

/**
 * Sweep F — AROUND THE PEAK. The card's aggregate turns over at one core per DRAM NoC
 * port, so this is the only region a serving design would ever be asked to sit in.
 *
 * Core count and reads-in-flight together, at the request size the matmul reader actually
 * uses, because "24 cores is the peak" and "depth 32 is the peak" are two claims and a
 * single diagonal through them is not evidence for either.
 */
static uint32_t build_peak_sweep(sweep_condition *out)
{
    static const uint32_t cores[] = {16u, 20u, 24u, 28u, 32u};
    static const uint32_t depths[] = {8u, 16u, 32u, 64u};
    uint32_t count = 0u;
    uint32_t index;
    uint32_t which;

    for (which = 0u; which < sizeof(depths) / sizeof(depths[0]); which++) {
        for (index = 0u; index < sizeof(cores) / sizeof(cores[0]); index++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "2K depth %2u, %2u cores disjoint",
                           depths[which], cores[index]);
            set_condition(&out[count++], label, cores[index], 8u, 3u, 0u, 256uLL * MIB, 2048u,
                          depths[which], depths[which], STREAM_TARGET);
        }
    }
    return count;
}

/**
 * Sweep G — THE CLIFF, finely, from two directions.
 *
 * 768 reads in flight across the card gives 429 GB/s and 1024 gives 329, reproduced both as
 * 32 cores x depth 32 and as 16 x 64 — so it is the PRODUCT. What that cannot say is
 * whether the turn is an EDGE or a ROLLOFF, and those are different mechanisms: a fixed
 * resource that runs out at a number (a queue, a credit pool, a tracking table) against
 * contention that degrades continuously.
 *
 * Both families cross the same products with the other factor held: depth swept at 24 cores
 * (one core per DRAM endpoint) and cores swept at depth 32. If the two agree at equal
 * products the product really is the variable; where they disagree, the factor is.
 */
static uint32_t build_cliff_sweep(sweep_condition *out)
{
    static const uint32_t depths[] = {8u,  12u, 16u, 20u, 24u, 28u, 32u,
                                      36u, 40u, 44u, 48u, 56u, 64u, 96u};
    static const uint32_t cores[] = {8u, 16u, 20u, 24u, 28u, 32u, 40u, 48u, 56u, 64u};
    uint32_t count = 0u;
    uint32_t index;

    for (index = 0u; index < sizeof(depths) / sizeof(depths[0]); index++) {
        char label[48];

        (void)snprintf(label, sizeof(label), "24 cores x depth %3u = %5u in flight",
                       depths[index], 24u * depths[index]);
        set_condition(&out[count++], label, 24u, 8u, 3u, 0u, 64uLL * MIB, 2048u, depths[index],
                      depths[index], STREAM_TARGET);
    }
    for (index = 0u; index < sizeof(cores) / sizeof(cores[0]); index++) {
        char label[48];

        (void)snprintf(label, sizeof(label), "%3u cores x depth  32 = %5u in flight",
                       cores[index], cores[index] * 32u);
        set_condition(&out[count++], label, cores[index], 8u, 3u, 0u, 64uLL * MIB, 2048u, 32u,
                      32u, STREAM_TARGET);
    }
    return count;
}

/**
 * Sweep H — THE SAME reads in flight, REDISTRIBUTED. This is the one comparison that
 * separates a card-wide resource from a per-endpoint one.
 *
 * Every condition here has the same product of cores and depth. What changes is how many
 * DRAM endpoints it is spread over: 24 (eight banks, three ports), 3 (one bank), or 1 (one
 * port of one bank). If the collapse is a card-level pool, the arms at equal product agree
 * whatever they are aimed at. If it is per-endpoint, concentrating the same number of
 * outstanding reads onto one door must collapse long before spreading them over 24 does.
 */
static uint32_t build_spread_sweep(sweep_condition *out)
{
    static const struct { uint32_t cores; uint32_t depth; } shapes[] = {
        {24u, 32u}, {12u, 64u}, {48u, 16u},              /* 768 in flight */
        {32u, 32u}, {16u, 64u}, {64u, 16u}, {128u, 8u},  /* 1024 in flight */
        {48u, 32u}, {24u, 64u}                           /* 1536 in flight */
    };
    static const struct { uint32_t banks; uint32_t ports; const char *name; } spread[] = {
        {8u, 3u, "24 endpoints"}, {8u, 1u, " 8 endpoints"}, {1u, 3u, " 3 endpoints"},
        {1u, 1u, " 1 endpoint "}
    };
    uint32_t count = 0u;
    uint32_t shape;
    uint32_t where;

    for (shape = 0u; shape < sizeof(shapes) / sizeof(shapes[0]); shape++)
        for (where = 0u; where < sizeof(spread) / sizeof(spread[0]); where++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "%5u flight, %s, %3ux%2u",
                           shapes[shape].cores * shapes[shape].depth, spread[where].name,
                           shapes[shape].cores, shapes[shape].depth);
            /* 16 MiB per region so even 128 disjoint regions fit one bank's addressable
             * span. Working set is flat above 512 KiB, so this costs nothing. */
            set_condition(&out[count++], label, shapes[shape].cores, spread[where].banks,
                          spread[where].ports, 0u, 16uLL * MIB, 2048u, shapes[shape].depth,
                          shapes[shape].depth, STREAM_TARGET);
        }
    return count;
}

/**
 * Sweep I — NoC DISTANCE, with the endpoint held still.
 *
 * 429 GB/s is a delivered-to-Tensix figure and placement alone swings it 16%, so the
 * question is whether the ceiling is in the route or in the memory. Here the bank and the
 * port are fixed and the only thing that moves is WHERE the readers sit — same request
 * size, same depth, same working set, same number of cores.
 *
 * Two core counts, because they ask different questions: ONE core cannot saturate a port
 * (39-41 GB/s against ~45), so it reports what the route costs a single stream, while TWO
 * cores sit exactly at the port ceiling and report whether the route can still deliver it.
 */
static uint32_t build_distance_sweep(sweep_condition *out, uint32_t first_index,
                                     uint32_t columns)
{
    static const uint32_t rows[] = {1u, 5u, 9u};
    static const uint32_t cols[] = {0u, 5u, 10u};
    static const uint32_t banks[] = {0u, 7u};
    uint32_t count = 0u;
    uint32_t which;
    uint32_t row;
    uint32_t col;
    uint32_t cores;

    for (which = 0u; which < sizeof(banks) / sizeof(banks[0]); which++)
        for (cores = 1u; cores <= 2u; cores++)
            for (row = 0u; row < sizeof(rows) / sizeof(rows[0]); row++)
                for (col = 0u; col < sizeof(cols) / sizeof(cols[0]); col++) {
                    uint32_t index = rows[row] * columns + cols[col];
                    char label[48];

                    if (index < first_index)
                        continue;
                    (void)snprintf(label, sizeof(label), "bank %u p0, %u core at grid r%u c%2u",
                                   banks[which], cores, rows[row], cols[col]);
                    set_condition(&out[count], label, cores, 1u, 1u, 0u, 64uLL * MIB, 2048u,
                                  32u, 32u, STREAM_TARGET);
                    out[count].bank_offset = banks[which];
                    out[count].core_offset = index - first_index;
                    count++;
                }
    return count;
}

/**
 * Sweep J — WHICH port, and HOW MANY per bank.
 *
 * Port 0 across all eight banks measures ~200 GB/s where 8 x 45 = 360 should be reachable.
 * The hypothesis on the table is that port 0's endpoints share a routing path the other
 * ports do not. That has exactly one cheap discriminator: run the SAME shape against port 1
 * only, port 2 only, and against one port per bank that is a DIFFERENT port on each bank.
 * If all four measure the same, port 0 is not special and the limit is "one endpoint per
 * bank"; if the mixed-port arm recovers, the endpoints' positions are the mechanism.
 *
 * The one-bank block underneath asks the same question inside a single bank: a bank does ~58
 * GB/s and a port ~45, so if two ports of one bank reach 58 the port limit is at the
 * endpoint and the bank limit is the memory behind it.
 */
static uint32_t build_ports_sweep(sweep_condition *out)
{
    static const uint32_t cores[] = {16u, 24u, 48u};
    uint32_t count = 0u;
    uint32_t which;
    uint32_t index;

    for (which = 0u; which < sizeof(cores) / sizeof(cores[0]); which++) {
        static const char *const names[] = {"port 0 only", "port 1 only", "port 2 only",
                                            "one port per bank, ROTATED", "ports 0+1",
                                            "ports 1+2", "all three ports"};
        for (index = 0u; index < 7u; index++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "8 banks, %-26s %2u cores", names[index],
                           cores[which]);
            set_condition(&out[count], label, cores[which], 8u,
                          index < 4u ? 1u : (index < 6u ? 2u : 3u), 0u, 64uLL * MIB, 2048u, 32u,
                          32u, STREAM_TARGET);
            out[count].port_offset = index < 3u ? index : (index == 4u ? 0u : 1u);
            if (index == 3u) {
                out[count].port_offset = 0u;
                out[count].port_by_bank = 1u;
            }
            if (index == 6u)
                out[count].port_offset = 0u;
            count++;
        }
    }
    for (which = 0u; which < 3u; which++) {
        uint32_t ports = which + 1u;
        uint32_t index_cores;

        for (index_cores = 2u; index_cores <= 8u; index_cores *= 2u) {
            char label[48];

            (void)snprintf(label, sizeof(label), "bank 0 alone, %u port%s, %u cores", ports,
                           ports == 1u ? " " : "s", index_cores);
            set_condition(&out[count++], label, index_cores, 1u, ports, 0u, 64uLL * MIB, 2048u,
                          32u, 32u, STREAM_TARGET);
        }
    }
    return count;
}

/**
 * Sweep K — DO TWO DRAM ENDPOINTS ON THE SAME NoC ROW SHARE A CEILING?
 *
 * The endpoint table says port p of bank b and of bank b+4 sit at the SAME y in the two
 * DRAM columns — (0,0) and (9,0) for port 0 of banks 0 and 4 — so a port index occupies 4
 * of the 12 rows and all three ports together occupy all 12. That is the only structure in
 * the layout that distinguishes "one port of every bank" from "every port of every bank",
 * and it is testable directly: saturate TWO endpoints that share a row and TWO that do not,
 * with everything else identical.
 *
 * Predictions, so the arms can disagree with them: if the row is a shared ceiling, a
 * same-row pair lands near ONE endpoint's rate and a different-row pair near twice it. If
 * the ceiling is the endpoint, both pairs double.
 */
static uint32_t build_row_sweep(sweep_condition *out)
{
    static const struct {
        const char *label;
        uint32_t cores;
        uint32_t bank_offset;
        uint32_t bank_count;
        uint32_t bank_stride;
        uint32_t ports;
        uint32_t port_offset;
        uint32_t rotate;
    } arms[] = {
        {"1 endpoint  b0p0            (row 0)",   3u, 0u, 1u, 1u, 1u, 0u, 0u},
        {"2 ep SAME ROW    b0p0 b4p0  (row 0)",   6u, 0u, 2u, 4u, 1u, 0u, 0u},
        {"2 ep same column b0p0 b1p0  (r0,r2)",   6u, 0u, 2u, 1u, 1u, 0u, 0u},
        {"2 ep neither     b0p0 b5p0  (r0,r2)",   6u, 0u, 2u, 5u, 1u, 0u, 0u},
        {"2 ep one bank    b0p0 b0p1  (r0,r1)",   6u, 0u, 1u, 1u, 2u, 0u, 0u},
        {"4 ep west column b0-3 p0  (4 rows)",   12u, 0u, 4u, 1u, 1u, 0u, 0u},
        {"4 ep east column b4-7 p0  (4 rows)",   12u, 4u, 4u, 1u, 1u, 0u, 0u},
        {"8 ep both columns p0      (4 rows)",   24u, 0u, 8u, 1u, 1u, 0u, 0u},
        {"8 ep ROTATED port        (8 rows)",    24u, 0u, 8u, 1u, 1u, 0u, 1u},
        {"4 ep west ROTATED b0-3   (4 rows)",    12u, 0u, 4u, 1u, 1u, 0u, 1u}
    };
    uint32_t count = 0u;
    uint32_t index;

    for (index = 0u; index < sizeof(arms) / sizeof(arms[0]); index++) {
        set_condition(&out[count], arms[index].label, arms[index].cores, arms[index].bank_count,
                      arms[index].ports, 0u, 16uLL * MIB, 2048u, 32u, 32u, STREAM_TARGET);
        out[count].bank_offset = arms[index].bank_offset;
        out[count].bank_stride = arms[index].bank_stride;
        out[count].port_offset = arms[index].port_offset;
        out[count].port_by_bank = arms[index].rotate;
        count++;
    }
    return count;
}

/**
 * Sweep L — reads in flight PER CORE, separated from reads per pass.
 *
 * Every sweep on this card so far has set `barrier_every` equal to `depth`, which makes
 * "how many reads a core has outstanding" and "how long its pass is" the same number. They
 * are not the same thing: `barrier_every` is what is actually in flight, and `depth` only
 * says how often the walk wraps its L1 slots. Sweep H found 128 cores x 8 and 32 cores x 32
 * — the same 1024 outstanding on the same 24 endpoints — 40% apart, so the two must be
 * separated before anything is attributed to "outstanding reads".
 */
static uint32_t build_burst_sweep(sweep_condition *out)
{
    static const uint32_t cores[] = {24u, 32u, 48u, 128u};
    static const uint32_t barriers[] = {4u, 8u, 16u, 32u};
    uint32_t count = 0u;
    uint32_t which;
    uint32_t index;

    for (which = 0u; which < sizeof(cores) / sizeof(cores[0]); which++)
        for (index = 0u; index < sizeof(barriers) / sizeof(barriers[0]); index++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "%3u cores, depth 32, %2u in flight each",
                           cores[which], barriers[index]);
            set_condition(&out[count++], label, cores[which], 8u, 3u, 0u, 16uLL * MIB, 2048u,
                          32u, barriers[index], STREAM_TARGET);
        }
    return count;
}

/**
 * Sweep M — the SAME work, the same endpoints, the readers spread further apart.
 *
 * Compact 24 cores measured 429 and stride-4 scattered 359 in the capacity round, which is
 * the strongest single piece of evidence that NoC routing is inside the card figure. Here
 * the core-to-endpoint assignment is identical in every arm — slot i reads the same endpoint
 * whatever the stride — so the only thing that changes is which Tensix rows and columns the
 * readers occupy, and therefore which links their responses travel over.
 */
static uint32_t build_scatter_sweep(sweep_condition *out)
{
    static const uint32_t strides[] = {1u, 2u, 3u, 4u, 5u};
    static const struct { uint32_t cores; uint32_t flight; } shapes[] = {
        {24u, 32u}, {24u, 16u}, {48u, 8u}
    };
    uint32_t count = 0u;
    uint32_t shape;
    uint32_t index;

    for (shape = 0u; shape < sizeof(shapes) / sizeof(shapes[0]); shape++)
        for (index = 0u; index < sizeof(strides) / sizeof(strides[0]); index++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "%2u cores x%2u flight, core stride %u",
                           shapes[shape].cores, shapes[shape].flight, strides[index]);
            set_condition(&out[count], label, shapes[shape].cores, 8u, 3u, 0u, 16uLL * MIB,
                          2048u, 32u, shapes[shape].flight, STREAM_TARGET);
            out[count].core_stride = strides[index];
            count++;
        }
    return count;
}

/**
 * Sweep N — WRITES, against READS of the same shape, interleaved.
 *
 * Everything known about this card's DRAM is read bandwidth, and Phase 7 writes KV cache
 * every token. Each shape appears twice, one arm each way, adjacent in the interleave so
 * the pair cannot be separated by host drift — the comparison is what is wanted, not the
 * absolute number. The shapes are the ones the read round found interesting: the per-core
 * rate, the port ceiling, the bank ceiling, the row-sharing pair, and the two card peaks.
 *
 * The serialised arm is here too, in the write direction, because a harness extended into a
 * new direction has to be shown able to fail in that direction as well.
 */
static uint32_t build_write_sweep(sweep_condition *out)
{
    static const struct {
        const char *label;
        uint32_t cores;
        uint32_t banks;
        uint32_t bank_stride;
        uint32_t ports;
        uint32_t flight;
        uint32_t block;
    } shapes[] = {
        {"1 core  b0p0            ",   1u, 1u, 1u, 1u, 32u, 2048u},
        {"1 core  b0p0  512 B     ",   1u, 1u, 1u, 1u, 32u, 512u},
        {"1 core  b0p0   16 KiB   ",   1u, 1u, 1u, 1u, 32u, 16384u},
        {"2 cores b0p0            ",   2u, 1u, 1u, 1u, 32u, 2048u},
        {"6 cores bank 0, 3 ports ",   6u, 1u, 1u, 3u, 32u, 2048u},
        {"6 cores SAME ROW b0 b4p0",   6u, 2u, 4u, 1u, 32u, 2048u},
        {"6 cores diff row b0 b1p0",   6u, 2u, 1u, 1u, 32u, 2048u},
        {"24 cores 8 banks 1 port ",  24u, 8u, 1u, 1u, 32u, 2048u},
        {"24 cores 8 banks 3 ports",  24u, 8u, 1u, 3u, 32u, 2048u},
        {"48 cores 8 banks 3 ports",  48u, 8u, 1u, 3u,  8u, 2048u},
        {"128 cores 8 banks 3 port", 128u, 8u, 1u, 3u,  4u, 2048u}
    };
    uint32_t count = 0u;
    uint32_t index;
    uint32_t direction;

    for (index = 0u; index < sizeof(shapes) / sizeof(shapes[0]); index++)
        for (direction = 0u; direction <= 1u; direction++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "%s %s", direction ? "WRITE" : "read ",
                           shapes[index].label);
            set_condition(&out[count], label, shapes[index].cores, shapes[index].banks,
                          shapes[index].ports, 0u, 16uLL * MIB, shapes[index].block, 32u,
                          shapes[index].flight, STREAM_TARGET);
            out[count].bank_stride = shapes[index].bank_stride;
            out[count].direction = direction;
            out[count].base_offset = DRAM_WRITE_BASE_OFFSET;
            count++;
        }
    for (direction = 0u; direction <= 1u; direction++) {
        char label[48];

        (void)snprintf(label, sizeof(label), "%s 1 core SERIALISED",
                       direction ? "WRITE" : "read ");
        set_condition(&out[count], label, 1u, 1u, 1u, 0u, 16uLL * MIB, 2048u, 32u, 1u,
                      STREAM_TARGET);
        out[count].direction = direction;
        out[count].base_offset = DRAM_WRITE_BASE_OFFSET;
        count++;
    }
    return count;
}

/**
 * Sweep O — the BANK ceiling and the CARD ceiling, taken the same way.
 *
 * "Is the card figure the memory or the NoC" has a direct arithmetic form: measure ONE bank
 * as high as it will go with nothing else running, then measure all eight the same way. If
 * the card lands at eight times the bank, nothing is being lost between them and the wall is
 * the memory. If it lands short, the shortfall is the delivery network and has a size.
 *
 * Both halves are swept over cores and reads-in-flight rather than taken at one shape,
 * because sweep L showed the optimum in-flight-per-core MOVES with core count — a single
 * shape would compare a saturated bank against an unsaturated card or the reverse.
 */
static uint32_t build_ceiling_sweep(sweep_condition *out)
{
    static const uint32_t bank_cores[] = {6u, 12u, 24u};
    static const uint32_t card_cores[] = {96u, 112u, 128u};
    static const uint32_t flights[] = {2u, 4u, 8u};
    uint32_t count = 0u;
    uint32_t which;
    uint32_t index;

    for (which = 0u; which < sizeof(bank_cores) / sizeof(bank_cores[0]); which++)
        for (index = 0u; index < sizeof(flights) / sizeof(flights[0]); index++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "BANK 0 alone, %3u cores x%2u flight",
                           bank_cores[which], flights[index]);
            set_condition(&out[count++], label, bank_cores[which], 1u, 3u, 0u, 16uLL * MIB,
                          2048u, 32u, flights[index], STREAM_TARGET);
        }
    for (which = 0u; which < sizeof(card_cores) / sizeof(card_cores[0]); which++)
        for (index = 0u; index < sizeof(flights) / sizeof(flights[0]); index++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "CARD, 8 banks, %3u cores x%2u flight",
                           card_cores[which], flights[index]);
            set_condition(&out[count++], label, card_cores[which], 8u, 3u, 0u, 16uLL * MIB,
                          2048u, 32u, flights[index], STREAM_TARGET);
        }
    return count;
}

/** Sweep E — how many reads in flight it takes, so "depth-bound" is excluded by
 *  measurement rather than by argument. */
static uint32_t build_depth_sweep(sweep_condition *out)
{
    static const uint32_t depths[] = {1u, 2u, 4u, 8u, 16u, 32u, 64u};
    uint32_t count = 0u;
    uint32_t index;

    for (index = 0u; index < sizeof(depths) / sizeof(depths[0]); index++) {
        char label[48];

        (void)snprintf(label, sizeof(label), "depth %2u, 1 core disjoint", depths[index]);
        set_condition(&out[count++], label, 1u, 1u, 1u, 0u, 256uLL * MIB, STREAM_BLOCK,
                      depths[index], depths[index], 512uLL * MIB);
    }
    return count;
}

/** Sweep M — THE DECODE MATMUL'S OWN READ SHAPE without its compute (M1, 2026-10-04).
 *  The model's matmul streams ~340 GB/s at 16, 20, 24 and 32 members alike, banks at 37-40
 *  of 59.2. Its reader: per member ONE bank's panel, 8704 B block-float requests (depth 8 x
 *  1088 B), 9 in flight (1 A + 8 B) then a barrier. If this shape alone reaches ~500 the
 *  gap is the reader/compute coupling; if it stops near 340 it is the access shape, and
 *  the other rows say which axis lifts it. */
static uint32_t build_mmshape_sweep(sweep_condition *out)
{
    static const struct {
        const char *label;
        uint32_t cores, ports, block, depth, barrier;
    } rows[] = {
        {"MM 16c 8704B x9 b9 3 ports (matmul)", 16u, 3u, 8704u, 9u, 9u},
        {"MM 16c 8704B x9 b9 1 port", 16u, 1u, 8704u, 9u, 9u},
        {"MM 16c 4352B x9 b9 3 ports (depth 4)", 16u, 3u, 4352u, 9u, 9u},
        {"MM 16c 17408B x9 b9 3p (depth 16)", 16u, 3u, 17408u, 9u, 9u},
        {"MM 16c 8704B x18 b18 3 ports", 16u, 3u, 8704u, 18u, 18u},
        {"MM 16c 8704B x36 b36 3 ports", 16u, 3u, 8704u, 36u, 36u},
        {"MM 16c 8704B x4 b4 3 ports", 16u, 3u, 8704u, 4u, 4u},
        {"MM 16c 2048B x32 b32 3 ports", 16u, 3u, 2048u, 32u, 32u},
        {"MM 32c 8704B x9 b9 3 ports", 32u, 3u, 8704u, 9u, 9u},
        {"MM 32c 2048B x32 b8 3p (peak)", 32u, 3u, 2048u, 32u, 8u},
        /* THE SPLIT READER'S OWN SHAPES (PN_TT_MATMUL_READ_KIB): 2 KiB -> B 2176 B x4 per
         * column x8 + A 2048 B x8 = 40 a barrier; 4 KiB -> 4352 B x2 x8 + A 4096 B x4 = 20. */
        {"MM 16c 2176B x40 b40 3p (KiB 2)", 16u, 3u, 2176u, 40u, 40u},
        {"MM 16c 4352B x20 b20 3p (KiB 4)", 16u, 3u, 4352u, 20u, 20u},
        {"MM 16c 1088B x72 b72 3p (KiB 1)", 16u, 3u, 1088u, 72u, 72u},
        {"MM 16c 2176B x20 b20 3p", 16u, 3u, 2176u, 20u, 20u},
    };
    /* ALIGNMENT (run 2): every block-float run is 1088 B tiles, so the matmul's requests
     * start off the 2 KiB grid; 2176 B x 40 measured 369 against 2048 B x 32's 475. These
     * rows move ONLY the alignment (base_offset) or ONLY the count, at fixed sizes. */
    static const struct {
        const char *label;
        uint32_t block, depth, base_offset;
    } aligned[] = {
        {"MA 16c 2048B x32 b32 +128B", 2048u, 32u, 128u},
        {"MA 16c 2048B x40 b40 aligned", 2048u, 40u, 0u},
        {"MA 16c 8192B x9 b9 aligned", 8192u, 9u, 0u},
        {"MA 16c 8192B x9 b9 +128B", 8192u, 9u, 128u},
        {"MA 16c 4096B x18 b18 aligned", 4096u, 18u, 0u},
        {"MA 16c 4096B x18 b18 +1024B", 4096u, 18u, 1024u},
        /* M4 (reader redesign): <= 2 KiB requests, more of them per barrier. A block of the
         * matmul is 16 KiB A + 8 x 8704 B B = ~86 KiB = ~43 x 2 KiB. */
        {"MS 16c 2048B x24 b24", 2048u, 24u, 0u},
        {"MS 16c 2048B x48 b48", 2048u, 48u, 0u},
        {"MS 16c 2048B x64 b64", 2048u, 64u, 0u},
        {"MS 16c 1024B x64 b64", 1024u, 64u, 0u},
        {"MS 16c 1024B x96 b96", 1024u, 96u, 0u},
        {"MS 16c 2048B x48 +1088B", 2048u, 48u, 1088u},
    };
    uint32_t index, extra;

    for (index = 0u; index < sizeof(rows) / sizeof(rows[0]); index++)
        set_condition(&out[index], rows[index].label, rows[index].cores, 8u, rows[index].ports,
                      0u, 16uLL * MIB, rows[index].block, rows[index].depth,
                      rows[index].barrier, STREAM_TARGET);
    /* PIPELINED (direction 2, transaction ids): a group's barrier waits for the PREVIOUS
     * group only, so two groups are in flight — the double-buffered reader M4 would build. */
    static const struct {
        const char *label;
        uint32_t block, depth;
    } piped[] = {
        {"MP 16c 8704B x9 PIPE (matmul)", 8704u, 9u},
        {"MP 16c 4352B x9 PIPE (depth 4)", 4352u, 9u},
        {"MP 16c 2048B x16 PIPE", 2048u, 16u},
        {"MP 16c 2048B x32 PIPE", 2048u, 32u},
        {"MP 16c 8704B x4 PIPE", 8704u, 4u},
        {"MP 16c 17408B x9 PIPE (depth 16)", 17408u, 9u},
    };
    for (extra = 0u; extra < sizeof(aligned) / sizeof(aligned[0]); extra++, index++) {
        set_condition(&out[index], aligned[extra].label, 16u, 8u, 3u, 0u, 16uLL * MIB,
                      aligned[extra].block, aligned[extra].depth, aligned[extra].depth,
                      STREAM_TARGET);
        out[index].base_offset = aligned[extra].base_offset;
    }
    for (extra = 0u; extra < sizeof(piped) / sizeof(piped[0]); extra++, index++) {
        set_condition(&out[index], piped[extra].label, 16u, 8u, 3u, 0u, 16uLL * MIB,
                      piped[extra].block, piped[extra].depth, piped[extra].depth,
                      STREAM_TARGET);
        out[index].direction = 2u;
    }
    return index;
}

/* ------------------------------------------------------------------------- */

/**
 * Put every condition of a sweep on the same fixed WALL-CLOCK window.
 *
 * This is what makes the aggregate and the overlap-window figures the same number instead
 * of a 60% band. With a fixed byte count the cores finish at different times — measured
 * spread across this grid is ~20% — so the sum of per-core rates assumes an overlap the
 * run does not have, and bytes-over-the-union counts a tail in which most cores are idle.
 */
static void set_sweep_duration(sweep_condition *conditions, uint32_t count,
                               uint32_t duration_ms)
{
    uint32_t index;

    for (index = 0u; index < count; index++)
        conditions[index].duration_ms = duration_ms;
}

/** How long each swept condition runs. 150 ms is ~200x the worst launch skew across 64
 *  cores and short enough that the 32-bit wall clock (5.37 s at 800 MHz) cannot wrap. */
#define SWEEP_DURATION_MS 150u

static bool stage_selected(const char *stages, const char *name)
{
    return strstr(stages, name) != NULL;
}

/* The placement stage (tt-place lane) lives in its own file so this one stops growing; it
 * uses the sweep driver above and nothing else. */
#include "dram_capacity_placement_sweeps.inc"


static void run_selected_sweeps(capacity_fixture *fixture, const char *stages)
{
    static sweep_condition conditions[MAX_CONDITIONS];
    uint32_t count;

    if (stage_selected(stages, "onebank")) {
        count = build_one_bank_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep A — ONE BANK: port vs bank, shared vs disjoint", conditions,
                  count);
    }
    if (stage_selected(stages, "workingset")) {
        count = build_working_set_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep B — WORKING SET, 6 cores on bank 0 across three ports",
                  conditions, count);
    }
    if (stage_selected(stages, "anomaly")) {
        count = build_anomaly_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep P — the 64 KiB anomaly: distinct addresses, in-flight depth, "
                           "span and request size", conditions, count);
    }
    if (stage_selected(stages, "address")) {
        count = build_anomaly_address_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep Q — the 64 KiB anomaly as an ADDRESS STREAM: stride period, "
                           "span at resolution, and one reader alone", conditions, count);
    }
    if (stage_selected(stages, "aggregate")) {
        count = build_aggregate_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep C — THE CARD: every bank, disjoint and large", conditions,
                  count);
    }
    if (stage_selected(stages, "blocksize")) {
        count = build_block_size_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep D — request size", conditions, count);
    }
    if (stage_selected(stages, "peak")) {
        count = build_peak_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep F — around the peak: core count x reads in flight",
                  conditions, count);
    }
    if (stage_selected(stages, "depth")) {
        count = build_depth_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep E — reads in flight", conditions, count);
    }
    if (stage_selected(stages, "cliff")) {
        count = build_cliff_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep G — the cliff, finely, from both factors", conditions, count);
    }
    if (stage_selected(stages, "spread")) {
        count = build_spread_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep H — equal reads in flight, redistributed over endpoints",
                  conditions, count);
    }
    if (stage_selected(stages, "distance")) {
        count = build_distance_sweep(conditions, fixture->first_index,
                                     ft_tt_coord_tensix_column_count(&fixture->session->grid));
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep I — NoC distance, endpoint held fixed", conditions, count);
    }
    if (stage_selected(stages, "ceiling")) {
        count = build_ceiling_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep O — one bank as high as it goes, then all eight",
                  conditions, count);
    }
    if (stage_selected(stages, "writes")) {
        count = build_write_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep N — writes against reads of the same shape", conditions,
                  count);
    }
    if (stage_selected(stages, "scatter")) {
        count = build_scatter_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep M — same endpoints, readers spread further apart",
                  conditions, count);
    }
    if (stage_selected(stages, "row")) {
        count = build_row_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep K — do two endpoints on one NoC row share a ceiling",
                  conditions, count);
    }
    if (stage_selected(stages, "burst")) {
        count = build_burst_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep L — reads in flight per core, apart from pass length",
                  conditions, count);
    }
    if (stage_selected(stages, "mmshape")) {
        count = build_mmshape_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep M — the decode matmul's read shape, no compute", conditions,
                  count);
    }
    run_placement_sweeps(fixture, stages, conditions);
    if (stage_selected(stages, "ports")) {
        count = build_ports_sweep(conditions);
        set_sweep_duration(conditions, count, SWEEP_DURATION_MS);
        run_sweep(fixture, "sweep J — which port, and how many per bank", conditions, count);
    }
}

/**
 * Every core this suite touched still BOOTS. Unconditional, even when an earlier stage
 * failed, because a read cannot tell a wedged core from a healthy idle one. The COMPUTE
 * half is `ft_tt_core probe`, a different triad, run by the suite script afterwards.
 */
static void run_stage_cores_still_boot(capacity_fixture *fixture)
{
    uint32_t slot;
    uint32_t failures = 0u;

    ft_tt_test_step("final — all %u cores still BOOT (run unconditionally)",
                    fixture->core_count);
    for (slot = 0u; slot < fixture->core_count; slot++) {
        int rc = boot_core(fixture, slot);

        if (rc != FT_TT_OK) {
            failures++;
            printf("        core index %u at translated (%u,%u): rc=%d (%s)\n",
                   fixture->core_index[slot], fixture->core[slot].x, fixture->core[slot].y, rc,
                   ft_tt_error_string(rc));
        }
    }
    ft_tt_test_check_fmt("every core this suite touched re-booted", failures == 0u,
                         "%u of %u failed", failures, fixture->core_count);
    printf("  The COMPUTE half of this check is `ft_tt_core probe`, which drives its own\n"
           "  vector_add triad and must be run next.\n\n");
}

/* ------------------------------------------------------------------------- */

/**
 * Resolve the core set, optionally SPREAD across the grid rather than contiguous.
 *
 * The spread is a control, not a convenience. Live index i is grid column i % 14, row
 * i / 14, so 64 contiguous indices are rows 0-5 of 10 — half the die, all of it on the same
 * side of the DRAM columns. If the card's aggregate ceiling were a NoC routing limit rather
 * than a memory one, spreading the same core COUNT over twice as many rows would move it.
 * Same count, same banks, same everything else: one variable.
 */
static int resolve_cores(capacity_fixture *fixture, uint32_t first_index, uint32_t count,
                         uint32_t stride)
{
    uint32_t slot;

    for (slot = 0u; slot < count; slot++) {
        uint32_t index = first_index + slot * stride;
        int rc = ft_tt_coord_tensix_by_index(&fixture->session->grid, index,
                                             FT_TT_COORD_TRANSLATED, &fixture->core[slot]);

        if (rc == FT_TT_OK)
            rc = ft_tt_coord_tensix_by_index(&fixture->session->grid, index, FT_TT_COORD_NOC0,
                                             &fixture->core_noc0[slot]);
        if (rc != FT_TT_OK)
            return rc;
        fixture->core_index[slot] = index;
    }
    fixture->core_count = count;
    return FT_TT_OK;
}

/**
 * Resolve an EXPLICIT core list, `FT_TT_DRAMCAP_CORE_LIST="71,96-110,20-35"` — booted slot k
 * is the k-th index named. The placement stage needs the production matmul members and a set
 * of idle cores, which no first/stride pair can name.
 */
static int resolve_core_list(capacity_fixture *fixture, const char *list)
{
    const char *cursor = list;
    uint32_t count = 0u;

    while (*cursor) {
        char *end = NULL;
        unsigned long first = strtoul(cursor, &end, 10);
        unsigned long last = first;
        unsigned long index;

        if (end == cursor)
            return FT_TT_EINVAL;
        if (*end == '-') {
            cursor = end + 1;
            last = strtoul(cursor, &end, 10);
            if (end == cursor || last < first)
                return FT_TT_EINVAL;
        }
        for (index = first; index <= last; index++) {
            int rc;

            if (count >= MAX_CORES)
                return FT_TT_EINVAL;
            rc = ft_tt_coord_tensix_by_index(&fixture->session->grid, (uint32_t)index,
                                             FT_TT_COORD_TRANSLATED, &fixture->core[count]);
            if (rc == FT_TT_OK)
                rc = ft_tt_coord_tensix_by_index(&fixture->session->grid, (uint32_t)index,
                                                 FT_TT_COORD_NOC0, &fixture->core_noc0[count]);
            if (rc != FT_TT_OK)
                return rc;
            fixture->core_index[count++] = (uint32_t)index;
        }
        cursor = *end == ',' ? end + 1 : end;
        if (*end != ',' && *end != '\0')
            return FT_TT_EINVAL;
    }
    fixture->core_count = count;
    return count > 0u ? FT_TT_OK : FT_TT_EINVAL;
}

static int resolve_dram(capacity_fixture *fixture)
{
    uint32_t bank;
    uint32_t port;

    if (ft_tt_coord_dram_bank_count(&fixture->session->grid) < MAX_BANKS ||
        ft_tt_coord_dram_ports_per_bank(&fixture->session->grid) < MAX_PORTS)
        return FT_TT_EUNAVAIL;
    for (bank = 0u; bank < MAX_BANKS; bank++)
        for (port = 0u; port < MAX_PORTS; port++) {
            int rc = ft_tt_coord_dram_port(&fixture->session->grid, bank, port,
                                           FT_TT_COORD_TRANSLATED, &fixture->dram[bank][port]);

            /* TRANSLATED is what a transfer must be aimed with; NOC 0 is the only one of
             * the two that says anything about where the endpoint physically IS, and the
             * routing questions are asked in that space. Both are derived from the same
             * telemetry — neither is probed. */
            if (rc == FT_TT_OK)
                rc = ft_tt_coord_dram_port(&fixture->session->grid, bank, port,
                                           FT_TT_COORD_NOC0, &fixture->dram_noc0[bank][port]);
            if (rc != FT_TT_OK)
                return rc;
        }
    return FT_TT_OK;
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_program_directory program;
    capacity_fixture fixture;
    const char *stages;
    const char *directory;
    uint32_t first_index;
    uint32_t slot;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <program-directory>\n", argv[0]);
        return 2;
    }
    directory = argv[1];
    stages = getenv("FT_TT_DRAMCAP_STAGES");
    if (stages == NULL || stages[0] == '\0')
        stages = "validate,onebank,workingset,aggregate";

    memset(&fixture, 0, sizeof(fixture));
    fixture.cycles_per_second = 800e6; /* replaced by stage 2's calibration */
    fixture.repetitions = environment_u32("FT_TT_DRAMCAP_REPS", 3u);
    if (fixture.repetitions == 0u || fixture.repetitions > MAX_REPETITIONS)
        fixture.repetitions = 3u;
    first_index = environment_u32("FT_TT_DRAMCAP_FIRST_INDEX", 8u);
    fixture.first_index = first_index;

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;
    fixture.session = &session;

    /* The session's own WC/BLOCK16 DRAM accessor, on whichever transport it opened. This
     * used to open a SECOND window through the tt-kmd context and refuse outright on udrv,
     * which made the suite unrunnable on a board where tt-kmd is never installed. Borrowing
     * is also the more honest arrangement: one window, and its byte counters are the ones
     * ft_tt_runtime already reports. */
    fixture.bulk = &session.runtime.bulk_dram;

    rc = resolve_dram(&fixture);
    if (rc == FT_TT_OK) {
        uint32_t wanted = environment_u32("FT_TT_DRAMCAP_CORES", 48u);

        if (wanted > MAX_CORES)
            wanted = MAX_CORES;
        if (getenv("FT_TT_DRAMCAP_CORE_LIST") && getenv("FT_TT_DRAMCAP_CORE_LIST")[0])
            rc = resolve_core_list(&fixture, getenv("FT_TT_DRAMCAP_CORE_LIST"));
        else
            rc = resolve_cores(&fixture, first_index, wanted,
                               environment_u32("FT_TT_DRAMCAP_CORE_STRIDE", 1u));
    }
    if (rc != FT_TT_OK) {
        ft_tt_test_check("cores and all 8 banks x 3 ports resolved from live telemetry", false,
                         ft_tt_error_string(rc));
        goto close_path;
    }

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

    printf("DRAM read capacity — %u cores from live index %u stride %u (last index %u), "
           "8 banks x 3 NoC ports\n", fixture.core_count, first_index,
           environment_u32("FT_TT_DRAMCAP_CORE_STRIDE", 1u),
           fixture.core_index[fixture.core_count - 1u]);
    printf("  stages: %s, %u repetitions\n", stages, fixture.repetitions);
    printf("  host load %.2f, board AICLK %u MHz at start\n", read_host_load(),
           read_board_aiclk_mhz());
    printf("  DRAM endpoints, derived from live telemetry — TRANSLATED is what a transfer is\n"
           "  aimed with, NOC 0 is where the endpoint physically sits:\n");
    for (slot = 0u; slot < MAX_BANKS; slot++)
        printf("  bank %u  translated (%2u,%2u) (%2u,%2u) (%2u,%2u)   noc0 (%2u,%2u) (%2u,%2u) "
               "(%2u,%2u)\n", slot, fixture.dram[slot][0].x, fixture.dram[slot][0].y,
               fixture.dram[slot][1].x, fixture.dram[slot][1].y, fixture.dram[slot][2].x,
               fixture.dram[slot][2].y, fixture.dram_noc0[slot][0].x,
               fixture.dram_noc0[slot][0].y, fixture.dram_noc0[slot][1].x,
               fixture.dram_noc0[slot][1].y, fixture.dram_noc0[slot][2].x,
               fixture.dram_noc0[slot][2].y);

    ft_tt_test_step("boot — %u cores, SEQUENTIALLY (concurrent boots at 12-way and above "
                    "wedge cores, DEV_NOTES.md §1)", fixture.core_count);
    for (slot = 0u; slot < fixture.core_count; slot++) {
        rc = boot_core(&fixture, slot);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("a core booted with the program loaded", false,
                                 "core index %u at translated (%u,%u): rc=%d (%s)",
                                 fixture.core_index[slot], fixture.core[slot].x,
                                 fixture.core[slot].y, rc, ft_tt_error_string(rc));
            goto health;
        }
    }
    ft_tt_test_check_fmt("every core booted with the program loaded", true, "%u cores",
                         fixture.core_count);

    if (stage_selected(stages, "validate")) {
        run_stage_transfer_round_trip(&fixture);
        run_stage_clock_calibration(&fixture);
        run_stage_bytes_moved(&fixture, 0u, 0u, 0u, 0u);
        /* The same proof driven through the placement and endpoint axes the cliff/NoC
         * sweeps depend on. Bank 3 port 2 from booted slot 5 is a different bank, a
         * different NoC endpoint and a different core than the run above; if any of the
         * three axes were ignored this goes red. */
        run_stage_bytes_moved(&fixture, 3u, 2u, 5u, 0u);
        /* And once more through the STRIDE axis, which the 64 KiB anomaly sweep depends on
         * in exactly the way the sweeps above depend on bank/port/core. The proof blocks are
         * staged 4 KiB apart and read 1 KiB at a time, so a harness that ignored the stride
         * would read four CONTIGUOUS KiB — three of which were never staged — and go red. */
        run_stage_bytes_moved(&fixture, 0u, 1u, 0u, 4096u);
        run_stage_writes_moved(&fixture);
        run_stage_falsifier(&fixture);
        run_stage_refusals(&fixture);
    }
    run_selected_sweeps(&fixture, stages);

health:
    run_stage_cores_still_boot(&fixture);
    printf("  host load %.2f, board AICLK %u MHz at end\n", read_host_load(),
           read_board_aiclk_mhz());

    ft_tt_program_directory_release(&program);
    ft_tt_firmware_set_release(&firmware);

close_path:
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
