/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_cooperative_matmul.c — one matmul across N Tensix cores, with the shared A block
 * multicast once instead of read from DRAM by every core.
 *
 * Every performance number in this backend is one Tensix of 140. This program is the first
 * that is not, and it is built to answer four questions in order:
 *
 *   1. Is a cooperative result the SAME result? Byte for byte against the single-core arm,
 *      not by PCC — PCC is blind to a consistent permutation and a split is exactly a
 *      permutation of where output tiles come from.
 *   2. What does splitting over N cores buy, WITHOUT the multicast? (the SPLIT arm)
 *   3. What does the MULTICAST add on top of that? (the COOPERATIVE arm)
 *   4. Where does it stop scaling, and what stops it?
 *
 * ===========================================================================
 * THE FOUR ARMS, AND WHY ALL FOUR ARE NEEDED
 * ===========================================================================
 * All four are ONE binary (kernels/cooperative_matmul_reader.c) one runtime word apart, so no
 * difference between them can be a codegen difference in disguise — the same rule
 * `weight_n_major` follows in matmul_reader.c.
 *
 *   SOLO         1 core, the whole Nt. The baseline every other number is a ratio of.
 *   SPLIT(N)     N cores, Nt/N columns each, every core reads A from DRAM itself. This is
 *                what "more cores" is worth on its own. WITHOUT it, a cooperative speedup
 *                cannot be attributed to the multicast at all.
 *   COOP(N)      N cores, Nt/N columns each, A multicast once per block by the leader.
 *                COOP/SPLIT is the multicast's contribution and nothing else.
 *   INDEPENDENT(N)  N cores each doing the WHOLE matmul into its own output. Not a split:
 *                it is N concurrent streams, and it is the DRAM/interconnect capacity
 *                question rather than this op's — how many cores it takes to saturate the
 *                memory system, which for a memory-bound decode matters more than how fast
 *                one matmul goes.
 *
 * ===========================================================================
 * WHAT IS TIMED, AND WHAT IS NOT
 * ===========================================================================
 * The reported op time is the window from "send GO to every member" to "every member reports
 * DONE", best of `repeats`. CB configuration and the runtime-arg blocks are staged ONCE per
 * arm, outside that window, because they are per-shape and not per-token: a serving path
 * dispatches an already-staged op, and the leanest go/done shape is what DEV_NOTES.md §6
 * records as 10.6-12.3 us. Staging is reported separately rather than hidden — at large N it
 * is the fixed cost that ends the scaling, and saying so is half the answer.
 *
 * Each reader ALSO records its own wall clock, so a device-side number exists that no host
 * scheduling can move. The card host is not a quiet host — a `pantheon serve` on the K80s takes
 * ~58% of two cores — and short best-of-N host measurements survive that while long ones do
 * not. Both columns are printed; when they disagree, the device one is the
 * op and the difference is dispatch.
 *
 * ===========================================================================
 * CORES, AND WHY THE GROUP CANNOT SIMPLY GROW
 * ===========================================================================
 * A multicast rectangle may not straddle NoC x = 8 and 9, which carry the ARC/L2CPU spine and
 * the east DRAM column, so the Tensix columns are 1..7 and 10..16 and a rectangle is at most
 * SEVEN wide. It is held to FOUR here because a 5-wide rectangle was measured on 2026-08-13 to
 * leave a minority of its receivers unbootable (1 of 5 on a 1x5, 3 of 10 on a 2x5) with width
 * and receiver count confounded and unresolved. `ft_tt_group_rectangle_coordinates()` refuses
 * the gap-crossing case outright; the 4-column limit is this program's, and deliberate.
 *
 * Usage:
 *   test_cooperative_matmul <program-directory> [repeats]
 * Environment:
 *   FT_TT_COOP_FIRST_INDEX   live Tensix index of the group's first core (default 100)
 *   FT_TT_COOP_ROWS          pool rows    (default 2)
 *   FT_TT_COOP_COLUMNS       pool columns (default 4; DO NOT raise past 4, see above)
 *   FT_TT_COOP_DRAM_A/B/C    DRAM logical banks for A, B and the output (default 5, 6, 7)
 *   FT_TT_FIRMWARE_DIR       where the five firmware ELFs live (the same directory)
 *
 * Build:
 *   make -f Makefile.cooperative PHASE1B=1 all
 */

/* clock_gettime(CLOCK_MONOTONIC) is POSIX.1-2008, not C99. */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_dram_shard.h"
#include "../ft_tt_dram_stripe.h"
#include "../ft_tt_group_shared_operand.h"
#include "../ft_tt_noc.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_matmul_cooperative.h"
#include "../ft_tt_tilize.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define TILE_DIM      FT_TT_OP_TILE_DIM
#define TILE_ELEMENTS (TILE_DIM * TILE_DIM)
#define PCC_ACCEPT    0.99
#define POISON_BYTE   0xA5
#define RUN_TIMEOUT_MS 30000u

/** Default DRAM logical banks. 5..7 rather than test_ops.c's 2..4 and the multicast bench's 3,
 *  so three suites can be live on one card without one overwriting another's operands. */
/** The DRAM banks A, B and the output live in. Separate banks by default: a single bank is
 *  the first thing that saturates, and putting two operands in one would make every scaling
 *  number a bank measurement without saying so. FT_TT_COOP_DRAM_B + FT_TT_COOP_DRAM_B_BASE
 *  together let a run deliberately collapse them into one, which is the experiment that
 *  separates a per-bank wall from a global one. */
#define DEFAULT_DRAM_BANK_A 5u
#define DEFAULT_DRAM_BANK_B 6u
#define DEFAULT_DRAM_BANK_OUTPUT 7u
#define DRAM_BUFFER_BASE 0x00100000u

/** Where the BANK-STRIPED copies go. Well clear of DRAM_BUFFER_BASE and of the weight base,
 *  because bank striping puts a member's panel in whichever bank its index names - including
 *  the banks the un-striped arms use for the activation, the weight and the OUTPUT - and two
 *  arms of one measurement may not overwrite each other's operands. 0x10000000 and 0x20000000
 *  are 268 and 537 MiB into a 4,278 MiB bank. */
#define SHARD_WEIGHT_BASE     0x10000000u
#define SHARD_ACTIVATION_BASE 0x20000000u

/**
 * The widest COOPERATIVE group this suite will run on silicon.
 *
 * MEASURED, twice, and it costs a core each time: a 2x4 group — seven receivers — left core
 * 115 unbootable on 2026-08-13 and again on 2026-08-14, both times at the re-boot AFTER a
 * clean, byte-exact run, both times the same slot, with the `soft reset 0x00047000` signature.
 * Groups of four or fewer ran clean through every stage on both days. Recovering it is a
 * WHOLE-CARD reset, so the sweep stops rather than pays that again; SPLIT and INDEPENDENT
 * issue no multicast and are swept to the full pool.
 */
#define COOP_MAX_MEMBERS_ON_SILICON 4u

#define DEFAULT_FIRST_INDEX 100u
#define DEFAULT_POOL_ROWS 2u
#define DEFAULT_POOL_COLUMNS 4u
#define DEFAULT_REPEATS 5u

/** Bounded, per resident-kernel rule 4: a wait that expires must report, never hang. At
 *  800 MHz a spin is a few cycles, so 200 million is well under a second and far longer than
 *  any legitimate wait here. */
#define SPIN_LIMIT 200000000u

/** The prefetch depth every shape runs at, shrunk to a divisor of Kt by the plan. 48 is what
 *  the single-core numbers on record were measured at, and it makes the shared A block 98304
 *  bytes — the size at which multicast was measured to cross over at N=3. */
#define PREFETCH_TILES 48u

/** One Tensix's derived peak at HiFi4 and this card's measured 800 MHz busy clock:
 *  2048 FLOP/cycle x 800e6. DERIVED, not measured — every "% of one Tensix" here inherits
 *  that. */
#define ONE_TENSIX_GFLOPS 1638.4

/** The composed one-core forward pass this project quotes: 3.745 ms/layer x 24 layers +
 *  22.0 ms lm_head = 111.9 ms = 8.94 tok/s. No end-to-end pass has ever been run; it is a
 *  composition of measured per-op costs, and every tok/s projection below inherits that. */
#define FORWARD_PASS_MS 111.9
#define LM_HEAD_MS 22.0
#define MODEL_LAYERS 24u

/* ------------------------------------------------------------------------- */

typedef struct coop_fixture {
    ft_tt_hardware_session *session;
    ft_tt_noc_path *dram_path;
    const ft_tt_program_directory *program;
    const ft_tt_firmware_set *firmware;
    ft_tt_noc_coord dram_a;
    ft_tt_noc_coord dram_b;
    ft_tt_noc_coord dram_output;
    /** Every NoC door of the activation's bank and of the weight's bank, from live telemetry.
     *  The STRIPED arms rotate members over these; the FLAT arms use dram_a/dram_b, which are
     *  port 0 of each — `ft_tt_coord_dram_bank()` IS port 0. */
    ft_tt_dram_port_set dram_a_ports;
    ft_tt_dram_port_set dram_b_ports;
    /**
     * Concurrent streams aimed at the busiest and the quietest DRAM bank by the LAST bank
     * shard plan staged, from ft_tt_dram_shard.h.
     *
     * Carried here so the sweep can print it, because it is the BALANCE rule made visible and
     * it is what the bank-striped arm's wall clock actually tracks: with 8 banks and two
     * operands per member the busiest bank holds ceil(2N/8) streams, and the arm's wall clock
     * steps at exactly the N where that number does. A plan whose two figures differ by more
     * than one is unbalanced, and an unbalanced map measured 30% worse than either
     * neighbouring balanced count (plan Phase 6.7).
     */
    uint32_t shard_busiest_bank;
    uint32_t shard_quietest_bank;
    /** The POOL: pool_rows x pool_columns cores in row-major order, booted once. */
    ft_tt_noc_coord members[FT_TT_OP_COOP_MAX_MEMBERS];
    uint32_t member_count;
    uint32_t first_index;
    uint32_t pool_rows;
    uint32_t pool_columns;
    uint32_t repeats;
    uint32_t weight_base;
    /** Spread stage (FT_TT_COOP_SPREAD_ROUNDS). 0 leaves the suite exactly as it was. */
    uint32_t spread_rounds;
    uint32_t spread_inner;
    bool sample_device_cycles;
    ft_tt_launch_config config[FT_TT_OP_COOP_MAX_MEMBERS];
    /**
     * The cores ONE run uses, and which pool slot each of them is.
     *
     * A run is not simply "the first N of the pool". The pool is stored row-major, so the
     * first 6 cores of a 2x4 pool are four of row 0 and two of row 1 — which is NOT a
     * rectangle, and a multicast to its bounding box would write into the two cores the wrap
     * picked up. `ft_tt_multicast_rectangle_from_cores()` refuses it, which is how this bug
     * was found rather than broadcast; select_rectangle() is the fix.
     */
    ft_tt_noc_coord active[FT_TT_OP_COOP_MAX_MEMBERS];
    uint32_t active_slot[FT_TT_OP_COOP_MAX_MEMBERS];
} coop_fixture;

/** The top-left rows x columns sub-block of the pool, which IS a rectangle because the pool
 *  is one. */
static void select_rectangle(coop_fixture *fixture, uint32_t rows, uint32_t columns)
{
    uint32_t next = 0u;
    uint32_t row;
    uint32_t column;

    for (row = 0u; row < rows; row++) {
        for (column = 0u; column < columns; column++) {
            uint32_t slot = row * fixture->pool_columns + column;

            fixture->active_slot[next] = slot;
            fixture->active[next] = fixture->members[slot];
            next++;
        }
    }
}

/** The first `count` cores of the pool, in pool order. Only for arms that never multicast —
 *  they need no rectangle, and demanding one would refuse group sizes that are perfectly
 *  legal for independent work. */
static void select_linear(coop_fixture *fixture, uint32_t count)
{
    uint32_t index;

    for (index = 0u; index < count; index++) {
        fixture->active_slot[index] = index;
        fixture->active[index] = fixture->members[index];
    }
}

/**
 * How many repeats of one arm the per-repeat sample arrays can hold.
 *
 * Best/median/worst is a summary, and a summary cannot tell a BIMODAL result (two behaviours,
 * one of them slow) from a broad unimodal one (contention). Only the distribution can, so the
 * spread stage keeps every sample rather than three order statistics of them.
 */
#define MAX_SPREAD_SAMPLES 64u

/** What one arm of one run produced. */
typedef struct arm_result {
    int rc;
    double best_ms;          /**< best-of-repeats GO -> all DONE, host clock */
    double worst_ms;         /**< the slowest repeat of the same run — the spread, reported */
    /** Every repeat's own host wall clock and, when the fixture asks for it, the same
     *  repeat's DEVICE cycles. Two clocks on ONE run is what separates a host-dispatch
     *  spread from an on-card one: a wall clock that moves while the device cycles do not
     *  is the 2-core host, and no histogram of a single clock can say that. */
    double sample_ms[MAX_SPREAD_SAMPLES];
    uint32_t sample_cycles[MAX_SPREAD_SAMPLES];
    uint32_t sample_count;
    double staging_ms;       /**< CB config + runtime args for every member, once */
    /** DRAM doors this arm's readers used, and how many pairs of them share a NoC row. 0 and 0
     *  on an un-striped arm, which is how the report distinguishes the two. */
    uint32_t distinct_endpoints;
    uint32_t row_collisions;
    uint32_t device_cycles;  /**< the slowest member's reader, entry to exit */
    uint32_t first_error;    /**< first non-zero kernel error word, in member order */
    uint32_t blocks_short;   /**< members that did not complete every block */
    /** Each member's own error word. A group red proof has to make exactly ONE member wrong
     *  and show the others stay green, and a single first-error cannot express that — a leader
     *  that timed out waiting for a refusing receiver reports first and hides it. */
    int32_t member_error[FT_TT_OP_COOP_MAX_MEMBERS];
    bool bytes_match_solo;
    /** Byte offset of the first difference from the solo result, or output_bytes if none.
     *  Which MEMBER'S PANEL that offset falls in is what says whether one member was wrong or
     *  all of them were. */
    size_t first_difference;
} arm_result;

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

static int compare_double(const void *left, const void *right)
{
    double a = *(const double *)left;
    double b = *(const double *)right;

    return (a > b) - (a < b);
}

/** One-minute load average. `pantheon` serves on the K80s of this same 2-core box and takes
 *  ~58% of it, so a spread stage that does not record the host is measuring two things. */
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

static uint64_t random_state = 1u;

static void random_seed(uint64_t seed)
{
    random_state = seed ? seed : 1u;
}

static float random_value(void)
{
    random_state ^= random_state >> 12;
    random_state ^= random_state << 25;
    random_state ^= random_state >> 27;
    return (float)((double)(random_state * 0x2545F4914F6CDD1Dull >> 11) / (double)(1ull << 53) *
                       2.0 -
                   1.0);
}

static void round_to_bf16(float *matrix, size_t count)
{
    size_t index;

    for (index = 0; index < count; index++)
        matrix[index] = ft_float_from_bf16(ft_bf16_from_float(matrix[index]));
}

/** Row-major fp32 -> face-ordered bf16 tiles. */
static void to_bf16_tiles(const float *matrix, uint32_t rows, uint32_t cols, uint16_t *out)
{
    uint16_t *strip = malloc((size_t)cols * TILE_DIM * sizeof(uint16_t));
    uint32_t tiles_across = ft_tt_tile_count_for(cols);
    uint32_t block;

    if (!strip)
        return;
    for (block = 0; block < rows / TILE_DIM; block++) {
        uint32_t row;
        uint32_t column;

        for (row = 0; row < TILE_DIM; row++)
            for (column = 0; column < cols; column++)
                strip[row * cols + column] =
                    ft_bf16_from_float(matrix[(block * TILE_DIM + row) * cols + column]);
        ft_tt_tilize_strip(strip, sizeof(uint16_t), TILE_DIM, cols, cols * sizeof(uint16_t),
                           out + (size_t)block * tiles_across * TILE_ELEMENTS);
    }
    free(strip);
}

static void from_bf16_tiles(const uint16_t *tiles, uint32_t rows, uint32_t cols, float *out)
{
    uint16_t *strip = malloc((size_t)cols * TILE_DIM * sizeof(uint16_t));
    uint32_t tiles_across = ft_tt_tile_count_for(cols);
    uint32_t block;

    if (!strip)
        return;
    for (block = 0; block < rows / TILE_DIM; block++) {
        uint32_t row;
        uint32_t column;

        ft_tt_untilize_strip(tiles + (size_t)block * tiles_across * TILE_ELEMENTS,
                             sizeof(uint16_t), TILE_DIM, cols, cols * sizeof(uint16_t), strip);
        for (row = 0; row < TILE_DIM; row++)
            for (column = 0; column < cols; column++)
                out[(block * TILE_DIM + row) * cols + column] =
                    ft_float_from_bf16(strip[row * cols + column]);
    }
    free(strip);
}

static void reference_matmul(const float *a, const float *b, uint32_t m, uint32_t k, uint32_t n,
                             float *out)
{
    double *accumulator = malloc((size_t)n * sizeof(double));
    uint32_t row;

    if (!accumulator)
        return;
    for (row = 0; row < m; row++) {
        uint32_t column;
        uint32_t inner;

        for (column = 0; column < n; column++)
            accumulator[column] = 0.0;
        for (inner = 0; inner < k; inner++) {
            double scale = (double)a[(size_t)row * k + inner];
            const float *b_row = b + (size_t)inner * n;

            for (column = 0; column < n; column++)
                accumulator[column] += scale * (double)b_row[column];
        }
        for (column = 0; column < n; column++)
            out[(size_t)row * n + column] = (float)accumulator[column];
    }
    free(accumulator);
}

static double pearson(const float *a, const float *b, size_t count)
{
    double mean_a = 0.0;
    double mean_b = 0.0;
    double cov = 0.0;
    double var_a = 0.0;
    double var_b = 0.0;
    size_t index;

    if (count == 0u)
        return 0.0;
    for (index = 0; index < count; index++) {
        mean_a += a[index];
        mean_b += b[index];
    }
    mean_a /= (double)count;
    mean_b /= (double)count;
    for (index = 0; index < count; index++) {
        double da = (double)a[index] - mean_a;
        double db = (double)b[index] - mean_b;

        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if (var_a <= 0.0 || var_b <= 0.0)
        return 0.0;
    return cov / sqrt(var_a * var_b);
}

/** GGUF-order (K-major) tile grid -> N-major, i.e. tile index nt*Kt + kt. Every arm here runs
 *  N-major, because that is what makes a prefetch block's B tiles one transaction AND what
 *  makes a member's column slice a CONTIGUOUS run of tiles. */
static void permute_tiles_to_n_major(const void *source, void *destination, uint32_t tile_rows,
                                     uint32_t tile_cols, size_t tile_bytes)
{
    const unsigned char *from = source;
    unsigned char *to = destination;
    uint32_t kt;

    for (kt = 0; kt < tile_rows; kt++) {
        uint32_t nt;

        for (nt = 0; nt < tile_cols; nt++)
            memcpy(to + ((size_t)nt * tile_rows + kt) * tile_bytes,
                   from + ((size_t)kt * tile_cols + nt) * tile_bytes, tile_bytes);
    }
}

static size_t poison_remaining(const unsigned char *bytes, size_t count)
{
    size_t left = 0u;
    size_t index;

    for (index = 0; index < count; index++)
        if (bytes[index] == POISON_BYTE)
            left++;
    return left;
}

/* ------------------------------------------------------------------------- */
/* Boot, stage, launch                                                       */
/* ------------------------------------------------------------------------- */

/**
 * Bring one core from unknown to "firmware running, program loaded".
 *
 * SEQUENTIAL by construction — one process, one core at a time. At 12-way and above, firmware
 * boots from INDEPENDENT host processes start returning `device I/O failed` and leave cores
 * mid-handshake; it wedged cores 110 and 111 during the DRAM fan-out. This
 * program is one process and never overlaps a boot.
 */
static int boot_core(coop_fixture *fixture, uint32_t slot)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    const uint16_t x = (uint16_t)fixture->members[slot].x;
    const uint16_t y = (uint16_t)fixture->members[slot].y;
    int rc;

    rc = ft_tt_core_park(noc, x, y);
    if (rc == FT_TT_OK)
        rc = ft_tt_core_boot_firmware(noc, x, y, fixture->firmware, NULL);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_op_program_load(noc, x, y, &fixture->program->program, NULL,
                               &fixture->config[slot]);
    if (rc != FT_TT_OK)
        return rc;
    fixture->config[slot].local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    /* The reader is on BRISC and was built with NOC_INDEX=0; the firmware re-reads this from
     * the launch message at run time, so the two must agree. */
    fixture->config[slot].brisc_noc_id = 0u;
    return FT_TT_OK;
}

/** Poison the status block and clear the stop flag, so "never ran" is its own bit pattern
 *  rather than the same zero as "finished". */
static int poison_control_block(const coop_fixture *fixture, uint32_t slot)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    uint32_t poison[FTCM_STATUS_WORDS];
    uint32_t zero = 0u;
    unsigned index;
    int rc;

    for (index = 0u; index < FTCM_STATUS_WORDS; index++)
        poison[index] = FTCM_STATUS_POISON;
    rc = noc->write(noc->user, (uint16_t)fixture->active[slot].x,
                    (uint16_t)fixture->active[slot].y, ft_tt_op_coop_status_address(), poison,
                    sizeof(poison));
    if (rc == FT_TT_OK)
        rc = noc->write(noc->user, (uint16_t)fixture->active[slot].x,
                        (uint16_t)fixture->active[slot].y, ft_tt_op_coop_stop_flag_address(),
                        &zero, sizeof(zero));
    return rc;
}

/**
 * Everything before the kick, for every member: the CB descriptor array, the reader's args,
 * the compute third's three and the writer's four.
 *
 * `arg_override` lets a red proof plant a deliberately wrong word on ONE member without a
 * second staging path — a mutation that had to be built specially would be testing the
 * mutation harness rather than the op.
 */
static int stage_members(const coop_fixture *fixture,
                         const ft_tt_op_matmul_cooperative_plan *plan,
                         const ft_tt_op_matmul_cooperative_request *request,
                         uint32_t override_member, uint32_t override_index,
                         uint32_t override_value)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    uint32_t slot;
    int rc = FT_TT_OK;

    for (slot = 0u; slot < plan->member_count && rc == FT_TT_OK; slot++) {
        uint32_t reader_args[FTCM_ARG_COUNT];
        uint32_t compute_args[3];
        uint32_t writer_args[4];
        ft_tt_op_args args[5];
        ft_tt_launch_config config = fixture->config[fixture->active_slot[slot]];

        rc = ft_tt_op_matmul_cooperative_reader_arguments(plan, request, fixture->active, slot,
                                                          reader_args, FTCM_ARG_COUNT);
        if (rc == FT_TT_OK)
            rc = ft_tt_op_matmul_cooperative_compute_arguments(plan, request, slot, compute_args,
                                                               3u);
        if (rc == FT_TT_OK)
            rc = ft_tt_op_matmul_cooperative_writer_arguments(plan, request, slot, writer_args,
                                                              4u);
        if (rc != FT_TT_OK)
            break;
        if (slot == override_member && override_index < FTCM_ARG_COUNT)
            reader_args[override_index] = override_value;

        config.local_cb_mask = plan->cb_mask;
        args[0].processor = FT_TT_PROC_DM0;
        args[0].values = reader_args;
        args[0].count = FTCM_ARG_COUNT;
        args[1].processor = FT_TT_PROC_DM1;
        args[1].values = writer_args;
        args[1].count = 4u;
        args[2].processor = FT_TT_PROC_MATH0;
        args[2].values = compute_args;
        args[2].count = 3u;
        args[3] = args[2];
        args[3].processor = FT_TT_PROC_MATH1;
        args[4] = args[2];
        args[4].processor = FT_TT_PROC_MATH2;

        rc = poison_control_block(fixture, slot);
        if (rc == FT_TT_OK)
            rc = ft_tt_op_stage(noc, (uint16_t)fixture->active[slot].x,
                                (uint16_t)fixture->active[slot].y, &config, plan->cb_buffer,
                                sizeof(plan->cb_buffer), args, 5u);
    }
    return rc;
}

/**
 * The timed window: GO to every member, then wait for every member.
 *
 * All the GOs go out before any DONE is polled, so the members overlap. The serialisation
 * that remains — one host write per member over an x1 link — is real and is part of the
 * number; it is also exactly the cost the resident work queue removes (1.36 us an op against
 * this path's 10.6-12.3), which is why it is reported rather than subtracted.
 */
static int go_and_wait(const coop_fixture *fixture, uint32_t member_count, double *out_ms)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    double started;
    uint32_t slot;
    int rc = FT_TT_OK;

    started = monotonic_seconds();
    for (slot = 0u; slot < member_count && rc == FT_TT_OK; slot++)
        rc = ft_tt_launch_send_go(noc, (uint16_t)fixture->active[slot].x,
                                  (uint16_t)fixture->active[slot].y);
    for (slot = 0u; slot < member_count && rc == FT_TT_OK; slot++)
        rc = ft_tt_launch_wait_done(noc, (uint16_t)fixture->active[slot].x,
                                    (uint16_t)fixture->active[slot].y, RUN_TIMEOUT_MS, NULL);
    *out_ms = (monotonic_seconds() - started) * 1e3;

    /* BRISC asks TRISC0 to re-zero all 64 circular-buffer counter registers AFTER writing
     * RUN_MSG_DONE, so a host that relaunches without waiting for that can have its next CB
     * configuration zeroed underneath it (ft_tt_hardware_session.h). Outside the timed
     * window, and on every member. */
    for (slot = 0u; slot < member_count && rc == FT_TT_OK; slot++)
        rc = ft_tt_core_wait_sync_registers_ready(noc, (uint16_t)fixture->active[slot].x,
                                                  (uint16_t)fixture->active[slot].y, 2000u);
    return rc;
}

/** Read every member's status block and fold it into the result. */
static int collect_status(const coop_fixture *fixture,
                          const ft_tt_op_matmul_cooperative_plan *plan, arm_result *out)
{
    const ft_tt_noc_access *noc = &fixture->session->noc;
    uint32_t slot;

    for (slot = 0u; slot < plan->member_count; slot++) {
        uint32_t status[FTCM_STATUS_WORDS];
        int rc = noc->read(noc->user, (uint16_t)fixture->active[slot].x,
                           (uint16_t)fixture->active[slot].y, ft_tt_op_coop_status_address(),
                           status, sizeof(status));

        if (rc != FT_TT_OK)
            return rc;
        if (status[FTCM_STATUS_MAGIC_WORD] != FTCM_STATUS_MAGIC) {
            /* Still the host poison: the kernel never ran, or never reached its publish. */
            if (out->first_error == 0u)
                out->first_error = status[FTCM_STATUS_MAGIC_WORD];
            out->member_error[slot] = (int32_t)status[FTCM_STATUS_MAGIC_WORD];
            out->blocks_short++;
            continue;
        }
        out->member_error[slot] = (int32_t)status[FTCM_STATUS_ERROR];
        if (status[FTCM_STATUS_ERROR] != 0u && out->first_error == 0u)
            out->first_error = status[FTCM_STATUS_ERROR];
        if (status[FTCM_STATUS_BLOCKS_COMPLETED] != status[FTCM_STATUS_BLOCKS_EXPECTED])
            out->blocks_short++;
        {
            uint32_t cycles =
                status[FTCM_STATUS_CLOCK_AT_EXIT] - status[FTCM_STATUS_CLOCK_AT_ENTRY];

            if (cycles > out->device_cycles)
                out->device_cycles = cycles;
        }
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* One arm                                                                   */
/* ------------------------------------------------------------------------- */

/** The shape and buffers one sweep entry shares across all its arms. */
typedef struct shape_run {
    const char *label;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t tile_rows;
    uint32_t tile_inner;
    uint32_t tile_cols;
    size_t output_bytes;  /**< bytes of ONE full result */
    uint16_t *solo_tiles; /**< the single-core result: the byte-comparison reference */
    uint16_t *readback;   /**< room for FT_TT_OP_COOP_MAX_MEMBERS results, for INDEPENDENT */
    float *reference;     /**< the CPU fp32 answer, used only to score the SOLO arm */
    /** The staged operands, KEPT rather than freed after the first upload: a bank-striped arm
     *  re-lays them out for every group size, because which bytes go in which bank is a
     *  function of N. Re-tilizing them each time would put host arithmetic in the loop. */
    uint16_t *a_tiles;
    uint16_t *b_n_major;
} shape_run;

/** Which addressing an arm uses. The three are the same bytes reached three ways, not three
 *  algorithms — every one of them is byte-compared against the same single-core result. */
typedef enum stripe_mode {
    /** One door per operand, both operands in one bank each. The recorded baseline, and what
     *  `ft_tt_coord_dram_bank()` gives every caller in the backend today. */
    STRIPE_NONE = 0,
    /** PART A. The same two banks, but all three doors of each, members rotated over them.
     *  Moves no data — a port is a door into one memory. Ceiling: one bank, ~64 GB/s. */
    STRIPE_PORTS,
    /** PART B. A bank per member: its weight PANEL relocated into its own bank and a copy of
     *  the activation placed in another, each on its own door. Moves data. Ceiling: N banks. */
    STRIPE_BANKS
} stripe_mode;

static const char *const g_stripe_suffix[] = {"", "+ports", "+banks"};
static const char *const g_stripe_name[] = {"one door each", "port-rotated",
                                           "bank-striped"};

/**
 * Stage, poison the output, run `repeats` times, read back, and compare.
 *
 * The output poison is written ONCE before the first repeat and not refreshed: every repeat
 * writes the same answer to the same place, so a repeat that silently did nothing still leaves
 * the previous one's result, and refreshing the poison would let the last repeat alone carry
 * the whole check (test_ops.c makes the same choice for the same reason).
 */
static int run_arm(coop_fixture *fixture, shape_run *shape,
                   const ft_tt_op_matmul_cooperative_request *request, uint32_t member_count,
                   uint32_t override_member, uint32_t override_index, uint32_t override_value,
                   bool compare_to_solo, arm_result *out)
{
    ft_tt_op_matmul_cooperative_plan plan;
    size_t total_bytes;
    double started;
    uint32_t repeat;
    int rc;

    memset(out, 0, sizeof(*out));
    out->bytes_match_solo = true;

    rc = ft_tt_op_matmul_cooperative_plan_build(fixture->active, member_count,
                                                0u /* leader = member 0 */, request, &plan);
    if (rc != FT_TT_OK) {
        out->rc = rc;
        return rc;
    }
    out->distinct_endpoints = plan.distinct_endpoints;
    out->row_collisions = plan.row_collisions;

    /* INDEPENDENT writes member_count whole results side by side; the split arms write one. */
    total_bytes = (request->mode == FT_TT_OP_MATMUL_INDEPENDENT)
                      ? shape->output_bytes * member_count
                      : shape->output_bytes;

    memset(shape->readback, POISON_BYTE, total_bytes);
    rc = ft_tt_write_dram(fixture->dram_path, fixture->dram_output, DRAM_BUFFER_BASE,
                          shape->readback, total_bytes);
    if (rc != FT_TT_OK) {
        out->rc = rc;
        return rc;
    }

    if (plan.shared_valid) {
        rc = ft_tt_group_shared_operand_reset(&fixture->session->noc, &plan.shared,
                                              fixture->active, member_count);
        if (rc != FT_TT_OK) {
            out->rc = rc;
            return rc;
        }
    }

    started = monotonic_seconds();
    rc = stage_members(fixture, &plan, request, override_member, override_index, override_value);
    out->staging_ms = (monotonic_seconds() - started) * 1e3;
    if (rc != FT_TT_OK) {
        out->rc = rc;
        return rc;
    }

    for (repeat = 0u; repeat < fixture->repeats; repeat++) {
        double milliseconds = 0.0;

        /* The barrier state has to be reset before EVERY run, not once at setup: a ready
         * semaphore left non-zero by an abandoned run lets the next leader believe receivers
         * are ready that have not started, and unlike a completion block a stale semaphore
         * value is indistinguishable from a fresh one (ft_tt_group_shared_operand.h). */
        if (plan.shared_valid && repeat > 0u) {
            rc = ft_tt_group_shared_operand_reset(&fixture->session->noc, &plan.shared,
                                                  fixture->active, member_count);
            if (rc != FT_TT_OK)
                break;
        }
        rc = go_and_wait(fixture, member_count, &milliseconds);
        if (rc != FT_TT_OK)
            break;
        if (repeat == 0u || milliseconds < out->best_ms)
            out->best_ms = milliseconds;
        if (repeat == 0u || milliseconds > out->worst_ms)
            out->worst_ms = milliseconds;
        if (out->sample_count < MAX_SPREAD_SAMPLES) {
            out->sample_ms[out->sample_count] = milliseconds;
            /* Read into a SCRATCH result, not `out`: collect_status folds error and
             * short-block state in, and a per-repeat call would make the arm's final verdict
             * the union of every repeat's rather than the last one's. It is a pure read, and
             * it is outside the timed window. */
            if (fixture->sample_device_cycles) {
                arm_result probe;

                memset(&probe, 0, sizeof(probe));
                if (collect_status(fixture, &plan, &probe) == FT_TT_OK)
                    out->sample_cycles[out->sample_count] = probe.device_cycles;
            }
            out->sample_count++;
        }
    }
    out->rc = rc;
    if (rc != FT_TT_OK)
        return rc;

    rc = collect_status(fixture, &plan, out);
    if (rc != FT_TT_OK) {
        out->rc = rc;
        return rc;
    }

    rc = ft_tt_read_dram(fixture->dram_path, fixture->dram_output, DRAM_BUFFER_BASE,
                         shape->readback, total_bytes);
    if (rc != FT_TT_OK) {
        out->rc = rc;
        return rc;
    }
    out->first_difference = shape->output_bytes;
    if (poison_remaining((const unsigned char *)shape->readback, total_bytes) * 100u >
        total_bytes)
        out->bytes_match_solo = false;
    if (compare_to_solo && shape->solo_tiles != NULL) {
        const unsigned char *left = (const unsigned char *)shape->solo_tiles;
        const unsigned char *right = (const unsigned char *)shape->readback;
        size_t offset;

        for (offset = 0u; offset < shape->output_bytes; offset++) {
            if (left[offset] != right[offset]) {
                out->first_difference = offset;
                out->bytes_match_solo = false;
                break;
            }
        }
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* Reporting                                                                 */
/* ------------------------------------------------------------------------- */

static double gflops_of(const shape_run *shape, double milliseconds)
{
    double flops = 2.0 * (double)shape->m * (double)shape->k * (double)shape->n;

    return milliseconds > 0.0 ? flops / (milliseconds * 1e6) : 0.0;
}

/**
 * The bytes this arm's readers actually pull out of DRAM for ONE result.
 *
 * Exact, from what the reader issues rather than from the shape alone: every member reads its
 * own B panel, and every member reads A for each of its own columns — except in COOPERATIVE,
 * where only the LEADER reads A and the rest receive it over the NoC. Getting that wrong would
 * make the striped and un-striped GB/s incomparable, which is the whole column.
 */
static double dram_read_bytes(const shape_run *shape, ft_tt_op_matmul_cooperative_mode mode,
                              uint32_t members)
{
    const double tile_bytes = (double)FT_TT_OP_TILE_BYTES_BF16;
    const uint32_t columns =
        (mode == FT_TT_OP_MATMUL_INDEPENDENT) ? shape->tile_cols : shape->tile_cols / members;
    const double a_per_member = (double)shape->tile_rows * shape->tile_inner * columns;
    const double b_per_member = (double)shape->tile_inner * columns;
    const double a_readers = (mode == FT_TT_OP_MATMUL_COOPERATIVE) ? 1.0 : (double)members;

    return (a_readers * a_per_member + (double)members * b_per_member) * tile_bytes;
}

static void print_row(const char *arm, uint32_t n, uint32_t rows, uint32_t columns,
                      const shape_run *shape, ft_tt_op_matmul_cooperative_mode mode,
                      const arm_result *result, double solo_ms)
{
    double aggregate = gflops_of(shape, result->best_ms);
    double read_gbs = result->best_ms > 0.0
                          ? dram_read_bytes(shape, mode, n) / (result->best_ms * 1e-3) / 1e9
                          : 0.0;

    printf("  %-12s %2u %ux%u | %8.3f | %6.1f%% | %9.2f | %8.1f | %2u/%u | %6.2fx | %s\n", arm,
           n, rows, columns, result->best_ms,
           result->best_ms > 0.0 ? (result->worst_ms - result->best_ms) / result->best_ms * 100.0
                                 : 0.0,
           aggregate, read_gbs, result->distinct_endpoints, result->row_collisions,
           result->best_ms > 0.0 ? solo_ms / result->best_ms : 0.0,
           result->first_error != 0u ? "KERNEL ERROR"
                                     : (result->bytes_match_solo ? "bytes == solo" : "BYTES DIFFER"));
    fflush(stdout);
}

/* ------------------------------------------------------------------------- */
/* Stages                                                                    */
/* ------------------------------------------------------------------------- */

/** The rectangle shapes the sweep walks, widest-last so a failure is reported at the smallest
 *  N that shows it. Never wider than 4 columns — see the file header. */
typedef struct sweep_shape {
    uint32_t rows;
    uint32_t columns;
} sweep_shape;

static const sweep_shape g_sweep[] = {{1u, 2u}, {1u, 3u}, {1u, 4u}, {2u, 2u},
                                      {2u, 3u}, {2u, 4u}, {3u, 4u}, {4u, 4u}};
#define SWEEP_COUNT ((uint32_t)(sizeof(g_sweep) / sizeof(g_sweep[0])))

/** Operand order inside a shard plan. It is the order the DOORS are assigned in — the
 *  planner is greedy in stream order — so it is named rather than left to a literal. */
#define SHARD_OPERAND_WEIGHT 0u
#define SHARD_OPERAND_ACTIVATION 1u
#define SHARD_OPERAND_COUNT 2u

/**
 * PART B — lay the operands out ACROSS BANKS for one group size, and stage the bytes.
 *
 * The WEIGHT is PARTITIONED and the ACTIVATION is REPLICATED, and the asymmetry is the whole
 * design: the weight is stored N-major, so member c's column slice is ALREADY a contiguous run
 * of bytes and relocating it is a copy rather than a re-tiling; the activation is needed whole
 * by every member and is 98 KiB against the weight's 14 MiB, so a copy per bank is affordable.
 *
 * **The placement itself is not this file's.** It is ft_tt_dram_shard.h, because the layout
 * depends on N and an artifact cannot be shipped once per core count (plan Phase 11) — so
 * deciding it has to be something a LOADER does on the machine that is about to run the
 * model, against the group size that machine chose. What is left here is what a loader really
 * does own: taking the plan's `source_offset` out of the CANONICAL bytes and writing them
 * where the plan says. Nothing below knows which bank anything is in.
 */
static int stage_bank_shards(coop_fixture *f, const shape_run *shape,
                             ft_tt_op_matmul_cooperative_mode mode, uint32_t members,
                             ft_tt_op_matmul_cooperative_request *request)
{
    /* INDEPENDENT is not a split: every member computes the WHOLE width, so its "panel" is a
     * whole copy of the weight, REPLICATED, and its source offset is 0. Getting this wrong
     * would give every member a different slice of the same answer and still complete. */
    const uint32_t columns = (mode == FT_TT_OP_MATMUL_INDEPENDENT)
                                 ? shape->tile_cols
                                 : shape->tile_cols / members;
    ft_tt_dram_shard_operand operands[SHARD_OPERAND_COUNT];
    ft_tt_dram_shard_plan plan;
    uint32_t member;
    int rc;

    memset(operands, 0, sizeof(operands));
    operands[SHARD_OPERAND_WEIGHT].spread = (mode == FT_TT_OP_MATMUL_INDEPENDENT)
                                                ? FT_TT_DRAM_SHARD_REPLICATED
                                                : FT_TT_DRAM_SHARD_PARTITIONED;
    operands[SHARD_OPERAND_WEIGHT].bank_base_address = SHARD_WEIGHT_BASE;
    operands[SHARD_OPERAND_WEIGHT].member_bytes =
        columns * shape->tile_inner * TILE_ELEMENTS * (uint32_t)sizeof(shape->b_n_major[0]);
    operands[SHARD_OPERAND_ACTIVATION].spread = FT_TT_DRAM_SHARD_REPLICATED;
    operands[SHARD_OPERAND_ACTIVATION].bank_base_address = SHARD_ACTIVATION_BASE;
    operands[SHARD_OPERAND_ACTIVATION].member_bytes =
        shape->tile_rows * shape->tile_inner * FT_TT_OP_TILE_BYTES_BF16;

    rc = ft_tt_dram_shard_plan_build(&f->session->grid, operands, SHARD_OPERAND_COUNT, members,
                                     FT_TT_COORD_TRANSLATED, &plan);
    if (rc != FT_TT_OK)
        return rc;
    f->shard_busiest_bank = plan.busiest_bank_streams;
    f->shard_quietest_bank = plan.quietest_bank_streams;

    for (member = 0u; member < members; member++) {
        const ft_tt_dram_shard_placement *weight =
            ft_tt_dram_shard_placement_of(&plan, SHARD_OPERAND_WEIGHT, member);
        const ft_tt_dram_shard_placement *activation =
            ft_tt_dram_shard_placement_of(&plan, SHARD_OPERAND_ACTIVATION, member);

        if (weight == NULL || activation == NULL)
            return FT_TT_EINVAL;

        request->weight_shards[member].endpoint = weight->endpoint;
        request->weight_shards[member].address = weight->address;
        request->activation_shards[member].endpoint = activation->endpoint;
        request->activation_shards[member].address = activation->address;

        /* The canonical artifact is `b_n_major` / `a_tiles` and is never re-laid-out; the
         * plan says which BYTES of it belong here, so the cast to a byte pointer is the whole
         * of "the loader's copy". */
        rc = ft_tt_write_dram(f->dram_path, weight->endpoint.coord, weight->address,
                              (const unsigned char *)shape->b_n_major + weight->source_offset,
                              weight->bytes);
        if (rc == FT_TT_OK)
            rc = ft_tt_write_dram(
                f->dram_path, activation->endpoint.coord, activation->address,
                (const unsigned char *)shape->a_tiles + activation->source_offset,
                activation->bytes);
        if (rc != FT_TT_OK)
            return rc;
    }
    request->weight_shard_count = members;
    request->activation_shard_count = members;
    return FT_TT_OK;
}

/**
 * The shape, the mode, and WHERE each member reads its operands from.
 *
 * STRIPE_NONE is the pre-striping request exactly: every member is given the operand's own
 * coordinate, which `ft_tt_coord_dram_bank()` made port 0. STRIPE_PORTS hands the planner all
 * three doors of the SAME two banks. STRIPE_BANKS relocates the bytes. Nothing else differs
 * between the arms - same shape, same kernels, same cores - which is what makes the byte
 * comparison a comparison of ADDRESSINGS rather than of programs.
 */
static int fill_request(ft_tt_op_matmul_cooperative_request *request, coop_fixture *f,
                        const shape_run *shape, ft_tt_op_matmul_cooperative_mode mode,
                        stripe_mode stripe, uint32_t members)
{
    memset(request, 0, sizeof(*request));
    request->tile_rows = shape->tile_rows;
    request->tile_inner = shape->tile_inner;
    request->tile_cols = shape->tile_cols;
    request->prefetch_tiles = PREFETCH_TILES;
    request->weight_n_major = 1u;
    request->mode = mode;
    request->spin_limit = SPIN_LIMIT;
    request->activation.x = (uint16_t)f->dram_a.x;
    request->activation.y = (uint16_t)f->dram_a.y;
    request->activation.address = DRAM_BUFFER_BASE;
    request->activation.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    request->weight = request->activation;
    request->weight.x = (uint16_t)f->dram_b.x;
    request->weight.y = (uint16_t)f->dram_b.y;
    request->weight.address = f->weight_base;
    request->output = request->activation;
    request->output.x = (uint16_t)f->dram_output.x;
    request->output.y = (uint16_t)f->dram_output.y;
    if (stripe == STRIPE_PORTS) {
        request->activation_ports = f->dram_a_ports;
        request->weight_ports = f->dram_b_ports;
    }
    if (stripe == STRIPE_BANKS)
        return stage_bank_shards(f, shape, mode, members, request);
    return FT_TT_OK;
}

static void release_shape(shape_run *shape)
{
    free(shape->solo_tiles);
    free(shape->readback);
    free(shape->reference);
    free(shape->a_tiles);
    free(shape->b_n_major);
    shape->solo_tiles = NULL;
    shape->readback = NULL;
    shape->reference = NULL;
    shape->a_tiles = NULL;
    shape->b_n_major = NULL;
}

/**
 * Allocate one shape's buffers, upload its operands, and compute the CPU reference.
 *
 * The weight is stored N-MAJOR in DRAM for every arm. Both halves or neither: the flag alone
 * reads the right bytes from the wrong places. N-major is also what makes a member's column
 * slice a CONTIGUOUS run of tiles, which is the whole reason a member needs only a column
 * OFFSET rather than a strided index.
 */
static int prepare_shape(coop_fixture *fixture, shape_run *shape)
{
    float *a = malloc((size_t)shape->m * shape->k * sizeof(float));
    float *b = malloc((size_t)shape->k * shape->n * sizeof(float));
    uint16_t *b_tiles = malloc((size_t)shape->tile_inner * shape->tile_cols * TILE_ELEMENTS *
                               sizeof(uint16_t));
    uint16_t *a_tiles;
    uint16_t *reordered;
    uint32_t index;
    int rc = FT_TT_ENOMEM;

    /* The tilized operands are KEPT on the shape rather than freed here: a bank-striped arm
     * re-lays them out for every group size, and re-tilizing on this 2-core 1.8 GHz host each
     * time would put host arithmetic inside the sweep. */
    shape->a_tiles = malloc((size_t)shape->tile_rows * shape->tile_inner * TILE_ELEMENTS *
                            sizeof(uint16_t));
    shape->b_n_major = malloc((size_t)shape->tile_inner * shape->tile_cols * TILE_ELEMENTS *
                              sizeof(uint16_t));
    shape->solo_tiles = malloc(shape->output_bytes);
    shape->readback = malloc(shape->output_bytes * FT_TT_OP_COOP_MAX_MEMBERS);
    shape->reference = malloc((size_t)shape->m * shape->n * sizeof(float));
    if (!a || !b || !b_tiles || !shape->a_tiles || !shape->b_n_major || !shape->solo_tiles ||
        !shape->readback || !shape->reference)
        goto cleanup;
    a_tiles = shape->a_tiles;
    reordered = shape->b_n_major;

    /* Scaled down so a Kt-long accumulation stays well inside bf16's range at the output; the
     * kernel accumulates in DST, not in fp32, and an overflow would be measuring the test's
     * arithmetic rather than the kernel's. */
    random_seed(0x5EEDu + shape->m + shape->k + shape->n);
    for (index = 0; index < shape->m * shape->k; index++)
        a[index] = random_value() * 0.5f;
    for (index = 0; index < shape->k * shape->n; index++)
        b[index] = random_value() * 0.5f;
    round_to_bf16(a, (size_t)shape->m * shape->k);
    round_to_bf16(b, (size_t)shape->k * shape->n);

    to_bf16_tiles(a, shape->m, shape->k, a_tiles);
    to_bf16_tiles(b, shape->k, shape->n, b_tiles);
    permute_tiles_to_n_major(b_tiles, reordered, shape->tile_inner, shape->tile_cols,
                             FT_TT_OP_TILE_BYTES_BF16);

    rc = ft_tt_write_dram(fixture->dram_path, fixture->dram_a, DRAM_BUFFER_BASE, a_tiles,
                          (size_t)shape->tile_rows * shape->tile_inner *
                              FT_TT_OP_TILE_BYTES_BF16);
    if (rc == FT_TT_OK)
        rc = ft_tt_write_dram(fixture->dram_path, fixture->dram_b, fixture->weight_base,
                              reordered,
                              (size_t)shape->tile_inner * shape->tile_cols *
                                  FT_TT_OP_TILE_BYTES_BF16);
    if (rc != FT_TT_OK)
        goto cleanup;

    reference_matmul(a, b, shape->m, shape->k, shape->n, shape->reference);

cleanup:
    free(a);
    free(b);
    free(b_tiles);
    return rc;
}

/**
 * Stage 1 — the SOLO baseline, scored against the CPU, and kept as the byte reference.
 *
 * A single member in any mode is the same program: one core, column 0, the whole width. It is
 * expressed as INDEPENDENT because that mode's slicing is "the whole width" by definition, so
 * the baseline needs no special case anywhere.
 */
static bool run_solo_baseline(coop_fixture *fixture, shape_run *shape, arm_result *out)
{
    ft_tt_op_matmul_cooperative_request request;
    float *actual = malloc((size_t)shape->m * shape->n * sizeof(float));
    bool ok = false;
    int rc;

    if (actual == NULL)
        return false;

    select_linear(fixture, 1u);
    (void)fill_request(&request, fixture, shape, FT_TT_OP_MATMUL_INDEPENDENT, STRIPE_NONE, 1u);
    rc = run_arm(fixture, shape, &request, 1u, UINT32_MAX, 0u, 0u, false, out);
    ft_tt_test_check_fmt("solo baseline ran", rc == FT_TT_OK && out->first_error == 0u &&
                                                  out->blocks_short == 0u,
                         "%s: rc=%d (%s), kernel error 0x%08x, %u members short of their "
                         "block count",
                         shape->label, rc, ft_tt_error_string(rc), out->first_error,
                         out->blocks_short);
    if (rc != FT_TT_OK || out->first_error != 0u) {
        free(actual);
        return false;
    }

    memcpy(shape->solo_tiles, shape->readback, shape->output_bytes);
    from_bf16_tiles(shape->solo_tiles, shape->m, shape->n, actual);
    {
        double pcc = pearson(shape->reference, actual, (size_t)shape->m * shape->n);
        size_t poison_left =
            poison_remaining((const unsigned char *)shape->solo_tiles, shape->output_bytes);

        ok = pcc >= PCC_ACCEPT && poison_left * 100u < shape->output_bytes;
        ft_tt_test_check_fmt("solo baseline agrees with the CPU reference", ok,
                             "%s [%ux%u]x[%ux%u]: PCC %.9f, %zu poison bytes left, %.3f ms, "
                             "%.2f GFLOP/s (%.1f%% of one Tensix)",
                             shape->label, shape->m, shape->k, shape->k, shape->n, pcc,
                             poison_left, out->best_ms, gflops_of(shape, out->best_ms),
                             gflops_of(shape, out->best_ms) / ONE_TENSIX_GFLOPS * 100.0);
    }
    free(actual);
    return ok;
}

/**
 * How many distinct DRAM doors an arm's readers should end up on.
 *
 * Asserted rather than merely printed, because it is the BALANCE half of the striping rules —
 * the outstanding-read budget is per endpoint — and because it is the cheapest way to catch an
 * allocator that quietly put two streams through one door.
 *
 * PORTS: two banks, each rotated over min(members, doors-per-bank) of its doors.
 * BANKS: one door per reader stream, and every member has two streams (its weight panel and
 *        its activation copy), even in COOPERATIVE where only the leader uses the A one —
 *        until the streams outnumber the CARD'S doors, at which point the chooser has to
 *        double up and the ceiling is the door count itself.
 */
static uint32_t expected_doors(const coop_fixture *f, stripe_mode stripe, uint32_t members)
{
    const uint32_t ports = f->dram_a_ports.port_count;

    if (stripe == STRIPE_PORTS)
        return 2u * (members < ports ? members : ports);
    if (stripe == STRIPE_BANKS) {
        const uint32_t doors = ft_tt_coord_dram_bank_count(&f->session->grid) * ports;

        return 2u * members < doors ? 2u * members : doors;
    }
    return 0u;
}

/** One arm of the sweep: lay the operands out for it, run it, assert it is byte-identical to
 *  the single-core result, and print its row. Every addressing goes through this, so none can
 *  be scored by a different rule than the others. */
static int run_sweep_arm(coop_fixture *fixture, shape_run *shape, const sweep_shape *rectangle,
                         ft_tt_op_matmul_cooperative_mode mode, stripe_mode stripe,
                         double solo_ms, const char *label, arm_result *out)
{
    const uint32_t members = rectangle->rows * rectangle->columns;
    ft_tt_op_matmul_cooperative_request request;
    int rc;

    select_rectangle(fixture, rectangle->rows, rectangle->columns);
    rc = fill_request(&request, fixture, shape, mode, stripe, members);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("an arm's operands were laid out and staged", false,
                             "%s %s N=%u: rc=%d (%s)", shape->label, label, members, rc,
                             ft_tt_error_string(rc));
        return rc;
    }
    rc = run_arm(fixture, shape, &request, members, UINT32_MAX, 0u, 0u, true, out);
    ft_tt_test_check_fmt("an arm is byte-identical to the single-core result",
                         rc == FT_TT_OK && out->first_error == 0u && out->blocks_short == 0u &&
                             out->bytes_match_solo,
                         "%s %s N=%u (%ux%u): rc=%d (%s), kernel error 0x%08x, %u short, "
                         "bytes %s",
                         shape->label, label, members, rectangle->rows, rectangle->columns, rc,
                         ft_tt_error_string(rc), out->first_error, out->blocks_short,
                         out->bytes_match_solo ? "IDENTICAL" : "DIFFER");
    /* Every reader gets its OWN door, which is the balance half of the striping rules. The ROW
     * half cannot always be met and is reported rather than asserted: the part has 24 DRAM
     * endpoints on only 12 rows, so any arm with more than 12 concurrent streams must double
     * up somewhere, and at these rates a shared row is far from its ~60 GB/s. */
    if (stripe != STRIPE_NONE)
        ft_tt_test_check_fmt("every reader stream of a striped arm gets its own door",
                             rc == FT_TT_OK &&
                                 out->distinct_endpoints ==
                                     expected_doors(fixture, stripe, members),
                             "%s %s N=%u: %u distinct doors, wanted %u; %u pairs of them share "
                             "a NoC row",
                             shape->label, label, members, out->distinct_endpoints,
                             expected_doors(fixture, stripe, members), out->row_collisions);
    if (rc == FT_TT_OK)
        print_row(label, members, rectangle->rows, rectangle->columns, shape, mode, out,
                  solo_ms);
    return rc;
}

/**
 * Stage 2 - the sweep. SPLIT and COOPERATIVE at every rectangle the pool allows, each in all
 * THREE addressings, every arm interleaved so a host that drifts over the sweep drifts through
 * all of them equally (DEV_NOTES.md section 8).
 *
 * COOPERATIVE stops at COOP_MAX_MEMBERS_ON_SILICON. A 2x4 group - seven receivers - wedged
 * core 115 on 2026-08-13 AND again on 2026-08-14, both times at the re-boot AFTER a clean run,
 * always the same slot. Groups of <= 4 members ran clean through both. SPLIT and INDEPENDENT
 * issue no multicast at all and are swept to the full pool.
 */
static void run_sweep(coop_fixture *fixture, shape_run *shape, double solo_ms)
{
    uint32_t entry;

    printf("\n  %-12s %2s %3s | %8s | %7s | %9s | %8s | %5s | %7s | %s\n", "arm", "N", "r x c",
           "ms", "spread", "aggregate", "read GB/s", "doors", "vs solo", "correctness");
    printf("  --------------------------+----------+---------+-----------+----------+-------+"
           "---------+---------------\n");

    for (entry = 0u; entry < SWEEP_COUNT; entry++) {
        const uint32_t members = g_sweep[entry].rows * g_sweep[entry].columns;
        arm_result split[3];
        arm_result coop[3];
        stripe_mode stripe;
        char label[24];
        int rc = FT_TT_OK;

        if (g_sweep[entry].rows > fixture->pool_rows ||
            g_sweep[entry].columns > fixture->pool_columns)
            continue;
        if ((shape->tile_cols % members) != 0u) {
            printf("  %-12s %2u %ux%u | Nt = %u does not divide by %u members - refused by the "
                   "planner, not rounded\n", "-", members, g_sweep[entry].rows,
                   g_sweep[entry].columns, shape->tile_cols, members);
            continue;
        }

        for (stripe = STRIPE_NONE; stripe <= STRIPE_BANKS && rc == FT_TT_OK; stripe++) {
            snprintf(label, sizeof(label), "SPLIT%s", g_stripe_suffix[stripe]);
            rc = run_sweep_arm(fixture, shape, &g_sweep[entry], FT_TT_OP_MATMUL_SPLIT, stripe,
                               solo_ms, label, &split[stripe]);
        }
        if (rc == FT_TT_OK && split[STRIPE_NONE].best_ms > 0.0)
            printf("               %2u %ux%u | against one door each: ports %.3fx, banks "
                   "%.3fx; device cycles %u -> %u -> %u; busiest bank %u streams, quietest "
                   "%u\n",
                   members, g_sweep[entry].rows, g_sweep[entry].columns,
                   split[STRIPE_NONE].best_ms / split[STRIPE_PORTS].best_ms,
                   split[STRIPE_NONE].best_ms / split[STRIPE_BANKS].best_ms,
                   split[STRIPE_NONE].device_cycles, split[STRIPE_PORTS].device_cycles,
                   split[STRIPE_BANKS].device_cycles, fixture->shard_busiest_bank,
                   fixture->shard_quietest_bank);

        if (rc == FT_TT_OK && members <= COOP_MAX_MEMBERS_ON_SILICON) {
            for (stripe = STRIPE_NONE; stripe <= STRIPE_BANKS && rc == FT_TT_OK; stripe++) {
                snprintf(label, sizeof(label), "COOP%s", g_stripe_suffix[stripe]);
                rc = run_sweep_arm(fixture, shape, &g_sweep[entry],
                                   FT_TT_OP_MATMUL_COOPERATIVE, stripe, solo_ms, label,
                                   &coop[stripe]);
            }
            if (rc == FT_TT_OK && split[STRIPE_NONE].best_ms > 0.0 &&
                coop[STRIPE_NONE].best_ms > 0.0)
                printf("               %2u %ux%u | multicast alone: %.3fx un-striped, %.3fx "
                       "bank-striped\n", members, g_sweep[entry].rows, g_sweep[entry].columns,
                       split[STRIPE_NONE].best_ms / coop[STRIPE_NONE].best_ms,
                       split[STRIPE_BANKS].best_ms / coop[STRIPE_BANKS].best_ms);
        } else if (members > COOP_MAX_MEMBERS_ON_SILICON) {
            printf("               %2u %ux%u | COOPERATIVE not run: %u members is %u receivers, "
                   "and a 2x4 group wedged core 115 twice\n", members, g_sweep[entry].rows,
                   g_sweep[entry].columns, members, members - 1u);
        }
        fflush(stdout);
    }
}

/* ------------------------------------------------------------------------- */
/* Spread stage — the DISTRIBUTION, not three order statistics of it          */
/* ------------------------------------------------------------------------- */

/** One arm's samples, gathered round-robin with the other arms' so a host that drifts
 *  through the stage drifts through all three equally. */
typedef struct spread_samples {
    double ms[MAX_SPREAD_SAMPLES];
    double cycles[MAX_SPREAD_SAMPLES];
    uint32_t count;
    uint32_t mismatches;
    uint32_t failures;
} spread_samples;

static void spread_statistics(const double *values, uint32_t count, double *out_min,
                              double *out_median, double *out_mean, double *out_max,
                              double *out_cv)
{
    static double sorted[MAX_SPREAD_SAMPLES];
    double total = 0.0;
    double variance = 0.0;
    uint32_t index;

    *out_min = 0.0;
    *out_median = 0.0;
    *out_mean = 0.0;
    *out_max = 0.0;
    *out_cv = 0.0;
    if (count == 0u)
        return;
    memcpy(sorted, values, count * sizeof(sorted[0]));
    qsort(sorted, count, sizeof(sorted[0]), compare_double);
    for (index = 0u; index < count; index++)
        total += values[index];
    *out_mean = total / (double)count;
    for (index = 0u; index < count; index++)
        variance += (values[index] - *out_mean) * (values[index] - *out_mean);
    *out_min = sorted[0];
    *out_median = sorted[count / 2u];
    *out_max = sorted[count - 1u];
    if (*out_mean > 0.0 && count > 1u)
        *out_cv = sqrt(variance / (double)(count - 1u)) / *out_mean * 100.0;
}

/**
 * The distribution as a picture, because the SHAPE is the whole question.
 *
 * A broad unimodal spread is contention — many small delays adding up. Two clusters is two
 * behaviours, one of them slow, and that is a different mechanism with a different fix. No
 * summary statistic distinguishes them, which is why min-of-N was the right thing to publish
 * and the wrong thing to stop at.
 */
static void print_spread_histogram(const char *what, const double *values, uint32_t count,
                                   double low, double high)
{
    enum { BINS = 24u };
    uint32_t histogram[BINS];
    uint32_t tallest = 1u;
    uint32_t bin;
    uint32_t index;

    memset(histogram, 0, sizeof(histogram));
    if (count == 0u || high <= low)
        return;
    for (index = 0u; index < count; index++) {
        double position = (values[index] - low) / (high - low) * (double)BINS;
        uint32_t slot = position <= 0.0 ? 0u : (uint32_t)position;

        if (slot >= BINS)
            slot = BINS - 1u;
        histogram[slot]++;
    }
    for (bin = 0u; bin < BINS; bin++)
        if (histogram[bin] > tallest)
            tallest = histogram[bin];
    printf("      %s, %u samples over %.3f .. %.3f\n", what, count, low, high);
    for (bin = 0u; bin < BINS; bin++) {
        uint32_t width = histogram[bin] * 40u / tallest;
        char bar[41];

        memset(bar, '#', width);
        bar[width] = '\0';
        printf("      %8.3f |%-40s %u\n", low + (high - low) * (double)bin / (double)BINS, bar,
               histogram[bin]);
    }
}

/**
 * Stage S — WHY do striped arms spread 10-50%% run to run where flat arms spread 1-3%%?
 *
 * Min-of-N was used and the spread reported, so no published number is wrong — but a 10-50%
 * spread on the same cores, the same host and the same session is the instrument saying
 * something. What distinguishes a striped arm is that its traffic crosses several banks and
 * several NoC rows, and the per-member map changes with N.
 *
 * TWO CLOCKS ON ONE RUN is the discriminator, and it is stronger than any histogram of a
 * single clock. The host wall clock is GO -> all DONE over an x1 link driven by two 1.8 GHz
 * cores that are also serving eight K80s; the device cycles are the slowest member's own
 * reader, entry to exit, which no host scheduling can move. Phase 6.7 already measured that
 * the bank-striped arm's wall clock is FLAT while its device cycles keep falling — i.e. that
 * arm is dispatch-bound — so if the spread is the host, the wall clock moves and the device
 * cycles do not. If both move, it is on the card.
 *
 * The three addressings are run ROUND-ROBIN, one sample each, not one arm to completion:
 * this host drifts enough over a stage to swamp the quantity.
 */
static void run_spread_stage(coop_fixture *fixture, shape_run *shape, uint32_t members)
{
    static spread_samples samples[3];
    const uint32_t rounds = fixture->spread_rounds;
    const uint32_t inner = fixture->spread_inner;
    const uint32_t saved_repeats = fixture->repeats;
    stripe_mode stripe;
    uint32_t round;

    ft_tt_test_step("stage S — the run-to-run DISTRIBUTION of all three addressings at N=%u, "
                    "%u rounds x %u BACK-TO-BACK repeats, ROUND-ROBIN over the arms, host "
                    "clock and device cycles on every sample",
                    members, rounds, inner);
    if (members == 0u || members > fixture->member_count || rounds == 0u || inner == 0u ||
        rounds * inner > MAX_SPREAD_SAMPLES || (shape->tile_cols % members) != 0u) {
        ft_tt_test_check_fmt("the spread stage's N and round count are expressible", false,
                             "N=%u of pool %u, %u x %u samples (max %u), Nt=%u", members,
                             fixture->member_count, rounds, inner, (unsigned)MAX_SPREAD_SAMPLES,
                             shape->tile_cols);
        return;
    }

    memset(samples, 0, sizeof(samples));
    /* `inner` repeats BACK TO BACK inside one staging is how the sweep has always measured
     * an arm, and it is what the 10-50%% spread was reported from. Re-staging between every
     * sample puts ~50 ms of host work in between and is a DIFFERENT experiment — it was tried
     * first and gave 2-4%% on every arm, which is itself a result and is why both shapes of
     * the measurement are named here. `rounds` interleaves the arms on top. */
    fixture->repeats = inner;
    fixture->sample_device_cycles = true;
    for (round = 0u; round < rounds; round++) {
        for (stripe = STRIPE_NONE; stripe <= STRIPE_BANKS; stripe++) {
            ft_tt_op_matmul_cooperative_request request;
            arm_result result;
            uint32_t sample;
            int rc;

            select_linear(fixture, members);
            rc = fill_request(&request, fixture, shape, FT_TT_OP_MATMUL_SPLIT, stripe, members);
            if (rc == FT_TT_OK)
                rc = run_arm(fixture, shape, &request, members, UINT32_MAX, 0u, 0u, true,
                             &result);
            if (rc != FT_TT_OK || result.sample_count == 0u) {
                samples[stripe].failures++;
                continue;
            }
            if (!result.bytes_match_solo || result.first_error != 0u)
                samples[stripe].mismatches++;
            for (sample = 0u; sample < result.sample_count; sample++) {
                samples[stripe].ms[samples[stripe].count] = result.sample_ms[sample];
                samples[stripe].cycles[samples[stripe].count] =
                    (double)result.sample_cycles[sample];
                samples[stripe].count++;
            }
        }
        printf("    round %2u of %u, host load %.2f\n", round, rounds, read_host_load());
        fflush(stdout);
    }
    fixture->repeats = saved_repeats;
    fixture->sample_device_cycles = false;

    printf("\n  %-14s %5s | %8s %8s %8s %8s %7s | %9s %9s %7s | %s\n", "arm", "n", "min ms",
           "median", "mean", "max", "CV%", "min cycles", "median", "CV%", "bytes");
    printf("  ------------------------------------------------------------------------------"
           "----------------------------------\n");
    for (stripe = STRIPE_NONE; stripe <= STRIPE_BANKS; stripe++) {
        double low;
        double median;
        double mean;
        double high;
        double cv;
        double cycle_low;
        double cycle_median;
        double cycle_mean;
        double cycle_high;
        double cycle_cv;
        char label[24];

        spread_statistics(samples[stripe].ms, samples[stripe].count, &low, &median, &mean, &high,
                          &cv);
        spread_statistics(samples[stripe].cycles, samples[stripe].count, &cycle_low,
                          &cycle_median, &cycle_mean, &cycle_high, &cycle_cv);
        snprintf(label, sizeof(label), "SPLIT%s", g_stripe_suffix[stripe]);
        printf("  %-14s %5u | %8.3f %8.3f %8.3f %8.3f %6.1f%% | %9.0f %9.0f %6.1f%% | %s\n",
               label, samples[stripe].count, low, median, mean, high, cv, cycle_low,
               cycle_median, cycle_cv,
               samples[stripe].mismatches == 0u ? "all == solo" : "MISMATCH");
        ft_tt_test_check_fmt("every sample of a spread arm is byte-identical to solo",
                             samples[stripe].mismatches == 0u && samples[stripe].failures == 0u,
                             "%s: %u mismatches, %u failures of %u rounds", label,
                             samples[stripe].mismatches, samples[stripe].failures, rounds);
    }
    printf("\n  CV%% is the sample standard deviation over the mean. The HOST column is GO -> all\n"
           "  DONE over the x1 link; the DEVICE column is the slowest member's reader, entry to\n"
           "  exit, on its own 800 MHz clock. A host CV far above the device CV means the\n"
           "  spread is this 2-core box, not the card.\n\n");

    /* The histogram is the answer to "bimodal or broad", and it is only worth the page when
     * there is a spread to shape. 5% is well above the 1-3% a flat arm has always shown. */
    for (stripe = STRIPE_NONE; stripe <= STRIPE_BANKS; stripe++) {
        double low;
        double median;
        double mean;
        double high;
        double cv;
        char what[64];

        spread_statistics(samples[stripe].ms, samples[stripe].count, &low, &median, &mean, &high,
                          &cv);
        if (cv < 5.0)
            continue;
        snprintf(what, sizeof(what), "N=%u SPLIT%s host ms", members, g_stripe_suffix[stripe]);
        print_spread_histogram(what, samples[stripe].ms, samples[stripe].count, low, high);
        spread_statistics(samples[stripe].cycles, samples[stripe].count, &low, &median, &mean,
                          &high, &cv);
        snprintf(what, sizeof(what), "N=%u SPLIT%s device cycles", members,
                 g_stripe_suffix[stripe]);
        print_spread_histogram(what, samples[stripe].cycles, samples[stripe].count, low, high);
    }
    fflush(stdout);
}

/** Every group size the spread stage asks about. The striped arms' map — which bank, which
 *  door, how many streams share one — is a function of N, so a spread that is a property of
 *  the MAP shows up at some N and not others, and one N cannot say that. */
static void run_spread_stage_all_n(coop_fixture *fixture, shape_run *shape)
{
    static const uint32_t group_sizes[] = {2u, 4u, 6u, 8u};
    uint32_t index;

    for (index = 0u; index < sizeof(group_sizes) / sizeof(group_sizes[0]); index++)
        if (group_sizes[index] <= fixture->member_count &&
            (shape->tile_cols % group_sizes[index]) == 0u)
            run_spread_stage(fixture, shape, group_sizes[index]);
}

/**
 * Stage 3 — N INDEPENDENT streams of the whole matmul.
 *
 * This is not a split and it is not scored against solo: it is the memory-system question.
 * Decode reads every weight per token, so what decides how many concurrent streams a card
 * supports is where AGGREGATE DRAM bandwidth flattens, not how fast one matmul goes. The
 * aggregate column is what to watch: while it keeps rising linearly the memory system is not
 * the limit, and the first N where it stops rising is the answer.
 */
static bool run_independent_arm(coop_fixture *fixture, shape_run *shape, uint32_t members,
                                stripe_mode stripe, arm_result *out)
{
    ft_tt_op_matmul_cooperative_request request;
    int rc;

    select_linear(fixture, members);
    rc = fill_request(&request, fixture, shape, FT_TT_OP_MATMUL_INDEPENDENT, stripe, members);
    if (rc == FT_TT_OK)
        rc = run_arm(fixture, shape, &request, members, UINT32_MAX, 0u, 0u, false, out);
    ft_tt_test_check_fmt("an independent-stream run completed on every member",
                         rc == FT_TT_OK && out->first_error == 0u && out->blocks_short == 0u,
                         "%s N=%u %s: rc=%d (%s), kernel error 0x%08x, %u short", shape->label,
                         members, g_stripe_name[stripe], rc, ft_tt_error_string(rc),
                         out->first_error, out->blocks_short);
    return rc == FT_TT_OK;
}

/**
 * Stage 3 - N INDEPENDENT streams of the whole matmul, in all three addressings.
 *
 * This is not a split and it is not scored against solo: it is the memory-system question, and
 * it is where striping is most legible because every core is pulling its own full copy. The
 * read GB/s column is the answer - the un-striped arm flattens at two endpoints' worth, the
 * port arm at two banks' worth, and the bank arm has a bank per core.
 *
 * The bank arm re-uploads N copies of the weight for every N, which is why this stage runs on
 * one shape only.
 */
static void run_independent_sweep(coop_fixture *fixture, shape_run *shape, double solo_ms)
{
    const uint32_t pool = fixture->pool_rows * fixture->pool_columns;
    uint32_t members;

    ft_tt_test_step("stage 3 - %s on N INDEPENDENT cores, each computing the WHOLE matmul, in "
                    "ALL THREE addressings. Aggregate is N results per run, so this is DRAM "
                    "throughput and it is where striping is most legible", shape->label);
    printf("\n  %-12s %2s | %8s | %7s | %9s | %10s | %5s | %s\n", "arm", "N", "ms", "spread",
           "aggregate", "read GB/s", "doors", "notes");
    printf("  ---------------+----------+---------+-----------+------------+-------+-------\n");

    for (members = 1u; members <= pool; members++) {
        const double read_bytes = dram_read_bytes(shape, FT_TT_OP_MATMUL_INDEPENDENT, members);
        arm_result result[3];
        stripe_mode stripe;
        bool ok = true;
        char label[24];

        /* Interleaved within one N, not one arm swept and then the other: two 1.8 GHz cores
         * also driving eight K80s drift enough over a sweep to swamp the difference being
         * measured (DEV_NOTES.md section 8). */
        for (stripe = STRIPE_NONE; stripe <= STRIPE_BANKS && ok; stripe++)
            ok = run_independent_arm(fixture, shape, members, stripe, &result[stripe]);
        if (!ok)
            continue;

        for (stripe = STRIPE_NONE; stripe <= STRIPE_BANKS; stripe++) {
            snprintf(label, sizeof(label), "INDEP%s", g_stripe_suffix[stripe]);
            printf("  %-12s %2u | %8.3f | %6.1f%% | %9.2f | %10.1f | %2u/%u | %s\n", label,
                   members, result[stripe].best_ms,
                   (result[stripe].worst_ms - result[stripe].best_ms) /
                       result[stripe].best_ms * 100.0,
                   gflops_of(shape, result[stripe].best_ms) * (double)members,
                   read_bytes / (result[stripe].best_ms * 1e-3) / 1e9,
                   result[stripe].distinct_endpoints, result[stripe].row_collisions,
                   stripe == STRIPE_NONE
                       ? (result[stripe].best_ms <= solo_ms * 1.05
                              ? "still flat - DRAM not the limit"
                              : "slower per core than solo")
                       : "");
        }
        printf("               %2u | against one door each: ports %.3fx, banks %.3fx\n",
               members, result[STRIPE_NONE].best_ms / result[STRIPE_PORTS].best_ms,
               result[STRIPE_NONE].best_ms / result[STRIPE_BANKS].best_ms);
        fflush(stdout);
    }
    printf("\n  read GB/s counts the A and B bytes the readers pull from DRAM, aggregated over\n"
           "  the group. `doors` is distinct DRAM endpoints / pairs of them sharing a NoC row.\n"
           "  The un-striped arm reads through TWO endpoints however many cores it has and\n"
           "  ~45 GB/s each is what it flattens at; the port arm has three doors of each of\n"
           "  two banks and a bank serves ~64; the bank arm has a bank per core and the CARD\n"
           "  does 502. Row collisions above 12 streams are structural - 24 endpoints, 12\n"
           "  rows - and a shared row is ~60 GB/s, far above what these streams ask of it.\n");
}

/**
 * Stage 4 — the RED HALF. Two mutations that must be CAUGHT, on silicon.
 *
 * A multicore test that only detects all-cores-broken is the failure mode to design against:
 * every red proof for a group has to make exactly ONE member wrong and show the others stay
 * green.
 *
 *  (a) ONE member is given the wrong first column. Its panel is then a different, valid piece
 *      of the same matmul — no error anywhere, and the byte comparison is the only thing that
 *      can see it. If this comes back byte-identical, the comparison is decorative.
 *  (b) ONE receiver is given a wrong CB_IN0 base. The kernel's per-block invariant must REFUSE
 *      with FTCM_ERROR_WRITE_POINTER rather than let the leader multicast into an address the
 *      receivers did not reserve — and the whole group must still finish, drain, and leave
 *      every core bootable, which is what the drain rule is for.
 */
static void run_red_proofs(coop_fixture *fixture, shape_run *shape)
{
    ft_tt_op_matmul_cooperative_request request;
    arm_result result;
    const uint32_t members = 2u;
    int rc;

    ft_tt_test_step("stage 4 — the red half: one member wrong, on silicon");

    if ((shape->tile_cols % members) != 0u || fixture->pool_columns < 2u) {
        ft_tt_test_check("the red proofs have a two-member group to run on", false,
                         "pool too small or Nt does not divide by 2");
        return;
    }

    /* (a) Member 1 is told its slice starts at column 0, which is member 0's slice. Nothing
     * fails: it computes a valid piece of the same matmul into its own panel. Only the byte
     * comparison can see it — and the FIRST DIFFERENCE must land at or after member 0's panel
     * ends, which is what says one member was wrong and the other was not. */
    select_rectangle(fixture, 1u, 2u);
    (void)fill_request(&request, fixture, shape, FT_TT_OP_MATMUL_SPLIT, STRIPE_NONE, members);
    rc = run_arm(fixture, shape, &request, members, 1u, FTCM_ARG_B_COLUMN_FIRST, 0u, true,
                 &result);
    {
        const size_t panel_bytes = shape->output_bytes / members;

        ft_tt_test_check_fmt("ONE member given the wrong first column differs in BYTES, and "
                             "only in ITS panel",
                             rc == FT_TT_OK && !result.bytes_match_solo &&
                                 result.first_difference >= panel_bytes,
                             "rc=%d (%s), kernel error 0x%08x, bytes %s, first difference at "
                             "%zu of %zu (member 0's panel ends at %zu) — a green comparison "
                             "here would mean the comparison cannot fail",
                             rc, ft_tt_error_string(rc), result.first_error,
                             result.bytes_match_solo ? "IDENTICAL (BAD)" : "differ (correct)",
                             result.first_difference, shape->output_bytes, panel_bytes);
    }

    /* (b) Member 1 — a RECEIVER, since the leader is member 0 — is given a CB_IN0 base that is
     * legally aligned and simply wrong. Its per-block invariant must REFUSE rather than let
     * the leader multicast into an address the receivers did not reserve. Two things then have
     * to hold and each is a separate check: the refusing member must report exactly
     * FTCM_ERROR_WRITE_POINTER, and the LEADER must give up on its bounded wait rather than
     * spin forever — a group where one member refuses has to end, not hang. */
    select_rectangle(fixture, 1u, 2u);
    (void)fill_request(&request, fixture, shape, FT_TT_OP_MATMUL_COOPERATIVE, STRIPE_NONE, members);
    rc = run_arm(fixture, shape, &request, members, 1u, FTCM_ARG_CB_IN0_BASE,
                 FT_TT_OP_CB_ARENA_BASE + 0x40u, false, &result);
    ft_tt_test_check_fmt("the receiver whose CB_IN0 base is wrong REFUSES with the write-"
                         "pointer code",
                         rc == FT_TT_OK && result.member_error[1] == FTCM_ERROR_WRITE_POINTER,
                         "rc=%d (%s), member 1 reported %d (wanted %d), member 0 reported %d",
                         rc, ft_tt_error_string(rc), (int)result.member_error[1],
                         FTCM_ERROR_WRITE_POINTER, (int)result.member_error[0]);
    ft_tt_test_check_fmt("the LEADER gives up on its bounded wait instead of hanging",
                         rc == FT_TT_OK && result.member_error[0] == FTCM_ERROR_WAIT_TIMEOUT,
                         "member 0 reported %d (wanted %d, the bounded-wait expiry). A hang "
                         "here is a core only assert_reset could end, which is how cores were "
                         "wedged (DEV_NOTES.md §1)",
                         (int)result.member_error[0], FTCM_ERROR_WAIT_TIMEOUT);

    /* (c) THE STRIPING RED PROOF. A striped run is byte-identical to the un-striped one, and
     * on its own that is exactly what a run whose door args were IGNORED would also look like.
     * So one member is given a door that is a real, live, telemetry-derived DRAM endpoint —
     * the ACTIVATION bank's port 0 — for its WEIGHT reads. Its panel then comes out of the
     * wrong memory while everything else is untouched, and the first difference has to land at
     * exactly its panel boundary. Green here would mean the whole striping arm is decorative.
     *
     * Only the y word is overridden, and only because the two banks were CHECKED at startup to
     * share a translated x; a half-overridden coordinate that did not would be a made-up
     * endpoint, which is the class of mistake that hangs the NoC rather than failing. */
    if (fixture->dram_a.x != fixture->dram_b.x) {
        ft_tt_test_check("the striping red proof has a same-column bank pair to use", false,
                         "A and B banks are in different translated columns, so overriding y "
                         "alone would name a coordinate neither of them is");
        return;
    }
    select_rectangle(fixture, 1u, 2u);
    (void)fill_request(&request, fixture, shape, FT_TT_OP_MATMUL_SPLIT, STRIPE_PORTS, members);
    rc = run_arm(fixture, shape, &request, members, 1u, FTCM_ARG_B_Y,
                 (uint32_t)fixture->dram_a.y, true, &result);
    {
        const size_t panel_bytes = shape->output_bytes / members;

        ft_tt_test_check_fmt("ONE striped member given a door on the WRONG BANK differs in "
                             "BYTES, and only in ITS panel",
                             rc == FT_TT_OK && !result.bytes_match_solo &&
                                 result.first_difference >= panel_bytes,
                             "rc=%d (%s), kernel error 0x%08x, bytes %s, first difference at "
                             "%zu of %zu (member 0's panel ends at %zu) — a green comparison "
                             "here would mean the reader never used the door it was given, "
                             "and every striped number above would be measuring nothing",
                             rc, ft_tt_error_string(rc), result.first_error,
                             result.bytes_match_solo ? "IDENTICAL (BAD)" : "differ (correct)",
                             result.first_difference, shape->output_bytes, panel_bytes);
    }

    /* (d) THE BANK-STRIPING RED PROOF. Same shape of argument one layer out: a bank-striped
     * run is byte-identical too, and a build that ignored the SHARD LIST would look the same.
     * So member 1's weight shard is pointed at member 0's — a real endpoint holding a real
     * panel of the same matmul, just not member 1's. Its panel then repeats member 0's answer.
     *
     * Planted in the REQUEST rather than as a reader-arg override because a shard is a host
     * layout decision, and overriding one arg word would leave the other three describing the
     * shard the layout actually made. */
    select_rectangle(fixture, 1u, 2u);
    rc = fill_request(&request, fixture, shape, FT_TT_OP_MATMUL_SPLIT, STRIPE_BANKS, members);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the bank-striping red proof staged its shards", false, "rc=%d (%s)",
                             rc, ft_tt_error_string(rc));
        return;
    }
    request.weight_shards[1] = request.weight_shards[0];
    rc = run_arm(fixture, shape, &request, members, UINT32_MAX, 0u, 0u, true, &result);
    {
        const size_t panel_bytes = shape->output_bytes / members;

        ft_tt_test_check_fmt("ONE bank-striped member pointed at ANOTHER member's shard differs "
                             "in BYTES, and only in ITS panel",
                             rc == FT_TT_OK && !result.bytes_match_solo &&
                                 result.first_difference >= panel_bytes,
                             "rc=%d (%s), kernel error 0x%08x, bytes %s, first difference at "
                             "%zu of %zu (member 0's panel ends at %zu) — a green comparison "
                             "here would mean the shard list was never read, and the whole "
                             "bank-striped column would be measuring the port rotation",
                             rc, ft_tt_error_string(rc), result.first_error,
                             result.bytes_match_solo ? "IDENTICAL (BAD)" : "differ (correct)",
                             result.first_difference, shape->output_bytes, panel_bytes);
    }
}

/**
 * Stage 5 — what state every core is in BEFORE anything re-boots it. READS ONLY, so it is safe
 * on a stuck core, and it runs before stage 6 touches anything. It exists because "the suite
 * left a core unbootable" has two very different causes and a boot failure alone cannot tell
 * the WORK from the RE-BOOT apart (DEV_NOTES.md §1, core 65).
 */
static void run_state_before_reboot(coop_fixture *fixture)
{
    uint32_t slot;

    ft_tt_test_step("stage 5 — every core's run state BEFORE any re-boot (reads only)");

    for (slot = 0u; slot < fixture->member_count; slot++) {
        ft_tt_launch_state state;
        int rc = ft_tt_launch_read_state(&fixture->session->noc,
                                         (uint16_t)fixture->members[slot].x,
                                         (uint16_t)fixture->members[slot].y, &state);

        ft_tt_test_check_fmt("a core answers with a plausible run state", rc == FT_TT_OK,
                             "slot %u at translated (%u,%u): rc=%d (%s), soft reset 0x%08x, "
                             "subordinate sync %02x %02x %02x %02x",
                             slot, fixture->members[slot].x, fixture->members[slot].y, rc,
                             ft_tt_error_string(rc), state.reset_register,
                             state.subordinate_sync[0], state.subordinate_sync[1],
                             state.subordinate_sync[2], state.subordinate_sync[3]);
    }
}

/**
 * Stage 6 — every core this suite touched still BOOTS. UNCONDITIONAL, even when an earlier
 * stage failed, because a read cannot tell a wedged core from a healthy idle one and a
 * multicast bug is exactly the kind that wedges a receiver quietly. The
 * COMPUTE half of the rule is `ft_tt_core probe`, a DIFFERENT triad, run by
 * run_cooperative_matmul_suite.sh immediately after this program.
 */
static void run_cores_still_boot(coop_fixture *fixture)
{
    uint32_t slot;

    ft_tt_test_step("stage 6 — every core touched still BOOTS (run unconditionally)");

    for (slot = 0u; slot < fixture->member_count; slot++) {
        int rc = boot_core(fixture, slot);

        ft_tt_test_check_fmt("a core this suite touched still boots", rc == FT_TT_OK,
                             "slot %u at translated (%u,%u): rc=%d (%s)", slot,
                             fixture->members[slot].x, fixture->members[slot].y, rc,
                             ft_tt_error_string(rc));
    }
    printf("  The COMPUTE half of this check is `ft_tt_core probe`, which drives its own\n"
           "  vector_add triad and must be run next — see run_cooperative_matmul_suite.sh.\n\n");
}

/* ------------------------------------------------------------------------- */

/**
 * The doors the striped arms will use, printed, and the two structural facts they must have.
 *
 * Both are properties of the LIVE layout rather than of this suite, and both decide whether
 * any GB/s below means anything: three doors of one bank must sit on three DIFFERENT NoC rows
 * (a rotation onto one row would just re-share a ~60 GB/s ceiling), and the two banks' doors
 * must not collide with each other's. Printed as well as asserted, because the row numbers are
 * what a later reader needs to check a different bank pair against.
 */
static void report_the_doors(const coop_fixture *fixture)
{
    ft_tt_dram_endpoint all[FT_TT_DRAM_STRIPE_MAX_PORTS * 2u];
    uint32_t count = 0u;
    uint32_t port;
    uint32_t collisions;

    ft_tt_test_step("the DRAM doors — three NoC ports per bank, from live telemetry");
    for (port = 0u; port < fixture->dram_a_ports.port_count; port++) {
        const ft_tt_dram_endpoint *a = &fixture->dram_a_ports.port[port];
        const ft_tt_dram_endpoint *b = &fixture->dram_b_ports.port[port];

        printf("    port %u   A bank %u -> translated (%u,%u), NoC 0 row %u   |   "
               "B bank %u -> translated (%u,%u), NoC 0 row %u\n",
               port, a->logical_bank, a->coord.x, a->coord.y, a->noc_row, b->logical_bank,
               b->coord.x, b->coord.y, b->noc_row);
        all[count++] = *a;
        all[count++] = *b;
    }
    collisions = ft_tt_dram_stripe_shared_rows(all, count, NULL, NULL);
    ft_tt_test_check_fmt("the six doors the striped arms use sit on six DIFFERENT NoC rows",
                         collisions == 0u,
                         "%u colliding pairs of %u doors — two DRAM endpoints on one row share "
                         "~60 GB/s, so a rotation onto a shared row buys nothing and the "
                         "measurement would be of the row, not of the ports",
                         collisions, count);
    ft_tt_test_check_fmt("port 0 of each bank IS what the un-striped arms address",
                         fixture->dram_a_ports.port[0].coord.x == fixture->dram_a.x &&
                             fixture->dram_a_ports.port[0].coord.y == fixture->dram_a.y &&
                             fixture->dram_b_ports.port[0].coord.x == fixture->dram_b.x &&
                             fixture->dram_b_ports.port[0].coord.y == fixture->dram_b.y,
                         "A port 0 (%u,%u) vs dram_a (%u,%u); B port 0 (%u,%u) vs dram_b "
                         "(%u,%u). If these ever differ the two arms are not the same bytes",
                         fixture->dram_a_ports.port[0].coord.x,
                         fixture->dram_a_ports.port[0].coord.y, fixture->dram_a.x,
                         fixture->dram_a.y, fixture->dram_b_ports.port[0].coord.x,
                         fixture->dram_b_ports.port[0].coord.y, fixture->dram_b.x,
                         fixture->dram_b.y);
}

/** The MiniCPM5-1B shapes, at 32 tokens (Mt = 1), as K x N in the decided orientation.
 *  `ffn_gate/up` is the best case for sharing (Nt = 144, so the split has room), `ffn_down`
 *  the middle (Nt = 48) and `attn_k/v` the worst (Nt = 8, where a group of 8 leaves one output
 *  tile per core). */
typedef struct model_shape {
    const char *label;
    uint32_t m;
    uint32_t k;
    uint32_t n;
} model_shape;

static const model_shape g_shapes[] = {
    {"ffn_gate/up", 32u, 1536u, 4608u},
    {"ffn_down", 32u, 4608u, 1536u},
    {"attn_k/v", 32u, 1536u, 256u},
};
#define SHAPE_COUNT ((uint32_t)(sizeof(g_shapes) / sizeof(g_shapes[0])))

static void run_one_shape(coop_fixture *fixture, const model_shape *model, bool independent)
{
    shape_run shape;
    arm_result solo;
    int rc;

    memset(&shape, 0, sizeof(shape));
    shape.label = model->label;
    shape.m = model->m;
    shape.k = model->k;
    shape.n = model->n;
    shape.tile_rows = model->m / TILE_DIM;
    shape.tile_inner = model->k / TILE_DIM;
    shape.tile_cols = model->n / TILE_DIM;
    shape.output_bytes = (size_t)shape.tile_rows * shape.tile_cols * FT_TT_OP_TILE_BYTES_BF16;

    ft_tt_test_step("shape %s — [%ux%u] x [%ux%u], Mt %u Kt %u Nt %u, prefetch %u",
                    shape.label, shape.m, shape.k, shape.k, shape.n, shape.tile_rows,
                    shape.tile_inner, shape.tile_cols, PREFETCH_TILES);

    rc = prepare_shape(fixture, &shape);
    ft_tt_test_check_fmt("operands staged in DRAM and the CPU reference computed",
                         rc == FT_TT_OK, "%s: rc=%d (%s)", shape.label, rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK) {
        release_shape(&shape);
        return;
    }

    if (fixture->spread_rounds != 0u) {
        /* One question, one stage: the sweep and the red proofs would put minutes of other
         * work between the samples this stage is trying to compare. */
        if (run_solo_baseline(fixture, &shape, &solo))
            run_spread_stage_all_n(fixture, &shape);
        release_shape(&shape);
        return;
    }
    if (run_solo_baseline(fixture, &shape, &solo)) {
        run_sweep(fixture, &shape, solo.best_ms);
        if (independent)
            run_independent_sweep(fixture, &shape, solo.best_ms);
        run_red_proofs(fixture, &shape);
    }
    release_shape(&shape);
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_program_directory program;
    ft_tt_noc_path dram_path;
    coop_fixture fixture;
    const char *directory;
    uint32_t slot;
    uint32_t shape_index;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <program-directory> [repeats]\n", argv[0]);
        return 2;
    }
    directory = argv[1];

    memset(&fixture, 0, sizeof(fixture));
    fixture.repeats = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 0) : DEFAULT_REPEATS;
    fixture.first_index = environment_u32("FT_TT_COOP_FIRST_INDEX", DEFAULT_FIRST_INDEX);
    fixture.pool_rows = environment_u32("FT_TT_COOP_ROWS", DEFAULT_POOL_ROWS);
    fixture.pool_columns = environment_u32("FT_TT_COOP_COLUMNS", DEFAULT_POOL_COLUMNS);
    /* Where the WEIGHT sits inside its bank. Overridable so the weight can be put in the SAME
     * DRAM bank as the activation at a different address, which is the only way this harness
     * can ask whether the scaling wall is per-BANK bandwidth or a global one: same cores, same
     * bytes, one bank instead of two. */
    fixture.weight_base = environment_u32("FT_TT_COOP_DRAM_B_BASE", DRAM_BUFFER_BASE);
    /* Non-zero replaces the whole suite with stage S — the run-to-run distribution of the
     * three addressings at one N. Zero (the default) leaves the suite exactly as it was. */
    fixture.spread_rounds = environment_u32("FT_TT_COOP_SPREAD_ROUNDS", 0u);
    fixture.spread_inner = environment_u32("FT_TT_COOP_SPREAD_INNER", 8u);
    if (fixture.repeats == 0u || fixture.pool_rows == 0u || fixture.pool_columns == 0u ||
        fixture.pool_rows * fixture.pool_columns > FT_TT_OP_COOP_MAX_MEMBERS) {
        fprintf(stderr, "repeats must be >= 1 and the pool must be 1..%u cores\n",
                FT_TT_OP_COOP_MAX_MEMBERS);
        return 2;
    }
    if (fixture.pool_columns > 4u)
        fprintf(stderr, "WARNING: a rectangle wider than 4 columns was MEASURED on 2026-08-13 "
                        "to leave a minority of its receivers unbootable. Proceeding because "
                        "you asked for %u.\n", fixture.pool_columns);
    fixture.member_count = fixture.pool_rows * fixture.pool_columns;

    /* The wire contract's refusal codes are reproduced by value in the protocol header so it
     * can stay include-free; if the two ever drift, every red proof silently passes. */
    ft_tt_test_step("cooperative matmul — %ux%u cores from live index %u, %u repeats",
                    fixture.pool_rows, fixture.pool_columns, fixture.first_index,
                    fixture.repeats);
    ft_tt_test_check("the protocol header's refusal codes match the multicast kernel's",
                     FTCM_ERROR_ZERO_LENGTH == -1 && FTCM_ERROR_ZERO_DESTINATIONS == -2 &&
                         FTCM_ERROR_ALIGNMENT == -3 && FTCM_ERROR_RECTANGLE == -4 &&
                         FTCM_ERROR_WAIT_TIMEOUT == -5 && FTCM_ERROR_STOPPED == -6,
                     "cooperative_matmul_protocol.h vs kernels/ft_tensix_noc_multicast.h");

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

    rc = ft_tt_coord_dram_bank(&session.grid, environment_u32("FT_TT_COOP_DRAM_A",
                                                              DEFAULT_DRAM_BANK_A),
                               FT_TT_COORD_TRANSLATED, &fixture.dram_a);
    if (rc == FT_TT_OK)
        rc = ft_tt_coord_dram_bank(&session.grid,
                                   environment_u32("FT_TT_COOP_DRAM_B", DEFAULT_DRAM_BANK_B),
                                   FT_TT_COORD_TRANSLATED, &fixture.dram_b);
    if (rc == FT_TT_OK)
        rc = ft_tt_coord_dram_bank(&session.grid,
                                   environment_u32("FT_TT_COOP_DRAM_C",
                                                   DEFAULT_DRAM_BANK_OUTPUT),
                                   FT_TT_COORD_TRANSLATED, &fixture.dram_output);
    /* Every door of the two banks the readers pull from. DERIVED from live telemetry, never
     * probed: a wrong DRAM endpoint does not fail a read, it hangs the chip.
     * fixture.dram_a / dram_b stay port 0 of the same banks, which is what the un-striped arms
     * use and what `ft_tt_coord_dram_bank()` has always returned. */
    if (rc == FT_TT_OK)
        rc = ft_tt_dram_port_set_build(&session.grid,
                                       environment_u32("FT_TT_COOP_DRAM_A", DEFAULT_DRAM_BANK_A),
                                       FT_TT_COORD_TRANSLATED, &fixture.dram_a_ports);
    if (rc == FT_TT_OK)
        rc = ft_tt_dram_port_set_build(&session.grid,
                                       environment_u32("FT_TT_COOP_DRAM_B", DEFAULT_DRAM_BANK_B),
                                       FT_TT_COORD_TRANSLATED, &fixture.dram_b_ports);
    /* The partitioner that refuses a run of consecutive live indices which wraps a row or
     * crosses the non-Tensix column gap at NoC x = 8, 9. Never walk indices by hand. */
    if (rc == FT_TT_OK)
        rc = ft_tt_group_rectangle_coordinates(&session.grid, fixture.first_index,
                                               fixture.pool_rows, fixture.pool_columns,
                                               fixture.members, FT_TT_OP_COOP_MAX_MEMBERS);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("cores and DRAM banks resolved from live telemetry", false,
                         ft_tt_error_string(rc));
        goto close_path;
    }

    rc = ft_tt_firmware_set_load(directory, &firmware);
    if (rc != FT_TT_OK)
        goto close_path;
    fixture.firmware = &firmware;
    rc = ft_tt_program_directory_load(directory, 0x1Fu, &program);
    if (rc != FT_TT_OK) {
        ft_tt_firmware_set_release(&firmware);
        goto close_path;
    }
    fixture.program = &program;

    printf("  group: %ux%u rectangle from live index %u, translated x %u..%u, y %u..%u\n",
           fixture.pool_rows, fixture.pool_columns, fixture.first_index, fixture.members[0].x,
           fixture.members[fixture.pool_columns - 1u].x, fixture.members[0].y,
           fixture.members[fixture.member_count - 1u].y);

    report_the_doors(&fixture);

    ft_tt_test_step("boot — %u cores, SEQUENTIALLY (concurrent boots at 12-way and above wedge "
                    "cores, DEV_NOTES.md §1)", fixture.member_count);
    for (slot = 0u; slot < fixture.member_count; slot++) {
        rc = boot_core(&fixture, slot);
        ft_tt_test_check_fmt("a core booted with the program loaded", rc == FT_TT_OK,
                             "slot %u at translated (%u,%u): rc=%d (%s)", slot,
                             fixture.members[slot].x, fixture.members[slot].y, rc,
                             ft_tt_error_string(rc));
        if (rc != FT_TT_OK)
            goto health;
    }

    for (shape_index = 0u; shape_index < SHAPE_COUNT; shape_index++) {
        run_one_shape(&fixture, &g_shapes[shape_index], shape_index == 0u);
        if (fixture.spread_rounds != 0u)
            break; /* the spread stage asks about ONE shape at ONE N; more is noise */
    }

health:
    run_state_before_reboot(&fixture);
    run_cores_still_boot(&fixture);

    ft_tt_program_directory_release(&program);
    ft_tt_firmware_set_release(&firmware);

close_path:
    (void)ft_tt_noc_path_close(&dram_path);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
