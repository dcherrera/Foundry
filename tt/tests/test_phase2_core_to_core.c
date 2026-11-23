/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_phase2_core_to_core.c — the first Tensix-to-OTHER-Tensix NoC traffic.
 *
 * Plan Phase 2 follow-up: "a core-to-*other*-core transaction is still untested". DRAM
 * proved the fabric and both directions, but every Tensix→Tensix transaction Foundry has
 * issued was SELF-addressed: step 12's reader and writer were pointed at the core they
 * ran on. Phase 5's spatial thesis is that the grid is a topology, not 140 independent
 * cores, and this is the smallest experiment that says so.
 *
 * Two cores, both derived from ft_tt_coord against live ARC telemetry and never guessed:
 * a COMPUTE core (FT_TT_TENSIX_INDEX, default 0) that runs the vector_add triad, and a
 * PEER core (FT_TT_TENSIX_PEER_INDEX, default 1) that holds data. The two indices must
 * differ — a "cross-core" test on one core is exactly the vacuous test this file exists
 * to replace, so equal indices are refused rather than run.
 *
 *   A  operands in the PEER's L1, result in the compute core's own L1.
 *      A cross-core READ, on NoC 0, and nothing the device writes leaves the core.
 *   B  result in the PEER's L1 too, so the writer crosses on NoC 1 as well.
 *   C  the PEER computes the operand. It runs the same triad first, entirely inside its
 *      own L1; then the compute core reads THAT result across the fabric. The bytes on
 *      the wire were produced by a Tensix, not staged by the host.
 *
 * **The trap this test is built around: the peer's L1 addresses are the SAME addresses
 * in the compute core's own L1.** A transaction that quietly stayed self-addressed would
 * read a valid buffer at a valid address and could produce the right answer. So every
 * stage POISONS the compute core's own copy of the source addresses with different data
 * first, and stage A additionally re-runs with the reader pointed at the compute core's
 * own L1 and REQUIRES that to fail. Byte-exactness here means "came from the other core",
 * not merely "came from somewhere".
 *
 * Usage:  test_phase2_core_to_core <program-directory>
 *   A build_tensix_program.sh output for the vector_add triad at the default 2048-byte
 *   tile. Firmware and kernels both come from it: a kernel is linked at its own
 *   firmware's __fw_export_text_end and the two cannot be mixed across builds.
 *
 * Build:
 *   make -f Makefile.phase2_followups PHASE1B=1 build-phase2b/test_phase2_core_to_core
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_hardware_session.h"
#include "ft_tt_phase2_l1_layout.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RUN_TIMEOUT_MS 5000u
#define TILE_COUNT     2u
#define TILE_ELEMENTS  (FT_TT_PHASE2_TILE_BYTES / 2u)
#define CB_CONFIG_BYTES (FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4u)

/** Repeats of the cross-core round trip. Enough to catch a path that works once and
 *  leaks state, cheap enough that a failure is found in seconds. */
#define REPEAT_COUNT 50u

/** Environment variable naming the core that HOLDS the data. */
#define PEER_INDEX_ENVIRONMENT "FT_TT_TENSIX_PEER_INDEX"

/** Poison written over the compute core's own copy of the source addresses. Every 16-bit
 *  element reads 0x5A5A, which is a large negative bfloat16 and cannot be mistaken for
 *  one of the small positive integers the reference builds. */
#define LOCAL_POISON_BYTE 0x5Au

/** Poison for a destination, so "unchanged" and "correct" are never the same bytes. */
#define DESTINATION_POISON_BYTE 0xA5u

/** The L1 map is derived from where the firmware images landed, not chosen. Both cores
 *  boot the same firmware set, so one map serves both. */
static ft_tt_phase2_l1_layout g_layout;

static uint16_t tile_in0[TILE_ELEMENTS];
static uint16_t tile_in1[TILE_ELEMENTS];
static uint16_t tile_expected[TILE_ELEMENTS];
static uint16_t tile_scratch[TILE_ELEMENTS];

/** One Tensix in this test: where it is and which index it came from. */
typedef struct tensix_endpoint {
    uint32_t index;
    uint16_t x;
    uint16_t y;
} tensix_endpoint;

/* ------------------------------------------------------------------------- */
/* Reference data                                                            */
/* ------------------------------------------------------------------------- */

/** bfloat16 is the top 16 bits of the binary32 pattern. Every value here is a small
 *  integer, so the truncation and the sum are both exact. */
static uint16_t bfloat16_from_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

#define PASS_ZEROED 0xFFFFu

static void build_reference_tiles(unsigned tile_index, unsigned pass)
{
    unsigned element;

    for (element = 0; element < TILE_ELEMENTS; element++) {
        float left;
        float right;

        if (pass == PASS_ZEROED) {
            tile_in0[element] = 0u;
            tile_in1[element] = 0u;
            tile_expected[element] = 0u;
            continue;
        }
        left = (float)((element + tile_index + pass) % 16u + 1u);
        right = (float)((element * 3u + tile_index + pass * 5u) % 16u + 1u);
        tile_in0[element] = bfloat16_from_float(left);
        tile_in1[element] = bfloat16_from_float(right);
        tile_expected[element] = bfloat16_from_float(left + right);
    }
}

static unsigned first_difference(const uint16_t *left, const uint16_t *right)
{
    unsigned element;

    for (element = 0; element < TILE_ELEMENTS; element++) {
        if (left[element] != right[element])
            return element;
    }
    return TILE_ELEMENTS;
}

/* ------------------------------------------------------------------------- */
/* L1 staging, on whichever core is named                                    */
/* ------------------------------------------------------------------------- */

static int write_tile(const ft_tt_noc_access *noc, const tensix_endpoint *core, uint32_t address,
                      const uint16_t *tile)
{
    return noc->write(noc->user, core->x, core->y, address, tile, FT_TT_PHASE2_TILE_BYTES);
}

static int read_tile(const ft_tt_noc_access *noc, const tensix_endpoint *core, uint32_t address,
                     uint16_t *tile)
{
    return noc->read(noc->user, core->x, core->y, address, tile, FT_TT_PHASE2_TILE_BYTES);
}

/** Put both source tile sets of `pass` into `core`'s L1 at the layout's source addresses. */
static int stage_sources(const ft_tt_noc_access *noc, const tensix_endpoint *core, unsigned pass)
{
    unsigned tile;

    for (tile = 0; tile < TILE_COUNT; tile++) {
        uint32_t offset = tile * FT_TT_PHASE2_TILE_BYTES;
        int rc;

        build_reference_tiles(tile, pass);
        rc = write_tile(noc, core, g_layout.source0_address + offset, tile_in0);
        if (rc == FT_TT_OK)
            rc = write_tile(noc, core, g_layout.source1_address + offset, tile_in1);
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/**
 * Fill `core`'s source addresses with poison.
 *
 * This is the assertion, not housekeeping. The peer's buffers live at the same L1
 * addresses in the compute core's own address space, so a transaction that never left
 * the tile would find a plausible buffer there. With poison in it, a self-addressed read
 * cannot produce the reference answer.
 */
static int poison_local_sources(const ft_tt_noc_access *noc, const tensix_endpoint *core)
{
    unsigned tile;

    memset(tile_scratch, LOCAL_POISON_BYTE, sizeof(tile_scratch));
    for (tile = 0; tile < TILE_COUNT; tile++) {
        uint32_t offset = tile * FT_TT_PHASE2_TILE_BYTES;
        int rc = write_tile(noc, core, g_layout.source0_address + offset, tile_scratch);

        if (rc == FT_TT_OK)
            rc = write_tile(noc, core, g_layout.source1_address + offset, tile_scratch);
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/**
 * Read the peer's staged operands back through the host and compare them to the
 * reference, BEFORE asking a kernel to fetch them.
 *
 * This separates "the host cannot write another core's L1" from "the kernel cannot read
 * it", which is the first fork any cross-core failure has to be pushed through. The DRAM
 * test has the same check for the same reason.
 */
static int verify_staged_sources(const ft_tt_noc_access *noc, const tensix_endpoint *core,
                                 unsigned pass, unsigned *out_bad_tile, unsigned *out_bad_element)
{
    unsigned tile;

    *out_bad_tile = TILE_COUNT;
    *out_bad_element = TILE_ELEMENTS;
    for (tile = 0; tile < TILE_COUNT; tile++) {
        uint32_t offset = tile * FT_TT_PHASE2_TILE_BYTES;
        unsigned difference;
        int rc;

        build_reference_tiles(tile, pass);
        rc = read_tile(noc, core, g_layout.source0_address + offset, tile_scratch);
        if (rc != FT_TT_OK)
            return rc;
        difference = first_difference(tile_scratch, tile_in0);
        if (difference == TILE_ELEMENTS) {
            rc = read_tile(noc, core, g_layout.source1_address + offset, tile_scratch);
            if (rc != FT_TT_OK)
                return rc;
            difference = first_difference(tile_scratch, tile_in1);
        }
        if (difference != TILE_ELEMENTS) {
            *out_bad_tile = tile;
            *out_bad_element = difference;
            return FT_TT_ERROR;
        }
    }
    return FT_TT_OK;
}

/**
 * Print, for the elements around a mismatch, every value that could explain it: what the
 * reference wanted, what the destination holds, what the PEER's operands hold, and what
 * the compute core's own poisoned copy holds. One of those four columns is always the
 * answer, and re-running to get them costs a card.
 */
static void dump_mismatch(const ft_tt_noc_access *noc, const tensix_endpoint *compute,
                          const tensix_endpoint *peer, const tensix_endpoint *destination,
                          unsigned tile, unsigned element, unsigned pass)
{
    static uint16_t destination_tile[TILE_ELEMENTS];
    static uint16_t peer0[TILE_ELEMENTS];
    static uint16_t peer1[TILE_ELEMENTS];
    static uint16_t local0[TILE_ELEMENTS];
    uint32_t offset = tile * FT_TT_PHASE2_TILE_BYTES;
    unsigned first = element > 4u ? element - 4u : 0u;
    unsigned index;

    if (tile >= TILE_COUNT || element >= TILE_ELEMENTS)
        return;
    if (read_tile(noc, destination, g_layout.destination_address + offset, destination_tile) !=
            FT_TT_OK ||
        read_tile(noc, peer, g_layout.source0_address + offset, peer0) != FT_TT_OK ||
        read_tile(noc, peer, g_layout.source1_address + offset, peer1) != FT_TT_OK ||
        read_tile(noc, compute, g_layout.source0_address + offset, local0) != FT_TT_OK) {
        printf("        (could not read the tiles back for a dump)\n");
        return;
    }
    build_reference_tiles(tile, pass);

    /* How MANY elements are wrong, and where, decides what kind of failure this is: four
     * stray bytes is not the same fault as half a tile. */
    {
        unsigned differing = 0;
        unsigned shown = 0;

        for (index = 0; index < TILE_ELEMENTS; index++) {
            if (destination_tile[index] == tile_expected[index])
                continue;
            differing++;
            if (shown < 16u) {
                printf("%s%u", shown == 0u ? "        differing elements: " : ", ", index);
                shown++;
            }
        }
        printf("%s(%u of %u differ in tile %u)\n", differing ? " ... " : "        ", differing,
               TILE_ELEMENTS, tile);
    }

    printf("        tile %u around element %u — expected / destination / peer in0 / peer in1 / "
           "compute's own in0\n", tile, element);
    for (index = first; index < first + 12u && index < TILE_ELEMENTS; index++)
        printf("          [%4u] %04x  %04x  %04x  %04x  %04x%s\n", index, tile_expected[index],
               destination_tile[index], peer0[index], peer1[index], local0[index],
               index == element ? "   <-- first difference" : "");
}

static int poison_destination(const ft_tt_noc_access *noc, const tensix_endpoint *core)
{
    unsigned tile;

    memset(tile_scratch, DESTINATION_POISON_BYTE, sizeof(tile_scratch));
    for (tile = 0; tile < TILE_COUNT; tile++) {
        int rc = write_tile(noc, core, g_layout.destination_address + tile * FT_TT_PHASE2_TILE_BYTES,
                            tile_scratch);

        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

static int check_destination(const ft_tt_noc_access *noc, const tensix_endpoint *core,
                             unsigned pass, unsigned *out_bad_tile, unsigned *out_bad_element)
{
    unsigned tile;

    *out_bad_tile = TILE_COUNT;
    *out_bad_element = TILE_ELEMENTS;
    for (tile = 0; tile < TILE_COUNT; tile++) {
        unsigned difference;
        int rc = read_tile(noc, core, g_layout.destination_address + tile * FT_TT_PHASE2_TILE_BYTES,
                           tile_scratch);

        if (rc != FT_TT_OK)
            return rc;
        build_reference_tiles(tile, pass);
        difference = first_difference(tile_scratch, tile_expected);
        if (difference != TILE_ELEMENTS) {
            *out_bad_tile = tile;
            *out_bad_element = difference;
            return FT_TT_ERROR;
        }
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* Program setup, per core                                                   */
/* ------------------------------------------------------------------------- */

/** Describe the three circular buffers. They are always in the RUNNING core's own L1 —
 *  a CB is a local FIFO, and only the reader's and writer's endpoints move. */
static int write_cb_config(const ft_tt_noc_access *noc, const tensix_endpoint *core,
                           uint64_t *out_mask)
{
    const uint32_t addresses[3] = {g_layout.cb_in0_address, g_layout.cb_in1_address,
                                   g_layout.cb_out_address};
    static const unsigned indices[3] = {FT_TT_PHASE2_CB_IN0, FT_TT_PHASE2_CB_IN1,
                                        FT_TT_PHASE2_CB_OUT};
    static uint8_t buffer[CB_CONFIG_BYTES];
    ft_tt_launch_cb_entry entries[3];
    unsigned entry;
    int rc;

    for (entry = 0; entry < 3u; entry++) {
        entries[entry].index = indices[entry];
        entries[entry].address = addresses[entry];
        entries[entry].size_bytes = FT_TT_PHASE2_CB_BYTES;
        entries[entry].page_size_bytes = FT_TT_PHASE2_TILE_BYTES;
        entries[entry].page_count = FT_TT_PHASE2_CB_PAGE_COUNT;
    }
    rc = ft_tt_launch_encode_cb_config(entries, 3u, buffer, sizeof(buffer), out_mask);
    if (rc != FT_TT_OK)
        return rc;
    return noc->write(noc->user, core->x, core->y, g_layout.cb_config_address, buffer,
                      sizeof(buffer));
}

static void build_config(const ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                         uint64_t cb_mask, ft_tt_launch_config *out_config)
{
    unsigned processor;

    memset(out_config, 0, sizeof(*out_config));
    out_config->kernel_config_base = g_layout.kernel_config_base;
    out_config->enables = 0x1fu;
    out_config->local_cb_config_address = g_layout.cb_config_address;
    out_config->local_cb_mask = cb_mask;
    out_config->brisc_noc_id = 0u;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        out_config->kernel_text_address[processor] = placements[processor].entry_address;
        out_config->kernel_text_bytes[processor] = placements[processor].text_bytes;
        out_config->runtime_args_address[processor] =
            ft_tt_phase2_runtime_args_address(&g_layout, processor);
    }
}

/**
 * Plant runtime args for one run of the triad on `running`.
 *
 * `source` names the core the reader pulls from and `destination` the core the writer
 * pushes to; pointing either at a core other than `running` is the whole point of this
 * file. `source0_address` is separate so the red half can shift it without touching a
 * coordinate — an address mistake is a wrong answer, a coordinate mistake is a power
 * cycle (spec §7.3).
 */
static int plant_runtime_args(const ft_tt_noc_access *noc, const tensix_endpoint *running,
                              const ft_tt_launch_config *config, const tensix_endpoint *source,
                              uint32_t source0_address, uint32_t source1_address,
                              const tensix_endpoint *destination)
{
    uint32_t reader_arguments[7];
    uint32_t writer_arguments[4];
    uint32_t tile_count = TILE_COUNT;
    unsigned processor;
    int rc;

    reader_arguments[0] = tile_count;
    reader_arguments[1] = source->x;
    reader_arguments[2] = source->y;
    reader_arguments[3] = source0_address;
    reader_arguments[4] = source->x;
    reader_arguments[5] = source->y;
    reader_arguments[6] = source1_address;

    writer_arguments[0] = tile_count;
    writer_arguments[1] = destination->x;
    writer_arguments[2] = destination->y;
    writer_arguments[3] = g_layout.destination_address;

    rc = ft_tt_launch_plant_runtime_args(noc, running->x, running->y,
                                         config->runtime_args_address[FT_TT_PROC_DM0],
                                         reader_arguments, 7u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_plant_runtime_args(noc, running->x, running->y,
                                             config->runtime_args_address[FT_TT_PROC_DM1],
                                             writer_arguments, 4u);
    for (processor = FT_TT_PROC_MATH0; processor < FT_TT_PROC_COUNT && rc == FT_TT_OK; processor++)
        rc = ft_tt_launch_plant_runtime_args(noc, running->x, running->y,
                                             config->runtime_args_address[processor], &tile_count,
                                             1u);
    return rc;
}

static int launch_and_wait(const ft_tt_noc_access *noc, const tensix_endpoint *core,
                           const ft_tt_launch_config *config)
{
    int rc = ft_tt_launch_write_launch_message(noc, core->x, core->y, config);

    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_send_go(noc, core->x, core->y);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_wait_done(noc, core->x, core->y, RUN_TIMEOUT_MS, NULL);
}

/** One full run: stage `pass` on `source`, poison the compute core's own copy and the
 *  destination, run, and compare. */
static int run_pass(const ft_tt_noc_access *noc, const tensix_endpoint *compute,
                    const tensix_endpoint *peer, const ft_tt_launch_config *config,
                    const tensix_endpoint *destination, unsigned pass, unsigned *out_bad_tile,
                    unsigned *out_bad_element)
{
    int rc = stage_sources(noc, peer, pass);

    if (rc == FT_TT_OK)
        rc = poison_local_sources(noc, compute);
    if (rc == FT_TT_OK)
        rc = poison_destination(noc, destination);
    if (rc == FT_TT_OK)
        rc = launch_and_wait(noc, compute, config);
    if (rc != FT_TT_OK)
        return rc;
    return check_destination(noc, destination, pass, out_bad_tile, out_bad_element);
}

/* ------------------------------------------------------------------------- */
/* Stage A — the reader crosses to the peer                                  */
/* ------------------------------------------------------------------------- */

static void run_stage_a_remote_read(const ft_tt_noc_access *noc, const tensix_endpoint *compute,
                                    const tensix_endpoint *peer,
                                    const ft_tt_image_placement placements[FT_TT_PROC_COUNT])
{
    ft_tt_launch_config config;
    uint64_t cb_mask = 0;
    unsigned bad_tile = 0;
    unsigned bad_element = 0;
    int rc;

    ft_tt_test_step("stage A — operands in Tensix %u at (%u, %u); the triad runs on Tensix %u "
                    "at (%u, %u) and reads across the fabric", peer->index, peer->x, peer->y,
                    compute->index, compute->x, compute->y);

    rc = write_cb_config(noc, compute, &cb_mask);
    ft_tt_test_check_fmt("CB config written on the compute core", rc == FT_TT_OK, "mask=0x%llx",
                         (unsigned long long)cb_mask);
    if (rc != FT_TT_OK)
        return;

    build_config(placements, cb_mask, &config);
    rc = plant_runtime_args(noc, compute, &config, peer, g_layout.source0_address,
                            g_layout.source1_address, compute);
    ft_tt_test_check_fmt("runtime args planted", rc == FT_TT_OK,
                         "reader src (%u,%u) 0x%05x / 0x%05x, writer dst (%u,%u) 0x%05x", peer->x,
                         peer->y, g_layout.source0_address, g_layout.source1_address, compute->x,
                         compute->y, g_layout.destination_address);
    if (rc != FT_TT_OK)
        return;

    /* Staging first, and separately: a cross-core failure is either the host's write or
     * the kernel's read, and these two checks are the fork between them. */
    rc = stage_sources(noc, peer, 0u);
    if (rc == FT_TT_OK)
        rc = verify_staged_sources(noc, peer, 0u, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("the host reads back what it wrote to the PEER's L1", rc == FT_TT_OK,
                         "so a later mismatch is the kernel's, not the staging's; first "
                         "difference tile %u element %u", bad_tile, bad_element);
    if (rc != FT_TT_OK)
        return;

    rc = run_pass(noc, compute, peer, &config, compute, 0u, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("the result is byte-exact against the C reference", rc == FT_TT_OK,
                         "%u tiles x %u elements; matched up to tile %u element %u", TILE_COUNT,
                         TILE_ELEMENTS, bad_tile, bad_element);
    if (rc != FT_TT_OK) {
        dump_mismatch(noc, compute, peer, compute, bad_tile, bad_element, 0u);
        return;
    }

    /* Read the poison back rather than asserting we wrote it: this is the claim that the
     * bytes came from the OTHER core, and it is worth a device read. */
    rc = read_tile(noc, compute, g_layout.source0_address, tile_scratch);
    ft_tt_test_check_fmt("the compute core's own source address still holds poison",
                         rc == FT_TT_OK && tile_scratch[0] == 0x5A5Au &&
                             tile_scratch[TILE_ELEMENTS - 1u] == 0x5A5Au,
                         "0x%04x..0x%04x — a self-addressed read could not have produced the "
                         "reference from this", tile_scratch[0], tile_scratch[TILE_ELEMENTS - 1u]);

    ft_tt_test_step("stage A — the result must TRACK the peer's contents, including zeros");
    rc = run_pass(noc, compute, peer, &config, compute, 7u, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("a second, different remote input gives its own answer", rc == FT_TT_OK,
                         "pass 7; first difference tile %u element %u", bad_tile, bad_element);

    rc = run_pass(noc, compute, peer, &config, compute, PASS_ZEROED, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("a ZEROED remote source produces a zeroed result", rc == FT_TT_OK,
                         "the destination held 0x%02x before the run; first difference tile %u "
                         "element %u", DESTINATION_POISON_BYTE, bad_tile, bad_element);
}

/**
 * The red half of stage A, in two forms, both of which must FAIL.
 *
 * 1. the reader pointed at the COMPUTE core's own L1 — the same addresses, the wrong
 *    core. This is the exact mistake the old self-addressed tests could not detect.
 * 2. the reader pointed at the peer but one tile too high — an address slip, never a
 *    coordinate slip.
 */
static void run_stage_a_red_half(const ft_tt_noc_access *noc, const tensix_endpoint *compute,
                                 const tensix_endpoint *peer,
                                 const ft_tt_image_placement placements[FT_TT_PROC_COUNT])
{
    ft_tt_launch_config config;
    uint64_t cb_mask = 0;
    unsigned bad_tile = 0;
    unsigned bad_element = 0;
    int rc;

    ft_tt_test_step("stage A red half — the same run with the source pointed somewhere else "
                    "must FAIL");

    if (write_cb_config(noc, compute, &cb_mask) != FT_TT_OK)
        return;
    build_config(placements, cb_mask, &config);

    rc = plant_runtime_args(noc, compute, &config, compute, g_layout.source0_address,
                            g_layout.source1_address, compute);
    if (rc == FT_TT_OK)
        rc = run_pass(noc, compute, peer, &config, compute, 0u, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("reading the COMPUTE core's own L1 instead of the peer's fails",
                         rc != FT_TT_OK, "rc=%d, first difference tile %u element %u", rc,
                         bad_tile, bad_element);

    rc = plant_runtime_args(noc, compute, &config, peer,
                            g_layout.source0_address + FT_TT_PHASE2_TILE_BYTES,
                            g_layout.source1_address, compute);
    if (rc == FT_TT_OK)
        rc = run_pass(noc, compute, peer, &config, compute, 0u, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("the peer's address shifted by one tile fails", rc != FT_TT_OK,
                         "rc=%d, first difference tile %u element %u", rc, bad_tile, bad_element);
}

/* ------------------------------------------------------------------------- */
/* Stage B — the writer crosses too                                          */
/* ------------------------------------------------------------------------- */

static void run_stage_b_remote_write(const ft_tt_noc_access *noc, const tensix_endpoint *compute,
                                     const tensix_endpoint *peer,
                                     const ft_tt_image_placement placements[FT_TT_PROC_COUNT])
{
    ft_tt_launch_config config;
    uint64_t cb_mask = 0;
    unsigned bad_tile = 0;
    unsigned bad_element = 0;
    unsigned completed = 0;
    unsigned pass;
    int rc;

    ft_tt_test_step("stage B — the writer crosses to Tensix %u as well, on NoC 1", peer->index);

    rc = write_cb_config(noc, compute, &cb_mask);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("CB config written", false, ft_tt_error_string(rc));
        return;
    }
    build_config(placements, cb_mask, &config);
    rc = plant_runtime_args(noc, compute, &config, peer, g_layout.source0_address,
                            g_layout.source1_address, peer);
    ft_tt_test_check_fmt("runtime args planted", rc == FT_TT_OK, "writer dst (%u,%u) 0x%05x",
                         peer->x, peer->y, g_layout.destination_address);
    if (rc != FT_TT_OK)
        return;

    rc = run_pass(noc, compute, peer, &config, peer, 0u, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("peer in, peer out is byte-exact", rc == FT_TT_OK,
                         "%u tiles x %u elements; matched up to tile %u element %u", TILE_COUNT,
                         TILE_ELEMENTS, bad_tile, bad_element);
    if (rc != FT_TT_OK)
        return;

    ft_tt_test_step("stage B — %u repeats, destination poisoned and checked every time",
                    REPEAT_COUNT);
    for (pass = 0; pass < REPEAT_COUNT; pass++) {
        rc = run_pass(noc, compute, peer, &config, peer, pass % 13u, &bad_tile, &bad_element);
        if (rc != FT_TT_OK)
            break;
        completed++;
    }
    ft_tt_test_check_fmt("every repeat was byte-exact", completed == REPEAT_COUNT,
                         "%u/%u, last rc=%d, first difference tile %u element %u", completed,
                         REPEAT_COUNT, rc, bad_tile, bad_element);
}

/* ------------------------------------------------------------------------- */
/* Stage C — the peer PRODUCES the operand                                   */
/* ------------------------------------------------------------------------- */

/**
 * The peer runs the triad entirely inside its own L1, then the compute core reads the
 * peer's result across the fabric and adds a zeroed second operand, so its destination
 * must equal the peer's result.
 *
 * This is the stage that is about Phase 5 rather than about NoC plumbing: the bytes that
 * crossed were computed by another Tensix, not staged by the host. Sequencing is the
 * host's — the peer's run completes before the compute core's is kicked — because
 * semaphores are still untested and a producer/consumer handshake without them would be
 * testing hope.
 */
static void run_stage_c_peer_produces(const ft_tt_noc_access *noc, const tensix_endpoint *compute,
                                      const tensix_endpoint *peer,
                                      const ft_tt_image_placement compute_placements[FT_TT_PROC_COUNT],
                                      const ft_tt_image_placement peer_placements[FT_TT_PROC_COUNT])
{
    ft_tt_launch_config peer_config;
    ft_tt_launch_config compute_config;
    uint64_t cb_mask = 0;
    unsigned bad_tile = 0;
    unsigned bad_element = 0;
    unsigned tile;
    int rc;

    ft_tt_test_step("stage C — Tensix %u computes the operand, Tensix %u consumes it across "
                    "the fabric", peer->index, compute->index);

    /* The peer's own run: everything in its L1, exactly step 12's shape. */
    rc = write_cb_config(noc, peer, &cb_mask);
    if (rc == FT_TT_OK) {
        build_config(peer_placements, cb_mask, &peer_config);
        rc = plant_runtime_args(noc, peer, &peer_config, peer, g_layout.source0_address,
                                g_layout.source1_address, peer);
    }
    if (rc == FT_TT_OK)
        rc = stage_sources(noc, peer, 3u);
    if (rc == FT_TT_OK)
        rc = poison_destination(noc, peer);
    if (rc == FT_TT_OK)
        rc = launch_and_wait(noc, peer, &peer_config);
    ft_tt_test_check_fmt("the peer ran the triad in its own L1", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    rc = check_destination(noc, peer, 3u, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("the peer's own result is byte-exact", rc == FT_TT_OK,
                         "first difference tile %u element %u", bad_tile, bad_element);
    if (rc != FT_TT_OK)
        return;

    /* Second operand: zeros, in the peer's L1, so the compute core's answer must be the
     * peer's result unchanged. */
    memset(tile_scratch, 0, sizeof(tile_scratch));
    for (tile = 0; tile < TILE_COUNT && rc == FT_TT_OK; tile++)
        rc = write_tile(noc, peer, g_layout.source1_address + tile * FT_TT_PHASE2_TILE_BYTES,
                        tile_scratch);
    if (rc == FT_TT_OK)
        rc = poison_local_sources(noc, compute);
    if (rc == FT_TT_OK)
        rc = poison_destination(noc, compute);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("staging the consumer run", false, ft_tt_error_string(rc));
        return;
    }

    rc = write_cb_config(noc, compute, &cb_mask);
    if (rc == FT_TT_OK) {
        build_config(compute_placements, cb_mask, &compute_config);
        /* Operand 0 is the PEER'S DESTINATION — the tiles the peer just wrote. */
        rc = plant_runtime_args(noc, compute, &compute_config, peer,
                                g_layout.destination_address, g_layout.source1_address, compute);
    }
    if (rc == FT_TT_OK)
        rc = launch_and_wait(noc, compute, &compute_config);
    ft_tt_test_check_fmt("the consumer run completed", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    rc = check_destination(noc, compute, 3u, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("the consumer reproduced the PRODUCER's result byte for byte",
                         rc == FT_TT_OK,
                         "pass 3 plus zeros; first difference tile %u element %u", bad_tile,
                         bad_element);
}

/* ------------------------------------------------------------------------- */

/** Read the peer index, defaulting to 1. Equal indices are refused: the whole file is
 *  about the difference between two cores. */
static int peer_index_from_environment(uint32_t compute_index, uint32_t *out_index)
{
    const char *value = getenv(PEER_INDEX_ENVIRONMENT);
    char *end = NULL;
    unsigned long parsed;

    if (!value || value[0] == '\0') {
        *out_index = compute_index + 1u;
        return FT_TT_OK;
    }
    parsed = strtoul(value, &end, 10);
    /* Syntax only; the live bound is ft_tt_coord_tensix_by_index()'s, which is derived
     * from the grid rather than from a constant that is wrong on a 120-core part. */
    if (end == value || *end != '\0' || parsed > UINT32_MAX) {
        ft_tt_log_error("core_to_core: %s=\"%s\" is not a Tensix index", PEER_INDEX_ENVIRONMENT,
                        value);
        return FT_TT_EINVAL;
    }
    *out_index = (uint32_t)parsed;
    return FT_TT_OK;
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_program_directory program;
    ft_tt_image_placement firmware_placements[FT_TT_PROC_COUNT];
    ft_tt_image_placement compute_placements[FT_TT_PROC_COUNT];
    ft_tt_image_placement peer_placements[FT_TT_PROC_COUNT];
    tensix_endpoint compute;
    tensix_endpoint peer;
    const char *directory;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <program-directory>\n", argv[0]);
        return 2;
    }
    directory = argv[1];

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;

    memset(&compute, 0, sizeof(compute));
    memset(&peer, 0, sizeof(peer));
    rc = ft_tt_hardware_session_tensix_from_environment(&session, &compute.x, &compute.y,
                                                        &compute.index);
    if (rc == FT_TT_OK)
        rc = peer_index_from_environment(compute.index, &peer.index);
    if (rc == FT_TT_OK && peer.index == compute.index) {
        ft_tt_log_error("core_to_core: the compute core and the peer are both index %u — a "
                        "cross-core test on one core is the vacuous test this file replaces",
                        compute.index);
        rc = FT_TT_EINVAL;
    }
    if (rc == FT_TT_OK)
        rc = ft_tt_hardware_session_tensix(&session, peer.index, &peer.x, &peer.y);
    /* The env helper claimed the compute core; the PEER is a second core this
     * test drives, and an unclaimed peer is exactly the silent gap the split is
     * meant to close. It is a separate claim because its index is only known
     * after the compute index has been resolved. */
    if (rc == FT_TT_OK)
        rc = ft_tt_hardware_session_claim_cores(&session, &peer.index, 1u,
                                                "core_to_core peer", NULL, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    printf("peer Tensix %u at translated (%u, %u)  [%s]\n", peer.index, (unsigned)peer.x,
           (unsigned)peer.y, PEER_INDEX_ENVIRONMENT);

    rc = ft_tt_firmware_set_load(directory, &firmware);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    rc = ft_tt_program_directory_load(directory, 0x1fu, &program);
    if (rc != FT_TT_OK) {
        ft_tt_firmware_set_release(&firmware);
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    ft_tt_test_step("setup — boot both cores and load the vector_add triad on each from %s",
                    directory);
    rc = ft_tt_core_boot_firmware(&session.noc, compute.x, compute.y, &firmware,
                                  firmware_placements);
    if (rc == FT_TT_OK)
        rc = ft_tt_phase2_l1_layout_derive(firmware_placements, FT_TT_PROC_COUNT, &g_layout);
    ft_tt_test_check_fmt("compute core booted and its L1 map derived", rc == FT_TT_OK,
                         "base 0x%05x, rc=%d (%s)", g_layout.kernel_config_base, rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    rc = ft_tt_core_boot_firmware(&session.noc, peer.x, peer.y, &firmware, NULL);
    ft_tt_test_check_fmt("peer core booted", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    rc = ft_tt_core_load_program(&session.noc, compute.x, compute.y, &program.program,
                                 compute_placements);
    if (rc == FT_TT_OK)
        rc = ft_tt_core_load_program(&session.noc, peer.x, peer.y, &program.program,
                                     peer_placements);
    ft_tt_test_check_fmt("both cores hold all five images", rc == FT_TT_OK, "%u images each",
                         program.program.image_count);
    if (rc != FT_TT_OK)
        goto done;

    run_stage_a_remote_read(&session.noc, &compute, &peer, compute_placements);
    if (ft_tt_test_failure_count() != 0)
        goto done;

    run_stage_a_red_half(&session.noc, &compute, &peer, compute_placements);
    if (ft_tt_test_failure_count() != 0)
        goto done;

    run_stage_b_remote_write(&session.noc, &compute, &peer, compute_placements);
    if (ft_tt_test_failure_count() != 0)
        goto done;

    run_stage_c_peer_produces(&session.noc, &compute, &peer, compute_placements, peer_placements);

done:
    /* Both cores, on every exit path: a core left running can still issue NoC
     * transactions, and this is the first program whose transactions target another
     * core's L1. */
    (void)ft_tt_core_park(&session.noc, compute.x, compute.y);
    (void)ft_tt_core_park(&session.noc, peer.x, peer.y);
    ft_tt_program_directory_release(&program);
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
