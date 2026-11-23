/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_phase2_dram.c — the first NoC traffic a kernel has issued that LEAVES the core.
 *
 * Plan Phase 2 follow-up 6. Everything Foundry has run on the card so far put every
 * buffer in the target core's own L1: step 12's reader and writer were pointed at the
 * core they run on, so the whole reader/writer split — and Phase 5's spatial thesis on
 * top of it — has never moved a byte between two endpoints. This test moves the source
 * data to **DRAM bank 0** and makes the reader fetch it over the NoC.
 *
 * Two stages, in this order, because they differ by which DIRECTION crosses the fabric:
 *
 *   A  sources in DRAM, destination in this core's L1.  The reader issues a real
 *      cross-endpoint READ; nothing the device writes leaves the core. If the DRAM
 *      endpoint were wrong this is the cheapest way to find out.
 *   B  sources in DRAM, destination in DRAM.  Adds the writer's cross-endpoint WRITE,
 *      on the other NoC, so both directions and both data-movement cores are covered.
 *
 * Every coordinate comes from ft_tt_coord against live ARC telemetry — the Tensix from
 * ft_tt_hardware_session_tensix() and the DRAM bank from ft_tt_coord_dram_bank() in
 * TRANSLATED form, which is the space a TLB window is aimed with and, on Blackhole,
 * the space that goes on the wire unmodified on both NoCs (DYNAMIC_NOC_X/Y are the
 * identity). Nothing here is probed: a wrong endpoint does
 * not fail, it hangs the chip (spec §7.3).
 *
 * Byte-exactness alone would not prove the data came from DRAM — a stale CB could
 * produce the same answer. So the run is repeated against three different DRAM
 * contents, including zeros, and the result must track every one of them. The zeroed
 * pass is the red half (coding-best-practices.md §9): if the output stayed correct with
 * the DRAM source wiped, the reader was reading something else.
 *
 * Usage:  test_phase2_dram <program-directory>
 *   The directory is a build_tensix_program.sh output for the vector_add triad at the
 *   default 2048-byte tile, e.g. programs/vector_add.
 *
 * Build:
 *   make -f Makefile.phase2_followups PHASE1B=1 build/test_phase2_dram
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_buffer.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_noc.h"
#include "ft_tt_phase2_l1_layout.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>

#define RUN_TIMEOUT_MS 5000u
#define TILE_COUNT     2u
#define TILE_ELEMENTS  (FT_TT_PHASE2_TILE_BYTES / 2u)
#define BUFFER_BYTES   (TILE_COUNT * FT_TT_PHASE2_TILE_BYTES)
#define CB_CONFIG_BYTES (FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4u)

/** Repeats of the DRAM-to-DRAM stage. Enough to catch a path that works once and leaks
 *  state, cheap enough that a failure is found in seconds. */
#define REPEAT_COUNT 100u

/** DRAM bank this test owns. Never derive a second one by arithmetic on this. */
#define DRAM_LOGICAL_BANK 0u

/** The L1 map, derived from where the firmware images landed rather than chosen. A
 *  file-scope object because every helper below needs it and threading one more
 *  parameter through all of them would say less than this comment does. Written once,
 *  in main(), before anything reads it. */
static ft_tt_phase2_l1_layout g_layout;

/**
 * Boot the firmware and DERIVE the L1 map from where its images landed.
 *
 * The map's base used to be the literal 0x14000. It is now the higher of "the top of the
 * five firmware images the loader just placed" and "the end of the regions the stock memory map
 * reserves above MEM_MAP_END", which is exactly what ft_tt_image_placement_l1_floor()
 * was written to compute and had no caller for.
 */
static int boot_and_derive_layout(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                  const ft_tt_firmware_set *firmware)
{
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    int rc = ft_tt_core_boot_firmware(noc, x, y, firmware, placements);

    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_phase2_l1_layout_derive(placements, FT_TT_PROC_COUNT, &g_layout);
}

static uint16_t tile_in0[TILE_ELEMENTS];
static uint16_t tile_in1[TILE_ELEMENTS];
static uint16_t tile_expected[TILE_ELEMENTS];
static uint16_t tile_readback[TILE_ELEMENTS];

/** Where the three DRAM buffers of one run live, plus the endpoint they live at. */
typedef struct dram_region {
    ft_tt_noc_coord coord;
    ft_tt_buffer source0;
    ft_tt_buffer source1;
    ft_tt_buffer destination;
} dram_region;

/** bfloat16 is the top 16 bits of the binary32 pattern. Every value here is a small
 *  integer, so the truncation and the sum are both exact. */
static uint16_t bfloat16_from_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

/**
 * Build the two input tiles and the reference result for one tile of one pass.
 *
 * `pass` shifts the values, so two passes over the same buffers cannot pass on each
 * other's answer. PASS_ZEROED builds an all-zero input, whose reference is all-zero too.
 */
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

/** Write both source tile sets into DRAM and wipe the DRAM destination. */
static int stage_dram_sources(ft_tt_noc_path *path, const dram_region *region, unsigned pass)
{
    unsigned tile;

    for (tile = 0; tile < TILE_COUNT; tile++) {
        uint64_t offset = (uint64_t)tile * FT_TT_PHASE2_TILE_BYTES;
        int rc;

        build_reference_tiles(tile, pass);
        rc = ft_tt_write_dram(path, region->coord, region->source0.address + offset, tile_in0,
                              sizeof(tile_in0));
        if (rc == FT_TT_OK)
            rc = ft_tt_write_dram(path, region->coord, region->source1.address + offset, tile_in1,
                                  sizeof(tile_in1));
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/** Prove the host's own view of DRAM before asking a kernel to read it. */
static int verify_dram_sources(ft_tt_noc_path *path, const dram_region *region, unsigned pass)
{
    unsigned tile;

    for (tile = 0; tile < TILE_COUNT; tile++) {
        uint64_t offset = (uint64_t)tile * FT_TT_PHASE2_TILE_BYTES;
        int rc = ft_tt_read_dram(path, region->coord, region->source0.address + offset,
                                 tile_readback, sizeof(tile_readback));

        if (rc != FT_TT_OK)
            return rc;
        build_reference_tiles(tile, pass);
        if (first_difference(tile_readback, tile_in0) != TILE_ELEMENTS)
            return FT_TT_ERROR;
    }
    return FT_TT_OK;
}

/** Fill the DRAM destination with a value the kernel must overwrite. */
static int poison_dram_destination(ft_tt_noc_path *path, const dram_region *region)
{
    unsigned tile;

    memset(tile_readback, 0xA5, sizeof(tile_readback));
    for (tile = 0; tile < TILE_COUNT; tile++) {
        int rc = ft_tt_write_dram(path, region->coord,
                                  region->destination.address +
                                      (uint64_t)tile * FT_TT_PHASE2_TILE_BYTES,
                                  tile_readback, sizeof(tile_readback));

        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/** Wipe the L1 destination the stage-A run writes into. */
static int wipe_l1_destination(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    unsigned tile;

    memset(tile_readback, 0, sizeof(tile_readback));
    for (tile = 0; tile < TILE_COUNT; tile++) {
        int rc = noc->write(noc->user, x, y,
                            g_layout.destination_address + tile * FT_TT_PHASE2_TILE_BYTES,
                            tile_readback, sizeof(tile_readback));

        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

static int check_l1_destination(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, unsigned pass,
                                unsigned *out_bad_tile, unsigned *out_bad_element)
{
    unsigned tile;

    for (tile = 0; tile < TILE_COUNT; tile++) {
        unsigned difference;
        int rc = noc->read(noc->user, x, y,
                           g_layout.destination_address + tile * FT_TT_PHASE2_TILE_BYTES,
                           tile_readback, sizeof(tile_readback));

        if (rc != FT_TT_OK)
            return rc;
        build_reference_tiles(tile, pass);
        difference = first_difference(tile_readback, tile_expected);
        if (difference != TILE_ELEMENTS) {
            *out_bad_tile = tile;
            *out_bad_element = difference;
            return FT_TT_ERROR;
        }
    }
    *out_bad_tile = TILE_COUNT;
    *out_bad_element = TILE_ELEMENTS;
    return FT_TT_OK;
}

static int check_dram_destination(ft_tt_noc_path *path, const dram_region *region, unsigned pass,
                                  unsigned *out_bad_tile, unsigned *out_bad_element)
{
    unsigned tile;

    for (tile = 0; tile < TILE_COUNT; tile++) {
        unsigned difference;
        int rc = ft_tt_read_dram(path, region->coord,
                                 region->destination.address +
                                     (uint64_t)tile * FT_TT_PHASE2_TILE_BYTES,
                                 tile_readback, sizeof(tile_readback));

        if (rc != FT_TT_OK)
            return rc;
        build_reference_tiles(tile, pass);
        difference = first_difference(tile_readback, tile_expected);
        if (difference != TILE_ELEMENTS) {
            *out_bad_tile = tile;
            *out_bad_element = difference;
            return FT_TT_ERROR;
        }
    }
    *out_bad_tile = TILE_COUNT;
    *out_bad_element = TILE_ELEMENTS;
    return FT_TT_OK;
}

/** Describe the three circular buffers, all still in this core's L1. */
static int write_cb_config(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint64_t *out_mask)
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
    return noc->write(noc->user, x, y, g_layout.cb_config_address, buffer, sizeof(buffer));
}

static void build_config(const ft_tt_image_placement placements[FT_TT_PROC_COUNT], uint64_t cb_mask,
                         ft_tt_launch_config *out_config)
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
        out_config->runtime_args_address[processor] = ft_tt_phase2_runtime_args_address(&g_layout, processor);
    }
}

/**
 * Plant runtime args for one stage.
 *
 * The reader always names the DRAM endpoint; the writer names whatever
 * `destination_*` says, which is how stage A and stage B differ.
 */
static int plant_runtime_args(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                              const ft_tt_launch_config *config, const dram_region *region,
                              uint32_t destination_x, uint32_t destination_y,
                              uint32_t destination_address)
{
    uint32_t reader_arguments[7];
    uint32_t writer_arguments[4];
    uint32_t tile_count = TILE_COUNT;
    unsigned processor;
    int rc;

    reader_arguments[0] = tile_count;
    reader_arguments[1] = region->coord.x;
    reader_arguments[2] = region->coord.y;
    reader_arguments[3] = (uint32_t)region->source0.address;
    reader_arguments[4] = region->coord.x;
    reader_arguments[5] = region->coord.y;
    reader_arguments[6] = (uint32_t)region->source1.address;

    writer_arguments[0] = tile_count;
    writer_arguments[1] = destination_x;
    writer_arguments[2] = destination_y;
    writer_arguments[3] = destination_address;

    rc = ft_tt_launch_plant_runtime_args(noc, x, y, config->runtime_args_address[FT_TT_PROC_DM0],
                                         reader_arguments, 7u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_plant_runtime_args(noc, x, y,
                                             config->runtime_args_address[FT_TT_PROC_DM1],
                                             writer_arguments, 4u);
    for (processor = FT_TT_PROC_MATH0; processor < FT_TT_PROC_COUNT && rc == FT_TT_OK; processor++)
        rc = ft_tt_launch_plant_runtime_args(noc, x, y, config->runtime_args_address[processor],
                                             &tile_count, 1u);
    return rc;
}

static int launch_and_wait(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                           const ft_tt_launch_config *config)
{
    int rc = ft_tt_launch_write_launch_message(noc, x, y, config);

    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_send_go(noc, x, y);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_wait_done(noc, x, y, RUN_TIMEOUT_MS, NULL);
}

/* ------------------------------------------------------------------------- */
/* Stage A — the reader crosses the fabric, the writer does not               */
/* ------------------------------------------------------------------------- */

static void run_stage_a_dram_to_l1(const ft_tt_noc_access *noc, ft_tt_noc_path *path, uint16_t x,
                                   uint16_t y,
                                   const ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                                   const dram_region *region)
{
    ft_tt_launch_config config;
    uint64_t cb_mask = 0;
    unsigned bad_tile = 0;
    unsigned bad_element = 0;
    int rc;

    ft_tt_test_step("stage A — reader pulls both operands from DRAM bank %u at translated "
                    "(%u, %u); writer stays in this core's L1",
                    DRAM_LOGICAL_BANK, (unsigned)region->coord.x, (unsigned)region->coord.y);

    rc = write_cb_config(noc, x, y, &cb_mask);
    ft_tt_test_check_fmt("CB config written", rc == FT_TT_OK, "mask=0x%llx",
                         (unsigned long long)cb_mask);
    if (rc != FT_TT_OK)
        return;

    build_config(placements, cb_mask, &config);
    rc = plant_runtime_args(noc, x, y, &config, region, x, y, g_layout.destination_address);
    ft_tt_test_check_fmt("runtime args planted", rc == FT_TT_OK,
                         "reader src DRAM 0x%llx / 0x%llx, writer dst L1 (%u,%u) 0x%05x",
                         (unsigned long long)region->source0.address,
                         (unsigned long long)region->source1.address, x, y,
                         g_layout.destination_address);
    if (rc != FT_TT_OK)
        return;

    rc = stage_dram_sources(path, region, 0u);
    ft_tt_test_check_fmt("source tiles written to DRAM", rc == FT_TT_OK, "%u tiles x %u B",
                         TILE_COUNT, FT_TT_PHASE2_TILE_BYTES);
    if (rc != FT_TT_OK)
        return;

    rc = verify_dram_sources(path, region, 0u);
    ft_tt_test_check("the host reads back what it wrote to DRAM", rc == FT_TT_OK,
                     "so a later mismatch is the kernel's, not the staging's");
    if (rc != FT_TT_OK)
        return;

    rc = wipe_l1_destination(noc, x, y);
    if (rc == FT_TT_OK)
        rc = launch_and_wait(noc, x, y, &config);
    ft_tt_test_check_fmt("the triad completed with a DRAM source", rc == FT_TT_OK, "rc=%d (%s)",
                         rc, ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    rc = check_l1_destination(noc, x, y, 0u, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("the L1 result is byte-exact against the C reference", rc == FT_TT_OK,
                         "%u tiles x %u elements; matched up to tile %u element %u", TILE_COUNT,
                         TILE_ELEMENTS, bad_tile, bad_element);
}

/* ------------------------------------------------------------------------- */
/* Stage B — both directions cross the fabric                                */
/* ------------------------------------------------------------------------- */

/** One full DRAM-to-DRAM run at input pass `pass`. Returns FT_TT_OK on a byte-exact
 *  result, and reports the first difference otherwise. */
static int run_dram_pass(const ft_tt_noc_access *noc, ft_tt_noc_path *path, uint16_t x, uint16_t y,
                         const ft_tt_launch_config *config, const dram_region *region,
                         unsigned pass, unsigned *out_bad_tile, unsigned *out_bad_element)
{
    int rc = stage_dram_sources(path, region, pass);

    if (rc == FT_TT_OK)
        rc = poison_dram_destination(path, region);
    if (rc == FT_TT_OK)
        rc = launch_and_wait(noc, x, y, config);
    if (rc != FT_TT_OK)
        return rc;
    return check_dram_destination(path, region, pass, out_bad_tile, out_bad_element);
}

static void run_stage_b_dram_to_dram(const ft_tt_noc_access *noc, ft_tt_noc_path *path, uint16_t x,
                                     uint16_t y,
                                     const ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                                     const dram_region *region)
{
    ft_tt_launch_config config;
    uint64_t cb_mask = 0;
    unsigned bad_tile = 0;
    unsigned bad_element = 0;
    unsigned completed = 0;
    unsigned pass;
    int rc;

    ft_tt_test_step("stage B — DRAM in, DRAM out: the writer crosses the fabric too, on NoC 1");

    rc = write_cb_config(noc, x, y, &cb_mask);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("CB config written", false, ft_tt_error_string(rc));
        return;
    }
    build_config(placements, cb_mask, &config);
    rc = plant_runtime_args(noc, x, y, &config, region, region->coord.x, region->coord.y,
                            (uint32_t)region->destination.address);
    ft_tt_test_check_fmt("runtime args planted", rc == FT_TT_OK,
                         "writer dst DRAM (%u,%u) 0x%llx", (unsigned)region->coord.x,
                         (unsigned)region->coord.y,
                         (unsigned long long)region->destination.address);
    if (rc != FT_TT_OK)
        return;

    rc = run_dram_pass(noc, path, x, y, &config, region, 0u, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("DRAM to DRAM is byte-exact against the C reference", rc == FT_TT_OK,
                         "%u tiles x %u elements; matched up to tile %u element %u", TILE_COUNT,
                         TILE_ELEMENTS, bad_tile, bad_element);
    if (rc != FT_TT_OK)
        return;

    /* Different DRAM contents, same everything else. If the output tracks the input the
     * reader really read DRAM; if it does not, it read something stale. */
    ft_tt_test_step("stage B — the result must TRACK the DRAM contents, including zeros");
    rc = run_dram_pass(noc, path, x, y, &config, region, 7u, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("a second, different DRAM input gives its own answer", rc == FT_TT_OK,
                         "pass 7; first difference tile %u element %u", bad_tile, bad_element);

    rc = run_dram_pass(noc, path, x, y, &config, region, PASS_ZEROED, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("a ZEROED DRAM source produces a zeroed result", rc == FT_TT_OK,
                         "the destination held 0xA5 before the run, so this is not staleness; "
                         "first difference tile %u element %u",
                         bad_tile, bad_element);

    ft_tt_test_step("stage B — %u repeats, destination poisoned and checked every time",
                    REPEAT_COUNT);
    for (pass = 0; pass < REPEAT_COUNT; pass++) {
        rc = run_dram_pass(noc, path, x, y, &config, region, pass % 13u, &bad_tile, &bad_element);
        if (rc != FT_TT_OK)
            break;
        completed++;
    }
    ft_tt_test_check_fmt("every repeat was byte-exact", completed == REPEAT_COUNT,
                         "%u/%u, last rc=%d, first difference tile %u element %u", completed,
                         REPEAT_COUNT, rc, bad_tile, bad_element);
}

/* ------------------------------------------------------------------------- */

/** Claim the three DRAM buffers and the endpoint that serves them. */
static int claim_dram_region(ft_tt_allocator *allocator, const ft_tt_grid_info *grid,
                             dram_region *out_region)
{
    int rc;

    memset(out_region, 0, sizeof(*out_region));
    rc = ft_tt_coord_dram_bank(grid, DRAM_LOGICAL_BANK, FT_TT_COORD_TRANSLATED,
                               &out_region->coord);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_buffer_alloc(allocator, FT_TT_MEM_SPACE_DRAM, DRAM_LOGICAL_BANK, BUFFER_BYTES,
                            &out_region->source0);
    if (rc == FT_TT_OK)
        rc = ft_tt_buffer_alloc(allocator, FT_TT_MEM_SPACE_DRAM, DRAM_LOGICAL_BANK, BUFFER_BYTES,
                                &out_region->source1);
    if (rc == FT_TT_OK)
        rc = ft_tt_buffer_alloc(allocator, FT_TT_MEM_SPACE_DRAM, DRAM_LOGICAL_BANK, BUFFER_BYTES,
                                &out_region->destination);
    if (rc != FT_TT_OK)
        return rc;

    /* The kernels take a 32-bit base address as a runtime arg and build the NoC address
     * themselves, so a buffer above 4 GiB in a bank would be truncated silently. */
    if ((out_region->source0.address | out_region->source1.address |
         out_region->destination.address) > 0xFFFFFFFFu) {
        ft_tt_log_error("dram: a buffer landed above 4 GiB, which the kernels' 32-bit "
                        "address argument cannot express");
        return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_program_directory program;
    ft_tt_allocator allocator;
    ft_tt_noc_path dram_path;
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    dram_region region;
    ft_tt_ctx *kmd;
    const char *directory;
    bool path_open = false;
    uint16_t x = 0;
    uint16_t y = 0;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <program-directory>\n", argv[0]);
        return 2;
    }
    directory = argv[1];

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;
    rc = ft_tt_hardware_session_tensix_from_environment(&session, &x, &y, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    /* This arm allocates DRAM and opens its own window through the TT-KMD CONTEXT,
     * so it is one of the two harnesses that cannot run on the userspace driver.
     * It says which transport it needs rather than dereferencing a handle that is
     * NULL there. */
    kmd = ft_tt_hardware_session_kmd_context(&session);
    if (!kmd) {
        ft_tt_log_error("this suite allocates DRAM through the tt-kmd context "
                        "(ft_tt_allocator_init); run it with FT_TT_DRIVER=kmd");
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    /* A SECOND window: ft_tt_noc_path must be the only thing aiming the window it is
     * attached to, and the session's window belongs to the launch accessor. */
    rc = ft_tt_noc_path_open(kmd, FT_TT_KMD_CACHE_WC, &dram_path);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    path_open = true;

    rc = ft_tt_allocator_init(&allocator, kmd, &session.grid, 0u);
    if (rc == FT_TT_OK)
        rc = claim_dram_region(&allocator, &session.grid, &region);
    if (rc != FT_TT_OK) {
        (void)ft_tt_noc_path_close(&dram_path);
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    printf("target Tensix 0 at translated (%u, %u); DRAM bank %u at translated (%u, %u)\n",
           (unsigned)x, (unsigned)y, DRAM_LOGICAL_BANK, (unsigned)region.coord.x,
           (unsigned)region.coord.y);

    rc = ft_tt_firmware_set_load(directory, &firmware);
    if (rc != FT_TT_OK)
        goto close_path;
    rc = ft_tt_program_directory_load(directory, 0x1fu, &program);
    if (rc != FT_TT_OK) {
        ft_tt_firmware_set_release(&firmware);
        goto close_path;
    }

    ft_tt_test_step("setup — boot the firmware and load the vector_add triad from %s", directory);
    rc = boot_and_derive_layout(&session.noc, x, y, &firmware);
    ft_tt_test_check_fmt("firmware booted", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    rc = ft_tt_core_load_program(&session.noc, x, y, &program.program, placements);
    ft_tt_test_check_fmt("all five images loaded", rc == FT_TT_OK, "%u images",
                         program.program.image_count);
    if (rc != FT_TT_OK)
        goto done;

    run_stage_a_dram_to_l1(&session.noc, &dram_path, x, y, placements, &region);
    if (ft_tt_test_failure_count() != 0)
        goto done;

    run_stage_b_dram_to_dram(&session.noc, &dram_path, x, y, placements, &region);

done:
    /* Assert reset on every exit path: a core left running can still issue NoC
     * transactions, and this is the first program whose transactions leave the tile. */
    (void)ft_tt_core_park(&session.noc, x, y);
    ft_tt_program_directory_release(&program);
    ft_tt_firmware_set_release(&firmware);
close_path:
    if (path_open)
        (void)ft_tt_noc_path_close(&dram_path);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
