/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_core_probe.c
 * @brief Implementation of the one-core health test: boot, run a vector-add triad, compare.
 *
 * The run itself is the Phase 2 checklist step 12 shape (tests/test_phase2_vector_add.c),
 * reduced to what a health check needs and with the L1 map DERIVED from where the firmware
 * images landed rather than chosen.
 *
 * @status Runs on the p150a test card.
 * @issues None
 * @todo None
 */

#include "ft_tt_core_probe.h"

#include <stdio.h>
#include <string.h>

#include "ft_tt_phase2_l1_layout.h"

#define PROBE_RUN_TIMEOUT_MS 5000u
#define PROBE_TILE_ELEMENTS  (FT_TT_PHASE2_TILE_BYTES / 2u)
#define PROBE_CB_CONFIG_BYTES (FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4u)

/** Every processor: the triad is reader on BRISC, writer on NCRISC, compute on all three
 *  TRISCs. A health check that left the backend out could not see the failure that
 *  damaged 14 cores. */
#define PROBE_PROCESSOR_MASK 0x1fu

static uint16_t tile_in0[PROBE_TILE_ELEMENTS];
static uint16_t tile_in1[PROBE_TILE_ELEMENTS];
static uint16_t tile_expected[PROBE_TILE_ELEMENTS];
static uint16_t tile_readback[PROBE_TILE_ELEMENTS];

/** bfloat16 is the top 16 bits of the binary32 pattern, truncated. */
static uint16_t bfloat16_from_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

/**
 * Build both operand tiles and the reference for one tile index.
 *
 * Small integers, so every sum is exactly representable in bfloat16 and "byte-exact" is
 * a fair demand rather than a rounding lottery. The per-tile offset is added OUTSIDE the
 * modulus so two tiles never collapse onto identical data.
 */
static void build_reference_tiles(unsigned tile_index)
{
    unsigned element;

    for (element = 0; element < PROBE_TILE_ELEMENTS; element++) {
        float left = (float)((element % 16u) + tile_index + 1u);
        float right = (float)(((element * 3u) % 16u) + tile_index + 1u);

        tile_in0[element] = bfloat16_from_float(left);
        tile_in1[element] = bfloat16_from_float(right);
        tile_expected[element] = bfloat16_from_float(left + right);
    }
}

static unsigned first_difference(const uint16_t *left, const uint16_t *right)
{
    unsigned element;

    for (element = 0; element < PROBE_TILE_ELEMENTS; element++)
        if (left[element] != right[element])
            return element;
    return PROBE_TILE_ELEMENTS;
}

/** Describe the three circular buffers the vector_add triad was built against. */
static int write_cb_config(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                           const ft_tt_phase2_l1_layout *layout, uint64_t *out_mask)
{
    static uint8_t buffer[PROBE_CB_CONFIG_BYTES];
    const uint32_t addresses[3] = {layout->cb_in0_address, layout->cb_in1_address,
                                   layout->cb_out_address};
    static const unsigned indices[3] = {FT_TT_PHASE2_CB_IN0, FT_TT_PHASE2_CB_IN1,
                                        FT_TT_PHASE2_CB_OUT};
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
    return noc->write(noc->user, x, y, layout->cb_config_address, buffer, sizeof(buffer));
}

static void build_launch_config(const ft_tt_phase2_l1_layout *layout,
                                const ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                                uint64_t cb_mask, ft_tt_launch_config *out_config)
{
    unsigned processor;

    memset(out_config, 0, sizeof(*out_config));
    out_config->kernel_config_base = layout->kernel_config_base;
    out_config->enables = PROBE_PROCESSOR_MASK;
    out_config->local_cb_config_address = layout->cb_config_address;
    out_config->local_cb_mask = cb_mask;
    /* vector_add_reader.c is compiled with NOC_INDEX=0 and BRISC firmware re-reads this
     * at run time. The two MUST agree. */
    out_config->brisc_noc_id = 0u;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        out_config->kernel_text_address[processor] = placements[processor].entry_address;
        out_config->kernel_text_bytes[processor] = placements[processor].text_bytes;
        out_config->runtime_args_address[processor] =
            ft_tt_phase2_runtime_args_address(layout, processor);
    }
}

static int plant_runtime_args(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                              const ft_tt_phase2_l1_layout *layout,
                              const ft_tt_launch_config *config)
{
    uint32_t reader_arguments[7];
    uint32_t writer_arguments[4];
    uint32_t tile_count = FT_TT_CORE_PROBE_TILE_COUNT;
    unsigned processor;
    int rc;

    reader_arguments[0] = tile_count;
    reader_arguments[1] = x;
    reader_arguments[2] = y;
    reader_arguments[3] = layout->source0_address;
    reader_arguments[4] = x;
    reader_arguments[5] = y;
    reader_arguments[6] = layout->source1_address;

    writer_arguments[0] = tile_count;
    writer_arguments[1] = x;
    writer_arguments[2] = y;
    writer_arguments[3] = layout->destination_address;

    rc = ft_tt_launch_plant_runtime_args(noc, x, y, config->runtime_args_address[FT_TT_PROC_DM0],
                                         reader_arguments, 7u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_plant_runtime_args(noc, x, y,
                                             config->runtime_args_address[FT_TT_PROC_DM1],
                                             writer_arguments, 4u);
    /* Every enabled TRISC needs a VALID runtime-arg block, not just the one you care
     * about — a stale block stalled a core once (dev notes §5). */
    for (processor = FT_TT_PROC_MATH0; processor < FT_TT_PROC_COUNT && rc == FT_TT_OK; processor++)
        rc = ft_tt_launch_plant_runtime_args(noc, x, y, config->runtime_args_address[processor],
                                             &tile_count, 1u);
    return rc;
}

static int stage_sources(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                         const ft_tt_phase2_l1_layout *layout)
{
    unsigned tile;

    for (tile = 0; tile < FT_TT_CORE_PROBE_TILE_COUNT; tile++) {
        uint32_t offset = tile * FT_TT_PHASE2_TILE_BYTES;
        int rc;

        build_reference_tiles(tile);
        rc = noc->write(noc->user, x, y, layout->source0_address + offset, tile_in0,
                        sizeof(tile_in0));
        if (rc != FT_TT_OK)
            return rc;
        rc = noc->write(noc->user, x, y, layout->source1_address + offset, tile_in1,
                        sizeof(tile_in1));
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/** Wipe the destination so a run that did nothing reads as zeros, not as last run's answer. */
static int wipe_destination(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                            const ft_tt_phase2_l1_layout *layout)
{
    unsigned tile;

    memset(tile_readback, 0, sizeof(tile_readback));
    for (tile = 0; tile < FT_TT_CORE_PROBE_TILE_COUNT; tile++) {
        int rc = noc->write(noc->user, x, y,
                            layout->destination_address + tile * FT_TT_PHASE2_TILE_BYTES,
                            tile_readback, sizeof(tile_readback));

        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

static int check_destination(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                             const ft_tt_phase2_l1_layout *layout, unsigned *out_bad_tile,
                             unsigned *out_bad_element)
{
    unsigned tile;

    *out_bad_tile = FT_TT_CORE_PROBE_TILE_COUNT;
    *out_bad_element = PROBE_TILE_ELEMENTS;
    for (tile = 0; tile < FT_TT_CORE_PROBE_TILE_COUNT; tile++) {
        unsigned difference;
        int rc = noc->read(noc->user, x, y,
                           layout->destination_address + tile * FT_TT_PHASE2_TILE_BYTES,
                           tile_readback, sizeof(tile_readback));

        if (rc != FT_TT_OK)
            return rc;
        build_reference_tiles(tile);
        difference = first_difference(tile_readback, tile_expected);
        if (difference != PROBE_TILE_ELEMENTS) {
            *out_bad_tile = tile;
            *out_bad_element = difference;
            return FT_TT_ERROR;
        }
    }
    return FT_TT_OK;
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
    return ft_tt_launch_wait_done(noc, x, y, PROBE_RUN_TIMEOUT_MS, NULL);
}

bool ft_tt_core_probe_passed(const ft_tt_core_probe_result *result)
{
    return result && !result->refused && result->repeats_requested != 0u &&
           result->repeats_passed == result->repeats_requested;
}

int ft_tt_core_probe_context_open(const char *directory, ft_tt_core_probe_context *out)
{
    int rc;

    if (!directory || !out)
        return FT_TT_EINVAL;
    memset(out, 0, sizeof(*out));

    rc = ft_tt_firmware_set_load(directory, &out->firmware);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_program_directory_load(directory, PROBE_PROCESSOR_MASK, &out->program);
    if (rc != FT_TT_OK) {
        ft_tt_firmware_set_release(&out->firmware);
        return rc;
    }
    return FT_TT_OK;
}

void ft_tt_core_probe_context_close(ft_tt_core_probe_context *context)
{
    if (!context)
        return;
    ft_tt_program_directory_release(&context->program);
    ft_tt_firmware_set_release(&context->firmware);
}

/** Record a failure and hand back FT_TT_OK: a failed probe is a RESULT, not an error. */
static int fail(ft_tt_core_probe_result *out, const char *stage, int code)
{
    out->failure_stage = stage;
    out->failure_code = code;
    return FT_TT_OK;
}

int ft_tt_core_probe_run(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                         ft_tt_core_probe_context *context, const ft_tt_core_status *status,
                         unsigned repeats, bool force, ft_tt_core_probe_result *out)
{
    ft_tt_image_placement firmware_placements[FT_TT_PROC_COUNT];
    ft_tt_image_placement kernel_placements[FT_TT_PROC_COUNT];
    ft_tt_phase2_l1_layout layout;
    ft_tt_launch_config config;
    ft_tt_noc_coord core;
    uint64_t cb_mask = 0;
    unsigned completed = 0;
    int rc;

    if (!noc || !grid || !context || !status || !out || repeats == 0u)
        return FT_TT_EINVAL;

    memset(out, 0, sizeof(*out));
    out->index = status->index;
    out->repeats_requested = repeats;
    out->first_bad_tile = FT_TT_CORE_PROBE_TILE_COUNT;
    out->first_bad_element = PROBE_TILE_ELEMENTS;

    /* Resolve independently of the status read rather than trusting its coordinate: the
     * one thing that must never be second-hand is the endpoint (spec §7.3). */
    rc = ft_tt_coord_tensix_by_index(grid, status->index, FT_TT_COORD_TRANSLATED, &core);
    if (rc != FT_TT_OK)
        return rc;
    out->x = (uint16_t)core.x;
    out->y = (uint16_t)core.y;

    /* Booting starts with assert_reset. Doing that to a core with a kernel still in it is
     * the exact move that wedged Tensix 0 (dev notes §1), so it is refused here. */
    if (!force &&
        (status->verdict == FT_TT_CORE_VERDICT_KERNEL_RUNNING || status->queue_heartbeat_ticking)) {
        out->refused = true;
        out->failure_stage = "refused: a kernel is running on this core";
        return FT_TT_EPERM;
    }

    rc = ft_tt_core_boot_firmware(noc, out->x, out->y, &context->firmware, firmware_placements);
    if (rc != FT_TT_OK)
        return fail(out, "firmware boot", rc);
    out->firmware_booted = true;

    rc = ft_tt_phase2_l1_layout_derive(firmware_placements, FT_TT_PROC_COUNT, &layout);
    if (rc != FT_TT_OK)
        return fail(out, "L1 layout", rc);

    rc = ft_tt_core_load_program(noc, out->x, out->y, &context->program.program,
                                 kernel_placements);
    if (rc != FT_TT_OK)
        return fail(out, "kernel load", rc);
    out->program_loaded = true;

    rc = write_cb_config(noc, out->x, out->y, &layout, &cb_mask);
    if (rc != FT_TT_OK)
        return fail(out, "CB config", rc);

    build_launch_config(&layout, kernel_placements, cb_mask, &config);
    rc = plant_runtime_args(noc, out->x, out->y, &layout, &config);
    if (rc != FT_TT_OK)
        return fail(out, "runtime args", rc);

    rc = stage_sources(noc, out->x, out->y, &layout);
    if (rc != FT_TT_OK)
        return fail(out, "staging", rc);

    while (completed < repeats) {
        rc = wipe_destination(noc, out->x, out->y, &layout);
        if (rc != FT_TT_OK)
            return fail(out, "destination wipe", rc);
        rc = launch_and_wait(noc, out->x, out->y, &config);
        if (rc != FT_TT_OK)
            return fail(out, "kernel did not complete", rc);
        out->completed = true;
        rc = check_destination(noc, out->x, out->y, &layout, &out->first_bad_tile,
                               &out->first_bad_element);
        if (rc != FT_TT_OK)
            return fail(out, "result is not byte-exact", rc);
        completed++;
        out->repeats_passed = completed;
    }
    /* The core is left with firmware up and idle. NOT parked — see the file header. */
    return FT_TT_OK;
}

void ft_tt_core_probe_print_line(const ft_tt_core_probe_result *result)
{
    if (!result)
        return;
    if (ft_tt_core_probe_passed(result)) {
        printf("%5u   (%2u,%2u)  PASS   %u/%u runs byte-exact, %u tiles x %u bf16 elements\n",
               result->index, result->x, result->y, result->repeats_passed,
               result->repeats_requested, FT_TT_CORE_PROBE_TILE_COUNT, PROBE_TILE_ELEMENTS);
        return;
    }
    printf("%5u   (%2u,%2u)  %s   %s (rc=%d)", result->index, result->x, result->y,
           result->refused ? "SKIP" : "FAIL",
           result->failure_stage ? result->failure_stage : "?", result->failure_code);
    if (result->first_bad_tile != FT_TT_CORE_PROBE_TILE_COUNT)
        printf(", first difference tile %u element %u", result->first_bad_tile,
               result->first_bad_element);
    printf(", %u/%u runs passed\n", result->repeats_passed, result->repeats_requested);
}
