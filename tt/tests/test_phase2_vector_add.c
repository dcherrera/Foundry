/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_phase2_vector_add.c — checklist steps 11 and 12 from docs/tt_launch_notes.md §6.
 *
 * The first arithmetic Foundry asks a Tensix to do, in two stages that differ by
 * exactly one thing: who moves the data.
 *
 *   11  HOST-STAGED. The host writes both input tiles straight into the circular-buffer
 *       FIFOs, primes the CB config, and hands the compute triad a full input CB by
 *       writing the CBs' page counters itself. Those counters are NoC-addressable
 *       overlay-stream registers on the target core (ft_tt_memmap.h), so the host can
 *       play producer without any data-movement kernel existing. THE DEVICE ISSUES NO
 *       NoC TRANSACTION AT ALL, so the blast radius is one core.
 *
 *   12  FULL TRIAD. Add vector_add_reader.c on BRISC and vector_add_writer.c on
 *       NCRISC. This is the first time the DEVICE issues NoC traffic. Both kernels are
 *       pointed at this core's own L1 — a self-addressed transaction — because the one
 *       endpoint we have proven exists in this coordinate space is the core we have
 *       been reading and writing all along (spec §7.3: a wrong endpoint hangs the chip).
 *
 * Both stages are checked byte-exact against a C reference. Inputs are small integers
 * so every sum is exactly representable in bfloat16 and "byte-exact" is a fair demand
 * rather than a rounding lottery.
 *
 * Build:
 *   cc -std=c99 -O1 -Wall -Wextra -Werror -pedantic -I.. -DFT_TT_PHASE1B_NOC_ENABLED \
 *      ../ft_tt_kmd.c ../ft_tt_device.c ../ft_tt_tlb.c ../ft_tt_arc_telemetry.c \
 *      ../ft_tt_coord.c ../ft_tt_elf.c ../ft_tt_kernel_blob.c ../ft_tt_launch.c \
 *      ft_tt_test_report.c ft_tt_hardware_session.c test_phase2_vector_add.c \
 *      -o build/test_phase2_vector_add
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_hardware_session.h"
#include "ft_tt_kernels.h"
#include "ft_tt_phase2_l1_layout.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>

#include <time.h>

#define RUN_TIMEOUT_MS   5000u
#define TILE_ELEMENTS    (FT_TT_PHASE2_TILE_BYTES / 2u)
#define CB_CONFIG_BYTES  (FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4u)

/** Tiles the full-triad run moves. Two, so the two-page CB ring wraps exactly once —
 *  a CB that is really a single slot passes a one-tile test. */
#define FULL_TRIAD_TILE_COUNT 2u

/** Step 12's acceptance: byte-exact, repeated. */
#define FULL_TRIAD_REPEAT_COUNT 1000u

/** How long the host will keep re-asserting a CB page counter against TRISC0's
 *  stock sync init before giving up. */
#define COUNTER_STAGE_ATTEMPTS 64u

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

/** bfloat16 is the top 16 bits of the IEEE-754 binary32 pattern, truncated. Every value
 *  this test uses is a small integer, so truncation is exact and so is the sum. */
static uint16_t bfloat16_from_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

/**
 * Build the two input tiles and the reference result.
 *
 * Element ORDER does not have to match the Tensix's 4-face tile layout: an elementwise
 * add preserves position whatever the permutation, as long as both operands use the
 * same one. What does matter is that every sum lands exactly on a bfloat16.
 */
static void build_reference_tiles(unsigned tile_index)
{
    unsigned element;

    for (element = 0; element < TILE_ELEMENTS; element++) {
        float left = (float)((element + tile_index) % 16u + 1u);
        float right = (float)((element * 3u + tile_index) % 16u + 1u);

        tile_in0[element] = bfloat16_from_float(left);
        tile_in1[element] = bfloat16_from_float(right);
        tile_expected[element] = bfloat16_from_float(left + right);
    }
}

/** Index of the first differing element, or TILE_ELEMENTS when they match. */
static unsigned first_difference(const uint16_t *left, const uint16_t *right)
{
    unsigned element;

    for (element = 0; element < TILE_ELEMENTS; element++) {
        if (left[element] != right[element])
            return element;
    }
    return TILE_ELEMENTS;
}

/** Describe the three circular buffers this program uses. */
static void describe_circular_buffers(ft_tt_launch_cb_entry entries[3])
{
    const uint32_t addresses[3] = {g_layout.cb_in0_address, g_layout.cb_in1_address,
                                  g_layout.cb_out_address};
    static const unsigned indices[3] = {FT_TT_PHASE2_CB_IN0, FT_TT_PHASE2_CB_IN1,
                                        FT_TT_PHASE2_CB_OUT};
    unsigned entry;

    for (entry = 0; entry < 3u; entry++) {
        entries[entry].index = indices[entry];
        entries[entry].address = addresses[entry];
        entries[entry].size_bytes = FT_TT_PHASE2_CB_BYTES;
        entries[entry].page_size_bytes = FT_TT_PHASE2_TILE_BYTES;
        entries[entry].page_count = FT_TT_PHASE2_CB_PAGE_COUNT;
    }
}

/** Write the CB config array into L1 and hand back the mask for the launch message. */
static int write_cb_config(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                           uint64_t *out_mask)
{
    static uint8_t buffer[CB_CONFIG_BYTES];
    ft_tt_launch_cb_entry entries[3];
    int rc;

    describe_circular_buffers(entries);
    rc = ft_tt_launch_encode_cb_config(entries, 3u, buffer, sizeof(buffer), out_mask);
    if (rc != FT_TT_OK)
        return rc;
    return noc->write(noc->user, x, y, g_layout.cb_config_address, buffer, sizeof(buffer));
}

/**
 * Set one CB's page counters and make them stick.
 *
 * TRISC0 re-zeroes all 64 counter pairs whenever BRISC asks it to, and BRISC asks after
 * every run — so a plain write here races that and loses silently, which shows up as a
 * hang, not an error. Write, read back, repeat.
 */
static int stage_cb_counters(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                             unsigned cb_index, uint32_t received, uint32_t acked)
{
    uint32_t received_address = FT_TT_CB_TILES_RECEIVED_ADDR(cb_index);
    uint32_t acked_address = FT_TT_CB_TILES_ACKED_ADDR(cb_index);
    unsigned attempt;

    for (attempt = 0; attempt < COUNTER_STAGE_ATTEMPTS; attempt++) {
        uint32_t observed_received = 0;
        uint32_t observed_acked = 0;
        int rc = noc->write(noc->user, x, y, acked_address, &acked, sizeof(acked));

        if (rc != FT_TT_OK)
            return rc;
        rc = noc->write(noc->user, x, y, received_address, &received, sizeof(received));
        if (rc != FT_TT_OK)
            return rc;
        rc = noc->read(noc->user, x, y, received_address, &observed_received,
                       sizeof(observed_received));
        if (rc != FT_TT_OK)
            return rc;
        rc = noc->read(noc->user, x, y, acked_address, &observed_acked, sizeof(observed_acked));
        if (rc != FT_TT_OK)
            return rc;
        if (observed_received == received && observed_acked == acked)
            return FT_TT_OK;
    }
    ft_tt_log_error("vector_add: CB %u counters would not hold their value", cb_index);
    return FT_TT_EIO;
}

/** Common launch config for the vector_add program. */
static void build_config(const ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                         uint32_t enables, uint64_t cb_mask, ft_tt_launch_config *out_config)
{
    unsigned processor;

    memset(out_config, 0, sizeof(*out_config));
    out_config->kernel_config_base = g_layout.kernel_config_base;
    out_config->enables = enables;
    out_config->local_cb_config_address = g_layout.cb_config_address;
    out_config->local_cb_mask = cb_mask;
    /* vector_add_reader.c is compiled with NOC_INDEX=0 and the firmware re-reads this
     * at run time. The two MUST agree. */
    out_config->brisc_noc_id = 0u;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if ((enables & (1u << processor)) == 0u)
            continue;
        out_config->kernel_text_address[processor] = placements[processor].entry_address;
        out_config->kernel_text_bytes[processor] = placements[processor].text_bytes;
        out_config->runtime_args_address[processor] = ft_tt_phase2_runtime_args_address(&g_layout, processor);
    }
}

static int launch_and_wait(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                           const ft_tt_launch_config *config, uint32_t timeout_ms)
{
    int rc = ft_tt_launch_write_launch_message(noc, x, y, config);

    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_send_go(noc, x, y);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_wait_done(noc, x, y, timeout_ms, NULL);
}

/* ------------------------------------------------------------------------- */
/* Step 11 — host-staged                                                     */
/* ------------------------------------------------------------------------- */

static void run_step_11_host_staged(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                    const ft_tt_image_placement placements[FT_TT_PROC_COUNT])
{
    ft_tt_launch_config config;
    uint64_t cb_mask = 0;
    uint32_t tile_count = 1u;
    uint32_t received_after = 0;
    unsigned processor;
    unsigned difference;
    int rc;

    ft_tt_test_step("step 11 — vector add with the HOST staging both input tiles "
                    "(no device NoC traffic at all)");

    build_reference_tiles(0u);

    rc = write_cb_config(noc, x, y, &cb_mask);
    ft_tt_test_check_fmt("CB config written", rc == FT_TT_OK, "mask=0x%llx at 0x%05x",
                         (unsigned long long)cb_mask, g_layout.cb_config_address);
    if (rc != FT_TT_OK)
        return;

    build_config(placements, 0x1cu, cb_mask, &config);
    ft_tt_test_check_fmt("enables covers the compute triad only", config.enables == 0x1cu,
                         "0x%02x", config.enables);

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if (config.runtime_args_address[processor] == 0u)
            continue;
        rc = ft_tt_launch_plant_runtime_args(noc, x, y, config.runtime_args_address[processor],
                                             &tile_count, 1u);
        if (rc != FT_TT_OK)
            break;
    }
    ft_tt_test_check_fmt("tile count planted for each TRISC", rc == FT_TT_OK, "%u tile", tile_count);
    if (rc != FT_TT_OK)
        return;

    /* Both input tiles straight into page 0 of their FIFOs, and a wiped output FIFO so
     * a stale result cannot pass for a computed one. */
    memset(tile_readback, 0, sizeof(tile_readback));
    rc = noc->write(noc->user, x, y, g_layout.cb_in0_address, tile_in0, sizeof(tile_in0));
    if (rc == FT_TT_OK)
        rc = noc->write(noc->user, x, y, g_layout.cb_in1_address, tile_in1, sizeof(tile_in1));
    if (rc == FT_TT_OK)
        rc = noc->write(noc->user, x, y, g_layout.cb_out_address, tile_readback,
                        sizeof(tile_readback));
    ft_tt_test_check_fmt("input tiles staged, output FIFO wiped", rc == FT_TT_OK, "%u B each",
                         FT_TT_PHASE2_TILE_BYTES);
    if (rc != FT_TT_OK)
        return;

    rc = ft_tt_core_wait_sync_registers_ready(noc, x, y, 1000u);
    ft_tt_test_check_fmt("TRISC0 finished re-initialising the CB counters", rc == FT_TT_OK,
                         "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;

    rc = stage_cb_counters(noc, x, y, FT_TT_PHASE2_CB_IN0, 1u, 0u);
    if (rc == FT_TT_OK)
        rc = stage_cb_counters(noc, x, y, FT_TT_PHASE2_CB_IN1, 1u, 0u);
    if (rc == FT_TT_OK)
        rc = stage_cb_counters(noc, x, y, FT_TT_PHASE2_CB_OUT, 0u, 0u);
    ft_tt_test_check_fmt("the host pushed one page into each input CB", rc == FT_TT_OK,
                         "tiles_received[%u]@0x%08x = 1", FT_TT_PHASE2_CB_IN0,
                         FT_TT_CB_TILES_RECEIVED_ADDR(FT_TT_PHASE2_CB_IN0));
    if (rc != FT_TT_OK)
        return;

    rc = launch_and_wait(noc, x, y, &config, RUN_TIMEOUT_MS);
    ft_tt_test_check_fmt("compute triad completed", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    /* Informational, not a check. The output CB's counter reads 0 here and that is
     * correct: BRISC asks TRISC0 to re-zero all 64 counter pairs as part of finishing a
     * run, and it does so before the host can observe the packer's push.
     * The evidence that the packer really wrote is the output FIFO itself, which was
     * wiped to zeros before the launch and is compared below. */
    rc = noc->read(noc->user, x, y, FT_TT_CB_TILES_RECEIVED_ADDR(FT_TT_PHASE2_CB_OUT),
                   &received_after, sizeof(received_after));
    printf("        output CB counter after completion: tiles_received[%u] = %u (rc=%d) — "
           "TRISC0 re-zeroes every counter at end of run, so 0 is expected\n",
           FT_TT_PHASE2_CB_OUT, received_after & 0xFFFFu, rc);

    rc = noc->read(noc->user, x, y, g_layout.cb_out_address, tile_readback,
                   sizeof(tile_readback));
    if (rc != FT_TT_OK) {
        ft_tt_test_check("output tile read back", false, ft_tt_error_string(rc));
        return;
    }
    difference = first_difference(tile_readback, tile_expected);
    if (difference != TILE_ELEMENTS) {
        /* WHICH elements are wrong, and how many, is the whole diagnosis: a handful of
         * stray words is a different fault from half a tile, and step 11 has no NoC
         * traffic at all, so anything wrong here is the compute. */
        unsigned index;
        unsigned differing = 0;
        unsigned shown = 0;

        for (index = 0; index < TILE_ELEMENTS; index++) {
            if (tile_readback[index] == tile_expected[index])
                continue;
            differing++;
            if (shown < 24u) {
                printf("%s%u(%04x!=%04x)", shown == 0u ? "        differing: " : " ", index,
                       tile_readback[index], tile_expected[index]);
                shown++;
            }
        }
        printf("%s%u of %u differ\n", differing > 24u ? " ... " : "  — ", differing,
               TILE_ELEMENTS);
    }
    ft_tt_test_check_fmt("output tile is byte-exact against the C reference",
                         difference == TILE_ELEMENTS,
                         "%u of %u bfloat16 elements matched before the first difference",
                         difference, TILE_ELEMENTS);
    if (difference != TILE_ELEMENTS)
        printf("        element %u: in0=0x%04x in1=0x%04x got=0x%04x expected=0x%04x\n",
               difference, tile_in0[difference], tile_in1[difference], tile_readback[difference],
               tile_expected[difference]);
}

/* ------------------------------------------------------------------------- */
/* Step 12 — full triad                                                      */
/* ------------------------------------------------------------------------- */

/** Fill both source buffers and the reference for a multi-tile run. */
static int stage_source_buffers(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    unsigned tile;

    for (tile = 0; tile < FULL_TRIAD_TILE_COUNT; tile++) {
        uint32_t offset = tile * FT_TT_PHASE2_TILE_BYTES;
        int rc;

        build_reference_tiles(tile);
        rc = noc->write(noc->user, x, y, g_layout.source0_address + offset, tile_in0,
                        sizeof(tile_in0));
        if (rc != FT_TT_OK)
            return rc;
        rc = noc->write(noc->user, x, y, g_layout.source1_address + offset, tile_in1,
                        sizeof(tile_in1));
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/** Wipe the destination, so a repeat cannot pass on the previous run's answer. */
static int wipe_destination(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    unsigned tile;

    memset(tile_readback, 0, sizeof(tile_readback));
    for (tile = 0; tile < FULL_TRIAD_TILE_COUNT; tile++) {
        int rc = noc->write(noc->user, x, y,
                            g_layout.destination_address + tile * FT_TT_PHASE2_TILE_BYTES,
                            tile_readback, sizeof(tile_readback));

        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/** Compare every destination tile against the reference. Returns FT_TT_OK on a match. */
static int check_destination(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                             unsigned *out_bad_tile, unsigned *out_bad_element)
{
    unsigned tile;

    for (tile = 0; tile < FULL_TRIAD_TILE_COUNT; tile++) {
        unsigned difference;
        int rc = noc->read(noc->user, x, y,
                           g_layout.destination_address + tile * FT_TT_PHASE2_TILE_BYTES,
                           tile_readback, sizeof(tile_readback));

        if (rc != FT_TT_OK)
            return rc;
        build_reference_tiles(tile);
        difference = first_difference(tile_readback, tile_expected);
        if (difference != TILE_ELEMENTS) {
            *out_bad_tile = tile;
            *out_bad_element = difference;
            return FT_TT_ERROR;
        }
    }
    *out_bad_tile = FULL_TRIAD_TILE_COUNT;
    *out_bad_element = TILE_ELEMENTS;
    return FT_TT_OK;
}

static uint64_t monotonic_milliseconds(void)
{
    struct timespec now;

    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000u + (uint64_t)(now.tv_nsec / 1000000L);
}

static void run_step_12_full_triad(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                   const ft_tt_image_placement placements[FT_TT_PROC_COUNT])
{
    ft_tt_launch_config config;
    uint64_t cb_mask = 0;
    uint32_t reader_arguments[7];
    uint32_t writer_arguments[4];
    uint32_t tile_count = FULL_TRIAD_TILE_COUNT;
    uint64_t started;
    uint64_t total_ms;
    unsigned bad_tile = 0;
    unsigned bad_element = 0;
    unsigned completed = 0;
    unsigned processor;
    int rc;

    ft_tt_test_step("step 12 — full triad: reader on BRISC (NoC 0), writer on NCRISC "
                    "(NoC 1), both self-addressed to this core");

    rc = write_cb_config(noc, x, y, &cb_mask);
    ft_tt_test_check_fmt("CB config written", rc == FT_TT_OK, "mask=0x%llx",
                         (unsigned long long)cb_mask);
    if (rc != FT_TT_OK)
        return;

    build_config(placements, 0x1fu, cb_mask, &config);
    ft_tt_test_check_fmt("enables covers all five processors", config.enables == 0x1fu, "0x%02x",
                         config.enables);
    ft_tt_test_check_fmt("brisc_noc_id matches the NOC the reader was built for",
                         config.brisc_noc_id == 0u, "%u", config.brisc_noc_id);

    reader_arguments[0] = tile_count;
    reader_arguments[1] = x;
    reader_arguments[2] = y;
    reader_arguments[3] = g_layout.source0_address;
    reader_arguments[4] = x;
    reader_arguments[5] = y;
    reader_arguments[6] = g_layout.source1_address;

    writer_arguments[0] = tile_count;
    writer_arguments[1] = x;
    writer_arguments[2] = y;
    writer_arguments[3] = g_layout.destination_address;

    rc = ft_tt_launch_plant_runtime_args(noc, x, y, config.runtime_args_address[FT_TT_PROC_DM0],
                                         reader_arguments, 7u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_plant_runtime_args(noc, x, y,
                                             config.runtime_args_address[FT_TT_PROC_DM1],
                                             writer_arguments, 4u);
    for (processor = FT_TT_PROC_MATH0; processor < FT_TT_PROC_COUNT && rc == FT_TT_OK; processor++)
        rc = ft_tt_launch_plant_runtime_args(noc, x, y, config.runtime_args_address[processor],
                                             &tile_count, 1u);
    ft_tt_test_check_fmt("runtime args planted", rc == FT_TT_OK,
                         "reader src (%u,%u) 0x%05x / 0x%05x, writer dst (%u,%u) 0x%05x", x, y,
                         g_layout.source0_address, g_layout.source1_address, x, y,
                         g_layout.destination_address);
    if (rc != FT_TT_OK)
        return;

    rc = stage_source_buffers(noc, x, y);
    if (rc == FT_TT_OK)
        rc = wipe_destination(noc, x, y);
    ft_tt_test_check_fmt("source buffers staged, destination wiped", rc == FT_TT_OK, "%u tiles",
                         FULL_TRIAD_TILE_COUNT);
    if (rc != FT_TT_OK)
        return;

    rc = launch_and_wait(noc, x, y, &config, RUN_TIMEOUT_MS);
    ft_tt_test_check_fmt("full triad completed", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    rc = check_destination(noc, x, y, &bad_tile, &bad_element);
    ft_tt_test_check_fmt("every destination tile is byte-exact against the C reference",
                         rc == FT_TT_OK,
                         "%u tiles x %u elements; matched up to tile %u element %u",
                         FULL_TRIAD_TILE_COUNT, TILE_ELEMENTS, bad_tile, bad_element);
    if (rc != FT_TT_OK)
        return;

    /* Acceptance: repeatable. Each repeat wipes the destination first, so a run that
     * silently did nothing shows up as zeros rather than as the previous answer. */
    ft_tt_test_step("step 12 — %u repeats, destination wiped and checked every time",
                    FULL_TRIAD_REPEAT_COUNT);
    started = monotonic_milliseconds();
    while (completed < FULL_TRIAD_REPEAT_COUNT) {
        rc = wipe_destination(noc, x, y);
        if (rc != FT_TT_OK)
            break;
        rc = launch_and_wait(noc, x, y, &config, RUN_TIMEOUT_MS);
        if (rc != FT_TT_OK)
            break;
        rc = check_destination(noc, x, y, &bad_tile, &bad_element);
        if (rc != FT_TT_OK)
            break;
        completed++;
    }
    total_ms = monotonic_milliseconds() - started;
    ft_tt_test_check_fmt("every repeat was byte-exact", completed == FULL_TRIAD_REPEAT_COUNT,
                         "%u/%u in %llu ms (%.2f ms each), last rc=%d", completed,
                         FULL_TRIAD_REPEAT_COUNT, (unsigned long long)total_ms,
                         completed ? (double)total_ms / (double)completed : 0.0, rc);
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    uint16_t x = 0;
    uint16_t y = 0;
    int rc;

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;
    rc = ft_tt_hardware_session_tensix_from_environment(&session, &x, &y, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &firmware);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    ft_tt_test_step("setup — boot the firmware and load the vector_add triad");
    rc = boot_and_derive_layout(&session.noc, x, y, &firmware);
    ft_tt_test_check_fmt("firmware booted", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    rc = ft_tt_core_load_program(&session.noc, x, y, &ft_tt_kernel_vector_add, placements);
    ft_tt_test_check_fmt("all five vector_add images loaded", rc == FT_TT_OK, "%u images",
                         ft_tt_kernel_vector_add.image_count);
    if (rc != FT_TT_OK)
        goto done;

    run_step_11_host_staged(&session.noc, x, y, placements);
    if (ft_tt_test_failure_count() != 0)
        goto done;

    run_step_12_full_triad(&session.noc, x, y, placements);

done:
    (void)ft_tt_core_park(&session.noc, x, y);
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
