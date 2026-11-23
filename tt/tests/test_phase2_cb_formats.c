/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_phase2_cb_formats.c — a SECOND tile format, end to end on the card.
 *
 * Plan Phase 2 follow-up 5. Until 2026-08-12 build_tensix_kernel.sh emitted one hardcoded
 * CB descriptor set — Float16_b, 32x32, 4 faces, 2048 B — so exactly one tile shape had
 * ever been compiled, and a kernel with any other page size or format got descriptors
 * describing a tile it does not have. That failure is silent: nothing in the build or the
 * launch protocol checks the descriptors against the CB config, because
 * ft_tensix_cb_formats.h IS the contract and there is no runtime negotiation.
 *
 * This runs the same vector_add triad on **Float32** tiles — 4096-byte pages, twice the
 * bytes, a different format code — under two builds that differ in NOTHING except which
 * descriptors were baked in:
 *
 *   vector_add_float32              CB_TABLE=kernels/cb_tables/vector_add_float32.cbtable
 *   vector_add_float32_legacydesc   no CB table, i.e. the old uniform Float16_b set
 *
 * Both are built with the same 4096-byte tile compile-time args, so the data movement is
 * identical and the descriptors are the only variable. Both are booted and run here, in
 * one process, against the same data, and the destinations are compared to each other as
 * BYTES — a pass/fail against a reference could not tell "the descriptors are right" from
 * "the descriptors never mattered".
 *
 * WHAT THE FIRST VERSION OF THIS TEST GOT WRONG, measured on the card 2026-08-12 and
 * worth keeping: with small-integer inputs BOTH builds return a byte-exact result. That
 * is not luck twice, it is arithmetic — a Float32 array whose values are all exactly
 * representable in bfloat16 has zero in every low 16 bits, so it is bit-identical to its
 * own bfloat16 rendering interleaved with zeros, and the wrong descriptors reproduce it
 * exactly. **Integer test data cannot distinguish two descriptor sets on this path.** The
 * fractional pass below is the one that separates them, and the integer pass is kept and
 * reported because the trap is the finding.
 *
 * The correct expectation for Float32 in, with dst_accum_mode off: the source and
 * destination registers hold Float16_b (the build-time generator derives
 * this, nobody chooses it), so every result is a bfloat16 value widened back to Float32,
 * i.e. its low 16 bits are ZERO. That is a rounding-mode-independent assertion, which
 * matters because whether the unpacker truncates or rounds to nearest is documented
 * nowhere we have read.
 *
 * This test uses its OWN L1 layout, not ft_tt_phase2_l1_layout.h's: at 4096-byte pages a
 * two-page CB is 8 KiB and the Phase 2 CBs are 4 KiB apart, so that map would overlap.
 * Only the kernel-config base and the runtime-arg blocks are shared.
 *
 * Usage:  test_phase2_cb_formats <table-driven-directory> <legacy-descriptor-directory>
 *
 * Build:
 *   make -f Makefile.phase2_followups PHASE1B=1 build-phase2b/test_phase2_cb_formats
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_hardware_session.h"
#include "ft_tt_phase2_l1_layout.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>

#define RUN_TIMEOUT_MS 5000u

/** 32x32 Float32. This is the number the whole test is about. */
#define FLOAT32_TILE_BYTES    4096u
#define FLOAT32_TILE_ELEMENTS (FLOAT32_TILE_BYTES / 4u)

/** Same CB indices vector_add is always built with: 0 and 1 in, 16 out. */
#define CB_IN0 0u
#define CB_IN1 1u
#define CB_OUT 16u

/** Two pages, so the ring wraps once over a two-tile run. */
#define CB_PAGE_COUNT 2u
#define CB_BYTES      (CB_PAGE_COUNT * FLOAT32_TILE_BYTES)

#define CB_IN0_ADDRESS 0x00030000u
#define CB_IN1_ADDRESS 0x00032000u
#define CB_OUT_ADDRESS 0x00034000u

#define SOURCE0_ADDRESS     0x00040000u
#define SOURCE1_ADDRESS     0x00044000u
#define DESTINATION_ADDRESS 0x00048000u

#define TILE_COUNT     2u
#define TOTAL_ELEMENTS (TILE_COUNT * FLOAT32_TILE_ELEMENTS)

#define CB_CONFIG_BYTES (FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4u)

/** Which input set a run uses. */
typedef enum input_kind {
    /** Small integers: every value and every sum is exactly a bfloat16. */
    INPUT_BFLOAT16_EXACT = 0,
    /** Values with mantissa bits below the bfloat16 boundary, so the format the
     *  descriptors name changes the answer. */
    INPUT_FRACTIONAL = 1
} input_kind;

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

static float tile_in0[FLOAT32_TILE_ELEMENTS];
static float tile_in1[FLOAT32_TILE_ELEMENTS];
static float tile_expected[FLOAT32_TILE_ELEMENTS];
static float tile_scratch[FLOAT32_TILE_ELEMENTS];

/** Whole destinations, one per program, so the two can be compared byte for byte. */
static uint32_t destination_words[2][TOTAL_ELEMENTS];

/**
 * Build both input tiles and the exact-arithmetic reference for one tile.
 *
 * Element ORDER does not have to match the Tensix's 4-face tile layout: an elementwise
 * add preserves position under any permutation applied to both operands.
 */
static void build_reference_tiles(unsigned tile_index, input_kind inputs)
{
    unsigned element;

    for (element = 0; element < FLOAT32_TILE_ELEMENTS; element++) {
        float left = (float)((element + tile_index) % 16u + 1u);
        float right = (float)((element * 3u + tile_index) % 16u + 1u);

        if (inputs == INPUT_FRACTIONAL) {
            left += (float)(element % 7u) / 128.0f;
            right += (float)(element % 5u) / 512.0f;
        }
        tile_in0[element] = left;
        tile_in1[element] = right;
        tile_expected[element] = left + right;
    }
}

/** Index of the first differing element, or FLOAT32_TILE_ELEMENTS when they match. */
static unsigned first_difference(const float *left, const float *right)
{
    unsigned element;

    for (element = 0; element < FLOAT32_TILE_ELEMENTS; element++) {
        if (memcmp(&left[element], &right[element], sizeof(float)) != 0)
            return element;
    }
    return FLOAT32_TILE_ELEMENTS;
}

/** Write the three CBs' config into L1 and hand back the launch message's mask. */
static int write_cb_config(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint64_t *out_mask)
{
    static const uint32_t addresses[3] = {CB_IN0_ADDRESS, CB_IN1_ADDRESS, CB_OUT_ADDRESS};
    static const unsigned indices[3] = {CB_IN0, CB_IN1, CB_OUT};
    static uint8_t buffer[CB_CONFIG_BYTES];
    ft_tt_launch_cb_entry entries[3];
    unsigned entry;
    int rc;

    for (entry = 0; entry < 3u; entry++) {
        entries[entry].index = indices[entry];
        entries[entry].address = addresses[entry];
        entries[entry].size_bytes = CB_BYTES;
        entries[entry].page_size_bytes = FLOAT32_TILE_BYTES;
        entries[entry].page_count = CB_PAGE_COUNT;
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

/** Plant the reader's, writer's and TRISCs' runtime args for a self-addressed run. */
static int plant_runtime_args(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                              const ft_tt_launch_config *config)
{
    uint32_t reader_arguments[7] = {TILE_COUNT, x, y, SOURCE0_ADDRESS, x, y, SOURCE1_ADDRESS};
    uint32_t writer_arguments[4] = {TILE_COUNT, x, y, DESTINATION_ADDRESS};
    uint32_t tile_count = TILE_COUNT;
    unsigned processor;
    int rc;

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

/** Fill both source buffers in this core's L1 and wipe the destination. */
static int stage_buffers(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, input_kind inputs)
{
    unsigned tile;

    for (tile = 0; tile < TILE_COUNT; tile++) {
        uint32_t offset = tile * FLOAT32_TILE_BYTES;
        int rc;

        build_reference_tiles(tile, inputs);
        rc = noc->write(noc->user, x, y, SOURCE0_ADDRESS + offset, tile_in0, sizeof(tile_in0));
        if (rc == FT_TT_OK)
            rc = noc->write(noc->user, x, y, SOURCE1_ADDRESS + offset, tile_in1, sizeof(tile_in1));
        if (rc == FT_TT_OK) {
            memset(tile_scratch, 0, sizeof(tile_scratch));
            rc = noc->write(noc->user, x, y, DESTINATION_ADDRESS + offset, tile_scratch,
                            sizeof(tile_scratch));
        }
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/**
 * Boot one program directory, run the Float32 vector add over `inputs`, and copy the
 * whole destination out.
 *
 * @param out_words  TOTAL_ELEMENTS words, the destination exactly as the card left it
 */
static int run_one_program(ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                           const char *directory, input_kind inputs, uint32_t *out_words)
{
    ft_tt_firmware_set firmware;
    ft_tt_program_directory program;
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_launch_config config;
    uint64_t cb_mask = 0;
    unsigned tile;
    int rc;

    rc = ft_tt_firmware_set_load(directory, &firmware);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_program_directory_load(directory, 0x1fu, &program);
    if (rc != FT_TT_OK) {
        ft_tt_firmware_set_release(&firmware);
        return rc;
    }

    rc = boot_and_derive_layout(&session->noc, x, y, &firmware);
    if (rc == FT_TT_OK)
        rc = ft_tt_core_load_program(&session->noc, x, y, &program.program, placements);
    if (rc == FT_TT_OK)
        rc = write_cb_config(&session->noc, x, y, &cb_mask);
    if (rc == FT_TT_OK) {
        build_config(placements, cb_mask, &config);
        rc = plant_runtime_args(&session->noc, x, y, &config);
    }
    if (rc == FT_TT_OK)
        rc = stage_buffers(&session->noc, x, y, inputs);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_write_launch_message(&session->noc, x, y, &config);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_send_go(&session->noc, x, y);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_wait_done(&session->noc, x, y, RUN_TIMEOUT_MS, NULL);

    for (tile = 0; tile < TILE_COUNT && rc == FT_TT_OK; tile++)
        rc = session->noc.read(session->noc.user, x, y,
                               DESTINATION_ADDRESS + tile * FLOAT32_TILE_BYTES,
                               &out_words[tile * FLOAT32_TILE_ELEMENTS], FLOAT32_TILE_BYTES);

    /* Park between programs: the next boot would otherwise load firmware over a core that
     * is still running the previous one. */
    (void)ft_tt_core_park(&session->noc, x, y);
    ft_tt_program_directory_release(&program);
    ft_tt_firmware_set_release(&firmware);
    return rc;
}

/** How many destination words are NOT a bfloat16 value widened to Float32. */
static unsigned count_words_with_low_mantissa_bits(const uint32_t *words)
{
    unsigned element;
    unsigned count = 0;

    for (element = 0; element < TOTAL_ELEMENTS; element++) {
        if ((words[element] & 0x0000FFFFu) != 0u)
            count++;
    }
    return count;
}

/** First element that differs from the exact-arithmetic reference, and its tile. */
static unsigned first_reference_difference(const uint32_t *words, input_kind inputs,
                                           unsigned *out_tile)
{
    unsigned tile;

    for (tile = 0; tile < TILE_COUNT; tile++) {
        unsigned difference;

        build_reference_tiles(tile, inputs);
        memcpy(tile_scratch, &words[tile * FLOAT32_TILE_ELEMENTS], sizeof(tile_scratch));
        difference = first_difference(tile_scratch, tile_expected);
        if (difference != FLOAT32_TILE_ELEMENTS) {
            *out_tile = tile;
            return difference;
        }
    }
    *out_tile = TILE_COUNT;
    return FLOAT32_TILE_ELEMENTS;
}

static void report_head(const char *label, const uint32_t *words)
{
    printf("        %-14s destination[0..3] = 0x%08x 0x%08x 0x%08x 0x%08x\n", label, words[0],
           words[1], words[2], words[3]);
}

/* ------------------------------------------------------------------------- */

static void run_integer_pass(ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                             const char *table_directory, const char *legacy_directory)
{
    unsigned bad_tile = 0;
    unsigned difference;
    int rc;

    ft_tt_test_step("Float32 tiles, bfloat16-exact inputs — correctness, and why this pass "
                    "cannot be the demonstration");

    rc = run_one_program(session, x, y, table_directory, INPUT_BFLOAT16_EXACT,
                         destination_words[0]);
    ft_tt_test_check_fmt("the table-driven build completed", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;
    difference = first_reference_difference(destination_words[0], INPUT_BFLOAT16_EXACT, &bad_tile);
    ft_tt_test_check_fmt("every Float32 element is byte-exact against the C reference",
                         difference == FLOAT32_TILE_ELEMENTS,
                         "%u tiles x %u elements; matched up to tile %u element %u", TILE_COUNT,
                         FLOAT32_TILE_ELEMENTS, bad_tile, difference);
    report_head("table-driven", destination_words[0]);

    rc = run_one_program(session, x, y, legacy_directory, INPUT_BFLOAT16_EXACT,
                         destination_words[1]);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("the legacy build completed", false, ft_tt_error_string(rc));
        return;
    }
    report_head("legacy", destination_words[1]);
    printf("        the two destinations are %s on integer data — which is why the fractional\n"
           "        pass exists: a bfloat16-exact Float32 array is bit-identical to its own\n"
           "        bfloat16 rendering, so WRONG descriptors reproduce it exactly\n",
           memcmp(destination_words[0], destination_words[1], sizeof(destination_words[0])) == 0
               ? "IDENTICAL"
               : "different");
}

static void run_fractional_pass(ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                                const char *table_directory, const char *legacy_directory)
{
    unsigned table_low_bits;
    unsigned legacy_low_bits;
    int rc;

    ft_tt_test_step("Float32 tiles, inputs with mantissa bits below the bfloat16 boundary — "
                    "now the descriptors are the only thing that can explain the answer");

    rc = run_one_program(session, x, y, table_directory, INPUT_FRACTIONAL, destination_words[0]);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("the table-driven build completed", false, ft_tt_error_string(rc));
        return;
    }
    rc = run_one_program(session, x, y, legacy_directory, INPUT_FRACTIONAL, destination_words[1]);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("the legacy build completed", false, ft_tt_error_string(rc));
        return;
    }
    report_head("table-driven", destination_words[0]);
    report_head("legacy", destination_words[1]);

    ft_tt_test_check("the two descriptor sets produce DIFFERENT device output",
                     memcmp(destination_words[0], destination_words[1],
                            sizeof(destination_words[0])) != 0,
                     "same kernel source, same CB config, same data — only "
                     "ft_tensix_cb_formats.h differs");

    /* Rounding-mode independent: whatever the unpacker does to the low mantissa bits, a
     * correctly described Float32 CB puts a bfloat16 value back into L1, so the low 16
     * bits of every destination word must be zero. */
    table_low_bits = count_words_with_low_mantissa_bits(destination_words[0]);
    legacy_low_bits = count_words_with_low_mantissa_bits(destination_words[1]);

    ft_tt_test_check_fmt("the table-driven result is a bfloat16 value widened to Float32, "
                         "which is what the derived formats require",
                         table_low_bits == 0u, "%u of %u words carry low mantissa bits",
                         table_low_bits, TOTAL_ELEMENTS);
    ft_tt_test_check_fmt("the legacy descriptors do NOT produce that, so they were really wrong",
                         legacy_low_bits > 0u, "%u of %u words carry low mantissa bits",
                         legacy_low_bits, TOTAL_ELEMENTS);
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    uint16_t x = 0;
    uint16_t y = 0;
    int rc;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <table-driven-directory> <legacy-descriptor-directory>\n"
                        "  both are build_tensix_program.sh outputs for the Float32 vector_add\n",
                argv[0]);
        return 2;
    }

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;
    rc = ft_tt_hardware_session_tensix_from_environment(&session, &x, &y, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    printf("target Tensix 0 at translated (%u, %u)\n  table-driven: %s\n  legacy:       %s\n",
           (unsigned)x, (unsigned)y, argv[1], argv[2]);

    run_integer_pass(&session, x, y, argv[1], argv[2]);
    run_fractional_pass(&session, x, y, argv[1], argv[2]);

    /* run_one_program() parks on every path; do it once more so a failure before the
     * first run still leaves the core in reset. */
    (void)ft_tt_core_park(&session.noc, x, y);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
