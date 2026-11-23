/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_phase2_multicore.c
 * @brief Checklist step 13 from docs/tt_launch_notes.md §6, and plan Phase 2's second
 *   acceptance: "8 cores, each writing a distinct value derived from its own args" — plus
 *   the half that acceptance did not ask for, which is that every core in the group
 *   COMPUTES correctly, not merely that it completed.
 *
 * Eight Tensix cores at DISCONTIGUOUS logical indices, so a bug that walks a contiguous
 * run cannot pass. Two stages over the same placement group:
 *
 *   A  RUNTIME-ARG PROBE. Each core gets its own (address, value) pair and stores it. The
 *      values encode both the core and the processor that wrote them, so a core writing
 *      another core's value is visible. This is the original acceptance and it checks the
 *      per-core ARGUMENT plumbing.
 *
 *   B  VECTOR ADD, FULL TRIAD. The same group runs a real op — reader on BRISC, compute
 *      on all three TRISCs, writer on NCRISC — on OPERANDS DERIVED FROM EACH CORE'S OWN
 *      INDEX, and every element of every output tile on every core is compared byte-exact
 *      against a C reference. Stage A cannot see a compute fault: `runtime_arg_probe`
 *      stores a planted word and does no matrix math, so a core whose unpack/math/pack
 *      path is corrupt passes it. The fault class that cost 2026-08-12 corrupts compute
 *      while leaving stores intact, and a multicore gate blind to it is barely a gate.
 *
 * Both stages stage the WHOLE group first and kick it second, then wait — otherwise this
 * is eight sequential single-core runs wearing a loop, which is what step 7 already
 * proved.
 *
 * Three properties beyond "every core finished":
 *   - operands are seeded from the CORE INDEX, not the slot, so a core reading another
 *     core's L1 produces a wrong answer rather than a coincidentally right one;
 *   - the eight results are checked PAIRWISE DISTINCT, which catches a whole group
 *     collapsing onto one core's data even if the reference were wrong;
 *   - each core's DERIVED L1 layout is checked identical to slot 0's, because every
 *     address in stage B is that layout and "it worked on one core" is how this project
 *     has been wrong before (dev notes §3).
 *
 * The placement loop itself is not here. It is ft_tt_launch's core-set API
 * (`ft_tt_launch_core_set_*`, `_write_launch_messages`, `_send_go_all`, `_wait_done_all`,
 * `_park_all`), and this file is a CALLER of it — a backend that can only launch one core
 * at a time is not a backend, it is a demo.
 *
 * FT_TT_TENSIX_INDICES overrides the eight indices (comma-separated). It exists so the
 * SECOND, DISJOINT core set is the same binary rather than an edited source: a result
 * believed on one set of cores does not generalise, and re-compiling to move cores makes
 * the two runs differ by more than the cores.
 *
 * SAFETY. Every coordinate comes from ft_tt_coord against live telemetry and is then
 * classified before use — inside ft_tt_launch_core_set_add_tensix(), so a coordinate that
 * does not classify as a Tensix cannot reach a launch at all (spec §7.3 — a wrong NoC
 * endpoint hangs the chip and costs a physical power cycle). Stage B's reader and writer
 * are pointed at each core's OWN L1, so no core issues a transaction to an endpoint this
 * test derived for another purpose. Every core in the group is parked in reset on every
 * exit path.
 *
 * @status Runs on the card. 82 checks — stage A 56, stage B 26 — green on two disjoint
 *   eight-core sets, 2026-08-12: 1, 3, 17, 38, 59, 84, 111, 139 and
 *   2, 11, 25, 46, 67, 93, 120, 133. Three red proofs, each making exactly ONE core of the
 *   eight wrong, are recorded in the plan's Phase 2 follow-ups.
 * @issues None
 * @todo None
 *
 * Build:
 *   make PHASE1B=1 build/test_phase2_multicore
 */

#include "ft_tt_hardware_session.h"
#include "ft_tt_kernels.h"
#include "ft_tt_phase2_l1_layout.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CORE_COUNT       8u
#define RUN_TIMEOUT_MS   5000u

/** Environment override for the eight indices, comma-separated. Exactly CORE_COUNT
 *  values or the run is refused — a short list would quietly weaken the acceptance. */
#define CORE_INDICES_ENVIRONMENT "FT_TT_TENSIX_INDICES"

/** Tiles stage B moves per core. Two, so the two-page CB ring wraps exactly once — a CB
 *  that is really a single slot passes a one-tile test. */
#define GROUP_TILE_COUNT 2u

/** Group launches stage B repeats, each with every destination wiped first. Enough to
 *  catch a result that only survives the first run; not so many that the read-back
 *  traffic (32 KiB per pass at ~4.7 MB/s card→host) dominates the run. */
#define GROUP_REPEAT_COUNT 10u

#define TILE_ELEMENTS    (FT_TT_PHASE2_TILE_BYTES / 2u)
#define CB_CONFIG_BYTES  (FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4u)

/** Deliberately not 0..7: a contiguous set would hide an off-by-one in the coordinate
 *  walk, and row-major index i means (column i % 14, row i / 14) — so these span rows
 *  0, 1, 2, 4, 5, 7 and 9 as well as columns.
 *
 *  Index 0 is deliberately NOT in the set. It is the single core the kernel-toolchain
 *  bring-up work uses, and two agents booting firmware on one Tensix at the same time
 *  produces a failure neither of them can attribute. */
static uint32_t g_core_indices[CORE_COUNT] = {1u, 3u, 17u, 38u, 59u, 84u, 111u, 139u};

/** The L1 map, derived from where the firmware images landed rather than chosen. A
 *  file-scope object because every helper below needs it and threading one more
 *  parameter through all of them would say less than this comment does. Written by
 *  slot 0's boot; every later core's derivation is checked against it. */
static ft_tt_phase2_l1_layout g_layout;

/** Caller-owned storage; static rather than on main()'s stack. The capacity is this
 *  test's own CORE_COUNT, not a part size — ft_tt_launch.h stopped carrying one in
 *  Phase 5, precisely so a group is sized by whoever knows how big it needs to be. */
static ft_tt_launch_core_placement core_storage[CORE_COUNT];
static ft_tt_launch_core_set core_set;

static uint16_t tile_in0[TILE_ELEMENTS];
static uint16_t tile_in1[TILE_ELEMENTS];
static uint16_t tile_readback[TILE_ELEMENTS];

/** Per-core reference results, and the first tile each core actually returned — kept so
 *  the eight answers can be compared to each OTHER as well as to the reference. */
static uint16_t g_expected[CORE_COUNT][GROUP_TILE_COUNT][TILE_ELEMENTS];
static uint16_t g_observed_first_tile[CORE_COUNT][TILE_ELEMENTS];

/* ------------------------------------------------------------------------- */
/* Setup                                                                     */
/* ------------------------------------------------------------------------- */

/**
 * Replace the default core set from CORE_INDICES_ENVIRONMENT, if it is set.
 *
 * @return true when the set to use is valid (whether default or overridden)
 */
static bool load_core_indices_from_environment(void)
{
    const char *text = getenv(CORE_INDICES_ENVIRONMENT);
    uint32_t parsed[CORE_COUNT];
    unsigned count = 0;
    const char *cursor;

    if (!text || text[0] == '\0')
        return true;

    for (cursor = text; *cursor != '\0'; ) {
        char *end = NULL;
        unsigned long value = strtoul(cursor, &end, 10);

        /* Syntax and count only. The live bound is ft_tt_coord_tensix_by_index()'s. */
        if (end == cursor || value > UINT32_MAX || count >= CORE_COUNT) {
            ft_tt_test_check_fmt(CORE_INDICES_ENVIRONMENT " parses", false, "at \"%s\"", cursor);
            return false;
        }
        parsed[count++] = (uint32_t)value;
        cursor = (*end == ',') ? end + 1 : end;
    }
    if (count != CORE_COUNT) {
        ft_tt_test_check_fmt(CORE_INDICES_ENVIRONMENT " names the full core set", false,
                             "%u given, %u required", count, CORE_COUNT);
        return false;
    }
    memcpy(g_core_indices, parsed, sizeof(g_core_indices));
    return true;
}

/**
 * Boot the firmware and DERIVE the L1 map from where its images landed.
 *
 * The map's base used to be the literal 0x14000. It is now the higher of "the top of the
 * five firmware images the loader just placed" and "the end of the regions the stock memory map
 * reserves above MEM_MAP_END", which is exactly what ft_tt_image_placement_l1_floor()
 * was written to compute and had no caller for.
 *
 * `slot` 0 writes g_layout; every later slot derives into a local and is checked equal to
 * it, because every stage-B address is this map and one core's derivation is not eight.
 */
static int boot_and_derive_layout(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                  const ft_tt_firmware_set *firmware, unsigned slot)
{
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_phase2_l1_layout derived;
    int rc = ft_tt_core_boot_firmware(noc, x, y, firmware, placements);

    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_phase2_l1_layout_derive(placements, FT_TT_PROC_COUNT, &derived);
    if (rc != FT_TT_OK)
        return rc;
    if (slot == 0u) {
        g_layout = derived;
        return FT_TT_OK;
    }
    ft_tt_test_check_fmt("derived L1 layout matches slot 0's",
                         memcmp(&derived, &g_layout, sizeof(derived)) == 0,
                         "index %3u base 0x%05x vs 0x%05x", g_core_indices[slot],
                         derived.kernel_config_base, g_layout.kernel_config_base);
    return FT_TT_OK;
}

/**
 * Build the placement group. Every coordinate is derived from live telemetry and
 * classified inside the API before it enters the set, and nothing has been written to
 * any core yet.
 *
 * The configs are filled in later, by the staging helpers, because a program's
 * per-processor text addresses are not known until its images have been loaded.
 */
static bool derive_cores(const ft_tt_hardware_session *session)
{
    unsigned slot;
    bool all_good = true;

    ft_tt_test_step("step 13 — derive and classify eight discontiguous Tensix coordinates");
    if (ft_tt_launch_core_set_init(&core_set, core_storage, CORE_COUNT) != FT_TT_OK)
        return false;

    for (slot = 0; slot < CORE_COUNT; slot++) {
        const ft_tt_launch_core_placement *placement;
        int rc = ft_tt_launch_core_set_add_tensix(&core_set, &session->grid, g_core_indices[slot],
                                                  NULL);

        placement = ft_tt_launch_core_set_at(&core_set, slot);
        ft_tt_test_check_fmt("coordinate is a Tensix and entered the group",
                             rc == FT_TT_OK && placement != NULL,
                             "index %3u -> translated (%2u, %2u), rc=%d", g_core_indices[slot],
                             placement ? placement->x : 0u, placement ? placement->y : 0u, rc);
        if (rc != FT_TT_OK || !placement)
            all_good = false;
    }
    return all_good;
}

/** Stage the whole group, kick the whole group, wait for all of it. */
static int launch_group(const ft_tt_noc_access *noc, unsigned *out_failed_slot)
{
    int rc = ft_tt_launch_write_launch_messages(noc, &core_set, out_failed_slot);

    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_send_go_all(noc, &core_set, out_failed_slot);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_wait_done_all(noc, &core_set, RUN_TIMEOUT_MS, out_failed_slot);
}

/* ------------------------------------------------------------------------- */
/* Stage A — per-core runtime args                                           */
/* ------------------------------------------------------------------------- */

/** Value core `slot`'s processor `p` is told to write. Encodes both. */
static uint32_t expected_value(unsigned slot, unsigned processor)
{
    return 0xC0DE0000u | (g_core_indices[slot] << 8) | processor;
}

/** Boot, load and configure one core of the group for the runtime-arg probe. */
static int stage_probe_core(const ft_tt_noc_access *noc, unsigned slot,
                            const ft_tt_firmware_set *firmware)
{
    const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&core_set, slot);
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_launch_config config;
    uint32_t reset_value = 0;
    uint16_t x;
    uint16_t y;
    unsigned processor;
    int rc;

    if (!placement)
        return FT_TT_EINVAL;
    x = placement->x;
    y = placement->y;

    /* A sanity read before any write: every RISC-V of an untouched core is held in
     * reset. A coordinate that answers with something else is not what we think it is. */
    rc = ft_tt_launch_read_reset(noc, x, y, &reset_value);
    if (rc != FT_TT_OK)
        return rc;
    ft_tt_test_check_fmt("soft-reset register reads a plausible value",
                         (reset_value & FT_TT_SOFT_RESET_ALL_RISCS) != 0u, "(%2u, %2u) 0x%08x", x,
                         y, reset_value);

    rc = boot_and_derive_layout(noc, x, y, firmware, slot);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_core_load_program(noc, x, y, &ft_tt_kernel_runtime_arg_probe, placements);
    if (rc != FT_TT_OK)
        return rc;

    memset(&config, 0, sizeof(config));
    config.kernel_config_base = g_layout.kernel_config_base;
    config.enables = ft_tt_kernel_program_enables(&ft_tt_kernel_runtime_arg_probe);
    config.brisc_noc_id = 0u;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        uint32_t arguments[2];
        uint32_t zero = 0;

        if (placements[processor].bytes_written == 0u)
            continue;
        config.kernel_text_address[processor] = placements[processor].entry_address;
        config.kernel_text_bytes[processor] = placements[processor].text_bytes;
        config.runtime_args_address[processor] = ft_tt_phase2_runtime_args_address(&g_layout, processor);

        arguments[0] = g_layout.probe_address + processor * FT_TT_PHASE2_PROBE_STRIDE;
        arguments[1] = expected_value(slot, processor);
        rc = noc->write(noc->user, x, y, arguments[0], &zero, sizeof(zero));
        if (rc != FT_TT_OK)
            return rc;
        rc = ft_tt_launch_plant_runtime_args(noc, x, y, config.runtime_args_address[processor],
                                             arguments, 2u);
        if (rc != FT_TT_OK)
            return rc;
    }
    return ft_tt_launch_core_set_configure(&core_set, slot, &config);
}

/** Every core wrote the value derived from its OWN args, at its own address. */
static void verify_probe_results(const ft_tt_noc_access *noc)
{
    unsigned slot;
    unsigned processor;

    ft_tt_test_step("stage A — every core wrote the value derived from its own args");
    for (slot = 0; slot < CORE_COUNT; slot++) {
        const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&core_set, slot);

        if (!placement)
            continue;
        for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
            uint32_t address = g_layout.probe_address + processor * FT_TT_PHASE2_PROBE_STRIDE;
            uint32_t readback = 0;
            int rc;

            if (placement->config.runtime_args_address[processor] == 0u)
                continue;
            rc = noc->read(noc->user, placement->x, placement->y, address, &readback,
                           sizeof(readback));
            ft_tt_test_check_fmt("value came back",
                                 rc == FT_TT_OK && readback == expected_value(slot, processor),
                                 "index %3u %-6s 0x%08x, expected 0x%08x", g_core_indices[slot],
                                 ft_tt_processor_name(processor), readback,
                                 expected_value(slot, processor));
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Stage B — a real op on every core, checked byte-exact on every core        */
/* ------------------------------------------------------------------------- */

/** bfloat16 is the top 16 bits of the IEEE-754 binary32 pattern, truncated. Every value
 *  this test uses is a small integer, so truncation is exact and so is the sum. */
static uint16_t bfloat16_from_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

/**
 * Build one core's operand tiles and its reference result.
 *
 * The seed is the CORE INDEX, not the slot: a core that reads another core's L1, or a
 * group that collapses onto one core's operands, then produces a result that matches no
 * reference rather than one that happens to be right.
 *
 * THE SEED IS AN OFFSET, NOT A MULTIPLIER INSIDE THE MODULUS, and that is the whole
 * design of this function. The first version used `(element + seed * 7) % 16`, which is
 * identical for every pair of indices congruent mod 16 — so cores 1 and 17, and 59 and
 * 139, were handed byte-identical operands while the test claimed they were per-core.
 * Every core still passed its own reference; the pairwise-distinctness check is what
 * saw it. As an offset the seed shifts every element by a different amount, so distinct
 * indices give distinct tiles by construction, for any set of indices.
 *
 * Element ORDER does not have to match the Tensix's 4-face tile layout — an elementwise
 * add preserves position under any permutation applied to both operands. What matters is
 * that every sum lands exactly on a bfloat16: the largest value here is
 * 16 + 1 + 139 + 16 + 1 = 173, and bfloat16's 8-bit significand represents every integer
 * up to 256 exactly, so "byte-exact" stays a fair demand rather than a rounding lottery.
 */
static void build_core_tiles(unsigned slot, unsigned tile)
{
    unsigned seed = g_core_indices[slot];
    unsigned element;

    for (element = 0; element < TILE_ELEMENTS; element++) {
        float left = (float)((element + tile) % 16u + 1u + seed);
        float right = (float)((element * 3u + tile) % 16u + 1u);

        tile_in0[element] = bfloat16_from_float(left);
        tile_in1[element] = bfloat16_from_float(right);
        g_expected[slot][tile][element] = bfloat16_from_float(left + right);
    }
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

/** Write the CB config array into one core's L1 and hand back the launch-message mask. */
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

/** Plant this core's operand tiles in its own L1 and build the matching reference. */
static int stage_core_operands(const ft_tt_noc_access *noc, unsigned slot, uint16_t x, uint16_t y)
{
    unsigned tile;

    for (tile = 0; tile < GROUP_TILE_COUNT; tile++) {
        uint32_t offset = tile * FT_TT_PHASE2_TILE_BYTES;
        int rc;

        build_core_tiles(slot, tile);
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

/** Wipe one core's destination, so a repeat cannot pass on the previous run's answer and
 *  a run that silently did nothing shows up as zeros. */
static int wipe_core_destination(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    static const uint16_t zeros[TILE_ELEMENTS];
    unsigned tile;

    for (tile = 0; tile < GROUP_TILE_COUNT; tile++) {
        int rc = noc->write(noc->user, x, y,
                            g_layout.destination_address + tile * FT_TT_PHASE2_TILE_BYTES, zeros,
                            sizeof(zeros));

        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/** Wipe every core's destination. */
static int wipe_group_destinations(const ft_tt_noc_access *noc)
{
    unsigned slot;

    for (slot = 0; slot < CORE_COUNT; slot++) {
        const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&core_set, slot);
        int rc;

        if (!placement)
            return FT_TT_EINVAL;
        rc = wipe_core_destination(noc, placement->x, placement->y);
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/** Boot, load and configure one core of the group for the vector_add triad. */
static int stage_vector_add_core(const ft_tt_noc_access *noc, unsigned slot,
                                 const ft_tt_firmware_set *firmware)
{
    const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&core_set, slot);
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_launch_config config;
    uint32_t reader_arguments[7];
    uint32_t writer_arguments[4];
    uint32_t tile_count = GROUP_TILE_COUNT;
    uint64_t cb_mask = 0;
    uint16_t x;
    uint16_t y;
    unsigned processor;
    int rc;

    if (!placement)
        return FT_TT_EINVAL;
    x = placement->x;
    y = placement->y;

    rc = boot_and_derive_layout(noc, x, y, firmware, slot);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_core_load_program(noc, x, y, &ft_tt_kernel_vector_add, placements);
    if (rc != FT_TT_OK)
        return rc;
    rc = write_cb_config(noc, x, y, &cb_mask);
    if (rc != FT_TT_OK)
        return rc;

    memset(&config, 0, sizeof(config));
    config.kernel_config_base = g_layout.kernel_config_base;
    config.enables = ft_tt_kernel_program_enables(&ft_tt_kernel_vector_add);
    config.local_cb_config_address = g_layout.cb_config_address;
    config.local_cb_mask = cb_mask;
    /* vector_add_reader.c is compiled with NOC_INDEX=0 and the firmware re-reads this
     * at run time. The two MUST agree. */
    config.brisc_noc_id = 0u;
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if ((config.enables & (1u << processor)) == 0u)
            continue;
        config.kernel_text_address[processor] = placements[processor].entry_address;
        config.kernel_text_bytes[processor] = placements[processor].text_bytes;
        config.runtime_args_address[processor] = ft_tt_phase2_runtime_args_address(&g_layout, processor);
    }

    /* Both kernels address THIS core's own L1 — the one endpoint each core has already
     * been read and written at (spec §7.3). */
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
    if (rc != FT_TT_OK)
        return rc;

    rc = stage_core_operands(noc, slot, x, y);
    if (rc != FT_TT_OK)
        return rc;
    rc = wipe_core_destination(noc, x, y);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_core_set_configure(&core_set, slot, &config);
}

/** Where a core's result first went wrong, and by how much. */
typedef struct core_result_diff {
    unsigned bad_tile;    /**< first tile that differed, or GROUP_TILE_COUNT */
    unsigned bad_element; /**< first differing element in it, or TILE_ELEMENTS */
    unsigned differing;   /**< total differing elements across all tiles */
    uint16_t observed;    /**< what that element actually held */
    uint16_t expected;    /**< what the C reference says it should have held */
} core_result_diff;

/**
 * Compare one core's destination against its own reference, element by element.
 *
 * @param out_diff  always written, even on a match
 * @return          FT_TT_OK on a byte-exact match, FT_TT_ERROR on a mismatch, or the
 *                  read's error code
 */
static int compare_core_result(const ft_tt_noc_access *noc, unsigned slot,
                               core_result_diff *out_diff)
{
    const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&core_set, slot);
    unsigned tile;

    memset(out_diff, 0, sizeof(*out_diff));
    out_diff->bad_tile = GROUP_TILE_COUNT;
    out_diff->bad_element = TILE_ELEMENTS;
    if (!placement)
        return FT_TT_EINVAL;

    for (tile = 0; tile < GROUP_TILE_COUNT; tile++) {
        unsigned element;
        int rc = noc->read(noc->user, placement->x, placement->y,
                           g_layout.destination_address + tile * FT_TT_PHASE2_TILE_BYTES,
                           tile_readback, sizeof(tile_readback));

        if (rc != FT_TT_OK)
            return rc;
        if (tile == 0u)
            memcpy(g_observed_first_tile[slot], tile_readback, sizeof(tile_readback));
        for (element = 0; element < TILE_ELEMENTS; element++) {
            if (tile_readback[element] == g_expected[slot][tile][element])
                continue;
            out_diff->differing++;
            if (out_diff->bad_tile != GROUP_TILE_COUNT)
                continue;
            out_diff->bad_tile = tile;
            out_diff->bad_element = element;
            out_diff->observed = tile_readback[element];
            out_diff->expected = g_expected[slot][tile][element];
        }
    }
    return (out_diff->differing == 0u) ? FT_TT_OK : FT_TT_ERROR;
}

/** One check per core: every element of every output tile, byte-exact against a C
 *  reference built from that core's own index. */
static void verify_group_results(const ft_tt_noc_access *noc)
{
    unsigned slot;

    ft_tt_test_step("stage B — every core's vector add is byte-exact against a C reference");
    for (slot = 0; slot < CORE_COUNT; slot++) {
        core_result_diff diff;
        int rc = compare_core_result(noc, slot, &diff);

        ft_tt_test_check_fmt("every output element matches", rc == FT_TT_OK,
                             "index %3u, %u tiles x %u elements, %u differ", g_core_indices[slot],
                             GROUP_TILE_COUNT, TILE_ELEMENTS, diff.differing);
        if (rc == FT_TT_ERROR)
            printf("        index %3u tile %u element %u: got 0x%04x expected 0x%04x\n",
                   g_core_indices[slot], diff.bad_tile, diff.bad_element, diff.observed,
                   diff.expected);
    }
}

/**
 * The eight answers are pairwise distinct.
 *
 * Every core was given operands seeded from its own index, so equal results mean the
 * group collapsed onto one core's data — a class of bug the per-core reference check
 * would also catch, but this states it without depending on the reference being right.
 */
static void verify_results_are_distinct(void)
{
    unsigned left;
    unsigned collisions = 0;
    unsigned first_left = 0;
    unsigned first_right = 0;

    ft_tt_test_step("stage B — the eight results are pairwise distinct");
    for (left = 0; left < CORE_COUNT; left++) {
        unsigned right;

        for (right = left + 1u; right < CORE_COUNT; right++) {
            if (memcmp(g_observed_first_tile[left], g_observed_first_tile[right],
                       sizeof(g_observed_first_tile[left])) != 0)
                continue;
            if (collisions == 0u) {
                first_left = g_core_indices[left];
                first_right = g_core_indices[right];
            }
            collisions++;
        }
    }
    ft_tt_test_check_fmt("no two cores returned the same tile", collisions == 0u,
                         "%u colliding pairs of %u (first: index %u == index %u)", collisions,
                         CORE_COUNT * (CORE_COUNT - 1u) / 2u, first_left, first_right);
}

/** Re-launch the group GROUP_REPEAT_COUNT times, wiping and re-checking every core each
 *  time. A result that only survives the first run is a different bug from a wrong one. */
static void repeat_group_runs(const ft_tt_noc_access *noc)
{
    core_result_diff diff;
    unsigned completed = 0;
    unsigned failed_slot = 0;
    unsigned bad_slot = CORE_COUNT;
    int rc = FT_TT_OK;

    memset(&diff, 0, sizeof(diff));
    ft_tt_test_step("stage B — %u more group launches, every destination wiped and every core "
                    "re-checked each time", GROUP_REPEAT_COUNT);
    while (completed < GROUP_REPEAT_COUNT) {
        unsigned slot;

        rc = wipe_group_destinations(noc);
        if (rc != FT_TT_OK)
            break;
        rc = launch_group(noc, &failed_slot);
        if (rc != FT_TT_OK)
            break;
        for (slot = 0; slot < CORE_COUNT && rc == FT_TT_OK; slot++) {
            rc = compare_core_result(noc, slot, &diff);
            if (rc != FT_TT_OK)
                bad_slot = slot;
        }
        if (rc != FT_TT_OK)
            break;
        completed++;
    }
    ft_tt_test_check_fmt("every core was byte-exact in every repeat",
                         completed == GROUP_REPEAT_COUNT, "%u/%u repeats, last rc=%d, slot %u "
                         "(index %u), %u elements differ", completed, GROUP_REPEAT_COUNT, rc,
                         bad_slot, bad_slot < CORE_COUNT ? g_core_indices[bad_slot] : 0u,
                         diff.differing);
}

/* ------------------------------------------------------------------------- */

/** Stage A: the same kernel on eight discontiguous cores with per-core arguments. */
static bool run_stage_a(ft_tt_hardware_session *session, const ft_tt_firmware_set *firmware)
{
    unsigned failed_slot = 0;
    unsigned slot;
    int rc;

    ft_tt_test_step("stage A — boot, load and configure all eight before staging any");
    for (slot = 0; slot < CORE_COUNT; slot++) {
        const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&core_set, slot);

        rc = stage_probe_core(&session->noc, slot, firmware);
        ft_tt_test_check_fmt("core staged", rc == FT_TT_OK, "index %3u at (%2u, %2u), rc=%d",
                             g_core_indices[slot], placement ? placement->x : 0u,
                             placement ? placement->y : 0u, rc);
        if (rc != FT_TT_OK)
            return false;
    }

    ft_tt_test_step("stage A — stage the whole group, kick the whole group, wait for all of it");
    rc = launch_group(&session->noc, &failed_slot);
    ft_tt_test_check_fmt("every core completed", rc == FT_TT_OK, "rc=%d, slot %u (index %u)", rc,
                         failed_slot,
                         failed_slot < CORE_COUNT ? g_core_indices[failed_slot] : 0u);
    if (rc != FT_TT_OK)
        return false;

    verify_probe_results(&session->noc);
    return true;
}

/** Stage B: a real op — the vector_add triad — on every core of the same group. */
static bool run_stage_b(ft_tt_hardware_session *session, const ft_tt_firmware_set *firmware)
{
    unsigned failed_slot = 0;
    unsigned slot;
    int rc;

    ft_tt_test_step("stage B — reboot the group onto the vector_add triad, per-core operands "
                    "seeded from each core's own index");
    for (slot = 0; slot < CORE_COUNT; slot++) {
        const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&core_set, slot);

        rc = stage_vector_add_core(&session->noc, slot, firmware);
        ft_tt_test_check_fmt("core staged with its own operands", rc == FT_TT_OK,
                             "index %3u at (%2u, %2u), rc=%d", g_core_indices[slot],
                             placement ? placement->x : 0u, placement ? placement->y : 0u, rc);
        if (rc != FT_TT_OK)
            return false;
    }

    ft_tt_test_step("stage B — stage the whole group, kick the whole group, wait for all of it");
    rc = launch_group(&session->noc, &failed_slot);
    ft_tt_test_check_fmt("every core completed", rc == FT_TT_OK, "rc=%d, slot %u (index %u)", rc,
                         failed_slot,
                         failed_slot < CORE_COUNT ? g_core_indices[failed_slot] : 0u);
    if (rc != FT_TT_OK)
        return false;

    verify_group_results(&session->noc);
    verify_results_are_distinct();
    repeat_group_runs(&session->noc);
    return true;
}

int main(void)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    unsigned slot;
    int rc;

    if (!load_core_indices_from_environment())
        return ft_tt_test_finish();

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;

    /* Claim all eight BEFORE reading firmware off disk: a refusal must cost
     * nothing, and it names the process holding the core rather than timing out.
     * The claim is PER-CORE, so a suite working the other 132 is unaffected —
     * which is the whole reason this does not take the whole-card lock. */
    rc = ft_tt_hardware_session_claim_cores(&session, g_core_indices, CORE_COUNT,
                                            "phase2 multicore group", NULL, NULL);
    ft_tt_test_check_fmt("all eight cores are claimed for this run", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return ft_tt_test_finish();
    }

    rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &firmware);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    printf("%u Tensix cores, %u enabled on this card — indices", CORE_COUNT,
           session.grid.tensix_core_count);
    for (slot = 0; slot < CORE_COUNT; slot++)
        printf(" %u", g_core_indices[slot]);
    printf("\n");

    if (!derive_cores(&session))
        goto done;
    if (!run_stage_a(&session, &firmware))
        goto done;
    (void)run_stage_b(&session, &firmware);

done:
    (void)ft_tt_launch_park_all(&session.noc, &core_set);
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
