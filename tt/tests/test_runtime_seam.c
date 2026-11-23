/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_runtime_seam.c
 * @brief Off-hardware gates for THE TRANSPORT SEAM — the three pieces the
 *   userspace-driver path added that a card cannot be borrowed to prove:
 *   transport selection, architecture selection with no driver opinion, and the
 *   now-SHARED ARC telemetry walk.
 *
 * ---------------------------------------------------------------------------
 * WHY THE TELEMETRY WALK IS THE CENTRE OF THIS FILE
 * ---------------------------------------------------------------------------
 * `ft_tt_arc_telemetry_read_grid_via()` is what BOTH transports now build the
 * grid out of. It used to be reachable only through a TLB window, so nothing
 * off a card could see it; behind an `ft_tt_arc_read32_fn` it can be driven
 * against a synthetic ARC image, and every refusal it makes — an unpublished
 * table, a pointer outside the CSM, a value offset outside the CSM, a corrupt
 * entry count, a missing tag — can be made to happen on purpose.
 *
 * ---------------------------------------------------------------------------
 * THE TAG VALUES ARE SEEDED FROM pid + clock, DELIBERATELY
 * ---------------------------------------------------------------------------
 * A reader that ignored the table and answered from a compiled-in constant, or
 * one that cached a row and returned it for every tag, would pass a test whose
 * expectations were hardcoded. They are not: every value this image publishes
 * is drawn from the process id and the clock, so the ONLY way to produce them
 * is to have read them out of the image. That is the same trap the rehearsal's
 * fixed L1 pattern fell into (`ft_tt_udrv.h` @status), one layer up.
 *
 * @status Off-hardware. Runs anywhere; touches no device and links the backend
 *   WITHOUT FT_TT_PHASE1B_NOC_ENABLED, so the NoC fence is closed and nothing
 *   here could reach a card even by mistake.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_arch.h"
#include "ft_tt_device.h"
#include "ft_tt_runtime.h"
#include "ft_tt_test_report.h"

/** How many 32-bit words of synthetic ARC CSM the image carries. Big enough for
 *  a header, an entry table and a value array with room to spare. */
#define IMAGE_WORDS 256u

/** Which tags the walk wants, in the order it fills the grid. */
#define TAG_COUNT 5u

/* ------------------------------------------------------------------------ *
 * A synthetic ARC
 * ------------------------------------------------------------------------ */

/**
 * Everything an `ft_tt_arc_read32_fn` needs to answer: the two scratch
 * registers, and a CSM image the walk reads the table and values out of.
 *
 * `reads` and `fail_at` exist so a test can prove the walker really issued the
 * reads it is credited with, and can make one of them fail.
 */
typedef struct fake_arc {
    const ft_tt_arch *arch;
    uint32_t table_pointer;
    uint32_t values_pointer;
    uint32_t csm[IMAGE_WORDS];
    unsigned reads;
    unsigned fail_at; /**< 0 = never; otherwise the 1-based read that returns EIO */
} fake_arc;

/** Word index into `csm` for a CSM byte address, or IMAGE_WORDS when it is out
 *  of the image — an out-of-image read answers all-ones, which is what a real
 *  NoC read of nowhere looks like. */
static uint32_t csm_index(const fake_arc *arc, uint32_t address)
{
    uint32_t base = arc->arch->telemetry.csm_base;

    if (address < base)
        return IMAGE_WORDS;
    if (((address - base) / 4u) >= IMAGE_WORDS)
        return IMAGE_WORDS;
    return (address - base) / 4u;
}

static int fake_arc_read32(void *user, uint32_t address, uint32_t *out_value)
{
    fake_arc *arc = (fake_arc *)user;
    uint32_t scratch = arc->arch->telemetry.scratch_base;
    uint32_t index;

    arc->reads++;
    if (arc->fail_at != 0u && arc->reads == arc->fail_at)
        return FT_TT_EIO;

    if (address == scratch + 4u * arc->arch->telemetry.table_pointer_scratch_index) {
        *out_value = arc->table_pointer;
        return FT_TT_OK;
    }
    if (address == scratch + 4u * arc->arch->telemetry.values_pointer_scratch_index) {
        *out_value = arc->values_pointer;
        return FT_TT_OK;
    }
    index = csm_index(arc, address);
    *out_value = (index < IMAGE_WORDS) ? arc->csm[index] : 0xFFFFFFFFu;
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------ *
 * Building an image whose contents cannot be guessed
 * ------------------------------------------------------------------------ */

/** The five tag values this run publishes, and what the grid should therefore
 *  hold. Seeded per run — see the file header. */
typedef struct image_plan {
    uint16_t tag[TAG_COUNT];
    uint32_t value[TAG_COUNT];
    uint32_t expect_column_count;
    uint32_t expect_core_count;
    uint32_t expect_dram_banks;
} image_plan;

static uint32_t popcount32(uint32_t value)
{
    uint32_t count = 0;

    while (value) {
        value &= value - 1u;
        count++;
    }
    return count;
}

/** A seed nothing in the tree can have baked in. */
static uint32_t run_seed(void)
{
    struct timespec ts;

    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)getpid() * 2654435761u + (uint32_t)ts.tv_nsec;
}

/**
 * Fill `arc` with a well-formed telemetry image and say what the grid must then
 * be.
 *
 * Two properties are built in on purpose:
 *   - the tensix column mask carries a DON'T-CARE BIT ABOVE the part's column
 *     count, so a reader that popcounts the raw word instead of masking first
 *     over-counts. That bug was real in this file's own history;
 *   - the five values are distinct and unguessable, so a swap between two tags
 *     is a failure rather than a coincidence.
 */
static void build_image(fake_arc *arc, image_plan *plan, uint32_t seed)
{
    const ft_tt_arch *arch = arc->arch;
    uint32_t columns = arch->tensix.column_count;
    uint32_t banks = arch->dram.bank_count;
    uint32_t column_mask;
    uint32_t bank_mask;
    uint32_t entry_base_word;
    uint32_t value_base_word;
    uint32_t i;

    /* At least one column and one bank, whatever the seed says, and a high bit
     * the part does not have so masking is load-bearing. */
    column_mask = ((seed | 1u) & ((1u << columns) - 1u)) | (1u << (columns + 1u));
    bank_mask = (seed >> 8) | 1u;
    bank_mask &= (1u << banks) - 1u;

    plan->tag[0] = arch->telemetry.tag_enabled_tensix_col;
    plan->tag[1] = arch->telemetry.tag_enabled_eth;
    plan->tag[2] = arch->telemetry.tag_enabled_gddr;
    plan->tag[3] = arch->telemetry.tag_enabled_l2cpu;
    plan->tag[4] = arch->telemetry.tag_noc_translation;
    plan->value[0] = column_mask;
    plan->value[1] = seed ^ 0x5A5A0001u;
    plan->value[2] = bank_mask;
    plan->value[3] = seed ^ 0x5A5A0003u;
    plan->value[4] = seed ^ 0x5A5A0004u;
    plan->expect_column_count = popcount32(column_mask & ((1u << columns) - 1u));
    plan->expect_core_count = plan->expect_column_count * arch->tensix.row_count;
    plan->expect_dram_banks = popcount32(bank_mask & ((1u << banks) - 1u));

    /* Layout inside the image: [0] table header, [2..] entries, [64..] values. */
    entry_base_word = 2u;
    value_base_word = 64u;
    memset(arc->csm, 0, sizeof(arc->csm));
    arc->table_pointer = arch->telemetry.csm_base;
    arc->values_pointer = arch->telemetry.csm_base + value_base_word * 4u;
    arc->csm[0] = 0x00010000u; /* version; the walk does not read it */
    arc->csm[1] = TAG_COUNT;   /* entry count */
    for (i = 0; i < TAG_COUNT; i++) {
        uint32_t offset = TAG_COUNT - 1u - i; /* deliberately NOT entry order */

        arc->csm[entry_base_word + i] = (uint32_t)plan->tag[i] | (offset << 16);
        arc->csm[value_base_word + offset] = plan->value[i];
    }
}

/* ------------------------------------------------------------------------ *
 * Stage 1 — transport selection
 * ------------------------------------------------------------------------ */

static void check_selection(const char *value, int expect_rc, ft_tt_runtime_driver expect_driver,
                            bool expect_explicit)
{
    ft_tt_runtime_driver driver = (ft_tt_runtime_driver)0xFF;
    bool explicit_choice = true;
    int rc;

    if (value)
        (void)setenv(FT_TT_RUNTIME_DRIVER_ENVIRONMENT, value, 1);
    else
        (void)unsetenv(FT_TT_RUNTIME_DRIVER_ENVIRONMENT);

    rc = ft_tt_runtime_driver_select(&driver, &explicit_choice);
    if (expect_rc != FT_TT_OK) {
        ft_tt_test_check_fmt("an unrecognised FT_TT_DRIVER is REFUSED, never defaulted",
                             rc == expect_rc, "%s='%s' -> rc=%d",
                             FT_TT_RUNTIME_DRIVER_ENVIRONMENT, value ? value : "(unset)", rc);
        return;
    }
    ft_tt_test_check_fmt("FT_TT_DRIVER selection", rc == FT_TT_OK && driver == expect_driver &&
                                                       explicit_choice == expect_explicit,
                         "%s='%s' -> %s, explicit %d (wanted %s, explicit %d)",
                         FT_TT_RUNTIME_DRIVER_ENVIRONMENT, value ? value : "(unset)",
                         ft_tt_runtime_driver_string(driver), (int)explicit_choice,
                         ft_tt_runtime_driver_string(expect_driver), (int)expect_explicit);
}

static void stage_transport_selection(void)
{
    ft_tt_test_step("Stage 1 — the transport is OPT-IN and a bad name is a refusal");

    check_selection(NULL, FT_TT_OK, FT_TT_RUNTIME_DRIVER_KMD, false);
    check_selection("", FT_TT_OK, FT_TT_RUNTIME_DRIVER_KMD, false);
    check_selection("kmd", FT_TT_OK, FT_TT_RUNTIME_DRIVER_KMD, true);
    check_selection("udrv", FT_TT_OK, FT_TT_RUNTIME_DRIVER_UDRV, true);
    /* The three ways somebody actually gets it wrong. Each must REFUSE: a typo
     * that silently selects tt-kmd runs a whole round on the transport the
     * operator was trying to leave, and every number in it is then attributed
     * to the other one. */
    check_selection("UDRV", FT_TT_EINVAL, FT_TT_RUNTIME_DRIVER_KMD, false);
    check_selection("udrv ", FT_TT_EINVAL, FT_TT_RUNTIME_DRIVER_KMD, false);
    check_selection("1", FT_TT_EINVAL, FT_TT_RUNTIME_DRIVER_KMD, false);

    (void)unsetenv(FT_TT_RUNTIME_DRIVER_ENVIRONMENT);
    ft_tt_test_check("the default with the variable unset is kmd, so nothing changes for "
                     "anyone who does not ask",
                     true, "");
}

/* ------------------------------------------------------------------------ *
 * Stage 2 — architecture selection with no driver opinion
 * ------------------------------------------------------------------------ */

static void stage_arch_selection(void)
{
    const ft_tt_arch *blackhole = ft_tt_arch_blackhole();
    const ft_tt_arch *selected;

    ft_tt_test_step("Stage 2 — the arch table from PCI ids alone (no tt-kmd chip_arch)");

    selected = ft_tt_arch_select_by_pci_ids(blackhole->pci_vendor_id, blackhole->pci_device_ids[0]);
    ft_tt_test_check_fmt("a known Blackhole id selects the Blackhole table", selected == blackhole,
                         "%04x:%04x -> %s", blackhole->pci_vendor_id,
                         blackhole->pci_device_ids[0], selected ? selected->name : "(none)");

    selected = ft_tt_arch_select_by_pci_ids(0x10DEu, blackhole->pci_device_ids[0]);
    ft_tt_test_check_fmt("the VENDOR is still checked with no driver to cross-check against",
                         selected == NULL, "10de:%04x -> %s", blackhole->pci_device_ids[0],
                         selected ? selected->name : "(refused)");

    selected = ft_tt_arch_select_by_pci_ids(blackhole->pci_vendor_id, 0xFFFEu);
    ft_tt_test_check_fmt("an id no table names is REFUSED rather than defaulted to the one "
                         "architecture we have",
                         selected == NULL, "%04x:fffe -> %s", blackhole->pci_vendor_id,
                         selected ? selected->name : "(refused)");
}

/* ------------------------------------------------------------------------ *
 * Stage 3 — the shared telemetry walk
 * ------------------------------------------------------------------------ */

static void stage_telemetry_walk(uint32_t seed)
{
    const ft_tt_arch *arch = ft_tt_arch_blackhole();
    static fake_arc arc;
    image_plan plan;
    ft_tt_grid_info grid;
    int rc;

    ft_tt_test_step("Stage 3 — ft_tt_arc_telemetry_read_grid_via(), the walk BOTH transports "
                    "now share (seed 0x%08X)", seed);

    memset(&arc, 0, sizeof(arc));
    arc.arch = arch;
    build_image(&arc, &plan, seed);

    memset(&grid, 0, sizeof(grid));
    rc = ft_tt_arc_telemetry_read_grid_via(arch, fake_arc_read32, &arc, &grid);
    ft_tt_test_check_fmt("a well-formed image reads", rc == FT_TT_OK, "rc=%d, %u reads", rc,
                         arc.reads);
    if (rc != FT_TT_OK)
        return;

    /* Every one of these five is a value this process invented moments ago. A
     * reader answering from a constant, or reusing one row for every tag,
     * cannot produce them. */
    ft_tt_test_check_fmt("tensix_enabled_col came from the table", grid.tensix_enabled_col ==
                                                                      plan.value[0],
                         "0x%08X (wanted 0x%08X)", grid.tensix_enabled_col, plan.value[0]);
    ft_tt_test_check_fmt("enabled_eth came from the table", grid.enabled_eth == plan.value[1],
                         "0x%08X (wanted 0x%08X)", grid.enabled_eth, plan.value[1]);
    ft_tt_test_check_fmt("enabled_gddr came from the table", grid.enabled_gddr == plan.value[2],
                         "0x%08X (wanted 0x%08X)", grid.enabled_gddr, plan.value[2]);
    ft_tt_test_check_fmt("enabled_l2cpu came from the table", grid.enabled_l2cpu == plan.value[3],
                         "0x%08X (wanted 0x%08X)", grid.enabled_l2cpu, plan.value[3]);
    ft_tt_test_check_fmt("noc_translation_enabled came from the table",
                         grid.noc_translation_enabled == plan.value[4], "0x%08X (wanted 0x%08X)",
                         grid.noc_translation_enabled, plan.value[4]);

    /* The mask is applied BEFORE the popcount. The image sets a bit above the
     * part's column count on purpose: a raw popcount over-counts, and the
     * derived core count then describes a card that does not exist. */
    ft_tt_test_check_fmt("the column count MASKS to the part's columns before counting",
                         grid.tensix_column_count == plan.expect_column_count, "%u (wanted %u) "
                         "from mask 0x%08X over %u columns",
                         grid.tensix_column_count, plan.expect_column_count, plan.value[0],
                         arch->tensix.column_count);
    ft_tt_test_check_fmt("the core count is columns x the part's row count",
                         grid.tensix_core_count == plan.expect_core_count, "%u (wanted %u)",
                         grid.tensix_core_count, plan.expect_core_count);
    ft_tt_test_check_fmt("the DRAM bank count is masked and counted the same way",
                         grid.dram_bank_count == plan.expect_dram_banks, "%u (wanted %u)",
                         grid.dram_bank_count, plan.expect_dram_banks);
    ft_tt_test_check_fmt("the grid is marked valid", grid.valid, "%d", (int)grid.valid);
    ft_tt_test_check_fmt("the grid names the architecture it was read against",
                         grid.arch == arch, "%s", grid.arch ? grid.arch->name : "(none)");
}

/** Every refusal the walk makes, each provoked on purpose. A reader that made
 *  none of them would return a plausible grid from a corrupt table, which is
 *  the failure that cannot be seen later. */
static void stage_refusals(uint32_t seed)
{
    const ft_tt_arch *arch = ft_tt_arch_blackhole();
    static fake_arc arc;
    image_plan plan;
    ft_tt_grid_info grid;
    int rc;

    ft_tt_test_step("Stage 4 — the refusals, provoked one at a time");

    /* (a) ARC has not published a table yet. */
    memset(&arc, 0, sizeof(arc));
    arc.arch = arch;
    build_image(&arc, &plan, seed);
    arc.table_pointer = 0u;
    rc = ft_tt_arc_telemetry_read_grid_via(arch, fake_arc_read32, &arc, &grid);
    ft_tt_test_check_fmt("a zero table pointer is UNAVAILABLE, not an error and not a guess",
                         rc == FT_TT_EUNAVAIL, "rc=%d", rc);

    /* (b) a pointer outside the CSM. tt-kmd passes this straight to a bounded
     * accessor and reports only -EIO; we distinguish it before dereferencing. */
    memset(&arc, 0, sizeof(arc));
    arc.arch = arch;
    build_image(&arc, &plan, seed);
    arc.table_pointer = arch->telemetry.csm_base + arch->telemetry.csm_size;
    rc = ft_tt_arc_telemetry_read_grid_via(arch, fake_arc_read32, &arc, &grid);
    ft_tt_test_check_fmt("a table pointer outside the CSM is refused BEFORE it is followed",
                         rc == FT_TT_EIO, "rc=%d", rc);

    /* (c) a corrupt entry count. A 16-bit tag space bounds a well-formed table. */
    memset(&arc, 0, sizeof(arc));
    arc.arch = arch;
    build_image(&arc, &plan, seed);
    arc.csm[1] = FT_TT_ARC_TELEMETRY_MAX_ENTRIES + 1u;
    rc = ft_tt_arc_telemetry_read_grid_via(arch, fake_arc_read32, &arc, &grid);
    ft_tt_test_check_fmt("an entry count past the tag space is refused", rc == FT_TT_EIO,
                         "%u entries -> rc=%d", arc.csm[1], rc);

    /* (d) a value offset that lands outside the CSM. The offset field is 16
     * bits, so the largest reach is 256 KiB — half this part's CSM. Escaping it
     * needs the value ARRAY near the top as well, which is why the pointer moves
     * too. (The first attempt used a large offset alone and did NOT leave the
     * CSM: the arm passed for the wrong reason and the check caught it.) */
    memset(&arc, 0, sizeof(arc));
    arc.arch = arch;
    build_image(&arc, &plan, seed);
    arc.values_pointer = arch->telemetry.csm_base + arch->telemetry.csm_size - 8u;
    arc.csm[2] = (uint32_t)plan.tag[0] | (0xFFFFu << 16);
    rc = ft_tt_arc_telemetry_read_grid_via(arch, fake_arc_read32, &arc, &grid);
    ft_tt_test_check_fmt("a tag whose value offset leaves the CSM is refused", rc == FT_TT_EIO,
                         "rc=%d", rc);

    /* (e) a firmware whose table simply does not carry a tag we need. Refusing
     * is the point: a missing mask silently read as 0 is a card with no cores,
     * and every loop over it then reports "0 wedged" on a card nobody looked
     * at — the exact vacuity the rehearsal caught. */
    memset(&arc, 0, sizeof(arc));
    arc.arch = arch;
    build_image(&arc, &plan, seed);
    arc.csm[2] = (uint32_t)(plan.tag[0] + 1000u) | (4u << 16);
    rc = ft_tt_arc_telemetry_read_grid_via(arch, fake_arc_read32, &arc, &grid);
    ft_tt_test_check_fmt("a tag missing from this firmware's table is refused, not defaulted "
                         "to zero",
                         rc == FT_TT_EIO, "rc=%d", rc);

    /* (f) the transport's own failure is propagated UNCHANGED, so a caller can
     * still tell a dead NoC from a malformed table. */
    memset(&arc, 0, sizeof(arc));
    arc.arch = arch;
    build_image(&arc, &plan, seed);
    arc.fail_at = 1u;
    rc = ft_tt_arc_telemetry_read_grid_via(arch, fake_arc_read32, &arc, &grid);
    ft_tt_test_check_fmt("a transport failure is propagated, not rewritten", rc == FT_TT_EIO,
                         "rc=%d after %u reads", rc, arc.reads);

    /* (g) arguments. */
    rc = ft_tt_arc_telemetry_read_grid_via(NULL, fake_arc_read32, &arc, &grid);
    ft_tt_test_check_fmt("a NULL arch is refused", rc == FT_TT_EINVAL, "rc=%d", rc);
    rc = ft_tt_arc_telemetry_read_grid_via(arch, NULL, &arc, &grid);
    ft_tt_test_check_fmt("a NULL transport is refused", rc == FT_TT_EINVAL, "rc=%d", rc);
}

/* ------------------------------------------------------------------------ *
 * Stage 5 — what a serve budgets against
 * ------------------------------------------------------------------------ */

static void stage_dram_bytes(uint32_t seed)
{
    const ft_tt_arch *arch = ft_tt_arch_blackhole();
    static fake_arc arc;
    static ft_tt_runtime runtime;
    image_plan plan;
    uint64_t expected;

    ft_tt_test_step("Stage 5 — ft_tt_runtime_dram_bytes(), the capacity a serve budgets "
                    "against on EITHER transport");

    memset(&arc, 0, sizeof(arc));
    arc.arch = arch;
    build_image(&arc, &plan, seed);
    memset(&runtime, 0, sizeof(runtime));

    ft_tt_test_check_fmt("an unread grid reports 0 rather than the nominal bank count — a "
                         "telemetry failure must end a scan, not budget an invented card",
                         ft_tt_runtime_dram_bytes(&runtime) == 0u, "%llu",
                         (unsigned long long)ft_tt_runtime_dram_bytes(&runtime));
    ft_tt_test_check_fmt("a NULL runtime reports 0", ft_tt_runtime_dram_bytes(NULL) == 0u, "%llu",
                         (unsigned long long)ft_tt_runtime_dram_bytes(NULL));

    if (ft_tt_arc_telemetry_read_grid_via(arch, fake_arc_read32, &arc, &runtime.grid) !=
        FT_TT_OK) {
        ft_tt_test_check("the grid reads for the capacity arm", false, "");
        return;
    }
    /* A grid whose COUNTS are populated but whose `valid` flag is clear. This
     * is the arm that makes the flag load-bearing, and the battery is what
     * found it missing: with only the zeroed-grid arm above, deleting the
     * `valid` check entirely left the suite GREEN, because a zeroed grid also
     * has a NULL arch. A grid can be populated by something that is not the
     * telemetry walk, and `valid` is the only thing that says it was not. */
    {
        ft_tt_runtime unread;

        memset(&unread, 0, sizeof(unread));
        unread.grid = runtime.grid;
        unread.grid.valid = false;
        ft_tt_test_check_fmt("a grid with counts but no `valid` flag reports 0 — the flag is "
                             "what says the telemetry walk put them there",
                             ft_tt_runtime_dram_bytes(&unread) == 0u, "%llu from %u banks",
                             (unsigned long long)ft_tt_runtime_dram_bytes(&unread),
                             unread.grid.dram_bank_count);
    }

    /* A grid marked valid but with no arch is the shape a half-filled struct
     * takes; it must not multiply out to a number. */
    {
        ft_tt_grid_info orphan = runtime.grid;
        ft_tt_runtime orphan_runtime;

        memset(&orphan_runtime, 0, sizeof(orphan_runtime));
        orphan.arch = NULL;
        orphan_runtime.grid = orphan;
        ft_tt_test_check_fmt("a valid grid that names no architecture reports 0",
                             ft_tt_runtime_dram_bytes(&orphan_runtime) == 0u, "%llu",
                             (unsigned long long)ft_tt_runtime_dram_bytes(&orphan_runtime));
    }

    expected = (uint64_t)plan.expect_dram_banks * arch->core.dram_bytes_per_bank;
    ft_tt_test_check_fmt("capacity is the LIVE bank count times the SELECTED part's bank size",
                         ft_tt_runtime_dram_bytes(&runtime) == expected,
                         "%llu (wanted %llu) from %u banks",
                         (unsigned long long)ft_tt_runtime_dram_bytes(&runtime),
                         (unsigned long long)expected, plan.expect_dram_banks);
}

int main(void)
{
    uint32_t seed = run_seed();

    printf("test_runtime_seam — the transport seam, off hardware. seed 0x%08X\n", seed);
    stage_transport_selection();
    stage_arch_selection();
    stage_telemetry_walk(seed);
    stage_refusals(seed);
    stage_dram_bytes(seed);
    return ft_tt_test_finish();
}
