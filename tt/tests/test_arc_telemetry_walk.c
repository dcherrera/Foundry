/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file test_arc_telemetry_walk.c
 * @brief Off-hardware gates for ft_tt_arc_telemetry_read_tags_via() and
 *   ft_tt_arc_telemetry_read_grid_via(), driven through a fake ARC whose CSM image this
 *   file lays out entry by entry: first match wins, missing tags, an unpublished table,
 *   pointers / entry arrays / values outside the CSM, an oversized entry count, the
 *   popcount restriction, transport-error propagation and argument refusals.
 *
 * The fake also counts every read that lands outside the CSM (other than the two scratch
 * registers), so each refusal is also proven to happen BEFORE the bad address is read.
 *
 * Built and run by `make -f Makefile.runtimeseam check`; `make -f Makefile.runtimeseam
 * arctelemetry-red` rebuilds a copy of the reader with the entry's tag/offset halves
 * swapped and requires this test to fail.
 *
 * @status Off-hardware; touches no device.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_arch.h"
#include "ft_tt_device.h"
#include "ft_tt_test_report.h"

/** CSM size of the synthetic part: big enough for a 65536-entry table plus values. */
#define FAKE_CSM_WORDS 0x20000u
#define FAKE_CSM_BASE 0x10000000u
#define FAKE_SCRATCH_BASE 0x80030400u
#define TABLE_SCRATCH_INDEX 13u
#define VALUES_SCRATCH_INDEX 12u

/** Where the default image puts things, in CSM words. */
#define TABLE_WORD 0u
#define VALUES_WORD 0x1000u

/** Tags of the synthetic part's five grid facts. */
#define TAG_COL 34u
#define TAG_ETH 35u
#define TAG_GDDR 36u
#define TAG_L2CPU 37u
#define TAG_NOC 40u

/** A synthetic ARC: two scratch registers, a CSM image, and a read census. */
typedef struct fake_arc {
    const ft_tt_arch *arch;
    uint32_t table_pointer;
    uint32_t values_pointer;
    uint32_t csm[FAKE_CSM_WORDS];
    unsigned reads;
    unsigned out_of_csm_reads;
    unsigned fail_at;  /**< 1-based read that fails; 0 = never */
    int fail_code;
} fake_arc;

static fake_arc g_arc;
static ft_tt_arch g_arch;

static int fake_arc_read32(void *user, uint32_t address, uint32_t *out_value)
{
    fake_arc *arc = (fake_arc *)user;
    const ft_tt_arch_telemetry *telemetry = &arc->arch->telemetry;

    arc->reads++;
    if (arc->fail_at != 0u && arc->reads == arc->fail_at)
        return arc->fail_code;
    if (address == telemetry->scratch_base + 4u * telemetry->table_pointer_scratch_index) {
        *out_value = arc->table_pointer;
        return FT_TT_OK;
    }
    if (address == telemetry->scratch_base + 4u * telemetry->values_pointer_scratch_index) {
        *out_value = arc->values_pointer;
        return FT_TT_OK;
    }
    if (address < telemetry->csm_base ||
        (uint64_t)address + 4u > (uint64_t)telemetry->csm_base + telemetry->csm_size ||
        (address - telemetry->csm_base) / 4u >= FAKE_CSM_WORDS) {
        arc->out_of_csm_reads++;
        *out_value = 0xFFFFFFFFu;
        return FT_TT_OK;
    }
    *out_value = arc->csm[(address - telemetry->csm_base) / 4u];
    return FT_TT_OK;
}

/** The synthetic part: Blackhole's table with this file's own telemetry layout. */
static void reset_arch(uint32_t column_count, uint32_t bank_count)
{
    g_arch = *ft_tt_arch_blackhole();
    g_arch.tensix.column_count = column_count;
    g_arch.dram.bank_count = bank_count;
    g_arch.telemetry.scratch_base = FAKE_SCRATCH_BASE;
    g_arch.telemetry.table_pointer_scratch_index = TABLE_SCRATCH_INDEX;
    g_arch.telemetry.values_pointer_scratch_index = VALUES_SCRATCH_INDEX;
    g_arch.telemetry.csm_base = FAKE_CSM_BASE;
    g_arch.telemetry.csm_size = FAKE_CSM_WORDS * 4u;
    g_arch.telemetry.tag_enabled_tensix_col = TAG_COL;
    g_arch.telemetry.tag_enabled_eth = TAG_ETH;
    g_arch.telemetry.tag_enabled_gddr = TAG_GDDR;
    g_arch.telemetry.tag_enabled_l2cpu = TAG_L2CPU;
    g_arch.telemetry.tag_noc_translation = TAG_NOC;
}

static uint32_t entry_word(uint32_t tag, uint32_t word_offset)
{
    return (tag & 0xFFFFu) | (word_offset << 16);
}

/**
 * Lay out a table at TABLE_WORD and values at VALUES_WORD. Offsets are chosen so the
 * tag and offset halves can never be confused: no offset equals any tag in the table.
 */
static void build_image(const uint32_t *tags, const uint32_t *offsets,
                        const uint32_t *values, unsigned entry_count)
{
    unsigned i;

    memset(&g_arc, 0, sizeof(g_arc));
    g_arc.arch = &g_arch;
    g_arc.table_pointer = FAKE_CSM_BASE + 4u * TABLE_WORD;
    g_arc.values_pointer = FAKE_CSM_BASE + 4u * VALUES_WORD;
    g_arc.csm[TABLE_WORD] = 0x00010002u;
    g_arc.csm[TABLE_WORD + 1u] = entry_count;
    for (i = 0; i < entry_count; i++) {
        g_arc.csm[TABLE_WORD + 2u + i] = entry_word(tags[i], offsets[i]);
        g_arc.csm[VALUES_WORD + offsets[i]] = values[i];
    }
}

/** The standard five-tag image, with a leading unrelated entry. */
static void build_grid_image(uint32_t column_mask, uint32_t gddr_mask)
{
    static const uint32_t tags[6] = { 7u, TAG_NOC, TAG_COL, TAG_GDDR, TAG_ETH, TAG_L2CPU };
    static const uint32_t offsets[6] = { 300u, 101u, 211u, 5u, 77u, 1023u };
    uint32_t values[6];

    values[0] = 0xDEAD0007u;
    values[1] = 0x00000001u;
    values[2] = column_mask;
    values[3] = gddr_mask;
    values[4] = 0x00003C3Cu;
    values[5] = 0x0000000Au;
    build_image(tags, offsets, values, 6u);
}

/* ------------------------------------------------------------------------- */

static void stage_grid_success(void)
{
    ft_tt_grid_info grid;
    int rc;

    ft_tt_test_step("grid_via over a hand-laid table");
    reset_arch(14u, 8u);
    build_grid_image(0xC0003A5Bu, 0x000001F3u);
    memset(&grid, 0, sizeof(grid));
    rc = ft_tt_arc_telemetry_read_grid_via(&g_arch, fake_arc_read32, &g_arc, &grid);
    ft_tt_test_check_fmt("reads", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("tensix mask", grid.tensix_enabled_col == 0xC0003A5Bu, "0x%08X",
                         grid.tensix_enabled_col);
    ft_tt_test_check_fmt("eth mask", grid.enabled_eth == 0x3C3Cu, "0x%08X", grid.enabled_eth);
    ft_tt_test_check_fmt("gddr mask", grid.enabled_gddr == 0x1F3u, "0x%08X", grid.enabled_gddr);
    ft_tt_test_check_fmt("l2cpu mask", grid.enabled_l2cpu == 0xAu, "0x%08X", grid.enabled_l2cpu);
    ft_tt_test_check_fmt("noc translation", grid.noc_translation_enabled == 1u, "%u",
                         grid.noc_translation_enabled);
    /* 0x3A5B & 0x3FFF = 0x3A5B -> 9 bits; the 0xC000_0000 bits are above 14 columns. */
    ft_tt_test_check_fmt("column count masks to 14 columns", grid.tensix_column_count == 9u,
                         "%u", grid.tensix_column_count);
    ft_tt_test_check_fmt("core count = columns x rows",
                         grid.tensix_core_count == 9u * g_arch.tensix.row_count, "%u",
                         grid.tensix_core_count);
    /* 0x1F3 & 0xFF = 0xF3 -> 6 bits. */
    ft_tt_test_check_fmt("dram banks mask to 8 banks", grid.dram_bank_count == 6u, "%u",
                         grid.dram_bank_count);
    ft_tt_test_check_fmt("valid and names its arch", grid.valid && grid.arch == &g_arch, "%d",
                         (int)grid.valid);
    ft_tt_test_check_fmt("no read left the CSM", g_arc.out_of_csm_reads == 0u, "%u",
                         g_arc.out_of_csm_reads);
}

/** Expected restricted popcount for one (mask, bit count) case. */
static void check_popcount_case(uint32_t columns, uint32_t banks, uint32_t mask,
                                uint32_t want_columns, uint32_t want_banks)
{
    ft_tt_grid_info grid;
    int rc;

    reset_arch(columns, banks);
    build_grid_image(mask, mask);
    memset(&grid, 0, sizeof(grid));
    rc = ft_tt_arc_telemetry_read_grid_via(&g_arch, fake_arc_read32, &g_arc, &grid);
    ft_tt_test_check_fmt("popcount restriction", rc == FT_TT_OK &&
                         grid.tensix_column_count == want_columns &&
                         grid.dram_bank_count == want_banks,
                         "mask 0x%08X cols %u banks %u -> rc=%d %u/%u (want %u/%u)", mask,
                         columns, banks, rc, grid.tensix_column_count, grid.dram_bank_count,
                         want_columns, want_banks);
}

static void stage_popcount_restriction(void)
{
    ft_tt_test_step("popcount restricted to the part's low bits");
    check_popcount_case(4u, 2u, 0xFFFFFFFFu, 4u, 2u);
    check_popcount_case(0u, 0u, 0xFFFFFFFFu, 0u, 0u);
    check_popcount_case(32u, 32u, 0x80000001u, 2u, 2u);
    check_popcount_case(40u, 31u, 0xFFFFFFFFu, 32u, 31u);
    check_popcount_case(14u, 8u, 0xFFFFC000u, 0u, 0u);
}

static void stage_tags_via(void)
{
    static const uint32_t tags[5] = { 50u, 60u, 50u, 70u, 60u };
    static const uint32_t offsets[5] = { 9u, 1000u, 3u, 444u, 12u };
    static const uint32_t values[5] = { 0x11111111u, 0x22222222u, 0x33333333u,
                                        0x44444444u, 0x55555555u };
    uint16_t want[5] = { 60u, 50u, 99u, 70u, 50u };
    uint32_t got[5];
    bool found[5];
    int rc;

    ft_tt_test_step("tags_via: first match wins, missing tag stays unfound");
    reset_arch(14u, 8u);
    build_image(tags, offsets, values, 5u);
    memset(got, 0xA5, sizeof(got));
    memset(found, 1, sizeof(found));
    rc = ft_tt_arc_telemetry_read_tags_via(&g_arch, fake_arc_read32, &g_arc, want, 5u, got,
                                           found);
    ft_tt_test_check_fmt("reads", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("tag 60: first entry wins", found[0] && got[0] == 0x22222222u,
                         "found=%d 0x%08X", (int)found[0], got[0]);
    ft_tt_test_check_fmt("tag 50: first entry wins", found[1] && got[1] == 0x11111111u,
                         "found=%d 0x%08X", (int)found[1], got[1]);
    ft_tt_test_check_fmt("tag 99 missing: unfound, zeroed", !found[2] && got[2] == 0u,
                         "found=%d 0x%08X", (int)found[2], got[2]);
    ft_tt_test_check_fmt("tag 70", found[3] && got[3] == 0x44444444u, "found=%d 0x%08X",
                         (int)found[3], got[3]);
    ft_tt_test_check_fmt("duplicate request gets the same first value",
                         found[4] && got[4] == 0x11111111u, "found=%d 0x%08X", (int)found[4],
                         got[4]);
}

/** Run grid_via on the current image and require `want_rc` with the grid untouched and no
 *  read outside the CSM. */
static void expect_grid_refusal(const char *what, int want_rc)
{
    ft_tt_grid_info grid;
    int rc;

    memset(&grid, 0x5C, sizeof(grid));
    rc = ft_tt_arc_telemetry_read_grid_via(&g_arch, fake_arc_read32, &g_arc, &grid);
    ft_tt_test_check_fmt(what, rc == want_rc && grid.tensix_enabled_col == 0x5C5C5C5Cu &&
                         g_arc.out_of_csm_reads == 0u,
                         "rc=%d (want %d), grid %s, %u out-of-CSM reads", rc, want_rc,
                         grid.tensix_enabled_col == 0x5C5C5C5Cu ? "untouched" : "WRITTEN",
                         g_arc.out_of_csm_reads);
}

static void stage_refusals(void)
{
    const uint32_t csm_end = FAKE_CSM_BASE + FAKE_CSM_WORDS * 4u;

    ft_tt_test_step("refusals");
    reset_arch(14u, 8u);

    build_grid_image(0x3FFFu, 0xFFu);
    g_arc.table_pointer = 0u;
    expect_grid_refusal("table pointer 0 is UNAVAILABLE", FT_TT_EUNAVAIL);

    build_grid_image(0x3FFFu, 0xFFu);
    g_arc.table_pointer = FAKE_CSM_BASE - 4u;
    expect_grid_refusal("table pointer below the CSM", FT_TT_EIO);

    build_grid_image(0x3FFFu, 0xFFu);
    g_arc.table_pointer = csm_end - 4u; /* header needs 8 bytes */
    expect_grid_refusal("table header straddling the CSM end", FT_TT_EIO);

    build_grid_image(0x3FFFu, 0xFFu);
    g_arc.values_pointer = csm_end;
    expect_grid_refusal("values pointer at the CSM end", FT_TT_EIO);

    build_grid_image(0x3FFFu, 0xFFu);
    g_arc.csm[TABLE_WORD + 1u] = FT_TT_ARC_TELEMETRY_MAX_ENTRIES + 1u;
    expect_grid_refusal("entry count 65537 is malformed", FT_TT_EIO);

    build_grid_image(0x3FFFu, 0xFFu);
    g_arc.table_pointer = csm_end - 8u - 4u * 3u;
    g_arc.csm[FAKE_CSM_WORDS - 4u] = 6u; /* 6 entries, only 3 fit */
    expect_grid_refusal("entry array running past the CSM", FT_TT_EIO);

    build_grid_image(0x3FFFu, 0xFFu);
    g_arc.values_pointer = csm_end - 8u;
    expect_grid_refusal("value offset leaving the CSM", FT_TT_EIO);

    build_grid_image(0x3FFFu, 0xFFu);
    g_arc.csm[TABLE_WORD + 2u + 4u] = entry_word(TAG_ETH + 100u, 77u);
    expect_grid_refusal("grid tag missing from the table", FT_TT_EIO);

    build_grid_image(0x3FFFu, 0xFFu);
    g_arc.fail_at = 6u;
    g_arc.fail_code = FT_TT_ENOMEM;
    expect_grid_refusal("transport error propagated unchanged", FT_TT_ENOMEM);
}

static void stage_entry_count_bound(void)
{
    uint16_t want = TAG_NOC;
    uint32_t got = 0;
    bool found = false;
    int rc;

    ft_tt_test_step("entry count exactly 65536 is accepted");
    reset_arch(14u, 8u);
    memset(&g_arc, 0, sizeof(g_arc));
    g_arc.arch = &g_arch;
    g_arc.table_pointer = FAKE_CSM_BASE;
    g_arc.values_pointer = FAKE_CSM_BASE + 4u * (FAKE_CSM_WORDS - 4u);
    g_arc.csm[1] = FT_TT_ARC_TELEMETRY_MAX_ENTRIES;
    g_arc.csm[2u + FT_TT_ARC_TELEMETRY_MAX_ENTRIES - 1u] = entry_word(TAG_NOC, 2u);
    g_arc.csm[FAKE_CSM_WORDS - 2u] = 0xFEEDF00Du;
    rc = ft_tt_arc_telemetry_read_tags_via(&g_arch, fake_arc_read32, &g_arc, &want, 1u, &got,
                                           &found);
    ft_tt_test_check_fmt("last of 65536 entries found", rc == FT_TT_OK && found &&
                         got == 0xFEEDF00Du, "rc=%d found=%d 0x%08X", rc, (int)found, got);
}

static void stage_arguments(void)
{
    uint16_t tag = TAG_COL;
    uint32_t value;
    bool found;
    ft_tt_grid_info grid;

    ft_tt_test_step("arguments");
    reset_arch(14u, 8u);
    build_grid_image(0x3FFFu, 0xFFu);
    ft_tt_test_check("tags_via NULL arch", ft_tt_arc_telemetry_read_tags_via(
                     NULL, fake_arc_read32, &g_arc, &tag, 1u, &value, &found) == FT_TT_EINVAL,
                     "");
    ft_tt_test_check("tags_via NULL read32", ft_tt_arc_telemetry_read_tags_via(
                     &g_arch, NULL, &g_arc, &tag, 1u, &value, &found) == FT_TT_EINVAL, "");
    ft_tt_test_check("tags_via NULL tags", ft_tt_arc_telemetry_read_tags_via(
                     &g_arch, fake_arc_read32, &g_arc, NULL, 1u, &value, &found) ==
                     FT_TT_EINVAL, "");
    ft_tt_test_check("tags_via NULL values", ft_tt_arc_telemetry_read_tags_via(
                     &g_arch, fake_arc_read32, &g_arc, &tag, 1u, NULL, &found) ==
                     FT_TT_EINVAL, "");
    ft_tt_test_check("tags_via NULL found", ft_tt_arc_telemetry_read_tags_via(
                     &g_arch, fake_arc_read32, &g_arc, &tag, 1u, &value, NULL) ==
                     FT_TT_EINVAL, "");
    ft_tt_test_check("grid_via NULL grid", ft_tt_arc_telemetry_read_grid_via(
                     &g_arch, fake_arc_read32, &g_arc, NULL) == FT_TT_EINVAL, "");
    ft_tt_test_check("read_grid NULL ctx", ft_tt_arc_telemetry_read_grid(NULL, &grid) ==
                     FT_TT_EINVAL, "");
}

int main(void)
{
    printf("test_arc_telemetry_walk: ARC telemetry walk against a hand-laid fake table\n");
    stage_grid_success();
    stage_popcount_restriction();
    stage_tags_via();
    stage_refusals();
    stage_entry_count_bound();
    stage_arguments();
    return ft_tt_test_finish();
}
