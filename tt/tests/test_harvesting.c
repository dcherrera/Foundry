/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file test_harvesting.c
 * @brief Off-hardware harvest matrix for ft_tt_coord: every single-column Tensix harvest
 *   and a set of multi-column ones (the p100's two-column case included), DRAM single-bank
 *   harvests (the doc's D2 example) and the "at most one bank" refusal, and
 *   ft_tt_coord_tensix_harvest_mask round trips.
 *
 * This is the only coverage for code a fully-enabled card never runs. Expectations come
 * from ft_tt_blackhole_doc_layout.h (the public docs: [COORD] "X is remapped to put the
 * fused columns at maximal X ... X = 8 always refers to the column containing CPUs, and
 * X = 9 always refers to a column containing DRAM", [LUA] bh_y211 / bh_y1223) and the
 * spec-given telemetry bit order — never from ft_tt_arch_blackhole.c.
 *
 * Grid convention assumed (no public page states it): GDDR word bit b = bank b.
 *
 * @status Covers the harvest paths of every ft_tt_coord.h lookup.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "ft_tt_arch.h"
#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_blackhole_doc_layout.h"

/* [COORD] / [LUA] bh_y211: the p100 example fuses the columns at NoC #0 x = 5 and 11. */
#define P100_FUSED_COLUMNS ((1u << 4) | (1u << 8))

/* [LUA] bh_y1223: the p100 example fuses DRAM bank D2. */
#define P100_FUSED_BANK 2u

static int checks_run;
static int checks_failed;

static void check(bool condition, const char *what)
{
    checks_run++;
    if (!condition) {
        checks_failed++;
        printf("  FAIL  %s\n", what);
    }
}

static void check_rc(int got, int want, const char *what)
{
    checks_run++;
    if (got != want) {
        checks_failed++;
        printf("  FAIL  %s: rc %d (expected %d)\n", what, got, want);
    }
}

static void check_u32(uint32_t got, uint32_t want, const char *what)
{
    checks_run++;
    if (got != want) {
        checks_failed++;
        printf("  FAIL  %s = %u (expected %u)\n", what, got, want);
    }
}

static void check_coord(int rc, ft_tt_noc_coord got, uint32_t want_x, uint32_t want_y,
                        const char *what, uint32_t index)
{
    checks_run++;
    if (rc != FT_TT_OK || got.x != want_x || got.y != want_y) {
        checks_failed++;
        printf("  FAIL  %s [%u]: rc %d (%u,%u), expected (%u,%u)\n", what, index, rc, got.x,
               got.y, want_x, want_y);
    }
}

static void check_at(bool condition, const char *what, uint32_t x, uint32_t y)
{
    checks_run++;
    if (!condition) {
        checks_failed++;
        printf("  FAIL  %s at (%u,%u)\n", what, x, y);
    }
}

static uint32_t count_bits(uint32_t value)
{
    uint32_t count = 0u;

    for (; value; value &= value - 1u)
        count++;
    return count;
}

static ft_tt_grid_info make_grid(uint32_t harvested_columns, uint32_t enabled_gddr)
{
    ft_tt_grid_info grid = { 0 };

    grid.valid = true;
    grid.arch = ft_tt_arch_blackhole();
    grid.tensix_enabled_col = bh_doc_enabled_column_word(harvested_columns);
    grid.enabled_eth = 0u; /* [COORD]: Ethernet "fused off in p100" */
    grid.enabled_gddr = enabled_gddr;
    grid.enabled_l2cpu = BH_DOC_ALL_L2CPU_MASK;
    grid.noc_translation_enabled = 1u;
    return grid;
}

/* ------------------------------------------------------------------------- */
/* Telemetry word <-> harvest mask                                           */
/* ------------------------------------------------------------------------- */

static void test_harvest_mask_round_trips(void)
{
    const ft_tt_arch *arch = ft_tt_arch_blackhole();
    uint32_t mask = 0u;
    uint32_t harvested;
    uint32_t column;
    uint32_t mismatches = 0u;

    printf("harvest_mask: telemetry word -> grid-column mask\n");
    check_rc(ft_tt_coord_tensix_harvest_mask(arch, bh_doc_enabled_column_word(0u), &mask),
             FT_TT_OK, "all enabled");
    check_u32(mask, 0u, "all enabled harvests nothing");
    check_rc(ft_tt_coord_tensix_harvest_mask(arch, 0u, &mask), FT_TT_OK, "none enabled");
    check_u32(mask, BH_DOC_ALL_COLUMNS_MASK, "none enabled harvests every column");

    for (column = 0; column < BH_DOC_TENSIX_COLUMNS; column++) {
        uint32_t word = BH_DOC_ALL_COLUMNS_MASK & ~(1u << bh_doc_tensix_telemetry_bit(column));

        check_rc(ft_tt_coord_tensix_harvest_mask(arch, word, &mask), FT_TT_OK,
                 "single cleared telemetry bit");
        check_u32(mask, 1u << column, "a cleared telemetry bit names its grid column");
    }

    /* The doc's p100: x = 5 is telemetry bit 8 and x = 11 is bit 11 (spec bit order). */
    check_rc(ft_tt_coord_tensix_harvest_mask(arch, BH_DOC_ALL_COLUMNS_MASK & ~0x900u, &mask),
             FT_TT_OK, "p100 telemetry word");
    check_u32(mask, P100_FUSED_COLUMNS, "p100 word names grid columns 4 (x 5) and 8 (x 11)");

    for (harvested = 0; harvested <= BH_DOC_ALL_COLUMNS_MASK; harvested++) {
        if (ft_tt_coord_tensix_harvest_mask(arch, bh_doc_enabled_column_word(harvested),
                                            &mask) != FT_TT_OK ||
            mask != harvested)
            mismatches++;
    }
    check_u32(mismatches, 0u, "all 16384 harvest patterns round-trip through the word");
}

/* ------------------------------------------------------------------------- */
/* Tensix layout under a harvest                                             */
/* ------------------------------------------------------------------------- */

/** The n-th translated Tensix x: 1..7 then 10..16 ([COORD] gap at 8, 9). */
static uint32_t translated_slot_x(uint32_t slot)
{
    return slot < 7u ? slot + 1u : slot + 3u;
}

/** Logical columns: NoC #0 position, translated position, renumbering rules. */
static void check_tensix_columns(const ft_tt_grid_info *grid, uint32_t harvested, uint32_t columns)
{
    ft_tt_noc_coord coord = { 0u, 0u };
    uint32_t logical = 0u;
    uint32_t column;
    uint32_t row;

    for (column = 0; column < BH_DOC_TENSIX_COLUMNS; column++) {
        uint32_t noc0_x = bh_doc_tensix_noc0_x(column);
        uint32_t translated_x = bh_doc_tensix_translated_x(noc0_x, harvested);
        bool fused = (harvested & (1u << column)) != 0u;

        check(translated_x != 8u && translated_x != 9u, "translated x never uses 8 or 9");
        check(fused == (translated_x > translated_slot_x(columns - 1u)),
              "fused columns, and only they, sit above the live translated range");
        for (row = 0; row < BH_DOC_TENSIX_ROWS; row++) {
            uint32_t y = BH_DOC_FIRST_TENSIX_ROW + row;

            check_at(ft_tt_coord_is_enabled(grid, FT_TT_COORD_NOC0, noc0_x, y) == !fused,
                     "NOC0 is_enabled follows the harvest", noc0_x, y);
            check_at(ft_tt_coord_classify(grid, FT_TT_COORD_NOC0, noc0_x, y) == FT_TT_CORE_TENSIX,
                     "a fused Tensix keeps its type", noc0_x, y);
            if (fused)
                continue;
            check_coord(ft_tt_coord_tensix_at(grid, logical, row, FT_TT_COORD_NOC0, &coord),
                        coord, noc0_x, y, "tensix_at NOC0", logical * 100u + row);
            check_coord(ft_tt_coord_tensix_at(grid, logical, row, FT_TT_COORD_TRANSLATED, &coord),
                        coord, translated_x, y, "tensix_at TRANSLATED", logical * 100u + row);
        }
        if (!fused) {
            check_u32(translated_x, translated_slot_x(logical),
                      "live columns take the lowest translated slots, in NoC #0 order");
            logical++;
        }
    }
    check_u32(logical, columns, "logical columns = enabled columns");
    check_rc(ft_tt_coord_tensix_at(grid, columns, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EINVAL,
             "tensix_at past the live columns");
}

/** Translated Tensix space: live slots below the enabled count, fused slots above. */
static void check_tensix_translated_slots(const ft_tt_grid_info *grid, uint32_t columns)
{
    uint32_t slot;
    uint32_t row;

    for (slot = 0; slot < BH_DOC_TENSIX_COLUMNS; slot++) {
        uint32_t x = translated_slot_x(slot);

        for (row = 0; row < BH_DOC_TENSIX_ROWS; row++) {
            uint32_t y = BH_DOC_FIRST_TENSIX_ROW + row;

            check_at(ft_tt_coord_is_enabled(grid, FT_TT_COORD_TRANSLATED, x, y) == (slot < columns),
                     "translated slot is live iff below the enabled count", x, y);
            check_at(ft_tt_coord_classify(grid, FT_TT_COORD_TRANSLATED, x, y) ==
                         FT_TT_CORE_TENSIX,
                     "translated Tensix slot classifies as Tensix", x, y);
        }
    }
}

/** By-index enumeration is dense: exactly the enabled nodes, each once, row-major. */
static void check_tensix_by_index(const ft_tt_grid_info *grid, uint32_t harvested,
                                  uint32_t columns)
{
    bool seen[BH_DOC_NOC_WIDTH][BH_DOC_NOC_HEIGHT] = { { false } };
    ft_tt_noc_coord coord = { 0u, 0u };
    ft_tt_noc_coord expected = { 0u, 0u };
    uint32_t cores = columns * BH_DOC_TENSIX_ROWS;
    uint32_t duplicates_or_fused = 0u;
    uint32_t index;

    for (index = 0; index < cores; index++) {
        uint32_t column;

        ft_tt_coord_tensix_at(grid, index % columns, index / columns, FT_TT_COORD_TRANSLATED,
                              &expected);
        check_coord(ft_tt_coord_tensix_by_index(grid, index, FT_TT_COORD_TRANSLATED, &coord),
                    coord, expected.x, expected.y, "by_index TRANSLATED is row-major", index);
        if (ft_tt_coord_tensix_by_index(grid, index, FT_TT_COORD_NOC0, &coord) != FT_TT_OK ||
            coord.x >= BH_DOC_NOC_WIDTH || coord.y >= BH_DOC_NOC_HEIGHT) {
            duplicates_or_fused++;
            continue;
        }
        column = bh_doc_tensix_column_at_x(coord.x);
        if (column == BH_DOC_TENSIX_COLUMNS || (harvested & (1u << column)) ||
            seen[coord.x][coord.y])
            duplicates_or_fused++;
        seen[coord.x][coord.y] = true;
    }
    check_u32(duplicates_or_fused, 0u, "by_index NOC0 hits live Tensix only, each once");
    check_u32(ft_tt_coord_tensix_core_count(grid), cores, "core count = live columns x 10");
    check_rc(ft_tt_coord_tensix_by_index(grid, cores, FT_TT_COORD_NOC0, &coord), FT_TT_EINVAL,
             "by_index at the core count is out of range");
}

static void check_tensix_harvest(uint32_t harvested, const char *what)
{
    ft_tt_grid_info grid = make_grid(harvested, BH_DOC_ALL_BANKS_MASK);
    uint32_t columns = BH_DOC_TENSIX_COLUMNS - count_bits(harvested);
    int failed_before = checks_failed;

    check_rc(ft_tt_coord_check_grid(&grid), FT_TT_OK, "harvested grid is accepted");
    check_u32(ft_tt_coord_tensix_column_count(&grid), columns, "column count");
    check_u32(ft_tt_coord_tensix_row_count(&grid), BH_DOC_TENSIX_ROWS, "rows are never fused");
    check_tensix_columns(&grid, harvested, columns);
    check_tensix_translated_slots(&grid, columns);
    check_tensix_by_index(&grid, harvested, columns);
    printf("  %-44s %s\n", what, checks_failed == failed_before ? "ok" : "FAILED");
}

static void test_single_column_harvests(void)
{
    char what[64];
    uint32_t column;

    printf("Tensix: every single-column harvest\n");
    for (column = 0; column < BH_DOC_TENSIX_COLUMNS; column++) {
        snprintf(what, sizeof(what), "column %u (NoC #0 x = %u) fused", column,
                 bh_doc_tensix_noc0_x(column));
        check_tensix_harvest(1u << column, what);
    }
}

/**
 * [LUA] bh_y211, read literally: on the p100 example the translated x of the column at
 * NoC #0 x is this table — checked as plain numbers, independent of the renumbering helper.
 */
static void test_p100_doc_numbering(void)
{
    static const uint8_t translated_by_noc0_x[BH_DOC_NOC_WIDTH] = { 0, 1, 2, 3, 4, 15, 5, 6, 0,
                                                                    0, 7, 16, 10, 11, 12, 13, 14 };
    ft_tt_grid_info grid = make_grid(P100_FUSED_COLUMNS, BH_DOC_ALL_BANKS_MASK);
    ft_tt_noc_coord coord = { 0u, 0u };
    uint32_t logical = 0u;
    uint32_t x;

    printf("Tensix: the doc's p100 numbering (x = 5 and 11 fused)\n");
    for (x = 1u; x < BH_DOC_NOC_WIDTH; x++) {
        uint32_t column = bh_doc_tensix_column_at_x(x);

        if (column == BH_DOC_TENSIX_COLUMNS)
            continue;
        check_u32(bh_doc_tensix_translated_x(x, P100_FUSED_COLUMNS), translated_by_noc0_x[x],
                  "renumbering helper agrees with the doc diagram");
        if (P100_FUSED_COLUMNS & (1u << column))
            continue;
        check_coord(ft_tt_coord_tensix_at(&grid, logical, 0u, FT_TT_COORD_TRANSLATED, &coord),
                    coord, translated_by_noc0_x[x], BH_DOC_FIRST_TENSIX_ROW,
                    "p100 tensix_at TRANSLATED (doc bh_y211)", logical);
        logical++;
    }
    check_u32(ft_tt_coord_tensix_core_count(&grid), 120u, "p100 has 120 Tensix ([COORD] 70+50)");
}

static void test_multi_column_harvests(void)
{
    static const struct {
        uint32_t harvested;
        const char *what;
    } cases[] = {
        { P100_FUSED_COLUMNS, "p100 doc example: x 5 + 11" },
        { (1u << 12) | (1u << 13), "x 15 + 16 (already at maximal X)" },
        { (1u << 0) | (1u << 1), "x 1 + 2 (westmost pair)" },
        { (1u << 0) | (1u << 13), "x 1 + 16 (both edges)" },
        { (1u << 6) | (1u << 7), "x 7 + 10 (either side of the gap)" },
        { 0x007Fu, "all seven west columns" },
        { 0x3F80u, "all seven east columns" },
        { 0x1555u, "alternate columns" },
        { BH_DOC_ALL_COLUMNS_MASK & ~1u, "all but x 1" },
        { BH_DOC_ALL_COLUMNS_MASK & ~(1u << 13), "all but x 16" },
        { (1u << 3) | (1u << 9) | (1u << 10), "three scattered (x 4, 12, 13)" },
    };
    uint32_t index;

    printf("Tensix: multi-column harvests\n");
    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); index++)
        check_tensix_harvest(cases[index].harvested, cases[index].what);
}

/* ------------------------------------------------------------------------- */
/* DRAM harvests                                                             */
/* ------------------------------------------------------------------------- */

/** Translated DRAM alias liveness: [COORD] puts the fused bank at (18, 21-23) whichever
 *  bank it is ("fused off in p100", and the D2 example's x = 0 -> 18, y -> 21-23). */
static void check_dram_alias_liveness(const ft_tt_grid_info *grid)
{
    uint32_t x;
    uint32_t y;

    for (x = BH_DOC_DRAM_ALIAS_FIRST_X; x <= BH_DOC_DRAM_ALIAS_FIRST_X + 1u; x++) {
        for (y = BH_DOC_DRAM_ALIAS_FIRST_Y; y <= BH_DOC_DRAM_ALIAS_LAST_Y; y++) {
            bool fused_slot = x == BH_DOC_DRAM_ALIAS_FIRST_X + 1u && y >= 21u;

            check_at(ft_tt_coord_classify(grid, FT_TT_COORD_TRANSLATED, x, y) == FT_TT_CORE_DRAM,
                     "DRAM alias classifies as DRAM", x, y);
            check_at(ft_tt_coord_is_enabled(grid, FT_TT_COORD_TRANSLATED, x, y) == !fused_slot,
                     "only (18, 21-23) is fused in translated space", x, y);
        }
    }
}

static void check_single_bank_harvest(uint32_t fused_bank)
{
    ft_tt_grid_info grid = make_grid(0u, BH_DOC_ALL_BANKS_MASK & ~(1u << fused_bank));
    ft_tt_noc_coord coord = { 0u, 0u };
    uint32_t logical = 0u;
    uint32_t bank;
    uint32_t port;
    int failed_before = checks_failed;

    check_rc(ft_tt_coord_check_grid(&grid), FT_TT_OK, "one fused bank is accepted");
    check_u32(ft_tt_coord_dram_bank_count(&grid), BH_DOC_DRAM_BANKS - 1u, "bank count");
    for (bank = 0; bank < BH_DOC_DRAM_BANKS; bank++) {
        for (port = 0; port < BH_DOC_DRAM_PORTS_PER_BANK; port++) {
            uint32_t x = bh_doc_dram_noc0_x(bank);
            uint32_t y = bh_doc_dram_port_noc0_y(bank, port);

            check_at(ft_tt_coord_is_enabled(&grid, FT_TT_COORD_NOC0, x, y) == (bank != fused_bank),
                     "NOC0 DRAM tile is_enabled follows the harvest", x, y);
            check_at(ft_tt_coord_classify(&grid, FT_TT_COORD_NOC0, x, y) == FT_TT_CORE_DRAM,
                     "a fused DRAM tile keeps its type", x, y);
            if (bank == fused_bank)
                continue;
            check_coord(ft_tt_coord_dram_port(&grid, logical, port, FT_TT_COORD_NOC0, &coord),
                        coord, x, y, "dram_port NOC0 skips the fused bank", logical * 10u + port);
        }
        if (bank != fused_bank)
            logical++;
    }
    check_rc(ft_tt_coord_dram_port(&grid, logical, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EINVAL,
             "dram_port past the live banks");
    /* ft_tt_coord.h contract: the harvested-bank alias numbering is not modelled. */
    check_rc(ft_tt_coord_dram_bank(&grid, 0u, FT_TT_COORD_TRANSLATED, &coord), FT_TT_EUNAVAIL,
             "dram_bank TRANSLATED refuses with a fused bank");
    check_dram_alias_liveness(&grid);
    printf("  bank %u fused%-33s %s\n", fused_bank, "",
           checks_failed == failed_before ? "ok" : "FAILED");
}

static void test_dram_harvests(void)
{
    ft_tt_grid_info p100 = make_grid(P100_FUSED_COLUMNS,
                                     BH_DOC_ALL_BANKS_MASK & ~(1u << P100_FUSED_BANK));
    uint32_t first;
    uint32_t second;
    uint32_t accepted_pairs = 0u;

    printf("DRAM: every single-bank harvest (D2 is the doc's p100 example)\n");
    for (first = 0; first < BH_DOC_DRAM_BANKS; first++)
        check_single_bank_harvest(first);

    printf("DRAM: at most one bank may be harvested\n");
    for (first = 0; first < BH_DOC_DRAM_BANKS; first++) {
        for (second = first + 1u; second < BH_DOC_DRAM_BANKS; second++) {
            ft_tt_grid_info grid = make_grid(0u, BH_DOC_ALL_BANKS_MASK & ~(1u << first) &
                                                     ~(1u << second));

            if (ft_tt_coord_check_grid(&grid) != FT_TT_EIO ||
                ft_tt_coord_dram_bank_count(&grid) != 0u)
                accepted_pairs++;
        }
    }
    check_u32(accepted_pairs, 0u, "all 28 two-bank harvests are refused and count 0 banks");

    printf("the doc's p100: x 5 + 11 and D2 fused, no Ethernet\n");
    check_rc(ft_tt_coord_check_grid(&p100), FT_TT_OK, "p100 grid is accepted");
    check_u32(ft_tt_coord_tensix_core_count(&p100), 120u, "p100 Tensix cores");
    check_u32(ft_tt_coord_dram_bank_count(&p100), 7u, "p100 DRAM banks (28 of 32 GiB)");
    check_u32(ft_tt_coord_eth_channel_count(&p100), 0u, "p100 Ethernet channels");
}

int main(void)
{
    test_harvest_mask_round_trips();
    test_single_column_harvests();
    test_p100_doc_numbering();
    test_multi_column_harvests();
    test_dram_harvests();

    printf("\ntest_harvesting: %d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}
