/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file test_coord.c
 * @brief Off-hardware coordinate oracle for ft_tt_coord on an unharvested p150 grid:
 *   every Tensix, DRAM port, Ethernet channel, ARC, PCIe and L2CPU position in NoC #0
 *   and translated space, classify / is_enabled over the whole grid and the alias
 *   ranges, argument refusals, and the refused-grid rule.
 *
 * Every expectation comes from ft_tt_blackhole_doc_layout.h (the public docs) or from a
 * table in this file citing the doc diagram it was read from — never from
 * ft_tt_arch_blackhole.c. Passing checks are silent; each failure prints what it saw.
 *
 * Grid conventions this file assumes (no public page states them): the GDDR word has
 * bit b = bank b, the Ethernet word bit c = channel Ec.
 *
 * @status Covers every public call of ft_tt_coord.h except
 *   ft_tt_coord_check_kernel_build_assumptions (a build constant, not a layout).
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

/* [LUA] bh_y25 ("one possible numbering for a p150 board"): E5 and E8 are fused. */
#define P150_FUSED_ETH_MASK ((1u << 5) | (1u << 8))
#define P150_ENABLED_ETH    (BH_DOC_ALL_CHANNELS_MASK & ~P150_FUSED_ETH_MASK)
#define P150_ETH_CHANNELS   12u

/* Probe range past the 17 x 12 grid on both axes, to reach the alias ranges. */
#define PROBE_WIDTH  34u
#define PROBE_HEIGHT 32u

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

/** One coordinate lookup: rc must be OK and the coordinate must be (want_x, want_y). */
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

static void check_type_at(ft_tt_core_type got, ft_tt_core_type want, const char *what,
                          uint32_t x, uint32_t y)
{
    checks_run++;
    if (got != want) {
        checks_failed++;
        printf("  FAIL  %s (%u,%u) = %s (expected %s)\n", what, x, y, ft_tt_core_type_string(got),
               ft_tt_core_type_string(want));
    }
}

static void check_enabled_at(bool got, bool want, const char *what, uint32_t x, uint32_t y)
{
    checks_run++;
    if (got != want) {
        checks_failed++;
        printf("  FAIL  %s (%u,%u) enabled = %d (expected %d)\n", what, x, y, got, want);
    }
}

/** A filled-in Blackhole grid. `harvested_columns` is in grid-column order. */
static ft_tt_grid_info make_grid(uint32_t harvested_columns, uint32_t enabled_gddr,
                                 uint32_t enabled_eth, uint32_t enabled_l2cpu, bool translation)
{
    ft_tt_grid_info grid = { 0 };

    grid.valid = true;
    grid.arch = ft_tt_arch_blackhole();
    grid.tensix_enabled_col = bh_doc_enabled_column_word(harvested_columns);
    grid.enabled_eth = enabled_eth;
    grid.enabled_gddr = enabled_gddr;
    grid.enabled_l2cpu = enabled_l2cpu;
    grid.noc_translation_enabled = translation ? 1u : 0u;
    return grid;
}

/** The p150 under test: nothing harvested but the two Ethernet tiles of the doc example. */
static ft_tt_grid_info make_p150_grid(bool translation)
{
    return make_grid(0u, BH_DOC_ALL_BANKS_MASK, P150_ENABLED_ETH, BH_DOC_ALL_L2CPU_MASK,
                     translation);
}

/* ------------------------------------------------------------------------- */
/* Counts                                                                    */
/* ------------------------------------------------------------------------- */

static void test_counts_and_noc_size(void)
{
    ft_tt_grid_info grid = make_p150_grid(true);
    uint32_t size_x = 0u;
    uint32_t size_y = 0u;

    printf("p150 counts and NoC size\n");
    check_rc(ft_tt_coord_check_grid(&grid), FT_TT_OK, "the p150 grid is accepted");
    check_u32(ft_tt_coord_tensix_column_count(&grid), BH_DOC_TENSIX_COLUMNS, "Tensix columns");
    check_u32(ft_tt_coord_tensix_row_count(&grid), BH_DOC_TENSIX_ROWS, "Tensix rows");
    check_u32(ft_tt_coord_tensix_core_count(&grid), 140u, "Tensix cores ([COORD] 70+50+20)");
    check_u32(ft_tt_coord_dram_bank_count(&grid), BH_DOC_DRAM_BANKS, "DRAM banks");
    check_u32(ft_tt_coord_dram_ports_per_bank(&grid), BH_DOC_DRAM_PORTS_PER_BANK,
              "DRAM ports per bank");
    check_u32(ft_tt_coord_eth_channel_count(&grid), P150_ETH_CHANNELS, "Ethernet channels");
    check_u32(ft_tt_coord_l2cpu_count(&grid), BH_DOC_L2CPU_TILES, "L2CPU tiles");
    check_rc(ft_tt_coord_noc_size(&grid, &size_x, &size_y), FT_TT_OK, "noc_size");
    check_u32(size_x, BH_DOC_NOC_WIDTH, "NoC width");
    check_u32(size_y, BH_DOC_NOC_HEIGHT, "NoC height");
}

/* ------------------------------------------------------------------------- */
/* Tensix                                                                    */
/* ------------------------------------------------------------------------- */

static void test_tensix_positions(void)
{
    ft_tt_grid_info grid = make_p150_grid(true);
    ft_tt_noc_coord coord = { 0u, 0u };
    uint32_t column;
    uint32_t row;
    uint32_t index;

    printf("Tensix: all 140 cores by (x, y) and by index, both systems\n");
    for (column = 0; column < BH_DOC_TENSIX_COLUMNS; column++) {
        uint32_t noc0_x = bh_doc_tensix_noc0_x(column);
        uint32_t translated_x = bh_doc_tensix_translated_x(noc0_x, 0u);

        for (row = 0; row < BH_DOC_TENSIX_ROWS; row++) {
            uint32_t y = BH_DOC_FIRST_TENSIX_ROW + row;
            int rc = ft_tt_coord_tensix_at(&grid, column, row, FT_TT_COORD_NOC0, &coord);

            check_coord(rc, coord, noc0_x, y, "tensix_at NOC0", column * 100u + row);
            rc = ft_tt_coord_tensix_at(&grid, column, row, FT_TT_COORD_TRANSLATED, &coord);
            check_coord(rc, coord, translated_x, y, "tensix_at TRANSLATED", column * 100u + row);
        }
    }
    check_u32(bh_doc_tensix_translated_x(10u, 0u), 10u,
              "doc: unharvested translated x is the NoC #0 x (x = 10 stays 10)");

    for (index = 0; index < 140u; index++) {
        uint32_t noc0_x = bh_doc_tensix_noc0_x(index % BH_DOC_TENSIX_COLUMNS);
        uint32_t y = BH_DOC_FIRST_TENSIX_ROW + index / BH_DOC_TENSIX_COLUMNS;
        int rc = ft_tt_coord_tensix_by_index(&grid, index, FT_TT_COORD_NOC0, &coord);

        check_coord(rc, coord, noc0_x, y, "tensix_by_index NOC0", index);
        rc = ft_tt_coord_tensix_by_index(&grid, index, FT_TT_COORD_TRANSLATED, &coord);
        check_coord(rc, coord, bh_doc_tensix_translated_x(noc0_x, 0u), y,
                    "tensix_by_index TRANSLATED", index);
    }

    check_rc(ft_tt_coord_tensix_at(&grid, BH_DOC_TENSIX_COLUMNS, 0u, FT_TT_COORD_NOC0, &coord),
             FT_TT_EINVAL, "tensix_at column 14 is out of range");
    check_rc(ft_tt_coord_tensix_at(&grid, 0u, BH_DOC_TENSIX_ROWS, FT_TT_COORD_NOC0, &coord),
             FT_TT_EINVAL, "tensix_at row 10 is out of range");
    check_rc(ft_tt_coord_tensix_by_index(&grid, 140u, FT_TT_COORD_TRANSLATED, &coord),
             FT_TT_EINVAL, "tensix_by_index 140 is out of range");
}

/* ------------------------------------------------------------------------- */
/* DRAM                                                                      */
/* ------------------------------------------------------------------------- */

/*
 * Translated: spec-given, banks 0-3 (x = 0) form alias column 17 and banks 4-7 (x = 9)
 * column 18; [COORD] gives each bank 3 consecutive y from 12. Within a column the banks
 * are in bank order — the D2 example only reorders because D2 is fused — and for banks
 * 0, 1, 4 and 5 (unmoved by that example) the exact doc y is checked as well.
 */
static void test_dram_positions(void)
{
    ft_tt_grid_info grid = make_p150_grid(true);
    ft_tt_noc_coord coord = { 0u, 0u };
    uint32_t bank;
    uint32_t port;

    printf("DRAM: 8 banks x 3 ports, both systems\n");
    for (bank = 0; bank < BH_DOC_DRAM_BANKS; bank++) {
        for (port = 0; port < BH_DOC_DRAM_PORTS_PER_BANK; port++) {
            uint32_t noc0_y = bh_doc_dram_port_noc0_y(bank, port);
            uint32_t alias_x = BH_DOC_DRAM_ALIAS_FIRST_X + bank / 4u;
            uint32_t alias_y = BH_DOC_DRAM_ALIAS_FIRST_Y + (bank % 4u) * 3u + port;
            uint32_t slot = bank * 10u + port;
            int rc = ft_tt_coord_dram_port(&grid, bank, port, FT_TT_COORD_NOC0, &coord);

            check_coord(rc, coord, bh_doc_dram_noc0_x(bank), noc0_y, "dram_port NOC0", slot);
            rc = ft_tt_coord_dram_port(&grid, bank, port, FT_TT_COORD_TRANSLATED, &coord);
            check_coord(rc, coord, alias_x, alias_y, "dram_port TRANSLATED", slot);
            if (bank % 4u < 2u)
                check_u32(alias_y, bh_doc_dram_example_translated_y(noc0_y),
                          "doc bh_y1223: unmoved bank keeps the example's translated y");
        }
        check_rc(ft_tt_coord_dram_bank(&grid, bank, FT_TT_COORD_NOC0, &coord), FT_TT_OK,
                 "dram_bank NOC0");
        check_u32(coord.y, bh_doc_dram_port_noc0_y(bank, 0u), "dram_bank is port 0 (NOC0 y)");
        check_rc(ft_tt_coord_dram_bank(&grid, bank, FT_TT_COORD_TRANSLATED, &coord), FT_TT_OK,
                 "dram_bank TRANSLATED");
        check_u32(coord.y, BH_DOC_DRAM_ALIAS_FIRST_Y + (bank % 4u) * 3u,
                  "dram_bank is port 0 (TRANSLATED y)");
    }
    check_rc(ft_tt_coord_dram_port(&grid, 0u, BH_DOC_DRAM_PORTS_PER_BANK, FT_TT_COORD_NOC0,
                                   &coord),
             FT_TT_EINVAL, "dram_port port 3 is out of range");
    check_rc(ft_tt_coord_dram_port(&grid, BH_DOC_DRAM_BANKS, 0u, FT_TT_COORD_NOC0, &coord),
             FT_TT_EINVAL, "dram_port bank 8 is out of range");
    check_rc(ft_tt_coord_dram_bank(&grid, BH_DOC_DRAM_BANKS, FT_TT_COORD_TRANSLATED, &coord),
             FT_TT_EINVAL, "dram_bank bank 8 is out of range");
}

/* ------------------------------------------------------------------------- */
/* Ethernet                                                                  */
/* ------------------------------------------------------------------------- */

/** [LUA] bh_y25: translated x the p150 example gives the Ethernet tile at NoC #0 x
 *  (0 = no translated position: x 0, 8, 9, and the fused E8 at x 5 and E5 at x 14). */
static uint32_t doc_p150_eth_translated_x(uint32_t noc0_x)
{
    static const uint8_t translated_by_x[BH_DOC_NOC_WIDTH] = { 0, 20, 22, 24, 25, 0, 28, 30, 0,
                                                               0, 31, 29, 27, 26, 0, 23, 21 };
    return translated_by_x[noc0_x];
}

static void test_eth_positions(void)
{
    ft_tt_grid_info grid = make_p150_grid(true);
    ft_tt_grid_info all_eth = make_grid(0u, BH_DOC_ALL_BANKS_MASK, BH_DOC_ALL_CHANNELS_MASK,
                                        BH_DOC_ALL_L2CPU_MASK, true);
    ft_tt_noc_coord coord = { 0u, 0u };
    uint32_t logical = 0u;
    uint32_t channel;

    printf("Ethernet: the doc's p150 numbering, and all 14 channels in NoC #0\n");
    for (channel = 0; channel < BH_DOC_ETH_CHANNELS; channel++) {
        uint32_t noc0_x = bh_doc_eth_noc0_x(channel);
        int rc = ft_tt_coord_eth_channel(&all_eth, channel, FT_TT_COORD_NOC0, &coord);

        check_coord(rc, coord, noc0_x, 1u, "eth_channel NOC0 (all 14 enabled)", channel);
        if (P150_FUSED_ETH_MASK & (1u << channel))
            continue;
        rc = ft_tt_coord_eth_channel(&grid, logical, FT_TT_COORD_NOC0, &coord);
        check_coord(rc, coord, noc0_x, 1u, "eth_channel NOC0 (p150)", logical);
        rc = ft_tt_coord_eth_channel(&grid, logical, FT_TT_COORD_TRANSLATED, &coord);
        check_coord(rc, coord, doc_p150_eth_translated_x(noc0_x), BH_DOC_ETH_ALIAS_Y,
                    "eth_channel TRANSLATED (doc bh_y25)", logical);
        logical++;
    }
    check_u32(logical, P150_ETH_CHANNELS, "the doc example leaves 12 channels");
    check_rc(ft_tt_coord_eth_channel(&grid, P150_ETH_CHANNELS, FT_TT_COORD_NOC0, &coord),
             FT_TT_EINVAL, "eth_channel 12 is out of range on the p150");
    check_rc(ft_tt_coord_eth_channel(&all_eth, BH_DOC_ETH_CHANNELS, FT_TT_COORD_NOC0, &coord),
             FT_TT_EINVAL, "eth_channel 14 is out of range with all enabled");
}

/* ------------------------------------------------------------------------- */
/* ARC, PCIe, L2CPU                                                          */
/* ------------------------------------------------------------------------- */

static void test_arc_and_pcie(void)
{
    const ft_tt_arch *arch = ft_tt_arch_blackhole();
    ft_tt_noc_coord coord = { 0u, 0u };

    printf("ARC and PCIe\n");
    /* [LUA] TileLabel_BH (8, 0) "ARC"; [LUA] bh_x8 maps it to translated (8, 0). */
    check_coord(ft_tt_coord_arc(arch, FT_TT_COORD_NOC0, &coord), coord, 8u, 0u, "arc NOC0", 0u);
    check_coord(ft_tt_coord_arc(arch, FT_TT_COORD_TRANSLATED, &coord), coord, 8u, 0u,
                "arc TRANSLATED", 0u);
    check_rc(ft_tt_coord_arc(NULL, FT_TT_COORD_NOC0, &coord), FT_TT_EINVAL, "arc NULL arch");
    check_rc(ft_tt_coord_arc(arch, FT_TT_COORD_NOC0, NULL), FT_TT_EINVAL, "arc NULL out");
    check_rc(ft_tt_coord_arc(arch, (ft_tt_coord_system)2, &coord), FT_TT_EINVAL,
             "arc unknown system");

    /* [LUA] TileLabel_BH row 0: "PCIe 0" at x 2, "PCIe 1" at x 11. */
    check_coord(ft_tt_coord_pcie(arch, 0u, FT_TT_COORD_NOC0, &coord), coord, 2u, 0u,
                "pcie NOC0", 0u);
    check_coord(ft_tt_coord_pcie(arch, 1u, FT_TT_COORD_NOC0, &coord), coord, 11u, 0u,
                "pcie NOC0", 1u);
    check_rc(ft_tt_coord_pcie(arch, 2u, FT_TT_COORD_NOC0, &coord), FT_TT_EINVAL,
             "pcie index 2 is out of range");
    /* [COORD]: which tile owns (19, 24) "will depend on the particular product". */
    check_rc(ft_tt_coord_pcie(arch, 0u, FT_TT_COORD_TRANSLATED, &coord), FT_TT_EUNAVAIL,
             "pcie TRANSLATED refuses: the board decides which tile is aliased");
    check_rc(ft_tt_coord_pcie(NULL, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EINVAL,
             "pcie NULL arch");
    check_rc(ft_tt_coord_pcie(arch, 0u, FT_TT_COORD_NOC0, NULL), FT_TT_EINVAL, "pcie NULL out");
}

/** L2CPU tiles with tile `fused_tile` reported disabled (BH_DOC_L2CPU_TILES = none). */
static void check_l2cpu_layout(uint32_t fused_tile)
{
    uint32_t enabled = BH_DOC_ALL_L2CPU_MASK;
    ft_tt_grid_info grid;
    ft_tt_noc_coord coord = { 0u, 0u };
    uint32_t logical = 0u;
    uint32_t tile;

    if (fused_tile < BH_DOC_L2CPU_TILES)
        enabled &= ~(1u << bh_doc_l2cpu_telemetry_bit(fused_tile));
    grid = make_grid(0u, BH_DOC_ALL_BANKS_MASK, P150_ENABLED_ETH, enabled, true);
    for (tile = 0; tile < BH_DOC_L2CPU_TILES; tile++) {
        uint32_t y = bh_doc_l2cpu_noc0_y(tile);

        check_enabled_at(ft_tt_coord_is_enabled(&grid, FT_TT_COORD_NOC0, 8u, y),
                         tile != fused_tile, "l2cpu tile", 8u, y);
        if (tile == fused_tile)
            continue;
        check_coord(ft_tt_coord_l2cpu(&grid, logical, FT_TT_COORD_NOC0, &coord), coord, 8u, y,
                    "l2cpu NOC0", logical);
        logical++;
    }
    check_u32(ft_tt_coord_l2cpu_count(&grid), logical, "l2cpu_count");
    check_rc(ft_tt_coord_l2cpu(&grid, logical, FT_TT_COORD_NOC0, &coord), FT_TT_EINVAL,
             "l2cpu past the enabled tiles");
}

static void test_l2cpu_positions(void)
{
    ft_tt_grid_info grid = make_p150_grid(true);
    ft_tt_noc_coord coord = { 0u, 0u };
    uint32_t fused;

    printf("L2CPU: the four clusters, and each one fused by its telemetry bit\n");
    for (fused = 0; fused <= BH_DOC_L2CPU_TILES; fused++)
        check_l2cpu_layout(fused);
    /* ft_tt_coord.h contract: no L2CPU alias positions are modelled. */
    check_rc(ft_tt_coord_l2cpu(&grid, 0u, FT_TT_COORD_TRANSLATED, &coord), FT_TT_EUNAVAIL,
             "l2cpu TRANSLATED with translation on refuses");
}

/* ------------------------------------------------------------------------- */
/* Classification                                                            */
/* ------------------------------------------------------------------------- */

static bool doc_p150_noc0_enabled(uint32_t x, uint32_t y)
{
    ft_tt_core_type type = bh_doc_noc0_tile_type(x, y);

    if (type == FT_TT_CORE_NONE)
        return false;
    if (type == FT_TT_CORE_ETH)
        return !(P150_FUSED_ETH_MASK & (1u << bh_doc_eth_channel_at_x(x)));
    return true;
}

static void test_classify_noc0(void)
{
    ft_tt_grid_info grid = make_p150_grid(true);
    uint32_t x;
    uint32_t y;

    printf("classify / is_enabled over every NoC #0 node (and past the edge)\n");
    for (y = 0; y < PROBE_HEIGHT; y++) {
        for (x = 0; x < PROBE_WIDTH; x++) {
            check_type_at(ft_tt_coord_classify(&grid, FT_TT_COORD_NOC0, x, y),
                          bh_doc_noc0_tile_type(x, y), "classify NOC0", x, y);
            check_enabled_at(ft_tt_coord_is_enabled(&grid, FT_TT_COORD_NOC0, x, y),
                             doc_p150_noc0_enabled(x, y), "is_enabled NOC0", x, y);
        }
    }
}

/**
 * What [COORD] puts at translated (x, y) on the p150. Inside the grid every node but
 * x = 8 at 2 <= y < 12 (no doc entry) keeps its NoC #0 identity — nothing is fused on
 * this grid, so the Tensix renumbering is the identity. Returns false (no expectation)
 * for nodes this test does not assert.
 */
static bool doc_p150_translated_type(uint32_t x, uint32_t y, ft_tt_core_type *out_type)
{
    bool in_dram_alias = (x == 17u || x == 18u) && y >= BH_DOC_DRAM_ALIAS_FIRST_Y &&
                         y <= BH_DOC_DRAM_ALIAS_LAST_Y;
    bool in_eth_alias = y == BH_DOC_ETH_ALIAS_Y && x >= BH_DOC_ETH_ALIAS_FIRST_X &&
                        x <= BH_DOC_ETH_ALIAS_LAST_X;

    if (x < BH_DOC_NOC_WIDTH && y < BH_DOC_NOC_HEIGHT) {
        if (x == BH_DOC_FIXED_COLUMN_X && y >= BH_DOC_FIRST_TENSIX_ROW)
            return false;
        *out_type = bh_doc_noc0_tile_type(x, y);
        return true;
    }
    if (x == BH_DOC_FIXED_COLUMN_X)
        return false; /* [COORD] 8, 26-30: L2CPU / security aliases, see the report */
    if (y == BH_DOC_ETH_ALIAS_Y && x > BH_DOC_ETH_ALIAS_LAST_X)
        return false; /* no doc entry past x = 31; only is_enabled is asserted there */
    if (in_dram_alias)
        *out_type = FT_TT_CORE_DRAM;
    else if (in_eth_alias)
        *out_type = FT_TT_CORE_ETH;
    else if (x == BH_DOC_PCIE_ALIAS_X && y == BH_DOC_PCIE_ALIAS_Y)
        *out_type = FT_TT_CORE_PCIE;
    else
        *out_type = FT_TT_CORE_NONE;
    return true;
}

static void test_classify_translated(void)
{
    ft_tt_grid_info grid = make_p150_grid(true);
    ft_tt_core_type want = FT_TT_CORE_NONE;
    uint32_t x;
    uint32_t y;

    printf("classify / is_enabled over translated space, alias ranges included\n");
    for (y = 0; y < PROBE_HEIGHT; y++) {
        for (x = 0; x < PROBE_WIDTH; x++) {
            bool want_enabled;

            if (!doc_p150_translated_type(x, y, &want)) {
                if (y == BH_DOC_ETH_ALIAS_Y && x > BH_DOC_ETH_ALIAS_LAST_X)
                    check_enabled_at(ft_tt_coord_is_enabled(&grid, FT_TT_COORD_TRANSLATED, x, y),
                                     false, "is_enabled TRANSLATED past the ETH alias row", x, y);
                continue;
            }
            want_enabled = x < BH_DOC_NOC_WIDTH && y < BH_DOC_NOC_HEIGHT
                               ? doc_p150_noc0_enabled(x, y)
                               : want != FT_TT_CORE_NONE;
            check_type_at(ft_tt_coord_classify(&grid, FT_TT_COORD_TRANSLATED, x, y), want,
                          "classify TRANSLATED", x, y);
            check_enabled_at(ft_tt_coord_is_enabled(&grid, FT_TT_COORD_TRANSLATED, x, y),
                             want_enabled, "is_enabled TRANSLATED", x, y);
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Translation disabled                                                      */
/* ------------------------------------------------------------------------- */

static void test_translation_disabled(void)
{
    ft_tt_grid_info grid = make_p150_grid(false);
    ft_tt_noc_coord translated = { 0u, 0u };
    uint32_t index;

    printf("translation reported off: TRANSLATED collapses onto NOC0\n");
    for (index = 0; index < 140u; index++) {
        check_coord(ft_tt_coord_tensix_by_index(&grid, index, FT_TT_COORD_TRANSLATED,
                                                &translated),
                    translated, bh_doc_tensix_noc0_x(index % BH_DOC_TENSIX_COLUMNS),
                    BH_DOC_FIRST_TENSIX_ROW + index / BH_DOC_TENSIX_COLUMNS,
                    "tensix TRANSLATED == NOC0", index);
    }
    for (index = 0; index < BH_DOC_DRAM_BANKS; index++) {
        check_coord(ft_tt_coord_dram_port(&grid, index, 2u, FT_TT_COORD_TRANSLATED, &translated),
                    translated, bh_doc_dram_noc0_x(index), bh_doc_dram_port_noc0_y(index, 2u),
                    "dram TRANSLATED == NOC0", index);
    }
    check_coord(ft_tt_coord_eth_channel(&grid, 0u, FT_TT_COORD_TRANSLATED, &translated),
                translated, bh_doc_eth_noc0_x(0u), 1u, "eth TRANSLATED == NOC0", 0u);
    check_coord(ft_tt_coord_l2cpu(&grid, 0u, FT_TT_COORD_TRANSLATED, &translated), translated,
                8u, 3u, "l2cpu TRANSLATED == NOC0", 0u);
    check_type_at(ft_tt_coord_classify(&grid, FT_TT_COORD_TRANSLATED, 17u, 12u), FT_TT_CORE_NONE,
                  "no DRAM alias without translation", 17u, 12u);
    check_type_at(ft_tt_coord_classify(&grid, FT_TT_COORD_TRANSLATED, 20u, 25u), FT_TT_CORE_NONE,
                  "no ETH alias without translation", 20u, 25u);
    check_enabled_at(ft_tt_coord_is_enabled(&grid, FT_TT_COORD_TRANSLATED, 19u, 24u), false,
                     "no PCIe alias without translation", 19u, 24u);
}

/* ------------------------------------------------------------------------- */
/* Argument refusals                                                         */
/* ------------------------------------------------------------------------- */

static void test_invalid_arguments(void)
{
    ft_tt_grid_info grid = make_p150_grid(true);
    ft_tt_grid_info unfilled = grid;
    ft_tt_grid_info no_arch = grid;
    ft_tt_noc_coord coord = { 0u, 0u };
    uint32_t size = 0u;
    uint32_t mask = 0u;
    const ft_tt_coord_system bad_system = (ft_tt_coord_system)2;

    printf("argument refusals\n");
    unfilled.valid = false;
    no_arch.arch = NULL;
    check_rc(ft_tt_coord_check_grid(NULL), FT_TT_EINVAL, "check_grid NULL");
    check_rc(ft_tt_coord_check_grid(&unfilled), FT_TT_EUNAVAIL, "check_grid unfilled");
    check_rc(ft_tt_coord_check_grid(&no_arch), FT_TT_EUNAVAIL, "check_grid with no arch");
    check_rc(ft_tt_coord_tensix_at(NULL, 0u, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EINVAL,
             "tensix_at NULL grid");
    check_rc(ft_tt_coord_tensix_at(&unfilled, 0u, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EUNAVAIL,
             "tensix_at unfilled grid");
    check_rc(ft_tt_coord_tensix_at(&no_arch, 0u, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EUNAVAIL,
             "tensix_at grid with no arch");
    check_rc(ft_tt_coord_tensix_at(&grid, 0u, 0u, FT_TT_COORD_NOC0, NULL), FT_TT_EINVAL,
             "tensix_at NULL out");
    check_rc(ft_tt_coord_tensix_at(&grid, 0u, 0u, bad_system, &coord), FT_TT_EINVAL,
             "tensix_at unknown system");
    check_rc(ft_tt_coord_tensix_by_index(&grid, 0u, bad_system, &coord), FT_TT_EINVAL,
             "tensix_by_index unknown system");
    check_rc(ft_tt_coord_dram_port(&grid, 0u, 0u, FT_TT_COORD_NOC0, NULL), FT_TT_EINVAL,
             "dram_port NULL out");
    check_rc(ft_tt_coord_dram_port(NULL, 0u, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EINVAL,
             "dram_port NULL grid");
    check_rc(ft_tt_coord_eth_channel(&grid, 0u, bad_system, &coord), FT_TT_EINVAL,
             "eth_channel unknown system");
    check_rc(ft_tt_coord_l2cpu(&unfilled, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EUNAVAIL,
             "l2cpu unfilled grid");
    check_rc(ft_tt_coord_noc_size(&grid, NULL, &size), FT_TT_EINVAL, "noc_size NULL x");
    check_rc(ft_tt_coord_noc_size(&grid, &size, NULL), FT_TT_EINVAL, "noc_size NULL y");
    check_rc(ft_tt_coord_noc_size(NULL, &size, &size), FT_TT_EINVAL, "noc_size NULL grid");
    check_rc(ft_tt_coord_tensix_harvest_mask(NULL, 0u, &mask), FT_TT_EINVAL,
             "harvest_mask NULL arch");
    check_rc(ft_tt_coord_tensix_harvest_mask(grid.arch, 0u, NULL), FT_TT_EINVAL,
             "harvest_mask NULL out");
    check_u32(ft_tt_coord_tensix_core_count(NULL), 0u, "core_count NULL grid");
    check_u32(ft_tt_coord_dram_bank_count(&unfilled), 0u, "dram_bank_count unfilled grid");
    check_type_at(ft_tt_coord_classify(NULL, FT_TT_COORD_NOC0, 8u, 0u), FT_TT_CORE_NONE,
                  "classify NULL grid", 8u, 0u);
    check_type_at(ft_tt_coord_classify(&grid, bad_system, 8u, 0u), FT_TT_CORE_NONE,
                  "classify unknown system", 8u, 0u);
    check_enabled_at(ft_tt_coord_is_enabled(&no_arch, FT_TT_COORD_NOC0, 8u, 0u), false,
                     "is_enabled grid with no arch", 8u, 0u);
    check(ft_tt_core_type_string((ft_tt_core_type)99) != NULL,
          "core_type_string never returns NULL");
}

/* ------------------------------------------------------------------------- */
/* Refused grids                                                             */
/* ------------------------------------------------------------------------- */

/** A grid ft_tt_coord_check_grid refuses must answer nothing. */
static void check_refused_grid(const ft_tt_grid_info *grid, const char *what)
{
    ft_tt_noc_coord coord = { 0u, 0u };
    uint32_t size = 0u;

    printf("refused grid: %s\n", what);
    check_rc(ft_tt_coord_check_grid(grid), FT_TT_EIO, "check_grid refuses");
    check_u32(ft_tt_coord_tensix_column_count(grid), 0u, "tensix_column_count");
    check_u32(ft_tt_coord_tensix_row_count(grid), 0u, "tensix_row_count");
    check_u32(ft_tt_coord_tensix_core_count(grid), 0u, "tensix_core_count");
    check_u32(ft_tt_coord_dram_bank_count(grid), 0u, "dram_bank_count");
    check_u32(ft_tt_coord_dram_ports_per_bank(grid), 0u, "dram_ports_per_bank");
    check_u32(ft_tt_coord_eth_channel_count(grid), 0u, "eth_channel_count");
    check_u32(ft_tt_coord_l2cpu_count(grid), 0u, "l2cpu_count");
    check_rc(ft_tt_coord_noc_size(grid, &size, &size), FT_TT_EIO, "noc_size");
    check_rc(ft_tt_coord_tensix_at(grid, 0u, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EIO,
             "tensix_at");
    check_rc(ft_tt_coord_tensix_by_index(grid, 0u, FT_TT_COORD_TRANSLATED, &coord), FT_TT_EIO,
             "tensix_by_index");
    check_rc(ft_tt_coord_dram_port(grid, 0u, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EIO,
             "dram_port");
    check_rc(ft_tt_coord_dram_bank(grid, 0u, FT_TT_COORD_TRANSLATED, &coord), FT_TT_EIO,
             "dram_bank");
    check_rc(ft_tt_coord_eth_channel(grid, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EIO,
             "eth_channel");
    check_rc(ft_tt_coord_l2cpu(grid, 0u, FT_TT_COORD_NOC0, &coord), FT_TT_EIO, "l2cpu");
    check_type_at(ft_tt_coord_classify(grid, FT_TT_COORD_NOC0, 8u, 0u), FT_TT_CORE_NONE,
                  "classify ARC", 8u, 0u);
    check_type_at(ft_tt_coord_classify(grid, FT_TT_COORD_NOC0, 1u, 2u), FT_TT_CORE_NONE,
                  "classify Tensix", 1u, 2u);
    check_type_at(ft_tt_coord_classify(grid, FT_TT_COORD_TRANSLATED, 17u, 12u), FT_TT_CORE_NONE,
                  "classify DRAM alias", 17u, 12u);
    check_enabled_at(ft_tt_coord_is_enabled(grid, FT_TT_COORD_NOC0, 8u, 0u), false,
                     "is_enabled ARC", 8u, 0u);
    check_enabled_at(ft_tt_coord_is_enabled(grid, FT_TT_COORD_TRANSLATED, 19u, 24u), false,
                     "is_enabled PCIe alias", 19u, 24u);
}

static void test_refused_grids(void)
{
    ft_tt_grid_info no_tensix = make_p150_grid(true);
    ft_tt_grid_info two_banks_fused = make_p150_grid(true);
    ft_tt_grid_info one_bank_fused = make_p150_grid(true);

    no_tensix.tensix_enabled_col = 0u;
    two_banks_fused.enabled_gddr = BH_DOC_ALL_BANKS_MASK & ~((1u << 2) | (1u << 6));
    one_bank_fused.enabled_gddr = BH_DOC_ALL_BANKS_MASK & ~(1u << 2);
    check_refused_grid(&no_tensix, "no Tensix column enabled");
    check_refused_grid(&two_banks_fused, "two DRAM banks harvested");
    printf("boundary: one harvested bank is allowed\n");
    check_rc(ft_tt_coord_check_grid(&one_bank_fused), FT_TT_OK, "one bank fused is accepted");
}

int main(void)
{
    test_counts_and_noc_size();
    test_tensix_positions();
    test_dram_positions();
    test_eth_positions();
    test_arc_and_pcie();
    test_l2cpu_positions();
    test_classify_noc0();
    test_classify_translated();
    test_translation_disabled();
    test_invalid_arguments();
    test_refused_grids();

    printf("\ntest_coord: %d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}
