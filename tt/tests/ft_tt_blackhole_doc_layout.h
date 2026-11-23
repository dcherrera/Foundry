/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_blackhole_doc_layout.h
 * @brief The Blackhole NoC layout as the public tt-isa-documentation states it, written
 *   out independently of ft_tt_arch_blackhole.c so the coordinate tests compare two
 *   separate readings of the docs.
 *
 * Sources (paths relative to the tt-isa-documentation root):
 *   [COORD] BlackholeA0/NoC/Coordinates.md — translated coordinate table and the
 *           harvested-tile placement rules
 *   [LUA]   Diagrams/Src/NoC.lua — TileLabel_BH (what sits at each NoC #0 node),
 *           NocSize_Arch.bh, and the bh_y211 / bh_y1223 / bh_y25 / bh_x8 examples
 * Values with no public page are the numbers given in the clean-room spec and say so.
 *
 * Header-only, every helper `static inline` so a test that uses part of it builds
 * warning-free.
 *
 * @status Used by tests/test_coord.c and tests/test_harvesting.c.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_BLACKHOLE_DOC_LAYOUT_H
#define FT_TT_BLACKHOLE_DOC_LAYOUT_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tt_coord.h"

/* [LUA] NocSize_Arch.bh = { w = 17, h = 12 }. */
#define BH_DOC_NOC_WIDTH  17u
#define BH_DOC_NOC_HEIGHT 12u

/* [COORD] Tensix at x 1-7 and 10-16, y 2-11: 14 columns x 10 rows. */
#define BH_DOC_TENSIX_COLUMNS   14u
#define BH_DOC_TENSIX_ROWS      10u
#define BH_DOC_FIRST_TENSIX_ROW 2u
#define BH_DOC_ALL_COLUMNS_MASK 0x3FFFu

/* [COORD] eight 4 GiB banks, "same 4 GiB exposed on each" of 3 tiles. */
#define BH_DOC_DRAM_BANKS          8u
#define BH_DOC_DRAM_PORTS_PER_BANK 3u
#define BH_DOC_ALL_BANKS_MASK      0xFFu
#define BH_DOC_DRAM_ALIAS_FIRST_X  17u
#define BH_DOC_DRAM_ALIAS_FIRST_Y  12u
#define BH_DOC_DRAM_ALIAS_LAST_Y   23u

/* [LUA] TileLabel_BH row y = 1 names E0..E13. */
#define BH_DOC_ETH_CHANNELS      14u
#define BH_DOC_ALL_CHANNELS_MASK 0x3FFFu
#define BH_DOC_ETH_ALIAS_FIRST_X 20u
#define BH_DOC_ETH_ALIAS_LAST_X  31u
#define BH_DOC_ETH_ALIAS_Y       25u

/* [LUA] x = 8: CPU clusters at y 3, 5, 7, 9. */
#define BH_DOC_L2CPU_TILES     4u
#define BH_DOC_ALL_L2CPU_MASK  0xFu
#define BH_DOC_FIXED_COLUMN_X  8u

/* [COORD] "19, 24: PCI Express tile (whichever one is connected to the host)". */
#define BH_DOC_PCIE_ALIAS_X 19u
#define BH_DOC_PCIE_ALIAS_Y 24u

/** NoC #0 x of Tensix grid column `column` (columns counted west to east). */
static inline uint32_t bh_doc_tensix_noc0_x(uint32_t column)
{
    return column < 7u ? column + 1u : column + 3u;
}

/** Grid column at NoC #0 x, or BH_DOC_TENSIX_COLUMNS when x holds no Tensix. */
static inline uint32_t bh_doc_tensix_column_at_x(uint32_t x)
{
    if (x >= 1u && x <= 7u)
        return x - 1u;
    if (x >= 10u && x <= 16u)
        return x - 3u;
    return BH_DOC_TENSIX_COLUMNS;
}

/** Spec-given (no public page): telemetry bit reporting grid column `column`. */
static inline uint32_t bh_doc_tensix_telemetry_bit(uint32_t column)
{
    static const uint8_t bit_by_column[BH_DOC_TENSIX_COLUMNS] = { 0, 2, 4, 6, 8, 10, 12,
                                                                  13, 11, 9, 7, 5, 3, 1 };
    return bit_by_column[column];
}

/** ARC's enabled-columns word for a part whose `harvested_columns` (grid-column bits) are
 *  fused off. */
static inline uint32_t bh_doc_enabled_column_word(uint32_t harvested_columns)
{
    uint32_t word = 0u;
    uint32_t column;

    for (column = 0; column < BH_DOC_TENSIX_COLUMNS; column++) {
        if (!(harvested_columns & (1u << column)))
            word |= 1u << bh_doc_tensix_telemetry_bit(column);
    }
    return word;
}

/**
 * Translated x of the Tensix column at NoC #0 x `noc0_x`, following [LUA] bh_y211 and the
 * [COORD] rule: surviving columns take translated x 1, 2, ... in NoC #0 order, x = 8 and
 * x = 9 are never handed out, and fused columns take the remaining (maximal) values.
 */
static inline uint32_t bh_doc_tensix_translated_x(uint32_t noc0_x, uint32_t harvested_columns)
{
    uint32_t translated_by_noc0_x[BH_DOC_NOC_WIDTH] = { 0 };
    uint32_t next = 1u;
    uint32_t pass;
    uint32_t x;

    for (pass = 0; pass < 2u; pass++) {
        for (x = 1u; x < BH_DOC_NOC_WIDTH; x++) {
            uint32_t column = bh_doc_tensix_column_at_x(x);
            bool fused;

            if (column == BH_DOC_TENSIX_COLUMNS || translated_by_noc0_x[x] != 0u)
                continue;
            fused = (harvested_columns & (1u << column)) != 0u;
            if (fused != (pass == 1u))
                continue;
            translated_by_noc0_x[x] = next++;
            if (next == 8u)
                next = 10u;
        }
    }
    return translated_by_noc0_x[noc0_x];
}

/** Bank whose tile sits at NoC #0 (x, y) for x = 0 or 9 ([LUA] TileLabel_BH "D" labels). */
static inline uint32_t bh_doc_dram_bank_at(uint32_t x, uint32_t y)
{
    static const uint8_t bank_by_y[BH_DOC_NOC_HEIGHT] = { 0, 0, 1, 1, 2, 3, 3, 3, 2, 2, 1, 0 };
    return bank_by_y[y] + (x == 9u ? 4u : 0u);
}

/** [LUA] bh_y1223 (p100, D2 fused): translated y given to the DRAM tile at NoC #0 row y. */
static inline uint32_t bh_doc_dram_example_translated_y(uint32_t noc0_y)
{
    static const uint8_t translated_by_y[BH_DOC_NOC_HEIGHT] = { 12, 13, 15, 17, 22, 18,
                                                                20, 19, 23, 21, 16, 14 };
    return translated_by_y[noc0_y];
}

/**
 * NoC #0 row of port `port` of `bank`. Port order within a bank is the order of the
 * translated y the doc's bh_y1223 example gives that bank's tiles, so port p is the tile
 * at bank-relative translated offset p.
 */
static inline uint32_t bh_doc_dram_port_noc0_y(uint32_t bank, uint32_t port)
{
    uint32_t column_x = bank < 4u ? 0u : 9u;
    uint32_t y;

    for (y = 0; y < BH_DOC_NOC_HEIGHT; y++) {
        uint32_t rank = 0u;
        uint32_t other;

        if (bh_doc_dram_bank_at(column_x, y) != bank)
            continue;
        for (other = 0; other < BH_DOC_NOC_HEIGHT; other++) {
            if (bh_doc_dram_bank_at(column_x, other) == bank &&
                bh_doc_dram_example_translated_y(other) < bh_doc_dram_example_translated_y(y))
                rank++;
        }
        if (rank == port)
            return y;
    }
    return BH_DOC_NOC_HEIGHT;
}

/** NoC #0 x of the bank's DRAM column. */
static inline uint32_t bh_doc_dram_noc0_x(uint32_t bank)
{
    return bank < 4u ? 0u : 9u;
}

/** Ethernet channel at NoC #0 (x, 1) per [LUA] TileLabel_BH, or BH_DOC_ETH_CHANNELS. */
static inline uint32_t bh_doc_eth_channel_at_x(uint32_t x)
{
    static const uint8_t channel_by_x[BH_DOC_NOC_WIDTH] = { 14, 0, 2, 4, 6, 8, 10, 12, 14,
                                                            14, 13, 11, 9, 7, 5, 3, 1 };
    return x < BH_DOC_NOC_WIDTH ? channel_by_x[x] : BH_DOC_ETH_CHANNELS;
}

/** NoC #0 x of Ethernet channel `channel`. */
static inline uint32_t bh_doc_eth_noc0_x(uint32_t channel)
{
    uint32_t x;

    for (x = 0; x < BH_DOC_NOC_WIDTH; x++) {
        if (bh_doc_eth_channel_at_x(x) == channel)
            return x;
    }
    return BH_DOC_NOC_WIDTH;
}

/** NoC #0 y of L2CPU tile `tile` (tiles ordered by y). */
static inline uint32_t bh_doc_l2cpu_noc0_y(uint32_t tile)
{
    return 3u + 2u * tile;
}

/** Spec-given (no public page): telemetry bit of L2CPU tile `tile` (y = 3, 5, 7, 9). */
static inline uint32_t bh_doc_l2cpu_telemetry_bit(uint32_t tile)
{
    static const uint8_t bit_by_tile[BH_DOC_L2CPU_TILES] = { 0, 2, 3, 1 };
    return bit_by_tile[tile];
}

/** What sits at NoC #0 (x, y), transcribed from [LUA] TileLabel_BH. An empty label is a
 *  router-only node. */
static inline ft_tt_core_type bh_doc_noc0_tile_type(uint32_t x, uint32_t y)
{
    if (x >= BH_DOC_NOC_WIDTH || y >= BH_DOC_NOC_HEIGHT)
        return FT_TT_CORE_NONE;
    if (x == 0u || x == 9u)
        return FT_TT_CORE_DRAM;
    if (x == BH_DOC_FIXED_COLUMN_X) {
        if (y == 0u)
            return FT_TT_CORE_ARC;
        if (y == 2u)
            return FT_TT_CORE_SECURITY;
        if (y == 3u || y == 5u || y == 7u || y == 9u)
            return FT_TT_CORE_L2CPU;
        return FT_TT_CORE_ROUTER;
    }
    if (y == 1u)
        return FT_TT_CORE_ETH;
    if (y == 0u)
        return (x == 2u || x == 11u) ? FT_TT_CORE_PCIE : FT_TT_CORE_ROUTER;
    return FT_TT_CORE_TENSIX;
}

#endif /* FT_TT_BLACKHOLE_DOC_LAYOUT_H */
