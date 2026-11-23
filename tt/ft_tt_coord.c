/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_coord.c
 * @brief Implementation of logical <-> NoC0 / translated coordinate mapping,
 *   classification and enabled checks, driven entirely by the grid's
 *   architecture table and ARC telemetry masks.
 *
 * Translation model (public docs, BlackholeA0/NoC/Coordinates.md):
 *   - Nodes inside the NoC grid keep their raw position in translated space,
 *     EXCEPT the Tensix columns: enabled columns pack, in NoC0 order, onto the
 *     first entries of the full Tensix column list, and harvested columns take the
 *     remaining (top) entries. Rows are never renumbered.
 *   - DRAM, the in-use PCIe tile and Ethernet get alias ranges OUTSIDE the grid
 *     (origins in the arch table). Enabled Ethernet channels pack in channel order;
 *     on a part with a harvested DRAM bank the alias layout is not modelled and the
 *     translated DRAM accessors refuse.
 *   - With translation reported disabled, TRANSLATED is NOC0.
 *
 * @status Complete; covered by tests/test_coord.c, tests/test_harvesting.c and
 *   tests/test_arch.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_coord.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_memmap.h"

/* ------------------------------------------------------------------------- */
/* Grid validation and mask helpers                                          */
/* ------------------------------------------------------------------------- */

/** Resolve a grid to its architecture: EINVAL for NULL, EUNAVAIL when ARC has
 *  not filled it or it names no architecture. No shape check (check_grid's own front door). */
static int resolve_arch_only(const ft_tt_grid_info *grid, const ft_tt_arch **out_arch)
{
    if (!grid)
        return FT_TT_EINVAL;
    if (!grid->valid || !grid->arch)
        return FT_TT_EUNAVAIL;
    *out_arch = grid->arch;
    return FT_TT_OK;
}

static uint32_t enabled_tensix_columns(const ft_tt_grid_info *grid, const ft_tt_arch *arch);
static bool dram_harvest_beyond_limit(const ft_tt_grid_info *grid, const ft_tt_arch *arch);

/**
 * Resolve a grid for a lookup. A grid ft_tt_coord_check_grid() refuses answers NOTHING
 * (EIO, counts 0, classify NONE): a coordinate derived from an impossible harvest picture
 * would address a tile that may not exist, and a wrong NoC target can wedge the card.
 * Silent here; ft_tt_coord_check_grid() is the call that explains the refusal.
 */
static int resolve_grid_arch(const ft_tt_grid_info *grid, const ft_tt_arch **out_arch)
{
    const ft_tt_arch *arch = NULL;
    int rc = resolve_arch_only(grid, &arch);

    if (rc != FT_TT_OK)
        return rc;
    if (enabled_tensix_columns(grid, arch) == 0u || dram_harvest_beyond_limit(grid, arch))
        return FT_TT_EIO;
    *out_arch = arch;
    return FT_TT_OK;
}

static bool is_known_system(ft_tt_coord_system system)
{
    return system == FT_TT_COORD_NOC0 || system == FT_TT_COORD_TRANSLATED;
}

/** Shared front door for the grid-based accessors. */
static int begin_grid_lookup(const ft_tt_grid_info *grid, ft_tt_coord_system system,
                             const ft_tt_noc_coord *out_coord, const ft_tt_arch **out_arch)
{
    if (!out_coord || !is_known_system(system))
        return FT_TT_EINVAL;
    return resolve_grid_arch(grid, out_arch);
}

/** TRANSLATED only means something when firmware enabled translation. */
static bool translation_applies(const ft_tt_grid_info *grid, ft_tt_coord_system system)
{
    return system == FT_TT_COORD_TRANSLATED && grid->noc_translation_enabled != 0u;
}

static uint32_t low_bits_mask(uint32_t count)
{
    return count >= 32u ? 0xFFFFFFFFu : (1u << count) - 1u;
}

static uint32_t count_set_bits(uint32_t value)
{
    uint32_t count = 0u;

    while (value) {
        value &= value - 1u;
        count++;
    }
    return count;
}

static bool bit_is_set(uint32_t word, uint32_t bit)
{
    return bit < 32u && (word & (1u << bit)) != 0u;
}

static void store_coord(ft_tt_noc_coord *out_coord, uint32_t x, uint32_t y)
{
    out_coord->x = x;
    out_coord->y = y;
}

/* ------------------------------------------------------------------------- */
/* Enabled-unit enumeration                                                  */
/* ------------------------------------------------------------------------- */

static bool tensix_column_enabled(const ft_tt_arch *arch, uint32_t enabled_col, uint32_t column)
{
    return bit_is_set(enabled_col, arch->tensix.telemetry_bit_by_column[column]);
}

static uint32_t enabled_tensix_columns(const ft_tt_grid_info *grid, const ft_tt_arch *arch)
{
    uint32_t column;
    uint32_t count = 0u;

    for (column = 0; column < arch->tensix.column_count; column++) {
        if (tensix_column_enabled(arch, grid->tensix_enabled_col, column))
            count++;
    }
    return count;
}

/** Grid column of the `logical`-th enabled column. False when out of range. */
static bool nth_enabled_tensix_column(const ft_tt_grid_info *grid, const ft_tt_arch *arch,
                                      uint32_t logical, uint32_t *out_column)
{
    uint32_t column;

    for (column = 0; column < arch->tensix.column_count; column++) {
        if (!tensix_column_enabled(arch, grid->tensix_enabled_col, column))
            continue;
        if (logical == 0u) {
            *out_column = column;
            return true;
        }
        logical--;
    }
    return false;
}

/** Index of the `logical`-th set bit of `enabled` below `limit`. Banks and
 *  Ethernet channels are reported in their own index order. */
static bool nth_enabled_index(uint32_t enabled, uint32_t limit, uint32_t logical,
                              uint32_t *out_index)
{
    uint32_t index;

    for (index = 0; index < limit; index++) {
        if (!bit_is_set(enabled, index))
            continue;
        if (logical == 0u) {
            *out_index = index;
            return true;
        }
        logical--;
    }
    return false;
}

static uint32_t enabled_dram_banks(const ft_tt_grid_info *grid, const ft_tt_arch *arch)
{
    return count_set_bits(grid->enabled_gddr & low_bits_mask(arch->dram.bank_count));
}

static uint32_t enabled_eth_channels(const ft_tt_grid_info *grid, const ft_tt_arch *arch)
{
    return count_set_bits(grid->enabled_eth & low_bits_mask(arch->eth.channel_count));
}

static bool l2cpu_tile_enabled(const ft_tt_grid_info *grid, const ft_tt_arch *arch, uint32_t tile)
{
    return bit_is_set(grid->enabled_l2cpu, arch->fixed.l2cpu_telemetry_bit_by_tile[tile]);
}

static uint32_t enabled_l2cpu_tiles(const ft_tt_grid_info *grid, const ft_tt_arch *arch)
{
    uint32_t tile;
    uint32_t count = 0u;

    for (tile = 0; tile < arch->fixed.l2cpu_count; tile++) {
        if (l2cpu_tile_enabled(grid, arch, tile))
            count++;
    }
    return count;
}

static bool dram_harvest_beyond_limit(const ft_tt_grid_info *grid, const ft_tt_arch *arch)
{
    return arch->dram.bank_count - enabled_dram_banks(grid, arch) > arch->dram.max_harvested_banks;
}

/* ------------------------------------------------------------------------- */
/* Grid-level checks and counts                                              */
/* ------------------------------------------------------------------------- */

int ft_tt_coord_check_grid(const ft_tt_grid_info *grid)
{
    const ft_tt_arch *arch = NULL;
    int rc = resolve_arch_only(grid, &arch);

    if (rc != FT_TT_OK)
        return rc;
    if (enabled_tensix_columns(grid, arch) == 0u) {
        ft_tt_log_error("grid refused: %s reports no enabled Tensix columns", arch->name);
        return FT_TT_EIO;
    }
    if (dram_harvest_beyond_limit(grid, arch)) {
        ft_tt_log_error("grid refused: %s reports %u of %u DRAM banks enabled; at most %u may "
                        "be harvested",
                        arch->name, enabled_dram_banks(grid, arch), arch->dram.bank_count,
                        arch->dram.max_harvested_banks);
        return FT_TT_EIO;
    }
    return FT_TT_OK;
}

int ft_tt_coord_check_kernel_build_assumptions(const ft_tt_grid_info *grid)
{
    int rc = ft_tt_coord_check_grid(grid);
    uint32_t cores;
    uint32_t banks;

    if (rc != FT_TT_OK)
        return rc;
    cores = ft_tt_coord_tensix_core_count(grid);
    banks = ft_tt_coord_dram_bank_count(grid);
    if (cores != FT_TT_KERNEL_BUILD_L1_BANK_COUNT || banks != FT_TT_KERNEL_BUILD_DRAM_BANK_COUNT) {
        ft_tt_log_error("card has %u Tensix cores / %u DRAM banks, but the kernels were built "
                        "for %u / %u — refusing to run them",
                        cores, banks, FT_TT_KERNEL_BUILD_L1_BANK_COUNT,
                        FT_TT_KERNEL_BUILD_DRAM_BANK_COUNT);
        return FT_TT_EIO;
    }
    return FT_TT_OK;
}

uint32_t ft_tt_coord_tensix_column_count(const ft_tt_grid_info *grid)
{
    const ft_tt_arch *arch = NULL;

    if (resolve_grid_arch(grid, &arch) != FT_TT_OK)
        return 0u;
    return enabled_tensix_columns(grid, arch);
}

uint32_t ft_tt_coord_tensix_row_count(const ft_tt_grid_info *grid)
{
    const ft_tt_arch *arch = NULL;

    if (resolve_grid_arch(grid, &arch) != FT_TT_OK)
        return 0u;
    return arch->tensix.row_count;
}

uint32_t ft_tt_coord_tensix_core_count(const ft_tt_grid_info *grid)
{
    return ft_tt_coord_tensix_column_count(grid) * ft_tt_coord_tensix_row_count(grid);
}

uint32_t ft_tt_coord_dram_bank_count(const ft_tt_grid_info *grid)
{
    const ft_tt_arch *arch = NULL;

    if (resolve_grid_arch(grid, &arch) != FT_TT_OK)
        return 0u;
    return enabled_dram_banks(grid, arch);
}

uint32_t ft_tt_coord_dram_ports_per_bank(const ft_tt_grid_info *grid)
{
    const ft_tt_arch *arch = NULL;

    if (resolve_grid_arch(grid, &arch) != FT_TT_OK)
        return 0u;
    return arch->dram.ports_per_bank;
}

uint32_t ft_tt_coord_eth_channel_count(const ft_tt_grid_info *grid)
{
    const ft_tt_arch *arch = NULL;

    if (resolve_grid_arch(grid, &arch) != FT_TT_OK)
        return 0u;
    return enabled_eth_channels(grid, arch);
}

uint32_t ft_tt_coord_l2cpu_count(const ft_tt_grid_info *grid)
{
    const ft_tt_arch *arch = NULL;

    if (resolve_grid_arch(grid, &arch) != FT_TT_OK)
        return 0u;
    return enabled_l2cpu_tiles(grid, arch);
}

int ft_tt_coord_noc_size(const ft_tt_grid_info *grid, uint32_t *out_size_x, uint32_t *out_size_y)
{
    const ft_tt_arch *arch = NULL;
    int rc;

    if (!out_size_x || !out_size_y)
        return FT_TT_EINVAL;
    rc = resolve_grid_arch(grid, &arch);
    if (rc != FT_TT_OK)
        return rc;
    *out_size_x = arch->noc.size_x;
    *out_size_y = arch->noc.size_y;
    return FT_TT_OK;
}

int ft_tt_coord_tensix_harvest_mask(const ft_tt_arch *arch, uint32_t tensix_enabled_col,
                                    uint32_t *out_harvest_mask)
{
    uint32_t column;
    uint32_t harvested = 0u;

    if (!arch || !out_harvest_mask)
        return FT_TT_EINVAL;
    if (!arch->tensix.telemetry_bit_by_column || arch->tensix.column_count >= 32u)
        return FT_TT_EIO;

    for (column = 0; column < arch->tensix.column_count; column++) {
        if (!tensix_column_enabled(arch, tensix_enabled_col, column))
            harvested |= 1u << column;
    }
    *out_harvest_mask = harvested;
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* Per-unit coordinates                                                      */
/* ------------------------------------------------------------------------- */

int ft_tt_coord_tensix_at(const ft_tt_grid_info *grid, uint32_t logical_x, uint32_t logical_y,
                          ft_tt_coord_system system, ft_tt_noc_coord *out_coord)
{
    const ft_tt_arch *arch = NULL;
    uint32_t column = 0u;
    int rc = begin_grid_lookup(grid, system, out_coord, &arch);

    if (rc != FT_TT_OK)
        return rc;
    if (logical_y >= arch->tensix.row_count ||
        !nth_enabled_tensix_column(grid, arch, logical_x, &column))
        return FT_TT_EINVAL;

    /* Translated: survivors take the low slots of the column list in order. */
    if (translation_applies(grid, system))
        column = logical_x;
    store_coord(out_coord, arch->tensix.noc_x_by_column[column],
                arch->tensix.noc_y_by_row[logical_y]);
    return FT_TT_OK;
}

int ft_tt_coord_tensix_by_index(const ft_tt_grid_info *grid, uint32_t index,
                                ft_tt_coord_system system, ft_tt_noc_coord *out_coord)
{
    const ft_tt_arch *arch = NULL;
    uint32_t columns;
    int rc = begin_grid_lookup(grid, system, out_coord, &arch);

    if (rc != FT_TT_OK)
        return rc;
    columns = enabled_tensix_columns(grid, arch);
    if (columns == 0u)
        return FT_TT_EINVAL;
    return ft_tt_coord_tensix_at(grid, index % columns, index / columns, system, out_coord);
}

int ft_tt_coord_dram_port(const ft_tt_grid_info *grid, uint32_t logical_bank, uint32_t port,
                          ft_tt_coord_system system, ft_tt_noc_coord *out_coord)
{
    const ft_tt_arch *arch = NULL;
    const ft_tt_arch_xy *entry;
    uint32_t bank = 0u;
    uint32_t banks_per_column;
    int rc = begin_grid_lookup(grid, system, out_coord, &arch);

    if (rc != FT_TT_OK)
        return rc;
    if (arch->dram.bank_count == 0u || !arch->dram.noc0)
        return FT_TT_EUNAVAIL;
    if (dram_harvest_beyond_limit(grid, arch))
        return FT_TT_EIO;
    if (port >= arch->dram.ports_per_bank ||
        !nth_enabled_index(grid->enabled_gddr, arch->dram.bank_count, logical_bank, &bank))
        return FT_TT_EINVAL;

    if (!translation_applies(grid, system)) {
        entry = &arch->dram.noc0[bank * arch->dram.ports_per_bank + port];
        store_coord(out_coord, entry->x, entry->y);
        return FT_TT_OK;
    }

    /* The harvested-bank alias layout is not modelled: refuse, never guess. */
    if (enabled_dram_banks(grid, arch) != arch->dram.bank_count)
        return FT_TT_EUNAVAIL;
    banks_per_column = arch->dram.bank_count / arch->dram.translated_column_count;
    store_coord(out_coord, arch->dram.translated_origin_x + bank / banks_per_column,
                arch->dram.translated_origin_y +
                    (bank % banks_per_column) * arch->dram.ports_per_bank + port);
    return FT_TT_OK;
}

int ft_tt_coord_dram_bank(const ft_tt_grid_info *grid, uint32_t logical_bank,
                          ft_tt_coord_system system, ft_tt_noc_coord *out_coord)
{
    return ft_tt_coord_dram_port(grid, logical_bank, 0u, system, out_coord);
}

int ft_tt_coord_eth_channel(const ft_tt_grid_info *grid, uint32_t logical_channel,
                            ft_tt_coord_system system, ft_tt_noc_coord *out_coord)
{
    const ft_tt_arch *arch = NULL;
    uint32_t channel = 0u;
    int rc = begin_grid_lookup(grid, system, out_coord, &arch);

    if (rc != FT_TT_OK)
        return rc;
    if (!nth_enabled_index(grid->enabled_eth, arch->eth.channel_count, logical_channel,
                           &channel))
        return FT_TT_EINVAL;

    /* Translated: enabled channels pack along one row in channel order. */
    if (translation_applies(grid, system))
        store_coord(out_coord, arch->eth.translated_origin_x + logical_channel,
                    arch->eth.translated_origin_y);
    else
        store_coord(out_coord, arch->eth.noc0[channel].x, arch->eth.noc0[channel].y);
    return FT_TT_OK;
}

int ft_tt_coord_arc(const ft_tt_arch *arch, ft_tt_coord_system system, ft_tt_noc_coord *out_coord)
{
    if (!arch || !out_coord || !is_known_system(system))
        return FT_TT_EINVAL;
    store_coord(out_coord, arch->fixed.arc.x, arch->fixed.arc.y);
    return FT_TT_OK;
}

int ft_tt_coord_pcie(const ft_tt_arch *arch, uint32_t index, ft_tt_coord_system system,
                     ft_tt_noc_coord *out_coord)
{
    if (!arch || !out_coord || !is_known_system(system))
        return FT_TT_EINVAL;
    if (index >= arch->fixed.pcie_count || !arch->fixed.pcie_noc0)
        return FT_TT_EINVAL;
    if (system == FT_TT_COORD_TRANSLATED)
        return FT_TT_EUNAVAIL;
    store_coord(out_coord, arch->fixed.pcie_noc0[index].x, arch->fixed.pcie_noc0[index].y);
    return FT_TT_OK;
}

int ft_tt_coord_l2cpu(const ft_tt_grid_info *grid, uint32_t logical_index,
                      ft_tt_coord_system system, ft_tt_noc_coord *out_coord)
{
    const ft_tt_arch *arch = NULL;
    uint32_t tile;
    int rc = begin_grid_lookup(grid, system, out_coord, &arch);

    if (rc != FT_TT_OK)
        return rc;
    for (tile = 0; tile < arch->fixed.l2cpu_count; tile++) {
        if (!l2cpu_tile_enabled(grid, arch, tile))
            continue;
        if (logical_index > 0u) {
            logical_index--;
            continue;
        }
        if (translation_applies(grid, system))
            return FT_TT_EUNAVAIL;
        store_coord(out_coord, arch->fixed.l2cpu_noc0[tile].x, arch->fixed.l2cpu_noc0[tile].y);
        return FT_TT_OK;
    }
    return FT_TT_EINVAL;
}

/* ------------------------------------------------------------------------- */
/* Classification                                                            */
/* ------------------------------------------------------------------------- */

/** Index of (x, y) in a coordinate list, or -1. */
static int find_xy(const ft_tt_arch_xy *list, uint32_t count, uint32_t x, uint32_t y)
{
    uint32_t index;

    if (!list)
        return -1;
    for (index = 0; index < count; index++) {
        if (list[index].x == x && list[index].y == y)
            return (int)index;
    }
    return -1;
}

/** Index of `value` in a byte list, or -1. */
static int find_byte(const uint8_t *list, uint32_t count, uint32_t value)
{
    uint32_t index;

    for (index = 0; index < count; index++) {
        if (list[index] == value)
            return (int)index;
    }
    return -1;
}

static bool is_tensix_position(const ft_tt_arch *arch, uint32_t x, uint32_t y)
{
    return find_byte(arch->tensix.noc_x_by_column, arch->tensix.column_count, x) >= 0 &&
           find_byte(arch->tensix.noc_y_by_row, arch->tensix.row_count, y) >= 0;
}

/** What sits at raw NoC #0 position (x, y). */
static ft_tt_core_type classify_noc0(const ft_tt_arch *arch, uint32_t x, uint32_t y)
{
    const ft_tt_arch_fixed_blocks *fixed = &arch->fixed;

    if (x >= arch->noc.size_x || y >= arch->noc.size_y)
        return FT_TT_CORE_NONE;
    if (is_tensix_position(arch, x, y))
        return FT_TT_CORE_TENSIX;
    if (find_xy(arch->dram.noc0, arch->dram.bank_count * arch->dram.ports_per_bank, x, y) >= 0)
        return FT_TT_CORE_DRAM;
    if (find_xy(arch->eth.noc0, arch->eth.channel_count, x, y) >= 0)
        return FT_TT_CORE_ETH;
    if (fixed->arc.x == x && fixed->arc.y == y)
        return FT_TT_CORE_ARC;
    if (fixed->security_count > 0u && fixed->security.x == x && fixed->security.y == y)
        return FT_TT_CORE_SECURITY;
    if (find_xy(fixed->pcie_noc0, fixed->pcie_count, x, y) >= 0)
        return FT_TT_CORE_PCIE;
    if (find_xy(fixed->l2cpu_noc0, fixed->l2cpu_count, x, y) >= 0)
        return FT_TT_CORE_L2CPU;
    if (find_xy(fixed->router_noc0, fixed->router_count, x, y) >= 0)
        return FT_TT_CORE_ROUTER;
    return FT_TT_CORE_NONE;
}

/** Slot of (x, y) in the translated DRAM alias block, or -1. */
static int dram_alias_slot(const ft_tt_arch *arch, uint32_t x, uint32_t y)
{
    const ft_tt_arch_dram *dram = &arch->dram;
    uint32_t banks_per_column;

    if (dram->bank_count == 0u || dram->translated_column_count == 0u ||
        dram->ports_per_bank == 0u)
        return -1;
    banks_per_column = dram->bank_count / dram->translated_column_count;
    if (x < dram->translated_origin_x ||
        x >= dram->translated_origin_x + dram->translated_column_count ||
        y < dram->translated_origin_y ||
        y >= dram->translated_origin_y + banks_per_column * dram->ports_per_bank)
        return -1;
    return (int)((x - dram->translated_origin_x) * banks_per_column +
                 (y - dram->translated_origin_y) / dram->ports_per_bank);
}

/** Slot of (x, y) in the translated Ethernet alias row, or -1. */
static int eth_alias_slot(const ft_tt_arch *arch, uint32_t x, uint32_t y)
{
    if (y != arch->eth.translated_origin_y || x < arch->eth.translated_origin_x ||
        x >= arch->eth.translated_origin_x + arch->eth.channel_count)
        return -1;
    return (int)(x - arch->eth.translated_origin_x);
}

/**
 * What sits at translated (x, y) with translation enabled. Inside the grid only
 * Tensix columns are renumbered, and they renumber onto the same set of column
 * positions, so the NoC #0 answer holds; outside it are the alias ranges.
 */
static ft_tt_core_type classify_translated(const ft_tt_arch *arch, uint32_t x, uint32_t y)
{
    if (x < arch->noc.size_x && y < arch->noc.size_y)
        return classify_noc0(arch, x, y);
    if (dram_alias_slot(arch, x, y) >= 0)
        return FT_TT_CORE_DRAM;
    if (arch->fixed.pcie_count > 0u && arch->fixed.pcie_translated.x == x &&
        arch->fixed.pcie_translated.y == y)
        return FT_TT_CORE_PCIE;
    if (eth_alias_slot(arch, x, y) >= 0)
        return FT_TT_CORE_ETH;
    return FT_TT_CORE_NONE;
}

ft_tt_core_type ft_tt_coord_classify(const ft_tt_grid_info *grid, ft_tt_coord_system system,
                                     uint32_t x, uint32_t y)
{
    const ft_tt_arch *arch = NULL;

    if (!is_known_system(system) || resolve_grid_arch(grid, &arch) != FT_TT_OK)
        return FT_TT_CORE_NONE;
    if (translation_applies(grid, system))
        return classify_translated(arch, x, y);
    return classify_noc0(arch, x, y);
}

const char *ft_tt_core_type_string(ft_tt_core_type type)
{
    switch (type) {
    case FT_TT_CORE_NONE:
        return "none";
    case FT_TT_CORE_TENSIX:
        return "tensix";
    case FT_TT_CORE_DRAM:
        return "dram";
    case FT_TT_CORE_ETH:
        return "eth";
    case FT_TT_CORE_ARC:
        return "arc";
    case FT_TT_CORE_PCIE:
        return "pcie";
    case FT_TT_CORE_L2CPU:
        return "l2cpu";
    case FT_TT_CORE_SECURITY:
        return "security";
    case FT_TT_CORE_ROUTER:
        return "router";
    default:
        return "unknown";
    }
}

/* ------------------------------------------------------------------------- */
/* Enabled checks                                                            */
/* ------------------------------------------------------------------------- */

/** Enabled check for a raw NoC #0 node already classified as `type`. */
static bool noc0_node_enabled(const ft_tt_grid_info *grid, const ft_tt_arch *arch,
                              ft_tt_core_type type, uint32_t x, uint32_t y)
{
    int index;

    switch (type) {
    case FT_TT_CORE_TENSIX:
        index = find_byte(arch->tensix.noc_x_by_column, arch->tensix.column_count, x);
        return tensix_column_enabled(arch, grid->tensix_enabled_col, (uint32_t)index);
    case FT_TT_CORE_DRAM:
        index = find_xy(arch->dram.noc0, arch->dram.bank_count * arch->dram.ports_per_bank, x, y);
        return bit_is_set(grid->enabled_gddr, (uint32_t)index / arch->dram.ports_per_bank);
    case FT_TT_CORE_ETH:
        index = find_xy(arch->eth.noc0, arch->eth.channel_count, x, y);
        return bit_is_set(grid->enabled_eth, (uint32_t)index);
    case FT_TT_CORE_L2CPU:
        index = find_xy(arch->fixed.l2cpu_noc0, arch->fixed.l2cpu_count, x, y);
        return l2cpu_tile_enabled(grid, arch, (uint32_t)index);
    case FT_TT_CORE_NONE:
        return false;
    default:
        return true; /* ARC, security, PCIe and routers are never harvested */
    }
}

/** Enabled check for translated (x, y) with translation on. Harvested units sit
 *  at the top of each renumbered range, so a slot is live iff it is below the
 *  enabled count. */
static bool translated_node_enabled(const ft_tt_grid_info *grid, const ft_tt_arch *arch,
                                    uint32_t x, uint32_t y)
{
    ft_tt_core_type type = classify_translated(arch, x, y);
    int slot;

    if (x < arch->noc.size_x && y < arch->noc.size_y) {
        if (type != FT_TT_CORE_TENSIX)
            return noc0_node_enabled(grid, arch, type, x, y);
        slot = find_byte(arch->tensix.noc_x_by_column, arch->tensix.column_count, x);
        return (uint32_t)slot < enabled_tensix_columns(grid, arch);
    }
    switch (type) {
    case FT_TT_CORE_DRAM:
        return (uint32_t)dram_alias_slot(arch, x, y) < enabled_dram_banks(grid, arch);
    case FT_TT_CORE_ETH:
        return (uint32_t)eth_alias_slot(arch, x, y) < enabled_eth_channels(grid, arch);
    case FT_TT_CORE_PCIE:
        return true;
    default:
        return false;
    }
}

bool ft_tt_coord_is_enabled(const ft_tt_grid_info *grid, ft_tt_coord_system system,
                            uint32_t x, uint32_t y)
{
    const ft_tt_arch *arch = NULL;

    if (!is_known_system(system) || resolve_grid_arch(grid, &arch) != FT_TT_OK)
        return false;
    if (translation_applies(grid, system))
        return translated_node_enabled(grid, arch, x, y);
    return noc0_node_enabled(grid, arch, classify_noc0(arch, x, y), x, y);
}
