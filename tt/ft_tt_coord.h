/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_coord.h
 * @brief NoC coordinates for every tile type, derived from a live grid
 *   (ARC telemetry masks + the selected architecture table).
 *
 * Two coordinate systems are exposed:
 *   - NOC0: raw NoC #0 positions, (0,0) top-left.
 *   - TRANSLATED: the renumbering firmware programs into the NIUs, in which
 *     enabled tiles of a type occupy a contiguous range and harvested ones sit at
 *     the top of it. When the grid reports translation disabled, TRANSLATED
 *     collapses onto NOC0.
 *
 * Every "logical" index (Tensix column, DRAM bank, ETH channel, L2CPU tile)
 * counts ENABLED units only, so logical numbering is always contiguous and a
 * harvested unit is never handed out. Where a translated layout depends on
 * information the grid does not carry, the call refuses (FT_TT_EUNAVAIL) rather
 * than guess.
 *
 * Return codes are the FT_TT_* status values of ft_tt_device.h. A grid that is
 * NULL is FT_TT_EINVAL; one that ARC telemetry has not filled, or that names no
 * architecture, is FT_TT_EUNAVAIL. Nothing here touches a device.
 *
 * @status Complete; covered by tests/test_coord.c, tests/test_harvesting.c and
 *   tests/test_arch.c.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_COORD_H
#define FT_TT_COORD_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tt_arch.h"
#include "ft_tt_device.h"

/** Which coordinate system a coordinate is expressed in. */
typedef enum ft_tt_coord_system {

    FT_TT_COORD_NOC0 = 0,

    FT_TT_COORD_TRANSLATED = 1
} ft_tt_coord_system;

/** What sits at a NoC node. ROUTER is a live node with no endpoint tile. */
typedef enum ft_tt_core_type {
    FT_TT_CORE_NONE = 0,
    FT_TT_CORE_TENSIX,
    FT_TT_CORE_DRAM,
    FT_TT_CORE_ETH,
    FT_TT_CORE_ARC,
    FT_TT_CORE_PCIE,
    FT_TT_CORE_L2CPU,
    FT_TT_CORE_SECURITY,
    FT_TT_CORE_ROUTER
} ft_tt_core_type;

/** A NoC coordinate in whichever system the producing call was asked for. */
typedef struct ft_tt_noc_coord {
    uint32_t x;
    uint32_t y;
} ft_tt_noc_coord;

/**
 * Is this grid a topology ft_tt_coord can map? Refuses zero enabled Tensix
 * columns and more harvested DRAM banks than the architecture allows.
 * @return FT_TT_OK, FT_TT_EINVAL, FT_TT_EUNAVAIL, or FT_TT_EIO.
 */
int ft_tt_coord_check_grid(const ft_tt_grid_info *grid);

/**
 * Does the live grid match what the RISC-V kernels were compiled for
 * (FT_TT_KERNEL_BUILD_L1_BANK_COUNT Tensix cores, FT_TT_KERNEL_BUILD_DRAM_BANK_COUNT
 * banks)? A valid but different topology is FT_TT_EIO.
 */
int ft_tt_coord_check_kernel_build_assumptions(const ft_tt_grid_info *grid);

/** Enabled Tensix columns; 0 for an unusable grid. */
uint32_t ft_tt_coord_tensix_column_count(const ft_tt_grid_info *grid);

/** Tensix rows (never harvested on a column-harvesting part); 0 for an unusable grid. */
uint32_t ft_tt_coord_tensix_row_count(const ft_tt_grid_info *grid);

/** Enabled Tensix cores = columns x rows; 0 for an unusable grid. */
uint32_t ft_tt_coord_tensix_core_count(const ft_tt_grid_info *grid);

/** Enabled DRAM banks; 0 for an unusable grid. */
uint32_t ft_tt_coord_dram_bank_count(const ft_tt_grid_info *grid);

/** NoC ports per DRAM bank; 0 for an unusable grid. */
uint32_t ft_tt_coord_dram_ports_per_bank(const ft_tt_grid_info *grid);

/** Enabled Ethernet channels; 0 for an unusable grid. */
uint32_t ft_tt_coord_eth_channel_count(const ft_tt_grid_info *grid);

/** Enabled L2CPU tiles; 0 for an unusable grid. */
uint32_t ft_tt_coord_l2cpu_count(const ft_tt_grid_info *grid);

/** NoC coordinate-space size of the grid's architecture. */
int ft_tt_coord_noc_size(const ft_tt_grid_info *grid, uint32_t *out_size_x, uint32_t *out_size_y);

/**
 * Convert ARC's enabled-columns telemetry word into a HARVESTED-columns mask in
 * grid-column order (bit i set = grid column i fused off).
 */
int ft_tt_coord_tensix_harvest_mask(const ft_tt_arch *arch, uint32_t tensix_enabled_col,
                                    uint32_t *out_harvest_mask);

/** Tensix at logical (x, y), where logical x counts enabled columns only. */
int ft_tt_coord_tensix_at(const ft_tt_grid_info *grid, uint32_t logical_x, uint32_t logical_y,
                          ft_tt_coord_system system, ft_tt_noc_coord *out_coord);

/** Tensix at a row-major logical index (stride = the LIVE column count). */
int ft_tt_coord_tensix_by_index(const ft_tt_grid_info *grid, uint32_t index,
                                ft_tt_coord_system system, ft_tt_noc_coord *out_coord);

/**
 * One NoC port of a logical (enabled-only) DRAM bank. TRANSLATED on a part with a
 * harvested bank is FT_TT_EUNAVAIL: that alias layout is not modelled.
 */
int ft_tt_coord_dram_port(const ft_tt_grid_info *grid, uint32_t logical_bank, uint32_t port,
                          ft_tt_coord_system system, ft_tt_noc_coord *out_coord);

/** Port 0 of a logical DRAM bank. */
int ft_tt_coord_dram_bank(const ft_tt_grid_info *grid, uint32_t logical_bank,
                          ft_tt_coord_system system, ft_tt_noc_coord *out_coord);

/** A logical (enabled-only) Ethernet channel. */
int ft_tt_coord_eth_channel(const ft_tt_grid_info *grid, uint32_t logical_channel,
                            ft_tt_coord_system system, ft_tt_noc_coord *out_coord);

/** The ARC tile. ARC is not renumbered, so both systems give the same answer. */
int ft_tt_coord_arc(const ft_tt_arch *arch, ft_tt_coord_system system, ft_tt_noc_coord *out_coord);

/**
 * PCIe tile `index` in NOC0. TRANSLATED is FT_TT_EUNAVAIL: which tile owns the
 * translated alias depends on the board, and the grid does not say.
 */
int ft_tt_coord_pcie(const ft_tt_arch *arch, uint32_t index, ft_tt_coord_system system,
                     ft_tt_noc_coord *out_coord);

/**
 * A logical (enabled-only) L2CPU tile. TRANSLATED with translation enabled is
 * FT_TT_EUNAVAIL: the architecture table carries no L2CPU alias positions.
 */
int ft_tt_coord_l2cpu(const ft_tt_grid_info *grid, uint32_t logical_index,
                      ft_tt_coord_system system, ft_tt_noc_coord *out_coord);

/** What sits at (x, y) in `system`. FT_TT_CORE_NONE for an unusable grid or a
 *  coordinate that names no node. Harvested tiles keep their type. */
ft_tt_core_type ft_tt_coord_classify(const ft_tt_grid_info *grid, ft_tt_coord_system system,
                                     uint32_t x, uint32_t y);

/** Printable core type name. Never NULL. */
const char *ft_tt_core_type_string(ft_tt_core_type type);

/** True when (x, y) is a node that exists and is not harvested. Router nodes are
 *  live (true) even though nothing can be targeted there. */
bool ft_tt_coord_is_enabled(const ft_tt_grid_info *grid, ft_tt_coord_system system,
                            uint32_t x, uint32_t y);

#endif
