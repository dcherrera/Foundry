/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_dram_nearest.h
 * @brief Which DRAM bank (and door) each reader core should own so its read RESPONSES travel
 *   the fewest NoC hops — a balanced reader -> bank assignment, nothing else.
 *
 * A NoC read's DATA comes back on the same NoC the request went out on, dimension-ordered:
 * NoC 0 runs +x then +y, NoC 1 runs -x then -y, both on a torus (Blackhole NoC 17 x 12). So
 * the response path from a DRAM door at (dx,dy) to a core at (cx,cy) is, in NoC 0 space,
 *
 *   NoC 0: ((cx - dx) mod W) + ((cy - dy) mod H)
 *   NoC 1: ((dx - cx) mod W) + ((dy - cy) mod H)
 *
 * and the row links it crosses are the ones every other response on that row shares. A
 * west-bank (x = 0) response to a core at x = 1..7 never reaches the east DRAM column, which
 * is the row-sharing ceiling ft_tt_dram_stripe.h measured (two endpoints of one y, ~60 GB/s).
 *
 * The assignment is BALANCED: at most `per_bank` readers a bank, so the per-bank load the
 * rotation gave is kept. Greedy on the cheapest (reader, bank) pair, then pairwise swaps
 * until no swap lowers the total — deterministic, so a load-time placement is reproducible.
 * Doors are chosen per bank among that bank's readers, nearest first, distinct while a bank
 * has <= 3 readers.
 *
 * Moving a reader's bank moves no VALUE: the caller uploads the same bytes to the new bank.
 *
 * @status NEW 2026-10-22 (tt-place lane). Used by tests/test_dram_capacity.c (stage `place`)
 *   and by Pantheon's PN_TT_WEIGHT_BANKS=near.
 * @issues None
 * @todo None
 */
#ifndef FT_TT_DRAM_NEAREST_H
#define FT_TT_DRAM_NEAREST_H

#include <stdint.h>

#include "ft_tt_coord.h"

#define FT_TT_DRAM_NEAREST_MAX_READERS 64u

/** Which assignment to build. FARTHEST is a control arm for measurement, not a placement. */
typedef enum ft_tt_dram_nearest_goal {
    FT_TT_DRAM_NEAREST_GOAL_NEAREST = 0,
    FT_TT_DRAM_NEAREST_GOAL_FARTHEST = 1
} ft_tt_dram_nearest_goal;

/**
 * Response hops from a DRAM door to a core on NoC `noc` (0 or 1). Both coordinates are NoC 0
 * coordinates; `size_x`/`size_y` the NoC size.
 */
uint32_t ft_tt_dram_response_hops(ft_tt_noc_coord door_noc0, ft_tt_noc_coord core_noc0,
                                  uint32_t noc, uint32_t size_x, uint32_t size_y);

/**
 * Assign `count` reader cores to banks, at most `per_bank` readers a bank.
 *
 * @param grid       topology from ARC telemetry
 * @param cores_noc0 `count` reader cores in NoC 0 coordinates
 * @param count      1..FT_TT_DRAM_NEAREST_MAX_READERS, <= per_bank * banks
 * @param noc        the NoC the readers issue on (0 or 1)
 * @param per_bank   capacity per bank, >= 1
 * @param goal       nearest (placement) or farthest (control)
 * @param out_banks  `count` logical banks
 * @param out_ports  `count` ports of those banks (0..2)
 * @return FT_TT_OK; FT_TT_EINVAL on a bad argument; a coord error from the grid
 */
int ft_tt_dram_nearest_assign(const ft_tt_grid_info *grid, const ft_tt_noc_coord *cores_noc0,
                              uint32_t count, uint32_t noc, uint32_t per_bank,
                              ft_tt_dram_nearest_goal goal, uint32_t *out_banks,
                              uint32_t *out_ports);

#endif /* FT_TT_DRAM_NEAREST_H */
