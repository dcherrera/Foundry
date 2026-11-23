/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * ft_tt_dram_nearest.c — balanced nearest-bank assignment of reader cores (header for the
 * reasoning).
 *
 * @status NEW 2026-10-22 (tt-place lane).
 * @issues None
 * @todo None
 */
#include "ft_tt_dram_nearest.h"

#include <string.h>

#define NEAREST_MAX_BANKS 8u
#define NEAREST_MAX_PORTS 3u

uint32_t ft_tt_dram_response_hops(ft_tt_noc_coord door_noc0, ft_tt_noc_coord core_noc0,
                                  uint32_t noc, uint32_t size_x, uint32_t size_y)
{
    uint32_t hx, hy;

    if (size_x == 0u || size_y == 0u)
        return 0u;
    if (noc == 0u) {
        hx = (core_noc0.x + size_x - door_noc0.x % size_x) % size_x;
        hy = (core_noc0.y + size_y - door_noc0.y % size_y) % size_y;
    } else {
        hx = (door_noc0.x + size_x - core_noc0.x % size_x) % size_x;
        hy = (door_noc0.y + size_y - core_noc0.y % size_y) % size_y;
    }
    return hx + hy;
}

/** The cost of a reader on a bank: its nearest door. Negated for the farthest goal so one
 *  minimiser serves both. */
static int32_t reader_bank_cost(const ft_tt_noc_coord doors[NEAREST_MAX_BANKS][NEAREST_MAX_PORTS],
                                uint32_t ports, ft_tt_noc_coord core, uint32_t bank,
                                uint32_t noc, uint32_t sx, uint32_t sy,
                                ft_tt_dram_nearest_goal goal)
{
    uint32_t port, best = 0xFFFFFFFFu;

    for (port = 0u; port < ports; port++) {
        uint32_t hops = ft_tt_dram_response_hops(doors[bank][port], core, noc, sx, sy);

        if (hops < best)
            best = hops;
    }
    return goal == FT_TT_DRAM_NEAREST_GOAL_FARTHEST ? -(int32_t)best : (int32_t)best;
}

/** Greedy: repeatedly give the globally cheapest unassigned (reader, bank) pair. */
static void assign_greedy(const int32_t cost[][NEAREST_MAX_BANKS], uint32_t count,
                          uint32_t banks, uint32_t per_bank, uint32_t *out_banks)
{
    uint32_t load[NEAREST_MAX_BANKS] = {0};
    uint8_t done[FT_TT_DRAM_NEAREST_MAX_READERS] = {0};
    uint32_t step, reader, bank;

    for (step = 0u; step < count; step++) {
        int32_t best = 0x7FFFFFFF;
        uint32_t best_reader = 0u, best_bank = 0u;

        for (reader = 0u; reader < count; reader++) {
            if (done[reader])
                continue;
            for (bank = 0u; bank < banks; bank++)
                if (load[bank] < per_bank && cost[reader][bank] < best) {
                    best = cost[reader][bank];
                    best_reader = reader;
                    best_bank = bank;
                }
        }
        done[best_reader] = 1u;
        load[best_bank]++;
        out_banks[best_reader] = best_bank;
    }
}

/** Pairwise swaps until no swap lowers the total. Capacity is preserved by construction. */
static void improve_by_swaps(const int32_t cost[][NEAREST_MAX_BANKS], uint32_t count,
                             uint32_t *banks)
{
    int improved = 1;
    uint32_t a, b;

    while (improved) {
        improved = 0;
        for (a = 0u; a < count; a++)
            for (b = a + 1u; b < count; b++) {
                int32_t now = cost[a][banks[a]] + cost[b][banks[b]];
                int32_t swapped = cost[a][banks[b]] + cost[b][banks[a]];

                if (swapped < now) {
                    uint32_t t = banks[a];

                    banks[a] = banks[b];
                    banks[b] = t;
                    improved = 1;
                }
            }
    }
}

/** Doors inside each bank: the bank's readers in reader order, each the nearest door no
 *  earlier reader of that bank took (wrapping once all doors are taken). */
static void choose_doors(const ft_tt_noc_coord doors[NEAREST_MAX_BANKS][NEAREST_MAX_PORTS],
                         uint32_t ports, const ft_tt_noc_coord *cores, uint32_t count,
                         const uint32_t *banks, uint32_t noc, uint32_t sx, uint32_t sy,
                         ft_tt_dram_nearest_goal goal, uint32_t *out_ports)
{
    uint32_t taken[NEAREST_MAX_BANKS] = {0}; /* bitmask of ports used */
    uint32_t reader, port;

    for (reader = 0u; reader < count; reader++) {
        const uint32_t bank = banks[reader];
        int32_t best = 0x7FFFFFFF;
        uint32_t best_port = 0u;

        if (taken[bank] == (1u << ports) - 1u)
            taken[bank] = 0u;
        for (port = 0u; port < ports; port++) {
            int32_t hops = (int32_t)ft_tt_dram_response_hops(doors[bank][port], cores[reader],
                                                             noc, sx, sy);

            if (goal == FT_TT_DRAM_NEAREST_GOAL_FARTHEST)
                hops = -hops;
            if (!(taken[bank] & (1u << port)) && hops < best) {
                best = hops;
                best_port = port;
            }
        }
        taken[bank] |= 1u << best_port;
        out_ports[reader] = best_port;
    }
}

int ft_tt_dram_nearest_assign(const ft_tt_grid_info *grid, const ft_tt_noc_coord *cores_noc0,
                              uint32_t count, uint32_t noc, uint32_t per_bank,
                              ft_tt_dram_nearest_goal goal, uint32_t *out_banks,
                              uint32_t *out_ports)
{
    ft_tt_noc_coord doors[NEAREST_MAX_BANKS][NEAREST_MAX_PORTS];
    int32_t cost[FT_TT_DRAM_NEAREST_MAX_READERS][NEAREST_MAX_BANKS];
    uint32_t banks, ports, sx = 0u, sy = 0u, bank, port, reader;
    int rc;

    if (!grid || !cores_noc0 || !out_banks || !out_ports || count == 0u ||
        count > FT_TT_DRAM_NEAREST_MAX_READERS || per_bank == 0u || noc > 1u)
        return FT_TT_EINVAL;
    banks = ft_tt_coord_dram_bank_count(grid);
    ports = ft_tt_coord_dram_ports_per_bank(grid);
    if (banks == 0u || banks > NEAREST_MAX_BANKS || ports == 0u || ports > NEAREST_MAX_PORTS ||
        count > per_bank * banks)
        return FT_TT_EINVAL;
    rc = ft_tt_coord_noc_size(grid, &sx, &sy);
    if (rc != FT_TT_OK)
        return rc;
    for (bank = 0u; bank < banks; bank++)
        for (port = 0u; port < ports; port++) {
            rc = ft_tt_coord_dram_port(grid, bank, port, FT_TT_COORD_NOC0, &doors[bank][port]);
            if (rc != FT_TT_OK)
                return rc;
        }
    for (reader = 0u; reader < count; reader++)
        for (bank = 0u; bank < banks; bank++)
            cost[reader][bank] = reader_bank_cost((const ft_tt_noc_coord(*)[NEAREST_MAX_PORTS])doors,
                                                  ports, cores_noc0[reader], bank, noc, sx, sy,
                                                  goal);
    assign_greedy((const int32_t(*)[NEAREST_MAX_BANKS])cost, count, banks, per_bank, out_banks);
    improve_by_swaps((const int32_t(*)[NEAREST_MAX_BANKS])cost, count, out_banks);
    choose_doors((const ft_tt_noc_coord(*)[NEAREST_MAX_PORTS])doors, ports, cores_noc0, count,
                 out_banks, noc, sx, sy, goal, out_ports);
    return FT_TT_OK;
}
