/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_dram_nearest.c — off-hardware checks of ft_tt_dram_nearest (tt-place lane): the
 * hop formula on both NoCs, the per-bank cap, nearest <= every production rotation <=
 * farthest, determinism; prints the map for the production decode members (71, 96..110).
 *
 *   make -f Makefile.dramnearest check
 *
 * Status: NEW 2026-10-22. Issues: none. Todo: none.
 */
#include "ft_tt_dram_nearest.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>

static ft_tt_grid_info card_grid(void)
{
    ft_tt_grid_info grid;

    memset(&grid, 0, sizeof(grid));
    grid.valid = true;
    grid.arch = ft_tt_arch_blackhole();
    grid.tensix_enabled_col = 0x3FFFu;
    grid.enabled_gddr = 0xFFu;
    grid.enabled_eth = 0x3FFFu;
    grid.enabled_l2cpu = 0xFu;
    grid.noc_translation_enabled = 1u;
    return grid;
}

static uint32_t total_hops(const ft_tt_grid_info *grid, const ft_tt_noc_coord *cores,
                           const uint32_t *banks, const uint32_t *ports, uint32_t noc)
{
    uint32_t r, total = 0u;

    for (r = 0u; r < 16u; r++) {
        ft_tt_noc_coord door;

        (void)ft_tt_coord_dram_port(grid, banks[r], ports[r], FT_TT_COORD_NOC0, &door);
        total += ft_tt_dram_response_hops(door, cores[r], noc, 17u, 12u);
    }
    return total;
}

static void check_noc(const ft_tt_grid_info *grid, const ft_tt_noc_coord *cores, uint32_t noc)
{
    uint32_t near_b[16], near_p[16], far_b[16], far_p[16], again_b[16], again_p[16];
    uint32_t load[8] = {0}, r, cursor, worst_rotation = 0u, best_rotation = 0xFFFFu;
    uint32_t near_total, far_total;
    int capped = 1;

    ft_tt_test_step("NoC %u: nearest / farthest over the 16 decode members", noc);
    ft_tt_test_check("nearest assigns", ft_tt_dram_nearest_assign(grid, cores, 16u, noc, 2u,
                     FT_TT_DRAM_NEAREST_GOAL_NEAREST, near_b, near_p) == FT_TT_OK, "");
    ft_tt_test_check("farthest assigns", ft_tt_dram_nearest_assign(grid, cores, 16u, noc, 2u,
                     FT_TT_DRAM_NEAREST_GOAL_FARTHEST, far_b, far_p) == FT_TT_OK, "");
    (void)ft_tt_dram_nearest_assign(grid, cores, 16u, noc, 2u, FT_TT_DRAM_NEAREST_GOAL_NEAREST,
                                    again_b, again_p);
    ft_tt_test_check("deterministic", memcmp(near_b, again_b, sizeof(near_b)) == 0 &&
                     memcmp(near_p, again_p, sizeof(near_p)) == 0, "");
    for (r = 0u; r < 16u; r++)
        if (++load[near_b[r]] > 2u)
            capped = 0;
    ft_tt_test_check("at most 2 readers a bank", capped, "");
    near_total = total_hops(grid, cores, near_b, near_p, noc);
    far_total = total_hops(grid, cores, far_b, far_p, noc);
    for (cursor = 0u; cursor < 8u; cursor++) {
        uint32_t b[16], p[16], t;

        for (r = 0u; r < 16u; r++) {
            b[r] = (cursor + r) % 8u;
            p[r] = r / 8u;
        }
        t = total_hops(grid, cores, b, p, noc);
        if (t > worst_rotation)
            worst_rotation = t;
        if (t < best_rotation)
            best_rotation = t;
    }
    printf("  NoC %u total response hops: nearest %u, rotations %u..%u, farthest %u\n", noc,
           near_total, best_rotation, worst_rotation, far_total);
    ft_tt_test_check("nearest beats every rotation", near_total < best_rotation, "");
    ft_tt_test_check("farthest is worse than every rotation", far_total > worst_rotation, "");
    printf("  map (member: bank.port hops):");
    for (r = 0u; r < 16u; r++) {
        ft_tt_noc_coord door;

        (void)ft_tt_coord_dram_port(grid, near_b[r], near_p[r], FT_TT_COORD_NOC0, &door);
        printf(" m%u:%u.%u/%u", r, near_b[r], near_p[r],
               ft_tt_dram_response_hops(door, cores[r], noc, 17u, 12u));
    }
    printf("\n");
}

int main(void)
{
    const ft_tt_grid_info grid = card_grid();
    ft_tt_noc_coord cores[16], a = {0u, 5u}, b = {3u, 9u};
    uint32_t r;

    ft_tt_test_step("the hop formula");
    ft_tt_test_check("NoC 0: (0,5) -> (3,9) is 3 + 4", ft_tt_dram_response_hops(a, b, 0u, 17u,
                     12u) == 7u, "");
    ft_tt_test_check("NoC 1: (0,5) -> (3,9) wraps: 14 + 8",
                     ft_tt_dram_response_hops(a, b, 1u, 17u, 12u) == 22u, "");
    ft_tt_test_check("refuses 17 readers at 2 a bank (8 x 2 = 16)",
                     ft_tt_dram_nearest_assign(&grid, cores, 17u, 0u, 2u,
                                               FT_TT_DRAM_NEAREST_GOAL_NEAREST, NULL, NULL) ==
                         FT_TT_EINVAL, "");
    (void)ft_tt_coord_tensix_by_index(&grid, 71u, FT_TT_COORD_NOC0, &cores[0]);
    for (r = 1u; r < 16u; r++)
        (void)ft_tt_coord_tensix_by_index(&grid, 95u + r, FT_TT_COORD_NOC0, &cores[r]);
    check_noc(&grid, cores, 0u);
    check_noc(&grid, cores, 1u);
    return ft_tt_test_finish();
}
