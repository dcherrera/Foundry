/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_dram_stripe.c
 * @brief Implementation of ft_tt_dram_stripe.h — see that header for the three measured
 *   ceilings and why choosing a port moves no data.
 *
 * @status Off-hardware in tests/test_cooperative_matmul_plan.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_dram_stripe.h"

#include <stdbool.h>
#include <string.h>

#include "ft_tt_device.h"

int ft_tt_dram_port_set_build(const ft_tt_grid_info *grid, uint32_t logical_bank,
                              ft_tt_coord_system system, ft_tt_dram_port_set *out)
{
    ft_tt_dram_port_set set;
    uint32_t port_count;
    uint32_t port;

    if (grid == NULL || out == NULL) {
        ft_tt_log_error("dram stripe: NULL grid/out");
        return FT_TT_EINVAL;
    }

    port_count = ft_tt_coord_dram_ports_per_bank(grid);
    if (port_count == 0u) {
        ft_tt_log_error("dram stripe: the grid reports 0 DRAM ports per bank");
        return FT_TT_EUNAVAIL;
    }
    if (port_count > FT_TT_DRAM_STRIPE_MAX_PORTS) {
        /* Named refusal rather than a truncated set: a set that silently holds fewer doors
         * than the part has would rotate over a subset and report a ceiling that is not the
         * bank's. */
        ft_tt_log_error("dram stripe: this part has %u DRAM ports per bank but "
                        "FT_TT_DRAM_STRIPE_MAX_PORTS is %u — raise the capacity",
                        port_count, FT_TT_DRAM_STRIPE_MAX_PORTS);
        return FT_TT_ENOMEM;
    }

    memset(&set, 0, sizeof(set));
    set.logical_bank = logical_bank;
    set.port_count = port_count;

    for (port = 0u; port < port_count; port++) {
        ft_tt_noc_coord noc0;
        int rc = ft_tt_coord_dram_port(grid, logical_bank, port, system,
                                       &set.port[port].coord);

        if (rc != FT_TT_OK)
            return rc;
        /* The row limit is a property of the physical node, so it is read in NOC 0 even when
         * the transfer will be aimed with a translated coordinate. */
        rc = ft_tt_coord_dram_port(grid, logical_bank, port, FT_TT_COORD_NOC0, &noc0);
        if (rc != FT_TT_OK)
            return rc;

        set.port[port].logical_bank = logical_bank;
        set.port[port].port = port;
        set.port[port].noc_row = noc0.y;
    }

    *out = set;
    return FT_TT_OK;
}

const ft_tt_dram_endpoint *ft_tt_dram_port_set_reader_endpoint(const ft_tt_dram_port_set *set,
                                                               uint32_t reader_index)
{
    if (set == NULL || set->port_count == 0u)
        return NULL;
    return &set->port[reader_index % set->port_count];
}

int ft_tt_dram_stripe_choose_ports(const ft_tt_grid_info *grid, const uint32_t *banks,
                                   uint32_t count, ft_tt_coord_system system,
                                   ft_tt_dram_endpoint *out)
{
    uint32_t reader;

    if (grid == NULL || banks == NULL || out == NULL || count == 0u) {
        ft_tt_log_error("dram stripe: NULL argument or zero reader count");
        return FT_TT_EINVAL;
    }

    for (reader = 0u; reader < count; reader++) {
        ft_tt_dram_port_set set;
        uint32_t best_port = 0u;
        uint32_t best_score = 0xFFFFFFFFu;
        uint32_t port;
        int rc = ft_tt_dram_port_set_build(grid, banks[reader], system, &set);

        if (rc != FT_TT_OK)
            return rc;

        for (port = 0u; port < set.port_count; port++) {
            /* A DOOR somebody already uses is the expensive collision and a ROW is the cheap
             * one, which is the opposite of the intuitive order and comes straight off the
             * ladder: two streams through one endpoint share ~45 GB/s, two streams through two
             * endpoints of one row share ~60. So take a fresh door on a used row before a used
             * door on a fresh one. */
            uint32_t score = 0u;
            uint32_t earlier;

            for (earlier = 0u; earlier < reader; earlier++) {
                if (out[earlier].logical_bank == banks[reader] && out[earlier].port == port) {
                    score += 1000u;
                    continue;
                }
                if (out[earlier].noc_row == set.port[port].noc_row)
                    score += 1u;
            }
            if (score < best_score) {
                best_score = score;
                best_port = port;
            }
        }
        out[reader] = set.port[best_port];
    }
    return FT_TT_OK;
}

/** Is `index` the first appearance of its (bank, port) in the array? Readers SHARE doors by
 *  design — a rotation over 3 ports with 8 readers repeats each door two or three times — so
 *  counting raw pairs would report a door used twice as a row collision with itself. */
static bool is_first_occurrence(const ft_tt_dram_endpoint *endpoints, uint32_t index)
{
    uint32_t earlier;

    for (earlier = 0u; earlier < index; earlier++) {
        if (endpoints[earlier].logical_bank == endpoints[index].logical_bank &&
            endpoints[earlier].port == endpoints[index].port)
            return false;
    }
    return true;
}

uint32_t ft_tt_dram_stripe_shared_rows(const ft_tt_dram_endpoint *endpoints, uint32_t count,
                                       uint32_t *out_first_left, uint32_t *out_first_right)
{
    uint32_t collisions = 0u;
    uint32_t left;

    if (endpoints == NULL)
        return 0u;

    for (left = 0u; left < count; left++) {
        uint32_t right;

        if (!is_first_occurrence(endpoints, left))
            continue;
        for (right = left + 1u; right < count; right++) {
            if (endpoints[left].noc_row != endpoints[right].noc_row)
                continue;
            /* Only first occurrences on BOTH sides, which also settles the same-door case:
             * `right` being the first of its door means its door differs from every earlier
             * one, `left` included. */
            if (!is_first_occurrence(endpoints, right))
                continue;
            if (collisions == 0u) {
                if (out_first_left != NULL)
                    *out_first_left = left;
                if (out_first_right != NULL)
                    *out_first_right = right;
            }
            collisions++;
        }
    }
    return collisions;
}
