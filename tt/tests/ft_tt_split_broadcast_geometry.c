/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_split_broadcast_geometry.c
 * @brief Implementation of the split-broadcast plan and its refusals. Pure arithmetic over
 *   coordinates; it opens nothing.
 *
 * @status Complete. Gated off-hardware by tests/test_split_broadcast_geometry.c, one
 *   mutation per refusal.
 * @issues None
 * @todo None
 */

#include "ft_tt_split_broadcast_geometry.h"

#include <stdio.h>
#include <string.h>

bool ft_tt_split_broadcast_rectangles_overlap(const ft_tt_multicast_rectangle *left,
                                              const ft_tt_multicast_rectangle *right)
{
    if (left == NULL || right == NULL)
        return false;
    return !(left->x_high < right->x_low || right->x_high < left->x_low ||
             left->y_high < right->y_low || right->y_high < left->y_low);
}

/** Every group's rectangle, or the first row that is not one. */
static int build_every_rectangle(const ft_tt_noc_coord *receivers, uint32_t rows,
                                 uint32_t columns, const ft_tt_noc_coord *senders,
                                 uint32_t max_destinations, ft_tt_split_broadcast_plan *out,
                                 char *why, size_t why_bytes)
{
    uint32_t group;
    int rc;

    for (group = 0u; group < rows; group++) {
        rc = ft_tt_multicast_rectangle_from_cores(&receivers[group * columns], columns,
                                                  senders[group], &out->rectangle[group]);
        if (rc != FT_TT_OK) {
            (void)snprintf(why, why_bytes, "group %u is not a rectangle: rc=%d (%s)", group,
                           rc, ft_tt_error_string(rc));
            return rc;
        }
        if (out->rectangle[group].destination_count > max_destinations) {
            (void)snprintf(why, why_bytes,
                           "group %u would broadcast to %u destinations, over the limit of "
                           "%u", group, out->rectangle[group].destination_count,
                           max_destinations);
            return FT_TT_ERROR;
        }
    }
    return FT_TT_OK;
}

/**
 * The two cross-group refusals.
 *
 * Both are about a broadcast reaching a node its group does not own. A sender inside ANY
 * rectangle — its own or the other one's — is written by that broadcast, and two
 * rectangles that overlap have a receiver written twice per round with no ordering between
 * the two writes.
 */
static int refuse_every_collision(const ft_tt_noc_coord *senders, uint32_t rows,
                                  const ft_tt_split_broadcast_plan *plan, char *why,
                                  size_t why_bytes)
{
    uint32_t group;
    uint32_t other;

    for (group = 0u; group < rows; group++)
        for (other = 0u; other < rows; other++) {
            if (ft_tt_multicast_rectangle_contains(&plan->rectangle[group], senders[other])) {
                (void)snprintf(why, why_bytes,
                               "sender %u at (%u,%u) is INSIDE group %u's rectangle, so the "
                               "broadcast would overwrite its own staging buffer", other,
                               senders[other].x, senders[other].y, group);
                return FT_TT_ERROR;
            }
            if (other > group &&
                ft_tt_split_broadcast_rectangles_overlap(&plan->rectangle[group],
                                                         &plan->rectangle[other])) {
                (void)snprintf(why, why_bytes,
                               "groups %u and %u overlap: (%u,%u)..(%u,%u) against "
                               "(%u,%u)..(%u,%u)", group, other, plan->rectangle[group].x_low,
                               plan->rectangle[group].y_low, plan->rectangle[group].x_high,
                               plan->rectangle[group].y_high, plan->rectangle[other].x_low,
                               plan->rectangle[other].y_low, plan->rectangle[other].x_high,
                               plan->rectangle[other].y_high);
                return FT_TT_ERROR;
            }
        }
    return FT_TT_OK;
}

int ft_tt_split_broadcast_plan_build(const ft_tt_noc_coord *receivers, uint32_t rows,
                                     uint32_t columns, const ft_tt_noc_coord *senders,
                                     uint32_t sender_count, uint32_t max_destinations,
                                     ft_tt_split_broadcast_plan *out, char *why,
                                     size_t why_bytes)
{
    int rc;

    if (why == NULL || why_bytes == 0u)
        return FT_TT_EINVAL;
    if (receivers == NULL || senders == NULL || out == NULL) {
        (void)snprintf(why, why_bytes, "a required argument is NULL");
        return FT_TT_EINVAL;
    }
    if (rows == 0u || rows > FT_TT_SPLIT_MAX_GROUPS || columns == 0u ||
        max_destinations == 0u) {
        (void)snprintf(why, why_bytes,
                       "%u groups of %u with a limit of %u is outside 1..%u groups", rows,
                       columns, max_destinations, FT_TT_SPLIT_MAX_GROUPS);
        return FT_TT_EINVAL;
    }
    if (sender_count != rows) {
        (void)snprintf(why, why_bytes,
                       "%u senders for %u groups; each group needs its own, because a "
                       "shared sender would serialise the broadcasts", sender_count, rows);
        return FT_TT_EINVAL;
    }

    memset(out, 0, sizeof(*out));
    out->group_count = rows;
    rc = build_every_rectangle(receivers, rows, columns, senders, max_destinations, out, why,
                               why_bytes);
    if (rc == FT_TT_OK)
        rc = refuse_every_collision(senders, rows, out, why, why_bytes);
    if (rc != FT_TT_OK) {
        memset(out, 0, sizeof(*out));
        return rc;
    }
    (void)snprintf(why, why_bytes,
                   "%u disjoint rectangles of %u destinations, every sender outside all of "
                   "them", rows, out->rectangle[0].destination_count);
    return FT_TT_OK;
}
