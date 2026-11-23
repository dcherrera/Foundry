/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_group_broadcast_partition.c
 * @brief Implementation of ft_tt_group_broadcast_partition.h — see that header for why
 *   the partition is by row and why the four-destination limit is a refusal rather than
 *   a comment. Nothing here touches the device.
 *
 * @status Off-hardware, driven red by tests/test_group_broadcast_partition.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_group_broadcast_partition.h"

#include "ft_tt_device.h"

/** Emit one piece covering `[x_first, x_last]` of row `y`, and refuse anything the
 *  rectangle arithmetic itself would refuse.
 *
 *  `leader` is passed to ft_tt_multicast_rectangle_from_corners as the SENDER so that
 *  `sender_inside` is computed rather than assumed — a piece that somehow contained the
 *  leader would come back with `destination_count == node_count - 1` and be caught by the
 *  check below instead of silently under-counting the barrier. */
static int append_row_run(ft_tt_group_broadcast_partition *partition, uint32_t y,
                          uint32_t x_first, uint32_t x_last, ft_tt_noc_coord leader)
{
    ft_tt_multicast_rectangle rectangle;
    int rc;

    if (partition->rectangle_count >= FT_TT_GROUP_BROADCAST_MAX_RECTANGLES) {
        ft_tt_log_error("broadcast partition: needs more than %u pieces",
                        FT_TT_GROUP_BROADCAST_MAX_RECTANGLES);
        return FT_TT_ERROR;
    }
    rc = ft_tt_multicast_rectangle_from_corners(x_first, y, x_last, y, leader, &rectangle);
    if (rc != FT_TT_OK)
        return rc;
    if (rectangle.sender_inside) {
        ft_tt_log_error("broadcast partition: piece (%u,%u)..(%u,%u) contains the leader "
                        "(%u,%u); the row split was supposed to remove it",
                        x_first, y, x_last, y, leader.x, leader.y);
        return FT_TT_ERROR;
    }
    partition->rectangle[partition->rectangle_count] = rectangle;
    partition->rectangle_count++;
    partition->destination_total += rectangle.destination_count;
    return FT_TT_OK;
}

/** Split `[x_first, x_last]` into chunks of at most `max_destinations` and emit each. */
static int append_row_chunks(ft_tt_group_broadcast_partition *partition, uint32_t y,
                             uint32_t x_first, uint32_t x_last, uint32_t max_destinations,
                             ft_tt_noc_coord leader)
{
    uint32_t x = x_first;

    while (x <= x_last) {
        const uint32_t remaining = x_last - x + 1u;
        const uint32_t chunk = remaining < max_destinations ? remaining : max_destinations;
        const int rc = append_row_run(partition, y, x, x + chunk - 1u, leader);

        if (rc != FT_TT_OK)
            return rc;
        x += chunk;
    }
    return FT_TT_OK;
}

/** One row of the group's bounding box, with the leader's cell removed. A row that does
 *  not hold the leader is one run; the leader's row is the run left of it, the run right
 *  of it, or both, or neither when the group is one column wide. */
static int append_row(ft_tt_group_broadcast_partition *partition,
                      const ft_tt_multicast_rectangle *group, uint32_t y,
                      uint32_t max_destinations, ft_tt_noc_coord leader)
{
    int rc;

    if (y != leader.y)
        return append_row_chunks(partition, y, group->x_low, group->x_high, max_destinations,
                                 leader);
    if (leader.x > group->x_low) {
        rc = append_row_chunks(partition, y, group->x_low, leader.x - 1u, max_destinations,
                               leader);
        if (rc != FT_TT_OK)
            return rc;
    }
    if (leader.x < group->x_high)
        return append_row_chunks(partition, y, leader.x + 1u, group->x_high, max_destinations,
                                 leader);
    return FT_TT_OK;
}

/** Whether `coord` is one of `members`. The leader has to be a member: a sender outside
 *  the group is a legal multicast shape and a DIFFERENT one, whose barrier and column
 *  split this module does not describe. */
static bool contains_coordinate(const ft_tt_noc_coord *members, uint32_t member_count,
                                ft_tt_noc_coord coord)
{
    uint32_t index;

    for (index = 0u; index < member_count; index++)
        if (members[index].x == coord.x && members[index].y == coord.y)
            return true;
    return false;
}

bool ft_tt_group_broadcast_partition_covers(const ft_tt_group_broadcast_partition *partition,
                                            const ft_tt_noc_coord *members,
                                            uint32_t member_count, ft_tt_noc_coord leader)
{
    uint32_t member;

    if (partition == NULL || members == NULL)
        return false;
    for (member = 0u; member < member_count; member++) {
        const bool is_leader =
            members[member].x == leader.x && members[member].y == leader.y;
        uint32_t hits = 0u;
        uint32_t piece;

        for (piece = 0u; piece < partition->rectangle_count; piece++)
            if (ft_tt_multicast_rectangle_contains(&partition->rectangle[piece], members[member]))
                hits++;
        if (hits != (is_leader ? 0u : 1u))
            return false;
    }
    return true;
}

int ft_tt_group_broadcast_partition_build(const ft_tt_noc_coord *members, uint32_t member_count,
                                          ft_tt_noc_coord leader, uint32_t max_destinations,
                                          ft_tt_group_broadcast_partition *out)
{
    ft_tt_multicast_rectangle group;
    uint32_t y;
    int rc;

    if (members == NULL || out == NULL)
        return FT_TT_EINVAL;
    if (member_count < 2u || member_count > FT_TT_GROUP_BROADCAST_MAX_MEMBERS) {
        ft_tt_log_error("broadcast partition: %u members is outside 2..%u; one member has "
                        "no receivers, which is a zero-destination multicast",
                        member_count, FT_TT_GROUP_BROADCAST_MAX_MEMBERS);
        return FT_TT_EINVAL;
    }
    if (max_destinations == 0u || max_destinations > FT_TT_GROUP_BROADCAST_HAZARD_LIMIT) {
        ft_tt_log_error("broadcast partition: %u destinations per packet is outside 1..%u; "
                        "five wedges a core at its next firmware boot",
                        max_destinations, FT_TT_GROUP_BROADCAST_HAZARD_LIMIT);
        return FT_TT_EINVAL;
    }
    /* The pigeonhole fill test AND the duplicate check, both of which this partition
     * depends on: a row of the bounding box is only a set of members if the members fill
     * the box. The leader is the sender, so `group.destination_count` is already
     * member_count - 1 and is checked against the partition's own total below. */
    rc = ft_tt_multicast_rectangle_from_cores(members, member_count, leader, &group);
    if (rc != FT_TT_OK)
        return rc;
    if (!contains_coordinate(members, member_count, leader)) {
        ft_tt_log_error("broadcast partition: leader (%u,%u) is not one of the %u members",
                        leader.x, leader.y, member_count);
        return FT_TT_ERROR;
    }

    out->rectangle_count = 0u;
    out->destination_total = 0u;
    for (y = group.y_low; y <= group.y_high; y++) {
        rc = append_row(out, &group, y, max_destinations, leader);
        if (rc != FT_TT_OK)
            return rc;
    }
    if (out->destination_total != member_count - 1u) {
        ft_tt_log_error("broadcast partition: %u destinations cover %u receivers",
                        out->destination_total, member_count - 1u);
        return FT_TT_ERROR;
    }
    if (!ft_tt_group_broadcast_partition_covers(out, members, member_count, leader)) {
        ft_tt_log_error("broadcast partition: the pieces do not cover every receiver "
                        "exactly once");
        return FT_TT_ERROR;
    }
    return FT_TT_OK;
}
