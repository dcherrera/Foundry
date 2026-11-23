/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_group_broadcast_partition.h
 * @brief Turn ONE group of cooperating Tensix into `ceil(receivers / max)` multicast
 *   rectangles, so a shared operand reaches every member without any single packet
 *   naming more destinations than the standing hazard rule allows.
 *
 *   Pure arithmetic. Nothing here touches the device, which is the point: a wrong
 *   rectangle is not a failed transaction, it is a write into a core somebody else owns,
 *   so every claim this module makes is provable off-hardware and is
 *   driven red by tests/test_group_broadcast_partition.c.
 *
 * ---------------------------------------------------------------------------
 * WHY IT EXISTS, AND WHY IT IS NOT ft_tt_split_broadcast_geometry.h
 * ---------------------------------------------------------------------------
 * `tests/ft_tt_split_broadcast_geometry.h` answers experiment B's question — several
 * SENDERS each broadcasting to their own group at once — and its central refusal is
 * "no sender may be inside any rectangle". That is exactly wrong for a cooperative
 * matmul: there is ONE sender, it is a working member of the group, and it sits inside
 * the group's bounding box. So the two modules share the arithmetic in ft_tt_multicast.h
 * and nothing else.
 *
 * The rule this one enforces instead is the standing operational hazard, measured on
 * silicon twice: **never more than four destinations in one multicast packet**. A
 * 5-destination broadcast leaves every receiver and the sender able to accept re-launches
 * indefinitely and then wedges a core at the NEXT firmware boot (`0x00047000`, BRISC
 * stalled in `device_setup()`), while N=8 issued as two disjoint <=4-destination
 * broadcasts through the same park-and-reboot shape came back 20/20 PASS. The cause is
 * believed to be our teardown rather than the packet, and the rule still stands as an
 * OPERATIONAL rule until a residency that is never re-booted proves otherwise.
 *
 * ---------------------------------------------------------------------------
 * THE PARTITION IS BY ROW, AND THAT IS NOT AN ARBITRARY CHOICE
 * ---------------------------------------------------------------------------
 * A multicast delivers to every node of a closed box, so a partition piece must be a box
 * containing group members and NOTHING else. Rows are the only shape that is always
 * available: the group is a `rows x columns` box, the leader occupies one cell, and
 * removing one cell from a box leaves a shape that is not a box — but every ROW of that
 * shape still is, either whole or as the single contiguous run either side of the leader.
 * Splitting each run into chunks of at most `max_destinations` then costs
 * `ceil(run / max)` packets and nothing else.
 *
 * The consequences worth stating:
 *  - a group laid out `rows x columns` with `columns <= 4` costs exactly `rows` packets,
 *    except the leader's row, which costs one or two;
 *  - the leader is NEVER inside a piece, so `destination_count == node_count` for every
 *    piece and the barrier arithmetic has no sender-exclusion special case left in it;
 *  - the pieces are disjoint by construction, and it is checked anyway, because
 *    "disjoint" is the property that stops one piece's payload landing twice.
 *
 * @status Off-hardware. Built by tests/test_group_broadcast_partition.c with a mutation
 *   battery; consumed by the resident cooperative matmul.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_GROUP_BROADCAST_PARTITION_H
#define FT_TT_GROUP_BROADCAST_PARTITION_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tt_multicast.h"

/** The standing operational limit on destinations in ONE multicast packet. Measured, not
 *  chosen: five wedges a core at its next firmware boot, four does not. A caller may ask
 *  for fewer; asking for more is refused here rather than in a comment. */
#define FT_TT_GROUP_BROADCAST_HAZARD_LIMIT 4u

/** Most pieces one partition may hold. A `rows x columns` group with `columns <= 4` needs
 *  at most `rows + 1`, so eight covers every group up to sixteen members that this
 *  backend can place. The bound exists so a partition is a value rather than an
 *  allocation. */
#define FT_TT_GROUP_BROADCAST_MAX_RECTANGLES 8u

/** Most members a partition will consider. Matches FT_TT_OP_COOP_MAX_MEMBERS. */
#define FT_TT_GROUP_BROADCAST_MAX_MEMBERS 32u

/**
 * `rectangle_count` pieces covering every member except the leader, exactly once each.
 *
 * `destination_total` is carried separately rather than recomputed by the caller because
 * it is the number the LEADER's software acknowledgement counter is advanced by across
 * the whole block, and a caller deriving it a second way is a caller deriving it wrong.
 */
typedef struct ft_tt_group_broadcast_partition {
    ft_tt_multicast_rectangle rectangle[FT_TT_GROUP_BROADCAST_MAX_RECTANGLES];
    uint32_t rectangle_count;
    uint32_t destination_total;
} ft_tt_group_broadcast_partition;

/**
 * Partition `members` minus `leader` into rectangles of at most `max_destinations` nodes.
 *
 * @param members           TRANSLATED coordinates, in any order, which must EXACTLY fill
 *                          their own bounding box (the pigeonhole test in
 *                          ft_tt_multicast_rectangle_from_cores)
 * @param member_count      2..FT_TT_GROUP_BROADCAST_MAX_MEMBERS. One member has no
 *                          receivers, which is the zero-destination multicast in
 *                          different clothes and is refused
 * @param leader            must be one of `members`; it sends and does not receive
 * @param max_destinations  1..FT_TT_GROUP_BROADCAST_HAZARD_LIMIT
 * @param out               written only on success
 * @return FT_TT_OK; FT_TT_EINVAL for a NULL argument or a count out of range;
 *         FT_TT_ERROR for a member set that is not a rectangle, a leader outside it, a
 *         partition needing more pieces than the capacity, or any piece that fails its
 *         own rectangle check
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_group_broadcast_partition_build(const ft_tt_noc_coord *members, uint32_t member_count,
                                          ft_tt_noc_coord leader, uint32_t max_destinations,
                                          ft_tt_group_broadcast_partition *out);

/** Whether any node belongs to two pieces, or to none. Exposed because "covers every
 *  receiver exactly once" is the property the partition turns on, and a caller checking
 *  it by hand is a caller getting it wrong. */
bool ft_tt_group_broadcast_partition_covers(const ft_tt_group_broadcast_partition *partition,
                                            const ft_tt_noc_coord *members,
                                            uint32_t member_count, ft_tt_noc_coord leader);

#endif /* FT_TT_GROUP_BROADCAST_PARTITION_H */
