/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_group_broadcast_partition.c — splitting ONE cooperating group's broadcast into
 * pieces of at most four destinations, OFF HARDWARE.
 *
 * The thing being protected is not a failed transaction. A multicast delivers to every
 * node of a closed box, so a piece that is one cell too wide writes into a core another
 * residency owns, and a piece that leaves a member out means that member computes from
 * the PREVIOUS block with nothing reporting an error. Both are silent, and both are
 * arithmetic — so they are proven here rather than on silicon.
 *
 * Three shapes of check, deliberately:
 *
 *  - **Literal expected corners.** A partition checked only against its own
 *    `destination_total` would agree with itself: a piece one column too wide and a piece
 *    one column too narrow sum correctly. So the corners of every piece are written out by
 *    hand for the fixed fixtures.
 *  - **Translation invariance**, taken at a base derived from the pid and the clock. The
 *    literal expectations above are a fixed pattern, and a fixed pattern is the vacuity
 *    trap this project has been bitten by four times: an implementation that answered from
 *    a constant would pass them. Moving the whole group and requiring the pieces to move
 *    with it cannot be satisfied by a constant.
 *  - **The refusals**, each written so that deleting the guard it names makes it red.
 *    `tools/run_broadcast_partition_gates.sh` is the mutation battery that proves it.
 *
 * Build and run:
 *   make -f Makefile.broadcastpartition check
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_group_broadcast_partition.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static ft_tt_noc_coord coord(uint32_t x, uint32_t y)
{
    ft_tt_noc_coord result;

    result.x = x;
    result.y = y;
    return result;
}

/** Fill `out` with a `rows x columns` block whose top-left cell is (x, y), row major —
 *  the order ft_tt_group_rectangle_coordinates() produces. */
static void fill_block(ft_tt_noc_coord *out, uint32_t x, uint32_t y, uint32_t rows,
                       uint32_t columns)
{
    uint32_t row;
    uint32_t column;

    for (row = 0u; row < rows; row++)
        for (column = 0u; column < columns; column++)
            out[row * columns + column] = coord(x + column, y + row);
}

/** True when `piece` is exactly the row run `[x_first, x_last]` of row `y` and names
 *  `x_last - x_first + 1` destinations. */
static bool piece_is(const ft_tt_multicast_rectangle *piece, uint32_t x_first, uint32_t y,
                     uint32_t x_last)
{
    return piece->x_low == x_first && piece->x_high == x_last && piece->y_low == y &&
           piece->y_high == y && piece->destination_count == x_last - x_first + 1u &&
           !piece->sender_inside;
}

/** No piece may name more than the limit. Checked separately from the corners because it
 *  is the whole hazard: five destinations in one packet wedges a core at its next
 *  firmware boot. */
static bool every_piece_within(const ft_tt_group_broadcast_partition *partition, uint32_t limit)
{
    uint32_t index;

    for (index = 0u; index < partition->rectangle_count; index++)
        if (partition->rectangle[index].destination_count > limit)
            return false;
    return true;
}

/* ------------------------------------------------------------------------- */

static void check_a_row_of_four_led_from_the_end(void)
{
    ft_tt_noc_coord members[4];
    ft_tt_group_broadcast_partition partition;
    int rc;

    ft_tt_test_step("1x4 group at (13,7), leader at the left end");

    fill_block(members, 13u, 7u, 1u, 4u);
    memset(&partition, 0, sizeof(partition));
    rc = ft_tt_group_broadcast_partition_build(members, 4u, coord(13u, 7u),
                                               FT_TT_GROUP_BROADCAST_HAZARD_LIMIT, &partition);
    ft_tt_test_check_fmt("build succeeds", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("one piece", partition.rectangle_count == 1u, "%u",
                         partition.rectangle_count);
    ft_tt_test_check_fmt("three destinations", partition.destination_total == 3u, "%u",
                         partition.destination_total);
    ft_tt_test_check("the piece is (14,7)..(16,7)",
                     partition.rectangle_count == 1u &&
                         piece_is(&partition.rectangle[0], 14u, 7u, 16u),
                     "the leader's own cell is removed and nothing else is");
    ft_tt_test_check("covers every receiver exactly once",
                     ft_tt_group_broadcast_partition_covers(&partition, members, 4u,
                                                            coord(13u, 7u)),
                     "and the leader zero times");
}

static void check_a_row_of_four_led_from_the_middle(void)
{
    ft_tt_noc_coord members[4];
    ft_tt_group_broadcast_partition partition;
    int rc;

    /* The leader inside the run is the case the row split exists for: removing a cell
     * from the middle of a box leaves TWO boxes, and a partition that emitted one would
     * broadcast to the leader's own staging block. */
    ft_tt_test_step("1x4 group at (13,7), leader in the middle at (14,7)");

    fill_block(members, 13u, 7u, 1u, 4u);
    memset(&partition, 0, sizeof(partition));
    rc = ft_tt_group_broadcast_partition_build(members, 4u, coord(14u, 7u),
                                               FT_TT_GROUP_BROADCAST_HAZARD_LIMIT, &partition);
    ft_tt_test_check_fmt("build succeeds", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("two pieces", partition.rectangle_count == 2u, "%u",
                         partition.rectangle_count);
    ft_tt_test_check("the left piece is (13,7)..(13,7)",
                     partition.rectangle_count == 2u &&
                         piece_is(&partition.rectangle[0], 13u, 7u, 13u),
                     "one cell, one destination");
    ft_tt_test_check("the right piece is (15,7)..(16,7)",
                     partition.rectangle_count == 2u &&
                         piece_is(&partition.rectangle[1], 15u, 7u, 16u),
                     "two cells, two destinations");
    ft_tt_test_check_fmt("three destinations", partition.destination_total == 3u, "%u",
                         partition.destination_total);
    ft_tt_test_check("no piece contains the leader",
                     ft_tt_group_broadcast_partition_covers(&partition, members, 4u,
                                                            coord(14u, 7u)),
                     "which is what stops the leader overwriting its own A block");
}

static void check_eight_as_two_by_four(void)
{
    ft_tt_noc_coord members[8];
    ft_tt_group_broadcast_partition partition;
    int rc;

    /* This is the shape the plan's experiment B ran 20/20 PASS: N=8 reached as two
     * disjoint <=4-destination broadcasts instead of one 8-destination one. */
    ft_tt_test_step("2x4 group at (3,7), leader at the top-left — N=8 as TWO packets");

    fill_block(members, 3u, 7u, 2u, 4u);
    memset(&partition, 0, sizeof(partition));
    rc = ft_tt_group_broadcast_partition_build(members, 8u, coord(3u, 7u),
                                               FT_TT_GROUP_BROADCAST_HAZARD_LIMIT, &partition);
    ft_tt_test_check_fmt("build succeeds", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("two pieces", partition.rectangle_count == 2u, "%u",
                         partition.rectangle_count);
    ft_tt_test_check("the leader's row is (4,7)..(6,7)",
                     partition.rectangle_count == 2u &&
                         piece_is(&partition.rectangle[0], 4u, 7u, 6u),
                     "three destinations");
    ft_tt_test_check("the second row is (3,8)..(6,8)",
                     partition.rectangle_count == 2u &&
                         piece_is(&partition.rectangle[1], 3u, 8u, 6u),
                     "four destinations, the limit exactly");
    ft_tt_test_check_fmt("seven destinations", partition.destination_total == 7u, "%u",
                         partition.destination_total);
    ft_tt_test_check("no piece exceeds four destinations",
                     every_piece_within(&partition, FT_TT_GROUP_BROADCAST_HAZARD_LIMIT),
                     "the standing hazard rule, as arithmetic");
    ft_tt_test_check("covers every receiver exactly once",
                     ft_tt_group_broadcast_partition_covers(&partition, members, 8u,
                                                            coord(3u, 7u)),
                     "no member left reading the previous block");
}

static void check_sixteen_as_four_by_four(void)
{
    ft_tt_noc_coord members[16];
    ft_tt_group_broadcast_partition partition;
    uint32_t index;
    bool corners_right;
    int rc;

    ft_tt_test_step("4x4 group at (3,4), leader at the top-left — N=16 as FOUR packets");

    fill_block(members, 3u, 4u, 4u, 4u);
    memset(&partition, 0, sizeof(partition));
    rc = ft_tt_group_broadcast_partition_build(members, 16u, coord(3u, 4u),
                                               FT_TT_GROUP_BROADCAST_HAZARD_LIMIT, &partition);
    ft_tt_test_check_fmt("build succeeds", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("four pieces", partition.rectangle_count == 4u, "%u",
                         partition.rectangle_count);
    corners_right = partition.rectangle_count == 4u &&
                    piece_is(&partition.rectangle[0], 4u, 4u, 6u);
    for (index = 1u; index < partition.rectangle_count; index++)
        corners_right = corners_right &&
                        piece_is(&partition.rectangle[index], 3u, 4u + index, 6u);
    ft_tt_test_check("the pieces are the four rows, the leader's short by one",
                     corners_right, "3 + 4 + 4 + 4");
    ft_tt_test_check_fmt("fifteen destinations", partition.destination_total == 15u, "%u",
                         partition.destination_total);
    ft_tt_test_check("no piece exceeds four destinations",
                     every_piece_within(&partition, FT_TT_GROUP_BROADCAST_HAZARD_LIMIT), "");
}

static void check_a_tighter_limit_splits_a_row(void)
{
    ft_tt_noc_coord members[8];
    ft_tt_group_broadcast_partition partition;
    int rc;

    /* `max_destinations` is a parameter and not a constant, so it has to be shown to
     * MOVE something. A limit of two turns the four-wide second row into two packets. */
    ft_tt_test_step("2x4 group, max_destinations = 2");

    fill_block(members, 3u, 7u, 2u, 4u);
    memset(&partition, 0, sizeof(partition));
    rc = ft_tt_group_broadcast_partition_build(members, 8u, coord(3u, 7u), 2u, &partition);
    ft_tt_test_check_fmt("build succeeds", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("four pieces", partition.rectangle_count == 4u, "%u",
                         partition.rectangle_count);
    ft_tt_test_check("the leader's row splits 2 + 1",
                     partition.rectangle_count == 4u &&
                         piece_is(&partition.rectangle[0], 4u, 7u, 5u) &&
                         piece_is(&partition.rectangle[1], 6u, 7u, 6u),
                     "the remainder is the LAST chunk, never the first");
    ft_tt_test_check("the second row splits 2 + 2",
                     partition.rectangle_count == 4u &&
                         piece_is(&partition.rectangle[2], 3u, 8u, 4u) &&
                         piece_is(&partition.rectangle[3], 5u, 8u, 6u),
                     "");
    ft_tt_test_check("no piece exceeds two destinations",
                     every_piece_within(&partition, 2u), "");
    ft_tt_test_check_fmt("still seven destinations", partition.destination_total == 7u, "%u",
                         partition.destination_total);
}

static void check_translation_invariance(void)
{
    ft_tt_noc_coord members[8];
    ft_tt_group_broadcast_partition partition;
    uint32_t base_x;
    uint32_t base_y;
    bool shifted_right;
    uint32_t index;
    int rc;

    /* SEEDED, because every expectation above is a fixed pattern and an implementation
     * answering from a constant would satisfy all of them. The group is moved to a
     * position neither the test nor the code can have been written around, and the pieces
     * are required to move with it. */
    base_x = 3u + (uint32_t)((unsigned long)getpid() % 9u);
    base_y = 2u + (uint32_t)((unsigned long)time(NULL) % 8u);
    ft_tt_test_step("2x4 group at a SEEDED base (%u,%u) — the pieces must move with it",
                    base_x, base_y);

    fill_block(members, base_x, base_y, 2u, 4u);
    memset(&partition, 0, sizeof(partition));
    rc = ft_tt_group_broadcast_partition_build(members, 8u, coord(base_x, base_y),
                                               FT_TT_GROUP_BROADCAST_HAZARD_LIMIT, &partition);
    ft_tt_test_check_fmt("build succeeds", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    shifted_right = partition.rectangle_count == 2u &&
                    piece_is(&partition.rectangle[0], base_x + 1u, base_y, base_x + 3u) &&
                    piece_is(&partition.rectangle[1], base_x, base_y + 1u, base_x + 3u);
    ft_tt_test_check_fmt("the pieces are the translated rows", shifted_right,
                         "(%u,%u)..(%u,%u) and (%u,%u)..(%u,%u)",
                         partition.rectangle[0].x_low, partition.rectangle[0].y_low,
                         partition.rectangle[0].x_high, partition.rectangle[0].y_high,
                         partition.rectangle[1].x_low, partition.rectangle[1].y_low,
                         partition.rectangle[1].x_high, partition.rectangle[1].y_high);
    for (index = 0u; index < partition.rectangle_count; index++) {
        char what[64];

        snprintf(what, sizeof(what), "piece %u names no coordinate above 63", index);
        ft_tt_test_check_fmt(what,
                             partition.rectangle[index].x_high <= FT_TT_MULTICAST_MAX_COORDINATE &&
                                 partition.rectangle[index].y_high <=
                                     FT_TT_MULTICAST_MAX_COORDINATE,
                             "(%u,%u)", partition.rectangle[index].x_high,
                             partition.rectangle[index].y_high);
    }
    ft_tt_test_check("covers every receiver exactly once",
                     ft_tt_group_broadcast_partition_covers(&partition, members, 8u,
                                                            coord(base_x, base_y)),
                     "");
}

/* ------------------------------------------------------------------------- */

static void check_the_refusals(void)
{
    ft_tt_noc_coord members[16];
    ft_tt_group_broadcast_partition partition;
    int rc;

    ft_tt_test_step("the refusals");

    fill_block(members, 3u, 7u, 2u, 4u);

    rc = ft_tt_group_broadcast_partition_build(NULL, 8u, coord(3u, 7u), 4u, &partition);
    ft_tt_test_check_fmt("NULL members refused", rc == FT_TT_EINVAL, "rc=%d", rc);

    rc = ft_tt_group_broadcast_partition_build(members, 8u, coord(3u, 7u), 4u, NULL);
    ft_tt_test_check_fmt("NULL output refused", rc == FT_TT_EINVAL, "rc=%d", rc);

    rc = ft_tt_group_broadcast_partition_build(members, 1u, coord(3u, 7u), 4u, &partition);
    ft_tt_test_check_fmt("one member refused", rc == FT_TT_EINVAL, "rc=%d", rc);

    rc = ft_tt_group_broadcast_partition_build(members, 17u, coord(3u, 7u), 4u, &partition);
    ft_tt_test_check_fmt("seventeen members refused", rc == FT_TT_EINVAL, "rc=%d", rc);

    rc = ft_tt_group_broadcast_partition_build(members, 8u, coord(3u, 7u), 0u, &partition);
    ft_tt_test_check_fmt("zero destinations per packet refused", rc == FT_TT_EINVAL, "rc=%d",
                         rc);

    /* THE HAZARD ITSELF. Five destinations in one packet is what wedges a core at its
     * next firmware boot, and this is the line that stops a caller asking for it. */
    rc = ft_tt_group_broadcast_partition_build(members, 8u, coord(3u, 7u), 5u, &partition);
    ft_tt_test_check_fmt("five destinations per packet refused", rc == FT_TT_EINVAL, "rc=%d",
                         rc);

    /* A set of cores that is not a rectangle, produced the way a real caller produces
     * one: a run of consecutive indices that wraps a row. */
    fill_block(members, 3u, 7u, 1u, 4u);
    members[3] = coord(3u, 8u);
    rc = ft_tt_group_broadcast_partition_build(members, 4u, coord(3u, 7u), 4u, &partition);
    ft_tt_test_check_fmt("a member set that is not a rectangle refused", rc != FT_TT_OK,
                         "rc=%d", rc);

    fill_block(members, 3u, 7u, 2u, 4u);
    rc = ft_tt_group_broadcast_partition_build(members, 8u, coord(9u, 7u), 4u, &partition);
    ft_tt_test_check_fmt("a leader outside the group refused", rc == FT_TT_ERROR, "rc=%d", rc);

    /* Sixteen members one column wide needs fifteen pieces, which is more than the
     * partition can hold. It has to be a refusal and not a truncated plan: a truncated
     * plan leaves members reading the previous block with nothing reporting an error. */
    fill_block(members, 3u, 2u, 16u, 1u);
    rc = ft_tt_group_broadcast_partition_build(members, 16u, coord(3u, 2u), 4u, &partition);
    ft_tt_test_check_fmt("a partition needing more pieces than the capacity refused",
                         rc == FT_TT_ERROR, "rc=%d", rc);
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    printf("ft_tt group broadcast partition — off hardware\n");
    printf("==============================================\n");

    check_a_row_of_four_led_from_the_end();
    check_a_row_of_four_led_from_the_middle();
    check_eight_as_two_by_four();
    check_sixteen_as_four_by_four();
    check_a_tighter_limit_splits_a_row();
    check_translation_invariance();
    check_the_refusals();

    return ft_tt_test_finish();
}
