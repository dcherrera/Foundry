/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_split_broadcast_geometry.c — the split-broadcast refusals, OFF HARDWARE.
 *
 * These are the checks that keep experiment B (test_multicast_bench.c stage 3d) safe to
 * issue: two senders broadcasting at once into two rectangles, on a card where a wedge
 * costs a whole-card reset plus a link retrain. Two of the refusals protect the card
 * directly and both are easy to violate by picking the wrong index:
 *
 *  - a SENDER INSIDE A RECTANGLE. A multicast writes every node in its bounding box, so a
 *    sender caught in one has its own staging buffer overwritten mid-round. The failure
 *    would show up as a corrupt payload and be blamed on the split.
 *  - TWO RECTANGLES THAT OVERLAP. A receiver written twice per round by two senders with
 *    no ordering between them.
 *
 * Every check below is written so that deleting the guard it tests makes it go red, and
 * the healthy case is checked first — a gate that refuses everything passes every red.
 *
 * Build and run:
 *   make -f Makefile.multicast check
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_split_broadcast_geometry.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>

#define MAX_DESTINATIONS 4u

static ft_tt_noc_coord coord(uint32_t x, uint32_t y)
{
    ft_tt_noc_coord result;

    result.x = x;
    result.y = y;
    return result;
}

/** Two rows of four, at y = 5 and y = 6, x = 3..6. The shape experiment B runs. */
static void fill_two_rows_of_four(ft_tt_noc_coord *receivers)
{
    uint32_t row;
    uint32_t column;

    for (row = 0u; row < 2u; row++)
        for (column = 0u; column < 4u; column++)
            receivers[row * 4u + column] = coord(3u + column, 5u + row);
}

static void check_build(const char *what, const ft_tt_noc_coord *receivers, uint32_t rows,
                        uint32_t columns, const ft_tt_noc_coord *senders,
                        uint32_t sender_count, int wanted_rc)
{
    ft_tt_split_broadcast_plan plan;
    char why[192];
    int rc = ft_tt_split_broadcast_plan_build(receivers, rows, columns, senders, sender_count,
                                              MAX_DESTINATIONS, &plan, why, sizeof(why));

    ft_tt_test_check_fmt(what, rc == wanted_rc, "rc=%d (%s), wanted %d (%s) — %s", rc,
                         ft_tt_error_string(rc), wanted_rc, ft_tt_error_string(wanted_rc),
                         why);
}

/* ------------------------------------------------------------------------- */

static void check_the_shape_experiment_b_runs_is_accepted(void)
{
    ft_tt_noc_coord receivers[8];
    ft_tt_noc_coord senders[2];
    ft_tt_split_broadcast_plan plan;
    char why[192];
    int rc;

    ft_tt_test_step("two rows of four with both senders outside — the shape experiment B "
                    "runs, and the reason every refusal below means something");

    fill_two_rows_of_four(receivers);
    senders[0] = coord(3u, 9u);
    senders[1] = coord(4u, 9u);
    rc = ft_tt_split_broadcast_plan_build(receivers, 2u, 4u, senders, 2u, MAX_DESTINATIONS,
                                          &plan, why, sizeof(why));
    ft_tt_test_check_fmt("accepted", rc == FT_TT_OK, "rc=%d (%s) — %s", rc,
                         ft_tt_error_string(rc), why);
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("two groups, four destinations each, NEITHER over the limit",
                         plan.group_count == 2u &&
                             plan.rectangle[0].destination_count == 4u &&
                             plan.rectangle[1].destination_count == 4u,
                         "%u groups, destinations %u and %u", plan.group_count,
                         plan.rectangle[0].destination_count,
                         plan.rectangle[1].destination_count);
    ft_tt_test_check_fmt("the rows became two rectangles one row apart",
                         plan.rectangle[0].y_low == 5u && plan.rectangle[0].y_high == 5u &&
                             plan.rectangle[1].y_low == 6u && plan.rectangle[1].y_high == 6u &&
                             plan.rectangle[0].x_low == 3u && plan.rectangle[0].x_high == 6u,
                         "(%u,%u)..(%u,%u) and (%u,%u)..(%u,%u)", plan.rectangle[0].x_low,
                         plan.rectangle[0].y_low, plan.rectangle[0].x_high,
                         plan.rectangle[0].y_high, plan.rectangle[1].x_low,
                         plan.rectangle[1].y_low, plan.rectangle[1].x_high,
                         plan.rectangle[1].y_high);
    ft_tt_test_check("and neither sender is inside either rectangle",
                     !plan.rectangle[0].sender_inside && !plan.rectangle[1].sender_inside,
                     "sender_inside is false on both");
    ft_tt_test_check("the two rectangles are disjoint",
                     !ft_tt_split_broadcast_rectangles_overlap(&plan.rectangle[0],
                                                               &plan.rectangle[1]),
                     "row 5 against row 6");
}

/** The two refusals that protect the card. */
static void check_a_broadcast_that_would_reach_the_wrong_core_is_refused(void)
{
    ft_tt_noc_coord receivers[8];
    ft_tt_noc_coord senders[2];

    ft_tt_test_step("a plan whose broadcast would reach a core it does not own is refused");

    /* Sender 1 sitting in row 0: legal as a coordinate, and it would be written by group
     * 0's broadcast. This is the mistake a wrong FT_TT_MULTICAST_SENDER_B_INDEX makes. */
    fill_two_rows_of_four(receivers);
    senders[0] = coord(3u, 9u);
    senders[1] = coord(5u, 5u);
    check_build("a sender INSIDE the other group's rectangle is refused", receivers, 2u, 4u,
                senders, 2u, FT_TT_ERROR);

    /* A sender inside its OWN rectangle. Legal for a multicast in general — it just does
     * not receive — but here it means one of the eight receivers is also the feeder, which
     * is a different experiment. */
    senders[0] = coord(4u, 5u);
    senders[1] = coord(4u, 9u);
    check_build("a sender inside its OWN rectangle is refused too", receivers, 2u, 4u,
                senders, 2u, FT_TT_ERROR);

    /* Both groups handed the SAME row: the rectangles are identical, so every receiver is
     * written twice per round by two senders with no ordering between them. */
    {
        ft_tt_noc_coord doubled[8];
        uint32_t index;

        fill_two_rows_of_four(doubled);
        for (index = 0u; index < 4u; index++)
            doubled[4u + index] = doubled[index];
        senders[0] = coord(3u, 9u);
        senders[1] = coord(4u, 9u);
        check_build("two groups over the SAME row are refused as overlapping", doubled, 2u,
                    4u, senders, 2u, FT_TT_ERROR);
    }
}

/** The limit itself, and the argument checks. */
static void check_the_destination_limit_and_the_arguments(void)
{
    ft_tt_noc_coord receivers[10];
    ft_tt_noc_coord senders[2];
    uint32_t row;
    uint32_t column;

    ft_tt_test_step("the per-packet limit and the argument checks");

    for (row = 0u; row < 2u; row++)
        for (column = 0u; column < 5u; column++)
            receivers[row * 5u + column] = coord(3u + column, 5u + row);
    senders[0] = coord(3u, 9u);
    senders[1] = coord(4u, 9u);
    check_build("a FIVE-wide group is refused — this is the standing rule, in code",
                receivers, 2u, 5u, senders, 2u, FT_TT_ERROR);

    check_build("and the same rows at four wide are accepted, so the refusal above is "
                "about the WIDTH and not the rows", receivers, 1u, 4u, senders, 1u,
                FT_TT_OK);

    check_build("one sender for two groups is refused", receivers, 2u, 4u, senders, 1u,
                FT_TT_EINVAL);
    check_build("zero groups is refused", receivers, 0u, 4u, senders, 0u, FT_TT_EINVAL);
    check_build("zero receivers per group is refused", receivers, 2u, 0u, senders, 2u,
                FT_TT_EINVAL);
    check_build("more groups than the plan holds is refused", receivers,
                FT_TT_SPLIT_MAX_GROUPS + 1u, 1u, senders, FT_TT_SPLIT_MAX_GROUPS + 1u,
                FT_TT_EINVAL);
    check_build("a NULL receiver array is refused", NULL, 2u, 4u, senders, 2u, FT_TT_EINVAL);

    /* A row that is not contiguous does not FILL its bounding box, so the box contains a
     * core nobody asked for. The multicast module is what catches it; this proves the plan
     * asks it rather than doing its own arithmetic. */
    {
        ft_tt_noc_coord gapped[4];

        gapped[0] = coord(3u, 5u);
        gapped[1] = coord(4u, 5u);
        gapped[2] = coord(5u, 5u);
        gapped[3] = coord(7u, 5u);
        check_build("a row with a GAP in it is refused: its bounding box holds a core the "
                    "group does not own", gapped, 1u, 4u, senders, 1u, FT_TT_ERROR);
    }
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    ft_tt_test_step("the split-broadcast geometry — off hardware");

    check_the_shape_experiment_b_runs_is_accepted();
    check_a_broadcast_that_would_reach_the_wrong_core_is_refused();
    check_the_destination_limit_and_the_arguments();

    return ft_tt_test_finish();
}
