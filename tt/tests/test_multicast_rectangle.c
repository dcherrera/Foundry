/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_multicast_rectangle.c — the multicast rectangle rules, OFF HARDWARE.
 *
 * Everything in ft_tt_multicast.c and the pure half of ft_tt_group_shared_operand.c is
 * arithmetic, and it is separated out precisely so it can be proven on the Mac while
 * somebody else owns the silicon. What it is protecting against is not a failed
 * transaction: a wrong rectangle writes into a core another group is using, and a
 * coordinate that does not fit the 6-bit node-id field lands on a DIFFERENT live endpoint,
 * which on this board is a hung NoC and a walk to the machine.
 *
 * The checks that matter here are the REFUSALS, and each one is written so that deleting
 * the guard it tests makes it go red. In particular:
 *
 *  - The 64-bit encoding is checked against LITERAL expected values, not against the
 *    kernel-side header. Two implementations that agree with each other and disagree with
 *    the hardware is exactly the failure a round trip cannot see, so the
 *    expected words are computed by hand from the field layout written out in the comments.
 *  - The NoC-1 corner swap is checked as a DIFFERENT value from NoC 0's, on a rectangle
 *    whose two corners are not equal — a square would make the swap invisible, which is
 *    the "an assert on a shape whose two dimensions happen to be equal is vacuous" trap.
 *  - The fill check is given a set that is NOT a rectangle in the exact way a real caller
 *    would produce one: a run of consecutive live indices that wraps a row.
 *
 * Build and run:
 *   make -f Makefile.multicast check
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_group_shared_operand.h"
#include "ft_tt_multicast.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>

#define L1_ADDRESS 0x00010000u
#define PAYLOAD_BYTES 2048u

static ft_tt_noc_coord coord(uint32_t x, uint32_t y)
{
    ft_tt_noc_coord result;

    result.x = x;
    result.y = y;
    return result;
}

/* ------------------------------------------------------------------------- */

static void check_rectangle_from_a_row(void)
{
    ft_tt_noc_coord members[4];
    ft_tt_multicast_rectangle rectangle;
    int rc;

    ft_tt_test_step("a 1x4 row of cores, sender outside it");

    members[0] = coord(3u, 7u);
    members[1] = coord(4u, 7u);
    members[2] = coord(5u, 7u);
    members[3] = coord(6u, 7u);
    rc = ft_tt_multicast_rectangle_from_cores(members, 4u, coord(2u, 7u), &rectangle);

    ft_tt_test_check_fmt("accepted", rc == FT_TT_OK, "rc=%d (%s)", rc, ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("the corners are the row's ends", rectangle.x_low == 3u &&
                             rectangle.x_high == 6u && rectangle.y_low == 7u &&
                             rectangle.y_high == 7u,
                         "(%u,%u)..(%u,%u)", rectangle.x_low, rectangle.y_low,
                         rectangle.x_high, rectangle.y_high);
    ft_tt_test_check_fmt("4 nodes, and 4 destinations because the sender is outside",
                         rectangle.node_count == 4u && rectangle.destination_count == 4u &&
                             !rectangle.sender_inside,
                         "nodes %u, destinations %u, sender %s", rectangle.node_count,
                         rectangle.destination_count,
                         rectangle.sender_inside ? "inside" : "outside");
}

static void check_sender_inside_costs_one_destination(void)
{
    ft_tt_noc_coord members[4];
    ft_tt_multicast_rectangle rectangle;
    int rc;

    ft_tt_test_step("the sender INSIDE the rectangle — a non-loopback multicast does not "
                    "write to self, so the destination count is one lower");

    members[0] = coord(3u, 7u);
    members[1] = coord(4u, 7u);
    members[2] = coord(5u, 7u);
    members[3] = coord(6u, 7u);
    rc = ft_tt_multicast_rectangle_from_cores(members, 4u, coord(5u, 7u), &rectangle);

    ft_tt_test_check_fmt("accepted, 4 nodes but 3 destinations",
                         rc == FT_TT_OK && rectangle.node_count == 4u &&
                             rectangle.destination_count == 3u && rectangle.sender_inside,
                         "rc=%d, nodes %u, destinations %u, sender %s", rc,
                         rectangle.node_count, rectangle.destination_count,
                         rectangle.sender_inside ? "inside" : "outside");
    ft_tt_test_check("this is the number the write BARRIER counts acknowledgements against; "
                     "one too many never returns and one too few returns early",
                     rectangle.destination_count == 3u, "destination_count");
}

static void check_a_wrapped_run_is_refused(void)
{
    ft_tt_noc_coord members[4];
    ft_tt_multicast_rectangle rectangle;
    int rc;

    ft_tt_test_step("a run of consecutive live indices that WRAPS a row is refused");

    /* Exactly what ft_tt_coord_tensix_by_index() produces for indices 110..113 on a
     * 14-column card: the last two cores of one row and the first two of the next. Their
     * bounding box is 14 wide and 2 tall — 28 nodes, 24 of which nobody asked for. */
    members[0] = coord(13u, 7u);
    members[1] = coord(14u, 7u);
    members[2] = coord(1u, 8u);
    members[3] = coord(2u, 8u);
    rc = ft_tt_multicast_rectangle_from_cores(members, 4u, coord(12u, 7u), &rectangle);

    ft_tt_test_check_fmt("REFUSED, because the set does not fill its bounding box",
                         rc == FT_TT_ERROR, "rc=%d (%s) — a multicast writes EVERY node in "
                         "the box, so accepting this would write into cores another group "
                         "owns", rc, ft_tt_error_string(rc));
}

static void check_a_duplicate_member_is_refused(void)
{
    ft_tt_noc_coord members[4];
    ft_tt_multicast_rectangle rectangle;
    int rc;

    ft_tt_test_step("a duplicated member is refused BEFORE the fill check");

    /* Without the distinctness check this passes: four members, a 2x2 box, four nodes —
     * and one node of the box is not a member at all. The fill test is a COUNT, so a
     * duplicate is exactly what defeats it. */
    members[0] = coord(3u, 7u);
    members[1] = coord(4u, 7u);
    members[2] = coord(3u, 8u);
    members[3] = coord(3u, 8u);
    rc = ft_tt_multicast_rectangle_from_cores(members, 4u, coord(2u, 7u), &rectangle);

    ft_tt_test_check_fmt("REFUSED", rc == FT_TT_EINVAL, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
}

static void check_a_lone_sender_is_refused(void)
{
    ft_tt_noc_coord members[1];
    ft_tt_multicast_rectangle rectangle;
    int rc;

    ft_tt_test_step("a rectangle that degenerates onto the SENDER ALONE is refused");

    members[0] = coord(5u, 7u);
    rc = ft_tt_multicast_rectangle_from_cores(members, 1u, coord(5u, 7u), &rectangle);

    ft_tt_test_check_fmt("REFUSED with zero destinations", rc == FT_TT_ERROR,
                         "rc=%d (%s) — this is the zero-length NoC transfer in different "
                         "clothes, the shape that took the card to `Mem- BusMaster-` on "
                         "2026-08-12", rc, ft_tt_error_string(rc));
}

static void check_an_out_of_range_coordinate_is_refused(void)
{
    ft_tt_noc_coord members[2];
    ft_tt_multicast_rectangle rectangle;
    int rc;

    ft_tt_test_step("a coordinate above the 6-bit node-id field is refused");

    members[0] = coord(3u, 7u);
    members[1] = coord(64u, 7u);
    rc = ft_tt_multicast_rectangle_from_cores(members, 2u, coord(2u, 7u), &rectangle);

    ft_tt_test_check_fmt("REFUSED", rc == FT_TT_EINVAL,
                         "rc=%d (%s) — 64 does not fail, it TRUNCATES to 0 and names a "
                         "different live endpoint", rc, ft_tt_error_string(rc));
}

static void check_inverted_corners_are_refused(void)
{
    ft_tt_multicast_rectangle rectangle;
    int rc = ft_tt_multicast_rectangle_from_corners(6u, 7u, 3u, 7u, coord(2u, 7u), &rectangle);

    ft_tt_test_step("inverted corners are refused");
    ft_tt_test_check_fmt("REFUSED", rc != FT_TT_OK, "rc=%d (%s)", rc, ft_tt_error_string(rc));
}

/**
 * The encoding, against literal expected words.
 *
 * Field layout (as measured on this card) —
 *   x_start << (36 + 12), y_start << (36 + 18), x_end << 36, y_end << (36 + 6), | addr
 * and the corner order — NoC 0 takes (low, high) as (start, end), NoC 1 takes
 * them reversed.
 *
 * The rectangle below is 3..6 x 7..9, deliberately NOT square, so the swap changes the
 * answer. A square rectangle would encode identically on both NoCs and the check would be
 * vacuous while looking thorough.
 */
static void check_the_encoding_and_the_noc1_swap(void)
{
    ft_tt_multicast_rectangle rectangle;
    uint64_t noc0 = 0u;
    uint64_t noc1 = 0u;
    uint64_t expected_noc0;
    uint64_t expected_noc1;
    int rc;

    ft_tt_test_step("the 64-bit encoding, and the NoC-1 corner swap");

    rc = ft_tt_multicast_rectangle_from_corners(3u, 7u, 6u, 9u, coord(2u, 7u), &rectangle);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("a 4x3 rectangle built", false, ft_tt_error_string(rc));
        return;
    }
    ft_tt_test_check_fmt("12 nodes, 12 destinations", rectangle.node_count == 12u &&
                             rectangle.destination_count == 12u,
                         "nodes %u, destinations %u", rectangle.node_count,
                         rectangle.destination_count);

    /* NoC 0: start = (3, 7), end = (6, 9). */
    expected_noc0 = ((uint64_t)3u << 48) | ((uint64_t)7u << 54) | ((uint64_t)6u << 36) |
                    ((uint64_t)9u << 42) | (uint64_t)L1_ADDRESS;
    /* NoC 1: start = (6, 9), end = (3, 7). */
    expected_noc1 = ((uint64_t)6u << 48) | ((uint64_t)9u << 54) | ((uint64_t)3u << 36) |
                    ((uint64_t)7u << 42) | (uint64_t)L1_ADDRESS;

    rc = ft_tt_multicast_encode_address(&rectangle, 0u, L1_ADDRESS, &noc0);
    ft_tt_test_check_fmt("NoC 0 encodes the LOW corner as `start`",
                         rc == FT_TT_OK && noc0 == expected_noc0,
                         "got 0x%016llx, expected 0x%016llx", (unsigned long long)noc0,
                         (unsigned long long)expected_noc0);

    rc = ft_tt_multicast_encode_address(&rectangle, 1u, L1_ADDRESS, &noc1);
    ft_tt_test_check_fmt("NoC 1 encodes them REVERSED", rc == FT_TT_OK && noc1 == expected_noc1,
                         "got 0x%016llx, expected 0x%016llx", (unsigned long long)noc1,
                         (unsigned long long)expected_noc1);

    ft_tt_test_check_fmt("and the two differ, so the swap is load-bearing on this shape",
                         noc0 != noc1, "0x%016llx vs 0x%016llx", (unsigned long long)noc0,
                         (unsigned long long)noc1);
}

static void check_payload_rules(void)
{
    ft_tt_test_step("payload refusals");

    ft_tt_test_check_fmt("a legal payload is accepted",
                         ft_tt_multicast_check_payload(L1_ADDRESS, PAYLOAD_BYTES) == FT_TT_OK,
                         "0x%08x + %u", L1_ADDRESS, PAYLOAD_BYTES);
    ft_tt_test_check("ZERO BYTES is refused",
                     ft_tt_multicast_check_payload(L1_ADDRESS, 0u) == FT_TT_EINVAL,
                     "the shape that took the card down on 2026-08-12");
    ft_tt_test_check("a mis-aligned address is refused",
                     ft_tt_multicast_check_payload(L1_ADDRESS + 8u, PAYLOAD_BYTES) ==
                         FT_TT_EINVAL,
                     "L1 wants 16 bytes in address AND length");
    ft_tt_test_check("a mis-aligned length is refused",
                     ft_tt_multicast_check_payload(L1_ADDRESS, PAYLOAD_BYTES + 8u) ==
                         FT_TT_EINVAL,
                     "16-byte length rule");
    ft_tt_test_check("a payload that leaves L1 is refused",
                     ft_tt_multicast_check_payload(1536u * 1024u - 16u, 2048u) == FT_TT_EINVAL,
                     "1536 KiB is the whole of L1");
}

/* ------------------------------------------------------------------------- */

static ft_tt_group_shared_operand_layout good_layout(void)
{
    ft_tt_group_shared_operand_layout layout;

    layout.operand_address = L1_ADDRESS;
    layout.operand_bytes = PAYLOAD_BYTES;
    layout.ready_semaphore_address = 0x2040u;
    layout.arrival_semaphore_address = 0x2050u;
    layout.valid_word_address = 0x2060u;
    return layout;
}

static void check_shared_operand_binding(void)
{
    ft_tt_noc_coord members[4];
    ft_tt_group_shared_operand_layout layout = good_layout();
    ft_tt_group_shared_operand operand;
    int rc;

    ft_tt_test_step("binding a shared operand onto a group");

    members[0] = coord(3u, 7u);
    members[1] = coord(4u, 7u);
    members[2] = coord(5u, 7u);
    members[3] = coord(6u, 7u);

    /* The leader INSIDE the rectangle: the natural cooperating-matmul layout. Four cores,
     * one of them also reads DRAM, so three receive. */
    rc = ft_tt_group_shared_operand_bind(members, 4u, members[0], &layout, &operand);
    ft_tt_test_check_fmt("a 4-core rectangle whose LEADER is one of them binds at 3 receivers",
                         rc == FT_TT_OK && operand.receiver_count == 3u &&
                             operand.member_count == 4u && operand.rectangle.sender_inside,
                         "rc=%d (%s), receivers %u, members %u", rc, ft_tt_error_string(rc),
                         operand.receiver_count, operand.member_count);

    /* The leader OUTSIDE it: a dedicated feeder. Four cores receive and the group is five. */
    rc = ft_tt_group_shared_operand_bind(members, 4u, coord(2u, 7u), &layout, &operand);
    ft_tt_test_check_fmt("a leader OUTSIDE the rectangle binds at 4 receivers and 5 members",
                         rc == FT_TT_OK && operand.receiver_count == 4u &&
                             operand.member_count == 5u && !operand.rectangle.sender_inside,
                         "rc=%d (%s), receivers %u, members %u", rc, ft_tt_error_string(rc),
                         operand.receiver_count, operand.member_count);

    rc = ft_tt_group_shared_operand_bind(members, 1u, members[0], &layout, &operand);
    ft_tt_test_check_fmt("ONE receiver that IS the leader is refused", rc == FT_TT_ERROR,
                         "rc=%d (%s) — that rectangle has zero destinations", rc,
                         ft_tt_error_string(rc));

    rc = ft_tt_group_shared_operand_bind(members, 0u, members[0], &layout, &operand);
    ft_tt_test_check_fmt("ZERO receivers is refused", rc == FT_TT_EINVAL, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
}

static void check_shared_operand_layout_refusals(void)
{
    ft_tt_noc_coord members[2];
    ft_tt_group_shared_operand operand;
    ft_tt_group_shared_operand_layout layout;
    int rc;

    ft_tt_test_step("layouts that cannot work are refused");

    members[0] = coord(3u, 7u);
    members[1] = coord(4u, 7u);

    /* The arrival semaphore inside the operand region. The leader multicasts into BOTH in
     * the same round, so one silently destroys the other — and the symptom is a receiver
     * that reads the previous round's bytes, not an error. */
    layout = good_layout();
    layout.arrival_semaphore_address = L1_ADDRESS + 64u;
    rc = ft_tt_group_shared_operand_bind(members, 2u, coord(2u, 7u), &layout, &operand);
    ft_tt_test_check_fmt("an arrival semaphore INSIDE the operand region is refused",
                         rc == FT_TT_ERROR, "rc=%d (%s)", rc, ft_tt_error_string(rc));

    layout = good_layout();
    layout.valid_word_address = layout.ready_semaphore_address;
    rc = ft_tt_group_shared_operand_bind(members, 2u, coord(2u, 7u), &layout, &operand);
    ft_tt_test_check_fmt("two control words at the same address are refused",
                         rc == FT_TT_ERROR, "rc=%d (%s) — on the LEADER all three live in "
                         "one L1, so an alias is a deadlock", rc, ft_tt_error_string(rc));

    layout = good_layout();
    layout.ready_semaphore_address += 2u;
    rc = ft_tt_group_shared_operand_bind(members, 2u, coord(2u, 7u), &layout, &operand);
    ft_tt_test_check_fmt("a semaphore that is not 4-byte aligned is refused",
                         rc == FT_TT_EINVAL, "rc=%d (%s)", rc, ft_tt_error_string(rc));

    layout = good_layout();
    layout.operand_bytes = 0u;
    rc = ft_tt_group_shared_operand_bind(members, 2u, coord(2u, 7u), &layout, &operand);
    ft_tt_test_check_fmt("a ZERO-BYTE operand is refused", rc == FT_TT_EINVAL, "rc=%d (%s)",
                         rc, ft_tt_error_string(rc));
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    check_rectangle_from_a_row();
    check_sender_inside_costs_one_destination();
    check_a_wrapped_run_is_refused();
    check_a_duplicate_member_is_refused();
    check_a_lone_sender_is_refused();
    check_an_out_of_range_coordinate_is_refused();
    check_inverted_corners_are_refused();
    check_the_encoding_and_the_noc1_swap();
    check_payload_rules();
    check_shared_operand_binding();
    check_shared_operand_layout_refusals();
    return ft_tt_test_finish();
}
