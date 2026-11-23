/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_resident_multicast_args.c — the RESIDENT cooperative matmul's host half, OFF
 * HARDWARE: the plan, the packed rectangle word the kernel unpacks, the per-member runtime
 * args, and the order the three rendezvous words are staged in.
 *
 * Everything here is arithmetic and three L1 writes, and all of it is protecting against
 * failures that are silent on silicon:
 *
 *  - a packed rectangle whose fields land on the wrong bits aims a multicast at a
 *    DIFFERENT live endpoint, which on this board is a hung NoC and a walk to the machine;
 *  - a receiver handed the leader's geometry is a receiver one bug away from broadcasting;
 *  - a destination total that disagrees with the receiver count makes the leader's write
 *    barrier either hang or return while a payload is still landing, and neither shows up
 *    as a wrong answer;
 *  - staging the leader's ready semaphore BEFORE a receiver's arrival word leaves a window
 *    in which the leader starts a block into a member that has not cleared its own.
 *
 * The packed word is checked against LITERAL expected values computed by hand from the
 * field widths, not against `ftcoop_pack_rectangle()` run backwards. Two implementations
 * that agree with each other and disagree with the hardware is exactly the failure a round
 * trip cannot see.
 *
 * Build and run:
 *   make -f Makefile.broadcastpartition check
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_op_matmul_resident_multicast.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>

/** Every write the reset made, in order, so the ORDER can be asserted. A test that only
 *  reads the words back afterwards is vacuous about the one property that matters: they
 *  end up identical whatever order they were written in. */
#define RECORDER_CAPACITY 64u

typedef struct write_recorder {
    struct {
        uint16_t x;
        uint16_t y;
        uint64_t address;
        uint32_t value;
    } entry[RECORDER_CAPACITY];
    unsigned count;
    int fail_after; /**< -1 never; otherwise the write index that returns an error */
} write_recorder;

static int recorder_write(void *user, uint16_t x, uint16_t y, uint64_t address, const void *src,
                          size_t bytes)
{
    write_recorder *recorder = (write_recorder *)user;

    if (recorder->fail_after >= 0 && (int)recorder->count >= recorder->fail_after)
        return FT_TT_ERROR;
    if (recorder->count >= RECORDER_CAPACITY || bytes != sizeof(uint32_t))
        return FT_TT_ERROR;
    recorder->entry[recorder->count].x = x;
    recorder->entry[recorder->count].y = y;
    recorder->entry[recorder->count].address = address;
    memcpy(&recorder->entry[recorder->count].value, src, sizeof(uint32_t));
    recorder->count++;
    return FT_TT_OK;
}

static ft_tt_noc_coord coord(uint32_t x, uint32_t y)
{
    ft_tt_noc_coord result;

    result.x = x;
    result.y = y;
    return result;
}

static void fill_block(ft_tt_noc_coord *out, uint32_t x, uint32_t y, uint32_t rows,
                       uint32_t columns)
{
    uint32_t row;
    uint32_t column;

    for (row = 0u; row < rows; row++)
        for (column = 0u; column < columns; column++)
            out[row * columns + column] = coord(x + column, y + row);
}

/* ------------------------------------------------------------------------- */

static void check_the_packed_rectangle_word(void)
{
    uint32_t packed;

    ft_tt_test_step("the packed rectangle word, against LITERAL expected bits");

    /* (13,7)..(16,7) with three destinations:
     *   x_low  13 -> bits  0..5   = 0x0000000D
     *   y_low   7 -> bits  6..11  = 0x000001C0
     *   x_high 16 -> bits 12..17  = 0x00010000
     *   y_high  7 -> bits 18..23  = 0x001C0000
     *   count   3 -> bits 24..31  = 0x03000000
     *   -----------------------------------------
     *                               0x031D01CD */
    packed = ftcoop_pack_rectangle(13u, 7u, 16u, 7u, 3u);
    ft_tt_test_check_fmt("(13,7)..(16,7) x3 packs to 0x031D01CD", packed == 0x031D01CDu,
                         "0x%08X", packed);
    ft_tt_test_check_fmt("x_low unpacks", ftcoop_rectangle_x_low(packed) == 13u, "%u",
                         ftcoop_rectangle_x_low(packed));
    ft_tt_test_check_fmt("y_low unpacks", ftcoop_rectangle_y_low(packed) == 7u, "%u",
                         ftcoop_rectangle_y_low(packed));
    ft_tt_test_check_fmt("x_high unpacks", ftcoop_rectangle_x_high(packed) == 16u, "%u",
                         ftcoop_rectangle_x_high(packed));
    ft_tt_test_check_fmt("y_high unpacks", ftcoop_rectangle_y_high(packed) == 7u, "%u",
                         ftcoop_rectangle_y_high(packed));
    ft_tt_test_check_fmt("the destination count unpacks",
                         ftcoop_rectangle_destinations(packed) == 3u, "%u",
                         ftcoop_rectangle_destinations(packed));

    /* An ASYMMETRIC rectangle, because a 1x1 or a square would make a swapped x/y pair
     * invisible — the "an assert on a shape whose two dimensions happen to be equal is
     * vacuous" trap. (3,8)..(6,9): 3 | 8<<6 | 6<<12 | 9<<18 | 4<<24 = 0x04246203. */
    packed = ftcoop_pack_rectangle(3u, 8u, 6u, 9u, 4u);
    ft_tt_test_check_fmt("(3,8)..(6,9) x4 packs to 0x04246203", packed == 0x04246203u,
                         "0x%08X", packed);

    ft_tt_test_check("an inverted rectangle packs to 0",
                     ftcoop_pack_rectangle(6u, 7u, 3u, 7u, 3u) == 0u,
                     "0 is never a legal piece, so the kernel refuses it");
    ft_tt_test_check("a coordinate above the 6-bit node-id field packs to 0",
                     ftcoop_pack_rectangle(3u, 7u, 64u, 7u, 3u) == 0u,
                     "it would TRUNCATE onto a different live endpoint");
    ft_tt_test_check("five destinations pack to 0",
                     ftcoop_pack_rectangle(3u, 7u, 7u, 7u, 5u) == 0u,
                     "the standing hazard limit, restated in the wire format");
    ft_tt_test_check("zero destinations pack to 0",
                     ftcoop_pack_rectangle(3u, 7u, 3u, 7u, 0u) == 0u, "");

    ft_tt_test_check("the counter never takes the INVALID value",
                     ftcoop_next_counter(0xFFFFFFFFu) == FT_TT_RESIDENT_COOP_FIRST_COUNTER &&
                         FT_TT_RESIDENT_COOP_FIRST_COUNTER != 0u,
                     "a receiver clears its arrival word to 0 before announcing");
}

static void check_the_plan_and_the_args(void)
{
    ft_tt_noc_coord members[8];
    ft_tt_op_matmul_resident_multicast plan;
    uint32_t args[kArgCoopCount];
    uint32_t leader_group;
    int rc;

    ft_tt_test_step("a 2x4 group at (3,7) — the plan and both kinds of arg block");

    fill_block(members, 3u, 7u, 2u, 4u);
    memset(&plan, 0, sizeof(plan));
    rc = ft_tt_op_matmul_resident_multicast_plan(members, 8u, 0u, 4096u, &plan);
    ft_tt_test_check_fmt("the plan builds", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("seven receivers", plan.receiver_count == 7u, "%u",
                         plan.receiver_count);
    ft_tt_test_check_fmt("TWO broadcast packets", plan.partition.rectangle_count == 2u, "%u",
                         plan.partition.rectangle_count);
    ft_tt_test_check("the rendezvous words are the cooperative control block's",
                     plan.ready_semaphore_address != plan.arrival_semaphore_address &&
                         plan.arrival_semaphore_address != plan.counter_word_address &&
                         plan.ready_semaphore_address != 0u,
                     "three distinct addresses, the same on every member");

    memset(args, 0xAA, sizeof(args));
    rc = ft_tt_op_matmul_resident_multicast_args(&plan, 0u, args);
    ft_tt_test_check_fmt("the leader's args build", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("the leader's role is LEADER",
                         args[kArgCoopRole] == FT_TT_RESIDENT_COOP_ROLE_LEADER, "%u",
                         args[kArgCoopRole]);
    ft_tt_test_check_fmt("the leader carries two pieces", args[kArgCoopGroupCount] == 2u, "%u",
                         args[kArgCoopGroupCount]);
    leader_group = args[kArgCoopGroup0];
    ft_tt_test_check_fmt("piece 0 is the leader's own row minus its own cell",
                         ftcoop_rectangle_x_low(leader_group) == 4u &&
                             ftcoop_rectangle_x_high(leader_group) == 6u &&
                             ftcoop_rectangle_y_low(leader_group) == 7u &&
                             ftcoop_rectangle_destinations(leader_group) == 3u,
                         "0x%08X", leader_group);
    ft_tt_test_check_fmt("the unused piece words are 0", args[kArgCoopGroup2] == 0u &&
                                                             args[kArgCoopGroup3] == 0u,
                         "0x%08X 0x%08X", args[kArgCoopGroup2], args[kArgCoopGroup3]);
    ft_tt_test_check_fmt("the receiver count is carried", args[kArgCoopReceiverCount] == 7u,
                         "%u", args[kArgCoopReceiverCount]);
    ft_tt_test_check_fmt("the spin limit is carried and is not 0",
                         args[kArgCoopSpinLimit] == 4096u, "%u", args[kArgCoopSpinLimit]);

    memset(args, 0xAA, sizeof(args));
    rc = ft_tt_op_matmul_resident_multicast_args(&plan, 5u, args);
    ft_tt_test_check_fmt("a receiver's args build", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("its role is RECEIVER",
                         args[kArgCoopRole] == FT_TT_RESIDENT_COOP_ROLE_RECEIVER, "%u",
                         args[kArgCoopRole]);
    /* A RECEIVER GETS NO GEOMETRY AT ALL, and that is the check. A receiver holding the
     * leader's rectangle is one wrong branch away from broadcasting into the group. */
    ft_tt_test_check_fmt("it is given NO rectangle",
                         args[kArgCoopGroupCount] == 0u && args[kArgCoopGroup0] == 0u &&
                             args[kArgCoopGroup1] == 0u,
                         "count=%u", args[kArgCoopGroupCount]);
    ft_tt_test_check_fmt("it is told where the leader is",
                         args[kArgCoopLeaderX] == 3u && args[kArgCoopLeaderY] == 7u, "(%u,%u)",
                         args[kArgCoopLeaderX], args[kArgCoopLeaderY]);
    ft_tt_test_check("every member is given the same three rendezvous addresses",
                     args[kArgCoopReadySemaphore] == plan.ready_semaphore_address &&
                         args[kArgCoopArrivalSemaphore] == plan.arrival_semaphore_address &&
                         args[kArgCoopCounterWord] == plan.counter_word_address,
                     "a multicast writes ONE address into every destination");
}

static void check_the_reset_order(void)
{
    ft_tt_noc_coord members[4];
    ft_tt_op_matmul_resident_multicast plan;
    ft_tt_noc_access noc;
    write_recorder recorder;
    bool arrivals_first = true;
    unsigned index;
    int rc;

    ft_tt_test_step("the reset stages the arrival words BEFORE the leader's ready semaphore");

    fill_block(members, 13u, 7u, 1u, 4u);
    memset(&plan, 0, sizeof(plan));
    if (ft_tt_op_matmul_resident_multicast_plan(members, 4u, 0u, 4096u, &plan) != FT_TT_OK) {
        ft_tt_test_check("the plan builds", false, "");
        return;
    }
    memset(&recorder, 0, sizeof(recorder));
    recorder.fail_after = -1;
    noc.write = recorder_write;
    noc.read = NULL;
    noc.user = &recorder;

    rc = ft_tt_op_matmul_resident_multicast_reset(&noc, &plan);
    ft_tt_test_check_fmt("the reset succeeds", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("six writes: four arrivals, the counter word, the ready semaphore",
                         recorder.count == 6u, "%u", recorder.count);
    if (recorder.count != 6u)
        return;
    for (index = 0u; index < 4u; index++)
        arrivals_first = arrivals_first &&
                         recorder.entry[index].address == plan.arrival_semaphore_address &&
                         recorder.entry[index].x == members[index].x &&
                         recorder.entry[index].value == 0u;
    ft_tt_test_check("writes 0..3 clear every member's arrival word", arrivals_first,
                     "including the leader's, which receives nothing but must not hold a "
                     "previous residency's counter");
    ft_tt_test_check_fmt("write 4 is the leader's counter word",
                         recorder.entry[4].address == plan.counter_word_address &&
                             recorder.entry[4].x == 13u && recorder.entry[4].value == 0u,
                         "addr=0x%X x=%u", (unsigned)recorder.entry[4].address,
                         recorder.entry[4].x);
    ft_tt_test_check_fmt("write 5 — LAST — is the leader's ready semaphore",
                         recorder.entry[5].address == plan.ready_semaphore_address &&
                             recorder.entry[5].value == 0u,
                         "addr=0x%X", (unsigned)recorder.entry[5].address);

    /* A failing write must be REPORTED, not swallowed: a reset that half-ran leaves the
     * group with one stale arrival word, which is the value that makes a desynchronised
     * first block undetectable. */
    memset(&recorder, 0, sizeof(recorder));
    recorder.fail_after = 2;
    rc = ft_tt_op_matmul_resident_multicast_reset(&noc, &plan);
    ft_tt_test_check_fmt("a failing write is reported", rc != FT_TT_OK, "rc=%d", rc);
}

static void check_the_refusals(void)
{
    ft_tt_noc_coord members[8];
    ft_tt_op_matmul_resident_multicast plan;
    uint32_t args[kArgCoopCount];
    int rc;

    ft_tt_test_step("the refusals");

    fill_block(members, 3u, 7u, 2u, 4u);

    rc = ft_tt_op_matmul_resident_multicast_plan(NULL, 8u, 0u, 4096u, &plan);
    ft_tt_test_check_fmt("NULL members refused", rc == FT_TT_EINVAL, "rc=%d", rc);

    rc = ft_tt_op_matmul_resident_multicast_plan(members, 1u, 0u, 4096u, &plan);
    ft_tt_test_check_fmt("one member refused", rc == FT_TT_EINVAL, "rc=%d", rc);

    rc = ft_tt_op_matmul_resident_multicast_plan(members, 8u, 8u, 4096u, &plan);
    ft_tt_test_check_fmt("a leader index outside the group refused", rc == FT_TT_EINVAL,
                         "rc=%d", rc);

    /* AN UNBOUNDED RENDEZVOUS WAIT is the one thing a resident kernel may not be
     * configured into: `assert_reset` is then the only way to end it, and asserting reset
     * on a blocked triad is how cores were wedged. */
    rc = ft_tt_op_matmul_resident_multicast_plan(members, 8u, 0u, 0u, &plan);
    ft_tt_test_check_fmt("a zero spin limit refused", rc == FT_TT_EINVAL, "rc=%d", rc);

    /* Consecutive live indices wrapping a row is how a real caller produces a set that is
     * not a rectangle, and it is exactly the shape the default member cores have. */
    fill_block(members, 3u, 7u, 1u, 4u);
    members[3] = coord(3u, 8u);
    rc = ft_tt_op_matmul_resident_multicast_plan(members, 4u, 0u, 4096u, &plan);
    ft_tt_test_check_fmt("a member set that is not a rectangle refused", rc != FT_TT_OK,
                         "rc=%d", rc);

    fill_block(members, 3u, 7u, 2u, 4u);
    if (ft_tt_op_matmul_resident_multicast_plan(members, 8u, 0u, 4096u, &plan) == FT_TT_OK) {
        rc = ft_tt_op_matmul_resident_multicast_args(&plan, 8u, args);
        ft_tt_test_check_fmt("args for a member outside the plan refused", rc == FT_TT_EINVAL,
                             "rc=%d", rc);
    }
    rc = ft_tt_op_matmul_resident_multicast_reset(NULL, &plan);
    ft_tt_test_check_fmt("a reset with no accessor refused", rc == FT_TT_EINVAL, "rc=%d", rc);
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    printf("ft_tt resident cooperative matmul — host half, off hardware\n");
    printf("===========================================================\n");

    check_the_packed_rectangle_word();
    check_the_plan_and_the_args();
    check_the_reset_order();
    check_the_refusals();

    return ft_tt_test_finish();
}
