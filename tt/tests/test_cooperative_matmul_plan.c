/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_cooperative_matmul_plan.c — the cooperative matmul's WORK SPLIT and its refusals,
 * off hardware.
 *
 * Everything ft_tt_op_matmul_cooperative.c does is arithmetic, and that is deliberate: a wrong
 * multicast rectangle is not a failed transaction, it is a write into a core somebody else is
 * using or a coordinate that hangs the chip, and the iteration loop after a NoC hang is
 * minutes plus physical access to the machine. So every claim the planner
 * makes is provable here, on the Mac, while somebody else owns the card.
 *
 * What this file does NOT cover, and cannot: whether the multicast actually lands, whether the
 * CB write pointer really is where the kernel computes it will be, and whether the answer is
 * right. Those need silicon and are tests/test_cooperative_matmul.c's.
 *
 *   make -f Makefile.cooperative check
 */

#include <stdio.h>
#include <string.h>

#include "../ft_tt_dram_shard.h"
#include "../ft_tt_op_matmul_cooperative.h"
#include "ft_tt_test_report.h"

#define TILE_BYTES FT_TT_OP_TILE_BYTES_BF16

/** A 2x2 block of TRANSLATED coordinates that really is a rectangle: NoC x 3,4 and y 2,3.
 *  Written out rather than derived, so this file has no grid and cannot inherit a grid bug. */
static const ft_tt_noc_coord g_square[4] = {{3u, 2u}, {4u, 2u}, {3u, 3u}, {4u, 3u}};

/** The same four cores with one moved across the non-Tensix column gap. Its bounding box then
 *  contains NoC x 8 and 9 — the ARC/L2CPU spine and the east DRAM column — and a multicast
 *  writes EVERY node in its bounding box. */
static const ft_tt_noc_coord g_across_the_gap[4] = {{3u, 2u}, {10u, 2u}, {3u, 3u}, {10u, 3u}};

static void fill_request(ft_tt_op_matmul_cooperative_request *request, uint32_t tile_cols,
                         ft_tt_op_matmul_cooperative_mode mode)
{
    memset(request, 0, sizeof(*request));
    request->tile_rows = 1u;   /* 32 tokens, which is every decode shape on record */
    request->tile_inner = 48u; /* Kt of ffn_gate/up */
    request->tile_cols = tile_cols;
    request->prefetch_tiles = 48u;
    request->weight_n_major = 1u;
    request->mode = mode;
    request->spin_limit = 1000u;
    request->activation.x = 17u;
    request->activation.y = 12u;
    request->activation.address = 0x00100000u;
    request->activation.tile_bytes = TILE_BYTES;
    request->weight = request->activation;
    request->output = request->activation;
    request->output.address = 0x00200000u;
}

/* ------------------------------------------------------------------------- */

/** The split itself: who owns which columns, and where their output goes. */
static void test_the_split(void)
{
    ft_tt_op_matmul_cooperative_request request;
    ft_tt_op_matmul_cooperative_plan plan;
    bool columns_partition = true;
    bool outputs_are_consecutive = true;
    uint32_t index;
    int rc;

    ft_tt_test_step("the split — 4 cores over Nt = 144");

    fill_request(&request, 144u, FT_TT_OP_MATMUL_COOPERATIVE);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &plan);
    ft_tt_test_check_fmt("a 2x2 group over Nt = 144 plans", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    for (index = 0u; index < 4u; index++) {
        if (plan.member[index].column_count != 36u ||
            plan.member[index].column_first != index * 36u)
            columns_partition = false;
        if (plan.member[index].output_address !=
            request.output.address + index * 36u * TILE_BYTES)
            outputs_are_consecutive = false;
    }
    ft_tt_test_check_fmt("the output columns are partitioned, not overlapped or dropped",
                         columns_partition,
                         "member first/count: %u/%u %u/%u %u/%u %u/%u (wanted 0/36 36/36 "
                         "72/36 108/36)",
                         plan.member[0].column_first, plan.member[0].column_count,
                         plan.member[1].column_first, plan.member[1].column_count,
                         plan.member[2].column_first, plan.member[2].column_count,
                         plan.member[3].column_first, plan.member[3].column_count);
    /* At Mt = 1 the panels concatenate into exactly the single-core output, which is the whole
     * reason the on-card gate can be a byte comparison rather than a PCC. */
    ft_tt_test_check_fmt("at Mt = 1 the panels concatenate into the single-core layout",
                         outputs_are_consecutive,
                         "member 1 writes at 0x%08x, expected 0x%08x",
                         plan.member[1].output_address,
                         request.output.address + 36u * TILE_BYTES);

    ft_tt_test_check_fmt("the leader is inside the rectangle, so it receives nothing",
                         plan.shared_valid && plan.shared.rectangle.sender_inside &&
                             plan.shared.rectangle.destination_count == 3u,
                         "sender_inside %d, destination_count %u (wanted 3 = members - 1), "
                         "node_count %u",
                         (int)plan.shared.rectangle.sender_inside,
                         plan.shared.rectangle.destination_count,
                         plan.shared.rectangle.node_count);
    ft_tt_test_check_fmt("the shared operand is one whole prefetch block of A",
                         plan.shared.layout.operand_bytes == 48u * TILE_BYTES,
                         "%u bytes (wanted %u = depth 48 x %u)",
                         plan.shared.layout.operand_bytes, 48u * TILE_BYTES, TILE_BYTES);
    ft_tt_test_check_fmt("the shared operand lands in CB_IN0, which is what the receivers "
                         "reserve", plan.shared.layout.operand_address == plan.cb[0].address,
                         "operand at 0x%08x, CB_IN0 at 0x%08x",
                         plan.shared.layout.operand_address, plan.cb[0].address);
}

/**
 * THE RAGGED SPLIT — an Nt the member count does not divide, in SPLIT mode.
 *
 * COOPERATIVE refuses this and must keep refusing it: its per-block multicast barrier is
 * correct only because every member walks the same number of blocks. SPLIT has no barrier
 * at all, so refusing there cost a whole tensor its split — MEASURED on MiniCPM5-1B, where
 * N=5 left 3,864 of 3,887 projections on one core and N=7 left every one of them.
 *
 * The property that has to hold is a PARTITION: the column ranges are consecutive, cover
 * [0, Nt) exactly, differ in width by at most one, and every output address is the running
 * sum. A check that only added the counts up would pass a plan whose members overlap.
 */
static void test_the_ragged_split(void)
{
    ft_tt_op_matmul_cooperative_request request;
    ft_tt_op_matmul_cooperative_plan plan;
    bool partitions = true;
    bool balanced = true;
    bool outputs_follow = true;
    uint32_t expected_first = 0u;
    uint32_t total = 0u;
    uint32_t index;
    int rc;

    ft_tt_test_step("the ragged split — 7 cores over Nt = 45, which 7 does not divide");

    fill_request(&request, 45u, FT_TT_OP_MATMUL_SPLIT);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 7u, 0u, &request, &plan);
    ft_tt_test_check_fmt("SPLIT plans a ragged Nt rather than refusing it", rc == FT_TT_OK,
                         "Nt 45 over 7 members gave rc=%d (%s)", rc, ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    for (index = 0u; index < 7u; index++) {
        const ft_tt_op_matmul_cooperative_member *member = &plan.member[index];

        if (member->column_first != expected_first)
            partitions = false;
        if (member->column_count != 6u && member->column_count != 7u)
            balanced = false;
        if (member->output_address !=
            request.output.address + expected_first * request.tile_rows * TILE_BYTES)
            outputs_follow = false;
        expected_first += member->column_count;
        total += member->column_count;
    }
    ft_tt_test_check_fmt("the ranges are consecutive, so nothing is computed twice or dropped",
                         partitions,
                         "member first/count: %u/%u %u/%u %u/%u %u/%u %u/%u %u/%u %u/%u",
                         plan.member[0].column_first, plan.member[0].column_count,
                         plan.member[1].column_first, plan.member[1].column_count,
                         plan.member[2].column_first, plan.member[2].column_count,
                         plan.member[3].column_first, plan.member[3].column_count,
                         plan.member[4].column_first, plan.member[4].column_count,
                         plan.member[5].column_first, plan.member[5].column_count,
                         plan.member[6].column_first, plan.member[6].column_count);
    ft_tt_test_check_fmt("and they COVER Nt exactly", total == 45u, "%u columns of 45", total);
    ft_tt_test_check_fmt("the widths differ by at most one, which is the balance a column "
                         "split can reach", balanced, "3 members of 7 should hold 7 columns");
    ft_tt_test_check_fmt("the output addresses are the RUNNING SUM, not index x width",
                         outputs_follow, "member 4 writes at 0x%08x, expected 0x%08x",
                         plan.member[4].output_address,
                         request.output.address +
                             (plan.member[0].column_count + plan.member[1].column_count +
                              plan.member[2].column_count + plan.member[3].column_count) *
                                 request.tile_rows * TILE_BYTES);

    /* And the COOPERATIVE refusal is unchanged, because its reason is the RENDEZVOUS and not
     * the arithmetic. Without this the change reads as "ragged splits are fine now". */
    fill_request(&request, 45u, FT_TT_OP_MATMUL_COOPERATIVE);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 7u, 0u, &request, &plan);
    ft_tt_test_check_fmt("COOPERATIVE still REFUSES the same shape", rc == FT_TT_EINVAL,
                         "rc=%d (%s)", rc, ft_tt_error_string(rc));

    /* Fewer columns than members would give one of them ZERO, which the resident reader
     * refuses on the card — and a refusal there strands a triad. */
    fill_request(&request, 3u, FT_TT_OP_MATMUL_SPLIT);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 7u, 0u, &request, &plan);
    ft_tt_test_check_fmt("an Nt SMALLER than the member count is REFUSED, because a member "
                         "with zero columns is a refusal on the card",
                         rc == FT_TT_EINVAL, "Nt 3 over 7 members gave rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
}

/** Nothing about the CB placement may depend on how many members there are — the multicast
 *  writes ONE L1 address into every node of the rectangle, so a placement that moved with the
 *  group size would be a different address on a differently-sized group. */
static void test_placement_is_group_independent(void)
{
    ft_tt_op_matmul_cooperative_request request;
    ft_tt_op_matmul_cooperative_plan two;
    ft_tt_op_matmul_cooperative_plan four;
    ft_tt_op_matmul_cooperative_plan solo;
    int rc_two;
    int rc_four;
    int rc_solo;

    ft_tt_test_step("the circular-buffer placement does not move with the group size");

    fill_request(&request, 144u, FT_TT_OP_MATMUL_COOPERATIVE);
    rc_two = ft_tt_op_matmul_cooperative_plan_build(g_square, 2u, 0u, &request, &two);
    rc_four = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &four);
    request.mode = FT_TT_OP_MATMUL_INDEPENDENT;
    rc_solo = ft_tt_op_matmul_cooperative_plan_build(NULL, 1u, 0u, &request, &solo);

    ft_tt_test_check_fmt("all three plans built", rc_two == FT_TT_OK && rc_four == FT_TT_OK &&
                                                      rc_solo == FT_TT_OK,
                         "rc %d / %d / %d", rc_two, rc_four, rc_solo);
    if (rc_two != FT_TT_OK || rc_four != FT_TT_OK || rc_solo != FT_TT_OK)
        return;

    ft_tt_test_check_fmt("CB_IN0 is at the same L1 address for 1, 2 and 4 members",
                         two.cb[0].address == four.cb[0].address &&
                             two.cb[0].address == solo.cb[0].address,
                         "0x%08x / 0x%08x / 0x%08x", solo.cb[0].address, two.cb[0].address,
                         four.cb[0].address);
    ft_tt_test_check_fmt("CB_IN0 is 64-byte aligned, which is what a DRAM read into it needs",
                         (four.cb[0].address % FT_TT_DRAM_ALIGNMENT) == 0u, "0x%08x",
                         four.cb[0].address);
    ft_tt_test_check_fmt("the input CBs hold exactly two prefetch blocks",
                         four.cb[0].page_count == 2u * four.depth &&
                             four.cb[1].page_count == 2u * four.depth,
                         "%u and %u pages at depth %u", four.cb[0].page_count,
                         four.cb[1].page_count, four.depth);
    ft_tt_test_check_fmt("the control region clears the circular-buffer arena",
                         FT_TT_OP_COOP_CONTROL_BASE + FT_TT_OP_COOP_CONTROL_BYTES <=
                             FT_TT_OP_CB_ARENA_BASE,
                         "control 0x%08x + %u vs arena 0x%08x",
                         (unsigned)FT_TT_OP_COOP_CONTROL_BASE,
                         (unsigned)FT_TT_OP_COOP_CONTROL_BYTES,
                         (unsigned)FT_TT_OP_CB_ARENA_BASE);
}

/** The refusals. Each one is a shape that produces no error on the card. */
static void test_the_refusals(void)
{
    ft_tt_op_matmul_cooperative_request request;
    ft_tt_op_matmul_cooperative_plan plan;
    int rc;

    ft_tt_test_step("the refusals — every one of these is silent on the card");

    /* A ragged split gives members different BLOCK COUNTS, and the per-block rendezvous is
     * only correct because every member walks the same number. It would deadlock on the last
     * blocks: the leader waiting for members that have already finished. */
    fill_request(&request, 145u, FT_TT_OP_MATMUL_COOPERATIVE);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &plan);
    ft_tt_test_check_fmt("an Nt that does not divide by the member count is REFUSED",
                         rc == FT_TT_EINVAL, "Nt 145 over 4 members gave rc=%d (%s)", rc,
                         ft_tt_error_string(rc));

    /* One member leaves the rectangle with zero destinations, which is the zero-length NoC
     * transfer in different clothes — the shape that took this card from working to
     * `Mem- BusMaster-`. */
    fill_request(&request, 144u, FT_TT_OP_MATMUL_COOPERATIVE);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 1u, 0u, &request, &plan);
    ft_tt_test_check_fmt("a COOPERATIVE group of ONE is REFUSED", rc == FT_TT_EINVAL,
                         "rc=%d (%s)", rc, ft_tt_error_string(rc));

    /* A multicast writes EVERY node in its bounding box, and x = 8, 9 are the ARC/L2CPU spine
     * and a DRAM column. This is the single most expensive mistake available here. */
    fill_request(&request, 144u, FT_TT_OP_MATMUL_COOPERATIVE);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_across_the_gap, 4u, 0u, &request, &plan);
    ft_tt_test_check_fmt("a group whose bounding box CROSSES the non-Tensix column gap is "
                         "REFUSED", rc != FT_TT_OK,
                         "x 3 and 10 with x 8, 9 between them gave rc=%d (%s)", rc,
                         ft_tt_error_string(rc));

    fill_request(&request, 144u, FT_TT_OP_MATMUL_COOPERATIVE);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 4u, &request, &plan);
    ft_tt_test_check_fmt("a leader outside the group is REFUSED", rc == FT_TT_EINVAL,
                         "leader 4 of 4 members gave rc=%d (%s)", rc, ft_tt_error_string(rc));

    fill_request(&request, 0u, FT_TT_OP_MATMUL_COOPERATIVE);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &plan);
    ft_tt_test_check_fmt("a zero Nt is REFUSED", rc == FT_TT_EINVAL, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
}

/** The runtime-arg blocks, which are the whole of what the kernel sees. */
static void test_the_argument_blocks(void)
{
    ft_tt_op_matmul_cooperative_request request;
    ft_tt_op_matmul_cooperative_plan plan;
    uint32_t leader_args[FTCM_ARG_COUNT];
    uint32_t receiver_args[FTCM_ARG_COUNT];
    uint32_t compute_args[3];
    uint32_t writer_args[4];
    int rc;

    ft_tt_test_step("the runtime-arg blocks");

    fill_request(&request, 144u, FT_TT_OP_MATMUL_COOPERATIVE);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &plan);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("the plan for the argument blocks built", false,
                         ft_tt_error_string(rc));
        return;
    }
    rc = ft_tt_op_matmul_cooperative_reader_arguments(&plan, &request, g_square, 0u,
                                                      leader_args, FTCM_ARG_COUNT);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_matmul_cooperative_reader_arguments(&plan, &request, g_square, 2u,
                                                          receiver_args, FTCM_ARG_COUNT);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_matmul_cooperative_compute_arguments(&plan, &request, 2u, compute_args,
                                                           3u);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_matmul_cooperative_writer_arguments(&plan, &request, 2u, writer_args, 4u);
    ft_tt_test_check_fmt("every argument block built", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    ft_tt_test_check_fmt("exactly one member is the LEADER and the rest RECEIVE",
                         leader_args[FTCM_ARG_ROLE] == FTCM_ROLE_LEADER &&
                             receiver_args[FTCM_ARG_ROLE] == FTCM_ROLE_RECEIVER,
                         "roles %u and %u (wanted %u and %u)", leader_args[FTCM_ARG_ROLE],
                         receiver_args[FTCM_ARG_ROLE], FTCM_ROLE_LEADER, FTCM_ROLE_RECEIVER);
    ft_tt_test_check_fmt("every member is told the SAME leader, rectangle and semaphores",
                         leader_args[FTCM_ARG_LEADER_X] == receiver_args[FTCM_ARG_LEADER_X] &&
                             leader_args[FTCM_ARG_LEADER_Y] ==
                                 receiver_args[FTCM_ARG_LEADER_Y] &&
                             leader_args[FTCM_ARG_RECTANGLE_X_LOW] ==
                                 receiver_args[FTCM_ARG_RECTANGLE_X_LOW] &&
                             leader_args[FTCM_ARG_ARRIVAL_SEMAPHORE] ==
                                 receiver_args[FTCM_ARG_ARRIVAL_SEMAPHORE] &&
                             leader_args[FTCM_ARG_READY_SEMAPHORE] ==
                                 receiver_args[FTCM_ARG_READY_SEMAPHORE],
                         "leader (%u,%u) vs (%u,%u), arrival 0x%08x vs 0x%08x",
                         leader_args[FTCM_ARG_LEADER_X], leader_args[FTCM_ARG_LEADER_Y],
                         receiver_args[FTCM_ARG_LEADER_X], receiver_args[FTCM_ARG_LEADER_Y],
                         leader_args[FTCM_ARG_ARRIVAL_SEMAPHORE],
                         receiver_args[FTCM_ARG_ARRIVAL_SEMAPHORE]);
    ft_tt_test_check_fmt("every member is told the SAME CB_IN0 base, which is the address the "
                         "multicast lands at",
                         leader_args[FTCM_ARG_CB_IN0_BASE] ==
                                 receiver_args[FTCM_ARG_CB_IN0_BASE] &&
                             leader_args[FTCM_ARG_CB_IN0_BASE] == plan.cb[0].address,
                         "0x%08x vs 0x%08x, CB_IN0 at 0x%08x",
                         leader_args[FTCM_ARG_CB_IN0_BASE],
                         receiver_args[FTCM_ARG_CB_IN0_BASE], plan.cb[0].address);
    ft_tt_test_check_fmt("the destination count EXCLUDES the leader",
                         leader_args[FTCM_ARG_DESTINATION_COUNT] == 3u &&
                             receiver_args[FTCM_ARG_DESTINATION_COUNT] == 3u,
                         "%u and %u for a group of 4 with the leader inside (wanted 3): too "
                         "many and the barrier never returns, too few and it returns while "
                         "writes are still landing",
                         leader_args[FTCM_ARG_DESTINATION_COUNT],
                         receiver_args[FTCM_ARG_DESTINATION_COUNT]);
    ft_tt_test_check_fmt("a member is told its own column slice and the FULL width",
                         receiver_args[FTCM_ARG_B_COLUMN_FIRST] == 72u &&
                             receiver_args[FTCM_ARG_TILE_COLS] == 36u &&
                             receiver_args[FTCM_ARG_B_FULL_COLS] == 144u,
                         "first %u, count %u, full %u (wanted 72, 36, 144)",
                         receiver_args[FTCM_ARG_B_COLUMN_FIRST],
                         receiver_args[FTCM_ARG_TILE_COLS],
                         receiver_args[FTCM_ARG_B_FULL_COLS]);
    ft_tt_test_check_fmt("the compute third is told its own width, and nothing else changes",
                         compute_args[0] == 1u && compute_args[1] == 48u &&
                             compute_args[2] == 36u,
                         "Mt %u, Kt %u, Nt %u (wanted 1, 48, 36)", compute_args[0],
                         compute_args[1], compute_args[2]);
    ft_tt_test_check_fmt("the writer third streams this member's tiles to its own panel",
                         writer_args[0] == 36u &&
                             writer_args[3] == request.output.address + 72u * TILE_BYTES,
                         "%u tiles to 0x%08x (wanted 36 to 0x%08x)", writer_args[0],
                         writer_args[3],
                         (unsigned)(request.output.address + 72u * TILE_BYTES));

    /* A SOLO member gets NO rectangle. A zeroed one is (0,0)..(0,0), a real live endpoint of
     * the wrong type, and leaving destination_count at zero is what makes the kernel refuse to
     * issue rather than issue somewhere plausible-looking. */
    request.mode = FT_TT_OP_MATMUL_SPLIT;
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &plan);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_matmul_cooperative_reader_arguments(&plan, &request, g_square, 0u,
                                                          leader_args, FTCM_ARG_COUNT);
    ft_tt_test_check_fmt("a SPLIT member is given NO rectangle and zero destinations",
                         rc == FT_TT_OK && leader_args[FTCM_ARG_ROLE] == FTCM_ROLE_SOLO &&
                             leader_args[FTCM_ARG_DESTINATION_COUNT] == 0u &&
                             leader_args[FTCM_ARG_RECTANGLE_X_HIGH] == 0u,
                         "rc=%d, role %u, destinations %u, x_high %u", rc,
                         leader_args[FTCM_ARG_ROLE], leader_args[FTCM_ARG_DESTINATION_COUNT],
                         leader_args[FTCM_ARG_RECTANGLE_X_HIGH]);

    /* INDEPENDENT is not a split: every member does the whole width into its own result. */
    request.mode = FT_TT_OP_MATMUL_INDEPENDENT;
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &plan);
    ft_tt_test_check_fmt("INDEPENDENT gives every member the WHOLE width and its own output",
                         rc == FT_TT_OK && plan.member[3].column_first == 0u &&
                             plan.member[3].column_count == 144u &&
                             plan.member[3].output_address ==
                                 request.output.address + 3u * 144u * TILE_BYTES,
                         "rc=%d, member 3 first %u count %u at 0x%08x", rc,
                         plan.member[3].column_first, plan.member[3].column_count,
                         plan.member[3].output_address);
}

/* ------------------------------------------------------------------------- */

/** Doors of one bank, spelled out rather than derived, for the same reason g_square is: this
 *  file has no grid and must not be able to inherit a grid bug. `coord` is a stand-in
 *  translated alias; `noc_row` is the NoC 0 y the row-sharing rule is expressed in. */
static void fill_port_set(ft_tt_dram_port_set *set, uint32_t bank, const uint32_t *rows,
                          uint32_t port_count)
{
    uint32_t port;

    memset(set, 0, sizeof(*set));
    set->logical_bank = bank;
    set->port_count = port_count;
    for (port = 0u; port < port_count; port++) {
        set->port[port].logical_bank = bank;
        set->port[port].port = port;
        set->port[port].coord.x = 17u;
        set->port[port].coord.y = bank * 3u + port;
        set->port[port].noc_row = rows[port];
    }
}

/**
 * The port rotation: which DOOR each member reads through, that NO other reader arg moves,
 * and that a bad bank pairing is COUNTED rather than silently accepted.
 *
 * The last is the one worth the most: two DRAM endpoints on one NoC row share ~60 GB/s where
 * two in one column do not interact at all (plan Phase 6.7), and the pairing that produces it
 * — the same port index of a bank and of bank+4 — looks completely innocent.
 */
static void test_the_port_rotation(void)
{
    /* Blackhole's real rows, ft_tt_arch_blackhole.c:93. Bank 5 (east) and bank 1 (west) carry
     * the SAME three y, which is the collision this must catch; bank 6 carries three others. */
    static const uint32_t bank5_rows[3] = {2u, 10u, 3u};
    static const uint32_t bank6_rows[3] = {9u, 4u, 8u};
    static const uint32_t bank1_rows[3] = {2u, 10u, 3u};
    ft_tt_op_matmul_cooperative_request request;
    ft_tt_op_matmul_cooperative_plan striped;
    ft_tt_op_matmul_cooperative_plan flat;
    uint32_t striped_args[FTCM_ARG_COUNT];
    uint32_t flat_args[FTCM_ARG_COUNT];
    uint32_t member;
    uint32_t endpoint_args_moved = 0u;
    uint32_t other_args_moved = 0u;
    uint32_t doors_not_in_the_args = 0u;
    int rc;

    ft_tt_test_step("the DRAM port rotation — the doors, and the row collision it must count");

    fill_request(&request, 144u, FT_TT_OP_MATMUL_SPLIT);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &flat);
    ft_tt_test_check_fmt("a request with NO port sets plans exactly as it always did",
                         rc == FT_TT_OK && flat.distinct_endpoints == 0u &&
                             flat.row_collisions == 0u &&
                             flat.member[3].weight_x == request.weight.x &&
                             flat.member[3].weight_y == request.weight.y,
                         "rc=%d, %u endpoints, %u collisions, member 3 reads B at (%u,%u) "
                         "against the request's (%u,%u)",
                         rc, flat.distinct_endpoints, flat.row_collisions,
                         flat.member[3].weight_x, flat.member[3].weight_y, request.weight.x,
                         request.weight.y);

    fill_port_set(&request.activation_ports, 5u, bank5_rows, 3u);
    fill_port_set(&request.weight_ports, 6u, bank6_rows, 3u);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &striped);
    ft_tt_test_check_fmt("four members over three doors take ports 0,1,2,0 — the balanced "
                         "assignment, since the outstanding-read budget is PER ENDPOINT",
                         rc == FT_TT_OK && striped.member[0].weight_y == 18u &&
                             striped.member[1].weight_y == 19u &&
                             striped.member[2].weight_y == 20u &&
                             striped.member[3].weight_y == 18u,
                         "rc=%d, B doors y %u %u %u %u (wanted 18 19 20 18)", rc,
                         striped.member[0].weight_y, striped.member[1].weight_y,
                         striped.member[2].weight_y, striped.member[3].weight_y);
    ft_tt_test_check_fmt("three doors of each of two banks is six distinct endpoints, on six "
                         "distinct rows",
                         rc == FT_TT_OK && striped.distinct_endpoints == 6u &&
                             striped.row_collisions == 0u,
                         "%u endpoints, %u row collisions", striped.distinct_endpoints,
                         striped.row_collisions);

    /* The whole claim of the rotation is that ONLY the endpoint moves. Every other reader arg
     * has to be bit-identical to the un-striped plan's, or the byte comparison on the card
     * would be scoring two different programs rather than two addressings of one. */
    for (member = 0u; member < 4u && rc == FT_TT_OK; member++) {
        uint32_t index;

        rc = ft_tt_op_matmul_cooperative_reader_arguments(&striped, &request, g_square, member,
                                                          striped_args, FTCM_ARG_COUNT);
        if (rc == FT_TT_OK)
            rc = ft_tt_op_matmul_cooperative_reader_arguments(&flat, &request, g_square, member,
                                                              flat_args, FTCM_ARG_COUNT);
        if (rc != FT_TT_OK)
            break;
        /* Named individually, because "some endpoint word moved" is satisfied by A alone and
         * would pass a build that dropped B's door entirely — the mutation that goes green
         * against the weaker form. */
        if (striped_args[FTCM_ARG_A_X] != striped.member[member].activation_x ||
            striped_args[FTCM_ARG_A_Y] != striped.member[member].activation_y ||
            striped_args[FTCM_ARG_B_X] != striped.member[member].weight_x ||
            striped_args[FTCM_ARG_B_Y] != striped.member[member].weight_y)
            doors_not_in_the_args++;
        for (index = 0u; index < FTCM_ARG_COUNT; index++) {
            if (striped_args[index] == flat_args[index])
                continue;
            if (index == FTCM_ARG_A_X || index == FTCM_ARG_A_Y || index == FTCM_ARG_B_X ||
                index == FTCM_ARG_B_Y)
                endpoint_args_moved++;
            else
                other_args_moved++;
        }
    }
    ft_tt_test_check_fmt("every member's reader block carries THAT member's four door words",
                         rc == FT_TT_OK && doors_not_in_the_args == 0u,
                         "rc=%d, %u of 4 members' arg blocks disagree with the plan's doors",
                         rc, doors_not_in_the_args);
    /* Absolute, not differential: the ADDRESSES are the claim striping makes — one memory,
     * several doors — and a differential check cannot see a change made to both arms. */
    ft_tt_test_check_fmt("the operand ADDRESSES are untouched by striping, on every member",
                         rc == FT_TT_OK &&
                             striped_args[FTCM_ARG_A_BASE] == request.activation.address &&
                             striped_args[FTCM_ARG_B_BASE] == request.weight.address,
                         "A base 0x%08x (wanted 0x%08x), B base 0x%08x (wanted 0x%08x)",
                         striped_args[FTCM_ARG_A_BASE], request.activation.address,
                         striped_args[FTCM_ARG_B_BASE], request.weight.address);
    ft_tt_test_check_fmt("striping moves ENDPOINT args and nothing else — the bases, the "
                         "column slice, the depth and the CB all stay",
                         rc == FT_TT_OK && endpoint_args_moved > 0u && other_args_moved == 0u,
                         "rc=%d, %u endpoint arg words moved over 4 members, %u OTHER arg "
                         "words moved (0 endpoint words would mean the rotation did nothing; "
                         "any other word is the failure this counts)",
                         rc, endpoint_args_moved, other_args_moved);

    /* THE RED HALF, off hardware: bank 1 west and bank 5 east share a NoC y per port index, so
     * each of the three doors collides with its opposite number. */
    fill_port_set(&request.activation_ports, 1u, bank1_rows, 3u);
    fill_port_set(&request.weight_ports, 5u, bank5_rows, 3u);
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &striped);
    ft_tt_test_check_fmt("a bank pairing whose ports share NoC rows is COUNTED, not silently "
                         "accepted",
                         rc == FT_TT_OK && striped.row_collisions == 3u &&
                             striped.distinct_endpoints == 6u,
                         "rc=%d, %u collisions (wanted 3, one per port index), %u endpoints — "
                         "a 0 here would mean the check cannot fail",
                         rc, striped.row_collisions, striped.distinct_endpoints);
}

/**
 * PART B — the per-member SHARDS: the member reads ITS OWN panel, from column 0 of a grid its
 * own width, at its own bank's door.
 *
 * The three arg words this changes are the ones a bank-striped plan lives or dies on, and each
 * of them is in range whatever it holds — a member reading the full grid's column offset out
 * of its own panel would run to completion and answer wrongly. So they are checked by value.
 */
static void test_the_bank_shards(void)
{
    static const uint32_t bank2_rows[3] = {9u, 4u, 8u};
    static const uint32_t bank3_rows[3] = {5u, 7u, 6u};
    ft_tt_op_matmul_cooperative_request request;
    ft_tt_op_matmul_cooperative_plan plan;
    ft_tt_dram_port_set banks[2];
    uint32_t args[FTCM_ARG_COUNT];
    uint32_t member;
    uint32_t wrong_grid = 0u;
    int rc;

    ft_tt_test_step("the per-member bank shards — own panel, own grid, own door");

    fill_request(&request, 144u, FT_TT_OP_MATMUL_SPLIT);
    fill_port_set(&banks[0], 2u, bank2_rows, 3u);
    fill_port_set(&banks[1], 3u, bank3_rows, 3u);
    for (member = 0u; member < 4u; member++) {
        request.weight_shards[member].endpoint = banks[member % 2u].port[member % 3u];
        request.weight_shards[member].address = 0x10000000u + member * 0x1000u;
        request.activation_shards[member].endpoint = banks[(member + 1u) % 2u].port[member % 3u];
        request.activation_shards[member].address = 0x20000000u;
    }

    /* A shard list that covers SOME members is refused, not partly applied: the covered ones
     * would read their own panel from column 0 while the rest read the full grid at the full
     * offset, and both are in range. */
    request.weight_shard_count = 3u;
    request.activation_shard_count = 4u;
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &plan);
    ft_tt_test_check_fmt("a shard list that does not cover every member is REFUSED",
                         rc == FT_TT_EINVAL, "rc=%d (%s), wanted FT_TT_EINVAL", rc,
                         ft_tt_error_string(rc));

    request.weight_shard_count = 4u;
    rc = ft_tt_op_matmul_cooperative_plan_build(g_square, 4u, 0u, &request, &plan);
    ft_tt_test_check_fmt("a full shard list plans", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    for (member = 0u; member < 4u; member++) {
        rc = ft_tt_op_matmul_cooperative_reader_arguments(&plan, &request, g_square, member,
                                                          args, FTCM_ARG_COUNT);
        if (rc != FT_TT_OK)
            break;
        if (args[FTCM_ARG_B_BASE] != request.weight_shards[member].address ||
            args[FTCM_ARG_B_COLUMN_FIRST] != 0u || args[FTCM_ARG_B_FULL_COLS] != 36u ||
            args[FTCM_ARG_A_BASE] != request.activation_shards[member].address ||
            args[FTCM_ARG_B_X] != request.weight_shards[member].endpoint.coord.x ||
            args[FTCM_ARG_B_Y] != request.weight_shards[member].endpoint.coord.y)
            wrong_grid++;
    }
    ft_tt_test_check_fmt("every member reads ITS OWN panel from column 0 of a 36-wide grid at "
                         "its own door",
                         rc == FT_TT_OK && wrong_grid == 0u,
                         "rc=%d, %u of 4 members disagree. Member 3 got base 0x%08x, first "
                         "column %u, full cols %u (wanted 0x%08x, 0, 36)",
                         rc, wrong_grid, args[FTCM_ARG_B_BASE], args[FTCM_ARG_B_COLUMN_FIRST],
                         args[FTCM_ARG_B_FULL_COLS], request.weight_shards[3].address);
    ft_tt_test_check_fmt("the member still knows which columns of the FULL answer it owns",
                         plan.member[3].column_first == 108u && plan.member[3].column_count == 36u,
                         "member 3 owns columns %u..%u of 144 — the OUTPUT panel depends on it, "
                         "and sharding must not fold it away",
                         plan.member[3].column_first,
                         plan.member[3].column_first + plan.member[3].column_count - 1u);
}

/* ------------------------------------------------------------------------- */

/** This card's telemetry masks, written out rather than read from a device, so the shard
 *  placement is provable on the Mac. All 14 Tensix columns, all 8 GDDR banks, translation on
 *  — the grid tests/test_coord.c is an oracle for. */
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

/** One operand pair to place: a PARTITIONED weight and a REPLICATED activation, which is the
 *  shape every bank-striped matmul arm asks for. */
static void fill_shard_operands(ft_tt_dram_shard_operand *operands, uint32_t weight_bytes,
                                uint32_t activation_bytes)
{
    memset(operands, 0, sizeof(operands[0]) * 2u);
    operands[0].spread = FT_TT_DRAM_SHARD_PARTITIONED;
    operands[0].bank_base_address = 0x10000000u;
    operands[0].member_bytes = weight_bytes;
    operands[1].spread = FT_TT_DRAM_SHARD_REPLICATED;
    operands[1].bank_base_address = 0x20000000u;
    operands[1].member_bytes = activation_bytes;
}

/**
 * LOAD-TIME placement — the fork Phase 11 settled, proved without silicon.
 *
 * The claim is that the ARTIFACT STAYS CANONICAL: nothing about it changes with N, and all
 * the group size decides is which contiguous run of its bytes goes where. So the checks are
 * about the MAP, not about any layout:
 *
 *   COVERAGE   at every N the partitioned source ranges tile the artifact exactly once — no
 *              gap, no overlap. A placement that dropped or duplicated a column slice would
 *              still complete on silicon and still read live endpoints.
 *   DISJOINT   no two members get the same (bank, address). This is the bug the byte
 *              comparison against single-core caught at N=12 and N=16 and NOTHING ELSE did:
 *              two members wrote panels to one base and silently overwrote each other, every
 *              kernel completed, every status word was clean and the GB/s did not move. It is
 *              an ARITHMETIC property, so it belongs here, where finding it costs a second
 *              instead of a run on the card.
 *   BALANCE    busiest and quietest bank differ by at most one stream at every N. The
 *              outstanding-read budget is per endpoint and an unbalanced map measured 30%
 *              worse than either neighbouring balanced count (plan Phase 6.7).
 *   SEPARATION no member pulls its two operands through one bank, which serves ~64 GB/s
 *              however many of its doors are used.
 */
static void test_the_shard_placement(void)
{
    static const uint32_t group_sizes[6] = {1u, 2u, 4u, 8u, 12u, 16u};
    const ft_tt_grid_info grid = card_grid();
    /* 48 columns of weight in the canonical artifact: divisible by every group size below, so
     * a member's slice is always a whole number of them. */
    const uint32_t canonical_weight_bytes = 48u * 0x4000u;
    ft_tt_dram_shard_operand operands[2];
    uint32_t entry;
    uint32_t coverage_wrong = 0u;
    uint32_t collisions = 0u;
    uint32_t unbalanced = 0u;
    uint32_t same_bank = 0u;
    uint32_t worst_gap = 0u;
    int rc = FT_TT_OK;

    ft_tt_test_step("the LOAD-TIME shard placement — canonical bytes in, per-member banks out");

    for (entry = 0u; entry < 6u && rc == FT_TT_OK; entry++) {
        const uint32_t members = group_sizes[entry];
        ft_tt_dram_shard_plan plan;
        uint32_t member;

        fill_shard_operands(operands, canonical_weight_bytes / members, 0x1000u);
        rc = ft_tt_dram_shard_plan_build(&grid, operands, 2u, members, FT_TT_COORD_TRANSLATED,
                                         &plan);
        if (rc != FT_TT_OK)
            break;

        if (plan.busiest_bank_streams - plan.quietest_bank_streams > 1u)
            unbalanced++;
        if (plan.busiest_bank_streams - plan.quietest_bank_streams > worst_gap)
            worst_gap = plan.busiest_bank_streams - plan.quietest_bank_streams;

        for (member = 0u; member < members; member++) {
            const ft_tt_dram_shard_placement *weight =
                ft_tt_dram_shard_placement_of(&plan, 0u, member);
            const ft_tt_dram_shard_placement *activation =
                ft_tt_dram_shard_placement_of(&plan, 1u, member);
            uint32_t other;

            if (weight == NULL || activation == NULL) {
                coverage_wrong++;
                continue;
            }
            /* Member c's slice starts exactly where c-1's ended and the last ends at the
             * artifact's end — coverage without gap or overlap, stated once. */
            if (weight->source_offset != member * weight->bytes ||
                weight->bytes * members != canonical_weight_bytes ||
                activation->source_offset != 0u)
                coverage_wrong++;
            if (weight->endpoint.logical_bank == activation->endpoint.logical_bank)
                same_bank++;
            for (other = 0u; other < member; other++) {
                const ft_tt_dram_shard_placement *earlier =
                    ft_tt_dram_shard_placement_of(&plan, 0u, other);

                if (earlier != NULL &&
                    earlier->endpoint.logical_bank == weight->endpoint.logical_bank &&
                    earlier->address == weight->address)
                    collisions++;
            }
        }
    }

    ft_tt_test_check_fmt("every group size plans", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    ft_tt_test_check_fmt("a member's slice is a CONTIGUOUS run of the canonical bytes, and the "
                         "slices tile it exactly once at N = 1,2,4,8,12,16",
                         coverage_wrong == 0u,
                         "%u placements disagree — the artifact is meant to be untouched and "
                         "N-independent, so a wrong offset here is a converter's decision that "
                         "has leaked into the loader",
                         coverage_wrong);
    ft_tt_test_check_fmt("no two members are given the SAME bank and address", collisions == 0u,
                         "%u colliding pairs. This is the failure that shows ONLY as BYTES "
                         "DIFFER at N=12 and N=16 on silicon: both members read a valid panel "
                         "from a live endpoint and complete",
                         collisions);
    ft_tt_test_check_fmt("the map is BALANCED at every group size — busiest and quietest bank "
                         "differ by at most one stream",
                         unbalanced == 0u, "%u group sizes unbalanced, worst gap %u streams",
                         unbalanced, worst_gap);
    ft_tt_test_check_fmt("no member pulls both operands through one bank", same_bank == 0u,
                         "%u members do; a bank serves ~64 GB/s however many doors are used",
                         same_bank);
}

/**
 * The refusals. Each one is a wrong DRAM ADDRESS if it is let through, and a wrong DRAM
 * address is a wrong answer with no error anywhere on the card.
 */
static void test_the_shard_refusals(void)
{
    const ft_tt_grid_info grid = card_grid();
    ft_tt_dram_shard_operand operands[2];
    ft_tt_dram_shard_plan plan;
    int rc;

    ft_tt_test_step("the shard placement's refusals");

    fill_shard_operands(operands, 0x4000u, 0x1000u);
    operands[0].member_bytes = 0u;
    rc = ft_tt_dram_shard_plan_build(&grid, operands, 2u, 4u, FT_TT_COORD_TRANSLATED, &plan);
    ft_tt_test_check_fmt("an operand that places ZERO bytes per member is refused",
                         rc == FT_TT_EINVAL, "rc=%d (%s), wanted FT_TT_EINVAL", rc,
                         ft_tt_error_string(rc));

    fill_shard_operands(operands, 0x4000u, 0x1000u);
    operands[1].bank_base_address = operands[0].bank_base_address;
    rc = ft_tt_dram_shard_plan_build(&grid, operands, 2u, 4u, FT_TT_COORD_TRANSLATED, &plan);
    ft_tt_test_check_fmt("two operands at ONE bank base are refused — their members would "
                         "stack over each other and both would still complete",
                         rc == FT_TT_EINVAL, "rc=%d (%s), wanted FT_TT_EINVAL", rc,
                         ft_tt_error_string(rc));

    fill_shard_operands(operands, 0x4000u, 0x1000u);
    rc = ft_tt_dram_shard_plan_build(&grid, operands, 2u, FT_TT_DRAM_SHARD_MAX_MEMBERS + 1u,
                                     FT_TT_COORD_TRANSLATED, &plan);
    ft_tt_test_check_fmt("more members than the plan can hold is refused, not truncated",
                         rc == FT_TT_EINVAL, "rc=%d (%s), wanted FT_TT_EINVAL", rc,
                         ft_tt_error_string(rc));

    /* 16 members over 8 banks is two 1.875 GiB pieces per bank on top of a 256 MiB base —
     * 4 GiB, past this part's 0xFF000000. The arithmetic is done in 64 bits on purpose: in 32
     * it would WRAP and place the second piece at a low, live address instead of refusing. */
    fill_shard_operands(operands, 0x78000000u, 0x1000u);
    rc = ft_tt_dram_shard_plan_build(&grid, operands, 2u, 16u, FT_TT_COORD_TRANSLATED, &plan);
    ft_tt_test_check_fmt("a placement that would run past the END of a bank is refused",
                         rc == FT_TT_ENOMEM, "rc=%d (%s), wanted FT_TT_ENOMEM", rc,
                         ft_tt_error_string(rc));
}

int main(void)
{
    test_the_split();
    test_the_ragged_split();
    test_placement_is_group_independent();
    test_the_refusals();
    test_the_argument_blocks();
    test_the_port_rotation();
    test_the_bank_shards();
    test_the_shard_placement();
    test_the_shard_refusals();
    return ft_tt_test_finish();
}
