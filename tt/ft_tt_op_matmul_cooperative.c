/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_matmul_cooperative.c
 * @brief Implementation of ft_tt_op_matmul_cooperative.h — see that header for the work split
 *   and what each refusal is protecting against.
 *
 * @status Off-hardware in tests/test_cooperative_matmul_plan.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_op_matmul_cooperative.h"

#include <string.h>

#include "ft_tt_device.h"

/** CB indices, matching the compile-time args the triad is built with (0,1,16) — the same
 *  three the single-core driver uses, because the compute and writer thirds are the same
 *  binaries. */
#define COOP_CB_IN0 0u
#define COOP_CB_IN1 1u
#define COOP_CB_OUT 16u

/** TWO blocks per input CB: the reader fills one while the compute thread consumes the other.
 *  It is also the invariant that keeps a reserved block contiguous AND the one that makes the
 *  CB write pointer a function of the block index, which is what the kernel checks before it
 *  multicasts into it (cooperative_matmul_protocol.h). */
#define COOP_CB_BLOCKS 2u
#define COOP_CB_OUT_PAGES 2u

/* C99 has no _Static_assert; negative array size is the portable equivalent this project
 * already uses (ft_tt_mailbox.h) and it is accepted under -std=c99 -pedantic -Werror.
 *
 * The control region has to sit between the kernel-config region and the CB arena. If the
 * Phase 4 L1 map ever moves, this is where it is noticed — at compile time, rather than as a
 * multicast into somebody's circular buffer. */
#define COOP_STATIC_ASSERT(cond, tag) typedef char ft_tt_coop_static_assert_##tag[(cond) ? 1 : -1]

COOP_STATIC_ASSERT(FT_TT_OP_COOP_CONTROL_BASE + FT_TT_OP_COOP_CONTROL_BYTES <=
                       FT_TT_OP_CB_ARENA_BASE,
                   control_region_fits_below_the_cb_arena);
COOP_STATIC_ASSERT(FT_TT_OP_COOP_CONTROL_BASE >=
                       FT_TT_OP_KERNEL_CONFIG_BASE + FT_TT_OP_KERNEL_CONFIG_BYTES,
                   control_region_clears_kernel_text_and_runtime_args);
/** The status block must not run into the ready semaphore. */
COOP_STATIC_ASSERT(FT_TT_OP_COOP_STATUS_OFFSET + FTCM_STATUS_WORDS * 4u <=
                       FT_TT_OP_COOP_READY_SEMAPHORE_OFFSET,
                   status_block_clears_the_ready_semaphore);

static int check_request(const ft_tt_op_matmul_cooperative_request *request,
                         uint32_t member_count, uint32_t leader)
{
    if (request->tile_rows == 0u || request->tile_inner == 0u || request->tile_cols == 0u) {
        ft_tt_log_error("cooperative matmul: a zero shape dimension (Mt %u, Kt %u, Nt %u)",
                        request->tile_rows, request->tile_inner, request->tile_cols);
        return FT_TT_EINVAL;
    }
    if (member_count == 0u || member_count > FT_TT_OP_COOP_MAX_MEMBERS) {
        ft_tt_log_error("cooperative matmul: %u members, must be 1..%u", member_count,
                        FT_TT_OP_COOP_MAX_MEMBERS);
        return FT_TT_EINVAL;
    }
    if (leader >= member_count) {
        ft_tt_log_error("cooperative matmul: leader %u is outside a group of %u", leader,
                        member_count);
        return FT_TT_EINVAL;
    }
    if (request->activation.tile_bytes == 0u || request->weight.tile_bytes == 0u ||
        request->output.tile_bytes == 0u) {
        ft_tt_log_error("cooperative matmul: an operand has a zero tile size");
        return FT_TT_EINVAL;
    }
    if (request->mode == FT_TT_OP_MATMUL_COOPERATIVE &&
        (request->tile_cols % member_count) != 0u) {
        /* COOPERATIVE ONLY, and the reason is the RENDEZVOUS rather than the arithmetic: a
         * ragged split gives members different BLOCK COUNTS, and the per-block multicast
         * barrier is correct only because every member walks the same number of blocks in
         * the same order. A remainder handed to the first members is exactly the shape that
         * would deadlock the group on its last blocks.
         *
         * SPLIT HAS NO RENDEZVOUS — it is N independent residencies handed disjoint columns,
         * each walking its own panel with nothing to synchronise — so a ragged split there
         * is merely uneven, and refusing it cost a whole tensor its split. MEASURED
         * 2026-08-16 on MiniCPM5-1B: at N=5 that refusal left 3,864 of 3,887 projections on
         * one core and at N=7 it left ALL of them, which is why 5, 6 and 7 read slower than
         * 4 and looked like a scaling ceiling rather than a divisibility one. */
        ft_tt_log_error("cooperative matmul: Nt %u does not divide by %u members. A ragged "
                        "split would give members different block counts, and the per-block "
                        "rendezvous requires every member to walk the same number",
                        request->tile_cols, member_count);
        return FT_TT_EINVAL;
    }
    if (request->mode != FT_TT_OP_MATMUL_INDEPENDENT && request->tile_cols < member_count) {
        /* Fewer columns than members means a member with ZERO columns, which the reader
         * refuses on the card — and a refusal there strands a triad. */
        ft_tt_log_error("cooperative matmul: Nt %u cannot be split over %u members without "
                        "giving one of them zero columns",
                        request->tile_cols, member_count);
        return FT_TT_EINVAL;
    }
    if ((request->activation_shard_count != 0u &&
         request->activation_shard_count != member_count) ||
        (request->weight_shard_count != 0u && request->weight_shard_count != member_count)) {
        /* Partly applied is the dangerous state: the members with a shard would read their own
         * panel from column 0 while the members without one read the full grid at the full
         * offset, and both are in range. That is a wrong answer with no error anywhere, which
         * is the class this file refuses rather than rounds. */
        ft_tt_log_error("cooperative matmul: %u activation shards and %u weight shards for %u "
                        "members — a shard list must cover every member or none",
                        request->activation_shard_count, request->weight_shard_count,
                        member_count);
        return FT_TT_EINVAL;
    }
    if (request->mode == FT_TT_OP_MATMUL_COOPERATIVE && member_count < 2u) {
        ft_tt_log_error("cooperative matmul: a COOPERATIVE run needs at least 2 members; one "
                        "member leaves the rectangle with zero destinations, which is the "
                        "zero-length NoC transfer in different clothes (DEV_NOTES.md §1)");
        return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

/** CB_IN0 must be 64-byte aligned because a block of `depth` A tiles is the LOCAL end of a
 *  DRAM read, and a transfer takes the STRICTER of its two ends' alignments. ft_tt_op_cb_place
 *  already leaves every address 64-byte aligned; this is the statement of why that matters
 *  here, checked rather than trusted (a 32-byte-aligned base read back byte-exact until its
 *  last 64 bytes, silently — tests/ft_tt_phase2_l1_layout.h). */
static int place_circular_buffers(const ft_tt_op_matmul_cooperative_request *request,
                                  uint32_t depth, ft_tt_op_matmul_cooperative_plan *plan)
{
    ft_tt_op_cb_request requests[3];
    int rc;

    requests[0].index = COOP_CB_IN0;
    requests[0].page_bytes = request->activation.tile_bytes;
    requests[0].page_count = depth * COOP_CB_BLOCKS;
    requests[1].index = COOP_CB_IN1;
    requests[1].page_bytes = request->weight.tile_bytes;
    requests[1].page_count = depth * COOP_CB_BLOCKS;
    requests[2].index = COOP_CB_OUT;
    requests[2].page_bytes = request->output.tile_bytes;
    requests[2].page_count = COOP_CB_OUT_PAGES;

    rc = ft_tt_op_cb_place(requests, 3u, plan->cb, plan->cb_buffer, sizeof(plan->cb_buffer),
                           &plan->cb_mask);
    if (rc != FT_TT_OK)
        return rc;

    if ((plan->cb[0].address % FT_TT_DRAM_ALIGNMENT) != 0u) {
        ft_tt_log_error("cooperative matmul: CB_IN0 landed at 0x%08x, which is not %u-byte "
                        "aligned — a DRAM read into it would be byte-exact until its last %u "
                        "bytes, with no error anywhere",
                        plan->cb[0].address, FT_TT_DRAM_ALIGNMENT, FT_TT_DRAM_ALIGNMENT);
        return FT_TT_ERROR;
    }
    return FT_TT_OK;
}

/** Bind the rectangle and the L1 layout, and check the A block against the multicast rules.
 *  The leader is INSIDE the rectangle — it computes a slice of its own — so
 *  `destination_count` is member_count-1, which is what a non-loopback multicast delivers. */
static int bind_shared_operand(const ft_tt_noc_coord *members, uint32_t member_count,
                               uint32_t leader, uint32_t depth,
                               const ft_tt_op_matmul_cooperative_request *request,
                               ft_tt_op_matmul_cooperative_plan *plan)
{
    ft_tt_group_shared_operand_layout layout;

    layout.operand_address = plan->cb[0].address;
    layout.operand_bytes = depth * request->activation.tile_bytes;
    layout.ready_semaphore_address = ft_tt_op_coop_ready_semaphore_address();
    layout.arrival_semaphore_address = ft_tt_op_coop_arrival_semaphore_address();
    layout.valid_word_address = ft_tt_op_coop_valid_word_address();

    return ft_tt_group_shared_operand_bind(members, member_count, members[leader], &layout,
                                           &plan->shared);
}

/**
 * Give every member its DOOR into each read operand's bank, and count the doors that would
 * share a NoC row.
 *
 * Member `c` takes port `c % port_count` of the bank, which is the balanced assignment the
 * per-endpoint outstanding-read budget wants (ft_tt_dram_stripe.h). An operand with no port
 * set keeps the request's own coordinate on every member — the pre-striping behaviour, and the
 * arm the measurement is against.
 *
 * The row check only sees endpoints that arrived as port sets: for an operand given a bare
 * (x, y) there is no bank, port or NOC 0 row to compare, and inventing one by reverse-looking
 * up a coordinate is the "never discover a DRAM endpoint by probing" mistake in slow motion.
 */
static void assign_dram_endpoints(const ft_tt_op_matmul_cooperative_request *request,
                                  ft_tt_op_matmul_cooperative_plan *plan)
{
    ft_tt_dram_endpoint concurrent[FT_TT_OP_COOP_MAX_MEMBERS * 2u];
    const bool activation_sharded = request->activation_shard_count == plan->member_count;
    const bool weight_sharded = request->weight_shard_count == plan->member_count;
    uint32_t concurrent_count = 0u;
    uint32_t first_left = 0u;
    uint32_t first_right = 0u;
    uint32_t index;

    for (index = 0u; index < plan->member_count; index++) {
        ft_tt_op_matmul_cooperative_member *member = &plan->member[index];
        const ft_tt_dram_endpoint *activation =
            activation_sharded ? &request->activation_shards[index].endpoint
                               : ft_tt_dram_port_set_reader_endpoint(&request->activation_ports,
                                                                     index);
        const ft_tt_dram_endpoint *weight =
            weight_sharded
                ? &request->weight_shards[index].endpoint
                : ft_tt_dram_port_set_reader_endpoint(&request->weight_ports, index);

        member->activation_address = activation_sharded
                                         ? request->activation_shards[index].address
                                         : request->activation.address;
        /* A sharded member owns its panel outright: it starts at column 0 of a grid exactly
         * its own width. The full-grid offset it would otherwise carry has been folded into
         * the address by whoever laid the panels out, which is what makes a bank-striped read
         * the SAME instruction stream as an un-striped one. */
        if (weight_sharded) {
            member->weight_address = request->weight_shards[index].address;
            member->weight_full_columns = member->column_count;
            member->weight_column_first = 0u;
        } else {
            member->weight_address = request->weight.address;
            member->weight_full_columns = request->tile_cols;
            member->weight_column_first = member->column_first;
        }

        if (activation != NULL) {
            member->activation_x = (uint16_t)activation->coord.x;
            member->activation_y = (uint16_t)activation->coord.y;
            concurrent[concurrent_count++] = *activation;
        } else {
            member->activation_x = request->activation.x;
            member->activation_y = request->activation.y;
        }
        if (weight != NULL) {
            member->weight_x = (uint16_t)weight->coord.x;
            member->weight_y = (uint16_t)weight->coord.y;
            concurrent[concurrent_count++] = *weight;
        } else {
            member->weight_x = request->weight.x;
            member->weight_y = request->weight.y;
        }
    }

    plan->row_collisions =
        ft_tt_dram_stripe_shared_rows(concurrent, concurrent_count, &first_left, &first_right);
    plan->distinct_endpoints = 0u;
    for (index = 0u; index < concurrent_count; index++) {
        uint32_t earlier;
        bool seen = false;

        for (earlier = 0u; earlier < index; earlier++)
            seen = seen || (concurrent[earlier].logical_bank == concurrent[index].logical_bank &&
                            concurrent[earlier].port == concurrent[index].port);
        if (!seen)
            plan->distinct_endpoints++;
    }

    if (plan->row_collisions != 0u)
        ft_tt_log_error("cooperative matmul: %u pairs of the DRAM doors this plan reads "
                        "through share a NoC row and so share ~60 GB/s — the first is bank %u "
                        "port %u against bank %u port %u, both on NoC 0 row %u. The plan is "
                        "CORRECT and slower; pick banks whose ports do not collide "
                        "(ft_tt_dram_stripe.h)",
                        plan->row_collisions, concurrent[first_left].logical_bank,
                        concurrent[first_left].port, concurrent[first_right].logical_bank,
                        concurrent[first_right].port, concurrent[first_left].noc_row);
}

int ft_tt_op_matmul_cooperative_plan_build(const ft_tt_noc_coord *members, uint32_t member_count,
                                           uint32_t leader,
                                           const ft_tt_op_matmul_cooperative_request *request,
                                           ft_tt_op_matmul_cooperative_plan *out)
{
    ft_tt_op_matmul_cooperative_plan plan;
    uint32_t columns;
    uint32_t remainder;
    uint32_t index;
    int rc;

    if (request == NULL || out == NULL) {
        ft_tt_log_error("cooperative matmul: NULL request/out");
        return FT_TT_EINVAL;
    }
    if (request->mode == FT_TT_OP_MATMUL_COOPERATIVE && members == NULL) {
        ft_tt_log_error("cooperative matmul: a COOPERATIVE plan needs the member coordinates");
        return FT_TT_EINVAL;
    }
    rc = check_request(request, member_count, leader);
    if (rc != FT_TT_OK)
        return rc;

    memset(&plan, 0, sizeof(plan));
    plan.member_count = member_count;
    plan.leader = leader;
    plan.depth = ft_tt_op_block_depth(request->tile_inner, request->prefetch_tiles);

    rc = place_circular_buffers(request, plan.depth, &plan);
    if (rc != FT_TT_OK)
        return rc;

    /* INDEPENDENT is not a split: every member does the whole width into its own output. */
    columns = (request->mode == FT_TT_OP_MATMUL_INDEPENDENT) ? request->tile_cols
                                                             : request->tile_cols / member_count;
    /* THE REMAINDER GOES TO THE FIRST MEMBERS, one column each. Widths then differ by at
     * most one, which is the closest a column split can come to balanced — and the OFFSET
     * has to be the running sum rather than `index * columns`, or the wider members' extra
     * columns would be computed twice and the last member's would fall off the end. */
    remainder = (request->mode == FT_TT_OP_MATMUL_INDEPENDENT) ? 0u
                                                               : request->tile_cols % member_count;

    for (index = 0u; index < member_count; index++) {
        ft_tt_op_matmul_cooperative_member *member = &plan.member[index];

        member->column_count = columns + (index < remainder ? 1u : 0u);
        member->column_first =
            (request->mode == FT_TT_OP_MATMUL_INDEPENDENT)
                ? 0u
                : index * columns + (index < remainder ? index : remainder);
        member->output_tiles = request->tile_rows * member->column_count;
        /* INDEPENDENT gives every member the whole width into a panel OF ITS OWN, so its
         * stride is the panel; a split writes into ONE shared output at its own column
         * offset. The two coincide for an even split, which is why the running sum is only
         * visibly necessary once the widths differ. */
        member->output_address =
            request->output.address +
            ((request->mode == FT_TT_OP_MATMUL_INDEPENDENT)
                 ? index * member->output_tiles
                 : member->column_first * request->tile_rows) *
                request->output.tile_bytes;
        if (request->mode != FT_TT_OP_MATMUL_COOPERATIVE)
            member->role = FTCM_ROLE_SOLO;
        else
            member->role = (index == leader) ? FTCM_ROLE_LEADER : FTCM_ROLE_RECEIVER;
    }

    assign_dram_endpoints(request, &plan);

    if (request->mode == FT_TT_OP_MATMUL_COOPERATIVE) {
        rc = bind_shared_operand(members, member_count, leader, plan.depth, request, &plan);
        if (rc != FT_TT_OK)
            return rc;
        if (plan.shared.receiver_count != member_count - 1u) {
            /* The leader is inside the rectangle, so the hardware delivers to member_count-1
             * nodes. Anything else means the rectangle is not the group, and the barrier's
             * software counter would then be advanced by the wrong number — too many hangs,
             * too few returns early (ft_tt_multicast.h). */
            ft_tt_log_error("cooperative matmul: the rectangle leaves %u destinations for a "
                            "group of %u with the leader inside it; expected %u",
                            plan.shared.receiver_count, member_count, member_count - 1u);
            return FT_TT_ERROR;
        }
        plan.shared_valid = true;
    }

    *out = plan;
    return FT_TT_OK;
}

int ft_tt_op_matmul_cooperative_reader_arguments(
    const ft_tt_op_matmul_cooperative_plan *plan,
    const ft_tt_op_matmul_cooperative_request *request, const ft_tt_noc_coord *members,
    uint32_t member, uint32_t *out, uint32_t capacity)
{
    const ft_tt_op_matmul_cooperative_member *slice;

    if (plan == NULL || request == NULL || out == NULL)
        return FT_TT_EINVAL;
    if (member >= plan->member_count)
        return FT_TT_EINVAL;
    if (capacity < FTCM_ARG_COUNT)
        return FT_TT_ENOMEM;
    if (plan->shared_valid && members == NULL)
        return FT_TT_EINVAL;

    slice = &plan->member[member];
    memset(out, 0, (size_t)FTCM_ARG_COUNT * sizeof(out[0]));

    out[FTCM_ARG_ROLE] = slice->role;
    out[FTCM_ARG_TILE_ROWS] = request->tile_rows;
    out[FTCM_ARG_TILE_INNER] = request->tile_inner;
    out[FTCM_ARG_TILE_COLS] = slice->column_count;
    /* The DOOR, not the operand's own coordinate: with a port set these differ per member and
     * name the same bytes through a different NoC port. The ADDRESSES are untouched, which is
     * the whole claim striping makes — a port is a door into one memory, not a memory. */
    out[FTCM_ARG_A_X] = slice->activation_x;
    out[FTCM_ARG_A_Y] = slice->activation_y;
    out[FTCM_ARG_A_BASE] = slice->activation_address;
    out[FTCM_ARG_B_X] = slice->weight_x;
    out[FTCM_ARG_B_Y] = slice->weight_y;
    out[FTCM_ARG_B_BASE] = slice->weight_address;
    out[FTCM_ARG_B_COLUMN_FIRST] = slice->weight_column_first;
    out[FTCM_ARG_B_FULL_COLS] = slice->weight_full_columns;
    out[FTCM_ARG_DEPTH] = plan->depth;
    out[FTCM_ARG_B_N_MAJOR] = request->weight_n_major ? 1u : 0u;
    out[FTCM_ARG_CB_IN0_BASE] = plan->cb[0].address;
    out[FTCM_ARG_STATUS] = ft_tt_op_coop_status_address();
    out[FTCM_ARG_STOP_FLAG] = ft_tt_op_coop_stop_flag_address();
    out[FTCM_ARG_SPIN_LIMIT] = request->spin_limit;

    /* A SOLO member is given no rectangle at all — not a degenerate one. A zeroed rectangle is
     * (0,0)..(0,0), a REAL live endpoint of the wrong type, and the family of mistake that
     * hangs the NoC; leaving destination_count at 0 is what makes the kernel refuse to issue
     * rather than issue at a plausible-looking wrong place. */
    if (plan->shared_valid) {
        out[FTCM_ARG_RECTANGLE_X_LOW] = plan->shared.rectangle.x_low;
        out[FTCM_ARG_RECTANGLE_Y_LOW] = plan->shared.rectangle.y_low;
        out[FTCM_ARG_RECTANGLE_X_HIGH] = plan->shared.rectangle.x_high;
        out[FTCM_ARG_RECTANGLE_Y_HIGH] = plan->shared.rectangle.y_high;
        out[FTCM_ARG_DESTINATION_COUNT] = plan->shared.rectangle.destination_count;
        out[FTCM_ARG_LEADER_X] = members[plan->leader].x;
        out[FTCM_ARG_LEADER_Y] = members[plan->leader].y;
        out[FTCM_ARG_READY_SEMAPHORE] = plan->shared.layout.ready_semaphore_address;
        out[FTCM_ARG_ARRIVAL_SEMAPHORE] = plan->shared.layout.arrival_semaphore_address;
        out[FTCM_ARG_VALID_WORD] = plan->shared.layout.valid_word_address;
    }
    return FT_TT_OK;
}

int ft_tt_op_matmul_cooperative_compute_arguments(
    const ft_tt_op_matmul_cooperative_plan *plan,
    const ft_tt_op_matmul_cooperative_request *request, uint32_t member, uint32_t *out,
    uint32_t capacity)
{
    if (plan == NULL || request == NULL || out == NULL)
        return FT_TT_EINVAL;
    if (member >= plan->member_count)
        return FT_TT_EINVAL;
    if (capacity < 3u)
        return FT_TT_ENOMEM;

    out[0] = request->tile_rows;
    out[1] = request->tile_inner;
    out[2] = plan->member[member].column_count;
    return FT_TT_OK;
}

int ft_tt_op_matmul_cooperative_writer_arguments(
    const ft_tt_op_matmul_cooperative_plan *plan,
    const ft_tt_op_matmul_cooperative_request *request, uint32_t member, uint32_t *out,
    uint32_t capacity)
{
    if (plan == NULL || request == NULL || out == NULL)
        return FT_TT_EINVAL;
    if (member >= plan->member_count)
        return FT_TT_EINVAL;
    if (capacity < 4u)
        return FT_TT_ENOMEM;

    out[0] = plan->member[member].output_tiles;
    out[1] = request->output.x;
    out[2] = request->output.y;
    out[3] = plan->member[member].output_address;
    return FT_TT_OK;
}
