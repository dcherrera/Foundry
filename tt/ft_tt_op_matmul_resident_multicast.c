/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_matmul_resident_multicast.c
 * @brief Implementation of ft_tt_op_matmul_resident_multicast.h — see that header for why
 *   each refusal exists and why the rendezvous words are the one-shot kernel's.
 *
 * @status Off-hardware except for the three reset writes.
 * @issues None
 * @todo None
 */

#include "ft_tt_op_matmul_resident_multicast.h"

#include "ft_tt_device.h"
#include "ft_tt_op_matmul_cooperative.h"

int ft_tt_op_matmul_resident_multicast_plan(const ft_tt_noc_coord *members,
                                            uint32_t member_count, uint32_t leader,
                                            uint32_t spin_limit,
                                            ft_tt_op_matmul_resident_multicast *out)
{
    ft_tt_op_matmul_resident_multicast plan;
    uint32_t index;
    int rc;

    if (members == NULL || out == NULL)
        return FT_TT_EINVAL;
    if (member_count < 2u || member_count > FT_TT_RESIDENT_MULTICAST_MAX_MEMBERS) {
        ft_tt_log_error("resident multicast: %u members is outside 2..%u", member_count,
                        FT_TT_RESIDENT_MULTICAST_MAX_MEMBERS);
        return FT_TT_EINVAL;
    }
    if (leader >= member_count) {
        ft_tt_log_error("resident multicast: leader %u of %u members", leader, member_count);
        return FT_TT_EINVAL;
    }
    /* 0 would have to mean "spin forever", and a resident kernel with an unbounded wait is
     * the one thing this backend may not be configured into: `assert_reset` is then the
     * only way to end it, and asserting reset on a blocked triad is how cores were
     * wedged. */
    if (spin_limit == 0u) {
        ft_tt_log_error("resident multicast: a zero rendezvous spin limit is unbounded");
        return FT_TT_EINVAL;
    }

    for (index = 0u; index < member_count; index++)
        plan.member[index] = members[index];
    for (; index < FT_TT_RESIDENT_MULTICAST_MAX_MEMBERS; index++) {
        plan.member[index].x = 0u;
        plan.member[index].y = 0u;
    }
    plan.member_count = member_count;
    plan.leader = leader;
    plan.receiver_count = member_count - 1u;
    plan.spin_limit = spin_limit;
    plan.ready_semaphore_address = ft_tt_op_coop_ready_semaphore_address();
    plan.arrival_semaphore_address = ft_tt_op_coop_arrival_semaphore_address();
    plan.counter_word_address = ft_tt_op_coop_valid_word_address();

    rc = ft_tt_group_broadcast_partition_build(members, member_count, members[leader],
                                               FT_TT_GROUP_BROADCAST_HAZARD_LIMIT,
                                               &plan.partition);
    if (rc != FT_TT_OK)
        return rc;
    /* The partition already proves this, and it is checked again HERE because the kernel
     * reads the two numbers from different runtime args and compares them itself: a plan
     * whose receiver count and destination total disagree would make the leader's write
     * barrier either hang or return early, and neither is visible as a wrong answer. */
    if (plan.partition.destination_total != plan.receiver_count) {
        ft_tt_log_error("resident multicast: %u destinations for %u receivers",
                        plan.partition.destination_total, plan.receiver_count);
        return FT_TT_ERROR;
    }

    *out = plan;
    return FT_TT_OK;
}

int ft_tt_op_matmul_resident_multicast_args(const ft_tt_op_matmul_resident_multicast *plan,
                                            uint32_t index, uint32_t out_args[kArgCoopCount])
{
    uint32_t group;

    if (plan == NULL || out_args == NULL || index >= plan->member_count)
        return FT_TT_EINVAL;

    out_args[kArgCoopRole] = index == plan->leader ? FT_TT_RESIDENT_COOP_ROLE_LEADER
                                                   : FT_TT_RESIDENT_COOP_ROLE_RECEIVER;
    out_args[kArgCoopLeaderX] = plan->member[plan->leader].x;
    out_args[kArgCoopLeaderY] = plan->member[plan->leader].y;
    out_args[kArgCoopReadySemaphore] = plan->ready_semaphore_address;
    out_args[kArgCoopArrivalSemaphore] = plan->arrival_semaphore_address;
    out_args[kArgCoopCounterWord] = plan->counter_word_address;
    out_args[kArgCoopReceiverCount] = plan->receiver_count;
    out_args[kArgCoopSpinLimit] = plan->spin_limit;

    /* THE GEOMETRY IS THE LEADER'S ALONE. A receiver given a rectangle it does not use is
     * a receiver one bug away from broadcasting, so it is given zeroes — which the kernel
     * refuses to issue rather than aiming at (0,0). */
    if (index != plan->leader) {
        out_args[kArgCoopGroupCount] = 0u;
        for (group = 0u; group < FT_TT_RESIDENT_COOP_MAX_GROUPS; group++)
            out_args[kArgCoopGroup0 + group] = 0u;
        return FT_TT_OK;
    }
    if (plan->partition.rectangle_count > FT_TT_RESIDENT_COOP_MAX_GROUPS) {
        ft_tt_log_error("resident multicast: %u pieces will not fit the %u arg words the "
                        "kernel reads", plan->partition.rectangle_count,
                        FT_TT_RESIDENT_COOP_MAX_GROUPS);
        return FT_TT_EINVAL;
    }
    out_args[kArgCoopGroupCount] = plan->partition.rectangle_count;
    for (group = 0u; group < FT_TT_RESIDENT_COOP_MAX_GROUPS; group++) {
        const ft_tt_multicast_rectangle *piece = &plan->partition.rectangle[group];
        uint32_t packed = 0u;

        if (group < plan->partition.rectangle_count) {
            packed = ftcoop_pack_rectangle(piece->x_low, piece->y_low, piece->x_high,
                                           piece->y_high, piece->destination_count);
            /* ftcoop_pack_rectangle() returns 0 — never a legal piece — for anything out
             * of range, so this catches a coordinate above the 6-bit node-id field before
             * it can TRUNCATE onto a different live endpoint on the card. */
            if (packed == 0u) {
                ft_tt_log_error("resident multicast: piece %u (%u,%u)..(%u,%u) x%u does not "
                                "fit the packed rectangle word", group, piece->x_low,
                                piece->y_low, piece->x_high, piece->y_high,
                                piece->destination_count);
                return FT_TT_EINVAL;
            }
        }
        out_args[kArgCoopGroup0 + group] = packed;
    }
    return FT_TT_OK;
}

int ft_tt_op_matmul_resident_multicast_reset(const ft_tt_noc_access *noc,
                                             const ft_tt_op_matmul_resident_multicast *plan)
{
    const uint32_t zero = 0u;
    uint32_t index;
    int rc;

    if (noc == NULL || noc->write == NULL || plan == NULL)
        return FT_TT_EINVAL;
    if (plan->member_count < 2u || plan->leader >= plan->member_count)
        return FT_TT_EINVAL;

    for (index = 0u; index < plan->member_count; index++) {
        rc = noc->write(noc->user, (uint16_t)plan->member[index].x,
                        (uint16_t)plan->member[index].y, plan->arrival_semaphore_address,
                        &zero, sizeof(zero));
        if (rc != FT_TT_OK) {
            ft_tt_log_error("resident multicast: clearing the arrival word on member %u at "
                            "(%u,%u) failed: rc=%d (%s)", index, plan->member[index].x,
                            plan->member[index].y, rc, ft_tt_error_string(rc));
            return rc;
        }
    }
    rc = noc->write(noc->user, (uint16_t)plan->member[plan->leader].x,
                    (uint16_t)plan->member[plan->leader].y, plan->counter_word_address, &zero,
                    sizeof(zero));
    if (rc == FT_TT_OK)
        rc = noc->write(noc->user, (uint16_t)plan->member[plan->leader].x,
                        (uint16_t)plan->member[plan->leader].y, plan->ready_semaphore_address,
                        &zero, sizeof(zero));
    if (rc != FT_TT_OK)
        ft_tt_log_error("resident multicast: staging the leader's control words at (%u,%u) "
                        "failed: rc=%d (%s)", plan->member[plan->leader].x,
                        plan->member[plan->leader].y, rc, ft_tt_error_string(rc));
    return rc;
}
