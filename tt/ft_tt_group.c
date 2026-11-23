/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_group.c
 * @brief Core-group partitioning — see ft_tt_group.h for what "disjoint" buys and why
 *   there is not a single core, column or row count in this file.
 *
 * @status Green off-hardware in tests/test_queue.c, including a synthetic 12-column grid,
 *   and on the card in tests/test_phase5_resident.c and tests/test_phase6_triad.c.
 * @issues None
 * @todo None
 */

/* clock_gettime and CLOCK_MONOTONIC are POSIX.1-2008; -std=c99 turns off glibc's default
 * feature set, so ask for them by name or the Linux card host's gcc hides them. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_group.h"

#include <string.h>
#include <time.h>

#include "ft_tt_device.h"

static uint64_t ft_tt_group_monotonic_microseconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0u;
    return (uint64_t)now.tv_sec * 1000000u + (uint64_t)(now.tv_nsec / 1000);
}

int ft_tt_group_partition(const ft_tt_grid_info *grid, uint32_t first_index, uint32_t index_count,
                          uint32_t group_count, ft_tt_core_group *groups,
                          ft_tt_launch_core_placement *placements, ft_tt_queue *queues,
                          uint32_t storage_count)
{
    uint32_t live_cores;
    uint32_t live_banks;
    uint32_t base_size;
    uint32_t remainder;
    uint32_t assigned = 0u;
    uint32_t group;

    if (!grid || !groups || !placements || !queues || group_count == 0u)
        return FT_TT_EINVAL;

    /* THE ONLY SOURCE OF SIZE IN THIS FILE. Everything below is arithmetic on it. */
    live_cores = ft_tt_coord_tensix_core_count(grid);
    if (live_cores == 0u) {
        ft_tt_log_error("group: the grid reports no enabled Tensix cores");
        return FT_TT_EINVAL;
    }
    if (first_index >= live_cores) {
        ft_tt_log_error("group: first index %u is at or past the live core count %u", first_index,
                        live_cores);
        return FT_TT_EINVAL;
    }
    if (index_count == 0u)
        index_count = live_cores - first_index;
    if (first_index + index_count > live_cores) {
        ft_tt_log_error("group: indices %u..%u run past the live core count %u", first_index,
                        first_index + index_count - 1u, live_cores);
        return FT_TT_EINVAL;
    }
    if (group_count > index_count) {
        ft_tt_log_error("group: %u groups cannot be carved out of %u cores", group_count,
                        index_count);
        return FT_TT_EINVAL;
    }
    if (storage_count < index_count)
        return FT_TT_ENOMEM;

    live_banks = grid->dram_bank_count;
    if (live_banks == 0u) {
        ft_tt_log_error("group: the grid reports no DRAM banks");
        return FT_TT_EINVAL;
    }

    base_size = index_count / group_count;
    remainder = index_count % group_count;

    memset(groups, 0, (size_t)group_count * sizeof(*groups));

    for (group = 0u; group < group_count; group++) {
        uint32_t size = base_size + (group < remainder ? 1u : 0u);
        uint32_t slot;
        int rc;

        groups[group].id = group;
        groups[group].first_tensix_index = first_index + assigned;
        groups[group].tensix_index_count = size;
        groups[group].queues = queues + assigned;

        /* Banks are handed out by STRIDE rather than by run, so a part with fewer banks
         * than groups still gives every group at least one and no two groups share one
         * unless there are genuinely fewer banks than groups. */
        groups[group].dram_bank_first = group % live_banks;
        groups[group].dram_bank_stride = group_count <= live_banks ? group_count : live_banks;
        groups[group].dram_bank_count =
            (live_banks - groups[group].dram_bank_first + groups[group].dram_bank_stride - 1u) /
            groups[group].dram_bank_stride;

        rc = ft_tt_launch_core_set_init(&groups[group].cores, placements + assigned, size);
        if (rc != FT_TT_OK)
            return rc;

        for (slot = 0u; slot < size; slot++) {
            rc = ft_tt_launch_core_set_add_tensix(&groups[group].cores, grid,
                                                  groups[group].first_tensix_index + slot, NULL);
            if (rc != FT_TT_OK) {
                ft_tt_log_error("group %u: Tensix index %u could not be placed (%d)", group,
                                groups[group].first_tensix_index + slot, rc);
                return rc;
            }
        }

        assigned += size;
    }

    return FT_TT_OK;
}

int ft_tt_group_init_allocators(ft_tt_core_group *groups, uint32_t group_count,
                                const ft_tt_grid_info *grid, ft_tt_ctx *ctx)
{
    uint32_t group;

    if (!groups || !grid || group_count == 0u)
        return FT_TT_EINVAL;

    for (group = 0u; group < group_count; group++) {
        /* Every group's L1 arena starts at the same place because groups own disjoint
         * CORES — two groups cannot reach one L1, so there is nothing to carve. */
        int rc = ft_tt_allocator_init(&groups[group].allocator, ctx, grid, FT_TT_OP_CB_ARENA_BASE);
        if (rc != FT_TT_OK) {
            ft_tt_log_error("group %u: allocator init failed (%d)", group, rc);
            return rc;
        }
    }
    return FT_TT_OK;
}

int ft_tt_group_dram_bank(const ft_tt_core_group *group, uint32_t index, uint32_t *out_bank)
{
    if (!group || !out_bank || index >= group->dram_bank_count)
        return FT_TT_EINVAL;
    *out_bank = group->dram_bank_first + index * group->dram_bank_stride;
    return FT_TT_OK;
}

int ft_tt_group_check_disjoint(const ft_tt_core_group *groups, uint32_t group_count,
                               const char **out_reason)
{
    const char *reason = NULL;
    uint32_t outer;

    if (!groups || group_count == 0u) {
        if (out_reason)
            *out_reason = "bad argument";
        return FT_TT_EINVAL;
    }

    for (outer = 0u; outer < group_count && !reason; outer++) {
        unsigned outer_count = ft_tt_launch_core_set_count(&groups[outer].cores);
        unsigned outer_slot;

        if (outer_count == 0u) {
            reason = "a group holds no cores";
            break;
        }
        for (outer_slot = 0u; outer_slot < outer_count && !reason; outer_slot++) {
            const ft_tt_launch_core_placement *a =
                ft_tt_launch_core_set_at(&groups[outer].cores, outer_slot);
            uint32_t inner;

            for (inner = outer + 1u; inner < group_count && !reason; inner++) {
                unsigned inner_count = ft_tt_launch_core_set_count(&groups[inner].cores);
                unsigned inner_slot;

                for (inner_slot = 0u; inner_slot < inner_count; inner_slot++) {
                    const ft_tt_launch_core_placement *b =
                        ft_tt_launch_core_set_at(&groups[inner].cores, inner_slot);
                    if (a->x == b->x && a->y == b->y) {
                        ft_tt_log_error("group: (%u, %u) is in group %u and group %u",
                                        (unsigned)a->x, (unsigned)a->y, outer, inner);
                        reason = "two groups hold the same core";
                        break;
                    }
                }
            }
        }
    }

    if (out_reason)
        *out_reason = reason;
    return reason ? FT_TT_ERROR : FT_TT_OK;
}

int ft_tt_group_queues_init(const ft_tt_noc_access *noc, ft_tt_core_group *group,
                            uint32_t control_address, uint32_t capacity, uint32_t run_nonce)
{
    unsigned count;
    unsigned slot;

    if (!noc || !group || !group->queues)
        return FT_TT_EINVAL;

    count = ft_tt_launch_core_set_count(&group->cores);
    if (count == 0u)
        return FT_TT_EINVAL;

    for (slot = 0u; slot < count; slot++) {
        const ft_tt_launch_core_placement *placement =
            ft_tt_launch_core_set_at(&group->cores, slot);
        int rc = ft_tt_queue_init(noc, placement->x, placement->y, control_address, capacity,
                                  run_nonce, &group->queues[slot]);
        if (rc != FT_TT_OK) {
            ft_tt_log_error("group %u: queue init failed on core (%u, %u) (%d)", group->id,
                            (unsigned)placement->x, (unsigned)placement->y, rc);
            return rc;
        }
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* Group-level completion                                                    */
/* ------------------------------------------------------------------------- */

/** The leader — the core the block lives on. Slot 0, by definition and by nothing else:
 *  every member already knows the whole core set, so any fixed choice serves. */
static const ft_tt_launch_core_placement *group_leader(const ft_tt_core_group *group)
{
    return ft_tt_launch_core_set_at(&group->cores, 0u);
}

int ft_tt_group_completion_init(const ft_tt_noc_access *noc, ft_tt_core_group *group,
                                uint32_t group_address)
{
    const ft_tt_launch_core_placement *leader;
    unsigned members;
    int rc;

    if (!noc || !group)
        return FT_TT_EINVAL;

    members = ft_tt_launch_core_set_count(&group->cores);
    if (members == 0u)
        return FT_TT_EINVAL;

    leader = group_leader(group);
    rc = ft_tt_queue_group_init(noc, leader->x, leader->y, group_address, (uint32_t)members);
    if (rc != FT_TT_OK) {
        ft_tt_log_error("group %u: writing the completion block at 0x%08x on (%u, %u) failed (%d)",
                        group->id, group_address, (unsigned)leader->x, (unsigned)leader->y, rc);
        return rc;
    }
    group->completion_address = group_address;
    return FT_TT_OK;
}

int ft_tt_group_kernel_binding(const ft_tt_core_group *group, uint32_t slot, unsigned processor,
                               unsigned publisher, uint32_t idle_spin_limit,
                               uint32_t cb_wait_spin_limit,
                               ft_tt_queue_kernel_binding *out_binding)
{
    const ft_tt_launch_core_placement *leader;

    if (!group || !out_binding)
        return FT_TT_EINVAL;
    if (processor >= FT_TT_PROC_COUNT || publisher >= FT_TT_PROC_COUNT)
        return FT_TT_EINVAL;
    if (slot >= ft_tt_launch_core_set_count(&group->cores))
        return FT_TT_EINVAL;

    memset(out_binding, 0, sizeof(*out_binding));
    out_binding->idle_spin_limit = idle_spin_limit;
    out_binding->cb_wait_spin_limit = cb_wait_spin_limit;
    out_binding->status_address =
        FT_TT_QUEUE_STATUS_ADDRESS(group->queues[slot].control_address, processor);

    /* ONLY the publishing processor is given the group address. A second poster would give
     * one slot two writers, which is the one property the slot array exists to have. */
    if (group->completion_address != 0u && processor == publisher) {
        leader = group_leader(group);
        out_binding->group_x = leader->x;
        out_binding->group_y = leader->y;
        out_binding->group_address = group->completion_address;
        out_binding->group_member = slot;
    }
    return FT_TT_OK;
}

int ft_tt_group_completion_read(const ft_tt_noc_access *noc, const ft_tt_core_group *group,
                                ft_tt_group_member_completion *out_members, uint32_t capacity)
{
    const ft_tt_launch_core_placement *leader;
    unsigned members;

    if (!noc || !group || !out_members)
        return FT_TT_EINVAL;
    if (group->completion_address == 0u)
        return FT_TT_EINVAL;

    members = ft_tt_launch_core_set_count(&group->cores);
    if (members == 0u || members > capacity)
        return FT_TT_EINVAL;

    leader = group_leader(group);
    return ft_tt_queue_group_read(noc, leader->x, leader->y, group->completion_address,
                                  (uint32_t)members, out_members);
}

int ft_tt_group_check_completion(const ft_tt_core_group *group,
                                 const ft_tt_group_member_completion *members, uint32_t count,
                                 uint32_t sequence, uint32_t expected_token,
                                 const char **out_reason, uint32_t *out_short_slot)
{
    const char *reason = NULL;
    uint32_t short_slot = 0u;
    uint32_t slot;

    if (!group || !members || count == 0u || sequence == 0u) {
        if (out_reason)
            *out_reason = "bad argument";
        return FT_TT_EINVAL;
    }
    if (count != ft_tt_launch_core_set_count(&group->cores)) {
        if (out_reason)
            *out_reason = "the member count does not match the group";
        return FT_TT_EINVAL;
    }

    for (slot = 0u; slot < count && !reason; slot++) {
        if (members[slot].sequence < sequence) {
            /* Includes the "never posted" case, because 0 is not a sequence any descriptor
             * may carry. */
            reason = "a member has not reached this sequence";
            short_slot = slot;
        } else if (members[slot].token != expected_token) {
            /* What refuses a stale block: a previous run's slots can hold a large enough
             * sequence, and cannot hold this run's nonce-derived token. */
            reason = "a member's token is not this run's token for that sequence";
            short_slot = slot;
        }
    }

    if (out_reason)
        *out_reason = reason;
    if (out_short_slot)
        *out_short_slot = short_slot;
    return reason ? FT_TT_ERROR : FT_TT_OK;
}

int ft_tt_group_wait_completion(const ft_tt_noc_access *noc, const ft_tt_core_group *group,
                                uint32_t sequence, uint32_t timeout_ms, uint32_t *out_reached)
{
    ft_tt_group_member_completion members[FT_TT_QUEUE_GROUP_MAX_MEMBERS];
    uint64_t deadline;
    unsigned count;

    if (!noc || !group)
        return FT_TT_EINVAL;
    /* Same rule as everywhere else here: there is no untimed path on a board where a hang
     * costs a physical power cycle (spec §7.3). */
    if (timeout_ms == 0u)
        return FT_TT_EINVAL;

    count = ft_tt_launch_core_set_count(&group->cores);
    if (count == 0u || count > FT_TT_QUEUE_GROUP_MAX_MEMBERS)
        return FT_TT_EINVAL;

    deadline = ft_tt_group_monotonic_microseconds() + (uint64_t)timeout_ms * 1000u;
    for (;;) {
        uint32_t lowest = 0xFFFFFFFFu;
        unsigned slot;
        int rc = ft_tt_group_completion_read(noc, group, members, count);

        if (rc != FT_TT_OK)
            return rc;
        for (slot = 0u; slot < count; slot++) {
            if (members[slot].sequence < lowest)
                lowest = members[slot].sequence;
        }
        if (lowest >= sequence) {
            if (out_reached)
                *out_reached = lowest;
            return FT_TT_OK;
        }
        if (ft_tt_group_monotonic_microseconds() >= deadline) {
            if (out_reached)
                *out_reached = lowest;
            return FT_TT_EIO;
        }
    }
}
