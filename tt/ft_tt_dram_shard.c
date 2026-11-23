/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_dram_shard.c
 * @brief Implementation of ft_tt_dram_shard.h — see that header for why the placement is the
 *   loader's decision and which measured rules it respects.
 *
 * @status Off-hardware in tests/test_cooperative_matmul_plan.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_dram_shard.h"

#include <string.h>

#include "ft_tt_device.h"

/**
 * The bank member `member` of operand `operand` reads from.
 *
 * Round-robin over members, with each operand offset a whole FRACTION of the bank list from
 * the last. The offset is what keeps a member's two operands in different banks — member c
 * would otherwise pull both through one bank's ~64 GB/s and the second door would buy it
 * nothing.
 */
static uint32_t bank_of(uint32_t bank_count, uint32_t operand_count, uint32_t operand,
                        uint32_t member)
{
    return (member + operand * (bank_count / operand_count)) % bank_count;
}

/**
 * Does every operand fit in the bank that will hold the most of it?
 *
 * Checked BEFORE anything is placed so the plan is written only on success, and computable in
 * closed form because the map is round-robin: the busiest bank holds exactly
 * `ceil(member_count / bank_count)` pieces of each operand.
 *
 * The arithmetic is 64-bit deliberately. In 32 it would WRAP and place a piece at a low, live
 * DRAM address — which is not an error anywhere, it is a wrong answer.
 */
static int check_extents(const ft_tt_dram_shard_operand *operands, uint32_t operand_count,
                         uint32_t member_count, uint32_t bank_count, uint64_t bank_bytes)
{
    const uint32_t busiest = (member_count + bank_count - 1u) / bank_count;
    uint32_t operand;

    for (operand = 0u; operand < operand_count; operand++) {
        const uint64_t end = (uint64_t)operands[operand].bank_base_address +
                             (uint64_t)busiest * operands[operand].member_bytes;

        if (end > bank_bytes) {
            ft_tt_log_error("dram shard: operand %u needs %u pieces of %u bytes from 0x%08x in "
                            "the busiest bank, ending at %llu of %llu — %u members do not fit "
                            "at this base",
                            operand, busiest, operands[operand].member_bytes,
                            operands[operand].bank_base_address, (unsigned long long)end,
                            (unsigned long long)bank_bytes, member_count);
            return FT_TT_ENOMEM;
        }
    }
    return FT_TT_OK;
}

/** Refuse before anything is computed: a bad count here becomes a wrong DRAM address, and a
 *  wrong DRAM address is a wrong answer with no error anywhere. */
static int check_arguments(const ft_tt_grid_info *grid,
                           const ft_tt_dram_shard_operand *operands, uint32_t operand_count,
                           uint32_t member_count, const ft_tt_dram_shard_plan *out)
{
    uint32_t operand;

    if (grid == NULL || operands == NULL || out == NULL) {
        ft_tt_log_error("dram shard: NULL grid/operands/out");
        return FT_TT_EINVAL;
    }
    if (operand_count == 0u || operand_count > FT_TT_DRAM_SHARD_MAX_OPERANDS) {
        ft_tt_log_error("dram shard: %u operands is outside 1..%u", operand_count,
                        FT_TT_DRAM_SHARD_MAX_OPERANDS);
        return FT_TT_EINVAL;
    }
    if (member_count == 0u || member_count > FT_TT_DRAM_SHARD_MAX_MEMBERS) {
        ft_tt_log_error("dram shard: %u members is outside 1..%u", member_count,
                        FT_TT_DRAM_SHARD_MAX_MEMBERS);
        return FT_TT_EINVAL;
    }

    for (operand = 0u; operand < operand_count; operand++) {
        uint32_t earlier;

        if (operands[operand].member_bytes == 0u) {
            ft_tt_log_error("dram shard: operand %u places 0 bytes per member", operand);
            return FT_TT_EINVAL;
        }
        /* Two operands sharing a base would stack their members over each other and both
         * would still read a live endpoint and complete. That exact shape — two pieces at
         * one base, everything green — is what the byte comparison against single-core
         * caught at N=12 and N=16 and nothing else did (plan Phase 6.7). */
        for (earlier = 0u; earlier < operand; earlier++) {
            if (operands[earlier].bank_base_address == operands[operand].bank_base_address) {
                ft_tt_log_error("dram shard: operands %u and %u share bank base 0x%08x — "
                                "their members would silently overwrite each other",
                                earlier, operand, operands[operand].bank_base_address);
                return FT_TT_EINVAL;
            }
        }
    }
    return FT_TT_OK;
}

/**
 * Fill in `distinct_endpoints`, `row_collisions` and the two balance figures.
 *
 * Reported rather than enforced, all four of them: a row collision above 12 streams is
 * structural on this part and a bank imbalance is a performance fact, so refusing either
 * would reject a plan that returns the right bytes.
 */
static void summarise(ft_tt_dram_shard_plan *plan, const ft_tt_dram_endpoint *doors,
                      uint32_t door_count, uint32_t bank_count)
{
    uint32_t streams[FT_TT_MAX_DRAM_BANKS] = {0u};
    uint32_t index;
    uint32_t bank;

    plan->row_collisions = ft_tt_dram_stripe_shared_rows(doors, door_count, NULL, NULL);

    plan->distinct_endpoints = 0u;
    for (index = 0u; index < door_count; index++) {
        uint32_t earlier;
        bool seen = false;

        for (earlier = 0u; earlier < index; earlier++) {
            if (doors[earlier].logical_bank == doors[index].logical_bank &&
                doors[earlier].port == doors[index].port) {
                seen = true;
                break;
            }
        }
        if (!seen)
            plan->distinct_endpoints++;
        if (doors[index].logical_bank < FT_TT_MAX_DRAM_BANKS)
            streams[doors[index].logical_bank]++;
    }

    plan->busiest_bank_streams = 0u;
    plan->quietest_bank_streams = 0xFFFFFFFFu;
    for (bank = 0u; bank < bank_count && bank < FT_TT_MAX_DRAM_BANKS; bank++) {
        if (streams[bank] > plan->busiest_bank_streams)
            plan->busiest_bank_streams = streams[bank];
        if (streams[bank] < plan->quietest_bank_streams)
            plan->quietest_bank_streams = streams[bank];
    }
    if (plan->quietest_bank_streams == 0xFFFFFFFFu)
        plan->quietest_bank_streams = 0u;
}

int ft_tt_dram_shard_plan_build(const ft_tt_grid_info *grid,
                                const ft_tt_dram_shard_operand *operands,
                                uint32_t operand_count, uint32_t member_count,
                                ft_tt_coord_system system, ft_tt_dram_shard_plan *out)
{
    /* One entry per (member, operand) concurrent stream, MEMBER-MAJOR. The order is part of
     * the contract: ft_tt_dram_stripe_choose_ports() is greedy in reader order, so a
     * different flattening is a different — still correct, differently fast — door map, and
     * a plan that is not reproducible cannot be the input to a byte comparison. */
    uint32_t banks[FT_TT_DRAM_SHARD_MAX_MEMBERS * FT_TT_DRAM_SHARD_MAX_OPERANDS];
    ft_tt_dram_endpoint doors[FT_TT_DRAM_SHARD_MAX_MEMBERS * FT_TT_DRAM_SHARD_MAX_OPERANDS];
    /* How many pieces of each operand are already placed in each bank. More members than
     * banks is normal, and two of them at one base would have the second SILENTLY overwrite
     * the first — both would then read a valid piece of the same model from a live endpoint
     * and complete. */
    uint32_t placed[FT_TT_DRAM_SHARD_MAX_OPERANDS][FT_TT_MAX_DRAM_BANKS];
    uint64_t bank_bytes;
    uint32_t bank_count;
    uint32_t stream_count;
    uint32_t member;
    uint32_t operand;
    int rc = check_arguments(grid, operands, operand_count, member_count, out);

    if (rc != FT_TT_OK)
        return rc;

    bank_count = ft_tt_coord_dram_bank_count(grid);
    if (bank_count == 0u || bank_count > FT_TT_MAX_DRAM_BANKS) {
        ft_tt_log_error("dram shard: the grid reports %u DRAM banks", bank_count);
        return FT_TT_EUNAVAIL;
    }
    if (grid->arch == NULL) {
        ft_tt_log_error("dram shard: the grid has no architecture, so a bank has no size");
        return FT_TT_EUNAVAIL;
    }
    bank_bytes = grid->arch->core.dram_bytes_per_bank;

    rc = check_extents(operands, operand_count, member_count, bank_count, bank_bytes);
    if (rc != FT_TT_OK)
        return rc;

    memset(placed, 0, sizeof(placed));
    stream_count = 0u;
    for (member = 0u; member < member_count; member++) {
        for (operand = 0u; operand < operand_count; operand++)
            banks[stream_count++] = bank_of(bank_count, operand_count, operand, member);
    }

    rc = ft_tt_dram_stripe_choose_ports(grid, banks, stream_count, system, doors);
    if (rc != FT_TT_OK)
        return rc;

    memset(out, 0, sizeof(*out));
    out->member_count = member_count;
    out->operand_count = operand_count;

    stream_count = 0u;
    for (member = 0u; member < member_count; member++) {
        for (operand = 0u; operand < operand_count; operand++) {
            ft_tt_dram_shard_placement *placement = &out->placement[operand][member];
            const uint32_t bank = banks[stream_count];
            const uint32_t slot = placed[operand][bank]++;
            const uint64_t address = (uint64_t)operands[operand].bank_base_address +
                                     (uint64_t)slot * operands[operand].member_bytes;

            placement->endpoint = doors[stream_count];
            placement->address = (uint32_t)address;
            placement->bytes = operands[operand].member_bytes;
            placement->source_offset =
                operands[operand].spread == FT_TT_DRAM_SHARD_PARTITIONED
                    ? member * operands[operand].member_bytes
                    : 0u;
            stream_count++;
        }
    }

    summarise(out, doors, stream_count, bank_count);
    return FT_TT_OK;
}

const ft_tt_dram_shard_placement *ft_tt_dram_shard_placement_of(
    const ft_tt_dram_shard_plan *plan, uint32_t operand, uint32_t member)
{
    if (plan == NULL || operand >= plan->operand_count || member >= plan->member_count)
        return NULL;
    return &plan->placement[operand][member];
}
