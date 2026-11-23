/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_group_shared_operand.c
 * @brief Implementation of ft_tt_group_shared_operand.h — see that header for what each
 *   refusal is protecting against.
 */

#include "ft_tt_group_shared_operand.h"

#include <stddef.h>

#include "ft_tt_device.h"

/** Two spans overlap if neither ends before the other begins. Used to keep the operand,
 *  the two semaphores and the valid word from destroying each other — the leader writes
 *  into two of them in the same round. */
static bool spans_overlap(uint32_t first_start, uint32_t first_bytes, uint32_t second_start,
                          uint32_t second_bytes)
{
    return first_start < second_start + second_bytes &&
           second_start < first_start + first_bytes;
}

static int check_layout(const ft_tt_group_shared_operand_layout *layout)
{
    int rc = ft_tt_multicast_check_payload(layout->operand_address, layout->operand_bytes);

    if (rc != FT_TT_OK)
        return rc;

    if ((layout->ready_semaphore_address % FT_TT_GROUP_SEMAPHORE_BYTES) != 0u ||
        (layout->arrival_semaphore_address % FT_TT_GROUP_SEMAPHORE_BYTES) != 0u ||
        (layout->valid_word_address % FT_TT_GROUP_SEMAPHORE_BYTES) != 0u) {
        ft_tt_log_error("shared operand: a semaphore address is not %u-byte aligned "
                        "(ready 0x%08x, arrival 0x%08x, valid 0x%08x)",
                        FT_TT_GROUP_SEMAPHORE_BYTES, layout->ready_semaphore_address,
                        layout->arrival_semaphore_address, layout->valid_word_address);
        return FT_TT_EINVAL;
    }

    if (spans_overlap(layout->operand_address, layout->operand_bytes,
                      layout->arrival_semaphore_address, FT_TT_GROUP_SEMAPHORE_BYTES)) {
        ft_tt_log_error("shared operand: the arrival semaphore at 0x%08x is inside the "
                        "operand region 0x%08x+%u. The leader multicasts into BOTH in the "
                        "same round, so one would silently destroy the other",
                        layout->arrival_semaphore_address, layout->operand_address,
                        layout->operand_bytes);
        return FT_TT_ERROR;
    }
    if (spans_overlap(layout->operand_address, layout->operand_bytes,
                      layout->ready_semaphore_address, FT_TT_GROUP_SEMAPHORE_BYTES) ||
        spans_overlap(layout->operand_address, layout->operand_bytes,
                      layout->valid_word_address, FT_TT_GROUP_SEMAPHORE_BYTES)) {
        ft_tt_log_error("shared operand: the ready semaphore (0x%08x) or the valid word "
                        "(0x%08x) is inside the operand region 0x%08x+%u on the leader",
                        layout->ready_semaphore_address, layout->valid_word_address,
                        layout->operand_address, layout->operand_bytes);
        return FT_TT_ERROR;
    }
    if (layout->ready_semaphore_address == layout->arrival_semaphore_address ||
        layout->ready_semaphore_address == layout->valid_word_address ||
        layout->arrival_semaphore_address == layout->valid_word_address) {
        ft_tt_log_error("shared operand: two of the three control words share an address "
                        "(ready 0x%08x, arrival 0x%08x, valid 0x%08x). On the LEADER all "
                        "three are in one L1, so an alias is a deadlock, not a saving",
                        layout->ready_semaphore_address, layout->arrival_semaphore_address,
                        layout->valid_word_address);
        return FT_TT_ERROR;
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */

int ft_tt_group_rectangle_coordinates(const ft_tt_grid_info *grid, uint32_t first_index,
                                      uint32_t rows, uint32_t columns,
                                      ft_tt_noc_coord *out_members, uint32_t capacity)
{
    uint32_t column_count;
    uint32_t core_count;
    uint32_t first_column;
    uint32_t needed;
    uint32_t row;
    uint32_t column;

    if (grid == NULL || out_members == NULL || rows == 0u || columns == 0u) {
        ft_tt_log_error("shared operand: NULL grid/out or a zero rectangle dimension");
        return FT_TT_EINVAL;
    }
    needed = rows * columns;
    if (capacity < needed) {
        ft_tt_log_error("shared operand: %u entries of storage for a %ux%u rectangle",
                        capacity, rows, columns);
        return FT_TT_ENOMEM;
    }

    /* Everything below is derived from the LIVE grid. A hardcoded 14 would not fail on a
     * 12-column part; it would address cores that are not there. */
    column_count = ft_tt_coord_tensix_column_count(grid);
    core_count = ft_tt_coord_tensix_core_count(grid);
    if (column_count == 0u || core_count == 0u) {
        ft_tt_log_error("shared operand: the live grid reports %u columns and %u cores",
                        column_count, core_count);
        return FT_TT_EINVAL;
    }

    first_column = first_index % column_count;
    if (first_column + columns > column_count) {
        ft_tt_log_error("shared operand: index %u is column %u of %u, so a %u-wide block "
                        "WRAPS the row. Consecutive live indices are not a rectangle, and a "
                        "multicast writes every node in the bounding box — including the "
                        "cores the wrap picked up",
                        first_index, first_column, column_count, columns);
        return FT_TT_ERROR;
    }
    if (first_index + (rows - 1u) * column_count + (columns - 1u) >= core_count) {
        ft_tt_log_error("shared operand: a %ux%u block at index %u leaves the live grid of "
                        "%u cores", rows, columns, first_index, core_count);
        return FT_TT_ERROR;
    }

    for (row = 0u; row < rows; row++) {
        for (column = 0u; column < columns; column++) {
            uint32_t index = first_index + row * column_count + column;
            int rc = ft_tt_coord_tensix_by_index(grid, index, FT_TT_COORD_TRANSLATED,
                                                 &out_members[row * columns + column]);

            if (rc != FT_TT_OK) {
                ft_tt_log_error("shared operand: live index %u has no coordinate (rc=%d %s)",
                                index, rc, ft_tt_error_string(rc));
                return rc;
            }
        }
    }

    /* A block that is a rectangle in LIVE-INDEX space is not automatically one in NoC
     * space, and on Blackhole it usually is not: the Tensix columns are x = 1..7 and
     * 10..16, because x = 8 and x = 9 carry the ARC/L2CPU spine and the east DRAM column
     * (ft_tt_arch_blackhole.c, blackhole_tensix_noc_x_by_column). A rectangle spanning
     * grid columns 6 and 7 therefore has a bounding box that INCLUDES those two — and a
     * multicast writes every node in the box, which would put a payload into the ARC spine
     * and a DRAM column. Nothing about that fails; it is a write to a live endpoint of the
     * wrong type, which is the family that hangs the NoC.
     *
     * MEASURED on the card 2026-08-13: asking for indices 100..111, one row of a 14-column
     * card, produced x 3..7 and 10..16 with x 8 and 9 missing, and the fill check refused
     * it. That is what this check is, hoisted to the point where the block is chosen. */
    {
        uint32_t span_x = out_members[columns - 1u].x - out_members[0].x + 1u;

        if (span_x != columns) {
            ft_tt_log_error("shared operand: grid columns %u..%u are NoC x %u..%u, which "
                            "spans %u nodes for %u cores — the block crosses the "
                            "non-Tensix column gap (x = 8, 9 carry the ARC/L2CPU spine and "
                            "the east DRAM column). A multicast writes EVERY node in its "
                            "bounding box, so this rectangle cannot be used",
                            first_column, first_column + columns - 1u, out_members[0].x,
                            out_members[columns - 1u].x, span_x, columns);
            return FT_TT_ERROR;
        }
    }
    return FT_TT_OK;
}

int ft_tt_group_shared_operand_bind(const ft_tt_noc_coord *receivers, uint32_t receiver_count,
                                    ft_tt_noc_coord leader,
                                    const ft_tt_group_shared_operand_layout *layout,
                                    ft_tt_group_shared_operand *out)
{
    ft_tt_group_shared_operand operand;
    int rc;

    if (receivers == NULL || layout == NULL || out == NULL) {
        ft_tt_log_error("shared operand: NULL receivers/layout/out");
        return FT_TT_EINVAL;
    }
    if (receiver_count == 0u) {
        ft_tt_log_error("shared operand: zero receivers. A multicast to nobody is the "
                        "zero-length NoC transfer in different clothes (DEV_NOTES.md §1), "
                        "not a degenerate case to optimise into a local copy");
        return FT_TT_EINVAL;
    }

    rc = check_layout(layout);
    if (rc != FT_TT_OK)
        return rc;

    operand.layout = *layout;
    operand.leader = leader;
    rc = ft_tt_multicast_rectangle_from_cores(receivers, receiver_count, leader,
                                              &operand.rectangle);
    if (rc != FT_TT_OK)
        return rc;

    /* The rectangle already excluded the leader from the destination count if it is inside;
     * that same fact is what decides whether the cooperating group is N cores or N+1. */
    operand.receiver_count = operand.rectangle.destination_count;
    operand.member_count =
        operand.rectangle.sender_inside ? receiver_count : receiver_count + 1u;

    *out = operand;
    return FT_TT_OK;
}

int ft_tt_group_shared_operand_reset(const ft_tt_noc_access *noc,
                                     const ft_tt_group_shared_operand *operand,
                                     const ft_tt_noc_coord *receivers, uint32_t receiver_count)
{
    const uint32_t zero = 0u;
    const uint32_t valid = FT_TT_GROUP_SEMAPHORE_VALID;
    const uint32_t invalid = FT_TT_GROUP_SEMAPHORE_INVALID;
    uint32_t index;
    int rc;

    if (noc == NULL || noc->write == NULL || operand == NULL || receivers == NULL) {
        ft_tt_log_error("shared operand: NULL accessor/operand/receivers");
        return FT_TT_EINVAL;
    }
    if (receiver_count == 0u) {
        ft_tt_log_error("shared operand: reset given zero receivers");
        return FT_TT_EINVAL;
    }

    /* Every member's arrival semaphore FIRST, the leader's ready semaphore last. A member
     * whose arrival word still reads VALID from a previous run would sail through its
     * first wait without any operand having arrived — and the leader cannot start until
     * the ready word is cleared, so clearing that last closes the window. */
    for (index = 0u; index < receiver_count; index++) {
        rc = noc->write(noc->user, (uint16_t)receivers[index].x, (uint16_t)receivers[index].y,
                        operand->layout.arrival_semaphore_address, &invalid, sizeof(invalid));
        if (rc != FT_TT_OK) {
            ft_tt_log_error("shared operand: clearing the arrival semaphore on receiver %u "
                            "at (%u,%u) failed: rc=%d (%s)", index, receivers[index].x,
                            receivers[index].y, rc, ft_tt_error_string(rc));
            return rc;
        }
    }

    rc = noc->write(noc->user, (uint16_t)operand->leader.x, (uint16_t)operand->leader.y,
                    operand->layout.valid_word_address, &valid, sizeof(valid));
    if (rc == FT_TT_OK)
        rc = noc->write(noc->user, (uint16_t)operand->leader.x, (uint16_t)operand->leader.y,
                        operand->layout.ready_semaphore_address, &zero, sizeof(zero));
    if (rc != FT_TT_OK)
        ft_tt_log_error("shared operand: staging the leader's control words at (%u,%u) "
                        "failed: rc=%d (%s)", operand->leader.x, operand->leader.y, rc,
                        ft_tt_error_string(rc));
    return rc;
}
