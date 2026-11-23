/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_split_broadcast_geometry.h
 * @brief The geometry of "reach N receivers with several SMALL broadcasts instead of one
 *   big one", and every refusal that keeps such a plan safe to issue.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS ITS OWN MODULE
 * ---------------------------------------------------------------------------
 * It is used by tests/test_multicast_bench.c's experiment B, which asks whether the
 * standing "never more than 4 destinations per multicast" rule is really about
 * DESTINATIONS PER PACKET. That experiment issues two broadcasts at once, and the two
 * things that make it safe are arithmetic:
 *
 *  - no rectangle may carry more destinations than the caller allows, which is the entire
 *    claim the arm is making about itself;
 *  - NO SENDER MAY BE INSIDE ANY RECTANGLE, including the other group's. A multicast
 *    writes every node in its bounding box, so a sender caught in a box has its own
 *    staging buffer overwritten mid-round — and the result would be a corrupt payload
 *    blamed on the split rather than on the layout.
 *
 * Arithmetic that protects the card must be provable without the card, so it lives here
 * and is driven red by tests/test_split_broadcast_geometry.c rather than being an `if`
 * inside a program that needs silicon to run at all.
 *
 * @status Complete and off-hardware. Touches no device; the only dependency is the
 *   rectangle arithmetic in ft_tt_multicast.h.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_SPLIT_BROADCAST_GEOMETRY_H
#define FT_TT_SPLIT_BROADCAST_GEOMETRY_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_multicast.h"

/** Most groups one plan may hold. Two is what experiment B needs; the bound exists so the
 *  plan is a value rather than an allocation. */
#define FT_TT_SPLIT_MAX_GROUPS 4u

/** A plan: one rectangle per group, in the order the rows were given. */
typedef struct ft_tt_split_broadcast_plan {
    ft_tt_multicast_rectangle rectangle[FT_TT_SPLIT_MAX_GROUPS];
    uint32_t group_count;
} ft_tt_split_broadcast_plan;

/**
 * Build a plan of `rows` broadcasts, group `g` being row `g` of the receiver block sent by
 * sender `g`, and REFUSE every geometry that is not safe to issue.
 *
 * @param receivers         `rows * columns` TRANSLATED coordinates, ROW MAJOR
 * @param rows              how many groups; 1..FT_TT_SPLIT_MAX_GROUPS
 * @param columns           receivers per group; at least 1
 * @param senders           one TRANSLATED coordinate per group, in the same order
 * @param sender_count      must equal `rows` — one sender per group, no sharing. A shared
 *                          sender would serialise the two broadcasts and the arm would no
 *                          longer be testing what it says it is
 * @param max_destinations  the standing per-packet limit the plan must respect
 * @param out               written only on success
 * @param why               a sentence for the transcript, written on success AND failure
 *                          so a caller can print it unconditionally; never NULL
 * @param why_bytes         its size
 * @return FT_TT_OK; FT_TT_EINVAL for a NULL argument or a count out of range;
 *         FT_TT_ERROR for a row that is not a rectangle, a rectangle over
 *         `max_destinations`, a sender inside any rectangle, or two rectangles that
 *         overlap. The reason is in `why` either way
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (ft_tt_multicast logs on failure)
 */
int ft_tt_split_broadcast_plan_build(const ft_tt_noc_coord *receivers, uint32_t rows,
                                     uint32_t columns, const ft_tt_noc_coord *senders,
                                     uint32_t sender_count, uint32_t max_destinations,
                                     ft_tt_split_broadcast_plan *out, char *why,
                                     size_t why_bytes);

/** Whether two rectangles share any node. Exposed because "disjoint" is the property the
 *  plan turns on, and a caller checking it by hand is a caller getting it wrong. */
bool ft_tt_split_broadcast_rectangles_overlap(const ft_tt_multicast_rectangle *left,
                                              const ft_tt_multicast_rectangle *right);

#endif /* FT_TT_SPLIT_BROADCAST_GEOMETRY_H */
