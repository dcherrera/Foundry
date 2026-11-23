/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_multicast.h
 * @brief The HOST half of NoC multicast: turn a set of cores into a multicast rectangle,
 *   refuse the ones the hardware would mis-deliver, and hand the kernel the numbers it
 *   cannot derive for itself.
 *
 *   Pure arithmetic. Nothing here touches the device — which is the point, because every
 *   claim below is then provable off-hardware, and a wrong multicast rectangle is not a
 *   failed transaction: it is a write into a core somebody else is using, or a coordinate
 *   that hangs the NoC (spec §7.3, DEV_NOTES.md §1).
 *
 *   Pair with kernels/ft_tt_multicast_kernel.h, which is the device half. The 64-bit
 *   encoding exists in BOTH, deliberately: the kernel needs it to issue and the host needs
 *   it to check, and `tests/test_multicast_rectangle.c` asserts the two agree bit for bit
 *   against literal expected values rather than against each other.
 *
 * ---------------------------------------------------------------------------
 * WHAT A MULTICAST RECTANGLE IS, AND THE THREE WAYS TO GET IT WRONG
 * ---------------------------------------------------------------------------
 * A Blackhole multicast write names TWO CORNERS and an address, and every node in the
 * closed box between them receives the payload at that address. Three consequences, each
 * of which this module refuses rather than documents:
 *
 *  1. **A set of cores is not a rectangle.** `ft_tt_group_partition()` hands out
 *     CONSECUTIVE live Tensix indices, and `ft_tt_coord_tensix_by_index()`'s stride is the
 *     live column count — so eight consecutive indices wrap a row and their bounding box
 *     contains cores that are not in the set. Multicasting to that box writes into a core
 *     another group owns. `ft_tt_multicast_rectangle_from_cores()` refuses it, by counting:
 *     a set of DISTINCT coordinates whose cardinality equals its bounding box's cardinality
 *     IS that box, by pigeonhole, and needs no grid to check.
 *
 *  2. **The destination count is barrier arithmetic, not a comment.** The kernel's write
 *     barrier compares the NIU's acknowledgement counter against a software counter
 *     advanced by `destination_count` per multicast. Too many and the barrier never
 *     returns; too few and it returns while writes are still landing in L1 the kernel is
 *     about to reuse. The count EXCLUDES the sender even when the sender is inside the
 *     rectangle, because a non-loopback multicast does not write to self.
 *
 *  3. **A rectangle that degenerates to the sender alone has ZERO destinations**, and a
 *     zero-destination multicast is the zero-length NoC transfer wearing different clothes
 *     — the shape that took this card from working to `Mem- BusMaster-` on 2026-08-12.
 *     It is refused here AND again in the kernel at the point of issue,
 *     because a host-side check cannot see a kernel-issued transfer.
 *
 * ---------------------------------------------------------------------------
 * THE CORNER ORDER IS PER-NoC
 * ---------------------------------------------------------------------------
 * The encoding has a "start" corner and an "end" corner and they are NOT simply the low
 * and high corners: NoC 1 broadcasts in the opposite direction and takes them reversed
 * (NoC 1 multicasts from bottom left to top right, so the coordinates are reversed). This
 * module takes LOW and HIGH — unambiguous, checkable — and applies the swap from the NoC
 * index, so a caller cannot get it silently wrong. Reversed corners are not a rejected
 * transaction; they are a rectangle covering the wrong nodes.
 *
 * @status Off-hardware in tests/test_multicast_rectangle.c; exercised on the card by
 *   tests/test_multicast_bench.c.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_MULTICAST_H
#define FT_TT_MULTICAST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_coord.h"

/** Largest coordinate the TLB's 6-bit x_end/y_end fields and the NoC's 6-bit node-id field
 *  can carry. A coordinate above this does not fail — it TRUNCATES onto a different live
 *  endpoint (ft_tt_noc.h says the same thing about a unicast target). */
#define FT_TT_MULTICAST_MAX_COORDINATE 63u

/** A multicast destination is always Tensix L1 — the hardware has no other legal target
 *  for a broadcast packet — so L1's 16 bytes is the whole alignment rule and DRAM's 64
 *  never applies. */
#define FT_TT_MULTICAST_ALIGNMENT_BYTES 16u

/** The rectangle plus the two counts a kernel needs. Build it with
 *  ft_tt_multicast_rectangle_from_cores(); the fields are read-only after that. */
typedef struct ft_tt_multicast_rectangle {
    uint32_t x_low;  /**< inclusive */
    uint32_t y_low;  /**< inclusive */
    uint32_t x_high; /**< inclusive */
    uint32_t y_high; /**< inclusive */
    /** Nodes inside the box: (x_high - x_low + 1) * (y_high - y_low + 1). */
    uint32_t node_count;
    /** What the kernel passes as `destination_count`: `node_count` minus the sender when
     *  the sender is inside the box. NEVER zero in a rectangle this module returned. */
    uint32_t destination_count;
    /** Whether the sender is one of the nodes in the box. Recorded because it is the whole
     *  reason `destination_count` and `node_count` differ, and a reader who does not know
     *  it cannot check either. */
    bool sender_inside;
} ft_tt_multicast_rectangle;

/**
 * Bounding rectangle of `members`, with the counts a multicast needs, or a refusal.
 *
 * @param members  TRANSLATED coordinates, from ft_tt_coord.h against live telemetry. Never
 *                 constructed, never guessed — a wrong endpoint hangs the chip
 * @param count    how many; at least 1
 * @param sender   the core that will issue the multicast. It MAY be inside the rectangle
 *                 (it just does not receive) and it may be outside it
 * @param out      written only on success
 * @return FT_TT_OK;
 *         FT_TT_EINVAL for a NULL argument, a zero count, a coordinate above
 *         FT_TT_MULTICAST_MAX_COORDINATE, or duplicate members;
 *         FT_TT_ERROR when the members do not FILL their bounding box (the set is not a
 *         rectangle) or when the rectangle leaves zero destinations once the sender is
 *         excluded. Both reasons are logged.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_multicast_rectangle_from_cores(const ft_tt_noc_coord *members, uint32_t count,
                                         ft_tt_noc_coord sender,
                                         ft_tt_multicast_rectangle *out);

/**
 * Build a rectangle directly from two corners, for a caller that already knows the shape
 * (a group partitioned as a rectangle, say) and has no member array to hand.
 *
 * Same refusals as above minus the fill check, which has no meaning without members.
 *
 * @return FT_TT_OK, FT_TT_EINVAL, or FT_TT_ERROR for zero destinations
 */
int ft_tt_multicast_rectangle_from_corners(uint32_t x_low, uint32_t y_low, uint32_t x_high,
                                           uint32_t y_high, ft_tt_noc_coord sender,
                                           ft_tt_multicast_rectangle *out);

/** Whether `coord` is one of the rectangle's nodes. */
bool ft_tt_multicast_rectangle_contains(const ft_tt_multicast_rectangle *rectangle,
                                        ft_tt_noc_coord coord);

/**
 * Re-check a rectangle a caller may have built or edited by hand.
 *
 * Separate from the constructors so that a struct arriving from anywhere can be validated
 * on the way into a launch, which is the last point before it becomes NoC traffic.
 *
 * @return FT_TT_OK, FT_TT_EINVAL, or FT_TT_ERROR (the reason is logged)
 */
int ft_tt_multicast_check_rectangle(const ft_tt_multicast_rectangle *rectangle);

/**
 * Whether a multicast PAYLOAD is legal: non-zero, 16-byte aligned in both address and
 * length, and inside L1.
 *
 * This is the host-side statement of a rule the kernel also enforces at the point of
 * issue. Both are needed and neither replaces the other:
 * `ft_tt_noc_check_transfer()` runs host-side and cannot see a kernel-issued transfer,
 * which is exactly how `noc_inline_dw_write` to DRAM cost two cores.
 *
 * @param l1_address  the SAME address on every receiver; the hardware requires that
 * @param bytes       payload length; ZERO IS A REFUSAL, not a no-op
 * @return FT_TT_OK, or FT_TT_EINVAL (the reason is logged)
 */
int ft_tt_multicast_check_payload(uint32_t l1_address, size_t bytes);

/**
 * The 64-bit multicast NoC address a kernel would issue against, for a host that wants to
 * check the encoding rather than take it on trust.
 *
 * @param noc_index  0 or 1. The corner order DIFFERS between them — see the header block
 * @param out        written only on success
 * @return FT_TT_OK, or FT_TT_EINVAL
 */
int ft_tt_multicast_encode_address(const ft_tt_multicast_rectangle *rectangle,
                                   uint32_t noc_index, uint32_t l1_address,
                                   uint64_t *out_address);

#endif /* FT_TT_MULTICAST_H */
