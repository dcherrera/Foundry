/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_matmul_resident_multicast.h
 * @brief The HOST half of a RESIDENT matmul triad that multicasts its shared A block:
 *   the per-member roles, the runtime-arg tail, and the pre-launch reset of the three
 *   rendezvous words.
 *
 *   Pure arithmetic plus three L1 word writes. It owns no L1 map of its own — the
 *   rendezvous words are `ft_tt_op_matmul_cooperative.h`'s, deliberately, because there
 *   should be ONE answer to "where does a cooperating group keep its semaphores" and the
 *   one-shot kernel already had it.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT IS FOR
 * ---------------------------------------------------------------------------
 * At Mt = 1 the shared activation is **1.76 GB of the 2.70 GB a token reads — 65%** — and
 * every member of a split group reads it from the ONE DRAM bank holding that strip. That
 * is the measured reason four cores buy 3.44x and sixteen buy 4.80x, and why door rotation
 * is worth 1.27x at N=8: a gain that GROWS with N is what an endpoint ceiling looks like.
 *
 * The fix is the leader reading A once and multicasting it, which
 * `cooperative_matmul_reader.c` has done since 2026-08-13 — as a ONE-SHOT program, whose
 * per-op image upload costs 218 µs per member per op, **147 ms a token at N=4** against a
 * 0.109 s token. So it was unreachable in the forward. This module is the same rendezvous
 * expressed as a tail on the RESIDENT queue's runtime args, where it costs nothing per op.
 *
 * ---------------------------------------------------------------------------
 * THREE THINGS THIS MODULE REFUSES, AND WHY EACH IS A REFUSAL
 * ---------------------------------------------------------------------------
 *  1. **A member set that is not a rectangle.** A multicast writes every node of its
 *     bounding box. The backend's default member cores are `{71, 75, 76, 77}` — an
 *     ARITHMETIC run, not a rectangle — whose box also contains 72, 73 and 74, which hold
 *     the reduce, eltwise and sample residencies. Cooperating over that set would write a
 *     matmul's activation block into three other residencies' L1. So a cooperative group
 *     has to be PLACED as a rectangle, and this refuses anything else.
 *  2. **More than four destinations in one packet.** The partition is
 *     ft_tt_group_broadcast_partition.h's, which is where that rule and its evidence live.
 *  3. **A group of one.** No receivers is the zero-destination multicast in different
 *     clothes, and the answer is a refusal rather than a silent fall back to solo.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS MODULE DOES NOT DECIDE
 * ---------------------------------------------------------------------------
 * Whether cooperation is SAFE for a given weight set. That is a property of the whole
 * descriptor stream — cooperation is all-or-nothing for a residency, so every matmul it
 * will execute must split evenly across the group or the members' CB_IN0 write pointers
 * drift apart. The caller knows its weights; this module only knows geometry, and the
 * kernel refuses a descriptor whose cooperate word disagrees with its role.
 *
 * @status Off-hardware except for the three reset writes.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_MATMUL_RESIDENT_MULTICAST_H
#define FT_TT_OP_MATMUL_RESIDENT_MULTICAST_H

#include <stdint.h>

#include "ft_tt_coord.h"
#include "ft_tt_group_broadcast_partition.h"
#include "ft_tt_launch.h"
#include "kernels/resident_matmul_multicast_protocol.h"

/** Members one plan will describe. The partition's own bound, restated so a caller sizing
 *  an array has one number to use. */
#define FT_TT_RESIDENT_MULTICAST_MAX_MEMBERS FT_TT_GROUP_BROADCAST_MAX_MEMBERS

/**
 * Everything a cooperating residency needs that does not change between descriptors.
 *
 * The roles, the geometry and the three L1 addresses are all fixed for the life of a
 * residency, which is why they live in the runtime-arg block and not in the 16-word
 * parameter block: the parameter block is per descriptor, is written on the hot path, and
 * already spells fourteen of its sixteen words.
 */
typedef struct ft_tt_op_matmul_resident_multicast {
    ft_tt_noc_coord member[FT_TT_RESIDENT_MULTICAST_MAX_MEMBERS];
    uint32_t member_count;
    /** Index into `member`, not a coordinate. The leader is a working member and sits
     *  inside the group's bounding box. */
    uint32_t leader;
    ft_tt_group_broadcast_partition partition;
    /** Receivers the leader waits for: `member_count - 1`, and equal to the partition's
     *  destination total. Carried because it is the number the kernel's ready-semaphore
     *  wait and the write barrier's accounting are both built on. */
    uint32_t receiver_count;
    uint32_t ready_semaphore_address;
    uint32_t arrival_semaphore_address;
    uint32_t counter_word_address;
    uint32_t spin_limit;
} ft_tt_op_matmul_resident_multicast;

/**
 * Plan a cooperating residency over `members`, led by `members[leader]`.
 *
 * @param members     TRANSLATED coordinates that EXACTLY fill their own bounding box, in
 *                    row-major order — what ft_tt_group_rectangle_coordinates() produces
 * @param member_count 2..FT_TT_RESIDENT_MULTICAST_MAX_MEMBERS
 * @param leader      index into `members`
 * @param spin_limit  spins one rendezvous wait takes before abandoning; never 0
 * @param out         written only on success
 * @return FT_TT_OK; FT_TT_EINVAL for a NULL argument or a count out of range;
 *         FT_TT_ERROR for a member set that is not a rectangle or a partition that cannot
 *         be built inside the four-destination limit
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_op_matmul_resident_multicast_plan(const ft_tt_noc_coord *members,
                                            uint32_t member_count, uint32_t leader,
                                            uint32_t spin_limit,
                                            ft_tt_op_matmul_resident_multicast *out);

/**
 * Fill member `index`'s runtime-arg tail — args kArgCoopRole..kArgCoopSpinLimit.
 *
 * `out_args` must already hold the queue's own ten words from
 * `ft_tt_queue_kernel_args()`; this writes words 10..22 of the SAME block and leaves
 * 0..9 alone. Only the reader (DM0) reads them, so only that processor's block needs the
 * wider count — the other four keep FT_TT_QUEUE_KERNEL_ARG_COUNT.
 *
 * @return FT_TT_OK, or FT_TT_EINVAL
 */
int ft_tt_op_matmul_resident_multicast_args(const ft_tt_op_matmul_resident_multicast *plan,
                                            uint32_t index,
                                            uint32_t out_args[kArgCoopCount]);

/**
 * Stage the three rendezvous words BEFORE the launch that will use them.
 *
 * Order is load-bearing and is `ft_tt_group_shared_operand_reset()`'s: every member's
 * arrival word first, the leader's counter word next, the leader's ready semaphore LAST. A
 * member whose arrival word still held a previous run's counter would sail through its
 * first wait with no block having arrived, and the leader cannot start until the ready word
 * is clear — so clearing that last closes the window.
 *
 * Called on every ARM, not once at setup: a residency that was retired and re-armed starts
 * its block counter over, and a stale arrival word is exactly the value that would make
 * that undetectable.
 *
 * @return FT_TT_OK, FT_TT_EINVAL, or the first failing write's code
 */
int ft_tt_op_matmul_resident_multicast_reset(const ft_tt_noc_access *noc,
                                             const ft_tt_op_matmul_resident_multicast *plan);

#endif /* FT_TT_OP_MATMUL_RESIDENT_MULTICAST_H */
