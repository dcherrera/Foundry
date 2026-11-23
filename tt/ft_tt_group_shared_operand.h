/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_group_shared_operand.h
 * @brief What a core group needs on top of `ft_tt_group.h` before it can express
 *   "these N cores cooperate on ONE matmul": a shared operand source, a barrier, and a
 *   rectangle to hold them.
 *
 * ---------------------------------------------------------------------------
 * WHAT ft_tt_group.h ALREADY DOES, AND THE THREE THINGS IT DOES NOT
 * ---------------------------------------------------------------------------
 * `ft_tt_core_group` already gives a group disjoint CORES, disjoint DRAM BANKS, a queue
 * per core, and group-level COMPLETION ("did every member reach item N, with this run's
 * token"). That is enough for N cores doing N INDEPENDENT things. It is not enough for N
 * cores doing ONE thing, and the gap is exactly three items:
 *
 *  1. **A group is not a rectangle, and a multicast needs one.**
 *     `ft_tt_group_partition()` hands out CONSECUTIVE live Tensix indices, and
 *     `ft_tt_coord_tensix_by_index()`'s stride is the live column count — so eight
 *     consecutive indices routinely wrap a row, and the bounding box of a wrapped run
 *     contains cores the group does not own. A multicast writes every node in the box.
 *     `ft_tt_group_rectangle_coordinates()` is the partitioner that cannot do that.
 *     It is a NEW entry point rather than a change to `ft_tt_group_partition()`, whose
 *     "consecutive indices, remainder to the first groups" contract is right for the
 *     independent-workload case and is depended on by tests/test_queue.c's synthetic
 *     12-column grid.
 *
 *  2. **A shared OPERAND SOURCE.** A group has no notion of a member's ROLE. Sharing an A
 *     tile means one member READS it (from DRAM, once) and every other member RECEIVES it,
 *     at the SAME L1 address on all of them — the hardware requires the address to match,
 *     and nothing in `ft_tt_core_group` says a per-core address must agree across cores.
 *     `ft_tt_group_shared_operand` is that contract: who the leader is, where the operand
 *     lands, and how big it is.
 *
 *  3. **A BARRIER, which is not the completion block.** Group completion answers "has every
 *     member finished item N" and it is read by the HOST, after the fact, over PCIe. A
 *     cooperative matmul needs a member to answer "is every member ready for me to
 *     overwrite the shared buffer" ON THE CARD, per K-block, in under a microsecond. Those
 *     are different questions with different readers and the second one does not exist yet.
 *     It is the two-semaphore rendezvous below — a ready semaphore on the leader that
 *     receivers atomically increment, and an arrival semaphore on every member that the
 *     leader multicasts VALID into. The protocol is in
 *     kernels/ft_tensix_noc_multicast.h's header block.
 *
 * ---------------------------------------------------------------------------
 * AND ONE THING THAT ALREADY WORKS AND SHOULD NOT BE CHANGED
 * ---------------------------------------------------------------------------
 * The COMPLETION RULE. `ft_tt_group_check_completion()`'s "every member's own slot reached
 * sequence N with this run's token" is already the right shape for a cooperative op: the
 * members produce different output tiles of one logical result, and "the group finished
 * output block N" is exactly per-slot agreement. A sum would not be — one core doing 2N
 * items and another doing none sums to the same 2N (ft_tt_group.h). The one thing worth
 * saying out loud is that ABANDONMENT stays visible: a member whose wait gave up does not
 * post, so the group wait times out and `out_reached` names the lowest sequence anyone got
 * to. That is the diagnosis you want and it needs no change.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS FILE DELIBERATELY DOES NOT DO
 * ---------------------------------------------------------------------------
 * It does not allocate L1. The four addresses come from the caller, because a resident op
 * owns its own L1 map (ft_tt_op_l1_layout.h) and a second allocator would be a second
 * source of truth for the same bytes. What this file does is REFUSE a layout that cannot
 * work: overlapping regions, a mis-aligned operand, a receiver set that is not a
 * rectangle.
 *
 * @status Off-hardware in tests/test_multicast_rectangle.c; exercised on the card by
 *   tests/test_multicast_bench.c.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_GROUP_SHARED_OPERAND_H
#define FT_TT_GROUP_SHARED_OPERAND_H

#include <stdint.h>

#include "ft_tt_coord.h"
#include "ft_tt_launch.h"
#include "ft_tt_multicast.h"

/** The two values the arrival semaphore takes. The vendor's semaphore values, mirrored in
 *  kernels/ft_tensix_noc_multicast.h as FT_TENSIX_SEMAPHORE_*; the three must agree. */
#define FT_TT_GROUP_SEMAPHORE_INVALID 0u
#define FT_TT_GROUP_SEMAPHORE_VALID 1u

/** A semaphore is one 4-byte word, and the NoC's register-class alignment is 4 bytes
 *  (NOC_REG_ALIGNMENT_BYTES). */
#define FT_TT_GROUP_SEMAPHORE_BYTES 4u

/**
 * Where the shared-operand machinery lives in each member's L1. The caller supplies it;
 * this module only checks it.
 *
 * Every address is the SAME on every member. That is a hardware requirement for the
 * operand (a multicast writes one address on all destinations) and a deliberate choice
 * for the semaphores, so that one runtime-arg block serves every member and a per-member
 * address cannot drift.
 */
typedef struct ft_tt_group_shared_operand_layout {
    /** Where the shared operand lands on every member. 16-byte aligned. */
    uint32_t operand_address;
    /** How big it is. Non-zero and a multiple of 16 — a zero here is the zero-length NoC
     *  transfer and is refused, not treated as "nothing to send". */
    uint32_t operand_bytes;
    /** On the LEADER: the word receivers atomically increment to say they are ready. */
    uint32_t ready_semaphore_address;
    /** On EVERY member: the word the leader multicasts VALID into when the operand has
     *  landed. */
    uint32_t arrival_semaphore_address;
    /** On the LEADER: a word holding FT_TT_GROUP_SEMAPHORE_VALID, which is what the
     *  leader multicasts FROM. It is a staged constant rather than an immediate because a
     *  semaphore multicast is a WRITE of an L1 word, not an atomic. */
    uint32_t valid_word_address;
} ft_tt_group_shared_operand_layout;

/** A group's shared-operand contract: the rectangle, the leader, and the layout. */
typedef struct ft_tt_group_shared_operand {
    ft_tt_multicast_rectangle rectangle;
    ft_tt_group_shared_operand_layout layout;
    /** TRANSLATED coordinate of the core that reads the operand and multicasts it. */
    ft_tt_noc_coord leader;
    /** How many cores RECEIVE the operand. Equals rectangle.destination_count. */
    uint32_t receiver_count;
    /** How many cores the cooperating group has, leader included — `receiver_count` when
     *  the leader is one of the receivers' rectangle, one more when it is a dedicated
     *  feeder outside it. */
    uint32_t member_count;
} ft_tt_group_shared_operand;

/**
 * Derive `rows * columns` TRANSLATED Tensix coordinates forming a RECTANGLE, starting at
 * live index `first_index`.
 *
 * This is the partitioner `ft_tt_group_partition()` is not: it refuses a block that would
 * wrap a row instead of silently handing back a set whose bounding box contains cores it
 * does not own.
 *
 * Every size comes from the LIVE grid — there is no core, row or column count in this
 * module — because "14x10" is exactly the constant that feels safe while partitioning a
 * grid and is wrong on the 120-core p150s other people run (ft_tt_group.h).
 *
 * @param grid         from ft_tt_arc_telemetry_read_grid(); never a constructed one
 * @param first_index  live Tensix index of the rectangle's first (lowest-index) core
 * @param rows         at least 1
 * @param columns      at least 1
 * @param out_members  receives `rows * columns` coordinates in live-index order
 * @param capacity     how many entries `out_members` has
 * @return FT_TT_OK;
 *         FT_TT_EINVAL for a NULL argument or a zero dimension;
 *         FT_TT_ENOMEM when `capacity` is smaller than `rows * columns`;
 *         FT_TT_ERROR when the block would wrap a row or leave the live grid — with the
 *         column count and the offending index in the log, because "why not" is the
 *         question a caller then has
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_group_rectangle_coordinates(const ft_tt_grid_info *grid, uint32_t first_index,
                                      uint32_t rows, uint32_t columns,
                                      ft_tt_noc_coord *out_members, uint32_t capacity);

/**
 * Bind the RECEIVERS and an L1 layout into a shared-operand contract, refusing every shape
 * that cannot work.
 *
 * **The leader is passed separately from the receivers on purpose.** The rectangle is what
 * the hardware broadcasts into, and the leader may be either inside it (the natural
 * cooperating-matmul layout: N cores, one of them also does the DRAM read, so
 * `destination_count` is N-1 because a non-loopback multicast does not write to self) or
 * outside it (a dedicated feeder core, `destination_count` is N). Folding the leader into
 * the member array would have made the first shape the only expressible one, and the
 * second is exactly what tests/test_multicast_bench.c needs so that both arms of its A/B
 * run on the SAME set of cores.
 *
 * Refuses: receivers that are not a rectangle (ft_tt_multicast_rectangle_from_cores());
 * a rectangle that leaves zero destinations once the leader is excluded; an operand that is
 * zero-length, mis-aligned or outside L1; a semaphore that is not 4-byte aligned; and ANY
 * overlap between the operand region, the two semaphores and the valid word — because the
 * leader multicasts into the arrival semaphore and into the operand region in the same
 * round, and an overlap makes one of them silently destroy the other.
 *
 * @param receivers       TRANSLATED coordinates, from ft_tt_coord.h against live telemetry
 * @param receiver_count  how many; at least 1
 * @param leader          the core that reads the operand and multicasts it
 * @param out             written only on success
 * @return FT_TT_OK, FT_TT_EINVAL, or FT_TT_ERROR (the reason is logged)
 */
int ft_tt_group_shared_operand_bind(const ft_tt_noc_coord *receivers, uint32_t receiver_count,
                                    ft_tt_noc_coord leader,
                                    const ft_tt_group_shared_operand_layout *layout,
                                    ft_tt_group_shared_operand *out);

/**
 * Put the barrier into its start-of-run state on the card: the leader's ready semaphore
 * to 0, every member's arrival semaphore to INVALID, and the leader's valid word to VALID.
 *
 * Call it before every run, not once at setup. A ready semaphore left non-zero by an
 * abandoned run lets the next leader believe receivers are ready that have not booted —
 * and unlike the completion block, whose zero is not a sequence any descriptor may carry,
 * a stale semaphore value is INDISTINGUISHABLE from a fresh one. That is why this is a
 * reset rather than an init.
 *
 * @param noc             an accessor over a claimed window
 * @param receivers       the same array passed to ft_tt_group_shared_operand_bind()
 * @param receiver_count  the same count
 * @return FT_TT_OK, or the first failing write's code (logged)
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_group_shared_operand_reset(const ft_tt_noc_access *noc,
                                     const ft_tt_group_shared_operand *operand,
                                     const ft_tt_noc_coord *receivers,
                                     uint32_t receiver_count);

#endif /* FT_TT_GROUP_SHARED_OPERAND_H */
