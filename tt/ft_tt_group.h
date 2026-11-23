/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_group.h
 * @brief Core groups — the SubDevices equivalent. Disjoint sets of Tensix cores, each
 *   with its own allocator and its own completion tracking, so two workloads run
 *   concurrently on one card without either one being able to see the other's state.
 *
 * ---------------------------------------------------------------------------
 * DERIVE FROM LIVE — an acceptance criterion, not a style note
 * ---------------------------------------------------------------------------
 * There is no core, column or row count in this file. Every size comes from
 * `ft_tt_coord_tensix_core_count()` against a grid read off ARC telemetry. "14x10" is
 * exactly the constant that feels safe while partitioning a grid and is wrong on the
 * 120-core p150s other people run — TT confirmed the 140→120 reduction is a
 * COMPATIBILITY measure, not yield binning, so those parts are a shipping configuration
 * and not damaged silicon. A hardcoded 140 does not fail there; it addresses cores that
 * are not present, silently.
 *
 * `tests/test_queue.c` partitions a SYNTHETIC 12-column grid into 120 cores with no code
 * change and asserts the shape, which is where a constant creeping back in goes red.
 *
 * Note also that `ft_tt_coord_tensix_by_index()`'s stride IS the live column count, so a
 * core index recorded on this card does not name the same core on another. Indices are
 * for reproducing a run on ONE part; coordinates are what a group actually holds.
 *
 * ---------------------------------------------------------------------------
 * WHAT "INDEPENDENT ALLOCATORS" MEANS HERE
 * ---------------------------------------------------------------------------
 * Two things, and only one of them needed building:
 *
 *  - **L1 is free.** Groups own disjoint CORES, and L1 is per-core, so two groups cannot
 *    collide in L1 at all. Every group uses the same per-core map (ft_tt_op_l1_layout.h),
 *    which is what lets one kernel binary serve every group.
 *  - **DRAM is by BANK.** Each group is handed a disjoint stride through the LIVE bank
 *    count and its own `ft_tt_allocator`. DRAM banks are already separate arenas inside
 *    the allocator, so disjoint banks is disjoint memory with nothing to enforce at
 *    allocation time — a group physically cannot hand out another group's bytes.
 *
 * @status Green. Off-hardware in tests/test_queue.c, including the synthetic 12-column
 *   grid that must partition into 120 cores with no code change; on the card in
 *   tests/test_phase5_resident.c, two groups running independent workloads concurrently,
 *   and in tests/test_phase6_triad.c for group-level completion — two resident triads, a
 *   sequence number their WRITERS post, and the check refusing while one member is behind.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_GROUP_H
#define FT_TT_GROUP_H

#include <stdint.h>

#include "ft_tt_buffer.h"
#include "ft_tt_coord.h"
#include "ft_tt_launch.h"
#include "ft_tt_queue.h"

/**
 * One group: a disjoint set of cores, their queues, and the memory they may use.
 *
 * `queues` is parallel to the core set — `queues[i]` belongs to
 * `ft_tt_launch_core_set_at(&group->cores, i)`. Both point into caller storage carved by
 * ft_tt_group_partition().
 */
typedef struct ft_tt_core_group {
    uint32_t id;                  /**< 0..group_count-1, for logs */
    uint32_t first_tensix_index;  /**< the live-grid index this group starts at */
    uint32_t tensix_index_count;  /**< how many consecutive indices it holds */
    ft_tt_launch_core_set cores;  /**< borrowed storage */
    ft_tt_queue *queues;          /**< borrowed storage, `tensix_index_count` entries */
    ft_tt_allocator allocator;    /**< this group's own DRAM arenas */
    uint32_t dram_bank_first;     /**< first LIVE bank this group may allocate from */
    uint32_t dram_bank_stride;    /**< and the stride between the ones it owns */
    uint32_t dram_bank_count;     /**< how many banks that comes to */
    /** L1 address of the group completion block ON THE GROUP'S FIRST CORE, or 0 if this
     *  group does not track group-level completion. Set by
     *  ft_tt_group_completion_init(). */
    uint32_t completion_address;
    /** Which FT_TT_PROC_* are resident on every core of this group, as a bit mask. 0 means
     *  "ask the control block", which is right for the BRISC-only kernel and WRONG for a
     *  triad: `state` there has exactly one writer, so a reader that has exited while the
     *  compute thread is still spinning out a bound reads EXIT and is a lie. A shutdown
     *  that believed it would relaunch a second kernel onto a core that still has one. */
    uint32_t resident_processor_mask;
} ft_tt_core_group;

/**
 * Split a run of live Tensix indices into `group_count` disjoint groups.
 *
 * Sizes are derived: `index_count / group_count`, with the remainder given one extra core
 * each to the first groups, so a 120-core part and a 140-core part both partition without
 * a code change and neither leaves cores unassigned.
 *
 * Coordinates come from `ft_tt_launch_core_set_add_tensix()`, which derives them from the
 * live grid and CLASSIFIES each one before it can enter a launch — a wrong NoC endpoint
 * is a hung chip and a physical power cycle, not a failed call (spec §7.3).
 *
 * @param grid          from ft_tt_arc_telemetry_read_grid(); never a constructed one
 * @param first_index   first live Tensix index to hand out
 * @param index_count   how many, or 0 for "everything from `first_index` to the end of
 *                      the live grid"
 * @param group_count   at least 1, at most `index_count`
 * @param groups        `group_count` caller-owned structs, zeroed and filled
 * @param placements    caller-owned, at least `index_count` entries; carved among groups
 * @param queues        caller-owned, at least `index_count` entries; carved among groups
 * @param storage_count how many entries `placements` and `queues` have
 * @return              FT_TT_OK; FT_TT_EINVAL on a bad argument; FT_TT_ENOMEM if the
 *                      storage is smaller than `index_count`; or a coordinate error
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_group_partition(const ft_tt_grid_info *grid, uint32_t first_index, uint32_t index_count,
                          uint32_t group_count, ft_tt_core_group *groups,
                          ft_tt_launch_core_placement *placements, ft_tt_queue *queues,
                          uint32_t storage_count);

/**
 * Give each group its own allocator over its own DRAM banks.
 *
 * Separate from partitioning because a caller that only wants cores should not have to
 * supply a device context, and because the allocator refuses a grid it considers
 * unusable — which is a different failure from "these indices do not exist".
 *
 * @param ctx  may be NULL; only used for ft_tt_mem_info accounting
 * @return     FT_TT_OK, or the first failing group's code
 */
int ft_tt_group_init_allocators(ft_tt_core_group *groups, uint32_t group_count,
                                const ft_tt_grid_info *grid, ft_tt_ctx *ctx);

/**
 * The `index`-th DRAM bank this group owns, in LIVE bank numbering.
 *
 * @param out_bank  written only on success
 * @return          FT_TT_OK, or FT_TT_EINVAL when the group owns fewer banks than that
 */
int ft_tt_group_dram_bank(const ft_tt_core_group *group, uint32_t index, uint32_t *out_bank);

/**
 * Check that no coordinate appears in two groups, and that no group is empty.
 *
 * This is the core-level form of the pairwise-distinctness check that caught the
 * multicore seed collision: when N units are supposed to be doing N different things,
 * check them against EACH OTHER and not only against a model. A partitioner that
 * accidentally hands two groups the same core produces two callers that both believe
 * they own it, and the symptom is a wrong answer somewhere else entirely.
 *
 * @param out_reason  may be NULL; a static string naming the first collision
 * @return            FT_TT_OK, or FT_TT_ERROR
 */
int ft_tt_group_check_disjoint(const ft_tt_core_group *groups, uint32_t group_count,
                               const char **out_reason);

/**
 * Set up every core's queue control block in a group, with one nonce for the whole group.
 *
 * @param control_address  the same per-core L1 address for every core; FT_TT_OP_QUEUE_BASE
 * @param capacity         ring slots, a power of two
 * @param run_nonce        ft_tt_queue_make_nonce()
 * @return                 FT_TT_OK, or the first failing core's code
 */
int ft_tt_group_queues_init(const ft_tt_noc_access *noc, ft_tt_core_group *group,
                            uint32_t control_address, uint32_t capacity, uint32_t run_nonce);

/* ================================================================
 * Group-level completion — "the group finished item N"
 * ================================================================
 *
 * Until this existed, completion was per CORE and the only group-level answer available was
 * a SUM of `completed_count` across the group's cores. A sum cannot express what a group
 * scheduler needs, and worse, it is satisfiable by the wrong thing: one core doing 2N items
 * and another doing none sums to the same 2N as every core doing N. That is the
 * "`RUN_MSG_DONE` is 0" family of lie again — a number that looks like evidence of work and
 * is evidence of arithmetic (dev notes §3).
 *
 * So the kernels write it. Each member's WRITER puts {sequence, token} into ITS OWN slot of
 * a block on the group's first core, after that descriptor's work; the host asks "has every
 * slot reached N, with THIS run's token for N". One writer per slot, one NoC transaction
 * per post, no atomics, no ordering to argue about — and because the block is on a core
 * rather than in host memory, another group on the device can read it too.
 */

/**
 * Write the group completion block on the group's FIRST core: the magic, the member count,
 * and every slot zeroed.
 *
 * The zero is not the poison the per-core protocol uses, and does not need to be: the
 * group check asks for a SEQUENCE, and 0 is not a sequence any descriptor may carry. What
 * a stale block from a previous run could offer is a large sequence, and that is what the
 * TOKEN check refuses.
 *
 * @param group_address  L1 address on the leader; FT_TT_OP_QUEUE_BASE + FT_TT_QUEUE_GROUP_OFFSET
 * @return               FT_TT_OK; FT_TT_EINVAL; FT_TT_ENOMEM if the group has more members
 *                       than FT_TT_QUEUE_GROUP_MAX_MEMBERS
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_group_completion_init(const ft_tt_noc_access *noc, ft_tt_core_group *group,
                                uint32_t group_address);

/**
 * Fill the per-processor kernel binding for one member of a group, including where its
 * group post goes.
 *
 * Separate from ft_tt_queue_kernel_args() because the GROUP is what knows the leader's
 * coordinates and this member's slot, and the queue is what knows the addresses.
 *
 * @param slot       this core's index within the group
 * @param processor  FT_TT_PROC_*; only the processor that publishes completion gets a
 *                   non-zero group address, because a second poster would give one slot
 *                   two writers
 * @param publisher  the FT_TT_PROC_* that publishes completion for this program: DM1 for
 *                   the resident triad, DM0 for the BRISC-only kernel
 */
int ft_tt_group_kernel_binding(const ft_tt_core_group *group, uint32_t slot, unsigned processor,
                               unsigned publisher, uint32_t idle_spin_limit,
                               uint32_t cb_wait_spin_limit,
                               ft_tt_queue_kernel_binding *out_binding);

/** What one member's slot currently says. The slot LAYOUT belongs to the queue protocol
 *  (ft_tt_queue.h), so this is that type under the name group callers already use — one
 *  definition, not two that have to be kept in step. */
typedef ft_tt_queue_group_slot ft_tt_group_member_completion;

/**
 * Read every member's slot out of the leader's L1 in ONE device access.
 *
 * @param out_members  at least `ft_tt_launch_core_set_count(&group->cores)` entries
 * @param capacity     how many entries it has
 */
int ft_tt_group_completion_read(const ft_tt_noc_access *noc, const ft_tt_core_group *group,
                                ft_tt_group_member_completion *out_members, uint32_t capacity);

/**
 * Did EVERY member reach group sequence `sequence`, with this run's token for it?
 *
 * Both halves are needed and each fails a different lie: the sequence alone is satisfiable
 * by a previous run's residue, and the token alone cannot say how far the group got.
 *
 * @param expected_token  the token the host stamped on the descriptor carrying `sequence`;
 *                        ft_tt_queue_token_at() of the member's own queue
 * @param out_reason      may be NULL; a static string naming the first disagreement
 * @param out_short_slot  may be NULL; the first member that is behind
 * @return                FT_TT_OK, or FT_TT_ERROR
 */
int ft_tt_group_check_completion(const ft_tt_core_group *group,
                                 const ft_tt_group_member_completion *members, uint32_t count,
                                 uint32_t sequence, uint32_t expected_token,
                                 const char **out_reason, uint32_t *out_short_slot);

/**
 * Poll until every member reaches `sequence`, or until `timeout_ms` elapses.
 *
 * @param out_reached  may be NULL; the lowest sequence any member had, on success or timeout
 * @return             FT_TT_OK; FT_TT_EIO on timeout; FT_TT_EINVAL for timeout_ms 0
 */
int ft_tt_group_wait_completion(const ft_tt_noc_access *noc, const ft_tt_core_group *group,
                                uint32_t sequence, uint32_t timeout_ms, uint32_t *out_reached);

#endif /* FT_TT_GROUP_H */
