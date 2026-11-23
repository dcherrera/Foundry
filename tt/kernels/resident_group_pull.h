/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_group_pull.h
 * @brief GROUP COMPLETION BY PULL for a writer outside the matmul triad: member 0 of a split
 *        family NoC-READS every other member's done-sequence before it publishes its own
 *        completion, so the host (or the puller) awaits member 0 alone.
 *
 * The protocol is resident_matmul_writer.c's L2 sweep, word for word:
 *   - the PULL TABLE at control + kPullTableOffset (resident_queue_protocol.h), written by the
 *     host on member 0: magic "FTQP", member count, then `x | y << 16` per member;
 *   - descriptor a3 = pulled member count | flag bits (bit 31 kPullRedSkip skips the pull),
 *     a4 = the group sequence every pulled member must have reached, a5 = this descriptor's
 *     own group sequence, which every writer stores in control word 56 (kFieldGroupDone)
 *     BEFORE it publishes completion — that word is what the pull reads.
 *
 * READS ONLY. Inline-write group posting (`ftq_group_post`) wedged the posting core at its
 * next boot; nothing here writes another core's L1.
 *
 * Each still-pending member is read into its own 16-byte landing line per sweep, with ONE read
 * barrier a sweep. Stoppable and bounded in every spin, the NoC command buffer and the read
 * barrier included: a stop request or the CB spin bound aborts the descriptor instead of
 * hanging the core, and the writer then exits, so the triad retires.
 *
 * CONTROL WORD 56 BELONGS TO THE WRITER ALONE on any core a pull can name. Nothing else may
 * store to it, add to it or keep a clock in it: the gdn reader's CONV barrier clock lived
 * there until S1 and satisfied member 0's pull before the member had finished
 * (tt/tools/check_group_done_word.sh holds every pulled program's kernels to this).
 *
 * @status S1 (design spec 4.2): included by
 *         resident_vector_writer.c, resident_attention_writer.c, resident_gdn_writer.c and
 *         resident_place_writer.c. Card gate pending.
 * @issues None
 * @todo resident_matmul_writer.c keeps its own copy of this sweep (plus its one-at-a-time
 *       walk): switching it to this header changes its sources' provenance on nine programs
 *       for no change in behaviour. Fold it in with the next change that moves those images.
 */

#ifndef FOUNDRY_RESIDENT_GROUP_PULL_H
#define FOUNDRY_RESIDENT_GROUP_PULL_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_queue_protocol.h"

/* Every a3 flag bit (31 red skip, 30 the matmul writer's serial walk); the rest is the member
 * count. Host mirror: Pantheon pn_tt_group_pull.h. */
enum { kGroupPullFlags = 0xC0000000u };
/* 16-byte landing per member (1..63), 0xFB00..0xFEF0: the matmul writer's kPullSweepOffset. */
enum { kGroupPullSweepOffset = 0xFB00 };

/** One more spin of a pull wait: false (descriptor aborted) on a stop request — checked every
 *  1024 spins — or once the CB spin bound is spent. Every spin of every wait below passes
 *  through here, so none of them can outlive a stop or the bound. */
static bool ftq_group_pull_spin(ftq_context *ctx, uint32_t *spins)
{
    if ((++*spins & 1023u) != 0u)
        return true;
    if (ftq_stop_requested(ctx)) {
        ftq_abort(ctx, kErrNone);
        return false;
    }
    if (*spins >= ctx->cb_spin_limit) {
        ftq_abort(ctx, kErrSemWaitTimeout);
        return false;
    }
    return true;
}

/** Wait until members 1..`members`-1 of `table` have each posted a group sequence at or past
 *  `sequence`. @return false (descriptor aborted) on a stop request or the spin bound.
 *
 *  BOUNDED TO THE LAST INSTRUCTION. ft_tensix_noc_read() and ft_tensix_noc_read_barrier() each
 *  spin with no bound (on the command buffer, and on the read-response counter), so this
 *  issues and settles its reads itself, through ftq_group_pull_spin(). A member that never
 *  answers costs the bound and an abort, never a core that cannot be retired. A read still in
 *  flight at an abort lands in this core's own landing lines, which nothing else uses. */
static bool ftq_group_pull_sweep(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *table,
                                 uint32_t members, uint32_t sequence)
{
    const uint32_t control = (uint32_t)(uintptr_t)ctx->control;
    volatile FTQ_L1_PTR uint32_t *landing =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(control + kGroupPullSweepOffset);
    uint32_t pending[2] = { 0u, 0u };
    uint32_t spins = 0u, member;

    for (member = 1u; member < members; member++)
        pending[member >> 5] |= 1u << (member & 31u);
    for (;;) {
        for (member = 1u; member < members; member++) {
            const uint32_t xy = table[2u + member];

            if ((pending[member >> 5] & (1u << (member & 31u))) == 0u)
                continue;
            while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_READ_CMD_BUF))
                if (!ftq_group_pull_spin(ctx, &spins))
                    return false;
            ft_tensix_noc_issue_read(NOC_INDEX, FT_BH_READ_CMD_BUF,
                                     ft_tensix_noc_address(xy & 0xFFFFu, xy >> 16,
                                                           control + kFieldGroupDone * 4u),
                                     (uint32_t)(uintptr_t)(landing + member * 4u), 16u, 1);
            if (!ftq_group_pull_spin(ctx, &spins))
                return false;
        }
        while (!ft_tensix_noc_reads_flushed(NOC_INDEX))
            if (!ftq_group_pull_spin(ctx, &spins))
                return false;
        ft_bh_l1_cache_invalidate();
        for (member = 1u; member < members; member++)
            if ((int32_t)(landing[member * 4u] - sequence) >= 0)
                pending[member >> 5] &= ~(1u << (member & 31u));
        if ((pending[0] | pending[1]) == 0u)
            return true;
    }
}

/** The pull a descriptor's a3/a4 ask for, after checking the table member 0 holds. A table
 *  that is absent or names fewer members than a3 aborts with kErrBadParameters. */
static bool ftq_group_pull(ftq_context *ctx, uint32_t members, uint32_t sequence)
{
    volatile FTQ_L1_PTR uint32_t *table =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)((uint32_t)(uintptr_t)ctx->control +
                                                     kPullTableOffset);

    if (table[0] != kPullMagic || members > table[1] || members > kPullMaxMembers) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    return ftq_group_pull_sweep(ctx, table, members, sequence);
}

#endif /* FOUNDRY_RESIDENT_GROUP_PULL_H */
