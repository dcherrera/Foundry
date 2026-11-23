/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_precondition.h
 * @brief S2 ON-CARD COMPLETION, the kernel half: ftq_precondition_wait(), which every resident
 *        reader calls at descriptor start, before it touches an input. The wire format and the
 *        parse / compare it runs are resident_precondition_protocol.h.
 *
 * What it does, per descriptor:
 *   1. read this descriptor's sidecar entry (local L1); no entry for this index: return;
 *   2. otherwise NoC-READ the producer's control words 4..7 into this core's own landing line
 *      until its completed count reaches the entry's and the token beside it is the entry
 *      nonce's;
 *   3. record when the precondition was met (the reader's counters and stamp slot).
 *
 * READS ONLY. Nothing here writes another core's L1: the inline-write group post wedged the
 * posting core at its next boot (DEV_NOTES.md "Do not use inline writes in a resident
 * kernel"). The one remote transaction is a 16-byte read, issued and settled here.
 *
 * BOUNDED AND STOPPABLE TO THE LAST INSTRUCTION, as resident_group_pull.h: the command-buffer
 * wait, the read-response wait and the poll each spin through ftq_precondition_spin(), which
 * checks the stop word every 1024 spins and gives up at the CB spin bound. A refusal ABORTS the
 * descriptor (the reader then leaves its loop with the descriptor unconsumed, rule 5), it never
 * hangs the core:
 *   - a malformed entry for this index (bad coordinate, zero count or nonce): refused before
 *     any read — a coordinate that does not fit the node-id field is a hung NoC;
 *   - the producer's block poisoned (re-armed under the wait), or its count going backwards:
 *     refused at once, the count the entry names belongs to a producer that is gone;
 *   - the count reached but the token never the nonce's for kPreMismatchLimit polls in a row:
 *     a producer of another nonce, refused (one mismatched read is the writer's token-before-
 *     count window and is retried).
 * A read still in flight at an abort lands in this core's own landing line, which nothing
 * else uses.
 *
 * POLLING IS PACED (kPrePollGap wall-clock ticks between reads, ~0.2 us) so the consumers of
 * one op do not saturate the producer's NoC port while it works.
 *
 * @status S2 2.2: included by every resident reader (tt/tools/check_precondition_wait.sh
 *         proves each calls it before its first input read). Card gate pending.
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_PRECONDITION_H
#define RESIDENT_PRECONDITION_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_precondition_protocol.h"
#include "resident_queue_protocol.h"

/* Consecutive reached-but-mismatched polls before the producer is called foreign. */
enum { kPreMismatchLimit = 256 };
/* Wall-clock ticks between two polls of the producer (~0.19 us at 1.35 GHz). */
enum { kPrePollGap = 256 };
/* Producer control words 4..7, byte offset from its control block. */
enum { kPreWindowOffset = 16 };

/** One more spin of a precondition wait: false (descriptor aborted) on a stop request, checked
 *  every 1024 spins, or once the CB spin bound is spent. */
static bool ftq_precondition_spin(ftq_context *ctx, uint32_t *spins)
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

/** Record that descriptor `index`'s precondition is resolved now (entry or not). */
static void ftq_precondition_stamp(volatile FTQ_L1_PTR uint32_t *counters, uint32_t control,
                                   uint32_t index)
{
    volatile FTQ_L1_PTR uint32_t *stamps =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(control + (uint32_t)kPreStampOffset);

    stamps[index & ((uint32_t)kPreStampSlots - 1u)] = ft_bh_wall_clock_low();
    counters[kPreCounterMet] = index + 1u;
}

/**
 * Wait for the precondition the sidecar names for the descriptor at the reader's cursor.
 * @return true to run the descriptor; false when it was aborted (stop, spin bound, or a
 *         refused entry — ctx->error says which), and the reader must leave its loop.
 */
static bool ftq_precondition_wait(ftq_context *ctx)
{
    const uint32_t control = (uint32_t)(uintptr_t)ctx->control;
    const uint32_t index = ctx->cursor;
    volatile FTQ_L1_PTR uint32_t *entry =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(control +
                                                     ftq_pre_entry_offset(index,
                                                                          ctx->capacity_mask));
    volatile FTQ_L1_PTR uint32_t *landing =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(control + (uint32_t)kPreLandingOffset);
    volatile FTQ_L1_PTR uint32_t *counters =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(control + (uint32_t)kPreCountersOffset);
    ftq_pre_entry pre;
    uint32_t verdict, spins = 0u, mismatches = 0u, highest = 0u, started, paced;
    bool waited = false;

    ft_bh_l1_cache_invalidate();
    verdict = ftq_pre_parse(entry, index, &pre);
    if (verdict == (uint32_t)kPreNone) {
        ftq_precondition_stamp(counters, control, index);
        return true;
    }
    if (verdict != (uint32_t)kPreValid) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    counters[kPreCounterHonoured] += 1u;
    ftq_status_set(ctx, kStateSemWait);
    started = ft_bh_wall_clock_low();
    for (;;) {
        while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_READ_CMD_BUF))
            if (!ftq_precondition_spin(ctx, &spins))
                return false;
        ft_tensix_noc_issue_read(NOC_INDEX, FT_BH_READ_CMD_BUF,
                                 ft_tensix_noc_address(pre.x, pre.y,
                                                       control + (uint32_t)kPreWindowOffset),
                                 (uint32_t)(uintptr_t)landing, 16u, 1);
        while (!ft_tensix_noc_reads_flushed(NOC_INDEX))
            if (!ftq_precondition_spin(ctx, &spins))
                return false;
        ft_bh_l1_cache_invalidate();
        verdict = ftq_pre_compare(&pre, landing);
        if (verdict == (uint32_t)kPreMet)
            break;
        if (verdict == (uint32_t)kPreGone || (int32_t)(landing[1] - highest) < 0) {
            ftq_abort(ctx, kErrBadParameters);
            return false;
        }
        highest = landing[1];
        mismatches = verdict == (uint32_t)kPreMismatch ? mismatches + 1u : 0u;
        if (mismatches >= (uint32_t)kPreMismatchLimit) {
            ftq_abort(ctx, kErrBadParameters);
            return false;
        }
        waited = true;
        paced = ft_bh_wall_clock_low();
        while (ft_bh_wall_clock_low() - paced < (uint32_t)kPrePollGap)
            ;
        if (!ftq_precondition_spin(ctx, &spins))
            return false;
    }
    if (waited)
        counters[kPreCounterWaited] += 1u;
    counters[kPreCounterCycles] += (ft_bh_wall_clock_low() - started) >> 4;
    ftq_precondition_stamp(counters, control, index);
    ftq_status_set(ctx, kStateExec);
    return true;
}

#endif /* RESIDENT_PRECONDITION_H */
