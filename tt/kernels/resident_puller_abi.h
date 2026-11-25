/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_puller_abi.h
 * @brief The puller program entries a host builder may emit beyond the A2/S2 set — shared by
 *        the kernel (resident_puller.c) and Pantheon's program builder (pn_tt_dispatch*.c), so a
 *        host that emits an entry and a kernel blob that runs it come from one definition.
 *
 * @status L10: FLUSH. S2 (on-card completion): PRECOND. The AWAIT step (host test
 *         tt/tests/test_puller_await.c).
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_PULLER_ABI_H
#define RESIDENT_PULLER_ABI_H

#include <stdint.h>

/** Program-entry type 9, one word: write every queued head now. The builder puts it before
 *  ring slots / parameter blocks it hoisted above an AWAIT (they then go out while the awaited
 *  core works). A puller built before it refuses the entry (error 2), never misreads it. */
#define FT_TT_PULLER_ENTRY_FLUSH 9u

/** Program-entry type 10, FT_TT_PULLER_PRECOND_WORDS words:
 *    10, target, index0, step, producer, count0, count_step
 *  Write the WAIT SIDECAR entry (kernels/resident_precondition_protocol.h) of descriptor
 *  index0 + step * k on queue `target` (bit 31 set: multicast group `target & 0xFFFF`, as a
 *  RING entry): {index, producer x | y << 8 | tag << 16, count0 + count_step * k, producer
 *  nonce}, at control + 0x900 + 16 * (index & the target's mask). A plain NoC write, landed by
 *  the barrier in front of the next head flush like a ring slot. The builder puts it after the
 *  descriptor's ring slot and before its head. A producer whose coordinate does not fit six
 *  bits is refused (error 2), never written. A puller built before it refuses the entry. */
#define FT_TT_PULLER_ENTRY_PRECOND 10u
#define FT_TT_PULLER_PRECOND_WORDS 7u

/** The token a queue with `nonce` gives descriptor `index` (ft_tt_queue_token_at). */
static inline uint32_t ft_tt_puller_token(uint32_t nonce, uint32_t index)
{
    uint32_t token = nonce ^ ((index + 1u) * 0x9E3779B9u);

    if (token == 0u || token == 0x7B1B1B1Bu)
        token = nonce ^ 0xA5A5A5A5u;
    return token;
}

/** What one AWAIT poll (program entry 4) asks the puller to do next. */
#define FT_TT_PULLER_AWAIT_POLL 0u   /* read the line again */
#define FT_TT_PULLER_AWAIT_MET 1u
#define FT_TT_PULLER_AWAIT_POISON 2u /* refuse: kPlrErrAwaitPoison */
#define FT_TT_PULLER_AWAIT_TOKEN 3u  /* refuse: kPlrErrAwaitToken */

/** Two words of the puller's L1 (just past its AWAIT census), zeroed when the puller boots:
 *  [0] mismatching AWAIT polls read again, [1] AWAITs that read one or more. Data only: a
 *  nonzero count with no refusal is a torn line the retry absorbed. */
#define FT_TT_PULLER_AWAIT_RETRIES 0x16D800u

/** Polls in a row whose count is reached but whose token is not that count's, before the
 *  AWAIT refuses (resident_precondition.h kPreMismatchLimit, the reader-side twin). */
#define FT_TT_PULLER_AWAIT_MISMATCH_LIMIT 256u

/**
 * One AWAIT poll on the target's control words 4..7 (`line`: stop, completed count, completed
 * token, state). MET once the count is >= `wanted` and the token beside it is the one `nonce`
 * gives count - 1 (the count READ: an AWAIT on a step mark names an older count than the
 * queue's last). The writer stores token before count, so a read between the two stores sees
 * the next descriptor's token beside the old count: that line is polled again, not refused. A
 * mismatch that persists for FT_TT_PULLER_AWAIT_MISMATCH_LIMIT polls is a queue of another
 * nonce and refuses. `mismatches` is the caller's count of mismatching polls in a row, 0 at
 * the AWAIT's start.
 *
 * FT_TT_PULLER_AWAIT_HOST_RED_NO_RETRY (host test only, never a kernel build): the first
 * mismatch refuses, the rule before the retry; the host test must catch it.
 */
static inline uint32_t ft_tt_puller_await_step(const volatile uint32_t *line, uint32_t wanted,
                                               uint32_t nonce, uint32_t *mismatches)
{
    uint32_t limit = FT_TT_PULLER_AWAIT_MISMATCH_LIMIT;

#if defined(FT_TT_PULLER_AWAIT_HOST_RED_NO_RETRY)
    limit = 1u;
#endif
    if (line[1] < wanted)
        return FT_TT_PULLER_AWAIT_POLL;
    if (line[3] == 0x7B1B1B1Bu)
        return FT_TT_PULLER_AWAIT_POISON;
    if (line[2] == ft_tt_puller_token(nonce, line[1] - 1u))
        return FT_TT_PULLER_AWAIT_MET;
    return ++*mismatches >= limit ? FT_TT_PULLER_AWAIT_TOKEN : FT_TT_PULLER_AWAIT_POLL;
}

#endif /* RESIDENT_PULLER_ABI_H */
