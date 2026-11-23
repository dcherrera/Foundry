/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_precondition_protocol.h
 * @brief S2 ON-CARD COMPLETION, the wire half: the WAIT SIDECAR a descriptor's precondition
 *        travels in, and the parse / compare every resident reader runs on it. Pure C over
 *        plain words, so the host (Pantheon's program builder, the foundry host test) and the
 *        kernels (resident_precondition.h) share one definition.
 *
 * THE RULE (design spec 4.3): the puller no longer
 * waits for window k-1 before it issues window k. Each descriptor that opens window k on its
 * queue carries "producer (x, y) has completed `count` descriptors", and the consumer's
 * READER checks it before it touches an input. Execution order is unchanged; only who
 * detects the producer's completion moves.
 *
 * THE SIDECAR: one 16-byte entry per ring slot, in the queue region at
 *   control + kPreSidecarOffset + 16 * (index & capacity_mask)
 * written by the puller (a plain NoC write, never inline) with the ring slot and before the
 * head, so the write barrier in front of every head flush lands it first. Words:
 *   0 index   the descriptor's own index (the queue's monotonic count): an entry for another
 *             index is STALE, not this descriptor's, and ignored;
 *   1 xy      producer x | y << 8 | kPreTag << 16 (x, y < 64: the NoC node-id field is six
 *             bits, and a coordinate that does not fit truncates onto a live endpoint);
 *   2 count   the producer's completed count to wait for (>= 1);
 *   3 nonce   the producer queue's run nonce. The token of whatever count the reader reads
 *             is derived from it (ft_tt_queue_token_at), so a producer that ran PAST `count`
 *             still verifies, and a re-armed producer (new nonce) never does.
 *
 * WHY THE REGION IS FREE: the ring is 0x100 + 32 * capacity and capacity is at most 64
 * (Pantheon pn_tt_residency.h PN_TT_RESIDENCY_QUEUE_CAPACITY; the matmul group arms 8), so
 * it ends at 0x900; the MoE expansion mailbox starts at 0x1000
 * (resident_matmul_moe_protocol.h FT_TT_MOE_MAILBOX_OFFSET) and the parameter-cache overflow
 * at 0x2000 (ft_tt_queue.h). 64 entries end at 0xD00; the reader's own words follow and end at
 * kPreRegionEnd = 0xE20. The host zeroes [kPreSidecarOffset, kPreRegionEnd) at every arm.
 *
 * READER-OWNED WORDS (each written by the queue's reader alone):
 *   kPreLandingOffset   16-byte landing for the producer's control words 4..7;
 *   kPreCountersOffset  [0] index + 1 of the last descriptor whose precondition was resolved,
 *                       [1] cycles >> 4 spent waiting, [2] entries honoured, [3] entries that
 *                       had to wait at all;
 *   kPreStampOffset     wall clock (low word) at which descriptor i's precondition was met,
 *                       at slot i & 63 — the matmul writer starts its busy clock there.
 *
 * WIRE CONTRACT FORM: one-line anonymous enums, as resident_queue_protocol.h.
 *
 * @status S2 2.1. Host test tt/tests/test_precondition_protocol.c (make -f
 *         Makefile.precondition run / red). Card gate pending.
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_PRECONDITION_PROTOCOL_H
#define RESIDENT_PRECONDITION_PROTOCOL_H

#include <stdbool.h>
#include <stdint.h>

#ifndef FTQ_PRE_INLINE
#define FTQ_PRE_INLINE static inline
#endif

/* ---- Layout, byte offsets from the queue control block ---- */
enum { kPreSidecarOffset = 0x900 };
enum { kPreEntryBytes = 16 };
enum { kPreMaxEntries = 64 };
enum { kPreLandingOffset = 0xD00 };
enum { kPreCountersOffset = 0xD10 };
enum { kPreStampOffset = 0xD20 };
enum { kPreStampSlots = 64 };
enum { kPreRegionEnd = 0xE20 };

/* ---- Entry words ---- */
enum { kPreWordIndex = 0 };
enum { kPreWordXy = 1 };
enum { kPreWordCount = 2 };
enum { kPreWordNonce = 3 };
enum { kPreTag = 0x5C1D };
enum { kPreCoordLimit = 64 };

/* ---- Reader counters ---- */
enum { kPreCounterMet = 0 };
enum { kPreCounterCycles = 1 };
enum { kPreCounterHonoured = 2 };
enum { kPreCounterWaited = 3 };

/* ---- Token arithmetic (ft_tt_queue_token_at, resident_puller.c pull_token) ---- */
#define FTQ_PRE_TOKEN_MULTIPLIER 0x9E3779B9u /* past INT_MAX: a define, not an enumerator */
#define FTQ_PRE_TOKEN_FALLBACK 0xA5A5A5A5u
enum { kPrePoison = 0x7B1B1B1B };

/* ---- Parse verdicts ---- */
enum { kPreNone = 0 };      /* no entry for this descriptor: run at once */
enum { kPreValid = 1 };     /* an entry: wait for it */
enum { kPreMalformed = 2 }; /* this descriptor's index with bad fields: refuse */

/* ---- Compare verdicts ---- */
enum { kPreWait = 0 };      /* count short of the entry's */
enum { kPreMet = 1 };
enum { kPreMismatch = 2 };  /* count reached, token not the nonce's (transient or foreign) */
enum { kPreGone = 3 };      /* the producer's block is poisoned: it is being re-armed */

/** One parsed entry. */
typedef struct ftq_pre_entry {
    uint32_t x, y;
    uint32_t count;
    uint32_t nonce;
} ftq_pre_entry;

/** The token a queue with `nonce` gives descriptor `index`. */
FTQ_PRE_INLINE uint32_t ftq_pre_token(uint32_t nonce, uint32_t index)
{
    uint32_t token = nonce ^ ((index + 1u) * FTQ_PRE_TOKEN_MULTIPLIER);

    if (token == 0u || token == (uint32_t)kPrePoison)
        token = nonce ^ FTQ_PRE_TOKEN_FALLBACK;
    return token;
}

/** Entry word 1 for producer (x, y); 0 when a coordinate does not fit six bits. */
FTQ_PRE_INLINE uint32_t ftq_pre_xy_word(uint32_t x, uint32_t y)
{
    if (x >= (uint32_t)kPreCoordLimit || y >= (uint32_t)kPreCoordLimit)
        return 0u;
    return x | (y << 8) | ((uint32_t)kPreTag << 16);
}

/** Byte offset of descriptor `index`'s entry from the control block. */
FTQ_PRE_INLINE uint32_t ftq_pre_entry_offset(uint32_t index, uint32_t capacity_mask)
{
    return (uint32_t)kPreSidecarOffset + (uint32_t)kPreEntryBytes * (index & capacity_mask);
}

/**
 * Read the entry at `entry` for descriptor `index`.
 *
 * kPreNone: no tag (a zeroed or never-written slot) or another index (a stale entry of an
 * earlier lap of the ring). kPreMalformed: this index, but a coordinate that does not fit,
 * stray bits in the coordinate word, a zero count or a nonce no queue is armed with — refused,
 * never polled, because a wrong endpoint is a hung NoC. kPreValid: `out` is filled.
 */
FTQ_PRE_INLINE uint32_t ftq_pre_parse(const volatile uint32_t *entry, uint32_t index,
                                      ftq_pre_entry *out)
{
    const uint32_t xy = entry[kPreWordXy];
    const uint32_t count = entry[kPreWordCount];
    const uint32_t nonce = entry[kPreWordNonce];

    if ((xy >> 16) != (uint32_t)kPreTag || entry[kPreWordIndex] != index)
        return kPreNone;
    if ((xy & 0xC0C0u) != 0u || count == 0u || nonce == 0u || nonce == (uint32_t)kPrePoison)
        return kPreMalformed;
    out->x = xy & 0x3Fu;
    out->y = (xy >> 8) & 0x3Fu;
    out->count = count;
    out->nonce = nonce;
    return kPreValid;
}

/**
 * Compare the producer's control words 4..7 (`window`: stop, completed count, completed
 * token, state) with a parsed entry.
 *
 * kPreMet only when the count has reached the entry's (wrap-safe) AND the token beside it is
 * the one the entry's nonce gives that count. The writer stores token before count, so a read
 * between the two sees the new token beside the old count: kPreMismatch, which the caller
 * retries. A persistent kPreMismatch is a producer of another nonce (re-armed, or a stale
 * entry) — never a satisfied wait. kPreGone: the producer's state is the host's poison, i.e.
 * its control block was re-initialised under this wait.
 *
 * FTQ_PRE_HOST_RED_OFF_BY_ONE (host test only, never a kernel build) accepts one count early:
 * the red arm the host test must catch.
 */
FTQ_PRE_INLINE uint32_t ftq_pre_compare(const ftq_pre_entry *entry, const volatile uint32_t *window)
{
    const uint32_t count = window[1];
    const uint32_t token = window[2];
    uint32_t wanted = entry->count;

    if (window[3] == (uint32_t)kPrePoison)
        return kPreGone;
#if defined(FTQ_PRE_HOST_RED_OFF_BY_ONE)
    wanted -= 1u;
#endif
    if ((int32_t)(count - wanted) < 0 || count == 0u)
        return kPreWait;
    return token == ftq_pre_token(entry->nonce, count - 1u) ? kPreMet : kPreMismatch;
}

#endif /* RESIDENT_PRECONDITION_PROTOCOL_H */
