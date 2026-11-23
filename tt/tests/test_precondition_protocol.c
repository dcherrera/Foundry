/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_precondition_protocol.c
 * @brief S2 on-card completion off hardware: the WAIT SIDECAR's layout, and the KERNEL'S OWN
 *        parse / compare (kernels/resident_precondition_protocol.h, the same inline every
 *        resident reader compiles) against the host's queue arithmetic.
 *
 * A precondition has no failure mode the card reports. Accept one count early and the
 * consumer reads its producer's half-written strip; honour a stale entry and it waits on a
 * producer that is gone; parse a zeroed slot as an entry and it polls core (0, 0). Each is
 * checked here, where checking is cheap:
 *   - LAYOUT: the sidecar starts where the largest ring ends and the region ends before the
 *     MoE mailbox and the parameter-cache overflow (the free-region proof, as numbers);
 *   - TOKENS: the inline token is ft_tt_queue_token_at(), fallbacks included;
 *   - PARSE: a zeroed slot, a stale index (either lap) and a foreign tag are not entries; a
 *     coordinate past six bits, a zero count or an unarmed nonce is refused, never polled;
 *   - COMPARE: one count short waits; the count with its token is met, and so is any later
 *     count with ITS token (the producer ran on); a wrong-nonce token and the writer's
 *     token-before-count window are mismatches, never met; a poisoned producer is gone;
 *     the count compares across the 32-bit wrap.
 *
 * RED (make -f Makefile.precondition red): the same suite built with
 * FTQ_PRE_HOST_RED_OFF_BY_ONE, a compare that accepts one count early, must FAIL.
 *
 *   make -f Makefile.precondition run
 *
 * @status S2 2.1.
 * @issues None
 * @todo None
 */
#include <stdio.h>
#include <string.h>

#include "ft_tt_queue.h"
#include "ft_tt_test_report.h"
#include "kernels/resident_matmul_moe_protocol.h"
#include "kernels/resident_precondition_protocol.h"

/** The token a host queue with `nonce` gives descriptor `index`. */
static uint32_t host_token(uint32_t nonce, uint32_t index)
{
    ft_tt_queue queue;

    memset(&queue, 0, sizeof(queue));
    queue.run_nonce = nonce;
    return ft_tt_queue_token_at(&queue, index);
}

static void put_entry(uint32_t *entry, uint32_t index, uint32_t x, uint32_t y, uint32_t count,
                      uint32_t nonce)
{
    entry[kPreWordIndex] = index;
    entry[kPreWordXy] = ftq_pre_xy_word(x, y);
    entry[kPreWordCount] = count;
    entry[kPreWordNonce] = nonce;
}

static void put_window(uint32_t *window, uint32_t count, uint32_t token, uint32_t state)
{
    window[0] = 0u;
    window[1] = count;
    window[2] = token;
    window[3] = state;
}

static void test_layout(void)
{
    ft_tt_test_step("layout: the sidecar region is free in every family's queue block");
    ft_tt_test_check_fmt("sidecar starts where the 64-slot ring ends",
                         kPreSidecarOffset == FT_TT_QUEUE_RING_OFFSET + 64u * 32u,
                         "0x%x vs 0x%x", (unsigned)kPreSidecarOffset,
                         FT_TT_QUEUE_RING_OFFSET + 64u * 32u);
    ft_tt_test_check_fmt("64 entries end at the landing line",
                         kPreSidecarOffset + kPreMaxEntries * kPreEntryBytes == kPreLandingOffset,
                         "0x%x", (unsigned)(kPreSidecarOffset + kPreMaxEntries * kPreEntryBytes));
    ft_tt_test_check_fmt("the counters follow the landing line, the stamps the counters",
                         kPreCountersOffset == kPreLandingOffset + 16 &&
                             kPreStampOffset == kPreCountersOffset + 16 &&
                             kPreRegionEnd == kPreStampOffset + kPreStampSlots * 4,
                         "0x%x 0x%x 0x%x", (unsigned)kPreCountersOffset,
                         (unsigned)kPreStampOffset, (unsigned)kPreRegionEnd);
    ft_tt_test_check_fmt("the region ends before the MoE expansion mailbox",
                         kPreRegionEnd <= FT_TT_MOE_MAILBOX_OFFSET, "0x%x <= 0x%x",
                         (unsigned)kPreRegionEnd, FT_TT_MOE_MAILBOX_OFFSET);
    ft_tt_test_check_fmt("and before the parameter-cache overflow",
                         kPreRegionEnd <= FT_TT_QUEUE_PARAM_CACHE_OVERFLOW_OFFSET, "0x%x",
                         FT_TT_QUEUE_PARAM_CACHE_OVERFLOW_OFFSET);
    ft_tt_test_check_fmt("entry slots wrap with the ring (capacity 8 and 64)",
                         ftq_pre_entry_offset(9u, 7u) == 0x910u &&
                             ftq_pre_entry_offset(64u + 5u, 63u) == 0x950u &&
                             ftq_pre_entry_offset(63u, 63u) == 0xCF0u,
                         "0x%x 0x%x 0x%x", ftq_pre_entry_offset(9u, 7u),
                         ftq_pre_entry_offset(69u, 63u), ftq_pre_entry_offset(63u, 63u));
}

static void test_tokens(void)
{
    static const uint32_t nonces[] = { 0x12345678u, 0xDEADBEEFu, 1u, 0x7B1B1B1Au };
    unsigned n, i, bad = 0u;
    uint32_t zero_nonce, poison_nonce;

    ft_tt_test_step("tokens: the inline token is the host queue's");
    for (n = 0; n < sizeof(nonces) / sizeof(*nonces); n++)
        for (i = 0; i < 2000u; i += 7u)
            if (ftq_pre_token(nonces[n], i) != host_token(nonces[n], i))
                bad++;
    ft_tt_test_check_fmt("1,144 (nonce, index) pairs agree", bad == 0u, "%u differ", bad);
    zero_nonce = 41u * FTQ_PRE_TOKEN_MULTIPLIER; /* index 40 would give token 0 */
    poison_nonce = (uint32_t)kPrePoison ^ (6u * FTQ_PRE_TOKEN_MULTIPLIER);
    ft_tt_test_check_fmt("the zero and poison fallbacks agree",
                         ftq_pre_token(zero_nonce, 40u) == host_token(zero_nonce, 40u) &&
                             ftq_pre_token(zero_nonce, 40u) != 0u &&
                             ftq_pre_token(poison_nonce, 5u) == host_token(poison_nonce, 5u) &&
                             ftq_pre_token(poison_nonce, 5u) != (uint32_t)kPrePoison,
                         "0x%08x 0x%08x", ftq_pre_token(zero_nonce, 40u),
                         ftq_pre_token(poison_nonce, 5u));
}

static void test_parse(void)
{
    uint32_t entry[4];
    ftq_pre_entry pre;
    uint32_t v;

    ft_tt_test_step("parse: only this descriptor's well-formed entry is an entry");
    memset(entry, 0, sizeof(entry));
    v = ftq_pre_parse(entry, 0u, &pre);
    ft_tt_test_check_fmt("a ZEROED slot is no entry, even for index 0", v == (uint32_t)kPreNone,
                         "verdict %u", v);
    put_entry(entry, 100u, 7u, 11u, 42u, 0x12345678u);
    v = ftq_pre_parse(entry, 100u, &pre);
    ft_tt_test_check_fmt("a well-formed entry parses", v == (uint32_t)kPreValid && pre.x == 7u &&
                             pre.y == 11u && pre.count == 42u && pre.nonce == 0x12345678u,
                         "verdict %u (%u,%u) count %u nonce 0x%08x", v, pre.x, pre.y, pre.count,
                         pre.nonce);
    v = ftq_pre_parse(entry, 164u, &pre);
    ft_tt_test_check_fmt("STALE: the entry of index - 64 is not index's", v == (uint32_t)kPreNone,
                         "verdict %u", v);
    v = ftq_pre_parse(entry, 36u, &pre);
    ft_tt_test_check_fmt("STALE: nor of index + 64", v == (uint32_t)kPreNone, "verdict %u", v);
    entry[kPreWordXy] ^= 0x10000u;
    v = ftq_pre_parse(entry, 100u, &pre);
    ft_tt_test_check_fmt("a foreign tag is no entry", v == (uint32_t)kPreNone, "verdict %u", v);
    put_entry(entry, 100u, 7u, 11u, 42u, 0x12345678u);
    entry[kPreWordXy] |= 0x40u;
    v = ftq_pre_parse(entry, 100u, &pre);
    ft_tt_test_check_fmt("x past six bits is refused", v == (uint32_t)kPreMalformed, "verdict %u",
                         v);
    put_entry(entry, 100u, 7u, 11u, 42u, 0x12345678u);
    entry[kPreWordXy] |= 0x8000u;
    v = ftq_pre_parse(entry, 100u, &pre);
    ft_tt_test_check_fmt("y past six bits is refused", v == (uint32_t)kPreMalformed, "verdict %u",
                         v);
    ft_tt_test_check_fmt("the encoder refuses a coordinate that does not fit",
                         ftq_pre_xy_word(64u, 1u) == 0u && ftq_pre_xy_word(1u, 64u) == 0u &&
                             ftq_pre_xy_word(63u, 63u) != 0u,
                         "0x%08x", ftq_pre_xy_word(63u, 63u));
    put_entry(entry, 100u, 7u, 11u, 0u, 0x12345678u);
    v = ftq_pre_parse(entry, 100u, &pre);
    ft_tt_test_check_fmt("a zero count is refused", v == (uint32_t)kPreMalformed, "verdict %u", v);
    put_entry(entry, 100u, 7u, 11u, 42u, 0u);
    v = ftq_pre_parse(entry, 100u, &pre);
    ft_tt_test_check_fmt("a zero nonce is refused", v == (uint32_t)kPreMalformed, "verdict %u", v);
    put_entry(entry, 100u, 7u, 11u, 42u, (uint32_t)kPrePoison);
    v = ftq_pre_parse(entry, 100u, &pre);
    ft_tt_test_check_fmt("the poison nonce is refused", v == (uint32_t)kPreMalformed, "verdict %u",
                         v);
}

static void test_compare(void)
{
    const uint32_t nonce = 0xC0FFEE11u;
    uint32_t entry[4], window[4];
    ftq_pre_entry pre;
    uint32_t v;

    ft_tt_test_step("compare: met only when the count is reached with its own token");
    put_entry(entry, 5u, 3u, 4u, 42u, nonce);
    (void)ftq_pre_parse(entry, 5u, &pre);
    put_window(window, 41u, host_token(nonce, 40u), 0x51010002u);
    v = ftq_pre_compare(&pre, window);
    ft_tt_test_check_fmt("ONE COUNT SHORT waits (the off-by-one the red arm accepts)",
                         v == (uint32_t)kPreWait, "count 41 of 42: verdict %u", v);
    put_window(window, 0u, (uint32_t)kPrePoison, 0x51010002u);
    v = ftq_pre_compare(&pre, window);
    ft_tt_test_check_fmt("nothing completed waits", v == (uint32_t)kPreWait, "verdict %u", v);
    put_window(window, 42u, host_token(nonce, 41u), 0x51010002u);
    v = ftq_pre_compare(&pre, window);
    ft_tt_test_check_fmt("the count with its token is met", v == (uint32_t)kPreMet, "verdict %u",
                         v);
    put_window(window, 45u, host_token(nonce, 44u), 0x51010003u);
    v = ftq_pre_compare(&pre, window);
    ft_tt_test_check_fmt("a producer that ran past it is met (its own token)",
                         v == (uint32_t)kPreMet, "count 45: verdict %u", v);
    put_window(window, 42u, host_token(nonce ^ 1u, 41u), 0x51010002u);
    v = ftq_pre_compare(&pre, window);
    ft_tt_test_check_fmt("WRONG TOKEN (another nonce) is refused", v == (uint32_t)kPreMismatch,
                         "verdict %u", v);
    put_window(window, 42u, host_token(nonce, 42u), 0x51010003u);
    v = ftq_pre_compare(&pre, window);
    ft_tt_test_check_fmt("the token-before-count window (token of 43 beside count 42) is not met",
                         v == (uint32_t)kPreMismatch, "verdict %u", v);
    put_window(window, 42u, host_token(nonce, 41u), (uint32_t)kPrePoison);
    v = ftq_pre_compare(&pre, window);
    ft_tt_test_check_fmt("a poisoned (re-armed) producer is gone", v == (uint32_t)kPreGone,
                         "verdict %u", v);
    put_entry(entry, 5u, 3u, 4u, 0xFFFFFFF0u, nonce);
    (void)ftq_pre_parse(entry, 5u, &pre);
    put_window(window, 5u, host_token(nonce, 4u), 0x51010002u);
    v = ftq_pre_compare(&pre, window);
    ft_tt_test_check_fmt("a count past the 32-bit wrap is met", v == (uint32_t)kPreMet,
                         "verdict %u", v);
    put_window(window, 0xFFFFFFEFu, host_token(nonce, 0xFFFFFFEEu), 0x51010002u);
    v = ftq_pre_compare(&pre, window);
    ft_tt_test_check_fmt("and one short of it before the wrap waits", v == (uint32_t)kPreWait,
                         "verdict %u", v);
}

int main(void)
{
#if defined(FTQ_PRE_HOST_RED_OFF_BY_ONE)
    printf("RED ARM: FTQ_PRE_HOST_RED_OFF_BY_ONE (the compare accepts one count early) — this "
           "suite must FAIL\n");
#endif
    test_layout();
    test_tokens();
    test_parse();
    test_compare();
    return ft_tt_test_finish();
}
