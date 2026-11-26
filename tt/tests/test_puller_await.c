/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_puller_await.c
 * @brief The puller's AWAIT off hardware: the KERNEL'S OWN poll step
 *        (kernels/resident_puller_abi.h ft_tt_puller_await_step, the inline resident_puller.c
 *        runs) against an emulated resident writer that publishes a completion the way every
 *        resident writer does — the token store, then the count store.
 *
 * A puller AWAIT that refuses a live queue kills the whole stream: the PULL never completes,
 * every core it feeds passes its idle exit, and the request fails. On the card that read as
 * error 4 (kPlrErrAwaitToken) on a step-mark AWAIT: wanted 1812, count read 1812, the core at
 * 1814 a moment later. The emulation puts a puller read between every pair of writer stores:
 *   - TORN: an AWAIT that names an older count while its core still completes later
 *     descriptors (the step mark) is met at every interleaving, including the read between a
 *     token store and its count store (count C beside descriptor C's token);
 *   - the same with the card's numbers (nonce 36a93ac3, wanted 1812, core 1812 -> 1814);
 *   - a core slower than the puller (many polls a store), and an AWAIT on the core's last
 *     count (nothing after it to tear) are met;
 *   - FOREIGN: a queue of another nonce (re-armed under the stream) is never met and refuses
 *     after exactly FT_TT_PULLER_AWAIT_MISMATCH_LIMIT polls;
 *   - a poisoned state refuses; a short count keeps polling;
 *   - the inline token is ft_tt_queue_token_at(), fallbacks included.
 *
 * RED (make -f Makefile.pullerawait red): the same suite built with
 * FT_TT_PULLER_AWAIT_HOST_RED_NO_RETRY, the rule before the retry (the first mismatch
 * refuses), must FAIL.
 *
 *   make -f Makefile.pullerawait run
 *
 * @status Added with the step-mark AWAIT retry.
 * @issues None
 * @todo None
 */
#include <stdio.h>
#include <string.h>

#include "ft_tt_queue.h"
#include "ft_tt_test_report.h"
#include "kernels/resident_puller_abi.h"

#define CARD_NONCE 0x36a93ac3u
#define POLL_BOUND 100000u

/** An emulated resident writer's control words 4..7 and its completion stores. */
typedef struct {
    uint32_t line[4]; /* stop, completed count, completed token, state */
    uint32_t nonce;
    uint32_t last;    /* the count it stops at (its last issued descriptor) */
    uint32_t stores;  /* stores done since the start */
} emulated_writer;

/** The token a host queue with `nonce` gives descriptor `index`. */
static uint32_t host_token(uint32_t nonce, uint32_t index)
{
    ft_tt_queue queue;

    memset(&queue, 0, sizeof(queue));
    queue.run_nonce = nonce;
    return ft_tt_queue_token_at(&queue, index);
}

static void writer_start(emulated_writer *w, uint32_t nonce, uint32_t count, uint32_t last)
{
    memset(w, 0, sizeof(*w));
    w->nonce = nonce;
    w->last = last;
    w->line[1] = count;
    w->line[2] = count ? host_token(nonce, count - 1u) : 0x7B1B1B1Bu;
    w->line[3] = 0x51010003u; /* EXEC */
}

/** One store of the writer's completion (resident_matmul_writer.c: token, then count).
 *  @return false once the writer has published `last`. */
static bool writer_store(emulated_writer *w)
{
    const uint32_t count = w->line[1];

    if (count >= w->last)
        return false;
    if (w->line[2] != host_token(w->nonce, count))
        w->line[2] = host_token(w->nonce, count); /* token of descriptor `count` */
    else
        w->line[1] = count + 1u;                  /* then the count it completes */
    w->stores++;
    return true;
}

/** Is the writer between its token store and its count store? */
static bool writer_torn(const emulated_writer *w)
{
    return w->line[1] < w->last && w->line[2] == host_token(w->nonce, w->line[1]);
}

typedef struct {
    uint32_t step;      /* the final verdict */
    uint32_t polls;
    uint32_t torn_reads; /* reads of a torn line whose count had reached `wanted` */
    uint32_t count_read;
} await_result;

/**
 * The puller's AWAIT loop (resident_puller.c entry 4) against the emulated writer: read the
 * line, take the step, and let the writer do `stores_per_poll` stores between two reads
 * (0 = one store every `polls_per_store` polls).
 */
static await_result run_await(emulated_writer *w, uint32_t wanted, uint32_t nonce,
                              uint32_t stores_per_poll, uint32_t polls_per_store)
{
    await_result r = { FT_TT_PULLER_AWAIT_POLL, 0u, 0u, 0u };
    uint32_t mismatches = 0u;

    while (r.polls < POLL_BOUND) {
        uint32_t read[4];

        memcpy(read, w->line, sizeof(read));
        r.torn_reads += writer_torn(w) && read[1] >= wanted ? 1u : 0u;
        r.polls++;
        r.step = ft_tt_puller_await_step(read, wanted, nonce, &mismatches);
        r.count_read = read[1];
        if (r.step != FT_TT_PULLER_AWAIT_POLL)
            break;
        if (stores_per_poll > 0u) {
            for (uint32_t s = 0; s < stores_per_poll; s++)
                (void)writer_store(w);
        } else if (r.polls % polls_per_store == 0u) {
            (void)writer_store(w);
        }
    }
    return r;
}

static const char *step_name(uint32_t step)
{
    switch (step) {
    case FT_TT_PULLER_AWAIT_POLL:
        return "POLL";
    case FT_TT_PULLER_AWAIT_MET:
        return "MET";
    case FT_TT_PULLER_AWAIT_POISON:
        return "POISON";
    case FT_TT_PULLER_AWAIT_TOKEN:
        return "TOKEN (kPlrErrAwaitToken)";
    default:
        return "?";
    }
}

static void test_tokens(void)
{
    static const uint32_t nonces[] = { 1u, CARD_NONCE, 0x9E3779B9u, 0xA5A5A5A5u, 0xFFFFFFFFu };
    uint32_t bad = 0u;

    ft_tt_test_step("tokens: the inline token is ft_tt_queue_token_at()");
    for (size_t n = 0; n < sizeof(nonces) / sizeof(nonces[0]); n++)
        for (uint32_t index = 0; index < 4096u; index++)
            bad += ft_tt_puller_token(nonces[n], index) != host_token(nonces[n], index);
    /* index 0 with nonce 0x9E3779B9 makes 0: the fallback, on both sides */
    bad += ft_tt_puller_token(0x9E3779B9u, 0u) != host_token(0x9E3779B9u, 0u);
    ft_tt_test_check_fmt("20481 tokens agree, the zero fallback included", bad == 0u,
                         "%u differ", bad);
}

/** Every interleaving of an AWAIT on `wanted` while the core runs on from `from` to `last`. */
static void sweep_step_mark(const char *what, uint32_t nonce, uint32_t wanted, uint32_t from,
                            uint32_t last)
{
    emulated_writer w;
    uint32_t phases = 0u, met = 0u, torn = 0u, refused_at = UINT32_MAX;
    await_result first_refusal = { 0u, 0u, 0u, 0u };

    writer_start(&w, nonce, from, last);
    while (writer_store(&w))
        phases++;
    for (uint32_t phase = 0; phase <= phases; phase++) {
        await_result r;

        writer_start(&w, nonce, from, last);
        for (uint32_t s = 0; s < phase; s++)
            (void)writer_store(&w);
        r = run_await(&w, wanted, nonce, 1u, 0u);
        torn += r.torn_reads;
        if (r.step == FT_TT_PULLER_AWAIT_MET && r.count_read >= wanted)
            met++;
        else if (refused_at == UINT32_MAX) {
            refused_at = phase;
            first_refusal = r;
        }
    }
    if (refused_at == UINT32_MAX)
        ft_tt_test_check_fmt(what, met == phases + 1u && torn > 0u,
                             "%u of %u interleavings met, %u torn reads at the count", met,
                             phases + 1u, torn);
    else
        ft_tt_test_check_fmt(what, false,
                             "%u of %u met; interleaving %u refused: %s, count read %u", met,
                             phases + 1u, refused_at, step_name(first_refusal.step),
                             first_refusal.count_read);
}

static void test_torn_line(void)
{
    ft_tt_test_step("torn line: a step-mark AWAIT while its core runs on");
    sweep_step_mark("AWAIT 5, core 3 -> 9: met at every interleaving", 7u, 5u, 3u, 9u);
    sweep_step_mark("AWAIT 5, core already 5 -> 9 (count past wanted): met", 7u, 5u, 5u, 9u);
    sweep_step_mark("card: nonce 36a93ac3, AWAIT 1812, core 1812 -> 1814: met", CARD_NONCE,
                    1812u, 1812u, 1814u);
    {
        /* The exact line the card refused: count 1812 beside descriptor 1812's token. */
        uint32_t line[4] = { 0u, 1812u, host_token(CARD_NONCE, 1812u), 0x51010003u };
        uint32_t mismatches = 0u;
        const uint32_t step = ft_tt_puller_await_step(line, 1812u, CARD_NONCE, &mismatches);

        ft_tt_test_check_fmt("the card's torn line (1812, token of 1812) polls again",
                             step == FT_TT_PULLER_AWAIT_POLL && mismatches == 1u, "%s, %u",
                             step_name(step), mismatches);
        line[1] = 1813u; /* the count store lands */
        ft_tt_test_check_fmt("... and the next read (1813, token of 1812) is met",
                             ft_tt_puller_await_step(line, 1812u, CARD_NONCE, &mismatches) ==
                                 FT_TT_PULLER_AWAIT_MET,
                             "count 1813");
    }
}

static void test_slow_core_and_last(void)
{
    emulated_writer w;
    await_result r;

    ft_tt_test_step("slow core, last count");
    writer_start(&w, 11u, 2u, 9u);
    r = run_await(&w, 6u, 11u, 0u, 37u);
    ft_tt_test_check_fmt("core 37x slower than the puller, AWAIT 6: met at 6",
                         r.step == FT_TT_PULLER_AWAIT_MET && r.count_read == 6u,
                         "%s at count %u after %u polls, %u torn reads", step_name(r.step),
                         r.count_read, r.polls, r.torn_reads);
    writer_start(&w, 11u, 2u, 6u);
    r = run_await(&w, 6u, 11u, 1u, 0u);
    ft_tt_test_check_fmt("AWAIT on the core's last count: met at 6",
                         r.step == FT_TT_PULLER_AWAIT_MET && r.count_read == 6u, "%s at %u",
                         step_name(r.step), r.count_read);
    writer_start(&w, 11u, 2u, 4u);
    r = run_await(&w, 6u, 11u, 1u, 0u);
    ft_tt_test_check_fmt("a core that stops short of the AWAIT keeps it polling",
                         r.step == FT_TT_PULLER_AWAIT_POLL && r.polls == POLL_BOUND,
                         "%s after %u polls", step_name(r.step), r.polls);
}

static void test_foreign_and_poison(void)
{
    emulated_writer w;
    await_result r;
    uint32_t line[4] = { 0u, 9u, host_token(11u, 8u), 0x7B1B1B1Bu };
    uint32_t mismatches = 0u;

    ft_tt_test_step("foreign queue, poison");
    writer_start(&w, 12u, 3u, 9u); /* the core was re-armed: nonce 12, the stream knows 11 */
    r = run_await(&w, 5u, 11u, 1u, 0u);
    ft_tt_test_check_fmt("a queue of another nonce is never met",
                         r.step == FT_TT_PULLER_AWAIT_TOKEN, "%s", step_name(r.step));
    writer_start(&w, 12u, 9u, 9u); /* idle: nothing moves, every poll mismatches */
    r = run_await(&w, 5u, 11u, 1u, 0u);
    ft_tt_test_check_fmt("an idle foreign queue refuses after exactly the mismatch limit",
                         r.step == FT_TT_PULLER_AWAIT_TOKEN &&
                             r.polls == FT_TT_PULLER_AWAIT_MISMATCH_LIMIT,
                         "%s after %u polls (limit %u)", step_name(r.step), r.polls,
                         FT_TT_PULLER_AWAIT_MISMATCH_LIMIT);
    ft_tt_test_check_fmt("a poisoned state refuses",
                         ft_tt_puller_await_step(line, 5u, 11u, &mismatches) ==
                             FT_TT_PULLER_AWAIT_POISON,
                         "state 0x%x", line[3]);
}

int main(void)
{
#if defined(FT_TT_PULLER_AWAIT_HOST_RED_NO_RETRY)
    printf("RED ARM: FT_TT_PULLER_AWAIT_HOST_RED_NO_RETRY (the first mismatch refuses) — this "
           "suite must FAIL\n");
#endif
    test_tokens();
    test_torn_line();
    test_slow_core_and_last();
    test_foreign_and_poison();
    return ft_tt_test_finish();
}
