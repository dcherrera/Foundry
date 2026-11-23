/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_xchg_2d.c
 * @brief fc_xchg_2d_async: a two-die exchange ordered by device counters moves both
 *        halves byte-exactly, waits for the partner, and replays inside a CUDA graph.
 * @status complete
 * @issues none
 * @todo none
 *
 * Tensor-parallel decode gathers two halves of a row on both dies of a board pair. The
 * exchange is one launch per die: copy the own rows, wait until the partner's launch for
 * the same slot has started (its sig counter), pull the partner's rows. The test gates:
 *   - direct: 4 slots x 50 rounds, host-synced, new sources every round; both dies hold
 *     [A | B] rows each round; every slot's counters end at 50;
 *   - waits: die B launched alone does not finish (its pull half untouched) until die A
 *     launches (the sensitivity proof: without the wait it would finish at once);
 *   - unsynced: 400 launches issued back to back on both dies with no host sync
 *     (8 rotating buffer sets), all correct at the end;
 *   - graph: 8 exchanges over both dies captured with no cross-die edge but the fork and
 *     join, replayed 3 times with new sources (counters advance; nothing baked);
 *   - refusals: misaligned width, a dst not owned by the stream's die.
 *
 * Build: make USE_CUDA=1 test-xchg-2d
 * Run (two dies of one board): CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8,9 ...
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The E3 hid gather at n = 1: 9 rows of 1 KiB halves into 2 KiB rows. */
#define W     1024
#define H     9
#define SP    W
#define DP    (2 * W)
#define SLOTS 4
#define SETS  8

static uint32_t rng_state = 2463534242u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static void fill(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)rng();
}

/* dst rows must be [a row | b row]. */
static int rows_ok(const uint8_t *got, const uint8_t *a, const uint8_t *b)
{
    for (size_t r = 0; r < H; r++)
        if (memcmp(got + r * DP, a + r * SP, W) || memcmp(got + r * DP + W, b + r * SP, W))
            return 0;
    return 1;
}

typedef struct {
    fc_cuda_ctx *a, *b;
    fc_buffer *sa[SETS], *sb[SETS], *da[SETS], *db[SETS];
    fc_buffer *ya, *yb;   /* sync words, SLOTS x FC_XCHG_SYNC_BYTES each */
    uint8_t *ha[SETS], *hb[SETS];
} pair;

static void pair_free(pair *p)
{
    for (int i = 0; i < SETS; i++) {
        fc_buffer_free(p->sa[i]); fc_buffer_free(p->sb[i]);
        fc_buffer_free(p->da[i]); fc_buffer_free(p->db[i]);
        free(p->ha[i]); free(p->hb[i]);
    }
    fc_buffer_free(p->ya);
    fc_buffer_free(p->yb);
    fc_cuda_ctx_free(p->b);
    memset(p, 0, sizeof(*p));
}

static int pair_init(pair *p)
{
    static const uint8_t zero[SLOTS * FC_XCHG_SYNC_BYTES];
    uint8_t *d0 = calloc(1, DP * H);
    memset(p, 0, sizeof(*p));
    p->a = fc_default_ctx();
    p->b = fc_cuda_init_device(1);
    if (!d0 || !p->b || fc_cuda_enable_peer(p->a, p->b) != 0) {
        free(d0);
        fc_cuda_ctx_free(p->b);
        p->b = NULL;
        return 0;
    }
    int ok = 1;
    for (int i = 0; ok && i < SETS; i++) {
        p->ha[i] = malloc(SP * H);
        p->hb[i] = malloc(SP * H);
        ok = p->ha[i] && p->hb[i];
        if (ok) {
            fill(p->ha[i], SP * H);
            fill(p->hb[i], SP * H);
        }
        p->sa[i] = ok ? fc_buffer_from_ptr_ctx(p->a, p->ha[i], SP * H) : NULL;
        p->sb[i] = ok ? fc_buffer_from_ptr_ctx(p->b, p->hb[i], SP * H) : NULL;
        p->da[i] = ok ? fc_buffer_from_ptr_ctx(p->a, d0, DP * H) : NULL;
        p->db[i] = ok ? fc_buffer_from_ptr_ctx(p->b, d0, DP * H) : NULL;
        ok = p->sa[i] && p->sb[i] && p->da[i] && p->db[i];
    }
    p->ya = ok ? fc_buffer_from_ptr_ctx(p->a, zero, sizeof(zero)) : NULL;
    p->yb = ok ? fc_buffer_from_ptr_ctx(p->b, zero, sizeof(zero)) : NULL;
    free(d0);
    return ok && p->ya && p->yb ? 1 : -1;
}

/* Exchange set i through slot s on both dies (A's half at 0, B's at W). */
static int xchg(pair *p, int i, int s)
{
    return fc_xchg_2d_async(p->da[i], 0, W, DP, p->sa[i], p->sb[i], SP, W, H, p->ya, p->yb,
                            s, p->a) == 0 &&
           fc_xchg_2d_async(p->db[i], W, 0, DP, p->sb[i], p->sa[i], SP, W, H, p->yb, p->ya,
                            s, p->b) == 0;
}

static int set_ok(pair *p, int i)
{
    uint8_t *got = malloc(DP * H);
    int ok = got != NULL;
    if (ok) {
        fc_buffer_to_cpu(p->da[i], got);
        ok = rows_ok(got, p->ha[i], p->hb[i]);
        fc_buffer_to_cpu(p->db[i], got);
        ok = ok && rows_ok(got, p->ha[i], p->hb[i]);
    }
    free(got);
    return ok;
}

static void new_sources(pair *p, int i)
{
    fill(p->ha[i], SP * H);
    fill(p->hb[i], SP * H);
    fc_buffer_from_cpu(p->sa[i], p->ha[i]);
    fc_buffer_from_cpu(p->sb[i], p->hb[i]);
}

/* Counters of every slot on both dies: sig = cnt = n, ticket 0. */
static int counters_are(pair *p, unsigned int n)
{
    unsigned int w[SLOTS * FC_XCHG_SYNC_BYTES / 4];
    for (int d = 0; d < 2; d++) {
        fc_buffer_to_cpu(d ? p->yb : p->ya, w);
        for (int s = 0; s < SLOTS; s++)
            if (w[4 * s] != n || w[4 * s + 1] != n || w[4 * s + 2] != 0)
                return 0;
    }
    return 1;
}

static void test_direct(void)
{
    pair p;
    int r = pair_init(&p);
    if (!r) {
        printf("    no peer-capable second die: skipped\n");
        return;
    }
    ASSERT_TRUE(r == 1);
    int ok = r == 1;
    for (int round = 0; ok && round < 50; round++)
        for (int s = 0; ok && s < SLOTS; s++) {
            new_sources(&p, s);
            ok = xchg(&p, s, s);
            fc_sync_ctx(p.a);
            fc_sync_ctx(p.b);
            ok = ok && set_ok(&p, s);
        }
    ASSERT_TRUE(ok);
    ASSERT_TRUE(counters_are(&p, 50));
    /* Refusals: misaligned width; dst owned by the partner. */
    ASSERT_TRUE(fc_xchg_2d_async(p.da[0], 0, W, DP, p.sa[0], p.sb[0], SP, W - 4, H, p.ya,
                                 p.yb, 0, p.a) == -1);
    ASSERT_TRUE(fc_xchg_2d_async(p.db[0], 0, W, DP, p.sa[0], p.sb[0], SP, W, H, p.ya, p.yb,
                                 0, p.a) == -1);
    printf("    4 slots x 50 rounds byte-exact on both dies; counters at 50\n");
    pair_free(&p);
}

static void test_waits(void)
{
    pair p;
    int r = pair_init(&p);
    if (!r) {
        printf("    no peer-capable second die: skipped\n");
        return;
    }
    ASSERT_TRUE(r == 1);
    fc_event *eb = fc_event_create_ctx(p.b);
    ASSERT_TRUE(eb != NULL);
    if (r != 1 || !eb) {
        fc_event_free(eb);
        pair_free(&p);
        return;
    }
    ASSERT_TRUE(fc_xchg_2d_async(p.db[0], W, 0, DP, p.sb[0], p.sa[0], SP, W, H, p.yb, p.ya,
                                 0, p.b) == 0);
    ASSERT_TRUE(fc_event_record_ctx(p.b, eb) == 0);
    struct timespec ts = { 0, 200 * 1000 * 1000 };
    nanosleep(&ts, NULL);
    int pending = fc_event_query(eb);   /* 1 = still waiting for die A */
    printf("    die B alone after 200 ms: %s\n", pending == 1 ? "waiting" : "finished");
    ASSERT_TRUE(pending == 1);
    ASSERT_TRUE(fc_xchg_2d_async(p.da[0], 0, W, DP, p.sa[0], p.sb[0], SP, W, H, p.ya, p.yb,
                                 0, p.a) == 0);
    fc_sync_ctx(p.a);
    fc_sync_ctx(p.b);
    ASSERT_TRUE(set_ok(&p, 0));
    fc_event_free(eb);
    pair_free(&p);
}

static void test_unsynced(void)
{
    pair p;
    int r = pair_init(&p);
    if (!r) {
        printf("    no peer-capable second die: skipped\n");
        return;
    }
    ASSERT_TRUE(r == 1);
    int ok = r == 1;
    for (int i = 0; ok && i < 400; i++)
        ok = xchg(&p, i % SETS, i % SLOTS);
    fc_sync_ctx(p.a);
    fc_sync_ctx(p.b);
    for (int i = 0; ok && i < SETS; i++)
        ok = set_ok(&p, i);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(counters_are(&p, 100));
    printf("    400 unsynced launches a die byte-exact; counters at 100\n");
    pair_free(&p);
}

static void test_graph(void)
{
    pair p;
    int r = pair_init(&p);
    if (!r) {
        printf("    no peer-capable second die: skipped\n");
        return;
    }
    ASSERT_TRUE(r == 1);
    fc_event *ea = fc_event_create_ctx(p.a), *eb = fc_event_create_ctx(p.b);
    int ok = r == 1 && ea && eb && fc_graph_begin_ctx(p.a) == 0;
    ASSERT_TRUE(ok);
    if (!ok) {
        fc_event_free(ea);
        fc_event_free(eb);
        pair_free(&p);
        return;
    }
    ok = fc_event_record_ctx(p.a, ea) == 0 && fc_event_wait_ctx(p.b, ea) == 0;
    for (int i = 0; ok && i < SETS; i++)
        ok = xchg(&p, i, i % SLOTS);
    ok = fc_event_record_ctx(p.b, eb) == 0 && fc_event_wait_ctx(p.a, eb) == 0 && ok;
    fc_graph *gr = fc_graph_end_ctx(p.a);
    ASSERT_TRUE(ok && gr);
    for (int rep = 0; ok && gr && rep < 3; rep++) {
        for (int i = 0; i < SETS; i++)
            new_sources(&p, i);
        ASSERT_TRUE(fc_graph_launch(gr) == 0);
        fc_sync_ctx(p.a);
        fc_sync_ctx(p.b);
        for (int i = 0; i < SETS; i++)
            ASSERT_TRUE(set_ok(&p, i));
    }
    ASSERT_TRUE(counters_are(&p, 6));
    printf("    8-exchange two-die graph replayed byte-exactly 3 times; counters at 6\n");
    fc_graph_free(gr);
    fc_event_free(ea);
    fc_event_free(eb);
    pair_free(&p);
}

int main(void)
{
    printf("Two-Die Exchange Tests\n");
    printf("======================\n");
    if (fc_init() != 0) {
        printf("  [SKIP] no CUDA device\n");
        TEST_REPORT();
        return __ft_failed > 0 ? 1 : 0;
    }
    RUN_TEST(direct);
    RUN_TEST(waits);
    RUN_TEST(unsynced);
    RUN_TEST(graph);
    fc_free();
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
