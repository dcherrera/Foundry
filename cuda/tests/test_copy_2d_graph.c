/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_copy_2d_graph.c
 * @brief fc_buffer_copy_2d_kernel_async moves strided blocks byte-exactly, directly and
 *        inside a CUDA graph captured across two dies.
 * @status complete
 * @issues none
 * @todo none
 *
 * Tensor-parallel decode captures both dies of a board pair into one graph: die B's stream
 * joins die A's capture through an event, pulls die A's rows with a 2-D copy, and rejoins.
 * The peer copy (fc_buffer_copy_2d_async, cuMemcpy3DPeerAsync) cannot be captured, and a
 * captured cross-die memcpy node is issued from the launching die (a push: on K80 ~78% of
 * the bytes landed). The kernel copy runs on the reader die. The test gates:
 *   - direct: same die, and die A -> die B / B -> A when the pair is peer-capable;
 *   - captured: a graph that pulls A's rows on B and B's rows on A (the TP gather shape,
 *     each die writing only its own buffer) replays byte-exactly, twice, with the source
 *     rewritten between replays (the graph re-reads, nothing is baked);
 *   - the peer copy inside a capture is reported (it is expected to be refused).
 *
 * Build: make USE_CUDA=1 test-copy-2d-graph
 * Run (two dies of one board): CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8,9 ...
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The GDN core gather's shape: 3 rows of 3 x 1 KiB runs into 8 KiB rows at +4 KiB. */
#define W   4096
#define H   9
#define SP  4096
#define DP  8192
#define OFF 4096

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

/* dst (DP x H, initially d0) must hold s's W-byte rows at OFF and d0 elsewhere. */
static int block_ok(const uint8_t *got, const uint8_t *s, const uint8_t *d0)
{
    for (size_t r = 0; r < H; r++)
        for (size_t i = 0; i < DP; i++) {
            uint8_t want = (i >= OFF && i < OFF + W) ? s[r * SP + i - OFF] : d0[r * DP + i];
            if (got[r * DP + i] != want)
                return 0;
        }
    return 1;
}

static int direct_ok(fc_cuda_ctx *src_ctx, fc_cuda_ctx *dst_ctx)
{
    uint8_t *s = malloc(SP * H), *d0 = malloc(DP * H), *got = malloc(DP * H);
    int ok = s && d0 && got;
    if (ok) {
        fill(s, SP * H);
        fill(d0, DP * H);
    }
    fc_buffer *bs = ok ? fc_buffer_from_ptr_ctx(src_ctx, s, SP * H) : NULL;
    fc_buffer *bd = ok ? fc_buffer_from_ptr_ctx(dst_ctx, d0, DP * H) : NULL;
    ok = bs && bd &&
         fc_buffer_copy_2d_kernel_async(bd, OFF, DP, bs, 0, SP, W, H, dst_ctx) == 0;
    /* Overflow refused, nothing queued. */
    ok = ok && fc_buffer_copy_2d_kernel_async(bd, OFF + 1, DP, bs, 0, SP, W, H, dst_ctx) == -1;
    if (ok) {
        fc_sync_ctx(dst_ctx);
        fc_buffer_to_cpu(bd, got);
        ok = block_ok(got, s, d0);
    }
    fc_buffer_free(bs);
    fc_buffer_free(bd);
    free(s); free(d0); free(got);
    return ok;
}

static void test_direct(void)
{
    fc_cuda_ctx *a = fc_default_ctx();
    ASSERT_TRUE(direct_ok(a, a));
    fc_cuda_ctx *b = fc_cuda_init_device(1);
    if (!b) {
        printf("    one die visible: cross-die arm skipped\n");
        return;
    }
    if (fc_cuda_enable_peer(a, b) != 0) {
        printf("    dies 0/1 not peer-capable: cross-die arm skipped\n");
    } else {
        ASSERT_TRUE(direct_ok(a, b));
        ASSERT_TRUE(direct_ok(b, a));
        printf("    die-to-die kernel 2-D copy verified both ways\n");
    }
    fc_cuda_ctx_free(b);
}

/* One graph over both dies: each pulls the other's rows into its own buffer. */
static void test_graph(void)
{
    fc_cuda_ctx *a = fc_default_ctx();
    fc_cuda_ctx *b = fc_cuda_init_device(1);
    if (!b || fc_cuda_enable_peer(a, b) != 0) {
        printf("    no peer-capable second die: graph arm skipped\n");
        fc_cuda_ctx_free(b);
        return;
    }
    uint8_t *sa = malloc(SP * H), *sb = malloc(SP * H), *d0 = malloc(DP * H);
    uint8_t *got = malloc(DP * H);
    ASSERT_TRUE(sa && sb && d0 && got);
    if (!sa || !sb || !d0 || !got)
        goto out_host;
    fill(sa, SP * H);
    fill(sb, SP * H);
    fill(d0, DP * H);
    fc_buffer *bsa = fc_buffer_from_ptr_ctx(a, sa, SP * H);
    fc_buffer *bsb = fc_buffer_from_ptr_ctx(b, sb, SP * H);
    fc_buffer *bda = fc_buffer_from_ptr_ctx(a, d0, DP * H);
    fc_buffer *bdb = fc_buffer_from_ptr_ctx(b, d0, DP * H);
    fc_event *ea = fc_event_create_ctx(a), *eb = fc_event_create_ctx(b);
    ASSERT_TRUE(bsa && bsb && bda && bdb && ea && eb);
    if (!bsa || !bsb || !bda || !bdb || !ea || !eb)
        goto out;

    /* The peer copy cannot be captured (reported, not gated). */
    if (fc_graph_begin_ctx(a) == 0) {
        int pr = fc_buffer_copy_2d_async(bda, OFF, DP, bsa, 0, SP, W, H, a);
        fc_graph *pg = fc_graph_end_ctx(a);
        printf("    peer 2-D copy inside a capture: %s\n",
               pr == 0 && pg ? "captured" : "refused (expected)");
        fc_graph_free(pg);
    }

    int cap = fc_graph_begin_ctx(a) == 0;
    ASSERT_TRUE(cap);
    if (!cap)
        goto out;
    int ok = fc_event_record_ctx(a, ea) == 0 && fc_event_wait_ctx(b, ea) == 0;
    ok = ok && fc_buffer_copy_2d_kernel_async(bdb, OFF, DP, bsa, 0, SP, W, H, b) == 0;
    ok = ok && fc_buffer_copy_2d_kernel_async(bda, OFF, DP, bsb, 0, SP, W, H, a) == 0;
    ok = fc_event_record_ctx(b, eb) == 0 && fc_event_wait_ctx(a, eb) == 0 && ok;
    fc_graph *gr = fc_graph_end_ctx(a);
    ASSERT_TRUE(ok && gr);
    if (!ok || !gr) {
        fc_graph_free(gr);
        goto out;
    }
    for (int rep = 0; rep < 2; rep++) {
        if (rep) {   /* new source bytes: the replay must re-read them */
            fill(sa, SP * H);
            fill(sb, SP * H);
            fc_buffer_from_cpu(bsa, sa);
            fc_buffer_from_cpu(bsb, sb);
        }
        ASSERT_TRUE(fc_graph_launch(gr) == 0);
        fc_sync_ctx(a);
        fc_buffer_to_cpu(bdb, got);
        if (!block_ok(got, sa, d0)) {
            size_t moved = 0;
            for (size_t r = 0; r < H; r++)
                for (size_t i = OFF; i < OFF + W; i++)
                    moved += got[r * DP + i] == sa[r * SP + i - OFF];
            fc_sync_ctx(b);
            fc_buffer_to_cpu(bdb, got);
            printf("    replay %d: die B window bytes moved %zu of %d after die A sync; "
                   "after die B sync too: %s\n", rep, moved, W * H,
                   block_ok(got, sa, d0) ? "complete" : "still wrong");
        }
        ASSERT_TRUE(block_ok(got, sa, d0));
        fc_buffer_to_cpu(bda, got);
        ASSERT_TRUE(block_ok(got, sb, d0));
    }
    printf("    two-die captured gather replayed byte-exactly twice\n");
    fc_graph_free(gr);
out:
    fc_event_free(ea);
    fc_event_free(eb);
    fc_buffer_free(bsa); fc_buffer_free(bsb); fc_buffer_free(bda); fc_buffer_free(bdb);
out_host:
    free(sa); free(sb); free(d0); free(got);
    fc_cuda_ctx_free(b);
}

int main(void)
{
    printf("Capturable 2-D Copy Tests\n");
    printf("=========================\n");
    if (fc_init() != 0) {
        printf("  [SKIP] no CUDA device\n");
        TEST_REPORT();
        return __ft_failed > 0 ? 1 : 0;
    }
    RUN_TEST(direct);
    RUN_TEST(graph);
    fc_free();
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
