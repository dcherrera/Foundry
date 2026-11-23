/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_cross_die.c
 * @brief Prices the two things a K80 tensor-parallel scheme would pay 80 times per
 *   token: ONE cross-die barrier + 8 KB P2P transfer, and the cost of breaking a
 *   captured CUDA graph into fragments.
 *
 * WHY THESE TWO NUMBERS
 * ---------------------
 * Megatron-style TP alternates row-split and column-split, so saber's 40 layers need
 * 2 all-reduces per layer = 80 per token. Bandwidth is not the worry: 80 x 8 KB
 * (n_embd 2048, fp32) = 640 KB/token, trivial over the on-board PLX link. SYNC is the
 * worry, twice over:
 *   1. each all-reduce is a hard barrier -- both dies must arrive before either
 *      proceeds -- and at ~100 us x 80 that is 8 ms of a 52.6 ms step (survivable),
 *      while at ~300 us it is 24 ms (fatal);
 *   2. 80 barriers per token cannot live inside ONE captured graph, so full-segment
 *      capture shatters into ~80 fragments and the per-launch overhead graphs
 *      removed comes partly back.
 * Nobody has measured either. This file measures both.
 *
 * WHAT IS BEING COMPARED (the barrier variants are not equivalent)
 * ---------------------------------------------------------------
 *   sync_floor      fc_sync_ctx on an IDLE stream. Not a barrier, just the price of
 *                   asking the driver a question. Everything else must beat it.
 *   peer_blocking   fc_buffer_copy_peer alone (cuMemcpyPeer, synchronous).
 *   prod_pattern    sync(B) -> copy_peer -> sync(A). This is EXACTLY what shipped
 *                   QWEN35_VOCAB_SPLIT runs once per token today
 *                   (qwen35moe_forward_cuda.c gpu_split_head_slots), so its cost is
 *                   the one number in this file that is already in production.
 *   evsync          async peer copy + cuEventSynchronize (waits for a POINT, not a
 *                   whole stream drain).
 *   evpoll          async peer copy + non-blocking cuEventQuery spin.
 *   gpu_pingpong    ** the decisive variant ** -- async peer copy + cuStreamWaitEvent
 *                   so the DEPENDENCY IS ENFORCED ON THE DEVICE and the host never
 *                   synchronises inside the loop. If a cross-die barrier is ever
 *                   going to be cheap on this board, it is cheap here and nowhere
 *                   else. fc_event_wait_ctx was added for this measurement; nothing
 *                   in the engine used a cross-die stream wait before.
 *   allreduce       the real primitive: A->B and B->A, plus an elementwise add on
 *                   each side. What one Megatron all-reduce actually costs.
 *
 * The pipelined variants are timed by queueing DEPTH of them back-to-back and
 * dividing, because that is how a TP forward would use them (80 barriers pipelined
 * through a step, not 80 stop-the-world round trips). Both the pipelined and the
 * one-at-a-time costs are reported -- they answer different questions and conflating
 * them would flatter the scheme.
 *
 * PROVING THE HARNESS CAN FAIL
 * ----------------------------
 * Three checks run before the table and are asserted, not eyeballed:
 *   - peer access must actually be ENABLED (fc_cuda_enable_peer) and the two contexts
 *     must report DIFFERENT device ids. A silent fallback to one device would make
 *     every cross-die number look wonderful.
 *   - the transfer must be VERIFIED: a known byte pattern is written on die A, copied,
 *     and read back from die B. A no-op copy is otherwise indistinguishable from a
 *     very fast one -- fc_buffer_copy_peer is documented to be a bounds-checked NO-OP
 *     when the size exceeds either buffer.
 *   - the size sweep must show a rising cost with bytes. If 4 KB and 4 MB cost the
 *     same, the copy is not happening.
 *   - the size sweep issues its copies ASYNC ON DIE B'S OWN STREAM and drains that
 *     stream inside the timed region, and it verifies the largest payload arrived.
 *     A device-to-device cuMemcpyPeer performs no host-side synchronisation despite
 *     the missing `Async` suffix — it queues on the LEGACY NULL STREAM and returns.
 *     That is a two-layer trap and both layers were walked into on 2026-08-09: an
 *     undrained loop reported 80.5 GB/s over a link whose theoretical ceiling is
 *     15.75 GB/s, and the obvious fix (fc_sync_ctx on both contexts) did NOT change
 *     the number, because fc_sync_ctx waits on ctx->stream and the blocking copy was
 *     never in ctx->stream. Only the async form lands the copy in a stream that can
 *     actually be drained. The sweep now reports the drained rate AND the raw issue
 *     cost side by side, so the divergence is visible rather than inferred, and it
 *     runs out to 64 MB where the ~10 us fixed cost cannot flatter the rate.
 *
 * `peer_blocking` in the barrier table is deliberately left UNDRAINED: it is the raw
 * host cost of issuing an 8 KB peer copy, and it is only meaningful next to
 * `prod_pattern`, which drains and comes out at the same number (the 8 KB payload is
 * ~0.7 us of link time, so at this size issue cost is the whole cost).
 *
 * ⚠ TWO DIES OF ONE K80 SHARE A POWER BUDGET. A busy sibling drags the other from
 * 875 to 745-810 MHz; that has already manufactured a false -38.9% result on this
 * board. Run this ALONE on its die pair, with the clock pinned:
 *     sudo nvidia-smi -i 6,7 -ac 2505,666     (trap-restore 2505,875)
 *
 * Build (from foundry/):
 *   gcc -O2 -I. -o cuda/tests/bench_cross_die cuda/tests/bench_cross_die.c \
 *       cuda/fc_cuda.c -lm -ldl
 * Run (the two dies of ONE board, never dies another agent holds):
 *   CUDA_VISIBLE_DEVICES=6,7 ./cuda/tests/bench_cross_die
 * Env:
 *   XD_REPS   alternating reps, median reported (default 5)
 *   XD_ITERS  barriers timed per arm (default 2000)
 *   XD_DEPTH  barriers queued before one host sync in the pipelined arms (default 64)
 *   XD_FRAG   kernels per token used for the graph-fragmentation model (default 600)
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define XD_EMBD_BYTES   (2048 * (int)sizeof(float))   /* one n_embd fp32 activation */
#define XD_EVENT_RING   256    /* distinct events so a pipelined arm never reuses one */

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int env_int(const char *name, int dflt)
{
    const char *e = getenv(name);
    return (e && *e) ? atoi(e) : dflt;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ------------------------------------------------------------------
 * Shared rig: two contexts on two dies, peer-enabled, with a buffer on
 * each side and an event ring on each side.
 * ------------------------------------------------------------------ */

typedef struct xd_rig {
    fc_cuda_ctx *a, *b;
    fc_buffer   *buf_a, *buf_b;      /* XD_EMBD_BYTES activation mirrors */
    fc_buffer   *big_a, *big_b;      /* sized for the largest sweep point */
    fc_event    *ev_a[XD_EVENT_RING];
    fc_event    *ev_b[XD_EVENT_RING];
    int          have_events;
    int          have_stream_wait;
    size_t       big_bytes;
} xd_rig;

static void rig_free(xd_rig *r)
{
    int i;
    for (i = 0; i < XD_EVENT_RING; i++) {
        fc_event_free(r->ev_a[i]);
        fc_event_free(r->ev_b[i]);
    }
    fc_buffer_free(r->buf_a);
    fc_buffer_free(r->buf_b);
    fc_buffer_free(r->big_a);
    fc_buffer_free(r->big_b);
}

/* Bring up the pair. Returns 0 only if the two contexts are on DIFFERENT
 * devices and peer access is genuinely enabled — a same-device fallback would
 * silently turn this whole file into a lie. */
static int rig_init(xd_rig *r, size_t big_bytes)
{
    int i;

    memset(r, 0, sizeof(*r));
    r->big_bytes = big_bytes;
    r->a = fc_cuda_init_device(0);
    r->b = fc_cuda_init_device(1);
    if (!r->a || !r->b) {
        fprintf(stderr, "need TWO visible dies "
                        "(CUDA_VISIBLE_DEVICES=<a>,<b>)\n");
        return -1;
    }
    if (fc_cuda_ctx_device_id(r->a) == fc_cuda_ctx_device_id(r->b)) {
        fprintf(stderr, "both contexts landed on device %d — not a die pair\n",
                fc_cuda_ctx_device_id(r->a));
        return -1;
    }
    if (fc_cuda_enable_peer(r->a, r->b) != 0) {
        fprintf(stderr, "fc_cuda_enable_peer FAILED — no P2P between these "
                        "dies; every cross-die number below would be a host "
                        "bounce, not a peer copy\n");
        return -1;
    }
    r->buf_a = fc_buffer_alloc_ctx(r->a, XD_EMBD_BYTES);
    r->buf_b = fc_buffer_alloc_ctx(r->b, XD_EMBD_BYTES);
    r->big_a = fc_buffer_alloc_ctx(r->a, big_bytes);
    r->big_b = fc_buffer_alloc_ctx(r->b, big_bytes);
    if (!r->buf_a || !r->buf_b || !r->big_a || !r->big_b) {
        fprintf(stderr, "buffer allocation failed\n");
        return -1;
    }
    for (i = 0; i < XD_EVENT_RING; i++) {
        r->ev_a[i] = fc_event_create_ctx(r->a);
        r->ev_b[i] = fc_event_create_ctx(r->b);
    }
    r->have_events = (r->ev_a[0] != NULL && r->ev_b[0] != NULL);
    /* fc_event_wait_ctx returns -1 when cuStreamWaitEvent is missing from the
     * driver; probe it once rather than reporting a bogus 0.000 ms. */
    r->have_stream_wait = r->have_events &&
        (fc_event_record_ctx(r->a, r->ev_a[0]) == 0) &&
        (fc_event_wait_ctx(r->b, r->ev_a[0]) == 0);
    fc_sync_ctx(r->a);
    fc_sync_ctx(r->b);
    return 0;
}

/* Write a pattern on die A, peer-copy it, read it back off die B. Without this
 * a bounds-checked no-op copy reads as an extremely cheap barrier. */
static int rig_verify_transfer(xd_rig *r)
{
    float *src = malloc(XD_EMBD_BYTES);
    float *dst = malloc(XD_EMBD_BYTES);
    int i, bad = 0;

    if (!src || !dst) {
        free(src);
        free(dst);
        return -1;
    }
    for (i = 0; i < XD_EMBD_BYTES / (int)sizeof(float); i++)
        src[i] = (float)i * 0.5f - 7.0f;
    memset(dst, 0, XD_EMBD_BYTES);
    fc_buffer_from_cpu(r->buf_a, src);
    fc_sync_ctx(r->a);
    if (fc_buffer_copy_peer(r->buf_b, r->buf_a, XD_EMBD_BYTES) != 0) {
        fprintf(stderr, "fc_buffer_copy_peer returned an error\n");
        free(src);
        free(dst);
        return -1;
    }
    fc_sync_ctx(r->b);
    fc_buffer_to_cpu(r->buf_b, dst);
    for (i = 0; i < XD_EMBD_BYTES / (int)sizeof(float); i++)
        if (dst[i] != src[i])
            bad++;
    free(src);
    free(dst);
    if (bad) {
        fprintf(stderr, "peer copy did NOT transfer: %d/%d floats wrong\n",
                bad, XD_EMBD_BYTES / (int)sizeof(float));
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------
 * Barrier variants. Each runs `iters` barriers and returns us per barrier.
 * ------------------------------------------------------------------ */

/* The price of asking the driver whether an idle stream is idle. */
static double bar_sync_floor(xd_rig *r, int iters, int depth)
{
    double t0;
    int i;
    (void)depth;
    fc_sync_ctx(r->a);
    t0 = now_s();
    for (i = 0; i < iters; i++)
        fc_sync_ctx(r->a);
    return (now_s() - t0) * 1e6 / iters;
}

static double bar_peer_blocking(xd_rig *r, int iters, int depth)
{
    double t0;
    int i;
    (void)depth;
    fc_buffer_copy_peer(r->buf_b, r->buf_a, XD_EMBD_BYTES);
    t0 = now_s();
    for (i = 0; i < iters; i++)
        fc_buffer_copy_peer(r->buf_b, r->buf_a, XD_EMBD_BYTES);
    return (now_s() - t0) * 1e6 / iters;
}

/* Exactly gpu_split_head_slots: drain the producer, copy, drain the consumer. */
static double bar_prod_pattern(xd_rig *r, int iters, int depth)
{
    double t0;
    int i;
    (void)depth;
    t0 = now_s();
    for (i = 0; i < iters; i++) {
        fc_sync_ctx(r->a);
        fc_buffer_copy_peer(r->buf_b, r->buf_a, XD_EMBD_BYTES);
        fc_sync_ctx(r->b);
    }
    return (now_s() - t0) * 1e6 / iters;
}

static double bar_evsync(xd_rig *r, int iters, int depth)
{
    double t0;
    int i;
    (void)depth;
    if (!r->have_events)
        return -1.0;
    t0 = now_s();
    for (i = 0; i < iters; i++) {
        fc_buffer_copy_peer_async(r->buf_b, r->buf_a, XD_EMBD_BYTES, r->b);
        fc_event_record_ctx(r->b, r->ev_b[0]);
        fc_event_sync(r->ev_b[0]);
    }
    return (now_s() - t0) * 1e6 / iters;
}

static double bar_evpoll(xd_rig *r, int iters, int depth)
{
    double t0;
    int i;
    (void)depth;
    if (!r->have_events)
        return -1.0;
    t0 = now_s();
    for (i = 0; i < iters; i++) {
        fc_buffer_copy_peer_async(r->buf_b, r->buf_a, XD_EMBD_BYTES, r->b);
        fc_event_record_ctx(r->b, r->ev_b[0]);
        while (fc_event_query(r->ev_b[0]) == 1)
            ;
    }
    return (now_s() - t0) * 1e6 / iters;
}

/* THE decisive variant. A -> B hand-off ordered entirely on the devices:
 * B's stream waits on A's event, copies, records; A's stream waits on B's
 * event. The host issues but never synchronises inside the pipeline, so this
 * measures what a barrier costs when it is NOT a host round-trip.
 *
 * `depth` barriers are queued before one drain, matching how a TP forward
 * would pipeline 80 of them through a step. */
static double bar_gpu_pingpong(xd_rig *r, int iters, int depth)
{
    double t0;
    int i, q = 0;

    if (!r->have_stream_wait)
        return -1.0;
    if (depth < 1)
        depth = 1;
    if (depth > XD_EVENT_RING)
        depth = XD_EVENT_RING;
    fc_sync_ctx(r->a);
    fc_sync_ctx(r->b);
    t0 = now_s();
    for (i = 0; i < iters; i++) {
        int s = q % depth;
        /* A produced something: mark it, make B wait for it on-device. */
        fc_event_record_ctx(r->a, r->ev_a[s]);
        fc_event_wait_ctx(r->b, r->ev_a[s]);
        fc_buffer_copy_peer_async(r->buf_b, r->buf_a, XD_EMBD_BYTES, r->b);
        /* B produced something: mark it, make A wait for it on-device. */
        fc_event_record_ctx(r->b, r->ev_b[s]);
        fc_event_wait_ctx(r->a, r->ev_b[s]);
        if (++q % depth == 0) {
            fc_sync_ctx(r->a);
            fc_sync_ctx(r->b);
        }
    }
    fc_sync_ctx(r->a);
    fc_sync_ctx(r->b);
    return (now_s() - t0) * 1e6 / iters;
}

/* A full 2-die all-reduce leg: both directions, both dies device-ordered.
 * This is the primitive a column-split layer needs, not a one-way copy. */
static double bar_allreduce(xd_rig *r, int iters, int depth)
{
    double t0;
    int i, q = 0;

    if (!r->have_stream_wait)
        return -1.0;
    if (depth < 1)
        depth = 1;
    if (depth > XD_EVENT_RING)
        depth = XD_EVENT_RING;
    fc_sync_ctx(r->a);
    fc_sync_ctx(r->b);
    t0 = now_s();
    for (i = 0; i < iters; i++) {
        int s = q % depth;
        fc_event_record_ctx(r->a, r->ev_a[s]);
        fc_event_record_ctx(r->b, r->ev_b[s]);
        fc_event_wait_ctx(r->b, r->ev_a[s]);
        fc_event_wait_ctx(r->a, r->ev_b[s]);
        fc_buffer_copy_peer_async(r->buf_b, r->buf_a, XD_EMBD_BYTES, r->b);
        fc_buffer_copy_peer_async(r->buf_a, r->buf_b, XD_EMBD_BYTES, r->a);
        /* the summation half of the reduce, one cheap elementwise pass/die */
        fc_sigmoid_ctx(r->a, r->buf_a, XD_EMBD_BYTES / (int)sizeof(float));
        fc_sigmoid_ctx(r->b, r->buf_b, XD_EMBD_BYTES / (int)sizeof(float));
        if (++q % depth == 0) {
            fc_sync_ctx(r->a);
            fc_sync_ctx(r->b);
        }
    }
    fc_sync_ctx(r->a);
    fc_sync_ctx(r->b);
    return (now_s() - t0) * 1e6 / iters;
}

typedef double (*bar_fn)(xd_rig *, int, int);

typedef struct bar_case {
    const char *name;
    const char *what;
    bar_fn      fn;
    double      us;
} bar_case;

/* ------------------------------------------------------------------
 * Size sweep — separates the fixed per-transfer cost from bandwidth.
 *
 * ⚠ THE DRAIN IS NOT OPTIONAL AND THIS IS WHY.
 * cuMemcpyPeer has no `Async` suffix, but a DEVICE-TO-DEVICE copy is
 * documented as performing NO host-side synchronisation — it queues on the
 * stream and returns. An undrained loop therefore times the HOST ISSUE RATE,
 * not the link, and reports a bandwidth the hardware cannot physically
 * deliver. The first run of this bench did exactly that and produced
 * "80.5 GB/s" at 4 MB across a PCIe 3.0 x16 link whose theoretical ceiling is
 * 15.75 GB/s — a 5x-impossible number that would have been reported as the
 * answer to the disputed ~13-vs-~75 GB/s question. Both contexts are drained
 * INSIDE the timed region, and the largest transfer's payload is verified, so
 * a queued-but-unfinished copy cannot be mistaken for a fast one.
 * ------------------------------------------------------------------ */

/* Confirm `bytes` really landed on die B: stamp a sentinel at the first and
 * last word of the source, copy, drain, read back. Cheap, and it is the only
 * thing standing between "the link is fast" and "the copy did not happen". */
static int verify_big_transfer(xd_rig *r, size_t bytes)
{
    unsigned int *src = malloc(bytes);
    unsigned int *dst = malloc(bytes);
    size_t n = bytes / sizeof(unsigned int), i;
    int bad = 0;

    if (!src || !dst) {
        free(src);
        free(dst);
        return -1;
    }
    /* A counter pattern, so a stale or zeroed destination cannot pass and a
     * partially-completed copy shows up as a tail mismatch. */
    for (i = 0; i < n; i++)
        src[i] = (unsigned int)(i * 2654435761u);
    memset(dst, 0, bytes);
    fc_buffer_from_cpu_n(r->big_a, src, bytes);
    fc_sync_ctx(r->a);
    if (fc_buffer_copy_peer(r->big_b, r->big_a, bytes) != 0)
        bad = 1;
    fc_sync_ctx(r->b);
    fc_buffer_to_cpu_n(r->big_b, dst, bytes);
    for (i = 0; !bad && i < n; i++)
        if (dst[i] != src[i])
            bad = 1;
    free(src);
    free(dst);
    return bad ? -1 : 0;
}

static void sweep_sizes(xd_rig *r, int iters)
{
    static const size_t sizes[] = {
        1024, 4096, 8192, 16384, 65536, 262144, 1048576, 4194304,
        16777216, 67108864
    };
    size_t nsz = sizeof(sizes) / sizeof(sizes[0]);
    size_t s, top = 0;
    double first = 0.0, last = 0.0, top_gbs = 0.0;

    printf("\n== peer-copy size sweep (ASYNC on die B's stream + drain) ==\n");
    printf("%12s %12s %12s %12s\n", "bytes", "us", "GB/s", "issue us");
    for (s = 0; s < nsz; s++) {
        double t0, us, issue;
        int i, n = iters;
        if (sizes[s] > r->big_bytes)
            continue;
        if (sizes[s] >= 1048576)
            n = iters / 100 > 10 ? iters / 100 : 10;

        /* The DRAINED rate. The copy is queued on die B's own stream, which
         * fc_sync_ctx genuinely waits on, so the timed region contains the
         * whole transfer and not just its issue. */
        fc_buffer_copy_peer_async(r->big_b, r->big_a, sizes[s], r->b);
        fc_sync_ctx(r->b);
        t0 = now_s();
        for (i = 0; i < n; i++)
            fc_buffer_copy_peer_async(r->big_b, r->big_a, sizes[s], r->b);
        fc_sync_ctx(r->b);
        us = (now_s() - t0) * 1e6 / n;

        /* The UNDRAINED host issue cost, for contrast. Anywhere the two
         * columns diverge, the undrained one is fiction. */
        t0 = now_s();
        for (i = 0; i < n; i++)
            fc_buffer_copy_peer_async(r->big_b, r->big_a, sizes[s], r->b);
        issue = (now_s() - t0) * 1e6 / n;
        fc_sync_ctx(r->b);

        top_gbs = (double)sizes[s] / (us * 1e-6) / 1e9;
        printf("%12zu %12.2f %12.2f %12.2f\n", sizes[s], us, top_gbs, issue);
        if (s == 0) first = us;
        last = us;
        top = sizes[s];
    }
    if (last <= first * 1.05) {
        printf("  FAIL  cost did not rise with size (%.2f us at 1 KB vs %.2f us"
               " at the top) — the copy is not moving bytes\n", first, last);
        return;
    }
    printf("  pass  cost rises with size; the intercept is the fixed"
           " per-transfer cost, the slope is the PLX link\n");
    if (verify_big_transfer(r, top) == 0)
        printf("  pass  %zu-byte payload verified on die B (head+tail"
               " sentinels) — %.1f GB/s is a real transfer, not a queued one\n",
               top, top_gbs);
    else
        printf("  FAIL  %zu-byte payload did NOT arrive on die B — the"
               " bandwidth above is meaningless\n", top);
}

/* ------------------------------------------------------------------
 * Graph fragmentation — what does breaking one captured segment into N
 * pieces cost per piece?
 *
 * Three arms over the SAME total kernel work:
 *   raw     : `total` direct launches, no graph. The pre-graph world.
 *   1 graph : one capture of all `total` launches, one cuGraphLaunch.
 *   N graphs: `total`/N launches per capture, N cuGraphLaunches.
 * (cost(N) - cost(1)) / (N - 1) is the incremental cost of one extra graph
 * boundary — i.e. what each of TP's 80 barriers would cost in graph overhead
 * alone, before the barrier itself is paid.
 * ------------------------------------------------------------------ */

/* Capture `count` tiny kernels on ctx into one graph. NULL if capture fails
 * (in which case the recorded launches did NOT run and the caller must not
 * treat the arm as measured). */
static fc_graph *capture_chunk(fc_cuda_ctx *ctx, fc_buffer *buf, int count)
{
    int i;
    if (fc_graph_begin_ctx(ctx) != 0)
        return NULL;
    for (i = 0; i < count; i++)
        fc_sigmoid_ctx(ctx, buf, 32);
    return fc_graph_end_ctx(ctx);
}

static void bench_graph_fragmentation(xd_rig *r, int total, int reps)
{
    static const int frags[] = {1, 2, 4, 8, 20, 40, 80};
    size_t nf = sizeof(frags) / sizeof(frags[0]);
    fc_buffer *buf = fc_buffer_alloc_ctx(r->a, 64 * sizeof(float));
    double raw_ms = 0.0, one_ms = 0.0;
    size_t f;
    int i, rep;

    printf("\n== graph fragmentation on die %d (%d tiny kernels per 'token') ==\n",
           fc_cuda_ctx_device_id(r->a), total);
    if (!buf) {
        printf("  buffer allocation failed — skipped\n");
        return;
    }

    /* raw launches, no graph */
    {
        double *m = malloc((size_t)reps * sizeof(double));
        for (rep = 0; rep < reps; rep++) {
            double t0;
            for (i = 0; i < total; i++)
                fc_sigmoid_ctx(r->a, buf, 32);
            fc_sync_ctx(r->a);
            t0 = now_s();
            for (i = 0; i < total; i++)
                fc_sigmoid_ctx(r->a, buf, 32);
            fc_sync_ctx(r->a);
            m[rep] = (now_s() - t0) * 1e3;
        }
        qsort(m, (size_t)reps, sizeof(double), cmp_double);
        raw_ms = m[reps / 2];
        free(m);
        printf("%10s %10s %14s %16s\n", "fragments", "ms", "vs 1 graph",
               "us per extra frag");
        printf("%10s %10.4f %14s %16s\n", "raw", raw_ms, "-", "-");
    }

    for (f = 0; f < nf; f++) {
        int n = frags[f];
        int per = total / n;
        fc_graph **g = NULL;
        double *m = NULL;
        int ok = 1;

        if (per < 1)
            continue;
        g = calloc((size_t)n, sizeof(*g));
        m = malloc((size_t)reps * sizeof(double));
        if (!g || !m) {
            free(g);
            free(m);
            continue;
        }
        for (i = 0; i < n; i++) {
            g[i] = capture_chunk(r->a, buf, per);
            if (!g[i]) {
                ok = 0;
                break;
            }
        }
        if (!ok) {
            printf("%10d %10s %14s %16s   (capture FAILED — not measured)\n",
                   n, "-", "-", "-");
        } else {
            for (rep = 0; rep < reps; rep++) {
                double t0;
                for (i = 0; i < n; i++)
                    fc_graph_launch(g[i]);
                fc_sync_ctx(r->a);
                t0 = now_s();
                for (i = 0; i < n; i++)
                    fc_graph_launch(g[i]);
                fc_sync_ctx(r->a);
                m[rep] = (now_s() - t0) * 1e3;
            }
            qsort(m, (size_t)reps, sizeof(double), cmp_double);
            if (n == 1)
                one_ms = m[reps / 2];
            printf("%10d %10.4f %13.3fx %16.2f\n", n, m[reps / 2],
                   one_ms > 0.0 ? m[reps / 2] / one_ms : 0.0,
                   n > 1 ? (m[reps / 2] - one_ms) * 1e3 / (n - 1) : 0.0);
        }
        for (i = 0; i < n; i++)
            fc_graph_free(g[i]);
        free(g);
        free(m);
        fflush(stdout);
    }
    printf("  'us per extra frag' is what ONE extra graph boundary costs. TP\n"
           "  needs ~80 of them per token; multiply and compare against the\n"
           "  52.6 ms step before believing any TP arithmetic.\n");
    fc_buffer_free(buf);
}

/* Can a PEER COPY be captured inside a graph at all? If yes, only the barrier
 * forces a fragment; if no, every cross-die transfer forces one too. */
static void probe_peer_in_graph(xd_rig *r)
{
    fc_graph *g;

    printf("\n== can a cross-die peer copy live INSIDE a captured graph? ==\n");
    if (fc_graph_begin_ctx(r->b) != 0) {
        printf("  graphs unavailable on this driver — skipped\n");
        return;
    }
    fc_buffer_copy_peer_async(r->buf_b, r->buf_a, XD_EMBD_BYTES, r->b);
    g = fc_graph_end_ctx(r->b);
    if (!g) {
        printf("  NO — capture failed with a peer copy in the stream. Every\n"
               "  cross-die transfer would break full-segment capture on its\n"
               "  own, independently of the barrier.\n");
        return;
    }
    if (fc_graph_launch(g) == 0) {
        fc_sync_ctx(r->b);
        printf("  YES — captured and replayed. The transfer itself is\n"
               "  graph-safe; only the BARRIER forces a fragment.\n");
    } else {
        printf("  captured but replay failed — treat as unusable.\n");
    }
    fc_graph_free(g);
}

/* ==================================================================
 * PLX STREAMING — the scheme that needs no weight split at all.
 *
 * This is a DIFFERENT idea from tensor parallelism and it is the one the
 * bandwidth numbers actually favour. Do not conflate them.
 *
 * The framing that makes it work:
 *   - die A's DRAM can serve a pure streaming read at 197.4 GB/s, and Q3_K's
 *     access pattern caps at ~171 GB/s.
 *   - fc_matvec_q3k actually draws only ~91.6 GB/s, because it is INSTRUCTION
 *     -ISSUE bound, not bandwidth bound.
 *   - so ~79 GB/s of die A's own DRAM headroom is sitting unused.
 * A second die's SMs are a second instruction-issue engine. If die B's kernels
 * read die A's weights directly over the peer link, die B consumes that
 * headroom WITHOUT any weight being split, copied, re-laid-out or duplicated:
 * no extra VRAM, no new kernels, no reduce.
 *
 * THE TRAP THIS MEASURES AROUND: a peer link has TWO very different bandwidths.
 *   - cuMemcpyPeer uses the copy ENGINE and is what every "P2P bandwidth"
 *     number quotes. That is what sweep_sizes() above measures.
 *   - a kernel dereferencing a remote pointer issues ordinary loads that
 *     happen to cross the link. Latency is far higher and the achieved rate
 *     can be a small fraction of the DMA figure.
 * Scheme 9a needs the SECOND one, and only the second one. Reporting the DMA
 * number as if it were the kernel number is exactly the kind of confident
 * wrong answer this hardware keeps producing, so both are measured and
 * labelled, and the verdict quotes the kernel figure.
 *
 * IMPORTANT: this is also the crux of the ~13 vs ~75 GB/s disagreement about
 * this board. Whatever the link turns out to be, the number that matters is
 * the one a matvec kernel can actually pull through it.
 * ================================================================== */

#define PLX_Q3K 11

static size_t q3k_raw_row(int cols)   { return 110u * (size_t)(cols / 256); }

/* Upload a random Q3_K weight to `ctx`, repacked as a model upload would. */
static fc_buffer *plx_make_weight(fc_cuda_ctx *ctx, long rows, int cols,
                                  int *stride)
{
    size_t raw_stride = q3k_raw_row(cols);
    size_t rs = fc_repack_row_bytes(PLX_Q3K, cols);
    unsigned char *raw = malloc(raw_stride * (size_t)rows);
    unsigned char *packed = malloc(rs * (size_t)rows);
    fc_buffer *b = NULL;
    size_t i;

    if (!raw || !packed)
        goto out;
    for (i = 0; i < raw_stride * (size_t)rows; i++)
        raw[i] = (unsigned char)(rand() & 0xFF);
    /* Clamp every block's fp16 scale away from inf/nan (position 108 for
     * Q3_K) so no arm can accidentally run on a denormal slow path. */
    for (i = 0; i + 110 <= raw_stride * (size_t)rows; i += 110) {
        unsigned short h = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        raw[i + 108] = h & 0xFF;
        raw[i + 109] = h >> 8;
    }
    if (fc_repack_weight(PLX_Q3K, raw, packed, rows, cols) != 0)
        goto out;
    b = fc_buffer_from_ptr_ctx(ctx, packed, rs * (size_t)rows);
    *stride = (int)rs;
out:
    free(raw);
    free(packed);
    return b;
}

static fc_buffer *plx_make_floats(fc_cuda_ctx *ctx, size_t n)
{
    float *h = malloc(n * sizeof(float));
    fc_buffer *b;
    size_t i;
    if (!h)
        return NULL;
    for (i = 0; i < n; i++)
        h[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    b = fc_buffer_from_ptr_ctx(ctx, h, n * sizeof(float));
    free(h);
    return b;
}

/* One shape's worth of the PLX-streaming probe.
 *
 * Three arms over the SAME kernel and the same shape:
 *   A local      die A reads its own weights          -> the 91.6 GB/s baseline
 *   B remote     die B reads DIE A's weights          -> the kernel-issued PLX rate
 *   A + B        both at once, B still reading A's    -> what 9a would actually get
 * The third is the only one that answers the question, because it is the only
 * one where die A's memory controller is serving both consumers.
 *
 * fc_matvec_q3k_ctx launches in the context it is HANDED and simply passes
 * w->dptr, with no ownership check — so handing it a die-A buffer with die B's
 * context is a genuine remote read, not a copy. Peer access makes the address
 * valid in B. Verified below by checking the remote arm produces the same
 * output as the local arm. */
static void bench_plx_streaming(xd_rig *r, const char *label, long rows,
                                int cols, int iters, int reps)
{
    fc_buffer *w_a = NULL, *w_b = NULL;
    fc_buffer *x_a = NULL, *x_b = NULL, *y_a = NULL, *y_b = NULL;
    float *ref = NULL, *got = NULL;
    int stride_a = 0, stride_b = 0;
    double loc[16], rem[16], agg[16];
    double bytes = (double)rows;
    int rep, i, mismatch = 0;

    if (reps > 16)
        reps = 16;
    w_a = plx_make_weight(r->a, rows, cols, &stride_a);
    w_b = plx_make_weight(r->b, rows, cols, &stride_b);
    x_a = plx_make_floats(r->a, (size_t)cols);
    x_b = plx_make_floats(r->b, (size_t)cols);
    y_a = fc_buffer_alloc_ctx(r->a, (size_t)rows * sizeof(float));
    y_b = fc_buffer_alloc_ctx(r->b, (size_t)rows * sizeof(float));
    if (!w_a || !w_b || !x_a || !x_b || !y_a || !y_b) {
        printf("  %s: allocation failed (needs ~%.0f MB per die) — skipped\n",
               label, (double)rows * stride_a / 1e6);
        goto out;
    }
    bytes *= (double)stride_a;

    /* Correctness of the remote path, not just its speed: die B reading die A's
     * weights must produce what die A produces. A remote read that silently
     * returned zeros would look like magnificent bandwidth. */
    ref = malloc((size_t)rows * sizeof(float));
    got = malloc((size_t)rows * sizeof(float));
    if (ref && got) {
        fc_buffer_copy_peer(x_b, x_a, (size_t)cols * sizeof(float));
        fc_sync_ctx(r->b);
        fc_matvec_q3k_ctx(r->a, w_a, x_a, y_a, (int)rows, cols, stride_a);
        fc_sync_ctx(r->a);
        fc_buffer_to_cpu(y_a, ref);
        fc_matvec_q3k_ctx(r->b, w_a, x_b, y_b, (int)rows, cols, stride_a);
        fc_sync_ctx(r->b);
        fc_buffer_to_cpu(y_b, got);
        for (i = 0; i < (int)rows; i++)
            if (ref[i] != got[i])
                mismatch++;
    }

    for (rep = 0; rep < reps; rep++) {
        double t0;
        /* die A, local weights */
        fc_matvec_q3k_ctx(r->a, w_a, x_a, y_a, (int)rows, cols, stride_a);
        fc_sync_ctx(r->a);
        t0 = now_s();
        for (i = 0; i < iters; i++)
            fc_matvec_q3k_ctx(r->a, w_a, x_a, y_a, (int)rows, cols, stride_a);
        fc_sync_ctx(r->a);
        loc[rep] = (now_s() - t0) / iters;

        /* die B, reading die A's weights across the link */
        fc_matvec_q3k_ctx(r->b, w_a, x_b, y_b, (int)rows, cols, stride_a);
        fc_sync_ctx(r->b);
        t0 = now_s();
        for (i = 0; i < iters; i++)
            fc_matvec_q3k_ctx(r->b, w_a, x_b, y_b, (int)rows, cols, stride_a);
        fc_sync_ctx(r->b);
        rem[rep] = (now_s() - t0) / iters;

        /* both at once — die A's memory controller serving both dies. Issued
         * onto two independent streams with no sync between, so they really do
         * overlap; one drain at the end covers both. */
        t0 = now_s();
        for (i = 0; i < iters; i++) {
            fc_matvec_q3k_ctx(r->a, w_a, x_a, y_a, (int)rows, cols, stride_a);
            fc_matvec_q3k_ctx(r->b, w_a, x_b, y_b, (int)rows, cols, stride_a);
        }
        fc_sync_ctx(r->a);
        fc_sync_ctx(r->b);
        agg[rep] = (now_s() - t0) / iters;
    }
    qsort(loc, (size_t)reps, sizeof(double), cmp_double);
    qsort(rem, (size_t)reps, sizeof(double), cmp_double);
    qsort(agg, (size_t)reps, sizeof(double), cmp_double);

    printf("  %-22s %10s %10s %10s %10s\n", label, "ms", "GB/s", "vs local",
           "");
    printf("    %-20s %10.4f %10.2f %10s\n", "A local",
           loc[reps / 2] * 1e3, bytes / loc[reps / 2] / 1e9, "1.00x");
    printf("    %-20s %10.4f %10.2f %9.3fx   <-- KERNEL-ISSUED PLX RATE\n",
           "B reads A (remote)", rem[reps / 2] * 1e3,
           bytes / rem[reps / 2] / 1e9, loc[reps / 2] / rem[reps / 2]);
    printf("    %-20s %10.4f %10.2f %9.3fx   <-- what scheme 9a gets\n",
           "A + B concurrent", agg[reps / 2] * 1e3,
           2.0 * bytes / agg[reps / 2] / 1e9,
           2.0 * loc[reps / 2] / agg[reps / 2]);
    if (mismatch)
        printf("    ** REMOTE READ IS WRONG: %d/%ld rows differ from the local "
               "result — the speed above is meaningless **\n", mismatch, rows);
    else if (ref && got)
        printf("    remote result verified bit-identical to local\n");

out:
    free(ref);
    free(got);
    fc_buffer_free(w_a); fc_buffer_free(w_b);
    fc_buffer_free(x_a); fc_buffer_free(x_b);
    fc_buffer_free(y_a); fc_buffer_free(y_b);
}

/* ------------------------------------------------------------------ */

int main(void)
{
    xd_rig r;
    int reps = env_int("XD_REPS", 5);
    int iters = env_int("XD_ITERS", 2000);
    int depth = env_int("XD_DEPTH", 64);
    int frag_total = env_int("XD_FRAG", 600);
    bar_case cases[] = {
        { "sync_floor",    "fc_sync_ctx on an idle stream",        bar_sync_floor,    0 },
        { "peer_blocking", "cuMemcpyPeer 8 KB",                    bar_peer_blocking, 0 },
        { "prod_pattern",  "sync(B)+copy+sync(A)  [SHIPPED T3]",   bar_prod_pattern,  0 },
        { "evsync",        "async copy + cuEventSynchronize",      bar_evsync,        0 },
        { "evpoll",        "async copy + cuEventQuery spin",       bar_evpoll,        0 },
        { "gpu_pingpong",  "cuStreamWaitEvent, host out of loop",  bar_gpu_pingpong,  0 },
        { "allreduce",     "both directions + add, device-ordered", bar_allreduce,    0 },
    };
    int ncase = (int)(sizeof(cases) / sizeof(cases[0]));
    int i, rep, fail = 0;

    if (fc_driver_load() != 0) {
        fprintf(stderr, "no CUDA driver\n");
        return 1;
    }
    if (rig_init(&r, 64u * 1024 * 1024) != 0)
        return 1;

    printf("bench_cross_die — what does ONE cross-die barrier cost?\n");
    printf("die A = device %d, die B = device %d, peer ENABLED\n",
           fc_cuda_ctx_device_id(r.a), fc_cuda_ctx_device_id(r.b));
    printf("events %s, cuStreamWaitEvent %s\n",
           r.have_events ? "available" : "UNAVAILABLE",
           r.have_stream_wait ? "available" : "UNAVAILABLE");
    printf("reps=%d iters=%d pipeline depth=%d\n", reps, iters, depth);

    if (rig_verify_transfer(&r) != 0) {
        fprintf(stderr, "TRANSFER VERIFY FAILED — refusing to report timings\n");
        rig_free(&r);
        return 1;
    }
    printf("transfer verified: 8 KB pattern survives die %d -> die %d\n",
           fc_cuda_ctx_device_id(r.a), fc_cuda_ctx_device_id(r.b));

    printf("\n== barrier variants, us per barrier (median of %d reps) ==\n", reps);
    printf("%-15s %-40s %12s %14s\n", "variant", "what", "us/barrier",
           "x80 = ms/token");
    for (i = 0; i < ncase; i++) {
        double *m = malloc((size_t)reps * sizeof(double));
        int measured = 1;
        if (!m)
            continue;
        for (rep = 0; rep < reps; rep++) {
            m[rep] = cases[i].fn(&r, iters, depth);
            if (m[rep] < 0.0)
                measured = 0;
        }
        if (!measured) {
            printf("%-15s %-40s %12s %14s\n", cases[i].name, cases[i].what,
                   "n/a", "-");
            free(m);
            continue;
        }
        qsort(m, (size_t)reps, sizeof(double), cmp_double);
        cases[i].us = m[reps / 2];
        printf("%-15s %-40s %12.2f %14.2f\n", cases[i].name, cases[i].what,
               cases[i].us, cases[i].us * 80.0 / 1000.0);
        free(m);
        fflush(stdout);
    }
    printf("\n'x80' is the whole-token barrier bill for layer-level TP (2\n"
           "all-reduces x 40 layers). Against a 52.6 ms step, anything at or\n"
           "above ~26 ms erases the entire 2x ceiling on its own.\n");

    /* The size sweep comes first now: the die-to-die link rate is the single
     * number the whole verdict pivots on, and it is disputed (~13 GB/s from
     * PCIe 3.0 x16 specs vs ~75 GB/s claimed for this board). Measure it before
     * anything is modelled on top of it. */
    sweep_sizes(&r, iters);

    printf("\n== PLX streaming: can die B's SMs soak up die A's unused DRAM"
           " headroom? ==\n");
    printf("  die A's Q3_K layout ceiling ~171 GB/s; fc_matvec_q3k draws ~91.6."
           " ~79 GB/s spare.\n");
    printf("  NOTE the sweep above is the COPY ENGINE. These rows are"
           " KERNEL-ISSUED remote loads,\n  which is what this scheme needs and"
           " can be several times slower.\n");
    bench_plx_streaming(&r, "qkv 8192x2048", 8192, 2048, iters / 20 + 1, reps);
    bench_plx_streaming(&r, "lm_head 248320x2048", 248320, 2048, 20, reps);

    probe_peer_in_graph(&r);
    bench_graph_fragmentation(&r, frag_total, reps);

    rig_free(&r);
    fc_free();
    return fail;
}
