/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_tp_q38.c
 * @brief Prices tensor-parallel (TP) decode for Qwen3.8-27B Q8_0 on 4 K80 dies.
 *
 * Today Qwen3.8 decodes as a layer split: die k runs layers 16k..16k+15, so one die
 * works while three idle. TP gives every die a slice of every layer and pays a
 * cross-die exchange inside each layer. This file measures the parts of that trade:
 *
 *   peer  P2P access matrix of the visible dies, a verified copy per pair, and the
 *         latency / bandwidth of one peer copy (on-board pair vs across boards).
 *   mv    die 0 alone: Q8_0 (G4) matvec time for every Qwen3.8 weight at its full
 *         shape, at 1/2 and 1/4 of the output rows, and at 1/2 and 1/4 of the input
 *         columns, for n = 1..5 rows (decode = 1, MTP verify = 2..5).
 *   mvc   the same quarter slices on all dies AT ONCE vs die 0 alone (a K80 board's
 *         two dies share a power budget; TP keeps both busy, the split never does).
 *   xchg  a dependent chain of exchanges (each die: one add over the payload, then
 *         the exchange) for an on-board pair, a cross-board pair, both pairs at once
 *         and all four dies. Reported as us per exchange over an add-only baseline.
 *   step  an emulated decode step: every matvec of all 64 layers + the LM head at
 *         its sliced shape, plus every exchange / stage hop, issued exactly as a TP
 *         forward would issue them (device-side event waits, no host sync inside the
 *         step), for today's split, TP2 within a board x 2 pipeline stages, and TP4.
 *         No attention core, SSM recurrence, conv or norms: those are priced from the
 *         Pantheon profile (QWEN35_PROF) by the caller.
 *
 * TP schemes (G = dies in a TP group):
 *   E4  bit-exact, existing kernels only: every matvec split by OUTPUT rows (each
 *       output is computed whole on one die, same chains and tree as today). Per
 *       layer: gather core (6144), gather hidden (5120) after the mixer out
 *       projection, gather FFN hidden (17408), gather hidden after down. 4 exchanges.
 *   E3  bit-exact with one new kernel: as E4, but FFN down is split by INPUT columns
 *       along the Q8 kernel's own chain tree (die m owns chain subtree m, i.e. every
 *       element with index%4 == m for TP4), gate/up rows assigned to match, and the 4
 *       partials summed in the tree's own order. 3 exchanges (gather, gather, reduce).
 *       The down matvec is timed here as a cols/G matvec; the real kernel reads one
 *       scale per 32-block on every die (~11% more bytes on down with f16 scales).
 *   M2  NOT exact (Megatron): mixer out and FFN down split by contiguous input
 *       columns, two all-reduces per layer. Prices what exactness costs.
 *
 * Weights are random Q8_0 bytes with sane fp16 scales, repacked with
 * fc_repack_weight (G4 when cols % 128 == 0, as every Qwen3.8 matrix is) and tiled
 * through one arena per die, every slice on its own row range. No correctness is
 * claimed; bench_q80_tile / test_matvec_nx gate these kernels.
 *
 * Build (from foundry/): make cuda/tests/bench_tp_q38
 * Run (dies that nobody else holds; PCI order):
 *   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8,9,10,11 \
 *       ./cuda/tests/bench_tp_q38 peer|mv|mvc|xchg|step
 * Env:
 *   TP_NS     row counts, e.g. "1,4" (default 1,2,3,4,5)
 *   TP_REPS   repetitions, median reported (default 5)
 *   TP_ITERS  launches per timing (mv/mvc, default 40) / chain length (xchg, 128)
 *   TP_STEPS  steps per step timing (default 6)
 *   TP_CFG    step configs, subset of "split,tp2e4,tp2e3,tp2m2,tp4e4,tp4e3,tp4m2"
 *   TP_SOAK   seconds: run the first TP_CFG config at the first TP_NS n, ms/step per 10 s
 *   TP_XB     cross-board method (drv|ce|zc|zcce|cezc, tp_xboard_copy.h). Set: a cross-
 *             board pair stages through that method, and a 4-die group exchanges
 *             hierarchically (xchg_hier). Unset: every copy is cuMemcpyPeerAsync.
 *   TP_NOXCHG 1 = step issues no exchanges or hops (matvec + head only). Without the hop
 *             nothing orders the stages, so a multi-stage config's stages run at once:
 *             only single-stage configs (tp4*) give a serial matvec time
 *
 * Mode xbv: verified cross-board copies by method, swap latency, one-way bandwidth.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "cuda/fc_cuda.h"
#include "cuda/tests/tp_xboard_copy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Qwen3.8-27B geometry (GGUF metadata). */
#define Q_E      5120     /* hidden */
#define Q_FF     17408    /* dense FFN */
#define Q_CORE   6144     /* GDN 48 v-heads x 128 == attention 24 heads x 256 */
#define Q_QKV    10240    /* GDN q 16x128 + k 16x128 + v 48x128 */
#define Q_BA     96       /* GDN beta + alpha, 48 each */
#define Q_AQ     12288    /* attention q + output gate, 24 x 256 x 2 */
#define Q_AKV    1024     /* attention k (or v), 4 kv heads x 256 */
#define Q_VOCAB  248320
#define Q_LAYERS 64

#define MAXD   4
#define MAXN   8
#define EVR    1024
#define ALIGNB 9216       /* 64 G4 groups: every slice starts on a group, 16-B aligned */

static int is_attn(int l) { return (l % 4) == 3; }

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

static double median(double *v, int n)
{
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    return v[n / 2];
}

static int g_ns[MAXN], g_nn;

static void parse_ns(void)
{
    const char *e = getenv("TP_NS");
    const char *p = (e && *e) ? e : "1,2,3,4,5";
    g_nn = 0;
    while (*p && g_nn < MAXN) {
        int v = atoi(p);
        if (v >= 1 && v <= MAXN)
            g_ns[g_nn++] = v;
        while (*p && *p != ',')
            p++;
        if (*p == ',')
            p++;
    }
}

/* ------------------------------------------------------------------
 * Dies
 * ------------------------------------------------------------------ */

typedef struct die {
    fc_cuda_ctx *ctx;
    fc_buffer   *arena;
    size_t       arena_bytes, bump;
    fc_buffer   *x;          /* matvec input, MAXN x Q_FF */
    fc_buffer   *y[2];       /* matvec output / exchange source, ping-pong */
    fc_buffer   *rcv;        /* exchange landing zone, MAXD x MAXN x Q_FF */
    fc_buffer   *sum;        /* combine output */
    fc_event    *ev[EVR];
    int          evi;
} die_t;

static die_t D[MAXD];
static int   ND;
static int   g_xb = -1;    /* TP_XB method, -1 = plain peer copies */
static const char *g_mode = "";

static size_t q8_row_bytes(int cols) { return fc_repack_row_bytes(8, cols); }

static fc_event *die_record(int d)
{
    fc_event *e = D[d].ev[D[d].evi++ % EVR];
    fc_event_record_ctx(D[d].ctx, e);
    return e;
}

/* A valid repacked Q8_0 pattern (64 rows x 5120, fp16 scales in a sane exponent
 * range), doubled through the arena on the device. Every multiple of 144 B is a G4
 * group start, so any slice at an ALIGNB offset reads well-formed groups. */
static int arena_fill(die_t *d)
{
    const int rows = 64, cols = Q_E;
    size_t raw_rs = (size_t)(cols / 32) * 34, rs = q8_row_bytes(cols);
    size_t pat = rs * rows, have;
    unsigned char *raw = malloc(raw_rs * rows), *packed = malloc(pat);
    fc_buffer *pb;
    size_t i;

    if (!raw || !packed)
        return -1;
    for (i = 0; i < raw_rs * rows; i++)
        raw[i] = (unsigned char)(rand() & 0xFF);
    for (i = 0; i + 34 <= raw_rs * rows; i += 34) {
        unsigned short h = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        raw[i] = h & 0xFF;
        raw[i + 1] = h >> 8;
    }
    if (fc_repack_weight(8, raw, packed, rows, cols) != 0)
        return -1;
    pb = fc_buffer_from_ptr_ctx(d->ctx, packed, pat);
    free(raw);
    free(packed);
    if (!pb)
        return -1;
    fc_buffer_copy_d2d_ctx(d->ctx, d->arena, 0, pb, 0, pat);
    for (have = pat; have < d->arena_bytes;) {
        size_t n = have;
        if (have + n > d->arena_bytes)
            n = d->arena_bytes - have;
        fc_buffer_copy_d2d_ctx(d->ctx, d->arena, have, d->arena, 0, n);
        have += n;
    }
    fc_sync_ctx(d->ctx);
    fc_buffer_free(pb);
    return 0;
}

static int dies_init(size_t arena_bytes)
{
    int i, j;
    size_t act = (size_t)MAXN * Q_FF * sizeof(float);
    float *h = malloc(act);

    if (!h)
        return -1;
    for (i = 0; i < MAXN * Q_FF; i++)
        h[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    for (ND = 0; ND < MAXD; ND++) {
        fc_cuda_ctx *c = fc_cuda_init_device(ND);
        if (!c)
            break;
        D[ND].ctx = c;
    }
    if (ND < 1) {
        free(h);
        return -1;
    }
    for (i = 0; i < ND; i++) {
        die_t *d = &D[i];
        d->arena_bytes = arena_bytes;
        d->arena = arena_bytes ? fc_buffer_alloc_ctx(d->ctx, arena_bytes) : NULL;
        d->x = fc_buffer_from_ptr_ctx(d->ctx, h, act);
        d->y[0] = fc_buffer_from_ptr_ctx(d->ctx, h, act);
        d->y[1] = fc_buffer_from_ptr_ctx(d->ctx, h, act);
        d->rcv = fc_buffer_alloc_ctx(d->ctx, act * MAXD);
        d->sum = fc_buffer_alloc_ctx(d->ctx, act);
        for (j = 0; j < EVR; j++)
            d->ev[j] = fc_event_create_ctx(d->ctx);
        if ((arena_bytes && !d->arena) || !d->x || !d->y[0] || !d->y[1] || !d->rcv ||
            !d->sum || !d->ev[EVR - 1]) {
            fprintf(stderr, "die %d: allocation failed\n", i);
            free(h);
            return -1;
        }
        if (arena_bytes && arena_fill(d) != 0) {
            fprintf(stderr, "die %d: arena fill failed\n", i);
            free(h);
            return -1;
        }
    }
    free(h);
    /* P2P only between the two dies of one board (PCI order: 2k, 2k+1). The reference host's
     * driver also grants cross-board access (through the host bridge), but those
     * copies arrive CORRUPT at ~0.3 GB/s (mode peer with TP_PEER_ALL=1 shows it), so
     * cross-board pairs stay unpeered and cuMemcpyPeerAsync stages through host. */
    for (i = 0; i < ND; i++)
        for (j = i + 1; j < ND; j++)
            if (env_int("TP_PEER_ALL", 0) || (i / 2 == j / 2))
                fc_cuda_enable_peer(D[i].ctx, D[j].ctx);
    g_xb = xb_parse(getenv("TP_XB"));
    if (getenv("TP_XB") && g_xb < 0) {
        fprintf(stderr, "TP_XB: unknown method %s\n", getenv("TP_XB"));
        return -1;
    }
    if (g_xb >= 0 || !strcmp(g_mode, "xbv")) {
        fc_cuda_ctx *c[MAXD];
        for (i = 0; i < ND; i++)
            c[i] = D[i].ctx;
        if (xb_init(c, ND, (size_t)MAXN * Q_FF * sizeof(float)) != 0) {
            fprintf(stderr, "xb_init failed\n");
            return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------
 * Weight slices and the matvec
 * ------------------------------------------------------------------ */

typedef struct wslice {
    fc_buffer *w;
    int rows, cols, stride;
} wslice;

static int take(int d, int rows, int cols, wslice *s)
{
    size_t rs = q8_row_bytes(cols), bytes = rs * (size_t)rows;
    if (D[d].bump + bytes > D[d].arena_bytes) {
        fprintf(stderr, "die %d: arena exhausted (%zu + %zu > %zu)\n", d, D[d].bump,
                bytes, D[d].arena_bytes);
        return -1;
    }
    s->w = fc_buffer_offset_view(D[d].arena, D[d].bump);
    s->rows = rows;
    s->cols = cols;
    s->stride = (int)rs;
    D[d].bump += (bytes + ALIGNB - 1) / ALIGNB * ALIGNB;
    return s->w ? 0 : -1;
}

static void mv(int d, const wslice *s, int n, int par)
{
    fc_matvec_q80_nx_ctx(D[d].ctx, s->w, D[d].x, D[d].y[par], s->rows, s->cols,
                         s->stride, n);
}

/* ------------------------------------------------------------------
 * Exchanges. grp[0..g-1] are die indices. Device-side ordering only: each source
 * records an event after its producer, each receiver waits on it and pulls the
 * bytes on ITS OWN stream (cuMemcpyPeerAsync; P2P where enabled, else the driver
 * stages through host memory).
 *   gather(P):  die i owns P/g floats per row; every die ends with all P.
 *   reduce(P):  die i owns a full P partial; every die pulls the g-1 others and
 *               sums in one fixed order (one add launch stands in for the combine).
 * ------------------------------------------------------------------ */

static void xchg_hier(const int *grp, int P, int n, int reduce, int par);

static int same_board(int a, int b) { return a / 2 == b / 2; }

static void xchg(const int *grp, int g, int P, int n, int reduce, int par)
{
    fc_event *ev[MAXD];
    size_t per = reduce ? (size_t)P : (size_t)(P / g);
    size_t bytes = per * (size_t)n * sizeof(float);
    int a, b;

    if (g == 4 && g_xb >= 0) {
        xchg_hier(grp, P, n, reduce, par);
        return;
    }
    for (a = 0; a < g; a++)
        ev[a] = die_record(grp[a]);
    for (b = 0; b < g; b++) {
        int j = grp[b];
        for (a = 0; a < g; a++) {
            fc_buffer *dst;
            if (a == b)
                continue;
            if (g_xb >= 0 && !same_board(grp[a], j)) {
                xb_copy(g_xb, grp[a], D[grp[a]].y[par], 0, j, D[j].rcv, (size_t)a * bytes,
                        bytes, ev[a], par);
                continue;
            }
            fc_event_wait_ctx(D[j].ctx, ev[a]);
            dst = fc_buffer_offset_view(D[j].rcv, (size_t)a * bytes);
            fc_buffer_copy_peer_async(dst, D[grp[a]].y[par], bytes, D[j].ctx);
            fc_buffer_free_view(dst);
        }
        if (reduce)
            fc_add_ctx(D[j].ctx, D[j].rcv, D[j].y[par], D[j].sum, (int)per * n);
    }
}

/* On-board pull: dst range of die j <- src range of its board partner (P2P), after ev. */
static void pull_on_board(int j, fc_buffer *dst, size_t dst_off, fc_buffer *src,
                          size_t src_off, size_t bytes, fc_event *ev)
{
    fc_buffer *t = fc_buffer_offset_view(dst, dst_off);
    fc_buffer *s = fc_buffer_offset_view(src, src_off);
    fc_event_wait_ctx(D[j].ctx, ev);
    fc_buffer_copy_peer_async(t, s, bytes, D[j].ctx);
    fc_buffer_free_view(t);
    fc_buffer_free_view(s);
}

/* out = a + b over floats of three ranges of one die. */
static void add_at(int d, fc_buffer *a, size_t ao, fc_buffer *b, size_t bo, fc_buffer *o,
                   size_t oo, int floats)
{
    fc_buffer *va = fc_buffer_offset_view(a, ao), *vb = fc_buffer_offset_view(b, bo);
    fc_buffer *vo = fc_buffer_offset_view(o, oo);
    fc_add_ctx(D[d].ctx, va, vb, vo, floats);
    fc_buffer_free_view(va);
    fc_buffer_free_view(vb);
    fc_buffer_free_view(vo);
}

/* TP4 over two boards with the fewest cross-board bytes (TP_XB). grp[k] and grp[k^1]
 * share a board; grp[k^2] is k's cross-board partner.
 *   gather(P): each die owns P/4 per row. Cross: k pulls k^2's slice (P/4); on-board:
 *              k pulls k^1's own slice and k^1's cross slice. A board takes P/2 per row
 *              across, the minimum.
 *   reduce(P): each die holds a full P partial, ranks placed so that board A sums the
 *              kernel tree's (V0+V2) and board B (V1+V3) (exact). On-board
 *              reduce-scatter (k keeps half k&1), cross swap of the half sums + add,
 *              on-board gather of the other half. A board takes P per row across. */
static void xchg_hier(const int *grp, int P, int n, int reduce, int par)
{
    const size_t fl = sizeof(float);
    fc_event *ev[4];
    int k;

    if (!reduce) {
        size_t q = (size_t)(P / 4) * (size_t)n * fl;
        for (k = 0; k < 4; k++)
            ev[k] = die_record(grp[k]);
        for (k = 0; k < 4; k++)
            xb_copy(g_xb, grp[k ^ 2], D[grp[k ^ 2]].y[par], 0, grp[k], D[grp[k]].rcv, 0, q,
                    ev[k ^ 2], par);
        for (k = 0; k < 4; k++)
            ev[k] = die_record(grp[k]);
        for (k = 0; k < 4; k++) {
            int j = grp[k], i = grp[k ^ 1];
            pull_on_board(j, D[j].rcv, q, D[i].y[par], 0, q, ev[k ^ 1]);
            pull_on_board(j, D[j].rcv, 2 * q, D[i].rcv, 0, q, ev[k ^ 1]);
        }
        return;
    }
    {
        size_t h = (size_t)(P / 2) * (size_t)n * fl;
        int hf = (P / 2) * n;
        for (k = 0; k < 4; k++)
            ev[k] = die_record(grp[k]);
        for (k = 0; k < 4; k++) {   /* on-board reduce-scatter into sum[0..h) */
            int j = grp[k], i = grp[k ^ 1];
            size_t mine = (size_t)(k & 1) * h;
            pull_on_board(j, D[j].rcv, 0, D[i].y[par], mine, h, ev[k ^ 1]);
            add_at(j, D[j].y[par], mine, D[j].rcv, 0, D[j].sum, 0, hf);
        }
        for (k = 0; k < 4; k++)
            ev[k] = die_record(grp[k]);
        for (k = 0; k < 4; k++) {   /* cross: the other board's half sum, add */
            int j = grp[k];
            xb_copy(g_xb, grp[k ^ 2], D[grp[k ^ 2]].sum, 0, j, D[j].rcv, h, h, ev[k ^ 2],
                    par);
            add_at(j, D[j].sum, 0, D[j].rcv, h, D[j].sum, 0, hf);
        }
        for (k = 0; k < 4; k++)
            ev[k] = die_record(grp[k]);
        for (k = 0; k < 4; k++) {   /* on-board gather of the other half */
            int j = grp[k], i = grp[k ^ 1];
            pull_on_board(j, D[j].rcv, 2 * h, D[i].sum, 0, h, ev[k ^ 1]);
        }
    }
}

/* Stage hop (pipeline boundary): die a's hidden state to die b. */
static void hop(int a, int b, int n, int par)
{
    fc_event *e = die_record(a);
    fc_event_wait_ctx(D[b].ctx, e);
    fc_buffer_copy_peer_async(D[b].x, D[a].y[par], (size_t)n * Q_E * sizeof(float),
                              D[b].ctx);
}

/* ------------------------------------------------------------------
 * mode peer
 * ------------------------------------------------------------------ */

static int mode_peer(void)
{
    int i, j, k;
    const size_t small = (size_t)Q_E * sizeof(float);
    const size_t big = (size_t)64 << 20;

    int bad = 0;

    printf("P2P enabled: %s (TP_PEER_ALL=1 enables every pair the driver grants)\n",
           env_int("TP_PEER_ALL", 0) ? "every pair" : "on-board pairs only");
    printf("\npair   verify  copy20KB_us(drained,1-at-a-time)  copy64MB_GBps\n");
    for (i = 0; i < ND; i++)
        for (j = 0; j < ND; j++) {
            fc_buffer *sa, *sb, *ba, *bb;
            float *hp, *hq;
            double t0, us, gbps;
            int ok = 1;
            if (i == j)
                continue;
            hp = malloc(small);
            hq = malloc(small);
            for (k = 0; k < Q_E; k++)
                hp[k] = (float)(k * 7 + i * 1000 + j);
            sa = fc_buffer_from_ptr_ctx(D[i].ctx, hp, small);
            sb = fc_buffer_alloc_ctx(D[j].ctx, small);
            ba = fc_buffer_alloc_ctx(D[i].ctx, big);
            bb = fc_buffer_alloc_ctx(D[j].ctx, big);
            fc_buffer_copy_peer_async(sb, sa, small, D[j].ctx);
            fc_sync_ctx(D[j].ctx);
            fc_buffer_to_cpu(sb, hq);
            for (k = 0; k < Q_E; k++)
                if (hq[k] != hp[k])
                    ok = 0;
            t0 = now_s();
            for (k = 0; k < 200; k++) {
                fc_buffer_copy_peer_async(sb, sa, small, D[j].ctx);
                fc_sync_ctx(D[j].ctx);
            }
            us = (now_s() - t0) * 1e6 / 200;
            fc_buffer_copy_peer_async(bb, ba, big, D[j].ctx);
            fc_sync_ctx(D[j].ctx);
            t0 = now_s();
            for (k = 0; k < 5; k++)
                fc_buffer_copy_peer_async(bb, ba, big, D[j].ctx);
            fc_sync_ctx(D[j].ctx);
            gbps = 5.0 * (double)big / (now_s() - t0) / 1e9;
            printf("d%d->d%d %-6s  %8.1f                          %6.2f\n", i, j,
                   ok ? "OK" : "FAIL", us, gbps);
            fc_buffer_free(sa);
            fc_buffer_free(sb);
            fc_buffer_free(ba);
            fc_buffer_free(bb);
            free(hp);
            free(hq);
            bad |= !ok;
        }
    return bad;
}

/* ------------------------------------------------------------------
 * mode mv / mvc
 * ------------------------------------------------------------------ */

typedef struct shape {
    const char *name;
    int rows, cols;
    int per_layers;   /* how many layers carry it (48 GDN / 16 attn / 64 all) */
} shape;

static const shape SH[] = {
    { "gdn_qkv",   Q_QKV,   Q_E,    48 },
    { "gdn_z",     Q_CORE,  Q_E,    48 },
    { "gdn_ba",    Q_BA,    Q_E,    48 },
    { "gdn_out",   Q_E,     Q_CORE, 48 },
    { "attn_q",    Q_AQ,    Q_E,    16 },
    { "attn_k",    Q_AKV,   Q_E,    16 },
    { "attn_v",    Q_AKV,   Q_E,    16 },
    { "attn_o",    Q_E,     Q_CORE, 16 },
    { "ffn_gate",  Q_FF,    Q_E,    64 },
    { "ffn_up",    Q_FF,    Q_E,    64 },
    { "ffn_down",  Q_E,     Q_FF,   64 },
    { "lm_head",   Q_VOCAB, Q_E,     1 },
};
#define NSH ((int)(sizeof(SH) / sizeof(SH[0])))

/* Time `iters` launches of one slice on die d (alone), ms per launch. */
static double time_mv(int d, const wslice *s, int n, int iters)
{
    int i;
    double t0;
    mv(d, s, n, 0);
    fc_sync_ctx(D[d].ctx);
    t0 = now_s();
    for (i = 0; i < iters; i++)
        mv(d, s, n, 0);
    fc_sync_ctx(D[d].ctx);
    return (now_s() - t0) * 1e3 / iters;
}

/* The five slices of one shape: full, rows/2, rows/4, cols/2, cols/4. */
static void slice_dims(const shape *sh, int k, int *r, int *c)
{
    static const int rdiv[5] = { 1, 2, 4, 1, 1 }, cdiv[5] = { 1, 1, 1, 2, 4 };
    *r = sh->rows / rdiv[k];
    *c = sh->cols / cdiv[k];
}

static int mode_mv(void)
{
    static const char *kn[5] = { "full", "rows/2", "rows/4", "cols/2", "cols/4" };
    int reps = env_int("TP_REPS", 5), iters = env_int("TP_ITERS", 40);
    int si, k, ni, r;
    double *t = malloc(sizeof(double) * (size_t)reps);

    printf("die 0 alone, Q8_0 G4, median of %d reps x %d launches\n", reps, iters);
    printf("%-9s %-7s %13s %3s %10s %8s\n", "weight", "slice", "shape", "n", "ms",
           "GB/s");
    for (si = 0; si < NSH; si++)
        for (k = 0; k < 5; k++) {
            wslice s;
            int rr, cc;
            slice_dims(&SH[si], k, &rr, &cc);
            D[0].bump = 0;
            if (take(0, rr, cc, &s) != 0)
                return 1;
            for (ni = 0; ni < g_nn; ni++) {
                int n = g_ns[ni];
                int it = SH[si].rows == Q_VOCAB ? iters / 4 + 1 : iters;
                double ms;
                for (r = 0; r < reps; r++)
                    t[r] = time_mv(0, &s, n, it);
                ms = median(t, reps);
                printf("%-9s %-7s %6dx%-6d %3d %10.4f %8.1f\n", SH[si].name, kn[k], rr,
                       cc, n, ms, (double)s.stride * rr / (ms * 1e6));
            }
            fc_buffer_free_view(s.w);
        }
    free(t);
    return 0;
}

/* Quarter slices of the per-layer FFN + GDN set, die 0 alone vs every die at once. */
static int mode_mvc(void)
{
    int reps = env_int("TP_REPS", 5), iters = env_int("TP_ITERS", 40);
    static const int pick[] = { 0, 1, 3, 8, 9, 10 };  /* qkv z out gate up down */
    wslice s[MAXD][6];
    int d, k, ni, r, i;
    double *ta = malloc(sizeof(double) * (size_t)reps);
    double *tc = malloc(sizeof(double) * (size_t)reps);

    for (d = 0; d < ND; d++) {
        D[d].bump = 0;
        for (k = 0; k < 6; k++) {
            const shape *sh = &SH[pick[k]];
            /* row slices for the column-parallel weights, col slices for out/down */
            int rr = (pick[k] == 3 || pick[k] == 10) ? sh->rows : sh->rows / 4;
            int cc = (pick[k] == 3 || pick[k] == 10) ? sh->cols / 4 : sh->cols;
            if (take(d, rr, cc, &s[d][k]) != 0)
                return 1;
        }
    }
    printf("one GDN layer's quarter slices (qkv z out gate up down), %d dies\n", ND);
    printf("%3s %12s %12s %8s\n", "n", "alone_ms", "concurrent", "ratio");
    for (ni = 0; ni < g_nn; ni++) {
        int n = g_ns[ni];
        for (r = 0; r < reps; r++) {
            double t0 = now_s();
            for (i = 0; i < iters; i++)
                for (k = 0; k < 6; k++)
                    mv(0, &s[0][k], n, 0);
            fc_sync_ctx(D[0].ctx);
            ta[r] = (now_s() - t0) * 1e3 / iters;
            t0 = now_s();
            for (i = 0; i < iters; i++)
                for (d = 0; d < ND; d++)
                    for (k = 0; k < 6; k++)
                        mv(d, &s[d][k], n, 0);
            for (d = 0; d < ND; d++)
                fc_sync_ctx(D[d].ctx);
            tc[r] = (now_s() - t0) * 1e3 / iters;
        }
        {
            double a = median(ta, reps), c = median(tc, reps);
            printf("%3d %12.4f %12.4f %8.3f\n", n, a, c, c / a);
        }
    }
    free(ta);
    free(tc);
    return 0;
}

/* ------------------------------------------------------------------
 * mode xchg
 * ------------------------------------------------------------------ */

typedef struct xgrp {
    const char *name;
    int g, ngrp;          /* group size, number of concurrent groups */
    int dies[MAXD];       /* ngrp groups of g, back to back */
} xgrp;

static double xchain(const xgrp *x, int P, int n, int reduce, int len, int do_x)
{
    int k, q, a;
    double t0;

    for (a = 0; a < x->g * x->ngrp; a++)
        fc_sync_ctx(D[x->dies[a]].ctx);
    t0 = now_s();
    for (k = 0; k < len; k++)
        for (q = 0; q < x->ngrp; q++) {
            const int *grp = x->dies + q * x->g;
            for (a = 0; a < x->g; a++) {
                int d = grp[a];
                fc_add_ctx(D[d].ctx, D[d].x, D[d].x, D[d].y[k & 1], P * n);
            }
            if (do_x)
                xchg(grp, x->g, P, n, reduce, k & 1);
        }
    for (a = 0; a < x->g * x->ngrp; a++)
        fc_sync_ctx(D[x->dies[a]].ctx);
    return (now_s() - t0) * 1e6 / len;
}

static int mode_xchg(void)
{
    const xgrp G[] = {
        { "pair d0-d1 (on-board)", 2, 1, { 0, 1 } },
        { "pair d2-d3 (on-board)", 2, 1, { 2, 3 } },
        { "pair d1-d2 (cross)",    2, 1, { 1, 2 } },
        { "cross 0-2 + 1-3",       2, 2, { 0, 2, 1, 3 } },
        { "both pairs at once",    2, 2, { 0, 1, 2, 3 } },
        { "quad d0-d3",            4, 1, { 0, 1, 2, 3 } },
    };
    const struct { const char *name; int P, reduce; } K[] = {
        { "gather 5120",  Q_E,    0 },
        { "gather 6144",  Q_CORE, 0 },
        { "gather 17408", Q_FF,   0 },
        { "reduce 5120",  Q_E,    1 },
    };
    int reps = env_int("TP_REPS", 5), len = env_int("TP_ITERS", 128);
    int gi, ki, ni, r;
    double *tb = malloc(sizeof(double) * (size_t)reps);
    double *tx = malloc(sizeof(double) * (size_t)reps);

    printf("dependent chain of %d exchanges, median of %d reps; us per exchange "
           "(chain minus add-only baseline)\ncross-board: %s\n", len, reps,
           g_xb >= 0 ? xb_name(g_xb) : "cuMemcpyPeerAsync (driver-staged), quad all-to-all");
    printf("%-22s %-13s %3s %9s %9s %9s\n", "group", "kind", "n", "base_us", "chain_us",
           "xchg_us");
    for (gi = 0; gi < (int)(sizeof(G) / sizeof(G[0])); gi++) {
        if (G[gi].g * G[gi].ngrp > ND || G[gi].dies[G[gi].g * G[gi].ngrp - 1] >= ND)
            continue;
        for (ki = 0; ki < (int)(sizeof(K) / sizeof(K[0])); ki++)
            for (ni = 0; ni < g_nn; ni++) {
                int n = g_ns[ni];
                double b, c;
                xchain(&G[gi], K[ki].P, n, K[ki].reduce, 8, 1);   /* warm */
                for (r = 0; r < reps; r++) {
                    tb[r] = xchain(&G[gi], K[ki].P, n, K[ki].reduce, len, 0);
                    tx[r] = xchain(&G[gi], K[ki].P, n, K[ki].reduce, len, 1);
                }
                b = median(tb, reps);
                c = median(tx, reps);
                printf("%-22s %-13s %3d %9.1f %9.1f %9.1f\n", G[gi].name, K[ki].name, n,
                       b, c, c - b);
            }
    }
    free(tb);
    free(tx);
    return 0;
}

/* ------------------------------------------------------------------
 * mode step
 * ------------------------------------------------------------------ */

enum { SC_NONE = 0, SC_E4, SC_E3, SC_M2 };

typedef struct cfg {
    const char *name;
    int stages, g, scheme;   /* stage s = dies s*g .. s*g+g-1, layers split evenly */
} cfg;

/* Per die, per layer: up to 7 matvec slices in issue order (see issue_layer). */
#define MAXM 8
static wslice W[MAXD][Q_LAYERS][MAXM];
static wslice WH[MAXD];

static int prep_cfg(const cfg *c)
{
    int s, a, l, lps = Q_LAYERS / c->stages, G = c->g;
    for (a = 0; a < ND; a++)
        D[a].bump = 0;
    for (s = 0; s < c->stages; s++)
        for (a = 0; a < G; a++) {
            int d = s * G + a;
            for (l = s * lps; l < (s + 1) * lps; l++) {
                wslice *w = W[d][l];
                int m = 0, out_rows = Q_E, out_cols = Q_CORE, dn_rows = Q_E,
                    dn_cols = Q_FF;
                if (G > 1 && c->scheme == SC_M2)
                    out_cols = Q_CORE / G;
                else if (G > 1)
                    out_rows = Q_E / G;
                if (G > 1 && c->scheme == SC_E4)
                    dn_rows = Q_E / G;
                else if (G > 1)
                    dn_cols = Q_FF / G;
                if (is_attn(l)) {
                    if (take(d, Q_AQ / G, Q_E, &w[m++]) || take(d, Q_AKV / G, Q_E, &w[m++]) ||
                        take(d, Q_AKV / G, Q_E, &w[m++]))
                        return -1;
                } else {
                    if (take(d, Q_QKV / G, Q_E, &w[m++]) || take(d, Q_CORE / G, Q_E, &w[m++]) ||
                        take(d, Q_BA / G, Q_E, &w[m++]))
                        return -1;
                }
                if (take(d, out_rows, out_cols, &w[m++]) ||
                    take(d, Q_FF / G, Q_E, &w[m++]) || take(d, Q_FF / G, Q_E, &w[m++]) ||
                    take(d, dn_rows, dn_cols, &w[m++]))
                    return -1;
            }
            if (s == c->stages - 1 && take(d, Q_VOCAB / G, Q_E, &WH[d]))
                return -1;
        }
    return 0;
}

static void free_cfg(const cfg *c)
{
    int s, a, l, m, lps = Q_LAYERS / c->stages;
    for (s = 0; s < c->stages; s++)
        for (a = 0; a < c->g; a++) {
            int d = s * c->g + a;
            for (l = s * lps; l < (s + 1) * lps; l++)
                for (m = 0; m < 7; m++)
                    fc_buffer_free_view(W[d][l][m].w);
            if (s == c->stages - 1)
                fc_buffer_free_view(WH[d].w);
        }
}

static int g_par;
static int g_noxchg;

/* One layer on one TP group: mixer projections, [exchanges], out, FFN, [exchanges]. */
static void issue_layer(const cfg *c, const int *grp, int l, int n)
{
    int G = c->g * !g_noxchg, a;
#define ALL_MV(m) for (a = 0; a < c->g; a++) mv(grp[a], &W[grp[a]][l][m], n, g_par)
    ALL_MV(0);
    ALL_MV(1);
    ALL_MV(2);
    if (G > 1 && c->scheme != SC_M2) {
        xchg(grp, G, Q_CORE, n, 0, g_par);
        g_par ^= 1;
    }
    ALL_MV(3);
    if (G > 1) {
        xchg(grp, G, Q_E, n, c->scheme == SC_M2, g_par);
        g_par ^= 1;
    }
    ALL_MV(4);
    ALL_MV(5);
    if (G > 1 && c->scheme == SC_E4) {
        xchg(grp, G, Q_FF, n, 0, g_par);
        g_par ^= 1;
    }
    ALL_MV(6);
    if (G > 1) {
        xchg(grp, G, Q_E, n, c->scheme != SC_E4, g_par);
        g_par ^= 1;
    }
#undef ALL_MV
}

/* One decode step. Stage s's dies start once stage s-1's dies hop the hidden state. */
static void issue_step(const cfg *c, int n)
{
    int s, a, l, lps = Q_LAYERS / c->stages, G = c->g;
    int grp[MAXD];
    for (s = 0; s < c->stages; s++) {
        for (a = 0; a < G; a++)
            grp[a] = s * G + a;
        if (s > 0 && !g_noxchg)
            for (a = 0; a < G; a++)
                hop((s - 1) * G + a, s * G + a, n, g_par);
        for (l = s * lps; l < (s + 1) * lps; l++)
            issue_layer(c, grp, l, n);
    }
    for (a = 0; a < G; a++) {
        int d = (c->stages - 1) * G + a;
        mv(d, &WH[d], n, g_par);
    }
}

/* TP_SOAK=<seconds>: one config (the first TP_CFG match) at the first TP_NS row count,
 * synced steps back to back, ms/step printed every ~10 s with a wall-clock stamp so the
 * nvidia-smi log can be lined up (heat and power caps act over minutes, not bursts). */
static int soak(const cfg *C, int nc, const char *want)
{
    int ci, d, n = g_ns[0], secs = env_int("TP_SOAK", 0);
    for (ci = 0; ci < nc; ci++)
        if (!want || !*want || strstr(want, C[ci].name))
            break;
    if (ci == nc || prep_cfg(&C[ci]) != 0)
        return 1;
    printf("soak %s n=%d for %d s\n%-8s %8s %8s\n", C[ci].name, n, secs, "time", "steps",
           "ms/step");
    {
        double t_end = now_s() + secs;
        while (now_s() < t_end) {
            double t0 = now_s();
            int k = 0;
            time_t wall;
            char ts[16];
            while (now_s() - t0 < 10.0) {
                issue_step(&C[ci], n);
                for (d = 0; d < ND; d++)
                    fc_sync_ctx(D[d].ctx);
                k++;
            }
            wall = time(NULL);
            strftime(ts, sizeof(ts), "%H:%M:%S", localtime(&wall));
            printf("%-8s %8d %8.2f\n", ts, k, (now_s() - t0) * 1e3 / k);
            fflush(stdout);
        }
    }
    free_cfg(&C[ci]);
    return 0;
}

static int mode_step(void)
{
    const cfg C[] = {
        { "split", 4, 1, SC_NONE },
        { "tp2e4", 2, 2, SC_E4 }, { "tp2e3", 2, 2, SC_E3 }, { "tp2m2", 2, 2, SC_M2 },
        { "tp4e4", 1, 4, SC_E4 }, { "tp4e3", 1, 4, SC_E3 }, { "tp4m2", 1, 4, SC_M2 },
    };
    const char *want = getenv("TP_CFG");
    int steps = env_int("TP_STEPS", 6), reps = env_int("TP_REPS", 3);

    g_noxchg = env_int("TP_NOXCHG", 0);
    int ci, ni, k, r, d;

    printf("emulated decode step (matvecs + LM head + exchanges/hops; no attention, "
           "SSM core or norms)\nhost syncs all dies after every step (as the serve "
           "does for sampling); median of %d reps x %d steps\nexchanges: %s; cross-board: "
           "%s\n", reps, steps, g_noxchg ? "OFF (TP_NOXCHG)" : "on",
           g_xb >= 0 ? xb_name(g_xb) : "cuMemcpyPeerAsync");
    if (env_int("TP_SOAK", 0) > 0)
        return soak(C, (int)(sizeof(C) / sizeof(C[0])), want);
    printf("%-6s %3s %10s %10s %10s\n", "cfg", "n", "step_ms", "issue_ms", "pipe_ms");
    for (ci = 0; ci < (int)(sizeof(C) / sizeof(C[0])); ci++) {
        const cfg *c = &C[ci];
        if (want && *want && !strstr(want, c->name))
            continue;
        if (c->stages * c->g > ND)
            continue;
        if (prep_cfg(c) != 0)
            return 1;
        for (ni = 0; ni < g_nn; ni++) {
            int n = g_ns[ni];
            double ts[16], ti[16], tp[16];
            issue_step(c, n);   /* warm */
            for (d = 0; d < ND; d++)
                fc_sync_ctx(D[d].ctx);
            for (r = 0; r < reps && r < 16; r++) {
                double t0, tiss = 0.0, t1;
                t0 = now_s();
                for (k = 0; k < steps; k++) {
                    double a0 = now_s();
                    issue_step(c, n);
                    tiss += now_s() - a0;
                    for (d = 0; d < ND; d++)
                        fc_sync_ctx(D[d].ctx);
                }
                ts[r] = (now_s() - t0) * 1e3 / steps;
                ti[r] = tiss * 1e3 / steps;
                /* back to back, the last stage's dies feeding the first stage's next
                 * step through device events: GPU-bound when issue keeps up */
                t1 = now_s();
                for (k = 0; k < steps; k++) {
                    int a;
                    if (k > 0)
                        for (a = 0; a < c->g; a++) {
                            int b;
                            for (b = 0; b < c->g; b++) {
                                fc_event *e = die_record((c->stages - 1) * c->g + b);
                                fc_event_wait_ctx(D[a].ctx, e);
                            }
                        }
                    issue_step(c, n);
                }
                for (d = 0; d < ND; d++)
                    fc_sync_ctx(D[d].ctx);
                tp[r] = (now_s() - t1) * 1e3 / steps;
            }
            {
                time_t wall = time(NULL);
                char tsb[16];
                strftime(tsb, sizeof(tsb), "%H:%M:%S", localtime(&wall));
                printf("%s ", tsb);
            }
            printf("%-6s %3d %10.2f %10.2f %10.2f\n", c->name, n, median(ts, reps),
                   median(ti, reps), median(tp, reps));
            fflush(stdout);
        }
        free_cfg(c);
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "peer";
    g_mode = mode;
    size_t arena = 0;
    int rc;

    parse_ns();
    srand(1234);
    if (!strcmp(mode, "mv"))
        arena = (size_t)1600 << 20;             /* the LM head is 1.37 GB */
    else if (!strcmp(mode, "mvc"))
        arena = (size_t)256 << 20;
    else if (!strcmp(mode, "step"))
        arena = (size_t)env_int("TP_ARENA_MB", 8200) << 20;
    if (dies_init(arena) != 0) {
        fprintf(stderr, "die init failed\n");
        return 2;
    }
    printf("bench_tp_q38 %s: %d dies\n", mode, ND);
    if (!strcmp(mode, "peer"))
        rc = mode_peer();
    else if (!strcmp(mode, "mv"))
        rc = mode_mv();
    else if (!strcmp(mode, "mvc"))
        rc = mode_mvc();
    else if (!strcmp(mode, "xchg"))
        rc = mode_xchg();
    else if (!strcmp(mode, "step"))
        rc = mode_step();
    else if (!strcmp(mode, "xbv")) {
        fc_cuda_ctx *c[MAXD];
        int i;
        for (i = 0; i < ND; i++)
            c[i] = D[i].ctx;
        rc = xb_mode_verify(c, ND);
    } else {
        fprintf(stderr, "unknown mode %s (peer|mv|mvc|xchg|step|xbv)\n", mode);
        rc = 2;
    }
    return rc;
}
