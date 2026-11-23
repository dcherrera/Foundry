/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* bench_peer_stream — does scheme 9a (PLX weight-streaming) actually convert?
 *
 * THE SCHEME. saber runs a
 * layer split at 20, so each die idles ~26 ms of a 52.6 ms step. 9a keeps every
 * weight exactly where it is and lets the IDLE die help: during die A's turn,
 * die B's kernels dereference die A's VRAM over the peer link and compute a
 * disjoint tail of each matvec's output rows, writing straight into die A's
 * output buffer. No weight split, no replication, no re-layout, no new kernels.
 *
 * WHAT IS ALREADY MEASURED and is NOT re-derived here (same doc, §5):
 *   - the peer link is 12.41 GB/s copy-engine, 9.3-11.3 GB/s kernel-issued.
 *   - fc_matvec_q3k draws 91.6 GB/s of die A's 171.2 GB/s Q3_K layout ceiling,
 *     so it is instruction-issue bound with ~79 GB/s of DRAM headroom spare.
 *   - die B is therefore ~8-9x slower per byte than die A on die A's weights,
 *     which puts the optimal split at f ~ 0.10 and the gross prize at ~1.11x on
 *     the 35.5 ms matvec pool (~3.5 ms of a 52.6 ms step).
 *
 * THE ONLY QUESTION LEFT, and the only one this bench answers: 3.5 ms of gross
 * saving spread across ~120-320 matvec launches per token is ~11-30 us per
 * launch, and a cross-die ordering costs 0.7-30 us depending on which primitive
 * you pick. THE ENTIRE LEVER LIVES INSIDE THAT GAP. So this measures the
 * economics directly, on the real kernel at the real shapes, rather than
 * modelling them:
 *
 *   P1  what does die B's remote read cost when x is ALSO remote? (a real 9a
 *       cannot afford a 10 us cuMemcpyPeer of x per group, so it must read x
 *       across the link too -- unless that is ruinous, which is measurable.)
 *   P2  the cost of the two orderings 9a can actually use, in a real
 *       producer/consumer where die B writes into die A's buffer:
 *       host drain (fc_sync_ctx) vs device-ordered (fc_event_wait_ctx).
 *       NOTE the campaign's "use prod_pattern, not cuStreamWaitEvent" verdict
 *       was measured on IDLE dies at a POINT barrier. 9a is different: it needs
 *       the host to keep running ahead, so a drain and a stream-wait are not
 *       interchangeable and both are measured under load.
 *   P3  can a cross-die fork/join be captured in ONE cuda graph? If yes, 9a
 *       pays no ordering and no fragmentation at all. Never probed before
 *       (§9 item 9) -- only the peer COPY was, and that cannot (error 900).
 *   P4  THE NUMBER: a 40-layer micro-model of saber's dense matvec chain,
 *       baseline (all on die A, one captured graph per layer, exactly as the deployment
 *       runs) vs 9a (split, G ordering groups per layer, fragment graphs on
 *       both dies), sweeping the split fraction f and the group count G.
 *
 * CONTROLS, because this project has produced green-but-vacuous gates:
 *   - the two contexts must land on DIFFERENT devices with peer ENABLED.
 *   - f = 0 must reproduce the baseline within noise (the null arm).
 *   - every split arm's output must be BIT-IDENTICAL to the baseline's. A row
 *     split reassociates nothing: each output row is computed by the same
 *     kernel over the same weights, only on a different die. Drift would mean
 *     the remote path is wrong, not that it is approximate.
 *   - PS_POISON=1 skips die B's launches while still claiming the split. The
 *     verifier MUST go red. Run it before believing any green.
 *
 * WHAT THIS BENCH DOES NOT CLAIM. The eight ops below are wired with private
 * x/y buffers, so the bench is free to group them any way it likes; a real
 * layer's dependency chain is what fixes G (about 4 for saber: {qg,k,v} |
 * {attn_out} | {shexp_gu,moe_gu} | {shexp_dn,moe_dn}). The bench prices the
 * cost of G orderings against the saving from the split -- it does not prove a
 * given G is dependency-legal. G=4 is the honest row to read.
 *
 * ==================================================================
 * PART 2 (2026-08-09) — the load-balance equation, and R_B as the target
 * ==================================================================
 *
 * Part 1 reported its verdict from f = 0.10, which is not an operating point
 * anyone would choose, and never wrote down the load balance. Total time is
 * max(t_A, t_B); die A takes share (1-f) at rate R_A, die B takes f at R_B, so
 * they finish together at
 *
 *     f* = R_B / (R_A + R_B)          speedup = 1 + R_B / R_A
 *
 * with the twist that R_B is itself a function of f (a thinner slice reads
 * SLOWER, because the remote path is latency-bound and a thin slice cannot
 * keep enough requests in flight): f* is a FIXED POINT, not a formula. P5
 * measures R_B(f) and R_A(f), P7 solves the fixed point and measures there.
 *
 * And the prize is not the balance, it is R_B itself:
 *
 *     link, copy engine (measured)      12.41 GB/s
 *     link, kernel-issued (measured)     9.3-11.3 GB/s
 *     die B achieved (part 1)            2.66-4.22 GB/s   <-- 21-34%
 *
 * A faster die B raises R_B, which raises f*, which earns die B a bigger
 * share: R_B 2.5 -> f* .035 -> +3.6%; R_B 6 -> f* .080 -> +8.7%; R_B 11
 * (link-saturated) -> f* .138 -> +16%. So P6 is an R_B LEDGER: one row per
 * optimisation, each measured in isolation, against the 9.3-12.41 GB/s the
 * link is known to deliver. The levers, all aimed at memory-level
 * parallelism rather than at moving fewer bytes:
 *
 *   PS_XMODE   how die B gets x. remote (part 1's choice) makes die B drag an
 *              8 KB activation across the link once PER BLOCK; mirror is the
 *              free-x bound; stage does ONE packed peer copy per ordering
 *              group into die B's own memory, which is what a real 9a would
 *              do (part 1 priced a PER-OP copy and correctly rejected it).
 *   PS_BKERN   which matvec die B runs. The stock kernel issues one
 *              super-block's loads per warp and then consumes them; the
 *              fc_matvec_q3k_ps_r{1,2}p{1,2,4,8} variants issue up to 8
 *              super-blocks ahead (bytes in flight scale with the prefetch
 *              depth) and optionally halve rows-per-warp to double the warp
 *              count. All are BIT-EXACT with the stock kernel, and the
 *              bit-exactness gate below is what proves it.
 *   PS_FSHAPE  where die B's rows come from. Uniform f gives the small ops
 *              (k, v: 512 rows) a slice of 25 rows = 2 blocks, which cannot
 *              fill a link; shape=1 concentrates die B's share of each
 *              dependency group onto that group's LARGEST op instead.
 */

#include "../fc_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PS_Q3K          11
#define PS_EVENT_RING   512
#define PS_MAX_OPS      8
#define PS_MAX_GROUPS   8

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

static double dmedian(double *v, int n)
{
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    return v[n / 2];
}

/* Mean and half-width of a 95% CI over n samples (normal approximation; n is
 * >= 20 everywhere it is used). Reporting a point estimate against a +-1.3%
 * noise floor was part 1's other reporting error. */
static void mean_ci95(const double *v, int n, double *mean, double *half)
{
    double s = 0.0, ss = 0.0;
    int i;
    for (i = 0; i < n; i++) s += v[i];
    *mean = s / n;
    for (i = 0; i < n; i++) ss += (v[i] - *mean) * (v[i] - *mean);
    *half = (n > 1) ? 1.96 * sqrt(ss / (n - 1)) / sqrt((double)n) : 0.0;
}

/* ---- how die B gets x, and which matvec it runs ---- */
enum { PS_X_REMOTE = 0, PS_X_MIRROR, PS_X_STAGE };
static const char *ps_xmode_name[] = { "remote", "mirror", "stage" };
static int ps_xmode = PS_X_REMOTE;
static int ps_bkern = -1;            /* -1 = stock fc_matvec_q3k */
static int ps_fshape = 0;            /* 0 = uniform f, 1 = biggest op first */
/* PS_XGARBLE=1 garbles die B's activation mirror even in the free-x bound
 * arm, which MUST then go red. It is the falsification of the staged-x gate:
 * without it, "staged x stayed bit-exact" only shows the mirror happened to
 * be right, not that the staging copy did anything. */
static int ps_xgarble = 0;

static const char *ps_bkern_name(int v)
{
    const char *n = fc_matvec_q3k_ps_name(v);
    return n ? n : "stock";
}

/* ------------------------------------------------------------------
 * Weight/activation construction (same recipe as bench_cross_die, so the
 * two benches' absolute numbers are comparable).
 * ------------------------------------------------------------------ */

static size_t q3k_raw_row(int cols) { return 110u * (size_t)(cols / 256); }

static fc_buffer *ps_make_weight(fc_cuda_ctx *ctx, int rows, int cols,
                                 int *stride)
{
    size_t raw_stride = q3k_raw_row(cols);
    size_t rs = fc_repack_row_bytes(PS_Q3K, cols);
    unsigned char *raw = malloc(raw_stride * (size_t)rows);
    unsigned char *packed = malloc(rs * (size_t)rows);
    fc_buffer *b = NULL;
    size_t i;

    if (!raw || !packed)
        goto out;
    for (i = 0; i < raw_stride * (size_t)rows; i++)
        raw[i] = (unsigned char)(rand() & 0xFF);
    /* Keep every block's fp16 scale away from inf/nan/denormal (Q3_K puts it
     * at byte 108 of each 110-byte block) so no arm lands on a slow path. */
    for (i = 0; i + 110 <= raw_stride * (size_t)rows; i += 110) {
        unsigned short h = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        raw[i + 108] = (unsigned char)(h & 0xFF);
        raw[i + 109] = (unsigned char)(h >> 8);
    }
    if (fc_repack_weight(PS_Q3K, raw, packed, rows, cols) != 0)
        goto out;
    b = fc_buffer_from_ptr_ctx(ctx, packed, rs * (size_t)rows);
    *stride = (int)rs;
out:
    free(raw);
    free(packed);
    return b;
}

/* ------------------------------------------------------------------
 * The rig: one saber-shaped dense matvec chain, all weights on die A.
 * ------------------------------------------------------------------ */

typedef struct ps_op {
    const char *name;
    int rows, cols;
    int stride;
    int group;                  /* dependency group at G = PS_MAX_GROUPS */
    fc_buffer *w;               /* on die A */
    fc_buffer *x_a, *x_b;       /* views into the packed x on die A / die B */
    fc_buffer *y;               /* output, on die A (die B writes here too) */
    fc_buffer *w_tail, *y_tail; /* offset views for die B's row slice */
    int rows_b;                 /* rows die B currently owns */
    size_t x_off;               /* byte offset of this op's x in the pack */
} ps_op;

/* saber's per-layer dense matvec chain. The last two are PROXIES sized to the
 * measured ms of the grouped MoE kernels (moe_gateup 0.1436 ms ~ qg 0.1492;
 * moe_down + moe_rowfb 0.110 ms ~ 5120 rows), because the routed-expert
 * kernels take a different argument set. Labelled as proxies everywhere. */
static ps_op ps_ops[PS_MAX_OPS] = {
    { "qg        8192x2048", 8192, 2048, 0, 0, 0,0,0,0,0,0, 0, 0 },
    { "k          512x2048",  512, 2048, 0, 0, 0,0,0,0,0,0, 0, 0 },
    { "v          512x2048",  512, 2048, 0, 0, 0,0,0,0,0,0, 0, 0 },
    { "attn_out  2048x4096", 2048, 4096, 0, 1, 0,0,0,0,0,0, 0, 0 },
    { "shexp_gu  1024x2048", 1024, 2048, 0, 2, 0,0,0,0,0,0, 0, 0 },
    { "moe_gu*   8192x2048", 8192, 2048, 0, 2, 0,0,0,0,0,0, 0, 0 },
    { "shexp_dn  2048x512",  2048,  512, 0, 3, 0,0,0,0,0,0, 0, 0 },
    { "moe_dn*   5120x2048", 5120, 2048, 0, 3, 0,0,0,0,0,0, 0, 0 },
};

typedef struct ps_rig {
    fc_cuda_ctx *a, *b;
    fc_event *ev_a[PS_EVENT_RING];
    fc_event *ev_b[PS_EVENT_RING];
    int have_events, have_stream_wait;
    size_t weight_bytes;
    /* ONE contiguous activation pack per die, so PS_X_STAGE can bring a whole
     * dependency group's x across the link in a SINGLE peer copy. Part 1
     * priced a per-op copy (8/layer) and rejected it; per-group is 1-4/layer
     * and is what a real build would issue. */
    fc_buffer *xpack_a, *xpack_b;
    size_t xpack_bytes;
    size_t grp_off[PS_MAX_GROUPS + 1];  /* group g owns [off[g], off[g+1]) */
} ps_rig;

static int rig_init(ps_rig *r)
{
    int i;
    memset(r, 0, sizeof(*r));
    r->a = fc_cuda_init_device(0);
    r->b = fc_cuda_init_device(1);
    if (!r->a || !r->b) {
        fprintf(stderr, "need TWO visible dies (CUDA_VISIBLE_DEVICES=<a>,<b>)\n");
        return -1;
    }
    if (fc_cuda_ctx_device_id(r->a) == fc_cuda_ctx_device_id(r->b)) {
        fprintf(stderr, "both contexts landed on device %d — not a die pair\n",
                fc_cuda_ctx_device_id(r->a));
        return -1;
    }
    if (fc_cuda_enable_peer(r->a, r->b) != 0) {
        fprintf(stderr, "fc_cuda_enable_peer FAILED — every cross-die number "
                        "below would be a host bounce, not a peer access\n");
        return -1;
    }
    for (i = 0; i < PS_EVENT_RING; i++) {
        r->ev_a[i] = fc_event_create_ctx(r->a);
        r->ev_b[i] = fc_event_create_ctx(r->b);
    }
    r->have_events = (r->ev_a[0] != NULL && r->ev_b[0] != NULL);
    r->have_stream_wait = r->have_events &&
        (fc_event_record_ctx(r->a, r->ev_a[0]) == 0) &&
        (fc_event_wait_ctx(r->b, r->ev_a[0]) == 0);
    fc_sync_ctx(r->a);
    fc_sync_ctx(r->b);
    return 0;
}

static int ops_alloc(ps_rig *r)
{
    float *h;
    size_t off = 0;
    int i, g;

    for (i = 0; i < PS_MAX_OPS; i++) {
        ps_ops[i].x_off = off;
        off += (size_t)ps_ops[i].cols * sizeof(float);
    }
    r->xpack_bytes = off;
    /* Groups are contiguous because ps_ops is declared in group order; assert
     * that rather than assume it, and record each group's byte range. */
    for (g = 0; g <= PS_MAX_GROUPS; g++)
        r->grp_off[g] = (g == 0) ? 0 : r->xpack_bytes;
    for (i = PS_MAX_OPS - 1; i >= 0; i--)
        r->grp_off[ps_ops[i].group] = ps_ops[i].x_off;
    for (g = 1; g < PS_MAX_GROUPS; g++)
        if (r->grp_off[g] < r->grp_off[g - 1]) {
            fprintf(stderr, "ps_ops must be declared in group order\n");
            return -1;
        }

    h = malloc(r->xpack_bytes);
    if (!h)
        return -1;
    for (i = 0; i < (int)(r->xpack_bytes / sizeof(float)); i++)
        h[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    r->xpack_a = fc_buffer_from_ptr_ctx(r->a, h, r->xpack_bytes);
    r->xpack_b = fc_buffer_alloc_ctx(r->b, r->xpack_bytes);
    free(h);
    if (!r->xpack_a || !r->xpack_b)
        return -1;
    /* Mirror once, so PS_X_MIRROR (the free-x bound) is available. */
    if (fc_buffer_copy_peer(r->xpack_b, r->xpack_a, r->xpack_bytes) != 0)
        return -1;

    r->weight_bytes = 0;
    for (i = 0; i < PS_MAX_OPS; i++) {
        ps_op *o = &ps_ops[i];
        o->w = ps_make_weight(r->a, o->rows, o->cols, &o->stride);
        o->y = fc_buffer_alloc_ctx(r->a, (size_t)o->rows * sizeof(float));
        o->x_a = fc_buffer_offset_view(r->xpack_a, o->x_off);
        o->x_b = fc_buffer_offset_view(r->xpack_b, o->x_off);
        if (!o->w || !o->x_a || !o->y || !o->x_b) {
            fprintf(stderr, "alloc failed for %s\n", o->name);
            return -1;
        }
        r->weight_bytes += (size_t)o->rows * (size_t)o->stride;
    }
    fc_sync_ctx(r->a);
    fc_sync_ctx(r->b);
    return 0;
}

static void ops_free(ps_rig *r)
{
    int i;
    for (i = 0; i < PS_MAX_OPS; i++) {
        ps_op *o = &ps_ops[i];
        fc_buffer_free_view(o->w_tail);
        fc_buffer_free_view(o->y_tail);
        fc_buffer_free_view(o->x_a);
        fc_buffer_free_view(o->x_b);
        o->w_tail = o->y_tail = o->x_a = o->x_b = NULL;
        fc_buffer_free(o->w);
        fc_buffer_free(o->y);
        o->w = o->y = NULL;
    }
    fc_buffer_free(r->xpack_a);
    fc_buffer_free(r->xpack_b);
    r->xpack_a = r->xpack_b = NULL;
}

/* Rebuild die B's weight/output views for its current rows_b. */
static int ops_rebuild_views(void)
{
    int i;
    for (i = 0; i < PS_MAX_OPS; i++) {
        ps_op *o = &ps_ops[i];
        fc_buffer_free_view(o->w_tail);
        fc_buffer_free_view(o->y_tail);
        o->w_tail = o->y_tail = NULL;
        if (o->rows_b <= 0)
            continue;
        o->w_tail = fc_buffer_offset_view(o->w,
                        (size_t)(o->rows - o->rows_b) * (size_t)o->stride);
        o->y_tail = fc_buffer_offset_view(o->y,
                        (size_t)(o->rows - o->rows_b) * sizeof(float));
        if (!o->w_tail || !o->y_tail)
            return -1;
    }
    return 0;
}

/* Set the split: die B takes the LAST rows_b rows of an op.
 *
 * shape 0 (uniform): every op gives up the same row FRACTION. Simple, and
 *   what part 1 measured — but it hands the 512-row ops a 25-row slice, which
 *   is 2 blocks and cannot keep a link busy.
 * shape 1 (concentrated): each dependency GROUP gives up the same BYTE
 *   fraction, taken from that group's largest op first. Die B ends up running
 *   fewer, fatter kernels for the same total bytes, which is strictly more
 *   memory-level parallelism per launch. The group-wise budget keeps the
 *   per-group balance that a real 9a has to hold.
 * shape 2 (one op): the whole budget goes to the single largest op in the
 *   chain — die B runs ONE fat kernel per layer. This is the CEILING of the
 *   concentration lever and it is only legal at G = 1, because every other
 *   group would have die A working alone. Reported as a bound. */
static int ops_set_split_shape(double f, int shape)
{
    int i, g;

    for (i = 0; i < PS_MAX_OPS; i++)
        ps_ops[i].rows_b = 0;
    if (f <= 0.0)
        return ops_rebuild_views();

    if (shape == 2) {
        double total = 0.0, budget;
        for (i = 0; i < PS_MAX_OPS; i++)
            total += (double)ps_ops[i].rows * ps_ops[i].stride;
        budget = f * total;
        for (;;) {
            int best = -1;
            for (i = 0; i < PS_MAX_OPS; i++)
                if (ps_ops[i].rows_b == 0 &&
                    (best < 0 || (double)ps_ops[i].rows * ps_ops[i].stride >
                                 (double)ps_ops[best].rows * ps_ops[best].stride))
                    best = i;
            if (best < 0 || budget <= 0.0)
                break;
            {
                ps_op *o = &ps_ops[best];
                int rb = (int)(budget / o->stride + 0.5);
                if (rb > o->rows) rb = o->rows;
                if (rb < 1) rb = 1;
                o->rows_b = rb;
                budget -= (double)rb * o->stride;
            }
        }
        return ops_rebuild_views();
    }

    if (shape == 0) {
        for (i = 0; i < PS_MAX_OPS; i++) {
            int rb = (int)(ps_ops[i].rows * f + 0.5);
            ps_ops[i].rows_b = rb > ps_ops[i].rows ? ps_ops[i].rows : rb;
        }
        return ops_rebuild_views();
    }

    for (g = 0; g < PS_MAX_GROUPS; g++) {
        double gbytes = 0.0, budget;
        for (i = 0; i < PS_MAX_OPS; i++)
            if (ps_ops[i].group == g)
                gbytes += (double)ps_ops[i].rows * ps_ops[i].stride;
        if (gbytes <= 0.0)
            continue;
        budget = f * gbytes;
        for (;;) {                       /* largest remaining op first */
            int best = -1;
            for (i = 0; i < PS_MAX_OPS; i++)
                if (ps_ops[i].group == g && ps_ops[i].rows_b == 0 &&
                    (best < 0 || ps_ops[i].rows > ps_ops[best].rows))
                    best = i;
            if (best < 0 || budget <= 0.0)
                break;
            {
                ps_op *o = &ps_ops[best];
                int rb = (int)(budget / o->stride + 0.5);
                if (rb > o->rows) rb = o->rows;
                if (rb < 1) rb = 1;
                o->rows_b = rb;
                budget -= (double)rb * o->stride;
            }
        }
    }
    return ops_rebuild_views();
}

static int ops_set_split(double f) { return ops_set_split_shape(f, ps_fshape); }

/* Bytes die B streams across the link per layer for its current split (weights
 * only; the x cost is what PS_XMODE is measuring and is reported separately). */
static double ops_split_bytes_b(void)
{
    double b = 0.0;
    int i;
    for (i = 0; i < PS_MAX_OPS; i++)
        b += (double)ps_ops[i].rows_b * ps_ops[i].stride;
    return b;
}

static double ops_split_bytes_a(void)
{
    double b = 0.0;
    int i;
    for (i = 0; i < PS_MAX_OPS; i++)
        b += (double)(ps_ops[i].rows - ps_ops[i].rows_b) * ps_ops[i].stride;
    return b;
}

/* Issue one op's die-A part (rows [0, rows-rows_b)) onto ctx a. */
static void issue_a(ps_rig *r, ps_op *o)
{
    int rows_a = o->rows - o->rows_b;
    if (rows_a <= 0)
        return;
    fc_matvec_q3k_ctx(r->a, o->w, o->x_a, o->y, rows_a, o->cols, o->stride);
}

static int op_group(int op, int groups);

/* PS_X_STAGE: bring group g's whole activation range across the link in ONE
 * async peer copy on die B's stream, before die B's kernels for that group.
 * Stream-ordered, so no host sync and no extra rendezvous. */
static void stage_group_x(ps_rig *r, int g, int groups)
{
    size_t lo = r->xpack_bytes, hi = 0;
    int i;
    if (ps_xmode != PS_X_STAGE)
        return;
    for (i = 0; i < PS_MAX_OPS; i++) {
        if (op_group(i, groups) != g || ps_ops[i].rows_b <= 0)
            continue;
        if (ps_ops[i].x_off < lo) lo = ps_ops[i].x_off;
        if (ps_ops[i].x_off + (size_t)ps_ops[i].cols * sizeof(float) > hi)
            hi = ps_ops[i].x_off + (size_t)ps_ops[i].cols * sizeof(float);
    }
    if (hi <= lo)
        return;
    {
        fc_buffer *dv = fc_buffer_offset_view(r->xpack_b, lo);
        fc_buffer *sv = fc_buffer_offset_view(r->xpack_a, lo);
        if (dv && sv)
            fc_buffer_copy_peer_async(dv, sv, hi - lo, r->b);
        fc_buffer_free_view(dv);
        fc_buffer_free_view(sv);
    }
}

/* Issue one op's die-B part (rows [rows-rows_b, rows)) onto ctx b, reading die
 * A's weights across the link and writing straight into die A's output buffer.
 * PS_XMODE picks where x comes from; PS_BKERN picks the matvec variant. */
static void issue_b(ps_rig *r, ps_op *o, int unused_x_remote)
{
    fc_buffer *x = (ps_xmode == PS_X_REMOTE) ? o->x_a : o->x_b;
    (void)unused_x_remote;
    if (o->rows_b <= 0)
        return;
    if (ps_bkern >= 0 &&
        fc_matvec_q3k_ps_ctx(r->b, o->w_tail, x, o->y_tail, o->rows_b, o->cols,
                             o->stride, ps_bkern) == 0)
        return;
    fc_matvec_q3k_ctx(r->b, o->w_tail, x, o->y_tail, o->rows_b, o->cols,
                      o->stride);
}

/* ==================================================================
 * P1 — what does die B pay for reading x across the link too?
 * ================================================================== */

static void p1_remote_x(ps_rig *r, int iters, int reps)
{
    double loc[32], rem[32], own[32];
    ps_op *o = &ps_ops[0];              /* qg 8192x2048, the big dense one */
    int rep, i, xmode0 = ps_xmode;
    if (reps > 32) reps = 32;

    printf("\n== P1 — die B remote matvec: does reading x remotely as well "
           "cost anything? ==\n");
    printf("  (9a cannot afford a 10 us cuMemcpyPeer of x per ordering group,\n"
           "   so it must read x across the link. This prices that choice.)\n");

    o->rows_b = o->rows;                /* whole tensor, so the rates compare */
    fc_buffer_free_view(o->w_tail);
    fc_buffer_free_view(o->y_tail);
    o->w_tail = fc_buffer_offset_view(o->w, 0);
    o->y_tail = fc_buffer_offset_view(o->y, 0);

    for (rep = 0; rep < reps; rep++) {
        double t0;
        fc_matvec_q3k_ctx(r->a, o->w, o->x_a, o->y, o->rows, o->cols, o->stride);
        fc_sync_ctx(r->a);
        t0 = now_s();
        for (i = 0; i < iters; i++)
            fc_matvec_q3k_ctx(r->a, o->w, o->x_a, o->y, o->rows, o->cols,
                              o->stride);
        fc_sync_ctx(r->a);
        own[rep] = (now_s() - t0) / iters;

        ps_xmode = PS_X_MIRROR;         /* w remote, x local on B */
        issue_b(r, o, 0);
        fc_sync_ctx(r->b);
        t0 = now_s();
        for (i = 0; i < iters; i++)
            issue_b(r, o, 0);
        fc_sync_ctx(r->b);
        loc[rep] = (now_s() - t0) / iters;

        ps_xmode = PS_X_REMOTE;         /* w remote, x remote */
        issue_b(r, o, 0);
        fc_sync_ctx(r->b);
        t0 = now_s();
        for (i = 0; i < iters; i++)
            issue_b(r, o, 0);
        fc_sync_ctx(r->b);
        rem[rep] = (now_s() - t0) / iters;
    }
    ps_xmode = xmode0;
    qsort(own, (size_t)reps, sizeof(double), cmp_double);
    qsort(loc, (size_t)reps, sizeof(double), cmp_double);
    qsort(rem, (size_t)reps, sizeof(double), cmp_double);
    printf("  %-34s %10s %12s\n", "arm", "ms", "vs A local");
    printf("  %-34s %10.4f %11.2fx\n", "die A, own weights + own x",
           own[reps / 2] * 1e3, 1.0);
    printf("  %-34s %10.4f %11.2fx\n", "die B: w REMOTE, x local",
           loc[reps / 2] * 1e3, loc[reps / 2] / own[reps / 2]);
    printf("  %-34s %10.4f %11.2fx  <-- what 9a must use\n",
           "die B: w REMOTE, x REMOTE", rem[reps / 2] * 1e3,
           rem[reps / 2] / own[reps / 2]);
    printf("  remote-x surcharge: %+.1f%%\n",
           100.0 * (rem[reps / 2] / loc[reps / 2] - 1.0));
    printf("  => optimal die-B row share f* = %.3f, gross matvec speedup "
           "%.3fx\n",
           1.0 / (1.0 + rem[reps / 2] / own[reps / 2]),
           1.0 + own[reps / 2] / rem[reps / 2]);
    o->rows_b = 0;
}

/* ==================================================================
 * P2 — the two orderings 9a can use, priced in a REAL producer/consumer.
 *
 * The campaign's barrier table timed a POINT barrier with both dies idle and
 * concluded host-sync (9.99 us) beats cuStreamWaitEvent (28.5 us). That is the
 * right answer for T3, which barriers once per token. It is NOT obviously the
 * right answer for 9a, which barriers ~160 times per token INSIDE the layer
 * loop: a host drain also stops the host from running ahead and launching, and
 * a stream-wait does not. So both are timed here with real matvec work in
 * flight, which is the only configuration 9a will ever see.
 * ================================================================== */

static double p2_order_sync(ps_rig *r, int iters)
{
    double t0 = now_s();
    int i;
    for (i = 0; i < iters; i++) {
        fc_sync_ctx(r->a);
        fc_sync_ctx(r->b);
    }
    return (now_s() - t0) / iters * 1e6;
}

static double p2_order_event(ps_rig *r, int iters)
{
    double t0;
    int i;
    if (!r->have_stream_wait)
        return -1.0;
    t0 = now_s();
    for (i = 0; i < iters; i++) {
        fc_event *ea = r->ev_a[i % PS_EVENT_RING];
        fc_event *eb = r->ev_b[i % PS_EVENT_RING];
        fc_event_record_ctx(r->a, ea);
        fc_event_record_ctx(r->b, eb);
        fc_event_wait_ctx(r->a, eb);
        fc_event_wait_ctx(r->b, ea);
    }
    fc_sync_ctx(r->a);
    fc_sync_ctx(r->b);
    return (now_s() - t0) / iters * 1e6;
}

static void p2_orderings(ps_rig *r, int iters, int reps)
{
    double s[32], e[32];
    int rep;
    if (reps > 32) reps = 32;
    printf("\n== P2 — cost of ONE cross-die rendezvous (both dies idle) ==\n");
    for (rep = 0; rep < reps; rep++) {
        s[rep] = p2_order_sync(r, iters);
        e[rep] = p2_order_event(r, iters);
    }
    qsort(s, (size_t)reps, sizeof(double), cmp_double);
    qsort(e, (size_t)reps, sizeof(double), cmp_double);
    printf("  %-40s %10s %14s\n", "primitive", "us", "x160/token ms");
    printf("  %-40s %10.2f %14.2f\n", "sync(A)+sync(B)  [host drain]",
           s[reps / 2], s[reps / 2] * 160.0 / 1000.0);
    if (e[reps / 2] < 0)
        printf("  %-40s %10s %14s\n", "record x2 + streamWait x2", "n/a", "-");
    else
        printf("  %-40s %10.2f %14.2f\n", "record x2 + streamWait x2  [device]",
               e[reps / 2], e[reps / 2] * 160.0 / 1000.0);
    printf("  NOTE: these are the OVERHEAD components only. Under load a drain\n"
           "  also blocks the host from launching ahead; P4 is where that bites.\n");
}

/* ==================================================================
 * P3 — can a cross-die fork/join live inside ONE captured graph?
 *
 * If it can, 9a pays neither ordering nor fragmentation and the whole gross
 * saving is net. Never probed: §9 item 9 of the campaign notes only the peer
 * COPY was tried (it cannot, error 900). This uses no copy at all -- die B's
 * kernel writes into die A's buffer directly -- so the copy's restriction may
 * not apply.
 * ================================================================== */

static void p3_probe_cross_die_graph(ps_rig *r)
{
    fc_graph *g;
    ps_op *o = &ps_ops[1];              /* k 512x2048, small and quick */
    printf("\n== P3 — can a cross-die fork/join be CAPTURED in one graph? ==\n");
    if (!r->have_stream_wait) {
        printf("  cuStreamWaitEvent unavailable — cannot even try.\n");
        return;
    }
    if (ops_set_split(0.10) != 0) {
        printf("  split view setup failed\n");
        return;
    }
    if (fc_graph_begin_ctx(r->a) != 0) {
        printf("  graphs unavailable on this driver\n");
        return;
    }
    issue_a(r, o);
    fc_event_record_ctx(r->a, r->ev_a[0]);
    fc_event_wait_ctx(r->b, r->ev_a[0]);     /* fork: die B joins the capture */
    issue_b(r, o, 1);
    fc_event_record_ctx(r->b, r->ev_b[0]);
    fc_event_wait_ctx(r->a, r->ev_b[0]);     /* join */
    g = fc_graph_end_ctx(r->a);
    if (!g) {
        printf("  NO — capture failed once die B's stream joined it. 9a must\n"
               "  pay a real ordering + a graph boundary at every group.\n");
        fc_sync_ctx(r->a);
        fc_sync_ctx(r->b);
        return;
    }
    printf("  capture SUCCEEDED — replaying to check it actually runs...\n");
    if (fc_graph_launch(g) != 0)
        printf("  ** replay FAILED — treat the capture as unusable **\n");
    else
        printf("  ** YES — a cross-die graph captured and replayed. This is the\n"
               "  best case for 9a: no per-group ordering, no fragmentation. **\n");
    fc_sync_ctx(r->a);
    fc_sync_ctx(r->b);
    fc_graph_free(g);
}

/* ==================================================================
 * P4 — the number. 40 layers of saber's dense matvec chain.
 * ================================================================== */

/* order_mode: how the two dies rendezvous G times per layer.
 *   PS_ORD_SYNC    host drains both streams               (fragments the graph)
 *   PS_ORD_EVENT   cuEventRecord x2 + cuStreamWaitEvent x2 (fragments the graph)
 *   PS_ORD_XGRAPH  the WHOLE layer -- both dies, all G rendezvous -- captured
 *                  as ONE cross-die graph and replayed with one launch. P3
 *                  showed this is possible; it is 9a's best case, because the
 *                  ordering becomes a graph edge and costs nothing at replay. */
enum { PS_ORD_SYNC = 0, PS_ORD_EVENT, PS_ORD_XGRAPH, PS_ORD_COUNT,
       /* DIAGNOSTIC ONLY, and deliberately outside the P4 sweep: both dies
        * run every group with NO rendezvous at all and are drained once per
        * token. It is not a legal scheme — die B would read a group's x
        * before die A wrote it — but it is the only way to separate the two
        * things that can eat a balanced split: CONTENTION (die B's peer
        * traffic slowing die A, and vice versa) from ORDERING (what the
        * rendezvous itself costs). Contention = t_none - max(t_A, t_B) solo;
        * ordering = t_split - t_none. */
       PS_ORD_NONE = PS_ORD_COUNT };
static const char *ps_ord_name[PS_ORD_COUNT + 1] = { "sync", "event", "xgraph",
                                                     "none" };

typedef struct p4_cfg {
    double f;                   /* die B's row share */
    int groups;                 /* ordering groups per layer */
    int order_mode;
    int graphs;                 /* capture fragments as graphs */
} p4_cfg;

/* Map the 8 ops onto `groups` ordering groups, preserving their dependency
 * order. groups=8 -> one per op; groups=4 -> saber's real chain. */
static int op_group(int op, int groups)
{
    if (groups >= PS_MAX_OPS)
        return op;                       /* one rendezvous per matvec */
    return ps_ops[op].group * groups / 4;   /* 4 -> as declared, 2 -> pairs,
                                             * 1 -> the whole layer */
}

/* Baseline: everything on die A. `graphs` captures one graph per layer. */
static double p4_run_base(ps_rig *r, int layers, int tokens, int graphs,
                          fc_graph **cache)
{
    double t0;
    int l, i, t;
    if (graphs && !*cache) {
        if (fc_graph_begin_ctx(r->a) == 0) {
            for (i = 0; i < PS_MAX_OPS; i++)
                issue_a(r, &ps_ops[i]);
            *cache = fc_graph_end_ctx(r->a);
        }
        if (!*cache) {                  /* capture failed: re-issue directly */
            for (i = 0; i < PS_MAX_OPS; i++)
                issue_a(r, &ps_ops[i]);
            fc_sync_ctx(r->a);
        }
    }
    t0 = now_s();
    for (t = 0; t < tokens; t++)
        for (l = 0; l < layers; l++) {
            if (graphs && *cache)
                fc_graph_launch(*cache);
            else
                for (i = 0; i < PS_MAX_OPS; i++)
                    issue_a(r, &ps_ops[i]);
        }
    fc_sync_ctx(r->a);
    return (now_s() - t0) / tokens;
}

/* 9a: die A and die B each take a slice of every op; `groups` rendezvous per
 * layer. When `graphs`, each group's die-A ops and die-B ops are captured as
 * their own fragment graph -- that is the most favourable implementation the
 * scheme allows, and it is what a real build would do. */
static double p4_run_split(ps_rig *r, int layers, int tokens, const p4_cfg *c,
                           fc_graph **fa, fc_graph **fb, int poison)
{
    double t0;
    int l, i, g, t;
    long evseq = 0;
    int G = c->groups;

    /* PS_ORD_XGRAPH: one graph for the whole layer, both dies inside it. The
     * initial fork brings die B's stream into die A's capture; each group ends
     * with a two-way rendezvous (A waits B, B waits A) so no die may run ahead
     * into the next group's inputs; a final join returns the capture to A. */
    if (c->order_mode == PS_ORD_XGRAPH) {
        /* A peer copy cannot be captured in a graph (error 900), so the
         * staged-x mode has no xgraph form. Report n/a rather than quietly
         * run die B on a stale mirror — in this bench x never changes, so a
         * stale mirror would still come out bit-exact. */
        if (ps_xmode == PS_X_STAGE)
            return -1.0;
        if (!fa[0]) {
            if (fc_graph_begin_ctx(r->a) == 0) {
                fc_event_record_ctx(r->a, r->ev_a[PS_MAX_GROUPS]);
                if (!poison)
                    fc_event_wait_ctx(r->b, r->ev_a[PS_MAX_GROUPS]);
                for (g = 0; g < G; g++) {
                    for (i = 0; i < PS_MAX_OPS; i++)
                        if (op_group(i, G) == g) issue_a(r, &ps_ops[i]);
                    if (!poison)
                        for (i = 0; i < PS_MAX_OPS; i++)
                            if (op_group(i, G) == g) issue_b(r, &ps_ops[i], 1);
                    if (!poison) {
                        fc_event_record_ctx(r->a, r->ev_a[g]);
                        fc_event_record_ctx(r->b, r->ev_b[g]);
                        fc_event_wait_ctx(r->b, r->ev_a[g]);
                        fc_event_wait_ctx(r->a, r->ev_b[g]);
                    }
                }
                fa[0] = fc_graph_end_ctx(r->a);
            }
            if (!fa[0]) {
                fc_sync_ctx(r->a);
                fc_sync_ctx(r->b);
                return -1.0;          /* capture failed: report n/a, not a lie */
            }
        }
        t0 = now_s();
        for (t = 0; t < tokens; t++)
            for (l = 0; l < layers; l++)
                fc_graph_launch(fa[0]);
        fc_sync_ctx(r->a);
        fc_sync_ctx(r->b);
        return (now_s() - t0) / tokens;
    }

    if (c->graphs && !fa[0]) {
        for (g = 0; g < G; g++) {
            if (fc_graph_begin_ctx(r->a) == 0) {
                for (i = 0; i < PS_MAX_OPS; i++)
                    if (op_group(i, G) == g) issue_a(r, &ps_ops[i]);
                fa[g] = fc_graph_end_ctx(r->a);
            }
            if (!fa[g]) {
                for (i = 0; i < PS_MAX_OPS; i++)
                    if (op_group(i, G) == g) issue_a(r, &ps_ops[i]);
                fc_sync_ctx(r->a);
            }
            if (!poison && fc_graph_begin_ctx(r->b) == 0) {
                for (i = 0; i < PS_MAX_OPS; i++)
                    if (op_group(i, G) == g) issue_b(r, &ps_ops[i], 1);
                fb[g] = fc_graph_end_ctx(r->b);
            }
            if (!poison && !fb[g]) {
                for (i = 0; i < PS_MAX_OPS; i++)
                    if (op_group(i, G) == g) issue_b(r, &ps_ops[i], 1);
                fc_sync_ctx(r->b);
            }
        }
    }

    t0 = now_s();
    for (t = 0; t < tokens; t++)
    for (l = 0; l < layers; l++) {
        for (g = 0; g < G; g++) {
            if (c->graphs && fa[g])
                fc_graph_launch(fa[g]);
            else
                for (i = 0; i < PS_MAX_OPS; i++)
                    if (op_group(i, G) == g) issue_a(r, &ps_ops[i]);
            if (!poison) {
                /* stream-ordered ahead of die B's kernels for this group; a
                 * peer copy cannot live inside the fragment graph, so it is
                 * issued on the stream in front of it */
                stage_group_x(r, g, G);
                if (c->graphs && fb[g])
                    fc_graph_launch(fb[g]);
                else
                    for (i = 0; i < PS_MAX_OPS; i++)
                        if (op_group(i, G) == g) issue_b(r, &ps_ops[i], 1);
            }
            if (c->order_mode == PS_ORD_EVENT) {
                int slot = (int)(evseq++ % PS_EVENT_RING);
                fc_event_record_ctx(r->a, r->ev_a[slot]);
                fc_event_record_ctx(r->b, r->ev_b[slot]);
                fc_event_wait_ctx(r->a, r->ev_b[slot]);
                fc_event_wait_ctx(r->b, r->ev_a[slot]);
            } else if (c->order_mode == PS_ORD_SYNC) {
                fc_sync_ctx(r->a);
                fc_sync_ctx(r->b);
            }   /* PS_ORD_NONE: no rendezvous — the attribution arm */
        }
    }
    fc_sync_ctx(r->a);
    fc_sync_ctx(r->b);
    return (now_s() - t0) / tokens;
}

/* Snapshot every op's output off die A. */
static float **p4_snapshot(void)
{
    float **s = calloc(PS_MAX_OPS, sizeof(float *));
    int i;
    if (!s) return NULL;
    for (i = 0; i < PS_MAX_OPS; i++) {
        s[i] = malloc((size_t)ps_ops[i].rows * sizeof(float));
        if (!s[i]) return s;
        fc_buffer_to_cpu(ps_ops[i].y, s[i]);
    }
    return s;
}

static int p4_compare(float **ref)
{
    int i, j, bad = 0;
    float *got = NULL;
    for (i = 0; i < PS_MAX_OPS; i++) {
        got = realloc(got, (size_t)ps_ops[i].rows * sizeof(float));
        if (!got || !ref[i]) continue;
        fc_buffer_to_cpu(ps_ops[i].y, got);
        for (j = 0; j < ps_ops[i].rows; j++)
            if (ref[i][j] != got[j])
                bad++;
    }
    free(got);
    return bad;
}

/* Wreck die B's activation mirror. The SECOND vacuity trap in this bench: x
 * never changes, so a staged-x arm that never issued its peer copy would read
 * a mirror that happens to be right and come out bit-exact. Garbling the
 * mirror before every verify makes the staging copy load-bearing.
 *   PS_X_REMOTE  must not touch the mirror at all   -> must stay bit-exact
 *   PS_X_STAGE   must repair it with its peer copy  -> must stay bit-exact
 *   PS_X_MIRROR  reads it                           -> MUST go red; it is the
 *                free-x BOUND, not an implementable configuration. */
static void x_mirror_garble(ps_rig *r)
{
    size_t n = r->xpack_bytes / sizeof(float);
    float *junk = malloc(r->xpack_bytes);
    size_t i;
    if (!junk)
        return;
    for (i = 0; i < n; i++) junk[i] = -7.5e3f;
    fc_buffer_from_cpu_n(r->xpack_b, junk, r->xpack_bytes);
    fc_sync_ctx(r->b);
    free(junk);
}

static void x_mirror_restore(ps_rig *r)
{
    fc_buffer_copy_peer(r->xpack_b, r->xpack_a, r->xpack_bytes);
    fc_sync_ctx(r->b);
}

/* Overwrite every output with a sentinel, run ONE layer of the split, and
 * compare. Without the sentinel the gate is VACUOUS: x never changes, so the
 * rows die B was supposed to write still hold the correct values from the
 * previous arm and a die B that did nothing at all passes. PS_POISON=1 proved
 * exactly that on 2026-08-09 before this was added. */
static int p4_verify(ps_rig *r, const p4_cfg *c, fc_graph **fa, fc_graph **fb,
                     float **ref, int poison)
{
    static const unsigned int sentinel = 0x7FBFDEADu;   /* a quiet NaN pattern */
    unsigned int *fill = NULL;
    int i, maxrows = 0, bad;
    if (!ref)
        return -1;
    for (i = 0; i < PS_MAX_OPS; i++)
        if (ps_ops[i].rows > maxrows) maxrows = ps_ops[i].rows;
    fill = malloc((size_t)maxrows * sizeof(unsigned int));
    if (!fill)
        return -1;
    for (i = 0; i < maxrows; i++) fill[i] = sentinel;
    for (i = 0; i < PS_MAX_OPS; i++)
        fc_buffer_from_cpu_n(ps_ops[i].y, fill,
                             (size_t)ps_ops[i].rows * sizeof(float));
    fc_sync_ctx(r->a);
    free(fill);
    if (ps_xmode == PS_X_MIRROR && !ps_xgarble)
        x_mirror_restore(r);            /* the bound arm is allowed its mirror */
    else
        x_mirror_garble(r);
    if (p4_run_split(r, 1, 1, c, fa, fb, poison) < 0.0) {
        x_mirror_restore(r);
        return -1;
    }
    bad = p4_compare(ref);
    x_mirror_restore(r);
    return bad;
}

static void p4_snapshot_free(float **s)
{
    int i;
    if (!s) return;
    for (i = 0; i < PS_MAX_OPS; i++) free(s[i]);
    free(s);
}

/* ==================================================================
 * P5 — R_B(f) and R_A(f): the two rates the load balance is made of.
 *
 * Each die's slice of the 40-layer chain, run ALONE, with the same one-graph-
 * per-group fragmentation the split arm uses. Solo is the right way to
 * measure a rate: under load the two are entangled and you cannot attribute a
 * millisecond to either. P7 then checks the solo rates against the loaded
 * split time and reports the gap rather than hiding it.
 * ================================================================== */

enum { PS_SIDE_A = 0, PS_SIDE_B = 1 };

static double run_solo(ps_rig *r, int layers, int tokens, int G, int side)
{
    fc_graph *frag[PS_MAX_GROUPS] = { 0 };
    fc_cuda_ctx *ctx = (side == PS_SIDE_B) ? r->b : r->a;
    double t0;
    int g, i, l, t;

    for (g = 0; g < G; g++) {
        if (fc_graph_begin_ctx(ctx) == 0) {
            for (i = 0; i < PS_MAX_OPS; i++)
                if (op_group(i, G) == g) {
                    if (side == PS_SIDE_B) issue_b(r, &ps_ops[i], 1);
                    else                   issue_a(r, &ps_ops[i]);
                }
            frag[g] = fc_graph_end_ctx(ctx);
        }
    }
    fc_sync_ctx(ctx);

    t0 = now_s();
    for (t = 0; t < tokens; t++)
        for (l = 0; l < layers; l++)
            for (g = 0; g < G; g++) {
                if (side == PS_SIDE_B)
                    stage_group_x(r, g, G);
                if (frag[g])
                    fc_graph_launch(frag[g]);
                else
                    for (i = 0; i < PS_MAX_OPS; i++)
                        if (op_group(i, G) == g) {
                            if (side == PS_SIDE_B) issue_b(r, &ps_ops[i], 1);
                            else                   issue_a(r, &ps_ops[i]);
                        }
            }
    fc_sync_ctx(ctx);
    t0 = (now_s() - t0) / tokens;
    for (g = 0; g < G; g++)
        fc_graph_free(frag[g]);
    return t0;
}

/* GB/s a side achieves on the weight bytes it streamed. */
static double solo_rate(ps_rig *r, int layers, int tokens, int G, int side,
                        int reps, double *ms_out)
{
    double m[32];
    double bytes = (side == PS_SIDE_B) ? ops_split_bytes_b()
                                       : ops_split_bytes_a();
    int rep;
    if (reps > 32) reps = 32;
    if (bytes <= 0.0) { if (ms_out) *ms_out = 0.0; return 0.0; }
    for (rep = 0; rep < reps; rep++)
        m[rep] = run_solo(r, layers, tokens, G, side);
    {
        double s = dmedian(m, reps);
        if (ms_out) *ms_out = s * 1e3;
        return bytes * (double)layers / s / 1e9;
    }
}

/* The measured R_B(f) curve, kept so the fixed point can be solved on data
 * rather than on a guess. */
#define PS_CURVE_MAX 16
static double curve_f[PS_CURVE_MAX], curve_rb[PS_CURVE_MAX];
static int curve_n = 0;

static double curve_rb_at(double f)
{
    int i;
    if (curve_n == 0) return 0.0;
    if (curve_n == 1 || f <= curve_f[0]) return curve_rb[0];
    if (f >= curve_f[curve_n - 1]) return curve_rb[curve_n - 1];
    for (i = 1; i < curve_n; i++)
        if (f <= curve_f[i]) {
            double u = (f - curve_f[i - 1]) / (curve_f[i] - curve_f[i - 1]);
            return curve_rb[i - 1] + u * (curve_rb[i] - curve_rb[i - 1]);
        }
    return curve_rb[curve_n - 1];
}

/* f* solves f = R_B(f) / (R_A + R_B(f)). R_B rises with f, so the map is
 * monotone and a plain iteration converges; it is run from both ends and the
 * two are reported, because a fixed point found from one side only is not
 * evidence that it is the only one. */
static double solve_fixed_point(double ra, double f0, double *rb_out)
{
    double f = f0;
    int it;
    for (it = 0; it < 200; it++) {
        double rb = curve_rb_at(f);
        double fn = rb / (ra + rb);
        if (fabs(fn - f) < 1e-6) { f = fn; break; }
        f = 0.5 * f + 0.5 * fn;         /* damped, so it cannot oscillate */
    }
    if (rb_out) *rb_out = curve_rb_at(f);
    return f;
}

static void p5_rate_curve(ps_rig *r, int layers, int tokens, int G, int reps,
                          const double *Fs, int nF)
{
    int fi;
    printf("\n== P5 — R_B(f) and R_A(f): the load-balance inputs "
           "(G=%d, %d layers) ==\n", G, layers);
    printf("  die B's rate is a FUNCTION of its share — a thin slice cannot\n"
           "  keep the link busy — so f* = R_B/(R_A+R_B) is a fixed point.\n");
    printf("  xmode=%s  bkern=%s  fshape=%d\n",
           ps_xmode_name[ps_xmode], ps_bkern_name(ps_bkern), ps_fshape);
    printf("\n  %-6s %10s %10s %10s %10s %10s %9s\n", "f", "MB_B/tok",
           "t_B ms", "R_B GB/s", "t_A ms", "R_A GB/s", "f_next");
    curve_n = 0;
    for (fi = 0; fi < nF; fi++) {
        double rb, ra, msb, msa, bytes_b;
        if (Fs[fi] <= 0.0) continue;
        if (ops_set_split(Fs[fi]) != 0) break;
        bytes_b = ops_split_bytes_b() * layers;
        rb = solo_rate(r, layers, tokens, G, PS_SIDE_B, reps, &msb);
        ra = solo_rate(r, layers, tokens, G, PS_SIDE_A, reps, &msa);
        printf("  %-6.3f %10.1f %10.3f %10.2f %10.3f %10.2f %9.4f\n",
               Fs[fi], bytes_b / 1e6, msb, rb, msa, ra, rb / (ra + rb));
        fflush(stdout);
        if (curve_n < PS_CURVE_MAX) {
            curve_f[curve_n] = Fs[fi];
            curve_rb[curve_n] = rb;
            curve_n++;
        }
    }
}

/* ==================================================================
 * P6 — the R_B ledger. One row per optimisation, measured in isolation.
 *
 * The number to watch is R_B against what the link is known to deliver:
 * 12.41 GB/s copy-engine, 9.3-11.3 GB/s kernel-issued. Part 1 achieved
 * 2.66-4.22.
 * ================================================================== */
#define PS_LINK_GBS 12.41

static void p6_rb_ledger(ps_rig *r, int layers, int tokens, int G, int reps,
                         double f)
{
    struct { int xmode, bkern, shape; const char *what; } arms[] = {
        { PS_X_REMOTE, -1,              0, "part 1 baseline" },
        { PS_X_STAGE,  -1,              0, "+ x staged per group" },
        { PS_X_MIRROR, -1,              0, "  (x free: the bound)" },
        { PS_X_STAGE,  FC_Q3K_PS_R2P1,  0, "+ 128-bit staged-x loads" },
        { PS_X_STAGE,  FC_Q3K_PS_R2P2,  0, "+ prefetch 2" },
        { PS_X_STAGE,  FC_Q3K_PS_R2P4,  0, "+ prefetch 4" },
        { PS_X_STAGE,  FC_Q3K_PS_R2P8,  0, "+ prefetch 8" },
        { PS_X_STAGE,  FC_Q3K_PS_R1P1,  0, "+ 1 row/warp" },
        { PS_X_STAGE,  FC_Q3K_PS_R1P4,  0, "+ 1 row/warp, prefetch 4" },
        { PS_X_STAGE,  FC_Q3K_PS_R1P8,  0, "+ 1 row/warp, prefetch 8" },
        { PS_X_STAGE,  FC_Q3K_PS_R2P4,  1, "+ concentrated slice, pf4" },
        { PS_X_STAGE,  FC_Q3K_PS_R1P4,  1, "+ concentrated slice, r1pf4" },
        { PS_X_STAGE,  FC_Q3K_PS_R2P8,  1, "+ concentrated slice, pf8" },
        { PS_X_STAGE,  FC_Q3K_PS_R2P4,  2, "+ ONE fat kernel/layer, pf4" },
        { PS_X_STAGE,  FC_Q3K_PS_R1P4,  2, "+ ONE fat kernel/layer, r1pf4" },
        { PS_X_STAGE,  -1,              2, "+ ONE fat kernel/layer, stock" },
        { PS_X_REMOTE, FC_Q3K_PS_R2P4,  0, "  (remote x, pf4)" },
    };
    int xmode0 = ps_xmode, bkern0 = ps_bkern, shape0 = ps_fshape;
    size_t k;
    double first = 0.0;

    printf("\n== P6 — R_B ledger at f = %.3f (G=%d, %d layers) ==\n",
           f, G, layers);
    printf("  link delivers 12.41 GB/s (copy engine), 9.3-11.3 (kernel-issued)."
           "\n  %-30s %10s %8s %8s %9s\n", "arm", "R_B GB/s", "vs base",
           "%link", "kernels/l");
    for (k = 0; k < sizeof(arms) / sizeof(arms[0]); k++) {
        double rb, ms;
        int nk = 0, i;
        ps_xmode = arms[k].xmode;
        ps_bkern = arms[k].bkern;
        ps_fshape = arms[k].shape;
        if (ps_bkern >= 0 && !fc_matvec_q3k_ps_name(ps_bkern))
            continue;
        if (ops_set_split(f) != 0) break;
        for (i = 0; i < PS_MAX_OPS; i++) if (ps_ops[i].rows_b > 0) nk++;
        rb = solo_rate(r, layers, tokens, G, PS_SIDE_B, reps, &ms);
        if (first == 0.0) first = rb;
        printf("  %-30s %10.2f %7.2fx %7.0f%% %9d\n", arms[k].what, rb,
               first > 0.0 ? rb / first : 0.0, 100.0 * rb / PS_LINK_GBS, nk);
        fflush(stdout);
    }
    ps_xmode = xmode0; ps_bkern = bkern0; ps_fshape = shape0;
}

/* ==================================================================
 * P7 — the balanced measurement. INTERLEAVED base/split pairs, so drift
 * cancels, and a 95% CI instead of a point estimate: part 1's +-1.3% noise
 * floor cannot resolve the ~+3% this scheme is worth at its own optimum.
 * ================================================================== */

static int ps_balord = PS_ORD_EVENT;  /* PS_BALORD: which rendezvous P7 uses */

static void p7_balanced(ps_rig *r, int layers, int tokens, int G, int pairs,
                        double f, fc_graph **base_graph, float **ref)
{
    p4_cfg c;
    fc_graph *fa[PS_MAX_GROUPS] = { 0 };
    fc_graph *fb[PS_MAX_GROUPS] = { 0 };
    double *ratio = malloc((size_t)pairs * sizeof(double));
    double mean, half, msb, msa, rb, ra;
    int i, bad;

    if (!ratio)
        return;
    printf("\n== P7 — the BALANCED arm: f* = %.4f, %d interleaved pairs ==\n",
           f, pairs);
    printf("  xmode=%s  bkern=%s  fshape=%d  G=%d  order=%s\n",
           ps_xmode_name[ps_xmode], ps_bkern_name(ps_bkern), ps_fshape, G,
           ps_ord_name[ps_balord]);

    if (ops_set_split(f) != 0) { free(ratio); return; }
    rb = solo_rate(r, layers, tokens, G, PS_SIDE_B, 5, &msb);
    ra = solo_rate(r, layers, tokens, G, PS_SIDE_A, 5, &msa);
    printf("  solo: t_A %.3f ms (R_A %.2f GB/s), t_B %.3f ms (R_B %.2f GB/s), "
           "max %.3f\n", msa, ra, msb, rb, msa > msb ? msa : msb);

    /* Attribution: what does the split cost BEFORE any rendezvous is paid? */
    {
        p4_cfg cn;
        fc_graph *na[PS_MAX_GROUPS] = { 0 }, *nb[PS_MAX_GROUPS] = { 0 };
        double m[9];
        int k;
        cn.f = f; cn.groups = G; cn.order_mode = PS_ORD_NONE; cn.graphs = 1;
        p4_run_split(r, layers, tokens, &cn, na, nb, 0);
        for (k = 0; k < 9; k++)
            m[k] = p4_run_split(r, layers, tokens, &cn, na, nb, 0) * 1e3;
        printf("  concurrent, NO rendezvous: %.3f ms  => contention costs "
               "%+.3f ms vs max(solo)\n", dmedian(m, 9),
               dmedian(m, 9) - (msa > msb ? msa : msb));
        for (k = 0; k < PS_MAX_GROUPS; k++) {
            fc_graph_free(na[k]);
            fc_graph_free(nb[k]);
        }
    }

    c.f = f; c.groups = G; c.order_mode = ps_balord; c.graphs = 1;
    /* warm both arms' graphs before the timed pairs. base_graph was captured
     * by the caller at f = 0, so it really is the all-on-die-A baseline; a
     * graph captured here would bake in the CURRENT split and measure nothing. */
    p4_run_base(r, layers, tokens, 1, base_graph);
    p4_run_split(r, layers, tokens, &c, fa, fb, 0);

    for (i = 0; i < pairs; i++) {
        double tb = p4_run_base(r, layers, tokens, 1, base_graph);
        double ts = p4_run_split(r, layers, tokens, &c, fa, fb, 0);
        double tb2 = p4_run_base(r, layers, tokens, 1, base_graph);
        /* ABBA: the split arm sits between two baselines, so a monotone
         * drift (clock ramp, thermals) cancels to first order. */
        ratio[i] = (0.5 * (tb + tb2)) / ts;
    }
    mean_ci95(ratio, pairs, &mean, &half);
    bad = p4_verify(r, &c, fa, fb, ref, 0);
    printf("  BALANCED SPEEDUP  %.4fx  95%% CI [%.4f, %.4f]   n=%d pairs   "
           "bitexact %s\n", mean, mean - half, mean + half, pairs,
           bad == 0 ? "yes" : (bad < 0 ? "?" : "NO"));
    printf("  predicted from the solo rates: %.4fx  (1 + R_B/R_A = 1 + "
           "%.2f/%.2f)\n", 1.0 + rb / ra, rb, ra);
    for (i = 0; i < PS_MAX_GROUPS; i++) {
        fc_graph_free(fa[i]);
        fc_graph_free(fb[i]);
    }
    free(ratio);
}

/* ------------------------------------------------------------------ */

int main(void)
{
    ps_rig r;
    int reps  = env_int("PS_REPS", 5);
    int iters = env_int("PS_ITERS", 200);
    int layers = env_int("PS_LAYERS", 40);
    int tokens = env_int("PS_TOKENS", 20);
    int poison = env_int("PS_POISON", 0);
    /* PS_PROBE is a bitmask: 1=P1 2=P2 4=P3 8=P4 16=P5 32=P6 64=P7.
     * Default is the part-2 set (P5|P6|P7) — P1-P4 are part 1's and are
     * already recorded; ask for them explicitly with PS_PROBE=255. */
    int probe = env_int("PS_PROBE", 16 | 32 | 64);
    int pairs = env_int("PS_PAIRS", 25);
    double ledger_f = env_int("PS_LEDGER_F_MILLI", 50) / 1000.0;
    int balG = env_int("PS_BALG", 1);
    fc_graph *base_graph = NULL;
    float **ref = NULL;
    double base_ms = 0.0, base_raw_ms = 0.0;
    double bt[32];
    int rep, gi, fi, oi, fail = 0, i;
    int Gs[8]      = { 1, 2, 4, 8 };
    double Fs[16]  = { 0.00, 0.05, 0.10, 0.15, 0.20 };
    int nG = 4, nF = 5;
    const char *fl = getenv("PS_FLIST");
    const char *gl = getenv("PS_GLIST");

    ps_xmode  = env_int("PS_XMODE", PS_X_REMOTE);
    ps_bkern  = env_int("PS_BKERN", -1);
    ps_fshape = env_int("PS_FSHAPE", 0);
    ps_xgarble = env_int("PS_XGARBLE", 0);
    ps_balord = env_int("PS_BALORD", PS_ORD_EVENT);
    if (ps_balord < 0 || ps_balord >= PS_ORD_COUNT) ps_balord = PS_ORD_EVENT;
    if (ps_xmode < 0 || ps_xmode > PS_X_STAGE) {
        fprintf(stderr, "PS_XMODE must be 0 (remote) 1 (mirror) 2 (stage)\n");
        return 1;
    }
    if (reps > 32) reps = 32;
    if (pairs < 2) pairs = 2;
    if (fl && *fl) {                     /* PS_FLIST=0.01,0.02,... */
        char *cp = strdup(fl), *tok, *sv = NULL;
        nF = 0;
        for (tok = strtok_r(cp, ",", &sv); tok && nF < 16;
             tok = strtok_r(NULL, ",", &sv))
            Fs[nF++] = atof(tok);
        free(cp);
    }
    if (gl && *gl) {                     /* PS_GLIST=1,4 */
        char *cp = strdup(gl), *tok, *sv = NULL;
        nG = 0;
        for (tok = strtok_r(cp, ",", &sv); tok && nG < 8;
             tok = strtok_r(NULL, ",", &sv))
            Gs[nG++] = atoi(tok);
        free(cp);
    }
    if (fc_driver_load() != 0) {
        fprintf(stderr, "no CUDA driver\n");
        return 1;
    }
    srand(1234);
    if (rig_init(&r) != 0)
        return 1;
    printf("bench_peer_stream — scheme 9a (PLX weight-streaming) economics\n");
    printf("die A = device %d, die B = device %d, peer ENABLED\n",
           fc_cuda_ctx_device_id(r.a), fc_cuda_ctx_device_id(r.b));
    printf("events %s, cuStreamWaitEvent %s\n",
           r.have_events ? "available" : "UNAVAILABLE",
           r.have_stream_wait ? "available" : "UNAVAILABLE");
    /* Which peer-stream variants this build's PTX actually carries. A missing
     * variant must never be papered over by the stock kernel — an arm that
     * silently measured the stock kernel twice is exactly the failure mode
     * this bench exists to avoid. */
    printf("peer-stream matvec variants:");
    for (i = 0; i < FC_Q3K_PS_COUNT; i++)
        printf(" %s", fc_matvec_q3k_ps_name(i));
    printf("\nselected: xmode=%s bkern=%s fshape=%d probe=0x%x\n",
           ps_xmode_name[ps_xmode], ps_bkern_name(ps_bkern), ps_fshape, probe);
    if (poison)
        printf("** PS_POISON=1 — die B's launches are SKIPPED. Every split arm\n"
               "** must report a MISMATCH below. If any reports 0, the bit-exact\n"
               "** gate is vacuous and no number in this file may be believed.\n");
    if (ops_alloc(&r) != 0) {
        ops_free(&r);
        return 1;
    }
    printf("weights on die A: %.1f MB across %d ops\n",
           r.weight_bytes / 1e6, PS_MAX_OPS);
    /* A selected variant that is absent from the PTX is a hard error: die B
     * would otherwise fall back to the stock kernel and the arm would be
     * mislabelled. */
    if (ps_bkern >= 0) {
        if (ops_set_split(0.05) != 0 ||
            fc_matvec_q3k_ps_ctx(r.b, ps_ops[0].w_tail, ps_ops[0].x_b,
                                 ps_ops[0].y_tail, ps_ops[0].rows_b,
                                 ps_ops[0].cols, ps_ops[0].stride,
                                 ps_bkern) != 0) {
            fprintf(stderr, "PS_BKERN=%d (%s) is not in this build's PTX — "
                    "rebuild cuda_kernels\n", ps_bkern, ps_bkern_name(ps_bkern));
            ops_free(&r);
            return 1;
        }
        fc_sync_ctx(r.b);
    }

    if (probe & 1)  p1_remote_x(&r, iters / 10 + 1, reps);
    if (probe & 2)  p2_orderings(&r, iters * 5, reps);
    if (probe & 4)  p3_probe_cross_die_graph(&r);

    if (probe & 32)
        p6_rb_ledger(&r, layers, tokens, balG, reps, ledger_f);
    if (probe & 16)
        p5_rate_curve(&r, layers, tokens, balG, reps, Fs, nF);

    if (!(probe & (8 | 64))) {
        ops_free(&r);
        fc_free();
        return 0;
    }

    /* ---- P4 ---- */
    printf("\n== P4 — %d-layer dense matvec chain, %d tokens/measurement ==\n",
           layers, tokens);
    if (ops_set_split(0.0) != 0) { ops_free(&r); return 1; }

    for (rep = 0; rep < reps; rep++)
        bt[rep] = p4_run_base(&r, layers, tokens, 0, &base_graph);
    qsort(bt, (size_t)reps, sizeof(double), cmp_double);
    base_raw_ms = bt[reps / 2] * 1e3;
    for (rep = 0; rep < reps; rep++)
        bt[rep] = p4_run_base(&r, layers, tokens, 1, &base_graph);
    qsort(bt, (size_t)reps, sizeof(double), cmp_double);
    base_ms = bt[reps / 2] * 1e3;
    ref = p4_snapshot();
    printf("  baseline (all die A), no graph : %8.3f ms/token\n", base_raw_ms);
    printf("  baseline (all die A), 1 graph  : %8.3f ms/token   (graph win "
           "%.2fx)\n", base_ms, base_raw_ms / base_ms);
    printf("  NOTE this chain is %.1f%% of saber's 52.6 ms step by construction"
           " (the\n  matvec pool is 35.5 ms); the end-to-end effect of any row"
           " below is\n  its saving on THIS chain, not on the whole step.\n",
           100.0 * base_ms / 52.6);

    if (probe & 8)
    printf("\n  %-6s %-6s %-8s %-8s %12s %10s %10s %8s\n", "f", "G", "order",
           "graphs", "ms/token", "vs base", "step ms", "bitexact");
    for (fi = 0; (probe & 8) && fi < nF; fi++) {
        if (ops_set_split(Fs[fi]) != 0) break;
        for (gi = 0; gi < nG; gi++) {
            for (oi = 0; oi < PS_ORD_COUNT; oi++) {
                p4_cfg c = { Fs[fi], Gs[gi], oi, 1 };
                fc_graph *fa[PS_MAX_GROUPS] = { 0 };
                fc_graph *fb[PS_MAX_GROUPS] = { 0 };
                double m[32];
                double ms, step;
                int bad, k;
                if (oi != PS_ORD_SYNC && !r.have_stream_wait) continue;
                for (rep = 0; rep < reps; rep++)
                    m[rep] = p4_run_split(&r, layers, tokens, &c, fa, fb, poison);
                qsort(m, (size_t)reps, sizeof(double), cmp_double);
                ms = m[reps / 2] * 1e3;
                if (ms < 0.0) {
                    printf("  %-6.2f %-6d %-8s %-8s %12s\n", Fs[fi], Gs[gi],
                           ps_ord_name[oi], "yes", "capture failed");
                    continue;
                }
                bad = p4_verify(&r, &c, fa, fb, ref, poison);
                step = 52.6 - (base_ms - ms);
                printf("  %-6.2f %-6d %-8s %-8s %12.3f %9.3fx %10.2f %8s\n",
                       Fs[fi], Gs[gi], ps_ord_name[oi], "yes", ms,
                       base_ms / ms, step,
                       bad == 0 ? "yes" : (bad < 0 ? "?" : "NO"));
                if (bad > 0 && !poison) {
                    printf("      ** %d output values differ from the baseline "
                           "— the split is WRONG **\n", bad);
                    fail = 1;
                }
                if (bad == 0 && poison && Fs[fi] > 0.0) {
                    printf("      ** POISON ARM PASSED — the bit-exact gate is "
                           "VACUOUS **\n");
                    fail = 1;
                }
                for (k = 0; k < PS_MAX_GROUPS; k++) {
                    fc_graph_free(fa[k]);
                    fc_graph_free(fb[k]);
                }
                fflush(stdout);
            }
        }
    }

    if (probe & 8)
        printf("\n  'step ms' projects the row onto saber's 52.6 ms step by\n"
               "  substituting this chain's time. tok/s = 1000/step.\n");

    /* ---- P7: solve the fixed point on the measured curve, measure there ---- */
    if (probe & 64) {
        double ra, rb_star, fstar_lo, fstar_hi;
        double fforce = env_int("PS_FSTAR_MILLI", 0) / 1000.0;
        ops_set_split(0.05);
        ra = solo_rate(&r, layers, tokens, balG, PS_SIDE_A, reps, NULL);
        if (fforce > 0.0) {
            /* f* pinned by the caller, so a repeat measurement lands on the
             * SAME operating point as the run that solved for it */
            printf("\n== f* pinned to %.4f by PS_FSTAR_MILLI ==\n", fforce);
            p7_balanced(&r, layers, tokens, balG, pairs, fforce,
                        &base_graph, ref);
        } else if (curve_n == 0) {
            printf("\n== P7 skipped — no R_B curve (run P5 too: PS_PROBE|=16)\n");
        } else {
            fstar_lo = solve_fixed_point(ra, curve_f[0], &rb_star);
            fstar_hi = solve_fixed_point(ra, curve_f[curve_n - 1], &rb_star);
            printf("\n== fixed point of f = R_B(f)/(R_A+R_B(f)) ==\n");
            printf("  R_A = %.2f GB/s; iterating from below -> f* = %.4f, "
                   "from above -> f* = %.4f\n", ra, fstar_lo, fstar_hi);
            printf("  R_B(f*) = %.2f GB/s  => predicted 1 + R_B/R_A = %.4fx\n",
                   rb_star, 1.0 + rb_star / ra);
            if (fabs(fstar_lo - fstar_hi) > 0.002)
                printf("  ** the two ends disagree — the curve has more than one\n"
                       "  ** fixed point in the swept range; both are reported.\n");
            p7_balanced(&r, layers, tokens, balG, pairs, fstar_lo,
                        &base_graph, ref);
        }
    }

    p4_snapshot_free(ref);
    fc_graph_free(base_graph);
    ops_free(&r);
    fc_free();
    return fail;
}
