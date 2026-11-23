/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_split_scaling.c
 * @brief Prices tensor-parallelism on one K80 die: does halving a kernel's work halve
 *   its time? Times every hot saber decode kernel at FULL width and at the width a
 *   2-die tensor-parallel split would give it, and reports eta = t_full / t_half.
 *
 * WHY THIS IS THE WHOLE TP QUESTION, ON ONE DIE
 * ---------------------------------------------
 * saber is layer-split at 20: die A runs layers 0-19, then die B runs 20-39, so the
 * two dies together spend ~52.6 ms on 40 layers at full width. Under tensor
 * parallelism both dies run ALL 40 layers at half width, concurrently:
 *
 *     t_TP    = 40 * t_half_layer
 *     t_today = 40 * t_full_layer = 52.6 ms
 *     speedup = t_full / t_half = eta
 *
 * So eta IS the TP speedup, before any cross-die barrier is paid. eta = 2 means the
 * kernels split cleanly and TP is a 2x lever; eta = 1 means splitting the tensor buys
 * nothing and TP is dead with no two-die measurement needed. Auto-memory
 * `k80-work-reduction-doesnt-convert` (a 4.15x work cut returning 1.097x, bound by
 * BLOCK COUNT) says the prior is bad -- a row split halves block count.
 *
 * The MoE rows additionally price EXPERT parallelism (halve k, the number of routed
 * experts per launch) separately from tensor parallelism (halve ff, the expert
 * intermediate width). They are different splits with different costs.
 *
 * PROVING THE HARNESS CAN FAIL (six green-but-vacuous gates shipped here in 48 hours)
 * ---------------------------------------------------------------------------------
 * Two controls run first and their expected values are asserted, not eyeballed:
 *   NULL control     -- the same shape in both arms; eta MUST land in [0.97, 1.03].
 *                       If a "half" arm is silently running the full shape, this is
 *                       the only thing that catches it.
 *   QUARTER control  -- rows/4 in the half arm; eta MUST exceed the rows/2 eta.
 *                       If the kernel is fully latency-bound, both collapse to ~1 and
 *                       this reports that honestly rather than hiding it.
 * A control failure sets the exit code; the timing table is still printed.
 *
 * Arms ALTERNATE full/half/full/half within a rep and the table reports the MEDIAN of
 * REPS reps, because this board's clock moves under load (an
 * unpinned clock drifts between arms). Pin the clock before trusting any number:
 *     sudo nvidia-smi -i <die> -ac 2505,666      (and trap-restore 2505,875)
 *
 * No model is loaded and no correctness is claimed: weights are random quant bytes
 * with sane fp16 block scales. bench_matvec.c already gates these kernels bit-exactly
 * against CPU replicas; this file only times them.
 *
 * Build (from foundry/):
 *   gcc -O2 -I. -o cuda/tests/bench_split_scaling cuda/tests/bench_split_scaling.c \
 *       cuda/fc_cuda.c -lm -ldl
 * Run (ONE die, never a die another agent is using):
 *   CUDA_VISIBLE_DEVICES=7 ./cuda/tests/bench_split_scaling
 * Env:
 *   SPLIT_REPS   number of alternating reps (default 5)
 *   SPLIT_ITERS  launches timed per arm (default 200; LM head uses 20)
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "cuda/fc_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------
 * saber (Qwen3.6-35B-A3B-NSC-ACE-SABER) decode geometry.
 * Dense shapes from a census of the model's GGUF; SSM/attention dims from
 * Pantheon's qwen35moe model definition.
 * ------------------------------------------------------------------ */
#define SB_N_EMBD        2048
#define SB_N_VOCAB       248320
#define SB_N_EXPERT      256
#define SB_EXPERT_FF     512
#define SB_TOPK          8
#define SB_SSM_STATE     128     /* per-head state matrix is 128 x 128 */
#define SB_SSM_V_HEADS   32
#define SB_SSM_K_HEADS   16
#define SB_CONV_DIM      8192    /* key*2 + value */
#define SB_CONV_KERNEL   4
#define SB_ATTN_HEADS    16
#define SB_HEAD_DIM      256
#define SB_ATTN_KV_HEADS 2       /* GQA 8:1 -- and the CEILING on TP width */
/* Decode flash split plan, mirroring qwen35moe_forward_cuda.c:114-115. */
#define SB_FLASH_KPS       256
#define SB_FLASH_SPLIT_MAX 32

/* GGUF quant type ids used below. */
#define Q3K 11
#define Q4K 12
#define Q6K 14

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

/* ------------------------------------------------------------------
 * Random weights (mirrors bench_matvec.c: fp16 block scales clamped to a
 * sane exponent so no arm ever hits a denormal/inf slow path -- an arm that
 * silently ran on inf data would be a timing artefact, not a measurement).
 * ------------------------------------------------------------------ */

static size_t raw_block_bytes(int t)
{
    switch (t) {
    case 8:  return 34;
    case 11: return 110;
    case 12: return 144;
    case 13: return 176;
    case 14: return 210;
    default: return 0;
    }
}

static void sanitize_quant(int t, unsigned char *p, size_t bytes)
{
    size_t bs = raw_block_bytes(t);
    size_t off;
    for (off = 0; bs && off + bs <= bytes; off += bs) {
        unsigned char *b = p + off;
        unsigned short h = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        unsigned short h2 = (unsigned short)(0x2C00 | (rand() & 0x3FF));
        switch (t) {
        case 11: b[108] = h & 0xFF; b[109] = h >> 8; break;
        case 14: b[208] = h & 0xFF; b[209] = h >> 8; break;
        case 12: case 13:
            b[0] = h & 0xFF;  b[1] = h >> 8;
            b[2] = h2 & 0xFF; b[3] = h2 >> 8;
            break;
        case 8:  b[0] = h & 0xFF;  b[1] = h >> 8; break;
        }
    }
}

static float *rand_floats(size_t n)
{
    float *p = malloc(n * sizeof(float));
    size_t i;
    if (!p)
        return NULL;
    for (i = 0; i < n; i++)
        p[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    return p;
}

/* Upload `rows` x `cols` of random quant bytes, repacked exactly as a model
 * upload does (fc_repack_weight). Writes the device row stride. */
static fc_buffer *make_weight(int t, long rows, int cols, int *dev_stride)
{
    size_t raw_stride = raw_block_bytes(t) * (size_t)((t == 8) ? cols / 32
                                                               : cols / 256);
    size_t rs = fc_repack_row_bytes(t, cols);
    unsigned char *raw = NULL, *packed = NULL;
    fc_buffer *b = NULL;
    size_t i;

    raw = malloc(raw_stride * (size_t)rows);
    packed = malloc(rs * (size_t)rows);
    if (!raw || !packed)
        goto out;
    for (i = 0; i < raw_stride * (size_t)rows; i++)
        raw[i] = (unsigned char)(rand() & 0xFF);
    sanitize_quant(t, raw, raw_stride * (size_t)rows);
    if (fc_repack_weight(t, raw, packed, rows, cols) != 0)
        goto out;
    b = fc_buffer_from_ptr(packed, rs * (size_t)rows);
    *dev_stride = (int)rs;
out:
    free(raw);
    free(packed);
    return b;
}

static fc_buffer *make_floats(size_t n)
{
    float *h = rand_floats(n);
    fc_buffer *b = h ? fc_buffer_from_ptr(h, n * sizeof(float)) : NULL;
    free(h);
    return b;
}

/* ------------------------------------------------------------------
 * The arm abstraction: one timed kernel configuration.
 *
 * Each row of the table is a pair of arms (full, half) sharing one prepared
 * workload. `run` issues ONE launch of that arm; the harness handles warmup,
 * iteration and alternation so both arms see identical treatment.
 * ------------------------------------------------------------------ */

typedef void (*arm_run_fn)(void *ctx, int half);

typedef struct split_case {
    const char *name;        /* kernel + split axis, e.g. "attn_qg  row-split" */
    const char *full_shape;
    const char *half_shape;
    arm_run_fn  run;
    void       *ctx;
    int         iters;
    double      t_full_ms;   /* median over reps */
    double      t_half_ms;
    double      eta;         /* t_full / t_half */
} split_case;

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Time `iters` launches of one arm, returning ms per launch. */
static double time_arm(split_case *c, int half)
{
    int i;
    double t0;

    c->run(c->ctx, half);          /* warm up outside the timed region */
    fc_sync();
    t0 = now_s();
    for (i = 0; i < c->iters; i++)
        c->run(c->ctx, half);
    fc_sync();
    return (now_s() - t0) * 1e3 / c->iters;
}

/* Run one case for `reps` alternating full/half rounds and keep the medians.
 * Alternating (not blocked) is what keeps a DVFS drift from landing entirely
 * on one arm -- the board's clock moves ~2% under load. */
static void measure_case(split_case *c, int reps)
{
    double *full = malloc((size_t)reps * sizeof(double));
    double *half = malloc((size_t)reps * sizeof(double));
    int r;

    if (!full || !half) {
        free(full);
        free(half);
        return;
    }
    for (r = 0; r < reps; r++) {
        full[r] = time_arm(c, 0);
        half[r] = time_arm(c, 1);
    }
    qsort(full, (size_t)reps, sizeof(double), cmp_double);
    qsort(half, (size_t)reps, sizeof(double), cmp_double);
    c->t_full_ms = full[reps / 2];
    c->t_half_ms = half[reps / 2];
    c->eta = (c->t_half_ms > 0.0) ? c->t_full_ms / c->t_half_ms : 0.0;
    free(full);
    free(half);
}

/* ------------------------------------------------------------------
 * Case 1 -- dense quant matvec, split by output ROWS or by input COLS.
 *
 * Row split  = Megatron column-parallel: each die owns half the output rows,
 *              reads the whole activation, produces a disjoint half (concat).
 * Col split  = Megatron row-parallel: each die owns half the input columns,
 *              produces a full-length PARTIAL that must be summed (all-reduce).
 * Both are modelled here purely as a smaller launch; the reduce is priced
 * separately by bench_cross_die.
 * ------------------------------------------------------------------ */

typedef struct mv_ctx {
    int         quant;
    fc_buffer  *w_full, *w_half;
    fc_buffer  *x_full, *x_half;
    fc_buffer  *y;
    int         rows_full, cols_full, stride_full;
    int         rows_half, cols_half, stride_half;
} mv_ctx;

static void mv_dispatch(fc_cuda_ctx *ctx, int quant, fc_buffer *w, fc_buffer *x,
                        fc_buffer *y, int rows, int cols, int stride)
{
    switch (quant) {
    case Q3K: fc_matvec_q3k_ctx(ctx, w, x, y, rows, cols, stride); break;
    case Q4K: fc_matvec_q4k_ctx(ctx, w, x, y, rows, cols, stride); break;
    case Q6K: fc_matvec_q6k_ctx(ctx, w, x, y, rows, cols, stride); break;
    default:  break;
    }
}

static void mv_run(void *vc, int half)
{
    mv_ctx *m = (mv_ctx *)vc;
    if (half)
        mv_dispatch(fc_default_ctx(), m->quant, m->w_half, m->x_half, m->y,
                    m->rows_half, m->cols_half, m->stride_half);
    else
        mv_dispatch(fc_default_ctx(), m->quant, m->w_full, m->x_full, m->y,
                    m->rows_full, m->cols_full, m->stride_full);
}

/* Build a matvec case. rows_half/cols_half give the post-split shape; passing
 * the full shape for both is the NULL control. */
static int mv_case(split_case *c, const char *name, int quant,
                   int rows_full, int cols_full, int rows_half, int cols_half,
                   int iters, char *shape_a, char *shape_b)
{
    mv_ctx *m = calloc(1, sizeof(*m));
    if (!m)
        return -1;
    m->quant = quant;
    m->rows_full = rows_full; m->cols_full = cols_full;
    m->rows_half = rows_half; m->cols_half = cols_half;
    m->w_full = make_weight(quant, rows_full, cols_full, &m->stride_full);
    m->w_half = make_weight(quant, rows_half, cols_half, &m->stride_half);
    m->x_full = make_floats((size_t)cols_full);
    m->x_half = make_floats((size_t)cols_half);
    m->y      = fc_buffer_alloc((size_t)rows_full * sizeof(float));
    if (!m->w_full || !m->w_half || !m->x_full || !m->x_half || !m->y) {
        fprintf(stderr, "%s: allocation failed (VRAM?)\n", name);
        free(m);
        return -1;
    }
    sprintf(shape_a, "%dx%d", rows_full, cols_full);
    sprintf(shape_b, "%dx%d", rows_half, cols_half);
    c->name = name;
    c->full_shape = shape_a;
    c->half_shape = shape_b;
    c->run = mv_run;
    c->ctx = m;
    c->iters = iters;
    return 0;
}

/* ------------------------------------------------------------------
 * Case 2 -- grouped MoE gate/up and down.
 *
 * Two independent split axes, and they are NOT interchangeable:
 *   TP (halve ff)  -- both dies see all 8 routed experts, each computing half
 *                     the expert intermediate. Perfectly balanced, needs an
 *                     all-reduce after `down`.
 *   EP (halve k)   -- each die owns 128 of the 256 experts and computes only
 *                     the ~4 that routed to it. One reduce, but the split is
 *                     data-dependent and therefore imbalanced (see the doc's
 *                     E[max] = 5.09/8 argument).
 * ------------------------------------------------------------------ */

typedef struct moe_ctx {
    fc_buffer *gate, *up, *down;
    fc_buffer *x, *hid, *out, *top_idx, *top_wt;
    int gu_stride, dn_stride;
    int k_full, k_half;
    int ff_full, ff_half;
    int rows;                /* down output rows == n_embd */
    int is_down;
} moe_ctx;

static void moe_run(void *vc, int half)
{
    moe_ctx *m = (moe_ctx *)vc;
    int k = half ? m->k_half : m->k_full;
    int ff = half ? m->ff_half : m->ff_full;

    if (m->is_down)
        fc_moe_down_accum_ctx(fc_default_ctx(), Q3K, m->down, m->hid,
                              m->top_idx, m->top_wt, m->out, k, m->rows, ff,
                              m->dn_stride);
    else
        fc_moe_gateup_q3k_ctx(fc_default_ctx(), m->gate, m->up, m->x,
                              m->top_idx, m->hid, k, ff, SB_N_EMBD,
                              m->gu_stride);
}

/* Shared expert-weight pool: allocating all 256 experts of gate/up/down at
 * Q3_K is ~1.1 GB, so the pool is built ONCE and every MoE case points at it.
 * Only the k selected experts are read per launch, so the pool's size does not
 * affect the timing -- but it must be real memory, not a small buffer indexed
 * out of bounds, or the "half" arm would get a hotter cache than the full one. */
typedef struct moe_pool {
    fc_buffer *gate, *up, *down;
    fc_buffer *x, *hid, *out, *top_idx, *top_wt;
    int gu_stride, dn_stride;
} moe_pool;

static int moe_pool_init(moe_pool *p)
{
    int idx[SB_TOPK];
    float wt[SB_TOPK];
    int i;

    memset(p, 0, sizeof(*p));
    p->gate = make_weight(Q3K, (long)SB_N_EXPERT * SB_EXPERT_FF, SB_N_EMBD,
                          &p->gu_stride);
    p->up   = make_weight(Q3K, (long)SB_N_EXPERT * SB_EXPERT_FF, SB_N_EMBD,
                          &p->gu_stride);
    p->down = make_weight(Q3K, (long)SB_N_EXPERT * SB_N_EMBD, SB_EXPERT_FF,
                          &p->dn_stride);
    p->x    = make_floats(SB_N_EMBD);
    p->hid  = fc_buffer_alloc((size_t)SB_TOPK * SB_EXPERT_FF * sizeof(float));
    p->out  = fc_buffer_alloc((size_t)SB_N_EMBD * sizeof(float));
    /* Spread the selected experts across the whole pool so neither arm gets a
     * cache-friendly contiguous run. */
    for (i = 0; i < SB_TOPK; i++) {
        idx[i] = (i * (SB_N_EXPERT / SB_TOPK)) + 3;
        wt[i] = 1.0f / (float)SB_TOPK;
    }
    p->top_idx = fc_buffer_from_ptr(idx, sizeof(idx));
    p->top_wt  = fc_buffer_from_ptr(wt, sizeof(wt));
    if (!p->gate || !p->up || !p->down || !p->x || !p->hid || !p->out ||
        !p->top_idx || !p->top_wt) {
        fprintf(stderr, "moe pool: allocation failed (VRAM?)\n");
        return -1;
    }
    return 0;
}

static int moe_case(split_case *c, const char *name, const moe_pool *p,
                    int is_down, int k_full, int k_half, int ff_full,
                    int ff_half, int iters, char *shape_a, char *shape_b)
{
    moe_ctx *m = calloc(1, sizeof(*m));
    if (!m)
        return -1;
    m->gate = p->gate; m->up = p->up; m->down = p->down;
    m->x = p->x; m->hid = p->hid; m->out = p->out;
    m->top_idx = p->top_idx; m->top_wt = p->top_wt;
    m->gu_stride = p->gu_stride; m->dn_stride = p->dn_stride;
    m->k_full = k_full; m->k_half = k_half;
    m->ff_full = ff_full; m->ff_half = ff_half;
    m->rows = SB_N_EMBD;
    m->is_down = is_down;
    sprintf(shape_a, "k=%d ff=%d", k_full, ff_full);
    sprintf(shape_b, "k=%d ff=%d", k_half, ff_half);
    c->name = name;
    c->full_shape = shape_a;
    c->half_shape = shape_b;
    c->run = moe_run;
    c->ctx = m;
    c->iters = iters;
    return 0;
}

/* ------------------------------------------------------------------
 * Case 3 -- the SSM block (Gated Delta Net), split by HEADS.
 *
 * A head split is the natural TP decomposition for a recurrent block: each die
 * owns half the value heads and their state matrices, and the recurrence never
 * crosses heads, so NO cross-die communication is needed inside the SSM at all
 * -- only at the block's input and output projections. That makes the SSM the
 * cheapest part of a TP scheme to communicate and, at 30 of saber's 40 layers,
 * the most important part to scale.
 * ------------------------------------------------------------------ */

typedef struct ssm_ctx {
    fc_buffer *s, *q, *k, *v, *g, *beta, *y;      /* recurrence */
    fc_buffer *qkv, *cw, *cs, *cout;              /* conv1d */
    fc_buffer *gn_y, *gn_w, *gn_z, *gn_out;       /* gated norm */
    int heads_full, heads_half;
    int conv_full, conv_half;
    int which;                                    /* 0 recur, 1 conv, 2 gnorm */
} ssm_ctx;

static void ssm_run(void *vc, int half)
{
    ssm_ctx *s = (ssm_ctx *)vc;
    fc_cuda_ctx *ctx = fc_default_ctx();
    int heads = half ? s->heads_half : s->heads_full;

    switch (s->which) {
    case 0:
        fc_ssm_recurrence_ctx(ctx, s->s, s->q, s->k, s->v, s->g, s->beta, s->y,
                              heads, SB_SSM_STATE,
                              1.0f / sqrtf((float)SB_SSM_STATE));
        break;
    case 1:
        fc_ssm_conv1d_silu_ctx(ctx, s->qkv, s->cw, s->cs, s->cout,
                               half ? s->conv_half : s->conv_full,
                               SB_CONV_KERNEL);
        break;
    default:
        fc_ssm_gated_norm_ctx(ctx, s->gn_y, s->gn_w, s->gn_z, s->gn_out,
                              heads, SB_SSM_STATE, 1e-6f);
        break;
    }
}

static int ssm_pool_init(ssm_ctx *s)
{
    size_t hs = (size_t)SB_SSM_V_HEADS * SB_SSM_STATE;

    memset(s, 0, sizeof(*s));
    s->heads_full = SB_SSM_V_HEADS;
    s->heads_half = SB_SSM_V_HEADS / 2;
    s->conv_full = SB_CONV_DIM;
    s->conv_half = SB_CONV_DIM / 2;
    s->s    = make_floats(hs * SB_SSM_STATE);
    s->q    = make_floats(hs);
    s->k    = make_floats(hs);
    s->v    = make_floats(hs);
    s->g    = make_floats(SB_SSM_V_HEADS);
    s->beta = make_floats(SB_SSM_V_HEADS);
    s->y    = fc_buffer_alloc(hs * sizeof(float));
    s->qkv  = make_floats(SB_CONV_DIM);
    s->cw   = make_floats((size_t)SB_CONV_DIM * SB_CONV_KERNEL);
    s->cs   = make_floats((size_t)SB_CONV_DIM * (SB_CONV_KERNEL - 1));
    s->cout = fc_buffer_alloc((size_t)SB_CONV_DIM * sizeof(float));
    s->gn_y = make_floats(hs);
    s->gn_w = make_floats(SB_SSM_STATE);
    s->gn_z = make_floats(hs);
    s->gn_out = fc_buffer_alloc(hs * sizeof(float));
    if (!s->s || !s->q || !s->k || !s->v || !s->g || !s->beta || !s->y ||
        !s->qkv || !s->cw || !s->cs || !s->cout || !s->gn_y || !s->gn_w ||
        !s->gn_z || !s->gn_out) {
        fprintf(stderr, "ssm pool: allocation failed\n");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------
 * Case 4 -- decode flash attention, split by HEAD.
 *
 * The most favourable phase in the whole model for TP: heads are independent,
 * each die would own 8 of the 16 q-heads and 1 of the 2 kv-heads, and NO
 * cross-die communication is needed -- the split lives entirely between the
 * (already-split) input and output projections. It is also 13.7% of the step.
 *
 * But the flash decode grid is (n_splits, n_head_kv), and n_splits is
 * ceil(seq_len/256) capped at 32. At short context that is a 2-block launch on
 * a 13-SM die, and halving 2 blocks to 1 cannot make anything faster. So eta
 * here is CONTEXT-DEPENDENT and the sweep below is the point of this case:
 *   seq_len   512 -> n_splits  2 -> grid  4 blocks  (full) /  2 (half)
 *   seq_len  4096 -> n_splits 16 -> grid 32 blocks  (full) / 16 (half)
 *   seq_len 32768 -> n_splits 32 -> grid 64 blocks  (full) / 32 (half)
 * Anything that reports eta ~ 1.0 at 512 and ~ 2.0 at 32768 is telling us TP's
 * attention win only exists at depth -- which is where the serve actually runs.
 *
 * fp32 KV, not the q4 path the deployment runs (QWEN35_KV_Q4=1). The q4 kernel adds sink
 * and window shadow buffers; the head split is the same shape either way, but
 * the absolute ms here are NOT the deployment's.
 * ------------------------------------------------------------------ */

typedef struct flash_ctx {
    fc_buffer *q_full, *q_half;
    fc_buffer *k_cache, *v_cache;      /* [n_head_kv][max_seq][head_dim] */
    fc_buffer *m_scr, *l_scr, *acc_scr, *out;
    int seq_len, max_seq;
    int n_splits, split_len;
} flash_ctx;

/* The engine's plan, verbatim (qwen35moe_forward_cuda.c:4592-4602). */
static void flash_plan(int seq_len, int *n_splits, int *split_len)
{
    int ns = (seq_len + SB_FLASH_KPS - 1) / SB_FLASH_KPS;
    int sl;
    if (ns < 1)
        ns = 1;
    if (ns > SB_FLASH_SPLIT_MAX)
        ns = SB_FLASH_SPLIT_MAX;
    sl = (seq_len + ns - 1) / ns;
    sl = (sl + 15) & ~15;
    *split_len = sl;
    *n_splits = (seq_len + sl - 1) / sl;
}

static void flash_run(void *vc, int half)
{
    flash_ctx *f = (flash_ctx *)vc;
    int nh = half ? SB_ATTN_HEADS / 2 : SB_ATTN_HEADS;
    int nkv = half ? SB_ATTN_KV_HEADS / 2 : SB_ATTN_KV_HEADS;

    fc_attn_flash_decode_ctx(fc_default_ctx(), half ? f->q_half : f->q_full,
                             f->k_cache, f->v_cache, f->m_scr, f->l_scr,
                             f->acc_scr, f->out, nh, nkv, SB_HEAD_DIM,
                             f->max_seq, f->seq_len, f->n_splits, f->split_len,
                             0, 0, 1.0f / 16.0f);
}

/* Build one flash case at `seq_len`. Returns -1 if the kernels are absent from
 * this PTX blob (they are optional) so the row is skipped rather than faked. */
static int flash_case(split_case *c, const char *name, int seq_len, int iters,
                      char *shape_a, char *shape_b)
{
    flash_ctx *f = calloc(1, sizeof(*f));
    size_t ring = (size_t)SB_ATTN_KV_HEADS * seq_len * SB_HEAD_DIM;
    size_t np = (size_t)SB_ATTN_HEADS * SB_FLASH_SPLIT_MAX;

    if (!f)
        return -1;
    if (!fc_attn_flash_decode_available_ctx(fc_default_ctx())) {
        free(f);
        return -1;
    }
    f->seq_len = seq_len;
    f->max_seq = seq_len;
    flash_plan(seq_len, &f->n_splits, &f->split_len);
    f->q_full   = make_floats((size_t)SB_ATTN_HEADS * SB_HEAD_DIM);
    f->q_half   = make_floats((size_t)(SB_ATTN_HEADS / 2) * SB_HEAD_DIM);
    f->k_cache  = make_floats(ring);
    f->v_cache  = make_floats(ring);
    f->m_scr    = fc_buffer_alloc(np * sizeof(float));
    f->l_scr    = fc_buffer_alloc(np * sizeof(float));
    f->acc_scr  = fc_buffer_alloc(np * SB_HEAD_DIM * sizeof(float));
    f->out      = fc_buffer_alloc((size_t)SB_ATTN_HEADS * SB_HEAD_DIM *
                                  sizeof(float));
    if (!f->q_full || !f->q_half || !f->k_cache || !f->v_cache || !f->m_scr ||
        !f->l_scr || !f->acc_scr || !f->out) {
        fprintf(stderr, "%s: allocation failed (VRAM?)\n", name);
        free(f);
        return -1;
    }
    sprintf(shape_a, "16h/2kv s=%d", seq_len);
    sprintf(shape_b, "8h/1kv s=%d", seq_len);
    c->name = name;
    c->full_shape = shape_a;
    c->half_shape = shape_b;
    c->run = flash_run;
    c->ctx = f;
    c->iters = iters;
    return 0;
}

/* ------------------------------------------------------------------
 * Reporting
 * ------------------------------------------------------------------ */

static void print_header(void)
{
    printf("\n%-34s %-14s %-14s %10s %10s %7s\n", "kernel / split axis",
           "full", "half", "t_full ms", "t_half ms", "eta");
    printf("%-34s %-14s %-14s %10s %10s %7s\n", "----------------------------------",
           "--------------", "--------------", "----------", "----------",
           "-------");
}

static void print_case(const split_case *c)
{
    printf("%-34s %-14s %-14s %10.4f %10.4f %6.3fx\n", c->name, c->full_shape,
           c->half_shape, c->t_full_ms, c->t_half_ms, c->eta);
}

int main(void)
{
    /* Shape strings are held by the cases, so they must outlive the table. */
    static char shp[64][2][32];
    static split_case cases[64];
    moe_pool mp;
    ssm_ctx ssm_r, ssm_c, ssm_g;
    int reps = env_int("SPLIT_REPS", 5);
    int iters = env_int("SPLIT_ITERS", 200);
    int n = 0, i, control_fail = 0;
    double null_eta = 0.0, half_eta = 0.0, quarter_eta = 0.0;

    if (fc_init() != 0) {
        fprintf(stderr, "no CUDA driver / no visible die\n");
        return 1;
    }
    srand(42);
    printf("bench_split_scaling -- does halving the work halve the time?\n");
    printf("device %d, reps=%d iters=%d\n", fc_cuda_ctx_device_id(fc_default_ctx()),
           reps, iters);

    /* ---- controls (see the file header: these are what make the table
     * trustworthy; a NULL control off 1.00 means an arm is not what it says) */
    if (mv_case(&cases[n], "CONTROL null (same shape both arms)", Q4K,
                8192, SB_N_EMBD, 8192, SB_N_EMBD, iters,
                shp[n][0], shp[n][1]) == 0) n++;
    if (mv_case(&cases[n], "CONTROL quarter (rows/4)", Q4K,
                8192, SB_N_EMBD, 2048, SB_N_EMBD, iters,
                shp[n][0], shp[n][1]) == 0) n++;

    /* ---- dense projections, decode width (n = 1) ---- */
    /* SSM in_proj / attention Q+gate: 8192 rows -> 4096 (column-parallel). */
    if (mv_case(&cases[n], "ssm_in_proj / attn_qg  ROW", Q4K,
                SB_CONV_DIM, SB_N_EMBD, SB_CONV_DIM / 2, SB_N_EMBD, iters,
                shp[n][0], shp[n][1]) == 0) n++;
    /* attn_k / attn_v: 512 rows -> 256. Already tiny; this is where a row
     * split is most likely to hit the block-count floor. */
    if (mv_case(&cases[n], "attn_k / attn_v  ROW", Q4K,
                512, SB_N_EMBD, 256, SB_N_EMBD, iters,
                shp[n][0], shp[n][1]) == 0) n++;
    /* attn_output: row-parallel, so the CONTRACTION halves, not the rows. */
    if (mv_case(&cases[n], "attn_output  COL", Q4K,
                SB_N_EMBD, 4096, SB_N_EMBD, 2048, iters,
                shp[n][0], shp[n][1]) == 0) n++;
    /* shared-expert gate+up (column-parallel) and down (row-parallel). */
    if (mv_case(&cases[n], "shexp_gate_up  ROW", Q3K,
                2 * SB_EXPERT_FF, SB_N_EMBD, SB_EXPERT_FF, SB_N_EMBD, iters,
                shp[n][0], shp[n][1]) == 0) n++;
    if (mv_case(&cases[n], "shexp_down  COL", Q4K,
                SB_N_EMBD, SB_EXPERT_FF, SB_N_EMBD, SB_EXPERT_FF / 2, iters,
                shp[n][0], shp[n][1]) == 0) n++;
    /* LM head: the shipped T3 vocab split. This row is the cross-check against
     * the live +1.01 ms/step that QWEN35_VOCAB_SPLIT measured on the serve. */
    if (mv_case(&cases[n], "lm_head  ROW (= shipped T3)", Q6K,
                SB_N_VOCAB, SB_N_EMBD, SB_N_VOCAB / 2, SB_N_EMBD, 20,
                shp[n][0], shp[n][1]) == 0) n++;

    /* ---- grouped MoE ---- */
    if (moe_pool_init(&mp) == 0) {
        if (moe_case(&cases[n], "moe_gateup  TP (halve ff)", &mp, 0,
                     SB_TOPK, SB_TOPK, SB_EXPERT_FF, SB_EXPERT_FF / 2, iters,
                     shp[n][0], shp[n][1]) == 0) n++;
        if (moe_case(&cases[n], "moe_gateup  EP (halve k)", &mp, 0,
                     SB_TOPK, SB_TOPK / 2, SB_EXPERT_FF, SB_EXPERT_FF, iters,
                     shp[n][0], shp[n][1]) == 0) n++;
        if (moe_case(&cases[n], "moe_down  TP (halve ff)", &mp, 1,
                     SB_TOPK, SB_TOPK, SB_EXPERT_FF, SB_EXPERT_FF / 2, iters,
                     shp[n][0], shp[n][1]) == 0) n++;
        if (moe_case(&cases[n], "moe_down  EP (halve k)", &mp, 1,
                     SB_TOPK, SB_TOPK / 2, SB_EXPERT_FF, SB_EXPERT_FF, iters,
                     shp[n][0], shp[n][1]) == 0) n++;
        /* EP at the WORST routed split the doc's binomial allows: 7 of 8
         * experts on one die. The busy die sets the critical path, so this is
         * the number an EP scheme actually pays on its unlucky tokens. */
        if (moe_case(&cases[n], "moe_gateup  EP worst (7 of 8)", &mp, 0,
                     SB_TOPK, 7, SB_EXPERT_FF, SB_EXPERT_FF, iters,
                     shp[n][0], shp[n][1]) == 0) n++;
    }

    /* ---- decode flash attention, head split, swept over context depth ----
     * The grid is (n_splits, n_head_kv) and n_splits comes from seq_len, so a
     * head split has almost no blocks to take away at short context and a
     * useful number at depth. Sweeping is the only honest way to report it. */
    {
        static const int depths[] = { 512, 4096, 16384, 32768 };
        static char fname[4][40];
        size_t d;
        for (d = 0; d < sizeof(depths) / sizeof(depths[0]); d++) {
            sprintf(fname[d], "attn_flash  HEAD @ %d ctx", depths[d]);
            if (flash_case(&cases[n], fname[d], depths[d], iters / 4 + 1,
                           shp[n][0], shp[n][1]) == 0)
                n++;
        }
    }

    /* ---- SSM block, head split ---- */
    if (ssm_pool_init(&ssm_r) == 0) {
        memcpy(&ssm_c, &ssm_r, sizeof(ssm_c));
        memcpy(&ssm_g, &ssm_r, sizeof(ssm_g));
        ssm_r.which = 0; ssm_c.which = 1; ssm_g.which = 2;
        cases[n].name = "ssm_recurrence  HEAD";
        cases[n].full_shape = "32 heads"; cases[n].half_shape = "16 heads";
        cases[n].run = ssm_run; cases[n].ctx = &ssm_r; cases[n].iters = iters; n++;
        cases[n].name = "ssm_conv1d_silu  HEAD";
        cases[n].full_shape = "conv 8192"; cases[n].half_shape = "conv 4096";
        cases[n].run = ssm_run; cases[n].ctx = &ssm_c; cases[n].iters = iters; n++;
        cases[n].name = "ssm_gated_norm  HEAD";
        cases[n].full_shape = "32 heads"; cases[n].half_shape = "16 heads";
        cases[n].run = ssm_run; cases[n].ctx = &ssm_g; cases[n].iters = iters; n++;
    }

    print_header();
    for (i = 0; i < n; i++) {
        measure_case(&cases[i], reps);
        print_case(&cases[i]);
        fflush(stdout);
    }

    /* ---- control verdict ---- */
    null_eta = cases[0].eta;
    quarter_eta = cases[1].eta;
    for (i = 0; i < n; i++)
        if (strstr(cases[i].name, "ssm_in_proj"))
            half_eta = cases[i].eta;

    printf("\n== controls ==\n");
    if (null_eta < 0.97 || null_eta > 1.03) {
        printf("  FAIL  null control eta = %.3f, expected 1.00 +/- 0.03 --"
               " the two arms are NOT running the same shape; the table above"
               " is not trustworthy\n", null_eta);
        control_fail = 1;
    } else {
        printf("  pass  null control eta = %.3f (arms are comparable)\n", null_eta);
    }
    if (quarter_eta <= half_eta) {
        printf("  NOTE  quarter control eta = %.3f is NOT above the rows/2 eta"
               " %.3f -- this kernel does not scale with work at all at these"
               " sizes (a real finding, not a harness fault)\n",
               quarter_eta, half_eta);
    } else {
        printf("  pass  quarter control eta = %.3f > rows/2 eta %.3f"
               " (the harness can see work reduction)\n", quarter_eta, half_eta);
    }

    printf("\neta is the tensor-parallel speedup ceiling before any cross-die"
           " barrier.\n2.00 = perfect split, 1.00 = splitting buys nothing.\n");
    fc_free();
    return control_fail;
}
