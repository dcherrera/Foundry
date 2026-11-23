/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_metal_attn.c — attention family (fm_attn_kv_append,
 * fm_attn_flash_decode + fm_attn_flash_decode_reduce, fm_attn_flash_prefill)
 *
 * THERE ARE NO GOLDEN VECTORS FOR ATTENTION. The oracle is
 * reference_attention_heads() below: a dense, double-precision, single-pass
 * softmax written from the definition
 *
 *     out[h] = sum_j softmax_j(scale * dot(q[h], k[h/groups][j])) * v[h/groups][j]
 *
 * with no tiling, no running max and no split — i.e. deliberately none of the
 * structure the kernels use. It was written before the kernels and is not
 * derived from them, so it cannot agree with a kernel by sharing its bug.
 *
 * EVERY assertion is against that reference or against the kernel's own input.
 * The one GPU-vs-GPU comparison (single-split vs multi-split decode) is a
 * SECONDARY check that runs only alongside the absolute one — Phase 1 showed
 * GPU-vs-GPU arms stay green through breaks that change both arms.
 *
 * Depths tested: 1, 2, 16, 17, 513, 4097 live keys. 16 is exactly one tile,
 * 17 is one full tile plus a one-key tail, 1 is the empty-cache case (the
 * first token attends only itself), 4097 is a partial tail at long context.
 *
 * Non-bit-exactness is expected (tile order); the bar is ~1e-3 with a stable
 * argmax (spec §11). Measured drift is printed per depth so a regression shows
 * up as a number moving, not as a tolerance being widened.
 *
 * Exits 77 when no Metal device is present so GPU-less CI skips cleanly.
 */

#include "lib/test.h"
#include "metal/kernels/fm_kernel_attn.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* qwen35moe attention shape (spec §6). */
#define ATTN_HEAD_DIM    256
#define ATTN_N_HEAD       16
#define ATTN_N_HEAD_KV     2

/* A second shape, so nothing can pass by hard-coding the first. */
#define ALT_HEAD_DIM      64
#define ALT_N_HEAD         8
#define ALT_N_HEAD_KV      4

/* Spec §11 bar for the by-design non-bit-exact kernels: max_abs_err/RMS. */
#define ATTN_TOL_ABS    1e-3f

/* The working bar. Deep-context outputs are a weighted mean of thousands of
 * bounded values, so they are SMALL — at depth 4097 the whole output has an
 * RMS around 0.02, and a plain 1e-3 absolute bar there is a 5% error budget.
 * Anything smaller than that would sail through, including a uniform 0.01%
 * scaling of the result. Normalizing by the reference RMS makes the bar mean
 * the same thing at every depth. Measured worst ratio is ~4e-6, so this sits
 * ~25x above the real drift and ~3x below a 0.01% scale error. */
#define ATTN_TOL_RMS    1e-4f

static const int attn_depths[] = { 1, 2, 16, 17, 513, 4097 };
#define ATTN_N_DEPTHS ((int)(sizeof(attn_depths) / sizeof(attn_depths[0])))

/* Worst drift seen across every comparison in the run, reported at the end. */
static double attn_worst_abs = 0.0;
static double attn_worst_ratio = 0.0;

/* ================================================================
 * Deterministic inputs
 * ================================================================ */

/* xorshift32: same bytes on every machine and every run, so a failure is
 * reproducible without shipping a data file. */
static uint32_t attn_rng_state = 0x2545F491u;

static void attn_rng_seed(uint32_t seed) {
    attn_rng_state = seed ? seed : 1u;
}

static float attn_rng_signed(void) {
    attn_rng_state ^= attn_rng_state << 13;
    attn_rng_state ^= attn_rng_state >> 17;
    attn_rng_state ^= attn_rng_state << 5;
    return (float)((double)attn_rng_state / 2147483648.0 - 1.0);
}

static void attn_fill_random(float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = attn_rng_signed();
}

/* isfinite() is compiled out by the -ffast-math this file is built with
 * (finite-math-only lets the compiler fold it to true), which would make the
 * NaN check in the empty-split test vacuous. Read the exponent field instead
 * — no floating-point comparison for the optimizer to reason about. */
static int attn_bits_are_finite(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return ((bits >> 23) & 0xFFu) != 0xFFu;
}

/* ================================================================
 * The oracle
 * ================================================================ */

/*
 * Dense attention for one query row, all heads, in double precision.
 *
 * q_row     [n_head][head_dim]
 * k_cache   [n_head_kv][max_seq][head_dim], rows [0, n_keys) live
 * out_row   [n_head][head_dim] (written)
 *
 * Query head h reads kv head h / kv_groups — the same GQA map the kernels
 * use, stated here independently. Single global max, one exp pass, one
 * normalize: no online rescaling, which is exactly what makes this a check
 * on the kernels' rescaling rather than a copy of it.
 */
static void reference_attention_heads(const float *q_row, const float *k_cache,
                                      const float *v_cache, float *out_row,
                                      int n_head, int kv_groups, int head_dim,
                                      int max_seq, int n_keys, float scale) {
    double *prob = (double *)malloc((size_t)(n_keys > 0 ? n_keys : 1) * sizeof(double));
    if (!prob) return;

    for (int h = 0; h < n_head; h++) {
        int kvh = h / kv_groups;
        const float *kh = k_cache + (size_t)kvh * max_seq * head_dim;
        const float *vh = v_cache + (size_t)kvh * max_seq * head_dim;
        const float *qh = q_row + (size_t)h * head_dim;

        /* -DBL_MAX, not -INFINITY: this file is built with -ffast-math and an
         * infinite operand is exactly what that tells the compiler cannot
         * happen. Every real score is far above it. */
        double peak = -DBL_MAX;
        for (int j = 0; j < n_keys; j++) {
            double s = 0.0;
            for (int e = 0; e < head_dim; e++)
                s += (double)qh[e] * (double)kh[(size_t)j * head_dim + e];
            s *= (double)scale;
            prob[j] = s;
            if (s > peak) peak = s;
        }

        double total = 0.0;
        for (int j = 0; j < n_keys; j++) {
            prob[j] = exp(prob[j] - peak);
            total += prob[j];
        }

        for (int e = 0; e < head_dim; e++) {
            double o = 0.0;
            for (int j = 0; j < n_keys; j++)
                o += prob[j] * (double)vh[(size_t)j * head_dim + e];
            out_row[(size_t)h * head_dim + e] = (float)(o / total);
        }
    }
    free(prob);
}

/* ================================================================
 * Comparison
 * ================================================================ */

/* Diff against the reference: worst absolute error, and that error over the
 * reference's RMS (the scale-free form, which is what actually gets asserted).
 * `label` names the case because several tests share this. */
static int case_compare(const char *label, const float *want, const float *got,
                        size_t n) {
    double worst_abs = 0.0, sum_sq = 0.0;
    size_t worst_i = 0;

    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)want[i] - (double)got[i]);
        if (d > worst_abs) { worst_abs = d; worst_i = i; }
        sum_sq += (double)want[i] * (double)want[i];
    }
    double rms = sqrt(sum_sq / (double)(n ? n : 1));
    double ratio = worst_abs / (rms + 1e-12);

    if (worst_abs > attn_worst_abs) attn_worst_abs = worst_abs;
    if (ratio > attn_worst_ratio) attn_worst_ratio = ratio;

    printf("    %-44s max_abs %.3e  /rms %.3e\n", label, worst_abs, ratio);
    if (worst_abs > (double)ATTN_TOL_ABS || ratio > (double)ATTN_TOL_RMS) {
        printf("      worst element %zu: want %.9g got %.9g (rms %.4g)\n",
               worst_i, (double)want[worst_i], (double)got[worst_i], rms);
        return 0;
    }
    return 1;
}

/* Per-head argmax over head_dim must be identical — the stable-argmax half
 * of the spec §11 bar. A drift that only shows up as a flipped greedy pick
 * would slip past a pure magnitude check on a near-tie. */
static int case_argmax_matches(const float *want, const float *got,
                               int n_head, int head_dim) {
    for (int h = 0; h < n_head; h++) {
        int wi = 0, gi = 0;
        for (int e = 1; e < head_dim; e++) {
            if (want[(size_t)h * head_dim + e] > want[(size_t)h * head_dim + wi]) wi = e;
            if (got[(size_t)h * head_dim + e] > got[(size_t)h * head_dim + gi]) gi = e;
        }
        if (wi != gi) {
            printf("      head %d argmax: want %d got %d\n", h, wi, gi);
            return 0;
        }
    }
    return 1;
}

/* ================================================================
 * Fixture
 * ================================================================ */

typedef struct attn_fixture {
    int n_head, n_head_kv, kv_groups, head_dim, max_seq;
    float scale;
    fm_buffer *k_cache;
    fm_buffer *v_cache;
    fm_buffer *q;
    fm_buffer *out;
    float *k_host;     /* mirror of the cache, for the reference */
    float *v_host;
    float *q_host;
    float *want;
} attn_fixture;

static void fixture_free(attn_fixture *f) {
    fm_buffer_free(f->k_cache);
    fm_buffer_free(f->v_cache);
    fm_buffer_free(f->q);
    fm_buffer_free(f->out);
    free(f->k_host);
    free(f->v_host);
    free(f->q_host);
    free(f->want);
    memset(f, 0, sizeof(*f));
}

/* Allocates cache + query buffers and fills them with the deterministic
 * stream. `n_rows` is how many query rows the buffers must hold (1 for
 * decode). The cache is filled to max_seq so a causality bug reads live
 * data, not zeros — zeros would contribute a plausible-looking output and
 * could hide the bug. */
static int fixture_open(attn_fixture *f, int n_head, int n_head_kv,
                        int head_dim, int max_seq, int n_rows, uint32_t seed) {
    memset(f, 0, sizeof(*f));
    f->n_head = n_head;
    f->n_head_kv = n_head_kv;
    f->kv_groups = n_head / n_head_kv;
    f->head_dim = head_dim;
    f->max_seq = max_seq;
    f->scale = 1.0f / sqrtf((float)head_dim);

    size_t cache_n = (size_t)n_head_kv * max_seq * head_dim;
    size_t row_n = (size_t)n_rows * n_head * head_dim;

    f->k_cache = fm_buffer_alloc(cache_n * sizeof(float));
    f->v_cache = fm_buffer_alloc(cache_n * sizeof(float));
    f->q = fm_buffer_alloc(row_n * sizeof(float));
    f->out = fm_buffer_alloc(row_n * sizeof(float));
    f->k_host = (float *)malloc(cache_n * sizeof(float));
    f->v_host = (float *)malloc(cache_n * sizeof(float));
    f->q_host = (float *)malloc(row_n * sizeof(float));
    f->want = (float *)malloc(row_n * sizeof(float));
    if (!f->k_cache || !f->v_cache || !f->q || !f->out ||
        !f->k_host || !f->v_host || !f->q_host || !f->want) {
        fixture_free(f);
        return -1;
    }

    attn_rng_seed(seed);
    attn_fill_random(f->k_host, cache_n);
    attn_fill_random(f->v_host, cache_n);
    attn_fill_random(f->q_host, row_n);
    memcpy(fm_buffer_data(f->k_cache), f->k_host, cache_n * sizeof(float));
    memcpy(fm_buffer_data(f->v_cache), f->v_host, cache_n * sizeof(float));
    memcpy(fm_buffer_data(f->q), f->q_host, row_n * sizeof(float));
    return 0;
}

/* ================================================================
 * KV append
 * ================================================================ */

/* Every appended element must land at [h][(pos0+t)%max_seq][d] and every
 * untouched slot must keep its sentinel. An off-by-one in the position math
 * moves a whole row, which both halves catch: the row that should have been
 * written still holds the sentinel, and a row that should not have been
 * written no longer does. */
static void test_kv_append_lands_at_the_ring_position(void) {
    const int n = 5, n_head_kv = 3, head_dim = 32, max_seq = 11, pos0 = 7;
    size_t cache_n = (size_t)n_head_kv * max_seq * head_dim;
    size_t new_n = (size_t)n * n_head_kv * head_dim;

    fm_buffer *k_new = fm_buffer_alloc(new_n * sizeof(float));
    fm_buffer *v_new = fm_buffer_alloc(new_n * sizeof(float));
    fm_buffer *k_cache = fm_buffer_alloc(cache_n * sizeof(float));
    fm_buffer *v_cache = fm_buffer_alloc(cache_n * sizeof(float));
    ASSERT_NOT_NULL(k_new);
    ASSERT_NOT_NULL(v_new);
    ASSERT_NOT_NULL(k_cache);
    ASSERT_NOT_NULL(v_cache);

    float *kn = (float *)fm_buffer_data(k_new);
    float *vn = (float *)fm_buffer_data(v_new);
    /* Value encodes (row, head, dim) so a misplaced element is identifiable. */
    for (int t = 0; t < n; t++)
        for (int h = 0; h < n_head_kv; h++)
            for (int d = 0; d < head_dim; d++) {
                size_t i = ((size_t)t * n_head_kv + h) * head_dim + d;
                kn[i] = (float)(1000 * (t + 1) + 10 * h + d);
                vn[i] = -kn[i];
            }
    const float sentinel = -12345.0f;
    for (size_t i = 0; i < cache_n; i++) {
        ((float *)fm_buffer_data(k_cache))[i] = sentinel;
        ((float *)fm_buffer_data(v_cache))[i] = sentinel;
    }

    fm_attn_kv_append(k_new, v_new, k_cache, v_cache, n, n_head_kv, head_dim,
                      max_seq, pos0);
    fm_sync();

    const float *kc = (const float *)fm_buffer_data(k_cache);
    const float *vc = (const float *)fm_buffer_data(v_cache);
    for (int t = 0; t < n; t++) {
        int cache_pos = (pos0 + t) % max_seq;
        for (int h = 0; h < n_head_kv; h++)
            for (int d = 0; d < head_dim; d++) {
                size_t src = ((size_t)t * n_head_kv + h) * head_dim + d;
                size_t dst = ((size_t)h * max_seq + cache_pos) * head_dim + d;
                ASSERT_FLT_NEAR(kn[src], kc[dst], 0.0f);
                ASSERT_FLT_NEAR(vn[src], vc[dst], 0.0f);
            }
    }

    /* Slots outside the appended set must be untouched. pos0=7, n=5,
     * max_seq=11 wraps: positions 7,8,9,10,0 are written, 1..6 are not. */
    for (int h = 0; h < n_head_kv; h++)
        for (int p = 1; p <= 6; p++)
            for (int d = 0; d < head_dim; d++) {
                size_t i = ((size_t)h * max_seq + p) * head_dim + d;
                ASSERT_FLT_NEAR(sentinel, kc[i], 0.0f);
                ASSERT_FLT_NEAR(sentinel, vc[i], 0.0f);
            }

    fm_buffer_free(k_new);
    fm_buffer_free(v_new);
    fm_buffer_free(k_cache);
    fm_buffer_free(v_cache);
}

/* n = 0 must write nothing at all. A dispatch that rounds the thread count
 * up would clobber cache row pos0. */
static void test_kv_append_of_zero_rows_writes_nothing(void) {
    const int n_head_kv = 2, head_dim = 32, max_seq = 4;
    size_t cache_n = (size_t)n_head_kv * max_seq * head_dim;

    fm_buffer *k_new = fm_buffer_alloc(head_dim * sizeof(float));
    fm_buffer *v_new = fm_buffer_alloc(head_dim * sizeof(float));
    fm_buffer *k_cache = fm_buffer_alloc(cache_n * sizeof(float));
    fm_buffer *v_cache = fm_buffer_alloc(cache_n * sizeof(float));
    ASSERT_NOT_NULL(k_cache);
    ASSERT_NOT_NULL(v_cache);

    float *kc = (float *)fm_buffer_data(k_cache);
    float *vc = (float *)fm_buffer_data(v_cache);
    for (size_t i = 0; i < cache_n; i++) { kc[i] = 3.5f; vc[i] = 3.5f; }

    fm_attn_kv_append(k_new, v_new, k_cache, v_cache, 0, n_head_kv, head_dim,
                      max_seq, 0);
    fm_sync();

    for (size_t i = 0; i < cache_n; i++) {
        ASSERT_FLT_NEAR(3.5f, kc[i], 0.0f);
        ASSERT_FLT_NEAR(3.5f, vc[i], 0.0f);
    }

    fm_buffer_free(k_new);
    fm_buffer_free(v_new);
    fm_buffer_free(k_cache);
    fm_buffer_free(v_cache);
}

/* ================================================================
 * Decode
 * ================================================================ */

/* Runs the two-kernel decode path and returns the output pointer. */
static const float *run_decode(attn_fixture *f, int seq_len, int n_splits,
                               int split_len, fm_buffer *m_buf,
                               fm_buffer *l_buf, fm_buffer *acc_buf) {
    fm_attn_flash_decode(f->q, f->k_cache, f->v_cache, m_buf, l_buf, acc_buf,
                         f->kv_groups, f->n_head_kv, f->head_dim, f->max_seq,
                         seq_len, n_splits, split_len, f->scale);
    fm_attn_flash_decode_reduce(m_buf, l_buf, acc_buf, f->out, f->n_head,
                                f->head_dim, n_splits);
    fm_sync();
    return (const float *)fm_buffer_data(f->out);
}

/* Allocates split-K scratch for the largest plan a test will use. */
static int alloc_scratch(int n_head, int head_dim, int n_splits,
                         fm_buffer **m, fm_buffer **l, fm_buffer **acc) {
    size_t m_floats = 0, acc_floats = 0;
    fm_attn_decode_scratch_floats(n_head, head_dim, n_splits, &m_floats,
                                  &acc_floats);
    *m = fm_buffer_alloc(m_floats * sizeof(float));
    *l = fm_buffer_alloc(m_floats * sizeof(float));
    *acc = fm_buffer_alloc(acc_floats * sizeof(float));
    return (*m && *l && *acc) ? 0 : -1;
}

/* One workgroup per kv head: the split-K machinery is present but inert, so
 * this isolates the tiling and the online softmax from the combine. */
static void case_decode_depths(int n_head, int n_head_kv, int head_dim,
                               const char *shape) {
    char label[96];
    for (int d = 0; d < ATTN_N_DEPTHS; d++) {
        int seq_len = attn_depths[d];
        attn_fixture f;
        ASSERT_INT_EQ(0, fixture_open(&f, n_head, n_head_kv, head_dim,
                                      seq_len + 3, 1, 0x1234u + (uint32_t)d));

        fm_buffer *m = NULL, *l = NULL, *acc = NULL;
        ASSERT_INT_EQ(0, alloc_scratch(n_head, head_dim, 1, &m, &l, &acc));

        reference_attention_heads(f.q_host, f.k_host, f.v_host, f.want,
                                  n_head, f.kv_groups, head_dim, f.max_seq,
                                  seq_len, f.scale);
        const float *got = run_decode(&f, seq_len, 1, seq_len, m, l, acc);

        snprintf(label, sizeof(label), "%s decode depth %d (1 split)", shape,
                 seq_len);
        int ok = case_compare(label, f.want, got,
                              (size_t)n_head * head_dim);
        int argmax_ok = case_argmax_matches(f.want, got, n_head, head_dim);

        fm_buffer_free(m);
        fm_buffer_free(l);
        fm_buffer_free(acc);
        fixture_free(&f);
        ASSERT_TRUE(ok);
        ASSERT_TRUE(argmax_ok);
    }
}

static void test_decode_matches_reference_at_every_depth(void) {
    case_decode_depths(ATTN_N_HEAD, ATTN_N_HEAD_KV, ATTN_HEAD_DIM, "qwen35");
}

static void test_decode_matches_reference_at_alt_shape(void) {
    case_decode_depths(ALT_N_HEAD, ALT_N_HEAD_KV, ALT_HEAD_DIM, "alt");
}

/* SPLIT-K. Forces more than one workgroup per kv head and checks the combined
 * result against the CPU reference (the assertion that matters) and against
 * the single-workgroup GPU result (secondary — it would stay green if a break
 * hit both arms, which is exactly why it is not the primary check).
 *
 * The rescale in the reduce is what this exercises: with one split it never
 * runs, with many splits every split carries a different running max and the
 * exp(m_split - M) weights have to put them back on a common scale. */
static void test_decode_split_k_matches_reference_and_single_split(void) {
    const int seq_len = 4097;
    char label[96];

    attn_fixture f;
    ASSERT_INT_EQ(0, fixture_open(&f, ATTN_N_HEAD, ATTN_N_HEAD_KV,
                                  ATTN_HEAD_DIM, seq_len + 3, 1, 0x9E37u));
    reference_attention_heads(f.q_host, f.k_host, f.v_host, f.want,
                              ATTN_N_HEAD, f.kv_groups, ATTN_HEAD_DIM,
                              f.max_seq, seq_len, f.scale);

    size_t out_n = (size_t)ATTN_N_HEAD * ATTN_HEAD_DIM;
    float *single = (float *)malloc(out_n * sizeof(float));
    ASSERT_NOT_NULL(single);

    fm_buffer *m = NULL, *l = NULL, *acc = NULL;
    ASSERT_INT_EQ(0, alloc_scratch(ATTN_N_HEAD, ATTN_HEAD_DIM,
                                   FM_ATTN_DECODE_TARGET_WORKGROUPS, &m, &l,
                                   &acc));

    memcpy(single, run_decode(&f, seq_len, 1, seq_len, m, l, acc),
           out_n * sizeof(float));

    /* Several plans, so a bug that only shows at one split count cannot hide.
     * 6 workgroups gives 3 splits, deliberately not a divisor of anything
     * here; 32 is the shipped default. */
    const int workgroup_counts[] = { 4, 6, 16, 64 };
    for (int s = 0; s < 4; s++) {
        int split_len = 0;
        int n_splits = fm_attn_decode_plan(seq_len, ATTN_N_HEAD_KV,
                                           workgroup_counts[s], &split_len);
        ASSERT_TRUE(n_splits > 1);

        const float *got = run_decode(&f, seq_len, n_splits, split_len, m, l,
                                      acc);
        snprintf(label, sizeof(label), "decode depth %d (%d splits x %d)",
                 seq_len, n_splits, split_len);
        int ok = case_compare(label, f.want, got, out_n);
        int argmax_ok = case_argmax_matches(f.want, got, ATTN_N_HEAD,
                                            ATTN_HEAD_DIM);
        snprintf(label, sizeof(label), "  ^ vs 1-split GPU (secondary)");
        int agree = case_compare(label, single, got, out_n);

        if (!ok || !argmax_ok || !agree) {
            fm_buffer_free(m); fm_buffer_free(l); fm_buffer_free(acc);
            free(single); fixture_free(&f);
            ASSERT_TRUE(ok);
            ASSERT_TRUE(argmax_ok);
            ASSERT_TRUE(agree);
            return;
        }
    }

    fm_buffer_free(m);
    fm_buffer_free(l);
    fm_buffer_free(acc);
    free(single);
    fixture_free(&f);
}

/* A plan whose splits run past seq_len leaves some workgroups with no keys.
 * Those must contribute nothing: m = -inf makes their weight exp(-inf) = 0.
 * Getting this wrong yields NaN, not a small error. */
static void test_decode_tolerates_empty_splits(void) {
    const int seq_len = 40;    /* 3 splits of 16 covers 48 > 40 */
    const int n_splits = 8, split_len = 16;

    attn_fixture f;
    ASSERT_INT_EQ(0, fixture_open(&f, ATTN_N_HEAD, ATTN_N_HEAD_KV,
                                  ATTN_HEAD_DIM, seq_len + 3, 1, 0xBEEFu));
    reference_attention_heads(f.q_host, f.k_host, f.v_host, f.want,
                              ATTN_N_HEAD, f.kv_groups, ATTN_HEAD_DIM,
                              f.max_seq, seq_len, f.scale);

    fm_buffer *m = NULL, *l = NULL, *acc = NULL;
    ASSERT_INT_EQ(0, alloc_scratch(ATTN_N_HEAD, ATTN_HEAD_DIM, n_splits, &m,
                                   &l, &acc));

    const float *got = run_decode(&f, seq_len, n_splits, split_len, m, l, acc);
    size_t out_n = (size_t)ATTN_N_HEAD * ATTN_HEAD_DIM;

    int finite = 1;
    for (size_t i = 0; i < out_n; i++)
        if (!attn_bits_are_finite(got[i])) { finite = 0; break; }
    int ok = case_compare("decode depth 40 (5 live + 3 empty splits)", f.want,
                          got, out_n);

    fm_buffer_free(m);
    fm_buffer_free(l);
    fm_buffer_free(acc);
    fixture_free(&f);
    ASSERT_TRUE(finite);
    ASSERT_TRUE(ok);
}

/* CAUSALITY, decode. The query attends [0, seq_len); rows at and beyond
 * seq_len are future. Poison them with a value large enough to dominate any
 * softmax it entered and require the output to be BIT-identical. The second
 * half proves the poison is reachable at all — the same overwrite, moved
 * inside the attended range, must change the output. */
static void test_decode_ignores_keys_beyond_seq_len(void) {
    const int seq_len = 100, max_seq = 160;
    attn_fixture f;
    ASSERT_INT_EQ(0, fixture_open(&f, ATTN_N_HEAD, ATTN_N_HEAD_KV,
                                  ATTN_HEAD_DIM, max_seq, 1, 0x5150u));

    fm_buffer *m = NULL, *l = NULL, *acc = NULL;
    ASSERT_INT_EQ(0, alloc_scratch(ATTN_N_HEAD, ATTN_HEAD_DIM, 4, &m, &l, &acc));

    size_t out_n = (size_t)ATTN_N_HEAD * ATTN_HEAD_DIM;
    float *before = (float *)malloc(out_n * sizeof(float));
    ASSERT_NOT_NULL(before);
    memcpy(before, run_decode(&f, seq_len, 4, 32, m, l, acc),
           out_n * sizeof(float));

    /* Poison every position >= seq_len. */
    float *kc = (float *)fm_buffer_data(f.k_cache);
    float *vc = (float *)fm_buffer_data(f.v_cache);
    for (int h = 0; h < ATTN_N_HEAD_KV; h++)
        for (int p = seq_len; p < max_seq; p++)
            for (int d = 0; d < ATTN_HEAD_DIM; d++) {
                size_t i = ((size_t)h * max_seq + p) * ATTN_HEAD_DIM + d;
                kc[i] = 50.0f;
                vc[i] = 777.0f;
            }
    const float *after = run_decode(&f, seq_len, 4, 32, m, l, acc);
    int identical = memcmp(before, after, out_n * sizeof(float)) == 0;

    /* Now poison position seq_len-1, which IS attended: the output must move.
     * Without this the test above would also pass on a kernel that reads no
     * keys at all. */
    for (int h = 0; h < ATTN_N_HEAD_KV; h++)
        for (int d = 0; d < ATTN_HEAD_DIM; d++) {
            size_t i = ((size_t)h * max_seq + (seq_len - 1)) * ATTN_HEAD_DIM + d;
            kc[i] = 50.0f;
            vc[i] = 777.0f;
        }
    const float *poisoned = run_decode(&f, seq_len, 4, 32, m, l, acc);
    int moved = memcmp(before, poisoned, out_n * sizeof(float)) != 0;

    fm_buffer_free(m);
    fm_buffer_free(l);
    fm_buffer_free(acc);
    free(before);
    fixture_free(&f);
    ASSERT_TRUE(identical);
    ASSERT_TRUE(moved);
}

/* ================================================================
 * Prefill
 * ================================================================ */

/* Compares every query row of a prefill batch against the reference. Query
 * row t attends pos0+t+1 keys, so one call sweeps a run of depths and the
 * per-row causal bound is checked at every one of them. */
static void case_prefill_batch(const char *shape, int n_head, int n_head_kv,
                               int head_dim, int n, int pos0) {
    char label[96];
    attn_fixture f;
    ASSERT_INT_EQ(0, fixture_open(&f, n_head, n_head_kv, head_dim,
                                  pos0 + n + 2, n, 0x77u + (uint32_t)pos0));

    for (int t = 0; t < n; t++)
        reference_attention_heads(f.q_host + (size_t)t * n_head * head_dim,
                                  f.k_host, f.v_host,
                                  f.want + (size_t)t * n_head * head_dim,
                                  n_head, f.kv_groups, head_dim, f.max_seq,
                                  pos0 + t + 1, f.scale);

    fm_attn_flash_prefill(f.q, f.k_cache, f.v_cache, f.out, n, f.kv_groups,
                          n_head_kv, head_dim, f.max_seq, pos0, f.scale);
    fm_sync();

    const float *got = (const float *)fm_buffer_data(f.out);
    snprintf(label, sizeof(label), "%s prefill n=%d pos0=%d (depth %d..%d)",
             shape, n, pos0, pos0 + 1, pos0 + n);
    int ok = case_compare(label, f.want, got,
                          (size_t)n * n_head * head_dim);
    int argmax_ok = case_argmax_matches(f.want, got, n * n_head, head_dim);

    fixture_free(&f);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(argmax_ok);
}

/* Depths 1..17 in one batch: covers the empty cache (row 0 sees one key),
 * exactly one full tile (row 15 sees 16) and a one-key tail (row 16). */
static void test_prefill_matches_reference_at_tile_boundaries(void) {
    case_prefill_batch("qwen35", ATTN_N_HEAD, ATTN_N_HEAD_KV, ATTN_HEAD_DIM,
                       17, 0);
}

/* Deep context, few rows: the O(N^2) cost of a full 4096-row prefill is not
 * needed to exercise the depth — pos0 puts the rows at the depth directly. */
static void test_prefill_matches_reference_at_depth_513_and_4097(void) {
    case_prefill_batch("qwen35", ATTN_N_HEAD, ATTN_N_HEAD_KV, ATTN_HEAD_DIM,
                       4, 509);
    if (__ft_current_failed) return;
    case_prefill_batch("qwen35", ATTN_N_HEAD, ATTN_N_HEAD_KV, ATTN_HEAD_DIM,
                       4, 4093);
}

static void test_prefill_matches_reference_at_alt_shape(void) {
    case_prefill_batch("alt", ALT_N_HEAD, ALT_N_HEAD_KV, ALT_HEAD_DIM, 33, 5);
}

/* CAUSALITY, prefill. Row 0 attends exactly one key (position pos0); every
 * later position is future for it. Poisoning positions pos0+1.. must leave
 * row 0 bit-identical while moving the last row, which does attend them. */
static void test_prefill_row_zero_ignores_future_positions(void) {
    const int n = 24, pos0 = 0, head_dim = ATTN_HEAD_DIM;
    attn_fixture f;
    ASSERT_INT_EQ(0, fixture_open(&f, ATTN_N_HEAD, ATTN_N_HEAD_KV, head_dim,
                                  n + 2, n, 0xC0DEu));

    size_t row_n = (size_t)ATTN_N_HEAD * head_dim;
    size_t all_n = (size_t)n * row_n;
    float *before = (float *)malloc(all_n * sizeof(float));
    ASSERT_NOT_NULL(before);

    fm_attn_flash_prefill(f.q, f.k_cache, f.v_cache, f.out, n, f.kv_groups,
                          ATTN_N_HEAD_KV, head_dim, f.max_seq, pos0, f.scale);
    fm_sync();
    memcpy(before, fm_buffer_data(f.out), all_n * sizeof(float));

    float *kc = (float *)fm_buffer_data(f.k_cache);
    float *vc = (float *)fm_buffer_data(f.v_cache);
    for (int h = 0; h < ATTN_N_HEAD_KV; h++)
        for (int p = pos0 + 1; p < f.max_seq; p++)
            for (int d = 0; d < head_dim; d++) {
                size_t i = ((size_t)h * f.max_seq + p) * head_dim + d;
                kc[i] = 40.0f;
                vc[i] = -999.0f;
            }

    fm_attn_flash_prefill(f.q, f.k_cache, f.v_cache, f.out, n, f.kv_groups,
                          ATTN_N_HEAD_KV, head_dim, f.max_seq, pos0, f.scale);
    fm_sync();
    const float *after = (const float *)fm_buffer_data(f.out);

    int row0_identical = memcmp(before, after, row_n * sizeof(float)) == 0;
    int last_row_moved = memcmp(before + (size_t)(n - 1) * row_n,
                                after + (size_t)(n - 1) * row_n,
                                row_n * sizeof(float)) != 0;

    free(before);
    fixture_free(&f);
    ASSERT_TRUE(row0_identical);
    ASSERT_TRUE(last_row_moved);
}

/* The append and the read have to agree on the ring rule, or the whole
 * attention block is silently reading somebody else's rows. Append a batch
 * through fm_attn_kv_append, then prefill over it, and check against a
 * reference built from the SOURCE rows rather than from the cache. */
static void test_append_then_prefill_reads_the_appended_rows(void) {
    const int n = 20, head_dim = ALT_HEAD_DIM, n_head = ALT_N_HEAD;
    const int n_head_kv = ALT_N_HEAD_KV, max_seq = n + 4, pos0 = 0;
    const int kv_groups = n_head / n_head_kv;
    const float scale = 1.0f / sqrtf((float)head_dim);

    size_t cache_n = (size_t)n_head_kv * max_seq * head_dim;
    size_t new_n = (size_t)n * n_head_kv * head_dim;
    size_t q_n = (size_t)n * n_head * head_dim;

    fm_buffer *k_new = fm_buffer_alloc(new_n * sizeof(float));
    fm_buffer *v_new = fm_buffer_alloc(new_n * sizeof(float));
    fm_buffer *k_cache = fm_buffer_alloc(cache_n * sizeof(float));
    fm_buffer *v_cache = fm_buffer_alloc(cache_n * sizeof(float));
    fm_buffer *q = fm_buffer_alloc(q_n * sizeof(float));
    fm_buffer *out = fm_buffer_alloc(q_n * sizeof(float));
    ASSERT_NOT_NULL(k_new);
    ASSERT_NOT_NULL(out);

    attn_rng_seed(0x4242u);
    attn_fill_random((float *)fm_buffer_data(k_new), new_n);
    attn_fill_random((float *)fm_buffer_data(v_new), new_n);
    attn_fill_random((float *)fm_buffer_data(q), q_n);
    memset(fm_buffer_data(k_cache), 0, cache_n * sizeof(float));
    memset(fm_buffer_data(v_cache), 0, cache_n * sizeof(float));

    /* Reference cache, laid out by the ring rule stated here rather than
     * read back from the kernel's output. */
    float *k_ref = (float *)calloc(cache_n, sizeof(float));
    float *v_ref = (float *)calloc(cache_n, sizeof(float));
    ASSERT_NOT_NULL(k_ref);
    ASSERT_NOT_NULL(v_ref);
    const float *kn = (const float *)fm_buffer_data(k_new);
    const float *vn = (const float *)fm_buffer_data(v_new);
    for (int t = 0; t < n; t++)
        for (int h = 0; h < n_head_kv; h++)
            for (int d = 0; d < head_dim; d++) {
                size_t src = ((size_t)t * n_head_kv + h) * head_dim + d;
                size_t dst = ((size_t)h * max_seq + ((pos0 + t) % max_seq)) *
                             head_dim + d;
                k_ref[dst] = kn[src];
                v_ref[dst] = vn[src];
            }

    float *want = (float *)malloc(q_n * sizeof(float));
    ASSERT_NOT_NULL(want);
    const float *qh = (const float *)fm_buffer_data(q);
    for (int t = 0; t < n; t++)
        reference_attention_heads(qh + (size_t)t * n_head * head_dim, k_ref,
                                  v_ref, want + (size_t)t * n_head * head_dim,
                                  n_head, kv_groups, head_dim, max_seq,
                                  pos0 + t + 1, scale);

    fm_attn_kv_append(k_new, v_new, k_cache, v_cache, n, n_head_kv, head_dim,
                      max_seq, pos0);
    fm_attn_flash_prefill(q, k_cache, v_cache, out, n, kv_groups, n_head_kv,
                          head_dim, max_seq, pos0, scale);
    fm_sync();

    int ok = case_compare("append -> prefill round trip", want,
                          (const float *)fm_buffer_data(out), q_n);

    fm_buffer_free(k_new); fm_buffer_free(v_new);
    fm_buffer_free(k_cache); fm_buffer_free(v_cache);
    fm_buffer_free(q); fm_buffer_free(out);
    free(k_ref); free(v_ref); free(want);
    ASSERT_TRUE(ok);
}

/* Decode and prefill must be the same function of the same inputs: a decode
 * over seq_len keys is a prefill of one row at pos0 = seq_len-1. Both are
 * checked against the reference here, so this catches a divergence between
 * the two paths that neither absolute check alone would attribute. */
static void test_decode_and_prefill_agree_with_the_same_reference(void) {
    const int seq_len = 300;
    attn_fixture f;
    ASSERT_INT_EQ(0, fixture_open(&f, ATTN_N_HEAD, ATTN_N_HEAD_KV,
                                  ATTN_HEAD_DIM, seq_len + 2, 1, 0xABCDu));
    reference_attention_heads(f.q_host, f.k_host, f.v_host, f.want,
                              ATTN_N_HEAD, f.kv_groups, ATTN_HEAD_DIM,
                              f.max_seq, seq_len, f.scale);
    size_t out_n = (size_t)ATTN_N_HEAD * ATTN_HEAD_DIM;

    fm_buffer *m = NULL, *l = NULL, *acc = NULL;
    ASSERT_INT_EQ(0, alloc_scratch(ATTN_N_HEAD, ATTN_HEAD_DIM, 8, &m, &l, &acc));
    int split_len = 0;
    int n_splits = fm_attn_decode_plan(seq_len, ATTN_N_HEAD_KV, 16, &split_len);
    int dec_ok = case_compare("decode  depth 300", f.want,
                              run_decode(&f, seq_len, n_splits, split_len, m,
                                         l, acc), out_n);

    fm_attn_flash_prefill(f.q, f.k_cache, f.v_cache, f.out, 1, f.kv_groups,
                          ATTN_N_HEAD_KV, ATTN_HEAD_DIM, f.max_seq,
                          seq_len - 1, f.scale);
    fm_sync();
    int pre_ok = case_compare("prefill depth 300 (1 row)", f.want,
                              (const float *)fm_buffer_data(f.out), out_n);

    fm_buffer_free(m);
    fm_buffer_free(l);
    fm_buffer_free(acc);
    fixture_free(&f);
    ASSERT_TRUE(dec_ok);
    ASSERT_TRUE(pre_ok);
}

/* THE NORMALIZATION TRAP, and the sharpest assertion in this file.
 *
 * Make every cached V row identical. The output is then sum_j p_j * v[d] over
 * sum_j p_j = v[d] EXACTLY, whatever the scores are — so the expected value is
 * known in closed form instead of being a small weighted mean, and the check
 * is at full relative precision rather than at the RMS of a deep-context
 * average. Anything that scales the result, drops the online-softmax rescale,
 * or mismatches the numerator and denominator moves this immediately; the
 * random-input comparisons above cannot see a uniform 0.01% scale because
 * their outputs are two orders of magnitude smaller than their tolerance.
 *
 * Run against a MULTI-SPLIT decode and against prefill, since the split
 * combine is where the numerator and denominator can drift apart. */
static void test_constant_v_makes_the_output_that_v_exactly(void) {
    const int seq_len = 1000, max_seq = 1024;
    const int n = 12;
    attn_fixture f;
    ASSERT_INT_EQ(0, fixture_open(&f, ATTN_N_HEAD, ATTN_N_HEAD_KV,
                                  ATTN_HEAD_DIM, max_seq, n, 0x1F1Fu));

    /* v[kvh][any j][d] = 0.5 + d/256 + kvh — distinct per dim and per kv head,
     * so a head reading the wrong kv head is caught too. */
    float *vc = (float *)fm_buffer_data(f.v_cache);
    for (int h = 0; h < ATTN_N_HEAD_KV; h++)
        for (int p = 0; p < max_seq; p++)
            for (int d = 0; d < ATTN_HEAD_DIM; d++)
                vc[((size_t)h * max_seq + p) * ATTN_HEAD_DIM + d] =
                    0.5f + (float)d / 256.0f + (float)h;

    fm_buffer *m = NULL, *l = NULL, *acc = NULL;
    ASSERT_INT_EQ(0, alloc_scratch(ATTN_N_HEAD, ATTN_HEAD_DIM, 16, &m, &l, &acc));
    int split_len = 0;
    int n_splits = fm_attn_decode_plan(seq_len, ATTN_N_HEAD_KV, 32, &split_len);
    ASSERT_TRUE(n_splits > 1);

    const float *dec = run_decode(&f, seq_len, n_splits, split_len, m, l, acc);
    for (int h = 0; h < ATTN_N_HEAD; h++)
        for (int d = 0; d < ATTN_HEAD_DIM; d++) {
            float want = 0.5f + (float)d / 256.0f +
                         (float)(h / f.kv_groups);
            if (fabsf(want - dec[(size_t)h * ATTN_HEAD_DIM + d]) > 2e-6f)
                printf("      decode head %d dim %d\n", h, d);
            ASSERT_FLT_NEAR(want, dec[(size_t)h * ATTN_HEAD_DIM + d], 2e-6f);
        }

    fm_attn_flash_prefill(f.q, f.k_cache, f.v_cache, f.out, n, f.kv_groups,
                          ATTN_N_HEAD_KV, ATTN_HEAD_DIM, f.max_seq, 0, f.scale);
    fm_sync();
    const float *pre = (const float *)fm_buffer_data(f.out);
    for (int t = 0; t < n; t++)
        for (int h = 0; h < ATTN_N_HEAD; h++)
            for (int d = 0; d < ATTN_HEAD_DIM; d++) {
                size_t i = ((size_t)t * ATTN_N_HEAD + h) * ATTN_HEAD_DIM + d;
                float want = 0.5f + (float)d / 256.0f +
                             (float)(h / f.kv_groups);
                if (fabsf(want - pre[i]) > 2e-6f)
                    printf("      prefill row %d head %d dim %d\n", t, h, d);
                ASSERT_FLT_NEAR(want, pre[i], 2e-6f);
            }

    fm_buffer_free(m);
    fm_buffer_free(l);
    fm_buffer_free(acc);
    fixture_free(&f);
}

/* ================================================================ */

int main(void) {
    printf("\n  Metal Attention Kernel Tests\n");
    printf("  ============================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "  Metal unavailable — skipping.\n\n");
        return 77;
    }

    RUN_TEST(kv_append_lands_at_the_ring_position);
    RUN_TEST(kv_append_of_zero_rows_writes_nothing);
    RUN_TEST(decode_matches_reference_at_every_depth);
    RUN_TEST(decode_matches_reference_at_alt_shape);
    RUN_TEST(decode_split_k_matches_reference_and_single_split);
    RUN_TEST(decode_tolerates_empty_splits);
    RUN_TEST(decode_ignores_keys_beyond_seq_len);
    RUN_TEST(prefill_matches_reference_at_tile_boundaries);
    RUN_TEST(prefill_matches_reference_at_depth_513_and_4097);
    RUN_TEST(prefill_matches_reference_at_alt_shape);
    RUN_TEST(prefill_row_zero_ignores_future_positions);
    RUN_TEST(append_then_prefill_reads_the_appended_rows);
    RUN_TEST(decode_and_prefill_agree_with_the_same_reference);
    RUN_TEST(constant_v_makes_the_output_that_v_exactly);

    fm_free();

    printf("\n  Worst drift vs CPU reference: abs %.3e   abs/rms %.3e"
           "   (bars: abs %.0e, abs/rms %.0e)\n",
           attn_worst_abs, attn_worst_ratio,
           (double)ATTN_TOL_ABS, (double)ATTN_TOL_RMS);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
