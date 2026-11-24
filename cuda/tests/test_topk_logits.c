/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_topk_logits.c — device top-k: exactness, no-winner rows, speed (ONE K80 die)
 *
 * Exactness: every (n, k, nslots) shape below, on rows full of ties, NaN,
 * -inf and signed zeros, must return exactly the host reference (sort the
 * values above -inf by value desc, index asc; take k) and leave the logits
 * bit-identical. Speed: the Qwen3.6 LM-head shape (k = 40 over 248320) must
 * finish under TOPK_BUDGET_US per call; the k-round grid argmax this
 * replaced took 1841 us there.
 *
 * A row whose values are all NaN (or has fewer than k values above -inf)
 * leaves a round with no winner. The kernels used to write -inf through that
 * round's index, INT_MAX: 8 GiB past the row, an Xid 31 MMU fault that kills
 * the context (2026-11-14, MTP draft head). These tests feed such rows
 * and require a clean -1, the row left bit-identical, and the context still
 * usable for a normal call afterwards. One context for the whole run on
 * purpose: a fault in one test must fail the tests after it.
 *
 * NaN flag (fc_topk_logits_slots_nan_ctx): 1 exactly for the rows holding a
 * NaN on the fast path, -1 only when the tree ran, never stale from a previous
 * call.
 *
 * Build: make USE_CUDA=1 test-topk-logits
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TOPK_N 65812     /* the Qwen3.6 MTP draft vocabulary (65536 + 276 tail) */
#define TOPK_K 8         /* the MTP draft's k: k > 1 is the clobbering path */

/* Distinct finite logits whose top-k is known: value i * 1e-3 at index
 * (i * 7919) % n, so the winners sit at scattered positions. */
static void fill_finite_row(float *row, int n)
{
    for (int i = 0; i < n; i++)
        row[(int)(((long)i * 7919) % n)] = (float)i * 1e-3f;
}

/* A normal call on the same context returns the known top-k. Proves the
 * context survived whatever the test before it did. */
static int context_still_answers(void)
{
    float *row = malloc(TOPK_N * sizeof(float));
    int ids[TOPK_K];
    float vals[TOPK_K];
    int ok = 0;

    if (!row)
        return 0;
    fill_finite_row(row, TOPK_N);
    fc_buffer *buf = fc_buffer_from_ptr(row, TOPK_N * sizeof(float));
    if (buf && fc_topk_logits_ctx(fc_default_ctx(), buf, TOPK_N, TOPK_K, ids, vals) == 0) {
        ok = 1;
        for (int i = 0; i < TOPK_K; i++)
            if (ids[i] != (int)(((long)(TOPK_N - 1 - i) * 7919) % TOPK_N) ||
                vals[i] != (float)(TOPK_N - 1 - i) * 1e-3f)
                ok = 0;
    }
    fc_buffer_free(buf);
    free(row);
    return ok;
}

/* Upload `row`, run single-row top-k, and check: -1, row bytes unchanged. */
static int rejects_row_untouched(const float *row, int n, int k)
{
    float *back = malloc((size_t)n * sizeof(float));
    int ids[FC_TOPK_MAX];
    float vals[FC_TOPK_MAX];
    int ok = 0;

    fc_buffer *buf = fc_buffer_from_ptr(row, (size_t)n * sizeof(float));
    if (back && buf) {
        int rc = fc_topk_logits_ctx(fc_default_ctx(), buf, n, k, ids, vals);
        fc_buffer_to_cpu(buf, back);
        ok = rc == -1 && memcmp(back, row, (size_t)n * sizeof(float)) == 0;
    }
    fc_buffer_free(buf);
    free(back);
    return ok;
}

/* ---- exactness against a host reference ---- */

/* The reference works on bit patterns, not float compares: the build uses
 * -ffast-math, under which NaN / signed-zero compares are not trustworthy. */
typedef struct { uint32_t key; float v; int i; } topk_pair;

/* Order-preserving key of a value above -inf (+0 and -0 equal); 0 = not
 * above -inf (NaN or -inf). */
static uint32_t order_key(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u && u != 0x7f800000u)
        return 0;                        /* NaN or -inf */
    if (u == 0x80000000u)
        u = 0;                           /* -0 ties +0 */
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* value desc, index asc: the host's serial strict-'>' argmax, k times */
static int pair_cmp(const void *a, const void *b)
{
    const topk_pair *x = a, *y = b;
    if (x->key != y->key) return x->key > y->key ? -1 : 1;
    return (x->i > y->i) - (x->i < y->i);
}

/* Does the row hold a NaN (bitwise; fast-math compares are not trusted)? */
static int host_has_nan(const float *row, int n)
{
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &row[i], sizeof(u));
        if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu))
            return 1;
    }
    return 0;
}

/* Reference top-k of one row. Returns the number of values above -inf found
 * (the device must reject the row when that is < k). */
static int host_topk(const float *row, int n, int k, int *ids, float *vals)
{
    topk_pair *p = malloc((size_t)n * sizeof(*p));
    int m = 0;
    for (int i = 0; i < n; i++) {
        uint32_t key = order_key(row[i]);
        if (key) { p[m].key = key; p[m].v = row[i]; p[m].i = i; m++; }
    }
    qsort(p, (size_t)m, sizeof(*p), pair_cmp);
    for (int r = 0; r < k && r < m; r++) { ids[r] = p[r].i; vals[r] = p[r].v; }
    free(p);
    return m;
}

static uint32_t rng_state = 12345u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* Few distinct levels so ties are everywhere; sprinkle NaN, -inf, +-0. */
static void fill_tied_row(float *row, int n)
{
    for (int i = 0; i < n; i++) {
        uint32_t r = rng();
        switch (r % 64) {
        case 0:  row[i] = NAN; break;
        case 1:  row[i] = -INFINITY; break;
        case 2:  row[i] = 0.0f; break;
        case 3:  row[i] = -0.0f; break;
        default: row[i] = (float)((int)((r >> 8) % 97) - 48) * 0.25f; break;
        }
    }
}

/* One shape: random tied rows; device must match the reference pair for
 * pair (bitwise on values) or reject exactly the rows that are short. */
static int exact_on_shape(int n, int k, int nslots)
{
    size_t total = (size_t)nslots * n;
    float *rows = malloc(total * sizeof(float));
    float *back = malloc(total * sizeof(float));
    int *ids = malloc((size_t)nslots * k * sizeof(int));
    float *vals = malloc((size_t)nslots * k * sizeof(float));
    int *rids = malloc((size_t)k * sizeof(int));
    float *rvals = malloc((size_t)k * sizeof(float));
    int nan_flag[4];
    int ok = rows && back && ids && vals && rids && rvals && nslots <= 4;
    int short_row = 0;

    if (ok) {
        for (int u = 0; u < nslots; u++)
            fill_tied_row(rows + (size_t)u * n, n);
        fc_buffer *buf = fc_buffer_from_ptr(rows, total * sizeof(float));
        int rc = buf ? fc_topk_logits_slots_nan_ctx(fc_default_ctx(), buf, nslots,
                                                    n, k, ids, vals, nan_flag) : -2;
        if (buf)
            fc_buffer_to_cpu(buf, back);
        ok = buf && memcmp(back, rows, total * sizeof(float)) == 0;
        for (int u = 0; ok && u < nslots; u++) {
            int m = host_topk(rows + (size_t)u * n, n, k, rids, rvals);
            if (m < k) { short_row = 1; continue; }
            if (rc != 0) continue;
            if (nan_flag[u] != -1 && nan_flag[u] != host_has_nan(rows + (size_t)u * n, n)) {
                printf("    n=%d k=%d slots=%d row %d: nan flag %d\n", n, k, nslots, u,
                       nan_flag[u]);
                ok = 0;
                break;
            }
            for (int r = 0; r < k; r++) {
                if (ids[u * k + r] != rids[r] ||
                    memcmp(&vals[u * k + r], &rvals[r], sizeof(float)) != 0) {
                    printf("    n=%d k=%d slots=%d row %d rank %d: got (%d, %g) want (%d, %g)\n",
                           n, k, nslots, u, r, ids[u * k + r], vals[u * k + r],
                           rids[r], rvals[r]);
                    ok = 0;
                    break;
                }
            }
        }
        if (ok && rc != (short_row ? -1 : 0)) {
            printf("    n=%d k=%d slots=%d: rc %d, short row %d\n", n, k, nslots,
                   rc, short_row);
            ok = 0;
        }
        fc_buffer_free(buf);
    }
    free(rows); free(back); free(ids); free(vals); free(rids); free(rvals);
    return ok;
}

static void test_exact_vs_host_reference(void)
{
    static const int ns[] = { 1, 7, 255, 256, 257, 4097, 65812, 248320, 600001 };
    static const int ks[] = { 1, 8, 40, 256, 257, 512 };
    for (size_t a = 0; a < sizeof(ns) / sizeof(ns[0]); a++)
        for (size_t b = 0; b < sizeof(ks) / sizeof(ks[0]); b++) {
            if (ks[b] > ns[a])
                continue;
            ASSERT_TRUE(exact_on_shape(ns[a], ks[b], 1));
            ASSERT_TRUE(exact_on_shape(ns[a], ks[b], 3));
        }
}

/* Distinct values: every row has its full k, so this is all-success. */
static void test_exact_distinct_slots(void)
{
    const int nslots = 4, n = 248320, k = 40;
    float *rows = malloc((size_t)nslots * n * sizeof(float));
    int ids[4 * 40], rids[40];
    float vals[4 * 40], rvals[40];
    ASSERT_NOT_NULL(rows);
    for (int u = 0; u < nslots; u++)
        for (int i = 0; i < n; i++)
            rows[(size_t)u * n + i] = (float)(int)(rng() % 2000000) * 1e-4f;
    fc_buffer *buf = fc_buffer_from_ptr(rows, (size_t)nslots * n * sizeof(float));
    ASSERT_NOT_NULL(buf);
    int rc = fc_topk_logits_slots_ctx(fc_default_ctx(), buf, nslots, n, k, ids, vals);
    fc_buffer_free(buf);
    int same = rc == 0;
    for (int u = 0; same && u < nslots; u++) {
        host_topk(rows + (size_t)u * n, n, k, rids, rvals);
        same = memcmp(ids + u * k, rids, sizeof(rids)) == 0 &&
               memcmp(vals + u * k, rvals, sizeof(rvals)) == 0;
    }
    free(rows);
    ASSERT_TRUE(same);
}

/* All-equal row: the threshold admits ~k chunks' worth of candidates, far
 * past the candidate cap, so the call must take the tree fallback and still
 * return the first k indices. */
static void test_all_equal_row_overflow_exact(void)
{
    const int n = 248320, k = 40;
    float *row = malloc((size_t)n * sizeof(float));
    int ids[40];
    float vals[40];
    ASSERT_NOT_NULL(row);
    for (int i = 0; i < n; i++)
        row[i] = 1.5f;
    fc_buffer *buf = fc_buffer_from_ptr(row, (size_t)n * sizeof(float));
    free(row);
    ASSERT_NOT_NULL(buf);
    int rc = fc_topk_logits_slots_ctx(fc_default_ctx(), buf, 1, n, k, ids, vals);
    fc_buffer_free(buf);
    ASSERT_INT_EQ(rc, 0);
    for (int r = 0; r < k; r++) {
        ASSERT_INT_EQ(ids[r], r);
        ASSERT_TRUE(vals[r] == 1.5f);
    }
}

/* Distinct values (fast path): one NaN anywhere flags exactly its row, and
 * the next call on clean rows reports 0 (the flag is not left over). The
 * pairs are the same as the plain call's. */
static void test_nan_flag_exact_and_not_stale(void)
{
    const int nslots = 3, n = 248320, k = 257;
    float *rows = malloc((size_t)nslots * n * sizeof(float));
    int *ids = malloc((size_t)nslots * k * sizeof(int));
    int *ids2 = malloc((size_t)nslots * k * sizeof(int));
    float *vals = malloc((size_t)nslots * k * sizeof(float));
    float *vals2 = malloc((size_t)nslots * k * sizeof(float));
    int flag[3], rc;
    ASSERT_NOT_NULL(rows);
    ASSERT_NOT_NULL(ids);
    ASSERT_NOT_NULL(ids2);
    ASSERT_NOT_NULL(vals);
    ASSERT_NOT_NULL(vals2);
    for (size_t i = 0; i < (size_t)nslots * n; i++)
        rows[i] = (float)(int)(rng() % 2000000) * 1e-4f;
    rows[(size_t)1 * n + 123457] = NAN;          /* row 1 only, far from the top */
    fc_buffer *buf = fc_buffer_from_ptr(rows, (size_t)nslots * n * sizeof(float));
    ASSERT_NOT_NULL(buf);
    rc = fc_topk_logits_slots_nan_ctx(fc_default_ctx(), buf, nslots, n, k, ids, vals, flag);
    ASSERT_INT_EQ(rc, 0);
    ASSERT_INT_EQ(flag[0], 0);
    ASSERT_INT_EQ(flag[1], 1);
    ASSERT_INT_EQ(flag[2], 0);
    rc = fc_topk_logits_slots_ctx(fc_default_ctx(), buf, nslots, n, k, ids2, vals2);
    ASSERT_INT_EQ(rc, 0);
    ASSERT_TRUE(memcmp(ids, ids2, (size_t)nslots * k * sizeof(int)) == 0);
    ASSERT_TRUE(memcmp(vals, vals2, (size_t)nslots * k * sizeof(float)) == 0);
    fc_buffer_free(buf);
    rows[(size_t)1 * n + 123457] = 0.5f;
    buf = fc_buffer_from_ptr(rows, (size_t)nslots * n * sizeof(float));
    ASSERT_NOT_NULL(buf);
    flag[0] = flag[1] = flag[2] = 7;
    rc = fc_topk_logits_slots_nan_ctx(fc_default_ctx(), buf, nslots, n, k, ids, vals, flag);
    fc_buffer_free(buf);
    ASSERT_INT_EQ(rc, 0);
    ASSERT_INT_EQ(flag[0], 0);
    ASSERT_INT_EQ(flag[1], 0);
    ASSERT_INT_EQ(flag[2], 0);
    free(rows); free(ids); free(ids2); free(vals); free(vals2);
}

/* ---- speed ---- */

#define TOPK_BUDGET_US 400.0   /* old k-round path: 1841 us */

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

static int dbl_cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Median wall time per call (incl. launches, sync, D2H): the median keeps a
 * busy host's scheduling hiccups out of the number. */
static double median_call_us(int nslots, int k, int with_nan)
{
    enum { ITERS = 31 };
    const int n = 248320;
    float *rows = malloc((size_t)nslots * n * sizeof(float));
    int *ids = malloc((size_t)nslots * k * sizeof(int));
    float *vals = malloc((size_t)nslots * k * sizeof(float));
    int flag[4];
    double us[ITERS];
    int rc = rows && ids && vals && nslots <= 4 ? 0 : -1;
    fc_buffer *buf = NULL;
    if (rc == 0) {
        for (size_t i = 0; i < (size_t)nslots * n; i++)
            rows[i] = (float)(int)(rng() % 2000000) * 1e-4f;
        buf = fc_buffer_from_ptr(rows, (size_t)nslots * n * sizeof(float));
        rc = buf ? 0 : -1;
    }
    for (int i = 0; rc == 0 && i < 3 + ITERS; i++) {
        double t0 = now_us();
        rc |= with_nan ? fc_topk_logits_slots_nan_ctx(fc_default_ctx(), buf, nslots, n, k,
                                                      ids, vals, flag)
                       : fc_topk_logits_slots_ctx(fc_default_ctx(), buf, nslots, n, k,
                                                  ids, vals);
        if (i >= 3)
            us[i - 3] = now_us() - t0;
    }
    fc_buffer_free(buf);
    free(rows); free(ids); free(vals);
    if (rc != 0)
        return -1.0;
    qsort(us, ITERS, sizeof(us[0]), dbl_cmp);
    return us[ITERS / 2];
}

static void test_lm_head_shape_under_budget(void)
{
    double us = median_call_us(1, 40, 0);
    printf("    top-k k=40 over 248320: median %.1f us/call (budget %.0f)\n", us,
           TOPK_BUDGET_US);
    ASSERT_TRUE(us >= 0.0 && us < TOPK_BUDGET_US);
    /* The short logits row: k = 257 with the NaN check, 1 and 4 rows. */
    for (int ns = 1; ns <= 4; ns *= 2) {
        us = median_call_us(ns, 257, 1);
        printf("    top-k k=257 + NaN check, %d row(s) over 248320: median %.1f us/call\n",
               ns, us);
        ASSERT_TRUE(us >= 0.0 && us < 2.0 * TOPK_BUDGET_US * ns);
    }
}

static void test_finite_row_baseline(void)
{
    ASSERT_TRUE(context_still_answers());
}

static void test_all_nan_row_rejected(void)
{
    float *row = malloc(TOPK_N * sizeof(float));
    ASSERT_NOT_NULL(row);
    for (int i = 0; i < TOPK_N; i++)
        row[i] = NAN;
    int rejected = rejects_row_untouched(row, TOPK_N, TOPK_K);
    free(row);
    ASSERT_TRUE(rejected);
    ASSERT_TRUE(context_still_answers());
}

static void test_fewer_than_k_finite_rejected(void)
{
    float *row = malloc(TOPK_N * sizeof(float));
    ASSERT_NOT_NULL(row);
    for (int i = 0; i < TOPK_N; i++)
        row[i] = -INFINITY;
    row[11] = 1.0f;
    row[4000] = 2.0f;
    row[TOPK_N - 1] = 3.0f;   /* 3 winners for k = 8 */
    int rejected = rejects_row_untouched(row, TOPK_N, TOPK_K);
    free(row);
    ASSERT_TRUE(rejected);
    ASSERT_TRUE(context_still_answers());
}

static void test_slots_one_nan_row_rejected(void)
{
    const int nslots = 2;
    float *rows = malloc((size_t)nslots * TOPK_N * sizeof(float));
    float *back = malloc((size_t)nslots * TOPK_N * sizeof(float));
    int ids[2 * TOPK_K];
    float vals[2 * TOPK_K];
    ASSERT_NOT_NULL(rows);
    ASSERT_NOT_NULL(back);
    fill_finite_row(rows, TOPK_N);
    for (int i = 0; i < TOPK_N; i++)
        rows[TOPK_N + i] = NAN;

    fc_buffer *buf = fc_buffer_from_ptr(rows, (size_t)nslots * TOPK_N * sizeof(float));
    ASSERT_NOT_NULL(buf);
    int rc = fc_topk_logits_slots_ctx(fc_default_ctx(), buf, nslots, TOPK_N, TOPK_K,
                                      ids, vals);
    fc_buffer_to_cpu(buf, back);
    int same = memcmp(back, rows, (size_t)nslots * TOPK_N * sizeof(float)) == 0;
    fc_buffer_free(buf);
    free(rows);
    free(back);
    ASSERT_INT_EQ(rc, -1);
    ASSERT_TRUE(same);
    ASSERT_TRUE(context_still_answers());
}

int main(void)
{
    printf("Top-k Logits Tests\n");
    printf("==================\n");
    if (fc_init() != 0) {
        printf("  [SKIP] no CUDA device\n");
        return 0;
    }
    RUN_TEST(finite_row_baseline);
    RUN_TEST(all_nan_row_rejected);
    RUN_TEST(fewer_than_k_finite_rejected);
    RUN_TEST(slots_one_nan_row_rejected);
    RUN_TEST(exact_vs_host_reference);
    RUN_TEST(exact_distinct_slots);
    RUN_TEST(all_equal_row_overflow_exact);
    RUN_TEST(nan_flag_exact_and_not_stale);
    RUN_TEST(lm_head_shape_under_budget);
    fc_free();
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
