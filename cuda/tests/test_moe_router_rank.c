/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_moe_router_rank.c
 * @brief Rank-select MoE router vs the block router: bit-exact, and faster (ONE K80 die).
 * @status complete
 * @issues none
 * @todo none
 *
 * fc_moe_router_topk_nx_ctx runs the rank-select router (no selection
 * rounds) when n_expert is a power of two <= 256. It must leave top_idx, top_wt and the in-place logits
 * (probabilities, -1 at the chosen experts) bit-identical to the block router
 * (fc_moe_router_topk_nx_block_ctx) on random logits with heavy ties, for
 * every eligible n_expert, several k and token counts; the single-token
 * fc_moe_router_topk_ctx likewise. Speed: the Qwen3.6 shape (256 experts,
 * k = 8, one token) must cost under ROUTER_BUDGET_US per stream launch; the
 * block router took ~55 us.
 *
 * Build: make USE_CUDA=1 test-moe-router-rank
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t rng_state = 2463534242u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* Router logits: coarse levels (ties after softmax) or fine spread. */
static void fill_logits(float *x, size_t n, int coarse)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t r = rng();
        x[i] = coarse ? (float)((int)(r % 9) - 4) * 0.5f
                      : ((float)(int)(r % 200001) - 100000.0f) * 1e-4f;
    }
}

typedef void (*router_fn)(fc_cuda_ctx *, fc_buffer *, fc_buffer *, fc_buffer *,
                          int, int, float, int);

/* Run one router over a copy of `x`; return logits/idx/wt on the host. */
static int run_router(router_fn fn, int single, const float *x, int n_expert,
                      int k, int ntok, float scale, float *lg_out, int *idx_out,
                      float *wt_out)
{
    size_t nl = (size_t)ntok * n_expert, nk = (size_t)ntok * k;
    fc_buffer *lg = fc_buffer_from_ptr(x, nl * sizeof(float));
    fc_buffer *ti = fc_buffer_alloc(nk * sizeof(int));
    fc_buffer *tw = fc_buffer_alloc(nk * sizeof(float));
    int ok = lg && ti && tw;
    if (ok) {
        if (single)
            fc_moe_router_topk_ctx(fc_default_ctx(), lg, ti, tw, n_expert, k, scale);
        else
            fn(fc_default_ctx(), lg, ti, tw, n_expert, k, scale, ntok);
        fc_buffer_to_cpu(lg, lg_out);
        fc_buffer_to_cpu(ti, idx_out);
        fc_buffer_to_cpu(tw, wt_out);
    }
    fc_buffer_free(lg);
    fc_buffer_free(ti);
    fc_buffer_free(tw);
    return ok;
}

/* Rank (or single-token) router vs block router on one shape. */
static int same_as_block(int n_expert, int k, int ntok, int coarse, int single)
{
    size_t nl = (size_t)ntok * n_expert, nk = (size_t)ntok * k;
    float *x = malloc(nl * sizeof(float));
    float *la = malloc(nl * sizeof(float)), *lb = malloc(nl * sizeof(float));
    int *ia = malloc(nk * sizeof(int)), *ib = malloc(nk * sizeof(int));
    float *wa = malloc(nk * sizeof(float)), *wb = malloc(nk * sizeof(float));
    int ok = x && la && lb && ia && ib && wa && wb;
    float scale = 2.5f;
    if (ok) {
        fill_logits(x, nl, coarse);
        ok = run_router(fc_moe_router_topk_nx_ctx, single, x, n_expert, k, ntok,
                        scale, la, ia, wa) &&
             run_router(fc_moe_router_topk_nx_block_ctx, 0, x, n_expert, k,
                        ntok, scale, lb, ib, wb);
        ok = ok && memcmp(ia, ib, nk * sizeof(int)) == 0 &&
             memcmp(wa, wb, nk * sizeof(float)) == 0 &&
             memcmp(la, lb, nl * sizeof(float)) == 0;
        if (!ok)
            printf("    n_expert=%d k=%d ntok=%d coarse=%d single=%d differ "
                   "(idx0 %d/%d wt0 %a/%a)\n", n_expert, k, ntok, coarse, single,
                   ia[0], ib[0], wa[0], wb[0]);
    }
    free(x); free(la); free(lb); free(ia); free(ib); free(wa); free(wb);
    return ok;
}

static void test_rank_matches_block_all_shapes(void)
{
    static const int ne[] = { 256, 128, 64, 32, 16, 8, 2, 1 };
    static const int ks[] = { 1, 2, 8, 32 };
    for (size_t a = 0; a < sizeof(ne) / sizeof(ne[0]); a++)
        for (size_t b = 0; b < sizeof(ks) / sizeof(ks[0]); b++) {
            if (ks[b] > ne[a])
                continue;
            for (int coarse = 0; coarse < 2; coarse++) {
                ASSERT_TRUE(same_as_block(ne[a], ks[b], 1, coarse, 0));
                ASSERT_TRUE(same_as_block(ne[a], ks[b], 7, coarse, 0));
                ASSERT_TRUE(same_as_block(ne[a], ks[b], 1, coarse, 1));
            }
        }
}

static void test_rank_matches_block_many_tokens(void)
{
    for (int rep = 0; rep < 8; rep++) {
        ASSERT_TRUE(same_as_block(256, 8, 512, rep & 1, 0));
    }
}

#define ROUTER_BUDGET_US 25.0   /* block router ~53 us per launch */

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

/* Mean GPU+launch time per stream launch at the Qwen3.6 shape. Each launch
 * gets its own fresh row: the router overwrites its logits in place, and a
 * re-routed row of probabilities is not a realistic input. */
static double router_us(router_fn fn, int iters)
{
    const int warm = 10, rows = warm + iters;
    float *x = malloc((size_t)rows * 256 * sizeof(float));
    double us = -1.0;
    if (!x)
        return us;
    fill_logits(x, (size_t)rows * 256, 0);
    fc_buffer *lg = fc_buffer_from_ptr(x, (size_t)rows * 256 * sizeof(float));
    fc_buffer *ti = fc_buffer_alloc(8 * sizeof(int));
    fc_buffer *tw = fc_buffer_alloc(8 * sizeof(float));
    fc_buffer **v = calloc((size_t)rows, sizeof(*v));
    int ok = lg && ti && tw && v;
    for (int r = 0; ok && r < rows; r++)
        ok = (v[r] = fc_buffer_offset_view(lg, (size_t)r * 256 * sizeof(float))) != NULL;
    if (ok) {
        for (int i = 0; i < warm; i++)
            fn(fc_default_ctx(), v[i], ti, tw, 256, 8, 1.0f, 1);
        fc_sync_ctx(fc_default_ctx());
        double t0 = now_us();
        for (int i = 0; i < iters; i++)
            fn(fc_default_ctx(), v[warm + i], ti, tw, 256, 8, 1.0f, 1);
        fc_sync_ctx(fc_default_ctx());
        us = (now_us() - t0) / iters;
    }
    for (int r = 0; v && r < rows; r++)
        fc_buffer_free_view(v[r]);
    free(v);
    fc_buffer_free(lg);
    fc_buffer_free(ti);
    fc_buffer_free(tw);
    free(x);
    return us;
}

static void test_qwen36_shape_under_budget(void)
{
    double rank = router_us(fc_moe_router_topk_nx_ctx, 400);
    double block = router_us(fc_moe_router_topk_nx_block_ctx, 400);
    printf("    router 256->8: rank %.1f us, block %.1f us per launch (budget %.0f)\n",
           rank, block, ROUTER_BUDGET_US);
    ASSERT_TRUE(rank > 0.0);
    ASSERT_TRUE(rank < ROUTER_BUDGET_US);
}

int main(void)
{
    printf("MoE Rank Router Tests\n");
    printf("=====================\n");
    if (fc_init() != 0) {
        printf("  [SKIP] no CUDA device\n");
        return 0;
    }
    RUN_TEST(rank_matches_block_all_shapes);
    RUN_TEST(rank_matches_block_many_tokens);
    RUN_TEST(qwen36_shape_under_budget);
    fc_free();
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
