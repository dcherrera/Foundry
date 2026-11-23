/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file test_metal_moe.c
 * @brief Absolute-correctness tests for the Metal MoE router, the
 *   sort-by-expert prepass, the fused silu*mul and the shared-expert gate.
 *
 *   The bar is ABSOLUTE, never GPU-against-GPU.  Every assertion compares the
 *   kernel either to a Pantheon golden vector (produced by a third
 *   implementation from a fixed seed) or to the CPU reference written below,
 *   which is a term-for-term transcription of `qwen35moe_moe.c`'s router and a
 *   from-scratch CSR inversion of the routing.  A GPU-against-GPU arm would
 *   stay green through a uniform scale, which is exactly the break Phase 1
 *   proved slips past self-comparison.
 *
 *   The tie-break gets its own fixture.  A test whose logits happen to be
 *   distinct does not test a tie-break at all, so `router_tie_break_*` makes
 *   sixteen experts share the top probability bit-for-bit and requires the
 *   eight LOWEST of them, in ascending order.
 *
 *   The sort prepass is exercised at 512 tokens, not at 1: its entire value is
 *   at prefill, and a single-token run cannot expose a scan or a per-expert
 *   offset that is off by one.
 *
 * @status All tests pass on M4 / macOS 26.  Each was proven able to fail
 *   against a deliberately broken kernel before being trusted — a 0.01% scaled
 *   weight, a tie broken toward the higher index, an off-by-one per-expert
 *   count and a top-k returning the lowest probabilities instead of the
 *   highest were all induced and all caught.
 * @issues None
 * @todo None
 */

#include "lib/test.h"
/* The family header directly, not the fm_metal.h umbrella: the umbrella lists
 * one include per family and is owned by the integration step, so a test that
 * went through it could not build until that line lands. */
#include "metal/kernels/fm_kernel_moe.h"
#include "metal/tests/fm_golden_vectors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Fixtures
 * ================================================================ */

#define MOE_N_EXPERT     256
#define MOE_EXPERT_USED    8
#define MOE_N_EMBD      2048

/* Tokens for the sort prepass. The prepass exists for prefill, so it is
 * measured at a prefill batch rather than at the decode batch of 1. */
#define MOE_SORT_TOKENS  512

/* Experts sharing the top probability in the tie fixture. Deliberately more
 * than expert_used, so which ones are picked is decided by the tie-break. */
#define MOE_TIED_EXPERTS  16

static const int kTiedExperts[MOE_TIED_EXPERTS] = {
    5, 9, 17, 33, 41, 60, 77, 88, 101, 130, 150, 171, 190, 201, 222, 250
};

/* Deterministic PRNG so a failure is always reproducible. */
static uint32_t moe_rand_state = 0x12345678u;

static uint32_t moe_rand_u32(void)
{
    moe_rand_state ^= moe_rand_state << 13;
    moe_rand_state ^= moe_rand_state >> 17;
    moe_rand_state ^= moe_rand_state << 5;
    return moe_rand_state;
}

static float moe_rand_unit(void)
{
    return (float)(moe_rand_u32() >> 8) / (float)(1u << 24);
}

/**
 * Reads a PNGOLDEN1 i32 vector. fm_golden_vectors.h reads f32 only, and the
 * router's selected indices — the one thing that must match exactly — are i32.
 */
static int *moe_golden_load_i32(const char *name, size_t expect_count)
{
    char path[1152];
    char header[256];
    char magic[16] = {0}, vec_name[64] = {0}, dtype[8] = {0};
    unsigned long count = 0;
    int *values = NULL;
    FILE *f = NULL;

    snprintf(path, sizeof(path), "%s/%s", fm_golden_dir, name);
    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "  golden: cannot open %s\n", path);
        return NULL;
    }
    if (!fgets(header, sizeof(header), f)) {
        fprintf(stderr, "  golden: %s has no header line\n", path);
        goto cleanup;
    }
    if (sscanf(header, "%15s %63s %7s %lu", magic, vec_name, dtype, &count) != 4 ||
        strcmp(magic, "PNGOLDEN1") != 0 || strcmp(dtype, "i32") != 0 ||
        count != expect_count) {
        fprintf(stderr, "  golden: %s has a bad header: %s", path, header);
        goto cleanup;
    }
    values = (int *)malloc(expect_count * sizeof(int));
    if (values && fread(values, sizeof(int), expect_count, f) != expect_count) {
        fprintf(stderr, "  golden: %s payload is truncated\n", path);
        free(values);
        values = NULL;
    }

cleanup:
    fclose(f);
    return values;
}

/* ================================================================
 * CPU reference — transcribed from Pantheon's qwen35moe_moe.c
 * ================================================================ */

/**
 * Softmax over the router logits, top-k by probability, norm_topk_prob and the
 * routed-weight scale. Mirrors qwen35moe_router_softmax + qwen35moe_select_topk
 * + the renormalization block: selection scans experts in ascending order with
 * strict `>`, so a tie resolves to the lower index.
 */
static void cpu_router_topk(const float *logits, int n_expert, int k, float scale,
                            int *top_idx, float *top_wt)
{
    float *probs = (float *)malloc((size_t)n_expert * sizeof(float));
    float max_logit = logits[0];
    for (int e = 1; e < n_expert; e++)
        if (logits[e] > max_logit)
            max_logit = logits[e];

    float sum_exp = 0.0f;
    for (int e = 0; e < n_expert; e++) {
        probs[e] = expf(logits[e] - max_logit);
        sum_exp += probs[e];
    }
    float inv = 1.0f / sum_exp;
    for (int e = 0; e < n_expert; e++)
        probs[e] *= inv;

    float wt_sum = 0.0f;
    for (int s = 0; s < k; s++) {
        int best = -1;
        float best_val = -1.0f;
        for (int e = 0; e < n_expert; e++) {
            int skip = 0;
            for (int j = 0; j < s; j++)
                if (top_idx[j] == e) {
                    skip = 1;
                    break;
                }
            if (!skip && probs[e] > best_val) {
                best = e;
                best_val = probs[e];
            }
        }
        top_idx[s] = best;
        top_wt[s] = best_val;
        wt_sum += best_val;
    }

    float inv_wt = (wt_sum > 0.0f) ? (1.0f / wt_sum) : 0.0f;
    for (int s = 0; s < k; s++)
        top_wt[s] *= scale * inv_wt;
    free(probs);
}

/**
 * Inverts a routing table into the CSR form the prepass must produce: per
 * expert, the tokens that selected it in ascending token order, plus the slot
 * each used. Written from the definition, not from the kernel's tiled scan.
 */
static void cpu_expert_sort(const int *top_idx, int n_tokens, int n_expert, int k,
                            int *counts, int *offsets, int *sorted_token,
                            int *sorted_slot, int *sorted_expert)
{
    for (int e = 0; e < n_expert; e++)
        counts[e] = 0;
    for (int t = 0; t < n_tokens; t++)
        for (int s = 0; s < k; s++) {
            int e = top_idx[t * k + s];
            if (e >= 0 && e < n_expert)
                counts[e]++;
        }

    offsets[0] = 0;
    for (int e = 0; e < n_expert; e++)
        offsets[e + 1] = offsets[e] + counts[e];

    for (int e = 0; e < n_expert; e++) {
        int pos = offsets[e];
        for (int t = 0; t < n_tokens; t++)
            for (int s = 0; s < k; s++)
                if (top_idx[t * k + s] == e) {
                    sorted_token[pos] = t;
                    sorted_slot[pos] = s;
                    sorted_expert[pos] = e;
                    pos++;
                }
    }
}

/* ================================================================
 * Router
 * ================================================================ */

/* Runs the router over one token's logits and hands back the device results. */
static int run_router(const float *logits, int n_tokens, int n_expert, int k,
                      float scale, int *out_idx, float *out_wt)
{
    fm_buffer *lg = fm_buffer_alloc((size_t)n_tokens * n_expert * sizeof(float));
    fm_buffer *ti = fm_buffer_alloc((size_t)n_tokens * k * sizeof(int));
    fm_buffer *tw = fm_buffer_alloc((size_t)n_tokens * k * sizeof(float));
    if (!lg || !ti || !tw)
        return -1;

    memcpy(fm_buffer_data(lg), logits,
           (size_t)n_tokens * n_expert * sizeof(float));
    memset(fm_buffer_data(ti), 0xFF, (size_t)n_tokens * k * sizeof(int));
    memset(fm_buffer_data(tw), 0, (size_t)n_tokens * k * sizeof(float));

    fm_moe_router_topk(lg, ti, tw, n_tokens, n_expert, k, scale);
    fm_sync();

    memcpy(out_idx, fm_buffer_data(ti), (size_t)n_tokens * k * sizeof(int));
    memcpy(out_wt, fm_buffer_data(tw), (size_t)n_tokens * k * sizeof(float));

    fm_buffer_free(lg);
    fm_buffer_free(ti);
    fm_buffer_free(tw);
    return 0;
}

static void test_router_topk_matches_golden(void)
{
    float *logits = fm_golden_load_f32("moe_router_logits.bin", MOE_N_EXPERT);
    float *golden_wt = fm_golden_load_f32("moe_topk_weights.bin", MOE_EXPERT_USED);
    int *golden_idx = moe_golden_load_i32("moe_topk_indices.bin", MOE_EXPERT_USED);
    ASSERT_NOT_NULL(logits);
    ASSERT_NOT_NULL(golden_wt);
    ASSERT_NOT_NULL(golden_idx);

    int gpu_idx[MOE_EXPERT_USED];
    float gpu_wt[MOE_EXPERT_USED];
    ASSERT_INT_EQ(0, run_router(logits, 1, MOE_N_EXPERT, MOE_EXPERT_USED, 1.0f,
                                gpu_idx, gpu_wt));

    float weight_sum = 0.0f;
    for (int s = 0; s < MOE_EXPERT_USED; s++) {
        ASSERT_INT_EQ(golden_idx[s], gpu_idx[s]);
        ASSERT_FLT_NEAR(golden_wt[s], gpu_wt[s], 2e-6f);
        weight_sum += gpu_wt[s];
    }
    ASSERT_FLT_NEAR(1.0f, weight_sum, 2e-6f);

    free(logits);
    free(golden_wt);
    free(golden_idx);
}

static void test_router_tie_break_prefers_lower_index(void)
{
    /* Sixteen experts share the maximum logit exactly, so their softmax
     * probabilities are bit-identical and the SELECTION ORDER is decided
     * entirely by the tie-break rule. */
    float logits[MOE_N_EXPERT];
    for (int e = 0; e < MOE_N_EXPERT; e++)
        logits[e] = 0.0f;
    for (int i = 0; i < MOE_TIED_EXPERTS; i++)
        logits[kTiedExperts[i]] = 1.0f;

    int gpu_idx[MOE_EXPERT_USED];
    float gpu_wt[MOE_EXPERT_USED];
    ASSERT_INT_EQ(0, run_router(logits, 1, MOE_N_EXPERT, MOE_EXPERT_USED, 1.0f,
                                gpu_idx, gpu_wt));

    /* Ties resolve to the LOWER index, so the eight lowest of the tied set are
     * selected, in ascending order. A rule that preferred the higher index
     * would return kTiedExperts[15..8] descending. */
    for (int s = 0; s < MOE_EXPERT_USED; s++) {
        ASSERT_INT_EQ(kTiedExperts[s], gpu_idx[s]);
        ASSERT_FLT_NEAR(1.0f / (float)MOE_EXPERT_USED, gpu_wt[s], 1e-6f);
    }
}

static void test_router_tie_break_matches_cpu_selection_sort(void)
{
    /* Same fixture through the CPU selection sort, which resolves ties by
     * first-seen-highest. The two must agree index for index. */
    float logits[MOE_N_EXPERT];
    for (int e = 0; e < MOE_N_EXPERT; e++)
        logits[e] = 0.0f;
    for (int i = 0; i < MOE_TIED_EXPERTS; i++)
        logits[kTiedExperts[i]] = 1.0f;

    int cpu_idx[MOE_EXPERT_USED];
    float cpu_wt[MOE_EXPERT_USED];
    cpu_router_topk(logits, MOE_N_EXPERT, MOE_EXPERT_USED, 1.0f, cpu_idx, cpu_wt);

    int gpu_idx[MOE_EXPERT_USED];
    float gpu_wt[MOE_EXPERT_USED];
    ASSERT_INT_EQ(0, run_router(logits, 1, MOE_N_EXPERT, MOE_EXPERT_USED, 1.0f,
                                gpu_idx, gpu_wt));

    for (int s = 0; s < MOE_EXPERT_USED; s++) {
        ASSERT_INT_EQ(cpu_idx[s], gpu_idx[s]);
        ASSERT_FLT_NEAR(cpu_wt[s], gpu_wt[s], 1e-6f);
    }
}

static void test_router_applies_expert_weights_scale(void)
{
    float *logits = fm_golden_load_f32("moe_router_logits.bin", MOE_N_EXPERT);
    ASSERT_NOT_NULL(logits);

    const float scale = 2.5f;
    int cpu_idx[MOE_EXPERT_USED];
    float cpu_wt[MOE_EXPERT_USED];
    cpu_router_topk(logits, MOE_N_EXPERT, MOE_EXPERT_USED, scale, cpu_idx, cpu_wt);

    int gpu_idx[MOE_EXPERT_USED];
    float gpu_wt[MOE_EXPERT_USED];
    ASSERT_INT_EQ(0, run_router(logits, 1, MOE_N_EXPERT, MOE_EXPERT_USED, scale,
                                gpu_idx, gpu_wt));

    float weight_sum = 0.0f;
    for (int s = 0; s < MOE_EXPERT_USED; s++) {
        ASSERT_INT_EQ(cpu_idx[s], gpu_idx[s]);
        ASSERT_FLT_NEAR(cpu_wt[s], gpu_wt[s], 2e-6f);
        weight_sum += gpu_wt[s];
    }
    ASSERT_FLT_NEAR(scale, weight_sum, 5e-6f);
    free(logits);
}

/* Per-token logits that are distinct by construction (a permutation on a
 * 0.03-wide grid), so index selection cannot be flipped by the last ulp of a
 * differently-associated softmax while still giving every token its own
 * expert set. */
static void fill_permuted_logits(float *logits, int n_tokens, int n_expert)
{
    for (int t = 0; t < n_tokens; t++) {
        int *perm = (int *)malloc((size_t)n_expert * sizeof(int));
        for (int e = 0; e < n_expert; e++)
            perm[e] = e;
        for (int e = n_expert - 1; e > 0; e--) {
            int j = (int)(moe_rand_u32() % (uint32_t)(e + 1));
            int tmp = perm[e];
            perm[e] = perm[j];
            perm[j] = tmp;
        }
        for (int e = 0; e < n_expert; e++)
            logits[t * n_expert + e] = (float)perm[e] * 0.03f - 4.0f;
        free(perm);
    }
}

static void test_router_batch_matches_cpu_reference(void)
{
    const int n_tokens = 64;
    float *logits = (float *)malloc((size_t)n_tokens * MOE_N_EXPERT * sizeof(float));
    int *gpu_idx = (int *)malloc((size_t)n_tokens * MOE_EXPERT_USED * sizeof(int));
    float *gpu_wt = (float *)malloc((size_t)n_tokens * MOE_EXPERT_USED * sizeof(float));
    ASSERT_NOT_NULL(logits);
    ASSERT_NOT_NULL(gpu_idx);
    ASSERT_NOT_NULL(gpu_wt);

    moe_rand_state = 0x2468ACE0u;
    fill_permuted_logits(logits, n_tokens, MOE_N_EXPERT);
    ASSERT_INT_EQ(0, run_router(logits, n_tokens, MOE_N_EXPERT, MOE_EXPERT_USED,
                                1.0f, gpu_idx, gpu_wt));

    for (int t = 0; t < n_tokens; t++) {
        int cpu_idx[MOE_EXPERT_USED];
        float cpu_wt[MOE_EXPERT_USED];
        cpu_router_topk(logits + (size_t)t * MOE_N_EXPERT, MOE_N_EXPERT,
                        MOE_EXPERT_USED, 1.0f, cpu_idx, cpu_wt);
        for (int s = 0; s < MOE_EXPERT_USED; s++) {
            ASSERT_INT_EQ(cpu_idx[s], gpu_idx[t * MOE_EXPERT_USED + s]);
            ASSERT_FLT_NEAR(cpu_wt[s], gpu_wt[t * MOE_EXPERT_USED + s], 2e-6f);
        }
    }
    free(logits);
    free(gpu_idx);
    free(gpu_wt);
}

/* ================================================================
 * Sort-by-expert prepass
 * ================================================================ */

/* Routing for the sort tests: real router output over per-token permuted
 * logits, so the token->expert map is as irregular as a real batch's. */
static int build_routing(int n_tokens, int *top_idx)
{
    float *logits = (float *)malloc((size_t)n_tokens * MOE_N_EXPERT * sizeof(float));
    float *wt = (float *)malloc((size_t)n_tokens * MOE_EXPERT_USED * sizeof(float));
    if (!logits || !wt)
        return -1;

    moe_rand_state = 0x0BADC0DEu;
    fill_permuted_logits(logits, n_tokens, MOE_N_EXPERT);
    int rc = run_router(logits, n_tokens, MOE_N_EXPERT, MOE_EXPERT_USED, 1.0f,
                        top_idx, wt);
    free(logits);
    free(wt);
    return rc;
}

/* Runs the prepass and copies its five outputs back. */
static int run_expert_sort(const int *top_idx, int n_tokens, int *counts,
                           int *offsets, int *sorted_token, int *sorted_slot,
                           int *sorted_expert)
{
    const size_t requests = (size_t)n_tokens * MOE_EXPERT_USED;
    fm_buffer *ti = fm_buffer_alloc(requests * sizeof(int));
    fm_buffer *bc = fm_buffer_alloc(MOE_N_EXPERT * sizeof(int));
    fm_buffer *bo = fm_buffer_alloc((MOE_N_EXPERT + 1) * sizeof(int));
    fm_buffer *bt = fm_buffer_alloc(requests * sizeof(int));
    fm_buffer *bs = fm_buffer_alloc(requests * sizeof(int));
    fm_buffer *be = fm_buffer_alloc(requests * sizeof(int));
    if (!ti || !bc || !bo || !bt || !bs || !be)
        return -1;

    memcpy(fm_buffer_data(ti), top_idx, requests * sizeof(int));
    memset(fm_buffer_data(bc), 0xFF, MOE_N_EXPERT * sizeof(int));
    memset(fm_buffer_data(bo), 0xFF, (MOE_N_EXPERT + 1) * sizeof(int));
    memset(fm_buffer_data(bt), 0xFF, requests * sizeof(int));
    memset(fm_buffer_data(bs), 0xFF, requests * sizeof(int));
    memset(fm_buffer_data(be), 0xFF, requests * sizeof(int));

    fm_moe_expert_sort(ti, bc, bo, bt, bs, be, n_tokens, MOE_N_EXPERT,
                       MOE_EXPERT_USED);
    fm_sync();

    memcpy(counts, fm_buffer_data(bc), MOE_N_EXPERT * sizeof(int));
    memcpy(offsets, fm_buffer_data(bo), (MOE_N_EXPERT + 1) * sizeof(int));
    memcpy(sorted_token, fm_buffer_data(bt), requests * sizeof(int));
    memcpy(sorted_slot, fm_buffer_data(bs), requests * sizeof(int));
    memcpy(sorted_expert, fm_buffer_data(be), requests * sizeof(int));

    fm_buffer_free(ti);
    fm_buffer_free(bc);
    fm_buffer_free(bo);
    fm_buffer_free(bt);
    fm_buffer_free(bs);
    fm_buffer_free(be);
    return 0;
}

static void test_expert_sort_matches_cpu_reference_at_prefill(void)
{
    const int n_tokens = MOE_SORT_TOKENS;
    const size_t requests = (size_t)n_tokens * MOE_EXPERT_USED;

    int *top_idx = (int *)malloc(requests * sizeof(int));
    int *counts = (int *)malloc(MOE_N_EXPERT * sizeof(int));
    int *offsets = (int *)malloc((MOE_N_EXPERT + 1) * sizeof(int));
    int *sorted_token = (int *)malloc(requests * sizeof(int));
    int *sorted_slot = (int *)malloc(requests * sizeof(int));
    int *sorted_expert = (int *)malloc(requests * sizeof(int));
    int *ref_counts = (int *)malloc(MOE_N_EXPERT * sizeof(int));
    int *ref_offsets = (int *)malloc((MOE_N_EXPERT + 1) * sizeof(int));
    int *ref_token = (int *)malloc(requests * sizeof(int));
    int *ref_slot = (int *)malloc(requests * sizeof(int));
    int *ref_expert = (int *)malloc(requests * sizeof(int));
    ASSERT_NOT_NULL(top_idx);
    ASSERT_NOT_NULL(ref_expert);

    ASSERT_INT_EQ(0, build_routing(n_tokens, top_idx));
    ASSERT_INT_EQ(0, run_expert_sort(top_idx, n_tokens, counts, offsets,
                                     sorted_token, sorted_slot, sorted_expert));
    cpu_expert_sort(top_idx, n_tokens, MOE_N_EXPERT, MOE_EXPERT_USED, ref_counts,
                    ref_offsets, ref_token, ref_slot, ref_expert);

    for (int e = 0; e < MOE_N_EXPERT; e++)
        ASSERT_INT_EQ(ref_counts[e], counts[e]);
    for (int e = 0; e <= MOE_N_EXPERT; e++)
        ASSERT_INT_EQ(ref_offsets[e], offsets[e]);
    ASSERT_INT_EQ((int)requests, offsets[MOE_N_EXPERT]);
    for (size_t r = 0; r < requests; r++) {
        ASSERT_INT_EQ(ref_token[r], sorted_token[r]);
        ASSERT_INT_EQ(ref_slot[r], sorted_slot[r]);
        ASSERT_INT_EQ(ref_expert[r], sorted_expert[r]);
    }

    free(top_idx);
    free(counts);
    free(offsets);
    free(sorted_token);
    free(sorted_slot);
    free(sorted_expert);
    free(ref_counts);
    free(ref_offsets);
    free(ref_token);
    free(ref_slot);
    free(ref_expert);
}

static void test_expert_sort_is_a_permutation_of_the_routing(void)
{
    /* Independent of the CPU reference: every (token, slot) request must appear
     * exactly once, at a position inside its own expert's CSR range, with the
     * expert the router actually chose. This catches a scan that drops or
     * duplicates an entry even if the reference made the same mistake. */
    const int n_tokens = MOE_SORT_TOKENS;
    const size_t requests = (size_t)n_tokens * MOE_EXPERT_USED;

    int *top_idx = (int *)malloc(requests * sizeof(int));
    int *counts = (int *)malloc(MOE_N_EXPERT * sizeof(int));
    int *offsets = (int *)malloc((MOE_N_EXPERT + 1) * sizeof(int));
    int *sorted_token = (int *)malloc(requests * sizeof(int));
    int *sorted_slot = (int *)malloc(requests * sizeof(int));
    int *sorted_expert = (int *)malloc(requests * sizeof(int));
    char *seen = (char *)calloc(requests, 1);
    ASSERT_NOT_NULL(top_idx);
    ASSERT_NOT_NULL(seen);

    ASSERT_INT_EQ(0, build_routing(n_tokens, top_idx));
    ASSERT_INT_EQ(0, run_expert_sort(top_idx, n_tokens, counts, offsets,
                                     sorted_token, sorted_slot, sorted_expert));

    for (int e = 0; e < MOE_N_EXPERT; e++) {
        ASSERT_INT_EQ(counts[e], offsets[e + 1] - offsets[e]);
        for (int r = offsets[e]; r < offsets[e + 1]; r++) {
            ASSERT_INT_EQ(e, sorted_expert[r]);
            int t = sorted_token[r];
            int s = sorted_slot[r];
            ASSERT_TRUE(t >= 0 && t < n_tokens);
            ASSERT_TRUE(s >= 0 && s < MOE_EXPERT_USED);
            ASSERT_INT_EQ(e, top_idx[t * MOE_EXPERT_USED + s]);
            ASSERT_INT_EQ(0, seen[(size_t)t * MOE_EXPERT_USED + s]);
            seen[(size_t)t * MOE_EXPERT_USED + s] = 1;
            /* Ascending token order within an expert is what makes the output
             * deterministic; an atomic-ordered scatter would break here. */
            if (r > offsets[e])
                ASSERT_TRUE(sorted_token[r] > sorted_token[r - 1]);
        }
    }
    for (size_t r = 0; r < requests; r++)
        ASSERT_INT_EQ(1, seen[r]);

    free(top_idx);
    free(counts);
    free(offsets);
    free(sorted_token);
    free(sorted_slot);
    free(sorted_expert);
    free(seen);
}

static void test_expert_sort_single_token(void)
{
    int top_idx[MOE_EXPERT_USED];
    int counts[MOE_N_EXPERT];
    int offsets[MOE_N_EXPERT + 1];
    int sorted_token[MOE_EXPERT_USED];
    int sorted_slot[MOE_EXPERT_USED];
    int sorted_expert[MOE_EXPERT_USED];

    ASSERT_INT_EQ(0, build_routing(1, top_idx));
    ASSERT_INT_EQ(0, run_expert_sort(top_idx, 1, counts, offsets, sorted_token,
                                     sorted_slot, sorted_expert));

    int total = 0;
    for (int e = 0; e < MOE_N_EXPERT; e++) {
        int expected = 0;
        for (int s = 0; s < MOE_EXPERT_USED; s++)
            if (top_idx[s] == e)
                expected = 1;
        ASSERT_INT_EQ(expected, counts[e]);
        total += counts[e];
    }
    ASSERT_INT_EQ(MOE_EXPERT_USED, total);
    for (int r = 0; r < MOE_EXPERT_USED; r++) {
        ASSERT_INT_EQ(0, sorted_token[r]);
        ASSERT_INT_EQ(sorted_expert[r], top_idx[sorted_slot[r]]);
    }
}

/* ================================================================
 * Shared expert
 * ================================================================ */

static void test_silu_mul_matches_cpu_reference(void)
{
    const int n = 4096;
    float *a = (float *)malloc((size_t)n * sizeof(float));
    float *b = (float *)malloc((size_t)n * sizeof(float));
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);

    moe_rand_state = 0x51A7ABCDu;
    for (int i = 0; i < n; i++) {
        a[i] = moe_rand_unit() * 8.0f - 4.0f;
        b[i] = moe_rand_unit() * 4.0f - 2.0f;
    }

    fm_buffer *ba = fm_buffer_alloc((size_t)n * sizeof(float));
    fm_buffer *bb = fm_buffer_alloc((size_t)n * sizeof(float));
    fm_buffer *bo = fm_buffer_alloc((size_t)n * sizeof(float));
    ASSERT_NOT_NULL(ba);
    ASSERT_NOT_NULL(bb);
    ASSERT_NOT_NULL(bo);
    memcpy(fm_buffer_data(ba), a, (size_t)n * sizeof(float));
    memcpy(fm_buffer_data(bb), b, (size_t)n * sizeof(float));
    memset(fm_buffer_data(bo), 0, (size_t)n * sizeof(float));

    fm_moe_silu_mul(ba, bb, bo, n);
    fm_sync();

    const float *got = (const float *)fm_buffer_data(bo);
    for (int i = 0; i < n; i++) {
        float expected = (a[i] / (1.0f + expf(-a[i]))) * b[i];
        ASSERT_FLT_NEAR(expected, got[i], 1e-5f);
    }

    fm_buffer_free(ba);
    fm_buffer_free(bb);
    fm_buffer_free(bo);
    free(a);
    free(b);
}

static void test_shexp_dot_gate_matches_golden_scalar(void)
{
    float *gate_w = fm_golden_load_f32("moe_shared_gate_weight.bin", MOE_N_EMBD);
    float *x = fm_golden_load_f32("moe_shared_gate_input.bin", MOE_N_EMBD);
    float *golden = fm_golden_load_f32("moe_shared_gate_scalar.bin", 1);
    ASSERT_NOT_NULL(gate_w);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(golden);

    fm_buffer *bw = fm_buffer_alloc(MOE_N_EMBD * sizeof(float));
    fm_buffer *bx = fm_buffer_alloc(MOE_N_EMBD * sizeof(float));
    fm_buffer *bs = fm_buffer_alloc(MOE_N_EMBD * sizeof(float));
    ASSERT_NOT_NULL(bw);
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bs);
    memcpy(fm_buffer_data(bw), gate_w, MOE_N_EMBD * sizeof(float));
    memcpy(fm_buffer_data(bx), x, MOE_N_EMBD * sizeof(float));

    /* Seeding the branch with 1.0 makes the in-place kernel report the scalar
     * gate itself, which is what the golden vector holds. */
    float *shexp = (float *)fm_buffer_data(bs);
    for (int i = 0; i < MOE_N_EMBD; i++)
        shexp[i] = 1.0f;

    fm_moe_shexp_dot_gate(bw, bx, bs, 1, MOE_N_EMBD);
    fm_sync();

    for (int i = 0; i < MOE_N_EMBD; i++)
        ASSERT_FLT_NEAR(golden[0], shexp[i], 5e-7f);

    fm_buffer_free(bw);
    fm_buffer_free(bx);
    fm_buffer_free(bs);
    free(gate_w);
    free(x);
    free(golden);
}

static void test_shexp_dot_gate_scales_the_branch_per_token(void)
{
    const int n_tokens = 5;
    float *gate_w = fm_golden_load_f32("moe_shared_gate_weight.bin", MOE_N_EMBD);
    float *x0 = fm_golden_load_f32("moe_shared_gate_input.bin", MOE_N_EMBD);
    ASSERT_NOT_NULL(gate_w);
    ASSERT_NOT_NULL(x0);

    fm_buffer *bw = fm_buffer_alloc(MOE_N_EMBD * sizeof(float));
    fm_buffer *bx = fm_buffer_alloc((size_t)n_tokens * MOE_N_EMBD * sizeof(float));
    fm_buffer *bs = fm_buffer_alloc((size_t)n_tokens * MOE_N_EMBD * sizeof(float));
    ASSERT_NOT_NULL(bw);
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bs);
    memcpy(fm_buffer_data(bw), gate_w, MOE_N_EMBD * sizeof(float));

    /* Each token scales the golden input differently, so every token must end
     * up with its own gate — a kernel that reduced token 0 for everyone passes
     * the single-token test and fails here. */
    float *x = (float *)fm_buffer_data(bx);
    float *shexp = (float *)fm_buffer_data(bs);
    for (int t = 0; t < n_tokens; t++)
        for (int i = 0; i < MOE_N_EMBD; i++) {
            x[t * MOE_N_EMBD + i] = x0[i] * (0.5f + 0.25f * (float)t);
            shexp[t * MOE_N_EMBD + i] = 2.0f;
        }

    fm_moe_shexp_dot_gate(bw, bx, bs, n_tokens, MOE_N_EMBD);
    fm_sync();

    for (int t = 0; t < n_tokens; t++) {
        double dot = 0.0;
        for (int i = 0; i < MOE_N_EMBD; i++)
            dot += (double)gate_w[i] * (double)x[t * MOE_N_EMBD + i];
        float expected = 2.0f * (float)(1.0 / (1.0 + exp(-dot)));
        for (int i = 0; i < MOE_N_EMBD; i += 257)
            ASSERT_FLT_NEAR(expected, shexp[t * MOE_N_EMBD + i], 1e-5f);
    }

    fm_buffer_free(bw);
    fm_buffer_free(bx);
    fm_buffer_free(bs);
    free(gate_w);
    free(x0);
}

/* ================================================================
 * main
 * ================================================================ */

int main(int argc, char **argv)
{
    (void)argc;
    printf("\n  Metal MoE — Router, Sort Prepass, Shared Expert\n");
    printf("  ==============================================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "  Metal unavailable — no device to test against.\n");
        return 77;
    }

    fm_golden_locate(argv[0]);
    float *probe = fm_golden_load_f32("moe_router_logits.bin", MOE_N_EXPERT);
    if (!probe) {
        fprintf(stderr, "  Golden vectors missing — every assertion below is "
                        "absolute and needs them.\n");
        fm_free();
        return 77;
    }
    free(probe);

    RUN_TEST(router_topk_matches_golden);
    RUN_TEST(router_tie_break_prefers_lower_index);
    RUN_TEST(router_tie_break_matches_cpu_selection_sort);
    RUN_TEST(router_applies_expert_weights_scale);
    RUN_TEST(router_batch_matches_cpu_reference);

    RUN_TEST(expert_sort_matches_cpu_reference_at_prefill);
    RUN_TEST(expert_sort_is_a_permutation_of_the_routing);
    RUN_TEST(expert_sort_single_token);

    RUN_TEST(silu_mul_matches_cpu_reference);
    RUN_TEST(shexp_dot_gate_matches_golden_scalar);
    RUN_TEST(shexp_dot_gate_scales_the_branch_per_token);

    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
