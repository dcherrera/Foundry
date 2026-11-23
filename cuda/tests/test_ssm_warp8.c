/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_ssm_warp8.c
 * @brief Digests + timing of the S == 128 GDN recurrence entry points, for the 8-lane vs
 *        32-lane layout gate (ONE K80 die).
 * @status complete
 * @issues none
 * @todo none
 *
 * fc_ssm_recurrence_seq_ctx (npos 1, 2, 4, 8; snaps off and on) and fc_ssm_gdn_step_slots_ctx
 * (K = 1, 2, 4; identity and permuted state rows), 16 and 32 v-heads, run on seeded inputs
 * through the production dispatch. Each case prints an FNV-1a 64 digest of every output
 * (final state, y, snaps) and the median launch time. The gate runs the binary twice:
 *   FC_SSM_WARP8=0 ./cuda/tests/test_ssm_warp8 > w32.txt    (32-lane kernels)
 *   ./cuda/tests/test_ssm_warp8 > w8.txt                     (8-lane kernels, the default)
 * and the "case" digests must be identical. A build with NVCCFLAGS_EXTRA=-DFC_SSM_W8_RED
 * (state update rounded twice in the 8-lane kernels) must differ: that proves the digests
 * can see a one-rounding change. Inputs mix signs and magnitudes, zeros, and decays near 0
 * and 1; the per-element special values are the superoptimizer's proof (TeamIDE-Foundry
 * superopt/targets/gdn).
 *
 * Build: gcc -O2 -I. -o cuda/tests/test_ssm_warp8 cuda/tests/test_ssm_warp8.c cuda/fc_cuda.c -lm -ldl
 */

#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define W8_S 128
#define W8_ITERS 21

static uint32_t w8_rng_state = 0x9e3779b9u;

static uint32_t w8_rng(void)
{
    w8_rng_state ^= w8_rng_state << 13;
    w8_rng_state ^= w8_rng_state >> 17;
    w8_rng_state ^= w8_rng_state << 5;
    return w8_rng_state;
}

/* Uniform in [lo, hi), with one value in 16 set to exactly 0. */
static float *w8_floats(size_t n, float lo, float hi)
{
    float *v = malloc(n * sizeof(float));
    for (size_t i = 0; v && i < n; i++)
        v[i] = (w8_rng() & 15) == 0 ? 0.0f : lo + (hi - lo) * (float)(w8_rng() >> 8) / 16777216.0f;
    return v;
}

static uint64_t w8_fnv(uint64_t h, const void *p, size_t bytes)
{
    const unsigned char *b = p;
    for (size_t i = 0; i < bytes; i++)
        h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

static uint64_t w8_fnv_buffer(uint64_t h, fc_buffer *buf, size_t bytes)
{
    void *host = malloc(bytes);
    fc_buffer_to_cpu_n(buf, host, bytes);
    h = w8_fnv(h, host, bytes);
    free(host);
    return h;
}

static double w8_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int w8_cmp_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* One fc_ssm_recurrence_seq_ctx case: digest of the state after one launch, y and snaps,
 * then the median of W8_ITERS launches (state keeps evolving; timing only). */
static int w8_case_seq(int heads, int npos, int with_snaps)
{
    const size_t vec = (size_t)npos * heads * W8_S, state = (size_t)heads * W8_S * W8_S;
    float *s0 = w8_floats(state, -1.0f, 1.0f), *qh = w8_floats(vec, -0.2f, 0.2f);
    float *kh = w8_floats(vec, -0.2f, 0.2f), *vh = w8_floats(vec, -2.0f, 2.0f);
    float *gh = w8_floats((size_t)npos * heads, -6.0f, 0.0f);
    float *bh = w8_floats((size_t)npos * heads, 0.0f, 1.0f);
    fc_buffer *s = fc_buffer_from_ptr(s0, state * sizeof(float));
    fc_buffer *q = fc_buffer_from_ptr(qh, vec * sizeof(float));
    fc_buffer *k = fc_buffer_from_ptr(kh, vec * sizeof(float));
    fc_buffer *v = fc_buffer_from_ptr(vh, vec * sizeof(float));
    fc_buffer *g = fc_buffer_from_ptr(gh, (size_t)npos * heads * sizeof(float));
    fc_buffer *beta = fc_buffer_from_ptr(bh, (size_t)npos * heads * sizeof(float));
    fc_buffer *y = fc_buffer_alloc(vec * sizeof(float));
    fc_buffer *snaps = with_snaps ? fc_buffer_alloc((size_t)npos * state * sizeof(float)) : NULL;
    if (!s || !q || !k || !v || !g || !beta || !y || (with_snaps && !snaps)) {
        printf("seq heads %d npos %d: alloc failed\n", heads, npos);
        return 1;
    }
    fc_cuda_ctx *ctx = fc_default_ctx();
    fc_ssm_recurrence_seq_ctx(ctx, s, q, k, v, g, beta, y, heads, W8_S, npos, 0.088388f, snaps);
    fc_sync_ctx(ctx);
    uint64_t h = w8_fnv_buffer(0xcbf29ce484222325ull, s, state * sizeof(float));
    h = w8_fnv_buffer(h, y, vec * sizeof(float));
    if (snaps)
        h = w8_fnv_buffer(h, snaps, (size_t)npos * state * sizeof(float));
    double t[W8_ITERS];
    for (int i = 0; i < W8_ITERS; i++) {
        const double t0 = w8_now();
        fc_ssm_recurrence_seq_ctx(ctx, s, q, k, v, g, beta, y, heads, W8_S, npos, 0.088388f, snaps);
        fc_sync_ctx(ctx);
        t[i] = w8_now() - t0;
    }
    qsort(t, W8_ITERS, sizeof t[0], w8_cmp_double);
    printf("case seq heads %2d npos %d snaps %d fnv %016llx | %.1f us\n", heads, npos, with_snaps,
           (unsigned long long)h, 1e6 * t[W8_ITERS / 2]);
    fc_buffer *bufs[] = { s, q, k, v, g, beta, y, snaps };
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        if (bufs[i])
            fc_buffer_free(bufs[i]);
    free(s0); free(qh); free(kh); free(vh); free(gh); free(bh);
    return 0;
}

/* One fc_ssm_gdn_step_slots_ctx case over K slots; permuted: slot u owns state row K-1-u. */
static int w8_case_slots(int heads, int nslots, int permuted)
{
    const size_t vec = (size_t)nslots * heads * W8_S;
    const size_t state = (size_t)nslots * heads * W8_S * W8_S, per = (size_t)nslots * heads;
    float *s0 = w8_floats(state, -1.0f, 1.0f), *qh = w8_floats(vec, -0.2f, 0.2f);
    float *kh = w8_floats(vec, -0.2f, 0.2f), *vh = w8_floats(vec, -2.0f, 2.0f);
    float *ah = w8_floats(per, -4.0f, 24.0f), *dth = w8_floats((size_t)heads, -1.0f, 1.0f);
    float *ach = w8_floats((size_t)heads, -2.0f, -0.01f), *brh = w8_floats(per, -6.0f, 6.0f);
    int rows_h[8];
    for (int u = 0; u < nslots; u++)
        rows_h[u] = permuted ? nslots - 1 - u : u;
    fc_buffer *s = fc_buffer_from_ptr(s0, state * sizeof(float));
    fc_buffer *q = fc_buffer_from_ptr(qh, vec * sizeof(float));
    fc_buffer *k = fc_buffer_from_ptr(kh, vec * sizeof(float));
    fc_buffer *v = fc_buffer_from_ptr(vh, vec * sizeof(float));
    fc_buffer *alpha = fc_buffer_from_ptr(ah, per * sizeof(float));
    fc_buffer *dt = fc_buffer_from_ptr(dth, (size_t)heads * sizeof(float));
    fc_buffer *a = fc_buffer_from_ptr(ach, (size_t)heads * sizeof(float));
    fc_buffer *braw = fc_buffer_from_ptr(brh, per * sizeof(float));
    fc_buffer *rows = fc_buffer_from_ptr(rows_h, (size_t)nslots * sizeof(int));
    fc_buffer *y = fc_buffer_alloc(vec * sizeof(float));
    if (!s || !q || !k || !v || !alpha || !dt || !a || !braw || !rows || !y) {
        printf("slots heads %d K %d: alloc failed\n", heads, nslots);
        return 1;
    }
    fc_cuda_ctx *ctx = fc_default_ctx();
    if (fc_ssm_gdn_step_slots_ctx(ctx, s, q, k, v, alpha, dt, a, braw, y, heads, W8_S, nslots,
                                  0.088388f, permuted ? rows : NULL) != 0) {
        printf("slots heads %d K %d: not launched\n", heads, nslots);
        return 1;
    }
    fc_sync_ctx(ctx);
    uint64_t h = w8_fnv_buffer(0xcbf29ce484222325ull, s, state * sizeof(float));
    h = w8_fnv_buffer(h, y, vec * sizeof(float));
    double t[W8_ITERS];
    for (int i = 0; i < W8_ITERS; i++) {
        const double t0 = w8_now();
        fc_ssm_gdn_step_slots_ctx(ctx, s, q, k, v, alpha, dt, a, braw, y, heads, W8_S, nslots,
                                  0.088388f, permuted ? rows : NULL);
        fc_sync_ctx(ctx);
        t[i] = w8_now() - t0;
    }
    qsort(t, W8_ITERS, sizeof t[0], w8_cmp_double);
    printf("case slots heads %2d K %d rows %d fnv %016llx | %.1f us\n", heads, nslots, permuted,
           (unsigned long long)h, 1e6 * t[W8_ITERS / 2]);
    fc_buffer *bufs[] = { s, q, k, v, alpha, dt, a, braw, rows, y };
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        fc_buffer_free(bufs[i]);
    free(s0); free(qh); free(kh); free(vh); free(ah); free(dth); free(ach); free(brh);
    return 0;
}

int main(void)
{
    if (fc_init() != 0) {
        fprintf(stderr, "no CUDA\n");
        return 1;
    }
    const char *e = getenv("FC_SSM_WARP8");
    printf("FC_SSM_WARP8=%s\n", e ? e : "(default on)");
    int fails = 0;
    static const int heads[] = { 16, 32 };
    for (int hi = 0; hi < 2; hi++) {
        static const int npos[] = { 1, 2, 4, 8 };
        for (int ni = 0; ni < 4; ni++)
            for (int sn = 0; sn < 2; sn++)
                fails += w8_case_seq(heads[hi], npos[ni], sn);
        static const int ks[] = { 1, 2, 4 };
        for (int ki = 0; ki < 3; ki++)
            for (int pr = 0; pr < 2; pr++)
                fails += w8_case_slots(heads[hi], ks[ki], pr);
    }
    printf(fails ? "RESULT: FAIL (%d cases)\n" : "RESULT: ran all cases\n", fails);
    fc_free();
    return fails ? 1 : 0;
}
