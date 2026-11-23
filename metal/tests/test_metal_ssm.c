/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_metal_ssm.c — gated-delta SSM kernels (fm_ssm_decay_gate,
 * fm_ssm_conv1d_silu(_seq), fm_ssm_gated_norm, fm_ssm_recurrence(_seq))
 *
 * Every check here is ABSOLUTE: each kernel is compared against the
 * CPU-reference golden vectors in metal/tests/golden (provenance in
 * fm_golden_vectors.h), or — for the decay gate, which the golden set does
 * not cover — against a reference computed here from the same formula the
 * CPU path uses. Nothing is validated by agreeing with another GPU result,
 * because a GPU-vs-GPU arm stays green through a change that moves both
 * sides: test_metal_region's matvec arms survived a 0.01% row-sum scale and
 * every other break induced in them.
 *
 * The two GPU-vs-GPU comparisons that DO appear are equivalence checks
 * between two forms of the same kernel, and each is additionally pinned to a
 * golden or a CPU reference by another case in this file:
 *   - the sequential scan run one position at a time vs the single-step
 *     decode kernel, and
 *   - the scan over n positions vs n single-step dispatches.
 * Both are asserted BIT-EXACT (tolerance 0), not "close": the per-position
 * arithmetic is identical term for term, so any drift at all is a real
 * divergence rather than reassociation noise. That is what lets the scan
 * serve as the permanent oracle for the chunked prefill path.
 *
 * STATE IS ASSERTED, NOT JUST OUTPUT. A recurrence that emits the right
 * token and corrupts its carried state passes a y-only test and then fails
 * forty layers later, so every stateful case checks the state image too.
 *
 * Helper functions are named case_* so lib/test.h's failure line, which
 * prints __func__ + 5, still reads as the case that failed.
 *
 * Exits 77 when no Metal device is present, and fails loudly rather than
 * skipping when the golden vectors are missing.
 */

#include "lib/test.h"
#include "metal/kernels/fm_kernel_ssm.h"
#include "metal/tests/fm_golden_vectors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* qwen35moe SSM shape the golden set was generated for. */
#define SSM_N_V_HEADS      32
#define SSM_STATE_SIZE    128
#define SSM_CONV_DIM     8192
#define SSM_CONV_KERNEL     4
#define SSM_NORM_EPS     1e-6f

/* Positions used by the multi-token scan cases. Enough that a state carried
 * wrongly between positions compounds into a visible difference. */
#define SSM_SCAN_TOKENS     8

/* Tolerance against the CPU goldens, as tol_abs + tol_rel * |expected|.
 *
 * These are not guesses. Measured on M4 / macOS 26 by running each case with
 * the two terms driven to zero in turn, the worst observed disagreements are:
 *   recurrence y      3.7e-9 absolute · state 1.4e-8 absolute
 *   conv1d out        1.2e-7 absolute = 2.3e-7 relative
 *   gated_norm        3.6e-7 absolute = 2.9e-7 relative
 *   decay_gate        1.9e-6 absolute = 2.3e-7 relative
 * so the band below carries roughly 7x headroom on the relative term and
 * 14x on the absolute one. Both terms are needed: the GPU reduces over 32
 * SIMD lanes where pn_dot_f32 reduces sequentially, and the recurrence's
 * near-cancelling elements (|y| ~ 2e-7) reach 2.6e-3 RELATIVE error at
 * 6.6e-10 absolute — a pure-relative band would reject them and a
 * pure-absolute band would let large elements drift.
 *
 * The relative term is deliberately kept two orders below 1e-4: a 0.01%
 * output scale is exactly the break that has slipped through here before,
 * and at this band it is caught on every element above 2e-3 in magnitude —
 * thousands of them in each comparison. */
#define SSM_TOL_ABS      2e-7f
#define SSM_TOL_REL      2e-6f

static uint32_t ssm_rng = 0x9e3779b9u;

static float next_random_float(void) {
    ssm_rng ^= ssm_rng << 13;
    ssm_rng ^= ssm_rng >> 17;
    ssm_rng ^= ssm_rng << 5;
    return (float)((int32_t)ssm_rng) / 2147483648.0f;
}

static fm_buffer *upload_floats(const float *src, size_t n) {
    fm_buffer *buf = fm_buffer_alloc(n * sizeof(float));
    if (buf) memcpy(fm_buffer_data(buf), src, n * sizeof(float));
    return buf;
}

/* Diffs against the reference and asserts on the element that comes closest
 * to breaking ITS OWN tolerance, not the one with the largest raw difference
 * — with a relative term in the band those are not the same element, and
 * asserting on the wrong one lets a small-magnitude element drift by orders
 * of magnitude unnoticed. tol_abs = tol_rel = 0 means bit-exact.
 *
 * `label` is printed because lib/test.h names the failing function and every
 * case here shares this one; without it a failure would not say which
 * comparison broke. */
static void case_compare(const char *label, const float *expected,
                         const float *got, size_t n, float tol_abs,
                         float tol_rel) {
    size_t worst = 0;
    double worst_ratio = -1.0;
    for (size_t i = 0; i < n; i++) {
        double diff = fabs((double)expected[i] - (double)got[i]);
        double tol = (double)tol_abs + (double)tol_rel * fabs((double)expected[i]);
        double ratio = (tol > 0.0) ? diff / tol : (diff > 0.0 ? HUGE_VAL : 0.0);
        if (ratio > worst_ratio) { worst_ratio = ratio; worst = i; }
    }
    float worst_tol = tol_abs + tol_rel * fabsf(expected[worst]);
    if (worst_ratio > 1.0)
        printf("    %s: worst element %zu of %zu, expected %.9g got %.9g\n",
               label, worst, n, (double)expected[worst], (double)got[worst]);
    ASSERT_FLT_NEAR(expected[worst], got[worst], worst_tol);
}

/* SiLU written as the divisor form both the CPU reference and the shader
 * use, so the three round the same way. */
static float reference_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ================================================================
 * fm_ssm_recurrence — the decode centerpiece
 * ================================================================ */

/* Operands of the golden recurrence case, loaded once per test. */
typedef struct recur_golden {
    float *q, *k, *v, *g, *beta, *state_in, *state_out, *y;
} recur_golden;

static int recur_golden_load(recur_golden *set) {
    const size_t hs = (size_t)SSM_N_V_HEADS * SSM_STATE_SIZE;
    const size_t state_n = hs * SSM_STATE_SIZE;
    memset(set, 0, sizeof(*set));
    set->q = fm_golden_load_f32("ssm_recur_q.bin", hs);
    set->k = fm_golden_load_f32("ssm_recur_k.bin", hs);
    set->v = fm_golden_load_f32("ssm_recur_v.bin", hs);
    set->g = fm_golden_load_f32("ssm_recur_g.bin", SSM_N_V_HEADS);
    set->beta = fm_golden_load_f32("ssm_recur_beta.bin", SSM_N_V_HEADS);
    set->state_in = fm_golden_load_f32("ssm_recur_state_in.bin", state_n);
    set->state_out = fm_golden_load_f32("ssm_recur_state_out.bin", state_n);
    set->y = fm_golden_load_f32("ssm_recur_y.bin", hs);
    return (set->q && set->k && set->v && set->g && set->beta &&
            set->state_in && set->state_out && set->y) ? 0 : -1;
}

static void recur_golden_free(recur_golden *set) {
    free(set->q); free(set->k); free(set->v); free(set->g);
    free(set->beta); free(set->state_in); free(set->state_out); free(set->y);
}

static void test_recurrence_matches_golden(void) {
    const size_t hs = (size_t)SSM_N_V_HEADS * SSM_STATE_SIZE;
    const size_t state_n = hs * SSM_STATE_SIZE;
    recur_golden gold;
    ASSERT_INT_EQ(0, recur_golden_load(&gold));

    fm_buffer *bs = upload_floats(gold.state_in, state_n);
    fm_buffer *bq = upload_floats(gold.q, hs);
    fm_buffer *bk = upload_floats(gold.k, hs);
    fm_buffer *bv = upload_floats(gold.v, hs);
    fm_buffer *bg = upload_floats(gold.g, SSM_N_V_HEADS);
    fm_buffer *bb = upload_floats(gold.beta, SSM_N_V_HEADS);
    fm_buffer *by = fm_buffer_alloc(hs * sizeof(float));
    ASSERT_NOT_NULL(bs);
    ASSERT_NOT_NULL(bq);
    ASSERT_NOT_NULL(bk);
    ASSERT_NOT_NULL(bv);
    ASSERT_NOT_NULL(bg);
    ASSERT_NOT_NULL(bb);
    ASSERT_NOT_NULL(by);

    fm_ssm_recurrence(bs, bq, bk, bv, bg, bb, by, SSM_N_V_HEADS,
                      SSM_STATE_SIZE, 1.0f / sqrtf((float)SSM_STATE_SIZE));
    fm_sync();

    case_compare("recurrence y vs ssm_recur_y.bin", gold.y,
                 (const float *)fm_buffer_data(by), hs,
                 SSM_TOL_ABS, SSM_TOL_REL);
    /* The carried state matters as much as the token: a kernel that gets y
     * right and the state wrong passes a y-only check and then diverges. */
    case_compare("recurrence state vs ssm_recur_state_out.bin", gold.state_out,
                 (const float *)fm_buffer_data(bs), state_n,
                 SSM_TOL_ABS, SSM_TOL_REL);

    fm_buffer_free(bs); fm_buffer_free(bq); fm_buffer_free(bk);
    fm_buffer_free(bv); fm_buffer_free(bg); fm_buffer_free(bb);
    fm_buffer_free(by);
    recur_golden_free(&gold);
}

/* The _at form is how the forward will actually call this: activations are
 * slices of one persistent staging buffer, not buffers of their own. Every
 * operand here sits at a nonzero 16-byte-aligned offset behind poison the
 * kernel must not read, and the tail past each operand is poison the kernel
 * must not write — so an off-by-one in either direction is caught rather
 * than landing harmlessly inside a private allocation. */
static void test_recurrence_at_offsets_reads_and_writes_in_range(void) {
    const size_t hs = (size_t)SSM_N_V_HEADS * SSM_STATE_SIZE;
    const size_t state_n = hs * SSM_STATE_SIZE;
    const size_t pad = 16;                      /* floats; 64 bytes */
    const float poison = -12345.0f;
    recur_golden gold;
    ASSERT_INT_EQ(0, recur_golden_load(&gold));

    fm_buffer *bs = fm_buffer_alloc((state_n + 2 * pad) * sizeof(float));
    fm_buffer *bstream = fm_buffer_alloc((3 * hs + 4 * pad) * sizeof(float));
    fm_buffer *bhead = fm_buffer_alloc((2 * SSM_N_V_HEADS + 3 * pad) * sizeof(float));
    fm_buffer *by = fm_buffer_alloc((hs + 2 * pad) * sizeof(float));
    ASSERT_NOT_NULL(bs);
    ASSERT_NOT_NULL(bstream);
    ASSERT_NOT_NULL(bhead);
    ASSERT_NOT_NULL(by);

    float *sp = (float *)fm_buffer_data(bs);
    float *stream = (float *)fm_buffer_data(bstream);
    float *head = (float *)fm_buffer_data(bhead);
    float *yp = (float *)fm_buffer_data(by);
    for (size_t i = 0; i < state_n + 2 * pad; i++) sp[i] = poison;
    for (size_t i = 0; i < 3 * hs + 4 * pad; i++) stream[i] = poison;
    for (size_t i = 0; i < 2 * SSM_N_V_HEADS + 3 * pad; i++) head[i] = poison;
    for (size_t i = 0; i < hs + 2 * pad; i++) yp[i] = poison;

    memcpy(sp + pad, gold.state_in, state_n * sizeof(float));
    memcpy(stream + pad, gold.q, hs * sizeof(float));
    memcpy(stream + 2 * pad + hs, gold.k, hs * sizeof(float));
    memcpy(stream + 3 * pad + 2 * hs, gold.v, hs * sizeof(float));
    memcpy(head + pad, gold.g, SSM_N_V_HEADS * sizeof(float));
    memcpy(head + 2 * pad + SSM_N_V_HEADS, gold.beta,
           SSM_N_V_HEADS * sizeof(float));

    const size_t f = sizeof(float);
    fm_ssm_recurrence_at(fm_bufref_at(bs, pad * f),
                         fm_bufref_at(bstream, pad * f),
                         fm_bufref_at(bstream, (2 * pad + hs) * f),
                         fm_bufref_at(bstream, (3 * pad + 2 * hs) * f),
                         fm_bufref_at(bhead, pad * f),
                         fm_bufref_at(bhead, (2 * pad + SSM_N_V_HEADS) * f),
                         fm_bufref_at(by, pad * f),
                         SSM_N_V_HEADS, SSM_STATE_SIZE,
                         1.0f / sqrtf((float)SSM_STATE_SIZE));
    fm_sync();

    case_compare("recurrence _at y vs ssm_recur_y.bin", gold.y, yp + pad, hs,
                 SSM_TOL_ABS, SSM_TOL_REL);
    case_compare("recurrence _at state vs ssm_recur_state_out.bin",
                 gold.state_out, sp + pad, state_n, SSM_TOL_ABS, SSM_TOL_REL);
    for (size_t i = 0; i < pad; i++) {
        ASSERT_FLT_NEAR(poison, yp[i], 0.0f);
        ASSERT_FLT_NEAR(poison, yp[pad + hs + i], 0.0f);
        ASSERT_FLT_NEAR(poison, sp[i], 0.0f);
        ASSERT_FLT_NEAR(poison, sp[pad + state_n + i], 0.0f);
    }

    fm_buffer_free(bs); fm_buffer_free(bstream); fm_buffer_free(bhead);
    fm_buffer_free(by);
    recur_golden_free(&gold);
}

/* ================================================================
 * fm_ssm_recurrence_seq — the permanent reference scan
 * ================================================================ */

static void test_recurrence_seq_one_token_matches_golden(void) {
    const size_t hs = (size_t)SSM_N_V_HEADS * SSM_STATE_SIZE;
    const size_t state_n = hs * SSM_STATE_SIZE;
    recur_golden gold;
    ASSERT_INT_EQ(0, recur_golden_load(&gold));

    fm_buffer *bs = upload_floats(gold.state_in, state_n);
    fm_buffer *bq = upload_floats(gold.q, hs);
    fm_buffer *bk = upload_floats(gold.k, hs);
    fm_buffer *bv = upload_floats(gold.v, hs);
    fm_buffer *bg = upload_floats(gold.g, SSM_N_V_HEADS);
    fm_buffer *bb = upload_floats(gold.beta, SSM_N_V_HEADS);
    fm_buffer *by = fm_buffer_alloc(hs * sizeof(float));
    ASSERT_NOT_NULL(bs);
    ASSERT_NOT_NULL(by);

    fm_ssm_recurrence_seq(bs, bq, bk, bv, bg, bb, by, SSM_N_V_HEADS,
                          SSM_STATE_SIZE, 1,
                          1.0f / sqrtf((float)SSM_STATE_SIZE));
    fm_sync();

    case_compare("recurrence_seq n=1 y vs ssm_recur_y.bin", gold.y,
                 (const float *)fm_buffer_data(by), hs,
                 SSM_TOL_ABS, SSM_TOL_REL);
    case_compare("recurrence_seq n=1 state vs ssm_recur_state_out.bin",
                 gold.state_out, (const float *)fm_buffer_data(bs), state_n,
                 SSM_TOL_ABS, SSM_TOL_REL);

    fm_buffer_free(bs); fm_buffer_free(bq); fm_buffer_free(bk);
    fm_buffer_free(bv); fm_buffer_free(bg); fm_buffer_free(bb);
    fm_buffer_free(by);
    recur_golden_free(&gold);
}

/* Fills a token-major recurrence input set with the magnitudes the real
 * block produces: q and k L2-normed per head, v unnormalized, g small and
 * negative so exp(g) lands in (0,1), beta already sigmoid'd. */
static void fill_scan_inputs(float *q, float *k, float *v, float *g,
                             float *beta, int tokens) {
    const int S = SSM_STATE_SIZE, H = SSM_N_V_HEADS;
    for (int t = 0; t < tokens; t++) {
        for (int h = 0; h < H; h++) {
            float *qh = q + ((size_t)t * H + h) * S;
            float *kh = k + ((size_t)t * H + h) * S;
            float *vh = v + ((size_t)t * H + h) * S;
            float qsq = 0.0f, ksq = 0.0f;
            for (int d = 0; d < S; d++) {
                qh[d] = next_random_float();
                kh[d] = next_random_float();
                vh[d] = next_random_float();
                qsq += qh[d] * qh[d];
                ksq += kh[d] * kh[d];
            }
            float qinv = 1.0f / fmaxf(sqrtf(qsq), SSM_NORM_EPS);
            float kinv = 1.0f / fmaxf(sqrtf(ksq), SSM_NORM_EPS);
            for (int d = 0; d < S; d++) { qh[d] *= qinv; kh[d] *= kinv; }
            g[(size_t)t * H + h] = -0.05f - 0.1f * (next_random_float() * 0.5f + 0.5f);
            beta[(size_t)t * H + h] = 1.0f / (1.0f + expf(-next_random_float()));
        }
    }
}

/* The scan over n positions against n single-step dispatches, BIT-EXACT.
 * This is the property that makes the scan usable as the chunked path's
 * oracle: it is not merely close to the decode kernel, it is the same
 * arithmetic with the state held in registers instead of round-tripped
 * through device memory. Both y at every position and the final carried
 * state are compared. */
static void test_recurrence_seq_scan_equals_repeated_decode(void) {
    const int T = SSM_SCAN_TOKENS;
    const size_t hs = (size_t)SSM_N_V_HEADS * SSM_STATE_SIZE;
    const size_t state_n = hs * SSM_STATE_SIZE;
    const float scale = 1.0f / sqrtf((float)SSM_STATE_SIZE);

    float *q = malloc((size_t)T * hs * sizeof(float));
    float *k = malloc((size_t)T * hs * sizeof(float));
    float *v = malloc((size_t)T * hs * sizeof(float));
    float *g = malloc((size_t)T * SSM_N_V_HEADS * sizeof(float));
    float *beta = malloc((size_t)T * SSM_N_V_HEADS * sizeof(float));
    float *state0 = malloc(state_n * sizeof(float));
    float *step_y = malloc((size_t)T * hs * sizeof(float));
    float *step_state = malloc(state_n * sizeof(float));
    ASSERT_NOT_NULL(q);
    ASSERT_NOT_NULL(k);
    ASSERT_NOT_NULL(v);
    ASSERT_NOT_NULL(g);
    ASSERT_NOT_NULL(beta);
    ASSERT_NOT_NULL(state0);
    ASSERT_NOT_NULL(step_y);
    ASSERT_NOT_NULL(step_state);

    fill_scan_inputs(q, k, v, g, beta, T);
    for (size_t i = 0; i < state_n; i++)
        state0[i] = 0.1f * next_random_float();

    fm_buffer *bs = upload_floats(state0, state_n);
    fm_buffer *bq = upload_floats(q, (size_t)T * hs);
    fm_buffer *bk = upload_floats(k, (size_t)T * hs);
    fm_buffer *bv = upload_floats(v, (size_t)T * hs);
    fm_buffer *bg = upload_floats(g, (size_t)T * SSM_N_V_HEADS);
    fm_buffer *bb = upload_floats(beta, (size_t)T * SSM_N_V_HEADS);
    fm_buffer *by = fm_buffer_alloc((size_t)T * hs * sizeof(float));
    ASSERT_NOT_NULL(bs);
    ASSERT_NOT_NULL(by);

    /* Arm 1: T single-step dispatches, each advancing the state in place. */
    const size_t f = sizeof(float);
    for (int t = 0; t < T; t++)
        fm_ssm_recurrence_at(fm_bufref_at(bs, 0),
                             fm_bufref_at(bq, (size_t)t * hs * f),
                             fm_bufref_at(bk, (size_t)t * hs * f),
                             fm_bufref_at(bv, (size_t)t * hs * f),
                             fm_bufref_at(bg, (size_t)t * SSM_N_V_HEADS * f),
                             fm_bufref_at(bb, (size_t)t * SSM_N_V_HEADS * f),
                             fm_bufref_at(by, (size_t)t * hs * f),
                             SSM_N_V_HEADS, SSM_STATE_SIZE, scale);
    fm_sync();
    memcpy(step_y, fm_buffer_data(by), (size_t)T * hs * f);
    memcpy(step_state, fm_buffer_data(bs), state_n * f);

    /* Arm 2: one scan over the same T positions from the same start state. */
    memcpy(fm_buffer_data(bs), state0, state_n * f);
    memset(fm_buffer_data(by), 0, (size_t)T * hs * f);
    fm_ssm_recurrence_seq(bs, bq, bk, bv, bg, bb, by, SSM_N_V_HEADS,
                          SSM_STATE_SIZE, T, scale);
    fm_sync();

    case_compare("scan y vs repeated single-step y", step_y,
                 (const float *)fm_buffer_data(by), (size_t)T * hs, 0.0f, 0.0f);
    case_compare("scan state vs repeated single-step state", step_state,
                 (const float *)fm_buffer_data(bs), state_n, 0.0f, 0.0f);

    fm_buffer_free(bs); fm_buffer_free(bq); fm_buffer_free(bk);
    fm_buffer_free(bv); fm_buffer_free(bg); fm_buffer_free(bb);
    fm_buffer_free(by);
    free(q); free(k); free(v); free(g); free(beta);
    free(state0); free(step_y); free(step_state);
}

/* ================================================================
 * fm_ssm_conv1d_silu
 * ================================================================ */

static void test_conv1d_silu_matches_golden(void) {
    const size_t taps = SSM_CONV_KERNEL - 1;
    const size_t state_n = (size_t)SSM_CONV_DIM * taps;
    const size_t w_n = (size_t)SSM_CONV_DIM * SSM_CONV_KERNEL;

    float *qkv = fm_golden_load_f32("ssm_conv_qkv.bin", SSM_CONV_DIM);
    float *weight = fm_golden_load_f32("ssm_conv_weight.bin", w_n);
    float *state_in = fm_golden_load_f32("ssm_conv_state_in.bin", state_n);
    float *state_out = fm_golden_load_f32("ssm_conv_state_out.bin", state_n);
    float *out = fm_golden_load_f32("ssm_conv_out.bin", SSM_CONV_DIM);
    ASSERT_NOT_NULL(qkv);
    ASSERT_NOT_NULL(weight);
    ASSERT_NOT_NULL(state_in);
    ASSERT_NOT_NULL(state_out);
    ASSERT_NOT_NULL(out);

    fm_buffer *bx = upload_floats(qkv, SSM_CONV_DIM);
    fm_buffer *bw = upload_floats(weight, w_n);
    fm_buffer *bst = upload_floats(state_in, state_n);
    fm_buffer *bo = fm_buffer_alloc(SSM_CONV_DIM * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bw);
    ASSERT_NOT_NULL(bst);
    ASSERT_NOT_NULL(bo);

    fm_ssm_conv1d_silu(bx, bw, bst, bo, SSM_CONV_DIM, SSM_CONV_KERNEL);
    fm_sync();

    case_compare("conv1d out vs ssm_conv_out.bin", out,
                 (const float *)fm_buffer_data(bo), SSM_CONV_DIM,
                 SSM_TOL_ABS, SSM_TOL_REL);
    /* The rolled window is the whole point of the kernel — it must drop the
     * oldest tap and append the RAW input, not the convolved value. */
    case_compare("conv1d state vs ssm_conv_state_out.bin", state_out,
                 (const float *)fm_buffer_data(bst), state_n,
                 SSM_TOL_ABS, SSM_TOL_REL);

    fm_buffer_free(bx); fm_buffer_free(bw); fm_buffer_free(bst);
    fm_buffer_free(bo);
    free(qkv); free(weight); free(state_in); free(state_out); free(out);
}

/* The conv scan against repeated single steps, BIT-EXACT — the same
 * register-vs-memory equivalence the recurrence scan asserts, and the
 * property the chunked prefill path relies on for its conv leg. */
static void test_conv1d_silu_seq_equals_repeated_single_step(void) {
    const int T = SSM_SCAN_TOKENS;
    const size_t taps = SSM_CONV_KERNEL - 1;
    const size_t state_n = (size_t)SSM_CONV_DIM * taps;
    const size_t w_n = (size_t)SSM_CONV_DIM * SSM_CONV_KERNEL;

    float *qkv = malloc((size_t)T * SSM_CONV_DIM * sizeof(float));
    float *weight = malloc(w_n * sizeof(float));
    float *state0 = malloc(state_n * sizeof(float));
    float *step_out = malloc((size_t)T * SSM_CONV_DIM * sizeof(float));
    float *step_state = malloc(state_n * sizeof(float));
    ASSERT_NOT_NULL(qkv);
    ASSERT_NOT_NULL(weight);
    ASSERT_NOT_NULL(state0);
    ASSERT_NOT_NULL(step_out);
    ASSERT_NOT_NULL(step_state);

    for (size_t i = 0; i < (size_t)T * SSM_CONV_DIM; i++)
        qkv[i] = next_random_float();
    for (size_t i = 0; i < w_n; i++)
        weight[i] = 0.5f * next_random_float();
    for (size_t i = 0; i < state_n; i++)
        state0[i] = next_random_float();

    fm_buffer *bx = upload_floats(qkv, (size_t)T * SSM_CONV_DIM);
    fm_buffer *bw = upload_floats(weight, w_n);
    fm_buffer *bst = upload_floats(state0, state_n);
    fm_buffer *bo = fm_buffer_alloc((size_t)T * SSM_CONV_DIM * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bw);
    ASSERT_NOT_NULL(bst);
    ASSERT_NOT_NULL(bo);

    const size_t f = sizeof(float);
    for (int t = 0; t < T; t++)
        fm_ssm_conv1d_silu_at(fm_bufref_at(bx, (size_t)t * SSM_CONV_DIM * f),
                              fm_bufref_at(bw, 0), fm_bufref_at(bst, 0),
                              fm_bufref_at(bo, (size_t)t * SSM_CONV_DIM * f),
                              SSM_CONV_DIM, SSM_CONV_KERNEL);
    fm_sync();
    memcpy(step_out, fm_buffer_data(bo), (size_t)T * SSM_CONV_DIM * f);
    memcpy(step_state, fm_buffer_data(bst), state_n * f);

    memcpy(fm_buffer_data(bst), state0, state_n * f);
    memset(fm_buffer_data(bo), 0, (size_t)T * SSM_CONV_DIM * f);
    fm_ssm_conv1d_silu_seq(bx, bw, bst, bo, SSM_CONV_DIM, SSM_CONV_KERNEL, T);
    fm_sync();

    case_compare("conv scan out vs repeated single-step out", step_out,
                 (const float *)fm_buffer_data(bo),
                 (size_t)T * SSM_CONV_DIM, 0.0f, 0.0f);
    case_compare("conv scan state vs repeated single-step state", step_state,
                 (const float *)fm_buffer_data(bst), state_n, 0.0f, 0.0f);

    fm_buffer_free(bx); fm_buffer_free(bw); fm_buffer_free(bst);
    fm_buffer_free(bo);
    free(qkv); free(weight); free(state0); free(step_out); free(step_state);
}

/* ================================================================
 * fm_ssm_gated_norm
 * ================================================================ */

static void test_gated_norm_matches_golden(void) {
    const size_t hs = (size_t)SSM_N_V_HEADS * SSM_STATE_SIZE;

    float *y = fm_golden_load_f32("ssm_gated_y.bin", hs);
    float *z = fm_golden_load_f32("ssm_gated_z.bin", hs);
    float *weight = fm_golden_load_f32("ssm_gated_weight.bin", SSM_STATE_SIZE);
    float *out = fm_golden_load_f32("ssm_gated_out.bin", hs);
    ASSERT_NOT_NULL(y);
    ASSERT_NOT_NULL(z);
    ASSERT_NOT_NULL(weight);
    ASSERT_NOT_NULL(out);

    fm_buffer *by = upload_floats(y, hs);
    fm_buffer *bz = upload_floats(z, hs);
    fm_buffer *bw = upload_floats(weight, SSM_STATE_SIZE);
    fm_buffer *bo = fm_buffer_alloc(hs * sizeof(float));
    ASSERT_NOT_NULL(by);
    ASSERT_NOT_NULL(bz);
    ASSERT_NOT_NULL(bw);
    ASSERT_NOT_NULL(bo);

    fm_ssm_gated_norm(by, bw, bz, bo, SSM_N_V_HEADS, SSM_STATE_SIZE,
                      SSM_NORM_EPS);
    fm_sync();

    case_compare("gated_norm vs ssm_gated_out.bin", out,
                 (const float *)fm_buffer_data(bo), hs,
                 SSM_TOL_ABS, SSM_TOL_REL);

    fm_buffer_free(by); fm_buffer_free(bz); fm_buffer_free(bw);
    fm_buffer_free(bo);
    free(y); free(z); free(weight); free(out);
}

/* THE SHARED-WEIGHT TRAP, the same one fm_qk_rms_norm needs. The weight
 * buffer is n_heads * head_dim long: the first head_dim floats are the real
 * shared weight and the rest — exactly the region a per-head-indexing kernel
 * would read for heads 1..3 — hold very different values, so wrong indexing
 * produces a specific wrong answer instead of maybe getting away with a read
 * past the buffer. A uniform weight would make both indexings agree. */
static void test_gated_norm_weight_is_shared_across_heads(void) {
    const int heads = 4, head_dim = 64;
    const size_t count = (size_t)heads * head_dim;
    float y[4 * 64], z[4 * 64], weight[4 * 64], expected[4 * 64];

    for (size_t i = 0; i < count; i++) {
        y[i] = next_random_float() * 3.0f;
        z[i] = next_random_float() * 2.0f;
    }
    for (int i = 0; i < head_dim; i++)
        weight[i] = 0.5f + 0.01f * (float)i;          /* the shared weight */
    for (size_t i = head_dim; i < count; i++)
        weight[i] = -4.0f - 0.02f * (float)i;         /* never to be read */

    for (int h = 0; h < heads; h++) {
        const float *yh = y + (size_t)h * head_dim;
        double sum_sq = 0.0;
        for (int d = 0; d < head_dim; d++)
            sum_sq += (double)yh[d] * (double)yh[d];
        float inv_rms = (float)(1.0 / sqrt(sum_sq / (double)head_dim +
                                           (double)SSM_NORM_EPS));
        for (int d = 0; d < head_dim; d++) {
            size_t i = (size_t)h * head_dim + d;
            expected[i] = y[i] * inv_rms * weight[d] * reference_silu(z[i]);
        }
    }

    fm_buffer *by = upload_floats(y, count);
    fm_buffer *bz = upload_floats(z, count);
    fm_buffer *bw = upload_floats(weight, count);
    fm_buffer *bo = fm_buffer_alloc(count * sizeof(float));
    ASSERT_NOT_NULL(by);
    ASSERT_NOT_NULL(bz);
    ASSERT_NOT_NULL(bw);
    ASSERT_NOT_NULL(bo);

    fm_ssm_gated_norm(by, bw, bz, bo, heads, head_dim, SSM_NORM_EPS);
    fm_sync();

    case_compare("gated_norm with a shared non-uniform weight", expected,
                 (const float *)fm_buffer_data(bo), count,
                 SSM_TOL_ABS, SSM_TOL_REL);

    fm_buffer_free(by); fm_buffer_free(bz); fm_buffer_free(bw);
    fm_buffer_free(bo);
}

/* ================================================================
 * fm_ssm_decay_gate
 *
 * The golden set does not cover the decay gate, so the reference is
 * computed here from the same expression the CPU path uses. It is still an
 * absolute check — a CPU oracle, not a second GPU opinion.
 * ================================================================ */

static float reference_decay_gate(float alpha, float dt, float a) {
    float v = alpha + dt;
    float softplus = (v > 20.0f) ? v : log1pf(expf(v));
    return a * softplus;
}

static void test_decay_gate_matches_cpu_reference(void) {
    const int rows = 3, heads = SSM_N_V_HEADS;
    const size_t count = (size_t)rows * heads;
    float alpha[3 * SSM_N_V_HEADS], expected[3 * SSM_N_V_HEADS];
    float dt[SSM_N_V_HEADS], a[SSM_N_V_HEADS];

    /* dt and a are per-head and BROADCAST over rows; giving them strongly
     * head-dependent values means a kernel that indexed them by the flat
     * (row, head) index would produce a different answer for every row
     * after the first, instead of silently agreeing. */
    for (int h = 0; h < heads; h++) {
        dt[h] = -1.0f + 0.13f * (float)h;
        a[h] = -(0.25f + 0.05f * (float)h);
    }
    for (size_t i = 0; i < count; i++)
        alpha[i] = next_random_float() * 4.0f;
    for (int r = 0; r < rows; r++)
        for (int h = 0; h < heads; h++) {
            size_t i = (size_t)r * heads + h;
            expected[i] = reference_decay_gate(alpha[i], dt[h], a[h]);
        }

    fm_buffer *ba = upload_floats(alpha, count);
    fm_buffer *bdt = upload_floats(dt, heads);
    fm_buffer *baa = upload_floats(a, heads);
    fm_buffer *bg = fm_buffer_alloc(count * sizeof(float));
    ASSERT_NOT_NULL(ba);
    ASSERT_NOT_NULL(bdt);
    ASSERT_NOT_NULL(baa);
    ASSERT_NOT_NULL(bg);

    fm_ssm_decay_gate(ba, bdt, baa, bg, heads, rows);
    fm_sync();

    case_compare("decay_gate rows=3 vs CPU reference", expected,
                 (const float *)fm_buffer_data(bg), count,
                 SSM_TOL_ABS, SSM_TOL_REL);

    fm_buffer_free(ba); fm_buffer_free(bdt); fm_buffer_free(baa);
    fm_buffer_free(bg);
}

/* The guard at 20 is load-bearing, not cosmetic: exp(89) overflows float32,
 * so an unguarded log1p(exp(v)) turns the head's decay into an infinity and
 * then the state into NaN. These inputs straddle the guard. */
static void test_decay_gate_guards_softplus_at_20(void) {
    const int heads = 4;
    float alpha[4] = { 19.5f, 20.5f, 90.0f, 1000.0f };
    float dt[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float a[4] = { -1.0f, -2.0f, -0.5f, -0.25f };

    fm_buffer *ba = upload_floats(alpha, heads);
    fm_buffer *bdt = upload_floats(dt, heads);
    fm_buffer *baa = upload_floats(a, heads);
    fm_buffer *bg = fm_buffer_alloc(heads * sizeof(float));
    ASSERT_NOT_NULL(ba);
    ASSERT_NOT_NULL(bdt);
    ASSERT_NOT_NULL(baa);
    ASSERT_NOT_NULL(bg);

    fm_ssm_decay_gate(ba, bdt, baa, bg, heads, 1);
    fm_sync();

    const float *got = (const float *)fm_buffer_data(bg);
    for (int h = 0; h < heads; h++) {
        ASSERT_TRUE(isfinite(got[h]));
        float expected = reference_decay_gate(alpha[h], dt[h], a[h]);
        ASSERT_FLT_NEAR(expected, got[h],
                        SSM_TOL_ABS + SSM_TOL_REL * fabsf(expected));
    }
    /* Above the guard softplus(v) is v to well within float32, so the
     * result is exactly a*v — assert that, not merely finiteness. */
    ASSERT_FLT_NEAR(a[3] * alpha[3], got[3], 0.0f);

    fm_buffer_free(ba); fm_buffer_free(bdt); fm_buffer_free(baa);
    fm_buffer_free(bg);
}

/* The other end of the range, and the reason the shader cannot just write
 * log(1 + exp(v)). Once exp(v) falls below a float32 ulp of 1.0 that
 * addition absorbs it and softplus collapses toward zero — an error that is
 * invisible in absolute terms but total in relative terms, which is why this
 * case asserts RELATIVELY (tol_abs = 0). It matters because the chunked
 * prefill path takes cumulative prefix sums of g along a chunk, so a decay
 * that is systematically rounded toward zero accumulates over CS positions
 * instead of staying a per-token rounding difference. */
static void test_decay_gate_small_softplus_is_not_absorbed(void) {
    const int heads = 8;
    float alpha[8], dt[8], a[8], expected[8];

    for (int h = 0; h < heads; h++) {
        alpha[h] = -4.0f - 2.0f * (float)h;      /* -4 down to -18 */
        dt[h] = 0.0f;
        a[h] = -(0.5f + 0.25f * (float)h);
        expected[h] = reference_decay_gate(alpha[h], dt[h], a[h]);
    }

    fm_buffer *ba = upload_floats(alpha, heads);
    fm_buffer *bdt = upload_floats(dt, heads);
    fm_buffer *baa = upload_floats(a, heads);
    fm_buffer *bg = fm_buffer_alloc(heads * sizeof(float));
    ASSERT_NOT_NULL(ba);
    ASSERT_NOT_NULL(bdt);
    ASSERT_NOT_NULL(baa);
    ASSERT_NOT_NULL(bg);

    fm_ssm_decay_gate(ba, bdt, baa, bg, heads, 1);
    fm_sync();

    case_compare("decay_gate small softplus, relative only", expected,
                 (const float *)fm_buffer_data(bg), heads, 0.0f, SSM_TOL_REL);

    fm_buffer_free(ba); fm_buffer_free(bdt); fm_buffer_free(baa);
    fm_buffer_free(bg);
}

int main(int argc, char **argv) {
    (void)argc;
    fm_golden_locate(argv[0]);

    printf("\n  Metal Gated-Delta SSM Kernel Tests\n");
    printf("  ==================================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "  Metal unavailable — skipping.\n\n");
        return 77;
    }

    RUN_TEST(recurrence_matches_golden);
    RUN_TEST(recurrence_at_offsets_reads_and_writes_in_range);
    RUN_TEST(recurrence_seq_one_token_matches_golden);
    RUN_TEST(recurrence_seq_scan_equals_repeated_decode);
    RUN_TEST(conv1d_silu_matches_golden);
    RUN_TEST(conv1d_silu_seq_equals_repeated_single_step);
    RUN_TEST(gated_norm_matches_golden);
    RUN_TEST(gated_norm_weight_is_shared_across_heads);
    RUN_TEST(decay_gate_matches_cpu_reference);
    RUN_TEST(decay_gate_guards_softplus_at_20);
    RUN_TEST(decay_gate_small_softplus_is_not_absorbed);

    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
