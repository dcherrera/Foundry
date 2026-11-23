/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_det_math.c
 * @brief Off-hardware acceptance of kernels/ft_det_math.h (integer-only fp32 gate math).
 *
 *   test_det_math                       full run: exhaustive unary sweeps + sampled checks
 *   test_det_math --stride N            sweep every Nth fp32 only (quick look)
 *   test_det_math --red                 run the mutated arms; each MUST fail (exit non-zero)
 *   test_det_math --binary              tables, edge cases and add/mul only (the x86 run)
 *   test_det_math --emit FILE N SEED    write on-card comparison vectors (format below)
 *
 * WHAT IS CHECKED (each failure counted; exit non-zero on any):
 *   1. the header's four tables, recomputed in double (within 1 unit of Q32/Q31);
 *   2. exp / softplus / sigmoid vs double libm, EVERY fp32 in range: exp [-103.9, 88.7]
 *      <= 1 ulp, softplus [-104, 20] <= 2 ulp, sigmoid [-104, 104] <= 1 ulp;
 *   3. the gate composites on 1e8 random (alpha, dt, ssm_a) / g: gdn_g vs a double ref with
 *      the host's fp32 steps (sum and softplus rounded+FTZ'd, product rounded once) <= 3 ulp;
 *      vs an all-double ref informationally (the fp32 add's rounding error is amplified by
 *      up to |x| in softplus' far tail, so that number measures the FORMULA, not this code);
 *   4. add / mul on 1e8 random pairs each vs a SOFTWARE model of x86 addss/mulss under
 *      FTZ|DAZ (double arithmetic is exact or innocuously rounded for fp32 operands; tininess
 *      judged after rounding at unbounded exponent) -- must be bit-exact. The Mac is arm64 so
 *      MXCSR cannot be set natively; native arm64 float + and * (FPCR default: no flush) are
 *      compared INFORMATIONALLY where no operand or result is tiny. mul must also equal
 *      ftgc_mul. The x86_64 build (`make -f Makefile.detmath x86`, run under Rosetta 2) sets
 *      FTZ|DAZ through Apple's fenv FE_DFL_DISABLE_SSE_DENORMS_ENV (no intrinsics) in every
 *      worker and compares against native addss/mulss on ALL pairs (informational: it
 *      measures Rosetta's emulation as much as this code);
 *   5. edge cases: +-0, +-Inf, NaN, denormals, the softplus 20.0f threshold, exp boundaries;
 *   6. FAST path: ft_det_gdn_gate on ALL 65,536 bf16 alpha x 4,096 random (dt in [-20, 20],
 *      ssm_a = -exp(U(ln 1e-3, ln 64))) <= 4 ulp vs exp(a * softplus(alpha + dt)) in double
 *      (softplus in its stable form max(x,0) + log1p(exp(-|x|)), so huge alpha stays finite);
 *      ft_det_gdn_beta on all 65,536 bf16 values and every fp32 in [-104, 104] <= 2 ulp.
 *
 * ULP DISTANCE is measured in the FTZ number line (denormals do not exist, so +0 and 2^-126
 * are adjacent). References: inputs DAZ'd, result rounded to fp32 then FTZ'd after rounding.
 *
 * EMIT FORMAT (all little-endian uint32), version 2:
 *   header  : 'F','T','D','M' magic, version (2), record count n, words per record (13)
 *   record  : in0, in1, in2,
 *             add(in0,in1), mul(in0,in1), exp(in0), softplus(in0), sigmoid(in0),
 *             gdn_beta_reference(in0), gdn_g(in0,in1,in2), gdn_decay(in0),   <- as version 1
 *             gdn_beta(in0) [fast], gdn_gate(in0 & 0xFFFF0000, in1, in2) [fast, bf16 alpha]
 *   The first records are the edge-case inputs; then in0 spans [-104, 104] (half uniform in
 *   value, half uniform in bit pattern), in1 in [-8, 8], in2 = -exp(U(-6, 3)).
 *
 * @status The full run (~9.2e9 exhaustive points) takes well under a minute on 8 threads (M4).
 * @issues None
 * @todo None
 */
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../kernels/ft_det_math.h"
#include "../kernels/ftgc_compact.h"
#include "det_math_arm.h"

#define THREADS 8u
#define SAMPLES_BINARY 100000000ull
#define SAMPLES_COMPOSITE 100000000ull
#define RED_STRIDE 97u
#define RED_SAMPLES 2000000ull
#define HISTOGRAM_BINS 5 /* 0, 1, 2, 3, >= 4 ulp */

/** x86 MXCSR FTZ|DAZ for the calling thread, through libc fenv only. 0 if unavailable. */
static int enable_native_ftz_daz(void)
{
#if defined(__x86_64__) && defined(FE_DFL_DISABLE_SSE_DENORMS_ENV)
    return fesetenv(FE_DFL_DISABLE_SSE_DENORMS_ENV) == 0;
#else
    return 0;
#endif
}

/* ---------------------------------------------------------------- bit helpers */

static float bits_to_float(uint32_t bits)
{
    float value;

    memcpy(&value, &bits, sizeof value);
    return value;
}

static uint32_t float_to_bits(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static int bits_is_nan(uint32_t bits)
{
    return (bits & 0x7FFFFFFFu) > 0x7F800000u;
}

/** Input as the hardware reads it: a denormal is a zero of the same sign (DAZ). */
static double daz_input(uint32_t bits)
{
    if ((bits & 0x7F800000u) == 0u)
        bits &= 0x80000000u;
    return (double)bits_to_float(bits);
}

/** Round to fp32, then flush if tiny AFTER rounding at unbounded exponent (x86 FTZ). */
static uint32_t round_ftz(double value)
{
    static const double tiny_bound = 0x1p-126 - 0x1p-151; /* rounds up to 2^-126 at/above */

    if (value != value)
        return 0x7FC00000u;
    if (fabs(value) < tiny_bound)
        return signbit(value) ? 0x80000000u : 0u;
    return float_to_bits((float)value);
}

/** Position on the FTZ number line: +-0 and denormals are 0, 2^-126 is +-1. */
static int64_t ftz_ordinal(uint32_t bits)
{
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const int64_t ordinal = magnitude < 0x00800000u ? 0 : (int64_t)magnitude - 0x007FFFFF;

    return (bits & 0x80000000u) ? -ordinal : ordinal;
}

static uint64_t ulp_distance(uint32_t got, uint32_t want)
{
    const int64_t d = ftz_ordinal(got) - ftz_ordinal(want);

    if (bits_is_nan(got) || bits_is_nan(want))
        return (bits_is_nan(got) && bits_is_nan(want)) ? 0u : UINT64_MAX;
    return (uint64_t)(d < 0 ? -d : d);
}

static uint64_t splitmix64(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);

    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/** Uniform double in [lo, hi). */
static double random_uniform(uint64_t *state, double lo, double hi)
{
    return lo + (hi - lo) * (double)(splitmix64(state) >> 11) * 0x1p-53;
}

/* ---------------------------------------------------------------- references */

static double reference_exp(double x)
{
    return exp(x);
}

static double reference_softplus(double x)
{
    return x > 20.0 ? x : log1p(exp(x));
}

static double reference_sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

/** x86 addss/mulss under FTZ|DAZ, modeled in double. */
static uint32_t reference_binary(uint32_t a, uint32_t b, int is_mul)
{
    double fa, fb;

    if (bits_is_nan(a))
        return a | 0x00400000u;
    if (bits_is_nan(b))
        return b | 0x00400000u;
    fa = daz_input(a);
    fb = daz_input(b);
    if (is_mul) {
        if ((isinf(fa) && fb == 0.0) || (isinf(fb) && fa == 0.0))
            return 0xFFC00000u;
        return round_ftz(fa * fb);
    }
    if (isinf(fa) && isinf(fb) && signbit(fa) != signbit(fb))
        return 0xFFC00000u;
    return round_ftz(fa + fb);
}

/* ---------------------------------------------------------------- error statistics */

typedef struct ErrorStats {
    uint64_t histogram[HISTOGRAM_BINS];
    uint64_t max_ulp;
    uint32_t worst_input;
    uint64_t count;
    uint64_t failures;
} ErrorStats;

static void stats_add(ErrorStats *stats, uint64_t ulp, uint64_t tolerance, uint32_t input)
{
    stats->histogram[ulp < HISTOGRAM_BINS - 1 ? ulp : HISTOGRAM_BINS - 1]++;
    if (stats->count++ == 0u || ulp > stats->max_ulp) {
        stats->max_ulp = ulp;
        stats->worst_input = input;
    }
    if (ulp > tolerance)
        stats->failures++;
}

static void stats_merge(ErrorStats *into, const ErrorStats *from)
{
    int bin;

    for (bin = 0; bin < HISTOGRAM_BINS; bin++)
        into->histogram[bin] += from->histogram[bin];
    if (from->max_ulp > into->max_ulp) {
        into->max_ulp = from->max_ulp;
        into->worst_input = from->worst_input;
    }
    into->count += from->count;
    into->failures += from->failures;
}

static void stats_print(const char *label, const ErrorStats *stats, uint64_t tolerance,
                        int informational)
{
    printf("  %-34s n=%-11llu max=%-3llu tol=%-3llu hist[0,1,2,3,4+]=%llu,%llu,%llu,%llu,%llu"
           "  worst=0x%08X%s\n",
           label, (unsigned long long)stats->count, (unsigned long long)stats->max_ulp,
           (unsigned long long)tolerance, (unsigned long long)stats->histogram[0],
           (unsigned long long)stats->histogram[1], (unsigned long long)stats->histogram[2],
           (unsigned long long)stats->histogram[3], (unsigned long long)stats->histogram[4],
           stats->worst_input, informational ? "  (informational)" : "");
}

/* ---------------------------------------------------------------- unary sweeps */

typedef struct UnarySweep {
    const char *name;
    uint32_t (*function)(uint32_t);
    double (*reference)(double);
    uint32_t negative_limit; /* sweep magnitudes 0..limit with the sign bit set */
    uint32_t positive_limit; /* and 0..limit without */
    uint64_t tolerance;
    uint32_t stride;
    unsigned thread_index;
    ErrorStats stats;
} UnarySweep;

static void *run_unary_sweep_thread(void *argument)
{
    UnarySweep *sweep = argument;
    const uint64_t negatives = (uint64_t)sweep->negative_limit + 1u;
    const uint64_t total = negatives + sweep->positive_limit + 1u;
    uint64_t index;

    for (index = (uint64_t)sweep->thread_index * sweep->stride; index < total;
         index += (uint64_t)THREADS * sweep->stride) {
        const uint32_t x = index < negatives ? (0x80000000u | (uint32_t)index)
                                             : (uint32_t)(index - negatives);
        const uint32_t want = round_ftz(sweep->reference(daz_input(x)));

        stats_add(&sweep->stats, ulp_distance(sweep->function(x), want), sweep->tolerance, x);
    }
    return NULL;
}

static uint64_t run_unary_sweep(const char *name, uint32_t (*function)(uint32_t),
                                double (*reference)(double), float lo, float hi,
                                uint64_t tolerance, uint32_t stride)
{
    UnarySweep sweeps[THREADS];
    pthread_t threads[THREADS];
    ErrorStats total;
    unsigned t;

    memset(&total, 0, sizeof total);
    for (t = 0; t < THREADS; t++) {
        memset(&sweeps[t], 0, sizeof sweeps[t]);
        sweeps[t].name = name;
        sweeps[t].function = function;
        sweeps[t].reference = reference;
        sweeps[t].negative_limit = float_to_bits(lo) & 0x7FFFFFFFu;
        sweeps[t].positive_limit = float_to_bits(hi);
        sweeps[t].tolerance = tolerance;
        sweeps[t].stride = stride;
        sweeps[t].thread_index = t;
        pthread_create(&threads[t], NULL, run_unary_sweep_thread, &sweeps[t]);
    }
    for (t = 0; t < THREADS; t++) {
        pthread_join(threads[t], NULL);
        stats_merge(&total, &sweeps[t].stats);
    }
    stats_print(name, &total, tolerance, 0);
    return total.failures;
}

/* ---------------------------------------------------------------- binary (add/mul) */

typedef struct BinaryRun {
    const DetMathArm *arm;
    uint64_t samples;
    uint64_t seed;
    uint64_t add_mismatches, mul_mismatches, ftgc_compared, ftgc_mismatches;
    uint64_t native_compared, native_mismatches, native_nan_payload;
    int native_ftz_daz;
    uint32_t first_bad_a, first_bad_b;
} BinaryRun;

/** A random operand pair from one of four strata. */
static void random_binary_pair(uint64_t *state, uint64_t index, uint32_t *a, uint32_t *b)
{
    const uint64_t r = splitmix64(state);
    uint32_t ea, eb;

    *a = (uint32_t)r;
    *b = (uint32_t)(r >> 32);
    switch (index & 3u) {
    case 0: /* raw bit patterns: every class, specials included */
        break;
    case 1: /* close exponents: cancellation, ties, carries */
        ea = (*a >> 23) & 0xFFu;
        eb = (ea + ((*b >> 23) & 3u)) & 0xFFu;
        *b = (*b & 0x807FFFFFu) | (eb << 23);
        break;
    case 2: /* results near 2^-126 (mul: exponents summing to ~1; add: tiny operands) */
        ea = 1u + ((*a >> 23) & 0x7Fu);
        eb = (uint32_t)((int32_t)128 - (int32_t)ea + (int32_t)((*b >> 23) & 3u) - 1);
        eb = (eb & 0xFFu) == 0xFFu ? 1u : (eb & 0xFFu);
        *a = (*a & 0x807FFFFFu) | (ea << 23);
        *b = (*b & 0x807FFFFFu) | (((index & 4u) ? eb : 1u + ((*b >> 23) & 1u)) << 23);
        if (!(index & 4u))
            *a = (*a & 0x807FFFFFu) | ((1u + ((*a >> 24) & 1u)) << 23);
        break;
    default: /* zero / denormal / Inf exponents mixed with normals */
        ea = (*a >> 23) & 1u ? 0xFFu : 0u;
        *a = (*a & 0x807FFFFFu) | (ea << 23);
        break;
    }
}

static int is_tiny_or_special(uint32_t bits)
{
    const uint32_t e = (bits >> 23) & 0xFFu;

    return e == 0u || e == 0xFFu;
}

static void *run_binary_thread(void *argument)
{
    BinaryRun *run = argument;
    uint64_t state = run->seed, index;

    run->native_ftz_daz = enable_native_ftz_daz();

    for (index = 0; index < run->samples; index++) {
        uint32_t a, b, want_add, want_mul;

        random_binary_pair(&state, index, &a, &b);
        want_add = reference_binary(a, b, 0);
        want_mul = reference_binary(a, b, 1);
        if (run->arm->add(a, b) != want_add) {
            if (run->add_mismatches++ == 0u) {
                run->first_bad_a = a;
                run->first_bad_b = b;
            }
        }
        if (run->arm->mul(a, b) != want_mul)
            run->mul_mismatches++;
        if (((a >> 23) & 0xFFu) != 0xFFu && ((b >> 23) & 0xFFu) != 0xFFu) {
            run->ftgc_compared++;
            if (ftgc_mul(a, b) != run->arm->mul(a, b))
                run->ftgc_mismatches++;
        }
        if (run->native_ftz_daz ||
            (!is_tiny_or_special(a) && !is_tiny_or_special(b) && !is_tiny_or_special(want_add) &&
             !is_tiny_or_special(want_mul))) {
            const volatile float fa = bits_to_float(a), fb = bits_to_float(b);
            const uint32_t native_add = float_to_bits(fa + fb), native_mul = float_to_bits(fa * fb);

            run->native_compared++;
            if (native_add != want_add || native_mul != want_mul) {
                if ((native_add == want_add || (bits_is_nan(native_add) && bits_is_nan(want_add))) &&
                    (native_mul == want_mul || (bits_is_nan(native_mul) && bits_is_nan(want_mul))))
                    run->native_nan_payload++;
                else
                    run->native_mismatches++;
            }
        }
    }
    return NULL;
}

static uint64_t run_binary_checks(const DetMathArm *arm, uint64_t samples)
{
    BinaryRun runs[THREADS];
    pthread_t threads[THREADS];
    uint64_t add_bad = 0, mul_bad = 0, compared = 0, native_bad = 0, ftgc_n = 0, ftgc_bad = 0;
    uint64_t nan_payload = 0;
    int ftz_daz = 0;
    unsigned t;

    for (t = 0; t < THREADS; t++) {
        memset(&runs[t], 0, sizeof runs[t]);
        runs[t].arm = arm;
        runs[t].samples = samples / THREADS;
        runs[t].seed = 0xD37A11ull + t * 0x1000193ull;
        pthread_create(&threads[t], NULL, run_binary_thread, &runs[t]);
    }
    for (t = 0; t < THREADS; t++) {
        pthread_join(threads[t], NULL);
        add_bad += runs[t].add_mismatches;
        mul_bad += runs[t].mul_mismatches;
        compared += runs[t].native_compared;
        native_bad += runs[t].native_mismatches;
        nan_payload += runs[t].native_nan_payload;
        ftz_daz = runs[t].native_ftz_daz;
        ftgc_n += runs[t].ftgc_compared;
        ftgc_bad += runs[t].ftgc_mismatches;
    }
    printf("  add vs x86 FTZ|DAZ model          n=%llu mismatches=%llu\n",
           (unsigned long long)(samples / THREADS * THREADS), (unsigned long long)add_bad);
    printf("  mul vs x86 FTZ|DAZ model          n=%llu mismatches=%llu\n",
           (unsigned long long)(samples / THREADS * THREADS), (unsigned long long)mul_bad);
    printf("  mul vs ftgc_mul (finite operands)  n=%llu mismatches=%llu\n",
           (unsigned long long)ftgc_n, (unsigned long long)ftgc_bad);
    printf("  native %s n=%llu mismatches=%llu nan-payload-only=%llu  (informational)\n",
           ftz_daz ? "x86 addss/mulss, MXCSR FTZ|DAZ, all pairs"
                   : "+,* default FP env (no tiny/special)",
           (unsigned long long)compared, (unsigned long long)native_bad,
           (unsigned long long)nan_payload);
    return add_bad + mul_bad + ftgc_bad;
}

/* ---------------------------------------------------------------- gate composites */

typedef struct CompositeRun {
    const DetMathArm *arm;
    uint64_t samples;
    uint64_t seed;
    ErrorStats g_given_add, g_end_to_end, decay_given_g, decay_end_to_end, beta;
} CompositeRun;

static void *run_composite_thread(void *argument)
{
    CompositeRun *run = argument;
    uint64_t state = run->seed, index;

    for (index = 0; index < run->samples; index++) {
        const float alpha = (float)random_uniform(&state, -104.0, 24.0);
        const float dt = (float)random_uniform(&state, -2.0, 2.0);
        const float ssm_a = (float)-exp(random_uniform(&state, -6.0, 3.0));
        const float b = (float)random_uniform(&state, -104.0, 104.0);
        const uint32_t sum = run->arm->add(float_to_bits(alpha), float_to_bits(dt));
        const uint32_t g = run->arm->gdn_g(float_to_bits(alpha), float_to_bits(dt),
                                           float_to_bits(ssm_a));
        const double g_exact = (double)ssm_a * reference_softplus((double)alpha + (double)dt);
        const uint32_t decay = run->arm->gdn_decay(g);

        const uint32_t sp_fp32 = round_ftz(reference_softplus(daz_input(sum)));

        stats_add(&run->g_given_add,
                  ulp_distance(g, round_ftz((double)ssm_a * daz_input(sp_fp32))), 3u,
                  float_to_bits(alpha));
        if (sp_fp32 != 0u) /* a flushed softplus makes the all-double gap meaningless */
            stats_add(&run->g_end_to_end, ulp_distance(g, round_ftz(g_exact)), UINT64_MAX,
                      float_to_bits(alpha));
        stats_add(&run->decay_given_g, ulp_distance(decay, round_ftz(exp(daz_input(g)))), 1u, g);
        stats_add(&run->decay_end_to_end, ulp_distance(decay, round_ftz(exp(g_exact))),
                  UINT64_MAX, g);
        stats_add(&run->beta, ulp_distance(run->arm->gdn_beta(float_to_bits(b)),
                                           round_ftz(reference_sigmoid(daz_input(float_to_bits(b))))),
                  2u, float_to_bits(b));
    }
    return NULL;
}

static uint64_t run_composite_checks(const DetMathArm *arm, uint64_t samples)
{
    CompositeRun runs[THREADS];
    pthread_t threads[THREADS];
    CompositeRun total;
    unsigned t;

    memset(&total, 0, sizeof total);
    for (t = 0; t < THREADS; t++) {
        memset(&runs[t], 0, sizeof runs[t]);
        runs[t].arm = arm;
        runs[t].samples = samples / THREADS;
        runs[t].seed = 0x6A7E5ull + t * 0x9E3779B9ull;
        pthread_create(&threads[t], NULL, run_composite_thread, &runs[t]);
    }
    for (t = 0; t < THREADS; t++) {
        pthread_join(threads[t], NULL);
        stats_merge(&total.g_given_add, &runs[t].g_given_add);
        stats_merge(&total.g_end_to_end, &runs[t].g_end_to_end);
        stats_merge(&total.decay_given_g, &runs[t].decay_given_g);
        stats_merge(&total.decay_end_to_end, &runs[t].decay_end_to_end);
        stats_merge(&total.beta, &runs[t].beta);
    }
    stats_print("gdn_g vs fp32-step double ref", &total.g_given_add, 3u, 0);
    stats_print("gdn_g vs all-double ref", &total.g_end_to_end, 0u, 1);
    stats_print("gdn_decay vs exp(fp32 g)", &total.decay_given_g, 1u, 0);
    stats_print("gdn_decay vs double exp(g_exact)", &total.decay_end_to_end, 0u, 1);
    stats_print("gdn_beta (fast) vs double sigmoid", &total.beta, 2u, 0);
    return total.g_given_add.failures + total.decay_given_g.failures + total.beta.failures;
}

/* ---------------------------------------------------------------- tables + edge cases */

static uint64_t check_one_table(const char *name, const uint32_t *table, int which)
{
    uint64_t failures = 0;
    int i;

    for (i = 0; i < 16; i++) {
        double value = 0.0, recip = 1.0 / (1.0 + (i + 0.5) / 16.0);

        if (which == 0)
            value = (exp2(i / 16.0) - 1.0) * 0x1p32;
        else if (which == 1)
            value = recip * 0x1p32;
        else if (which == 2)
            value = -log((double)ft_det_log_recip_table[i] * 0x1p-32) * 0x1p32;
        else
            value = -log2((double)ft_det_log_recip_table[i] * 0x1p-32) * 0x1p31;
        if (fabs(value - (double)table[i]) > 1.0) {
            printf("  FAIL table %s[%d] = 0x%08X, want %.1f\n", name, i, table[i], value);
            failures++;
        }
    }
    return failures;
}

static uint64_t check_tables(void)
{
    const uint64_t failures = check_one_table("exp2", ft_det_exp2_table, 0) +
                              check_one_table("log_recip", ft_det_log_recip_table, 1) +
                              check_one_table("log_neg", ft_det_log_neg_table, 2) +
                              check_one_table("log2_neg", ft_det_log2_neg_table, 3);

    printf("  tables recomputed in double: %llu failures\n", (unsigned long long)failures);
    return failures;
}

typedef enum { kOpAdd, kOpMul, kOpExp, kOpSoftplus, kOpSigmoid, kOpBeta } EdgeOp;

typedef struct EdgeCase {
    EdgeOp op;
    uint32_t a, b;
    uint32_t want;
    const char *what;
} EdgeCase;

static const EdgeCase kEdgeCases[] = {
    {kOpExp, 0x00000000u, 0, 0x3F800000u, "exp(+0) = 1"},
    {kOpExp, 0x80000000u, 0, 0x3F800000u, "exp(-0) = 1"},
    {kOpExp, 0x00000001u, 0, 0x3F800000u, "exp(denormal) = 1 (DAZ)"},
    {kOpExp, 0x807FFFFFu, 0, 0x3F800000u, "exp(-denormal) = 1 (DAZ)"},
    {kOpExp, 0x7F800000u, 0, 0x7F800000u, "exp(+Inf) = +Inf"},
    {kOpExp, 0xFF800000u, 0, 0x00000000u, "exp(-Inf) = +0"},
    {kOpExp, 0x7FC00000u, 0, 0x7FC00000u, "exp(qNaN) = qNaN"},
    {kOpExp, 0xFF800001u, 0, 0xFFC00001u, "exp(sNaN) = quieted"},
    {kOpExp, 0x43000000u, 0, 0x7F800000u, "exp(128) = +Inf"},
    {kOpExp, 0xC3000000u, 0, 0x00000000u, "exp(-128) = +0"},
    {kOpExp, 0x3F800000u, 0, 0x402DF854u, "exp(1) = e"},
    {kOpSoftplus, 0x41A00000u, 0, 0x41A00000u, "softplus(20) = 20 (log1p branch)"},
    {kOpSoftplus, 0x41A00001u, 0, 0x41A00001u, "softplus(next(20)) = x"},
    {kOpSoftplus, 0x419FFFFFu, 0, 0x419FFFFFu, "softplus(prev(20)) = x"},
    {kOpSoftplus, 0x7F7FFFFFu, 0, 0x7F7FFFFFu, "softplus(FLT_MAX) = x"},
    {kOpSoftplus, 0x00000000u, 0, 0x3F317218u, "softplus(+0) = ln 2"},
    {kOpSoftplus, 0x80000000u, 0, 0x3F317218u, "softplus(-0) = ln 2"},
    {kOpSoftplus, 0x00400000u, 0, 0x3F317218u, "softplus(denormal) = ln 2 (DAZ)"},
    {kOpSoftplus, 0x7F800000u, 0, 0x7F800000u, "softplus(+Inf) = +Inf"},
    {kOpSoftplus, 0xFF800000u, 0, 0x00000000u, "softplus(-Inf) = +0"},
    {kOpSoftplus, 0xFFC00000u, 0, 0xFFC00000u, "softplus(NaN) = NaN"},
    {kOpSoftplus, 0xC2D00000u, 0, 0x00000000u, "softplus(-104) = +0 (FTZ)"},
    {kOpSigmoid, 0x00000000u, 0, 0x3F000000u, "sigmoid(+0) = 1/2"},
    {kOpSigmoid, 0x80000001u, 0, 0x3F000000u, "sigmoid(-denormal) = 1/2"},
    {kOpSigmoid, 0x7F800000u, 0, 0x3F800000u, "sigmoid(+Inf) = 1"},
    {kOpSigmoid, 0xFF800000u, 0, 0x00000000u, "sigmoid(-Inf) = +0"},
    {kOpSigmoid, 0x7FA00000u, 0, 0x7FE00000u, "sigmoid(sNaN) = quieted"},
    {kOpSigmoid, 0x418B0000u, 0, 0x3F800000u, "sigmoid(17.375) = 1"},
    {kOpSigmoid, 0xC2D00000u, 0, 0x00000000u, "sigmoid(-104) = +0 (FTZ)"},
    {kOpBeta, 0x00000000u, 0, 0x3F000000u, "beta(+0) = 1/2"},
    {kOpBeta, 0x80000001u, 0, 0x3F000000u, "beta(-denormal) = 1/2"},
    {kOpBeta, 0x7F800000u, 0, 0x3F800000u, "beta(+Inf) = 1"},
    {kOpBeta, 0xFF800000u, 0, 0x00000000u, "beta(-Inf) = +0"},
    {kOpBeta, 0x7FA00000u, 0, 0x7FE00000u, "beta(sNaN) = quieted"},
    {kOpBeta, 0x41C00000u, 0, 0x3F800000u, "beta(24) = 1"},
    {kOpBeta, 0xC2D00000u, 0, 0x00000000u, "beta(-104) = +0 (FTZ)"},
    {kOpAdd, 0x00000000u, 0x80000000u, 0x00000000u, "+0 + -0 = +0"},
    {kOpAdd, 0x80000000u, 0x80000000u, 0x80000000u, "-0 + -0 = -0"},
    {kOpAdd, 0x3FC00000u, 0xBFC00000u, 0x00000000u, "x + -x = +0"},
    {kOpAdd, 0x80000001u, 0x80000002u, 0x80000000u, "-den + -den = -0 (DAZ)"},
    {kOpAdd, 0x00000001u, 0x3F800000u, 0x3F800000u, "den + 1 = 1 (DAZ)"},
    {kOpAdd, 0x00C00000u, 0x80800000u, 0x00000000u, "1.5m - m = 0.5m -> +0 (FTZ)"},
    {kOpAdd, 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F800000u, "FLT_MAX + FLT_MAX = +Inf"},
    {kOpAdd, 0x7F800000u, 0xFF800000u, 0xFFC00000u, "Inf - Inf = default NaN"},
    {kOpAdd, 0x7F800001u, 0x7FC00002u, 0x7FC00001u, "NaN + NaN = first, quieted"},
    {kOpAdd, 0x3F800000u, 0xFFC00003u, 0xFFC00003u, "1 + NaN = the NaN"},
    {kOpAdd, 0x3F800000u, 0x33800000u, 0x3F800000u, "1 + 2^-24 = 1 (tie to even)"},
    {kOpAdd, 0x3F800001u, 0x33800000u, 0x3F800002u, "(1+u) + 2^-24 rounds up (tie to even)"},
    {kOpMul, 0x00800000u, 0x3F000000u, 0x00000000u, "FLT_MIN * 0.5 = +0 (FTZ)"},
    {kOpMul, 0x3F7FFFFFu, 0x00800000u, 0x00000000u, "(1-2^-24) * FLT_MIN = +0 (tiny)"},
    {kOpMul, 0x00000001u, 0x40000000u, 0x00000000u, "den * 2 = +0 (DAZ)"},
    {kOpMul, 0x7F800000u, 0x00000000u, 0xFFC00000u, "Inf * 0 = default NaN"},
    {kOpMul, 0x7F800000u, 0x80000001u, 0xFFC00000u, "Inf * den = default NaN (DAZ)"},
    {kOpMul, 0xFF800000u, 0x40000000u, 0xFF800000u, "-Inf * 2 = -Inf"},
    {kOpMul, 0x7F7FFFFFu, 0x40000000u, 0x7F800000u, "FLT_MAX * 2 = +Inf"},
    {kOpMul, 0xBF800000u, 0x00000000u, 0x80000000u, "-1 * +0 = -0"},
    {kOpMul, 0x7FC00000u, 0x7F800001u, 0x7FC00000u, "qNaN * sNaN = first"},
};

static uint32_t apply_edge_op(const DetMathArm *arm, EdgeOp op, uint32_t a, uint32_t b)
{
    switch (op) {
    case kOpAdd: return arm->add(a, b);
    case kOpMul: return arm->mul(a, b);
    case kOpExp: return arm->exp(a);
    case kOpSoftplus: return arm->softplus(a);
    case kOpBeta: return arm->gdn_beta(a);
    default: return arm->sigmoid(a);
    }
}

/**
 * Walk x (bits growing away from zero) to the first input whose reference saturates (+Inf
 * going up, +0 going down); <= 1 ulp at edge-2..edge, exact saturation at edge+1, edge+2.
 */
static uint64_t check_exp_boundary(const DetMathArm *arm, uint32_t start, uint32_t saturated)
{
    uint64_t failures = 0;
    uint32_t edge = start;
    int k;

    while (round_ftz(exp(daz_input(edge))) != saturated)
        edge++;
    for (k = -2; k <= 2; k++) {
        const uint32_t probe = (uint32_t)((int64_t)edge + k);
        const uint32_t got = arm->exp(probe), want = round_ftz(exp(daz_input(probe)));

        if (ulp_distance(got, want) > (k <= 0 ? 1u : 0u)) {
            printf("  FAIL exp boundary x=0x%08X got 0x%08X want 0x%08X\n", probe, got, want);
            failures++;
        }
    }
    return failures;
}

/**
 * A product whose exact value lies in [2^-126 - 2^-151, 2^-126) is NOT tiny after rounding,
 * so x86 FTZ returns 2^-126, not 0. Find such a pair by search and check it.
 */
static uint64_t check_mul_rounds_up_to_normal(const DetMathArm *arm)
{
    uint32_t a;

    for (a = 0x3F800001u; a < 0x3F900000u; a++) {
        const uint32_t guess = float_to_bits((float)(1.0 / (double)bits_to_float(a)));
        uint32_t b;

        for (b = guess - 2u; b <= guess + 2u; b++) {
            const double product = (double)bits_to_float(a) * (double)bits_to_float(b);

            if (product < 1.0 && product >= 1.0 - 0x1p-25) {
                const uint32_t tiny_a = (a & 0x807FFFFFu) | (1u << 23); /* a * 2^-126 */
                const uint32_t got = arm->mul(tiny_a, b);

                if (got != 0x00800000u || reference_binary(tiny_a, b, 1) != 0x00800000u) {
                    printf("  FAIL mul 0x%08X * 0x%08X got 0x%08X want 0x00800000\n", tiny_a,
                           b, got);
                    return 1u;
                }
                return 0u;
            }
        }
    }
    printf("  FAIL no round-up-to-normal mul pair found\n");
    return 1u;
}

typedef struct GateEdgeCase {
    uint32_t alpha, dt, ssm_a, want;
    const char *what;
} GateEdgeCase;

static const GateEdgeCase kGateEdgeCases[] = {
    {0x00000000u, 0x00000000u, 0xBF800000u, 0x3F000000u, "a=-1, x=0: 2^-1"},
    {0x7F800000u, 0x3F800000u, 0xBF800000u, 0x00000000u, "x=+Inf, a<0: 0"},
    {0xFF800000u, 0x3F800000u, 0xBF800000u, 0x3F800000u, "x=-Inf: 1"},
    {0x7F800000u, 0xFF800000u, 0xBF800000u, 0xFFC00000u, "Inf - Inf: default NaN"},
    {0x7FC00001u, 0x3F800000u, 0xBF800000u, 0x7FC00001u, "NaN alpha: quieted"},
    {0x3F800000u, 0x7F800001u, 0xBF800000u, 0x7FC00001u, "sNaN dt: quieted"},
    {0x3F800000u, 0x00000000u, 0x80000000u, 0x3F800000u, "a=-0: 1"},
    {0x3F800000u, 0x00000000u, 0x80000001u, 0x3F800000u, "a=-denormal: 1 (DAZ)"},
    {0x7F800000u, 0x00000000u, 0x00000000u, 0xFFC00000u, "x=+Inf, a=0: NaN"},
    {0x3F800000u, 0x00000000u, 0xFF800000u, 0x00000000u, "a=-Inf: 0"},
    {0x7F7F0000u, 0x00000000u, 0xBA83126Fu, 0x00000000u, "x=bf16 max, a=-1e-3: 0"},
    {0xFF7F0000u, 0x41A00000u, 0xC2800000u, 0x3F800000u, "x=-bf16 max, a=-64: 1"},
    {0x3F800000u, 0xBF800000u, 0xC2800000u, 0x1F800000u, "x=0 exactly, a=-64: 2^-64"},
};

/** The fast gate's specials (exact bits) -- see ft_det_fast_gate_special. */
static uint64_t check_gate_edge_cases(const DetMathArm *arm)
{
    uint64_t failures = 0;
    size_t i;

    for (i = 0; i < sizeof kGateEdgeCases / sizeof kGateEdgeCases[0]; i++) {
        const GateEdgeCase *edge = &kGateEdgeCases[i];
        const uint32_t got = arm->gdn_gate(edge->alpha, edge->dt, edge->ssm_a);

        if (got != edge->want) {
            printf("  FAIL gate edge %-34s got 0x%08X want 0x%08X\n", edge->what, got,
                   edge->want);
            failures++;
        }
    }
    return failures;
}

static uint64_t check_edge_cases(const DetMathArm *arm)
{
    uint64_t failures = 0;
    size_t i;

    for (i = 0; i < sizeof kEdgeCases / sizeof kEdgeCases[0]; i++) {
        const EdgeCase *edge = &kEdgeCases[i];
        const uint32_t got = apply_edge_op(arm, edge->op, edge->a, edge->b);

        if (got != edge->want) {
            printf("  FAIL edge %-40s got 0x%08X want 0x%08X\n", edge->what, got, edge->want);
            failures++;
        }
    }
    failures += check_exp_boundary(arm, float_to_bits(88.0f), 0x7F800000u);
    failures += check_exp_boundary(arm, float_to_bits(-87.0f), 0u);
    failures += check_mul_rounds_up_to_normal(arm);
    failures += check_gate_edge_cases(arm);
    printf("  %zu edge cases + exp boundaries + mul round-up-to-normal: %llu failures\n",
           sizeof kEdgeCases / sizeof kEdgeCases[0], (unsigned long long)failures);
    return failures;
}

/* ---------------------------------------------------------------- fast path */

#define GATE_PAIRS 4096u

/** softplus in double, stable for any x (huge alpha must not overflow exp). */
static double reference_softplus_stable(double x)
{
    return (x > 0.0 ? x : 0.0) + log1p(exp(-fabs(x)));
}

typedef struct GateRun {
    const DetMathArm *arm;
    const uint32_t *dt, *ssm_a;
    uint32_t pairs;
    unsigned thread_index;
    ErrorStats stats;
    uint32_t worst_dt, worst_a;
} GateRun;

static void *run_gate_thread(void *argument)
{
    GateRun *run = argument;
    uint32_t high, pair;

    for (high = run->thread_index; high < 65536u; high += THREADS) {
        const uint32_t alpha = high << 16;

        for (pair = 0; pair < run->pairs; pair++) {
            const double x = daz_input(alpha) + daz_input(run->dt[pair]);
            const double g = daz_input(run->ssm_a[pair]) * reference_softplus_stable(x);
            const uint32_t want = round_ftz(exp(g));
            const uint64_t ulp =
                ulp_distance(run->arm->gdn_gate(alpha, run->dt[pair], run->ssm_a[pair]), want);
            const uint64_t before = run->stats.max_ulp;

            stats_add(&run->stats, ulp, 4u, alpha);
            if (run->stats.max_ulp != before || run->stats.count == 1u) {
                run->worst_dt = run->dt[pair];
                run->worst_a = run->ssm_a[pair];
            }
        }
    }
    return NULL;
}

/** ft_det_gdn_gate: every bf16 alpha x `pairs` random (dt, ssm_a), <= 4 ulp. */
static uint64_t run_gate_checks(const DetMathArm *arm, uint32_t pairs)
{
    static uint32_t dt[GATE_PAIRS], ssm_a[GATE_PAIRS];
    GateRun runs[THREADS];
    pthread_t threads[THREADS];
    ErrorStats total;
    uint64_t state = 0x6A7Eull;
    uint32_t worst_dt = 0, worst_a = 0, pair;
    unsigned t;

    for (pair = 0; pair < GATE_PAIRS; pair++) {
        dt[pair] = float_to_bits((float)random_uniform(&state, -20.0, 20.0));
        ssm_a[pair] = float_to_bits((float)-exp(random_uniform(&state, log(1e-3), log(64.0))));
    }
    memset(&total, 0, sizeof total);
    for (t = 0; t < THREADS; t++) {
        memset(&runs[t], 0, sizeof runs[t]);
        runs[t].arm = arm;
        runs[t].dt = dt;
        runs[t].ssm_a = ssm_a;
        runs[t].pairs = pairs;
        runs[t].thread_index = t;
        pthread_create(&threads[t], NULL, run_gate_thread, &runs[t]);
    }
    for (t = 0; t < THREADS; t++) {
        pthread_join(threads[t], NULL);
        if (runs[t].stats.max_ulp >= total.max_ulp) {
            worst_dt = runs[t].worst_dt;
            worst_a = runs[t].worst_a;
        }
        stats_merge(&total, &runs[t].stats);
    }
    stats_print("gdn_gate (fast) all bf16 alpha", &total, 4u, 0);
    printf("  %-34s worst alpha=0x%08X dt=0x%08X ssm_a=0x%08X\n", "", total.worst_input,
           worst_dt, worst_a);
    return total.failures;
}

/** ft_det_gdn_beta on all 65,536 bf16 patterns (NaN/Inf included), <= 2 ulp. */
static uint64_t run_beta_bf16_check(const DetMathArm *arm)
{
    ErrorStats stats;
    uint32_t high;

    memset(&stats, 0, sizeof stats);
    for (high = 0; high < 65536u; high++) {
        const uint32_t b = high << 16;

        stats_add(&stats, ulp_distance(arm->gdn_beta(b), round_ftz(reference_sigmoid(daz_input(b)))),
                  2u, b);
    }
    stats_print("gdn_beta (fast) all bf16", &stats, 2u, 0);
    return stats.failures;
}

/* ---------------------------------------------------------------- one full pass */

/** Every check against one arm. stride > 1 (or a reduced sample count) is for quick runs. */
static uint64_t run_all_checks(const DetMathArm *arm, uint32_t stride, uint64_t binary_samples,
                               uint64_t composite_samples)
{
    uint64_t failures = 0;

    printf("[%s] stride=%u\n", arm->name, stride);
    failures += check_tables();
    failures += check_edge_cases(arm);
    failures += run_unary_sweep("exp [-103.9, 88.7]", arm->exp, reference_exp, -103.9f, 88.7f,
                                1u, stride);
    failures += run_unary_sweep("softplus [-104, 20]", arm->softplus, reference_softplus,
                                -104.0f, 20.0f, 2u, stride);
    failures += run_unary_sweep("sigmoid [-104, 104]", arm->sigmoid, reference_sigmoid, -104.0f,
                                104.0f, 1u, stride);
    failures += run_beta_bf16_check(arm);
    failures += run_unary_sweep("gdn_beta (fast) [-104, 104]", arm->gdn_beta, reference_sigmoid,
                                -104.0f, 104.0f, 2u, stride);
    failures += run_gate_checks(arm, stride > 1u ? 64u : GATE_PAIRS);
    failures += run_composite_checks(arm, composite_samples);
    failures += run_binary_checks(arm, binary_samples);
    printf("[%s] %llu failures\n", arm->name, (unsigned long long)failures);
    return failures;
}

/* ---------------------------------------------------------------- emit */

static void write_word(FILE *file, uint32_t word)
{
    const unsigned char bytes[4] = {(unsigned char)word, (unsigned char)(word >> 8),
                                    (unsigned char)(word >> 16), (unsigned char)(word >> 24)};

    fwrite(bytes, 1, sizeof bytes, file);
}

static void write_record(FILE *file, const DetMathArm *arm, uint32_t in0, uint32_t in1,
                         uint32_t in2)
{
    const uint32_t words[13] = {in0,
                                in1,
                                in2,
                                arm->add(in0, in1),
                                arm->mul(in0, in1),
                                arm->exp(in0),
                                arm->softplus(in0),
                                arm->sigmoid(in0),
                                arm->gdn_beta_reference(in0),
                                arm->gdn_g(in0, in1, in2),
                                arm->gdn_decay(in0),
                                arm->gdn_beta(in0),
                                arm->gdn_gate(in0 & 0xFFFF0000u, in1, in2)};
    int i;

    for (i = 0; i < 13; i++)
        write_word(file, words[i]);
}

/** Write `count` records (edge-case inputs first) to `path`. Returns 0 on success. */
static int emit_vectors(const char *path, uint32_t count, uint64_t seed)
{
    const DetMathArm *arm = det_math_arm_production();
    const size_t edges = sizeof kEdgeCases / sizeof kEdgeCases[0];
    FILE *file = fopen(path, "wb");
    uint64_t state = seed;
    uint32_t index;

    if (file == NULL) {
        fprintf(stderr, "emit: cannot open %s\n", path);
        return 1;
    }
    write_word(file, 0x4D445446u); /* "FTDM" little-endian */
    write_word(file, 2u);
    write_word(file, count);
    write_word(file, 13u);
    for (index = 0; index < count; index++) {
        const uint32_t dt = float_to_bits((float)random_uniform(&state, -8.0, 8.0));
        const uint32_t ssm_a = float_to_bits((float)-exp(random_uniform(&state, -6.0, 3.0)));
        uint32_t in0;

        if (index < edges)
            in0 = kEdgeCases[index].a;
        else if (index & 1u)
            in0 = float_to_bits((float)random_uniform(&state, -104.0, 104.0));
        else
            for (in0 = (uint32_t)splitmix64(&state); (in0 & 0x7FFFFFFFu) > 0x42D00000u;)
                in0 = (uint32_t)splitmix64(&state);
        write_record(file, arm, in0, index < edges ? kEdgeCases[index].b : dt, ssm_a);
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "emit: write to %s failed\n", path);
        return 1;
    }
    printf("emit: %u records (%zu edge) -> %s\n", count, edges < count ? edges : count, path);
    return 0;
}

/* ---------------------------------------------------------------- main */

/** Each mutated arm must fail; returns the total failure count (non-zero is the pass). */
static uint64_t run_red_arms(void)
{
    const DetMathArm *arms[4];
    uint64_t total = 0;
    int i;

    arms[0] = det_math_arm_red_round_half();
    arms[1] = det_math_arm_red_exp_coefficient();
    arms[2] = det_math_arm_red_fast_log1p();
    arms[3] = det_math_arm_red_fast_newton();
    for (i = 0; i < 4; i++) {
        const uint64_t failures = run_all_checks(arms[i], RED_STRIDE, RED_SAMPLES, RED_SAMPLES);

        printf("RED ARM %-16s %llu failures%s\n", arms[i]->name, (unsigned long long)failures,
               failures ? "" : "  <-- VACUOUS: the mutation was not detected");
        total += failures;
        if (failures == 0u)
            return 0u;
    }
    return total;
}

int main(int argc, char **argv)
{
    uint32_t stride = 1u;
    uint64_t failures;

    if (argc == 5 && strcmp(argv[1], "--emit") == 0)
        return emit_vectors(argv[2], (uint32_t)strtoul(argv[3], NULL, 0),
                            strtoull(argv[4], NULL, 0));
    if (argc == 2 && strcmp(argv[1], "--red") == 0) {
        failures = run_red_arms();
        printf("RED: %llu failures (non-zero exit expected)\n", (unsigned long long)failures);
        return failures ? 1 : 0;
    }
    if (argc == 2 && strcmp(argv[1], "--binary") == 0) {
        const DetMathArm *arm = det_math_arm_production();

        failures = check_tables() + check_edge_cases(arm) + run_binary_checks(arm, SAMPLES_BINARY);
        printf("%s: %llu failures\n", failures ? "FAIL" : "PASS", (unsigned long long)failures);
        return failures ? 1 : 0;
    }
    if (argc == 3 && strcmp(argv[1], "--stride") == 0)
        stride = (uint32_t)strtoul(argv[2], NULL, 0);
    else if (argc != 1) {
        fprintf(stderr, "usage: %s [--stride N | --red | --binary | --emit FILE N SEED]\n",
                argv[0]);
        return 2;
    }
    if (stride == 0u)
        stride = 1u;
    failures = run_all_checks(det_math_arm_production(), stride, SAMPLES_BINARY,
                              stride > 1u ? SAMPLES_COMPOSITE / stride : SAMPLES_COMPOSITE);
    printf("%s: %llu failures\n", failures ? "FAIL" : "PASS", (unsigned long long)failures);
    return failures ? 1 : 0;
}
