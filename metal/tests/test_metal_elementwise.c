/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_metal_elementwise.c — elementwise family (fm_silu, fm_sigmoid,
 * fm_add, fm_mul and their float4 _4 forms)
 *
 * Every assertion is against a CPU reference computed here. The float4
 * forms are NOT compared against the scalar GPU kernels: a bug shared by
 * both — a wrong activation, say — would survive that comparison, which is
 * exactly the failure mode this repo has been bitten by before.
 *
 * Each op is exercised at a length that is a multiple of 4 and at one that
 * is not, so the _4 kernels' scalar tail path is covered.
 *
 * Helper functions are named case_* so lib/test.h's failure line, which
 * prints __func__ + 5, still reads as the case that failed.
 *
 * Exits 77 when no Metal device is present so GPU-less CI skips cleanly.
 */

#include "lib/test.h"
#include "metal/fm_metal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* fast::exp on the GPU vs expf() on the host: measured on M4 / macOS 26 the
 * difference is sub-ULP, so this band is tight enough that any arithmetic
 * mistake shows up rather than being absorbed. add/mul are in fact
 * bit-exact. */
#define ELEM_TOL_ABS 1e-7f
#define ELEM_TOL_REL 1e-6f

/* Lengths: a multiple of 4, and one that leaves a 3-element tail. */
#define ELEM_N_ALIGNED 1024
#define ELEM_N_TAIL    1027
#define ELEM_N_MAX     ELEM_N_TAIL

typedef void (*unary_kernel)(fm_buffer *x, fm_buffer *out, int n);
typedef void (*binary_kernel)(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n);

static uint32_t elem_rng = 0x9e3779b9u;

static float next_random_float(void) {
    elem_rng ^= elem_rng << 13;
    elem_rng ^= elem_rng >> 17;
    elem_rng ^= elem_rng << 5;
    return (float)((int32_t)elem_rng) / 2147483648.0f;
}

static float reference_sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }
static float reference_silu(float v)    { return v / (1.0f + expf(-v)); }
static float reference_add(float a, float b) { return a + b; }
static float reference_mul(float a, float b) { return a * b; }

static fm_buffer *upload_floats(const float *src, size_t n) {
    fm_buffer *buf = fm_buffer_alloc(n * sizeof(float));
    if (buf) memcpy(fm_buffer_data(buf), src, n * sizeof(float));
    return buf;
}

/* Diffs against the reference. `label` is printed just before the assertion
 * fires, because lib/test.h names the failing function and every case here
 * shares this one — without it a failure would not say which. */
static void case_compare(const char *label, const float *expected,
                         const float *got, int n) {
    for (int i = 0; i < n; i++) {
        float tol = ELEM_TOL_ABS + ELEM_TOL_REL * fabsf(expected[i]);
        if (fabsf(expected[i] - got[i]) > tol)
            printf("    %s: element %d of %d\n", label, i, n);
        ASSERT_FLT_NEAR(expected[i], got[i], tol);
    }
}

/* Runs one unary kernel over n random inputs and diffs against the CPU
 * reference. The output buffer is pre-filled with a sentinel so an element
 * the kernel never writes fails loudly instead of matching by luck. */
static void case_unary(const char *label, unary_kernel kernel,
                       float (*reference)(float), int n) {
    float input[ELEM_N_MAX], expected[ELEM_N_MAX];
    for (int i = 0; i < n; i++) {
        input[i] = next_random_float() * 8.0f;
        expected[i] = reference(input[i]);
    }

    fm_buffer *bx = upload_floats(input, (size_t)n);
    fm_buffer *bo = fm_buffer_alloc((size_t)n * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bo);
    float *out = (float *)fm_buffer_data(bo);
    for (int i = 0; i < n; i++) out[i] = -12345.0f;

    kernel(bx, bo, n);
    fm_sync();

    case_compare(label, expected, out, n);

    fm_buffer_free(bx);
    fm_buffer_free(bo);
}

static void case_binary(const char *label, binary_kernel kernel,
                        float (*reference)(float, float), int n) {
    float a[ELEM_N_MAX], b[ELEM_N_MAX], expected[ELEM_N_MAX];
    for (int i = 0; i < n; i++) {
        a[i] = next_random_float() * 6.0f;
        b[i] = next_random_float() * 6.0f;
        expected[i] = reference(a[i], b[i]);
    }

    fm_buffer *ba = upload_floats(a, (size_t)n);
    fm_buffer *bb = upload_floats(b, (size_t)n);
    fm_buffer *bo = fm_buffer_alloc((size_t)n * sizeof(float));
    ASSERT_NOT_NULL(ba);
    ASSERT_NOT_NULL(bb);
    ASSERT_NOT_NULL(bo);
    float *out = (float *)fm_buffer_data(bo);
    for (int i = 0; i < n; i++) out[i] = -12345.0f;

    kernel(ba, bb, bo, n);
    fm_sync();

    case_compare(label, expected, out, n);

    fm_buffer_free(ba);
    fm_buffer_free(bb);
    fm_buffer_free(bo);
}

/* Each op runs at both lengths: the aligned one, then the one with a tail. */
#define RUN_UNARY_BOTH_LENGTHS(fn, reference)                              \
    do {                                                                   \
        case_unary(#fn " n=aligned", fn, reference, ELEM_N_ALIGNED);       \
        if (__ft_current_failed) return;                                   \
        case_unary(#fn " n=with-tail", fn, reference, ELEM_N_TAIL);        \
    } while (0)

#define RUN_BINARY_BOTH_LENGTHS(fn, reference)                             \
    do {                                                                   \
        case_binary(#fn " n=aligned", fn, reference, ELEM_N_ALIGNED);      \
        if (__ft_current_failed) return;                                   \
        case_binary(#fn " n=with-tail", fn, reference, ELEM_N_TAIL);       \
    } while (0)

static void test_silu_matches_cpu_reference(void) {
    RUN_UNARY_BOTH_LENGTHS(fm_silu, reference_silu);
}

static void test_silu_4_matches_cpu_reference(void) {
    RUN_UNARY_BOTH_LENGTHS(fm_silu_4, reference_silu);
}

static void test_sigmoid_matches_cpu_reference(void) {
    RUN_UNARY_BOTH_LENGTHS(fm_sigmoid, reference_sigmoid);
}

static void test_sigmoid_4_matches_cpu_reference(void) {
    RUN_UNARY_BOTH_LENGTHS(fm_sigmoid_4, reference_sigmoid);
}

static void test_add_matches_cpu_reference(void) {
    RUN_BINARY_BOTH_LENGTHS(fm_add, reference_add);
}

static void test_add_4_matches_cpu_reference(void) {
    RUN_BINARY_BOTH_LENGTHS(fm_add_4, reference_add);
}

static void test_mul_matches_cpu_reference(void) {
    RUN_BINARY_BOTH_LENGTHS(fm_mul, reference_mul);
}

static void test_mul_4_matches_cpu_reference(void) {
    RUN_BINARY_BOTH_LENGTHS(fm_mul_4, reference_mul);
}

/* The _4 kernels must not scribble past element n-1: the grid is rounded up
 * to ceil(n/4), so the final thread owns a partial float4. */
static void test_vec4_kernels_do_not_write_past_n(void) {
    const int n = ELEM_N_TAIL;
    const int slack = 4;
    float input[ELEM_N_MAX];
    for (int i = 0; i < n; i++) input[i] = next_random_float();

    fm_buffer *bx = upload_floats(input, (size_t)n);
    fm_buffer *bo = fm_buffer_alloc((size_t)(n + slack) * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bo);
    float *out = (float *)fm_buffer_data(bo);
    for (int i = 0; i < n + slack; i++) out[i] = -12345.0f;

    fm_silu_4(bx, bo, n);
    fm_sync();

    for (int i = n; i < n + slack; i++)
        ASSERT_FLT_NEAR(-12345.0f, out[i], 0.0f);

    fm_buffer_free(bx);
    fm_buffer_free(bo);
}

int main(void) {
    printf("\n  Metal Elementwise Kernel Tests\n");
    printf("  =============================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "  Metal unavailable — skipping.\n\n");
        return 77;
    }

    RUN_TEST(silu_matches_cpu_reference);
    RUN_TEST(silu_4_matches_cpu_reference);
    RUN_TEST(sigmoid_matches_cpu_reference);
    RUN_TEST(sigmoid_4_matches_cpu_reference);
    RUN_TEST(add_matches_cpu_reference);
    RUN_TEST(add_4_matches_cpu_reference);
    RUN_TEST(mul_matches_cpu_reference);
    RUN_TEST(mul_4_matches_cpu_reference);
    RUN_TEST(vec4_kernels_do_not_write_past_n);

    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
