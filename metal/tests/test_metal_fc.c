/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_metal_fc.c
 * @brief Tests for function-constant pipeline specialization in the Metal
 *   backend: the value-set helpers, the (name + constants) pipeline cache,
 *   and proof that a specialized constant actually reaches the shader.
 *
 * Exits 77 when no Metal device is present so GPU-less CI skips cleanly.
 *
 * @status All tests pass on M4 / macOS 26; each was proven able to fail.
 * @issues None
 * @todo None
 */

#include "lib/test.h"
#include "metal/fm_metal.h"

#include <stdio.h>

/* out[i] the constant_probe kernel is contracted to write, per fm_metal.h. */
static float expected_probe_value(int repeat_count, int scale, bool flag, int i)
{
    float sum = 0.0f;
    for (int n = 1; n <= repeat_count; n++)
        sum += (float)n;
    return sum * (float)scale + (flag ? 1000.0f : 0.0f) + (float)i;
}

/* Runs constant_probe over `count` elements and returns the GPU buffer's
 * contents.  Caller frees via fm_buffer_free. */
static fm_buffer *run_probe(const fm_fc_values *constants, int count)
{
    fm_buffer *out = fm_buffer_alloc((size_t)count * sizeof(float));
    if (!out) return NULL;
    fm_constant_probe(out, count, constants);
    fm_sync();
    return out;
}

/* ================================================================
 * Constant-set value type
 * ================================================================ */

static void test_fc_set_rejects_null_and_overflow(void)
{
    fm_fc_values values;
    fm_fc_reset(&values);

    ASSERT_INT_EQ(-1, fm_fc_set_int(NULL, 0, 1));

    for (unsigned i = 0; i < FM_FC_MAX_VALUES; i++)
        ASSERT_INT_EQ(0, fm_fc_set_int(&values, i, (int32_t)i));
    ASSERT_INT_EQ(FM_FC_MAX_VALUES, values.count);

    /* Full set: a new index is refused, an existing one still overwrites. */
    ASSERT_INT_EQ(-1, fm_fc_set_int(&values, FM_FC_MAX_VALUES, 99));
    ASSERT_INT_EQ(0, fm_fc_set_int(&values, 0, 42));
    ASSERT_INT_EQ(FM_FC_MAX_VALUES, values.count);
}

static void test_fc_set_is_order_independent(void)
{
    fm_fc_values ascending, descending;
    fm_fc_reset(&ascending);
    fm_fc_reset(&descending);

    fm_fc_set_int(&ascending, FM_FC_PROBE_REPEAT_COUNT, 4);
    fm_fc_set_short(&ascending, FM_FC_PROBE_SCALE, 3);
    fm_fc_set_bool(&ascending, FM_FC_PROBE_FLAG, true);

    fm_fc_set_bool(&descending, FM_FC_PROBE_FLAG, true);
    fm_fc_set_short(&descending, FM_FC_PROBE_SCALE, 3);
    fm_fc_set_int(&descending, FM_FC_PROBE_REPEAT_COUNT, 4);

    ASSERT_INT_EQ(ascending.count, descending.count);
    for (int i = 0; i < ascending.count; i++) {
        ASSERT_INT_EQ(ascending.index[i], descending.index[i]);
        ASSERT_INT_EQ(ascending.type[i],  descending.type[i]);
        ASSERT_INT_EQ(ascending.value[i], descending.value[i]);
    }

    /* Same values set in either order must hit the same cached pipeline. */
    const void *from_ascending  = fm_pipeline_for_constants("constant_probe", &ascending);
    const void *from_descending = fm_pipeline_for_constants("constant_probe", &descending);
    ASSERT_NOT_NULL(from_ascending);
    ASSERT_TRUE(from_ascending == from_descending);
}

/* ================================================================
 * Pipeline cache keyed on (name + constant values)
 * ================================================================ */

static void test_identical_constants_reuse_one_pipeline(void)
{
    fm_fc_values first, second;
    fm_fc_reset(&first);
    fm_fc_reset(&second);
    fm_fc_set_int(&first,  FM_FC_PROBE_REPEAT_COUNT, 7);
    fm_fc_set_int(&second, FM_FC_PROBE_REPEAT_COUNT, 7);

    const void *a = fm_pipeline_for_constants("constant_probe", &first);
    const void *b = fm_pipeline_for_constants("constant_probe", &second);
    const void *c = fm_pipeline_for_constants("constant_probe", &first);

    ASSERT_NOT_NULL(a);
    ASSERT_TRUE(a == b);
    ASSERT_TRUE(a == c);
}

static void test_distinct_constants_give_distinct_pipelines(void)
{
    fm_fc_values repeat_3, repeat_4, repeat_3_scaled;
    fm_fc_reset(&repeat_3);
    fm_fc_reset(&repeat_4);
    fm_fc_reset(&repeat_3_scaled);

    fm_fc_set_int(&repeat_3, FM_FC_PROBE_REPEAT_COUNT, 3);
    fm_fc_set_int(&repeat_4, FM_FC_PROBE_REPEAT_COUNT, 4);
    fm_fc_set_int(&repeat_3_scaled, FM_FC_PROBE_REPEAT_COUNT, 3);
    fm_fc_set_short(&repeat_3_scaled, FM_FC_PROBE_SCALE, 2);

    const void *p3        = fm_pipeline_for_constants("constant_probe", &repeat_3);
    const void *p4        = fm_pipeline_for_constants("constant_probe", &repeat_4);
    const void *p3_scaled = fm_pipeline_for_constants("constant_probe", &repeat_3_scaled);
    const void *plain     = fm_pipeline_for_constants("constant_probe", NULL);

    ASSERT_NOT_NULL(p3);
    ASSERT_NOT_NULL(p4);
    ASSERT_NOT_NULL(p3_scaled);
    ASSERT_NOT_NULL(plain);

    /* A different value, an extra constant, and no constants at all must
     * each be their own pipeline. */
    ASSERT_TRUE(p3 != p4);
    ASSERT_TRUE(p3 != p3_scaled);
    ASSERT_TRUE(p3 != plain);
    ASSERT_TRUE(p4 != p3_scaled);
    ASSERT_TRUE(p4 != plain);
    ASSERT_TRUE(p3_scaled != plain);
}

static void test_unknown_kernel_name_returns_null(void)
{
    ASSERT_NULL(fm_pipeline_for_constants("no_such_kernel_exists", NULL));
}

/* ================================================================
 * The specialized constants actually reach the shader
 * ================================================================ */

static void test_unspecialized_probe_uses_shader_defaults(void)
{
    int count = 8;
    fm_buffer *out = run_probe(NULL, count);
    ASSERT_NOT_NULL(out);

    /* Defaults: repeat 0, scale 1, flag false -> out[i] == i. */
    float *result = (float *)fm_buffer_data(out);
    for (int i = 0; i < count; i++)
        ASSERT_FLT_NEAR(expected_probe_value(0, 1, false, i), result[i], 1e-4f);

    fm_buffer_free(out);
}

static void test_probe_observes_int_constant(void)
{
    int count = 8;
    int repeat = 5;

    fm_fc_values constants;
    fm_fc_reset(&constants);
    fm_fc_set_int(&constants, FM_FC_PROBE_REPEAT_COUNT, repeat);

    fm_buffer *out = run_probe(&constants, count);
    ASSERT_NOT_NULL(out);

    /* sum(1..5) == 15, so out[i] == 15 + i — a value the unspecialized
     * pipeline (out[i] == i) cannot produce. */
    float *result = (float *)fm_buffer_data(out);
    for (int i = 0; i < count; i++)
        ASSERT_FLT_NEAR(expected_probe_value(repeat, 1, false, i), result[i], 1e-4f);

    fm_buffer_free(out);
}

static void test_probe_observes_all_three_constant_types(void)
{
    int count = 16;
    int repeat = 6;
    int scale = 3;

    fm_fc_values constants;
    fm_fc_reset(&constants);
    fm_fc_set_int(&constants, FM_FC_PROBE_REPEAT_COUNT, repeat);
    fm_fc_set_short(&constants, FM_FC_PROBE_SCALE, (int16_t)scale);
    fm_fc_set_bool(&constants, FM_FC_PROBE_FLAG, true);

    fm_buffer *out = run_probe(&constants, count);
    ASSERT_NOT_NULL(out);

    /* sum(1..6) * 3 + 1000 + i == 1063 + i */
    float *result = (float *)fm_buffer_data(out);
    for (int i = 0; i < count; i++)
        ASSERT_FLT_NEAR(expected_probe_value(repeat, scale, true, i), result[i], 1e-4f);

    fm_buffer_free(out);
}

static void test_two_specializations_stay_independent(void)
{
    int count = 4;

    fm_fc_values small, large;
    fm_fc_reset(&small);
    fm_fc_reset(&large);
    fm_fc_set_int(&small, FM_FC_PROBE_REPEAT_COUNT, 2);
    fm_fc_set_int(&large, FM_FC_PROBE_REPEAT_COUNT, 10);

    /* Interleave the dispatches: a cache that keyed on the name alone
     * would serve whichever pipeline was built first to both. */
    fm_buffer *out_small_first = run_probe(&small, count);
    fm_buffer *out_large       = run_probe(&large, count);
    fm_buffer *out_small_again = run_probe(&small, count);
    ASSERT_NOT_NULL(out_small_first);
    ASSERT_NOT_NULL(out_large);
    ASSERT_NOT_NULL(out_small_again);

    float *small_first = (float *)fm_buffer_data(out_small_first);
    float *large_vals  = (float *)fm_buffer_data(out_large);
    float *small_again = (float *)fm_buffer_data(out_small_again);

    for (int i = 0; i < count; i++) {
        ASSERT_FLT_NEAR(expected_probe_value(2, 1, false, i),  small_first[i], 1e-4f);
        ASSERT_FLT_NEAR(expected_probe_value(10, 1, false, i), large_vals[i],  1e-4f);
        ASSERT_FLT_NEAR(expected_probe_value(2, 1, false, i),  small_again[i], 1e-4f);
    }

    fm_buffer_free(out_small_first);
    fm_buffer_free(out_large);
    fm_buffer_free(out_small_again);
}

/* ================================================================
 * main
 * ================================================================ */

int main(void)
{
    printf("\n  Metal Function-Constant Specialization Tests\n");
    printf("  ============================================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "Metal unavailable\n");
        return 77;
    }

    RUN_TEST(fc_set_rejects_null_and_overflow);
    RUN_TEST(fc_set_is_order_independent);

    RUN_TEST(identical_constants_reuse_one_pipeline);
    RUN_TEST(distinct_constants_give_distinct_pipelines);
    RUN_TEST(unknown_kernel_name_returns_null);

    RUN_TEST(unspecialized_probe_uses_shader_defaults);
    RUN_TEST(probe_observes_int_constant);
    RUN_TEST(probe_observes_all_three_constant_types);
    RUN_TEST(two_specializations_stay_independent);

    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
