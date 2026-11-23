/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../export.h"
#include "../quantization.h"
#include "../../safetensors/safetensors.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

/* ============================================================= */
/* Helpers: create in-memory test model                          */
/* ============================================================= */

/* Create a minimal safetensors file with F32 tensors and save to disk */
static const char* TEST_MODEL = "/tmp/fq_test_model.safetensors";
static const char* TEST_OUTPUT = "/tmp/fq_test_output.safetensors";

static bool create_test_model(void) {
    fs_writer* w = fs_writer_new();
    if (!w) return false;

    /* 2D weight tensor (large enough to quantize) */
    float weight[2048];
    for (int i = 0; i < 2048; i++) {
        weight[i] = sinf((float)i * 0.01f) * 2.0f;
    }
    int64_t weight_shape[] = {32, 64};
    fs_writer_add_f32(w, "layer.0.weight", weight_shape, 2, weight);

    /* Another weight */
    float weight2[2048];
    for (int i = 0; i < 2048; i++) {
        weight2[i] = cosf((float)i * 0.02f) * 3.0f;
    }
    fs_writer_add_f32(w, "layer.1.weight", weight_shape, 2, weight2);

    /* Bias (1D, should be skipped with default config) */
    float bias[64];
    for (int i = 0; i < 64; i++) bias[i] = (float)i * 0.01f;
    int64_t bias_shape[] = {64};
    fs_writer_add_f32(w, "layer.0.bias", bias_shape, 1, bias);

    /* Norm layer (should be skipped with default config) */
    float norm[64];
    for (int i = 0; i < 64; i++) norm[i] = 1.0f;
    fs_writer_add_f32(w, "ln_0.weight", bias_shape, 1, norm);

    /* Embedding (should be skipped with default config) */
    float embed[4096];
    for (int i = 0; i < 4096; i++) embed[i] = (float)i * 0.001f;
    int64_t embed_shape[] = {64, 64};
    fs_writer_add_f32(w, "embed.weight", embed_shape, 2, embed);

    /* Small tensor (below min_elements, should be skipped) */
    float small[16];
    for (int i = 0; i < 16; i++) small[i] = (float)i;
    int64_t small_shape[] = {4, 4};
    fs_writer_add_f32(w, "tiny.weight", small_shape, 2, small);

    bool ok = fs_writer_save(w, TEST_MODEL);
    fs_writer_free(w);
    return ok;
}

static void cleanup_test_files(void) {
    unlink(TEST_MODEL);
    unlink(TEST_OUTPUT);
}

/* ============================================================= */
/* Config Tests                                                  */
/* ============================================================= */

static void test_config_defaults(void) {
    fq_export_config cfg = fq_export_config_defaults();
    ASSERT_INT_EQ(FQ_INT8, cfg.target_dtype);
    ASSERT_INT_EQ(FQ_PER_TENSOR, cfg.granularity);
    ASSERT_INT_EQ(128, cfg.group_size);
    ASSERT_FLT_NEAR(0.0f, cfg.ternary_threshold, 0.001f);
    ASSERT_TRUE(cfg.skip_embeddings);
    ASSERT_TRUE(cfg.skip_norms);
    ASSERT_TRUE(cfg.skip_biases);
    ASSERT_TRUE(cfg.skip_1d);
    ASSERT_INT_EQ(1024, cfg.min_elements);
    ASSERT_NULL(cfg.progress_cb);
}

/* ============================================================= */
/* INT8 Export Tests                                              */
/* ============================================================= */

static void test_int8_export(void) {
    ASSERT_TRUE(create_test_model());

    fq_export_config cfg = fq_export_config_defaults();
    cfg.target_dtype = FQ_INT8;

    bool ok = fq_export_quantized(TEST_MODEL, TEST_OUTPUT, &cfg);
    ASSERT_TRUE(ok);

    /* Read back and verify */
    fs_file* f = fs_open(TEST_OUTPUT);
    ASSERT_NOT_NULL(f);

    /* Should have quantized weight tensors with scale metadata */
    ASSERT_TRUE(fs_has_tensor(f, "layer.0.weight"));
    ASSERT_TRUE(fs_has_tensor(f, "layer.0.weight.__scale"));
    ASSERT_TRUE(fs_has_tensor(f, "layer.0.weight.__shape"));
    ASSERT_TRUE(fs_has_tensor(f, "layer.1.weight"));
    ASSERT_TRUE(fs_has_tensor(f, "layer.1.weight.__scale"));

    /* INT8 data tensor should be I8 dtype */
    fs_tensor_info* info = fs_get_info(f, "layer.0.weight");
    ASSERT_NOT_NULL(info);
    ASSERT_INT_EQ(FS_DTYPE_I8, info->dtype);
    fs_free_info(info);

    /* Scale should be F32 */
    fs_tensor_info* scale_info = fs_get_info(f, "layer.0.weight.__scale");
    ASSERT_NOT_NULL(scale_info);
    ASSERT_INT_EQ(FS_DTYPE_F32, scale_info->dtype);
    fs_free_info(scale_info);

    /* Bias should be passed through as F32 */
    ASSERT_TRUE(fs_has_tensor(f, "layer.0.bias"));
    fs_tensor_info* bias_info = fs_get_info(f, "layer.0.bias");
    ASSERT_NOT_NULL(bias_info);
    ASSERT_INT_EQ(FS_DTYPE_F32, bias_info->dtype);
    fs_free_info(bias_info);

    /* Check metadata */
    ASSERT_STR_EQ("INT8", fs_get_metadata(f, "quantization.dtype"));
    ASSERT_STR_EQ("per_tensor", fs_get_metadata(f, "quantization.granularity"));
    ASSERT_STR_EQ("1", fs_get_metadata(f, "quantization.format_version"));

    fs_close(f);
    cleanup_test_files();
}

/* ============================================================= */
/* INT4 Export Tests                                              */
/* ============================================================= */

static void test_int4_export(void) {
    ASSERT_TRUE(create_test_model());

    fq_export_config cfg = fq_export_config_defaults();
    cfg.target_dtype = FQ_INT4;
    cfg.granularity = FQ_PER_GROUP;
    cfg.group_size = 64;

    bool ok = fq_export_quantized(TEST_MODEL, TEST_OUTPUT, &cfg);
    ASSERT_TRUE(ok);

    fs_file* f = fs_open(TEST_OUTPUT);
    ASSERT_NOT_NULL(f);

    /* Quantized tensors should be U8 (packed INT4) */
    fs_tensor_info* info = fs_get_info(f, "layer.0.weight");
    ASSERT_NOT_NULL(info);
    ASSERT_INT_EQ(FS_DTYPE_U8, info->dtype);
    fs_free_info(info);

    ASSERT_TRUE(fs_has_tensor(f, "layer.0.weight.__scale"));
    ASSERT_TRUE(fs_has_tensor(f, "layer.0.weight.__shape"));

    ASSERT_STR_EQ("INT4", fs_get_metadata(f, "quantization.dtype"));
    ASSERT_STR_EQ("per_group", fs_get_metadata(f, "quantization.granularity"));

    fs_close(f);
    cleanup_test_files();
}

/* ============================================================= */
/* Ternary Export Tests                                           */
/* ============================================================= */

static void test_ternary_export(void) {
    ASSERT_TRUE(create_test_model());

    fq_export_config cfg = fq_export_config_defaults();
    cfg.target_dtype = FQ_TERNARY;

    bool ok = fq_export_quantized(TEST_MODEL, TEST_OUTPUT, &cfg);
    ASSERT_TRUE(ok);

    fs_file* f = fs_open(TEST_OUTPUT);
    ASSERT_NOT_NULL(f);

    /* Ternary data should be U8 (packed) */
    fs_tensor_info* info = fs_get_info(f, "layer.0.weight");
    ASSERT_NOT_NULL(info);
    ASSERT_INT_EQ(FS_DTYPE_U8, info->dtype);

    /* Packed size: 2048 values / 4 per byte = 512 bytes */
    ASSERT_INT_EQ(512, (int)info->shape[0]);
    fs_free_info(info);

    ASSERT_TRUE(fs_has_tensor(f, "layer.0.weight.__scale"));
    ASSERT_TRUE(fs_has_tensor(f, "layer.0.weight.__shape"));
    ASSERT_STR_EQ("TERNARY", fs_get_metadata(f, "quantization.dtype"));

    fs_close(f);
    cleanup_test_files();
}

/* ============================================================= */
/* Skip Logic Tests                                              */
/* ============================================================= */

static void test_skip_norms(void) {
    ASSERT_TRUE(create_test_model());

    fq_export_config cfg = fq_export_config_defaults();
    cfg.skip_norms = true;

    fq_export_quantized(TEST_MODEL, TEST_OUTPUT, &cfg);

    fs_file* f = fs_open(TEST_OUTPUT);
    ASSERT_NOT_NULL(f);

    /* Norm should be passed through as F32 */
    ASSERT_TRUE(fs_has_tensor(f, "ln_0.weight"));
    fs_tensor_info* info = fs_get_info(f, "ln_0.weight");
    ASSERT_NOT_NULL(info);
    ASSERT_INT_EQ(FS_DTYPE_F32, info->dtype);
    fs_free_info(info);

    /* Should NOT have scale tensor for norm */
    ASSERT_FALSE(fs_has_tensor(f, "ln_0.weight.__scale"));

    fs_close(f);
    cleanup_test_files();
}

static void test_skip_embeddings(void) {
    ASSERT_TRUE(create_test_model());

    fq_export_config cfg = fq_export_config_defaults();
    cfg.skip_embeddings = true;

    fq_export_quantized(TEST_MODEL, TEST_OUTPUT, &cfg);

    fs_file* f = fs_open(TEST_OUTPUT);
    ASSERT_NOT_NULL(f);

    /* Embed should be passed through */
    ASSERT_TRUE(fs_has_tensor(f, "embed.weight"));
    ASSERT_FALSE(fs_has_tensor(f, "embed.weight.__scale"));

    fs_close(f);
    cleanup_test_files();
}

static void test_skip_biases(void) {
    ASSERT_TRUE(create_test_model());

    fq_export_config cfg = fq_export_config_defaults();
    cfg.skip_biases = true;

    fq_export_quantized(TEST_MODEL, TEST_OUTPUT, &cfg);

    fs_file* f = fs_open(TEST_OUTPUT);
    ASSERT_NOT_NULL(f);

    /* Bias should be passed through */
    ASSERT_TRUE(fs_has_tensor(f, "layer.0.bias"));
    ASSERT_FALSE(fs_has_tensor(f, "layer.0.bias.__scale"));

    fs_close(f);
    cleanup_test_files();
}

static void test_skip_small_tensors(void) {
    ASSERT_TRUE(create_test_model());

    fq_export_config cfg = fq_export_config_defaults();
    cfg.min_elements = 1024;
    cfg.skip_1d = false;
    cfg.skip_norms = false;
    cfg.skip_biases = false;
    cfg.skip_embeddings = false;

    fq_export_quantized(TEST_MODEL, TEST_OUTPUT, &cfg);

    fs_file* f = fs_open(TEST_OUTPUT);
    ASSERT_NOT_NULL(f);

    /* tiny.weight has 16 elements < 1024, should be passed through */
    ASSERT_TRUE(fs_has_tensor(f, "tiny.weight"));
    ASSERT_FALSE(fs_has_tensor(f, "tiny.weight.__scale"));

    fs_close(f);
    cleanup_test_files();
}

/* ============================================================= */
/* Scale Naming Tests                                            */
/* ============================================================= */

static void test_scale_naming(void) {
    ASSERT_TRUE(create_test_model());

    fq_export_config cfg = fq_export_config_defaults();
    fq_export_quantized(TEST_MODEL, TEST_OUTPUT, &cfg);

    fs_file* f = fs_open(TEST_OUTPUT);
    ASSERT_NOT_NULL(f);

    /* Verify __scale suffix naming convention */
    ASSERT_TRUE(fs_has_tensor(f, "layer.0.weight.__scale"));
    ASSERT_TRUE(fs_has_tensor(f, "layer.1.weight.__scale"));
    ASSERT_TRUE(fs_has_tensor(f, "layer.0.weight.__shape"));
    ASSERT_TRUE(fs_has_tensor(f, "layer.1.weight.__shape"));

    fs_close(f);
    cleanup_test_files();
}

/* ============================================================= */
/* Progress Callback Tests                                       */
/* ============================================================= */

typedef struct {
    int calls;
    int last_current;
    int last_total;
} progress_state;

static void test_progress_callback_fn(int current, int total, const char* name, void* ud) {
    progress_state* ps = (progress_state*)ud;
    ps->calls++;
    ps->last_current = current;
    ps->last_total = total;
    (void)name;
}

static void test_progress_callback(void) {
    ASSERT_TRUE(create_test_model());

    progress_state ps = {0, 0, 0};

    fq_export_config cfg = fq_export_config_defaults();
    cfg.progress_cb = test_progress_callback_fn;
    cfg.userdata = &ps;

    fq_export_quantized(TEST_MODEL, TEST_OUTPUT, &cfg);

    /* Callback should have been called for each tensor */
    ASSERT_TRUE(ps.calls > 0);
    ASSERT_INT_EQ(ps.last_current, ps.last_total);

    cleanup_test_files();
}

/* ============================================================= */
/* Step-by-step API Tests                                        */
/* ============================================================= */

static void test_step_api(void) {
    ASSERT_TRUE(create_test_model());

    fq_export_config cfg = fq_export_config_defaults();
    fq_export_state* s = fq_export_begin(TEST_MODEL, &cfg);
    ASSERT_NOT_NULL(s);

    int count = fq_export_tensor_count(s);
    ASSERT_TRUE(count > 0);

    /* Process each tensor one at a time */
    int steps = 0;
    while (fq_export_step(s)) {
        steps++;
        float prog = fq_export_progress(s);
        ASSERT_TRUE(prog > 0.0f);
        ASSERT_TRUE(prog <= 1.0f);
    }
    /* One more step for the last tensor (returns false) */
    ASSERT_INT_EQ(count - 1, steps);

    /* Progress should be 1.0 */
    ASSERT_FLT_NEAR(1.0f, fq_export_progress(s), 0.01f);

    bool ok = fq_export_finish(s, TEST_OUTPUT);
    ASSERT_TRUE(ok);

    /* Verify output exists and is valid */
    fs_file* f = fs_open(TEST_OUTPUT);
    ASSERT_NOT_NULL(f);
    ASSERT_TRUE(fs_num_tensors(f) > 0);
    fs_close(f);

    cleanup_test_files();
}

/* ============================================================= */
/* Abort Tests                                                   */
/* ============================================================= */

static void test_abort(void) {
    ASSERT_TRUE(create_test_model());

    fq_export_config cfg = fq_export_config_defaults();
    fq_export_state* s = fq_export_begin(TEST_MODEL, &cfg);
    ASSERT_NOT_NULL(s);

    /* Process a couple tensors then abort */
    fq_export_step(s);
    fq_export_abort(s); /* Should not crash or leak */

    cleanup_test_files();
    ASSERT_TRUE(1);
}

/* ============================================================= */
/* NULL Safety Tests                                             */
/* ============================================================= */

static void test_null_safety(void) {
    ASSERT_FALSE(fq_export_quantized(NULL, NULL, NULL));
    ASSERT_NULL(fq_export_begin(NULL, NULL));
    ASSERT_INT_EQ(0, fq_export_tensor_count(NULL));
    ASSERT_FALSE(fq_export_step(NULL));
    ASSERT_FLT_NEAR(0.0f, fq_export_progress(NULL), 0.001f);
    ASSERT_NULL(fq_export_current_tensor(NULL));
    ASSERT_FALSE(fq_export_finish(NULL, NULL));
    fq_export_abort(NULL); /* Should not crash */

    ASSERT_TRUE(1);
}

/* ============================================================= */
/* Metadata Tests                                                */
/* ============================================================= */

static void test_metadata_written(void) {
    ASSERT_TRUE(create_test_model());

    fq_export_config cfg = fq_export_config_defaults();
    cfg.target_dtype = FQ_TERNARY;
    cfg.granularity = FQ_PER_CHANNEL;
    cfg.group_size = 256;

    fq_export_quantized(TEST_MODEL, TEST_OUTPUT, &cfg);

    fs_file* f = fs_open(TEST_OUTPUT);
    ASSERT_NOT_NULL(f);

    ASSERT_STR_EQ("TERNARY", fs_get_metadata(f, "quantization.dtype"));
    ASSERT_STR_EQ("per_channel", fs_get_metadata(f, "quantization.granularity"));
    ASSERT_STR_EQ("256", fs_get_metadata(f, "quantization.group_size"));
    ASSERT_STR_EQ("1", fs_get_metadata(f, "quantization.format_version"));

    fs_close(f);
    cleanup_test_files();
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running export tests...\n\n");

    /* Config */
    RUN_TEST(config_defaults);

    /* Round-trip tests */
    RUN_TEST(int8_export);
    RUN_TEST(int4_export);
    RUN_TEST(ternary_export);

    /* Skip logic */
    RUN_TEST(skip_norms);
    RUN_TEST(skip_embeddings);
    RUN_TEST(skip_biases);
    RUN_TEST(skip_small_tensors);

    /* Naming */
    RUN_TEST(scale_naming);

    /* Metadata */
    RUN_TEST(metadata_written);

    /* Progress */
    RUN_TEST(progress_callback);

    /* Step-by-step API */
    RUN_TEST(step_api);
    RUN_TEST(abort);

    /* NULL safety */
    RUN_TEST(null_safety);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
