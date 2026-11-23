/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../safetensors.h"

#include <string.h>
#include <stdlib.h>
#include <unistd.h>

/* ============================================================= */
/* Dtype Tests                                                   */
/* ============================================================= */

static void test_dtype_names(void) {
    ASSERT_STR_EQ("F32", fs_dtype_name(FS_DTYPE_F32));
    ASSERT_STR_EQ("F64", fs_dtype_name(FS_DTYPE_F64));
    ASSERT_STR_EQ("I32", fs_dtype_name(FS_DTYPE_I32));
    ASSERT_STR_EQ("U8", fs_dtype_name(FS_DTYPE_U8));
    ASSERT_STR_EQ("UNKNOWN", fs_dtype_name(FS_DTYPE_UNKNOWN));
}

static void test_dtype_sizes(void) {
    ASSERT_INT_EQ(4, (int)fs_dtype_size(FS_DTYPE_F32));
    ASSERT_INT_EQ(8, (int)fs_dtype_size(FS_DTYPE_F64));
    ASSERT_INT_EQ(2, (int)fs_dtype_size(FS_DTYPE_F16));
    ASSERT_INT_EQ(4, (int)fs_dtype_size(FS_DTYPE_I32));
    ASSERT_INT_EQ(1, (int)fs_dtype_size(FS_DTYPE_U8));
}

static void test_dtype_from_string(void) {
    ASSERT_INT_EQ(FS_DTYPE_F32, fs_dtype_from_string("F32"));
    ASSERT_INT_EQ(FS_DTYPE_F64, fs_dtype_from_string("F64"));
    ASSERT_INT_EQ(FS_DTYPE_I32, fs_dtype_from_string("I32"));
    ASSERT_INT_EQ(FS_DTYPE_UNKNOWN, fs_dtype_from_string("INVALID"));
    ASSERT_INT_EQ(FS_DTYPE_UNKNOWN, fs_dtype_from_string(NULL));
}

/* ============================================================= */
/* Writer Tests                                                  */
/* ============================================================= */

static void test_writer_create(void) {
    fs_writer* w = fs_writer_new();
    ASSERT_NOT_NULL(w);
    fs_writer_free(w);
}

static void test_writer_add_tensor(void) {
    fs_writer* w = fs_writer_new();

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int64_t shape[] = {2, 2};

    fs_writer_add_f32(w, "weights", shape, 2, data);

    fs_writer_free(w);
    ASSERT_TRUE(1);  /* Didn't crash */
}

static void test_writer_add_metadata(void) {
    fs_writer* w = fs_writer_new();

    fs_writer_add_metadata(w, "format", "pt");
    fs_writer_add_metadata(w, "author", "test");

    fs_writer_free(w);
    ASSERT_TRUE(1);
}

/* ============================================================= */
/* Roundtrip Tests                                               */
/* ============================================================= */

static void test_roundtrip_single_tensor(void) {
    fs_writer* w = fs_writer_new();

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int64_t shape[] = {2, 3};

    fs_writer_add_f32(w, "test_tensor", shape, 2, data);

    /* Write to buffer */
    size_t size;
    void* buf = fs_writer_to_buffer(w, &size);
    ASSERT_NOT_NULL(buf);
    ASSERT_TRUE(size > 0);

    fs_writer_free(w);

    /* Read back */
    fs_file* f = fs_open_buffer(buf, size);
    ASSERT_NOT_NULL(f);

    ASSERT_INT_EQ(1, fs_num_tensors(f));
    ASSERT_TRUE(fs_has_tensor(f, "test_tensor"));

    /* Check info */
    fs_tensor_info* info = fs_get_info(f, "test_tensor");
    ASSERT_NOT_NULL(info);
    ASSERT_STR_EQ("test_tensor", info->name);
    ASSERT_INT_EQ(FS_DTYPE_F32, info->dtype);
    ASSERT_INT_EQ(2, info->ndim);
    ASSERT_INT_EQ(2, (int)info->shape[0]);
    ASSERT_INT_EQ(3, (int)info->shape[1]);
    fs_free_info(info);

    /* Check data */
    const float* read_data = fs_get_f32(f, "test_tensor");
    ASSERT_NOT_NULL(read_data);

    for (int i = 0; i < 6; i++) {
        ASSERT_FLT_NEAR(data[i], read_data[i], 0.0001f);
    }

    fs_close(f);
    free(buf);
}

static void test_roundtrip_multiple_tensors(void) {
    fs_writer* w = fs_writer_new();

    float weights[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float bias[] = {0.1f, 0.2f};
    int64_t w_shape[] = {2, 2};
    int64_t b_shape[] = {2};

    fs_writer_add_f32(w, "layer1.weight", w_shape, 2, weights);
    fs_writer_add_f32(w, "layer1.bias", b_shape, 1, bias);

    size_t size;
    void* buf = fs_writer_to_buffer(w, &size);
    fs_writer_free(w);

    fs_file* f = fs_open_buffer(buf, size);
    ASSERT_NOT_NULL(f);
    ASSERT_INT_EQ(2, fs_num_tensors(f));

    ASSERT_TRUE(fs_has_tensor(f, "layer1.weight"));
    ASSERT_TRUE(fs_has_tensor(f, "layer1.bias"));

    const float* w_data = fs_get_f32(f, "layer1.weight");
    const float* b_data = fs_get_f32(f, "layer1.bias");

    ASSERT_NOT_NULL(w_data);
    ASSERT_NOT_NULL(b_data);

    ASSERT_FLT_NEAR(1.0f, w_data[0], 0.0001f);
    ASSERT_FLT_NEAR(0.1f, b_data[0], 0.0001f);
    ASSERT_FLT_NEAR(0.2f, b_data[1], 0.0001f);

    fs_close(f);
    free(buf);
}

static void test_roundtrip_with_metadata(void) {
    fs_writer* w = fs_writer_new();

    fs_writer_add_metadata(w, "format", "pt");
    fs_writer_add_metadata(w, "model", "test_model");

    float data[] = {1.0f};
    int64_t shape[] = {1};
    fs_writer_add_f32(w, "x", shape, 1, data);

    size_t size;
    void* buf = fs_writer_to_buffer(w, &size);
    fs_writer_free(w);

    fs_file* f = fs_open_buffer(buf, size);
    ASSERT_NOT_NULL(f);

    const char* format = fs_get_metadata(f, "format");
    const char* model = fs_get_metadata(f, "model");

    ASSERT_NOT_NULL(format);
    ASSERT_NOT_NULL(model);
    ASSERT_STR_EQ("pt", format);
    ASSERT_STR_EQ("test_model", model);

    ASSERT_NULL(fs_get_metadata(f, "nonexistent"));

    fs_close(f);
    free(buf);
}

/* ============================================================= */
/* File I/O Tests                                                */
/* ============================================================= */

static void test_save_and_load(void) {
    const char* path = "/tmp/test_foundry_safetensors.safetensors";

    fs_writer* w = fs_writer_new();

    float data[] = {1.0f, 2.0f, 3.0f};
    int64_t shape[] = {3};
    fs_writer_add_f32(w, "vector", shape, 1, data);

    bool saved = fs_writer_save(w, path);
    ASSERT_TRUE(saved);
    fs_writer_free(w);

    /* Load back */
    fs_file* f = fs_open(path);
    ASSERT_NOT_NULL(f);
    ASSERT_INT_EQ(1, fs_num_tensors(f));

    const float* read_data = fs_get_f32(f, "vector");
    ASSERT_NOT_NULL(read_data);
    ASSERT_FLT_NEAR(1.0f, read_data[0], 0.0001f);
    ASSERT_FLT_NEAR(2.0f, read_data[1], 0.0001f);
    ASSERT_FLT_NEAR(3.0f, read_data[2], 0.0001f);

    fs_close(f);

    /* Cleanup */
    unlink(path);
}

/* ============================================================= */
/* Tensor Names Tests                                            */
/* ============================================================= */

static void test_tensor_names(void) {
    fs_writer* w = fs_writer_new();

    float data[] = {1.0f};
    int64_t shape[] = {1};
    fs_writer_add_f32(w, "alpha", shape, 1, data);
    fs_writer_add_f32(w, "beta", shape, 1, data);
    fs_writer_add_f32(w, "gamma", shape, 1, data);

    size_t size;
    void* buf = fs_writer_to_buffer(w, &size);
    fs_writer_free(w);

    fs_file* f = fs_open_buffer(buf, size);
    ASSERT_NOT_NULL(f);

    int count;
    const char** names = fs_tensor_names(f, &count);
    ASSERT_NOT_NULL(names);
    ASSERT_INT_EQ(3, count);

    /* Check names exist (order may vary) */
    bool found_alpha = false, found_beta = false, found_gamma = false;
    for (int i = 0; i < count; i++) {
        if (strcmp(names[i], "alpha") == 0) found_alpha = true;
        if (strcmp(names[i], "beta") == 0) found_beta = true;
        if (strcmp(names[i], "gamma") == 0) found_gamma = true;
    }
    ASSERT_TRUE(found_alpha);
    ASSERT_TRUE(found_beta);
    ASSERT_TRUE(found_gamma);

    free(names);
    fs_close(f);
    free(buf);
}

/* ============================================================= */
/* Type Checking Tests                                           */
/* ============================================================= */

static void test_dtype_mismatch_returns_null(void) {
    fs_writer* w = fs_writer_new();

    float data[] = {1.0f};
    int64_t shape[] = {1};
    fs_writer_add_f32(w, "float_tensor", shape, 1, data);

    size_t size;
    void* buf = fs_writer_to_buffer(w, &size);
    fs_writer_free(w);

    fs_file* f = fs_open_buffer(buf, size);
    ASSERT_NOT_NULL(f);

    /* Correct type works */
    ASSERT_NOT_NULL(fs_get_f32(f, "float_tensor"));

    /* Wrong types return NULL */
    ASSERT_NULL(fs_get_f64(f, "float_tensor"));
    ASSERT_NULL(fs_get_i32(f, "float_tensor"));
    ASSERT_NULL(fs_get_i64(f, "float_tensor"));

    fs_close(f);
    free(buf);
}

/* ============================================================= */
/* Different Dtypes                                              */
/* ============================================================= */

static void test_i32_tensor(void) {
    fs_writer* w = fs_writer_new();

    int32_t data[] = {10, 20, 30};
    int64_t shape[] = {3};
    fs_writer_add_tensor(w, "ints", FS_DTYPE_I32, shape, 1, data);

    size_t size;
    void* buf = fs_writer_to_buffer(w, &size);
    fs_writer_free(w);

    fs_file* f = fs_open_buffer(buf, size);
    ASSERT_NOT_NULL(f);

    fs_tensor_info* info = fs_get_info(f, "ints");
    ASSERT_NOT_NULL(info);
    ASSERT_INT_EQ(FS_DTYPE_I32, info->dtype);
    fs_free_info(info);

    const int32_t* read_data = fs_get_i32(f, "ints");
    ASSERT_NOT_NULL(read_data);
    ASSERT_INT_EQ(10, read_data[0]);
    ASSERT_INT_EQ(20, read_data[1]);
    ASSERT_INT_EQ(30, read_data[2]);

    fs_close(f);
    free(buf);
}

/* ============================================================= */
/* Edge Cases                                                    */
/* ============================================================= */

static void test_null_safety(void) {
    ASSERT_INT_EQ(0, fs_num_tensors(NULL));
    ASSERT_NULL(fs_tensor_names(NULL, NULL));
    ASSERT_FALSE(fs_has_tensor(NULL, "test"));
    ASSERT_NULL(fs_get_info(NULL, "test"));
    ASSERT_NULL(fs_get_data(NULL, "test"));
    ASSERT_NULL(fs_get_f32(NULL, "test"));
    ASSERT_NULL(fs_get_metadata(NULL, "test"));

    fs_close(NULL);  /* Should not crash */
    fs_free_info(NULL);  /* Should not crash */
    fs_writer_free(NULL);  /* Should not crash */

    ASSERT_TRUE(1);
}

static void test_nonexistent_tensor(void) {
    fs_writer* w = fs_writer_new();

    float data[] = {1.0f};
    int64_t shape[] = {1};
    fs_writer_add_f32(w, "exists", shape, 1, data);

    size_t size;
    void* buf = fs_writer_to_buffer(w, &size);
    fs_writer_free(w);

    fs_file* f = fs_open_buffer(buf, size);
    ASSERT_NOT_NULL(f);

    ASSERT_FALSE(fs_has_tensor(f, "does_not_exist"));
    ASSERT_NULL(fs_get_info(f, "does_not_exist"));
    ASSERT_NULL(fs_get_data(f, "does_not_exist"));
    ASSERT_NULL(fs_get_f32(f, "does_not_exist"));

    fs_close(f);
    free(buf);
}

static void test_open_invalid_file(void) {
    fs_file* f = fs_open("/nonexistent/path/to/file.safetensors");
    ASSERT_NULL(f);
}

static void test_scalar_tensor(void) {
    fs_writer* w = fs_writer_new();

    float data[] = {42.0f};
    int64_t shape[] = {1};
    fs_writer_add_f32(w, "scalar", shape, 1, data);

    size_t size;
    void* buf = fs_writer_to_buffer(w, &size);
    fs_writer_free(w);

    fs_file* f = fs_open_buffer(buf, size);
    ASSERT_NOT_NULL(f);

    const float* read_data = fs_get_f32(f, "scalar");
    ASSERT_NOT_NULL(read_data);
    ASSERT_FLT_NEAR(42.0f, read_data[0], 0.0001f);

    fs_close(f);
    free(buf);
}

static void test_empty_file_with_metadata_only(void) {
    fs_writer* w = fs_writer_new();

    /* Only metadata, no tensors */
    fs_writer_add_metadata(w, "test", "value");

    float dummy[] = {0.0f};
    int64_t shape[] = {1};
    fs_writer_add_f32(w, "placeholder", shape, 1, dummy);

    size_t size;
    void* buf = fs_writer_to_buffer(w, &size);
    fs_writer_free(w);

    fs_file* f = fs_open_buffer(buf, size);
    ASSERT_NOT_NULL(f);

    const char* val = fs_get_metadata(f, "test");
    ASSERT_NOT_NULL(val);
    ASSERT_STR_EQ("value", val);

    fs_close(f);
    free(buf);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running safetensors tests...\n\n");

    /* Dtype tests */
    RUN_TEST(dtype_names);
    RUN_TEST(dtype_sizes);
    RUN_TEST(dtype_from_string);

    /* Writer tests */
    RUN_TEST(writer_create);
    RUN_TEST(writer_add_tensor);
    RUN_TEST(writer_add_metadata);

    /* Roundtrip tests */
    RUN_TEST(roundtrip_single_tensor);
    RUN_TEST(roundtrip_multiple_tensors);
    RUN_TEST(roundtrip_with_metadata);

    /* File I/O tests */
    RUN_TEST(save_and_load);

    /* Tensor names */
    RUN_TEST(tensor_names);

    /* Type checking */
    RUN_TEST(dtype_mismatch_returns_null);

    /* Different dtypes */
    RUN_TEST(i32_tensor);

    /* Edge cases */
    RUN_TEST(null_safety);
    RUN_TEST(nonexistent_tensor);
    RUN_TEST(open_invalid_file);
    RUN_TEST(scalar_tensor);
    RUN_TEST(empty_file_with_metadata_only);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
