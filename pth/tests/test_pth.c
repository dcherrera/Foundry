/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for PyTorch .pth reader module
 */

#include "lib/test.h"
#include "pth/pth.h"
#include "safetensors/safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ============================================================= */
/* No-model tests (always run)                                    */
/* ============================================================= */

static void test_pt_open_nonexistent(void) {
    pt_file *f = pt_open("/tmp/nonexistent_file_12345.pth");
    ASSERT_NULL(f);
}

static void test_pt_open_not_zip(void) {
    /* Create a garbage file */
    const char *path = "/tmp/test_pth_garbage.bin";
    FILE *fp = fopen(path, "wb");
    ASSERT_NOT_NULL(fp);
    fprintf(fp, "this is not a zip file");
    fclose(fp);

    pt_file *f = pt_open(path);
    ASSERT_NULL(f);
    unlink(path);
}

static void test_pt_open_null(void) {
    pt_file *f = pt_open(NULL);
    ASSERT_NULL(f);
}

static void test_pt_tensor_count_null(void) {
    ASSERT_INT_EQ(pt_tensor_count(NULL), 0);
}

static void test_pt_tensor_at_null(void) {
    ASSERT_NULL(pt_tensor_at(NULL, 0));
}

static void test_pt_tensor_names_null(void) {
    int count = -1;
    const char **names = pt_tensor_names(NULL, &count);
    ASSERT_NULL(names);
    ASSERT_INT_EQ(count, 0);
}

/* ============================================================= */
/* Model-gated tests (skip if .pth not present)                   */
/* ============================================================= */

#define RWKV7_FILENAME "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"

static const char *rwkv7_pth_path(void) {
    /* Check env var first */
    const char *env = getenv("RWKV7_PTH");
    if (env && access(env, R_OK) == 0) return env;

    /* Try relative paths (from different CWDs) */
    static const char *paths[] = {
        "../models/base/" RWKV7_FILENAME,
        "../../models/base/" RWKV7_FILENAME,
        "../RWKVx/models/base/" RWKV7_FILENAME,
    };
    for (int i = 0; i < 3; i++) {
        if (access(paths[i], R_OK) == 0) return paths[i];
    }
    return NULL;
}

static int has_model(void) {
    return rwkv7_pth_path() != NULL;
}

static void test_pt_open_rwkv7(void) {
    if (!has_model()) {
        printf("    (skipped — model not found)\n");
        return;
    }
    pt_file *f = pt_open(rwkv7_pth_path());
    ASSERT_NOT_NULL(f);
    ASSERT_TRUE(pt_tensor_count(f) > 0);
    printf("    found %d tensors\n", pt_tensor_count(f));
    pt_close(f);
}

static void test_pt_tensor_names_rwkv7(void) {
    if (!has_model()) {
        printf("    (skipped — model not found)\n");
        return;
    }
    pt_file *f = pt_open(rwkv7_pth_path());
    ASSERT_NOT_NULL(f);

    int count = 0;
    const char **names = pt_tensor_names(f, &count);
    ASSERT_NOT_NULL(names);
    ASSERT_TRUE(count > 0);

    /* Check for expected tensor names */
    int found_emb = 0, found_blocks = 0, found_head = 0;
    for (int i = 0; i < count; i++) {
        if (strstr(names[i], "emb.weight")) found_emb = 1;
        if (strstr(names[i], "blocks.0.")) found_blocks = 1;
        if (strstr(names[i], "head.")) found_head = 1;
    }
    ASSERT_TRUE(found_emb);
    ASSERT_TRUE(found_blocks);
    ASSERT_TRUE(found_head);

    /* Print first few names */
    printf("    first 5 tensors:\n");
    for (int i = 0; i < 5 && i < count; i++)
        printf("      [%d] %s\n", i, names[i]);

    free(names);
    pt_close(f);
}

static void test_pt_tensor_dtypes(void) {
    if (!has_model()) {
        printf("    (skipped — model not found)\n");
        return;
    }
    pt_file *f = pt_open(rwkv7_pth_path());
    ASSERT_NOT_NULL(f);

    int n = pt_tensor_count(f);
    for (int i = 0; i < n; i++) {
        const pt_tensor *t = pt_tensor_at(f, i);
        ASSERT_NOT_NULL(t);
        /* All RWKV-7 tensors should be BF16 */
        ASSERT_TRUE(t->dtype == FS_DTYPE_BF16 || t->dtype == FS_DTYPE_F32 ||
                    t->dtype == FS_DTYPE_F16);
        ASSERT_NOT_NULL(t->data);
        ASSERT_TRUE(t->data_size > 0);
    }

    pt_close(f);
}

static void test_pt_to_safetensors(void) {
    if (!has_model()) {
        printf("    (skipped — model not found)\n");
        return;
    }
    pt_file *f = pt_open(rwkv7_pth_path());
    ASSERT_NOT_NULL(f);

    int n = pt_tensor_count(f);
    ASSERT_TRUE(n > 0);

    /* Convert to safetensors */
    const char *out_path = "/tmp/test_pth_convert.safetensors";
    fs_writer *w = fs_writer_new();
    ASSERT_NOT_NULL(w);

    int written = 0;
    for (int i = 0; i < n; i++) {
        const pt_tensor *t = pt_tensor_at(f, i);
        if (!t || !t->data || t->data_size == 0) continue;
        fs_writer_add_tensor(w, t->name, t->dtype, t->shape, t->ndim, t->data);
        written++;
    }

    ASSERT_TRUE(fs_writer_save(w, out_path));
    fs_writer_free(w);

    /* Read back and verify */
    fs_file *sf = fs_open(out_path);
    ASSERT_NOT_NULL(sf);
    ASSERT_INT_EQ(fs_num_tensors(sf), written);

    /* Spot check: verify first tensor shape matches */
    const pt_tensor *first = pt_tensor_at(f, 0);
    if (first && first->data) {
        fs_tensor_info *info = fs_get_info(sf, first->name);
        ASSERT_NOT_NULL(info);
        ASSERT_INT_EQ(info->ndim, first->ndim);
        for (int d = 0; d < first->ndim; d++)
            ASSERT_TRUE(info->shape[d] == first->shape[d]);
        ASSERT_TRUE(info->dtype == first->dtype);
        fs_free_info(info);
    }

    fs_close(sf);
    pt_close(f);
    unlink(out_path);
    printf("    converted %d tensors successfully\n", written);
}

/* ============================================================= */
/* Main                                                           */
/* ============================================================= */

int main(void) {
    printf("\n  pth module tests\n\n");

    /* Always-run tests */
    RUN_TEST(pt_open_nonexistent);
    RUN_TEST(pt_open_not_zip);
    RUN_TEST(pt_open_null);
    RUN_TEST(pt_tensor_count_null);
    RUN_TEST(pt_tensor_at_null);
    RUN_TEST(pt_tensor_names_null);

    /* Model-gated tests */
    printf("\n  model-gated tests (RWKV-7 0.1B)\n\n");
    RUN_TEST(pt_open_rwkv7);
    RUN_TEST(pt_tensor_names_rwkv7);
    RUN_TEST(pt_tensor_dtypes);
    RUN_TEST(pt_to_safetensors);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
