/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../data.h"

#include <string.h>
#include <stdlib.h>
#include <unistd.h>

/* ============================================================= */
/* Sample Tests                                                  */
/* ============================================================= */

static void test_sample_create(void) {
    fd_sample* s = fd_sample_new();
    ASSERT_NOT_NULL(s);
    ASSERT_INT_EQ(0, s->num_fields);
    fd_sample_free(s);
}

static void test_sample_set_get(void) {
    fd_sample* s = fd_sample_new();

    fd_sample_set(s, "text", "hello world");
    fd_sample_set(s, "label", "positive");

    ASSERT_INT_EQ(2, s->num_fields);

    const char* text = fd_sample_get(s, "text");
    const char* label = fd_sample_get(s, "label");

    ASSERT_NOT_NULL(text);
    ASSERT_NOT_NULL(label);
    ASSERT_STR_EQ("hello world", text);
    ASSERT_STR_EQ("positive", label);

    ASSERT_NULL(fd_sample_get(s, "nonexistent"));

    fd_sample_free(s);
}

static void test_sample_has(void) {
    fd_sample* s = fd_sample_new();
    fd_sample_set(s, "key", "value");

    ASSERT_TRUE(fd_sample_has(s, "key"));
    ASSERT_FALSE(fd_sample_has(s, "missing"));

    fd_sample_free(s);
}

static void test_sample_update(void) {
    fd_sample* s = fd_sample_new();

    fd_sample_set(s, "key", "original");
    ASSERT_STR_EQ("original", fd_sample_get(s, "key"));

    fd_sample_set(s, "key", "updated");
    ASSERT_STR_EQ("updated", fd_sample_get(s, "key"));
    ASSERT_INT_EQ(1, s->num_fields);

    fd_sample_free(s);
}

static void test_sample_clone(void) {
    fd_sample* s = fd_sample_new();
    fd_sample_set(s, "a", "1");
    fd_sample_set(s, "b", "2");

    fd_sample* clone = fd_sample_clone(s);
    ASSERT_NOT_NULL(clone);
    ASSERT_INT_EQ(2, clone->num_fields);
    ASSERT_STR_EQ("1", fd_sample_get(clone, "a"));
    ASSERT_STR_EQ("2", fd_sample_get(clone, "b"));

    /* Modify original, clone should be unaffected */
    fd_sample_set(s, "a", "changed");
    ASSERT_STR_EQ("1", fd_sample_get(clone, "a"));

    fd_sample_free(s);
    fd_sample_free(clone);
}

/* ============================================================= */
/* Dataset Tests                                                 */
/* ============================================================= */

static void test_dataset_create(void) {
    fd_dataset* ds = fd_new();
    ASSERT_NOT_NULL(ds);
    ASSERT_INT_EQ(0, fd_len(ds));
    fd_free(ds);
}

static void test_dataset_add_samples(void) {
    fd_dataset* ds = fd_new();

    fd_sample* s1 = fd_sample_new();
    fd_sample_set(s1, "text", "sample 1");
    fd_add_sample(ds, s1);

    fd_sample* s2 = fd_sample_new();
    fd_sample_set(s2, "text", "sample 2");
    fd_add_sample(ds, s2);

    ASSERT_INT_EQ(2, fd_len(ds));

    fd_sample* retrieved = fd_get(ds, 0);
    ASSERT_NOT_NULL(retrieved);
    ASSERT_STR_EQ("sample 1", fd_sample_get(retrieved, "text"));

    retrieved = fd_get(ds, 1);
    ASSERT_STR_EQ("sample 2", fd_sample_get(retrieved, "text"));

    ASSERT_NULL(fd_get(ds, 2));
    ASSERT_NULL(fd_get(ds, -1));

    fd_free(ds);
}

static void test_dataset_shuffle(void) {
    fd_dataset* ds = fd_new();

    /* Add 100 samples */
    for (int i = 0; i < 100; i++) {
        fd_sample* s = fd_sample_new();
        char val[16];
        snprintf(val, sizeof(val), "%d", i);
        fd_sample_set(s, "idx", val);
        fd_add_sample(ds, s);
    }

    fd_seed(42);
    fd_shuffle(ds);

    /* After shuffle, order should likely be different */
    /* At minimum, check that we still have all 100 samples */
    ASSERT_INT_EQ(100, fd_len(ds));

    /* Check all values are still present (counting 0-99) */
    int found[100] = {0};
    for (int i = 0; i < 100; i++) {
        const char* val = fd_sample_get(fd_get(ds, i), "idx");
        int idx = atoi(val);
        found[idx] = 1;
    }

    for (int i = 0; i < 100; i++) {
        ASSERT_TRUE(found[i]);
    }

    fd_free(ds);
}

static void test_dataset_split(void) {
    fd_dataset* ds = fd_new();

    for (int i = 0; i < 100; i++) {
        fd_sample* s = fd_sample_new();
        char val[16];
        snprintf(val, sizeof(val), "%d", i);
        fd_sample_set(s, "idx", val);
        fd_add_sample(ds, s);
    }

    fd_dataset* train = NULL;
    fd_dataset* test = NULL;

    fd_split(ds, 0.8f, &train, &test);

    ASSERT_NOT_NULL(train);
    ASSERT_NOT_NULL(test);
    ASSERT_INT_EQ(80, fd_len(train));
    ASSERT_INT_EQ(20, fd_len(test));

    fd_free(ds);
    fd_free(train);
    fd_free(test);
}

/* ============================================================= */
/* JSONL Loading Tests                                           */
/* ============================================================= */

static void test_load_jsonl(void) {
    const char* path = "/tmp/test_foundry_data.jsonl";

    /* Write test file */
    FILE* f = fopen(path, "w");
    fprintf(f, "{\"text\": \"hello\", \"label\": \"positive\"}\n");
    fprintf(f, "{\"text\": \"goodbye\", \"label\": \"negative\"}\n");
    fprintf(f, "{\"text\": \"test\", \"label\": \"neutral\"}\n");
    fclose(f);

    fd_dataset* ds = fd_load_jsonl(path);
    ASSERT_NOT_NULL(ds);
    ASSERT_INT_EQ(3, fd_len(ds));

    fd_sample* s0 = fd_get(ds, 0);
    ASSERT_STR_EQ("hello", fd_sample_get(s0, "text"));
    ASSERT_STR_EQ("positive", fd_sample_get(s0, "label"));

    fd_sample* s1 = fd_get(ds, 1);
    ASSERT_STR_EQ("goodbye", fd_sample_get(s1, "text"));

    fd_free(ds);
    unlink(path);
}

static void test_load_jsonl_with_numbers(void) {
    const char* path = "/tmp/test_foundry_data2.jsonl";

    FILE* f = fopen(path, "w");
    fprintf(f, "{\"value\": 42, \"score\": 3.14}\n");
    fprintf(f, "{\"value\": -10, \"score\": 2.71}\n");
    fclose(f);

    fd_dataset* ds = fd_load_jsonl(path);
    ASSERT_NOT_NULL(ds);
    ASSERT_INT_EQ(2, fd_len(ds));

    /* Numbers are stored as strings */
    ASSERT_STR_EQ("42", fd_sample_get(fd_get(ds, 0), "value"));
    ASSERT_STR_EQ("3.14", fd_sample_get(fd_get(ds, 0), "score"));

    fd_free(ds);
    unlink(path);
}

static void test_load_jsonl_empty_lines(void) {
    const char* path = "/tmp/test_foundry_data3.jsonl";

    FILE* f = fopen(path, "w");
    fprintf(f, "{\"a\": \"1\"}\n");
    fprintf(f, "\n");
    fprintf(f, "   \n");
    fprintf(f, "{\"a\": \"2\"}\n");
    fclose(f);

    fd_dataset* ds = fd_load_jsonl(path);
    ASSERT_NOT_NULL(ds);
    ASSERT_INT_EQ(2, fd_len(ds));

    fd_free(ds);
    unlink(path);
}

static void test_load_nonexistent_file(void) {
    fd_dataset* ds = fd_load_jsonl("/nonexistent/path/data.jsonl");
    ASSERT_NULL(ds);
}

/* ============================================================= */
/* Data Loader Tests                                             */
/* ============================================================= */

static void test_loader_create(void) {
    fd_dataset* ds = fd_new();
    for (int i = 0; i < 10; i++) {
        fd_sample* s = fd_sample_new();
        fd_sample_set(s, "x", "y");
        fd_add_sample(ds, s);
    }

    fd_loader* loader = fd_loader_new(ds, 4, false);
    ASSERT_NOT_NULL(loader);
    ASSERT_INT_EQ(3, fd_loader_num_batches(loader));  /* ceil(10/4) = 3 */

    fd_loader_free(loader);
    fd_free(ds);
}

static void test_loader_iterate(void) {
    fd_dataset* ds = fd_new();
    for (int i = 0; i < 10; i++) {
        fd_sample* s = fd_sample_new();
        char val[16];
        snprintf(val, sizeof(val), "%d", i);
        fd_sample_set(s, "idx", val);
        fd_add_sample(ds, s);
    }

    fd_loader* loader = fd_loader_new(ds, 4, false);

    /* First batch: 4 samples */
    fd_batch* batch = fd_loader_next(loader);
    ASSERT_NOT_NULL(batch);
    ASSERT_INT_EQ(4, batch->size);
    ASSERT_STR_EQ("0", fd_sample_get(batch->samples[0], "idx"));
    fd_batch_free(batch);

    /* Second batch: 4 samples */
    batch = fd_loader_next(loader);
    ASSERT_NOT_NULL(batch);
    ASSERT_INT_EQ(4, batch->size);
    ASSERT_STR_EQ("4", fd_sample_get(batch->samples[0], "idx"));
    fd_batch_free(batch);

    /* Third batch: 2 samples (remaining) */
    batch = fd_loader_next(loader);
    ASSERT_NOT_NULL(batch);
    ASSERT_INT_EQ(2, batch->size);
    ASSERT_STR_EQ("8", fd_sample_get(batch->samples[0], "idx"));
    fd_batch_free(batch);

    /* No more batches */
    batch = fd_loader_next(loader);
    ASSERT_NULL(batch);

    fd_loader_free(loader);
    fd_free(ds);
}

static void test_loader_reset(void) {
    fd_dataset* ds = fd_new();
    for (int i = 0; i < 5; i++) {
        fd_sample* s = fd_sample_new();
        fd_sample_set(s, "x", "y");
        fd_add_sample(ds, s);
    }

    fd_loader* loader = fd_loader_new(ds, 2, false);

    /* Exhaust the loader */
    while (fd_loader_next(loader)) {}

    /* Reset and check we can iterate again */
    fd_loader_reset(loader);

    fd_batch* batch = fd_loader_next(loader);
    ASSERT_NOT_NULL(batch);
    ASSERT_INT_EQ(2, batch->size);
    fd_batch_free(batch);

    fd_loader_free(loader);
    fd_free(ds);
}

static void test_loader_shuffle(void) {
    fd_dataset* ds = fd_new();
    for (int i = 0; i < 100; i++) {
        fd_sample* s = fd_sample_new();
        char val[16];
        snprintf(val, sizeof(val), "%d", i);
        fd_sample_set(s, "idx", val);
        fd_add_sample(ds, s);
    }

    fd_seed(123);
    fd_loader* loader = fd_loader_new(ds, 10, true);

    /* First batch with shuffle - unlikely to start with "0" */
    fd_batch* batch = fd_loader_next(loader);
    ASSERT_NOT_NULL(batch);
    ASSERT_INT_EQ(10, batch->size);

    /* Just verify we get 100 samples total in random order */
    int count = batch->size;
    fd_batch_free(batch);

    while ((batch = fd_loader_next(loader)) != NULL) {
        count += batch->size;
        fd_batch_free(batch);
    }

    ASSERT_INT_EQ(100, count);

    fd_loader_free(loader);
    fd_free(ds);
}

/* ============================================================= */
/* Edge Cases                                                    */
/* ============================================================= */

static void test_null_safety(void) {
    ASSERT_INT_EQ(0, fd_len(NULL));
    ASSERT_NULL(fd_get(NULL, 0));
    ASSERT_NULL(fd_sample_get(NULL, "key"));
    ASSERT_FALSE(fd_sample_has(NULL, "key"));
    ASSERT_NULL(fd_sample_clone(NULL));
    ASSERT_NULL(fd_loader_next(NULL));
    ASSERT_INT_EQ(0, fd_loader_num_batches(NULL));

    fd_sample_free(NULL);  /* Should not crash */
    fd_free(NULL);         /* Should not crash */
    fd_loader_free(NULL);  /* Should not crash */
    fd_batch_free(NULL);   /* Should not crash */
    fd_shuffle(NULL);      /* Should not crash */
    fd_loader_reset(NULL); /* Should not crash */

    ASSERT_TRUE(1);
}

static void test_empty_dataset_loader(void) {
    fd_dataset* ds = fd_new();
    fd_loader* loader = fd_loader_new(ds, 4, false);

    ASSERT_NOT_NULL(loader);
    ASSERT_INT_EQ(0, fd_loader_num_batches(loader));

    fd_batch* batch = fd_loader_next(loader);
    ASSERT_NULL(batch);

    fd_loader_free(loader);
    fd_free(ds);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running data tests...\n\n");

    /* Sample tests */
    RUN_TEST(sample_create);
    RUN_TEST(sample_set_get);
    RUN_TEST(sample_has);
    RUN_TEST(sample_update);
    RUN_TEST(sample_clone);

    /* Dataset tests */
    RUN_TEST(dataset_create);
    RUN_TEST(dataset_add_samples);
    RUN_TEST(dataset_shuffle);
    RUN_TEST(dataset_split);

    /* JSONL loading tests */
    RUN_TEST(load_jsonl);
    RUN_TEST(load_jsonl_with_numbers);
    RUN_TEST(load_jsonl_empty_lines);
    RUN_TEST(load_nonexistent_file);

    /* Data loader tests */
    RUN_TEST(loader_create);
    RUN_TEST(loader_iterate);
    RUN_TEST(loader_reset);
    RUN_TEST(loader_shuffle);

    /* Edge cases */
    RUN_TEST(null_safety);
    RUN_TEST(empty_dataset_loader);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
