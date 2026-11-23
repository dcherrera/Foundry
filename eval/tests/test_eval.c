/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../eval.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

/* ============================================================= */
/* Metrics Tests                                                 */
/* ============================================================= */

static void test_perplexity(void) {
    /* logprobs = [-1, -1, -1, -1] -> avg=-1 -> ppl=e^1 ≈ 2.718 */
    float logprobs[] = {-1.0f, -1.0f, -1.0f, -1.0f};
    double ppl = fe_perplexity(logprobs, 4);

    ASSERT_FLT_NEAR(exp(1.0), ppl, 0.01);
}

static void test_perplexity_perfect(void) {
    /* logprobs = [0, 0, 0] -> avg=0 -> ppl=e^0 = 1 */
    float logprobs[] = {0.0f, 0.0f, 0.0f};
    double ppl = fe_perplexity(logprobs, 3);

    ASSERT_FLT_NEAR(1.0, ppl, 0.001);
}

static void test_accuracy(void) {
    int predictions[] = {0, 1, 1, 0, 1};
    int labels[]      = {0, 1, 0, 0, 1};  /* 4 correct out of 5 */

    double acc = fe_accuracy(predictions, labels, 5);
    ASSERT_FLT_NEAR(0.8, acc, 0.001);
}

static void test_accuracy_perfect(void) {
    int predictions[] = {1, 2, 3, 4};
    int labels[]      = {1, 2, 3, 4};

    double acc = fe_accuracy(predictions, labels, 4);
    ASSERT_FLT_NEAR(1.0, acc, 0.001);
}

static void test_f1_score(void) {
    /* TP=2, FP=1, FN=1 */
    int predictions[] = {1, 1, 1, 0};
    int labels[]      = {1, 1, 0, 1};

    double f1 = fe_f1_score(predictions, labels, 4);

    /* precision = 2/3, recall = 2/3, f1 = 2/3 */
    ASSERT_FLT_NEAR(2.0/3.0, f1, 0.01);
}

/* ============================================================= */
/* Task Tests                                                    */
/* ============================================================= */

static void test_task_create(void) {
    fd_dataset* ds = fd_new();
    fe_task* task = fe_task_new("test_task", FE_TASK_ACCURACY, ds);

    ASSERT_NOT_NULL(task);
    ASSERT_STR_EQ("test_task", task->name);
    ASSERT_INT_EQ(FE_TASK_ACCURACY, task->type);

    fe_task_free(task);
    fd_free(ds);
}

static void test_task_set_fields(void) {
    fe_task* task = fe_task_new("test", FE_TASK_ACCURACY, NULL);

    fe_task_set_fields(task, "input_text", "output_label");

    ASSERT_STR_EQ("input_text", task->input_field);
    ASSERT_STR_EQ("output_label", task->target_field);

    fe_task_free(task);
}

static void test_task_fewshot(void) {
    fe_task* task = fe_task_new("test", FE_TASK_ACCURACY, NULL);

    ASSERT_INT_EQ(0, task->num_fewshot);

    fe_task_set_fewshot(task, 5);
    ASSERT_INT_EQ(5, task->num_fewshot);

    fe_task_set_fewshot(task, -1);  /* Should clamp to 0 */
    ASSERT_INT_EQ(0, task->num_fewshot);

    fe_task_free(task);
}

/* ============================================================= */
/* Evaluator Tests                                               */
/* ============================================================= */

static void test_evaluator_create(void) {
    ft_tokenizer* tok = ft_new();
    fe_evaluator* eval = fe_new(tok);

    ASSERT_NOT_NULL(eval);

    fe_free(eval);
    ft_free(tok);
}

static void test_evaluator_add_task(void) {
    ft_tokenizer* tok = ft_new();
    fe_evaluator* eval = fe_new(tok);

    fd_dataset* ds = fd_new();
    fe_task* task = fe_task_new("task1", FE_TASK_ACCURACY, ds);

    fe_add_task(eval, task);

    fe_free(eval);
    fe_task_free(task);
    fd_free(ds);
    ft_free(tok);

    ASSERT_TRUE(1);  /* Didn't crash */
}

/* Dummy model function for testing */
static float* dummy_model(int* input_ids, int input_len, int* output_len, void* ctx) {
    (void)input_ids;
    (void)ctx;

    *output_len = input_len;
    float* output = calloc(input_len, sizeof(float));

    /* Return log probs of -1 */
    for (int i = 0; i < input_len; i++) {
        output[i] = -1.0f;
    }

    return output;
}

static void test_run_perplexity_task(void) {
    ft_tokenizer* tok = ft_new();
    ft_add_token(tok, "h");
    ft_add_token(tok, "e");
    ft_add_token(tok, "l");
    ft_add_token(tok, "o");

    fd_dataset* ds = fd_new();
    fd_sample* s = fd_sample_new();
    fd_sample_set(s, "text", "hello");
    fd_add_sample(ds, s);

    fe_task* task = fe_task_new("ppl", FE_TASK_PERPLEXITY, ds);
    fe_task_set_fields(task, "text", NULL);

    fe_evaluator* eval = fe_new(tok);
    fe_add_task(eval, task);

    fe_result* result = fe_run_task(eval, task, dummy_model, NULL);

    ASSERT_NOT_NULL(result);
    ASSERT_STR_EQ("ppl", result->task_name);
    ASSERT_TRUE(result->score > 0);
    ASSERT_INT_EQ(1, result->num_samples);

    fe_result_free(result);
    fe_free(eval);
    fe_task_free(task);
    fd_free(ds);
    ft_free(tok);
}

/* ============================================================= */
/* Result Tests                                                  */
/* ============================================================= */

static void test_result_free(void) {
    fe_result_free(NULL);  /* Should not crash */
    ASSERT_TRUE(1);
}

static void test_results_free(void) {
    fe_results_free(NULL, 0);  /* Should not crash */
    ASSERT_TRUE(1);
}

/* ============================================================= */
/* Export Tests                                                  */
/* ============================================================= */

static void test_export_json(void) {
    const char* path = "/tmp/test_foundry_eval.json";

    /* Create dummy results */
    fe_result** results = calloc(2, sizeof(fe_result*));

    results[0] = calloc(1, sizeof(fe_result));
    results[0]->task_name = strdup("task1");
    results[0]->score = 0.95;
    results[0]->num_samples = 100;

    results[1] = calloc(1, sizeof(fe_result));
    results[1]->task_name = strdup("task2");
    results[1]->score = 0.80;
    results[1]->num_samples = 50;

    bool success = fe_export_json(results, 2, path);
    ASSERT_TRUE(success);

    /* Check file exists */
    FILE* f = fopen(path, "r");
    ASSERT_NOT_NULL(f);
    fclose(f);

    /* Cleanup */
    fe_results_free(results, 2);
    unlink(path);
}

/* ============================================================= */
/* Built-in Loaders Tests                                        */
/* ============================================================= */

static void test_load_perplexity_task(void) {
    const char* path = "/tmp/test_foundry_ppl.jsonl";

    FILE* f = fopen(path, "w");
    fprintf(f, "{\"text\": \"hello world\"}\n");
    fprintf(f, "{\"text\": \"test data\"}\n");
    fclose(f);

    fe_task* task = fe_load_perplexity(path, "text");

    ASSERT_NOT_NULL(task);
    ASSERT_STR_EQ("perplexity", task->name);
    ASSERT_INT_EQ(FE_TASK_PERPLEXITY, task->type);
    ASSERT_NOT_NULL(task->dataset);
    ASSERT_INT_EQ(2, fd_len(task->dataset));

    fd_free(task->dataset);
    fe_task_free(task);
    unlink(path);
}

static void test_load_classification_task(void) {
    const char* path = "/tmp/test_foundry_cls.jsonl";

    FILE* f = fopen(path, "w");
    fprintf(f, "{\"sentence\": \"good\", \"label\": \"1\"}\n");
    fprintf(f, "{\"sentence\": \"bad\", \"label\": \"0\"}\n");
    fclose(f);

    fe_task* task = fe_load_classification(path, "sentence", "label");

    ASSERT_NOT_NULL(task);
    ASSERT_STR_EQ("classification", task->name);
    ASSERT_INT_EQ(FE_TASK_ACCURACY, task->type);

    fd_free(task->dataset);
    fe_task_free(task);
    unlink(path);
}

/* ============================================================= */
/* Edge Cases                                                    */
/* ============================================================= */

static void test_null_safety(void) {
    ASSERT_FLT_NEAR(0.0, fe_perplexity(NULL, 0), 0.001);
    ASSERT_FLT_NEAR(0.0, fe_accuracy(NULL, NULL, 0), 0.001);
    ASSERT_FLT_NEAR(0.0, fe_f1_score(NULL, NULL, 0), 0.001);

    ASSERT_NULL(fe_run_task(NULL, NULL, NULL, NULL));
    ASSERT_NULL(fe_run(NULL, NULL, NULL));

    fe_task_free(NULL);
    fe_free(NULL);

    ASSERT_TRUE(1);
}

static void test_empty_dataset(void) {
    ft_tokenizer* tok = ft_new();
    fd_dataset* ds = fd_new();  /* Empty */

    fe_task* task = fe_task_new("empty", FE_TASK_ACCURACY, ds);
    fe_evaluator* eval = fe_new(tok);

    fe_result* result = fe_run_task(eval, task, dummy_model, NULL);

    /* Should handle gracefully */
    ASSERT_NULL(result);  /* Empty dataset returns NULL */

    fe_free(eval);
    fe_task_free(task);
    fd_free(ds);
    ft_free(tok);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running eval tests...\n\n");

    /* Metrics tests */
    RUN_TEST(perplexity);
    RUN_TEST(perplexity_perfect);
    RUN_TEST(accuracy);
    RUN_TEST(accuracy_perfect);
    RUN_TEST(f1_score);

    /* Task tests */
    RUN_TEST(task_create);
    RUN_TEST(task_set_fields);
    RUN_TEST(task_fewshot);

    /* Evaluator tests */
    RUN_TEST(evaluator_create);
    RUN_TEST(evaluator_add_task);
    RUN_TEST(run_perplexity_task);

    /* Result tests */
    RUN_TEST(result_free);
    RUN_TEST(results_free);

    /* Export tests */
    RUN_TEST(export_json);

    /* Built-in loaders */
    RUN_TEST(load_perplexity_task);
    RUN_TEST(load_classification_task);

    /* Edge cases */
    RUN_TEST(null_safety);
    RUN_TEST(empty_dataset);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
