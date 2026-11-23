/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_EVAL_H
#define FOUNDRY_EVAL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../data/data.h"
#include "../tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Model evaluation framework.
 * Replaces lm-eval-harness basics.
 */

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* Evaluation task types */
typedef enum {
    FE_TASK_PERPLEXITY,      /* Language modeling perplexity */
    FE_TASK_ACCURACY,        /* Classification accuracy */
    FE_TASK_MULTIPLE_CHOICE, /* Multiple choice (A/B/C/D) */
    FE_TASK_GENERATION,      /* Text generation */
} fe_task_type;

/* Single evaluation result */
typedef struct {
    char* task_name;
    fe_task_type type;
    double score;            /* Main metric (perplexity, accuracy, etc.) */
    double* subscores;       /* Additional metrics */
    int num_subscores;
    int num_samples;
    double elapsed_seconds;
} fe_result;

/* Evaluation task definition */
typedef struct {
    char* name;
    fe_task_type type;
    fd_dataset* dataset;
    char* input_field;       /* Field name for input text */
    char* target_field;      /* Field name for target/label */
    int num_fewshot;         /* Number of few-shot examples */
} fe_task;

/* Model callback: takes input tokens, returns logits or output tokens */
typedef float* (*fe_model_fn)(int* input_ids, int input_len, int* output_len, void* ctx);

/* Evaluator */
typedef struct fe_evaluator fe_evaluator;

/* ============================================================= */
/* Evaluator                                                     */
/* ============================================================= */

/* Create evaluator with tokenizer */
fe_evaluator* fe_new(ft_tokenizer* tokenizer);

/* Free evaluator */
void fe_free(fe_evaluator* eval);

/* Add evaluation task */
void fe_add_task(fe_evaluator* eval, fe_task* task);

/* Run evaluation on all tasks */
fe_result** fe_run(fe_evaluator* eval, fe_model_fn model_fn, void* model_ctx);

/* Run single task */
fe_result* fe_run_task(fe_evaluator* eval, fe_task* task,
                       fe_model_fn model_fn, void* model_ctx);

/* Free result */
void fe_result_free(fe_result* result);

/* Free result array */
void fe_results_free(fe_result** results, int num_results);

/* ============================================================= */
/* Built-in Metrics                                              */
/* ============================================================= */

/* Compute perplexity from log probabilities */
double fe_perplexity(float* logprobs, int len);

/* Compute accuracy from predictions and labels */
double fe_accuracy(int* predictions, int* labels, int n);

/* Compute top-k accuracy */
double fe_accuracy_topk(float** logits, int* labels, int n, int vocab_size, int k);

/* F1 score */
double fe_f1_score(int* predictions, int* labels, int n);

/* ============================================================= */
/* Task Creation                                                 */
/* ============================================================= */

/* Create task from dataset */
fe_task* fe_task_new(const char* name, fe_task_type type, fd_dataset* dataset);

/* Free task */
void fe_task_free(fe_task* task);

/* Set task fields */
void fe_task_set_fields(fe_task* task, const char* input_field, const char* target_field);

/* Set few-shot count */
void fe_task_set_fewshot(fe_task* task, int num_fewshot);

/* ============================================================= */
/* Built-in Tasks                                                */
/* ============================================================= */

/* Load HellaSwag-style multiple choice task from JSONL */
fe_task* fe_load_hellaswag(const char* path);

/* Load simple text classification task */
fe_task* fe_load_classification(const char* path, const char* input_field,
                                 const char* label_field);

/* Load language modeling perplexity task */
fe_task* fe_load_perplexity(const char* path, const char* text_field);

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

/* Print result */
void fe_print_result(fe_result* result);

/* Print all results */
void fe_print_results(fe_result** results, int num_results);

/* Export results to JSON file */
bool fe_export_json(fe_result** results, int num_results, const char* path);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_EVAL_H */
