/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================= */
/* Task                                                          */
/* ============================================================= */

fe_task* fe_task_new(const char* name, fe_task_type type, fd_dataset* dataset) {
    if (!name) return NULL;

    fe_task* task = calloc(1, sizeof(fe_task));
    if (!task) return NULL;

    task->name = strdup(name);
    task->type = type;
    task->dataset = dataset;  /* Does not take ownership */
    task->input_field = strdup("text");
    task->target_field = strdup("label");
    task->num_fewshot = 0;

    return task;
}

void fe_task_free(fe_task* task) {
    if (!task) return;

    free(task->name);
    free(task->input_field);
    free(task->target_field);
    free(task);
}

void fe_task_set_fields(fe_task* task, const char* input_field, const char* target_field) {
    if (!task) return;

    if (input_field) {
        free(task->input_field);
        task->input_field = strdup(input_field);
    }
    if (target_field) {
        free(task->target_field);
        task->target_field = strdup(target_field);
    }
}

void fe_task_set_fewshot(fe_task* task, int num_fewshot) {
    if (task) {
        task->num_fewshot = num_fewshot >= 0 ? num_fewshot : 0;
    }
}

/* ============================================================= */
/* Result                                                        */
/* ============================================================= */

static fe_result* result_new(const char* task_name, fe_task_type type) {
    fe_result* result = calloc(1, sizeof(fe_result));
    if (!result) return NULL;

    result->task_name = task_name ? strdup(task_name) : NULL;
    result->type = type;

    return result;
}

void fe_result_free(fe_result* result) {
    if (!result) return;

    free(result->task_name);
    free(result->subscores);
    free(result);
}

void fe_results_free(fe_result** results, int num_results) {
    if (!results) return;

    for (int i = 0; i < num_results; i++) {
        fe_result_free(results[i]);
    }
    free(results);
}

/* ============================================================= */
/* Metrics                                                       */
/* ============================================================= */

double fe_perplexity(float* logprobs, int len) {
    if (!logprobs || len <= 0) return 0.0;

    double sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += logprobs[i];
    }

    double avg_logprob = sum / len;
    return exp(-avg_logprob);
}

double fe_accuracy(int* predictions, int* labels, int n) {
    if (!predictions || !labels || n <= 0) return 0.0;

    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
    }

    return (double)correct / n;
}

double fe_accuracy_topk(float** logits, int* labels, int n, int vocab_size, int k) {
    if (!logits || !labels || n <= 0 || k <= 0) return 0.0;

    int correct = 0;

    for (int i = 0; i < n; i++) {
        /* Find top-k predictions */
        int* top_k = calloc(k, sizeof(int));
        float* top_vals = calloc(k, sizeof(float));

        for (int j = 0; j < k; j++) {
            top_vals[j] = -INFINITY;
        }

        for (int v = 0; v < vocab_size; v++) {
            float val = logits[i][v];
            /* Insert into top-k if larger than smallest */
            if (val > top_vals[k - 1]) {
                int pos = k - 1;
                while (pos > 0 && val > top_vals[pos - 1]) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_k[pos] = top_k[pos - 1];
                    pos--;
                }
                top_vals[pos] = val;
                top_k[pos] = v;
            }
        }

        /* Check if label is in top-k */
        for (int j = 0; j < k; j++) {
            if (top_k[j] == labels[i]) {
                correct++;
                break;
            }
        }

        free(top_k);
        free(top_vals);
    }

    return (double)correct / n;
}

double fe_f1_score(int* predictions, int* labels, int n) {
    if (!predictions || !labels || n <= 0) return 0.0;

    /* Binary F1 score */
    int tp = 0, fp = 0, fn = 0;

    for (int i = 0; i < n; i++) {
        if (predictions[i] == 1 && labels[i] == 1) tp++;
        else if (predictions[i] == 1 && labels[i] == 0) fp++;
        else if (predictions[i] == 0 && labels[i] == 1) fn++;
    }

    if (tp == 0) return 0.0;

    double precision = (double)tp / (tp + fp);
    double recall = (double)tp / (tp + fn);

    return 2.0 * precision * recall / (precision + recall);
}

/* ============================================================= */
/* Evaluator                                                     */
/* ============================================================= */

struct fe_evaluator {
    ft_tokenizer* tokenizer;
    fe_task** tasks;
    int num_tasks;
    int capacity;
};

fe_evaluator* fe_new(ft_tokenizer* tokenizer) {
    fe_evaluator* eval = calloc(1, sizeof(fe_evaluator));
    if (!eval) return NULL;

    eval->tokenizer = tokenizer;
    eval->capacity = 8;
    eval->tasks = calloc(eval->capacity, sizeof(fe_task*));

    return eval;
}

void fe_free(fe_evaluator* eval) {
    if (!eval) return;

    /* Note: doesn't free tasks, just the pointers */
    free(eval->tasks);
    free(eval);
}

void fe_add_task(fe_evaluator* eval, fe_task* task) {
    if (!eval || !task) return;

    if (eval->num_tasks >= eval->capacity) {
        eval->capacity *= 2;
        eval->tasks = realloc(eval->tasks, eval->capacity * sizeof(fe_task*));
    }

    eval->tasks[eval->num_tasks++] = task;
}

fe_result* fe_run_task(fe_evaluator* eval, fe_task* task,
                       fe_model_fn model_fn, void* model_ctx) {
    if (!eval || !task || !model_fn) return NULL;
    if (!task->dataset || fd_len(task->dataset) == 0) return NULL;

    fe_result* result = result_new(task->name, task->type);
    if (!result) return NULL;

    clock_t start = clock();

    int n = fd_len(task->dataset);
    result->num_samples = n;

    switch (task->type) {
        case FE_TASK_PERPLEXITY: {
            double total_logprob = 0.0;
            int total_tokens = 0;

            for (int i = 0; i < n; i++) {
                fd_sample* sample = fd_get(task->dataset, i);
                const char* text = fd_sample_get(sample, task->input_field);
                if (!text) continue;

                /* Tokenize */
                int len;
                int* ids = ft_encode(eval->tokenizer, text, &len);
                if (!ids || len == 0) {
                    free(ids);
                    continue;
                }

                /* Get model output (logits) */
                int output_len;
                float* logits = model_fn(ids, len, &output_len, model_ctx);

                if (logits && output_len > 0) {
                    /* Sum log probabilities (simplified) */
                    for (int t = 0; t < output_len; t++) {
                        /* Assume logits[t] is log prob of correct token */
                        total_logprob += logits[t];
                    }
                    total_tokens += output_len;
                    free(logits);
                }

                free(ids);
            }

            if (total_tokens > 0) {
                result->score = exp(-total_logprob / total_tokens);
            }
            break;
        }

        case FE_TASK_ACCURACY: {
            int correct = 0;

            for (int i = 0; i < n; i++) {
                fd_sample* sample = fd_get(task->dataset, i);
                const char* text = fd_sample_get(sample, task->input_field);
                const char* label_str = fd_sample_get(sample, task->target_field);
                if (!text || !label_str) continue;

                int len;
                int* ids = ft_encode(eval->tokenizer, text, &len);
                if (!ids) continue;

                int output_len;
                float* output = model_fn(ids, len, &output_len, model_ctx);

                if (output && output_len > 0) {
                    /* Find predicted class (argmax) */
                    int pred = 0;
                    float max_val = output[0];
                    for (int j = 1; j < output_len; j++) {
                        if (output[j] > max_val) {
                            max_val = output[j];
                            pred = j;
                        }
                    }

                    int label = atoi(label_str);
                    if (pred == label) correct++;

                    free(output);
                }

                free(ids);
            }

            result->score = (double)correct / n;
            break;
        }

        case FE_TASK_MULTIPLE_CHOICE:
        case FE_TASK_GENERATION:
            /* Simplified: just return placeholder */
            result->score = 0.0;
            break;
    }

    clock_t end = clock();
    result->elapsed_seconds = (double)(end - start) / CLOCKS_PER_SEC;

    return result;
}

fe_result** fe_run(fe_evaluator* eval, fe_model_fn model_fn, void* model_ctx) {
    if (!eval || !model_fn || eval->num_tasks == 0) return NULL;

    fe_result** results = calloc(eval->num_tasks, sizeof(fe_result*));
    if (!results) return NULL;

    for (int i = 0; i < eval->num_tasks; i++) {
        results[i] = fe_run_task(eval, eval->tasks[i], model_fn, model_ctx);
    }

    return results;
}

/* ============================================================= */
/* Built-in Task Loaders                                         */
/* ============================================================= */

fe_task* fe_load_hellaswag(const char* path) {
    fd_dataset* ds = fd_load_jsonl(path);
    if (!ds) return NULL;

    fe_task* task = fe_task_new("hellaswag", FE_TASK_MULTIPLE_CHOICE, ds);
    if (task) {
        fe_task_set_fields(task, "ctx", "label");
    }

    return task;
}

fe_task* fe_load_classification(const char* path, const char* input_field,
                                 const char* label_field) {
    fd_dataset* ds = fd_load_jsonl(path);
    if (!ds) return NULL;

    fe_task* task = fe_task_new("classification", FE_TASK_ACCURACY, ds);
    if (task) {
        fe_task_set_fields(task, input_field, label_field);
    }

    return task;
}

fe_task* fe_load_perplexity(const char* path, const char* text_field) {
    fd_dataset* ds = fd_load_jsonl(path);
    if (!ds) return NULL;

    fe_task* task = fe_task_new("perplexity", FE_TASK_PERPLEXITY, ds);
    if (task) {
        fe_task_set_fields(task, text_field, NULL);
    }

    return task;
}

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

void fe_print_result(fe_result* result) {
    if (!result) {
        printf("(null result)\n");
        return;
    }

    const char* type_name = "unknown";
    switch (result->type) {
        case FE_TASK_PERPLEXITY: type_name = "perplexity"; break;
        case FE_TASK_ACCURACY: type_name = "accuracy"; break;
        case FE_TASK_MULTIPLE_CHOICE: type_name = "multiple_choice"; break;
        case FE_TASK_GENERATION: type_name = "generation"; break;
    }

    printf("%s (%s): %.4f", result->task_name ? result->task_name : "unnamed",
           type_name, result->score);

    if (result->num_samples > 0) {
        printf(" (%d samples, %.2fs)", result->num_samples, result->elapsed_seconds);
    }
    printf("\n");
}

void fe_print_results(fe_result** results, int num_results) {
    if (!results) return;

    printf("\n=== Evaluation Results ===\n");
    for (int i = 0; i < num_results; i++) {
        fe_print_result(results[i]);
    }
    printf("==========================\n");
}

bool fe_export_json(fe_result** results, int num_results, const char* path) {
    if (!results || !path) return false;

    FILE* f = fopen(path, "w");
    if (!f) return false;

    fprintf(f, "{\n  \"results\": [\n");

    for (int i = 0; i < num_results; i++) {
        fe_result* r = results[i];
        if (!r) continue;

        fprintf(f, "    {\n");
        fprintf(f, "      \"task\": \"%s\",\n", r->task_name ? r->task_name : "");
        fprintf(f, "      \"score\": %.6f,\n", r->score);
        fprintf(f, "      \"num_samples\": %d,\n", r->num_samples);
        fprintf(f, "      \"elapsed_seconds\": %.2f\n", r->elapsed_seconds);
        fprintf(f, "    }%s\n", (i < num_results - 1) ? "," : "");
    }

    fprintf(f, "  ]\n}\n");
    fclose(f);

    return true;
}
