# eval

Model evaluation framework. Python counterpart: lm-eval-harness basics.

**Header:** `eval/eval.h`
**Source:** `eval/eval.c`
**Status:** API Complete - Pending Verification

---

## Overview

The eval module provides:
- Perplexity evaluation
- Accuracy and Top-K accuracy
- F1 score
- Multiple choice tasks (HellaSwag-style)
- Few-shot prompting support

---

## Quick Start

```c
#include "eval/eval.h"

// Create evaluator
fe_evaluator* eval = fe_new(tokenizer);

// Load task
fe_task* task = fe_load_hellaswag("hellaswag.jsonl");
fe_add_task(eval, task);

// Define model callback
float* my_model(int* input_ids, int input_len, int* output_len, void* ctx) {
    // Run model, return logits
    *output_len = vocab_size;
    return logits;
}

// Run evaluation
fe_result** results = fe_run(eval, my_model, model_ctx);

// Print results
fe_print_results(results, 1);

// Cleanup
fe_results_free(results, 1);
fe_free(eval);
```

---

## API Reference

### Evaluator

```c
fe_evaluator* fe_new(ft_tokenizer* tokenizer);
void fe_free(fe_evaluator* eval);

void fe_add_task(fe_evaluator* eval, fe_task* task);
fe_result** fe_run(fe_evaluator* eval, fe_model_fn model_fn, void* model_ctx);
fe_result* fe_run_task(fe_evaluator* eval, fe_task* task,
                       fe_model_fn model_fn, void* model_ctx);
```

### Model Callback

```c
// Takes input tokens, returns logits or output tokens
typedef float* (*fe_model_fn)(int* input_ids, int input_len, int* output_len, void* ctx);
```

### Task Types

```c
typedef enum {
    FE_TASK_PERPLEXITY,      // Language modeling perplexity
    FE_TASK_ACCURACY,        // Classification accuracy
    FE_TASK_MULTIPLE_CHOICE, // Multiple choice (A/B/C/D)
    FE_TASK_GENERATION,      // Text generation
} fe_task_type;
```

### Task Creation

```c
fe_task* fe_task_new(const char* name, fe_task_type type, fd_dataset* dataset);
void fe_task_free(fe_task* task);
void fe_task_set_fields(fe_task* task, const char* input_field, const char* target_field);
void fe_task_set_fewshot(fe_task* task, int num_fewshot);

// Built-in loaders
fe_task* fe_load_hellaswag(const char* path);
fe_task* fe_load_classification(const char* path, const char* input_field,
                                 const char* label_field);
fe_task* fe_load_perplexity(const char* path, const char* text_field);
```

### Results

```c
typedef struct {
    char* task_name;
    fe_task_type type;
    double score;            // Main metric
    double* subscores;       // Additional metrics
    int num_subscores;
    int num_samples;
    double elapsed_seconds;
} fe_result;

void fe_result_free(fe_result* result);
void fe_results_free(fe_result** results, int num_results);
void fe_print_result(fe_result* result);
void fe_print_results(fe_result** results, int num_results);
bool fe_export_json(fe_result** results, int num_results, const char* path);
```

### Built-in Metrics

```c
double fe_perplexity(float* logprobs, int len);
double fe_accuracy(int* predictions, int* labels, int n);
double fe_accuracy_topk(float** logits, int* labels, int n, int vocab_size, int k);
double fe_f1_score(int* predictions, int* labels, int n);
```

---

## See Also

- [data](data.md) - Dataset loading
- [tokenizer](tokenizer.md) - Tokenization
