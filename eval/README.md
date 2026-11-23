# eval/

C implementation of LLM evaluation and benchmarking.

## Python counterpart

**lm-eval-harness** - https://github.com/EleutherAI/lm-evaluation-harness

## Why lm-eval-harness?

lm-eval-harness is the standard for evaluating language models:

- **Standardized benchmarks** - MMLU, HellaSwag, ARC, TruthfulQA, etc.
- **Reproducible results** - Consistent evaluation methodology
- **Multiple metrics** - Accuracy, perplexity, F1, exact match
- **Few-shot evaluation** - Test with N examples in context

Alternatives (HELM, OpenAI evals) cover different benchmark suites.

## Scope

This module implements:

1. **Core metrics** - Accuracy, perplexity, exact match
2. **Multiple choice** - For benchmarks like MMLU, HellaSwag
3. **Generation eval** - For open-ended tasks
4. **Few-shot prompting** - Construct prompts with examples
5. **Common benchmarks** - MMLU, HellaSwag, ARC, Winogrande

## Getting Started

### Dependencies

- tensor/ module (for model inference)
- tokenizer/ module (for text processing)
- data/ module (for loading benchmark data)

### Key Files to Study

From lm-eval-harness:
- `lm_eval/tasks/` - Task definitions
- `lm_eval/api/model.py` - Model interface
- `lm_eval/evaluator.py` - Evaluation loop

### Core API

> Original design sketch, not the implemented API. The real API is `eval/eval.h`; see
> [`docs/eval.md`](../docs/eval.md).

```c
// Initialize evaluator with model
fe_evaluator* fe_new(tm_model* model, ft_tokenizer* tokenizer);

// Run specific benchmark
fe_result* fe_run_task(fe_evaluator* e, const char* task, int num_fewshot);

// Available tasks
const char** fe_list_tasks(void);

// Results
typedef struct {
    const char* task;
    float accuracy;
    float stderr;
    int num_samples;
} fe_result;

void fe_print_result(fe_result* r);
void fe_save_results(fe_result** results, int n, const char* path);

// Metrics
float fe_accuracy(int* predictions, int* labels, int n);
float fe_perplexity(tm_model* model, const char** texts, int n);
float fe_exact_match(const char** predictions, const char** references, int n);

// Multiple choice evaluation
int fe_loglikelihood_choice(tm_model* model, const char* context,
                             const char** choices, int num_choices);
```

## Benchmark Format

Tasks are defined as JSON:
```json
{
  "task": "hellaswag",
  "type": "multiple_choice",
  "dataset": "hellaswag",
  "prompt_template": "{context}",
  "choices_key": "endings",
  "label_key": "label",
  "num_fewshot": 10
}
```

## Implementation Order

1. Core metrics (accuracy, perplexity)
2. Multiple choice evaluation
3. Few-shot prompt construction
4. HellaSwag benchmark
5. MMLU benchmark
6. ARC benchmark
7. Results reporting

## References

- [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HELM](https://crfm.stanford.edu/helm/)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
