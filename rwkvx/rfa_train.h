/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rfa_train.h — LoRA training for RFA hybrid model (Qwen blocks only)
 *
 * Trains Qwen GQA attention blocks to work with frozen RWKV-7 blocks.
 * 7 LoRA layers per Qwen block: Q, K, V, O, gate, up, down.
 * Gradient flows through RWKV blocks via residual connections.
 *
 * Prefix: rfa_ (for RFA training)
 */

#ifndef RFA_TRAIN_H
#define RFA_TRAIN_H

#include <stdint.h>
#include <stdbool.h>
#include "rwkvx/rfa.h"
#include "lora/lora.h"
#include "optim/optim.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Training example (pre-tokenized binary format)
 * ================================================================ */

typedef struct {
    uint32_t *input_ids;   /* [seq_len] */
    uint32_t *target_ids;  /* [seq_len] */
    int seq_len;
} rfa_train_example;

/* Free a single training example */
void rfa_train_example_free(rfa_train_example *ex);

/* Free array of training examples */
void rfa_train_examples_free(rfa_train_example *examples, int n);

/* Load pre-tokenized binary file (from pre_tokenize.py).
 * Format: [n_examples: u32] [seq_len: u32, tokens: u32[seq_len]] x N
 * Creates input/target pairs shifted by 1.
 * Returns malloc'd array, sets *n_examples_out. */
rfa_train_example* rfa_load_pretokenized(
    const char *path, int max_seq_len, int *n_examples_out
);

/* ================================================================
 * Saved activations (for backward pass through Qwen blocks)
 * ================================================================ */

typedef struct {
    float *x_input;        /* [C] block input (saved during forward) */
    int cache_len;         /* KV cache length at this token */
} rfa_qwen_saved;

/* ================================================================
 * Training config
 * ================================================================ */

typedef struct {
    /* LoRA */
    int lora_rank;             /* LoRA rank (default: 32) */
    float lora_alpha;          /* LoRA alpha (default: 64) */

    /* Training */
    float lr;                  /* Learning rate (default: 2e-5) */
    float weight_decay;        /* AdamW weight decay (default: 0.01) */
    int epochs;                /* Number of epochs (default: 3) */
    int max_seq_len;           /* Max sequence length (default: 256) */
    int warmup_steps;          /* LR warmup steps (default: 100) */
    int save_every;            /* Checkpoint every N steps (default: 1000) */
    int log_every;             /* Log every N steps (default: 10) */
    float grad_clip;           /* Max gradient norm (default: 1.0) */
    int max_cache;             /* Max KV cache per Qwen block (default: 2048) */
    int n_threads;             /* Parallel training threads (default: 1) */
    bool quantize;             /* Use INT8 base weights (default: false) */

    /* Paths */
    const char *model_path;
    const char *data_path;     /* Pre-tokenized binary file */
    const char *output_dir;    /* Checkpoint output directory */
} rfa_train_config;

/* Return default training config */
rfa_train_config rfa_train_config_defaults(void);

/* ================================================================
 * Backward pass primitives
 * ================================================================ */

/* RMSNorm backward: given d_output, original input x, weight
 * Computes d_x (gradient to propagate through) */
void rfa_rms_norm_backward(
    const float *d_output,  /* [C] gradient from above */
    const float *x,         /* [C] original input to RMSNorm */
    const float *weight,    /* [C] RMSNorm weight */
    float *d_x,             /* [C] gradient to propagate (output) */
    int c, float eps
);

/* SwiGLU backward: given d_output [C], computes d_input [C].
 * Requires recomputed gate/up activations. */
void rfa_swiglu_backward(
    const float *d_output,  /* [C] gradient from above */
    const float *x_input,   /* [C] input to SwiGLU */
    const rfa_qwen_weights *w,
    fl_layer **lora,        /* [7] LoRA layers: Q,K,V,O,gate,up,down */
    float *d_input,         /* [C] gradient to propagate (output) */
    int c, int inter
);

/* GQA attention backward: given d_y [C], computes d_q, d_k, d_v.
 * Uses saved attention weights from forward. */
void rfa_gqa_attention_backward(
    const float *d_y,       /* [C] gradient into attention output */
    const float *q,         /* [C] query vector */
    const float *k_cache,   /* [cache_len, kv_dim] */
    const float *v_cache,   /* [cache_len, kv_dim] */
    const float *attn_weights, /* [n_q_heads, cache_len] saved softmax */
    int cache_len,
    float *d_q,             /* [C] output */
    float *d_k_cur,         /* [kv_dim] output: gradient for current token's K */
    float *d_v_cur,         /* [kv_dim] output: gradient for current token's V */
    int n_q, int n_kv, int hd
);

/* ================================================================
 * Full Qwen block backward
 * ================================================================ */

/* Backward through one Qwen block. Computes LoRA gradients.
 * d_output: [C] gradient from above
 * d_input: [C] gradient to propagate down (output) */
void rfa_qwen_backward(
    const rfa_model *model,
    const rfa_state *state,
    int qwen_idx,
    const rfa_qwen_saved *saved,
    fl_layer **lora,        /* [7] LoRA layers for this block */
    const float *d_output,  /* [C] gradient from above */
    float *d_input,         /* [C] gradient to propagate down */
    rfa_config *cfg
);

/* ================================================================
 * Training step result
 * ================================================================ */

typedef struct {
    int step;
    float loss;
    float lr;
    int tokens_seen;
    double elapsed_sec;
} rfa_train_step_result;

/* ================================================================
 * LoRA adapter + training
 * ================================================================ */

/* Create LoRA adapter for Qwen blocks.
 * Creates 7 LoRA layers per Qwen block: Q, K, V, O, gate, up, down.
 * Returns adapter with all layers. */
fl_adapter* rfa_create_qwen_lora(const rfa_model *model, int rank, float alpha);

/* Collect all trainable parameters from adapter.
 * Returns malloc'd array of tm_tensor* pointers (A and B per layer).
 * Sets *n_params_out. */
tm_tensor** rfa_collect_params(fl_adapter *adapter, int *n_params_out);

/* Single training step: forward + loss + backward + optimizer step. */
rfa_train_step_result rfa_train_step(
    rfa_model *model,
    rfa_state *state,
    fl_adapter *adapter,
    fo_optimizer *optimizer,
    rfa_train_example *example,
    rfa_train_config *config
);

/* Full training loop over all examples for all epochs. */
void rfa_train_loop(
    rfa_model *model,
    rfa_train_config *config,
    rfa_train_example *examples,
    int n_examples
);

/* ================================================================
 * Parallel training + LoRA averaging
 * ================================================================ */

/* Average N LoRA adapters element-wise.
 * All adapters must have the same structure (same layers, same dimensions).
 * Returns a new adapter with averaged A and B matrices. */
fl_adapter* rfa_lora_average(fl_adapter **adapters, int n_adapters);

/* Parallel training: spawns n_threads workers, each training on a chunk
 * of the dataset. Merges adapters via averaging when all complete.
 * Equivalent to calling rfa_train_loop on each chunk, then averaging. */
void rfa_train_parallel(
    rfa_model *model,
    rfa_train_config *config,
    rfa_train_example *examples,
    int n_examples,
    int n_threads
);

#ifdef __cplusplus
}
#endif

#endif /* RFA_TRAIN_H */
