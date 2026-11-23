/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rfa.h — RWKV-7 + Full Attention hybrid inference engine
 *
 * Pure C inference for hybrid models combining RWKV-7 recurrent blocks with
 * Qwen2 dense attention blocks (GQA, RoPE, SwiGLU). Built on Foundry.
 *
 * Designed to test whether pretrained Qwen attention weights solve the B=C
 * problem observed in HybridMem research with RWKV-X MoBA sparse attention.
 *
 * Prefix: rfa_
 *
 * Reuses rx_tokenizer, rx_sampling_params, and rx_sample from rwkvx.h
 * (same vocabulary and sampling logic — no duplication).
 */

#ifndef RFA_H
#define RFA_H

#include <stdint.h>
#include <stdbool.h>
#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Types
 * ================================================================ */

/* Opaque types */
typedef struct rfa_model rfa_model;
typedef struct rfa_state rfa_state;

/* Model configuration (extracted from safetensors metadata) */
typedef struct {
    int n_embd;          /* hidden dimension: 1536 for 1.5B */
    int n_q_heads;       /* query heads: 12 for 1.5B (GQA) */
    int n_kv_heads;      /* key/value heads: 2 for 1.5B (GQA) */
    int head_dim;        /* per-head dim for attention: 128 */
    int n_wkv_heads;     /* WKV-7 heads: 24 for 1.5B */
    int wkv_head_size;   /* WKV-7 head dim: 64 for 1.5B */
    int intermediate_size; /* SwiGLU FFN intermediate: 8960 */
    int n_layers;        /* total layers: 28 for 1.5B */
    int n_rwkv_layers;   /* RWKV-7 layers: 21 */
    int n_qwen_layers;   /* Qwen attention layers: 7 */
    int vocab_size;      /* vocabulary size: 151936 */
    float rope_theta;    /* RoPE theta: 1000000.0 */
    float rms_norm_eps;  /* RMSNorm epsilon: 1e-6 */
} rfa_config;

/* ================================================================
 * Model lifecycle
 * ================================================================ */

/* Load hybrid model from merged safetensors file.
 * The file must contain metadata identifying block types (block_types).
 * Returns NULL on failure. */
rfa_model* rfa_model_load(const char *safetensors_path);

/* Load and quantize all linear weights to INT8. */
rfa_model* rfa_model_load_q8(const char *safetensors_path);

/* Free model and all weights. */
void rfa_model_free(rfa_model *model);

/* Get model configuration. */
rfa_config rfa_model_config(const rfa_model *model);

/* ================================================================
 * State management
 * ================================================================ */

/* Create zeroed state. Returns NULL on failure.
 * max_cache_len: max KV cache per Qwen attention layer (default 2048). */
rfa_state* rfa_state_new(const rfa_model *model, int max_cache_len);

/* Free state. */
void rfa_state_free(rfa_state *state);

/* Reset state to initial zeros. */
void rfa_state_reset(rfa_state *state);

/* ================================================================
 * Inference
 * ================================================================ */

/* Forward pass for a single token. Returns logits [vocab_size].
 * Caller must tm_free() the returned tensor. */
tm_tensor* rfa_forward(rfa_model *model, uint32_t token, rfa_state *state);

/* ================================================================
 * Forward with callback (for training)
 * ================================================================ */

/* Block callback for training hooks.
 * phase: 0=pre-attention (before Qwen block runs), 1=post-attention
 * block_idx: absolute layer index [0, n_layers)
 * qwen_idx: index among Qwen blocks only [0, n_qwen_layers)
 * x: current hidden state [C]
 * Return 0 to continue normal forward. */
typedef int (*rfa_block_callback)(int phase, int block_idx, int qwen_idx,
                                   float *x, void *user_data);

/* Forward pass with per-block callback. Returns logits [vocab_size]. */
tm_tensor* rfa_forward_ex(rfa_model *model, uint32_t token, rfa_state *state,
                           rfa_block_callback callback, void *user_data);

/* ================================================================
 * Weight accessors (for training backward pass)
 * ================================================================ */

/* Qwen block weight pointers (not owned — do not free) */
typedef struct {
    /* RMSNorm weights [C] */
    float *ln1_w;      /* input_layernorm */
    float *ln2_w;      /* post_attention_layernorm */
    /* Attention projections (transposed: [in, out] for fast_matvec) */
    float *q_proj_w;   /* [C, C] */
    float *q_proj_b;   /* [C] or NULL */
    float *k_proj_w;   /* [C, kv_dim] */
    float *k_proj_b;   /* [kv_dim] or NULL */
    float *v_proj_w;   /* [C, kv_dim] */
    float *v_proj_b;   /* [kv_dim] or NULL */
    float *o_proj_w;   /* [C, C] */
    /* SwiGLU FFN (transposed for fast_matvec) */
    float *gate_proj;  /* [C, inter] */
    float *up_proj;    /* [C, inter] */
    float *down_proj;  /* [inter, C] */
    int inter_size;    /* intermediate_size */
} rfa_qwen_weights;

/* Get raw weight pointers for a Qwen block by its Qwen index [0, n_qwen).
 * Returns 0 on success, -1 on failure. */
int rfa_model_get_qwen_weights(const rfa_model *model, int qwen_idx,
                                rfa_qwen_weights *out);

/* Get head projection weight. Returns pointer to [C, vocab] (not owned). */
const float* rfa_model_get_head_weight(const rfa_model *model);

/* Get final RMSNorm weight. Returns pointer to [C] (not owned). */
const float* rfa_model_get_ln_out_w(const rfa_model *model);

/* Get KV cache for a Qwen block. Returns cache_len.
 * Pointers set to internal cache buffers (not owned). */
int rfa_state_get_kv_cache(const rfa_state *state, int qwen_idx,
                            const float **k_cache, const float **v_cache);

/* Get absolute position counter */
int rfa_state_get_position(const rfa_state *state);

/* ================================================================
 * Internal types (exposed for testing, not part of public API)
 * ================================================================ */

/* RWKV-7 time mixer weights (per RWKV block) */
typedef struct {
    /* Token shift lerp coefficients [1, 1, C] squeezed to [C] */
    tm_tensor *x_r, *x_w, *x_k, *x_v, *x_a;
    /* Decay LoRA: w = exp(-softplus(w0 + tanh(xw @ w1) @ w2) - 0.5) */
    tm_tensor *w0, *w1, *w2;
    /* Gate a LoRA: a = sigmoid(a0 + (xa @ a1) @ a2) */
    tm_tensor *a0, *a1, *a2;
    /* Value first-token blend LoRA (used for layer > 0) */
    tm_tensor *v0, *v1, *v2;
    /* Key normalization */
    tm_tensor *k_k, *k_a, *r_k;
    /* Linear projections [C, C] */
    tm_tensor *receptance_w, *key_w, *value_w, *output_w;
} rfa_rwkv_attn;

/* Qwen2 dense attention weights (per Qwen block) */
typedef struct {
    /* Q/K/V/O projections — Q has bias, K has bias, V has bias, O has no bias */
    tm_tensor *q_proj_w, *q_proj_b;   /* [C, C] + [C] */
    tm_tensor *k_proj_w, *k_proj_b;   /* [C, kv_dim] + [kv_dim] */
    tm_tensor *v_proj_w, *v_proj_b;   /* [C, kv_dim] + [kv_dim] */
    tm_tensor *o_proj_w;              /* [C, C] */
} rfa_qwen_attn;

/* Unified block (all 28 layers share this struct) */
typedef struct {
    /* RMSNorm weights [C] — no bias */
    tm_tensor *ln1_w;   /* input_layernorm */
    tm_tensor *ln2_w;   /* post_attention_layernorm */

    /* SwiGLU FFN (all layers, same structure) */
    tm_tensor *gate_proj;  /* [intermediate, C] = [8960, 1536] */
    tm_tensor *up_proj;    /* [intermediate, C] = [8960, 1536] */
    tm_tensor *down_proj;  /* [C, intermediate] = [1536, 8960] */

    /* Block type */
    int is_qwen;  /* 0 = RWKV-7, 1 = Qwen attention */

    /* Exactly one of these is populated (the other is zeroed) */
    rfa_rwkv_attn rwkv;
    rfa_qwen_attn qwen;
} rfa_block;

/* Per-block RWKV-7 recurrent state */
typedef struct {
    tm_tensor *att_x_prev;   /* [C] previous token for shift */
    tm_tensor *att_kv;       /* [n_wkv_heads, wkv_head_size, wkv_head_size] */
} rfa_rwkv_state;

/* Per-block Qwen KV cache */
typedef struct {
    float *k_cache;       /* [max_cache, n_kv_heads * head_dim] */
    float *v_cache;       /* [max_cache, n_kv_heads * head_dim] */
    int cache_len;        /* current number of tokens in cache */
    int max_cache_len;    /* capacity */
} rfa_kv_cache;

#ifdef __cplusplus
}
#endif

#endif /* RFA_H */
