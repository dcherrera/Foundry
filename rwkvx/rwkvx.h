/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rwkvx.h — RWKV-X inference engine built on Foundry
 *
 * Pure C implementation of the RWKV-X hybrid model (RWKV-7 recurrent blocks
 * interleaved with sparse attention blocks). Loads safetensors weights,
 * runs inference, and generates text.
 *
 * Prefix: rx_
 */

#ifndef RWKVX_H
#define RWKVX_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "tensor/tensor.h"
#include "hybridmem/hybridmem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Types
 * ================================================================ */

/* Opaque types */
typedef struct rx_model rx_model;
typedef struct rx_state rx_state;
typedef struct rx_tokenizer rx_tokenizer;

/* Model configuration (extracted from weight shapes) */
typedef struct {
    int n_embd;          /* embedding dim: 768 for 0.2B */
    int n_head;          /* attention heads: 12 for 0.2B */
    int head_size;       /* per-head dim: always 64 */
    int n_rwkv_layer;    /* RWKV-7 recurrent layers: 12 for 0.2B */
    int n_attn_layer;    /* sparse attention layers: 4 for 0.2B */
    int vocab_size;      /* vocabulary size: 65536 */
} rx_config;

/* HybridMem configuration for disk-backed attention memory */
typedef struct {
    const char *store_dir;   /* Directory for persistent storage (NULL = disabled) */
    int max_cache_len;       /* KV cache size per attention layer (0 = use default 2048) */
    int top_k;               /* Number of entries to retrieve from disk (default: 5) */
    float evict_fraction;    /* Fraction of cache to evict when full (default: 0.5) */
} rx_hybridmem_config;

/* Sampling parameters */
typedef struct {
    float temperature;      /* 0 = greedy */
    float top_p;            /* nucleus sampling threshold */
    int   top_k;            /* top-k filtering (0 = disabled) */
    float frequency_penalty;
    float presence_penalty;
} rx_sampling_params;

/* ================================================================
 * Model lifecycle
 * ================================================================ */

/* Load model from safetensors file. Returns NULL on failure. */
rx_model* rx_model_load(const char *safetensors_path);

/* Load model from safetensors and quantize all linear weights to INT8.
 * Reduces memory by ~4x and improves bandwidth-bound inference speed.
 * Output quality may differ slightly from f32 — check greedy decode. */
rx_model* rx_model_load_q8(const char *safetensors_path);

/* Free model and all associated weights. */
void rx_model_free(rx_model *model);

/* Get model configuration. */
rx_config rx_model_config(const rx_model *model);

/* ================================================================
 * State management
 * ================================================================ */

/* Create a new zeroed state for the given model. Returns NULL on failure. */
rx_state* rx_state_new(const rx_model *model);

/* Create a new state with HybridMem disk-backed memory enabled.
 * When the KV cache fills, old entries are evicted to disk via fm_put().
 * Before attention, relevant past context is retrieved via fm_query().
 * Returns NULL on failure. */
rx_state* rx_state_new_with_hybridmem(const rx_model *model,
                                       const rx_hybridmem_config *hm_config);

/* Get default HybridMem config. */
rx_hybridmem_config rx_hybridmem_config_defaults(void);

/* Free state (also frees HybridMem store if present). */
void rx_state_free(rx_state *state);

/* Flush HybridMem store to disk. No-op if HybridMem is disabled.
 * Returns 0 on success, -1 on error. */
int rx_state_flush(rx_state *state);

/* Reset state to initial zeroed values. */
void rx_state_reset(rx_state *state);

/* ================================================================
 * Instrumentation (optional, zero-cost when disabled)
 * ================================================================ */

/* Enable per-head attention weight + entropy logging to a CSV file.
 * Pass NULL to disable. Resets the step counter.
 * CSV format: token_step,layer,head,ret_mass,cache_mass,entropy */
void rx_state_set_attn_log(rx_state *state, FILE *fp);

/* Enable embedding capture from attention layers.
 * buffer must have space for max_entries * n_embd floats.
 * layer_idx: which attention layer to capture from (-1 = first one encountered).
 * Call rx_state_embed_capture_count() after inference to get count. */
void rx_state_set_embed_capture(rx_state *state, float *buffer,
                                 int max_entries, int layer_idx);

/* Get number of embeddings captured so far. */
int rx_state_embed_capture_count(const rx_state *state);

/* Get the HybridMem store from a state (for direct querying).
 * Returns NULL if HybridMem is disabled. */
fm_store* rx_state_get_hm_store(const rx_state *state);

/* ================================================================
 * Inference
 * ================================================================ */

/* Forward pass for a single token. Returns logits [vocab_size].
 * Caller must tm_free() the returned tensor. */
tm_tensor* rx_forward(rx_model *model, uint32_t token, rx_state *state);

/* Forward pass for a batch of tokens (prefill). Returns logits for the
 * last token only: [vocab_size]. Caller must tm_free(). */
tm_tensor* rx_forward_batch(rx_model *model, const uint32_t *tokens,
                            int n_tokens, rx_state *state);

/* ================================================================
 * Training support
 * ================================================================ */

/* Block callback for training forward pass.
 * Called BEFORE and AFTER each attention block.
 * phase: 0 = before attention block, 1 = after attention block
 * attn_idx: which attention block (0-based)
 * x: current activation vector [n_embd], can be read/modified
 * user_data: opaque pointer passed through from rx_forward_ex
 *
 * Return 0 to continue normal processing, 1 to skip this block
 * (caller handles the block processing instead). */
typedef int (*rx_block_callback)(int phase, int attn_idx, float *x,
                                  void *user_data);

/* Extended forward pass with block-level callbacks.
 * Same as rx_forward but calls cb before/after each attention block.
 * If cb returns 1 for phase=0, the attention block forward is SKIPPED
 * and the caller is expected to process x in-place.
 * logits_out: if non-NULL, receives pointer to logits data [vocab_size]
 *             (only valid until next rx_forward call — not a new allocation).
 * Returns logits tensor (caller must tm_free()). */
tm_tensor* rx_forward_ex(rx_model *model, uint32_t token, rx_state *state,
                          rx_block_callback cb, void *user_data);

/* Get raw k_cache and v_cache pointers for an attention layer.
 * Useful for training backward pass. Returns cache_len. */
int rx_state_get_kv_cache(const rx_state *state, int attn_idx,
                           const float **k_cache_out,
                           const float **v_cache_out);

/* ================================================================
 * Weight accessors (for training backward pass)
 *
 * These expose internal weight pointers needed by training code.
 * All returned pointers are owned by the model — do NOT free them.
 * ================================================================ */

/* Get attention block weight pointers for a given attention layer index.
 * All weights are [in_dim, out_dim] row-major (pre-transposed for fast_matvec).
 * Returns 0 on success, -1 on invalid index. */
typedef struct {
    const float *ln1_w, *ln1_b;              /* [C] pre-attn LayerNorm */
    const float *ln2_w, *ln2_b;              /* [C] pre-FFN LayerNorm */
    const float *receptance_w;               /* [C, C] Q projection */
    const float *key_w;                      /* [C, C] K projection */
    const float *value_w;                    /* [C, C] V projection */
    const float *output_w;                   /* [C, C] output projection */
    const float *ffn_x_k;                    /* [C] token shift lerp */
    const float *ffn_key_w;                  /* [C, ffn_dim] FFN key */
    const float *ffn_value_w;                /* [ffn_dim, C] FFN value */
    int ffn_dim;                             /* FFN intermediate dimension */
} rx_attn_weights;

int rx_model_get_attn_weights(const rx_model *model, int attn_idx,
                               rx_attn_weights *out);

/* Get head projection weight pointer. Returns [C, vocab] row-major.
 * Returns NULL if model is NULL. */
const float* rx_model_get_head_weight(const rx_model *model);

/* Get final layer norm weights. Returns 0 on success. */
int rx_model_get_ln_out(const rx_model *model,
                         const float **w_out, const float **b_out);

/* ================================================================
 * Tokenizer
 * ================================================================ */

/* Load RWKV World tokenizer from vocab file. Returns NULL on failure. */
rx_tokenizer* rx_tokenizer_new(const char *vocab_path);

/* Free tokenizer. */
void rx_tokenizer_free(rx_tokenizer *tok);

/* Encode text into token IDs. Returns number of tokens written.
 * tokens_out must have space for at least max_tokens entries. */
int rx_tokenizer_encode(const rx_tokenizer *tok, const char *text,
                        uint32_t *tokens_out, int max_tokens);

/* Decode token IDs into text. Returns malloc'd string (caller must free). */
char* rx_tokenizer_decode(const rx_tokenizer *tok, const uint32_t *tokens,
                          int n_tokens);

/* ================================================================
 * Sampling
 * ================================================================ */

/* Return default sampling parameters. */
rx_sampling_params rx_sampling_defaults(void);

/* Sample a token from logits using the given parameters.
 * token_counts: per-token occurrence counts for penalties (can be NULL).
 * vocab_size: size of the logits/token_counts arrays. */
uint32_t rx_sample(const tm_tensor *logits, rx_sampling_params *params,
                   const uint32_t *token_counts, int vocab_size);

#ifdef __cplusplus
}
#endif

#endif /* RWKVX_H */
