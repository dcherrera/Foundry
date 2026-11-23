/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_ATTENTION_H
#define FOUNDRY_ATTENTION_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Efficient attention implementations.
 * Replaces FlashAttention.
 */

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* Attention configuration */
typedef struct {
    int num_heads;
    int head_dim;
    float dropout;
    bool causal;           /* Causal masking for autoregressive */
    float scale;           /* Attention scale (default: 1/sqrt(head_dim)) */
} fa_config;

/* Attention context for caching */
typedef struct fa_context fa_context;

/* ============================================================= */
/* Standard Attention                                            */
/* ============================================================= */

/* Scaled dot-product attention: softmax(QK^T / sqrt(d)) * V
 * Q, K, V: (batch, seq_len, num_heads, head_dim)
 * Returns: (batch, seq_len, num_heads, head_dim)
 */
tm_tensor* fa_attention(tm_tensor* Q, tm_tensor* K, tm_tensor* V, fa_config* config);

/* Attention with explicit mask
 * mask: (batch, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
 * -inf values are masked out
 */
tm_tensor* fa_attention_mask(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                             tm_tensor* mask, fa_config* config);

/* ============================================================= */
/* Multi-Head Attention                                          */
/* ============================================================= */

/* Multi-head attention with projections
 * x: (batch, seq_len, embed_dim)
 * Wq, Wk, Wv: (embed_dim, num_heads * head_dim)
 * Wo: (num_heads * head_dim, embed_dim)
 */
tm_tensor* fa_multi_head(tm_tensor* x, tm_tensor* Wq, tm_tensor* Wk, tm_tensor* Wv,
                         tm_tensor* Wo, fa_config* config);

/* Multi-head self-attention (Q=K=V from same source) */
tm_tensor* fa_self_attention(tm_tensor* x, tm_tensor* Wqkv, tm_tensor* Wo,
                             fa_config* config);

/* ============================================================= */
/* KV Cache (for inference)                                      */
/* ============================================================= */

/* Create KV cache context */
fa_context* fa_context_new(int batch_size, int max_seq_len, int num_heads,
                           int head_dim, int num_layers);

/* Free context */
void fa_context_free(fa_context* ctx);

/* Attention with KV cache (for autoregressive generation)
 * Updates cache in-place, returns attention output
 */
tm_tensor* fa_attention_cached(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                               fa_context* ctx, int layer_idx, int pos,
                               fa_config* config);

/* Clear cache */
void fa_context_clear(fa_context* ctx);

/* Get current sequence length in cache */
int fa_context_seq_len(fa_context* ctx);

/* ============================================================= */
/* Rotary Position Embedding (RoPE)                              */
/* ============================================================= */

/* Apply rotary position embedding to Q and K
 * Modifies tensors in-place
 */
void fa_rope(tm_tensor* Q, tm_tensor* K, int start_pos, int head_dim);

/* Precompute RoPE frequencies */
tm_tensor* fa_rope_freqs(int max_seq_len, int head_dim);

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

/* Default config */
fa_config fa_config_default(int num_heads, int head_dim);

/* Create causal mask for sequence length */
tm_tensor* fa_causal_mask(int seq_len);

/* Print attention info */
void fa_print_config(fa_config* config);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_ATTENTION_H */
