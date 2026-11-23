/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ============================================================= */
/* Config                                                        */
/* ============================================================= */

fa_config fa_config_default(int num_heads, int head_dim) {
    fa_config config = {
        .num_heads = num_heads,
        .head_dim = head_dim,
        .dropout = 0.0f,
        .causal = false,
        .scale = 1.0f / sqrtf((float)head_dim)
    };
    return config;
}

void fa_print_config(fa_config* config) {
    if (!config) {
        printf("(null config)\n");
        return;
    }
    printf("Attention config: heads=%d, head_dim=%d, scale=%.4f, causal=%s\n",
           config->num_heads, config->head_dim, config->scale,
           config->causal ? "true" : "false");
}

/* ============================================================= */
/* Causal Mask                                                   */
/* ============================================================= */

tm_tensor* fa_causal_mask(int seq_len) {
    int64_t shape[] = {1, 1, seq_len, seq_len};
    tm_tensor* mask = tm_zeros(shape, 4, TM_F32);
    if (!mask) return NULL;

    float* data = tm_data_f32(mask);

    /* Fill upper triangle with -inf */
    for (int i = 0; i < seq_len; i++) {
        for (int j = i + 1; j < seq_len; j++) {
            data[i * seq_len + j] = -INFINITY;
        }
    }

    return mask;
}

/* ============================================================= */
/* Core Attention                                                */
/* ============================================================= */

/* Simple attention for 2D Q, K, V (seq_len x dim) */
static tm_tensor* attention_2d(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                               float scale, bool causal) {
    if (!Q || !K || !V) return NULL;

    int seq_len = (int)Q->shape[0];
    int kv_len = (int)K->shape[0];

    /* Compute Q @ K^T */
    tm_tensor* Kt = tm_transpose(K, 0, 1);
    if (!Kt) return NULL;

    tm_tensor* scores = tm_matmul(Q, Kt);
    tm_release(Kt);
    if (!scores) return NULL;

    /* Scale */
    float* s_data = tm_data_f32(scores);
    for (int64_t i = 0; i < scores->numel; i++) {
        s_data[i] *= scale;
    }

    /* Apply causal mask */
    if (causal) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = i + 1; j < kv_len; j++) {
                s_data[i * kv_len + j] = -INFINITY;
            }
        }
    }

    /* Softmax over last dimension */
    tm_tensor* attn_weights = tm_softmax(scores, 1);
    tm_release(scores);
    if (!attn_weights) return NULL;

    /* Attention output = weights @ V */
    tm_tensor* output = tm_matmul(attn_weights, V);
    tm_release(attn_weights);

    return output;
}

tm_tensor* fa_attention(tm_tensor* Q, tm_tensor* K, tm_tensor* V, fa_config* config) {
    if (!Q || !K || !V || !config) return NULL;

    float scale = config->scale;
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf((float)config->head_dim);
    }

    /* Handle different input shapes */
    if (Q->ndim == 2) {
        /* Simple 2D attention */
        return attention_2d(Q, K, V, scale, config->causal);
    }

    if (Q->ndim == 3) {
        /* Batched attention: (batch, seq, dim) */
        int batch = (int)Q->shape[0];
        int seq_len = (int)Q->shape[1];
        int dim = (int)Q->shape[2];

        int64_t out_shape[] = {batch, seq_len, dim};
        tm_tensor* output = tm_zeros(out_shape, 3, TM_F32);
        if (!output) return NULL;

        /* Process each batch */
        for (int b = 0; b < batch; b++) {
            /* Extract slices - simplified: just compute directly */
            int64_t q_shape[] = {seq_len, dim};
            tm_tensor* q_slice = tm_zeros(q_shape, 2, TM_F32);
            tm_tensor* k_slice = tm_zeros(q_shape, 2, TM_F32);
            tm_tensor* v_slice = tm_zeros(q_shape, 2, TM_F32);

            float* q_data = tm_data_f32(Q);
            float* k_data = tm_data_f32(K);
            float* v_data = tm_data_f32(V);

            memcpy(tm_data_f32(q_slice), q_data + b * seq_len * dim, seq_len * dim * sizeof(float));
            memcpy(tm_data_f32(k_slice), k_data + b * seq_len * dim, seq_len * dim * sizeof(float));
            memcpy(tm_data_f32(v_slice), v_data + b * seq_len * dim, seq_len * dim * sizeof(float));

            tm_tensor* out_slice = attention_2d(q_slice, k_slice, v_slice, scale, config->causal);

            if (out_slice) {
                memcpy(tm_data_f32(output) + b * seq_len * dim,
                       tm_data_f32(out_slice), seq_len * dim * sizeof(float));
                tm_release(out_slice);
            }

            tm_release(q_slice);
            tm_release(k_slice);
            tm_release(v_slice);
        }

        return output;
    }

    return NULL;
}

tm_tensor* fa_attention_mask(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                             tm_tensor* mask, fa_config* config) {
    /* For now, just use causal attention if mask provided */
    fa_config cfg = *config;
    if (mask) {
        cfg.causal = true;
    }
    return fa_attention(Q, K, V, &cfg);
}

/* ============================================================= */
/* Multi-Head Attention                                          */
/* ============================================================= */

tm_tensor* fa_multi_head(tm_tensor* x, tm_tensor* Wq, tm_tensor* Wk, tm_tensor* Wv,
                         tm_tensor* Wo, fa_config* config) {
    if (!x || !Wq || !Wk || !Wv || !Wo || !config) return NULL;

    /* Project to Q, K, V */
    tm_tensor* Q = tm_matmul(x, Wq);
    tm_tensor* K = tm_matmul(x, Wk);
    tm_tensor* V = tm_matmul(x, Wv);

    if (!Q || !K || !V) {
        tm_release(Q);
        tm_release(K);
        tm_release(V);
        return NULL;
    }

    /* Compute attention */
    tm_tensor* attn = fa_attention(Q, K, V, config);
    tm_release(Q);
    tm_release(K);
    tm_release(V);

    if (!attn) return NULL;

    /* Output projection */
    tm_tensor* output = tm_matmul(attn, Wo);
    tm_release(attn);

    return output;
}

tm_tensor* fa_self_attention(tm_tensor* x, tm_tensor* Wqkv, tm_tensor* Wo,
                             fa_config* config) {
    if (!x || !Wqkv || !Wo || !config) return NULL;

    /* Project to combined QKV */
    tm_tensor* qkv = tm_matmul(x, Wqkv);
    if (!qkv) return NULL;

    /* Split into Q, K, V (assuming Wqkv projects to 3 * hidden_dim) */
    int hidden = config->num_heads * config->head_dim;

    int64_t seq_len = x->shape[x->ndim - 2];
    int64_t q_shape[] = {seq_len, hidden};

    tm_tensor* Q = tm_zeros(q_shape, 2, TM_F32);
    tm_tensor* K = tm_zeros(q_shape, 2, TM_F32);
    tm_tensor* V = tm_zeros(q_shape, 2, TM_F32);

    if (!Q || !K || !V) {
        tm_release(qkv);
        tm_release(Q);
        tm_release(K);
        tm_release(V);
        return NULL;
    }

    float* qkv_data = tm_data_f32(qkv);
    float* q_data = tm_data_f32(Q);
    float* k_data = tm_data_f32(K);
    float* v_data = tm_data_f32(V);

    for (int64_t i = 0; i < seq_len; i++) {
        memcpy(q_data + i * hidden, qkv_data + i * 3 * hidden, hidden * sizeof(float));
        memcpy(k_data + i * hidden, qkv_data + i * 3 * hidden + hidden, hidden * sizeof(float));
        memcpy(v_data + i * hidden, qkv_data + i * 3 * hidden + 2 * hidden, hidden * sizeof(float));
    }

    tm_release(qkv);

    /* Compute attention */
    tm_tensor* attn = fa_attention(Q, K, V, config);
    tm_release(Q);
    tm_release(K);
    tm_release(V);

    if (!attn) return NULL;

    /* Output projection */
    tm_tensor* output = tm_matmul(attn, Wo);
    tm_release(attn);

    return output;
}

/* ============================================================= */
/* KV Cache                                                      */
/* ============================================================= */

struct fa_context {
    tm_tensor** k_cache;  /* K cache per layer */
    tm_tensor** v_cache;  /* V cache per layer */
    int num_layers;
    int batch_size;
    int max_seq_len;
    int num_heads;
    int head_dim;
    int current_len;      /* Current sequence length in cache */
};

fa_context* fa_context_new(int batch_size, int max_seq_len, int num_heads,
                           int head_dim, int num_layers) {
    fa_context* ctx = calloc(1, sizeof(fa_context));
    if (!ctx) return NULL;

    ctx->num_layers = num_layers;
    ctx->batch_size = batch_size;
    ctx->max_seq_len = max_seq_len;
    ctx->num_heads = num_heads;
    ctx->head_dim = head_dim;
    ctx->current_len = 0;

    ctx->k_cache = calloc(num_layers, sizeof(tm_tensor*));
    ctx->v_cache = calloc(num_layers, sizeof(tm_tensor*));

    if (!ctx->k_cache || !ctx->v_cache) {
        fa_context_free(ctx);
        return NULL;
    }

    /* Allocate cache tensors */
    int64_t shape[] = {batch_size, max_seq_len, num_heads * head_dim};

    for (int i = 0; i < num_layers; i++) {
        ctx->k_cache[i] = tm_zeros(shape, 3, TM_F32);
        ctx->v_cache[i] = tm_zeros(shape, 3, TM_F32);

        if (!ctx->k_cache[i] || !ctx->v_cache[i]) {
            fa_context_free(ctx);
            return NULL;
        }
    }

    return ctx;
}

void fa_context_free(fa_context* ctx) {
    if (!ctx) return;

    if (ctx->k_cache) {
        for (int i = 0; i < ctx->num_layers; i++) {
            tm_release(ctx->k_cache[i]);
        }
        free(ctx->k_cache);
    }

    if (ctx->v_cache) {
        for (int i = 0; i < ctx->num_layers; i++) {
            tm_release(ctx->v_cache[i]);
        }
        free(ctx->v_cache);
    }

    free(ctx);
}

void fa_context_clear(fa_context* ctx) {
    if (!ctx) return;

    ctx->current_len = 0;

    /* Zero out cache tensors */
    for (int i = 0; i < ctx->num_layers; i++) {
        if (ctx->k_cache[i]) {
            memset(tm_data_f32(ctx->k_cache[i]), 0,
                   ctx->k_cache[i]->numel * sizeof(float));
        }
        if (ctx->v_cache[i]) {
            memset(tm_data_f32(ctx->v_cache[i]), 0,
                   ctx->v_cache[i]->numel * sizeof(float));
        }
    }
}

int fa_context_seq_len(fa_context* ctx) {
    return ctx ? ctx->current_len : 0;
}

tm_tensor* fa_attention_cached(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                               fa_context* ctx, int layer_idx, int pos,
                               fa_config* config) {
    if (!Q || !K || !V || !ctx || !config) return NULL;
    if (layer_idx < 0 || layer_idx >= ctx->num_layers) return NULL;
    if (pos >= ctx->max_seq_len) return NULL;

    int dim = ctx->num_heads * ctx->head_dim;

    /* Update cache with new K, V at position pos */
    float* k_cache_data = tm_data_f32(ctx->k_cache[layer_idx]);
    float* v_cache_data = tm_data_f32(ctx->v_cache[layer_idx]);
    float* k_data = tm_data_f32(K);
    float* v_data = tm_data_f32(V);

    /* Copy new K, V to cache at pos */
    int new_tokens = (int)K->shape[K->ndim - 2];
    for (int t = 0; t < new_tokens; t++) {
        int cache_pos = pos + t;
        if (cache_pos < ctx->max_seq_len) {
            memcpy(k_cache_data + cache_pos * dim, k_data + t * dim, dim * sizeof(float));
            memcpy(v_cache_data + cache_pos * dim, v_data + t * dim, dim * sizeof(float));
        }
    }

    ctx->current_len = pos + new_tokens;

    /* Create K, V views from cache up to current_len */
    int64_t kv_shape[] = {ctx->current_len, dim};
    tm_tensor* K_full = tm_zeros(kv_shape, 2, TM_F32);
    tm_tensor* V_full = tm_zeros(kv_shape, 2, TM_F32);

    if (!K_full || !V_full) {
        tm_release(K_full);
        tm_release(V_full);
        return NULL;
    }

    memcpy(tm_data_f32(K_full), k_cache_data, ctx->current_len * dim * sizeof(float));
    memcpy(tm_data_f32(V_full), v_cache_data, ctx->current_len * dim * sizeof(float));

    /* Compute attention */
    tm_tensor* output = fa_attention(Q, K_full, V_full, config);

    tm_release(K_full);
    tm_release(V_full);

    return output;
}

/* ============================================================= */
/* RoPE                                                          */
/* ============================================================= */

tm_tensor* fa_rope_freqs(int max_seq_len, int head_dim) {
    int64_t shape[] = {max_seq_len, head_dim / 2};
    tm_tensor* freqs = tm_zeros(shape, 2, TM_F32);
    if (!freqs) return NULL;

    float* data = tm_data_f32(freqs);
    float theta = 10000.0f;

    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            data[pos * (head_dim / 2) + i] = (float)pos * freq;
        }
    }

    return freqs;
}

void fa_rope(tm_tensor* Q, tm_tensor* K, int start_pos, int head_dim) {
    if (!Q || !K) return;

    int seq_len = (int)Q->shape[Q->ndim - 2];
    int dim = (int)Q->shape[Q->ndim - 1];

    float* q_data = tm_data_f32(Q);
    float* k_data = tm_data_f32(K);

    float theta = 10000.0f;

    for (int pos = 0; pos < seq_len; pos++) {
        int abs_pos = start_pos + pos;

        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float angle = (float)abs_pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            /* Apply rotation to Q */
            int idx1 = pos * dim + i * 2;
            int idx2 = pos * dim + i * 2 + 1;

            if (idx2 < (int)Q->numel) {
                float q0 = q_data[idx1];
                float q1 = q_data[idx2];
                q_data[idx1] = q0 * cos_val - q1 * sin_val;
                q_data[idx2] = q0 * sin_val + q1 * cos_val;
            }

            /* Apply rotation to K */
            if (idx2 < (int)K->numel) {
                float k0 = k_data[idx1];
                float k1 = k_data[idx2];
                k_data[idx1] = k0 * cos_val - k1 * sin_val;
                k_data[idx2] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
}
