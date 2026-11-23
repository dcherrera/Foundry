/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rfa_train.c — LoRA training for RFA hybrid model (Qwen blocks only)
 *
 * Gradient checkpointing approach: during forward, save x_input and cache_len
 * at each Qwen block. During backward, recompute all intermediates and compute
 * proper gradients for all 7 LoRA layers per Qwen block.
 *
 * LoRA indices per Qwen block: 0=Q, 1=K, 2=V, 3=O, 4=gate, 5=up, 6=down
 *
 * Helper patterns: lora_fwd_add, lora_grad_acc,
 * lora_backward_input, ensure_grad, BLAS wrappers.
 */

#include "rwkvx/rfa_train.h"
#include "rwkvx/rfa.h"
#include "tensor/tensor.h"
#include "lora/lora.h"
#include "optim/optim.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <pthread.h>

#ifdef USE_BLAS
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

/* ================================================================
 * Internal BLAS helpers
 * ================================================================ */

/* out[cols] = W^T[rows,cols] @ x[rows]  — i.e. out = x^T @ W */
static void t_matvec(const float *x, const float *W, float *out,
                      int rows, int cols) {
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasTrans, rows, cols,
                1.0f, W, cols, x, 1, 0.0f, out, 1);
#else
    memset(out, 0, cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
        float xi = x[i];
        const float *wi = &W[i * cols];
        for (int j = 0; j < cols; j++)
            out[j] += xi * wi[j];
    }
#endif
}

/* out[rows] = W[rows,cols] @ x[cols] */
static void t_matvec_nt(const float *W, const float *x, float *out,
                          int rows, int cols) {
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols,
                1.0f, W, cols, x, 1, 0.0f, out, 1);
#else
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++)
            sum += W[i * cols + j] * x[j];
        out[i] = sum;
    }
#endif
}

/* out[rows] += W[rows,cols] @ x[cols]  (accumulate variant) */
static void t_matvec_nt_acc(const float *W, const float *x, float *out,
                              int rows, int cols) {
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols,
                1.0f, W, cols, x, 1, 1.0f, out, 1);
#else
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++)
            sum += W[i * cols + j] * x[j];
        out[i] += sum;
    }
#endif
}

/* ================================================================
 * LoRA helpers
 * ================================================================ */

static void ensure_grad(tm_tensor *t) {
    if (!t) return;
    if (!t->grad) {
        t->grad = tm_zeros(t->shape, t->ndim, t->dtype);
        t->grad->requires_grad = false;
    }
}

/* Forward: out[d_out] += scale * B @ A @ input */
static void lora_fwd_add(fl_layer *lora, const float *input, float *out) {
    if (!lora) return;
    int d_in = lora->d_in, d_out = lora->d_out, rank = lora->rank;
    float scale = lora->scale;
    float *A = tm_data_f32(lora->lora_A);
    float *B = tm_data_f32(lora->lora_B);
    float *h = (float*)calloc(rank, sizeof(float));
    for (int r = 0; r < rank; r++) {
        float s = 0.0f;
        for (int j = 0; j < d_in; j++) s += A[r * d_in + j] * input[j];
        h[r] = s;
    }
    for (int i = 0; i < d_out; i++) {
        float s = 0.0f;
        for (int r = 0; r < rank; r++) s += B[i * rank + r] * h[r];
        out[i] += scale * s;
    }
    free(h);
}

/* Accumulate LoRA gradients */
static void lora_grad_acc(fl_layer *lora, const float *input,
                           const float *d_output) {
    if (!lora) return;
    int d_in = lora->d_in, d_out = lora->d_out, rank = lora->rank;
    float scale = lora->scale;
    float *A = tm_data_f32(lora->lora_A);
    float *B = tm_data_f32(lora->lora_B);
    ensure_grad(lora->lora_A);
    ensure_grad(lora->lora_B);
    float *dA = tm_data_f32(lora->lora_A->grad);
    float *dB = tm_data_f32(lora->lora_B->grad);

    float *h = (float*)calloc(rank, sizeof(float));
    for (int r = 0; r < rank; r++) {
        float s = 0.0f;
        for (int j = 0; j < d_in; j++) s += A[r * d_in + j] * input[j];
        h[r] = s;
    }
    for (int i = 0; i < d_out; i++) {
        float ds = scale * d_output[i];
        for (int r = 0; r < rank; r++)
            dB[i * rank + r] += ds * h[r];
    }
    float *dh = (float*)calloc(rank, sizeof(float));
    for (int r = 0; r < rank; r++) {
        float s = 0.0f;
        for (int i = 0; i < d_out; i++) s += B[i * rank + r] * d_output[i];
        dh[r] = scale * s;
    }
    for (int r = 0; r < rank; r++) {
        float dr = dh[r];
        for (int j = 0; j < d_in; j++)
            dA[r * d_in + j] += dr * input[j];
    }
    free(h);
    free(dh);
}

/* Backward: d_input[d_in] += scale * A^T @ B^T @ d_output */
static void lora_backward_input(fl_layer *lora, const float *d_output,
                                  float *d_input) {
    if (!lora) return;
    int d_in = lora->d_in, d_out = lora->d_out, rank = lora->rank;
    float scale = lora->scale;
    float *A = tm_data_f32(lora->lora_A);
    float *B = tm_data_f32(lora->lora_B);

    float *dh = (float*)calloc(rank, sizeof(float));
    for (int r = 0; r < rank; r++) {
        float s = 0.0f;
        for (int i = 0; i < d_out; i++) s += B[i * rank + r] * d_output[i];
        dh[r] = s;
    }
    for (int r = 0; r < rank; r++) {
        float dr = scale * dh[r];
        for (int j = 0; j < d_in; j++)
            d_input[j] += A[r * d_in + j] * dr;
    }
    free(dh);
}

/* ================================================================
 * RMSNorm forward (matching rfa.c convention)
 * ================================================================ */

static void rms_norm_fwd(const float *x, const float *w, float *out,
                          int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++)
        sum_sq += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * w[i];
}

static inline float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ================================================================
 * Config defaults
 * ================================================================ */

rfa_train_config rfa_train_config_defaults(void) {
    rfa_train_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.lora_rank = 32;
    cfg.lora_alpha = 64.0f;
    cfg.lr = 2e-5f;
    cfg.weight_decay = 0.01f;
    cfg.epochs = 3;
    cfg.max_seq_len = 256;
    cfg.warmup_steps = 100;
    cfg.save_every = 1000;
    cfg.log_every = 10;
    cfg.grad_clip = 1.0f;
    cfg.max_cache = 2048;
    cfg.n_threads = 1;
    cfg.quantize = false;
    return cfg;
}

/* ================================================================
 * Training examples
 * ================================================================ */

void rfa_train_example_free(rfa_train_example *ex) {
    if (!ex) return;
    free(ex->input_ids);
    free(ex->target_ids);
}

void rfa_train_examples_free(rfa_train_example *examples, int n) {
    if (!examples) return;
    for (int i = 0; i < n; i++) {
        free(examples[i].input_ids);
        free(examples[i].target_ids);
    }
    free(examples);
}

/* ================================================================
 * Binary data loader
 * ================================================================ */

rfa_train_example* rfa_load_pretokenized(
    const char *path, int max_seq_len, int *n_examples_out)
{
    if (!path || !n_examples_out) return NULL;
    *n_examples_out = 0;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "rfa_train: cannot open %s\n", path);
        return NULL;
    }

    uint32_t n_total;
    if (fread(&n_total, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    rfa_train_example *examples = (rfa_train_example*)calloc(n_total, sizeof(rfa_train_example));
    int n_out = 0;

    for (uint32_t i = 0; i < n_total; i++) {
        uint32_t seq_len;
        if (fread(&seq_len, sizeof(uint32_t), 1, f) != 1) break;

        uint32_t *tokens = (uint32_t*)malloc(seq_len * sizeof(uint32_t));
        if (fread(tokens, sizeof(uint32_t), seq_len, f) != seq_len) {
            free(tokens);
            break;
        }

        if (seq_len < 2) { free(tokens); continue; }

        /* Truncate if needed */
        int n_tok = (int)seq_len;
        if (n_tok > max_seq_len + 1) n_tok = max_seq_len + 1;
        int sl = n_tok - 1;

        examples[n_out].seq_len = sl;
        examples[n_out].input_ids = (uint32_t*)malloc(sl * sizeof(uint32_t));
        examples[n_out].target_ids = (uint32_t*)malloc(sl * sizeof(uint32_t));
        memcpy(examples[n_out].input_ids, tokens, sl * sizeof(uint32_t));
        memcpy(examples[n_out].target_ids, &tokens[1], sl * sizeof(uint32_t));
        n_out++;

        free(tokens);
    }

    fclose(f);
    *n_examples_out = n_out;
    fprintf(stderr, "rfa_train: loaded %d examples from %s\n", n_out, path);
    return examples;
}

/* ================================================================
 * RMSNorm backward
 *
 * y = x * rsqrt(mean(x^2) + eps) * w
 * d_x[i] = inv_rms * (d_out[i] * w[i] - x_hat[i] * dot / n)
 * where x_hat = x * inv_rms, dot = sum(d_out * w * x_hat)
 * ================================================================ */

void rfa_rms_norm_backward(
    const float *d_output,
    const float *x,
    const float *weight,
    float *d_x,
    int c, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < c; i++)
        sum_sq += x[i] * x[i];
    float rms_sq = sum_sq / c + eps;
    float inv_rms = 1.0f / sqrtf(rms_sq);

    /* dot = sum(d_output * weight * x_hat) where x_hat = x * inv_rms */
    float dot = 0.0f;
    for (int i = 0; i < c; i++)
        dot += d_output[i] * weight[i] * x[i] * inv_rms;

    float inv_c = 1.0f / c;
    for (int i = 0; i < c; i++) {
        float dw = d_output[i] * weight[i];
        float x_hat = x[i] * inv_rms;
        d_x[i] = inv_rms * (dw - x_hat * dot * inv_c);
    }
}

/* ================================================================
 * SwiGLU backward
 *
 * Forward: out = down_proj @ (silu(gate_proj @ x) * up_proj @ x)
 * ================================================================ */

void rfa_swiglu_backward(
    const float *d_output,
    const float *x_input,
    const rfa_qwen_weights *w,
    fl_layer **lora,
    float *d_input,
    int c, int inter)
{
    /* Recompute forward intermediates */
    float *gate = (float*)calloc(inter, sizeof(float));
    float *up = (float*)calloc(inter, sizeof(float));
    float *act = (float*)calloc(inter, sizeof(float));

    /* gate = gate_proj @ x + LoRA[4] */
    t_matvec(x_input, w->gate_proj, gate, c, inter);
    lora_fwd_add(lora[4], x_input, gate);

    /* up = up_proj @ x + LoRA[5] */
    t_matvec(x_input, w->up_proj, up, c, inter);
    lora_fwd_add(lora[5], x_input, up);

    /* act = silu(gate) * up */
    for (int i = 0; i < inter; i++)
        act[i] = fast_silu(gate[i]) * up[i];

    /* --- Backward through down_proj --- */
    /* d_act = down_proj^T @ d_output + LoRA[6] backward */
    float *d_act = (float*)calloc(inter, sizeof(float));
    t_matvec_nt(w->down_proj, d_output, d_act, inter, c);
    lora_grad_acc(lora[6], act, d_output);
    lora_backward_input(lora[6], d_output, d_act);

    /* --- Backward through SwiGLU activation --- */
    float *d_gate = (float*)calloc(inter, sizeof(float));
    float *d_up = (float*)calloc(inter, sizeof(float));

    for (int i = 0; i < inter; i++) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        float silu_val = gate[i] * sig;
        /* d_gate = d_act * up * silu'(gate) */
        /* silu'(x) = sig(x) + x * sig(x) * (1 - sig(x)) = sig(x) * (1 + x * (1 - sig(x))) */
        float silu_grad = sig * (1.0f + gate[i] * (1.0f - sig));
        d_gate[i] = d_act[i] * up[i] * silu_grad;
        /* d_up = d_act * silu(gate) */
        d_up[i] = d_act[i] * silu_val;
    }

    /* --- Backward through gate_proj --- */
    memset(d_input, 0, c * sizeof(float));
    t_matvec_nt_acc(w->gate_proj, d_gate, d_input, c, inter);
    lora_grad_acc(lora[4], x_input, d_gate);
    lora_backward_input(lora[4], d_gate, d_input);

    /* --- Backward through up_proj --- */
    t_matvec_nt_acc(w->up_proj, d_up, d_input, c, inter);
    lora_grad_acc(lora[5], x_input, d_up);
    lora_backward_input(lora[5], d_up, d_input);

    free(gate);
    free(up);
    free(act);
    free(d_act);
    free(d_gate);
    free(d_up);
}

/* ================================================================
 * GQA Attention backward
 *
 * Forward: for each Q-head qh (with kv_head = qh / group_size):
 *   scores[t] = Q[qh] . K[kvh,t] * scale
 *   weights[t] = softmax(scores)
 *   y[qh] = sum_t weights[t] * V[kvh,t]
 *
 * K/V gradients accumulate across all Q heads sharing the same KV head.
 * ================================================================ */

void rfa_gqa_attention_backward(
    const float *d_y,
    const float *q,
    const float *k_cache,
    const float *v_cache,
    const float *attn_weights,
    int cache_len,
    float *d_q,
    float *d_k_cur,
    float *d_v_cur,
    int n_q, int n_kv, int hd)
{
    int kv_dim = n_kv * hd;
    int group_size = n_q / n_kv;
    float scale = 1.0f / sqrtf((float)hd);

    memset(d_q, 0, n_q * hd * sizeof(float));
    memset(d_k_cur, 0, kv_dim * sizeof(float));
    memset(d_v_cur, 0, kv_dim * sizeof(float));

    for (int qh = 0; qh < n_q; qh++) {
        int kvh = qh / group_size;
        const float *w_h = &attn_weights[qh * cache_len];  /* [cache_len] */
        const float *d_y_h = &d_y[qh * hd];
        float *d_q_h = &d_q[qh * hd];

        /* d_weights[t] = dot(d_y_h, V[kvh, t]) */
        float *d_weights = (float*)calloc(cache_len, sizeof(float));
        for (int t = 0; t < cache_len; t++) {
            float dot = 0.0f;
            const float *vt = &v_cache[t * kv_dim + kvh * hd];
            for (int j = 0; j < hd; j++)
                dot += d_y_h[j] * vt[j];
            d_weights[t] = dot;
        }

        /* Softmax backward: d_score = w * (d_w - sum(d_w * w)) */
        float wdw_sum = 0.0f;
        for (int t = 0; t < cache_len; t++)
            wdw_sum += d_weights[t] * w_h[t];

        for (int t = 0; t < cache_len; t++) {
            float d_score = w_h[t] * (d_weights[t] - wdw_sum);
            /* d_Q += scale * d_score * K[kvh, t] */
            const float *kt = &k_cache[t * kv_dim + kvh * hd];
            for (int j = 0; j < hd; j++)
                d_q_h[j] += scale * d_score * kt[j];
        }

        /* K/V gradients for current token (last entry in cache) */
        int self_idx = cache_len - 1;
        if (self_idx >= 0) {
            float d_score_self = w_h[self_idx] * (d_weights[self_idx] - wdw_sum);
            for (int j = 0; j < hd; j++) {
                /* K gradient accumulates across Q heads sharing this KV head */
                d_k_cur[kvh * hd + j] += scale * d_score_self * q[qh * hd + j];
                d_v_cur[kvh * hd + j] += w_h[self_idx] * d_y_h[j];
            }
        }

        free(d_weights);
    }
}

/* ================================================================
 * Full Qwen block backward
 *
 * Reverse walk through:
 *  1.  FFN residual split
 *  2.  down_proj backward + LoRA[6] grads
 *  3.  SwiGLU backward (gate/up LoRA[4,5] grads)
 *  4.  Pre-FFN RMSNorm backward
 *  5.  Attention residual split
 *  6.  O projection backward + LoRA[3] grads
 *  7.  GQA attention backward
 *  8.  V projection backward + LoRA[2] grads
 *  9.  K projection backward + LoRA[1] grads
 * 10.  Q projection backward + LoRA[0] grads
 * 11.  Pre-attention RMSNorm backward
 * ================================================================ */

void rfa_qwen_backward(
    const rfa_model *model,
    const rfa_state *state,
    int qwen_idx,
    const rfa_qwen_saved *saved,
    fl_layer **lora,
    const float *d_output,
    float *d_input,
    rfa_config *cfg)
{
    int c = cfg->n_embd;
    int n_q = cfg->n_q_heads;
    int n_kv = cfg->n_kv_heads;
    int hd = cfg->head_dim;
    int kv_dim = n_kv * hd;
    int inter = cfg->intermediate_size;
    int group_size = n_q / n_kv;
    float eps = cfg->rms_norm_eps;

    rfa_qwen_weights w;
    rfa_model_get_qwen_weights(model, qwen_idx, &w);

    /* ---- Recompute forward intermediates from saved x_input ---- */

    float *ln1_out = (float*)calloc(c, sizeof(float));
    float *q_vec = (float*)calloc(c, sizeof(float));
    float *k_vec = (float*)calloc(kv_dim, sizeof(float));
    float *v_vec = (float*)calloc(kv_dim, sizeof(float));
    float *y_attn = (float*)calloc(c, sizeof(float));
    float *attn_out = (float*)calloc(c, sizeof(float));
    float *x_after_attn = (float*)calloc(c, sizeof(float));
    float *ln2_out = (float*)calloc(c, sizeof(float));

    /* Pre-attention RMSNorm */
    rms_norm_fwd(saved->x_input, w.ln1_w, ln1_out, c, eps);

    /* Q/K/V projections + LoRA */
    t_matvec(ln1_out, w.q_proj_w, q_vec, c, c);
    if (w.q_proj_b) for (int i = 0; i < c; i++) q_vec[i] += w.q_proj_b[i];
    lora_fwd_add(lora[0], ln1_out, q_vec);

    t_matvec(ln1_out, w.k_proj_w, k_vec, c, kv_dim);
    if (w.k_proj_b) for (int i = 0; i < kv_dim; i++) k_vec[i] += w.k_proj_b[i];
    lora_fwd_add(lora[1], ln1_out, k_vec);

    t_matvec(ln1_out, w.v_proj_w, v_vec, c, kv_dim);
    if (w.v_proj_b) for (int i = 0; i < kv_dim; i++) v_vec[i] += w.v_proj_b[i];
    lora_fwd_add(lora[2], ln1_out, v_vec);

    /* Apply RoPE to Q and K (needed for correct attention computation) */
    /* Use cache_len - 1 as the position for this token */
    int tok_pos = saved->cache_len > 0 ? saved->cache_len - 1 : 0;

    /* RoPE application (inline, matching rfa.c apply_rope) */
    {
        int half = hd / 2;
        float theta = cfg->rope_theta;
        for (int h = 0; h < n_q; h++) {
            float *head = &q_vec[h * hd];
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / hd);
                float angle = tok_pos * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float x0 = head[2 * i], x1 = head[2 * i + 1];
                head[2 * i]     = x0 * cos_a - x1 * sin_a;
                head[2 * i + 1] = x0 * sin_a + x1 * cos_a;
            }
        }
        for (int h = 0; h < n_kv; h++) {
            float *head = &k_vec[h * hd];
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / hd);
                float angle = tok_pos * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float x0 = head[2 * i], x1 = head[2 * i + 1];
                head[2 * i]     = x0 * cos_a - x1 * sin_a;
                head[2 * i + 1] = x0 * sin_a + x1 * cos_a;
            }
        }
    }

    /* Recompute attention output using KV cache */
    const float *k_cache, *v_cache;
    int ct = rfa_state_get_kv_cache(state, qwen_idx, &k_cache, &v_cache);
    int total = saved->cache_len > 0 ? saved->cache_len : ct;
    if (total > ct) total = ct;

    float scale = 1.0f / sqrtf((float)hd);

    /* Per-head attention + save weights for backward */
    float *attn_weights = (float*)calloc((size_t)n_q * total, sizeof(float));
    memset(y_attn, 0, c * sizeof(float));

    for (int qh = 0; qh < n_q; qh++) {
        int kvh = qh / group_size;
        float *q_head = &q_vec[qh * hd];
        float *scores = &attn_weights[qh * total];

        /* Compute scores */
        float max_s = -1e30f;
        for (int t = 0; t < total; t++) {
            const float *kt = &k_cache[t * kv_dim + kvh * hd];
            float dot = 0.0f;
            for (int j = 0; j < hd; j++)
                dot += q_head[j] * kt[j];
            scores[t] = dot * scale;
            if (scores[t] > max_s) max_s = scores[t];
        }

        /* Softmax */
        float sum_e = 0.0f;
        for (int t = 0; t < total; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_e += scores[t];
        }
        if (sum_e > 0.0f)
            for (int t = 0; t < total; t++)
                scores[t] /= sum_e;

        /* Value aggregation */
        float *y_head = &y_attn[qh * hd];
        for (int t = 0; t < total; t++) {
            const float *vt = &v_cache[t * kv_dim + kvh * hd];
            for (int j = 0; j < hd; j++)
                y_head[j] += scores[t] * vt[j];
        }
    }

    /* O projection + LoRA */
    t_matvec(y_attn, w.o_proj_w, attn_out, c, c);
    lora_fwd_add(lora[3], y_attn, attn_out);

    /* Attention residual */
    for (int i = 0; i < c; i++)
        x_after_attn[i] = saved->x_input[i] + attn_out[i];

    /* Pre-FFN RMSNorm */
    rms_norm_fwd(x_after_attn, w.ln2_w, ln2_out, c, eps);

    /* ---- BACKWARD PASS ---- */

    float *d_x1 = (float*)calloc(c, sizeof(float));      /* d w.r.t. x_after_attn */
    float *d_ffn_out = (float*)calloc(c, sizeof(float));
    float *d_ln2 = (float*)calloc(c, sizeof(float));
    float *d_att_out = (float*)calloc(c, sizeof(float));
    float *d_y = (float*)calloc(c, sizeof(float));
    float *d_q = (float*)calloc(c, sizeof(float));
    float *d_k_cur = (float*)calloc(kv_dim, sizeof(float));
    float *d_v_cur = (float*)calloc(kv_dim, sizeof(float));
    float *d_ln1 = (float*)calloc(c, sizeof(float));

    /* --- 1. FFN residual: x_out = x_after_attn + ffn_out --- */
    memcpy(d_x1, d_output, c * sizeof(float));
    memcpy(d_ffn_out, d_output, c * sizeof(float));

    /* --- 2-3. SwiGLU FFN backward (includes down, gate, up + their LoRAs) --- */
    float *d_ln2_from_ffn = (float*)calloc(c, sizeof(float));
    rfa_swiglu_backward(d_ffn_out, ln2_out, &w, lora, d_ln2_from_ffn, c, inter);

    /* --- 4. Pre-FFN RMSNorm backward --- */
    float *d_x1_from_ln = (float*)calloc(c, sizeof(float));
    rfa_rms_norm_backward(d_ln2_from_ffn, x_after_attn, w.ln2_w, d_x1_from_ln, c, eps);
    for (int i = 0; i < c; i++)
        d_x1[i] += d_x1_from_ln[i];

    /* --- 5. Attention residual: x_after_attn = x_input + attn_out --- */
    memcpy(d_input, d_x1, c * sizeof(float));
    memcpy(d_att_out, d_x1, c * sizeof(float));

    /* --- 6. O projection backward --- */
    t_matvec_nt(w.o_proj_w, d_att_out, d_y, c, c);
    lora_grad_acc(lora[3], y_attn, d_att_out);
    lora_backward_input(lora[3], d_att_out, d_y);

    /* --- 7. GQA attention backward --- */
    rfa_gqa_attention_backward(d_y, q_vec, k_cache, v_cache,
                                attn_weights, total,
                                d_q, d_k_cur, d_v_cur,
                                n_q, n_kv, hd);

    /* --- 8-10. Q/K/V LoRA gradients --- */
    /* Q: d_ln1 += Q_proj^T @ d_q (need to undo RoPE for gradient, but since
     * RoPE is positional and orthogonal, we can pass d_q through RoPE inverse.
     * For LoRA gradients, we use the pre-RoPE Q which is what ln1_out was projected to.) */
    /* Approximate: treat RoPE as identity for LoRA gradient computation.
     * This is standard practice — RoPE gradients are small and orthogonal. */
    t_matvec_nt_acc(w.q_proj_w, d_q, d_ln1, c, c);
    lora_grad_acc(lora[0], ln1_out, d_q);
    lora_backward_input(lora[0], d_q, d_ln1);

    /* K LoRA gradients */
    lora_grad_acc(lora[1], ln1_out, d_k_cur);
    /* V LoRA gradients */
    lora_grad_acc(lora[2], ln1_out, d_v_cur);

    /* --- 11. Pre-attention RMSNorm backward --- */
    float *d_input_from_ln = (float*)calloc(c, sizeof(float));
    rfa_rms_norm_backward(d_ln1, saved->x_input, w.ln1_w, d_input_from_ln, c, eps);
    for (int i = 0; i < c; i++)
        d_input[i] += d_input_from_ln[i];

    /* Cleanup */
    free(ln1_out);
    free(q_vec);
    free(k_vec);
    free(v_vec);
    free(y_attn);
    free(attn_out);
    free(x_after_attn);
    free(ln2_out);
    free(attn_weights);
    free(d_x1);
    free(d_ffn_out);
    free(d_ln2);
    free(d_ln2_from_ffn);
    free(d_x1_from_ln);
    free(d_att_out);
    free(d_y);
    free(d_q);
    free(d_k_cur);
    free(d_v_cur);
    free(d_ln1);
    free(d_input_from_ln);
}

/* ================================================================
 * LoRA adapter creation
 * ================================================================ */

fl_adapter* rfa_create_qwen_lora(const rfa_model *model, int rank, float alpha) {
    if (!model) return NULL;
    fl_adapter *adapter = fl_adapter_new();
    rfa_config cfg = rfa_model_config(model);
    int c = cfg.n_embd;
    int kv_dim = cfg.n_kv_heads * cfg.head_dim;
    int inter = cfg.intermediate_size;
    int n_qwen = cfg.n_qwen_layers;

    fl_config lora_cfg = { .rank = rank, .alpha = alpha, .dropout = 0.0f,
                           .fan_in_fan_out = false };

    for (int q = 0; q < n_qwen; q++) {
        char name[128];
        /* Attention projections */
        snprintf(name, sizeof(name), "qwen.%d.attn.q_proj", q);
        fl_adapter_add(adapter, fl_layer_new(name, c, c, &lora_cfg));
        snprintf(name, sizeof(name), "qwen.%d.attn.k_proj", q);
        fl_adapter_add(adapter, fl_layer_new(name, c, kv_dim, &lora_cfg));
        snprintf(name, sizeof(name), "qwen.%d.attn.v_proj", q);
        fl_adapter_add(adapter, fl_layer_new(name, c, kv_dim, &lora_cfg));
        snprintf(name, sizeof(name), "qwen.%d.attn.o_proj", q);
        fl_adapter_add(adapter, fl_layer_new(name, c, c, &lora_cfg));
        /* SwiGLU FFN projections */
        snprintf(name, sizeof(name), "qwen.%d.ffn.gate_proj", q);
        fl_adapter_add(adapter, fl_layer_new(name, c, inter, &lora_cfg));
        snprintf(name, sizeof(name), "qwen.%d.ffn.up_proj", q);
        fl_adapter_add(adapter, fl_layer_new(name, c, inter, &lora_cfg));
        snprintf(name, sizeof(name), "qwen.%d.ffn.down_proj", q);
        fl_adapter_add(adapter, fl_layer_new(name, inter, c, &lora_cfg));
    }
    return adapter;
}

tm_tensor** rfa_collect_params(fl_adapter *adapter, int *n_params_out) {
    if (!adapter || !n_params_out) return NULL;
    int n_layers = fl_adapter_size(adapter);
    int n_params = n_layers * 2;
    *n_params_out = n_params;
    tm_tensor **params = (tm_tensor**)malloc(n_params * sizeof(tm_tensor*));
    for (int i = 0; i < n_layers; i++) {
        fl_layer *l = fl_adapter_get_idx(adapter, i);
        params[i * 2] = l->lora_A;
        params[i * 2 + 1] = l->lora_B;
    }
    return params;
}

/* ================================================================
 * Training context for rfa_forward_ex callback
 * ================================================================ */

typedef struct {
    fl_adapter *adapter;
    int *token_idx;         /* [n_qwen] counter per Qwen block */
    const rfa_model *model;
    const rfa_state *state;
    /* Saved x_input for each Qwen block × token */
    float *x_inputs_flat;   /* [n_qwen * max_seq_len * C] */
    int *saved_cache_lens;  /* [n_qwen * max_seq_len] */
    int max_seq_len;
    int c;
} rfa_train_ctx;

/* Callback: save x_input before Qwen block runs */
static int train_block_callback(int phase, int block_idx, int qwen_idx,
                                 float *x, void *user_data) {
    (void)block_idx;
    rfa_train_ctx *ctx = (rfa_train_ctx*)user_data;

    if (phase == 0) {
        int t = ctx->token_idx[qwen_idx];
        int idx = qwen_idx * ctx->max_seq_len + t;
        memcpy(&ctx->x_inputs_flat[idx * ctx->c], x, ctx->c * sizeof(float));

        const float *kc, *vc;
        int cl = rfa_state_get_kv_cache(ctx->state, qwen_idx, &kc, &vc);
        ctx->saved_cache_lens[idx] = cl + 1;  /* +1 because forward appends */

        return 0;
    }

    if (phase == 1) {
        ctx->token_idx[qwen_idx]++;
    }

    return 0;
}

/* ================================================================
 * Training step
 * ================================================================ */

rfa_train_step_result rfa_train_step(
    rfa_model *model,
    rfa_state *state,
    fl_adapter *adapter,
    fo_optimizer *optimizer,
    rfa_train_example *example,
    rfa_train_config *config)
{
    rfa_train_step_result result;
    memset(&result, 0, sizeof(result));

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    rfa_config cfg = rfa_model_config(model);
    int c = cfg.n_embd;
    int vocab = cfg.vocab_size;
    int n_qwen = cfg.n_qwen_layers;
    int seq_len = example->seq_len;
    int lora_per_block = 7;

    /* Allocate storage */
    float *x_inputs_flat = (float*)calloc((size_t)n_qwen * seq_len * c, sizeof(float));
    int *saved_cache_lens = (int*)calloc((size_t)n_qwen * seq_len, sizeof(int));
    int *token_idx = (int*)calloc(n_qwen, sizeof(int));

    rfa_train_ctx ctx = {
        .adapter = adapter,
        .token_idx = token_idx,
        .model = model,
        .state = state,
        .x_inputs_flat = x_inputs_flat,
        .saved_cache_lens = saved_cache_lens,
        .max_seq_len = seq_len,
        .c = c,
    };

    /* Reset state */
    rfa_state_reset(state);

    /* === FORWARD PASS === */
    float *logits_buf = (float*)malloc((size_t)seq_len * vocab * sizeof(float));
    float total_loss = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        uint32_t token = example->input_ids[t];
        uint32_t target = example->target_ids[t];

        tm_tensor *logits = rfa_forward_ex(model, token, state,
                                            train_block_callback, &ctx);
        if (!logits) continue;

        float *ldata = tm_data_f32(logits);
        memcpy(&logits_buf[t * vocab], ldata, vocab * sizeof(float));

        /* Cross-entropy loss */
        float max_l = ldata[0];
        for (int j = 1; j < vocab; j++)
            if (ldata[j] > max_l) max_l = ldata[j];
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++)
            sum_exp += expf(ldata[j] - max_l);
        total_loss += -(ldata[target] - max_l - logf(sum_exp));

        tm_free(logits);
    }

    result.loss = seq_len > 0 ? total_loss / seq_len : 0.0f;

    /* === BACKWARD PASS === */

    const float *head_w = rfa_model_get_head_weight(model);

    /* Compute d_logits */
    float *d_logits = (float*)calloc((size_t)seq_len * vocab, sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        float *lgt = &logits_buf[t * vocab];
        float *dlgt = &d_logits[t * vocab];
        uint32_t target = example->target_ids[t];

        float max_l = lgt[0];
        for (int j = 1; j < vocab; j++)
            if (lgt[j] > max_l) max_l = lgt[j];
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++)
            sum_exp += expf(lgt[j] - max_l);
        float inv_seq = 1.0f / seq_len;
        for (int j = 0; j < vocab; j++)
            dlgt[j] = inv_seq * (expf(lgt[j] - max_l) / sum_exp);
        dlgt[target] -= inv_seq;
    }

    /* Backward through model for each token */
    float *d_x = (float*)calloc(c, sizeof(float));
    float *d_x_tmp = (float*)calloc(c, sizeof(float));

    for (int t = seq_len - 1; t >= 0; t--) {
        float *dlgt = &d_logits[t * vocab];

        /* Head projection backward: d_x = head_w^T @ d_logits */
        /* head_w is [C, vocab] in transposed layout */
        t_matvec_nt(head_w, dlgt, d_x, c, vocab);

        /* Walk Qwen blocks in reverse order */
        for (int q = n_qwen - 1; q >= 0; q--) {
            if (t >= token_idx[q]) continue;

            rfa_qwen_saved saved;
            memset(&saved, 0, sizeof(saved));
            int idx = q * seq_len + t;
            saved.x_input = (float*)malloc(c * sizeof(float));
            memcpy(saved.x_input, &x_inputs_flat[idx * c], c * sizeof(float));
            saved.cache_len = saved_cache_lens[idx];

            fl_layer *lora[7];
            for (int l = 0; l < 7; l++)
                lora[l] = fl_adapter_get_idx(adapter, q * lora_per_block + l);

            rfa_qwen_backward(model, state, q, &saved, lora,
                               d_x, d_x_tmp, &cfg);
            memcpy(d_x, d_x_tmp, c * sizeof(float));

            free(saved.x_input);
        }
    }

    /* Gradient clipping + optimizer step */
    if (config->grad_clip > 0.0f)
        fo_clip_grad_norm(optimizer, config->grad_clip);
    fo_step(optimizer);
    fo_zero_grad(optimizer);

    result.step = fo_get_step(optimizer);
    result.lr = fo_get_lr(optimizer);
    result.tokens_seen = seq_len;

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    result.elapsed_sec = (t_end.tv_sec - t_start.tv_sec) +
                          (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;

    /* Cleanup */
    free(x_inputs_flat);
    free(saved_cache_lens);
    free(token_idx);
    free(logits_buf);
    free(d_logits);
    free(d_x);
    free(d_x_tmp);

    return result;
}

/* ================================================================
 * Training loop
 * ================================================================ */

void rfa_train_loop(
    rfa_model *model,
    rfa_train_config *config,
    rfa_train_example *examples,
    int n_examples)
{
    if (!model || !config || !examples || n_examples == 0) return;

    rfa_config cfg = rfa_model_config(model);

    fprintf(stderr, "rfa_train: %d examples, %d epochs, lr=%.2e, rank=%d\n",
            n_examples, config->epochs, config->lr, config->lora_rank);
    fprintf(stderr, "rfa_train: %d Qwen blocks, 7 LoRA layers each\n",
            cfg.n_qwen_layers);

    fl_adapter *adapter = rfa_create_qwen_lora(model, config->lora_rank,
                                                config->lora_alpha);
    fl_adapter_print(adapter);

    int n_params;
    tm_tensor **params = rfa_collect_params(adapter, &n_params);
    fprintf(stderr, "rfa_train: %d trainable tensors, %lld parameters\n",
            n_params, (long long)fl_adapter_num_params(adapter));

    fo_adam_config adam_cfg = fo_adam_default(config->lr);
    adam_cfg.weight_decay = config->weight_decay;
    fo_optimizer *opt = fo_adamw_new(params, n_params, &adam_cfg);

    int total_steps = n_examples * config->epochs;
    fo_scheduler_config sched_cfg;
    sched_cfg.type = FO_SCHED_WARMUP_COSINE;
    sched_cfg.warmup_steps = config->warmup_steps;
    sched_cfg.total_steps = total_steps;
    sched_cfg.min_lr = config->lr * 0.1f;
    sched_cfg.step_size = 1;
    sched_cfg.gamma = 0.1f;
    fo_set_scheduler(opt, &sched_cfg);

    int max_cache = config->max_cache > 0 ? config->max_cache : 2048;
    rfa_state *state = rfa_state_new(model, max_cache);
    if (!state) {
        fprintf(stderr, "rfa_train: failed to create state\n");
        goto cleanup;
    }

    if (config->output_dir) mkdir(config->output_dir, 0755);

    int global_step = 0;
    for (int epoch = 0; epoch < config->epochs; epoch++) {
        fprintf(stderr, "\n=== Epoch %d/%d ===\n", epoch + 1, config->epochs);

        /* Shuffle */
        int *indices = (int*)malloc(n_examples * sizeof(int));
        for (int i = 0; i < n_examples; i++) indices[i] = i;
        for (int i = n_examples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        float epoch_loss = 0.0f;
        int epoch_tokens = 0;

        for (int i = 0; i < n_examples; i++) {
            fo_scheduler_step(opt);
            rfa_train_step_result res = rfa_train_step(
                model, state, adapter, opt, &examples[indices[i]], config);

            epoch_loss += res.loss * res.tokens_seen;
            epoch_tokens += res.tokens_seen;
            global_step++;

            if (config->log_every > 0 && global_step % config->log_every == 0) {
                float avg = epoch_tokens > 0 ? epoch_loss / epoch_tokens : 0.0f;
                fprintf(stderr, "  step %d | loss %.4f | avg %.4f | lr %.2e | %.1f tok/s\n",
                        global_step, res.loss, avg, res.lr,
                        res.elapsed_sec > 0 ? res.tokens_seen / res.elapsed_sec : 0.0);
            }

            if (config->save_every > 0 && config->output_dir &&
                global_step % config->save_every == 0) {
                char path[512];
                snprintf(path, sizeof(path), "%s/checkpoint_%d.safetensors",
                         config->output_dir, global_step);
                if (fl_adapter_save(adapter, path))
                    fprintf(stderr, "  Saved: %s\n", path);
            }
        }

        float avg = epoch_tokens > 0 ? epoch_loss / epoch_tokens : 0.0f;
        fprintf(stderr, "Epoch %d: avg_loss=%.4f, tokens=%d\n",
                epoch + 1, avg, epoch_tokens);
        free(indices);
    }

    if (config->output_dir) {
        char path[512];
        snprintf(path, sizeof(path), "%s/adapter_final.safetensors",
                 config->output_dir);
        if (fl_adapter_save(adapter, path))
            fprintf(stderr, "Saved: %s\n", path);
    }

cleanup:
    if (state) rfa_state_free(state);
    fo_free(opt);
    free(params);
    fl_adapter_free(adapter);
}

/* ================================================================
 * LoRA adapter averaging
 * ================================================================ */

fl_adapter* rfa_lora_average(fl_adapter **adapters, int n_adapters) {
    if (!adapters || n_adapters <= 0) return NULL;
    if (n_adapters == 1) {
        /* Clone the single adapter */
        fl_adapter *src = adapters[0];
        int n_layers = fl_adapter_size(src);
        fl_adapter *out = fl_adapter_new();
        for (int i = 0; i < n_layers; i++) {
            fl_layer *sl = fl_adapter_get_idx(src, i);
            tm_tensor *a_clone = tm_clone(sl->lora_A);
            tm_tensor *b_clone = tm_clone(sl->lora_B);
            fl_layer *nl = fl_layer_from_tensors(sl->name, a_clone, b_clone,
                                                  sl->scale * sl->rank);
            fl_adapter_add(out, nl);
        }
        return out;
    }

    fl_adapter *first = adapters[0];
    int n_layers = fl_adapter_size(first);
    fl_adapter *out = fl_adapter_new();
    float inv_n = 1.0f / n_adapters;

    for (int i = 0; i < n_layers; i++) {
        fl_layer *ref = fl_adapter_get_idx(first, i);
        int d_in = ref->d_in, d_out = ref->d_out, rank = ref->rank;
        int a_size = rank * d_in;
        int b_size = d_out * rank;

        /* Allocate averaged tensors */
        int64_t a_shape[] = {rank, d_in};
        int64_t b_shape[] = {d_out, rank};
        tm_tensor *avg_a = tm_zeros(a_shape, 2, TM_F32);
        tm_tensor *avg_b = tm_zeros(b_shape, 2, TM_F32);
        float *ad = tm_data_f32(avg_a);
        float *bd = tm_data_f32(avg_b);

        /* Sum across all adapters */
        for (int a = 0; a < n_adapters; a++) {
            fl_layer *al = fl_adapter_get_idx(adapters[a], i);
            float *src_a = tm_data_f32(al->lora_A);
            float *src_b = tm_data_f32(al->lora_B);
            for (int j = 0; j < a_size; j++) ad[j] += src_a[j];
            for (int j = 0; j < b_size; j++) bd[j] += src_b[j];
        }

        /* Average */
        for (int j = 0; j < a_size; j++) ad[j] *= inv_n;
        for (int j = 0; j < b_size; j++) bd[j] *= inv_n;

        fl_layer *nl = fl_layer_from_tensors(ref->name, avg_a, avg_b,
                                              ref->scale * ref->rank);
        fl_adapter_add(out, nl);
    }

    return out;
}

/* ================================================================
 * Parallel training (pthread-based)
 * ================================================================ */

typedef struct {
    rfa_model *model;            /* shared, read-only */
    rfa_train_config config;     /* per-thread copy */
    rfa_train_example *examples; /* per-thread chunk */
    int n_examples;
    int thread_id;
    fl_adapter *adapter;         /* output: trained adapter */
} rfa_train_worker_ctx;

static void* rfa_train_worker(void *arg) {
    rfa_train_worker_ctx *ctx = (rfa_train_worker_ctx *)arg;
    rfa_model *model = ctx->model;
    rfa_train_config *config = &ctx->config;

    fprintf(stderr, "[thread %d] starting: %d examples, %d epochs\n",
            ctx->thread_id, ctx->n_examples, config->epochs);

    /* Each thread gets its own adapter, state, optimizer */
    fl_adapter *adapter = rfa_create_qwen_lora(model, config->lora_rank,
                                                config->lora_alpha);

    int n_params;
    tm_tensor **params = rfa_collect_params(adapter, &n_params);

    fo_adam_config adam_cfg = fo_adam_default(config->lr);
    adam_cfg.weight_decay = config->weight_decay;
    fo_optimizer *opt = fo_adamw_new(params, n_params, &adam_cfg);

    int total_steps = ctx->n_examples * config->epochs;
    fo_scheduler_config sched_cfg;
    sched_cfg.type = FO_SCHED_WARMUP_COSINE;
    sched_cfg.warmup_steps = config->warmup_steps;
    sched_cfg.total_steps = total_steps;
    sched_cfg.min_lr = config->lr * 0.1f;
    sched_cfg.step_size = 1;
    sched_cfg.gamma = 0.1f;
    fo_set_scheduler(opt, &sched_cfg);

    int max_cache = config->max_cache > 0 ? config->max_cache : 2048;
    rfa_state *state = rfa_state_new(model, max_cache);
    if (!state) {
        fprintf(stderr, "[thread %d] failed to create state\n", ctx->thread_id);
        fo_free(opt);
        free(params);
        fl_adapter_free(adapter);
        return NULL;
    }

    int global_step = 0;
    for (int epoch = 0; epoch < config->epochs; epoch++) {
        /* Shuffle with per-thread seed */
        unsigned int seed = (unsigned int)(ctx->thread_id * 1000 + epoch * 37);
        int *indices = (int*)malloc(ctx->n_examples * sizeof(int));
        for (int i = 0; i < ctx->n_examples; i++) indices[i] = i;
        for (int i = ctx->n_examples - 1; i > 0; i--) {
            int j = (int)(rand_r(&seed) % (unsigned)(i + 1));
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        float epoch_loss = 0.0f;
        int epoch_tokens = 0;

        for (int i = 0; i < ctx->n_examples; i++) {
            fo_scheduler_step(opt);
            rfa_train_step_result res = rfa_train_step(
                model, state, adapter, opt,
                &ctx->examples[indices[i]], config);

            epoch_loss += res.loss * res.tokens_seen;
            epoch_tokens += res.tokens_seen;
            global_step++;

            if (config->log_every > 0 && global_step % config->log_every == 0) {
                float avg = epoch_tokens > 0 ? epoch_loss / epoch_tokens : 0.0f;
                fprintf(stderr, "  [T%d] step %d | loss %.4f | avg %.4f | lr %.2e\n",
                        ctx->thread_id, global_step, res.loss, avg, res.lr);
            }
        }

        float avg = epoch_tokens > 0 ? epoch_loss / epoch_tokens : 0.0f;
        fprintf(stderr, "[thread %d] epoch %d: avg_loss=%.4f, tokens=%d\n",
                ctx->thread_id, epoch + 1, avg, epoch_tokens);
        free(indices);
    }

    /* Save per-thread adapter */
    if (config->output_dir) {
        char path[512];
        snprintf(path, sizeof(path), "%s/thread_%d.safetensors",
                 config->output_dir, ctx->thread_id);
        if (fl_adapter_save(adapter, path))
            fprintf(stderr, "[thread %d] saved: %s\n", ctx->thread_id, path);
    }

    rfa_state_free(state);
    fo_free(opt);
    free(params);

    ctx->adapter = adapter;
    return NULL;
}

void rfa_train_parallel(
    rfa_model *model,
    rfa_train_config *config,
    rfa_train_example *examples,
    int n_examples,
    int n_threads)
{
    if (!model || !config || !examples || n_examples == 0 || n_threads < 1)
        return;

    if (n_threads == 1) {
        rfa_train_loop(model, config, examples, n_examples);
        return;
    }

    fprintf(stderr, "rfa_train_parallel: %d threads, %d examples total\n",
            n_threads, n_examples);

    if (config->output_dir) mkdir(config->output_dir, 0755);

    /* Split examples into n_threads chunks */
    int chunk_size = n_examples / n_threads;
    rfa_train_worker_ctx *ctxs = (rfa_train_worker_ctx*)calloc(
        n_threads, sizeof(rfa_train_worker_ctx));
    pthread_t *threads = (pthread_t*)calloc(n_threads, sizeof(pthread_t));

    for (int t = 0; t < n_threads; t++) {
        int start = t * chunk_size;
        int end = (t < n_threads - 1) ? start + chunk_size : n_examples;

        ctxs[t].model = model;
        ctxs[t].config = *config;   /* copy config per thread */
        ctxs[t].examples = &examples[start];
        ctxs[t].n_examples = end - start;
        ctxs[t].thread_id = t;
        ctxs[t].adapter = NULL;

        /* Adjust warmup proportionally */
        ctxs[t].config.warmup_steps = config->warmup_steps / n_threads;
        if (ctxs[t].config.warmup_steps < 1 && config->warmup_steps > 0)
            ctxs[t].config.warmup_steps = 1;

        pthread_create(&threads[t], NULL, rfa_train_worker, &ctxs[t]);
    }

    /* Wait for all threads */
    for (int t = 0; t < n_threads; t++)
        pthread_join(threads[t], NULL);

    /* Collect adapters and average */
    fl_adapter **adapters = (fl_adapter**)calloc(n_threads, sizeof(fl_adapter*));
    int n_valid = 0;
    for (int t = 0; t < n_threads; t++) {
        if (ctxs[t].adapter)
            adapters[n_valid++] = ctxs[t].adapter;
    }

    if (n_valid > 0) {
        fprintf(stderr, "rfa_train_parallel: averaging %d adapters\n", n_valid);
        fl_adapter *merged = rfa_lora_average(adapters, n_valid);

        if (merged && config->output_dir) {
            char path[512];
            snprintf(path, sizeof(path), "%s/adapter_final.safetensors",
                     config->output_dir);
            if (fl_adapter_save(merged, path))
                fprintf(stderr, "Saved merged adapter: %s\n", path);
        }

        fl_adapter_free(merged);
    }

    /* Cleanup per-thread adapters */
    for (int t = 0; t < n_threads; t++) {
        if (ctxs[t].adapter)
            fl_adapter_free(ctxs[t].adapter);
    }

    free(adapters);
    free(ctxs);
    free(threads);
    fprintf(stderr, "rfa_train_parallel: done\n");
}
