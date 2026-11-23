/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rfa.c — RWKV-7 + Full Attention hybrid inference engine
 *
 * Hybrid model: RWKV-7 recurrent blocks + Qwen2 dense attention (GQA + RoPE).
 * All layers share RMSNorm + SwiGLU FFN; only the attention mechanism differs.
 *
 * Built on Foundry tensor ops. Designed for single-token decode (no batching).
 */

#include "rfa.h"
#include "safetensors/safetensors.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __ARM_NEON
  #include <arm_neon.h>
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* ================================================================
 * INT8 quantized weight matrix (per-row symmetric)
 * ================================================================ */

typedef struct {
    int8_t *data;       /* [rows * cols] quantized data */
    float *scales;      /* [rows] per-row scale: float = int8 * scale */
    int rows, cols;
} rfa_q8_weight;

/* Aligned malloc helper — 64-byte alignment for NEON cache line ops */
static void* rfa_aligned_malloc(size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
    return ptr;
}

static rfa_q8_weight* rfa_q8_from_f32(const float *src, int rows, int cols) {
    rfa_q8_weight *q = (rfa_q8_weight*)calloc(1, sizeof(rfa_q8_weight));
    q->rows = rows;
    q->cols = cols;
    q->data = (int8_t*)rfa_aligned_malloc(rows * cols * sizeof(int8_t));
    q->scales = (float*)rfa_aligned_malloc(rows * sizeof(float));

    for (int i = 0; i < rows; i++) {
        const float *row = &src[i * cols];
        float amax = 0.0f;
        for (int j = 0; j < cols; j++) {
            float a = row[j] < 0 ? -row[j] : row[j];
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        q->scales[i] = scale;
        float inv_scale = (scale > 0.0f) ? 127.0f / amax : 0.0f;
        int8_t *dst = &q->data[i * cols];
        for (int j = 0; j < cols; j++) {
            float v = row[j] * inv_scale;
            int vi = (int)(v + (v >= 0 ? 0.5f : -0.5f));
            if (vi > 127) vi = 127;
            if (vi < -127) vi = -127;
            dst[j] = (int8_t)vi;
        }
    }
    return q;
}

static void rfa_q8_free(rfa_q8_weight *q) {
    if (!q) return;
    free(q->data);
    free(q->scales);
    free(q);
}

/* Quantize a float vector to int8. Returns scale factor. */
static float rfa_quantize_input(const float *x, int8_t *x_q, int n) {
    float x_amax = 0.0f;
    for (int j = 0; j < n; j++) {
        float a = x[j] < 0 ? -x[j] : x[j];
        if (a > x_amax) x_amax = a;
    }
    float x_scale = x_amax / 127.0f;
    float x_inv_scale = (x_amax > 0.0f) ? 127.0f / x_amax : 0.0f;
    for (int j = 0; j < n; j++) {
        float v = x[j] * x_inv_scale;
        int vi = (int)(v + (v >= 0 ? 0.5f : -0.5f));
        if (vi > 127) vi = 127;
        if (vi < -127) vi = -127;
        x_q[j] = (int8_t)vi;
    }
    return x_scale;
}

/* INT8 matvec with pre-quantized input. NEON 4x unrolled. */
static void rfa_matvec_q8_preq(const int8_t *x_q, float x_scale,
                                const rfa_q8_weight *q, float *out,
                                int rows, int cols) {
#ifdef __ARM_NEON
    int cols64 = cols & ~63;
    int cols16 = cols & ~15;
#endif
    for (int i = 0; i < rows; i++) {
        const int8_t *row = &q->data[i * cols];
        float w_scale = q->scales[i];
        if (i + 1 < rows)
            __builtin_prefetch(&q->data[(i + 1) * cols], 0, 1);
#ifdef __ARM_NEON
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        for (int j = 0; j < cols64; j += 64) {
            acc0 = vdotq_s32(acc0, vld1q_s8(&x_q[j]),      vld1q_s8(&row[j]));
            acc1 = vdotq_s32(acc1, vld1q_s8(&x_q[j + 16]), vld1q_s8(&row[j + 16]));
            acc2 = vdotq_s32(acc2, vld1q_s8(&x_q[j + 32]), vld1q_s8(&row[j + 32]));
            acc3 = vdotq_s32(acc3, vld1q_s8(&x_q[j + 48]), vld1q_s8(&row[j + 48]));
        }
        int32x4_t acc = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        for (int j = cols64; j < cols16; j += 16)
            acc = vdotq_s32(acc, vld1q_s8(&x_q[j]), vld1q_s8(&row[j]));
        int32_t dot = vaddvq_s32(acc);
        for (int j = cols16; j < cols; j++)
            dot += (int32_t)x_q[j] * (int32_t)row[j];
#else
        int32_t dot = 0;
        for (int j = 0; j < cols; j++)
            dot += (int32_t)x_q[j] * (int32_t)row[j];
#endif
        out[i] = (float)dot * w_scale * x_scale;
    }
}

/* INT8 matvec: quantizes input on the fly, then dispatches. */
static void rfa_matvec_q8(const float *x, const rfa_q8_weight *q, float *out,
                           int rows, int cols, int8_t *q8_buf) {
    float x_scale = rfa_quantize_input(x, q8_buf, cols);
    rfa_matvec_q8_preq(q8_buf, x_scale, q, out, rows, cols);
}

/* ================================================================
 * Per-block INT8 quantized weights (populated when model->quantized)
 * ================================================================ */

typedef struct {
    /* Qwen attention (only for Qwen blocks) */
    rfa_q8_weight *q_proj_q8, *k_proj_q8, *v_proj_q8, *o_proj_q8;
    /* RWKV attention (only for RWKV blocks) */
    rfa_q8_weight *receptance_q8, *key_q8, *value_q8, *output_q8;
    /* SwiGLU FFN (all blocks) */
    rfa_q8_weight *gate_q8, *up_q8, *down_q8;
} rfa_block_q8;

/* ================================================================
 * Scratch workspace (pre-allocated, zero malloc in hot path)
 * ================================================================ */

typedef struct {
    float *buf[16];       /* general-purpose scratch buffers [C] */
    float *ffn_buf;       /* FFN intermediate buffer [intermediate_size] */
    float *lora_buf;      /* LoRA intermediate buffer [max_lora_dim] */
    float *x_buf[2];      /* alternating input/output buffers [C] */
    float *attn_buf;      /* attention scores buffer [max_cache] */
    float *rope_cos;      /* precomputed RoPE cos [head_dim/2] */
    float *rope_sin;      /* precomputed RoPE sin [head_dim/2] */
    int8_t *q8_buf;       /* INT8 quantization scratch */
} rfa_workspace;

/* ================================================================
 * Model struct
 * ================================================================ */

struct rfa_model {
    rfa_config config;

    /* Embedding + head */
    tm_tensor *emb_weight;     /* [vocab, C] */
    tm_tensor *ln_out_w;       /* [C] final RMSNorm */
    tm_tensor *head_weight;    /* [C, vocab] transposed for fast matvec */

    /* All blocks in execution order */
    rfa_block *blocks;         /* array of n_layers */

    /* INT8 quantized weights (populated when quantized=true) */
    rfa_block_q8 *blocks_q8;  /* array of n_layers, NULL if not quantized */
    rfa_q8_weight *head_q8;   /* lm_head quantized */

    /* Quantization flag */
    bool quantized;
};

/* ================================================================
 * State struct
 * ================================================================ */

struct rfa_state {
    rfa_rwkv_state *rwkv_states;  /* array indexed by RWKV block sequence */
    rfa_kv_cache *kv_caches;      /* array indexed by Qwen block sequence */
    float *v_first;               /* [C] first-token value for RWKV-7 */
    int n_rwkv;                   /* count of RWKV blocks */
    int n_qwen;                   /* count of Qwen blocks */
    int position;                 /* absolute position counter for RoPE */
    rfa_config config;
    rfa_workspace *workspace;
};

/* ================================================================
 * Fast math helpers
 * ================================================================ */

static inline float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight */
static void fast_rms_norm(const float *x, const float *w, float *out,
                          int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++)
        sum_sq += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * w[i];
}

/* Matrix-vector multiply: out[rows] = mat[rows, cols] @ x[cols] */
static void fast_matvec(const float *x, const float *mat, float *out,
                        int cols, int rows) {
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols,
                1.0f, mat, cols, x, 1, 0.0f, out, 1);
#else
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const float *row = &mat[r * cols];
        for (int c = 0; c < cols; c++)
            sum += row[c] * x[c];
        out[r] = sum;
    }
#endif
}

/* Add bias: out[i] += bias[i] */
static void fast_add_bias(float *out, const float *bias, int n) {
    for (int i = 0; i < n; i++)
        out[i] += bias[i];
}

/* Residual: out[i] += residual[i] */
static void fast_residual_add(float *out, const float *residual, int n) {
    for (int i = 0; i < n; i++)
        out[i] += residual[i];
}

/* Token shift: out[i] = x[i] + (prev[i] - x[i]) * lerp[i] */
static void fast_token_shift(const float *x, const float *prev,
                             const float *lerp, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = x[i] + (prev[i] - x[i]) * lerp[i];
}

/* L2 normalize per head */
static void fast_l2_normalize_per_head(const float *in, float *out,
                                       int n_head, int head_size) {
    for (int h = 0; h < n_head; h++) {
        int off = h * head_size;
        float sum_sq = 0.0f;
        for (int j = 0; j < head_size; j++)
            sum_sq += in[off + j] * in[off + j];
        float scale = 1.0f / sqrtf(sum_sq + 1e-12f);
        for (int j = 0; j < head_size; j++)
            out[off + j] = in[off + j] * scale;
    }
}

/* ================================================================
 * RoPE (Rotary Position Embedding)
 * ================================================================ */

/* Apply RoPE to a single vector in-place.
 * vec: [n_heads, head_dim], applied per-head.
 * pos: absolute position index.
 * theta: rope_theta (1e6 for Qwen2.5). */
static void apply_rope(float *vec, int n_heads, int head_dim,
                       int pos, float theta) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = &vec[h * head_dim];
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float x0 = head[2 * i];
            float x1 = head[2 * i + 1];
            head[2 * i]     = x0 * cos_a - x1 * sin_a;
            head[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/* ================================================================
 * SwiGLU FFN forward
 * ================================================================ */

/* SwiGLU: out = down_proj(silu(gate_proj(x)) * up_proj(x))
 * gate_proj, up_proj: [intermediate, C]
 * down_proj: [C, intermediate] */
static void swiglu_ffn(const float *x, const rfa_block *blk,
                       const rfa_block_q8 *bq8,
                       float *out, float *ffn_buf, float *ffn_buf2,
                       int c, int inter, int8_t *q8_buf) {
    /* gate = gate_proj @ x */
    if (bq8 && bq8->gate_q8)
        rfa_matvec_q8(x, bq8->gate_q8, ffn_buf, inter, c, q8_buf);
    else
        fast_matvec(x, tm_data_f32(blk->gate_proj), ffn_buf, c, inter);
    /* up = up_proj @ x */
    if (bq8 && bq8->up_q8)
        rfa_matvec_q8(x, bq8->up_q8, ffn_buf2, inter, c, q8_buf);
    else
        fast_matvec(x, tm_data_f32(blk->up_proj), ffn_buf2, c, inter);
    /* act = silu(gate) * up */
    for (int i = 0; i < inter; i++)
        ffn_buf[i] = fast_silu(ffn_buf[i]) * ffn_buf2[i];
    /* out = down_proj @ act */
    if (bq8 && bq8->down_q8)
        rfa_matvec_q8(ffn_buf, bq8->down_q8, out, c, inter, q8_buf);
    else
        fast_matvec(ffn_buf, tm_data_f32(blk->down_proj), out, inter, c);
}

/* ================================================================
 * RWKV-7 Time Mix Forward (single token)
 * ================================================================ */

/* WKV-7 kernel for a single token: updates state matrix in-place.
 * State: [n_head, head_size, head_size]
 * Inputs: r, k, v, w, a, b all [C] reshaped to [n_head, head_size] */
static void rfa_wkv7_one(const float *r, const float *w,
                          const float *k, const float *v,
                          const float *neg_kk, const float *kk_a,
                          float *state, float *y,
                          int n_head, int head_size) {
    for (int h = 0; h < n_head; h++) {
        int ho = h * head_size;
        int so = h * head_size * head_size;
        for (int i = 0; i < head_size; i++) {
            float ri = r[ho + i];
            float dot = 0.0f;
            for (int j = 0; j < head_size; j++)
                dot += state[so + i * head_size + j] * ri;
            /* This isn't quite right — accumulate across i for y[ho+j] */
            /* Actually WKV7 output: y[h,j] = sum_i(r[h,i] * state[h,i,j]) */
            (void)dot; /* handled below */
        }
        /* Correct: y[h,j] = sum_i(r[h,i] * state[h,i,j]) */
        for (int j = 0; j < head_size; j++) {
            float sum = 0.0f;
            for (int i = 0; i < head_size; i++)
                sum += r[ho + i] * state[so + i * head_size + j];
            y[ho + j] = sum;
        }
        /* State update: state[h,i,j] = state[h,i,j] * w[h,i]
         *   + k_mod[h,i] * v[h,j]
         *   + state[h,i,j] * neg_kk[h,i] * kk_a_sum... */
        /* Simplified: state[h,i,j] = w[h,i] * state[h,i,j]
         *   + kk_a[h,i] * sum_k(neg_kk[h,k] * state[h,k,j])
         *   + k[h,i] * v[h,j] */
        float sa_buf[head_size]; /* temp for state @ neg_kk per column */
        for (int j = 0; j < head_size; j++) {
            /* sa = sum_i(neg_kk[h,i] * state[h,i,j]) */
            float sa = 0.0f;
            for (int i = 0; i < head_size; i++)
                sa += neg_kk[ho + i] * state[so + i * head_size + j];
            sa_buf[j] = sa;
        }
        for (int i = 0; i < head_size; i++) {
            float wi = w[ho + i];
            float ki = k[ho + i];
            float ai = kk_a[ho + i];
            for (int j = 0; j < head_size; j++) {
                int idx = so + i * head_size + j;
                state[idx] = wi * state[idx]
                           + ai * sa_buf[j]
                           + ki * v[ho + j];
            }
        }
    }
}

static void rwkv7_time_mix(const rfa_rwkv_attn *attn, int layer_id,
                           const float *xx, rfa_rwkv_state *st,
                           float *v_first, rfa_config *cfg,
                           const rfa_block_q8 *bq8,
                           rfa_workspace *ws, float *output) {
    int c = cfg->n_embd;
    int n_head = cfg->n_wkv_heads;
    int head_size = cfg->wkv_head_size;

    float *prev = tm_data_f32(st->att_x_prev);
    float *shifted = ws->buf[0];
    float *r_vec   = ws->buf[1];
    float *k_vec   = ws->buf[2];
    float *v_vec   = ws->buf[3];
    float *w_vec   = ws->buf[4];
    float *a_vec   = ws->buf[5];
    float *kk      = ws->buf[6];
    float *k_mod   = ws->buf[7];
    float *v_final = ws->buf[8];
    float *y       = ws->buf[9];
    float *tmp     = ws->buf[10];
    float *tmp2    = ws->buf[11];
    float *xv_save = ws->buf[12];
    float *lora    = ws->lora_buf;

    /* Token shift + projections */
    fast_token_shift(xx, prev, tm_data_f32(attn->x_r), shifted, c);
    if (bq8 && bq8->receptance_q8)
        rfa_matvec_q8(shifted, bq8->receptance_q8, r_vec, c, c, ws->q8_buf);
    else
        fast_matvec(shifted, tm_data_f32(attn->receptance_w), r_vec, c, c);

    fast_token_shift(xx, prev, tm_data_f32(attn->x_k), shifted, c);
    if (bq8 && bq8->key_q8)
        rfa_matvec_q8(shifted, bq8->key_q8, k_vec, c, c, ws->q8_buf);
    else
        fast_matvec(shifted, tm_data_f32(attn->key_w), k_vec, c, c);

    fast_token_shift(xx, prev, tm_data_f32(attn->x_v), shifted, c);
    memcpy(xv_save, shifted, c * sizeof(float));
    if (bq8 && bq8->value_q8)
        rfa_matvec_q8(shifted, bq8->value_q8, v_vec, c, c, ws->q8_buf);
    else
        fast_matvec(shifted, tm_data_f32(attn->value_w), v_vec, c, c);

    /* Decay: w = exp(-softplus(w0 + tanh(xw @ w1) @ w2) - 0.5) */
    {
        int ld = (int)tm_size(attn->w1, 1);
        float *w0d = tm_data_f32(attn->w0);
        fast_token_shift(xx, prev, tm_data_f32(attn->x_w), shifted, c);
        fast_matvec(shifted, tm_data_f32(attn->w1), lora, c, ld);
        for (int i = 0; i < ld; i++) lora[i] = tanhf(lora[i]);
        fast_matvec(lora, tm_data_f32(attn->w2), w_vec, ld, c);
        for (int i = 0; i < c; i++) {
            float sp = logf(1.0f + expf(w0d[i] + w_vec[i]));
            w_vec[i] = expf(-sp - 0.5f);
        }
    }

    /* Gate a: a = sigmoid(a0 + (xa @ a1) @ a2) */
    {
        int ld = (int)tm_size(attn->a1, 1);
        float *a0d = tm_data_f32(attn->a0);
        fast_token_shift(xx, prev, tm_data_f32(attn->x_a), shifted, c);
        fast_matvec(shifted, tm_data_f32(attn->a1), lora, c, ld);
        fast_matvec(lora, tm_data_f32(attn->a2), a_vec, ld, c);
        for (int i = 0; i < c; i++)
            a_vec[i] = 1.0f / (1.0f + expf(-(a0d[i] + a_vec[i])));
    }

    /* Update prev (safe now — all shifts done) */
    memcpy(prev, xx, c * sizeof(float));

    /* Key normalization: kk = L2_normalize(k * k_k) per head */
    {
        float *kkd = tm_data_f32(attn->k_k);
        for (int i = 0; i < c; i++) tmp[i] = k_vec[i] * kkd[i];
        fast_l2_normalize_per_head(tmp, kk, n_head, head_size);
    }

    /* Modify key: k_mod = k * (1 + (a-1) * k_a) */
    {
        float *kad = tm_data_f32(attn->k_a);
        for (int i = 0; i < c; i++)
            k_mod[i] = k_vec[i] * (1.0f + (a_vec[i] - 1.0f) * kad[i]);
    }

    /* Value first-token gating */
    if (layer_id == 0) {
        memcpy(v_final, v_vec, c * sizeof(float));
        memcpy(v_first, v_vec, c * sizeof(float));
    } else {
        int vld = (int)tm_size(attn->v1, 1);
        float *v0d = tm_data_f32(attn->v0);
        fast_matvec(xv_save, tm_data_f32(attn->v1), lora, c, vld);
        fast_matvec(lora, tm_data_f32(attn->v2), tmp, vld, c);
        for (int i = 0; i < c; i++) {
            float blend = 1.0f / (1.0f + expf(-(v0d[i] + tmp[i])));
            v_final[i] = v_vec[i] + (v_first[i] - v_vec[i]) * blend;
        }
    }

    /* WKV-7 kernel */
    for (int i = 0; i < c; i++) {
        tmp[i] = -kk[i];
        tmp2[i] = kk[i] * a_vec[i];
    }
    rfa_wkv7_one(r_vec, w_vec, k_mod, v_final,
                 tmp, tmp2, tm_data_f32(st->att_kv), y,
                 n_head, head_size);

    /* Bonus term: y += (r * k * r_k).sum(-1, keepdim=True) * v */
    {
        float *rkd = tm_data_f32(attn->r_k);
        for (int h = 0; h < n_head; h++) {
            int ho = h * head_size;
            float dot = 0.0f;
            for (int j = 0; j < head_size; j++)
                dot += r_vec[ho + j] * k_vec[ho + j] * rkd[h * head_size + j];
            for (int j = 0; j < head_size; j++)
                y[ho + j] += dot * v_final[ho + j];
        }
    }

    /* Output projection */
    if (bq8 && bq8->output_q8)
        rfa_matvec_q8(y, bq8->output_q8, output, c, c, ws->q8_buf);
    else
        fast_matvec(y, tm_data_f32(attn->output_w), output, c, c);
}

/* ================================================================
 * Qwen GQA Attention Forward (single token, decode mode)
 * ================================================================ */

static void qwen_attention(const rfa_qwen_attn *attn, const float *x,
                           rfa_kv_cache *cache, rfa_config *cfg,
                           const rfa_block_q8 *bq8,
                           int position, rfa_workspace *ws, float *output) {
    int c = cfg->n_embd;
    int n_q = cfg->n_q_heads;
    int n_kv = cfg->n_kv_heads;
    int hd = cfg->head_dim;
    int kv_dim = n_kv * hd;
    int group_size = n_q / n_kv;

    float *q = ws->buf[1];    /* [C] = [n_q * hd] */
    float *k = ws->buf[2];    /* [kv_dim] = [n_kv * hd] */
    float *v = ws->buf[3];    /* [kv_dim] */
    float *y = ws->buf[9];    /* [C] output */

    /* Q/K/V projections */
    if (bq8 && bq8->q_proj_q8)
        rfa_matvec_q8(x, bq8->q_proj_q8, q, c, c, ws->q8_buf);
    else
        fast_matvec(x, tm_data_f32(attn->q_proj_w), q, c, c);
    if (attn->q_proj_b) fast_add_bias(q, tm_data_f32(attn->q_proj_b), c);

    if (bq8 && bq8->k_proj_q8)
        rfa_matvec_q8(x, bq8->k_proj_q8, k, kv_dim, c, ws->q8_buf);
    else
        fast_matvec(x, tm_data_f32(attn->k_proj_w), k, c, kv_dim);
    if (attn->k_proj_b) fast_add_bias(k, tm_data_f32(attn->k_proj_b), kv_dim);

    if (bq8 && bq8->v_proj_q8)
        rfa_matvec_q8(x, bq8->v_proj_q8, v, kv_dim, c, ws->q8_buf);
    else
        fast_matvec(x, tm_data_f32(attn->v_proj_w), v, c, kv_dim);
    if (attn->v_proj_b) fast_add_bias(v, tm_data_f32(attn->v_proj_b), kv_dim);

    /* Apply RoPE to Q and K */
    apply_rope(q, n_q, hd, position, cfg->rope_theta);
    apply_rope(k, n_kv, hd, position, cfg->rope_theta);

    /* Append to KV cache */
    int ct = cache->cache_len;
    if (ct >= cache->max_cache_len) {
        /* Simple: drop oldest half */
        int keep = cache->max_cache_len / 2;
        int drop = ct - keep;
        memmove(cache->k_cache, &cache->k_cache[drop * kv_dim],
                keep * kv_dim * sizeof(float));
        memmove(cache->v_cache, &cache->v_cache[drop * kv_dim],
                keep * kv_dim * sizeof(float));
        ct = keep;
        cache->cache_len = keep;
    }
    memcpy(&cache->k_cache[ct * kv_dim], k, kv_dim * sizeof(float));
    memcpy(&cache->v_cache[ct * kv_dim], v, kv_dim * sizeof(float));
    cache->cache_len = ct + 1;
    int total = ct + 1;

    /* GQA attention: per Q-head, dot with matching KV head across cache */
    float scale = 1.0f / sqrtf((float)hd);
    memset(y, 0, c * sizeof(float));

    for (int qh = 0; qh < n_q; qh++) {
        int kvh = qh / group_size;  /* which KV head this Q head uses */
        float *q_head = &q[qh * hd];

        /* Compute attention scores: Q[qh] @ K[kvh, t]^T for all t */
        float *scores = ws->attn_buf;
        float max_s = -1e30f;
        for (int t = 0; t < total; t++) {
            float *kt = &cache->k_cache[t * kv_dim + kvh * hd];
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
        float inv_sum = 1.0f / sum_e;
        for (int t = 0; t < total; t++)
            scores[t] *= inv_sum;

        /* Value aggregation: y[qh] = sum_t(scores[t] * V[kvh, t]) */
        float *y_head = &y[qh * hd];
        for (int t = 0; t < total; t++) {
            float *vt = &cache->v_cache[t * kv_dim + kvh * hd];
            float s = scores[t];
            for (int j = 0; j < hd; j++)
                y_head[j] += s * vt[j];
        }
    }

    /* Output projection */
    if (bq8 && bq8->o_proj_q8)
        rfa_matvec_q8(y, bq8->o_proj_q8, output, c, c, ws->q8_buf);
    else
        fast_matvec(y, tm_data_f32(attn->o_proj_w), output, c, c);
}

/* ================================================================
 * Block forward (unified for RWKV and Qwen)
 * ================================================================ */

static void rfa_block_forward(rfa_block *blk, int layer_id,
                              float *x, rfa_rwkv_state *rwkv_st,
                              rfa_kv_cache *kv_cache, float *v_first,
                              rfa_config *cfg, int position,
                              const rfa_block_q8 *bq8,
                              rfa_workspace *ws, float *x_out) {
    int c = cfg->n_embd;
    int inter = cfg->intermediate_size;
    float eps = cfg->rms_norm_eps;
    float *ln_out = ws->buf[15];
    float *attn_out = ws->buf[14];
    float *ffn_out = ws->buf[13];

    /* Pre-attention RMSNorm */
    fast_rms_norm(x, tm_data_f32(blk->ln1_w), ln_out, c, eps);

    /* Attention (RWKV-7 or Qwen GQA) */
    if (blk->is_qwen) {
        qwen_attention(&blk->qwen, ln_out, kv_cache, cfg,
                       bq8, position, ws, attn_out);
    } else {
        rwkv7_time_mix(&blk->rwkv, layer_id, ln_out, rwkv_st,
                       v_first, cfg, bq8, ws, attn_out);
    }

    /* Residual: x1 = x + attn_out */
    for (int i = 0; i < c; i++)
        x_out[i] = x[i] + attn_out[i];

    /* Pre-FFN RMSNorm */
    fast_rms_norm(x_out, tm_data_f32(blk->ln2_w), ln_out, c, eps);

    /* SwiGLU FFN */
    swiglu_ffn(ln_out, blk, bq8, ffn_out, ws->ffn_buf,
               ws->buf[0], /* reuse buf[0] as ffn_buf2 */
               c, inter, ws->q8_buf);

    /* Residual: x2 = x1 + ffn_out */
    fast_residual_add(x_out, ffn_out, c);
}

/* ================================================================
 * Weight loading
 * ================================================================ */

/* Convert BF16 raw bytes to F32 float array. Caller must free. */
static float* bf16_to_f32(const void *raw, int64_t numel) {
    const uint16_t *src = (const uint16_t*)raw;
    float *dst = (float*)malloc(numel * sizeof(float));
    for (int64_t i = 0; i < numel; i++) {
        uint32_t bits = (uint32_t)src[i] << 16;
        memcpy(&dst[i], &bits, sizeof(float));
    }
    return dst;
}

/* Load tensor from safetensors, handle BF16->F32 conversion and transpose.
 * Returns owned tensor (caller must tm_free). NULL on missing. */
static tm_tensor* load_weight(fs_file *f, const char *name, bool required) {
    if (!fs_has_tensor(f, name)) {
        if (required)
            fprintf(stderr, "rfa: missing weight: %s\n", name);
        return NULL;
    }

    fs_tensor_info *info = fs_get_info(f, name);
    if (!info) return NULL;

    tm_tensor *t = NULL;

    if (info->dtype == FS_DTYPE_F32) {
        const float *data = fs_get_f32(f, name);
        if (!data) { fs_free_info(info); return NULL; }
        tm_tensor *t_view = tm_from_data((void*)data, info->shape, info->ndim, TM_F32);
        t = tm_clone(t_view);
        tm_free(t_view);
    } else if (info->dtype == FS_DTYPE_BF16) {
        const void *raw = fs_get_data(f, name);
        if (!raw) { fs_free_info(info); return NULL; }
        int64_t numel = 1;
        for (int d = 0; d < info->ndim; d++) numel *= info->shape[d];
        float *f32_data = bf16_to_f32(raw, numel);
        tm_tensor *t_view = tm_from_data(f32_data, info->shape, info->ndim, TM_F32);
        t = tm_clone(t_view);
        tm_free(t_view);
        free(f32_data);
    } else {
        fprintf(stderr, "rfa: unsupported dtype for %s\n", name);
        fs_free_info(info);
        return NULL;
    }

    /* Transpose 2D linear weights: [out, in] -> [in, out] for fast_matvec */
    bool is_linear = strstr(name, ".weight") != NULL &&
                     (strstr(name, "proj") || strstr(name, "receptance") ||
                      strstr(name, "key.weight") || strstr(name, "value.weight") ||
                      strstr(name, "output.weight") || strstr(name, "head.weight") ||
                      strstr(name, "gate_proj") || strstr(name, "up_proj") ||
                      strstr(name, "down_proj"));
    if (is_linear && info->ndim == 2) {
        tm_tensor *tt = tm_t(t);
        tm_tensor *tc = tm_contiguous(tt);
        tm_free(t); tm_free(tt);
        t = tc;
    }

    /* Squeeze singleton dims: [1,1,C] -> [C] */
    for (int d = tm_ndim(t) - 1; d >= 0; d--) {
        if (tm_size(t, d) == 1 && tm_ndim(t) > 1) {
            tm_tensor *sq = tm_squeeze(t, d);
            tm_free(t);
            t = sq;
        }
    }

    fs_free_info(info);
    return t;
}

rfa_model* rfa_model_load(const char *path) {
    fs_file *f = fs_open(path);
    if (!f) {
        fprintf(stderr, "rfa: cannot open %s\n", path);
        return NULL;
    }

    rfa_config cfg = {0};

    /* Parse config from metadata (simple sscanf-based) */
    /* Default values for 1.5B */
    cfg.n_embd = 1536;
    cfg.n_q_heads = 12;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 128;
    cfg.n_wkv_heads = 24;
    cfg.wkv_head_size = 64;
    cfg.intermediate_size = 8960;
    cfg.n_layers = 28;
    cfg.vocab_size = 151936;
    cfg.rope_theta = 1000000.0f;
    cfg.rms_norm_eps = 1e-6f;

    /* Override from metadata if present */
    {
        const char *v;
        #define META_INT(key, field) do { \
            v = fs_get_metadata(f, key); \
            if (v) cfg.field = atoi(v); \
        } while(0)
        #define META_FLOAT(key, field) do { \
            v = fs_get_metadata(f, key); \
            if (v) cfg.field = strtof(v, NULL); \
        } while(0)
        META_INT("hidden_size", n_embd);
        META_INT("num_attention_heads", n_q_heads);
        META_INT("num_key_value_heads", n_kv_heads);
        META_INT("head_dim", head_dim);
        META_INT("num_wkv_heads", n_wkv_heads);
        META_INT("wkv_head_size", wkv_head_size);
        META_INT("intermediate_size", intermediate_size);
        META_INT("num_hidden_layers", n_layers);
        META_INT("vocab_size", vocab_size);
        META_FLOAT("rope_theta", rope_theta);
        META_FLOAT("rms_norm_eps", rms_norm_eps);
        #undef META_INT
        #undef META_FLOAT
    }

    /* Determine block types from metadata or tensor names */
    int *block_types = (int*)calloc(cfg.n_layers, sizeof(int)); /* 0=rwkv, 1=qwen */
    int n_rwkv = 0, n_qwen = 0;

    /* Try metadata first */
    const char *bt_str = fs_get_metadata(f, "block_types");
    if (bt_str) {
        /* Parse JSON array of "rwkv"/"qwen" strings */
        const char *arr = strchr(bt_str, '[');
        if (arr) {
            arr++;
            for (int i = 0; i < cfg.n_layers && *arr; i++) {
                const char *q = strstr(arr, "qwen");
                const char *next_comma = strchr(arr, ',');
                const char *next_bracket = strchr(arr, ']');
                const char *end = next_comma ? next_comma : next_bracket;
                if (!end) break;
                if (q && q < end) {
                    block_types[i] = 1;
                    n_qwen++;
                } else {
                    block_types[i] = 0;
                    n_rwkv++;
                }
                arr = end + 1;
            }
        }
    }

    /* Fallback: detect from tensor names */
    if (n_rwkv == 0 && n_qwen == 0) {
        for (int i = 0; i < cfg.n_layers; i++) {
            char qw_key[256];
            snprintf(qw_key, sizeof(qw_key),
                     "model.layers.%d.self_attn.q_proj.weight", i);
            if (fs_has_tensor(f, qw_key)) {
                block_types[i] = 1;
                n_qwen++;
            } else {
                block_types[i] = 0;
                n_rwkv++;
            }
        }
    }

    cfg.n_rwkv_layers = n_rwkv;
    cfg.n_qwen_layers = n_qwen;

    printf("rfa: loading %d layers (%d RWKV, %d Qwen), %dd, vocab=%d\n",
           cfg.n_layers, n_rwkv, n_qwen, cfg.n_embd, cfg.vocab_size);

    /* Allocate model */
    rfa_model *m = (rfa_model*)calloc(1, sizeof(rfa_model));
    m->config = cfg;
    m->blocks = (rfa_block*)calloc(cfg.n_layers, sizeof(rfa_block));

    /* Global weights */
    m->emb_weight = load_weight(f, "model.embed_tokens.weight", true);
    m->ln_out_w = load_weight(f, "model.norm.weight", true);

    /* Head weight: tied or separate */
    m->head_weight = load_weight(f, "lm_head.weight", false);
    if (!m->head_weight) {
        /* Tied embeddings: head = embed_tokens transposed */
        printf("rfa: using tied embeddings for lm_head\n");
        m->head_weight = tm_clone(m->emb_weight);
        /* Transpose [vocab, C] -> [C, vocab] for matvec */
        int v = cfg.vocab_size, c = cfg.n_embd;
        float *src = tm_data_f32(m->head_weight);
        float *transposed = (float*)malloc(v * c * sizeof(float));
        for (int i = 0; i < v; i++)
            for (int j = 0; j < c; j++)
                transposed[j * v + i] = src[i * c + j];
        memcpy(src, transposed, v * c * sizeof(float));
        free(transposed);
    }

    /* Load each block */
    for (int i = 0; i < cfg.n_layers; i++) {
        rfa_block *blk = &m->blocks[i];
        blk->is_qwen = block_types[i];
        char buf[256];

        /* Norms (all layers) */
        snprintf(buf, sizeof(buf), "model.layers.%d.input_layernorm.weight", i);
        blk->ln1_w = load_weight(f, buf, true);

        snprintf(buf, sizeof(buf), "model.layers.%d.post_attention_layernorm.weight", i);
        blk->ln2_w = load_weight(f, buf, true);

        /* SwiGLU FFN (all layers) */
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.gate_proj.weight", i);
        blk->gate_proj = load_weight(f, buf, true);

        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.up_proj.weight", i);
        blk->up_proj = load_weight(f, buf, true);

        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.down_proj.weight", i);
        blk->down_proj = load_weight(f, buf, true);

        if (blk->is_qwen) {
            /* Qwen GQA attention */
            rfa_qwen_attn *qa = &blk->qwen;
            snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_proj.weight", i);
            qa->q_proj_w = load_weight(f, buf, true);
            snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_proj.bias", i);
            qa->q_proj_b = load_weight(f, buf, false);
            snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_proj.weight", i);
            qa->k_proj_w = load_weight(f, buf, true);
            snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_proj.bias", i);
            qa->k_proj_b = load_weight(f, buf, false);
            snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.v_proj.weight", i);
            qa->v_proj_w = load_weight(f, buf, true);
            snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.v_proj.bias", i);
            qa->v_proj_b = load_weight(f, buf, false);
            snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.o_proj.weight", i);
            qa->o_proj_w = load_weight(f, buf, true);
            printf("  Layer %d: Qwen GQA\n", i);
        } else {
            /* RWKV-7 time mixer */
            rfa_rwkv_attn *ra = &blk->rwkv;
            #define LOAD_RWKV(field, fmt, ...) do { \
                snprintf(buf, sizeof(buf), fmt, __VA_ARGS__); \
                ra->field = load_weight(f, buf, true); \
            } while(0)

            LOAD_RWKV(x_r, "model.layers.%d.self_attn.time_mixer.x_r", i);
            LOAD_RWKV(x_w, "model.layers.%d.self_attn.time_mixer.x_w", i);
            LOAD_RWKV(x_k, "model.layers.%d.self_attn.time_mixer.x_k", i);
            LOAD_RWKV(x_v, "model.layers.%d.self_attn.time_mixer.x_v", i);
            LOAD_RWKV(x_a, "model.layers.%d.self_attn.time_mixer.x_a", i);
            LOAD_RWKV(w0,  "model.layers.%d.self_attn.time_mixer.w0", i);
            LOAD_RWKV(w1,  "model.layers.%d.self_attn.time_mixer.w1", i);
            LOAD_RWKV(w2,  "model.layers.%d.self_attn.time_mixer.w2", i);
            LOAD_RWKV(a0,  "model.layers.%d.self_attn.time_mixer.a0", i);
            LOAD_RWKV(a1,  "model.layers.%d.self_attn.time_mixer.a1", i);
            LOAD_RWKV(a2,  "model.layers.%d.self_attn.time_mixer.a2", i);
            LOAD_RWKV(v0,  "model.layers.%d.self_attn.time_mixer.v0", i);
            LOAD_RWKV(v1,  "model.layers.%d.self_attn.time_mixer.v1", i);
            LOAD_RWKV(v2,  "model.layers.%d.self_attn.time_mixer.v2", i);
            LOAD_RWKV(k_k, "model.layers.%d.self_attn.time_mixer.k_k", i);
            LOAD_RWKV(k_a, "model.layers.%d.self_attn.time_mixer.k_a", i);
            LOAD_RWKV(r_k, "model.layers.%d.self_attn.time_mixer.r_k", i);

            LOAD_RWKV(receptance_w, "model.layers.%d.self_attn.time_mixer.receptance.weight", i);
            LOAD_RWKV(key_w,       "model.layers.%d.self_attn.time_mixer.key.weight", i);
            LOAD_RWKV(value_w,     "model.layers.%d.self_attn.time_mixer.value.weight", i);
            LOAD_RWKV(output_w,    "model.layers.%d.self_attn.time_mixer.output.weight", i);

            #undef LOAD_RWKV
        }
    }

    free(block_types);
    fs_close(f);

    printf("rfa: model loaded successfully\n");
    return m;
}

/* Quantize all linear weights to INT8 */
static void rfa_model_quantize_weights(rfa_model *m) {
    rfa_config cfg = m->config;
    int c = cfg.n_embd;
    int inter = cfg.intermediate_size;
    int kv_dim = cfg.n_kv_heads * cfg.head_dim;

    m->blocks_q8 = (rfa_block_q8*)calloc(cfg.n_layers, sizeof(rfa_block_q8));

    for (int i = 0; i < cfg.n_layers; i++) {
        rfa_block *blk = &m->blocks[i];
        rfa_block_q8 *bq8 = &m->blocks_q8[i];

        /* FFN weights (all blocks) — [rows, cols] already transposed */
        bq8->gate_q8 = rfa_q8_from_f32(tm_data_f32(blk->gate_proj), inter, c);
        bq8->up_q8 = rfa_q8_from_f32(tm_data_f32(blk->up_proj), inter, c);
        bq8->down_q8 = rfa_q8_from_f32(tm_data_f32(blk->down_proj), c, inter);

        if (blk->is_qwen) {
            /* Qwen attention projections */
            bq8->q_proj_q8 = rfa_q8_from_f32(tm_data_f32(blk->qwen.q_proj_w), c, c);
            bq8->k_proj_q8 = rfa_q8_from_f32(tm_data_f32(blk->qwen.k_proj_w), kv_dim, c);
            bq8->v_proj_q8 = rfa_q8_from_f32(tm_data_f32(blk->qwen.v_proj_w), kv_dim, c);
            bq8->o_proj_q8 = rfa_q8_from_f32(tm_data_f32(blk->qwen.o_proj_w), c, c);
        } else {
            /* RWKV attention projections */
            bq8->receptance_q8 = rfa_q8_from_f32(tm_data_f32(blk->rwkv.receptance_w), c, c);
            bq8->key_q8 = rfa_q8_from_f32(tm_data_f32(blk->rwkv.key_w), c, c);
            bq8->value_q8 = rfa_q8_from_f32(tm_data_f32(blk->rwkv.value_w), c, c);
            bq8->output_q8 = rfa_q8_from_f32(tm_data_f32(blk->rwkv.output_w), c, c);
        }
    }

    /* Head weight: [vocab, C] transposed layout */
    m->head_q8 = rfa_q8_from_f32(tm_data_f32(m->head_weight),
                                   cfg.vocab_size, c);

    m->quantized = true;
    printf("rfa: quantized all weights to INT8\n");
}

rfa_model* rfa_model_load_q8(const char *path) {
    rfa_model *m = rfa_model_load(path);
    if (!m) return NULL;
    rfa_model_quantize_weights(m);
    return m;
}

void rfa_model_free(rfa_model *model) {
    if (!model) return;
    rfa_config cfg = model->config;

    tm_free(model->emb_weight);
    tm_free(model->ln_out_w);
    tm_free(model->head_weight);

    /* Free INT8 weights */
    if (model->blocks_q8) {
        for (int i = 0; i < cfg.n_layers; i++) {
            rfa_block_q8 *bq8 = &model->blocks_q8[i];
            rfa_q8_free(bq8->gate_q8);
            rfa_q8_free(bq8->up_q8);
            rfa_q8_free(bq8->down_q8);
            rfa_q8_free(bq8->q_proj_q8);
            rfa_q8_free(bq8->k_proj_q8);
            rfa_q8_free(bq8->v_proj_q8);
            rfa_q8_free(bq8->o_proj_q8);
            rfa_q8_free(bq8->receptance_q8);
            rfa_q8_free(bq8->key_q8);
            rfa_q8_free(bq8->value_q8);
            rfa_q8_free(bq8->output_q8);
        }
        free(model->blocks_q8);
    }
    rfa_q8_free(model->head_q8);

    for (int i = 0; i < cfg.n_layers; i++) {
        rfa_block *blk = &model->blocks[i];
        tm_free(blk->ln1_w);
        tm_free(blk->ln2_w);
        tm_free(blk->gate_proj);
        tm_free(blk->up_proj);
        tm_free(blk->down_proj);

        if (blk->is_qwen) {
            tm_free(blk->qwen.q_proj_w);
            tm_free(blk->qwen.q_proj_b);
            tm_free(blk->qwen.k_proj_w);
            tm_free(blk->qwen.k_proj_b);
            tm_free(blk->qwen.v_proj_w);
            tm_free(blk->qwen.v_proj_b);
            tm_free(blk->qwen.o_proj_w);
        } else {
            rfa_rwkv_attn *ra = &blk->rwkv;
            tm_free(ra->x_r); tm_free(ra->x_w); tm_free(ra->x_k);
            tm_free(ra->x_v); tm_free(ra->x_a);
            tm_free(ra->w0); tm_free(ra->w1); tm_free(ra->w2);
            tm_free(ra->a0); tm_free(ra->a1); tm_free(ra->a2);
            tm_free(ra->v0); tm_free(ra->v1); tm_free(ra->v2);
            tm_free(ra->k_k); tm_free(ra->k_a); tm_free(ra->r_k);
            tm_free(ra->receptance_w); tm_free(ra->key_w);
            tm_free(ra->value_w); tm_free(ra->output_w);
        }
    }
    free(model->blocks);
    free(model);
}

rfa_config rfa_model_config(const rfa_model *model) {
    return model->config;
}

/* ================================================================
 * State lifecycle
 * ================================================================ */

static rfa_workspace* workspace_new(rfa_config cfg, int max_cache) {
    int c = cfg.n_embd;
    int inter = cfg.intermediate_size;
    rfa_workspace *ws = (rfa_workspace*)calloc(1, sizeof(rfa_workspace));
    for (int i = 0; i < 16; i++)
        ws->buf[i] = (float*)calloc(c, sizeof(float));
    ws->ffn_buf = (float*)calloc(inter, sizeof(float));
    ws->lora_buf = (float*)calloc(256, sizeof(float)); /* max LoRA dim */
    ws->x_buf[0] = (float*)calloc(c, sizeof(float));
    ws->x_buf[1] = (float*)calloc(c, sizeof(float));
    ws->attn_buf = (float*)calloc(max_cache + 64, sizeof(float));
    /* q8_buf needs to be large enough for max(C, intermediate_size) */
    int q8_size = inter > c ? inter : c;
    ws->q8_buf = (int8_t*)calloc(q8_size + 64, sizeof(int8_t));
    return ws;
}

static void workspace_free(rfa_workspace *ws) {
    if (!ws) return;
    for (int i = 0; i < 16; i++) free(ws->buf[i]);
    free(ws->ffn_buf);
    free(ws->lora_buf);
    free(ws->x_buf[0]);
    free(ws->x_buf[1]);
    free(ws->attn_buf);
    free(ws->q8_buf);
    free(ws);
}

rfa_state* rfa_state_new(const rfa_model *model, int max_cache_len) {
    if (!model) return NULL;
    rfa_config cfg = model->config;
    if (max_cache_len <= 0) max_cache_len = 2048;

    rfa_state *s = (rfa_state*)calloc(1, sizeof(rfa_state));
    s->config = cfg;
    s->position = 0;

    /* Count RWKV and Qwen blocks */
    s->n_rwkv = 0;
    s->n_qwen = 0;
    for (int i = 0; i < cfg.n_layers; i++) {
        if (model->blocks[i].is_qwen) s->n_qwen++;
        else s->n_rwkv++;
    }

    /* Allocate RWKV states */
    s->rwkv_states = (rfa_rwkv_state*)calloc(s->n_rwkv, sizeof(rfa_rwkv_state));
    for (int i = 0; i < s->n_rwkv; i++) {
        int64_t prev_shape[] = {cfg.n_embd};
        s->rwkv_states[i].att_x_prev = tm_zeros(prev_shape, 1, TM_F32);

        int64_t kv_shape[] = {cfg.n_wkv_heads, cfg.wkv_head_size, cfg.wkv_head_size};
        s->rwkv_states[i].att_kv = tm_zeros(kv_shape, 3, TM_F32);
    }

    /* Allocate Qwen KV caches */
    int kv_dim = cfg.n_kv_heads * cfg.head_dim;
    s->kv_caches = (rfa_kv_cache*)calloc(s->n_qwen, sizeof(rfa_kv_cache));
    for (int i = 0; i < s->n_qwen; i++) {
        s->kv_caches[i].k_cache = (float*)calloc(max_cache_len * kv_dim, sizeof(float));
        s->kv_caches[i].v_cache = (float*)calloc(max_cache_len * kv_dim, sizeof(float));
        s->kv_caches[i].cache_len = 0;
        s->kv_caches[i].max_cache_len = max_cache_len;
    }

    /* v_first */
    s->v_first = (float*)calloc(cfg.n_embd, sizeof(float));

    /* Workspace */
    s->workspace = workspace_new(cfg, max_cache_len);

    return s;
}

void rfa_state_free(rfa_state *state) {
    if (!state) return;

    for (int i = 0; i < state->n_rwkv; i++) {
        tm_free(state->rwkv_states[i].att_x_prev);
        tm_free(state->rwkv_states[i].att_kv);
    }
    free(state->rwkv_states);

    for (int i = 0; i < state->n_qwen; i++) {
        free(state->kv_caches[i].k_cache);
        free(state->kv_caches[i].v_cache);
    }
    free(state->kv_caches);

    free(state->v_first);
    workspace_free(state->workspace);
    free(state);
}

void rfa_state_reset(rfa_state *state) {
    if (!state) return;
    rfa_config cfg = state->config;

    state->position = 0;

    for (int i = 0; i < state->n_rwkv; i++) {
        memset(tm_data_f32(state->rwkv_states[i].att_x_prev), 0,
               cfg.n_embd * sizeof(float));
        memset(tm_data_f32(state->rwkv_states[i].att_kv), 0,
               cfg.n_wkv_heads * cfg.wkv_head_size * cfg.wkv_head_size * sizeof(float));
    }

    for (int i = 0; i < state->n_qwen; i++) {
        state->kv_caches[i].cache_len = 0;
    }

    memset(state->v_first, 0, cfg.n_embd * sizeof(float));
}

/* ================================================================
 * Forward pass
 * ================================================================ */

tm_tensor* rfa_forward(rfa_model *m, uint32_t token, rfa_state *s) {
    return rfa_forward_ex(m, token, s, NULL, NULL);
}

tm_tensor* rfa_forward_ex(rfa_model *m, uint32_t token, rfa_state *s,
                           rfa_block_callback callback, void *user_data) {
    if (!m || !s) return NULL;
    rfa_config cfg = m->config;
    int c = cfg.n_embd;
    int vocab = cfg.vocab_size;
    rfa_workspace *ws = s->workspace;

    /* Embedding lookup */
    float *emb = tm_data_f32(m->emb_weight);
    float *x_cur = ws->x_buf[0];
    float *x_next = ws->x_buf[1];
    memcpy(x_cur, &emb[token * c], c * sizeof(float));

    /* Block loop */
    int rwkv_idx = 0, qwen_idx = 0;
    for (int i = 0; i < cfg.n_layers; i++) {
        rfa_block *blk = &m->blocks[i];

        /* Determine which state to use */
        rfa_rwkv_state *rst = NULL;
        rfa_kv_cache *kvc = NULL;
        int layer_id = 0;

        if (blk->is_qwen) {
            /* Pre-block callback for Qwen blocks */
            if (callback)
                callback(0, i, qwen_idx, x_cur, user_data);

            kvc = &s->kv_caches[qwen_idx];
            qwen_idx++;
        } else {
            rst = &s->rwkv_states[rwkv_idx];
            layer_id = rwkv_idx; /* for v_first gating */
            rwkv_idx++;
        }

        const rfa_block_q8 *bq8 = (m->quantized && m->blocks_q8) ?
                                    &m->blocks_q8[i] : NULL;
        rfa_block_forward(blk, layer_id, x_cur, rst, kvc,
                          s->v_first, &cfg, s->position,
                          bq8, ws, x_next);

        /* Post-block callback for Qwen blocks */
        if (blk->is_qwen && callback)
            callback(1, i, qwen_idx - 1, x_next, user_data);

        /* Swap buffers */
        float *tmp = x_cur;
        x_cur = x_next;
        x_next = tmp;
    }

    /* Final RMSNorm */
    fast_rms_norm(x_cur, tm_data_f32(m->ln_out_w), x_next,
                  c, cfg.rms_norm_eps);

    /* Head projection: logits = x @ head_weight */
    int64_t logits_shape[] = {vocab};
    tm_tensor *logits = tm_empty(logits_shape, 1, TM_F32);
    if (m->quantized && m->head_q8)
        rfa_matvec_q8(x_next, m->head_q8, tm_data_f32(logits),
                      vocab, c, ws->q8_buf);
    else
        fast_matvec(x_next, tm_data_f32(m->head_weight), tm_data_f32(logits),
                    c, vocab);

    s->position++;
    return logits;
}

/* ================================================================
 * Weight accessors for training
 * ================================================================ */

int rfa_model_get_qwen_weights(const rfa_model *model, int qwen_idx,
                                rfa_qwen_weights *out) {
    if (!model || !out || qwen_idx < 0) return -1;
    rfa_config cfg = model->config;
    memset(out, 0, sizeof(*out));

    /* Find the qwen_idx-th Qwen block */
    int qi = 0;
    for (int i = 0; i < cfg.n_layers; i++) {
        const rfa_block *blk = &model->blocks[i];
        if (!blk->is_qwen) continue;
        if (qi == qwen_idx) {
            out->ln1_w = tm_data_f32(blk->ln1_w);
            out->ln2_w = tm_data_f32(blk->ln2_w);
            out->q_proj_w = tm_data_f32(blk->qwen.q_proj_w);
            out->q_proj_b = blk->qwen.q_proj_b ? tm_data_f32(blk->qwen.q_proj_b) : NULL;
            out->k_proj_w = tm_data_f32(blk->qwen.k_proj_w);
            out->k_proj_b = blk->qwen.k_proj_b ? tm_data_f32(blk->qwen.k_proj_b) : NULL;
            out->v_proj_w = tm_data_f32(blk->qwen.v_proj_w);
            out->v_proj_b = blk->qwen.v_proj_b ? tm_data_f32(blk->qwen.v_proj_b) : NULL;
            out->o_proj_w = tm_data_f32(blk->qwen.o_proj_w);
            out->gate_proj = tm_data_f32(blk->gate_proj);
            out->up_proj = tm_data_f32(blk->up_proj);
            out->down_proj = tm_data_f32(blk->down_proj);
            out->inter_size = cfg.intermediate_size;
            return 0;
        }
        qi++;
    }
    return -1;  /* qwen_idx out of range */
}

const float* rfa_model_get_head_weight(const rfa_model *model) {
    if (!model || !model->head_weight) return NULL;
    return tm_data_f32(model->head_weight);
}

const float* rfa_model_get_ln_out_w(const rfa_model *model) {
    if (!model || !model->ln_out_w) return NULL;
    return tm_data_f32(model->ln_out_w);
}

int rfa_state_get_kv_cache(const rfa_state *state, int qwen_idx,
                            const float **k_cache, const float **v_cache) {
    if (!state || qwen_idx < 0 || qwen_idx >= state->n_qwen) return 0;
    const rfa_kv_cache *kvc = &state->kv_caches[qwen_idx];
    if (k_cache) *k_cache = kvc->k_cache;
    if (v_cache) *v_cache = kvc->v_cache;
    return kvc->cache_len;
}

int rfa_state_get_position(const rfa_state *state) {
    if (!state) return 0;
    return state->position;
}
