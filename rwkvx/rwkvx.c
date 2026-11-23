/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rwkvx.c — RWKV-X inference engine built on Foundry
 *
 * Port of app/rwkvx-core/ (Rust/Candle) to pure C using Foundry's tensor,
 * safetensors, and attention modules.
 */

#include "rwkvx/rwkvx.h"
#include "tensor/tensor.h"
#include "safetensors/safetensors.h"
#include "hybridmem/hybridmem.h"
#include "hdc/hdc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef __ARM_NEON
  #include <arm_neon.h>
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef USE_THREADS
  #include "threadpool/threadpool.h"
#endif

#ifdef USE_BLAS
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#ifdef USE_METAL
  #include "metal/fm_metal.h"
  /* Helper to pass persistent Metal buffer to fast_matmul_batch without
   * #ifdef at every call site. Evaluates to the buffer when Metal is
   * compiled, NULL otherwise. */
  #define MTL_W(ptr) ((void*)(ptr))
  #define MTL_BUF(ptr) ((void*)(ptr))
#else
  #define MTL_W(ptr) NULL
  #define MTL_BUF(ptr) NULL
#endif

#ifdef USE_CUDA
  #include "cuda/fc_cuda.h"
#endif

#ifdef USE_THREADS
/* File-scope pool pointer — set during rx_forward(), used by internal
 * parallel functions. Avoids changing all static function signatures. */
static _Thread_local fw_pool *__rx_pool = NULL;
#endif

/* Aligned malloc helper — 64-byte alignment for NEON/AVX cache line ops */
static void* aligned_malloc(size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
    return ptr;
}

/* Page-aligned malloc — enables zero-copy Metal buffer wrapping via
 * newBufferWithBytesNoCopy on Apple Silicon (requires page alignment). */
static void* page_aligned_malloc(size_t size) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) page_size = 16384;
    void *ptr = NULL;
    if (posix_memalign(&ptr, (size_t)page_size, size) != 0) return NULL;
    return ptr;
}

/* ================================================================
 * Internal: INT8 quantized weight matrix (per-row symmetric)
 * ================================================================ */

typedef struct {
    int8_t *data;       /* [rows * cols] quantized data */
    float *scales;      /* [rows] per-row scale: float = int8 * scale */
    int rows, cols;
#ifdef USE_METAL
    fm_buffer *mtl_data;    /* Metal buffer wrapping int8 weights */
    fm_buffer *mtl_scales;  /* Metal buffer wrapping per-row scales */
#endif
#ifdef USE_CUDA
    fc_buffer *cuda_data;   /* Persistent GPU copy of int8 weights */
    fc_buffer *cuda_scales; /* Persistent GPU copy of per-row scales */
#endif
} rx_q8_weight;

static rx_q8_weight* rx_q8_from_f32(const float *src, int rows, int cols) {
    rx_q8_weight *q = (rx_q8_weight*)calloc(1, sizeof(rx_q8_weight));
    q->rows = rows;
    q->cols = cols;
    q->data = (int8_t*)aligned_malloc(rows * cols * sizeof(int8_t));
    q->scales = (float*)aligned_malloc(rows * sizeof(float));

    for (int i = 0; i < rows; i++) {
        const float *row = &src[i * cols];
        /* Find abs-max for this row */
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
#ifdef USE_METAL
    if (fm_available()) {
        q->mtl_data = fm_buffer_from_ptr(q->data, rows * cols * sizeof(int8_t));
        q->mtl_scales = fm_buffer_from_ptr(q->scales, rows * sizeof(float));
    } else {
        q->mtl_data = NULL;
        q->mtl_scales = NULL;
    }
#endif
#ifdef USE_CUDA
    if (fc_available()) {
        q->cuda_data = fc_buffer_from_ptr(q->data, rows * cols * sizeof(int8_t));
        q->cuda_scales = fc_buffer_from_ptr(q->scales, rows * sizeof(float));
    } else {
        q->cuda_data = NULL;
        q->cuda_scales = NULL;
    }
#endif
    return q;
}

static void rx_q8_free(rx_q8_weight *q) {
    if (!q) return;
#ifdef USE_METAL
    if (q->mtl_data) fm_buffer_free(q->mtl_data);
    if (q->mtl_scales) fm_buffer_free(q->mtl_scales);
#endif
#ifdef USE_CUDA
    if (q->cuda_data) fc_buffer_free(q->cuda_data);
    if (q->cuda_scales) fc_buffer_free(q->cuda_scales);
#endif
    free(q->data);
    free(q->scales);
    free(q);
}

/* Quantize a float vector to int8 in-place into a pre-allocated buffer.
 * Returns the scale factor (amax / 127). */
static float fast_quantize_input(const float *x, int8_t *x_q, int n) {
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

#ifdef USE_THREADS
/* Context for q8 row worker callback */
typedef struct {
    const int8_t *x_q;
    float x_scale;
    const rx_q8_weight *q;
    float *out;
    int cols;
} q8_row_ctx;

static void q8_row_worker(int start, int end, void *arg);
#endif

/* Quantized matvec with pre-quantized input.
 * x_q[cols] and x_scale must come from fast_quantize_input().
 *
 * Uses 4x unrolled NEON accumulators for ILP, prefetches next weight row,
 * and parallelizes the row loop for large matrices. */
static void fast_matvec_q8_preq(const int8_t *x_q, float x_scale,
                                 const rx_q8_weight *q, float *out,
                                 int rows, int cols) {
#ifdef USE_METAL
    if (fm_available() && q->mtl_data && (int64_t)rows * cols >= FM_Q8_THRESHOLD) {
        fm_buffer *bxq = fm_buffer_from_ptr((void*)x_q, cols * sizeof(int8_t));
        fm_buffer *bout = fm_buffer_from_ptr(out, rows * sizeof(float));
        fm_matvec_q8(bxq, q->mtl_data, q->mtl_scales, bout,
                     x_scale, rows, cols);
        float *gout = (float*)fm_buffer_data(bout);
        if (gout != out) memcpy(out, gout, rows * sizeof(float));
        fm_buffer_free(bxq);
        fm_buffer_free(bout);
        return;
    }
#endif
#ifdef USE_CUDA
    if (fc_available() && q->cuda_data && (int64_t)rows * cols >= FC_Q8_THRESHOLD) {
        fc_buffer *bxq = fc_buffer_from_ptr(x_q, cols * sizeof(int8_t));
        fc_buffer *bout = fc_buffer_alloc(rows * sizeof(float));
        fc_matvec_q8(bxq, q->cuda_data, q->cuda_scales, bout,
                     x_scale, rows, cols);
        fc_sync();
        fc_buffer_to_cpu(bout, out);
        fc_buffer_free(bxq);
        fc_buffer_free(bout);
        return;
    }
#endif

#ifdef __ARM_NEON
    int cols64 = cols & ~63;
    int cols16 = cols & ~15;
#endif

    /* Parallelize rows for large matrices (bandwidth-bound on 3.6B+).
     * Threshold: 512 rows avoids overhead on small projections. */
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(rows >= 512)
#elif defined(USE_THREADS)
    if (__rx_pool && rows >= 512) {
        q8_row_ctx ctx = { x_q, x_scale, q, out, cols };
        fw_for(__rx_pool, rows, q8_row_worker, &ctx);
        return;
    }
#endif
    for (int i = 0; i < rows; i++) {
        const int8_t *row = &q->data[i * cols];
        float w_scale = q->scales[i];

        /* Prefetch next weight row */
        if (i + 1 < rows)
            __builtin_prefetch(&q->data[(i + 1) * cols], 0, 1);

#ifdef __ARM_NEON
        /* 4x unrolled accumulators for instruction-level parallelism */
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
        /* Handle remainder 16-element chunks */
        for (int j = cols64; j < cols16; j += 16)
            acc = vdotq_s32(acc, vld1q_s8(&x_q[j]), vld1q_s8(&row[j]));
        int32_t dot = vaddvq_s32(acc);
        /* Scalar tail */
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

#ifdef USE_THREADS
/* Worker callback for parallel q8 matvec — processes rows [start, end) */
static void q8_row_worker(int start, int end, void *arg) {
    q8_row_ctx *ctx = (q8_row_ctx *)arg;
    const int8_t *x_q = ctx->x_q;
    float x_scale = ctx->x_scale;
    const rx_q8_weight *q = ctx->q;
    float *out = ctx->out;
    int cols = ctx->cols;
    int rows = q->rows;
#ifdef __ARM_NEON
    int cols64 = cols & ~63;
    int cols16 = cols & ~15;
#endif
    for (int i = start; i < end; i++) {
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
#endif

/* Quantized matvec: out[rows] = x[cols] @ w_q8[rows, cols]
 * Convenience wrapper that quantizes x on the fly using the workspace q8 buffer. */
static void fast_matvec_q8(const float *x, const rx_q8_weight *q, float *out,
                           int rows, int cols, int8_t *q8_buf) {
    float x_scale = fast_quantize_input(x, q8_buf, cols);
    fast_matvec_q8_preq(q8_buf, x_scale, q, out, rows, cols);
}

/* ================================================================
 * Internal: RWKV-7 block weights
 * ================================================================ */

typedef struct {
    /* Layer norms */
    tm_tensor *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    /* Token shift lerp coefficients */
    tm_tensor *x_r, *x_w, *x_k, *x_v, *x_a, *x_g;
    /* Decay LoRA: w = exp(-0.606531 * sigmoid(w0 + tanh(xw @ w1) @ w2)) */
    tm_tensor *w0, *w1, *w2;
    /* Gate a LoRA: a = sigmoid(a0 + (xa @ a1) @ a2) */
    tm_tensor *a0, *a1, *a2;
    /* Value first-token blend LoRA (NULL for layer 0) */
    tm_tensor *v0, *v1, *v2;
    /* Output gate LoRA: g = sigmoid(xg @ g1) @ g2 */
    tm_tensor *g1, *g2;
    /* Key normalization */
    tm_tensor *k_k, *k_a, *r_k;
    /* Linear projections (already transposed: [in, out]) */
    tm_tensor *receptance_w, *key_w, *value_w, *output_w;
    /* Group norm */
    tm_tensor *ln_x_w, *ln_x_b;
    /* FFN */
    tm_tensor *ffn_x_k, *ffn_key_w, *ffn_value_w;

    /* INT8 quantized versions (NULL when not quantized) */
    rx_q8_weight *receptance_q8, *key_q8, *value_q8, *output_q8;
    rx_q8_weight *ffn_key_q8, *ffn_value_q8;

#ifdef USE_METAL
    /* Persistent Metal buffers for weight tensors (created once at model load) */
    fm_buffer *mtl_receptance_w, *mtl_key_w, *mtl_value_w, *mtl_output_w;
    fm_buffer *mtl_ffn_key_w, *mtl_ffn_value_w;
#endif
} rx_rwkv_block;

/* ================================================================
 * Internal: Sparse attention block weights
 * ================================================================ */

typedef struct {
    tm_tensor *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    tm_tensor *receptance_w, *key_w, *value_w, *output_w;
    tm_tensor *ffn_x_k, *ffn_key_w, *ffn_value_w;

    /* INT8 quantized versions (NULL when not quantized) */
    rx_q8_weight *receptance_q8, *key_q8, *value_q8, *output_q8;
    rx_q8_weight *ffn_key_q8, *ffn_value_q8;

#ifdef USE_METAL
    /* Persistent Metal buffers for weight tensors (created once at model load) */
    fm_buffer *mtl_receptance_w, *mtl_key_w, *mtl_value_w, *mtl_output_w;
    fm_buffer *mtl_ffn_key_w, *mtl_ffn_value_w;
#endif
} rx_attn_block;

/* ================================================================
 * Internal: Model structure
 * ================================================================ */

struct rx_model {
    rx_config config;

    /* Embedding + head */
    tm_tensor *emb_weight;      /* [vocab, n_embd] pre-normalized */
    tm_tensor *ln_out_w;        /* [n_embd] */
    tm_tensor *ln_out_b;        /* [n_embd] */
    tm_tensor *head_weight;     /* [n_embd, vocab] transposed */
    rx_q8_weight *head_q8;      /* INT8 quantized head (NULL when not quantized) */

    /* Blocks */
    rx_rwkv_block *rwkv_blocks; /* array of n_rwkv_layer */
    rx_attn_block *attn_blocks; /* array of n_attn_layer */

    /* Block execution order */
    int *block_order;           /* 0=RWKV, 1=ATTN per slot */
    int *block_index;           /* index within RWKV or ATTN arrays */
    int n_blocks;               /* total = n_rwkv + n_attn */

    /* Quantization flag */
    bool quantized;             /* true if INT8 quantized */

#ifdef USE_METAL
    fm_buffer *mtl_head_w;      /* Persistent Metal buffer for head weight */
#endif

#ifdef USE_THREADS
    fw_pool *pool;              /* pthread thread pool (NULL if threading disabled) */
#endif
};

/* ================================================================
 * Internal: State structure
 * ================================================================ */

typedef struct {
    tm_tensor *att_x_prev;  /* [C] */
    tm_tensor *att_kv;      /* [n_head * head_size * head_size] flat */
    tm_tensor *ffn_x_prev;  /* [C] */
} rx_rwkv_state;

typedef struct {
    tm_tensor *k_cache;     /* [max_cache * C] flat */
    tm_tensor *v_cache;     /* [max_cache * C] flat */
    int cache_len;
    int max_cache_len;
    tm_tensor *ffn_x_prev;  /* [C] */
    /* Incremental HDC projection accumulators (one per cache slot) */
    float *proj_accum;      /* [max_cache * hdc_dim] — R@k accumulated per slot */
    int proj_accum_dim;     /* hdc_dim (0 when disabled) */
} rx_attn_state;

typedef struct rx_workspace rx_workspace;  /* forward declaration */

struct rx_state {
    rx_rwkv_state *rwkv;    /* array of n_rwkv_layer */
    rx_attn_state *attn;    /* array of n_attn_layer */
    tm_tensor *v_first;     /* [C] */
    float *v_first_buf;     /* [C] raw float buffer for fast path */
    rx_config config;

    /* Scratch workspace for zero-allocation inference */
    rx_workspace *workspace;

    /* HybridMem (Tier 3 disk-backed memory) — NULL when disabled */
    fm_store *hm_store;
    int hm_top_k;           /* entries to retrieve per query (default 5) */
    float evict_fraction;   /* fraction of cache to evict (default 0.5) */

    /* Attention weight logging — NULL when disabled */
    FILE *attn_log;         /* CSV: token_step,layer,head,ret_mass,cache_mass,entropy */
    int attn_log_step;      /* current token step counter */

    /* Embedding capture — NULL when disabled */
    float *embed_capture;        /* buffer for [max_capture, C] embeddings */
    int embed_capture_count;     /* how many captured so far */
    int embed_capture_max;       /* buffer capacity */
    int embed_capture_layer;     /* which attention layer to capture from (-1 = all) */

    /* Pre-allocated workspace for retrieved K/V data (zero malloc in hot path) */
    float *ret_k_workspace;      /* [ret_workspace_cap * C] */
    float *ret_v_workspace;      /* [ret_workspace_cap * C] */
    int ret_workspace_cap;       /* max tokens that fit in workspace */
};

/* ================================================================
 * Internal: Tokenizer trie
 * ================================================================ */

#define RX_VOCAB_SIZE 65536
#define RX_MAX_TOKEN_LEN 128

typedef struct rx_trie_node {
    struct rx_trie_node *children[256];
    int32_t token_id;  /* -1 if not a terminal */
} rx_trie_node;

struct rx_tokenizer {
    rx_trie_node *root;
    uint8_t *token_bytes[RX_VOCAB_SIZE]; /* byte sequence per token */
    int token_lens[RX_VOCAB_SIZE];       /* length of each byte sequence */
};

/* ================================================================
 * Scratch workspace — pre-allocated buffers to avoid malloc per token
 *
 * All intermediate vectors in the forward pass are either [C] (n_embd)
 * or [lora_dim] (much smaller). We pre-allocate a set of scratch
 * buffers once and reuse them across all tokens.
 * ================================================================ */

#define RX_NUM_SCRATCH 16   /* number of [C]-sized scratch buffers (using 0-15) */

struct rx_workspace {
    float **buf;            /* array of RX_NUM_SCRATCH float[C] buffers */
    float *lora_buf;        /* single LoRA-sized buffer (max_lora_dim) */
    float *ffn_buf;         /* FFN intermediate buffer (ffn_dim, typically 3.5x C) */
    float *x_buf[2];       /* ping-pong buffers for block I/O in rx_forward */
    int8_t *q8_buf;        /* reusable INT8 quantization buffer [max(C, ffn_dim)] */
    int c;                  /* n_embd */
    int max_lora_dim;       /* largest LoRA intermediate dim */
    int ffn_dim;            /* largest FFN intermediate dim */
};

static rx_workspace* rx_workspace_new(int c, int max_lora_dim, int ffn_dim) {
    rx_workspace *ws = (rx_workspace*)calloc(1, sizeof(rx_workspace));
    ws->c = c;
    ws->max_lora_dim = max_lora_dim > 0 ? max_lora_dim : 256;
    ws->ffn_dim = ffn_dim > 0 ? ffn_dim : c * 4;
    ws->buf = (float**)malloc(RX_NUM_SCRATCH * sizeof(float*));
    for (int i = 0; i < RX_NUM_SCRATCH; i++)
        ws->buf[i] = (float*)aligned_malloc(c * sizeof(float));
    ws->lora_buf = (float*)aligned_malloc(ws->max_lora_dim * sizeof(float));
    ws->ffn_buf = (float*)aligned_malloc(ws->ffn_dim * sizeof(float));
    ws->x_buf[0] = (float*)aligned_malloc(c * sizeof(float));
    ws->x_buf[1] = (float*)aligned_malloc(c * sizeof(float));
    /* q8 buffer sized for largest vector that gets quantized (ffn_dim or c) */
    int q8_size = ws->ffn_dim > c ? ws->ffn_dim : c;
    ws->q8_buf = (int8_t*)aligned_malloc(q8_size * sizeof(int8_t));
    return ws;
}

static void rx_workspace_free(rx_workspace *ws) {
    if (!ws) return;
    for (int i = 0; i < RX_NUM_SCRATCH; i++)
        free(ws->buf[i]);
    free(ws->buf);
    free(ws->lora_buf);
    free(ws->ffn_buf);
    free(ws->x_buf[0]);
    free(ws->x_buf[1]);
    free(ws->q8_buf);
    free(ws);
}

/* ================================================================
 * Batch workspace — pre-allocated buffers for batched prefill
 *
 * All N tokens through one block at a time, converting N×SGEMV
 * into single SGEMM per linear projection.
 * ================================================================ */

typedef struct {
    float *x[2];        /* [N, C] ping-pong block I/O */
    float *ln;          /* [N, C] layer norm output */
    float *shifted;     /* [N, C] token shift results */
    float *proj[3];     /* [N, C] for R/K/V or Q/K/V projections */
    float *wkv_out;     /* [N, C] per-token output after WKV+gate or attn */
    float *v_first;     /* [N, C] per-token v_first (set during RWKV layer 0) */
    float *ffn_k;       /* [N, ffn_dim] FFN key intermediate */
    int8_t *q8_buf;     /* [max(c, ffn_dim)] INT8 quantization scratch */
    int n;              /* number of tokens */
    int c;              /* embedding dim */
    int ffn_dim;        /* FFN intermediate dim */
#ifdef USE_METAL
    /* Persistent Metal buffers wrapping workspace allocations (zero-copy) */
    fm_buffer *mtl_x[2], *mtl_ln, *mtl_shifted;
    fm_buffer *mtl_proj[3], *mtl_wkv_out, *mtl_ffn_k;
#endif
} rx_batch_workspace;

static rx_batch_workspace* rx_batch_workspace_new(int n, int c, int ffn_dim) {
    rx_batch_workspace *bws = (rx_batch_workspace*)calloc(1, sizeof(rx_batch_workspace));
    bws->n = n;
    bws->c = c;
    bws->ffn_dim = ffn_dim;

    size_t nc = (size_t)n * c * sizeof(float);
    size_t nf = (size_t)n * ffn_dim * sizeof(float);
    /* Page-aligned allocation enables zero-copy Metal buffer wrapping */
    bws->x[0]    = (float*)page_aligned_malloc(nc);
    bws->x[1]    = (float*)page_aligned_malloc(nc);
    bws->ln       = (float*)page_aligned_malloc(nc);
    bws->shifted   = (float*)page_aligned_malloc(nc);
    bws->proj[0]  = (float*)page_aligned_malloc(nc);
    bws->proj[1]  = (float*)page_aligned_malloc(nc);
    bws->proj[2]  = (float*)page_aligned_malloc(nc);
    bws->wkv_out  = (float*)page_aligned_malloc(nc);
    bws->v_first  = (float*)page_aligned_malloc(nc);
    bws->ffn_k    = (float*)page_aligned_malloc(nf);
    size_t q8_size = (size_t)(ffn_dim > c ? ffn_dim : c);
    bws->q8_buf   = (int8_t*)aligned_malloc(q8_size * sizeof(int8_t));

#ifdef USE_METAL
    if (fm_available()) {
        bws->mtl_x[0]   = fm_buffer_from_ptr(bws->x[0], nc);
        bws->mtl_x[1]   = fm_buffer_from_ptr(bws->x[1], nc);
        bws->mtl_ln      = fm_buffer_from_ptr(bws->ln, nc);
        bws->mtl_shifted  = fm_buffer_from_ptr(bws->shifted, nc);
        bws->mtl_proj[0] = fm_buffer_from_ptr(bws->proj[0], nc);
        bws->mtl_proj[1] = fm_buffer_from_ptr(bws->proj[1], nc);
        bws->mtl_proj[2] = fm_buffer_from_ptr(bws->proj[2], nc);
        bws->mtl_wkv_out = fm_buffer_from_ptr(bws->wkv_out, nc);
        bws->mtl_ffn_k   = fm_buffer_from_ptr(bws->ffn_k, nf);
    }
#endif

    return bws;
}

static void rx_batch_workspace_free(rx_batch_workspace *bws) {
    if (!bws) return;
#ifdef USE_METAL
    if (bws->mtl_x[0]) fm_buffer_free(bws->mtl_x[0]);
    if (bws->mtl_x[1]) fm_buffer_free(bws->mtl_x[1]);
    if (bws->mtl_ln) fm_buffer_free(bws->mtl_ln);
    if (bws->mtl_shifted) fm_buffer_free(bws->mtl_shifted);
    if (bws->mtl_proj[0]) fm_buffer_free(bws->mtl_proj[0]);
    if (bws->mtl_proj[1]) fm_buffer_free(bws->mtl_proj[1]);
    if (bws->mtl_proj[2]) fm_buffer_free(bws->mtl_proj[2]);
    if (bws->mtl_wkv_out) fm_buffer_free(bws->mtl_wkv_out);
    if (bws->mtl_ffn_k) fm_buffer_free(bws->mtl_ffn_k);
#endif
    free(bws->x[0]);
    free(bws->x[1]);
    free(bws->ln);
    free(bws->shifted);
    free(bws->proj[0]);
    free(bws->proj[1]);
    free(bws->proj[2]);
    free(bws->wkv_out);
    free(bws->v_first);
    free(bws->ffn_k);
    free(bws->q8_buf);
    free(bws);
}

/* ================================================================
 * Zero-allocation helpers — operate on raw float pointers
 * ================================================================ */

/* Matrix-vector: out[D] = x[C] @ w[C, D]  (w is row-major [C, D]) */
static void fast_matvec(const float *x, const float *w, float *out,
                        int rows, int cols) {
#ifdef USE_BLAS
    /* w is [rows, cols] row-major. We want x^T * W = (W^T * x)^T
     * which is sgemv with Trans on the [rows, cols] matrix. */
    cblas_sgemv(CblasRowMajor, CblasTrans,
                rows, cols,
                1.0f, w, cols, x, 1, 0.0f, out, 1);
#else
    /* Naive: out[j] = sum_i x[i] * w[i * cols + j] */
    memset(out, 0, cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
        float xi = x[i];
        const float *wi = &w[i * cols];
        for (int j = 0; j < cols; j++)
            out[j] += xi * wi[j];
    }
#endif
}

/* Batch matrix multiply: Out[N, D] = X[N, C] @ W[C, D]
 * W is row-major [C, D] (same layout as fast_matvec).
 * mtl_W: persistent Metal buffer for weight matrix (NULL to skip Metal).
 * mtl_X, mtl_out: persistent Metal buffers for input/output (NULL to create temp).
 * Non-BLAS fallback loops N individual SGEMV calls. */
static void fast_matmul_batch(const float *X, const float *W, float *out,
                              int N, int C, int D,
                              __attribute__((unused)) void *mtl_W,
                              __attribute__((unused)) void *mtl_X,
                              __attribute__((unused)) void *mtl_out) {
#ifdef USE_METAL
    if (fm_available() && mtl_W && (int64_t)N * C * D >= FM_GEMM_THRESHOLD) {
        fm_buffer *bx = mtl_X ? (fm_buffer*)mtl_X
                               : fm_buffer_from_ptr((void*)X, (size_t)N * C * sizeof(float));
        fm_buffer *bo = mtl_out ? (fm_buffer*)mtl_out
                                 : fm_buffer_from_ptr(out, (size_t)N * D * sizeof(float));
        if (bx && bo) {
            fm_sgemm(bx, (fm_buffer*)mtl_W, bo, N, D, C, 1.0f, 0.0f);
            float *gout = (float*)fm_buffer_data(bo);
            if (gout != out) memcpy(out, gout, (size_t)N * D * sizeof(float));
            if (!mtl_X) fm_buffer_free(bx);
            if (!mtl_out) fm_buffer_free(bo);
            return;
        }
        if (bx && !mtl_X) fm_buffer_free(bx);
        if (bo && !mtl_out) fm_buffer_free(bo);
        /* Fall through to CPU path */
    }
#endif
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, D, C,
                1.0f, X, C, W, D, 0.0f, out, D);
#else
    for (int t = 0; t < N; t++)
        fast_matvec(&X[t * C], W, &out[t * D], C, D);
#endif
}

/* Batch matrix multiply with INT8 quantized weights.
 * Out[N, D] = X[N, C] @ dequant(q->data[D, C], q->scales[D])^T
 * q->data is [D, C] int8 (transposed layout from original [C, D]).
 * CPU fallback: N individual q8 matvec calls.
 * ws_q8_buf: pre-allocated int8 buffer [C] for input quantization.
 * mtl_X, mtl_out: persistent Metal buffers for input/output (NULL to create temp). */
static void fast_matmul_batch_q8(const float *X, const rx_q8_weight *q,
                                  float *out, int N, int C, int D,
                                  int8_t *ws_q8_buf,
                                  __attribute__((unused)) void *mtl_X,
                                  __attribute__((unused)) void *mtl_out) {
#ifdef USE_METAL
    if (fm_available() && q->mtl_data && q->mtl_scales &&
        (int64_t)N * C * D >= FM_Q8_GEMM_THRESHOLD) {
        fm_buffer *bx = mtl_X ? (fm_buffer*)mtl_X
                               : fm_buffer_from_ptr((void*)X, (size_t)N * C * sizeof(float));
        fm_buffer *bo = mtl_out ? (fm_buffer*)mtl_out
                                 : fm_buffer_from_ptr(out, (size_t)N * D * sizeof(float));
        if (bx && bo) {
            fm_sgemm_q8(bx, q->mtl_data, q->mtl_scales, bo, N, D, C);
            float *gout = (float*)fm_buffer_data(bo);
            if (gout != out) memcpy(out, gout, (size_t)N * D * sizeof(float));
            if (!mtl_X) fm_buffer_free(bx);
            if (!mtl_out) fm_buffer_free(bo);
            return;
        }
        if (bx && !mtl_X) fm_buffer_free(bx);
        if (bo && !mtl_out) fm_buffer_free(bo);
        /* Fall through to CPU path */
    }
#endif
    /* CPU fallback: N individual quantized matvec calls */
    for (int t = 0; t < N; t++) {
        float x_scale = fast_quantize_input(&X[t * C], ws_q8_buf, C);
        fast_matvec_q8_preq(ws_q8_buf, x_scale, q, &out[t * D], D, C);
    }
}

/* Batch SGEMM dispatch: uses q8 path if quantized, f32 otherwise.
 * q8: rx_q8_weight* (NULL when not quantized), f32_w: tm_tensor* weight,
 * mtl: persistent Metal buffer for f32 weight (via MTL_W macro),
 * q8_buf: int8_t scratch buffer for input quantization.
 * mtl_in, mtl_out: persistent Metal buffers for input/output workspace. */
#define BATCH_SGEMM(input, q8, f32_w, output, N, C, D, mtl, q8_buf, mtl_in, mtl_out) do { \
    if (q8) \
        fast_matmul_batch_q8(input, q8, output, N, C, D, q8_buf, mtl_in, mtl_out); \
    else \
        fast_matmul_batch(input, tm_data_f32(f32_w), output, N, C, D, mtl, mtl_in, mtl_out); \
} while(0)

/* Layer norm in-place: out = (x - mean) / sqrt(var + eps) * weight + bias */
static void fast_layer_norm(const float *x, const float *w, const float *b,
                            float *out, int c) {
#ifdef __ARM_NEON
    /* NEON: process 4 floats at a time */
    int c4 = c & ~3;

    /* Pass 1: mean */
    float32x4_t sum_v = vdupq_n_f32(0.0f);
    for (int i = 0; i < c4; i += 4)
        sum_v = vaddq_f32(sum_v, vld1q_f32(&x[i]));
    float mean = vaddvq_f32(sum_v);
    for (int i = c4; i < c; i++) mean += x[i];
    mean /= c;

    /* Pass 2: variance */
    float32x4_t mean_v = vdupq_n_f32(mean);
    float32x4_t var_v = vdupq_n_f32(0.0f);
    for (int i = 0; i < c4; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(&x[i]), mean_v);
        var_v = vfmaq_f32(var_v, d, d);
    }
    float var = vaddvq_f32(var_v);
    for (int i = c4; i < c; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= c;

    /* Pass 3: normalize */
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    float32x4_t inv_std_v = vdupq_n_f32(inv_std);
    for (int i = 0; i < c4; i += 4) {
        float32x4_t xv = vsubq_f32(vld1q_f32(&x[i]), mean_v);
        float32x4_t normed = vmulq_f32(xv, inv_std_v);
        vst1q_f32(&out[i], vfmaq_f32(vld1q_f32(&b[i]),
                                       normed, vld1q_f32(&w[i])));
    }
    for (int i = c4; i < c; i++)
        out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
#else
    float mean = 0.0f;
    for (int i = 0; i < c; i++) mean += x[i];
    mean /= c;

    float var = 0.0f;
    for (int i = 0; i < c; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= c;

    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < c; i++)
        out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
#endif
}

/* Group norm: C elements in num_groups groups, eps=64e-5 per RWKV convention */
static void fast_group_norm(const float *x, int num_groups, int group_size,
                            const float *weight, const float *bias,
                            float *out) {
    for (int g = 0; g < num_groups; g++) {
        const float *gx = &x[g * group_size];
        float *go = &out[g * group_size];
        int off = g * group_size;

#ifdef __ARM_NEON
        int gs4 = group_size & ~3;

        float32x4_t sum_v = vdupq_n_f32(0.0f);
        for (int i = 0; i < gs4; i += 4)
            sum_v = vaddq_f32(sum_v, vld1q_f32(&gx[i]));
        float mean = vaddvq_f32(sum_v);
        for (int i = gs4; i < group_size; i++) mean += gx[i];
        mean /= group_size;

        float32x4_t mean_v = vdupq_n_f32(mean);
        float32x4_t var_v = vdupq_n_f32(0.0f);
        for (int i = 0; i < gs4; i += 4) {
            float32x4_t d = vsubq_f32(vld1q_f32(&gx[i]), mean_v);
            var_v = vfmaq_f32(var_v, d, d);
        }
        float var = vaddvq_f32(var_v);
        for (int i = gs4; i < group_size; i++) {
            float d = gx[i] - mean;
            var += d * d;
        }
        var /= group_size;

        float inv_std = 1.0f / sqrtf(var + 64e-5f);
        float32x4_t inv_std_v = vdupq_n_f32(inv_std);
        for (int i = 0; i < gs4; i += 4) {
            float32x4_t xv = vsubq_f32(vld1q_f32(&gx[i]), mean_v);
            float32x4_t normed = vmulq_f32(xv, inv_std_v);
            vst1q_f32(&go[i], vfmaq_f32(vld1q_f32(&bias[off + i]),
                                          normed, vld1q_f32(&weight[off + i])));
        }
        for (int i = gs4; i < group_size; i++)
            go[i] = (gx[i] - mean) * inv_std * weight[off + i] + bias[off + i];
#else
        float mean = 0.0f;
        for (int i = 0; i < group_size; i++) mean += gx[i];
        mean /= group_size;

        float var = 0.0f;
        for (int i = 0; i < group_size; i++) {
            float d = gx[i] - mean;
            var += d * d;
        }
        var /= group_size;

        float inv_std = 1.0f / sqrtf(var + 64e-5f);
        for (int i = 0; i < group_size; i++)
            go[i] = (gx[i] - mean) * inv_std * weight[off + i] + bias[off + i];
#endif
    }
}

/* L2 normalize per head: x[H*N] -> normalize each head row in place */
static void fast_l2_normalize_per_head(const float *x, float *out,
                                        int n_head, int head_size) {
    for (int h = 0; h < n_head; h++) {
        const float *hx = &x[h * head_size];
        float *ho = &out[h * head_size];
#ifdef __ARM_NEON
        int hs4 = head_size & ~3;
        float32x4_t sq_v = vdupq_n_f32(0.0f);
        for (int i = 0; i < hs4; i += 4) {
            float32x4_t v = vld1q_f32(&hx[i]);
            sq_v = vfmaq_f32(sq_v, v, v);
        }
        float sq_sum = vaddvq_f32(sq_v);
        for (int i = hs4; i < head_size; i++)
            sq_sum += hx[i] * hx[i];

        float inv_norm = 1.0f / (sqrtf(sq_sum) + 1e-12f);
        float32x4_t inv_v = vdupq_n_f32(inv_norm);
        for (int i = 0; i < hs4; i += 4)
            vst1q_f32(&ho[i], vmulq_f32(vld1q_f32(&hx[i]), inv_v));
        for (int i = hs4; i < head_size; i++)
            ho[i] = hx[i] * inv_norm;
#else
        float sq_sum = 0.0f;
        for (int i = 0; i < head_size; i++)
            sq_sum += hx[i] * hx[i];
        float inv_norm = 1.0f / (sqrtf(sq_sum) + 1e-12f);
        for (int i = 0; i < head_size; i++)
            ho[i] = hx[i] * inv_norm;
#endif
    }
}

/* Token shift + lerp: out = xx + (prev - xx) * lerp_coeff */
static void fast_token_shift(const float *xx, const float *prev,
                              const float *lerp, float *out, int c) {
#ifdef __ARM_NEON
    int c4 = c & ~3;
    for (int i = 0; i < c4; i += 4) {
        float32x4_t xx_v = vld1q_f32(&xx[i]);
        float32x4_t diff = vsubq_f32(vld1q_f32(&prev[i]), xx_v);
        vst1q_f32(&out[i], vfmaq_f32(xx_v, diff, vld1q_f32(&lerp[i])));
    }
    for (int i = c4; i < c; i++)
        out[i] = xx[i] + (prev[i] - xx[i]) * lerp[i];
#else
    for (int i = 0; i < c; i++)
        out[i] = xx[i] + (prev[i] - xx[i]) * lerp[i];
#endif
}


#ifdef USE_THREADS
/* Context for WKV-7 per-head worker */
typedef struct {
    const float *r_d, *w_d, *k_d, *v_d, *nkk_d, *kka_d;
    float *s_d, *y_out;
    int n_head, head_size;
} wkv7_ctx;

static void wkv7_head_worker(int start, int end, void *arg);
#endif

/* ================================================================
 * WKV-7 kernel (single timestep, zero-allocation)
 *
 * State evolution (Generalized Delta Rule):
 *   sa = sum_j(-kk[j] * state[h][i][j])
 *   state[h][i][j] = state[h][i][j]*w[j] + k[j]*v[i] + sa*kk_a[j]
 *   y[i] = sum_j(state[h][i][j] * r[j])
 *
 * Writes output into y_out[C]. State is modified in place.
 * ================================================================ */

static void rx_wkv7_one(
    const float *r_d, const float *w_d, const float *k_d, const float *v_d,
    const float *nkk_d, const float *kka_d,
    float *s_d,    /* [n_head * head_size * head_size] flat, modified in place */
    float *y_out,  /* [C] output buffer (pre-allocated) */
    int n_head, int head_size)
{
    int n = head_size;
    int C = n_head * head_size;

#ifdef USE_METAL
    if (fm_available() && C >= FM_WKV7_THRESHOLD) {
        fm_buffer *br   = fm_buffer_from_ptr((void*)r_d,   C * sizeof(float));
        fm_buffer *bw   = fm_buffer_from_ptr((void*)w_d,   C * sizeof(float));
        fm_buffer *bk   = fm_buffer_from_ptr((void*)k_d,   C * sizeof(float));
        fm_buffer *bv   = fm_buffer_from_ptr((void*)v_d,   C * sizeof(float));
        fm_buffer *bnkk = fm_buffer_from_ptr((void*)nkk_d, C * sizeof(float));
        fm_buffer *bkka = fm_buffer_from_ptr((void*)kka_d, C * sizeof(float));
        fm_buffer *bst  = fm_buffer_from_ptr(s_d, n_head * n * n * sizeof(float));
        fm_buffer *by   = fm_buffer_from_ptr(y_out, C * sizeof(float));
        fm_wkv7_one(br, bw, bk, bv, bnkk, bkka, bst, by, n_head, head_size);
        /* Copy back state and output if buffers were copies */
        float *gs = (float*)fm_buffer_data(bst);
        float *gy = (float*)fm_buffer_data(by);
        if (gs != s_d) memcpy(s_d, gs, n_head * n * n * sizeof(float));
        if (gy != y_out) memcpy(y_out, gy, C * sizeof(float));
        fm_buffer_free(br); fm_buffer_free(bw); fm_buffer_free(bk);
        fm_buffer_free(bv); fm_buffer_free(bnkk); fm_buffer_free(bkka);
        fm_buffer_free(bst); fm_buffer_free(by);
        return;
    }
#endif

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#elif defined(USE_THREADS)
    if (__rx_pool) {
        wkv7_ctx ctx = { r_d, w_d, k_d, v_d, nkk_d, kka_d,
                         s_d, y_out, n_head, head_size };
        fw_for(__rx_pool, n_head, wkv7_head_worker, &ctx);
        return;
    }
#endif
    for (int h = 0; h < n_head; h++) {
        int off = h * n;
        for (int i = 0; i < n; i++) {
            int sbase = (h * n + i) * n;

            /* Prefetch next state row */
            if (i + 1 < n)
                __builtin_prefetch(&s_d[(h * n + (i + 1)) * n], 1, 3);

#ifdef __ARM_NEON
            int n4 = n & ~3;

            /* sa = sum_j(-kk[j] * state[h][i][j]) */
            float32x4_t sa_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < n4; j += 4)
                sa_v = vfmaq_f32(sa_v, vld1q_f32(&nkk_d[off + j]),
                                       vld1q_f32(&s_d[sbase + j]));
            float sa = vaddvq_f32(sa_v);
            for (int j = n4; j < n; j++)
                sa += nkk_d[off + j] * s_d[sbase + j];

            float vv = v_d[off + i];
            float32x4_t vv_v = vdupq_n_f32(vv);
            float32x4_t sa_bcast = vdupq_n_f32(sa);
            float32x4_t yy_v = vdupq_n_f32(0.0f);

            /* state update + y accumulation */
            for (int j = 0; j < n4; j += 4) {
                float32x4_t s = vld1q_f32(&s_d[sbase + j]);
                float32x4_t w = vld1q_f32(&w_d[off + j]);
                float32x4_t k = vld1q_f32(&k_d[off + j]);
                float32x4_t kka = vld1q_f32(&kka_d[off + j]);
                /* s = s*w + k*vv + sa*kka */
                s = vfmaq_f32(vfmaq_f32(vmulq_f32(s, w), k, vv_v), kka, sa_bcast);
                vst1q_f32(&s_d[sbase + j], s);
                yy_v = vfmaq_f32(yy_v, s, vld1q_f32(&r_d[off + j]));
            }
            float yy = vaddvq_f32(yy_v);
            for (int j = n4; j < n; j++) {
                s_d[sbase + j] = s_d[sbase + j] * w_d[off + j]
                                + k_d[off + j] * vv
                                + sa * kka_d[off + j];
                yy += s_d[sbase + j] * r_d[off + j];
            }
#else
            float sa = 0.0f;
            for (int j = 0; j < n; j++)
                sa += nkk_d[off + j] * s_d[sbase + j];

            float vv = v_d[off + i];
            float yy = 0.0f;
            for (int j = 0; j < n; j++) {
                s_d[sbase + j] = s_d[sbase + j] * w_d[off + j]
                                + k_d[off + j] * vv
                                + sa * kka_d[off + j];
                yy += s_d[sbase + j] * r_d[off + j];
            }
#endif
            y_out[off + i] = yy;
        }
    }
}

#ifdef USE_THREADS
/* Worker callback for WKV-7 per-head parallelism — processes heads [start, end) */
static void wkv7_head_worker(int start, int end, void *arg) {
    wkv7_ctx *ctx = (wkv7_ctx *)arg;
    int n = ctx->head_size;
    for (int h = start; h < end; h++) {
        int off = h * n;
        for (int i = 0; i < n; i++) {
            int sbase = (h * n + i) * n;
            if (i + 1 < n)
                __builtin_prefetch(&ctx->s_d[(h * n + (i + 1)) * n], 1, 3);
#ifdef __ARM_NEON
            int n4 = n & ~3;
            float32x4_t sa_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < n4; j += 4)
                sa_v = vfmaq_f32(sa_v, vld1q_f32(&ctx->nkk_d[off + j]),
                                       vld1q_f32(&ctx->s_d[sbase + j]));
            float sa = vaddvq_f32(sa_v);
            for (int j = n4; j < n; j++)
                sa += ctx->nkk_d[off + j] * ctx->s_d[sbase + j];

            float vv = ctx->v_d[off + i];
            float32x4_t vv_v = vdupq_n_f32(vv);
            float32x4_t sa_bcast = vdupq_n_f32(sa);
            float32x4_t yy_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < n4; j += 4) {
                float32x4_t s = vld1q_f32(&ctx->s_d[sbase + j]);
                float32x4_t w = vld1q_f32(&ctx->w_d[off + j]);
                float32x4_t k = vld1q_f32(&ctx->k_d[off + j]);
                float32x4_t kka = vld1q_f32(&ctx->kka_d[off + j]);
                s = vfmaq_f32(vfmaq_f32(vmulq_f32(s, w), k, vv_v), kka, sa_bcast);
                vst1q_f32(&ctx->s_d[sbase + j], s);
                yy_v = vfmaq_f32(yy_v, s, vld1q_f32(&ctx->r_d[off + j]));
            }
            float yy = vaddvq_f32(yy_v);
            for (int j = n4; j < n; j++) {
                ctx->s_d[sbase + j] = ctx->s_d[sbase + j] * ctx->w_d[off + j]
                                + ctx->k_d[off + j] * vv
                                + sa * ctx->kka_d[off + j];
                yy += ctx->s_d[sbase + j] * ctx->r_d[off + j];
            }
#else
            float sa = 0.0f;
            for (int j = 0; j < n; j++)
                sa += ctx->nkk_d[off + j] * ctx->s_d[sbase + j];
            float vv = ctx->v_d[off + i];
            float yy = 0.0f;
            for (int j = 0; j < n; j++) {
                ctx->s_d[sbase + j] = ctx->s_d[sbase + j] * ctx->w_d[off + j]
                                + ctx->k_d[off + j] * vv
                                + sa * ctx->kka_d[off + j];
                yy += ctx->s_d[sbase + j] * ctx->r_d[off + j];
            }
#endif
            ctx->y_out[off + i] = yy;
        }
    }
}
#endif

/* ================================================================
 * RWKV-7 block forward (single token)
 * ================================================================ */

/* Time mixing for a single token (zero-allocation fast path)
 *
 * All intermediate results use scratch buffers from the workspace.
 * The final output is written into output_buf[C].
 */
static void rwkv7_time_mix_fast(
    rx_rwkv_block *blk, int layer_id,
    const float *xx, rx_rwkv_state *st, float *v_first, /* [C] in/out */
    int n_head, int head_size, rx_workspace *ws,
    float *output_buf)  /* [C] output */
{
    int c = n_head * head_size;
    float *prev = tm_data_f32(st->att_x_prev);

    /* Scratch aliases */
    float *shifted = ws->buf[0];
    float *r_vec   = ws->buf[1];
    float *k_vec   = ws->buf[2];
    float *v_vec   = ws->buf[3];
    float *w_vec   = ws->buf[4];
    float *a_vec   = ws->buf[5];
    float *g_vec   = ws->buf[6];
    float *kk      = ws->buf[7];
    float *k_mod   = ws->buf[8];
    float *v_final = ws->buf[9];
    float *y       = ws->buf[10];
    float *y_gn    = ws->buf[11];
    float *tmp     = ws->buf[12];
    float *tmp2    = ws->buf[13];
    float *xv_save = ws->buf[14];  /* save xv for v_first gating */
    float *lora    = ws->lora_buf;

    /* ---- Token shift + linear projections ----
     * All token shifts use the OLD prev, so we do them all before updating prev.
     *
     * Dispatch macro: use q8 path when available, f32 otherwise. */
    int8_t *q8_buf = ws->q8_buf;
    #define MATVEC(x_arg, w_tensor, q8_arg, out_arg, r_arg) do { \
        if (q8_arg) fast_matvec_q8((x_arg), (q8_arg), (out_arg), (q8_arg)->rows, (q8_arg)->cols, q8_buf); \
        else fast_matvec((x_arg), tm_data_f32(w_tensor), (out_arg), (r_arg), (int)tm_size((w_tensor), 1)); \
    } while(0)

    fast_token_shift(xx, prev, tm_data_f32(blk->x_r), shifted, c);
    MATVEC(shifted, blk->receptance_w, blk->receptance_q8, r_vec, c);

    fast_token_shift(xx, prev, tm_data_f32(blk->x_k), shifted, c);
    MATVEC(shifted, blk->key_w, blk->key_q8, k_vec, c);

    fast_token_shift(xx, prev, tm_data_f32(blk->x_v), shifted, c);
    /* Save shifted xv for v_first gating before it gets overwritten */
    memcpy(xv_save, shifted, c * sizeof(float));
    MATVEC(shifted, blk->value_w, blk->value_q8, v_vec, c);

    /* ---- Decay gate: w = exp(-0.606531 * sigmoid(w0 + tanh(xw @ w1) @ w2)) ---- */
    {
        int lora_dim = (int)tm_size(blk->w1, 1);
        float *w0_d = tm_data_f32(blk->w0);

        fast_token_shift(xx, prev, tm_data_f32(blk->x_w), shifted, c);
        fast_matvec(shifted, tm_data_f32(blk->w1), lora, c, lora_dim);

        /* tanh over lora_dim */
#if defined(USE_BLAS) && defined(__APPLE__)
        vvtanhf(lora, lora, &lora_dim);
#else
        for (int i = 0; i < lora_dim; i++)
            lora[i] = tanhf(lora[i]);
#endif

        fast_matvec(lora, tm_data_f32(blk->w2), w_vec, lora_dim, c);

        /* sigmoid(w0 + w_vec), then exp(-0.606531 * sig) */
#if defined(USE_BLAS) && defined(__APPLE__)
        /* w_vec = w0 + w_vec */
        vDSP_vadd(w0_d, 1, w_vec, 1, w_vec, 1, c);
        /* negate for sigmoid: w_vec = -w_vec */
        vDSP_vneg(w_vec, 1, w_vec, 1, c);
        /* exp(-x) */
        vvexpf(w_vec, w_vec, &c);
        /* w_vec = 1 + exp(-x) */
        float one = 1.0f;
        vDSP_vsadd(w_vec, 1, &one, w_vec, 1, c);
        /* w_vec = 1 / (1 + exp(-x)) = sigmoid */
        vvrecf(w_vec, w_vec, &c);
        /* w_vec = -0.606531 * sigmoid */
        float neg_coeff = -0.606531f;
        vDSP_vsmul(w_vec, 1, &neg_coeff, w_vec, 1, c);
        /* w_vec = exp(-0.606531 * sigmoid) */
        vvexpf(w_vec, w_vec, &c);
#else
        for (int i = 0; i < c; i++) {
            float sig = 1.0f / (1.0f + expf(-(w0_d[i] + w_vec[i])));
            w_vec[i] = expf(-0.606531f * sig);
        }
#endif
    }

    /* ---- Gate a: a = sigmoid(a0 + (xa @ a1) @ a2) ---- */
    {
        int lora_dim = (int)tm_size(blk->a1, 1);
        float *a0_d = tm_data_f32(blk->a0);

        fast_token_shift(xx, prev, tm_data_f32(blk->x_a), shifted, c);
        fast_matvec(shifted, tm_data_f32(blk->a1), lora, c, lora_dim);
        fast_matvec(lora, tm_data_f32(blk->a2), a_vec, lora_dim, c);

#if defined(USE_BLAS) && defined(__APPLE__)
        /* a_vec = sigmoid(a0 + a_vec) */
        vDSP_vadd(a0_d, 1, a_vec, 1, a_vec, 1, c);
        vDSP_vneg(a_vec, 1, a_vec, 1, c);
        vvexpf(a_vec, a_vec, &c);
        float one_a = 1.0f;
        vDSP_vsadd(a_vec, 1, &one_a, a_vec, 1, c);
        vvrecf(a_vec, a_vec, &c);
#else
        for (int i = 0; i < c; i++)
            a_vec[i] = 1.0f / (1.0f + expf(-(a0_d[i] + a_vec[i])));
#endif
    }

    /* ---- Gate g: g = sigmoid(xg @ g1) @ g2 ---- */
    {
        int lora_dim = (int)tm_size(blk->g1, 1);

        fast_token_shift(xx, prev, tm_data_f32(blk->x_g), shifted, c);
        fast_matvec(shifted, tm_data_f32(blk->g1), lora, c, lora_dim);

#if defined(USE_BLAS) && defined(__APPLE__)
        /* sigmoid over lora_dim */
        vDSP_vneg(lora, 1, lora, 1, lora_dim);
        vvexpf(lora, lora, &lora_dim);
        float one_g = 1.0f;
        vDSP_vsadd(lora, 1, &one_g, lora, 1, lora_dim);
        vvrecf(lora, lora, &lora_dim);
#else
        for (int i = 0; i < lora_dim; i++)
            lora[i] = 1.0f / (1.0f + expf(-lora[i]));
#endif

        fast_matvec(lora, tm_data_f32(blk->g2), g_vec, lora_dim, c);
    }

    /* Now safe to update prev for next token */
    memcpy(prev, xx, c * sizeof(float));

    /* ---- Key normalization: kk = L2_normalize(k * k_k), per head ---- */
    {
        float *kk_d = tm_data_f32(blk->k_k);
#ifdef __ARM_NEON
        int c4 = c & ~3;
        for (int i = 0; i < c4; i += 4)
            vst1q_f32(&tmp[i], vmulq_f32(vld1q_f32(&k_vec[i]),
                                           vld1q_f32(&kk_d[i])));
        for (int i = c4; i < c; i++)
            tmp[i] = k_vec[i] * kk_d[i];
#else
        for (int i = 0; i < c; i++)
            tmp[i] = k_vec[i] * kk_d[i];
#endif
        fast_l2_normalize_per_head(tmp, kk, n_head, head_size);
    }

    /* ---- Modify key: k_mod = k * (1 + (a-1) * k_a) ---- */
    {
        float *ka_d = tm_data_f32(blk->k_a);
#ifdef __ARM_NEON
        int c4 = c & ~3;
        float32x4_t one_v = vdupq_n_f32(1.0f);
        for (int i = 0; i < c4; i += 4) {
            float32x4_t am1 = vsubq_f32(vld1q_f32(&a_vec[i]), one_v);
            float32x4_t factor = vfmaq_f32(one_v, am1, vld1q_f32(&ka_d[i]));
            vst1q_f32(&k_mod[i], vmulq_f32(vld1q_f32(&k_vec[i]), factor));
        }
        for (int i = c4; i < c; i++)
            k_mod[i] = k_vec[i] * (1.0f + (a_vec[i] - 1.0f) * ka_d[i]);
#else
        for (int i = 0; i < c; i++)
            k_mod[i] = k_vec[i] * (1.0f + (a_vec[i] - 1.0f) * ka_d[i]);
#endif
    }

    /* ---- Value first-token gating ---- */
    if (layer_id == 0) {
        memcpy(v_final, v_vec, c * sizeof(float));
        memcpy(v_first, v_vec, c * sizeof(float));
    } else {
        /* v_blend = sigmoid(v0 + (xv @ v1) @ v2) */
        int lora_dim = (int)tm_size(blk->v1, 1);
        float *v0_d = tm_data_f32(blk->v0);

        /* Use saved xv_save (computed with old prev before update) */
        fast_matvec(xv_save, tm_data_f32(blk->v1), lora, c, lora_dim);
        fast_matvec(lora, tm_data_f32(blk->v2), tmp, lora_dim, c);

#if defined(USE_BLAS) && defined(__APPLE__)
        /* sigmoid(v0 + tmp) into tmp */
        vDSP_vadd(v0_d, 1, tmp, 1, tmp, 1, c);
        vDSP_vneg(tmp, 1, tmp, 1, c);
        vvexpf(tmp, tmp, &c);
        float one_v = 1.0f;
        vDSP_vsadd(tmp, 1, &one_v, tmp, 1, c);
        vvrecf(tmp, tmp, &c);
        /* v_final = v_vec + (v_first - v_vec) * blend */
  #ifdef __ARM_NEON
        int c4 = c & ~3;
        for (int i = 0; i < c4; i += 4) {
            float32x4_t vv = vld1q_f32(&v_vec[i]);
            float32x4_t diff = vsubq_f32(vld1q_f32(&v_first[i]), vv);
            vst1q_f32(&v_final[i], vfmaq_f32(vv, diff, vld1q_f32(&tmp[i])));
        }
        for (int i = c4; i < c; i++)
            v_final[i] = v_vec[i] + (v_first[i] - v_vec[i]) * tmp[i];
  #else
        for (int i = 0; i < c; i++)
            v_final[i] = v_vec[i] + (v_first[i] - v_vec[i]) * tmp[i];
  #endif
#else
        for (int i = 0; i < c; i++) {
            float blend = 1.0f / (1.0f + expf(-(v0_d[i] + tmp[i])));
            v_final[i] = v_vec[i] + (v_first[i] - v_vec[i]) * blend;
        }
#endif
    }

    /* ---- WKV-7 kernel ---- */
    {
#ifdef __ARM_NEON
        int c4 = c & ~3;
        for (int i = 0; i < c4; i += 4) {
            float32x4_t kk_v = vld1q_f32(&kk[i]);
            vst1q_f32(&tmp[i], vnegq_f32(kk_v));
            vst1q_f32(&tmp2[i], vmulq_f32(kk_v, vld1q_f32(&a_vec[i])));
        }
        for (int i = c4; i < c; i++) {
            tmp[i] = -kk[i];
            tmp2[i] = kk[i] * a_vec[i];
        }
#else
        for (int i = 0; i < c; i++) {
            tmp[i] = -kk[i];
            tmp2[i] = kk[i] * a_vec[i];
        }
#endif
        rx_wkv7_one(r_vec, w_vec, k_mod, v_final,
                     tmp, tmp2, tm_data_f32(st->att_kv), y,
                     n_head, head_size);
    }

    /* ---- Group norm ---- */
    fast_group_norm(y, n_head, head_size,
                    tm_data_f32(blk->ln_x_w), tm_data_f32(blk->ln_x_b),
                    y_gn);

    /* ---- Bonus: per-head (r * k_mod * r_k).sum() * v_final ---- */
    {
        float *rk_d = tm_data_f32(blk->r_k);
        for (int h = 0; h < n_head; h++) {
            int off = h * head_size;
#ifdef __ARM_NEON
            int hs4 = head_size & ~3;
            float32x4_t sum_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < hs4; j += 4) {
                float32x4_t rv = vld1q_f32(&r_vec[off + j]);
                float32x4_t km = vld1q_f32(&k_mod[off + j]);
                float32x4_t rk = vld1q_f32(&rk_d[off + j]);
                sum_v = vfmaq_f32(sum_v, vmulq_f32(rv, km), rk);
            }
            float sum = vaddvq_f32(sum_v);
            for (int j = hs4; j < head_size; j++)
                sum += r_vec[off + j] * k_mod[off + j] * rk_d[off + j];

            float32x4_t sum_bcast = vdupq_n_f32(sum);
            for (int j = 0; j < hs4; j += 4) {
                float32x4_t cur = vld1q_f32(&y_gn[off + j]);
                vst1q_f32(&y_gn[off + j],
                           vfmaq_f32(cur, sum_bcast, vld1q_f32(&v_final[off + j])));
            }
            for (int j = hs4; j < head_size; j++)
                y_gn[off + j] += sum * v_final[off + j];
#else
            float sum = 0.0f;
            for (int j = 0; j < head_size; j++)
                sum += r_vec[off + j] * k_mod[off + j] * rk_d[off + j];
            for (int j = 0; j < head_size; j++)
                y_gn[off + j] += sum * v_final[off + j];
#endif
        }
    }

    /* ---- Output: (y_bonus * g) @ output_w ---- */
    {
#ifdef __ARM_NEON
        int c4 = c & ~3;
        for (int i = 0; i < c4; i += 4)
            vst1q_f32(&tmp[i], vmulq_f32(vld1q_f32(&y_gn[i]),
                                           vld1q_f32(&g_vec[i])));
        for (int i = c4; i < c; i++)
            tmp[i] = y_gn[i] * g_vec[i];
#else
        for (int i = 0; i < c; i++)
            tmp[i] = y_gn[i] * g_vec[i];
#endif
        MATVEC(tmp, blk->output_w, blk->output_q8, output_buf, c);
    }

    #undef MATVEC
}

/* Channel mix (ReLU^2 FFN) for a single token (zero-allocation fast path) */
static void rwkv7_channel_mix_fast(
    rx_rwkv_block *blk, const float *xx, rx_rwkv_state *st,
    rx_workspace *ws, float *output_buf)  /* [C] output */
{
    int c = ws->c;
    float *prev = tm_data_f32(st->ffn_x_prev);
    float *shifted = ws->buf[0];
    int ffn_dim = blk->ffn_key_q8 ? blk->ffn_key_q8->rows
                                  : (int)tm_size(blk->ffn_key_w, 1);

    /* Token shift + lerp */
    fast_token_shift(xx, prev, tm_data_f32(blk->ffn_x_k), shifted, c);

    /* Update prev for next token */
    memcpy(prev, xx, c * sizeof(float));

    /* k = relu(xk @ K)^2, then output = k @ V
     * ffn_key_w is [C, ffn_dim], ffn_value_w is [ffn_dim, C] */
    float *fk_buf = ws->ffn_buf;
    int8_t *q8_buf = ws->q8_buf;

    if (blk->ffn_key_q8)
        fast_matvec_q8(shifted, blk->ffn_key_q8, fk_buf,
                       blk->ffn_key_q8->rows, blk->ffn_key_q8->cols, q8_buf);
    else
        fast_matvec(shifted, tm_data_f32(blk->ffn_key_w), fk_buf, c, ffn_dim);

    /* relu^2 in place */
#ifdef __ARM_NEON
    {
        int fd4 = ffn_dim & ~3;
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (int i = 0; i < fd4; i += 4) {
            float32x4_t v = vmaxq_f32(vld1q_f32(&fk_buf[i]), zero);
            vst1q_f32(&fk_buf[i], vmulq_f32(v, v));
        }
        for (int i = fd4; i < ffn_dim; i++) {
            float v = fk_buf[i] > 0.0f ? fk_buf[i] : 0.0f;
            fk_buf[i] = v * v;
        }
    }
#else
    for (int i = 0; i < ffn_dim; i++) {
        float v = fk_buf[i] > 0.0f ? fk_buf[i] : 0.0f;
        fk_buf[i] = v * v;
    }
#endif

    if (blk->ffn_value_q8)
        fast_matvec_q8(fk_buf, blk->ffn_value_q8, output_buf,
                       blk->ffn_value_q8->rows, blk->ffn_value_q8->cols, q8_buf);
    else
        fast_matvec(fk_buf, tm_data_f32(blk->ffn_value_w), output_buf, ffn_dim, c);
}

/* NEON residual add helper: out[i] += src[i] */
static inline void fast_residual_add(float *out, const float *src, int c) {
#ifdef __ARM_NEON
    int c4 = c & ~3;
    for (int i = 0; i < c4; i += 4)
        vst1q_f32(&out[i], vaddq_f32(vld1q_f32(&out[i]), vld1q_f32(&src[i])));
    for (int i = c4; i < c; i++)
        out[i] += src[i];
#else
    for (int i = 0; i < c; i++)
        out[i] += src[i];
#endif
}

/* Full RWKV-7 block forward (zero-allocation fast path)
 * x_data[C] is input, x_out[C] is output (can be the same buffer). */
static void rwkv7_block_forward_fast(
    rx_rwkv_block *blk, int layer_id,
    float *x_data, rx_rwkv_state *st, float *v_first,
    int n_head, int head_size, rx_workspace *ws,
    float *x_out)
{
    int c = n_head * head_size;
    float *ln_out = ws->buf[15];  /* use buf[15] for layer norm output */

    /* Pre-attention LayerNorm */
    fast_layer_norm(x_data, tm_data_f32(blk->ln1_w), tm_data_f32(blk->ln1_b),
                    ln_out, c);

    /* Time mixing writes output to x_out. Then we add the residual.
     * time_mix uses buf[0..14] internally, buf[15] (ln_out) is read
     * early and not touched by time_mix. */
    rwkv7_time_mix_fast(blk, layer_id, ln_out, st, v_first,
                         n_head, head_size, ws, x_out);

    /* x1 = x + time_mix_output (residual connection) */
    fast_residual_add(x_out, x_data, c);

    /* Pre-FFN LayerNorm */
    fast_layer_norm(x_out, tm_data_f32(blk->ln2_w), tm_data_f32(blk->ln2_b),
                    ln_out, c);

    /* Channel mix -> writes to a temp, then add residual */
    float *cmix_out = ws->buf[0]; /* safe: channel_mix only uses buf[0] + ffn_buf */
    rwkv7_channel_mix_fast(blk, ln_out, st, ws, cmix_out);

    /* x2 = x1 + channel_mix_output */
    fast_residual_add(x_out, cmix_out, c);
}

/* ================================================================
 * Batched RWKV-7 block forward — process N tokens through one block
 *
 * Converts 6×N SGEMV into 6 SGEMM (R, K, V, output, FFN key, FFN value).
 * WKV-7 state evolution + LoRA paths remain sequential per token.
 * ================================================================ */

static void rwkv7_block_forward_batch(
    rx_rwkv_block *blk, int layer_id,
    const float *x_batch,   /* [N, C] input */
    rx_rwkv_state *st,
    float *v_first_batch,   /* [N, C] per-token v_first buffer */
    int n_head, int head_size,
    rx_workspace *ws,       /* per-token scratch for WKV loop */
    rx_batch_workspace *bws,
    float *x_out,           /* [N, C] output */
    __attribute__((unused)) void *mtl_x_out)  /* persistent Metal buf for x_out */
{
    int c = n_head * head_size;
    int N = bws->n;
    float *ln     = bws->ln;       /* [N, C] */
    float *shifted = bws->shifted; /* [N, C] */
    float *R_batch = bws->proj[0]; /* [N, C] */
    float *K_batch = bws->proj[1]; /* [N, C] */
    float *V_batch = bws->proj[2]; /* [N, C] */
    float *wkv_out = bws->wkv_out; /* [N, C] */

    /* ---- Phase A: Batch LayerNorm ---- */
    float *ln1_w = tm_data_f32(blk->ln1_w);
    float *ln1_b = tm_data_f32(blk->ln1_b);
    for (int t = 0; t < N; t++)
        fast_layer_norm(&x_batch[t * c], ln1_w, ln1_b, &ln[t * c], c);

    /* ---- Phase A: Token shifts + SGEMM projections ---- */
    float *att_prev = tm_data_f32(st->att_x_prev);
    float *x_r_d = tm_data_f32(blk->x_r);
    float *x_k_d = tm_data_f32(blk->x_k);
    float *x_v_d = tm_data_f32(blk->x_v);

    /* Save original att_prev for Phase B's sequential LoRA paths (shifts for
     * decay/gate-a/gate-g at t=0 need the original prev, not ln[N-1]). */
    float att_prev_save[c];
    memcpy(att_prev_save, att_prev, c * sizeof(float));

    /* R/K/V shifts + SGEMM projections.
     * Pre-compute all 3 shifts into SEPARATE buffers to allow GPU to process
     * all 3 SGEMMs in one command buffer (MLX-style auto-batching).
     * R shift → wkv_out, K shift → first N*c of ffn_k, V shift → shifted.
     * V stays in shifted because xv_batch aliases it for Phase B. */
    float *xv_batch = shifted; /* alias: shifted IS xv_batch */
    for (int t = 0; t < N; t++) {
        const float *prev = (t == 0) ? att_prev : &ln[(t - 1) * c];
        fast_token_shift(&ln[t * c], prev, x_r_d, &wkv_out[t * c], c);
        fast_token_shift(&ln[t * c], prev, x_k_d, &bws->ffn_k[t * c], c);
        fast_token_shift(&ln[t * c], prev, x_v_d, &shifted[t * c], c);
    }
    /* All 3 SGEMMs use different input buffers → auto-batch together */
    BATCH_SGEMM(wkv_out, blk->receptance_q8, blk->receptance_w,
                R_batch, N, c, c, MTL_W(blk->mtl_receptance_w), bws->q8_buf,
                MTL_BUF(bws->mtl_wkv_out), MTL_BUF(bws->mtl_proj[0]));
    BATCH_SGEMM(bws->ffn_k, blk->key_q8, blk->key_w,
                K_batch, N, c, c, MTL_W(blk->mtl_key_w), bws->q8_buf,
                MTL_BUF(bws->mtl_ffn_k), MTL_BUF(bws->mtl_proj[1]));
    BATCH_SGEMM(shifted, blk->value_q8, blk->value_w,
                V_batch, N, c, c, MTL_W(blk->mtl_value_w), bws->q8_buf,
                MTL_BUF(bws->mtl_shifted), MTL_BUF(bws->mtl_proj[2]));
    /* xv_batch[t*c] still holds token-shifted xv values (V shift in shifted) */
#ifdef USE_METAL
    fm_sync(); /* CPU needs R/K/V projections for WKV-7 */
#endif

    /* Update att_x_prev state: last token's LN output */
    memcpy(att_prev, &ln[(N - 1) * c], c * sizeof(float));

    /* ---- Phase B: Sequential WKV loop ---- */
    /* Per-token scratch aliases from workspace */
    float *shifted_t = ws->buf[0];
    float *w_vec   = ws->buf[4];
    float *a_vec   = ws->buf[5];
    float *g_vec   = ws->buf[6];
    float *kk      = ws->buf[7];
    float *k_mod   = ws->buf[8];
    float *v_final = ws->buf[9];
    float *y       = ws->buf[10];
    float *y_gn    = ws->buf[11];
    float *tmp     = ws->buf[12];
    float *tmp2    = ws->buf[13];
    float *lora    = ws->lora_buf;

    float *x_w_d = tm_data_f32(blk->x_w);
    float *x_a_d = tm_data_f32(blk->x_a);
    float *x_g_d = tm_data_f32(blk->x_g);
    float *w0_d  = tm_data_f32(blk->w0);
    float *a0_d  = tm_data_f32(blk->a0);
    float *kk_d  = tm_data_f32(blk->k_k);
    float *ka_d  = tm_data_f32(blk->k_a);
    float *rk_d  = tm_data_f32(blk->r_k);

    int w_lora_dim = (int)tm_size(blk->w1, 1);
    int a_lora_dim = (int)tm_size(blk->a1, 1);
    int g_lora_dim = (int)tm_size(blk->g1, 1);

    for (int t = 0; t < N; t++) {
        float *r_t = &R_batch[t * c];
        float *k_t = &K_batch[t * c];
        float *v_t = &V_batch[t * c];
        const float *ln_t = &ln[t * c];
        const float *ln_prev = (t == 0) ? att_prev_save : &ln[(t - 1) * c];

        /* ---- Decay gate: w = exp(-0.606531 * sigmoid(w0 + tanh(xw @ w1) @ w2)) ---- */
        fast_token_shift(ln_t, ln_prev, x_w_d, shifted_t, c);
        fast_matvec(shifted_t, tm_data_f32(blk->w1), lora, c, w_lora_dim);
        for (int i = 0; i < w_lora_dim; i++)
            lora[i] = tanhf(lora[i]);
        fast_matvec(lora, tm_data_f32(blk->w2), w_vec, w_lora_dim, c);
        for (int i = 0; i < c; i++) {
            float sig = 1.0f / (1.0f + expf(-(w0_d[i] + w_vec[i])));
            w_vec[i] = expf(-0.606531f * sig);
        }

        /* ---- Gate a: a = sigmoid(a0 + (xa @ a1) @ a2) ---- */
        fast_token_shift(ln_t, ln_prev, x_a_d, shifted_t, c);
        fast_matvec(shifted_t, tm_data_f32(blk->a1), lora, c, a_lora_dim);
        fast_matvec(lora, tm_data_f32(blk->a2), a_vec, a_lora_dim, c);
        for (int i = 0; i < c; i++)
            a_vec[i] = 1.0f / (1.0f + expf(-(a0_d[i] + a_vec[i])));

        /* ---- Gate g: g = sigmoid(xg @ g1) @ g2 ---- */
        fast_token_shift(ln_t, ln_prev, x_g_d, shifted_t, c);
        fast_matvec(shifted_t, tm_data_f32(blk->g1), lora, c, g_lora_dim);
        for (int i = 0; i < g_lora_dim; i++)
            lora[i] = 1.0f / (1.0f + expf(-lora[i]));
        fast_matvec(lora, tm_data_f32(blk->g2), g_vec, g_lora_dim, c);

        /* ---- Key normalization: kk = L2_normalize(k * k_k), per head ---- */
        for (int i = 0; i < c; i++)
            tmp[i] = k_t[i] * kk_d[i];
        fast_l2_normalize_per_head(tmp, kk, n_head, head_size);

        /* ---- Modify key: k_mod = k * (1 + (a-1) * k_a) ---- */
        for (int i = 0; i < c; i++)
            k_mod[i] = k_t[i] * (1.0f + (a_vec[i] - 1.0f) * ka_d[i]);

        /* ---- Value first-token gating ---- */
        if (layer_id == 0) {
            memcpy(v_final, v_t, c * sizeof(float));
            memcpy(&v_first_batch[t * c], v_t, c * sizeof(float));
        } else {
            int v_lora_dim = (int)tm_size(blk->v1, 1);
            float *v0_d = tm_data_f32(blk->v0);
            /* Use xv_batch[t] — the token-shifted xv computed with old prev */
            fast_matvec(&xv_batch[t * c], tm_data_f32(blk->v1), lora, c, v_lora_dim);
            fast_matvec(lora, tm_data_f32(blk->v2), tmp, v_lora_dim, c);
            for (int i = 0; i < c; i++) {
                float blend = 1.0f / (1.0f + expf(-(v0_d[i] + tmp[i])));
                v_final[i] = v_t[i] + (v_first_batch[t * c + i] - v_t[i]) * blend;
            }
        }

        /* ---- WKV-7 kernel ---- */
        for (int i = 0; i < c; i++) {
            tmp[i] = -kk[i];
            tmp2[i] = kk[i] * a_vec[i];
        }
        rx_wkv7_one(r_t, w_vec, k_mod, v_final,
                     tmp, tmp2, tm_data_f32(st->att_kv), y,
                     n_head, head_size);

        /* ---- Group norm ---- */
        fast_group_norm(y, n_head, head_size,
                        tm_data_f32(blk->ln_x_w), tm_data_f32(blk->ln_x_b),
                        y_gn);

        /* ---- Bonus: per-head (r * k_mod * r_k).sum() * v_final ---- */
        for (int h = 0; h < n_head; h++) {
            int off = h * head_size;
            float sum = 0.0f;
            for (int j = 0; j < head_size; j++)
                sum += r_t[off + j] * k_mod[off + j] * rk_d[off + j];
            for (int j = 0; j < head_size; j++)
                y_gn[off + j] += sum * v_final[off + j];
        }

        /* ---- Output: (y_bonus * g) -> wkv_out[t] ---- */
        for (int i = 0; i < c; i++)
            wkv_out[t * c + i] = y_gn[i] * g_vec[i];
    }

    /* ---- Phase C: Batch output projection + residual ---- */
    BATCH_SGEMM(wkv_out, blk->output_q8, blk->output_w,
                x_out, N, c, c, MTL_W(blk->mtl_output_w), bws->q8_buf,
                MTL_BUF(bws->mtl_wkv_out), mtl_x_out);
#ifdef USE_METAL
    fm_sync(); /* CPU needs output projection for residual add */
#endif
    for (int t = 0; t < N; t++)
        fast_residual_add(&x_out[t * c], &x_batch[t * c], c);

    /* ---- Phase D: Batch FFN ---- */
    float *ln2_w = tm_data_f32(blk->ln2_w);
    float *ln2_b = tm_data_f32(blk->ln2_b);
    /* Get ffn_dim from whichever weight source exists */
    int ffn_dim = blk->ffn_key_q8 ? blk->ffn_key_q8->rows : (int)tm_size(blk->ffn_key_w, 1);
    float *ffn_x_k_d = tm_data_f32(blk->ffn_x_k);
    float *ffn_prev = tm_data_f32(st->ffn_x_prev);

    /* Batch pre-FFN LayerNorm */
    for (int t = 0; t < N; t++)
        fast_layer_norm(&x_out[t * c], ln2_w, ln2_b, &ln[t * c], c);

    /* FFN token shifts */
    for (int t = 0; t < N; t++) {
        const float *prev = (t == 0) ? ffn_prev : &ln[(t - 1) * c];
        fast_token_shift(&ln[t * c], prev, ffn_x_k_d, &shifted[t * c], c);
    }
    /* Update ffn_x_prev state */
    memcpy(ffn_prev, &ln[(N - 1) * c], c * sizeof(float));

    /* SGEMM: shifted[N, C] @ ffn_key_w[C, ffn_dim] -> ffn_k[N, ffn_dim] */
    float *ffn_k = bws->ffn_k;
    BATCH_SGEMM(shifted, blk->ffn_key_q8, blk->ffn_key_w,
                ffn_k, N, c, ffn_dim, MTL_W(blk->mtl_ffn_key_w), bws->q8_buf,
                MTL_BUF(bws->mtl_shifted), MTL_BUF(bws->mtl_ffn_k));
#ifdef USE_METAL
    fm_sync(); /* CPU needs ffn_k for ReLU² */
#endif

    /* Batch ReLU² */
    {
        int total = N * ffn_dim;
#ifdef __ARM_NEON
        int t4 = total & ~3;
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (int i = 0; i < t4; i += 4) {
            float32x4_t v = vmaxq_f32(vld1q_f32(&ffn_k[i]), zero);
            vst1q_f32(&ffn_k[i], vmulq_f32(v, v));
        }
        for (int i = t4; i < total; i++) {
            float v = ffn_k[i] > 0.0f ? ffn_k[i] : 0.0f;
            ffn_k[i] = v * v;
        }
#else
        for (int i = 0; i < total; i++) {
            float v = ffn_k[i] > 0.0f ? ffn_k[i] : 0.0f;
            ffn_k[i] = v * v;
        }
#endif
    }

    /* SGEMM: ffn_k[N, ffn_dim] @ ffn_value_w[ffn_dim, C] -> temp[N, C] */
    /* Reuse wkv_out as temp buffer for FFN output */
    BATCH_SGEMM(ffn_k, blk->ffn_value_q8, blk->ffn_value_w,
                wkv_out, N, ffn_dim, c, MTL_W(blk->mtl_ffn_value_w), bws->q8_buf,
                MTL_BUF(bws->mtl_ffn_k), MTL_BUF(bws->mtl_wkv_out));
#ifdef USE_METAL
    fm_sync(); /* CPU needs ffn_value output for residual add */
#endif

    /* Batch residual: x_out += ffn_out */
    for (int t = 0; t < N; t++)
        fast_residual_add(&x_out[t * c], &wkv_out[t * c], c);
}

#ifdef USE_THREADS
/* Context for attention per-head worker */
typedef struct {
    const float *q_vec, *k_cache, *v_cache;
    const float *ret_k_data, *ret_v_data;
    float *y_data;
    int n_head, head_size, c, ct, n_retrieved_tokens, total;
    float scale;
    float *attn_log_buf;    /* [n_head * 3] — per-head cache_mass, ret_mass, entropy (NULL=disabled) */
} attn_head_ctx;

static void attn_head_worker(int start, int end, void *arg);
#endif

/* ================================================================
 * Sparse attention block forward (single token, decode mode)
 * ================================================================ */

/* Sparse attention block forward (zero-allocation fast path)
 * x_data[C] is input, x_out[C] is output. */
static void attn_block_forward_fast(
    rx_attn_block *blk, float *x_data, rx_attn_state *st,
    fm_store *hm_store, int layer_idx,
    int hm_top_k, float evict_fraction,
    int n_head, int head_size, rx_workspace *ws,
    float *x_out,
    FILE *attn_log, int attn_log_step,
    float *embed_capture, int *embed_capture_count,
    int embed_capture_max, int embed_capture_layer,
    float *ret_k_ws, float *ret_v_ws, int ret_ws_cap)
{
    int c = n_head * head_size;
    float *ln_out  = ws->buf[15];
    float *q_vec   = ws->buf[1];
    float *k_vec   = ws->buf[2];
    float *v_vec   = ws->buf[3];
    float *y_data  = ws->buf[10];
    float *att_out = ws->buf[11];

    /* Pre-attention LayerNorm */
    fast_layer_norm(x_data, tm_data_f32(blk->ln1_w), tm_data_f32(blk->ln1_b),
                    ln_out, c);

    /* Embedding capture: copy ln_out for HDC correlation test */
    if (embed_capture && embed_capture_count &&
        *embed_capture_count < embed_capture_max &&
        (embed_capture_layer < 0 || embed_capture_layer == layer_idx)) {
        memcpy(&embed_capture[*embed_capture_count * c], ln_out, c * sizeof(float));
        (*embed_capture_count)++;
    }

    /* Q/K/V projections (dispatch to q8 if quantized)
     * All 3 share the same input (ln_out), so quantize once and reuse. */
    int8_t *q8_buf = ws->q8_buf;
    if (blk->receptance_q8) {
        /* Quantize ln_out once, reuse for K and V */
        float ln_q8_scale = fast_quantize_input(ln_out, q8_buf, c);
        fast_matvec_q8_preq(q8_buf, ln_q8_scale, blk->receptance_q8, q_vec,
                            blk->receptance_q8->rows, blk->receptance_q8->cols);
        fast_matvec_q8_preq(q8_buf, ln_q8_scale, blk->key_q8, k_vec,
                            blk->key_q8->rows, blk->key_q8->cols);
        fast_matvec_q8_preq(q8_buf, ln_q8_scale, blk->value_q8, v_vec,
                            blk->value_q8->rows, blk->value_q8->cols);
    } else {
        fast_matvec(ln_out, tm_data_f32(blk->receptance_w), q_vec, c, c);
        fast_matvec(ln_out, tm_data_f32(blk->key_w), k_vec, c, c);
        fast_matvec(ln_out, tm_data_f32(blk->value_w), v_vec, c, c);
    }

    /* Append K/V to cache — evict to disk if full */
    float *k_cache = tm_data_f32(st->k_cache);
    float *v_cache = tm_data_f32(st->v_cache);

    if (st->cache_len >= st->max_cache_len) {
        int evict_count = (int)(st->max_cache_len * evict_fraction);
        if (evict_count < 1) evict_count = 1;
        if (evict_count > st->cache_len) evict_count = st->cache_len;

        if (hm_store) {
            int64_t ev_shape[] = {evict_count, c};
            tm_tensor *ev_k = tm_from_data(k_cache, ev_shape, 2, TM_F32);
            tm_tensor *ev_v = tm_from_data(v_cache, ev_shape, 2, TM_F32);

            /* Use pre-computed incremental HDC signature if available */
            if (st->proj_accum && st->proj_accum_dim > 0) {
                int hd = st->proj_accum_dim;
                float *sum_accum = (float*)calloc(hd, sizeof(float));
                for (int ei = 0; ei < evict_count; ei++) {
                    float *slot = &st->proj_accum[ei * hd];
                    for (int j = 0; j < hd; j++) sum_accum[j] += slot[j];
                }
                fv_vector *sig = fv_project_threshold(
                    fm_store_projector(hm_store), sum_accum);
                free(sum_accum);
                fm_put_presig(hm_store, ev_k, ev_v, layer_idx, 1.0f, sig);
            } else {
                fm_put(hm_store, ev_k, ev_v, layer_idx, 1.0f);
            }
            tm_free(ev_k); tm_free(ev_v);
        }

        int remain = st->cache_len - evict_count;
        if (remain > 0) {
            memmove(k_cache, &k_cache[evict_count * c], remain * c * sizeof(float));
            memmove(v_cache, &v_cache[evict_count * c], remain * c * sizeof(float));
            /* Shift projection accumulators too */
            if (st->proj_accum && st->proj_accum_dim > 0) {
                int hd = st->proj_accum_dim;
                memmove(st->proj_accum, &st->proj_accum[evict_count * hd],
                        (size_t)remain * hd * sizeof(float));
            }
        }
        st->cache_len = remain;
    }

    memcpy(&k_cache[st->cache_len * c], k_vec, c * sizeof(float));
    memcpy(&v_cache[st->cache_len * c], v_vec, c * sizeof(float));

    /* Incrementally project this token's K into its cache slot accumulator */
    if (st->proj_accum && st->proj_accum_dim > 0) {
        int hd = st->proj_accum_dim;
        float *slot = &st->proj_accum[st->cache_len * hd];
        memset(slot, 0, hd * sizeof(float));
        fv_project_accum(fm_store_projector(hm_store), k_vec, slot);
    }

    st->cache_len++;

    /* Retrieve from disk (HybridMem) — this part still uses tensor API
     * because fm_query returns tm_tensors. Only happens when HybridMem is on. */
    fm_results *retrieved = NULL;
    int n_retrieved_tokens = 0;
    float *ret_k_data = NULL;
    float *ret_v_data = NULL;
    int ret_k_allocated = 0;  /* track if we had to fallback-malloc */

    if (hm_store && hm_top_k > 0) {
        /* fm_query needs a tm_tensor for the query — wrap ln_out temporarily */
        int64_t shape_c[] = {c};
        tm_tensor *query_t = tm_from_data(ln_out, shape_c, 1, TM_F32);
        retrieved = fm_query(hm_store, query_t, layer_idx, hm_top_k);
        tm_free(query_t);

        if (retrieved) {
            for (int r = 0; r < retrieved->count; r++)
                n_retrieved_tokens += (int)tm_size(retrieved->items[r].keys, 0);
            if (n_retrieved_tokens > 0) {
                /* Use pre-allocated workspace if it fits, fallback to malloc */
                if (ret_k_ws && ret_v_ws && n_retrieved_tokens <= ret_ws_cap) {
                    ret_k_data = ret_k_ws;
                    ret_v_data = ret_v_ws;
                } else {
                    ret_k_data = (float*)malloc(n_retrieved_tokens * c * sizeof(float));
                    ret_v_data = (float*)malloc(n_retrieved_tokens * c * sizeof(float));
                    ret_k_allocated = 1;
                }
                int offset = 0;
                for (int r = 0; r < retrieved->count; r++) {
                    int ntok = (int)tm_size(retrieved->items[r].keys, 0);
                    memcpy(&ret_k_data[offset * c],
                           tm_data_f32(retrieved->items[r].keys),
                           ntok * c * sizeof(float));
                    memcpy(&ret_v_data[offset * c],
                           tm_data_f32(retrieved->items[r].values),
                           ntok * c * sizeof(float));
                    offset += ntok;
                }
            }
        }
    }

    int ct = st->cache_len;
    int total = ct + n_retrieved_tokens;
    float scale = 1.0f / sqrtf((float)head_size);

    /* Zero output */
    memset(y_data, 0, c * sizeof(float));

    /* Per-head attention with threading parallelism */
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#elif defined(USE_THREADS)
    if (__rx_pool) {
        /* Allocate log buffer if logging enabled */
        float *log_buf = NULL;
        if (attn_log)
            log_buf = (float*)calloc(n_head * 3, sizeof(float));
        attn_head_ctx ctx = { q_vec, k_cache, v_cache, ret_k_data, ret_v_data,
                              y_data, n_head, head_size, c, ct,
                              n_retrieved_tokens, total, scale, log_buf };
        fw_for(__rx_pool, n_head, attn_head_worker, &ctx);
        /* Write accumulated log results from threaded workers */
        if (attn_log && log_buf) {
            for (int h = 0; h < n_head; h++) {
                fprintf(attn_log, "%d,%d,%d,%.6f,%.6f,%.6f\n",
                        attn_log_step, layer_idx, h,
                        log_buf[h * 3 + 1],   /* ret_mass */
                        log_buf[h * 3],        /* cache_mass */
                        log_buf[h * 3 + 2]);   /* entropy */
            }
            free(log_buf);
        }
        goto attn_cleanup;
    }
#endif
    for (int h = 0; h < n_head; h++) {
        int off = h * head_size;
        /* Thread-local attn_scores on stack (max ~16KB for 4096 cache) */
        float attn_scores_local[total];

        /* Q @ K^T: dot product per cached token */
        for (int t = 0; t < ct; t++) {
#ifdef __ARM_NEON
            int hs4 = head_size & ~3;
            float32x4_t dot_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < hs4; j += 4)
                dot_v = vfmaq_f32(dot_v, vld1q_f32(&q_vec[off + j]),
                                         vld1q_f32(&k_cache[t * c + off + j]));
            float dot = vaddvq_f32(dot_v);
            for (int j = hs4; j < head_size; j++)
                dot += q_vec[off + j] * k_cache[t * c + off + j];
#else
            float dot = 0.0f;
            for (int j = 0; j < head_size; j++)
                dot += q_vec[off + j] * k_cache[t * c + off + j];
#endif
            /* Prefetch next K-cache row */
            if (t + 1 < ct)
                __builtin_prefetch(&k_cache[(t + 1) * c + off], 0, 1);
            attn_scores_local[t] = dot * scale;
        }
        for (int t = 0; t < n_retrieved_tokens; t++) {
#ifdef __ARM_NEON
            int hs4 = head_size & ~3;
            float32x4_t dot_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < hs4; j += 4)
                dot_v = vfmaq_f32(dot_v, vld1q_f32(&q_vec[off + j]),
                                         vld1q_f32(&ret_k_data[t * c + off + j]));
            float dot = vaddvq_f32(dot_v);
            for (int j = hs4; j < head_size; j++)
                dot += q_vec[off + j] * ret_k_data[t * c + off + j];
#else
            float dot = 0.0f;
            for (int j = 0; j < head_size; j++)
                dot += q_vec[off + j] * ret_k_data[t * c + off + j];
#endif
            attn_scores_local[ct + t] = dot * scale;
        }

        /* Softmax */
        float max_s = attn_scores_local[0];
        for (int t = 1; t < total; t++)
            if (attn_scores_local[t] > max_s) max_s = attn_scores_local[t];

#if defined(USE_BLAS) && defined(__APPLE__) && !defined(_OPENMP) && !defined(USE_THREADS)
        /* vDSP softmax (only safe in single-threaded mode) */
        float neg_max = -max_s;
        vDSP_vsadd(attn_scores_local, 1, &neg_max, attn_scores_local, 1, total);
        vvexpf(attn_scores_local, attn_scores_local, &total);
        float sum_e;
        vDSP_sve(attn_scores_local, 1, &sum_e, total);
        vDSP_vsdiv(attn_scores_local, 1, &sum_e, attn_scores_local, 1, total);
#else
        float sum_e = 0.0f;
        for (int t = 0; t < total; t++) {
            attn_scores_local[t] = expf(attn_scores_local[t] - max_s);
            sum_e += attn_scores_local[t];
        }
        for (int t = 0; t < total; t++)
            attn_scores_local[t] /= sum_e;
#endif

        /* Value aggregation: y[off+j] = sum_t(attn_scores[t] * v_cache[t,off+j]) */
        for (int j = 0; j < head_size; j++) {
            float val = 0.0f;
            for (int t = 0; t < ct; t++)
                val += attn_scores_local[t] * v_cache[t * c + off + j];
            for (int t = 0; t < n_retrieved_tokens; t++)
                val += attn_scores_local[ct + t] * ret_v_data[t * c + off + j];
            y_data[off + j] = val;
        }

        /* Attention weight + entropy logging (non-threaded path) */
        if (attn_log) {
            float cache_mass = 0.0f, ret_mass = 0.0f, entropy = 0.0f;
            for (int t = 0; t < ct; t++)
                cache_mass += attn_scores_local[t];
            for (int t = 0; t < n_retrieved_tokens; t++)
                ret_mass += attn_scores_local[ct + t];
            for (int t = 0; t < total; t++) {
                if (attn_scores_local[t] > 1e-10f)
                    entropy -= attn_scores_local[t] * logf(attn_scores_local[t]);
            }
            fprintf(attn_log, "%d,%d,%d,%.6f,%.6f,%.6f\n",
                    attn_log_step, layer_idx, h, ret_mass, cache_mass, entropy);
        }
    }
#ifdef USE_THREADS
attn_cleanup:
#endif
    if (ret_k_allocated) { free(ret_k_data); free(ret_v_data); }
    if (retrieved) fm_results_free(retrieved);

    /* Output projection: att_out = y @ output_w */
    if (blk->output_q8)
        fast_matvec_q8(y_data, blk->output_q8, att_out,
                       blk->output_q8->rows, blk->output_q8->cols, q8_buf);
    else
        fast_matvec(y_data, tm_data_f32(blk->output_w), att_out, c, c);

    /* Residual: x1 = x + att_out */
#ifdef __ARM_NEON
    {
        int c4 = c & ~3;
        for (int i = 0; i < c4; i += 4)
            vst1q_f32(&x_out[i], vaddq_f32(vld1q_f32(&x_data[i]),
                                             vld1q_f32(&att_out[i])));
        for (int i = c4; i < c; i++)
            x_out[i] = x_data[i] + att_out[i];
    }
#else
    for (int i = 0; i < c; i++)
        x_out[i] = x_data[i] + att_out[i];
#endif

    /* Pre-FFN LayerNorm */
    fast_layer_norm(x_out, tm_data_f32(blk->ln2_w), tm_data_f32(blk->ln2_b),
                    ln_out, c);

    /* Channel mix: token shift + relu^2 FFN */
    {
        float *ffn_prev = tm_data_f32(st->ffn_x_prev);
        float *shifted = ws->buf[0];
        int ffn_dim = blk->ffn_key_q8 ? blk->ffn_key_q8->rows
                                      : (int)tm_size(blk->ffn_key_w, 1);
        float *fk_buf = ws->ffn_buf;

        fast_token_shift(ln_out, ffn_prev, tm_data_f32(blk->ffn_x_k), shifted, c);
        memcpy(ffn_prev, ln_out, c * sizeof(float));

        if (blk->ffn_key_q8)
            fast_matvec_q8(shifted, blk->ffn_key_q8, fk_buf,
                           blk->ffn_key_q8->rows, blk->ffn_key_q8->cols, q8_buf);
        else
            fast_matvec(shifted, tm_data_f32(blk->ffn_key_w), fk_buf, c, ffn_dim);

        /* relu^2 in place */
#ifdef __ARM_NEON
        {
            int fd4 = ffn_dim & ~3;
            float32x4_t zero = vdupq_n_f32(0.0f);
            for (int i = 0; i < fd4; i += 4) {
                float32x4_t v = vmaxq_f32(vld1q_f32(&fk_buf[i]), zero);
                vst1q_f32(&fk_buf[i], vmulq_f32(v, v));
            }
            for (int i = fd4; i < ffn_dim; i++) {
                float v = fk_buf[i] > 0.0f ? fk_buf[i] : 0.0f;
                fk_buf[i] = v * v;
            }
        }
#else
        for (int i = 0; i < ffn_dim; i++) {
            float v = fk_buf[i] > 0.0f ? fk_buf[i] : 0.0f;
            fk_buf[i] = v * v;
        }
#endif

        if (blk->ffn_value_q8)
            fast_matvec_q8(fk_buf, blk->ffn_value_q8, att_out,
                           blk->ffn_value_q8->rows, blk->ffn_value_q8->cols, q8_buf);
        else
            fast_matvec(fk_buf, tm_data_f32(blk->ffn_value_w), att_out, ffn_dim, c);

        /* Residual: x2 = x1 + ffn_out */
        fast_residual_add(x_out, att_out, c);
    }
}

/* ================================================================
 * Batched sparse attention block forward — process N tokens
 *
 * Converts 6×N SGEMV into 6 SGEMM (Q, K, V, output, FFN key, FFN value).
 * Attention computation (cache + softmax) remains sequential per token.
 * ================================================================ */

static void attn_block_forward_batch(
    rx_attn_block *blk,
    const float *x_batch,   /* [N, C] input */
    rx_attn_state *st,
    fm_store *hm_store, int layer_idx,
    int hm_top_k, float evict_fraction,
    int n_head, int head_size,
    rx_batch_workspace *bws,
    float *x_out,           /* [N, C] output */
    __attribute__((unused)) void *mtl_x_out,  /* persistent Metal buf for x_out */
    FILE *attn_log, int *attn_log_step_ptr,
    float *embed_capture, int *embed_capture_count,
    int embed_capture_max, int embed_capture_layer,
    float *ret_k_ws, float *ret_v_ws, int ret_ws_cap)
{
    int c = n_head * head_size;
    int N = bws->n;
    float *ln      = bws->ln;
    float *Q_batch = bws->proj[0];
    float *K_batch = bws->proj[1];
    float *V_batch = bws->proj[2];
    float *wkv_out = bws->wkv_out;

    /* ---- Phase A: Batch LayerNorm + SGEMM Q/K/V ---- */
    float *ln1_w = tm_data_f32(blk->ln1_w);
    float *ln1_b = tm_data_f32(blk->ln1_b);
    for (int t = 0; t < N; t++)
        fast_layer_norm(&x_batch[t * c], ln1_w, ln1_b, &ln[t * c], c);

    /* Embedding capture (batch path) */
    if (embed_capture && embed_capture_count &&
        (embed_capture_layer < 0 || embed_capture_layer == layer_idx)) {
        for (int t = 0; t < N && *embed_capture_count < embed_capture_max; t++) {
            memcpy(&embed_capture[*embed_capture_count * c], &ln[t * c], c * sizeof(float));
            (*embed_capture_count)++;
        }
    }

    /* No token shifts in attention blocks — Q/K/V come directly from ln. */
    BATCH_SGEMM(ln, blk->receptance_q8, blk->receptance_w,
                Q_batch, N, c, c, MTL_W(blk->mtl_receptance_w), bws->q8_buf,
                MTL_BUF(bws->mtl_ln), MTL_BUF(bws->mtl_proj[0]));
    BATCH_SGEMM(ln, blk->key_q8, blk->key_w,
                K_batch, N, c, c, MTL_W(blk->mtl_key_w), bws->q8_buf,
                MTL_BUF(bws->mtl_ln), MTL_BUF(bws->mtl_proj[1]));
    BATCH_SGEMM(ln, blk->value_q8, blk->value_w,
                V_batch, N, c, c, MTL_W(blk->mtl_value_w), bws->q8_buf,
                MTL_BUF(bws->mtl_ln), MTL_BUF(bws->mtl_proj[2]));
#ifdef USE_METAL
    fm_sync(); /* CPU needs Q/K/V projections for attention */
#endif

    /* ---- Phase B: Sequential attention per token ---- */
    float *k_cache = tm_data_f32(st->k_cache);
    float *v_cache = tm_data_f32(st->v_cache);

    for (int t = 0; t < N; t++) {
        float *q_t = &Q_batch[t * c];
        float *k_t = &K_batch[t * c];
        float *v_t = &V_batch[t * c];
        float *y_t = &wkv_out[t * c];

        /* Append K/V to cache — evict to disk if full */
        if (st->cache_len >= st->max_cache_len) {
            int evict_count = (int)(st->max_cache_len * evict_fraction);
            if (evict_count < 1) evict_count = 1;
            if (evict_count > st->cache_len) evict_count = st->cache_len;

            if (hm_store) {
                int64_t ev_shape[] = {evict_count, c};
                tm_tensor *ev_k = tm_from_data(k_cache, ev_shape, 2, TM_F32);
                tm_tensor *ev_v = tm_from_data(v_cache, ev_shape, 2, TM_F32);

                if (st->proj_accum && st->proj_accum_dim > 0) {
                    int hd = st->proj_accum_dim;
                    float *sum_accum = (float*)calloc(hd, sizeof(float));
                    for (int ei = 0; ei < evict_count; ei++) {
                        float *slot = &st->proj_accum[ei * hd];
                        for (int j = 0; j < hd; j++) sum_accum[j] += slot[j];
                    }
                    fv_vector *sig = fv_project_threshold(
                        fm_store_projector(hm_store), sum_accum);
                    free(sum_accum);
                    fm_put_presig(hm_store, ev_k, ev_v, layer_idx, 1.0f, sig);
                } else {
                    fm_put(hm_store, ev_k, ev_v, layer_idx, 1.0f);
                }
                tm_free(ev_k); tm_free(ev_v);
            }

            int remain = st->cache_len - evict_count;
            if (remain > 0) {
                memmove(k_cache, &k_cache[evict_count * c], remain * c * sizeof(float));
                memmove(v_cache, &v_cache[evict_count * c], remain * c * sizeof(float));
                if (st->proj_accum && st->proj_accum_dim > 0) {
                    int hd = st->proj_accum_dim;
                    memmove(st->proj_accum, &st->proj_accum[evict_count * hd],
                            (size_t)remain * hd * sizeof(float));
                }
            }
            st->cache_len = remain;
        }

        memcpy(&k_cache[st->cache_len * c], k_t, c * sizeof(float));
        memcpy(&v_cache[st->cache_len * c], v_t, c * sizeof(float));

        if (st->proj_accum && st->proj_accum_dim > 0) {
            int hd = st->proj_accum_dim;
            float *slot = &st->proj_accum[st->cache_len * hd];
            memset(slot, 0, hd * sizeof(float));
            fv_project_accum(fm_store_projector(hm_store), k_t, slot);
        }

        st->cache_len++;

        /* HybridMem retrieval */
        fm_results *retrieved = NULL;
        int n_retrieved_tokens = 0;
        float *ret_k_data = NULL;
        float *ret_v_data = NULL;
        int ret_k_allocated = 0;

        if (hm_store && hm_top_k > 0) {
            int64_t shape_c[] = {c};
            tm_tensor *query_t_tensor = tm_from_data(&ln[t * c], shape_c, 1, TM_F32);
            retrieved = fm_query(hm_store, query_t_tensor, layer_idx, hm_top_k);
            tm_free(query_t_tensor);

            if (retrieved) {
                for (int r = 0; r < retrieved->count; r++)
                    n_retrieved_tokens += (int)tm_size(retrieved->items[r].keys, 0);
                if (n_retrieved_tokens > 0) {
                    if (ret_k_ws && ret_v_ws && n_retrieved_tokens <= ret_ws_cap) {
                        ret_k_data = ret_k_ws;
                        ret_v_data = ret_v_ws;
                    } else {
                        ret_k_data = (float*)malloc(n_retrieved_tokens * c * sizeof(float));
                        ret_v_data = (float*)malloc(n_retrieved_tokens * c * sizeof(float));
                        ret_k_allocated = 1;
                    }
                    int offset = 0;
                    for (int r = 0; r < retrieved->count; r++) {
                        int ntok = (int)tm_size(retrieved->items[r].keys, 0);
                        memcpy(&ret_k_data[offset * c],
                               tm_data_f32(retrieved->items[r].keys),
                               ntok * c * sizeof(float));
                        memcpy(&ret_v_data[offset * c],
                               tm_data_f32(retrieved->items[r].values),
                               ntok * c * sizeof(float));
                        offset += ntok;
                    }
                }
            }
        }

        int ct = st->cache_len;
        int total = ct + n_retrieved_tokens;
        float scale = 1.0f / sqrtf((float)head_size);

        memset(y_t, 0, c * sizeof(float));

        /* Per-head attention */
        for (int h = 0; h < n_head; h++) {
            int off = h * head_size;
            float attn_scores[total];

            for (int tt = 0; tt < ct; tt++) {
                float dot = 0.0f;
                for (int j = 0; j < head_size; j++)
                    dot += q_t[off + j] * k_cache[tt * c + off + j];
                attn_scores[tt] = dot * scale;
            }
            for (int tt = 0; tt < n_retrieved_tokens; tt++) {
                float dot = 0.0f;
                for (int j = 0; j < head_size; j++)
                    dot += q_t[off + j] * ret_k_data[tt * c + off + j];
                attn_scores[ct + tt] = dot * scale;
            }

            float max_s = attn_scores[0];
            for (int tt = 1; tt < total; tt++)
                if (attn_scores[tt] > max_s) max_s = attn_scores[tt];
            float sum_e = 0.0f;
            for (int tt = 0; tt < total; tt++) {
                attn_scores[tt] = expf(attn_scores[tt] - max_s);
                sum_e += attn_scores[tt];
            }
            for (int tt = 0; tt < total; tt++)
                attn_scores[tt] /= sum_e;

            for (int j = 0; j < head_size; j++) {
                float val = 0.0f;
                for (int tt = 0; tt < ct; tt++)
                    val += attn_scores[tt] * v_cache[tt * c + off + j];
                for (int tt = 0; tt < n_retrieved_tokens; tt++)
                    val += attn_scores[ct + tt] * ret_v_data[tt * c + off + j];
                y_t[off + j] = val;
            }

            /* Attention weight + entropy logging (batch path) */
            if (attn_log) {
                float cache_mass = 0.0f, ret_mass = 0.0f, entropy = 0.0f;
                for (int tt = 0; tt < ct; tt++)
                    cache_mass += attn_scores[tt];
                for (int tt = 0; tt < n_retrieved_tokens; tt++)
                    ret_mass += attn_scores[ct + tt];
                for (int tt = 0; tt < total; tt++) {
                    if (attn_scores[tt] > 1e-10f)
                        entropy -= attn_scores[tt] * logf(attn_scores[tt]);
                }
                int log_step = attn_log_step_ptr ? *attn_log_step_ptr : t;
                fprintf(attn_log, "%d,%d,%d,%.6f,%.6f,%.6f\n",
                        log_step, layer_idx, h, ret_mass, cache_mass, entropy);
            }
        }

        if (ret_k_allocated) { free(ret_k_data); free(ret_v_data); }
        if (retrieved) fm_results_free(retrieved);

        /* Increment batch step counter */
        if (attn_log_step_ptr)
            (*attn_log_step_ptr)++;
    }

    /* ---- Phase C: Batch output projection + residual ---- */
    BATCH_SGEMM(wkv_out, blk->output_q8, blk->output_w,
                x_out, N, c, c, MTL_W(blk->mtl_output_w), bws->q8_buf,
                MTL_BUF(bws->mtl_wkv_out), mtl_x_out);
#ifdef USE_METAL
    fm_sync(); /* CPU needs output projection for residual add */
#endif
    for (int t = 0; t < N; t++)
        fast_residual_add(&x_out[t * c], &x_batch[t * c], c);

    /* ---- Phase D: Batch FFN ---- */
    float *ln2_w = tm_data_f32(blk->ln2_w);
    float *ln2_b = tm_data_f32(blk->ln2_b);
    int ffn_dim = blk->ffn_key_q8 ? blk->ffn_key_q8->rows : (int)tm_size(blk->ffn_key_w, 1);
    float *ffn_x_k_d = tm_data_f32(blk->ffn_x_k);
    float *ffn_prev = tm_data_f32(st->ffn_x_prev);
    float *shifted = bws->shifted;

    /* Batch pre-FFN LayerNorm */
    for (int t = 0; t < N; t++)
        fast_layer_norm(&x_out[t * c], ln2_w, ln2_b, &ln[t * c], c);

    /* FFN token shifts */
    for (int t = 0; t < N; t++) {
        const float *prev = (t == 0) ? ffn_prev : &ln[(t - 1) * c];
        fast_token_shift(&ln[t * c], prev, ffn_x_k_d, &shifted[t * c], c);
    }
    memcpy(ffn_prev, &ln[(N - 1) * c], c * sizeof(float));

    /* SGEMM: shifted[N, C] @ ffn_key_w[C, ffn_dim] -> ffn_k[N, ffn_dim] */
    float *ffn_k = bws->ffn_k;
    BATCH_SGEMM(shifted, blk->ffn_key_q8, blk->ffn_key_w,
                ffn_k, N, c, ffn_dim, MTL_W(blk->mtl_ffn_key_w), bws->q8_buf,
                MTL_BUF(bws->mtl_shifted), MTL_BUF(bws->mtl_ffn_k));
#ifdef USE_METAL
    fm_sync(); /* CPU needs ffn_k for ReLU² */
#endif

    /* Batch ReLU² */
    {
        int total_ffn = N * ffn_dim;
#ifdef __ARM_NEON
        int t4 = total_ffn & ~3;
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (int i = 0; i < t4; i += 4) {
            float32x4_t v = vmaxq_f32(vld1q_f32(&ffn_k[i]), zero);
            vst1q_f32(&ffn_k[i], vmulq_f32(v, v));
        }
        for (int i = t4; i < total_ffn; i++) {
            float v = ffn_k[i] > 0.0f ? ffn_k[i] : 0.0f;
            ffn_k[i] = v * v;
        }
#else
        for (int i = 0; i < total_ffn; i++) {
            float v = ffn_k[i] > 0.0f ? ffn_k[i] : 0.0f;
            ffn_k[i] = v * v;
        }
#endif
    }

    /* SGEMM: ffn_k[N, ffn_dim] @ ffn_value_w[ffn_dim, C] -> temp[N, C] */
    BATCH_SGEMM(ffn_k, blk->ffn_value_q8, blk->ffn_value_w,
                wkv_out, N, ffn_dim, c, MTL_W(blk->mtl_ffn_value_w), bws->q8_buf,
                MTL_BUF(bws->mtl_ffn_k), MTL_BUF(bws->mtl_wkv_out));
#ifdef USE_METAL
    fm_sync(); /* CPU needs ffn_value output for residual add */
#endif

    /* Batch residual: x_out += ffn_out */
    for (int t = 0; t < N; t++)
        fast_residual_add(&x_out[t * c], &wkv_out[t * c], c);
}

#ifdef USE_THREADS
/* Worker callback for attention per-head parallelism — processes heads [start, end) */
static void attn_head_worker(int start, int end, void *arg) {
    attn_head_ctx *ctx = (attn_head_ctx *)arg;
    int head_size = ctx->head_size;
    int c = ctx->c;
    int ct = ctx->ct;
    int total = ctx->total;
    int n_retrieved_tokens = ctx->n_retrieved_tokens;
    float scale = ctx->scale;

    for (int h = start; h < end; h++) {
        int off = h * head_size;
        /* Thread-local attn_scores on stack */
        float attn_scores_local[total];

        /* Q @ K^T: dot product per cached token */
        for (int t = 0; t < ct; t++) {
#ifdef __ARM_NEON
            int hs4 = head_size & ~3;
            float32x4_t dot_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < hs4; j += 4)
                dot_v = vfmaq_f32(dot_v, vld1q_f32(&ctx->q_vec[off + j]),
                                         vld1q_f32(&ctx->k_cache[t * c + off + j]));
            float dot = vaddvq_f32(dot_v);
            for (int j = hs4; j < head_size; j++)
                dot += ctx->q_vec[off + j] * ctx->k_cache[t * c + off + j];
#else
            float dot = 0.0f;
            for (int j = 0; j < head_size; j++)
                dot += ctx->q_vec[off + j] * ctx->k_cache[t * c + off + j];
#endif
            if (t + 1 < ct)
                __builtin_prefetch(&ctx->k_cache[(t + 1) * c + off], 0, 1);
            attn_scores_local[t] = dot * scale;
        }
        for (int t = 0; t < n_retrieved_tokens; t++) {
#ifdef __ARM_NEON
            int hs4 = head_size & ~3;
            float32x4_t dot_v = vdupq_n_f32(0.0f);
            for (int j = 0; j < hs4; j += 4)
                dot_v = vfmaq_f32(dot_v, vld1q_f32(&ctx->q_vec[off + j]),
                                         vld1q_f32(&ctx->ret_k_data[t * c + off + j]));
            float dot = vaddvq_f32(dot_v);
            for (int j = hs4; j < head_size; j++)
                dot += ctx->q_vec[off + j] * ctx->ret_k_data[t * c + off + j];
#else
            float dot = 0.0f;
            for (int j = 0; j < head_size; j++)
                dot += ctx->q_vec[off + j] * ctx->ret_k_data[t * c + off + j];
#endif
            attn_scores_local[ct + t] = dot * scale;
        }

        /* Softmax */
        float max_s = attn_scores_local[0];
        for (int t = 1; t < total; t++)
            if (attn_scores_local[t] > max_s) max_s = attn_scores_local[t];

        float sum_e = 0.0f;
        for (int t = 0; t < total; t++) {
            attn_scores_local[t] = expf(attn_scores_local[t] - max_s);
            sum_e += attn_scores_local[t];
        }
        for (int t = 0; t < total; t++)
            attn_scores_local[t] /= sum_e;

        /* Value aggregation */
        for (int j = 0; j < head_size; j++) {
            float val = 0.0f;
            for (int t = 0; t < ct; t++)
                val += attn_scores_local[t] * ctx->v_cache[t * c + off + j];
            for (int t = 0; t < n_retrieved_tokens; t++)
                val += attn_scores_local[ct + t] * ctx->ret_v_data[t * c + off + j];
            ctx->y_data[off + j] = val;
        }

        /* Attention weight + entropy logging (threaded path) */
        if (ctx->attn_log_buf) {
            float cache_mass = 0.0f, ret_mass = 0.0f, entropy = 0.0f;
            for (int t = 0; t < ct; t++)
                cache_mass += attn_scores_local[t];
            for (int t = 0; t < n_retrieved_tokens; t++)
                ret_mass += attn_scores_local[ct + t];
            for (int t = 0; t < total; t++) {
                if (attn_scores_local[t] > 1e-10f)
                    entropy -= attn_scores_local[t] * logf(attn_scores_local[t]);
            }
            ctx->attn_log_buf[h * 3]     = cache_mass;
            ctx->attn_log_buf[h * 3 + 1] = ret_mass;
            ctx->attn_log_buf[h * 3 + 2] = entropy;
        }
    }
}
#endif

/* ================================================================
 * Weight loading helpers
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

/* Get a tensor from safetensors, with transpose if it's a linear weight.
 * Supports F32 and BF16 dtypes (BF16 is converted to F32).
 * Returns a new owned tm_tensor (caller must free). */
static tm_tensor* load_weight(fs_file *f, const char *name) {
    if (!fs_has_tensor(f, name))
        return NULL;

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
        fprintf(stderr, "rwkvx: unsupported dtype %s for weight %s\n",
                fs_dtype_name(info->dtype), name);
        fs_free_info(info);
        return NULL;
    }

    /* Transpose linear weights: [out, in] -> [in, out] */
    bool is_linear = strstr(name, "key.weight") || strstr(name, "value.weight") ||
                     strstr(name, "receptance.weight") || strstr(name, "output.weight") ||
                     strstr(name, "head.weight");
    if (is_linear && info->ndim == 2) {
        tm_tensor *tt = tm_t(t);
        tm_tensor *tc = tm_contiguous(tt);
        tm_free(t); tm_free(tt);
        t = tc;
    }

    /* Squeeze singleton dims */
    for (int d = tm_ndim(t) - 1; d >= 0; d--) {
        if (tm_size(t, d) == 1 && tm_ndim(t) > 1) {
            tm_tensor *sq = tm_squeeze(t, d);
            tm_free(t);
            t = sq;
        }
    }

    /* Flatten att.r_k from (H, N) to (H*N) */
    if (strstr(name, "att.r_k") && tm_ndim(t) == 2) {
        int64_t flat_shape[] = {tm_numel(t)};
        tm_tensor *flat = tm_reshape(t, flat_shape, 1);
        tm_free(t);
        t = flat;
    }

    fs_free_info(info);
    return t;
}

/* Get a required weight. Prints error and returns NULL on missing. */
static tm_tensor* require_weight(fs_file *f, const char *name) {
    tm_tensor *t = load_weight(f, name);
    if (!t) {
        fprintf(stderr, "rwkvx: missing required weight: %s\n", name);
    }
    return t;
}

/* Pre-normalize embeddings: emb = layer_norm(emb, ln0.w, ln0.b) per row */
static void pre_normalize_embeddings(tm_tensor **emb,
                                     tm_tensor *ln0_w, tm_tensor *ln0_b,
                                     int vocab_size, int n_embd)
{
    if (!ln0_w || !ln0_b) return;

    float *e = tm_data_f32(*emb);
    float *w = tm_data_f32(ln0_w);
    float *b = tm_data_f32(ln0_b);
    float eps = 1e-5f;

    for (int v = 0; v < vocab_size; v++) {
        float *row = &e[v * n_embd];
        /* mean */
        float mean = 0.0f;
        for (int j = 0; j < n_embd; j++) mean += row[j];
        mean /= n_embd;
        /* variance */
        float var = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            float d = row[j] - mean;
            var += d * d;
        }
        var /= n_embd;
        float inv_std = 1.0f / sqrtf(var + eps);
        /* normalize */
        for (int j = 0; j < n_embd; j++) {
            row[j] = (row[j] - mean) * inv_std * w[j] + b[j];
        }
    }
}

/* ================================================================
 * Public: Model loading
 * ================================================================ */

rx_model* rx_model_load(const char *path) {
#ifdef USE_METAL
    fm_init();  /* Initialize Metal if available (no-op if already init) */
#endif
#ifdef USE_CUDA
    fc_init();  /* Initialize CUDA if available (no-op if already init) */
#endif
    fs_file *f = fs_open(path);
    if (!f) {
        fprintf(stderr, "rwkvx: failed to open %s\n", path);
        return NULL;
    }

    /* Determine config from emb.weight shape */
    fs_tensor_info *emb_info = fs_get_info(f, "rwkv.emb.weight");
    if (!emb_info) {
        fprintf(stderr, "rwkvx: missing rwkv.emb.weight\n");
        fs_close(f);
        return NULL;
    }
    int vocab_size = (int)emb_info->shape[0];
    int n_embd = (int)emb_info->shape[1];
    int head_size = 64;
    int n_head = n_embd / head_size;
    fs_free_info(emb_info);

    /* Count layers from tensor names */
    int count;
    const char **names = fs_tensor_names(f, &count);
    int max_rwkv = -1, max_attn = -1;
    for (int i = 0; i < count; i++) {
        int idx;
        if (sscanf(names[i], "rwkv.blocks.%d.", &idx) == 1) {
            if (idx > max_rwkv) max_rwkv = idx;
        } else if (sscanf(names[i], "moba.%d.", &idx) == 1) {
            if (idx > max_attn) max_attn = idx;
        }
    }
    int n_rwkv_layer = max_rwkv + 1;
    int n_attn_layer = max_attn >= 0 ? max_attn + 1 : 0;

    fprintf(stderr, "rwkvx: config n_embd=%d n_head=%d n_rwkv=%d n_attn=%d vocab=%d\n",
            n_embd, n_head, n_rwkv_layer, n_attn_layer, vocab_size);

    /* Allocate model */
    rx_model *m = (rx_model*)calloc(1, sizeof(rx_model));
    m->config.n_embd = n_embd;
    m->config.n_head = n_head;
    m->config.head_size = head_size;
    m->config.n_rwkv_layer = n_rwkv_layer;
    m->config.n_attn_layer = n_attn_layer;
    m->config.vocab_size = vocab_size;

    /* Load global weights */
    m->emb_weight = require_weight(f, "rwkv.emb.weight");
    m->ln_out_w = require_weight(f, "rwkv.ln_out.weight");
    m->ln_out_b = require_weight(f, "rwkv.ln_out.bias");
    m->head_weight = require_weight(f, "rwkv.head.weight");

    if (!m->emb_weight || !m->ln_out_w || !m->ln_out_b || !m->head_weight) {
        rx_model_free(m);
        fs_close(f);
        return NULL;
    }

    /* Pre-normalize embeddings */
    tm_tensor *ln0_w = load_weight(f, "rwkv.blocks.0.ln0.weight");
    tm_tensor *ln0_b = load_weight(f, "rwkv.blocks.0.ln0.bias");
    pre_normalize_embeddings(&m->emb_weight, ln0_w, ln0_b, vocab_size, n_embd);
    if (ln0_w) tm_free(ln0_w);
    if (ln0_b) tm_free(ln0_b);

    /* Load RWKV-7 blocks */
    m->rwkv_blocks = (rx_rwkv_block*)calloc(n_rwkv_layer, sizeof(rx_rwkv_block));
    for (int i = 0; i < n_rwkv_layer; i++) {
        char buf[256];
        rx_rwkv_block *b = &m->rwkv_blocks[i];

        #define LOAD_RWKV(field, fmt, ...) do { \
            snprintf(buf, sizeof(buf), fmt, __VA_ARGS__); \
            b->field = require_weight(f, buf); \
        } while(0)

        LOAD_RWKV(ln1_w, "rwkv.blocks.%d.ln1.weight", i);
        LOAD_RWKV(ln1_b, "rwkv.blocks.%d.ln1.bias", i);
        LOAD_RWKV(ln2_w, "rwkv.blocks.%d.ln2.weight", i);
        LOAD_RWKV(ln2_b, "rwkv.blocks.%d.ln2.bias", i);

        LOAD_RWKV(x_r, "rwkv.blocks.%d.att.x_r", i);
        LOAD_RWKV(x_w, "rwkv.blocks.%d.att.x_w", i);
        LOAD_RWKV(x_k, "rwkv.blocks.%d.att.x_k", i);
        LOAD_RWKV(x_v, "rwkv.blocks.%d.att.x_v", i);
        LOAD_RWKV(x_a, "rwkv.blocks.%d.att.x_a", i);
        LOAD_RWKV(x_g, "rwkv.blocks.%d.att.x_g", i);

        LOAD_RWKV(w0, "rwkv.blocks.%d.att.w0", i);
        LOAD_RWKV(w1, "rwkv.blocks.%d.att.w1", i);
        LOAD_RWKV(w2, "rwkv.blocks.%d.att.w2", i);
        LOAD_RWKV(a0, "rwkv.blocks.%d.att.a0", i);
        LOAD_RWKV(a1, "rwkv.blocks.%d.att.a1", i);
        LOAD_RWKV(a2, "rwkv.blocks.%d.att.a2", i);

        LOAD_RWKV(g1, "rwkv.blocks.%d.att.g1", i);
        LOAD_RWKV(g2, "rwkv.blocks.%d.att.g2", i);
        LOAD_RWKV(k_k, "rwkv.blocks.%d.att.k_k", i);
        LOAD_RWKV(k_a, "rwkv.blocks.%d.att.k_a", i);
        LOAD_RWKV(r_k, "rwkv.blocks.%d.att.r_k", i);

        LOAD_RWKV(receptance_w, "rwkv.blocks.%d.att.receptance.weight", i);
        LOAD_RWKV(key_w, "rwkv.blocks.%d.att.key.weight", i);
        LOAD_RWKV(value_w, "rwkv.blocks.%d.att.value.weight", i);
        LOAD_RWKV(output_w, "rwkv.blocks.%d.att.output.weight", i);

        LOAD_RWKV(ln_x_w, "rwkv.blocks.%d.att.ln_x.weight", i);
        LOAD_RWKV(ln_x_b, "rwkv.blocks.%d.att.ln_x.bias", i);

        LOAD_RWKV(ffn_x_k, "rwkv.blocks.%d.ffn.x_k", i);
        LOAD_RWKV(ffn_key_w, "rwkv.blocks.%d.ffn.key.weight", i);
        LOAD_RWKV(ffn_value_w, "rwkv.blocks.%d.ffn.value.weight", i);

        /* v0/v1/v2 only for layer > 0 */
        if (i > 0) {
            LOAD_RWKV(v0, "rwkv.blocks.%d.att.v0", i);
            LOAD_RWKV(v1, "rwkv.blocks.%d.att.v1", i);
            LOAD_RWKV(v2, "rwkv.blocks.%d.att.v2", i);
        }

        #undef LOAD_RWKV

        fprintf(stderr, "rwkvx: loaded RWKV-7 block %d\n", i);
    }

    /* Load sparse attention blocks */
    m->attn_blocks = (rx_attn_block*)calloc(n_attn_layer > 0 ? n_attn_layer : 1,
                                            sizeof(rx_attn_block));
    for (int i = 0; i < n_attn_layer; i++) {
        char buf[256];
        rx_attn_block *b = &m->attn_blocks[i];

        #define LOAD_ATTN(field, fmt, ...) do { \
            snprintf(buf, sizeof(buf), fmt, __VA_ARGS__); \
            b->field = require_weight(f, buf); \
        } while(0)

        LOAD_ATTN(ln1_w, "moba.%d.ln1.weight", i);
        LOAD_ATTN(ln1_b, "moba.%d.ln1.bias", i);
        LOAD_ATTN(ln2_w, "moba.%d.ln2.weight", i);
        LOAD_ATTN(ln2_b, "moba.%d.ln2.bias", i);
        LOAD_ATTN(receptance_w, "moba.%d.att.receptance.weight", i);
        LOAD_ATTN(key_w, "moba.%d.att.key.weight", i);
        LOAD_ATTN(value_w, "moba.%d.att.value.weight", i);
        LOAD_ATTN(output_w, "moba.%d.att.output.weight", i);
        LOAD_ATTN(ffn_x_k, "moba.%d.ffn.x_k", i);
        LOAD_ATTN(ffn_key_w, "moba.%d.ffn.key.weight", i);
        LOAD_ATTN(ffn_value_w, "moba.%d.ffn.value.weight", i);

        #undef LOAD_ATTN

        fprintf(stderr, "rwkvx: loaded sparse attention block %d\n", i);
    }

    /* Build block execution order: interleaved RWKV + ATTN */
    int n_blocks = n_rwkv_layer + n_attn_layer;
    m->n_blocks = n_blocks;
    m->block_order = (int*)calloc(n_blocks, sizeof(int));
    m->block_index = (int*)calloc(n_blocks, sizeof(int));

    if (n_attn_layer == 0) {
        for (int i = 0; i < n_rwkv_layer; i++) {
            m->block_order[i] = 0; /* RWKV */
            m->block_index[i] = i;
        }
    } else {
        int interval = n_rwkv_layer / n_attn_layer;
        int pos = 0;
        for (int a = 0; a < n_attn_layer; a++) {
            int start = a * interval;
            int end = (a + 1) * interval;
            for (int j = start; j < end; j++) {
                m->block_order[pos] = 0; /* RWKV */
                m->block_index[pos] = j;
                pos++;
            }
            m->block_order[pos] = 1; /* ATTN */
            m->block_index[pos] = a;
            pos++;
        }
    }

#ifdef USE_THREADS
    m->pool = fw_pool_new(0); /* auto-detect core count */
    if (m->pool)
        fprintf(stderr, "rwkvx: pthread thread pool created\n");
#endif

    fs_close(f);

#ifdef USE_METAL
    if (fm_available()) {
        /* Check if weight tensor data is page-aligned (determines zero-copy vs copy) */
        long ps = sysconf(_SC_PAGESIZE);
        int aligned = 0, total = 0;
        for (int i = 0; i < n_rwkv_layer; i++) {
            rx_rwkv_block *b = &m->rwkv_blocks[i];
            void *ptrs[] = { tm_data_f32(b->receptance_w), tm_data_f32(b->key_w),
                             tm_data_f32(b->value_w), tm_data_f32(b->output_w),
                             tm_data_f32(b->ffn_key_w), tm_data_f32(b->ffn_value_w) };
            for (int j = 0; j < 6; j++) {
                total++;
                if (((uintptr_t)ptrs[j] % (uintptr_t)ps) == 0) aligned++;
            }
        }
        fprintf(stderr, "rwkvx: Metal weight alignment: %d/%d page-aligned (page=%ld)\n",
                aligned, total, ps);

        /* Create persistent Metal buffers for f32 weight tensors (zero-copy) */
        for (int i = 0; i < n_rwkv_layer; i++) {
            rx_rwkv_block *b = &m->rwkv_blocks[i];
            int c_dim = (int)tm_size(b->receptance_w, 0);
            int d_dim = (int)tm_size(b->receptance_w, 1);
            size_t wb = (size_t)c_dim * d_dim * sizeof(float);
            b->mtl_receptance_w = fm_buffer_from_ptr(tm_data_f32(b->receptance_w), wb);
            b->mtl_key_w        = fm_buffer_from_ptr(tm_data_f32(b->key_w), wb);
            b->mtl_value_w      = fm_buffer_from_ptr(tm_data_f32(b->value_w), wb);
            b->mtl_output_w     = fm_buffer_from_ptr(tm_data_f32(b->output_w), wb);
            int ffn_c = (int)tm_size(b->ffn_key_w, 0);
            int ffn_d = (int)tm_size(b->ffn_key_w, 1);
            b->mtl_ffn_key_w    = fm_buffer_from_ptr(tm_data_f32(b->ffn_key_w),
                                                      (size_t)ffn_c * ffn_d * sizeof(float));
            int fv_c = (int)tm_size(b->ffn_value_w, 0);
            int fv_d = (int)tm_size(b->ffn_value_w, 1);
            b->mtl_ffn_value_w  = fm_buffer_from_ptr(tm_data_f32(b->ffn_value_w),
                                                      (size_t)fv_c * fv_d * sizeof(float));
        }
        for (int i = 0; i < n_attn_layer; i++) {
            rx_attn_block *b = &m->attn_blocks[i];
            int c_dim = (int)tm_size(b->receptance_w, 0);
            int d_dim = (int)tm_size(b->receptance_w, 1);
            size_t wb = (size_t)c_dim * d_dim * sizeof(float);
            b->mtl_receptance_w = fm_buffer_from_ptr(tm_data_f32(b->receptance_w), wb);
            b->mtl_key_w        = fm_buffer_from_ptr(tm_data_f32(b->key_w), wb);
            b->mtl_value_w      = fm_buffer_from_ptr(tm_data_f32(b->value_w), wb);
            b->mtl_output_w     = fm_buffer_from_ptr(tm_data_f32(b->output_w), wb);
            int ffn_c = (int)tm_size(b->ffn_key_w, 0);
            int ffn_d = (int)tm_size(b->ffn_key_w, 1);
            b->mtl_ffn_key_w    = fm_buffer_from_ptr(tm_data_f32(b->ffn_key_w),
                                                      (size_t)ffn_c * ffn_d * sizeof(float));
            int fv_c = (int)tm_size(b->ffn_value_w, 0);
            int fv_d = (int)tm_size(b->ffn_value_w, 1);
            b->mtl_ffn_value_w  = fm_buffer_from_ptr(tm_data_f32(b->ffn_value_w),
                                                      (size_t)fv_c * fv_d * sizeof(float));
        }
        /* Head weight */
        {
            int h_c = (int)tm_size(m->head_weight, 0);
            int h_d = (int)tm_size(m->head_weight, 1);
            m->mtl_head_w = fm_buffer_from_ptr(tm_data_f32(m->head_weight),
                                                (size_t)h_c * h_d * sizeof(float));
        }
        fprintf(stderr, "rwkvx: Metal persistent f32 weight buffers created\n");
    }
#endif

    fprintf(stderr, "rwkvx: model loaded successfully (%d blocks)\n", n_blocks);
    return m;
}

/* Quantize a 2D weight tensor [in_dim, out_dim] to INT8.
 * The quantized weight is stored transposed as [out_dim, in_dim] so that
 * fast_matvec_q8 can compute out[i] = dot(x, row_i) efficiently. */
static rx_q8_weight* rx_quantize_weight(tm_tensor *t) {
    if (!t || tm_ndim(t) != 2) return NULL;
    int in_dim = (int)tm_size(t, 0);   /* rows of original */
    int out_dim = (int)tm_size(t, 1);  /* cols of original */
    const float *src = tm_data_f32(t);

    /* Transpose: [in, out] -> [out, in] */
    float *transposed = (float*)malloc(out_dim * in_dim * sizeof(float));
    for (int i = 0; i < in_dim; i++)
        for (int j = 0; j < out_dim; j++)
            transposed[j * in_dim + i] = src[i * out_dim + j];

    rx_q8_weight *q = rx_q8_from_f32(transposed, out_dim, in_dim);
    free(transposed);
    return q;
}

/* Post-load quantization: quantize all large linear weight matrices to INT8 */
static void rx_model_quantize_weights(rx_model *m) {
    if (!m || m->quantized) return;

    for (int i = 0; i < m->config.n_rwkv_layer; i++) {
        rx_rwkv_block *b = &m->rwkv_blocks[i];
        b->receptance_q8 = rx_quantize_weight(b->receptance_w);
        b->key_q8 = rx_quantize_weight(b->key_w);
        b->value_q8 = rx_quantize_weight(b->value_w);
        b->output_q8 = rx_quantize_weight(b->output_w);
        b->ffn_key_q8 = rx_quantize_weight(b->ffn_key_w);
        b->ffn_value_q8 = rx_quantize_weight(b->ffn_value_w);

        /* Free f32 weight tensors and their Metal buffers to save memory */
#ifdef USE_METAL
        if (b->mtl_receptance_w) { fm_buffer_free(b->mtl_receptance_w); b->mtl_receptance_w = NULL; }
        if (b->mtl_key_w) { fm_buffer_free(b->mtl_key_w); b->mtl_key_w = NULL; }
        if (b->mtl_value_w) { fm_buffer_free(b->mtl_value_w); b->mtl_value_w = NULL; }
        if (b->mtl_output_w) { fm_buffer_free(b->mtl_output_w); b->mtl_output_w = NULL; }
        if (b->mtl_ffn_key_w) { fm_buffer_free(b->mtl_ffn_key_w); b->mtl_ffn_key_w = NULL; }
        if (b->mtl_ffn_value_w) { fm_buffer_free(b->mtl_ffn_value_w); b->mtl_ffn_value_w = NULL; }
#endif
        tm_free(b->receptance_w); b->receptance_w = NULL;
        tm_free(b->key_w); b->key_w = NULL;
        tm_free(b->value_w); b->value_w = NULL;
        tm_free(b->output_w); b->output_w = NULL;
        tm_free(b->ffn_key_w); b->ffn_key_w = NULL;
        tm_free(b->ffn_value_w); b->ffn_value_w = NULL;
    }

    for (int i = 0; i < m->config.n_attn_layer; i++) {
        rx_attn_block *b = &m->attn_blocks[i];
        b->receptance_q8 = rx_quantize_weight(b->receptance_w);
        b->key_q8 = rx_quantize_weight(b->key_w);
        b->value_q8 = rx_quantize_weight(b->value_w);
        b->output_q8 = rx_quantize_weight(b->output_w);
        b->ffn_key_q8 = rx_quantize_weight(b->ffn_key_w);
        b->ffn_value_q8 = rx_quantize_weight(b->ffn_value_w);

#ifdef USE_METAL
        if (b->mtl_receptance_w) { fm_buffer_free(b->mtl_receptance_w); b->mtl_receptance_w = NULL; }
        if (b->mtl_key_w) { fm_buffer_free(b->mtl_key_w); b->mtl_key_w = NULL; }
        if (b->mtl_value_w) { fm_buffer_free(b->mtl_value_w); b->mtl_value_w = NULL; }
        if (b->mtl_output_w) { fm_buffer_free(b->mtl_output_w); b->mtl_output_w = NULL; }
        if (b->mtl_ffn_key_w) { fm_buffer_free(b->mtl_ffn_key_w); b->mtl_ffn_key_w = NULL; }
        if (b->mtl_ffn_value_w) { fm_buffer_free(b->mtl_ffn_value_w); b->mtl_ffn_value_w = NULL; }
#endif
        tm_free(b->receptance_w); b->receptance_w = NULL;
        tm_free(b->key_w); b->key_w = NULL;
        tm_free(b->value_w); b->value_w = NULL;
        tm_free(b->output_w); b->output_w = NULL;
        tm_free(b->ffn_key_w); b->ffn_key_w = NULL;
        tm_free(b->ffn_value_w); b->ffn_value_w = NULL;
    }

    /* Quantize head weight */
    m->head_q8 = rx_quantize_weight(m->head_weight);
#ifdef USE_METAL
    if (m->mtl_head_w) { fm_buffer_free(m->mtl_head_w); m->mtl_head_w = NULL; }
#endif
    tm_free(m->head_weight); m->head_weight = NULL;

    m->quantized = true;
    fprintf(stderr, "rwkvx: weights quantized to INT8 (per-row symmetric)\n");
}

rx_model* rx_model_load_q8(const char *path) {
    rx_model *m = rx_model_load(path);
    if (!m) return NULL;
    rx_model_quantize_weights(m);
    return m;
}

void rx_model_free(rx_model *m) {
    if (!m) return;

    if (m->emb_weight) tm_free(m->emb_weight);
    if (m->ln_out_w) tm_free(m->ln_out_w);
    if (m->ln_out_b) tm_free(m->ln_out_b);
    if (m->head_weight) tm_free(m->head_weight);
    rx_q8_free(m->head_q8);
#ifdef USE_METAL
    if (m->mtl_head_w) fm_buffer_free(m->mtl_head_w);
#endif

    for (int i = 0; i < m->config.n_rwkv_layer; i++) {
        rx_rwkv_block *b = &m->rwkv_blocks[i];
        if (b->ln1_w) tm_free(b->ln1_w);
        if (b->ln1_b) tm_free(b->ln1_b);
        if (b->ln2_w) tm_free(b->ln2_w);
        if (b->ln2_b) tm_free(b->ln2_b);
        if (b->x_r) tm_free(b->x_r);
        if (b->x_w) tm_free(b->x_w);
        if (b->x_k) tm_free(b->x_k);
        if (b->x_v) tm_free(b->x_v);
        if (b->x_a) tm_free(b->x_a);
        if (b->x_g) tm_free(b->x_g);
        if (b->w0) tm_free(b->w0);
        if (b->w1) tm_free(b->w1);
        if (b->w2) tm_free(b->w2);
        if (b->a0) tm_free(b->a0);
        if (b->a1) tm_free(b->a1);
        if (b->a2) tm_free(b->a2);
        if (b->v0) tm_free(b->v0);
        if (b->v1) tm_free(b->v1);
        if (b->v2) tm_free(b->v2);
        if (b->g1) tm_free(b->g1);
        if (b->g2) tm_free(b->g2);
        if (b->k_k) tm_free(b->k_k);
        if (b->k_a) tm_free(b->k_a);
        if (b->r_k) tm_free(b->r_k);
        if (b->receptance_w) tm_free(b->receptance_w);
        if (b->key_w) tm_free(b->key_w);
        if (b->value_w) tm_free(b->value_w);
        if (b->output_w) tm_free(b->output_w);
        if (b->ln_x_w) tm_free(b->ln_x_w);
        if (b->ln_x_b) tm_free(b->ln_x_b);
        if (b->ffn_x_k) tm_free(b->ffn_x_k);
        if (b->ffn_key_w) tm_free(b->ffn_key_w);
        if (b->ffn_value_w) tm_free(b->ffn_value_w);
        rx_q8_free(b->receptance_q8);
        rx_q8_free(b->key_q8);
        rx_q8_free(b->value_q8);
        rx_q8_free(b->output_q8);
        rx_q8_free(b->ffn_key_q8);
        rx_q8_free(b->ffn_value_q8);
#ifdef USE_METAL
        if (b->mtl_receptance_w) fm_buffer_free(b->mtl_receptance_w);
        if (b->mtl_key_w) fm_buffer_free(b->mtl_key_w);
        if (b->mtl_value_w) fm_buffer_free(b->mtl_value_w);
        if (b->mtl_output_w) fm_buffer_free(b->mtl_output_w);
        if (b->mtl_ffn_key_w) fm_buffer_free(b->mtl_ffn_key_w);
        if (b->mtl_ffn_value_w) fm_buffer_free(b->mtl_ffn_value_w);
#endif
    }
    free(m->rwkv_blocks);

    for (int i = 0; i < m->config.n_attn_layer; i++) {
        rx_attn_block *b = &m->attn_blocks[i];
        if (b->ln1_w) tm_free(b->ln1_w);
        if (b->ln1_b) tm_free(b->ln1_b);
        if (b->ln2_w) tm_free(b->ln2_w);
        if (b->ln2_b) tm_free(b->ln2_b);
        if (b->receptance_w) tm_free(b->receptance_w);
        if (b->key_w) tm_free(b->key_w);
        if (b->value_w) tm_free(b->value_w);
        if (b->output_w) tm_free(b->output_w);
        if (b->ffn_x_k) tm_free(b->ffn_x_k);
        if (b->ffn_key_w) tm_free(b->ffn_key_w);
        if (b->ffn_value_w) tm_free(b->ffn_value_w);
        rx_q8_free(b->receptance_q8);
        rx_q8_free(b->key_q8);
        rx_q8_free(b->value_q8);
        rx_q8_free(b->output_q8);
        rx_q8_free(b->ffn_key_q8);
        rx_q8_free(b->ffn_value_q8);
#ifdef USE_METAL
        if (b->mtl_receptance_w) fm_buffer_free(b->mtl_receptance_w);
        if (b->mtl_key_w) fm_buffer_free(b->mtl_key_w);
        if (b->mtl_value_w) fm_buffer_free(b->mtl_value_w);
        if (b->mtl_output_w) fm_buffer_free(b->mtl_output_w);
        if (b->mtl_ffn_key_w) fm_buffer_free(b->mtl_ffn_key_w);
        if (b->mtl_ffn_value_w) fm_buffer_free(b->mtl_ffn_value_w);
#endif
    }
    free(m->attn_blocks);

    free(m->block_order);
    free(m->block_index);

#ifdef USE_THREADS
    fw_pool_free(m->pool);
#endif
#ifdef USE_CUDA
    fc_free();
#endif

    free(m);
}

rx_config rx_model_config(const rx_model *m) {
    return m->config;
}

/* ================================================================
 * Public: State management
 * ================================================================ */

#define DEFAULT_MAX_CACHE_LEN 2048

rx_state* rx_state_new(const rx_model *m) {
    if (!m) return NULL;
    rx_config cfg = m->config;
    int c = cfg.n_embd;

    rx_state *s = (rx_state*)calloc(1, sizeof(rx_state));
    s->config = cfg;

    /* RWKV states */
    s->rwkv = (rx_rwkv_state*)calloc(cfg.n_rwkv_layer, sizeof(rx_rwkv_state));
    int64_t shape_c[] = {c};
    int state_size = cfg.n_head * cfg.head_size * cfg.head_size;
    int64_t shape_s[] = {state_size};
    for (int i = 0; i < cfg.n_rwkv_layer; i++) {
        s->rwkv[i].att_x_prev = tm_zeros(shape_c, 1, TM_F32);
        s->rwkv[i].att_kv = tm_zeros(shape_s, 1, TM_F32);
        s->rwkv[i].ffn_x_prev = tm_zeros(shape_c, 1, TM_F32);
    }

    /* Attention states */
    s->attn = (rx_attn_state*)calloc(cfg.n_attn_layer > 0 ? cfg.n_attn_layer : 1,
                                     sizeof(rx_attn_state));
    for (int i = 0; i < cfg.n_attn_layer; i++) {
        int64_t cache_shape[] = {DEFAULT_MAX_CACHE_LEN * c};
        s->attn[i].k_cache = tm_zeros(cache_shape, 1, TM_F32);
        s->attn[i].v_cache = tm_zeros(cache_shape, 1, TM_F32);
        s->attn[i].cache_len = 0;
        s->attn[i].max_cache_len = DEFAULT_MAX_CACHE_LEN;
        s->attn[i].ffn_x_prev = tm_zeros(shape_c, 1, TM_F32);
    }

    /* v_first */
    s->v_first = tm_zeros(shape_c, 1, TM_F32);
    s->v_first_buf = tm_data_f32(s->v_first);

    /* Scratch workspace — scan model for max LoRA and FFN dims */
    int max_lora = 0, max_ffn = 0;
    for (int i = 0; i < cfg.n_rwkv_layer; i++) {
        rx_rwkv_block *blk = &m->rwkv_blocks[i];
        int d;
        d = (int)tm_size(blk->w1, 1); if (d > max_lora) max_lora = d;
        d = (int)tm_size(blk->a1, 1); if (d > max_lora) max_lora = d;
        d = (int)tm_size(blk->g1, 1); if (d > max_lora) max_lora = d;
        if (blk->v1) { d = (int)tm_size(blk->v1, 1); if (d > max_lora) max_lora = d; }
        d = blk->ffn_key_q8 ? blk->ffn_key_q8->rows : (int)tm_size(blk->ffn_key_w, 1);
        if (d > max_ffn) max_ffn = d;
    }
    for (int i = 0; i < cfg.n_attn_layer; i++) {
        rx_attn_block *blk = &m->attn_blocks[i];
        int d = blk->ffn_key_q8 ? blk->ffn_key_q8->rows : (int)tm_size(blk->ffn_key_w, 1);
        if (d > max_ffn) max_ffn = d;
    }
    s->workspace = rx_workspace_new(c, max_lora, max_ffn);

    /* HybridMem disabled by default */
    s->hm_store = NULL;
    s->hm_top_k = 0;
    s->evict_fraction = 0.5f;

    /* Instrumentation disabled by default */
    s->attn_log = NULL;
    s->attn_log_step = 0;
    s->embed_capture = NULL;
    s->embed_capture_count = 0;
    s->embed_capture_max = 0;
    s->embed_capture_layer = -1;

    return s;
}

rx_hybridmem_config rx_hybridmem_config_defaults(void) {
    rx_hybridmem_config cfg;
    cfg.store_dir = NULL;
    cfg.max_cache_len = 0;   /* 0 = use DEFAULT_MAX_CACHE_LEN */
    cfg.top_k = 5;
    cfg.evict_fraction = 0.5f;
    return cfg;
}

rx_state* rx_state_new_with_hybridmem(const rx_model *m,
                                       const rx_hybridmem_config *hm_config) {
    if (!m || !hm_config) return NULL;

    /* Determine cache size */
    int cache_len = hm_config->max_cache_len > 0
                    ? hm_config->max_cache_len
                    : DEFAULT_MAX_CACHE_LEN;

    rx_config cfg = m->config;
    int c = cfg.n_embd;

    rx_state *s = (rx_state*)calloc(1, sizeof(rx_state));
    s->config = cfg;

    /* RWKV states (same as rx_state_new) */
    s->rwkv = (rx_rwkv_state*)calloc(cfg.n_rwkv_layer, sizeof(rx_rwkv_state));
    int64_t shape_c[] = {c};
    int state_size = cfg.n_head * cfg.head_size * cfg.head_size;
    int64_t shape_s[] = {state_size};
    for (int i = 0; i < cfg.n_rwkv_layer; i++) {
        s->rwkv[i].att_x_prev = tm_zeros(shape_c, 1, TM_F32);
        s->rwkv[i].att_kv = tm_zeros(shape_s, 1, TM_F32);
        s->rwkv[i].ffn_x_prev = tm_zeros(shape_c, 1, TM_F32);
    }

    /* Attention states with custom cache size */
    s->attn = (rx_attn_state*)calloc(cfg.n_attn_layer > 0 ? cfg.n_attn_layer : 1,
                                     sizeof(rx_attn_state));
    for (int i = 0; i < cfg.n_attn_layer; i++) {
        int64_t cache_shape[] = {cache_len * c};
        s->attn[i].k_cache = tm_zeros(cache_shape, 1, TM_F32);
        s->attn[i].v_cache = tm_zeros(cache_shape, 1, TM_F32);
        s->attn[i].cache_len = 0;
        s->attn[i].max_cache_len = cache_len;
        s->attn[i].ffn_x_prev = tm_zeros(shape_c, 1, TM_F32);
    }

    /* v_first */
    s->v_first = tm_zeros(shape_c, 1, TM_F32);
    s->v_first_buf = tm_data_f32(s->v_first);

    /* Scratch workspace */
    {
        int max_lora = 0, max_ffn = 0;
        for (int i = 0; i < cfg.n_rwkv_layer; i++) {
            rx_rwkv_block *blk = &m->rwkv_blocks[i];
            int d;
            d = (int)tm_size(blk->w1, 1); if (d > max_lora) max_lora = d;
            d = (int)tm_size(blk->a1, 1); if (d > max_lora) max_lora = d;
            d = (int)tm_size(blk->g1, 1); if (d > max_lora) max_lora = d;
            if (blk->v1) { d = (int)tm_size(blk->v1, 1); if (d > max_lora) max_lora = d; }
            d = blk->ffn_key_q8 ? blk->ffn_key_q8->rows : (int)tm_size(blk->ffn_key_w, 1);
            if (d > max_ffn) max_ffn = d;
        }
        for (int ii = 0; ii < cfg.n_attn_layer; ii++) {
            rx_attn_block *blk = &m->attn_blocks[ii];
            int d = blk->ffn_key_q8 ? blk->ffn_key_q8->rows : (int)tm_size(blk->ffn_key_w, 1);
            if (d > max_ffn) max_ffn = d;
        }
        s->workspace = rx_workspace_new(c, max_lora, max_ffn);
    }

    /* Instrumentation disabled by default */
    s->attn_log = NULL;
    s->attn_log_step = 0;
    s->embed_capture = NULL;
    s->embed_capture_count = 0;
    s->embed_capture_max = 0;
    s->embed_capture_layer = -1;

    /* HybridMem store */
    s->hm_top_k = hm_config->top_k;
    s->evict_fraction = hm_config->evict_fraction;
    if (s->evict_fraction <= 0.0f) s->evict_fraction = 0.5f;
    if (s->evict_fraction > 1.0f) s->evict_fraction = 1.0f;

    if (hm_config->store_dir) {
        fm_config fm_cfg = fm_config_default(c);
        fm_cfg.store_dir = hm_config->store_dir;
        s->hm_store = fm_store_new(&fm_cfg);
        if (!s->hm_store) {
            fprintf(stderr, "rwkvx: failed to create HybridMem store at %s\n",
                    hm_config->store_dir);
            rx_state_free(s);
            return NULL;
        }
        fprintf(stderr, "rwkvx: HybridMem enabled (dir=%s, cache=%d, top_k=%d, evict=%.0f%%)\n",
                hm_config->store_dir, cache_len, s->hm_top_k,
                s->evict_fraction * 100.0f);
    } else {
        s->hm_store = NULL;
    }

    /* Pre-allocate workspace for retrieved K/V data.
     * Worst case: top_k entries, each up to cache_len tokens.
     * Typical case is much smaller, but we size for worst case to
     * guarantee zero malloc in the hot path. */
    if (s->hm_store && s->hm_top_k > 0) {
        int ws_cap = s->hm_top_k * cache_len;
        s->ret_workspace_cap = ws_cap;
        s->ret_k_workspace = (float*)malloc((size_t)ws_cap * c * sizeof(float));
        s->ret_v_workspace = (float*)malloc((size_t)ws_cap * c * sizeof(float));
    }

    /* Per-slot incremental HDC projection accumulators.
     * Each cache slot gets hdc_dim floats to hold R@k_i (pre-threshold).
     * At eviction, sum slots 0..evict_count-1 and threshold → binary sig. */
    if (s->hm_store) {
        int hdc_dim = fv_projector_hdc_dim(fm_store_projector(s->hm_store));
        for (int i = 0; i < cfg.n_attn_layer; i++) {
            s->attn[i].proj_accum_dim = hdc_dim;
            s->attn[i].proj_accum = (float*)calloc(
                (size_t)cache_len * hdc_dim, sizeof(float));
        }
    }

    return s;
}

void rx_state_free(rx_state *s) {
    if (!s) return;
    for (int i = 0; i < s->config.n_rwkv_layer; i++) {
        tm_free(s->rwkv[i].att_x_prev);
        tm_free(s->rwkv[i].att_kv);
        tm_free(s->rwkv[i].ffn_x_prev);
    }
    free(s->rwkv);
    for (int i = 0; i < s->config.n_attn_layer; i++) {
        tm_free(s->attn[i].k_cache);
        tm_free(s->attn[i].v_cache);
        tm_free(s->attn[i].ffn_x_prev);
        free(s->attn[i].proj_accum);
    }
    free(s->attn);
    tm_free(s->v_first);
    rx_workspace_free(s->workspace);
    if (s->hm_store) {
        fm_store_flush(s->hm_store);
        fm_store_free(s->hm_store);
    }
    free(s->ret_k_workspace);
    free(s->ret_v_workspace);
    free(s);
}

int rx_state_flush(rx_state *s) {
    if (!s) return -1;
    if (!s->hm_store) return 0;
    return fm_store_flush(s->hm_store);
}

void rx_state_reset(rx_state *s) {
    if (!s) return;
    rx_config cfg = s->config;

    for (int i = 0; i < cfg.n_rwkv_layer; i++) {
        tm_fill_(s->rwkv[i].att_x_prev, 0.0f);
        tm_fill_(s->rwkv[i].att_kv, 0.0f);
        tm_fill_(s->rwkv[i].ffn_x_prev, 0.0f);
    }
    for (int i = 0; i < cfg.n_attn_layer; i++) {
        tm_fill_(s->attn[i].k_cache, 0.0f);
        tm_fill_(s->attn[i].v_cache, 0.0f);
        s->attn[i].cache_len = 0;
        tm_fill_(s->attn[i].ffn_x_prev, 0.0f);
        if (s->attn[i].proj_accum && s->attn[i].proj_accum_dim > 0) {
            memset(s->attn[i].proj_accum, 0,
                   (size_t)s->attn[i].max_cache_len * s->attn[i].proj_accum_dim * sizeof(float));
        }
    }
    tm_fill_(s->v_first, 0.0f);
    /* Note: HybridMem store is NOT reset — disk data persists across resets */
    /* Reset instrumentation step counter */
    s->attn_log_step = 0;
    s->embed_capture_count = 0;
}

/* ================================================================
 * Public: Instrumentation
 * ================================================================ */

void rx_state_set_attn_log(rx_state *state, FILE *fp) {
    if (!state) return;
    state->attn_log = fp;
    state->attn_log_step = 0;
}

void rx_state_set_embed_capture(rx_state *state, float *buffer,
                                 int max_entries, int layer_idx) {
    if (!state) return;
    state->embed_capture = buffer;
    state->embed_capture_count = 0;
    state->embed_capture_max = max_entries;
    state->embed_capture_layer = layer_idx;
}

int rx_state_embed_capture_count(const rx_state *state) {
    if (!state) return 0;
    return state->embed_capture_count;
}

fm_store* rx_state_get_hm_store(const rx_state *state) {
    if (!state) return NULL;
    return state->hm_store;
}

/* ================================================================
 * Public: Inference
 * ================================================================ */

tm_tensor* rx_forward(rx_model *m, uint32_t token, rx_state *s) {
    if (!m || !s) return NULL;
#ifdef USE_THREADS
    __rx_pool = m->pool;
#endif
    rx_config cfg = m->config;
    int c = cfg.n_embd;
    int vocab = cfg.vocab_size;
    rx_workspace *ws = s->workspace;

    /* Embedding lookup: copy row from emb_weight [vocab, C] into workspace */
    float *emb_data = tm_data_f32(m->emb_weight);
    /* Use two alternating buffers for block input/output (separate from scratch) */
    float *x_cur = ws->x_buf[0];
    float *x_next = ws->x_buf[1];
    memcpy(x_cur, &emb_data[token * c], c * sizeof(float));

    /* Interleaved block execution: RWKV-7 and sparse attention blocks
     * For 0.2B (12 RWKV + 4 ATTN):
     *   [R0,R1,R2,A0, R3,R4,R5,A1, R6,R7,R8,A2, R9,R10,R11,A3] */
    for (int pos = 0; pos < m->n_blocks; pos++) {
        int type = m->block_order[pos];  /* 0=RWKV, 1=ATTN */
        int idx  = m->block_index[pos];

        if (type == 0) {
            rwkv7_block_forward_fast(
                &m->rwkv_blocks[idx], idx, x_cur,
                &s->rwkv[idx], s->v_first_buf,
                cfg.n_head, cfg.head_size, ws, x_next);
        } else {
            attn_block_forward_fast(
                &m->attn_blocks[idx], x_cur, &s->attn[idx],
                s->hm_store, idx,
                s->hm_top_k, s->evict_fraction,
                cfg.n_head, cfg.head_size, ws, x_next,
                s->attn_log, s->attn_log_step,
                s->embed_capture, &s->embed_capture_count,
                s->embed_capture_max, s->embed_capture_layer,
                s->ret_k_workspace, s->ret_v_workspace, s->ret_workspace_cap);
        }
        /* Swap buffers */
        float *tmp = x_cur;
        x_cur = x_next;
        x_next = tmp;
    }

    /* Final layer norm -> head projection -> logits
     * These produce a [vocab]-sized output, so we must allocate a tensor. */
    fast_layer_norm(x_cur, tm_data_f32(m->ln_out_w), tm_data_f32(m->ln_out_b),
                    x_next, c);

    /* Head projection: logits = x_ln @ head_weight [C, vocab] */
    int64_t logits_shape[] = {vocab};
    tm_tensor *logits = tm_empty(logits_shape, 1, TM_F32);
    if (m->head_q8)
        fast_matvec_q8(x_next, m->head_q8, tm_data_f32(logits),
                       m->head_q8->rows, m->head_q8->cols, ws->q8_buf);
    else
        fast_matvec(x_next, tm_data_f32(m->head_weight),
                    tm_data_f32(logits), c, vocab);

    s->attn_log_step++;
    return logits;
}

/* Extended forward with block callbacks (for training) */
tm_tensor* rx_forward_ex(rx_model *m, uint32_t token, rx_state *s,
                          rx_block_callback cb, void *user_data) {
    if (!m || !s) return NULL;
    if (!cb) return rx_forward(m, token, s);

#ifdef USE_THREADS
    __rx_pool = m->pool;
#endif
    rx_config cfg = m->config;
    int c = cfg.n_embd;
    int vocab = cfg.vocab_size;
    rx_workspace *ws = s->workspace;

    /* Embedding lookup */
    float *emb_data = tm_data_f32(m->emb_weight);
    float *x_cur = ws->x_buf[0];
    float *x_next = ws->x_buf[1];
    memcpy(x_cur, &emb_data[token * c], c * sizeof(float));

    /* Block execution with callbacks */
    for (int pos = 0; pos < m->n_blocks; pos++) {
        int type = m->block_order[pos];
        int idx  = m->block_index[pos];

        if (type == 0) {
            /* RWKV-7 block: standard forward */
            rwkv7_block_forward_fast(
                &m->rwkv_blocks[idx], idx, x_cur,
                &s->rwkv[idx], s->v_first_buf,
                cfg.n_head, cfg.head_size, ws, x_next);
        } else {
            /* Attention block: call pre-callback */
            int skip = cb(0, idx, x_cur, user_data);
            if (skip) {
                /* Caller handled this block — x_cur was modified in-place */
                memcpy(x_next, x_cur, c * sizeof(float));
            } else {
                /* Standard attention forward */
                attn_block_forward_fast(
                    &m->attn_blocks[idx], x_cur, &s->attn[idx],
                    s->hm_store, idx,
                    s->hm_top_k, s->evict_fraction,
                    cfg.n_head, cfg.head_size, ws, x_next,
                    s->attn_log, s->attn_log_step,
                    s->embed_capture, &s->embed_capture_count,
                    s->embed_capture_max, s->embed_capture_layer,
                    s->ret_k_workspace, s->ret_v_workspace, s->ret_workspace_cap);
            }
            /* Post-callback */
            cb(1, idx, x_next, user_data);
        }
        /* Swap buffers */
        float *tmp = x_cur;
        x_cur = x_next;
        x_next = tmp;
    }

    /* Final layer norm + head projection */
    fast_layer_norm(x_cur, tm_data_f32(m->ln_out_w), tm_data_f32(m->ln_out_b),
                    x_next, c);

    int64_t logits_shape[] = {vocab};
    tm_tensor *logits = tm_empty(logits_shape, 1, TM_F32);
    if (m->head_q8)
        fast_matvec_q8(x_next, m->head_q8, tm_data_f32(logits),
                       m->head_q8->rows, m->head_q8->cols, ws->q8_buf);
    else
        fast_matvec(x_next, tm_data_f32(m->head_weight),
                    tm_data_f32(logits), c, vocab);

    s->attn_log_step++;
    return logits;
}

/* Get KV cache pointers from state for a given attention layer */
int rx_state_get_kv_cache(const rx_state *state, int attn_idx,
                           const float **k_cache_out,
                           const float **v_cache_out) {
    if (!state || attn_idx < 0 || attn_idx >= state->config.n_attn_layer)
        return 0;
    if (k_cache_out) *k_cache_out = tm_data_f32(state->attn[attn_idx].k_cache);
    if (v_cache_out) *v_cache_out = tm_data_f32(state->attn[attn_idx].v_cache);
    return state->attn[attn_idx].cache_len;
}

/* ================================================================
 * Weight accessors for training
 * ================================================================ */

int rx_model_get_attn_weights(const rx_model *model, int attn_idx,
                               rx_attn_weights *out) {
    if (!model || !out || attn_idx < 0 || attn_idx >= model->config.n_attn_layer)
        return -1;
    const rx_attn_block *b = &model->attn_blocks[attn_idx];
    out->ln1_w = tm_data_f32(b->ln1_w);
    out->ln1_b = tm_data_f32(b->ln1_b);
    out->ln2_w = tm_data_f32(b->ln2_w);
    out->ln2_b = tm_data_f32(b->ln2_b);
    out->receptance_w = tm_data_f32(b->receptance_w);
    out->key_w = tm_data_f32(b->key_w);
    out->value_w = tm_data_f32(b->value_w);
    out->output_w = tm_data_f32(b->output_w);
    out->ffn_x_k = tm_data_f32(b->ffn_x_k);
    out->ffn_key_w = tm_data_f32(b->ffn_key_w);
    out->ffn_value_w = tm_data_f32(b->ffn_value_w);
    out->ffn_dim = (int)tm_size(b->ffn_key_w, 1);
    return 0;
}

const float* rx_model_get_head_weight(const rx_model *model) {
    if (!model || !model->head_weight) return NULL;
    return tm_data_f32(model->head_weight);
}

int rx_model_get_ln_out(const rx_model *model,
                         const float **w_out, const float **b_out) {
    if (!model) return -1;
    if (w_out) *w_out = tm_data_f32(model->ln_out_w);
    if (b_out) *b_out = tm_data_f32(model->ln_out_b);
    return 0;
}

tm_tensor* rx_forward_batch(rx_model *m, const uint32_t *tokens,
                            int n_tokens, rx_state *s) {
    if (!m || !s || n_tokens <= 0) return NULL;

    /* For very short sequences, fall back to serial */
    if (n_tokens <= 2) {
        tm_tensor *logits = NULL;
        for (int i = 0; i < n_tokens; i++) {
            if (logits) tm_free(logits);
            logits = rx_forward(m, tokens[i], s);
        }
        return logits;
    }

#ifdef USE_METAL
    fm_set_auto_batch(true);  /* Accumulate GPU work, sync at dependency points */
#endif

#ifdef USE_THREADS
    __rx_pool = m->pool;
#endif
    rx_config cfg = m->config;
    int c = cfg.n_embd;
    int vocab = cfg.vocab_size;
    rx_workspace *ws = s->workspace;

    /* Determine FFN dim from first RWKV block */
    int ffn_dim = ws->ffn_dim;

    /* Allocate batch workspace */
    rx_batch_workspace *bws = rx_batch_workspace_new(n_tokens, c, ffn_dim);

    /* Embedding lookup for all tokens */
    float *emb = tm_data_f32(m->emb_weight);
    for (int t = 0; t < n_tokens; t++)
        memcpy(&bws->x[0][t * c], &emb[tokens[t] * c], c * sizeof(float));

    /* Process each block for ALL tokens */
    int cur = 0;
    for (int pos = 0; pos < m->n_blocks; pos++) {
        int type = m->block_order[pos];
        int idx  = m->block_index[pos];
        int nxt = cur ^ 1;

        if (type == 0) {
            rwkv7_block_forward_batch(
                &m->rwkv_blocks[idx], idx,
                bws->x[cur], &s->rwkv[idx],
                bws->v_first,
                cfg.n_head, cfg.head_size, ws, bws, bws->x[nxt],
                MTL_BUF(bws->mtl_x[nxt]));
        } else {
            attn_block_forward_batch(
                &m->attn_blocks[idx],
                bws->x[cur], &s->attn[idx],
                s->hm_store, idx,
                s->hm_top_k, s->evict_fraction,
                cfg.n_head, cfg.head_size, bws, bws->x[nxt],
                MTL_BUF(bws->mtl_x[nxt]),
                s->attn_log, &s->attn_log_step,
                s->embed_capture, &s->embed_capture_count,
                s->embed_capture_max, s->embed_capture_layer,
                s->ret_k_workspace, s->ret_v_workspace, s->ret_workspace_cap);
        }
        cur = nxt;
    }

    /* Persist v_first from last token for future incremental decoding */
    memcpy(s->v_first_buf, &bws->v_first[(n_tokens - 1) * c], c * sizeof(float));

    /* Final LN + head projection — LAST TOKEN ONLY */
    float *x_last = &bws->x[cur][(n_tokens - 1) * c];
    float ln_out[c];
    fast_layer_norm(x_last, tm_data_f32(m->ln_out_w), tm_data_f32(m->ln_out_b),
                    ln_out, c);

    int64_t logits_shape[] = {vocab};
    tm_tensor *logits = tm_empty(logits_shape, 1, TM_F32);
    if (m->head_q8)
        fast_matvec_q8(ln_out, m->head_q8, tm_data_f32(logits),
                       m->head_q8->rows, m->head_q8->cols, ws->q8_buf);
    else
        fast_matvec(ln_out, tm_data_f32(m->head_weight),
                    tm_data_f32(logits), c, vocab);

#ifdef USE_METAL
    fm_set_auto_batch(false);  /* Back to immediate mode */
#endif

    rx_batch_workspace_free(bws);
    return logits;
}

/* ================================================================
 * Tokenizer
 * ================================================================ */

static rx_trie_node* trie_node_new(void) {
    rx_trie_node *n = (rx_trie_node*)calloc(1, sizeof(rx_trie_node));
    n->token_id = -1;
    return n;
}

static void trie_node_free(rx_trie_node *n) {
    if (!n) return;
    for (int i = 0; i < 256; i++)
        trie_node_free(n->children[i]);
    free(n);
}

static void trie_insert(rx_trie_node *root, const uint8_t *key, int len, int32_t token_id) {
    rx_trie_node *node = root;
    for (int i = 0; i < len; i++) {
        int idx = key[i];
        if (!node->children[idx])
            node->children[idx] = trie_node_new();
        node = node->children[idx];
    }
    node->token_id = token_id;
}

/* Parse a Python string/bytes literal into raw bytes.
 * Returns number of bytes written, or -1 on error. */
static int parse_python_literal(const char *s, uint8_t *out, int max_out) {
    const char *p = s;
    bool is_bytes = false;
    int out_len = 0;

    /* Skip leading whitespace */
    while (*p == ' ' || *p == '\t') p++;

    /* Detect b'...' or b"..." */
    if (*p == 'b') {
        is_bytes = true;
        p++;
    }

    /* Get quote char */
    char quote = *p;
    if (quote != '\'' && quote != '"') return -1;
    p++;

    /* Find end quote */
    const char *end = p;
    while (*end && *end != quote) {
        if (*end == '\\') end++; /* skip escaped char */
        if (*end) end++;
    }
    if (*end != quote) return -1;

    /* Parse content between quotes */
    while (p < end && out_len < max_out) {
        if (*p == '\\' && p + 1 < end) {
            switch (p[1]) {
                case 'x':
                    if (p + 3 < end) {
                        char hex[3] = {p[2], p[3], 0};
                        out[out_len++] = (uint8_t)strtol(hex, NULL, 16);
                        p += 4;
                    } else {
                        return -1;
                    }
                    break;
                case 'n': out[out_len++] = '\n'; p += 2; break;
                case 'r': out[out_len++] = '\r'; p += 2; break;
                case 't': out[out_len++] = '\t'; p += 2; break;
                case '\\': out[out_len++] = '\\'; p += 2; break;
                case '\'': out[out_len++] = '\''; p += 2; break;
                case '"': out[out_len++] = '"'; p += 2; break;
                case '0': out[out_len++] = 0; p += 2; break;
                default:
                    out[out_len++] = '\\';
                    p++;
                    break;
            }
        } else {
            if (is_bytes) {
                out[out_len++] = (uint8_t)*p;
            } else {
                /* UTF-8 encode the character */
                unsigned char ch = (unsigned char)*p;
                /* For the RWKV vocab, most entries are either ASCII or \x escapes.
                 * Handle multi-byte UTF-8 by copying raw bytes. */
                out[out_len++] = ch;
            }
            p++;
        }
    }
    return out_len;
}

rx_tokenizer* rx_tokenizer_new(const char *vocab_path) {
    FILE *fp = fopen(vocab_path, "r");
    if (!fp) {
        fprintf(stderr, "rwkvx: failed to open vocab: %s\n", vocab_path);
        return NULL;
    }

    rx_tokenizer *tok = (rx_tokenizer*)calloc(1, sizeof(rx_tokenizer));
    tok->root = trie_node_new();

    /* Token 0 = <|endoftext|> */
    const char *eot = "<|endoftext|>";
    int eot_len = (int)strlen(eot);
    tok->token_bytes[0] = (uint8_t*)malloc(eot_len);
    memcpy(tok->token_bytes[0], eot, eot_len);
    tok->token_lens[0] = eot_len;

    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        /* Strip newline */
        int ll = (int)strlen(line);
        while (ll > 0 && (line[ll-1] == '\n' || line[ll-1] == '\r')) line[--ll] = 0;
        if (ll == 0) continue;

        /* Parse: "token_id python_literal byte_length" */
        char *first_space = strchr(line, ' ');
        if (!first_space) continue;
        char *last_space = strrchr(line, ' ');
        if (last_space == first_space) continue;

        int idx = atoi(line);
        if (idx < 0 || idx >= RX_VOCAB_SIZE) continue;

        /* Extract the literal between first_space+1 and last_space */
        *last_space = 0;
        const char *literal = first_space + 1;

        uint8_t bytes[RX_MAX_TOKEN_LEN];
        int blen = parse_python_literal(literal, bytes, RX_MAX_TOKEN_LEN);
        if (blen < 0) continue;

        trie_insert(tok->root, bytes, blen, idx);
        tok->token_bytes[idx] = (uint8_t*)malloc(blen);
        memcpy(tok->token_bytes[idx], bytes, blen);
        tok->token_lens[idx] = blen;
    }

    fclose(fp);
    return tok;
}

void rx_tokenizer_free(rx_tokenizer *tok) {
    if (!tok) return;
    trie_node_free(tok->root);
    for (int i = 0; i < RX_VOCAB_SIZE; i++)
        free(tok->token_bytes[i]);
    free(tok);
}

int rx_tokenizer_encode(const rx_tokenizer *tok, const char *text,
                        uint32_t *tokens_out, int max_tokens)
{
    if (!tok || !text) return 0;
    const uint8_t *src = (const uint8_t*)text;
    int src_len = (int)strlen(text);
    int n = 0;
    int i = 0;

    while (i < src_len && n < max_tokens) {
        rx_trie_node *node = tok->root;
        int last_match_pos = i;
        int32_t last_match_id = -1;
        int j = i;

        while (j < src_len) {
            int byte = src[j];
            if (!node->children[byte]) break;
            node = node->children[byte];
            j++;
            if (node->token_id >= 0) {
                last_match_pos = j;
                last_match_id = node->token_id;
            }
        }

        if (last_match_id >= 0) {
            tokens_out[n++] = (uint32_t)last_match_id;
            i = last_match_pos;
        } else {
            /* Single byte fallback: tokens 1-256 */
            tokens_out[n++] = (uint32_t)(src[i] + 1);
            i++;
        }
    }
    return n;
}

char* rx_tokenizer_decode(const rx_tokenizer *tok, const uint32_t *tokens,
                          int n_tokens)
{
    if (!tok || !tokens) return NULL;

    /* Calculate total length */
    int total = 0;
    for (int i = 0; i < n_tokens; i++) {
        uint32_t tid = tokens[i];
        if (tid < RX_VOCAB_SIZE)
            total += tok->token_lens[tid];
    }

    char *out = (char*)malloc(total + 1);
    int pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        uint32_t tid = tokens[i];
        if (tid < RX_VOCAB_SIZE && tok->token_bytes[tid]) {
            memcpy(&out[pos], tok->token_bytes[tid], tok->token_lens[tid]);
            pos += tok->token_lens[tid];
        }
    }
    out[pos] = 0;
    return out;
}

/* ================================================================
 * Public: Sampling
 * ================================================================ */

rx_sampling_params rx_sampling_defaults(void) {
    rx_sampling_params p;
    p.temperature = 1.0f;
    p.top_p = 0.7f;
    p.top_k = 100;
    p.frequency_penalty = 0.2f;
    p.presence_penalty = 0.1f;
    return p;
}

/* Comparison function for sorting (index, logit) pairs descending by logit */
typedef struct { int idx; float val; } rx_indexed_logit;

static int cmp_logit_desc(const void *a, const void *b) {
    float va = ((const rx_indexed_logit*)a)->val;
    float vb = ((const rx_indexed_logit*)b)->val;
    if (vb > va) return 1;
    if (vb < va) return -1;
    return 0;
}

uint32_t rx_sample(const tm_tensor *logits, rx_sampling_params *params,
                   const uint32_t *token_counts, int vocab_size)
{
    float *ld = tm_data_f32((tm_tensor*)logits);
    int vs = vocab_size > 0 ? vocab_size : (int)tm_numel((tm_tensor*)logits);

    /* Work on a copy */
    float *work = (float*)malloc(vs * sizeof(float));
    memcpy(work, ld, vs * sizeof(float));

    /* Apply penalties */
    if (token_counts && (params->frequency_penalty != 0.0f || params->presence_penalty != 0.0f)) {
        for (int i = 0; i < vs; i++) {
            if (token_counts[i] > 0) {
                work[i] -= params->frequency_penalty * (float)token_counts[i];
                work[i] -= params->presence_penalty;
            }
        }
    }

    /* Temperature scaling */
    if (params->temperature != 1.0f && params->temperature > 0.0f) {
        float inv_t = 1.0f / params->temperature;
        for (int i = 0; i < vs; i++)
            work[i] *= inv_t;
    }

    /* Greedy */
    if (params->temperature <= 0.0f) {
        int max_i = 0;
        for (int i = 1; i < vs; i++)
            if (work[i] > work[max_i]) max_i = i;
        free(work);
        return (uint32_t)max_i;
    }

    /* Sort by logit descending */
    rx_indexed_logit *sorted = (rx_indexed_logit*)malloc(vs * sizeof(rx_indexed_logit));
    for (int i = 0; i < vs; i++) {
        sorted[i].idx = i;
        sorted[i].val = work[i];
    }
    qsort(sorted, vs, sizeof(rx_indexed_logit), cmp_logit_desc);

    /* Top-k */
    int n_cand = vs;
    if (params->top_k > 0 && params->top_k < vs)
        n_cand = params->top_k;

    /* Softmax over candidates */
    float max_l = sorted[0].val;
    float sum = 0.0f;
    for (int i = 0; i < n_cand; i++) {
        sorted[i].val = expf(sorted[i].val - max_l);
        sum += sorted[i].val;
    }
    for (int i = 0; i < n_cand; i++)
        sorted[i].val /= sum;

    /* Top-p */
    if (params->top_p > 0.0f && params->top_p < 1.0f) {
        float cumsum = 0.0f;
        int cutoff = n_cand;
        for (int i = 0; i < n_cand; i++) {
            cumsum += sorted[i].val;
            if (cumsum > params->top_p) {
                cutoff = i + 1;
                break;
            }
        }
        n_cand = cutoff;

        /* Re-normalize */
        sum = 0.0f;
        for (int i = 0; i < n_cand; i++) sum += sorted[i].val;
        for (int i = 0; i < n_cand; i++) sorted[i].val /= sum;
    }

    /* Random sample */
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    uint32_t result = (uint32_t)sorted[n_cand - 1].idx;
    for (int i = 0; i < n_cand; i++) {
        cumsum += sorted[i].val;
        if (r < cumsum) {
            result = (uint32_t)sorted[i].idx;
            break;
        }
    }

    free(work);
    free(sorted);
    return result;
}
