/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_TENSOR_H
#define FOUNDRY_TENSOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tensor operations with automatic differentiation.
 * Replaces PyTorch core functionality.
 */

#define TM_MAX_DIMS 8

/* Data types */
typedef enum {
    TM_F32,     /* float32 - default */
    TM_F64,     /* float64 */
    TM_F16,     /* float16 */
    TM_BF16,    /* bfloat16 */
    TM_I32,     /* int32 */
    TM_I64,     /* int64 */
} tm_dtype;

/* Sparse formats */
typedef enum {
    TM_SPARSE_COO,   /* Coordinate format: (row, col, value) triples */
    TM_SPARSE_CSR,   /* Compressed Sparse Row: row_ptr, col_ind, values */
} tm_sparse_format;

/* Forward declarations */
typedef struct tm_tensor tm_tensor;
typedef struct tm_grad_fn tm_grad_fn;

/* Gradient function - computes gradients during backward pass */
struct tm_grad_fn {
    void (*backward)(tm_tensor* out);  /* Backward function */
    tm_tensor* inputs[2];               /* Input tensors (for binary ops) */
    int num_inputs;
    void* ctx;                          /* Optional context */
};

/* Tensor structure */
struct tm_tensor {
    void* data;                         /* Raw data pointer */
    int64_t shape[TM_MAX_DIMS];        /* Shape of each dimension */
    int64_t strides[TM_MAX_DIMS];      /* Strides in elements */
    int ndim;                           /* Number of dimensions */
    int64_t numel;                      /* Total number of elements */
    tm_dtype dtype;                     /* Data type */

    /* Autograd */
    bool requires_grad;                 /* Track gradients? */
    tm_tensor* grad;                    /* Gradient tensor */
    tm_grad_fn* grad_fn;               /* How to compute gradient */

    /* Memory management */
    int refcount;                       /* Reference count */
    bool owns_data;                     /* Do we own the data? */

    /* Sparse tensor support */
    bool is_sparse;                     /* Is this a sparse tensor? */
    tm_sparse_format sparse_format;    /* COO or CSR */
    int64_t nnz;                        /* Number of non-zero elements */
    int64_t* indices;                   /* COO: [2, nnz] for 2D (row, col indices) */
    int64_t* row_ptr;                   /* CSR: [rows + 1] row pointers */
    int64_t* col_ind;                   /* CSR: [nnz] column indices */
};

/* ============================================================= */
/* Tensor Creation                                               */
/* ============================================================= */

/* Create tensor with uninitialized data */
tm_tensor* tm_empty(int64_t* shape, int ndim, tm_dtype dtype);

/* Create tensor filled with zeros */
tm_tensor* tm_zeros(int64_t* shape, int ndim, tm_dtype dtype);

/* Create tensor filled with ones */
tm_tensor* tm_ones(int64_t* shape, int ndim, tm_dtype dtype);

/* Create tensor filled with value */
tm_tensor* tm_full(int64_t* shape, int ndim, tm_dtype dtype, double value);

/* Create tensor from existing data (does not copy) */
tm_tensor* tm_from_data(void* data, int64_t* shape, int ndim, tm_dtype dtype);

/* Create tensor with random normal values */
tm_tensor* tm_randn(int64_t* shape, int ndim, tm_dtype dtype);

/* Create tensor with random uniform values [0, 1) */
tm_tensor* tm_rand(int64_t* shape, int ndim, tm_dtype dtype);

/* Clone tensor (deep copy) */
tm_tensor* tm_clone(tm_tensor* t);

/* Create tensor with range of values [start, end) with step */
tm_tensor* tm_arange(double start, double end, double step, tm_dtype dtype);

/* Create tensor with random integers in [low, high) */
tm_tensor* tm_randint(int64_t low, int64_t high, int64_t* shape, int ndim);

/* Create identity matrix */
tm_tensor* tm_eye(int64_t n, tm_dtype dtype);

/* Create linearly spaced values */
tm_tensor* tm_linspace(double start, double end, int64_t steps, tm_dtype dtype);

/* Like functions - create tensor with same shape/dtype as input */
tm_tensor* tm_full_like(tm_tensor* t, double value);
tm_tensor* tm_empty_like(tm_tensor* t);
tm_tensor* tm_rand_like(tm_tensor* t);
tm_tensor* tm_randn_like(tm_tensor* t);

/* ============================================================= */
/* Memory Management                                             */
/* ============================================================= */

/* Increment reference count */
tm_tensor* tm_retain(tm_tensor* t);

/* Decrement reference count, free if zero */
void tm_release(tm_tensor* t);

/* Convenience alias */
#define tm_free(t) tm_release(t)

/* ============================================================= */
/* Accessors                                                     */
/* ============================================================= */

/* Get element at index (1D) */
float tm_get_f32(tm_tensor* t, int64_t i);
double tm_get_f64(tm_tensor* t, int64_t i);
int32_t tm_get_i32(tm_tensor* t, int64_t i);

/* Set element at index (1D) */
void tm_set_f32(tm_tensor* t, int64_t i, float val);
void tm_set_f64(tm_tensor* t, int64_t i, double val);
void tm_set_i32(tm_tensor* t, int64_t i, int32_t val);

/* Get pointer to data */
float* tm_data_f32(tm_tensor* t);
double* tm_data_f64(tm_tensor* t);

/* Get size info */
int64_t tm_numel(tm_tensor* t);
int tm_ndim(tm_tensor* t);
int64_t tm_size(tm_tensor* t, int dim);
size_t tm_dtype_size(tm_dtype dtype);

/* ============================================================= */
/* Shape Operations                                              */
/* ============================================================= */

/* Reshape (returns new view if possible, otherwise copies) */
tm_tensor* tm_reshape(tm_tensor* t, int64_t* shape, int ndim);

/* View (alias for reshape, for PyTorch compatibility) */
tm_tensor* tm_view(tm_tensor* t, int64_t* shape, int ndim);

/* Transpose (swap two dimensions) */
tm_tensor* tm_transpose(tm_tensor* t, int dim0, int dim1);

/* Permute dimensions (generalized transpose) */
tm_tensor* tm_permute(tm_tensor* t, int* dims, int ndim);

/* Squeeze (remove dimensions of size 1) */
tm_tensor* tm_squeeze(tm_tensor* t, int dim);

/* Unsqueeze (add dimension of size 1) */
tm_tensor* tm_unsqueeze(tm_tensor* t, int dim);

/* Flatten dimensions from start_dim to end_dim (inclusive) */
tm_tensor* tm_flatten(tm_tensor* t, int start_dim, int end_dim);

/* Select along dimension (reduces ndim by 1) */
tm_tensor* tm_select(tm_tensor* t, int dim, int64_t index);

/* Contiguous (ensure data is contiguous in memory) */
tm_tensor* tm_contiguous(tm_tensor* t);

/* Check if contiguous */
bool tm_is_contiguous(tm_tensor* t);

/* Concatenate tensors along dimension */
tm_tensor* tm_cat(tm_tensor** tensors, int num_tensors, int dim);

/* Stack tensors along new dimension */
tm_tensor* tm_stack(tm_tensor** tensors, int num_tensors, int dim);

/* Split tensor into chunks along dimension */
tm_tensor** tm_split(tm_tensor* t, int64_t split_size, int dim, int* num_chunks);

/* Split tensor into n equal chunks along dimension */
tm_tensor** tm_chunk(tm_tensor* t, int num_chunks, int dim, int* actual_chunks);

/* Narrow (slice) along dimension */
tm_tensor* tm_narrow(tm_tensor* t, int dim, int64_t start, int64_t length);

/* Expand tensor to new shape (broadcasting) */
tm_tensor* tm_expand(tm_tensor* t, int64_t* shape, int ndim);

/* Repeat tensor along dimensions */
tm_tensor* tm_repeat(tm_tensor* t, int64_t* repeats, int ndim);

/* Flip tensor along dimensions */
tm_tensor* tm_flip(tm_tensor* t, int* dims, int ndim);

/* Roll (circular shift) tensor */
tm_tensor* tm_roll(tm_tensor* t, int64_t shift, int dim);

/* 2D transpose alias */
tm_tensor* tm_t(tm_tensor* t);

/* ============================================================= */
/* Indexing Operations                                           */
/* ============================================================= */

/* Gather elements along dimension using index tensor */
tm_tensor* tm_gather(tm_tensor* t, int dim, tm_tensor* index);

/* Scatter elements along dimension using index tensor */
tm_tensor* tm_scatter(tm_tensor* t, int dim, tm_tensor* index, tm_tensor* src);

/* Select elements along dimension using index tensor */
tm_tensor* tm_index_select(tm_tensor* t, int dim, tm_tensor* index);

/* Scatter with addition */
tm_tensor* tm_scatter_add(tm_tensor* t, int dim, tm_tensor* index, tm_tensor* src);

/* Select elements where mask is true */
tm_tensor* tm_masked_select(tm_tensor* t, tm_tensor* mask);

/* Get indices of non-zero elements */
tm_tensor* tm_nonzero(tm_tensor* t);

/* Fill at indices */
tm_tensor* tm_index_fill(tm_tensor* t, int dim, tm_tensor* index, float value);

/* Copy to indices */
tm_tensor* tm_index_copy(tm_tensor* t, int dim, tm_tensor* index, tm_tensor* src);

/* Add at indices */
tm_tensor* tm_index_add(tm_tensor* t, int dim, tm_tensor* index, tm_tensor* src);

/* Take elements by flat indices */
tm_tensor* tm_take(tm_tensor* t, tm_tensor* indices);

/* ============================================================= */
/* Element-wise Operations                                       */
/* ============================================================= */

/* Binary ops */
tm_tensor* tm_add(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_sub(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_mul(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_div(tm_tensor* a, tm_tensor* b);

/* Scalar ops */
tm_tensor* tm_add_scalar(tm_tensor* a, double s);
tm_tensor* tm_sub_scalar(tm_tensor* a, double s);
tm_tensor* tm_mul_scalar(tm_tensor* a, double s);
tm_tensor* tm_div_scalar(tm_tensor* a, double s);

/* Unary ops */
tm_tensor* tm_neg(tm_tensor* a);
tm_tensor* tm_abs(tm_tensor* a);
tm_tensor* tm_exp(tm_tensor* a);
tm_tensor* tm_log(tm_tensor* a);
tm_tensor* tm_sqrt(tm_tensor* a);
tm_tensor* tm_rsqrt(tm_tensor* a);
tm_tensor* tm_pow(tm_tensor* a, double exponent);
tm_tensor* tm_sin(tm_tensor* a);
tm_tensor* tm_cos(tm_tensor* a);
tm_tensor* tm_tanh(tm_tensor* a);
tm_tensor* tm_relu(tm_tensor* a);
tm_tensor* tm_sigmoid(tm_tensor* a);
tm_tensor* tm_gelu(tm_tensor* a);
tm_tensor* tm_silu(tm_tensor* a);

/* Clamp values */
tm_tensor* tm_clamp(tm_tensor* a, double min_val, double max_val);

/* Conditional operations */
tm_tensor* tm_where(tm_tensor* condition, tm_tensor* a, tm_tensor* b);
tm_tensor* tm_masked_fill(tm_tensor* t, tm_tensor* mask, float value);

/* Element-wise max/min */
tm_tensor* tm_maximum(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_minimum(tm_tensor* a, tm_tensor* b);

/* Comparison operations (return boolean tensors as TM_I32) */
tm_tensor* tm_eq(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_ne(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_lt(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_le(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_gt(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_ge(tm_tensor* a, tm_tensor* b);

/* Check for special values */
tm_tensor* tm_isnan(tm_tensor* t);
tm_tensor* tm_isinf(tm_tensor* t);

/* Leaky ReLU activation */
tm_tensor* tm_leaky_relu(tm_tensor* a, float negative_slope);

/* ELU activation */
tm_tensor* tm_elu(tm_tensor* a, float alpha);

/* Rounding operations */
tm_tensor* tm_floor(tm_tensor* a);
tm_tensor* tm_ceil(tm_tensor* a);
tm_tensor* tm_round(tm_tensor* a);
tm_tensor* tm_trunc(tm_tensor* a);

/* Sign and reciprocal */
tm_tensor* tm_sign(tm_tensor* a);
tm_tensor* tm_reciprocal(tm_tensor* a);

/* Additional math functions */
tm_tensor* tm_log2(tm_tensor* a);
tm_tensor* tm_log10(tm_tensor* a);
tm_tensor* tm_exp2(tm_tensor* a);
tm_tensor* tm_expm1(tm_tensor* a);
tm_tensor* tm_log1p(tm_tensor* a);
tm_tensor* tm_tan(tm_tensor* a);
tm_tensor* tm_asin(tm_tensor* a);
tm_tensor* tm_acos(tm_tensor* a);
tm_tensor* tm_atan(tm_tensor* a);
tm_tensor* tm_sinh(tm_tensor* a);
tm_tensor* tm_cosh(tm_tensor* a);

/* Fused operations */
tm_tensor* tm_addcmul(tm_tensor* t, tm_tensor* a, tm_tensor* b, double value);
tm_tensor* tm_addcdiv(tm_tensor* t, tm_tensor* a, tm_tensor* b, double value);
tm_tensor* tm_lerp(tm_tensor* a, tm_tensor* b, double weight);

/* Logical operations */
tm_tensor* tm_logical_and(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_logical_or(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_logical_not(tm_tensor* a);
tm_tensor* tm_logical_xor(tm_tensor* a, tm_tensor* b);

/* Check functions */
tm_tensor* tm_isfinite(tm_tensor* t);
bool tm_equal(tm_tensor* a, tm_tensor* b);
bool tm_allclose(tm_tensor* a, tm_tensor* b, double rtol, double atol);

/* ============================================================= */
/* Matrix Operations                                             */
/* ============================================================= */

/* Matrix multiplication */
tm_tensor* tm_matmul(tm_tensor* a, tm_tensor* b);

/* Batched matrix multiplication */
tm_tensor* tm_bmm(tm_tensor* a, tm_tensor* b);

/* Matrix multiplication aliases */
tm_tensor* tm_mm(tm_tensor* a, tm_tensor* b);

/* Fused t + a @ b */
tm_tensor* tm_addmm(tm_tensor* t, tm_tensor* a, tm_tensor* b, double beta, double alpha);

/* Vector operations */
tm_tensor* tm_dot(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_mv(tm_tensor* mat, tm_tensor* vec);
tm_tensor* tm_outer(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_inner(tm_tensor* a, tm_tensor* b);

/* ============================================================= */
/* Reduction Operations                                          */
/* ============================================================= */

/* Sum over dimension (dim=-1 means all) */
tm_tensor* tm_sum(tm_tensor* t, int dim, bool keepdim);

/* Mean over dimension */
tm_tensor* tm_mean(tm_tensor* t, int dim, bool keepdim);

/* Max over dimension */
tm_tensor* tm_max(tm_tensor* t, int dim, bool keepdim);

/* Min over dimension */
tm_tensor* tm_min(tm_tensor* t, int dim, bool keepdim);

/* Argmax - index of maximum value */
tm_tensor* tm_argmax(tm_tensor* t, int dim, bool keepdim);

/* Argmin - index of minimum value */
tm_tensor* tm_argmin(tm_tensor* t, int dim, bool keepdim);

/* Log-sum-exp (numerically stable) */
tm_tensor* tm_logsumexp(tm_tensor* t, int dim, bool keepdim);

/* Cumulative sum along dimension */
tm_tensor* tm_cumsum(tm_tensor* t, int dim);

/* Standard deviation and variance */
tm_tensor* tm_std(tm_tensor* t, int dim, bool keepdim, bool unbiased);
tm_tensor* tm_var(tm_tensor* t, int dim, bool keepdim, bool unbiased);

/* Product over dimension */
tm_tensor* tm_prod(tm_tensor* t, int dim, bool keepdim);

/* Cumulative product along dimension */
tm_tensor* tm_cumprod(tm_tensor* t, int dim);

/* Any/all along dimension */
tm_tensor* tm_any(tm_tensor* t, int dim, bool keepdim);
tm_tensor* tm_all(tm_tensor* t, int dim, bool keepdim);

/* Lp norm */
tm_tensor* tm_norm(tm_tensor* t, double p, int dim, bool keepdim);

/* Count non-zero elements */
tm_tensor* tm_count_nonzero(tm_tensor* t, int dim);

/* ============================================================= */
/* Type Conversion                                               */
/* ============================================================= */

/* Convert tensor to different dtype */
tm_tensor* tm_to(tm_tensor* t, tm_dtype dtype);

/* Convenience type conversion functions */
tm_tensor* tm_float(tm_tensor* t);   /* Convert to TM_F32 */
tm_tensor* tm_double(tm_tensor* t);  /* Convert to TM_F64 */
tm_tensor* tm_half(tm_tensor* t);    /* Convert to TM_F16 */
tm_tensor* tm_int(tm_tensor* t);     /* Convert to TM_I32 */
tm_tensor* tm_long(tm_tensor* t);    /* Convert to TM_I64 */

/* ============================================================= */
/* Neural Network Operations                                     */
/* ============================================================= */

/* Softmax over dimension */
tm_tensor* tm_softmax(tm_tensor* t, int dim);

/* Log softmax */
tm_tensor* tm_log_softmax(tm_tensor* t, int dim);

/* Layer normalization */
tm_tensor* tm_layer_norm(tm_tensor* t, tm_tensor* weight, tm_tensor* bias, float eps);

/* Embedding lookup */
tm_tensor* tm_embedding(tm_tensor* weight, tm_tensor* indices);

/* Dropout (training mode) */
tm_tensor* tm_dropout(tm_tensor* t, float p, bool training);

/* RMS Normalization (used in Llama, modern LLMs) */
tm_tensor* tm_rms_norm(tm_tensor* t, tm_tensor* weight, float eps);

/* Linear layer: x @ weight.T + bias */
tm_tensor* tm_linear(tm_tensor* x, tm_tensor* weight, tm_tensor* bias);

/* Upper triangular matrix (for causal masks) */
tm_tensor* tm_triu(tm_tensor* t, int diagonal);

/* Lower triangular matrix */
tm_tensor* tm_tril(tm_tensor* t, int diagonal);

/* Hardswish activation: x * relu6(x + 3) / 6 */
tm_tensor* tm_hardswish(tm_tensor* t);

/* Hardtanh activation: clamp(x, min, max) */
tm_tensor* tm_hardtanh(tm_tensor* t, float min_val, float max_val);

/* GELU approximate (tanh approximation) */
tm_tensor* tm_gelu_approximate(tm_tensor* t);

/* Softplus: log(1 + exp(x)) */
tm_tensor* tm_softplus(tm_tensor* t, float beta, float threshold);

/* Mish activation: x * tanh(softplus(x)) */
tm_tensor* tm_mish(tm_tensor* t);

/* SELU activation: scale * (max(0,x) + min(0, alpha*(exp(x)-1))) */
tm_tensor* tm_selu(tm_tensor* t);

/* CELU activation: max(0,x) + min(0, alpha*(exp(x/alpha)-1)) */
tm_tensor* tm_celu(tm_tensor* t, float alpha);

/* ReLU6 activation: min(max(0,x), 6) */
tm_tensor* tm_relu6(tm_tensor* t);

/* GLU (Gated Linear Unit): a * sigmoid(b) where input split in half */
tm_tensor* tm_glu(tm_tensor* t, int dim);

/* Softsign activation: x / (1 + |x|) */
tm_tensor* tm_softsign(tm_tensor* t);

/* Tanhshrink activation: x - tanh(x) */
tm_tensor* tm_tanhshrink(tm_tensor* t);

/* Hardsigmoid: 0 if x<=-3, 1 if x>=3, else x/6 + 0.5 */
tm_tensor* tm_hardsigmoid(tm_tensor* t);

/* Threshold: x if x > threshold, else value */
tm_tensor* tm_threshold(tm_tensor* t, float threshold, float value);

/* Pad tensor (constant padding) */
tm_tensor* tm_pad(tm_tensor* t, int64_t* padding, int pad_len, float value);

/* Normalize (Lp normalization along dimension) */
tm_tensor* tm_normalize(tm_tensor* t, float p, int dim, float eps);

/* Interpolate/upsample (nearest neighbor) */
tm_tensor* tm_interpolate(tm_tensor* t, int64_t* size, int size_len, const char* mode);

/* Batch normalization: (x - mean) / sqrt(var + eps) * weight + bias */
tm_tensor* tm_batch_norm(tm_tensor* t, tm_tensor* weight, tm_tensor* bias,
                         tm_tensor* running_mean, tm_tensor* running_var,
                         bool training, float momentum, float eps);

/* Group normalization */
tm_tensor* tm_group_norm(tm_tensor* t, int num_groups, tm_tensor* weight,
                         tm_tensor* bias, float eps);

/* Instance normalization */
tm_tensor* tm_instance_norm(tm_tensor* t, tm_tensor* weight, tm_tensor* bias, float eps);

/* Scaled dot-product attention:
 * Q: [batch, heads, seq_q, head_dim]
 * K: [batch, heads, seq_k, head_dim]
 * V: [batch, heads, seq_k, head_dim]
 * attn_mask: optional [batch, heads, seq_q, seq_k] or NULL
 * Returns: [batch, heads, seq_q, head_dim] */
tm_tensor* tm_scaled_dot_product_attention(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                                           tm_tensor* attn_mask, float dropout_p,
                                           bool is_causal);

/* Multi-head attention helper: split heads
 * x: [batch, seq, embed_dim] -> [batch, heads, seq, head_dim] */
tm_tensor* tm_split_heads(tm_tensor* x, int num_heads);

/* Multi-head attention helper: merge heads
 * x: [batch, heads, seq, head_dim] -> [batch, seq, embed_dim] */
tm_tensor* tm_merge_heads(tm_tensor* x);

/* 1D Convolution */
tm_tensor* tm_conv1d(tm_tensor* input, tm_tensor* weight, tm_tensor* bias,
                     int stride, int padding, int dilation, int groups);

/* 2D Convolution */
tm_tensor* tm_conv2d(tm_tensor* input, tm_tensor* weight, tm_tensor* bias,
                     int stride_h, int stride_w, int padding_h, int padding_w,
                     int dilation_h, int dilation_w, int groups);

/* 1D Average Pooling */
tm_tensor* tm_avg_pool1d(tm_tensor* input, int kernel_size, int stride, int padding);

/* 1D Max Pooling */
tm_tensor* tm_max_pool1d(tm_tensor* input, int kernel_size, int stride, int padding);

/* 2D Average Pooling */
tm_tensor* tm_avg_pool2d(tm_tensor* input, int kernel_h, int kernel_w,
                         int stride_h, int stride_w, int padding_h, int padding_w);

/* 2D Max Pooling */
tm_tensor* tm_max_pool2d(tm_tensor* input, int kernel_h, int kernel_w,
                         int stride_h, int stride_w, int padding_h, int padding_w);

/* Adaptive Average Pooling 1D: output has shape [batch, channels, output_size] */
tm_tensor* tm_adaptive_avg_pool1d(tm_tensor* input, int output_size);

/* Adaptive Average Pooling 2D: output has shape [batch, channels, output_h, output_w] */
tm_tensor* tm_adaptive_avg_pool2d(tm_tensor* input, int output_h, int output_w);

/* ============================================================= */
/* Loss Functions                                                */
/* ============================================================= */

/* Cross-entropy loss: logits [batch, classes], targets [batch] (class indices)
 * Returns scalar loss tensor */
tm_tensor* tm_cross_entropy(tm_tensor* logits, tm_tensor* targets);

/* Cross-entropy loss with soft targets: both [batch, classes]
 * Returns scalar loss tensor */
tm_tensor* tm_cross_entropy_soft(tm_tensor* logits, tm_tensor* targets);

/* Mean squared error loss: pred [*], target [*] (same shape)
 * Returns scalar loss tensor */
tm_tensor* tm_mse_loss(tm_tensor* pred, tm_tensor* target);

/* L1 (MAE) loss: sum(|pred - target|) / n */
tm_tensor* tm_l1_loss(tm_tensor* pred, tm_tensor* target);

/* Smooth L1 (Huber) loss: 0.5*x^2/beta if |x| < beta, else |x| - 0.5*beta */
tm_tensor* tm_smooth_l1_loss(tm_tensor* pred, tm_tensor* target, float beta);

/* Negative log likelihood loss: expects log probabilities and class indices */
tm_tensor* tm_nll_loss(tm_tensor* log_probs, tm_tensor* targets);

/* KL divergence: KL(target || input), input is log probs, target is probs */
tm_tensor* tm_kl_div(tm_tensor* input, tm_tensor* target);

/* Binary cross entropy: for binary classification */
tm_tensor* tm_binary_cross_entropy(tm_tensor* pred, tm_tensor* target);

/* Binary cross entropy with logits: applies sigmoid internally */
tm_tensor* tm_binary_cross_entropy_with_logits(tm_tensor* logits, tm_tensor* target);

/* Huber loss: smooth combination of L1 and L2 */
tm_tensor* tm_huber_loss(tm_tensor* pred, tm_tensor* target, float delta);

/* Cosine embedding loss: for similarity/dissimilarity learning */
tm_tensor* tm_cosine_embedding_loss(tm_tensor* x1, tm_tensor* x2, tm_tensor* y, float margin);

/* Triplet margin loss: for metric learning */
tm_tensor* tm_triplet_margin_loss(tm_tensor* anchor, tm_tensor* positive, tm_tensor* negative, float margin, float p);

/* Cross entropy with label smoothing */
tm_tensor* tm_cross_entropy_label_smooth(tm_tensor* logits, tm_tensor* targets, float smoothing);

/* ============================================================= */
/* Linear Algebra                                                */
/* ============================================================= */

/* Matrix inverse (Gauss-Jordan elimination) */
tm_tensor* tm_inverse(tm_tensor* t);

/* Determinant (LU decomposition) */
tm_tensor* tm_det(tm_tensor* t);

/* Singular Value Decomposition: returns [U, S, V] where A = U * diag(S) * V^T */
tm_tensor** tm_svd(tm_tensor* t, int* num_outputs);

/* Eigenvalues (QR iteration, symmetric matrices) */
tm_tensor* tm_eig(tm_tensor* t);

/* ============================================================= */
/* Autograd                                                      */
/* ============================================================= */

/* Enable gradient tracking */
void tm_set_requires_grad(tm_tensor* t, bool requires);

/* Compute gradients (backpropagation) */
void tm_backward(tm_tensor* loss);

/* Zero all gradients */
void tm_zero_grad(tm_tensor** params, int n);

/* Detach from computation graph */
tm_tensor* tm_detach(tm_tensor* t);

/* No-grad context (returns previous state) */
bool tm_no_grad_begin(void);
void tm_no_grad_end(bool prev);

/* Retain gradient for non-leaf tensors */
void tm_retain_grad(tm_tensor* t);

/* Compute gradient of outputs w.r.t. inputs explicitly */
tm_tensor** tm_grad(tm_tensor** outputs, int num_outputs, tm_tensor** inputs, int num_inputs,
                    tm_tensor** grad_outputs);

/* Gradient checkpointing: recompute forward during backward to save memory
 * Note: Full memory savings only on GPU; CPU version runs forward directly */
tm_tensor* tm_checkpoint(tm_tensor* (*fn)(tm_tensor**, int, void*),
                         tm_tensor** inputs, int num_inputs, void* ctx);

/* ============================================================= */
/* Higher-Order Gradients                                        */
/* ============================================================= */

/* Compute Hessian matrix of loss w.r.t. parameters using finite differences
 * loss_fn: function that computes scalar loss from parameters
 * Returns n x n Hessian matrix where n = total number of parameters */
tm_tensor* tm_hessian(tm_tensor** params, int num_params,
                      tm_tensor* (*loss_fn)(tm_tensor**, int, void*), void* ctx);

/* Hessian-vector product: H @ v without computing full Hessian
 * More memory efficient than tm_hessian for large models
 * Returns 1D tensor of length = total parameters */
tm_tensor* tm_hvp(tm_tensor** params, int num_params, tm_tensor** vec, int num_vec,
                  tm_tensor* (*loss_fn)(tm_tensor**, int, void*), void* ctx);

/* Jacobian-vector product (forward-mode AD)
 * Computes J @ tangents where J is Jacobian of fn w.r.t. inputs
 * Returns array of output tensors (use num_outputs_out to get count) */
tm_tensor** tm_jvp(tm_tensor* (*fn)(tm_tensor**, int, void*), tm_tensor** inputs, int num_inputs,
                   tm_tensor** tangents, void* ctx, int* num_outputs_out);

/* ============================================================= */
/* Custom Autograd Functions                                     */
/* ============================================================= */

/* Custom context for saving tensors during forward pass */
typedef struct tm_custom_ctx tm_custom_ctx;

/* Custom autograd function definition */
typedef struct tm_autograd_function tm_autograd_function;

/* Create a new custom context */
tm_custom_ctx* tm_custom_ctx_new(void);

/* Save tensor for backward pass */
void tm_save_for_backward(tm_custom_ctx* ctx, tm_tensor* t);

/* Get saved tensors (returns array, sets num_saved) */
tm_tensor** tm_get_saved_tensors(tm_custom_ctx* ctx, int* num_saved);

/* Set/get user data in context */
void tm_custom_ctx_set_data(tm_custom_ctx* ctx, void* user_data);
void* tm_custom_ctx_get_data(tm_custom_ctx* ctx);

/* Free custom context */
void tm_custom_ctx_free(tm_custom_ctx* ctx);

/* Register a custom autograd function
 * forward: takes inputs and context, returns output (save tensors to ctx)
 * backward: takes grad_output and context, returns array of input gradients
 * Returns function ID (>=0) on success, -1 on failure */
int tm_register_autograd_function(const char* name,
                                   tm_tensor* (*forward)(tm_tensor**, int, tm_custom_ctx*),
                                   tm_tensor** (*backward)(tm_tensor*, tm_custom_ctx*, int*));

/* Apply a registered custom function by name */
tm_tensor* tm_apply_custom_function(const char* name, tm_tensor** inputs, int num_inputs);

/* Apply a registered custom function by ID (faster) */
tm_tensor* tm_apply_custom_function_id(int fn_id, tm_tensor** inputs, int num_inputs);

/* Clean up custom function registry */
void tm_cleanup_autograd_registry(void);

/* ============================================================= */
/* In-place Operations                                           */
/* ============================================================= */

/* Create zeros tensor with same shape as input */
tm_tensor* tm_zeros_like(tm_tensor* t);

/* Create ones tensor with same shape as input */
tm_tensor* tm_ones_like(tm_tensor* t);

/* Fill tensor with value (in-place) */
void tm_fill_(tm_tensor* t, float value);

/* Add tensors in-place: a += b */
void tm_add_(tm_tensor* a, tm_tensor* b);

/* Scale tensor in-place: t *= s */
void tm_scale_(tm_tensor* t, float s);

/* Zero tensor in-place */
void tm_zero_(tm_tensor* t);

/* Copy data in-place: dst = src */
void tm_copy_(tm_tensor* dst, tm_tensor* src);

/* Clamp in-place */
void tm_clamp_(tm_tensor* t, float min_val, float max_val);

/* More in-place operations */
void tm_sub_(tm_tensor* a, tm_tensor* b);
void tm_mul_(tm_tensor* a, tm_tensor* b);
void tm_div_(tm_tensor* a, tm_tensor* b);
void tm_exp_(tm_tensor* t);
void tm_log_(tm_tensor* t);
void tm_sqrt_(tm_tensor* t);
void tm_abs_(tm_tensor* t);
void tm_neg_(tm_tensor* t);
void tm_relu_(tm_tensor* t);
void tm_sigmoid_(tm_tensor* t);

/* ============================================================= */
/* Sparse Tensor Operations                                      */
/* ============================================================= */

/* Create sparse COO tensor from indices and values
 * indices: [2, nnz] tensor of (row, col) pairs
 * values: [nnz] tensor of non-zero values
 * shape: dense shape [rows, cols] */
tm_tensor* tm_sparse_coo(tm_tensor* indices, tm_tensor* values, int64_t* shape, int ndim);

/* Create sparse CSR tensor from row_ptr, col_ind, and values
 * row_ptr: [rows + 1] row pointers
 * col_ind: [nnz] column indices
 * values: [nnz] tensor of non-zero values
 * shape: dense shape [rows, cols] */
tm_tensor* tm_sparse_csr(tm_tensor* row_ptr, tm_tensor* col_ind, tm_tensor* values,
                         int64_t* shape, int ndim);

/* Convert dense tensor to sparse COO format */
tm_tensor* tm_to_sparse(tm_tensor* t);

/* Convert dense tensor to sparse CSR format */
tm_tensor* tm_to_sparse_csr(tm_tensor* t);

/* Convert sparse tensor to dense */
tm_tensor* tm_to_dense(tm_tensor* t);

/* Check if tensor is sparse */
bool tm_is_sparse(tm_tensor* t);

/* Get number of non-zero elements */
int64_t tm_sparse_nnz(tm_tensor* t);

/* Get sparse format */
tm_sparse_format tm_get_sparse_format(tm_tensor* t);

/* Convert between sparse formats */
tm_tensor* tm_sparse_coo_to_csr(tm_tensor* t);
tm_tensor* tm_sparse_csr_to_coo(tm_tensor* t);

/* Sparse-dense matrix multiplication: sparse @ dense -> dense */
tm_tensor* tm_spmm(tm_tensor* sparse, tm_tensor* dense);

/* Sparse-sparse addition: sparse + sparse -> sparse */
tm_tensor* tm_sparse_add(tm_tensor* a, tm_tensor* b);

/* Sparse-scalar multiplication: sparse * scalar -> sparse */
tm_tensor* tm_sparse_mul_scalar(tm_tensor* t, double s);

/* Sparse matrix-vector multiplication: sparse @ vector -> vector */
tm_tensor* tm_spmv(tm_tensor* sparse, tm_tensor* vec);

/* Get values tensor from sparse tensor */
tm_tensor* tm_sparse_values(tm_tensor* t);

/* Get indices tensor from sparse COO tensor */
tm_tensor* tm_sparse_indices(tm_tensor* t);

/* Coalesce sparse tensor (combine duplicate indices) */
tm_tensor* tm_sparse_coalesce(tm_tensor* t);

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

/* Print tensor info */
void tm_print(tm_tensor* t);

/* Print tensor with data */
void tm_print_data(tm_tensor* t);

/* Seed random number generator */
void tm_manual_seed(uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_TENSOR_H */
