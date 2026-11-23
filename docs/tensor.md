# tensor

Tensor operations with automatic differentiation. Python counterpart: PyTorch core functionality.

**Header:** `tensor/tensor.h`
**Source:** `tensor/tensor.c`
**Status:** 141 unit tests passing; PyTorch comparison suite 80/83 (`make compare-test`, needs python3 + torch)

---

## Overview

The tensor module provides:
- N-dimensional tensor storage
- Element-wise and matrix operations
- Automatic differentiation (autograd)
- Broadcasting support
- Multiple data types (F32, F64, F16, BF16, I32, I64)

---

## Quick Start

```c
#include "tensor/tensor.h"

// Create tensors
int64_t shape[] = {32, 64};
tm_tensor* a = tm_randn(shape, 2, TM_F32);
tm_tensor* b = tm_randn(shape, 2, TM_F32);

// Enable gradient tracking
tm_set_requires_grad(a, true);
tm_set_requires_grad(b, true);

// Operations
tm_tensor* c = tm_add(a, b);
tm_tensor* d = tm_relu(c);
tm_tensor* loss = tm_mean(d, -1, false);

// Backpropagation
tm_backward(loss);

// Access gradients
float* grad_a = tm_data_f32(a->grad);

// Cleanup
tm_free(loss);
tm_free(d);
tm_free(c);
tm_free(b);
tm_free(a);
```

---

## Data Types

| Type | Constant | Size | Description |
|------|----------|------|-------------|
| float32 | `TM_F32` | 4 bytes | Default, most operations |
| float64 | `TM_F64` | 8 bytes | High precision |
| float16 | `TM_F16` | 2 bytes | Memory efficient |
| bfloat16 | `TM_BF16` | 2 bytes | Training efficient |
| int32 | `TM_I32` | 4 bytes | Indices |
| int64 | `TM_I64` | 8 bytes | Large indices |

---

## API Reference

### Tensor Creation

#### tm_empty
```c
tm_tensor* tm_empty(int64_t* shape, int ndim, tm_dtype dtype);
```
Create tensor with uninitialized data.

#### tm_zeros
```c
tm_tensor* tm_zeros(int64_t* shape, int ndim, tm_dtype dtype);
```
Create tensor filled with zeros.

#### tm_ones
```c
tm_tensor* tm_ones(int64_t* shape, int ndim, tm_dtype dtype);
```
Create tensor filled with ones.

#### tm_full
```c
tm_tensor* tm_full(int64_t* shape, int ndim, tm_dtype dtype, double value);
```
Create tensor filled with specified value.

#### tm_randn
```c
tm_tensor* tm_randn(int64_t* shape, int ndim, tm_dtype dtype);
```
Create tensor with random normal values (mean=0, std=1).

#### tm_rand
```c
tm_tensor* tm_rand(int64_t* shape, int ndim, tm_dtype dtype);
```
Create tensor with random uniform values [0, 1).

#### tm_from_data
```c
tm_tensor* tm_from_data(void* data, int64_t* shape, int ndim, tm_dtype dtype);
```
Create tensor from existing data (does not copy).

#### tm_clone
```c
tm_tensor* tm_clone(tm_tensor* t);
```
Deep copy tensor.

#### tm_zeros_like
```c
tm_tensor* tm_zeros_like(tm_tensor* t);
```
Create zeros tensor with same shape as input.

#### tm_ones_like
```c
tm_tensor* tm_ones_like(tm_tensor* t);
```
Create ones tensor with same shape as input.

#### tm_arange
```c
tm_tensor* tm_arange(double start, double end, double step, tm_dtype dtype);
```
Create tensor with range of values [start, end) with step.

#### tm_eye
```c
tm_tensor* tm_eye(int64_t n, tm_dtype dtype);
```
Create n×n identity matrix.

#### tm_linspace
```c
tm_tensor* tm_linspace(double start, double end, int64_t steps, tm_dtype dtype);
```
Create linearly spaced values from start to end.

#### tm_randint
```c
tm_tensor* tm_randint(int64_t low, int64_t high, int64_t* shape, int ndim);
```
Create tensor with random integers in [low, high).

#### tm_full_like
```c
tm_tensor* tm_full_like(tm_tensor* t, double value);
```
Create tensor filled with value, same shape as input.

#### tm_empty_like
```c
tm_tensor* tm_empty_like(tm_tensor* t);
```
Create uninitialized tensor with same shape as input.

#### tm_rand_like
```c
tm_tensor* tm_rand_like(tm_tensor* t);
```
Create random uniform tensor with same shape as input.

#### tm_randn_like
```c
tm_tensor* tm_randn_like(tm_tensor* t);
```
Create random normal tensor with same shape as input.

---

### Memory Management

#### tm_retain
```c
tm_tensor* tm_retain(tm_tensor* t);
```
Increment reference count.

#### tm_release / tm_free
```c
void tm_release(tm_tensor* t);
#define tm_free(t) tm_release(t)
```
Decrement reference count, free if zero.

---

### Accessors

#### tm_get_f32 / tm_set_f32
```c
float tm_get_f32(tm_tensor* t, int64_t i);
void tm_set_f32(tm_tensor* t, int64_t i, float val);
```
Get/set element by flat index.

#### tm_data_f32
```c
float* tm_data_f32(tm_tensor* t);
```
Get pointer to data.

#### tm_numel
```c
int64_t tm_numel(tm_tensor* t);
```
Get total number of elements.

#### tm_ndim
```c
int tm_ndim(tm_tensor* t);
```
Get number of dimensions.

#### tm_size
```c
int64_t tm_size(tm_tensor* t, int dim);
```
Get size of dimension.

---

### Shape Operations

#### tm_reshape
```c
tm_tensor* tm_reshape(tm_tensor* t, int64_t* shape, int ndim);
```
Reshape tensor (returns view if possible).

#### tm_transpose
```c
tm_tensor* tm_transpose(tm_tensor* t, int dim0, int dim1);
```
Swap two dimensions.

#### tm_squeeze
```c
tm_tensor* tm_squeeze(tm_tensor* t, int dim);
```
Remove dimension of size 1.

#### tm_unsqueeze
```c
tm_tensor* tm_unsqueeze(tm_tensor* t, int dim);
```
Add dimension of size 1.

#### tm_contiguous
```c
tm_tensor* tm_contiguous(tm_tensor* t);
```
Ensure data is contiguous in memory.

#### tm_view
```c
tm_tensor* tm_view(tm_tensor* t, int64_t* shape, int ndim);
```
View tensor with new shape (alias for reshape).

#### tm_permute
```c
tm_tensor* tm_permute(tm_tensor* t, int* dims, int ndim);
```
Permute dimensions (generalized transpose).

#### tm_flatten
```c
tm_tensor* tm_flatten(tm_tensor* t, int start_dim, int end_dim);
```
Flatten dimensions from start_dim to end_dim.

#### tm_select
```c
tm_tensor* tm_select(tm_tensor* t, int dim, int64_t index);
```
Select along dimension (reduces ndim by 1).

#### tm_cat
```c
tm_tensor* tm_cat(tm_tensor** tensors, int num_tensors, int dim);
```
Concatenate tensors along dimension.

#### tm_stack
```c
tm_tensor* tm_stack(tm_tensor** tensors, int num_tensors, int dim);
```
Stack tensors along new dimension.

#### tm_split
```c
tm_tensor** tm_split(tm_tensor* t, int64_t split_size, int dim, int* num_chunks);
```
Split tensor into chunks along dimension.

#### tm_chunk
```c
tm_tensor** tm_chunk(tm_tensor* t, int n, int dim, int* actual);
```
Split tensor into n equal chunks along dimension.

#### tm_expand
```c
tm_tensor* tm_expand(tm_tensor* t, int64_t* shape, int ndim);
```
Expand tensor to larger shape (broadcasting).

#### tm_repeat
```c
tm_tensor* tm_repeat(tm_tensor* t, int64_t* sizes, int ndim);
```
Repeat tensor along dimensions.

#### tm_narrow
```c
tm_tensor* tm_narrow(tm_tensor* t, int dim, int64_t start, int64_t length);
```
Narrow tensor along dimension (slice view).

#### tm_flip
```c
tm_tensor* tm_flip(tm_tensor* t, int* dims, int ndim);
```
Reverse tensor along specified dimensions.

#### tm_roll
```c
tm_tensor* tm_roll(tm_tensor* t, int64_t shift, int dim);
```
Roll tensor elements along dimension.

#### tm_t
```c
tm_tensor* tm_t(tm_tensor* t);
```
Transpose 2D tensor (alias for transpose(0, 1)).

---

### Element-wise Operations

All element-wise operations support broadcasting.

| Function | Operation |
|----------|-----------|
| `tm_add(a, b)` | a + b |
| `tm_sub(a, b)` | a - b |
| `tm_mul(a, b)` | a * b |
| `tm_div(a, b)` | a / b |
| `tm_add_scalar(a, s)` | a + s |
| `tm_sub_scalar(a, s)` | a - s |
| `tm_mul_scalar(a, s)` | a * s |
| `tm_div_scalar(a, s)` | a / s |
| `tm_neg(a)` | -a |
| `tm_abs(a)` | \|a\| |
| `tm_exp(a)` | exp(a) |
| `tm_log(a)` | log(a) |
| `tm_sqrt(a)` | sqrt(a) |
| `tm_rsqrt(a)` | 1/sqrt(a) |
| `tm_reciprocal(a)` | 1/a |
| `tm_pow(a, n)` | a^n |
| `tm_sin(a)` | sin(a) |
| `tm_cos(a)` | cos(a) |
| `tm_tan(a)` | tan(a) |
| `tm_asin(a)` | arcsin(a) |
| `tm_acos(a)` | arccos(a) |
| `tm_atan(a)` | arctan(a) |
| `tm_sinh(a)` | sinh(a) |
| `tm_cosh(a)` | cosh(a) |
| `tm_tanh(a)` | tanh(a) |
| `tm_exp2(a)` | 2^a |
| `tm_expm1(a)` | exp(a) - 1 |
| `tm_log2(a)` | log2(a) |
| `tm_log10(a)` | log10(a) |
| `tm_log1p(a)` | log(1 + a) |
| `tm_floor(a)` | floor(a) |
| `tm_ceil(a)` | ceil(a) |
| `tm_round(a)` | round(a) |
| `tm_trunc(a)` | truncate(a) |
| `tm_sign(a)` | sign(a) |
| `tm_relu(a)` | max(0, a) |
| `tm_leaky_relu(a, slope)` | Leaky ReLU |
| `tm_elu(a, alpha)` | ELU activation |
| `tm_sigmoid(a)` | 1/(1+exp(-a)) |
| `tm_gelu(a)` | GELU activation |
| `tm_silu(a)` | x * sigmoid(x) (SwiGLU) |
| `tm_hardswish(a)` | x * relu6(x+3) / 6 |
| `tm_hardtanh(a, min, max)` | Clamped linear |
| `tm_softplus(a, beta, thresh)` | Smooth ReLU |
| `tm_clamp(a, min, max)` | Clamp values |
| `tm_maximum(a, b)` | Element-wise max |
| `tm_minimum(a, b)` | Element-wise min |
| `tm_where(cond, a, b)` | Conditional select |
| `tm_masked_fill(t, mask, val)` | Fill where mask is true |
| `tm_lerp(a, b, w)` | Linear interpolation |
| `tm_addcmul(t, a, b, val)` | t + val * a * b |
| `tm_addcdiv(t, a, b, val)` | t + val * a / b |

---

### Matrix Operations

#### tm_matmul
```c
tm_tensor* tm_matmul(tm_tensor* a, tm_tensor* b);
```
Matrix multiplication. Supports 2D and batched 3D tensors.

#### tm_bmm
```c
tm_tensor* tm_bmm(tm_tensor* a, tm_tensor* b);
```
Batched matrix multiplication.

#### tm_mm
```c
tm_tensor* tm_mm(tm_tensor* a, tm_tensor* b);
```
2D matrix multiplication (alias for matmul).

#### tm_addmm
```c
tm_tensor* tm_addmm(tm_tensor* t, tm_tensor* a, tm_tensor* b, float beta, float alpha);
```
Additive matrix multiply: beta*t + alpha*(a @ b).

#### tm_dot
```c
tm_tensor* tm_dot(tm_tensor* a, tm_tensor* b);
```
Dot product of 1D tensors.

#### tm_mv
```c
tm_tensor* tm_mv(tm_tensor* mat, tm_tensor* vec);
```
Matrix-vector multiplication.

#### tm_outer
```c
tm_tensor* tm_outer(tm_tensor* a, tm_tensor* b);
```
Outer product of 1D tensors.

#### tm_inner
```c
tm_tensor* tm_inner(tm_tensor* a, tm_tensor* b);
```
Inner product of tensors.

#### tm_inverse
```c
tm_tensor* tm_inverse(tm_tensor* t);
```
Matrix inverse using Gauss-Jordan elimination.

#### tm_det
```c
tm_tensor* tm_det(tm_tensor* t);
```
Matrix determinant using LU decomposition.

#### tm_svd
```c
tm_tensor** tm_svd(tm_tensor* t, int* num_outputs);
```
Singular value decomposition. Returns array of 3 tensors: U, S, V.

#### tm_eig
```c
tm_tensor* tm_eig(tm_tensor* t);
```
Eigenvalues of a square matrix using QR iteration.

---

### Comparison Operations

| Function | Operation |
|----------|-----------|
| `tm_eq(a, b)` | a == b (element-wise) |
| `tm_ne(a, b)` | a != b (element-wise) |
| `tm_lt(a, b)` | a < b (element-wise) |
| `tm_le(a, b)` | a <= b (element-wise) |
| `tm_gt(a, b)` | a > b (element-wise) |
| `tm_ge(a, b)` | a >= b (element-wise) |
| `tm_equal(a, b)` | All elements equal (bool) |
| `tm_allclose(a, b, rtol, atol)` | Approximately equal |
| `tm_isnan(a)` | Check for NaN |
| `tm_isinf(a)` | Check for Inf |
| `tm_isfinite(a)` | Check for finite |
| `tm_logical_and(a, b)` | Logical AND |
| `tm_logical_or(a, b)` | Logical OR |
| `tm_logical_not(a)` | Logical NOT |
| `tm_logical_xor(a, b)` | Logical XOR |

---

### Indexing Operations

#### tm_gather
```c
tm_tensor* tm_gather(tm_tensor* t, int dim, tm_tensor* idx);
```
Gather elements by index along dimension.

#### tm_scatter
```c
tm_tensor* tm_scatter(tm_tensor* t, int dim, tm_tensor* idx, tm_tensor* src);
```
Scatter elements by index along dimension.

#### tm_scatter_add
```c
tm_tensor* tm_scatter_add(tm_tensor* t, int dim, tm_tensor* idx, tm_tensor* src);
```
Scatter with addition.

#### tm_index_select
```c
tm_tensor* tm_index_select(tm_tensor* t, int dim, tm_tensor* idx);
```
Select elements by index along dimension.

#### tm_masked_select
```c
tm_tensor* tm_masked_select(tm_tensor* t, tm_tensor* mask);
```
Select elements where mask is true.

#### tm_take
```c
tm_tensor* tm_take(tm_tensor* t, tm_tensor* idx);
```
Take elements by flat index.

#### tm_index_fill
```c
tm_tensor* tm_index_fill(tm_tensor* t, int dim, tm_tensor* idx, float val);
```
Fill at indices with value.

#### tm_index_copy
```c
tm_tensor* tm_index_copy(tm_tensor* t, int dim, tm_tensor* idx, tm_tensor* src);
```
Copy to indices.

#### tm_index_add
```c
tm_tensor* tm_index_add(tm_tensor* t, int dim, tm_tensor* idx, tm_tensor* src);
```
Add at indices.

#### tm_nonzero
```c
tm_tensor* tm_nonzero(tm_tensor* t);
```
Indices of non-zero elements.

---

### Reductions

#### tm_sum
```c
tm_tensor* tm_sum(tm_tensor* t, int dim, bool keepdim);
```
Sum over dimension. `dim=-1` means sum all elements.

#### tm_mean
```c
tm_tensor* tm_mean(tm_tensor* t, int dim, bool keepdim);
```
Mean over dimension.

#### tm_max / tm_min
```c
tm_tensor* tm_max(tm_tensor* t, int dim, bool keepdim);
tm_tensor* tm_min(tm_tensor* t, int dim, bool keepdim);
```
Max/min over dimension.

#### tm_argmax / tm_argmin
```c
tm_tensor* tm_argmax(tm_tensor* t, int dim, bool keepdim);
tm_tensor* tm_argmin(tm_tensor* t, int dim, bool keepdim);
```
Index of max/min value.

#### tm_logsumexp
```c
tm_tensor* tm_logsumexp(tm_tensor* t, int dim, bool keepdim);
```
Log-sum-exp (numerically stable).

#### tm_prod
```c
tm_tensor* tm_prod(tm_tensor* t, int dim, bool keepdim);
```
Product over dimension.

#### tm_cumsum
```c
tm_tensor* tm_cumsum(tm_tensor* t, int dim);
```
Cumulative sum along dimension.

#### tm_cumprod
```c
tm_tensor* tm_cumprod(tm_tensor* t, int dim);
```
Cumulative product along dimension.

#### tm_std / tm_var
```c
tm_tensor* tm_std(tm_tensor* t, int dim, bool keepdim, bool unbiased);
tm_tensor* tm_var(tm_tensor* t, int dim, bool keepdim, bool unbiased);
```
Standard deviation and variance.

#### tm_any / tm_all
```c
tm_tensor* tm_any(tm_tensor* t, int dim, bool keepdim);
tm_tensor* tm_all(tm_tensor* t, int dim, bool keepdim);
```
Any/all true along dimension.

#### tm_norm
```c
tm_tensor* tm_norm(tm_tensor* t, float p, int dim, bool keepdim);
```
Lp norm along dimension.

#### tm_count_nonzero
```c
tm_tensor* tm_count_nonzero(tm_tensor* t, int dim);
```
Count non-zero elements.

---

### Neural Network Operations

#### tm_softmax
```c
tm_tensor* tm_softmax(tm_tensor* t, int dim);
```
Softmax over dimension.

#### tm_log_softmax
```c
tm_tensor* tm_log_softmax(tm_tensor* t, int dim);
```
Log softmax (numerically stable).

#### tm_layer_norm
```c
tm_tensor* tm_layer_norm(tm_tensor* t, tm_tensor* weight, tm_tensor* bias, float eps);
```
Layer normalization.

#### tm_embedding
```c
tm_tensor* tm_embedding(tm_tensor* weight, tm_tensor* indices);
```
Embedding lookup.

#### tm_dropout
```c
tm_tensor* tm_dropout(tm_tensor* t, float p, bool training);
```
Dropout (training mode).

#### tm_rms_norm
```c
tm_tensor* tm_rms_norm(tm_tensor* t, tm_tensor* weight, float eps);
```
RMS normalization (used in Llama, modern LLMs).

#### tm_linear
```c
tm_tensor* tm_linear(tm_tensor* x, tm_tensor* weight, tm_tensor* bias);
```
Linear layer: x @ weight.T + bias.

#### tm_triu / tm_tril
```c
tm_tensor* tm_triu(tm_tensor* t, int diagonal);
tm_tensor* tm_tril(tm_tensor* t, int diagonal);
```
Upper/lower triangular matrix (for causal masks).

#### tm_batch_norm
```c
tm_tensor* tm_batch_norm(tm_tensor* t, tm_tensor* weight, tm_tensor* bias,
                         tm_tensor* running_mean, tm_tensor* running_var,
                         bool training, float momentum, float eps);
```
Batch normalization.

#### tm_group_norm
```c
tm_tensor* tm_group_norm(tm_tensor* t, int num_groups, tm_tensor* weight,
                         tm_tensor* bias, float eps);
```
Group normalization.

#### tm_instance_norm
```c
tm_tensor* tm_instance_norm(tm_tensor* t, tm_tensor* weight, tm_tensor* bias,
                            float eps);
```
Instance normalization.

#### tm_scaled_dot_product_attention
```c
tm_tensor* tm_scaled_dot_product_attention(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                                           tm_tensor* attn_mask, float dropout_p,
                                           bool is_causal);
```
Scaled dot-product attention with optional causal mask.

#### tm_split_heads / tm_merge_heads
```c
tm_tensor* tm_split_heads(tm_tensor* x, int num_heads);
tm_tensor* tm_merge_heads(tm_tensor* x);
```
Multi-head attention helpers.

#### tm_conv1d
```c
tm_tensor* tm_conv1d(tm_tensor* input, tm_tensor* weight, tm_tensor* bias,
                     int stride, int padding, int dilation, int groups);
```
1D convolution with full parameter support.

#### tm_conv2d
```c
tm_tensor* tm_conv2d(tm_tensor* input, tm_tensor* weight, tm_tensor* bias,
                     int stride_h, int stride_w, int padding_h, int padding_w,
                     int dilation_h, int dilation_w, int groups);
```
2D convolution with full parameter support.

#### tm_avg_pool1d / tm_max_pool1d
```c
tm_tensor* tm_avg_pool1d(tm_tensor* input, int kernel_size, int stride, int padding);
tm_tensor* tm_max_pool1d(tm_tensor* input, int kernel_size, int stride, int padding);
```
1D pooling operations.

#### tm_avg_pool2d / tm_max_pool2d
```c
tm_tensor* tm_avg_pool2d(tm_tensor* input, int kernel_h, int kernel_w,
                         int stride_h, int stride_w, int padding_h, int padding_w);
tm_tensor* tm_max_pool2d(tm_tensor* input, int kernel_h, int kernel_w,
                         int stride_h, int stride_w, int padding_h, int padding_w);
```
2D pooling operations.

#### tm_adaptive_avg_pool1d / tm_adaptive_avg_pool2d
```c
tm_tensor* tm_adaptive_avg_pool1d(tm_tensor* input, int output_size);
tm_tensor* tm_adaptive_avg_pool2d(tm_tensor* input, int output_h, int output_w);
```
Adaptive average pooling (output size specified).

---

### Loss Functions

#### tm_cross_entropy
```c
tm_tensor* tm_cross_entropy(tm_tensor* logits, tm_tensor* targets);
```
Cross-entropy loss with integer targets.

#### tm_cross_entropy_soft
```c
tm_tensor* tm_cross_entropy_soft(tm_tensor* logits, tm_tensor* targets);
```
Cross-entropy loss with soft targets.

#### tm_mse_loss
```c
tm_tensor* tm_mse_loss(tm_tensor* pred, tm_tensor* target);
```
Mean squared error loss.

#### tm_l1_loss
```c
tm_tensor* tm_l1_loss(tm_tensor* pred, tm_tensor* target);
```
L1 (Mean Absolute Error) loss.

#### tm_smooth_l1_loss
```c
tm_tensor* tm_smooth_l1_loss(tm_tensor* pred, tm_tensor* target, float beta);
```
Smooth L1 (Huber) loss. Uses quadratic for small errors, linear for large.

#### tm_nll_loss
```c
tm_tensor* tm_nll_loss(tm_tensor* log_probs, tm_tensor* targets);
```
Negative log likelihood loss. Expects log probabilities as input.

#### tm_kl_div
```c
tm_tensor* tm_kl_div(tm_tensor* input, tm_tensor* target);
```
KL divergence loss. Input is log probabilities, target is probabilities.

#### tm_binary_cross_entropy
```c
tm_tensor* tm_binary_cross_entropy(tm_tensor* pred, tm_tensor* target);
```
Binary cross entropy loss for binary classification.

#### tm_cosine_embedding_loss
```c
tm_tensor* tm_cosine_embedding_loss(tm_tensor* x1, tm_tensor* x2, tm_tensor* y, float margin);
```
Cosine embedding loss. y=1 encourages similarity, y=-1 encourages dissimilarity.

#### tm_triplet_margin_loss
```c
tm_tensor* tm_triplet_margin_loss(tm_tensor* anchor, tm_tensor* positive, tm_tensor* negative, float margin, float p);
```
Triplet margin loss for metric learning. Encourages anchor-positive distance < anchor-negative distance.

#### tm_cross_entropy_label_smooth
```c
tm_tensor* tm_cross_entropy_label_smooth(tm_tensor* logits, tm_tensor* targets, float smoothing);
```
Cross-entropy loss with label smoothing. Reduces overconfidence.

---

### Autograd

#### tm_set_requires_grad
```c
void tm_set_requires_grad(tm_tensor* t, bool requires);
```
Enable/disable gradient tracking.

#### tm_backward
```c
void tm_backward(tm_tensor* loss);
```
Compute gradients via backpropagation.

#### tm_zero_grad
```c
void tm_zero_grad(tm_tensor** params, int n);
```
Zero gradients for all parameters.

#### tm_detach
```c
tm_tensor* tm_detach(tm_tensor* t);
```
Detach from computation graph.

#### tm_no_grad_begin / tm_no_grad_end
```c
bool tm_no_grad_begin(void);
void tm_no_grad_end(bool prev);
```
Temporarily disable gradient tracking.

#### tm_retain_grad
```c
void tm_retain_grad(tm_tensor* t);
```
Retain gradient for non-leaf tensors. By default, gradients are only kept for leaf tensors.

#### tm_grad
```c
tm_tensor** tm_grad(tm_tensor** outputs, int num_outputs, tm_tensor** inputs, int num_inputs, tm_tensor** grad_outputs);
```
Compute gradients explicitly. Returns array of gradients for each input.

#### tm_checkpoint
```c
tm_tensor* tm_checkpoint(tm_tensor* (*fn)(tm_tensor**, int, void*), tm_tensor** inputs, int num_inputs, void* ctx);
```
Gradient checkpointing for memory-efficient training. Recomputes forward pass during backward.

---

### Higher-Order Gradients

#### tm_hessian
```c
tm_tensor* tm_hessian(tm_tensor** params, int num_params,
                      tm_tensor* (*loss_fn)(tm_tensor**, int, void*), void* ctx);
```
Compute Hessian matrix of loss w.r.t. parameters using finite differences. Returns n x n matrix.

#### tm_hvp
```c
tm_tensor* tm_hvp(tm_tensor** params, int num_params, tm_tensor** vec, int num_vec,
                  tm_tensor* (*loss_fn)(tm_tensor**, int, void*), void* ctx);
```
Hessian-vector product without computing full Hessian. More memory efficient for large models.

#### tm_jvp
```c
tm_tensor** tm_jvp(tm_tensor* (*fn)(tm_tensor**, int, void*), tm_tensor** inputs, int num_inputs,
                   tm_tensor** tangents, void* ctx, int* num_outputs_out);
```
Jacobian-vector product (forward-mode AD). Computes J @ tangents.

---

### Custom Autograd Functions

#### tm_custom_ctx
Context for saving tensors during forward pass for use in backward.

```c
tm_custom_ctx* tm_custom_ctx_new(void);
void tm_save_for_backward(tm_custom_ctx* ctx, tm_tensor* t);
tm_tensor** tm_get_saved_tensors(tm_custom_ctx* ctx, int* num_saved);
void tm_custom_ctx_free(tm_custom_ctx* ctx);
```

#### tm_register_autograd_function
```c
int tm_register_autograd_function(const char* name,
                                   tm_tensor* (*forward)(tm_tensor**, int, tm_custom_ctx*),
                                   tm_tensor** (*backward)(tm_tensor*, tm_custom_ctx*, int*));
```
Register a custom autograd function with forward and backward implementations.

#### tm_apply_custom_function
```c
tm_tensor* tm_apply_custom_function(const char* name, tm_tensor** inputs, int num_inputs);
tm_tensor* tm_apply_custom_function_id(int fn_id, tm_tensor** inputs, int num_inputs);
```
Apply a registered custom function by name or ID.

---

### Type Conversion

#### tm_to
```c
tm_tensor* tm_to(tm_tensor* t, tm_dtype dtype);
```
Convert tensor to specified dtype.

#### tm_float
```c
tm_tensor* tm_float(tm_tensor* t);
```
Convert to float32.

#### tm_double
```c
tm_tensor* tm_double(tm_tensor* t);
```
Convert to float64.

#### tm_half
```c
tm_tensor* tm_half(tm_tensor* t);
```
Convert to float16.

#### tm_int
```c
tm_tensor* tm_int(tm_tensor* t);
```
Convert to int32.

#### tm_long
```c
tm_tensor* tm_long(tm_tensor* t);
```
Convert to int64.

---

### In-place Operations

**Note:** All in-place operations support autograd. When `requires_grad` is true, gradients are properly tracked and computed during backward pass.

#### tm_fill_
```c
void tm_fill_(tm_tensor* t, float value);
```
Fill tensor with value in-place.

#### tm_add_
```c
void tm_add_(tm_tensor* a, tm_tensor* b);
```
Add tensors in-place: `a += b`.

#### tm_scale_
```c
void tm_scale_(tm_tensor* t, float s);
```
Scale tensor in-place: `t *= s`.

#### tm_zero_
```c
void tm_zero_(tm_tensor* t);
```
Zero tensor in-place.

#### tm_copy_
```c
void tm_copy_(tm_tensor* dst, tm_tensor* src);
```
Copy data in-place.

#### tm_clamp_
```c
void tm_clamp_(tm_tensor* t, float min_val, float max_val);
```
Clamp values in-place.

#### tm_sub_
```c
void tm_sub_(tm_tensor* a, tm_tensor* b);
```
Subtract in-place: `a -= b`.

#### tm_mul_
```c
void tm_mul_(tm_tensor* a, tm_tensor* b);
```
Multiply in-place: `a *= b`.

#### tm_div_
```c
void tm_div_(tm_tensor* a, tm_tensor* b);
```
Divide in-place: `a /= b`.

#### tm_exp_
```c
void tm_exp_(tm_tensor* t);
```
Exponentiate in-place.

#### tm_log_
```c
void tm_log_(tm_tensor* t);
```
Logarithm in-place.

#### tm_sqrt_
```c
void tm_sqrt_(tm_tensor* t);
```
Square root in-place.

#### tm_abs_
```c
void tm_abs_(tm_tensor* t);
```
Absolute value in-place.

#### tm_neg_
```c
void tm_neg_(tm_tensor* t);
```
Negate in-place.

#### tm_relu_
```c
void tm_relu_(tm_tensor* t);
```
ReLU in-place.

#### tm_sigmoid_
```c
void tm_sigmoid_(tm_tensor* t);
```
Sigmoid in-place.

---

## Examples

### Matrix Multiplication

```c
int64_t shape_a[] = {32, 64};
int64_t shape_b[] = {64, 128};

tm_tensor* a = tm_randn(shape_a, 2, TM_F32);
tm_tensor* b = tm_randn(shape_b, 2, TM_F32);
tm_tensor* c = tm_matmul(a, b);  // [32, 128]

tm_print(c);  // Print shape and info
tm_free(c);
tm_free(b);
tm_free(a);
```

### Simple Neural Network Forward

```c
// Input: [batch, features]
// Weights: [features, hidden]
tm_tensor* h = tm_matmul(input, weights);
tm_tensor* h_bias = tm_add(h, bias);
tm_tensor* h_relu = tm_relu(h_bias);

// Cleanup intermediates
tm_free(h);
tm_free(h_bias);
```

### Broadcasting

```c
int64_t shape_a[] = {32, 64};
int64_t shape_b[] = {64};  // Will broadcast to [32, 64]

tm_tensor* a = tm_ones(shape_a, 2, TM_F32);
tm_tensor* b = tm_full(shape_b, 1, TM_F32, 2.0);
tm_tensor* c = tm_add(a, b);  // All elements are 3.0
```

---

## Sparse Tensors

Foundry supports sparse tensors in both COO (Coordinate) and CSR (Compressed Sparse Row) formats.

### Sparse Formats

```c
typedef enum {
    TM_SPARSE_COO,   /* Coordinate format: (row, col, value) triples */
    TM_SPARSE_CSR,   /* Compressed Sparse Row: row_ptr, col_ind, values */
} tm_sparse_format;
```

### Creating Sparse Tensors

```c
// Create sparse COO tensor from indices and values
// indices: [2, nnz] tensor of (row, col) pairs
// values: [nnz] tensor of non-zero values
tm_tensor* tm_sparse_coo(tm_tensor* indices, tm_tensor* values, int64_t* shape, int ndim);

// Create sparse CSR tensor
tm_tensor* tm_sparse_csr(tm_tensor* row_ptr, tm_tensor* col_ind, tm_tensor* values,
                         int64_t* shape, int ndim);

// Convert dense to sparse
tm_tensor* tm_to_sparse(tm_tensor* t);      // Dense -> COO
tm_tensor* tm_to_sparse_csr(tm_tensor* t);  // Dense -> CSR
tm_tensor* tm_to_dense(tm_tensor* t);       // Sparse -> Dense
```

### Sparse Operations

```c
// Check if sparse
bool tm_is_sparse(tm_tensor* t);
int64_t tm_sparse_nnz(tm_tensor* t);
tm_sparse_format tm_get_sparse_format(tm_tensor* t);

// Format conversion
tm_tensor* tm_sparse_coo_to_csr(tm_tensor* t);
tm_tensor* tm_sparse_csr_to_coo(tm_tensor* t);

// Sparse-dense matrix multiplication
tm_tensor* tm_spmm(tm_tensor* sparse, tm_tensor* dense);  // sparse @ dense
tm_tensor* tm_spmv(tm_tensor* sparse, tm_tensor* vec);    // sparse @ vector

// Sparse arithmetic
tm_tensor* tm_sparse_add(tm_tensor* a, tm_tensor* b);     // sparse + sparse
tm_tensor* tm_sparse_mul_scalar(tm_tensor* t, double s);  // sparse * scalar

// Access components
tm_tensor* tm_sparse_values(tm_tensor* t);   // Get values tensor
tm_tensor* tm_sparse_indices(tm_tensor* t);  // Get indices (COO only)
tm_tensor* tm_sparse_coalesce(tm_tensor* t); // Combine duplicate indices
```

### Example

```c
// Create sparse identity matrix
tm_tensor* dense_eye = tm_eye(1000, TM_F32);  // 1000x1000 identity
tm_tensor* sparse = tm_to_sparse(dense_eye);   // Only 1000 non-zeros stored

// Sparse-dense multiplication (efficient for large sparse matrices)
int64_t shape[] = {1000, 64};
tm_tensor* x = tm_randn(shape, 2, TM_F32);
tm_tensor* result = tm_spmm(sparse, x);  // I @ x = x

tm_free(result);
tm_free(x);
tm_free(sparse);
tm_free(dense_eye);
```

---

## Differences from PyTorch

| Feature | PyTorch | Foundry | Why |
|---------|---------|---------|-----|
| Device | CPU/CUDA/MPS | CPU only | GPU requires writing CUDA/Metal kernels for every op. Planned. |
| Autocast | Yes | No | Mixed-precision casting. Planned after GPU support. |
| torch.compile | Yes | N/A | PyTorch's JIT for Python. Foundry is already compiled C code. |
| In-place grad | Yes | **Yes** | All in-place ops (add_, mul_, exp_, relu_, etc.) support autograd. |
| Sparse | Yes | **Yes** | COO and CSR formats with spmm, spmv, and sparse arithmetic. |

---

## See Also

- [optim](optim.md) - Optimizers for training
- [attention](attention.md) - Attention with tensor operations
- [Examples](examples.md) - Complete training examples
