# tensor/

C implementation of tensor operations with automatic differentiation.

## Python counterpart

**PyTorch** - https://github.com/pytorch/pytorch

## Why PyTorch?

PyTorch is the dominant framework for LLM training. TensorFlow and JAX are alternatives but PyTorch has won the LLM space. Core features:

- N-dimensional tensor operations
- Automatic differentiation (autograd)
- GPU acceleration
- Neural network primitives

## Scope

This module implements:

1. **Tensor data structure** - N-dimensional arrays with strides, shapes, dtypes
2. **Basic operations** - add, sub, mul, div, matmul, transpose, reshape
3. **Reductions** - sum, mean, max, min
4. **Autograd** - Computation graph, backward pass, gradient accumulation
5. **NN operations** - softmax, layer norm, embedding lookup

## Getting Started

### Dependencies

- C99 compiler
- BLAS library (OpenBLAS, MKL, or Accelerate)
- Optional: CUDA toolkit, Metal SDK

### Key Files to Study

From PyTorch:
- `c10/core/TensorImpl.h` - Tensor implementation
- `torch/csrc/autograd/` - Autograd engine
- `aten/src/ATen/native/` - Native operations

From ggml:
- `ggml.h` / `ggml.c` - Simpler tensor implementation
- Already C, good reference

### Core API

```c
// Tensor creation
tm_tensor* tm_zeros(int64_t* shape, int ndim, tm_dtype dtype);
tm_tensor* tm_randn(int64_t* shape, int ndim, tm_dtype dtype);

// Operations (all support autograd)
tm_tensor* tm_add(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_matmul(tm_tensor* a, tm_tensor* b);
tm_tensor* tm_softmax(tm_tensor* x, int dim);

// Autograd
void tm_backward(tm_tensor* loss);
void tm_zero_grad(tm_tensor** params, int n);
```

## Implementation Order

1. Tensor struct and memory management
2. Basic ops (add, mul, matmul) - CPU only
3. Autograd graph building
4. Backward pass for basic ops
5. NN ops (softmax, layernorm)
6. GPU backends

## References

- [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [ggml](https://github.com/ggerganov/ggml)
- [tinygrad](https://github.com/tinygrad/tinygrad) - Minimal autograd
- [micrograd](https://github.com/karpathy/micrograd) - Educational autograd
