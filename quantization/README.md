# quantization/

C implementation of model quantization for efficient inference and training.

## Python counterpart

**bitsandbytes** - https://github.com/TimDettmers/bitsandbytes

## Why bitsandbytes?

bitsandbytes enables:

- **8-bit optimizers** - Train with quantized optimizer states (saves memory)
- **8-bit inference** - Run models with int8 weights
- **4-bit inference** - QLoRA with 4-bit base model
- **NF4 format** - Information-theoretically optimal 4-bit format

Alternatives (AutoGPTQ, AWQ) focus on specific quantization methods. bitsandbytes is most general for training.

## Scope

This module implements:

1. **INT8 quantization** - Absmax and zero-point quantization
2. **INT4 quantization** - 4-bit weights (NF4, FP4)
3. **Mixed precision** - FP16 compute with quantized storage
4. **Dequantization** - Restore full precision for compute
5. **Quantized matmul** - Efficient int8/int4 matrix multiply

## Getting Started

### Dependencies

- tensor/ module (tensor operations)
- CUDA for GPU quantized ops

### Key Files to Study

From bitsandbytes:
- `csrc/kernels.cu` - CUDA kernels
- `bitsandbytes/functional.py` - Quantization functions

From llama.cpp:
- `ggml-quants.c` - Quantization formats (Q4_0, Q4_1, Q8_0, etc.)

### Core API

> Original design sketch, not the implemented API. The real API is `quantization/quantization.h`; see
> [`docs/quantization.md`](../docs/quantization.md).

```c
// Quantize to INT8
fq_quantized* fq_quantize_int8(tm_tensor* t, fq_method method);
tm_tensor* fq_dequantize_int8(fq_quantized* q);

// Quantize to INT4 (NF4 format)
fq_quantized* fq_quantize_nf4(tm_tensor* t);
tm_tensor* fq_dequantize_nf4(fq_quantized* q);

// Quantized matrix multiply
tm_tensor* fq_matmul_int8(tm_tensor* a, fq_quantized* b);
tm_tensor* fq_matmul_nf4(tm_tensor* a, fq_quantized* b);

// Quantization methods
typedef enum {
    FQ_ABSMAX,      // Scale by max absolute value
    FQ_ZEROPOINT,   // Scale and shift
    FQ_NF4,         // Normal Float 4-bit
    FQ_FP4,         // Float Point 4-bit
} fq_method;

// Block-wise quantization (better accuracy)
fq_quantized* fq_quantize_blockwise(tm_tensor* t, int block_size, fq_method method);
```

## Quantization Formats

**INT8 (absmax):**
```
scale = max(abs(tensor)) / 127
quantized = round(tensor / scale)
dequantized = quantized * scale
```

**NF4 (4-bit normal float):**
- 16 values optimally placed for normal distribution
- Used in QLoRA

**Block-wise:**
- Quantize in blocks of 64-256 elements
- Each block has its own scale
- Better accuracy than per-tensor

## Implementation Order

1. INT8 absmax quantization
2. INT8 dequantization
3. Quantized matmul (CPU)
4. NF4 quantization
5. Block-wise quantization
6. CUDA kernels

## References

- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [ggml quantization](https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c)
