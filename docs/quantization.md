# quantization

Tensor quantization for model compression. Python counterpart: Bitsandbytes.

**Header:** `quantization/quantization.h`
**Source:** `quantization/quantization.c`
**Status:** API Complete

---

## Overview

The quantization module provides:
- INT8 quantization (symmetric, per-tensor or per-channel)
- INT4 per-group quantization
- Ternary quantization (2-bit, BitNet b1.58 absmean)
- Quantized matrix multiplication (with native ternary kernel)
- Dequantization
- Offline model export (see [quantize-model](quantize-model.md))

---

## Quick Start

```c
#include "quantization/quantization.h"

// Quantize weights to INT8
int64_t shape[] = {4096, 4096};
float* weights = load_weights();

fq_tensor* qw = fq_quantize_int8(weights, shape, 2);
printf("Compressed to %.1f MB (was %.1f MB)\n",
       fq_memory_usage(qw) / 1e6,
       4096 * 4096 * 4 / 1e6);

// Quantized matmul
float* activations = get_activations();
float* output = fq_matmul(activations, qw, batch, 4096, 4096);

free(output);
fq_free(qw);
```

### Ternary Quantization

```c
// Quantize to {-1, 0, +1} using absmean threshold
fq_tensor* qt = fq_quantize_ternary(weights, shape, 2);

// Or with manual threshold
fq_tensor* qt2 = fq_quantize_ternary_threshold(weights, shape, 2, 0.05f);

// Per-channel ternary (different threshold per output channel)
fq_tensor* qt3 = fq_quantize_ternary_per_channel(weights, shape, 2, 0);

// Dequantize: ternary_value * scale
float* restored = fq_dequantize(qt);
```

---

## Ternary Format

### Encoding

Each value is encoded in 2 bits: `-1 = 0b00`, `0 = 0b01`, `+1 = 0b10`. Four values are packed per byte, LSB-first:

```
byte = (val3 << 6) | (val2 << 4) | (val1 << 2) | val0
```

Packed size: `(numel + 3) / 4` bytes.

### Absmean Thresholding (BitNet b1.58)

```
threshold = mean(|w|)
w > threshold   ->  +1
w < -threshold  ->  -1
otherwise       ->   0
scale = threshold  (for reconstruction: ternary_value * scale)
```

### Compression

| Dtype | Bits/value | Compression vs F32 |
|-------|------------|-------------------|
| INT8 | 8 | 4x |
| INT4 | 4 | 8x |
| Ternary | 2 | 16x |

Real-world result on RWKV-X 0.2B: 837 MB -> 53 MB (15.8x).

---

## Ternary Quantization (BitNet b1.58)

2-bit ternary quantization constrains weights to `{-1, 0, +1}`, enabling integer-only matmul (no floating-point multiply in the inner loop). This is the foundation for 1-bit LLM architectures like BitNet b1.58.

### How It Works

1. **Absmean scale**: `gamma = mean(|weights|)` over all elements
2. **Quantize**: `q = round(w / gamma)`, clamped to `{-1, 0, +1}`
3. **Pack**: 4 values per byte (2 bits each), encoding `00=-1, 01=0, 10=+1`
4. **Matmul**: inner loop uses only add/subtract/skip (no multiply), then scales result by `gamma`

### Compression

| Dtype | Bits/weight | Compression vs FP32 |
|-------|------------|---------------------|
| INT8 | 8 | 4x |
| INT4 | 4 | 8x |
| Ternary | 2 | 16x |

### Usage

```c
#include "quantization/quantization.h"

// Quantize a weight matrix to ternary
int64_t shape[] = {4096, 4096};
float* weights = load_weights();

fq_tensor* qw = fq_quantize_ternary(weights, shape, 2);
// 4096x4096 = 16M weights -> 4 MB (was 64 MB in FP32)

// Native ternary matmul (no dequantization, integer-only inner loop)
float* activations = get_activations();  // [batch, 4096]
float* output = fq_matmul_ternary(activations, qw, batch, 4096, 4096);

// Also works through the generic fq_matmul / fq_matmul_acc interface
// (auto-dispatches to ternary kernel when B is FQ_TERNARY)
float* output2 = fq_matmul(activations, qw, batch, 4096, 4096);

free(output);
free(output2);
fq_free(qw);
```

### Native Matmul Kernel

The ternary matmul kernel unpacks weights on-the-fly during the inner loop. Each 2-bit weight maps to `{-1, 0, +1}` via a lookup table, so the inner loop body is effectively:

```
sum += A[k] * w   // where w is -1, 0, or +1
```

The compiler optimizes this to conditional add/subtract/nop. No temporary dequantized buffer is allocated.

---

## API Reference

### Quantization

```c
// INT8
fq_tensor* fq_quantize_int8(const float* data, int64_t* shape, int ndim);
fq_tensor* fq_quantize_int8_per_channel(const float* data, int64_t* shape,
                                         int ndim, int channel_dim);

// INT4
fq_tensor* fq_quantize_int4(const float* data, int64_t* shape,
                             int ndim, int group_size);

// Ternary
fq_tensor* fq_quantize_ternary(const float* data, int64_t* shape, int ndim);
fq_tensor* fq_quantize_ternary_threshold(const float* data, int64_t* shape,
                                          int ndim, float threshold);
fq_tensor* fq_quantize_ternary_per_channel(const float* data, int64_t* shape,
                                            int ndim, int channel_dim);

// Generic dispatcher
fq_tensor* fq_quantize(const float* data, int64_t* shape, int ndim,
                       fq_dtype dtype, fq_mode mode, fq_granularity granularity,
                       int channel_dim, int group_size);

void fq_free(fq_tensor* qt);
```

### Dequantization

```c
float* fq_dequantize(fq_tensor* qt);
float* fq_dequantize_range(fq_tensor* qt, int64_t start, int64_t end);
```

### Quantized Operations

```c
float* fq_matmul(const float* A, fq_tensor* B, int M, int K, int N);
void fq_matmul_acc(const float* A, fq_tensor* B, float* C, int M, int K, int N);

// Native ternary matmul (integer-only, no dequant)
float* fq_matmul_ternary(const float* A, fq_tensor* B, int M, int K, int N);
void fq_matmul_ternary_acc(const float* A, fq_tensor* B, float* C, int M, int K, int N);
```

### Types

```c
typedef enum {
    FQ_INT8,    // 8-bit signed
    FQ_INT4,    // 4-bit signed (packed)
    FQ_UINT8,   // 8-bit unsigned
    FQ_UINT4,   // 4-bit unsigned (packed)
    FQ_TERNARY, // 2-bit ternary {-1, 0, +1} (packed, 4 per byte)
} fq_dtype;

typedef enum {
    FQ_SYMMETRIC,   // Zero-point is always 0
    FQ_ASYMMETRIC,  // Zero-point can be non-zero
} fq_mode;

typedef enum {
    FQ_PER_TENSOR,  // Single scale for entire tensor
    FQ_PER_CHANNEL, // Different scale for each channel
    FQ_PER_GROUP,   // Group quantization (like GPTQ)
} fq_granularity;
```

### Utility

```c
const char* fq_dtype_name(fq_dtype dtype);
float fq_dtype_bytes(fq_dtype dtype);  // 1 for int8, 0.5 for int4, 0.25 for ternary
void fq_compute_scale(const float* data, int64_t n, fq_mode mode,
                      float* scale, int32_t* zero_point);
size_t fq_memory_usage(fq_tensor* qt);
void fq_print(fq_tensor* qt);
float fq_quantization_error(const float* original, fq_tensor* quantized);
```

---

## Export API

The export module provides offline model quantization — loads a safetensors model, quantizes all weight tensors, and saves a new safetensors file.

**Header:** `quantization/export.h`
**Source:** `quantization/export.c`

### One-shot Export

```c
#include "quantization/export.h"

fq_export_config cfg = fq_export_config_defaults();
cfg.target_dtype = FQ_TERNARY;

fq_export_quantized("model.safetensors", "model-ternary.safetensors", &cfg);
```

### Step-by-step Export (for GUI)

```c
fq_export_config cfg = fq_export_config_defaults();
cfg.target_dtype = FQ_INT8;

fq_export_state* s = fq_export_begin("model.safetensors", &cfg);

while (fq_export_step(s)) {
    printf("%.0f%% - %s\n",
           fq_export_progress(s) * 100,
           fq_export_current_tensor(s));
}

fq_export_finish(s, "model-q8.safetensors");
```

### Export Config

```c
typedef struct {
    fq_dtype target_dtype;        // FQ_INT8, FQ_INT4, FQ_TERNARY
    fq_granularity granularity;   // PER_TENSOR, PER_CHANNEL, PER_GROUP
    int group_size;               // for per-group (default 128)
    float ternary_threshold;      // 0.0 = auto absmean, >0 = manual

    bool skip_embeddings;         // don't quantize embed layers (default true)
    bool skip_norms;              // don't quantize norm layers (default true)
    bool skip_head;               // don't quantize lm_head (default false)
    bool skip_biases;             // don't quantize biases (default true)
    bool skip_1d;                 // don't quantize 1D tensors (default true)
    int min_elements;             // skip tensors < this size (default 1024)

    void (*progress_cb)(int current, int total, const char* name, void* ud);
    void* userdata;
} fq_export_config;
```

### Export Functions

```c
fq_export_config fq_export_config_defaults(void);

// One-shot
bool fq_export_quantized(const char* input, const char* output, fq_export_config* cfg);

// Step-by-step
fq_export_state* fq_export_begin(const char* input, fq_export_config* cfg);
int fq_export_tensor_count(fq_export_state* s);
bool fq_export_step(fq_export_state* s);
float fq_export_progress(fq_export_state* s);
const char* fq_export_current_tensor(fq_export_state* s);
bool fq_export_finish(fq_export_state* s, const char* output);
void fq_export_abort(fq_export_state* s);
```

### Output Format

Quantized tensors are stored in safetensors format with auxiliary metadata tensors:

| Tensor | Dtype | Description |
|--------|-------|-------------|
| `layer.weight` | `I8` / `U8` | Quantized data (INT8 raw, INT4/ternary packed) |
| `layer.weight.__scale` | `F32` | Scale factors for reconstruction |
| `layer.weight.__shape` | `I64` | Original dimensions |
| `layer.weight.__zp` | `I32` | Zero points (asymmetric only) |

Skipped tensors (norms, biases, embeddings) are passed through unchanged.

File-level metadata:

```json
{
    "quantization.dtype": "TERNARY",
    "quantization.granularity": "per_tensor",
    "quantization.group_size": "128",
    "quantization.format_version": "1"
}
```

---

## Testing

```bash
cd foundry

# Quantization core (INT8, INT4, ternary)
make test-quantization    # 43 tests

# Export pipeline (round-trip, skip logic, metadata, step API)
make test-export          # 14 tests
```

---

## See Also

- [quantize-model](quantize-model.md) - CLI tool for offline model quantization
- [tensor](tensor.md) - Tensor operations
- [safetensors](safetensors.md) - Model file I/O
- [lora](lora.md) - LoRA for efficient fine-tuning
