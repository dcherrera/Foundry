# quantize_model

CLI tool for offline model quantization. Converts safetensors models to INT8, INT4, or ternary format.

**Source:** `examples/quantize_model.c`
**Build:** `make quantize-model`

---

## Overview

Loads a safetensors model, quantizes all weight tensors to the target dtype, and saves a new safetensors file. Automatically skips norm layers, biases, embeddings, and small tensors. Reports compression ratio when done.

---

## Usage

```bash
./examples/quantize_model \
  --input model.safetensors \
  --output model-quantized.safetensors \
  --dtype int8|int4|ternary \
  [--granularity per_tensor|per_channel|per_group] \
  [--group-size 128] \
  [--threshold 0.0]
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | (required) | Input safetensors model |
| `--output PATH` | (required) | Output quantized safetensors |
| `--dtype TYPE` | (required) | `int8`, `int4`, or `ternary` |
| `--granularity MODE` | `per_tensor` | `per_tensor`, `per_channel`, or `per_group` |
| `--group-size N` | `128` | Group size for per-group quantization |
| `--threshold F` | `0.0` | Ternary threshold (0 = auto absmean) |

---

## Examples

### INT8 quantization (4x compression)

```bash
make USE_BLAS=1 quantize-model

./examples/quantize_model \
  --input ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --output ../models/base/RWKV-X-0.2B-q8.safetensors \
  --dtype int8
```

### INT4 per-group quantization (8x compression)

```bash
./examples/quantize_model \
  --input ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --output ../models/base/RWKV-X-0.2B-q4.safetensors \
  --dtype int4 \
  --granularity per_group \
  --group-size 64
```

### Ternary quantization (16x compression)

```bash
./examples/quantize_model \
  --input ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --output ../models/base/RWKV-X-0.2B-ternary.safetensors \
  --dtype ternary
```

### Ternary with manual threshold

```bash
./examples/quantize_model \
  --input ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --output ../models/base/RWKV-X-0.2B-ternary-t05.safetensors \
  --dtype ternary \
  --threshold 0.05
```

---

## Output

```
Quantizing model:
  Input:       ../models/base/RWKV-X-0.2B-64k-Base.safetensors (836.8 MB)
  Output:      /tmp/rwkvx-ternary.safetensors
  Dtype:       TERNARY
  Granularity: per_tensor

  [========================================] 100% (443/443) rwkv.ln_out.weight

  Input:  836.8 MB
  Output: 53.1 MB
  Ratio:  15.8x compression

Done.
```

---

## Skip Logic

The following tensors are passed through without quantization (preserving original F32):

| Pattern | Example | Why |
|---------|---------|-----|
| Norm layers | `ln_0.weight`, `rmsnorm.weight` | Norms are sensitive to precision |
| Biases | `layer.0.bias` | Small tensors, negligible savings |
| Embeddings | `embed.weight`, `wte.weight` | Critical for token representation |
| 1D tensors | Any 1D param | Typically norms/biases |
| Small tensors | < 1024 elements | Not worth the overhead |
| Non-float | I8, I32, etc. | Already quantized or integer params |

All skip behavior is configurable via the C export API (`fq_export_config`).

---

## Output Format

The quantized safetensors file contains:

- **Quantized weight tensors** stored as `I8` (INT8) or `U8` (INT4/ternary packed)
- **Scale tensors** (`name.__scale`) as `F32` for reconstruction
- **Shape tensors** (`name.__shape`) as `I64` storing original dimensions
- **Passthrough tensors** in original dtype (norms, biases, embeddings)
- **Metadata** with quantization parameters (`quantization.dtype`, `quantization.granularity`, etc.)

---

## See Also

- [quantization](quantization.md) - Quantization module API and ternary format details
- [safetensors](safetensors.md) - Safetensors file format
