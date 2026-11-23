# rwkvx_generate

Text generation CLI for RWKV-X models. Supports INT8 quantization and HybridMem disk-backed memory.

**Source:** `examples/rwkvx_generate.c`
**Build:** `make USE_BLAS=1 rwkvx-generate`

---

## Overview

Interactive text generation from RWKV-X hybrid models (RWKV-7 recurrent + sparse attention). Processes a prompt, then generates tokens autoregressively with configurable sampling.

Features:
- Batched prefill for fast prompt processing
- INT8 quantized inference (`--quantize`)
- HybridMem disk-backed KV cache (`--hybridmem-dir`)
- Nucleus (top-p), top-k, and temperature sampling
- Streaming output (tokens printed as generated)

---

## Usage

```bash
./examples/rwkvx_generate \
  --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "Hello world" \
  --max-tokens 64 \
  --temperature 1.0 \
  --top-p 0.7
```

### With INT8 quantization (faster decode, less memory)

```bash
./examples/rwkvx_generate \
  --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "Hello world" \
  --max-tokens 64 \
  --quantize
```

### With HybridMem disk-backed memory

```bash
./examples/rwkvx_generate \
  --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
  --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "Hello world" \
  --max-tokens 64 \
  --hybridmem-dir /tmp/hm_store \
  --cache-size 512 \
  --hm-top-k 10
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | (required) | Path to .safetensors model file |
| `--vocab PATH` | (required) | Path to RWKV vocab file |
| `--prompt TEXT` | `"Hello"` | Input prompt text |
| `--max-tokens N` | 64 | Maximum tokens to generate |
| `--temperature F` | 1.0 | Sampling temperature (0 = greedy) |
| `--top-p F` | 0.7 | Nucleus sampling threshold |
| `--top-k N` | 100 | Top-k filtering |
| `--quantize` | off | Quantize weights to INT8 |
| `--hybridmem-dir D` | off | Enable HybridMem at directory D |
| `--cache-size N` | 2048 | KV cache size per attention layer |
| `--hm-top-k N` | 5 | HybridMem retrieval count |

---

## Build Variants

```bash
# 0.2B: BLAS + INT8 (best for small models)
make clean && make USE_BLAS=1 rwkvx-generate

# 3.6B: BLAS + threads (essential for large models)
make clean && make USE_BLAS=1 USE_THREADS=1 rwkvx-generate

# 3.6B: BLAS + threads + Metal GPU
make clean && make USE_BLAS=1 USE_THREADS=1 USE_METAL=1 rwkvx-generate
```

---

## Output

Prompt text is printed immediately, then generated tokens stream to stdout. Performance stats go to stderr:

```
Loading tokenizer from rwkvx/data/rwkv_vocab_v20230424.txt...
Loading model from model.safetensors (INT8 quantized)...
Model: n_embd=768 n_head=12 n_rwkv=12 n_attn=4 vocab=65536
Prompt: "Hello" -> 1 tokens
Hello, I'm a new member of this forum...
Generated: 64 tokens in 224.3ms (285.3 tok/s)
```

---

## See Also

- [rfa-generate](rfa-generate.md) - Text generation for RFA hybrid models
