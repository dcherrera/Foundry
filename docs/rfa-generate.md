# rfa_generate

Text generation CLI for RFA hybrid models (RWKV-7 + Qwen GQA).

**Source:** `examples/rfa_generate.c`
**Build:** `make USE_BLAS=1 rfa-generate`

---

## Overview

Interactive text generation from the RWKV-7 + Qwen hybrid model. Processes a prompt token-by-token through interleaved RWKV-7 recurrent blocks and Qwen GQA attention blocks, then generates tokens autoregressively.

The RFA model uses Qwen's vocabulary (151K tokens) but the RWKV vocab file for tokenization (shared encoding).

---

## Usage

### Base model inference

```bash
./examples/rfa_generate \
  --model ../models/base/rwkv-qwen-hybrid-1.5B.safetensors \
  --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
  --prompt "Hello world" \
  --max-tokens 64 \
  --temperature 1.0 \
  --top-p 0.7
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | (required) | Merged hybrid safetensors file |
| `--vocab PATH` | (required) | RWKV vocab file |
| `--prompt TEXT` | `"Hello"` | Input prompt text |
| `--max-tokens N` | 64 | Maximum tokens to generate |
| `--max-cache N` | 2048 | KV cache size per Qwen layer |
| `--temperature F` | 1.0 | Sampling temperature (0 = greedy) |
| `--top-p F` | 0.7 | Nucleus sampling threshold |
| `--top-k N` | 0 (off) | Top-k filtering |

---

## Output

Prompt and generated text go to stdout. Stats printed after generation:

```
Loading model: rwkv-qwen-hybrid-1.5B.safetensors
Config: 1536d, 28 layers (21 RWKV + 7 Qwen), vocab=151936
Prompt: "Hello" (1 tokens)

Hello, I am a student at the University of...

--- Stats ---
Prefill: 1 tokens in 0.05s (20.0 tok/s)
Decode:  64 tokens in 3.20s (20.0 tok/s)
```

---

## See Also

- [rfa-train](rfa-train.md) - LoRA training for RFA hybrid models
- [rwkvx-generate](rwkvx-generate.md) - Text generation for RWKV-X models
