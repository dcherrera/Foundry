# rfa_train

LoRA training CLI for RFA hybrid models (RWKV-7 + Qwen GQA). Supports multi-threaded parallel training and INT8 quantized base weights.

**Source:** `examples/rfa_train.c`
**Build:** `make USE_BLAS=1 rfa-train`
**Tests:** `make USE_BLAS=1 test-rfa-train` (9 tests), `make USE_BLAS=1 test-rfa-parallel` (6 tests)

---

## Overview

Trains LoRA adapters on the 7 Qwen GQA blocks in the RWKV-7 + Qwen hybrid model. The 21 RWKV-7 blocks are frozen. Uses pre-tokenized binary data (format under [Pre-Tokenization](#pre-tokenization)).

7 LoRA layers per Qwen block (Q, K, V, O, gate, up, down) = **49 LoRA layers** total.

At rank 32: ~9.2M trainable parameters.

Features:
- Multi-threaded data-parallel training (`--parallel N`)
- INT8 quantized base weights (`--quantize`)
- Cross-machine LoRA adapter merging (via `rfa_lora_merge`)
- Gradient checkpointing (recompute vs store intermediates)
- AdamW optimizer with warmup + cosine annealing

---

## Quick Start

### 1. Pre-tokenize data

Tokenize your JSONL with the Qwen tokenizer into the binary format under
[Pre-Tokenization](#pre-tokenization). The tokenizer script is not part of this repo.

### 2. Train (single-threaded)

```bash
./examples/rfa_train \
  --model ../models/base/rwkv-qwen-hybrid-1.5B.safetensors \
  --data ../models/datasets/oasst1_qwen_tokens.bin \
  --output ../models/loras/rfa-qwen-lora/ \
  --lora-rank 32 --lora-alpha 64 \
  --lr 2e-5 --epochs 3 --max-seq-len 256 --log-every 10
```

### 3. Train (parallel with INT8)

```bash
./examples/rfa_train \
  --model ../models/base/rwkv-qwen-hybrid-1.5B.safetensors \
  --data ../models/datasets/oasst1_qwen_tokens.bin \
  --output ../models/loras/rfa-qwen-lora/ \
  --lora-rank 32 --lora-alpha 64 \
  --lr 2e-5 --epochs 3 --max-seq-len 256 \
  --parallel 6 --quantize
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | (required) | Hybrid model safetensors file |
| `--data PATH` | (required) | Pre-tokenized binary data file |
| `--output DIR` | (required) | Output directory for LoRA checkpoints |
| `--lora-rank N` | 32 | LoRA rank |
| `--lora-alpha F` | 64 | LoRA alpha scaling |
| `--lr F` | 2e-5 | Learning rate |
| `--epochs N` | 3 | Training epochs |
| `--max-seq-len N` | 256 | Max sequence length per example |
| `--warmup N` | 100 | LR warmup steps |
| `--log-every N` | 10 | Log metrics every N steps |
| `--save-every N` | 1000 | Save checkpoint every N steps |
| `--grad-clip F` | 1.0 | Gradient clipping norm |
| `--max-cache N` | 2048 | Max KV cache per Qwen block |
| `--parallel N` | 1 | Number of training threads |
| `--quantize` | off | Use INT8 quantized base weights |

---

## Parallel Training

When `--parallel N` is set (N > 1), the training data is split into N chunks. Each thread independently trains a LoRA adapter on its chunk with its own state, optimizer, and adapter. After all threads complete, the N adapters are averaged element-wise (FedAvg) and saved as `adapter_final.safetensors`.

```
Main Thread
  1. Load model (shared, read-only)
  2. Split dataset into N chunks
  3. Spawn N pthreads
  4. Wait for all threads to finish
  5. Average N adapters -> save merged

  Thread 0          Thread 1          Thread N
  ┌─────────┐      ┌─────────┐      ┌─────────┐
  │ state    │      │ state    │      │ state    │
  │ adapter  │      │ adapter  │      │ adapter  │
  │ optimizer│      │ optimizer│      │ optimizer│
  │ chunk    │      │ chunk    │      │ chunk    │
  └─────────┘      └─────────┘      └─────────┘
       ↑ shared model (read-only) ↑
```

Output files with parallel training:
```
output_dir/
├── thread_0.safetensors          # Per-thread adapter
├── thread_1.safetensors
├── ...
└── adapter_final.safetensors     # Averaged adapter
```

---

## INT8 Quantized Training

The `--quantize` flag quantizes all base model linear weights from f32 to INT8 (per-row symmetric). Forward-pass matvecs use NEON 4x-unrolled INT8 dot products (~3x faster). LoRA weights remain f32 for gradient accuracy.

Both f32 and INT8 weights are kept in memory — f32 is needed for backward pass gradient computation through base weights.

---

## Pre-Tokenization

The RFA model uses Qwen's vocabulary (151K tokens). Data must be pre-tokenized (with the
model's `tokenizer.json`; the tokenizer script is not part of this repo) into:

Binary format: `[n_examples: u32] [seq_len: u32, tokens: u32[seq_len]] x N`

### Dataset splitting for multi-machine training

For cross-machine training, split the examples into one file per machine (same format),
weighted by each machine's throughput.

---

## Cross-Machine Training

Train on separate machines, then merge adapters:

```bash
# Machine 1 (24 GB):
./examples/rfa_train --model model.safetensors \
  --data chunk_0.bin --output loras_a/ \
  --parallel 6 --quantize --epochs 3

# Machine 2 (8 GB):
./examples/rfa_train --model model.safetensors \
  --data chunk_1.bin --output loras_b/ \
  --parallel 2 --quantize --epochs 3

# Merge:
scp machine2:loras_b/adapter_final.safetensors ./
./examples/rfa_lora_merge \
  --output merged.safetensors \
  loras_a/adapter_final.safetensors loras_b/adapter_final.safetensors
```

---

## Memory Requirements

| Config | Memory (1.5B model) |
|--------|-------------------|
| Single-thread, f32 | ~8 GB |
| Single-thread, INT8 (`--quantize`) | ~9.5 GB |
| 6 threads, INT8 (`--parallel 6 --quantize`) | ~20 GB |

Reduce memory with shorter sequences (`--max-seq-len 128`) or fewer threads.

---

## Output Files

```
output_dir/
├── checkpoint_1000.safetensors   # Step 1000
├── checkpoint_2000.safetensors   # Step 2000
└── adapter_final.safetensors     # End of training (or merged from threads)
```

---

## See Also

- [rfa-generate](rfa-generate.md) - Inference with trained adapters
- [rfa-lora-merge](rfa-lora-merge.md) - Merge adapters from multiple machines
- [lora](lora.md) - LoRA adapter module
