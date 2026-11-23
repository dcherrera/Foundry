# Foundry Roadmap

Pure C implementations of 10+ Python ML tools with **full feature parity**.

---

## Module Status Overview

| Module | Replaces | Status | Notes |
|--------|----------|--------|-------|
| **tensor** | PyTorch/NumPy | ✅ API Complete | Autograd, broadcasting, all core ops |
| **optim** | torch.optim | ✅ API Complete | SGD, Adam, AdamW, schedulers, clipping |
| **tokenizer** | Tokenizers | ✅ API Complete | BPE, HuggingFace JSON loading |
| **safetensors** | Safetensors | ✅ API Complete | Read/write, mmap, metadata |
| **data** | Datasets | ✅ API Complete | JSONL loading, batching, shuffling |
| **progress** | Tqdm | ✅ API Complete | Progress bars, training metrics |
| **quantization** | Bitsandbytes | ✅ API Complete | INT8/INT4, per-channel, per-group |
| **attention** | FlashAttention | ✅ API Complete | Multi-head, KV cache, RoPE |
| **lora** | PEFT | ✅ API Complete | LoRA layers, adapters, merging |
| **distributed** | DeepSpeed/FSDP | ✅ API Complete | Collectives, gradient accum |
| **eval** | lm-eval-harness | ✅ API Complete | Perplexity, accuracy, F1 |
| **swarm** | Custom | ✅ Complete | Docker Swarm distributed training |
| **tracker** | Wandb | ❌ Missing | Experiment tracking |
| **transformer** | Transformers | 🟡 Partial | RWKV-X / RFA hybrids in `rwkvx/`; no generic GPT/Llama block library |

Also in the tree: rwkvx, potluck, hdc, hybridmem, ingot, pth, convert, threadpool, barswap, and the
CUDA, Metal and Tenstorrent backends — see the top-level README.

---

## Module Details

### 1. tensor (PyTorch/NumPy) ✅

**File:** `tensor/tensor.h`

**Implemented:**
- [x] Tensor creation (zeros, ones, full, randn, rand, empty)
- [x] Data types (F32, F64, F16, BF16, I32, I64)
- [x] Shape operations (reshape, transpose, squeeze, unsqueeze)
- [x] Element-wise ops (add, sub, mul, div, neg, exp, log, sqrt, tanh)
- [x] Activations (relu, sigmoid, gelu)
- [x] Matrix ops (matmul, bmm)
- [x] Reductions (sum, mean, max, min)
- [x] NN ops (softmax, log_softmax, layer_norm, embedding, dropout)
- [x] Loss functions (cross_entropy, mse_loss, l1_loss, smooth_l1_loss, nll_loss, kl_div, binary_cross_entropy)
- [x] Autograd (requires_grad, backward, zero_grad, detach, no_grad)
- [x] In-place ops (zeros_like, ones_like, fill_, add_, scale_)
- [x] Broadcasting support

**Needs verification against PyTorch:**
- [ ] Edge cases in broadcasting
- [ ] Gradient correctness for all ops
- [ ] Numerical stability

---

### 2. optim (torch.optim) ✅

**File:** `optim/optim.h`

**Implemented:**
- [x] SGD with momentum, weight decay, Nesterov
- [x] Adam with AMSGrad
- [x] AdamW (decoupled weight decay)
- [x] Schedulers: step, cosine, linear, warmup_cosine
- [x] Gradient clipping (norm and value)

**Needs verification:**
- [ ] Optimizer state save/load
- [ ] Comparison with PyTorch implementations

---

### 3. tokenizer (HuggingFace Tokenizers) ✅

**File:** `tokenizer/tokenizer.h`

**Implemented:**
- [x] Load from HuggingFace JSON format
- [x] Load BPE from vocab/merges files
- [x] Encode/decode text
- [x] Special tokens (BOS, EOS, PAD, UNK)
- [x] Batch encoding

**Needs verification:**
- [ ] Unicode handling edge cases
- [ ] Large vocabulary performance
- [ ] Comparison with `tokenizers` library

---

### 4. safetensors (HuggingFace Safetensors) ✅

**File:** `safetensors/safetensors.h`

**Implemented:**
- [x] Read safetensors files (mmap)
- [x] Read from memory buffer
- [x] Write safetensors files
- [x] All data types (F32, F64, F16, BF16, I8-I64, U8-U64, BOOL)
- [x] Metadata read/write
- [x] Zero-copy data access

**Needs verification:**
- [ ] Sharded checkpoint support
- [ ] Large file handling
- [ ] Comparison with Python safetensors

---

### 5. data (HuggingFace Datasets) ✅

**File:** `data/data.h`

**Implemented:**
- [x] Load JSONL files
- [x] Sample creation/manipulation
- [x] Dataset shuffling
- [x] Train/test splitting
- [x] DataLoader with batching

**Needs verification:**
- [ ] Streaming large datasets
- [ ] Memory-mapped loading
- [ ] Comparison with `datasets` library

---

### 6. progress (Tqdm) ✅

**File:** `progress/progress.h`

**Implemented:**
- [x] Progress bars with ETA
- [x] Training state logging
- [x] Nested progress bars
- [x] Postfix formatting
- [x] Rate calculation

**Needs verification:**
- [ ] Terminal width handling
- [ ] Nested bar rendering

---

### 7. quantization (Bitsandbytes) ✅

**File:** `quantization/quantization.h`

**Implemented:**
- [x] INT8 quantization (symmetric)
- [x] INT8 per-channel quantization
- [x] INT4 per-group quantization
- [x] Dequantization
- [x] Quantized matmul

**Needs verification:**
- [ ] NF4 quantization (QLoRA)
- [ ] Dynamic quantization
- [ ] Comparison with bitsandbytes

---

### 8. attention (FlashAttention) ✅

**File:** `attention/attention.h`

**Implemented:**
- [x] Scaled dot-product attention
- [x] Multi-head attention
- [x] Self-attention
- [x] Attention with mask
- [x] KV cache for inference
- [x] RoPE (Rotary Position Embedding)
- [x] Causal masking

**Needs verification:**
- [ ] Flash Attention algorithm (memory-efficient)
- [ ] Numerical stability at scale
- [ ] Comparison with FlashAttention

---

### 9. lora (PEFT) ✅

**File:** `lora/lora.h`

**Implemented:**
- [x] LoRA layer creation
- [x] LoRA forward pass
- [x] Weight merging
- [x] Adapter save/load (safetensors)
- [x] Configurable rank/alpha/dropout

**Needs verification:**
- [ ] QLoRA support
- [ ] Comparison with PEFT library

---

### 10. distributed (DeepSpeed/FSDP basics) ✅

**File:** `distributed/distributed.h`

**Implemented:**
- [x] Process group initialization
- [x] All-reduce, broadcast, reduce, all-gather
- [x] Barrier synchronization
- [x] Gradient accumulation
- [x] Data sharding

**Needs verification:**
- [ ] MPI backend integration
- [ ] Multi-node support

---

### 11. eval (lm-eval-harness) ✅

**File:** `eval/eval.h`

**Implemented:**
- [x] Perplexity evaluation
- [x] Accuracy/Top-K accuracy
- [x] F1 score
- [x] Multiple choice tasks
- [x] Few-shot support
- [x] HellaSwag-style tasks

**Needs verification:**
- [ ] Standard benchmark compatibility
- [ ] Comparison with lm-eval-harness

---

### 12. swarm (Distributed Training) ✅ COMPLETE

**Files:** `swarm/` directory

**Implemented:**
- [x] Parameter server architecture
- [x] Coordinator (weight distribution, gradient aggregation)
- [x] Worker (gradient computation, compression)
- [x] TopK gradient compression
- [x] FP16 compression
- [x] Hardware profile auto-detection
- [x] Docker Swarm deployment
- [x] Fault tolerance (reconnection)
- [x] TOML config parsing
- [x] 14 passing tests

---

## Missing Modules

### 13. tracker (Wandb) ❌ NOT IMPLEMENTED

**Should implement:**
- [ ] Experiment initialization
- [ ] Metric logging (scalars, histograms)
- [ ] Config tracking
- [ ] Artifact storage
- [ ] Local logging (SQLite/JSON fallback)
- [ ] Optional cloud sync

---

### 14. transformer (Transformers) 🟡 PARTIAL

RWKV-X and RFA (RWKV-7 + Qwen GQA) hybrids exist in `rwkvx/`; `convert/` maps Llama-family
safetensors. A generic architecture library does not.

**Should implement:**
- [ ] Embedding layer (token + position)
- [ ] TransformerBlock (attention + MLP)
- [ ] GPT-2 architecture
- [ ] Llama architecture (RMSNorm, SwiGLU, RoPE)
- [ ] RWKV architecture
- [ ] Weight loading from safetensors
- [ ] HuggingFace weight name mapping

---

### 15. corpus-archaeology (research idea — back-pocket, not committed) 💡

A model-forensics tool: take **open weights** and infer *what a model was trained on* and *which data actually moved which capability* — across many models, comparatively. Not data reconstruction (training is lossy compression; the full corpus is information-theoretically unrecoverable past the memorized slice) — it's **distributional/provenance analysis via forward passes only** (no gradients). Core signal: **comparative perplexity/loss fingerprinting** — score each model over a battery of candidate datasets; the loss profile is a fingerprint of what it knows comfortably. From that: cluster models by **data lineage**, run **membership/dataset inference** ("was source X in training?"), regress **dataset presence → eval score** to find what's *effective* (and what's redundant). Optional tier: a **memorization-extraction sweep** (sample heavily, flag abnormally-low-loss verbatim spans) to recover the small memorized fraction. Natural Foundry fit (it's the tensor/compute/research base, not serving) and a **capacity-bound batch workload** — hold many models resident, fire probes — i.e. *exactly* what a slow-but-huge-VRAM multi-K80 host is good at. Turns such a host into a model-archaeology rig, not just an inference server.

- [ ] Perplexity/loss fingerprinting harness (model × dataset matrix)
- [ ] Membership / dataset inference scorers
- [ ] dataset-presence → capability correlation (regress vs eval scores)
- [ ] (tier 2) memorization-extraction sweep (low-loss verbatim detection + dedup)

---

## Verification Checklist

Before each module is production-ready, verify against the original Python implementation:

1. **API Compatibility** - Does it support the same core operations?
2. **Numerical Accuracy** - Do outputs match within tolerance?
3. **Edge Cases** - Does it handle empty inputs, large inputs, special values?
4. **Performance** - Is it at least as fast for common operations?
5. **Memory** - Does it avoid unnecessary copies/allocations?

---

## Next Steps

1. **transformer** - Generic model architectures (GPT, Llama); RWKV-X/RFA exist in `rwkvx/`
2. **tracker** - Experiment tracking (Wandb replacement)
3. **Full feature parity passes** - Implement ALL missing PyTorch/library features
4. **GPU support** - CUDA, Metal and Tenstorrent backends exist; widen op coverage, Vulkan planned
5. Verification passes on all modules

---

## Test Coverage

`make test`: 504 tests across 30 suites, all passing (2026-11-23); per-suite counts are in the
top-level README. `make test-swarm`: 14.

---

## Hardware Targets

Primary target: Train 1B RWKV-x model across ~40 heterogeneous nodes:
- Raspberry Pi 4/5
- Laptops (Intel/AMD)
- Desktop PCs
- Small servers
- Connected via WiFi/Starlink

The swarm module is designed specifically for this use case with:
- Async gradient updates (workers don't block each other)
- TopK compression (90% bandwidth reduction)
- Hardware-adaptive batch sizes
- Fault tolerance (nodes can drop/rejoin)
