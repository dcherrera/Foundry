# RADLADS in Foundry

> Adding transformer-to-RWKV distillation tooling to Foundry as a first-class tool.

**Status:** proposal; no `radlads/` module exists yet.

---

## What is RADLADS?

[RADLADS](https://arxiv.org/abs/2505.03005) (Rapid Attention Distillation to Linear Attention Decoders at Scale) is a protocol for converting pretrained transformer models into RWKV linear attention models. Developed by [Recursal](https://github.com/recursal/RADLADS-paper) (Apache 2.0 license).

It has been proven at scale:
- Converted Qwen2.5 models (7B, 32B, 72B) into RWKV-6 and RWKV-7 variants
- Uses only 350-700M tokens (~0.005% of original pretraining data)
- Retains ~95-98% of teacher model quality
- The 72B conversion cost less than $2,000

Currently implemented in Python/PyTorch. We want to bring this capability into Foundry as pure C.

---

## Why Add RADLADS to Foundry?

### 1. CPU Cluster Training

The original RADLADS code assumes GPU (A100/H100). A Foundry implementation enables:
- Training on CPU clusters with 8GB+ per node
- Using our existing Docker Swarm infrastructure
- Model parallelism via Foundry's `distributed/` and `swarm/` modules
- Quantized teacher inference (Q8) to fit in memory

### 2. No Python Stack

Foundry means no Python and no PyTorch at run time, and no CUDA toolkit to build (see NOTICE for what Foundry does link). The entire conversion pipeline runs as a single compiled binary or across a distributed cluster.

### 3. Tighter Memory Control

C gives us byte-level control over memory layout. Critical for fitting a 7B teacher + student in 8GB per node via:
- Memory-mapped weights (safetensors mmap)
- Layer-by-layer processing (don't load full model)
- Mixed precision (Q8 frozen weights, FP32 trainable)
- Gradient checkpointing

### 4. BootAI / Bare Metal Integration

A Foundry-native RADLADS means the conversion tooling can eventually run on BootAI itself — converting models directly on bare metal hardware.

---

## What RADLADS Needs from Foundry

### Already Implemented

| Foundry Module | RADLADS Use | Status |
|----------------|-------------|--------|
| `tensor/` | All tensor ops, autograd, backward pass | ✅ |
| `attention/` | Multi-head attention, KV cache, RoPE | ✅ |
| `safetensors/` | Load teacher/student weights, save checkpoints | ✅ |
| `tokenizer/` | BPE tokenizer (load Qwen HuggingFace JSON) | ✅ |
| `data/` | JSONL data loading, batching, shuffling | ✅ |
| `optim/` | Adam/AdamW optimizers, cosine scheduler | ✅ |
| `distributed/` | All-reduce, gradient accumulation | ✅ |
| `swarm/` | Parameter server, gradient compression, Docker Swarm | ✅ |
| `quantization/` | INT8/INT4 for frozen teacher weights | ✅ |
| `progress/` | Training progress bars, metrics | ✅ |
| `lora/` | LoRA layers (may be useful for parameter-efficient stages) | ✅ |
| `eval/` | Perplexity evaluation between runs | ✅ |

### Not Yet Implemented

| Need | Description | Where |
|------|-------------|-------|
| **Transformer forward pass** | Qwen2.5 architecture (RMSNorm, SwiGLU, GQA, RoPE) | `transformer/` module (planned) |
| **RWKV-7 forward pass** | WKV-7 kernel, token shift, decay/key/output gates | `transformer/` module (planned) |
| **RWKV-X sparse attention** | Top-k chunk sparse attention with bounded KV cache | `transformer/` module |
| **KL divergence loss** | Already have `kl_div` in tensor — verify it supports distillation use | `tensor/` |
| **Pipeline parallelism** | Split model layers across nodes, pass activations | `distributed/` or `swarm/` |
| **Mixed-precision training** | FP32 gradients on Q8 frozen weights | `tensor/` + `quantization/` |

---

## Proposed Module: `radlads/`

A new top-level Foundry module implementing the three-step RADLADS pipeline.

### Structure

```
foundry/
└── radlads/
    ├── radlads.h            # Public API
    ├── radlads.c            # Pipeline orchestration
    ├── alignment.c          # Step 1: Hidden state alignment
    ├── distillation.c       # Step 2: KL divergence distillation
    ├── extension.c          # Step 3: Context length extension
    ├── weight_mapping.c     # Transformer -> RWKV weight transfer
    ├── config.h             # Conversion configuration
    └── tests/
        └── test_radlads.c
```

### Public API (Draft)

```c
#include "radlads/radlads.h"

// Configuration
typedef struct {
    const char* teacher_path;       // Path to teacher safetensors
    const char* student_output;     // Path for student output
    const char* tokenizer_path;     // Path to tokenizer.json
    const char* data_path;          // Path to training data

    // Architecture
    fr_arch_t teacher_arch;         // FR_ARCH_QWEN2, FR_ARCH_LLAMA, etc.
    fr_arch_t student_arch;         // FR_ARCH_RWKV7, FR_ARCH_RWKVX

    // Training
    int alignment_tokens;           // Step 1 token budget (default: 100M)
    int distillation_tokens;        // Step 2 token budget (default: 500M)
    int extension_tokens;           // Step 3 token budget (default: 100M)
    int extension_context_len;      // Target context length (default: 16384)
    int batch_size;                 // Per-node batch size
    float learning_rate;            // Default: 3e-4

    // Memory
    bool quantize_teacher;          // Use Q8 for teacher (saves ~50% RAM)
    bool gradient_checkpointing;    // Trade compute for memory

    // Distributed
    int world_size;                 // Number of nodes (0 = auto-detect)
    fr_parallel_t parallel_mode;    // FR_DATA_PARALLEL, FR_PIPELINE_PARALLEL
} fr_radlads_config;

// Run full conversion pipeline
int fr_radlads_convert(fr_radlads_config* config);

// Run individual steps
int fr_radlads_align(fr_radlads_config* config);       // Step 1 only
int fr_radlads_distill(fr_radlads_config* config);     // Step 2 only
int fr_radlads_extend(fr_radlads_config* config);      // Step 3 only

// Weight mapping (no training, just transfer weights)
int fr_radlads_init_student(
    const char* teacher_path,
    const char* student_path,
    fr_arch_t teacher_arch,
    fr_arch_t student_arch
);
```

### Usage Example

```c
#include "radlads/radlads.h"

int main(void) {
    fr_radlads_config config = {
        .teacher_path = "models/Qwen2.5-7B-Instruct",
        .student_output = "models/QRWKV7-7B",
        .tokenizer_path = "models/Qwen2.5-7B-Instruct/tokenizer.json",
        .data_path = "data/minipile.jsonl",
        .teacher_arch = FR_ARCH_QWEN2,
        .student_arch = FR_ARCH_RWKV7,
        .alignment_tokens = 100000000,
        .distillation_tokens = 500000000,
        .extension_tokens = 100000000,
        .extension_context_len = 16384,
        .batch_size = 1,
        .learning_rate = 3e-4,
        .quantize_teacher = true,
        .gradient_checkpointing = true,
        .world_size = 0,  // auto-detect
        .parallel_mode = FR_PIPELINE_PARALLEL
    };

    return fr_radlads_convert(&config);
}
```

---

## Dependencies on Transformer Module

RADLADS requires model forward passes for both teacher and student architectures. This means the `transformer/` module (currently missing) must be built first. The needed architectures:

### Teacher Architectures (inference only, no gradients)

| Architecture | Components | Priority |
|-------------|------------|----------|
| Qwen2.5 | RMSNorm, SwiGLU FFN, GQA, RoPE | High (our target teacher) |
| Llama 3 | RMSNorm, SwiGLU FFN, GQA, RoPE | Medium (nearly identical to Qwen) |

### Student Architectures (training, needs gradients)

| Architecture | Components | Priority |
|-------------|------------|----------|
| RWKV-7 | WKV-7 kernel, token shift, decay/key/output gates, GroupNorm, ReLU-squared FFN | High |
| RWKV-X | RWKV-7 blocks + Top-k chunk sparse attention + bounded KV cache | High |

### Build Order

```
transformer/ (core building blocks)
    ├── rmsnorm, layernorm, embedding
    ├── swiglu_ffn, relu_squared_ffn
    ├── gqa_attention (+ RoPE)
    ├── wkv7_kernel
    ├── rwkv7_block (token shift, decay, gates)
    └── sparse_attention_block (top-k chunk, bounded cache)
         │
         ▼
radlads/ (conversion pipeline)
    ├── weight_mapping (transformer -> RWKV weight init)
    ├── alignment (step 1)
    ├── distillation (step 2)
    └── extension (step 3)
```

---

## Memory Budget for CPU Cluster

Targeting 8GB-per-node machines:

### Option A: Pipeline Parallel (3 nodes)

```
Node 0: Teacher layers 0-9 (Q8) + Student layers 0-9     ~7 GB
Node 1: Teacher layers 10-19 (Q8) + Student layers 10-19  ~7 GB
Node 2: Teacher layers 20-27 (Q8) + Student layers 20-27  ~5 GB
         + Embeddings + LM head + Optimizer state
```

Activations passed between nodes via TCP (swarm module).

### Option B: Layer-by-Layer Alignment (1 node at a time)

For Step 1 (alignment), we only need one teacher layer and one student layer at a time:
```
Load teacher layer N (Q8):    ~250 MB
Load student layer N (FP32):  ~500 MB
Activations + gradients:      ~200 MB
Optimizer state:              ~1 GB
Total per layer:              ~2 GB  (fits easily in 8 GB)
```

Process layers sequentially. This is the most memory-efficient approach for Step 1.

### Option C: Hybrid

- Step 1 (alignment): Layer-by-layer on single node (Option B)
- Step 2 (distillation): Pipeline parallel across 3+ nodes (Option A)
- Step 3 (extension): Same as Step 2

---

## Estimated Timeline

| Phase | Work | Depends On |
|-------|------|------------|
| 1. `transformer/` core blocks | RMSNorm, SwiGLU, GQA, embedding | tensor/ (done) |
| 2. `transformer/` Qwen2.5 | Full Qwen forward pass (inference) | Phase 1 |
| 3. `transformer/` RWKV-7 | WKV-7 kernel, full block (with autograd) | Phase 1 |
| 4. `transformer/` RWKV-X | Block expansion, sparse attention | Phase 3 |
| 5. `radlads/` weight mapping | Init student from teacher weights | Phase 2, 3 |
| 6. `radlads/` alignment | Step 1 hidden state matching | Phase 5 |
| 7. `radlads/` distillation | Step 2 KL divergence | Phase 6 |
| 8. `radlads/` extension | Step 3 long context | Phase 7 |
| 9. Pipeline parallelism | Multi-node model splitting | distributed/ (done) |
| 10. Integration testing | End-to-end Qwen -> RWKV-X | All above |

---

## References

- [RADLADS Paper](https://arxiv.org/abs/2505.03005) (May 2025)
- [RADLADS Code](https://github.com/recursal/RADLADS-paper) (Apache 2.0)
- [RADLADS Models](https://huggingface.co/collections/recursal/radlads-6818ee69e99e729ba8a87102)
- [RWKV-X Paper](https://arxiv.org/abs/2504.21463) (April 2025)
- [HALO: Hybrid Linear Attention Done Right](https://arxiv.org/abs/2601.22156) (January 2026)
- [MiniCPM-SALA](https://arxiv.org/abs/2602.11761) (February 2026)

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-19 | Initial proposal |
