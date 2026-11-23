# Foundry Documentation

Pure C implementation of essential ML tools for training large language models on CPU clusters.

---

## Quick Links

- [Getting Started](getting-started.md)
- [API Reference](#api-reference)
- [Examples](examples.md)
- [Architecture](architecture.md)

---

## API Reference

### Core Modules

| Module | Description | Docs |
|--------|-------------|------|
| **tensor** | Tensor operations with autograd | [tensor.md](tensor.md) |
| **optim** | Optimizers (SGD, Adam, AdamW) | [optim.md](optim.md) |
| **tokenizer** | BPE tokenization | [tokenizer.md](tokenizer.md) |
| **safetensors** | Model serialization | [safetensors.md](safetensors.md) |
| **data** | Dataset loading and batching | [data.md](data.md) |

### Training Modules

| Module | Description | Docs |
|--------|-------------|------|
| **attention** | Efficient attention with KV cache | [attention.md](attention.md) |
| **lora** | Low-Rank Adaptation fine-tuning | [lora.md](lora.md) |
| **quantization** | INT8/INT4/ternary quantization, training + export | [quantization.md](quantization.md) |
| **distributed** | Distributed training primitives | [distributed.md](distributed.md) |
| **swarm** | Docker Swarm distributed training | [swarm.md](swarm.md) |

### GPU Acceleration

| Module | Description | Docs |
|--------|-------------|------|
| **cuda** | CUDA backend (`dlopen`, embedded PTX, sm_37+) | [getting-started.md](getting-started.md) |
| **metal** | Apple Metal GPU compute backend | [metal.md](metal.md) |
| **tt** | Tenstorrent Blackhole compute backend (p150a; own userspace driver, tt-kmd optional) | [tenstorrent.md](tenstorrent.md) |
| **barswap** | K80 BAR reassignment (kernel module + userspace) | [bar-swap.md](bar-swap.md) |

### Utility Modules

| Module | Description | Docs |
|--------|-------------|------|
| **progress** | Progress bars and metrics | [progress.md](progress.md) |
| **eval** | Model evaluation framework | [eval.md](eval.md) |
| **threadpool** | pthread `parallel_for` (`USE_THREADS=1`) | — |
| **pth** | PyTorch `.pth` state_dict reader (mmap) | — |
| **convert** | HuggingFace safetensors → `.btw`/`.btv` converter | — |

### Models, Memory and Compression

| Module | Description | Docs |
|--------|-------------|------|
| **rwkvx** | RWKV-X and RFA hybrid LMs: inference, training, backward passes | CLI pages below |
| **potluck** | RWKV-7 LoRA training pieces (loader, Q8_0 base, data, batching) | — |
| **hdc** | Hyperdimensional computing (binary hypervectors) | [hdc/README.md](../hdc/README.md) |
| **hybridmem** | Disk-backed K/V memory tier, indexed with hdc | [hybridmem/README.md](../hybridmem/README.md) |
| **ingot** | Lossless model file compression | [ingot/README.md](../ingot/README.md) |
| **dedup**, **retrieval** | Interfaces only (header + README), not implemented | [dedup](../dedup/README.md), [retrieval](../retrieval/README.md) |

### CLI Tools

| Tool | Description | Docs |
|------|-------------|------|
| **rwkvx_generate** | RWKV-X text generation (INT8, HybridMem) | [rwkvx-generate.md](rwkvx-generate.md) |
| **rfa_generate** | RFA hybrid text generation (RWKV-7 + Qwen) | [rfa-generate.md](rfa-generate.md) |
| **rfa_train** | RFA hybrid LoRA training (parallel, INT8) | [rfa-train.md](rfa-train.md) |
| **rfa_lora_merge** | LoRA adapter averaging (cross-machine merge) | [rfa-lora-merge.md](rfa-lora-merge.md) |
| **quantize_model** | Offline model quantization (INT8/INT4/ternary) | [quantize-model.md](quantize-model.md) |
| **pth_convert** | `.pth` → safetensors (`make pth-convert`) | — |
| **model_convert** | HF safetensors → `.btw`/`.btv` (`make model-convert`) | — |

---

## Building Foundry

### Prerequisites

- GCC or Clang with C11 support
- Make
- POSIX-compatible system (Linux, macOS)

### Build Commands

```bash
# Build all modules
make all

# Build specific module
make tensor
make swarm

# Run all tests
make test

# Clean build artifacts
make clean
```

### Build Options

| Option | Description |
|--------|-------------|
| `DEBUG=1` | `-O0 -g` debug build (default is `-O3`) |
| `PORTABLE=1` | Copyable binary (no `-march=native`) |
| `USE_BLAS=1` | Accelerate (macOS) / OpenBLAS (Linux) |
| `USE_THREADS=1` / `USE_OMP=1` | pthread pool / OpenMP |
| `USE_CUDA=1`, `USE_METAL=1`, `USE_TT=1` | GPU backends |

---

## Project Structure

```
foundry/
├── tensor/          # Core tensor operations
│   ├── tensor.h
│   ├── tensor.c
│   └── tests/
├── optim/           # Optimizers
├── tokenizer/       # Tokenization
├── safetensors/     # Model I/O
├── data/            # Data loading
├── attention/       # Attention mechanisms
├── lora/            # LoRA adapters
├── quantization/    # Quantization
├── rwkvx/           # RWKV-X / RFA hybrid models
├── potluck/         # RWKV-7 LoRA training pieces
├── hdc/ hybridmem/  # Hypervectors, disk-backed K/V memory
├── ingot/           # Lossless model file compression (own Makefile)
├── pth/ convert/    # .pth reader, safetensors converter
├── threadpool/      # pthread parallel_for
├── cuda/ cuda_kernels/  # CUDA backend (host + kernels)
├── metal/           # Apple Metal GPU backend
├── tt/              # Tenstorrent Blackhole backend (host C + Tensix kernels + converter)
├── barswap/         # K80 BAR reassignment (kernel module + userspace lib)
├── distributed/     # Distributed primitives
├── swarm/           # Docker Swarm training
├── progress/        # Progress bars
├── eval/            # Evaluation
├── examples/        # Example programs and CLI tools
├── scripts/         # License header, CLA and DCO checks
├── docs/            # Documentation (you are here)
├── Makefile
└── roadmap.md
```

---

## Python Counterparts

Each module covers the core function of one Python library, in C:

| Python Tool | Foundry Module | Notes |
|-------------|----------------|-------|
| PyTorch | tensor, optim | tensor ops + autograd; PyTorch parity 80/83 comparison tests |
| HuggingFace Tokenizers | tokenizer | BPE (WordPiece planned) |
| HuggingFace Safetensors | safetensors | read/write, mmap |
| HuggingFace Datasets | data | JSONL, batching, sqlite chat loading |
| tqdm | progress | |
| bitsandbytes | quantization | INT8/INT4, export |
| PyTorch attention | attention | standard attention, KV cache, RoPE; no fused flash kernel yet |
| PEFT | lora | |
| Accelerate/DeepSpeed | distributed | single-process primitives; real multi-node planned |
| lm-eval-harness | eval | perplexity, accuracy, F1 |

---

## Design Principles

1. **Zero Python Dependencies** - Pure C, no Python runtime required
2. **Minimal External Dependencies** - POSIX APIs, plus opt-in vendor SDKs (CUDA, Metal, BLAS); see NOTICE
3. **Memory Efficient** - No unnecessary copies, mmap support
4. **Embeddable** - Designed to be embedded in other applications
5. **Portable** - Works on x86_64 and ARM (Raspberry Pi)

---

## Target Use Case

Training 1B+ parameter RWKV-x models across heterogeneous CPU clusters:
- Raspberry Pi 4/5
- Laptops and desktops
- Small servers
- Connected via WiFi/Starlink

The swarm module provides distributed training with:
- Async gradient updates
- TopK gradient compression (90% bandwidth reduction)
- Hardware-adaptive batch sizes
- Fault tolerance

---

## License

See [`../LICENSE`](../LICENSE) and the
License section of the top-level [`README.md`](../README.md).

---

## Contributing

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
