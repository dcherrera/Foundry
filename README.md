# Foundry

Pure C implementations of 11 core Python ML tools. CPU is the baseline; GPU backends exist for CUDA, Metal and Tenstorrent Blackhole (see [GPU Support](#gpu-support)). Vulkan is planned.

## Why Foundry?

The Python ML ecosystem is powerful but has limitations:

- **Dependency hell** - Complex Python environments, version conflicts

- **Not embeddable** - Can't easily integrate training into native apps

- **Performance overhead** - Python GIL, interpreter overhead

- **Deployment complexity** - Requires Python runtime everywhere

Foundry provides C implementations that can be:

- Embedded in any application

- Compiled to a single binary

- Used from any language via C FFI

- Deployed without Python

## The Tools

We identified 11 core Python tools that everything else builds on. Each has alternatives, but these are the essential functions:

| Foundry Module | Python counterpart | Function                               |
| -------------- | --------------- | ----------------------------------------- |
| tensor/        | PyTorch         | Tensor operations + autograd              |
| optim/         | PyTorch         | Optimizers (SGD, Adam) + LR schedulers    |
| distributed/   | DeepSpeed       | Gradient accumulation, data sharding (single-process; real distributed planned) |
| lora/          | PEFT            | LoRA and adapter training                 |
| tokenizer/     | tokenizers      | BPE tokenization (WordPiece planned)      |
| data/          | datasets        | Dataset loading and processing            |
| safetensors/   | safetensors     | Safe model weight serialization           |
| attention/     | PyTorch attention | Standard (unfused) multi-head attention, KV cache, RoPE, TurboQuant KV compression; no flash kernel yet |
| quantization/  | bitsandbytes    | Model quantization (int8, int4)           |
| eval/          | lm-eval-harness | Model evaluation and benchmarks           |
| progress/      | tqdm            | Progress reporting                        |

## Why These Tools?

Many Python ML tools are alternatives or convenience wrappers:

**Frameworks** - PyTorch vs TensorFlow vs JAX (pick one) **Distributed** - DeepSpeed vs Megatron vs FairScale (pick one) **LoRA** - PEFT vs Unsloth vs Axolotl (PEFT is the core, others wrap it) **Tokenizers** - tokenizers vs SentencePiece vs tiktoken (same function)

We chose the most fundamental tool in each category. Everything else can be built on top.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Your Application                           │
├─────────────────────────────────────────────────────────────────┤
│                       Foundry C API                             │
├─────────────────────────────────────────────────────────────────┤
│  tensor/  │  lora/  │  tokenizer/  │  data/  │  eval/  │  ...  │
├─────────────────────────────────────────────────────────────────┤
│        Backends: CPU · CUDA · Metal · Tenstorrent Blackhole     │
│                       (Vulkan planned)                          │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

**Prerequisites:** a C11 compiler and `make`. BLAS is optional (`USE_BLAS=1`: Accelerate on
macOS, OpenBLAS on Linux). The default flags include `-march=native`, so build on the machine
that will run the code, or add `PORTABLE=1` for a copyable build (`-march=x86-64-v2` on x86-64).

```
# Build all modules
make all

# Run all CPU tests (504 tests, 30 suites)
make test

# Run training example
make train
```

## Training Example

```
$ make train

===========================================
  Foundry Training Example: XOR with Adam
===========================================

Model: 2-layer MLP (2 -> 16 -> 1)
Optimizer: Adam (lr=0.010, beta1=0.9, beta2=0.999)
Epochs: 500

Epoch    1 | Loss: 0.216718 | LR: 0.0100
...
Epoch  500 | Loss: 0.001138 | LR: 0.0100

Predictions vs Targets:
-----------------------
  0 XOR 0 = 0.060 (target: 0) [OK]
  0 XOR 1 = 0.980 (target: 1) [OK]
  1 XOR 0 = 0.983 (target: 1) [OK]
  1 XOR 1 = 0.017 (target: 0) [OK]

Accuracy: 4/4 (100%)

[SUCCESS] Model learned XOR perfectly!
```

## Status

| Module        | Status   | Tests | Description                                      |
| ------------- | -------- | ----- | ------------------------------------------------ |
| tensor/       | Complete | 141   | Tensor ops with autograd (PyTorch parity: 80/83) |
| optim/        | Complete | 15    | SGD, Adam, AdamW, LR schedulers                  |
| progress/     | Complete | 18    | Training progress bars & logging                 |
| tokenizer/    | Complete | 18    | BPE tokenization                                 |
| safetensors/  | Complete | 18    | Safe tensor serialization                        |
| lora/         | Complete | 15    | Low-rank adaptation                              |
| data/         | Complete | 19    | Dataset loading & batching                       |
| quantization/ | Complete | 57    | INT8/INT4 quantization, export (test_quantization 43 + test_export 14) |
| attention/    | Complete | 23    | Multi-head attention, KV cache, RoPE, TurboQuant KV compression (fa_tq_) (test_attention 12 + test_turboquant 11) |
| distributed/  | Complete | 18    | Gradient accumulation, data sharding             |
| eval/         | Complete | 18    | Perplexity, accuracy, F1 metrics                 |

**Core modules: 352 tests.** `make test` also runs the extension suites — hdc 18, hybridmem 20,
threadpool 7, rwkvx 27 (3 suites), barswap 14, pth 10, ingot 56 (9 suites) plus 43 CLI checks,
and an exhaustive Q2_0 rounding check — for **504 tests across 30 suites, all passing** (macOS
arm64, `USE_BLAS=1 USE_THREADS=1`, 2026-11-23). Not in `make test`: `make test-swarm` (14),
the `test-potluck-*` suites and `test-byte-data`. The rwkvx suites need the RWKV vocabulary,
which `make test` downloads once via `rwkvx/data/fetch_rwkv_vocab.sh` (network, `curl`,
`sha256sum` or `shasum`). Hardware suites are separate: `make USE_CUDA=1 test-cuda`,
`make USE_METAL=1 test-metal` (17, passing on Apple Silicon 2026-11-23), and the Tenstorrent suites in `tt/tests/` (see
[`tt/README.md`](./tt/README.md)); each needs the hardware.

**PyTorch numerical parity: 80/83 comparison tests (tensor module)**

## Other Modules

Modules in the tree that are not in the 11-tool table:

| Module | What it is |
| ------ | ---------- |
| rwkvx/ | RWKV-X hybrid LM (RWKV-7 + sparse attention): inference + LoRA training |
| rwkvx/rfa.* | RWKV-7 + full-attention (Qwen2-style GQA) hybrid inference and training |
| convert/ | HuggingFace safetensors → `.btw`/`.btv` converter for Llama-family models |
| pth/ | PyTorch `.pth` (ZIP + pickle) state_dict reader, zero-copy via mmap |
| threadpool/ | Minimal pthread `parallel_for` (`USE_THREADS=1`) |
| hdc/ | Hyperdimensional computing: binary hypervectors, Hamming-distance search |
| hybridmem/ | Disk-backed K/V memory tier for RWKV-X, indexed with hdc/ |
| barswap/ | K80 die time-multiplexing through one PCIe BAR slot on a specific board (BTC-S37) |
| ingot/ | Lossless model file compression (BF16 safetensors, GGUF Q8_0/K-quants), streaming decode; own Makefile — see [`ingot/README.md`](./ingot/README.md) |
| potluck/ | RWKV-7 LoRA training pieces: `.pth` loader, Q8_0 frozen base, JSONL data, token-budget batching (`make test-potluck-*`) |
| swarm/ | Distributed CPU training across Docker Swarm (`make test-swarm`, not in `make test`); see [`swarm/README.md`](./swarm/README.md) |
| dedup/ | MinHash/LSH deduplication — interface only (header + README), not implemented |
| retrieval/ | Hash-based document retrieval — interface only (header + README), not implemented |

## Build Options

```
# Build specific module
make tensor
make lora

# Run specific test suite
make test-tensor
make test-attention

# Debug build
make DEBUG=1 all

# Clean
make clean
```

## GPU Support

CPU is the baseline — every operation works on CPU first, and the GPU backends
extend them without changing the public API. Each backend is opt-in at build time.

- **CUDA** (`USE_CUDA=1`, Linux; `cuda/`, `cuda_kernels/`) — loaded via `dlopen`, kernels
  compiled to PTX and embedded, so Foundry builds without nvcc present. Targets `sm_37`
  (Kepler, e.g. K80) upward; that is deliberate, since most current runtimes have dropped
  below-Pascal support. `barswap/` adds K80 die time-multiplexing on one specific board.
- **Metal** (`USE_METAL=1`, macOS; `metal/`) — Apple Silicon.
- **Tenstorrent Blackhole** (`USE_TT=1`, Linux; `tt/`) — a p150a's Tensix cores, through
  Foundry's own firmware, kernels and userspace driver, plain C. Rewritten from public
  documentation and measured hardware behaviour: no tt-metal, TT-NN, tt-umd or tt-llk library is
  linked or vendored, and no Tenstorrent source is compiled by the current build (one measurement
  test carries tt-umd-derived code, Apache-2.0; see `NOTICE`). Kernels are hand-written
  reader/compute/writer triads, compiled ahead of time and embedded, so a Foundry build needs no
  toolchain; the SFPI compiler (and `python3` for the `tt/tools/emit_tensix_*.py` generators) is
  needed only to rebuild the device images. 8 op families run on silicon, plus Gated DeltaNet;
  multi-core matmul exists. [Pantheon](https://git.teamide.dev/David/Pantheon.git) runs a 9B
  `qwen35` model end-to-end on the card through this backend. See
  [`tt/README.md`](./tt/README.md) and [`docs/tenstorrent.md`](./docs/tenstorrent.md).
- **Vulkan** — planned.

Contributions to the GPU backends are very welcome.

## License

Foundry is licensed under the **GNU AGPL-3.0** — see [`LICENSE`](./LICENSE).

Contributions are accepted under the CLA in [`CLA.md`](./CLA.md), with a DCO
sign-off on every commit — see [`CONTRIBUTING.md`](./CONTRIBUTING.md). You keep
your copyright; signing is one entry in
[`.cla/signatures.md`](./.cla/signatures.md), added in your first pull request.

Copyright © 2026 David (Corey) Herrera and the Foundry contributors. See
[`NOTICE`](./NOTICE).