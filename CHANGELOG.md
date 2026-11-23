# Changelog

All notable changes to Foundry will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **tt/** - Tenstorrent Blackhole backend (p150a, 140 Tensix cores). Host side is C99 against **tt-kmd** ioctls; device kernels are hand-written reader/compute/writer triads compiled ahead of time by **SFPI** and embedded as ELF (`ft_tt_kernels.h`), so a Foundry build needs no toolchain present. **No tt-metal runtime, no TT-NN, no C++ in host code.** (Since rewritten from public documentation and measured hardware behaviour; no tt-metal, TT-NN, tt-umd or tt-llk source is compiled, linked or vendored.) Eight op families run on silicon: matmul (bf16/Bfp8_b/Bfp4_b), elementwise, reduce (softmax/RMS norm/layer norm/router normalise), embedding, KV-cache append, RoPE, flash-decode attention, on-card top-k + sampling. Ops execute as **resident kernels** pulling 32-byte descriptors from an L1 ring (`ft_tt_queue.h`) rather than per-op dispatch: 1.363 µs/op host-side at batch 32, 0.391 µs device-side op-to-op, against upstream's 2.08 µs `bh_p150` CI gate. One-core matmul 468.8-625.9 GFLOP/s (28.6-38.2% of one Tensix's *derived* 1.64 TFLOP/s HiFi4 peak at a measured 800 MHz); attention 0.842 ms/descriptor at 8k KV = 24.7 tok/s; top-k 2.590 ms/token over a 130,560 vocabulary. Weights convert from GGUF to a tiled `FTTW` artifact (`tt/convert/`), streaming at 210 MiB peak RSS. `tt/tools/ft_tt_core` reports, probes and recovers individual cores (140/140 healthy). The figures above are one core of 140; multi-core matmul has since been added. `make USE_TT=1 tt` builds the host backend; the suites build from their own Makefiles in `tt/tests/` and `tt/convert/tests/`, and on-card suites need the card. See `tt/README.md` and `docs/tenstorrent.md`.
- **attention/turboquant** - TurboQuant KV cache compression (`fa_tq_`): randomized Hadamard rotation, Lloyd-Max scalar quantization against precomputed Beta(d/2, d/2) codebooks, 2/3/4-bit packing. Tensor-library-free raw-buffer API. Default K4/V2 is 9.14x smaller than FP32. 11 unit tests (`make test-turboquant`) plus an attention-output quality report (`make validate-turboquant`). Note: K4/V2 is lossy (attention-output cosine ~0.94 synthetic / ~0.975 realistic); validate the value bit width per model.
- **metal** - Q4_K GEMM kernels: `fm_sgemm_q4_k` (tiled simdgroup_matrix Q4_K x B GEMM for batched decode, F32 accumulation, ~8x over a naive version) and `fm_gemm_q4_k_f16` (large-M Q4_K prefill GEMM via dequant-to-F16 + MPS F16 matmul). Test harnesses: `make test-gemm-q4-k`, `make test-gemm-q4-k-f16`.

### Changed
- **cuda_kernels/matvec_kquants.cu** - MIN-LIFTED Q4_K matvecs (`fc_matvec_q4k`, `_nx`, `_nx2`, `_nx8`). `d` is applied once per super-block instead of once per element, and the per-sub-block minimum is factored out of the element loop against a per-chunk table of activation sub-block sums (row-independent, so the block builds it once). Inner loop 10 -> 6 instructions per element+row. Measured on one K80 die at 875 MHz, 3 reps: 248320x2048 4.676 -> 3.640 ms (1.285x), 8192x2048 0.171 -> 0.134 (1.276x), 2048x2048 0.055 -> 0.046 (1.196x), batched 2048x512 n=12 0.096 -> 0.088 (1.09x). **This reassociates the sum, so Q4_K matvec results are no longer bit-identical to the previous term order** (max 896 ULP apart at 2048 cols; fp32 relative error against an fp64 reference is unchanged or better). `cuda/tests/bench_matvec.c` gates the new order bit-exactly against a matching CPU replica plus an fp64 error bound (`verify_mv_reassoc`), and `cuda/tests/test_matvec_nx.c` still gates per-column bit-exactness between the tiled and single-column kernels. `cuda_kernels/fused_layer.cu` and `cuda_kernels/moe_expert_group.cu` keep their own Q4_K decoders on the OLD association, so those paths now differ from `fc_matvec_q4k` by the same margin.
- **cuda/fc_cuda.c** - Q4_K's batched dispatch no longer uses the 8-wide tile. The min-lift's extra accumulators put `fc_matvec_q4k_nx8` at 125 registers (from 80), dropping it from 6 to 4 resident blocks per SM; the 4-wide tile is bit-exact per column and measured faster (2048x512, n=12: 8-wide 0.108 ms vs 4-wide 0.088 ms).

## [0.1.0] - 2026-03-02

### Added
- **tensor/** - Tensor operations with autograd, matmul, softmax, layer_norm (141 tests, 83/83 PyTorch parity)
- **optim/** - SGD, Adam, AdamW optimizers with LR schedulers (15 tests)
- **progress/** - Progress bars with rate calculation, ETA, postfix metrics (18 tests)
- **tokenizer/** - BPE tokenization, vocab management, encode/decode (18 tests)
- **safetensors/** - Model weight serialization, memory-mapped reading (18 tests)
- **lora/** - LoRA adapter layers, forward pass, weight merging (15 tests)
- **data/** - JSONL loading, batching, shuffling, train/test split (19 tests)
- **quantization/** - INT8/INT4 quantization, per-channel, per-group (18 tests)
- **attention/** - Multi-head attention, KV cache, RoPE (12 tests)
- **distributed/** - Gradient accumulation, data sharding (18 tests)
- **eval/** - Perplexity, accuracy, F1 evaluation metrics (18 tests)
- Training example (`examples/train_mlp.c`) - XOR problem with MLP
- PyTorch numerical parity test suite (83 comparison tests)
