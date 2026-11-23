# attention/

CPU attention for the tensor library: scaled dot-product attention, multi-head
attention with projections, causal and explicit masks, a KV cache for
autoregressive decode, RoPE, and TurboQuant KV-cache compression.

## What it is (and is not)

The Python counterpart is the attention layer in PyTorch. This module computes
attention the standard way — it materialises the score matrix. A fused, tiled
FlashAttention-style kernel is **not implemented** (planned). GPU attention
kernels live in the backends (`cuda_kernels/attention.cu`, `metal/`), not here.

## Dependencies

- tensor/ module (tensor operations)

## API

```c
// Scaled dot-product attention: softmax(QK^T / sqrt(d)) V
tm_tensor* fa_attention(tm_tensor* Q, tm_tensor* K, tm_tensor* V, fa_config* config);
tm_tensor* fa_attention_mask(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                             tm_tensor* mask, fa_config* config);

// Multi-head attention with projections
tm_tensor* fa_multi_head(tm_tensor* x, tm_tensor* Wq, tm_tensor* Wk, tm_tensor* Wv,
                         tm_tensor* Wo, fa_config* config);
tm_tensor* fa_self_attention(tm_tensor* x, tm_tensor* Wqkv, tm_tensor* Wo,
                             fa_config* config);

// KV cache for autoregressive generation
fa_context* fa_context_new(int batch_size, int max_seq_len, int num_heads,
                           int head_dim, int num_layers);
tm_tensor* fa_attention_cached(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                               fa_context* ctx, int layer_idx, int pos,
                               fa_config* config);

// RoPE, masks, config
void fa_rope(tm_tensor* Q, tm_tensor* K, int start_pos, int head_dim);
tm_tensor* fa_causal_mask(int seq_len);
fa_config fa_config_default(int num_heads, int head_dim);
```

Full declarations in `attention.h`; tests: `make test-attention`.

## TurboQuant KV Cache Compression

`turboquant.{h,c}` (prefix `fa_tq_`) compresses the KV cache from FP32 to 2-4
bits per channel: randomized Hadamard rotation, then Lloyd-Max scalar
quantization against precomputed `Beta(d/2, d/2)` codebooks, then bit-packing.
It is tensor-library-free (raw `float*` / `uint8_t*` API) so it can be driven
from these attention loops or from external engines. Default config K4/V2 is
9.14x smaller than FP32.

Quality caveat: K4/V2 is lossy. 4-bit keys are near-lossless, but 2-bit values
drive the error (attention-output cosine ~0.94 synthetic / ~0.975 realistic;
K4/V4 reaches ~0.99). Validate the value bit width per model on real activations
before relying on K4/V2. Two deliberate deviations from the spec, both
correctness-driven: SRHT signs are applied before the Hadamard transform (spreads
the DC component), and normalization uses the L2 norm (the true `Beta(d/2, d/2)`
mapping) rather than per-head min/max.

Full design in [spec.md](spec.md). Tests: `make test-turboquant` (11 unit tests);
quality report: `make validate-turboquant`.

## References

- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [TurboQuant](https://arxiv.org/abs/2504.19874) - Online vector quantization (KV compression)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [Online Softmax Trick](https://arxiv.org/abs/1805.02867)
- [xFormers](https://github.com/facebookresearch/xformers)
