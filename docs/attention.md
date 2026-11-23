# attention

CPU attention: scaled dot-product, multi-head, KV cache, RoPE. No fused flash-attention kernel yet.

**Header:** `attention/attention.h`
**Source:** `attention/attention.c`
**Status:** API Complete - Pending Verification

---

## Overview

The attention module provides:
- Scaled dot-product attention
- Multi-head attention with projections
- KV cache for efficient inference
- Rotary Position Embedding (RoPE)
- Causal masking

---

## Quick Start

```c
#include "attention/attention.h"

// Configure attention
fa_config cfg = fa_config_default(8, 64);  // 8 heads, 64 dim per head
cfg.causal = true;

// Attention: Q, K, V are [batch, seq, heads, head_dim]
tm_tensor* output = fa_attention(Q, K, V, &cfg);

// With KV cache for generation
fa_context* ctx = fa_context_new(batch, max_seq, 8, 64, num_layers);
for (int pos = 0; pos < seq_len; pos++) {
    tm_tensor* out = fa_attention_cached(Q, K, V, ctx, layer, pos, &cfg);
    // ... use output ...
}
fa_context_free(ctx);
```

---

## API Reference

### Configuration

```c
typedef struct {
    int num_heads;
    int head_dim;
    float dropout;
    bool causal;       // Causal masking for autoregressive
    float scale;       // Attention scale (default: 1/sqrt(head_dim))
} fa_config;

fa_config fa_config_default(int num_heads, int head_dim);
```

### Standard Attention

```c
// Scaled dot-product attention
// Q, K, V: (batch, seq_len, num_heads, head_dim)
// Returns: (batch, seq_len, num_heads, head_dim)
tm_tensor* fa_attention(tm_tensor* Q, tm_tensor* K, tm_tensor* V, fa_config* config);

// With explicit mask
// mask: (batch, 1, seq_len, seq_len) - use -inf for masked positions
tm_tensor* fa_attention_mask(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                             tm_tensor* mask, fa_config* config);
```

### Multi-Head Attention

```c
// Multi-head attention with projections
// x: (batch, seq_len, embed_dim)
// Wq, Wk, Wv: (embed_dim, num_heads * head_dim)
// Wo: (num_heads * head_dim, embed_dim)
tm_tensor* fa_multi_head(tm_tensor* x, tm_tensor* Wq, tm_tensor* Wk, tm_tensor* Wv,
                         tm_tensor* Wo, fa_config* config);

// Self-attention (Q=K=V from same source)
// Wqkv: (embed_dim, 3 * num_heads * head_dim)
tm_tensor* fa_self_attention(tm_tensor* x, tm_tensor* Wqkv, tm_tensor* Wo,
                             fa_config* config);
```

### KV Cache

```c
fa_context* fa_context_new(int batch_size, int max_seq_len, int num_heads,
                           int head_dim, int num_layers);
void fa_context_free(fa_context* ctx);

tm_tensor* fa_attention_cached(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                               fa_context* ctx, int layer_idx, int pos,
                               fa_config* config);

void fa_context_clear(fa_context* ctx);
int fa_context_seq_len(fa_context* ctx);
```

### Rotary Position Embedding (RoPE)

```c
// Apply RoPE to Q and K in-place
void fa_rope(tm_tensor* Q, tm_tensor* K, int start_pos, int head_dim);

// Precompute frequencies
tm_tensor* fa_rope_freqs(int max_seq_len, int head_dim);
```

### Utility

```c
tm_tensor* fa_causal_mask(int seq_len);
void fa_print_config(fa_config* config);
```

---

## TurboQuant KV cache compression

`attention/turboquant.{h,c}` (prefix `fa_tq_`) compresses the KV cache from FP32
to 2-4 bits per channel, based on TurboQuant (Zandieh et al. 2025,
arXiv:2504.19874). Full design in [attention/spec.md](../attention/spec.md).

Per K or V vector the pipeline is: randomized Hadamard rotation
(`D * H * x / sqrt(d)`, `O(d log d)`) to spread information across coordinates,
a per-head range map, Lloyd-Max scalar quantization against precomputed
`Beta(d/2, d/2)` codebooks (shipped as a static table in `tq_codebooks.h`, with a
Lloyd's-algorithm fallback), then bit-packing to 2/3/4 bits. Dequant reverses
each step. The module is tensor-library-free: it operates on raw `float*` /
`uint8_t*` buffers so the foundry attention loops and external engines can call
it directly. Bit allocation is asymmetric because keys route attention while
values are averaged. The default is K4/V2 (4-bit keys, 2-bit values), which is
9.14x smaller than FP32 including range metadata.

```c
#include "attention/turboquant.h"

fa_tq_config cfg = fa_tq_config_default();          // K4 / V2
fa_tq_state* tq  = fa_tq_init(head_dim, cfg, seed); // seed drives the rotation

fa_tq_cache* cache = fa_tq_cache_new(tq, num_layers, num_heads, max_seq_len);

// Per token: quantize K and V into the cache, dequantize a range back out.
fa_tq_quantize(tq, cache, layer, pos, k_vec, /*is_key=*/true);
fa_tq_quantize(tq, cache, layer, pos, v_vec, /*is_key=*/false);
fa_tq_dequantize(tq, cache, layer, /*start=*/0, count, /*is_key=*/true, out);

fa_tq_cache_free(cache);
fa_tq_free(tq);
```

Low-level single-head `fa_tq_quantize_head` / `fa_tq_dequantize_head`,
`fa_tq_rotate` / `fa_tq_rotate_inverse`, and the codebook/introspection helpers
are also public (see `turboquant.h`).

### Quality: K4/V2 is lossy, validate the value bit width per model

4-bit keys are near-lossless, but 2-bit values are the error driver. Measured
attention-output cosine similarity (FP vs compressed):

| Config | Synthetic (Gaussian) | Realistic (correlated) |
|--------|----------------------|------------------------|
| K4/V2  | ~0.94                | ~0.975                 |
| K4/V3  | ~0.98                | -                      |
| K4/V4  | ~0.99                | ~0.99                  |

The default K4/V2 is a real quality hit, not lossless. Pick the value bit width
per model on real activations before relying on K4/V2; raise it to V3 or V4 where
output fidelity matters. `make test-turboquant` runs the 11 unit tests (all
pass); `make validate-turboquant` prints the attention-output quality report
(informational, not a build gate).

---

## See Also

- [tensor](tensor.md) - Tensor operations
- [lora](lora.md) - LoRA for attention layers
- [attention/spec.md](../attention/spec.md) - TurboQuant design spec
