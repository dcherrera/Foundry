# TurboQuant KV Cache Compression — Spec

Module: `foundry/attention/`
Prefix: `fa_tq_` (Foundry Attention — TurboQuant)

Based on: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., Google Research / DeepMind, April 2025)

---

## Overview

TurboQuant compresses the KV cache from FP32 to 2-4 bits per channel with near-zero quality loss. The algorithm is:

1. **Rotate** the vector by a fixed random orthogonal matrix (spreads information uniformly)
2. **Scalar quantize** each coordinate independently using precomputed Lloyd-Max levels for the Beta distribution
3. **Dequantize** by reversing: look up reconstruction values, then rotate back

No codebook training. No preprocessing. No iterative optimization. Just a rotation matrix and a lookup table. The entire quantizer is deterministic given the dimension and bit width.

### Why not QJL?

The paper proposes QJL (1-bit quantized Johnson-Lindenstrauss) on residuals for unbiased inner products. Six independent community implementations confirmed that for softmax attention, **MSE-only quantization without QJL outperforms the full method**. Softmax exponentially amplifies variance, making low MSE more valuable than unbiasedness. We skip QJL entirely.

### Asymmetric bit allocation

Keys determine attention routing — errors in keys misroute attention to the wrong tokens. Values just get averaged — errors cancel during the weighted sum. This gives us:

| Component | Bits | Levels | Compression vs FP32 |
|-----------|------|--------|---------------------|
| Keys      | 4    | 16     | 8x                  |
| Values    | 2    | 4      | 16x                 |
| Average   | 3    | —      | ~10.7x              |

Conservative default: K4/V2. Optional K3/V2 for aggressive compression (~12.8x).

---

## As-Built Notes (implementation vs this spec)

The shipped `turboquant.{h,c}` follows this design with two deliberate,
correctness-driven deviations from the text below:

1. **SRHT signs before the Hadamard transform.** The rotation applies the random
   `+-1` sign diagonal `D` before the FWHT rather than after (`H * (D * x)`).
   This spreads the DC component across coordinates and gave measurably better
   roundtrip error. The transform is still orthogonal and self-inverse up to the
   `1/sqrt(d)` scaling.
2. **L2-norm normalization, not per-head min/max.** Step 2 below describes mapping
   rotated coordinates into `[0,1]` via per-head min/max. The implementation
   instead normalizes by the vector L2 norm, which is the true `Beta(d/2, d/2)`
   mapping the codebooks are optimized for. The stored per-head range metadata
   carries the norm-based scale accordingly.

### Measured quality: K4/V2 is lossy

The default K4/V2 is **not** near-lossless in practice. 4-bit keys are
near-lossless, but 2-bit values are the error driver. Attention-output cosine
similarity (FP vs compressed):

| Config | Synthetic (Gaussian) | Realistic (correlated) |
|--------|----------------------|------------------------|
| K4/V2  | ~0.94                | ~0.975                 |
| K4/V3  | ~0.98                | -                      |
| K4/V4  | ~0.99                | ~0.99                  |

Treat the value bit width as a per-model knob: validate on real activations and
raise it to V3 or V4 where output fidelity matters. The "attention score
preservation >0.999" target in the Testing Strategy below is aspirational for
K4/V2 and is reported by `validate_tq_attention.c` rather than gated. The 11
unit tests in `test_turboquant.c` (codebook, rotation orthogonality, roundtrip
MSE, dot preservation, determinism, packing, cache) all pass.

---

## Algorithm Detail

### Step 1: Random Orthogonal Rotation

Generate a fixed `d x d` orthogonal matrix `R` where `d = head_dim`. This is generated once at init from a seed and reused for all tokens and layers.

**Implementation**: Use a randomized Hadamard transform (fast `O(d log d)`) instead of dense matrix multiply (`O(d^2)`):

```
rotate(x) = D * H * x / sqrt(d)
```

Where:
- `H` is the Walsh-Hadamard matrix (applied in-place via butterfly ops, no storage)
- `D` is a diagonal of random +1/-1 signs (one `d`-element array, generated from seed)

This is `O(d log d)` per vector vs `O(d^2)` for a full orthogonal matrix. For `head_dim=128`, that's ~896 FLOPs vs ~16,384.

The rotation spreads information across all coordinates, making each coordinate approximately follow a `Beta(d/2, d/2)` distribution centered at 0.5 (after shifting to [0,1]). In high dimensions this concentrates sharply around the mean, making simple scalar quantization nearly optimal.

### Step 2: Lloyd-Max Scalar Quantization

After rotation, each coordinate is quantized independently using optimal decision boundaries and reconstruction levels for `Beta(d/2, d/2)`.

These are **precomputed offline** for each (d, bits) pair and stored as constant lookup tables:

```c
typedef struct {
    int bits;                    // 2, 3, or 4
    int levels;                  // 2^bits
    float boundaries[15];        // Decision boundaries (levels - 1 values)
    float centroids[16];         // Reconstruction centroids (levels values)
} fa_tq_codebook;
```

For quantization: binary search into `boundaries[]` to get the level index.
For dequantization: index into `centroids[]` to get the reconstruction value.

The Beta distribution parameters depend only on `head_dim`, which is constant per model. So we need one codebook per (head_dim, bits) pair — precomputed at compile time or at `fa_tq_init()`.

### Step 3: Packing

Quantized values are packed into bytes:

| Bits | Values per byte | Packing         |
|------|-----------------|-----------------|
| 2    | 4               | `[v0:2][v1:2][v2:2][v3:2]` |
| 3    | 2 (+ 2 waste)   | `[v0:3][v1:3][__:2]`       |
| 4    | 2               | `[v0:4][v1:4]`             |

3-bit packing wastes 25% of bits. Alternative: pack 8 values into 3 bytes (no waste, more complex addressing). Start with the simple per-byte packing; optimize later if memory-bound.

---

## Memory Layout

### Compressed cache entry (one token, one layer)

```
key_packed:    [ceil(head_dim * k_bits / 8)] bytes per head, n_heads total
value_packed:  [ceil(head_dim * v_bits / 8)] bytes per head, n_heads total
key_range:     [2 floats per head] — min/max before rotation (for denormalization)
value_range:   [2 floats per head] — min/max before rotation
```

The min/max per head are needed to map the rotated coordinates into [0, 1] for the Beta quantizer.

### Memory per token per layer

| Config | Keys | Values | Range metadata | Total | vs FP32 (`2 * n_heads * head_dim * 4`) |
|--------|------|--------|----------------|-------|-----------------------------------------|
| K4/V2, 16 heads, dim 128 | 1024 B | 512 B | 256 B | 1792 B | vs 16,384 B → **9.1x** |
| K4/V2, 32 heads, dim 128 | 2048 B | 1024 B | 512 B | 3584 B | vs 32,768 B → **9.1x** |
| K3/V2, 16 heads, dim 128 | 768 B | 512 B | 256 B | 1536 B | vs 16,384 B → **10.7x** |

### Total cache savings

RWKV-X 0.2B (4 attention layers, 16 heads, dim 128, max_cache 2048):

| | FP32 | TurboQuant K4/V2 |
|---|---|---|
| Per layer | 2 * 2048 * 2048 * 4 = **32 MB** | 2048 * 1792 = **3.5 MB** |
| Total (4 layers) | **128 MB** | **14 MB** |

Llama-style 3.6B (24 attention layers, 32 heads, dim 128, max_seq 4096):

| | FP32 | TurboQuant K4/V2 |
|---|---|---|
| Per layer | 2 * 4096 * 4096 * 4 = **128 MB** | 4096 * 3584 = **14.3 MB** |
| Total (24 layers) | **3 GB** | **343 MB** |

---

## API

### Types

```c
/* Bit allocation configuration */
typedef struct {
    int key_bits;       /* 2, 3, or 4 (default: 4) */
    int value_bits;     /* 2, 3, or 4 (default: 2) */
} fa_tq_config;

/* Opaque quantizer state (holds rotation signs + codebooks) */
typedef struct fa_tq_state fa_tq_state;

/* Compressed KV cache (replaces fa_context for quantized mode) */
typedef struct fa_tq_cache fa_tq_cache;
```

### Lifecycle

```c
/* Initialize quantizer for a given head_dim and bit config.
 * seed determines the random rotation (deterministic, reproducible).
 * Call once at model load time. */
fa_tq_state* fa_tq_init(int head_dim, fa_tq_config config, uint64_t seed);
void          fa_tq_free(fa_tq_state *state);

/* Default config: K4/V2 */
fa_tq_config fa_tq_config_default(void);
```

### Cache

```c
/* Create compressed KV cache.
 * Same shape semantics as fa_context but stores quantized data. */
fa_tq_cache* fa_tq_cache_new(fa_tq_state *state, int num_layers,
                              int num_heads, int max_seq_len);
void          fa_tq_cache_free(fa_tq_cache *cache);
void          fa_tq_cache_clear(fa_tq_cache *cache);
int           fa_tq_cache_seq_len(fa_tq_cache *cache);
```

### Quantize / Dequantize

```c
/* Quantize a single K or V vector into the cache at position pos.
 * vec: [num_heads * head_dim] float input
 * is_key: true for key, false for value (determines bit width) */
void fa_tq_quantize(fa_tq_state *state, fa_tq_cache *cache,
                    int layer, int pos, const float *vec, bool is_key);

/* Dequantize a range of cached vectors back to float.
 * out: [count * num_heads * head_dim] float output
 * Returns pointer to out for convenience. */
float* fa_tq_dequantize(fa_tq_state *state, fa_tq_cache *cache,
                        int layer, int start, int count,
                        bool is_key, float *out);
```

### Attention with TurboQuant cache

```c
/* Drop-in replacement for fa_attention_cached.
 * Quantizes K, V into cache, dequantizes cached K, V for attention.
 * Q is always full precision (current token only, never cached). */
tm_tensor* fa_tq_attention_cached(tm_tensor *Q, tm_tensor *K, tm_tensor *V,
                                   fa_tq_state *tq, fa_tq_cache *cache,
                                   int layer_idx, int pos, fa_config *config);
```

---

## Integration Points

### RWKV-X (`foundry/rwkvx/rwkvx.c`)

The sparse attention blocks (MOBA layers) maintain per-layer `rx_attn_state` with flat `k_cache`/`v_cache` buffers. Integration:

1. Add `fa_tq_state*` and `fa_tq_cache*` to `rx_model` or `rx_state`
2. In the attention block forward pass (~line 1163), after computing `k_vec`/`v_vec`:
   - Call `fa_tq_quantize()` to store into compressed cache
   - Call `fa_tq_dequantize()` to reconstruct K/V for the attention dot products
3. Remove the FP32 `k_cache`/`v_cache` from `rx_attn_state` when TQ is enabled
4. Cache eviction (line 1181) works the same — just operates on packed bytes instead of floats, `memmove` still works

### Pantheon / Llama (`Pantheon/src/models/llama.c`)

The per-head `[n_kv_head, max_seq, head_dim]` layout maps directly:

1. Add `fa_tq_state*` to `pn_llama_model`, `fa_tq_cache*` to `pn_llama_kv_cache`
2. In `pn_llama_forward()` (~line 1484), replace the `memcpy` into K/V cache with `fa_tq_quantize()`
3. Before the attention head loop (~line 1370), `fa_tq_dequantize()` the needed range
4. Circular buffer addressing: `cache_pos = pos % max_seq` still works — the compressed cache just stores packed bytes at each position

### Runtime toggle

Both engines already have a `--quantize` flag for INT8 weights. Add `--tq-cache` (or `--kv-compress`) to enable TurboQuant on the KV cache independently. The two are orthogonal: INT8 weights + TQ cache is the highest compression combo.

---

## File Plan

```
foundry/attention/
├── attention.h          # Existing — add fa_tq_* declarations
├── attention.c          # Existing — unchanged
├── turboquant.h         # TurboQuant types + API (public header)
├── turboquant.c         # Implementation
├── tq_codebooks.h       # Precomputed Lloyd-Max tables (generated, static const)
├── spec.md              # This file
└── tests/
    └── test_turboquant.c
```

`turboquant.h/c` are self-contained. They depend only on `<math.h>`, `<string.h>`, `<stdint.h>`. No tensor library dependency — they operate on raw `float*` and `uint8_t*` buffers. This keeps them usable from both the foundry attention module and the hand-rolled attention loops in RWKV-X and Pantheon.

---

## Codebook Generation

Lloyd-Max optimal levels for `Beta(d/2, d/2)` can be computed offline via iterative Lloyd's algorithm:

1. Start with uniform boundaries in [0, 1]
2. Compute centroids as conditional expectations `E[X | boundary_i < X < boundary_{i+1}]` under `Beta(d/2, d/2)`
3. Update boundaries as midpoints between adjacent centroids
4. Repeat until converged (~20 iterations)

Ship a Python script `tools/gen_codebooks.py` that generates `tq_codebooks.h` for all needed (d, bits) pairs. The output is a static C header — no Python dependency at runtime.

Target dimensions: 64, 96, 128, 160 (covers RWKV-X 0.2B through 3.6B and common Llama configs). Bit widths: 2, 3, 4.

That's 4 * 3 = 12 codebooks, each under 200 bytes. Total: ~2 KB of static const data.

---

## Hadamard Transform Detail

The Walsh-Hadamard transform of dimension `d` (must be power of 2) is:

```c
/* In-place Fast Walsh-Hadamard Transform, O(d log d) */
static void fwht(float *x, int d) {
    for (int len = 1; len < d; len <<= 1) {
        for (int i = 0; i < d; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
}
```

Full rotation: multiply each element by the random sign (`D`), apply FWHT, scale by `1/sqrt(d)`. Inverse: same operation (FWHT is its own inverse up to scaling, and `D` is its own inverse).

For `head_dim` not a power of 2: pad to the next power of 2, apply FWHT, truncate. The extra coordinates are discarded — they don't affect the quantized output. Alternatively, use a dense random orthogonal matrix for non-power-of-2 dims (rare in practice).

---

## Performance Budget

Target: TurboQuant overhead < 5% of total inference time.

| Operation | FLOPs per token per layer | Notes |
|-----------|--------------------------|-------|
| Hadamard rotate K | `d * log2(d)` = 896 | d=128 |
| Hadamard rotate V | 896 | |
| Quantize K (binary search) | `d * log2(levels)` = 512 | 4-bit, 4 comparisons |
| Quantize V | `d * log2(levels)` = 256 | 2-bit, 2 comparisons |
| Pack K into bytes | `d` = 128 | Shift + OR |
| Pack V into bytes | `d` = 128 | |
| **Total quantize** | **~2,816** | Per head |
| | | |
| Dequantize K (all cached) | `seq * d` = 262,144 | seq=2048, lookup + unpack |
| Hadamard inverse K | `seq * d * log2(d)` = 1,835,008 | |
| Dequantize V (all cached) | `seq * d` = 262,144 | |
| Hadamard inverse V | `seq * d * log2(d)` = 1,835,008 | |
| **Total dequantize** | **~4.2M** | Dominates |
| | | |
| Attention Q@K^T + softmax + @V | `~2 * seq * d` = 524,288 | Standard attention |

Dequantize is ~8x the attention cost per head, which is too expensive if done naively for the full cache every token.

### Optimization: Fused dequant-dot

Instead of materializing the full dequantized K/V matrix, fuse dequantization into the attention loop:

```
For each cached position t:
    k_packed = cache[t]
    k_quant = unpack(k_packed)           // byte ops
    k_rotated = centroids[k_quant]       // lookup, no multiply
    score[t] = dot(Q_rotated, k_rotated) // dot in rotated space
```

**Key insight**: `dot(Q, R^T * k_rotated) = dot(R * Q, k_rotated)`. Pre-rotate Q once (`O(d log d)`), then compute dots directly in the rotated domain. **No inverse Hadamard on cached vectors at all.**

Same for values: accumulate `sum(attn[t] * v_rotated[t])` in rotated space, then inverse-rotate the final output once.

This reduces dequant to just unpack + table lookup per element — essentially free compared to the dot product itself.

**Revised cost with fused dequant-dot:**

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| Rotate Q once | 896 | Hadamard |
| Quantize K, V (new token) | ~2,816 | Per head |
| Attention loop (unpack + lookup + dot) | `~2 * seq * d` | Same as FP32 attention |
| Inverse rotate output | 896 | Hadamard |
| **Overhead vs FP32 attention** | **~0.7%** | Just two Hadamard transforms |

---

## Testing Strategy

### Unit tests (`test_turboquant.c`)

1. **Roundtrip accuracy**: Quantize random vectors, dequantize, measure MSE. Assert MSE < threshold for each bit width.
2. **Rotation orthogonality**: Verify `R * R^T = I` (within float tolerance).
3. **Dot product preservation**: `|dot(q, k) - dot(q_dequant, k_dequant)| < epsilon` for random q, k.
4. **Attention score preservation**: Full attention with FP32 cache vs TQ cache, compare output cosine similarity. Target: >0.999 for K4/V2.
5. **Determinism**: Same seed produces identical quantization across runs.
6. **Edge cases**: All-zero vectors, all-same vectors, very large/small magnitudes.
7. **Cache eviction**: Verify memmove on packed data preserves correct values after dequant.
8. **Bit packing**: Verify pack/unpack roundtrip for all bit widths (2, 3, 4) with all possible level values.

### Integration test

Run RWKV-X 0.2B inference with and without TQ cache on the same prompt. Compare:
- Output token agreement (should match for short sequences)
- Perplexity difference on a held-out text (target: <0.5% increase)

---

## Implementation Order

1. **`tq_codebooks.h`** — Generate Lloyd-Max tables with Python script, ship as static header
2. **`turboquant.c`** — Core: Hadamard, quantize, dequantize, pack/unpack, cache struct
3. **`test_turboquant.c`** — Unit tests for roundtrip, accuracy, dot preservation
4. **RWKV-X integration** — Wire into `rwkvx.c` attention blocks behind `--tq-cache` flag
5. **Pantheon integration** — Wire into `llama.c` behind same flag
6. **Benchmarks** — tok/s with and without, memory usage comparison
