# metal

Apple Metal GPU compute backend. Accelerates tensor operations on macOS with Apple Silicon.

**Header:** `metal/fm_metal.h` (umbrella; per-family headers in `metal/kernels/`)
**Source:** `metal/fm_metal.m` (core) + one unit per kernel family in `metal/kernels/`
**Status:** Complete (17 tests passing)

---

## Overview

The metal module provides GPU-accelerated tensor operations via Apple's Metal framework:
- MPS GEMM/GEMV (MetalPerformanceShaders matrix multiply)
- Custom compute shaders (WKV-7, softmax, RMS norm, SiLU, elementwise ops)
- INT8 quantized matrix-vector product
- Command batching (accumulate ops into a single GPU submission)
- Unified memory buffers (zero-copy on Apple Silicon when page-aligned)

All operations use threshold-based dispatch: small ops stay on CPU, large batched ops go to GPU. On Apple Silicon's unified memory, single-token decode (M=1) is always faster on CPU due to ~100-200us GPU dispatch overhead.

---

## Build

```bash
# Build with Metal
make clean && make USE_BLAS=1 USE_METAL=1 rwkvx-generate

# Run Metal tests
make USE_METAL=1 test-metal

# Q4_K GEMM correctness + timing harnesses
make test-gemm-q4-k        # batched-decode GEMM (fm_sgemm_q4_k)
make test-gemm-q4-k-f16    # prefill GEMM (fm_gemm_q4_k_f16)
```

The `USE_METAL=1` flag links `-framework Metal -framework MetalPerformanceShaders -framework Foundation` and compiles `metal/fm_metal.m` and `metal/kernels/*.m` with Objective-C flags (no `-std=c11`, adds `-fobjc-arc`).

---

## Quick Start

```c
#include "metal/fm_metal.h"

// Initialize (auto-called by rx_model_load)
fm_init();

// Wrap existing data as Metal buffers (zero-copy if page-aligned)
float A[4] = {1,2,3,4};
float B[4] = {5,6,7,8};
float C[4] = {0};

fm_buffer *ba = fm_buffer_from_ptr(A, sizeof(A));
fm_buffer *bb = fm_buffer_from_ptr(B, sizeof(B));
fm_buffer *bc = fm_buffer_from_ptr(C, sizeof(C));

// GEMM: C = A @ B (2x2 matrices)
fm_sgemm(ba, bb, bc, 2, 2, 2, 1.0f, 0.0f);

// Read result back (zero-copy — same pointer if page-aligned)
float *result = (float*)fm_buffer_data(bc);

fm_buffer_free(ba);
fm_buffer_free(bb);
fm_buffer_free(bc);
fm_free();
```

---

## API Reference

### Lifecycle

```c
int  fm_init(void);       // Create device + command queue + pipeline cache
void fm_free(void);       // Cleanup all Metal resources
bool fm_available(void);  // Metal initialized and ready?
```

### Buffers

```c
// Wrap existing pointer (zero-copy if page-aligned, copies otherwise)
fm_buffer* fm_buffer_from_ptr(void *ptr, size_t bytes);

// Allocate new Metal buffer with shared memory
fm_buffer* fm_buffer_alloc(size_t bytes);

// Free buffer wrapper (does NOT free underlying data from fm_buffer_from_ptr)
void fm_buffer_free(fm_buffer *buf);

// Access buffer data from CPU
void*  fm_buffer_data(fm_buffer *buf);
size_t fm_buffer_size(fm_buffer *buf);
```

### GEMM / GEMV (via MPS)

```c
// C = alpha * A @ B + beta * C — all row-major float32
// A: [M,K], B: [K,N], C: [M,N]
void fm_sgemm(fm_buffer *A, fm_buffer *B, fm_buffer *C,
              int M, int N, int K, float alpha, float beta);

// y = alpha * A @ x + beta * y — A: [M,K] row-major, x: [K], y: [M]
void fm_sgemv(fm_buffer *A, fm_buffer *x, fm_buffer *y,
              int M, int K, float alpha, float beta);
```

### Custom Compute Kernels

```c
// WKV-7 single timestep — state: [n_head, head_size, head_size], modified in-place
void fm_wkv7_one(fm_buffer *r, fm_buffer *w, fm_buffer *k, fm_buffer *v,
                 fm_buffer *neg_kk, fm_buffer *kk_a,
                 fm_buffer *state, fm_buffer *y_out,
                 int n_head, int head_size);

// WKV-7 sequence — inputs: [T, n_head * head_size]
void fm_wkv7_seq(fm_buffer *r, fm_buffer *w, fm_buffer *k, fm_buffer *v,
                 fm_buffer *neg_kk, fm_buffer *kk_a,
                 fm_buffer *state, fm_buffer *y_out,
                 int n_head, int head_size, int t_len);

// Softmax over last dimension — x: [rows, cols], out: [rows, cols]
void fm_softmax(fm_buffer *x, fm_buffer *out, int rows, int cols);

// RMS normalization — x: [n], weight: [n] (may be NULL), out: [n]
void fm_rms_norm(fm_buffer *x, fm_buffer *weight, fm_buffer *out,
                 int n, float eps);

// SiLU activation: out = x * sigmoid(x)
void fm_silu(fm_buffer *x, fm_buffer *out, int n);

// Elementwise
void fm_add(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n);
void fm_mul(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n);

// INT8 quantized matvec — x_q: [cols] int8, w: [rows,cols] int8
void fm_matvec_q8(fm_buffer *x_q, fm_buffer *w, fm_buffer *w_scales,
                  fm_buffer *out, float x_scale, int rows, int cols);

// INT8 quantized GEMM (batched prefill). X: [M,K], W_q8: [N,K], scales: [N]
void fm_sgemm_q8(fm_buffer *X, fm_buffer *W_q8, fm_buffer *scales,
                 fm_buffer *Out, int M, int N, int K);
```

### Q4_K quantized GEMM (weights stay compressed on the GPU)

These kernels operate directly on raw Q4_K super-block bytes (144 bytes per 256
weights, `row_stride_bytes = (cols/256)*144`). The weight never leaves its
quantized form in memory, so only ~0.5625 bytes per weight are read.

```c
// Q4_K matvec (M=1 decode). Fused dequant + dot, one threadgroup per row.
void fm_matvec_q4_k(fm_buffer *w, fm_buffer *x, fm_buffer *out,
                    int rows, int cols, int row_stride_bytes);

// Q4_K x B GEMM (batched decode): out[B, rows] = x[B, cols] @ dequant(w)^T.
// Tiled simdgroup_matrix GEMM (F32 accumulation), B padded to a multiple of 8
// internally, tested up to B = 32.
void fm_sgemm_q4_k(fm_buffer *w, fm_buffer *x, fm_buffer *out,
                   int rows, int cols, int row_stride_bytes, int B);

// Q4_K GEMM for PREFILL (large M): out[M, N] = x[M, K] @ dequant(w)[N, K]^T.
// Dequantizes the weight to F16 and runs an F16 MPS matmul.
void fm_gemm_q4_k_f16(fm_buffer *w, fm_buffer *x_f32, fm_buffer *out_f32,
                      int M, int N, int K, int row_stride_bytes);
```

**Decode GEMM (`fm_sgemm_q4_k`).** One threadgroup owns a 64-row output tile
(256 threads, 8 SIMD groups). Per 64-column slice it stages the batch
activations into threadgroup memory once (coalesced) and cooperatively
dequantizes the Q4_K weights into a transposed tile, both reused across every
batch and row sub-tile, then runs the inner product on `simdgroup_float8x8`
with F32 accumulation. This amortizes the Q4_K dequant across the batch (the GPU
analog of a CPU AMX panel). Correct to err/RMS ~6e-6 against a CPU dequant-dot
reference. About 8x faster than a naive one-row-per-SIMD-group version, and
every tested shape clears the CPU AMX per-weight budget in isolation. This is a
per-weight microbenchmark result; it is the seam for a concurrent CPU+GPU decode
batch, not on its own a full decode-step win.

**Prefill GEMM (`fm_gemm_q4_k_f16`).** For large M (hundreds to thousands of
tokens) the weight is dequantized to F16 and the matmul runs as an F16 MPS GEMM
(F16 has ~2x the GPU throughput of F32). F16 is safe here because prefill
activations are post-RMSNorm and bounded, unlike the outlier-prone vision
residuals that force F32 elsewhere. Correct to err/RMS ~0.004 against a CPU F32
dequant-dot reference. At M=1369 it runs ~4 ms/call versus ~22 ms/weight for the
CPU AMX prefill path (~5.5x per weight); at prefill sizes the per-call
dispatch/sync overhead is negligible.

### Command Batching

```c
int  fm_batch_begin(void);   // Start accumulating GPU commands
void fm_batch_end(void);     // Commit + wait once
bool fm_batch_active(void);  // Check if batching
```

---

## Dispatch Thresholds

Operations below these thresholds stay on CPU:

| Threshold | Value | What it controls |
|-----------|-------|------------------|
| `FM_GEMM_THRESHOLD` | 128M ops (M*K*N) | `tm_matmul()` — needs M >= ~20 for 2560-dim |
| `FM_GEMV_THRESHOLD` | 16M ops (M*K) | `fm_sgemv()` |
| `FM_ELEM_THRESHOLD` | 1M elements | `tm_softmax()`, `tm_rms_norm()`, `tm_silu()` |
| `FM_WKV7_THRESHOLD` | disabled | `rx_wkv7_one()` — always M=1, always CPU |
| `FM_Q8_THRESHOLD` | disabled | INT8 matvec — always M=1, always CPU |

### Why single-token decode stays on CPU

On Apple Silicon, CPU and GPU share the same memory bus. Single-vector operations (M=1) are memory-bandwidth bound — both CPU and GPU hit the same bandwidth ceiling. But GPU adds ~100-200us dispatch overhead per kernel. With ~160 kernel calls per token, that's 16-32ms of pure overhead, turning a 50ms/token baseline into 66-82ms/token.

Metal only wins when batch size (M) is large enough to amortize dispatch overhead — typically M >= 16-20 for GEMM on 2560-dim matrices.

---

## Integration Points

Metal dispatch is integrated at these call sites in the inference engine:

| File | Function | Kernel |
|------|----------|--------|
| `tensor/tensor.c` | `tm_matmul()` | `fm_sgemm` |
| `tensor/tensor.c` | `tm_softmax()` | `fm_softmax` |
| `tensor/tensor.c` | `tm_rms_norm()` | `fm_rms_norm` |
| `tensor/tensor.c` | `tm_silu()` | `fm_silu` |
| `rwkvx/rwkvx.c` | `rx_wkv7_one()` | `fm_wkv7_one` |
| `rwkvx/rwkvx.c` | `fast_matvec_q8_preq()` | `fm_matvec_q8` |
| `rwkvx/rwkvx.c` | `rx_model_load()` | `fm_init()` (auto-init) |

### C API embedding

```bash
# macOS with Metal
cc -DUSE_BLAS -DACCELERATE_NEW_LAPACK -DUSE_METAL -DUSE_THREADS \
   -O3 -march=native -ffast-math \
   your_app.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c \
   hybridmem/hybridmem.c hdc/hdc.c threadpool/threadpool.c \
   metal/fm_metal.m metal/kernels/*.m \
   -framework Accelerate -framework Metal \
   -framework MetalPerformanceShaders -framework Foundation \
   -lm -lpthread
```

Note: `fm_metal.m` and `metal/kernels/*.m` are Objective-C — must not be compiled with `-std=c11`.

---

## Metal Shaders

Shaders are embedded as string fragments, one per kernel family, in the matching
`metal/kernels/fm_kernel_<family>.m`. `fm_kernel_library_source()` concatenates them
into a single source compiled once at `fm_init()`, so nothing has to ship or locate a
`.metallib`. Pipeline states are cached by kernel name plus function-constant values.

The fragments share one MSL translation unit, so a `#define` in one family is visible
to every family after it — prefix macros with the family name and `#undef` them at the
end of the fragment.

| Kernel | Thread layout | Description |
|--------|--------------|-------------|
| `wkv7_one` | 1 threadgroup/head, head_size threads | Single-timestep WKV-7 |
| `wkv7_seq` | 1 threadgroup/head, head_size threads | Sequence WKV-7 |
| `softmax_row` | 1 threadgroup/row, 256 threads | SIMD-reduced row softmax |
| `rms_norm` | 1 threadgroup, 256 threads | SIMD-reduced RMS norm |
| `silu` | 1 thread/element | SiLU activation |
| `elementwise_add` | 1 thread/element | Vector addition |
| `elementwise_mul` | 1 thread/element | Vector multiplication |
| `matvec_q8` | 1 threadgroup/row, 256 threads | INT8 quantized matvec |
| `gemm_q8` | tiled | INT8 quantized GEMM (batched prefill) |
| `matvec_q4_k` | 1 threadgroup/row, 1 SIMD group | Q4_K fused dequant+dot matvec (decode) |
| `gemm_q4_k` | 1 threadgroup/64-row tile, 256 threads | Q4_K x B tiled simdgroup_matrix GEMM (batched decode) |
| `dequant_q4_k_f16` | 1 thread/super-block | Q4_K weight to F16 (prefill GEMM, feeds MPS F16 matmul) |
| `cvt_f32_f16` / `cvt_f16_f32` | 1 thread/element | F32/F16 conversion (prefill GEMM staging) |

---

## See Also

- [tensor](tensor.md) - Tensor operations (Metal dispatch integrated here)
- [quantization](quantization.md) - INT8/INT4 quantization
