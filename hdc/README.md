# HDC Module (`fv_`)

> Hyperdimensional Computing operations for Foundry.

---

## Overview

General-purpose Hyperdimensional Computing (HDC) library that provides binary hypervector manipulation, random projection encoding, and fast similarity search via Hamming distance.

HDC encodes information as high-dimensional binary vectors (typically 4096 bits) and performs computation using simple bitwise operations. This provides extremely fast similarity search with compact storage — a 4096-dim binary vector requires only 512 bytes.

## API Reference

### Types

```c
/* Binary hypervector — bit-packed storage */
typedef struct {
    uint64_t *bits;     /* ceil(dim/64) uint64 words */
    int dim;
} fv_vector;

/* Opaque projection matrix for float→binary encoding */
typedef struct fv_projector fv_projector;

/* Configuration */
typedef struct {
    int dim;            /* Hypervector dimension (default: 4096) */
    uint64_t seed;      /* RNG seed for deterministic projection */
} fv_config;
```

### Vector Lifecycle

| Function | Description |
|----------|-------------|
| `fv_alloc(dim)` | Allocate zero-initialized vector |
| `fv_random(dim, seed)` | Deterministic random vector (~50% bits set) |
| `fv_zeros(dim)` | All-zero vector |
| `fv_clone(v)` | Deep copy |
| `fv_free(v)` | Free (NULL-safe) |

### Core Operations

| Function | HDC Name | Description |
|----------|----------|-------------|
| `fv_xor(a, b)` | **Bind** | Associates two concepts. Self-inverse: `XOR(XOR(a,b), b) = a` |
| `fv_permute(v, shift)` | **Permute** | Cyclic bit shift for position encoding |
| `fv_bundle(vecs, n)` | **Bundle** | Majority vote — creates superposition of concepts |
| `fv_bundle_weighted(vecs, w, n)` | **Weighted Bundle** | Weighted majority vote |
| `fv_hamming(a, b)` | **Distance** | Hamming distance (count differing bits) |
| `fv_similarity(a, b)` | **Similarity** | `1.0 - hamming/dim`, range [0, 1] |

### Random Projection

```c
/* Create projector: maps input_dim floats → hdc_dim binary vector */
fv_projector* fv_projector_new(int input_dim, int hdc_dim, uint64_t seed);
void fv_projector_free(fv_projector *proj);

/* Project: sign(R @ x) where R is Gaussian random matrix */
fv_vector* fv_project(const fv_projector *proj, const tm_tensor *embedding);
```

### Tensor Conversion

```c
tm_tensor* fv_to_tensor(const fv_vector *v);     /* → [dim] float, 0.0/1.0 */
fv_vector* fv_from_tensor(const tm_tensor *t);   /* → binary, threshold > 0 */
```

### Serialization

```c
int fv_save(const fv_vector *v, const char *path);   /* 4 bytes dim + bit data */
fv_vector* fv_load(const char *path);
```

### Batch Search

```c
/* Find top-k most similar vectors. Returns count found. */
int fv_search(fv_vector **corpus, int corpus_size,
              const fv_vector *query, int top_k,
              int *indices_out, float *scores_out);
```

## Usage Example

```c
#include "hdc/hdc.h"
#include "tensor/tensor.h"

/* Encode float embedding to binary hypervector */
fv_projector *proj = fv_projector_new(768, 4096, 42);

int64_t shape[] = {768};
tm_tensor *emb = tm_randn(shape, 1, TM_F32);
fv_vector *hv = fv_project(proj, emb);

/* Search corpus */
int indices[5];
float scores[5];
fv_search(corpus, 1000, hv, 5, indices, scores);

/* Key-value binding: associate concept with position */
fv_vector *key_hv = fv_random(4096, key_seed);
fv_vector *bound = fv_xor(key_hv, hv);    /* bind key to value */
fv_vector *recovered = fv_xor(bound, key_hv); /* unbind to get value back */

fv_free(hv);
fv_free(bound);
fv_free(recovered);
fv_free(key_hv);
fv_projector_free(proj);
tm_free(emb);
```

## Performance

- **Hamming distance**: XOR + `popcountll` over 64 uint64 words (4096-bit). ~nanoseconds per comparison.
- **Search (brute force)**: Linear scan with partial sort. ~1M comparisons/second for 4096-bit vectors.
- **Memory**: 512 bytes per 4096-bit vector. 1M vectors = 512 MB.
- **Projection**: One matrix-vector multiply per encoding. Dominated by `input_dim * hdc_dim` multiply-adds.

## Build

```bash
cd foundry
make hdc           # Build module
make test-hdc      # Run 18 tests
```

## Dependencies

- `tensor/` — For `tm_tensor` type used in projection and conversion
