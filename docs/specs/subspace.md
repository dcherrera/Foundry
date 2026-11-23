# Share-Inspired LoRA Subspace Memory for RWKV-X

**Status:** Research / Future Work
**Date:** 2026-02-26
**Based on:** [Shared LoRA Subspaces for almost Strict Continual Learning](https://arxiv.org/abs/2602.06043) (Kaushik et al., Feb 2026)

---

## Summary

Replace independent per-topic LoRA adapters (the LoRA HybridMem proposal) with a single shared SVD-derived basis + tiny per-topic coefficient vectors. Achieves ~11,000x storage reduction per topic while solving the B=C problem from HybridMem research.

**Core idea:** LoRA adapters across tasks converge to a shared low-rank subspace (>0.95 alignment by task 5). One set of basis vectors can represent all task-specific knowledge through per-task coefficient vectors alone. Merging new topics is gradient-free (SVD + analytical reprojection).

---

## Prior Research

Earlier HybridMem experiments (their code and write-ups are not in this repo) ran a 7-phase
viability suite on RWKV-X at 0.2B and 3.6B (INT8):

- Phases 1-4: PASS (HDC projection r=0.9667, attention heads use injected entries, single-fact recall works)
- Phase 5: FAIL (multi-fact: 1/10 improved, 2/10 worse)
- Phase 7b: **B=C** (random retrieval = HDC retrieval, both 2/3)

**Conclusion:** HDC overhead adds zero value over random injection. The model treats injected KV
entries as generic extra window regardless of content. A follow-up proposal (disk-backed LoRA
pages, ~160 KB each, trained online during conversation) is superseded by this spec's shared
subspace.

**Relevant to subspace:** the HDC projection and LSH infrastructure in `hdc/` and `hybridmem/`
can be repurposed for topic routing (selecting which subspace coefficients to activate).

### Foundry HDC/HybridMem Implementation

| File | What |
|------|------|
| `hdc/hdc.h` | HDC vector API (fv_vector, fv_projector, fv_similarity) |
| `hdc/hdc.c` | Dense Gaussian projection, Hamming similarity, LSH |
| `hybridmem/hybridmem.h` | HybridMem store API (fm_store, fm_query, fm_put) |
| `hybridmem/hybridmem.c` | Disk-backed KV store with LSH indexing, shard management |
| `hdc/tests/test_hdc.c` | HDC unit tests |
| `hybridmem/tests/test_hybridmem.c` | HybridMem unit tests |

---

## Why Share Solves B=C

The B=C problem: pretrained attention layers treat injected KV entries as generic extra context regardless of relevance. Random retrieval performs identically to HDC-matched retrieval.

Share avoids this because knowledge goes INTO the weights via LoRA (`y = (W + BA)x`), not into the KV cache as foreign entries. Different topics produce different weight modifications. The model doesn't need to "distinguish" relevant context — each topic's coefficients encode different learned knowledge in the computation path itself.

---

## Storage Comparison (per LoRA layer, d=3584, rank=32)

| Method | Per-Topic Storage | Shared Overhead |
|--------|------------------|-----------------|
| Standard LoRA | 917 KB | -- |
| LoRA HybridMem (proposed) | ~160 KB (rank 8) | -- |
| **Share (k=10, p=1)** | **80 bytes** | 286 KB (one-time) |

For 24 LoRA layers (6 per block x 4 attention blocks):
- Standard LoRA: 22 MB/topic
- **Share: 1.9 KB/topic + 6.8 MB shared basis**

---

## Existing Infrastructure

| Component | File | Status |
|-----------|------|--------|
| SVD (power iteration) | `tensor/tensor.c:7144` `tm_svd()` | Works, tested |
| LoRA A/B storage | `lora/lora.h` `fl_layer` | Done |
| Adapter save/load | `lora/lora.c` `fl_adapter_save/load` | Safetensors I/O |
| Adapter averaging (FedAvg) | `rwkvx/rfa_train.c:1093` `rfa_lora_average()` | Done |
| Training forward+backward | `rwkvx/rfa_train.c` `rfa_train_step()` | Tested |
| AdamW optimizer | `optim/optim.h` `fo_adamw_new()` | Done |
| HDC projection + LSH | `hdc/hdc.c` `fv_project()`, `hybridmem/` | Done |
| LAPACK via Accelerate | Makefile: `-DACCELERATE_NEW_LAPACK` | Linked |

---

## Architecture

### Data Structure

```c
typedef struct {
    int k;              // Number of basis vectors (principal factors)
    int p;              // Pseudo-rank per topic (typically 1)
    int n_topics;       // Number of topics stored
    int topic_capacity; // Allocated capacity for topics

    int n_layers;
    struct fl_subspace_layer {
        char *name;
        int d_in, d_out;

        tm_tensor *beta;    // [d_out x k] shared B basis (orthonormal columns)
        tm_tensor *alpha;   // [d_in x k]  shared A basis (orthonormal columns)

        // Per-topic coefficients (dynamic array, grows with topics)
        tm_tensor **eps_beta;   // [n_topics] each [k x p]
        tm_tensor **eps_alpha;  // [n_topics] each [k x p]
    } *layers;

    float scale;

    char **topic_names;
    fv_vector **topic_sigs;     // HDC signatures for routing (optional)
} fl_subspace;
```

### API

```c
// Build subspace from T trained adapters
fl_subspace* fl_subspace_from_adapters(fl_adapter **adapters, int n_adapters,
                                       int k, int p);

// Forward: output += scale * (beta * eps_beta[t]) @ (alpha * eps_alpha[t])^T @ input
void fl_subspace_forward_add(fl_subspace *sub, int topic_id,
                              const float *input, float *output, int layer_idx);

// Reconstruct full fl_adapter for topic t (compatibility with existing training)
fl_adapter* fl_subspace_reconstruct(fl_subspace *sub, int topic_id);

// Begin new topic: returns temporary adapter for training
fl_adapter* fl_subspace_begin_topic(fl_subspace *sub);

// Merge trained adapter back into subspace (SVD + analytical reprojection)
int fl_subspace_commit_topic(fl_subspace *sub, fl_adapter *trained, const char *name);

// Recalculate all coefficients (gradient-free)
void fl_subspace_reproject(fl_subspace *sub);

// Save/load
bool fl_subspace_save(fl_subspace *sub, const char *path);
fl_subspace* fl_subspace_load(const char *path);
void fl_subspace_free(fl_subspace *sub);

// Route: find best topic(s) for current input via HDC similarity
int* fl_subspace_route(fl_subspace *sub, const float *embedding, int embed_dim,
                       int top_k, float *scores_out);
```

### SVD Basis Extraction (core algorithm)

For each LoRA layer across T adapters:

```
1. Stack B matrices: B_tilde = [scale*B_1, ..., scale*B_T]  in R^(d_out x T*r)
2. SVD: U, S, V = tm_svd(B_tilde)
3. Choose k: smallest k where sum(S[:k]^2) / sum(S^2) >= 0.60
4. Extract basis: beta = U[:, :k]                // [d_out x k]
5. Compute coefficients: eps_beta[t] = beta^T @ (scale*B_t)  // [k x r]
   If p=1: eps_beta[t] = beta^T @ (scale*B_t @ ones)          // [k x 1]
6. Same for A -> alpha, eps_alpha
```

### Forward Pass

**Option A (general, any p):**
```c
effective_B = beta @ eps_beta[t]   // [d_out x k] @ [k x p] -> [d_out x p]
effective_A = alpha @ eps_alpha[t] // [d_in x k] @ [k x p]  -> [d_in x p]
output += scale * effective_B @ effective_A^T @ input
```

**Option B (optimized, p=1):**
```c
b_vec = beta @ eps_beta[t]     // [d_out] vector
a_vec = alpha @ eps_alpha[t]   // [d_in] vector
output += scale * b_vec * dot(a_vec, input)   // rank-1 outer product
```

### New Topic Training

**Step 1 — Temporary expansion:**
- Create standard LoRA adapter with rank phi = k/4
- Initialize from top-phi basis directions
- Return adapter for training via existing `rfa_train_step()`

**Step 2 — Train using existing infrastructure (no changes needed):**
```c
rfa_train_step(model, state, temp_adapter, optimizer, &example, &config);
```

**Step 3 — Merge back (gradient-free):**
```c
// For each layer:
//   1. Reconstruct old: B_hat_i = beta @ eps_beta[i]
//   2. Stack with new: B_tilde = [B_hat_1, ..., B_hat_T, trained_B]
//   3. SVD -> new beta, new k
//   4. Analytical reprojection: eps_beta[i] = beta^T @ B_hat_i  (no gradients!)
//   5. Same for A
```

### Topic Routing (reuse HDC)

```c
// When topic is committed: mean-pool effective embeddings -> fv_project() -> signature
// At inference: mean-pool input -> fv_project() -> query
// fv_similarity(query, topic_sigs[i]) for each topic -> activate top-k
```

Reuses `fv_projector` and `fv_similarity` from `hdc/hdc.c`, and LSH from `hybridmem/` for sub-linear search when topic count grows large.

### Persistence Format (safetensors)

```
Metadata:
  "type": "fl_subspace"
  "k": "10", "p": "1", "n_topics": "25", "scale": "2.0"
  "topic_names": "topic_0,topic_1,..."

Tensors:
  "layer.0.beta":        [d_out, k]
  "layer.0.alpha":       [d_in, k]
  "layer.0.eps_beta.0":  [k, p]
  "layer.0.eps_beta.1":  [k, p]
  ...
  "layer.0.eps_alpha.0": [k, p]
  ...
  "topic_sigs":          [n_topics, hdc_dim/64]
```

---

## Validation Test Suite

Extends the HybridMem 7-phase framework:

| Phase | Test | Pass Criterion |
|-------|------|----------------|
| 1 | Subspace convergence | Alignment >0.90 by topic 5 (reproduce Share Fig. 1) |
| 2 | Reconstruction error | norm(B_hat - beta * eps) / norm(B) < 0.10 at 60% variance |
| 3 | Forward equivalence | Subspace output ~ original LoRA output (RMSE < 0.01) |
| 4 | B != C test | Correct topic coefficients outperform random topic |
| 5 | Online learning | New topic doesn't degrade existing topic quality |
| 6 | Storage efficiency | Measure actual bytes: subspace vs N separate adapters |
| 7 | Routing accuracy | HDC routing selects correct topic >80% |

---

## New Files

| File | Purpose |
|------|---------|
| `lora/lora_subspace.h` | fl_subspace API |
| `lora/lora_subspace.c` | Implementation |
| `lora/tests/test_lora_subspace.c` | Unit tests |

## Modified Files

| File | Change |
|------|--------|
| `Makefile` | Add lora_subspace build and test targets |
| `tensor/tensor.c` | Optional: LAPACK SVD path for larger matrices |

---

## Implementation Order

1. `lora_subspace.h` -- API definition
2. `lora_subspace.c` -- `fl_subspace_from_adapters` (SVD extraction)
3. Unit tests -- basis extraction + reconstruction error
4. Forward pass -- `fl_subspace_forward_add` + equivalence test
5. Commit/merge -- `fl_subspace_commit_topic` + reprojection
6. Save/load -- safetensors persistence
7. Topic routing -- HDC signature integration
8. Validation -- 7-phase test suite against 3.6B model

## Prerequisites

- 2-3 trained LoRA adapters (train on different OASST1 topic splits)
- Training infrastructure works (rfa_train_step, tested)
- SVD works (tm_svd, tested)

## Risks

1. **SVD accuracy** -- Power iteration (100 iter) may need LAPACK upgrade for larger matrices. Accelerate LAPACK already linked.
2. **p=1 capacity** -- Share shows p=1 works for NLU/classification. Conversation memory may need p=2-4.
3. **Training data** -- Need topic-diverse adapters to seed subspace. Can use synthetic splits.

---

## Share Paper Details

**Title:** Shared LoRA Subspaces for almost Strict Continual Learning
**Authors:** Prakhar Kaushik, Ankit Vaidya, Shravan Chaudhari, Rama Chellappa, Alan Yuille
**ArXiv:** [2602.06043](https://arxiv.org/abs/2602.06043) (Feb 5, 2026)

### Key Results from Paper

- 100x parameter reduction vs standard LoRA
- 281x memory savings
- Matches jointly-trained baseline on GLUE, CIFAR-100, Pascal3D+, Flux
- CIFAR-100: 94.20% accuracy (= upper bound), 0.40% forgetting
- Processes 500 LoRA adapters incrementally, maintains 90-99% on IID tasks
- Subspace alignment >0.95 by task 5 (empirically validated)

### Share Algorithm Summary

**Step 1 (Build subspace):** Stack LoRA B matrices -> SVD -> top-k basis vectors (beta). Coefficients: eps = beta^T @ B_hat (pseudoinverse, gradient-free).

**Step 2 (New task):** Expand basis temporarily by phi directions. Train on new data. Only coefficients are learned.

**Step 3 (Merge):** Stack reconstructed old + new -> SVD -> updated basis. Analytical reprojection for all existing topics (no replay, no gradients).

**Forward:** h = W_0 * x + (beta * eps_beta) @ (alpha * eps_alpha)^T @ x

**Key insight:** When beta has orthonormal columns, reprojection is just eps = beta^T @ B_hat -- a single matrix multiply, no optimization needed.
