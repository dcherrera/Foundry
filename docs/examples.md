# CLI Tools & Examples

Production CLI tools and example programs built on Foundry.

---

## Production CLI Tools

| Tool | Description | Docs |
|------|-------------|------|
| **rwkvx_generate** | RWKV-X text generation (INT8, HybridMem) | [rwkvx-generate.md](rwkvx-generate.md) |
| **rfa_generate** | RFA hybrid text generation (RWKV-7 + Qwen) | [rfa-generate.md](rfa-generate.md) |
| **rfa_train** | RFA hybrid LoRA training (parallel, INT8) | [rfa-train.md](rfa-train.md) |
| **rfa_lora_merge** | LoRA adapter averaging (cross-machine merge) | [rfa-lora-merge.md](rfa-lora-merge.md) |

---

## Build All Tools

```bash
# RWKV-X inference + training
make clean && make USE_BLAS=1 rwkvx-generate

# RFA hybrid inference + training + merge
make clean && make USE_BLAS=1 rfa-generate rfa-train rfa-lora-merge
```

---

## Research & Test Programs

Located in `examples/` but not intended for production use:

| Program | Description |
|---------|-------------|
| `train_mlp.c` | Simple MLP training demo |
| `bench_hybridmem.c` | HybridMem performance benchmark |
| `test_hdc_correlation.c` | HDC vector correlation test |
| `test_hybridmem_retrieval.c` | HybridMem retrieval accuracy test |
| `test_pooling_loss.c` | Mean-pool vs token embedding loss |
| `test_query_vector.c` | HDC query vector generation |
| `test_attn_weights.c` | Attention weight distribution analysis |
| `test_context_scaling.c` | Context length scaling test |
| `test_eviction_boundary.c` | KV cache eviction boundary test |
| `test_false_positive.c` | HDC false positive rate test |
| `test_filler_sensitivity.c` | Filler token sensitivity test |
| `test_multi_fact.c` | Multi-fact retrieval test |
| `test_precision_at_scale.c` | Precision at scale test |
