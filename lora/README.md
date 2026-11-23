# lora/

C implementation of Low-Rank Adaptation (LoRA) for efficient fine-tuning.

## Python counterpart

**PEFT** - https://github.com/huggingface/peft

## Why PEFT?

PEFT (Parameter-Efficient Fine-Tuning) is HuggingFace's library for adapter-based training. Core technique is LoRA:

- Freeze base model weights
- Add small trainable low-rank matrices
- Train only ~0.1% of parameters
- Merge adapters back into base model

Alternatives (Unsloth, Axolotl, LLaMA-Factory) are wrappers around PEFT with extra features.

## Scope

This module implements:

1. **LoRA layers** - Low-rank A and B matrices
2. **Target selection** - Apply LoRA to specific layers
3. **Merge/unmerge** - Combine adapters with base weights
4. **Adapter save/load** - safetensors format
5. **Multi-adapter** - Load and switch between adapters

## Getting Started

### Dependencies

- tensor/ module (for autograd)
- safetensors/ module (for save/load)

### Key Files to Study

From PEFT:
- `src/peft/tuners/lora/layer.py` - LoRA layer implementation
- `src/peft/tuners/lora/model.py` - Model wrapping

From the paper:
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### Core API

> Original design sketch, not the implemented API. The real API is `lora/lora.h`; see
> [`docs/lora.md`](../docs/lora.md).

```c
// Create LoRA layer wrapping a linear layer
fl_lora* fl_lora_new(tm_linear* base, int rank, float alpha, float dropout);

// Forward pass (base + lora contribution)
tm_tensor* fl_lora_forward(fl_lora* layer, tm_tensor* x);

// Merge LoRA into base weights (for inference)
void fl_lora_merge(fl_lora* layer);
void fl_lora_unmerge(fl_lora* layer);

// Save/load adapters
void fl_lora_save(fl_lora** layers, int n, const char* path);
fl_lora** fl_lora_load(const char* path, int* n);

// Get trainable parameters only
tm_tensor** fl_lora_params(fl_lora** layers, int n, int* num_params);
```

## Implementation Order

1. LoRA linear layer (forward only)
2. LoRA backward pass
3. Initialization (Kaiming for A, zero for B)
4. Merge/unmerge
5. Save/load (safetensors format)
6. Multi-adapter support

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
