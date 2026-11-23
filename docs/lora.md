# lora

Low-Rank Adaptation (LoRA) for efficient fine-tuning. Python counterpart: HuggingFace PEFT.

**Header:** `lora/lora.h`
**Source:** `lora/lora.c`
**Status:** API Complete - Pending Verification

---

## Overview

The LoRA module provides:
- LoRA layer creation and forward pass
- Weight merging for inference
- Adapter save/load (safetensors format)
- Configurable rank, alpha, and dropout

For weight matrix W (d_out x d_in), LoRA adds:
```
output = Wx + (alpha/r) * B @ A @ x
```
where A is (r x d_in) and B is (d_out x r)

---

## Quick Start

```c
#include "lora/lora.h"

// Configure LoRA
fl_config cfg = fl_config_default();  // rank=8, alpha=16, dropout=0
cfg.rank = 16;

// Create LoRA layer
fl_layer* layer = fl_layer_new("model.attn.q_proj", 4096, 4096, &cfg);

// Forward pass (adds LoRA output to base output)
tm_tensor* lora_out = fl_layer_forward(layer, x);

// Merge into base weights for inference
tm_tensor* merged = fl_layer_merge(layer, base_weight);

// Save adapter
fl_adapter* adapter = fl_adapter_new();
fl_adapter_add(adapter, layer);
fl_adapter_save(adapter, "adapter.safetensors");

fl_adapter_free(adapter);
```

---

## API Reference

### Configuration

```c
typedef struct {
    int rank;            // LoRA rank (r)
    float alpha;         // Scaling factor
    float dropout;       // Dropout rate (0.0 = no dropout)
    bool fan_in_fan_out; // For Conv1D compatibility
} fl_config;

fl_config fl_config_default(void);  // rank=8, alpha=16, dropout=0
```

### LoRA Layer

```c
fl_layer* fl_layer_new(const char* name, int d_in, int d_out, fl_config* config);
fl_layer* fl_layer_from_tensors(const char* name, tm_tensor* A, tm_tensor* B, float alpha);
void fl_layer_free(fl_layer* layer);

// Forward: returns BAx scaled by alpha/r
tm_tensor* fl_layer_forward(fl_layer* layer, tm_tensor* x);

// Merge: W_new = W + scale * B @ A
tm_tensor* fl_layer_merge(fl_layer* layer, tm_tensor* W);

// Get merged without modifying original
tm_tensor* fl_layer_get_merged(fl_layer* layer, tm_tensor* W);
```

### Adapter (Collection of Layers)

```c
fl_adapter* fl_adapter_new(void);
void fl_adapter_free(fl_adapter* adapter);

void fl_adapter_add(fl_adapter* adapter, fl_layer* layer);
fl_layer* fl_adapter_get(fl_adapter* adapter, const char* name);
int fl_adapter_size(fl_adapter* adapter);
fl_layer* fl_adapter_get_idx(fl_adapter* adapter, int idx);

fl_adapter* fl_adapter_load(const char* path, fl_config* config);
bool fl_adapter_save(fl_adapter* adapter, const char* path);
```

### Utility

```c
int64_t fl_adapter_num_params(fl_adapter* adapter);
void fl_adapter_print(fl_adapter* adapter);
```

---

## See Also

- [tensor](tensor.md) - Tensor operations
- [safetensors](safetensors.md) - Adapter file format
- [attention](attention.md) - Attention layers to adapt
