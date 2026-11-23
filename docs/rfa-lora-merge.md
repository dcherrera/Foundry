# rfa_lora_merge

CLI tool for averaging multiple LoRA adapters. Used for cross-machine parallel training.

**Source:** `examples/rfa_lora_merge.c`
**Build:** `make USE_BLAS=1 rfa-lora-merge`

---

## Overview

Loads N LoRA adapter safetensors files, averages A and B weight matrices element-wise across all adapters (FedAvg), and saves the merged result. All input adapters must have the same structure (same number of layers, same dimensions).

Supports up to 64 input adapters.

---

## Usage

```bash
./examples/rfa_lora_merge \
  --output merged_adapter.safetensors \
  adapter_0.safetensors adapter_1.safetensors [... adapter_N.safetensors]
```

### Cross-machine merge workflow

```bash
# After training on two machines separately:
scp machine2:loras_b/adapter_final.safetensors ./loras_b/

./examples/rfa_lora_merge \
  --output adapter_merged.safetensors \
  loras_a/adapter_final.safetensors \
  loras_b/adapter_final.safetensors
```

### Merging many thread adapters

```bash
./examples/rfa_lora_merge \
  --output adapter_merged.safetensors \
  thread_0.safetensors thread_1.safetensors \
  thread_2.safetensors thread_3.safetensors \
  thread_4.safetensors thread_5.safetensors
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output PATH` | (required) | Output merged adapter path |
| `ADAPTER1 ADAPTER2 ...` | (required) | At least 2 input adapter files |

---

## Averaging Method

Standard FedAvg — element-wise average of all A and B matrices:

```
For each layer i:
    merged.A[i] = (adapter[0].A[i] + adapter[1].A[i] + ... ) / N
    merged.B[i] = (adapter[0].B[i] + adapter[1].B[i] + ... ) / N
```

The tool verifies that all input adapters have the same number of layers before merging.

---

## Output

```
Loading 3 adapters...
  [0] adapter_0.safetensors: 49 layers, 9200000 params
  [1] adapter_1.safetensors: 49 layers, 9200000 params
  [2] adapter_2.safetensors: 49 layers, 9200000 params
Averaging 3 adapters (49 layers each)...
Saved: merged.safetensors (49 layers, 9200000 params)
```

---

## See Also

- [rfa-train](rfa-train.md) - Training that produces the adapters to merge
- [rfa-generate](rfa-generate.md) - Inference with merged adapters
- [lora](lora.md) - LoRA adapter module
