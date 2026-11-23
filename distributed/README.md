# distributed/

C implementation of memory-efficient distributed training.

## Python counterpart

**DeepSpeed** - https://github.com/microsoft/DeepSpeed

## Why DeepSpeed?

DeepSpeed enables training models that don't fit in GPU memory through:

- ZeRO optimizer stages (partition optimizer states, gradients, parameters)
- Gradient checkpointing
- Mixed precision training
- Pipeline parallelism

Alternatives (Megatron-LM, FairScale, FSDP) solve similar problems differently. DeepSpeed is most widely used for LoRA training.

## Scope

This module implements:

1. **Memory-efficient optimizer** - Partition Adam states across GPUs
2. **Gradient checkpointing** - Recompute activations to save memory
3. **Mixed precision** - FP16/BF16 training with FP32 master weights
4. **Communication primitives** - AllReduce, AllGather for gradients

## Getting Started

### Dependencies

- tensor/ module (foundation)
- MPI or NCCL for multi-GPU
- Optional: CUDA-aware MPI

### Key Files to Study

From DeepSpeed:
- `deepspeed/runtime/zero/` - ZeRO implementation
- `deepspeed/runtime/activation_checkpointing/` - Gradient checkpointing

### Core API

> Original design sketch, not the implemented API. The real API is `distributed/distributed.h`; see
> [`docs/distributed.md`](../docs/distributed.md).

```c
// Initialize distributed
fd_init(int* argc, char*** argv);
int fd_rank(void);
int fd_world_size(void);

// ZeRO optimizer wrapper
fd_zero_optimizer* fd_zero_wrap(tm_adamw* opt, int stage);
void fd_zero_step(fd_zero_optimizer* opt);

// Gradient checkpointing
tm_tensor* fd_checkpoint(tm_tensor* (*fn)(tm_tensor*), tm_tensor* input);

// Communication
void fd_all_reduce(tm_tensor* t);
void fd_broadcast(tm_tensor* t, int root);
```

## Implementation Order

1. Single-GPU gradient checkpointing
2. Mixed precision training
3. Multi-GPU communication (MPI)
4. ZeRO Stage 1 (optimizer state partitioning)
5. ZeRO Stage 2 (gradient partitioning)
6. ZeRO Stage 3 (parameter partitioning)

## References

- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
