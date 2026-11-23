# optim

Optimizers for training neural networks. Python counterpart: PyTorch `torch.optim`.

**Header:** `optim/optim.h`
**Source:** `optim/optim.c`
**Status:** API Complete - Pending Verification

---

## Overview

The optim module provides:
- SGD with momentum and Nesterov
- Adam with AMSGrad
- AdamW (decoupled weight decay)
- Learning rate schedulers
- Gradient clipping

---

## Quick Start

```c
#include "optim/optim.h"
#include "tensor/tensor.h"

// Create parameters
tm_tensor* params[] = {weight, bias};
int num_params = 2;

// Create Adam optimizer
fo_adam_config cfg = fo_adam_default(1e-3);
fo_optimizer* opt = fo_adam_new(params, num_params, &cfg);

// Add warmup cosine scheduler
fo_scheduler_config sched = {
    .type = FO_SCHED_WARMUP_COSINE,
    .warmup_steps = 1000,
    .total_steps = 100000,
    .min_lr = 1e-6
};
fo_set_scheduler(opt, &sched);

// Training loop
for (int step = 0; step < 100000; step++) {
    fo_zero_grad(opt);

    // ... forward and backward pass ...

    fo_clip_grad_norm(opt, 1.0);
    fo_step(opt);
    fo_scheduler_step(opt);
}

fo_free(opt);
```

---

## API Reference

### Configuration

#### fo_sgd_config
```c
typedef struct {
    float lr;           // Learning rate
    float momentum;     // Momentum factor (0 = no momentum)
    float weight_decay; // L2 regularization
    float dampening;    // Dampening for momentum
    bool nesterov;      // Use Nesterov momentum
} fo_sgd_config;
```

#### fo_adam_config
```c
typedef struct {
    float lr;           // Learning rate
    float beta1;        // First moment decay (default: 0.9)
    float beta2;        // Second moment decay (default: 0.999)
    float eps;          // Numerical stability (default: 1e-8)
    float weight_decay; // Weight decay (0 for Adam, >0 for AdamW)
    bool amsgrad;       // Use AMSGrad variant
} fo_adam_config;
```

### Default Configurations

```c
fo_sgd_config fo_sgd_default(float lr);
fo_adam_config fo_adam_default(float lr);
```

### Optimizer Creation

```c
fo_optimizer* fo_sgd_new(tm_tensor** params, int num_params, fo_sgd_config* config);
fo_optimizer* fo_adam_new(tm_tensor** params, int num_params, fo_adam_config* config);
fo_optimizer* fo_adamw_new(tm_tensor** params, int num_params, fo_adam_config* config);
void fo_free(fo_optimizer* opt);
```

### Optimizer Operations

```c
void fo_step(fo_optimizer* opt);           // Update parameters
void fo_zero_grad(fo_optimizer* opt);      // Zero gradients
float fo_get_lr(fo_optimizer* opt);        // Get current LR
void fo_set_lr(fo_optimizer* opt, float lr); // Set LR
int fo_get_step(fo_optimizer* opt);        // Get step count
```

### Learning Rate Schedulers

```c
typedef enum {
    FO_SCHED_NONE,
    FO_SCHED_STEP,
    FO_SCHED_COSINE,
    FO_SCHED_LINEAR,
    FO_SCHED_WARMUP_COSINE,
} fo_scheduler_type;

void fo_set_scheduler(fo_optimizer* opt, fo_scheduler_config* config);
void fo_scheduler_step(fo_optimizer* opt);
float fo_get_scheduled_lr(fo_optimizer* opt, int step);
```

### Gradient Clipping

```c
float fo_clip_grad_norm(fo_optimizer* opt, float max_norm);  // Returns norm
void fo_clip_grad_value(fo_optimizer* opt, float clip_value);
```

---

## Scheduler Types

| Type | Description |
|------|-------------|
| `FO_SCHED_STEP` | Decay by gamma every step_size steps |
| `FO_SCHED_COSINE` | Cosine annealing to min_lr |
| `FO_SCHED_LINEAR` | Linear decay to min_lr |
| `FO_SCHED_WARMUP_COSINE` | Warmup then cosine decay |

---

## See Also

- [tensor](tensor.md) - Tensor operations
- [distributed](distributed.md) - Distributed gradient handling
