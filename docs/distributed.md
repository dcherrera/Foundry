# distributed

Distributed training primitives. Python counterpart: DeepSpeed/FSDP basics.

**Header:** `distributed/distributed.h`
**Source:** `distributed/distributed.c`
**Status:** API Complete - Pending Verification

---

## Overview

The distributed module provides:
- Process group initialization
- Collective operations (all-reduce, broadcast, etc.)
- Gradient accumulation
- Data sharding

This is a simplified single-process implementation that can be extended with MPI/NCCL for multi-node training. For Docker Swarm distributed training, see [swarm](swarm.md).

---

## Quick Start

```c
#include "distributed/distributed.h"

// Initialize (single process or from env)
fd_config cfg = fd_init_from_env();
// Or: fd_config cfg = fd_init(world_size, rank);

printf("Rank %d of %d\n", cfg.rank, cfg.world_size);

// Gradient accumulation
fd_grad_accum* accum = fd_grad_accum_new(params, num_params, 4);  // 4 steps

for (int step = 0; step < total_steps; step++) {
    // Forward/backward
    tm_backward(loss);

    // Accumulate gradients
    fd_grad_accum_step(accum, params);

    if (fd_grad_accum_should_step(accum)) {
        fd_grad_accum_finish(accum, params);
        fd_all_reduce_mean(gradients);  // Sync across workers
        fo_step(optimizer);
    }
}

fd_grad_accum_free(accum);
fd_cleanup();
```

---

## API Reference

### Initialization

```c
fd_config fd_init(int world_size, int rank);
fd_config fd_init_from_env(void);  // Uses WORLD_SIZE, RANK, LOCAL_RANK
bool fd_is_initialized(void);
fd_config fd_get_config(void);
void fd_cleanup(void);
```

### Configuration

```c
typedef struct {
    int world_size;   // Total number of processes
    int rank;         // This process's rank (0 to world_size-1)
    int local_rank;   // Rank within node
    bool is_main;     // Is this the main process (rank 0)
} fd_config;
```

### Collective Operations

```c
void fd_all_reduce(tm_tensor* tensor);       // Sum across all processes
void fd_all_reduce_mean(tm_tensor* tensor);  // Average across all processes
void fd_broadcast(tm_tensor* tensor, int src_rank);
void fd_reduce(tm_tensor* tensor, int dst_rank);
tm_tensor** fd_all_gather(tm_tensor* tensor);
void fd_barrier(void);  // Synchronize all processes
```

### Gradient Accumulation

```c
fd_grad_accum* fd_grad_accum_new(tm_tensor** params, int num_params,
                                  int accumulation_steps);
void fd_grad_accum_free(fd_grad_accum* accum);
void fd_grad_accum_step(fd_grad_accum* accum, tm_tensor** params);
bool fd_grad_accum_should_step(fd_grad_accum* accum);
void fd_grad_accum_finish(fd_grad_accum* accum, tm_tensor** params);
void fd_grad_accum_reset(fd_grad_accum* accum);
```

### Data Parallel

```c
int* fd_shard_indices(int total_size, int* shard_size);
tm_tensor* fd_shard_batch(tm_tensor* batch, int batch_dim);
```

### Checkpointing

```c
bool fd_should_save(void);       // Only rank 0 should save
void fd_checkpoint_barrier(void); // Synchronize around checkpoints
```

### Utility

```c
void fd_print(const char* fmt, ...);  // Print on main process only
uint64_t fd_get_seed(uint64_t base_seed);  // Rank-specific seed
void fd_print_config(fd_config* config);
```

---

## See Also

- [swarm](swarm.md) - Docker Swarm distributed training
- [optim](optim.md) - Optimizers
