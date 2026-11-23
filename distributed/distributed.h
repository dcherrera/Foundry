/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_DISTRIBUTED_H
#define FOUNDRY_DISTRIBUTED_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Distributed training primitives.
 * Replaces DeepSpeed/FSDP basics.
 *
 * This is a simplified single-process implementation that can be
 * extended with MPI/NCCL for multi-node training.
 */

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* Process group */
typedef struct fd_group fd_group;

/* Distributed config */
typedef struct {
    int world_size;       /* Total number of processes */
    int rank;             /* This process's rank (0 to world_size-1) */
    int local_rank;       /* Rank within node */
    bool is_main;         /* Is this the main process (rank 0) */
} fd_config;

/* Gradient accumulation state */
typedef struct {
    tm_tensor** grads;    /* Accumulated gradients */
    int num_params;
    int accumulation_steps;
    int current_step;
} fd_grad_accum;

/* ============================================================= */
/* Initialization                                                */
/* ============================================================= */

/* Initialize distributed environment
 * For single-process: world_size=1, rank=0
 */
fd_config fd_init(int world_size, int rank);

/* Initialize from environment variables (WORLD_SIZE, RANK, LOCAL_RANK) */
fd_config fd_init_from_env(void);

/* Check if initialized */
bool fd_is_initialized(void);

/* Get current config */
fd_config fd_get_config(void);

/* Cleanup */
void fd_cleanup(void);

/* ============================================================= */
/* Collective Operations                                         */
/* ============================================================= */

/* All-reduce: sum gradients across all processes
 * In single-process mode, this is a no-op
 */
void fd_all_reduce(tm_tensor* tensor);

/* All-reduce with averaging */
void fd_all_reduce_mean(tm_tensor* tensor);

/* Broadcast from rank 0 to all other ranks */
void fd_broadcast(tm_tensor* tensor, int src_rank);

/* Reduce to single rank */
void fd_reduce(tm_tensor* tensor, int dst_rank);

/* All-gather: gather tensors from all ranks */
tm_tensor** fd_all_gather(tm_tensor* tensor);

/* Barrier: synchronize all processes */
void fd_barrier(void);

/* ============================================================= */
/* Gradient Accumulation                                         */
/* ============================================================= */

/* Create gradient accumulator */
fd_grad_accum* fd_grad_accum_new(tm_tensor** params, int num_params,
                                  int accumulation_steps);

/* Free accumulator */
void fd_grad_accum_free(fd_grad_accum* accum);

/* Accumulate gradients from current backward pass */
void fd_grad_accum_step(fd_grad_accum* accum, tm_tensor** params);

/* Check if we should update weights */
bool fd_grad_accum_should_step(fd_grad_accum* accum);

/* Get averaged gradients and reset accumulator */
void fd_grad_accum_finish(fd_grad_accum* accum, tm_tensor** params);

/* Reset accumulator */
void fd_grad_accum_reset(fd_grad_accum* accum);

/* ============================================================= */
/* Data Parallel                                                 */
/* ============================================================= */

/* Shard dataset indices across ranks */
int* fd_shard_indices(int total_size, int* shard_size);

/* Get this rank's portion of a batch */
tm_tensor* fd_shard_batch(tm_tensor* batch, int batch_dim);

/* ============================================================= */
/* Checkpointing                                                 */
/* ============================================================= */

/* Only rank 0 should save checkpoints */
bool fd_should_save(void);

/* Synchronize before/after checkpointing */
void fd_checkpoint_barrier(void);

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

/* Print on main process only */
void fd_print(const char* fmt, ...);

/* Get rank-specific seed */
uint64_t fd_get_seed(uint64_t base_seed);

/* Print distributed config */
void fd_print_config(fd_config* config);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_DISTRIBUTED_H */
