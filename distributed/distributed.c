/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "distributed.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/* ============================================================= */
/* Global State                                                  */
/* ============================================================= */

static fd_config g_config = {0};
static bool g_initialized = false;

/* ============================================================= */
/* Initialization                                                */
/* ============================================================= */

fd_config fd_init(int world_size, int rank) {
    g_config.world_size = world_size > 0 ? world_size : 1;
    g_config.rank = rank >= 0 && rank < g_config.world_size ? rank : 0;
    g_config.local_rank = g_config.rank;  /* Simplified: assume single node */
    g_config.is_main = (g_config.rank == 0);
    g_initialized = true;
    return g_config;
}

fd_config fd_init_from_env(void) {
    int world_size = 1;
    int rank = 0;

    const char* ws = getenv("WORLD_SIZE");
    const char* r = getenv("RANK");

    if (ws) world_size = atoi(ws);
    if (r) rank = atoi(r);

    return fd_init(world_size, rank);
}

bool fd_is_initialized(void) {
    return g_initialized;
}

fd_config fd_get_config(void) {
    if (!g_initialized) {
        return fd_init(1, 0);
    }
    return g_config;
}

void fd_cleanup(void) {
    g_initialized = false;
    memset(&g_config, 0, sizeof(g_config));
}

/* ============================================================= */
/* Collective Operations                                         */
/* ============================================================= */

void fd_all_reduce(tm_tensor* tensor) {
    /* Single-process: no-op */
    if (!tensor || g_config.world_size <= 1) return;

    /* In multi-process mode, this would use MPI_Allreduce */
    /* For now, just a placeholder */
}

void fd_all_reduce_mean(tm_tensor* tensor) {
    if (!tensor) return;

    fd_all_reduce(tensor);

    /* Divide by world size */
    if (g_config.world_size > 1) {
        float* data = tm_data_f32(tensor);
        float scale = 1.0f / g_config.world_size;
        for (int64_t i = 0; i < tensor->numel; i++) {
            data[i] *= scale;
        }
    }
}

void fd_broadcast(tm_tensor* tensor, int src_rank) {
    /* Single-process: no-op */
    if (!tensor || g_config.world_size <= 1) return;
    (void)src_rank;
}

void fd_reduce(tm_tensor* tensor, int dst_rank) {
    /* Single-process: no-op */
    if (!tensor || g_config.world_size <= 1) return;
    (void)dst_rank;
}

tm_tensor** fd_all_gather(tm_tensor* tensor) {
    if (!tensor) return NULL;

    tm_tensor** gathered = calloc(g_config.world_size, sizeof(tm_tensor*));
    if (!gathered) return NULL;

    /* Single-process: just clone the tensor */
    gathered[0] = tm_clone(tensor);

    /* In multi-process mode, would gather from all ranks */
    for (int i = 1; i < g_config.world_size; i++) {
        gathered[i] = tm_clone(tensor);  /* Placeholder */
    }

    return gathered;
}

void fd_barrier(void) {
    /* Single-process: no-op */
    /* In multi-process mode: MPI_Barrier */
}

/* ============================================================= */
/* Gradient Accumulation                                         */
/* ============================================================= */

fd_grad_accum* fd_grad_accum_new(tm_tensor** params, int num_params,
                                  int accumulation_steps) {
    if (!params || num_params <= 0 || accumulation_steps <= 0) return NULL;

    fd_grad_accum* accum = calloc(1, sizeof(fd_grad_accum));
    if (!accum) return NULL;

    accum->num_params = num_params;
    accum->accumulation_steps = accumulation_steps;
    accum->current_step = 0;

    accum->grads = calloc(num_params, sizeof(tm_tensor*));
    if (!accum->grads) {
        free(accum);
        return NULL;
    }

    /* Initialize gradient accumulators to zeros */
    for (int i = 0; i < num_params; i++) {
        if (params[i]) {
            accum->grads[i] = tm_zeros(params[i]->shape, params[i]->ndim, params[i]->dtype);
        }
    }

    return accum;
}

void fd_grad_accum_free(fd_grad_accum* accum) {
    if (!accum) return;

    if (accum->grads) {
        for (int i = 0; i < accum->num_params; i++) {
            tm_release(accum->grads[i]);
        }
        free(accum->grads);
    }

    free(accum);
}

void fd_grad_accum_step(fd_grad_accum* accum, tm_tensor** params) {
    if (!accum || !params) return;

    /* Accumulate gradients */
    for (int i = 0; i < accum->num_params; i++) {
        if (params[i] && params[i]->grad && accum->grads[i]) {
            float* acc = tm_data_f32(accum->grads[i]);
            float* grad = tm_data_f32(params[i]->grad);

            for (int64_t j = 0; j < params[i]->grad->numel; j++) {
                acc[j] += grad[j];
            }
        }
    }

    accum->current_step++;
}

bool fd_grad_accum_should_step(fd_grad_accum* accum) {
    if (!accum) return false;
    return accum->current_step >= accum->accumulation_steps;
}

void fd_grad_accum_finish(fd_grad_accum* accum, tm_tensor** params) {
    if (!accum || !params) return;

    float scale = 1.0f / accum->accumulation_steps;

    /* Copy averaged gradients back to params */
    for (int i = 0; i < accum->num_params; i++) {
        if (params[i] && params[i]->grad && accum->grads[i]) {
            float* dst = tm_data_f32(params[i]->grad);
            float* src = tm_data_f32(accum->grads[i]);

            for (int64_t j = 0; j < params[i]->grad->numel; j++) {
                dst[j] = src[j] * scale;
            }
        }
    }

    fd_grad_accum_reset(accum);
}

void fd_grad_accum_reset(fd_grad_accum* accum) {
    if (!accum) return;

    accum->current_step = 0;

    for (int i = 0; i < accum->num_params; i++) {
        if (accum->grads[i]) {
            memset(tm_data_f32(accum->grads[i]), 0,
                   accum->grads[i]->numel * sizeof(float));
        }
    }
}

/* ============================================================= */
/* Data Parallel                                                 */
/* ============================================================= */

int* fd_shard_indices(int total_size, int* shard_size) {
    if (total_size <= 0 || !shard_size) return NULL;

    int base_size = total_size / g_config.world_size;
    int remainder = total_size % g_config.world_size;

    /* This rank gets base_size + 1 if rank < remainder */
    int my_size = base_size + (g_config.rank < remainder ? 1 : 0);
    *shard_size = my_size;

    int* indices = malloc(my_size * sizeof(int));
    if (!indices) return NULL;

    /* Calculate starting index for this rank */
    int start = 0;
    for (int r = 0; r < g_config.rank; r++) {
        start += base_size + (r < remainder ? 1 : 0);
    }

    for (int i = 0; i < my_size; i++) {
        indices[i] = start + i;
    }

    return indices;
}

tm_tensor* fd_shard_batch(tm_tensor* batch, int batch_dim) {
    (void)batch_dim;
    if (!batch || g_config.world_size <= 1) {
        return tm_retain(batch);
    }

    /* In multi-process mode, would slice the batch */
    /* For single process, return the whole batch */
    return tm_retain(batch);
}

/* ============================================================= */
/* Checkpointing                                                 */
/* ============================================================= */

bool fd_should_save(void) {
    return g_config.is_main;
}

void fd_checkpoint_barrier(void) {
    fd_barrier();
}

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

void fd_print(const char* fmt, ...) {
    if (!g_config.is_main) return;

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

uint64_t fd_get_seed(uint64_t base_seed) {
    return base_seed + g_config.rank * 1000;
}

void fd_print_config(fd_config* config) {
    if (!config) {
        printf("(null config)\n");
        return;
    }

    printf("Distributed config:\n");
    printf("  World size: %d\n", config->world_size);
    printf("  Rank: %d\n", config->rank);
    printf("  Local rank: %d\n", config->local_rank);
    printf("  Is main: %s\n", config->is_main ? "true" : "false");
}
