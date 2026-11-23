/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_SWARM_WORKER_H
#define FOUNDRY_SWARM_WORKER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "protocol.h"
#include "network.h"
#include "../tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Swarm Worker - Training worker client.
 *
 * Responsibilities:
 * - Connect to coordinator
 * - Receive model weights
 * - Load and iterate training data
 * - Compute forward/backward passes
 * - Send gradients to coordinator
 * - Handle pause/resume/shutdown signals
 */

/* ============================================================= */
/* Configuration                                                 */
/* ============================================================= */

typedef struct {
    /* Network */
    char coordinator_host[256];
    uint16_t coordinator_port;
    int reconnect_attempts;         /* Default: 10 */
    int reconnect_delay_sec;        /* Default: 5 */

    /* Hardware */
    int num_threads;                /* 0 = auto-detect */
    int max_memory_mb;              /* 0 = auto-detect */

    /* Training */
    int batch_size;                 /* 0 = auto from profile */
    int accumulation_steps;         /* 0 = auto from profile */

    /* Data */
    char data_path[SW_MAX_PATH];
    int shard_index;                /* -1 = auto-assign from coordinator */

    /* Compression */
    sw_compress_type compression;   /* Overridden by coordinator */
    float topk_ratio;

    /* Identity */
    char worker_id[SW_MAX_WORKER_ID]; /* Empty = auto-generate */

} sw_worker_config;

/* Get default configuration */
sw_worker_config sw_worker_default_config(void);

/* ============================================================= */
/* Forward/Backward Function Types                               */
/* ============================================================= */

/*
 * The worker doesn't know about the model architecture.
 * You provide callbacks for forward and backward passes.
 */

/* Forward function: takes input batch, returns loss */
typedef float (*sw_forward_fn)(
    tm_tensor** weights,
    int num_weights,
    void* batch_data,
    void* ctx
);

/* Backward function: computes gradients for current batch */
typedef void (*sw_backward_fn)(
    tm_tensor** weights,
    int num_weights,
    void* batch_data,
    void* ctx
);

/* Data loader: returns next batch (NULL when epoch done) */
typedef void* (*sw_next_batch_fn)(void* ctx);

/* Reset data loader for new epoch */
typedef void (*sw_reset_data_fn)(void* ctx);

/* Free batch data */
typedef void (*sw_free_batch_fn)(void* batch, void* ctx);

/* ============================================================= */
/* Worker                                                        */
/* ============================================================= */

typedef struct sw_worker sw_worker;

/* Worker state */
typedef enum {
    SW_WSTATE_DISCONNECTED,
    SW_WSTATE_CONNECTING,
    SW_WSTATE_REGISTERED,
    SW_WSTATE_WAITING,      /* Waiting for training to start */
    SW_WSTATE_TRAINING,
    SW_WSTATE_PAUSED,
    SW_WSTATE_FINISHED,
    SW_WSTATE_ERROR,
} sw_wstate;

/* Create worker */
sw_worker* sw_worker_new(sw_worker_config* config);

/* Free worker */
void sw_worker_free(sw_worker* w);

/* ============================================================= */
/* Connection                                                    */
/* ============================================================= */

/* Connect to coordinator */
bool sw_worker_connect(sw_worker* w);

/* Disconnect from coordinator */
void sw_worker_disconnect(sw_worker* w);

/* Check if connected */
bool sw_worker_is_connected(sw_worker* w);

/* Get current state */
sw_wstate sw_worker_get_state(sw_worker* w);

/* ============================================================= */
/* Training                                                      */
/* ============================================================= */

/* Set model weights (worker will update these in place) */
void sw_worker_set_weights(sw_worker* w, tm_tensor** weights, int num_weights);

/* Set training callbacks */
void sw_worker_set_forward(sw_worker* w, sw_forward_fn fn, void* ctx);
void sw_worker_set_backward(sw_worker* w, sw_backward_fn fn, void* ctx);
void sw_worker_set_data_loader(sw_worker* w, sw_next_batch_fn next,
                                sw_reset_data_fn reset, sw_free_batch_fn free_batch,
                                void* ctx);

/* Request initial weights from coordinator */
bool sw_worker_fetch_weights(sw_worker* w);

/* Run training loop (blocking) */
void sw_worker_train(sw_worker* w);

/* Stop training */
void sw_worker_stop(sw_worker* w);

/* ============================================================= */
/* State Access                                                  */
/* ============================================================= */

/* Get worker ID */
const char* sw_worker_get_id(sw_worker* w);

/* Get current local step */
int sw_worker_get_step(sw_worker* w);

/* Get current loss */
float sw_worker_get_loss(sw_worker* w);

/* Get samples per second */
float sw_worker_get_throughput(sw_worker* w);

/* Get assigned shard index */
int sw_worker_get_shard(sw_worker* w);

/* Get batch size (may be set by coordinator) */
int sw_worker_get_batch_size(sw_worker* w);

/* ============================================================= */
/* Callbacks                                                     */
/* ============================================================= */

typedef void (*sw_on_state_change)(sw_worker* w, sw_wstate old_state,
                                    sw_wstate new_state, void* ctx);
typedef void (*sw_on_weights_received)(sw_worker* w, void* ctx);
typedef void (*sw_on_step_complete)(sw_worker* w, int step, float loss, void* ctx);

void sw_worker_on_state_change(sw_worker* w, sw_on_state_change cb, void* ctx);
void sw_worker_on_weights_received(sw_worker* w, sw_on_weights_received cb, void* ctx);
void sw_worker_on_step_complete(sw_worker* w, sw_on_step_complete cb, void* ctx);

/* ============================================================= */
/* Logging                                                       */
/* ============================================================= */

void sw_worker_set_log_level(sw_worker* w, sw_log_level level);
void sw_worker_set_log_file(sw_worker* w, const char* path);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_SWARM_WORKER_H */
