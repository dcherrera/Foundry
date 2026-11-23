/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_SWARM_COORDINATOR_H
#define FOUNDRY_SWARM_COORDINATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "protocol.h"
#include "network.h"
#include "../tensor/tensor.h"
#include "../optim/optim.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Swarm Coordinator - Central training server.
 *
 * Responsibilities:
 * - Store master copy of model weights
 * - Receive and aggregate gradients from workers
 * - Apply optimizer updates
 * - Broadcast updated weights
 * - Track worker health
 * - Save checkpoints
 */

/* ============================================================= */
/* Configuration                                                 */
/* ============================================================= */

typedef struct {
    /* Network */
    uint16_t port;                      /* Default: 7100 */

    /* Training */
    float learning_rate;
    float lr_min;
    int lr_warmup_steps;
    int total_steps;

    /* Optimizer */
    float beta1;                        /* Adam beta1, default: 0.9 */
    float beta2;                        /* Adam beta2, default: 0.999 */
    float weight_decay;                 /* Default: 0.0 */

    /* Synchronization */
    int min_workers;                    /* Wait for N workers before starting */
    int sync_every_n_grads;             /* Aggregate after N gradient pushes */

    /* Checkpointing */
    char checkpoint_dir[SW_MAX_PATH];
    int checkpoint_every_steps;

    /* Fault tolerance */
    int heartbeat_timeout_sec;          /* Mark worker dead after N seconds */
    bool allow_rejoin;                  /* Allow workers to rejoin */

    /* Compression */
    sw_compress_type compression;
    float topk_ratio;

} sw_coordinator_config;

/* Get default configuration */
sw_coordinator_config sw_coordinator_default_config(void);

/* ============================================================= */
/* Worker Info                                                   */
/* ============================================================= */

typedef enum {
    SW_WORKER_PENDING,      /* Registered, waiting to start */
    SW_WORKER_ACTIVE,       /* Training */
    SW_WORKER_PAUSED,       /* Paused */
    SW_WORKER_DEAD,         /* Missed heartbeats */
    SW_WORKER_DISCONNECTED, /* Disconnected */
} sw_worker_state;

typedef struct {
    char id[SW_MAX_WORKER_ID];
    sw_socket* socket;
    sw_worker_state state;
    sw_hardware_profile profile;

    uint32_t shard_index;
    uint32_t batch_size;
    uint32_t accumulation_steps;

    uint32_t local_step;
    uint32_t grads_contributed;
    float last_loss;
    float samples_per_sec;

    uint64_t last_heartbeat;
    uint64_t connected_at;
    uint64_t bytes_sent;
    uint64_t bytes_received;
} sw_worker_info;

/* ============================================================= */
/* Coordinator                                                   */
/* ============================================================= */

typedef struct sw_coordinator sw_coordinator;

/* Create coordinator with model weights */
sw_coordinator* sw_coordinator_new(
    tm_tensor** weights,
    int num_weights,
    sw_coordinator_config* config
);

/* Free coordinator */
void sw_coordinator_free(sw_coordinator* coord);

/* ============================================================= */
/* Lifecycle                                                     */
/* ============================================================= */

/* Start the coordinator (blocking - runs main loop) */
void sw_coordinator_run(sw_coordinator* coord);

/* Start the coordinator in background (non-blocking) */
bool sw_coordinator_start(sw_coordinator* coord);

/* Stop the coordinator */
void sw_coordinator_stop(sw_coordinator* coord);

/* Check if coordinator is running */
bool sw_coordinator_is_running(sw_coordinator* coord);

/* ============================================================= */
/* Training Control                                              */
/* ============================================================= */

/* Wait for minimum workers to connect */
bool sw_coordinator_wait_for_workers(sw_coordinator* coord, int timeout_sec);

/* Signal all workers to start training */
void sw_coordinator_start_training(sw_coordinator* coord);

/* Pause all workers */
void sw_coordinator_pause_training(sw_coordinator* coord);

/* Resume all workers */
void sw_coordinator_resume_training(sw_coordinator* coord);

/* Signal training complete */
void sw_coordinator_finish_training(sw_coordinator* coord);

/* ============================================================= */
/* State Access                                                  */
/* ============================================================= */

/* Get current global step */
int sw_coordinator_get_step(sw_coordinator* coord);

/* Get current learning rate */
float sw_coordinator_get_lr(sw_coordinator* coord);

/* Get number of connected workers */
int sw_coordinator_num_workers(sw_coordinator* coord);

/* Get number of active workers */
int sw_coordinator_num_active_workers(sw_coordinator* coord);

/* Get worker info by index */
sw_worker_info* sw_coordinator_get_worker(sw_coordinator* coord, int index);

/* Get worker info by ID */
sw_worker_info* sw_coordinator_get_worker_by_id(sw_coordinator* coord, const char* id);

/* Get total samples processed */
uint64_t sw_coordinator_total_samples(sw_coordinator* coord);

/* Get average loss across workers */
float sw_coordinator_avg_loss(sw_coordinator* coord);

/* Get aggregate throughput (samples/sec) */
float sw_coordinator_throughput(sw_coordinator* coord);

/* ============================================================= */
/* Checkpointing                                                 */
/* ============================================================= */

/* Save checkpoint now */
bool sw_coordinator_save_checkpoint(sw_coordinator* coord, const char* path);

/* Load checkpoint */
bool sw_coordinator_load_checkpoint(sw_coordinator* coord, const char* path);

/* Get last checkpoint path */
const char* sw_coordinator_last_checkpoint(sw_coordinator* coord);

/* ============================================================= */
/* Callbacks                                                     */
/* ============================================================= */

/* Callback types */
typedef void (*sw_on_worker_join)(sw_coordinator* coord, sw_worker_info* worker, void* ctx);
typedef void (*sw_on_worker_leave)(sw_coordinator* coord, sw_worker_info* worker, void* ctx);
typedef void (*sw_on_step)(sw_coordinator* coord, int step, float loss, void* ctx);
typedef void (*sw_on_checkpoint)(sw_coordinator* coord, const char* path, void* ctx);

/* Set callbacks */
void sw_coordinator_on_worker_join(sw_coordinator* coord, sw_on_worker_join cb, void* ctx);
void sw_coordinator_on_worker_leave(sw_coordinator* coord, sw_on_worker_leave cb, void* ctx);
void sw_coordinator_on_step(sw_coordinator* coord, sw_on_step cb, void* ctx);
void sw_coordinator_on_checkpoint(sw_coordinator* coord, sw_on_checkpoint cb, void* ctx);

/* ============================================================= */
/* Logging                                                       */
/* ============================================================= */

/* Set log level (sw_log_level defined in protocol.h) */
void sw_coordinator_set_log_level(sw_coordinator* coord, sw_log_level level);

/* Set log file (NULL for stdout) */
void sw_coordinator_set_log_file(sw_coordinator* coord, const char* path);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_SWARM_COORDINATOR_H */
