/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "coordinator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>

/* ============================================================= */
/* Internal Structures                                           */
/* ============================================================= */

#define MAX_WORKERS 128

struct sw_coordinator {
    sw_coordinator_config config;

    /* Network */
    sw_socket* server;
    sw_socket_set* socket_set;

    /* Model */
    tm_tensor** weights;
    int num_weights;
    tm_tensor** grad_accum;         /* Accumulated gradients */
    int grad_count;                 /* Number of gradient contributions */

    /* Optimizer */
    fo_optimizer* optimizer;

    /* Workers */
    sw_worker_info workers[MAX_WORKERS];
    int num_workers;
    pthread_mutex_t workers_lock;

    /* State */
    int global_step;
    float current_lr;
    bool running;
    bool training_started;
    uint64_t start_time;
    uint64_t total_samples;
    float avg_loss;

    /* Checkpointing */
    char last_checkpoint[SW_MAX_PATH];

    /* Callbacks */
    sw_on_worker_join on_join;
    void* on_join_ctx;
    sw_on_worker_leave on_leave;
    void* on_leave_ctx;
    sw_on_step on_step;
    void* on_step_ctx;
    sw_on_checkpoint on_checkpoint;
    void* on_checkpoint_ctx;

    /* Logging */
    sw_log_level log_level;
    FILE* log_file;

    /* Threading */
    pthread_t thread;
    bool thread_started;
};

/* ============================================================= */
/* Logging                                                       */
/* ============================================================= */

static void coord_log(sw_coordinator* c, sw_log_level level, const char* fmt, ...) {
    if (!c || level < c->log_level) return;

    FILE* f = c->log_file ? c->log_file : stdout;

    time_t now = time(NULL);
    struct tm* tm = localtime(&now);
    char timebuf[32];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", tm);

    const char* level_str = "???";
    switch (level) {
        case SW_LOG_DEBUG: level_str = "DEBUG"; break;
        case SW_LOG_INFO:  level_str = "INFO "; break;
        case SW_LOG_WARN:  level_str = "WARN "; break;
        case SW_LOG_ERROR: level_str = "ERROR"; break;
    }

    fprintf(f, "[%s] [%s] ", timebuf, level_str);

    va_list args;
    va_start(args, fmt);
    vfprintf(f, fmt, args);
    va_end(args);

    fprintf(f, "\n");
    fflush(f);
}

/* ============================================================= */
/* Configuration                                                 */
/* ============================================================= */

sw_coordinator_config sw_coordinator_default_config(void) {
    sw_coordinator_config cfg = {0};

    cfg.port = SW_DEFAULT_PORT;

    cfg.learning_rate = 1e-4f;
    cfg.lr_min = 1e-6f;
    cfg.lr_warmup_steps = 100;
    cfg.total_steps = 10000;

    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.999f;
    cfg.weight_decay = 0.0f;

    cfg.min_workers = 1;
    cfg.sync_every_n_grads = 1;

    strcpy(cfg.checkpoint_dir, "./checkpoints");
    cfg.checkpoint_every_steps = 1000;

    cfg.heartbeat_timeout_sec = 60;
    cfg.allow_rejoin = true;

    cfg.compression = SW_COMPRESS_TOPK;
    cfg.topk_ratio = 0.1f;

    return cfg;
}

/* ============================================================= */
/* Gradient Accumulation                                         */
/* ============================================================= */

static void init_grad_accum(sw_coordinator* c) {
    c->grad_accum = calloc(c->num_weights, sizeof(tm_tensor*));
    for (int i = 0; i < c->num_weights; i++) {
        c->grad_accum[i] = tm_zeros(c->weights[i]->shape, c->weights[i]->ndim, TM_F32);
    }
    c->grad_count = 0;
}

static void free_grad_accum(sw_coordinator* c) {
    if (!c->grad_accum) return;
    for (int i = 0; i < c->num_weights; i++) {
        tm_release(c->grad_accum[i]);
    }
    free(c->grad_accum);
    c->grad_accum = NULL;
}

static void zero_grad_accum(sw_coordinator* c) {
    for (int i = 0; i < c->num_weights; i++) {
        float* data = tm_data_f32(c->grad_accum[i]);
        memset(data, 0, c->grad_accum[i]->numel * sizeof(float));
    }
    c->grad_count = 0;
}

static void accumulate_grads(sw_coordinator* c, tm_tensor** grads) {
    for (int i = 0; i < c->num_weights; i++) {
        float* acc = tm_data_f32(c->grad_accum[i]);
        float* g = tm_data_f32(grads[i]);
        for (int64_t j = 0; j < c->grad_accum[i]->numel; j++) {
            acc[j] += g[j];
        }
    }
    c->grad_count++;
}

static void average_grads(sw_coordinator* c) {
    if (c->grad_count == 0) return;
    float scale = 1.0f / c->grad_count;
    for (int i = 0; i < c->num_weights; i++) {
        float* acc = tm_data_f32(c->grad_accum[i]);
        for (int64_t j = 0; j < c->grad_accum[i]->numel; j++) {
            acc[j] *= scale;
        }
    }
}

static void apply_grads_to_weights(sw_coordinator* c) {
    /* Copy accumulated grads to weight grad tensors */
    for (int i = 0; i < c->num_weights; i++) {
        if (!c->weights[i]->grad) {
            c->weights[i]->grad = tm_zeros(c->weights[i]->shape, c->weights[i]->ndim, TM_F32);
        }
        float* wg = tm_data_f32(c->weights[i]->grad);
        float* ag = tm_data_f32(c->grad_accum[i]);
        memcpy(wg, ag, c->grad_accum[i]->numel * sizeof(float));
    }

    /* Apply optimizer step */
    fo_step(c->optimizer);
    c->global_step++;
    c->current_lr = fo_get_lr(c->optimizer);
}

/* ============================================================= */
/* Worker Management                                             */
/* ============================================================= */

static sw_worker_info* find_worker_by_socket(sw_coordinator* c, sw_socket* sock) {
    for (int i = 0; i < c->num_workers; i++) {
        if (c->workers[i].socket == sock) {
            return &c->workers[i];
        }
    }
    return NULL;
}

static sw_worker_info* find_worker_by_id(sw_coordinator* c, const char* id) {
    for (int i = 0; i < c->num_workers; i++) {
        if (strcmp(c->workers[i].id, id) == 0) {
            return &c->workers[i];
        }
    }
    return NULL;
}

static int count_active_workers(sw_coordinator* c) {
    int count = 0;
    for (int i = 0; i < c->num_workers; i++) {
        if (c->workers[i].state == SW_WORKER_ACTIVE) {
            count++;
        }
    }
    return count;
}

static void remove_worker(sw_coordinator* c, sw_worker_info* worker) {
    if (!worker) return;

    coord_log(c, SW_LOG_INFO, "Worker %s disconnected", worker->id);

    if (c->on_leave) {
        c->on_leave(c, worker, c->on_leave_ctx);
    }

    /* Close socket */
    if (worker->socket) {
        sw_socket_set_remove(c->socket_set, worker->socket);
        sw_socket_close(worker->socket);
    }

    /* Remove from array */
    int idx = worker - c->workers;
    memmove(&c->workers[idx], &c->workers[idx + 1],
            (c->num_workers - idx - 1) * sizeof(sw_worker_info));
    c->num_workers--;
}

static void check_worker_heartbeats(sw_coordinator* c) {
    uint64_t now = (uint64_t)time(NULL);
    uint64_t timeout = c->config.heartbeat_timeout_sec;

    for (int i = c->num_workers - 1; i >= 0; i--) {
        sw_worker_info* w = &c->workers[i];
        if (w->state == SW_WORKER_ACTIVE || w->state == SW_WORKER_PENDING) {
            if (now - w->last_heartbeat > timeout) {
                coord_log(c, SW_LOG_WARN, "Worker %s timed out (no heartbeat for %lus)",
                          w->id, now - w->last_heartbeat);
                w->state = SW_WORKER_DEAD;
                remove_worker(c, w);
            }
        }
    }
}

/* ============================================================= */
/* Message Handling                                              */
/* ============================================================= */

static void handle_register(sw_coordinator* c, sw_socket* sock, uint8_t* msg) {
    sw_register_payload* payload = sw_msg_get_payload(msg);

    coord_log(c, SW_LOG_INFO, "Worker %s registering (profile: %s, cores: %d, mem: %lu MB)",
              payload->worker_id, sw_profile_name(payload->profile),
              payload->num_cores, (unsigned long)payload->memory_mb);

    /* Check if already registered */
    sw_worker_info* existing = find_worker_by_id(c, payload->worker_id);
    if (existing) {
        if (c->config.allow_rejoin) {
            coord_log(c, SW_LOG_INFO, "Worker %s rejoining", payload->worker_id);
            remove_worker(c, existing);
        } else {
            coord_log(c, SW_LOG_WARN, "Worker %s already registered, rejecting", payload->worker_id);
            /* Send error */
            uint8_t* err = sw_msg_alloc(SW_MSG_ERROR, sizeof(sw_error_payload));
            sw_error_payload* ep = sw_msg_get_payload(err);
            ep->error_code = 1;
            strcpy(ep->message, "Already registered");
            sw_send_msg(sock, err);
            sw_msg_free(err);
            return;
        }
    }

    /* Check capacity */
    if (c->num_workers >= MAX_WORKERS) {
        coord_log(c, SW_LOG_WARN, "Max workers reached, rejecting %s", payload->worker_id);
        uint8_t* err = sw_msg_alloc(SW_MSG_ERROR, sizeof(sw_error_payload));
        sw_error_payload* ep = sw_msg_get_payload(err);
        ep->error_code = 2;
        strcpy(ep->message, "Max workers reached");
        sw_send_msg(sock, err);
        sw_msg_free(err);
        return;
    }

    /* Add worker */
    pthread_mutex_lock(&c->workers_lock);
    sw_worker_info* w = &c->workers[c->num_workers++];
    memset(w, 0, sizeof(sw_worker_info));
    strncpy(w->id, payload->worker_id, SW_MAX_WORKER_ID - 1);
    w->socket = sock;
    w->state = SW_WORKER_PENDING;
    w->profile = payload->profile;
    w->shard_index = c->num_workers - 1;
    w->batch_size = payload->suggested_batch_size > 0 ?
                    payload->suggested_batch_size :
                    sw_profile_batch_size(payload->profile);
    w->accumulation_steps = sw_profile_accumulation_steps(payload->profile);
    w->last_heartbeat = (uint64_t)time(NULL);
    w->connected_at = w->last_heartbeat;
    pthread_mutex_unlock(&c->workers_lock);

    /* Send ack */
    uint8_t* ack = sw_msg_alloc(SW_MSG_REGISTER_ACK, sizeof(sw_register_ack_payload));
    sw_register_ack_payload* ap = sw_msg_get_payload(ack);
    ap->accepted = true;
    ap->assigned_shard = w->shard_index;
    ap->total_shards = c->num_workers;
    ap->batch_size = w->batch_size;
    ap->accumulation_steps = w->accumulation_steps;
    ap->compression = c->config.compression;
    ap->topk_ratio = c->config.topk_ratio;
    ap->global_step = c->global_step;

    sw_send_msg(sock, ack);
    sw_msg_free(ack);

    sw_socket_set_add(c->socket_set, sock);

    if (c->on_join) {
        c->on_join(c, w, c->on_join_ctx);
    }

    coord_log(c, SW_LOG_INFO, "Worker %s registered (shard %d, batch %d)",
              w->id, w->shard_index, w->batch_size);
}

static void handle_heartbeat(sw_coordinator* c, sw_socket* sock, uint8_t* msg) {
    sw_heartbeat_payload* payload = sw_msg_get_payload(msg);
    sw_worker_info* w = find_worker_by_id(c, payload->worker_id);

    if (!w) {
        coord_log(c, SW_LOG_WARN, "Heartbeat from unknown worker %s", payload->worker_id);
        return;
    }

    w->last_heartbeat = (uint64_t)time(NULL);
    w->local_step = payload->local_step;
    w->last_loss = payload->current_loss;
    w->samples_per_sec = payload->samples_per_sec;

    /* Send ack */
    uint8_t* ack = sw_msg_alloc(SW_MSG_HEARTBEAT_ACK, 0);
    sw_send_msg(sock, ack);
    sw_msg_free(ack);
}

static void handle_gradients(sw_coordinator* c, sw_socket* sock, uint8_t* msg) {
    sw_gradients_payload* header = sw_msg_get_payload(msg);
    sw_worker_info* w = find_worker_by_id(c, header->worker_id);

    if (!w) {
        coord_log(c, SW_LOG_WARN, "Gradients from unknown worker %s", header->worker_id);
        return;
    }

    coord_log(c, SW_LOG_DEBUG, "Received gradients from %s (step %d)",
              w->id, header->local_step);

    /* Deserialize gradients */
    uint8_t* grad_data = (uint8_t*)header + sizeof(sw_gradients_payload);
    tm_tensor** grads = calloc(c->num_weights, sizeof(tm_tensor*));

    size_t offset = 0;
    for (int i = 0; i < c->num_weights; i++) {
        grads[i] = tm_zeros(c->weights[i]->shape, c->weights[i]->ndim, TM_F32);
        float* data = tm_data_f32(grads[i]);
        offset += sw_deserialize_tensor(grad_data + offset, data,
                                         grads[i]->numel, header->compression);
    }

    /* Accumulate */
    pthread_mutex_lock(&c->workers_lock);
    accumulate_grads(c, grads);
    w->grads_contributed++;

    /* Check if we should update */
    bool should_update = (c->grad_count >= c->config.sync_every_n_grads);
    if (should_update) {
        average_grads(c);
        apply_grads_to_weights(c);

        /* Update average loss */
        float total_loss = 0;
        int loss_count = 0;
        for (int i = 0; i < c->num_workers; i++) {
            if (c->workers[i].state == SW_WORKER_ACTIVE) {
                total_loss += c->workers[i].last_loss;
                loss_count++;
            }
        }
        if (loss_count > 0) c->avg_loss = total_loss / loss_count;

        coord_log(c, SW_LOG_INFO, "Step %d | LR: %.2e | Loss: %.4f | Workers: %d",
                  c->global_step, c->current_lr, c->avg_loss, count_active_workers(c));

        if (c->on_step) {
            c->on_step(c, c->global_step, c->avg_loss, c->on_step_ctx);
        }

        /* Check for checkpoint */
        if (c->config.checkpoint_every_steps > 0 &&
            c->global_step % c->config.checkpoint_every_steps == 0) {
            char path[SW_MAX_PATH];
            snprintf(path, sizeof(path), "%s/step_%d",
                     c->config.checkpoint_dir, c->global_step);
            sw_coordinator_save_checkpoint(c, path);
        }

        /* Check for completion */
        if (c->global_step >= c->config.total_steps) {
            coord_log(c, SW_LOG_INFO, "Training complete!");
            sw_coordinator_finish_training(c);
        }

        zero_grad_accum(c);
    }
    pthread_mutex_unlock(&c->workers_lock);

    /* Free temporary gradients */
    for (int i = 0; i < c->num_weights; i++) {
        tm_release(grads[i]);
    }
    free(grads);

    /* Send ack */
    uint8_t* ack = sw_msg_alloc(SW_MSG_GRADIENTS_ACK, 0);
    sw_send_msg(sock, ack);
    sw_msg_free(ack);
}

static void handle_weights_req(sw_coordinator* c, sw_socket* sock) {
    coord_log(c, SW_LOG_DEBUG, "Sending weights to worker");

    /* Send metadata first */
    uint8_t* meta = sw_msg_alloc(SW_MSG_WEIGHTS_META,
        sizeof(sw_weights_meta_payload) + c->num_weights * sizeof(sw_tensor_meta));
    sw_weights_meta_payload* mp = sw_msg_get_payload(meta);
    mp->num_tensors = c->num_weights;
    mp->global_step = c->global_step;
    mp->learning_rate = c->current_lr;

    sw_tensor_meta* tmeta = (sw_tensor_meta*)(mp + 1);
    for (int i = 0; i < c->num_weights; i++) {
        snprintf(tmeta[i].name, sizeof(tmeta[i].name), "weight_%d", i);
        tmeta[i].ndim = c->weights[i]->ndim;
        for (int d = 0; d < c->weights[i]->ndim; d++) {
            tmeta[i].shape[d] = c->weights[i]->shape[d];
        }
        tmeta[i].numel = c->weights[i]->numel;
        tmeta[i].dtype = c->weights[i]->dtype;
    }

    sw_send_msg(sock, meta);
    sw_msg_free(meta);

    /* Send weights */
    for (int i = 0; i < c->num_weights; i++) {
        size_t data_size = c->weights[i]->numel * sizeof(float);
        uint8_t* wmsg = sw_msg_alloc(SW_MSG_WEIGHTS, data_size);
        memcpy(sw_msg_get_payload(wmsg), c->weights[i]->data, data_size);
        sw_send_msg(sock, wmsg);
        sw_msg_free(wmsg);
    }
}

static void handle_message(sw_coordinator* c, sw_socket* sock, uint8_t* msg) {
    sw_msg_header* hdr = sw_msg_get_header(msg);

    switch (hdr->type) {
        case SW_MSG_REGISTER:
            handle_register(c, sock, msg);
            break;
        case SW_MSG_HEARTBEAT:
            handle_heartbeat(c, sock, msg);
            break;
        case SW_MSG_GRADIENTS:
            handle_gradients(c, sock, msg);
            break;
        case SW_MSG_WEIGHTS_REQ:
            handle_weights_req(c, sock);
            break;
        case SW_MSG_DEREGISTER: {
            sw_worker_info* w = find_worker_by_socket(c, sock);
            if (w) remove_worker(c, w);
            break;
        }
        default:
            coord_log(c, SW_LOG_WARN, "Unknown message type: %d", hdr->type);
    }
}

/* ============================================================= */
/* Main Loop                                                     */
/* ============================================================= */

static void* coordinator_thread(void* arg) {
    sw_coordinator* c = (sw_coordinator*)arg;
    sw_coordinator_run(c);
    return NULL;
}

void sw_coordinator_run(sw_coordinator* c) {
    if (!c) return;

    c->running = true;
    c->start_time = (uint64_t)time(NULL);

    coord_log(c, SW_LOG_INFO, "Coordinator starting on port %d", c->config.port);

    while (c->running) {
        /* Accept new connections */
        sw_socket* new_client = sw_server_accept(c->server, 1);
        if (new_client) {
            coord_log(c, SW_LOG_DEBUG, "New connection from %s",
                      sw_socket_get_info(new_client).remote_addr);
            /* Don't add to set yet - wait for registration */

            /* Read registration message */
            uint8_t* msg = sw_recv_msg(new_client, 10);
            if (msg && sw_msg_validate(msg, sw_msg_total_size(msg))) {
                handle_message(c, new_client, msg);
                sw_msg_free(msg);
            } else {
                coord_log(c, SW_LOG_WARN, "Invalid registration, closing connection");
                sw_socket_close(new_client);
            }
        }

        /* Check existing workers */
        if (c->num_workers > 0) {
            int ready = sw_socket_set_wait(c->socket_set, 100);
            if (ready > 0) {
                for (int i = 0; i < c->num_workers; i++) {
                    sw_socket* sock = c->workers[i].socket;
                    if (sw_socket_set_is_readable(c->socket_set, sock)) {
                        uint8_t* msg = sw_recv_msg(sock, 1);
                        if (msg) {
                            if (sw_msg_validate(msg, sw_msg_total_size(msg))) {
                                handle_message(c, sock, msg);
                            }
                            sw_msg_free(msg);
                        } else if (!sw_socket_is_connected(sock)) {
                            /* Worker disconnected */
                            remove_worker(c, &c->workers[i]);
                            i--;  /* Adjust index after removal */
                        }
                    }
                }
            }
        }

        /* Periodic tasks */
        check_worker_heartbeats(c);
    }

    coord_log(c, SW_LOG_INFO, "Coordinator stopped");
}

/* ============================================================= */
/* Public API                                                    */
/* ============================================================= */

sw_coordinator* sw_coordinator_new(
    tm_tensor** weights,
    int num_weights,
    sw_coordinator_config* config
) {
    if (!weights || num_weights <= 0) return NULL;

    sw_coordinator* c = calloc(1, sizeof(sw_coordinator));
    if (!c) return NULL;

    c->config = config ? *config : sw_coordinator_default_config();
    c->weights = weights;
    c->num_weights = num_weights;
    c->log_level = SW_LOG_INFO;
    c->current_lr = c->config.learning_rate;

    pthread_mutex_init(&c->workers_lock, NULL);

    /* Create server socket */
    c->server = sw_server_create(c->config.port);
    if (!c->server) {
        coord_log(c, SW_LOG_ERROR, "Failed to create server: %s", sw_socket_error());
        free(c);
        return NULL;
    }

    c->socket_set = sw_socket_set_new();

    /* Initialize gradient accumulator */
    init_grad_accum(c);

    /* Create optimizer */
    fo_adam_config adam_cfg = fo_adam_default(c->config.learning_rate);
    adam_cfg.beta1 = c->config.beta1;
    adam_cfg.beta2 = c->config.beta2;
    adam_cfg.weight_decay = c->config.weight_decay;

    /* Enable gradients on weights */
    for (int i = 0; i < num_weights; i++) {
        tm_set_requires_grad(weights[i], true);
    }

    c->optimizer = fo_adamw_new(weights, num_weights, &adam_cfg);

    /* Set up learning rate schedule */
    if (c->config.lr_warmup_steps > 0) {
        fo_scheduler_config sched = fo_scheduler_default();
        sched.type = FO_SCHED_WARMUP_COSINE;
        sched.warmup_steps = c->config.lr_warmup_steps;
        sched.total_steps = c->config.total_steps;
        sched.min_lr = c->config.lr_min;
        fo_set_scheduler(c->optimizer, &sched);
    }

    /* Create checkpoint directory */
    mkdir(c->config.checkpoint_dir, 0755);

    return c;
}

void sw_coordinator_free(sw_coordinator* c) {
    if (!c) return;

    sw_coordinator_stop(c);

    /* Close all worker connections */
    for (int i = 0; i < c->num_workers; i++) {
        if (c->workers[i].socket) {
            sw_socket_close(c->workers[i].socket);
        }
    }

    if (c->socket_set) sw_socket_set_free(c->socket_set);
    if (c->server) sw_socket_close(c->server);
    if (c->optimizer) fo_free(c->optimizer);

    free_grad_accum(c);
    pthread_mutex_destroy(&c->workers_lock);

    if (c->log_file && c->log_file != stdout) {
        fclose(c->log_file);
    }

    free(c);
}

bool sw_coordinator_start(sw_coordinator* c) {
    if (!c || c->thread_started) return false;

    if (pthread_create(&c->thread, NULL, coordinator_thread, c) != 0) {
        return false;
    }

    c->thread_started = true;
    return true;
}

void sw_coordinator_stop(sw_coordinator* c) {
    if (!c) return;
    c->running = false;

    if (c->thread_started) {
        pthread_join(c->thread, NULL);
        c->thread_started = false;
    }
}

bool sw_coordinator_is_running(sw_coordinator* c) {
    return c && c->running;
}

bool sw_coordinator_wait_for_workers(sw_coordinator* c, int timeout_sec) {
    if (!c) return false;

    time_t start = time(NULL);
    while (c->num_workers < c->config.min_workers) {
        if (time(NULL) - start > timeout_sec) {
            return false;
        }
        usleep(100000);  /* 100ms */
    }
    return true;
}

void sw_coordinator_start_training(sw_coordinator* c) {
    if (!c) return;

    c->training_started = true;

    /* Notify all workers */
    uint8_t* msg = sw_msg_alloc(SW_MSG_RESUME, 0);
    for (int i = 0; i < c->num_workers; i++) {
        if (c->workers[i].state == SW_WORKER_PENDING) {
            c->workers[i].state = SW_WORKER_ACTIVE;
            sw_send_msg(c->workers[i].socket, msg);
        }
    }
    sw_msg_free(msg);

    coord_log(c, SW_LOG_INFO, "Training started with %d workers", c->num_workers);
}

void sw_coordinator_pause_training(sw_coordinator* c) {
    if (!c) return;

    uint8_t* msg = sw_msg_alloc(SW_MSG_PAUSE, 0);
    for (int i = 0; i < c->num_workers; i++) {
        if (c->workers[i].state == SW_WORKER_ACTIVE) {
            c->workers[i].state = SW_WORKER_PAUSED;
            sw_send_msg(c->workers[i].socket, msg);
        }
    }
    sw_msg_free(msg);

    coord_log(c, SW_LOG_INFO, "Training paused");
}

void sw_coordinator_resume_training(sw_coordinator* c) {
    if (!c) return;

    uint8_t* msg = sw_msg_alloc(SW_MSG_RESUME, 0);
    for (int i = 0; i < c->num_workers; i++) {
        if (c->workers[i].state == SW_WORKER_PAUSED) {
            c->workers[i].state = SW_WORKER_ACTIVE;
            sw_send_msg(c->workers[i].socket, msg);
        }
    }
    sw_msg_free(msg);

    coord_log(c, SW_LOG_INFO, "Training resumed");
}

void sw_coordinator_finish_training(sw_coordinator* c) {
    if (!c) return;

    uint8_t* msg = sw_msg_alloc(SW_MSG_SHUTDOWN, 0);
    for (int i = 0; i < c->num_workers; i++) {
        sw_send_msg(c->workers[i].socket, msg);
    }
    sw_msg_free(msg);

    /* Save final checkpoint */
    char path[SW_MAX_PATH];
    snprintf(path, sizeof(path), "%s/final", c->config.checkpoint_dir);
    sw_coordinator_save_checkpoint(c, path);

    c->running = false;
}

int sw_coordinator_get_step(sw_coordinator* c) {
    return c ? c->global_step : 0;
}

float sw_coordinator_get_lr(sw_coordinator* c) {
    return c ? c->current_lr : 0.0f;
}

int sw_coordinator_num_workers(sw_coordinator* c) {
    return c ? c->num_workers : 0;
}

int sw_coordinator_num_active_workers(sw_coordinator* c) {
    return c ? count_active_workers(c) : 0;
}

sw_worker_info* sw_coordinator_get_worker(sw_coordinator* c, int index) {
    if (!c || index < 0 || index >= c->num_workers) return NULL;
    return &c->workers[index];
}

sw_worker_info* sw_coordinator_get_worker_by_id(sw_coordinator* c, const char* id) {
    return c ? find_worker_by_id(c, id) : NULL;
}

uint64_t sw_coordinator_total_samples(sw_coordinator* c) {
    return c ? c->total_samples : 0;
}

float sw_coordinator_avg_loss(sw_coordinator* c) {
    return c ? c->avg_loss : 0.0f;
}

float sw_coordinator_throughput(sw_coordinator* c) {
    if (!c) return 0.0f;
    float total = 0.0f;
    for (int i = 0; i < c->num_workers; i++) {
        if (c->workers[i].state == SW_WORKER_ACTIVE) {
            total += c->workers[i].samples_per_sec;
        }
    }
    return total;
}

bool sw_coordinator_save_checkpoint(sw_coordinator* c, const char* path) {
    if (!c || !path) return false;

    mkdir(path, 0755);

    coord_log(c, SW_LOG_INFO, "Saving checkpoint to %s", path);

    /* TODO: Use safetensors module to save weights */
    /* For now, save as raw binary */
    char weights_path[SW_MAX_PATH];
    snprintf(weights_path, sizeof(weights_path), "%s/weights.bin", path);
    FILE* f = fopen(weights_path, "wb");
    if (!f) return false;

    fwrite(&c->num_weights, sizeof(int), 1, f);
    for (int i = 0; i < c->num_weights; i++) {
        fwrite(&c->weights[i]->ndim, sizeof(int), 1, f);
        fwrite(c->weights[i]->shape, sizeof(int64_t), c->weights[i]->ndim, f);
        fwrite(&c->weights[i]->numel, sizeof(int64_t), 1, f);
        fwrite(c->weights[i]->data, sizeof(float), c->weights[i]->numel, f);
    }
    fclose(f);

    /* Save training state */
    char state_path[SW_MAX_PATH];
    snprintf(state_path, sizeof(state_path), "%s/state.json", path);
    f = fopen(state_path, "w");
    if (f) {
        fprintf(f, "{\n");
        fprintf(f, "  \"global_step\": %d,\n", c->global_step);
        fprintf(f, "  \"learning_rate\": %e,\n", c->current_lr);
        fprintf(f, "  \"avg_loss\": %f,\n", c->avg_loss);
        fprintf(f, "  \"total_samples\": %llu\n", (unsigned long long)c->total_samples);
        fprintf(f, "}\n");
        fclose(f);
    }

    strncpy(c->last_checkpoint, path, SW_MAX_PATH - 1);

    if (c->on_checkpoint) {
        c->on_checkpoint(c, path, c->on_checkpoint_ctx);
    }

    return true;
}

bool sw_coordinator_load_checkpoint(sw_coordinator* c, const char* path) {
    if (!c || !path) return false;

    char weights_path[SW_MAX_PATH];
    snprintf(weights_path, sizeof(weights_path), "%s/weights.bin", path);
    FILE* f = fopen(weights_path, "rb");
    if (!f) return false;

    int num_weights;
    fread(&num_weights, sizeof(int), 1, f);
    if (num_weights != c->num_weights) {
        fclose(f);
        return false;
    }

    for (int i = 0; i < c->num_weights; i++) {
        int ndim;
        int64_t shape[8], numel;
        fread(&ndim, sizeof(int), 1, f);
        fread(shape, sizeof(int64_t), ndim, f);
        fread(&numel, sizeof(int64_t), 1, f);

        if (numel != c->weights[i]->numel) {
            fclose(f);
            return false;
        }
        fread(c->weights[i]->data, sizeof(float), numel, f);
    }
    fclose(f);

    coord_log(c, SW_LOG_INFO, "Loaded checkpoint from %s", path);
    return true;
}

const char* sw_coordinator_last_checkpoint(sw_coordinator* c) {
    return c ? c->last_checkpoint : NULL;
}

void sw_coordinator_on_worker_join(sw_coordinator* c, sw_on_worker_join cb, void* ctx) {
    if (!c) return;
    c->on_join = cb;
    c->on_join_ctx = ctx;
}

void sw_coordinator_on_worker_leave(sw_coordinator* c, sw_on_worker_leave cb, void* ctx) {
    if (!c) return;
    c->on_leave = cb;
    c->on_leave_ctx = ctx;
}

void sw_coordinator_on_step(sw_coordinator* c, sw_on_step cb, void* ctx) {
    if (!c) return;
    c->on_step = cb;
    c->on_step_ctx = ctx;
}

void sw_coordinator_on_checkpoint(sw_coordinator* c, sw_on_checkpoint cb, void* ctx) {
    if (!c) return;
    c->on_checkpoint = cb;
    c->on_checkpoint_ctx = ctx;
}

void sw_coordinator_set_log_level(sw_coordinator* c, sw_log_level level) {
    if (c) c->log_level = level;
}

void sw_coordinator_set_log_file(sw_coordinator* c, const char* path) {
    if (!c) return;
    if (c->log_file && c->log_file != stdout) {
        fclose(c->log_file);
    }
    c->log_file = path ? fopen(path, "a") : NULL;
}
