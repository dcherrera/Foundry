/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Swarm Worker - Training worker client implementation.
 *
 * Connects to coordinator, receives weights, computes gradients,
 * and sends them back for aggregation.
 */

#include "worker.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#include <unistd.h>
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

/* ============================================================= */
/* Internal Structures                                           */
/* ============================================================= */

struct sw_worker {
    sw_worker_config config;
    sw_socket* socket;
    sw_wstate state;

    /* Model */
    tm_tensor** weights;
    int num_weights;
    tm_tensor** gradients;      /* Accumulated gradients */

    /* Training callbacks */
    sw_forward_fn forward_fn;
    void* forward_ctx;
    sw_backward_fn backward_fn;
    void* backward_ctx;

    /* Data loader callbacks */
    sw_next_batch_fn next_batch_fn;
    sw_reset_data_fn reset_data_fn;
    sw_free_batch_fn free_batch_fn;
    void* data_ctx;

    /* Event callbacks */
    sw_on_state_change on_state_change;
    void* on_state_change_ctx;
    sw_on_weights_received on_weights_received;
    void* on_weights_received_ctx;
    sw_on_step_complete on_step_complete;
    void* on_step_complete_ctx;

    /* State */
    int local_step;
    int accumulation_count;
    float current_loss;
    float total_loss;
    int loss_count;

    /* Assigned by coordinator */
    int assigned_shard;
    int assigned_batch_size;
    int assigned_accumulation_steps;

    /* Timing */
    uint64_t last_heartbeat_sent;
    uint64_t step_start_time;
    float samples_per_sec;

    /* Logging */
    sw_log_level log_level;
    FILE* log_file;

    /* Control */
    bool should_stop;
    bool running;
};

/* ============================================================= */
/* Logging Helpers                                               */
/* ============================================================= */

static uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void worker_log(sw_worker* w, sw_log_level level, const char* fmt, ...) {
    if (level < w->log_level) return;

    FILE* out = w->log_file ? w->log_file : stdout;

    const char* level_str = "???";
    switch (level) {
        case SW_LOG_DEBUG: level_str = "DEBUG"; break;
        case SW_LOG_INFO:  level_str = "INFO";  break;
        case SW_LOG_WARN:  level_str = "WARN";  break;
        case SW_LOG_ERROR: level_str = "ERROR"; break;
    }

    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    fprintf(out, "[%02d:%02d:%02d] [%s] [%s] ",
            tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec,
            level_str, w->config.worker_id);

    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);

    fprintf(out, "\n");
    fflush(out);
}

/* ============================================================= */
/* State Management                                              */
/* ============================================================= */

static void set_state(sw_worker* w, sw_wstate new_state) {
    sw_wstate old_state = w->state;
    if (old_state == new_state) return;

    w->state = new_state;

    const char* state_names[] = {
        "DISCONNECTED", "CONNECTING", "REGISTERED",
        "WAITING", "TRAINING", "PAUSED", "FINISHED", "ERROR"
    };
    worker_log(w, SW_LOG_DEBUG, "State: %s -> %s",
               state_names[old_state], state_names[new_state]);

    if (w->on_state_change) {
        w->on_state_change(w, old_state, new_state, w->on_state_change_ctx);
    }
}

/* ============================================================= */
/* Configuration                                                 */
/* ============================================================= */

sw_worker_config sw_worker_default_config(void) {
    sw_worker_config config;
    memset(&config, 0, sizeof(config));

    strcpy(config.coordinator_host, "localhost");
    config.coordinator_port = SW_DEFAULT_PORT;
    config.reconnect_attempts = 10;
    config.reconnect_delay_sec = 5;

    config.num_threads = 0;     /* Auto-detect */
    config.max_memory_mb = 0;   /* Auto-detect */

    config.batch_size = 0;      /* Auto from profile */
    config.accumulation_steps = 0;

    config.shard_index = -1;    /* Auto-assign */
    config.compression = SW_COMPRESS_TOPK;
    config.topk_ratio = 0.1f;

    return config;
}

/* ============================================================= */
/* Worker Creation                                               */
/* ============================================================= */

static void generate_worker_id(char* id, size_t max_len) {
    /* Generate simple unique ID: hostname-pid-random */
    char hostname[64] = "unknown";
    gethostname(hostname, sizeof(hostname));

    uint32_t random_val = (uint32_t)time(NULL) ^ (uint32_t)getpid();
    snprintf(id, max_len, "%s-%d-%04x", hostname, getpid(), random_val & 0xFFFF);
}

sw_worker* sw_worker_new(sw_worker_config* config) {
    sw_worker* w = calloc(1, sizeof(sw_worker));
    if (!w) return NULL;

    w->config = *config;
    w->state = SW_WSTATE_DISCONNECTED;
    w->log_level = SW_LOG_INFO;
    w->assigned_shard = -1;

    /* Generate worker ID if not provided */
    if (w->config.worker_id[0] == '\0') {
        generate_worker_id(w->config.worker_id, SW_MAX_WORKER_ID);
    }

    return w;
}

void sw_worker_free(sw_worker* w) {
    if (!w) return;

    sw_worker_disconnect(w);

    /* Free accumulated gradients */
    if (w->gradients) {
        for (int i = 0; i < w->num_weights; i++) {
            if (w->gradients[i]) {
                tm_free(w->gradients[i]);
            }
        }
        free(w->gradients);
    }

    if (w->log_file && w->log_file != stdout) {
        fclose(w->log_file);
    }

    free(w);
}

/* ============================================================= */
/* Connection                                                    */
/* ============================================================= */

static bool send_register(sw_worker* w) {
    sw_hardware_profile profile = sw_detect_profile();
    uint32_t batch_size = w->config.batch_size > 0 ?
                          (uint32_t)w->config.batch_size :
                          sw_profile_batch_size(profile);

    worker_log(w, SW_LOG_INFO, "Registering with coordinator (profile=%s, batch=%d)",
               sw_profile_name(profile), batch_size);

    uint8_t* msg = sw_msg_alloc(SW_MSG_REGISTER, sizeof(sw_register_payload));
    if (!msg) return false;

    sw_register_payload* payload = sw_msg_get_payload(msg);
    memset(payload, 0, sizeof(*payload));
    strncpy(payload->worker_id, w->config.worker_id, SW_MAX_WORKER_ID - 1);
    payload->protocol_version = SW_PROTOCOL_VERSION;
    payload->profile = profile;
    payload->suggested_batch_size = batch_size;

    bool ok = sw_send_msg(w->socket, msg);
    sw_msg_free(msg);

    return ok;
}

static bool handle_register_ack(sw_worker* w, uint8_t* msg) {
    sw_register_ack_payload* ack = sw_msg_get_payload(msg);

    if (!ack->accepted) {
        worker_log(w, SW_LOG_ERROR, "Registration rejected by coordinator");
        return false;
    }

    w->assigned_shard = ack->assigned_shard;
    w->assigned_batch_size = ack->batch_size;
    w->assigned_accumulation_steps = ack->accumulation_steps;

    worker_log(w, SW_LOG_INFO, "Registered! shard=%d, batch=%d, accum=%d",
               ack->assigned_shard, ack->batch_size, ack->accumulation_steps);

    set_state(w, SW_WSTATE_REGISTERED);
    return true;
}

bool sw_worker_connect(sw_worker* w) {
    if (w->socket) {
        sw_socket_close(w->socket);
        w->socket = NULL;
    }

    set_state(w, SW_WSTATE_CONNECTING);

    for (int attempt = 0; attempt < w->config.reconnect_attempts; attempt++) {
        if (attempt > 0) {
            worker_log(w, SW_LOG_INFO, "Reconnect attempt %d/%d in %d seconds...",
                       attempt + 1, w->config.reconnect_attempts,
                       w->config.reconnect_delay_sec);
            sleep_ms(w->config.reconnect_delay_sec * 1000);
        }

        worker_log(w, SW_LOG_INFO, "Connecting to %s:%d...",
                   w->config.coordinator_host, w->config.coordinator_port);

        w->socket = sw_client_connect(w->config.coordinator_host,
                                       w->config.coordinator_port,
                                       5);  /* 5 second timeout */

        if (!w->socket) {
            worker_log(w, SW_LOG_WARN, "Connection failed: %s", sw_socket_error());
            continue;
        }

        worker_log(w, SW_LOG_INFO, "Connected!");

        /* Send registration */
        if (!send_register(w)) {
            worker_log(w, SW_LOG_ERROR, "Failed to send registration");
            sw_socket_close(w->socket);
            w->socket = NULL;
            continue;
        }

        /* Wait for ack */
        uint8_t* msg = sw_recv_msg(w->socket, 10);  /* 10 sec timeout */
        if (!msg) {
            worker_log(w, SW_LOG_ERROR, "No registration response");
            sw_socket_close(w->socket);
            w->socket = NULL;
            continue;
        }

        sw_msg_header* hdr = sw_msg_get_header(msg);
        if (hdr->type != SW_MSG_REGISTER_ACK) {
            worker_log(w, SW_LOG_ERROR, "Unexpected message type: 0x%02x", hdr->type);
            sw_msg_free(msg);
            sw_socket_close(w->socket);
            w->socket = NULL;
            continue;
        }

        bool ok = handle_register_ack(w, msg);
        sw_msg_free(msg);

        if (ok) {
            return true;
        }

        sw_socket_close(w->socket);
        w->socket = NULL;
    }

    set_state(w, SW_WSTATE_ERROR);
    return false;
}

void sw_worker_disconnect(sw_worker* w) {
    if (w->socket) {
        sw_socket_close(w->socket);
        w->socket = NULL;
    }
    set_state(w, SW_WSTATE_DISCONNECTED);
}

bool sw_worker_is_connected(sw_worker* w) {
    return w->socket != NULL && sw_socket_is_connected(w->socket);
}

sw_wstate sw_worker_get_state(sw_worker* w) {
    return w->state;
}

/* ============================================================= */
/* Weight Management                                             */
/* ============================================================= */

void sw_worker_set_weights(sw_worker* w, tm_tensor** weights, int num_weights) {
    w->weights = weights;
    w->num_weights = num_weights;

    /* Allocate gradient accumulators */
    if (w->gradients) {
        for (int i = 0; i < w->num_weights; i++) {
            if (w->gradients[i]) {
                tm_free(w->gradients[i]);
            }
        }
        free(w->gradients);
    }

    w->gradients = calloc(num_weights, sizeof(tm_tensor*));
    for (int i = 0; i < num_weights; i++) {
        /* Create zero-initialized gradient tensor matching weight shape */
        w->gradients[i] = tm_zeros_like(weights[i]);
    }
}

bool sw_worker_fetch_weights(sw_worker* w) {
    if (!sw_worker_is_connected(w)) {
        worker_log(w, SW_LOG_ERROR, "Not connected");
        return false;
    }

    worker_log(w, SW_LOG_INFO, "Requesting initial weights...");

    uint8_t* req = sw_msg_alloc(SW_MSG_WEIGHTS_REQ, 0);
    if (!req) return false;

    bool ok = sw_send_msg(w->socket, req);
    sw_msg_free(req);

    if (!ok) {
        worker_log(w, SW_LOG_ERROR, "Failed to send weights request");
        return false;
    }

    /* Wait for weights metadata */
    uint8_t* meta_msg = sw_recv_msg(w->socket, 60);  /* 60 sec for large models */
    if (!meta_msg) {
        worker_log(w, SW_LOG_ERROR, "No weights metadata response");
        return false;
    }

    sw_msg_header* hdr = sw_msg_get_header(meta_msg);
    if (hdr->type != SW_MSG_WEIGHTS_META) {
        worker_log(w, SW_LOG_ERROR, "Unexpected message type: 0x%02x", hdr->type);
        sw_msg_free(meta_msg);
        return false;
    }

    sw_weights_meta_payload* meta = sw_msg_get_payload(meta_msg);
    if ((int)meta->num_tensors != w->num_weights) {
        worker_log(w, SW_LOG_ERROR, "Weight count mismatch: got %d, expected %d",
                   meta->num_tensors, w->num_weights);
        sw_msg_free(meta_msg);
        return false;
    }

    worker_log(w, SW_LOG_INFO, "Receiving %d weight tensors (step=%d)",
               meta->num_tensors, meta->global_step);
    sw_msg_free(meta_msg);

    /* Receive each weight tensor */
    for (int i = 0; i < w->num_weights; i++) {
        uint8_t* weight_msg = sw_recv_msg(w->socket, 60);
        if (!weight_msg) {
            worker_log(w, SW_LOG_ERROR, "Failed to receive weight tensor %d", i);
            return false;
        }

        hdr = sw_msg_get_header(weight_msg);
        if (hdr->type != SW_MSG_WEIGHTS) {
            worker_log(w, SW_LOG_ERROR, "Unexpected message type: 0x%02x", hdr->type);
            sw_msg_free(weight_msg);
            return false;
        }

        /* Copy weight data */
        void* data = sw_msg_get_payload(weight_msg);
        size_t size = tm_numel(w->weights[i]) * sizeof(float);
        memcpy(w->weights[i]->data, data, size);

        sw_msg_free(weight_msg);
    }

    worker_log(w, SW_LOG_INFO, "Received all weights");

    if (w->on_weights_received) {
        w->on_weights_received(w, w->on_weights_received_ctx);
    }

    return true;
}

/* ============================================================= */
/* Training Callbacks                                            */
/* ============================================================= */

void sw_worker_set_forward(sw_worker* w, sw_forward_fn fn, void* ctx) {
    w->forward_fn = fn;
    w->forward_ctx = ctx;
}

void sw_worker_set_backward(sw_worker* w, sw_backward_fn fn, void* ctx) {
    w->backward_fn = fn;
    w->backward_ctx = ctx;
}

void sw_worker_set_data_loader(sw_worker* w, sw_next_batch_fn next,
                                sw_reset_data_fn reset, sw_free_batch_fn free_batch,
                                void* ctx) {
    w->next_batch_fn = next;
    w->reset_data_fn = reset;
    w->free_batch_fn = free_batch;
    w->data_ctx = ctx;
}

/* ============================================================= */
/* Gradient Handling                                             */
/* ============================================================= */

static void zero_gradients(sw_worker* w) {
    for (int i = 0; i < w->num_weights; i++) {
        tm_fill_(w->gradients[i], 0.0f);
    }
}

static void accumulate_gradients(sw_worker* w) {
    /* After backward pass, accumulate gradients from weights->grad */
    for (int i = 0; i < w->num_weights; i++) {
        tm_tensor* grad = w->weights[i]->grad;
        if (grad) {
            tm_add_(w->gradients[i], grad);
        }
    }
    w->accumulation_count++;
}

static bool send_gradients(sw_worker* w) {
    if (w->accumulation_count == 0) return true;

    /* Average accumulated gradients */
    float scale = 1.0f / w->accumulation_count;
    for (int i = 0; i < w->num_weights; i++) {
        tm_scale_(w->gradients[i], scale);
    }

    /* Calculate total gradient size */
    size_t total_elements = 0;
    for (int i = 0; i < w->num_weights; i++) {
        total_elements += tm_numel(w->gradients[i]);
    }

    /* Determine compressed size */
    sw_compress_type compression = w->config.compression;
    size_t data_size;

    if (compression == SW_COMPRESS_TOPK) {
        size_t k = (size_t)(total_elements * w->config.topk_ratio);
        data_size = k * (sizeof(uint32_t) + sizeof(float));
    } else if (compression == SW_COMPRESS_FP16) {
        data_size = total_elements * sizeof(uint16_t);
    } else {
        data_size = total_elements * sizeof(float);
    }

    size_t payload_size = sizeof(sw_gradients_payload) + data_size;
    uint8_t* msg = sw_msg_alloc(SW_MSG_GRADIENTS, payload_size);
    if (!msg) return false;

    sw_gradients_payload* gp = sw_msg_get_payload(msg);
    memset(gp, 0, sizeof(*gp));
    strncpy(gp->worker_id, w->config.worker_id, SW_MAX_WORKER_ID - 1);
    gp->local_step = w->local_step;
    gp->num_tensors = w->num_weights;
    gp->compression = compression;
    gp->uncompressed_size = total_elements * sizeof(float);

    /* Serialize gradients (after the header struct) */
    uint8_t* data = (uint8_t*)(gp + 1);

    if (compression == SW_COMPRESS_TOPK) {
        size_t k = (size_t)(total_elements * w->config.topk_ratio);

        /* Flatten all gradients */
        float* all_grads = malloc(total_elements * sizeof(float));
        size_t offset = 0;
        for (int i = 0; i < w->num_weights; i++) {
            size_t n = tm_numel(w->gradients[i]);
            memcpy(all_grads + offset, tm_data_f32(w->gradients[i]), n * sizeof(float));
            offset += n;
        }

        /* Get TopK */
        uint32_t* indices = (uint32_t*)data;
        float* values = (float*)(data + k * sizeof(uint32_t));
        sw_topk_compress(all_grads, total_elements, k, indices, values);

        free(all_grads);
    } else if (compression == SW_COMPRESS_FP16) {
        for (int i = 0; i < w->num_weights; i++) {
            size_t n = tm_numel(w->gradients[i]);
            sw_fp32_to_fp16(tm_data_f32(w->gradients[i]), (uint16_t*)data, n);
            data += n * sizeof(uint16_t);
        }
    } else {
        for (int i = 0; i < w->num_weights; i++) {
            size_t n = tm_numel(w->gradients[i]) * sizeof(float);
            memcpy(data, tm_data_f32(w->gradients[i]), n);
            data += n;
        }
    }

    worker_log(w, SW_LOG_DEBUG, "Sending gradients (step=%d, samples=%d)",
               w->local_step, w->accumulation_count * w->assigned_batch_size);

    bool ok = sw_send_msg(w->socket, msg);
    sw_msg_free(msg);

    /* Reset accumulators */
    zero_gradients(w);
    w->accumulation_count = 0;
    w->total_loss = 0;
    w->loss_count = 0;

    return ok;
}

/* ============================================================= */
/* Message Handling                                              */
/* ============================================================= */

static bool send_heartbeat(sw_worker* w) {
    uint8_t* msg = sw_msg_alloc(SW_MSG_HEARTBEAT, sizeof(sw_heartbeat_payload));
    if (!msg) return false;

    sw_heartbeat_payload* hb = sw_msg_get_payload(msg);
    memset(hb, 0, sizeof(*hb));
    strncpy(hb->worker_id, w->config.worker_id, SW_MAX_WORKER_ID - 1);
    hb->local_step = w->local_step;
    hb->current_loss = w->current_loss;
    hb->samples_per_sec = w->samples_per_sec;

    bool ok = sw_send_msg(w->socket, msg);
    sw_msg_free(msg);

    if (ok) {
        w->last_heartbeat_sent = get_time_ms();
    }

    return ok;
}

static bool handle_message(sw_worker* w, uint8_t* msg) {
    sw_msg_header* hdr = sw_msg_get_header(msg);

    switch (hdr->type) {
        case SW_MSG_HEARTBEAT_ACK:
            /* Just confirms connection is alive */
            return true;

        case SW_MSG_START:
            worker_log(w, SW_LOG_INFO, "Received START command");
            set_state(w, SW_WSTATE_TRAINING);
            return true;

        case SW_MSG_PAUSE:
            worker_log(w, SW_LOG_INFO, "Received PAUSE command");
            set_state(w, SW_WSTATE_PAUSED);
            return true;

        case SW_MSG_RESUME:
            worker_log(w, SW_LOG_INFO, "Received RESUME command");
            set_state(w, SW_WSTATE_TRAINING);
            return true;

        case SW_MSG_SHUTDOWN:
            worker_log(w, SW_LOG_INFO, "Received SHUTDOWN command");
            set_state(w, SW_WSTATE_FINISHED);
            w->should_stop = true;
            return true;

        case SW_MSG_GRADIENTS_ACK:
            worker_log(w, SW_LOG_DEBUG, "Gradients acknowledged");
            return true;

        default:
            worker_log(w, SW_LOG_WARN, "Unknown message type: 0x%02x", hdr->type);
            return true;
    }
}

static void process_pending_messages(sw_worker* w) {
    /* Process any pending messages (non-blocking) */
    while (sw_data_available(w->socket)) {
        uint8_t* msg = sw_recv_msg(w->socket, 0);
        if (msg) {
            handle_message(w, msg);
            sw_msg_free(msg);
        } else {
            break;
        }
    }
}

/* ============================================================= */
/* Training Loop                                                 */
/* ============================================================= */

void sw_worker_train(sw_worker* w) {
    if (!w->forward_fn || !w->backward_fn) {
        worker_log(w, SW_LOG_ERROR, "Forward/backward callbacks not set");
        set_state(w, SW_WSTATE_ERROR);
        return;
    }

    if (!w->next_batch_fn) {
        worker_log(w, SW_LOG_ERROR, "Data loader not set");
        set_state(w, SW_WSTATE_ERROR);
        return;
    }

    if (!sw_worker_is_connected(w)) {
        worker_log(w, SW_LOG_ERROR, "Not connected to coordinator");
        set_state(w, SW_WSTATE_ERROR);
        return;
    }

    w->running = true;
    w->should_stop = false;
    w->local_step = 0;
    w->accumulation_count = 0;
    zero_gradients(w);

    int accum_steps = w->assigned_accumulation_steps > 0 ?
                      w->assigned_accumulation_steps : 50;

    worker_log(w, SW_LOG_INFO, "Starting training loop (accum_steps=%d)", accum_steps);

    /* Wait for START signal */
    set_state(w, SW_WSTATE_WAITING);
    while (w->state == SW_WSTATE_WAITING && !w->should_stop) {
        process_pending_messages(w);

        /* Send heartbeat periodically */
        uint64_t now = get_time_ms();
        if (now - w->last_heartbeat_sent > 5000) {  /* Every 5 seconds */
            send_heartbeat(w);
        }

        sleep_ms(100);
    }

    if (w->should_stop) {
        w->running = false;
        return;
    }

    /* Reset data loader */
    if (w->reset_data_fn) {
        w->reset_data_fn(w->data_ctx);
    }

    /* Main training loop */
    w->step_start_time = get_time_ms();

    while (!w->should_stop) {
        /* Handle pause state */
        while (w->state == SW_WSTATE_PAUSED && !w->should_stop) {
            process_pending_messages(w);

            uint64_t now = get_time_ms();
            if (now - w->last_heartbeat_sent > 5000) {
                send_heartbeat(w);
            }

            sleep_ms(100);
        }

        if (w->should_stop) break;

        /* Get next batch */
        void* batch = w->next_batch_fn(w->data_ctx);
        if (!batch) {
            /* Epoch complete - reset and continue */
            worker_log(w, SW_LOG_INFO, "Epoch complete, resetting data loader");
            if (w->reset_data_fn) {
                w->reset_data_fn(w->data_ctx);
            }
            batch = w->next_batch_fn(w->data_ctx);
            if (!batch) {
                worker_log(w, SW_LOG_ERROR, "No data available");
                break;
            }
        }

        /* Forward pass */
        float loss = w->forward_fn(w->weights, w->num_weights, batch, w->forward_ctx);
        w->current_loss = loss;
        w->total_loss += loss;
        w->loss_count++;

        /* Backward pass */
        w->backward_fn(w->weights, w->num_weights, batch, w->backward_ctx);

        /* Accumulate gradients */
        accumulate_gradients(w);

        /* Free batch */
        if (w->free_batch_fn) {
            w->free_batch_fn(batch, w->data_ctx);
        }

        w->local_step++;

        /* Calculate throughput */
        uint64_t now = get_time_ms();
        float elapsed_sec = (now - w->step_start_time) / 1000.0f;
        if (elapsed_sec > 0) {
            w->samples_per_sec = (w->local_step * w->assigned_batch_size) / elapsed_sec;
        }

        /* Step complete callback */
        if (w->on_step_complete) {
            w->on_step_complete(w, w->local_step, loss, w->on_step_complete_ctx);
        }

        /* Send gradients after accumulation steps */
        if (w->accumulation_count >= accum_steps) {
            if (!send_gradients(w)) {
                worker_log(w, SW_LOG_ERROR, "Failed to send gradients");
                /* Try to reconnect */
                if (!sw_worker_connect(w)) {
                    set_state(w, SW_WSTATE_ERROR);
                    break;
                }
                /* Request fresh weights after reconnect */
                sw_worker_fetch_weights(w);
            }
        }

        /* Process incoming messages */
        process_pending_messages(w);

        /* Send heartbeat periodically */
        if (now - w->last_heartbeat_sent > 5000) {
            send_heartbeat(w);
        }
    }

    /* Send any remaining gradients */
    if (w->accumulation_count > 0) {
        send_gradients(w);
    }

    w->running = false;
    worker_log(w, SW_LOG_INFO, "Training loop ended (steps=%d)", w->local_step);
}

void sw_worker_stop(sw_worker* w) {
    w->should_stop = true;
}

/* ============================================================= */
/* State Accessors                                               */
/* ============================================================= */

const char* sw_worker_get_id(sw_worker* w) {
    return w->config.worker_id;
}

int sw_worker_get_step(sw_worker* w) {
    return w->local_step;
}

float sw_worker_get_loss(sw_worker* w) {
    return w->current_loss;
}

float sw_worker_get_throughput(sw_worker* w) {
    return w->samples_per_sec;
}

int sw_worker_get_shard(sw_worker* w) {
    return w->assigned_shard;
}

int sw_worker_get_batch_size(sw_worker* w) {
    return w->assigned_batch_size;
}

/* ============================================================= */
/* Callbacks                                                     */
/* ============================================================= */

void sw_worker_on_state_change(sw_worker* w, sw_on_state_change cb, void* ctx) {
    w->on_state_change = cb;
    w->on_state_change_ctx = ctx;
}

void sw_worker_on_weights_received(sw_worker* w, sw_on_weights_received cb, void* ctx) {
    w->on_weights_received = cb;
    w->on_weights_received_ctx = ctx;
}

void sw_worker_on_step_complete(sw_worker* w, sw_on_step_complete cb, void* ctx) {
    w->on_step_complete = cb;
    w->on_step_complete_ctx = ctx;
}

/* ============================================================= */
/* Logging                                                       */
/* ============================================================= */

void sw_worker_set_log_level(sw_worker* w, sw_log_level level) {
    w->log_level = level;
}

void sw_worker_set_log_file(sw_worker* w, const char* path) {
    if (w->log_file && w->log_file != stdout) {
        fclose(w->log_file);
    }

    if (path) {
        w->log_file = fopen(path, "a");
    } else {
        w->log_file = NULL;
    }
}
