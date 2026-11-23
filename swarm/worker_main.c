/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Swarm Worker - Main entry point.
 *
 * Usage:
 *   swarm_worker --config <path>
 *   swarm_worker --host coordinator --port 7100
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <math.h>

#include "worker.h"
#include "config.h"
#include "../tensor/tensor.h"

/* Global worker for signal handling */
static sw_worker* g_worker = NULL;

static void print_usage(const char* prog) {
    printf("Foundry Swarm Worker\n");
    printf("\n");
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  --config <path>      Load configuration from file\n");
    printf("  --host <hostname>    Coordinator hostname (default: localhost)\n");
    printf("  --port <port>        Coordinator port (default: 7100)\n");
    printf("  --batch-size <n>     Batch size (0 = auto)\n");
    printf("  --threads <n>        Number of threads (0 = auto)\n");
    printf("  --data <path>        Path to training data\n");
    printf("  --shard <n>          Data shard index (-1 = auto)\n");
    printf("  --id <id>            Worker ID (default: auto-generated)\n");
    printf("  --help               Show this help\n");
}

static void signal_handler(int sig) {
    (void)sig;
    printf("\nReceived signal, stopping worker...\n");
    if (g_worker) {
        sw_worker_stop(g_worker);
    }
}

/* Callbacks */
static void on_state_change(sw_worker* w, sw_wstate old_state,
                            sw_wstate new_state, void* ctx) {
    (void)ctx;
    const char* state_names[] = {
        "DISCONNECTED", "CONNECTING", "REGISTERED",
        "WAITING", "TRAINING", "PAUSED", "FINISHED", "ERROR"
    };
    printf("[%s] State: %s -> %s\n",
           sw_worker_get_id(w), state_names[old_state], state_names[new_state]);
}

static void on_weights_received(sw_worker* w, void* ctx) {
    (void)ctx;
    printf("[%s] Received weights from coordinator\n", sw_worker_get_id(w));
}

static void on_step_complete(sw_worker* w, int step, float loss, void* ctx) {
    (void)ctx;
    if (step % 100 == 0 || step == 1) {
        printf("[%s] Step %d | loss=%.4f | throughput=%.1f samples/sec\n",
               sw_worker_get_id(w), step, loss, sw_worker_get_throughput(w));
    }
}

/* ============================================================= */
/* Dummy Model for Testing                                       */
/* ============================================================= */

typedef struct {
    tm_tensor** weights;
    int num_weights;
    tm_tensor* loss_tensor;
} dummy_model;

typedef struct {
    tm_tensor* input;
    tm_tensor* target;
} dummy_batch;

typedef struct {
    int batch_size;
    int input_size;
    int current_batch;
    int max_batches;
} dummy_data_loader;

/* Forward pass: simple 2-layer MLP */
static float dummy_forward(tm_tensor** weights, int num_weights,
                           void* batch_data, void* ctx) {
    (void)num_weights;
    dummy_model* model = (dummy_model*)ctx;
    dummy_batch* batch = (dummy_batch*)batch_data;

    /* Simple forward: x @ W1 -> relu -> @ W2 -> mse_loss */
    tm_tensor* h1 = tm_matmul(batch->input, weights[0]);
    tm_tensor* h1_bias = tm_add(h1, weights[1]);
    tm_tensor* h1_relu = tm_relu(h1_bias);

    tm_tensor* out = tm_matmul(h1_relu, weights[2]);
    tm_tensor* out_bias = tm_add(out, weights[3]);

    /* MSE loss */
    tm_tensor* loss = tm_mse_loss(out_bias, batch->target);

    float loss_val = tm_get_f32(loss, 0);

    /* Store loss tensor for backward */
    if (model->loss_tensor) tm_free(model->loss_tensor);
    model->loss_tensor = loss;

    /* Free intermediates */
    tm_free(h1);
    tm_free(h1_bias);
    tm_free(h1_relu);
    tm_free(out);
    tm_free(out_bias);

    return loss_val;
}

/* Backward pass */
static void dummy_backward(tm_tensor** weights, int num_weights,
                           void* batch_data, void* ctx) {
    (void)weights;
    (void)num_weights;
    (void)batch_data;
    dummy_model* model = (dummy_model*)ctx;

    if (model->loss_tensor) {
        tm_backward(model->loss_tensor);
    }
}

/* Data loader */
static void* dummy_next_batch(void* ctx) {
    dummy_data_loader* loader = (dummy_data_loader*)ctx;

    if (loader->current_batch >= loader->max_batches) {
        return NULL;  /* Epoch complete */
    }

    /* Create random batch */
    dummy_batch* batch = malloc(sizeof(dummy_batch));

    int64_t input_shape[] = {loader->batch_size, loader->input_size};
    int64_t target_shape[] = {loader->batch_size, loader->input_size};

    batch->input = tm_randn(input_shape, 2, TM_F32);
    batch->target = tm_randn(target_shape, 2, TM_F32);

    loader->current_batch++;
    return batch;
}

static void dummy_reset_data(void* ctx) {
    dummy_data_loader* loader = (dummy_data_loader*)ctx;
    loader->current_batch = 0;
}

static void dummy_free_batch(void* batch_data, void* ctx) {
    (void)ctx;
    dummy_batch* batch = (dummy_batch*)batch_data;
    if (batch) {
        tm_free(batch->input);
        tm_free(batch->target);
        free(batch);
    }
}

/* Create dummy weights */
static void create_dummy_weights(tm_tensor*** weights, int* num_weights, int size) {
    *num_weights = 4;
    *weights = malloc(*num_weights * sizeof(tm_tensor*));

    int64_t shape_w1[] = {size, size};
    int64_t shape_b1[] = {size};
    int64_t shape_w2[] = {size, size};
    int64_t shape_b2[] = {size};

    /* Xavier initialization */
    float scale = sqrtf(2.0f / size);

    (*weights)[0] = tm_randn(shape_w1, 2, TM_F32);
    tm_scale_((*weights)[0], scale);
    tm_set_requires_grad((*weights)[0], true);

    (*weights)[1] = tm_zeros(shape_b1, 1, TM_F32);
    tm_set_requires_grad((*weights)[1], true);

    (*weights)[2] = tm_randn(shape_w2, 2, TM_F32);
    tm_scale_((*weights)[2], scale);
    tm_set_requires_grad((*weights)[2], true);

    (*weights)[3] = tm_zeros(shape_b2, 1, TM_F32);
    tm_set_requires_grad((*weights)[3], true);
}

int main(int argc, char** argv) {
    sw_worker_config config = sw_worker_default_config();
    const char* config_path = NULL;
    int model_size = 256;  /* Hidden size for dummy model */

    /* Parse command line */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            strncpy(config.coordinator_host, argv[++i], 255);
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            config.coordinator_port = (uint16_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            config.batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            config.num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            strncpy(config.data_path, argv[++i], SW_MAX_PATH - 1);
        } else if (strcmp(argv[i], "--shard") == 0 && i + 1 < argc) {
            config.shard_index = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--id") == 0 && i + 1 < argc) {
            strncpy(config.worker_id, argv[++i], SW_MAX_WORKER_ID - 1);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Load config file if specified */
    if (config_path) {
        sw_config* cfg = sw_config_load(config_path);
        if (!cfg) {
            fprintf(stderr, "Failed to load config: %s\n", config_path);
            return 1;
        }

        const char* host = sw_config_get_str(cfg, "network", "coordinator_host", NULL);
        if (host) strncpy(config.coordinator_host, host, 255);
        config.coordinator_port = (uint16_t)sw_config_get_int(cfg, "network", "coordinator_port", config.coordinator_port);
        config.reconnect_attempts = sw_config_get_int(cfg, "network", "reconnect_attempts", config.reconnect_attempts);
        config.reconnect_delay_sec = sw_config_get_int(cfg, "network", "reconnect_delay_sec", config.reconnect_delay_sec);

        config.num_threads = sw_config_get_int(cfg, "hardware", "num_threads", config.num_threads);
        config.max_memory_mb = sw_config_get_int(cfg, "hardware", "max_memory_mb", config.max_memory_mb);

        config.batch_size = sw_config_get_int(cfg, "training", "batch_size", config.batch_size);
        config.accumulation_steps = sw_config_get_int(cfg, "training", "accumulation_steps", config.accumulation_steps);

        const char* data = sw_config_get_str(cfg, "data", "path", NULL);
        if (data) strncpy(config.data_path, data, SW_MAX_PATH - 1);
        config.shard_index = sw_config_get_int(cfg, "data", "shard_index", config.shard_index);

        const char* comp = sw_config_get_str(cfg, "compression", "type", "topk");
        if (strcmp(comp, "none") == 0) config.compression = SW_COMPRESS_NONE;
        else if (strcmp(comp, "fp16") == 0) config.compression = SW_COMPRESS_FP16;
        else if (strcmp(comp, "topk") == 0) config.compression = SW_COMPRESS_TOPK;
        else if (strcmp(comp, "qsgd") == 0) config.compression = SW_COMPRESS_QSGD;

        config.topk_ratio = sw_config_get_float(cfg, "compression", "topk_ratio", config.topk_ratio);

        const char* id = sw_config_get_str(cfg, "identity", "worker_id", NULL);
        if (id && id[0]) strncpy(config.worker_id, id, SW_MAX_WORKER_ID - 1);

        sw_config_free(cfg);
    }

    /* Create worker */
    sw_worker* worker = sw_worker_new(&config);
    if (!worker) {
        fprintf(stderr, "Failed to create worker\n");
        return 1;
    }

    g_worker = worker;

    /* Set up signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Set callbacks */
    sw_worker_on_state_change(worker, on_state_change, NULL);
    sw_worker_on_weights_received(worker, on_weights_received, NULL);
    sw_worker_on_step_complete(worker, on_step_complete, NULL);

    /* Print config */
    printf("\n");
    printf("=== Foundry Swarm Worker ===\n");
    printf("Worker ID:      %s\n", sw_worker_get_id(worker));
    printf("Coordinator:    %s:%d\n", config.coordinator_host, config.coordinator_port);
    printf("Batch size:     %s\n", config.batch_size ? "auto" : "from profile");
    printf("Compression:    %s\n",
           config.compression == SW_COMPRESS_NONE ? "none" :
           config.compression == SW_COMPRESS_FP16 ? "fp16" :
           config.compression == SW_COMPRESS_TOPK ? "topk" : "qsgd");
    printf("\n");

    /* Create dummy model for testing */
    printf("Creating dummy model for testing...\n");
    tm_tensor** weights = NULL;
    int num_weights = 0;
    create_dummy_weights(&weights, &num_weights, model_size);
    printf("  Model size: %d x %d\n", model_size, model_size);
    printf("  Weights: %d tensors\n", num_weights);

    dummy_model model = {
        .weights = weights,
        .num_weights = num_weights,
        .loss_tensor = NULL
    };

    int batch_size = config.batch_size > 0 ? config.batch_size : 32;
    dummy_data_loader loader = {
        .batch_size = batch_size,
        .input_size = model_size,
        .current_batch = 0,
        .max_batches = 1000000  /* Effectively infinite */
    };

    /* Set weights */
    sw_worker_set_weights(worker, weights, num_weights);

    /* Set training callbacks */
    sw_worker_set_forward(worker, dummy_forward, &model);
    sw_worker_set_backward(worker, dummy_backward, &model);
    sw_worker_set_data_loader(worker, dummy_next_batch, dummy_reset_data,
                               dummy_free_batch, &loader);

    /* Connect to coordinator */
    printf("Connecting to coordinator...\n");
    if (!sw_worker_connect(worker)) {
        fprintf(stderr, "Failed to connect to coordinator\n");
        sw_worker_free(worker);
        return 1;
    }

    /* Fetch initial weights */
    printf("Fetching weights from coordinator...\n");
    if (!sw_worker_fetch_weights(worker)) {
        fprintf(stderr, "Failed to fetch weights\n");
        sw_worker_free(worker);
        return 1;
    }

    /* Run training loop */
    printf("Starting training...\n\n");
    sw_worker_train(worker);

    /* Cleanup */
    printf("\nWorker stopped.\n");
    sw_worker_free(worker);

    if (model.loss_tensor) tm_free(model.loss_tensor);
    for (int i = 0; i < num_weights; i++) {
        tm_free(weights[i]);
    }
    free(weights);

    return 0;
}
