/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Swarm Coordinator - Main entry point.
 *
 * Usage:
 *   swarm_coordinator --config <path>
 *   swarm_coordinator --port 7100 --min-workers 2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#include "coordinator.h"
#include "config.h"
#include "../tensor/tensor.h"

/* Global coordinator for signal handling */
static sw_coordinator* g_coordinator = NULL;

static void print_usage(const char* prog) {
    printf("Foundry Swarm Coordinator\n");
    printf("\n");
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  --config <path>      Load configuration from file\n");
    printf("  --port <port>        Port to listen on (default: 7100)\n");
    printf("  --min-workers <n>    Minimum workers before training starts\n");
    printf("  --lr <rate>          Learning rate (default: 1e-4)\n");
    printf("  --steps <n>          Total training steps\n");
    printf("  --checkpoint <dir>   Checkpoint directory\n");
    printf("  --weights <path>     Load initial weights from safetensors file\n");
    printf("  --help               Show this help\n");
}

static void signal_handler(int sig) {
    (void)sig;
    printf("\nReceived signal, shutting down...\n");
    if (g_coordinator) {
        sw_coordinator_stop(g_coordinator);
    }
}

/* Callbacks */
static void on_worker_join(sw_coordinator* coord, sw_worker_info* worker, void* ctx) {
    (void)ctx;
    printf("[+] Worker joined: %s (profile=%d, shard=%d)\n",
           worker->id, worker->profile, worker->shard_index);
    printf("    Active workers: %d\n", sw_coordinator_num_active_workers(coord));
}

static void on_worker_leave(sw_coordinator* coord, sw_worker_info* worker, void* ctx) {
    (void)ctx;
    printf("[-] Worker left: %s\n", worker->id);
    printf("    Active workers: %d\n", sw_coordinator_num_active_workers(coord));
}

static void on_step(sw_coordinator* coord, int step, float loss, void* ctx) {
    (void)ctx;
    if (step % 100 == 0 || step == 1) {
        printf("[Step %d] loss=%.4f lr=%.2e throughput=%.1f samples/sec\n",
               step, loss, sw_coordinator_get_lr(coord),
               sw_coordinator_throughput(coord));
    }
}

static void on_checkpoint(sw_coordinator* coord, const char* path, void* ctx) {
    (void)coord;
    (void)ctx;
    printf("[Checkpoint] Saved to: %s\n", path);
}

/* Create dummy weights for testing (replace with actual model loading) */
static void create_dummy_weights(tm_tensor*** weights, int* num_weights) {
    /* Simple test model: 2 layers */
    *num_weights = 4;
    *weights = malloc(*num_weights * sizeof(tm_tensor*));

    int64_t shape1[] = {1024, 1024};
    int64_t shape2[] = {1024};
    int64_t shape3[] = {1024, 1024};
    int64_t shape4[] = {1024};

    (*weights)[0] = tm_randn(shape1, 2, TM_F32);  /* layer1.weight */
    (*weights)[1] = tm_zeros(shape2, 1, TM_F32); /* layer1.bias */
    (*weights)[2] = tm_randn(shape3, 2, TM_F32);  /* layer2.weight */
    (*weights)[3] = tm_zeros(shape4, 1, TM_F32); /* layer2.bias */

    printf("Created dummy weights: %d tensors\n", *num_weights);
    for (int i = 0; i < *num_weights; i++) {
        printf("  [%d] shape=[", i);
        for (int d = 0; d < (*weights)[i]->ndim; d++) {
            printf("%lld%s", (long long)(*weights)[i]->shape[d],
                   d < (*weights)[i]->ndim - 1 ? ", " : "");
        }
        printf("] numel=%lld\n", (long long)(*weights)[i]->numel);
    }
}

int main(int argc, char** argv) {
    sw_coordinator_config config = sw_coordinator_default_config();
    const char* config_path = NULL;
    const char* weights_path = NULL;

    /* Parse command line */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            config.port = (uint16_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-workers") == 0 && i + 1 < argc) {
            config.min_workers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            config.learning_rate = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            config.total_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
            strncpy(config.checkpoint_dir, argv[++i], SW_MAX_PATH - 1);
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
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

        config.port = (uint16_t)sw_config_get_int(cfg, "network", "port", config.port);

        config.learning_rate = sw_config_get_float(cfg, "training", "learning_rate", config.learning_rate);
        config.lr_min = sw_config_get_float(cfg, "training", "lr_min", config.lr_min);
        config.lr_warmup_steps = sw_config_get_int(cfg, "training", "lr_warmup_steps", config.lr_warmup_steps);
        config.total_steps = sw_config_get_int(cfg, "training", "total_steps", config.total_steps);

        config.beta1 = sw_config_get_float(cfg, "optimizer", "beta1", config.beta1);
        config.beta2 = sw_config_get_float(cfg, "optimizer", "beta2", config.beta2);
        config.weight_decay = sw_config_get_float(cfg, "optimizer", "weight_decay", config.weight_decay);

        config.min_workers = sw_config_get_int(cfg, "sync", "min_workers", config.min_workers);
        config.sync_every_n_grads = sw_config_get_int(cfg, "sync", "sync_every_n_grads", config.sync_every_n_grads);

        const char* ckpt_dir = sw_config_get_str(cfg, "checkpoint", "dir", NULL);
        if (ckpt_dir) strncpy(config.checkpoint_dir, ckpt_dir, SW_MAX_PATH - 1);
        config.checkpoint_every_steps = sw_config_get_int(cfg, "checkpoint", "every_steps", config.checkpoint_every_steps);

        config.heartbeat_timeout_sec = sw_config_get_int(cfg, "fault_tolerance", "heartbeat_timeout_sec", config.heartbeat_timeout_sec);
        config.allow_rejoin = sw_config_get_bool(cfg, "fault_tolerance", "allow_rejoin", config.allow_rejoin);

        const char* comp = sw_config_get_str(cfg, "compression", "type", "topk");
        if (strcmp(comp, "none") == 0) config.compression = SW_COMPRESS_NONE;
        else if (strcmp(comp, "fp16") == 0) config.compression = SW_COMPRESS_FP16;
        else if (strcmp(comp, "topk") == 0) config.compression = SW_COMPRESS_TOPK;
        else if (strcmp(comp, "qsgd") == 0) config.compression = SW_COMPRESS_QSGD;

        config.topk_ratio = sw_config_get_float(cfg, "compression", "topk_ratio", config.topk_ratio);

        sw_config_free(cfg);
    }

    /* Load or create weights */
    tm_tensor** weights = NULL;
    int num_weights = 0;

    if (weights_path) {
        /* TODO: Load from safetensors file */
        fprintf(stderr, "Safetensors loading not yet implemented\n");
        fprintf(stderr, "Creating dummy weights for testing...\n");
        create_dummy_weights(&weights, &num_weights);
    } else {
        printf("No weights file specified, creating dummy weights for testing...\n");
        create_dummy_weights(&weights, &num_weights);
    }

    /* Create coordinator */
    sw_coordinator* coord = sw_coordinator_new(weights, num_weights, &config);
    if (!coord) {
        fprintf(stderr, "Failed to create coordinator\n");
        return 1;
    }

    g_coordinator = coord;

    /* Set up signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Set callbacks */
    sw_coordinator_on_worker_join(coord, on_worker_join, NULL);
    sw_coordinator_on_worker_leave(coord, on_worker_leave, NULL);
    sw_coordinator_on_step(coord, on_step, NULL);
    sw_coordinator_on_checkpoint(coord, on_checkpoint, NULL);

    /* Print config */
    printf("\n");
    printf("=== Foundry Swarm Coordinator ===\n");
    printf("Port:           %d\n", config.port);
    printf("Min workers:    %d\n", config.min_workers);
    printf("Learning rate:  %.2e\n", config.learning_rate);
    printf("Total steps:    %d\n", config.total_steps);
    printf("Checkpoint dir: %s\n", config.checkpoint_dir[0] ? config.checkpoint_dir : "(none)");
    printf("Compression:    %s\n",
           config.compression == SW_COMPRESS_NONE ? "none" :
           config.compression == SW_COMPRESS_FP16 ? "fp16" :
           config.compression == SW_COMPRESS_TOPK ? "topk" : "qsgd");
    printf("\n");
    printf("Waiting for workers to connect...\n");
    printf("\n");

    /* Run coordinator (blocking) */
    sw_coordinator_run(coord);

    /* Cleanup */
    printf("Coordinator stopped.\n");
    sw_coordinator_free(coord);

    for (int i = 0; i < num_weights; i++) {
        tm_free(weights[i]);
    }
    free(weights);

    return 0;
}
