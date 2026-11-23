/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rfa_train — CLI training tool for RFA hybrid model LoRA fine-tuning
 *
 * Trains Qwen GQA blocks to work with frozen RWKV-7 blocks.
 * Uses pre-tokenized binary data (format: docs/rfa-train.md).
 *
 * Supports parallel training (--parallel N) and INT8 quantized base weights
 * (--quantize) for faster training with lower memory usage.
 *
 * Usage:
 *   ./examples/rfa_train \
 *     --model ../models/base/rwkv-qwen-hybrid-1.5B.safetensors \
 *     --data ../models/datasets/oasst1_qwen_tokens.bin \
 *     --output ../models/loras/rfa-qwen-lora/ \
 *     --lora-rank 32 --lora-alpha 64 \
 *     --lr 2e-5 --epochs 3 --max-seq-len 256 --log-every 10 \
 *     --parallel 6 --quantize
 */

#include "rwkvx/rfa_train.h"
#include "rwkvx/rfa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  --model PATH       Model safetensors file\n");
    fprintf(stderr, "  --data PATH        Pre-tokenized binary data file\n");
    fprintf(stderr, "  --output DIR       Output directory for LoRA checkpoints\n");
    fprintf(stderr, "\nOptional:\n");
    fprintf(stderr, "  --lora-rank N      LoRA rank (default: 32)\n");
    fprintf(stderr, "  --lora-alpha F     LoRA alpha (default: 64)\n");
    fprintf(stderr, "  --lr F             Learning rate (default: 2e-5)\n");
    fprintf(stderr, "  --epochs N         Number of epochs (default: 3)\n");
    fprintf(stderr, "  --max-seq-len N    Max sequence length (default: 256)\n");
    fprintf(stderr, "  --warmup N         Warmup steps (default: 100)\n");
    fprintf(stderr, "  --log-every N      Log every N steps (default: 10)\n");
    fprintf(stderr, "  --save-every N     Save checkpoint every N steps (default: 1000)\n");
    fprintf(stderr, "  --grad-clip F      Gradient clipping norm (default: 1.0)\n");
    fprintf(stderr, "  --max-cache N      Max KV cache per Qwen block (default: 2048)\n");
    fprintf(stderr, "  --parallel N       Number of training threads (default: 1)\n");
    fprintf(stderr, "  --quantize         Use INT8 quantized base weights\n");
}

int main(int argc, char **argv) {
    rfa_train_config config = rfa_train_config_defaults();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            config.data_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (strcmp(argv[i], "--lora-rank") == 0 && i + 1 < argc) {
            config.lora_rank = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lora-alpha") == 0 && i + 1 < argc) {
            config.lora_alpha = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            config.lr = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            config.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-seq-len") == 0 && i + 1 < argc) {
            config.max_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            config.warmup_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--log-every") == 0 && i + 1 < argc) {
            config.log_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--save-every") == 0 && i + 1 < argc) {
            config.save_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--grad-clip") == 0 && i + 1 < argc) {
            config.grad_clip = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--max-cache") == 0 && i + 1 < argc) {
            config.max_cache = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--parallel") == 0 && i + 1 < argc) {
            config.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--quantize") == 0) {
            config.quantize = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!config.model_path || !config.data_path || !config.output_dir) {
        fprintf(stderr, "Error: --model, --data, and --output are required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Load model (optionally quantized) */
    fprintf(stderr, "Loading model: %s%s\n", config.model_path,
            config.quantize ? " (INT8)" : "");
    rfa_model *model = config.quantize ?
        rfa_model_load_q8(config.model_path) :
        rfa_model_load(config.model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    rfa_config mcfg = rfa_model_config(model);
    fprintf(stderr, "Model: %dd, %d layers (%d RWKV, %d Qwen), vocab=%d\n",
            mcfg.n_embd, mcfg.n_layers, mcfg.n_rwkv_layers,
            mcfg.n_qwen_layers, mcfg.vocab_size);

    /* Load pre-tokenized data */
    int n_examples;
    rfa_train_example *examples = rfa_load_pretokenized(
        config.data_path, config.max_seq_len, &n_examples);
    if (!examples || n_examples == 0) {
        fprintf(stderr, "Failed to load training data\n");
        rfa_model_free(model);
        return 1;
    }

    /* Train */
    if (config.n_threads > 1) {
        fprintf(stderr, "Parallel training: %d threads\n", config.n_threads);
        rfa_train_parallel(model, &config, examples, n_examples,
                           config.n_threads);
    } else {
        rfa_train_loop(model, &config, examples, n_examples);
    }

    /* Cleanup */
    rfa_train_examples_free(examples, n_examples);
    rfa_model_free(model);

    fprintf(stderr, "Done.\n");
    return 0;
}
