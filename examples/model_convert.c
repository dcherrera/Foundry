/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * model_convert.c — Convert HuggingFace models to BootAI .btw/.btv format
 *
 * Pure C, no Python dependencies. Supports any Llama-family model:
 * Llama 2/3, Qwen 2/2.5/3, Mistral, Phi-3, Gemma 2.
 *
 * Usage:
 *   ./model_convert --model-dir path/to/hf/model/ --output model.btw
 *   ./model_convert --weights model.safetensors --config config.json \
 *                   --tokenizer tokenizer.json --output model.btw
 */

#include "../convert/convert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --model-dir DIR    HuggingFace model directory (auto-detects files)\n"
        "  --weights PATH     Path to model.safetensors\n"
        "  --config PATH      Path to config.json\n"
        "  --tokenizer PATH   Path to tokenizer.json\n"
        "  --vocab PATH       Path to vocab.json (alternative to tokenizer.json)\n"
        "  --merges PATH      Path to merges.txt (used with --vocab)\n"
        "  --output PATH      Output .btw file path (required)\n"
        "  --vocab-only       Export only vocab (.btv), skip weights\n"
        "  --help             Show this help\n"
        "\n"
        "Examples:\n"
        "  %s --model-dir ../models/base/Qwen2.5-1.5B/ --output model.btw\n"
        "  %s --weights model.safetensors --config config.json --output model.btw\n"
        , prog, prog, prog);
}

/* Check if file exists */
static int file_exists(const char *path) {
    FILE *f = fopen(path, "r");
    if (f) { fclose(f); return 1; }
    return 0;
}

int main(int argc, char **argv) {
    const char *model_dir  = NULL;
    const char *weights    = NULL;
    const char *config     = NULL;
    const char *tokenizer  = NULL;
    const char *vocab      = NULL;
    const char *merges     = NULL;
    const char *output     = NULL;
    int vocab_only = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc)
            model_dir = argv[++i];
        else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc)
            weights = argv[++i];
        else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc)
            config = argv[++i];
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc)
            tokenizer = argv[++i];
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc)
            vocab = argv[++i];
        else if (strcmp(argv[i], "--merges") == 0 && i + 1 < argc)
            merges = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output = argv[++i];
        else if (strcmp(argv[i], "--vocab-only") == 0)
            vocab_only = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!output) {
        fprintf(stderr, "ERROR: --output is required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Auto-detect files from model directory */
    char weights_buf[512], config_buf[512], tokenizer_buf[512];
    char vocab_buf[512], merges_buf[512];

    if (model_dir) {
        size_t dlen = strlen(model_dir);
        /* Strip trailing slash */
        char dir[512];
        snprintf(dir, sizeof(dir), "%s", model_dir);
        if (dlen > 0 && dir[dlen - 1] == '/') dir[dlen - 1] = '\0';

        if (!weights) {
            snprintf(weights_buf, sizeof(weights_buf), "%s/model.safetensors", dir);
            if (file_exists(weights_buf)) weights = weights_buf;
        }
        if (!config) {
            snprintf(config_buf, sizeof(config_buf), "%s/config.json", dir);
            if (file_exists(config_buf)) config = config_buf;
        }
        if (!tokenizer) {
            snprintf(tokenizer_buf, sizeof(tokenizer_buf), "%s/tokenizer.json", dir);
            if (file_exists(tokenizer_buf)) tokenizer = tokenizer_buf;
        }
        if (!vocab) {
            snprintf(vocab_buf, sizeof(vocab_buf), "%s/vocab.json", dir);
            if (file_exists(vocab_buf)) vocab = vocab_buf;
        }
        if (!merges) {
            snprintf(merges_buf, sizeof(merges_buf), "%s/merges.txt", dir);
            if (file_exists(merges_buf)) merges = merges_buf;
        }

        fprintf(stderr, "Model directory: %s\n", dir);
        if (weights)   fprintf(stderr, "  weights:   %s\n", weights);
        if (config)    fprintf(stderr, "  config:    %s\n", config);
        if (tokenizer) fprintf(stderr, "  tokenizer: %s\n", tokenizer);
        if (vocab)     fprintf(stderr, "  vocab:     %s\n", vocab);
        if (merges)    fprintf(stderr, "  merges:    %s\n", merges);
        fprintf(stderr, "\n");
    }

    /* Validate required files */
    if (!vocab_only && !weights) {
        fprintf(stderr, "ERROR: --weights or --model-dir with model.safetensors required\n");
        return 1;
    }
    if (!vocab_only && !config) {
        fprintf(stderr, "ERROR: --config or --model-dir with config.json required\n");
        return 1;
    }
    if (!tokenizer && !vocab) {
        fprintf(stderr, "ERROR: --tokenizer or --vocab required for vocab export\n");
        return 1;
    }

    /* Parse config */
    fc_model_config *cfg = NULL;
    if (config) {
        cfg = fc_parse_config(config);
        if (!cfg) {
            fprintf(stderr, "ERROR: failed to parse %s\n", config);
            return 1;
        }
    }

    /* Export weights */
    if (!vocab_only) {
        if (!fc_export_weights(weights, cfg, output)) {
            fprintf(stderr, "ERROR: weight export failed\n");
            free(cfg);
            return 1;
        }
    }

    /* Export vocab (.btv) — same directory as .btw, named "vocab.btv" */
    char btv_path[512];
    snprintf(btv_path, sizeof(btv_path), "%s", output);
    /* Find last slash to get directory */
    char *last_slash = strrchr(btv_path, '/');
    if (last_slash) {
        strcpy(last_slash + 1, "vocab.btv");
    } else {
        strcpy(btv_path, "vocab.btv");
    }

    if (!fc_export_vocab(tokenizer, merges, vocab, btv_path)) {
        fprintf(stderr, "WARNING: vocab export failed\n");
    }

    free(cfg);
    fprintf(stderr, "\nAll done!\n");
    return 0;
}
