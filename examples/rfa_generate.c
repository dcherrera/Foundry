/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rfa_generate.c — CLI text generation for RWKV-7 + Qwen hybrid models
 *
 * Usage:
 *   ./examples/rfa_generate \
 *     --model ../models/base/rwkv-qwen-hybrid-1.5B.safetensors \
 *     --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
 *     --prompt "Hello world" \
 *     --max-tokens 64 \
 *     --temperature 1.0 \
 *     --top-p 0.7
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rwkvx/rfa.h"
#include "rwkvx/rwkvx.h"  /* for rx_tokenizer, rx_sample, rx_sampling_params */

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  --model PATH        Merged hybrid safetensors file\n");
    fprintf(stderr, "  --vocab PATH        RWKV vocab file\n");
    fprintf(stderr, "  --prompt TEXT        Input prompt\n");
    fprintf(stderr, "  --max-tokens N      Maximum tokens to generate (default 64)\n");
    fprintf(stderr, "  --max-cache N       KV cache size per Qwen layer (default 2048)\n");
    fprintf(stderr, "  --temperature F     Sampling temperature (default 1.0, 0=greedy)\n");
    fprintf(stderr, "  --top-p F           Nucleus sampling threshold (default 0.7)\n");
    fprintf(stderr, "  --top-k N           Top-k filtering (default 0=off)\n");
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *vocab_path = NULL;
    const char *prompt = "Hello";
    int max_tokens = 64;
    int max_cache = 2048;
    float temperature = 1.0f;
    float top_p = 0.7f;
    int top_k = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--vocab") == 0 && i+1 < argc) vocab_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i+1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i+1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-cache") == 0 && i+1 < argc) max_cache = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i+1 < argc) temperature = strtof(argv[++i], NULL);
        else if (strcmp(argv[i], "--top-p") == 0 && i+1 < argc) top_p = strtof(argv[++i], NULL);
        else if (strcmp(argv[i], "--top-k") == 0 && i+1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (!model_path || !vocab_path) {
        fprintf(stderr, "Error: --model and --vocab are required\n");
        usage(argv[0]);
        return 1;
    }

    /* Load model */
    printf("Loading model: %s\n", model_path);
    rfa_model *model = rfa_model_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    rfa_config cfg = rfa_model_config(model);
    printf("Config: %dd, %d layers (%d RWKV + %d Qwen), vocab=%d\n",
           cfg.n_embd, cfg.n_layers, cfg.n_rwkv_layers, cfg.n_qwen_layers,
           cfg.vocab_size);

    /* Load tokenizer */
    rx_tokenizer *tok = rx_tokenizer_new(vocab_path);
    if (!tok) {
        fprintf(stderr, "Failed to load tokenizer\n");
        rfa_model_free(model);
        return 1;
    }

    /* Create state */
    rfa_state *state = rfa_state_new(model, max_cache);
    if (!state) {
        fprintf(stderr, "Failed to create state\n");
        rx_tokenizer_free(tok);
        rfa_model_free(model);
        return 1;
    }

    /* Tokenize prompt */
    uint32_t tokens[8192];
    int n_prompt = rx_tokenizer_encode(tok, prompt, tokens, 8192);
    printf("Prompt: \"%s\" (%d tokens)\n\n", prompt, n_prompt);

    /* Sampling params */
    rx_sampling_params params = rx_sampling_defaults();
    params.temperature = temperature;
    params.top_p = top_p;
    params.top_k = top_k;

    /* Prefill: process prompt tokens */
    struct timespec t_start, t_prefill, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    tm_tensor *logits = NULL;
    for (int i = 0; i < n_prompt; i++) {
        if (logits) tm_free(logits);
        logits = rfa_forward(model, tokens[i], state);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_prefill);

    /* Print prompt */
    char *prompt_text = rx_tokenizer_decode(tok, tokens, n_prompt);
    if (prompt_text) {
        printf("%s", prompt_text);
        free(prompt_text);
    }
    fflush(stdout);

    /* Generate */
    uint32_t *counts = (uint32_t*)calloc(cfg.vocab_size, sizeof(uint32_t));
    int generated = 0;

    for (int i = 0; i < max_tokens; i++) {
        uint32_t next = rx_sample(logits, &params, counts, cfg.vocab_size);
        tm_free(logits);
        logits = NULL;

        if (next == 0) break; /* EOS */
        counts[next]++;

        /* Decode and print token */
        char *tok_str = rx_tokenizer_decode(tok, &next, 1);
        if (tok_str) {
            printf("%s", tok_str);
            fflush(stdout);
            free(tok_str);
        }

        /* Forward next token */
        logits = rfa_forward(model, next, state);
        generated++;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    if (logits) tm_free(logits);

    /* Stats */
    double prefill_s = (t_prefill.tv_sec - t_start.tv_sec) +
                       (t_prefill.tv_nsec - t_start.tv_nsec) / 1e9;
    double decode_s = (t_end.tv_sec - t_prefill.tv_sec) +
                      (t_end.tv_nsec - t_prefill.tv_nsec) / 1e9;

    printf("\n\n--- Stats ---\n");
    printf("Prefill: %d tokens in %.2fs (%.1f tok/s)\n",
           n_prompt, prefill_s, n_prompt / prefill_s);
    printf("Decode:  %d tokens in %.2fs (%.1f tok/s)\n",
           generated, decode_s, generated / decode_s);

    /* Cleanup */
    free(counts);
    rfa_state_free(state);
    rx_tokenizer_free(tok);
    rfa_model_free(model);

    return 0;
}
