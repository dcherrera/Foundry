/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rwkvx_generate.c — CLI text generation with RWKV-X
 *
 * Usage:
 *   ./examples/rwkvx_generate \
 *     --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
 *     --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
 *     --prompt "Hello world" \
 *     --max-tokens 32
 */

#include "rwkvx/rwkvx.h"
#include "tensor/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --model PATH       Path to .safetensors model file (required)\n"
        "  --vocab PATH       Path to vocab file (required)\n"
        "  --prompt TEXT       Prompt text (default: \"Hello\")\n"
        "  --max-tokens N     Max tokens to generate (default: 64)\n"
        "  --temperature F    Sampling temperature, 0=greedy (default: 1.0)\n"
        "  --top-p F          Nucleus sampling threshold (default: 0.7)\n"
        "  --top-k N          Top-k filtering (default: 100)\n"
        "  --quantize         Quantize weights to INT8 (reduces memory, may affect quality)\n"
        "  --hybridmem-dir D  Enable HybridMem disk-backed memory at directory D\n"
        "  --cache-size N     KV cache size per attention layer (default: 2048)\n"
        "  --hm-top-k N       HybridMem retrieval count (default: 5)\n",
        prog);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *vocab_path = NULL;
    const char *prompt = "Hello";
    int max_tokens = 64;
    float temperature = 1.0f;
    float top_p = 0.7f;
    int top_k = 100;
    int quantize = 0;
    const char *hybridmem_dir = NULL;
    int cache_size = 0;  /* 0 = default */
    int hm_top_k = 5;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc)
            vocab_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            prompt = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
            temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc)
            top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc)
            top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--quantize") == 0)
            quantize = 1;
        else if (strcmp(argv[i], "--hybridmem-dir") == 0 && i + 1 < argc)
            hybridmem_dir = argv[++i];
        else if (strcmp(argv[i], "--cache-size") == 0 && i + 1 < argc)
            cache_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hm-top-k") == 0 && i + 1 < argc)
            hm_top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !vocab_path) {
        fprintf(stderr, "Error: --model and --vocab are required\n");
        print_usage(argv[0]);
        return 1;
    }

    srand((unsigned)time(NULL));

    /* Load tokenizer */
    fprintf(stderr, "Loading tokenizer from %s...\n", vocab_path);
    rx_tokenizer *tok = rx_tokenizer_new(vocab_path);
    if (!tok) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    /* Load model */
    fprintf(stderr, "Loading model from %s%s...\n", model_path,
            quantize ? " (INT8 quantized)" : "");
    rx_model *model = quantize ? rx_model_load_q8(model_path)
                               : rx_model_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        rx_tokenizer_free(tok);
        return 1;
    }

    rx_config cfg = rx_model_config(model);
    fprintf(stderr, "Model: n_embd=%d n_head=%d n_rwkv=%d n_attn=%d vocab=%d\n",
            cfg.n_embd, cfg.n_head, cfg.n_rwkv_layer, cfg.n_attn_layer, cfg.vocab_size);

    /* Create state (with optional HybridMem) */
    rx_state *state;
    if (hybridmem_dir) {
        rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
        hm_cfg.store_dir = hybridmem_dir;
        hm_cfg.max_cache_len = cache_size;
        hm_cfg.top_k = hm_top_k;
        state = rx_state_new_with_hybridmem(model, &hm_cfg);
    } else {
        state = rx_state_new(model);
    }
    if (!state) {
        fprintf(stderr, "Failed to create state\n");
        rx_model_free(model);
        rx_tokenizer_free(tok);
        return 1;
    }

    /* Encode prompt */
    uint32_t prompt_tokens[4096];
    int n_prompt = rx_tokenizer_encode(tok, prompt, prompt_tokens, 4096);
    fprintf(stderr, "Prompt: \"%s\" -> %d tokens\n", prompt, n_prompt);

    /* Print prompt */
    printf("%s", prompt);
    fflush(stdout);

    /* Sampling params */
    rx_sampling_params params = rx_sampling_defaults();
    params.temperature = temperature;
    params.top_p = top_p;
    params.top_k = top_k;

    /* Token frequency counts for penalties */
    uint32_t *counts = (uint32_t*)calloc(cfg.vocab_size, sizeof(uint32_t));

    /* Prefill: process prompt tokens */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Batched prefill for prompt processing */
    for (int i = 0; i < n_prompt; i++)
        counts[prompt_tokens[i]]++;
    tm_tensor *logits = rx_forward_batch(model, prompt_tokens, n_prompt, state);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double prefill_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e6;
    fprintf(stderr, "Prefill: %d tokens in %.1fms (%.1f tok/s)\n",
            n_prompt, prefill_ms,
            prefill_ms > 0 ? n_prompt * 1000.0 / prefill_ms : 0);

    /* Generate tokens */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int generated = 0;

    for (int i = 0; i < max_tokens; i++) {
        if (!logits) break;

        uint32_t token = rx_sample(logits, &params, counts, cfg.vocab_size);
        tm_free(logits);

        /* Stop on end-of-text */
        if (token == 0) break;

        /* Decode and print */
        char *text = rx_tokenizer_decode(tok, &token, 1);
        if (text) {
            printf("%s", text);
            fflush(stdout);
            free(text);
        }

        counts[token]++;
        generated++;

        /* Forward next token */
        logits = rx_forward(model, token, state);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gen_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                    (t1.tv_nsec - t0.tv_nsec) / 1e6;

    printf("\n");
    fprintf(stderr, "Generated: %d tokens in %.1fms (%.1f tok/s)\n",
            generated, gen_ms,
            gen_ms > 0 ? generated * 1000.0 / gen_ms : 0);

    /* Flush HybridMem to disk if enabled */
    rx_state_flush(state);

    /* Cleanup */
    if (logits) tm_free(logits);
    free(counts);
    rx_state_free(state);
    rx_model_free(model);
    rx_tokenizer_free(tok);

    return 0;
}
