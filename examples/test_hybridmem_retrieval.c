/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_hybridmem_retrieval.c — A/B test of HybridMem retrieval with real inference
 *
 * Feeds: [fact] [filler from file] [question about fact]
 * Run A: small cache, NO disk backup → fact is evicted and lost
 * Run B: small cache, WITH disk backup → fact is evicted but retrievable
 *
 * Both runs use identical cache size so the only variable is disk retrieval.
 *
 * Usage:
 *   ./examples/test_hybridmem_retrieval \
 *     --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
 *     --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
 *     --filler docs/architecture/rwkv-x-deep-dive.md
 */

#include "rwkvx/rwkvx.h"
#include "tensor/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

/* Read entire file into malloc'd string. Returns NULL on error. */
static char* read_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = (char*)malloc(len + 1);
    size_t read = fread(buf, 1, len, fp);
    buf[read] = '\0';
    fclose(fp);
    return buf;
}

/* Generate tokens from a state that has already been prefilled.
 * Returns malloc'd string of generated text. */
static char* generate(rx_model *model, rx_tokenizer *tok, rx_state *state,
                      tm_tensor *logits, int max_tokens, rx_config cfg) {
    rx_sampling_params params = rx_sampling_defaults();
    params.temperature = 0.0f; /* greedy for reproducibility */

    uint32_t *gen_tokens = (uint32_t*)calloc(max_tokens, sizeof(uint32_t));
    int n_gen = 0;
    tm_tensor *cur_logits = logits;

    for (int i = 0; i < max_tokens; i++) {
        if (!cur_logits) break;
        uint32_t token = rx_sample(cur_logits, &params, NULL, cfg.vocab_size);
        if (i > 0) tm_free(cur_logits);
        if (token == 0) { cur_logits = NULL; break; } /* end-of-text */
        gen_tokens[n_gen++] = token;
        cur_logits = rx_forward(model, token, state);
    }
    if (cur_logits && cur_logits != logits) tm_free(cur_logits);

    char *text = rx_tokenizer_decode(tok, gen_tokens, n_gen);
    free(gen_tokens);
    return text;
}

/* Recursive rm -rf for cleanup */
static void rmdir_recursive(const char *path) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", path);
    system(cmd);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *vocab_path = NULL;
    const char *filler_path = NULL;
    int cache_size = 8;
    int hm_top_k = 5;
    int max_filler_chars = 2000;
    int max_tokens = 32;
    int quantize = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc)
            vocab_path = argv[++i];
        else if (strcmp(argv[i], "--filler") == 0 && i + 1 < argc)
            filler_path = argv[++i];
        else if (strcmp(argv[i], "--cache-size") == 0 && i + 1 < argc)
            cache_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hm-top-k") == 0 && i + 1 < argc)
            hm_top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-filler") == 0 && i + 1 < argc)
            max_filler_chars = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--quantize") == 0)
            quantize = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: %s --model PATH --vocab PATH --filler PATH [options]\n"
                "  --cache-size N     KV cache slots per attn layer (default: 8)\n"
                "  --hm-top-k N       Retrieval count (default: 5)\n"
                "  --max-filler N     Max chars of filler text (default: 2000)\n"
                "  --max-tokens N     Tokens to generate (default: 32)\n"
                "  --quantize         Quantize weights to INT8\n",
                argv[0]);
            return 0;
        }
    }

    if (!model_path || !vocab_path || !filler_path) {
        fprintf(stderr, "Error: --model, --vocab, and --filler are required\n");
        return 1;
    }

    srand(42);

    /* Load shared resources */
    fprintf(stderr, "Loading tokenizer...\n");
    rx_tokenizer *tok = rx_tokenizer_new(vocab_path);
    if (!tok) { fprintf(stderr, "Failed to load tokenizer\n"); return 1; }

    fprintf(stderr, "Loading model%s...\n", quantize ? " (INT8 quantized)" : "");
    rx_model *model = quantize ? rx_model_load_q8(model_path)
                               : rx_model_load(model_path);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    rx_config cfg = rx_model_config(model);

    /* Build prompt: fact + filler + question */
    const char *fact = "The secret project codename is AURORA-7 and the lead researcher is Dr. Kimura.";
    const char *question = "\n\nQuestion: What is the secret project codename and who leads it?\nAnswer:";

    char *filler_raw = read_file(filler_path);
    if (!filler_raw) {
        fprintf(stderr, "Failed to read filler file: %s\n", filler_path);
        return 1;
    }

    /* Truncate filler */
    int filler_len = (int)strlen(filler_raw);
    if (filler_len > max_filler_chars) {
        filler_raw[max_filler_chars] = '\0';
        filler_len = max_filler_chars;
    }

    /* Assemble full prompt */
    int prompt_len = (int)(strlen(fact) + 2 + filler_len + strlen(question) + 1);
    char *prompt = (char*)malloc(prompt_len);
    snprintf(prompt, prompt_len, "%s\n\n%s%s", fact, filler_raw, question);
    free(filler_raw);

    /* Tokenize */
    uint32_t *tokens = (uint32_t*)malloc(32768 * sizeof(uint32_t));
    int n_tokens = rx_tokenizer_encode(tok, prompt, tokens, 32768);

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "HybridMem A/B Retrieval Test\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "Prompt tokens:  %d\n", n_tokens);
    fprintf(stderr, "Cache size:     %d (per attn layer)\n", cache_size);
    fprintf(stderr, "HM top-k:       %d\n", hm_top_k);
    fprintf(stderr, "Generate:       %d tokens (greedy)\n", max_tokens);
    fprintf(stderr, "Fact: \"%s\"\n", fact);
    fprintf(stderr, "Question: \"%s\"\n", question);
    fprintf(stderr, "\n");

    /* ============================================================
     * Run A: Small cache, NO HybridMem (evicted entries are lost)
     * ============================================================ */
    fprintf(stderr, "--- Run A: cache_size=%d, NO HybridMem ---\n", cache_size);
    {
        rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
        hm_cfg.store_dir = NULL;  /* no disk backup */
        hm_cfg.max_cache_len = cache_size;

        rx_state *state_a = rx_state_new_with_hybridmem(model, &hm_cfg);
        if (!state_a) { fprintf(stderr, "Failed to create state A\n"); return 1; }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        tm_tensor *logits_a = NULL;
        for (int i = 0; i < n_tokens; i++) {
            if (logits_a) tm_free(logits_a);
            logits_a = rx_forward(model, tokens[i], state_a);
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        fprintf(stderr, "Prefill: %d tokens in %.1fms (%.1f tok/s)\n",
                n_tokens, ms, n_tokens * 1000.0 / ms);

        char *output_a = generate(model, tok, state_a, logits_a, max_tokens, cfg);
        tm_free(logits_a);
        rx_state_free(state_a);

        fprintf(stderr, "Output A: %s\n\n", output_a ? output_a : "(null)");
        free(output_a);
    }

    /* ============================================================
     * Run B: Small cache, WITH HybridMem (evicted entries retrievable)
     * ============================================================ */
    const char *hm_dir = "/tmp/hm_ab_test";
    rmdir_recursive(hm_dir);

    fprintf(stderr, "--- Run B: cache_size=%d, WITH HybridMem (top_k=%d) ---\n",
            cache_size, hm_top_k);
    {
        rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
        hm_cfg.store_dir = hm_dir;
        hm_cfg.max_cache_len = cache_size;
        hm_cfg.top_k = hm_top_k;

        rx_state *state_b = rx_state_new_with_hybridmem(model, &hm_cfg);
        if (!state_b) { fprintf(stderr, "Failed to create state B\n"); return 1; }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        tm_tensor *logits_b = NULL;
        for (int i = 0; i < n_tokens; i++) {
            if (logits_b) tm_free(logits_b);
            logits_b = rx_forward(model, tokens[i], state_b);
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        fprintf(stderr, "Prefill: %d tokens in %.1fms (%.1f tok/s)\n",
                n_tokens, ms, n_tokens * 1000.0 / ms);

        char *output_b = generate(model, tok, state_b, logits_b, max_tokens, cfg);
        tm_free(logits_b);
        rx_state_flush(state_b);
        rx_state_free(state_b);

        fprintf(stderr, "Output B: %s\n\n", output_b ? output_b : "(null)");

        /* Check disk artifacts exist */
        struct stat sb;
        char shard_path[512];
        snprintf(shard_path, sizeof(shard_path), "%s/shard_0000.safetensors", hm_dir);
        if (stat(shard_path, &sb) == 0) {
            fprintf(stderr, "HybridMem shard: %s (%lld bytes)\n", shard_path, (long long)sb.st_size);
        } else {
            fprintf(stderr, "WARNING: No shard file found at %s\n", shard_path);
        }
        free(output_b);
    }

    /* Cleanup */
    rmdir_recursive(hm_dir);
    free(prompt);
    free(tokens);
    rx_model_free(model);
    rx_tokenizer_free(tok);

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "Compare outputs above. If HybridMem retrieval works,\n");
    fprintf(stderr, "Run B should reference AURORA-7 / Dr. Kimura more than Run A.\n");
    fprintf(stderr, "==========================================\n");

    return 0;
}
