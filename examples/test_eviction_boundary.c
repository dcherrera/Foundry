/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_eviction_boundary.c — HDC Memory Viability Test 11
 *
 * Eviction Boundary Analysis
 *
 * Method:
 *   1. Run 3.6B (or 0.2B) needle-in-haystack with HybridMem
 *   2. After prefill, query the HDC store for the fact embedding
 *   3. Report:
 *      - Total entries stored to disk
 *      - Per-layer entry count
 *      - Rank of fact entry per layer (how well can we retrieve it?)
 *      - Score distribution of retrieved entries
 *
 * Usage:
 *   ./examples/test_eviction_boundary \
 *     --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
 *     --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
 *     --filler ../docs/architecture/rwkv-x-deep-dive.md
 */

#include "rwkvx/rwkvx.h"
#include "hybridmem/hybridmem.h"
#include "tensor/tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Read entire file into malloc'd string. */
static char* read_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = (char*)malloc(len + 1);
    size_t nread = fread(buf, 1, len, fp);
    buf[nread] = '\0';
    fclose(fp);
    return buf;
}

static void rmdir_recursive(const char *path) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", path);
    system(cmd);
}

/* Cosine similarity between two float vectors */
static float cosine_sim(const float *a, const float *b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float d = sqrtf(na) * sqrtf(nb);
    return d > 1e-12f ? dot / d : 0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *vocab_path = NULL;
    const char *filler_path = NULL;
    int cache_size = 8;
    int hm_top_k = 10;  /* retrieve more for rank analysis */
    int max_filler_chars = 2000;
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
        else if (strcmp(argv[i], "--quantize") == 0)
            quantize = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: %s --model PATH --vocab PATH --filler PATH [options]\n"
                "  --cache-size N     KV cache slots per attn layer (default: 8)\n"
                "  --hm-top-k N       Retrieval count for rank analysis (default: 10)\n"
                "  --max-filler N     Max chars of filler text (default: 2000)\n"
                "  --quantize         Quantize weights to INT8\n",
                argv[0]);
            return 0;
        }
    }

    if (!model_path || !vocab_path || !filler_path) {
        fprintf(stderr, "Error: --model, --vocab, and --filler are required\n");
        return 1;
    }

    /* Load model + tokenizer */
    fprintf(stderr, "Loading tokenizer...\n");
    rx_tokenizer *tok = rx_tokenizer_new(vocab_path);
    if (!tok) { fprintf(stderr, "Failed to load tokenizer\n"); return 1; }

    fprintf(stderr, "Loading model%s...\n", quantize ? " (INT8)" : "");
    rx_model *model = quantize ? rx_model_load_q8(model_path)
                               : rx_model_load(model_path);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    rx_config cfg = rx_model_config(model);
    int c = cfg.n_embd;

    /* Build prompt */
    const char *fact = "The secret project codename is AURORA-7 and the lead researcher is Dr. Kimura.";
    const char *question = "\n\nQuestion: What is the secret project codename and who leads it?\nAnswer:";

    char *filler_raw = read_file(filler_path);
    if (!filler_raw) {
        fprintf(stderr, "Failed to read filler file: %s\n", filler_path);
        return 1;
    }
    if ((int)strlen(filler_raw) > max_filler_chars)
        filler_raw[max_filler_chars] = '\0';

    int prompt_len = (int)(strlen(fact) + 2 + strlen(filler_raw) + strlen(question) + 1);
    char *prompt = (char*)malloc(prompt_len);
    snprintf(prompt, prompt_len, "%s\n\n%s%s", fact, filler_raw, question);
    free(filler_raw);

    /* Tokenize fact separately (to get fact token range) */
    uint32_t *fact_tokens = (uint32_t*)malloc(1024 * sizeof(uint32_t));
    int n_fact_tokens = rx_tokenizer_encode(tok, fact, fact_tokens, 1024);

    uint32_t *tokens = (uint32_t*)malloc(32768 * sizeof(uint32_t));
    int n_tokens = rx_tokenizer_encode(tok, prompt, tokens, 32768);

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "HDC Memory Viability — Test 11\n");
    fprintf(stderr, "Eviction Boundary Analysis\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "Prompt tokens:  %d\n", n_tokens);
    fprintf(stderr, "Fact tokens:    %d (first %d tokens of prompt)\n",
            n_fact_tokens, n_fact_tokens);
    fprintf(stderr, "Cache size:     %d\n", cache_size);
    fprintf(stderr, "HM top-k:       %d\n", hm_top_k);
    fprintf(stderr, "Attn layers:    %d\n\n", cfg.n_attn_layer);

    /* Run prefill with HybridMem */
    const char *hm_dir = "/tmp/hm_eviction_test";
    rmdir_recursive(hm_dir);

    rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
    hm_cfg.store_dir = hm_dir;
    hm_cfg.max_cache_len = cache_size;
    hm_cfg.top_k = hm_top_k;

    rx_state *state = rx_state_new_with_hybridmem(model, &hm_cfg);
    if (!state) { fprintf(stderr, "Failed to create state\n"); return 1; }

    /* Also capture fact embeddings for querying */
    int max_capture = n_fact_tokens + 10;
    float *fact_embeddings = (float*)malloc((size_t)max_capture * c * sizeof(float));
    rx_state_set_embed_capture(state, fact_embeddings, max_capture, -1);

    fprintf(stderr, "Running prefill...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    tm_tensor *logits = NULL;
    for (int i = 0; i < n_tokens; i++) {
        if (logits) tm_free(logits);
        logits = rx_forward(model, tokens[i], state);

        /* Stop capturing after fact tokens */
        if (i == n_fact_tokens - 1) {
            rx_state_set_embed_capture(state, NULL, 0, -1);
        }
    }
    if (logits) tm_free(logits);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    int n_captured = rx_state_embed_capture_count(state);
    fprintf(stderr, "Prefill: %d tokens in %.1fms (%.1f tok/s)\n",
            n_tokens, ms, n_tokens * 1000.0 / ms);
    fprintf(stderr, "Captured %d fact embeddings\n\n", n_captured);

    /* Store stats */
    fm_stats stats = fm_stats_get(rx_state_get_hm_store(state));
    fprintf(stderr, "HybridMem store stats:\n");
    fprintf(stderr, "  Total stores:    %zu\n", stats.total_stores);
    fprintf(stderr, "  Total entries:   %zu\n", stats.num_entries);
    fprintf(stderr, "  Disk bytes:      %zu\n", stats.disk_bytes_used);
    fprintf(stderr, "\n");

    /* Flush store before querying */
    rx_state_flush(state);

    /* Query the HDC store for each captured fact embedding */
    fprintf(stderr, "--- Fact Embedding Retrieval Analysis ---\n\n");

    if (n_captured == 0) {
        fprintf(stderr, "No fact embeddings captured (model may have 0 attn layers in fact range).\n");
        fprintf(stderr, "With %d fact tokens and cache_size=%d, the fact may still be in cache.\n",
                n_fact_tokens, cache_size);
    }

    /* For each captured embedding, query the store */
    for (int e = 0; e < n_captured && e < 4; e++) {
        float *query_emb = &fact_embeddings[e * c];

        fprintf(stderr, "Fact embedding %d:\n", e);

        /* Query each layer */
        for (int l = 0; l < cfg.n_attn_layer; l++) {
            int64_t shape_q[] = {c};
            tm_tensor *query_t = tm_from_data(query_emb, shape_q, 1, TM_F32);
            fm_results *results = fm_query(rx_state_get_hm_store(state), query_t, l, hm_top_k);
            tm_free(query_t);

            if (results && results->count > 0) {
                fprintf(stderr, "  Layer %d: %d results\n", l, results->count);
                for (int r = 0; r < results->count && r < 5; r++) {
                    int n_tok = (int)tm_size(results->items[r].keys, 0);

                    /* Compute cosine similarity between query and first key in result */
                    float *rk = tm_data_f32(results->items[r].keys);
                    float cos = cosine_sim(query_emb, rk, c);

                    fprintf(stderr, "    Rank %d: score=%.4f, cos_sim=%.4f, n_tokens=%d\n",
                            r, results->items[r].score, cos, n_tok);
                }
                fm_results_free(results);
            } else {
                fprintf(stderr, "  Layer %d: no results\n", l);
                if (results) fm_results_free(results);
            }
        }
        fprintf(stderr, "\n");
    }

    /* Eviction count analysis */
    int eviction_count = n_tokens / cache_size;  /* approximate */
    fprintf(stderr, "--- Eviction Analysis ---\n");
    fprintf(stderr, "Approximate evictions: %d (tokens=%d / cache=%d)\n",
            eviction_count, n_tokens, cache_size);
    fprintf(stderr, "Entries in store: %zu\n", stats.num_entries);
    fprintf(stderr, "Tokens per eviction batch: %d (cache_size * evict_fraction)\n",
            (int)(cache_size * hm_cfg.evict_fraction));

    fprintf(stderr, "\nVERDICT:\n");
    if (stats.num_entries > 0 && n_captured > 0) {
        fprintf(stderr, "  Store populated with %zu entries — eviction pipeline works.\n",
                stats.num_entries);
        fprintf(stderr, "  Check retrieval scores above: high scores indicate fact is retrievable.\n");
    } else if (stats.num_entries > 0) {
        fprintf(stderr, "  Store populated but no fact embeddings captured.\n");
        fprintf(stderr, "  The fact may be in cache (within first %d tokens).\n", cache_size);
    } else {
        fprintf(stderr, "  FAIL — No entries stored. Check eviction path.\n");
    }

    /* Cleanup */
    free(fact_embeddings);
    free(fact_tokens);
    free(tokens);
    free(prompt);
    rx_state_free(state);
    rx_model_free(model);
    rx_tokenizer_free(tok);
    rmdir_recursive(hm_dir);

    fprintf(stderr, "\n==========================================\n");
    return 0;
}
