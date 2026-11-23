/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_attn_weights.c — HDC Memory Viability Tests 1+2
 *
 * Test 1: Attention Weight on Retrieved Entries
 *   Measures what fraction of softmax mass goes to retrieved (disk) entries
 *   vs cache entries, per head, per layer, per token.
 *
 * Test 2: Retrieval Changes Attention Distribution
 *   Compares attention entropy with and without HybridMem.
 *
 * Method:
 *   Pass A: HybridMem disabled (small cache, entries lost) — baseline entropy
 *   Pass B: HybridMem enabled (small cache, entries retrievable) — entropy + mass
 *   Both passes use identical prompt (fact + filler + question).
 *
 * Output: CSV log files + summary analysis printed to stderr.
 *
 * Usage:
 *   ./examples/test_attn_weights \
 *     --model ../models/base/RWKV-X-3.6B-64k-Base.safetensors \
 *     --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
 *     --filler ../docs/architecture/rwkv-x-deep-dive.md \
 *     --cache-size 8 --hm-top-k 5 --quantize \
 *     --log-file /tmp/attn_weights.csv
 */

#include "rwkvx/rwkvx.h"
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

/* Generate tokens greedily from a prefilled state. */
static char* generate(rx_model *model, rx_tokenizer *tok, rx_state *state,
                      tm_tensor *logits, int max_tokens, rx_config cfg) {
    rx_sampling_params params = rx_sampling_defaults();
    params.temperature = 0.0f;

    uint32_t *gen_tokens = (uint32_t*)calloc(max_tokens, sizeof(uint32_t));
    int n_gen = 0;
    tm_tensor *cur_logits = logits;

    for (int i = 0; i < max_tokens; i++) {
        if (!cur_logits) break;
        uint32_t token = rx_sample(cur_logits, &params, NULL, cfg.vocab_size);
        if (i > 0) tm_free(cur_logits);
        if (token == 0) { cur_logits = NULL; break; }
        gen_tokens[n_gen++] = token;
        cur_logits = rx_forward(model, token, state);
    }
    if (cur_logits && cur_logits != logits) tm_free(cur_logits);

    char *text = rx_tokenizer_decode(tok, gen_tokens, n_gen);
    free(gen_tokens);
    return text;
}

/* Recursive cleanup */
static void rmdir_recursive(const char *path) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", path);
    system(cmd);
}

/* Parse one CSV line: token_step,layer,head,ret_mass,cache_mass,entropy */
typedef struct {
    int token_step;
    int layer;
    int head;
    float ret_mass;
    float cache_mass;
    float entropy;
} csv_row;

/* Load CSV log into array. Returns count. */
static int load_csv(const char *path, csv_row **rows_out) {
    FILE *fp = fopen(path, "r");
    if (!fp) { *rows_out = NULL; return 0; }

    int cap = 4096, count = 0;
    csv_row *rows = (csv_row*)malloc(cap * sizeof(csv_row));
    char line[256];

    /* Skip header */
    if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        free(rows);
        *rows_out = NULL;
        return 0;
    }

    while (fgets(line, sizeof(line), fp)) {
        if (count >= cap) {
            cap *= 2;
            rows = (csv_row*)realloc(rows, cap * sizeof(csv_row));
        }
        csv_row *r = &rows[count];
        if (sscanf(line, "%d,%d,%d,%f,%f,%f",
                   &r->token_step, &r->layer, &r->head,
                   &r->ret_mass, &r->cache_mass, &r->entropy) == 6) {
            count++;
        }
    }
    fclose(fp);
    *rows_out = rows;
    return count;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *vocab_path = NULL;
    const char *filler_path = NULL;
    const char *log_file = "/tmp/attn_weights.csv";
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
        else if (strcmp(argv[i], "--log-file") == 0 && i + 1 < argc)
            log_file = argv[++i];
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: %s --model PATH --vocab PATH --filler PATH [options]\n"
                "  --cache-size N     KV cache slots per attn layer (default: 8)\n"
                "  --hm-top-k N       Retrieval count (default: 5)\n"
                "  --max-filler N     Max chars of filler text (default: 2000)\n"
                "  --max-tokens N     Tokens to generate (default: 32)\n"
                "  --quantize         Quantize weights to INT8\n"
                "  --log-file PATH    CSV output path (default: /tmp/attn_weights.csv)\n",
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

    fprintf(stderr, "Loading model%s...\n", quantize ? " (INT8)" : "");
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
    int filler_len = (int)strlen(filler_raw);
    if (filler_len > max_filler_chars) {
        filler_raw[max_filler_chars] = '\0';
        filler_len = max_filler_chars;
    }

    int prompt_len = (int)(strlen(fact) + 2 + filler_len + strlen(question) + 1);
    char *prompt = (char*)malloc(prompt_len);
    snprintf(prompt, prompt_len, "%s\n\n%s%s", fact, filler_raw, question);
    free(filler_raw);

    /* Tokenize */
    uint32_t *tokens = (uint32_t*)malloc(32768 * sizeof(uint32_t));
    int n_tokens = rx_tokenizer_encode(tok, prompt, tokens, 32768);

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "HDC Memory Viability — Tests 1+2\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "Prompt tokens:  %d\n", n_tokens);
    fprintf(stderr, "Cache size:     %d (per attn layer)\n", cache_size);
    fprintf(stderr, "HM top-k:       %d\n", hm_top_k);
    fprintf(stderr, "Generate:       %d tokens (greedy)\n", max_tokens);
    fprintf(stderr, "Attn layers:    %d\n", cfg.n_attn_layer);
    fprintf(stderr, "Heads:          %d\n", cfg.n_head);
    fprintf(stderr, "\n");

    /* ============================================================
     * Pass A: Small cache, NO HybridMem (baseline entropy)
     * ============================================================ */
    char log_a_path[512], log_b_path[512];
    snprintf(log_a_path, sizeof(log_a_path), "%s.baseline", log_file);
    snprintf(log_b_path, sizeof(log_b_path), "%s", log_file);

    fprintf(stderr, "--- Pass A: cache_size=%d, NO HybridMem (baseline) ---\n", cache_size);
    {
        rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
        hm_cfg.store_dir = NULL;
        hm_cfg.max_cache_len = cache_size;

        rx_state *state_a = rx_state_new_with_hybridmem(model, &hm_cfg);
        if (!state_a) { fprintf(stderr, "Failed to create state A\n"); return 1; }

        /* Enable attention logging */
        FILE *log_a = fopen(log_a_path, "w");
        fprintf(log_a, "token_step,layer,head,ret_mass,cache_mass,entropy\n");
        rx_state_set_attn_log(state_a, log_a);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        /* Prefill + generate */
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

        rx_state_set_attn_log(state_a, NULL);
        fclose(log_a);
        rx_state_free(state_a);

        fprintf(stderr, "Output A: %s\n\n", output_a ? output_a : "(null)");
        free(output_a);
    }

    /* ============================================================
     * Pass B: Small cache, WITH HybridMem (entropy + mass)
     * ============================================================ */
    const char *hm_dir = "/tmp/hm_attn_test";
    rmdir_recursive(hm_dir);

    fprintf(stderr, "--- Pass B: cache_size=%d, WITH HybridMem (top_k=%d) ---\n",
            cache_size, hm_top_k);
    {
        rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
        hm_cfg.store_dir = hm_dir;
        hm_cfg.max_cache_len = cache_size;
        hm_cfg.top_k = hm_top_k;

        rx_state *state_b = rx_state_new_with_hybridmem(model, &hm_cfg);
        if (!state_b) { fprintf(stderr, "Failed to create state B\n"); return 1; }

        FILE *log_b = fopen(log_b_path, "w");
        fprintf(log_b, "token_step,layer,head,ret_mass,cache_mass,entropy\n");
        rx_state_set_attn_log(state_b, log_b);

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

        rx_state_set_attn_log(state_b, NULL);
        fclose(log_b);
        rx_state_flush(state_b);
        rx_state_free(state_b);

        fprintf(stderr, "Output B: %s\n\n", output_b ? output_b : "(null)");
        free(output_b);
    }

    rmdir_recursive(hm_dir);

    /* ============================================================
     * Analysis
     * ============================================================ */
    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "ANALYSIS\n");
    fprintf(stderr, "==========================================\n\n");

    /* Load both CSV logs */
    csv_row *rows_a = NULL, *rows_b = NULL;
    int count_a = load_csv(log_a_path, &rows_a);
    int count_b = load_csv(log_b_path, &rows_b);

    fprintf(stderr, "Pass A log: %d rows (%s)\n", count_a, log_a_path);
    fprintf(stderr, "Pass B log: %d rows (%s)\n\n", count_b, log_b_path);

    /* ---- Test 1: Attention weight on retrieved entries ---- */
    fprintf(stderr, "--- Test 1: Attention Weight on Retrieved Entries ---\n");
    {
        int heads_gt1 = 0, heads_gt5 = 0, total_heads = 0;
        float total_ret_mass = 0.0f;
        int generation_rows = 0;

        for (int i = 0; i < count_b; i++) {
            total_heads++;
            if (rows_b[i].ret_mass > 0.01f) heads_gt1++;
            if (rows_b[i].ret_mass > 0.05f) heads_gt5++;

            /* Count generation tokens (step > n_tokens - 1) */
            if (rows_b[i].token_step >= n_tokens) {
                total_ret_mass += rows_b[i].ret_mass;
                generation_rows++;
            }
        }

        float pct_gt1 = total_heads > 0 ? 100.0f * heads_gt1 / total_heads : 0;
        float pct_gt5 = total_heads > 0 ? 100.0f * heads_gt5 / total_heads : 0;
        float avg_ret_mass_gen = generation_rows > 0
            ? total_ret_mass / generation_rows : 0;

        fprintf(stderr, "  Heads with >1%% retrieved mass: %d/%d (%.1f%%)\n",
                heads_gt1, total_heads, pct_gt1);
        fprintf(stderr, "  Heads with >5%% retrieved mass: %d/%d (%.1f%%)\n",
                heads_gt5, total_heads, pct_gt5);
        fprintf(stderr, "  Avg retrieved mass (generation tokens): %.4f\n",
                avg_ret_mass_gen);

        /* Per-layer summary */
        fprintf(stderr, "\n  Per-layer breakdown (generation tokens only):\n");
        for (int l = 0; l < cfg.n_attn_layer; l++) {
            int layer_heads = 0, layer_gt1 = 0, layer_gt5 = 0;
            float layer_mass = 0.0f;
            for (int i = 0; i < count_b; i++) {
                if (rows_b[i].layer == l && rows_b[i].token_step >= n_tokens) {
                    layer_heads++;
                    layer_mass += rows_b[i].ret_mass;
                    if (rows_b[i].ret_mass > 0.01f) layer_gt1++;
                    if (rows_b[i].ret_mass > 0.05f) layer_gt5++;
                }
            }
            float avg = layer_heads > 0 ? layer_mass / layer_heads : 0;
            fprintf(stderr, "    Layer %d: avg_ret_mass=%.4f, >1%%=%d/%d, >5%%=%d/%d\n",
                    l, avg, layer_gt1, layer_heads, layer_gt5, layer_heads);
        }

        fprintf(stderr, "\n  VERDICT: ");
        if (pct_gt5 >= 20.0f) {
            fprintf(stderr, "PASS (%.1f%% of heads allocate >5%% to retrieved)\n", pct_gt5);
        } else if (pct_gt1 >= 5.0f) {
            fprintf(stderr, "MARGINAL (%.1f%% >1%%, %.1f%% >5%% — needs investigation)\n",
                    pct_gt1, pct_gt5);
        } else {
            fprintf(stderr, "FAIL (<1%% attention weight on retrieved entries)\n");
        }
    }

    /* ---- Test 2: Entropy shift ---- */
    fprintf(stderr, "\n--- Test 2: Retrieval Changes Attention Distribution ---\n");
    {
        /* Build per-(step, layer, head) entropy maps for both passes.
         * Simple approach: iterate rows, match by (step, layer, head). */
        int n_shifts = 0, shifts_gt01 = 0;
        float total_delta = 0.0f;

        /* For each row in B, find matching row in A by (step, layer, head) */
        for (int i = 0; i < count_b; i++) {
            for (int j = 0; j < count_a; j++) {
                if (rows_a[j].token_step == rows_b[i].token_step &&
                    rows_a[j].layer == rows_b[i].layer &&
                    rows_a[j].head == rows_b[i].head) {
                    float delta = rows_b[i].entropy - rows_a[j].entropy;
                    total_delta += delta;
                    n_shifts++;
                    if (fabsf(delta) > 0.1f) shifts_gt01++;
                    break;
                }
            }
        }

        float avg_delta = n_shifts > 0 ? total_delta / n_shifts : 0;
        float pct_shifted = n_shifts > 0 ? 100.0f * shifts_gt01 / n_shifts : 0;

        fprintf(stderr, "  Matched (step,layer,head) pairs: %d\n", n_shifts);
        fprintf(stderr, "  Avg entropy shift (B - A): %.4f nats\n", avg_delta);
        fprintf(stderr, "  Heads with |delta_H| > 0.1 nats: %d/%d (%.1f%%)\n",
                shifts_gt01, n_shifts, pct_shifted);

        fprintf(stderr, "\n  VERDICT: ");
        if (pct_shifted >= 25.0f) {
            fprintf(stderr, "PASS (%.1f%% of heads show entropy shift >0.1 nats)\n",
                    pct_shifted);
        } else if (pct_shifted >= 10.0f) {
            fprintf(stderr, "MARGINAL (%.1f%% shifted — some effect but weak)\n",
                    pct_shifted);
        } else {
            fprintf(stderr, "FAIL (retrieval doesn't measurably affect attention)\n");
        }
    }

    /* Cleanup */
    free(rows_a);
    free(rows_b);
    free(prompt);
    free(tokens);
    rx_model_free(model);
    rx_tokenizer_free(tok);

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "CSV logs saved to:\n");
    fprintf(stderr, "  Baseline: %s\n", log_a_path);
    fprintf(stderr, "  HybridMem: %s\n", log_b_path);
    fprintf(stderr, "==========================================\n");

    return 0;
}
