/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_context_scaling.c — HDC Memory Viability Test 7
 *
 * Context Length Scaling
 *
 * Method:
 *   Vary prompt length (200, 500, 1000, 2000, 4000 tokens via filler)
 *   Vary cache_size (8, 64, 256)
 *   Same AURORA-7 fact + question.
 *   Find crossover point where HybridMem starts helping.
 *
 * Usage:
 *   ./examples/test_context_scaling \
 *     --model ../models/base/RWKV-X-3.6B-64k-Base.safetensors \
 *     --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
 *     --filler ../docs/architecture/rwkv-x-deep-dive.md
 */

#include "rwkvx/rwkvx.h"
#include "tensor/tensor.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

/* Case-insensitive substring search */
static int contains_ci(const char *haystack, const char *needle) {
    if (!haystack || !needle) return 0;
    int hlen = (int)strlen(haystack);
    int nlen = (int)strlen(needle);
    for (int i = 0; i <= hlen - nlen; i++) {
        int match = 1;
        for (int j = 0; j < nlen; j++) {
            if (tolower((unsigned char)haystack[i + j]) !=
                tolower((unsigned char)needle[j])) {
                match = 0;
                break;
            }
        }
        if (match) return 1;
    }
    return 0;
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

/* Extend filler to target character count by repeating */
static char* extend_filler(const char *base, int target_chars) {
    int base_len = (int)strlen(base);
    if (base_len == 0) return strdup("");
    char *buf = (char*)malloc(target_chars + base_len + 1);
    int pos = 0;
    while (pos < target_chars) {
        int chunk = base_len;
        if (pos + chunk > target_chars) chunk = target_chars - pos;
        memcpy(buf + pos, base, chunk);
        pos += chunk;
    }
    buf[target_chars] = '\0';
    return buf;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *vocab_path = NULL;
    const char *filler_path = NULL;
    int hm_top_k = 5;
    int max_tokens = 48;
    int quantize = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) vocab_path = argv[++i];
        else if (strcmp(argv[i], "--filler") == 0 && i + 1 < argc) filler_path = argv[++i];
        else if (strcmp(argv[i], "--hm-top-k") == 0 && i + 1 < argc) hm_top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--quantize") == 0) quantize = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "Usage: %s --model PATH --vocab PATH --filler PATH [options]\n", argv[0]);
            return 0;
        }
    }

    if (!model_path || !vocab_path || !filler_path) {
        fprintf(stderr, "Error: --model, --vocab, and --filler are required\n");
        return 1;
    }

    /* Load shared resources */
    fprintf(stderr, "Loading tokenizer...\n");
    rx_tokenizer *tok = rx_tokenizer_new(vocab_path);
    if (!tok) { fprintf(stderr, "Failed to load tokenizer\n"); return 1; }

    fprintf(stderr, "Loading model%s...\n", quantize ? " (INT8)" : "");
    rx_model *model = quantize ? rx_model_load_q8(model_path)
                               : rx_model_load(model_path);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    rx_config cfg = rx_model_config(model);

    /* Load filler */
    char *filler_raw = read_file(filler_path);
    if (!filler_raw) { fprintf(stderr, "Failed to read filler\n"); return 1; }

    const char *fact = "The secret project codename is AURORA-7 and the lead researcher is Dr. Kimura.";
    const char *question = "\n\nQuestion: What is the secret project codename and who leads it?\nAnswer:";
    const char *key_detail = "AURORA";

    /* Target token counts and cache sizes to test */
    int target_tokens[] = {200, 500, 1000, 2000, 4000};
    int n_targets = 5;
    int cache_sizes[] = {8, 64, 256};
    int n_caches = 3;

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "HDC Memory Viability — Test 7\n");
    fprintf(stderr, "Context Length Scaling\n");
    fprintf(stderr, "==========================================\n\n");

    /* Results: [target_idx][cache_idx] => 0=both_miss, 1=A_only, 2=B_only, 3=both_hit */
    int results[5][3];
    memset(results, 0, sizeof(results));

    /* Header */
    fprintf(stderr, "%-10s", "tokens");
    for (int c = 0; c < n_caches; c++)
        fprintf(stderr, "  cache=%-6d", cache_sizes[c]);
    fprintf(stderr, "\n");
    for (int i = 0; i < 10 + n_caches * 14; i++) fprintf(stderr, "-");
    fprintf(stderr, "\n");

    for (int ti = 0; ti < n_targets; ti++) {
        int target = target_tokens[ti];

        /* Estimate chars needed for target token count (~3.5 chars/token) */
        int filler_chars = (int)(target * 3.5);
        char *filler = extend_filler(filler_raw, filler_chars);

        /* Build prompt */
        int pl = (int)(strlen(fact) + 2 + strlen(filler) + strlen(question) + 1);
        char *prompt = (char*)malloc(pl);
        snprintf(prompt, pl, "%s\n\n%s%s", fact, filler, question);

        uint32_t *tokens = (uint32_t*)malloc(65536 * sizeof(uint32_t));
        int n_tokens = rx_tokenizer_encode(tok, prompt, tokens, 65536);

        fprintf(stderr, "%-10d", n_tokens);

        for (int ci = 0; ci < n_caches; ci++) {
            int cs = cache_sizes[ci];

            /* Run A: no HM */
            rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
            hm_cfg.store_dir = NULL;
            hm_cfg.max_cache_len = cs;
            rx_state *state_a = rx_state_new_with_hybridmem(model, &hm_cfg);

            tm_tensor *logits = NULL;
            for (int i = 0; i < n_tokens; i++) {
                if (logits) tm_free(logits);
                logits = rx_forward(model, tokens[i], state_a);
            }
            char *out_a = generate(model, tok, state_a, logits, max_tokens, cfg);
            tm_free(logits);
            rx_state_free(state_a);
            int hit_a = contains_ci(out_a, key_detail);

            /* Run B: with HM */
            const char *hm_dir = "/tmp/hm_context_scaling";
            rmdir_recursive(hm_dir);
            hm_cfg.store_dir = hm_dir;
            hm_cfg.max_cache_len = cs;
            hm_cfg.top_k = hm_top_k;
            rx_state *state_b = rx_state_new_with_hybridmem(model, &hm_cfg);

            logits = NULL;
            for (int i = 0; i < n_tokens; i++) {
                if (logits) tm_free(logits);
                logits = rx_forward(model, tokens[i], state_b);
            }
            char *out_b = generate(model, tok, state_b, logits, max_tokens, cfg);
            tm_free(logits);
            rx_state_free(state_b);
            int hit_b = contains_ci(out_b, key_detail);
            rmdir_recursive(hm_dir);

            /* Encode result */
            if (hit_a && hit_b) results[ti][ci] = 3;
            else if (hit_a) results[ti][ci] = 1;
            else if (hit_b) results[ti][ci] = 2;
            else results[ti][ci] = 0;

            const char *labels[] = {"--/--", "A/--", "--/B", "A/B"};
            fprintf(stderr, "  %-12s", labels[results[ti][ci]]);

            free(out_a);
            free(out_b);
        }
        fprintf(stderr, "\n");

        free(prompt);
        free(tokens);
        free(filler);
    }

    /* Analysis */
    fprintf(stderr, "\nLegend: A=no_HM hit, B=HM hit, --=miss\n\n");

    /* Find crossover points */
    fprintf(stderr, "--- Crossover Analysis ---\n");
    for (int ci = 0; ci < n_caches; ci++) {
        int crossover = -1;
        for (int ti = 0; ti < n_targets; ti++) {
            if (results[ti][ci] == 2) { /* B hits but A doesn't */
                crossover = target_tokens[ti];
                break;
            }
        }
        if (crossover > 0) {
            fprintf(stderr, "  cache=%d: HybridMem starts helping at ~%d tokens\n",
                    cache_sizes[ci], crossover);
        } else {
            /* Check if both always hit or both always miss */
            int all_both = 1, all_miss = 1;
            for (int ti = 0; ti < n_targets; ti++) {
                if (results[ti][ci] != 3) all_both = 0;
                if (results[ti][ci] != 0) all_miss = 0;
            }
            if (all_both) {
                fprintf(stderr, "  cache=%d: Both methods always succeed (cache large enough)\n",
                        cache_sizes[ci]);
            } else if (all_miss) {
                fprintf(stderr, "  cache=%d: Both methods always fail\n", cache_sizes[ci]);
            } else {
                fprintf(stderr, "  cache=%d: No clear crossover pattern\n", cache_sizes[ci]);
            }
        }
    }

    fprintf(stderr, "\nVERDICT:\n");
    /* Count how many conditions HybridMem helps (result==2) */
    int hm_helps = 0, total_cells = n_targets * n_caches;
    for (int ti = 0; ti < n_targets; ti++)
        for (int ci = 0; ci < n_caches; ci++)
            if (results[ti][ci] == 2) hm_helps++;

    if (hm_helps > 0) {
        fprintf(stderr, "  PASS — HybridMem helps in %d/%d conditions\n", hm_helps, total_cells);
    } else {
        /* Check if cache always sufficient */
        int both_hit = 0;
        for (int ti = 0; ti < n_targets; ti++)
            for (int ci = 0; ci < n_caches; ci++)
                if (results[ti][ci] == 3) both_hit++;
        if (both_hit == total_cells) {
            fprintf(stderr, "  N/A — Cache always sufficient at tested sizes\n");
        } else {
            fprintf(stderr, "  FAIL — HybridMem doesn't help at any tested condition\n");
        }
    }

    /* Cleanup */
    free(filler_raw);
    rx_model_free(model);
    rx_tokenizer_free(tok);

    fprintf(stderr, "\n==========================================\n");
    return 0;
}
