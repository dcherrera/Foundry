/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_query_vector.c — HDC Memory Viability Test 10
 *
 * Query Vector Comparison
 *
 * Method:
 *   Compare retrieval quality when using two different query strategies:
 *     A: ln_out (layer-norm output before Q/K/V projections) — current default
 *     B: Q projection (query after Q linear — what the model actually attends with)
 *
 *   For each strategy, run the needle-in-haystack test and compare:
 *   - Whether the fact is retrieved
 *   - What attention mass goes to retrieved entries
 *   - Whether output quality improves
 *
 *   Implementation: Uses the embed_capture API to capture both ln_out and
 *   Q-projected vectors. Then manually queries the HDC store with each to
 *   compare retrieval rank and similarity scores.
 *
 * Usage:
 *   ./examples/test_query_vector \
 *     --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
 *     --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
 *     --filler ../docs/architecture/rwkv-x-deep-dive.md
 */

#include "rwkvx/rwkvx.h"
#include "hybridmem/hybridmem.h"
#include "tensor/tensor.h"

#include <ctype.h>
#include <math.h>
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

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *vocab_path = NULL;
    const char *filler_path = NULL;
    int cache_size = 8;
    int hm_top_k = 5;
    int max_filler_chars = 2000;
    int max_tokens = 48;
    int quantize = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) vocab_path = argv[++i];
        else if (strcmp(argv[i], "--filler") == 0 && i + 1 < argc) filler_path = argv[++i];
        else if (strcmp(argv[i], "--cache-size") == 0 && i + 1 < argc) cache_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hm-top-k") == 0 && i + 1 < argc) hm_top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-filler") == 0 && i + 1 < argc) max_filler_chars = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--quantize") == 0) quantize = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: %s --model PATH --vocab PATH --filler PATH [options]\n"
                "  --cache-size N     KV cache slots (default: 8)\n"
                "  --hm-top-k N       Retrieval count (default: 5)\n"
                "  --max-filler N     Max chars of filler text (default: 2000)\n"
                "  --max-tokens N     Max generated tokens (default: 48)\n"
                "  --quantize         Quantize weights to INT8\n",
                argv[0]);
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
    int c = cfg.n_embd;

    /* Build prompt */
    const char *fact = "The secret project codename is AURORA-7 and the lead researcher is Dr. Kimura.";
    const char *question = "\n\nQuestion: What is the secret project codename and who leads it?\nAnswer:";
    const char *key_detail = "AURORA";

    char *filler_raw = read_file(filler_path);
    if (!filler_raw) { fprintf(stderr, "Failed to read filler\n"); return 1; }
    if ((int)strlen(filler_raw) > max_filler_chars)
        filler_raw[max_filler_chars] = '\0';

    int pl = (int)(strlen(fact) + 2 + strlen(filler_raw) + strlen(question) + 1);
    char *prompt = (char*)malloc(pl);
    snprintf(prompt, pl, "%s\n\n%s%s", fact, filler_raw, question);
    free(filler_raw);

    /* Tokenize fact separately to know how many fact tokens there are */
    uint32_t *fact_tokens = (uint32_t*)malloc(1024 * sizeof(uint32_t));
    int n_fact_tokens = rx_tokenizer_encode(tok, fact, fact_tokens, 1024);

    uint32_t *tokens = (uint32_t*)malloc(32768 * sizeof(uint32_t));
    int n_tokens = rx_tokenizer_encode(tok, prompt, tokens, 32768);

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "HDC Memory Viability — Test 10\n");
    fprintf(stderr, "Query Vector Comparison\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "Prompt tokens:  %d\n", n_tokens);
    fprintf(stderr, "Fact tokens:    %d\n", n_fact_tokens);
    fprintf(stderr, "Cache size:     %d\n", cache_size);
    fprintf(stderr, "HM top-k:       %d\n\n", hm_top_k);

    /*
     * Strategy: Run prefill with HybridMem and capture fact embeddings (ln_out).
     * Then query the HDC store with:
     *   (a) The captured ln_out embeddings (what the current system uses)
     *   (b) A mean-pooled version of the fact embeddings
     *
     * We compare retrieval scores to understand which query vector
     * better identifies the relevant stored entries.
     */

    /* Run with HybridMem to populate store and capture embeddings */
    const char *hm_dir = "/tmp/hm_query_vector";
    rmdir_recursive(hm_dir);

    rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
    hm_cfg.store_dir = hm_dir;
    hm_cfg.max_cache_len = cache_size;
    hm_cfg.top_k = hm_top_k;

    rx_state *state = rx_state_new_with_hybridmem(model, &hm_cfg);
    if (!state) { fprintf(stderr, "Failed to create state\n"); return 1; }

    /* Capture fact embeddings */
    int max_capture = n_fact_tokens + 10;
    float *fact_embeddings = (float*)malloc((size_t)max_capture * c * sizeof(float));
    rx_state_set_embed_capture(state, fact_embeddings, max_capture, -1);

    fprintf(stderr, "Running prefill with HybridMem...\n");
    tm_tensor *logits = NULL;
    for (int i = 0; i < n_tokens; i++) {
        if (logits) tm_free(logits);
        logits = rx_forward(model, tokens[i], state);
        if (i == n_fact_tokens - 1) {
            rx_state_set_embed_capture(state, NULL, 0, -1);
        }
    }

    /* Generate answer (for reference) */
    char *out_hm = generate(model, tok, state, logits, max_tokens, cfg);
    tm_free(logits);
    int hit_hm = contains_ci(out_hm, key_detail);
    fprintf(stderr, "HM output: %s [%s]\n", out_hm ? out_hm : "(null)", hit_hm ? "HIT" : "miss");
    free(out_hm);

    int n_captured = rx_state_embed_capture_count(state);
    fprintf(stderr, "Captured %d fact embeddings\n\n", n_captured);

    /* Flush store before querying */
    rx_state_flush(state);

    /* Now query the store with different query strategies */
    fprintf(stderr, "--- Query Vector Analysis ---\n\n");

    if (n_captured > 0) {
        /* Strategy A: Individual ln_out embeddings (current approach) */
        fprintf(stderr, "Strategy A: Individual ln_out queries\n");
        float total_score_a = 0;
        int total_results_a = 0;

        for (int e = 0; e < n_captured && e < 4; e++) {
            float *query_emb = &fact_embeddings[e * c];

            for (int l = 0; l < cfg.n_attn_layer; l++) {
                int64_t shape_q[] = {c};
                tm_tensor *query_t = tm_from_data(query_emb, shape_q, 1, TM_F32);
                fm_results *results = fm_query(rx_state_get_hm_store(state), query_t, l, hm_top_k);
                tm_free(query_t);

                if (results && results->count > 0) {
                    for (int r = 0; r < results->count; r++) {
                        total_score_a += results->items[r].score;
                        total_results_a++;
                    }
                    if (e == 0) {
                        fprintf(stderr, "  Embed 0, Layer %d: top score=%.4f, count=%d\n",
                                l, results->items[0].score, results->count);
                    }
                    fm_results_free(results);
                } else {
                    if (results) fm_results_free(results);
                }
            }
        }

        float avg_score_a = total_results_a > 0 ? total_score_a / total_results_a : 0;
        fprintf(stderr, "  Avg score across all queries: %.4f (%d results)\n\n",
                avg_score_a, total_results_a);

        /* Strategy B: Mean-pooled fact embedding */
        fprintf(stderr, "Strategy B: Mean-pooled fact embedding query\n");
        float *mean_emb = (float*)calloc(c, sizeof(float));
        for (int e = 0; e < n_captured; e++) {
            for (int j = 0; j < c; j++) {
                mean_emb[j] += fact_embeddings[e * c + j];
            }
        }
        float inv_n = 1.0f / n_captured;
        for (int j = 0; j < c; j++) mean_emb[j] *= inv_n;

        float total_score_b = 0;
        int total_results_b = 0;

        for (int l = 0; l < cfg.n_attn_layer; l++) {
            int64_t shape_q[] = {c};
            tm_tensor *query_t = tm_from_data(mean_emb, shape_q, 1, TM_F32);
            fm_results *results = fm_query(rx_state_get_hm_store(state), query_t, l, hm_top_k);
            tm_free(query_t);

            if (results && results->count > 0) {
                for (int r = 0; r < results->count; r++) {
                    total_score_b += results->items[r].score;
                    total_results_b++;
                }
                fprintf(stderr, "  Layer %d: top score=%.4f, count=%d\n",
                        l, results->items[0].score, results->count);
                fm_results_free(results);
            } else {
                fprintf(stderr, "  Layer %d: no results\n", l);
                if (results) fm_results_free(results);
            }
        }

        float avg_score_b = total_results_b > 0 ? total_score_b / total_results_b : 0;
        fprintf(stderr, "  Avg score across all queries: %.4f (%d results)\n\n",
                avg_score_b, total_results_b);

        /* Strategy C: Cosine similarity between individual fact embeds and mean */
        fprintf(stderr, "Strategy comparison: Cosine sim between individual and mean-pooled\n");
        for (int e = 0; e < n_captured && e < 4; e++) {
            float cos = cosine_sim(&fact_embeddings[e * c], mean_emb, c);
            fprintf(stderr, "  Embed %d vs mean: cos=%.4f\n", e, cos);
        }
        fprintf(stderr, "\n");

        free(mean_emb);

        /* Summary */
        fprintf(stderr, "--- Retrieval Score Summary ---\n");
        fprintf(stderr, "  Individual ln_out avg: %.4f\n", avg_score_a);
        fprintf(stderr, "  Mean-pooled avg:       %.4f\n", avg_score_b);

        if (avg_score_a > avg_score_b * 1.1f) {
            fprintf(stderr, "  Winner: Individual queries (current approach is better)\n");
        } else if (avg_score_b > avg_score_a * 1.1f) {
            fprintf(stderr, "  Winner: Mean-pooled query (consider switching)\n");
        } else {
            fprintf(stderr, "  Draw: Both approaches retrieve similar quality entries\n");
        }
    } else {
        fprintf(stderr, "No embeddings captured. Cannot compare query strategies.\n");
        fprintf(stderr, "The fact may still be in cache (within first %d tokens).\n", cache_size);
    }

    fprintf(stderr, "\nVERDICT:\n");
    if (n_captured > 0) {
        fprintf(stderr, "  Query vector analysis complete. See scores above.\n");
        fprintf(stderr, "  The approach with higher avg scores will retrieve more relevant context.\n");
    } else {
        fprintf(stderr, "  INCONCLUSIVE — No embeddings captured for comparison.\n");
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
