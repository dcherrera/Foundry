/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_contiguous_chunk.c — Contiguous Chunk Retrieval Experiments
 *
 * Follow-up to HybridMem Phase 7b (B=C=2/3).
 * Tests whether fragment count (seams between chunks from different contexts)
 * is the root cause of retrieval failure.
 *
 * Experiment 1: top_k sweep — fewer chunks = fewer seams = better?
 * Experiment 2: evict_fraction sweep — larger chunks = more coherence = better?
 * Experiment 3: best contiguous config vs Phase 7b baseline
 *
 * Usage:
 *   ./examples/test_contiguous_chunk \
 *     --model ../models/base/RWKV-X-3.6B-64k-Base.safetensors \
 *     --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
 *     --filler ../docs/architecture/rwkv-x-deep-dive.md \
 *     --max-filler 8000 --max-tokens 48 --cache-size 8 --quantize
 */

#include "rwkvx/rwkvx.h"
#include "tensor/tensor.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *fact;
    const char *question;
    const char *key_detail;
} fact_entry;

static const fact_entry FACTS[] = {
    {
        "The secret project codename is AURORA-7 and the lead researcher is Dr. Kimura.",
        "What is the secret project codename and who leads it?",
        "AURORA"
    },
    {
        "The server room temperature must be maintained at exactly 18.5 degrees Celsius.",
        "What temperature must the server room be maintained at?",
        "18.5"
    },
    {
        "Employee badge number 4472 belongs to Sarah Chen from the quantum computing division.",
        "What is Sarah Chen's badge number?",
        "4472"
    }
};
#define N_FACTS 3

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

/* Run a single test pass with given HybridMem config.
 * Returns 1 if key_detail found in output, 0 otherwise. */
static int run_pass(rx_model *model, rx_tokenizer *tok, rx_config cfg,
                    uint32_t *tokens, int n_tokens, int max_gen_tokens,
                    rx_hybridmem_config *hm_cfg, const char *key_detail,
                    const char *label) {
    rx_state *state = rx_state_new_with_hybridmem(model, hm_cfg);
    if (!state) {
        fprintf(stderr, "    %s: failed to create state\n", label);
        return 0;
    }

    tm_tensor *logits = NULL;
    for (int i = 0; i < n_tokens; i++) {
        if (logits) tm_free(logits);
        logits = rx_forward(model, tokens[i], state);
    }
    char *out = generate(model, tok, state, logits, max_gen_tokens, cfg);
    tm_free(logits);
    rx_state_free(state);

    int hit = contains_ci(out, key_detail);
    fprintf(stderr, "    %s: %s [%s]\n", label, out ? out : "(null)",
            hit ? "HIT" : "miss");
    free(out);
    return hit;
}

/* Build prompt: fact + filler + question */
static char* build_prompt(const fact_entry *fe, const char *filler) {
    char question_full[512];
    snprintf(question_full, sizeof(question_full),
             "\n\nQuestion: %s\nAnswer:", fe->question);

    int pl = (int)(strlen(fe->fact) + 2 + strlen(filler) + strlen(question_full) + 1);
    char *prompt = (char*)malloc(pl);
    snprintf(prompt, pl, "%s\n\n%s%s", fe->fact, filler, question_full);
    return prompt;
}

/* ================================================================
 * Experiment 1: top_k sweep
 * ================================================================ */

static void experiment_1(rx_model *model, rx_tokenizer *tok, rx_config cfg,
                         const char *filler, int cache_size, int max_tokens) {
    int top_k_values[] = {0, 1, 2, 3, 5};
    int n_tk = 5;
    float evict_fraction = 0.5f;

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "EXPERIMENT 1: top_k Sweep (Fragment Count)\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "cache_size=%d, evict_fraction=%.2f\n", cache_size, evict_fraction);
    fprintf(stderr, "top_k values: 0 (baseline), 1, 2, 3, 5\n\n");

    int hits[5][N_FACTS];
    memset(hits, 0, sizeof(hits));

    for (int f = 0; f < N_FACTS; f++) {
        const fact_entry *fe = &FACTS[f];
        fprintf(stderr, "--- Fact %d: \"%s\" ---\n", f + 1, fe->key_detail);

        char *prompt = build_prompt(fe, filler);
        uint32_t *tokens = (uint32_t*)malloc(32768 * sizeof(uint32_t));
        int n_tokens = rx_tokenizer_encode(tok, prompt, tokens, 32768);
        fprintf(stderr, "  Prompt: %d tokens\n", n_tokens);

        for (int ti = 0; ti < n_tk; ti++) {
            int tk = top_k_values[ti];
            char label[64];
            char hm_dir[256];
            snprintf(hm_dir, sizeof(hm_dir), "/tmp/hm_contiguous_exp1_%d_%d", f, tk);
            rmdir_recursive(hm_dir);

            rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
            if (tk == 0) {
                hm_cfg.store_dir = NULL;
                snprintf(label, sizeof(label), "top_k=0 (no HM)");
            } else {
                hm_cfg.store_dir = hm_dir;
                hm_cfg.top_k = tk;
                hm_cfg.evict_fraction = evict_fraction;
                snprintf(label, sizeof(label), "top_k=%d         ", tk);
            }
            hm_cfg.max_cache_len = cache_size;

            hits[ti][f] = run_pass(model, tok, cfg, tokens, n_tokens,
                                   max_tokens, &hm_cfg, fe->key_detail, label);
            rmdir_recursive(hm_dir);
        }
        free(prompt);
        free(tokens);
        fprintf(stderr, "\n");
    }

    /* Summary table */
    fprintf(stderr, "--- Experiment 1 Summary ---\n");
    fprintf(stderr, "%-20s", "Fact");
    for (int ti = 0; ti < n_tk; ti++)
        fprintf(stderr, "  tk=%d", top_k_values[ti]);
    fprintf(stderr, "\n");

    int totals[5] = {0};
    for (int f = 0; f < N_FACTS; f++) {
        fprintf(stderr, "%-20s", FACTS[f].key_detail);
        for (int ti = 0; ti < n_tk; ti++) {
            fprintf(stderr, "  %s  ", hits[ti][f] ? "HIT" : "miss");
            totals[ti] += hits[ti][f];
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "%-20s", "TOTAL");
    for (int ti = 0; ti < n_tk; ti++)
        fprintf(stderr, "  %d/%d ", totals[ti], N_FACTS);
    fprintf(stderr, "\n\n");

    /* Verdict */
    fprintf(stderr, "VERDICT: ");
    if (totals[1] >= totals[4]) {
        fprintf(stderr, "PASS — top_k=1 (%d) >= top_k=5 (%d). Fewer seams helps.\n",
                totals[1], totals[4]);
    } else {
        fprintf(stderr, "FAIL — top_k=5 (%d) > top_k=1 (%d). More entries helps more than coherence.\n",
                totals[4], totals[1]);
    }
}

/* ================================================================
 * Experiment 2: evict_fraction sweep
 * ================================================================ */

static void experiment_2(rx_model *model, rx_tokenizer *tok, rx_config cfg,
                         const char *filler, int cache_size, int max_tokens) {
    float ef_values[] = {0.25f, 0.5f, 0.75f, 1.0f};
    int n_ef = 4;
    int top_k = 1;

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "EXPERIMENT 2: Chunk Size Sweep\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "cache_size=%d, top_k=%d\n", cache_size, top_k);
    fprintf(stderr, "evict_fraction values: 0.25 (%d tok), 0.5 (%d tok), 0.75 (%d tok), 1.0 (%d tok)\n\n",
            (int)(cache_size * 0.25f), (int)(cache_size * 0.5f),
            (int)(cache_size * 0.75f), cache_size);

    int hits[4][N_FACTS];
    memset(hits, 0, sizeof(hits));

    for (int f = 0; f < N_FACTS; f++) {
        const fact_entry *fe = &FACTS[f];
        fprintf(stderr, "--- Fact %d: \"%s\" ---\n", f + 1, fe->key_detail);

        char *prompt = build_prompt(fe, filler);
        uint32_t *tokens = (uint32_t*)malloc(32768 * sizeof(uint32_t));
        int n_tokens = rx_tokenizer_encode(tok, prompt, tokens, 32768);
        fprintf(stderr, "  Prompt: %d tokens\n", n_tokens);

        for (int ei = 0; ei < n_ef; ei++) {
            float ef = ef_values[ei];
            int evict_count = (int)(cache_size * ef);
            if (evict_count < 1) evict_count = 1;

            char label[64];
            char hm_dir[256];
            snprintf(hm_dir, sizeof(hm_dir), "/tmp/hm_contiguous_exp2_%d_%d", f, ei);
            rmdir_recursive(hm_dir);

            rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
            hm_cfg.store_dir = hm_dir;
            hm_cfg.max_cache_len = cache_size;
            hm_cfg.top_k = top_k;
            hm_cfg.evict_fraction = ef;
            snprintf(label, sizeof(label), "ef=%.2f (%d tok)", ef, evict_count);

            hits[ei][f] = run_pass(model, tok, cfg, tokens, n_tokens,
                                   max_tokens, &hm_cfg, fe->key_detail, label);
            rmdir_recursive(hm_dir);
        }
        free(prompt);
        free(tokens);
        fprintf(stderr, "\n");
    }

    /* Summary table */
    fprintf(stderr, "--- Experiment 2 Summary ---\n");
    fprintf(stderr, "%-20s", "Fact");
    for (int ei = 0; ei < n_ef; ei++)
        fprintf(stderr, "  ef=%.2f", ef_values[ei]);
    fprintf(stderr, "\n");

    int totals[4] = {0};
    for (int f = 0; f < N_FACTS; f++) {
        fprintf(stderr, "%-20s", FACTS[f].key_detail);
        for (int ei = 0; ei < n_ef; ei++) {
            fprintf(stderr, "  %s   ", hits[ei][f] ? "HIT" : "miss");
            totals[ei] += hits[ei][f];
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "%-20s", "TOTAL");
    for (int ei = 0; ei < n_ef; ei++)
        fprintf(stderr, "  %d/%d  ", totals[ei], N_FACTS);
    fprintf(stderr, "\n\n");

    /* Verdict */
    fprintf(stderr, "VERDICT: ");
    if (totals[3] >= totals[0] && totals[3] >= totals[1]) {
        fprintf(stderr, "PASS — ef=1.0 (%d) >= smaller chunks. Larger chunks help.\n", totals[3]);
    } else if (totals[0] == totals[1] && totals[1] == totals[2] && totals[2] == totals[3]) {
        fprintf(stderr, "INCONCLUSIVE — all chunk sizes tied at %d/%d.\n", totals[0], N_FACTS);
    } else {
        fprintf(stderr, "MIXED — no clear trend. ef=0.25:%d, 0.5:%d, 0.75:%d, 1.0:%d\n",
                totals[0], totals[1], totals[2], totals[3]);
    }
}

/* ================================================================
 * Experiment 3: Best contiguous vs Phase 7b baseline
 * ================================================================ */

static void experiment_3(rx_model *model, rx_tokenizer *tok, rx_config cfg,
                         const char *filler, int cache_size, int max_tokens) {
    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "EXPERIMENT 3: Best Contiguous vs Phase 7b\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "A: No HybridMem (baseline)\n");
    fprintf(stderr, "B: Phase 7b config (top_k=5, ef=0.5)\n");
    fprintf(stderr, "C: Best contiguous (top_k=1, ef=1.0)\n\n");

    int hits_a[N_FACTS], hits_b[N_FACTS], hits_c[N_FACTS];
    memset(hits_a, 0, sizeof(hits_a));
    memset(hits_b, 0, sizeof(hits_b));
    memset(hits_c, 0, sizeof(hits_c));

    for (int f = 0; f < N_FACTS; f++) {
        const fact_entry *fe = &FACTS[f];
        fprintf(stderr, "--- Fact %d: \"%s\" ---\n", f + 1, fe->key_detail);

        char *prompt = build_prompt(fe, filler);
        uint32_t *tokens = (uint32_t*)malloc(32768 * sizeof(uint32_t));
        int n_tokens = rx_tokenizer_encode(tok, prompt, tokens, 32768);
        fprintf(stderr, "  Prompt: %d tokens\n", n_tokens);

        /* Condition A: no HybridMem */
        {
            rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
            hm_cfg.store_dir = NULL;
            hm_cfg.max_cache_len = cache_size;
            hits_a[f] = run_pass(model, tok, cfg, tokens, n_tokens,
                                 max_tokens, &hm_cfg, fe->key_detail,
                                 "A (no HM)        ");
        }

        /* Condition B: Phase 7b config */
        {
            const char *hm_dir = "/tmp/hm_contiguous_exp3_b";
            rmdir_recursive(hm_dir);
            rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
            hm_cfg.store_dir = hm_dir;
            hm_cfg.max_cache_len = cache_size;
            hm_cfg.top_k = 5;
            hm_cfg.evict_fraction = 0.5f;
            hits_b[f] = run_pass(model, tok, cfg, tokens, n_tokens,
                                 max_tokens, &hm_cfg, fe->key_detail,
                                 "B (7b: tk=5 ef=.5)");
            rmdir_recursive(hm_dir);
        }

        /* Condition C: best contiguous */
        {
            const char *hm_dir = "/tmp/hm_contiguous_exp3_c";
            rmdir_recursive(hm_dir);
            rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
            hm_cfg.store_dir = hm_dir;
            hm_cfg.max_cache_len = cache_size;
            hm_cfg.top_k = 1;
            hm_cfg.evict_fraction = 1.0f;
            hits_c[f] = run_pass(model, tok, cfg, tokens, n_tokens,
                                 max_tokens, &hm_cfg, fe->key_detail,
                                 "C (tk=1, ef=1.0) ");
            rmdir_recursive(hm_dir);
        }

        free(prompt);
        free(tokens);
        fprintf(stderr, "\n");
    }

    /* Summary */
    int total_a = 0, total_b = 0, total_c = 0;
    fprintf(stderr, "--- Experiment 3 Summary ---\n");
    fprintf(stderr, "%-12s  A(noHM)  B(7b)  C(contig)\n", "Fact");
    for (int f = 0; f < N_FACTS; f++) {
        fprintf(stderr, "%-12s  %s     %s    %s\n", FACTS[f].key_detail,
                hits_a[f] ? "HIT" : "miss",
                hits_b[f] ? "HIT" : "miss",
                hits_c[f] ? "HIT" : "miss");
        total_a += hits_a[f];
        total_b += hits_b[f];
        total_c += hits_c[f];
    }
    fprintf(stderr, "%-12s  %d/%d     %d/%d    %d/%d\n\n", "TOTAL",
            total_a, N_FACTS, total_b, N_FACTS, total_c, N_FACTS);

    fprintf(stderr, "VERDICT: ");
    if (total_c > total_b) {
        fprintf(stderr, "PASS — Contiguous (%d) > Phase 7b (%d). Fragment reduction works!\n",
                total_c, total_b);
    } else if (total_c == total_b) {
        fprintf(stderr, "TIED — Contiguous (%d) = Phase 7b (%d). No improvement from contiguity.\n",
                total_c, total_b);
    } else {
        fprintf(stderr, "FAIL — Phase 7b (%d) > Contiguous (%d). More entries beats coherence.\n",
                total_b, total_c);
    }
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *vocab_path = NULL;
    const char *filler_path = NULL;
    int cache_size = 8;
    int max_filler_chars = 8000;
    int max_tokens = 48;
    int quantize = 0;
    int experiment = 0;  /* 0 = all, 1/2/3 = specific */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc)
            vocab_path = argv[++i];
        else if (strcmp(argv[i], "--filler") == 0 && i + 1 < argc)
            filler_path = argv[++i];
        else if (strcmp(argv[i], "--cache-size") == 0 && i + 1 < argc)
            cache_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-filler") == 0 && i + 1 < argc)
            max_filler_chars = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--experiment") == 0 && i + 1 < argc)
            experiment = atoi(argv[++i]);
        else if (strcmp(argv[i], "--quantize") == 0)
            quantize = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: %s --model PATH --vocab PATH --filler PATH [options]\n"
                "  --cache-size N     KV cache slots per attn layer (default: 8)\n"
                "  --max-filler N     Max chars of filler text (default: 8000)\n"
                "  --max-tokens N     Max generated tokens (default: 48)\n"
                "  --experiment N     Run only experiment N (1, 2, or 3; default: all)\n"
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

    /* Load filler */
    char *filler_raw = read_file(filler_path);
    if (!filler_raw) { fprintf(stderr, "Failed to read filler: %s\n", filler_path); return 1; }
    if ((int)strlen(filler_raw) > max_filler_chars)
        filler_raw[max_filler_chars] = '\0';

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "Contiguous Chunk Retrieval Experiments\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "Model: %s%s\n", model_path, quantize ? " (INT8)" : "");
    fprintf(stderr, "Filler: %d chars\n", (int)strlen(filler_raw));
    fprintf(stderr, "Cache size: %d, Max gen tokens: %d\n", cache_size, max_tokens);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    if (experiment == 0 || experiment == 1)
        experiment_1(model, tok, cfg, filler_raw, cache_size, max_tokens);

    if (experiment == 0 || experiment == 2)
        experiment_2(model, tok, cfg, filler_raw, cache_size, max_tokens);

    if (experiment == 0 || experiment == 3)
        experiment_3(model, tok, cfg, filler_raw, cache_size, max_tokens);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                     (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "Total wall time: %.1f min\n", elapsed / 60.0);
    fprintf(stderr, "==========================================\n");

    /* Cleanup */
    free(filler_raw);
    rx_model_free(model);
    rx_tokenizer_free(tok);
    return 0;
}
