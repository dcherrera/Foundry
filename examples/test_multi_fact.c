/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_multi_fact.c — HDC Memory Viability Test 5
 *
 * Multi-Fact Battery
 *
 * Method:
 *   10 facts, each with matching question.
 *   For each fact: A/B test (no HM vs HM), cache_size=8.
 *   Binary scoring: output contains specific detail?
 *   Summary: X/10 improved, Y/10 same, Z/10 worse.
 *
 * Usage:
 *   ./examples/test_multi_fact \
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

typedef struct {
    const char *fact;
    const char *question;
    const char *key_detail;  /* substring to check in output */
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
    },
    {
        "The annual budget for Project Helios is 2.7 million dollars, allocated by Director Williams.",
        "What is the annual budget for Project Helios?",
        "2.7"
    },
    {
        "The backup protocol requires data replication to three separate data centers: Tokyo, Frankfurt, and Sao Paulo.",
        "Which three data centers are used for backup replication?",
        "Tokyo"
    },
    {
        "The new encryption standard uses a 384-bit key length with the codename Obsidian Shield.",
        "What key length does the Obsidian Shield encryption use?",
        "384"
    },
    {
        "Dr. Patel discovered that compound X-891 reduces neural inflammation by 67 percent in clinical trials.",
        "What percentage does compound X-891 reduce neural inflammation?",
        "67"
    },
    {
        "The satellite launch window for Mission Vanguard is March 15 to March 22, from Pad 39B.",
        "When is the launch window for Mission Vanguard?",
        "March"
    },
    {
        "The quantum processor has 127 qubits and operates at 15 millikelvin, manufactured by Rigetti.",
        "How many qubits does the quantum processor have?",
        "127"
    },
    {
        "The ancient manuscript, catalogued as MS-7749, was found in a monastery near Thessaloniki in 1987.",
        "Where was manuscript MS-7749 found?",
        "Thessaloniki"
    }
};
#define N_FACTS 10

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
    if ((int)strlen(filler_raw) > max_filler_chars)
        filler_raw[max_filler_chars] = '\0';

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "HDC Memory Viability — Test 5\n");
    fprintf(stderr, "Multi-Fact Battery (%d facts)\n", N_FACTS);
    fprintf(stderr, "==========================================\n\n");

    int improved = 0, same = 0, worse = 0;

    for (int f = 0; f < N_FACTS; f++) {
        const fact_entry *fe = &FACTS[f];
        fprintf(stderr, "--- Fact %d: \"%s\" ---\n", f + 1, fe->key_detail);

        /* Build prompt */
        char question_full[512];
        snprintf(question_full, sizeof(question_full),
                 "\n\nQuestion: %s\nAnswer:", fe->question);

        int pl = (int)(strlen(fe->fact) + 2 + strlen(filler_raw) + strlen(question_full) + 1);
        char *prompt = (char*)malloc(pl);
        snprintf(prompt, pl, "%s\n\n%s%s", fe->fact, filler_raw, question_full);

        uint32_t *tokens = (uint32_t*)malloc(32768 * sizeof(uint32_t));
        int n_tokens = rx_tokenizer_encode(tok, prompt, tokens, 32768);

        /* Run A: no HM */
        {
            rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
            hm_cfg.store_dir = NULL;
            hm_cfg.max_cache_len = cache_size;
            rx_state *state = rx_state_new_with_hybridmem(model, &hm_cfg);

            tm_tensor *logits = NULL;
            for (int i = 0; i < n_tokens; i++) {
                if (logits) tm_free(logits);
                logits = rx_forward(model, tokens[i], state);
            }
            char *out_a = generate(model, tok, state, logits, max_tokens, cfg);
            tm_free(logits);
            rx_state_free(state);

            int hit_a = contains_ci(out_a, fe->key_detail);
            fprintf(stderr, "  A (no HM): %s [%s]\n",
                    out_a ? out_a : "(null)", hit_a ? "HIT" : "miss");
            free(out_a);

            /* Run B: with HM */
            const char *hm_dir = "/tmp/hm_multi_fact";
            rmdir_recursive(hm_dir);

            hm_cfg.store_dir = hm_dir;
            hm_cfg.max_cache_len = cache_size;
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

            int hit_b = contains_ci(out_b, fe->key_detail);
            fprintf(stderr, "  B (HM):    %s [%s]\n",
                    out_b ? out_b : "(null)", hit_b ? "HIT" : "miss");
            free(out_b);
            rmdir_recursive(hm_dir);

            if (hit_b && !hit_a) improved++;
            else if (!hit_b && hit_a) worse++;
            else same++;
        }

        free(prompt);
        free(tokens);
        fprintf(stderr, "\n");
    }

    /* Summary */
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "SUMMARY\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "Improved (B hit, A miss): %d/%d\n", improved, N_FACTS);
    fprintf(stderr, "Same:                     %d/%d\n", same, N_FACTS);
    fprintf(stderr, "Worse (A hit, B miss):    %d/%d\n\n", worse, N_FACTS);

    float gen_rate = (float)(improved + same) / N_FACTS * 100.0f;
    fprintf(stderr, "VERDICT: ");
    if (improved >= 6) {
        fprintf(stderr, "PASS — %d/10 improved, generalization = %.0f%%\n", improved, gen_rate);
    } else if (improved >= 3) {
        fprintf(stderr, "MARGINAL — %d/10 improved, check which facts fail\n", improved);
    } else {
        fprintf(stderr, "FAIL — %d/10 improved, HybridMem doesn't generalize\n", improved);
    }
    if (worse > 0) {
        fprintf(stderr, "WARNING: %d/10 WORSE with HybridMem — false retrieval may be harmful\n",
                worse);
    }

    /* Cleanup */
    free(filler_raw);
    rx_model_free(model);
    rx_tokenizer_free(tok);

    fprintf(stderr, "\n==========================================\n");
    return 0;
}
