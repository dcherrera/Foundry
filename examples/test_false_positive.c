/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_false_positive.c — HDC Memory Viability Test 8
 *
 * False Positive Impact
 *
 * Method:
 *   3 conditions on the same needle-in-haystack prompt:
 *     A: No HybridMem (baseline)
 *     B: Normal HybridMem (real retrieval)
 *     C: Random retrieval HybridMem (stores normally, but queries return
 *        random entries instead of HDC-matched ones)
 *
 *   If C performs similarly to B, then HDC matching is irrelevant.
 *   If C is worse than A, false positives are harmful.
 *   If B >> C > A, retrieval is helpful but quality matters less.
 *   If B >> A >= C, HDC matching quality is critical.
 *
 * Usage:
 *   ./examples/test_false_positive \
 *     --model ../models/base/RWKV-X-3.6B-64k-Base.safetensors \
 *     --vocab rwkvx/data/rwkv_vocab_v20230424.txt \
 *     --filler ../docs/architecture/rwkv-x-deep-dive.md
 */

#include "rwkvx/rwkvx.h"
#include "hybridmem/hybridmem.h"
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

/* Run a single prefill + generate pass. Returns whether key_detail was found. */
static int run_pass(rx_model *model, rx_tokenizer *tok, rx_config cfg,
                    uint32_t *tokens, int n_tokens, int max_gen_tokens,
                    rx_hybridmem_config *hm_cfg, const char *key_detail,
                    const char *label) {
    rx_state *state = rx_state_new_with_hybridmem(model, hm_cfg);
    if (!state) {
        fprintf(stderr, "  %s: failed to create state\n", label);
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
    fprintf(stderr, "  %s: %s [%s]\n", label, out ? out : "(null)", hit ? "HIT" : "miss");
    free(out);
    return hit;
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
                "  --cache-size N     KV cache slots per attn layer (default: 8)\n"
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

    /* Load filler */
    char *filler_raw = read_file(filler_path);
    if (!filler_raw) { fprintf(stderr, "Failed to read filler\n"); return 1; }
    if ((int)strlen(filler_raw) > max_filler_chars)
        filler_raw[max_filler_chars] = '\0';

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "HDC Memory Viability — Test 8\n");
    fprintf(stderr, "False Positive Impact\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "3 conditions: A=no_HM, B=normal_HM, C=random_retrieval\n\n");

    int hits_a = 0, hits_b = 0, hits_c = 0;

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

        /* Condition A: no HybridMem */
        {
            rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
            hm_cfg.store_dir = NULL;
            hm_cfg.max_cache_len = cache_size;
            hits_a += run_pass(model, tok, cfg, tokens, n_tokens, max_tokens,
                              &hm_cfg, fe->key_detail, "A (no HM)   ");
        }

        /* Condition B: normal HybridMem */
        {
            const char *hm_dir = "/tmp/hm_false_positive_b";
            rmdir_recursive(hm_dir);
            rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
            hm_cfg.store_dir = hm_dir;
            hm_cfg.max_cache_len = cache_size;
            hm_cfg.top_k = hm_top_k;
            hits_b += run_pass(model, tok, cfg, tokens, n_tokens, max_tokens,
                              &hm_cfg, fe->key_detail, "B (normal HM)");
            rmdir_recursive(hm_dir);
        }

        /* Condition C: "random retrieval" HybridMem.
         *
         * Implementation approach: We simulate random retrieval by using a
         * completely different store for retrieval than the one used for storage.
         * We create an HM state, prefill with a DIFFERENT prompt (unrelated text)
         * to populate the store, then use the SAME store dir for the actual prompt.
         * This means the retrieved entries will be from a different context entirely.
         *
         * Simpler approach: Just use a very different prompt as the source,
         * then re-use the store with our actual prompt. This creates "wrong
         * context" retrieval which is the same as false positives.
         */
        {
            const char *hm_dir = "/tmp/hm_false_positive_c";
            rmdir_recursive(hm_dir);

            /* Step 1: Populate store with unrelated text */
            rx_hybridmem_config hm_cfg = rx_hybridmem_config_defaults();
            hm_cfg.store_dir = hm_dir;
            hm_cfg.max_cache_len = cache_size;
            hm_cfg.top_k = hm_top_k;

            /* Create a "poison" prompt — completely unrelated */
            const char *poison = "Mathematics is the study of numbers, shapes, and patterns. "
                "The Pythagorean theorem states that a squared plus b squared equals c squared. "
                "Calculus was developed independently by Newton and Leibniz in the 17th century. "
                "Linear algebra deals with vector spaces and linear transformations. "
                "Statistics provides methods for data analysis and inference. "
                "Number theory explores the properties of integers and prime numbers. "
                "Topology studies properties preserved under continuous deformation. "
                "Group theory provides a framework for studying symmetry. "
                "Differential equations model rates of change in physical systems. "
                "Probability theory quantifies uncertainty in random processes.";

            /* Repeat poison to fill cache and force evictions to store */
            int poison_len = (int)strlen(poison);
            int repeat_len = poison_len * 4 + 4 + 1;
            char *poison_long = (char*)malloc(repeat_len);
            int pos = 0;
            for (int r = 0; r < 4; r++) {
                memcpy(poison_long + pos, poison, poison_len);
                pos += poison_len;
                poison_long[pos++] = ' ';
            }
            poison_long[pos] = '\0';

            uint32_t *poison_tokens = (uint32_t*)malloc(16384 * sizeof(uint32_t));
            int n_poison = rx_tokenizer_encode(tok, poison_long, poison_tokens, 16384);

            rx_state *state_poison = rx_state_new_with_hybridmem(model, &hm_cfg);
            tm_tensor *logits = NULL;
            for (int i = 0; i < n_poison; i++) {
                if (logits) tm_free(logits);
                logits = rx_forward(model, poison_tokens[i], state_poison);
            }
            if (logits) tm_free(logits);
            rx_state_flush(state_poison);
            rx_state_free(state_poison);
            free(poison_long);
            free(poison_tokens);

            /* Step 2: Now run the real prompt with the store pre-filled with junk */
            hits_c += run_pass(model, tok, cfg, tokens, n_tokens, max_tokens,
                              &hm_cfg, fe->key_detail, "C (random HM)");
            rmdir_recursive(hm_dir);
        }

        free(prompt);
        free(tokens);
        fprintf(stderr, "\n");
    }

    /* Summary */
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "SUMMARY (%d facts)\n", N_FACTS);
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "A (no HM):        %d/%d hits\n", hits_a, N_FACTS);
    fprintf(stderr, "B (normal HM):    %d/%d hits\n", hits_b, N_FACTS);
    fprintf(stderr, "C (random HM):    %d/%d hits\n\n", hits_c, N_FACTS);

    fprintf(stderr, "VERDICT:\n");
    if (hits_b > hits_a && hits_b > hits_c) {
        fprintf(stderr, "  PASS — Normal HM (%d) > no HM (%d) and random (%d)\n",
                hits_b, hits_a, hits_c);
        fprintf(stderr, "  HDC matching quality matters for retrieval benefit.\n");
    } else if (hits_b > hits_a && hits_c >= hits_b) {
        fprintf(stderr, "  INTERESTING — Random retrieval matches normal HM\n");
        fprintf(stderr, "  Any extra KV context helps, but HDC matching isn't critical.\n");
    } else if (hits_c < hits_a) {
        fprintf(stderr, "  WARNING — Random retrieval (%d) is worse than no HM (%d)\n",
                hits_c, hits_a);
        fprintf(stderr, "  False positives are actively harmful.\n");
    } else if (hits_b <= hits_a) {
        fprintf(stderr, "  FAIL — Normal HM doesn't improve over baseline\n");
    } else {
        fprintf(stderr, "  MIXED — No clear pattern. Inspect per-fact results.\n");
    }

    /* Cleanup */
    free(filler_raw);
    rx_model_free(model);
    rx_tokenizer_free(tok);

    fprintf(stderr, "\n==========================================\n");
    return 0;
}
