/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_filler_sensitivity.c — HDC Memory Viability Test 6
 *
 * Filler Sensitivity
 *
 * Method:
 *   Same AURORA-7 fact, 5 different filler types.
 *   A/B test (no HM vs HM) per filler type, cache_size=8.
 *   Binary scoring: output contains "AURORA"?
 *
 * Usage:
 *   ./examples/test_filler_sensitivity \
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

/* Generate filler text of given type and approximate char length */
static char* generate_filler(const char *type, const char *file_filler, int target_chars) {
    char *filler = NULL;

    if (strcmp(type, "prose") == 0) {
        /* Use file-based filler (real prose) */
        if (file_filler) {
            int len = (int)strlen(file_filler);
            int use = len < target_chars ? len : target_chars;
            filler = (char*)malloc(use + 1);
            memcpy(filler, file_filler, use);
            filler[use] = '\0';
        } else {
            filler = strdup("The architecture of modern neural networks has evolved significantly over the past decade, with transformer-based models achieving remarkable performance across diverse tasks including natural language processing, computer vision, and multimodal reasoning.");
        }
    } else if (strcmp(type, "code") == 0) {
        /* Code-like filler */
        int len = target_chars + 128;
        filler = (char*)malloc(len);
        int pos = 0;
        const char *snippets[] = {
            "def compute_loss(predictions, targets):\n    loss = 0.0\n    for i in range(len(predictions)):\n        diff = predictions[i] - targets[i]\n        loss += diff * diff\n    return loss / len(predictions)\n\n",
            "class DataLoader:\n    def __init__(self, data, batch_size=32):\n        self.data = data\n        self.batch_size = batch_size\n        self.idx = 0\n\n    def next_batch(self):\n        batch = self.data[self.idx:self.idx+self.batch_size]\n        self.idx += self.batch_size\n        return batch\n\n",
            "import numpy as np\nfrom typing import List, Tuple\n\ndef normalize(vec: np.ndarray) -> np.ndarray:\n    norm = np.linalg.norm(vec)\n    return vec / (norm + 1e-8)\n\ndef cosine_similarity(a, b):\n    return np.dot(normalize(a), normalize(b))\n\n",
            "# Configuration parameters\nMAX_SEQUENCE_LENGTH = 4096\nHIDDEN_SIZE = 768\nNUM_HEADS = 12\nNUM_LAYERS = 24\nDROPOUT_RATE = 0.1\nLEARNING_RATE = 3e-4\nWARMUP_STEPS = 1000\n\n"
        };
        int n_snippets = sizeof(snippets) / sizeof(snippets[0]);
        for (int i = 0; pos < target_chars && i < n_snippets * 10; i++) {
            const char *s = snippets[i % n_snippets];
            int slen = (int)strlen(s);
            if (pos + slen >= len - 1) break;
            memcpy(filler + pos, s, slen);
            pos += slen;
        }
        filler[pos < target_chars ? pos : target_chars] = '\0';
    } else if (strcmp(type, "numbers") == 0) {
        /* Numeric/tabular filler */
        int len = target_chars + 128;
        filler = (char*)malloc(len);
        int pos = 0;
        pos += snprintf(filler + pos, len - pos,
            "| ID | Value | Category | Score |\n"
            "|-----|-------|----------|-------|\n");
        for (int i = 0; pos < target_chars && i < 200; i++) {
            pos += snprintf(filler + pos, len - pos,
                "| %04d | %.4f | CAT-%c | %d.%02d |\n",
                1000 + i, (float)(i * 17 % 100) / 10.0f,
                'A' + (i % 26), i % 10, (i * 7) % 100);
        }
        filler[pos < target_chars ? pos : target_chars] = '\0';
    } else if (strcmp(type, "repetitive") == 0) {
        /* Highly repetitive filler */
        const char *unit = "The quick brown fox jumps over the lazy dog. ";
        int ulen = (int)strlen(unit);
        int reps = target_chars / ulen + 1;
        filler = (char*)malloc(reps * ulen + 1);
        int pos = 0;
        for (int i = 0; i < reps && pos < target_chars; i++) {
            memcpy(filler + pos, unit, ulen);
            pos += ulen;
        }
        filler[pos < target_chars ? pos : target_chars] = '\0';
    } else if (strcmp(type, "mixed") == 0) {
        /* Mixed: some prose, some numbers, some code */
        int third = target_chars / 3;
        char *a = generate_filler("prose", file_filler, third);
        char *b = generate_filler("numbers", NULL, third);
        char *c = generate_filler("code", NULL, third);
        int total = (int)(strlen(a) + strlen(b) + strlen(c) + 8);
        filler = (char*)malloc(total);
        snprintf(filler, total, "%s\n\n%s\n\n%s", a, b, c);
        if ((int)strlen(filler) > target_chars)
            filler[target_chars] = '\0';
        free(a); free(b); free(c);
    } else {
        filler = strdup("Unknown filler type.");
    }
    return filler;
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

    if (!model_path || !vocab_path) {
        fprintf(stderr, "Error: --model and --vocab are required\n");
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

    /* Load file-based filler (for prose type) */
    char *file_filler = NULL;
    if (filler_path) {
        file_filler = read_file(filler_path);
        if (!file_filler)
            fprintf(stderr, "Warning: could not read filler file %s\n", filler_path);
    }

    const char *fact = "The secret project codename is AURORA-7 and the lead researcher is Dr. Kimura.";
    const char *question = "\n\nQuestion: What is the secret project codename and who leads it?\nAnswer:";
    const char *key_detail = "AURORA";

    const char *filler_types[] = {"prose", "code", "numbers", "repetitive", "mixed"};
    int n_types = 5;

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "HDC Memory Viability — Test 6\n");
    fprintf(stderr, "Filler Sensitivity\n");
    fprintf(stderr, "==========================================\n\n");

    int improved = 0, same = 0, worse = 0;

    for (int t = 0; t < n_types; t++) {
        const char *ftype = filler_types[t];
        fprintf(stderr, "--- Filler: %s ---\n", ftype);

        /* Generate filler */
        char *filler = generate_filler(ftype, file_filler, max_filler_chars);
        int filler_len = (int)strlen(filler);
        fprintf(stderr, "  Filler length: %d chars\n", filler_len);

        /* Build prompt */
        int pl = (int)(strlen(fact) + 2 + filler_len + strlen(question) + 1);
        char *prompt = (char*)malloc(pl);
        snprintf(prompt, pl, "%s\n\n%s%s", fact, filler, question);

        uint32_t *tokens = (uint32_t*)malloc(32768 * sizeof(uint32_t));
        int n_tokens = rx_tokenizer_encode(tok, prompt, tokens, 32768);
        fprintf(stderr, "  Prompt tokens: %d\n", n_tokens);

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

            int hit_a = contains_ci(out_a, key_detail);
            fprintf(stderr, "  A (no HM): %s [%s]\n",
                    out_a ? out_a : "(null)", hit_a ? "HIT" : "miss");

            /* Run B: with HM */
            const char *hm_dir = "/tmp/hm_filler_test";
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

            int hit_b = contains_ci(out_b, key_detail);
            fprintf(stderr, "  B (HM):    %s [%s]\n",
                    out_b ? out_b : "(null)", hit_b ? "HIT" : "miss");
            free(out_b);
            rmdir_recursive(hm_dir);

            if (hit_b && !hit_a) improved++;
            else if (!hit_b && hit_a) worse++;
            else same++;

            free(out_a);
        }

        free(prompt);
        free(tokens);
        free(filler);
        fprintf(stderr, "\n");
    }

    /* Summary */
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "SUMMARY\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "Improved (B hit, A miss): %d/%d\n", improved, n_types);
    fprintf(stderr, "Same:                     %d/%d\n", same, n_types);
    fprintf(stderr, "Worse (A hit, B miss):    %d/%d\n\n", worse, n_types);

    fprintf(stderr, "VERDICT: ");
    if (improved >= 3) {
        fprintf(stderr, "PASS — HybridMem helps across %d/5 filler types\n", improved);
    } else if (improved >= 1) {
        fprintf(stderr, "MARGINAL — HybridMem helps for %d/5 filler types\n", improved);
    } else if (worse == 0 && same == n_types) {
        fprintf(stderr, "NEUTRAL — No difference across filler types (both may succeed or fail)\n");
    } else {
        fprintf(stderr, "FAIL — HybridMem doesn't help across filler types\n");
    }
    if (worse > 0) {
        fprintf(stderr, "WARNING: %d/5 WORSE with HybridMem\n", worse);
    }

    /* Cleanup */
    free(file_filler);
    rx_model_free(model);
    rx_tokenizer_free(tok);

    fprintf(stderr, "\n==========================================\n");
    return 0;
}
