/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_hdc_correlation.c — HDC Memory Viability Test 3
 *
 * Verifies HDC Hamming similarity correlates with cosine similarity
 * on real RWKV-X embeddings (not synthetic vectors).
 *
 * Method:
 *   1. Load 0.2B model + tokenizer
 *   2. Encode a ~500 token prompt
 *   3. Run prefill, capturing ln_out embeddings from attention layers
 *   4. For ~200 random pairs:
 *      a. Compute cosine similarity (float space)
 *      b. Project via fv_project, compute HDC Hamming similarity
 *   5. Compute Pearson correlation between the two
 *
 * Success: Pearson r > 0.85. Cosine spread across [0.0, 1.0].
 *
 * Usage:
 *   ./examples/test_hdc_correlation \
 *     --model ../models/base/RWKV-X-0.2B-64k-Base.safetensors \
 *     --vocab rwkvx/data/rwkv_vocab_v20230424.txt
 */

#include "rwkvx/rwkvx.h"
#include "hdc/hdc.h"
#include "tensor/tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Simple PRNG for reproducible pair selection */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Cosine similarity between two float vectors */
static float cosine_sim(const float *a, const float *b, int dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return denom > 1e-12f ? dot / denom : 0.0f;
}

/* Pearson correlation between two float arrays */
static float pearson(const float *x, const float *y, int n) {
    float mx = 0, my = 0;
    for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;

    float cov = 0, sx = 0, sy = 0;
    for (int i = 0; i < n; i++) {
        float dx = x[i] - mx;
        float dy = y[i] - my;
        cov += dx * dy;
        sx += dx * dx;
        sy += dy * dy;
    }
    float denom = sqrtf(sx) * sqrtf(sy);
    return denom > 1e-12f ? cov / denom : 0.0f;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *vocab_path = NULL;
    int max_pairs = 200;
    int capture_layer = 0;  /* which attention layer to capture from */
    int hdc_dim = 4096;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc)
            vocab_path = argv[++i];
        else if (strcmp(argv[i], "--pairs") == 0 && i + 1 < argc)
            max_pairs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc)
            capture_layer = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hdc-dim") == 0 && i + 1 < argc)
            hdc_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: %s --model PATH --vocab PATH [options]\n"
                "  --pairs N       Number of random pairs to compare (default: 200)\n"
                "  --layer N       Attention layer to capture from (default: 0)\n"
                "  --hdc-dim N     HDC projection dimension (default: 4096)\n",
                argv[0]);
            return 0;
        }
    }

    if (!model_path || !vocab_path) {
        fprintf(stderr, "Error: --model and --vocab are required\n");
        return 1;
    }

    /* Load model + tokenizer */
    fprintf(stderr, "Loading tokenizer...\n");
    rx_tokenizer *tok = rx_tokenizer_new(vocab_path);
    if (!tok) { fprintf(stderr, "Failed to load tokenizer\n"); return 1; }

    fprintf(stderr, "Loading model...\n");
    rx_model *model = rx_model_load(model_path);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    rx_config cfg = rx_model_config(model);
    int c = cfg.n_embd;

    /* Build a diverse prompt with varied content */
    const char *prompt_text =
        "The history of mathematics spans thousands of years. Ancient civilizations like "
        "the Babylonians and Egyptians developed early number systems. The Greeks made "
        "fundamental contributions: Euclid's Elements, Archimedes' work on areas and "
        "volumes, and Pythagoras' theorem. In the 17th century, Newton and Leibniz "
        "independently invented calculus. The 19th century saw the development of "
        "abstract algebra, non-Euclidean geometry, and set theory. Modern mathematics "
        "encompasses analysis, topology, number theory, combinatorics, and more.\n\n"
        "Machine learning uses mathematical optimization to find patterns in data. "
        "Neural networks stack layers of linear transformations and nonlinearities. "
        "The attention mechanism computes similarity between query and key vectors. "
        "Transformers use self-attention to process sequences in parallel. RWKV models "
        "combine recurrent state with selective attention for efficient inference.\n\n"
        "In physics, quantum mechanics describes behavior at atomic scales. The "
        "Schrodinger equation governs wave function evolution. Heisenberg's uncertainty "
        "principle limits simultaneous knowledge of position and momentum. General "
        "relativity describes gravity as spacetime curvature. String theory attempts "
        "to unify quantum mechanics and gravity in a single framework.\n\n"
        "Programming languages have evolved from machine code to high-level abstractions. "
        "C provides direct hardware access. Rust ensures memory safety at compile time. "
        "Python offers rapid prototyping. JavaScript runs in browsers. Each language "
        "makes different tradeoffs between performance, safety, and expressiveness.";

    uint32_t *tokens = (uint32_t*)malloc(4096 * sizeof(uint32_t));
    int n_tokens = rx_tokenizer_encode(tok, prompt_text, tokens, 4096);

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "HDC Memory Viability — Test 3\n");
    fprintf(stderr, "HDC vs Cosine Correlation on Real Embeddings\n");
    fprintf(stderr, "==========================================\n");
    fprintf(stderr, "Model:          %s\n", model_path);
    fprintf(stderr, "Embedding dim:  %d\n", c);
    fprintf(stderr, "Prompt tokens:  %d\n", n_tokens);
    fprintf(stderr, "Capture layer:  %d\n", capture_layer);
    fprintf(stderr, "HDC dim:        %d\n", hdc_dim);
    fprintf(stderr, "Random pairs:   %d\n\n", max_pairs);

    /* Allocate embedding capture buffer */
    int max_capture = n_tokens + 10;
    float *embeddings = (float*)malloc((size_t)max_capture * c * sizeof(float));

    /* Create state and enable embedding capture */
    rx_state *state = rx_state_new(model);
    if (!state) { fprintf(stderr, "Failed to create state\n"); return 1; }
    rx_state_set_embed_capture(state, embeddings, max_capture, capture_layer);

    /* Prefill — each forward call captures ln_out from the target attention layer */
    fprintf(stderr, "Running prefill to capture embeddings...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    tm_tensor *logits = NULL;
    for (int i = 0; i < n_tokens; i++) {
        if (logits) tm_free(logits);
        logits = rx_forward(model, tokens[i], state);
    }
    if (logits) tm_free(logits);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    int n_captured = rx_state_embed_capture_count(state);
    fprintf(stderr, "Captured %d embeddings in %.1fms\n\n", n_captured, ms);

    if (n_captured < 20) {
        fprintf(stderr, "ERROR: Too few embeddings captured (%d). Need at least 20.\n",
                n_captured);
        fprintf(stderr, "Check that capture_layer=%d is valid (model has %d attn layers)\n",
                capture_layer, cfg.n_attn_layer);
        rx_state_free(state);
        rx_model_free(model);
        rx_tokenizer_free(tok);
        free(tokens);
        free(embeddings);
        return 1;
    }

    /* Create HDC projector */
    fv_projector *proj = fv_projector_new(c, hdc_dim, 42);

    /* Generate random pairs and compute both similarities */
    int n_pairs = max_pairs;
    if (n_pairs > n_captured * (n_captured - 1) / 2)
        n_pairs = n_captured * (n_captured - 1) / 2;

    float *cosine_sims = (float*)malloc(n_pairs * sizeof(float));
    float *hdc_sims = (float*)malloc(n_pairs * sizeof(float));

    uint32_t rng_state = 12345;
    fprintf(stderr, "Computing %d similarity pairs...\n", n_pairs);

    for (int p = 0; p < n_pairs; p++) {
        int a_idx, b_idx;
        do {
            a_idx = xorshift32(&rng_state) % n_captured;
            b_idx = xorshift32(&rng_state) % n_captured;
        } while (a_idx == b_idx);

        const float *emb_a = &embeddings[a_idx * c];
        const float *emb_b = &embeddings[b_idx * c];

        /* Cosine similarity in float space */
        cosine_sims[p] = cosine_sim(emb_a, emb_b, c);

        /* HDC: project to binary, compute Hamming similarity */
        int64_t shape_c[] = {c};
        tm_tensor *ta = tm_from_data((void*)emb_a, shape_c, 1, TM_F32);
        tm_tensor *tb = tm_from_data((void*)emb_b, shape_c, 1, TM_F32);

        fv_vector *ha = fv_project(proj, ta);
        fv_vector *hb = fv_project(proj, tb);
        hdc_sims[p] = fv_similarity(ha, hb);

        fv_free(ha);
        fv_free(hb);
        tm_free(ta);
        tm_free(tb);
    }

    /* Compute statistics */
    float cos_min = cosine_sims[0], cos_max = cosine_sims[0];
    float hdc_min = hdc_sims[0], hdc_max = hdc_sims[0];
    for (int i = 1; i < n_pairs; i++) {
        if (cosine_sims[i] < cos_min) cos_min = cosine_sims[i];
        if (cosine_sims[i] > cos_max) cos_max = cosine_sims[i];
        if (hdc_sims[i] < hdc_min) hdc_min = hdc_sims[i];
        if (hdc_sims[i] > hdc_max) hdc_max = hdc_sims[i];
    }

    float r = pearson(cosine_sims, hdc_sims, n_pairs);
    float cos_spread = cos_max - cos_min;

    fprintf(stderr, "\n==========================================\n");
    fprintf(stderr, "RESULTS\n");
    fprintf(stderr, "==========================================\n\n");
    fprintf(stderr, "Cosine similarity range:  [%.4f, %.4f] (spread: %.4f)\n",
            cos_min, cos_max, cos_spread);
    fprintf(stderr, "HDC similarity range:     [%.4f, %.4f]\n", hdc_min, hdc_max);
    fprintf(stderr, "Pearson correlation r:    %.4f\n\n", r);

    /* Distribution buckets */
    fprintf(stderr, "Cosine similarity distribution:\n");
    int buckets[10] = {0};
    for (int i = 0; i < n_pairs; i++) {
        int b = (int)(cosine_sims[i] * 10);
        if (b < 0) b = 0;
        if (b > 9) b = 9;
        buckets[b]++;
    }
    for (int b = 0; b < 10; b++) {
        fprintf(stderr, "  [%.1f, %.1f): %d", b * 0.1f, (b + 1) * 0.1f, buckets[b]);
        /* Mini bar */
        int bar = buckets[b] * 40 / (n_pairs > 0 ? n_pairs : 1);
        fprintf(stderr, " ");
        for (int x = 0; x < bar; x++) fprintf(stderr, "#");
        fprintf(stderr, "\n");
    }

    /* Print some example pairs */
    fprintf(stderr, "\nExample pairs (first 10):\n");
    fprintf(stderr, "  %-12s %-12s\n", "cosine", "hdc");
    for (int i = 0; i < 10 && i < n_pairs; i++) {
        fprintf(stderr, "  %-12.4f %-12.4f\n", cosine_sims[i], hdc_sims[i]);
    }

    /* Verdict */
    fprintf(stderr, "\nVERDICT: ");
    if (r > 0.85f && cos_spread > 0.3f) {
        fprintf(stderr, "PASS (r=%.4f > 0.85, spread=%.4f > 0.3)\n", r, cos_spread);
    } else if (r > 0.70f) {
        fprintf(stderr, "MARGINAL (r=%.4f — moderate correlation, may need higher HDC dim)\n", r);
    } else {
        fprintf(stderr, "FAIL (r=%.4f — HDC does not preserve similarity structure)\n", r);
    }
    if (cos_spread < 0.3f) {
        fprintf(stderr, "WARNING: Embeddings are tightly clustered (spread=%.4f).\n",
                cos_spread);
        fprintf(stderr, "  This may indicate the embeddings are too similar for HDC to discriminate.\n");
    }

    /* Cleanup */
    fv_projector_free(proj);
    free(cosine_sims);
    free(hdc_sims);
    free(embeddings);
    free(tokens);
    rx_state_free(state);
    rx_model_free(model);
    rx_tokenizer_free(tok);

    fprintf(stderr, "\n==========================================\n");
    return 0;
}
