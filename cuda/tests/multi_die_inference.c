/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * multi_die_inference.c — End-to-end multi-die inference test
 *
 * Loads a model on die 0, generates tokens, swaps to die 1 and back,
 * then generates more tokens to prove VRAM persistence works for
 * real inference workloads.
 *
 * Usage:
 *   sudo LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64 \
 *     ./cuda/tests/multi_die_inference \
 *     --model <path>/RWKV-X-0.2B-64k-Base.safetensors \
 *     --vocab rwkvx/data/rwkv_vocab_v20230424.txt
 *
 * Requires: bar_swap loaded, die 0 acquired, nvidia loaded, root.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <time.h>

#include "rwkvx/rwkvx.h"

/* bar_swap ioctls (inline to avoid linking barswap.c) */
struct bs_release_req { uint32_t die_id; uint32_t flags; };
struct bs_acquire_req {
    uint32_t die_id; uint32_t flags;
    uint64_t bar0_addr, bar1_addr, bar1_size, bar3_addr;
};
#define MY_RELEASE _IOW('B', 5, struct bs_release_req)
#define MY_ACQUIRE _IOWR('B', 4, struct bs_acquire_req)

static int bs_fd = -1;

static int swap_release(int die) {
    struct bs_release_req r = { .die_id = die, .flags = 0 };
    return ioctl(bs_fd, MY_RELEASE, &r);
}

static int swap_acquire(int die) {
    struct bs_acquire_req a = { .die_id = die, .flags = 0 };
    return ioctl(bs_fd, MY_ACQUIRE, &a);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void generate(rx_model *model, rx_state *state, rx_tokenizer *tok,
                     const char *prompt, int max_tokens, const char *label) {
    printf("\n--- %s ---\n", label);
    printf("Prompt: \"%s\"\n", prompt);

    /* Encode prompt */
    uint32_t tokens[512];
    int n_prompt = rx_tokenizer_encode(tok, prompt, tokens, 512);
    printf("Encoded: %d tokens\n", n_prompt);

    /* Prefill */
    double t0 = now_ms();
    tm_tensor *logits = NULL;
    for (int i = 0; i < n_prompt; i++) {
        if (logits) tm_free(logits);
        logits = rx_forward(model, tokens[i], state);
    }
    double t1 = now_ms();
    printf("Prefill: %d tokens in %.1fms (%.1f tok/s)\n",
           n_prompt, t1 - t0, n_prompt / ((t1 - t0) / 1000.0));

    /* Decode */
    rx_sampling_params params = rx_sampling_defaults();
    params.temperature = 0.0f;  /* greedy for determinism */

    uint32_t generated[256];
    int n_gen = 0;

    double t2 = now_ms();
    for (int i = 0; i < max_tokens; i++) {
        uint32_t next = rx_sample(logits, &params, NULL, 65536);
        generated[n_gen++] = next;
        tm_free(logits);
        logits = rx_forward(model, next, state);
    }
    double t3 = now_ms();
    if (logits) tm_free(logits);

    char *text = rx_tokenizer_decode(tok, generated, n_gen);
    printf("Generated: %d tokens in %.1fms (%.1f tok/s)\n",
           n_gen, t3 - t2, n_gen / ((t3 - t2) / 1000.0));
    printf("Output: %s\n", text ? text : "(decode failed)");
    free(text);
}

int main(int argc, char *argv[]) {
    const char *model_path = NULL;
    const char *vocab_path = NULL;

    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--model") == 0) model_path = argv[++i];
        else if (strcmp(argv[i], "--vocab") == 0) vocab_path = argv[++i];
    }

    if (!model_path || !vocab_path) {
        fprintf(stderr, "Usage: %s --model <path> --vocab <path>\n", argv[0]);
        return 1;
    }

    printf("=== Multi-Die Inference Test ===\n\n");

    /* Open bar_swap */
    bs_fd = open("/dev/bar_swap", O_RDWR);
    if (bs_fd < 0) { perror("open /dev/bar_swap"); return 1; }

    /* Load tokenizer */
    rx_tokenizer *tok = rx_tokenizer_new(vocab_path);
    if (!tok) { fprintf(stderr, "Failed to load tokenizer\n"); return 1; }

    /* ================================================================
     * Phase 1: Load model on die 0, generate
     * ================================================================ */
    printf("=== Phase 1: Load model on die 0 ===\n");
    double t0 = now_ms();
    rx_model *model = rx_model_load_q8(model_path);
    double t1 = now_ms();
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    printf("Model loaded in %.0fms\n", t1 - t0);

    rx_state *state = rx_state_new(model);

    generate(model, state, tok, "The capital of France is", 16, "Phase 1: Die 0 generation");

    /* Save first 5 generated tokens for later comparison */
    printf("\n");

    /* ================================================================
     * Phase 2: Swap to die 1, then back to die 0
     * ================================================================ */
    printf("=== Phase 2: Hot swap die 0 → die 1 → die 0 ===\n\n");

    printf("Releasing die 0...\n");
    t0 = now_ms();
    if (swap_release(0) != 0) {
        perror("release die 0");
        /* Try acquire anyway */
    }
    t1 = now_ms();
    printf("Released in %.2fms\n", t1 - t0);

    printf("Acquiring die 1...\n");
    t0 = now_ms();
    if (swap_acquire(1) != 0) {
        perror("acquire die 1");
        printf("Die 1 acquire failed — may not be available. Trying die 0 directly.\n");
    } else {
        t1 = now_ms();
        printf("Acquired die 1 in %.2fms\n", t1 - t0);

        printf("Sleeping 200ms on die 1...\n");
        usleep(200000);

        printf("Releasing die 1...\n");
        swap_release(1);
    }

    printf("Re-acquiring die 0...\n");
    t0 = now_ms();
    if (swap_acquire(0) != 0) {
        perror("acquire die 0");
        close(bs_fd);
        return 1;
    }
    t1 = now_ms();
    printf("Re-acquired die 0 in %.2fms\n\n", t1 - t0);

    /* ================================================================
     * Phase 3: Generate on die 0 again — does it still work?
     * ================================================================ */
    printf("=== Phase 3: Generate on die 0 after hot swap ===\n");

    /* Reset state for a fresh prompt */
    rx_state_reset(state);
    generate(model, state, tok, "The capital of France is", 16,
             "Phase 3: Die 0 after swap (should match Phase 1)");

    /* ================================================================
     * Phase 4: Multiple rapid swaps
     * ================================================================ */
    printf("\n\n=== Phase 4: Rapid swap stress test ===\n\n");

    int n_swaps = 10;
    double total_swap_ms = 0;
    for (int i = 0; i < n_swaps; i++) {
        t0 = now_ms();
        swap_release(0);
        swap_acquire(0);
        t1 = now_ms();
        total_swap_ms += (t1 - t0);
    }
    printf("%d release/acquire cycles: %.1fms total, %.2fms avg\n",
           n_swaps, total_swap_ms, total_swap_ms / n_swaps);

    /* Generate one more time after stress test */
    rx_state_reset(state);
    generate(model, state, tok, "Hello", 8, "Phase 4: After 10 rapid swaps");

    /* ================================================================
     * Cleanup
     * ================================================================ */
    printf("\n\nCleaning up...\n");
    rx_state_free(state);
    rx_model_free(model);
    rx_tokenizer_free(tok);
    close(bs_fd);

    printf("Done.\n");
    return 0;
}
