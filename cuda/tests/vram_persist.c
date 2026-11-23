/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * vram_persist.c — Test whether K80 VRAM survives nvidia unload/reload
 *
 * This is the critical experiment for multi-die model persistence.
 * If VRAM persists across nvidia unload/reload cycles, we can pre-load
 * 6 models on 6 dies and switch between them in ~2.5s without re-uploading.
 *
 * Test procedure:
 *   Phase 1 (Write):
 *     1. fc_init() — loads nvidia, creates CUDA context
 *     2. Allocate 1 MB GPU memory
 *     3. Fill with known pattern (0xDEADBEEF repeated)
 *     4. Verify pattern reads back correctly
 *     5. fc_free() — destroys context (does NOT unload nvidia)
 *
 *   Phase 2 (Read back after nvidia reload):
 *     1. fc_init() again — creates new CUDA context
 *     2. Allocate 1 MB GPU memory (hopefully same physical VRAM)
 *     3. Read it back
 *     4. Check if 0xDEADBEEF pattern survived
 *
 * Usage:
 *   # Die must be acquired and nvidia loaded before running
 *   sudo ./cuda/tests/vram_persist                  # basic test
 *   sudo ./cuda/tests/vram_persist --with-rmmod     # unload/reload nvidia between phases
 *
 * The --with-rmmod variant is the real test: it unloads nvidia entirely
 * between the write and read phases, simulating a die swap cycle.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "cuda/fc_cuda.h"

#define TEST_SIZE (1024 * 1024)  /* 1 MB */
#define PATTERN   0xDEADBEEF

static int run_system(const char *cmd) {
    printf("  > %s\n", cmd);
    return system(cmd);
}

int main(int argc, char *argv[]) {
    int with_rmmod = (argc > 1 && strcmp(argv[1], "--with-rmmod") == 0);

    printf("=== VRAM Persistence Test ===\n");
    printf("Mode: %s\n\n", with_rmmod ? "with nvidia unload/reload" : "context destroy only");

    int n_words = TEST_SIZE / sizeof(uint32_t);
    uint32_t *write_buf = (uint32_t *)malloc(TEST_SIZE);
    uint32_t *read_buf  = (uint32_t *)malloc(TEST_SIZE);

    /* Fill write buffer with pattern */
    for (int i = 0; i < n_words; i++) {
        write_buf[i] = PATTERN;
    }

    /* ============================================================
     * Phase 1: Write pattern to VRAM
     * ============================================================ */
    printf("--- Phase 1: Write pattern to VRAM ---\n");

    if (fc_init() != 0) {
        fprintf(stderr, "ERROR: fc_init failed (is nvidia loaded?)\n");
        free(write_buf); free(read_buf);
        return 1;
    }

    fc_buffer *buf = fc_buffer_from_ptr(write_buf, TEST_SIZE);
    if (!buf) {
        fprintf(stderr, "ERROR: fc_buffer_from_ptr failed\n");
        fc_free();
        free(write_buf); free(read_buf);
        return 1;
    }

    /* Verify write was correct */
    fc_sync();
    fc_buffer_to_cpu(buf, read_buf);
    int write_ok = 1;
    for (int i = 0; i < n_words; i++) {
        if (read_buf[i] != PATTERN) {
            printf("  Write verify FAILED at word %d: got 0x%08X expected 0x%08X\n",
                   i, read_buf[i], PATTERN);
            write_ok = 0;
            break;
        }
    }
    if (write_ok) {
        printf("  Write verify OK: %d words of 0x%08X written to VRAM\n", n_words, PATTERN);
    }

    /* Intentionally do NOT free the buffer — we want to see if the
     * VRAM contents survive context destruction */
    printf("  Destroying CUDA context (keeping VRAM allocated)...\n");
    fc_free();
    printf("  Phase 1 complete.\n\n");

    /* ============================================================
     * Optional: Unload and reload nvidia
     * ============================================================ */
    if (with_rmmod) {
        printf("--- Unloading nvidia driver ---\n");
        run_system("rmmod nvidia_uvm 2>/dev/null");
        run_system("rmmod nvidia_drm 2>/dev/null");
        run_system("rmmod nvidia_modeset 2>/dev/null");
        run_system("rmmod nvidia 2>/dev/null");
        printf("  nvidia unloaded.\n\n");

        printf("--- Reloading nvidia driver ---\n");
        run_system("modprobe --ignore-install nvidia");
        run_system("modprobe --ignore-install nvidia_modeset");
        run_system("modprobe --ignore-install nvidia_drm");
        run_system("modprobe --ignore-install nvidia_uvm");
        /* Give driver time to initialize */
        run_system("sleep 1");
        printf("  nvidia reloaded.\n\n");
    }

    /* ============================================================
     * Phase 2: Read back VRAM
     * ============================================================ */
    printf("--- Phase 2: Read back VRAM ---\n");

    if (fc_init() != 0) {
        fprintf(stderr, "ERROR: fc_init failed on second init\n");
        free(write_buf); free(read_buf);
        return 1;
    }

    /* Allocate same size — CUDA may or may not give us the same physical VRAM.
     * We allocate WITHOUT initializing to see what's already there. */
    fc_buffer *buf2 = fc_buffer_alloc(TEST_SIZE);
    if (!buf2) {
        fprintf(stderr, "ERROR: fc_buffer_alloc failed\n");
        fc_free();
        free(write_buf); free(read_buf);
        return 1;
    }

    memset(read_buf, 0, TEST_SIZE);
    fc_buffer_to_cpu(buf2, read_buf);

    /* Count how many words match the pattern */
    int match_count = 0;
    int first_mismatch = -1;
    for (int i = 0; i < n_words; i++) {
        if (read_buf[i] == PATTERN) {
            match_count++;
        } else if (first_mismatch < 0) {
            first_mismatch = i;
        }
    }

    float match_pct = 100.0f * match_count / n_words;

    printf("\n");
    printf("=== RESULTS ===\n");
    printf("  Words checked:  %d\n", n_words);
    printf("  Pattern matches: %d (%.1f%%)\n", match_count, match_pct);

    if (match_count == n_words) {
        printf("\n  >>> VRAM PERSISTS! <<<\n");
        printf("  All %d words survived %s.\n", n_words,
               with_rmmod ? "nvidia unload/reload" : "context destroy/recreate");
        printf("  Architecture A is viable: pre-load models, switch dies freely.\n");
    } else if (match_count > n_words / 2) {
        printf("\n  >>> PARTIAL PERSISTENCE <<<\n");
        printf("  %d/%d words survived (%.1f%%). First mismatch at word %d.\n",
               match_count, n_words, match_pct, first_mismatch);
        printf("  Some VRAM regions may be reused by the driver.\n");
    } else {
        printf("\n  >>> VRAM WIPED <<<\n");
        printf("  Only %d/%d words survived (%.1f%%).\n",
               match_count, n_words, match_pct);
        if (first_mismatch >= 0) {
            printf("  First mismatch at word %d: got 0x%08X\n",
                   first_mismatch, read_buf[first_mismatch]);
        }
        printf("  Architecture B required: must re-upload weights each switch.\n");
        printf("  Switch cost: ~0.5s (0.2B) or ~9s (3.6B) per die swap.\n");
    }

    fc_buffer_free(buf2);
    fc_free();
    free(write_buf);
    free(read_buf);

    printf("\nDone.\n");
    return (match_count == n_words) ? 0 : 1;
}
