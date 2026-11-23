/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_metal_rope.c — partial NEOX RoPE kernels (fm_rope_neox_partial,
 * fm_rope_neox_partial_pos)
 *
 * The oracle is the CPU-reference golden set in metal/tests/golden (see
 * fm_golden_vectors.h for provenance). Every assertion here is against
 * those absolute values or against the kernel's own input — never against
 * another GPU result, so a kernel cannot pass by being consistently wrong.
 *
 * Two properties get their own tests because a golden diff of the rotated
 * span alone would not notice them:
 *   - dims [rope_dim, head_dim) must pass through BIT-identical,
 *   - the NEOX pair is j <-> j + rope_dim/2, not j <-> j + 1.
 *
 * Helper functions are named case_* so lib/test.h's failure line, which
 * prints __func__ + 5, still reads as the case that failed.
 *
 * Exits 77 when no Metal device is present so GPU-less CI skips cleanly.
 */

#include "lib/test.h"
#include "metal/fm_metal.h"
#include "metal/tests/fm_golden_vectors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* qwen35moe attention shape the golden set was generated for. */
#define ROPE_HEAD_DIM   256
#define ROPE_DIM         64
#define ROPE_Q_HEADS     16
#define ROPE_K_HEADS      2
#define ROPE_THETA      1e7f

/* Deliberately far tighter than the 1e-4..1e-3 relative band the golden set
 * documents for Kepler: measured on M4 / macOS 26, precise:: trig against
 * the host libm leaves sub-ULP differences, so anything the tolerance would
 * have to be loosened for is a real divergence, not GPU trig noise. */
#define ROPE_EPS_ABS    1e-5f
#define ROPE_EPS_REL    1e-5f

static const int rope_positions[] = { 0, 1, 7, 100 };
#define ROPE_N_POSITIONS ((int)(sizeof(rope_positions) / sizeof(rope_positions[0])))

/* Uploads `n` floats into a fresh device buffer. */
static fm_buffer *upload_floats(const float *src, size_t n) {
    fm_buffer *buf = fm_buffer_alloc(n * sizeof(float));
    if (buf) memcpy(fm_buffer_data(buf), src, n * sizeof(float));
    return buf;
}

/* Diffs against a golden vector. The golden file name is printed just
 * before the assertion fires, because lib/test.h names the failing function
 * and several cases share this one — without it a failure would not say
 * which position diverged. */
static void case_compare(const char *golden_file, const float *golden,
                         const float *got, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float tol = ROPE_EPS_ABS + ROPE_EPS_REL * fabsf(golden[i]);
        if (fabsf(golden[i] - got[i]) > tol)
            printf("    %s: element %zu of %zu (head %zu, dim %zu)\n",
                   golden_file, i, n, i / ROPE_HEAD_DIM, i % ROPE_HEAD_DIM);
        ASSERT_FLT_NEAR(golden[i], got[i], tol);
    }
}

/* Rotates one head layout at `pos` and diffs every element against the
 * matching golden file. */
static void case_rope_position(const char *input_file, const char *golden_file,
                               int n_heads, int pos) {
    size_t count = (size_t)n_heads * ROPE_HEAD_DIM;
    float *input = fm_golden_load_f32(input_file, count);
    float *golden = fm_golden_load_f32(golden_file, count);
    ASSERT_NOT_NULL(input);
    ASSERT_NOT_NULL(golden);

    fm_buffer *data = upload_floats(input, count);
    ASSERT_NOT_NULL(data);

    fm_rope_neox_partial(data, n_heads, ROPE_HEAD_DIM, ROPE_DIM, pos, ROPE_THETA);
    fm_sync();

    case_compare(golden_file, golden, (const float *)fm_buffer_data(data),
                 count);

    fm_buffer_free(data);
    free(input);
    free(golden);
}

static void test_rope_q_matches_golden(void) {
    char golden_file[64];
    for (int p = 0; p < ROPE_N_POSITIONS; p++) {
        snprintf(golden_file, sizeof(golden_file), "rope_q_pos%d.bin",
                 rope_positions[p]);
        case_rope_position("rope_q_input.bin", golden_file, ROPE_Q_HEADS,
                           rope_positions[p]);
        if (__ft_current_failed) return;
    }
}

static void test_rope_k_matches_golden(void) {
    char golden_file[64];
    for (int p = 0; p < ROPE_N_POSITIONS; p++) {
        snprintf(golden_file, sizeof(golden_file), "rope_k_pos%d.bin",
                 rope_positions[p]);
        case_rope_position("rope_k_input.bin", golden_file, ROPE_K_HEADS,
                           rope_positions[p]);
        if (__ft_current_failed) return;
    }
}

/* THE PARTIAL-ROTATION TRAP. A kernel that rotates all 256 dims instead of
 * the first 64 still matches the golden on [0,64) — the rotation there is
 * identical — and only differs in the tail. Compare the tail bit-for-bit
 * against the input, not approximately: the kernel is supposed to leave
 * those elements completely untouched. */
static void test_rope_pass_through_region_is_bit_identical(void) {
    size_t count = (size_t)ROPE_Q_HEADS * ROPE_HEAD_DIM;
    float *input = fm_golden_load_f32("rope_q_input.bin", count);
    ASSERT_NOT_NULL(input);

    fm_buffer *data = upload_floats(input, count);
    ASSERT_NOT_NULL(data);

    fm_rope_neox_partial(data, ROPE_Q_HEADS, ROPE_HEAD_DIM, ROPE_DIM, 7,
                         ROPE_THETA);
    fm_sync();

    const float *got = (const float *)fm_buffer_data(data);
    for (int h = 0; h < ROPE_Q_HEADS; h++) {
        size_t base = (size_t)h * ROPE_HEAD_DIM;
        int differing = memcmp(got + base + ROPE_DIM, input + base + ROPE_DIM,
                               (ROPE_HEAD_DIM - ROPE_DIM) * sizeof(float));
        ASSERT_INT_EQ(0, differing);
    }
    /* And the rotated span really was rotated, so the check above is not
     * passing simply because the kernel did nothing. */
    ASSERT_TRUE(memcmp(got, input, ROPE_DIM * sizeof(float)) != 0);

    fm_buffer_free(data);
    free(input);
}

/* THE NEOX PAIRING TRAP. One head, a single 1.0 at dim 0, everything else
 * zero. Correct NEOX pairing sends cos(angle) to dim 0 and sin(angle) to
 * dim 0 + 32; the adjacent-pair convention would send it to dim 1. */
static void test_rope_neox_pairs_j_with_j_plus_half(void) {
    const int pos = 3;
    float head[ROPE_HEAD_DIM];
    memset(head, 0, sizeof(head));
    head[0] = 1.0f;

    fm_buffer *data = upload_floats(head, ROPE_HEAD_DIM);
    ASSERT_NOT_NULL(data);

    fm_rope_neox_partial(data, 1, ROPE_HEAD_DIM, ROPE_DIM, pos, ROPE_THETA);
    fm_sync();

    /* j = 0 => freq = theta^0 = 1, so the angle is just the position. */
    const float *got = (const float *)fm_buffer_data(data);
    ASSERT_FLT_NEAR(cosf((float)pos), got[0], 1e-5f);
    ASSERT_FLT_NEAR(sinf((float)pos), got[ROPE_DIM / 2], 1e-5f);
    ASSERT_FLT_NEAR(0.0f, got[1], 1e-6f);

    fm_buffer_free(data);
}

/* The batched form against the same absolute goldens: four rows of the
 * golden Q input, one per position, checked row by row. */
static void test_rope_pos_variant_matches_golden(void) {
    size_t row = (size_t)ROPE_Q_HEADS * ROPE_HEAD_DIM;
    float *input = fm_golden_load_f32("rope_q_input.bin", row);
    ASSERT_NOT_NULL(input);

    fm_buffer *data = fm_buffer_alloc(ROPE_N_POSITIONS * row * sizeof(float));
    fm_buffer *pos_buf = fm_buffer_alloc(ROPE_N_POSITIONS * sizeof(int32_t));
    ASSERT_NOT_NULL(data);
    ASSERT_NOT_NULL(pos_buf);

    float *rows = (float *)fm_buffer_data(data);
    int32_t *positions = (int32_t *)fm_buffer_data(pos_buf);
    for (int p = 0; p < ROPE_N_POSITIONS; p++) {
        memcpy(rows + (size_t)p * row, input, row * sizeof(float));
        positions[p] = rope_positions[p];
    }

    fm_rope_neox_partial_pos(data, pos_buf, ROPE_N_POSITIONS, ROPE_Q_HEADS,
                             ROPE_HEAD_DIM, ROPE_DIM, ROPE_THETA);
    fm_sync();

    char golden_file[64];
    for (int p = 0; p < ROPE_N_POSITIONS; p++) {
        snprintf(golden_file, sizeof(golden_file), "rope_q_pos%d.bin",
                 rope_positions[p]);
        float *golden = fm_golden_load_f32(golden_file, row);
        ASSERT_NOT_NULL(golden);
        case_compare(golden_file, golden, rows + (size_t)p * row, row);
        free(golden);
        if (__ft_current_failed) return;
    }

    fm_buffer_free(data);
    fm_buffer_free(pos_buf);
    free(input);
}

int main(int argc, char **argv) {
    (void)argc;
    fm_golden_locate(argv[0]);

    printf("\n  Metal RoPE Kernel Tests\n");
    printf("  =======================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "  Metal unavailable — skipping.\n\n");
        return 77;
    }

    RUN_TEST(rope_q_matches_golden);
    RUN_TEST(rope_k_matches_golden);
    RUN_TEST(rope_pass_through_region_is_bit_identical);
    RUN_TEST(rope_neox_pairs_j_with_j_plus_half);
    RUN_TEST(rope_pos_variant_matches_golden);

    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
