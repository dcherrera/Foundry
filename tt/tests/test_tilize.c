/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tilize.c
 * @brief Phase 3 off-device test: the 32x32 face-ordered tile layout, its
 *   swizzled sibling, and the round trip through both on every shape the
 *   target model uses.
 *
 * NO DEVICE IS TOUCHED. This is address arithmetic and memcpy.
 *
 * ---------------------------------------------------------------------------
 * WHY "ROUND-TRIP IS THE IDENTITY" IS NOT ENOUGH ON ITS OWN
 * ---------------------------------------------------------------------------
 * tilize -> untilize is the identity for ANY consistent permutation, including
 * a wrong one. A backend that stored tiles row-major and read them back
 * row-major would pass a round-trip test and then feed a Tensix garbage. So
 * this file pins the layout ITSELF first — the four face base offsets, the
 * exact destination index of a handful of known (row, col) pairs, and the fact
 * that face order is NOT row-major over the whole tile — and only then checks
 * the round trip over the model's shapes.
 *
 * @status Green on macOS/arm64 under
 *   `cc -std=c99 -Wall -Wextra -Werror -pedantic -Wshadow -Wconversion`.
 *   Mutation-proven: see docs/tt_bringup_log.md §12.
 * @issues None
 * @todo None
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_tilize.h"

/* ------------------------------------------------------------------------- */
/* Tiny harness                                                              */
/* ------------------------------------------------------------------------- */

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (condition) {
        printf("  ok    %s\n", what);
    } else {
        checks_failed++;
        printf("  FAIL  %s\n", what);
    }
}

static void check_u64(uint64_t got, uint64_t want, const char *what)
{
    checks_run++;
    if (got == want) {
        printf("  ok    %s = %" PRIu64 "\n", what, got);
    } else {
        checks_failed++;
        printf("  FAIL  %s = %" PRIu64 ", expected %" PRIu64 "\n", what, got, want);
    }
}

static void section(const char *name)
{
    printf("\n== %s ==\n", name);
}

/* ------------------------------------------------------------------------- */
/* The layout itself                                                         */
/* ------------------------------------------------------------------------- */

static void test_face_index(void)
{
    section("face ordering is what tt-metal's index expression says");

    /* Face bases: face_y stride 512, face_x stride 256. */
    check_u64(ft_tt_tile_face_index(0, 0), 0, "face (0,0) starts at 0");
    check_u64(ft_tt_tile_face_index(0, 16), 256, "face (0,1) starts at 256");
    check_u64(ft_tt_tile_face_index(16, 0), 512, "face (1,0) starts at 512");
    check_u64(ft_tt_tile_face_index(16, 16), 768, "face (1,1) starts at 768");
    check_u64(ft_tt_tile_face_index(31, 31), 1023, "last element is 1023");

    /* Inside a face it is row-major with a stride of 16, NOT 32. */
    check_u64(ft_tt_tile_face_index(0, 15), 15, "row 0 col 15");
    check_u64(ft_tt_tile_face_index(1, 0), 16, "row 1 col 0 is 16, not 32");
    check_u64(ft_tt_tile_face_index(15, 15), 255, "last element of face (0,0)");
    check_u64(ft_tt_tile_face_index(1, 16), 272, "row 1 col 16 is in face (0,1)");
    /* 512 (face (1,0)) + 1 row into the face * 16 + 3. */
    check_u64(ft_tt_tile_face_index(17, 3), 531, "row 17 col 3 is in face (1,0)");

    /* The whole point: this is NOT the row-major index. */
    check(ft_tt_tile_face_index(0, 16) != 16, "face order differs from row-major at (0,16)");
    check(ft_tt_tile_face_index(1, 0) != 32, "face order differs from row-major at (1,0)");

    section("the index is a permutation of 0..1023");
    {
        static unsigned char seen[FT_TT_TILE_VALUES];
        size_t collisions = 0;

        memset(seen, 0, sizeof(seen));
        for (uint32_t row = 0; row < FT_TT_TILE_HEIGHT; row++) {
            for (uint32_t col = 0; col < FT_TT_TILE_WIDTH; col++) {
                uint32_t index = ft_tt_tile_face_index(row, col);

                if (index >= FT_TT_TILE_VALUES || seen[index])
                    collisions++;
                else
                    seen[index] = 1;
            }
        }
        check_u64(collisions, 0, "every (row,col) maps to a distinct index in range");
    }
}

static void test_tile_counts(void)
{
    section("tile counts round up");

    check_u64(ft_tt_tile_count_for(0), 0, "0 columns needs 0 tiles");
    check_u64(ft_tt_tile_count_for(1), 1, "1 column needs 1 tile");
    check_u64(ft_tt_tile_count_for(32), 1, "32 columns needs 1 tile");
    check_u64(ft_tt_tile_count_for(33), 2, "33 columns needs 2 tiles");
    check_u64(ft_tt_tile_count_for(1536), 48, "1536 columns needs 48 tiles");

    check_u64(ft_tt_tilize_bytes(4, 32, 32), 4096, "one fp32 tile is 4096 B");
    check_u64(ft_tt_tilize_bytes(2, 32, 32), 2048, "one bf16 tile is 2048 B");
    check_u64(ft_tt_tilize_bytes(4, 1, 1), 4096, "a 1x1 matrix still costs a whole tile");
    check_u64(ft_tt_tilize_bytes(3, 32, 32), 0, "an unsupported element size sizes to 0");
}

/* ------------------------------------------------------------------------- */
/* Round trip over the model's shapes                                        */
/* ------------------------------------------------------------------------- */

/*
 * Every distinct (rows, cols) in MiniCPM5-1B-Q8_0.gguf, plus the awkward
 * shapes around the tile boundary. dims are (rows = GGUF dims[1],
 * cols = GGUF dims[0]).
 */
typedef struct {
    uint32_t rows;
    uint32_t cols;
    const char *what;
} shape;

static const shape SHAPES[] = {
    {130560, 1536, "token_embd / output (vocab x n_embd)"},
    {2048, 1536, "attn_q (16 heads x 128)"},
    {1536, 2048, "attn_output"},
    {256, 1536, "attn_k, attn_v (GQA 2 kv heads x 128)"},
    {4608, 1536, "ffn_gate, ffn_up"},
    {1536, 4608, "ffn_down"},
    {1, 1536, "norms as a 1-row matrix"},
    {1, 1, "degenerate 1x1"},
    {31, 31, "one short of a tile in both axes"},
    {32, 32, "exactly one tile"},
    {33, 33, "one past a tile in both axes"},
    {32, 17, "partial face on the column edge"},
    {17, 32, "partial rows"},
    {64, 48, "two tile rows, one and a half tile columns"},
};

#define SHAPE_COUNT ((int)(sizeof(SHAPES) / sizeof(SHAPES[0])))

static uint32_t lcg_next(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

/**
 * Fill with a value that depends on BOTH indices, so a layout bug that swaps
 * or duplicates a face cannot survive by luck.
 */
static void fill_pattern(uint32_t *matrix, uint32_t rows, uint32_t cols, uint32_t seed)
{
    uint32_t state = seed;

    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t col = 0; col < cols; col++)
            matrix[(size_t)row * cols + col] = (row * 2654435761u) ^ (col * 40503u) ^ lcg_next(&state);
    }
}

static int round_trip_shape(const shape *s, size_t element_bytes, int swizzled)
{
    size_t values = (size_t)s->rows * s->cols;
    size_t tile_bytes = ft_tt_tilize_bytes(element_bytes, s->rows, s->cols);
    unsigned char *source = malloc(values * element_bytes);
    unsigned char *tiles = malloc(tile_bytes);
    unsigned char *restored = malloc(values * element_bytes);
    uint32_t tile_rows = ft_tt_tile_count_for(s->rows);
    uint32_t tile_cols = ft_tt_tile_count_for(s->cols);
    size_t strip_bytes = (size_t)tile_cols * FT_TT_TILE_VALUES * element_bytes;
    int ok = 0;

    if (!source || !tiles || !restored)
        goto done;

    {
        uint32_t *pattern = malloc(values * sizeof(uint32_t));

        if (!pattern)
            goto done;
        fill_pattern(pattern, s->rows, s->cols, (uint32_t)(s->rows * 31u + s->cols));
        for (size_t i = 0; i < values; i++)
            memcpy(source + i * element_bytes, &pattern[i], element_bytes);
        free(pattern);
    }

    memset(restored, 0xa5, values * element_bytes);

    for (uint32_t strip = 0; strip < tile_rows; strip++) {
        uint32_t first = strip * FT_TT_TILE_HEIGHT;
        uint32_t valid = s->rows - first < FT_TT_TILE_HEIGHT ? s->rows - first : FT_TT_TILE_HEIGHT;
        const unsigned char *src = source + (size_t)first * s->cols * element_bytes;
        unsigned char *dst = restored + (size_t)first * s->cols * element_bytes;
        unsigned char *strip_tiles = tiles + (size_t)strip * strip_bytes;
        size_t stride = (size_t)s->cols * element_bytes;
        int rc;

        if (swizzled) {
            rc = ft_tt_tile_swizzle_strip(src, element_bytes, valid, s->cols, stride, strip_tiles);
            if (rc != FT_TT_TILIZE_OK)
                goto done;
            rc = ft_tt_tile_unswizzle_strip(strip_tiles, element_bytes, valid, s->cols, stride, dst);
        } else {
            rc = ft_tt_tilize_strip(src, element_bytes, valid, s->cols, stride, strip_tiles);
            if (rc != FT_TT_TILIZE_OK)
                goto done;
            rc = ft_tt_untilize_strip(strip_tiles, element_bytes, valid, s->cols, stride, dst);
        }
        if (rc != FT_TT_TILIZE_OK)
            goto done;
    }

    ok = memcmp(source, restored, values * element_bytes) == 0;

done:
    free(source);
    free(tiles);
    free(restored);
    return ok;
}

static void test_round_trip(void)
{
    section("tilize -> untilize is the identity on every shape we use");

    for (int i = 0; i < SHAPE_COUNT; i++) {
        char label[160];

        snprintf(label, sizeof(label), "fp32 %ux%u  %s", SHAPES[i].rows, SHAPES[i].cols, SHAPES[i].what);
        check(round_trip_shape(&SHAPES[i], 4, 0), label);
    }

    section("the same, at 2 bytes per element (bf16)");
    for (int i = 0; i < SHAPE_COUNT; i++) {
        char label[160];

        snprintf(label, sizeof(label), "bf16 %ux%u", SHAPES[i].rows, SHAPES[i].cols);
        check(round_trip_shape(&SHAPES[i], 2, 0), label);
    }

    section("swizzled (no face reorder) round trip, the block-float input path");
    for (int i = 0; i < SHAPE_COUNT; i++) {
        char label[160];

        snprintf(label, sizeof(label), "swizzled fp32 %ux%u", SHAPES[i].rows, SHAPES[i].cols);
        check(round_trip_shape(&SHAPES[i], 4, 1), label);
    }
}

/* ------------------------------------------------------------------------- */
/* Padding and placement                                                     */
/* ------------------------------------------------------------------------- */

static void test_padding_is_zero(void)
{
    static uint32_t source[17 * 17];
    static uint32_t tiles[FT_TT_TILE_VALUES];
    size_t nonzero_pad = 0;

    section("padding is zero, and the valid region lands where it should");

    for (uint32_t i = 0; i < 17 * 17; i++)
        source[i] = 0xdeadbeefu;

    memset(tiles, 0x5a, sizeof(tiles));
    check(ft_tt_tilize_strip(source, 4, 17, 17, 17 * 4, tiles) == FT_TT_TILIZE_OK, "17x17 tilizes");

    for (uint32_t row = 0; row < FT_TT_TILE_HEIGHT; row++) {
        for (uint32_t col = 0; col < FT_TT_TILE_WIDTH; col++) {
            uint32_t value = tiles[ft_tt_tile_face_index(row, col)];

            if (row < 17 && col < 17) {
                if (value != 0xdeadbeefu)
                    nonzero_pad++;
            } else if (value != 0) {
                nonzero_pad++;
            }
        }
    }
    check_u64(nonzero_pad, 0, "misplaced or non-zero elements in the padded 17x17 tile");

    section("swizzled and face-ordered agree once the faces are applied");
    {
        static uint32_t swizzled[FT_TT_TILE_VALUES];
        static uint32_t faced[FT_TT_TILE_VALUES];
        static uint32_t dense[FT_TT_TILE_VALUES];
        size_t differing = 0;

        fill_pattern(dense, FT_TT_TILE_HEIGHT, FT_TT_TILE_WIDTH, 7u);
        check(ft_tt_tile_swizzle_strip(dense, 4, 32, 32, 32 * 4, swizzled) == FT_TT_TILIZE_OK, "swizzle a full tile");
        check(ft_tt_tilize_strip(dense, 4, 32, 32, 32 * 4, faced) == FT_TT_TILIZE_OK, "tilize the same tile");

        for (uint32_t row = 0; row < FT_TT_TILE_HEIGHT; row++) {
            for (uint32_t col = 0; col < FT_TT_TILE_WIDTH; col++) {
                if (swizzled[row * FT_TT_TILE_WIDTH + col] != faced[ft_tt_tile_face_index(row, col)])
                    differing++;
            }
        }
        check_u64(differing, 0, "face order applied to the swizzled tile reproduces the tilized tile");

        /* And they are genuinely different byte strings, so the check above is
         * not passing because both layouts happen to be the same thing. */
        check(memcmp(swizzled, faced, sizeof(faced)) != 0, "the two layouts are not the same bytes");
    }
}

static void test_argument_validation(void)
{
    static uint32_t source[FT_TT_TILE_VALUES];
    static uint32_t tiles[FT_TT_TILE_VALUES];

    section("argument validation");

    check(ft_tt_tilize_strip(NULL, 4, 32, 32, 128, tiles) == FT_TT_TILIZE_EINVAL, "NULL source rejected");
    check(ft_tt_tilize_strip(source, 4, 32, 32, 128, NULL) == FT_TT_TILIZE_EINVAL, "NULL destination rejected");
    check(ft_tt_tilize_strip(source, 3, 32, 32, 96, tiles) == FT_TT_TILIZE_EINVAL, "element size 3 rejected");
    check(ft_tt_tilize_strip(source, 4, 0, 32, 128, tiles) == FT_TT_TILIZE_EINVAL, "zero rows rejected");
    check(ft_tt_tilize_strip(source, 4, 33, 32, 128, tiles) == FT_TT_TILIZE_EINVAL, "more than 32 rows rejected");
    check(ft_tt_tilize_strip(source, 4, 32, 0, 128, tiles) == FT_TT_TILIZE_EINVAL, "zero columns rejected");
    check(ft_tt_tilize_strip(source, 4, 32, 32, 64, tiles) == FT_TT_TILIZE_EINVAL, "a stride shorter than a row rejected");
    check(ft_tt_tile_swizzle_strip(source, 4, 32, 32, 64, tiles) == FT_TT_TILIZE_EINVAL, "swizzle validates too");
}

int main(void)
{
    printf("test_tilize — Blackhole 32x32 face-ordered tile layout\n");

    test_face_index();
    test_tile_counts();
    test_round_trip();
    test_padding_is_zero();
    test_argument_validation();

    printf("\n%d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}
