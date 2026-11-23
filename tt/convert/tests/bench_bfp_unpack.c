/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_bfp_unpack.c
 * @brief ns a value of ft_bfp_unpack_tiles for Bfp8_b / Bfp6_b / Bfp4_b over 4,096 tiles — the
 *   cost the card-format CPU oracle pays a weight (qwen36_blackhole_plan.md Phase 4).
 *
 * @status None
 * @issues None
 * @todo None
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ft_bfp.h"

#define BENCH_TILES 4096u
#define BENCH_REPEATS 5

int main(void)
{
    static const ft_bfp_format formats[] = {FT_BFP_BFP8_B, FT_BFP_BFP6_B, FT_BFP_BFP4_B};
    static const char *names[] = {"bfp8_b", "bfp6_b", "bfp4_b"};
    float *tiles = malloc((size_t)BENCH_TILES * 1024u * sizeof(float));
    unsigned char *packed = malloc((size_t)BENCH_TILES * 1088u);

    if (!tiles || !packed)
        return 1;
    for (size_t i = 0; i < (size_t)BENCH_TILES * 1024u; i++)
        tiles[i] = (float)((i * 2654435761u) % 2001u) - 1000.0f;
    for (size_t f = 0; f < sizeof(formats) / sizeof(formats[0]); f++) {
        struct timespec a, b;
        double seconds;

        ft_bfp_pack_tiles(formats[f], tiles, BENCH_TILES, packed);
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int r = 0; r < BENCH_REPEATS; r++)
            ft_bfp_unpack_tiles(formats[f], packed, BENCH_TILES, tiles);
        clock_gettime(CLOCK_MONOTONIC, &b);
        seconds = (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) * 1e-9;
        printf("%s unpack: %.2f ns a value\n", names[f], seconds * 1e9 / (BENCH_REPEATS * BENCH_TILES * 1024.0));
    }
    free(tiles);
    free(packed);
    return 0;
}
