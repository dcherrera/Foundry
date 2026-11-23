/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_gdngate_layout.c
 * @brief Off-hardware check of the GDN GATE block layouts (kernels/resident_gdngate_protocol.h)
 *   at 1..64 heads: alpha/beta span ftgg_tiles(heads) side-by-side tiles a band.
 *
 * The reference never touches the protocol's addressing: it keeps alpha/beta as row-major
 * [32 rows][heads] matrices, tilizes them with its own face arithmetic, and computes every
 * gate straight from ft_det_math.h. Checked, failures counted, exit non-zero on any:
 *   1. ftgg_build_flat (all heads, and split in 4-head-multiple member slices) == reference;
 *   2. ftgg_build_block, both 16-row sub-chunks of a 32-row band == reference;
 *   3. at <= 32 heads ftgg_element(row, head) == ftgg_face_index(row, head) (Ornith's layout).
 *
 *   make -f Makefile.gdngate run    the check
 *   make -f Makefile.gdngate red    rebuilt with GDNGATE_LAYOUT_RED (the reference swaps the
 *                                   two tiles): MUST fail
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../kernels/resident_gdngate_protocol.h"

enum { kRows = 32, kFace = 16 };

static uint32_t next_random(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

/** Element (row, col) of one 32x32 tile in face order, written out independently. */
static uint32_t reference_face(uint32_t row, uint32_t col)
{
    const uint32_t face = (row / kFace) * 2u + col / kFace;

    return face * kFace * kFace + (row % kFace) * kFace + col % kFace;
}

/** Row-major [kRows][heads] into side-by-side tiles. */
static void reference_tilize(const uint16_t *matrix, uint32_t heads, uint16_t *tiles)
{
    memset(tiles, 0, (size_t)kGgMaxTiles * kGgTileElements * sizeof(uint16_t));
    for (uint32_t row = 0; row < kRows; row++)
        for (uint32_t head = 0; head < heads; head++) {
            uint32_t tile = head / 32u;
#ifdef GDNGATE_LAYOUT_RED
            tile = heads > 32u ? 1u - tile : tile;
#endif
            tiles[tile * 1024u + reference_face(row, head % 32u)] = matrix[row * heads + head];
        }
}

typedef struct case_data {
    uint32_t heads;
    uint16_t alpha[kRows * kGgMaxHeads], beta[kRows * kGgMaxHeads];
    uint16_t alpha_tiles[kGgMaxTiles * kGgTileElements], beta_tiles[kGgMaxTiles * kGgTileElements];
    uint32_t constants[kGgConstWords];
} case_data;

static void make_case(case_data *c, uint32_t heads, uint32_t seed)
{
    uint32_t state = seed;

    memset(c, 0, sizeof(*c));
    c->heads = heads;
    for (uint32_t i = 0; i < kRows * heads; i++) {
        /* bf16 in roughly [-8, 8]: sign, exponent 120..130, any mantissa. */
        const uint32_t r = next_random(&state);

        c->alpha[i] = (uint16_t)(((r >> 31) << 15) | ((120u + (r >> 8) % 11u) << 7) | (r & 0x7Fu));
        c->beta[i] = (uint16_t)((((r >> 30) & 1u) << 15) | ((118u + (r >> 12) % 14u) << 7) |
                                ((r >> 16) & 0x7Fu));
    }
    for (uint32_t head = 0; head < heads; head++) {
        float ssm_a = -0.01f * (float)(1u + next_random(&state) % 4000u);
        float ssm_dt = 0.001f * (float)((int32_t)(next_random(&state) % 8000u) - 4000);

        memcpy(&c->constants[head], &ssm_a, 4);
        memcpy(&c->constants[kGgMaxHeads + head], &ssm_dt, 4);
    }
    reference_tilize(c->alpha, heads, c->alpha_tiles);
    reference_tilize(c->beta, heads, c->beta_tiles);
}

static uint32_t reference_decay(const case_data *c, uint32_t row, uint32_t head)
{
    return ft_det_gdn_gate((uint32_t)c->alpha[row * c->heads + head] << 16,
                           c->constants[kGgMaxHeads + head], c->constants[head]);
}

static uint32_t reference_beta(const case_data *c, uint32_t row, uint32_t head)
{
    return ft_det_gdn_beta((uint32_t)c->beta[row * c->heads + head] << 16);
}

static long check_flat(const case_data *c, uint32_t members)
{
    const uint32_t stride = 2u * c->heads * 4u, words = kRows * stride / 4u;
    const uint32_t per = c->heads / members;
    uint32_t out[kRows * 2u * kGgMaxHeads];
    long bad = 0;

    memset(out, 0xA5, sizeof(out));
    for (uint32_t m = 0; m < members; m++)
        ftgg_build_flat(out, m * per, per, c->heads, c->alpha_tiles, c->beta_tiles, c->constants,
                        kRows, stride, 0);
    for (uint32_t row = 0; row < kRows; row++)
        for (uint32_t head = 0; head < c->heads; head++) {
            bad += out[row * stride / 4u + head] != reference_decay(c, row, head);
            bad += out[row * stride / 4u + c->heads + head] != reference_beta(c, row, head);
        }
    (void)words;
    return bad;
}

static long check_block(const case_data *c)
{
    uint32_t block[kGgBlockBytes / 4u];
    long bad = 0;

    for (uint32_t sub = 0; sub < 2u; sub++) {
        ftgg_build_block(block, c->heads, 0x3E000000u, c->alpha_tiles, c->beta_tiles,
                         c->constants, sub * 16u, 16u, 0);
        bad += block[0] != 0x3E000000u;
        for (uint32_t head = 0; head < c->heads; head++) {
            const uint32_t *run = block + kGgHeaderWords + head * kGgHeadWords;

            for (uint32_t row = 0; row < 16u; row++) {
                bad += run[row] != reference_decay(c, sub * 16u + row, head);
                bad += run[kGgTileRows + row] != reference_beta(c, sub * 16u + row, head);
            }
            for (uint32_t row = 16u; row < kGgTileRows; row++)
                bad += run[row] != 0u || run[kGgTileRows + row] != 0u;
        }
    }
    return bad;
}

int main(void)
{
    static case_data c;
    static const uint32_t heads_list[] = { 1u, 16u, 17u, 32u, 33u, 48u, 64u };
    long total = 0;

    for (uint32_t i = 0; i < sizeof(heads_list) / sizeof(heads_list[0]); i++) {
        const uint32_t heads = heads_list[i];
        long flat = 0, split = 0, block;

        make_case(&c, heads, 0x9E3779B9u * (i + 1u));
        flat = check_flat(&c, 1u);
        if (heads % 16u == 0u)
            split = check_flat(&c, 4u);
        block = check_block(&c);
        printf("heads %2u (%u tile%s): flat %ld, 4-member flat %ld, block %ld word(s) differ\n",
               heads, ftgg_tiles(heads), ftgg_tiles(heads) > 1u ? "s" : "", flat, split, block);
        total += flat + split + block;
    }
    {
        long moved = 0;

        for (uint32_t row = 0; row < kRows; row++)
            for (uint32_t head = 0; head < 32u; head++)
                moved += ftgg_element(row, head) != ftgg_face_index(row, head);
        printf("<= 32 heads: %ld element(s) moved from the one-tile layout\n", moved);
        total += moved;
    }
    printf("%s: %ld failure(s)\n", total ? "FAIL" : "PASS", total);
    return total ? 1 : 0;
}
