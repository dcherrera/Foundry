/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bfp6_expand_tile.h
 * @brief The scalar BFP6_B -> BFP8_B tile expansion for any Tensix RISC-V (BRISC, NCRISC or a
 *   TRISC): routes 1 and 3 of qwen36 spec §4.1b. One 832 B tile in, one 1088 B tile out; the
 *   64 exponent bytes are copied and every 6-bit datum lands in the top six bits of its byte
 *   (BFP8 datum = BFP6 datum << 2, exact; host reference ft_bfp6_expand_tiles()).
 *
 * SWAR, four datums per 32-bit output word, words only (L1 is word-addressed hot):
 *   stream layout  24 bits (f0 f1 f2 f3, 6 bits each) -> 12-bit halves to bits 0 / 16 -> each
 *                  half's two fields to bytes -> << 2. Nine ALU ops a word.
 *   planes layout  16 bits of high nibbles and 8 bits of low pairs, each spread to one per
 *                  byte by two shift-or-mask steps, then (high << 4) | (low << 2).
 *   fold layout    (qwen36 plan S3, ft_bfp.h FT_BFP6_LAYOUT_FOLD) three words carry three BFP8
 *                  words in their top six bits per byte and the fourth word's bits in their low
 *                  two: o_i = w_i & 0xFCFCFCFC, o3 = ((w0&m)<<6)|((w1&m)<<4)|((w2&m)<<2).
 *                  Eleven ALU ops a 16-datum group (stream ~46). The RED build (shift 1) also
 *                  shifts o0 right by one, the host's red reference (tests/test_bfp6.c).
 *
 * @status New 2026-10-28 (qwen36 Phase 1.4 price bench). Byte-compared on the card against
 *   ft_bfp6_expand_tiles() by tests/test_q36_expander_bench.c.
 * @issues None
 * @todo None
 */
#ifndef BFP6_EXPAND_TILE_H
#define BFP6_EXPAND_TILE_H

#include <stdint.h>

#define BFP6_TILE_BYTES 832u
#define BFP8_TILE_BYTES 1088u
#define BFP6_LAYOUT_STREAM 0u
#define BFP6_LAYOUT_PLANES 1u
#define BFP6_LAYOUT_FOLD 2u

/** Four 6-bit fields (bits 0..23 of x) -> four bytes, each field in its byte's top six bits. */
static inline __attribute__((always_inline)) uint32_t bfp6_spread_four_stream(uint32_t x, uint32_t shift)
{
    x = ((x & 0x00FFF000u) << 4) | (x & 0x00000FFFu);
    x = ((x & 0x0FC00FC0u) << 2) | (x & 0x003F003Fu);
    return x << shift;
}

/** Four high nibbles (bits 0..15 of h) and four low pairs (bits 0..7 of l) -> four BFP8 bytes. */
static inline __attribute__((always_inline)) uint32_t bfp6_spread_four_planes(uint32_t h, uint32_t l, uint32_t shift)
{
    h = (h | (h << 8)) & 0x00FF00FFu;
    h = (h | (h << 4)) & 0x0F0F0F0Fu;
    l = (l | (l << 12)) & 0x000F000Fu;
    l = (l | (l << 6)) & 0x03030303u;
    return (h << (shift + 2u)) | (l << shift);
}

/** Expand one tile. `in` and `out` are word-aligned L1 addresses. `shift` is 2; the RED build
 *  (compile-time arg 0 = 1) passes 1, an expander off by one bit that the byte gate must catch. */
static inline __attribute__((always_inline)) void bfp6_expand_one_tile(const volatile uint32_t *in,
                                                                       volatile uint32_t *out, uint32_t layout,
                                                                       uint32_t shift)
{
    for (uint32_t w = 0; w < 16u; w++)
        out[w] = in[w];
    in += 16;
    out += 16;
    if (layout == BFP6_LAYOUT_FOLD) {
        const uint32_t m = 0x03030303u, keep = 0xFCFCFCFCu;

        /* Four groups a step: the twelve loads issue back to back, so L1's load latency is
         * paid once a step instead of once a group. */
        for (uint32_t group = 0; group < 64u; group += 4u, in += 12, out += 16) {
            uint32_t w[12];

#pragma GCC unroll 12
            for (uint32_t i = 0; i < 12u; i++)
                w[i] = in[i];
#pragma GCC unroll 4
            for (uint32_t g = 0; g < 4u; g++) {
                const uint32_t w0 = w[3u * g], w1 = w[3u * g + 1u], w2 = w[3u * g + 2u];

                out[4u * g] = shift == 2u ? (w0 & keep) : ((w0 & keep) >> 1);
                out[4u * g + 1u] = w1 & keep;
                out[4u * g + 2u] = w2 & keep;
                out[4u * g + 3u] =
                    ((w0 & m) << (shift + 4u)) | ((w1 & m) << (shift + 2u)) | ((w2 & m) << shift);
            }
        }
    } else if (layout == BFP6_LAYOUT_STREAM) {
        for (uint32_t group = 0; group < 64u; group++, in += 3, out += 4) {
            uint32_t w0 = in[0], w1 = in[1], w2 = in[2];

            out[0] = bfp6_spread_four_stream(w0 & 0x00FFFFFFu, shift);
            out[1] = bfp6_spread_four_stream((w0 >> 24) | ((w1 & 0x0000FFFFu) << 8), shift);
            out[2] = bfp6_spread_four_stream((w1 >> 16) | ((w2 & 0x000000FFu) << 16), shift);
            out[3] = bfp6_spread_four_stream(w2 >> 8, shift);
        }
    } else {
        const volatile uint32_t *low = in + 128;

        for (uint32_t group = 0; group < 64u; group++, out += 4) {
            uint32_t h0 = in[2u * group], h1 = in[2u * group + 1u], l = low[group];

            out[0] = bfp6_spread_four_planes(h0 & 0xFFFFu, l & 0xFFu, shift);
            out[1] = bfp6_spread_four_planes(h0 >> 16, (l >> 8) & 0xFFu, shift);
            out[2] = bfp6_spread_four_planes(h1 & 0xFFFFu, (l >> 16) & 0xFFu, shift);
            out[3] = bfp6_spread_four_planes(h1 >> 16, l >> 24, shift);
        }
    }
}

#endif
