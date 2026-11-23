/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_kv_block_float.c
 * @brief Off-hardware gate for kernels/ft_kv_block_float.h, the block-float KV row packer the
 *   KV append writer runs: its bytes must equal convert/ft_bfp.c's (the weight packer, the
 *   format the unpacker already reads) on bf16 inputs, a partial pack must keep every other
 *   byte, and the unpack must equal ft_bfp_unpack_tiles. Plus the KV append encoding: the
 *   cache format flag and the bf16 source strides on a block-float cache.
 *
 * Build: make -f Makefile.kvbf && ./build-kvbf/test_kv_block_float
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "convert/ft_bfp.h"
#include "ft_tt_device.h"
#include "ft_tt_op_memory.h"
#include "kernels/ft_kv_block_float.h"

static int g_checks;
static int g_failures;

#define CHECK(condition, ...)                                                                   \
    do {                                                                                        \
        g_checks++;                                                                             \
        if (!(condition)) {                                                                     \
            g_failures++;                                                                       \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                \
            fprintf(stderr, __VA_ARGS__);                                                       \
            fputc('\n', stderr);                                                                \
        }                                                                                       \
    } while (0)

static uint32_t g_rng = 0x2545F491u;

static uint32_t next_random(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

/** A bf16 bit pattern of the kind a K/V row carries, plus the edges every 8th value. */
static uint16_t random_bf16(uint32_t index)
{
    static const uint16_t edges[] = { 0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x7F7F,
                                      0x0080, 0x3FC0, 0x4040, 0x3E00, 0xC2F7 };
    const uint32_t r = next_random();

    if ((index & 7u) == 7u)
        return edges[(r >> 8) % (sizeof(edges) / sizeof(edges[0]))];
    /* Exponents spread over ~2^-20..2^12, the range a projection output lives in. */
    return (uint16_t)(((r & 1u) << 15) | (((107u + (r >> 1) % 33u) & 0xFFu) << 7) |
                      ((r >> 9) & 0x7Fu));
}

static float bf16_to_float(uint16_t bits)
{
    uint32_t wide = (uint32_t)bits << 16;
    float value;

    memcpy(&value, &wide, sizeof(value));
    return value;
}

/** One random tile: face-ordered bf16 (what the writer's CB page holds) and row-major fp32
 *  (what ft_bfp_pack_tiles takes). Rows of one tile get different scales on purpose. */
static void make_tile(uint16_t face_ordered[1024], float row_major[1024])
{
    for (uint32_t row = 0; row < 32u; row++) {
        for (uint32_t col = 0; col < 32u; col++) {
            const uint16_t bits = random_bf16(row * 32u + col);
            const uint32_t face = (row / 16u) * 2u + col / 16u;

            face_ordered[face * 256u + (row % 16u) * 16u + col % 16u] = bits;
            row_major[row * 32u + col] = bf16_to_float(bits);
        }
    }
}

static void check_format(uint32_t kv_format, ft_bfp_format reference_format, const char *name)
{
    const uint32_t tile_bytes = ft_kv_bf_tile_bytes(kv_format);
    uint16_t face_ordered[1024];
    float row_major[1024], reference_out[1024];
    uint8_t reference[1088], packed[1088], before[1088];
    uint32_t mismatched_tiles = 0u, mismatched_rows = 0u, unpack_mismatches = 0u;

    CHECK(tile_bytes == ft_bfp_tile_bytes(reference_format), "%s tile bytes %u vs %zu", name,
          tile_bytes, ft_bfp_tile_bytes(reference_format));
    for (uint32_t trial = 0; trial < 2000u; trial++) {
        uint32_t row0, rows;

        make_tile(face_ordered, row_major);
        ft_bfp_pack_tiles(reference_format, row_major, 1, reference);
        memset(packed, 0xA5, sizeof(packed));
        ft_kv_bf_pack_tile_rows(kv_format, face_ordered, packed, 0u, 32u);
        if (memcmp(packed, reference, tile_bytes) != 0)
            mismatched_tiles++;

        /* A partial pack (an append) lands its rows and keeps every other byte. */
        row0 = next_random() % 32u;
        rows = 1u + next_random() % (32u - row0);
        memset(packed, 0x5A, sizeof(packed));
        memcpy(before, packed, sizeof(before));
        ft_kv_bf_pack_tile_rows(kv_format, face_ordered, packed, row0, rows);
        for (uint32_t row = 0; row < 32u; row++) {
            const bool written = row >= row0 && row < row0 + rows;
            const uint8_t *expected = written ? reference : before;

            for (uint32_t half = 0; half < 2u; half++) {
                const uint32_t face = (row / 16u) * 2u + half, face_row = row % 16u;
                const uint32_t e = ft_kv_bf_exponent_offset(face, face_row);
                const uint32_t d = ft_kv_bf_datum_offset(kv_format, face, face_row);

                if (packed[e] != expected[e] ||
                    memcmp(packed + d, expected + d, ft_kv_bf_row_datum_bytes(kv_format)) != 0)
                    mismatched_rows++;
            }
        }

        /* Unpack equals the reference decode. */
        ft_bfp_unpack_tiles(reference_format, reference, 1, reference_out);
        for (uint32_t row = 0; row < 32u; row++) {
            for (uint32_t half = 0; half < 2u; half++) {
                uint32_t bits[16];

                ft_kv_bf_unpack_row(kv_format, reference, (row / 16u) * 2u + half, row % 16u,
                                    bits);
                for (uint32_t i = 0; i < 16u; i++) {
                    uint32_t want;

                    memcpy(&want, &reference_out[row * 32u + half * 16u + i], sizeof(want));
                    if (want != bits[i])
                        unpack_mismatches++;
                }
            }
        }
    }
    CHECK(mismatched_tiles == 0u, "%s: %u of 2000 whole tiles differ from ft_bfp", name,
          mismatched_tiles);
    CHECK(mismatched_rows == 0u, "%s: %u face rows wrong after a partial pack", name,
          mismatched_rows);
    CHECK(unpack_mismatches == 0u, "%s: %u values decode differently", name, unpack_mismatches);
    printf("  %s: 2000 tiles bit-exact vs ft_bfp, partial packs keep their neighbours\n", name);
}

/** Error of the format on a K-like row, for the record (no bar: quality is the card gate). */
static void report_error(uint32_t kv_format, const char *name)
{
    double squared_error = 0.0, squared_signal = 0.0;
    uint16_t values[16];
    uint8_t datums[16];

    for (uint32_t trial = 0; trial < 20000u; trial++) {
        uint32_t shared, bits[16];
        uint8_t tile[1088];

        for (uint32_t i = 0; i < 16u; i++) {
            const float v = (float)((int32_t)(next_random() % 20001u) - 10000) / 3000.0f;
            uint32_t wide;

            memcpy(&wide, &v, sizeof(wide));
            values[i] = (uint16_t)(wide >> 16);
        }
        shared = ft_kv_bf_pack_row(kv_format, values, datums);
        memset(tile, 0, sizeof(tile));
        tile[0] = (uint8_t)shared;
        memcpy(tile + ft_kv_bf_datum_offset(kv_format, 0, 0), datums,
               ft_kv_bf_row_datum_bytes(kv_format));
        ft_kv_bf_unpack_row(kv_format, tile, 0, 0, bits);
        for (uint32_t i = 0; i < 16u; i++) {
            float decoded;

            memcpy(&decoded, &bits[i], sizeof(decoded));
            squared_error += pow((double)decoded - (double)bf16_to_float(values[i]), 2.0);
            squared_signal += pow((double)bf16_to_float(values[i]), 2.0);
        }
    }
    printf("  %s: relative RMS error on a uniform row %.4f (record only)\n", name,
           sqrt(squared_error / squared_signal));
}

/** The KV append encoding on a block-float cache: format flag, bf16 source strides. */
static void check_append_encoding(void)
{
    static const uint32_t formats[] = { kKvFormatBf16, kKvFormatBfp8, kKvFormatBfp4 };

    for (uint32_t f = 0; f < 3u; f++) {
        ft_tt_kv_cache_layout layout;
        ft_tt_op_kv_append_request request;
        uint32_t words[FT_TT_OP_MEMORY_WORDS];
        const uint32_t tile_bytes = ft_kv_bf_tile_bytes(formats[f]);
        int rc = ft_tt_kv_cache_layout_init(4, 256, 30720, tile_bytes, &layout);

        CHECK(rc == FT_TT_OK, "layout init rc %d at tile %u", rc, tile_bytes);
        if (rc != FT_TT_OK)
            continue;
        CHECK(layout.region_bytes == 4ull * 960u * 8u * tile_bytes,
              "region bytes %llu at tile %u", (unsigned long long)layout.region_bytes,
              tile_bytes);
        memset(&request, 0, sizeof(request));
        request.cache = &layout;
        request.position = 70;
        request.token_count = 1;
        request.source_row0 = 6;
        request.source_tiles = 32;
        request.source = (ft_tt_op_operand){ .x = 1, .y = 2, .address = 0x10000,
                                             .tile_bytes = FT_TT_OP_TILE_BYTES_BF16 };
        request.region = (ft_tt_op_operand){ .x = 3, .y = 4, .address = 0x200000,
                                             .tile_bytes = tile_bytes };
        rc = ft_tt_op_kv_append_parameters(&request, words);
        CHECK(rc == FT_TT_OK, "append parameters rc %d at tile %u", rc, tile_bytes);
        if (rc != FT_TT_OK)
            continue;
        CHECK(((words[FT_TT_OP_MEMORY_FLAGS] >> FT_TT_OP_MEMORY_KV_FORMAT_SHIFT) & 3u) ==
                  formats[f],
              "format flag %u, want %u", words[FT_TT_OP_MEMORY_FLAGS], formats[f]);
        CHECK(words[FT_TT_OP_MEMORY_SRC_GROUP_STRIDE] == 8u * FT_TT_OP_TILE_BYTES_BF16,
              "source group stride %u is not the bf16 strip's", words[FT_TT_OP_MEMORY_SRC_GROUP_STRIDE]);
        CHECK(words[FT_TT_OP_MEMORY_DST_GROUP_STRIDE] == (uint32_t)layout.head_stride,
              "dst group stride %u vs head stride %llu", words[FT_TT_OP_MEMORY_DST_GROUP_STRIDE],
              (unsigned long long)layout.head_stride);
    }
    printf("  append encoding: format flag + bf16 source strides on bf16/bfp8/bfp4 caches\n");
}

int main(void)
{
    printf("test_kv_block_float\n");
    check_format(kKvFormatBfp8, FT_BFP_BFP8_B, "Bfp8_b");
    check_format(kKvFormatBfp4, FT_BFP_BFP4_B, "Bfp4_b");
    report_error(kKvFormatBfp8, "Bfp8_b");
    report_error(kKvFormatBfp4, "Bfp4_b");
    check_append_encoding();
    printf("%d checks, %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
