/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_bfp_split.c
 * @brief Implementation of the BFP4_B high + BFP2_B residual split (qwen36 plan 1.4b). The
 *   per-datum encodes and decodes are ft_bfp.c's own, so the high tile is byte-for-byte what
 *   ft_bfp_pack_tiles(FT_BFP_BFP4_B) writes; only the residual is new.
 *
 * @status New 2026-10-28. test_bfp_split green.
 * @issues None
 * @todo None
 */

#include "ft_bfp_split.h"

#include <math.h>
#include <string.h>

#include "ft_bfp.h"

#define BLOCK_VALUES FT_BFP_BLOCK_VALUES
#define TILE_SIDE 32u
#define FACE_SIDE 16u
#define TILE_VALUES (TILE_SIDE * TILE_SIDE)
#define TILE_BLOCKS (TILE_VALUES / BLOCK_VALUES)

/** One quantized block: its two exponent bytes, 16 datums each, and the values they decode to. */
typedef struct split_block {
    uint8_t high_exponent;
    uint8_t low_exponent;
    uint8_t high_datum[BLOCK_VALUES];
    uint8_t low_datum[BLOCK_VALUES];
    float value[BLOCK_VALUES];
} split_block;

static float float_of_bits(uint32_t bits)
{
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

/** Residual datums under exponent byte `low_exponent`; returns the block's squared error. */
static double quantize_residual(const float *in, const float *high_value, uint8_t low_exponent,
                                uint8_t *datum_out, float *value_out)
{
    double unit = ldexp(1.0, (int)low_exponent - 127), error = 0.0;

    for (uint32_t i = 0; i < BLOCK_VALUES; i++) {
        double residual = (double)in[i] - (double)high_value[i];
        uint8_t magnitude = (low_exponent != 0u && fabs(residual) > 0.5 * unit) ? 1u : 0u;
        double low = magnitude ? (residual < 0.0 ? -unit : unit) : 0.0;

        datum_out[i] = (uint8_t)(magnitude ? ((residual < 0.0 ? 2u : 0u) | 1u) : 0u);
        value_out[i] = (float)((double)high_value[i] + low);
        error += ((double)in[i] - (double)value_out[i]) * ((double)in[i] - (double)value_out[i]);
    }
    return error;
}

static void quantize_block(ft_bfp_split_residual residual, const float *in, split_block *block)
{
    uint32_t bits[BLOCK_VALUES];
    float source[BLOCK_VALUES], high_value[BLOCK_VALUES], trial_value[BLOCK_VALUES];
    uint8_t trial_datum[BLOCK_VALUES];
    double error;

    for (uint32_t i = 0; i < BLOCK_VALUES; i++) {
        source[i] = in[i];
        memcpy(&bits[i], &source[i], sizeof(bits[i]));
    }
    block->high_exponent = ft_bfp_block_shared_exponent(FT_BFP_BFP4_B, bits);
    for (uint32_t i = 0; i < BLOCK_VALUES; i++) {
        block->high_datum[i] = ft_bfp_convert_float_bits(FT_BFP_BFP4_B, bits[i], block->high_exponent);
        high_value[i] = float_of_bits(ft_bfp_convert_datum(FT_BFP_BFP4_B, block->high_datum[i], block->high_exponent));
    }
    block->low_exponent = block->high_exponent >= 4u ? (uint8_t)(block->high_exponent - 4u) : 0u;
    error = quantize_residual(source, high_value, block->low_exponent, block->low_datum, block->value);
    if (residual == FT_BFP_SPLIT_RESIDUAL_BEST && block->high_exponent >= 3u) {
        uint8_t wider = (uint8_t)(block->high_exponent - 3u);

        if (quantize_residual(source, high_value, wider, trial_datum, trial_value) < error) {
            block->low_exponent = wider;
            memcpy(block->low_datum, trial_datum, sizeof(trial_datum));
            memcpy(block->value, trial_value, sizeof(trial_value));
        }
    }
}

/** Row-major offset inside a tile of the first value of packed block `block` (ft_bfp.c order). */
static uint32_t block_origin_in_tile(uint32_t block)
{
    uint32_t face = block / FACE_SIDE, face_row = block % FACE_SIDE;

    return ((face / 2u) * FACE_SIDE + face_row) * TILE_SIDE + (face % 2u) * FACE_SIDE;
}

static int residual_mode_valid(ft_bfp_split_residual residual)
{
    return residual == FT_BFP_SPLIT_RESIDUAL_E4 || residual == FT_BFP_SPLIT_RESIDUAL_BEST;
}

int ft_bfp_split_pack_tiles(ft_bfp_split_residual residual, const float *tiles, size_t tile_count,
                            void *high_out, void *low_out)
{
    if (!residual_mode_valid(residual))
        return FT_BFP_EINVAL;
    if (tile_count == 0)
        return FT_BFP_OK;
    if (!tiles || !high_out || !low_out)
        return FT_BFP_EINVAL;
    for (size_t t = 0; t < tile_count; t++) {
        unsigned char *high = (unsigned char *)high_out + t * FT_BFP_SPLIT_HIGH_TILE_BYTES;
        unsigned char *low = (unsigned char *)low_out + t * FT_BFP_SPLIT_LOW_TILE_BYTES;

        memset(high, 0, FT_BFP_SPLIT_HIGH_TILE_BYTES);
        memset(low, 0, FT_BFP_SPLIT_LOW_TILE_BYTES);
        for (uint32_t b = 0; b < TILE_BLOCKS; b++) {
            split_block block;

            quantize_block(residual, tiles + t * TILE_VALUES + block_origin_in_tile(b), &block);
            high[b] = block.high_exponent;
            low[b] = block.low_exponent;
            for (uint32_t i = 0; i < BLOCK_VALUES; i++) {
                uint32_t index = b * BLOCK_VALUES + i;

                high[FT_BFP_TILE_EXPONENT_BYTES + index / 2u] |= (unsigned char)((block.high_datum[i] & 0x0fu)
                                                                                  << ((index % 2u) * 4u));
                low[FT_BFP_TILE_EXPONENT_BYTES + index / 4u] |= (unsigned char)((block.low_datum[i] & 0x03u)
                                                                                 << ((index % 4u) * 2u));
            }
        }
    }
    return FT_BFP_OK;
}

int ft_bfp_split_unpack_tiles(const void *high, const void *low, size_t tile_count, float *tiles_out)
{
    float low_values[TILE_VALUES];

    if (tile_count == 0)
        return FT_BFP_OK;
    if (!high || !low || !tiles_out)
        return FT_BFP_EINVAL;
    for (size_t t = 0; t < tile_count; t++) {
        float *out = tiles_out + t * TILE_VALUES;

        if (ft_bfp_unpack_tiles(FT_BFP_BFP4_B, (const unsigned char *)high + t * FT_BFP_SPLIT_HIGH_TILE_BYTES, 1, out) !=
                FT_BFP_OK ||
            ft_bfp_unpack_tiles(FT_BFP_BFP2_B, (const unsigned char *)low + t * FT_BFP_SPLIT_LOW_TILE_BYTES, 1,
                                low_values) != FT_BFP_OK)
            return FT_BFP_EINVAL;
        for (uint32_t i = 0; i < TILE_VALUES; i++)
            out[i] += low_values[i];
    }
    return FT_BFP_OK;
}

int ft_bfp_split_round_trip_blocks(ft_bfp_split_residual residual, const float *in, float *out,
                                   size_t block_count)
{
    if (!residual_mode_valid(residual))
        return FT_BFP_EINVAL;
    if (block_count == 0)
        return FT_BFP_OK;
    if (!in || !out)
        return FT_BFP_EINVAL;
    for (size_t b = 0; b < block_count; b++) {
        split_block block;

        quantize_block(residual, in + b * BLOCK_VALUES, &block);
        memcpy(out + b * BLOCK_VALUES, block.value, sizeof(block.value));
    }
    return FT_BFP_OK;
}
