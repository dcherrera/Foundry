/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_bfp_split.h
 * @brief The qwen36 plan 1.4b "no-unpack" weight format: each value stored as a BFP4_B HIGH
 *   part plus a BFP2_B LOW residual, two tiles the Tensix unpacker reads natively, so a
 *   matmul consumes the weight as two native passes accumulating into one Dst.
 *
 * VALUE. For a 16-value block (16 consecutive OUTPUT features at a fixed input, a face row of
 * the K x N tile) with largest exponent field E:
 *   high datum  = BFP4_B of the value under E (ft_bfp.c semantics: truncating align, round to
 *                 nearest even, saturate) -> a multiple of u4 = 2^(E - 129), |high| <= 7 u4;
 *   low datum   = BFP2_B under its OWN exponent byte E_low: the residual (value - high)
 *                 rounded to {-1, 0, +1} x 2^(E_low - 127), ties to 0.
 * The weight the matmul sees is high + low (exact in fp32). E_low = E - 4 (plan 1.4b: the
 * BFP6_B step) unless `FT_BFP_SPLIT_RESIDUAL_BEST` is asked for, which picks E - 3 or E - 4 per
 * block by the smaller squared error. A block whose E < 4 gets a zero residual.
 *
 * IT IS NOT BFP6_B. The representable set under E - 4 is {4k - 1, 4k, 4k + 1} x 2^(E - 131):
 * values congruent to 2 mod 4 steps of BFP6_B are not representable, and |high + low| tops out
 * at 29 steps where BFP6_B reaches 31. So its quality is measured, never assumed (spec §8).
 *
 * BYTES. 576 (BFP4_B) + 320 (BFP2_B) = 896 B a 32x32 tile = 7.0 bits a weight, against 832 B
 * (6.5) for BFP6_B: the two exponent arrays are the cost.
 *
 * @status New 2026-10-28 (qwen36 Phase 1.4b). Bit-for-bit pinned by tests/test_bfp_split.c.
 * @issues None
 * @todo None
 */

#ifndef FT_BFP_SPLIT_H
#define FT_BFP_SPLIT_H

#include <stddef.h>

/** How the BFP2_B residual's exponent byte is chosen, per block. */
typedef enum {
    FT_BFP_SPLIT_RESIDUAL_E4 = 0,  /**< always E - 4 (plan 1.4b as written) */
    FT_BFP_SPLIT_RESIDUAL_BEST = 1 /**< E - 3 or E - 4, whichever has the smaller block SSE */
} ft_bfp_split_residual;

/** Bytes of the high (BFP4_B) and low (BFP2_B) tiles. */
#define FT_BFP_SPLIT_HIGH_TILE_BYTES 576u
#define FT_BFP_SPLIT_LOW_TILE_BYTES 320u

/**
 * Pack `tile_count` row-major 32x32 fp32 tiles into a high tile stream (576 B a tile) and a low
 * tile stream (320 B a tile). Returns FT_BFP_OK, or FT_BFP_EINVAL for a NULL buffer or an
 * unknown residual mode.
 */
int ft_bfp_split_pack_tiles(ft_bfp_split_residual residual, const float *tiles, size_t tile_count,
                            void *high_out, void *low_out);

/** The weight the card multiplies by: decode(high) + decode(low), row-major fp32 tiles. */
int ft_bfp_split_unpack_tiles(const void *high, const void *low, size_t tile_count, float *tiles_out);

/**
 * The same quantization on `block_count` CONTIGUOUS 16-value blocks (the eval's codec layout:
 * caller transposes so a block is 16 output features). `out` gets high + low.
 */
int ft_bfp_split_round_trip_blocks(ft_bfp_split_residual residual, const float *in, float *out,
                                   size_t block_count);

#endif
