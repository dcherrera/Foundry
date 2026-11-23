/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_attention_splitk_model.h
 * @brief Host model of SPLIT-K flash-decode attention (resident_attention_protocol.h §7) at
 *   the kernel's own rounding points, beside a float64 reference.
 *
 * Three answers to one problem:
 *   - ft_tt_splitk_reference(): attention from its definition, in double, per row live length;
 *   - ft_tt_splitk_lanes_model(): the ONE-CORE descriptor the lanes path sends today (one
 *     chunk over the whole live range, normalised on the core);
 *   - ft_tt_splitk_model(): the split path — one PARTIAL per chunk, then the MERGE's online
 *     fold over the partials in chunk order, then out = O / l.
 *
 * Both models replay the compute kernel step for step: the same block loop, the same mask
 * (dead positions -1e30 before the max), the same running-max seed, and a STORE after every
 * value the kernel packs into a circular buffer. `FT_TT_SPLITK_STORE_BF16` rounds there to
 * bf16, as the Float16_b CBs do; `FT_TT_SPLITK_STORE_FP32` keeps fp32, which isolates the
 * algorithm (chunking, partials, merge) from the storage format.
 *
 * Rows are independent in every tile op the kernel runs (row max, row sum, column
 * broadcast), so a verify's packed rows (p * q_rows + h) are modelled one row at a time,
 * each with its own live length.
 *
 * @status Used by tests/test_attention_splitk.c.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_ATTENTION_SPLITK_MODEL_H
#define FT_TT_ATTENTION_SPLITK_MODEL_H

#include <stdbool.h>
#include <stdint.h>

/** Where the kernel packs a value into a CB, round it to this. */
typedef enum ft_tt_splitk_store {
    FT_TT_SPLITK_STORE_FP32 = 0,
    FT_TT_SPLITK_STORE_BF16 = 1
} ft_tt_splitk_store;

/** One chunk: position tiles [first_tile, first_tile + tile_count) of one KV head. */
typedef struct ft_tt_splitk_chunk {
    uint32_t first_tile;
    uint32_t tile_count;
} ft_tt_splitk_chunk;

/** One problem. K and V are already rounded to what the cache holds (bf16 or block float). */
typedef struct ft_tt_splitk_problem {
    uint32_t rows;              /**< query rows: Q heads x verify positions */
    uint32_t head_dim;          /**< elements, a multiple of 32, at most 256 */
    const uint32_t *live;       /**< rows: positions row r attends to (its causal limit) */
    const float *query;         /**< rows * head_dim, bf16-rounded, NOT yet scaled */
    const float *keys;          /**< capacity_positions * head_dim */
    const float *values;        /**< capacity_positions * head_dim */
    uint32_t capacity_positions; /**< readable K/V rows: the padded windows must fit */
} ft_tt_splitk_problem;

/** Attention from its definition, in double. out: rows * head_dim. */
bool ft_tt_splitk_reference(const ft_tt_splitk_problem *problem, float *out);

/** Today's one-core descriptor at the kernel's rounding points. */
bool ft_tt_splitk_lanes_model(const ft_tt_splitk_problem *problem, uint32_t block_tiles,
                              ft_tt_splitk_store store, float *out);

/**
 * The split path at the kernel's rounding points: a partial per chunk, merged in chunk order.
 *
 * @param chunks       must cover [0, live tiles of the longest row) exactly once, in order,
 *                     with every row's causal limit in the last chunk
 *                     (ft_tt_splitk_chunks_are_valid())
 * @param drop_chunk   a chunk index whose partial the merge skips — the negative control;
 *                     UINT32_MAX for none
 * @return             false for a shape it cannot serve or an invalid chunk list
 */
bool ft_tt_splitk_model(const ft_tt_splitk_problem *problem, uint32_t block_tiles,
                        const ft_tt_splitk_chunk *chunks, uint32_t chunk_count,
                        ft_tt_splitk_store store, uint32_t drop_chunk, float *out);

/** Do `chunks` tile [0, live_tiles) exactly once, in order, each non-empty, with the last
 *  one holding position `last_tile_first` (every row's causal limit)? */
bool ft_tt_splitk_chunks_are_valid(const ft_tt_splitk_chunk *chunks, uint32_t chunk_count,
                                   uint32_t live_tiles, uint32_t last_tile_first);

/** Round `count` values (a multiple of 16, bf16-rounded already) through Bfp8_b, or Bfp4_b
 *  when `bfp4`: one shared exponent per 16 consecutive values (a face row), as the KV append
 *  packs a K or V row and as the unpacker reads it back. */
void ft_tt_splitk_round_block_float(float *values, uint32_t count, bool bfp4);

#endif /* FT_TT_ATTENTION_SPLITK_MODEL_H */
