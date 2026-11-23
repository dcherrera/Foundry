/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_attention.c
 * @brief Implementation of flash-decode attention's host driver. See ft_tt_op_attention.h,
 *   which carries the decision that one core owns a KV GROUP rather than a Q head, and the
 *   capacity argument that makes the online softmax a requirement.
 *
 * Everything here is PURE — no device access, no allocation. The enqueue itself is
 * ft_tt_queue_write_op_parameters() plus ft_tt_queue_enqueue(), which the caller does,
 * because a resident op has no launch of its own to own.
 *
 * @status See docs/tt_bringup_log.md and the Phase 6 attention report.
 * @issues None
 * @todo None
 */

#include "ft_tt_op_attention.h"

#include <math.h>
#include <string.h>

#include "convert/ft_float_convert.h"
#include "ft_tt_device.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_tilize.h"

/** K and V hold TWO blocks: the reader fills one while the compute thread drains the
 *  other. Two is also what makes `pages % depth == 0` hold with a `depth`-sized push. */
#define ATTENTION_KV_BLOCKS 2u

/** Pages in the buffers that carry one tile at a time. */
#define ATTENTION_CB_PAGES_SMALL 1u

/** Elements in one 32x32 tile. */
#define ATTENTION_TILE_ELEMENTS (FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM)

/** `(y << 16) | x`, the packed coordinate the parameter block carries. */
static uint32_t pack_coordinate(const ft_tt_op_operand *operand)
{
    return ((uint32_t)operand->y << 16) | (uint32_t)operand->x;
}

/** Is `operand` filled in, and does it carry the tile size the whole op runs at? */
static bool operand_is_valid(const ft_tt_op_operand *operand, uint32_t tile_bytes)
{
    return operand->tile_bytes == tile_bytes && tile_bytes != 0u;
}

uint32_t ft_tt_op_attention_live_tiles(uint32_t kv_positions)
{
    return (kv_positions + FT_TT_OP_TILE_DIM - 1u) / FT_TT_OP_TILE_DIM;
}

uint32_t ft_tt_op_attention_padded_tiles(uint32_t kv_positions, uint32_t block_tiles)
{
    uint32_t live;
    uint32_t window;

    if (block_tiles == 0u)
        return 0u;
    live = ft_tt_op_attention_live_tiles(kv_positions);
    window = ATTENTION_KV_BLOCKS * block_tiles;
    return ((live + window - 1u) / window) * window;
}

/** Pages each of the sixteen buffers wants, in ft_tt_op_attention_place_cbs()' order. This
 *  is the ONE place the page counts live; the L1 arithmetic and the placement both read it,
 *  so a residency's cost and its layout cannot disagree. */
static void page_counts(const ft_tt_op_attention_residency *residency,
                        uint32_t out_pages[FT_TT_OP_ATTENTION_CB_COUNT],
                        unsigned out_index[FT_TT_OP_ATTENTION_CB_COUNT])
{
    const uint32_t depth = residency->block_tiles * residency->dim_tiles;

    out_index[0] = FT_TT_OP_ATTENTION_CB_Q;
    out_pages[0] = residency->dim_tiles;
    out_index[1] = FT_TT_OP_ATTENTION_CB_QS;
    out_pages[1] = residency->dim_tiles;
    out_index[2] = FT_TT_OP_ATTENTION_CB_K;
    out_pages[2] = ATTENTION_KV_BLOCKS * depth;
    out_index[3] = FT_TT_OP_ATTENTION_CB_V;
    out_pages[3] = ATTENTION_KV_BLOCKS * depth;
    out_index[4] = FT_TT_OP_ATTENTION_CB_CONST;
    out_pages[4] = FT_TT_OP_ATTENTION_CONST_PAGES;
    out_index[5] = FT_TT_OP_ATTENTION_CB_S;
    out_pages[5] = residency->block_tiles;
    out_index[6] = FT_TT_OP_ATTENTION_CB_SCRATCH;
    out_pages[6] = ATTENTION_CB_PAGES_SMALL;
    out_index[7] = FT_TT_OP_ATTENTION_CB_M_A;
    out_pages[7] = ATTENTION_CB_PAGES_SMALL;
    out_index[8] = FT_TT_OP_ATTENTION_CB_M_B;
    out_pages[8] = ATTENTION_CB_PAGES_SMALL;
    out_index[9] = FT_TT_OP_ATTENTION_CB_L;
    out_pages[9] = ATTENTION_CB_PAGES_SMALL;
    out_index[10] = FT_TT_OP_ATTENTION_CB_LSUM;
    out_pages[10] = ATTENTION_CB_PAGES_SMALL;
    out_index[11] = FT_TT_OP_ATTENTION_CB_C;
    out_pages[11] = ATTENTION_CB_PAGES_SMALL;
    out_index[12] = FT_TT_OP_ATTENTION_CB_O;
    out_pages[12] = residency->dim_tiles;
    out_index[13] = FT_TT_OP_ATTENTION_CB_PV;
    out_pages[13] = residency->dim_tiles;
    /* The reader's private staging for the group's raw Q tiles. Declared as a circular
     * buffer so its L1 comes out of the arena allocator like everything else's and shows up
     * in the same accounting — nothing flows through it, and it is never pushed or popped. */
    out_index[14] = FT_TT_OP_ATTENTION_CB_QSTAGE;
    out_pages[14] = residency->q_rows * residency->dim_tiles;
    out_index[15] = FT_TT_OP_ATTENTION_CB_OUT;
    out_pages[15] = 2u * residency->dim_tiles;
}

/** Is this a residency that can be placed at all, before the arena is consulted? */
static bool residency_is_valid(const ft_tt_op_attention_residency *residency)
{
    if (!residency)
        return false;
    if (residency->q_rows == 0u || residency->q_rows > FT_TT_OP_TILE_DIM)
        return false;
    if (residency->dim_tiles == 0u || residency->block_tiles == 0u)
        return false;
    if (residency->tile_bytes != FT_TT_OP_TILE_BYTES_BF16 &&
        residency->tile_bytes != FT_TT_OP_TILE_BYTES_FP32)
        return false;
    return true;
}

/** The K/V CBs' page size: the cache's tile, which may be block float (ft_tt_op_memory.h
 *  §1.6); every other CB keeps the residency's. */
static bool kv_tile_bytes_is_valid(uint32_t kv_tile_bytes)
{
    return kv_tile_bytes == FT_TT_OP_TILE_BYTES_BF16 || kv_tile_bytes == FT_TT_OP_TILE_BYTES_FP32 ||
           kv_tile_bytes == FT_TT_OP_TILE_BYTES_BFP8_B ||
           kv_tile_bytes == FT_TT_OP_TILE_BYTES_BFP4_B;
}

/** K and V may each be their own format (a mixed cache: K Bfp8_b / V Bfp4_b and the like). */
static uint32_t page_bytes_of(const ft_tt_op_attention_residency *residency, unsigned cb_index,
                              uint32_t k_tile_bytes, uint32_t v_tile_bytes)
{
    if (cb_index == FT_TT_OP_ATTENTION_CB_K)
        return k_tile_bytes;
    if (cb_index == FT_TT_OP_ATTENTION_CB_V)
        return v_tile_bytes;
    return residency->tile_bytes;
}

size_t ft_tt_op_attention_l1_bytes_mixed_kv(const ft_tt_op_attention_residency *residency,
                                            uint32_t k_tile_bytes, uint32_t v_tile_bytes)
{
    uint32_t pages[FT_TT_OP_ATTENTION_CB_COUNT];
    unsigned indices[FT_TT_OP_ATTENTION_CB_COUNT];
    size_t total = 0;
    unsigned index;

    if (!residency_is_valid(residency) || !kv_tile_bytes_is_valid(k_tile_bytes) ||
        !kv_tile_bytes_is_valid(v_tile_bytes))
        return 0;
    page_counts(residency, pages, indices);
    for (index = 0; index < FT_TT_OP_ATTENTION_CB_COUNT; index++) {
        /* The allocator rounds each buffer up to 64 bytes, so the estimate must too or it
         * under-reports the residency that will actually be placed. */
        const size_t bytes = (size_t)pages[index] *
                             page_bytes_of(residency, indices[index], k_tile_bytes, v_tile_bytes);

        total += (bytes + 63u) & ~(size_t)63u;
    }
    return total;
}

size_t ft_tt_op_attention_l1_bytes_kv(const ft_tt_op_attention_residency *residency,
                                      uint32_t kv_tile_bytes)
{
    return ft_tt_op_attention_l1_bytes_mixed_kv(residency, kv_tile_bytes, kv_tile_bytes);
}

size_t ft_tt_op_attention_l1_bytes(const ft_tt_op_attention_residency *residency)
{
    return residency ? ft_tt_op_attention_l1_bytes_kv(residency, residency->tile_bytes) : 0;
}

uint32_t ft_tt_op_attention_max_block_tiles(uint32_t q_rows, uint32_t dim_tiles,
                                            uint32_t tile_bytes)
{
    ft_tt_op_attention_residency residency;
    uint32_t best = 0u;
    uint32_t candidate;

    residency.q_rows = q_rows;
    residency.dim_tiles = dim_tiles;
    residency.tile_bytes = tile_bytes;
    /* Searched rather than solved: the arena cost is piecewise because every buffer is
     * rounded up to 64 bytes, and a closed form would have to reproduce that rounding —
     * which is exactly the kind of second copy of an arithmetic that drifts. 256 is far
     * past anything that fits. */
    for (candidate = 1u; candidate <= 256u; candidate++) {
        residency.block_tiles = candidate;
        if (!residency_is_valid(&residency))
            return 0u;
        if (ft_tt_op_attention_l1_bytes(&residency) >
            (size_t)(FT_TT_OP_CB_ARENA_END - FT_TT_OP_CB_ARENA_BASE))
            break;
        best = candidate;
    }
    return best;
}

int ft_tt_op_attention_place_cbs_mixed_kv(
    const ft_tt_op_attention_residency *residency, uint32_t k_tile_bytes, uint32_t v_tile_bytes,
    ft_tt_launch_cb_entry out_entries[FT_TT_OP_ATTENTION_CB_COUNT], void *buffer,
    size_t buffer_bytes, uint64_t *out_mask)
{
    ft_tt_op_cb_request requests[FT_TT_OP_ATTENTION_CB_COUNT];
    uint32_t pages[FT_TT_OP_ATTENTION_CB_COUNT];
    unsigned indices[FT_TT_OP_ATTENTION_CB_COUNT];
    unsigned index;

    if (!residency_is_valid(residency) || !kv_tile_bytes_is_valid(k_tile_bytes) ||
        !kv_tile_bytes_is_valid(v_tile_bytes) || !out_entries || !buffer || !out_mask)
        return FT_TT_EINVAL;

    page_counts(residency, pages, indices);
    for (index = 0; index < FT_TT_OP_ATTENTION_CB_COUNT; index++) {
        requests[index].index = indices[index];
        requests[index].page_bytes =
            page_bytes_of(residency, indices[index], k_tile_bytes, v_tile_bytes);
        requests[index].page_count = pages[index];
    }
    return ft_tt_op_cb_place(requests, FT_TT_OP_ATTENTION_CB_COUNT, out_entries, buffer,
                             buffer_bytes, out_mask);
}

int ft_tt_op_attention_place_cbs_kv(const ft_tt_op_attention_residency *residency,
                                    uint32_t kv_tile_bytes,
                                    ft_tt_launch_cb_entry out_entries[FT_TT_OP_ATTENTION_CB_COUNT],
                                    void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    return ft_tt_op_attention_place_cbs_mixed_kv(residency, kv_tile_bytes, kv_tile_bytes,
                                                 out_entries, buffer, buffer_bytes, out_mask);
}

int ft_tt_op_attention_place_cbs(const ft_tt_op_attention_residency *residency,
                                 ft_tt_launch_cb_entry out_entries[FT_TT_OP_ATTENTION_CB_COUNT],
                                 void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    if (!residency)
        return FT_TT_EINVAL;
    return ft_tt_op_attention_place_cbs_kv(residency, residency->tile_bytes, out_entries,
                                           buffer, buffer_bytes, out_mask);
}

/**
 * A split-K MERGE (resident_attention_protocol.h §7): word 0 as decode / ROWS (the writer
 * scatters with it), word 2 the partial count, Q the partials, OUT the output. K, V and the
 * constants are unread, so their words stay zero.
 */
static int merge_parameters(const ft_tt_op_attention_request *request,
                            uint32_t out_words[FT_TT_OP_ATTENTION_WORDS])
{
    const uint32_t tile_bytes = request->query.tile_bytes;
    const bool rows = (request->flags & (uint32_t)FT_TT_OP_ATTENTION_FLAG_ROWS) != 0u;

    if ((request->flags & ~(uint32_t)(FT_TT_OP_ATTENTION_FLAG_MERGE |
                                      FT_TT_OP_ATTENTION_FLAG_ROWS)) != 0u)
        return FT_TT_EINVAL;
    if (request->q_rows == 0u || request->q_rows > FT_TT_OP_TILE_DIM ||
        request->dim_tiles == 0u || request->block_tiles == 0u ||
        request->query_row >= FT_TT_OP_TILE_DIM || request->partial_count == 0u ||
        request->partial_count > FT_TT_OP_ATTENTION_MAX_PARTIALS)
        return FT_TT_EINVAL;
    if (rows && (request->positions == 0u ||
                 request->q_rows * request->positions > FT_TT_OP_TILE_DIM ||
                 request->query_row + request->positions > FT_TT_OP_TILE_DIM))
        return FT_TT_EINVAL;
    if (!operand_is_valid(&request->query, tile_bytes) ||
        !operand_is_valid(&request->output, tile_bytes))
        return FT_TT_EINVAL;

    memset(out_words, 0, FT_TT_OP_ATTENTION_WORDS * sizeof(out_words[0]));
    out_words[FT_TT_OP_ATTENTION_HEADS] = request->q_rows | (request->query_row << 8) |
                                          (request->flags << 16) |
                                          (rows ? (request->positions & 0xFFu) << 24 : 0u);
    out_words[FT_TT_OP_ATTENTION_DIM_TILES] = request->dim_tiles;
    out_words[FT_TT_OP_ATTENTION_KV_POSITIONS] = request->partial_count;
    out_words[FT_TT_OP_ATTENTION_BLOCK_TILES] = request->block_tiles;
    out_words[FT_TT_OP_ATTENTION_CAPACITY_TILES] = request->capacity_tiles;
    out_words[FT_TT_OP_ATTENTION_Q_COORD] = pack_coordinate(&request->query);
    out_words[FT_TT_OP_ATTENTION_Q_BASE] = request->query.address;
    out_words[FT_TT_OP_ATTENTION_OUT_COORD] = pack_coordinate(&request->output);
    out_words[FT_TT_OP_ATTENTION_OUT_BASE] = request->output.address;
    return FT_TT_OK;
}

uint32_t ft_tt_op_attention_partial_bytes(uint32_t dim_tiles, uint32_t tile_bytes)
{
    return (dim_tiles + FT_TT_OP_ATTENTION_PARTIAL_STAT_TILES) * tile_bytes;
}

int ft_tt_op_attention_parameters(const ft_tt_op_attention_request *request,
                                  uint32_t out_words[FT_TT_OP_ATTENTION_WORDS])
{
    uint32_t tile_bytes;
    uint32_t padded;

    if (!request || !out_words)
        return FT_TT_EINVAL;
    if ((request->flags & (uint32_t)FT_TT_OP_ATTENTION_FLAG_MERGE) != 0u)
        return merge_parameters(request, out_words);
    /* A PARTIAL is a decode or ROWS descriptor with its output emitted un-normalised; every
     * check below is that descriptor's. */
    if ((request->flags & (uint32_t)FT_TT_OP_ATTENTION_FLAG_PARTIAL) != 0u &&
        (request->flags & (uint32_t)FT_TT_OP_ATTENTION_FLAG_CHUNK) != 0u)
        return FT_TT_EINVAL;
    tile_bytes = request->query.tile_bytes;
    if (request->q_rows == 0u || request->q_rows > FT_TT_OP_TILE_DIM)
        return FT_TT_EINVAL;
    if (request->dim_tiles == 0u || request->block_tiles == 0u || request->kv_positions == 0u)
        return FT_TT_EINVAL;
    if (request->query_row >= FT_TT_OP_TILE_DIM)
        return FT_TT_EINVAL;
    if ((request->flags & ~(uint32_t)(FT_TT_OP_ATTENTION_FLAG_CHUNK |
                                      FT_TT_OP_ATTENTION_FLAG_ROWS |
                                      FT_TT_OP_ATTENTION_FLAG_PARTIAL)) != 0u)
        return FT_TT_EINVAL;
    if ((request->flags & (uint32_t)FT_TT_OP_ATTENTION_FLAG_ROWS) != 0u &&
        (request->positions == 0u || request->q_rows * request->positions > FT_TT_OP_TILE_DIM ||
         request->query_row + request->positions > FT_TT_OP_TILE_DIM ||
         request->kv_positions < request->positions ||
         (request->kv_positions - request->positions) / FT_TT_OP_TILE_DIM !=
             (request->kv_positions - 1u) / FT_TT_OP_TILE_DIM ||
         request->rows_mask_address == 0u))
        return FT_TT_EINVAL;
    /* THE CHUNK GEOMETRY, REFUSED HERE AS WELL AS ON THE DEVICE. The chunk's C positions
     * END the live range and must START on a position-tile boundary, so the chunk occupies
     * exactly the LAST LIVE TILE and row t is position `first + t`. A chunk straddling two
     * position tiles would need two masks and there is one mask slot — and it would be a
     * wrong mask on a live tile, i.e. a plausible wrong answer with nothing out of range. */
    if ((request->flags & (uint32_t)FT_TT_OP_ATTENTION_FLAG_CHUNK) != 0u) {
        if (request->query_row != 0u)
            return FT_TT_EINVAL;
        if (request->kv_positions < request->q_rows)
            return FT_TT_EINVAL;
        if (((request->kv_positions - request->q_rows) % FT_TT_OP_TILE_DIM) != 0u)
            return FT_TT_EINVAL;
    }
    /* K and V carry their CACHE's tile (block float allowed, §1.6, and K and V may differ:
     * the program's CB table says which pair it reads); the rest the residency's. */
    if (!operand_is_valid(&request->query, tile_bytes) ||
        !kv_tile_bytes_is_valid(request->keys.tile_bytes) ||
        !kv_tile_bytes_is_valid(request->values.tile_bytes) ||
        !operand_is_valid(&request->constants, tile_bytes) ||
        !operand_is_valid(&request->output, tile_bytes))
        return FT_TT_EINVAL;

    /* The PADDED window is what the reader fetches, so it — not the live length — is what
     * has to fit the region. Getting this wrong is a read outside the KV cache. */
    padded = ft_tt_op_attention_padded_tiles(request->kv_positions, request->block_tiles);
    if (padded == 0u || padded > request->capacity_tiles)
        return FT_TT_EINVAL;

    memset(out_words, 0, FT_TT_OP_ATTENTION_WORDS * sizeof(out_words[0]));
    out_words[FT_TT_OP_ATTENTION_HEADS] =
        request->q_rows | (request->query_row << 8) | (request->flags << 16) |
        ((request->flags & (uint32_t)FT_TT_OP_ATTENTION_FLAG_ROWS) != 0u
             ? (request->positions & 0xFFu) << 24 : 0u);
    if ((request->flags & (uint32_t)FT_TT_OP_ATTENTION_FLAG_ROWS) != 0u)
        out_words[FT_TT_OP_ATTENTION_RESERVED] = request->rows_mask_address;
    out_words[FT_TT_OP_ATTENTION_DIM_TILES] = request->dim_tiles;
    out_words[FT_TT_OP_ATTENTION_KV_POSITIONS] = request->kv_positions;
    out_words[FT_TT_OP_ATTENTION_BLOCK_TILES] = request->block_tiles;
    out_words[FT_TT_OP_ATTENTION_CAPACITY_TILES] = request->capacity_tiles;
    out_words[FT_TT_OP_ATTENTION_Q_COORD] = pack_coordinate(&request->query);
    out_words[FT_TT_OP_ATTENTION_Q_BASE] = request->query.address;
    out_words[FT_TT_OP_ATTENTION_K_COORD] = pack_coordinate(&request->keys);
    out_words[FT_TT_OP_ATTENTION_K_BASE] = request->keys.address;
    out_words[FT_TT_OP_ATTENTION_V_COORD] = pack_coordinate(&request->values);
    out_words[FT_TT_OP_ATTENTION_V_BASE] = request->values.address;
    out_words[FT_TT_OP_ATTENTION_CONST_COORD] = pack_coordinate(&request->constants);
    out_words[FT_TT_OP_ATTENTION_CONST_BASE] = request->constants.address;
    out_words[FT_TT_OP_ATTENTION_OUT_COORD] = pack_coordinate(&request->output);
    out_words[FT_TT_OP_ATTENTION_OUT_BASE] = request->output.address;
    return FT_TT_OK;
}

size_t ft_tt_op_attention_constants_bytes(uint32_t tile_bytes)
{
    return (size_t)FT_TT_OP_ATTENTION_CONST_TILES * tile_bytes;
}

/** Fill one whole face-ordered tile with `value`. Only column 0 of the scaler is read by
 *  the reduce LLK and only the mask's columns matter for a mask, but filling the whole tile
 *  is a superset and costs nothing at build time. */
static void fill_tile(uint16_t *tile, float value)
{
    const uint16_t encoded = ft_bf16_from_float(value);
    uint32_t index;

    for (index = 0; index < ATTENTION_TILE_ELEMENTS; index++)
        tile[index] = encoded;
}

/** Mask tile `tail`: 0 in columns below `tail`, the mask value at and above it. `tail` 0 is
 *  every column dead, which is what a position tile entirely past the end needs. */
static void fill_mask_tile(uint16_t *tile, uint32_t tail)
{
    const uint16_t dead = ft_bf16_from_float(FT_TT_OP_ATTENTION_MASK_VALUE);
    const uint16_t live = ft_bf16_from_float(0.0f);
    uint32_t row;

    for (row = 0; row < FT_TT_OP_TILE_DIM; row++) {
        uint32_t column;

        for (column = 0; column < FT_TT_OP_TILE_DIM; column++)
            tile[ft_tt_tile_face_index(row, column)] = (column < tail) ? live : dead;
    }
}

/** Causal mask tile for a chunk of `live` positions: element (row r, column c) is 0 when
 *  `c <= r && c < live` and the mask value otherwise.
 *
 *  ROW r IS POSITION `first + r` AND COLUMN c IS POSITION `first + c`, which is the whole
 *  of why one constant tile is the entire causality mechanism: `c <= r` is "position r may
 *  attend to position c", and `c < live` kills the columns of a ragged chunk that hold no
 *  position at all. Rows at and above `live` are the strip's padding rows; they see the
 *  whole chunk, which keeps their softmax denominator non-zero and their output finite —
 *  garbage in a row every consumer discards, rather than a NaN that could propagate. */
static void fill_causal_mask_tile(uint16_t *tile, uint32_t live)
{
    const uint16_t dead = ft_bf16_from_float(FT_TT_OP_ATTENTION_MASK_VALUE);
    const uint16_t alive = ft_bf16_from_float(0.0f);
    uint32_t row;

    for (row = 0; row < FT_TT_OP_TILE_DIM; row++) {
        uint32_t column;

        for (column = 0; column < FT_TT_OP_TILE_DIM; column++)
            tile[ft_tt_tile_face_index(row, column)] =
                (column <= row && column < live) ? alive : dead;
    }
}

int ft_tt_op_attention_build_constants(uint32_t head_dim, uint32_t tile_bytes, void *out,
                                       size_t out_bytes)
{
    uint16_t *tiles = (uint16_t *)out;
    uint32_t tail;

    if (!out || head_dim == 0u || tile_bytes != FT_TT_OP_TILE_BYTES_BF16)
        return FT_TT_EINVAL;
    if (out_bytes < ft_tt_op_attention_constants_bytes(tile_bytes))
        return FT_TT_EINVAL;

    memset(out, 0, ft_tt_op_attention_constants_bytes(tile_bytes));
    fill_tile(tiles + FT_TT_OP_ATTENTION_CONST_SCALER_TILE * ATTENTION_TILE_ELEMENTS, 1.0f);
    fill_tile(tiles + FT_TT_OP_ATTENTION_CONST_SCALE_TILE * ATTENTION_TILE_ELEMENTS,
              (float)(1.0 / sqrt((double)head_dim)));
    for (tail = 0; tail < FT_TT_OP_TILE_DIM; tail++)
        fill_mask_tile(tiles + (FT_TT_OP_ATTENTION_CONST_MASK_TILE + tail) *
                                   ATTENTION_TILE_ELEMENTS,
                       tail);
    /* Indexed by C - 1, so C = 1..32 rather than 0..31: a chunk of no positions is not a
     * descriptor, and the all-dead tile already exists for a tile past the end. */
    for (tail = 0; tail < FT_TT_OP_TILE_DIM; tail++)
        fill_causal_mask_tile(tiles + (FT_TT_OP_ATTENTION_CONST_CAUSAL_TILE + tail) *
                                          ATTENTION_TILE_ELEMENTS,
                              tail + 1u);
    return FT_TT_OK;
}
