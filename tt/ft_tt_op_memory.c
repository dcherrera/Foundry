/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_memory.c
 * @brief Implementation of the memory op family's host driver — the KV cache layout
 *   arithmetic and the parameter encoders for EMBEDDING and KV_APPEND.
 *
 * See ft_tt_op_memory.h. PART 1 of that header is the KV layout contract; everything in
 * this file is that contract as executable arithmetic, so a reader who trusts the code and
 * a reader who trusts the prose get the same answer.
 *
 * Everything here is PURE — no device access, no allocation. The enqueue itself is
 * ft_tt_queue_write_op_parameters() plus ft_tt_queue_enqueue(), which the caller does,
 * because a resident op has no launch of its own to own.
 *
 * @status Off-hardware suite GREEN in tests/test_memory_ops.c, on the Mac and the card host;
 *   on-card acceptance GREEN on Tensix 30 and 32, and again on Tensix 40 after the
 *   one-CB-per-group-size change — ft_tt_op_memory.h @status has the figures, including the
 *   32-byte DRAM write measurement §1.4 rests on and the negative control the CB split is
 *   proved against.
 * @issues None
 * @todo None
 */

#include "ft_tt_op_memory.h"

#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_tilize.h"

/** Blocks of `group_tiles` pages in each output CB: the writer drains one while the reader
 *  fills the other, and it is also what keeps a reserved block contiguous — every push
 *  through one CB is the same size, and the CB holds a whole multiple of it. The "same
 *  size" half is what the per-group-size CB split (ft_tt_op_memory.h) makes true. */
#define MEMORY_CB_OUT_BLOCKS 2u

/** The DRAM read alignment. A source row read must satisfy it at BOTH ends, and the whole
 *  reason the embedding table is row-major is that a face row (32 B) does not. */
#define MEMORY_DRAM_READ_ALIGNMENT 64u

/** bf16 and fp32: every STAGING buffer and every non-KV destination. */
static bool tile_size_is_supported(uint32_t tile_bytes)
{
    return tile_bytes == FT_TT_OP_TILE_BYTES_BF16 || tile_bytes == FT_TT_OP_TILE_BYTES_FP32;
}

/** A KV CACHE may also be Bfp8_b or Bfp4_b: the writer packs each appended face row, and a
 *  face row is one position (ft_tt_op_memory.h §1.6). */
static bool kv_tile_size_is_supported(uint32_t tile_bytes)
{
    return tile_size_is_supported(tile_bytes) || tile_bytes == FT_TT_OP_TILE_BYTES_BFP8_B ||
           tile_bytes == FT_TT_OP_TILE_BYTES_BFP4_B;
}

uint32_t ft_tt_kv_cache_format(const ft_tt_kv_cache_layout *layout)
{
    if (!layout)
        return FT_TT_KV_FORMAT_BF16;
    return layout->tile_bytes == FT_TT_OP_TILE_BYTES_BFP8_B   ? FT_TT_KV_FORMAT_BFP8
           : layout->tile_bytes == FT_TT_OP_TILE_BYTES_BFP4_B ? FT_TT_KV_FORMAT_BFP4
                                                             : FT_TT_KV_FORMAT_BF16;
}

/** An operand is usable when it names a tile size. The COORDINATE is not checked: it comes
 *  from ft_tt_coord and a caller that invented one has a bigger problem than a range check
 *  (spec §7.3). */
static bool operand_is_set(const ft_tt_op_operand *operand)
{
    return operand->tile_bytes != 0u;
}

/* ================================================================
 * The KV cache layout
 * ================================================================ */

int ft_tt_kv_cache_layout_init(uint32_t head_count, uint32_t head_dim,
                               uint32_t capacity_positions, uint32_t tile_bytes,
                               ft_tt_kv_cache_layout *out_layout)
{
    uint64_t head_stride;

    if (!out_layout)
        return FT_TT_EINVAL;
    if (head_count == 0u || head_dim == 0u || (head_dim % FT_TT_OP_TILE_DIM) != 0u)
        return FT_TT_EINVAL;
    if (capacity_positions == 0u)
        return FT_TT_EINVAL;
    if (!kv_tile_size_is_supported(tile_bytes)) {
        ft_tt_log_error("kv cache: tile size %u is not bf16, fp32, Bfp8_b or Bfp4_b", tile_bytes);
        return FT_TT_EINVAL;
    }

    memset(out_layout, 0, sizeof(*out_layout));
    out_layout->head_count = head_count;
    out_layout->head_dim = head_dim;
    out_layout->capacity_positions =
        ((capacity_positions + FT_TT_OP_TILE_DIM - 1u) / FT_TT_OP_TILE_DIM) * FT_TT_OP_TILE_DIM;
    out_layout->tile_bytes = tile_bytes;
    out_layout->dim_tiles = head_dim / FT_TT_OP_TILE_DIM;
    out_layout->capacity_tiles = out_layout->capacity_positions / FT_TT_OP_TILE_DIM;
    out_layout->position_tile_bytes = out_layout->dim_tiles * tile_bytes;
    head_stride = (uint64_t)out_layout->capacity_tiles * out_layout->position_tile_bytes;
    out_layout->head_stride = head_stride;
    out_layout->region_bytes = (uint64_t)head_count * head_stride;
    return FT_TT_OK;
}

static bool layout_is_initialised(const ft_tt_kv_cache_layout *layout)
{
    return layout && layout->dim_tiles != 0u && layout->capacity_tiles != 0u &&
           layout->tile_bytes != 0u;
}

uint64_t ft_tt_kv_cache_tile_offset(const ft_tt_kv_cache_layout *layout, uint32_t head,
                                    uint32_t position, uint32_t dim_tile)
{
    if (!layout_is_initialised(layout))
        return UINT64_MAX;
    if (head >= layout->head_count || position >= layout->capacity_positions ||
        dim_tile >= layout->dim_tiles)
        return UINT64_MAX;
    return (uint64_t)head * layout->head_stride +
           (uint64_t)(position / FT_TT_OP_TILE_DIM) * layout->position_tile_bytes +
           (uint64_t)dim_tile * layout->tile_bytes;
}

int ft_tt_kv_cache_run(const ft_tt_kv_cache_layout *layout, uint32_t head,
                       uint32_t first_position, uint32_t position_count, uint64_t *out_offset,
                       uint32_t *out_bytes)
{
    uint64_t bytes;

    if (!layout_is_initialised(layout) || !out_offset || !out_bytes)
        return FT_TT_EINVAL;
    if (head >= layout->head_count || position_count == 0u)
        return FT_TT_EINVAL;
    if ((first_position % FT_TT_OP_TILE_DIM) != 0u || (position_count % FT_TT_OP_TILE_DIM) != 0u)
        return FT_TT_EINVAL;
    if ((uint64_t)first_position + position_count > layout->capacity_positions)
        return FT_TT_EINVAL;

    bytes = (uint64_t)(position_count / FT_TT_OP_TILE_DIM) * layout->position_tile_bytes;
    if (bytes > UINT32_MAX)
        return FT_TT_EINVAL;
    *out_offset = (uint64_t)head * layout->head_stride +
                  (uint64_t)(first_position / FT_TT_OP_TILE_DIM) * layout->position_tile_bytes;
    *out_bytes = (uint32_t)bytes;
    return FT_TT_OK;
}

/* ================================================================
 * Tile row geometry
 * ================================================================ */

uint32_t ft_tt_op_memory_chunk_offset(uint32_t row, uint32_t half, uint32_t tile_bytes)
{
    if (row >= FT_TT_OP_TILE_DIM || half >= FT_TT_OP_MEMORY_FACE_HALVES ||
        !tile_size_is_supported(tile_bytes))
        return UINT32_MAX;
    return ((row / FT_TT_OP_MEMORY_FACE_ROWS) * 2u + half) * FT_TT_OP_MEMORY_FACE_BYTES(tile_bytes) +
           (row % FT_TT_OP_MEMORY_FACE_ROWS) * FT_TT_OP_MEMORY_CHUNK_BYTES(tile_bytes);
}

uint32_t ft_tt_op_memory_position_row(uint32_t position)
{
    return position % FT_TT_OP_TILE_DIM;
}

/** Bytes one row of ONE group occupies in a row-major source. Exactly a 32nd of the
 *  group's tiles, which is why the staging buffer is `group_tiles * tile_bytes` and holds
 *  a full 32 rows without a second size to keep in step. */
static uint32_t group_row_bytes_of(uint32_t group_tiles, uint32_t tile_bytes)
{
    return group_tiles * FT_TT_OP_MEMORY_CHUNK_BYTES(tile_bytes) * FT_TT_OP_MEMORY_FACE_HALVES;
}

/* ================================================================
 * EMBEDDING
 * ================================================================ */

int ft_tt_op_embedding_parameters(const ft_tt_op_embedding_request *request,
                                  uint32_t out_words[FT_TT_OP_MEMORY_WORDS])
{
    uint32_t group_tiles;
    uint32_t row_bytes;

    if (!request || !out_words)
        return FT_TT_EINVAL;
    if (request->token_count == 0u || request->token_count > FT_TT_OP_MEMORY_MAX_ROWS)
        return FT_TT_EINVAL;
    if (request->embedding_dim == 0u || (request->embedding_dim % FT_TT_OP_TILE_DIM) != 0u)
        return FT_TT_EINVAL;
    if (!operand_is_set(&request->table) || !operand_is_set(&request->output))
        return FT_TT_EINVAL;
    if (!tile_size_is_supported(request->table.tile_bytes) ||
        request->table.tile_bytes != request->output.tile_bytes)
        return FT_TT_EINVAL;
    if ((request->output_row0 % FT_TT_OP_TILE_DIM) + request->token_count > FT_TT_OP_TILE_DIM)
        return FT_TT_EINVAL;

    group_tiles = request->embedding_dim / FT_TT_OP_TILE_DIM;
    row_bytes = group_row_bytes_of(group_tiles, request->table.tile_bytes);
    if (request->table_row_stride < row_bytes)
        return FT_TT_EINVAL;
    /* Both ends of a source row read must satisfy the 64-byte DRAM read alignment; the row
     * length always does (a face row pair is 64 B for bf16 and 128 B for fp32), so what is
     * left to check is where the rows START. */
    if ((request->table_row_stride % MEMORY_DRAM_READ_ALIGNMENT) != 0u ||
        (request->table.address % MEMORY_DRAM_READ_ALIGNMENT) != 0u)
        return FT_TT_EINVAL;
    /* THE TWO BOUNDS THE KERNEL REFUSES WITHOUT, and neither is derivable from anything else
     * in the request: a token id is a raw wire value and `output_row0` is absolute, so
     * without these the kernel turns a wrong number into a DRAM address at full length
     * rather than into an error. A zero is refused HERE as well, because an encoder that let
     * one through would be handing the kernel a descriptor it is bound to reject. */
    if (request->table_row_count == 0u || request->output_bytes == 0u)
        return FT_TT_EINVAL;
    {
        const uint64_t group_bytes = (uint64_t)group_tiles * request->table.tile_bytes;
        const uint64_t tile_row = request->output_row0 / FT_TT_OP_TILE_DIM;

        if (tile_row * group_bytes + group_bytes > request->output_bytes)
            return FT_TT_EINVAL;
    }

    memset(out_words, 0, FT_TT_OP_MEMORY_WORDS * sizeof(out_words[0]));
    /* GATHER always: an embedding whose rows were consecutive would be a copy, not a
     * lookup, and one path is one thing to get right. A caller that wants rows in order
     * writes the ids in order. */
    out_words[FT_TT_OP_MEMORY_FLAGS] =
        (uint32_t)FT_TT_OP_MEMORY_FLAG_GATHER |
        (request->zero_fill ? (uint32_t)FT_TT_OP_MEMORY_FLAG_DST_WHOLE_TILES : 0u);
    out_words[FT_TT_OP_MEMORY_ROW_COUNT] = request->token_count;
    out_words[FT_TT_OP_MEMORY_GROUP_COUNT] = 1u;
    out_words[FT_TT_OP_MEMORY_GROUP_TILES] = group_tiles;
    out_words[FT_TT_OP_MEMORY_SRC_X] = request->table.x;
    out_words[FT_TT_OP_MEMORY_SRC_Y] = request->table.y;
    out_words[FT_TT_OP_MEMORY_SRC_BASE] = request->table.address;
    out_words[FT_TT_OP_MEMORY_SRC_ROW_STRIDE] = request->table_row_stride;
    out_words[FT_TT_OP_MEMORY_DST_X] = request->output.x;
    out_words[FT_TT_OP_MEMORY_DST_Y] = request->output.y;
    out_words[FT_TT_OP_MEMORY_DST_BASE] = request->output.address;
    out_words[FT_TT_OP_MEMORY_DST_ROW0] = request->output_row0;
    out_words[FT_TT_OP_MEMORY_SRC_ROW_LIMIT] = request->table_row_count;
    /* ONE destination group, so the bytes it owns ARE the region's extent. This is what the
     * writer bounds the tile-row offset against. */
    out_words[FT_TT_OP_MEMORY_DST_GROUP_STRIDE] = request->output_bytes;
    return FT_TT_OK;
}

/* ================================================================
 * KV_APPEND
 * ================================================================ */

/** BANKED destination: one byte of DST_X / DST_Y per bank group, heads per bank in FLAGS.
 *  Every bank region must sit at `region.address` — the kernel has one base word. */
static int kv_append_encode_banks(const ft_tt_op_kv_append_request *request,
                                  uint32_t head_count, uint32_t out_words[FT_TT_OP_MEMORY_WORDS])
{
    const uint32_t banks = request->bank_count;
    uint32_t packed_x = 0u, packed_y = 0u;

    if (banks > FT_TT_OP_MEMORY_MAX_BANKS || head_count % banks != 0u ||
        head_count / banks > 0xFFu)
        return FT_TT_EINVAL;
    for (uint32_t b = 0; b < banks; b++) {
        const ft_tt_op_operand *bank = &request->bank_region[b];

        if (bank->address != request->region.address || bank->x > 0xFFu || bank->y > 0xFFu ||
            bank->tile_bytes != request->region.tile_bytes)
            return FT_TT_EINVAL;
        packed_x |= (uint32_t)bank->x << (8u * b);
        packed_y |= (uint32_t)bank->y << (8u * b);
    }
    out_words[FT_TT_OP_MEMORY_FLAGS] |= (uint32_t)FT_TT_OP_MEMORY_FLAG_DST_BANKED |
                                        ((head_count / banks) << FT_TT_OP_MEMORY_BANK_HEADS_SHIFT);
    out_words[FT_TT_OP_MEMORY_DST_X] = packed_x;
    out_words[FT_TT_OP_MEMORY_DST_Y] = packed_y;
    return FT_TT_OK;
}

int ft_tt_op_kv_append_parameters(const ft_tt_op_kv_append_request *request,
                                  uint32_t out_words[FT_TT_OP_MEMORY_WORDS])
{
    const ft_tt_kv_cache_layout *cache;
    uint64_t source_base;
    uint64_t last_tile;
    uint32_t kv_format;

    if (!request || !out_words)
        return FT_TT_EINVAL;
    cache = request->cache;
    if (!layout_is_initialised(cache))
        return FT_TT_EINVAL;
    if (request->token_count == 0u || request->token_count > FT_TT_OP_MEMORY_MAX_ROWS)
        return FT_TT_EINVAL;
    if (!operand_is_set(&request->source) || !operand_is_set(&request->region))
        return FT_TT_EINVAL;
    /* A block-float cache is fed from the bf16 projection strip: the writer packs. */
    kv_format = ft_tt_kv_cache_format(cache);
    if (request->region.tile_bytes != cache->tile_bytes)
        return FT_TT_EINVAL;
    if (request->source.tile_bytes !=
        (kv_format == FT_TT_KV_FORMAT_BF16 ? cache->tile_bytes : FT_TT_OP_TILE_BYTES_BF16))
        return FT_TT_EINVAL;
    /* The split rule, on both ends. ft_tt_op_kv_append_split() is what a caller with a
     * longer run uses; refusing here is what stops a silent wrap in face arithmetic. */
    if ((request->source_row0 % FT_TT_OP_TILE_DIM) + request->token_count > FT_TT_OP_TILE_DIM)
        return FT_TT_EINVAL;
    if ((request->position % FT_TT_OP_TILE_DIM) + request->token_count > FT_TT_OP_TILE_DIM)
        return FT_TT_EINVAL;
    if ((uint64_t)request->position + request->token_count > cache->capacity_positions)
        return FT_TT_EINVAL;
    if (request->source_tiles < cache->head_count * cache->dim_tiles)
        return FT_TT_EINVAL;
    if ((request->source.address % MEMORY_DRAM_READ_ALIGNMENT) != 0u)
        return FT_TT_EINVAL;

    source_base = (uint64_t)request->source.address +
                  (uint64_t)(request->source_row0 / FT_TT_OP_TILE_DIM) * request->source_tiles *
                      request->source.tile_bytes;
    last_tile = (uint64_t)request->region.address + cache->region_bytes;
    if (source_base > UINT32_MAX || last_tile > UINT32_MAX || cache->head_stride > UINT32_MAX) {
        ft_tt_log_error("kv append: a 32-bit device address cannot hold this cache "
                        "(head_stride %llu, region end %llu)",
                        (unsigned long long)cache->head_stride, (unsigned long long)last_tile);
        return FT_TT_EINVAL;
    }

    memset(out_words, 0, FT_TT_OP_MEMORY_WORDS * sizeof(out_words[0]));
    out_words[FT_TT_OP_MEMORY_FLAGS] = (uint32_t)FT_TT_OP_MEMORY_FLAG_SRC_TILED |
                                       (kv_format << FT_TT_OP_MEMORY_KV_FORMAT_SHIFT);
    out_words[FT_TT_OP_MEMORY_ROW_COUNT] = request->token_count;
    out_words[FT_TT_OP_MEMORY_GROUP_COUNT] = cache->head_count;
    out_words[FT_TT_OP_MEMORY_GROUP_TILES] = cache->dim_tiles;
    out_words[FT_TT_OP_MEMORY_SRC_X] = request->source.x;
    out_words[FT_TT_OP_MEMORY_SRC_Y] = request->source.y;
    out_words[FT_TT_OP_MEMORY_SRC_BASE] = (uint32_t)source_base;
    out_words[FT_TT_OP_MEMORY_SRC_ROW0] = request->source_row0 % FT_TT_OP_TILE_DIM;
    out_words[FT_TT_OP_MEMORY_SRC_GROUP_STRIDE] = cache->dim_tiles * request->source.tile_bytes;
    out_words[FT_TT_OP_MEMORY_DST_X] = request->region.x;
    out_words[FT_TT_OP_MEMORY_DST_Y] = request->region.y;
    out_words[FT_TT_OP_MEMORY_DST_BASE] = request->region.address;
    out_words[FT_TT_OP_MEMORY_DST_GROUP_STRIDE] = (uint32_t)cache->head_stride;
    out_words[FT_TT_OP_MEMORY_DST_ROW0] = request->position;
    if (request->bank_count > 1u &&
        kv_append_encode_banks(request, cache->head_count, out_words) != FT_TT_OK)
        return FT_TT_EINVAL;
    /* DST_GROUP_STRIDE above is the head stride, which is exactly the bytes one head owns —
     * so the writer's bound refuses a position past `capacity_positions` and also stops head
     * h's append landing in head h+1. SRC_ROW_LIMIT stays 0: this source is TILED, so no
     * index ever becomes a row address. */
    return FT_TT_OK;
}

int ft_tt_op_kv_append_split(uint32_t position, uint32_t token_count, uint32_t source_row0,
                             uint32_t *out_positions, uint32_t *out_source_rows,
                             uint32_t *out_token_counts, uint32_t capacity, uint32_t *out_count)
{
    uint32_t remaining = token_count;
    uint32_t count = 0u;
    int rc = FT_TT_OK;

    if (!out_count || token_count == 0u)
        return FT_TT_EINVAL;
    if ((out_positions || out_source_rows || out_token_counts) &&
        !(out_positions && out_source_rows && out_token_counts))
        return FT_TT_EINVAL;

    while (remaining > 0u) {
        const uint32_t destination_room =
            FT_TT_OP_TILE_DIM - (position % FT_TT_OP_TILE_DIM);
        const uint32_t source_room = FT_TT_OP_TILE_DIM - (source_row0 % FT_TT_OP_TILE_DIM);
        uint32_t piece = (destination_room < source_room) ? destination_room : source_room;

        if (piece > remaining)
            piece = remaining;
        if (out_positions && count < capacity) {
            out_positions[count] = position;
            out_source_rows[count] = source_row0;
            out_token_counts[count] = piece;
        } else if (out_positions) {
            rc = FT_TT_ENOMEM;
        }
        count++;
        position += piece;
        source_row0 += piece;
        remaining -= piece;
    }
    *out_count = count;
    return rc;
}

/* ================================================================
 * Placement
 * ================================================================ */

int ft_tt_op_memory_place_cbs(uint32_t embedding_group_tiles, uint32_t kv_group_tiles,
                              uint32_t tile_bytes,
                              ft_tt_launch_cb_entry out_entries[FT_TT_OP_MEMORY_CB_COUNT],
                              void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    ft_tt_op_cb_request requests[FT_TT_OP_MEMORY_CB_COUNT];
    uint32_t widest_group_tiles;

    if (!out_entries || !buffer || !out_mask)
        return FT_TT_EINVAL;
    if (embedding_group_tiles == 0u || kv_group_tiles == 0u ||
        !tile_size_is_supported(tile_bytes))
        return FT_TT_EINVAL;

    widest_group_tiles = (embedding_group_tiles > kv_group_tiles) ? embedding_group_tiles
                                                                  : kv_group_tiles;
    /* CB_STAGE holds either one group's source tiles OR 32 source rows of one group, and
     * those are the SAME number of bytes — 32 rows of a group is exactly the group's tiles.
     * That is not a coincidence to be grateful for, it is why there is one staging size.
     * It is the one buffer both opcodes DO share, and sharing it is safe because nothing is
     * ever pushed or popped through it (the header's CB note): it is addressed, not cycled,
     * so it has no write pointer to leave off a grid. */
    requests[0].index = FT_TT_OP_MEMORY_CB_STAGE;
    requests[0].page_bytes = widest_group_tiles * tile_bytes;
    requests[0].page_count = 1u;
    requests[1].index = FT_TT_OP_MEMORY_CB_INDEX;
    requests[1].page_bytes = FT_TT_OP_MEMORY_INDEX_BYTES;
    requests[1].page_count = 1u;
    /* ONE GROUP SIZE PER CB. Two blocks of the op's OWN group makes each page count a whole
     * multiple of the chunk pushed through it, so the pointer is on that grid at every
     * descriptor boundary — the invariant ft_tensix_cb_push_grid_is_safe() tests, held by
     * construction rather than by a rule the host has to remember. */
    requests[2].index = FT_TT_OP_MEMORY_CB_OUT_EMBEDDING;
    requests[2].page_bytes = tile_bytes;
    requests[2].page_count = embedding_group_tiles * MEMORY_CB_OUT_BLOCKS;
    requests[3].index = FT_TT_OP_MEMORY_CB_OUT_KV;
    requests[3].page_bytes = tile_bytes;
    requests[3].page_count = kv_group_tiles * MEMORY_CB_OUT_BLOCKS;
    /* Writer-private, addressed: two packed tiles, so a block-float append reads tile t+1's
     * destination while it packs tile t. Placed for every cache format (4.3 KiB). */
    requests[4].index = FT_TT_OP_MEMORY_CB_KV_PACK;
    requests[4].page_bytes = FT_TT_OP_MEMORY_KV_PACK_BYTES;
    requests[4].page_count = 1u;

    return ft_tt_op_cb_place(requests, FT_TT_OP_MEMORY_CB_COUNT, out_entries, buffer,
                             buffer_bytes, out_mask);
}

uint32_t ft_tt_op_memory_index_address(const ft_tt_launch_cb_entry *entries, unsigned count)
{
    const ft_tt_launch_cb_entry *entry =
        ft_tt_op_cb_find(entries, count, FT_TT_OP_MEMORY_CB_INDEX);

    return entry ? entry->address : 0u;
}

const char *ft_tt_op_memory_opcode_name(uint32_t opcode)
{
    switch (opcode) {
    case FT_TT_QUEUE_OP_EMBEDDING:
        return "EMBEDDING";
    case FT_TT_QUEUE_OP_KV_APPEND:
        return "KV_APPEND";
    default:
        return "unknown";
    }
}
