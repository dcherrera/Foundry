/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_rope.c
 * @brief Implementation of the ROPE host driver. See ft_tt_op_rope.h, which carries the
 *   face-space argument, the table-placement decision and the row-phase contract this file
 *   assumes.
 *
 * Everything except ft_tt_op_rope_build_table() is PURE — no device access, no allocation.
 * The enqueue itself is ft_tt_queue_write_op_parameters() plus ft_tt_queue_enqueue(), which
 * the caller does, because a resident op has no launch of its own to own.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "ft_tt_op_rope.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "convert/ft_float_convert.h"
#include "ft_tt_device.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_tilize.h"

/** Pages per streamed input CB, as a multiple of the block depth. TWO blocks: the reader
 *  fills one while the compute thread consumes the other. */
#define ROPE_CB_BLOCKS 2u

/** Pages in the CBs produced and consumed a tile at a time. */
#define ROPE_CB_PAGES_SMALL 2u

/** Positions in one table row block — a tile is 32 rows and a row is a position. */
#define ROPE_POSITIONS_PER_BLOCK FT_TT_OP_TILE_DIM

uint32_t ft_tt_op_rope_rotary_width(uint32_t head_dim, uint32_t rope_dim)
{
    return (rope_dim > 0u && rope_dim <= head_dim) ? rope_dim : head_dim;
}

bool ft_tt_op_rope_shape_is_supported(uint32_t head_tiles, uint32_t rot_tiles)
{
    if (head_tiles == 0u || rot_tiles == 0u || rot_tiles > head_tiles)
        return false;
    /* rot_tiles even <=> half = rot/2 is a whole number of tiles <=> the partner element is
     * the same intra-tile column of a different tile. This one clause IS the face-space
     * argument, and the kernel's ftr_shape_is_supported() is the same three lines. */
    return (rot_tiles % 2u) == 0u;
}

/** An operand is usable when it names a tile size. The COORDINATE is not checked here: it
 *  comes from ft_tt_coord and a caller that invented one has a bigger problem than a range
 *  check (spec §7.3). */
static bool operand_is_set(const ft_tt_op_operand *operand)
{
    return operand->tile_bytes != 0u;
}

/** The tile counts a request implies, or false if the request is not expressible. */
static bool resolve_shape(const ft_tt_op_rope_request *request, uint32_t *out_head_tiles,
                          uint32_t *out_rot_tiles)
{
    uint32_t rot;

    if (request->row_blocks == 0u || request->heads == 0u || request->head_dim == 0u)
        return false;
    if ((request->head_dim % FT_TT_OP_TILE_DIM) != 0u)
        return false;
    rot = ft_tt_op_rope_rotary_width(request->head_dim, request->rope_dim);
    if ((rot % FT_TT_OP_TILE_DIM) != 0u)
        return false;

    *out_head_tiles = request->head_dim / FT_TT_OP_TILE_DIM;
    *out_rot_tiles = rot / FT_TT_OP_TILE_DIM;
    return ft_tt_op_rope_shape_is_supported(*out_head_tiles, *out_rot_tiles);
}

/** The depth the reader will use: it must divide BOTH runs it blocks over, because cb_in
 *  advances over head_tiles while the three rotated streams advance over rot_tiles, and a
 *  block that does not divide its run straddles the CB wrap. */
static uint32_t effective_depth(const ft_tt_op_rope_request *request, uint32_t head_tiles,
                                uint32_t rot_tiles)
{
    uint32_t depth = ft_tt_op_block_depth(head_tiles, request->prefetch_tiles);

    return ft_tt_op_block_depth(rot_tiles, depth);
}

int ft_tt_op_rope_parameters(const ft_tt_op_rope_request *request,
                             uint32_t out_words[FT_TT_OP_ROPE_WORDS])
{
    uint32_t head_tiles = 0u;
    uint32_t rot_tiles = 0u;

    if (!request || !out_words)
        return FT_TT_EINVAL;
    if (!resolve_shape(request, &head_tiles, &rot_tiles))
        return FT_TT_EINVAL;
    if (!operand_is_set(&request->input) || !operand_is_set(&request->table) ||
        !operand_is_set(&request->output))
        return FT_TT_EINVAL;
    /* THE ROW-PHASE CONTRACT. A caller with a decode token at position p passes p and puts
     * the token in tile row p % 32; the row block is derived below. A first_position that is
     * not a multiple of 32 would mean row 0 of the activation is not row 0 of a table block,
     * and there is no field that could express the difference — so it is a refusal rather
     * than a rounding. */
    if ((request->first_position % ROPE_POSITIONS_PER_BLOCK) != 0u &&
        request->row_blocks != 1u)
        return FT_TT_EINVAL;

    memset(out_words, 0, FT_TT_OP_ROPE_WORDS * sizeof(out_words[0]));
    out_words[FT_TT_OP_ROPE_TILE_COUNT] = request->row_blocks * request->heads * head_tiles;
    out_words[FT_TT_OP_ROPE_HEAD_TILES] = head_tiles;
    out_words[FT_TT_OP_ROPE_ROT_TILES] = rot_tiles;
    out_words[FT_TT_OP_ROPE_HEADS] = request->heads;
    out_words[FT_TT_OP_ROPE_DEPTH] = effective_depth(request, head_tiles, rot_tiles);
    out_words[FT_TT_OP_ROPE_A_X] = request->input.x;
    out_words[FT_TT_OP_ROPE_A_Y] = request->input.y;
    out_words[FT_TT_OP_ROPE_A_BASE] = request->input.address;
    out_words[FT_TT_OP_ROPE_TABLE_X] = request->table.x;
    out_words[FT_TT_OP_ROPE_TABLE_Y] = request->table.y;
    out_words[FT_TT_OP_ROPE_TABLE_BASE] =
        request->table.address + ft_tt_op_rope_table_offset(request->first_position, rot_tiles,
                                                            request->table.tile_bytes);
    out_words[FT_TT_OP_ROPE_C_X] = request->output.x;
    out_words[FT_TT_OP_ROPE_C_Y] = request->output.y;
    out_words[FT_TT_OP_ROPE_C_BASE] = request->output.address;
    return FT_TT_OK;
}

int ft_tt_op_rope_place_cbs(const ft_tt_op_rope_request *request,
                            ft_tt_launch_cb_entry out_entries[FT_TT_OP_ROPE_CB_COUNT],
                            void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    ft_tt_op_cb_request requests[FT_TT_OP_ROPE_CB_COUNT];
    uint32_t head_tiles = 0u;
    uint32_t rot_tiles = 0u;
    uint32_t depth;

    if (!request || !out_entries || !buffer || !out_mask)
        return FT_TT_EINVAL;
    if (!resolve_shape(request, &head_tiles, &rot_tiles))
        return FT_TT_EINVAL;
    if (!operand_is_set(&request->input) || !operand_is_set(&request->table) ||
        !operand_is_set(&request->output))
        return FT_TT_EINVAL;

    depth = effective_depth(request, head_tiles, rot_tiles);

    requests[0].index = FT_TT_OP_ROPE_CB_IN;
    requests[0].page_bytes = request->input.tile_bytes;
    requests[0].page_count = depth * ROPE_CB_BLOCKS;
    requests[1].index = FT_TT_OP_ROPE_CB_ROT;
    requests[1].page_bytes = request->input.tile_bytes;
    requests[1].page_count = depth * ROPE_CB_BLOCKS;
    requests[2].index = FT_TT_OP_ROPE_CB_COS;
    requests[2].page_bytes = request->table.tile_bytes;
    requests[2].page_count = depth * ROPE_CB_BLOCKS;
    requests[3].index = FT_TT_OP_ROPE_CB_SIN;
    requests[3].page_bytes = request->table.tile_bytes;
    requests[3].page_count = depth * ROPE_CB_BLOCKS;
    requests[4].index = FT_TT_OP_ROPE_CB_PROD_COS;
    requests[4].page_bytes = request->output.tile_bytes;
    requests[4].page_count = ROPE_CB_PAGES_SMALL;
    requests[5].index = FT_TT_OP_ROPE_CB_PROD_SIN;
    requests[5].page_bytes = request->output.tile_bytes;
    requests[5].page_count = ROPE_CB_PAGES_SMALL;
    requests[6].index = FT_TT_OP_ROPE_CB_OUT;
    requests[6].page_bytes = request->output.tile_bytes;
    requests[6].page_count = ROPE_CB_PAGES_SMALL;

    return ft_tt_op_cb_place(requests, FT_TT_OP_ROPE_CB_COUNT, out_entries, buffer,
                             buffer_bytes, out_mask);
}

/* ================================================================
 * The table
 * ================================================================ */

uint32_t ft_tt_op_rope_table_row_block_bytes(uint32_t rot_tiles, uint32_t tile_bytes)
{
    return 2u * rot_tiles * tile_bytes;
}

size_t ft_tt_op_rope_table_bytes(uint32_t position_count, uint32_t rot_tiles,
                                 uint32_t tile_bytes)
{
    const size_t blocks =
        ((size_t)position_count + ROPE_POSITIONS_PER_BLOCK - 1u) / ROPE_POSITIONS_PER_BLOCK;

    return blocks * ft_tt_op_rope_table_row_block_bytes(rot_tiles, tile_bytes);
}

uint32_t ft_tt_op_rope_table_offset(uint32_t position, uint32_t rot_tiles,
                                    uint32_t tile_bytes)
{
    return (position / ROPE_POSITIONS_PER_BLOCK) *
           ft_tt_op_rope_table_row_block_bytes(rot_tiles, tile_bytes);
}

double ft_tt_op_rope_angle(uint32_t position, uint32_t j, uint32_t rot, double theta_base)
{
    const uint32_t half = rot / 2u;
    double frequency;

    if (half == 0u)
        return 0.0;
    /* pow() in double, and no explicit range reduction: libm's cos/sin reduce correctly for
     * any |x| < 2^63 and the largest angle here is 131071 radians. That exactness is the
     * whole reason this is not an SFPU pass (ft_tt_op_rope.h, THE TABLE). */
    frequency = pow(theta_base, -2.0 * (double)(j % half) / (double)rot);
    return (double)position * frequency;
}

/** One row block: 32 positions x `rot` columns of cos, then the same of SIGNED sin. */
static void fill_table_block(uint16_t *cos_plain, uint16_t *sin_plain, uint32_t position,
                             uint32_t rot, double theta_base)
{
    const uint32_t half = rot / 2u;
    uint32_t row;

    for (row = 0; row < ROPE_POSITIONS_PER_BLOCK; row++) {
        uint32_t column;

        for (column = 0; column < rot; column++) {
            const double angle = ft_tt_op_rope_angle(position + row, column, rot, theta_base);
            const double sine = sin(angle);
            /* THE SIGN LIVES HERE. Below the half the output is x*cos - partner*sin, above it
             * x*cos + partner*sin, so a signed table makes both the same instruction. */
            const double signed_sine = (column < half) ? -sine : sine;

            cos_plain[row * rot + column] = ft_bf16_from_float((float)cos(angle));
            sin_plain[row * rot + column] = ft_bf16_from_float((float)signed_sine);
        }
    }
}

int ft_tt_op_rope_build_table(uint32_t first_position, uint32_t position_count,
                              uint32_t head_dim, uint32_t rope_dim, double theta_base,
                              void *out, size_t out_bytes)
{
    unsigned char *cursor = out;
    uint16_t *cos_plain = NULL;
    uint16_t *sin_plain = NULL;
    uint32_t rot;
    uint32_t rot_tiles;
    uint32_t blocks;
    uint32_t block;
    size_t strip_bytes;
    int rc = FT_TT_EINVAL;

    if (!out || position_count == 0u || head_dim == 0u)
        return FT_TT_EINVAL;
    if ((first_position % ROPE_POSITIONS_PER_BLOCK) != 0u ||
        (position_count % ROPE_POSITIONS_PER_BLOCK) != 0u)
        return FT_TT_EINVAL;
    if ((head_dim % FT_TT_OP_TILE_DIM) != 0u)
        return FT_TT_EINVAL;
    rot = ft_tt_op_rope_rotary_width(head_dim, rope_dim);
    if ((rot % FT_TT_OP_TILE_DIM) != 0u)
        return FT_TT_EINVAL;
    rot_tiles = rot / FT_TT_OP_TILE_DIM;
    if (!ft_tt_op_rope_shape_is_supported(head_dim / FT_TT_OP_TILE_DIM, rot_tiles))
        return FT_TT_EINVAL;
    if (out_bytes <
        ft_tt_op_rope_table_bytes(position_count, rot_tiles, FT_TT_OP_TILE_BYTES_BF16))
        return FT_TT_EINVAL;

    cos_plain = malloc((size_t)ROPE_POSITIONS_PER_BLOCK * rot * sizeof(*cos_plain));
    sin_plain = malloc((size_t)ROPE_POSITIONS_PER_BLOCK * rot * sizeof(*sin_plain));
    if (!cos_plain || !sin_plain) {
        rc = FT_TT_ENOMEM;
        goto cleanup;
    }

    blocks = position_count / ROPE_POSITIONS_PER_BLOCK;
    strip_bytes = (size_t)rot_tiles * FT_TT_OP_TILE_BYTES_BF16;

    for (block = 0; block < blocks; block++) {
        const uint32_t position = first_position + block * ROPE_POSITIONS_PER_BLOCK;

        fill_table_block(cos_plain, sin_plain, position, rot, theta_base);
        /* ft_tt_tilize_strip(), i.e. the SAME face map the activations go through. A table
         * built flat row-major would hand three quarters of every tile the angle of a
         * different (position, frequency) pair — a wrong answer with nothing to cancel it. */
        if (ft_tt_tilize_strip(cos_plain, sizeof(*cos_plain), ROPE_POSITIONS_PER_BLOCK, rot,
                               (size_t)rot * sizeof(*cos_plain), cursor) != 0)
            goto cleanup;
        if (ft_tt_tilize_strip(sin_plain, sizeof(*sin_plain), ROPE_POSITIONS_PER_BLOCK, rot,
                               (size_t)rot * sizeof(*sin_plain), cursor + strip_bytes) != 0)
            goto cleanup;
        cursor += 2u * strip_bytes;
    }
    rc = FT_TT_OK;

cleanup:
    free(cos_plain);
    free(sin_plain);
    return rc;
}
