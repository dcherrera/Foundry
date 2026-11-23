/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_tilize.c
 * @brief Implementation of the row-major <-> tile layout conversions (ft_tt_tilize.h).
 *
 * Every conversion is the same walk: for each tile of a strip, for each valid row, copy
 * runs of contiguous elements between the row-major side and the tile side. A "run" is
 * the widest span that is contiguous on BOTH sides: one face row (16 elements) in the
 * face-ordered layout, one tile row (32 elements) in the swizzled layout. The two layouts
 * differ only in the run width and in where a (row, col) lands inside the tile.
 *
 * @status Clean-room rewrite. Green against tests/test_tilize.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_tilize.h"

#include <string.h>

/** Which in-tile ordering a conversion uses. */
typedef enum tile_element_order {
    TILE_ORDER_FACES,    /* four 16x16 faces, each row-major */
    TILE_ORDER_ROW_MAJOR /* one 32x32 row-major block (the "swizzled" form) */
} tile_element_order;

/** Which way bytes move. */
typedef enum tile_copy_direction {
    COPY_INTO_TILES,
    COPY_OUT_OF_TILES
} tile_copy_direction;

uint32_t ft_tt_tile_face_index(uint32_t row, uint32_t col)
{
    uint32_t tile_row = row % FT_TT_TILE_HEIGHT;
    uint32_t tile_col = col % FT_TT_TILE_WIDTH;
    uint32_t face_row_of_tile = tile_row / FT_TT_FACE_HEIGHT; /* 0 = top faces, 1 = bottom */
    uint32_t face_col_of_tile = tile_col / FT_TT_FACE_WIDTH;  /* 0 = left faces, 1 = right */
    uint32_t face_number = face_row_of_tile * 2u + face_col_of_tile;

    return face_number * FT_TT_FACE_VALUES + (tile_row % FT_TT_FACE_HEIGHT) * FT_TT_FACE_WIDTH +
           (tile_col % FT_TT_FACE_WIDTH);
}

uint32_t ft_tt_tile_count_for(uint32_t count)
{
    /* Written without `count + 31` so a count near UINT32_MAX cannot wrap. */
    return count / FT_TT_TILE_WIDTH + (count % FT_TT_TILE_WIDTH != 0u ? 1u : 0u);
}

static int element_size_is_supported(size_t element_bytes)
{
    return element_bytes == 1u || element_bytes == 2u || element_bytes == 4u;
}

size_t ft_tt_tilize_bytes(size_t element_bytes, uint32_t rows, uint32_t cols)
{
    if (!element_size_is_supported(element_bytes))
        return 0u;
    return (size_t)ft_tt_tile_count_for(rows) * ft_tt_tile_count_for(cols) * FT_TT_TILE_VALUES *
           element_bytes;
}

/** The argument contract shared by all four strip functions. */
static int strip_arguments_are_valid(const void *source, const void *destination, size_t element_bytes,
                                     uint32_t valid_rows, uint32_t cols, size_t row_stride)
{
    if (!source || !destination || !element_size_is_supported(element_bytes))
        return 0;
    if (valid_rows == 0u || valid_rows > FT_TT_TILE_HEIGHT || cols == 0u)
        return 0;
    return row_stride >= (size_t)cols * element_bytes;
}

static uint32_t in_tile_index(tile_element_order order, uint32_t row, uint32_t col)
{
    if (order == TILE_ORDER_FACES)
        return ft_tt_tile_face_index(row, col);
    return row * FT_TT_TILE_WIDTH + col;
}

/**
 * Move one strip between row-major memory and tiles, in either direction.
 *
 * Arguments must already be validated. When copying into tiles, the whole strip of tiles
 * is zeroed first so every padding element is 0; when copying out, only the valid region
 * of the row-major side is written.
 */
static void copy_strip(tile_element_order order, tile_copy_direction direction, unsigned char *row_major,
                       unsigned char *tiles, size_t element_bytes, uint32_t valid_rows, uint32_t cols,
                       size_t row_stride)
{
    const uint32_t run_width = (order == TILE_ORDER_FACES) ? FT_TT_FACE_WIDTH : FT_TT_TILE_WIDTH;
    const uint32_t tile_count = ft_tt_tile_count_for(cols);
    const size_t tile_bytes = (size_t)FT_TT_TILE_VALUES * element_bytes;
    uint32_t tile, row, run_start;

    if (direction == COPY_INTO_TILES)
        memset(tiles, 0, (size_t)tile_count * tile_bytes);

    for (tile = 0; tile < tile_count; tile++) {
        unsigned char *tile_base = tiles + (size_t)tile * tile_bytes;
        uint32_t first_col = tile * FT_TT_TILE_WIDTH;

        for (row = 0; row < valid_rows; row++) {
            unsigned char *row_base = row_major + (size_t)row * row_stride;

            for (run_start = 0; run_start < FT_TT_TILE_WIDTH; run_start += run_width) {
                uint32_t matrix_col = first_col + run_start;
                uint32_t run_length;
                unsigned char *tile_side;
                unsigned char *matrix_side;

                if (matrix_col >= cols)
                    break;
                run_length = cols - matrix_col < run_width ? cols - matrix_col : run_width;
                tile_side = tile_base + (size_t)in_tile_index(order, row, run_start) * element_bytes;
                matrix_side = row_base + (size_t)matrix_col * element_bytes;
                if (direction == COPY_INTO_TILES)
                    memcpy(tile_side, matrix_side, (size_t)run_length * element_bytes);
                else
                    memcpy(matrix_side, tile_side, (size_t)run_length * element_bytes);
            }
        }
    }
}

/*
 * The const casts below are confined to the direction that only READS through the
 * pointer: copy_strip writes to `tiles` only when copying into tiles and to `row_major`
 * only when copying out.
 */

int ft_tt_tilize_strip(const void *row_major, size_t element_bytes, uint32_t valid_rows, uint32_t cols,
                       size_t src_row_stride, void *tiles_out)
{
    if (!strip_arguments_are_valid(row_major, tiles_out, element_bytes, valid_rows, cols, src_row_stride))
        return FT_TT_TILIZE_EINVAL;
    copy_strip(TILE_ORDER_FACES, COPY_INTO_TILES, (unsigned char *)(uintptr_t)row_major, tiles_out,
               element_bytes, valid_rows, cols, src_row_stride);
    return FT_TT_TILIZE_OK;
}

int ft_tt_tile_swizzle_strip(const void *row_major, size_t element_bytes, uint32_t valid_rows, uint32_t cols,
                             size_t src_row_stride, void *tiles_out)
{
    if (!strip_arguments_are_valid(row_major, tiles_out, element_bytes, valid_rows, cols, src_row_stride))
        return FT_TT_TILIZE_EINVAL;
    copy_strip(TILE_ORDER_ROW_MAJOR, COPY_INTO_TILES, (unsigned char *)(uintptr_t)row_major, tiles_out,
               element_bytes, valid_rows, cols, src_row_stride);
    return FT_TT_TILIZE_OK;
}

int ft_tt_tile_unswizzle_strip(const void *tiles, size_t element_bytes, uint32_t valid_rows, uint32_t cols,
                               size_t dst_row_stride, void *row_major_out)
{
    if (!strip_arguments_are_valid(tiles, row_major_out, element_bytes, valid_rows, cols, dst_row_stride))
        return FT_TT_TILIZE_EINVAL;
    copy_strip(TILE_ORDER_ROW_MAJOR, COPY_OUT_OF_TILES, row_major_out, (unsigned char *)(uintptr_t)tiles,
               element_bytes, valid_rows, cols, dst_row_stride);
    return FT_TT_TILIZE_OK;
}

int ft_tt_untilize_strip(const void *tiles, size_t element_bytes, uint32_t valid_rows, uint32_t cols,
                         size_t dst_row_stride, void *row_major_out)
{
    if (!strip_arguments_are_valid(tiles, row_major_out, element_bytes, valid_rows, cols, dst_row_stride))
        return FT_TT_TILIZE_EINVAL;
    copy_strip(TILE_ORDER_FACES, COPY_OUT_OF_TILES, row_major_out, (unsigned char *)(uintptr_t)tiles,
               element_bytes, valid_rows, cols, dst_row_stride);
    return FT_TT_TILIZE_OK;
}

/** Walk a dense matrix one 32-row strip at a time, in either direction. */
static int convert_matrix(tile_copy_direction direction, unsigned char *row_major, unsigned char *tiles,
                          size_t element_bytes, uint32_t rows, uint32_t cols)
{
    const size_t row_stride = (size_t)cols * element_bytes;
    const size_t strip_tile_bytes = (size_t)ft_tt_tile_count_for(cols) * FT_TT_TILE_VALUES * element_bytes;
    uint32_t strip_count, strip;

    if (!row_major || !tiles || !element_size_is_supported(element_bytes) || rows == 0u || cols == 0u)
        return FT_TT_TILIZE_EINVAL;

    strip_count = ft_tt_tile_count_for(rows);
    for (strip = 0; strip < strip_count; strip++) {
        uint32_t first_row = strip * FT_TT_TILE_HEIGHT;
        uint32_t rows_left = rows - first_row;
        uint32_t valid_rows = rows_left < FT_TT_TILE_HEIGHT ? rows_left : FT_TT_TILE_HEIGHT;

        copy_strip(TILE_ORDER_FACES, direction, row_major + (size_t)first_row * row_stride,
                   tiles + (size_t)strip * strip_tile_bytes, element_bytes, valid_rows, cols, row_stride);
    }
    return FT_TT_TILIZE_OK;
}

int ft_tt_tilize_matrix(const void *row_major, size_t element_bytes, uint32_t rows, uint32_t cols, void *tiles_out)
{
    return convert_matrix(COPY_INTO_TILES, (unsigned char *)(uintptr_t)row_major, tiles_out, element_bytes, rows,
                          cols);
}

int ft_tt_untilize_matrix(const void *tiles, size_t element_bytes, uint32_t rows, uint32_t cols, void *row_major_out)
{
    return convert_matrix(COPY_OUT_OF_TILES, row_major_out, (unsigned char *)(uintptr_t)tiles, element_bytes, rows,
                          cols);
}
