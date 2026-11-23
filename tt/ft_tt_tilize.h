/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_tilize.h
 * @brief Host-side conversion between row-major matrices and the 32x32 tile layout a
 *   Tensix core consumes: four 16x16 faces per tile, tiles row-major over the grid.
 *
 * Layout (element indices inside one tile, `ft_tt_tile_face_index`):
 *
 *     face 0 = rows  0-15, cols  0-15   -> elements    0..255
 *     face 1 = rows  0-15, cols 16-31   -> elements  256..511
 *     face 2 = rows 16-31, cols  0-15   -> elements  512..767
 *     face 3 = rows 16-31, cols 16-31   -> elements  768..1023
 *
 * Inside a face the elements are row-major with a row length of 16. The 16x16 face is
 * the operand granule of the matrix unit (tt-isa-documentation, WormholeB0
 * TensixCoprocessor/MVMUL.md and SrcASrcB.md: "16x16 matrix" operands), which is why a
 * tile is stored as faces rather than as one 32-wide row-major block.
 *
 * The "swizzled" variants produce the same tile grid and the same zero padding but leave
 * each tile row-major (row length 32, no face reordering).
 *
 * Supported element sizes are 1, 2 and 4 bytes. Everything here is pure arithmetic and
 * memcpy; no device is touched.
 *
 * @status Clean-room rewrite. Oracle: tests/test_tilize.c (`make check-tilize`).
 * @issues None
 * @todo None
 */

#ifndef FT_TT_TILIZE_H
#define FT_TT_TILIZE_H

#include <stddef.h>
#include <stdint.h>

/** Tile and face geometry, in elements. */
enum {
    FT_TT_TILE_HEIGHT = 32,
    FT_TT_TILE_WIDTH = 32,
    FT_TT_FACE_HEIGHT = 16,
    FT_TT_FACE_WIDTH = 16,
    FT_TT_TILE_VALUES = FT_TT_TILE_HEIGHT * FT_TT_TILE_WIDTH,
    FT_TT_FACE_VALUES = FT_TT_FACE_HEIGHT * FT_TT_FACE_WIDTH,
    FT_TT_FACES_PER_TILE = FT_TT_TILE_VALUES / FT_TT_FACE_VALUES
};

/** Return codes. EINVAL shares its value with FT_TT_EINVAL (ft_tt_device.h). */
enum {
    FT_TT_TILIZE_OK = 0,
    FT_TT_TILIZE_EINVAL = -3
};

/**
 * Element index of (row, col) inside one face-ordered tile.
 *
 * @param row  0..31
 * @param col  0..31
 * @return     0..1023 (inputs outside the tile are reduced modulo 32)
 */
uint32_t ft_tt_tile_face_index(uint32_t row, uint32_t col);

/**
 * Number of 32-wide tiles needed to cover `count` rows or columns (rounds up).
 */
uint32_t ft_tt_tile_count_for(uint32_t count);

/**
 * Tilize one row strip (up to 32 rows) into face-ordered tiles.
 *
 * Writes ft_tt_tile_count_for(cols) consecutive tiles, each FT_TT_TILE_VALUES *
 * element_bytes bytes. Rows past `valid_rows` and columns past `cols` are zero.
 *
 * @param row_major       First element of the strip's first row.
 * @param element_bytes   1, 2 or 4.
 * @param valid_rows      Real rows in the strip, 1..32.
 * @param cols            Real columns, >= 1.
 * @param src_row_stride  Bytes between consecutive source rows, >= cols * element_bytes.
 * @param tiles_out       Destination for the strip's tiles.
 * @return FT_TT_TILIZE_OK, or FT_TT_TILIZE_EINVAL on a NULL pointer or bad geometry.
 */
int ft_tt_tilize_strip(const void *row_major, size_t element_bytes, uint32_t valid_rows, uint32_t cols,
                       size_t src_row_stride, void *tiles_out);

/**
 * As ft_tt_tilize_strip(), but each tile is left row-major (32 elements per row, no face
 * reordering). Same tile grid, same zero padding, same validation.
 */
int ft_tt_tile_swizzle_strip(const void *row_major, size_t element_bytes, uint32_t valid_rows, uint32_t cols,
                             size_t src_row_stride, void *tiles_out);

/**
 * Inverse of ft_tt_tile_swizzle_strip(): copy the valid `valid_rows` x `cols` region out
 * of a strip of row-major tiles. Padding in the tiles is ignored; nothing outside the
 * valid region of the destination is written.
 */
int ft_tt_tile_unswizzle_strip(const void *tiles, size_t element_bytes, uint32_t valid_rows, uint32_t cols,
                               size_t dst_row_stride, void *row_major_out);

/**
 * Inverse of ft_tt_tilize_strip(): copy the valid `valid_rows` x `cols` region out of a
 * strip of face-ordered tiles into row-major memory with `dst_row_stride` bytes per row.
 */
int ft_tt_untilize_strip(const void *tiles, size_t element_bytes, uint32_t valid_rows, uint32_t cols,
                         size_t dst_row_stride, void *row_major_out);

/**
 * Tilize a whole dense row-major matrix (row stride = cols * element_bytes). The tile
 * strips are written one after another, so tiles are row-major over the tile grid.
 * `tiles_out` must hold ft_tt_tilize_bytes(element_bytes, rows, cols) bytes.
 *
 * @return FT_TT_TILIZE_OK, or FT_TT_TILIZE_EINVAL (NULL, zero rows/cols, bad element size).
 */
int ft_tt_tilize_matrix(const void *row_major, size_t element_bytes, uint32_t rows, uint32_t cols, void *tiles_out);

/**
 * Inverse of ft_tt_tilize_matrix(): writes the dense rows x cols matrix only.
 */
int ft_tt_untilize_matrix(const void *tiles, size_t element_bytes, uint32_t rows, uint32_t cols, void *row_major_out);

/**
 * Bytes a rows x cols matrix occupies once tilized (whole tiles, padding included).
 *
 * @return The size, or 0 for an unsupported element size.
 */
size_t ft_tt_tilize_bytes(size_t element_bytes, uint32_t rows, uint32_t cols);

#endif
