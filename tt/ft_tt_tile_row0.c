/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_tile_row0.c
 * @brief Implementation of the row-0-only tile transfers (see ft_tt_tile_row0.h).
 *
 * @status ON THE CARD — see ft_tt_tile_row0.h for the numbers and the mutation table.
 * @issues None
 * @todo None
 */

#include "ft_tt_tile_row0.h"

#include <string.h>

uint32_t ft_tt_tile_row0_chunk_offset(uint32_t tile_index, uint32_t chunk)
{
    /* Chunk 0 is columns 0-15, which live at the START of face 0; chunk 1 is columns
     * 16-31, which live at the start of face 1, one whole face further in. Both facts
     * come from ft_tt_tilize.h's face ordering; a stride of 256 here (half a face, which
     * is what the FACE-INDEX arithmetic uses in ELEMENTS) reads the middle of face 0 and
     * returns a plausible wrong answer with no error anywhere. */
    return tile_index * FT_TT_TILE_ROW0_TILE_BYTES + chunk * FT_TT_TILE_ROW0_FACE_BYTES;
}

int ft_tt_tile_row0_read(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                         uint32_t base_address, uint32_t tile_count, uint16_t *out_row)
{
    /* One 64 B chunk at a time: the DRAM read alignment, of which the leading 32 B are
     * row 0. Reading the two chunks into one 128 B buffer would not help — they are
     * 512 B apart, so they are two transactions either way. */
    unsigned char chunk_bytes[FT_TT_TILE_ROW0_READ_BYTES];
    uint32_t tile;

    if (!noc || !noc->read || !out_row)
        return FT_TT_EINVAL;

    for (tile = 0; tile < tile_count; tile++) {
        uint32_t chunk;

        for (chunk = 0; chunk < FT_TT_TILE_ROW0_CHUNKS; chunk++) {
            uint32_t offset = ft_tt_tile_row0_chunk_offset(tile, chunk);
            int rc = noc->read(noc->user, x, y, (uint64_t)base_address + offset, chunk_bytes,
                               sizeof(chunk_bytes));

            if (rc != FT_TT_OK)
                return rc;
            memcpy(out_row + (size_t)tile * (FT_TT_TILE_ROW0_CHUNKS * FT_TT_TILE_ROW0_HALF_ELEMS) +
                       (size_t)chunk * FT_TT_TILE_ROW0_HALF_ELEMS,
                   chunk_bytes, FT_TT_TILE_ROW0_HALF_BYTES);
        }
    }
    return FT_TT_OK;
}

int ft_tt_tile_row0_write(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                          uint32_t base_address, uint32_t tile_count, const uint16_t *row)
{
    uint32_t tile;

    if (!noc || !noc->write || !row)
        return FT_TT_EINVAL;

    for (tile = 0; tile < tile_count; tile++) {
        uint32_t chunk;

        for (chunk = 0; chunk < FT_TT_TILE_ROW0_CHUNKS; chunk++) {
            uint32_t offset = ft_tt_tile_row0_chunk_offset(tile, chunk);
            const uint16_t *source =
                row + (size_t)tile * (FT_TT_TILE_ROW0_CHUNKS * FT_TT_TILE_ROW0_HALF_ELEMS) +
                (size_t)chunk * FT_TT_TILE_ROW0_HALF_ELEMS;
            /* 32 B, not 64: the DRAM WRITE alignment is 16 B, so this direction moves
             * exactly row 0 and nothing else. */
            int rc = noc->write(noc->user, x, y, (uint64_t)base_address + offset, source,
                                FT_TT_TILE_ROW0_HALF_BYTES);

            if (rc != FT_TT_OK)
                return rc;
        }
    }
    return FT_TT_OK;
}
