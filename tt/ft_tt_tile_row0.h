/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_tile_row0.h
 * @brief Move ONLY logical row 0 of a strip of 32x32 bf16 tiles between host memory and
 *   a DRAM bank — the shape a matvec's activation and result actually occupy.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS
 * ---------------------------------------------------------------------------
 * A matvec is Mt=1: one row of a 32-row tile carries the answer and the other 31 are
 * zero padding the FPU computed anyway. That waste is free ON the card — it is the tile
 * quantum — but it is NOT free across the link, where a token's 15,216 output tiles are
 * 29.719 MiB moved against 0.929 MiB needed, exactly 32.00x, on a bus that does ~4.7 MB/s
 * card->host with no device->host DMA.
 *
 * ---------------------------------------------------------------------------
 * THE RECOVERABLE FACTOR IS 16, NOT 32, AND THE REASON IS THE READ ALIGNMENT
 * ---------------------------------------------------------------------------
 * A tile is four 16x16 faces, each row-major, faces in row-major order of the 2x2 face
 * grid (ft_tt_tilize.h). So logical row 0 is NOT contiguous: cols 0-15 are elements
 * 0-15 of face 0 and cols 16-31 are elements 256-271, i.e. bf16 bytes `[0,32)` and
 * `[512,544)` — two 32-byte pieces 512 bytes apart.
 *
 * **A 32-byte DRAM READ IS REFUSED.** Blackhole's card->host alignment is 64 B
 * (ft_tt_noc_alignment(), and confirmed on silicon), so the smallest legal row-0 read is
 * 2 x 64 B = **128 B of 2048 = 16.00x**, not 32x. Half of each 64 B chunk is discarded
 * and that is the correct trade: per-call cost is flat from 64 B to 256 KiB on this link,
 * so the cost is the bytes and 128 beats 2048.
 *
 * The WRITE direction takes the looser 16 B DRAM write alignment, so it moves exactly the
 * 2 x 32 B it needs — 64 B of 2048 = 32.00x, the full factor.
 *
 * ---------------------------------------------------------------------------
 * THE WRITE HAS A PRECONDITION THE CALLER OWNS
 * ---------------------------------------------------------------------------
 * ft_tt_tile_row0_write() writes ONLY row 0. Rows 1-31 keep whatever the bank already
 * held, and for a matvec they must be ZERO — the reader feeds all 32 rows to the FPU.
 * A caller must therefore stage the strip zeroed ONCE (a full-size write of zeros at
 * setup) before using this per token. Staging is a setup cost; the per-token cost is the
 * 64 bytes. Skipping the staging does not fail, it returns arithmetic computed against
 * whatever was in DRAM — so the requirement is stated here and asserted nowhere, because
 * this layer cannot see it.
 *
 * ---------------------------------------------------------------------------
 * WHY IT TAKES AN ACCESSOR AND NOT A PATH
 * ---------------------------------------------------------------------------
 * Everything here is offset arithmetic over ft_tt_noc_access, so the whole module runs
 * against a simulated bank on any machine — which is what makes the wrong-face-stride
 * and swapped-chunk mutations provable off-hardware. The accessor MUST be a bulk DRAM
 * one (BLOCK16 over ft_tt_{read,write}_dram); handing it the WORD32 register accessor
 * would work and would defeat the entire point.
 *
 * @status ON THE CARD. 92 off-hardware checks green with three source mutations red
 *   (face stride 256, chunks swapped, tile stride 1024 — tests/run_tile_row0_red.sh), and
 *   two on-card mutations that turn the argmax probe from MATCH to MISMATCH. In the live
 *   forward it takes readback 6.58 s -> 0.385 s and upload 0.077 s -> 0.003 s per token,
 *   with all 338 per-op PCC values bit-identical to the full-tile path.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_TILE_ROW0_H
#define FT_TT_TILE_ROW0_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_launch.h"

/** Bytes of one 16x16 bf16 face, i.e. the distance from row 0's left half to its right
 *  half. This is the constant a wrong value silently corrupts the answer with. */
#define FT_TT_TILE_ROW0_FACE_BYTES 512u

/** The two pieces logical row 0 is split across: face 0 and face 1. */
#define FT_TT_TILE_ROW0_CHUNKS 2u

/** bf16 elements in one of those pieces. */
#define FT_TT_TILE_ROW0_HALF_ELEMS 16u

/** Bytes of one piece — what the WRITE direction moves per chunk (16 B aligned). */
#define FT_TT_TILE_ROW0_HALF_BYTES 32u

/** Bytes the READ direction must move per chunk: the 64 B DRAM read alignment, of which
 *  the first FT_TT_TILE_ROW0_HALF_BYTES are row 0 and the rest is discarded. */
#define FT_TT_TILE_ROW0_READ_BYTES 64u

/** Bytes of one whole 32x32 bf16 tile — what NOT reading row 0 alone would cost. */
#define FT_TT_TILE_ROW0_TILE_BYTES 2048u

/**
 * Byte offset, within a strip of consecutive bf16 tiles, of one of row 0's two pieces.
 *
 * The one place the layout is expressed. Both mutations that matter — a wrong face
 * stride and the two chunks swapped — are wrong values out of this function, which is
 * why it is separate and pure rather than inlined into the loops below.
 *
 * @param tile_index  which tile in the strip
 * @param chunk       0 for columns 0-15 (face 0), 1 for columns 16-31 (face 1)
 * @return            byte offset from the strip's base address
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_tile_row0_chunk_offset(uint32_t tile_index, uint32_t chunk);

/**
 * Read logical row 0 out of `tile_count` consecutive bf16 tiles.
 *
 * @param noc            a BULK DRAM accessor (BLOCK16); see the header note
 * @param x, y           TRANSLATED coordinate of the DRAM bank port. NEVER a guess
 * @param base_address   byte offset of tile 0 within the bank, 64 B aligned
 * @param tile_count     tiles in the strip
 * @param out_row        receives `tile_count * 32` bf16 elements
 * @return               FT_TT_OK, or FT_TT_EINVAL, or the accessor's failure
 *
 * Thread-safe: no (the accessor is stateful)    Reentrant: no    Signal-safe: no
 */
int ft_tt_tile_row0_read(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                         uint32_t base_address, uint32_t tile_count, uint16_t *out_row);

/**
 * Write logical row 0 into `tile_count` consecutive bf16 tiles, leaving rows 1-31 alone.
 *
 * @param row  `tile_count * 32` bf16 elements
 * @return     as ft_tt_tile_row0_read()
 *
 * The caller must have staged rows 1-31 as zero. See the header note.
 */
int ft_tt_tile_row0_write(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                          uint32_t base_address, uint32_t tile_count, const uint16_t *row);

#endif /* FT_TT_TILE_ROW0_H */
