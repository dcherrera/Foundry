/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_matmul_resadd_protocol.h
 * @brief The RESIDUAL-ADD matmul descriptor (fusion S3a; program `rtriad_bfp8_swiglu`, built
 *   with -DFT_RTRIAD_SWIGLU=1 -DFT_RTRIAD_RESADD=1): one descriptor per member of an
 *   out-projection or FFN down writes residual + bf16(input . W) into the residual stream's
 *   next strip, so the eltwise ADD after it is gone.
 *
 *   Included by BOTH sides verbatim, like resident_matmul_swiglu_protocol.h: the kernels and
 *   the host that writes the words (Pantheon pn_tt_resadd_fuse.c, tests/test_resadd_fusion.c).
 *   Only the FT_RTRIAD_RESADD build includes it, so every other image is unchanged.
 *
 * THE WORDS. An ordinary member descriptor of the projection (A = its input, B = its shard,
 *   C = the OUTPUT strip at the member's first column, Nt = the member's columns) plus:
 *     word 15 (kMatmulColumnBlock) bit 27  RESADD: C = residual + bf16(A . B)
 *                                  bit 28  RED: the add is skipped (C = bf16(A . B)); the
 *                                          answer must DIFFER
 *     descriptor a1                the RESIDUAL strip's address at the member's first column
 *     descriptor a2                its bank coordinate, x | y << 16
 *   Word 15's low 24 bits keep the block word; a1/a2 are free for the reason the fused SwiGLU
 *   gives (Mt = 1: a2, the row pitch, is multiplied by zero; never stashed or starved). The
 *   reader refuses a RESADD descriptor that is not Mt = 1, MB = 1, pipelined, N-major, solo,
 *   or that also carries the SwiGLU flag.
 *
 * THE ARITHMETIC IS TODAY'S, ROUNDING POINT FOR ROUNDING POINT. Today: the matmul packs DST
 *   (fp32) bf16 into `project`; eltwise ADD then unpacks hidden (SrcA) and project (SrcB),
 *   ELWADD into fp32 DST, packs bf16. Fused, per column block of w columns: the projection's K
 *   loop packs bf16 into the PANEL CB (the same pack of the same DST: the first rounding),
 *   SrcA goes to bf16, and eltwise ADD's own verbs run on (residual tile j as A, panel tile j
 *   as B), packed bf16 to CB_OUT (the second rounding). Same CB formats (Float16_b), same fp32
 *   DST, same ELWADD word, same operand order.
 *
 * THE CB GRID. The panel CB (2 * NB pages) is shared with the fused SwiGLU, whose indexed
 *   unpacks need every block on the 2 * NB grid (card fix 1 of S3b), so a RESADD block also
 *   takes 2 * NB panel pages: w projection tiles, then pad pages. The RESIDUAL CB (2 * NB
 *   pages) takes NB pages a block (w residual tiles, then pad), read at index j < NB from a
 *   read pointer on the NB grid, so no index runs past its end.
 *
 * @status Added with S3a (q38dec-p5-s3).
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_MATMUL_RESADD_PROTOCOL_H
#define RESIDENT_MATMUL_RESADD_PROTOCOL_H

#include <stdint.h>

#ifndef FTRA_INLINE
#define FTRA_INLINE static inline
#endif

/** Word 15 bit 27: C = residual + bf16(A . B). */
#define FT_TT_MATMUL_RESADD_FLAG 0x08000000u
/** Word 15 bit 28: RED, the add is skipped (C = bf16(A . B)). */
#define FT_TT_MATMUL_RESADD_RED_SKIP 0x10000000u
/** Every RESADD bit; disjoint from the block word (low 24) and the SwiGLU bits (24..26). */
#define FT_TT_MATMUL_RESADD_BITS 0x18000000u

/** CB the reader lands a column block's residual tiles in, bf16. */
#define FT_TT_MATMUL_RESADD_CB_RESIDUAL 4u

/** a2 of a RESADD descriptor: the residual strip's bank coordinate. */
FTRA_INLINE uint32_t ftra_residual_xy(uint32_t x, uint32_t y)
{
    return (x & 0xFFFFu) | ((y & 0xFFFFu) << 16);
}

FTRA_INLINE uint32_t ftra_residual_x(uint32_t a2)
{
    return a2 & 0xFFFFu;
}

FTRA_INLINE uint32_t ftra_residual_y(uint32_t a2)
{
    return a2 >> 16;
}

/** Column blocks (reader passes) of a RESADD descriptor: one per block, as unfused. */
FTRA_INLINE uint32_t ftra_blocks(uint32_t tile_cols, uint32_t column_block)
{
    return column_block == 0u ? 0u : (tile_cols + column_block - 1u) / column_block;
}

/** Columns of block `block` (short only on the ragged last block). */
FTRA_INLINE uint32_t ftra_block_columns(uint32_t block, uint32_t tile_cols, uint32_t column_block)
{
    const uint32_t nt = block * column_block;

    return (tile_cols - nt) < column_block ? (tile_cols - nt) : column_block;
}

/** Panel-CB pages a RESADD block takes: the fused SwiGLU's whole block (2 * NB). */
FTRA_INLINE uint32_t ftra_panel_block_pages(uint32_t column_block)
{
    return 2u * column_block;
}

/** Residual-CB pages a block takes (pad included), and the CB's size: two blocks. */
FTRA_INLINE uint32_t ftra_residual_block_pages(uint32_t column_block)
{
    return column_block;
}

FTRA_INLINE uint32_t ftra_residual_cb_pages(uint32_t column_block)
{
    return 2u * column_block;
}

/** Byte offset of column `column` (from the member's first) in the residual strip. */
FTRA_INLINE uint32_t ftra_residual_offset(uint32_t column, uint32_t tile_bytes)
{
    return column * tile_bytes;
}

#endif /* RESIDENT_MATMUL_RESADD_PROTOCOL_H */
