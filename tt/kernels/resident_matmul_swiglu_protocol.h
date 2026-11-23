/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_matmul_swiglu_protocol.h
 * @brief The FUSED-SWIGLU matmul descriptor (program `rtriad_bfp8_swiglu`, built with
 *   -DFT_RTRIAD_SWIGLU=1): one descriptor per member carries the gate AND the up panel of the
 *   member's columns and writes silu(gate) * up, with the output strip's mirrors.
 *
 *   Included by BOTH sides verbatim, like resident_matmul_block_protocol.h: the kernels
 *   (reader, compute, writer) and the host that writes the words (Pantheon
 *   pn_tt_swiglu_fuse.c, tests/test_swiglu_fusion.c). Only the FT_RTRIAD_SWIGLU build includes
 *   it, so every shipped image preprocesses to the tokens it had before this file existed.
 *
 * THE WORDS. The sixteen parameter words are full, so the fused descriptor is an ordinary
 *   member descriptor of the GATE projection (A = the input mirror, B = the gate shard,
 *   C = the output at the member's first column, Nt = the member's column count) plus:
 *     word 15 (kMatmulColumnBlock) bit 24  FUSED: the descriptor is a SwiGLU pair
 *                                  bit 25  silu on row 0 only (the eltwise kVecEltwiseRow0 rule)
 *                                  bit 26  RED: silu kept fp32 into the multiply (no bf16
 *                                          staging); the answer must DIFFER by bytes
 *     descriptor a1                the UP shard's DRAM base
 *     descriptor a2                the UP shard's bank coordinate, x | y << 16
 *   Word 15's low 24 bits keep the column and row block (ftmm_column_block/ftmm_row_block mask
 *   them), and a1/a2 are free on a fused descriptor because it is Mt = 1 (a2, the writer's row
 *   pitch, is multiplied by zero) and never stashes or starves (a1). The reader refuses a fused
 *   descriptor that is not Mt = 1, MB = 1, pipelined, N-major and solo.
 *
 * THE ORDER. The reader walks 2 * ceil(Nt / NB) passes: pass q reads panel q & 1 (0 gate,
 *   1 up) of column block q >> 1, every K block of it, exactly as an unfused pass. So the
 *   compute sees, per column block of w columns, w gate tiles then w up tiles, packs each pass
 *   bf16 into NB pages of the PANEL CB (pad pages after a short pass), and runs eltwise
 *   SILU_MUL's own sequence on tile pair (j, NB + j):
 *   datacopy -> SFPU silu (fp32 DST) -> pack bf16 to the STAGE CB -> MUL at HiFi4 -> pack bf16.
 *   Every rounding point of gate -> ffn_gate, up -> ffn_up, silu_mul is reproduced in order.
 *
 * @status Added 2026-11-17 (Nex-n2 qwen38_tt_chain_fusion_plan.md 3.3, S3b).
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_MATMUL_SWIGLU_PROTOCOL_H
#define RESIDENT_MATMUL_SWIGLU_PROTOCOL_H

#include <stdint.h>

#ifndef FTSW_INLINE
#define FTSW_INLINE static inline
#endif

/** Word 15 bit 24: the descriptor is a fused gate/up pair. */
#define FT_TT_MATMUL_SWIGLU_FLAG 0x01000000u
/** Word 15 bit 25: silu on row 0 only (the live row of a single-token tile). */
#define FT_TT_MATMUL_SWIGLU_SILU_ROW0 0x02000000u
/** Word 15 bit 26: RED, the silu is not staged bf16; the multiply takes it fp32. */
#define FT_TT_MATMUL_SWIGLU_RED_NO_STAGE 0x04000000u
/** Every fused bit; the low 24 bits stay the block word. */
#define FT_TT_MATMUL_SWIGLU_BITS 0x07000000u

/** CB holding silu(gate) staged bf16 (the eltwise cb_scratch). */
#define FT_TT_MATMUL_SWIGLU_CB_STAGE 2u
/** CB holding a column block's gate tiles then its up tiles, bf16 (2 * NB pages). */
#define FT_TT_MATMUL_SWIGLU_CB_PANEL 3u
/** Pages the stage CB needs: one tile in flight, two so PACK never waits on UNPACK. */
#define FT_TT_MATMUL_SWIGLU_STAGE_PAGES 2u

/** a2 of a fused descriptor: the up shard's bank coordinate. */
FTSW_INLINE uint32_t ftsw_up_xy(uint32_t x, uint32_t y)
{
    return (x & 0xFFFFu) | ((y & 0xFFFFu) << 16);
}

FTSW_INLINE uint32_t ftsw_up_x(uint32_t a2)
{
    return a2 & 0xFFFFu;
}

FTSW_INLINE uint32_t ftsw_up_y(uint32_t a2)
{
    return a2 >> 16;
}

/** Pages of one column block in the panel CB (two passes), and the CB's size: one block. */
FTSW_INLINE uint32_t ftsw_panel_pages(uint32_t column_block)
{
    return 2u * column_block;
}

/** Reader passes of a fused descriptor: a gate pass and an up pass per column block. */
FTSW_INLINE uint32_t ftsw_passes(uint32_t tile_cols, uint32_t column_block)
{
    return column_block == 0u ? 0u : 2u * ((tile_cols + column_block - 1u) / column_block);
}

/** The panel pass `pass` reads: 0 gate (descriptor B), 1 up (a1/a2). */
FTSW_INLINE uint32_t ftsw_pass_panel(uint32_t pass)
{
    return pass & 1u;
}

/** The first output column of pass `pass`. */
FTSW_INLINE uint32_t ftsw_pass_column(uint32_t pass, uint32_t column_block)
{
    return (pass >> 1) * column_block;
}

/** Columns of pass `pass` (short only on the ragged last block). */
FTSW_INLINE uint32_t ftsw_pass_columns(uint32_t pass, uint32_t tile_cols, uint32_t column_block)
{
    const uint32_t nt = ftsw_pass_column(pass, column_block);

    return (tile_cols - nt) < column_block ? (tile_cols - nt) : column_block;
}

/* THE PANEL LAYOUT IS FIXED AT NB PAGES A PASS (card fix 1, 2026-11-17). The SILU_MUL half
 * reads the panel by INDEX (ft_tensix_unpack_a / _ab: read pointer + page * index), and an
 * indexed unpack does NOT wrap at the CB's end. So every column block must start on the
 * ring's 2*NB-page grid: a pass of w < NB columns (the ragged last block) pushes NB - w pad
 * pages, the block is always 2*NB pages, and the panel CB is a whole number of blocks. The
 * first layout (w gate tiles, w up tiles, pop 2w) left the pointers 2w pages off the grid after
 * a ragged block, so every LATER descriptor's indices ran past the CB (Qwen3.8 Nt 34 at NB 8:
 * 4 pages off after the first FFN; output garbage from the 2nd decode token). */

/** Pages one pass occupies in the panel CB: always the column block (pad pages included). */
FTSW_INLINE uint32_t ftsw_panel_pass_pages(uint32_t column_block)
{
    return column_block;
}

/** Panel-CB index of the gate tile of column `j` of a block (the block's first pass). */
FTSW_INLINE uint32_t ftsw_panel_gate_index(uint32_t j)
{
    return j;
}

/** Panel-CB index of the up tile paired with gate column `j`: the second pass starts NB pages
 *  in, whatever the block's width. */
FTSW_INLINE uint32_t ftsw_panel_up_index(uint32_t j, uint32_t column_block)
{
    return ftsw_panel_pass_pages(column_block) + j;
}

#endif /* RESIDENT_MATMUL_SWIGLU_PROTOCOL_H */
