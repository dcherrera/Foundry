// SPDX-License-Identifier: AGPL-3.0-or-later
//
// resident_rope_protocol.h
//   The parameter block the ROPE op reads out of L1, shared by the resident RoPE triad
//   (resident_rope_{reader,compute,writer}.cpp). The host half is ft_tt_op_rope.h; the two
//   must agree word for word and tests/test_rope_ops.c pins the pair off-hardware by
//   PARSING this file.
//
// ===========================================================================
// THE FACE-SPACE ARGUMENT — THE WHOLE REASON THIS OP IS SHAPED LIKE THIS
// ===========================================================================
// A 32x32 tile is four 16x16 faces stored in FACE order, so element (r, c) sits at
//
//     offset(r, c) = (r/16)*512 + (c/16)*256 + (r%16)*16 + (c%16)
//
// which is NOT r*32 + c (docs/tt_kernel_model.md 9). Any op that moves elements around
// inside a tile has to move FACES, and getting it wrong is invisible to PCC.
//
// RoPE pairs feature j with feature j + half (half = rot/2) and rotates the pair. Activations
// are [tokens x features] with a feature row along W, so
//
//     feature f  ->  tile column f/32, intra-tile column f%32
//     token   t  ->  tile row      t/32, intra-tile row    t%32
//
// THEREFORE, WHEN half IS A MULTIPLE OF 32:
//
//     j and j + half have the SAME intra-tile column and the SAME row.
//     They differ ONLY in which TILE they are in.
//
// So the pairing is a permutation of TILES and, inside a tile, the IDENTITY. The rotation
// never crosses a face boundary, and the face order does not enter the arithmetic at all:
// every operation below is a tile-elementwise multiply or add, which is offset-wise and
// therefore face-order agnostic. The reader implements the whole pairing by ADDRESSING —
// it fetches the partner tile stream — and the compute thread never indexes an element.
//
// This is not "we avoided the hard case": it is the statement that for a tile-aligned split
// the hard case does not exist. What IS face-critical is the TABLE (below).
//
// ===========================================================================
// WHAT IS FACE-CRITICAL: THE SIN/COS TABLE
// ===========================================================================
// The table value varies with BOTH the row (position) and the column (frequency), so a table
// tilized with a different face map than the activations assigns the wrong ANGLE to three
// quarters of every tile — a wrong answer, not a permuted one. The host builds it with the
// same ft_tt_tilize_matrix() the activations go through, and tests/test_rope_ops.c reads
// individual table elements back through ft_tt_tile_face_index() to say so.
//
// ===========================================================================
// WHAT IS REFUSED, AND WHY IT IS A REFUSAL
// ===========================================================================
//   half % 32 == 16   j and j+half land in different FACE COLUMNS of their tiles. Buildable
//                     — a face is 256 contiguous elements, so a reader could assemble the
//                     partner tile from four face-sized NoC reads — but not built. REFUSED.
//   half % 16 != 0    the partner is at an arbitrary intra-face column offset. No NoC read
//                     shape produces it; it needs an element gather or a transpose. REFUSED.
//   INTERLEAVED RoPE  the GGUF `llama`/NORM style pairs (2i, 2i+1), i.e. ADJACENT COLUMNS
//                     inside one face row. That is not a tile permutation at all. The fix is
//                     offline and free: permute the wq/wk output rows in the converter so the
//                     activations arrive in split-half order (llama.cpp's own permutation,
//                     run backwards). Q.K is invariant under a permutation applied to BOTH
//                     Q and K, so it costs nothing and changes no math. REFUSED here.
//
// A refusal is kErrBadParameters, never a silent fallback: this op's failure mode is a
// plausible wrong answer.
//
// ===========================================================================
// THE TABLE IS AN OPERAND, NOT A KERNEL CONSTANT
// ===========================================================================
// TRISC2 has 1760 bytes of private local memory (dev notes 5). One row block of MiniCPM5-1B's
// table is 4 cos tiles + 4 sin tiles = 16 KiB, and the whole 131072-position table is 67 MB.
// It is a DRAM operand streamed through circular buffers like any other input, and it is
// PRECOMPUTED ON THE HOST — see ft_tt_op_rope.h for the measurement and the argument (the
// deciding one is precision, not bandwidth: angle = pos * freq reaches 1.3e5 radians and
// nothing on the SFPU reduces that argument usefully).
//
#ifndef FOUNDRY_RESIDENT_ROPE_PROTOCOL_H
#define FOUNDRY_RESIDENT_ROPE_PROTOCOL_H

#include <stdint.h>

/* Header-only helper linkage, spelled the same way in both languages -- `inline` in C++ and
 * `static inline` in C. NOT cosmetic: internal linkage changes GCC's inlining and cloning
 * decisions, and the COMPUTE images that also include this header must not move for a
 * change that is about the data-movement ones. */
#ifndef FTQ_INLINE
#ifdef __cplusplus
#define FTQ_INLINE inline
#else
#define FTQ_INLINE static inline
#endif
#endif

#ifndef __cplusplus
#include <stdbool.h>
#endif

// ---------------------------------------------------------------------------
// The sixteen words. Mirrors ft_tt_op_rope.h -- change both or neither.
// ---------------------------------------------------------------------------

/** TOTAL activation tiles this descriptor moves, in and out. The writer reads this and
 *  nothing else about the shape. = row_blocks * heads * head_tiles. */
enum { kRopeTileCount = 0 };

/** Tiles across ONE head: head_dim / 32. */
enum { kRopeHeadTiles = 1 };

/** Tiles of a head that are ROTATED: rot / 32, where rot = rope_dim (0 < rope_dim <=
 *  head_dim) else head_dim. Tiles [rot_tiles, head_tiles) pass through untouched — that is
 *  the whole of "partial". Must be EVEN, so the half-split is tile-aligned. */
enum { kRopeRotTiles = 2 };

/** Heads in one row block of the activation. Needed because the TABLE advances once per row
 *  block while the activation advances once per (row block, head). */
enum { kRopeHeads = 3 };

/** Tiles per NoC barrier in the reader. Must DIVIDE head_tiles AND rot_tiles or a block
 *  straddles the CB wrap; the kernel repeats the host's arithmetic and falls back to 1. */
enum { kRopeDepth = 4 };

/** The activation. */
enum { kRopeAX = 5 };
enum { kRopeAY = 6 };
enum { kRopeABase = 7 };

/** The TABLE, already offset to this descriptor's first position row block. One row block is
 *  2*rot_tiles tiles: rot_tiles of cos, then rot_tiles of SIGNED sin. */
enum { kRopeTableX = 8 };
enum { kRopeTableY = 9 };
enum { kRopeTableBase = 10 };

/** The output. Never the input: this op is NOT in place, because the partner tile of an
 *  already-rotated tile is the wrong operand and a reader has no ordering that fixes it. */
enum { kRopeCX = 11 };
enum { kRopeCY = 12 };
enum { kRopeCBase = 13 };

/** Sixteen, matching FT_TT_QUEUE_PARAM_WORDS. Two spare. */
enum { kRopeWords = 16 };

/**
 * The block depth a reader will actually use over a run of `run` tiles.
 *
 * Transliterated from ft_tt_op_block_depth(); the kernel repeats the host's arithmetic
 * rather than trusting it, exactly as resident_eltwise_reader.c does. A depth that does
 * not divide the run moves the CB write pointer off the block grid and every later block
 * straddles the wrap. Falling back to 1 is always correct.
 */
FTQ_INLINE uint32_t ftr_block_depth(uint32_t run, uint32_t requested)
{
    return (requested > 0u && run > 0u && run % requested == 0u) ? requested : 1u;
}

/**
 * Is this shape one the tile-aligned pairing can express?
 *
 * THE ONLY definition, host side and device side both (ft_tt_op_rope.c carries the same
 * three clauses and tests/test_rope_ops.c asserts the two agree). A shape that fails this
 * is refused with kErrBadParameters — never rotated with a partner stream that is merely
 * plausible.
 */
FTQ_INLINE bool ftr_shape_is_supported(uint32_t head_tiles, uint32_t rot_tiles)
{
    if (head_tiles == 0u || rot_tiles == 0u || rot_tiles > head_tiles)
        return false;
    /* rot_tiles even <=> half = rot/2 is a whole number of tiles <=> the pair is the same
     * intra-tile column of a different tile. This one clause IS the face-space argument. */
    return (rot_tiles % 2u) == 0u;
}

/**
 * The partner column-tile of column-tile `c` inside one head: a rotation by half.
 *
 * c < half  ->  c + half   (its output is x*cos - partner*sin; the table carries the minus)
 * c >= half ->  c - half   (its output is x*cos + partner*sin)
 *
 * The SIGN IS IN THE TABLE, not here — see ft_tt_op_rope_build_table(). That is what makes
 * every rotated tile the same three instructions with no branch, no scalar circular buffer
 * and no negation pass, where upstream's own kernel
 * (ttnn/.../rotary_embedding_hf/device/kernels/compute/rotary_embedding_hf.cpp) spends an
 * extra multiply-by-(-1) pass on half its tiles.
 */
FTQ_INLINE uint32_t ftr_partner_tile(uint32_t c, uint32_t rot_tiles)
{
    const uint32_t half = rot_tiles / 2u;

    return (c < half) ? (c + half) : (c - half);
}

#endif  // FOUNDRY_RESIDENT_ROPE_PROTOCOL_H
