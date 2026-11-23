/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_gdngate_protocol.h
 * @brief The parameter block and the block layout of the GDN GATE op family — the Gated
 *   DeltaNet's per-position gates computed ON THE CARD from the alpha/beta projections,
 *   shared by resident_gdngate_reader.c, resident_gdngate_writer.c and the host half
 *   ft_tt_op_gdngate.h (tests/test_gdngate_ops.c pins the pair off-hardware).
 *
 * WHAT IT COMPUTES (Pantheon tt_ornith_optimized_spec.md, "ITEM 5"): for every position r of
 * a layer's rows and every value head h,
 *     beta[r][h]  = ft_det_gdn_beta(b[r][h])
 *     decay[r][h] = ft_det_gdn_gate(alpha[r][h], ssm_dt[h], ssm_a[h])   (fused, ft_det_math.h)
 * with ft_det_math.h — the SAME integer code the host compiles — and lays each 16-row
 * sub-chunk out as the gdnc DELTA reader's COMPACT gate block (ftgc_compact.h), byte for byte
 * what the host's `gdnc_compact_stage()` builds from host-computed values:
 *     word 0           the host's 1/sqrt(S) bits (passed in, never recomputed)
 *     words 1..15      zero
 *     per head h       decay[0..31] | beta[0..31]; rows >= the sub-chunk's live rows zero
 *
 * THE INPUTS. alpha and beta are each ftgg_tiles(heads) bf16 32x32 tiles a band, side by side
 * (32 positions x 32 heads a tile, face-ordered: element (row, head) at ftgg_element(row,
 * head)), so band b of a strip is at `base + b * band_stride`. The per-layer constants are a
 * 128-word block in DRAM: ssm_a[64] then ssm_dt[64] as fp32 bits, uploaded once.
 *
 * THE OUTPUT. Block (band, sub) goes to `dst_base + (2 * band + sub) * block_stride` — two
 * slots a band, always, so a slot's address is a function of its sub-chunk alone. sub 1 is
 * written only when the band's rows exceed `sub_rows` (the gdnc band loop's own split: a band
 * of <= sub_rows rows is ONE sub-chunk of `rows` rows at first_row 0).
 *
 * DM0|DM1, NO COMPUTE THREAD — the memory family's shape and its argument
 * (resident_memory_protocol.h): a resident BRISC never returns, so TRISC0 never zeroes the CB
 * page counters underneath the reader/writer handshake.
 */
#ifndef FOUNDRY_RESIDENT_GDNGATE_PROTOCOL_H
#define FOUNDRY_RESIDENT_GDNGATE_PROTOCOL_H

#include <stdint.h>

#include "ft_det_math.h"

/** 0x16: the next free opcode after ROWCOPY 0x15 (ft_tt_queue.h). */
enum { kOpGdnGate = 0x16 };

/* ----- the 16-word parameter block ----- */
enum { kGgFlags = 0 };       /**< bit 0 red `gdngate_beta_as_decay`; bit 1 FLAT (decode) */
enum { kGgRows = 1 };        /**< positions this descriptor covers, from band 0 row 0 */
enum { kGgBandCount = 2 };   /**< ceil(rows / 32) */
enum { kGgSubRows = 3 };     /**< the DELTA sub-chunk length: 16 or 32 */
enum { kGgHeads = 4 };       /**< value heads, 1..64 (two tile columns) */
enum { kGgAlphaXY = 5 };     /**< x | y << 16 */
enum { kGgAlphaBase = 6 };   /**< band 0's tile */
enum { kGgBetaXY = 7 };
enum { kGgBetaBase = 8 };
enum { kGgBandStride = 9 };  /**< bytes between consecutive bands of a strip */
enum { kGgConstXY = 10 };
enum { kGgConstBase = 11 };  /**< ssm_a[64] | ssm_dt[64], fp32 bits */
enum { kGgDstXY = 12 };
enum { kGgDstBase = 13 };    /**< slot (band 0, sub 0) */
enum { kGgBlockStride = 14 };
enum { kGgScaleBits = 15 };  /**< header word 0 */
enum { kGgWords = 16 };

enum { kGgFlagRedBetaAsDecay = 0x1 };
/** ITEM 6: the DECODE gate layout instead of COMPACT blocks — row r (< rows <= 32, one band)
 *  as flat decay[H] | beta[H] fp32 at `dst_base + r * block_stride`, exactly what
 *  `pn_tt_gdn_gate_upload{,_rows}()` wrote from the host. A descriptor covers the heads
 *  [kGgSubRows, kGgSubRows + kGgHeads) of kGgBandStride total (the words FLAT has no other use
 *  for), so the members split the heads. */
enum { kGgFlagFlat = 0x2 };

/* ----- shapes ----- */
enum { kGgTileRows = 32 };
enum { kGgFaceRows = 16 };
enum { kGgTileHeads = 32 };   /**< heads a tile column holds */
enum { kGgTileElements = 1024 };
enum { kGgMaxTiles = 2 };
enum { kGgMaxHeads = kGgMaxTiles * kGgTileHeads };
enum { kGgHeaderWords = 16 };
enum { kGgHeadWords = 64 };
enum { kGgConstWords = 2 * kGgMaxHeads };
/** One block at the full 64 heads: 16 + 64 * 64 words. */
enum { kGgBlockBytes = (16 + kGgMaxHeads * 64) * 4 };
/** Bands one descriptor may cover: a super-band of 26 bands with room to spare. */
enum { kGgMaxBands = 64 };

#if defined(__cplusplus)
#define FTGG_INLINE inline
#else
#define FTGG_INLINE static inline
#endif

/** Element (row, col) of a face-ordered 32x32 tile: faces 0/1 hold rows 0-15 (cols 0-15 /
 *  16-31), faces 2/3 rows 16-31. The same arithmetic as ft_tt_tilize.h. */
FTGG_INLINE uint32_t ftgg_face_index(uint32_t row, uint32_t col)
{
    const uint32_t face = (row >= kGgFaceRows ? 2u : 0u) + (col >= kGgFaceRows ? 1u : 0u);

    return face * kGgFaceRows * kGgFaceRows + (row % kGgFaceRows) * kGgFaceRows +
           (col % kGgFaceRows);
}

/** Tiles a band of an alpha or beta strip spans at `heads` heads. */
FTGG_INLINE uint32_t ftgg_tiles(uint32_t heads)
{
    return (heads + kGgTileHeads - 1u) / kGgTileHeads;
}

/** Element (row, head) of a band's side-by-side tiles: tile head / 32, column head % 32. */
FTGG_INLINE uint32_t ftgg_element(uint32_t row, uint32_t head)
{
    return (head / kGgTileHeads) * kGgTileElements + ftgg_face_index(row, head % kGgTileHeads);
}

/** Live rows of sub-chunk `sub` of a band of `band_rows` rows (0 = the slot is not written).
 *  The gdnc band loop's split: `sub_rows >= band_rows` is one sub-chunk of every row. */
FTGG_INLINE uint32_t ftgg_sub_live_rows(uint32_t band_rows, uint32_t sub_rows, uint32_t sub)
{
    if (sub_rows == 0u || sub_rows >= band_rows)
        return sub == 0u ? band_rows : 0u;
    if (sub == 0u)
        return sub_rows;
    return band_rows > sub_rows ? band_rows - sub_rows : 0u;
}

/** Block bytes at `heads` heads (64-byte aligned slots are the host's business). */
FTGG_INLINE uint32_t ftgg_block_bytes(uint32_t heads)
{
    return (kGgHeaderWords + heads * kGgHeadWords) * 4u;
}

/**
 * Build one sub-chunk's COMPACT block — THE one definition, compiled by the card reader and by
 * the host (its reference and its verify), so the layout cannot drift between them.
 *
 * @param block       ftgg_block_bytes(heads) bytes, fully written
 * @param alpha,beta  one band's ftgg_tiles(heads) face-ordered bf16 tiles, side by side
 * @param constants   ssm_a[64] | ssm_dt[64], fp32 bits
 * @param first,live  the sub-chunk: band rows [first, first + live)
 * @param red         kGgFlagRedBetaAsDecay: the decay run carries beta (must DIFFER)
 */
FTGG_INLINE void ftgg_build_block(uint32_t *block, uint32_t heads, uint32_t scale_bits,
                                  const uint16_t *alpha, const uint16_t *beta,
                                  const uint32_t *constants, uint32_t first, uint32_t live,
                                  int red)
{
    const uint32_t words = ftgg_block_bytes(heads) / 4u;

    for (uint32_t word = 0; word < words; word++)
        block[word] = 0u;
    block[0] = scale_bits;
    for (uint32_t head = 0; head < heads; head++) {
        uint32_t *run = block + kGgHeaderWords + head * kGgHeadWords;
        const uint32_t ssm_a = constants[head];
        const uint32_t ssm_dt = constants[kGgMaxHeads + head];

        for (uint32_t row = 0; row < live; row++) {
            const uint32_t at = ftgg_element(first + row, head);
            const uint32_t b = ft_det_gdn_beta((uint32_t)beta[at] << 16);
            const uint32_t decay = ft_det_gdn_gate((uint32_t)alpha[at] << 16, ssm_dt, ssm_a);

            run[row] = red ? b : decay;
            run[kGgTileRows + row] = b;
        }
    }
}

/** Bytes the FLAT layout of `rows` rows spans at `total_heads` heads: the last row ends at its
 *  live prefix, so a single row is exactly the 2 * heads words the decode gate buffer holds. */
FTGG_INLINE uint32_t ftgg_flat_bytes(uint32_t rows, uint32_t total_heads, uint32_t row_stride)
{
    return (rows - 1u) * row_stride + 2u * total_heads * 4u;
}

/**
 * Build the FLAT (decode) gate words of heads [head_first, head_first + heads) of band rows
 * [0, rows) at `out` (laid out as the whole block: row r at word r * row_stride / 4, decay[h]
 * at h, beta[h] at total_heads + h). Only this slice's words are written — the members split
 * the heads and the writer puts only the slice's two runs a row in DRAM. Same arithmetic as
 * ftgg_build_block().
 */
FTGG_INLINE void ftgg_build_flat(uint32_t *out, uint32_t head_first, uint32_t heads,
                                 uint32_t total_heads, const uint16_t *alpha,
                                 const uint16_t *beta, const uint32_t *constants, uint32_t rows,
                                 uint32_t row_stride, int red)
{
    for (uint32_t row = 0; row < rows; row++) {
        uint32_t *line = out + row * (row_stride / 4u);

        for (uint32_t head = head_first; head < head_first + heads; head++) {
            const uint32_t at = ftgg_element(row, head);
            const uint32_t b = ft_det_gdn_beta((uint32_t)beta[at] << 16);
            const uint32_t decay = ft_det_gdn_gate((uint32_t)alpha[at] << 16,
                                                   constants[kGgMaxHeads + head], constants[head]);

            line[head] = red ? b : decay;
            line[total_heads + head] = b;
        }
    }
}

#endif /* FOUNDRY_RESIDENT_GDNGATE_PROTOCOL_H */
