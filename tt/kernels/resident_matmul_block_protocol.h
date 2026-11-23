/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_matmul_block_protocol.h
 * @brief The output-BLOCK contract of the resident matmul triad: how many output rows and
 *   how many output columns one pass of the reader supplies, packed into the one parameter
 *   word `kMatmulColumnBlock` already carries.
 *
 *   **Included by BOTH sides verbatim**, like resident_matmul_multicast_protocol.h and for
 *   the same reason: a wire contract that is duplicated is one edit away from a kernel
 *   reading a different word than the host wrote. Nothing here but preprocessor
 *   definitions and pure inline arithmetic, so <stdint.h> is the only include.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS A FAMILY HEADER AND NOT AN EDIT TO resident_queue_protocol.h
 * ---------------------------------------------------------------------------
 * `resident_queue_protocol.h` is a content-provenance input to EVERY embedded image: a
 * comment added there makes all 23+ programs stale and forces a full blob regeneration.
 * The parameter WORD (`kMatmulColumnBlock`, index 15) is unchanged and stays registered
 * there; only its ENCODING lives here, and only the five matmul images read it.
 *
 * ---------------------------------------------------------------------------
 * THE ARITHMETIC THIS WORD IS FOR, ITS CEILING, AND WHY MB IS WORTH 2%
 * ---------------------------------------------------------------------------
 * One pass of the reader holds `MB` A tile-rows and `NB` B columns and sweeps K past both.
 * Over a whole descriptor that costs
 *
 *     A tiles = Mt * ceil(Nt/NB) * Kt          (A is read once per row per column BLOCK)
 *     B tiles = ceil(Mt/MB) * Nt * Kt          (B is read once per column per row BLOCK)
 *     bytes   = Mt * T * (a/NB + b/MB)         T = sum of Kt*Nt, when the blocks divide
 *
 * where `a` is the ACTIVATION tile size and `b` the WEIGHT tile size. **The two are
 * symmetric in TILES and are not symmetric in BYTES**, and that is the whole of the
 * result: NB divides the operand whose tile is `a` and MB divides the one whose tile is
 * `b`, so the blocking budget belongs to whichever operand has the bigger tile.
 *
 * **THE CEILING IS THE DST SECTION AND IT IS A PRODUCT.** A pass accumulates `MB * NB`
 * output tiles across the whole K loop, so every one of them is live in DST at once.
 * The bound is `dst_index + ct_dim*rt_dim - 1 < dst_tiles`, and at SyncHalf with
 * `dst_accum_mode 1` the section is `(FT_TENSIX_DEST_REGISTER_HALF_SIZE >> 1) >> 6 =
 * (512 >> 1) >> 6 = 4` tiles (measured; see ft_tensix_compute.h).
 * So `MB * NB <= 4`, and over that lattice, in bytes a pass:
 *
 *     weights          MB=1 NB=4        MB=2 NB=2          MB=4 NB=1
 *     Bfp8_b 1088    1600 (shipped)   1568  BEST, +2.0%   2320  -31%
 *     Bfp4_b  576    1088  BEST       1312  -17%          2192  -50%
 *     bf16   2048    2560             2048  BEST, +25%    2560   0%
 *
 * **SO ON A Q8_0 MODEL THE ROW BLOCK IS WORTH TWO PER CENT AND THE SHIPPED MB=1/NB=4 IS
 * ALREADY ALMOST OPTIMAL — and at Bfp4_b it is EXACTLY optimal and MB only loses.** The
 * activation tile is bf16 2048 B and a Bfp8_b weight tile is 1088, so re-reading A costs
 * 1.88x what re-reading B costs and the budget belongs to NB. MB pays in proportion as `b`
 * approaches `a`, and fully only at bf16 weights. The continuous optimum of `a/NB + b/MB`
 * on `MB*NB = 4` is `sqrt(a*b) = 1493` at MB = 1.46 — between the two integer points and 7%
 * under today. **That 7% is the entire headroom this word has at a four-tile DST.**
 *
 * VALIDATED AGAINST SILICON WITH NO FREE PARAMETER: at Mt=1, MB=1, NB=4, Bfp8_b, the law
 * says `A/B = a/(4b) = 2048/4352 = 0.4706`. Ornith's measured decode split is
 * A 3.978 GB / B 8.432 GB = **0.4718** — 0.25% apart.
 *
 * **THE LEVER THAT WOULD MATTER IS A BIGGER DST, NOT A BIGGER MB.** `DstSync::SyncFull`
 * gives eight tiles at the same `dst_accum_mode 1` (FULL rather than HALF, so
 * `(1024 >> 1) >> 6 = 8`), and it is `FT_TT_MATMUL_DST_SYNC` below.
 * `resident_matmul_dst_sync.h` is the rendering; it costs the MATH/PACK double buffer.
 *
 * **AND AT `Mt == 1` THE EIGHT TILES ARE ALL NB'S, BECAUSE MB HAS NOTHING TO DIVIDE.** The
 * `MB=2/NB=4 -> 1056 B, 1.52x` reading assumes `Mt >= 2`, and no descriptor in the tree has
 * ever had one: `pn_tt_matmul_group.c` writes `FT_TT_QUEUE_MATMUL_TILE_ROWS = 1` from a
 * literal, in decode AND in the batched prefill (`rows` is capped at 32 positions = one
 * tile-row). At `Mt = 1` the row loop runs once whatever MB says, so the reachable DST-8
 * point is **MB=1/NB=8 at `256 + 1088 = 1344` against the shipped 1600 — 1.19x**, and the
 * 1.52x is gated on `Mt >= 2` rather than on this header.
 *
 *     weights          MB=1 NB=8        MB=2 NB=4       reachable at Mt=1?
 *     Bfp8_b 1088    1344  (1.19x)    1056  (1.52x)    only MB=1 NB=8
 *     Bfp4_b  576     832  (1.31x)     800  (1.36x)    only MB=1 NB=8
 *     bf16   2048    2304  (1.11x)    1536  (1.67x)    only MB=1 NB=8
 *
 * **L1 IS NOT THE BINDING CONSTRAINT AND NEVER WAS**: at depth 8 the three CBs are
 * `MB*depth*2*2048 + depth*NB*2*weight_tile + 2*2048` bytes of a 1,351,680 B arena
 * (ft_tt_op_l1_layout.h), which admits MB = 39 at NB=4 with Bfp8_b weights and 37 with
 * bf16 — an order of magnitude past what DST allows.
 *
 * ---------------------------------------------------------------------------
 * ZERO IS WHAT EVERY DESCRIPTOR WRITTEN BEFORE THIS ENCODING EXISTED SAYS
 * ---------------------------------------------------------------------------
 * The column block keeps the LOW sixteen bits, so any legal value the word has ever
 * carried (1..4) decodes to itself. The row block is bits 16..23 and a zero there means
 * ONE, which is the shape every descriptor in the tree has today. At `MB == 1` the
 * reader's, the compute thread's and the writer's loops collapse to exactly the loops they
 * ran before the field existed — the row loop steps by one, `rows` is one, the A chunk is
 * `depth`, the DST base is zero and the destination tile index is the running counter.
 * That collapse is what makes a decode `cmp` the evidence that this change is inert.
 *
 * ---------------------------------------------------------------------------
 * THE DESTINATION ROW PITCH IS A DESCRIPTOR WORD, NOT A PARAMETER WORD
 * ---------------------------------------------------------------------------
 * At `MB > 1` the writer emits output tile `(m, n)` and has to place it, which needs the
 * pitch of the destination tile grid — the FULL Nt of the weight, not this member's slice
 * of it. The sixteen-word parameter block is full and a seventeenth word would move the
 * shared slot stride every resident family is placed on, so the pitch rides in descriptor
 * word `a2`, which `kOpMatmul` has never used (a0 is the parameter block, a1 the debug
 * starve count, a5 the group sequence). **Zero means `tile_cols`**, which is what a
 * single-tile-row descriptor computes anyway.
 *
 * @status Encoding complete; MB > 1 is exercised off hardware by tests/test_matmul_block.c.
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_MATMUL_BLOCK_PROTOCOL_H
#define RESIDENT_MATMUL_BLOCK_PROTOCOL_H

#include <stdint.h>

/** The DST section in 32x32 tiles at `dst_accum_mode 1`:
 *  `(FT_TENSIX_DEST_REGISTER_{HALF,FULL}_SIZE >> 1) >> 6`, i.e. `(512 >> 1) >> 6 = 4`
 *  at SyncHalf and `(1024 >> 1) >> 6 = 8` at SyncFull. */
#define FT_TT_MATMUL_DST_SECTION_TILES_HALF 4u
#define FT_TT_MATMUL_DST_SECTION_TILES_FULL 8u

/**
 * WHICH DST SYNC MODE THIS TRANSLATION UNIT WAS BUILT FOR, and the reason it is a
 * compile-time value rather than a descriptor word: the section size decides which DST
 * tiles a MOP may address, and a pass that writes past the section is a silently wrong
 * answer with every kernel retiring clean. A descriptor cannot be allowed to widen it.
 *
 *   HALF         the shipped stream. MATH owns one half of DST while PACK drains the
 *                other; `MB*NB <= 4`.
 *   FULL         MATH owns all of DST; `MB*NB <= 8`. The MATH/PACK double buffer is gone,
 *                so a section's MATH cannot start until the previous section's PACK has
 *                retired (the semaphore's max count falls from 2 to 1).
 *   HALF_SERIAL  HALF in every respect EXCEPT the max count, which is 1. Bit-identical
 *                arithmetic — same section size, same DST tiles, same pack order — so it
 *                is the ARM THAT PRICES FULL's serialisation ON ITS OWN, with the byte
 *                win held out. It is a measurement, never a shipped configuration.
 */
#define FT_TT_MATMUL_DST_SYNC_HALF 0
#define FT_TT_MATMUL_DST_SYNC_FULL 1
#define FT_TT_MATMUL_DST_SYNC_HALF_SERIAL 2

#ifndef FT_TT_MATMUL_DST_SYNC
#define FT_TT_MATMUL_DST_SYNC FT_TT_MATMUL_DST_SYNC_HALF
#endif

#if FT_TT_MATMUL_DST_SYNC == FT_TT_MATMUL_DST_SYNC_FULL
#define FT_TT_MATMUL_DST_SECTION_TILES FT_TT_MATMUL_DST_SECTION_TILES_FULL
#else
#define FT_TT_MATMUL_DST_SECTION_TILES FT_TT_MATMUL_DST_SECTION_TILES_HALF
#endif

/** Column block: the low sixteen bits, so every value the word has ever held is itself. */
#define FT_TT_MATMUL_BLOCK_COLUMN_SHIFT 0u
#define FT_TT_MATMUL_BLOCK_COLUMN_MASK 0xFFFFu
/** Row block: bits 16..23. Zero means one. */
#define FT_TT_MATMUL_BLOCK_ROW_SHIFT 16u
#define FT_TT_MATMUL_BLOCK_ROW_MASK 0xFFu

/** Output COLUMN tiles one pass supplies. 0 and 1 are the same shape. */
static inline uint32_t ftmm_column_block(uint32_t word)
{
    const uint32_t value = (word >> FT_TT_MATMUL_BLOCK_COLUMN_SHIFT) &
                           FT_TT_MATMUL_BLOCK_COLUMN_MASK;

    return value > 0u ? value : 1u;
}

/** Output ROW tiles one pass supplies. 0 and 1 are the same shape. */
static inline uint32_t ftmm_row_block(uint32_t word)
{
    const uint32_t value = (word >> FT_TT_MATMUL_BLOCK_ROW_SHIFT) & FT_TT_MATMUL_BLOCK_ROW_MASK;

    return value > 0u ? value : 1u;
}

/**
 * Build the word. `row_block == 1` reproduces the pre-encoding word BIT FOR BIT, which is
 * why a host that has not been taught about rows writes descriptors no kernel can tell
 * apart from the ones it wrote yesterday.
 */
static inline uint32_t ftmm_block_word(uint32_t row_block, uint32_t column_block)
{
    const uint32_t rows = row_block > 1u ? row_block : 0u;

    return ((column_block & FT_TT_MATMUL_BLOCK_COLUMN_MASK) << FT_TT_MATMUL_BLOCK_COLUMN_SHIFT) |
           ((rows & FT_TT_MATMUL_BLOCK_ROW_MASK) << FT_TT_MATMUL_BLOCK_ROW_SHIFT);
}

/** The section a build of the given sync mode gets, for a caller that is not that build —
 *  i.e. the HOST, which is compiled once and must ask about whichever kernel is armed. */
static inline uint32_t ftmm_dst_section_tiles(int sync_full)
{
    return sync_full ? FT_TT_MATMUL_DST_SECTION_TILES_FULL : FT_TT_MATMUL_DST_SECTION_TILES_HALF;
}

/**
 * Does this pair fit a section of `section_tiles` tiles?
 *
 * Asked on the HOST before a descriptor is written and in the KERNEL before a pass is run,
 * because the two failures cost differently: refused on the host is a return value, and a
 * pass that writes past the section is a silently wrong answer with every kernel retiring
 * clean — DST has no bounds check and the tiles past the end are the OTHER half, which the
 * packer is reading.
 *
 * The size is a PARAMETER because the host and the kernel can disagree about it: the host
 * is one binary and the kernel it drives may have been built either way. The host must ask
 * about the armed program, never about itself.
 */
static inline int ftmm_block_fits_section(uint32_t row_block, uint32_t column_block,
                                          uint32_t section_tiles)
{
    if (row_block == 0u || column_block == 0u)
        return 0;
    return row_block * column_block <= section_tiles;
}

/**
 * The same question against THIS translation unit's own section — what the KERNEL asks.
 *
 * SPELLED OUT RATHER THAN FORWARDED TO `ftmm_block_fits_section()`, and the reason is
 * measured: writing it as that call moved **22 bytes of `resident_matmul_reader.c`'s
 * `.text`** with no change in behaviour — same size, different register allocation through
 * `-flto`. The build is otherwise bit-reproducible (two rebuilds of one tree are identical
 * on all thirteen images), so a byte-identity `cmp` is this family's staleness test, and
 * four lines of duplicated arithmetic are cheaper than losing it.
 */
static inline int ftmm_block_fits_dst(uint32_t row_block, uint32_t column_block)
{
    if (row_block == 0u || column_block == 0u)
        return 0;
    return row_block * column_block <= FT_TT_MATMUL_DST_SECTION_TILES;
}

#endif /* RESIDENT_MATMUL_BLOCK_PROTOCOL_H */
