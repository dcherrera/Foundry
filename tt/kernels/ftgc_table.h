/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ftgc_table.h
 * @brief The chunkwise GDN's EXACT card gates: the gdnc DELTA reader builds a head's COMPACT
 *        run (decay[32] | beta[32], ftgc_compact.h) from the alpha/beta bf16 strips through
 *        HOST-BUILT LOOKUP TABLES, so the prefill reads no alpha/beta back to the host.
 *
 * WHY THE BITS ARE THE HOST'S. alpha and beta reach the host as bf16 widened by `<< 16`, and
 * the host's gate is elementwise — decay[r][h] = f_h(alpha[r][h]), beta[r][h] = g(beta[r][h]),
 * no cross-row arithmetic (the products and ratios are ftgc_compact.h's, downstream, the same
 * in both arms). A function of a 16-bit code is a 65,536-entry table, so the host evaluates
 * its own compiled gate on every code and the card looks the answer up. NaN codes are outside
 * the claim: the host refuses a non-finite gate (gdnc_compact_stage) before it reaches a
 * block.
 *
 * ONE TABLE = ONE FUNCTION OF A CODE, WINDOWED. With `m = code & 0x7FFF` and the sign bit
 * choosing a side, every side is constant below a window (tiny |x| vanishes in `x + dt`) and
 * constant above it (the exp saturates to 0 or 1), so a side stores only [lo, hi):
 *     m <  lo    the side's LOW value       m >= hi    the side's HIGH value
 *     else       entry (side's start + m - lo)
 * The host builder proves it by enumerating every non-NaN code against the table.
 *
 * THE CARD SIDE. Per descriptor the reader reads the layer's TABLE BLOCK (kGdncDeltaGate, in
 * the constant bank) and the beta table's header. Each layer's decay tables are STRIPED over S
 * banks (head h in stripe h % S) and every stripe's bank holds a copy of the beta table, so no
 * bank serves every core. Per head: the head's decay header and the alpha/beta face rows of
 * the sub-chunk (one round trip, faces cached across heads of one 16-column face), then one
 * 64 B line per looked-up entry (the NoC's DRAM granule), 16 in flight at a time. The decode
 * reader (ftgc_table_rows) builds its gate block from the same tables.
 *
 * The front end is written ONCE here against `ftgc_table_io` (a read and a barrier), so the
 * host test runs the card's own code over an emulated DRAM (tests/test_gdnc_table.c).
 *
 * @status Prefill card-proven exact (p150 2026-11-19, striped). Decode host-tested; card gate pending.
 * @issues None
 * @todo None
 */
#ifndef FTGC_TABLE_H
#define FTGC_TABLE_H

#include <stdint.h>

#include "ftgc_compact.h"

/* ----- the per-layer TABLE BLOCK: 48 words at kGdncDeltaGate in the constant bank, one per
 *       layer every kFtgtBlockStride bytes ----- */
enum { kFtgtBlockScale = 0 };     /**< the compact header's word 0: the host's 1/sqrt(S) bits */
enum { kFtgtBlockAlphaXy = 1 };   /**< x | y << 16 */
enum { kFtgtBlockAlphaBase = 2 }; /**< band 0's first tile of the alpha strip (bf16 tiles) */
enum { kFtgtBlockBetaXy = 3 };
enum { kFtgtBlockBetaBase = 4 };
enum { kFtgtBlockBandBytes = 5 }; /**< bytes between consecutive bands (both strips) */
enum { kFtgtBlockStripes = 6 };   /**< S in 1..kFtgtMaxStripes: head h's table is in stripe h % S */
enum { kFtgtBlockDecaySlot = 7 }; /**< bytes between a stripe's heads (64 B multiple) */
enum { kFtgtBlockBetaBytes = 8 }; /**< the beta table's bytes (64 B multiple) */
/** Stripe s at word kFtgtBlockStripe + 4 s: its bank's decay tables (head h at
 *  base + (h / S) * slot) and its bank's copy of the beta table. */
enum { kFtgtBlockStripe = 16 };
enum { kFtgtStripeDecayXy = 0, kFtgtStripeDecayBase = 1 };
enum { kFtgtStripeBetaXy = 2, kFtgtStripeBetaBase = 3 };
enum { kFtgtMaxStripes = 8 };
enum { kFtgtBlockWords = 48 };
enum { kFtgtBlockStride = 256 };

/* ----- one TABLE: a 16-word header, then the entries (fp32 bits) ----- */
enum { kFtgtLoPos = 0, kFtgtHiPos = 1, kFtgtLoNeg = 2, kFtgtHiNeg = 3 };
enum { kFtgtLowPos = 4, kFtgtHighPos = 5, kFtgtLowNeg = 6, kFtgtHighNeg = 7 };
enum { kFtgtNegStart = 8 }; /**< entry index of the negative side's window */
enum { kFtgtEntries = 9 };  /**< entries in all */
enum { kFtgtHeaderWords = 16 };

/** Magnitudes 0..0x7F80 are the non-NaN codes of a side (0x7F80 is inf). */
#define FTGT_LAST_MAGNITUDE 0x7F80u
/** `ftgc_table_resolve()`: the answer is a header constant, nothing to fetch. */
#define FTGT_CONSTANT 0xFFFFFFFFu

/* ----- the reader's CB_SCRATCH map in table mode (64 B aligned) ----- */
#define FTGT_SCRATCH_SLOTS 0u          /**< 16 x 64 B lookup lines (gate runs reuse it later) */
#define FTGT_SCRATCH_SLOT_COUNT 16u
#define FTGT_SCRATCH_COMPACT 1024u     /**< = the reader's COMPACT_SCRATCH_OFFSET */
#define FTGT_SCRATCH_DECAY_HEADER 1344u
#define FTGT_SCRATCH_BETA_HEADER 1408u
#define FTGT_SCRATCH_BLOCK 1472u       /**< 192 B */
#define FTGT_SCRATCH_ALPHA 2048u       /**< face rows 0-15 at +0, 16-31 at +512 */
#define FTGT_SCRATCH_BETA 3072u
#define FTGT_SCRATCH_BYTES 4096u

/** bf16 tile bytes, face bytes. */
#define FTGT_TILE_BYTES 2048u
#define FTGT_FACE_BYTES 512u

/**
 * Where `code` lands in a table. @return the entry index, or FTGT_CONSTANT with `*value` set.
 */
FTGC_INLINE uint32_t ftgc_table_resolve(const volatile uint32_t *header, uint32_t code,
                                        uint32_t *value)
{
    const uint32_t side = (code >> 15) & 1u;
    const uint32_t m = code & 0x7FFFu;
    const uint32_t lo = header[kFtgtLoPos + 2u * side];
    const uint32_t hi = header[kFtgtHiPos + 2u * side];

    if (m < lo) {
        *value = header[kFtgtLowPos + 2u * side];
        return FTGT_CONSTANT;
    }
    if (m >= hi) {
        *value = header[kFtgtHighPos + 2u * side];
        return FTGT_CONSTANT;
    }
    return (side ? header[kFtgtNegStart] : 0u) + (m - lo);
}

/** Byte offset of entry `index` from its table's start. */
FTGC_INLINE uint32_t ftgc_table_entry_offset(uint32_t index)
{
    return (kFtgtHeaderWords + index) * 4u;
}

/** The two device operations the front end needs. `l1` is an L1 address; on the card the
 *  pointer is the address itself, the host test passes its emulated L1's base as `bias`.
 *  THE CARD DEFINES FTGT_READ / FTGT_BARRIER as direct NoC calls before including this file:
 *  a function pointer in .rodata keeps its LINK address, which relocated kernel text
 *  (RELOCATE_KERNEL_TEXT) does not fix up. The host keeps the pointers. */
#ifndef FTGT_READ
#define FTGT_READ(io, xy, address, l1, bytes) (io)->read((io)->user, xy, address, l1, bytes)
#define FTGT_BARRIER(io) (io)->barrier((io)->user)
#endif

typedef struct ftgc_table_io {
    void *user;
    uintptr_t bias;
    void (*read)(void *user, uint32_t xy, uint32_t address, uint32_t l1, uint32_t bytes);
    void (*barrier)(void *user);
} ftgc_table_io;

/** Per-descriptor state: which face pair is staged (0 = none). */
typedef struct ftgc_table_state {
    uint32_t staged;
} ftgc_table_state;

FTGC_INLINE volatile uint32_t *ftgc_table_words(const ftgc_table_io *io, uint32_t l1)
{
    return (volatile uint32_t *)(io->bias + l1);
}

/**
 * Descriptor start for heads from `head_first`: the layer's table block, then the beta
 * table's header from the copy in stripe head_first % S. Two round trips; nothing staged yet.
 * @return 0 for a malformed block (the reader aborts), else 1.
 */
static inline uint32_t ftgc_table_begin(const ftgc_table_io *io, ftgc_table_state *state,
                                        uint32_t scratch, uint32_t block_xy,
                                        uint32_t block_address, uint32_t head_first)
{
    const volatile uint32_t *block = ftgc_table_words(io, scratch + FTGT_SCRATCH_BLOCK);
    uint32_t stripes, slot, r;

    FTGT_READ(io, block_xy, block_address, scratch + FTGT_SCRATCH_BLOCK,
             kFtgtBlockWords * 4u);
    FTGT_BARRIER(io);
    stripes = block[kFtgtBlockStripes];
    slot = block[kFtgtBlockDecaySlot];
    state->staged = 0u;
    if (stripes == 0u || stripes > kFtgtMaxStripes || slot == 0u || (slot % 64u) != 0u)
        return 0u;
    r = kFtgtBlockStripe + 4u * (head_first % stripes);
    FTGT_READ(io, block[r + kFtgtStripeBetaXy], block[r + kFtgtStripeBetaBase],
             scratch + FTGT_SCRATCH_BETA_HEADER, kFtgtHeaderWords * 4u);
    FTGT_BARRIER(io);
    return 1u;
}

/**
 * Build head `head`'s compact header + run at `scratch + FTGT_SCRATCH_COMPACT` for the
 * sub-chunk [first_row, first_row + chunk_rows) of band `band`: exactly the words
 * gdnc_compact_stage() writes from host gates (header word 0 the scale, run[row] the decay of
 * strip row first_row + row, run[32 + row] its beta, every other word zero).
 */
static inline void ftgc_table_head(const ftgc_table_io *io, ftgc_table_state *state,
                                   uint32_t scratch, uint32_t head, uint32_t band,
                                   uint32_t first_row, uint32_t chunk_rows)
{
    const volatile uint32_t *block = ftgc_table_words(io, scratch + FTGT_SCRATCH_BLOCK);
    volatile uint32_t *compact = ftgc_table_words(io, scratch + FTGT_SCRATCH_COMPACT);
    volatile uint32_t *run = compact + FTGC_COMPACT_HEADER_WORDS;
    const volatile uint16_t *alpha_stage =
        (const volatile uint16_t *)(io->bias + scratch + FTGT_SCRATCH_ALPHA);
    const volatile uint16_t *beta_stage =
        (const volatile uint16_t *)(io->bias + scratch + FTGT_SCRATCH_BETA);
    const uint32_t tile = head / 32u, column = head % 32u, half = column / 16u;
    const uint32_t key = tile * 2u + half + 1u;
    const uint32_t band_offset = band * block[kFtgtBlockBandBytes];
    const uint32_t stripes = block[kFtgtBlockStripes];
    const uint32_t slot = block[kFtgtBlockDecaySlot];
    const uint32_t stripe = kFtgtBlockStripe + 4u * (head % stripes);
    const uint32_t decay_xy = block[stripe + kFtgtStripeDecayXy];
    const uint32_t decay_base = block[stripe + kFtgtStripeDecayBase] + (head / stripes) * slot;
    const uint32_t beta_xy = block[stripe + kFtgtStripeBetaXy];
    const uint32_t beta_base = block[stripe + kFtgtStripeBetaBase];
    const volatile uint32_t *decay_header =
        ftgc_table_words(io, scratch + FTGT_SCRATCH_DECAY_HEADER);
    const volatile uint32_t *beta_header =
        ftgc_table_words(io, scratch + FTGT_SCRATCH_BETA_HEADER);
    uint32_t slot_word[FTGT_SCRATCH_SLOT_COUNT];
    uint32_t slot_dest[FTGT_SCRATCH_SLOT_COUNT];
    uint32_t pending = 0u, row, word;

    if (state->staged != key) {
        for (uint32_t rows = 0; rows < 2u; rows++) {
            /* Face rows 0-15 (rows = 0) or 16-31 (rows = 1), only those the sub-chunk owns. */
            const uint32_t face_offset =
                band_offset + tile * FTGT_TILE_BYTES + (rows * 2u + half) * FTGT_FACE_BYTES;

            if ((rows == 0u && first_row >= 16u) || (rows == 1u && first_row + chunk_rows <= 16u))
                continue;
            FTGT_READ(io, block[kFtgtBlockAlphaXy], block[kFtgtBlockAlphaBase] + face_offset,
                     scratch + FTGT_SCRATCH_ALPHA + rows * FTGT_FACE_BYTES, FTGT_FACE_BYTES);
            FTGT_READ(io, block[kFtgtBlockBetaXy], block[kFtgtBlockBetaBase] + face_offset,
                     scratch + FTGT_SCRATCH_BETA + rows * FTGT_FACE_BYTES, FTGT_FACE_BYTES);
        }
        state->staged = key;
    }
    FTGT_READ(io, decay_xy, decay_base, scratch + FTGT_SCRATCH_DECAY_HEADER,
             kFtgtHeaderWords * 4u);
    FTGT_BARRIER(io);

    for (word = 0; word < FTGC_COMPACT_HEADER_WORDS + FTGC_COMPACT_HEAD_WORDS; word++)
        compact[word] = 0u;
    compact[0] = block[kFtgtBlockScale];

    for (row = 0; row < chunk_rows; row++) {
        const uint32_t strip_row = first_row + row;
        const uint32_t at = (strip_row / 16u) * 256u + (strip_row % 16u) * 16u + (column % 16u);

        for (uint32_t which = 0; which < 2u; which++) {
            const uint32_t code = which ? beta_stage[at] : alpha_stage[at];
            const uint32_t dest = which * 32u + row;
            uint32_t value = 0u;
            const uint32_t index =
                ftgc_table_resolve(which ? beta_header : decay_header, code, &value);

            if (index == FTGT_CONSTANT) {
                run[dest] = value;
                continue;
            }
            {
                const uint32_t address =
                    (which ? beta_base : decay_base) + ftgc_table_entry_offset(index);

                FTGT_READ(io, which ? beta_xy : decay_xy, address & ~63u,
                         scratch + FTGT_SCRATCH_SLOTS + pending * 64u, 64u);
                slot_word[pending] = pending * 16u + (address & 63u) / 4u;
                slot_dest[pending] = dest;
                pending++;
            }
            if (pending == FTGT_SCRATCH_SLOT_COUNT) {
                const volatile uint32_t *slots =
                    ftgc_table_words(io, scratch + FTGT_SCRATCH_SLOTS);

                FTGT_BARRIER(io);
                for (uint32_t k = 0; k < pending; k++)
                    run[slot_dest[k]] = slots[slot_word[k]];
                pending = 0u;
            }
        }
    }
    if (pending != 0u) {
        const volatile uint32_t *slots = ftgc_table_words(io, scratch + FTGT_SCRATCH_SLOTS);

        FTGT_BARRIER(io);
        for (uint32_t k = 0; k < pending; k++)
            run[slot_dest[k]] = slots[slot_word[k]];
    }
}

/* ----- DECODE: the gdn family's DELTA reader (kGdnFlagTable) builds its gate block ----- */

/** The decode front end's scratch (64 B aligned, free L1 before the head loop). */
#define FTGT_ROWS_BLOCK 0u          /**< 192 B */
#define FTGT_ROWS_BETA_HEADER 256u
#define FTGT_ROWS_HEADERS 512u      /**< a decay header a head, 64 B each */
#define FTGT_ROWS_MAX_HEADS 56u
#define FTGT_ROWS_FACES 4096u       /**< (key * 2 + row group) * 1 KiB: alpha face, beta face */
#define FTGT_ROWS_SLOTS 12288u      /**< 64 x 64 B lookup lines */
#define FTGT_ROWS_SLOT_COUNT 64u
#define FTGT_ROWS_BYTES 16384u

/**
 * Rows [row_first, row_first + row_count) of band 0 of the strips, heads [head_first,
 * head_first + head_count), into the decode gate layout: for row r (0-based in the range)
 * `gate[r * stride / 4 + h]` = decay, `gate[r * stride / 4 + v_heads + h]` = beta, fp32 bits
 * — the host's gates, exactly. Striped DRAM lookups in three round trips (the block; the
 * headers and faces; the entries, 64 at a time). Other heads' words are not written.
 * `scratch` holds FTGT_ROWS_BYTES. @return 0 for a malformed block or shape (the reader
 * aborts), else 1.
 */
static inline uint32_t ftgc_table_rows(const ftgc_table_io *io, uint32_t scratch,
                                       uint32_t block_xy, uint32_t block_address,
                                       uint32_t head_first, uint32_t head_count,
                                       uint32_t row_first, uint32_t row_count, uint32_t gate,
                                       uint32_t stride, uint32_t v_heads)
{
    const volatile uint32_t *block = ftgc_table_words(io, scratch + FTGT_ROWS_BLOCK);
    volatile uint32_t *out = ftgc_table_words(io, gate);
    uint32_t slot_word[FTGT_ROWS_SLOT_COUNT], slot_dest[FTGT_ROWS_SLOT_COUNT];
    uint32_t stripes, slot, pending = 0u;

    FTGT_READ(io, block_xy, block_address, scratch + FTGT_ROWS_BLOCK, kFtgtBlockWords * 4u);
    FTGT_BARRIER(io);
    stripes = block[kFtgtBlockStripes];
    slot = block[kFtgtBlockDecaySlot];
    if (stripes == 0u || stripes > kFtgtMaxStripes || (slot % 64u) != 0u || head_count == 0u ||
        head_count > FTGT_ROWS_MAX_HEADS || head_first + head_count > 64u || row_count == 0u ||
        row_first + row_count > 32u)
        return 0u;
    {
        const uint32_t w = kFtgtBlockStripe + 4u * (head_first % stripes);

        FTGT_READ(io, block[w + kFtgtStripeBetaXy], block[w + kFtgtStripeBetaBase],
                 scratch + FTGT_ROWS_BETA_HEADER, kFtgtHeaderWords * 4u);
    }
    for (uint32_t i = 0; i < head_count; i++) {
        const uint32_t head = head_first + i, w = kFtgtBlockStripe + 4u * (head % stripes);

        FTGT_READ(io, block[w + kFtgtStripeDecayXy],
                 block[w + kFtgtStripeDecayBase] + (head / stripes) * slot,
                 scratch + FTGT_ROWS_HEADERS + i * 64u, kFtgtHeaderWords * 4u);
    }
    /* Faces: key = tile * 2 + half of every head owned, row group of every row asked. */
    for (uint32_t key = (head_first / 16u); key <= (head_first + head_count - 1u) / 16u; key++)
        for (uint32_t group = row_first / 16u; group <= (row_first + row_count - 1u) / 16u;
             group++) {
            const uint32_t face = (key / 2u) * FTGT_TILE_BYTES +
                                  (group * 2u + key % 2u) * FTGT_FACE_BYTES;
            const uint32_t l1 = scratch + FTGT_ROWS_FACES + (key * 2u + group) * 1024u;

            FTGT_READ(io, block[kFtgtBlockAlphaXy], block[kFtgtBlockAlphaBase] + face, l1,
                     FTGT_FACE_BYTES);
            FTGT_READ(io, block[kFtgtBlockBetaXy], block[kFtgtBlockBetaBase] + face,
                     l1 + FTGT_FACE_BYTES, FTGT_FACE_BYTES);
        }
    FTGT_BARRIER(io);

    /* One flat walk — head, row, decay then beta — so the batch flush exists once. */
    for (uint32_t i = 0, r = 0, which = 0, n = 2u * head_count * row_count; n != 0u; n--) {
        const uint32_t head = head_first + i, w = kFtgtBlockStripe + 4u * (head % stripes);
        const uint32_t row = row_first + r;
        const volatile uint16_t *codes = (const volatile uint16_t *)(
            io->bias + scratch + FTGT_ROWS_FACES + ((head / 16u) * 2u + row / 16u) * 1024u +
            which * FTGT_FACE_BYTES);
        const uint32_t dest = r * (stride / 4u) + which * v_heads + head;
        uint32_t value = 0u;
        const uint32_t index = ftgc_table_resolve(
            ftgc_table_words(io, scratch + (which ? FTGT_ROWS_BETA_HEADER
                                                  : FTGT_ROWS_HEADERS + i * 64u)),
            codes[(row % 16u) * 16u + head % 16u], &value);

        if (index == FTGT_CONSTANT) {
            out[dest] = value;
        } else {
            const uint32_t address = (which ? block[w + kFtgtStripeBetaBase]
                                            : block[w + kFtgtStripeDecayBase] +
                                                  (head / stripes) * slot) +
                                     ftgc_table_entry_offset(index);

            FTGT_READ(io, block[w + (which ? kFtgtStripeBetaXy : kFtgtStripeDecayXy)],
                     address & ~63u, scratch + FTGT_ROWS_SLOTS + pending * 64u, 64u);
            slot_word[pending] = pending * 16u + (address & 63u) / 4u;
            slot_dest[pending++] = dest;
        }
        if (pending != 0u && (pending == FTGT_ROWS_SLOT_COUNT || n == 1u)) {
            const volatile uint32_t *slots = ftgc_table_words(io, scratch + FTGT_ROWS_SLOTS);

            FTGT_BARRIER(io);
            for (uint32_t k = 0; k < pending; k++)
                out[slot_dest[k]] = slots[slot_word[k]];
            pending = 0u;
        }
        if (++which == 2u) {
            which = 0u;
            if (++r == row_count) {
                r = 0u;
                i++;
            }
        }
    }
    return 1u;
}

/* ----- the HOST's builder (never compiled into a kernel: nothing on the card calls it) ----- */

/** Codes a table covers: every bf16 bit pattern. */
#define FTGT_CODES 65536u

/** One side's window: [lo, hi) of magnitudes, constant below and above. */
static inline void ftgc_table_side(const uint32_t *values, uint32_t side, uint32_t *lo,
                                   uint32_t *hi)
{
    const uint32_t base = side << 15;
    const uint32_t low = values[base];
    const uint32_t high = values[base | FTGT_LAST_MAGNITUDE];
    uint32_t first = FTGT_LAST_MAGNITUDE + 1u, end = 0u, m;

    for (m = 0; m <= FTGT_LAST_MAGNITUDE; m++)
        if (values[base | m] != low) {
            first = m;
            break;
        }
    for (m = FTGT_LAST_MAGNITUDE + 1u; m-- > 0u;)
        if (values[base | m] != high) {
            end = m + 1u;
            break;
        }
    /* first > end: every magnitude is LOW below `end` and HIGH from it (values in [end, first)
     * equal both), so the window is empty at `end`. */
    if (first > end)
        first = end;
    *lo = first;
    *hi = end;
}

/**
 * Build one table from `values[code]` — the host gate's fp32 bits for every bf16 code — and
 * PROVE it: every non-NaN code is resolved through the built table (ftgc_table_resolve, the
 * reader's own) and compared. NaN codes are outside the claim (see the file note).
 *
 * @param out  NULL to size only
 * @return words (header + entries), or 0 when `out_words` is too small or the proof failed
 */
static inline uint32_t ftgc_table_build(const uint32_t *values, uint32_t *out, uint32_t out_words)
{
    uint32_t lo[2], hi[2], side, m, words, word;

    if (!values)
        return 0u;
    ftgc_table_side(values, 0u, &lo[0], &hi[0]);
    ftgc_table_side(values, 1u, &lo[1], &hi[1]);
    words = kFtgtHeaderWords + (hi[0] - lo[0]) + (hi[1] - lo[1]);
    if (!out)
        return words;
    if (out_words < words)
        return 0u;
    for (word = 0; word < kFtgtHeaderWords; word++)
        out[word] = 0u;
    out[kFtgtLoPos] = lo[0];
    out[kFtgtHiPos] = hi[0];
    out[kFtgtLoNeg] = lo[1];
    out[kFtgtHiNeg] = hi[1];
    out[kFtgtLowPos] = values[0];
    out[kFtgtHighPos] = values[FTGT_LAST_MAGNITUDE];
    out[kFtgtLowNeg] = values[0x8000u];
    out[kFtgtHighNeg] = values[0x8000u | FTGT_LAST_MAGNITUDE];
    out[kFtgtNegStart] = hi[0] - lo[0];
    out[kFtgtEntries] = words - kFtgtHeaderWords;
    for (side = 0; side < 2u; side++)
        for (m = lo[side]; m < hi[side]; m++)
            out[kFtgtHeaderWords + (side ? out[kFtgtNegStart] : 0u) + (m - lo[side])] =
                values[(side << 15) | m];
    for (side = 0; side < 2u; side++)
        for (m = 0; m <= FTGT_LAST_MAGNITUDE; m++) {
            const uint32_t code = (side << 15) | m;
            uint32_t value = 0u;
            const uint32_t index = ftgc_table_resolve(out, code, &value);

            if (index != FTGT_CONSTANT)
                value = out[kFtgtHeaderWords + index];
            if (value != values[code])
                return 0u;
        }
    return words;
}

#endif /* FTGC_TABLE_H */
