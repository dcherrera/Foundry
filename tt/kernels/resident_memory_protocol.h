// SPDX-License-Identifier: AGPL-3.0-or-later
//
// resident_memory_protocol.h
//   The parameter block and the face arithmetic of the MEMORY op family — EMBEDDING and
//   KV_APPEND — shared by resident_gather_reader.c and resident_place_writer.c.
//   The host half is ft_tt_op_memory.h; the two must agree word for word and
//   tests/test_memory_ops.c pins the pair off-hardware.
//
//   **The KV cache layout contract lives in ft_tt_op_memory.h PART 1.** Read it there;
//   it is not repeated here, because a contract with two copies has two versions.
//
// ===========================================================================
// ONE OPERATION, TWO OPCODES
// ===========================================================================
// Both ops are:
//
//   place ROW_COUNT rows into a face-ordered tile destination starting at row DST_ROW0,
//   for GROUP_COUNT independent destination groups of GROUP_TILES tiles each.
//
// EMBEDDING gathers its rows out of a ROW-MAJOR table by token id and owns the whole
// destination tile row; KV_APPEND takes its rows out of a TILED projection output, splits
// them across one group per KV head, and must PRESERVE every destination row it does not
// write. Two flag bits, one binary.
//
// ===========================================================================
// WHY A PARTIAL ROW IS WRITABLE AT ALL — the arithmetic the whole family rests on
// ===========================================================================
// A tile is four 16x16 faces in face order. One ROW of a tile is therefore not contiguous:
// it is TWO face rows, one per face column-half, `tile_bytes/4` apart. Each is 16 elements
// = `tile_bytes/64` bytes — 32 B for bf16.
//
// 32 bytes is a legal DRAM WRITE (the NoC's write alignment is 16 B; only the READ
// direction needs 64 B — ft_tt_noc.c, ft_tt_noc_alignment()). That single asymmetry is
// what makes a one-position KV append a copy instead of a read-modify-write, and it is
// MEASURED by tests/test_phase6_memory.c stage 1 rather than assumed: the test writes one
// row into a poisoned destination and checks BOTH that the row landed byte-exact AND that
// its neighbours are untouched. If a future part refuses a 32-byte DRAM write, this family
// needs a read-modify-write path and the KV layout does not change.
//
// Consecutive rows MERGE: rows [r, r+n) of one face half are `n * chunk` contiguous bytes,
// so a 32-row block is one write of a whole tile and a whole group of tiles is one write of
// `group_tiles * tile_bytes`. The single-row case and the prefill case are the same code.
//
// ===========================================================================
// NO COMPUTE THREAD, AND WHY THAT IS SAFE RATHER THAN JUST SMALL
// ===========================================================================
// `enables` is DM0|DM1. The hazard that would make that wrong is TRISC0's
// stock sync init, which zeroes all 64 CB page-counter pairs and would break a
// BRISC<->NCRISC handshake underneath it. It runs ONLY when BRISC writes
// RUN_SYNC_MSG_INIT_SYNC_REGISTERS into trisc0's sync slot (at boot, and
// after a kernel RETURNS). A resident BRISC never returns, so the only
// zeroing happens at firmware boot, before the queue is armed.
//
#ifndef FOUNDRY_RESIDENT_MEMORY_PROTOCOL_H
#define FOUNDRY_RESIDENT_MEMORY_PROTOCOL_H

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
// The sixteen words. Mirrors ft_tt_op_memory.h -- change both or neither.
// ---------------------------------------------------------------------------

enum { kMemFlags = 0 };
enum { kMemRowCount = 1 };
enum { kMemGroupCount = 2 };
enum { kMemGroupTiles = 3 };
enum { kMemSrcX = 4 };
enum { kMemSrcY = 5 };
/** ROW-MAJOR: base of table row 0. TILED: base of the ONE source TILE ROW this descriptor
 *  reads — the host folds (source_row0 / 32) * source_tile_row_bytes in, because the split
 *  rule already guarantees one tile row per descriptor. */
enum { kMemSrcBase = 6 };
/** ROW-MAJOR only: bytes between table rows. Unread when the source is tiled. */
enum { kMemSrcRowStride = 7 };
/** TILED: the first source row WITHIN its tile row, 0..31. ROW-MAJOR without GATHER: the
 *  first table row index. GATHER: unread — the index array names every row. */
enum { kMemSrcRow0 = 8 };
enum { kMemSrcGroupStride = 9 };
enum { kMemDstX = 10 };
enum { kMemDstY = 11 };
enum { kMemDstBase = 12 };
/** Bytes each destination group OWNS: the stride when there are several, and the EXTENT
 *  when there is one. Both readings are the same number whenever the groups are contiguous,
 *  which they always are here — a KV region is `head_count` head strides end to end and an
 *  embedding output is one group. It is the only field that knows how big the destination
 *  is, and `destination_row0` is an ABSOLUTE position whose tile row multiplies into a raw
 *  byte offset, so without it a position past the cache writes into whatever DRAM follows,
 *  at full length, with nothing faulting. 0 is a REFUSAL. */
enum { kMemDstGroupStride = 13 };
/** ABSOLUTE destination row — the KV position. Its tile row is DST_ROW0 / 32 and its row
 *  within that tile is DST_ROW0 % 32. */
enum { kMemDstRow0 = 14 };
/** ROW-MAJOR sources only: how many rows the source table HAS. The gather index and the
 *  sequential row index are both raw wire values that become a DRAM address as
 *  `source_base + index * source_row_stride`, and nothing else in this block bounds them —
 *  so without this word a wrong index is an arbitrary DRAM read at full length rather than
 *  an error. 0 is a REFUSAL in row-major mode, never "unchecked": a bound a zero turns off
 *  is a bound a zeroed descriptor turns off. Unread when the source is tiled.
 *
 *  IT TAKES THE LAST SPARE WORD, and the destination bound had to be derived rather than
 *  given one of its own: the L1 parameter slot is FT_TT_QUEUE_PARAM_WORDS = 16 for EVERY op
 *  family, so a seventeenth word is a change to the shared slot stride and to all six
 *  families, not to this one. kMemDstGroupStride is what carries the destination bound. */
enum { kMemSrcRowLimit = 15 };
enum { kMemWords = 16 };

// Written as literals, not as shifts: tests/test_memory_ops.c pins these by PARSING this
// file, and a parser that had to evaluate `1u << 2` would be a second thing to get wrong.
enum { kMemFlagSrcTiled = 0x1 };
enum { kMemFlagDstWholeTiles = 0x2 };
enum { kMemFlagGather = 0x4 };
/** BANKED destination (KV_APPEND): bank group b of the groups — `heads_per_bank` each, bits
 *  kMemBankHeadsShift.. of FLAGS — lands at (byte b of DST_X, byte b of DST_Y, DST_BASE).
 *  One descriptor then writes heads that live in several DRAM banks at one common address
 *  (Pantheon's KV bank groups) instead of one descriptor per bank. */
enum { kMemFlagDstBanked = 0x8 };
/** KV_APPEND: the cache element format (kKvFormat*, ft_kv_block_float.h) in bits 4..5; non-zero
 *  means the writer packs each appended face row to block float. */
enum { kMemKvFormatShift = 4 };
enum { kMemKvFormatMask = 0x3 };
/** Writer-private CB: two packed-tile slots for the block-float read-modify-write. */
enum { kMemCbKvPack = 2 };
enum { kMemBankHeadsShift = 8 };
enum { kMemMaxBanks = 4 };

/** One descriptor stays inside one source tile row and one destination tile row. The host
 *  splits; the kernel REFUSES rather than looping, because a wrap case inside face
 *  arithmetic is invisible in an output that is nearly right. */
enum { kMemMaxRows = 32 };

/**
 * A NoC transfer length that may actually be issued.
 *
 * **A ZERO-LENGTH NoC READ IS SUSPECTED OF TAKING THE WHOLE CARD DOWN.** On 2026-08-12 the
 * RoPE round's card-down happened at exactly the first descriptor that reached a branch
 * computing a length of zero; every earlier stage had passed and the next access returned
 * 0xFFFFFFFF everywhere. INFERRED, and deliberately NOT re-confirmed — confirming it costs
 * the card for every worker on it.
 *
 * This family is the one most exposed to it: both ops are gathers built out of computed
 * lengths, and a token count of zero, a collapsed range, or an underflowed `count - start`
 * all produce a zero-length read by arithmetic rather than by a typo. So EVERY transfer
 * here is guarded at the point of issue, in the KERNEL and not only on the host, and a zero
 * length is a REFUSAL (kErrBadParameters) rather than a no-op — a no-op would advance the
 * descriptor and publish completion for work nobody did.
 */
FTQ_INLINE bool ftm_length_is_issuable(uint32_t bytes)
{
    return bytes != 0u;
}

/** Alignment a DRAM READ needs in address AND length (`ft_tt_noc_alignment()` host-side).
 *  Spelled here as well as in ft_tensix_transfer_guard.h because the COMPUTE images include
 *  this header and must not be made to include a data-movement one. */
enum { kMemDramReadAlignment = 64 };

/**
 * A ROW-MAJOR source row this reader may actually fetch.
 *
 * TWO SEPARATE DEFECTS, and the second one is invisible half the time. The index arrives on
 * the wire — for a GATHER it is a word the host wrote into CB_INDEX — and becomes a raw DRAM
 * address as `base + index * stride`, so an unbounded index is an arbitrary DRAM read.
 * And the STRIDE is never required to be 64-byte aligned, which is what a DRAM read needs in
 * address as well as length: a 1500-wide bf16 table has a 3000-byte row stride, so
 * `index * 3000` is 64-aligned for even indices and ILLEGAL for odd ones. Upstream's
 * analogue does not hope — it rounds the stride up explicitly
 * (`(page_size + 31u) & ~31u`).
 *
 * @param base        kMemSrcBase, the table's own base — checked too, because an aligned
 *                    stride off an unaligned base is still an unaligned read
 * @param stride      kMemSrcRowStride
 * @param row_limit   kMemSrcRowLimit; 0 refuses
 * @param index       the row this descriptor wants
 * @param row_bytes   bytes one row read moves
 */
FTQ_INLINE bool ftm_row_major_source_is_readable(uint32_t base, uint32_t stride,
                                                 uint32_t row_limit, uint32_t index,
                                                 uint32_t row_bytes)
{
    if (row_limit == 0u || index >= row_limit)
        return false;
    if (!ftm_length_is_issuable(row_bytes) || (row_bytes % kMemDramReadAlignment) != 0u)
        return false;
    if ((base % kMemDramReadAlignment) != 0u || (stride % kMemDramReadAlignment) != 0u)
        return false;
    /* The address must not wrap, or the bound above proves nothing about where it lands. */
    if (stride != 0u && index > 0xFFFFFFFFu / stride)
        return false;
    return base <= 0xFFFFFFFFu - index * stride;
}

/**
 * A destination byte range this writer may actually put a group on, INSIDE THE GROUP'S OWN
 * BYTES.
 *
 * `destination_row0` is an ABSOLUTE position, so `tile_row * group_tiles * tile_bytes` is an
 * unbounded byte offset added to a raw base — past the end of the cache it wrote into
 * whatever DRAM follows, at full length, with nothing faulting. The bound is
 * kMemDstGroupStride, the bytes each group owns: for a KV region that is the head stride,
 * so this refuses a position past `capacity_positions` exactly; for a one-group embedding it
 * is the whole output. Checking the group rather than the region is the TIGHTER of the two
 * statements — it also stops head h's append landing in head h+1.
 *
 * @param group_bytes_owned  kMemDstGroupStride; 0 refuses
 * @param offset             bytes into this group the write starts at
 * @param span_bytes         bytes this group writes
 */
FTQ_INLINE bool ftm_destination_span_fits(uint32_t group_bytes_owned, uint32_t offset,
                                          uint32_t span_bytes)
{
    if (group_bytes_owned == 0u || !ftm_length_is_issuable(span_bytes))
        return false;
    if (offset > group_bytes_owned)
        return false;
    return span_bytes <= group_bytes_owned - offset;
}

/**
 * A BANKED destination's group `group`: its bank's NoC (x, y) and its slot inside that bank.
 *
 * @param flags      kMemFlags (heads per bank in bits kMemBankHeadsShift..+7)
 * @param packed_x   kMemDstX: byte b = bank group b's x
 * @param packed_y   kMemDstY: byte b = bank group b's y
 * @param group      the group (KV head) being written
 * @return false on a zero heads-per-bank or a bank group past kMemMaxBanks — a refusal,
 *         never a write to coordinate 0
 */
FTQ_INLINE bool ftm_banked_destination(uint32_t flags, uint32_t packed_x, uint32_t packed_y,
                                       uint32_t group, uint32_t* out_x, uint32_t* out_y,
                                       uint32_t* out_slot)
{
    const uint32_t per = (flags >> kMemBankHeadsShift) & 0xFFu;
    uint32_t bank;

    if (per == 0u)
        return false;
    bank = group / per;
    if (bank >= (uint32_t)kMemMaxBanks)
        return false;
    *out_x = (packed_x >> (8u * bank)) & 0xFFu;
    *out_y = (packed_y >> (8u * bank)) & 0xFFu;
    *out_slot = group % per;
    return true;
}

enum { kMemTileRows = 32 };
enum { kMemFaceRows = 16 };

// ---------------------------------------------------------------------------
// Face arithmetic
// ---------------------------------------------------------------------------

/** Bytes of one face row: 16 elements of one tile row, on one side of the face column
 *  split. 32 for bf16, 64 for fp32. */
FTQ_INLINE uint32_t ftm_chunk_bytes(uint32_t tile_bytes)
{
    return tile_bytes / 64u;
}

/** Bytes of one face. */
FTQ_INLINE uint32_t ftm_face_bytes(uint32_t tile_bytes)
{
    return tile_bytes / 4u;
}

/**
 * Byte offset, inside a tile, of the face row holding tile row `row` and columns
 * [16*half, 16*half+16).
 *
 * This is ft_tt_tile_face_index(row, 16*half) * element_bytes with the multiply folded
 * away. tests/test_memory_ops.c checks it against ft_tt_tilize.c for all 64 (row, half)
 * pairs rather than trusting the rewrite — a face-order mistake is invisible to any metric
 * computed through this same code (docs/tt_kernel_model.md §9).
 */
FTQ_INLINE uint32_t ftm_chunk_offset(uint32_t row, uint32_t half, uint32_t tile_bytes)
{
    return ((row / kMemFaceRows) * 2u + half) * ftm_face_bytes(tile_bytes) +
           (row % kMemFaceRows) * ftm_chunk_bytes(tile_bytes);
}

/** Bytes one row of ONE group occupies in a ROW-MAJOR source: `group_tiles` tiles' worth
 *  of one row, i.e. two face rows per tile. Always a multiple of 64 for bf16 and fp32
 *  alike, which is what keeps a row read legal against the 64-byte DRAM read alignment. */
FTQ_INLINE uint32_t ftm_group_row_bytes(uint32_t group_tiles, uint32_t tile_bytes)
{
    return group_tiles * ftm_chunk_bytes(tile_bytes) * 2u;
}

/** A word copy inside L1. L1 is ordinary memory to the RISC-V and its data cache is
 *  disabled (ft_tt_queue.h, THE ORDERING ASSUMPTION), so this is a plain loop; it is the
 *  cheapest way to move a 32-byte face row, an order below the ~50 ns a NoC transaction
 *  costs to issue. Both ends are 4-byte aligned by construction: a chunk offset is a
 *  multiple of `chunk_bytes`, itself a multiple of 4. */
FTQ_INLINE void ftm_copy_words(uint32_t* destination, const uint32_t* source, uint32_t words)
{
    for (uint32_t index = 0; index < words; index++)
        destination[index] = source[index];
}

FTQ_INLINE void ftm_zero_words(uint32_t* destination, uint32_t words)
{
    for (uint32_t index = 0; index < words; index++)
        destination[index] = 0u;
}

/**
 * Place ONE row into a group of `group_tiles` face-ordered destination tiles.
 *
 * `source_row` is a pointer to `ftm_group_row_bytes()` bytes laid out ROW-MAJOR: face row
 * (tile t, half h) at offset (t*2 + h) * chunk_bytes. That is the natural element order of
 * a row-major table row, which is why an untilized embedding table needs no permutation
 * beyond this scatter.
 */
FTQ_INLINE void ftm_place_row_from_row_major(uint32_t* destination_block,
                                             const uint32_t* source_row,
                                         uint32_t destination_row, uint32_t group_tiles,
                                         uint32_t tile_bytes)
{
    const uint32_t chunk = ftm_chunk_bytes(tile_bytes);
    const uint32_t words = chunk / 4u;

    for (uint32_t tile = 0; tile < group_tiles; tile++) {
        for (uint32_t half = 0; half < 2u; half++) {
            const uint32_t destination_offset =
                tile * tile_bytes + ftm_chunk_offset(destination_row, half, tile_bytes);
            const uint32_t source_offset = (tile * 2u + half) * chunk;

            ftm_copy_words(destination_block + destination_offset / 4u,
                           source_row + source_offset / 4u, words);
        }
    }
}

/**
 * Place ONE row of a group of face-ordered SOURCE tiles into the same-shaped destination
 * group, moving it from `source_row` to `destination_row`.
 *
 * When the two rows are equal this is the identity, which is the prefill fast path the
 * reader takes by skipping the staging buffer entirely.
 */
FTQ_INLINE void ftm_place_row_from_tiles(uint32_t* destination_block, const uint32_t* source_block,
                                     uint32_t source_row, uint32_t destination_row,
                                     uint32_t group_tiles, uint32_t tile_bytes)
{
    const uint32_t words = ftm_chunk_bytes(tile_bytes) / 4u;

    for (uint32_t tile = 0; tile < group_tiles; tile++) {
        for (uint32_t half = 0; half < 2u; half++) {
            const uint32_t destination_offset =
                tile * tile_bytes + ftm_chunk_offset(destination_row, half, tile_bytes);
            const uint32_t source_offset =
                tile * tile_bytes + ftm_chunk_offset(source_row, half, tile_bytes);

            ftm_copy_words(destination_block + destination_offset / 4u,
                           source_block + source_offset / 4u, words);
        }
    }
}

// ---------------------------------------------------------------------------
// The destination runs a partial write is made of
// ---------------------------------------------------------------------------

/** Most runs one tile's partial write needs: two face row groups x two column halves. */
enum { kMemMaxRuns = 4 };

typedef struct ftm_run {
    uint32_t offset; /**< from the tile base */
    uint32_t bytes;
} ftm_run;

/**
 * The face-row runs covering destination rows [row0, row0 + row_count) of ONE tile.
 *
 * Rows inside one face row group and one column half are CONTIGUOUS, so a range collapses
 * to at most four runs — and the caller checks `row0 == 0 && row_count == 32` first, which
 * collapses it further to the whole tile and lets a whole GROUP of tiles go out as one
 * transaction.
 *
 * @return how many runs were written, 0 for an out-of-range request
 */
FTQ_INLINE uint32_t ftm_destination_runs(uint32_t row0, uint32_t row_count, uint32_t tile_bytes,
                                     ftm_run runs[kMemMaxRuns])
{
    const uint32_t chunk = ftm_chunk_bytes(tile_bytes);
    uint32_t count = 0u;

    if (row_count == 0u || row0 + row_count > kMemTileRows)
        return 0u;

    for (uint32_t group = 0; group < 2u; group++) {
        const uint32_t group_first = group * kMemFaceRows;
        const uint32_t group_last = group_first + kMemFaceRows;
        const uint32_t first = (row0 > group_first) ? row0 : group_first;
        const uint32_t last = (row0 + row_count < group_last) ? row0 + row_count : group_last;

        if (first >= last)
            continue;
        for (uint32_t half = 0; half < 2u; half++) {
            runs[count].offset = ftm_chunk_offset(first, half, tile_bytes);
            runs[count].bytes = (last - first) * chunk;
            count++;
        }
    }
    return count;
}

#endif  // FOUNDRY_RESIDENT_MEMORY_PROTOCOL_H
