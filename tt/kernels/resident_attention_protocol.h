// SPDX-License-Identifier: AGPL-3.0-or-later
//
// resident_attention_protocol.h
//   The device half of ft_tt_op_attention.h: the sixteen parameter words the FLASH-DECODE
//   ATTENTION triad reads out of L1, the circular-buffer contract the three kernels share,
//   and the face arithmetic the reader and the writer both need. The host half is
//   ft_tt_op_attention.h; the two must agree word for word and tests/test_attention_ops.c
//   pins the pair off-hardware.
//
// ===========================================================================
// 1. WHAT ONE DESCRIPTOR IS: ONE KV HEAD'S GROUP, AT ONE DECODE POSITION
// ===========================================================================
// MiniCPM5-1B is GQA: 16 Q heads, 2 KV heads, head_dim 128 (docs/tt_phase0_decisions.md
// §3, read from the GGUF header). So each KV head serves EIGHT Q heads, and the unit of
// work here is ONE KV GROUP -- one KV head's K and V, and the eight Q heads that read it.
//
// THAT CHOICE IS THE OP'S MAIN DESIGN DECISION AND IT IS AN ARITHMETIC ONE. Decode reads
// the whole KV cache and does almost no arithmetic per byte, so the only thing that matters
// is how many times the cache is read. A core that owned ONE Q HEAD would re-read its KV
// head's cache eight times per layer per token; a core that owns the GROUP reads it once
// and serves all eight. At 8k context that is the difference between 8 MiB and 64 MiB of
// NoC traffic per layer per token.
//
// The eight Q heads become the eight ROWS of one tile (§3), so the score matmul that a
// single-head design would issue eight times is issued ONCE. That matters at the other end
// too: one HiFi4 tile matmul is 64 cycles = 80 ns at 800 MHz against a reader delivering a
// tile pair every ~104 ns (docs/tt_kernel_model.md §12, §16), so eight of them per pair
// would move this op from data-movement bound to MATH bound. Packing keeps it at one.
//
// ===========================================================================
// 2. THE ONLINE SOFTMAX IS REQUIRED, AND THE REASON IS CAPACITY
// ===========================================================================
// A standalone softmax holds its row TWICE over -- the activations to subtract the max
// from, and the exponentials to divide by the sum -- so at 2048 B a tile it tops out at
// Wt = 325 (ft_tt_op_reduce_max_tile_cols(), docs/tt_bringup_log.md §22.3). A 32k-context
// attention row is Wt = 1024 and DOES NOT FIT AT ANY TILE SIZE.
//
// So the score row is never materialised. This op walks the KV cache in BLOCKS and carries
// three running quantities across them -- the row max `m`, the row sum `l`, and the output
// accumulator `O` -- rescaling the last two whenever the max moves:
//
//     m_new = max(m_old, rowmax(S_block))
//     c     = exp(m_old - m_new)
//     l_new = c * l_old + rowsum(exp(S_block - m_new))
//     O_new = c * O_old + exp(S_block - m_new) . V_block
//     out   = O_final / l_final
//
// L1 then holds ONE BLOCK of scores, not a row, and the context length stops being a
// capacity question at all.
//
// **`dst_accum_mode 1` is not optional here and it is not a transplant.** The running sum
// and max are accumulated across EVERY block, so a 16-bit destination accumulator
// re-rounds them once per block -- docs/tt_bringup_log.md §19's failure mode multiplied by
// the block count, which at 32k context is 128 blocks rather than the 144 K-tiles that
// made a matmul's error grow 159x.
//
// ===========================================================================
// 3. THE THREE LAYOUTS, STATED COMPLETELY
// ===========================================================================
// **K and V** are ft_tt_op_memory.h PART 1's contract, unchanged and relied on: tile row =
// POSITION, tile column = HEAD DIMENSION, axis order HEAD -> POSITION-TILE -> DIM-TILE. A
// position range of one head is therefore ONE CONTIGUOUS BYTE RUN (F2), which is what lets
// a whole block be one `noc_async_read` -- the shape the measured fast path was built for.
//
// **Q and the output** are the model's ordinary activation layout: a tile grid
// `q_head_count * dim_tiles` wide, head h's dim-tile d at tile column `h*dim_tiles + d`,
// and the decode token's own row is `position % 32` (ft_tt_op_memory_position_row(), the
// §1.5b row-phase invariant this op inherits rather than re-decides).
//
// **The packed Q the compute thread consumes** is `dim_tiles` tiles in which ROW h is Q
// head `q_first + h` and the column is the head dimension. The READER builds it, by
// reading the group's `q_rows * dim_tiles` consecutive Q tiles in ONE transaction and
// copying the two 32-byte face rows of row `query_row` out of each. The WRITER performs
// the inverse scatter on the way out. Neither is a new idea: it is
// resident_place_writer.c's placed copy, and the face arithmetic below is the same
// arithmetic (tests/test_attention_ops.c asserts it against ft_tt_tilize.c rather than
// trusting the rewrite).
//
// ===========================================================================
// 4. THE TAIL, AND WHY IT IS AN ADDITIVE MASK RATHER THAN A SHORTER LOOP
// ===========================================================================
// The live length is not a multiple of 32, and the cache beyond it holds whatever the
// previous sequence left -- so it must be MASKED, not merely ignored, and masked BEFORE the
// row max. A 0/1 multiply after the exponential cannot do it: one stale score of +500 in a
// dead column would set the running max and drive every live weight to zero. So every
// position tile that is not entirely live has a tile of `0` / `-1e30` ADDED to its scores,
// selected by `kv_positions % 32`, and a tile entirely past the end gets the all-dead mask.
//
// **EVERY BLOCK THEREFORE PROCESSES EXACTLY `block_tiles` POSITION TILES**, including the
// ones past the end, and that uniformity is load-bearing rather than lazy: it is what keeps
// every circular buffer's push and pop counts identical for every descriptor, which is what
// the block-prefetch invariant (§5) requires of an op whose depth varies with the context.
// The waste is at most `2*block_tiles - 1` tiles per descriptor.
//
// ===========================================================================
// 5. THE BLOCK-PREFETCH INVARIANT, WHICH IS STRONGER THAN §12 SAYS
// ===========================================================================
// `cb_push_back` wraps the FIFO write pointer only when it lands EXACTLY on the limit. On a RESIDENT op that runs many descriptors, a descriptor that
// leaves the pointer off the next descriptor's block grid makes the next block step OVER
// the limit and walk up L1 -- MEASURED on the topk reader, where a read landed on the queue
// control block at 0x170000 and nothing faulted (resident_topk_reader.c). Two conditions
// make it impossible by induction:
//
//     pages % depth == 0        every block wrap lands exactly on the limit
//     tile_count % pages == 0   the descriptor leaves the pointer where it found it
//
// Both are checked HERE, in the kernel, as refusals -- and both are why §4's uniform block
// exists. K and V hold `2 * block_tiles * dim_tiles` pages and are pushed a whole block at
// a time; the padded position-tile count is a multiple of `2 * block_tiles`, so a
// descriptor ends where it started.
//
// ===========================================================================
// 6. NEVER ISSUE A ZERO-LENGTH NoC TRANSFER
// ===========================================================================
// A zero-length `noc_async_read` most likely took the whole card down on 2026-08-12 (dev
// notes §1: INFERRED, and deliberately not re-confirmed, because confirming it costs the
// card for every worker). Attention is full of computed lengths -- a block that comes out
// empty, a last partial block, a range that collapses -- so every length is checked against
// zero at the point of issue by fta_length_is_issuable(), and a zero is a REFUSAL rather
// than a skipped transfer.
//
#ifndef FOUNDRY_RESIDENT_ATTENTION_PROTOCOL_H
#define FOUNDRY_RESIDENT_ATTENTION_PROTOCOL_H

#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

/* The SFPI GCC fork's own pointer attribute — an access through it goes to L1 rather than
 * to ordinary memory, and dropping it changes the code that is emitted. Identical to
 * tt-metal's `tt_l1_ptr`; spelled here as well as in resident_queue_protocol.h so this
 * header does not depend on include order. */
#ifndef FTQ_L1_PTR
#define FTQ_L1_PTR __attribute__((rvtt_l1_ptr))
#endif

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

// ---------------------------------------------------------------------------
// The sixteen words. Mirrors ft_tt_op_attention.h -- change both or neither.
// ---------------------------------------------------------------------------

/** byte 0 = q_rows (DECODE: Q heads in this group, 1..32; CHUNK: live positions C, 1..32);
 *  byte 1 = query_row (position % 32; 0 in CHUNK mode); byte 2 = flags. Three small numbers
 *  in one word because the block has no room for three, and bytes rather than bit fields
 *  because a field a human reads out of a hex dump on a stuck core is worth a nibble pair. */
enum { kAttnHeads = 0 };

/** CHUNK MODE — flags bit 0. One descriptor is then ONE Q HEAD over C consecutive
 *  positions, not one KV group at one position.
 *
 *  WHY THE ROW AXIS FLIPS MEANING RATHER THAN GROWING. Decode packs the group's q_rows Q
 *  heads into the rows of one tile so the score matmul is issued once for all of them; the
 *  other 32 - q_rows rows are padding the FPU computes anyway. Prefill wants C positions of
 *  ONE head in those same rows, which is the SAME tile ops -- so a chunk descriptor costs
 *  what a decode descriptor costs and serves C positions instead of one. Keeping the GROUP
 *  as the unit instead would need q_rows*C = 128 rows, i.e. Mt = 4 in the compute thread:
 *  four running maxima, four row sums, four output accumulators, against a DST section of
 *  four tiles. That buys 4x fewer KV reads on an op this family measures as COMPUTE bound
 *  (ft_tt_op_attention.h @issues) and costs the whole compute kernel. Not taken.
 *
 *  WHAT IT CHANGES, EXHAUSTIVELY:
 *   - the READER reads `dim_tiles` Q tiles straight into CB_Q. No staging buffer, no
 *     face-row pack: in chunk geometry the activation tile ALREADY has position t in row t,
 *     which is exactly the packed Q the compute thread wants (ft_tt_op_gdnc.h's "a chunk is
 *     the same strip a single position is", applied to attention);
 *   - the WRITER writes `dim_tiles` WHOLE tiles. No scatter, same reason;
 *   - the last LIVE position tile takes a CAUSAL mask instead of a tail mask (below);
 *   - the COMPUTE thread does not change at all. It never reads q_rows or query_row.
 *
 *  WHAT IT REQUIRES OF THE CALLER: the chunk's first position is a multiple of 32, so the
 *  chunk occupies exactly the LAST live position tile and row t is position first + t. The
 *  host refuses anything else, because a chunk straddling two position tiles would need two
 *  masks and there is one mask slot. */
enum { kAttnFlagChunk = 1 };
/** S2f ROWS mode (the speculative verify): DECODE packing for `positions` consecutive strip
 *  rows at once — packed row p*q_rows + h is head h of strip row query_row + p, written back
 *  the same way. Every position's live length must end in the SAME (last) position tile; the
 *  tail slot then takes a per-row STAIRCASE mask the host writes at word 5 (on the constants'
 *  coordinate), applied to the last live tile even when the longest row fills it. */
enum { kAttnFlagRows = 2 };
enum { kAttnFlagsKnown = kAttnFlagChunk | kAttnFlagRows };

// ===========================================================================
// 7. SPLIT-K (FLASH-DECODE ACROSS CORES) -- the `_split` builds only (-DATTN_SPLITK)
// ===========================================================================
// One KV group's positions split into chunks over many cores, each chunk ONE descriptor:
//
//   PARTIAL  a decode or ROWS descriptor over ONE CHUNK. The host folds the chunk's first
//            position tile into K_BASE / V_BASE and sends the chunk-relative live length, so
//            reader and masks are unchanged. Instead of out = O / l the compute thread emits
//            the un-normalised O (dim_tiles tiles), then m (the running max) and l (the row
//            sum) -- dim_tiles + 2 WHOLE tiles, row p*q_rows + h, the writer writes them to
//            OUT unscattered. A non-final chunk is entirely live: decode needs no mask
//            (tail 0), ROWS points word 5 at an all-zero tile.
//   MERGE    the group's P partials (contiguous at Q_BASE, dim_tiles + 2 tiles apiece, word 2
//            = P) folded IN ORDER, each as one more block of the online softmax: m' =
//            max(m, m_i); c_i = e^(m_i - m'); then the block merge the one-core loop runs,
//            l = e^(m - m') l + c_i l_i, O = e^(m - m') O + c_i O_i; finally out = O / l.
//            K, V and the constants are unread. The writer scatters it as a decode / ROWS
//            output. One merge core per group: the host awaits every partial first, so no
//            core waits on another (a tree would need on-card completion between cores).
//
// Every intermediate is packed from the fp32 DST into a Float16_b CB -- the same rounding
// points the one-core online softmax already has between blocks. A Float32 CB cannot be
// unpacked at fp32 in a program whose K / V CBs are block float (gdn_float32.cbtable,
// measured), so the partials are bf16 tiles, not fp32.
//
// The shipped programs never see these flags: they refuse them as unknown (kAttnFlagsKnown).
enum { kAttnFlagPartial = 4 };
enum { kAttnFlagMerge = 8 };
enum { kAttnFlagsKnownSplit = kAttnFlagsKnown | kAttnFlagPartial | kAttnFlagMerge };
/** Tiles of one partial: dim_tiles of O, then the max tile, then the sum tile. */
enum { kAttnPartialStatTiles = 2 };
/** The most partials one MERGE descriptor folds (the reader's and compute's trip count). */
enum { kAttnMaxPartials = 64 };

/** head_dim / 32. Four for MiniCPM5-1B. */
enum { kAttnDimTiles = 1 };

/** LIVE positions in the cache for this sequence, i.e. how many the query attends to.
 *  The live position-tile count and the tail width are DERIVED from it and never sent
 *  beside it -- two fields that must agree are two fields that can disagree. */
enum { kAttnKvPositions = 2 };

/** Position tiles per K/V block. Fixed by the RESIDENCY, because it fixes the circular
 *  buffers' page counts; the kernel refuses a descriptor whose value disagrees with the
 *  buffers it was staged with. */
enum { kAttnBlockTiles = 3 };

/** Position tiles the cache region actually holds. The padded read window may exceed the
 *  live length by up to 2*block_tiles - 1 tiles (§4), and reading past the region is the
 *  one thing that turns a masked-away tile into an out-of-bounds NoC read. */
enum { kAttnCapacityTiles = 4 };

/** Written 0 and unread, except in ROWS mode: the byte address of the staircase mask tile. */
enum { kAttnReserved = 5 };

/** Every operand's NoC coordinate is ONE word, `(y << 16) | x`. The vector and matmul
 *  families spend two words on a coordinate; this op has FIVE operands and sixteen words,
 *  so it packs. The pack is symmetric on both sides of the wire and pinned by the
 *  off-hardware suite. */
enum { kAttnQCoord = 6 };
/** Byte address of THIS GROUP's first Q tile: head `q_first`, dim tile 0, of the row block
 *  the decode token sits in. The host folds the head offset and any row-block offset in. */
enum { kAttnQBase = 7 };

enum { kAttnKCoord = 8 };
/** Byte address of THIS KV HEAD's K region at position 0 (ft_tt_kv_cache_run()). */
enum { kAttnKBase = 9 };

enum { kAttnVCoord = 10 };
enum { kAttnVBase = 11 };

enum { kAttnConstCoord = 12 };
/** The constants operand, ft_tt_op_attention_build_constants(): tile 0 the reduce scaler,
 *  tile 1 the 1/sqrt(head_dim) scale, tiles 2..33 the 32 tail masks. Built once per model,
 *  not per call. */
enum { kAttnConstBase = 13 };

enum { kAttnOutCoord = 14 };
/** Byte address of the output's first tile for this group -- the same layout as Q. */
enum { kAttnOutBase = 15 };

enum { kAttnWords = 16 };

/** Tile layout of the constants operand. Fixed for every descriptor: a layout that moved
 *  with the shape would be a wrong address for one call in four.
 *
 *  THE CAUSAL BLOCK IS APPENDED, NOT INTERLEAVED, so tiles 0..33 are byte for byte what
 *  decode already reads and the decode arm's constants upload is unchanged. */
enum { kAttnConstScalerTile = 0 };
enum { kAttnConstScaleTile = 1 };
enum { kAttnConstMaskTile = 2 }; /* + tail, 32 of them */
/** Causal mask `kAttnConstCausalTile + (C - 1)`: element (row r, column c) is live iff
 *  `c <= r && c < C`, dead otherwise. Row r IS position `first + r`, column c IS position
 *  `first + c`, so `c <= r` is causality inside the chunk and `c < C` kills the columns of
 *  a ragged chunk that hold no position at all. 32 of them, C = 1..32. */
enum { kAttnConstCausalTile = 34 };
enum { kAttnConstTiles = kAttnConstCausalTile + 32 };

/** How the reader stacks the four constants it pushes, and how the compute thread indexes
 *  them. Only four of the 34 constant tiles are fetched per descriptor -- the two that
 *  every call needs, and the two masks §4 can select. */
enum { kAttnCbConstScaler = 0 };
enum { kAttnCbConstScale = 1 };
enum { kAttnCbConstTailMask = 2 };
enum { kAttnCbConstDeadMask = 3 };
enum { kAttnCbConstPages = 4 };

/** A tile is 32x32 of four 16x16 faces; a face row is 16 elements on one side of the face
 *  column split, and it is the granule every partial-row copy in this op is made of. */
enum { kAttnTileRows = 32 };
enum { kAttnFaceRows = 16 };

/** One decoded descriptor. Everything the three kernels need, derived ONCE at the top of
 *  the op so no loop re-derives a bound and the three TRISC threads cannot disagree. */
typedef struct fta_shape {
    uint32_t q_rows;
    uint32_t query_row;
    uint32_t flags;
    uint32_t positions;    /**< ROWS mode: strip rows a descriptor serves (byte 3); else 1 */
    uint32_t dim_tiles;
    uint32_t kv_positions;
    uint32_t block_tiles;
    uint32_t capacity_tiles;
    uint32_t live_tiles;   /**< ceil(kv_positions / 32) */
    uint32_t tail;         /**< kv_positions % 32; 0 = the last live tile is full */
    uint32_t padded_tiles; /**< live_tiles rounded up to a multiple of 2*block_tiles */
    uint32_t blocks;       /**< padded_tiles / block_tiles, always even */
} fta_shape;

/** Is this a CHUNK descriptor -- one Q head over C positions -- rather than a decode one? */
FTQ_INLINE bool fta_is_chunk(const fta_shape *shape)
{
    return (shape->flags & (uint32_t)kAttnFlagChunk) != 0u;
}

FTQ_INLINE uint32_t fta_byte(uint32_t word, uint32_t index)
{
    return (word >> (index * 8u)) & 0xFFu;
}

FTQ_INLINE uint32_t fta_coord_x(uint32_t coord)
{
    return coord & 0xFFFFu;
}

FTQ_INLINE uint32_t fta_coord_y(uint32_t coord)
{
    return (coord >> 16) & 0xFFFFu;
}

/** Round `value` up to a multiple of `multiple`. `multiple` is never 0 here -- it is
 *  2*block_tiles and block_tiles is refused at 0 by fta_shape_valid(). */
FTQ_INLINE uint32_t fta_round_up(uint32_t value, uint32_t multiple)
{
    return ((value + multiple - 1u) / multiple) * multiple;
}

/** Decode one descriptor's parameter block. Pure arithmetic; validity is a separate
 *  question and fta_shape_valid() is the only thing that answers it. */
FTQ_INLINE fta_shape fta_decode(volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t heads = parameters[kAttnHeads];
    fta_shape shape;

    shape.q_rows = fta_byte(heads, 0);
    shape.query_row = fta_byte(heads, 1);
    shape.flags = fta_byte(heads, 2);
    shape.positions = (shape.flags & (uint32_t)kAttnFlagRows) != 0u ? fta_byte(heads, 3) : 1u;
    shape.dim_tiles = parameters[kAttnDimTiles];
    shape.kv_positions = parameters[kAttnKvPositions];
    shape.block_tiles = parameters[kAttnBlockTiles];
    shape.capacity_tiles = parameters[kAttnCapacityTiles];
    shape.live_tiles = (shape.kv_positions + kAttnTileRows - 1u) / kAttnTileRows;
    shape.tail = shape.kv_positions % kAttnTileRows;
    shape.padded_tiles =
        (shape.block_tiles == 0u) ? 0u : fta_round_up(shape.live_tiles, 2u * shape.block_tiles);
    shape.blocks = (shape.block_tiles == 0u) ? 0u : shape.padded_tiles / shape.block_tiles;
    return shape;
}

/**
 * Is this a descriptor the program can execute at all?
 *
 * Checked on the DEVICE and not only on the host, because none of these failures is a
 * wrong answer that shows up as a wrong number:
 *   - a zero anywhere becomes a zero-length NoC read (§6);
 *   - a padded window past the region becomes a read outside the cache;
 *   - a query row or head count past a tile becomes a copy outside the circular buffer.
 * A refusal names it in one word instead.
 */
/* HOW A DECODED SHAPE IS PASSED. A POINTER, and the wrapper that used to bind the same
 * name to a C++ REFERENCE is gone with the last C++ compute kernel (plan Phase 6.6,
 * 2026-08-13).
 *
 * All three forms were MEASURED against the pre-conversion images and the ranking was not
 * obvious: pointer-plus-wrapper left 18 of the 21 compute images byte-identical, a body
 * that copies `*ptr` into a local left 15, and passing the struct BY VALUE -- which needed
 * no wrapper at all -- left only 6. Kept as the record of why the pointer form was chosen;
 * with no C++ arm left there is nothing to bind twice. */
FTQ_INLINE bool fta_shape_is_valid(const fta_shape *shape)
{
    if (shape->q_rows == 0u || shape->q_rows > kAttnTileRows)
        return false;
    if (shape->query_row >= kAttnTileRows)
        return false;
    if (shape->dim_tiles == 0u || shape->kv_positions == 0u || shape->block_tiles == 0u)
        return false;
    if ((shape->flags & ~(uint32_t)kAttnFlagsKnown) != 0u)
        return false;
    /* CHUNK geometry, checked on the DEVICE because getting it wrong is a wrong mask on a
     * live tile -- a plausible wrong answer with nothing out of range. The chunk's C
     * positions END the live range and START at a tile boundary, so
     * `kv_positions - q_rows` is the chunk's first position and it must be 32-aligned. */
    if ((shape->flags & (uint32_t)kAttnFlagChunk) != 0u) {
        if (shape->query_row != 0u)
            return false;
        if (shape->kv_positions < shape->q_rows)
            return false;
        if (((shape->kv_positions - shape->q_rows) % kAttnTileRows) != 0u)
            return false;
    }
    if ((shape->flags & (uint32_t)kAttnFlagRows) != 0u) {
        if ((shape->flags & (uint32_t)kAttnFlagChunk) != 0u || shape->positions == 0u ||
            shape->q_rows * shape->positions > kAttnTileRows ||
            shape->query_row + shape->positions > kAttnTileRows ||
            shape->kv_positions < shape->positions ||
            (shape->kv_positions - shape->positions) / kAttnTileRows !=
                (shape->kv_positions - 1u) / kAttnTileRows)
            return false;
    }
    if (shape->live_tiles == 0u || shape->blocks == 0u)
        return false;
    /* The padded window is what the reader actually fetches -- it, not the live length, is
     * what has to fit the cache region. */
    if (shape->padded_tiles > shape->capacity_tiles)
        return false;
    return true;
}

/** Which mask tile position tile `global_tile` needs: the tail mask, the all-dead mask, or
 *  none at all. Returns kAttnCbConstPages for "no mask".
 *
 *  IN CHUNK MODE THE LAST LIVE TILE IS ALWAYS MASKED, INCLUDING WHEN `tail` IS ZERO. That
 *  is the one place the two modes disagree and it is the whole of causality: a full 32-wide
 *  chunk has `tail == 0` and decode would call that tile entirely live, which is every
 *  position in the chunk attending to every other one. The reader loads the causal tile into
 *  the TAIL slot, so this function answers "which CB page", not "which DRAM tile", and the
 *  compute thread needs no chunk awareness at all. */
FTQ_INLINE uint32_t fta_mask_for_global_tile(const fta_shape *shape, uint32_t global_tile)
{
    if (global_tile + 1u < shape->live_tiles)
        return kAttnCbConstPages; /* entirely live */
    if (global_tile + 1u == shape->live_tiles) {
        if ((shape->flags & (uint32_t)(kAttnFlagChunk | kAttnFlagRows)) != 0u)
            return kAttnCbConstTailMask; /* the causal / staircase tile, in the tail slot */
        return (shape->tail == 0u) ? kAttnCbConstPages : kAttnCbConstTailMask;
    }
    return kAttnCbConstDeadMask; /* entirely past the end */
}

/** The DRAM constant tile the reader must land in the TAIL slot for this descriptor: the
 *  causal tile in chunk mode, the tail mask otherwise. One function so the reader and the
 *  off-hardware suite cannot spell the selection differently. */
FTQ_INLINE uint32_t fta_tail_slot_source_tile(const fta_shape *shape)
{
    if ((shape->flags & (uint32_t)kAttnFlagChunk) != 0u)
        return (uint32_t)kAttnConstCausalTile + shape->q_rows - 1u;
    return (uint32_t)kAttnConstMaskTile + shape->tail;
}

// ---------------------------------------------------------------------------
// Face arithmetic -- the same as resident_memory_protocol.h's, and asserted against
// ft_tt_tilize.c rather than trusted
// ---------------------------------------------------------------------------

/** Bytes of one FACE ROW: 16 elements of one row of a tile, on one side of the face column
 *  split. 32 B for bf16. A multiple of the NoC's 16-byte DRAM write alignment, which is the
 *  whole reason a single row can be written at all (ft_tt_op_memory.h §1.4). */
FTQ_INLINE uint32_t fta_chunk_bytes(uint32_t tile_bytes)
{
    return tile_bytes / 64u;
}

FTQ_INLINE uint32_t fta_face_bytes(uint32_t tile_bytes)
{
    return tile_bytes / 4u;
}

/** Byte offset, inside a tile, of the face row holding `row` and columns
 *  [16*half, 16*half+16). Equals ft_tt_tile_face_index(row, 16*half) * element_bytes. */
FTQ_INLINE uint32_t fta_chunk_offset(uint32_t row, uint32_t half, uint32_t tile_bytes)
{
    return ((row / kAttnFaceRows) * 2u + half) * fta_face_bytes(tile_bytes) +
           (row % kAttnFaceRows) * fta_chunk_bytes(tile_bytes);
}

/** §6, as the one function every issue site calls. A length of zero is a REFUSAL. */
FTQ_INLINE bool fta_length_is_issuable(uint32_t bytes)
{
    return bytes != 0u;
}

/** A word copy inside L1. L1 is ordinary memory to the RISC-V and its data cache is
 *  disabled (ft_tt_queue.h, THE ORDERING ASSUMPTION), so this is a plain loop -- an order
 *  below the ~50 ns a NoC transaction costs to issue. Both ends are 4-byte aligned by
 *  construction: a chunk offset is a multiple of chunk_bytes, itself a multiple of 4. */
FTQ_INLINE void fta_copy_words(uint32_t* destination, const uint32_t* source, uint32_t words)
{
    for (uint32_t index = 0; index < words; index++)
        destination[index] = source[index];
}

// ---------------------------------------------------------------------------
// Split-K (§7). Defined for every build, CALLED only by the -DATTN_SPLITK one, so the
// shipped images do not move (an unused static inline emits nothing).
// ---------------------------------------------------------------------------

FTQ_INLINE bool fta_is_partial(const fta_shape *shape)
{
    return (shape->flags & (uint32_t)kAttnFlagPartial) != 0u;
}

FTQ_INLINE bool fta_is_merge(const fta_shape *shape)
{
    return (shape->flags & (uint32_t)kAttnFlagMerge) != 0u;
}

/** Tiles one partial occupies: dim_tiles of O, then m, then l. */
FTQ_INLINE uint32_t fta_partial_tiles(uint32_t dim_tiles)
{
    return dim_tiles + (uint32_t)kAttnPartialStatTiles;
}

/**
 * The split build's validity. PARTIAL is a decode or ROWS descriptor with the flag added and
 * is checked as that descriptor. MERGE has no KV window: word 2 is the partial count and the
 * ROWS live-length rule does not apply; the packing (q_rows, query_row, positions) must still
 * fit one tile, because the writer scatters the merged output with it.
 */
FTQ_INLINE bool fta_split_shape_is_valid(const fta_shape *shape)
{
    fta_shape base = *shape;

    if ((shape->flags & ~(uint32_t)kAttnFlagsKnownSplit) != 0u)
        return false;
    if (fta_is_merge(shape)) {
        if ((shape->flags & (uint32_t)(kAttnFlagChunk | kAttnFlagPartial)) != 0u)
            return false;
        if (shape->q_rows == 0u || shape->q_rows > kAttnTileRows ||
            shape->query_row >= kAttnTileRows || shape->dim_tiles == 0u ||
            shape->kv_positions == 0u || shape->kv_positions > (uint32_t)kAttnMaxPartials)
            return false;
        if ((shape->flags & (uint32_t)kAttnFlagRows) != 0u &&
            (shape->positions == 0u || shape->q_rows * shape->positions > kAttnTileRows ||
             shape->query_row + shape->positions > kAttnTileRows))
            return false;
        return true;
    }
    if (fta_is_partial(shape) && (shape->flags & (uint32_t)kAttnFlagChunk) != 0u)
        return false;
    base.flags &= ~(uint32_t)kAttnFlagPartial;
    return fta_shape_is_valid(&base);
}

/* The compute kernel's two spellings. They used to be bound TWICE -- a C++ reference
 * overload beside these macros -- because `resident_attention_compute.cpp` was written
 * against references and this header did not own it. Plan Phase 6.6 converted that kernel
 * (2026-08-13) and the C++ half is gone; the body has always taken a POINTER, which is what
 * C wants and what the measurement in the block above picked. The split build answers with
 * its own validity (§7). */
#ifdef ATTN_SPLITK
#define fta_shape_valid(shape) fta_split_shape_is_valid(shape)
#else
#define fta_shape_valid(shape) fta_shape_is_valid(shape)
#endif
#define fta_mask_for_tile(shape, global_tile) fta_mask_for_global_tile(shape, global_tile)

#endif  // FOUNDRY_RESIDENT_ATTENTION_PROTOCOL_H
