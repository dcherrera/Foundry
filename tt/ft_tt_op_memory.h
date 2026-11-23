/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_memory.h
 * @brief Host driver for the MEMORY op family — EMBEDDING (gather rows of a table by token
 *   id) and KV_APPEND (place this step's K and V into the cache) — and, more importantly,
 *   **the KV cache memory layout contract that flash-decode attention is written against.**
 *
 * The device half is `kernels/resident_memory_protocol.h`; the two must agree word for word
 * and `tests/test_memory_ops.c` pins the pair off-hardware.
 *
 * ===========================================================================
 * PART 1 — THE KV CACHE LAYOUT CONTRACT
 * ===========================================================================
 * Read this before writing attention. It is the reason KV_APPEND is sequenced BEFORE
 * flash-decode in the plan: the layout is a contract, not an implementation detail, and
 * discovering it late means writing attention twice.
 *
 * ---------------------------------------------------------------------------
 * 1.1 The layout, stated completely
 * ---------------------------------------------------------------------------
 * K and V are stored in **the SAME orientation**, in two separate regions with identical
 * geometry. Per (layer, sequence) there is one K region and one V region.
 *
 *   element      bf16, 2 bytes. (fp32 works; Bfp8_b/Bfp4_b are opt-in — see §1.6.)
 *   tile         32x32, four 16x16 faces in face order
 *   tile ROW (H) = POSITION      — 32 positions per tile
 *   tile COL (W) = HEAD DIMENSION — 32 dims per tile
 *
 *   axis order, slowest to fastest:   HEAD -> POSITION-TILE -> DIM-TILE -> (face, row, col)
 *
 *   offset(head h, position p, dims [32d, 32d+32)) =
 *        h * head_stride  +  (p / 32) * dim_tiles * tile_bytes  +  d * tile_bytes
 *
 *        dim_tiles      = head_dim / 32                 (MiniCPM5-1B: 128 / 32 = 4)
 *        tile_bytes     = 2048                          (bf16)
 *        capacity_tiles = ceil(capacity_positions / 32)
 *        head_stride    = capacity_tiles * dim_tiles * tile_bytes
 *        region bytes   = head_count * head_stride
 *
 *   Element (p, e) sits at row p % 32, column e % 32 of that tile, face-ordered:
 *        byte = tile_base + ((row/16)*2 + col/16) * (tile_bytes/4)
 *                         + (row%16) * (tile_bytes/64) ... + (col%16) * element_bytes
 *   which is exactly ft_tt_tile_face_index(row, col) * element_bytes.
 *
 * MiniCPM5-1B, read from the GGUF header and not assumed:
 * **16 Q heads / 2 KV heads (GQA), head_dim 128, 24 layers, n_embd 1536, context 131072.**
 * So one layer's K region at full context is
 * 2 heads x 4096 position-tiles x 4 dim-tiles x 2048 B = **64 MiB**, and K+V over 24 layers
 * is **3 GiB** of the card's 31.88 GiB. Capacity is a parameter here, never a constant.
 *
 * ---------------------------------------------------------------------------
 * 1.2 What attention gets, and it is the strongest form of the property asked for
 * ---------------------------------------------------------------------------
 * **A position range of ONE head is ONE contiguous byte range, walked LINEARLY in exactly
 * the order the matmul reader already wants.** Not "few transactions" — one, and no stride:
 *
 *     positions [32a, 32b) of head h  ==  bytes [ h*head_stride + a*dim_tiles*tile_bytes ,
 *                                                 h*head_stride + b*dim_tiles*tile_bytes )
 *
 * `ft_tt_kv_cache_run()` returns exactly that pair. Both matmuls of flash-decode consume it
 * as a **B operand in N-major (coalesced) order**, which is the layout the matmul reader's
 * measured fast path was built for:
 *
 *   scores  Sq[q, p]  = sum_e  Q[q, e] * K[p, e]
 *       tile form:  S(0, pt) = sum_dt  Q(0, dt) x K(pt, dt)^T
 *       so in1 = the K tile at (pt, dt) with `matmul_init(in0, in1, transpose = 1)`,
 *       the kt loop running over dt — which is the FAST axis, hence one contiguous read
 *       per position tile (dim_tiles tiles = 8 KiB for this model).
 *
 *   out     O[q, e]   = sum_p  P[q, p] * V[p, e]
 *       tile form:  O(0, dt) += P(0, pt) x V(pt, dt), accumulating over pt
 *       so in1 = the V tile at (pt, dt), transpose 0, and a flash block over positions
 *       reads V(pt, 0..dim_tiles-1) = the same contiguous 8 KiB run.
 *
 * Both operands therefore stream as one linear walk of one region per head. **The cost of
 * that property is zero** — it is not traded against anything, because the append side
 * (§1.4) wanted the same orientation for its own reasons.
 *
 * ---------------------------------------------------------------------------
 * 1.3 THE ONE ASSUMPTION THAT IS NOT MEASURED — and why the layout survives it anyway
 * ---------------------------------------------------------------------------
 * `matmul_init(in0, in1, transpose)` documents `transpose` as "perform transpose operation
 * on B" (the vendor's matmul API, reaching its
 * unpacker matmul-init), i.e. the UNPACKER transposes the in1
 * tile on its way to SrcA. Nothing here has executed it. **INFERRED, not measured.**
 *
 * If it does not work on Blackhole, the LAYOUT DOES NOT CHANGE — only attention's inner
 * loop does, and the cheap fallback is already known:
 *
 *   transpose **Q**, not K.  S^T(pt, 0) = K(pt, dt) x Qt(dt, 0) with in0 = K and
 *   in1 = Q^T, no transpose flag. Q^T is dim_tiles tiles PER HEAD PER LAYER PER TOKEN
 *   (4 tile transposes), against K^T which would be dim_tiles x Pt (4096 tile transposes at
 *   an 8k context). Scores then come out column-oriented, so the softmax reduces with
 *   REDUCE_COL instead of REDUCE_ROW. That is a different inner loop, not a different cache.
 *
 * The reason this is safe to leave unmeasured is §1.4: the orientation is FORCED from the
 * write side. There is no version of this contract where K is stored transposed and the
 * append stays a copy.
 *
 * ---------------------------------------------------------------------------
 * 1.4 Why K is NOT stored transposed, which is the decision most likely to be re-litigated
 * ---------------------------------------------------------------------------
 * The textbook flash-decode K cache is [head_dim x position] — K already transposed, so the
 * QK matmul needs no transpose at all. It was rejected, and the reason is arithmetic:
 *
 *  - **A decode step appends ONE position.** In a [dim x position] tile that is one COLUMN:
 *    32 elements of 2 bytes at a 64-byte stride. The NoC's minimum DRAM write granule is
 *    16 B (`ft_tt_noc_alignment()`), so a column cannot be written at all — every insert
 *    becomes a read-modify-write of dim_tiles whole tiles (8 KiB read + 8 KiB write per head
 *    per layer per token = 768 KiB/token for this model), plus an element-by-element
 *    transpose in L1.
 *  - **In [position x dim] the same insert is a ROW**: per tile, two 32-byte runs (one per
 *    face column-half), 32-byte aligned, which the write alignment permits. No read, no
 *    transpose, no state. `dim_tiles * 2` writes of 32 B per head, and consecutive positions
 *    MERGE into one run per face-half, so a 32-token prefill block is `dim_tiles` writes of
 *    2048 B — a plain tile copy.
 *  - The transposed layout's advantage is one transpose flag in attention. The
 *    position-major layout's advantage is that the write path is a copy. **The write path
 *    happens once per position; attention re-reads that position on every later token.** The
 *    cost that scales with P belongs in the layout; the cost that does not belongs in the
 *    kernel. Here BOTH land on the cheap side, because §1.2 shows the read is already
 *    optimal.
 *
 * ---------------------------------------------------------------------------
 * 1.5 What is FIXED and what is NEGOTIABLE
 * ---------------------------------------------------------------------------
 * FIXED — attention may rely on these, and changing one is a re-write of both ops:
 *   F1  K and V have IDENTICAL geometry and the SAME orientation (position on the tile's
 *       H axis, head dimension on W). Every other choice here follows from this one.
 *   F2  axis order HEAD -> POSITION-TILE -> DIM-TILE, so a position range of one head is
 *       one contiguous linear run (§1.2).
 *   F3  32x32 face-ordered tiles, `ft_tt_tile_face_index()` element order.
 *   F4  the cache is preallocated to `capacity_positions`; growth NEVER relayouts, it fills
 *       forward inside a fixed slab. `head_stride` does not depend on the live length.
 *   F5  positions are DENSE and zero-based per sequence. There is no page table and no free
 *       list. (Paging is the multi-session lever; see NEGOTIABLE.)
 *
 * NEGOTIABLE — these may change without attention changing more than a constant:
 *   N1  `capacity_positions`, per sequence, per model. A parameter everywhere.
 *   N2  the element type: bf16 by default; fp32, Bfp8_b or Bfp4_b by changing `tile_bytes`
 *       alone (§1.6).
 *   N3  padding `capacity_tiles` to make `head_stride` a non-power-of-two. Today it is a
 *       power of two at any power-of-two context, and a power-of-two tile stride measured
 *       10-14% on the one matmul shape that had one. It is
 *       NOT on the inner loop here — the inner walk is contiguous and strideless, and
 *       `head_stride` is crossed once per op — so this is a hazard to measure, not a defect
 *       to pre-pay 25% of the cache for.
 *   N4  whether K and V are two regions or one region with a fixed offset. Two, today,
 *       because it makes each one a plain `ft_tt_op_operand`.
 *   N5  paging: replacing the per-head slab with fixed-size position PAGES chained by an
 *       index. That is the multi-session capacity lever and it preserves F1-F3; it breaks
 *       F2 at the page boundary and turns attention's one read into one read per page.
 *       Not needed for one session at 131072 positions.
 *
 * ---------------------------------------------------------------------------
 * 1.5b THE ROW-PHASE INVARIANT — a CROSS-OP constraint, and the place it would get lost
 * ---------------------------------------------------------------------------
 * **A token at position `p` occupies tile row `p % 32`, everywhere in this family.**
 * `ft_tt_op_memory_position_row()` is that sentence as a function; use it rather than
 * writing the modulo out, so a grep finds every site.
 *
 * It is free here — KV_APPEND takes an absolute position and derives the row from it, so
 * the invariant is the layout — but it is NOT free downstream, and that is why it is
 * written in the contract rather than left implicit:
 *
 *  - **RoPE multiplies activation tile row `r` by sin/cos table row `r`.** So the hidden
 *    state of the token at position `p` must ALREADY be at row `p % 32` before RoPE runs,
 *    which means EMBEDDING has to put it there: pass
 *    `output_row0 = ft_tt_op_memory_position_row(p)` for a decode step, or the absolute
 *    first position for a prefill block (whose rows are then `p % 32` by induction).
 *  - Getting it wrong is a rotation by the wrong angle, which is a plausible wrong answer
 *    with no error anywhere — the failure this whole file is shaped against.
 *
 * The constraint is one line in two ops and it lives BETWEEN them, which is exactly the
 * kind of thing that is lost when each op's author only reads their own header.
 *
 * ---------------------------------------------------------------------------
 * 1.6 Block float: the writer packs, because a face row is one position
 * ---------------------------------------------------------------------------
 * A Bfp8_b/Bfp4_b tile shares one exponent per 16 elements ALONG A FACE ROW.
 * In this layout a face row is 16 head dimensions of ONE
 * position, so an exponent never spans two positions and an append never re-scales an older
 * row. The rows arrive as bf16 from the projection; the KV_APPEND writer packs them
 * (kernels/ft_kv_block_float.h, integer code on NCRISC, bit-exact with convert/ft_bfp.c) and
 * read-modify-writes the destination tile, since a packed face row is smaller than the NoC's
 * write granule. The cache is OPT-IN block float (`tile_bytes` 1088 / 576): bf16 stays the
 * default, and the attention reader takes the same tile size (a separate program whose K/V
 * CBs are block float and which reconfigures unpacker A around the two matmuls). The cost is
 * the format's rounding, which is a quality question for the model's gate, not this layout.
 * K and V MAY DIFFER (a K Bfp8_b / V Bfp4_b cache and the like): each KV_APPEND descriptor
 * writes ONE region and carries that region's format in FLAGS, so the K and V appends pack to
 * their own tiles, and attention runs a mixed program (`attention_k8v4`, ...) whose K and V CBs
 * are paged separately (ft_tt_op_attention_place_cbs_mixed_kv()).
 *
 * ===========================================================================
 * PART 2 — THE TWO OPS, WHICH ARE ONE KERNEL
 * ===========================================================================
 * EMBEDDING and KV_APPEND are the same operation with different sources:
 *
 *   "place `row_count` rows into a face-ordered tile destination at row `dst_row0`,
 *    for `group_count` independent destination groups of `group_tiles` tiles each."
 *
 * EMBEDDING gathers its rows from a ROW-MAJOR table by token id (one group: the whole
 * hidden width). KV_APPEND takes its rows from a TILED source — the K or V projection
 * output — and splits them across one group per KV head. That is the whole difference, and
 * it is two flag bits, so both are answered by ONE resident binary
 * (`resident_gather_reader.c` + `resident_place_writer.c`).
 *
 * **There is no compute thread.** `enables` is DM0|DM1 only. That is safe rather than
 * merely small: TRISC0's stock sync init — which zeroes all 64 CB page-counter
 * pairs — runs only when BRISC writes `RUN_SYNC_MSG_INIT_SYNC_REGISTERS` to it, and a resident BRISC never returns to the firmware loop that does.
 * So a BRISC+NCRISC CB handshake with the TRISCs parked in their GO wait cannot have its
 * counters zeroed underneath it.
 *
 * @status GREEN ON SILICON, on Tensix 30 and again on Tensix 32 — 17 checks, 0 failures
 *   each, byte-exact throughout, both cores probing PASS 4/4 afterwards. **§1.4's
 *   load-bearing fact is now MEASURED, not inferred: a 32-byte write to DRAM at a
 *   32-byte-aligned address lands, and leaves its neighbours untouched** — one position
 *   appended into a poisoned 64 KiB region, 0 bytes wrong and 0 bytes of an untouched
 *   position disturbed. Both opcodes ran on ONE resident image load, on two processors with
 *   no compute thread. Off-hardware: 95 checks, 0 failures, on the Mac (clang) and the card
 *   host (gcc), with TWELVE mutations run against it and all twelve red (1-15 failures each, six
 *   of them in the kernel's own header). Images: reader 2264 B on BRISC, writer 2656 B +
 *   24 B bss on NCRISC, against a 12 KiB slab.
 *
 *   **THE ONE-CB-PER-GROUP-SIZE FIX IS PROVED BOTH WAYS, on silicon, 2026-08-13.** The
 *   fixed build (CB 16 embedding, CB 17 KV) ran 28 descriptors on ONE image load on Tensix
 *   40, 0 checks failed, the 1536-wide gather byte-exact at 0 of 98304 bytes wrong. The
 *   NEGATIVE CONTROL is the SAME binary built with CB_OUT_KV = 16, i.e. exactly the old
 *   one-buffer design, on Tensix 41: it retires the 27 KV descriptors and then REFUSES the
 *   embedding one — completed=27 of 28, kernel error kErrBadParameters, tail=27 head=28 —
 *   which is the defect, still caught, by an unweakened guard. Both cores probe PASS 4/4
 *   afterwards.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_MEMORY_H
#define FT_TT_OP_MEMORY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_launch.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_queue.h"

/** KV cache element formats (the wire values of kernels/ft_kv_block_float.h). */
#define FT_TT_KV_FORMAT_BF16 0u
#define FT_TT_KV_FORMAT_BFP8 1u
#define FT_TT_KV_FORMAT_BFP4 2u

/* ================================================================
 * The KV cache layout
 * ================================================================ */

/**
 * One K region or one V region, for one layer of one sequence. The two are identical in
 * geometry, which is contract item F1 — build one and use it for both.
 *
 * Fields below `tile_bytes` are DERIVED by ft_tt_kv_cache_layout_init(); write them
 * yourself and the helpers will disagree with the kernel.
 */
typedef struct ft_tt_kv_cache_layout {
    uint32_t head_count;         /**< KV heads, not Q heads. MiniCPM5-1B: 2 */
    uint32_t head_dim;           /**< 128; must be a multiple of 32 */
    uint32_t capacity_positions; /**< rounded UP to a multiple of 32 by _init() */
    uint32_t tile_bytes;         /**< 2048 bf16, 4096 fp32, 1088 Bfp8_b, 576 Bfp4_b (§1.6) */

    uint32_t dim_tiles;      /**< head_dim / 32 */
    uint32_t capacity_tiles; /**< capacity_positions / 32 */
    uint32_t position_tile_bytes; /**< dim_tiles * tile_bytes — one position tile, all dims */
    uint64_t head_stride;    /**< capacity_tiles * position_tile_bytes */
    uint64_t region_bytes;   /**< head_count * head_stride */
} ft_tt_kv_cache_layout;

/**
 * Derive a cache layout from a model shape.
 *
 * `capacity_positions` is rounded up to a multiple of 32 because a position tile is the
 * quantum of the layout — a cache that stops mid-tile has no address for its own last
 * position.
 *
 * @param head_count          KV heads (GQA: NOT the Q head count)
 * @param head_dim            multiple of 32
 * @param capacity_positions  > 0; rounded up to 32
 * @param tile_bytes          FT_TT_OP_TILE_BYTES_BF16, _FP32, _BFP8 or _BFP4
 * @param out_layout          zeroed and filled on success
 * @return                    FT_TT_OK; FT_TT_EINVAL on a bad argument
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_kv_cache_layout_init(uint32_t head_count, uint32_t head_dim,
                               uint32_t capacity_positions, uint32_t tile_bytes,
                               ft_tt_kv_cache_layout *out_layout);

/** The cache's element format, FT_TT_KV_FORMAT_*, read off its tile size. */
uint32_t ft_tt_kv_cache_format(const ft_tt_kv_cache_layout *layout);

/**
 * Byte offset, from the region base, of the tile holding position `position` and head
 * dimensions [32*dim_tile, 32*dim_tile+32) of head `head`.
 *
 * This is contract §1.1 as code. Everything else in this header, and both kernels, agree
 * with it by construction rather than by comment.
 *
 * @return the offset, or UINT64_MAX if any index is out of range
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t ft_tt_kv_cache_tile_offset(const ft_tt_kv_cache_layout *layout, uint32_t head,
                                    uint32_t position, uint32_t dim_tile);

/**
 * **The call attention is written against.** The single contiguous byte run holding
 * positions [first_position, first_position + position_count) of head `head`, for K or V
 * alike.
 *
 * `first_position` and `position_count` must both be multiples of 32: a run is a whole
 * number of position tiles, because a tile is the smallest thing the unpacker reads.
 * Attention's live sequence length is not a multiple of 32 in general — round the count UP
 * and mask the tail scores, exactly as a partial tile is handled everywhere else.
 *
 * @param out_offset  byte offset from the region base
 * @param out_bytes   length of the run
 * @return            FT_TT_OK; FT_TT_EINVAL on a bad argument or a non-multiple of 32
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_kv_cache_run(const ft_tt_kv_cache_layout *layout, uint32_t head,
                       uint32_t first_position, uint32_t position_count, uint64_t *out_offset,
                       uint32_t *out_bytes);

/* ================================================================
 * Tile row geometry — the face arithmetic both kernels and the tests share
 * ================================================================ */

/** A face is 16x16 (ft_tt_tilize.h), so a tile holds 4 faces and 64 face-rows. */
#define FT_TT_OP_MEMORY_FACE_ROWS 16u
#define FT_TT_OP_MEMORY_FACE_HALVES 2u

/** Bytes of one FACE ROW — 16 elements of one row of a tile, on one side of the face
 *  column split. 32 B for bf16, 64 B for fp32. This is the granule every partial-row write
 *  in this family is made of, and the reason the layout works: it is a multiple of the
 *  16-byte DRAM write alignment (ft_tt_noc_alignment()). */
#define FT_TT_OP_MEMORY_CHUNK_BYTES(tile_bytes) ((tile_bytes) / 64u)

/** Bytes of one face. */
#define FT_TT_OP_MEMORY_FACE_BYTES(tile_bytes) ((tile_bytes) / 4u)

/**
 * Byte offset, inside a tile, of the face row holding row `row` and columns
 * [16*half, 16*half+16).
 *
 * Equals ft_tt_tile_face_index(row, 16*half) * element_bytes, and tests/test_memory_ops.c
 * asserts that against ft_tt_tilize.c for every (row, half) rather than trusting the
 * rewrite.
 *
 * @param row         0..31
 * @param half        0 or 1
 * @param tile_bytes  2048 or 4096
 * @return            the offset, or UINT32_MAX for a bad argument
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_op_memory_chunk_offset(uint32_t row, uint32_t half, uint32_t tile_bytes);

/**
 * The tile row a token at sequence position `position` must occupy — §1.5b's row-phase
 * invariant, as a function so a grep finds every site that depends on it.
 *
 * It is `position % 32` and it always will be. It exists as a call because it is a
 * CROSS-OP constraint: RoPE multiplies activation tile row `r` by sin/cos table row `r`,
 * so the embedding op has to place a decode token here and nowhere else, and a modulo
 * written out by hand in two files is a constraint that can drift in one of them.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_op_memory_position_row(uint32_t position);

/* ================================================================
 * The parameter block — mirrors kernels/resident_memory_protocol.h
 * ================================================================ */

/** The sixteen words a FT_TT_QUEUE_OP_EMBEDDING or FT_TT_QUEUE_OP_KV_APPEND descriptor
 *  points at. A word an op does not use is written 0 by the host and never read by the
 *  kernel; it is NOT free for that op to redefine (the rule ft_tt_op_vector.h states, for
 *  the same reason — a block whose meaning depends on which binary is resident is a wrong
 *  answer with no error anywhere). */
typedef enum ft_tt_op_memory_field {
    FT_TT_OP_MEMORY_FLAGS = 0,            /**< ft_tt_op_memory_flag bits */
    FT_TT_OP_MEMORY_ROW_COUNT = 1,        /**< rows to place; 1..32 (see the split rule) */
    FT_TT_OP_MEMORY_GROUP_COUNT = 2,      /**< independent destinations; KV heads, or 1 */
    FT_TT_OP_MEMORY_GROUP_TILES = 3,      /**< tiles across ONE group's row */
    FT_TT_OP_MEMORY_SRC_X = 4,
    FT_TT_OP_MEMORY_SRC_Y = 5,
    FT_TT_OP_MEMORY_SRC_BASE = 6,
    FT_TT_OP_MEMORY_SRC_ROW_STRIDE = 7,   /**< ROW-MAJOR only: bytes between table rows */
    FT_TT_OP_MEMORY_SRC_ROW0 = 8,         /**< TILED: first source row. GATHER: unused */
    FT_TT_OP_MEMORY_SRC_GROUP_STRIDE = 9, /**< bytes between source groups */
    FT_TT_OP_MEMORY_DST_X = 10,
    FT_TT_OP_MEMORY_DST_Y = 11,
    FT_TT_OP_MEMORY_DST_BASE = 12,
    /** Bytes each destination group OWNS: the stride when there are several, the EXTENT
     *  when there is one. Both readings are the same number here because the groups are
     *  contiguous. It is the only field that knows how big the destination is, and DST_ROW0
     *  is ABSOLUTE — its tile row multiplies into a raw byte offset — so the kernel bounds
     *  the write against this and REFUSES a 0. */
    FT_TT_OP_MEMORY_DST_GROUP_STRIDE = 13,
    FT_TT_OP_MEMORY_DST_ROW0 = 14,         /**< absolute destination row = KV position */
    /** ROW-MAJOR only: rows the source table HAS. The gather index and the sequential row
     *  index are raw wire values that become a DRAM address as
     *  `src_base + index * src_row_stride`, and nothing else in this block bounds them — so
     *  without this word a wrong token id is an arbitrary DRAM read rather than an error.
     *  The kernel REFUSES a 0 in row-major mode rather than reading it as "unchecked". */
    FT_TT_OP_MEMORY_SRC_ROW_LIMIT = 15,
    FT_TT_OP_MEMORY_WORDS = 16
} ft_tt_op_memory_field;

/** Flag bits of FT_TT_OP_MEMORY_FLAGS. */
typedef enum ft_tt_op_memory_flag {
    /** The SOURCE is face-ordered tiles (KV_APPEND). Clear means row-major (EMBEDDING). */
    FT_TT_OP_MEMORY_FLAG_SRC_TILED = 1u << 0,
    /** The descriptor OWNS the destination tile row: the rows it does not place are zeroed
     *  and the whole group goes out as ONE transaction. Clear writes only the face-row runs
     *  the placed rows occupy and preserves everything else — which is what an APPEND into a
     *  live cache must do, and why KV_APPEND never sets this. */
    FT_TT_OP_MEMORY_FLAG_DST_WHOLE_TILES = 1u << 1,
    /** Source rows are named by an index array in this core's L1 (EMBEDDING's token ids).
     *  Clear means rows SRC_ROW0.. are taken in order. */
    FT_TT_OP_MEMORY_FLAG_GATHER = 1u << 2,
    /** BANKED destination (KV_APPEND only): the groups are split into equal BANK GROUPS of
     *  `heads_per_bank` (bits FT_TT_OP_MEMORY_BANK_HEADS_SHIFT..+7 of FLAGS) and bank group b
     *  lands at (byte b of DST_X, byte b of DST_Y, DST_BASE): one descriptor writes heads that
     *  live in up to FT_TT_OP_MEMORY_MAX_BANKS banks, all at the SAME address in each. Inside
     *  a bank group, head h is at DST_BASE + (h % heads_per_bank) * DST_GROUP_STRIDE. */
    FT_TT_OP_MEMORY_FLAG_DST_BANKED = 1u << 3
} ft_tt_op_memory_flag;

/** KV_APPEND: the CACHE element format (FT_TT_KV_FORMAT_*) in FLAGS bits 4..5. Non-zero means
 *  the writer packs each appended face row to block float (ft_kv_block_float.h). */
#define FT_TT_OP_MEMORY_KV_FORMAT_SHIFT 4u

/** BANKED: where `heads_per_bank` sits in FLAGS (8 bits). */
#define FT_TT_OP_MEMORY_BANK_HEADS_SHIFT 8u
/** BANKED: most bank groups one descriptor can name (one byte each of DST_X / DST_Y). */
#define FT_TT_OP_MEMORY_MAX_BANKS 4u

/**
 * THE SPLIT RULE, and it is a refusal rather than a loop.
 *
 * One descriptor places rows inside ONE source tile row and ONE destination tile row:
 *   (SRC_ROW0 % 32) + ROW_COUNT <= 32   and   (DST_ROW0 % 32) + ROW_COUNT <= 32
 *
 * A kernel that spanned tile rows would carry two nested wrap cases in a loop that also
 * does face arithmetic, and neither is visible in an output that is "nearly right". The
 * host splits instead — ft_tt_op_kv_append_split() does it — and each piece is one enqueue
 * at 1.36 µs, so a 512-token prefill costs 16 descriptors per region per layer.
 */
#define FT_TT_OP_MEMORY_MAX_ROWS 32u

/* ================================================================
 * Circular buffers — no compute thread, so these are a staging map
 * ================================================================ */

/**
 * CB indices, matching the compile-time args both kernels are built with.
 *
 * ===========================================================================
 * ONE OUTPUT CB PER GROUP SIZE — and the alternative is not merely worse, it is FALSE
 * ===========================================================================
 * There used to be one CB_OUT, sized for the widest group the residency would serve, with
 * both opcodes pushing through it. **That overruns L1**, and the corrected CB grid guard
 * (`kernels/ft_tensix_transfer_guard.h`, `ft_tensix_cb_push_grid_is_safe()`) caught it on
 * silicon: 27 KV_APPEND descriptors of 2 groups x 4 tiles leave the write pointer 24 pages
 * into a 96-page buffer, and the next EMBEDDING descriptor's 48-page block is refused —
 * correctly, because a SECOND one at pointer 72 writes pages 72..119 of a 96-page buffer.
 * The shipped test escaped only by issuing one.
 *
 * **The obvious host rule — "every group size divides the coarsest" — does not fix it, and
 * MiniCPM5-1B is the counterexample rather than a hypothetical.** Its two group sizes are
 * 48 (embedding, 1536/32) and 4 (KV, head_dim/32), and 4 already divides 48. The failure
 * above is that configuration. The arithmetic says why: with one buffer of `P` pages, a
 * chunk `c1` pushed `k` times leaves the pointer at `k*c1 mod P`, and a later chunk `c2`
 * is legal only when `c2` divides that for EVERY reachable `k`. Since `k` is chosen by the
 * host at run time, that forces `c2 | c1` — the COARSEST must divide the FINEST, which for
 * a set of sizes means they are all EQUAL. So the host rule collapses into this one: a CB
 * carries exactly one group size, for its whole life, and two group sizes need two CBs.
 *
 * The invariant then holds by construction and needs no host discipline at all: each CB is
 * placed as a whole multiple of its own group, so its pointer is on that grid at every
 * descriptor boundary and can never leave it. The guard stays, unweakened, as the check
 * that this is still true — a build that routes both opcodes to one CB is refused on the
 * card, which is the negative control the fix is proved against.
 *
 * The cost is one extra CB and, for MiniCPM5-1B, 16 KiB of a 1,351,680-byte arena.
 */
#define FT_TT_OP_MEMORY_CB_STAGE 0u  /**< reader-private: one source row or tile group */
#define FT_TT_OP_MEMORY_CB_INDEX 1u  /**< reader-private: the gather index array */
/** reader -> writer, EMBEDDING only: `embedding_dim / 32` tiles a group, one group. */
#define FT_TT_OP_MEMORY_CB_OUT_EMBEDDING 16u
/** reader -> writer, KV_APPEND only: `dim_tiles` tiles a group, one group per KV head. */
#define FT_TT_OP_MEMORY_CB_OUT_KV        17u
/** writer-private: the block-float append's two packed-tile slots (kMemCbKvPack). */
#define FT_TT_OP_MEMORY_CB_KV_PACK       2u
#define FT_TT_OP_MEMORY_KV_PACK_BYTES    (2u * FT_TT_OP_TILE_BYTES_BFP8_B)
#define FT_TT_OP_MEMORY_CB_COUNT 5u

/**
 * CB_STAGE and CB_INDEX are addressed with get_write_ptr()/get_read_ptr() and NEVER
 * pushed or popped. They are declared as circular buffers so their L1 comes out of the
 * arena allocator like everything else's and shows up in the same accounting — not because
 * anything flows through them. Only CB_OUT carries the producer/consumer protocol.
 */
#define FT_TT_OP_MEMORY_INDEX_BYTES (FT_TT_OP_MEMORY_MAX_ROWS * 4u)

/* ================================================================
 * EMBEDDING
 * ================================================================ */

/** One embedding call: gather `token_count` rows of a row-major table and place them at
 *  destination rows [output_row0, output_row0 + token_count). */
typedef struct ft_tt_op_embedding_request {
    uint32_t token_count;    /**< 1..32 */
    uint32_t embedding_dim;  /**< multiple of 32; MiniCPM5-1B: 1536 */
    /** The table. ROW-MAJOR and untilized, which is a contract of its own: a gather reads
     *  ONE row, and a row of a row-major table is one aligned contiguous read, while a row
     *  of a tilized table is `embedding_dim/16` scattered face rows of 32 B — and a 32-byte
     *  DRAM READ is not legal at all (64 B, ft_tt_noc_alignment()). So a tilized embedding
     *  table cannot be gathered from directly, at any speed. */
    ft_tt_op_operand table;
    uint32_t table_row_stride; /**< bytes between table rows; >= embedding_dim*element */
    /** Rows the table HAS — the vocabulary size. **Required, and a 0 is refused.** A token
     *  id is a raw wire value that the kernel turns into `table.address + id * row_stride`,
     *  so this is the only thing standing between a wrong id and an arbitrary DRAM read.
     *  It rides the descriptor as FT_TT_OP_MEMORY_SRC_ROW_LIMIT because the check has to be
     *  made where the transfer is ISSUED — `ft_tt_noc_check_transfer()` runs host-side and
     *  every resident op issues its own transfers. */
    uint32_t table_row_count;
    ft_tt_op_operand output;   /**< face-ordered tiles, embedding_dim/32 across */
    /** Bytes the OUTPUT region has, from `output.address`. **Required, and a 0 is refused.**
     *  `output_row0` is absolute, so its tile row becomes a raw byte offset with nothing
     *  above it. It rides the descriptor as FT_TT_OP_MEMORY_DST_GROUP_STRIDE — an embedding
     *  has ONE destination group, so the bytes it owns and the region's extent are the same
     *  number, and the parameter block has no spare word for a second name. */
    uint32_t output_bytes;
    /** Absolute destination row. **For a decode step this MUST be
     *  ft_tt_op_memory_position_row(position)** — §1.5b: RoPE multiplies activation tile row
     *  `r` by table row `r`, so the token has to arrive at row `position % 32`. For a prefill
     *  block it is the absolute first position, whose rows are then `p % 32` by induction. */
    uint32_t output_row0;
    /** The token ids are NOT in this struct: they are `token_count` uint32 words the caller
     *  writes to ft_tt_op_memory_index_address() before enqueuing, because this encoder is
     *  pure and an L1 write is not. Every embedding descriptor is a GATHER — an embedding
     *  whose rows were consecutive would be a copy and not a lookup, and one path is one
     *  thing to get right; a caller that wants rows in order writes the ids in order. */
    /** This call OWNS the destination tile row: the rows it does not place come out zero and
     *  the whole row goes out as one transaction. Costs a memset of `embedding_dim/32` tiles
     *  in L1 when token_count < 32 — ~98 KiB of RISC-V stores for a 1536-wide decode step.
     *  Clear it and the untouched rows keep whatever the destination held, which is correct
     *  for this model (a matmul's output rows are independent, and a norm is per row) and
     *  cheaper, but leaves the buffer untestable byte for byte. */
    bool zero_fill;
} ft_tt_op_embedding_request;

/**
 * Encode `request` into the sixteen words a FT_TT_QUEUE_OP_EMBEDDING descriptor points at.
 *
 * Pure: no device access, no allocation.
 *
 * @param out_words  FT_TT_OP_MEMORY_WORDS values, fully written (unused words zeroed)
 * @return           FT_TT_OK; FT_TT_EINVAL on a zero or >32 token count, a dimension that
 *                   is not a multiple of 32, a zero tile size, or a row stride shorter
 *                   than one row
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_embedding_parameters(const ft_tt_op_embedding_request *request,
                                  uint32_t out_words[FT_TT_OP_MEMORY_WORDS]);

/* ================================================================
 * KV_APPEND
 * ================================================================ */

/** One KV append: take rows [source_row0, source_row0 + token_count) of a tiled projection
 *  output and place them at cache positions [position, position + token_count), for every
 *  head. Call it twice per layer per step — once for K, once for V. */
typedef struct ft_tt_op_kv_append_request {
    const ft_tt_kv_cache_layout *cache; /**< the contract; K and V share one layout */
    uint32_t position;                  /**< absolute first position */
    uint32_t token_count;               /**< 1..32 */
    /** The K or V projection output: face-ordered tiles, `head_count * dim_tiles` across,
     *  heads concatenated along the width in head order — which is how a fused KV
     *  projection already lays them out. */
    ft_tt_op_operand source;
    uint32_t source_row0;      /**< first row within the source */
    uint32_t source_tiles;     /**< tiles across ONE source tile row */
    ft_tt_op_operand region;   /**< base of the K region, or of the V region */
    /** BANKED (optional, 0 or 1 = off): the heads split into `bank_count` equal groups,
     *  group b's heads at `bank_region[b]` — every one at `region.address`, only the bank
     *  (x, y) differs — written by ONE descriptor (FT_TT_OP_MEMORY_FLAG_DST_BANKED). */
    uint32_t bank_count;
    ft_tt_op_operand bank_region[FT_TT_OP_MEMORY_MAX_BANKS];
} ft_tt_op_kv_append_request;

/**
 * Encode `request` into the sixteen words a FT_TT_QUEUE_OP_KV_APPEND descriptor points at.
 *
 * @param out_words  FT_TT_OP_MEMORY_WORDS values, fully written
 * @return           FT_TT_OK; FT_TT_EINVAL on a NULL or uninitialised layout, a token count
 *                   of 0 or > 32, a source/destination row range that crosses a tile row
 *                   (FT_TT_OP_MEMORY_MAX_ROWS — the split rule), a last position past the
 *                   cache capacity, a source narrower than head_count*dim_tiles, or a tile
 *                   size that disagrees with the layout; BANKED: a bank count above
 *                   FT_TT_OP_MEMORY_MAX_BANKS or not dividing the heads, a bank region at
 *                   another address than `region`, or a coordinate above 255
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_kv_append_parameters(const ft_tt_op_kv_append_request *request,
                                  uint32_t out_words[FT_TT_OP_MEMORY_WORDS]);

/**
 * How many descriptors `token_count` tokens starting at `position` need, given the split
 * rule, and where each one starts.
 *
 * A caller appending a prefill block asks this, then loops. It exists so the split is one
 * function with one test rather than an open-coded loop in every caller.
 *
 * @param position      absolute first position
 * @param token_count   tokens to append; > 0
 * @param source_row0   first source row
 * @param out_pieces    may be NULL; receives up to `capacity` {position, source_row0,
 *                      token_count} triples as three parallel arrays
 * @param capacity      entries available in each output array
 * @param out_count     receives the number of pieces, whether or not they fit
 * @return              FT_TT_OK; FT_TT_EINVAL on a bad argument; FT_TT_ENOMEM if the
 *                      pieces do not fit `capacity` (out_count still says how many)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_kv_append_split(uint32_t position, uint32_t token_count, uint32_t source_row0,
                             uint32_t *out_positions, uint32_t *out_source_rows,
                             uint32_t *out_token_counts, uint32_t capacity, uint32_t *out_count);

/* ================================================================
 * Placement
 * ================================================================ */

/**
 * Place the memory family's four circular buffers in the arena.
 *
 * **The two output CBs are separate because a CB carries exactly ONE group size** — see the
 * CB-index block above for the arithmetic, and for why "make the sizes divide each other"
 * is not a weaker version of this rule but a false one. Each output CB gets two blocks of
 * its own group so the writer drains one while the reader fills the other, which also makes
 * its page count a whole multiple of its group and puts its pointer permanently on that
 * grid.
 *
 * CB_STAGE is ONE page of `max(group_tiles) * tile_bytes`, which serves both source shapes:
 * 32 rows of a group is exactly the group's tiles worth of bytes, so there is one staging
 * size and not two to keep in step.
 *
 * Both group sizes are REQUIRED even for a residency that will only ever run one opcode.
 * A zero would place no CB and the kernel's grid guard would then refuse that opcode at the
 * first descriptor, which is a run-time failure standing in for a build-time one; and the
 * narrow CB costs 16 KiB of a 1,351,680-byte arena, so there is nothing to buy.
 *
 * @param embedding_group_tiles  tiles across one EMBEDDING group, i.e. embedding_dim/32; >= 1
 * @param kv_group_tiles         tiles across one KV_APPEND group, i.e. dim_tiles; >= 1
 * @param tile_bytes             2048 or 4096
 * @param out_entries   FT_TT_OP_MEMORY_CB_COUNT entries with addresses filled in
 * @param buffer        receives the encoded CB descriptor array
 * @param out_mask      receives the launch message's local_cb_mask
 * @return              FT_TT_OK; FT_TT_EINVAL; FT_TT_ENOMEM if it does not fit the arena
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_memory_place_cbs(uint32_t embedding_group_tiles, uint32_t kv_group_tiles,
                              uint32_t tile_bytes,
                              ft_tt_launch_cb_entry out_entries[FT_TT_OP_MEMORY_CB_COUNT],
                              void *buffer, size_t buffer_bytes, uint64_t *out_mask);

/**
 * The L1 address the host writes the gather index array to — CB_INDEX's page, which the
 * reader reads with get_read_ptr(). Derived from the placement rather than remembered, so a
 * caller cannot write the ids somewhere the kernel does not look.
 *
 * @param entries  from ft_tt_op_memory_place_cbs()
 * @param count    FT_TT_OP_MEMORY_CB_COUNT
 * @return         the address, or 0 if CB_INDEX is not in `entries`
 */
uint32_t ft_tt_op_memory_index_address(const ft_tt_launch_cb_entry *entries, unsigned count);

/** "EMBEDDING" / "KV_APPEND" for logs. Never NULL. */
const char *ft_tt_op_memory_opcode_name(uint32_t opcode);

#endif /* FT_TT_OP_MEMORY_H */
