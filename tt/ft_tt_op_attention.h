/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_attention.h
 * @brief Host driver for FLASH-DECODE ATTENTION: one GQA group's query against the KV
 *   cache, with an ONLINE softmax so the score row is never materialised. Places the
 *   triad's circular buffers, encodes its parameter block, and builds the constants — the
 *   reduce scaler, the 1/sqrt(head_dim) scale and the 32 tail masks.
 *
 * The device half is `kernels/resident_attention_protocol.h`; the two must agree word for
 * word and `tests/test_attention_ops.c` pins the pair off-hardware.
 *
 * ---------------------------------------------------------------------------
 * THE UNIT OF WORK IS A KV GROUP, NOT A Q HEAD, AND THAT IS THE DESIGN DECISION
 * ---------------------------------------------------------------------------
 * MiniCPM5-1B is GQA — **16 Q heads, 2 KV heads, head_dim 128, 24 layers, context
 * 131072**, read from the GGUF header and not assumed. So
 * each KV head serves EIGHT Q heads, and the choice of what one core owns is the choice of
 * how many times the KV cache is read:
 *
 *  - **a core owns a Q HEAD** — the cache is read 8x per layer per token, because each of
 *    the 8 Q heads sharing a KV head walks it separately;
 *  - **a core owns the KV GROUP** — the cache is read ONCE and serves all 8.
 *
 * Decode is bandwidth-shaped: at 8k context one layer's K+V for one KV head is 8 MiB, so
 * the first design pays 64 MiB per layer per token against the second's 8. The group is
 * the unit. The 8 Q heads become the 8 ROWS of one tile, so the score matmul is issued once
 * rather than eight times — which also keeps the op data-movement bound rather than MATH
 * bound (8 HiFi4 tile matmuls per K tile pair would be 640 ns against a reader delivering a
 * pair every ~104 ns, docs/tt_kernel_model.md §12, §16).
 *
 * ---------------------------------------------------------------------------
 * THE ONLINE SOFTMAX IS A CAPACITY REQUIREMENT, NOT A PREFERENCE
 * ---------------------------------------------------------------------------
 * A standalone softmax holds its row twice over and tops out at Wt = 325, i.e. W ≤ 10,400
 * (`ft_tt_op_reduce_max_tile_cols()`, docs/tt_bringup_log.md §22.3). A 32k-context
 * attention row is **Wt = 1024 and does not fit at any tile size**. So this op accumulates
 * a running max, row sum and output across KV blocks and never holds more than one block:
 * `ft_tt_op_attention_l1_bytes()` is flat in the context length and depends only on the
 * residency. That is the whole reason attention is its own op rather than "matmul, then
 * softmax, then matmul".
 *
 * ---------------------------------------------------------------------------
 * WHAT IT CONSUMES, AND WHAT IT IS ALLOWED TO ASSUME
 * ---------------------------------------------------------------------------
 * K and V are `ft_tt_op_memory.h` PART 1's contract, unchanged: F1 (K and V share one
 * orientation, position on the tile's H axis), F2 (axis order HEAD → POSITION-TILE →
 * DIM-TILE, so a position range of one head is ONE contiguous run), F3 (face-ordered
 * tiles), F4 (preallocated, growth never relayouts). `ft_tt_kv_cache_run()` gives the
 * (offset, bytes) pair this driver folds into the parameter block.
 *
 * §1.3's ONE inferred assumption — that `matmul_init(in0, in1, transpose)` transposes B on
 * Blackhole — is **MEASURED and true** as of this op: `tests/test_attention_transpose_probe.c`
 * runs both arms of one binary against a non-symmetric operand pair and the transposed arm
 * reproduces A·Bᵀ (PCC 0.999998668) while the plain arm reproduces A·B. The fallback §1.3
 * describes (transpose Q instead) is therefore not needed and not implemented.
 *
 * Q and the output are the model's ordinary activation layout: a tile grid
 * `q_head_count * dim_tiles` wide, head h's dim tile d at tile column `h*dim_tiles + d`,
 * the decode token at row `ft_tt_op_memory_position_row(position)` — §1.5b's row-phase
 * invariant, inherited rather than re-decided.
 *
 * @status GREEN ON SILICON — Tensix 70, 71, 72 and 73, 0 checks failed on every one, PCC
 *   0.99998-1.0 and max-abs <= 0.002 against a CPU fp32 reference at kv = 1, 33, 256, 1000
 *   and 2048. Core 71 reproduced core 70's PCCs to NINE DIGITS on five shapes. Four kernel
 *   mutations were built into their own program directories and driven through the SAME
 *   unmodified test binary: all four go red (6, 6, 10 and 2 failures). Timing at
 *   block_tiles = 16: **0.133 / 0.236 / 0.842 ms per descriptor at kv = 1024 / 2048 / 8192**,
 *   i.e. 40.4 ms and 24.7 tok/s per token at 8k on ONE core (24 layers x 2 KV groups), with
 *   the two groups independent so two cores halve it.
 *   SPLIT-K (2026-11-16, Qwen3.8 TT usable plan 3.1): FT_TT_OP_ATTENTION_FLAG_PARTIAL /
 *   _MERGE, served only by the `attention_k8v4_split` program. Host-tested
 *   (tests/test_attention_splitk.c); not yet run on silicon.
 * @issues The op is COMPUTE bound, not reader bound: at kv = 8192 it moves 4 MiB of KV in
 *   0.842 ms = 5.0 GB/s, against the matmul reader's measured ~28 GB/s at the same tile
 *   size. INFERRED, not measured: the SFPU exponential and the two row reduces, which the
 *   matmul does not have. That makes LoFi and an approximate exp real levers here for the
 *   first time in this backend — the opposite of docs/tt_bringup_log.md §16's finding on
 *   the matmul, and it should be measured before it is believed.
 * @todo None
 */

#ifndef FT_TT_OP_ATTENTION_H
#define FT_TT_OP_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_launch.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_queue.h"

/* ================================================================
 * The parameter block — mirrors kernels/resident_attention_protocol.h
 * ================================================================ */

/** The sixteen words a FT_TT_QUEUE_OP_ATTENTION descriptor points at.
 *
 *  Every operand's NoC coordinate is ONE word, `(y << 16) | x`. The other families spend
 *  two words on a coordinate; this op has five operands and sixteen words, so it packs —
 *  symmetrically on both sides of the wire, and pinned by the off-hardware suite. */
typedef enum ft_tt_op_attention_field {
    /** byte 0 q_rows, byte 1 query_row, byte 2 flags. */
    FT_TT_OP_ATTENTION_HEADS = 0,
    FT_TT_OP_ATTENTION_DIM_TILES = 1,
    /** LIVE positions attended to. The live tile count and the tail width are DERIVED from
     *  it on both sides — two fields that must agree are two fields that can disagree. */
    FT_TT_OP_ATTENTION_KV_POSITIONS = 2,
    /** Position tiles per K/V block. Fixed by the RESIDENCY: it fixes the page counts, and
     *  the kernel refuses a descriptor that disagrees with the buffers it was staged with. */
    FT_TT_OP_ATTENTION_BLOCK_TILES = 3,
    /** Position tiles the cache region holds, so the kernel can refuse a padded read that
     *  would run past it. */
    FT_TT_OP_ATTENTION_CAPACITY_TILES = 4,
    FT_TT_OP_ATTENTION_RESERVED = 5,
    FT_TT_OP_ATTENTION_Q_COORD = 6,
    FT_TT_OP_ATTENTION_Q_BASE = 7,
    FT_TT_OP_ATTENTION_K_COORD = 8,
    FT_TT_OP_ATTENTION_K_BASE = 9,
    FT_TT_OP_ATTENTION_V_COORD = 10,
    FT_TT_OP_ATTENTION_V_BASE = 11,
    FT_TT_OP_ATTENTION_CONST_COORD = 12,
    FT_TT_OP_ATTENTION_CONST_BASE = 13,
    FT_TT_OP_ATTENTION_OUT_COORD = 14,
    FT_TT_OP_ATTENTION_OUT_BASE = 15,
    FT_TT_OP_ATTENTION_WORDS = 16
} ft_tt_op_attention_field;

/** Tile layout of the constants operand, built once per model.
 *
 *  THE CAUSAL BLOCK IS APPENDED. Tiles 0..33 are byte for byte what the decode arm already
 *  uploads, so adding chunk mode does not move a single constant decode reads. */
#define FT_TT_OP_ATTENTION_CONST_SCALER_TILE 0u
#define FT_TT_OP_ATTENTION_CONST_SCALE_TILE  1u
#define FT_TT_OP_ATTENTION_CONST_MASK_TILE   2u
/** Causal mask `+ (C - 1)`: element (row r, column c) live iff `c <= r && c < C`. */
#define FT_TT_OP_ATTENTION_CONST_CAUSAL_TILE (FT_TT_OP_ATTENTION_CONST_MASK_TILE + 32u)
#define FT_TT_OP_ATTENTION_CONST_TILES       (FT_TT_OP_ATTENTION_CONST_CAUSAL_TILE + 32u)

/** `flags` byte 2 of word 0. CHUNK: one descriptor is ONE Q head over `q_rows` consecutive
 *  positions instead of one KV group at one position — kernels/resident_attention_protocol.h
 *  carries the design and the exhaustive list of what it changes. */
#define FT_TT_OP_ATTENTION_FLAG_CHUNK 1u
/** S2f ROWS mode — mirrors kAttnFlagRows (resident_attention_protocol.h). */
#define FT_TT_OP_ATTENTION_FLAG_ROWS 2u
/** SPLIT-K (resident_attention_protocol.h §7; only the `_split` programs accept them).
 *  PARTIAL: a decode or ROWS descriptor over one chunk whose output is the chunk's
 *  un-normalised O, its max and its sum — FT_TT_OP_ATTENTION_PARTIAL_STAT_TILES more tiles
 *  than dim_tiles, written whole at `output`. MERGE: `partial_count` partials at `query`
 *  (contiguous, ft_tt_op_attention_partial_bytes() apiece) folded into the ordinary output;
 *  `keys`, `values` and `constants` are unread and need not be set. */
#define FT_TT_OP_ATTENTION_FLAG_PARTIAL 4u
#define FT_TT_OP_ATTENTION_FLAG_MERGE   8u
/** Tiles of one partial past its dim_tiles of O: the max tile, then the sum tile. */
#define FT_TT_OP_ATTENTION_PARTIAL_STAT_TILES 2u
/** The most partials one MERGE descriptor folds — mirrors kAttnMaxPartials. */
#define FT_TT_OP_ATTENTION_MAX_PARTIALS 64u

/** How the reader stacks the four constants it fetches per descriptor. */
#define FT_TT_OP_ATTENTION_CONST_PAGES 4u

/** The value a masked-out score is driven to before the row max sees it. Large enough that
 *  `exp(dead - live_max)` is zero in bf16 at any plausible score, small enough that the
 *  subtraction cannot overflow. */
#define FT_TT_OP_ATTENTION_MASK_VALUE (-1.0e30f)

/* CB indices, matching the compile-time args the triad is built with. */
#define FT_TT_OP_ATTENTION_CB_Q       0u
#define FT_TT_OP_ATTENTION_CB_K       1u
#define FT_TT_OP_ATTENTION_CB_V       2u
#define FT_TT_OP_ATTENTION_CB_CONST   3u
#define FT_TT_OP_ATTENTION_CB_S       4u
#define FT_TT_OP_ATTENTION_CB_SCRATCH 5u
#define FT_TT_OP_ATTENTION_CB_M_A     6u
#define FT_TT_OP_ATTENTION_CB_M_B     7u
#define FT_TT_OP_ATTENTION_CB_L       8u
#define FT_TT_OP_ATTENTION_CB_LSUM    9u
#define FT_TT_OP_ATTENTION_CB_C       10u
#define FT_TT_OP_ATTENTION_CB_O       11u
#define FT_TT_OP_ATTENTION_CB_PV      12u
#define FT_TT_OP_ATTENTION_CB_QSTAGE  13u
#define FT_TT_OP_ATTENTION_CB_QS      14u
#define FT_TT_OP_ATTENTION_CB_OUT     16u
#define FT_TT_OP_ATTENTION_CB_COUNT   16u

/* ================================================================
 * The residency — what ONE staged program can serve
 * ================================================================ */

/**
 * The circular buffers are placed once, at launch, so the RESIDENCY and not the descriptor
 * fixes the group width, the head dimension and the KV block size. A descriptor that
 * disagrees is REFUSED by the kernel rather than run against buffers sized for something
 * else — the page counts are what a block-prefetch reader's correctness rests on
 * (kernels/resident_attention_protocol.h §5).
 */
typedef struct ft_tt_op_attention_residency {
    uint32_t q_rows;      /**< Q heads per KV group; 8 for MiniCPM5-1B. 1..32 */
    uint32_t dim_tiles;   /**< head_dim / 32; 4 for MiniCPM5-1B */
    uint32_t block_tiles; /**< position tiles per K/V block — the prefetch depth */
    uint32_t tile_bytes;  /**< 2048 Float16_b. The K/V CBs take the cache's tile instead when
                           *   placed through the `_kv` calls (block float, ft_tt_op_memory.h §1.6) */
} ft_tt_op_attention_residency;

/** L1 the residency's circular buffers occupy. FLAT IN THE CONTEXT LENGTH — that is the
 *  online softmax's whole payoff, and the number that makes 32k possible at all. */
size_t ft_tt_op_attention_l1_bytes(const ft_tt_op_attention_residency *residency);

/**
 * The largest `block_tiles` that fits the CB arena for this group shape.
 *
 * @return 0 if even one position tile per block does not fit
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_op_attention_max_block_tiles(uint32_t q_rows, uint32_t dim_tiles,
                                            uint32_t tile_bytes);

/**
 * Place the triad's sixteen circular buffers in the arena.
 *
 * K and V get TWO blocks of `block_tiles * dim_tiles` pages, so the reader fills one while
 * the compute thread drains the other — and two, rather than any other multiple, is what
 * makes `pages % depth == 0` hold (§5).
 *
 * @return FT_TT_OK; FT_TT_EINVAL; FT_TT_ENOMEM if it does not fit the arena
 */
int ft_tt_op_attention_place_cbs(const ft_tt_op_attention_residency *residency,
                                 ft_tt_launch_cb_entry out_entries[FT_TT_OP_ATTENTION_CB_COUNT],
                                 void *buffer, size_t buffer_bytes, uint64_t *out_mask);

/**
 * The same placement with the K and V CBs paged at `kv_tile_bytes`: the cache's tile, which
 * may be Bfp8_b (1088) or Bfp4_b (576). The program must be the matching block-float build
 * (its CB table and its reader's K/V tile size); nothing on the card checks that they agree.
 */
int ft_tt_op_attention_place_cbs_kv(const ft_tt_op_attention_residency *residency,
                                    uint32_t kv_tile_bytes,
                                    ft_tt_launch_cb_entry out_entries[FT_TT_OP_ATTENTION_CB_COUNT],
                                    void *buffer, size_t buffer_bytes, uint64_t *out_mask);

/** ft_tt_op_attention_l1_bytes() with the K/V CBs paged at `kv_tile_bytes`. */
size_t ft_tt_op_attention_l1_bytes_kv(const ft_tt_op_attention_residency *residency,
                                      uint32_t kv_tile_bytes);

/**
 * The placement for a MIXED cache: the K CB paged at `k_tile_bytes`, the V CB at
 * `v_tile_bytes` (each 2048, 4096, 1088 or 576). The program must be the matching mixed build
 * (`attention_k8v4`, `_k16v4`, `_k4v16`, `_k16v8`: its CB table and its reader's
 * ATTN_K_TILE_BYTES / ATTN_V_TILE_BYTES). `_kv` is this call with the two equal.
 */
int ft_tt_op_attention_place_cbs_mixed_kv(
    const ft_tt_op_attention_residency *residency, uint32_t k_tile_bytes, uint32_t v_tile_bytes,
    ft_tt_launch_cb_entry out_entries[FT_TT_OP_ATTENTION_CB_COUNT], void *buffer,
    size_t buffer_bytes, uint64_t *out_mask);

/** ft_tt_op_attention_l1_bytes() with the K CB paged at `k_tile_bytes`, the V CB at
 *  `v_tile_bytes`. */
size_t ft_tt_op_attention_l1_bytes_mixed_kv(const ft_tt_op_attention_residency *residency,
                                            uint32_t k_tile_bytes, uint32_t v_tile_bytes);

/* ================================================================
 * One call
 * ================================================================ */

/** One attention call: the group's query at `position`, against `kv_positions` cached
 *  positions of ONE KV head. Call it once per KV group per layer per token. */
typedef struct ft_tt_op_attention_request {
    /** DECODE: Q heads in this group. CHUNK: live positions C in this chunk, 1..32. */
    uint32_t q_rows;
    uint32_t dim_tiles;    /**< head_dim / 32 */
    uint32_t block_tiles;  /**< must equal the residency's */
    uint32_t kv_positions; /**< LIVE positions; the query attends to all of them */
    /** Position tiles the K (and V) region holds — `ft_tt_kv_cache_layout.capacity_tiles`.
     *  The padded read window may exceed the live length by up to `2*block_tiles - 1`
     *  tiles, and this is what stops that becoming a read past the region. */
    uint32_t capacity_tiles;
    /** Where this token's query row sits: `ft_tt_op_memory_position_row(position)`, and it
     *  is not optional — RoPE and the embedding put it there (§1.5b). */
    uint32_t query_row;
    /** FT_TT_OP_ATTENTION_FLAG_CHUNK, or 0 for an ordinary decode descriptor. */
    uint32_t flags;
    /** ROWS mode: strip rows served (query_row .. +positions), and the byte address of the
     *  staircase mask tile on the constants' coordinate. */
    uint32_t positions;
    uint32_t rows_mask_address;
    /** MERGE: partials folded, 1..FT_TT_OP_ATTENTION_MAX_PARTIALS (sent in word 2). */
    uint32_t partial_count;
    /** Q, already rotated: the FIRST tile of this group, i.e. head `group * q_rows`, dim
     *  tile 0, of the row block the token sits in. `address` carries the head offset.
     *  CHUNK: the first tile of ONE Q head, i.e. head `h`, dim tile 0. */
    ft_tt_op_operand query;
    /** This KV head's K region at position 0 — base + `ft_tt_kv_cache_run()`'s offset. */
    ft_tt_op_operand keys;
    /** This KV head's V region, same geometry (contract F1). */
    ft_tt_op_operand values;
    /** ft_tt_op_attention_build_constants()' buffer, built once per model. */
    ft_tt_op_operand constants;
    /** The attention output, in the same layout as `query`. */
    ft_tt_op_operand output;
} ft_tt_op_attention_request;

/** Position tiles the reader actually fetches: the live count rounded up to a whole number
 *  of DOUBLE blocks. The rounding is what keeps every descriptor's circular-buffer push and
 *  pop counts identical, which is the block-prefetch invariant's second condition. */
uint32_t ft_tt_op_attention_padded_tiles(uint32_t kv_positions, uint32_t block_tiles);

/** Position tiles that hold at least one live position: ceil(kv_positions / 32). */
uint32_t ft_tt_op_attention_live_tiles(uint32_t kv_positions);

/**
 * Encode `request` into the sixteen words a FT_TT_QUEUE_OP_ATTENTION descriptor points at.
 *
 * Pure: no device access, no allocation. That is what makes the wire format testable
 * off-hardware, which matters more here than usual — the card is the expensive place to
 * find a field in the wrong word.
 *
 * @param out_words  FT_TT_OP_ATTENTION_WORDS values, fully written (unused words zeroed)
 * @return           FT_TT_OK; FT_TT_EINVAL on a zero or oversized group, a zero head
 *                   dimension, a zero KV length, a query row past a tile, a tile size that
 *                   is not the cache's, or a padded read window past `capacity_tiles`; a
 *                   PARTIAL with CHUNK, a MERGE with any other mode bit but ROWS, or a MERGE
 *                   partial count of 0 or past FT_TT_OP_ATTENTION_MAX_PARTIALS
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_attention_parameters(const ft_tt_op_attention_request *request,
                                  uint32_t out_words[FT_TT_OP_ATTENTION_WORDS]);

/** Bytes one split-K partial occupies: (dim_tiles + 2) tiles of `tile_bytes`.
 *  Thread-safe: yes    Reentrant: yes    Signal-safe: yes */
uint32_t ft_tt_op_attention_partial_bytes(uint32_t dim_tiles, uint32_t tile_bytes);

/** Bytes ft_tt_op_attention_build_constants() writes. */
size_t ft_tt_op_attention_constants_bytes(uint32_t tile_bytes);

/**
 * Build the constants operand: the reduce scaler, the query scale, and all 32 tail masks.
 *
 * THE MASKS ARE BUILT ONCE PER MODEL AND SELECTED BY ARITHMETIC. Mask `t` is 0 in columns
 * below `t` and FT_TT_OP_ATTENTION_MASK_VALUE at and above it, so mask `kv_positions % 32`
 * kills the dead columns of the last live position tile and mask 0 — every column dead —
 * kills a tile entirely past the end. The kernel derives which one it needs from
 * `kv_positions`; nothing about the mask is on the wire. Uploading a per-call mask instead
 * would be 2 KiB per layer per token over a 287 MB/s link, i.e. ~7 µs of pure host cost per
 * layer that buys nothing.
 *
 * THE SCALER IS 1.0, FOR BOTH REDUCES. Upstream's reduce scaler is 1.0 for SUM and for MAX
 * alike (`reduce_helpers_dataflow.hpp`), and both of this op's reduces want a plain sum and
 * a plain max — so one tile serves both, exactly as in the REDUCE family.
 *
 * @param head_dim   the real head dimension in ELEMENTS: the query scale is 1/sqrt of it,
 *                   so a dim_tiles-derived value would silently be wrong for a head_dim
 *                   that is not a multiple of 32
 * @param out_bytes  at least ft_tt_op_attention_constants_bytes()
 * @return           FT_TT_OK, FT_TT_EINVAL
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_op_attention_build_constants(uint32_t head_dim, uint32_t tile_bytes, void *out,
                                       size_t out_bytes);

#endif /* FT_TT_OP_ATTENTION_H */
