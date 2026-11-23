/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_rope.h
 * @brief Host driver for ROPE — partial NeoX rotary position embedding. Places the triad's
 *   circular buffers, encodes the parameter block, and BUILDS THE SIN/COS TABLE that is the
 *   op's fourth operand.
 *
 * The device half is `kernels/resident_rope_protocol.h`; the two must agree word for word
 * and `tests/test_rope_ops.c` pins the pair off-hardware by parsing that header.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE OP IS, VALUE FOR VALUE
 * ---------------------------------------------------------------------------
 * The K80 reference is `fc_rope_neox_partial` (`foundry/cuda_kernels/rope.cu`), which mirrors
 * Pantheon's CPU oracle. For a head of `head_dim` features at token position `pos`, with
 * `rot = rope_dim` when `0 < rope_dim <= head_dim` and `head_dim` otherwise, and
 * `half = rot/2`:
 *
 *     freq_j  = theta_base^(-2j/rot)                       j in [0, half)
 *     angle   = pos * freq_j
 *     out[j]        = x[j]*cos(angle) - x[j+half]*sin(angle)
 *     out[j+half]   = x[j]*sin(angle) + x[j+half]*cos(angle)
 *     out[f]        = x[f]                                 f in [rot, head_dim)
 *
 * That last line is the whole of "partial". Both are one expression here:
 *
 *     out[c] = x[c] * COS[c] + x[partner(c)] * SIN[c]
 *
 * with `partner(c) = (c + half) mod rot`, `COS[c] = cos(angle_{c mod half})` and
 * **`SIN[c] = -sin(angle) below the half, +sin(angle) above it`**. The sign is in the TABLE,
 * which is what makes every rotated tile the same three FPU passes with no branch, no
 * scalar-broadcast circular buffer and no negation pass — where upstream's own kernel
 * (`ttnn/.../rotary_embedding_hf/.../rotary_embedding_hf.cpp`) spends an extra
 * multiply-by-(-1) on half its tiles.
 *
 * ---------------------------------------------------------------------------
 * THE FACE-SPACE ARGUMENT, IN ONE PARAGRAPH
 * ---------------------------------------------------------------------------
 * A tile is four 16x16 faces in FACE order, not flat row-major
 * (`docs/tt_kernel_model.md` §9). Activations are [tokens x features] with a feature row
 * along W, so feature `f` is intra-tile column `f % 32` of tile column `f / 32`. When
 * `half` is a multiple of 32, `j` and `j + half` therefore have the SAME intra-tile column
 * and the SAME row and differ ONLY in which tile they are in: **the pairing is a permutation
 * of TILES and the identity inside a tile.** No face crosses anything, and every operation
 * in the kernel is a tile-elementwise multiply or add, which is offset-wise and so
 * face-order agnostic. `kernels/resident_rope_protocol.h` carries the derivation.
 *
 * What IS face-critical is this file's TABLE: its value varies with both the row (position)
 * and the column (frequency), so it must be tilized with the SAME map as the activations, and
 * `ft_tt_op_rope_build_table()` uses `ft_tt_tilize_strip()` to guarantee it.
 *
 * Refused, rather than approximated (`ft_tt_op_rope_shape_is_supported()`):
 *   - `rot_tiles` odd, i.e. `half % 32 == 16`: the partner sits in a different FACE COLUMN.
 *     Buildable — a face is 256 contiguous elements, so a reader could assemble the partner
 *     tile from four face-sized NoC reads — but not built.
 *   - `rot` not a multiple of 32 at all: an intra-face column offset no NoC read expresses.
 *   - INTERLEAVED RoPE (GGUF `llama`/NORM style, pairs `(2i, 2i+1)`): adjacent columns
 *     inside one face row, not a tile permutation at any granularity. **And this is the style
 *     MiniCPM5-1B actually takes** in Pantheon (`llama_forward_cuda.c` sends PN_ARCH_LLAMA
 *     through `llama_rope_raw`). The fix is offline and free: permute the `wq`/`wk` output
 *     rows in the converter so the activations arrive in split-half order — llama.cpp's own
 *     HF->GGUF permutation, run backwards — after which interleaved RoPE IS this kernel.
 *     `Q.K` is invariant under a permutation applied to BOTH Q and K, so it changes no math.
 *
 * ---------------------------------------------------------------------------
 * THE TABLE: PRECOMPUTED HOST-SIDE, UPLOADED ONCE, HELD
 * ---------------------------------------------------------------------------
 * It cannot live in kernel-local memory — TRISC2 has 1760 bytes (dev notes §5) and one row
 * block of MiniCPM5-1B's table is 16 KiB — so it is a DRAM operand streamed through circular
 * buffers. The remaining question is host-precomputed versus computed on the card, and there
 * are two independent reasons and one non-reason:
 *
 *  1. **PRECISION, which is decisive.** `angle = pos * freq` reaches `131071 * 1.0` radians at
 *     the model's context length. Computing `cos` of that on the SFPU needs the argument
 *     reduced mod 2*pi, and the reduction needs ~18 bits of integer part before any fraction:
 *     bf16 has 8 mantissa bits and cannot hold it at all, fp32 has 24 and holds it to ~0.008
 *     rad. The host reduces in double and is exact. This is not a cost trade — the on-card
 *     version is WRONG at long context, and long context is the whole point of the card.
 *  2. **BANDWIDTH, which agrees but does not decide.** The whole 131072-position table is
 *     67.1 MB (two bf16 tables of 131072 x 128), which is 234 ms ONCE at the measured
 *     287 MB/s host->card (dev notes §6) and then never again — Pantheon loads and holds. A
 *     per-token upload of the 8 tiles a decode step needs is 16 KiB, i.e. ~56 us per token
 *     forever, and it puts a host round trip inside the decode loop, which is the shape spec
 *     §9.7 exists to avoid.
 *  3. The non-reason: SFPU cost. Not measured, and it does not matter, because (1) settles it.
 *
 * MEASURED here: 287 MB/s, and the table's byte count (arithmetic). INFERRED, not measured:
 * that the SFPU's trig would be unusable at 1.3e5 radians — that is read off the format's
 * mantissa width, not off the card.
 *
 * **The bf16 table costs about 2^-9 of relative angle amplitude**, which is the same order as
 * the bf16 activations it multiplies. An fp32 table is a mixed-format CB contract away
 * (`rope_float16b.cbtable` says where) and doubles to 134 MB; do it if a measurement asks.
 *
 * ---------------------------------------------------------------------------
 * THE ROW-PHASE CONTRACT — READ THIS BEFORE CALLING
 * ---------------------------------------------------------------------------
 * The kernel does not know what a position is. It multiplies activation tile row `r` by table
 * tile row `r`. So:
 *
 *   **the token in activation row `r` must be at the position the table has in row `r`.**
 *
 * The table is built for positions `first_position + 32*b + r`, `first_position % 32 == 0`.
 * Therefore:
 *   - PREFILL, 32 tokens at positions `p0 .. p0+31` with `p0 % 32 == 0`, in rows 0..31:
 *     aligned for free, point at `ft_tt_op_rope_table_offset(p0, ...)`.
 *   - DECODE, ONE token at position `p`: it must sit in **row `p % 32`** of its tile, not row
 *     0, and the table operand is `ft_tt_op_rope_table_offset(p, ...)`. That row choice costs
 *     nothing and is made ONCE per token, upstream: the embedding op writes the token's
 *     hidden state at row `p % 32` and every downstream tensor keeps the phase, because a
 *     matmul's output row is its input row. The alternative — a per-call table tile broadcast
 *     from the host — is the 56 us/token round trip above.
 *
 * `tests/test_phase6_rope.c` runs a decode case at `p = 7` precisely so that a kernel which
 * silently assumed row 0 goes red.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_ROPE_H
#define FT_TT_OP_ROPE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_launch.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_queue.h"

/* ================================================================
 * The parameter block — mirrors kernels/resident_rope_protocol.h
 * ================================================================ */

/** Words of one ROPE parameter block. A word this op does not use is written 0 by the host
 *  and never read by the kernel. */
typedef enum ft_tt_op_rope_field {
    FT_TT_OP_ROPE_TILE_COUNT = 0, /**< TOTAL tiles in and out — all the writer reads */
    FT_TT_OP_ROPE_HEAD_TILES = 1, /**< head_dim / 32 */
    FT_TT_OP_ROPE_ROT_TILES = 2,  /**< rot / 32; must be EVEN; tiles past it pass through */
    FT_TT_OP_ROPE_HEADS = 3,      /**< heads per row block; the table advances per row block */
    FT_TT_OP_ROPE_DEPTH = 4,      /**< reader block depth; divides head_tiles AND rot_tiles */
    FT_TT_OP_ROPE_A_X = 5,
    FT_TT_OP_ROPE_A_Y = 6,
    FT_TT_OP_ROPE_A_BASE = 7,
    FT_TT_OP_ROPE_TABLE_X = 8,
    FT_TT_OP_ROPE_TABLE_Y = 9,
    FT_TT_OP_ROPE_TABLE_BASE = 10, /**< already offset to this call's first row block */
    FT_TT_OP_ROPE_C_X = 11,
    FT_TT_OP_ROPE_C_Y = 12,
    FT_TT_OP_ROPE_C_BASE = 13,
    FT_TT_OP_ROPE_WORDS = 16
} ft_tt_op_rope_field;

/** CB indices, matching the compile-time args the RoPE triad is built with. */
#define FT_TT_OP_ROPE_CB_IN       0u
#define FT_TT_OP_ROPE_CB_ROT      1u
#define FT_TT_OP_ROPE_CB_COS      2u
#define FT_TT_OP_ROPE_CB_SIN      3u
#define FT_TT_OP_ROPE_CB_PROD_COS 4u
#define FT_TT_OP_ROPE_CB_PROD_SIN 5u
#define FT_TT_OP_ROPE_CB_OUT      16u
#define FT_TT_OP_ROPE_CB_COUNT    7u

/** MiniCPM5-1B, read from `MiniCPM5-1B-Q8_0.gguf`'s header on the card host 2026-08-12 — not
 *  assumed, and not inherited from another model. `llama.rope.dimension_count` equals
 *  `llama.attention.key_length`, so for THIS model the rotary dimension is the WHOLE head and
 *  the partial path is degenerate; the op still implements it because Qwen3.6-class models
 *  rotate 64 of 256 and because a pass-through that is never exercised is a pass-through that
 *  does not work. */
#define FT_TT_OP_ROPE_MINICPM5_HEAD_DIM   128u
#define FT_TT_OP_ROPE_MINICPM5_ROPE_DIM   128u
#define FT_TT_OP_ROPE_MINICPM5_THETA      5000000.0
#define FT_TT_OP_ROPE_MINICPM5_CONTEXT    131072u
#define FT_TT_OP_ROPE_MINICPM5_HEADS_Q    16u
#define FT_TT_OP_ROPE_MINICPM5_HEADS_KV   2u

/** One RoPE call: `row_blocks` blocks of 32 tokens, `heads` heads each, `head_dim` features
 *  per head, the first `rot` of them rotated at positions starting at `first_position`. */
typedef struct ft_tt_op_rope_request {
    uint32_t row_blocks;      /**< tokens / 32 */
    uint32_t heads;           /**< heads in one row block (16 for Q, 2 for KV) */
    uint32_t head_dim;        /**< features per head; a multiple of 32 */
    uint32_t rope_dim;        /**< rotated features; 0 means the whole head. See rot above */
    uint32_t first_position;  /**< position of activation ROW 0; see THE ROW-PHASE CONTRACT */
    uint32_t prefetch_tiles;  /**< reader block depth, shrunk to a divisor of both runs */
    ft_tt_op_operand input;
    /** The table, at its BASE. This driver adds `ft_tt_op_rope_table_offset(first_position)`
     *  so a caller cannot forget to. */
    ft_tt_op_operand table;
    ft_tt_op_operand output;
} ft_tt_op_rope_request;

/**
 * The rotary width in features: `rope_dim` when `0 < rope_dim <= head_dim`, else `head_dim`.
 *
 * Transliterated from `llama_rope_neox()` in `Pantheon/src/models/llama_forward_cuda.c` and
 * `fc_rope_neox_partial` in `foundry/cuda_kernels/rope.cu`. Note the two differ: the CUDA
 * kernel writes `rope_dim < head_dim` and the CPU path `rope_dim <= head_dim`, which is the
 * same value everywhere except at `rope_dim == head_dim`, where both then use the whole head
 * anyway. This follows the CPU path, which is Pantheon's oracle.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_op_rope_rotary_width(uint32_t head_dim, uint32_t rope_dim);

/**
 * Is this shape one the tile-aligned pairing can express?
 *
 * THE ONLY definition, and the kernel carries the same three clauses in
 * `ftr_shape_is_supported()` (tests/test_rope_ops.c asserts the two agree over a sweep). A
 * shape that fails this is FT_TT_EINVAL, never rotated with a plausible-looking partner.
 *
 * @param head_tiles  head_dim / 32
 * @param rot_tiles   rot / 32
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool ft_tt_op_rope_shape_is_supported(uint32_t head_tiles, uint32_t rot_tiles);

/**
 * Encode `request` into the sixteen words a FT_TT_QUEUE_OP_ROPE descriptor points at.
 *
 * Pure: no device access, no allocation.
 *
 * @param out_words  FT_TT_OP_ROPE_WORDS values, fully written (unused words zeroed)
 * @return           FT_TT_OK; FT_TT_EINVAL on a zero shape, a head_dim or rope_dim that is
 *                   not a multiple of 32, an odd rot_tiles (the face-space refusal), a zero
 *                   tile size, or a MULTI-row-block call whose `first_position` is not a
 *                   multiple of 32. A SINGLE row block may start anywhere — that is the
 *                   decode case: pass the token's real position `p`, place the token in tile
 *                   row `p % 32`, and this driver derives the table row block. Two or more
 *                   row blocks cannot express a phase, because row 0 of the second block is
 *                   position `p - p%32 + 32`, so it is refused rather than rounded.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_rope_parameters(const ft_tt_op_rope_request *request,
                             uint32_t out_words[FT_TT_OP_ROPE_WORDS]);

/**
 * Place the RoPE triad's seven circular buffers in the arena.
 *
 * The four input CBs get TWO BLOCKS of the effective depth each, so the reader fills one
 * while the compute thread drains the other. The two staging CBs and the output are two
 * pages, because they are produced and consumed a tile at a time.
 *
 * @param out_entries   FT_TT_OP_ROPE_CB_COUNT entries, with addresses filled in
 * @param buffer        receives the encoded CB descriptor array
 * @param out_mask      receives the launch message's local_cb_mask
 * @return              FT_TT_OK; FT_TT_EINVAL; FT_TT_ENOMEM if it does not fit the arena
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_rope_place_cbs(const ft_tt_op_rope_request *request,
                            ft_tt_launch_cb_entry out_entries[FT_TT_OP_ROPE_CB_COUNT],
                            void *buffer, size_t buffer_bytes, uint64_t *out_mask);

/* ================================================================
 * The table
 * ================================================================ */

/** Bytes of one row block of the table: `rot_tiles` cos tiles then `rot_tiles` signed-sin
 *  tiles, covering 32 consecutive positions. */
uint32_t ft_tt_op_rope_table_row_block_bytes(uint32_t rot_tiles, uint32_t tile_bytes);

/** Bytes of a table covering `position_count` positions (rounded up to whole row blocks). */
size_t ft_tt_op_rope_table_bytes(uint32_t position_count, uint32_t rot_tiles,
                                 uint32_t tile_bytes);

/** Byte offset, within a table built from position 0, of the row block holding `position`.
 *  The token itself must be in tile row `position % 32` — THE ROW-PHASE CONTRACT. */
uint32_t ft_tt_op_rope_table_offset(uint32_t position, uint32_t rot_tiles,
                                    uint32_t tile_bytes);

/**
 * The rotation angle for feature index `j` at `position`.
 *
 * `angle = position * theta_base^(-2j/rot)`, in DOUBLE — see the table note above. `j` is
 * taken modulo `rot/2`, which is what makes a table column `c >= half` carry the same angle
 * as column `c - half`.
 *
 * Exposed because it is the one number a test can check directly: rotate the basis vector
 * `(1, 0)` and the output IS `(cos, sin)`. A check on the norm cannot see a wrong angle at
 * all, because a rotation preserves it.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
double ft_tt_op_rope_angle(uint32_t position, uint32_t j, uint32_t rot, double theta_base);

/**
 * Build the bf16 sin/cos table for `position_count` positions starting at `first_position`.
 *
 * Layout, per row block of 32 positions: `rot_tiles` tiles of `cos(angle)` then `rot_tiles`
 * tiles of SIGNED `sin(angle)` — negative below the half, positive above it. Both are tilized
 * with `ft_tt_tilize_strip()`, i.e. with the SAME face map the activations use, because the
 * table's value varies along both axes and a face-permuted table assigns the wrong ANGLE
 * rather than permuting the answer.
 *
 * Called ONCE per model load for the whole context (67.1 MB at MiniCPM5-1B's 131072 x 128),
 * which is why it may allocate and why nothing here is on a per-token path.
 *
 * @param first_position  MUST be a multiple of 32 (the row-phase contract)
 * @param position_count  MUST be a non-zero multiple of 32
 * @param head_dim        features per head, to resolve `rope_dim` the way the model does
 * @param rope_dim        rotated features; 0 means the whole head
 * @param theta_base      `rope.freq_base` — 5e6 for MiniCPM5-1B, READ from the artifact
 * @param out             receives ft_tt_op_rope_table_bytes() bytes
 * @param out_bytes       its size
 * @return                FT_TT_OK; FT_TT_EINVAL on a bad shape or a short buffer;
 *                        FT_TT_ENOMEM
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (allocates)
 */
int ft_tt_op_rope_build_table(uint32_t first_position, uint32_t position_count,
                              uint32_t head_dim, uint32_t rope_dim, double theta_base,
                              void *out, size_t out_bytes);

#endif /* FT_TT_OP_ROPE_H */
