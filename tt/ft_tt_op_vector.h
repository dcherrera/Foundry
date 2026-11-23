/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_vector.h
 * @brief Host driver for the VECTOR op family — the elementwise triad (SiLU, sigmoid,
 *   add, mul, sub, SiLU-gated multiply) and the REDUCE triad (softmax, RMS norm, layer
 *   norm, router normalise). Places their circular buffers, encodes their parameter
 *   blocks, builds the reduce family's constant tiles.
 *
 * The device half is `kernels/resident_vector_protocol.h` plus
 * `kernels/resident_reduce_protocol.h`; the halves must agree word for word and
 * `tests/test_vector_ops.c` pins the pair off-hardware.
 *
 * ---------------------------------------------------------------------------
 * WHAT "THE VECTOR FAMILY" IS, AND WHY IT IS ONE MODULE
 * ---------------------------------------------------------------------------
 * One input tile stream (sometimes two), one output tile stream, the same tile count in as
 * out, activations laid out [tokens x features] so a feature row lies along a tile's W —
 * which is the orientation ft_tt_op_matmul.h settled and everything downstream inherits.
 * That description covers most of what Phase 6 still owes: softmax, SiLU, sigmoid, add,
 * mul, RoPE, the MoE weighted add, the gated norms. Matmul is the exception.
 *
 * They share this file because they share a parameter block and a WRITER, and because the
 * interesting decision — which ops are one kernel and which are separate — is a statement
 * about the family rather than about any member of it.
 *
 * ---------------------------------------------------------------------------
 * THE DECISION: TWO BINARIES — SIX ELEMENTWISE FUNCTIONS, FOUR REDUCTIONS
 * ---------------------------------------------------------------------------
 * SiLU and sigmoid differ by one SFPU verb. Add, mul and sub differ by one FPU opcode.
 * None of that is what decides it. **What decides it is that a resident binary is linked
 * at a fixed per-processor slab, so only ONE program can be resident on a core at a time**
 * (docs/tt_bringup_log.md §17.8 note 4). A core that runs a SwiGLU has to do silu and then
 * mul; if those are two programs it reloads five kernel images between them, and image
 * upload over the x1 link — not dispatch — is what the 0.245 ms fixed cost turned out to
 * be (§20.4: ~0.19 ms of it). So the grouping question is "what does one core have to do
 * in sequence", and for elementwise the answer is all of it.
 *
 * The same question answers the REDUCE family the same way. Softmax, RMS norm, layer norm
 * and the router normalise are one skeleton — reduce along the row into DST column 0, one
 * SFPU verb on that column, broadcast it back with *_tiles_bcast_cols — differing in a
 * pass count and two verbs, and a transformer block does RMS norm, then attention's
 * softmax, then RMS norm again ON ONE CORE. As separate binaries that is two image reloads
 * per block. So they are ONE binary with the pass count and the verbs as descriptor
 * fields, and RMS norm's Phase 4 one-shot triad is superseded by it.
 *
 * ELTWISE and REDUCE stay two binaries because nothing sequences a softmax between a silu
 * and a mul, and because the reduce contract is different in kind (eleven CBs, three of
 * them a whole row deep). They already share this file, the parameter block and the
 * WRITER; merging the two compute sources would be a source change and not a re-layout,
 * and the only question would be the 12 KiB text slab.
 *
 * ---------------------------------------------------------------------------
 * THE ONE FUSION, AND WHY IT IS NOT THE K80's KIND
 * ---------------------------------------------------------------------------
 * FT_TT_OP_VECTOR_FN_SILU_MUL computes silu(a)*b in one descriptor. docs/tt_bringup_log.md
 * §17.8 note 1 is explicit that fusing to save DISPATCH is porting a K80 workaround for a
 * problem this backend does not have — dispatch is 0.39 µs of device time. This fusion is
 * licensed by the other half of that sentence: as two descriptors it costs a full DRAM
 * round trip of the intermediate, and as one it costs an L1 round trip through a scratch
 * CB page. It avoids a round trip, which is the stated reason to fuse.
 *
 * @status Green on silicon, Tensix 20, 0 checks failed: six elementwise functions, and
 *   softmax / RMS norm / layer norm / router normalise on ONE reduce residency. Softmax
 *   reproduced its pre-merge numbers exactly (Wt=1 0.999983709, Wt=48 0.999985087,
 *   Wt=128 0.999980702) and RMS norm BEAT the Phase 4 one-shot it replaces (max-abs
 *   0.01046 against 0.02381 on identical input), because the merged cbtable carries
 *   dst_accum_mode 1 and the one-shot did not.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_VECTOR_H
#define FT_TT_OP_VECTOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_launch.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_queue.h"

/* ================================================================
 * The parameter block — mirrors kernels/resident_vector_protocol.h
 * ================================================================ */

/** Words of one vector-family parameter block. A word an op does not use is written 0 by
 *  the host and never read by the kernel; it is NOT free for that op to redefine, because
 *  a block whose meaning depends on which kernel is resident is a wrong answer with no
 *  error anywhere. */
typedef enum ft_tt_op_vector_field {
    FT_TT_OP_VECTOR_FUNCTION = 0,   /**< ELTWISE: which function. SOFTMAX: 0 */
    FT_TT_OP_VECTOR_TILE_COUNT = 1, /**< TOTAL tiles in and out — all the writer reads */
    FT_TT_OP_VECTOR_TILE_COLS = 2,  /**< softmax Wt; ELTWISE: flags (FT_TT_OP_ELTWISE_FLAG_*) */
    FT_TT_OP_VECTOR_DEPTH = 3,      /**< reader block depth; divides the run it blocks over */
    FT_TT_OP_VECTOR_A_X = 4,
    FT_TT_OP_VECTOR_A_Y = 5,
    FT_TT_OP_VECTOR_A_BASE = 6,
    FT_TT_OP_VECTOR_B_X = 7,     /**< eltwise: B stream. softmax: the ONE scaler tile */
    FT_TT_OP_VECTOR_B_Y = 8,
    FT_TT_OP_VECTOR_B_BASE = 9,
    FT_TT_OP_VECTOR_C_X = 10,
    FT_TT_OP_VECTOR_C_Y = 11,
    FT_TT_OP_VECTOR_C_BASE = 12,
    FT_TT_OP_VECTOR_WORDS = 16
} ft_tt_op_vector_field;

/** Which function an ELTWISE descriptor applies. Values are wire values. */
typedef enum ft_tt_op_vector_function {
    FT_TT_OP_VECTOR_FN_SILU = 1,     /**< y = x * sigmoid(x)   unary */
    FT_TT_OP_VECTOR_FN_SIGMOID = 2,  /**< y = sigmoid(x)       unary */
    FT_TT_OP_VECTOR_FN_ADD = 3,      /**< y = a + b            binary */
    FT_TT_OP_VECTOR_FN_MUL = 4,      /**< y = a * b            binary */
    FT_TT_OP_VECTOR_FN_SUB = 5,      /**< y = a - b            binary */
    FT_TT_OP_VECTOR_FN_SILU_MUL = 6  /**< y = silu(a) * b      binary, the SwiGLU gate */
} ft_tt_op_vector_function;

/**
 * Does `function` take a second input stream?
 *
 * THE ONLY definition of arity, host side and device side both (the kernel has its own
 * copy of this one line and tests/test_vector_ops.c asserts the two agree for every
 * function). The host deliberately does not SEND an arity field: two fields that must
 * agree are two fields that can disagree, and the disagreement here would be a reader
 * fetching a B tile nobody wrote — a plausible wrong answer, not an error.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool ft_tt_op_vector_function_is_binary(ft_tt_op_vector_function function);

/** "SILU", "ADD", … for logs. Never NULL. */
const char *ft_tt_op_vector_function_name(ft_tt_op_vector_function function);

/* ================================================================
 * Elementwise
 * ================================================================ */

/** CB indices, matching the compile-time args the elementwise triad is built with. */
#define FT_TT_OP_ELTWISE_CB_IN0     0u
#define FT_TT_OP_ELTWISE_CB_IN1     1u
#define FT_TT_OP_ELTWISE_CB_SCRATCH 2u
#define FT_TT_OP_ELTWISE_CB_OUT     16u
#define FT_TT_OP_ELTWISE_CB_COUNT   4u

/** One elementwise call: `tile_count` tiles of A (and of B, if binary) through `function`
 *  into `output`. Shape is a tile count and nothing else — an elementwise op has no row
 *  structure to get wrong. */
typedef struct ft_tt_op_eltwise_request {
    ft_tt_op_vector_function function;
    uint32_t tile_count;
    /** K-tiles per NoC barrier in the reader. Shrunk to a divisor of tile_count by
     *  ft_tt_op_block_depth(); 0 or 1 is the per-tile barrier, which is always correct and
     *  is what cost 5.9x on the matmul. */
    uint32_t prefetch_tiles;
    ft_tt_op_operand input_a;
    ft_tt_op_operand input_b; /**< ignored for a unary function */
    ft_tt_op_operand output;
    /** Nonzero: silu/sigmoid compute ROW 0 of each tile only (a decode strip's live row);
     *  rows 4-31 of the output are the input copied, not the function. FPU verbs ignore it. */
    uint32_t row0;
} ft_tt_op_eltwise_request;

/** Word-2 flag of an ELTWISE descriptor. Mirrors kVecEltwiseRow0 -- change both or neither. */
#define FT_TT_OP_ELTWISE_FLAG_ROW0 1u

/**
 * Encode `request` into the sixteen words a FT_TT_QUEUE_OP_ELTWISE descriptor points at.
 *
 * Pure: no device access, no allocation. That is what makes the wire format testable
 * off-hardware, which matters more here than usual — the card is the expensive place to
 * find a field in the wrong word.
 *
 * @param out_words  FT_TT_OP_VECTOR_WORDS values, fully written (unused words zeroed)
 * @return           FT_TT_OK; FT_TT_EINVAL on a zero tile count, an unknown function, a
 *                   zero tile size, or a binary function with a zero-sized B operand
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_eltwise_parameters(const ft_tt_op_eltwise_request *request,
                                uint32_t out_words[FT_TT_OP_VECTOR_WORDS]);

/**
 * Place the elementwise triad's four circular buffers in the arena.
 *
 * The two input CBs get TWO BLOCKS of the effective prefetch depth, so the reader fills
 * one while the compute thread drains the other. cb_in1 is placed even for a unary
 * function: a spare bf16 input CB is a few KiB of a 1.29 MiB arena, and the alternative is
 * a host-side arity table that has to agree with the kernel's.
 *
 * @param out_entries   FT_TT_OP_ELTWISE_CB_COUNT entries, with addresses filled in
 * @param buffer        receives the encoded CB descriptor array
 * @param out_mask      receives the launch message's local_cb_mask
 * @return              FT_TT_OK; FT_TT_EINVAL; FT_TT_ENOMEM if it does not fit the arena
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_eltwise_place_cbs(const ft_tt_op_eltwise_request *request,
                               ft_tt_launch_cb_entry out_entries[FT_TT_OP_ELTWISE_CB_COUNT],
                               void *buffer, size_t buffer_bytes, uint64_t *out_mask);

/* ================================================================
 * The REDUCE family — softmax, RMS norm, layer norm, router normalise
 * ================================================================ */

/** CB indices, matching the compile-time args the reduce triad is built with. */
#define FT_TT_OP_REDUCE_CB_IN     0u
#define FT_TT_OP_REDUCE_CB_SCALER 1u
#define FT_TT_OP_REDUCE_CB_COL0   2u
#define FT_TT_OP_REDUCE_CB_COL1   3u
#define FT_TT_OP_REDUCE_CB_MID    4u
#define FT_TT_OP_REDUCE_CB_EPS    5u
#define FT_TT_OP_REDUCE_CB_GAMMA  6u
#define FT_TT_OP_REDUCE_CB_BETA   7u
#define FT_TT_OP_REDUCE_CB_STAGE0 8u
#define FT_TT_OP_REDUCE_CB_STAGE1 9u
#define FT_TT_OP_REDUCE_CB_OUT    16u
#define FT_TT_OP_REDUCE_CB_COUNT  11u

/** The three program words of a reduce parameter block, on top of the thirteen the vector
 *  family defines. Mirrors kernels/resident_reduce_protocol.h. */
#define FT_TT_OP_REDUCE_PASS0 13u
#define FT_TT_OP_REDUCE_PASS1 14u
#define FT_TT_OP_REDUCE_FINAL 15u

/** Byte positions inside a program word. */
#define FT_TT_OP_REDUCE_BYTE_POOL       0u
#define FT_TT_OP_REDUCE_BYTE_MAP        1u
#define FT_TT_OP_REDUCE_BYTE_VERB       2u
#define FT_TT_OP_REDUCE_BYTE_ADD_EPS    3u
#define FT_TT_OP_REDUCE_BYTE_PASS_COUNT 0u
#define FT_TT_OP_REDUCE_BYTE_SOURCE     1u
#define FT_TT_OP_REDUCE_BYTE_CHAIN      2u

/** Wire values. 0 is deliberately not a pool, so a zeroed block is an invalid program. */
#define FT_TT_OP_REDUCE_POOL_SUM 1u
#define FT_TT_OP_REDUCE_POOL_MAX 2u

#define FT_TT_OP_REDUCE_MAP_NONE        0u
#define FT_TT_OP_REDUCE_MAP_SQUARE_SELF 1u
#define FT_TT_OP_REDUCE_MAP_SUB_EXP     2u
#define FT_TT_OP_REDUCE_MAP_SUB_SQUARE  3u

#define FT_TT_OP_REDUCE_VERB_NONE  0u
#define FT_TT_OP_REDUCE_VERB_RECIP 1u
#define FT_TT_OP_REDUCE_VERB_RSQRT 2u

#define FT_TT_OP_REDUCE_SOURCE_IN  0u
#define FT_TT_OP_REDUCE_SOURCE_MID 1u

#define FT_TT_OP_REDUCE_CHAIN_CENTER 0x1u
#define FT_TT_OP_REDUCE_CHAIN_GAMMA  0x2u
#define FT_TT_OP_REDUCE_CHAIN_SCALE  0x4u
#define FT_TT_OP_REDUCE_CHAIN_BETA   0x8u

/** Constants-operand tile layout. Fixed for every member: a layout that moved with the
 *  flags would be a wrong address for one descriptor in four. */
#define FT_TT_OP_REDUCE_CONST_SCALER_TILE 0u
#define FT_TT_OP_REDUCE_CONST_EPS_TILE    1u
#define FT_TT_OP_REDUCE_CONST_STREAM_TILE 2u

/** P3 SCHEDULE: word 0 of a reduce block (the vector family's FUNCTION slot, unused by
 *  reduce). It says WHEN tiles move and are computed, never what is computed — 0 is the
 *  pre-P3 schedule, and every schedule is bit-exact against it. Mirrors
 *  kernels/resident_reduce_protocol.h (kRedSchedule and the kRedSched* bits). */
#define FT_TT_OP_REDUCE_SCHEDULE           0u
#define FT_TT_OP_REDUCE_SCHED_CHUNK_MASK   0xFFu  /**< final-chain tiles per step run */
#define FT_TT_OP_REDUCE_SCHED_STREAM       0x100u /**< pipelined input, gamma after row 0 */
#define FT_TT_OP_REDUCE_SCHED_HALF_READ    0x200u /**< faces 0-1 of input and gamma only */
#define FT_TT_OP_REDUCE_SCHED_RED_FACE0    0x400u /**< RED: a half read lands face 0 only */
#define FT_TT_OP_REDUCE_CHAIN_CHUNK_MAX    255u

/**
 * Which member of the family a descriptor asks for.
 *
 * These values are NOT on the wire. What goes on the wire is the PROGRAM the member
 * compiles to — a pass count, two pass words and a final-chain word — and the kernel reads
 * nothing else, so there is no function id beside the program to disagree with it (dev
 * notes §3 has the catalogue of what two fields that must agree have cost here).
 */
typedef enum ft_tt_op_reduce_function {
    /** y = exp(x - max) / sum exp(x - max). Two passes: MAX, then SUM with a reciprocal. */
    FT_TT_OP_REDUCE_FN_SOFTMAX = 1,
    /** y = x * gamma * rsqrt(mean(x^2) + eps). One pass: SUM of squares with an rsqrt. */
    FT_TT_OP_REDUCE_FN_RMS_NORM = 2,
    /** y = (x - mean) * gamma * rsqrt(mean((x - mean)^2) + eps) + beta. Two passes. */
    FT_TT_OP_REDUCE_FN_LAYER_NORM = 3,
    /** y = x / sum x — the MoE router's renormalise over the selected experts' gates. One
     *  pass: SUM with a reciprocal. The cheapest member, and the one that shows the family
     *  is a family: it is softmax's second pass and nothing else. */
    FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE = 4,
    /**
     * y = x * rsqrt(sum(x^2) + eps) — the Gated-Delta-Net's per-head q/k norm. One pass,
     * like RMS norm, and it differs from it in exactly two table cells: the reduce scaler
     * is 1.0 rather than 1/W (a SUM, not a MEAN) and the final chain carries no gamma.
     *
     * **This is an L2 norm and NOT an RMS norm.** Pantheon's `pn_qwen35moe_l2_norm()` is the
     * definition and divides by `max(||x||, eps)` — the vector magnitude with no 1/n factor,
     * and a CLAMP rather than an epsilon under the root. Folding `eps*eps` into the sum is
     * the same function to fp32 wherever `sum(x^2) >> eps*eps`, and lands on exactly `1/eps`
     * where the clamp does, so **the caller passes `eps*eps` here, never `eps`**.
     */
    FT_TT_OP_REDUCE_FN_L2_NORM = 5
} ft_tt_op_reduce_function;

/** "SOFTMAX", "RMS_NORM", … for logs. Never NULL. */
const char *ft_tt_op_reduce_function_name(ft_tt_op_reduce_function function);

/** How many reduce passes this member runs: 1 or 2, and 0 for an unknown function. */
uint32_t ft_tt_op_reduce_pass_count(ft_tt_op_reduce_function function);

/** Does this member multiply by a per-column gamma? (RMS norm, layer norm.) */
bool ft_tt_op_reduce_uses_gamma(ft_tt_op_reduce_function function);

/** Does this member add a per-column beta? (Layer norm.) */
bool ft_tt_op_reduce_uses_beta(ft_tt_op_reduce_function function);

/** Does this member fold eps into a reduce? (RMS norm, layer norm.) */
bool ft_tt_op_reduce_uses_eps(ft_tt_op_reduce_function function);

/** Is this member's reduce scaler 1/W rather than 1.0? True exactly when a pass wants a
 *  MEAN rather than a SUM or a MAX. */
bool ft_tt_op_reduce_scaler_is_mean(ft_tt_op_reduce_function function);

/**
 * The three program words this member compiles to.
 *
 * Pure, and the single source of truth for what the kernel will do — which is why
 * tests/test_vector_ops.c pins all four members' words as LITERALS rather than
 * re-deriving them. A derived pin re-derives the mistake and passes (dev notes §3).
 *
 * @param out_pass0  the first pass's program word
 * @param out_pass1  the second's, or 0 for a one-pass member
 * @param out_final  pass count, final-chain source and the chain's step flags
 * @return           FT_TT_OK, or FT_TT_EINVAL for an unknown function
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_reduce_program(ft_tt_op_reduce_function function, uint32_t *out_pass0,
                            uint32_t *out_pass1, uint32_t *out_final);

/**
 * What ONE resident reduce program is configured to serve.
 *
 * The circular buffers are placed once, at launch, and then serve every descriptor — so
 * the residency, not the descriptor, is what fixes the deepest row and whether gamma and
 * beta are carried at all. A residency sized for the union of the members a core will run
 * is what makes the merge worth having: four ops, ONE image load.
 *
 * A stream the residency does not carry is still PLACED, at one page. A descriptor that
 * asks for it then blocks in cb_reserve_back and is retired by its bound, instead of the
 * reader writing into a circular buffer the host never configured — which would be a NoC
 * write to an address nobody chose (spec §7.3).
 */
typedef struct ft_tt_op_reduce_residency {
    uint32_t max_tile_cols; /**< the deepest Wt any descriptor will use */
    uint32_t tile_bytes;    /**< 2048 for Float16_b */
    bool carries_gamma;
    bool carries_beta;
    /** P3: pages in EACH final-chain scratch CB, and in cb_out when that is more than its
     *  two; 0 means 1. The largest `chain_chunk` a descriptor on this residency may ask
     *  for — the reader refuses a larger one. */
    uint32_t stage_pages;
    /** P3: CB_IN depth in whole rows of `max_tile_cols`; 0 means 1. How many row blocks the
     *  reader may run ahead of the compute thread. Every descriptor on a residency with more
     *  than one row MUST carry `max_tile_cols` itself, or a row straddles the wrap. */
    uint32_t input_rows;
} ft_tt_op_reduce_residency;

/** Fill a residency that serves every member in `functions`. */
int ft_tt_op_reduce_residency_for(const ft_tt_op_reduce_function *functions, size_t count,
                                  uint32_t max_tile_cols, uint32_t tile_bytes,
                                  ft_tt_op_reduce_residency *out_residency);

/** One reduce call: `row_blocks` blocks of 32 rows, each `tile_cols` tiles wide. W must be
 *  a whole number of tiles — there is no partial-tile mask here, and every dimension
 *  MiniCPM5-1B reduces over is a multiple of 32. */
typedef struct ft_tt_op_reduce_request {
    ft_tt_op_reduce_function function;
    uint32_t row_blocks; /**< rows / 32 */
    uint32_t tile_cols;  /**< Wt = W / 32 */
    uint32_t prefetch_tiles;
    ft_tt_op_operand input;
    /** The constants buffer from ft_tt_op_reduce_build_constants(): the scaler tile, the
     *  eps tile, then gamma's Wt tiles and beta's Wt tiles. ONE operand rather than three
     *  because the parameter block has room for one and because rms_norm_reader.c
     *  already put its scaler and eps adjacent "so they cost one address rather than two". */
    ft_tt_op_operand constants;
    ft_tt_op_operand output;
    /** P3 schedule; all zero is the pre-P3 schedule. `chain_chunk` <= the residency's
     *  stage_pages (0/1 = per tile); `half_read` is honoured by the kernel only for a
     *  norm-shaped program on a half-tile descriptor (ftr_half_read()). */
    uint32_t chain_chunk;
    bool stream_input;
    bool half_read;
} ft_tt_op_reduce_request;

/**
 * The widest Wt that fits the CB arena for a given residency.
 *
 * This is a HARD CEILING and it is the reason flash-decode attention is its own op rather
 * than "matmul, then softmax": a standalone softmax holds the whole row twice over (the
 * activations, to subtract the row max from, and the exponentials, to divide by the row
 * sum), so at 2048 bytes a tile it tops out around Wt = 326, i.e. W ≈ 10,432. A
 * 32k-context attention row is Wt = 1024 and does not fit at any tile size. That is a
 * capacity fact, not a performance preference. A residency that also carries gamma and
 * beta holds FOUR rows and halves it again.
 *
 * @return the largest max_tile_cols ft_tt_op_reduce_place_cbs() will accept, or 0
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_op_reduce_max_tile_cols(const ft_tt_op_reduce_residency *residency);

/**
 * Encode `request` into the sixteen words a FT_TT_QUEUE_OP_REDUCE descriptor points at.
 *
 * Pure: no device access, no allocation. That is what makes the wire format testable
 * off-hardware, which matters more here than usual — the card is the expensive place to
 * find a field in the wrong word.
 *
 * @param out_words  FT_TT_OP_VECTOR_WORDS values, fully written (unused words zeroed)
 * @return           FT_TT_OK; FT_TT_EINVAL on a zero row-block count, a zero tile_cols, an
 *                   unknown function, a zero tile size, a tile_cols past
 *                   ft_tt_op_reduce_max_tile_cols() for a softmax-shaped residency, or a
 *                   chain_chunk past FT_TT_OP_REDUCE_CHAIN_CHUNK_MAX
 */
int ft_tt_op_reduce_parameters(const ft_tt_op_reduce_request *request,
                               uint32_t out_words[FT_TT_OP_VECTOR_WORDS]);

/**
 * Place the reduce triad's eleven circular buffers in the arena.
 *
 * cb_in and cb_mid are `max_tile_cols` pages each — a whole row block resident, twice
 * over — and so are gamma and beta when the residency carries them. The rest are one or
 * two pages because they are produced and consumed a tile at a time. P3: cb_in is
 * `input_rows` rows, and each scratch CB and cb_out `stage_pages` pages, when those are set.
 *
 * @return FT_TT_OK; FT_TT_EINVAL; FT_TT_ENOMEM if it does not fit the arena
 */
int ft_tt_op_reduce_place_cbs(const ft_tt_op_reduce_residency *residency,
                              ft_tt_launch_cb_entry out_entries[FT_TT_OP_REDUCE_CB_COUNT],
                              void *buffer, size_t buffer_bytes, uint64_t *out_mask);

/** Bytes ft_tt_op_reduce_build_constants() writes for this member at this width. */
size_t ft_tt_op_reduce_constants_bytes(ft_tt_op_reduce_function function, uint32_t tile_cols,
                                       uint32_t tile_bytes);

/**
 * Build the constants buffer: the scaler tile, the eps tile, then gamma and beta expanded
 * to full tiles.
 *
 * THE SCALER. ONE tile serves BOTH reduces of a descriptor. Upstream's scaler is 1.0 for
 * SUM and for MAX (ttnn/kernel_lib/reduce_helpers_dataflow.hpp: "For SUM/MAX, the
 * reduce_factor is ignored and the scaler is 1.0"), so the two separate scaler buffers
 * moreh_softmax carries are two names for one tile; a member that wants a MEAN gets 1/W in
 * the same slot. Only the first row of each face is read by the reduce LLK; filling the
 * whole tile is a superset and costs nothing.
 *
 * THE EPS TILE. eps is one more accumulated row SUM against a tile whose every element is
 * eps*W/32 — each row has 32 columns, so 32 * (eps*W/32) * (1/W) = eps. That buys eps for
 * one extra reduce_tile instead of an add_tiles_bcast_scalar, a CB round trip and a pack.
 * The slot exists even for a member that does not use it, so the layout never moves.
 *
 * GAMMA AND BETA ARRIVE AS FULL 32x32 TILES, not broadcast rows. A norm weight is 1-D and
 * the FTTW artifact stores 1-D tensors untiled (bringup log §12.7); expanding here (or in
 * the loader, ft_tt_fttw_upload_broadcast_rows()) turns the multiply into a plain mul_tiles
 * instead of a bcast op, and costs 96 KiB of DRAM for a 1536-wide model.
 *
 * THE ELEMENT FORMAT IS THE PAGE SIZE'S, AND IT IS A PARAMETER FOR THE SAME REASON THE GDN
 * family's page size had to become one. `reduce_float16b.cbtable` places every buffer at
 * 2048 bytes and `reduce_float32.cbtable` at 4096; the reader issues `TILE_BYTES` reads
 * either way and has no descriptor header to check them against. A constants buffer built at
 * the wrong page size is refused nowhere — the scaler tile is simply half a tile of whatever
 * followed it — so the caller passes the page size the residency was PLACED at, which is the
 * same word `ft_tt_op_reduce_constants_bytes()` already takes.
 *
 * @param width      W in ELEMENTS, not tiles — the eps tile's value depends on it
 * @param eps        ignored by a member that does not use eps
 * @param gamma      W floats, or NULL for a member that does not use gamma
 * @param beta       W floats, or NULL
 * @param tile_bytes FT_TT_OP_TILE_BYTES_BF16 or _FP32; anything else is refused
 * @param out_bytes  at least ft_tt_op_reduce_constants_bytes() at the same `tile_bytes`
 * @return           FT_TT_OK, FT_TT_EINVAL
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_op_reduce_build_constants(ft_tt_op_reduce_function function, uint32_t width, float eps,
                                    const float *gamma, const float *beta, uint32_t tile_bytes,
                                    void *out, size_t out_bytes);

/* ================================================================
 * A RAGGED W — served by PADDING, and only for the members it is exact for
 * ================================================================ */

/**
 * The value a padded lane carries when it must vanish under a MAX or an exp.
 *
 * The same constant attention masks a dead score with, for the same reason and with the
 * same argument: large enough that `exp(pad - row_max)` is zero in bf16 at any plausible
 * activation, small enough that the subtraction cannot overflow
 * (FT_TT_OP_ATTENTION_MASK_VALUE, ft_tt_op_attention.h). It is on silicon there already, in
 * the same SFPU `exp`, which is why this is not a new numerical claim.
 */
#define FT_TT_OP_REDUCE_PAD_MASK_VALUE (-1.0e30f)

/** The value a padded lane carries when it must vanish under a SUM. */
#define FT_TT_OP_REDUCE_PAD_ZERO_VALUE (0.0f)

/**
 * Wt for a W that is not a whole number of tiles: ceil(width / 32).
 *
 * There is no ragged W on the wire — `ft_tt_op_reduce_request` carries `tile_cols` and the
 * reduce runs over `tile_cols * 32` columns unconditionally (kernels/resident_reduce_
 * reader.c). So a ragged W is served by making the padded width the real one and keeping
 * the DIVISOR honest, which is what the two calls below do.
 *
 * @return ceil(width/32), or 0 for width 0
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_op_reduce_padded_tile_cols(uint32_t width);

/**
 * The constant every padded lane of this member's input must hold — or a refusal, because
 * for one member of the four no constant exists.
 *
 * DERIVED PER PASS, NOT TABULATED, so a fifth member of the family gets the right answer
 * from its own row in the definition table. A padded lane is harmless exactly when the
 * value the pass MAPS it to is the pool's identity, and every pass of the member must want
 * the same constant:
 *
 *   | pass                     | padded lane maps to | vanishes when     |
 *   |--------------------------|---------------------|-------------------|
 *   | SUM, map NONE            | pad                 | pad == 0          |
 *   | SUM, map SQUARE_SELF     | pad*pad             | pad == 0          |
 *   | MAX, map NONE            | pad                 | pad << row max    |
 *   | SUM, map SUB_EXP         | exp(pad - col0)     | pad << col0       |
 *   | SUM, map SUB_SQUARE      | (pad - col0)^2      | **pad == col0**   |
 *
 * The last row is the whole finding, and it is worse than "no constant works". `col0` is
 * the ROW MEAN, a different number for every row, so no constant can be it — and A PER-ROW
 * PAD DOES NOT RESCUE IT EITHER: layer norm's FIRST pass is a plain SUM over the padded
 * row, so a pad equal to that row's mean `m` makes pass 0 report `m * padded/W` instead of
 * `m`. The two passes want different padding of the SAME buffer. **Nothing a caller writes
 * into the input serves layer norm at a ragged W**, which is why it is refused here rather
 * than shipped with a caveat, and why a mask tile is the only close. Measured off-hardware
 * in tests/test_vector_ops.c: every constant pad AND the per-row mean land 23-24x the
 * arrangement's own bf16 rounding floor.
 *
 * AND THE ERROR IS DATA-DEPENDENT, WHICH IS THE SECOND REASON IT IS A REFUSAL AND NOT A
 * WARNING. On zero-centred rows the same hole measures AT the rounding floor and would pass
 * any score anyone ran; on rows whose means spread over 0..31 it is 23x it. "It looked fine
 * on our activations" is not evidence about this one.
 *
 * @param out_value  receives FT_TT_OP_REDUCE_PAD_ZERO_VALUE or
 *                   FT_TT_OP_REDUCE_PAD_MASK_VALUE
 * @return           FT_TT_OK; FT_TT_EINVAL for an unknown member, or for one whose passes
 *                   cannot agree on a constant (layer norm) — ask
 *                   ft_tt_op_reduce_padding_is_exact() to tell the two apart
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_reduce_pad_value(ft_tt_op_reduce_function function, float *out_value);

/** Can this member be served at a ragged W by padding ALONE, exactly? True for softmax,
 *  RMS norm and the router normalise; FALSE for layer norm, and false for an unknown
 *  member. This is the named reason behind the refusal above, so a caller and a test can
 *  assert the reason rather than the return value (dev notes §3). */
bool ft_tt_op_reduce_padding_is_exact(ft_tt_op_reduce_function function);

/**
 * Copy a [rows x width] fp32 activation into the [rows x padded_width] the wire needs,
 * filling every padded lane with this member's pad value.
 *
 * @param in         rows*width floats, row-major
 * @param out        rows * ft_tt_op_reduce_padded_tile_cols(width) * 32 floats
 * @param out_count  elements `out` can hold; a short buffer is refused, never truncated
 * @return           FT_TT_OK; FT_TT_EINVAL on a null buffer, a zero row or width count, a
 *                   short output, or a member with no constant pad value
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_reduce_pad_rows(ft_tt_op_reduce_function function, const float *in, uint32_t rows,
                             uint32_t width, float *out, size_t out_count);

/**
 * ft_tt_op_reduce_build_constants() for a W that need not be a whole number of tiles.
 *
 * THE ONE THING THAT MAKES IT CORRECT IS THAT THE DIVISOR STAYS TRUE. The scaler tile is
 * 1/W and the eps tile is eps*W/32 for the REAL W, while gamma and beta expand to
 * ceil(W/32) tiles with the tail zeroed — so a mean is a mean over the columns that exist,
 * and the padded lanes contribute nothing to it. Building the constants at the padded width
 * instead is the silent wrong answer this call exists to prevent, and it is a SCALE error,
 * which PCC cannot see (dev notes §3).
 *
 * @param width      the REAL W in elements — not rounded, not the padded one
 * @param tile_bytes FT_TT_OP_TILE_BYTES_BF16 or _FP32; anything else is refused
 * @param out_bytes  at least ft_tt_op_reduce_constants_bytes() at
 *                   ft_tt_op_reduce_padded_tile_cols(width)
 * @return           FT_TT_OK; FT_TT_EINVAL for a ragged W on a member that padding cannot
 *                   serve exactly (layer norm), and everything
 *                   ft_tt_op_reduce_build_constants() refuses
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_op_reduce_build_constants_ragged(ft_tt_op_reduce_function function, uint32_t width,
                                           float eps, const float *gamma, const float *beta,
                                           uint32_t tile_bytes, void *out, size_t out_bytes);

#endif /* FT_TT_OP_VECTOR_H */
