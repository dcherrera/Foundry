// SPDX-License-Identifier: AGPL-3.0-or-later
//
// resident_vector_protocol.h
//   The parameter block the VECTOR op family reads out of L1, shared by the resident
//   elementwise triad, the resident softmax triad, and the ONE writer both of them use.
//   The host half is ft_tt_op_vector.h; the two must agree word for word and
//   tests/test_vector_ops.c pins the pair off-hardware.
//
//   "Vector op family" means: one input tile stream (sometimes two), one output tile
//   stream, the same tile count in as out, activations laid out [tokens x features] so a
//   feature row lies along a tile's W. That covers SiLU, sigmoid, add, mul, sub, the
//   SiLU-gated multiply, and softmax — and it is the shape most of the remaining ~35 ops
//   have. Matmul is the exception, not the rule.
//
// ===========================================================================
// WHY ONE BLOCK LAYOUT ACROSS THE FAMILY, AND WHAT IT BUYS
// ===========================================================================
// docs/tt_bringup_log.md 17.8 note 2: "one binary per OPCODE SET, not per op", because a
// kernel's compile-time args, CB formats and DST_ACCUM_MODE are baked at build time while
// anything that varies per call has to be a runtime arg or a descriptor field. The 32-byte
// descriptor carries six free words; a shape that does not fit them points at a block.
//
// This header takes that one step further: the ELTWISE and SOFTMAX blocks are the SAME
// SIXTEEN WORDS with the same meanings, so
//
//   1. `resident_vector_writer.c` is written ONCE and serves both. A writer is
//      "drain `tile_count` tiles from cb_out to (x, y, base)" and knows nothing about the
//      op — which is a fact about all ~40 ops, not about these two, and is worth having
//      in one file rather than forty. (resident_matmul_writer.c is the same function
//      over a DIFFERENT block layout; unifying the two is a follow-up, not this round.)
//   2. A future combined binary — one program that answers both opcodes — is a source
//      merge and not a re-layout, should the text budget allow it. Measured text sizes are
//      in the report; the 12 KiB slab (ft_tt_op_l1_layout.h) is the ceiling that decides.
//
// A word a given op does not use is written as 0 by the host and never read by the kernel.
// That is deliberately NOT "reserved for that op to redefine": a block whose meaning
// depends on which kernel is resident is a wrong answer with no error anywhere, which is
// exactly what ft_tt_op_matmul.h's `weight_n_major` note warns about.
//
// ===========================================================================
// THE ARITY RULE
// ===========================================================================
// Whether a descriptor has a SECOND input operand is derived from the FUNCTION by
// ftv_function_is_binary() below, and by nothing else. The host does not send an arity
// field, because two fields that must agree are two fields that can disagree — and the
// disagreement here is "the reader fetches a B tile nobody wrote", i.e. a plausible wrong
// answer. The host sizes both input circular buffers unconditionally (a spare bf16 input
// CB is a few KiB of a 1.29 MiB arena), so a unary descriptor costs arena and nothing else.
//
#ifndef FOUNDRY_RESIDENT_VECTOR_PROTOCOL_H
#define FOUNDRY_RESIDENT_VECTOR_PROTOCOL_H

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
// The sixteen words. Mirrors ft_tt_op_vector.h -- change both or neither.
// ---------------------------------------------------------------------------

/** Which function an ELTWISE descriptor applies. Unused (0) by SOFTMAX. */
enum { kVecFunction = 0 };

/** TOTAL tiles this descriptor moves, in and out. The writer reads this and nothing
 *  else about the shape, which is what lets one writer serve the family. */
enum { kVecTileCount = 1 };

/** Tiles across one softmax row (Wt). 0 for elementwise, which has no row structure.
 *  Row blocks are kVecTileCount / kVecTileCols -- derived, never sent twice. */
enum { kVecTileCols = 2 };

/** ELTWISE reuses word 2 (it has no row structure) as a flags word. kVecEltwiseRow0: the
 *  SFPU verbs (silu, sigmoid) compute ROW 0 of each tile only -- a decode strip's one live
 *  row (rows 0-3 are computed); rows 4-31 keep the copied input. FPU verbs and data movement are unchanged. */
enum { kVecEltwiseFlags = 2 };
enum { kVecEltwiseRow0 = 1u };

/** K-tiles per NoC barrier in the reader. Must DIVIDE the run it blocks over
 *  (kVecTileCount for elementwise, kVecTileCols for softmax) or a block straddles the CB
 *  wrap; the kernel refuses to guess and falls back to 1. See ft_tt_op_block_depth(). */
enum { kVecDepth = 3 };

/** E3 SPLIT PROBE (timing only — an armed bit's OUTPUT IS NOT EVIDENCE of anything): the
 *  top byte of word 3. The depth is the low 24 bits; every reader of word 3 masks. */
enum { kVecProbeShift = 24 };
enum { kVecProbeNoRead = 1u };    /* readers skip their NoC reads and barrier */
enum { kVecProbeNoWrite = 2u };   /* the vector writer skips its NoC writes and barrier */
enum { kVecProbeLegacyBarrier = 4u }; /* writer: the OLD per-tile ack barrier (the control) */
enum { kVecProbeNoCompute = 8u }; /* eltwise compute moves pages, no unpack/math/pack */

/** First input: the activation. */
enum { kVecAX = 4 };
enum { kVecAY = 5 };
enum { kVecABase = 6 };

/** Second operand. ELTWISE: the B tile stream, for a binary function. SOFTMAX: the ONE
 *  reduce scaler tile (1.0 everywhere), which is a single tile and not a stream. */
enum { kVecBX = 7 };
enum { kVecBY = 8 };
enum { kVecBBase = 9 };

/** Output. */
enum { kVecCX = 10 };
enum { kVecCY = 11 };
enum { kVecCBase = 12 };

/** Sixteen so a batch of blocks is one clean host write and a shape that grows moves
 *  nothing. Matches FT_TT_QUEUE_PARAM_WORDS. */
enum { kVecWords = 16 };

/** Word 3 split: the depth (low 24 bits) and the E3 probe bits (top byte). */
static inline uint32_t ftv_depth_word(uint32_t word3) { return word3 & 0x007FFFFFu; }
/** M3 (REDUCE only): bit 23 of word 3 = HALF TILE — the elementwise FPU passes (the map's
 *  square, the gamma multiply, the scale broadcast) unpack and compute faces 0-1 only (rows
 *  0-15); the row reduce stays whole. A single-token descriptor's live row is row 0. */
static inline bool ftv_half_tile(uint32_t word3) { return ((word3 >> 23) & 1u) != 0u; }
static inline uint32_t ftv_probe_bits(uint32_t word3) { return word3 >> kVecProbeShift; }

// ---------------------------------------------------------------------------
// The elementwise function set -- ONE binary, six functions
// ---------------------------------------------------------------------------
//
// SiLU and sigmoid are the same kernel with a different SFPU verb; add, mul and sub are
// the same kernel with a different FPU opcode; and all five are the same kernel as each
// other because a resident binary is BOUND TO ONE SLAB, so two binaries cannot be resident
// on one core at once (docs/tt_bringup_log.md 17.8 note 4). Splitting silu from mul would
// mean reloading five images -- ~0.19 ms over the x1 link, measured (20.4) -- between the
// two halves of every SwiGLU. That is the argument, and it is a residency argument rather
// than a code-size one.
//
enum { kVecFnSilu = 1 };     // y = x * sigmoid(x)                      unary
enum { kVecFnSigmoid = 2 };  // y = sigmoid(x)                          unary
enum { kVecFnAdd = 3 };      // y = a + b                               binary
enum { kVecFnMul = 4 };      // y = a * b                               binary
enum { kVecFnSub = 5 };      // y = a - b                               binary
/** y = silu(a) * b -- the SwiGLU gate, and the ONE fusion 17.8 note 1 licenses here: as
 *  two descriptors it costs a full DRAM round trip of the intermediate, and as one it
 *  costs an L1 round trip through a scratch CB. It is NOT fused to save dispatch; there is
 *  no dispatch to save. */
enum { kVecFnSiluMul = 6 };

/** Whether `function` takes a second input stream. THE ONLY definition of arity; see the
 *  header note. */
FTQ_INLINE bool ftv_function_is_binary(uint32_t function)
{
    return function >= kVecFnAdd;
}

/** Whether `function` needs the scratch CB (silu into L1, then multiply). */
FTQ_INLINE bool ftv_function_uses_scratch(uint32_t function)
{
    return function == kVecFnSiluMul;
}

/**
 * The depth a reader will actually block at.
 *
 * Transliterated from ft_tt_op_matmul_prefetch_depth() / ft_tt_op_block_depth(), and the
 * kernel repeats the host's arithmetic rather than trusting it for the same reason
 * resident_matmul_reader.c does: a depth that does not divide the run moves the CB write
 * pointer off the block grid and every later block straddles the wrap. Falling back to 1 is always correct, just slower.
 */
FTQ_INLINE uint32_t ftv_block_depth(uint32_t run, uint32_t requested)
{
    return (requested > 0u && run > 0u && run % requested == 0u) ? requested : 1u;
}

#endif  // FOUNDRY_RESIDENT_VECTOR_PROTOCOL_H
