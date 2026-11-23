/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_sample.h
 * @brief Host driver for the SAMPLING op pair — exact on-card top-N over a vocabulary
 *   logit row (`FT_TT_QUEUE_OP_TOPK`, the `fc_topk_logits` equivalent) and the token draw
 *   that consumes it (`FT_TT_QUEUE_OP_SAMPLE`). Places the circular buffer and the two L1
 *   blocks, encodes the parameter block, builds and reads the sampler state.
 *
 * The device half is `kernels/resident_topk_protocol.h`; the two must agree word for word
 * and `tests/test_topk_ops.c` pins the pair off-hardware by PARSING the device header.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS OP IS NOT OPTIONAL AND CANNOT BE A HOST FUNCTION
 * ---------------------------------------------------------------------------
 * Card → host is **2.3 MB/s at 8-byte loads, 4.7 at 16** (dev notes §6) and device→host
 * DMA does not exist on Blackhole at all. MiniCPM5-1B's vocabulary is **130,560** —
 * MEASURED, read out of the production artifact's `output.weight` directory entry, not the
 * ~73k this op was scoped against — so one bf16 logit row is **261,120 bytes** and costs
 * **55–113 ms per token** just to fetch, before the 2-core no-AVX host does a 130k-wide
 * softmax on it. The whole point of the op is that **what crosses the link is a 4-byte
 * token id**.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE SEMANTICS ARE COPIED FROM, AND WHY THAT IS THE RIGHT REFERENCE
 * ---------------------------------------------------------------------------
 * Pantheon already runs exactly this shape on the K80: `fc_topk_logits` reduces on the die
 * and returns k (id, value) pairs, `gpu_present_topk()` scatters them into an all-`-INFINITY`
 * vocabulary row, and `sample_token_core()` samples THAT. So the on-card sampler here is
 * not a new approximation — it is the deployed path, and the acceptance reference is that
 * same function (`src/core/engine.c` `sparse_sample()` / `sparse_argmax()`).
 *
 * Two things are inherited from it deliberately:
 *   - **The comparator.** `(a > b) || (a == b && index(a) < index(b))` — LOWEST index wins
 *     a tie, which is what `fc_topk_logits`'s iterative argmax reproduces from the host's
 *     serial scan. With bf16's 8 mantissa bits over 130,560 logits, ties are the common
 *     case and not an edge one.
 *   - **The greedy ORDER.** Temperature ≤ 0 is a pure argmax on the RAW values taken
 *     BEFORE any repetition penalty. Applying the penalty first is a real bug that was
 *     shipped once: it demoted the true argmax below a near-tied runner-up and lost the
 *     answer with a forward pass that was faithful throughout. The kernel keeps the two
 *     orders in SEPARATE functions so they cannot be merged by accident, and the on-card
 *     test asserts it with a case that can actually go red — the first version could not,
 *     because a fresh state has an empty recent ring and the penalty was a no-op.
 *
 * ---------------------------------------------------------------------------
 * WHAT DOES *NOT* HAPPEN ON THE CARD, STATED SO NOBODY ASSUMES IT DOES
 * ---------------------------------------------------------------------------
 * `logit_bias`, the presence/frequency penalties and grammar masking all need the FULL
 * vocabulary row and a client-supplied token set, so they are not implemented here and are
 * not silently approximated. A request that needs them is a request for the full-logits
 * path — which on this link is 55–113 ms a token and is a reason to constrain differently,
 * not a mode to fall into by accident.
 *
 * @status GREEN ON SILICON — Tensix 51, 57 and 59, 40 checks and 0 failures, including the
 *   full MiniCPM5-1B row at 2.590 ms a token. Off-hardware: 132 checks, 0 failures. Five
 *   kernel mutations run through the unmodified test binary, all five red.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_SAMPLE_H
#define FT_TT_OP_SAMPLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_launch.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_queue.h"

/* ================================================================
 * The parameter block — mirrors kernels/resident_topk_protocol.h
 * ================================================================ */

/** Words of one sampling-family parameter block. A word an op does not use is written 0 by
 *  the host and never read by the kernel; it is NOT free for that op to redefine. */
typedef enum ft_tt_op_sample_field {
    FT_TT_OP_SAMPLE_TILE_COUNT = 0,   /**< tiles in the logit row (Wt) */
    FT_TT_OP_SAMPLE_ROW = 1,          /**< which of a tile's 32 rows is the decoded token */
    FT_TT_OP_SAMPLE_SELECT_COUNT = 2, /**< N pairs to select; <= FT_TT_OP_SAMPLE_MAX_SELECT */
    FT_TT_OP_SAMPLE_DEPTH = 3,        /**< reader block depth; divides the tile count */
    FT_TT_OP_SAMPLE_LOGITS_X = 4,
    FT_TT_OP_SAMPLE_LOGITS_Y = 5,
    FT_TT_OP_SAMPLE_LOGITS_BASE = 6,
    FT_TT_OP_SAMPLE_RESULT_ADDRESS = 7, /**< L1 on the op's core */
    FT_TT_OP_SAMPLE_STATE_ADDRESS = 8,  /**< L1 on the op's core */
    FT_TT_OP_SAMPLE_TOKEN_OUT_X = 9,
    FT_TT_OP_SAMPLE_TOKEN_OUT_Y = 10,
    FT_TT_OP_SAMPLE_TOKEN_OUT_BASE = 11, /**< 0 = leave the id in the result block only */
    FT_TT_OP_SAMPLE_FLAGS = 12,
    FT_TT_OP_SAMPLE_TAIL_BINS = 13, /**< with FT_TT_OP_SAMPLE_FLAG_TAIL_STATS: threshold count */
    FT_TT_OP_SAMPLE_ROWS = 14,      /**< TOPK rows `row`..`row+rows-1` in one pass; 0 or 1 = one */
    FT_TT_OP_SAMPLE_HELPER_AREA = 15, /**< TOPK: L1 helper area (the TRISCs scan too); 0 = none */
    FT_TT_OP_SAMPLE_WORDS = 16
} ft_tt_op_sample_field;

/** Pop every logit block WITHOUT selecting over it. The measurement arm that isolates the
 *  DRAM stream from the scalar scan — a RUNTIME word, so both arms are the same binary.
 * It writes an empty result block on purpose. */
#define FT_TT_OP_SAMPLE_FLAG_DRAIN_ONLY 1u

/** TAIL STATS: while selecting, also count every logit of the row into the bins the host's
 *  threshold keys define (the tail area after the pairs; kernels/resident_topk_protocol.h §6).
 *  The pairs are unchanged. ft_tt_op_sample_tail_keys_encode / _tail_counts_decode. */
#define FT_TT_OP_SAMPLE_FLAG_TAIL_STATS 2u
/** RED (proof only): NCRISC merges without helper 2's selections — the output must DIFFER. */
#define FT_TT_OP_SAMPLE_FLAG_RED_DROP_HELPER 4u
#define FT_TT_OP_SAMPLE_TAIL_MAX_BINS 32u
/** Result-block word offsets of the tail area: the keys the host writes, then the counts
 *  (count[j] = keys in [key[j], key[j-1]), count[bins] = below the last), then the status. */
#define FT_TT_OP_SAMPLE_TAIL_KEYS_OFFSET 136u
#define FT_TT_OP_SAMPLE_TAIL_COUNTS_OFFSET 168u
#define FT_TT_OP_SAMPLE_TAIL_COUNTS_WORDS 36u
#define FT_TT_OP_SAMPLE_TAIL_STATUS_WORD 35u
#define FT_TT_OP_SAMPLE_TAIL_STATUS_MAGIC 0x46545431u /* "FTT1" */

/** Most pairs a result block holds. */
#define FT_TT_OP_SAMPLE_MAX_SELECT 64u

/** MULTI-ROW TOPK (2026-10-21): one descriptor selects over up to this many consecutive token
 *  rows of the SAME logit tiles, reading each tile once; row k's result block is the k-th
 *  FT_TT_OP_SAMPLE_RESULT_BYTES block from `result_address`. Each row's selection is the
 *  single-row algorithm over the same elements in the same order, so its block is the block a
 *  single-row descriptor for that row writes. No tail stats with rows > 1. */
#define FT_TT_OP_SAMPLE_MAX_ROWS 4u

/** HELPERS (2026-10-21): the three TRISCs of the sampling core scan tiles t & 3 == 1..3 while
 *  NCRISC scans t & 3 == 0, and NCRISC merges the four exactly (resident_topk_protocol.h
 *  "HELPERS"). The area is 64 B of handshake words + 3 x FT_TT_OP_SAMPLE_MAX_ROWS selections
 *  of 784 B (the kernel's sizeof(ftk_select)), placed right after the result blocks. Not
 *  used with tail stats or a drain. */
#define FT_TT_OP_SAMPLE_HELPER_BYTES (64u + 3u * FT_TT_OP_SAMPLE_MAX_ROWS * 784u)
/** Result CB pages: the row blocks, then the helper area. */
#define FT_TT_OP_SAMPLE_RESULT_PAGES                                                       \
    (FT_TT_OP_SAMPLE_MAX_ROWS +                                                            \
     (FT_TT_OP_SAMPLE_HELPER_BYTES + FT_TT_OP_SAMPLE_RESULT_BYTES - 1u) /                  \
         FT_TT_OP_SAMPLE_RESULT_BYTES)

/** The single CB: the logit tile stream, BRISC → NCRISC. */
#define FT_TT_OP_SAMPLE_CB_LOGITS 0u
#define FT_TT_OP_SAMPLE_CB_COUNT  3u

/** Result-block words. Header, then N pairs of [index, fp32 value bits]. */
typedef enum ft_tt_op_sample_result_field {
    FT_TT_OP_SAMPLE_RESULT_MAGIC = 0,
    FT_TT_OP_SAMPLE_RESULT_COUNT = 1,
    FT_TT_OP_SAMPLE_RESULT_SEQUENCE = 2,
    FT_TT_OP_SAMPLE_RESULT_TOKEN_VALID = 3,
    FT_TT_OP_SAMPLE_RESULT_TOKEN = 4,
    FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS = 8
} ft_tt_op_sample_result_field;

/**
 * The token hand-off's slot: `token_out.address` names 16 BYTES, not one word, and must be
 * 16-byte aligned. The id lands in the first word and the other three are zeroed.
 *
 * This is a hardware rule, not a convention. `noc_inline_dw_write` — the obvious way to move
 * four bytes — excludes DRAM destinations by its own contract, and `token_out` normally names
 * DRAM; on Blackhole the call degrades into a 4-byte write, which is below the 16 bytes
 * `ft_tt_noc_alignment()` requires of a DRAM write in address AND in length. The value landed
 * correctly on every run and the core could not be booted afterwards.
 * `kernels/resident_topk_protocol.h` carries the measurement and the cores it cost.
 */
#define FT_TT_OP_SAMPLE_TOKEN_SLOT_WORDS ((uint32_t)FT_TT_OP_SAMPLE_RESULT_TOKEN)
#define FT_TT_OP_SAMPLE_TOKEN_SLOT_BYTES 16u
#define FT_TT_OP_SAMPLE_TOKEN_OUT_ALIGNMENT 16u

#define FT_TT_OP_SAMPLE_PAIR_WORDS 2u
#define FT_TT_OP_SAMPLE_RESULT_MAGIC_VALUE 0x46544B31u /* "FTK1" */
/** Header, FT_TT_OP_SAMPLE_MAX_SELECT pairs, then the tail area. */
#define FT_TT_OP_SAMPLE_RESULT_BYTES \
    ((FT_TT_OP_SAMPLE_TAIL_COUNTS_OFFSET + FT_TT_OP_SAMPLE_TAIL_COUNTS_WORDS) * 4u)

/* ================================================================
 * The sampler state — what a closed decode loop carries between tokens
 * ================================================================
 * Every field here is per-TURN or per-SEQUENCE, never per-token, which is exactly why it is
 * a card-resident block and not a descriptor field: a host that had to refresh it per token
 * would be back in the path this op exists to leave.
 */

typedef enum ft_tt_op_sample_state_field {
    FT_TT_OP_SAMPLE_STATE_MAGIC = 0,
    FT_TT_OP_SAMPLE_STATE_RNG = 1,               /**< xorshift32; never 0 */
    FT_TT_OP_SAMPLE_STATE_TEMPERATURE_BITS = 2,  /**< fp32 bits; <= 0 means greedy */
    FT_TT_OP_SAMPLE_STATE_TOP_P_BITS = 3,
    FT_TT_OP_SAMPLE_STATE_REPEAT_PENALTY_BITS = 4,
    FT_TT_OP_SAMPLE_STATE_TOP_K = 5,
    FT_TT_OP_SAMPLE_STATE_REPEAT_WINDOW = 6,
    FT_TT_OP_SAMPLE_STATE_RECENT_CURSOR = 7,
    FT_TT_OP_SAMPLE_STATE_RECENT_COUNT = 8,
    FT_TT_OP_SAMPLE_STATE_EMITTED_COUNT = 9,
    FT_TT_OP_SAMPLE_STATE_EOS_COUNT = 10,
    FT_TT_OP_SAMPLE_STATE_STOPPED = 11,
    FT_TT_OP_SAMPLE_STATE_TOKEN_BUDGET = 12,
    FT_TT_OP_SAMPLE_STATE_LAST_TOKEN = 13,
    FT_TT_OP_SAMPLE_STATE_HEADER_WORDS = 16
} ft_tt_op_sample_state_field;

#define FT_TT_OP_SAMPLE_RECENT_CAPACITY 64u
#define FT_TT_OP_SAMPLE_EOS_CAPACITY 4u
#define FT_TT_OP_SAMPLE_EMITTED_CAPACITY 256u
#define FT_TT_OP_SAMPLE_RECENT_OFFSET_WORDS FT_TT_OP_SAMPLE_STATE_HEADER_WORDS
#define FT_TT_OP_SAMPLE_EOS_OFFSET_WORDS \
    (FT_TT_OP_SAMPLE_RECENT_OFFSET_WORDS + FT_TT_OP_SAMPLE_RECENT_CAPACITY)
#define FT_TT_OP_SAMPLE_EMITTED_OFFSET_WORDS \
    (FT_TT_OP_SAMPLE_EOS_OFFSET_WORDS + FT_TT_OP_SAMPLE_EOS_CAPACITY)
#define FT_TT_OP_SAMPLE_STATE_WORDS \
    (FT_TT_OP_SAMPLE_EMITTED_OFFSET_WORDS + FT_TT_OP_SAMPLE_EMITTED_CAPACITY)
#define FT_TT_OP_SAMPLE_STATE_BYTES (FT_TT_OP_SAMPLE_STATE_WORDS * 4u)
#define FT_TT_OP_SAMPLE_STATE_MAGIC_VALUE 0x46544B53u /* "FTKS" */

/** The knobs, as a caller states them. Mirrors `pn_sampling_params`' meaningful subset. */
typedef struct ft_tt_op_sample_params {
    float temperature;    /**< <= 0 selects greedy: argmax on RAW values, NO penalty */
    float top_p;          /**< 1.0 disables */
    float repeat_penalty; /**< 1.0 disables */
    uint32_t top_k;       /**< 0 disables */
    uint32_t repeat_window;
    uint32_t rng_seed;    /**< must be non-zero: xorshift32 has a fixed point at 0 */
    uint32_t token_budget; /**< 0 = unbounded */
    uint32_t eos_ids[FT_TT_OP_SAMPLE_EOS_CAPACITY];
    uint32_t eos_count;
} ft_tt_op_sample_params;

/* ================================================================
 * Requests
 * ================================================================ */

/** One TOPK call: select `select_count` pairs from the logit row, for token row `row`. */
typedef struct ft_tt_op_topk_request {
    uint32_t tile_count;   /**< Wt; 4080 for MiniCPM5-1B */
    uint32_t row;          /**< 0..31 */
    uint32_t select_count; /**< <= FT_TT_OP_SAMPLE_MAX_SELECT */
    uint32_t prefetch_tiles;
    uint32_t flags;
    ft_tt_op_operand logits;
    uint32_t result_address; /**< L1 on the op's core, from ft_tt_op_sample_place_cbs() */
    uint32_t state_address;  /**< L1 on the op's core, from ft_tt_op_sample_place_cbs() */
    /** tile_bytes ignored; a zero address means "no write". A non-zero address names a
     *  FT_TT_OP_SAMPLE_TOKEN_SLOT_BYTES slot and must be
     *  FT_TT_OP_SAMPLE_TOKEN_OUT_ALIGNMENT-aligned — see that constant. */
    ft_tt_op_operand token_out;
    /** With FT_TT_OP_SAMPLE_FLAG_TAIL_STATS in `flags`: 1..FT_TT_OP_SAMPLE_TAIL_MAX_BINS. */
    uint32_t tail_bins;
    /** TOPK only: rows selected from `row` up, 0 or 1 = one; <= FT_TT_OP_SAMPLE_MAX_ROWS. */
    uint32_t rows;
    /** TOPK only: the helper area (result_address + FT_TT_OP_SAMPLE_MAX_ROWS result blocks),
     *  0 = NCRISC scans alone. */
    uint32_t helper_area;
} ft_tt_op_topk_request;

/**
 * Encode `request` into the sixteen words a FT_TT_QUEUE_OP_TOPK or FT_TT_QUEUE_OP_SAMPLE
 * descriptor points at. ONE encoder for both opcodes: the block is the same block, and two
 * encoders would be two things to keep in step.
 *
 * Pure: no device access, no allocation.
 *
 * @param out_words  FT_TT_OP_SAMPLE_WORDS values, fully written (unused words zeroed)
 * @return           FT_TT_OK; FT_TT_EINVAL on a zero tile count, a row past 31, a row span
 *                   past 31 or wider than FT_TT_OP_SAMPLE_MAX_ROWS, rows > 1 with tail
 *                   stats, a zero or
 *                   oversized select count, a zero tile size, a missing result/state
 *                   address, or a `token_out` address that is not
 *                   FT_TT_OP_SAMPLE_TOKEN_OUT_ALIGNMENT-aligned
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_sample_parameters(const ft_tt_op_topk_request *request,
                               uint32_t out_words[FT_TT_OP_SAMPLE_WORDS]);

/**
 * The tail area's KEYS, as the words the host writes at FT_TT_OP_SAMPLE_TAIL_KEYS_OFFSET of
 * the result block before a FT_TT_OP_SAMPLE_FLAG_TAIL_STATS descriptor.
 *
 * @param keys  `bins` bf16 order keys (the kernel's key(), resident_topk_protocol.h §4),
 *              strictly DESCENDING
 * @return      FT_TT_OK; FT_TT_EINVAL on a bin count out of 1..FT_TT_OP_SAMPLE_TAIL_MAX_BINS
 *              or keys that are not strictly descending 16-bit values
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_sample_tail_keys_encode(const uint32_t *keys, uint32_t bins,
                                     uint32_t out_words[FT_TT_OP_SAMPLE_TAIL_MAX_BINS]);

/**
 * Decode the tail area's counts from a result block read back after the descriptor.
 *
 * @param area        the FT_TT_OP_SAMPLE_TAIL_COUNTS_WORDS words at
 *                    FT_TT_OP_SAMPLE_TAIL_COUNTS_OFFSET
 * @param out_counts  bins + 1 counts
 * @return            FT_TT_OK; FT_TT_EINVAL when the status word says the kernel counted
 *                    nothing (unusable keys, an aborted descriptor)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_sample_tail_counts_decode(const uint32_t *area, uint32_t bins, uint32_t *out_counts);

/**
 * Place the program's circular buffer and its two L1 blocks in the arena.
 *
 * cb_logits gets TWO BLOCKS of the effective prefetch depth, so the reader fills one while
 * the selector drains the other. The result and state blocks are placed through the SAME
 * arena allocator as CB entries 1 and 2 — they are ALLOCATIONS, not streams: nothing pushes
 * or pops them and the kernel reaches them by the address in the parameter block. Going
 * through the allocator rather than hardcoding an offset is the point; an op that names its
 * own L1 address is an op that collides with the next one.
 *
 * **THE BLOCK-PREFETCH INVARIANT IS STRONGER FOR A RESIDENT OP THAN THE ONE THE KERNEL
 * MODEL STATES, AND THE DIFFERENCE OVERWRITES L1.** `docs/tt_kernel_model.md` §12 says
 * "depth DIVIDES the run", which is correct for a ONE-SHOT kernel because its FIFO write
 * pointer starts at the base. A resident kernel runs many descriptors, `cb_push_back` only
 * wraps when the pointer lands EXACTLY on the limit, and a
 * descriptor that leaves the pointer off the NEXT descriptor's block grid makes the next
 * block step OVER the limit rather than wrap — after which every block walks further up L1
 * with no fault and no error. MEASURED here, on the card: 128 tiles at depth 32 through a
 * 96-page buffer, then 4080 tiles at depth 48, and a `noc_async_read` landed on the queue
 * control block at 0x170000.
 *
 * So a caller must satisfy, for EVERY descriptor of one residency:
 *   - `depth <= page_count`             — otherwise `cb_reserve_back` is unsatisfiable;
 *   - `page_count % depth == 0`         — every wrap lands exactly on the limit;
 *   - `tile_count % page_count == 0`    — the descriptor returns the pointer to the base.
 *
 * All three are now REFUSED BY THE KERNEL with `FT_TT_QUEUE_KERR_BAD_PARAMETERS`, because
 * the kernel can read `fifo_num_pages` and the host's placement is the one thing it cannot
 * see. Satisfying them is still the caller's job: place at the largest depth any descriptor
 * will use, and choose shapes whose tile counts are whole multiples of the page count.
 *
 * **This is a fact about the resident FAMILY and not about this op.** Any resident reader
 * whose descriptors vary the depth, or whose tile count is not a whole number of buffer
 * cycles, has the same hole; it is latent rather than live in the ops that happen to use one
 * depth throughout.
 *
 * @param request       shape; only tile_count, prefetch_tiles and logits.tile_bytes are read
 * @param out_entries   FT_TT_OP_SAMPLE_CB_COUNT entries with addresses filled in; entry 1 is
 *                      the result block and entry 2 the state block
 * @param buffer        receives the encoded CB descriptor array
 * @param out_mask      receives the launch message's local_cb_mask
 * @return              FT_TT_OK; FT_TT_EINVAL; FT_TT_ENOMEM if it does not fit the arena
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_sample_place_cbs(const ft_tt_op_topk_request *request,
                              ft_tt_launch_cb_entry out_entries[FT_TT_OP_SAMPLE_CB_COUNT],
                              void *buffer, size_t buffer_bytes, uint64_t *out_mask);

/**
 * Build the initial sampler state block from `params`.
 *
 * @param out         receives FT_TT_OP_SAMPLE_STATE_WORDS words, fully written
 * @param out_words   its length in words
 * @return            FT_TT_OK; FT_TT_EINVAL on a zero RNG seed, a repeat window past
 *                    FT_TT_OP_SAMPLE_RECENT_CAPACITY, or an EOS count past its capacity
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_sample_state_build(const ft_tt_op_sample_params *params, uint32_t *out,
                                size_t out_words);

/**
 * Decode a result block the card produced: `count`, the pairs, and the sampled token.
 *
 * Refuses a block whose magic is wrong rather than reporting whatever L1 held — which is
 * the difference between "the address was wrong" and a plausible answer.
 *
 * @param block         FT_TT_OP_SAMPLE_RESULT_BYTES read back off the card
 * @param out_indices   at least `max_pairs` entries
 * @param out_values    at least `max_pairs` entries, fp32
 * @param out_count     receives the number of pairs written
 * @param out_token     may be NULL; receives the sampled id, or is left alone when the
 *                      block's token-valid word is 0
 * @return              FT_TT_OK; FT_TT_EINVAL on a bad magic or a count past `max_pairs`
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_sample_result_decode(const uint32_t *block, uint32_t max_pairs,
                                  uint32_t *out_indices, float *out_values,
                                  uint32_t *out_count, uint32_t *out_token);

/**
 * The tile count a vocabulary of `vocab` entries occupies, and the refusal that goes with
 * it. MiniCPM5-1B's 130,560 is 4080 tiles EXACTLY; a vocabulary that is not a multiple of
 * 32 would put padding in the last tile, and zero-padding a logit row is not harmless —
 * a pad of 0.0 outranks every negative logit and would be selected as a token id that does
 * not exist. So it is refused here rather than masked in the kernel's inner loop, which is
 * the one place in this op where a per-element test would cost real time.
 *
 * @param vocab  vocabulary size
 * @return       tiles, or 0 if `vocab` is 0 or not a multiple of 32
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_op_sample_row_tiles(uint32_t vocab);

#endif /* FT_TT_OP_SAMPLE_H */
