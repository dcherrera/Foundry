// SPDX-License-Identifier: AGPL-3.0-or-later
//
// resident_topk_protocol.h
//   The device half of ft_tt_op_sample.h: the parameter block, the result block, the
//   per-turn sampler state, the ORDER COMPARATOR, and the pseudo-random generator, shared
//   by the resident TOPK/SAMPLE program's three images. The host half is ft_tt_op_sample.h;
//   the two must agree word for word and tests/test_topk_ops.c pins the pair off-hardware
//   by PARSING this file rather than by retyping it.
//
// ===========================================================================
// 1. WHY THIS OP EXISTS AT ALL, AND WHY IT CANNOT BE A HOST FUNCTION
// ===========================================================================
// Card -> host is 2.3 MB/s at 8-byte loads and 4.7 MB/s at 16 (dev notes §6), and
// device -> host DMA does not exist on Blackhole. MiniCPM5-1B's vocabulary is
// 130,560 (MEASURED, read out of the production artifact's `output.weight` directory
// entry -- not the 73k this op was scoped against), so ONE bf16 logit row is 261,120
// bytes and costs 55-113 ms to fetch. Decode cannot pay that per token, and the 2-core
// no-AVX host cannot afford the 130k-wide softmax on the other side of it either.
//
// So the requirement is a shape, not a preference: **the only thing that crosses the link
// per token is a 4-byte token id.** Everything above -- selection, the penalty, the
// temperature, the two nucleus filters and the draw -- happens here.
//
// ===========================================================================
// 2. THE SHAPE: A STREAMING SELECTION NETWORK, NOT A TILE MAP
// ===========================================================================
// Every other op in this backend is reader -> compute -> writer with the TRISCs doing the
// arithmetic. Top-k is not that op and should not be forced into it, for three
// independent reasons:
//
//   a. IT RETURNS AN INDEX. The FPU and SFPU produce VALUES; `reduce_tile<MAX>` gives the
//      row maximum and cannot say which column it came from. Blackhole does ship a bitonic
//      top-k SFPU (`topk_local_sort`/`topk_merge`/`topk_rebuild`, plus
//      `max_reduce_with_indices`) which carries index tiles alongside the data -- but it
//      sorts 64 values spanning TWO tiles along the column axis, wants K in {4,8,16,32,64},
//      runs in APPROXIMATE mode, and needs the sorted run to live in DST. Our row lies
//      along W, is 4080 tiles wide, and Wt is not a power of two.
//   b. THE ROW DOES NOT FIT L1. 4080 tiles x 2048 B is 8,355,840 B against a 1,351,680 B
//      CB arena (ft_tt_op_l1_layout.h). Any exact answer is therefore a STREAMING
//      reduction with bounded state, whatever unit computes it -- 6.2x too big is not a
//      tuning problem.
//   c. THE STATE IS TINY AND SCALAR. N <= 64 (index, key) pairs is 512 bytes. A sorted
//      insert is a branch and a shift, which is what a RISC-V is for and what a 32-lane
//      SIMD unit is not.
//
// Hence: **the reader block-prefetches logit tiles into a circular buffer and the
// data-movement processor beside it runs a scalar streaming selection over them.** The
// compute thread is deliberately idle (resident_topk_compute.c says so in its own
// header, and says what would put it back to work).
//
// ===========================================================================
// 3. THE HOT LOOP IS ONE SUBTRACT AND ONE UNSIGNED COMPARE PER LOGIT
// ===========================================================================
// The naive scalar test is "map this bf16 to an order-preserving integer, compare against
// the current N-th best". The map (§4) is four operations, and at 130,560 elements a token
// that is the whole budget.
//
// It is not needed on the REJECT path, which is 99.97% of elements. "Does this bf16 beat
// the incumbent `worst`" is a test on the RAW BITS, and the winning set is always ONE
// CONTIGUOUS INTERVAL of the 16-bit pattern space:
//
//   worst is non-negative (bits < 0x8000): a candidate wins iff it is positive AND its
//        bits are larger, i.e. bits in (worst, 0x8000).
//   worst is negative (bits >= 0x8000): every positive wins, and a negative wins iff its
//        bits are SMALLER (larger bits = larger magnitude = more negative), i.e. the
//        winning set is bits in [0, worst) -- the two ranges are contiguous.
//
// So the whole test is `(uint32_t)(bits - accept_low) < accept_span` with the pair
// recomputed only when `worst` changes -- ~ONE subtract and ONE branch per logit. The
// order-preserving map is then paid only by the winners. Measured effect is in the report;
// the point here is that the reject path never converts a float.
//
// ===========================================================================
// 4. THE ORDER, AND WHY TIES ARE THE MAIN EVENT AND NOT AN EDGE CASE
// ===========================================================================
// bf16 has EIGHT mantissa bits. Over a 130,560-element row, exact ties are certain, not
// exotic -- there are only 65,536 bf16 patterns, so the pigeonhole gives an average of two
// logits per representable value before you look at the distribution. Tie handling is
// therefore the dominant correctness question for this op, which is why the acceptance
// gate is exact index agreement and not a probability score: a top-k whose values are all
// right and whose indices are off by a tie correlates at 1.0 with the truth and is wrong.
//
// THE COMPARATOR, matching Pantheon's K80 `fc_topk_logits` (topk_logits.cu:46-50) and the
// host serial scan it reproduces (`if (logits[i] > logits[best]) best = i`):
//
//     a beats b  <=>  key(a) > key(b) || (key(a) == key(b) && index(a) < index(b))
//
// i.e. **LOWEST INDEX WINS A TIE**. Scanning in ascending index order, that is exactly
// "insert only on strictly greater", which is what ftk_select_offer() does.
//
// key() is the standard order-preserving map from IEEE bits to unsigned:
//     key(u) = (u & 0x8000) ? (uint16)~u : (uint16)(u | 0x8000)
//
// KNOWN AND DELIBERATE: key() orders -0.0 (0x8000 -> 0x7FFF) BELOW +0.0 (0x0000 -> 0x8000),
// where IEEE says they are equal. The host reference is defined on the same key order so
// the gate is exact, and tests/test_topk_ops.c sweeps ALL 65,536 bf16 patterns to prove
// key order and float order agree everywhere else. NaN is refused, not ordered: the host
// checks the row and this kernel does not, because a per-element NaN test is exactly the
// four operations §3 exists to avoid.
//
// ===========================================================================
// 5. WHAT A CLOSED DECODE LOOP HAS TO CARRY BETWEEN TOKENS
// ===========================================================================
// This is the reason the sampler's knobs live in a STATE block on the card and not in the
// descriptor. A descriptor field is per-call; every one of these is per-TURN or
// per-SEQUENCE, and a host that had to refresh them per token would be back in the path
// the op exists to leave. ftk_state is that list, and it is deliberately the whole list:
//
//   - the PRNG state, so successive tokens are not the same draw and so a turn is
//     reproducible from a seed the host set once;
//   - the sampler knobs (temperature, top_k, top_p, repeat penalty and its window), which
//     change when a request changes, not when a token is emitted;
//   - the RECENT-TOKEN RING and its cursor, appended to BY THE SAMPLER, because the
//     repetition penalty's input is the loop's own output;
//   - the EMITTED-TOKEN RING and its count, so the host drains N tokens per poll instead
//     of being in the path of each one;
//   - the stop condition -- an EOS id set and a token budget -- because a loop the host
//     must stop is a loop the host is in.
//
// The sampled id is ALSO written to `token_out`, an address the descriptor names, which is
// where the next step's embedding lookup reads its input. That is the actual closure: the
// token never leaves the card to come back as the next input.
//
#ifndef FOUNDRY_RESIDENT_TOPK_PROTOCOL_H
#define FOUNDRY_RESIDENT_TOPK_PROTOCOL_H

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
// The sixteen parameter words. Mirrors ft_tt_op_sample.h -- change both or neither.
// ---------------------------------------------------------------------------

/** Tiles in the logit row (Wt). 4080 for MiniCPM5-1B: 130,560 / 32, exactly. */
enum { kTopkTileCount = 0 };

/** Which of a tile's 32 rows carries the token being decoded, 0..31. The LM head runs at
 *  Mt = 1, so 32 token rows are produced and one is wanted; this names it. */
enum { kTopkRow = 1 };

/** How many (index, value) pairs to select. <= kTopkMaxSelect. */
enum { kTopkSelectCount = 2 };

/** Tiles per NoC barrier in the reader; must DIVIDE kTopkTileCount or a block straddles
 *  the CB wrap. The kernel repeats the host's arithmetic rather than trusting it. */
enum { kTopkDepth = 3 };

/** The logit row in DRAM: a NoC endpoint from ft_tt_coord, never a guess (spec §7.3). */
enum { kTopkLogitsX = 4 };
enum { kTopkLogitsY = 5 };
enum { kTopkLogitsBase = 6 };

/** The result block, in THIS core's L1. TOPK writes it, SAMPLE reads it. */
enum { kTopkResultAddress = 7 };

/** The per-turn sampler state, in THIS core's L1. See §5. */
enum { kTopkStateAddress = 8 };

/** Where SAMPLE writes the chosen token id -- normally the next step's embedding input, so
 *  the loop closes on the card. A zero address means "result block only". It names a
 *  kTopkTokenSlotBytes SLOT, not a word: see the hand-off note below. */
enum { kTopkTokenOutX = 9 };
enum { kTopkTokenOutY = 10 };
enum { kTopkTokenOutBase = 11 };

/**
 * Measurement flags. A RUNTIME word and not a build flag, on purpose: the A/B arms of the
 * cost split have to be the SAME BINARY or the measurement is of two implementations
 * rather than of the thing being measured (docs/tt_kernel_model.md §14, the
 * `weight_n_major` precedent).
 */
enum { kTopkFlags = 12 };

/** Pop every logit block WITHOUT selecting over it. The arm that measures the DRAM stream
 *  underneath the scan: full-arm minus this arm IS the scalar selection's cost. It writes
 *  an empty result block, so nothing can mistake it for an answer. */
enum { kTopkFlagDrainOnly = 1 };

/**
 * TAIL STATS: while selecting, also COUNT every logit of the row into the bins the host's
 * threshold keys define (the "tail stats" area after the pairs, §6). The selection and the
 * pairs are exactly what they are without the flag. A sampler whose top-k is off or reaches
 * past the N bounds the mass outside the N from these counts instead of reading the row.
 */
enum { kTopkFlagTailStats = 2 };

/** RED (proof only): the merge drops helper 2's selections. Mirrors FT_TT_OP_SAMPLE_FLAG_RED_DROP_HELPER. */
enum { kTopkFlagRedDropHelper = 4 };

/** With kTopkFlagTailStats: how many threshold keys the tail area holds, 1..kTopkTailMaxBins. */
enum { kTopkTailBins = 13 };

/** Word 14 is kTopkRows (below). Word 15 is kTopkHelperArea (below). */
enum { kTopkWords = 16 };

/** Most pairs one result block holds. 64 keeps the block inside one 512-byte L1 region and
 *  covers every top_k Pantheon serves (its production value is 40). */
enum { kTopkMaxSelect = 64 };

/** MULTI-ROW TOPK (2026-10-21): rows `row`..`row+rows-1` over the same tiles in one pass,
 *  row k's result block at result_address + k * kTopkResultBytes. 0 or 1 = one row.
 *  Mirrors FT_TT_OP_SAMPLE_ROWS / FT_TT_OP_SAMPLE_MAX_ROWS. */
enum { kTopkRows = 14 };
enum { kTopkMaxRows = 4 };

// ---------------------------------------------------------------------------
// The result block: a header the host can refuse, then N pairs
// ---------------------------------------------------------------------------
// Descending by value, ties by ascending index -- the ORDER `fc_topk_logits` returns, so a
// host that already consumes the K80's output consumes this unchanged. SAMPLE re-sorts to
// ascending index internally, because Pantheon's sparse sampler scans the vocabulary in
// index order and the cumulative draw is order-sensitive (engine.c sparse_sample()).

enum { kTopkResultMagic = 0 };  // kTopkResultMagicValue; refuses a wrong address
enum { kTopkResultCount = 1 };  // pairs actually written
enum { kTopkResultSequence = 2 };  // ++ per TOPK, so a stale block is visible
enum { kTopkResultTokenValid = 3 };  // SAMPLE writes 1 here after a draw
enum { kTopkResultToken = 4 };       // the sampled id, mirrored for the host
enum { kTopkResultHeaderWords = 8 };

/** Two words per pair: [vocabulary index, value as fp32 BITS]. The value is stored as fp32
 *  rather than bf16 because bf16 -> fp32 is exact (a 16-bit left shift) and every consumer
 *  -- this kernel's own sampler and the host reference both -- does fp32 arithmetic. */
enum { kTopkPairWords = 2 };
enum { kTopkPairIndex = 0 };
enum { kTopkPairValueBits = 1 };

enum { kTopkResultMagicValue = 0x46544B31 };  // "FTK1"

// ---------------------------------------------------------------------------
// THE TOKEN HAND-OFF IS AN ALIGNED 16-BYTE NoC WRITE, AND THAT COST TWO CORES
// ---------------------------------------------------------------------------
// The obvious way to hand four bytes to the next op is `noc_inline_dw_write`, and it is
// wrong here for a documented reason: **"The destination node can
// be either a Tensix core+L1 memory address or a PCIe controller; this API does not support
// DRAM addresses"** (the stock dataflow API; its watcher build has a sanitizer whose
// only job is to catch this). `token_out` normally names
// DRAM -- the embedding table's index array is not in anyone's L1 -- so the op was using the
// one destination class the call excludes.
//
// It does not fail loudly. On Blackhole a non-register destination takes the SPOOF path
// (the stock inline-write routine): the value is staged in
// a 16-byte L1 scratch slot and issued as a **4-byte** `noc_async_write`. The token LANDS --
// the on-card check that reads it back passed every time -- and the core is left unable to
// boot: the next `assert_reset` + firmware load comes up with BRISC out of reset, hanging
// before it releases its subordinates (`soft reset 0x00047000`, `ft_tt_core status`'s
// "boot handshake never finished"). MEASURED on Tensix 20 and 21, bisected to this one write
// on Tensix 27, which survives the identical run with the write suppressed.
//
// Four bytes is also below what this backend's OWN transfer contract allows: a DRAM WRITE
// takes the L1 alignment, 16 B in address AND in length (`ft_tt_noc_alignment()`,
// ft_tt_noc.c). `ft_tt_noc_check_transfer()` would have refused this transfer -- but it runs
// on the HOST, and nothing re-checks on the kernel side, which is where the transfer is
// actually issued. So the hand-off is now a legal transfer instead of a forbidden call:
// **16 bytes, 16-byte aligned at both ends, through plain `noc_async_write`** -- legal to L1
// and to DRAM alike, and needing no knowledge of which one the descriptor named (the kernel
// has an (x, y) and cannot tell).
//
// The consequence for a caller: `token_out` names a 16-byte SLOT. The id lands in its first
// word and the other three are zeroed.

/** First word of the 16-byte source slot the hand-off writes FROM, inside the result block.
 *  It IS kTopkResultToken, so the mirrored id and the shipped id are the same word and
 *  cannot disagree. */
enum { kTopkTokenSlotWords = kTopkResultToken };

/** Bytes moved by the hand-off, and the alignment both ends must satisfy. */
enum { kTopkTokenSlotBytes = 16 };
enum { kTopkTokenOutAlignment = 16 };

FTQ_STATIC_ASSERT((kTopkTokenSlotWords * 4u) % kTopkTokenOutAlignment == 0u,
              "the hand-off's L1 source must be aligned within a result block that is");
FTQ_STATIC_ASSERT(kTopkTokenSlotWords + kTopkTokenSlotBytes / 4u <= kTopkResultHeaderWords,
              "the hand-off slot must stay inside the result header, clear of the pairs");

// ---------------------------------------------------------------------------
// §6. The tail stats area, after the pairs
// ---------------------------------------------------------------------------
// The HOST writes `bins` threshold KEYS (ftk_key_from_bf16 order, strictly DESCENDING) at
// kTopkTailKeysOffset before it enqueues a kTopkFlagTailStats descriptor; the kernel reads
// them once and writes, at kTopkTailCountsOffset,
//     count[j]    = logits of the row whose key lies in [key[j], key[j-1])  (key[-1] = +inf)
//     count[bins] = logits whose key is below key[bins-1]
// and kTopkTailStatusMagic in the area's last word. Keys that are not strictly descending or
// a bin count out of range leave the status word 0 (no counts) instead of refusing the
// descriptor: the reader cannot see the area, and a refusal on one processor only would
// leave the other pushing into a buffer nobody drains (descriptor_is_executable's note).
// Cost: a 5-step binary search per logit on top of the reject test, paid only with the flag.

enum { kTopkTailMaxBins = 32 };
/** = kTopkResultHeaderWords + kTopkMaxSelect * kTopkPairWords (asserted below). */
enum { kTopkTailKeysOffset = 136 };
/** = kTopkTailKeysOffset + kTopkTailMaxBins (asserted below). */
enum { kTopkTailCountsOffset = 168 };
/** kTopkTailMaxBins + 1 counts, padded to 16 B; the LAST word is the status. */
enum { kTopkTailCountsWords = 36 };
enum { kTopkTailStatusWord = 35 };
enum { kTopkTailStatusMagic = 0x46545431 };  // "FTT1"

FTQ_STATIC_ASSERT(kTopkTailKeysOffset == kTopkResultHeaderWords + kTopkMaxSelect * kTopkPairWords,
                  "the tail area starts right after the last pair");
FTQ_STATIC_ASSERT(kTopkTailCountsOffset == kTopkTailKeysOffset + kTopkTailMaxBins,
                  "the counts follow the keys");
FTQ_STATIC_ASSERT(kTopkTailStatusWord >= kTopkTailMaxBins + 1 &&
                  kTopkTailStatusWord < kTopkTailCountsWords, "status clear of the counts");

/** Bytes of a result block: header, kTopkMaxSelect pairs, the tail stats area. */
enum { kTopkResultBytes = (kTopkTailCountsOffset + kTopkTailCountsWords) * 4u };

// ---------------------------------------------------------------------------
// The sampler state block (§5)
// ---------------------------------------------------------------------------

enum { kSampleStateMagic = 0 };
/** xorshift32. Never zero -- the host refuses a zero seed, because xorshift32 has a fixed
 *  point there and a "random" sampler that returns the same draw forever looks like a
 *  temperature bug for a long time. */
enum { kSampleStateRngState = 1 };
/** All four knobs are fp32 BITS, so the block is a plain uint32 array on both sides and
 *  the host never has to agree with the device about float endianness or promotion. */
enum { kSampleStateTemperatureBits = 2 };
enum { kSampleStateTopPBits = 3 };
enum { kSampleStateRepeatPenaltyBits = 4 };
enum { kSampleStateTopK = 5 };
enum { kSampleStateRepeatWindow = 6 };   // entries of the recent ring to consult
enum { kSampleStateRecentCursor = 7 };   // next write position in the recent ring
enum { kSampleStateRecentCount = 8 };    // entries written so far, saturating
enum { kSampleStateEmittedCount = 9 };   // tokens this state has produced
enum { kSampleStateEosCount = 10 };
enum { kSampleStateStopped = 11 };       // 1 once EOS or the budget was hit
enum { kSampleStateTokenBudget = 12 };
enum { kSampleStateLastToken = 13 };
enum { kSampleStateHeaderWords = 16 };

enum { kSampleStateMagicValue = 0x46544B53 };  // "FTKS"

/** The recent-token ring, appended to by the sampler itself. */
enum { kSampleRecentOffsetWords = kSampleStateHeaderWords };
enum { kSampleRecentCapacity = 64 };

/** The EOS id set. Small and linear; a model has one or two. */
enum { kSampleEosOffsetWords = kSampleRecentOffsetWords + kSampleRecentCapacity };
enum { kSampleEosCapacity = 4 };

/** The emitted-token ring the host drains lazily. */
enum { kSampleEmittedOffsetWords = kSampleEosOffsetWords + kSampleEosCapacity };
enum { kSampleEmittedCapacity = 256 };

enum { kSampleStateWords = kSampleEmittedOffsetWords + kSampleEmittedCapacity };
enum { kSampleStateBytes = kSampleStateWords * 4u };

// ---------------------------------------------------------------------------
// The order (§4)
// ---------------------------------------------------------------------------

/** bf16 bits -> an unsigned key that orders exactly as the float does. Branchless: the
 *  sign-broadcast form, so the winners' path has no unpredictable branch either. */
FTQ_INLINE uint32_t ftk_key_from_bf16(uint32_t bits)
{
    const uint32_t sign_mask = (uint32_t)((int32_t)(bits << 16) >> 31);
    return (bits ^ 0x8000u ^ (sign_mask & 0x7FFFu)) & 0xFFFFu;
}

/** bf16 bits -> fp32 bits. Exact: bf16 IS the top half of fp32. */
FTQ_INLINE uint32_t ftk_fp32_bits_from_bf16(uint32_t bits)
{
    return (bits & 0xFFFFu) << 16;
}

/**
 * The reject-path interval of §3: the set of raw bf16 patterns that BEAT `worst_bits`,
 * as one contiguous unsigned range `[low, low + span)`.
 *
 * `full` asks for the range that accepts everything, which is what the selection uses
 * before it has N entries.
 */
FTQ_INLINE void ftk_accept_range(uint32_t worst_bits, bool accept_all, uint32_t *low,
                                    uint32_t *span)
{
    if (accept_all) {
        *low = 0u;
        *span = 0x10000u;
    } else if (worst_bits < 0x8000u) {
        *low = worst_bits + 1u;
        *span = 0x8000u - *low;
    } else {
        *low = 0u;
        *span = worst_bits;
    }
}

/** One selected pair, held in the selector's registers/stack rather than in L1. */
typedef struct ftk_entry {
    uint32_t key;
    uint32_t index;
    uint32_t bf16_bits;
} ftk_entry;

/**
 * The selection state: `entries` sorted DESCENDING by key, ties by ascending index.
 *
 * `accept_low` / `accept_span` are the §3 interval, kept in step with `entries[count-1]`
 * and recomputed only when the N-th best changes.
 */
typedef struct ftk_select {
    ftk_entry entries[kTopkMaxSelect];
    uint32_t count;
    uint32_t capacity;
    uint32_t accept_low;
    uint32_t accept_span;
} ftk_select;

FTQ_INLINE void ftk_select_init(ftk_select *state, uint32_t capacity)
{
    state->count = 0u;
    state->capacity = (capacity > (uint32_t)kTopkMaxSelect) ? (uint32_t)kTopkMaxSelect : capacity;
    ftk_accept_range(0u, true, &state->accept_low, &state->accept_span);
}

FTQ_INLINE void ftk_select_refresh_range(ftk_select *state)
{
    const bool accept_all = state->count < state->capacity;

    ftk_accept_range(accept_all ? 0u : state->entries[state->capacity - 1u].bf16_bits, accept_all,
                     &state->accept_low, &state->accept_span);
}

/**
 * Offer one logit to the selection. THE ONLY place the comparator lives.
 *
 * Insertion is "walk back while the incumbent is strictly smaller", so an EQUAL key stops
 * the walk and the candidate lands AFTER it. Scanning in ascending index order, that is
 * exactly lowest-index-wins (§4) -- the tie rule is a property of where the walk stops,
 * not a separate test, which is why it cannot be forgotten in one of two branches.
 */
FTQ_INLINE void ftk_select_offer(ftk_select *state, uint32_t bf16_bits, uint32_t index)
{
    const uint32_t key = ftk_key_from_bf16(bf16_bits);
    uint32_t position = (state->count < state->capacity) ? state->count : state->capacity - 1u;

    if (state->count < state->capacity)
        state->count++;
    while (position > 0u && state->entries[position - 1u].key < key) {
        state->entries[position] = state->entries[position - 1u];
        position--;
    }
    state->entries[position].key = key;
    state->entries[position].index = index;
    state->entries[position].bf16_bits = bf16_bits;
    ftk_select_refresh_range(state);
}

/** §6: the tail bin of `key` among `bins` DESCENDING threshold keys — the first j with
 *  keys[j] <= key, or `bins` when every threshold is above it. Binary search, 5 steps at 32. */
FTQ_INLINE uint32_t ftk_tail_bin(const uint32_t *keys, uint32_t bins, uint32_t key)
{
    uint32_t low = 0u, high = bins;

    while (low < high) {
        const uint32_t mid = (low + high) >> 1;

        if (keys[mid] <= key)
            high = mid;
        else
            low = mid + 1u;
    }
    return low;
}

/** Whether `bf16_bits` is worth offering. One subtract and one unsigned compare (§3). */
FTQ_INLINE bool ftk_select_admits(const ftk_select *state, uint32_t bf16_bits)
{
    return (bf16_bits - state->accept_low) < state->accept_span;
}

// ---------------------------------------------------------------------------
// HELPERS (2026-10-21): the three TRISCs select too
// ---------------------------------------------------------------------------
// The scan is the op's cost (the selector is ~600 us a row on a 1,552-tile slice, the
// stream underneath ~430), and the three TRISCs of this core sat idle. With a helper area
// named, tile t of the row is scanned by processor t % 4 (0 = NCRISC, 1..3 = TRISC 0..2),
// each into its own selection per row; NCRISC merges the four exactly at the end. Every
// partial selection is the exact top-N of its tiles under the total order (key desc, index
// asc) because each processor still offers its tiles in ascending index order, so the
// merged N is the single-pass N, pair for pair.
//
// The area, in THIS core's L1 (the host places it after the result blocks):
//   word 0  page address of the block in flight          (NCRISC writes)
//   word 1  stamp of the block in flight                  (NCRISC writes, after word 0)
//   word 2  tile bytes, word 3 block depth               (NCRISC, before stamp 1)
//   word 4 + h  helper h's done stamp                     (helper h writes)
//   byte 64 + (h * kTopkMaxRows + k) * sizeof(ftk_select)  helper h's selection for row k
// Helpers take tiles by `t & 3` (stride 4: NCRISC + 3 TRISCs). A stamp is (descriptor cursor & 0xFFFF) << 16 | block + 1, so a stale word from an
// earlier descriptor never reads as current; kTopkStampFinal as the block = selections
// published. No helpers with tail stats or a drain (the predicate is the same on all four).

/** Word 15: L1 address of the helper area, 0 = NCRISC selects alone. */
enum { kTopkHelperArea = 15 };
enum { kTopkHelpers = 3 };
enum { kTopkHelperPage = 0, kTopkHelperStamp = 1, kTopkHelperTileBytes = 2, kTopkHelperDepth = 3 };
enum { kTopkHelperDone = 4 };
enum { kTopkHelperSelectOffset = 64 };
enum { kTopkStampFinal = 0xFFFF };
/** = FT_TT_OP_SAMPLE_HELPER_BYTES (ft_tt_op_sample.h). */
enum { kTopkHelperBytes = kTopkHelperSelectOffset + kTopkHelpers * kTopkMaxRows * 784 };
FTQ_STATIC_ASSERT(sizeof(ftk_select) == 784u, "the host sizes the helper area at 784 B a selection");

FTQ_INLINE uint32_t ftk_stamp(uint32_t cursor, uint32_t block)
{
    return ((cursor & 0xFFFFu) << 16) | (block & 0xFFFFu);
}

/** Whether a descriptor's parameters ask for helpers. Both sides compute it. */
FTQ_INLINE bool ftk_helpers_wanted(volatile FTQ_L1_PTR uint32_t *parameters)
{
    return parameters[kTopkHelperArea] != 0u &&
           (parameters[kTopkFlags] & (uint32_t)(kTopkFlagDrainOnly | kTopkFlagTailStats)) == 0u;
}

/** Offer the 16 logits of ONE face row; the accept interval is held in registers and
 *  reloaded only after an offer (the same decisions as testing `state` every time). */
FTQ_INLINE void ftk_offer_face_row(ftk_select *selection, const uint32_t *words,
                                   uint32_t first_index)
{
    uint32_t low = selection->accept_low;
    uint32_t span = selection->accept_span;

    for (uint32_t pair = 0u; pair < 8u; pair++) {
        const uint32_t packed = words[pair];
        const uint32_t lo = packed & 0xFFFFu;
        const uint32_t hi = packed >> 16;

        if ((lo - low) < span) {
            ftk_select_offer(selection, lo, first_index + pair * 2u);
            low = selection->accept_low;
            span = selection->accept_span;
        }
        if ((hi - low) < span) {
            ftk_select_offer(selection, hi, first_index + pair * 2u + 1u);
            low = selection->accept_low;
            span = selection->accept_span;
        }
    }
}

/** Offer one tile's 32 logits of token row `row` (two face rows, 512 bytes apart). */
FTQ_INLINE void ftk_offer_tile_row(ftk_select *selection, uint32_t tile_address, uint32_t row,
                                   uint32_t first_index)
{
    const uint32_t low_offset = (row >> 4) * 1024u + (row & 15u) * 32u;

    ftk_offer_face_row(selection, (const uint32_t *)(uintptr_t)(tile_address + low_offset),
                       first_index);
    ftk_offer_face_row(selection,
                       (const uint32_t *)(uintptr_t)(tile_address + low_offset + 512u),
                       first_index + 16u);
}

// ---------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------

/** xorshift32 (Marsaglia 2003, the 13/17/5 triple). Chosen because the device and the host
 *  reference must produce the SAME stream for the sampled-token gate to be exact, and a
 *  libc `rand()` cannot be that on a kernel with no libc. */
FTQ_INLINE uint32_t ftk_rng_next(uint32_t *state)
{
    uint32_t value = *state;

    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

/** A draw in [0, 1), matching the reference's `(float)rand() / (RAND_MAX + 1.0f)` in RANGE
 *  and in half-openness. 24 bits, because that is fp32's mantissa and anything finer is
 *  discarded by the conversion anyway. */
FTQ_INLINE float ftk_rng_next_unit_float(uint32_t *state)
{
    return (float)(ftk_rng_next(state) >> 8) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------------
// exp, in software, because the data-movement RISC-V has no FPU
// ---------------------------------------------------------------------------
// MEASURED: the data-movement -mcpu is `tt-bh` = `rv32im_zmmul_zaamo_zba_zbb`, ilp32 --
// integer only, no F extension. So every float operation here is a libgcc call and there is
// no libm to reach for. `expf` is therefore not available and is not merely slow.
//
// exp(x) = 2^(x*log2 e), split into an integer power of two applied to the exponent field
// and a degree-6 minimax polynomial for 2^f on f in [-0.5, 0.5]. The coefficients are
// PARSED out of this file by tests/test_topk_ops.c and compared against the host
// reference's, which is what keeps the two implementations from drifting -- a retyped
// constant is exactly the failure mode dev notes §4 catalogues.
//
// Only x <= 0 is ever passed (the sampler always subtracts the row maximum first), so the
// overflow half of the range is a refusal rather than a saturation.

static const float kFtkLog2E = 1.44269504088896340736f;
static const float kFtkExp2C1 = 0.69314718055994531f;
static const float kFtkExp2C2 = 0.24022650695910072f;
static const float kFtkExp2C3 = 0.05550410866482158f;
static const float kFtkExp2C4 = 0.00961812910762848f;
static const float kFtkExp2C5 = 0.00133335581467035f;
static const float kFtkExp2C6 = 0.00015403530393381f;

/** exp(x) for x <= 0. Returns 0.0f on underflow, exactly as expf() does. */
FTQ_INLINE float ftk_exp_nonpositive(float x)
{
    float scaled = x * kFtkLog2E;
    int32_t power;
    float fraction;
    float poly;
    uint32_t exponent_bits;
    float scale;

    if (scaled < -126.0f)
        return 0.0f;
    // `scaled` is non-positive, so truncation-toward-zero of (scaled - 0.5) IS the nearest
    // integer -- no rounding-mode call, and |fraction| <= 0.5 by construction.
    power = (int32_t)(scaled - 0.5f);
    fraction = scaled - (float)power;
    poly = kFtkExp2C6;
    poly = poly * fraction + kFtkExp2C5;
    poly = poly * fraction + kFtkExp2C4;
    poly = poly * fraction + kFtkExp2C3;
    poly = poly * fraction + kFtkExp2C2;
    poly = poly * fraction + kFtkExp2C1;
    poly = poly * fraction + 1.0f;
    exponent_bits = (uint32_t)(power + 127) << 23;
    __builtin_memcpy(&scale, &exponent_bits, sizeof(scale));
    return poly * scale;
}

#endif  // FOUNDRY_RESIDENT_TOPK_PROTOCOL_H
