// SPDX-License-Identifier: AGPL-3.0-or-later
//
// resident_gdn_protocol.h
//   The sixteen PROGRAM words the GATED-DELTA-NET op family reads out of its parameter
//   block. The host half is ft_tt_op_gdn.h; the two must agree word for word, and
//   tests/test_gdn_ops.c parses BOTH declarations off-hardware and asserts they do.
//
// ===========================================================================
// WHAT THE FAMILY IS, AND WHY IT IS AN OPCODE RATHER THAN A COMPOSITION
// ===========================================================================
// Qwen3.5's linear layers are a Gated-Delta-Net. Per SSM layer, per decode token, with
// S = 128 the state size, 16 k-heads and 32 v-heads (Pantheon's qwen35moe_ssm.c):
//
//   conv1d+SiLU over the whole conv_dim stream, primed from a rolling window
//   split into q | k | v, L2-normalise each q/k head over S, GQA-repeat q/k to 32 heads
//   per v-head h, with decay = exp(g_h) and beta_h:
//       (a) S *= decay                       scalar over the whole [S,S]
//       (b) pred_j = sum_i S[j][i] * k[i]     a row reduce
//       (c) delta_j = (v_j - pred_j) * beta
//       (d) S[j][i] += k[i] * delta_j         rank-1
//       (e) y_j = scale * sum_i S[j][i] * q[i]   READ AFTER THE WRITE
//
// Pantheon stores the state TRANSPOSED (s[h][j][i], row j contiguous over i) and that
// layout is what makes it a tile machine's problem: with rows=j and cols=i, (a) is a
// broadcast multiply, (b) and (e) are ROW REDUCES and (d) is a broadcast multiply plus an
// add. No new primitive is needed for any of them.
//
// It is nonetheless its own opcode for two reasons that are not tidiness:
//
//   - DISPATCH COUNT. 32 heads x 4 tile-rows expressed as separate descriptors is ~3,800
//     dispatches a token. The thing this op exists to remove is 0.225 s of host
//     recurrence; 3,800 dispatches would cost more than that. The head/tile-row loops are
//     INSIDE the kernel.
//   - IT CARRIES STATE ACROSS TOKENS. Two of its operands are read AND written -- the conv
//     window and the [S,S] fp32 state per v-head. Every other op in this backend is a pure
//     function of its inputs. A wrong answer here does not stay wrong for one token, it
//     compounds, which is why the acceptance gate is a 48-token completion.
//
// ===========================================================================
// (b) AND (e) ARE THE SHIPPED REDUCE, AND THAT IS MEASURED RATHER THAN ASSUMED
// ===========================================================================
// `ft_tensix_unpack_ab_reduce_row(data_cb, scaler_cb, data_tile, scaler_tile)` takes an
// INDEPENDENT scaler tile index per call; rms_norm passes 0 every iteration only because
// its scaler is one constant 1/W tile. Whether the scaler contributes PER REDUCE POSITION
// or as one broadcast constant decided whether (b) and (e) needed a kernel at all.
//
// MEASURED ON SILICON 2026-08-18 (tests/test_ops.c, FT_TT_OPS=reduce_scaler): the scaler is
// PER-POSITION. A 32-element non-uniform scaler seeded from pid+clock was recovered element
// for element, in the same index order, worst relative error 0.0077 and 0.0063 on two
// independent seeds, against 0.88 for the broadcast reading. The uniform-1/32 control
// recovered 1/32 on all 32 rows to 0.0002, so the probe could have failed and did not.
//
// So (b) is `reduce_row(cb_state, cb_k, tile, tile)` and (e) is the same call with q in the
// scaler CB. The scaler for those arms is 1.0 rather than 1/W. NO elementwise product and no
// pre-broadcast. q/k are staged ROW-REPLICATED, not as column tiles: the probe's own scaler
// tile carried s[c] at every [r][c], so it never tested the axis, and a row-replicated tile
// is ALSO the (d) rank-1 operand -- one staged tile for two uses.
//
// ===========================================================================
// WHY THERE ARE TWO PHASES AND NOT ONE MONOLITH
// ===========================================================================
// The two per-head norms in this block -- the q/k L2 norm and the output gated norm -- are
// 4 tile columns wide, over 16 and 32 row blocks. Every RMSNorm elsewhere in this model is
// n_embd/32 = 128 tile columns, and a reduce residency is placed for exactly ONE width
// (Pantheon's pn_tt_rms_norm refuses a mismatch by design). They therefore cannot run on
// the model's own norm core -- but they are ORDINARY RMS NORMS at a second width, not new
// arithmetic, so they belong to the shipped reduce family on a second narrow residency
// rather than inside this kernel. `L2(x) = x/sqrt(sum x^2)` and `RMSNorm(x,g) =
// x*g*rsqrt(mean x^2)` are the same function at `g = 1/sqrt(S)`.
//
// That leaves this family two phases, `kGdnPhaseConv` and `kGdnPhaseDelta`, run as two
// descriptors a layer with the narrow norms between and after them. Five descriptors a
// layer, ~120 a token -- which honours the dispatch-count reason above completely, since
// the thing that would have cost 3,800 was the per-head loop and that is still inside.
//
// THE SPLIT-POINT TABLE DOES NOT APPLY TO THIS SPLIT. Plan Phase 7.9 records that moving
// conv1d alone buys 1.00x and conv1d+L2+GQA is 1.33x WORSE. That table is about where the
// HOST takes over, and it is about the PCIe link. Here the host never takes over mid-block:
// q/k/v round-trip through CARD DRAM between phases, which is on-card bandwidth, not the
// 5.2 MB/s link. Nothing in the table is about a descriptor boundary.
//
// ===========================================================================
// THE ONE THING THAT STAYS ON THE HOST, AND WHY IT IS NOT A COMPROMISE
// ===========================================================================
// `decay = exp(A * softplus(z))` = `(1 + e^z)^A`, which needs a `pow` and therefore a
// `log`. The SFPU surface here is rsqrt / sigmoid / silu / exp / recip / square /
// binary_max and there is NO log and no pow. So the host computes decay and beta and sends
// them down: 32 scalars each, 256 B a layer in each direction, ~6 KB a token, ~1.2 ms.
// Against the 1,185,792 B and 0.225 s a host recurrence costs, that is 0.5%, and every
// value still crosses the link exactly once.
//
// ===========================================================================
// A CIRCULAR BUFFER MUST CLOSE ON THE CONSUMER'S INDEXED READ, NOT ONLY ON THE PUSH GRID
// ===========================================================================
// `ft_tensix_unpack_ab(cb, ..., tile)` is `read_pointer + tile * page_size` WITH NO WRAP, and
// a FIFO pointer moves back to its base on an EXACT HIT of the limit and on nothing else. So
// a compute thread that waits for `n` pages and then names tiles `0..n-1` needs the front low
// enough that the whole window still fits -- which is `pages % n == 0` for the INDEXED chunk
// of EVERY phase the buffer serves, on top of the push-grid rule
// `ft_tensix_cb_push_grid_is_safe()` already states for the producer.
//
// AND `pages % n == 0` IS NOT SUFFICIENT ON ITS OWN for a RESIDENT op. A descriptor's traffic
// is not a whole number of buffers -- this family moves 512 CB_STATE pages in PHASE DELTA and
// 1,280 in PHASE CONV -- so descriptor N+1 starts wherever N left off. The three kernels
// therefore REBASE every indexed buffer to its base at the top of every descriptor, the
// producers behind a whole-buffer reservation and the consumers unguarded (they have already
// popped everything). `ft_tensix_cb_rebase_{read,write}_pointer()` carries the argument.
//
// MEASURED ON SILICON 2026-08-18 as this family's whole wrong answer, on two buffers at once.
// CB_VECTOR held 8 pages against a DELTA window of `state_tiles + kGdnVectorConstants` = 7, so
// 24 of 32 v-heads read their v / decay / beta / scale out of the buffer and into CB_STAGE --
// exactly the 8 heads whose front was 0 or 1 scored 4e-3 against the position-0 closed form
// and the rest scored 1.0. CB_STATE held 8 against a CONV window of `depth*(taps-1)` = 3, so
// 2 blocks in 8 read past the end; that one is invisible at layer 0 of token 0 ALONE, because
// the L1 above CB_STATE is CB_SCALER and it is still zero until the first DELTA descriptor
// writes k/q replicas into it -- conv scored 6.9e-4 at layer 0 and 1e-2 to 2e-1 after it.
//
// ===========================================================================
// A BUFFER HAS ONE PRODUCER FOR THE LIFE OF THE RESIDENCY, NOT ONE PER DESCRIPTOR
// ===========================================================================
// The two push primitives keep INCOMPATIBLE BOOKS for the same `tiles_received` stream
// register. `ft_tensix_cb_push_back()` on a dataflow processor does `register += n` and keeps
// no local copy. `ft_tensix_compute_cb_push_back()` on a TRISC keeps a LOCAL
// `tiles.counters.tiles_received`, adds to that, and STORES it over the register. So the
// moment a buffer is pushed by the reader in one phase and by the compute thread in another,
// the compute's first push discards every increment the reader ever made, the register goes
// BACKWARDS, and the consumer's `(uint16_t)(received - acked)` in the stock wait-for-tiles routine wraps to
// ~65000: `cb_wait_front` on that buffer returns immediately for the rest of the run.
//
// MEASURED ON SILICON 2026-08-18. PHASE DELTA's compute-private working state shared CB_IN
// with PHASE CONV's qkv stream. PHASE CONV's PASS A survived, because it also waits on
// CB_VECTOR and CB_STATE and the reader pushes those AFTER the qkv read under the same
// barrier; PASS B's last tap waits on CB_IN and nothing else, so the compute thread free-ran
// and copied whatever the eight pages held. With an INJECTIVE stream staged (`qkv[c] = c`)
// the map read `s = t (mod 8)` for source tile `s` of output tile `t` at all 256 tiles, the
// lag growing from 8 tiles to 96 as the compute outran the reader. The one descriptor in a
// run that was CORRECT was SSM layer 0 at position 0 -- the only CONV never preceded by a
// DELTA. PHASE DELTA's working state is CB_WORK now.
//
// ===========================================================================
// THE GQA REPEAT IS AN ADDRESS COMPUTATION AND MUST NEVER BE MATERIALISED
// ===========================================================================
// The repeat is INTERLEAVED: `vh = r * n_k_heads + kh`, i.e. `kh = vh % n_k_heads`, NOT a
// blocked mapping -- Pantheon's own docs name a real gibberish bug from getting this
// backwards. In the reader it is one modulo on a head index: v-head `vh` stages the k/q
// columns of k-head `vh % n_k_heads`. Materialising it would DOUBLE q and k, which is the
// 1.33x-worse row of the split-point table.
//
#ifndef FOUNDRY_RESIDENT_GDN_PROTOCOL_H
#define FOUNDRY_RESIDENT_GDN_PROTOCOL_H

#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

/* The SFPI GCC fork's own pointer attribute -- an access through it goes to L1 rather than
 * to ordinary memory. Spelled here as well as in resident_queue_protocol.h so this header
 * does not depend on include order. */
#ifndef FTQ_L1_PTR
#define FTQ_L1_PTR __attribute__((rvtt_l1_ptr))
#endif

#ifndef FTQ_INLINE
#ifdef __cplusplus
#define FTQ_INLINE inline
#else
#define FTQ_INLINE static inline
#endif
#endif

// ---------------------------------------------------------------------------
// The sixteen words. Mirrors ft_tt_op_gdn.h -- change both or neither.
//
// WHY THE COORDINATES ARE PACKED TWO TO A WORD, stated because every other family in this
// backend spends a word each. This op has FIVE DRAM endpoints (an activation in, an
// activation out, two per-layer constants and two pieces of carried state) where rope has
// three. Spelled the rope way that is 5 pairs + 6 addresses + shape = 17 words, and
// FT_TT_QUEUE_PARAM_WORDS is 16 for EVERY family -- a seventeenth word changes the shared
// slot stride and breaks on the card. The pack is forced, not preferred, and x/y are 16-bit
// NoC coordinates so nothing is lost.
// ---------------------------------------------------------------------------

/** byte 0 phase, byte 1 state tiles (S/32), byte 2 v-heads, byte 3 k-heads. */
enum { kGdnShape = 0 };

/** Tile columns of the conv_dim stream: conv_dim/32, in the LOW HALF-WORD -- 8192/32 = 256
 *  does not fit in the byte the other shape fields use, and a shape that silently truncates
 *  is the failure this family can least afford. The HIGH half-word is the CONV column
 *  slice's high bytes: see kGdnByteColFirstHigh. */
enum { kGdnConvTileCols = 1 };
enum { kGdnConvTileColsMask = 0xFFFF };

/** byte 0 conv taps (ssm_conv, 4); the rolling window is taps-1 deep. */
enum { kGdnConvTaps = 2 };

/** Activation bank: x in the low half-word, y in the high.
 *
 *  **ONE WORD FOR TWO BUFFERS, SO THE INPUT AND THE OUTPUT ARE ONE DRAM ENDPOINT.** That is
 *  a CONTRACT the stager has to honour and NOTHING ON THE DEVICE CAN CHECK: the reader reads
 *  the input's bank correctly and the writer then puts the answer at the OUTPUT's offset
 *  inside that same bank. The descriptor completes, the retire is clean, the counters are
 *  right, and the output buffer is never written at all — the host reads back whatever it was
 *  allocated with. Measured on silicon 2026-08-18 as the whole of this op's first wrong
 *  answer, and named only because the conv result came back bit-identical with the compute
 *  kernel's SiLU deleted. `ft_tt_op_gdn_parameters()` refuses a mismatched pair. */
enum { kGdnActivationXy = 3 };
/** Phase CONV: the qkv projection, one tilized row of `conv_tile_cols` tiles.
 *  Phase DELTA: the staged q|k|v, three tilized matrices -- see the layout section. */
enum { kGdnActivationIn = 4 };
/** Phase CONV: conv_out. Phase DELTA: y, one row per v-head. */
enum { kGdnActivationOut = 5 };

/** Per-layer constants bank. */
enum { kGdnConstantXy = 6 };
/** Phase CONV only: the depthwise conv1d weight, as `taps` TAP-MAJOR PLANES of
 *  `conv_tile_cols` tiles -- NOT the oracle's row-major [conv_dim, taps]. See the layout
 *  section below; the stager transposes. */
enum { kGdnConstantWeight = 7 };
/** Phase DELTA only: decay[v_heads] then beta[v_heads], fp32, the host softplus's answer. */
enum { kGdnConstantGate = 8 };

/** Carried-state bank. Both operands below are READ AND WRITTEN by one descriptor. */
enum { kGdnStateXy = 9 };
/** Phase CONV only: the rolling conv window, `taps-1` TAP-MAJOR PLANES laid out exactly
 *  like the weight. Read AND written, so the roll is layout-preserving. */
enum { kGdnStateConv = 10 };
/** Phase DELTA only: the recurrent state, [v_heads, S, S] fp32, TRANSPOSED (row j over i). */
enum { kGdnStateRecurrent = 11 };

/** fp32 bit pattern of the (e) scale, 1/sqrt(S). Sent rather than derived because a
 *  reciprocal square root computed two ways is two answers that must agree. */
enum { kGdnScaleBits = 12 };

/** Bytes per tile of the state and activation operands. Float32 tiles are 4096 B where the
 *  bf16 the rest of this backend uses are 2048; see the note on formats below. */
enum { kGdnTileBytes = 13 };

/** Reserved for variant bits. Written 0 and never read, exactly as the vector protocol
 *  requires of a word a family does not use. */
enum { kGdnFlags = 14 };
/** M6 (reader only): build the DELTA phase's replicated k/q tiles with local NoC copies
 *  instead of RISC-V stores. Same bytes. */
enum { kGdnFlagNocReplicate = 8u };
enum { kGdnFlagRedNoCopy = 16u }; /* RED ONLY: skip the faces 2-3 copy */
/** M9 (reader + writer): PHASE CONV moves only row 0 of each tile (faces 0-1, 2 x 64 B). Its
 *  arithmetic is elementwise, so row 0 is exact whatever rows 1-31 of a CB page hold. */
enum { kGdnFlagConvStrip = 32u };
enum { kGdnFlagRedStripHalf = 64u }; /* RED ONLY: the reader skips the face-1 half */
/** S2 (writer only): write the carried state to the OTHER phase's state word (CONV: word 11,
 *  DELTA: word 10) instead of back over the one it read — same bank (word 9). The speculative
 *  verify keeps every row's state this way, with no copy. */
enum { kGdnFlagStateOut = 128u };
/** THE EXACT CARD GATES (reader, PHASE DELTA): kGdnConstantGate is the layer's TABLE BLOCK
 *  (ftgc_table.h) in the constant bank, and the reader builds the gate block itself from rows
 *  [row, row + rows) of band 0 of the alpha/beta strips the block names — the host's libm
 *  gates as lookup tables, no gate upload and no alpha/beta read back. */
enum { kGdnFlagTable = 512u };
/** S2c (reader + writer): THE ROW SELECT, flags bits 16-20. CONV (strip mode only) reads its
 *  stream input from row r and writes its stream output to row r — the window stays row 0;
 *  DELTA takes q/k/v from row r of the staged stream and writes y to row r. 0 = the shipped
 *  row-0 walk, byte for byte. Lets a k-row verify run the decode mixers on a k-row strip. */
enum { kGdnRowShift = 16 };
enum { kGdnRowMask = 31 };

// ---------------------------------------------------------------------------
// MULTI-ROW CONV (PM 6, strip mode only): the same row count runs the CONV of rows
// [row, row + R) in ONE descriptor, row by row, each row's work exactly the one-row
// descriptor's: row r convolves stream row `row + r` against the window it would have seen —
// tap t is element t + r of [live window planes 0..taps-2, stream rows row, row+1, ...] (the
// reader reads a stream row into row 0, the strip layout) — and its rolled window (elements
// r+1 .. r+taps-1) goes to EXTENSION word r, its output to row `row + r`. Nothing is carried in
// L1: every operand of row r is re-read, so the arithmetic and its operands are the per-row
// walk's, bit for bit; what is saved is R-1 descriptors' issue, completion and puller turns.
// ---------------------------------------------------------------------------
// MULTI-ROW DELTA (PM 5): flags bits 21-25 carry a ROW COUNT. 0 and 1 are the one-row walk,
// byte for byte. R > 1 runs the decode recurrence over rows [row, row + R) of the staged
// stream in ONE descriptor: per head the q/k/v tiles and the state are read ONCE, each row's
// updated state is carried in L1 (CB_CARRY, compute-owned) to the next row, and every row's
// state still goes out — row r's to the address in EXTENSION word r — and its y to row
// `row + r`. The per-element arithmetic is the one-row descriptor's, in the same order, on
// the same bits: the carry pages are packed from the SAME DST section the writer's state
// pages are (the fused state emit, delta_extras bit 0, which ships), so row r+1 unpacks
// exactly what it would have read back from DRAM.
//
// THE EXTENSION BLOCK is the NEXT parameter slot (16 words at `parameters + kGdnWords`): the
// DRAM address of row r's state output in the state bank (word 9), r < R. The host writes it
// before the descriptor; every processor reads it from its own L1.
//
// The gate blocks are row-major at `gate + r * ftg_gate_stride()`: decay[v] then beta[v] per
// row, each block rounded up to the 64 B read granule (Pantheon's per-row gate blocks).
// ---------------------------------------------------------------------------
/** RED ONLY (reader): every row of a multi-row descriptor stages row `row`'s q/k/v — must
 *  DIFFER, which proves the rows' own operands reach the recurrence. */
enum { kGdnFlagRedRowsRow0 = 1u << 26 };
/** G2a (reader only, PHASE CONV in strip mode): the stream input is the bf16 PROJECTION strip
 *  itself, not a widened fp32 copy -- word 4 its address, word 8 (DELTA-only, free in CONV)
 *  its NoC coordinate, 2048 B tiles. The reader reads row r's two 32 B face-row halves (each
 *  at its source's lane, ftg_bf16_half_lane) and widens each element into CB_IN row 0 by render_widen's arithmetic (`x + 0` on bf16 into
 *  fp32): a zero or denormal exponent gives +0, anything else is the bf16 bits << 16. Saves
 *  the render_widen descriptor. Word 3 stays the OUTPUT's. */
enum { kGdnFlagConvBf16In = 1u << 27 };
/** bf16 tile bytes of the G2a stream input. */
enum { kGdnBf16TileBytes = 2048 };

/** render_widen's element: bf16 `h` as the fp32 bits `h + 0` packs (DAZ/FTZ, -0 + 0 = +0). */
FTQ_INLINE uint32_t ftg_widen_bf16(uint32_t h)
{
    return (h & 0x7F80u) == 0u ? 0u : (h & 0xFFFFu) << 16;
}

/** Elements in one face row (a 16 x 16 face). */
enum { kGdnFaceRowElements = 16 };

/** The NoC's DRAM-read window: both ends of a read sit at the SAME offset inside it. */
enum { kGdnDramReadWindow = 64 };

/**
 * G2a: which 32 B half of a 64 B fp32 face row a bf16 face-row half read from `source` must
 * land in -- the half at the source's own offset in the 64 B window (0 or 32; bf16 face rows
 * are 32 B aligned). A read whose two ends are NOT congruent mod 64 is not refused; it lands
 * wrong, and an even row's half sits at 0 mod 64.
 */
FTQ_INLINE uint32_t ftg_bf16_half_lane(uint32_t source)
{
    return source & ((uint32_t)kGdnDramReadWindow / 2u);
}

/**
 * G2a, in place: one 64 B fp32 face row whose 16 bf16 elements sit in its 32 B half at `lane`
 * (ftg_bf16_half_lane) becomes those elements widened, element i landing in the word at byte
 * 4i. TOP half (lane 32, bf16 i at 32 + 2i): ascending i, each store only covers halves
 * already read (4i + 3 < 32 + 2(i + 1), i < 16). BOTTOM half (lane 0, bf16 i at 2i):
 * descending i, word i covers halves 2i and 2i + 1, both above i for i > 0 and so already
 * read; i = 0 reads before it stores.
 */
FTQ_INLINE void ftg_widen_face_row_at(volatile FTQ_L1_PTR uint32_t *row, uint32_t lane)
{
    volatile FTQ_L1_PTR uint16_t *halves =
        (volatile FTQ_L1_PTR uint16_t *)row + (lane ? (uint32_t)kGdnFaceRowElements : 0u);
    uint32_t i;

#ifdef FTG_WIDEN_RED_DESCENDING
    lane = !lane;
#endif
    if (lane) {
        for (i = 0; i < (uint32_t)kGdnFaceRowElements; i++)
            row[i] = ftg_widen_bf16(halves[i]);
    } else {
        for (i = (uint32_t)kGdnFaceRowElements; i-- > 0u;)
            row[i] = ftg_widen_bf16(halves[i]);
    }
}
enum { kGdnRowCountShift = 21 };
enum { kGdnRowCountMask = 31 };
enum { kGdnMultiRowMax = 16 };
/** The compute-owned carry buffer's CB index: a protocol constant, not a compile-time arg, so
 *  the shipped build lines are unchanged. Mirrors FT_TT_OP_GDN_CB_CARRY. */
enum { kGdnCbCarry = 9 };

/** Bytes between two rows' gate blocks: decay[v] + beta[v], fp32, rounded to 64 B. */
FTQ_INLINE uint32_t ftg_gate_stride(uint32_t v_heads)
{
    return (2u * v_heads * 4u + 63u) & ~63u;
}

/** TOTAL tiles this descriptor writes out -- all the writer reads. */
enum { kGdnTileCount = 15 };

enum { kGdnWords = 16 };

/** Where a byte lives in a packed word. Bytes rather than bit fields because a field a
 *  human has to read out of a hex dump on a stuck core is worth a nibble pair. */
enum { kGdnBytePhase = 0 };
enum { kGdnByteStateTiles = 1 };
enum { kGdnByteVHeads = 2 };
enum { kGdnByteKHeads = 3 };
enum { kGdnByteTaps = 0 };

// ---------------------------------------------------------------------------
// THE HEAD SLICE, AND WHY IT LIVES IN THE CONV-TAPS WORD
//
// PHASE DELTA divides EXACTLY by v-head: `DELTA(N heads) = 59.8 us + N x 103.4` measured over
// six points, and the op parallelises 1.96 / 3.79 / 7.12x on 2 / 4 / 8 cores. Dividing the
// heads gives the same series as replicating the whole descriptor across cores, so the work is
// embarrassingly parallel from both sides and the only term that does not divide is the 59.8 us
// host round trip. Every buffer this family places is sized per HEAD -- CB_STATE holds one
// tile-row, CB_SCALER one head's replicas -- so a member that owns four heads instead of
// thirty-two needs the IDENTICAL placement. Nothing in `resolve_cb_pages()` reads v_heads.
//
// `FT_TT_QUEUE_PARAM_WORDS` is 16 for every family and a seventeenth word moves the SHARED
// slot stride, so the slice cannot have a word of its own. The shape word's four bytes are
// full (phase, state tiles, v-heads, k-heads); the CONV-taps word uses ONE of its four and is
// meaningless in PHASE DELTA, so bytes 1 and 2 of it are where the slice goes.
//
// A COUNT OF ZERO MEANS THE WHOLE DESCRIPTOR, and that is the compatibility hinge rather than
// a convenience: every encoder that predates the slice leaves those bytes zero, and every one
// of them means "all v_heads from 0". A slice is opt-in on the HOST side and invisible to a
// descriptor that does not ask for one.
// ---------------------------------------------------------------------------
enum { kGdnByteHeadFirst = 1 };
enum { kGdnByteHeadCount = 2 };

// ---------------------------------------------------------------------------
// THE WIDE COLUMN SLICE: THE HIGH BYTES RIDE IN THE TILE-COLUMN WORD
//
// In PHASE CONV bytes 1 and 2 above are the column slice's LOW bytes, so on their own no slice
// may start or span past column 255 -- and Qwen3.8's stream is 320 tile columns (conv_dim
// 10240), so an 8- or 16-way split was refused. The tile-column word carries conv_dim/32 in
// its low half-word and nothing above it (65,535 tile columns is a 2M-wide stream), so its
// byte 2 is `col_first >> 8` and its byte 3 `col_count >> 8`. A slice under 256 columns, and
// every encoder written before this one, leaves those bytes zero: such descriptors are the
// same bytes and decode to the same slice. DELTA leaves the whole word zero.
// ---------------------------------------------------------------------------
enum { kGdnByteColFirstHigh = 2 };
enum { kGdnByteColCountHigh = 3 };

// ---------------------------------------------------------------------------
// THE DECLARED CONV DEPTH, AND WHY THE THREE-SIDED CONTRACT NEEDED A CARD-SIDE CHECK
//
// `FT_TT_OP_GDN_CONV_DEPTH` is one number on three sides: the HOST sizes CB_VECTOR, CB_STATE
// and CB_IN from it, the READER is built with it as compile-time arg 7, and the COMPUTE
// kernel as compile-time arg 9. Only ONE DIRECTION of a drift was ever caught. If the host
// places for a depth SMALLER than the kernels were built at, `conv_block_depth()` finds the
// buffers too shallow and returns 0, and every CONV descriptor refuses -- loud. If the host
// places for a depth LARGER, every one of those checks passes: a 112-page CB_VECTOR is a
// perfectly legal home for depth-1 blocks, so the kernels quietly run at their own smaller
// depth and the descriptor completes with the right answer. Nothing is wrong except the
// number you are about to write down.
//
// THAT IS THE SHAPE THAT HAS ALREADY BITTEN THIS PROGRAM. The reduce family's constants were
// built at 2048 under a 4096 program, scored 1.0e+00, and NOTHING REFUSED ANYWHERE. Here the
// silent direction does not corrupt an answer, it corrupts a MEASUREMENT -- which is worse
// for a depth sweep, because the arm reads green and reports the control's time.
//
// So the host DECLARES its own depth in byte 3 of the CONV-taps word -- the last free byte of
// the sixteen (byte 0 is the taps, 1 and 2 are the DELTA head slice) -- and both kernels
// compare it against the constant they were compiled with. The comparison is STRICT and there
// is no "unset" escape: a zero byte is a mismatch against any real depth, so an encoder that
// forgets to declare REFUSES instead of passing, which is the opposite of the head slice's
// compatibility hinge and deliberately so. PHASE DELTA does not read the depth and its taps
// word is zero by construction, so the check lives in the CONV branch only and a DELTA
// descriptor is byte-for-byte what it was.
// ---------------------------------------------------------------------------
enum { kGdnByteConvDepth = 3 };

/**
 * Does the depth the HOST sized its placement for match the one THIS image was built with?
 *
 * ONE implementation for the reader and the compute kernel, so the two cannot disagree about
 * the disagreement. Strict equality on purpose -- see the note above on why there is no
 * "0 means unset" escape.
 */
FTQ_INLINE bool ftg_conv_depth_agrees(uint32_t declared, uint32_t built)
{
    return declared == built;
}

/** Which half of the block this descriptor runs. 0 is deliberately not a phase, so a zeroed
 *  parameter block is an invalid program rather than a plausible one. */
enum { kGdnPhaseConv = 1 };  /**< conv1d + SiLU, rolls the window, splits into q|k|v */
enum { kGdnPhaseDelta = 2 }; /**< the four-pass delta rule over every v-head */

// ---------------------------------------------------------------------------
// WHERE A v-HEAD'S CONSTANTS SIT IN CB_VECTOR
//
// The reader pushes `state_tiles` v columns and then these three, in this order; the
// compute thread indexes them at `state_tiles + kGdnVector*`. They lived as a private enum
// in BOTH kernels, each carrying a comment saying the other had to agree — which is the
// exact shape of a contract that drifts. They are here now because the HOST needs the count
// too: `ft_tt_op_gdn_place_cbs()` cannot size CB_VECTOR without it, and the reader refuses a
// descriptor whose CB_VECTOR cannot hold `state_tiles + kGdnVectorConstants` pages.
// ---------------------------------------------------------------------------
enum { kGdnVectorDecay = 0 };
enum { kGdnVectorBeta = 1 };
enum { kGdnVectorScale = 2 };
enum { kGdnVectorConstants = 3 };

/** One descriptor's shape, decoded once at the top of the op. */
typedef struct ftg_shape {
    uint32_t phase;
    uint32_t state_tiles; /**< S / 32 */
    uint32_t v_heads;
    uint32_t k_heads;
    uint32_t conv_tile_cols;
    uint32_t conv_taps;
    uint32_t tile_bytes;
    uint32_t tile_count;
    uint32_t head_first; /**< PHASE DELTA: first v-head this member owns */
    uint32_t head_count; /**< PHASE DELTA: v-heads this member owns; `v_heads` if unsliced */
    /** PHASE CONV: the block depth the HOST sized the circular buffers for. Compared against
     *  each image's own compile-time constant -- see kGdnByteConvDepth. Zero in PHASE DELTA,
     *  which does not read the depth. */
    uint32_t declared_conv_depth;
    /** PHASE CONV column slice: the tile columns THIS member convolves, [col_first,
     *  col_first + col_count). The plane STRIDE stays `conv_tile_cols`. Whole stream if
     *  unsliced. See kGdnByteHeadFirst: in CONV the same two bytes carry the column range's
     *  low bytes, and kGdnByteColFirstHigh its high bytes. */
    uint32_t col_first;
    uint32_t col_count;
    /** PHASE DELTA rows: the first (the row select) and how many (1 = the one-row walk). */
    uint32_t row_first;
    uint32_t row_count;
} ftg_shape;

FTQ_INLINE uint32_t ftg_byte(uint32_t word, uint32_t index)
{
    return (word >> (index * 8u)) & 0xFFu;
}

FTQ_INLINE uint32_t ftg_xy_x(uint32_t word)
{
    return word & 0xFFFFu;
}

FTQ_INLINE uint32_t ftg_xy_y(uint32_t word)
{
    return (word >> 16) & 0xFFFFu;
}

FTQ_INLINE ftg_shape ftg_decode(volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t shape_word = parameters[kGdnShape];
    ftg_shape shape;

    shape.phase = ftg_byte(shape_word, kGdnBytePhase);
    shape.state_tiles = ftg_byte(shape_word, kGdnByteStateTiles);
    shape.v_heads = ftg_byte(shape_word, kGdnByteVHeads);
    shape.k_heads = ftg_byte(shape_word, kGdnByteKHeads);
    shape.conv_tile_cols = parameters[kGdnConvTileCols] & (uint32_t)kGdnConvTileColsMask;
    shape.conv_taps = ftg_byte(parameters[kGdnConvTaps], kGdnByteTaps);
    shape.tile_bytes = parameters[kGdnTileBytes];
    shape.tile_count = parameters[kGdnTileCount];
    /* The head slice. A count of ZERO is "the whole descriptor" -- see the note beside
     * kGdnByteHeadFirst; it is what every encoder written before the slice emits, and it is
     * what makes an unsliced descriptor byte-identical to the one that shipped. */
    shape.head_first = ftg_byte(parameters[kGdnConvTaps], kGdnByteHeadFirst);
    shape.head_count = ftg_byte(parameters[kGdnConvTaps], kGdnByteHeadCount);
    /* The host's own CONV block depth. Read UNCONDITIONALLY and compared only in the CONV
     * branch: a DELTA descriptor's taps word is zero and DELTA does not read the depth. */
    shape.declared_conv_depth = ftg_byte(parameters[kGdnConvTaps], kGdnByteConvDepth);
    /* PHASE CONV READS BYTES 1 AND 2 AS A COLUMN RANGE. The convolution divides by CHANNEL:
     * its weight and window are tap-major planes of `conv_tile_cols` tiles, so a member owns
     * a range of tile columns in EVERY plane at the full plane stride. A count of zero is the
     * whole stream — every CONV encoder before the slice left these bytes zero, because the
     * encoder refused a CONV head slice — and a CONV descriptor owns every head. */
    shape.col_first = 0u;
    shape.col_count = shape.conv_tile_cols;
    if (shape.phase == (uint32_t)kGdnPhaseConv) {
        /* The high bytes from the tile-column word (kGdnByteColFirstHigh): zero for any
         * slice under 256 columns, so the narrow encoding decodes exactly as before. */
        const uint32_t count = shape.head_count |
                               (ftg_byte(parameters[kGdnConvTileCols], kGdnByteColCountHigh) << 8);

        shape.col_first = shape.head_first |
                          (ftg_byte(parameters[kGdnConvTileCols], kGdnByteColFirstHigh) << 8);
        if (count != 0u)
            shape.col_count = count;
        shape.head_first = 0u;
        shape.head_count = 0u;
    }
    if (shape.head_count == 0u) {
        shape.head_first = 0u;
        shape.head_count = shape.v_heads;
    }
    shape.row_first = (parameters[kGdnFlags] >> kGdnRowShift) & (uint32_t)kGdnRowMask;
    shape.row_count = (parameters[kGdnFlags] >> kGdnRowCountShift) & (uint32_t)kGdnRowCountMask;
    if (shape.row_count == 0u)
        shape.row_count = 1u;
    return shape;
}

/**
 * Is this a shape this binary can execute?
 *
 * Checked on the DEVICE and not only on the host, for the reason the reduce family's
 * equivalent gives: the failure a bad shape produces is not a wrong answer but a blocked
 * triad -- a writer waiting on an output tile nobody will produce, until its bound expires.
 * A refusal names it in one word instead.
 *
 * The GQA divisibility check is here rather than only on the host because getting the
 * repeat backwards is a KNOWN bug in this architecture, not a hypothetical one.
 */
FTQ_INLINE bool ftg_shape_is_supported(const ftg_shape *shape)
{
    if (shape->phase != kGdnPhaseConv && shape->phase != kGdnPhaseDelta)
        return false;
    if (shape->state_tiles == 0u || shape->state_tiles > 8u)
        return false;
    if (shape->v_heads == 0u || shape->k_heads == 0u)
        return false;
    if (shape->v_heads % shape->k_heads != 0u)
        return false;
    if (shape->tile_count == 0u)
        return false;
    /* The head slice has to name heads this descriptor's operands actually have. A slice
     * that runs off the end would address a state region belonging to no head and scatter a
     * y row into a tile the layout never allocated -- and both are clean retires. */
    if (shape->head_count == 0u || shape->head_first + shape->head_count > shape->v_heads)
        return false;
    /* MULTI-ROW: at most kGdnMultiRowMax rows, all of them inside one tile. DELTA since PM 5;
     * CONV since PM 6 (strip mode only — the reader and writer refuse it otherwise). */
    if (shape->row_count > 1u &&
        (shape->row_count > (uint32_t)kGdnMultiRowMax || shape->row_first + shape->row_count > 32u))
        return false;
    if (shape->phase == kGdnPhaseConv) {
        if (shape->conv_tile_cols == 0u)
            return false;
        /* The column slice has to name columns the stream has. */
        if (shape->col_count == 0u || shape->col_first + shape->col_count > shape->conv_tile_cols)
            return false;
        /* A one-tap conv has no window to roll, which would make the carried state
         * write-only -- almost certainly a shape that was meant to be something else. */
        if (shape->conv_taps < 2u || shape->conv_taps > 8u)
            return false;
    }
    return true;
}

// ===========================================================================
// THE DRAM LAYOUTS, SETTLED. Every address arithmetic below is HERE and not in a kernel,
// because the host stager and the reader are the two ends of a contract nothing checks at
// run time: a stager that disagrees produces plausible numbers from the first token instead
// of an error. The reference is `pn_qwen35moe_ssm_recurrence()` in Pantheon's
// `src/models/qwen35moe_ssm.c`, which defines the op value for value.
//
// A "TILIZED [rows, cols] MATRIX" below means what `ft_tt_tilize_strip()` produces:
// `ceil(rows/32)` row blocks in order, each `ceil(cols/32)` face-ordered 32x32 tiles in
// order, so element [r][c] lands at tile `(r/32)*ceil(cols/32) + c/32`, position
// [r%32][c%32]. That is the layout every activation and every weight in this backend already
// uses; nothing here is a private convention.
//
// ---------------------------------------------------------------------------
// kGdnActivationIn, PHASE DELTA -- ONE ROW-0 STRIP, head-major: q | k | v
// ---------------------------------------------------------------------------
//     q : `k_heads * state_tiles` tiles   L2-normalised, NOT GQA-repeated
//     k : `k_heads * state_tiles` tiles   L2-normalised, NOT GQA-repeated
//     v : `v_heads * state_tiles` tiles
//
// A head's S values live in `state_tiles` CONSECUTIVE tiles, all of them in ROW 0 -- i.e. the
// whole block is the tilized `[1, heads*S]` strip `ft_tt_tilize_strip(rows = 1)` produces, and
// head `h` starts at tile `h * state_tiles`. Rows 1..31 are zero and nothing reads them.
//
// **THIS IS THE SAME GEOMETRY AS PHASE CONV'S OUTPUT, AND THAT IS THE WHOLE POINT.** conv_out
// is a `[1, conv_dim]` strip whose channel `c` sits at tile `c/32`, row 0; the delta rule's
// q, k and v ARE channels [0, key_dim), [key_dim, 2*key_dim) and [2*key_dim, conv_dim) of that
// same strip. So PHASE CONV's output buffer IS PHASE DELTA's input buffer with no copy, no
// re-tilization and no host round trip, and the per-head L2 norm that sits between them is an
// ordinary `row_blocks = 2*k_heads, tile_cols = state_tiles` reduce over its first
// `2 * k_heads * state_tiles` tiles, written back over its own input.
//
// It used to be three tilized `[heads, S]` matrices with ONE ROW PER HEAD, 12 tiles at
// Ornith's shape against 256 now. That packing was 32x denser in DRAM and cost nothing to
// read -- a head still needed the whole `state_tiles`-tile row block to get one row of it --
// but it was a DIFFERENT GEOMETRY from the strip every projection, every reduce and every
// elementwise op in this backend produces, so the host had to de-tilize conv_out, L2-normalise
// on the CPU and re-tilize, which is 49,408 B a layer over PCIe and 86% of a decode token.
// Head-major row-0 tiles are what let the block run WHOLE on the card.
//
// The three blocks are not the same size -- q and k are per K-HEAD and v is per V-HEAD.
//
// ---------------------------------------------------------------------------
// kGdnConstantWeight and kGdnStateConv, PHASE CONV -- TAP-MAJOR PLANES, not [conv_dim, taps]
// ---------------------------------------------------------------------------
// The compute thread multiplies a whole tile of the qkv stream by a whole tile of the weight,
// so a weight tile has to line channel c up with its OWN tap-t coefficient at the SAME
// position. That makes the operand `taps` PLANES of `conv_tile_cols` tiles, plane t holding
// w[.][t] laid out exactly like the stream -- and the oracle's `layer->ssm_conv1d` is
// row-major `[conv_dim][taps]`, `w[c*taps + tap]`. THE STAGER MUST TRANSPOSE IT; a reshape
// gives a wrong answer with no error anywhere. This header used to say "[conv_dim, taps]",
// which is the oracle's order and not the card's.
//
// The rolling window is the same shape, `taps-1` planes, and it is READ AND WRITTEN in that
// layout -- so the transpose is a one-time staging cost and the roll is layout-preserving.
//
// ---------------------------------------------------------------------------
// kGdnConstantGate, PHASE DELTA -- A FLAT fp32 ARRAY, the one operand that is not tiles
// ---------------------------------------------------------------------------
// `decay[v_heads]` then `beta[v_heads]`, contiguous fp32, indexed by head. The reader pulls it
// with ONE NoC read whose length is rounded UP to the 64-byte DRAM granule, so the allocation
// must be at least `(2*v_heads*4 + 63) & ~63` bytes or the read runs off the end of it.
//
// ---------------------------------------------------------------------------
// kGdnActivationOut and kGdnStateRecurrent
// ---------------------------------------------------------------------------
// y is a ROW-0 STRIP, head-major: v-head `h`'s S values are ROW 0 of tiles
// [h*state_tiles, +state_tiles). Same geometry as the delta input and as conv_out, for the
// same reason -- the gated RMS norm that follows is then a `row_blocks = v_heads,
// tile_cols = state_tiles` reduce over the strip, and the `silu(z) *` that follows THAT is an
// elementwise op against `ssm_z`, which is a strip because a projection cannot produce
// anything else. The writer still transposes each head's answer out of the column the (e)
// reduce produces; what changed is only WHERE it lands.
//
// The recurrent state is [v_heads, S, S] fp32, TRANSPOSED per head (the oracle's
// `s[h*S*S + j*S + i] == S_h[i][j]`, row j contiguous over i), each head tilized on its own
// into `state_tiles * state_tiles` tiles in row-major tile order. Head-major.
// ===========================================================================

/** Which of the three staged streams, in the order they sit at kGdnActivationIn. */
enum { kGdnStreamQ = 0 };
enum { kGdnStreamK = 1 };
enum { kGdnStreamV = 2 };

/** Tiles of the q, k or v block at kGdnActivationIn. q and k are per-K-HEAD. */
FTQ_INLINE uint32_t ftg_delta_stream_tiles(const ftg_shape *shape, uint32_t stream)
{
    const uint32_t heads = (stream == (uint32_t)kGdnStreamV) ? shape->v_heads : shape->k_heads;

    return heads * shape->state_tiles;
}

/** First tile of the q, k or v block at kGdnActivationIn. */
FTQ_INLINE uint32_t ftg_delta_stream_base(const ftg_shape *shape, uint32_t stream)
{
    const uint32_t k_block = shape->k_heads * shape->state_tiles;

    if (stream == (uint32_t)kGdnStreamV)
        return 2u * k_block;
    return stream * k_block;
}

/** First tile of head `head`'s S values inside the stream block that carries it. */
FTQ_INLINE uint32_t ftg_delta_head_tile(const ftg_shape *shape, uint32_t stream, uint32_t head)
{
    return ftg_delta_stream_base(shape, stream) + head * shape->state_tiles;
}

/** Tiles the whole staged q|k|v input occupies -- what a stager allocates. */
FTQ_INLINE uint32_t ftg_delta_input_tiles(const ftg_shape *shape)
{
    return ftg_delta_stream_base(shape, (uint32_t)kGdnStreamV) +
           ftg_delta_stream_tiles(shape, (uint32_t)kGdnStreamV);
}

/** Tile `column` of tap plane `tap`, in the conv weight OR in the rolling window. */
FTQ_INLINE uint32_t ftg_conv_plane_tile(const ftg_shape *shape, uint32_t tap, uint32_t column)
{
    return tap * shape->conv_tile_cols + column;
}

/** Tile of the carried state holding rows [32*tile_row, +32) and columns [32*tile_col, +32)
 *  of v-head `head`. Head-major, then row-major over the tiles inside the head. */
FTQ_INLINE uint32_t ftg_state_tile(const ftg_shape *shape, uint32_t head, uint32_t tile_row,
                                   uint32_t tile_col)
{
    return head * shape->state_tiles * shape->state_tiles + tile_row * shape->state_tiles +
           tile_col;
}

/** Output tile of y carrying v-head `head`'s elements [32*tile_row, +32). y is a ROW-0 STRIP,
 *  head-major, so the head's row inside that tile is always 0 (`kGdnYRow`). */
FTQ_INLINE uint32_t ftg_y_tile(const ftg_shape *shape, uint32_t head, uint32_t tile_row)
{
    return head * shape->state_tiles + tile_row;
}

/** The row a strip's data occupies inside every tile of it -- the delta input's q, k and v and
 *  the y output alike. Named rather than spelled `0` because it used to be `head % 32` on both
 *  sides, and both the reader's source row and the writer's destination row are arguments. */
enum { kGdnStripRow = 0 };

/**
 * The k-head whose q/k columns v-head `v_head` reads: the INTERLEAVED GQA repeat.
 *
 * `vh = r * n_k_heads + kh` in Pantheon's qwen35moe_ssm.c, so `kh = vh % n_k_heads`. This
 * is the whole of the repeat -- an address computation, never a materialised tensor. It is
 * a named function rather than an inline modulo because the blocked mapping
 * (`vh / kv_groups`) is the plausible wrong answer and has produced gibberish before.
 */
FTQ_INLINE uint32_t ftg_k_head_for(uint32_t v_head, uint32_t k_heads)
{
    return v_head % k_heads;
}

/** Tiles of carried recurrent state one v-head owns: [S,S] in 32x32 tiles. */
FTQ_INLINE uint32_t ftg_state_tiles_per_head(const ftg_shape *shape)
{
    return shape->state_tiles * shape->state_tiles;
}

// ---------------------------------------------------------------------------
// HOW MANY TILES PHASE DELTA PUTS IN DST BETWEEN ONE ACQUIRE AND ONE COMMIT
//
// MEASURED ON SILICON 2026-08-18. The DELTA descriptor costs 3,368 us and is bound by
// NEITHER compute (95 GFLOP/s on a Tensix that can issue ~1,400) NOR bandwidth (1.27 GB/s
// against 3-5% of one core's DRAM). It is bound by STRUCTURE: ~2,688 single-tile
// `wait_for_dest -> one tile op -> commit_dst -> wait_front` round trips a descriptor, each
// with an unpacker `*_init` in front, at ~1,250 ns where the tile arithmetic is ~64 cycles.
// The fixed part of that round trip is per ACQUIRE, not per tile: a `ZEROACC CLR_HALF` over
// the whole DST half, two cross-thread semaphore hops and a full ADDR_MOD + MOP reprogram,
// none of which get bigger when four tiles ride in one section instead of one.
//
// `reduce_row_sum` in the compute kernel already batches this way -- one acquire, `tiles`
// accumulating reduce ops, one commit -- so the pattern is not new to the family; the three
// per-tile loops around it simply never used more than DST index 0.
//
// THE CEILING IS FOUR AND IT IS NOT A TUNING CHOICE. DST is 16 tiles of 32x32; DST_SYNC half
// gives MATH one half at a time, and `dst_accum_mode 1` -- MANDATORY for this family, since
// it is the only op carrying state ACROSS TOKENS -- halves it again. Four is also exactly
// Ornith's `state_tiles`, so every DELTA loop batches whole. A `dst_batch_max` above the
// ceiling would silently alias DST tiles onto each other and produce a smoothly wrong
// recurrence, so the clamp is here rather than in either kernel's copy of the arithmetic.
//
// THE BATCH MUST ALSO DIVIDE `state_tiles`, because every buffer it touches closes on the
// CONSUMER'S INDEXED READ (see the CB grid note above): a block that does not divide the
// tile-row leaves CB_WORK's front somewhere other than where the next block expects it.
// ---------------------------------------------------------------------------
enum { kGdnDstBatchCeiling = 4 };

/**
 * DST tiles one PHASE DELTA block writes, from the shared compile-time `dst_batch_max`.
 *
 * ONE implementation for the reader, the compute kernel and the host, because it is the same
 * three-sided contract `FT_TT_OP_GDN_CONV_DEPTH` is and that one drifted. `1` is the identity
 * value: every batched loop collapses to the per-tile sequence that shipped, which is what
 * makes a `dst_batch_max = 1` build the control arm for a `dst_batch_max = 4` one.
 */
FTQ_INLINE uint32_t ftg_dst_batch(uint32_t state_tiles, uint32_t dst_batch_max)
{
    uint32_t batch = dst_batch_max;

    if (batch == 0u || state_tiles == 0u)
        return 1u;
    if (batch > (uint32_t)kGdnDstBatchCeiling)
        batch = (uint32_t)kGdnDstBatchCeiling;
    if (batch > state_tiles)
        batch = state_tiles;
    while (batch > 1u && (state_tiles % batch) != 0u)
        batch--;
    return batch;
}

// ---------------------------------------------------------------------------
// HOW MANY CB_OUT PAGES THE WRITER PUTS BEHIND ONE NoC WRITE BARRIER
//
// A NoC write barrier is a PER-CORE DRAIN, not a per-transaction wait: it spins until every
// write this core has issued has been ACKED by its destination. That is why the block
// reader takes one for a whole block instead of one per tile pair and is worth 5.9x
// (docs/tt_kernel_model.md 12), and it is why "one barrier per stream group, never per tile"
// is a rule of this backend rather than a preference. PHASE DELTA's writer took ONE PER
// TILE -- `v_heads * state_tiles * (state_tiles + 1)` = 640 a descriptor at Ornith's shape.
//
// THE GROUP CANNOT SIMPLY BE MADE BIGGER, and the reason is the SOURCE of the write. Every
// write's source is the CB_OUT page itself, so a page popped while its write is still in
// flight is a page the compute thread is free to pack over -- a barrier batched without
// moving the pops behind it trades a drain for silent corruption. So the pops move behind
// the barrier and CB_OUT has to hold TWO groups: one being drained, one being filled.
//
// RUNGS, NOT A FREE INTEGER, because the host sizes CB_OUT from this same function and the
// two must agree:
//
//   0  one barrier per TILE     -- the sequence that shipped, and what a SHORT BUILD LINE
//                                  folds to, since an out-of-range compile-time arg reads a
//                                  literal 0 with no diagnostic (FT_TT_OP_GDN_CONV_DEPTH)
//   1  one barrier per TILE-ROW -- `state_tiles` state pages then one y page: exactly the
//                                  stream group the writer's own split is defined over
//   2  one barrier per HEAD     -- `state_tiles` tile-rows of the same
//
// Rung 0 is the identity, which is what makes a rung-0 build the exact control arm for a
// rung-1 one. The kernel CLAMPS the result to half the placed page count, so a build at
// rung 2 under a host that did not grow CB_OUT degrades to the shipped sequence instead of
// hanging on an unsatisfiable `cb_wait_front` -- the failure the DST-batch round hit on
// silicon when `reserve_back(cb_out, 4)` met a two-page buffer.
//
// PHASE CONV STAYS AT RUNG 0 ON EVERY ARM of this round. It drains the same CB_OUT, so a
// CONV descriptor that does not move is the measurement that says the RESIZE on its own
// bought nothing and the barrier count is what moved DELTA.
// THE CEILING IS AN ARENA BOUND, NOT A TASTE ONE. CB_OUT is placed at twice the group, and
// at S = 256 (8 state tiles, 8 taps) an unclamped rung 2 asks for 144 pages -- 576 KiB, which
// puts that shape's ten buffers 13 pages over the 330-page arena and makes
// `ft_tt_op_gdn_place_cbs()` return ENOMEM for a shape the family supports. Measured
// off-hardware by test_gdn_ops.c before it ever reached a card. 20 pages is rung 2 at
// Ornith exactly, so the ceiling costs the shape being measured nothing.
enum { kGdnWriterBlockCeiling = 20 };

FTQ_INLINE uint32_t ftg_writer_block(uint32_t state_tiles, uint32_t rung)
{
    uint32_t block;

    if (state_tiles == 0u || rung == 0u)
        return 1u;
    block = (rung == 1u) ? state_tiles + 1u : state_tiles * (state_tiles + 1u);
    if (block > (uint32_t)kGdnWriterBlockCeiling)
        block = (uint32_t)kGdnWriterBlockCeiling;
    return block;
}

#endif  // FOUNDRY_RESIDENT_GDN_PROTOCOL_H
