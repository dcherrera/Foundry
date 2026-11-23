// SPDX-License-Identifier: AGPL-3.0-or-later
//
// resident_gdnc_protocol.h
//   The sixteen PROGRAM words the CHUNKWISE GATED-DELTA-NET op family reads out of its
//   parameter block. The host half is ft_tt_op_gdnc.h; the two must agree word for word, and
//   tests/test_gdnc_ops.c parses BOTH declarations off-hardware and asserts they do.
//
// ===========================================================================
// WHY THIS IS A SECOND FAMILY AND NOT A THIRD PHASE OF `gdn`
// ===========================================================================
// The decode family (`resident_gdn_protocol.h`) runs ONE position per descriptor and is
// bound to that by construction: the conv is an elementwise tile multiply of a `[1, N]`
// ROW-0 STRIP, and the delta rule's carried state is advanced exactly once. Prefill wants C
// positions at a time, which turns the convolution into a BANDED MATMUL and needs
// `matmul_tiles` in the compute image, a different circular-buffer map and a different
// reader. So it is a separate program:
//
//   - THE SHIPPED `gdn` ELFs STAY BYTE-IDENTICAL. Decode's acceptance is a 48-token `cmp`
//     against the CPU oracle, and the cheapest way to keep that true is to not rebuild the
//     binary it runs on. `make check-program-provenance` says so mechanically.
//   - The compute image needs the matmul REPLAY BUFFER and its two MOP templates
//     (ft_tensix_compute_matmul.h). The per-processor text slab is 0x3000 = 12,288 B and
//     `gdn`'s TRISC images already reach 9,020 B; adding matmul to them is a size question
//     with a silent failure mode (an over-slab image does not load and nothing says so).
//
// ===========================================================================
// THE GEOMETRY, WHICH IS THE WHOLE DESIGN DECISION
// ===========================================================================
// **A CHUNK OF C POSITIONS IS THE SAME TILIZED STRIP A SINGLE POSITION IS, WITH ROWS 1..C-1
// CARRYING POSITIONS 1..C-1 INSTEAD OF PADDING.**
//
// `ft_tt_tilize_strip(rows = 1, cols = N)` and `ft_tt_tilize_strip(rows = C, cols = N)` for
// C <= 32 produce the SAME `ceil(N/32)` tiles at the SAME addresses; the second one fills
// rows 1..C-1 where the first leaves zeros. So for C <= 32:
//
//   - the DRAM layout of every activation is UNCHANGED. `kGdncActivationIn` is
//     `conv_tile_cols` tiles, exactly as `kGdnActivationIn` is;
//   - the four SSM projections that produce it are UNCHANGED — a matvec is `Mt = 1` and one
//     row of a 32-row tile carries the answer while the other 31 are padding the FPU computed
//     anyway. MEASURED: 1 live row 448.4 us against 32 live rows 446.7 us, ratio 0.9960;
//   - the per-head L2 and RMS norms are UNCHANGED. They are `row_blocks = heads` reduces over
//     a `[rows, heads*S]` strip, and a ROW reduce reduces ALONG the row — so row `t` of row
//     block `h` is position `t` of head `h`, normalised independently. C positions come out
//     of one descriptor for free;
//   - `render_widen` / `render_narrow` / `silu_mul` / the residual adds are whole-tile
//     elementwise ops and batch for free.
//
// **THE ONLY TWO THINGS IN AN SSM LAYER THAT DO NOT BATCH ARE THE CONVOLUTION AND THE DELTA
// RULE**, because both are recurrences in the row axis. This family is the convolution.
//
// C = 32 IS FORCED, NOT CHOSEN. A tile is 32 rows and the broadcast/matmul surface has no
// sub-tile row addressing, so a chunk larger than 32 is two tile-rows and a chunk smaller
// than 32 is a 32-row tile with a short live prefix. C is therefore a free parameter only
// inside [taps-1, 32]; the sweep is over that range and over nothing else.
//
// ===========================================================================
// THE CONVOLUTION AS A BANDED MATMUL
// ===========================================================================
// The oracle (`pn_qwen35moe_conv1d_silu` in Pantheon's qwen35moe_ssm.c) is, per channel c and
// absolute position u, with T = taps and the window plane p holding position -(T-1)+p:
//
//     out[u][c] = SiLU( sum_{tau<T} x[u + tau - (T-1)][c] * w[c][tau] )
//
// Write d = (T-1) - tau, so the input row is u - d. Inside a chunk of C rows, row t needs:
//
//     t >= d          row t-d of the chunk's own input X
//     t <  d          window plane p = t + tau, which is row 0 of that plane's tile
//
// Both are LINEAR MAPS ON THE ROW AXIS, i.e. 32x32 0/1 matrices, and both are the SAME for
// every channel, every layer and every chunk. Stack the window's `T-1` planes into ONE tile
// per column, row p = plane p (this is what the reader assembles), and the whole convolution
// for tile column c is
//
//     out_c = SiLU( sum_{tau<T} [ S_tau . (X_c (x) W_tau) + C_tau . (Win_c (x) W_tau) ] )
//
// with `(x)` an elementwise tile multiply against the ROW-REPLICATED weight plane and `.` a
// `matmul_tiles`. `S_tau[t][u] = 1 iff u = t-d and t >= d`; `C_tau[t][p] = 1 iff p = t+tau
// and t < d`. `C_{T-1}` is the zero tile and is kept anyway, because a uniform `2*T` products
// puts CB_PROD on a clean indexed-read grid and 2*T is a power of two at every supported tap
// count while 2*T-1 is prime at T = 4.
//
// **THE WEIGHT IS ROW-REPLICATED, WHICH IS THE ONE STAGING CHANGE.** The decode family's
// tap-major planes carry `w[c][tau]` in ROW 0 and zeros below; here every row of the plane
// carries it, because the tap coefficient is constant across positions and the broadcast
// surface is BROADCAST_COL only — there is no row broadcast. Same tile count, same addresses,
// same transpose out of the oracle's row-major `[conv_dim, taps]`: only `valid_rows` changes
// from 1 to 32. The two weights are DIFFERENT DRAM BUFFERS in this round, so nothing the
// decode family reads moves.
//
// **THE 2*T BAND TILES RIDE AT THE END OF THE WEIGHT BUFFER**, at tile offset
// `taps * conv_tile_cols`, rather than in a sixth DRAM endpoint. `FT_TT_QUEUE_PARAM_WORDS`
// is 16 for every family and this one already spends three packed coordinate words; a
// seventh endpoint would need an eighth. They are 8 tiles of 4 KiB per layer against a 4 MiB
// weight, and they are the same 8 tiles in every layer.
//
// ===========================================================================
// THE ROLLING WINDOW IS THE CHUNK-BOUNDARY CARRY, AND ITS LAYOUT IS THE DECODE FAMILY'S
// ===========================================================================
// `kGdncStateConv` is `taps-1` TAP-MAJOR ROW-0 PLANES of `conv_tile_cols` tiles — BYTE FOR
// BYTE the layout of `kGdnStateConv`. That is deliberate and it is what makes a prompt that
// prefills in chunks and then decodes one token at a time a single sequence with no
// transcription step: the two families read and write the same bytes with the same meaning.
//
// The reader ASSEMBLES the `[C, ...]` window tile from those planes with `2*(taps-1)` 64-byte
// NoC reads a column — row 0 of an fp32 tile is two 64-byte pieces 1,024 B apart, which is
// exactly the DRAM read granule, so no gather and no word copy is needed. Rows `taps-1..31`
// of the assembled tile are ZEROED ONCE at residency boot and never written again, because
// they are multiplied by zero COLUMNS of every `C_tau` and `0 * NaN` is NaN.
//
// The writer rolls it: window plane p receives ROW `chunk_rows - (taps-1) + p` of the chunk's
// RAW input, two 64-byte writes. **THIS IS WHY `chunk_rows >= taps - 1` IS A REFUSAL RATHER
// THAN A CLAMP** — a chunk shorter than the window cannot replace it, it has to shift it, and
// a partial roll is a different writer. The host runs a tail of 1..taps-2 positions through
// the decode family, which is the same arithmetic on the same bytes.
//
// ===========================================================================
// WHAT THE HOST OWES: ROWS BEYOND `chunk_rows` MUST BE ZERO IN THE INPUT
// ===========================================================================
// Output rows `>= chunk_rows` are not read by anyone, but they are COMPUTED, and `S_tau`
// reaches input rows up to `chunk_rows - 1 + (taps-1)`. If the host leaves stale values
// there the arithmetic is still finite and still discarded — unless a stale row holds an inf
// or a NaN, which propagates through the SiLU into rows nobody inspects and then into the
// next chunk through nothing at all. A zeroed tail is `SiLU(0) = 0` and cannot surprise
// anyone. `pn_tt_activation_upload_rows()` into a zeroed strip already does this.
//
// ===========================================================================
// PHASE DELTA -- THE CHUNKWISE DELTA RULE AS A UT TRANSFORM
// ===========================================================================
// The oracle is `qwen35moe_compute_recur_heads()` in Pantheon's qwen35moe_ssm.c. Per v-head
// the carried state is stored TRANSPOSED, `N[j][i] == M[i][j]`, and one position does
//
//     N      *= d_t                                   (scalar decay of the whole state)
//     pred_j  = sum_i N[j][i] k_t[i]
//     u_t[j]  = (v_t[j] - pred_j) * b_t
//     N[j][i]+= k_t[i] * u_t[j]                       (rank-1 write)
//     y_t[j]  = scale * sum_i N[j][i] q_t[i]          (READ AFTER WRITE -- see below)
//
// Unrolled over a chunk of C positions with `P_t = prod_{r<=t} d_r` and
// `D[t][s] = P_t / P_s = prod_{r=s+1..t} d_r` (t >= s, and every entry is in (0,1]):
//
//     U = (I + diag(b) . StrictLower(D o K K^T))^-1  diag(b) (V - diag(P) K N_0^T)
//     Y = scale * ( diag(P) Q N_0^T + (D o Q K^T) U )
//     N_C = P_{C-1} N_0 + U^T diag(D[C-1][.]) K
//
// VERIFIED AGAINST THE ORACLE OFF-HARDWARE at 8.8e-16 max-norm relative in fp64, over four
// decay regimes including `d ~ 0.01` where a cumulative-product RECIPROCAL would overflow.
// `tests/test_gdnc_ops.c` stage 6 is that check and it runs on every build.
//
// (e) READS THE UPDATED STATE. `Y`'s second term is `LowerInclusive(Q K^T)`, diagonal
// INCLUDED. Excluding it is the "(e) before (d)" recurrence the decode family measured: it
// looks plausible for exactly one token and is wrong from the second.
//
// ---------------------------------------------------------------------------
// THE [C,C] SOLVE IS NEWTON, AND IT IS EXACT
// ---------------------------------------------------------------------------
// `A = I + A_off` with `A_off` STRICTLY LOWER, so `A_off` is nilpotent of index C. Started at
// `X_0 = 2I - A = I - A_off`, the error `E_k = I - A X_k` satisfies `E_0 = A_off^2` and
// `E_{k+1} = E_k^2`, i.e. `E_k = A_off^(2^(k+1))`. At C = 32 that is ZERO at k = 4 --
// **FOUR doublings, algebraically exact**, eight matmuls and four subtracts. The model
// reaches fp64 round-off at three and the fourth is kept because three is only an accident of
// the data: `A_off^16` is small, not zero. `newton = 0` is a red arm and scores 2.2e-02.
//
// ---------------------------------------------------------------------------
// THE DECAY-RATIO MATRIX IS BUILT ON THE HOST, AND THAT WAS MEASURED, NOT ASSUMED
// ---------------------------------------------------------------------------
// `D` arrives as ONE fp32 tile per v-head per chunk (`kGdncDeltaConstDecay + head`), built by
// `ft_tt_op_gdnc_build_decay()` as a PRODUCT RECURRENCE `D[t][s] = D[t][s+1] * d_{s+1}` --
// every entry in (0,1], no cumulative-product reciprocal formed anywhere, and no dependence
// on a log.
//
// THE ALTERNATIVE WAS PRICED AND REJECTED. `D` can be built ON CARD from 32 floats a head as
// `D = sum_{m<C} Z^m` with `Z` the decay subdiagonal, by doubling
// (`S_{k+1} = (I + P_k) S_k`, `P_{k+1} = P_k^2`) -- ~16 tile ops and it takes the operand off
// the link. Modelled through the same Tf32 unpack it costs the WHOLE precision advantage of
// going chunkwise: y goes 8.6e-03 -> 1.7e-02, which is exactly where 32 sequential decode
// steps already are (2.1e-02). Five truncating multiplies on the longest chain against one.
// So the 4 KiB a head rides on the link, and what it buys is measured.
//
// AND A `log g` CROSSING IS WORSE THAN EITHER. `D[t][s] = exp(log P_t - log P_s)` is safe
// against OVERFLOW and not against the UNPACK: the operands of that subtract are cumulative
// LOGS, which reach -50 at Ornith's decays, and a Tf32 truncation of -50 is an absolute error
// of 0.024 -- a 2.4% error in D before the SFPU exp is reached. The linear domain is where
// the ratios are small numbers and the truncation is relative.
//
// ---------------------------------------------------------------------------
// A CHUNK IS THE SAME STRIP HERE TOO, WHICH IS WHY NO DRAM LAYOUT MOVES
// ---------------------------------------------------------------------------
// `kGdncActivationIn` is the DECODE family's `kGdnActivationIn` byte for byte: q | k | v as
// ONE tilized strip, head-major, q and k per K-HEAD and NOT GQA-repeated. The decode family
// fills ROW 0 and leaves 1..31 zero; this one fills rows 0..chunk_rows-1 with positions. Same
// tiles, same addresses, same `ftg_delta_*` arithmetic. `kGdncActivationOut` is y in the same
// geometry, and `kGdncStateConv` carries the `[v_heads, S, S]` recurrent state in the decode
// family's own transposed, head-major, tile-row-major order.
//
// **SO A PROMPT THAT PREFILLS IN CHUNKS AND THEN DECODES ONE TOKEN AT A TIME IS ONE SEQUENCE
// WITH NO TRANSCRIPTION**, on the recurrent state exactly as it already is on the conv window.
//
// THE ONE THING THE WRITER DOES NOT DO THAT THE DECODE FAMILY'S DOES IS TRANSPOSE. Decode's
// (e) produces y as a COLUMN and the writer scatters it into a row; here y is a whole
// `[C, S]` tile whose rows are the positions, so it is a whole-tile write.
//
// ---------------------------------------------------------------------------
// ROWS AT AND ABOVE `chunk_rows` COST NOTHING AND MUST STILL BE ZERO
// ---------------------------------------------------------------------------
// A ragged chunk is handled ENTIRELY IN THE HOST'S GATE ARITHMETIC and needs no kernel branch.
// With q, k and v zero above the chunk: `K K^T` and `Q K^T` are zero in those rows and
// columns, so `A` stays unit lower triangular with `e_t` rows; `V - diag(P) K N_0^T` is zero
// there, so `U` is zero there; and `U^T diag(F) K` therefore takes no contribution from them.
// What the host owes is that `P`, `F` and `G` are computed over the LIVE rows only -- `G` is
// `P_{chunk_rows-1}`, not `P_31`. `ft_tt_op_gdnc_build_gate()` is the one implementation.
//
// This is also why PHASE DELTA has NO `chunk_rows >= taps-1` floor: that refusal belongs to
// the window ROLL, which is PHASE CONV's.
//
// ===========================================================================
// THE THREE RULES THIS INHERITS, UNCHANGED
// ===========================================================================
// A CB closes on the CONSUMER'S INDEXED READ (`pages % n == 0` per phase plus a rebase at
// every descriptor); a buffer has ONE PRODUCER for the life of the residency; and the fp32
// unpack is a WHOLE-PROGRAM decision, so every CB in `gdnc_float32.cbtable` is Float32 and
// `dst_accum_mode` is 1. All three are stated at length in resident_gdn_protocol.h and every
// one of them was a measured wrong answer there.
//
#ifndef FOUNDRY_RESIDENT_GDNC_PROTOCOL_H
#define FOUNDRY_RESIDENT_GDNC_PROTOCOL_H

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

/* ---------------------------------------------------------------------------
 * The sixteen words. Mirrors ft_tt_op_gdnc.h -- change both or neither.
 * --------------------------------------------------------------------------- */

/** byte 0 phase, byte 1 chunk rows (live positions, 1..32; PHASE CONV floors it at `taps-1`),
 *  byte 2 conv taps, byte 3 PHASE DELTA's state tiles (S/32). */
enum { kGdncShape = 0 };

/** PHASE CONV: tile columns of the conv_dim stream, conv_dim/32. A full word -- 8192/32 = 256
 *  does not fit in a byte, and a shape that silently truncates is the failure this family can
 *  least afford.
 *
 *  PHASE DELTA: v-heads in the low half-word, k-heads in the high. The two readings share a
 *  word because `FT_TT_QUEUE_PARAM_WORDS` is 16 for EVERY family and a seventeenth moves the
 *  shared slot stride; they cannot collide because the phase byte selects which one is read,
 *  and `ftgc_decode()` fills BOTH so a kernel that reads the wrong one reads a number that is
 *  obviously not a head count rather than a plausible one. */
enum { kGdncConvTileCols = 1 };

/** The SLICE this descriptor owns: first in the low half-word, COUNT in the high half-word.
 *  A count of ZERO means the whole stream, which is what every unsliced encoder emits and what
 *  makes an unsliced descriptor byte-identical to the one that shipped.
 *
 *  PHASE CONV slices TILE COLUMNS: the convolution divides EXACTLY by channel -- no column of
 *  it reads another column's operands, and the window roll is per column too. PHASE DELTA
 *  slices V-HEADS, for the same reason and with the same measurement behind it: the decode
 *  family's DELTA is `59.8 us + N x 103.4` over six points, i.e. embarrassingly parallel in
 *  the head. Half-words rather than bytes because `conv_tile_cols` is 256 at Ornith. */
enum { kGdncColSlice = 2 };

/** Activation bank: x in the low half-word, y in the high.
 *
 *  ONE WORD FOR TWO BUFFERS, so the input and the output are ONE DRAM endpoint -- the same
 *  contract `kGdnActivationXy` carries, and the same one that produced the decode family's
 *  first wrong answer when a stager put the pair in different banks: the writer lands the
 *  answer at the OUTPUT's offset INSIDE THE INPUT's bank, the descriptor retires clean, and
 *  the output buffer is never written at all. `ft_tt_op_gdnc_parameters()` refuses it. */
enum { kGdncActivationXy = 3 };
/** PHASE CONV: the chunk's conv input, a tilized `[chunk_rows, conv_dim]` strip of
 *  `conv_tile_cols` tiles. PHASE DELTA: the staged `q | k | v` strip, head-major, q and k per
 *  K-HEAD -- the decode family's `kGdnActivationIn` byte for byte.
 *  ROWS `chunk_rows..31` MUST BE ZERO in both -- see the header. */
enum { kGdncActivationIn = 4 };
/** PHASE CONV: conv_out, the same shape; rows `chunk_rows..31` are `SiLU(0) = 0`.
 *  PHASE DELTA: y, a `[chunk_rows, v_heads*S]` strip, head-major. */
enum { kGdncActivationOut = 5 };

/** Per-layer constants bank. */
enum { kGdncConstantXy = 6 };
/** PHASE CONV: the depthwise conv1d weight as `taps` ROW-REPLICATED TAP-MAJOR PLANES of
 *  `conv_tile_cols` tiles, FOLLOWED BY the `2*taps` band tiles.
 *  PHASE DELTA: `kGdncDeltaConstDecay` program-constant tiles, FOLLOWED BY one decay-ratio
 *  tile per v-head. See the layout sections. */
enum { kGdncConstantWeight = 7 };

/** Carried-state bank. */
enum { kGdncStateXy = 8 };
/** PHASE CONV: the rolling conv window, `taps-1` TAP-MAJOR ROW-0 PLANES of `conv_tile_cols`
 *  tiles -- byte for byte the decode family's `kGdnStateConv`.
 *  PHASE DELTA: the recurrent state, `[v_heads, S, S]` fp32 TRANSPOSED per head and head-major
 *  -- byte for byte the decode family's `kGdnStateRecurrent`.
 *  Read AND written in both. */
enum { kGdncStateConv = 9 };

/** Bytes per tile of every operand. Float32 tiles are 4096 B. */
enum { kGdncTileBytes = 10 };

/**
 * VARIANT BITS. Byte 0 is THE HOST'S OWN CONV BLOCK DEPTH; the rest is reserved and written 0.
 *
 * WHY THE DEPTH IS ON THE WIRE AT ALL. Every device check this family makes asks whether the
 * PLACED BUFFERS CAN SERVE the image's depth, and that question has a SILENT DIRECTION: a host
 * that placed for a DEEPER block than the kernels were built at passes every one of them — a
 * 40-page CB_VECTOR is a perfectly legal home for depth-1 blocks — so the arm runs at the
 * KERNELS' depth inside the HOST'S placement, produces the RIGHT ANSWER and THE CONTROL'S
 * TIMING, reported as the arm's. That corrupts a MEASUREMENT rather than an answer, which is
 * the one failure a sweep cannot see from its own output. The decode family recorded the
 * identical hazard at `kGdnByteConvDepth` and closed it the same way.
 *
 * STRICT EQUALITY, WITH NO "0 MEANS UNSET" ESCAPE. This is deliberately the OPPOSITE
 * convention from `kGdncColSlice`'s `count == 0 means all`: a slice has a meaningful default
 * and a depth does not, so an encoder that forgets to declare must REFUSE rather than pass.
 * PHASE DELTA never reads the depth, so a DELTA descriptor still writes the whole word zero
 * and its shipped encoding is unchanged.
 */
enum { kGdncFlags = 11 };
enum { kGdncByteConvDepth = 0 };
/** L11b, PHASE DELTA only: the gate block is COMPACT (ftgc_compact.h) and the reader builds
 *  the decay-ratio tiles and gate runs itself. */
enum { kGdncDeltaFlagCompact = 0x100 };
/** PM 8, either phase: the reader waits for the previous descriptor's PUBLISHED completion
 *  (its window / state writes landed) before reading — queued descriptors that carry state
 *  through DRAM need no host await between them. */
enum { kGdncFlagSerialize = 0x200 };
/** PHASE DELTA only, implies COMPACT: there is no gate block. kGdncDeltaGate is the layer's
 *  TABLE BLOCK (ftgc_table.h) in the constant bank, kGdncDeltaBand the band of the alpha/beta
 *  strips, and the reader builds each head's compact run from the strips through the host's
 *  lookup tables — the host's gate bits, no alpha/beta read back. */
enum { kGdncDeltaFlagTable = 0x400 };

/** TOTAL tiles this descriptor writes through CB_OUT -- all the writer reads.
 *  PHASE CONV: two a column, the conv_out tile then the RAW input tile the roll comes from.
 *  PHASE DELTA: `state_tiles * (state_tiles + 1)` a head, the y tiles then the state tiles. */
enum { kGdncTileCount = 12 };

/** PHASE DELTA: the GATE BLOCK's address, INSIDE THE CONSTANT BANK (`kGdncConstantXy`).
 *  A flat fp32 array, `kGdncGateStride` floats per v-head -- the one operand of this family
 *  that is not tiles, exactly as `kGdnConstantGate` is for the decode family. Written 0 and
 *  never read in PHASE CONV. */
enum { kGdncDeltaGate = 13 };

/**
 * PHASE DELTA: THE FIRST ROW OF THE 32-ROW STRIP THIS DESCRIPTOR OWNS. Written 0 and refused
 * nonzero in PHASE CONV.
 *
 * ===========================================================================
 * WHY A SUB-CHUNK IN PLACE IS A WRITER FIELD AND NOT A NEW KERNEL
 * ===========================================================================
 * The chunk length C is the whole of PHASE DELTA's numerical variance. `A = I + A_off` is well
 * conditioned (`max|A^-1| = 1.0`, `cond ~ 30`) but `A_off`'s powers are NOT monotone: at
 * C = 32 `max|A_off^8| = 3.6e+04` and the Newton iterates reach 1.7e+06 before collapsing to
 * ~1. The answer is a cancellation of 1e5-sized intermediates and a 4.9e-4 Tf32 unpack
 * truncation leaves a residue it cannot remove. At C = 16 the intermediates never get large
 * enough for that to bite -- measured 2026-08-24: C = 16 passes all six gate criteria with the
 * best figures of the whole ladder, C = 32 fails worst-document.
 *
 * A SHORTER C DOES NOT NEED A SHORTER PASS. Everything else in the layer batches at 32, so the
 * rung is `ceil(rows/C)` DELTA descriptors over the SAME 32-row strip, each owning rows
 * `[first_row, first_row + chunk_rows)` IN PLACE. That needs exactly two things:
 *
 *   1. THE HOST KILLS THE ROWS THIS DESCRIPTOR DOES NOT OWN, in the gate block and the decay
 *      tile it already rebuilds every chunk. `beta_t = 0` outside the sub-chunk makes row t of
 *      `A` exactly `e_t`, hence row t of `A^-1` exactly `e_t`, hence `U` row t exactly zero --
 *      so the carried state takes no contribution from it and `Y`'s `(D o QK^T) U` term takes
 *      none either. `D[t][s] = 0` for `s < first_row` then confines `A_off` to the sub-chunk's
 *      own `[C,C]` block, WHICH IS THE ENTIRE POINT: without it the Newton doubling still sees
 *      a 32-wide nilpotent and the intermediates blow up exactly as they do at C = 32. And
 *      `P_t = 0` on the dead rows makes `Y`'s `diag(P) Q N_0^T` term zero there. All three are
 *      `ft_tt_op_gdnc_build_gate_at()` / `_build_decay_at()` -- host arithmetic, on operands
 *      that were rebuilt per chunk anyway.
 *   2. THE WRITER WRITES ONLY THE ROWS IT OWNS, or the second sub-chunk's whole-tile `y` write
 *      lands zeros over the first one's answer. That is this word.
 *
 * A 32-ROW TILE'S TOP 16 ROWS ARE ITS FIRST TWO FACES, i.e. THE FIRST HALF OF THE TILE'S
 * BYTES, so a row range starting on a face boundary is a CONTIGUOUS SUFFIX of the tile and
 * needs no gather -- `ftgc_y_write_offset()`. That is why a `first_row` which is not a
 * multiple of `kGdncFaceRows` is REFUSED: the alignment is not a convenience, it is what keeps
 * this to one offset and one length on a write that was already being issued.
 *
 * PHASE CONV NEEDS NONE OF IT AND STILL RUNS ONCE PER PASS. The convolution is a fixed `taps`
 * FIR, not a length-sensitive recurrence: output row t reads input rows `t-(taps-1)..t`, which
 * for `t >= taps-1` are inside the strip whatever the chunking, and the band already selects
 * them (`S_tau[t][u] = 1 iff u = t-d and t >= d`). A sub-chunked CONV and a whole-pass CONV
 * accumulate the SAME `2*taps` terms in the SAME tap order, differing only in which of each
 * pair contributes the value and which contributes an exact zero.
 */
enum { kGdncFirstRow = 14 };
/** PHASE DELTA under kGdncDeltaFlagTable: the band of the alpha/beta strips the sub-chunk
 *  reads its gates from. Written 0 otherwise. */
enum { kGdncDeltaBand = 15 };

enum { kGdncWords = 16 };

/** Where a byte lives in a packed word. */
enum { kGdncBytePhase = 0 };
enum { kGdncByteChunkRows = 1 };
enum { kGdncByteTaps = 2 };
enum { kGdncByteStateTiles = 3 };

/**
 * THE OPCODE, AND WHY IT IS DECLARED HERE RATHER THAN IN THE SHARED REGISTRY.
 *
 * Every other family's opcode lives in `resident_queue_protocol.h` beside `kOpGdn`, and that
 * is where a registry belongs. It cannot live there: **that header is a CONTENT-PROVENANCE
 * INPUT TO EVERY EMBEDDED IMAGE.** Adding one line to it makes all 23 programs STALE by
 * content, `emit_tensix_program_header.py` refuses to regenerate the blob until every one of
 * them is rebuilt, and `make check-program-provenance` fails with 74 findings. Measured on
 * a second card host 2026-08-21, on the first attempt to add this family: the cost of registering an
 * opcode centrally is a 23-program rebuild, and the rebuild it forces is of the very binaries
 * -- `gdn` among them -- whose byte-identity is what protects the shipped decode gate.
 *
 * WHAT KEEPS THE REGISTRY HONEST INSTEAD is stronger than the single pin it replaces:
 * `tests/test_gdnc_ops.c` PARSES `resident_queue_protocol.h` for every `kOp*` constant it
 * declares and asserts that NONE of them is this number, and pins this one against the host's
 * `FT_TT_QUEUE_OP_GDN_CHUNK` in `ft_tt_queue.h` -- which is not a kernel input and so may
 * carry the authoritative enumeration at no provenance cost.
 *
 * 0x13 was the next free opcode after `kOpGdn = 0x12`.
 */
enum { kOpGdnChunk = 0x13 };

/** Which half of the block this descriptor runs. 0 is deliberately not a phase, so a zeroed
 *  parameter block is an invalid program rather than a plausible one.
 *
 *  `kGdncPhaseDelta` is DECLARED AND NOT IMPLEMENTED. It is here so the shape word's phase
 *  byte has its final meaning from the first build -- the chunkwise delta rule is the other
 *  half of this family and it lands in the same program. A descriptor naming it is refused
 *  by `ftgc_shape_is_supported()` today, loudly, which is the correct answer for an
 *  unimplemented phase and the wrong one for an undeclared number. */
enum { kGdncPhaseConv = 1 };  /**< the banded conv1d + SiLU over C positions, window rolled */
enum { kGdncPhaseDelta = 2 }; /**< the chunkwise parallel delta rule, as a UT transform */

/* ---------------------------------------------------------------------------
 * PHASE DELTA's operand shapes. Every number here is used by the reader, the compute thread,
 * the writer AND the host stager, which is why it is stated once.
 * --------------------------------------------------------------------------- */

/** The PROGRAM-CONSTANT tiles at the head of `kGdncConstantWeight` in PHASE DELTA. They are
 *  identical in every layer and every chunk; only the per-head decay tiles behind them move.
 *
 *  FOUR of them and not three: the compute thread indexes this buffer four wide, so four is
 *  what puts CB_BAND on a clean indexed-read grid that ALSO divides the `2*taps` PHASE CONV
 *  needs at every supported tap count. */
enum { kGdncDeltaConstIdentity = 0 };  /**< I -- the Newton start, and the transpose operand */
enum { kGdncDeltaConstTwoI = 1 };      /**< 2I -- the Newton residual */
enum { kGdncDeltaConstStrict = 2 };    /**< 1 strictly below the diagonal, 0 on and above it */
enum { kGdncDeltaConstScale = 3 };     /**< a COLUMN of `1/sqrt(S)`; the host owns the rsqrt */
enum { kGdncDeltaConstTiles = 4 };
/** First per-head decay-ratio tile inside the same buffer. */
enum { kGdncDeltaConstDecay = 4 };

/** Floats per v-head in the gate block, and what each run of `kGdncTileRows` carries.
 *
 *  A FIXED STRIDE OF FOUR RUNS, so the reader's per-head read is one NoC transaction at a
 *  64-byte-aligned offset and its scatter loop does not branch. `P` and `PS` are separate runs
 *  rather than one plus a multiply because `scale` multiplies the `Q` path and not the `K`
 *  path, and a kernel that recomputed either would be a second derivation of a factor that
 *  multiplies every output of a recurrence. */
enum { kGdncGateP = 0 };  /**< P_t  = prod_{r<=t} decay_r, over the LIVE rows only */
enum { kGdncGatePS = 1 }; /**< scale * P_t -- the Q path's, so `1/sqrt(S)` is derived ONCE */
enum { kGdncGateB = 2 };  /**< beta_t, the sigmoid's answer; 0 at and above chunk_rows */
enum { kGdncGateF = 3 };  /**< F_t  = P_{chunk_rows-1} / P_t, every entry in (0,1] */
enum { kGdncGateG = 4 };  /**< P_{chunk_rows-1}, REPLICATED down the run */
enum { kGdncGateRuns = 5 };
enum { kGdncGateStride = 5 * 32 }; /**< kGdncGateRuns * kGdncTileRows floats; 640 B, a
                                    *   multiple of the 64-byte DRAM granule by construction */

/** Circular-buffer INDEX SPANS a PHASE DELTA descriptor uses. Each is what the compute thread
 *  names absolutely off a buffer's front, so each is what `pages % span == 0` is checked
 *  against -- on the reader, which is the only processor that can see a page count. */
enum { kGdncDeltaVectorSpan = 5 };  /**< CB_VECTOR: the five gate columns of one head */
enum { kGdncDeltaCcSpan = 24 };     /**< CB_CC: the [C,C] working tiles, listed in the compute */
enum { kGdncDeltaWorkPerRow = 9 };  /**< CB_WORK: nine [C,S] groups of `state_tiles` tiles */

/** Rows of one face, and the row count of a tile. Fixed by gdnc_float32.cbtable
 *  (face_r 16, num_faces 4). */
enum { kGdncFaceRows = 16 };
enum { kGdncTileRows = 32 };

/** The row a ROW-0 STRIP's data occupies inside every tile of it -- the window planes, and
 *  the decode family's whole geometry. Named rather than spelled 0 because both the writer's
 *  destination row and the reader's source row are arguments. */
enum { kGdncStripRow = 0 };

/** One descriptor's shape, decoded once at the top of the op.
 *
 *  The CONV fields and the DELTA fields are decoded UNCONDITIONALLY out of the words they
 *  share, so a kernel that reads the wrong one for its phase reads a number that is obviously
 *  not what it wants (a head count of 256, a tile-column count of 16) rather than a plausible
 *  one. The refusal below is what actually keeps them apart. */
typedef struct ftgc_shape {
    uint32_t phase;
    uint32_t chunk_rows;     /**< live positions in this chunk, 1..32 (CONV floors at taps-1) */
    uint32_t conv_taps;
    uint32_t conv_tile_cols; /**< PHASE CONV: conv_dim / 32 */
    uint32_t first_col;      /**< PHASE CONV: first tile column THIS descriptor owns */
    uint32_t col_count;      /**< PHASE CONV: tile columns THIS descriptor owns */
    uint32_t state_tiles;    /**< PHASE DELTA: S / 32 */
    uint32_t v_heads;        /**< PHASE DELTA */
    uint32_t k_heads;        /**< PHASE DELTA; the GQA repeat is `kh = vh % k_heads` */
    uint32_t head_first;     /**< PHASE DELTA: first v-head THIS descriptor owns */
    uint32_t head_count;     /**< PHASE DELTA: v-heads THIS descriptor owns */
    uint32_t first_row;      /**< PHASE DELTA: first strip ROW this descriptor owns; 0 in CONV */
    uint32_t tile_bytes;
    uint32_t tile_count;
    /** PHASE CONV: the CONV block depth THE HOST placed for, off `kGdncByteConvDepth`.
     *  Decoded unconditionally so a DELTA descriptor's zero is visible rather than absent. */
    uint32_t declared_conv_depth;
} ftgc_shape;

FTQ_INLINE uint32_t ftgc_byte(uint32_t word, uint32_t index)
{
    return (word >> (index * 8u)) & 0xFFu;
}

FTQ_INLINE uint32_t ftgc_xy_x(uint32_t word)
{
    return word & 0xFFFFu;
}

FTQ_INLINE uint32_t ftgc_xy_y(uint32_t word)
{
    return (word >> 16) & 0xFFFFu;
}

FTQ_INLINE ftgc_shape ftgc_decode(volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t shape_word = parameters[kGdncShape];
    const uint32_t slice_word = parameters[kGdncColSlice];
    ftgc_shape shape;

    shape.phase = ftgc_byte(shape_word, kGdncBytePhase);
    shape.chunk_rows = ftgc_byte(shape_word, kGdncByteChunkRows);
    shape.conv_taps = ftgc_byte(shape_word, kGdncByteTaps);
    shape.state_tiles = ftgc_byte(shape_word, kGdncByteStateTiles);
    shape.conv_tile_cols = parameters[kGdncConvTileCols];
    shape.v_heads = parameters[kGdncConvTileCols] & 0xFFFFu;
    shape.k_heads = (parameters[kGdncConvTileCols] >> 16) & 0xFFFFu;
    shape.first_col = slice_word & 0xFFFFu;
    shape.col_count = (slice_word >> 16) & 0xFFFFu;
    shape.head_first = shape.first_col;
    shape.head_count = shape.col_count;
    shape.first_row = parameters[kGdncFirstRow];
    shape.tile_bytes = parameters[kGdncTileBytes];
    shape.tile_count = parameters[kGdncTileCount];
    shape.declared_conv_depth = ftgc_byte(parameters[kGdncFlags], kGdncByteConvDepth);
    /* A COUNT OF ZERO MEANS THE WHOLE STREAM, and it is the compatibility hinge rather than
     * a convenience: an encoder that does not know about the slice leaves the word zero and
     * means "all of it from element 0". */
    if (shape.col_count == 0u) {
        shape.first_col = 0u;
        shape.col_count = shape.conv_tile_cols;
    }
    if (shape.head_count == 0u) {
        shape.head_first = 0u;
        shape.head_count = shape.v_heads;
    }
    return shape;
}

/**
 * Does the depth the HOST declared match the depth THIS IMAGE was compiled with?
 *
 * ONE implementation, called by the reader AND the compute thread, so neither can be left
 * waiting on pages the other decided not to send. Strict equality — see `kGdncFlags`.
 */
FTQ_INLINE bool ftgc_conv_depth_agrees(uint32_t declared, uint32_t built)
{
    return declared == built;
}

/** Band tiles this shape uses: `S_tau` and `C_tau` for every tap. */
FTQ_INLINE uint32_t ftgc_band_tiles(const ftgc_shape *shape)
{
    return 2u * shape->conv_taps;
}

/** First tile of the band block inside the constant buffer: it rides at the END of the
 *  `taps` weight planes. ONE implementation, shared by the reader and by the host stager,
 *  because the two are the ends of a contract nothing on the device can check. */
FTQ_INLINE uint32_t ftgc_band_base_tile(const ftgc_shape *shape)
{
    return shape->conv_taps * shape->conv_tile_cols;
}

/** Tile `column` of tap plane `tap` in the conv WEIGHT (row-replicated) or in the rolling
 *  WINDOW (row 0). The two operands are laid out identically, which is what makes the roll
 *  layout-preserving. */
FTQ_INLINE uint32_t ftgc_plane_tile(const ftgc_shape *shape, uint32_t tap, uint32_t column)
{
    return tap * shape->conv_tile_cols + column;
}

/** Tiles the constant buffer occupies -- what a stager allocates: the weight planes and the
 *  band. */
FTQ_INLINE uint32_t ftgc_constant_tiles(const ftgc_shape *shape)
{
    return ftgc_band_base_tile(shape) + ftgc_band_tiles(shape);
}

/* ---------------------------------------------------------------------------
 * PHASE DELTA's DRAM arithmetic. Every one of these is the decode family's
 * `ftg_delta_*` / `ftg_state_tile` / `ftg_y_tile` expression, respelled here because the two
 * families do not include each other's headers -- and `tests/test_gdnc_ops.c` asserts the two
 * agree over the whole Ornith shape rather than trusting the respelling.
 * --------------------------------------------------------------------------- */

/** Which of the three staged streams, in the order they sit at `kGdncActivationIn`. */
enum { kGdncStreamQ = 0 };
enum { kGdncStreamK = 1 };
enum { kGdncStreamV = 2 };

/** First tile of stream `stream`'s block. q and k are per-K-HEAD and v is per-V-HEAD, so the
 *  three blocks are NOT the same size. */
FTQ_INLINE uint32_t ftgc_delta_stream_base(const ftgc_shape *shape, uint32_t stream)
{
    const uint32_t k_block = shape->k_heads * shape->state_tiles;

    if (stream == (uint32_t)kGdncStreamV)
        return 2u * k_block;
    return stream * k_block;
}

/**
 * The k-head whose q/k tiles v-head `v_head` reads: the INTERLEAVED GQA repeat.
 *
 * `vh = r * n_k_heads + kh` in Pantheon's qwen35moe_ssm.c, so `kh = vh % n_k_heads`. A named
 * function rather than an inline modulo because the BLOCKED mapping (`vh / kv_groups`) is the
 * plausible wrong answer and has produced gibberish in this architecture before.
 */
FTQ_INLINE uint32_t ftgc_delta_k_head(const ftgc_shape *shape, uint32_t v_head)
{
    return v_head % shape->k_heads;
}

/** First tile of v-head `head`'s slice of stream `stream`, with the GQA repeat applied. */
FTQ_INLINE uint32_t ftgc_delta_head_tile(const ftgc_shape *shape, uint32_t stream,
                                         uint32_t v_head)
{
    const uint32_t head = (stream == (uint32_t)kGdncStreamV)
                              ? v_head
                              : ftgc_delta_k_head(shape, v_head);

    return ftgc_delta_stream_base(shape, stream) + head * shape->state_tiles;
}

/** Tiles the whole staged `q|k|v` input occupies -- what a stager allocates. */
FTQ_INLINE uint32_t ftgc_delta_input_tiles(const ftgc_shape *shape)
{
    return (2u * shape->k_heads + shape->v_heads) * shape->state_tiles;
}

/** Tiles the y output occupies: one row block of `v_heads * state_tiles` tiles. */
FTQ_INLINE uint32_t ftgc_delta_y_tiles(const ftgc_shape *shape)
{
    return shape->v_heads * shape->state_tiles;
}

/** Tile of the carried state holding rows [32*tile_row, +32) and columns [32*tile_col, +32) of
 *  v-head `head`. Head-major, then row-major inside the head -- the decode family's order. */
FTQ_INLINE uint32_t ftgc_state_tile(const ftgc_shape *shape, uint32_t head, uint32_t tile_row,
                                    uint32_t tile_col)
{
    return head * shape->state_tiles * shape->state_tiles + tile_row * shape->state_tiles +
           tile_col;
}

/** Output tile of y carrying v-head `head`'s elements [32*tile_row, +32) for every position of
 *  the chunk. Unlike the decode family's, this is a WHOLE tile: its rows are the positions. */
FTQ_INLINE uint32_t ftgc_y_tile(const ftgc_shape *shape, uint32_t head, uint32_t tile_row)
{
    return head * shape->state_tiles + tile_row;
}

/**
 * Byte offset INSIDE a `y` tile at which this descriptor's write starts, and the length that
 * follows it is `tile_bytes - offset`. Zero for a whole-tile write, which is what every
 * `first_row = 0` descriptor emits and what makes the unsliced case byte-identical.
 *
 * A tile is four `kGdncFaceRows x kGdncFaceRows` faces in the order (0,0) (0,1) (1,0) (1,1),
 * so rows `[0,16)` are faces 0 and 1 -- the FIRST HALF of the tile's bytes -- and rows
 * `[16,32)` are the second. A row range that starts on a face boundary is therefore a
 * contiguous suffix, and `ftgc_shape_is_supported()` refuses anything else.
 *
 * The write is a SUFFIX rather than an exact `[first_row, first_row+chunk_rows)` window on
 * purpose: rows above the sub-chunk are y-zero by construction (the host sets `P_t = 0` and
 * `D[t][.] = 0` there), so the only thing a longer write can land on them is the zero they
 * already hold, and one length is one fewer thing for a later descriptor to get wrong.
 */
FTQ_INLINE uint32_t ftgc_y_write_offset(const ftgc_shape *shape)
{
    return (shape->first_row / (uint32_t)kGdncFaceRows) * (shape->tile_bytes / 2u);
}

/** Tiles of the recurrent state one layer owns -- what a stager allocates. */
FTQ_INLINE uint32_t ftgc_delta_state_tiles(const ftgc_shape *shape)
{
    return shape->v_heads * shape->state_tiles * shape->state_tiles;
}

/** Tiles the PHASE DELTA constant buffer occupies: the four program constants and one
 *  decay-ratio tile per v-head. */
FTQ_INLINE uint32_t ftgc_delta_constant_tiles(const ftgc_shape *shape)
{
    return (uint32_t)kGdncDeltaConstTiles + shape->v_heads;
}

/** The decay-ratio tile of v-head `head`. */
FTQ_INLINE uint32_t ftgc_delta_decay_tile(uint32_t head)
{
    return (uint32_t)kGdncDeltaConstDecay + head;
}

/** Float offset of v-head `head`'s run `run` inside the flat gate block. */
FTQ_INLINE uint32_t ftgc_gate_offset(uint32_t head, uint32_t run)
{
    return head * (uint32_t)kGdncGateStride + run * (uint32_t)kGdncTileRows;
}

/** Tiles the writer drains for this descriptor.
 *
 *  PHASE CONV: per column one conv_out tile and then the RAW input tile the roll is taken
 *  from. PHASE DELTA: per head the `state_tiles` y tiles and then the `state_tiles^2` updated
 *  state tiles -- THAT ORDER IS THE WIRE CONTRACT the writer's split is defined over. Y goes
 *  FIRST because it is what the compute thread finishes first; the decode family put the state
 *  first for the opposite reason and neither order is more correct than the other, so the one
 *  that costs no staging wins. */
FTQ_INLINE uint32_t ftgc_out_tiles(const ftgc_shape *shape)
{
    if (shape->phase == (uint32_t)kGdncPhaseDelta)
        return shape->head_count * shape->state_tiles * (shape->state_tiles + 1u);
    return 2u * shape->col_count;
}

/**
 * The chunk row whose values become window plane `plane` after the roll.
 *
 * Plane 0 is the OLDEST, so after a chunk of `chunk_rows` positions the window holds the last
 * `taps-1` of them in order. `chunk_rows >= taps-1` is a REFUSAL (see the header), which is
 * exactly what makes this expression non-negative.
 */
FTQ_INLINE uint32_t ftgc_roll_source_row(const ftgc_shape *shape, uint32_t plane)
{
    return shape->chunk_rows - (shape->conv_taps - 1u) + plane;
}

/**
 * Is this a shape this binary can execute?
 *
 * Checked on the DEVICE and not only on the host, for the reason the reduce family's
 * equivalent gives: the failure a bad shape produces is not a wrong answer but a blocked
 * triad -- a writer waiting on an output tile nobody will produce, until its bound expires.
 * A refusal names it in one word instead.
 */
FTQ_INLINE bool ftgc_shape_is_supported(const ftgc_shape *shape)
{
    if (shape->chunk_rows == 0u || shape->chunk_rows > (uint32_t)kGdncTileRows)
        return false;
    if (shape->tile_count != ftgc_out_tiles(shape))
        return false;

    if (shape->phase == (uint32_t)kGdncPhaseConv) {
        /* THE ROLL SOURCE ROW IS `chunk_rows - (taps-1) + plane` AND IT KNOWS NOTHING ABOUT
         * AN OFFSET, so a CONV descriptor naming one would roll the wrong three rows into the
         * window and be wrong from the NEXT chunk onward -- invisible in chunk 0, which is the
         * failure shape this family has already paid for twice. CONV does not need the field
         * (see `kGdncFirstRow`), so it is a refusal rather than a meaning. */
        if (shape->first_row != 0u)
            return false;
        if (shape->conv_taps < 2u || shape->conv_taps > 8u)
            return false;
        /* A chunk shorter than the window cannot REPLACE it, and a partial roll is a
         * different writer. The host runs a 1..taps-2 tail through the decode family
         * instead. THIS FLOOR IS THE ROLL'S, so PHASE DELTA does not carry it. */
        if (shape->chunk_rows < shape->conv_taps - 1u)
            return false;
        if (shape->conv_tile_cols == 0u)
            return false;
        if (shape->col_count == 0u ||
            shape->first_col + shape->col_count > shape->conv_tile_cols)
            return false;
        return true;
    }

    if (shape->phase == (uint32_t)kGdncPhaseDelta) {
        /* `state_tiles` bounds the [S,S] state a head holds in L1 at `state_tiles^2` pages,
         * and it squares -- 8 is 64 pages of 4 KiB, which is where the arena stops. */
        if (shape->state_tiles == 0u || shape->state_tiles > 8u)
            return false;
        if (shape->v_heads == 0u || shape->k_heads == 0u)
            return false;
        /* The GQA divisibility is checked ON THE DEVICE and not only on the host because
         * getting the repeat backwards is a KNOWN bug in this architecture. */
        if (shape->v_heads % shape->k_heads != 0u)
            return false;
        /* A slice that ran off the end would read a state region belonging to no head and
         * scatter y into a tile the layout never allocated -- both clean retires. */
        if (shape->head_count == 0u || shape->head_first + shape->head_count > shape->v_heads)
            return false;
        /* THE SUB-CHUNK WINDOW. A `first_row` off a face boundary would make the y write a
         * gather instead of a contiguous suffix, and one running past the tile would write
         * a negative length -- both clean retires that corrupt a neighbouring head's y. */
        if (shape->first_row % (uint32_t)kGdncFaceRows != 0u)
            return false;
        if (shape->first_row + shape->chunk_rows > (uint32_t)kGdncTileRows)
            return false;
        return true;
    }
    return false;
}

#endif  // FOUNDRY_RESIDENT_GDNC_PROTOCOL_H
