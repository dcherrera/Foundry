/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_gdn.h
 * @brief Host driver for the GATED-DELTA-NET block — Qwen3.5's linear-layer recurrence.
 *   Places the triad's circular buffers and encodes the parameter block. The two phases
 *   (conv1d+SiLU, then the delta rule) share one binary and one parameter layout.
 *
 * The device half is `kernels/resident_gdn_protocol.h`; the two must agree word for word and
 * `tests/test_gdn_ops.c` pins the pair off-hardware by parsing that header.
 *
 * THE DRAM LAYOUTS ARE NOT HERE, ON PURPOSE. Where q|k|v sit at `kGdnActivationIn`, that the
 * conv weight is TAP-MAJOR PLANES rather than the oracle's row-major `[conv_dim, taps]`, that
 * the gate block is a flat fp32 array and not tiles, and how y and the carried state are
 * tilized, are all stated ONCE — in `resident_gdn_protocol.h`'s layout section, beside the
 * `ftg_*` accessors both dataflow kernels compute their offsets with. A host stager reads
 * that section. Duplicating it here is how the CB map in this file came to contradict the
 * kernels for a whole round.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE OP IS, VALUE FOR VALUE
 * ---------------------------------------------------------------------------
 * The oracle is `pn_qwen35moe_ssm_recurrence()` in Pantheon's `src/models/qwen35moe_ssm.c`,
 * which the CPU arm and every device arm call — so a device arm that disagrees with it
 * disagrees about the matmuls and nothing else. Per SSM layer, per decode token, with
 * `S = 128`, 16 k-heads, 32 v-heads and `conv_dim = 2*key_dim + val_dim = 8192`:
 *
 *   PHASE CONV
 *     out[c] = SiLU( qkv[c]*w[c][k-1] + sum_{t<k-1} window[c][t]*w[c][t] )    c < conv_dim
 *     window[c] rolls: drop the oldest tap, append qkv[c]
 *   then the stream splits q | k | v at key_dim, 2*key_dim.
 *
 *   PHASE DELTA, per v-head h, with decay = exp(g_h), beta_h, scale = 1/sqrt(S):
 *     (a) S *= decay
 *     (b) pred_j  = sum_i S[j][i] * k[i]
 *     (c) delta_j = (v_j - pred_j) * beta
 *     (d) S[j][i] += k[i] * delta_j
 *     (e) y_j     = scale * sum_i S[j][i] * q[i]        READ AFTER THE WRITE
 *
 * Step (e) reading the UPDATED state is not a detail: doing it before (d) is a different
 * recurrence that happens to look plausible for one token.
 *
 * The two per-head NORMS of this block — the q/k L2 norm between the phases, and the output
 * gated norm after them — are NOT in this op. They are ordinary RMS norms at a second width
 * (4 tile columns against the model's 128) and belong to the shipped reduce family on a
 * second narrow residency. `resident_gdn_protocol.h` carries that argument in full.
 *
 * ---------------------------------------------------------------------------
 * WHY THE STATE IS fp32 AND WHAT THE BAR ACTUALLY IS
 * ---------------------------------------------------------------------------
 * The carried state is fp32 in card DRAM: 128x128 per head x 32 heads x 24 layers =
 * 50.25 MiB, already charged to the capacity plan's run-time reserve. bf16 would halve the
 * traffic and is not on the table, because this is a RECURRENCE — the CPU oracle
 * accumulates its dots in fp64 on purpose and its own comment names a token-60-90
 * divergence onset for doing them in float32.
 *
 * THE BAR A TENSIX ARM HAS TO MEET IS CUDA'S, NOT THE CPU'S. A Tensix cannot do fp64, so
 * bit-equality with the oracle was never available; the CUDA arm is fp32 and is gated at
 * 1e-3 relative, and that is the number this op is held to. Decided before the gate was
 * written rather than after it failed.
 *
 * AND THE FPU SEES bf16 REGARDLESS OF THE CB FORMAT — measured, and it is the constraint
 * that shapes the whole design. Float32 circular buffers exist (4096 B/tile against bf16's
 * 2048, `kernels/cb_tables/vector_add_float32.cbtable` ships one), but a Float32 CB unpacks
 * into `Float16_b` in the source registers unless EVERY CB of the program is Float32.
 * So fp32 buys accurate STORAGE and fp32 destination accumulation; the multiplies inside
 * the dots run on bf16 mantissas at MATH_FIDELITY 4 whatever the CB says. That is why
 * `dst_accum_mode 1` is mandatory in this family's CB table rather than optional.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE READER AND THE WRITER OWE, because the compute surface cannot express it
 * ---------------------------------------------------------------------------
 * The broadcast headers hold BROADCAST_COL only — `mul`/`sub` against column 0 of a second
 * tile. There is no row broadcast and no scalar broadcast. That is sufficient, because the
 * TRANSPOSED state layout makes every per-head scalar a per-ROW scalar, but it fixes what
 * has to be staged:
 *
 *   - `k` and `q` ROW-REPLICATED — [r][i] = k[i] for EVERY row r. THIS PARAGRAPH USED TO SAY
 *     "as COLUMN tiles (value at [i][0])" AND THAT WAS AN OVER-READ OF THE SILICON PROBE.
 *     The probe that proved the reduce scaler is PER-POSITION staged its scaler with
 *     `tests/test_ops.c build_column_scaler_tile()`, which writes `plain[row*32 + column] =
 *     s[column]` — the value for reduce position `c` is present at every `[r][c]`, so the
 *     measurement cannot tell "column 0 of row c" from "row 0, column c" from "replicated".
 *     Row-replicated is the layout that was actually measured, so it is the one the triad
 *     stages; column-0-only would have fed every reduce position past the first a zero.
 *   - `v` as a TRUE COLUMN tile (value at [j][0]), so `v - pred` is a plain eltwise sub of
 *     two columns — a reduce result's non-column-0 lanes are packed as zero, so `pred` is a
 *     genuine column vector;
 *   - `decay` and `beta` as CONSTANT column tiles (the same scalar in all 32 rows), so
 *     `bcast_col` applies a per-head scalar to a whole tile, and `scale` the same way,
 *     splatted from `kGdnScaleBits` without arithmetic;
 *   - the row-replicated `k` tile is SIMULTANEOUSLY the (b) reduce's scaler operand and the
 *     (d) rank-1 operand, so ONE tile serves both uses. That collapse is why this family has
 *     no separate replica buffer and why CB index 4 is free for the reader's private
 *     scratch. The rank-1 staging is not optional and the two documented alternatives are
 *     both gone: there is no FMA on this machine (ELWMUL hardcodes `acc_to_dest` to 0, which
 *     is why `resident_rope_compute.c` stages two products into separate CBs and adds them),
 *     and `matmul_tiles` here is legal only at `ct_dim == rt_dim == 1`, so the `kt_dim = 1`
 *     matmul the scoping offered as the alternative is not a shape this backend compiles.
 *   - the WRITER transposes `y` from column 0 back into a ROW, because the gated norm that
 *     follows wants one row per v-head. 32 strided word moves a head.
 *
 * ---------------------------------------------------------------------------
 * WHAT CROSSES THE LINK, WHICH IS THE POINT OF THE OP
 * ---------------------------------------------------------------------------
 * Per SSM layer the host currently pulls 49,408 B and the recurrence costs 0.225 s/token,
 * 35% of a 0.636 s token. With the block on the card the only things crossing are `alpha`
 * and `beta` DOWN and `decay` and `beta` UP — 32 scalars each, 256 B a layer in each
 * direction, ~6 KB a token, ~1.2 ms. Every value still crosses exactly once.
 *
 * The partial moves are traps and the numbers say so: after conv1d alone the host still
 * pulls 49,408 B (a depthwise conv is width-preserving, so moving it alone is 1.00x), and
 * after conv1d + L2 + the GQA repeat it pulls 65,536 — 1.33x WORSE than doing nothing,
 * because a materialised repeat doubles q and k. The block moves whole or not at all.
 *
 * @status Parameter layout and CB placement. The reader/compute/writer triad IS written
 *   (`kernels/resident_gdn_{reader,compute,writer}.c`), builds, and is embedded in the
 *   program blob — but it has NO CALLER, so nothing in this file has ever executed on the
 *   card. `tests/test_gdn_ops.c` is the whole of the evidence behind it.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_GDN_H
#define FT_TT_OP_GDN_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_launch.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_queue.h"

/* ================================================================
 * The parameter block — mirrors kernels/resident_gdn_protocol.h
 * ================================================================ */

/** Words of one GDN parameter block. A word a phase does not use is written 0 by the host
 *  and never read by the kernel.
 *
 *  THE COORDINATES ARE PACKED TWO TO A WORD and every other family here spends a word each.
 *  This op has FIVE DRAM endpoints where rope has three; spelled the rope way that is 17
 *  words, and `FT_TT_QUEUE_PARAM_WORDS` is 16 for EVERY family — a seventeenth word changes
 *  the shared slot stride and breaks on the card. The pack is forced, not preferred. */
typedef enum ft_tt_op_gdn_field {
    FT_TT_OP_GDN_SHAPE = 0,           /**< b0 phase, b1 S/32, b2 v-heads, b3 k-heads */
    FT_TT_OP_GDN_CONV_TILE_COLS = 1,  /**< conv_dim/32 in the low half-word (256 is not a
                                       *   byte); bytes 2, 3 the CONV column slice's high
                                       *   bytes (kGdnByteColFirstHigh) */
    FT_TT_OP_GDN_CONV_TAPS = 2,       /**< b0 ssm_conv; the window is taps-1 deep */
    FT_TT_OP_GDN_ACTIVATION_XY = 3,   /**< x low half-word, y high */
    FT_TT_OP_GDN_ACTIVATION_IN = 4,   /**< CONV: qkv.  DELTA: the staged q|k|v */
    FT_TT_OP_GDN_ACTIVATION_OUT = 5,  /**< CONV: conv_out.  DELTA: y, one row per v-head */
    FT_TT_OP_GDN_CONSTANT_XY = 6,
    FT_TT_OP_GDN_CONSTANT_WEIGHT = 7, /**< CONV only: the conv1d weight [conv_dim, taps] */
    FT_TT_OP_GDN_CONSTANT_GATE = 8,   /**< DELTA only: decay[H] then beta[H], fp32 */
    FT_TT_OP_GDN_STATE_XY = 9,
    FT_TT_OP_GDN_STATE_CONV = 10,     /**< CONV only: the rolling window; READ AND WRITTEN */
    FT_TT_OP_GDN_STATE_RECURRENT = 11,/**< DELTA only: [H,S,S] fp32; READ AND WRITTEN */
    FT_TT_OP_GDN_SCALE_BITS = 12,     /**< fp32 bits of 1/sqrt(S) */
    FT_TT_OP_GDN_TILE_BYTES = 13,
    FT_TT_OP_GDN_FLAGS = 14,          /**< variant bits, FT_TT_OP_GDN_FLAG_* */
    FT_TT_OP_GDN_TILE_COUNT = 15,     /**< TOTAL tiles out — all the writer reads */
    FT_TT_OP_GDN_WORDS = 16
} ft_tt_op_gdn_field;

/** Which half of the block a descriptor runs. Mirrors kGdnPhase* on the device. 0 is
 *  deliberately not a phase, so a zeroed parameter block is invalid rather than plausible. */
#define FT_TT_OP_GDN_PHASE_CONV  1u
#define FT_TT_OP_GDN_PHASE_DELTA 2u

/** CB indices, matching the compile-time args the GDN triad is built with.
 *
 *  INDEX 2 AND INDEX 4 ARE NOT WHAT THE PAPER DESIGN SAID, and the difference is the whole
 *  of the SCALER/REPLICA collapse above. Index 2 carries the ROW-REPLICATED k and q — one
 *  tile serving both the reduce scaler and the rank-1 update — which frees index 4, and the
 *  reader takes it for the private staging buffer its bulk-read-then-word-copy idiom needs.
 *  That buffer is never reserved, never pushed and never popped by ANY processor, which is
 *  exactly what makes writing through its write pointer safe: the pointer can never have
 *  moved off the buffer base. The round's spec named index 7 for it, which is CB_PROD — the
 *  compute thread pushes and pops that inside the same descriptor, so scratch there would
 *  land on live pages.
 *
 *  INDEX 8 EXISTS BECAUSE A BUFFER MAY HAVE ONLY ONE PRODUCER FOR THE LIFE OF THE RESIDENCY,
 *  not merely one per descriptor. PHASE DELTA's compute-private working state used to share
 *  index 0 with PHASE CONV's qkv stream, and the two phases push it from DIFFERENT
 *  PROCESSORS: the reader's `cb_push_back` INCREMENTS the `tiles_received` stream register,
 *  while the compute's `cb_push_back` keeps its own local counter and STORES that over the
 *  register. Every increment the reader had made was therefore discarded by the first DELTA
 *  push, the register jumped backwards, and the consumer's `received - acked` wrapped to a
 *  huge unsigned — so `cb_wait_front` on that buffer never blocked again for the rest of the
 *  run. Measured on silicon 2026-08-18: PHASE CONV's second pass copied whatever the eight
 *  CB_IN pages happened to hold, source tile `s` for output tile `t` satisfying
 *  `s ≡ t (mod 8)` at every one of 256 tiles. See kernels/resident_gdn_protocol.h. */
#define FT_TT_OP_GDN_CB_IN      0u  /**< CONV ONLY: the qkv stream. The READER is its only
                                     *   producer, in either phase. */
#define FT_TT_OP_GDN_CB_STATE   1u  /**< CONV: the rolling window. DELTA: one state tile-row */
#define FT_TT_OP_GDN_CB_SCALER  2u  /**< DELTA: k then q, ROW-REPLICATED — scaler AND rank-1 */
#define FT_TT_OP_GDN_CB_VECTOR  3u  /**< CONV: the weight taps. DELTA: v, decay, beta, scale */
#define FT_TT_OP_GDN_CB_STAGE   4u  /**< the READER's private scratch — never pushed or popped */
#define FT_TT_OP_GDN_CB_PRED    5u  /**< staging: the (b) and (e) reduce columns */
#define FT_TT_OP_GDN_CB_DELTA   6u  /**< CONV: the running tap sum. DELTA: (v - pred) * beta */
#define FT_TT_OP_GDN_CB_PROD    7u  /**< staging: k_replica * delta — there is no FMA */
#define FT_TT_OP_GDN_CB_WORK    8u  /**< DELTA ONLY: the compute thread's private working
                                     *   state, which it both packs into and unpacks from */
#define FT_TT_OP_GDN_CB_OUT     16u
/** MULTI-ROW DELTA ONLY: the compute thread's carry of one head's whole state from one verify
 *  row to the next (kGdnCbCarry). Compute-produced AND compute-consumed, like CB_WORK. */
#define FT_TT_OP_GDN_CB_CARRY   9u
#define FT_TT_OP_GDN_CB_COUNT   11u

/** Where a v-head's constants sit in CB_VECTOR, after its `state_tiles` v columns. Mirrors
 *  `kGdnVector*` in kernels/resident_gdn_protocol.h; the reader REFUSES a descriptor whose
 *  CB_VECTOR cannot hold `state_tiles + FT_TT_OP_GDN_VECTOR_CONSTANTS` pages, so this is a
 *  number `ft_tt_op_gdn_place_cbs()` has to know rather than a device-side detail. */
#define FT_TT_OP_GDN_VECTOR_DECAY     0u
#define FT_TT_OP_GDN_VECTOR_BETA      1u
#define FT_TT_OP_GDN_VECTOR_SCALE     2u
#define FT_TT_OP_GDN_VECTOR_CONSTANTS 3u

/**
 * Tile columns one CONV block covers — the THIRD two-sided compile-time contract in this
 * family, after the CB indices and the tile size.
 *
 * The compute kernel cannot read it off the CB interface (`cb_interface` does not exist on
 * the MATH thread and the reference fails at LINK time), and resident-protocol rule 3 forbids
 * a trip count taken from runtime buffer state in any case. So the depth is baked into BOTH
 * dataflow kernels as a compile-time arg — and the host needs the same number, because it is
 * the host that sizes the buffers the reader then checks the depth against.
 *
 * ---------------------------------------------------------------------------
 * TWO SINCE 2026-08-27, AND THE READER IS PHASE CONV'S WALL
 * ---------------------------------------------------------------------------
 * PHASE CONV is 20.1% of the decode token and MATH was ELIMINATED as its wall — deleting the
 * SiLU from the image entirely moved it 0.09%. What was left was the reader, and at depth 1
 * `read_one_conv()` issues 1,792 read barriers and 2,816 single-tile 4 KiB NoC reads a
 * descriptor. Halving that count is worth, measured on silicon against a control re-taken in
 * the same session at a 0.07% drift:
 *
 *     depth 1   CONV 1162.0 / 1162.2 us a descriptor   WALL 0.1378 / 0.1379 s/token
 *     depth 2   CONV  939.4 us                         WALL 0.1325 s/token
 *
 * **-19.2% on PHASE CONV and -3.85% on the token, 7.26 -> 7.55 tok/s**, `cmp` IDENTICAL at
 * 189 B / 80fc6259 on every rung. CONV fell 5.4 ms/token and the wall fell 5.3 — the saving
 * converts at ~98%, which is what a term that is not overlapped with anything looks like. It
 * is the same shape PHASE DELTA had: reader 1.76x while its compute arm moved 0.33%.
 *
 * AND THE SAVING SATURATES AT TWO — DEPTH 4 IS REFUTED AS A RUNG, not merely blocked. On the
 * one head-split width where a depth-4 build runs clean, PHASE CONV measures **945.6 / 945.7
 * us** against depth 2's **939.5** on the same binary at the same one-await bracket: 0.7%
 * WORSE. Halving the barrier count again buys nothing, so 2 is the end of this ladder whatever
 * happens to the grid collision below.
 *
 * DEPTH 4 IS ALSO REFUSED ON SILICON AT THE SHIPPED HEAD SPLIT, and the refusal is correct. CB_VECTOR is
 * 112 pages there and CONV's push chunk is `depth*taps` = 16; a CONV descriptor leaves the
 * write pointer at page 16, and a DELTA descriptor on a FOUR-MEMBER head split then pushes
 * `8 heads * (state_tiles + 3)` = 56 more, landing at 72 — which is not a multiple of 16, so
 * `ft_tensix_cb_push_grid_is_safe()` refuses the next CONV. At depth 1 and 2 the arithmetic
 * lands back on the grid (16 mod 4 and 16 mod 8), which is why the same code is clean there.
 * The observed shape matches exactly: 1,175 of 1,176 CONV descriptors refused, each recovering
 * on a re-arm that rebases the pointer, and one clean descriptor per re-arm. **The expression
 * was then tested as a FOUR-POINT PREDICTION and held 4 of 4** — clean at N = 1 and 2 (0
 * refusals, `cmp` IDENTICAL), refusing at N = 4 (1,175) and N = 8 (1,559), which is the
 * arithmetic `16 + (32/N)*7 mod 112` landing on a multiple of 16 or not.
 *
 * THE DEPTH IS NOT A PLACEHOLDER AT ANY VALUE. The reader refuses any depth whose blocks do
 * not fit the placed buffers, and CB_VECTOR must hold `2 * depth * taps` weight pages rounded
 * onto `lcm(state_tiles + 3, depth * taps)`. Moving it is a measured change to this constant,
 * `place_cbs` and both kernel build lines together.
 *
 * ---------------------------------------------------------------------------
 * OVERRIDABLE AT BUILD TIME, AND NOW CHECKED ON THE CARD
 * ---------------------------------------------------------------------------
 * A depth SWEEP needs the host and both kernels moved together, so this is `#ifndef`-guarded
 * exactly as `FT_TT_MATMUL_DST_SYNC` is: `-DFT_TT_OP_GDN_CONV_DEPTH=N` on the host build, and
 * reader compile-time arg 7 / compute compile-time arg 9 set to the same N in a program
 * directory built for that arm. Nothing derives one side from another.
 *
 * THE DIRECTION THAT USED TO BE SILENT IS NOW REFUSED ON THE CARD. Every device check this
 * family had asks whether the PLACED BUFFERS CAN SERVE the image's depth, so a host that
 * placed for a LARGER depth than the kernels were built at passed all of them: the arm ran at
 * the kernels' depth inside the host's placement and produced a correct answer and THE
 * CONTROL'S TIMING, reported as the arm's. `ft_tt_op_gdn_parameters()` now writes this value
 * into byte 3 of the CONV-taps word and both kernels refuse a CONV descriptor that does not
 * match the constant they were compiled with — see `kGdnByteConvDepth` in
 * kernels/resident_gdn_protocol.h.
 *
 * IT MUST FIT A BYTE, because that is where it rides, and nothing reachable is near the
 * limit. THE ARENA ENDS THIS LADDER, NOT THE READER — measured off hardware by calling
 * `ft_tt_op_gdn_place_cbs()` at each rung, not modelled: at Ornith's 4 taps and `state_tiles`
 * 4, CB_VECTOR is rounded onto `lcm(state_tiles + 3, depth * taps)` = `lcm(7, 4*depth)`, a 7
 * that shares no factor with anything, so the buffer grows FOURFOLD a doubling. The family's
 * ten buffers place **85 / 115 / 187 pages at depth 1 / 2 / 4** of the 330-page arena and
 * **depth 8 returns ENOMEM**.
 *
 * AND THE RUNGS BETWEEN THE POWERS OF TWO ARE NOT RUNGS. Both kernels shrink the depth until
 * it divides `conv_tile_cols` (256 at Ornith), so a build at 3 or 6 runs at 2 or 4 while the
 * host placed onto `lcm(7, 12)` or `lcm(7, 24)` — a placement the shrunk depth's index grid
 * does not divide, which the reader refuses. The reachable sweep is **1, 2, 4**.
 */
#ifndef FT_TT_OP_GDN_CONV_DEPTH
#define FT_TT_OP_GDN_CONV_DEPTH 2u
#endif

/** The knob that moves the HOST'S side of the contract without a host rebuild. See
 *  `ft_tt_op_gdn_conv_depth()`. */
#define FT_TT_OP_GDN_CONV_DEPTH_ENV "FT_TT_OP_GDN_CONV_DEPTH"

/**
 * The CONV block depth this host places its circular buffers for and DECLARES on the wire.
 *
 * `FT_TT_OP_GDN_CONV_DEPTH` unless `FT_TT_OP_GDN_CONV_DEPTH` names another in 1..255, so a
 * depth sweep runs several program directories against ONE host binary — which is what makes
 * its per-descriptor figures comparable, since every `us each` in this backend is
 * host-bracketed. Asking for a depth the armed program was not built at is REFUSED on the
 * card, not silently ignored.
 *
 * @return the depth, or 0 for a knob set to something unusable — in which case
 *         `ft_tt_op_gdn_parameters()` and `ft_tt_op_gdn_place_cbs()` both refuse
 *
 * Thread-safe: after the first call    Reentrant: no    Signal-safe: no
 */
uint32_t ft_tt_op_gdn_conv_depth(void);

/**
 * The knob's PARSER, split out of the cached accessor so it can be tested at more than one
 * value in one process.
 *
 * @param value  the environment string, or NULL/empty for "not set"
 * @return the depth, `FT_TT_OP_GDN_CONV_DEPTH` when unset, or 0 for anything unusable —
 *         never a silent fallback to the default, which is how a sweep measures its control
 *         five times and reports a ladder
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_op_gdn_conv_depth_parse(const char *value);

/**
 * PHASE DELTA's DST batch width — the FOURTH compile-time contract of this family, and the
 * only one whose failure mode is slow rather than wrong.
 *
 * MEASURED ON SILICON 2026-08-18: the DELTA descriptor costs 3,368 µs and is bound by neither
 * compute (95 GFLOP/s on a Tensix that can issue ~1,400) nor bandwidth (1.27 GB/s, 3–5% of one
 * core's DRAM). It is bound by ~2,688 single-tile DST sections a descriptor at ~1,250 ns each,
 * against ~64 cycles of tile arithmetic — a `ZEROACC` over the whole DST half, two cross-thread
 * semaphore hops and a full ADDR_MOD + MOP reprogram, none of which get bigger when four tiles
 * ride in one section. `reduce_row_sum` in the compute kernel already batched this way; the
 * three per-tile loops around it never used a DST index above 0.
 *
 * THE CEILING IS FOUR AND IT IS NOT A TUNING KNOB. DST is 16 tiles of 32x32, DST_SYNC half
 * hands MATH one half at a time, and `dst_accum_mode 1` — MANDATORY for the one op in this
 * backend that carries state ACROSS TOKENS — halves it again. Four is also exactly Ornith's
 * `state_tiles`, so every DELTA loop batches whole. `ftg_dst_batch()` in
 * kernels/resident_gdn_protocol.h is the single implementation of the clamp, shared by the
 * reader, the compute kernel and `resolve_cb_pages()` here.
 *
 * 1 IS THE IDENTITY. Every batched loop collapses to the per-tile sequence that shipped, which
 * is what makes a `dst_batch_max = 1` build the exact control arm — and it is also what a short
 * build line silently produces, since an out-of-range `ft_tensix_compile_time_arg` folds to a
 * literal 0 under SFPI GCC 15.1.0 and `ftg_dst_batch()` maps 0 to 1. Unlike the CONV depth,
 * whose zero-fold makes every CONV descriptor REFUSE, this one cannot be caught by watching for
 * an error: the arg COUNT beside each build line, pinned by test_gdn_ops.c, is the only guard.
 */
#define FT_TT_OP_GDN_DST_BATCH 4u

/**
 * The highest PHASE DELTA WRITER RUNG this placement supports — the SEVENTH compile-time
 * contract of this family, and the one that sizes CB_OUT.
 *
 * A NoC write barrier is a per-core DRAIN. The DELTA writer took one PER TILE —
 * `v_heads * state_tiles * (state_tiles + 1)` = 640 a descriptor at Ornith's shape — against
 * this backend's own rule of one barrier per stream group, the rule that makes the block
 * reader's barrier worth 5.9x. `ftg_writer_block()` in kernels/resident_gdn_protocol.h is the
 * one implementation of the group, shared by the writer and by `resolve_cb_pages()` here.
 *
 * THIS CONSTANT SIZES THE BUFFER, IT DOES NOT SELECT THE BEHAVIOUR. The writer's rung is
 * compile-time arg 3 of `WRITER_CTARGS` and the writer clamps it to half of what is placed
 * here, so the two sides cannot disagree into a hang — the failure mode `reserve_back(cb_out,
 * 4)` produced against a two-page CB_OUT on silicon 2026-08-19. Sizing for the ceiling is
 * what `FT_TT_OP_GDN_DST_BATCH` already does for CB_PROD, and for the same reason: it keeps a
 * rung-0 build an exact control arm for a rung-1 one, on one host binary.
 *
 * ---------------------------------------------------------------------------
 * 1, AND THE MEASUREMENT SAYS SO ON BOTH SIDES. Silicon 2026-08-19, `cmp` IDENTICAL and
 * `company=0` on every arm.
 *
 * THE BARRIER COUNT IS A NULL. Rungs 0/1/2 leave 640/128/32 barriers in a DELTA descriptor
 * and it costs 1836.0 / 1835.3 / 1845.1 us at N=1 and 680.8 / 680.4 / 684.8 at N=4 — a 20x
 * reduction in a per-core drain buys 0.0%, and the largest group is reproducibly ~0.5% WORSE
 * because the writer waits longer before it starts draining. The ~413 us the scoping put on
 * these barriers was writer-THREAD time that was never on the critical path.
 *
 * WHAT THE BUFFER BUYS IS CAPACITY, AND IT SATURATES AT TWO TILE-ROW GROUPS. One kernel image
 * (the per-tile sequence) over three placements — CB_OUT at 2, 10 and 40 pages — costs
 * 1908.3 / 1835.2 / 1835.1 us at N=1 and 697.3 / 679.4 / 679.1 at N=4. So 2 -> 10 pages is
 * worth 3.8% / 2.6% and 10 -> 40 is worth NOTHING: with only two pages the compute thread's
 * `reserve_back(cb_out, 1)` hard-handshakes the writer on every tile, and ten pages is all it
 * takes to decouple them. PHASE CONV is flat at 1222.3-1223.2 across all of it, which is the
 * null that says this is the DELTA cadence and not a general effect.
 *
 * 1 = one barrier per TILE-ROW, `state_tiles + 1` = 5 pages at Ornith, so CB_OUT is 10 pages
 * of 4 KiB = 40 KiB. Rung 2 would place 40 pages for 160 KiB of a 1,320 KiB arena and buy
 * nothing measurable, so the 120 KiB stays in the arena. The rung-2 code path is kept because
 * the clamp makes it free and it is the third point of the ladder above.
 */
#define FT_TT_OP_GDN_WRITER_RUNG 1u

/**
 * The two PHASE DELTA restructuring bitmasks the shipped kernels are built with — the fifth
 * and sixth compile-time contracts of this family, and the two that carry the measurement.
 *
 * MEASURED ON SILICON 2026-08-19, all arms `cmp` IDENTICAL against the 48-token oracle, on a
 * card with no other process on it, baseline reproduced to 0.2 µs (3351.2 / 3351.0):
 *
 *   READER bit 0, the column-major replica build      3351.2 → 1918.7 µs   −42.7%
 *   COMPUTE bit 0, the fused state emit               3351.2 → 3340.2 µs   −0.33%
 *   both                                              3351.2 → 1905.4 µs   −43.1%
 *
 * THAT PAIR OF NUMBERS IS THE WHOLE DIAGNOSIS. The compute arm removes FOUR of the
 * twenty-seven tile operations a tile-row — 15% of the MATH and UNPACK work — and moves the
 * descriptor by a third of a percent. The reader arm removes 960 of the 1,024 LOADS in each
 * of 256 replica tiles, touches `brisc` and nothing else, produces byte-identical circular
 * buffer contents, and moves it by 43%. PHASE DELTA was never bound by its compute thread:
 * it was bound by ~262,000 scalar L1 word copies on the READER, and the ~958 ns a
 * "single-tile operation" was that, seen through a division.
 *
 * COMPUTE bit 1 is NOT in the shipped mask, and the reason is evidence rather than doubt. Its
 * first build refused on silicon (it applied the batched push to CB_OUT, which holds two
 * pages); with CB_OUT excluded it is `cmp` IDENTICAL and worth **0.56%** — 1,905.3 → 1,894.7 µs
 * against its own baseline in the same session. That is a real gain on ONE run, of the same
 * size as the DST batch's 1.2%, and both are what is left once the reader stops being the
 * wall. It stays out of the default until it has a second run behind it.
 */
#define FT_TT_OP_GDN_READER_EXTRAS 1u
#define FT_TT_OP_GDN_DELTA_EXTRAS  1u

/** Every circular buffer of this family is Float32 (kernels/cb_tables/gdn_float32.cbtable)
 *  and every DRAM operand it touches is therefore an fp32 tile.
 *
 *  THIS IS THE ONE THE HOST GOT WRONG. `place_cbs` used to size five of the nine buffers
 *  from `activation_in.tile_bytes` while the kernels are compiled with TILE_BYTES = 4096 and
 *  validate only the word `parameters()` happens to have taken from the STATE operand. A
 *  request with a bf16 activation — which is what the shipped strip allocator produces and
 *  what this suite's own fixture used — therefore placed 2048-byte pages under a reader that
 *  writes 4096-byte tiles into them, and the tile_bytes refusal did not fire because it was
 *  reading the other operand. `ft_tt_op_gdn_parameters()` now refuses any operand that is
 *  not this size, so the two cannot disagree. */
#define FT_TT_OP_GDN_FP32_TILE_BYTES FT_TT_OP_TILE_BYTES_FP32

/** Qwen3.5 (Ornith 9B), read off Pantheon's `pn_qwen35moe_config` — not assumed. */
#define FT_TT_OP_GDN_QWEN35_STATE     128u /**< head_k_dim == head_v_dim */
#define FT_TT_OP_GDN_QWEN35_K_HEADS   16u  /**< ssm_n_group */
#define FT_TT_OP_GDN_QWEN35_V_HEADS   32u  /**< ssm_dt_rank */
#define FT_TT_OP_GDN_QWEN35_CONV_TAPS 4u   /**< ssm.conv_kernel */
#define FT_TT_OP_GDN_QWEN35_CONV_DIM  8192u /**< 2*key_dim + val_dim */

/** Mirrors kGdnMultiRowMax: rows one multi-row DELTA descriptor may carry. */
#define FT_TT_OP_GDN_MULTI_ROW_MAX 16u
/** The widest state (in tiles, S/32) whose multi-row buffers are placed: Ornith's S = 128. */
#define FT_TT_OP_GDN_MULTI_ROW_STATE_TILES_MAX 4u

/** One GDN call. The operands a phase does not read may be left zeroed. */
typedef struct ft_tt_op_gdn_request {
    uint32_t phase;      /**< FT_TT_OP_GDN_PHASE_* */
    uint32_t state_size; /**< S; a multiple of 32 */
    uint32_t v_heads;
    uint32_t k_heads;    /**< must divide v_heads — the GQA repeat is vh % k_heads */
    uint32_t conv_dim;   /**< a multiple of 32 */
    uint32_t conv_taps;
    /** PHASE DELTA head slice — the v-heads THIS member owns, for a head-split across a
     *  group of GDN residencies. `head_count` 0 means the whole descriptor and is what an
     *  unsliced caller leaves it at; a 0 count encodes as zero bytes, so an unsliced
     *  descriptor is BYTE-IDENTICAL to the one that shipped. `head_first + head_count` must
     *  not exceed `v_heads`, and a CONV descriptor may not carry a slice at all — the
     *  convolution divides by CHANNEL, not by head, and its operands are tap-major planes a
     *  column slice is not contiguous in. See kGdnByteHeadFirst in
     *  kernels/resident_gdn_protocol.h for why the slice rides in the CONV-taps word. */
    uint32_t head_first;
    uint32_t head_count;
    /** PHASE CONV column slice — the TILE COLUMNS this member convolves, at the full plane
     *  stride. `col_count` 0 means the whole stream and encodes as zero bytes, so an unsliced
     *  CONV descriptor is BYTE-IDENTICAL to the one that shipped. Each is up to
     *  FT_TT_OP_GDN_COL_MAX: the low bytes ride in the same two bytes of the CONV-taps word as
     *  the DELTA head slice (each phase reads its own meaning), the high bytes in the
     *  tile-column word's high half — zero, and the shipped bytes, for any slice under 256. */
    uint32_t col_first;
    uint32_t col_count;
    ft_tt_op_operand activation_in;
    ft_tt_op_operand activation_out;
    ft_tt_op_operand conv_weight;   /**< CONV */
    ft_tt_op_operand gate;          /**< DELTA: decay then beta, fp32 */
    ft_tt_op_operand state_conv;    /**< CONV; read and written */
    ft_tt_op_operand state_recurrent; /**< DELTA; read and written */
    /** S2: when set, the phase's state is READ from state_conv / state_recurrent and WRITTEN
     *  here (same NoC endpoint — one coordinate word) — FT_TT_OP_GDN_FLAG_STATE_OUT. */
    ft_tt_op_operand state_out;
    /** S2c: the row select (0-31; 0 = the shipped row-0 walk). CONV needs the strip flag. */
    uint32_t row;
    /** MULTI-ROW (DELTA PM 5, CONV PM 6 in strip mode): rows [row, row + row_count) in one
     *  descriptor; 0 or 1 is the one-row walk, byte for byte. Row r's carried state (DELTA:
     *  the recurrent state, CONV: the rolled window) goes to `state_rows_out[r]` (same bank as
     *  the state it read), carried by the EXTENSION block ft_tt_op_gdn_row_extension()
     *  encodes into the next parameter slot. DELTA's gate is row-major at ftg_gate_stride(). */
    uint32_t row_count;
    ft_tt_op_operand state_rows_out[FT_TT_OP_GDN_MULTI_ROW_MAX];
    /** G2a (CONV, strip mode): when set, the reader takes its stream input from this bf16
     *  strip (the projection itself, 2048 B tiles) and widens it, instead of the fp32
     *  `activation_in` (which still names the input buffer the output shares a bank with).
     *  Encodes word 4 = its address, word 8 = its NoC coordinate, FT_TT_OP_GDN_FLAG_CONV_BF16_IN. */
    ft_tt_op_operand stream_bf16;
    /** FT_TT_OP_GDN_FLAG_* bits, encoded into word 14. 0 = the shipped walk. */
    uint32_t flags;
} ft_tt_op_gdn_request;

/** The widest stream, and the furthest CONV column slice, a descriptor encodes: 16 bits each
 *  (kGdnConvTileColsMask; the slice's high bytes are kGdnByteColFirstHigh/ColCountHigh). */
#define FT_TT_OP_GDN_COL_MAX 0xFFFFu

/**
 * FT_TT_OP_GDN_CONV_SLICE_GRANULE — the column count every CONV slice must be a multiple of.
 *
 * MEASURED ON SILICON 2026-11-17 (p150, Qwen3.8, 16-way split): a member given an 18-column
 * slice [300, 318) RAN its CONV to completion, and the DELTA that followed on the same member
 * failed; member 0 (the S1 puller, Tensix 90) then never finished that DELTA, ignored the stop
 * request and refused to boot until `ft_tt_core recover 90 --rung 2`. No kernel check refuses
 * 18 columns (the block depth 2 divides it), so the host has to.
 *
 * What separates that slice from every shape that has run clean: its CONV traffic leaves
 * CB_STATE (`taps-1` + `taps-2` pages a column) and CB_PROD (`taps-1` a column) off the grid
 * PHASE DELTA indexes them on (`state_tiles`, `dst_batch`) — 18 x 5 = 90 = 6 mod 12 against a
 * 4-page grid. Qwen3.8's 320/80/40/20 and every 256-column model's slices land on it. Both
 * buffers are rebased at the top of every descriptor, so the kernels SHOULD tolerate it; the
 * card-side mechanism is not pinned, and the path is not one this family has ever run clean.
 * So a slice must keep both on grid, and be a multiple of the CONV depth: 4 at Qwen3.8's
 * placement (S 128, 4 taps, depth 2). The whole stream (`col_count` 0) is not a slice.
 *
 * @param state_size  S (state tiles = S / 32)
 * @param conv_taps   the CONV taps
 * @return the granule in tile columns, or 0 for a shape with no CONV placement (refused)
 *
 * Thread-safe: after the first ft_tt_op_gdn_conv_depth()    Reentrant: no    Signal-safe: no
 */
uint32_t ft_tt_op_gdn_conv_slice_granule(uint32_t state_size, uint32_t conv_taps);

/** Mirrors kGdnFlagNocReplicate — change both or neither. */
#define FT_TT_OP_GDN_FLAG_NOC_REPLICATE 8u
#define FT_TT_OP_GDN_FLAG_RED_NO_COPY 16u
/** Mirrors kGdnFlagConvStrip / kGdnFlagRedStripHalf (M9). */
#define FT_TT_OP_GDN_FLAG_CONV_STRIP 32u
#define FT_TT_OP_GDN_FLAG_RED_STRIP_HALF 64u
/** Mirrors kGdnFlagStateOut; set by the encoder from `state_out`, never by a caller. */
#define FT_TT_OP_GDN_FLAG_STATE_OUT 128u
/** Mirrors kGdnFlagTable (PHASE DELTA only): `gate` is the layer's TABLE BLOCK
 *  (kernels/ftgc_table.h) and the reader builds the gate block from the strips it names. */
#define FT_TT_OP_GDN_FLAG_TABLE 512u
/** Mirrors kGdnRowShift: the row select rides in flags bits 16-20 (`row`). */
#define FT_TT_OP_GDN_ROW_SHIFT 16u
/** Mirrors kGdnFlagRedRowsRow0 (RED ONLY): a multi-row reader stages row `row` for every row. */
#define FT_TT_OP_GDN_FLAG_RED_ROWS_ROW0 (1u << 26)
/** Mirrors kGdnFlagConvBf16In; set by the encoder from `stream_bf16`, never by a caller. */
#define FT_TT_OP_GDN_FLAG_CONV_BF16_IN (1u << 27)
/** The bf16 tile bytes `stream_bf16` must carry. */
#define FT_TT_OP_GDN_BF16_TILE_BYTES 2048u
/** Mirrors kGdnRowCountShift: the multi-row count rides in flags bits 21-25 (`row_count`). */
#define FT_TT_OP_GDN_ROW_COUNT_SHIFT 21u

/**
 * Is this a shape the triad can execute? The same predicate the kernel applies, so both
 * sides refuse the same descriptors independently.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_gdn_shape_is_supported(const ft_tt_op_gdn_request *request);

/**
 * Encode `request` into the sixteen parameter words. Pure — touches no device.
 *
 * @return FT_TT_OK, or FT_TT_EINVAL for a null argument or an unsupported shape.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_gdn_parameters(const ft_tt_op_gdn_request *request,
                            uint32_t out_words[FT_TT_OP_GDN_WORDS]);

/**
 * Encode a MULTI-ROW DELTA descriptor's extension block — the sixteen words written into the
 * parameter slot AFTER the descriptor's own: word r = the DRAM address of row r's state output,
 * r < row_count, the rest zero. Pure.
 *
 * @return FT_TT_OK, or FT_TT_EINVAL for a request that is not multi-row DELTA, an unset row
 *         operand, or one outside the recurrent state's bank (one coordinate word carries both)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_gdn_row_extension(const ft_tt_op_gdn_request *request,
                               uint32_t out_words[FT_TT_OP_GDN_WORDS]);

/**
 * Place this op's circular buffers in the L1 arena and encode the descriptor blob.
 *
 * @param out_mask  receives the local CB mask the launch message wants
 * @return FT_TT_OK, FT_TT_EINVAL, or FT_TT_ENOMEM when the arena cannot hold them
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_op_gdn_place_cbs(const ft_tt_op_gdn_request *request,
                           ft_tt_launch_cb_entry out_entries[FT_TT_OP_GDN_CB_COUNT],
                           void *buffer, size_t buffer_bytes, uint64_t *out_mask);

/* ================================================================
 * WHAT ONE LAYER COSTS IN CARD DRAM — every operand, one expression each
 * ================================================================
 *
 * A HOST STAGER ALLOCATES ALL SIX OF THESE AND NOT TWO, which is why there are now six
 * functions where there were two. The two that existed answered the CAPACITY question and
 * were then reached for as buffer sizes, and for the conv window those are not the same
 * number: `conv_dim * (taps-1)` fp32 ELEMENTS is 96 KiB a layer, and the window the reader
 * addresses is `(taps-1)` TAP-MAJOR PLANES of `conv_dim/32` fp32 TILES — 3 MiB a layer, 32x
 * larger, because a plane is a one-row tilized strip and the tile quantum is 32 rows.
 * `ft_tt_op_gdn_conv_state_bytes()` now returns the DRAM figure, because a buffer that is
 * 1/32 of what the kernel reads is not a capacity discrepancy, it is a NoC read landing in
 * whatever the allocator put above it.
 *
 * The recurrent state is unaffected: `[S,S]` per head is a whole number of 32x32 tiles with
 * no padding, so `v_heads*S*S*4` is both the element count and the tile count times 4096.
 *
 * Every one of these returns 0 for a request whose shape this family will not execute, so an
 * unchecked caller allocates nothing rather than something wrong.
 */

/**
 * Bytes of carried recurrent state one LAYER owns: `v_heads * S * S * 4`, which is also
 * `v_heads * state_tiles^2` fp32 tiles.
 *
 * Exposed because the caller allocates it, and because the capacity number this op is
 * charged against should be computed from the same expression that sizes the buffer rather
 * than restated beside it.
 */
size_t ft_tt_op_gdn_state_bytes(const ft_tt_op_gdn_request *request);

/**
 * Bytes of the rolling conv window one LAYER owns, AS THE READER ADDRESSES IT: `(taps-1)`
 * tap-major planes of `conv_dim/32` fp32 tiles.
 *
 * NOT `conv_dim * (taps-1) * 4`. See the note above; that expression is 1/32 of this one.
 */
size_t ft_tt_op_gdn_conv_state_bytes(const ft_tt_op_gdn_request *request);

/** Bytes of the tap-major conv WEIGHT one LAYER owns: `taps` planes of `conv_dim/32` fp32
 *  tiles. A constant — staged once, transposed out of the oracle's row-major [conv_dim,taps]. */
size_t ft_tt_op_gdn_conv_weight_bytes(const ft_tt_op_gdn_request *request);

/** Bytes of ONE conv stream — the qkv input or the conv_out output of PHASE CONV. One row
 *  block of `conv_dim/32` fp32 tiles, so the two are the same size. */
size_t ft_tt_op_gdn_conv_stream_bytes(const ft_tt_op_gdn_request *request);

/** Bytes of the staged `q | k | v` input of PHASE DELTA. q and k are per-K-HEAD and v is
 *  per-V-head, so the three blocks are NOT the same size whenever they need different row
 *  block counts. */
size_t ft_tt_op_gdn_delta_input_bytes(const ft_tt_op_gdn_request *request);

/** Bytes of `y`, the output of PHASE DELTA: a tilized [v_heads, S] fp32 matrix. */
size_t ft_tt_op_gdn_delta_output_bytes(const ft_tt_op_gdn_request *request);

/**
 * Bytes the DELTA gate block's allocation must be at least: `(2*v_heads*4 + 63) & ~63`.
 *
 * The block itself is `decay[v_heads]` then `beta[v_heads]` fp32 and nothing reads past it —
 * but the reader pulls it with ONE NoC read whose length is rounded UP to the 64-byte DRAM
 * granule, so a tight allocation is read off the end of.
 */
size_t ft_tt_op_gdn_gate_bytes(const ft_tt_op_gdn_request *request);

#endif /* FT_TT_OP_GDN_H */
