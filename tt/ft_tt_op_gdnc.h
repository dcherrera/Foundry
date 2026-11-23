/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_gdnc.h
 * @brief Host driver for the CHUNKWISE GATED-DELTA-NET block — Qwen3.5's linear layer over
 *   C POSITIONS AT ONCE. Places the triad's circular buffers, encodes the parameter block,
 *   and builds the constant BAND matrices the convolution becomes a matmul against.
 *
 * The device half is `kernels/resident_gdnc_protocol.h`; the two must agree word for word and
 * `tests/test_gdnc_ops.c` pins the pair off-hardware by parsing that header.
 *
 * THE DRAM LAYOUTS ARE NOT HERE, ON PURPOSE — they are stated ONCE, in
 * `resident_gdnc_protocol.h`'s layout section, beside the `ftgc_*` accessors both dataflow
 * kernels compute their offsets with. Duplicating them here is how the decode family's CB
 * map came to contradict its kernels for a whole round.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE OP IS, VALUE FOR VALUE
 * ---------------------------------------------------------------------------
 * The oracle is `pn_qwen35moe_conv1d_silu()` in Pantheon's `src/models/qwen35moe_ssm.c`,
 * called once per position. Per channel `c`, with `k = ssm_conv` taps and a rolling window
 * `cs[0..k-2]` (0 oldest):
 *
 *     acc        = qkv[c]*w[c][k-1] + sum_{t<k-1} cs[t]*w[c][t]
 *     conv_out[c] = SiLU(acc)
 *     cs rolls:   cs[t] = cs[t+1] for t < k-2;  cs[k-2] = qkv[c]      (the RAW qkv, not SiLU)
 *
 * This op runs that for C positions in ONE descriptor and rolls the window ONCE, to the last
 * `k-1` positions of the chunk. It is bit-for-bit the same recurrence; what changes is that
 * the tap shift, which is a shift along the POSITION axis, becomes a multiply by a constant
 * 0/1 matrix — and a chunk's positions are the 32 ROWS of a tile the projections already
 * produce.
 *
 * ---------------------------------------------------------------------------
 * THE GEOMETRY DECISION, IN ONE SENTENCE
 * ---------------------------------------------------------------------------
 * **A CHUNK OF C ≤ 32 POSITIONS IS THE SAME TILIZED STRIP A SINGLE POSITION IS, WITH ROWS
 * 1..C-1 CARRYING POSITIONS INSTEAD OF PADDING.** `ft_tt_tilize_strip(rows = 1)` and
 * `(rows = C)` produce the same tile count at the same addresses. So the DRAM layout of every
 * activation is unchanged, the four SSM projections that produce it are unchanged (a matvec
 * is `Mt = 1` and the other 31 rows are padding the FPU computed anyway — measured, 1 live
 * row 448.4 µs against 32 live rows 446.7 µs, ratio 0.9960), and the per-head L2/RMS norms
 * batch for free because a ROW reduce over a `row_blocks = heads` strip reduces ALONG the
 * row, one answer per row, i.e. one per position.
 *
 * ---------------------------------------------------------------------------
 * THE TWO OPERANDS THAT ARE NOT THE DECODE FAMILY'S
 * ---------------------------------------------------------------------------
 *   - **THE CONV WEIGHT IS ROW-REPLICATED.** The decode family's tap-major planes carry
 *     `w[c][tap]` in ROW 0 and zeros below; here every row of the plane carries it, because
 *     the tap coefficient is constant across positions and the broadcast surface is
 *     BROADCAST_COL only — there is no row broadcast. Same tile count, same addresses, same
 *     transpose out of the oracle's row-major `[conv_dim, taps]`. Only `valid_rows` changes.
 *   - **THE `2*taps` BAND TILES RIDE AT THE END OF THAT BUFFER**, at tile offset
 *     `taps * conv_tile_cols`. They are `taps`-dependent CONSTANTS, the same eight tiles in
 *     every layer of every model with this tap count, and `ft_tt_op_gdnc_build_band()` is the
 *     one implementation. They ride in the weight buffer rather than in a sixth DRAM endpoint
 *     because `FT_TT_QUEUE_PARAM_WORDS` is 16 for every family.
 *
 * **THE ROLLING WINDOW IS THE DECODE FAMILY'S, BYTE FOR BYTE** — `taps-1` tap-major ROW-0
 * planes of `conv_dim/32` tiles. That is the point: a prompt that prefills in chunks and then
 * decodes one token at a time is ONE sequence with no transcription at the seam. The reader
 * ASSEMBLES the `[C, ·]` window tile out of those planes with `2*(taps-1)` 64-byte reads,
 * because row 0 of an fp32 tile is two 64-byte pieces and 64 bytes is the DRAM granule.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE HOST OWES
 * ---------------------------------------------------------------------------
 *   1. **ROWS `chunk_rows..31` OF `activation_in` MUST BE ZERO.** Output rows beyond the
 *      chunk are computed and discarded, and `SiLU(0) = 0` cannot surprise anyone — a stale
 *      inf or NaN there would propagate through arithmetic nobody inspects.
 *   2. `chunk_rows >= conv_taps - 1`. A chunk shorter than the window cannot REPLACE it, only
 *      shift it, and a partial roll is a different writer. Run a 1..taps-2 tail through the
 *      decode family, which is the same arithmetic on the same bytes.
 *   3. `activation_in` and `activation_out` share ONE packed NoC coordinate, so they must sit
 *      in the SAME DRAM bank. Nothing on the device can catch a violation: the writer lands
 *      the answer at the output's offset INSIDE the input's bank, retires clean, and the
 *      output buffer is never written. Measured on the decode family, silicon 2026-08-18.
 *
 * @status NEW. Parameter layout, CB placement, band construction and the per-operand DRAM
 *   sizes — pure host arithmetic, pinned off hardware by tests/test_gdnc_ops.c.
 * @issues None
 * @todo `FT_TT_OP_GDNC_PHASE_DELTA` is declared in the protocol and refused everywhere. The
 *   chunkwise parallel delta rule is the other half of this family.
 */

#ifndef FT_TT_OP_GDNC_H
#define FT_TT_OP_GDNC_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_launch.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_queue.h"

/* ================================================================
 * The parameter block — mirrors kernels/resident_gdnc_protocol.h
 * ================================================================ */

/** Words of one GDNC parameter block. A word this phase does not use is written 0 by the
 *  host and never read by the kernel. */
typedef enum ft_tt_op_gdnc_field {
    FT_TT_OP_GDNC_SHAPE = 0,          /**< b0 phase, b1 chunk rows, b2 conv taps */
    FT_TT_OP_GDNC_CONV_TILE_COLS = 1, /**< conv_dim/32; a full word — 256 is not a byte */
    FT_TT_OP_GDNC_COL_SLICE = 2,      /**< first column low half-word, count high; 0 = all */
    FT_TT_OP_GDNC_ACTIVATION_XY = 3,  /**< x low half-word, y high; ONE bank for in AND out */
    FT_TT_OP_GDNC_ACTIVATION_IN = 4,  /**< the chunk's tilized [chunk_rows, conv_dim] strip */
    FT_TT_OP_GDNC_ACTIVATION_OUT = 5, /**< conv_out, the same shape */
    FT_TT_OP_GDNC_CONSTANT_XY = 6,
    FT_TT_OP_GDNC_CONSTANT_WEIGHT = 7,/**< taps ROW-REPLICATED planes, then the 2*taps band */
    FT_TT_OP_GDNC_STATE_XY = 8,
    FT_TT_OP_GDNC_STATE_CONV = 9,     /**< the rolling window; READ AND WRITTEN */
    FT_TT_OP_GDNC_TILE_BYTES = 10,
    FT_TT_OP_GDNC_FLAGS = 11,         /**< reserved; written 0, never read */
    FT_TT_OP_GDNC_TILE_COUNT = 12,    /**< TOTAL tiles out */
    FT_TT_OP_GDNC_DELTA_GATE = 13,    /**< DELTA: the gate block, INSIDE the constant bank */
    FT_TT_OP_GDNC_FIRST_ROW = 14,     /**< DELTA: first strip ROW this descriptor owns; 0 in
                                       *   CONV. `kGdncFirstRow` carries the derivation */
    FT_TT_OP_GDNC_DELTA_BAND = 15,    /**< DELTA under the TABLE flag: the strips' band; else 0 */
    FT_TT_OP_GDNC_WORDS = 16
} ft_tt_op_gdnc_field;

/** Which half of the block a descriptor runs. Mirrors kGdncPhase* on the device. 0 is
 *  deliberately not a phase, so a zeroed parameter block is invalid rather than plausible. */
#define FT_TT_OP_GDNC_PHASE_CONV  1u
#define FT_TT_OP_GDNC_PHASE_DELTA 2u

/** CB indices, matching the compile-time args the GDNC triad is built with. */
#define FT_TT_OP_GDNC_CB_IN     0u  /**< CONV: the input tile column. DELTA: a head's q|k|v */
#define FT_TT_OP_GDNC_CB_STATE  1u  /**< CONV: the ASSEMBLED window. DELTA: a head's [S,S] */
#define FT_TT_OP_GDNC_CB_BAND   2u  /**< CONV: the 2*taps band. DELTA: the four constants */
#define FT_TT_OP_GDNC_CB_VECTOR 3u  /**< CONV: taps weight tiles. DELTA: five gate columns */
#define FT_TT_OP_GDNC_CB_PROD   4u  /**< CONV: the 2*taps products — there is no FMA */
#define FT_TT_OP_GDNC_CB_WORK   5u  /**< DELTA: the nine [C,S] working groups */
#define FT_TT_OP_GDNC_CB_SCRATCH 6u /**< DELTA: the READER's private page. NEVER PUSHED */
#define FT_TT_OP_GDNC_CB_DECAY  7u  /**< DELTA: this head's [C,C] decay-ratio tile */
#define FT_TT_OP_GDNC_CB_CC     8u  /**< DELTA: the [C,C] working tiles */
#define FT_TT_OP_GDNC_CB_OUT    16u
#define FT_TT_OP_GDNC_CB_COUNT  10u

/** Every circular buffer of this family is Float32 (kernels/cb_tables/gdnc_float32.cbtable)
 *  and every DRAM operand it touches is therefore an fp32 tile. The decode family got this
 *  wrong once by sizing five of nine buffers from an operand's `tile_bytes` while the kernels
 *  were compiled for 4096, so the host refuses anything else here rather than propagating it. */
#define FT_TT_OP_GDNC_FP32_TILE_BYTES FT_TT_OP_TILE_BYTES_FP32

/** The tile row count, which is also the largest chunk this family can carry. C = 32 IS
 *  FORCED, NOT CHOSEN: a tile is 32 rows and the matmul surface has no sub-tile row
 *  addressing, so a chunk above 32 is two tile-rows and a different op. */
#define FT_TT_OP_GDNC_MAX_CHUNK_ROWS FT_TT_OP_TILE_DIM

/** THE SUB-CHUNK'S ROW ALIGNMENT, and it is a FACE HEIGHT rather than a taste. A tile is four
 *  16x16 faces in the order (0,0) (0,1) (1,0) (1,1), so a row range that starts on a multiple
 *  of 16 is a CONTIGUOUS suffix of the tile's bytes and the writer's `y` write stays one
 *  offset and one length. Mirrors `kGdncFaceRows`. */
#define FT_TT_OP_GDNC_SUB_CHUNK_ALIGN 16u

/** PHASE DELTA's circular-buffer INDEX SPANS. Mirrors `kGdncDelta*` in the device protocol,
 *  which is where they are derived; `tests/test_gdnc_ops.c` pins the pair by parsing it. */
#define FT_TT_OP_GDNC_DELTA_CC_SPAN     24u
#define FT_TT_OP_GDNC_DELTA_VECTOR_SPAN 5u
#define FT_TT_OP_GDNC_DELTA_CONST_TILES 4u
/** Indices inside the program-constant block, and inside a gate run. Mirrors the device's
 *  `kGdncDeltaConst*` and `kGdncGate*`. */
#define FT_TT_OP_GDNC_DELTA_CONST_IDENTITY 0u
#define FT_TT_OP_GDNC_DELTA_CONST_TWO_I    1u
#define FT_TT_OP_GDNC_DELTA_CONST_STRICT   2u
#define FT_TT_OP_GDNC_DELTA_CONST_SCALE    3u
#define FT_TT_OP_GDNC_DELTA_CONST_DECAY    4u
#define FT_TT_OP_GDNC_GATE_P  0u
#define FT_TT_OP_GDNC_GATE_PS 1u
#define FT_TT_OP_GDNC_GATE_B  2u
#define FT_TT_OP_GDNC_GATE_F  3u
#define FT_TT_OP_GDNC_GATE_G  4u
#define FT_TT_OP_GDNC_GATE_RUNS 5u
#define FT_TT_OP_GDNC_DELTA_WORK_SPAN(state_tiles) (9u * (state_tiles))

/**
 * THE CONV BLOCK DEPTH: how many tile columns this family reads behind ONE NoC read barrier.
 *
 * `read_one_column()` issues `1 + 2*(taps-1) + taps` NoC reads and then ONE barrier
 * (`kernels/resident_gdnc_reader.c`), `conv_tile_cols` times — 256 at Ornith. The descriptor
 * costs **680.2 / 681.8 / 682.5 us at 32 / 16 / 8 LIVE ROWS**, 0.34% over a 4x range, because
 * NONE of that cost depends on live rows: it is 256 serialized barrier round trips at
 * 680.2/256 = **2.66 us each**. A barrier is a per-core DRAIN, so batching D columns behind
 * one of them removes `(D-1)/D` of them outright. The decode family's identical change was
 * worth **19.2% of its CONV phase and SATURATED at D = 2**.
 *
 * WHAT IT COSTS IS L1, AND THE ARENA IS WHAT ENDS THE LADDER. At depth D a CONV group reserves
 * `D` pages of CB_IN, `D` of CB_STATE and `D*taps` of CB_VECTOR, and every count is a CONSUMER
 * GRID — `ft_tensix_unpack_ab()` is `read_pointer + tile*page_size` with NO WRAP — so each must
 * both hold two blocks and be a multiple of the span the compute thread indexes across.
 * CB_VECTOR is the one that grows: PHASE DELTA indexes it 5 wide and PHASE CONV `D*taps` wide,
 * so it is rounded onto `lcm(5, D*taps)`, and 5 shares no factor with `4D`. **PRICE EVERY RUNG
 * OFF HARDWARE by calling `ft_tt_op_gdnc_place_cbs()` at it** — that is how the decode family
 * found its depth 8 was ENOMEM before a card arm existed — never by modelling it.
 *
 * AND THE RUNGS BETWEEN THE POWERS OF TWO ARE NOT RUNGS, for the reason the decode family
 * records: both kernels shrink the depth until it divides `conv_tile_cols`, so a build at 3 or
 * 6 runs at 2 or 4 under a placement rounded onto `lcm(5, 12)` or `lcm(5, 24)` that the shrunk
 * depth's index grid does not divide — which the reader refuses.
 *
 * OVERRIDABLE AT BUILD TIME, AND CHECKED ON THE CARD. A depth sweep needs the host and both
 * kernels moved together: `-DFT_TT_OP_GDNC_CONV_DEPTH=N` on the host build and the reader's and
 * compute's depth compile-time arg set to the same N in a program directory built for that arm.
 * Nothing derives one side from another, and the direction that would otherwise be silent — a
 * HOST that placed for a DEEPER block than the kernels were built at — passes every capacity
 * check, runs at the KERNELS' depth inside the host's placement, and returns the RIGHT ANSWER
 * with THE CONTROL'S TIMING REPORTED AS THE ARM'S. That corrupts a measurement rather than an
 * answer, which is why the depth is DECLARED on the wire and compared for STRICT EQUALITY by
 * both kernels — see `kGdncByteConvDepth` in kernels/resident_gdnc_protocol.h.
 */
#ifndef FT_TT_OP_GDNC_CONV_DEPTH
#define FT_TT_OP_GDNC_CONV_DEPTH 1u
#endif

/** The knob that moves the HOST'S side of the contract without a host rebuild.
 *  See `ft_tt_op_gdnc_conv_depth()`. */
#define FT_TT_OP_GDNC_CONV_DEPTH_ENV "FT_TT_OP_GDNC_CONV_DEPTH"

/**
 * The CONV block depth this host places its circular buffers for and DECLARES on the wire.
 *
 * AN UNUSABLE VALUE RETURNS 0 AND EVERY ENTRY POINT REFUSES — it does NOT fall back to the
 * compile-time default, because a knob that silently ignores what it was told is how a sweep
 * measures its control four times and reports a ladder. Read once; ONE LINE PER PROCESS either
 * way, so the rung that actually placed is a printed number rather than an inference from the
 * environment line of whatever launched the process.
 */
uint32_t ft_tt_op_gdnc_conv_depth(void);

/** The parser behind it, exposed so `tests/test_gdnc_ops.c` can pin the refusal without an
 *  environment. Returns 0 for anything unusable, including the empty string's opposite — an
 *  unset variable is the compile-time default and is NOT a refusal. */
uint32_t ft_tt_op_gdnc_conv_depth_parse(const char *value);

/**
 * The pages this family would place for one shape AT ONE DEPTH, without reading the
 * environment and without an arena — the off-hardware price of a rung.
 *
 * It exists because "does depth 8 fit" is a question about `ft_tt_op_cb_place()` and a
 * 330-page arena, and answering it on silicon costs a model load and a program directory built
 * at that depth. `out_pages` is indexed the way `ft_tt_op_gdnc_place_cbs()` indexes it.
 *
 * @return FT_TT_OK, or FT_TT_EINVAL for a shape or a depth this family does not serve
 */
int ft_tt_op_gdnc_cb_pages_at_depth(uint32_t taps, uint32_t state_tiles, uint32_t depth,
                                    uint32_t out_pages[FT_TT_OP_GDNC_CB_COUNT]);

/** Qwen3.5 (Ornith 9B), read off Pantheon's `pn_qwen35moe_config` — not assumed. */
#define FT_TT_OP_GDNC_QWEN35_CONV_TAPS 4u
#define FT_TT_OP_GDNC_QWEN35_CONV_DIM  8192u
/** The GDN state's own shape, which is NOT the attention head's: `ssm_state` is 128, so
 *  `state_tiles` is 4, and `ssm_dt_rank`/`ssm_n_group` are 32 v-heads over 16 k-heads. The
 *  256 that appears elsewhere in this backend is `key_length`, the ATTENTION head dim. */
#define FT_TT_OP_GDNC_QWEN35_STATE_TILES 4u
#define FT_TT_OP_GDNC_QWEN35_V_HEADS     32u
#define FT_TT_OP_GDNC_QWEN35_K_HEADS     16u

/** One chunkwise GDN call. */
typedef struct ft_tt_op_gdnc_request {
    uint32_t phase;      /**< FT_TT_OP_GDNC_PHASE_* */
    uint32_t chunk_rows; /**< live positions in this chunk: 1..32 (CONV floors at conv_taps-1) */
    uint32_t conv_dim;   /**< PHASE CONV: a multiple of 32 */
    uint32_t conv_taps;  /**< PHASE CONV: 2..8; the window is taps-1 deep */
    /** The slice THIS descriptor owns, for a split across a group of gdnc residencies. A
     *  `count` of 0 means the whole stream from 0 and is what an unsliced caller leaves it at,
     *  so an unsliced descriptor encodes a zero word. PHASE CONV slices TILE COLUMNS and PHASE
     *  DELTA slices V-HEADS; both divide exactly, and the decode family measured its DELTA at
     *  `59.8 us + N x 103.4` over six points, i.e. embarrassingly parallel in the head. */
    uint32_t first_col;
    uint32_t col_count;
    /**
     * PHASE DELTA ONLY: the first ROW of the 32-row strip this descriptor owns, so the two
     * mixers can run at a SHORTER chunk length than the pass without a row shuttle. A multiple
     * of 16, with `first_row + chunk_rows <= 32`; 0 is the whole-strip descriptor that shipped
     * and encodes byte-identically to one that predates the field.
     *
     * C = 16 is the rung: PHASE DELTA owns 100% of the chunkwise arm's variance and the chunk
     * length is the whole of it. `kGdncFirstRow` states why in full, including why PHASE CONV
     * neither needs nor accepts it.
     */
    uint32_t first_row;
    /* PHASE DELTA's shape. `state_tiles` is S/32. */
    uint32_t state_tiles;
    uint32_t v_heads;
    uint32_t k_heads; /**< must divide v_heads; the repeat is INTERLEAVED, `vh % k_heads` */
    ft_tt_op_operand activation_in;  /**< CONV: the chunk strip. DELTA: the q|k|v strip */
    ft_tt_op_operand activation_out; /**< must share a bank with activation_in */
    ft_tt_op_operand constant;       /**< CONV: weight planes then band. DELTA: constants then
                                      *   one decay-ratio tile per v-head */
    ft_tt_op_operand state_conv;     /**< CONV: the rolling window. DELTA: the [v_heads,S,S]
                                      *   recurrent state. Read AND written either way */
    /** PHASE DELTA: the flat fp32 gate block. It must sit in the CONSTANT bank — it shares
     *  that operand's packed NoC coordinate word — and nothing on the device can catch a
     *  violation, for the same reason the activation pair cannot be split. */
    uint32_t gate_address;
    /** PHASE DELTA variant bits into the flags word (FT_TT_OP_GDNC_DELTA_FLAG_*); 0 shipped. */
    uint32_t delta_flags;
    /** PHASE DELTA under FT_TT_OP_GDNC_DELTA_FLAG_TABLE: the alpha/beta strips' band (and
     *  `gate_address` is the layer's table block). Must be 0 without the flag. */
    uint32_t gate_band;
    /** PM 8: set FT_TT_OP_GDNC_FLAG_SERIALIZE (either phase). */
    bool serialize;
} ft_tt_op_gdnc_request;

/** PM 8: mirrors kGdncFlagSerialize — either phase. The reader does not start this descriptor
 *  until the previous one on its core has PUBLISHED completion (its writes landed), so a
 *  host may queue several descriptors that carry state through DRAM (the CONV window, the
 *  DELTA state) without awaiting each. */
#define FT_TT_OP_GDNC_FLAG_SERIALIZE 0x200u

/** L11b: mirrors kGdncDeltaFlagCompact — the gate block at `gate_address` is COMPACT
 *  (kernels/ftgc_compact.h) and the reader builds the decay tiles and gate runs. */
#define FT_TT_OP_GDNC_DELTA_FLAG_COMPACT 0x100u

/** Mirrors kGdncDeltaFlagTable — DELTA only, implies COMPACT: `gate_address` is the layer's
 *  TABLE BLOCK (kernels/ftgc_table.h) and the reader looks the gates up from the strips. */
#define FT_TT_OP_GDNC_DELTA_FLAG_TABLE 0x400u


/**
 * Is this a shape the triad can execute? The same predicate the kernels apply, so all four
 * sides refuse the same descriptors independently.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_gdnc_shape_is_supported(const ft_tt_op_gdnc_request *request);

/**
 * Encode `request` into the sixteen parameter words. Pure — touches no device.
 *
 * @return FT_TT_OK, or FT_TT_EINVAL for a null argument, an unsupported shape, an operand
 *         that is not an fp32 tile, or an activation pair in two different banks.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_gdnc_parameters(const ft_tt_op_gdnc_request *request,
                             uint32_t out_words[FT_TT_OP_GDNC_WORDS]);

/**
 * Place this op's circular buffers in the L1 arena and encode the descriptor blob.
 *
 * @param out_mask  receives the local CB mask the launch message wants
 * @return FT_TT_OK, FT_TT_EINVAL, or FT_TT_ENOMEM when the arena cannot hold them
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_op_gdnc_place_cbs(const ft_tt_op_gdnc_request *request,
                            ft_tt_launch_cb_entry out_entries[FT_TT_OP_GDNC_CB_COUNT],
                            void *buffer, size_t buffer_bytes, uint64_t *out_mask);

/* ================================================================
 * THE BAND MATRICES — the whole of what makes the convolution a matmul
 * ================================================================ */

/**
 * Build the `2*taps` constant band tiles, TILIZED, into `out_tiles`.
 *
 * ---------------------------------------------------------------------------
 * THE DERIVATION, so nobody has to re-do it from the kernel
 * ---------------------------------------------------------------------------
 * Output row `t` of a chunk needs, for weight tap `tau`, the input at position `t - d` where
 * `d = (taps-1) - tau`. Inside the chunk that is row `t-d` of the input tile whenever
 * `t >= d`; below it, it is window plane `p = t + tau`, which the reader has put at ROW `p`
 * of the assembled window tile. Both are LINEAR MAPS ON THE ROW AXIS and neither depends on
 * the channel, the layer, the chunk or `chunk_rows`:
 *
 *     tile 2*tau     S_tau[t][u] = 1  iff  u == t - d  and  t >= d      (against the input)
 *     tile 2*tau + 1 C_tau[t][p] = 1  iff  p == t + tau and  t <  d      (against the window)
 *
 * `C_{taps-1}` is the ZERO TILE — at `tau = taps-1`, `d = 0` and no row qualifies. It is
 * emitted anyway: `2*taps` products put CB_PROD on a clean indexed-read grid at every
 * supported tap count, where `2*taps - 1` is prime at taps = 4, and the uniform loop costs one
 * multiply and one matmul of a zero matrix per column against a special case in three kernels
 * and the host.
 *
 * INDEPENDENT OF `chunk_rows`, which is why this takes only `taps`: rows at or above the
 * chunk produce values nobody reads, and the host owes zeros in the input above the chunk so
 * that what they produce is `SiLU(0)`.
 *
 * @param taps      2..8
 * @param out_tiles at least `ft_tt_op_gdnc_band_bytes(taps)` bytes, tilized fp32
 * @return FT_TT_OK or FT_TT_EINVAL
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (it uses stack scratch only)
 */
int ft_tt_op_gdnc_build_band(uint32_t taps, void *out_tiles, size_t out_bytes);

/** Bytes of the band block: `2*taps` fp32 tiles. */
size_t ft_tt_op_gdnc_band_bytes(uint32_t taps);

/* ================================================================
 * What one layer costs in card DRAM — every operand, one expression each
 * ================================================================ */

/** Bytes of ONE conv stream — the chunk's input or its conv_out. One row block of
 *  `conv_dim/32` fp32 tiles, so the two are the same size AND the same size as the decode
 *  family's, because a 32-row chunk and a 1-row position occupy the same tiles. */
size_t ft_tt_op_gdnc_conv_stream_bytes(const ft_tt_op_gdnc_request *request);

/** Bytes of the CONSTANT buffer one layer owns: `taps` row-replicated tap-major planes of
 *  `conv_dim/32` fp32 tiles, PLUS the `2*taps` band tiles that ride at the end of them. */
size_t ft_tt_op_gdnc_constant_bytes(const ft_tt_op_gdnc_request *request);

/** Bytes of the rolling conv window one layer owns, AS THE READER ADDRESSES IT: `(taps-1)`
 *  tap-major ROW-0 planes of `conv_dim/32` fp32 tiles.
 *
 *  NOT `conv_dim * (taps-1) * 4`. That expression is 1/32 of this one and it is the DRAM
 *  under-allocation the decode family shipped for a round: a plane is a one-row tilized strip
 *  and the tile quantum is 32 rows. This is byte for byte `ft_tt_op_gdn_conv_state_bytes()`,
 *  which is what lets the two families share the buffer. */
size_t ft_tt_op_gdnc_conv_state_bytes(const ft_tt_op_gdnc_request *request);

/* ================================================================
 * PHASE DELTA's operands
 * ================================================================ */

/**
 * Build the four PHASE DELTA PROGRAM CONSTANT tiles, TILIZED, into `out_tiles`.
 *
 * `I`, `2I`, the STRICTLY-LOWER 0/1 mask, and a COLUMN of `scale`. They are identical in every
 * layer, every chunk and every model with this state size, and they ride at the HEAD of the
 * constant buffer with the per-head decay tiles behind them.
 *
 * `scale` is `1/sqrt(S)` and it is computed HERE, once, rather than on the card: a reciprocal
 * square root derived two ways is two answers that must agree about a factor multiplying every
 * output of a recurrence. The decode family sends the same number as raw fp32 bits for the
 * same reason.
 *
 * @param state_size  S — 32*state_tiles
 * @param out_tiles   at least `ft_tt_op_gdnc_delta_constant_tile_bytes()` bytes
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (stack scratch only)
 */
int ft_tt_op_gdnc_build_delta_constants(uint32_t state_size, void *out_tiles, size_t out_bytes);

/** Bytes of the four program-constant tiles. */
size_t ft_tt_op_gdnc_delta_constant_tile_bytes(void);

/**
 * Build ONE v-head's decay-ratio tile `D[t][s] = prod_{r=s+1..t} decay_r` (t >= s), TILIZED.
 *
 * ---------------------------------------------------------------------------
 * A PRODUCT RECURRENCE, AND THAT IS THE WHOLE NUMERICAL ARGUMENT
 * ---------------------------------------------------------------------------
 * `D[t][t] = 1` and `D[t][s] = D[t][s+1] * decay_{s+1}`. Every entry is in (0,1], nothing is
 * ever divided by a cumulative product, and no logarithm appears. The two alternatives were
 * both modelled through this card's Tf32 truncating unpack and both are WORSE:
 *
 *   - building D ON CARD by doubling costs the whole precision advantage of going chunkwise
 *     (y 8.6e-03 -> 1.7e-02, which is where 32 sequential decode steps already are), to save
 *     4 KiB a head on a link that carries 128 KiB a layer a chunk for it;
 *   - crossing the ratios as `exp(log P_t - log P_s)` is safe against OVERFLOW and not against
 *     the UNPACK — cumulative logs reach -50 at Ornith's decays and a Tf32 truncation of -50
 *     is an absolute error of 0.024, i.e. 2.4% in D before the exp is reached.
 *
 * Rows at and above `chunk_rows` are not read by the answer; they are written as the identity's
 * continuation so nothing in the tile is uninitialised.
 *
 * @param decay       `chunk_rows` per-position decays, `exp(g_t)`, in (0,1]
 * @param chunk_rows  1..32
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (stack scratch only)
 */
int ft_tt_op_gdnc_build_decay(const float *decay, uint32_t chunk_rows, void *out_tile,
                              size_t out_bytes);

/**
 * The same tile for the IN-PLACE SUB-CHUNK owning rows `[first_row, first_row + chunk_rows)`.
 * `first_row = 0` is `ft_tt_op_gdnc_build_decay()` and produces a bit-identical tile.
 *
 * TWO THINGS CHANGE AND BOTH ARE LOAD-BEARING:
 *
 *   - **`D[t][s] = 0` FOR EVERY `s` BELOW THE SUB-CHUNK.** `A_off = diag(beta) StrictLower(D o
 *     KK^T)` is what the kernel's Newton doubling inverts, and its powers are what the whole
 *     C-length argument is about. Leaving `D[t][s]` alive for `s < first_row` would leave
 *     `A_off` a 32-WIDE nilpotent whose iterates reach 1.7e+06 — i.e. it would reproduce the
 *     C = 32 failure while every host-side row count said C = 16. **The zeroing is the
 *     mechanism, not tidiness.** It costs nothing on the y path: those columns multiply `U`
 *     rows that `beta = 0` has already made exactly zero.
 *   - **`D[t][.] = 0` FOR EVERY DEAD ROW `t`.** Together with `P_t = 0` from the gate, that
 *     makes `y` row t exactly zero, which is what lets the writer put the whole tile suffix
 *     down without inventing values for rows it does not own.
 *
 * `decay` is `chunk_rows` entries for THIS sub-chunk — the caller slices, so the ratio inside
 * the block is a product over the sub-chunk's own decays and nothing crosses its first row.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (stack scratch only)
 */
int ft_tt_op_gdnc_build_decay_at(const float *decay, uint32_t first_row, uint32_t chunk_rows,
                                 void *out_tile, size_t out_bytes);

/**
 * Build ONE v-head's run of the flat fp32 GATE BLOCK: `P`, `scale*P`, `beta`, `F` and `G`.
 *
 * **THE RAGGED CHUNK IS HANDLED ENTIRELY HERE**, which is why no kernel has a `chunk_rows`
 * branch in PHASE DELTA. `P_t` is the cumulative decay over the LIVE rows only and `G` is
 * `P_{chunk_rows-1}` and NOT `P_31`. Rows at and above `chunk_rows` get `beta = 0` and
 * `F = 1`, so they contribute nothing to `U` and nothing to the carried state — which is the
 * same statement as "the host owes zeros above the chunk in the activation".
 *
 * **`F` IS A BACKWARD PRODUCT RECURRENCE AND NOT `P_{chunk_rows-1} / P_t`.** The ratio is the
 * same number in exact arithmetic and it is a NaN factory in fp32: Ornith's decay is
 * `exp(ssm_a * softplus(alpha))` and a head can carry 1e-7, so `P` underflows to ZERO after
 * about five positions of a 32-position chunk and the ratio becomes `0/0`. Measured on
 * silicon 2026-08-21 — the arm was `cmp` IDENTICAL to the control at 4 rows a pass and
 * produced a completely different completion at 32. `F_{chunk_rows-1} = 1`,
 * `F_t = F_{t+1} * decay_{t+1}`: no denominator, every partial in (0,1], and an underflow
 * gives the zero the mathematics asks for. `P`, `PS` and `G` still underflow and that is
 * CORRECT — a decay that small does annihilate the incoming state.
 *
 * @param decay       `chunk_rows` per-position decays, in (0,1]
 * @param beta        `chunk_rows` per-position betas
 * @param state_size  S, for the `1/sqrt(S)` that rides on the `PS` run
 * @param out_floats  at least `FT_TT_OP_GDNC_GATE_STRIDE` floats
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_gdnc_build_gate(const float *decay, const float *beta, uint32_t chunk_rows,
                             uint32_t state_size, float *out_floats, size_t out_count);

/**
 * The same run for the IN-PLACE SUB-CHUNK owning rows `[first_row, first_row + chunk_rows)`.
 * `first_row = 0` is `ft_tt_op_gdnc_build_gate()` and produces a bit-identical run.
 *
 * A DEAD ROW IS NOT A RAGGED ROW AND THE DIFFERENCE IS `P`. The ragged tail above a short
 * chunk gets `P_t = P_{chunk_rows-1}` frozen, because nothing reads it; a dead row of a
 * sub-chunk gets **`P_t = 0`**, because `Y`'s first term is `scale * diag(P) Q N_0^T` and the
 * writer lands `y` rows that this descriptor does not own. Zero there is what makes those rows
 * harmless; the frozen value would put a whole extra chunk's worth of `Q N_0^T` into them.
 *
 * `beta_t = 0` off the sub-chunk is the same rule the ragged tail uses and it does the same
 * work — row t of `A` is `e_t`, so row t of `A^-1` is `e_t`, so `U` row t is exactly zero and
 * the carried state takes nothing from a position this descriptor does not own. `F` is 1
 * there for the same reason it is above a ragged chunk.
 *
 * `decay` and `beta` are `chunk_rows` entries for THIS sub-chunk. `G` is therefore the product
 * over the sub-chunk's own rows, which is exactly what the state carry `N_C = G N_0 + ...`
 * wants when `N_0` is the state the PREVIOUS sub-chunk left.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_gdnc_build_gate_at(const float *decay, const float *beta, uint32_t first_row,
                                uint32_t chunk_rows, uint32_t state_size, float *out_floats,
                                size_t out_count);

/** Floats one v-head occupies in the gate block. Mirrors `kGdncGateStride`. */
#define FT_TT_OP_GDNC_GATE_STRIDE 160u

/** Bytes of the whole gate block a layer's descriptor reads: `v_heads` runs. The reader pulls
 *  a head's run with ONE NoC transaction of exactly `FT_TT_OP_GDNC_GATE_STRIDE * 4` = 640
 *  bytes, a multiple of the 64-byte granule, so no rounding up is needed and no read leaves
 *  the head's own slot. */
size_t ft_tt_op_gdnc_gate_bytes(const ft_tt_op_gdnc_request *request);

/** Bytes of the PHASE DELTA constant buffer one layer owns: the four program constants and one
 *  decay-ratio tile per v-head. */
size_t ft_tt_op_gdnc_delta_constant_bytes(const ft_tt_op_gdnc_request *request);

/** Bytes of the staged `q|k|v` input, of the y output, and of the recurrent state one layer
 *  owns. The state is byte for byte `ft_tt_op_gdn_state_recurrent_bytes()` — which is the
 *  point: a prompt that prefills in chunks and then decodes is ONE sequence with no
 *  transcription at the seam. */
size_t ft_tt_op_gdnc_delta_input_bytes(const ft_tt_op_gdnc_request *request);
size_t ft_tt_op_gdnc_delta_output_bytes(const ft_tt_op_gdnc_request *request);
size_t ft_tt_op_gdnc_delta_state_bytes(const ft_tt_op_gdnc_request *request);

#endif /* FT_TT_OP_GDNC_H */
