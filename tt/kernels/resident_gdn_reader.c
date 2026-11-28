/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_gdn_reader.c
 * @brief Reader third of the RESIDENT GATED-DELTA-NET triad, and THE PROCESSOR THAT BUILDS
 *   EVERY BROADCAST SHAPE THE COMPUTE SURFACE CANNOT EXPRESS.
 *
 * The compute thread of this op never indexes an element. It is handed aligned tile streams
 * and issues broadcast multiplies, row reduces, subs and adds. Everything that makes those
 * legal -- the row-replicated k/q, the column-shaped v, the constant decay/beta/scale
 * columns, the GQA repeat -- is ADDRESSING and L1 word copies, and it all lives here.
 *
 * ===========================================================================
 * THE STAGING CONTRACT, AND THE ONE CORRECTION THAT SIMPLIFIES IT
 * ===========================================================================
 * The broadcast surface is BROADCAST_COL only: it reads column 0 of the second operand and
 * applies that row's scalar across every column of the first operand's row. There is no row
 * broadcast and no scalar broadcast. So:
 *
 *   k, q   ROW-REPLICATED  -- [r][i] = k[i] for EVERY row r.
 *   v      a TRUE COLUMN   -- v[j] at [j][0].
 *   decay  a CONSTANT COLUMN -- the same scalar in all 32 rows of column 0.
 *   beta   likewise.
 *   scale  likewise, splatted from parameters[kGdnScaleBits] WITHOUT arithmetic.
 *
 * ONE TILE SERVES BOTH USES OF k. The silicon probe that proved the reduce scaler is
 * PER-POSITION built its scaler with tests/test_ops.c `build_column_scaler_tile()`, which
 * writes `plain[row*32 + column] = s[column]` -- i.e. the value for reduce position c is
 * present at EVERY [r][c]. The proven staging is therefore ROW-REPLICATED, not "the value at
 * [i][0]". A row-replicated k tile is simultaneously the (b) reduce's scaler operand and the
 * (d) rank-1 operand, so FT_TT_OP_GDN_CB_SCALER absorbs what the paper design called
 * CB_REPLICA and index 4 is free for something else (see the CB map below).
 * ft_tt_op_gdn.h now DOCUMENTS that collapse rather than contradicting it.
 *
 * THE NON-COLUMN-0 LANES OF v / decay / beta / scale ARE DELIBERATELY NOT WRITTEN. Only
 * column 0 of any of them is ever consumed: `bcast_col` reads column 0, and the only full
 * eltwise op they reach is `(c)`'s `v - pred`, whose result is itself consumed by a
 * `bcast_col`. Zeroing 1,024 words a tile to make the other 31 columns tidy would roughly
 * double this reader's L1 store count for no observable difference. If a future step ever
 * consumes a lane other than column 0, this is the line that has to change.
 *
 * ===========================================================================
 * THE GQA REPEAT IS NEVER MATERIALISED
 * ===========================================================================
 * `ftg_k_head_for(h, k_heads)` is one modulo. v-head h stages the k/q columns of k-head
 * `h % k_heads`; the two v-heads that share a k-head rebuild the same replica tiles rather
 * than a doubled tensor being read from DRAM. Materialising it is the 1.33x-WORSE row of the
 * split-point table.
 *
 * ===========================================================================
 * HOW A REPLICA IS BUILT: BULK READ, THEN PURE L1 WORD COPIES
 * ===========================================================================
 * The NoC cannot gather with stride, so this is resident_attention_reader.c's
 * `read_and_pack_q()` idiom: ONE bulk `noc_read` of the source row block into a scratch CB
 * that is reserved but NEVER pushed, then face-arithmetic word copies into the consumed CB's
 * pages. A 32x32 Float32 tile is 4096 B: four 1024 B faces in row-major order of the 2x2
 * face grid, sixteen 64 B rows inside each face.
 *
 * The cost is honest and worth stating: a replica tile is 1,024 word stores because every
 * word of it is written, and one DELTA descriptor builds `2 * v_heads * state_tiles` of them.
 * At the shipped shape that is 256 tiles, ~262 K word stores a layer. Against the 0.225 s a
 * host recurrence costs it is noise; against the rest of this kernel it is the dominant term,
 * and the named optimisation is to cache the staged k/q row block across the `v_heads /
 * k_heads` heads that share it (deliberately not taken here -- it needs `2 * state_tiles`
 * scratch pages held live, which is the whole scratch buffer).
 *
 * ===========================================================================
 * THE CB MAP
 * ===========================================================================
 *   cb 0  CB_IN     CONV ONLY: the qkv stream. THIS PROCESSOR IS ITS ONLY PRODUCER, in
 *                   either phase. PHASE DELTA's compute-private working state used to share
 *                   this index; it is cb 8 now. "One producer per DESCRIPTOR" was the rule
 *                   this file used to state and it is not strong enough -- a CB's books
 *                   outlive a descriptor, and the dataflow push INCREMENTS the
 *                   tiles_received register while the compute push STORES a local counter
 *                   over it. See the compute kernel's cb_work note.
 *   cb 1  CB_STATE  CONV: the rolling window.  DELTA: one tile-row of the carried state.
 *   cb 2  CB_SCALER DELTA: this head's k replicas then its q replicas, `2*state_tiles`.
 *   cb 3  CB_VECTOR CONV: the conv1d weight taps.  DELTA: v columns, then decay, beta, scale.
 *   cb 4  CB_STAGE  READER-PRIVATE SCRATCH: never reserved, never pushed and never popped,
 *                   by this processor or any other. That is exactly why it is safe to write
 *                   through its write pointer -- the pointer can never have moved off the
 *                   buffer base. THE ROUND'S SPEC NAMED cb 7 FOR THIS AND THAT IS WRONG:
 *                   cb 7 is CB_PROD, which the compute thread pushes and pops inside the
 *                   same descriptor, so the two would fight over the same L1 pages.
 *
 * THIS PROCESSOR OWNS `tail`, AND NOTHING ELSE IN THE CONTROL BLOCK
 * (resident_queue_protocol.h rule 1). An aborted descriptor advances nothing.
 *
 * ONE BARRIER PER STREAM GROUP, NEVER PER TILE. A barrier is a per-CORE drain, and paying it
 * per tile cost 5.9x on the matmul. CONV cannot put all of a block's reads under ONE barrier
 * because the window's `taps-1` tiles cannot be reserved as one chunk -- `taps-1` = 3 does
 * not divide the 8-page window buffer, and a chunk that does not divide the depth walks the
 * write pointer off the end (ft_tensix_cb_push_grid_is_safe). So each window tap is its own
 * reserve/read/barrier/push group and the qkv+weights are one more: `taps` groups a block,
 * not `2*taps` transactions each with a barrier.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` -- no tt-metal include (plan Phase 6.6).
 * Compile-time args:
 *   0: CB_IN index   1: CB_STATE index   2: CB_SCALER index   3: CB_VECTOR index
 *   4: CB_STAGE index (reader-private staging; never pushed, never popped)
 *   5: tile size in bytes   6: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 *   7: CONV block depth, SHARED with the compute kernel and with the host's
 *      FT_TT_OP_GDN_CONV_DEPTH. EIGHT args, not seven, and the count is load-bearing:
 *      `ft_tensix_compile_time_arg(7)` against a seven-arg line reads one word PAST the
 *      args array, and SFPI GCC 15.1.0 FOLDS THAT TO A LITERAL ZERO. Measured on a second card host
 *      2026-08-18: a seven-arg build of this file is BYTE-IDENTICAL to one passing an
 *      explicit depth of 0, at -Os and at -O3, with no diagnostic even under
 *      `-Warray-bounds=2` -- so `-Werror` would not have caught it either, and the arg
 *      COUNT written beside this build line is the only guard that exists. Depth 0 makes
 *      conv_block_depth() return 0, which aborts every CONV descriptor with
 *      kErrBadParameters.
 *   8: CB_WORK index -- the COMPUTE thread's private PHASE DELTA working buffer. This
 *      processor never touches it; the index is here so `delta_is_executable()` can refuse a
 *      placement the compute thread's indexed reads would walk off, which is the same
 *      three-sided refusal every other buffer gets. NINE args.
 *   9: CB_PROD index -- the compute thread's rank-1 staging buffer, here for exactly the
 *      reason CB_WORK is. The batched (d) indexes it `batch` tiles wide instead of naming
 *      tile 0, so it acquired an indexed-read grid, and this processor is the only one of
 *      the three that can read a page count at all.
 *  10: dst_batch_max -- PHASE DELTA's DST batch width before `ftg_dst_batch()` clamps it.
 *      SHARED with the compute kernel: the grid refused here is a function of it, so a pair
 *      that disagreed would refuse the wrong placements and admit the ones that matter.
 *      ELEVEN args.
 *  11: reader extras, a bitmask -- see READER_EXTRAS. TWELVE args.
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build (the whole program; this file is the BRISC/NoC 0 third):
 *   RELOCATE_KERNEL_TEXT=1 \
 *     READER_CTARGS=0,1,2,3,4,4096,1,2,8,7,4,1 COMPUTE_CTARGS=0,1,2,3,5,6,7,16,1,2,8,4,1 \
 *     WRITER_CTARGS=16,4096,1,0 CB_TABLE=kernels/cb_tables/gdn_float32.cbtable \
 *     bash foundry/tt/build_tensix_program.sh \
 *       kernels/resident_gdn_reader.c kernels/resident_gdn_compute.c \
 *       kernels/resident_gdn_writer.c programs/gdn
 *
 * The round's spec wrote `READER_CTARGS=0,1,2,3,7,4096,1`. The 7 is corrected to 4 above for
 * the reason in the CB map: index 7 is the compute thread's CB_PROD.
 *
 * @status EXECUTED ON SILICON 2026-08-18, and every address this file computes is CONFIRMED
 *   by readback rather than inferred: the host echoes the staged qkv row and all four conv
 *   weight tap planes out of DRAM with 0 of 8192 channels differing, and PASS B's newest
 *   window plane comes back equal to qkv tile for tile, which exercises the whole reader ->
 *   CB -> compute -> writer path on Float32 data with an answer that needs no arithmetic.
 * @issues None. The DELTA input and CONV operand layouts were an INFERENCE living in this
 *   file's private address arithmetic; they are now STATED in resident_gdn_protocol.h, and
 *   every offset this file computes comes from the `ftg_*` accessors there, so the stager
 *   and the reader read one declaration. There is still no stager and no device check can
 *   catch a disagreement — `tests/test_gdn_ops.c` pins the arithmetic off-hardware instead.
 *
 *   THE PUSH GRID WAS ONLY HALF THE RULE, and the missing half was this family's whole wrong
 *   answer (measured 2026-08-18). A consumer that indexes `n` tiles off its front needs
 *   `pages % n == 0` as well, and a RESIDENT op needs its pointers put back on the base at
 *   every descriptor because a descriptor's traffic is not a whole number of buffers.
 *   `index_grid_is_safe()` refuses the first and `rebase_produced_buffers()` supplies the
 *   second; see the section in resident_gdn_protocol.h.
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_gdn_protocol.h"
/* Direct NoC calls, not ftgc_table_io's pointers (ftgc_table.h says why) — through ONE
 * out-of-line read: six inlined copies do not fit this reader's text slab. */
static __attribute__((noinline)) void gdn_table_read(uint32_t xy, uint32_t address,
                                                      uint32_t l1, uint32_t bytes)
{
    ft_tensix_noc_read(ft_tensix_noc_address(ftg_xy_x(xy), ftg_xy_y(xy), address), l1, bytes);
}
#define FTGT_READ(io, xy, address, l1, bytes) gdn_table_read((xy), (address), (l1), (bytes))
#define FTGT_BARRIER(io) ft_tensix_noc_read_barrier()
#include "ftgc_table.h"
#include "resident_queue_protocol.h"
#include "resident_precondition.h"

#define CB_IN ft_tensix_compile_time_arg(0)
#define CB_STATE ft_tensix_compile_time_arg(1)
#define CB_SCALER ft_tensix_compile_time_arg(2)
#define CB_VECTOR ft_tensix_compile_time_arg(3)
#define CB_STAGE ft_tensix_compile_time_arg(4)
#define TILE_BYTES ft_tensix_compile_time_arg(5)
#define WAIT_MODE ft_tensix_compile_time_arg(6)
/** Tile columns one CONV block covers. SHARED with the compute kernel, which is built with
 *  the same value and cannot derive it any other way — `cb_interface` does not exist on the
 *  MATH thread, so a depth read off the CB interface is an undefined reference there. Same
 *  two-sided compile-time contract TILE_BYTES has, and nothing checks it for you. */
#define CONV_DEPTH ft_tensix_compile_time_arg(7)
/** The COMPUTE thread's private PHASE DELTA working buffer. Named here ONLY so this
 *  processor's refusal covers it; nothing below reads or writes it. */
#define CB_WORK ft_tensix_compile_time_arg(8)
/** The COMPUTE thread's rank-1 staging buffer. Named here for the same reason CB_WORK is:
 *  the batched (d) indexes it, so it has a consumer grid, and this processor is the only one
 *  of the three that can see a page count. */
#define CB_PROD ft_tensix_compile_time_arg(9)
/** PHASE DELTA's DST batch width before the clamp. SHARED with the compute kernel; the grid
 *  refused below is a function of it. */
#define DST_BATCH_MAX ft_tensix_compile_time_arg(10)
/** Reader restructurings, as a BITMASK. Every bit is a measurement arm, and each produces
 *  BYTE-IDENTICAL circular-buffer contents to the arm below it -- what moves is how many
 *  scalar L1 accesses this processor makes to produce them, which is the term the DELTA
 *  descriptor may or may not be bound by. TWELVE args.
 *
 *  bit 0 BUILD A ROW-REPLICATED TILE COLUMN-MAJOR. `push_replicated_tile` copies one 64-byte
 *        source row into all sixteen rows of each of four faces, and it does it row by row,
 *        re-reading the source every time -- 1,024 loads and 1,024 stores a tile, times
 *        `2 * v_heads * state_tiles` = 256 tiles a DELTA descriptor, which is ~262,000 word
 *        copies. Loading each source word once and storing it down the column drops the loads
 *        to 64 a tile. */
#define READER_EXTRAS ft_tensix_compile_time_arg(11)

enum { kReaderReplicaByColumn = 1 };

/** Rows of one face, and faces of one tile: fixed by gdn_float32.cbtable (face_r 16,
 *  num_faces 4). Named because a bare 16 in an offset expression is unreadable. */
enum { kGdnFaceRows = 16 };
enum { kGdnTileRows = 32 };

/** The five DRAM endpoints a descriptor names, decoded once. */
typedef struct gdn_endpoints {
    uint32_t activation_x;
    uint32_t activation_y;
    uint32_t activation_in;
    uint32_t constant_x;
    uint32_t constant_y;
    uint32_t constant_base;
    uint32_t state_x;
    uint32_t state_y;
    uint32_t state_base;
} gdn_endpoints;

/* M5 GDN READER CLOCKS (control words 57-60, this processor's alone; zeroed at boot):
 *   57 cycles >> 4 waiting in NoC read barriers, 58 building tiles in software (replicate /
 *   column / constant), 59 whole CONV descriptors, 60 whole DELTA descriptors. */
enum { kGdnClockBarrier = 57, kGdnClockBuild = 58, kGdnClockConv = 59, kGdnClockDelta = 60 };
/* M9: PHASE CONV's barriers go to their own word, so 57 is DELTA's alone. Word 12 (free on a
 * gdn core: the matmul writer's busy clock, cleared by the host's queue init). NOT 56, which
 * it was until S1: that is kFieldGroupDone, the done-sequence a member 0 pull reads, and a
 * clock added into it satisfied the pull before the member had finished. */
enum { kGdnClockConvBarrier = 12 };
static bool gdn_conv_phase;
static bool gdn_conv_strip;
/** S2c: the row this descriptor's stream rows live in (kGdnRowShift). */
static uint32_t gdn_row;
static bool gdn_red_strip_half;
/** RED ONLY (kGdnFlagRedRowsRow0): a multi-row CONV convolves row 0's stream for every row. */
static bool gdn_red_rows_row0;
/** G2a (kGdnFlagConvBf16In): the CONV stream is the bf16 projection strip. Its rows land in
 *  the half of each 64 B face row at the source's DRAM-window lane and are widened in place
 *  after the group's barrier. */
static bool gdn_stream_bf16;
static uint32_t gdn_widen_destination;
static uint32_t gdn_widen_tiles;
static uint32_t gdn_widen_lane;
static volatile FTQ_L1_PTR uint32_t *gdn_clock_control;

static FT_BH_INLINE void gdn_clock_add(uint32_t word, uint32_t started)
{
    if (gdn_clock_control)
        gdn_clock_control[word] += (ft_bh_wall_clock_low() - started) >> 4;
}

static void conv_widen_pending(void);

static FT_BH_INLINE void gdn_read_barrier(void)
{
    const uint32_t started = ft_bh_wall_clock_low();

    ft_tensix_noc_read_barrier();
    gdn_clock_add(gdn_conv_phase ? kGdnClockConvBarrier : kGdnClockBarrier, started);
    if (gdn_widen_tiles)
        conv_widen_pending();
}

static void load_endpoints(gdn_endpoints *endpoints, const ftg_shape *shape,
                           volatile FTQ_L1_PTR uint32_t *parameters)
{
    endpoints->activation_x = ftg_xy_x(parameters[kGdnActivationXy]);
    endpoints->activation_y = ftg_xy_y(parameters[kGdnActivationXy]);
    endpoints->activation_in = parameters[kGdnActivationIn];
    endpoints->constant_x = ftg_xy_x(parameters[kGdnConstantXy]);
    endpoints->constant_y = ftg_xy_y(parameters[kGdnConstantXy]);
    endpoints->state_x = ftg_xy_x(parameters[kGdnStateXy]);
    endpoints->state_y = ftg_xy_y(parameters[kGdnStateXy]);
    if (shape->phase == (uint32_t)kGdnPhaseConv && gdn_stream_bf16) {
        endpoints->activation_x = ftg_xy_x(parameters[kGdnConstantGate]);
        endpoints->activation_y = ftg_xy_y(parameters[kGdnConstantGate]);
    }
    if (shape->phase == (uint32_t)kGdnPhaseConv) {
        endpoints->constant_base = parameters[kGdnConstantWeight];
        endpoints->state_base = parameters[kGdnStateConv];
    } else {
        endpoints->constant_base = parameters[kGdnConstantGate];
        endpoints->state_base = parameters[kGdnStateRecurrent];
    }
}

/* ===========================================================================
 * Face arithmetic. Modelled on fta_chunk_offset(); spelled locally because
 * resident_gdn_protocol.h is the wire contract and does not carry L1 layout.
 * =========================================================================== */

/** Bytes of one face: a 2x2 face grid over the tile. */
static uint32_t ftg_face_bytes(void)
{
    return TILE_BYTES / 4u;
}

/** Bytes of one face ROW: 16 elements of a 32-element row, i.e. tile_bytes/64. */
static uint32_t ftg_face_row_bytes(void)
{
    return TILE_BYTES / 64u;
}

/** Words of one face row -- the unit every copy below moves. 16 for a Float32 tile. */
static uint32_t ftg_face_row_words(void)
{
    return TILE_BYTES / 256u;
}

/** Byte offset, inside a tile, of the face row holding `row` and columns [16*half, +16). */
static uint32_t ftg_face_row_offset(uint32_t row, uint32_t half)
{
    return ((row / (uint32_t)kGdnFaceRows) * 2u + half) * ftg_face_bytes() +
           (row % (uint32_t)kGdnFaceRows) * ftg_face_row_bytes();
}

/** Byte offset, inside a tile, of element [row][column]. */
static uint32_t ftg_element_offset(uint32_t row, uint32_t column)
{
    return ftg_face_row_offset(row, column / (uint32_t)kGdnFaceRows) +
           (column % (uint32_t)kGdnFaceRows) * (TILE_BYTES / 1024u);
}

/**
 * M9: `tiles` consecutive CONV tiles from (x, y, address) into L1 at `destination` — whole, or
 * under kGdnFlagConvStrip only row 0 (faces 0-1: 2 x 64 B a tile). Barrier is the caller's.
 */
static void conv_read(uint32_t x, uint32_t y, uint32_t address, uint32_t destination,
                      uint32_t tiles)
{
    uint32_t tile;
    uint32_t half;

    if (!gdn_conv_strip) {
        ft_tensix_noc_read(ft_tensix_noc_address(x, y, address), destination,
                           tiles * TILE_BYTES);
        return;
    }
    for (tile = 0; tile < tiles; tile++)
        for (half = 0; half < (gdn_red_strip_half ? 1u : 2u); half++) {
            const uint32_t offset = tile * TILE_BYTES + ftg_face_row_offset(0u, half);

            ft_tensix_noc_read(ft_tensix_noc_address(x, y, address + offset),
                               destination + offset, ftg_face_row_bytes());
        }
}

/** Byte offset, inside a bf16 tile, of the face row holding `row`, columns [16*half, +16). */
static uint32_t ftg_bf16_face_row_offset(uint32_t row, uint32_t half)
{
    return ((row / (uint32_t)kGdnFaceRows) * 2u + half) * ((uint32_t)kGdnBf16TileBytes / 4u) +
           (row % (uint32_t)kGdnFaceRows) * ((uint32_t)kGdnBf16TileBytes / 64u);
}

/**
 * G2a: row `row` of `tiles` bf16 stream tiles into L1 row 0 of fp32 pages at `destination`:
 * each 32 B face-row half lands in the half of its 64 B fp32 face row at the SOURCE's offset
 * in the 64 B DRAM-read window (ftg_bf16_half_lane: the two ends of a NoC DRAM read must be
 * congruent mod 64), and the caller's barrier widens it in place (conv_widen_pending).
 * `address` is the bf16 strip's tile. Every half of one call shares the lane: the tile
 * (2048 B) and face (512 B) strides are multiples of 64.
 */
static void conv_read_bf16_row(uint32_t x, uint32_t y, uint32_t address, uint32_t destination,
                               uint32_t tiles, uint32_t row)
{
    const uint32_t half_bytes = (uint32_t)kGdnBf16TileBytes / 64u;
    const uint32_t lane = ftg_bf16_half_lane(address + ftg_bf16_face_row_offset(row, 0u));
    uint32_t tile, half;

    for (tile = 0; tile < tiles; tile++)
        for (half = 0; half < (gdn_red_strip_half ? 1u : 2u); half++)
            ft_tensix_noc_read(ft_tensix_noc_address(x, y, address +
                                                               tile * (uint32_t)kGdnBf16TileBytes +
                                                               ftg_bf16_face_row_offset(row, half)),
                               destination + tile * TILE_BYTES + ftg_face_row_offset(0u, half) +
                                   lane,
                               half_bytes);
    gdn_widen_destination = destination;
    gdn_widen_tiles = tiles;
    gdn_widen_lane = lane;
}

/** G2a, after the barrier: widen every staged half of the pending pages in place. */
static void conv_widen_pending(void)
{
    uint32_t tile, half;

    for (tile = 0; tile < gdn_widen_tiles; tile++)
        for (half = 0; half < (gdn_red_strip_half ? 1u : 2u); half++)
            ftg_widen_face_row_at((volatile FTQ_L1_PTR uint32_t *)(gdn_widen_destination +
                                                                   tile * TILE_BYTES +
                                                                   ftg_face_row_offset(0u, half)),
                                  gdn_widen_lane);
    gdn_widen_tiles = 0u;
}

/** The stream's tile `column` in DRAM: fp32 `stream_in`, or the G2a bf16 projection strip. */
static uint32_t conv_stream_tile(const gdn_endpoints *endpoints, uint32_t column)
{
    return endpoints->activation_in +
           column * (gdn_stream_bf16 ? (uint32_t)kGdnBf16TileBytes : TILE_BYTES);
}

/** S2c: conv_read for the STREAM at DRAM row `row` into L1 row 0 (the strip-mode layout the
 *  compute thread reads). Row 0 is conv_read() itself. */
static void conv_read_stream_row(uint32_t x, uint32_t y, uint32_t address, uint32_t destination,
                                 uint32_t tiles, uint32_t row)
{
    uint32_t tile;
    uint32_t half;

    if (gdn_stream_bf16) {
        conv_read_bf16_row(x, y, address, destination, tiles, row);
        return;
    }
    if (row == 0u || !gdn_conv_strip) {
        conv_read(x, y, address, destination, tiles);
        return;
    }
    for (tile = 0; tile < tiles; tile++)
        for (half = 0; half < (gdn_red_strip_half ? 1u : 2u); half++)
            ft_tensix_noc_read(ft_tensix_noc_address(x, y, address + tile * TILE_BYTES +
                                                               ftg_face_row_offset(row, half)),
                               destination + tile * TILE_BYTES + ftg_face_row_offset(0u, half),
                               ftg_face_row_bytes());
}

/** The row select's stream read: DRAM row `gdn_row`. */
static void conv_read_stream(uint32_t x, uint32_t y, uint32_t address, uint32_t destination,
                             uint32_t tiles)
{
    conv_read_stream_row(x, y, address, destination, tiles, gdn_row);
}

/** A word copy inside L1. L1 is ordinary memory to the RISC-V; both ends are 4-byte aligned
 *  by construction, since every offset above is a multiple of the element size. */
static void ftg_copy_words(uint32_t *destination, const uint32_t *source, uint32_t words)
{
    uint32_t index;

    for (index = 0; index < words; index++)
        destination[index] = source[index];
}

/* ===========================================================================
 * PHASE DELTA -- staging
 * =========================================================================== */

/** Pull `tiles` consecutive activation tiles into the scratch buffer. ONE transaction. */
static void stage_activation_tiles(ftq_context *ctx, const gdn_endpoints *endpoints,
                                   uint32_t scratch, uint32_t first_tile, uint32_t tiles)
{
    if (ctx->aborting)
        return;
    ft_tensix_noc_read(ft_tensix_noc_address(endpoints->activation_x, endpoints->activation_y,
                                             endpoints->activation_in + first_tile * TILE_BYTES),
                       scratch, tiles * TILE_BYTES);
    gdn_read_barrier();
}

/**
 * One ROW-REPLICATED tile into CB_SCALER: [r][i] = source_row[i] for every row r.
 *
 * Serves BOTH the (b)/(e) reduce scaler and the (d) rank-1 update -- see the file header on
 * why that is one tile rather than two.
 */
/* M6: build replicated tiles with the NoC instead of 1,024 RISC-V stores (kGdnFlagNocReplicate,
 * set per descriptor by the host). Measured 2026-10-05: the software build is 3.56 ms/token of
 * the DELTA reader (~4.6 us a tile, ~6 cycles a store). */
static bool gdn_noc_replicate;
static bool gdn_red_no_copy;
static uint32_t gdn_self_x, gdn_self_y;

/** Row 0 of faces 0 and 1 by RISC-V stores (32 words), then doubling inside each face with
 *  local NoC reads (64, 128, 256, 512 B), then faces 2-3 as one copy of faces 0-1. The bytes
 *  are the store loop's exactly: faces 0/2 repeat the source half for columns 0-15, faces
 *  1/3 the half for 16-31. */
static void build_replicated_by_noc(uint32_t destination, uint32_t source, uint32_t source_row)
{
    const uint32_t face = ftg_face_bytes();
    const uint32_t row = ftg_face_row_bytes();
    const uint32_t words = ftg_face_row_words();
    uint32_t half, span;

    for (half = 0; half < 2u; half++)
        ftg_copy_words((uint32_t *)(uintptr_t)(destination + half * face),
                       (const uint32_t *)(uintptr_t)(source + ftg_face_row_offset(source_row, half)),
                       words);
    for (span = row; span < face; span <<= 1) {
        for (half = 0; half < 2u; half++) {
            const uint32_t base = destination + half * face;

            ft_tensix_noc_read(ft_tensix_noc_address(gdn_self_x, gdn_self_y, base), base + span,
                               span);
        }
        gdn_read_barrier();
    }
    if (!gdn_red_no_copy) { /* RED (kGdnFlagRedNoCopy): faces 2-3 stay stale; must DIFFER */
        ft_tensix_noc_read(ft_tensix_noc_address(gdn_self_x, gdn_self_y, destination),
                           destination + 2u * face, 2u * face);
        gdn_read_barrier();
    }
}

static bool push_replicated_tile(ftq_context *ctx, uint32_t scratch, uint32_t source_row,
                                 uint32_t source_tile)
{
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_SCALER, 1, WAIT_MODE))
        return false;

    if (!ctx->aborting && gdn_noc_replicate) {
        const uint32_t build_started = ft_bh_wall_clock_low();

        build_replicated_by_noc(ft_tensix_cb_write_pointer(CB_SCALER),
                                scratch + source_tile * TILE_BYTES, source_row);
        gdn_clock_add(kGdnClockBuild, build_started);
    } else if (!ctx->aborting) {
        const uint32_t build_started = ft_bh_wall_clock_low();

        const uint32_t source = scratch + source_tile * TILE_BYTES;
        const uint32_t destination = ft_tensix_cb_write_pointer(CB_SCALER);
        const uint32_t words = ftg_face_row_words();
        uint32_t face;

        /* Faces 0 and 2 hold columns 0..15, faces 1 and 3 columns 16..31, so a face takes
         * the source half its column range names and repeats it down all 16 of its rows. */
        for (face = 0; face < 4u; face++) {
            const uint32_t from = source + ftg_face_row_offset(source_row, face & 1u);
            uint32_t *const to =
                (uint32_t *)(uintptr_t)(destination + face * ftg_face_bytes());
            uint32_t line;

            if ((READER_EXTRAS & (uint32_t)kReaderReplicaByColumn) != 0u) {
                /* COLUMN-MAJOR: one source word is loaded ONCE and stored down all sixteen
                 * rows of the face, instead of the whole 64-byte row being re-read for every
                 * row. Same 1,024 stores, SIXTEEN loads instead of 1,024.
                 *
                 * The compiler cannot do this for you and the reason is aliasing: `source`
                 * and `destination` are both plain L1 addresses arrived at through
                 * `uintptr_t`, so a store through `to` may, as far as GCC knows, have landed
                 * on the very word `from` is about to be read from again. It must reload on
                 * every row. Hoisting the load into a local is the only way to say it cannot.
                 *
                 * The stores go out strided by a face row instead of sequentially, which
                 * costs nothing: L1 is SRAM with no line to fill and no prefetcher to
                 * confuse. */
                const uint32_t stride = ftg_face_row_bytes() / 4u;
                uint32_t word;

                for (word = 0; word < words; word++) {
                    const uint32_t value = *(const uint32_t *)(uintptr_t)(from + word * 4u);

                    for (line = 0; line < (uint32_t)kGdnFaceRows; line++)
                        to[line * stride + word] = value;
                }
                continue;
            }
            for (line = 0; line < (uint32_t)kGdnFaceRows; line++)
                ftg_copy_words((uint32_t *)(uintptr_t)(destination + face * ftg_face_bytes() +
                                                       line * ftg_face_row_bytes()),
                               (const uint32_t *)(uintptr_t)from, words);
        }
        gdn_clock_add(kGdnClockBuild, build_started);
    }
    ft_tensix_cb_push_back(CB_SCALER, 1);
    return true;
}

/**
 * One TRUE COLUMN tile into CB_VECTOR: [j][0] = source_row[j], the other lanes untouched.
 *
 * This is a transpose of 32 elements, so it is 32 word copies and not a face-row memcpy.
 */
static bool push_column_tile(ftq_context *ctx, uint32_t scratch, uint32_t source_row,
                             uint32_t source_tile)
{
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_VECTOR, 1, WAIT_MODE))
        return false;

    if (!ctx->aborting) {
        const uint32_t build_started = ft_bh_wall_clock_low();

        const uint32_t source = scratch + source_tile * TILE_BYTES;
        const uint32_t destination = ft_tensix_cb_write_pointer(CB_VECTOR);
        uint32_t j;

        for (j = 0; j < (uint32_t)kGdnTileRows; j++)
            *(uint32_t *)(uintptr_t)(destination + ftg_element_offset(j, 0u)) =
                *(const uint32_t *)(uintptr_t)(source + ftg_element_offset(source_row, j));
        gdn_clock_add(kGdnClockBuild, build_started);
    }
    ft_tensix_cb_push_back(CB_VECTOR, 1);
    return true;
}

/** One CONSTANT COLUMN tile into CB_VECTOR: the same fp32 bit pattern in all 32 rows of
 *  column 0, so `bcast_col` applies a per-head scalar to a whole tile. */
static bool push_constant_column(ftq_context *ctx, uint32_t bits)
{
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_VECTOR, 1, WAIT_MODE))
        return false;

    if (!ctx->aborting) {
        const uint32_t build_started = ft_bh_wall_clock_low();

        const uint32_t destination = ft_tensix_cb_write_pointer(CB_VECTOR);
        uint32_t j;

        for (j = 0; j < (uint32_t)kGdnTileRows; j++)
            *(uint32_t *)(uintptr_t)(destination + ftg_element_offset(j, 0u)) = bits;
        gdn_clock_add(kGdnClockBuild, build_started);
    }
    ft_tensix_cb_push_back(CB_VECTOR, 1);
    return true;
}

/** One tile-row of one head's carried state: `state_tiles` CONSECUTIVE tiles, one read. */
static bool push_state_row(ftq_context *ctx, const gdn_endpoints *endpoints,
                           const ftg_shape *shape, uint32_t head, uint32_t tile_row)
{
    const uint32_t tiles = shape->state_tiles;
    const uint32_t first = ftg_state_tile(shape, head, tile_row, 0u);

    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_STATE, (int32_t)tiles, WAIT_MODE))
        return false;

    if (!ctx->aborting) {
        ft_tensix_noc_read(ft_tensix_noc_address(endpoints->state_x, endpoints->state_y,
                                                 endpoints->state_base + first * TILE_BYTES),
                           ft_tensix_cb_write_pointer(CB_STATE), tiles * TILE_BYTES);
        gdn_read_barrier();
    }
    ft_tensix_cb_push_back(CB_STATE, tiles);
    return true;
}

/**
 * Everything one v-head needs, in the order the compute thread consumes it.
 *
 * CB_SCALER: `state_tiles` k replicas then `state_tiles` q replicas -- the whole buffer, held
 * for the head's every tile-row, because (b) and (d) want k and (e) wants q at every one.
 * CB_VECTOR: `state_tiles` v columns then decay, beta and scale.
 */
static bool read_one_head(ftq_context *ctx, const gdn_endpoints *endpoints,
                          const ftg_shape *shape, uint32_t scratch, uint32_t gate,
                          uint32_t head)
{
    const uint32_t tiles = shape->state_tiles;
    const uint32_t k_head = ftg_k_head_for(head, shape->k_heads);
    uint32_t index;

    /* THE THREE STAGED STREAMS ARE ROW-0 STRIPS, HEAD-MAJOR. A head's S values are
     * `state_tiles` CONSECUTIVE tiles at `ftg_delta_head_tile()` and the row inside each of
     * them is always 0. They used to be one row per head inside a shared row block, which
     * read the SAME `state_tiles` tiles for every head and picked row `head % 32` out of
     * them -- identical read volume, denser DRAM. What the strip buys is that PHASE CONV's
     * output IS this buffer, so nothing crosses PCIe to be re-tilized between the phases. */
    stage_activation_tiles(ctx, endpoints, scratch,
                           ftg_delta_head_tile(shape, (uint32_t)kGdnStreamK, k_head), tiles);
    for (index = 0; index < tiles; index++)
        if (!push_replicated_tile(ctx, scratch, gdn_row, index))
            return false;

    stage_activation_tiles(ctx, endpoints, scratch,
                           ftg_delta_head_tile(shape, (uint32_t)kGdnStreamQ, k_head), tiles);
    for (index = 0; index < tiles; index++)
        if (!push_replicated_tile(ctx, scratch, gdn_row, index))
            return false;

    stage_activation_tiles(ctx, endpoints, scratch,
                           ftg_delta_head_tile(shape, (uint32_t)kGdnStreamV, head), tiles);
    for (index = 0; index < tiles; index++)
        if (!push_column_tile(ctx, scratch, gdn_row, index))
            return false;

    /* decay[h] then beta[h], out of the gate block this descriptor staged once. */
    if (!push_constant_column(ctx, *(const uint32_t *)(uintptr_t)(gate + head * 4u)))
        return false;
    if (!push_constant_column(ctx, *(const uint32_t *)(uintptr_t)(gate +
                                                                  (shape->v_heads + head) * 4u)))
        return false;
    return true;
}

static const ftgc_table_io gdn_table_io = {0, 0u, 0, 0};

/**
 * The gate block(s) for rows [row, row + rows) into `gate`, `stride` bytes a row: read as the
 * host staged it, or (kGdnFlagTable) built from the strips through the host's tables in the
 * staging area below `gate`, which nothing holds before the head loop. A malformed block
 * aborts; the head loop then drains.
 */
static void stage_gate(ftq_context *ctx, const gdn_endpoints *endpoints, const ftg_shape *shape,
                       volatile FTQ_L1_PTR uint32_t *parameters, uint32_t scratch,
                       uint32_t gate, uint32_t stride, uint32_t rows)
{
    if (ctx->aborting)
        return;
    if ((parameters[kGdnFlags] & (uint32_t)kGdnFlagTable) == 0u) {
        ft_tensix_noc_read(ft_tensix_noc_address(endpoints->constant_x, endpoints->constant_y,
                                                 endpoints->constant_base),
                           gate, rows * stride);
        gdn_read_barrier();
        return;
    }
    if (gate - scratch < FTGT_ROWS_BYTES || (scratch % 64u) != 0u ||
        !ftgc_table_rows(&gdn_table_io, scratch, parameters[kGdnConstantXy],
                         endpoints->constant_base, shape->head_first, shape->head_count,
                         shape->row_first, rows, gate, stride, shape->v_heads))
        ftq_abort(ctx, kErrBadParameters);
}

/**
 * The reader half of one DELTA descriptor.
 *
 * The gate block -- `decay[v_heads]` then `beta[v_heads]`, fp32 -- is read ONCE into the
 * scratch page ABOVE the row-block staging area, because it is 2 KiB at the widest shape this
 * binary accepts and re-reading four bytes a head would be 64 sub-alignment transactions.
 */
static bool read_one_delta(ftq_context *ctx, const gdn_endpoints *endpoints,
                           const ftg_shape *shape, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t scratch = ft_tensix_cb_write_pointer(CB_STAGE);
    const uint32_t gate = scratch + shape->state_tiles * TILE_BYTES;
    const uint32_t scale_bits = parameters[kGdnScaleBits];
    /* Rounded UP to the NoC's 64-byte read granule: the gate block is 2*v_heads fp32 words,
     * which is only a multiple of 64 B for an even head count, and an under-aligned length is
     * a refused transaction rather than a short one. 2048 B at the widest shape this binary
     * accepts, so it always fits the one scratch page reserved for it. */
    const uint32_t gate_bytes = (2u * shape->v_heads * 4u + 63u) & ~63u;
    uint32_t head;

    stage_gate(ctx, endpoints, shape, parameters, scratch, gate, gate_bytes, 1u);

    /* THIS MEMBER'S HEADS, not every head. The gate block above is read WHOLE regardless --
     * it is 2*v_heads fp32 words, 256 B at Ornith, and one 64 B-granule read of the whole
     * thing is cheaper than computing a sliced offset that has to stay 64 B-aligned. */
    for (head = shape->head_first; head < shape->head_first + shape->head_count; head++) {
        uint32_t tile_row;

        if (!read_one_head(ctx, endpoints, shape, scratch, gate, head))
            return false;
        if (!push_constant_column(ctx, scale_bits))
            return false;
        for (tile_row = 0; tile_row < shape->state_tiles; tile_row++)
            if (!push_state_row(ctx, endpoints, shape, head, tile_row))
                return false;
    }
    return !ctx->aborting;
}

/* ===========================================================================
 * PHASE DELTA -- MULTI-ROW (PM 5)
 * =========================================================================== */

/** One head's k, q and v row blocks into scratch at [0, t), [t, 2t), [2t, 3t): three reads,
 *  ONE barrier. Every verify row's operands are then built from these 12 tiles. */
static void stage_head_streams(ftq_context *ctx, const gdn_endpoints *endpoints,
                               const ftg_shape *shape, uint32_t scratch, uint32_t head)
{
    const uint32_t tiles = shape->state_tiles;
    const uint32_t k_head = ftg_k_head_for(head, shape->k_heads);
    const uint32_t first[3] = {
        ftg_delta_head_tile(shape, (uint32_t)kGdnStreamK, k_head),
        ftg_delta_head_tile(shape, (uint32_t)kGdnStreamQ, k_head),
        ftg_delta_head_tile(shape, (uint32_t)kGdnStreamV, head) };
    uint32_t block;

    if (ctx->aborting)
        return;
    for (block = 0; block < 3u; block++)
        ft_tensix_noc_read(ft_tensix_noc_address(endpoints->activation_x, endpoints->activation_y,
                                                 endpoints->activation_in +
                                                     first[block] * TILE_BYTES),
                           scratch + block * tiles * TILE_BYTES, tiles * TILE_BYTES);
    gdn_read_barrier();
}

/** One verify row's operands, in the one-row walk's order: k replicas, q replicas, v columns,
 *  decay, beta, scale. `gate` is THIS row's gate block. */
static bool push_row_operands(ftq_context *ctx, const ftg_shape *shape, uint32_t scratch,
                              uint32_t gate, uint32_t head, uint32_t row, uint32_t scale_bits)
{
    const uint32_t tiles = shape->state_tiles;
    uint32_t index;

    for (index = 0; index < tiles; index++)
        if (!push_replicated_tile(ctx, scratch, row, index))
            return false;
    for (index = 0; index < tiles; index++)
        if (!push_replicated_tile(ctx, scratch + tiles * TILE_BYTES, row, index))
            return false;
    for (index = 0; index < tiles; index++)
        if (!push_column_tile(ctx, scratch + 2u * tiles * TILE_BYTES, row, index))
            return false;
    return push_constant_column(ctx, *(const uint32_t *)(uintptr_t)(gate + head * 4u)) &&
           push_constant_column(ctx, *(const uint32_t *)(uintptr_t)(gate +
                                                              (shape->v_heads + head) * 4u)) &&
           push_constant_column(ctx, scale_bits);
}

/**
 * The reader half of a MULTI-ROW DELTA descriptor: per head, q/k/v and the state read ONCE,
 * then every row's operands built from them. The state goes to the compute thread for row 0
 * only; later rows decay the compute thread's own carry.
 */
static bool read_one_delta_rows(ftq_context *ctx, const gdn_endpoints *endpoints,
                                const ftg_shape *shape, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t scratch = ft_tensix_cb_write_pointer(CB_STAGE);
    const uint32_t gate = scratch + 3u * shape->state_tiles * TILE_BYTES;
    const uint32_t stride = ftg_gate_stride(shape->v_heads);
    const uint32_t scale_bits = parameters[kGdnScaleBits];
    const bool red_rows_row0 = (parameters[kGdnFlags] & (uint32_t)kGdnFlagRedRowsRow0) != 0u;
    uint32_t head;

    stage_gate(ctx, endpoints, shape, parameters, scratch, gate, stride, shape->row_count);
    for (head = shape->head_first; head < shape->head_first + shape->head_count; head++) {
        uint32_t row, tile_row;

        stage_head_streams(ctx, endpoints, shape, scratch, head);
        for (row = 0; row < shape->row_count; row++) {
            if (!push_row_operands(ctx, shape, scratch, gate + row * stride, head,
                                   shape->row_first + (red_rows_row0 ? 0u : row), scale_bits))
                return false;
            if (row != 0u)
                continue;
            for (tile_row = 0; tile_row < shape->state_tiles; tile_row++)
                if (!push_state_row(ctx, endpoints, shape, head, tile_row))
                    return false;
        }
    }
    return !ctx->aborting;
}

/* ===========================================================================
 * PHASE CONV -- streaming
 * =========================================================================== */

/**
 * Tile columns one CONV block covers.
 *
 * Bounded by three things at once: half of each buffer, so the reader can fill one block
 * while the compute thread drains the other; the PUSH GRID, because a chunk that does not
 * divide the buffer depth walks the write pointer off the limit; and the stream width, so
 * there is no short last block. Zero means "no depth satisfies all three", which is a
 * refusal, not a fallback.
 */
static uint32_t conv_block_depth(uint32_t tile_cols, uint32_t taps)
{
    const uint32_t in_pages = ft_tensix_cb(CB_IN)->fifo_num_pages;
    const uint32_t weight_pages = ft_tensix_cb(CB_VECTOR)->fifo_num_pages;
    const uint32_t window_pages = ft_tensix_cb(CB_STATE)->fifo_num_pages;
    uint32_t depth = CONV_DEPTH;

    if (taps < 2u || tile_cols == 0u || depth == 0u)
        return 0u;
    /* THE DEPTH ITSELF IS THE SHARED COMPILE-TIME ARG, not a page count, because the COMPUTE
     * kernel has to reach the identical number and cannot read `cb_interface` at all — it
     * does not exist on the MATH thread (link failure, measured 2026-08-18), and rule 3
     * forbids a trip count taken from runtime buffer state regardless. What the page counts
     * are still good for here is a REFUSAL: this processor can see whether the host actually
     * placed buffers deep enough for that depth, and a descriptor that does not fit is
     * refused by name instead of walking the FIFO write pointer off its limit. */
    if (depth > tile_cols)
        depth = tile_cols;
    while (depth > 0u && (tile_cols % depth) != 0u)
        depth--;
    if (depth == 0u)
        return 0u;
    if (in_pages < 2u * depth || weight_pages < 2u * depth * taps ||
        window_pages < 2u * depth * (taps - 1u))
        return 0u;
    if ((in_pages % depth) != 0u || (weight_pages % (depth * taps)) != 0u ||
        (window_pages % depth) != 0u)
        return 0u;
    return depth;
}

/** One window tap for one block: its own reserve/read/barrier/push group -- see the header on
 *  why `taps-1` tiles cannot share one reservation. */
static bool read_window_tap(ftq_context *ctx, const gdn_endpoints *endpoints,
                            const ftg_shape *shape, uint32_t depth, uint32_t first_col,
                            uint32_t tap)
{
    const uint32_t first = ftg_conv_plane_tile(shape, tap, first_col);

    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_STATE, (int32_t)depth, WAIT_MODE))
        return false;
    if (!ctx->aborting) {
        conv_read(endpoints->state_x, endpoints->state_y,
                  endpoints->state_base + first * TILE_BYTES,
                  ft_tensix_cb_write_pointer(CB_STATE), depth);
        gdn_read_barrier();
    }
    ft_tensix_cb_push_back(CB_STATE, depth);
    return true;
}

/** qkv and every weight tap for one block: `1 + taps` reads under ONE barrier. */
static bool read_conv_operands(ftq_context *ctx, const gdn_endpoints *endpoints,
                               const ftg_shape *shape, uint32_t depth, uint32_t first_col)
{
    const uint32_t taps = shape->conv_taps;
    uint32_t weight_write;
    uint32_t tap;

    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN, (int32_t)depth, WAIT_MODE))
        return false;
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_VECTOR, (int32_t)(depth * taps),
                                       WAIT_MODE))
        return false;
    weight_write = ft_tensix_cb_write_pointer(CB_VECTOR);

    if (!ctx->aborting) {
        conv_read_stream(endpoints->activation_x, endpoints->activation_y,
                         conv_stream_tile(endpoints, first_col),
                         ft_tensix_cb_write_pointer(CB_IN), depth);
        for (tap = 0; tap < taps; tap++) {
            const uint32_t first = ftg_conv_plane_tile(shape, tap, first_col);

            conv_read(endpoints->constant_x, endpoints->constant_y,
                      endpoints->constant_base + first * TILE_BYTES,
                      weight_write + tap * depth * TILE_BYTES, depth);
        }
        gdn_read_barrier();
    }
    ft_tensix_cb_push_back(CB_IN, depth);
    ft_tensix_cb_push_back(CB_VECTOR, depth * taps);
    return true;
}

/**
 * The reader half of one CONV descriptor: TWO passes over the stream.
 *
 * Pass A feeds the convolution itself. Pass B feeds the ROLLED WINDOW, and it is a second
 * pass rather than an interleave because CB_OUT is a FIFO and the writer's split puts every
 * conv_out tile before every window tile. The rolled window is `window_new[t] =
 * window_old[t+1]` for t < taps-2 and `window_new[taps-2] = qkv`, so pass B re-reads the
 * window one tap higher and finally re-reads qkv.
 */
static bool read_one_conv(ftq_context *ctx, const gdn_endpoints *endpoints,
                          const ftg_shape *shape, uint32_t depth)
{
    /* THIS MEMBER'S COLUMNS, at the full plane stride: `ftg_conv_plane_tile()` and the qkv
     * address both take an ABSOLUTE column, so the slice is only the loop bounds here. */
    const uint32_t start = shape->col_first;
    const uint32_t end = shape->col_first + shape->col_count;
    const uint32_t taps = shape->conv_taps;
    uint32_t first_col;
    uint32_t tap;

    for (first_col = start; first_col < end; first_col += depth) {
        if (!read_conv_operands(ctx, endpoints, shape, depth, first_col))
            return false;
        for (tap = 0; tap + 1u < taps; tap++)
            if (!read_window_tap(ctx, endpoints, shape, depth, first_col, tap))
                return false;
    }

    for (tap = 0; tap + 1u < taps; tap++) {
        for (first_col = start; first_col < end; first_col += depth) {
            if (tap + 2u < taps) {
                if (!read_window_tap(ctx, endpoints, shape, depth, first_col, tap + 1u))
                    return false;
                continue;
            }
            if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN, (int32_t)depth, WAIT_MODE))
                return false;
            if (!ctx->aborting) {
                conv_read_stream(endpoints->activation_x, endpoints->activation_y,
                                 conv_stream_tile(endpoints, first_col),
                                 ft_tensix_cb_write_pointer(CB_IN), depth);
                gdn_read_barrier();
            }
            ft_tensix_cb_push_back(CB_IN, depth);
        }
    }
    return !ctx->aborting;
}

static bool rebase_produced_buffers(ftq_context *ctx, const ftg_shape *shape);

/**
 * MULTI-ROW CONV (PM 6): element `j` of row-sequence [live window planes 0..taps-2, stream rows
 * row_first, row_first+1, ...] into CB_STATE for one block — a window plane, or a stream row
 * read into row 0 (strip mode). Row r's tap t is element t + r: exactly the window the one-row
 * descriptor of row r reads, after rows row_first..r-1 rolled it.
 */
static bool read_conv_sequence_tap(ftq_context *ctx, const gdn_endpoints *endpoints,
                                   const ftg_shape *shape, uint32_t depth, uint32_t first_col,
                                   uint32_t element)
{
    if (element + 1u < shape->conv_taps)
        return read_window_tap(ctx, endpoints, shape, depth, first_col, element);
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_STATE, (int32_t)depth, WAIT_MODE))
        return false;
    if (!ctx->aborting) {
        conv_read_stream_row(endpoints->activation_x, endpoints->activation_y,
                             conv_stream_tile(endpoints, first_col),
                             ft_tensix_cb_write_pointer(CB_STATE), depth,
                             shape->row_first + element - (shape->conv_taps - 1u));
        gdn_read_barrier();
    }
    ft_tensix_cb_push_back(CB_STATE, depth);
    return true;
}

/**
 * MULTI-ROW CONV: the one-row reader's two passes once a row, rows in order. Pass A is
 * read_one_conv()'s with the window taps taken from the row sequence; pass B's rolled window
 * is elements r+1 .. r+taps-1 of the same sequence, its newest tap the row's own stream row
 * (through CB_IN, exactly as the one-row pass B re-reads qkv).
 */
static bool read_one_conv_rows(ftq_context *ctx, const gdn_endpoints *endpoints,
                               const ftg_shape *shape, uint32_t depth)
{
    const uint32_t start = shape->col_first;
    const uint32_t end = shape->col_first + shape->col_count;
    const uint32_t taps = shape->conv_taps;
    uint32_t row, first_col, tap;

    for (row = 0; row < shape->row_count; row++) {
        /* EVERY ROW STARTS ON THE BUFFERS' BASES, as every descriptor does: pass B leaves
         * CB_STATE off the `depth*(taps-1)` grid pass A indexes (2 taps x the columns), and
         * the next row's window would then be read across the wrap. The compute thread
         * rebases its read pointers at the same row boundary. */
        if (row > 0u && !rebase_produced_buffers(ctx, shape))
            return false;
        gdn_row = shape->row_first + (gdn_red_rows_row0 ? 0u : row);
        for (first_col = start; first_col < end; first_col += depth) {
            if (!read_conv_operands(ctx, endpoints, shape, depth, first_col))
                return false;
            for (tap = 0; tap + 1u < taps; tap++)
                if (!read_conv_sequence_tap(ctx, endpoints, shape, depth, first_col, tap + row))
                    return false;
        }
        for (tap = 0; tap + 1u < taps; tap++) {
            for (first_col = start; first_col < end; first_col += depth) {
                if (tap + 2u < taps) {
                    if (!read_conv_sequence_tap(ctx, endpoints, shape, depth, first_col,
                                                tap + 1u + row))
                        return false;
                    continue;
                }
                if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN, (int32_t)depth,
                                                   WAIT_MODE))
                    return false;
                if (!ctx->aborting) {
                    conv_read_stream(endpoints->activation_x, endpoints->activation_y,
                                     conv_stream_tile(endpoints, first_col),
                                     ft_tensix_cb_write_pointer(CB_IN), depth);
                    gdn_read_barrier();
                }
                ft_tensix_cb_push_back(CB_IN, depth);
            }
        }
    }
    gdn_row = shape->row_first;
    return !ctx->aborting;
}

/* ===========================================================================
 * CONV ROW BATCH (resident_gdn_protocol.h): every row in ONE pass
 * =========================================================================== */

/** The CONV row batch count of the descriptor in flight; 0 = off. */
static uint32_t gdn_conv_batch;

/**
 * Elements [base, base + gdn_conv_batch) of the row sequence [live window planes 0..taps-2,
 * stream rows row_first, row_first+1, ...] into L1 rows 0.. of `tiles` CONV pages at
 * `destination`, columns from `first_col`. A window element is row 0 of its plane; the stream
 * elements are consecutive DRAM rows of one tile, so they go as ONE read a face half.
 * Barrier is the caller's.
 */
static void conv_read_sequence_rows(const gdn_endpoints *endpoints, const ftg_shape *shape,
                                    uint32_t first_col, uint32_t tiles, uint32_t base,
                                    uint32_t destination)
{
    const uint32_t window = shape->conv_taps - 1u;
    const uint32_t end = base + gdn_conv_batch;
    const uint32_t stream_first = base > window ? base : window;
    uint32_t tile, half, element;

    for (tile = 0; tile < tiles; tile++) {
        const uint32_t page = destination + tile * TILE_BYTES;
        const uint32_t column = first_col + tile;

        for (half = 0; half < 2u; half++) {
            for (element = base; element < end && element < window; element++)
                ft_tensix_noc_read(
                    ft_tensix_noc_address(endpoints->state_x, endpoints->state_y,
                                          endpoints->state_base +
                                              ftg_conv_plane_tile(shape, element, column) *
                                                  TILE_BYTES +
                                              ftg_face_row_offset(0u, half)),
                    page + ftg_face_row_offset(element - base, half), ftg_face_row_bytes());
            /* RED (kGdnFlagRedRowsRow0): every stream element reads row `row_first` — must
             * DIFFER, which proves each row's own stream row reaches the pass. */
            for (element = stream_first; gdn_red_rows_row0 && element < end; element++)
                ft_tensix_noc_read(
                    ft_tensix_noc_address(endpoints->activation_x, endpoints->activation_y,
                                          conv_stream_tile(endpoints, column) +
                                              ftg_face_row_offset(shape->row_first, half)),
                    page + ftg_face_row_offset(element - base, half), ftg_face_row_bytes());
            if (stream_first < end && !gdn_red_rows_row0)
                ft_tensix_noc_read(
                    ft_tensix_noc_address(endpoints->activation_x, endpoints->activation_y,
                                          conv_stream_tile(endpoints, column) +
                                              ftg_face_row_offset(shape->row_first +
                                                                      stream_first - window,
                                                                  half)),
                    page + ftg_face_row_offset(stream_first - base, half),
                    (end - stream_first) * ftg_face_row_bytes());
        }
    }
}

/** One batched sequence plane for one block into `cb`: reserve, read, barrier, push. */
static bool read_batch_plane(ftq_context *ctx, const gdn_endpoints *endpoints,
                             const ftg_shape *shape, uint32_t cb, uint32_t depth,
                             uint32_t first_col, uint32_t base)
{
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, (int32_t)depth, WAIT_MODE))
        return false;
    if (!ctx->aborting) {
        conv_read_sequence_rows(endpoints, shape, first_col, depth, base,
                                ft_tensix_cb_write_pointer(cb));
        gdn_read_barrier();
    }
    ft_tensix_cb_push_back(cb, depth);
    return true;
}

/** Row 0 of each of `tiles` weight pages copied into rows 1 .. gdn_conv_batch-1. Exact: a
 *  word copy. */
static void conv_replicate_weight_rows(uint32_t destination, uint32_t tiles)
{
    uint32_t tile, half, row;

    for (tile = 0; tile < tiles; tile++)
        for (half = 0; half < 2u; half++) {
            const uint32_t source = destination + tile * TILE_BYTES + ftg_face_row_offset(0u, half);

            for (row = 1; row < gdn_conv_batch; row++)
                ftg_copy_words((uint32_t *)(uintptr_t)(destination + tile * TILE_BYTES +
                                                       ftg_face_row_offset(row, half)),
                               (const uint32_t *)(uintptr_t)source, ftg_face_row_words());
        }
}

/** The batched stream rows and every weight tap for one block: `1 + taps` reads under ONE
 *  barrier, then the weight rows replicated, as read_conv_operands() does for one row. */
static bool read_batch_operands(ftq_context *ctx, const gdn_endpoints *endpoints,
                                const ftg_shape *shape, uint32_t depth, uint32_t first_col)
{
    const uint32_t taps = shape->conv_taps;
    uint32_t weight_write;
    uint32_t tap;

    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN, (int32_t)depth, WAIT_MODE))
        return false;
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_VECTOR, (int32_t)(depth * taps),
                                       WAIT_MODE))
        return false;
    weight_write = ft_tensix_cb_write_pointer(CB_VECTOR);

    if (!ctx->aborting) {
        conv_read_sequence_rows(endpoints, shape, first_col, depth, taps - 1u,
                                ft_tensix_cb_write_pointer(CB_IN));
        for (tap = 0; tap < taps; tap++)
            conv_read(endpoints->constant_x, endpoints->constant_y,
                      endpoints->constant_base +
                          ftg_conv_plane_tile(shape, tap, first_col) * TILE_BYTES,
                      weight_write + tap * depth * TILE_BYTES, depth);
        gdn_read_barrier();
        conv_replicate_weight_rows(weight_write, depth * taps);
    }
    ft_tensix_cb_push_back(CB_IN, depth);
    ft_tensix_cb_push_back(CB_VECTOR, depth * taps);
    return true;
}

/**
 * The reader half of one CONV ROW BATCH descriptor: read_one_conv()'s two passes, once, with
 * every page carrying rows 0 .. gdn_conv_batch-1. Pass A: CB_IN = elements taps-1.. (the
 * stream rows), window plane t = elements t... Pass B: rolled plane t = elements t+1.., its
 * newest plane the stream rows again through CB_IN.
 */
static bool read_one_conv_batch(ftq_context *ctx, const gdn_endpoints *endpoints,
                                const ftg_shape *shape, uint32_t depth)
{
    const uint32_t start = shape->col_first;
    const uint32_t end = shape->col_first + shape->col_count;
    const uint32_t taps = shape->conv_taps;
    uint32_t first_col, tap;

    for (first_col = start; first_col < end; first_col += depth) {
        if (!read_batch_operands(ctx, endpoints, shape, depth, first_col))
            return false;
        for (tap = 0; tap + 1u < taps; tap++)
            if (!read_batch_plane(ctx, endpoints, shape, CB_STATE, depth, first_col, tap))
                return false;
    }
    for (tap = 0; tap + 1u < taps; tap++)
        for (first_col = start; first_col < end; first_col += depth)
            if (!read_batch_plane(ctx, endpoints, shape, tap + 2u < taps ? CB_STATE : CB_IN,
                                  depth, first_col, tap + 1u))
                return false;
    return !ctx->aborting;
}

/* ===========================================================================
 * Dispatch
 * =========================================================================== */

/**
 * Refuse a descriptor the circular buffers this residency was STAGED with cannot serve.
 *
 * The page counts are fixed when the program is staged; the shape is a per-descriptor word,
 * so the host CAN ask for a block bigger than the buffer -- it did, on another op's first
 * hardware run, and the result was ten seconds of blocked polling reported as an idle
 * timeout. Every chunk this reader pushes is checked against the push grid as well as
 * against capacity, because the FIFO write pointer wraps on an EXACT hit of the limit and
 * nothing else.
 */
/**
 * Does `cb_index` hold a whole number of `chunk_pages` blocks?
 *
 * THE GRID A CONSUMER INDEXES ACROSS, which is a DIFFERENT rule from the push grid and the
 * one this family did not have. `ft_tensix_unpack_ab(cb, ..., tile)` resolves to
 * `read_pointer + tile * page_size` with no wrap, so a compute thread that waits for `n`
 * pages and then names tiles `0..n-1` needs the front low enough that the whole window fits.
 * The front is only ever a multiple of the chunk if the chunk divides the buffer — and the
 * three kernels rebase to the base at every descriptor so the multiple starts at zero.
 *
 * The pointer clause of `ft_tensix_cb_push_grid_is_safe()` is deliberately not reused here:
 * it tests the PRODUCER's pointer, and this is a statement about the CONSUMER's.
 */
static bool index_grid_is_safe(uint32_t cb_index, uint32_t chunk_pages)
{
    const uint32_t pages = ft_tensix_cb(cb_index)->fifo_num_pages;

    return chunk_pages != 0u && pages >= chunk_pages && (pages % chunk_pages) == 0u;
}

static bool delta_is_executable(const ftg_shape *shape)
{
    const uint32_t tiles = shape->state_tiles;

    if (ft_tensix_cb(CB_SCALER)->fifo_num_pages < 2u * tiles)
        return false;
    if (ft_tensix_cb(CB_VECTOR)->fifo_num_pages < tiles + (uint32_t)kGdnVectorConstants)
        return false;
    /* `state_tiles` staging tiles for a row block, plus ONE page above them for the gate
     * block this descriptor reads once. */
    if (ft_tensix_cb(CB_STAGE)->fifo_num_pages < tiles + 1u)
        return false;
    /* CB_WORK holds the decayed tile-row AND the updated one: (b) indexes off the first and
     * (e) off the second, so the front cycles 0 -> tiles -> 0 inside a tile-row and the
     * buffer needs both tile-rows live. */
    if (ft_tensix_cb(CB_WORK)->fifo_num_pages < 2u * tiles)
        return false;
    if (!ft_tensix_cb_push_grid_is_safe(CB_SCALER, 1u) ||
        !ft_tensix_cb_push_grid_is_safe(CB_VECTOR, 1u) ||
        !ft_tensix_cb_push_grid_is_safe(CB_STATE, tiles))
        return false;
    /* What the compute thread INDEXES ACROSS in this phase: a whole state tile-row out of
     * CB_WORK and CB_STATE, this head's k AND q replicas out of CB_SCALER, and its v columns
     * plus decay/beta/scale out of CB_VECTOR. CB_IN is NOT named: this phase does not touch
     * it on either processor. */
    if (!index_grid_is_safe(CB_WORK, tiles) || !index_grid_is_safe(CB_STATE, tiles) ||
        !index_grid_is_safe(CB_SCALER, 2u * tiles) ||
        !index_grid_is_safe(CB_VECTOR, tiles + (uint32_t)kGdnVectorConstants))
        return false;
    /* THE BATCHED (d)'s OWN GRIDS. It stages `batch` rank-1 products in CB_PROD and then
     * indexes all of them off its front, and it rewrites CB_WORK a whole block at a time --
     * so both need `batch` to divide them, on top of the tile-row grid above. `batch` divides
     * `tiles` by construction (`ftg_dst_batch`), so CB_WORK's clause is implied by the one
     * above it and CB_PROD's is not: nothing else in this function names CB_PROD, and before
     * the batch it had no grid at all because every access to it named tile 0. */
    {
        const uint32_t batch = ftg_dst_batch(tiles, DST_BATCH_MAX);

        if (ft_tensix_cb(CB_PROD)->fifo_num_pages < 2u * batch)
            return false;
        if (!index_grid_is_safe(CB_PROD, batch))
            return false;
        if (!index_grid_is_safe(CB_WORK, batch))
            return false;
    }
    /* MULTI-ROW: three staged row blocks plus the gate page, every row's gate block inside
     * that one page, and the compute thread's carry able to hold a head's whole state on its
     * tile-row grid. The carry is the compute thread's, but only this processor can see a
     * page count. */
    if (shape->row_count > 1u &&
        (ft_tensix_cb(CB_STAGE)->fifo_num_pages < 3u * tiles + 1u ||
         shape->row_count * ftg_gate_stride(shape->v_heads) > TILE_BYTES ||
         ft_tensix_cb((uint32_t)kGdnCbCarry)->fifo_num_pages < tiles * tiles ||
         !index_grid_is_safe((uint32_t)kGdnCbCarry, tiles)))
        return false;
    return shape->tile_count >= shape->head_count * ftg_state_tiles_per_head(shape);
}

static bool conv_is_executable(const ftg_shape *shape, uint32_t depth)
{
    if (depth == 0u)
        return false;
    /* THE THIRD SIDE OF THE THREE-SIDED CONTRACT. Every check below asks whether the buffers
     * the HOST placed can SERVE this image's depth, and a host that placed for a LARGER depth
     * passes all of them -- so the drift that silently costs a depth sweep its number is
     * invisible to capacity. This clause compares the two numbers themselves. See
     * kGdnByteConvDepth in resident_gdn_protocol.h. */
    if (!ftg_conv_depth_agrees(shape->declared_conv_depth, (uint32_t)CONV_DEPTH))
        return false;
    /* THE PUSH GRID IS CHECKED AFTER THE REBASE (conv_push_grid_is_safe(), read_one_gdn()),
     * not here. It reads the CURRENT write pointers, and `rebase_produced_buffers()` is about
     * to put every one of them back on its base — so checking first refused a CONV after a
     * DELTA that left CB_VECTOR off the CONV grid, a pointer the rebase was about to fix.
     * MEASURED 2026-10-04: a DELTA over 8 members pushes (32/8)*7 = 28 pages, off the 8-page
     * CONV grid, and every CONV refused and replayed (1,151 of 1,152). */
    /* `conv_one_column` names EVERY tap of a column, so the whole block's window and weights
     * are one indexed window even though the window is pushed a tap at a time. */
    if (!index_grid_is_safe(CB_IN, depth) ||
        !index_grid_is_safe(CB_STATE, depth * (shape->conv_taps - 1u)) ||
        !index_grid_is_safe(CB_VECTOR, depth * shape->conv_taps))
        return false;
    return shape->tile_count >= shape->conv_tile_cols * (shape->conv_taps - 1u);
}

/**
 * Put this processor's write pointers back on their buffers' bases, for the buffers it
 * produces into IN THIS PHASE.
 *
 * The whole-buffer reservation is the precondition `ft_tensix_cb_rebase_write_pointer()`
 * states: it does not move a pointer, it waits until nothing live is behind it. Only the
 * buffers this phase writes are named — reserving a buffer this phase does not touch would
 * block on a consumer that has no reason to drain it.
 *
 * @return false if a wait was abandoned, in which case the descriptor is not retired
 */
static bool rebase_produced_buffers(ftq_context *ctx, const ftg_shape *shape)
{
    const uint32_t delta[3] = { CB_STATE, CB_SCALER, CB_VECTOR };
    const uint32_t conv[3] = { CB_IN, CB_STATE, CB_VECTOR };
    const uint32_t *buffers = (shape->phase == (uint32_t)kGdnPhaseDelta) ? delta : conv;
    uint32_t index;

    for (index = 0; index < 3u; index++) {
        const uint32_t cb = buffers[index];

        if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb,
                                           (int32_t)ft_tensix_cb(cb)->fifo_num_pages,
                                           WAIT_MODE))
            return false;
        ft_tensix_cb_rebase_write_pointer(cb);
    }
    return true;
}

static bool descriptor_is_executable(ftq_context *ctx, const ftg_shape *shape, uint32_t depth)
{
    bool executable;

    if (TILE_BYTES == 0u || shape->tile_bytes != TILE_BYTES)
        executable = false;
    else if (shape->phase == (uint32_t)kGdnPhaseDelta)
        executable = delta_is_executable(shape);
    else
        executable = conv_is_executable(shape, depth);

    if (!executable)
        ctx->error = kErrBadParameters;
    return executable;
}

/**
 * The reader half of one GDN descriptor.
 *
 * @return true if the whole descriptor was read; false if a wait was abandoned, in which case
 *         the caller must not retire it (rule 5). A failed reservation returns here rather
 *         than pushing anyway, exactly as resident_rope_reader.c and
 *         resident_attention_reader.c do: a push with no satisfied reservation advances a
 *         page counter past the buffer's capacity, which is corruption rather than a drain.
 */
static bool read_one_gdn(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const ftg_shape shape = ftg_decode(parameters);
    gdn_endpoints endpoints;
    uint32_t depth = 0u;

    if (shape.phase == (uint32_t)kGdnPhaseConv)
        depth = conv_block_depth(shape.col_count, shape.conv_taps);
    if (!ftg_shape_is_supported(&shape) || !descriptor_is_executable(ctx, &shape, depth)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }

    gdn_stream_bf16 = shape.phase == (uint32_t)kGdnPhaseConv &&
                      (parameters[kGdnFlags] & (uint32_t)kGdnFlagConvBf16In) != 0u;
    gdn_widen_tiles = 0u;
    load_endpoints(&endpoints, &shape, parameters);
    if (!rebase_produced_buffers(ctx, &shape))
        return false;
    gdn_noc_replicate = (parameters[kGdnFlags] & (uint32_t)kGdnFlagNocReplicate) != 0u;
    gdn_red_no_copy = (parameters[kGdnFlags] & (uint32_t)kGdnFlagRedNoCopy) != 0u;
    gdn_conv_strip = (parameters[kGdnFlags] & (uint32_t)kGdnFlagConvStrip) != 0u;
    gdn_red_strip_half = (parameters[kGdnFlags] & (uint32_t)kGdnFlagRedStripHalf) != 0u;
    gdn_row = (parameters[kGdnFlags] >> kGdnRowShift) & (uint32_t)kGdnRowMask;
    gdn_red_rows_row0 = (parameters[kGdnFlags] & (uint32_t)kGdnFlagRedRowsRow0) != 0u;
    /* The CONV row select — and so the multi-row CONV — exists in strip mode only (a
     * whole-tile read has no row). */
    if ((gdn_row != 0u || shape.row_count > 1u || gdn_stream_bf16) &&
        shape.phase == (uint32_t)kGdnPhaseConv && !gdn_conv_strip) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    gdn_conv_batch = ftg_conv_batch_rows(parameters[kGdnFlags]);
    if (!ftg_conv_batch_is_supported(&shape, parameters[kGdnFlags])) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    if (shape.phase == (uint32_t)kGdnPhaseDelta) {
        const uint32_t started = ft_bh_wall_clock_low();
        const bool done = shape.row_count > 1u
                              ? read_one_delta_rows(ctx, &endpoints, &shape, parameters)
                              : read_one_delta(ctx, &endpoints, &shape, parameters);

        gdn_clock_add(kGdnClockDelta, started);
        return done;
    }
    /* THE PUSH GRID, on the REBASED pointers — where it is a property of the placement, which
     * is the thing it exists to check, instead of of the previous descriptor. */
    if (!ft_tensix_cb_push_grid_is_safe(CB_IN, depth) ||
        !ft_tensix_cb_push_grid_is_safe(CB_VECTOR, depth * shape.conv_taps) ||
        !ft_tensix_cb_push_grid_is_safe(CB_STATE, depth)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    {
        const uint32_t started = ft_bh_wall_clock_low();
        bool done;

        gdn_conv_phase = true;
        if (gdn_conv_batch != 0u)
            done = read_one_conv_batch(ctx, &endpoints, &shape, depth);
        else
            done = shape.row_count > 1u ? read_one_conv_rows(ctx, &endpoints, &shape, depth)
                                        : read_one_conv(ctx, &endpoints, &shape, depth);
        gdn_conv_phase = false;
        gdn_clock_add(kGdnClockConv, started);
        return done;
    }
}

void kernel_main(void)
{
    ftq_context ctx;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);
    ctx.control[kFieldState] = kStateBoot;

    if (!ftq_check_control(&ctx)) {
        ctx.control[kFieldError] = ctx.error;
        ctx.control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    // RESUME rather than restart. All five processors resume from completed_count, the ONE
    // number that means "every stage finished this descriptor" — resident_matmul_reader.c
    // carries the full argument and the refusal that enforces it.
    if (ctx.control[kFieldTail] != ctx.control[kFieldCompletedCount]) {
        ctx.error = kErrBadParameters;
        ctx.control[kFieldError] = ctx.error;
        ctx.control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }
    ctx.cursor = ctx.control[kFieldCompletedCount];
    gdn_clock_control = ctx.control;
    {
        /* This core's own NoC coordinate, for the local copies (the stock firmware derives it
         * from NOC_NODE_ID). */
        const uint32_t id = ft_bh_noc_cmd_read(NOC_INDEX, 0u,
                                                       FT_BH_NOC_REG_BASE + 0x44u);

        gdn_self_x = id & FT_BH_NOC_NODE_ID_MASK;
        gdn_self_y = (id >> FT_BH_NOC_NODE_ID_BITS) & FT_BH_NOC_NODE_ID_MASK;
    }
    ctx.control[kGdnClockBarrier] = 0u;
    ctx.control[kGdnClockBuild] = 0u;
    ctx.control[kGdnClockConv] = 0u;
    ctx.control[kGdnClockConvBarrier] = 0u;
    ctx.control[kGdnClockDelta] = 0u;

    while (running) {
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);
        /* S2: the producer this descriptor's sidecar entry names has completed, before any
         * input is read (resident_precondition.h). Refused or stopped: leave, unconsumed. */
        if (!ftq_precondition_wait(&ctx)) {
            running = false;
            continue;
        }

        volatile FTQ_L1_PTR uint32_t *descriptor = ftq_descriptor(&ctx);
        const uint32_t opcode = descriptor[kDescOpcode];
        bool done;

        ctx.control[kFieldLastOpcode] = opcode;
        ftq_status_opcode(&ctx, opcode);

        if (opcode == kOpGdn) {
            ftq_status_set(&ctx, kStateCbReserveBack);
            done = read_one_gdn(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
            if (!done)
                running = false;
        } else if (opcode == kOpNop) {
            done = true;
        } else if (opcode == kOpStop) {
            done = true;
            running = false;
        } else {
            // A binary is bound to one slab, so an opcode another program implements is
            // genuinely not available here. Refusing names that; executing nothing would
            // retire work that never happened.
            ctx.error = kErrBadOpcode;
            done = false;
            running = false;
        }

        // Rule 5. An abandoned descriptor was handed to nobody, so it advances NOTHING and
        // the next launch re-attempts it.
        if (done) {
            ctx.cursor++;
            ctx.control[kFieldTail] = ctx.cursor;
        }
    }

    // Nothing is outstanding here: the drain path issues no transactions. The barrier is
    // the cheap statement of that invariant rather than a fix for a case that can happen.
    gdn_read_barrier();
    ft_tensix_noc_write_barrier();
    ctx.control[kFieldError] = ctx.error;
    ctx.control[kFieldState] = kStateExit;
    ftq_status_set(&ctx, kStateExit);
}
