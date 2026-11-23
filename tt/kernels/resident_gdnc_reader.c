/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_gdnc_reader.c
 * @brief Reader third of the RESIDENT CHUNKWISE GATED-DELTA-NET triad, and the processor
 *   that ASSEMBLES THE WINDOW TILE -- the one shape the compute surface cannot express.
 *
 * The compute thread of this op never indexes an element. It is handed aligned tile streams
 * and issues elementwise multiplies, `matmul_tiles` against constant 0/1 band tiles and one
 * SiLU. Everything that makes those legal lives here, and for PHASE CONV that is exactly
 * three things:
 *
 *   X_c    the chunk's input tile column c -- ONE whole-tile read, unchanged from decode
 *   Win_c  the rolling window's `taps-1` ROW-0 planes STACKED into rows 0..taps-2 of one
 *          tile. `2*(taps-1)` 64-byte NoC reads, straight into the circular buffer page.
 *   W_tau  the `taps` ROW-REPLICATED weight planes at column c -- `taps` whole-tile reads
 *
 * plus the `2*taps` constant band tiles, read ONCE per descriptor.
 *
 * ===========================================================================
 * WHY THE WINDOW ASSEMBLY IS FREE, AND THE ONE THING THAT IS NOT
 * ===========================================================================
 * Row 0 of a Float32 32x32 tile is TWO 64-BYTE PIECES 1,024 B apart -- the top row of face 0
 * (columns 0..15) and the top row of face 1 (columns 16..31). 64 bytes is exactly the NoC's
 * DRAM read granule and exactly the alignment it requires, so "take row 0 of plane p and put
 * it at row p of a tile" is two ordinary reads with a computed destination offset. No
 * scratch buffer, no word copy, no gather. That is why the window keeps the decode family's
 * ROW-0 PLANE layout: the layout that is cheapest for one position is also free to stack.
 *
 * WHAT IS NOT FREE IS THE TAIL. Rows `taps-1..31` of the assembled tile are never written,
 * and they are multiplied by ZERO COLUMNS of every band matrix `C_tau` -- which is fine for
 * any finite value and NOT fine for a NaN, because `0 * NaN` is NaN and it would propagate
 * through the SiLU into a chunk nobody inspects. So CB_STATE IS ZEROED ONCE, AT RESIDENCY
 * BOOT, over both its pages, and this processor writes only rows `0..taps-2` afterwards. The
 * zeros are therefore invariant for the life of the residency. Zeroing per descriptor would
 * be 29 rows x 2 face rows x 256 columns of word stores -- the ~262,000 scalar L1 copies
 * that were PHASE DELTA's whole cost in the decode family, seen through a division.
 *
 * ===========================================================================
 * ONE BARRIER PER COLUMN, AND EVERY READ OF THAT COLUMN BEHIND IT
 * ===========================================================================
 * A NoC read barrier is a per-CORE drain. This reader issues all `1 + 2*(taps-1) + taps`
 * reads of a column, takes ONE barrier, and then pushes all three buffers. The decode
 * family's CONV reader cannot do that -- its window is `taps-1` separate reservations
 * because `taps-1` does not divide its window buffer -- and here it can, because the window
 * is ONE page.
 *
 * ===========================================================================
 * THE CB MAP
 * ===========================================================================
 *   cb 0  CB_IN      the chunk's input tile column. THIS PROCESSOR IS ITS ONLY PRODUCER.
 *   cb 1  CB_STATE   the assembled window tile. Zeroed once at boot; rows taps-1..31 stay
 *                    zero for the life of the residency.
 *   cb 2  CB_BAND    the `2*taps` constant band tiles, pushed ONCE per descriptor and popped
 *                    by the compute thread at the END of the descriptor, so the buffer is
 *                    empty at every descriptor boundary and its push is an exact hit.
 *   cb 3  CB_VECTOR  the `taps` row-replicated weight tiles of this column.
 *   cb 4  CB_PROD    the COMPUTE thread's product staging. This processor never touches it;
 *                    the index is here so the refusal below covers its indexed-read grid,
 *                    which is the same three-sided refusal every other buffer gets and which
 *                    this processor is the only one of the three that can evaluate (a page
 *                    count comes off `cb_interface`, which does not exist on the MATH
 *                    thread).
 *
 * THIS PROCESSOR OWNS `tail`, AND NOTHING ELSE IN THE CONTROL BLOCK
 * (resident_queue_protocol.h rule 1). An aborted descriptor advances nothing.
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` -- no tt-metal include.
 * ===========================================================================
 * PHASE DELTA ASKS THIS PROCESSOR FOR ONE THING THE COMPUTE SURFACE CANNOT EXPRESS TOO
 * ===========================================================================
 * Everything a head needs is CONTIGUOUS -- q, k, v and the `[S,S]` carried state are all runs
 * of consecutive tiles, because this family kept the DECODE family's layouts byte for byte.
 * Four bulk reads a head. What it cannot express is the GATE COLUMNS: `bcast_col` reads
 * COLUMN 0 of its second operand, and element [j][0] of a Float32 tile is 128 bytes from
 * element [j+1][0], so a column is a 32-element TRANSPOSE and therefore 32 word stores.
 *
 * THE GQA REPEAT IS AN ADDRESS HERE AND NEVER A TENSOR: `ftgc_delta_head_tile()` sends v-head
 * `vh`'s q and k reads to k-head `vh % k_heads`.
 *
 * Compile-time args (positional):
 *   0: CB_IN index   1: CB_STATE index   2: CB_BAND index   3: CB_VECTOR index
 *   4: CB_PROD index (the compute thread's; named only so the grid refusal covers it)
 *   5: tile size in bytes
 *   6: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 *   7: CB_WORK index (the compute thread's; grid refusal only)
 *   8: CB_DECAY index
 *   9: CB_CC index   (the compute thread's; grid refusal only)
 *  10: CB_SCRATCH index -- THIS PROCESSOR'S PRIVATE PAGE, and an index nobody pushes
 *  11: CONV_DEPTH -- tile columns behind ONE NoC read barrier. The HOST declares its own on
 *      every CONV descriptor and `conv_is_executable()` refuses one that disagrees; a SHORT
 *      line reads 0 here and `conv_block_depth()` turns that into a refusal of every CONV
 *      descriptor rather than a silent depth, which is the same shape the decode family's
 *      `resident_gdn_reader.c` chose for the identical argument.
 *   TWELVE args, and the count is load-bearing: `ft_tensix_compile_time_arg(n)` against a
 *   short line reads one word PAST the args array and SFPI GCC 15.1.0 FOLDS THAT TO A
 *   LITERAL ZERO with no diagnostic at any `-Warray-bounds` level. Wait mode is left at index
 *   6 for exactly that reason: a short line then reads mode 1 and a CB index of 0, which
 *   `delta_is_executable()` refuses loudly -- rather than mode 0, upstream's UNSTOPPABLE wait,
 *   which is how a Tensix gets wedged.
 *   `tools/check_tensix_kernel_compile_time_arg_counts.py` is the only other guard there is.
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build (the whole program; this file is the BRISC/NoC 0 third):
 *   RELOCATE_KERNEL_TEXT=1 \
 *     READER_CTARGS=0,1,2,3,4,4096,1,5,7,8,6,1 COMPUTE_CTARGS=0,1,2,3,4,16,1,5,7,8,1 \
 *     WRITER_CTARGS=16,4096,1 CB_TABLE=kernels/cb_tables/gdnc_float32.cbtable \
 *     bash foundry/tt/build_tensix_program.sh \
 *       kernels/resident_gdnc_reader.c kernels/resident_gdnc_compute.c \
 *       kernels/resident_gdnc_writer.c programs/gdnc
 *
 * @status PHASE CONV green on silicon 2026-08-21. PHASE DELTA NEW.
 * @issues The block depth is now a knob (`CONV_DEPTH`, arg 11) and ships at ONE. Raising it is
 *   still a change to this file, the compute kernel and `place_cbs` TOGETHER, and the host must
 *   be told the same number -- which is why the host DECLARES its own on the wire and both
 *   kernels compare for strict equality. The rungs the arena admits are priced off hardware by
 *   `tools/price_gdnc_conv_depth.c`.
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "ftgc_compact.h"
#include "resident_gdnc_protocol.h"
/* Direct NoC calls, not ftgc_table_io's pointers (ftgc_table.h says why). */
#define FTGT_READ(io, xy, address, l1, bytes)                                                  \
    ft_tensix_noc_read(ft_tensix_noc_address(ftgc_xy_x(xy), ftgc_xy_y(xy), (address)), (l1),  \
                       (bytes))
#define FTGT_BARRIER(io) ft_tensix_noc_read_barrier()
#include "ftgc_table.h"
#include "resident_queue_protocol.h"
#include "resident_precondition.h"

#define CB_IN ft_tensix_compile_time_arg(0)
#define CB_STATE ft_tensix_compile_time_arg(1)
#define CB_BAND ft_tensix_compile_time_arg(2)
#define CB_VECTOR ft_tensix_compile_time_arg(3)
/** The COMPUTE thread's product staging buffer. Named here ONLY so this processor's refusal
 *  covers it; nothing below reads or writes it. */
#define CB_PROD ft_tensix_compile_time_arg(4)
#define TILE_BYTES ft_tensix_compile_time_arg(5)
#define WAIT_MODE ft_tensix_compile_time_arg(6)
/** PHASE DELTA's four. CB_WORK and CB_CC are the COMPUTE thread's and this processor never
 *  touches them; they are named so this processor's page-count refusal covers their indexed
 *  read grids, which it is the only one of the three that can evaluate -- `fifo_num_pages`
 *  comes off `cb_interface`, and that does not exist on the MATH thread. */
#define CB_WORK ft_tensix_compile_time_arg(7)
#define CB_DECAY ft_tensix_compile_time_arg(8)
#define CB_CC ft_tensix_compile_time_arg(9)
/** THIS PROCESSOR'S PRIVATE SCRATCH, and it must be an index NOBODY PUSHES: the staging idiom
 *  writes through a CB's write pointer WITHOUT reserving, which is safe only while that
 *  pointer is provably at the buffer base. */
#define CB_SCRATCH ft_tensix_compile_time_arg(10)
/** PHASE CONV's BLOCK DEPTH: tile columns read behind ONE NoC read barrier. TWELVE args now,
 *  and the count is load-bearing for the reason stated above — a short line reads one word
 *  PAST the array and SFPI GCC folds it to a literal 0, which `conv_block_depth()` turns into
 *  a loud refusal rather than a silent depth. The host declares ITS OWN depth on every CONV
 *  descriptor and `conv_is_executable()` refuses one that disagrees with this constant; see
 *  `kGdncByteConvDepth` for the measurement that direction corrupts. */
#define CONV_DEPTH ft_tensix_compile_time_arg(11)

/** The three DRAM endpoints a descriptor names, decoded once. */
typedef struct gdnc_endpoints {
    uint32_t activation_x;
    uint32_t activation_y;
    uint32_t activation_in;
    uint32_t constant_x;
    uint32_t constant_y;
    uint32_t constant_base;
    uint32_t gate_base; /**< PHASE DELTA only, and INSIDE the constant bank */
    uint32_t state_x;
    uint32_t state_y;
    uint32_t state_base;
} gdnc_endpoints;

static void load_endpoints(gdnc_endpoints *endpoints, volatile FTQ_L1_PTR uint32_t *parameters)
{
    endpoints->activation_x = ftgc_xy_x(parameters[kGdncActivationXy]);
    endpoints->activation_y = ftgc_xy_y(parameters[kGdncActivationXy]);
    endpoints->activation_in = parameters[kGdncActivationIn];
    endpoints->constant_x = ftgc_xy_x(parameters[kGdncConstantXy]);
    endpoints->constant_y = ftgc_xy_y(parameters[kGdncConstantXy]);
    endpoints->constant_base = parameters[kGdncConstantWeight];
    endpoints->gate_base = parameters[kGdncDeltaGate];
    endpoints->state_x = ftgc_xy_x(parameters[kGdncStateXy]);
    endpoints->state_y = ftgc_xy_y(parameters[kGdncStateXy]);
    endpoints->state_base = parameters[kGdncStateConv];
}

/* ===========================================================================
 * Face arithmetic. Modelled on resident_gdn_reader.c's; spelled locally because
 * resident_gdnc_protocol.h is the WIRE contract and does not carry L1 layout.
 * =========================================================================== */

/** Bytes of one face: a 2x2 face grid over the tile. */
static uint32_t ftgc_face_bytes(void)
{
    return TILE_BYTES / 4u;
}

/** Bytes of one face ROW: 16 elements of a 32-element row, i.e. tile_bytes/64. 64 for a
 *  Float32 tile, which is exactly the NoC's DRAM read granule -- the whole reason the window
 *  assembly is two transactions and not a gather. */
static uint32_t ftgc_face_row_bytes(void)
{
    return TILE_BYTES / 64u;
}

/** Byte offset, inside a tile, of the face row holding `row` and columns [16*half, +16). */
static uint32_t ftgc_face_row_offset(uint32_t row, uint32_t half)
{
    return ((row / (uint32_t)kGdncFaceRows) * 2u + half) * ftgc_face_bytes() +
           (row % (uint32_t)kGdncFaceRows) * ftgc_face_row_bytes();
}

/* ===========================================================================
 * Staging
 * =========================================================================== */

/**
 * The `2*taps` constant band tiles, ONE transaction, ONCE per descriptor.
 *
 * They ride at the END of the weight buffer (`ftgc_band_base_tile`) rather than in a sixth
 * DRAM endpoint, because `FT_TT_QUEUE_PARAM_WORDS` is 16 for every family and a seventeenth
 * word moves the shared slot stride.
 */
static bool read_band(ftq_context *ctx, const gdnc_endpoints *endpoints,
                      const ftgc_shape *shape)
{
    const uint32_t tiles = ftgc_band_tiles(shape);
    const uint32_t first = ftgc_band_base_tile(shape);

    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_BAND, (int32_t)tiles, WAIT_MODE))
        return false;
    if (!ctx->aborting) {
        ft_tensix_noc_read(ft_tensix_noc_address(endpoints->constant_x, endpoints->constant_y,
                                                 endpoints->constant_base +
                                                     first * TILE_BYTES),
                           ft_tensix_cb_write_pointer(CB_BAND), tiles * TILE_BYTES);
        ft_tensix_noc_read_barrier();
    }
    ft_tensix_cb_push_back(CB_BAND, tiles);
    return true;
}

/**
 * One tile column: the chunk's input tile, the assembled window tile and the `taps` weight
 * tiles. Every read behind ONE barrier; every push after it.
 *
 * The reservations all happen FIRST, because a reservation is the only thing that says the
 * destination L1 pages are not still being read by the compute thread, and a read issued
 * into an unreserved page is corruption a barrier cannot undo.
 */
static bool read_column_group(ftq_context *ctx, const gdnc_endpoints *endpoints,
                              const ftgc_shape *shape, uint32_t first_column, uint32_t depth)
{
    const uint32_t taps = shape->conv_taps;
    uint32_t in_write;
    uint32_t window_write;
    uint32_t weight_write;
    uint32_t tap;
    uint32_t plane;
    uint32_t slot;

    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN, (int32_t)depth, WAIT_MODE))
        return false;
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_STATE, (int32_t)depth, WAIT_MODE))
        return false;
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_VECTOR, (int32_t)(depth * taps),
                                       WAIT_MODE))
        return false;

    in_write = ft_tensix_cb_write_pointer(CB_IN);
    window_write = ft_tensix_cb_write_pointer(CB_STATE);
    weight_write = ft_tensix_cb_write_pointer(CB_VECTOR);

    if (!ctx->aborting) {
        /* The chunk's own input tiles: rows 0..chunk_rows-1 are the C positions, and the host
         * owes zeros above that (see the protocol header). `depth` CONSECUTIVE tile columns
         * are CONSECUTIVE tiles in the strip, so the whole block is ONE read whatever the
         * depth — which is the reason the depth costs no extra transactions here. */
        ft_tensix_noc_read(ft_tensix_noc_address(endpoints->activation_x,
                                                 endpoints->activation_y,
                                                 endpoints->activation_in +
                                                     first_column * TILE_BYTES),
                           in_write, depth * TILE_BYTES);

        /* THE WINDOW ASSEMBLY. Plane p's ROW 0 becomes ROW p of one tile: two 64-byte pieces
         * 1,024 B apart at the source, two computed face-row offsets at the destination.
         * Rows taps-1..31 of every page were zeroed at boot and are not touched.
         *
         * THIS is the part that does NOT collapse with depth: a 64-byte face row of column
         * `c` and of column `c+1` are a whole tile apart at BOTH ends, so it stays
         * `depth * 2 * (taps-1)` reads. The lever is the BARRIER, not the transaction
         * count — at 680.2 us over 256 columns the descriptor is spending 2.66 us a column on
         * a serialized round trip, and 13 posted reads do not cost that. */
        for (slot = 0; slot < depth; slot++) {
            for (plane = 0; plane + 1u < taps; plane++) {
                const uint32_t source =
                    endpoints->state_base +
                    ftgc_plane_tile(shape, plane, first_column + slot) * TILE_BYTES;
                uint32_t half;

                for (half = 0; half < 2u; half++) {
                    ft_tensix_noc_read(
                        ft_tensix_noc_address(endpoints->state_x, endpoints->state_y,
                                              source +
                                                  ftgc_face_row_offset((uint32_t)kGdncStripRow,
                                                                       half)),
                        window_write + slot * TILE_BYTES + ftgc_face_row_offset(plane, half),
                        ftgc_face_row_bytes());
                }
            }
        }

        /* The `taps` ROW-REPLICATED weight planes, TAP-MAJOR then column, which is the order
         * `CB_VECTOR` is indexed in: tile `tap*depth + slot`. A plane's `depth` columns are
         * consecutive tiles (`ftgc_plane_tile` is `tap*conv_tile_cols + column`), so this is
         * `taps` reads whatever the depth. */
        for (tap = 0; tap < taps; tap++) {
            ft_tensix_noc_read(ft_tensix_noc_address(endpoints->constant_x,
                                                     endpoints->constant_y,
                                                     endpoints->constant_base +
                                                         ftgc_plane_tile(shape, tap,
                                                                         first_column) *
                                                             TILE_BYTES),
                               weight_write + tap * depth * TILE_BYTES, depth * TILE_BYTES);
        }
        ft_tensix_noc_read_barrier();
    }

    ft_tensix_cb_push_back(CB_IN, (int32_t)depth);
    ft_tensix_cb_push_back(CB_STATE, (int32_t)depth);
    ft_tensix_cb_push_back(CB_VECTOR, (int32_t)(depth * taps));
    return true;
}

/**
 * Tile columns one CONV block covers, or ZERO for a refusal.
 *
 * Bounded by four things at once: the compile-time constant; the SLICE this descriptor owns,
 * so there is no short last block; half of each buffer, so this processor can fill one block
 * while the compute thread drains the other; and the INDEX GRID, because a group that does not
 * divide the page count walks a consumer's `read_pointer + tile*page_size` off the end.
 *
 * The shrink is a SHRINK and not a fallback: it is the same expression the compute thread runs
 * — that thread cannot see `cb_interface` at all, so the two must agree from the same inputs —
 * and a shrunk depth under a placement rounded onto the UNSHRUNK one fails the grid test right
 * below, loudly. That is why the reachable rungs are the divisors and not every integer.
 */
static uint32_t conv_block_depth(const ftgc_shape *shape)
{
    const uint32_t taps = shape->conv_taps;
    const uint32_t in_pages = ft_tensix_cb(CB_IN)->fifo_num_pages;
    const uint32_t window_pages = ft_tensix_cb(CB_STATE)->fifo_num_pages;
    const uint32_t weight_pages = ft_tensix_cb(CB_VECTOR)->fifo_num_pages;
    uint32_t depth = CONV_DEPTH;

    if (taps < 2u || shape->col_count == 0u || depth == 0u)
        return 0u;
    if (depth > shape->col_count)
        depth = shape->col_count;
    while (depth > 0u && (shape->col_count % depth) != 0u)
        depth--;
    if (depth == 0u)
        return 0u;
    if (in_pages < 2u * depth || window_pages < 2u * depth ||
        weight_pages < 2u * depth * taps)
        return 0u;
    if ((in_pages % depth) != 0u || (window_pages % depth) != 0u ||
        (weight_pages % (depth * taps)) != 0u)
        return 0u;
    return depth;
}

/* ===========================================================================
 * PHASE DELTA -- staging
 *
 * Everything a head needs is CONTIGUOUS in DRAM, which is the whole dividend of keeping the
 * decode family's layouts: q, k and v are `state_tiles` consecutive tiles each, the carried
 * state is `state_tiles^2` consecutive tiles, and the decay-ratio matrix is one tile. Four
 * bulk reads and one 640-byte read a head, all behind ONE barrier.
 *
 * The ONLY thing this processor builds rather than moves is the five gate columns, and that is
 * a TRANSPOSE of 32 scalars: `bcast_col` reads column 0 of its second operand, and element
 * [j][0] of a Float32 tile is not contiguous with element [j+1][0]. 32 word stores a column.
 * The other 31 columns are NEVER READ and are left as they are -- zeroing them would be 1,024
 * word stores a column against 32, and the decode family's reader states the same thing about
 * the same operand.
 * =========================================================================== */

/** Byte offset, inside a tile, of element [row][column]. */
static uint32_t ftgc_element_offset(uint32_t row, uint32_t column)
{
    return ftgc_face_row_offset(row, column / (uint32_t)kGdncFaceRows) +
           (column % (uint32_t)kGdncFaceRows) * (TILE_BYTES / 1024u);
}

/**
 * One gate column into CB_VECTOR: `[j][0] = scratch[run][j]`, the other lanes untouched.
 *
 * The reservation is the caller's -- a head reserves all five at once so its five pushes
 * cannot interleave with the compute thread's absolute indexing of them.
 */
static void push_gate_column(uint32_t scratch, uint32_t run)
{
    const uint32_t destination = ft_tensix_cb_write_pointer(CB_VECTOR);
    const uint32_t source = scratch + run * (uint32_t)kGdncTileRows * 4u;
    uint32_t row;

    for (row = 0; row < (uint32_t)kGdncTileRows; row++)
        *(uint32_t *)(uintptr_t)(destination + ftgc_element_offset(row, 0u)) =
            *(const uint32_t *)(uintptr_t)(source + row * 4u);
    ft_tensix_cb_push_back(CB_VECTOR, 1);
}

/** The four PROGRAM CONSTANT tiles, ONE transaction, ONCE per descriptor. They sit at the head
 *  of the constant buffer, ahead of the per-head decay tiles. */
static bool read_delta_constants(ftq_context *ctx, const gdnc_endpoints *endpoints)
{
    const uint32_t tiles = (uint32_t)kGdncDeltaConstTiles;

    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_BAND, (int32_t)tiles, WAIT_MODE))
        return false;
    if (!ctx->aborting) {
        ft_tensix_noc_read(ft_tensix_noc_address(endpoints->constant_x, endpoints->constant_y,
                                                 endpoints->constant_base),
                           ft_tensix_cb_write_pointer(CB_BAND), tiles * TILE_BYTES);
        ft_tensix_noc_read_barrier();
    }
    ft_tensix_cb_push_back(CB_BAND, tiles);
    return true;
}

/** L11b (kGdncDeltaFlagCompact): this descriptor's gate block is COMPACT — header (scale bits
 *  at word 0) then decay[32] | beta[32] a head — and the decay-ratio tile and gate runs are
 *  built here (ftgc_compact.h) instead of read. */
static bool gdnc_compact;
static uint32_t gdnc_first_row;

/** Where the compact header and the head's run land in CB_SCRATCH, past the 640 B of runs. */
#define COMPACT_SCRATCH_OFFSET 1024u

/** kGdncDeltaFlagTable: the compact run is looked up from the alpha/beta strips through the
 *  host's tables (ftgc_table.h) instead of read from a gate block; `gdnc_band` is the strips'
 *  band. The front end writes the same header + run at COMPACT_SCRATCH_OFFSET. */
static bool gdnc_table;
static uint32_t gdnc_band;
static ftgc_table_state gdnc_table_state;

static const ftgc_table_io gdnc_table_io = {0, 0u, 0, 0};

/** One v-head: q | k | v, the carried state, the decay-ratio tile and the five gate columns. */
static bool read_one_head(ftq_context *ctx, const gdnc_endpoints *endpoints,
                          const ftgc_shape *shape, uint32_t head)
{
    const uint32_t tiles = shape->state_tiles;
    const uint32_t state = tiles * tiles;
    uint32_t in_write;
    uint32_t state_write;
    uint32_t decay_write;
    uint32_t scratch;
    uint32_t stream;

    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN, (int32_t)(3u * tiles), WAIT_MODE))
        return false;
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_STATE, (int32_t)state, WAIT_MODE))
        return false;
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_DECAY, 1, WAIT_MODE))
        return false;
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_VECTOR,
                                       (int32_t)kGdncDeltaVectorSpan, WAIT_MODE))
        return false;

    in_write = ft_tensix_cb_write_pointer(CB_IN);
    state_write = ft_tensix_cb_write_pointer(CB_STATE);
    decay_write = ft_tensix_cb_write_pointer(CB_DECAY);
    scratch = ft_tensix_cb_write_pointer(CB_SCRATCH);

    if (!ctx->aborting && gdnc_table) {
        /* The same header + run, looked up (its own barriers, before the big reads). */
        ftgc_table_head(&gdnc_table_io, &gdnc_table_state, scratch, head, gdnc_band,
                        gdnc_first_row, shape->chunk_rows);
    } else if (!ctx->aborting && gdnc_compact) {
        /* The compact header and this head's decay|beta first, behind their own barrier: the
         * build below needs them, and it runs while the big reads are in flight. */
        const uint32_t compact = scratch + COMPACT_SCRATCH_OFFSET;

        ft_tensix_noc_read(ft_tensix_noc_address(endpoints->constant_x, endpoints->constant_y,
                                                 endpoints->gate_base),
                           compact, FTGC_COMPACT_HEADER_WORDS * 4u);
        ft_tensix_noc_read(ft_tensix_noc_address(endpoints->constant_x, endpoints->constant_y,
                                                 endpoints->gate_base +
                                                     (FTGC_COMPACT_HEADER_WORDS +
                                                      head * FTGC_COMPACT_HEAD_WORDS) * 4u),
                           compact + FTGC_COMPACT_HEADER_WORDS * 4u,
                           FTGC_COMPACT_HEAD_WORDS * 4u);
        ft_tensix_noc_read_barrier();
    }
    if (!ctx->aborting) {
        /* q, then k, then v. THE GQA REPEAT IS AN ADDRESS and never a materialised tensor:
         * `ftgc_delta_head_tile` sends the q and k reads to k-head `vh % k_heads`, which is
         * the INTERLEAVED mapping the oracle uses. The blocked one is the plausible wrong
         * answer and has produced gibberish in this architecture before. */
        for (stream = 0; stream < 3u; stream++)
            ft_tensix_noc_read(
                ft_tensix_noc_address(endpoints->activation_x, endpoints->activation_y,
                                      endpoints->activation_in +
                                          ftgc_delta_head_tile(shape, stream, head) *
                                              TILE_BYTES),
                in_write + stream * tiles * TILE_BYTES, tiles * TILE_BYTES);

        ft_tensix_noc_read(
            ft_tensix_noc_address(endpoints->state_x, endpoints->state_y,
                                  endpoints->state_base +
                                      ftgc_state_tile(shape, head, 0u, 0u) * TILE_BYTES),
            state_write, state * TILE_BYTES);

        if (gdnc_compact) {
            const volatile uint32_t *header =
                (const volatile uint32_t *)(uintptr_t)(scratch + COMPACT_SCRATCH_OFFSET);
            const volatile uint32_t *run = header + FTGC_COMPACT_HEADER_WORDS;

            ftgc_compact_decay_tile(run, gdnc_first_row, shape->chunk_rows,
                                    (volatile uint32_t *)(uintptr_t)decay_write);
            ftgc_compact_gate(run, run + 32u, gdnc_first_row, shape->chunk_rows, header[0],
                              (volatile uint32_t *)(uintptr_t)scratch);
        } else {
            ft_tensix_noc_read(
                ft_tensix_noc_address(endpoints->constant_x, endpoints->constant_y,
                                      endpoints->constant_base +
                                          ftgc_delta_decay_tile(head) * TILE_BYTES),
                decay_write, TILE_BYTES);

            /* The gate block's run for this head. `kGdncGateStride * 4` is 640 bytes, a
             * multiple of the NoC's 64-byte granule by construction, so no rounding is needed
             * and no read runs past the head's own slot. */
            ft_tensix_noc_read(
                ft_tensix_noc_address(endpoints->constant_x, endpoints->constant_y,
                                      endpoints->gate_base + ftgc_gate_offset(head, 0u) * 4u),
                scratch, (uint32_t)kGdncGateStride * 4u);
        }
        ft_tensix_noc_read_barrier();

        for (stream = 0; stream < (uint32_t)kGdncGateRuns; stream++)
            push_gate_column(scratch, stream);
    } else {
        /* AN ABORT DRAINS. The compute thread is waiting on `kGdncDeltaVectorSpan` pages and
         * gets them; what it does not get is a barrier, and it will abort on its own words. */
        for (stream = 0; stream < (uint32_t)kGdncGateRuns; stream++)
            ft_tensix_cb_push_back(CB_VECTOR, 1);
    }

    ft_tensix_cb_push_back(CB_IN, 3u * tiles);
    ft_tensix_cb_push_back(CB_STATE, state);
    ft_tensix_cb_push_back(CB_DECAY, 1);
    return true;
}

/* ===========================================================================
 * Refusals
 * =========================================================================== */

/**
 * Does `cb_index` hold a whole number of `chunk_pages` blocks?
 *
 * THE GRID A CONSUMER INDEXES ACROSS, which is a different rule from the push grid and the
 * one the decode family did not have -- `ft_tensix_unpack_ab(cb, ..., tile)` resolves to
 * `read_pointer + tile * page_size` with NO WRAP, so a compute thread that waits for `n`
 * pages and then names tiles `0..n-1` needs the front low enough that the whole window fits.
 */
static bool index_grid_is_safe(uint32_t cb_index, uint32_t chunk_pages)
{
    const uint32_t pages = ft_tensix_cb(cb_index)->fifo_num_pages;

    return chunk_pages != 0u && pages >= chunk_pages && (pages % chunk_pages) == 0u;
}

static bool conv_is_executable(const ftgc_shape *shape)
{
    const uint32_t taps = shape->conv_taps;
    const uint32_t band = ftgc_band_tiles(shape);
    const uint32_t depth = conv_block_depth(shape);

    /* THE DIRECTION THAT WOULD OTHERWISE BE SILENT. Every check below asks whether the PLACED
     * BUFFERS CAN SERVE this image's depth, and a host that placed for a DEEPER block passes
     * all of them — it runs at THIS image's depth inside the host's placement and reports the
     * control's timing as the arm's. Strict equality against the declaration, on this
     * processor AND on the compute thread, so neither is left waiting on pages the other
     * decided not to send. See `kGdncByteConvDepth`. */
    if (!ftgc_conv_depth_agrees(shape->declared_conv_depth, (uint32_t)CONV_DEPTH))
        return false;
    /* A depth no block satisfies is a REFUSAL, not a fallback to one column. */
    if (depth == 0u)
        return false;

    /* Two BLOCKS on every streamed buffer, so this processor can fill one while the compute
     * thread drains the other. One block is a hard handshake per group. The block-sized
     * capacity and grid questions are `conv_block_depth()`'s and it has already asked them;
     * these are the depth-independent floors. */
    if (ft_tensix_cb(CB_IN)->fifo_num_pages < 2u)
        return false;
    if (ft_tensix_cb(CB_STATE)->fifo_num_pages < 2u)
        return false;
    if (ft_tensix_cb(CB_BAND)->fifo_num_pages < band)
        return false;
    if (ft_tensix_cb(CB_VECTOR)->fifo_num_pages < 2u * taps)
        return false;
    if (ft_tensix_cb(CB_PROD)->fifo_num_pages < 2u * band)
        return false;

    /* THE PUSH GRIDS AS A PAGE-COUNT QUESTION, NOT A POINTER ONE — THE SAME DEFECT PHASE
     * DELTA'S REFUSAL CARRIED, IN THE DIRECTION NOTHING EXERCISED.
     *
     * These four were `ft_tensix_cb_push_grid_is_safe()`, which tests the CURRENT write
     * pointer's alignment to the chunk grid — and this refusal runs BEFORE
     * `rebase_produced_buffers()`, so at this point every pointer is wherever the PREVIOUS
     * descriptor left it. PHASE DELTA's twin was corrected for exactly this reason on
     * 2026-08-21 and PHASE CONV's was not, because the suite's `stage_delta` is its LAST
     * stage: no CONV descriptor in it ever followed a DELTA one.
     *
     * MEASURED ON SILICON 2026-08-21, on the first caller that alternates them — a real
     * prefill runs CONV, DELTA, CONV, DELTA, one pair per SSM layer. A PHASE DELTA descriptor
     * pushes CB_BAND in groups of `kGdncDeltaConstTiles` = 4 into an 8-page buffer and leaves
     * the pointer at page 4; `4 % 8` is 4, so **EVERY PHASE CONV DESCRIPTOR AFTER ANY PHASE
     * DELTA DESCRIPTOR WAS REFUSED**. The host's re-arm-once recovery hid it as a number
     * rather than as a failure: 215 refusals and 215 residency re-arms over a 260-position
     * prefill, every one recovering, with PHASE CONV reading 4,502 us a descriptor against
     * the 681.4 us the same kernel measures in `tests/test_phase6_gdnc.c`.
     *
     * What the reader needs is that the page COUNT admits the group; the POINTER is
     * guaranteed by the rebase that runs immediately after this returns. */
    if (!index_grid_is_safe(CB_IN, depth) || !index_grid_is_safe(CB_STATE, depth))
        return false;

    /* What the compute thread INDEXES ACROSS: the whole band in one window, this BLOCK's
     * `depth*taps` weight tiles, its `depth` input tiles, its `depth` window tiles, and the
     * `2*taps` products it stages and reduces in one DST section PER COLUMN — CB_PROD keeps a
     * per-column grid at every depth, which is what leaves the DST accumulation argument
     * exactly where it was. */
    if (!index_grid_is_safe(CB_BAND, band) || !index_grid_is_safe(CB_VECTOR, depth * taps) ||
        !index_grid_is_safe(CB_PROD, band))
        return false;
    return true;
}

/**
 * PHASE DELTA's placement refusal.
 *
 * EVERY ONE OF THESE IS AN INDEXED-READ GRID and not a comfort setting. The compute thread
 * names tiles ABSOLUTELY off each buffer's front for the whole of one head and pops once at
 * the end of it, and `ft_tensix_unpack_ab(cb, ..., tile)` is `read_pointer + tile*page_size`
 * with NO WRAP -- so a buffer whose page count is not a multiple of the span reads a tile out
 * of the buffer entirely, at some head that is not the first.
 *
 * This processor is the only one of the three that can evaluate them: a page count comes off
 * `cb_interface`, which does not exist on the MATH thread and fails at LINK time there.
 */
static bool delta_is_executable(const ftgc_shape *shape)
{
    const uint32_t tiles = shape->state_tiles;
    const uint32_t input = 3u * tiles;
    const uint32_t state = tiles * tiles;
    const uint32_t work = (uint32_t)kGdncDeltaWorkPerRow * tiles;

    if (!index_grid_is_safe(CB_IN, input) || !index_grid_is_safe(CB_STATE, state) ||
        !index_grid_is_safe(CB_BAND, (uint32_t)kGdncDeltaConstTiles) ||
        !index_grid_is_safe(CB_VECTOR, (uint32_t)kGdncDeltaVectorSpan) ||
        !index_grid_is_safe(CB_CC, (uint32_t)kGdncDeltaCcSpan) ||
        !index_grid_is_safe(CB_WORK, work))
        return false;
    if (ft_tensix_cb(CB_DECAY)->fifo_num_pages < 1u)
        return false;
    if (ft_tensix_cb(CB_SCRATCH)->fifo_num_pages < 1u)
        return false;
    /* THE PUSH GRIDS TOO -- BUT AS A PAGE-COUNT QUESTION AND NOT AS A POINTER ONE, AND THAT
     * DISTINCTION COST A RUN.
     *
     * `ft_tensix_cb_push_grid_is_safe()` tests the CURRENT write pointer's alignment to the
     * chunk grid, and this refusal is made BEFORE `rebase_produced_buffers()` -- deliberately,
     * because rule 1 says a refused descriptor must leave no circular-buffer traffic behind
     * it. So at this point every write pointer is wherever the PREVIOUS DESCRIPTOR left it,
     * and asking whether THAT is aligned to THIS descriptor's group size is asking the wrong
     * question about the wrong descriptor.
     *
     * MEASURED ON SILICON 2026-08-21: a PHASE CONV descriptor pushes CB_IN one page at a time,
     * 256 of them into a 24-page buffer, and leaves the pointer at page 16. `16 % 12` is 4, so
     * the first PHASE DELTA descriptor after any CONV descriptor was refused with
     * `kErrBadParameters` -- the reader exited, the compute thread spun out its whole
     * 400,000,000-iteration bound on CB_BAND, and the triad retired clean with nothing wrong
     * with it. PHASE CONV never saw this because its own group is ONE page and every pointer
     * is one-page aligned.
     *
     * What the reader actually needs is that the page COUNT admits the group, which is
     * `index_grid_is_safe`; the POINTER is guaranteed by the rebase that runs immediately
     * after this returns. */
    if (!index_grid_is_safe(CB_DECAY, 1u) || !index_grid_is_safe(CB_SCRATCH, 1u))
        return false;
    /* CB_SCRATCH IS WRITTEN WITHOUT A RESERVATION, which is safe only while its write pointer
     * is provably at the base -- i.e. only if NOBODY EVER PUSHES IT. Nothing on the device can
     * check "nobody pushes"; what it can check is that the index is not one of the eight this
     * program does push, which is the same statement by exhaustion. */
    if (CB_SCRATCH == CB_IN || CB_SCRATCH == CB_STATE || CB_SCRATCH == CB_BAND ||
        CB_SCRATCH == CB_VECTOR || CB_SCRATCH == CB_PROD || CB_SCRATCH == CB_WORK ||
        CB_SCRATCH == CB_DECAY || CB_SCRATCH == CB_CC)
        return false;
    return true;
}

/**
 * Put this processor's write pointers back on their buffers' bases.
 *
 * A FIFO pointer wraps on an EXACT HIT of the limit and on nothing else, and a descriptor's
 * traffic is not a whole number of buffers: this one moves `col_count` CB_IN pages and
 * `taps * col_count` CB_VECTOR pages against page counts of 2 and `2*taps`. Descriptor N+1
 * would otherwise start wherever N left off, and the consumer's indexed reads are absolute
 * off its front.
 *
 * The whole-buffer reservation is the precondition `ft_tensix_cb_rebase_write_pointer()`
 * states: it does not move a pointer, it waits until nothing live is behind it.
 */
static bool rebase_produced_buffers(ftq_context *ctx)
{
    const uint32_t buffers[4] = { CB_IN, CB_STATE, CB_VECTOR, CB_DECAY };
    uint32_t index;

    for (index = 0; index < 4u; index++) {
        const uint32_t cb = buffers[index];

        if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb,
                                           (int32_t)ft_tensix_cb(cb)->fifo_num_pages,
                                           WAIT_MODE))
            return false;
        ft_tensix_cb_rebase_write_pointer(cb);
    }
    /* CB_BAND's traffic is `2*taps` into a `2*taps`-page buffer, i.e. an EXACT HIT, so it
     * returns to its base by arithmetic. Rebasing it anyway is what makes that a property of
     * the code rather than of the shape -- the same accidental agreement made the decode
     * family's CB_PROD look safe while CB_STATE and CB_VECTOR read out of their buffers for a
     * whole run. */
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_BAND,
                                       (int32_t)ft_tensix_cb(CB_BAND)->fifo_num_pages,
                                       WAIT_MODE))
        return false;
    ft_tensix_cb_rebase_write_pointer(CB_BAND);
    return true;
}

/**
 * The reader half of one GDNC descriptor.
 *
 * @return true if the whole descriptor was read; false if a wait was abandoned, in which case
 *         the caller must not retire it (rule 5).
 */
/**
 * kGdncFlagSerialize: wait until every descriptor before this one has PUBLISHED completion —
 * the writer publishes after its write barrier, so the previous descriptor's window / state
 * writes are in DRAM. Stop-aware and bounded; wrap-safe on the counts.
 */
static bool wait_previous_completed(ftq_context *ctx)
{
    uint32_t spins = 0u;

    for (;;) {
        ft_bh_l1_cache_invalidate();
        if ((int32_t)(ctx->control[kFieldCompletedCount] - ctx->cursor) >= 0)
            return true;
        if (ftq_stop_requested(ctx)) {
            ftq_abort(ctx, kErrNone);
            return false;
        }
        if (++spins >= ctx->cb_spin_limit) {
            ftq_abort(ctx, kErrCbWaitTimeout);
            return false;
        }
    }
}

static bool read_one_gdnc(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const ftgc_shape shape = ftgc_decode(parameters);
    const bool delta = shape.phase == (uint32_t)kGdncPhaseDelta;
    gdnc_endpoints endpoints;
    uint32_t index;

    if (TILE_BYTES == 0u || shape.tile_bytes != TILE_BYTES ||
        !ftgc_shape_is_supported(&shape) ||
        !(delta ? delta_is_executable(&shape) : conv_is_executable(&shape))) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }

    if ((parameters[kGdncFlags] & (uint32_t)kGdncFlagSerialize) != 0u &&
        !wait_previous_completed(ctx))
        return false;
    load_endpoints(&endpoints, parameters);
    gdnc_table = delta && (parameters[kGdncFlags] & (uint32_t)kGdncDeltaFlagTable) != 0u;
    gdnc_compact = gdnc_table ||
                   (delta && (parameters[kGdncFlags] & (uint32_t)kGdncDeltaFlagCompact) != 0u);
    gdnc_first_row = parameters[kGdncFirstRow];
    gdnc_band = parameters[kGdncDeltaBand];
    /* The table map needs the whole 4 KiB scratch page at 64 B (the DRAM granule). */
    if (gdnc_table && (TILE_BYTES < FTGT_SCRATCH_BYTES ||
                       (ft_tensix_cb_write_pointer(CB_SCRATCH) % 64u) != 0u)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    if (!rebase_produced_buffers(ctx))
        return false;
    /* A malformed table block aborts, and the heads below drain. */
    if (gdnc_table && !ctx->aborting &&
        !ftgc_table_begin(&gdnc_table_io, &gdnc_table_state,
                          ft_tensix_cb_write_pointer(CB_SCRATCH), parameters[kGdncConstantXy],
                          endpoints.gate_base, shape.head_first))
        ftq_abort(ctx, kErrBadParameters);

    if (delta) {
        if (!read_delta_constants(ctx, &endpoints))
            return false;
        for (index = 0; index < shape.head_count; index++)
            if (!read_one_head(ctx, &endpoints, &shape, shape.head_first + index))
                return false;
        return !ctx->aborting;
    }

    if (!read_band(ctx, &endpoints, &shape))
        return false;
    {
        /* The depth is re-derived rather than carried from the refusal, because the refusal is
         * the OTHER caller's and this one must not depend on having been through it. It is a
         * pure function of the shape and the placed page counts, both of which are fixed for
         * the descriptor. */
        const uint32_t depth = conv_block_depth(&shape);

        if (depth == 0u)
            return false;
        for (index = 0; index < shape.col_count; index += depth)
            if (!read_column_group(ctx, &endpoints, &shape, shape.first_col + index, depth))
                return false;
    }
    return !ctx->aborting;
}

/**
 * Zero CB_STATE, once, before any descriptor.
 *
 * Rows `taps-1..31` of the assembled window tile are multiplied by ZERO COLUMNS of the band
 * matrices, and `0 * NaN` is NaN. This processor writes only rows `0..taps-2` afterwards, so
 * one pass over the buffer makes the tail zero for the life of the residency.
 *
 * It RESERVES and does not PUSH: the write pointer only moves on a push, so the buffer is
 * still empty and still based when the first descriptor arrives. `taps` is not known here --
 * it is a per-descriptor word -- so the whole buffer is zeroed rather than only the tail,
 * which is 2,048 word stores once against ~237,000 a descriptor for the alternative.
 */
static bool zero_window_buffer(ftq_context *ctx)
{
    const uint32_t pages = ft_tensix_cb(CB_STATE)->fifo_num_pages;
    uint32_t *base;
    uint32_t words;
    uint32_t index;

    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_STATE, (int32_t)pages, WAIT_MODE))
        return false;
    base = (uint32_t *)(uintptr_t)ft_tensix_cb_write_pointer(CB_STATE);
    words = (pages * TILE_BYTES) / 4u;
    for (index = 0; index < words; index++)
        base[index] = 0u;
    return true;
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

    /* RESUME rather than restart. All five processors resume from completed_count, the ONE
     * number that means "every stage finished this descriptor". */
    if (ctx.control[kFieldTail] != ctx.control[kFieldCompletedCount]) {
        ctx.error = kErrBadParameters;
        ctx.control[kFieldError] = ctx.error;
        ctx.control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    if (!zero_window_buffer(&ctx)) {
        ctx.control[kFieldError] = ctx.error;
        ctx.control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    ctx.cursor = ctx.control[kFieldCompletedCount];

    while (running) {
        volatile FTQ_L1_PTR uint32_t *descriptor;
        uint32_t opcode;
        bool done;

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

        descriptor = ftq_descriptor(&ctx);
        opcode = descriptor[kDescOpcode];

        ctx.control[kFieldLastOpcode] = opcode;
        ftq_status_opcode(&ctx, opcode);

        if (opcode == kOpGdnChunk) {
            ftq_status_set(&ctx, kStateCbReserveBack);
            done = read_one_gdnc(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
            if (!done)
                running = false;
        } else if (opcode == kOpNop) {
            done = true;
        } else if (opcode == kOpStop) {
            done = true;
            running = false;
        } else {
            /* A binary is bound to one slab, so an opcode another program implements is
             * genuinely not available here. Refusing names that; executing nothing would
             * retire work that never happened. */
            ctx.error = kErrBadOpcode;
            done = false;
            running = false;
        }

        /* Rule 5. An abandoned descriptor was handed to nobody, so it advances NOTHING. */
        if (done) {
            ctx.cursor++;
            ctx.control[kFieldTail] = ctx.cursor;
        }
    }

    ft_tensix_noc_read_barrier();
    ft_tensix_noc_write_barrier();
    ctx.control[kFieldError] = ctx.error;
    ctx.control[kFieldState] = kStateExit;
    ftq_status_set(&ctx, kStateExit);
}
