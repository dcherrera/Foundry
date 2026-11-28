/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_gdn_writer.c
 * @brief Writer third of the RESIDENT GATED-DELTA-NET triad: the processor that PUBLISHES
 *   COMPLETION, and the one that PUTS THE CARRIED STATE BACK.
 *
 * ===========================================================================
 * WHY THE CARRIED STATE LEAVES THROUGH HERE AND NOT THROUGH THE READER
 * ===========================================================================
 * This is the only op in the backend whose operands are READ AND WRITTEN, and the obvious
 * shape -- the reader, which already has the state's address, writing it back -- is wrong.
 * Rule 1 says completion is published by the LAST processor in the pipeline, when the result
 * has left L1. The updated state IS a result. If the reader wrote it, the result would leave
 * L1 under the processor that owns `tail` while completion was published by the processor
 * that owns `completed_count`: two writers for one result, which is exactly the lie rules 1
 * and 5 exist to make impossible. So the state goes out through CB_OUT, like every other
 * result in this backend, and the reader owns `tail` and nothing else.
 *
 * ===========================================================================
 * THE SPLIT IS RE-DERIVED FROM THE SHAPE WORD, NOT SENT
 * ===========================================================================
 * `FT_TT_QUEUE_PARAM_WORDS` is 16 for EVERY family and a seventeenth word changes the shared
 * slot stride and breaks on the card, so there is no room for a "where the state ends" word.
 * There is no need for one either: `v_heads` and `state_tiles` are already in the shape word.
 *
 *   DELTA  per v-head h, per tile-row jt: `state_tiles` updated state tiles, then ONE y
 *          column. `v_heads * state_tiles * (state_tiles + 1)` tiles in all.
 *   CONV   `conv_tile_cols` conv_out tiles, then `conv_tile_cols * (taps-1)` window tiles.
 *
 * THE DELTA ORDER IS INTERLEAVED PER TILE-ROW AND THE ROUND'S SPEC SAID "AFTER ALL HEADS",
 * WHICH IS NOT IMPLEMENTABLE. Holding every head's y until the last head's state had gone out
 * means `v_heads * state_tiles` column tiles resident in L1 -- 512 KiB at the shipped shape,
 * against a whole-core L1 budget that has to hold nine circular buffers as well. The compute
 * thread emits y the instant it has it, which is the end of the tile-row that produced it.
 * The split RULE is unchanged: `v_heads * state_tiles * state_tiles` tiles go to
 * (kGdnStateXy, kGdnStateRecurrent) and the rest to (kGdnActivationXy, kGdnActivationOut).
 * Only their interleaving differs, and this file derives both from the shape.
 *
 * ===========================================================================
 * y IS TRANSPOSED HERE, AND IT COSTS TWO WRITES A TILE
 * ===========================================================================
 * y arrives from the (e) reduce as a COLUMN: head h's tile-row jt holds y[32*jt + j] at
 * [j][0]. y in DRAM is a ROW-0 STRIP, head-major, so element [j][0] belongs at [0][j] of
 * output tile `h*state_tiles + jt`. That is the inverse of resident_attention_reader.c's Q
 * pack and the same shape as resident_attention_writer.c's scatter.
 *
 * THE DESTINATION ROW USED TO BE `h % 32` AND THE TILE `(h/32)*state_tiles + jt` -- y was a
 * tilized [v_heads, S] matrix, one row per head, 4 tiles at Ornith's shape against 128 now.
 * The write COUNT is identical either way (two 64-byte face rows per column tile, 256 a
 * descriptor); what the strip buys is that y is then the same geometry as `ssm_z` and as every
 * other activation in the backend, so the gated norm and the `silu(z) *` after it run on the
 * card against operands nobody has to re-tilize on the host.
 *
 * IT STAGES THROUGH THE CB PAGE ITSELF rather than through a local array, and that is not
 * cleverness for its own sake: a NoC write's source must be L1, and a data-movement kernel's
 * private .data/.bss is a per-core LOCAL memory the NoC cannot read. The 32 source values are
 * column 0 of every row, which lives entirely in faces 0 and 2; faces 1 and 3 of the same
 * page are therefore free scratch, and the writer owns the page until it pops it. So the
 * gather runs into face 1 and face 3 and each becomes ONE 64-byte face-row write. Two writes
 * and never one per element -- a barrier is a per-core drain.
 *
 * ===========================================================================
 * THE BARRIER IS PER STREAM GROUP, AND THE POPS MOVE BEHIND IT
 * ===========================================================================
 * PHASE DELTA used to take one write barrier PER TILE -- 640 a descriptor at Ornith's 32
 * v-heads -- against this family's own "one barrier per stream group, never per tile" rule,
 * the rule that makes the block reader's barrier worth 5.9x. `WRITER_RUNG` (compile-time arg
 * 3) groups them: rung 0 is the per-tile sequence that shipped, rung 1 one barrier per
 * TILE-ROW, rung 2 one per HEAD. `ftg_writer_block()` in resident_gdn_protocol.h is the ONE
 * implementation, shared with `resolve_cb_pages()` on the host, because the host is what
 * sizes CB_OUT to hold two groups.
 *
 * THE POPS ARE WHAT MAKES IT CORRECT, not the barrier count. Every write's source is the
 * CB_OUT page, so a page popped with its write still in flight is a page the compute thread
 * may pack over. So a group is: wait for `group` pages, issue `group` writes, ONE barrier,
 * then `group` pops -- and the pops are ONE PAGE AT A TIME, and the sources are addressed
 * with the wrap SPELLED OUT (`page_address()`), because a FIFO pointer returns to its base
 * on an EXACT HIT of the limit and on nothing else. Doing either the upstream way would put
 * this buffer on a page grid that neither phase's traffic closes on. With both spelled out
 * the group divides nothing and CB_OUT needs no rebase.
 *
 * PHASE CONV IS RUNG 0 ON EVERY ARM of this round -- it drains the same CB_OUT, so a CONV
 * descriptor that does not move is the measurement that says the resize alone bought nothing.
 *
 * A 64-BYTE WRITE TO DRAM AT A 64-BYTE-ALIGNED ADDRESS LANDS AND LEAVES ITS NEIGHBOURS ALONE.
 * That is measured, not assumed: it is the same write KV_APPEND makes, checked on silicon
 * against a poisoned region with 0 bytes of an untouched neighbour disturbed.
 *
 * WHY IT FETCHES ON `head` BUT PUBLISHES ON `tail`: fetching on `tail` is a pipeline
 * deadlock, not a precaution -- the output CB fills, compute blocks in cb_reserve_back, the
 * reader blocks because nothing drains the inputs, and the reader can then never reach the
 * end of the descriptor to bump `tail`.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` -- no tt-metal include (plan Phase 6.6).
 * Compile-time args:
 *   0: CB_OUT index   1: tile size in bytes
 *   2: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 *   3: PHASE DELTA writer rung -- 0 a barrier per tile (ships), 1 per tile-row, 2 per head
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 * Descriptor words: a3/a4 the S1 group pull on member 0 of a split descriptor
 *   (resident_group_pull.h); a5 the group sequence, stored in control word 56 before
 *   completion is published.
 *
 * Build (the whole program; this file is the NCRISC/NoC 1 third):
 *   RELOCATE_KERNEL_TEXT=1 \
 *     READER_CTARGS=0,1,2,3,4,4096,1,2,8,7,4,1 COMPUTE_CTARGS=0,1,2,3,5,6,7,16,1,2,8,4,1 \
 *     WRITER_CTARGS=16,4096,1,0 CB_TABLE=kernels/cb_tables/gdn_float32.cbtable \
 *     bash foundry/tt/build_tensix_program.sh \
 *       kernels/resident_gdn_reader.c kernels/resident_gdn_compute.c \
 *       kernels/resident_gdn_writer.c programs/gdn
 *
 * @status EXECUTED ON SILICON 2026-08-18. Both sinks land: the CONV split writes
 *   `conv_tile_cols` conv_out tiles and then the rolled window, and the window's newest plane
 *   reads back equal to qkv with 0 of 8192 channels differing.
 *
 *   IT FOUND THE FAMILY'S FIRST WRONG ANSWER, from the other side. `load_sinks()` takes ONE
 *   packed coordinate for both activation sinks (`kGdnActivationXy`), so when the host staged
 *   the pair in different DRAM banks this processor wrote every answer at the output's offset
 *   INSIDE THE INPUT'S BANK -- a clean retire, a right completion count, and an output buffer
 *   that was never touched. Nothing on this side can detect that; the host refuses it now.
 * @issues PHASE CONV still takes one write barrier per tile -- 1,024 a descriptor -- and
 *   resident_rope_writer.c and resident_attention_writer.c still do too. CONV is held at
 *   rung 0 deliberately this round, as the built-in null for the CB_OUT resize; raising it
 *   is the next item and belongs in a round that measures it separately.
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_gdn_protocol.h"
#include "resident_group_pull.h"
#include "resident_queue_protocol.h"

#define CB_OUT ft_tensix_compile_time_arg(0)
#define TILE_BYTES ft_tensix_compile_time_arg(1)
#define WAIT_MODE ft_tensix_compile_time_arg(2)
/** PHASE DELTA's barrier group, as a RUNG -- see ftg_writer_block() in the protocol header.
 *  A build line short of this arg reads a literal 0 with no diagnostic, and 0 is the per-tile
 *  sequence that shipped, so a short line here is slow-and-correct rather than wrong. */
#define WRITER_RUNG ft_tensix_compile_time_arg(3)

/** Rows of one face, and rows of one tile: fixed by gdn_float32.cbtable. */
enum { kGdnFaceRows = 16 };
enum { kGdnTileRows = 32 };

/** The two DRAM banks a descriptor writes: the carried state and the activation out. */
typedef struct gdn_sinks {
    uint32_t state_x;
    uint32_t state_y;
    uint32_t state_base;
    uint32_t activation_x;
    uint32_t activation_y;
    uint32_t activation_base;
} gdn_sinks;

static void load_sinks(gdn_sinks *sinks, const ftg_shape *shape,
                       volatile FTQ_L1_PTR uint32_t *parameters)
{
    sinks->state_x = ftg_xy_x(parameters[kGdnStateXy]);
    sinks->state_y = ftg_xy_y(parameters[kGdnStateXy]);
    /* kGdnFlagStateOut: the state goes to the OTHER phase's (unused) state word. */
    sinks->state_base =
        ((shape->phase == (uint32_t)kGdnPhaseConv) !=
         ((parameters[kGdnFlags] & (uint32_t)kGdnFlagStateOut) != 0u))
            ? parameters[kGdnStateConv]
            : parameters[kGdnStateRecurrent];
    sinks->activation_x = ftg_xy_x(parameters[kGdnActivationXy]);
    sinks->activation_y = ftg_xy_y(parameters[kGdnActivationXy]);
    sinks->activation_base = parameters[kGdnActivationOut];
}

/* Face arithmetic, identical to the reader's. Spelled locally rather than in the protocol
 * header, which is the WIRE contract and carries no L1 layout. */

static uint32_t ftg_face_bytes(void)
{
    return TILE_BYTES / 4u;
}

static uint32_t ftg_face_row_bytes(void)
{
    return TILE_BYTES / 64u;
}

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

/** One whole tile, ISSUED. The barrier belongs to the group, not to the tile. */
static void issue_whole_tile(uint32_t source, uint32_t x, uint32_t y, uint32_t address)
{
    ft_tensix_noc_write(source, ft_tensix_noc_address(x, y, address), TILE_BYTES);
}

/** One whole tile, issued and drained. PHASE CONV's form, and rung 0's. */
static void write_whole_tile(uint32_t source, uint32_t x, uint32_t y, uint32_t address)
{
    issue_whole_tile(source, x, y, address);
    ft_tensix_noc_write_barrier();
}

/** M9: kGdnFlagConvStrip — PHASE CONV writes only row 0 of each tile (faces 0-1). */
static bool gdn_conv_strip;
/** S2c: the row this descriptor's stream rows live in (kGdnRowShift). */
static uint32_t gdn_row;

static void write_conv_tile(uint32_t source, uint32_t x, uint32_t y, uint32_t address);

/** S2c: one CONV OUTPUT tile under the row select — L1 row 0 to DRAM row `gdn_row`. */
static void write_conv_out_tile(uint32_t source, uint32_t x, uint32_t y, uint32_t address)
{
    uint32_t half;

    if (gdn_row == 0u) {
        write_conv_tile(source, x, y, address);
        return;
    }
    for (half = 0; half < 2u; half++)
        ft_tensix_noc_write(source + ftg_face_row_offset(0u, half),
                            ft_tensix_noc_address(x, y, address + ftg_face_row_offset(gdn_row, half)),
                            ftg_face_row_words() * (TILE_BYTES / 1024u));
    ft_tensix_noc_write_barrier();
}

static void write_conv_tile(uint32_t source, uint32_t x, uint32_t y, uint32_t address)
{
    uint32_t half;

    if (!gdn_conv_strip) {
        write_whole_tile(source, x, y, address);
        return;
    }
    for (half = 0; half < 2u; half++) {
        const uint32_t offset = ftg_face_row_offset(0u, half);

        ft_tensix_noc_write(source + offset, ft_tensix_noc_address(x, y, address + offset),
                            ftg_face_row_words() * (TILE_BYTES / 1024u));
    }
    ft_tensix_noc_write_barrier();
}

/**
 * L1 address of page `index` of a group whose first page is at `front`.
 *
 * THE WRAP IS SPELLED OUT ON PURPOSE. `front + index * page` is what every indexed CB read
 * in this backend does, and a FIFO pointer returns to its base on an EXACT HIT of the limit
 * and on NOTHING ELSE -- so the indexed form reads straight past the buffer whenever a group
 * straddles the end, which is the whole of what cost this family its first wrong answer,
 * from the consumer side. Spelling the wrap here means the group size divides nothing: no
 * page grid, no rebase, and a rung may be changed without re-deriving either. ONE
 * subtraction is enough because `index` is always below the page count, so the offset is
 * always below `fifo_size`.
 */
static uint32_t page_address(uint32_t front, uint32_t index)
{
    uint32_t address = front + index * TILE_BYTES;

    if (address >= ft_tensix_cb(CB_OUT)->fifo_limit)
        address -= ft_tensix_cb(CB_OUT)->fifo_size;
    return address;
}

/**
 * CB_OUT pages behind one write barrier, for THIS descriptor.
 *
 * CLAMPED TO HALF THE PLACED BUFFER, and that clamp is the guard rather than a comment. The
 * host sizes CB_OUT for the rung this family is built at, but the two sides are compiled
 * separately: a kernel at rung 2 under a host that never grew CB_OUT would otherwise wait
 * for more pages than the buffer can ever hold, spin to its bound and REFUSE the descriptor
 * -- which is exactly what `reserve_back(cb_out, 4)` did against a two-page buffer on
 * silicon 2026-08-19. With the clamp it degrades to the shipped per-tile sequence instead.
 * Half, not all: the other half is what the compute thread fills while this one drains.
 */
static uint32_t writer_block(const ftg_shape *shape)
{
    const uint32_t pages = ft_tensix_cb(CB_OUT)->fifo_num_pages;
    uint32_t block = ftg_writer_block(shape->state_tiles, WRITER_RUNG);

    if (block > pages / 2u)
        block = pages / 2u;
    if (block == 0u)
        block = 1u;
    return block;
}

/**
 * One y column tile, transposed into row `row` of output tile `out_tile`, ISSUED.
 *
 * The gather runs into faces 1 and 3 of the SOURCE page, which hold columns 16..31 and are
 * therefore not sources -- every source is column 0. Two 64-byte writes, and the barrier
 * belongs to the group.
 *
 * THE FENCE IS NOT DECORATION. This is the writer's only READ of L1 the compute thread wrote,
 * and it used to sit immediately behind the previous tile's write barrier, whose own
 * `invalidate_l1_cache()` ordered it by accident. Grouping the barrier removes that
 * accident, so the ordering is stated here where the read is.
 */
static void issue_y_tile(uint32_t source, const gdn_sinks *sinks, uint32_t out_tile,
                         uint32_t row)
{
    const uint32_t words = ftg_face_row_words();
    uint32_t half;

    ft_bh_l1_cache_invalidate();
    for (half = 0; half < 2u; half++) {
        const uint32_t staging = source + (2u * half + 1u) * ftg_face_bytes();
        uint32_t index;

        for (index = 0; index < words; index++)
            *(uint32_t *)(uintptr_t)(staging + index * (TILE_BYTES / 1024u)) =
                *(const uint32_t *)(uintptr_t)(source +
                                               ftg_element_offset(half * (uint32_t)kGdnFaceRows +
                                                                      index,
                                                                  0u));
        ft_tensix_noc_write(staging,
                            ft_tensix_noc_address(sinks->activation_x, sinks->activation_y,
                                                  sinks->activation_base +
                                                      out_tile * TILE_BYTES +
                                                      ftg_face_row_offset(row, half)),
                            words * (TILE_BYTES / 1024u));
    }
}

/**
 * The writer half of one DELTA descriptor.
 *
 * Trip counts come from the SHAPE, never from a wait and never from `kGdnTileCount`, so this
 * processor stays in lockstep with the compute thread whatever the host wrote.
 */
static bool write_one_delta(ftq_context *ctx, const gdn_sinks *sinks, const ftg_shape *shape,
                            volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t tiles = shape->state_tiles;
    /* The stream group the compute thread emits and this one drains: `tiles` state pages
     * then ONE y page, `tiles` times over, per head and verify row. A page's DESTINATION is
     * decoded from its position in that sequence, which is what lets a barrier group span
     * tile-rows without the loop knowing which rung it is running at. */
    const uint32_t row_pages = tiles + 1u;
    const uint32_t head_pages = tiles * row_pages;
    const uint32_t block = writer_block(shape);
    /* MULTI-ROW: row r's state goes to EXTENSION word r and its y to row `row_first + r`. */
    volatile FTQ_L1_PTR uint32_t *extension = parameters + kGdnWords;
    uint32_t head, row;

    /* THIS MEMBER'S HEADS. The state region and the y row this loop addresses are both
     * derived from the ABSOLUTE head index, so a slice writes exactly the DRAM its member
     * owns and two members never overlap: state is head-major, and y is one ROW of a shared
     * tile per head, written at row granularity. */
    for (head = shape->head_first; head < shape->head_first + shape->head_count; head++)
        for (row = 0; row < shape->row_count; row++) {
            const uint32_t state_base = shape->row_count > 1u ? extension[row] : sinks->state_base;
            const uint32_t y_row = shape->row_first + row;
            uint32_t emitted = 0u;

            while (emitted < head_pages) {
                uint32_t group = head_pages - emitted;
                uint32_t index;

                if (group > block)
                    group = block;

                /* THE ABORT PATH IS THE SHIPPED ONE: a wait that gives up issues nothing and
                 * still pops, so the producer drains rather than deadlocking, and the barrier
                 * is skipped because there is then nothing outstanding to drain. */
                if (ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, (int32_t)group,
                                                WAIT_MODE)) {
                    const uint32_t front = ft_tensix_cb_read_pointer(CB_OUT);

                    for (index = 0; index < group; index++) {
                        const uint32_t page = emitted + index;
                        const uint32_t tile_row = page / row_pages;
                        const uint32_t slot = page % row_pages;
                        const uint32_t source = page_address(front, index);

                        if (slot < tiles)
                            issue_whole_tile(source, sinks->state_x, sinks->state_y,
                                             state_base +
                                                 ftg_state_tile(shape, head, tile_row, slot) *
                                                     TILE_BYTES);
                        else
                            issue_y_tile(source, sinks, ftg_y_tile(shape, head, tile_row),
                                         y_row);
                    }
                    ft_tensix_noc_write_barrier();
                }
                /* ONE PAGE AT A TIME, for the reason page_address() states: a multi-page pop
                 * advances the read pointer by the whole group and wraps only on an exact
                 * hit, so it would put this buffer on a grid neither phase's traffic closes
                 * on. */
                for (index = 0; index < group; index++)
                    ft_tensix_cb_pop_front(CB_OUT, 1);
                emitted += group;
            }
        }
    return !ctx->aborting;
}

/**
 * The writer half of one CONV descriptor: the convolved stream, then the rolled window.
 *
 * Both are whole tiles in the layout they arrived in, which is what makes the roll
 * self-consistent from one token to the next.
 */
static bool write_one_conv(ftq_context *ctx, const gdn_sinks *sinks, const ftg_shape *shape,
                           uint32_t state_base)
{
    /* THIS MEMBER'S COLUMNS, in the order the compute thread produces them: conv_out for
     * columns [col_first, +col_count), then the rolled window plane by plane over the same
     * columns, each at its ABSOLUTE tile — `plane * conv_tile_cols + column`. Unsliced, this is
     * the linear walk it replaced, tile for tile. */
    const uint32_t cols = shape->conv_tile_cols;
    const uint32_t first = shape->col_first;
    const uint32_t count = shape->col_count;
    uint32_t plane;
    uint32_t index;

    for (index = 0; index < count; index++) {
        if (ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE))
            write_conv_out_tile(ft_tensix_cb_read_pointer(CB_OUT), sinks->activation_x,
                                sinks->activation_y,
                                sinks->activation_base + (first + index) * TILE_BYTES);
        ft_tensix_cb_pop_front(CB_OUT, 1);
    }
    for (plane = 0; plane + 1u < shape->conv_taps; plane++) {
        for (index = 0; index < count; index++) {
            if (ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE))
                write_conv_tile(ft_tensix_cb_read_pointer(CB_OUT), sinks->state_x,
                                 sinks->state_y,
                                 state_base + (plane * cols + first + index) * TILE_BYTES);
            ft_tensix_cb_pop_front(CB_OUT, 1);
        }
    }
    return !ctx->aborting;
}

/** CONV ROW BATCH: rows 0 .. `rows`-1 of one page to DRAM rows `dram_row`.. of the tile at
 *  `address`. Consecutive rows of one face are contiguous at both ends, so ONE write a face
 *  half, under ONE barrier. */
static void write_batch_rows(uint32_t source, uint32_t x, uint32_t y, uint32_t address,
                             uint32_t dram_row, uint32_t rows)
{
    uint32_t half;

    for (half = 0; half < 2u; half++)
        ft_tensix_noc_write(source + ftg_face_row_offset(0u, half),
                            ft_tensix_noc_address(x, y,
                                                  address + ftg_face_row_offset(dram_row, half)),
                            rows * ftg_face_row_words() * (TILE_BYTES / 1024u));
    ft_tensix_noc_write_barrier();
}

/** CONV ROW BATCH: page row r of a window tile to row 0 of the same tile in row r's slot
 *  (EXTENSION word r), under ONE barrier. */
static void write_batch_window(uint32_t source, uint32_t x, uint32_t y, uint32_t tile_offset,
                               volatile FTQ_L1_PTR uint32_t *extension, uint32_t rows)
{
    uint32_t row, half;

    for (row = 0; row < rows; row++)
        for (half = 0; half < 2u; half++)
            ft_tensix_noc_write(source + ftg_face_row_offset(row, half),
                                ft_tensix_noc_address(x, y,
                                                      extension[row] + tile_offset +
                                                          ftg_face_row_offset(0u, half)),
                                ftg_face_row_words() * (TILE_BYTES / 1024u));
    ft_tensix_noc_write_barrier();
}

/**
 * The writer half of one CONV ROW BATCH descriptor (resident_gdn_protocol.h): ONE pass of
 * pages, as write_one_conv() drains them, each page carrying `rows` rows. Output row r goes to
 * DRAM row `row_first + r`; window row r to row r's slot. Pops every page even after an abort.
 */
static bool write_one_conv_batch(ftq_context *ctx, const gdn_sinks *sinks,
                                 const ftg_shape *shape, volatile FTQ_L1_PTR uint32_t *extension,
                                 uint32_t rows)
{
    const uint32_t cols = shape->conv_tile_cols;
    const uint32_t first = shape->col_first;
    const uint32_t count = shape->col_count;
    uint32_t plane;
    uint32_t index;

    for (index = 0; index < count; index++) {
        if (ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE))
            write_batch_rows(ft_tensix_cb_read_pointer(CB_OUT), sinks->activation_x,
                             sinks->activation_y,
                             sinks->activation_base + (first + index) * TILE_BYTES,
                             shape->row_first, rows);
        ft_tensix_cb_pop_front(CB_OUT, 1);
    }
    for (plane = 0; plane + 1u < shape->conv_taps; plane++) {
        for (index = 0; index < count; index++) {
            if (ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE))
                write_batch_window(ft_tensix_cb_read_pointer(CB_OUT), sinks->state_x,
                                   sinks->state_y, (plane * cols + first + index) * TILE_BYTES,
                                   extension, rows);
            ft_tensix_cb_pop_front(CB_OUT, 1);
        }
    }
    return !ctx->aborting;
}

/**
 * The writer half of one GDN descriptor.
 *
 * The refusal is the same one the reader and the compute thread make, on the same words,
 * before any circular buffer is touched -- so a refused descriptor leaves nobody waiting.
 * `kGdnTileCount` is checked against the STATE part only, per the family contract: a count
 * that cannot even cover the carried state is a descriptor the host built wrong, and running
 * it would leave the recurrence half updated.
 */
static bool write_one_gdn(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const ftg_shape shape = ftg_decode(parameters);
    gdn_sinks sinks;

    if (!ftg_shape_is_supported(&shape) || TILE_BYTES == 0u || shape.tile_bytes != TILE_BYTES) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    if (shape.phase == (uint32_t)kGdnPhaseDelta) {
        if (shape.tile_count < shape.head_count * ftg_state_tiles_per_head(&shape)) {
            ftq_abort(ctx, kErrBadParameters);
            return false;
        }
    } else if (shape.tile_count < shape.conv_tile_cols * (shape.conv_taps - 1u)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }

    load_sinks(&sinks, &shape, parameters);
    gdn_conv_strip = (parameters[kGdnFlags] & (uint32_t)kGdnFlagConvStrip) != 0u;
    gdn_row = (parameters[kGdnFlags] >> kGdnRowShift) & (uint32_t)kGdnRowMask;
    if ((gdn_row != 0u || shape.row_count > 1u) && shape.phase == (uint32_t)kGdnPhaseConv &&
        !gdn_conv_strip) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    if (!ftg_conv_batch_is_supported(&shape, parameters[kGdnFlags])) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    if (shape.phase == (uint32_t)kGdnPhaseDelta)
        return write_one_delta(ctx, &sinks, &shape, parameters);
    if (ftg_conv_batch_rows(parameters[kGdnFlags]) != 0u) {
        const uint32_t waited = ctx->status ? ctx->status[kStatusPayload] : 0u;
        const bool done = write_one_conv_batch(ctx, &sinks, &shape, parameters + kGdnWords,
                                               ftg_conv_batch_rows(parameters[kGdnFlags]));

        if (ctx->status)
            ctx->control[61] += ctx->status[kStatusPayload] - waited;
        return done;
    }
    {
        /* M9: the writer's CB-wait cycles during CONV (control word 61). */
        const uint32_t waited = ctx->status ? ctx->status[kStatusPayload] : 0u;
        /* MULTI-ROW CONV (PM 6): row r's output to row `row_first + r`, its rolled window to
         * EXTENSION word r. One row: the shipped walk. */
        volatile FTQ_L1_PTR uint32_t *extension = parameters + kGdnWords;
        uint32_t row;
        bool done = true;

        /* EVERY row even after an abort: write_one_conv() pops what it cannot write, so the
         * compute thread's later rows drain instead of blocking on CB_OUT (rule 2). */
        for (row = 0; row < shape.row_count; row++) {
            gdn_row = shape.row_first + row;
            done = write_one_conv(ctx, &sinks, &shape,
                                  shape.row_count > 1u ? extension[row] : sinks.state_base) &&
                   done;
        }
        gdn_row = shape.row_first;

        if (ctx->status)
            ctx->control[61] += ctx->status[kStatusPayload] - waited;
        return done;
    }
}

void kernel_main(void)
{
    ftq_context ctx;
    uint32_t completed;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);

    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    completed = ctx.control[kFieldCompletedCount];
    ctx.cursor = completed;
    ctx.control[61] = 0u;

    while (running) {
        // FETCH on `head`, so the writer drains the output CB while the reader is still
        // filling the input ones. Publishing waits on `tail`, below.
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);

        volatile FTQ_L1_PTR uint32_t *descriptor = ftq_descriptor(&ctx);
        const uint32_t opcode = descriptor[kDescOpcode];
        const uint32_t token = descriptor[kDescToken];
        const uint32_t group_sequence = descriptor[kDescA5];
        bool done = true;

        ftq_status_opcode(&ctx, opcode);
        if (opcode == kOpGdn) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            done = write_one_gdn(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
        } else if (opcode == kOpStop) {
            running = false;
        }

        if (!done) {
            running = false;
            continue;
        }

        // THE PRECONDITION, and the only place it belongs: the reader must have finished
        // this descriptor before its completion is published.
        if (!ftq_wait_until_reaches(&ctx, kFieldTail, ctx.cursor + 1u)) {
            running = false;
            continue;
        }

        ctx.cursor++;
        /* S1 GROUP PULL (resident_group_pull.h): on member 0 of a split descriptor, a3 names
         * the members and a4 the sequence they must reach; it publishes only after every one
         * of them has. a3 = 0 (every unsplit descriptor) skips it. */
        {
            const uint32_t pull = descriptor[kDescA3];

            if (opcode == kOpGdn && (pull & ~kGroupPullFlags) > 1u &&
                (pull & kPullRedSkip) == 0u &&
                !ftq_group_pull(&ctx, pull & ~kGroupPullFlags, descriptor[kDescA4])) {
                running = false;
                continue;
            }
        }
        // The done-sequence a member 0 pull reads (control word 56), before the count.
        ctx.control[kFieldGroupDone] = group_sequence;
        // Token before count: a host that sees the count is entitled to trust the token it
        // reads next. Both are plain stores from ONE in-order RISC-V, which is the only
        // ordering argument this protocol makes and the only one it needs.
        ctx.control[kFieldCompletedToken] = token;
        ctx.control[kFieldCompletedCount] = ++completed;
        ftq_group_post(&ctx, group_sequence, token);
    }

    // The writer does NOT touch kFieldError or kFieldState: the reader owns those, and one
    // writer per word is the whole ordering argument.
    ftq_status_set(&ctx, kStateExit);
}
