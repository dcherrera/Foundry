/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_gdnc_writer.c
 * @brief Writer third of the RESIDENT CHUNKWISE GATED-DELTA-NET triad: the processor that
 *   PUBLISHES COMPLETION, and the one that ROLLS THE WINDOW ACROSS THE CHUNK BOUNDARY.
 *
 * ===========================================================================
 * THE ROLL IS THE CHUNK-BOUNDARY CARRY, AND IT IS THIS FILE
 * ===========================================================================
 * `resident_gdn_writer.c` rolls a ONE-POSITION window by copying whole tiles: plane t gets
 * plane t+1 and the newest plane gets the raw input, `conv_tile_cols * (taps-1)` whole-tile
 * writes a descriptor. A C-position chunk cannot do that, because the `taps-1` values the
 * next chunk needs are the LAST `taps-1` ROWS of this chunk's input -- they are not planes,
 * they are rows of one tile.
 *
 * So the compute thread sends the RAW input tile out behind each conv_out tile, and this
 * processor writes ROW `chunk_rows - (taps-1) + p` of it into ROW 0 of window plane `p`.
 * Row 0 of a Float32 tile is TWO 64-BYTE PIECES 1,024 B apart, and 64 bytes is exactly the
 * DRAM write granule, so a plane costs TWO writes and a column costs `2*(taps-1)`. The
 * source is the CB_OUT page itself -- a NoC write's source must be L1, and a data-movement
 * kernel's private `.data`/`.bss` is a per-core LOCAL memory the NoC cannot read.
 *
 * **THE WINDOW COMES OUT IN THE DECODE FAMILY'S EXACT LAYOUT**, `taps-1` tap-major ROW-0
 * planes of `conv_tile_cols` tiles, which is what makes a prompt that prefills in chunks and
 * then decodes one token at a time ONE sequence with no transcription step. That is a
 * property of this file: a different roll destination would have been cheaper to write and
 * would have cost a 98 KiB-a-layer round trip at the prefill/decode seam.
 *
 * `chunk_rows >= taps-1` IS THE PROTOCOL'S REFUSAL and it is what makes the source row
 * non-negative. A shorter chunk would have to SHIFT the window rather than replace it, which
 * is a different writer; the host runs a 1..taps-2 tail through the decode family instead.
 *
 * ===========================================================================
 * A 64-BYTE WRITE AT A 64-BYTE-ALIGNED ADDRESS LANDS AND LEAVES ITS NEIGHBOURS ALONE
 * ===========================================================================
 * Measured rather than assumed: it is the same write KV_APPEND makes, checked on silicon
 * against a poisoned region with 0 bytes of an untouched neighbour disturbed. Both the
 * source offset (`face_row_offset`, a multiple of 64 for a 4,096-byte tile) and the
 * destination (`plane_tile * 4096 + {0, 1024}`) are aligned by construction.
 *
 * ===========================================================================
 * ONE BARRIER PER COLUMN, AND THE POPS BEHIND IT
 * ===========================================================================
 * A NoC write barrier is a per-CORE DRAIN. A column is `1 + 2*(taps-1)` writes and takes ONE
 * barrier. THE POPS ARE WHAT MAKES IT CORRECT rather than the barrier count: every write's
 * source is a CB_OUT page, so a page popped with its write still in flight is a page the
 * compute thread may pack over. Wait for both pages of the column, issue every write, ONE
 * barrier, then pop -- ONE PAGE AT A TIME, and with the wrap SPELLED OUT (`page_address()`),
 * because a FIFO pointer returns to its base on an EXACT HIT of the limit and on nothing
 * else.
 *
 * WHY IT FETCHES ON `head` BUT PUBLISHES ON `tail`: fetching on `tail` is a pipeline
 * deadlock, not a precaution -- the output CB fills, compute blocks in cb_reserve_back, the
 * reader blocks because nothing drains the inputs, and the reader can then never reach the
 * end of the descriptor to bump `tail`.
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` -- no tt-metal include.
 * Compile-time args (positional):
 *   0: CB_OUT index   1: tile size in bytes
 *   2: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 *   THREE args, and the count is load-bearing: an out-of-range
 *   `ft_tensix_compile_time_arg` folds to a LITERAL ZERO under SFPI GCC 15.1.0 with no
 *   diagnostic, and wait mode 0 is upstream's UNSTOPPABLE wait.
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build (the whole program; this file is the NCRISC/NoC 1 third):
 *   RELOCATE_KERNEL_TEXT=1 \
 *     READER_CTARGS=0,1,2,3,4,4096,1,5,7,8,6,1 COMPUTE_CTARGS=0,1,2,3,4,16,1,5,7,8,1 \
 *     WRITER_CTARGS=16,4096,1 CB_TABLE=kernels/cb_tables/gdnc_float32.cbtable \
 *     bash foundry/tt/build_tensix_program.sh \
 *       kernels/resident_gdnc_reader.c kernels/resident_gdnc_compute.c \
 *       kernels/resident_gdnc_writer.c programs/gdnc
 *
 * @status NEW. Not yet executed on silicon.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_gdnc_protocol.h"
#include "resident_queue_protocol.h"

#define CB_OUT ft_tensix_compile_time_arg(0)
#define TILE_BYTES ft_tensix_compile_time_arg(1)
#define WAIT_MODE ft_tensix_compile_time_arg(2)

/** The two DRAM banks a descriptor writes: the convolved stream and the rolled window. */
typedef struct gdnc_sinks {
    uint32_t activation_x;
    uint32_t activation_y;
    uint32_t activation_base;
    uint32_t state_x;
    uint32_t state_y;
    uint32_t state_base;
} gdnc_sinks;

static void load_sinks(gdnc_sinks *sinks, volatile FTQ_L1_PTR uint32_t *parameters)
{
    sinks->activation_x = ftgc_xy_x(parameters[kGdncActivationXy]);
    sinks->activation_y = ftgc_xy_y(parameters[kGdncActivationXy]);
    sinks->activation_base = parameters[kGdncActivationOut];
    sinks->state_x = ftgc_xy_x(parameters[kGdncStateXy]);
    sinks->state_y = ftgc_xy_y(parameters[kGdncStateXy]);
    sinks->state_base = parameters[kGdncStateConv];
}

/* Face arithmetic, identical to the reader's. Spelled locally rather than in the protocol
 * header, which is the WIRE contract and carries no L1 layout. */

static uint32_t ftgc_face_bytes(void)
{
    return TILE_BYTES / 4u;
}

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

/**
 * L1 address of page `index` of a group whose first page is at `front`.
 *
 * THE WRAP IS SPELLED OUT ON PURPOSE. `front + index * page` is what every indexed CB read in
 * this backend does, and a FIFO pointer returns to its base on an EXACT HIT of the limit and
 * on NOTHING ELSE. Spelling it here means the group size divides nothing: no page grid on
 * CB_OUT and no rebase.
 */
static uint32_t page_address(uint32_t front, uint32_t index)
{
    uint32_t address = front + index * TILE_BYTES;

    if (address >= ft_tensix_cb(CB_OUT)->fifo_limit)
        address -= ft_tensix_cb(CB_OUT)->fifo_size;
    return address;
}

/** One whole tile, ISSUED. The barrier belongs to the group, not to the tile. */
static void issue_whole_tile(uint32_t source, uint32_t x, uint32_t y, uint32_t address)
{
    ft_tensix_noc_write(source, ft_tensix_noc_address(x, y, address), TILE_BYTES);
}

/**
 * ONE `y` TILE FROM `first_row` DOWN, ISSUED -- the in-place SUB-CHUNK write.
 *
 * `ftgc_y_write_offset()` is 0 for every unsliced descriptor and this is then byte for byte
 * `issue_whole_tile()`: same source, same destination, same length. That equality is the whole
 * compatibility argument -- a `first_row = 0` descriptor cannot tell this function from the
 * one it replaces, so the shipped chunkwise arm is unchanged by construction rather than by
 * inspection.
 *
 * At `first_row = 16` it is the SECOND HALF of the tile, source and destination alike, because
 * rows [16,32) of a 32-row tile are faces 2 and 3 and the four faces are laid out in order.
 * BOTH ENDS MOVE BY THE SAME OFFSET: the compute thread packs the answer at its natural rows
 * in the CB page, so row 16 of the page is 2,048 B in exactly as row 16 of DRAM is.
 *
 * WITHOUT THIS THE SECOND SUB-CHUNK ERASES THE FIRST. `y` is a whole-tile write and the rows a
 * DELTA descriptor does not own come out ZERO (the host gives them `P_t = 0` and `D[t][.] = 0`
 * so that they contribute nothing to the state) -- so a whole-tile write from sub-chunk 1
 * lands 16 rows of zeros over sub-chunk 0's answer, retires clean, and produces a model that
 * reads as half a prompt.
 */
static void issue_y_tile(uint32_t source, uint32_t x, uint32_t y, uint32_t address,
                         const ftgc_shape *shape)
{
    const uint32_t offset = ftgc_y_write_offset(shape);

    ft_tensix_noc_write(source + offset, ft_tensix_noc_address(x, y, address + offset),
                        TILE_BYTES - offset);
}

/**
 * The window roll for ONE tile column, ISSUED: `taps-1` planes, two 64-byte writes each.
 *
 * Row `ftgc_roll_source_row(shape, p)` of the chunk's RAW input becomes ROW 0 of window plane
 * `p`. Plane 0 is the OLDEST, so after this the window holds the chunk's last `taps-1`
 * positions in order -- exactly what position `chunk_rows` of the NEXT chunk (or of the next
 * decode token, in the decode family's own layout) reads.
 */
static void issue_window_roll(uint32_t source, const gdnc_sinks *sinks,
                              const ftgc_shape *shape, uint32_t column)
{
    uint32_t plane;

    for (plane = 0; plane + 1u < shape->conv_taps; plane++) {
        const uint32_t row = ftgc_roll_source_row(shape, plane);
        const uint32_t destination =
            sinks->state_base + ftgc_plane_tile(shape, plane, column) * TILE_BYTES;
        uint32_t half;

        for (half = 0; half < 2u; half++)
            ft_tensix_noc_write(source + ftgc_face_row_offset(row, half),
                                ft_tensix_noc_address(sinks->state_x, sinks->state_y,
                                                      destination +
                                                          ftgc_face_row_offset(
                                                              (uint32_t)kGdncStripRow, half)),
                                ftgc_face_row_bytes());
    }
}

/**
 * The writer half of one CONV descriptor: per column, the convolved tile and then the roll.
 *
 * TWO PAGES A COLUMN, IN THAT ORDER, IS THE WIRE CONTRACT the compute thread's emit order is
 * defined over. A group is one column, so the wait, the issue, the barrier and the pops are
 * all `2` and nothing has to divide anything.
 */
static bool write_one_conv(ftq_context *ctx, const gdnc_sinks *sinks, const ftgc_shape *shape)
{
    uint32_t index;

    for (index = 0; index < shape->col_count; index++) {
        const uint32_t column = shape->first_col + index;

        /* THE ABORT PATH IS THE SHIPPED ONE: a wait that gives up issues nothing and still
         * pops, so the producer drains rather than deadlocking, and the barrier is skipped
         * because there is then nothing outstanding to drain. */
        if (ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 2, WAIT_MODE)) {
            const uint32_t front = ft_tensix_cb_read_pointer(CB_OUT);

            issue_whole_tile(page_address(front, 0u), sinks->activation_x, sinks->activation_y,
                             sinks->activation_base + column * TILE_BYTES);
            issue_window_roll(page_address(front, 1u), sinks, shape, column);
            ft_tensix_noc_write_barrier();
        }
        /* ONE PAGE AT A TIME, for the reason page_address() states. */
        ft_tensix_cb_pop_front(CB_OUT, 1);
        ft_tensix_cb_pop_front(CB_OUT, 1);
    }
    return !ctx->aborting;
}

/**
 * The writer half of one PHASE DELTA descriptor: per head, `state_tiles` y tiles and then
 * `state_tiles^2` updated state tiles.
 *
 * ===========================================================================
 * THIS ONE DOES NOT TRANSPOSE, AND THAT IS THE CHUNK GEOMETRY PAYING OUT
 * ===========================================================================
 * `resident_gdn_writer.c` gets y as a COLUMN -- the (e) reduce's answer for ONE position --
 * and has to scatter 32 elements into row 0 of a tile. Here the (e) term is a whole `[C,S]`
 * matmul answer whose ROWS ARE THE POSITIONS, so y is `state_tiles` WHOLE-TILE writes. So is
 * the state. A head is `state_tiles * (state_tiles + 1)` whole tiles and nothing else.
 *
 * ===========================================================================
 * THE GROUP IS `state_tiles`, NOT A WHOLE HEAD, AND THAT IS A DEADLOCK AVOIDED
 * ===========================================================================
 * A head is `state_tiles * (state_tiles + 1)` = 20 pages at Ornith's shape. Waiting for all 20
 * before issuing anything would require CB_OUT to HOLD 20, and it holds eight: the compute
 * thread would block in `cb_reserve_back` on page nine while this processor waited for page
 * twenty, and neither would ever move. So the drain is `state_tiles + 1` GROUPS OF
 * `state_tiles` -- the y row first, then one state tile-row at a time -- which needs CB_OUT to
 * hold only `2 * state_tiles` and gives the two threads a page of overlap besides.
 *
 * `page_address()` spells the wrap, so no group size has to divide the buffer.
 */
static bool write_one_delta(ftq_context *ctx, const gdnc_sinks *sinks, const ftgc_shape *shape)
{
    const uint32_t tiles = shape->state_tiles;
    uint32_t index;

    for (index = 0; index < shape->head_count; index++) {
        const uint32_t head = shape->head_first + index;
        uint32_t group;

        /* Group 0 is y; group `1 + r` is state tile-row `r`. THE ORDER IS THE WIRE CONTRACT
         * the compute thread's emit order is defined over. */
        for (group = 0; group <= tiles; group++) {
            /* THE ABORT PATH IS THE SHIPPED ONE: a wait that gives up issues nothing and
             * still pops, so the producer drains rather than deadlocking. */
            if (ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, (int32_t)tiles, WAIT_MODE)) {
                const uint32_t front = ft_tensix_cb_read_pointer(CB_OUT);
                uint32_t page;

                for (page = 0; page < tiles; page++) {
                    /* THE STATE IS ALWAYS A WHOLE TILE AND THE SUB-CHUNK DOES NOT CHANGE THAT.
                     * `y`'s rows are POSITIONS, so a sub-chunk owns a row range of it; the
                     * carried state's rows are STATE DIMENSIONS and every sub-chunk rewrites
                     * all of them. Sharing one issue helper between the two would have been
                     * the bug. */
                    if (group == 0u) {
                        issue_y_tile(page_address(front, page), sinks->activation_x,
                                     sinks->activation_y,
                                     sinks->activation_base +
                                         ftgc_y_tile(shape, head, page) * TILE_BYTES,
                                     shape);
                    } else {
                        issue_whole_tile(page_address(front, page), sinks->state_x,
                                         sinks->state_y,
                                         sinks->state_base +
                                             ftgc_state_tile(shape, head, group - 1u, page) *
                                                 TILE_BYTES);
                    }
                }
                ft_tensix_noc_write_barrier();
            }
            /* ONE PAGE AT A TIME, for the reason page_address() states. */
            {
                uint32_t page;

                for (page = 0; page < tiles; page++)
                    ft_tensix_cb_pop_front(CB_OUT, 1);
            }
        }
    }
    return !ctx->aborting;
}

/**
 * The writer half of one GDNC descriptor.
 *
 * The refusal is the same one the reader and the compute thread make, on the same words,
 * before any circular buffer is touched -- so a refused descriptor leaves nobody waiting.
 */
static bool write_one_gdnc(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const ftgc_shape shape = ftgc_decode(parameters);
    gdnc_sinks sinks;

    if (!ftgc_shape_is_supported(&shape) || TILE_BYTES == 0u ||
        shape.tile_bytes != TILE_BYTES) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    load_sinks(&sinks, parameters);
    if (shape.phase == (uint32_t)kGdncPhaseDelta)
        return write_one_delta(ctx, &sinks, &shape);
    return write_one_conv(ctx, &sinks, &shape);
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

    while (running) {
        volatile FTQ_L1_PTR uint32_t *descriptor;
        uint32_t opcode;
        uint32_t token;
        uint32_t group_sequence;
        bool done = true;

        /* FETCH on `head`, so this processor drains the output CB while the reader is still
         * filling the input ones. Publishing waits on `tail`, below. */
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);

        descriptor = ftq_descriptor(&ctx);
        opcode = descriptor[kDescOpcode];
        token = descriptor[kDescToken];
        group_sequence = descriptor[kDescA5];

        ftq_status_opcode(&ctx, opcode);
        if (opcode == kOpGdnChunk) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            done = write_one_gdnc(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
        } else if (opcode == kOpStop) {
            running = false;
        }

        if (!done) {
            running = false;
            continue;
        }

        /* THE PRECONDITION, and the only place it belongs: the reader must have finished this
         * descriptor before its completion is published. */
        if (!ftq_wait_until_reaches(&ctx, kFieldTail, ctx.cursor + 1u)) {
            running = false;
            continue;
        }

        ctx.cursor++;
        /* Token before count: a host that sees the count is entitled to trust the token it
         * reads next. Both are plain stores from ONE in-order RISC-V. */
        ctx.control[kFieldCompletedToken] = token;
        ctx.control[kFieldCompletedCount] = ++completed;
        ftq_group_post(&ctx, group_sequence, token);
    }

    /* This processor does NOT touch kFieldError or kFieldState: the reader owns those, and
     * one writer per word is the whole ordering argument. */
    ftq_status_set(&ctx, kStateExit);
}
