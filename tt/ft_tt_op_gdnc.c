/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_gdnc.c
 * @brief Implementation of the chunkwise Gated-Delta-Net host driver. See ft_tt_op_gdnc.h.
 *
 * @status NEW. Parameter layout, CB placement, band construction and the per-operand DRAM
 *   sizes a host stager allocates from — pure host arithmetic, pinned off hardware by
 *   tests/test_gdnc_ops.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_op_gdnc.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_tilize.h"

/* The device's L1 pointer attribute belongs to the SFPI GCC fork, not to the contract; the
 * protocol header guards it so a HOST compiler can read the file. Including it here is what
 * lets the sizing functions below compute their tile counts with the SAME `ftgc_*` accessors
 * both dataflow kernels compute their offsets with, instead of respelling the layout a third
 * time. */
#define FTQ_L1_PTR
#include "kernels/ftgc_table.h"
#include "kernels/resident_gdnc_protocol.h"

/** Pages of the streamed per-column operands. FOUR columns of look-ahead, not the usual two:
 *  this reader issues `1 + 2*(taps-1) + taps` NoC reads a column against a compute thread
 *  that issues `4*taps + 2` tile operations, so the two are close enough in cost that a
 *  two-page buffer would hard-handshake them. Four pages of 4 KiB is 16 KiB against a
 *  1.29 MiB arena. */
#define GDNC_CB_COLUMNS 4u

/** Pages of CB_OUT: four column groups of two. The decode family measured CB_OUT capacity
 *  worth 3.8% between two pages and ten and NOTHING beyond that — with only one group live
 *  the compute thread hard-handshakes the writer on every column. */
#define GDNC_CB_OUT_PAGES 8u

static bool operand_is_set(const ft_tt_op_operand *operand)
{
    return operand->tile_bytes != 0u;
}

/* ================================================================
 * The CONV block depth — one number, three sides. See ft_tt_op_gdnc.h.
 * ================================================================ */

uint32_t ft_tt_op_gdnc_conv_depth_parse(const char *value)
{
    char *end = NULL;
    unsigned long parsed;

    if (!value || !*value)
        return FT_TT_OP_GDNC_CONV_DEPTH;
    parsed = strtoul(value, &end, 10);
    /* NO SILENT FALLBACK TO THE DEFAULT. A knob that ignores what it was told is how a sweep
     * measures its control four times and reports a ladder. It must also fit a BYTE, because
     * that is where it rides on the wire. */
    if (end == value || *end != '\0' || parsed < 1u || parsed > 255u)
        return 0u;
    return (uint32_t)parsed;
}

uint32_t ft_tt_op_gdnc_conv_depth(void)
{
    static uint32_t resolved;
    static bool known;
    const char *value;

    if (known)
        return resolved;
    known = true;
    value = getenv(FT_TT_OP_GDNC_CONV_DEPTH_ENV);
    resolved = ft_tt_op_gdnc_conv_depth_parse(value);
    /* ONE LINE PER PROCESS, ALWAYS — this is the line that says which rung actually placed. */
    if (resolved == 0u)
        fprintf(stderr,
                "ft_tt: gdnc CONV block depth REFUSED — %s=\"%s\" is not a depth in 1..255, "
                "so every GDNC descriptor refuses rather than quietly placing for %u\n",
                FT_TT_OP_GDNC_CONV_DEPTH_ENV, value ? value : "",
                (unsigned)FT_TT_OP_GDNC_CONV_DEPTH);
    else
        fprintf(stderr,
                "ft_tt: gdnc CONV block depth %u (compile-time default %u, %s%s) — declared on "
                "every CONV descriptor; a program built at another depth refuses\n",
                (unsigned)resolved, (unsigned)FT_TT_OP_GDNC_CONV_DEPTH,
                FT_TT_OP_GDNC_CONV_DEPTH_ENV, (value && *value) ? " set" : " unset");
    return resolved;
}

static uint32_t gdnc_gcd(uint32_t a, uint32_t b)
{
    while (b != 0u) {
        const uint32_t r = a % b;

        a = b;
        b = r;
    }
    return a;
}

static uint32_t gdnc_lcm(uint32_t a, uint32_t b)
{
    if (a == 0u || b == 0u)
        return a > b ? a : b;
    return a / gdnc_gcd(a, b) * b;
}

static uint32_t gdnc_round_up(uint32_t value, uint32_t grid)
{
    if (grid == 0u)
        return value;
    return ((value + grid - 1u) / grid) * grid;
}

/**
 * EVERY OPERAND IS AN fp32 TILE, and this refusal is what makes the device's one-word check
 * sufficient.
 *
 * The family publishes ONE `kGdncTileBytes` word covering four DRAM endpoints, so the device
 * cannot tell a 2048-byte activation from a 4096-byte one when the word came from a different
 * operand — which is exactly the defect the decode family shipped for a round: bf16 pages
 * placed under a reader issuing 4096-byte tile reads, with the tile-size refusal reading the
 * other operand and passing.
 */
static bool operands_are_fp32(const ft_tt_op_gdnc_request *request)
{
    const ft_tt_op_operand *operands[4] = { &request->activation_in, &request->activation_out,
                                            &request->constant, &request->state_conv };
    unsigned index;

    for (index = 0; index < 4u; index++) {
        if (operand_is_set(operands[index]) &&
            operands[index]->tile_bytes != FT_TT_OP_GDNC_FP32_TILE_BYTES)
            return false;
    }
    return true;
}

/** Pack a NoC coordinate pair into one word. */
static uint32_t pack_xy(const ft_tt_op_operand *operand)
{
    return (uint32_t)operand->x | ((uint32_t)operand->y << 16);
}

/** Tile columns of the stream. */
static uint32_t gdnc_tile_cols(const ft_tt_op_gdnc_request *request)
{
    return request->conv_dim / FT_TT_OP_TILE_DIM;
}

/** The tile columns a descriptor owns. A `col_count` of 0 means the whole stream — the
 *  convention the device decodes, and what an unsliced caller leaves the field at. */
static uint32_t gdnc_col_count(const ft_tt_op_gdnc_request *request)
{
    return request->col_count ? request->col_count : gdnc_tile_cols(request);
}

/** The v-heads a DELTA descriptor owns. A `col_count` of 0 means all of them. */
static uint32_t gdnc_head_count(const ft_tt_op_gdnc_request *request)
{
    return request->col_count ? request->col_count : request->v_heads;
}

int ft_tt_op_gdnc_shape_is_supported(const ft_tt_op_gdnc_request *request)
{
    if (!request)
        return 0;
    if (request->chunk_rows == 0u || request->chunk_rows > FT_TT_OP_GDNC_MAX_CHUNK_ROWS)
        return 0;
    /* The slice is a HALF-WORD each because `conv_tile_cols` is 256 at Ornith and a byte
     * would truncate to 0 — silently making every descriptor a whole-stream one. */
    if (request->first_col > 0xFFFFu || request->col_count > 0xFFFFu)
        return 0;

    if (request->phase == FT_TT_OP_GDNC_PHASE_CONV) {
        /* THE ROLL SOURCE ROW IS `chunk_rows - (taps-1) + plane` AND IT KNOWS NOTHING ABOUT AN
         * OFFSET, so a CONV descriptor naming one rolls the wrong rows into the window: right
         * in chunk 0 and wrong from chunk 1 on. CONV does not need the field — see
         * `kGdncFirstRow` — so it is refused rather than given a meaning. */
        if (request->first_row != 0u || request->gate_band != 0u ||
            (request->delta_flags & (FT_TT_OP_GDNC_DELTA_FLAG_TABLE |
                                     FT_TT_OP_GDNC_DELTA_FLAG_STATE_ROUND)) != 0u)
            return 0;
        if (request->conv_taps < 2u || request->conv_taps > 8u)
            return 0;
        /* A chunk shorter than the window cannot REPLACE it, only shift it, and a partial
         * roll is a different writer. THAT FLOOR BELONGS TO THE ROLL, so PHASE DELTA — which
         * rolls nothing — does not carry it. See ft_tt_op_gdnc.h. */
        if (request->chunk_rows < request->conv_taps - 1u)
            return 0;
        if (request->conv_dim == 0u || (request->conv_dim % FT_TT_OP_TILE_DIM) != 0u)
            return 0;
        if (request->first_col + gdnc_col_count(request) > gdnc_tile_cols(request))
            return 0;
        return 1;
    }

    if (request->phase == FT_TT_OP_GDNC_PHASE_DELTA) {
        /* `state_tiles` squares into L1 — 8 is 64 pages of 4 KiB for ONE head's state, which
         * is where the arena stops. It must also fit the shape word's byte. */
        if (request->state_tiles == 0u || request->state_tiles > 8u)
            return 0;
        if (request->v_heads == 0u || request->v_heads > 0xFFFFu)
            return 0;
        if (request->k_heads == 0u || request->k_heads > 0xFFFFu)
            return 0;
        /* The GQA divisibility, checked on BOTH sides because getting the repeat backwards is
         * a KNOWN bug in this architecture rather than a hypothetical one. */
        if (request->v_heads % request->k_heads != 0u)
            return 0;
        if (request->first_col + gdnc_head_count(request) > request->v_heads)
            return 0;
        /* THE IN-PLACE SUB-CHUNK'S WINDOW. Off a face boundary the y write is a gather rather
         * than a contiguous suffix; past the tile it is a negative length. Both retire clean
         * and corrupt a neighbouring head's y. */
        if ((request->first_row % FT_TT_OP_GDNC_SUB_CHUNK_ALIGN) != 0u)
            return 0;
        if (request->first_row + request->chunk_rows > FT_TT_OP_GDNC_MAX_CHUNK_ROWS)
            return 0;
        /* A band without the table flag would be a word the kernel never reads. */
        if ((request->delta_flags & FT_TT_OP_GDNC_DELTA_FLAG_TABLE) == 0u &&
            request->gate_band != 0u)
            return 0;
        return 1;
    }
    return 0;
}

int ft_tt_op_gdnc_parameters(const ft_tt_op_gdnc_request *request,
                             uint32_t out_words[FT_TT_OP_GDNC_WORDS])
{
    if (!request || !out_words)
        return FT_TT_EINVAL;
    /* An unusable depth knob refuses HERE, before a descriptor exists, rather than declaring a
     * zero the kernels would refuse one descriptor at a time. */
    if (ft_tt_op_gdnc_conv_depth() == 0u)
        return FT_TT_EINVAL;
    if (!ft_tt_op_gdnc_shape_is_supported(request))
        return FT_TT_EINVAL;
    if (!operand_is_set(&request->activation_in) ||
        !operand_is_set(&request->activation_out) || !operand_is_set(&request->constant) ||
        !operand_is_set(&request->state_conv))
        return FT_TT_EINVAL;
    if (!operands_are_fp32(request))
        return FT_TT_EINVAL;
    /* THE ACTIVATION PAIR SHARES ONE NoC COORDINATE WORD, so both buffers must sit on the same
     * DRAM endpoint. NOTHING ON THE DEVICE CAN CATCH A VIOLATION: the reader reads the right
     * bank and the writer puts the answer at the OUTPUT's offset inside the INPUT's bank, so
     * the descriptor completes normally, the output buffer is never written, and the host
     * reads back whatever was already there. Measured on the decode family, silicon
     * 2026-08-18 — and what named it was that the conv answer came back BIT-IDENTICAL with
     * the kernel's SiLU removed. */
    if (request->activation_in.x != request->activation_out.x ||
        request->activation_in.y != request->activation_out.y)
        return FT_TT_EINVAL;

    /* PHASE DELTA's gate block shares the CONSTANT operand's packed NoC coordinate, exactly as
     * the activation pair shares its own, and it fails the same way: the reader would pull
     * `kGdncGateStride` floats from the right bank at an address that belongs to nothing, the
     * descriptor would retire clean, and every gate column would be whatever was there. */
    if (request->phase == FT_TT_OP_GDNC_PHASE_DELTA && request->gate_address == 0u)
        return FT_TT_EINVAL;

    memset(out_words, 0, sizeof(uint32_t) * FT_TT_OP_GDNC_WORDS);

    out_words[FT_TT_OP_GDNC_SHAPE] = (request->phase & 0xFFu) |
                                     ((request->chunk_rows & 0xFFu) << 8) |
                                     ((request->conv_taps & 0xFFu) << 16) |
                                     ((request->state_tiles & 0xFFu) << 24);
    /* ONE WORD, TWO READINGS, selected by the phase byte. PHASE CONV keeps it byte-identical
     * to what shipped; PHASE DELTA packs the two head counts into it, because
     * `FT_TT_QUEUE_PARAM_WORDS` is 16 for EVERY family and a seventeenth moves the shared slot
     * stride. */
    out_words[FT_TT_OP_GDNC_CONV_TILE_COLS] =
        (request->phase == FT_TT_OP_GDNC_PHASE_DELTA)
            ? ((request->v_heads & 0xFFFFu) | ((request->k_heads & 0xFFFFu) << 16))
            : gdnc_tile_cols(request);
    /* The COUNT is written as the caller gave it, NOT as resolved: a 0 count means "the whole
     * stream" on both sides, and writing the resolved value would make an unsliced descriptor
     * encode differently from the one an encoder that predates the slice would emit. */
    out_words[FT_TT_OP_GDNC_COL_SLICE] =
        (request->first_col & 0xFFFFu) | ((request->col_count & 0xFFFFu) << 16);

    out_words[FT_TT_OP_GDNC_ACTIVATION_XY] = pack_xy(&request->activation_in);
    out_words[FT_TT_OP_GDNC_ACTIVATION_IN] = request->activation_in.address;
    out_words[FT_TT_OP_GDNC_ACTIVATION_OUT] = request->activation_out.address;

    out_words[FT_TT_OP_GDNC_CONSTANT_XY] = pack_xy(&request->constant);
    out_words[FT_TT_OP_GDNC_CONSTANT_WEIGHT] = request->constant.address;

    out_words[FT_TT_OP_GDNC_STATE_XY] = pack_xy(&request->state_conv);
    out_words[FT_TT_OP_GDNC_STATE_CONV] = request->state_conv.address;

    out_words[FT_TT_OP_GDNC_DELTA_GATE] =
        (request->phase == FT_TT_OP_GDNC_PHASE_DELTA) ? request->gate_address : 0u;
    /* PHASE CONV encodes a ZERO here whatever the request said, and the request is separately
     * refused for a nonzero one: the word must be zero on a CONV descriptor so this family's
     * shipped encoding is unchanged for the phase that does not use it. */
    out_words[FT_TT_OP_GDNC_FIRST_ROW] =
        (request->phase == FT_TT_OP_GDNC_PHASE_DELTA) ? request->first_row : 0u;
    out_words[FT_TT_OP_GDNC_DELTA_BAND] =
        (request->phase == FT_TT_OP_GDNC_PHASE_DELTA) ? request->gate_band : 0u;

    /* ONE tile size for the whole family, not whichever operand this branch reached for. */
    out_words[FT_TT_OP_GDNC_TILE_BYTES] = FT_TT_OP_GDNC_FP32_TILE_BYTES;
    /* BYTE 0 OF THE FLAGS WORD IS THIS HOST'S OWN CONV BLOCK DEPTH, and both kernels compare
     * it for STRICT EQUALITY against the constant they were compiled with. It is declared on
     * CONV descriptors ONLY, because PHASE DELTA never reads it — so a DELTA descriptor still
     * leaves the whole word zero and its shipped encoding is unchanged. There is deliberately
     * NO "0 means unset" escape: an encoder that forgets to declare must REFUSE, not pass.
     * `kGdncByteConvDepth` in kernels/resident_gdnc_protocol.h carries the drift this
     * catches — a host that placed DEEPER than the kernels were built at passes every capacity
     * check, runs at the kernels' depth, and reports the control's timing as the arm's. */
    out_words[FT_TT_OP_GDNC_FLAGS] = ((request->phase == FT_TT_OP_GDNC_PHASE_CONV)
                                          ? (ft_tt_op_gdnc_conv_depth() & 0xFFu)
                                          : (request->delta_flags &
                                             (FT_TT_OP_GDNC_DELTA_FLAG_COMPACT |
                                              FT_TT_OP_GDNC_DELTA_FLAG_TABLE |
                                              FT_TT_OP_GDNC_DELTA_FLAG_STATE_ROUND))) |
                                     (request->serialize ? FT_TT_OP_GDNC_FLAG_SERIALIZE : 0u);
    /* PHASE CONV: two pages a column, the conv_out tile and then the RAW input tile the writer
     * takes the window roll's rows out of. PHASE DELTA: `state_tiles*(state_tiles+1)` a head.
     * The kernels test `tile_count` for EQUALITY against this, not for `>=`: it is fully
     * determined by the slice, so an inequality would be a host that disagreed with itself. */
    out_words[FT_TT_OP_GDNC_TILE_COUNT] =
        (request->phase == FT_TT_OP_GDNC_PHASE_DELTA)
            ? gdnc_head_count(request) * request->state_tiles * (request->state_tiles + 1u)
            : 2u * gdnc_col_count(request);
    return FT_TT_OK;
}

/* ================================================================
 * Circular buffer placement
 * ================================================================ */

/**
 * Pages per buffer, for a shape with `taps` taps.
 *
 * EVERY COUNT IS A CONSUMER GRID, not a comfort setting. `ft_tensix_unpack_ab(cb, ..., tile)`
 * is `read_pointer + tile*page_size` with NO WRAP, so a buffer a consumer indexes `n` tiles
 * across needs `pages % n == 0` as well as `pages >= n`:
 *
 *   CB_BAND    indexed `2*taps` wide  -> exactly `2*taps` pages, which also makes its push an
 *              EXACT HIT on the limit and therefore self-rebasing
 *   CB_VECTOR  indexed `taps` wide    -> a multiple of `taps`
 *   CB_PROD    indexed `2*taps` wide  -> a multiple of `2*taps`, and at least TWO of them so
 *              the compute thread can stage the next column's products while the matmul
 *              reduces this one
 *   CB_IN, CB_STATE  indexed one tile at a time -> any count; four for the look-ahead
 *   CB_OUT     never indexed (the writer spells its own wrap) -> any count
 *
 * The reader re-checks all of it against the PLACED counts and refuses a descriptor that does
 * not fit, because the host and the kernels are compiled separately.
 */
static void resolve_cb_pages_at(uint32_t taps, uint32_t state_tiles, uint32_t depth,
                                uint32_t out_pages[FT_TT_OP_GDNC_CB_COUNT])
{
    const uint32_t band = 2u * taps;
    const uint32_t input = 3u * state_tiles;
    const uint32_t state = state_tiles * state_tiles;
    /* THE CONV BLOCK'S THREE INDEX SPANS AT THIS DEPTH. A group is `depth` input tiles,
     * `depth` assembled window tiles and `depth*taps` weight planes, and the compute thread
     * indexes across the whole of each — so these are grids, not capacities. */
    const uint32_t conv_group = depth;
    const uint32_t conv_weight_group = depth * taps;

    /* ONE PLACEMENT SERVES BOTH PHASES, because a residency is armed once and then answers
     * whatever the queue hands it. Every count below is therefore a COMMON MULTIPLE of the two
     * phases' index spans, not the larger of them:
     *
     *   CB_IN     CONV indexes 1 wide, DELTA `3*state_tiles` -> two DELTA groups
     *   CB_STATE  CONV 1, DELTA `state_tiles^2`             -> two DELTA groups
     *   CB_BAND   CONV `2*taps`, DELTA 4                    -> lcm, so both divide
     *   CB_VECTOR CONV `taps`, DELTA 5                      -> `5*taps` divides both
     *   CB_PROD   CONV `2*taps`, DELTA unused
     *   CB_CC     DELTA 24 -> exactly one span, so its push is an EXACT HIT and self-rebasing
     *   CB_WORK   DELTA `9*state_tiles` -> the same
     *   CB_OUT    never indexed (the writer spells its own wrap) -> any count
     *
     * A count that satisfied only one phase would be a residency that runs one kind of
     * descriptor and reads out of its buffers on the other, at some head that is not the
     * first -- and the reader refuses exactly that, which is what makes this arithmetic
     * checked rather than asserted. */
    /* AT DEPTH D EVERY CONV SPAN MULTIPLIES BY D, so each count is rounded onto the lcm of the
     * two phases' spans and then held to two blocks of the larger one. At D = 1 every
     * expression below collapses to exactly what shipped — 24 / 32 / 8 / 20 / 16 at Ornith —
     * which is what makes `FT_TT_OP_GDNC_CONV_DEPTH=1` a control and not a second sizing. */
    out_pages[0] = gdnc_round_up(2u * (input > conv_group ? input : conv_group),
                                 gdnc_lcm(input, conv_group));               /* CB_IN */
    out_pages[1] = gdnc_round_up(2u * (state > conv_group ? state : conv_group),
                                 gdnc_lcm(state, conv_group));               /* CB_STATE */
    out_pages[2] = (band % 4u == 0u) ? band : 4u * band;                     /* CB_BAND */
    out_pages[3] = gdnc_round_up(2u * (FT_TT_OP_GDNC_DELTA_VECTOR_SPAN > conv_weight_group
                                           ? FT_TT_OP_GDNC_DELTA_VECTOR_SPAN
                                           : conv_weight_group),
                                 gdnc_lcm(FT_TT_OP_GDNC_DELTA_VECTOR_SPAN,
                                          conv_weight_group));               /* CB_VECTOR */
    out_pages[4] = 2u * band;                                                /* CB_PROD */
    out_pages[5] = FT_TT_OP_GDNC_DELTA_WORK_SPAN(state_tiles); /* CB_WORK */
    out_pages[6] = 1u;                            /* CB_SCRATCH — one gate run */
    out_pages[7] = 2u;                            /* CB_DECAY */
    out_pages[8] = FT_TT_OP_GDNC_DELTA_CC_SPAN;   /* CB_CC */
    /* CB_OUT holds TWO of the WRITER'S GROUPS and not one head. A head is
     * `state_tiles*(state_tiles+1)` = 20 pages at Ornith and this buffer holds eight: a writer
     * that waited for a whole head would block on page twenty while the compute thread blocked
     * reserving page nine, and neither would ever move. The writer drains `state_tiles` at a
     * time -- the y row, then one state tile-row each -- so this is the count that matters. */
    out_pages[9] = (GDNC_CB_OUT_PAGES > 2u * state_tiles) ? GDNC_CB_OUT_PAGES
                                                          : 2u * state_tiles; /* CB_OUT */
    (void)GDNC_CB_COLUMNS;
}

int ft_tt_op_gdnc_cb_pages_at_depth(uint32_t taps, uint32_t state_tiles, uint32_t depth,
                                    uint32_t out_pages[FT_TT_OP_GDNC_CB_COUNT])
{
    if (!out_pages || taps < 2u || taps > 8u || state_tiles == 0u || state_tiles > 8u ||
        depth < 1u || depth > 255u)
        return FT_TT_EINVAL;
    resolve_cb_pages_at(taps, state_tiles, depth, out_pages);
    return FT_TT_OK;
}

/** The sizing the SHIPPING path uses: this process's own depth, read from the environment
 *  once. Split from the arithmetic above so a pricing caller can ask about a rung it is not
 *  running. */
static void resolve_cb_pages(uint32_t taps, uint32_t state_tiles,
                             uint32_t out_pages[FT_TT_OP_GDNC_CB_COUNT])
{
    resolve_cb_pages_at(taps, state_tiles, ft_tt_op_gdnc_conv_depth(), out_pages);
}

int ft_tt_op_gdnc_place_cbs(const ft_tt_op_gdnc_request *request,
                            ft_tt_launch_cb_entry out_entries[FT_TT_OP_GDNC_CB_COUNT],
                            void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    static const unsigned indices[FT_TT_OP_GDNC_CB_COUNT] = {
        FT_TT_OP_GDNC_CB_IN,      FT_TT_OP_GDNC_CB_STATE,   FT_TT_OP_GDNC_CB_BAND,
        FT_TT_OP_GDNC_CB_VECTOR,  FT_TT_OP_GDNC_CB_PROD,    FT_TT_OP_GDNC_CB_WORK,
        FT_TT_OP_GDNC_CB_SCRATCH, FT_TT_OP_GDNC_CB_DECAY,   FT_TT_OP_GDNC_CB_CC,
        FT_TT_OP_GDNC_CB_OUT
    };
    ft_tt_op_cb_request requests[FT_TT_OP_GDNC_CB_COUNT];
    uint32_t pages[FT_TT_OP_GDNC_CB_COUNT];
    unsigned index;

    if (!request || !out_entries || !buffer || !out_mask)
        return FT_TT_EINVAL;
    /* An unusable depth refuses HERE, because a placement sized from one is the single thing
     * this file must never produce: it is the placement the kernels then find perfectly
     * legal. */
    if (ft_tt_op_gdnc_conv_depth() == 0u)
        return FT_TT_EINVAL;
    if (!ft_tt_op_gdnc_shape_is_supported(request))
        return FT_TT_EINVAL;
    /* A PLACEMENT IS THE WHOLE FAMILY'S, NOT ONE PHASE'S. The residency is armed once and then
     * answers whatever the queue hands it, so a request that names only its own phase's shape
     * would place buffers the OTHER phase reads out of. Both shapes are required here even
     * though `shape_is_supported` only checks the phase's own. */
    if (request->conv_taps < 2u || request->conv_taps > 8u)
        return FT_TT_EINVAL;
    if (request->state_tiles == 0u || request->state_tiles > 8u)
        return FT_TT_EINVAL;

    resolve_cb_pages(request->conv_taps, request->state_tiles, pages);

    /* EVERY page is FT_TT_OP_GDNC_FP32_TILE_BYTES and none of them is taken from an operand.
     * The CB table makes all six buffers Float32 and all three kernels are built with
     * TILE_BYTES = 4096; a page sized from a bf16 operand would put 2048-byte pages under a
     * reader that writes 4096-byte tiles, which overruns the buffer into whatever
     * `ft_tt_op_cb_place` put above it. */
    for (index = 0; index < FT_TT_OP_GDNC_CB_COUNT; index++) {
        requests[index].index = indices[index];
        requests[index].page_bytes = FT_TT_OP_GDNC_FP32_TILE_BYTES;
        requests[index].page_count = pages[index];
    }

    return ft_tt_op_cb_place(requests, FT_TT_OP_GDNC_CB_COUNT, out_entries, buffer,
                             buffer_bytes, out_mask);
}

/* ================================================================
 * The band matrices
 * ================================================================ */

size_t ft_tt_op_gdnc_band_bytes(uint32_t taps)
{
    if (taps < 2u || taps > 8u)
        return 0u;
    return (size_t)(2u * taps) * FT_TT_OP_GDNC_FP32_TILE_BYTES;
}

int ft_tt_op_gdnc_build_band(uint32_t taps, void *out_tiles, size_t out_bytes)
{
    /* One 32x32 row-major fp32 matrix at a time, tilized into place. The stack cost is 4 KiB,
     * which is the same tile it produces. */
    float plane[FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM];
    unsigned char *out = (unsigned char *)out_tiles;
    uint32_t tap;

    if (!out_tiles || taps < 2u || taps > 8u)
        return FT_TT_EINVAL;
    if (out_bytes < ft_tt_op_gdnc_band_bytes(taps))
        return FT_TT_EINVAL;

    for (tap = 0; tap < taps; tap++) {
        const uint32_t shift = (taps - 1u) - tap; /* `d` in the derivation */
        uint32_t row;
        uint32_t which;

        for (which = 0; which < 2u; which++) {
            memset(plane, 0, sizeof(plane));
            for (row = 0; row < FT_TT_OP_TILE_DIM; row++) {
                if (which == 0u) {
                    /* S_tau: the chunk's OWN input, row t-d, for rows at or past the shift. */
                    if (row >= shift)
                        plane[row * FT_TT_OP_TILE_DIM + (row - shift)] = 1.0f;
                } else {
                    /* C_tau: the WINDOW, plane p = t + tau, for rows below the shift. Empty
                     * at tau = taps-1, where the shift is 0 — the zero tile, emitted anyway
                     * so the product count stays `2*taps`. */
                    if (row < shift)
                        plane[row * FT_TT_OP_TILE_DIM + (row + tap)] = 1.0f;
                }
            }
            if (ft_tt_tilize_strip(plane, sizeof(float), FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM,
                                   (size_t)FT_TT_OP_TILE_DIM * sizeof(float),
                                   out + (size_t)(2u * tap + which) *
                                             FT_TT_OP_GDNC_FP32_TILE_BYTES) != FT_TT_TILIZE_OK)
                return FT_TT_EINVAL;
        }
    }
    return FT_TT_OK;
}

/* ================================================================
 * What one layer costs in card DRAM
 * ================================================================
 *
 * EVERY SIZE BELOW IS `<an ftgc_* TILE COUNT> * FT_TT_OP_GDNC_FP32_TILE_BYTES`, and the tile
 * counts come from `resident_gdnc_protocol.h` itself rather than from arithmetic respelled
 * here. That header is the settled layout and both dataflow kernels compute their offsets
 * from those same accessors, so a stager sized from them cannot disagree with the reader
 * about where a plane ends — which is the one disagreement nothing checks at run time.
 */

/** The device's own view of this request. The slice and `tile_count` are not size inputs. */
static ftgc_shape shape_of(const ft_tt_op_gdnc_request *request)
{
    ftgc_shape shape;

    shape.phase = request->phase;
    shape.chunk_rows = request->chunk_rows;
    shape.conv_taps = request->conv_taps;
    shape.conv_tile_cols = gdnc_tile_cols(request);
    shape.first_col = 0u;
    shape.col_count = shape.conv_tile_cols;
    shape.tile_bytes = FT_TT_OP_GDNC_FP32_TILE_BYTES;
    shape.tile_count = 0u;
    return shape;
}

static size_t gdnc_tile_bytes(uint32_t tiles)
{
    return (size_t)tiles * FT_TT_OP_GDNC_FP32_TILE_BYTES;
}

size_t ft_tt_op_gdnc_conv_stream_bytes(const ft_tt_op_gdnc_request *request)
{
    if (!ft_tt_op_gdnc_shape_is_supported(request))
        return 0u;
    return gdnc_tile_bytes(gdnc_tile_cols(request));
}

size_t ft_tt_op_gdnc_constant_bytes(const ft_tt_op_gdnc_request *request)
{
    ftgc_shape shape;

    if (!ft_tt_op_gdnc_shape_is_supported(request))
        return 0u;
    shape = shape_of(request);
    return gdnc_tile_bytes(ftgc_constant_tiles(&shape));
}

size_t ft_tt_op_gdnc_conv_state_bytes(const ft_tt_op_gdnc_request *request)
{
    if (!ft_tt_op_gdnc_shape_is_supported(request))
        return 0u;
    return gdnc_tile_bytes((request->conv_taps - 1u) * gdnc_tile_cols(request));
}

/* ================================================================
 * PHASE DELTA's operands
 * ================================================================ */

size_t ft_tt_op_gdnc_delta_constant_tile_bytes(void)
{
    return (size_t)FT_TT_OP_GDNC_DELTA_CONST_TILES * FT_TT_OP_GDNC_FP32_TILE_BYTES;
}

int ft_tt_op_gdnc_build_delta_constants(uint32_t state_size, void *out_tiles, size_t out_bytes)
{
    float plane[FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM];
    unsigned char *out = (unsigned char *)out_tiles;
    float scale;
    uint32_t which;

    if (!out_tiles || state_size == 0u)
        return FT_TT_EINVAL;
    if (out_bytes < ft_tt_op_gdnc_delta_constant_tile_bytes())
        return FT_TT_EINVAL;

    /* DERIVED ONCE, HERE. The card never computes a reciprocal square root, so the two arms of
     * `Y = scale (Q_hat N_0^T) + (scale Pm) U` cannot disagree about a factor that multiplies
     * every output of a recurrence. The decode family sends the same number as raw fp32 bits
     * for the same reason. */
    scale = 1.0f / sqrtf((float)state_size);

    for (which = 0; which < FT_TT_OP_GDNC_DELTA_CONST_TILES; which++) {
        uint32_t row;
        uint32_t column;

        memset(plane, 0, sizeof(plane));
        for (row = 0; row < FT_TT_OP_TILE_DIM; row++) {
            for (column = 0; column < FT_TT_OP_TILE_DIM; column++) {
                float value = 0.0f;

                if (which == FT_TT_OP_GDNC_DELTA_CONST_IDENTITY)
                    value = (row == column) ? 1.0f : 0.0f;
                else if (which == FT_TT_OP_GDNC_DELTA_CONST_TWO_I)
                    value = (row == column) ? 2.0f : 0.0f;
                else if (which == FT_TT_OP_GDNC_DELTA_CONST_STRICT)
                    value = (column < row) ? 1.0f : 0.0f;
                else
                    /* A COLUMN, not a full tile: `bcast_col` reads column 0 and replicates it
                     * across the row. The other lanes are never read; they are written zero
                     * so a dump of this tile is readable rather than because it matters. */
                    value = (column == 0u) ? scale : 0.0f;
                plane[row * FT_TT_OP_TILE_DIM + column] = value;
            }
        }
        if (ft_tt_tilize_strip(plane, sizeof(float), FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM,
                               (size_t)FT_TT_OP_TILE_DIM * sizeof(float),
                               out + (size_t)which * FT_TT_OP_GDNC_FP32_TILE_BYTES) !=
            FT_TT_TILIZE_OK)
            return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

int ft_tt_op_gdnc_build_decay_at(const float *decay, uint32_t first_row, uint32_t chunk_rows,
                                 void *out_tile, size_t out_bytes)
{
    float plane[FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM];
    uint32_t row;

    if (!decay || !out_tile || chunk_rows == 0u)
        return FT_TT_EINVAL;
    if (first_row + chunk_rows > FT_TT_OP_GDNC_MAX_CHUNK_ROWS)
        return FT_TT_EINVAL;
    if ((first_row % FT_TT_OP_GDNC_SUB_CHUNK_ALIGN) != 0u)
        return FT_TT_EINVAL;
    if (out_bytes < FT_TT_OP_GDNC_FP32_TILE_BYTES)
        return FT_TT_EINVAL;
    /* THE UNSLICED CASE IS THE SHIPPED FUNCTION AND NOT A RE-DERIVATION OF IT. Delegating is
     * what makes `first_row = 0` BIT-IDENTICAL by construction instead of by inspection —
     * this file has already paid once for two spellings of one recurrence. */
    if (first_row == 0u)
        return ft_tt_op_gdnc_build_decay(decay, chunk_rows, out_tile, out_bytes);

    /* EVERYTHING THIS DESCRIPTOR DOES NOT OWN IS ZERO, AND THAT INCLUDES THE COLUMNS.
     * Zeroing the whole plane first is what confines `A_off` to the sub-chunk's own [C,C]
     * block — see ft_tt_op_gdnc_build_decay_at()'s declaration. */
    memset(plane, 0, sizeof(plane));
    for (row = first_row; row < first_row + chunk_rows; row++) {
        uint32_t column = row;

        /* THE PRODUCT RECURRENCE, run over the sub-chunk's own rows and stopped at its first:
         * `D[t][t] = 1`, then walk LEFT multiplying by one decay at a time, every partial in
         * (0,1] and nothing divided by a cumulative product. */
        plane[row * FT_TT_OP_TILE_DIM + row] = 1.0f;
        while (column > first_row) {
            plane[row * FT_TT_OP_TILE_DIM + (column - 1u)] =
                plane[row * FT_TT_OP_TILE_DIM + column] * decay[column - first_row];
            column--;
        }
    }
    if (ft_tt_tilize_strip(plane, sizeof(float), FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM,
                           (size_t)FT_TT_OP_TILE_DIM * sizeof(float),
                           out_tile) != FT_TT_TILIZE_OK)
        return FT_TT_EINVAL;
    return FT_TT_OK;
}

int ft_tt_op_gdnc_build_decay(const float *decay, uint32_t chunk_rows, void *out_tile,
                              size_t out_bytes)
{
    float plane[FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM];
    uint32_t row;

    if (!decay || !out_tile || chunk_rows == 0u || chunk_rows > FT_TT_OP_GDNC_MAX_CHUNK_ROWS)
        return FT_TT_EINVAL;
    if (out_bytes < FT_TT_OP_GDNC_FP32_TILE_BYTES)
        return FT_TT_EINVAL;

    memset(plane, 0, sizeof(plane));
    for (row = 0; row < FT_TT_OP_TILE_DIM; row++) {
        uint32_t column = row;

        /* THE PRODUCT RECURRENCE. `D[t][t] = 1`, then walk LEFT multiplying by one decay at a
         * time: every partial is in (0,1], nothing is divided by a cumulative product, and no
         * logarithm is taken. Above the diagonal stays zero, which is what makes the same tile
         * serve as the CAUSAL MASK on the y path. */
        plane[row * FT_TT_OP_TILE_DIM + row] = 1.0f;
        while (column > 0u) {
            const float step = (column <= chunk_rows - 1u) ? decay[column] : 1.0f;

            plane[row * FT_TT_OP_TILE_DIM + (column - 1u)] =
                plane[row * FT_TT_OP_TILE_DIM + column] * step;
            column--;
        }
    }
    if (ft_tt_tilize_strip(plane, sizeof(float), FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM,
                           (size_t)FT_TT_OP_TILE_DIM * sizeof(float),
                           out_tile) != FT_TT_TILIZE_OK)
        return FT_TT_EINVAL;
    return FT_TT_OK;
}

int ft_tt_op_gdnc_build_gate(const float *decay, const float *beta, uint32_t chunk_rows,
                             uint32_t state_size, float *out_floats, size_t out_count)
{
    const uint32_t rows = FT_TT_OP_TILE_DIM;
    float scale;
    float last;
    uint32_t row;

    if (!decay || !beta || !out_floats || state_size == 0u)
        return FT_TT_EINVAL;
    if (chunk_rows == 0u || chunk_rows > FT_TT_OP_GDNC_MAX_CHUNK_ROWS)
        return FT_TT_EINVAL;
    if (out_count < FT_TT_OP_GDNC_GATE_STRIDE)
        return FT_TT_EINVAL;

    scale = 1.0f / sqrtf((float)state_size);

    /* P over the LIVE rows only, and then FROZEN. A ragged chunk's rows at and above
     * `chunk_rows` carry zero q, k and v, so what they must not do is move the carried state's
     * decay -- `G` is `P_{chunk_rows-1}` and not `P_31`. This is the whole of the ragged-chunk
     * handling, and it is why no kernel has a `chunk_rows` branch in PHASE DELTA. */
    for (row = 0; row < rows; row++) {
        const float step = (row < chunk_rows) ? decay[row] : 1.0f;

        out_floats[FT_TT_OP_GDNC_GATE_P * rows + row] =
            (row == 0u) ? step : out_floats[FT_TT_OP_GDNC_GATE_P * rows + row - 1u] * step;
    }
    last = out_floats[FT_TT_OP_GDNC_GATE_P * rows + chunk_rows - 1u];

    /* ---------------------------------------------------------------------------
     * F IS ITS OWN BACKWARD PRODUCT RECURRENCE, AND `last / P_t` IS THE BUG IT REPLACES
     * ---------------------------------------------------------------------------
     * `F_t = prod_{r = t+1 .. chunk_rows-1} decay_r` — mathematically `P_{chunk_rows-1}/P_t`,
     * and that spelling is the one thing in this file that is not a product recurrence.
     *
     * MEASURED ON SILICON 2026-08-21, at Ornith's own decays and C = 32: P UNDERFLOWS. The
     * decay is `exp(ssm_a * softplus(alpha))` and `ssm_a` reaches the tens, so a head can
     * carry a decay of 1e-7 — and 32 of those multiply to 1e-224, which is ZERO in fp32 after
     * about five positions. `last / P_t` is then `0/0 = NaN`, the NaN reaches the carried
     * state, and the whole prefill is garbage. It is INVISIBLE at small C — at C = 4 the same
     * decay gives P_3 = 1.5e-28, comfortably normal — which is exactly the shape the arm
     * showed: `cmp` IDENTICAL to the control at 4 rows a pass and a completely different
     * completion at 32.
     *
     * Nothing had exercised it: `tests/test_gdnc_ops.c` and `tests/test_phase6_gdnc.c` both
     * draw decays in [0.90, 0.999], where `0.90^32 = 0.034` and no denominator is ever small.
     * The suites are green at C = 32 and the model is not.
     *
     * The backward recurrence has no denominator at all: `F_{chunk_rows-1} = 1` and
     * `F_t = F_{t+1} * decay_{t+1}`, so every entry is a product of decays in (0,1], every
     * partial is in (0,1], and an underflow gives the ZERO the mathematics asks for instead
     * of a NaN. It is the same argument `ft_tt_op_gdnc_build_decay()` is built on, applied to
     * the one column that was left out of it.
     *
     * P, PS and G still underflow to zero and that is CORRECT rather than tolerated: a decay
     * that small really does annihilate the incoming state, and zero is the value the
     * recurrence wants. Only a ratio could turn it into a NaN.
     * --------------------------------------------------------------------------- */
    out_floats[FT_TT_OP_GDNC_GATE_F * rows + chunk_rows - 1u] = 1.0f;
    for (row = chunk_rows - 1u; row > 0u; row--)
        out_floats[FT_TT_OP_GDNC_GATE_F * rows + row - 1u] =
            out_floats[FT_TT_OP_GDNC_GATE_F * rows + row] * decay[row];

    for (row = 0; row < rows; row++) {
        const float p = out_floats[FT_TT_OP_GDNC_GATE_P * rows + row];

        out_floats[FT_TT_OP_GDNC_GATE_PS * rows + row] = scale * p;
        /* beta ZERO above the chunk: `U`'s rows there are then exactly zero rather than
         * merely small, so the carried state takes no contribution from a position that does
         * not exist. */
        out_floats[FT_TT_OP_GDNC_GATE_B * rows + row] = (row < chunk_rows) ? beta[row] : 0.0f;
        if (row >= chunk_rows)
            out_floats[FT_TT_OP_GDNC_GATE_F * rows + row] = 1.0f;
        out_floats[FT_TT_OP_GDNC_GATE_G * rows + row] = last;
    }
    return FT_TT_OK;
}

int ft_tt_op_gdnc_build_gate_at(const float *decay, const float *beta, uint32_t first_row,
                                uint32_t chunk_rows, uint32_t state_size, float *out_floats,
                                size_t out_count)
{
    const uint32_t rows = FT_TT_OP_TILE_DIM;
    const uint32_t last_row = first_row + chunk_rows - 1u;
    float scale;
    float last;
    uint32_t row;

    if (!decay || !beta || !out_floats || state_size == 0u || chunk_rows == 0u)
        return FT_TT_EINVAL;
    if (first_row + chunk_rows > FT_TT_OP_GDNC_MAX_CHUNK_ROWS)
        return FT_TT_EINVAL;
    if ((first_row % FT_TT_OP_GDNC_SUB_CHUNK_ALIGN) != 0u)
        return FT_TT_EINVAL;
    if (out_count < FT_TT_OP_GDNC_GATE_STRIDE)
        return FT_TT_EINVAL;
    /* THE UNSLICED CASE IS THE SHIPPED FUNCTION, for the reason build_decay_at() states. */
    if (first_row == 0u)
        return ft_tt_op_gdnc_build_gate(decay, beta, chunk_rows, state_size, out_floats,
                                        out_count);

    scale = 1.0f / sqrtf((float)state_size);

    /* `P` OVER THE SUB-CHUNK'S OWN ROWS, AND EXACTLY ZERO EVERYWHERE ELSE. Zero and not the
     * ragged tail's frozen `P_{last}`: `Y`'s first term is `scale * diag(P) Q N_0^T` and the
     * writer lands whole tile suffixes, so a nonzero `P` on a row this descriptor does not own
     * writes a whole chunk's worth of `Q N_0^T` into another sub-chunk's answer. */
    for (row = 0; row < rows; row++) {
        if (row < first_row || row > last_row) {
            out_floats[FT_TT_OP_GDNC_GATE_P * rows + row] = 0.0f;
            continue;
        }
        out_floats[FT_TT_OP_GDNC_GATE_P * rows + row] =
            (row == first_row) ? decay[row - first_row]
                               : out_floats[FT_TT_OP_GDNC_GATE_P * rows + row - 1u] *
                                     decay[row - first_row];
    }
    last = out_floats[FT_TT_OP_GDNC_GATE_P * rows + last_row];

    /* `F` IS THE SAME BACKWARD PRODUCT RECURRENCE, run inside the sub-chunk. No denominator,
     * every partial in (0,1], and an underflow gives the zero the mathematics asks for — see
     * ft_tt_op_gdnc_build_gate() for what the `last / P_t` spelling cost on silicon. */
    for (row = 0; row < rows; row++)
        out_floats[FT_TT_OP_GDNC_GATE_F * rows + row] = 1.0f;
    for (row = last_row; row > first_row; row--)
        out_floats[FT_TT_OP_GDNC_GATE_F * rows + row - 1u] =
            out_floats[FT_TT_OP_GDNC_GATE_F * rows + row] * decay[row - first_row];

    for (row = 0; row < rows; row++) {
        const bool live = (row >= first_row && row <= last_row);

        out_floats[FT_TT_OP_GDNC_GATE_PS * rows + row] =
            scale * out_floats[FT_TT_OP_GDNC_GATE_P * rows + row];
        /* beta ZERO off the sub-chunk: row t of `A` is then exactly `e_t`, so row t of `A^-1`
         * is exactly `e_t`, so `U` row t is exactly zero and the carried state takes no
         * contribution from a position this descriptor does not own. */
        out_floats[FT_TT_OP_GDNC_GATE_B * rows + row] = live ? beta[row - first_row] : 0.0f;
        out_floats[FT_TT_OP_GDNC_GATE_G * rows + row] = last;
    }
    return FT_TT_OK;
}

/** The device's own view of a DELTA request. */
static ftgc_shape delta_shape_of(const ft_tt_op_gdnc_request *request)
{
    ftgc_shape shape;

    memset(&shape, 0, sizeof(shape));
    shape.phase = request->phase;
    shape.chunk_rows = request->chunk_rows;
    shape.conv_taps = request->conv_taps;
    shape.state_tiles = request->state_tiles;
    shape.v_heads = request->v_heads;
    shape.k_heads = request->k_heads;
    shape.head_first = 0u;
    shape.head_count = request->v_heads;
    shape.first_row = request->first_row;
    shape.tile_bytes = FT_TT_OP_GDNC_FP32_TILE_BYTES;
    return shape;
}

size_t ft_tt_op_gdnc_gate_bytes(const ft_tt_op_gdnc_request *request)
{
    if (!ft_tt_op_gdnc_shape_is_supported(request) ||
        request->phase != FT_TT_OP_GDNC_PHASE_DELTA)
        return 0u;
    return (size_t)request->v_heads * FT_TT_OP_GDNC_GATE_STRIDE * sizeof(float);
}

size_t ft_tt_op_gdnc_delta_constant_bytes(const ft_tt_op_gdnc_request *request)
{
    ftgc_shape shape;

    if (!ft_tt_op_gdnc_shape_is_supported(request) ||
        request->phase != FT_TT_OP_GDNC_PHASE_DELTA)
        return 0u;
    shape = delta_shape_of(request);
    return gdnc_tile_bytes(ftgc_delta_constant_tiles(&shape));
}

size_t ft_tt_op_gdnc_delta_input_bytes(const ft_tt_op_gdnc_request *request)
{
    ftgc_shape shape;

    if (!ft_tt_op_gdnc_shape_is_supported(request) ||
        request->phase != FT_TT_OP_GDNC_PHASE_DELTA)
        return 0u;
    shape = delta_shape_of(request);
    return gdnc_tile_bytes(ftgc_delta_input_tiles(&shape));
}

size_t ft_tt_op_gdnc_delta_output_bytes(const ft_tt_op_gdnc_request *request)
{
    ftgc_shape shape;

    if (!ft_tt_op_gdnc_shape_is_supported(request) ||
        request->phase != FT_TT_OP_GDNC_PHASE_DELTA)
        return 0u;
    shape = delta_shape_of(request);
    return gdnc_tile_bytes(ftgc_delta_y_tiles(&shape));
}

size_t ft_tt_op_gdnc_delta_state_bytes(const ft_tt_op_gdnc_request *request)
{
    ftgc_shape shape;

    if (!ft_tt_op_gdnc_shape_is_supported(request) ||
        request->phase != FT_TT_OP_GDNC_PHASE_DELTA)
        return 0u;
    shape = delta_shape_of(request);
    return gdnc_tile_bytes(ftgc_delta_state_tiles(&shape));
}
