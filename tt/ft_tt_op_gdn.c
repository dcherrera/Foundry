/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_gdn.c
 * @brief Implementation of the Gated-Delta-Net host driver. See ft_tt_op_gdn.h.
 *
 * @status Parameter layout, CB placement and the per-operand DRAM sizes a host stager
 *   allocates from — pure host arithmetic, pinned off hardware by tests/test_gdn_ops.c.
 *   Pantheon's `src/accel/tt/pn_tt_gdn.c` is the first caller; nothing here has run on the
 *   card, because the on-card gate has not been run.
 * @issues None
 * @todo None
 */

#include "ft_tt_op_gdn.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_op_l1_layout.h"

/* The device's L1 pointer attribute belongs to the SFPI GCC fork, not to the contract; the
 * header guards it so a HOST compiler can read the file, and tests/test_gdn_ops.c already
 * uses this exact escape. Including it here is what lets the sizing functions at the bottom
 * of this file compute their tile counts with the SAME `ftg_*` accessors both dataflow
 * kernels compute their offsets with, instead of respelling the layout a third time. */
#define FTQ_L1_PTR
#include "kernels/resident_gdn_protocol.h"

/** Pages of the streamed operands. Two blocks so the reader can fill one while the compute
 *  thread drains the other — the same policy every streamed CB in this backend uses. */
#define GDN_CB_BLOCKS 2u

/** Pages of a staging or output CB. Two, not one: a single page serialises the producer
 *  against the consumer on every tile. */
#define GDN_CB_PAGES_SMALL 2u

/**
 * The CONV block depth THIS HOST places its circular buffers for.
 *
 * ONE BINARY FOR THE WHOLE SWEEP, WHICH IS WHY THIS IS A KNOB AND NOT ONLY A MACRO. Every
 * per-descriptor `us each` this backend prints is host-bracketed — it carries one host wait
 * per awaited member — so figures are comparable across ARMS OF ONE BINARY and not across
 * binaries. A depth sweep that needed a host rebuild per rung would be comparing binaries,
 * and the only way to hold the host constant while the kernels move is to make the host's
 * side of the contract data. The kernel side stays compile-time, because the MATH thread
 * cannot read a runtime value at all.
 *
 * THE ENV DOES NOT WEAKEN THE CONTRACT, IT IS WHAT MAKES IT CHECKABLE. Whatever this returns
 * is written into byte 3 of the CONV-taps word, and both kernels refuse a CONV descriptor
 * that does not match the constant they were built with — so asking for a depth the armed
 * program directory does not carry is a REFUSAL, not a silently-slow arm.
 *
 * AN UNUSABLE VALUE RETURNS 0 AND EVERY ENTRY POINT REFUSES. It does not fall back to the
 * compile-time default: a knob that silently ignores what it was told is how a sweep measures
 * its control five times.
 *
 * Read once. Thread-safe after the first call, which happens at arm time on one thread.
 */
uint32_t ft_tt_op_gdn_conv_depth_parse(const char *value)
{
    char *end = NULL;
    unsigned long parsed;

    if (!value || !*value)
        return FT_TT_OP_GDN_CONV_DEPTH;
    parsed = strtoul(value, &end, 10);
    /* NO SILENT FALLBACK TO THE DEFAULT. A knob that ignores what it was told is how a sweep
     * measures its control five times and reports a ladder. */
    if (end == value || *end != '\0' || parsed < 1u || parsed > 255u)
        return 0u;
    return (uint32_t)parsed;
}

uint32_t ft_tt_op_gdn_conv_depth(void)
{
    static uint32_t resolved;
    static bool known;
    const char *value;

    if (known)
        return resolved;
    known = true;
    value = getenv(FT_TT_OP_GDN_CONV_DEPTH_ENV);
    resolved = ft_tt_op_gdn_conv_depth_parse(value);
    /* ONE LINE PER PROCESS, ALWAYS. An arm that reports a depth is worth more than an arm
     * whose depth is inferred from the environment line of the runner that launched it —
     * this is the line that says which rung actually placed. */
    if (resolved == 0u)
        fprintf(stderr,
                "ft_tt: gdn CONV block depth REFUSED — %s=\"%s\" is not a depth in 1..255, so "
                "every GDN descriptor refuses rather than quietly placing for %u\n",
                FT_TT_OP_GDN_CONV_DEPTH_ENV, value ? value : "",
                (unsigned)FT_TT_OP_GDN_CONV_DEPTH);
    else
        fprintf(stderr,
                "ft_tt: gdn CONV block depth %u (compile-time default %u, %s%s) — declared on "
                "every CONV descriptor; a program built at another depth refuses\n",
                (unsigned)resolved, (unsigned)FT_TT_OP_GDN_CONV_DEPTH,
                FT_TT_OP_GDN_CONV_DEPTH_ENV, (value && *value) ? " set" : " unset");
    return resolved;
}

static bool operand_is_set(const ft_tt_op_operand *operand)
{
    return operand->tile_bytes != 0u;
}

/** Is this a taps count a CONV descriptor could carry? A residency placed from a request
 *  that does not name one is sized for PHASE DELTA alone. */
static bool conv_taps_are_usable(uint32_t taps)
{
    return taps >= 2u && taps <= 8u;
}

/** The v-heads a descriptor owns. A `head_count` of 0 means the whole descriptor — the
 *  convention the device decodes, and what every caller written before the head slice
 *  leaves the field at. */
static uint32_t gdn_head_count(const ft_tt_op_gdn_request *request)
{
    return request->head_count ? request->head_count : request->v_heads;
}

static uint32_t gdn_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

/** The smallest multiple of `multiple` at least as large as `value`. Every call site below
 *  passes a tile count or a block depth, so `multiple` is never zero. */
static uint32_t gdn_round_up(uint32_t value, uint32_t multiple)
{
    return ((value + multiple - 1u) / multiple) * multiple;
}

/**
 * Lowest common multiple — for a buffer that has to close on TWO block grids at once.
 *
 * CB_STATE is pushed in chunks of `state_tiles` by PHASE DELTA and in chunks of the CONV
 * block depth by PHASE CONV, and `ft_tensix_cb_push_grid_is_safe()` refuses unless the page
 * count is a whole multiple of the chunk. One binary answers both phases out of one
 * placement, so the buffer has to satisfy both.
 */
static uint32_t gdn_lcm(uint32_t a, uint32_t b)
{
    uint32_t x = a;
    uint32_t y = b;

    while (y != 0u) {
        const uint32_t remainder = x % y;

        x = y;
        y = remainder;
    }
    return (a / x) * b;
}

uint32_t ft_tt_op_gdn_conv_slice_granule(uint32_t state_size, uint32_t conv_taps)
{
    const uint32_t state_tiles = state_size / FT_TT_OP_TILE_DIM;
    const uint32_t depth = ft_tt_op_gdn_conv_depth();
    /* CB_STATE: pass A pushes `taps-1` window pages a column, pass B `taps-2`. */
    const uint32_t window_pages = 2u * conv_taps - 3u;

    if (state_tiles == 0u || !conv_taps_are_usable(conv_taps) || depth == 0u)
        return 0u;
    /* The fewest columns whose CB_STATE traffic is a whole number of DELTA's `state_tiles`
     * grid. CB_PROD (`taps-1` pages a column, DELTA grid `dst_batch`) needs nothing more:
     * `dst_batch` divides `state_tiles` (ftg_dst_batch()) and so does every granule here. */
    return gdn_lcm(depth, gdn_lcm(window_pages, state_tiles) / window_pages);
}

/**
 * Is every operand this request names an fp32 tile?
 *
 * THE DEVICE CANNOT CATCH THIS FOR ITSELF and that is the whole reason it is here. The
 * kernels compare `kGdnTileBytes` against the TILE_BYTES they were built with — but the
 * parameter block has ONE such word for FIVE DRAM endpoints, so the check only ever sees the
 * operand this encoder happened to take it from, and every other operand goes unexamined.
 * A bf16 activation beside an fp32 state passed that check and then had 2048-byte circular
 * buffers placed under a reader issuing 4096-byte tile reads. Refusing here is what makes the
 * device's one-word check sufficient.
 */
static bool operands_are_fp32(const ft_tt_op_gdn_request *request)
{
    const ft_tt_op_operand *operands[6] = {
        &request->activation_in, &request->activation_out, &request->conv_weight,
        &request->gate,          &request->state_conv,     &request->state_recurrent};
    unsigned index;

    for (index = 0; index < 6u; index++) {
        if (operand_is_set(operands[index]) &&
            operands[index]->tile_bytes != FT_TT_OP_GDN_FP32_TILE_BYTES)
            return false;
    }
    return true;
}

/** Pack a NoC coordinate pair into one word. See the note in ft_tt_op_gdn.h on why. */
static uint32_t pack_xy(const ft_tt_op_operand *operand)
{
    return (uint32_t)operand->x | ((uint32_t)operand->y << 16);
}

/**
 * G2a: point a CONV descriptor's stream read at the bf16 projection strip (word 4 its address,
 * word 8 its coordinate; word 8 is DELTA's gate and unused by CONV). Strip mode only (the
 * reader widens row halves), bf16 tiles only, CONV only.
 */
static int encode_stream_bf16(const ft_tt_op_gdn_request *request, uint32_t *out_words)
{
    if (!operand_is_set(&request->stream_bf16))
        return FT_TT_OK;
    if (request->phase != FT_TT_OP_GDN_PHASE_CONV ||
        (request->flags & FT_TT_OP_GDN_FLAG_CONV_STRIP) == 0u ||
        request->stream_bf16.tile_bytes != FT_TT_OP_GDN_BF16_TILE_BYTES)
        return FT_TT_EINVAL;
    out_words[FT_TT_OP_GDN_ACTIVATION_IN] = request->stream_bf16.address;
    out_words[FT_TT_OP_GDN_CONSTANT_GATE] = pack_xy(&request->stream_bf16);
    out_words[FT_TT_OP_GDN_FLAGS] |= FT_TT_OP_GDN_FLAG_CONV_BF16_IN;
    return FT_TT_OK;
}

/** A CONV slice of whole granules (FT_TT_OP_GDN_CONV_SLICE_GRANULE). */
static bool conv_slice_is_granular(const ft_tt_op_gdn_request *request)
{
    const uint32_t granule =
        ft_tt_op_gdn_conv_slice_granule(request->state_size, request->conv_taps);

    return granule != 0u && request->col_count % granule == 0u;
}

int ft_tt_op_gdn_shape_is_supported(const ft_tt_op_gdn_request *request)
{
    if (!request)
        return 0;
    if (request->phase != FT_TT_OP_GDN_PHASE_CONV && request->phase != FT_TT_OP_GDN_PHASE_DELTA)
        return 0;
    if (request->state_size == 0u || (request->state_size % FT_TT_OP_TILE_DIM) != 0u)
        return 0;
    /* The state tile count is a BYTE of the shape word, so S/32 must fit in one. 8 is the
     * ceiling this binary is willing to claim rather than the byte's 255: a [256,256] state
     * per head is 256 KiB of L1 traffic a head and has never been shaped or measured. */
    if (request->state_size / FT_TT_OP_TILE_DIM > 8u)
        return 0;
    if (request->v_heads == 0u || request->v_heads > 255u)
        return 0;
    if (request->k_heads == 0u || request->k_heads > 255u)
        return 0;
    /* The GQA repeat is `kh = vh % k_heads`, which is only a repeat when it divides.
     * Checked here AND on the device: getting this mapping backwards is a known bug in this
     * architecture, not a hypothetical one. */
    if ((request->v_heads % request->k_heads) != 0u)
        return 0;
    /* THE HEAD SLICE. `head_count` 0 is "all of them"; anything else has to fit inside
     * v_heads, and both fields have to fit the BYTE each occupies in the CONV-taps word —
     * a slice that silently truncated would address another member's state. */
    if (request->head_first > 255u || request->head_count > 255u)
        return 0;
    if (request->head_count != 0u &&
        request->head_first + request->head_count > request->v_heads)
        return 0;
    if (request->head_count == 0u && request->head_first != 0u)
        return 0;
    if (request->phase == FT_TT_OP_GDN_PHASE_CONV) {
        /* PHASE CONV divides by CHANNEL, not by head, so a HEAD slice on it is refused — its
         * own slice is the column range below. */
        if (request->head_count != 0u)
            return 0;
        if (request->conv_dim == 0u || (request->conv_dim % FT_TT_OP_TILE_DIM) != 0u)
            return 0;
        /* The stream width rides in the low half-word of its word; the high half carries
         * the column slice's high bytes (kGdnByteColFirstHigh). */
        if (request->conv_dim / FT_TT_OP_TILE_DIM > FT_TT_OP_GDN_COL_MAX)
            return 0;
        /* THE COLUMN SLICE, the convolution's own: tile columns at the full plane stride
         * (the kernels index `tap * conv_tile_cols + column` with an ABSOLUTE column). It has
         * to fit the stream — so both fields fit their 16 bits — and a 0 count means the
         * whole. */
        if (request->col_count == 0u && request->col_first != 0u)
            return 0;
        if (request->col_count != 0u &&
            request->col_first + request->col_count > request->conv_dim / FT_TT_OP_TILE_DIM)
            return 0;
        /* A slice whose CONV traffic leaves CB_STATE or CB_PROD off PHASE DELTA's page grid —
         * never run clean on silicon (FT_TT_OP_GDN_CONV_SLICE_GRANULE). The whole stream (a 0
         * count) is not a slice and keeps its shipped bytes. */
        if (request->col_count != 0u && !conv_slice_is_granular(request))
            return 0;
        /* A one-tap conv has no window to roll, which would leave the carried conv state
         * write-only — a shape that was almost certainly meant to be something else. */
        if (request->conv_taps < 2u || request->conv_taps > 8u)
            return 0;
    } else if (request->col_count != 0u || request->col_first != 0u) {
        /* A DELTA descriptor reads bytes 1 and 2 as its HEAD slice — a column slice there
         * would be read as heads. Refused rather than reinterpreted. */
        return 0;
    }
    /* MULTI-ROW (PM 5): DELTA only, every row inside one tile, and the row-major gate blocks
     * of every row inside the reader's one scratch page. */
    if (request->row_count > 1u &&
        (request->row_count > FT_TT_OP_GDN_MULTI_ROW_MAX || request->row + request->row_count > 32u))
        return 0;
    if (request->row_count > 1u && request->phase == FT_TT_OP_GDN_PHASE_DELTA &&
        request->row_count * ftg_gate_stride(request->v_heads) > FT_TT_OP_GDN_FP32_TILE_BYTES)
        return 0;
    /* THE TABLE GATES build a DELTA gate block; a CONV descriptor has none. */
    if (request->phase == FT_TT_OP_GDN_PHASE_CONV &&
        (request->flags & FT_TT_OP_GDN_FLAG_TABLE) != 0u)
        return 0;
    /* MULTI-ROW CONV (PM 6): strip mode only — the row select reads and writes row r. */
    if (request->row_count > 1u && request->phase == FT_TT_OP_GDN_PHASE_CONV &&
        (request->flags & FT_TT_OP_GDN_FLAG_CONV_STRIP) == 0u)
        return 0;
    /* CONV ROW BATCH: a multi-row CONV whose rows all sit in faces 0-1, read from the fp32
     * stream. The device refuses the same shapes (ftg_conv_batch_is_supported) in the reader
     * and writer only, so a shape refused there would leave the compute thread waiting: the
     * host is the refusal that matters. */
    if (request->conv_row_batch != 0u &&
        (request->phase != FT_TT_OP_GDN_PHASE_CONV || request->row_count < 2u ||
         request->row + request->row_count > FT_TT_OP_GDN_CONV_BATCH_ROW_LIMIT ||
         operand_is_set(&request->stream_bf16)))
        return 0;
    return 1;
}

int ft_tt_op_gdn_row_extension(const ft_tt_op_gdn_request *request,
                               uint32_t out_words[FT_TT_OP_GDN_WORDS])
{
    uint32_t r;

    const ft_tt_op_operand *bank;

    if (!request || !out_words || !ft_tt_op_gdn_shape_is_supported(request) ||
        request->row_count < 2u)
        return FT_TT_EINVAL;
    /* Row r's carried state lands in the bank the descriptor's state word names: the
     * recurrent state (DELTA) or the conv window (CONV). */
    bank = request->phase == FT_TT_OP_GDN_PHASE_CONV ? &request->state_conv
                                                     : &request->state_recurrent;
    memset(out_words, 0, sizeof(uint32_t) * FT_TT_OP_GDN_WORDS);
    for (r = 0; r < request->row_count; r++) {
        const ft_tt_op_operand *out = &request->state_rows_out[r];

        if (!operand_is_set(out) || pack_xy(out) != pack_xy(bank))
            return FT_TT_EINVAL;
        out_words[r] = out->address;
    }
    return FT_TT_OK;
}

/** CONV passes the compute thread runs: one a row, or ONE for a row batch. */
static uint32_t conv_pass_count(const ft_tt_op_gdn_request *request)
{
    if (request->conv_row_batch != 0u || request->row_count < 2u)
        return 1u;
    return request->row_count;
}

/**
 * Tiles this descriptor's writer will emit — INCLUDING THE CARRIED STATE.
 *
 * This op is the only one in the backend whose operands are read AND written, and the
 * updated state is a RESULT, not a side effect. Resident-protocol rule 1 says completion is
 * published by the writer, when the result has left L1; rule 5 says an abandoned descriptor
 * retires nothing. If the reader wrote the state back, the state would leave L1 under the
 * processor that owns `tail` while completion was published by the processor that owns
 * `completed_count` — two writers for one result, which is exactly the lie those rules
 * exist to make impossible. So the state goes out through CB_OUT and is counted here.
 *
 * It costs nothing in page size because every CB of this family is Float32
 * (kernels/cb_tables/gdn_float32.cbtable): a state tile and a y tile are both 4096 B, so one
 * CB_OUT page size serves both and the writer needs no second buffer.
 *
 * THE WRITER RE-DERIVES THE SPLIT rather than being told it. `FT_TT_QUEUE_PARAM_WORDS` is 16
 * for every family and a seventeenth word changes the shared slot stride, so there is no room
 * for a "state tiles" word — but there is no need for one either, because `v_heads` and
 * `state_size` are already in the shape word the writer decodes.
 */
static uint32_t resolve_tile_count(const ft_tt_op_gdn_request *request)
{
    const uint32_t state_tiles = request->state_size / FT_TT_OP_TILE_DIM;

    if (request->phase == FT_TT_OP_GDN_PHASE_CONV) {
        const uint32_t conv_tile_cols = request->conv_dim / FT_TT_OP_TILE_DIM;

        /* conv_out is the whole convolved stream, one row of tiles; the rolled window that
         * follows it is taps-1 deep over the same stream — once a row (PM 6 multi-row). */
        return (conv_tile_cols + conv_tile_cols * (request->conv_taps - 1u)) *
               conv_pass_count(request);
    }
    /* Per v-head, per tile-row: `state_tiles` updated state tiles, then ONE y column tile.
     *
     * THIS COUNTS CB_OUT PAGES, NOT DRAM TILES, and the two differ for y. In DRAM y is one
     * ROW per v-head — v_heads/32 row blocks of S/32 tile columns, 4 tiles at the shipped
     * shape — because the writer scatters each head's column-0 answer into a row, which is
     * what makes the gated norm that follows an ordinary RMS norm over 4 tile columns. But
     * every head's answer leaves the (e) reduce as its OWN column tile, so
     * `v_heads * state_tiles` = 128 of them cross the buffer to produce those 4. Counting the
     * 4 here would starve the writer by 124 pages and hang the triad on its bound.
     *
     * The interleave is forced rather than chosen: emitting all the state and then all of y
     * would mean holding every head's y live, which is 128 column tiles = 512 KiB of L1.
     * Emitting each head's tile-row as it is finished bounds CB_OUT at two pages instead.
     *
     * THIS MEMBER'S HEADS, not every head: a head-split descriptor produces only the pages
     * of the slice it owns, and the writer's trip count is derived from the SAME slice. A
     * count taken from `v_heads` here would leave the writer waiting for pages a sliced
     * compute thread is never going to push, on its bound — and a triad that exits on a
     * circular-buffer spin bound WEDGES its Tensix permanently. */
    return gdn_head_count(request) * state_tiles * (state_tiles + 1u) *
           (request->row_count > 1u ? request->row_count : 1u);
}

int ft_tt_op_gdn_parameters(const ft_tt_op_gdn_request *request,
                           uint32_t out_words[FT_TT_OP_GDN_WORDS])
{
    uint32_t scale_bits;
    float scale;

    if (!request || !out_words)
        return FT_TT_EINVAL;
    /* An unusable depth knob refuses HERE, before a descriptor exists, rather than declaring
     * a zero the kernels would refuse one descriptor at a time. */
    if (ft_tt_op_gdn_conv_depth() == 0u)
        return FT_TT_EINVAL;
    if (!ft_tt_op_gdn_shape_is_supported(request))
        return FT_TT_EINVAL;
    if (!operand_is_set(&request->activation_in) || !operand_is_set(&request->activation_out))
        return FT_TT_EINVAL;
    if (request->phase == FT_TT_OP_GDN_PHASE_CONV &&
        (!operand_is_set(&request->conv_weight) || !operand_is_set(&request->state_conv)))
        return FT_TT_EINVAL;
    if (request->phase == FT_TT_OP_GDN_PHASE_DELTA &&
        (!operand_is_set(&request->gate) || !operand_is_set(&request->state_recurrent)))
        return FT_TT_EINVAL;
    if (!operands_are_fp32(request))
        return FT_TT_EINVAL;
    /* THE ACTIVATION PAIR SHARES ONE NoC COORDINATE WORD, so both buffers must sit on the
     * same DRAM endpoint. `FT_TT_QUEUE_PARAM_WORDS` is 16 for every family and this op has
     * five endpoints, so `kGdnActivationXy` covers the input AND the output — see the packing
     * note in resident_gdn_protocol.h. NOTHING ON THE DEVICE CAN CATCH A VIOLATION: the
     * reader reads the right bank and the writer puts the answer at the OUTPUT's offset
     * inside the INPUT's bank, so the descriptor completes normally, the output buffer is
     * never written, and the host reads back whatever was already there. Measured on silicon
     * 2026-08-18 — a stager that round-robined banks produced exactly that, and what named it
     * was that the conv answer came back BIT-IDENTICAL with the kernel's SiLU removed. */
    if (request->activation_in.x != request->activation_out.x ||
        request->activation_in.y != request->activation_out.y)
        return FT_TT_EINVAL;

    memset(out_words, 0, sizeof(uint32_t) * FT_TT_OP_GDN_WORDS);

    out_words[FT_TT_OP_GDN_SHAPE] =
        (request->phase & 0xFFu) |
        (((request->state_size / FT_TT_OP_TILE_DIM) & 0xFFu) << 8) |
        ((request->v_heads & 0xFFu) << 16) | ((request->k_heads & 0xFFu) << 24);
    /* Low half-word the stream's tile columns; bytes 2 and 3 the CONV column slice's high
     * bytes (kGdnByteColFirstHigh), zero for any slice under 256 columns -- so every narrow
     * descriptor is the same bytes it always was. */
    out_words[FT_TT_OP_GDN_CONV_TILE_COLS] =
        request->phase == FT_TT_OP_GDN_PHASE_CONV
            ? (request->conv_dim / FT_TT_OP_TILE_DIM) |
                  (((request->col_first >> 8) & 0xFFu) << 16) |
                  (((request->col_count >> 8) & 0xFFu) << 24)
            : 0u;
    /* Byte 0 is the taps (CONV only); bytes 1 and 2 are the DELTA head slice; byte 3 is THIS
     * HOST'S OWN CONV block depth, which both kernels compare against the constant they were
     * compiled with. The depth is declared on CONV descriptors ONLY, because PHASE DELTA
     * never reads it — so an unsliced DELTA descriptor still leaves the whole word zero and
     * the shipped DELTA encoding and this one are still the same bytes. See kGdnByteConvDepth
     * in kernels/resident_gdn_protocol.h for the drift this catches. */
    out_words[FT_TT_OP_GDN_CONV_TAPS] =
        (request->phase == FT_TT_OP_GDN_PHASE_CONV ? (request->conv_taps & 0xFFu) : 0u) |
        (request->phase == FT_TT_OP_GDN_PHASE_CONV
             ? ((request->col_first & 0xFFu) << 8) | ((request->col_count & 0xFFu) << 16)
             : ((request->head_first & 0xFFu) << 8) | ((request->head_count & 0xFFu) << 16)) |
        (request->phase == FT_TT_OP_GDN_PHASE_CONV
             ? ((ft_tt_op_gdn_conv_depth() & 0xFFu) << 24)
             : 0u);

    out_words[FT_TT_OP_GDN_ACTIVATION_XY] = pack_xy(&request->activation_in);
    out_words[FT_TT_OP_GDN_ACTIVATION_IN] = request->activation_in.address;
    out_words[FT_TT_OP_GDN_ACTIVATION_OUT] = request->activation_out.address;

    if (request->phase == FT_TT_OP_GDN_PHASE_CONV) {
        out_words[FT_TT_OP_GDN_CONSTANT_XY] = pack_xy(&request->conv_weight);
        out_words[FT_TT_OP_GDN_CONSTANT_WEIGHT] = request->conv_weight.address;
        out_words[FT_TT_OP_GDN_STATE_XY] = pack_xy(&request->state_conv);
        out_words[FT_TT_OP_GDN_STATE_CONV] = request->state_conv.address;
        if (operand_is_set(&request->state_out))
            out_words[FT_TT_OP_GDN_STATE_RECURRENT] = request->state_out.address;
    } else {
        out_words[FT_TT_OP_GDN_CONSTANT_XY] = pack_xy(&request->gate);
        out_words[FT_TT_OP_GDN_CONSTANT_GATE] = request->gate.address;
        out_words[FT_TT_OP_GDN_STATE_XY] = pack_xy(&request->state_recurrent);
        out_words[FT_TT_OP_GDN_STATE_RECURRENT] = request->state_recurrent.address;
        if (operand_is_set(&request->state_out))
            out_words[FT_TT_OP_GDN_STATE_CONV] = request->state_out.address;
    }
    if (operand_is_set(&request->state_out) &&
        pack_xy(&request->state_out) != out_words[FT_TT_OP_GDN_STATE_XY])
        return FT_TT_EINVAL; /* one coordinate word: the out state shares the in state's bank */

    /* ONE tile size for the whole family, not whichever operand this branch reached for.
     * CONV used to publish the activation's and DELTA the recurrent state's, so the two
     * phases could disagree about the size of a page in the same nine buffers. Every CB is
     * Float32 and `operands_are_fp32()` has already refused anything else, so there is
     * exactly one answer and this is it. */
    out_words[FT_TT_OP_GDN_TILE_BYTES] = FT_TT_OP_GDN_FP32_TILE_BYTES;

    /* scale = 1/sqrt(S), sent as BITS rather than derived on the device: a reciprocal square
     * root computed two ways is two answers that have to agree, and this one multiplies
     * every output element of a recurrence. Spelled with the SAME expression Pantheon's
     * oracle uses — `1.0f / sqrtf((float)S)` in qwen35moe_ssm.c's recur_args — so the host
     * and the oracle cannot drift, and the device does no arithmetic on it at all. */
    scale = 1.0f / sqrtf((float)request->state_size);
    memcpy(&scale_bits, &scale, sizeof(scale_bits));
    out_words[FT_TT_OP_GDN_SCALE_BITS] = scale_bits;

    out_words[FT_TT_OP_GDN_FLAGS] =
        (request->flags & (FT_TT_OP_GDN_FLAG_NOC_REPLICATE | FT_TT_OP_GDN_FLAG_RED_NO_COPY |
                           FT_TT_OP_GDN_FLAG_CONV_STRIP | FT_TT_OP_GDN_FLAG_RED_STRIP_HALF |
                           FT_TT_OP_GDN_FLAG_RED_ROWS_ROW0 | FT_TT_OP_GDN_FLAG_TABLE |
                           FT_TT_OP_GDN_FLAG_STATE_ROUND)) |
        (operand_is_set(&request->state_out) ? FT_TT_OP_GDN_FLAG_STATE_OUT : 0u) |
        ((request->row & 31u) << FT_TT_OP_GDN_ROW_SHIFT) |
        ((request->row_count > 1u && request->conv_row_batch == 0u ? request->row_count & 31u
                                                                    : 0u)
         << FT_TT_OP_GDN_ROW_COUNT_SHIFT) |
        ((request->conv_row_batch != 0u ? request->row_count & 31u : 0u)
         << FT_TT_OP_GDN_CONV_BATCH_SHIFT);
    if (request->row > 31u ||
        (request->row != 0u && request->phase == FT_TT_OP_GDN_PHASE_CONV &&
         (request->flags & FT_TT_OP_GDN_FLAG_CONV_STRIP) == 0u))
        return FT_TT_EINVAL;
    /* CONV packs no recurrent state; the round bias there would be a bit nothing reads. */
    if ((request->flags & FT_TT_OP_GDN_FLAG_STATE_ROUND) != 0u &&
        request->phase != FT_TT_OP_GDN_PHASE_DELTA)
        return FT_TT_EINVAL;
    out_words[FT_TT_OP_GDN_TILE_COUNT] = resolve_tile_count(request);
    if (out_words[FT_TT_OP_GDN_TILE_COUNT] == 0u)
        return FT_TT_EINVAL;
    return encode_stream_bf16(request, out_words);
}

/**
 * Page counts that satisfy EVERY refusal the device makes, at every admitted shape.
 *
 * ---------------------------------------------------------------------------
 * THE GRID THAT MATTERS IS THE CONSUMER'S INDEXED READ, NOT THE PRODUCER'S PUSH
 * ---------------------------------------------------------------------------
 * Every count below used to be rounded onto the PUSH grid alone, and that is only half of
 * the rule. A compute thread that does `cb_wait_front(cb, n)` and then names tiles `0..n-1`
 * gets `fifo_read_pointer + i * page_size` with NO WRAP — the FIFO pointer moves back to the
 * base on an EXACT HIT of the limit and on nothing else. So the front has to be low enough
 * that the whole indexed window still fits, at every position the front can take, which is
 * `pages % n == 0` for the INDEXED chunk `n` of every phase.
 *
 * MEASURED ON SILICON 2026-08-18, and it is this family's whole wrong answer:
 *   - CB_VECTOR held 8 pages and PHASE DELTA consumes `state_tiles + 3` = 7 a head, so the
 *     front walked 0, 7, 6, 5, ... and 24 of 32 heads read their v / decay / beta / scale out
 *     of the buffer and into CB_STAGE. Exactly the 8 heads whose front was 0 or 1 — 0, 7, 8,
 *     15, 16, 23, 24, 31 — scored 4e-3 against the closed form; the other 24 scored 1.0.
 *   - CB_STATE held 8 pages and PHASE CONV indexes `depth * (taps-1)` = 3 window tiles a
 *     block, so 2 blocks in 8 read past the end. It is right at layer 0 of token 0 only,
 *     because the L1 above it is CB_SCALER and that is still zero until the first DELTA
 *     descriptor writes k/q replicas into it — conv scored 6.9e-4 at layer 0 and 1e-2 to
 *     2e-1 at every layer after it.
 *
 * The counts below round onto BOTH grids. What makes the rounding sufficient is that the
 * three kernels REBASE their pointers at the top of every descriptor: a descriptor's traffic
 * is not a whole number of buffers (PHASE DELTA moves 512 CB_STATE pages and PHASE CONV
 * 1280), so without the rebase the next descriptor would start off-grid however the buffer
 * is sized. See `ftg_rebase_*()` in the three kernels.
 *
 * ---------------------------------------------------------------------------
 * THE HOST WAS THE WRONG SIDE, TWICE, AND BOTH WERE INVISIBLE AT ORNITH
 * ---------------------------------------------------------------------------
 * `delta_is_executable()` in resident_gdn_reader.c refuses unless CB_VECTOR holds
 * `state_tiles + kGdnVectorConstants` pages. The old placement gave it
 * `state_tiles * 2`, which is only enough from `state_tiles >= 3` — so S = 32 and S = 64 were
 * admitted by `ft_tt_op_gdn_shape_is_supported()` and refused by the kernel. At Ornith
 * `state_tiles` is 4 and 8 >= 7, which is why nothing showed it.
 *
 * `conv_block_depth()` refuses unless CB_VECTOR holds `2 * depth * taps` weight pages and
 * CB_STATE holds `2 * depth * (taps - 1)` window pages. Sizing BOTH from `state_tiles` —
 * a DELTA-phase number with nothing to do with the convolution — made the CONV phase's
 * legality an accident of the state size. At Ornith 2*1*4 = 8 = 2*state_tiles exactly, so
 * again nothing showed it, and every taps count above 4 was silently refused.
 *
 * ONE PLACEMENT SERVES BOTH PHASES because one binary does, so every count below is the max
 * of the two phases' demands, rounded onto whichever push grids apply.
 *
 * @param taps  the CONV taps, or 0 for a residency that will only ever run PHASE DELTA
 */
static void resolve_cb_pages(uint32_t state_tiles, uint32_t taps,
                             uint32_t out_pages[FT_TT_OP_GDN_CB_COUNT])
{
    const uint32_t depth = ft_tt_op_gdn_conv_depth();
    /* The chunk each phase INDEXES ACROSS, which is the grid its buffer has to close on.
     * A `taps` of 0 means a residency that will only ever run PHASE DELTA, so its CONV
     * grids collapse to 1 and leave the DELTA ones alone. */
    const uint32_t conv_window_chunk = taps ? depth * (taps - 1u) : 1u;
    const uint32_t conv_weight_chunk = taps ? depth * taps : 1u;
    const uint32_t vector_chunk = state_tiles + FT_TT_OP_GDN_VECTOR_CONSTANTS;
    const uint32_t state_grid = gdn_lcm(state_tiles, conv_window_chunk);
    const uint32_t vector_grid = gdn_lcm(vector_chunk, conv_weight_chunk);
    /* PHASE DELTA's DST batch, from the ONE implementation of the clamp both kernels call.
     * It is CB_PROD's indexed-read chunk, and CB_PROD had no grid at all before the batch
     * because every access to it named tile 0. */
    const uint32_t dst_batch = ftg_dst_batch(state_tiles, FT_TT_OP_GDN_DST_BATCH);
    /* MULTI-ROW DELTA's buffers (PM 5) are placed only where they fit beside the CONV ones:
     * at S = 256 the carry alone is 64 pages and 6-8 taps no longer place. Above the bound the
     * placement is the one-row one and the reader refuses a multi-row descriptor (its CB_STAGE
     * and CB_CARRY checks), so the bound is a capacity statement, not a second contract. */
    const bool multi_row = state_tiles <= FT_TT_OP_GDN_MULTI_ROW_STATE_TILES_MAX;

    /* CB_IN. CONV ONLY, and the "only" is load-bearing: the qkv stream, `depth` pages a
     * block, double-buffered. PHASE DELTA's working buffer used to share this index, and a
     * buffer pushed by the READER in one phase and by the COMPUTE in the other has two
     * incompatible sets of books — see FT_TT_OP_GDN_CB_WORK. */
    out_pages[0] = GDN_CB_BLOCKS * depth;
    /* CB_STATE. DELTA: ONE TILE-ROW of a head's state at a time, double-buffered — that is
     * what makes the L1 cost independent of the head count, since (b) through (e) all run
     * inside a tile-row. CONV: `taps-1` window planes of `depth` pages, ALL LIVE AT ONCE
     * because `conv_one_column` names every tap of a column, so `depth*(taps-1)` is an
     * indexed chunk and not just a capacity. */
    out_pages[1] = gdn_round_up(gdn_max(state_tiles * GDN_CB_BLOCKS,
                                        GDN_CB_BLOCKS * conv_window_chunk),
                                state_grid);
    /* CB_SCALER. DELTA only: this head's k replicas then its q replicas, held for every one
     * of its tile-rows and indexed `2*state_tiles` wide. TWO rows' sets (PM 5): a multi-row
     * descriptor's reader builds row r+1's replicas while the compute thread runs row r. */
    out_pages[2] = (multi_row ? 2u : 1u) * state_tiles * GDN_CB_BLOCKS;
    /* CB_VECTOR. DELTA: `state_tiles` v columns plus decay, beta and scale, ALL held for the
     * head and indexed `state_tiles + 3` wide. CONV: `taps` weight planes of `depth`, pushed
     * as ONE chunk and indexed across the whole of it. */
    out_pages[3] = gdn_round_up(gdn_max(GDN_CB_BLOCKS * vector_chunk,
                                        GDN_CB_BLOCKS * conv_weight_chunk),
                                vector_grid);
    /* CB_STAGE. The reader's private scratch: the k, q and v row blocks of one head (a
     * multi-row descriptor builds every row's operands from ONE read of them; the one-row walk
     * uses the first block), plus ONE page above them for the gate block(s) it reads once.
     * Never pushed, so never double. */
    out_pages[4] = (multi_row ? 3u : 1u) * state_tiles + 1u;
    out_pages[5] = GDN_CB_PAGES_SMALL; /* CB_PRED  */
    out_pages[6] = GDN_CB_PAGES_SMALL; /* CB_DELTA */
    /* CB_PROD. CONV: one tap's product, a page at a time. DELTA: the batched (d) stages
     * `dst_batch` rank-1 products here and then INDEXES all of them off the front, so this
     * buffer now has a consumer grid — `dst_batch` must divide it, and it must hold two
     * blocks so the pack of one block is not serialised against the unpack of the last. */
    out_pages[7] = gdn_max(GDN_CB_PAGES_SMALL, GDN_CB_BLOCKS * dst_batch);
    /* CB_WORK. DELTA ONLY: the compute thread's working buffer, one tile-row, reserved a page
     * at a time and INDEXED `state_tiles` wide by both reduces. The front cycles 0 -> tiles
     * -> 0 inside a tile-row — (b) indexes off the decayed row and (e) off the updated one —
     * so it needs `2 * state_tiles` pages and it needs `state_tiles` to divide them. This is
     * the sizing CB_IN used to carry for this phase, moved to a buffer with ONE producer. */
    out_pages[8] = state_tiles * GDN_CB_BLOCKS;
    /* CB_OUT. TWO of the WRITER's barrier groups, because the writer holds a whole group
     * un-popped while its NoC writes drain — a page popped with its write still in flight is
     * a page the compute thread may pack over. Sized for the highest rung the family
     * supports rather than the rung the kernels happen to be built at, exactly as CB_PROD is
     * sized for `FT_TT_OP_GDN_DST_BATCH` whatever the compute build asks for: the kernel
     * clamps its group to half of what is placed here, so a low build under a high placement
     * is merely slow, and a high build under a low placement degrades to the per-tile
     * sequence instead of hanging on a wait the buffer can never satisfy. The producer still
     * pushes ONE PAGE AT A TIME on every rung — CB_OUT's cadence is `state_tiles` state pages
     * then one y page and no page count makes that block-aligned — so this buffer gains a
     * consumer group and no producer grid. */
    out_pages[9] = gdn_max(GDN_CB_PAGES_SMALL,
                           GDN_CB_BLOCKS *
                               ftg_writer_block(state_tiles, FT_TT_OP_GDN_WRITER_RUNG));
    /* CB_CARRY. MULTI-ROW DELTA ONLY: one head's whole state, `state_tiles` tile-rows, from
     * one verify row to the next. A tile-row is popped (decayed) before its successor is
     * pushed, so `state_tiles^2` pages is enough, and `state_tiles` divides it. */
    out_pages[10] = multi_row ? state_tiles * state_tiles : GDN_CB_PAGES_SMALL;
}

int ft_tt_op_gdn_place_cbs(const ft_tt_op_gdn_request *request,
                          ft_tt_launch_cb_entry out_entries[FT_TT_OP_GDN_CB_COUNT],
                          void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    static const unsigned indices[FT_TT_OP_GDN_CB_COUNT] = {
        FT_TT_OP_GDN_CB_IN,    FT_TT_OP_GDN_CB_STATE, FT_TT_OP_GDN_CB_SCALER,
        FT_TT_OP_GDN_CB_VECTOR, FT_TT_OP_GDN_CB_STAGE, FT_TT_OP_GDN_CB_PRED,
        FT_TT_OP_GDN_CB_DELTA, FT_TT_OP_GDN_CB_PROD,  FT_TT_OP_GDN_CB_WORK,
        FT_TT_OP_GDN_CB_OUT,   FT_TT_OP_GDN_CB_CARRY};
    ft_tt_op_cb_request requests[FT_TT_OP_GDN_CB_COUNT];
    uint32_t pages[FT_TT_OP_GDN_CB_COUNT];
    unsigned index;

    if (!request || !out_entries || !buffer || !out_mask)
        return FT_TT_EINVAL;
    /* Same refusal as the encoder's: a placement sized from an unusable depth is the one
     * thing this file must never produce, because it is the placement the kernels then find
     * perfectly legal. */
    if (ft_tt_op_gdn_conv_depth() == 0u)
        return FT_TT_EINVAL;
    if (!ft_tt_op_gdn_shape_is_supported(request))
        return FT_TT_EINVAL;
    if (!operand_is_set(&request->activation_in) || !operand_is_set(&request->activation_out))
        return FT_TT_EINVAL;

    resolve_cb_pages(request->state_size / FT_TT_OP_TILE_DIM,
                     conv_taps_are_usable(request->conv_taps) ? request->conv_taps : 0u, pages);

    /* EVERY page is FT_TT_OP_GDN_TILE_BYTES and none of them is taken from an operand. The
     * CB table makes all nine buffers Float32 and both dataflow kernels are built with
     * TILE_BYTES = 4096; a page sized from a bf16 operand would put 2048-byte pages under a
     * reader that writes 4096-byte tiles, which overruns the buffer into whatever
     * `ft_tt_op_cb_place` put above it. `ft_tt_op_gdn_parameters()` refuses an operand that
     * is not this size, so nothing can reach here claiming otherwise. */
    for (index = 0; index < FT_TT_OP_GDN_CB_COUNT; index++) {
        requests[index].index = indices[index];
        requests[index].page_bytes = FT_TT_OP_GDN_FP32_TILE_BYTES;
        requests[index].page_count = pages[index];
    }

    return ft_tt_op_cb_place(requests, FT_TT_OP_GDN_CB_COUNT, out_entries, buffer, buffer_bytes,
                             out_mask);
}

/* ================================================================
 * What one layer costs in card DRAM
 * ================================================================
 *
 * EVERY SIZE BELOW IS `<an ftg_* TILE COUNT> * FT_TT_OP_GDN_FP32_TILE_BYTES`, and the tile
 * counts come from `resident_gdn_protocol.h` itself rather than from arithmetic respelled
 * here. That header is the settled layout and both dataflow kernels compute their offsets
 * from those same accessors, so a stager sized from them cannot disagree with the reader
 * about where a plane ends — which is the one disagreement nothing checks at run time.
 */

/** The device's own view of this request. `tile_count` is not a size input, so it is 0. */
static ftg_shape shape_of(const ft_tt_op_gdn_request *request)
{
    ftg_shape shape;

    shape.phase = request->phase;
    shape.state_tiles = request->state_size / FT_TT_OP_TILE_DIM;
    shape.v_heads = request->v_heads;
    shape.k_heads = request->k_heads;
    shape.conv_tile_cols = request->conv_dim / FT_TT_OP_TILE_DIM;
    shape.conv_taps = request->conv_taps;
    shape.tile_bytes = FT_TT_OP_GDN_FP32_TILE_BYTES;
    shape.tile_count = 0u;
    /* WHOLE, not this member's slice. Everything below sizes a LAYER's DRAM, which every
     * member of a head-split shares — the slice decides which of those bytes a member
     * touches, never how many exist. */
    shape.head_first = 0u;
    shape.head_count = request->v_heads;
    shape.row_first = 0u;
    shape.row_count = 1u;
    return shape;
}

/** Is this a request the sizes below mean anything for? Deliberately NOT the full phase
 *  predicate: a caller sizing a residency's buffers holds ONE request describing the layer,
 *  and asks it for both phases' operands. */
static bool shape_is_sizeable(const ft_tt_op_gdn_request *request)
{
    return request && request->state_size != 0u &&
           (request->state_size % FT_TT_OP_TILE_DIM) == 0u && request->v_heads != 0u &&
           request->k_heads != 0u && (request->v_heads % request->k_heads) == 0u;
}

/** As above, plus the CONV half: a conv operand needs a stream width and a tap count. */
static bool conv_is_sizeable(const ft_tt_op_gdn_request *request)
{
    return shape_is_sizeable(request) && request->conv_dim != 0u &&
           (request->conv_dim % FT_TT_OP_TILE_DIM) == 0u &&
           conv_taps_are_usable(request->conv_taps);
}

static size_t gdn_tile_bytes(uint32_t tiles)
{
    return (size_t)tiles * (size_t)FT_TT_OP_GDN_FP32_TILE_BYTES;
}

size_t ft_tt_op_gdn_state_bytes(const ft_tt_op_gdn_request *request)
{
    ftg_shape shape;

    if (!shape_is_sizeable(request))
        return 0u;
    shape = shape_of(request);
    return gdn_tile_bytes(shape.v_heads * ftg_state_tiles_per_head(&shape));
}

size_t ft_tt_op_gdn_conv_state_bytes(const ft_tt_op_gdn_request *request)
{
    ftg_shape shape;

    if (!conv_is_sizeable(request))
        return 0u;
    shape = shape_of(request);
    /* One past the LAST window plane's last tile — the same expression the reader's
     * `ftg_conv_plane_tile(taps-2, conv_tile_cols-1)` addresses, plus one. */
    return gdn_tile_bytes(ftg_conv_plane_tile(&shape, shape.conv_taps - 1u, 0u));
}

size_t ft_tt_op_gdn_conv_weight_bytes(const ft_tt_op_gdn_request *request)
{
    ftg_shape shape;

    if (!conv_is_sizeable(request))
        return 0u;
    shape = shape_of(request);
    return gdn_tile_bytes(ftg_conv_plane_tile(&shape, shape.conv_taps, 0u));
}

size_t ft_tt_op_gdn_conv_stream_bytes(const ft_tt_op_gdn_request *request)
{
    if (!conv_is_sizeable(request))
        return 0u;
    return gdn_tile_bytes(request->conv_dim / FT_TT_OP_TILE_DIM);
}

size_t ft_tt_op_gdn_delta_input_bytes(const ft_tt_op_gdn_request *request)
{
    ftg_shape shape;

    if (!shape_is_sizeable(request))
        return 0u;
    shape = shape_of(request);
    return gdn_tile_bytes(ftg_delta_input_tiles(&shape));
}

size_t ft_tt_op_gdn_delta_output_bytes(const ft_tt_op_gdn_request *request)
{
    ftg_shape shape;

    if (!shape_is_sizeable(request))
        return 0u;
    shape = shape_of(request);
    /* One past the LAST head's last tile-row: y is a ROW-0 STRIP, head-major, so every head
     * owns `state_tiles` tiles of its own and the buffer is `v_heads * state_tiles`. */
    return gdn_tile_bytes(ftg_y_tile(&shape, shape.v_heads - 1u, shape.state_tiles - 1u) + 1u);
}

size_t ft_tt_op_gdn_gate_bytes(const ft_tt_op_gdn_request *request)
{
    if (!shape_is_sizeable(request))
        return 0u;
    /* THE READER'S OWN EXPRESSION, spelled here because the host is the side that allocates:
     * resident_gdn_reader.c rounds its one gate read up to the 64-byte DRAM granule. */
    return (size_t)ftg_gate_stride(request->v_heads);
}
