/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_gdnc_ops.c
 * @brief Off-hardware acceptance for the CHUNKWISE GATED-DELTA-NET op family: the wire
 *   contract between `ft_tt_op_gdnc.h` and `kernels/resident_gdnc_protocol.h`, the host
 *   driver's encoding / refusals / CB placement, and — the part that matters —
 *   **A SOFTWARE MODEL OF THE KERNEL, RUN ACROSS FOUR CHUNK BOUNDARIES AGAINST THE ORACLE.**
 *
 * ---------------------------------------------------------------------------
 * THE ARM THAT IS WORTH SOMETHING IS STAGE 5, AND IT NEEDS NO CARD
 * ---------------------------------------------------------------------------
 * The whole chunkwise idea is one claim: that the depthwise conv1d's TAP SHIFT, which is a
 * shift along the POSITION axis, is a multiply by a constant 0/1 matrix, and that the
 * `taps-1` values a chunk owes the next one are the last `taps-1` ROWS of its input. That
 * claim is exactly checkable in C, in fp64, with no Tensix anywhere:
 *
 *   - build the band with the SHIPPING `ft_tt_op_gdnc_build_band()`, untilize it, and use it;
 *   - run the kernel's own expression `out = SiLU(sum_i band_i . prod_i)` over C rows;
 *   - roll the window the way the WRITER rolls it;
 *   - compare against a straight `pn_qwen35moe_conv1d_silu`-shaped scalar oracle stepped one
 *     position at a time over the SAME stream.
 *
 * **THE CHUNK SPLIT IS 32 + 32 + 32 + 7**, not one chunk, because a single-chunk test cannot
 * see a boundary bug: at position 0 the window is zero and the whole phase is
 * `SiLU(x * w[taps-1])`, which is the decode family's own recorded blind spot (its PHASE CONV
 * scored 6.9e-4 at layer 0 of token 0 and 1e-2..2e-1 after it, on a defect that was invisible
 * in the first descriptor). Four chunks and a ragged one is the smallest split that exercises
 * a full roll, a roll into a chunk shorter than 32, and three carries.
 *
 * ---------------------------------------------------------------------------
 * FOUR RED ARMS, AND EVERY ONE OF THEM IS RUN
 * ---------------------------------------------------------------------------
 * A gate that has never produced a LOW number is not a gate. `PN_TT_RED`-style mutations are
 * applied to the MODEL, not to the oracle, and each must break it:
 *
 *   band_swap      pair `S_tau` with the window and `C_tau` with the input — the plausible
 *                  transcription error, and the one a uniform loop invites
 *   tap_reverse    read the weight planes in the opposite order — the oracle's `w[c][k-1]`
 *                  multiplies the NEWEST sample, and getting that backwards is a convolution
 *                  that still looks like a convolution
 *   roll_first     roll the window from rows `0..taps-2` instead of the LAST `taps-1` — a
 *                  chunk-boundary bug and NOTHING ELSE, invisible inside any single chunk
 *   no_window      drop the window contribution entirely — the control that says the carry
 *                  is contributing at all, and the one that must be worst at position 0 of
 *                  every chunk after the first
 *
 * ---------------------------------------------------------------------------
 * WHY THE WIRE PIN PARSES THE HEADER INSTEAD OF INCLUDING IT
 * ---------------------------------------------------------------------------
 * The two halves are declared TWICE — once as a C enum the host compiles and once as the
 * `enum { name = value };` form the device kernels compile — and nothing in either build
 * makes them agree. Including the device header here would make the test compile against the
 * same declaration the kernel does, which is the right thing to want and the wrong thing to
 * test: the failure this suite exists to catch is one of the two being edited alone. A
 * constant that is MISSING is a failure rather than a default.
 *
 * The LAYOUT accessors (`ftgc_plane_tile`, `ftgc_band_base_tile`, `ftgc_roll_source_row`) are
 * declared ONCE and are included, not parsed: there is no second declaration for them to
 * drift from, and the drift that matters is between the accessor and the host stager.
 *
 * Usage:
 *   make -f Makefile.gdnc all && FT_TT_KERNEL_DIR=../kernels ./build-gdnc/test_gdnc_ops
 *
 * @status Off-hardware. Opens no device; runs on the Mac, which is where the code is written.
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_launch.h"
#include "ft_tt_mailbox.h"
#include "ft_tt_op_gdnc.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_queue.h"
#include "ft_tt_test_report.h"
#include "ft_tt_tilize.h"

/* The device's L1 pointer attribute belongs to the SFPI GCC fork. The protocol header guards
 * it so a HOST compiler can read the file. */
#define FTQ_L1_PTR
#include "../kernels/resident_gdnc_protocol.h"

#define KERNEL_DIRECTORY_ENVIRONMENT "FT_TT_KERNEL_DIR"
#define KERNEL_DIRECTORY_DEFAULT "../kernels"
#define PATH_MAX_BYTES 512u

static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG *
                                 4];

/* ================================================================
 * Reading a constant out of a device header
 * ================================================================ */

/** The value of `enum { <name> = <value> };` in `path`. A missing constant is a FAILURE. */
static int read_device_constant(const char *path, const char *name, uint32_t *out_value)
{
    char line[512];
    FILE *file = fopen(path, "r");
    int found = -1;

    if (!file)
        return -1;
    while (fgets(line, sizeof(line), file)) {
        char parsed_name[128];
        char parsed_value[64];

        if (sscanf(line, " enum { %127[A-Za-z0-9_] = %63[^}] };", parsed_name,
                   parsed_value) != 2)
            continue;
        if (strcmp(parsed_name, name) != 0)
            continue;
        *out_value = (uint32_t)strtoul(parsed_value, NULL, 0);
        found = 0;
        break;
    }
    fclose(file);
    return found;
}

static void device_header_path(char *out, size_t out_bytes, const char *file)
{
    const char *directory = getenv(KERNEL_DIRECTORY_ENVIRONMENT);

    if (!directory || directory[0] == '\0')
        directory = KERNEL_DIRECTORY_DEFAULT;
    snprintf(out, out_bytes, "%s/%s", directory, file);
}

static void check_device_constant(const char *file, const char *name, uint32_t expected)
{
    char path[PATH_MAX_BYTES];
    uint32_t value = 0xFFFFFFFFu;
    char label[192];
    int rc;

    device_header_path(path, sizeof(path), file);
    rc = read_device_constant(path, name, &value);
    snprintf(label, sizeof(label), "%s::%s is the host's value", file, name);
    ft_tt_test_check_fmt(label, rc == 0 && value == expected, "device %u, host %u (rc=%d)",
                         value, expected, rc);
}


/**
 * No `kOp*` in the SHARED registry already claims `opcode`.
 *
 * The registry lives in `kernels/resident_queue_protocol.h` and this family's opcode
 * deliberately does not (see the note beside the call site). So the collision is checked by
 * READING every opcode that header declares — which also makes the check say HOW MANY it
 * found, so a header that stopped declaring any would read as a failure rather than as a
 * vacuous pass.
 */
static void check_opcode_is_unclaimed(uint32_t opcode)
{
    char path[PATH_MAX_BYTES];
    char line[512];
    FILE *file;
    unsigned seen = 0;
    unsigned collisions = 0;
    char collided[128];

    collided[0] = '\0';
    device_header_path(path, sizeof(path), "resident_queue_protocol.h");
    file = fopen(path, "r");
    if (!file) {
        ft_tt_test_check_fmt("the shared opcode registry is readable", false, "%s", path);
        return;
    }
    while (fgets(line, sizeof(line), file)) {
        char name[128];
        char value[64];

        if (sscanf(line, " enum { %127[A-Za-z0-9_] = %63[^}] };", name, value) != 2)
            continue;
        if (strncmp(name, "kOp", 3) != 0)
            continue;
        seen++;
        if ((uint32_t)strtoul(value, NULL, 0) == opcode) {
            collisions++;
            snprintf(collided, sizeof(collided), "%s", name);
        }
    }
    fclose(file);
    ft_tt_test_check_fmt("the shared registry declares opcodes to compare against", seen >= 16u,
                         "%u found", seen);
    ft_tt_test_check_fmt("no family in the shared registry claims this opcode",
                         collisions == 0u, "0x%02x collides with %s (%u of %u)",
                         (unsigned)opcode, collided[0] ? collided : "nothing", collisions, seen);
}

/* ================================================================
 * Stage 1 — the wire contract
 * ================================================================ */

static void stage_wire_contract(void)
{
    ft_tt_test_step("stage 1 — the sixteen words, parsed out of the device header");

    check_device_constant("resident_gdnc_protocol.h", "kGdncShape",
                          (uint32_t)FT_TT_OP_GDNC_SHAPE);
    check_device_constant("resident_gdnc_protocol.h", "kGdncConvTileCols",
                          (uint32_t)FT_TT_OP_GDNC_CONV_TILE_COLS);
    check_device_constant("resident_gdnc_protocol.h", "kGdncColSlice",
                          (uint32_t)FT_TT_OP_GDNC_COL_SLICE);
    check_device_constant("resident_gdnc_protocol.h", "kGdncActivationXy",
                          (uint32_t)FT_TT_OP_GDNC_ACTIVATION_XY);
    check_device_constant("resident_gdnc_protocol.h", "kGdncActivationIn",
                          (uint32_t)FT_TT_OP_GDNC_ACTIVATION_IN);
    check_device_constant("resident_gdnc_protocol.h", "kGdncActivationOut",
                          (uint32_t)FT_TT_OP_GDNC_ACTIVATION_OUT);
    check_device_constant("resident_gdnc_protocol.h", "kGdncConstantXy",
                          (uint32_t)FT_TT_OP_GDNC_CONSTANT_XY);
    check_device_constant("resident_gdnc_protocol.h", "kGdncConstantWeight",
                          (uint32_t)FT_TT_OP_GDNC_CONSTANT_WEIGHT);
    check_device_constant("resident_gdnc_protocol.h", "kGdncStateXy",
                          (uint32_t)FT_TT_OP_GDNC_STATE_XY);
    check_device_constant("resident_gdnc_protocol.h", "kGdncStateConv",
                          (uint32_t)FT_TT_OP_GDNC_STATE_CONV);
    check_device_constant("resident_gdnc_protocol.h", "kGdncTileBytes",
                          (uint32_t)FT_TT_OP_GDNC_TILE_BYTES);
    check_device_constant("resident_gdnc_protocol.h", "kGdncFlags",
                          (uint32_t)FT_TT_OP_GDNC_FLAGS);
    check_device_constant("resident_gdnc_protocol.h", "kGdncTileCount",
                          (uint32_t)FT_TT_OP_GDNC_TILE_COUNT);
    check_device_constant("resident_gdnc_protocol.h", "kGdncFirstRow",
                          (uint32_t)FT_TT_OP_GDNC_FIRST_ROW);
    check_device_constant("resident_gdnc_protocol.h", "kGdncDeltaBand",
                          (uint32_t)FT_TT_OP_GDNC_DELTA_BAND);
    check_device_constant("resident_gdnc_protocol.h", "kGdncDeltaFlagCompact",
                          FT_TT_OP_GDNC_DELTA_FLAG_COMPACT);
    check_device_constant("resident_gdnc_protocol.h", "kGdncDeltaFlagTable",
                          FT_TT_OP_GDNC_DELTA_FLAG_TABLE);
    check_device_constant("resident_gdnc_protocol.h", "kGdncWords",
                          (uint32_t)FT_TT_OP_GDNC_WORDS);
    check_device_constant("resident_gdnc_protocol.h", "kGdncPhaseConv",
                          FT_TT_OP_GDNC_PHASE_CONV);
    check_device_constant("resident_gdnc_protocol.h", "kGdncPhaseDelta",
                          FT_TT_OP_GDNC_PHASE_DELTA);

    /* THE OPCODE, ACROSS THE TWO FILES THAT CARRY IT, AND AGAINST THE WHOLE REGISTRY.
     *
     * `kOpGdnChunk` is declared in THIS family's protocol header rather than in the shared
     * `resident_queue_protocol.h`, because that header is a CONTENT-PROVENANCE INPUT TO EVERY
     * EMBEDDED IMAGE: one added line makes all 23 programs stale, the blob emitter refuses,
     * and registering an opcode costs a 23-program rebuild of binaries whose byte-identity is
     * what protects the shipped decode gate. Measured on a second card host 2026-08-21.
     *
     * SO THE COLLISION CHECK IS DONE HERE INSTEAD, and it is STRONGER than the single pin it
     * replaces: every `kOp*` the shared registry declares is read out of it and asserted to
     * be a DIFFERENT number. Two families on one opcode is a descriptor executed by the wrong
     * kernel, and this catches it whichever side moved. */
    check_device_constant("resident_gdnc_protocol.h", "kOpGdnChunk",
                          (uint32_t)FT_TT_QUEUE_OP_GDN_CHUNK);
    ft_tt_test_check_fmt("kOpGdnChunk sits above kOpGdn",
                         FT_TT_QUEUE_OP_GDN_CHUNK > FT_TT_QUEUE_OP_GDN, "chunk 0x%02x, gdn 0x%02x",
                         (unsigned)FT_TT_QUEUE_OP_GDN_CHUNK, (unsigned)FT_TT_QUEUE_OP_GDN);
    check_opcode_is_unclaimed((uint32_t)FT_TT_QUEUE_OP_GDN_CHUNK);

    /* A SEVENTEENTH WORD DOES NOT OVERFLOW A BUFFER HERE, IT MOVES THE NEXT FAMILY'S SLOT ON
     * THE CARD. `ft_tt_queue_write_op_parameters()` writes exactly FT_TT_QUEUE_PARAM_WORDS
     * into a slot whose STRIDE is shared with every other family. */
    ft_tt_test_check_fmt("the parameter block is FT_TT_QUEUE_PARAM_WORDS wide",
                         (unsigned)FT_TT_OP_GDNC_WORDS == (unsigned)FT_TT_QUEUE_PARAM_WORDS,
                         "gdnc %u, queue %u", (unsigned)FT_TT_OP_GDNC_WORDS,
                         (unsigned)FT_TT_QUEUE_PARAM_WORDS);
}

/* ================================================================
 * A request the family accepts — Ornith's shape
 * ================================================================ */

static ft_tt_op_operand operand(uint16_t x, uint16_t y, uint32_t address)
{
    ft_tt_op_operand out;

    out.x = x;
    out.y = y;
    out.address = address;
    out.tile_bytes = FT_TT_OP_GDNC_FP32_TILE_BYTES;
    return out;
}

static ft_tt_op_gdnc_request ornith_request(void)
{
    ft_tt_op_gdnc_request request;

    memset(&request, 0, sizeof(request));
    request.phase = FT_TT_OP_GDNC_PHASE_CONV;
    request.chunk_rows = 32u;
    request.conv_dim = FT_TT_OP_GDNC_QWEN35_CONV_DIM;
    request.conv_taps = FT_TT_OP_GDNC_QWEN35_CONV_TAPS;
    request.first_col = 0u;
    request.col_count = 0u; /* the whole stream */
    /* THE PLACEMENT IS THE WHOLE FAMILY'S, so a CONV request still carries the DELTA shape:
     * a residency is armed ONCE and then answers whatever the queue hands it. The GDN state's
     * own dims -- `ssm_state` 128, so 4 tiles, over 32 v-heads and 16 k-heads -- and NOT the
     * 256 that appears elsewhere in this backend, which is the ATTENTION head dim. */
    request.state_tiles = FT_TT_OP_GDNC_QWEN35_STATE_TILES;
    request.v_heads = FT_TT_OP_GDNC_QWEN35_V_HEADS;
    request.k_heads = FT_TT_OP_GDNC_QWEN35_K_HEADS;
    request.activation_in = operand(1, 0, 0x10000000u);
    request.activation_out = operand(1, 0, 0x10200000u);
    request.constant = operand(2, 0, 0x20000000u);
    request.state_conv = operand(3, 0, 0x30000000u);
    request.gate_address = 0x20800000u;
    return request;
}

/** The same fixture, PHASE DELTA. */
static ft_tt_op_gdnc_request ornith_delta_request(void)
{
    ft_tt_op_gdnc_request request = ornith_request();

    request.phase = FT_TT_OP_GDNC_PHASE_DELTA;
    return request;
}

/* ================================================================
 * Stage 2 — the encoding round trip
 * ================================================================ */

static void stage_encoding(void)
{
    ft_tt_op_gdnc_request request = ornith_request();
    uint32_t words[FT_TT_OP_GDNC_WORDS];
    ftgc_shape shape;

    ft_tt_test_step("stage 2 — the encoding, read back with the DEVICE's own decoder");

    ft_tt_test_check("an Ornith CONV request encodes",
                     ft_tt_op_gdnc_parameters(&request, words) == FT_TT_OK, NULL);

    shape = ftgc_decode(words);
    ft_tt_test_check_fmt("phase", shape.phase == FT_TT_OP_GDNC_PHASE_CONV, "%u", shape.phase);
    ft_tt_test_check_fmt("chunk rows", shape.chunk_rows == 32u, "%u", shape.chunk_rows);
    ft_tt_test_check_fmt("taps", shape.conv_taps == 4u, "%u", shape.conv_taps);
    ft_tt_test_check_fmt("tile columns", shape.conv_tile_cols == 256u, "%u",
                         shape.conv_tile_cols);
    ft_tt_test_check_fmt("an unsliced descriptor decodes to the whole stream",
                         shape.first_col == 0u && shape.col_count == 256u, "%u..%u",
                         shape.first_col, shape.col_count);
    ft_tt_test_check_fmt("tile count is two a column", shape.tile_count == 512u, "%u",
                         shape.tile_count);
    ft_tt_test_check_fmt("tile bytes", shape.tile_bytes == FT_TT_OP_GDNC_FP32_TILE_BYTES, "%u",
                         shape.tile_bytes);
    ft_tt_test_check("the device accepts the shape it decoded",
                     ftgc_shape_is_supported(&shape), NULL);

    /* THE COLUMN SLICE IS A HALF-WORD BECAUSE 256 IS NOT A BYTE. A byte-wide count would
     * truncate 256 to 0, which decodes as "the whole stream" — a slice that silently becomes
     * no slice at all, on every member of a split. */
    request.first_col = 192u;
    request.col_count = 64u;
    ft_tt_test_check("a sliced request encodes",
                     ft_tt_op_gdnc_parameters(&request, words) == FT_TT_OK, NULL);
    shape = ftgc_decode(words);
    ft_tt_test_check_fmt("the slice survives the half-word pack",
                         shape.first_col == 192u && shape.col_count == 64u, "%u..%u",
                         shape.first_col, shape.col_count);
    ft_tt_test_check_fmt("a sliced descriptor's tile count is two a SLICED column",
                         shape.tile_count == 128u, "%u", shape.tile_count);
    ft_tt_test_check("the device accepts a sliced shape", ftgc_shape_is_supported(&shape),
                     NULL);

    /* THE BAND RIDES AT THE END OF THE WEIGHT PLANES, and this is the one place the host and
     * the reader could disagree about where. Both compute it with `ftgc_band_base_tile()`. */
    ft_tt_test_check_fmt("the band base is past every weight plane",
                         ftgc_band_base_tile(&shape) == 4u * 256u, "%u",
                         ftgc_band_base_tile(&shape));
    ft_tt_test_check_fmt("the constant buffer holds the planes AND the band",
                         ftgc_constant_tiles(&shape) == 4u * 256u + 8u, "%u",
                         ftgc_constant_tiles(&shape));
}

/* ================================================================
 * Stage 3 — the refusals
 * ================================================================ */

static void refuse(const char *what, ft_tt_op_gdnc_request request)
{
    uint32_t words[FT_TT_OP_GDNC_WORDS];

    ft_tt_test_check(what, ft_tt_op_gdnc_parameters(&request, words) != FT_TT_OK, NULL);
}

static void stage_refusals(void)
{
    ft_tt_op_gdnc_request request;

    ft_tt_test_step("stage 3 — what the host refuses");

    /* PHASE DELTA IS IMPLEMENTED NOW, so what has to be refused is a phase that is not a
     * phase — a byte that meant nothing today and something else tomorrow is how two families
     * end up executing each other's descriptors. */
    request = ornith_request();
    request.phase = 3u;
    refuse("a phase this family does not declare is refused by number", request);

    request = ornith_request();
    request.phase = 0u;
    refuse("a zeroed parameter block is not a plausible program", request);

    request = ornith_request();
    request.chunk_rows = request.conv_taps - 2u;
    refuse("a chunk shorter than the window cannot roll it", request);

    request = ornith_request();
    request.chunk_rows = 33u;
    refuse("a chunk longer than a tile row is a different op", request);

    request = ornith_request();
    request.chunk_rows = 0u;
    refuse("a zero-row chunk", request);

    request = ornith_request();
    request.conv_taps = 1u;
    refuse("a one-tap conv has no window to roll", request);

    request = ornith_request();
    request.conv_taps = 9u;
    refuse("more taps than the family will claim", request);

    request = ornith_request();
    request.conv_dim = 8190u;
    refuse("a conv_dim that is not a whole number of tile columns", request);

    request = ornith_request();
    request.first_col = 250u;
    request.col_count = 10u;
    refuse("a slice that runs off the end of the stream", request);

    request = ornith_request();
    request.activation_out = operand(9, 9, 0x40000000u);
    refuse("an activation pair in two different DRAM banks — ONE packed coordinate", request);

    request = ornith_request();
    request.state_conv.tile_bytes = 2048u;
    refuse("a bf16 operand under a program compiled for 4096-byte pages", request);

    request = ornith_request();
    request.constant.tile_bytes = 0u;
    refuse("an unset constant operand", request);

    /* ---------------------------------------------------------------------------
     * THE IN-PLACE SUB-CHUNK's window
     * --------------------------------------------------------------------------- */

    /* PHASE CONV's window ROLL takes rows `chunk_rows - (taps-1) + p` and knows nothing about
     * an offset, so a CONV descriptor naming one rolls the wrong rows — right in chunk 0 and
     * wrong from chunk 1 on, which is this family's established invisible-failure shape. */
    request = ornith_request();
    request.first_row = 16u;
    refuse("PHASE CONV does not take a first row — the roll would take the wrong rows",
           request);

    /* Off a FACE boundary the y write stops being a contiguous suffix of the tile. */
    request = ornith_delta_request();
    request.first_row = 8u;
    refuse("a first row off a tile face boundary", request);

    request = ornith_delta_request();
    request.first_row = 17u;
    refuse("...and one that is not even a multiple of the face height", request);

    /* Past the tile the write length goes negative and lands in the next head's y. */
    request = ornith_delta_request();
    request.first_row = 16u;
    request.chunk_rows = 32u;
    refuse("a sub-chunk that runs off the end of the strip", request);

    request = ornith_delta_request();
    request.first_row = 32u;
    request.chunk_rows = 1u;
    refuse("a first row at the end of the strip owns nothing", request);

    /* AND THE TWO THAT MUST BE ACCEPTED, or the refusals above are a function that refuses
     * every first row. `first_row = 16` at C = 16 is the SHIPPED rung. */
    {
        uint32_t words[FT_TT_OP_GDNC_WORDS];
        ftgc_shape decoded;

        request = ornith_delta_request();
        request.first_row = 16u;
        request.chunk_rows = 16u;
        ft_tt_test_check_fmt("the shipped rung — DELTA at first_row 16, C = 16 — is accepted",
                             ft_tt_op_gdnc_parameters(&request, words) == FT_TT_OK,
                             "word %d encodes %u", (int)FT_TT_OP_GDNC_FIRST_ROW,
                             words[FT_TT_OP_GDNC_FIRST_ROW]);
        /* THE DEVICE'S OWN DECODER AND ITS OWN REFUSAL, on the words the host emitted — the
         * two halves of this contract are checked against each other and not against a third
         * spelling. */
        decoded = ftgc_decode(words);
        ft_tt_test_check_fmt("the device decodes the first row it was sent",
                             decoded.first_row == 16u && ftgc_shape_is_supported(&decoded),
                             "%u, supported %d", decoded.first_row,
                             ftgc_shape_is_supported(&decoded) ? 1 : 0);
        /* THE WRITE IS THE SECOND HALF OF THE TILE, source and destination alike. */
        ft_tt_test_check_fmt("...and its y write starts at half a tile in",
                             ftgc_y_write_offset(&decoded) ==
                                 FT_TT_OP_GDNC_FP32_TILE_BYTES / 2u,
                             "%u of %u bytes", ftgc_y_write_offset(&decoded),
                             (unsigned)FT_TT_OP_GDNC_FP32_TILE_BYTES);

        request = ornith_delta_request();
        ft_tt_test_check_fmt("an unsliced DELTA descriptor still encodes a ZERO first row",
                             ft_tt_op_gdnc_parameters(&request, words) == FT_TT_OK &&
                                 words[FT_TT_OP_GDNC_FIRST_ROW] == 0u,
                             "word %d encodes %u", (int)FT_TT_OP_GDNC_FIRST_ROW,
                             words[FT_TT_OP_GDNC_FIRST_ROW]);
        /* AND ITS WRITE IS THE WHOLE TILE — the equality that makes the shipped descriptor
         * indistinguishable from the one that predates this field. */
        decoded = ftgc_decode(words);
        ft_tt_test_check_fmt("...and its y write is the WHOLE tile, as it always was",
                             ftgc_y_write_offset(&decoded) == 0u, "offset %u",
                             ftgc_y_write_offset(&decoded));
    }

    /* THE TABLE GATES: the band rides word 15 only under the flag, and only in DELTA. */
    request = ornith_delta_request();
    request.gate_band = 3u;
    refuse("a band without the table flag", request);
    request = ornith_request();
    request.delta_flags = FT_TT_OP_GDNC_DELTA_FLAG_TABLE;
    refuse("PHASE CONV does not take the table flag", request);
    request = ornith_request();
    request.gate_band = 1u;
    refuse("...nor a band", request);
    {
        uint32_t words[FT_TT_OP_GDNC_WORDS];

        request = ornith_delta_request();
        request.delta_flags = FT_TT_OP_GDNC_DELTA_FLAG_TABLE;
        request.gate_band = 1023u;
        request.first_row = 16u;
        request.chunk_rows = 16u;
        ft_tt_test_check_fmt("a table DELTA descriptor encodes its flag and band",
                             ft_tt_op_gdnc_parameters(&request, words) == FT_TT_OK &&
                                 (words[FT_TT_OP_GDNC_FLAGS] & FT_TT_OP_GDNC_DELTA_FLAG_TABLE) &&
                                 words[FT_TT_OP_GDNC_DELTA_BAND] == 1023u,
                             "flags 0x%x band %u", words[FT_TT_OP_GDNC_FLAGS],
                             words[FT_TT_OP_GDNC_DELTA_BAND]);
        request = ornith_delta_request();
        ft_tt_test_check_fmt("a shipped DELTA descriptor still encodes word 15 zero",
                             ft_tt_op_gdnc_parameters(&request, words) == FT_TT_OK &&
                                 words[FT_TT_OP_GDNC_DELTA_BAND] == 0u,
                             "word 15 = %u", words[FT_TT_OP_GDNC_DELTA_BAND]);
    }

    /* AND THE GREEN CONTROL, so the refusals above are not a function that always refuses. */
    request = ornith_request();
    {
        uint32_t words[FT_TT_OP_GDNC_WORDS];

        ft_tt_test_check("...and the unmutated request is still accepted",
                         ft_tt_op_gdnc_parameters(&request, words) == FT_TT_OK, NULL);
    }
}

/* ================================================================
 * Stage 4 — CB placement, against the grids the READER refuses on
 * ================================================================ */

/** The consumer's indexed-read grid, spelled exactly as the reader spells it. */
static bool index_grid_is_safe(uint32_t pages, uint32_t chunk)
{
    return chunk != 0u && pages >= chunk && (pages % chunk) == 0u;
}

static uint32_t placed_pages(const ft_tt_launch_cb_entry *entries, unsigned count,
                             unsigned index)
{
    const ft_tt_launch_cb_entry *entry = ft_tt_op_cb_find(entries, count, index);

    return entry ? entry->page_count : 0u;
}

static void stage_placement(void)
{
    ft_tt_op_gdnc_request request = ornith_request();
    ft_tt_launch_cb_entry entries[FT_TT_OP_GDNC_CB_COUNT];
    uint64_t mask = 0u;
    uint32_t taps;
    uint32_t band;
    uint32_t pages[FT_TT_OP_GDNC_CB_COUNT];
    unsigned index;

    ft_tt_test_step("stage 4 — CB placement, against the grids the reader refuses on");

    ft_tt_test_check("the Ornith shape places",
                     ft_tt_op_gdnc_place_cbs(&request, entries, g_cb_buffer,
                                             sizeof(g_cb_buffer), &mask) == FT_TT_OK,
                     NULL);
    taps = request.conv_taps;
    band = 2u * taps;

    pages[0] = placed_pages(entries, FT_TT_OP_GDNC_CB_COUNT, FT_TT_OP_GDNC_CB_IN);
    pages[1] = placed_pages(entries, FT_TT_OP_GDNC_CB_COUNT, FT_TT_OP_GDNC_CB_STATE);
    pages[2] = placed_pages(entries, FT_TT_OP_GDNC_CB_COUNT, FT_TT_OP_GDNC_CB_BAND);
    pages[3] = placed_pages(entries, FT_TT_OP_GDNC_CB_COUNT, FT_TT_OP_GDNC_CB_VECTOR);
    pages[4] = placed_pages(entries, FT_TT_OP_GDNC_CB_COUNT, FT_TT_OP_GDNC_CB_PROD);
    /* ALL TEN, and the count is the point: this array is iterated over
     * `FT_TT_OP_GDNC_CB_COUNT` below, so a list that stops at the six PHASE CONV buffers
     * reads uninitialised stack for the other four. It passed on the Mac's clang, where that
     * stack happened to be non-zero, and failed on a second card host's GCC, where it was not. */
    pages[5] = placed_pages(entries, FT_TT_OP_GDNC_CB_COUNT, FT_TT_OP_GDNC_CB_WORK);
    pages[6] = placed_pages(entries, FT_TT_OP_GDNC_CB_COUNT, FT_TT_OP_GDNC_CB_SCRATCH);
    pages[7] = placed_pages(entries, FT_TT_OP_GDNC_CB_COUNT, FT_TT_OP_GDNC_CB_DECAY);
    pages[8] = placed_pages(entries, FT_TT_OP_GDNC_CB_COUNT, FT_TT_OP_GDNC_CB_CC);
    pages[9] = placed_pages(entries, FT_TT_OP_GDNC_CB_COUNT, FT_TT_OP_GDNC_CB_OUT);

    for (index = 0; index < FT_TT_OP_GDNC_CB_COUNT; index++)
        ft_tt_test_check_fmt("every buffer is placed with pages", pages[index] != 0u,
                             "buffer %u has %u pages", index, pages[index]);

    ft_tt_test_check_fmt("CB_IN holds more than one column", pages[0] >= 2u, "%u", pages[0]);
    ft_tt_test_check_fmt("CB_STATE holds more than one column", pages[1] >= 2u, "%u", pages[1]);
    /* CB_BAND IS EXACTLY THE BAND, which makes its push an EXACT HIT on the limit and
     * therefore self-rebasing — a FIFO pointer wraps on an exact hit and on nothing else. */
    ft_tt_test_check_fmt("CB_BAND is exactly the band", pages[2] == band, "%u against %u",
                         pages[2], band);
    ft_tt_test_check_fmt("CB_VECTOR is at least two columns of taps", pages[3] >= 2u * taps,
                         "%u", pages[3]);
    ft_tt_test_check_fmt("CB_PROD holds two whole product blocks", pages[4] >= 2u * band, "%u",
                         pages[4]);
    ft_tt_test_check_fmt("CB_OUT holds more than one column group", pages[9] >= 4u, "%u",
                         pages[9]);

    ft_tt_test_check_fmt("CB_BAND closes on the consumer's indexed read",
                         index_grid_is_safe(pages[2], band), "%u pages, %u wide", pages[2],
                         band);
    ft_tt_test_check_fmt("CB_VECTOR closes on the consumer's indexed read",
                         index_grid_is_safe(pages[3], taps), "%u pages, %u wide", pages[3],
                         taps);
    ft_tt_test_check_fmt("CB_PROD closes on the consumer's indexed read",
                         index_grid_is_safe(pages[4], band), "%u pages, %u wide", pages[4],
                         band);

    /* EVERY PAGE IS 4096 AND NONE OF THEM COMES FROM AN OPERAND. The decode family sized five
     * of nine buffers from `activation_in.tile_bytes` against kernels compiled for 4096, and
     * the device's one-word tile-size check could not catch it because it was reading a
     * different operand. */
    for (index = 0; index < FT_TT_OP_GDNC_CB_COUNT; index++) {
        const ft_tt_launch_cb_entry *entry = &entries[index];

        ft_tt_test_check_fmt("every page is an fp32 tile",
                             entry->page_size_bytes == FT_TT_OP_GDNC_FP32_TILE_BYTES, "%u",
                             entry->page_size_bytes);
    }

    /* Eight taps is the widest shape the family claims; it must still fit the arena. */
    request.conv_taps = 8u;
    request.chunk_rows = 32u;
    ft_tt_test_check("the widest tap count this family claims still places",
                     ft_tt_op_gdnc_place_cbs(&request, entries, g_cb_buffer,
                                             sizeof(g_cb_buffer), &mask) == FT_TT_OK,
                     NULL);
}

/* ================================================================
 * Stage 5 — THE ARITHMETIC. The kernel, modelled, across four chunk boundaries.
 * ================================================================ */

#define MODEL_TAPS 4u
#define MODEL_CHANNELS 64u /* two tile columns; the model runs one at a time */
#define MODEL_POSITIONS 103u
#define TILE_DIM ((uint32_t)FT_TT_OP_TILE_DIM)

/** Which of the model's mutations is live. `kRedNone` is the shipped arithmetic. */
typedef enum model_red {
    kRedNone = 0,
    kRedBandSwap,
    kRedTapReverse,
    kRedRollFirst,
    kRedNoWindow,
    kRedCount
} model_red;

static const char *const g_red_name[kRedCount] = { "none", "band_swap", "tap_reverse",
                                                   "roll_first", "no_window" };

static double model_silu(double x)
{
    return x / (1.0 + exp(-x));
}

/**
 * THE ORACLE: `pn_qwen35moe_conv1d_silu()`, one position at a time, transcribed.
 *
 * `acc = qkv[c]*w[c][k-1] + sum_{t<k-1} cs[t]*w[c][t]`, then the window rolls and the newest
 * tap becomes the RAW `qkv[c]` — not the SiLU'd output. Getting that last part backwards is a
 * convolution that still looks like one, which is why it is transcribed rather than
 * paraphrased.
 */
static void oracle_step(const double *weight, const double *x, double *window, double *out,
                        uint32_t channels, uint32_t taps)
{
    uint32_t channel;

    for (channel = 0; channel < channels; channel++) {
        const double *w = weight + (size_t)channel * taps;
        double *cs = window + (size_t)channel * (taps - 1u);
        double acc = x[channel] * w[taps - 1u];
        uint32_t tap;

        for (tap = 0; tap + 1u < taps; tap++)
            acc += cs[tap] * w[tap];
        out[channel] = model_silu(acc);
        for (tap = 0; tap + 2u < taps; tap++)
            cs[tap] = cs[tap + 1u];
        cs[taps - 2u] = x[channel];
    }
}

/**
 * THE KERNEL, MODELLED: one CONV descriptor over one tile column of `chunk_rows` positions.
 *
 * `band` is the untilized `2*taps` matrices in the order the SHIPPING builder wrote them.
 * `input` is a `[32, 32]` tile whose rows `chunk_rows..31` are zero, exactly as the host owes.
 * `window_tile` is the reader's assembly: row `p` is window plane `p`'s row 0. `out` receives
 * `[32, 32]`, and `window_tile` is rolled the way the WRITER rolls it.
 */
static void model_descriptor(const double *band, const double *weight_replicated,
                             const double *input, double *window_tile, double *out,
                             uint32_t chunk_rows, uint32_t taps, model_red red)
{
    double product[2u * 8u][TILE_DIM * TILE_DIM];
    uint32_t tap;
    uint32_t row;
    uint32_t column;

    /* The `2*taps` weighted products, in BAND ORDER: `2*tau` against the input, `2*tau+1`
     * against the window. */
    for (tap = 0; tap < taps; tap++) {
        const uint32_t weight_tap = (red == kRedTapReverse) ? (taps - 1u - tap) : tap;

        for (row = 0; row < TILE_DIM; row++) {
            for (column = 0; column < TILE_DIM; column++) {
                const double w = weight_replicated[(size_t)weight_tap * TILE_DIM * TILE_DIM +
                                                   (size_t)row * TILE_DIM + column];
                const size_t at = (size_t)row * TILE_DIM + column;

                product[2u * tap][at] = input[at] * w;
                product[2u * tap + 1u][at] =
                    (red == kRedNoWindow) ? 0.0 : window_tile[at] * w;
            }
        }
    }

    /* `out = SiLU( sum_i band_i . product_i )`, one DST section on the card. */
    for (row = 0; row < TILE_DIM; row++) {
        for (column = 0; column < TILE_DIM; column++) {
            double accumulator = 0.0;
            uint32_t term;

            for (term = 0; term < 2u * taps; term++) {
                /* THE RED ARM THAT A UNIFORM LOOP INVITES: pair `S_tau` with the window and
                 * `C_tau` with the input. */
                const uint32_t band_index =
                    (red == kRedBandSwap) ? (term ^ 1u) : term;
                uint32_t inner;

                for (inner = 0; inner < TILE_DIM; inner++)
                    accumulator += band[(size_t)band_index * TILE_DIM * TILE_DIM +
                                        (size_t)row * TILE_DIM + inner] *
                                   product[term][(size_t)inner * TILE_DIM + column];
            }
            out[(size_t)row * TILE_DIM + column] = model_silu(accumulator);
        }
    }

    /* THE ROLL, as the writer does it: window plane `p` receives ROW
     * `chunk_rows - (taps-1) + p` of the chunk's RAW input. */
    {
        double rolled[TILE_DIM * TILE_DIM];
        uint32_t plane;

        memset(rolled, 0, sizeof(rolled));
        for (plane = 0; plane + 1u < taps; plane++) {
            /* THE CHUNK-BOUNDARY RED ARM, and it is invisible inside any one chunk. */
            const uint32_t source = (red == kRedRollFirst)
                                        ? plane
                                        : chunk_rows - (taps - 1u) + plane;

            for (column = 0; column < TILE_DIM; column++)
                rolled[(size_t)plane * TILE_DIM + column] =
                    input[(size_t)source * TILE_DIM + column];
        }
        memcpy(window_tile, rolled, sizeof(rolled));
    }
}

/** Untilize the SHIPPING band builder's output into `2*taps` row-major 32x32 matrices. */
static bool model_load_band(uint32_t taps, double *out)
{
    unsigned char *tiles = (unsigned char *)malloc(ft_tt_op_gdnc_band_bytes(taps));
    float row_major[TILE_DIM * TILE_DIM];
    uint32_t tile;
    bool ok = true;

    if (!tiles)
        return false;
    if (ft_tt_op_gdnc_build_band(taps, tiles, ft_tt_op_gdnc_band_bytes(taps)) != FT_TT_OK) {
        free(tiles);
        return false;
    }
    for (tile = 0; tile < 2u * taps && ok; tile++) {
        uint32_t index;

        ok = ft_tt_untilize_strip(tiles + (size_t)tile * FT_TT_OP_GDNC_FP32_TILE_BYTES,
                                  sizeof(float), TILE_DIM, TILE_DIM,
                                  (size_t)TILE_DIM * sizeof(float),
                                  row_major) == FT_TT_TILIZE_OK;
        for (index = 0; index < TILE_DIM * TILE_DIM; index++)
            out[(size_t)tile * TILE_DIM * TILE_DIM + index] = (double)row_major[index];
    }
    free(tiles);
    return ok;
}

/** A deterministic stream with no special structure, so the compare is not measuring zeros. */
static double model_sample(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return ((double)(*state >> 8) / 8388608.0) - 1.0;
}

/**
 * Run `MODEL_POSITIONS` positions through both, in chunks of 32, and return the worst
 * absolute difference over EVERY position and channel.
 *
 * @param out_first_bad_chunk receives the first chunk index that disagreed, or -1
 */
static double model_run(model_red red, int *out_first_bad_chunk)
{
    static double band[2u * 8u * TILE_DIM * TILE_DIM];
    static double weight_replicated[8u * TILE_DIM * TILE_DIM];
    static double weight[MODEL_CHANNELS * MODEL_TAPS];
    static double stream[MODEL_POSITIONS][MODEL_CHANNELS];
    static double expected[MODEL_POSITIONS][MODEL_CHANNELS];
    static double oracle_window[MODEL_CHANNELS * (MODEL_TAPS - 1u)];
    /* One window tile per tile column, carried across chunks — the model's whole point. */
    static double window_tile[2][TILE_DIM * TILE_DIM];
    static double input[TILE_DIM * TILE_DIM];
    static double output[TILE_DIM * TILE_DIM];
    uint32_t rng = 0x5eed1234u;
    uint32_t position;
    uint32_t channel;
    uint32_t tap;
    uint32_t done;
    int chunk_index = 0;
    double worst = 0.0;

    *out_first_bad_chunk = -1;
    if (!model_load_band(MODEL_TAPS, band))
        return 1.0e30;

    for (channel = 0; channel < MODEL_CHANNELS; channel++)
        for (tap = 0; tap < MODEL_TAPS; tap++)
            weight[(size_t)channel * MODEL_TAPS + tap] = model_sample(&rng);
    for (position = 0; position < MODEL_POSITIONS; position++)
        for (channel = 0; channel < MODEL_CHANNELS; channel++)
            stream[position][channel] = model_sample(&rng);

    /* THE ROW-REPLICATED, TAP-MAJOR WEIGHT: plane `tap`, tile column `c`, every row carrying
     * `w[32*c + column][tap]`. Two tile columns at 64 channels. */
    for (tap = 0; tap < MODEL_TAPS; tap++)
        for (channel = 0; channel < TILE_DIM; channel++)
            for (position = 0; position < TILE_DIM; position++)
                weight_replicated[(size_t)tap * TILE_DIM * TILE_DIM +
                                  (size_t)position * TILE_DIM + channel] = 0.0;

    memset(oracle_window, 0, sizeof(oracle_window));
    memset(window_tile, 0, sizeof(window_tile));

    for (position = 0; position < MODEL_POSITIONS; position++)
        oracle_step(weight, stream[position], oracle_window, expected[position],
                    MODEL_CHANNELS, MODEL_TAPS);

    done = 0;
    while (done < MODEL_POSITIONS) {
        uint32_t rows = MODEL_POSITIONS - done;
        uint32_t column_block;

        if (rows > TILE_DIM)
            rows = TILE_DIM;
        /* A tail shorter than the window goes through the DECODE family, not this one — the
         * protocol refuses it, so the model must not manufacture one either. */
        if (rows < MODEL_TAPS - 1u)
            break;

        for (column_block = 0; column_block < MODEL_CHANNELS / TILE_DIM; column_block++) {
            uint32_t row;

            /* This tile column's row-replicated weight planes. */
            for (tap = 0; tap < MODEL_TAPS; tap++)
                for (row = 0; row < TILE_DIM; row++)
                    for (channel = 0; channel < TILE_DIM; channel++)
                        weight_replicated[(size_t)tap * TILE_DIM * TILE_DIM +
                                          (size_t)row * TILE_DIM + channel] =
                            weight[(size_t)(column_block * TILE_DIM + channel) * MODEL_TAPS +
                                   tap];

            /* THE HOST OWES ZEROS ABOVE THE CHUNK. */
            memset(input, 0, sizeof(input));
            for (row = 0; row < rows; row++)
                for (channel = 0; channel < TILE_DIM; channel++)
                    input[(size_t)row * TILE_DIM + channel] =
                        stream[done + row][column_block * TILE_DIM + channel];

            model_descriptor(band, weight_replicated, input, window_tile[column_block], output,
                             rows, MODEL_TAPS, red);

            for (row = 0; row < rows; row++) {
                for (channel = 0; channel < TILE_DIM; channel++) {
                    const double got = output[(size_t)row * TILE_DIM + channel];
                    const double want = expected[done + row][column_block * TILE_DIM + channel];
                    const double difference = fabs(got - want);

                    if (difference > worst)
                        worst = difference;
                    if (difference > 1.0e-9 && *out_first_bad_chunk < 0)
                        *out_first_bad_chunk = chunk_index;
                }
            }
        }
        done += rows;
        chunk_index++;
    }
    return worst;
}

static void stage_arithmetic(void)
{
    int first_bad = -1;
    double worst;
    uint32_t red;

    ft_tt_test_step("stage 5 — the kernel modelled against the oracle, 103 positions in "
                    "chunks of 32 + 32 + 32 + 7");

    worst = model_run(kRedNone, &first_bad);
    /* The model is fp64 end to end, so the only difference it can show is a STRUCTURAL one.
     * The Tf32 residue the card will add is a separate question and is measured on silicon. */
    ft_tt_test_check_fmt("the banded matmul reproduces the oracle EXACTLY, across three "
                         "chunk boundaries and a ragged chunk",
                         worst < 1.0e-12 && first_bad < 0, "worst %.3e, first bad chunk %d",
                         worst, first_bad);

    for (red = kRedBandSwap; red < kRedCount; red++) {
        int bad = -1;
        double red_worst = model_run((model_red)red, &bad);
        char label[192];

        snprintf(label, sizeof(label), "RED %s breaks it", g_red_name[red]);
        ft_tt_test_check_fmt(label, red_worst > 1.0e-6, "worst %.3e, first bad chunk %d",
                             red_worst, bad);
    }

    /* THE ROLL RED ARM MUST BE INVISIBLE IN THE FIRST CHUNK AND VISIBLE AFTER IT. That is the
     * quantitative form of "a single-chunk test cannot see a boundary bug": at position 0 the
     * window is zero and the whole phase is SiLU(x * w[taps-1]), so a wrong roll destination
     * cannot show up until the SECOND chunk reads what the first one wrote. */
    {
        int bad = -1;

        (void)model_run(kRedRollFirst, &bad);
        ft_tt_test_check_fmt("RED roll_first is INVISIBLE in chunk 0 and appears at chunk 1 "
                             "— which is why one chunk is not a test",
                             bad == 1, "first bad chunk %d", bad);
    }
}

/* ================================================================
 * Stage 6 — the band, element for element
 * ================================================================ */

static void stage_band(void)
{
    static double band[2u * 8u * TILE_DIM * TILE_DIM];
    uint32_t taps = MODEL_TAPS;
    uint32_t tap;
    uint32_t row;
    uint32_t column;
    unsigned wrong = 0;
    unsigned ones = 0;

    ft_tt_test_step("stage 6 — the band tiles, element for element against the derivation");

    ft_tt_test_check("the band builds and untilizes", model_load_band(taps, band), NULL);

    for (tap = 0; tap < taps; tap++) {
        const uint32_t shift = (taps - 1u) - tap;

        for (row = 0; row < TILE_DIM; row++) {
            for (column = 0; column < TILE_DIM; column++) {
                const size_t at = (size_t)row * TILE_DIM + column;
                const double s = band[(size_t)(2u * tap) * TILE_DIM * TILE_DIM + at];
                const double c = band[(size_t)(2u * tap + 1u) * TILE_DIM * TILE_DIM + at];
                const double want_s =
                    (row >= shift && column == row - shift) ? 1.0 : 0.0;
                const double want_c = (row < shift && column == row + tap) ? 1.0 : 0.0;

                if (s != want_s || c != want_c)
                    wrong++;
                ones += (unsigned)((s != 0.0) ? 1 : 0) + (unsigned)((c != 0.0) ? 1 : 0);
            }
        }
    }
    ft_tt_test_check_fmt("every element of every band tile is the derivation's value",
                         wrong == 0u, "%u wrong", wrong);
    /* THE NON-ZERO COUNT IS A SECOND, INDEPENDENT INSTRUMENT. Every output row takes exactly
     * ONE input from every tap — from the chunk if it is far enough in, from the window
     * otherwise — so the two matrices of a tap partition the 32 rows and the total is
     * `taps * 32`. A band that was all zeros would pass the element check above against a
     * derivation that was also all zeros; it cannot pass this one. */
    ft_tt_test_check_fmt("and the band carries exactly taps*32 ones",
                         ones == taps * TILE_DIM, "%u against %u", ones, taps * TILE_DIM);

    /* `C_{taps-1}` IS THE ZERO TILE and is emitted anyway, so that the product count is
     * `2*taps` — a clean indexed-read grid at every tap count, where `2*taps-1` is prime at
     * taps = 4. */
    {
        unsigned nonzero = 0;

        for (row = 0; row < TILE_DIM * TILE_DIM; row++)
            nonzero +=
                band[(size_t)(2u * (taps - 1u) + 1u) * TILE_DIM * TILE_DIM + row] != 0.0;
        ft_tt_test_check_fmt("the newest tap's window matrix is the ZERO tile", nonzero == 0u,
                             "%u non-zero", nonzero);
    }

    ft_tt_test_check_fmt("the band refuses a short buffer",
                         ft_tt_op_gdnc_build_band(taps, band,
                                                  ft_tt_op_gdnc_band_bytes(taps) - 1u) !=
                             FT_TT_OK,
                         "%zu", ft_tt_op_gdnc_band_bytes(taps));
    ft_tt_test_check("the band refuses a tap count the family will not claim",
                     ft_tt_op_gdnc_build_band(9u, band, sizeof(band)) != FT_TT_OK, NULL);
}

/* ================================================================
 * Stage 7 — the DRAM sizes a stager allocates from
 * ================================================================ */

static void stage_sizes(void)
{
    ft_tt_op_gdnc_request request = ornith_request();
    const size_t tile = FT_TT_OP_GDNC_FP32_TILE_BYTES;

    ft_tt_test_step("stage 7 — what one layer costs in card DRAM");

    ft_tt_test_check_fmt("one conv stream is conv_dim/32 fp32 tiles",
                         ft_tt_op_gdnc_conv_stream_bytes(&request) == 256u * tile, "%zu",
                         ft_tt_op_gdnc_conv_stream_bytes(&request));
    ft_tt_test_check_fmt("the constant buffer is taps planes PLUS the band",
                         ft_tt_op_gdnc_constant_bytes(&request) == (4u * 256u + 8u) * tile,
                         "%zu", ft_tt_op_gdnc_constant_bytes(&request));
    /* THE WINDOW IS `taps-1` PLANES OF TILES, NOT `conv_dim*(taps-1)` ELEMENTS. That
     * expression is 1/32 of this one, and the decode family shipped it for a round: a plane is
     * a ONE-ROW tilized strip and the tile quantum is 32 rows, so a buffer sized from the
     * element count is a NoC read landing in whatever the allocator put above it. */
    ft_tt_test_check_fmt("the window is (taps-1) tap-major PLANES of tiles",
                         ft_tt_op_gdnc_conv_state_bytes(&request) == 3u * 256u * tile, "%zu",
                         ft_tt_op_gdnc_conv_state_bytes(&request));
    ft_tt_test_check_fmt("and NOT the element count, which is 32x smaller",
                         ft_tt_op_gdnc_conv_state_bytes(&request) !=
                             (size_t)FT_TT_OP_GDNC_QWEN35_CONV_DIM * 3u * 4u,
                         "%zu against %zu", ft_tt_op_gdnc_conv_state_bytes(&request),
                         (size_t)FT_TT_OP_GDNC_QWEN35_CONV_DIM * 3u * 4u);
    ft_tt_test_check_fmt("the band block is 2*taps tiles",
                         ft_tt_op_gdnc_band_bytes(4u) == 8u * tile, "%zu",
                         ft_tt_op_gdnc_band_bytes(4u));

    /* THE WINDOW IS BYTE FOR BYTE THE DECODE FAMILY'S, which is what makes a prompt that
     * prefills in chunks and then decodes one token at a time ONE sequence. */
    ft_tt_test_check_fmt("an unsupported request sizes to ZERO rather than to something wrong",
                         (request.phase = 0u,
                          ft_tt_op_gdnc_conv_state_bytes(&request) == 0u),
                         "%zu", ft_tt_op_gdnc_conv_state_bytes(&request));
}

/* ================================================================
 * Stage 9 — PHASE DELTA's host arithmetic: encoding, refusals, placement, DRAM sizes
 * ================================================================ */

static void stage_delta_host(void)
{
    ft_tt_op_gdnc_request request = ornith_delta_request();
    const size_t tile = FT_TT_OP_GDNC_FP32_TILE_BYTES;
    uint32_t words[FT_TT_OP_GDNC_WORDS];
    uint32_t conv_words[FT_TT_OP_GDNC_WORDS];
    ft_tt_op_gdnc_request conv = ornith_request();
    ft_tt_launch_cb_entry entries[FT_TT_OP_GDNC_CB_COUNT];
    unsigned char arena[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG *
                        sizeof(uint32_t)];
    uint64_t mask = 0;

    ft_tt_test_step("stage 9 — PHASE DELTA's host arithmetic");

    ft_tt_test_check("the DELTA shape is accepted",
                     ft_tt_op_gdnc_shape_is_supported(&request) == 1, NULL);
    ft_tt_test_check("and encodes",
                     ft_tt_op_gdnc_parameters(&request, words) == FT_TT_OK, NULL);

    ft_tt_test_check_fmt("the shape word carries phase, rows, taps AND state tiles",
                         words[FT_TT_OP_GDNC_SHAPE] ==
                             (FT_TT_OP_GDNC_PHASE_DELTA | (32u << 8) | (4u << 16) |
                              (FT_TT_OP_GDNC_QWEN35_STATE_TILES << 24)),
                         "0x%08x", words[FT_TT_OP_GDNC_SHAPE]);
    /* ONE WORD, TWO READINGS. The head counts ride where PHASE CONV puts its tile-column
     * count, because FT_TT_QUEUE_PARAM_WORDS is 16 for EVERY family. */
    ft_tt_test_check_fmt("word 1 carries v_heads | k_heads<<16 in DELTA",
                         words[FT_TT_OP_GDNC_CONV_TILE_COLS] ==
                             (FT_TT_OP_GDNC_QWEN35_V_HEADS |
                              (FT_TT_OP_GDNC_QWEN35_K_HEADS << 16)),
                         "0x%08x", words[FT_TT_OP_GDNC_CONV_TILE_COLS]);
    ft_tt_test_check_fmt("the tile count is state_tiles*(state_tiles+1) a head",
                         words[FT_TT_OP_GDNC_TILE_COUNT] == 32u * 4u * 5u, "%u",
                         words[FT_TT_OP_GDNC_TILE_COUNT]);
    ft_tt_test_check_fmt("the gate address is carried", 
                         words[FT_TT_OP_GDNC_DELTA_GATE] == request.gate_address, "0x%08x",
                         words[FT_TT_OP_GDNC_DELTA_GATE]);

    /* THE CONV DESCRIPTOR MUST NOT HAVE MOVED. PHASE DELTA added meaning to two reserved
     * words and to one byte of the shape word; if it had moved anything else, the arm that is
     * already green on silicon would be a different arm. */
    ft_tt_test_check("the CONV descriptor still encodes",
                     ft_tt_op_gdnc_parameters(&conv, conv_words) == FT_TT_OK, NULL);
    ft_tt_test_check_fmt("and its shape word carries a ZERO state-tiles byte in CONV... no: the "
                         "placement needs it, so CONV carries it too and DELTA reads it",
                         (conv_words[FT_TT_OP_GDNC_SHAPE] & 0x00FFFFFFu) ==
                             (FT_TT_OP_GDNC_PHASE_CONV | (32u << 8) | (4u << 16)),
                         "0x%08x", conv_words[FT_TT_OP_GDNC_SHAPE]);
    ft_tt_test_check_fmt("the CONV tile-column word is untouched",
                         conv_words[FT_TT_OP_GDNC_CONV_TILE_COLS] == 256u, "%u",
                         conv_words[FT_TT_OP_GDNC_CONV_TILE_COLS]);
    ft_tt_test_check_fmt("and the CONV tile count is still two a column",
                         conv_words[FT_TT_OP_GDNC_TILE_COUNT] == 512u, "%u",
                         conv_words[FT_TT_OP_GDNC_TILE_COUNT]);

    /* Refusals. */
    request = ornith_delta_request();
    request.k_heads = 15u; /* does not divide 32 */
    ft_tt_test_check("a GQA repeat that does not divide is refused",
                     ft_tt_op_gdnc_shape_is_supported(&request) == 0, NULL);
    request = ornith_delta_request();
    request.state_tiles = 9u;
    ft_tt_test_check("a state wider than L1 can hold is refused",
                     ft_tt_op_gdnc_shape_is_supported(&request) == 0, NULL);
    request = ornith_delta_request();
    request.gate_address = 0u;
    ft_tt_test_check("a DELTA descriptor with no gate block is refused",
                     ft_tt_op_gdnc_parameters(&request, words) == FT_TT_EINVAL, NULL);
    request = ornith_delta_request();
    request.first_col = 30u;
    request.col_count = 4u; /* runs off the end of 32 heads */
    ft_tt_test_check("a head slice past the end is refused",
                     ft_tt_op_gdnc_shape_is_supported(&request) == 0, NULL);
    /* PHASE DELTA rolls NOTHING, so the window's `chunk_rows >= taps-1` floor is not its. */
    request = ornith_delta_request();
    request.chunk_rows = 1u;
    ft_tt_test_check("a one-row DELTA chunk is ACCEPTED — the taps-1 floor belongs to the roll",
                     ft_tt_op_gdnc_shape_is_supported(&request) == 1, NULL);
    conv.chunk_rows = 1u;
    ft_tt_test_check("and the same one-row CONV chunk is REFUSED",
                     ft_tt_op_gdnc_shape_is_supported(&conv) == 0, NULL);

    /* Placement: ONE arming serves both phases, so every count is a common multiple. */
    request = ornith_delta_request();
    memset(entries, 0, sizeof(entries));
    ft_tt_test_check("the DELTA placement fits the arena",
                     ft_tt_op_gdnc_place_cbs(&request, entries, arena, sizeof(arena), &mask) ==
                         FT_TT_OK,
                     NULL);
    {
        const uint32_t input = 3u * FT_TT_OP_GDNC_QWEN35_STATE_TILES;
        const uint32_t state = FT_TT_OP_GDNC_QWEN35_STATE_TILES *
                               FT_TT_OP_GDNC_QWEN35_STATE_TILES;
        unsigned index;
        uint32_t pages[FT_TT_OP_GDNC_CB_COUNT];

        for (index = 0; index < FT_TT_OP_GDNC_CB_COUNT; index++)
            pages[index] = entries[index].page_count;

        /* EVERY ONE OF THESE IS AN INDEXED-READ GRID. `ft_tensix_unpack_ab(cb, ..., tile)` is
         * `read_pointer + tile*page_size` WITH NO WRAP, so a page count that is not a multiple
         * of the span reads a tile out of the buffer entirely — at some head that is not the
         * first, which is why a one-head test would not see it. */
        ft_tt_test_check_fmt("CB_IN holds a whole number of q|k|v groups",
                             pages[0] % input == 0u && pages[0] >= input, "%u pages", pages[0]);
        ft_tt_test_check_fmt("CB_STATE holds a whole number of [S,S] states",
                             pages[1] % state == 0u && pages[1] >= state, "%u pages", pages[1]);
        ft_tt_test_check_fmt("CB_BAND divides BOTH the CONV band and the four DELTA constants",
                             pages[2] % 8u == 0u &&
                                 pages[2] % FT_TT_OP_GDNC_DELTA_CONST_TILES == 0u,
                             "%u pages", pages[2]);
        ft_tt_test_check_fmt("CB_VECTOR divides BOTH taps and the five gate columns",
                             pages[3] % 4u == 0u &&
                                 pages[3] % FT_TT_OP_GDNC_DELTA_VECTOR_SPAN == 0u,
                             "%u pages", pages[3]);
        ft_tt_test_check_fmt("CB_WORK is exactly the nine-group span, so its push is an exact hit",
                             pages[5] ==
                                 FT_TT_OP_GDNC_DELTA_WORK_SPAN(FT_TT_OP_GDNC_QWEN35_STATE_TILES),
                             "%u pages", pages[5]);
        ft_tt_test_check_fmt("CB_CC is exactly the [C,C] span", 
                             pages[8] == FT_TT_OP_GDNC_DELTA_CC_SPAN, "%u pages", pages[8]);
        /* THE DEADLOCK CHECK. The writer drains `state_tiles` at a time -- a head is 20 pages
         * and this buffer is eight, so a writer that waited for a whole head would block on
         * page twenty while the compute thread blocked reserving page nine. */
        ft_tt_test_check_fmt("CB_OUT holds TWO of the writer's groups, not one head",
                             pages[9] >= 2u * FT_TT_OP_GDNC_QWEN35_STATE_TILES, "%u pages",
                             pages[9]);
    }

    /* DRAM sizes. THE STATE IS THE DECODE FAMILY'S, byte for byte — 32 heads of 4x4 fp32
     * tiles — which is what makes a prompt that prefills in chunks and then decodes ONE
     * sequence with no transcription at the seam. */
    request = ornith_delta_request();
    ft_tt_test_check_fmt("the staged q|k|v is (2*k_heads + v_heads)*state_tiles tiles",
                         ft_tt_op_gdnc_delta_input_bytes(&request) == (2u * 16u + 32u) * 4u * tile,
                         "%zu", ft_tt_op_gdnc_delta_input_bytes(&request));
    ft_tt_test_check_fmt("y is v_heads*state_tiles tiles",
                         ft_tt_op_gdnc_delta_output_bytes(&request) == 32u * 4u * tile, "%zu",
                         ft_tt_op_gdnc_delta_output_bytes(&request));
    ft_tt_test_check_fmt("the recurrent state is v_heads*state_tiles^2 tiles",
                         ft_tt_op_gdnc_delta_state_bytes(&request) == 32u * 16u * tile, "%zu",
                         ft_tt_op_gdnc_delta_state_bytes(&request));
    ft_tt_test_check_fmt("the DELTA constant buffer is four constants plus one tile a head",
                         ft_tt_op_gdnc_delta_constant_bytes(&request) == (4u + 32u) * tile,
                         "%zu", ft_tt_op_gdnc_delta_constant_bytes(&request));
    ft_tt_test_check_fmt("the gate block is v_heads runs of 160 floats",
                         ft_tt_op_gdnc_gate_bytes(&request) ==
                             32u * FT_TT_OP_GDNC_GATE_STRIDE * sizeof(float),
                         "%zu", ft_tt_op_gdnc_gate_bytes(&request));
    /* The reader pulls a head's run with ONE transaction and the NoC's granule is 64 bytes. */
    ft_tt_test_check_fmt("and a head's run is a whole number of 64-byte granules",
                         (FT_TT_OP_GDNC_GATE_STRIDE * sizeof(float)) % 64u == 0u, "%zu bytes",
                         FT_TT_OP_GDNC_GATE_STRIDE * sizeof(float));
}

/* ================================================================
 * Stage 8 — PHASE DELTA: a software model of the kernel, four chunks, against the oracle
 * ================================================================
 *
 * THE CLAIM IS THE UT TRANSFORM, and it is exactly checkable in C with no Tensix anywhere:
 *
 *     U   = (I + diag(b) StrictLower(D o K K^T))^-1 diag(b) (V - diag(P) K N_0^T)
 *     Y   = scale ( diag(P) Q N_0^T + (D o Q K^T) U )
 *     N_C = P_last N_0 + U^T diag(F) K
 *
 * against `qwen35moe_compute_recur_heads()` stepped ONE POSITION AT A TIME over the SAME
 * stream. Everything here is fp64, so the only difference it can show is a STRUCTURAL one; the
 * Tf32 residue the card adds is a separate question and is measured on silicon.
 *
 * THE DECAY MATRIX AND THE GATE COLUMNS ARE THE SHIPPING BUILDERS' OUTPUT, untilized — not a
 * second derivation. That is what makes this a check of the CODE rather than of the algebra: a
 * host that built `F` or `G` wrong for a ragged chunk would pass an algebra check and fail
 * this one.
 *
 * **THE SPLIT IS 32 + 32 + 32 + 7**, three full chunks and a ragged one, because a
 * single-chunk test cannot see a boundary bug: the incoming state is ZERO at chunk 0, so
 * `K_hat N_0^T` and `Q_hat N_0^T` vanish and HALF THE TRANSFORM IS UNTESTED. The decode
 * family's CONV recorded exactly this blind spot on the window, and `state_wipe` below
 * reproduces it on the recurrent state.
 *
 * FOUR RED ARMS, EVERY ONE RUN EVERY TIME:
 *   newton_none    stop the doubling at zero steps — the [C,C] solve stops being a solve
 *   strict_incl    an INCLUSIVE lower mask for A — k is L2-normalised, so this puts
 *                  `1 + beta_t` on the diagonal of a matrix whose structure is that it is UNIT
 *   causal_strict  a STRICT mask on the y path — that IS "(e) before (d)", the recurrence that
 *                  looks plausible for exactly one token
 *   state_wipe     zero the carried state between chunks — INVISIBLE IN CHUNK 0 by
 *                  construction, which is the quantitative form of "the carry is load-bearing"
 */

#define DELTA_S 128
#define DELTA_C 32
#define DELTA_HEADS 2

typedef enum {
    kDeltaRedNone = 0,
    kDeltaRedNewtonNone,
    kDeltaRedStrictIncl,
    kDeltaRedCausalStrict,
    kDeltaRedStateWipe,
    kDeltaRedCount
} delta_red;

static const char *const g_delta_red_name[kDeltaRedCount] = { "none", "newton_none",
                                                              "strict_incl", "causal_strict",
                                                              "state_wipe" };

static unsigned g_delta_rng = 0x1234567u;

/**
 * THE DECAY REGIME THE MODEL ARM DRAWS FROM, and it is a variable because the suite being
 * green while the model was not came down to this one range.
 *
 * The suite drew [0.90, 0.999] and nothing else. `0.90^32 = 0.034`, so `P` never came near a
 * denominator worth worrying about and `F_t = P_{chunk_rows-1} / P_t` looked exact. Ornith's
 * own decay is `exp(ssm_a * softplus(alpha))` and `ssm_a` reaches the tens, so a head can
 * carry 1e-7 — and thirty-two of those multiply to 1e-224, which is ZERO in fp32 after about
 * five positions. The ratio was then `0/0`.
 */
static double g_delta_decay_lo = 0.90;
static double g_delta_decay_hi = 0.999;

static double delta_sample(void)
{
    g_delta_rng = g_delta_rng * 1664525u + 1013904223u;
    return ((double)(g_delta_rng >> 8) / 16777216.0) * 2.0 - 1.0;
}

/** The oracle: `qwen35moe_compute_recur_heads()`, one position, state TRANSPOSED. */
static void delta_oracle_step(double *state, const double *q, const double *k, const double *v,
                              double decay, double beta, double scale, double *y)
{
    int i;
    int j;

    for (j = 0; j < DELTA_S; j++) {
        double *row = state + (size_t)j * DELTA_S;
        double pred = 0.0;
        double u;
        double out = 0.0;

        for (i = 0; i < DELTA_S; i++)
            row[i] *= decay;
        for (i = 0; i < DELTA_S; i++)
            pred += row[i] * k[i];
        u = (v[j] - pred) * beta;
        for (i = 0; i < DELTA_S; i++)
            row[i] += k[i] * u;
        /* (e) READS THE UPDATED STATE. Doing it before (d) is a different recurrence that
         * looks plausible for exactly one token — and `causal_strict` below is that arm. */
        for (i = 0; i < DELTA_S; i++)
            out += row[i] * q[i];
        y[j] = scale * out;
    }
}

/** Untilize one fp32 tile a shipping builder produced, into a `[32,32]` fp64 matrix. */
static void delta_untilize(const unsigned char *tile, double out[DELTA_C][DELTA_C])
{
    unsigned row;
    unsigned column;

    for (row = 0; row < DELTA_C; row++)
        for (column = 0; column < DELTA_C; column++) {
            const unsigned face = (row / 16u) * 2u + (column / 16u);
            const size_t offset =
                (size_t)face * 1024u + (size_t)(row % 16u) * 64u + (size_t)(column % 16u) * 4u;
            float value;

            memcpy(&value, tile + offset, sizeof(value));
            out[row][column] = (double)value;
        }
}

/** ONE CHUNK of the kernel's own op sequence, for one head. `state` is read and REWRITTEN. */
static void delta_model_chunk(double *state, /* NOT `const double (*)[N]`: ISO C before C23
                               * refuses a `double (*)[N]` argument for one, and GCC enforces
                               * that under -pedantic where clang does not. */
                              double q[DELTA_C][DELTA_S], double k[DELTA_C][DELTA_S],
                              double v[DELTA_C][DELTA_S],
                              double decay_matrix[DELTA_C][DELTA_C], const float *gate,
                              double scale, unsigned rows, delta_red red,
                              double y[DELTA_C][DELTA_S])
{
    static double a_matrix[DELTA_C][DELTA_C];
    static double inverse[DELTA_C][DELTA_C];
    static double residual[DELTA_C][DELTA_C];
    static double product[DELTA_C][DELTA_C];
    static double causal[DELTA_C][DELTA_C];
    static double staged[DELTA_C][DELTA_S];
    static double solved[DELTA_C][DELTA_S];
    const unsigned steps = (red == kDeltaRedNewtonNone) ? 0u : 4u;
    unsigned step;
    int t;
    int s;
    int i;
    int j;

    for (t = 0; t < DELTA_C; t++)
        for (s = 0; s < DELTA_C; s++) {
            double kk = 0.0;
            double qk = 0.0;
            int strict_ok = (red == kDeltaRedStrictIncl) ? (s <= t) : (s < t);
            int causal_ok = (red == kDeltaRedCausalStrict) ? (s < t) : (s <= t);

            for (i = 0; i < DELTA_S; i++) {
                kk += k[t][i] * k[s][i];
                qk += q[t][i] * k[s][i];
            }
            a_matrix[t][s] = (t == s ? 1.0 : 0.0) +
                             (strict_ok ? (double)gate[FT_TT_OP_GDNC_GATE_B * DELTA_C + (unsigned)t] *
                                              decay_matrix[t][s] * kk
                                        : 0.0);
            causal[t][s] = causal_ok ? decay_matrix[t][s] * qk : 0.0;
        }

    /* Newton, started at `2I - A`. `E_k = A_off^(2^(k+1))`, so four doublings are EXACT. */
    for (t = 0; t < DELTA_C; t++)
        for (s = 0; s < DELTA_C; s++)
            inverse[t][s] = (t == s ? 2.0 : 0.0) - a_matrix[t][s];
    for (step = 0; step < steps; step++) {
        for (t = 0; t < DELTA_C; t++)
            for (s = 0; s < DELTA_C; s++) {
                double accumulator = 0.0;

                for (i = 0; i < DELTA_C; i++)
                    accumulator += a_matrix[t][i] * inverse[i][s];
                residual[t][s] = (t == s ? 2.0 : 0.0) - accumulator;
            }
        for (t = 0; t < DELTA_C; t++)
            for (s = 0; s < DELTA_C; s++) {
                double accumulator = 0.0;

                for (i = 0; i < DELTA_C; i++)
                    accumulator += inverse[t][i] * residual[i][s];
                product[t][s] = accumulator;
            }
        memcpy(inverse, product, sizeof(inverse));
    }

    /* U = A^-1 diag(beta) (V - diag(P) K N_0^T). */
    for (t = 0; t < DELTA_C; t++)
        for (j = 0; j < DELTA_S; j++) {
            double accumulator = 0.0;

            for (i = 0; i < DELTA_S; i++)
                accumulator += (double)gate[FT_TT_OP_GDNC_GATE_P * DELTA_C + (unsigned)t] * k[t][i] *
                               state[(size_t)j * DELTA_S + (size_t)i];
            staged[t][j] =
                (double)gate[FT_TT_OP_GDNC_GATE_B * DELTA_C + (unsigned)t] * (v[t][j] - accumulator);
        }
    for (t = 0; t < DELTA_C; t++)
        for (j = 0; j < DELTA_S; j++) {
            double accumulator = 0.0;

            for (i = 0; i < DELTA_C; i++)
                accumulator += inverse[t][i] * staged[i][j];
            solved[t][j] = accumulator;
        }

    /* Y = scale (diag(P) Q N_0^T + (D o QK^T) U). The scale rides on the `PS` gate run for
     * the first term, so a wrong `PS` shows here and nowhere else. */
    for (t = 0; t < DELTA_C; t++)
        for (j = 0; j < DELTA_S; j++) {
            double from_state = 0.0;
            double from_chunk = 0.0;

            for (i = 0; i < DELTA_S; i++)
                from_state += (double)gate[FT_TT_OP_GDNC_GATE_PS * DELTA_C + (unsigned)t] * q[t][i] *
                              state[(size_t)j * DELTA_S + (size_t)i];
            for (i = 0; i < DELTA_C; i++)
                from_chunk += causal[t][i] * solved[i][j];
            y[t][j] = from_state + scale * from_chunk;
        }

    /* N_C = G N_0 + U^T diag(F) K. Rows at and above the chunk carry `beta = 0`, so they are
     * exactly zero in U and take no part in this sum whatever `F` says. */
    for (j = 0; j < DELTA_S; j++)
        for (i = 0; i < DELTA_S; i++) {
            double accumulator = (double)gate[FT_TT_OP_GDNC_GATE_G * DELTA_C + 0] *
                                 state[(size_t)j * DELTA_S + (size_t)i];

            for (t = 0; t < (int)rows; t++)
                accumulator +=
                    solved[t][j] * (double)gate[FT_TT_OP_GDNC_GATE_F * DELTA_C + (unsigned)t] * k[t][i];
            state[(size_t)j * DELTA_S + (size_t)i] = accumulator;
        }
}

static double delta_run(delta_red red)
{
    static const unsigned split[4] = { 32u, 32u, 32u, 7u };
    static double oracle_state[DELTA_HEADS][DELTA_S * DELTA_S];
    static double model_state[DELTA_HEADS][DELTA_S * DELTA_S];
    static double q[DELTA_C][DELTA_S];
    static double k[DELTA_C][DELTA_S];
    static double v[DELTA_C][DELTA_S];
    static double y_model[DELTA_C][DELTA_S];
    static double y_oracle[DELTA_S];
    static float decay[DELTA_C];
    static float beta[DELTA_C];
    static float gate[FT_TT_OP_GDNC_GATE_STRIDE];
    static unsigned char decay_tile[FT_TT_OP_GDNC_FP32_TILE_BYTES];
    static double decay_matrix[DELTA_C][DELTA_C];
    const double scale = 1.0 / sqrt((double)DELTA_S);
    double worst = 0.0;
    double reference = 0.0;
    unsigned chunk;
    unsigned head;

    g_delta_rng = 0x1234567u;
    memset(oracle_state, 0, sizeof(oracle_state));
    memset(model_state, 0, sizeof(model_state));

    for (chunk = 0; chunk < 4u; chunk++) {
        const unsigned rows = split[chunk];

        for (head = 0; head < DELTA_HEADS; head++) {
            unsigned t;
            int i;

            for (t = 0; t < DELTA_C; t++) {
                double norm_q = 0.0;
                double norm_k = 0.0;

                for (i = 0; i < DELTA_S; i++) {
                    /* THE HOST OWES ZEROS ABOVE THE CHUNK, and the model honours that rather
                     * than assuming it: a ragged chunk's dead rows are zero in q, k and v. */
                    q[t][i] = (t < rows) ? delta_sample() : 0.0;
                    k[t][i] = (t < rows) ? delta_sample() : 0.0;
                    v[t][i] = (t < rows) ? delta_sample() : 0.0;
                    norm_q += q[t][i] * q[t][i];
                    norm_k += k[t][i] * k[t][i];
                }
                if (t < rows) {
                    /* q and k are L2-NORMALISED before the delta rule, which is what makes
                     * `D[t][t] (k_t.k_t)` exactly 1 and A exactly unit triangular. */
                    norm_q = 1.0 / sqrt(norm_q + 1e-6);
                    norm_k = 1.0 / sqrt(norm_k + 1e-6);
                    for (i = 0; i < DELTA_S; i++) {
                        q[t][i] *= norm_q;
                        k[t][i] *= norm_k;
                    }
                }
                decay[t] = (float)(g_delta_decay_lo +
                                   (g_delta_decay_hi - g_delta_decay_lo) *
                                       (delta_sample() * 0.5 + 0.5));
                beta[t] = (float)(1.0 / (1.0 + exp(-delta_sample() * 2.0)));
            }

            /* THE SHIPPING BUILDERS, not a second derivation. */
            if (ft_tt_op_gdnc_build_decay(decay, rows, decay_tile, sizeof(decay_tile)) !=
                    FT_TT_OK ||
                ft_tt_op_gdnc_build_gate(decay, beta, rows, DELTA_S, gate,
                                         FT_TT_OP_GDNC_GATE_STRIDE) != FT_TT_OK)
                return 1.0e9;
            delta_untilize(decay_tile, decay_matrix);

            /* A NaN IS INVISIBLE TO A MAXIMUM, AND THAT IS WHY THIS COUNT EXISTS.
             *
             * The score below is `if (difference > worst) worst = difference`, and every
             * comparison against a NaN is FALSE — so a gate column full of NaNs propagates
             * into `y_model`, never becomes `worst`, and the arm reports the small number the
             * finite entries produced. MEASURED 2026-08-21: at Ornith's decays `P` underflows
             * and `F_t = P_{chunk_rows-1}/P_t` was `0/0` for 20 of 32 rows, and this suite
             * scored it 7.775e-15 and called it green. The tenth vacuous-gate class recorded
             * in this project. */
            {
                unsigned column;

                for (column = 0; column < FT_TT_OP_GDNC_GATE_STRIDE; column++)
                    if (!(gate[column] > -1e30f && gate[column] < 1e30f))
                        return 1.0e9;
                for (t = 0; t < DELTA_C; t++)
                    for (i = 0; i < DELTA_C; i++)
                        if (!(decay_matrix[t][i] > -1e30 && decay_matrix[t][i] < 1e30))
                            return 1.0e9;
            }

            if (red == kDeltaRedStateWipe && chunk > 0u)
                memset(model_state[head], 0, sizeof(model_state[head]));

            delta_model_chunk(model_state[head], q, k, v, decay_matrix, gate, scale, rows, red,
                              y_model);

            for (t = 0; t < rows; t++) {
                delta_oracle_step(oracle_state[head], q[t], k[t], v[t], (double)decay[t],
                                  (double)beta[t], scale, y_oracle);
                for (i = 0; i < DELTA_S; i++) {
                    const double difference = fabs(y_model[t][i] - y_oracle[i]);

                    if (difference > worst)
                        worst = difference;
                    if (fabs(y_oracle[i]) > reference)
                        reference = fabs(y_oracle[i]);
                }
            }
        }
    }
    return (reference > 0.0) ? worst / reference : worst;
}

static void stage_delta_model(void)
{
    /* THE ARM'S FLOOR IS fp32, NOT fp64, and that is not a compromise: the model runs its
     * arithmetic in fp64 but takes `D` and the gate columns from THE SHIPPING BUILDERS, which
     * produce the card's fp32 operands. A 32-long cumulative product of fp32 decays carries
     * ~1e-7 of relative rounding that the oracle — which multiplies the same fp32 decays one
     * at a time — does not. The measured arm is 1.1e-07 and the CLOSEST red arm is 2.1e-02,
     * five orders above it, so the gate discriminates by a wide margin at 1e-5. Setting it at
     * fp64 round-off would be gating on a difference the operands cannot express. */
    static const double arm_gate = 1e-5;
    double arm;
    unsigned red;

    ft_tt_test_step("stage 8 — PHASE DELTA's UT transform, 32+32+32+7, against the oracle");

    arm = delta_run(kDeltaRedNone);
    ft_tt_test_check_fmt("the chunkwise transform reproduces the sequential oracle",
                         arm < arm_gate, "%.3e max-norm relative (gate %.0e)", arm, arm_gate);

    /* THE SECOND REGIME, AND IT IS THE ONE THE MODEL ACTUALLY HAS.
     *
     * MEASURED ON SILICON 2026-08-21: at Ornith's decays and C = 32 the chunkwise prefill
     * produced a completely different completion, while being `cmp` IDENTICAL to the control
     * at 4 rows a pass. `P` underflows to zero and `F_t = P_{chunk_rows-1}/P_t` becomes
     * `0/0`. This suite could not see it because it only ever drew decays near 1 — so the
     * regime is now a variable and BOTH are run. `F` is a backward product recurrence now and
     * has no denominator at all; before that change this arm scores NaN, which fails every
     * comparison it is put into. */
    g_delta_decay_lo = 1.0e-7;
    g_delta_decay_hi = 1.0e-3;
    {
        const double tiny = delta_run(kDeltaRedNone);

        ft_tt_test_check_fmt("...and at ORNITH'S OWN DECAYS, where P underflows in fp32",
                             tiny < arm_gate, "%.3e max-norm relative (gate %.0e)", tiny,
                             arm_gate);
    }
    g_delta_decay_lo = 0.90;
    g_delta_decay_hi = 0.999;

    /* A GATE THAT HAS NEVER PRODUCED A LOW NUMBER IS NOT A GATE. Every arm is run. */
    for (red = (unsigned)kDeltaRedNewtonNone; red < (unsigned)kDeltaRedCount; red++) {
        const double score = delta_run((delta_red)red);

        ft_tt_test_check_fmt("a red arm breaks it", score > 1e-4, "%s scores %.3e against %.3e",
                             g_delta_red_name[red], score, arm);
    }
}

/* ================================================================
 * Stage 9 — THE IN-PLACE SUB-CHUNK
 * ================================================================ */

/**
 * The three ways a caller can get `first_row` wrong, every one of which retires clean.
 *
 *   beta_live     leave `beta` alive on the rows this descriptor does not own. `U` is then
 *                 nonzero there, it feeds the carried state through `U^T diag(F) K` and the y
 *                 path through `(D o QK^T) U`, and the answer is wrong from the SECOND
 *                 sub-chunk on.
 *   whole_y       the writer ignores `first_row` and writes the whole tile, so the last
 *                 sub-chunk's zeros erase every earlier sub-chunk's answer.
 *   prefix_write  the writer takes `[0, first_row + live)` instead of the suffix
 *                 `[first_row, 32)` — the plausible other reading of a row range, and it
 *                 clobbers every earlier sub-chunk with THIS one's dead rows.
 *
 * TWO THINGS IN THIS DESIGN ARE DELIBERATELY NOT RED ARMS, BECAUSE THIS HARNESS CANNOT SEE
 * THEM AND A GATE THAT CANNOT FAIL IS THE ELEVENTH VACUOUS CLASS ALL OVER AGAIN:
 *
 *   - `P_t = 0` ON THE DEAD ROWS IS UNOBSERVABLE UNDER A SUFFIX WRITE. The rows BELOW the
 *     sub-chunk are never written, and the rows above it carry q = 0 (the host owes zeros
 *     above the pass) or are overwritten by the next sub-chunk. Zero is kept because it is the
 *     value that stays harmless if the write ever widens — not because it is load-bearing
 *     today, and it is not dressed up as a mechanism.
 *   - `D[t][s] = 0` FOR `s` BELOW THE SUB-CHUNK IS INERT IN fp64 AND LETHAL IN Tf32. With
 *     `beta = 0` on the dead rows `A` is block lower triangular with an identity diagonal
 *     block, so `(A^-1)_live = (A_live)^-1` EXACTLY and leaving those columns alive changes
 *     nothing an fp64 model can measure. What it changes is the NEWTON PATH: `A_off` stays a
 *     32-wide nilpotent whose iterates reach 1.7e+06 before collapsing, which is precisely the
 *     C = 32 failure. That is a Tf32-truncation effect and the instrument for it is the
 *     silicon parity gate, not this file.
 */
typedef enum {
    kSubRedNone = 0,
    kSubRedBetaLive,
    kSubRedWholeY,
    kSubRedPrefixWrite,
    kSubRedCount
} sub_red;

static const char *const g_sub_red_name[kSubRedCount] = { "none", "beta_live", "whole_y",
                                                          "prefix_write" };

/**
 * Worst relative y error over the pass, and through `out_dead` the largest |y| a SUB-CHUNK
 * (`first_row > 0`) leaves on a row above its own live range.
 *
 * THAT SECOND FIGURE HAS TO BE EXACTLY ZERO AND THE `first_row > 0` QUALIFIER IS THE WHOLE OF
 * WHY. The y write is a contiguous SUFFIX `[first_row, 32)`, so a descriptor lands every row
 * above itself. For `first_row = 0` those rows are the ragged tail the whole-pass descriptor
 * has always written junk into and nobody has ever read — unchanged behaviour, and demanding
 * zero there would be gating on something this rung did not touch. For `first_row > 0` they
 * are rows an EARLIER sub-chunk may have written, so the value is not free: it has to be the
 * zero the host's `P_t = 0` and `D[t][.] = 0` produce.
 */
static double delta_run_sub(const unsigned *split, unsigned sub, sub_red red, double *out_dead)
{
    static double oracle_state[DELTA_HEADS][DELTA_S * DELTA_S];
    static double model_state[DELTA_HEADS][DELTA_S * DELTA_S];
    static double q[DELTA_C][DELTA_S];
    static double k[DELTA_C][DELTA_S];
    static double v[DELTA_C][DELTA_S];
    static double y_model[DELTA_C][DELTA_S];
    static double y_pass[DELTA_C][DELTA_S];
    static double y_oracle[DELTA_S];
    static float decay[DELTA_C];
    static float beta[DELTA_C];
    static float gate[FT_TT_OP_GDNC_GATE_STRIDE];
    static unsigned char decay_tile[FT_TT_OP_GDNC_FP32_TILE_BYTES];
    static double decay_matrix[DELTA_C][DELTA_C];
    const double scale = 1.0 / sqrt((double)DELTA_S);
    double worst = 0.0;
    double reference = 0.0;
    double dead = 0.0;
    unsigned chunk;
    unsigned head;

    g_delta_rng = 0x1234567u;
    memset(oracle_state, 0, sizeof(oracle_state));
    memset(model_state, 0, sizeof(model_state));

    for (chunk = 0; chunk < 4u; chunk++) {
        const unsigned rows = split[chunk];

        for (head = 0; head < DELTA_HEADS; head++) {
            unsigned first;
            unsigned t;
            int i;

            for (t = 0; t < DELTA_C; t++) {
                double norm_q = 0.0;
                double norm_k = 0.0;

                for (i = 0; i < DELTA_S; i++) {
                    q[t][i] = (t < rows) ? delta_sample() : 0.0;
                    k[t][i] = (t < rows) ? delta_sample() : 0.0;
                    v[t][i] = (t < rows) ? delta_sample() : 0.0;
                    norm_q += q[t][i] * q[t][i];
                    norm_k += k[t][i] * k[t][i];
                }
                if (t < rows) {
                    norm_q = 1.0 / sqrt(norm_q + 1e-6);
                    norm_k = 1.0 / sqrt(norm_k + 1e-6);
                    for (i = 0; i < DELTA_S; i++) {
                        q[t][i] *= norm_q;
                        k[t][i] *= norm_k;
                    }
                }
                decay[t] = (float)(g_delta_decay_lo +
                                   (g_delta_decay_hi - g_delta_decay_lo) *
                                       (delta_sample() * 0.5 + 0.5));
                beta[t] = (float)(1.0 / (1.0 + exp(-delta_sample() * 2.0)));
            }
            memset(y_pass, 0, sizeof(y_pass));

            /* THE WHOLE POINT: ONE STRIP, `ceil(rows/sub)` DESCRIPTORS, NOTHING SHUTTLED. q, k
             * and v are untouched between sub-chunks and the carried state walks forward
             * exactly as it walks across passes. */
            for (first = 0; first < rows; first += sub) {
                const unsigned live = (rows - first < sub) ? rows - first : sub;
                unsigned row;

                /* THE SHIPPING BUILDERS, sliced — not a second derivation of either. */
                if (ft_tt_op_gdnc_build_decay_at(decay + first, first, live, decay_tile,
                                                 sizeof(decay_tile)) != FT_TT_OK ||
                    ft_tt_op_gdnc_build_gate_at(decay + first, beta + first, first, live,
                                                DELTA_S, gate,
                                                FT_TT_OP_GDNC_GATE_STRIDE) != FT_TT_OK)
                    return 1.0e9;
                delta_untilize(decay_tile, decay_matrix);

                if (red == kSubRedBetaLive)
                    for (row = 0; row < first; row++)
                        gate[FT_TT_OP_GDNC_GATE_B * DELTA_C + row] = 0.5f;
                /* `rows = DELTA_C`: the kernel's state sum is a whole `[C,S]` matmul over all
                 * 32 rows, so passing the live count would make the model KINDER than the
                 * card. Summing all 32 is what proves `U` really is zero off the sub-chunk. */
                delta_model_chunk(model_state[head], q, k, v, decay_matrix, gate, scale,
                                  DELTA_C, kDeltaRedNone, y_model);

                /* THE WRITER'S ROW RANGE, at the host end: the SUFFIX `[first_row, 32)`. */
                {
                    const unsigned from = (red == kSubRedWholeY || red == kSubRedPrefixWrite)
                                              ? 0u
                                              : first;
                    const unsigned to =
                        (red == kSubRedPrefixWrite) ? first + live : (unsigned)DELTA_C;

                    for (row = from; row < to; row++)
                        memcpy(y_pass[row], y_model[row], sizeof(y_pass[row]));
                }
                if (first > 0u)
                    for (row = first + live; row < DELTA_C; row++)
                        for (i = 0; i < DELTA_S; i++)
                            if (fabs(y_model[row][i]) > dead)
                                dead = fabs(y_model[row][i]);
            }

            for (t = 0; t < rows; t++) {
                delta_oracle_step(oracle_state[head], q[t], k[t], v[t], (double)decay[t],
                                  (double)beta[t], scale, y_oracle);
                for (i = 0; i < DELTA_S; i++) {
                    const double difference = fabs(y_pass[t][i] - y_oracle[i]);

                    if (difference > worst)
                        worst = difference;
                    if (fabs(y_oracle[i]) > reference)
                        reference = fabs(y_oracle[i]);
                }
            }
        }
    }
    if (out_dead)
        *out_dead = dead;
    return (reference > 0.0) ? worst / reference : worst;
}

static void stage_delta_sub_chunk(void)
{
    /* The established prefill shape, and one that makes the LAST sub-chunk of a pass ragged —
     * 20 rows at C = 16 is `first_row = 16` with four live rows, the only configuration in
     * which a sub-chunk writes rows above itself that an earlier one already wrote. */
    static const unsigned even[4] = { 32u, 32u, 32u, 7u };
    static const unsigned ragged[4] = { 32u, 20u, 32u, 7u };
    static const double arm_gate = 1e-5;
    double dead = 0.0;
    double arm;
    unsigned red;

    ft_tt_test_step("stage 9 — the IN-PLACE SUB-CHUNK: C = 16 over one 32-row strip");

    /* THE CONTROL FIRST. `sub = 32` covers every pass, so `first_row` is 0 throughout and the
     * `_at` builders delegate to the shipping ones — this must reproduce stage 8's number
     * exactly, or the sub-chunk harness is measuring its own scaffolding. */
    arm = delta_run_sub(even, 32u, kSubRedNone, &dead);
    ft_tt_test_check_fmt("sub = 32 is the whole-pass descriptor, unchanged", arm < arm_gate,
                         "%.3e max-norm relative (gate %.0e)", arm, arm_gate);

    arm = delta_run_sub(even, 16u, kSubRedNone, &dead);
    ft_tt_test_check_fmt("C = 16 IN PLACE reproduces the sequential oracle", arm < arm_gate,
                         "%.3e max-norm relative (gate %.0e)", arm, arm_gate);

    arm = delta_run_sub(ragged, 16u, kSubRedNone, &dead);
    ft_tt_test_check_fmt("...and with a RAGGED last sub-chunk, 20 rows as 16 + 4",
                         arm < arm_gate, "%.3e max-norm relative (gate %.0e)", arm, arm_gate);
    ft_tt_test_check_fmt("a sub-chunk leaves EXACTLY zero above itself, since the y write is a "
                         "suffix",
                         dead == 0.0, "largest |y| above a sub-chunk is %.3e", dead);

    /* AND AT ORNITH'S OWN DECAYS, where P underflows in fp32 — the regime the model has. */
    g_delta_decay_lo = 1.0e-7;
    g_delta_decay_hi = 1.0e-3;
    arm = delta_run_sub(ragged, 16u, kSubRedNone, &dead);
    ft_tt_test_check_fmt("...and at ORNITH'S OWN DECAYS, where P underflows in fp32",
                         arm < arm_gate && dead == 0.0,
                         "%.3e max-norm relative, %.3e above the sub-chunk", arm, dead);
    g_delta_decay_lo = 0.90;
    g_delta_decay_hi = 0.999;

    /* A GATE THAT HAS NEVER READ LOW IS NOT A GATE, and neither is one that has never read
     * high. Every arm is run, on the ragged split, which is the only one where all three can
     * reach the code they mutate. */
    for (red = (unsigned)kSubRedBetaLive; red < (unsigned)kSubRedCount; red++) {
        double red_dead = 0.0;
        const double score = delta_run_sub(ragged, 16u, (sub_red)red, &red_dead);
        const int fired = (score > 1e-4) || (red_dead > 0.0);

        ft_tt_test_check_fmt("a red arm breaks it", fired,
                             "%s scores %.3e with %.3e above the sub-chunk",
                             g_sub_red_name[red], score, red_dead);
    }
}

int main(void)
{
    ft_tt_test_step("test_gdnc_ops — the CHUNKWISE Gated-Delta-Net family, off hardware");

    stage_wire_contract();
    stage_encoding();
    stage_refusals();
    stage_placement();
    stage_band();
    stage_arithmetic();
    stage_sizes();
    stage_delta_host();
    stage_delta_model();
    stage_delta_sub_chunk();

    return ft_tt_test_finish();
}
