/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_gdn_ops.c
 * @brief Off-hardware acceptance for the GATED-DELTA-NET op family: the wire contract
 *   between `ft_tt_op_gdn.h` and `kernels/resident_gdn_protocol.h`, and the host driver's
 *   parameter encoding, shape refusals and CB placement.
 *
 * ---------------------------------------------------------------------------
 * WHY THE WIRE PIN PARSES THE HEADER INSTEAD OF INCLUDING IT
 * ---------------------------------------------------------------------------
 * The two halves of the contract are declared TWICE — once as a C enum the host compiles
 * and once as the `enum { name = value };` form the device kernels compile — and nothing in
 * either build makes them agree. Including the device header here would make the test
 * compile against the same declaration the kernel does, which is the right thing to want
 * and the wrong thing to test: the failure this suite exists to catch is one of the two
 * being edited alone. So it READS the file, exactly as `test_rope_ops.c`, `test_vector_ops.c`
 * and `test_topk_ops.c` do for their own families, and a constant that is MISSING is a
 * failure rather than a default — a default passes against a header that no longer declares
 * the field.
 *
 * The opcode itself is pinned the same way, across the two files that carry it: the host's
 * `FT_TT_QUEUE_OP_GDN` in `ft_tt_queue.h` against the device's `kOpGdn` in
 * `kernels/resident_queue_protocol.h`.
 *
 * ---------------------------------------------------------------------------
 * AND `FT_TT_OP_GDN_WORDS == FT_TT_QUEUE_PARAM_WORDS` IS NOT PEDANTRY
 * ---------------------------------------------------------------------------
 * Every family's parameter block is 16 words because `ft_tt_queue_write_op_parameters()`
 * writes exactly `FT_TT_QUEUE_PARAM_WORDS` of them into a slot whose STRIDE is shared with
 * every other family. A seventeenth word does not overflow a buffer here; it silently moves
 * the next family's slot on the card. This op is the one most tempted to want a
 * seventeenth — it has five DRAM endpoints where rope has three — which is exactly why its
 * coordinates are packed two to a word, and why that pack is asserted rather than trusted.
 *
 * ---------------------------------------------------------------------------
 * AND THE LAYOUT ARMS INCLUDE THE DEVICE HEADER, WHICH IS NOT A CONTRADICTION
 * ---------------------------------------------------------------------------
 * The layout accessors (`ftg_delta_stream_base`, `ftg_state_tile`, `ftg_conv_plane_tile`,
 * `ftg_y_tile`) are declared ONCE, in the device header, and both dataflow kernels call
 * them — there is no second declaration for them to drift from. The drift that matters
 * there is between the accessor and whatever a future host stager writes into DRAM, and the
 * only way to test THAT off-hardware is to compute the expected tile index independently
 * from the oracle and compare. So those arms include the header and check its arithmetic;
 * the wire-constant arms parse it and check its text. Different failure, different method.
 *
 * ---------------------------------------------------------------------------
 * THE DELTA GEOMETRY THIS SUITE NOW PINS: HEAD-MAJOR ROW-0 STRIPS
 * ---------------------------------------------------------------------------
 * `kGdnActivationIn` (PHASE DELTA's q | k | v) and `kGdnActivationOut` (y) are ROW-0 STRIPS,
 * head-major: a head's S values are `state_tiles` CONSECUTIVE tiles, all of the data in ROW 0
 * (`kGdnStripRow`) of each. Head `h` starts at `h * state_tiles` inside its block, which is
 * what `ftg_delta_head_tile()` and `ftg_y_tile()` return.
 *
 * They used to be three tilized `[heads, S]` matrices with ONE ROW PER HEAD — 12 input tiles
 * and 4 y tiles at Ornith, against 256 and 128 now. That packing was 32x denser and cost
 * nothing to read, but it was a DIFFERENT GEOMETRY from the strip every projection, reduce and
 * elementwise op in this backend produces, so the host had to de-tilize conv_out, L2-normalise
 * on the CPU and re-tilize between the phases. The strip is what lets the block run WHOLE on
 * the card, and the literals below are the new contract.
 *
 * The distinguishing property, and the one the arms below assert directly, is INJECTIVITY:
 * under the old `ftg_y_tile()` all 32 v-heads collapsed onto the same four tiles and were told
 * apart by their ROW, so a check that (head, tile_row) -> tile is one-to-one is red against it.
 *
 * @status Off-hardware only, by construction — no device is opened. The triad is written,
 *   built and blob-embedded but has never executed, so every claim here is host arithmetic
 *   and header text.
 * @issues None
 * @todo An on-card arm gating the delta rule against pn_qwen35moe_ssm_recurrence at 1e-3
 *   relative, once the op has a caller.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ft_tt_op_gdn.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_queue.h"
#include "ft_tt_test_report.h"

/* The device's L1 pointer attribute is a property of the SFPI GCC fork, not of the contract.
 * The header guards it, so naming it as nothing lets a HOST compiler read the file. */
#define FTQ_L1_PTR
#include "../kernels/resident_gdn_protocol.h"

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
    char label[160];
    int rc;

    device_header_path(path, sizeof(path), file);
    rc = read_device_constant(path, name, &value);
    snprintf(label, sizeof(label), "%s::%s is the host's value", file, name);
    ft_tt_test_check_fmt(label, rc == 0 && value == expected, "device %u, host %u (rc=%d)",
                         value, expected, rc);
}

/**
 * The `position`'th comma-separated value (0-based) after `key` on any line of `path`, e.g.
 * `read_ctarg_value(path, "READER_CTARGS=", 7, &v)` against `... READER_CTARGS=0,1,2,3,4,
 * 4096,1,1,8 ...` reads `1`. A missing key or a list shorter than `position` is a FAILURE,
 * the same discipline as `read_device_constant` above and for the same reason: a build line
 * that no longer says what this test expects must be a red, not a value nobody checked.
 */
static int read_ctarg_value(const char *path, const char *key, unsigned position,
                            uint32_t *out_value)
{
    char line[512];
    FILE *file = fopen(path, "r");
    int found = -1;

    if (!file)
        return -1;
    while (found != 0 && fgets(line, sizeof(line), file)) {
        char *cursor = strstr(line, key);
        unsigned index;

        if (!cursor)
            continue;
        cursor += strlen(key);
        for (index = 0; index <= position; index++) {
            char *end;
            unsigned long value = strtoul(cursor, &end, 10);

            if (end == cursor)
                break; /* not a digit where the next list value was expected */
            if (index == position) {
                *out_value = (uint32_t)value;
                found = 0;
                break;
            }
            if (*end != ',')
                break; /* the list ended before `position` */
            cursor = end + 1;
        }
    }
    fclose(file);
    return found;
}

static void check_ctarg(const char *file, const char *key, unsigned position, const char *what,
                        uint32_t expected)
{
    char path[PATH_MAX_BYTES];
    uint32_t value = 0xFFFFFFFFu;
    int rc;

    device_header_path(path, sizeof(path), file);
    rc = read_ctarg_value(path, key, position, &value);
    ft_tt_test_check_fmt(what, rc == 0 && value == expected,
                         "%s[%u] in %s: device %u, host %u (rc=%d)", key, position, file,
                         value, expected, rc);
}

static void test_wire_constants(void)
{
    ft_tt_test_step("the host and the KERNEL agree on every wire word — PARSED, not claimed");

    check_device_constant("resident_gdn_protocol.h", "kGdnShape",
                          (uint32_t)FT_TT_OP_GDN_SHAPE);
    check_device_constant("resident_gdn_protocol.h", "kGdnConvTileCols",
                          (uint32_t)FT_TT_OP_GDN_CONV_TILE_COLS);
    check_device_constant("resident_gdn_protocol.h", "kGdnConvTaps",
                          (uint32_t)FT_TT_OP_GDN_CONV_TAPS);
    check_device_constant("resident_gdn_protocol.h", "kGdnActivationXy",
                          (uint32_t)FT_TT_OP_GDN_ACTIVATION_XY);
    check_device_constant("resident_gdn_protocol.h", "kGdnActivationIn",
                          (uint32_t)FT_TT_OP_GDN_ACTIVATION_IN);
    check_device_constant("resident_gdn_protocol.h", "kGdnActivationOut",
                          (uint32_t)FT_TT_OP_GDN_ACTIVATION_OUT);
    check_device_constant("resident_gdn_protocol.h", "kGdnConstantXy",
                          (uint32_t)FT_TT_OP_GDN_CONSTANT_XY);
    check_device_constant("resident_gdn_protocol.h", "kGdnConstantWeight",
                          (uint32_t)FT_TT_OP_GDN_CONSTANT_WEIGHT);
    check_device_constant("resident_gdn_protocol.h", "kGdnConstantGate",
                          (uint32_t)FT_TT_OP_GDN_CONSTANT_GATE);
    check_device_constant("resident_gdn_protocol.h", "kGdnStateXy",
                          (uint32_t)FT_TT_OP_GDN_STATE_XY);
    check_device_constant("resident_gdn_protocol.h", "kGdnStateConv",
                          (uint32_t)FT_TT_OP_GDN_STATE_CONV);
    check_device_constant("resident_gdn_protocol.h", "kGdnStateRecurrent",
                          (uint32_t)FT_TT_OP_GDN_STATE_RECURRENT);
    check_device_constant("resident_gdn_protocol.h", "kGdnScaleBits",
                          (uint32_t)FT_TT_OP_GDN_SCALE_BITS);
    check_device_constant("resident_gdn_protocol.h", "kGdnTileBytes",
                          (uint32_t)FT_TT_OP_GDN_TILE_BYTES);
    check_device_constant("resident_gdn_protocol.h", "kGdnFlags",
                          (uint32_t)FT_TT_OP_GDN_FLAGS);
    check_device_constant("resident_gdn_protocol.h", "kGdnFlagTable",
                          FT_TT_OP_GDN_FLAG_TABLE);
    check_device_constant("resident_gdn_protocol.h", "kGdnTileCount",
                          (uint32_t)FT_TT_OP_GDN_TILE_COUNT);
    check_device_constant("resident_gdn_protocol.h", "kGdnWords",
                          (uint32_t)FT_TT_OP_GDN_WORDS);
    check_device_constant("resident_gdn_protocol.h", "kGdnConvTileColsMask",
                          (uint32_t)FT_TT_OP_GDN_COL_MAX);

    /* The two phase ids, because a descriptor that names the wrong half of the block would
     * run the conv over the recurrent state's address. */
    check_device_constant("resident_gdn_protocol.h", "kGdnPhaseConv", FT_TT_OP_GDN_PHASE_CONV);
    check_device_constant("resident_gdn_protocol.h", "kGdnPhaseDelta", FT_TT_OP_GDN_PHASE_DELTA);

    /* The opcode, across the two files that carry it. */
    check_device_constant("resident_queue_protocol.h", "kOpGdn",
                          (uint32_t)FT_TT_QUEUE_OP_GDN);

    ft_tt_test_check_fmt("the parameter block is FT_TT_QUEUE_PARAM_WORDS wide",
                         (uint32_t)FT_TT_OP_GDN_WORDS == FT_TT_QUEUE_PARAM_WORDS,
                         "gdn %u, queue %u", (unsigned)FT_TT_OP_GDN_WORDS,
                         (unsigned)FT_TT_QUEUE_PARAM_WORDS);

    /* 0x12 was the next free opcode. Pinned so that a family added later cannot silently
     * take it: two families on one opcode is a descriptor executed by the wrong kernel. */
    ft_tt_test_check_fmt("kOpGdn sits above every opcode that existed before it",
                         FT_TT_QUEUE_OP_GDN > FT_TT_QUEUE_OP_ATTENTION, "0x%02X > 0x%02X",
                         (unsigned)FT_TT_QUEUE_OP_GDN, (unsigned)FT_TT_QUEUE_OP_ATTENTION);
}

/**
 * FT_TT_OP_GDN_CONV_DEPTH and FT_TT_OP_GDN_CB_STAGE are not read back off any device
 * structure the way the wire words above are. ft_tt_op_gdn.h says outright that raising the
 * depth is "a measured change to this constant, `place_cbs` and both kernel build lines
 * together" -- three sides, and nothing enforces that they move together. A header edit
 * alone changes `resolve_cb_pages()` and this file's own `placement_satisfies_device()` in
 * lockstep, since both read the same macro, so re-deriving the expectation from the header
 * cannot catch a mutation of the header: it has to be pinned against a text the header does
 * not own. That text is the CTARGS the kernel SOURCE COMMENTS document as the actual build
 * line -- what `programs/gdn` on the card host was built with -- and it does not move just
 * because the header does.
 */
static void test_compile_time_arg_contract(void)
{
    ft_tt_test_step("FT_TT_OP_GDN_CONV_DEPTH and FT_TT_OP_GDN_CB_STAGE, pinned against the "
                    "kernel build-line comments and not re-derived from the header they pin");

    ft_tt_test_check_fmt("FT_TT_OP_GDN_CONV_DEPTH is the depth every kernel build line "
                         "compiles in",
                         FT_TT_OP_GDN_CONV_DEPTH == 2u, "%u",
                         (unsigned)FT_TT_OP_GDN_CONV_DEPTH);
    ft_tt_test_check_fmt("FT_TT_OP_GDN_CB_STAGE is the CB index the host and the reader "
                         "both use for the private staging buffer",
                         FT_TT_OP_GDN_CB_STAGE == 4u, "%u", (unsigned)FT_TT_OP_GDN_CB_STAGE);

    /* THE THIRD AND FOURTH SIDES: the reader's own compile-time arg 7 and the compute
     * kernel's own compile-time arg 9, read out of the build lines their own file headers
     * document (resident_gdn_reader.c:121, resident_gdn_compute.c:141). */
    check_ctarg("resident_gdn_reader.c", "READER_CTARGS=", 7u,
               "resident_gdn_reader.c's own READER_CTARGS[7] is FT_TT_OP_GDN_CONV_DEPTH",
               (uint32_t)FT_TT_OP_GDN_CONV_DEPTH);
    check_ctarg("resident_gdn_compute.c", "COMPUTE_CTARGS=", 9u,
               "resident_gdn_compute.c's own COMPUTE_CTARGS[9] is FT_TT_OP_GDN_CONV_DEPTH",
               (uint32_t)FT_TT_OP_GDN_CONV_DEPTH);
    check_ctarg("resident_gdn_reader.c", "READER_CTARGS=", 4u,
               "resident_gdn_reader.c's own READER_CTARGS[4] is FT_TT_OP_GDN_CB_STAGE",
               (uint32_t)FT_TT_OP_GDN_CB_STAGE);

    /* THE DST BATCH IS THE ONE CONTRACT WHOSE BREAKAGE IS SILENT AND SLOW. A short build line
     * folds its arg to a literal 0, `ftg_dst_batch()` maps 0 to 1, and the op then runs the
     * per-tile sequence at the shipped 3,368 µs a DELTA descriptor while every host-side
     * number says 4 — no refusal, no wrong answer, nothing on the card to look at. The arg
     * COUNT beside each build line is the only guard, so it is pinned on BOTH kernels here
     * and CB_PROD's index with it, since the reader's refusal is what stops a placement the
     * batched indexed read would walk off. */
    ft_tt_test_check_fmt("FT_TT_OP_GDN_DST_BATCH is the batch every kernel build line "
                         "compiles in",
                         FT_TT_OP_GDN_DST_BATCH == 4u, "%u",
                         (unsigned)FT_TT_OP_GDN_DST_BATCH);
    check_ctarg("resident_gdn_reader.c", "READER_CTARGS=", 9u,
               "resident_gdn_reader.c's own READER_CTARGS[9] is FT_TT_OP_GDN_CB_PROD",
               (uint32_t)FT_TT_OP_GDN_CB_PROD);
    check_ctarg("resident_gdn_reader.c", "READER_CTARGS=", 10u,
               "resident_gdn_reader.c's own READER_CTARGS[10] is FT_TT_OP_GDN_DST_BATCH",
               (uint32_t)FT_TT_OP_GDN_DST_BATCH);
    check_ctarg("resident_gdn_compute.c", "COMPUTE_CTARGS=", 11u,
               "resident_gdn_compute.c's own COMPUTE_CTARGS[11] is FT_TT_OP_GDN_DST_BATCH",
               (uint32_t)FT_TT_OP_GDN_DST_BATCH);

    /* THE TWO RESTRUCTURING MASKS, and they are the ones that carry this op's 1.76x. They
     * fail the same silent way the batch does — a short build line folds the arg to 0 and the
     * kernel quietly runs the slow shape — except that here the slow shape is 3,351 µs
     * against 1,905, so the cost of the silence is 43% of PHASE DELTA rather than 1%. */
    ft_tt_test_check_fmt("FT_TT_OP_GDN_READER_EXTRAS is the mask every reader build line "
                         "compiles in",
                         FT_TT_OP_GDN_READER_EXTRAS == 1u, "%u",
                         (unsigned)FT_TT_OP_GDN_READER_EXTRAS);
    ft_tt_test_check_fmt("FT_TT_OP_GDN_DELTA_EXTRAS is the mask every compute build line "
                         "compiles in, and it does NOT carry the push-batch bit that refused "
                         "on silicon",
                         FT_TT_OP_GDN_DELTA_EXTRAS == 1u, "%u",
                         (unsigned)FT_TT_OP_GDN_DELTA_EXTRAS);
    check_ctarg("resident_gdn_reader.c", "READER_CTARGS=", 11u,
               "resident_gdn_reader.c's own READER_CTARGS[11] is FT_TT_OP_GDN_READER_EXTRAS",
               (uint32_t)FT_TT_OP_GDN_READER_EXTRAS);
    check_ctarg("resident_gdn_compute.c", "COMPUTE_CTARGS=", 12u,
               "resident_gdn_compute.c's own COMPUTE_CTARGS[12] is FT_TT_OP_GDN_DELTA_EXTRAS",
               (uint32_t)FT_TT_OP_GDN_DELTA_EXTRAS);
}

/**
 * `ftg_dst_batch()` is the ONE implementation of the clamp, called by the reader, by the
 * compute kernel and by `resolve_cb_pages()` — so what it returns is a three-sided fact and
 * the arms below are the only thing that says what it should be.
 *
 * The CEILING is the part that cannot be inferred from anything else in the tree. DST is 16
 * tiles of 32x32, DST_SYNC half hands MATH one half, and `dst_accum_mode 1` — mandatory for
 * the one op carrying state across tokens — halves it again, so four is the largest index the
 * hardware has. A batch above it does not fail: it ALIASES DST tiles onto each other and
 * produces a smoothly wrong recurrence, which is the failure this family has already been
 * bitten by twice under other names.
 */
static void test_dst_batch_clamp(void)
{
    uint32_t state_tiles;

    ft_tt_test_step("ftg_dst_batch() clamps to the fp32 half-DST ceiling, to the tile-row, "
                    "and onto the divisor grid — at every shape the family admits");

    ft_tt_test_check_fmt("the ceiling is four — DST is 16 tiles, half sync halves it and "
                         "dst_accum_mode 1 halves it again",
                         (uint32_t)kGdnDstBatchCeiling == 4u, "%u",
                         (unsigned)kGdnDstBatchCeiling);
    ft_tt_test_check_fmt("a zero max — what a SHORT build line folds to — is the identity, "
                         "not a division by zero",
                         ftg_dst_batch(4u, 0u) == 1u, "%u", ftg_dst_batch(4u, 0u));
    ft_tt_test_check_fmt("a max above the ceiling is clamped to it, not honoured",
                         ftg_dst_batch(8u, 64u) == (uint32_t)kGdnDstBatchCeiling, "%u",
                         ftg_dst_batch(8u, 64u));
    ft_tt_test_check_fmt("Ornith's own shape batches WHOLE — state_tiles 4, batch 4, one DST "
                         "section a tile-row loop",
                         ftg_dst_batch(FT_TT_OP_GDN_QWEN35_STATE / FT_TT_OP_TILE_DIM,
                                       FT_TT_OP_GDN_DST_BATCH) == 4u,
                         "%u", ftg_dst_batch(FT_TT_OP_GDN_QWEN35_STATE / FT_TT_OP_TILE_DIM,
                                             FT_TT_OP_GDN_DST_BATCH));

    for (state_tiles = 1u; state_tiles <= 8u; state_tiles++) {
        const uint32_t batch = ftg_dst_batch(state_tiles, FT_TT_OP_GDN_DST_BATCH);
        char label[160];

        snprintf(label, sizeof(label),
                 "S=%u (%u state tiles): batch %u is >=1, <= the ceiling, <= the tile-row and "
                 "DIVIDES it",
                 state_tiles * 32u, state_tiles, batch);
        ft_tt_test_check_fmt(label,
                             batch >= 1u && batch <= (uint32_t)kGdnDstBatchCeiling &&
                                 batch <= state_tiles && (state_tiles % batch) == 0u,
                             "batch=%u tiles=%u", batch, state_tiles);
    }
}

/**
 * `ftg_writer_block()` is the writer's barrier group and the host's CB_OUT size, from ONE
 * implementation — the same two-sided shape `ftg_dst_batch()` has and for the same reason.
 *
 * WHAT ONLY THIS TEST SAYS is the BARRIER COUNT each rung buys, because nothing on the card
 * reports it: 640 / 128 / 32 a DELTA descriptor at Ornith's 32 v-heads. A rung that quietly
 * returned the wrong group would be a slower kernel and a correct answer, which is the one
 * failure class this family has no on-card signal for.
 */
static void test_writer_block_rungs(void)
{
    const uint32_t tiles = FT_TT_OP_GDN_QWEN35_STATE / FT_TT_OP_TILE_DIM;
    const uint32_t heads = FT_TT_OP_GDN_QWEN35_V_HEADS;
    const uint32_t pages = heads * tiles * (tiles + 1u);
    uint32_t rung;

    ft_tt_test_step("ftg_writer_block() — the rungs, the arena ceiling, and the barrier count "
                    "each one leaves in a DELTA descriptor");

    ft_tt_test_check_fmt("rung 0 is the IDENTITY: one barrier per tile, the sequence that "
                         "shipped and what a short build line folds to",
                         ftg_writer_block(tiles, 0u) == 1u, "%u", ftg_writer_block(tiles, 0u));
    ft_tt_test_check_fmt("rung 1 is one TILE-ROW — `state_tiles` state pages then one y",
                         ftg_writer_block(tiles, 1u) == tiles + 1u, "%u",
                         ftg_writer_block(tiles, 1u));
    ft_tt_test_check_fmt("rung 2 is one HEAD — every tile-row of it",
                         ftg_writer_block(tiles, 2u) == tiles * (tiles + 1u), "%u",
                         ftg_writer_block(tiles, 2u));
    ft_tt_test_check_fmt("a state size of zero cannot divide, so it is the identity",
                         ftg_writer_block(0u, 2u) == 1u, "%u", ftg_writer_block(0u, 2u));
    ft_tt_test_check_fmt("the ceiling binds at S=256, where an unclamped rung 2 would ask for "
                         "144 CB_OUT pages and put the placement over the arena",
                         ftg_writer_block(8u, 2u) == (uint32_t)kGdnWriterBlockCeiling, "%u",
                         ftg_writer_block(8u, 2u));

    for (rung = 0u; rung <= 2u; rung++) {
        const uint32_t block = ftg_writer_block(tiles, rung);
        const uint32_t barriers = pages / block;
        char label[176];

        snprintf(label, sizeof(label),
                 "rung %u: %u CB_OUT pages a group BEFORE the placement clamp, so a 32-head "
                 "DELTA descriptor takes %u write barriers, not %u",
                 rung, block, barriers, pages);
        ft_tt_test_check_fmt(label,
                             block >= 1u && (pages % block) == 0u &&
                                 barriers == (rung == 0u ? 640u : (rung == 1u ? 128u : 32u)),
                             "block=%u barriers=%u", block, barriers);
    }
}

/* ================================================================
 * THE WRITER'S PAGE WALK — the half of the barrier group that silicon cannot check
 * ================================================================
 *
 * Grouping the barrier moves the POPS behind it, and a batched consumer then has to name
 * pages by INDEX off its front. `read_pointer + index * page` is what every indexed CB read
 * in this backend does, and a FIFO pointer returns to its base on an EXACT HIT of the limit
 * and on nothing else — so the indexed form reads straight past the buffer whenever a group
 * straddles the end. `page_address()` in resident_gdn_writer.c spells the wrap out for that
 * reason, and this is the model that says the wrap is REACHED rather than defensive.
 *
 * IT IS REACHED BECAUSE OF PHASE CONV, and that is not obvious from the DELTA side alone.
 * A DELTA descriptor moves `head_count * state_tiles * (state_tiles + 1)` pages — always a
 * multiple of 20, which divides a 40-page CB_OUT, so DELTA after DELTA always starts on the
 * base. But CONV moves `conv_tile_cols * taps` = 1,024 pages, and 1,024 mod 40 is 24, so
 * EVERY DELTA descriptor in a real layer stack starts 24 pages in and every group after the
 * fourth straddles the end.
 *
 * WHAT THE MODEL ASSERTS is the only thing that matters: the writer reads page `k` of the
 * stream from the SAME L1 address the producer pushed page `k` into. The producer advances
 * one page at a time with an exact-hit wrap, so that address is `base + (k mod pages)*page`
 * and nothing about the rung may change it. Measured on silicon 2026-08-19 from the other
 * side: the same writer with the wrap patched out returned 56 bytes and `cmp` DIFFERS.
 */

typedef struct wb_fifo {
    uint32_t base;
    uint32_t page;
    uint32_t pages;
    uint32_t size;
    uint32_t limit; /* one past the end, as ft_tensix_cb_pop_front() compares it */
    uint32_t read;
} wb_fifo;

static void wb_fifo_init(wb_fifo *fifo, uint32_t base, uint32_t page, uint32_t pages)
{
    fifo->base = base;
    fifo->page = page;
    fifo->pages = pages;
    fifo->size = page * pages;
    fifo->limit = base + fifo->size;
    fifo->read = base;
}

/** Where the PRODUCER put stream page `k`: it pushes one page at a time from the base. */
static uint32_t wb_producer_address(const wb_fifo *fifo, uint32_t k)
{
    return fifo->base + (k % fifo->pages) * fifo->page;
}

/** resident_gdn_writer.c `page_address()`, transliterated. `wrap` 0 is the RED arm. */
static uint32_t wb_page_address(const wb_fifo *fifo, uint32_t front, uint32_t index, int wrap)
{
    uint32_t address = front + index * fifo->page;

    if (wrap && address >= fifo->limit)
        address -= fifo->size;
    return address;
}

/**
 * `ft_tensix_cb_pop_front(cb, pages)` — advance, wrap on an EXACT hit and NOTHING ELSE.
 *
 * The exact-hit comparison is upstream's and is the whole reason the writer pops ONE PAGE AT
 * A TIME instead of popping its group. At `pages == 1` the pointer visits every page boundary
 * and therefore always hits the limit exactly; at `pages == group` it steps over the limit
 * whenever the group is not aligned to it, the wrap never fires, and the pointer walks out of
 * the buffer and never comes back. `wb_pop_group_would_escape()` below is the model of that.
 */
static void wb_pop(wb_fifo *fifo, uint32_t pages)
{
    fifo->read += pages * fifo->page;
    if (fifo->read == fifo->limit)
        fifo->read -= fifo->size;
}

static void wb_pop_one(wb_fifo *fifo)
{
    wb_pop(fifo, 1u);
}

/**
 * `writer_block()` in resident_gdn_writer.c: the rung's group, CLAMPED to half the placed
 * pages. The clamp is what stops a kernel built at a rung the host did not size for from
 * waiting on more pages than the buffer can ever hold — and the model has to apply it too,
 * or it tests a state the card cannot reach. Proven on silicon 2026-08-19: the rung-1 kernel
 * under a two-page CB_OUT ran `cmp` IDENTICAL at rung-0 speed (1910.7 us against 1908.3)
 * rather than refusing the descriptor.
 */
static uint32_t wb_clamped_block(uint32_t state_tiles, uint32_t rung, uint32_t cb_pages)
{
    uint32_t block = ftg_writer_block(state_tiles, rung);

    if (block > cb_pages / 2u)
        block = cb_pages / 2u;
    if (block == 0u)
        block = 1u;
    return block;
}

/**
 * Drain one descriptor's `pages` CB_OUT pages in groups of `group`, as write_one_delta() and
 * write_one_conv() do. Returns the barrier count; advances `*k`, the stream page counter.
 *
 * `out_bad` counts pages whose source address is NOT where the producer put them — which is
 * either a read past the buffer or a read of the wrong page, and both are wrong answers.
 */
static uint32_t wb_drain(wb_fifo *fifo, uint32_t pages, uint32_t group, int wrap, uint32_t *k,
                         uint32_t *out_bad, uint32_t *out_off_buffer)
{
    uint32_t emitted = 0u;
    uint32_t barriers = 0u;

    while (emitted < pages) {
        uint32_t n = pages - emitted;
        uint32_t front;
        uint32_t index;

        if (n > group)
            n = group;
        front = fifo->read;
        for (index = 0; index < n; index++) {
            const uint32_t source = wb_page_address(fifo, front, index, wrap);

            if (source < fifo->base || source >= fifo->limit)
                (*out_off_buffer)++;
            if (source != wb_producer_address(fifo, *k + index))
                (*out_bad)++;
        }
        barriers++;
        for (index = 0; index < n; index++)
            wb_pop_one(fifo);
        emitted += n;
        *k += n;
    }
    return barriers;
}

/** The DELTA destination for stream page `slot_page` of head `head`, as the writer decodes it.
 *  Encoded so a whole descriptor's destinations can be compared between rungs. */
static uint32_t wb_delta_destination(const ftg_shape *shape, uint32_t head, uint32_t page)
{
    const uint32_t row_pages = shape->state_tiles + 1u;
    const uint32_t tile_row = page / row_pages;
    const uint32_t slot = page % row_pages;

    if (slot < shape->state_tiles)
        return ftg_state_tile(shape, head, tile_row, slot);
    return 0x40000000u | ftg_y_tile(shape, head, tile_row);
}

static void test_writer_page_walk(void)
{
    const uint32_t tiles = FT_TT_OP_GDN_QWEN35_STATE / FT_TT_OP_TILE_DIM;
    const uint32_t heads = FT_TT_OP_GDN_QWEN35_V_HEADS;
    const uint32_t conv_pages =
        (FT_TT_OP_GDN_QWEN35_CONV_DIM / FT_TT_OP_TILE_DIM) * FT_TT_OP_GDN_QWEN35_CONV_TAPS;
    const uint32_t delta_pages = heads * tiles * (tiles + 1u);
    const uint32_t cb_pages = 2u * ftg_writer_block(tiles, FT_TT_OP_GDN_WRITER_RUNG);
    ftg_shape shape;
    uint32_t rung;
    uint32_t reference[1024];
    uint32_t page;

    ft_tt_test_step("the WRITER's page walk — the barrier group changes WHEN it drains, never "
                    "WHICH page it reads or where that page goes");

    memset(&shape, 0, sizeof(shape));
    shape.state_tiles = tiles;
    shape.v_heads = heads;

    /* The destinations, decoded from a page's position in the stream. Rung-independent by
     * construction, so recording them once is enough — what the loop below proves is that
     * the SOURCES stay matched to them however the groups fall. */
    for (page = 0; page < tiles * (tiles + 1u); page++)
        reference[page] = wb_delta_destination(&shape, 0u, page);
    ft_tt_test_check_fmt("a head's 20 pages are 4 state tiles then one y, four times over",
                         reference[0] == ftg_state_tile(&shape, 0u, 0u, 0u) &&
                             reference[4] == (0x40000000u | 0u) &&
                             reference[19] == (0x40000000u | 3u),
                         "%08x %08x %08x", reference[0], reference[4], reference[19]);

    /* AND THE LAST HEAD'S y PAGES ARE NOT THE FIRST HEAD'S. y is a HEAD-MAJOR ROW-0 STRIP, so
     * head 31's four y destinations are tiles 124..127 and they share nothing with head 0's
     * 0..3. Under the old one-row-per-head y every head drained into the SAME four tiles and
     * this pair compared equal, which is why the arm is spelled as a disagreement. */
    {
        uint32_t last[32];
        int disjoint = 1;
        uint32_t index;

        for (page = 0; page < tiles * (tiles + 1u); page++)
            last[page] = wb_delta_destination(&shape, heads - 1u, page);
        for (page = 0; page < tiles * (tiles + 1u); page++)
            for (index = 0; index < tiles * (tiles + 1u); index++)
                if ((last[page] & 0x40000000u) != 0u && last[page] == reference[index])
                    disjoint = 0;
        ft_tt_test_check_fmt("head 31's y pages are tiles 124..127 and share none of head 0's",
                             disjoint && last[4] == (0x40000000u | 124u) &&
                                 last[19] == (0x40000000u | 127u),
                             "%08x %08x disjoint=%d", last[4], last[19], disjoint);
    }

    ft_tt_test_check_fmt("CONV leaves the read pointer OFF the base, and that is what makes "
                         "the wrap reachable at all",
                         (conv_pages % cb_pages) != 0u, "conv_pages=%u cb_pages=%u rem=%u",
                         conv_pages, cb_pages, conv_pages % cb_pages);

    for (rung = 0u; rung <= 2u; rung++) {
        const uint32_t group = wb_clamped_block(tiles, rung, cb_pages);
        uint32_t bad = 0u;
        uint32_t off = 0u;
        uint32_t k = 0u;
        uint32_t delta_barriers = 0u;
        uint32_t layer;
        wb_fifo fifo;
        char label[208];

        wb_fifo_init(&fifo, 0x30000u, FT_TT_OP_GDN_FP32_TILE_BYTES, cb_pages);
        /* Twenty-four SSM layers of a real token: CONV then DELTA, CONV then DELTA. */
        for (layer = 0; layer < 24u; layer++) {
            (void)wb_drain(&fifo, conv_pages, 1u, 1, &k, &bad, &off);
            delta_barriers = wb_drain(&fifo, delta_pages, group, 1, &k, &bad, &off);
        }
        snprintf(label, sizeof(label),
                 "rung %u (group %u after the clamp): 24 layers of CONV+DELTA — every page "
                 "read where the producer put it, none off the buffer, %u barriers a DELTA "
                 "descriptor",
                 rung, group, delta_barriers);
        ft_tt_test_check_fmt(label,
                             bad == 0u && off == 0u && group >= 1u &&
                                 delta_barriers == (delta_pages + group - 1u) / group,
                             "bad=%u off_buffer=%u barriers=%u", bad, off, delta_barriers);
    }

    /* THE OTHER HALF OF THE SAME RULE: popping the GROUP instead of one page at a time.
     * `ft_tensix_cb_pop_front()` wraps on an EXACT hit of the limit and on nothing else, so a
     * pointer that STEPS OVER the limit never comes back. CONV leaves the front 24 pages in
     * and a group of 5 from there lands on 29, 34, 39, 44 — 44 is past the 40-page limit and
     * is not equal to it, so the wrap does not fire. This is why the writer's pops are
     * spelled one at a time and it is invisible in any DELTA-only test. */
    for (rung = 1u; rung <= 2u; rung++) {
        const uint32_t group = wb_clamped_block(tiles, rung, cb_pages);
        wb_fifo fifo;
        uint32_t emitted;
        int escaped = 0;
        char label[192];

        wb_fifo_init(&fifo, 0x30000u, FT_TT_OP_GDN_FP32_TILE_BYTES, cb_pages);
        for (emitted = 0; emitted < conv_pages; emitted++)
            wb_pop_one(&fifo);
        for (emitted = 0; emitted < delta_pages; emitted += group) {
            wb_pop(&fifo, group);
            if (fifo.read < fifo.base || fifo.read >= fifo.limit)
                escaped = 1;
        }
        snprintf(label, sizeof(label),
                 "rung %u: popping the whole %u-page group instead of one page at a time walks "
                 "the read pointer OUT of the buffer — the exact-hit wrap never fires",
                 rung, group);
        ft_tt_test_check_fmt(label, escaped, "read=%08x base=%08x limit=%08x", fifo.read,
                             fifo.base, fifo.limit);
    }

    /* THE RED ARM, off hardware. The same walk with page_address()'s wrap removed — the build
     * that returned 56 bytes and `cmp` DIFFERS on silicon. Rung 0 must be UNAFFECTED, because
     * a group of one never straddles anything; that is what says the model discriminates the
     * wrap rather than the rung. */
    for (rung = 0u; rung <= 2u; rung++) {
        uint32_t bad = 0u;
        uint32_t off = 0u;
        uint32_t k = 0u;
        uint32_t layer;
        wb_fifo fifo;
        char label[192];

        wb_fifo_init(&fifo, 0x30000u, FT_TT_OP_GDN_FP32_TILE_BYTES, cb_pages);
        for (layer = 0; layer < 24u; layer++) {
            (void)wb_drain(&fifo, conv_pages, 1u, 0, &k, &bad, &off);
            (void)wb_drain(&fifo, delta_pages, wb_clamped_block(tiles, rung, cb_pages), 0,
                           &k, &bad, &off);
        }
        snprintf(label, sizeof(label),
                 "rung %u WITHOUT the wrap: %s",
                 rung,
                 rung == 0u ? "a group of one cannot straddle, so nothing moves — the model "
                              "discriminates the wrap, not the rung"
                            : "reads land off the buffer, which is the 56-byte `cmp` DIFFERS "
                              "arm reproduced off hardware");
        ft_tt_test_check_fmt(label, rung == 0u ? (bad == 0u && off == 0u) : (bad > 0u && off > 0u),
                             "bad=%u off_buffer=%u", bad, off);
    }
}

/* ================================================================
 * A request at Ornith's real shape
 * ================================================================ */

static ft_tt_op_operand operand(uint16_t x, uint16_t y, uint32_t address, uint32_t tile_bytes)
{
    ft_tt_op_operand out;

    out.x = x;
    out.y = y;
    out.address = address;
    out.tile_bytes = tile_bytes;
    return out;
}

static void fill_delta_request(ft_tt_op_gdn_request *request)
{
    memset(request, 0, sizeof(*request));
    request->phase = FT_TT_OP_GDN_PHASE_DELTA;
    request->state_size = FT_TT_OP_GDN_QWEN35_STATE;
    request->v_heads = FT_TT_OP_GDN_QWEN35_V_HEADS;
    request->k_heads = FT_TT_OP_GDN_QWEN35_K_HEADS;
    request->conv_dim = FT_TT_OP_GDN_QWEN35_CONV_DIM;
    request->conv_taps = FT_TT_OP_GDN_QWEN35_CONV_TAPS;
    /* EVERY operand is fp32. The whole family's CB table is Float32 and both dataflow
     * kernels are built with TILE_BYTES = 4096; this fixture used to hand three of the four
     * operands the bf16 size, which is exactly the request that placed 2048-byte pages under
     * a reader issuing 4096-byte tile reads. */
    request->activation_in = operand(1u, 2u, 0x00100000u, FT_TT_OP_TILE_BYTES_FP32);
    request->activation_out = operand(1u, 2u, 0x00200000u, FT_TT_OP_TILE_BYTES_FP32);
    request->gate = operand(3u, 4u, 0x00300000u, FT_TT_OP_TILE_BYTES_FP32);
    request->state_recurrent = operand(5u, 6u, 0x00400000u, FT_TT_OP_TILE_BYTES_FP32);
}

static void fill_conv_request(ft_tt_op_gdn_request *request)
{
    fill_delta_request(request);
    request->phase = FT_TT_OP_GDN_PHASE_CONV;
    request->gate = operand(0u, 0u, 0u, 0u);
    request->state_recurrent = operand(0u, 0u, 0u, 0u);
    request->conv_weight = operand(3u, 4u, 0x00500000u, FT_TT_OP_TILE_BYTES_FP32);
    request->state_conv = operand(5u, 6u, 0x00600000u, FT_TT_OP_TILE_BYTES_FP32);
}

/**
 * THE CARD-SIDE HALF OF THE CONV-DEPTH CONTRACT — the one the checks above cannot reach.
 *
 * `check_ctarg` above pins the build-line COMMENTS, which is a statement about text in this
 * checkout. It says nothing about the images a program directory on the card host was actually
 * built from, and a depth sweep runs several directories against one host binary on purpose.
 * The drift that matters is the SILENT direction: a host that placed for a LARGER depth than
 * the kernels carry passes every capacity and grid check this family has, runs at the
 * kernels' depth, and returns the right answer at the CONTROL'S SPEED. The arm reads green
 * and reports a number that belongs to a different build.
 *
 * `kGdnByteConvDepth` closes it by putting the host's own constant on the wire, and
 * `ftg_conv_depth_agrees()` is the single comparison both kernels make. This step proves the
 * encoder writes it, that a decode recovers it, and — the part that makes the gate non-vacuous
 * — that the comparison can FAIL, on both a wrong depth and the zero a pre-declaration encoder
 * would leave.
 */
static void test_declared_conv_depth(void)
{
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];
    ftg_shape shape;
    int rc;

    ft_tt_test_step("the CONV block depth is DECLARED on the wire and the kernels' comparison "
                    "is proven able to fail");

    /* THE COMPARISON ITSELF, BEFORE ANY ENCODER. Both kernels call this and nothing else. */
    ft_tt_test_check_fmt("ftg_conv_depth_agrees() is true only on equality",
                         ftg_conv_depth_agrees(1u, 1u) && ftg_conv_depth_agrees(4u, 4u),
                         "1/1 %d 4/4 %d", (int)ftg_conv_depth_agrees(1u, 1u),
                         (int)ftg_conv_depth_agrees(4u, 4u));
    /* RED: the silent direction. Host placed for 4, kernels built at 1. Every capacity and
     * grid check in conv_is_executable() passes this pair; only the declared depth sees it. */
    ft_tt_test_check_fmt("RED — a host at depth 4 against kernels at depth 1 DISAGREES",
                         !ftg_conv_depth_agrees(4u, 1u), "%d",
                         (int)ftg_conv_depth_agrees(4u, 1u));
    /* RED: the loud direction, which conv_block_depth() already caught. Both now refuse. */
    ft_tt_test_check_fmt("RED — a host at depth 1 against kernels at depth 4 DISAGREES",
                         !ftg_conv_depth_agrees(1u, 4u), "%d",
                         (int)ftg_conv_depth_agrees(1u, 4u));
    /* RED: there is deliberately NO "0 means unset" escape, which is the opposite of the head
     * slice's compatibility hinge. An encoder that forgets to declare must refuse, not pass. */
    ft_tt_test_check_fmt("RED — an UNDECLARED depth (byte 3 zero) DISAGREES with every real "
                         "build, so a forgetful encoder refuses instead of passing",
                         !ftg_conv_depth_agrees(0u, 1u) && !ftg_conv_depth_agrees(0u, 4u), "%d %d",
                         (int)ftg_conv_depth_agrees(0u, 1u),
                         (int)ftg_conv_depth_agrees(0u, 4u));

    /* THE ENCODER. A CONV descriptor carries this host's constant in byte 3 of the taps word,
     * and `ftg_decode()` — the kernels' own decode, not a respelling of it — recovers it. */
    fill_conv_request(&request);
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("a CONV request encodes", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("byte 3 of the CONV-taps word is this host's own CONV block depth",
                         ((words[FT_TT_OP_GDN_CONV_TAPS] >> 24) & 0xFFu) ==
                             (ft_tt_op_gdn_conv_depth() & 0xFFu),
                         "0x%08X vs depth %u", words[FT_TT_OP_GDN_CONV_TAPS],
                         ft_tt_op_gdn_conv_depth());
    shape = ftg_decode(words);
    ft_tt_test_check_fmt("the kernels' own ftg_decode() recovers it, and it AGREES with a "
                         "matching build",
                         shape.declared_conv_depth == ft_tt_op_gdn_conv_depth() &&
                             ftg_conv_depth_agrees(shape.declared_conv_depth,
                                                   ft_tt_op_gdn_conv_depth()),
                         "%u", shape.declared_conv_depth);
    /* AND THE DEPTH DID NOT LAND ON A FIELD THAT WAS ALREADY THERE. Bytes 0-2 are the taps and
     * the head slice; a shift that overlapped one of them would be caught here and nowhere
     * else, because every other arm reads them through the same accessor that wrote them. */
    ft_tt_test_check_fmt("declaring the depth left the taps and the head slice alone",
                         shape.conv_taps == FT_TT_OP_GDN_QWEN35_CONV_TAPS &&
                             shape.head_first == 0u && shape.head_count == request.v_heads,
                         "taps %u first %u count %u", shape.conv_taps, shape.head_first,
                         shape.head_count);

    /* PHASE DELTA DOES NOT READ THE DEPTH, so its taps word stays zero and its descriptor is
     * byte-for-byte the one that shipped. A declaration on DELTA would have refused every
     * DELTA descriptor the moment the kernels started comparing. */
    fill_delta_request(&request);
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("a DELTA descriptor declares NOTHING and its taps word is still zero",
                         rc == FT_TT_OK && words[FT_TT_OP_GDN_CONV_TAPS] == 0u, "rc=%d 0x%08X",
                         rc, words[FT_TT_OP_GDN_CONV_TAPS]);

    /* THE DEPTH HAS TO FIT THE BYTE IT RIDES IN. Nothing else in the tree says so. */
    ft_tt_test_check_fmt("FT_TT_OP_GDN_CONV_DEPTH fits byte 3 and is not zero",
                         FT_TT_OP_GDN_CONV_DEPTH >= 1u && FT_TT_OP_GDN_CONV_DEPTH <= 255u, "%u",
                         (unsigned)FT_TT_OP_GDN_CONV_DEPTH);

    /* THE KNOB THAT LETS ONE BINARY RUN THE WHOLE SWEEP, tested through the PARSER rather
     * than the cached accessor: the accessor answers once per process and a sweep needs more
     * than one answer per test run. */
    ft_tt_test_check_fmt("an unset knob is the compile-time default",
                         ft_tt_op_gdn_conv_depth_parse(NULL) == FT_TT_OP_GDN_CONV_DEPTH &&
                             ft_tt_op_gdn_conv_depth_parse("") == FT_TT_OP_GDN_CONV_DEPTH,
                         "%u %u", ft_tt_op_gdn_conv_depth_parse(NULL),
                         ft_tt_op_gdn_conv_depth_parse(""));
    ft_tt_test_check_fmt("every rung of the sweep parses to itself",
                         ft_tt_op_gdn_conv_depth_parse("1") == 1u &&
                             ft_tt_op_gdn_conv_depth_parse("2") == 2u &&
                             ft_tt_op_gdn_conv_depth_parse("4") == 4u &&
                             ft_tt_op_gdn_conv_depth_parse("255") == 255u,
                         "%u %u %u %u", ft_tt_op_gdn_conv_depth_parse("1"),
                         ft_tt_op_gdn_conv_depth_parse("2"), ft_tt_op_gdn_conv_depth_parse("4"),
                         ft_tt_op_gdn_conv_depth_parse("255"));
    /* RED, and it is the clause that matters: a knob that quietly falls back to the default
     * on a value it cannot read measures the CONTROL once per rung and prints a flat ladder
     * that reads exactly like a refuted lever. */
    ft_tt_test_check_fmt("RED — 0, 256, a negative, a trailing character and pure text ALL "
                         "return 0, never the default",
                         ft_tt_op_gdn_conv_depth_parse("0") == 0u &&
                             ft_tt_op_gdn_conv_depth_parse("256") == 0u &&
                             ft_tt_op_gdn_conv_depth_parse("-1") == 0u &&
                             ft_tt_op_gdn_conv_depth_parse("4x") == 0u &&
                             ft_tt_op_gdn_conv_depth_parse("eight") == 0u,
                         "%u %u %u %u %u", ft_tt_op_gdn_conv_depth_parse("0"),
                         ft_tt_op_gdn_conv_depth_parse("256"),
                         ft_tt_op_gdn_conv_depth_parse("-1"),
                         ft_tt_op_gdn_conv_depth_parse("4x"),
                         ft_tt_op_gdn_conv_depth_parse("eight"));
    /* AND WHAT THE ACCESSOR ANSWERS IS WHAT THE DESCRIPTOR CARRIED, on this process's own
     * value — the two halves of the knob checked against each other rather than separately. */
    ft_tt_test_check_fmt("the descriptor declared the accessor's answer, not the macro's",
                         shape.declared_conv_depth == ft_tt_op_gdn_conv_depth(), "%u vs %u",
                         shape.declared_conv_depth, ft_tt_op_gdn_conv_depth());
}

static void test_parameters(void)
{
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];
    float scale;
    int rc;

    ft_tt_test_step("the parameter encoding, at Ornith's real shape (S=128, 32 v-heads, "
                    "16 k-heads, conv_dim 8192, 4 taps)");

    fill_delta_request(&request);
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("a DELTA request encodes", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;

    ft_tt_test_check_fmt("the shape word packs phase, S/32, v-heads and k-heads",
                         (words[FT_TT_OP_GDN_SHAPE] & 0xFFu) == FT_TT_OP_GDN_PHASE_DELTA &&
                             ((words[FT_TT_OP_GDN_SHAPE] >> 8) & 0xFFu) == 4u &&
                             ((words[FT_TT_OP_GDN_SHAPE] >> 16) & 0xFFu) == 32u &&
                             ((words[FT_TT_OP_GDN_SHAPE] >> 24) & 0xFFu) == 16u,
                         "0x%08X", words[FT_TT_OP_GDN_SHAPE]);

    ft_tt_test_check_fmt("a coordinate pair round-trips through one word",
                         (words[FT_TT_OP_GDN_STATE_XY] & 0xFFFFu) == 5u &&
                             ((words[FT_TT_OP_GDN_STATE_XY] >> 16) & 0xFFFFu) == 6u,
                         "0x%08X", words[FT_TT_OP_GDN_STATE_XY]);

    memcpy(&scale, &words[FT_TT_OP_GDN_SCALE_BITS], sizeof(scale));
    /* 1/sqrt(128) = 0.088388347... The bits are compared against the value the CPU oracle
     * multiplies y by, not against a rounded decimal. */
    ft_tt_test_check_fmt("the scale word is the fp32 bits of 1/sqrt(S)",
                         scale > 0.0883883f && scale < 0.0883884f, "%.9f", (double)scale);

    /* The writer publishes the CARRIED STATE as well as y, because the updated state is a
     * RESULT and rule 1 gives results to the writer. This counts CB_OUT PAGES:
     * 32 heads x 4 tile-rows x (4 state tiles + 1 y column) = 640.
     *
     * NOT 4 — that would leave the updated state in L1 and let the next token read a stale
     * recurrence, which a single-step comparison passes and the 48-token gate does not.
     * NOT 516 — y is 4 tiles in DRAM but crosses the buffer as 128 per-head COLUMN tiles,
     * because the (e) reduce produces one column per head and the writer scatters it into a
     * row. Counting the DRAM figure starves the writer by 124 pages and hangs the triad on
     * its bound. */
    ft_tt_test_check_fmt("a DELTA descriptor passes 640 CB_OUT pages, state and y interleaved",
                         words[FT_TT_OP_GDN_TILE_COUNT] == 640u, "%u",
                         words[FT_TT_OP_GDN_TILE_COUNT]);

    /* The words a DELTA descriptor does not use are ZERO, which is what
     * resident_vector_protocol.h requires of every family. */
    ft_tt_test_check_fmt("a DELTA descriptor zeroes the CONV-only words",
                         words[FT_TT_OP_GDN_CONV_TILE_COLS] == 0u &&
                             words[FT_TT_OP_GDN_CONV_TAPS] == 0u &&
                             words[FT_TT_OP_GDN_CONSTANT_WEIGHT] == 0u &&
                             words[FT_TT_OP_GDN_STATE_CONV] == 0u,
                         "%u %u %u %u", words[FT_TT_OP_GDN_CONV_TILE_COLS],
                         words[FT_TT_OP_GDN_CONV_TAPS], words[FT_TT_OP_GDN_CONSTANT_WEIGHT],
                         words[FT_TT_OP_GDN_STATE_CONV]);
    ft_tt_test_check_fmt("the reserved word is 0", words[FT_TT_OP_GDN_FLAGS] == 0u, "%u",
                         words[FT_TT_OP_GDN_FLAGS]);

    fill_conv_request(&request);
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("a CONV request encodes", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("conv_dim/32 needs a whole word and gets one",
                         words[FT_TT_OP_GDN_CONV_TILE_COLS] == 256u, "%u",
                         words[FT_TT_OP_GDN_CONV_TILE_COLS]);
    /* The convolved stream, then the ROLLED WINDOW — taps-1 deep over the same 256 columns,
     * and carried state for the same reason the recurrent matrix is. 256 + 256*3 = 1024. */
    ft_tt_test_check_fmt("a CONV descriptor writes the convolved stream THEN the rolled window",
                         words[FT_TT_OP_GDN_TILE_COUNT] == 1024u, "%u",
                         words[FT_TT_OP_GDN_TILE_COUNT]);
    ft_tt_test_check_fmt("a CONV descriptor zeroes the DELTA-only words",
                         words[FT_TT_OP_GDN_CONSTANT_GATE] == 0u &&
                             words[FT_TT_OP_GDN_STATE_RECURRENT] == 0u,
                         "%u %u", words[FT_TT_OP_GDN_CONSTANT_GATE],
                         words[FT_TT_OP_GDN_STATE_RECURRENT]);
}

static void test_refusals(void)
{
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];

    ft_tt_test_step("the shapes this op REFUSES, one mutation at a time from a request that "
                    "encodes");

    fill_delta_request(&request);
    request.phase = 0u;
    ft_tt_test_check("phase 0 — a zeroed block is invalid, not plausible",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    /* THE ONE THAT MATTERS MOST. The GQA repeat is `kh = vh % k_heads`, which is a repeat
     * only when k_heads divides v_heads. Pantheon's own docs name a real gibberish bug from
     * getting this mapping backwards, so a shape where the modulo is not a repeat has to be
     * refused rather than executed into nonsense. */
    fill_delta_request(&request);
    request.k_heads = 12u;
    ft_tt_test_check("k_heads that does not divide v_heads — the GQA repeat is not a repeat",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    fill_delta_request(&request);
    request.state_size = 100u;
    ft_tt_test_check("a state size that is not a multiple of the tile dimension",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    fill_delta_request(&request);
    request.state_size = 32u * 9u;
    ft_tt_test_check("a state size whose tile count does not fit the shape byte",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    fill_delta_request(&request);
    request.flags = FT_TT_OP_GDN_FLAG_TABLE;
    ft_tt_test_check_fmt("THE TABLE GATES: a DELTA descriptor encodes the flag",
                         ft_tt_op_gdn_parameters(&request, words) == FT_TT_OK &&
                             (words[FT_TT_OP_GDN_FLAGS] & FT_TT_OP_GDN_FLAG_TABLE) != 0u,
                         "flags 0x%x", words[FT_TT_OP_GDN_FLAGS]);
    fill_conv_request(&request);
    request.flags = FT_TT_OP_GDN_FLAG_TABLE;
    ft_tt_test_check("...and a CONV descriptor with it refuses",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    fill_delta_request(&request);
    request.state_recurrent = operand(0u, 0u, 0u, 0u);
    ft_tt_test_check("a DELTA request with no recurrent state operand",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    fill_conv_request(&request);
    request.conv_taps = 1u;
    ft_tt_test_check("a one-tap conv — the carried window would be write-only",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    fill_conv_request(&request);
    request.state_conv = operand(0u, 0u, 0u, 0u);
    ft_tt_test_check("a CONV request with no window operand",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    ft_tt_test_check("a null request", ft_tt_op_gdn_parameters(NULL, words) == FT_TT_EINVAL,
                     NULL);
    fill_delta_request(&request);
    ft_tt_test_check("a null output", ft_tt_op_gdn_parameters(&request, NULL) == FT_TT_EINVAL,
                     NULL);
}

static void test_cb_placement(void)
{
    ft_tt_op_gdn_request request;
    ft_tt_launch_cb_entry entries[FT_TT_OP_GDN_CB_COUNT];
    uint64_t mask = 0u;
    unsigned index;
    int rc;

    ft_tt_test_step("circular-buffer placement fits the arena and names every buffer once");

    fill_delta_request(&request);
    rc = ft_tt_op_gdn_place_cbs(&request, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("a DELTA request places its CBs", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;

    for (index = 0; index < FT_TT_OP_GDN_CB_COUNT; index++) {
        char label[96];

        snprintf(label, sizeof(label), "CB %u sits inside the arena", entries[index].index);
        ft_tt_test_check_fmt(label,
                             entries[index].address >= FT_TT_OP_CB_ARENA_BASE &&
                                 entries[index].address < FT_TT_OP_CB_ARENA_END,
                             "0x%08X", entries[index].address);
    }

    ft_tt_test_check_fmt("the mask names exactly the eleven buffers placed",
                         mask == ((1ull << FT_TT_OP_GDN_CB_IN) | (1ull << FT_TT_OP_GDN_CB_STATE) |
                                  (1ull << FT_TT_OP_GDN_CB_SCALER) |
                                  (1ull << FT_TT_OP_GDN_CB_VECTOR) |
                                  (1ull << FT_TT_OP_GDN_CB_STAGE) |
                                  (1ull << FT_TT_OP_GDN_CB_PRED) |
                                  (1ull << FT_TT_OP_GDN_CB_DELTA) |
                                  (1ull << FT_TT_OP_GDN_CB_PROD) |
                                  (1ull << FT_TT_OP_GDN_CB_WORK) |
                                  (1ull << FT_TT_OP_GDN_CB_OUT) |
                                  (1ull << FT_TT_OP_GDN_CB_CARRY)),
                         "0x%016llX", (unsigned long long)mask);

    /* THE ONE-PRODUCER RULE, AS AN INDEX TEST. PHASE CONV's qkv stream is pushed by the
     * READER and PHASE DELTA's working buffer by the COMPUTE thread, and the two push
     * primitives keep incompatible books: the dataflow one INCREMENTS the tiles_received
     * stream register, the compute one STORES a local counter over it. Sharing one index
     * therefore makes `cb_wait_front` on that buffer stop blocking after the first DELTA
     * descriptor — measured on silicon 2026-08-18 as PHASE CONV's second pass copying stale
     * pages, `s = t (mod 8)` at all 256 tiles. Nothing else in this suite would notice the
     * two constants being equal again. */
    ft_tt_test_check_fmt("CB_IN and CB_WORK are DIFFERENT indices — one producer per buffer",
                         FT_TT_OP_GDN_CB_IN != FT_TT_OP_GDN_CB_WORK, "in=%u work=%u",
                         FT_TT_OP_GDN_CB_IN, FT_TT_OP_GDN_CB_WORK);

    fill_conv_request(&request);
    rc = ft_tt_op_gdn_place_cbs(&request, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("a CONV request places the SAME eleven buffers — one binary, one "
                         "layout, two phases",
                         rc == FT_TT_OK, "rc=%d", rc);
}

/* ================================================================
 * The CB placement, against the refusals the DEVICE makes
 * ================================================================ */

/**
 * The device's own guards, transliterated.
 *
 * `delta_is_executable()` and `conv_is_executable()` in `kernels/resident_gdn_reader.c` refuse
 * a descriptor the placed buffers cannot serve, and `conv_block_depth()` refuses a block depth
 * they cannot hold. NOTHING made the host's placement satisfy them — the two were written a
 * commit apart and agreed only at Ornith's shape, by arithmetic coincidence
 * (2 * state_tiles = 8 = 2 * conv_depth * taps, and state_tiles + 3 = 7 <= 8). This is the
 * check that would have said so.
 *
 * `ft_tensix_cb_push_grid_is_safe()` is reduced here to its page-count half — "chunk <= pages
 * and pages % chunk == 0". Its third clause tests a live FIFO pointer, which is a runtime fact
 * no host placement can decide; the two static clauses are the ones a placement owns.
 */
static int placement_satisfies_device(const ft_tt_launch_cb_entry *entries, uint32_t state_tiles,
                                      uint32_t taps, const char **out_reason)
{
    const uint32_t depth = FT_TT_OP_GDN_CONV_DEPTH;
    const ft_tt_launch_cb_entry *in =
        ft_tt_op_cb_find(entries, FT_TT_OP_GDN_CB_COUNT, FT_TT_OP_GDN_CB_IN);
    const ft_tt_launch_cb_entry *state =
        ft_tt_op_cb_find(entries, FT_TT_OP_GDN_CB_COUNT, FT_TT_OP_GDN_CB_STATE);
    const ft_tt_launch_cb_entry *scaler =
        ft_tt_op_cb_find(entries, FT_TT_OP_GDN_CB_COUNT, FT_TT_OP_GDN_CB_SCALER);
    const ft_tt_launch_cb_entry *vector =
        ft_tt_op_cb_find(entries, FT_TT_OP_GDN_CB_COUNT, FT_TT_OP_GDN_CB_VECTOR);
    const ft_tt_launch_cb_entry *stage =
        ft_tt_op_cb_find(entries, FT_TT_OP_GDN_CB_COUNT, FT_TT_OP_GDN_CB_STAGE);
    const ft_tt_launch_cb_entry *work =
        ft_tt_op_cb_find(entries, FT_TT_OP_GDN_CB_COUNT, FT_TT_OP_GDN_CB_WORK);
    unsigned index;

    *out_reason = "";
    if (!in || !state || !scaler || !vector || !stage || !work) {
        *out_reason = "a buffer the kernels name was not placed";
        return 0;
    }
    /* Every page is an fp32 tile — the CB table says so for all ten. */
    for (index = 0; index < FT_TT_OP_GDN_CB_COUNT; index++) {
        if (entries[index].page_size_bytes != FT_TT_OP_GDN_FP32_TILE_BYTES) {
            *out_reason = "a page is not an fp32 tile";
            return 0;
        }
    }

    /* delta_is_executable() */
    if (scaler->page_count < 2u * state_tiles) {
        *out_reason = "CB_SCALER cannot hold this head's k and q replicas";
        return 0;
    }
    if (vector->page_count < state_tiles + FT_TT_OP_GDN_VECTOR_CONSTANTS) {
        *out_reason = "CB_VECTOR cannot hold the v columns plus decay/beta/scale";
        return 0;
    }
    if (stage->page_count < state_tiles + 1u) {
        *out_reason = "CB_STAGE cannot hold a row block plus the gate page";
        return 0;
    }
    if (work->page_count < 2u * state_tiles) {
        *out_reason = "CB_WORK cannot hold the decayed tile-row AND the updated one";
        return 0;
    }
    if (state->page_count < state_tiles || (state->page_count % state_tiles) != 0u) {
        *out_reason = "CB_STATE is off the DELTA tile-row push grid";
        return 0;
    }

    /* conv_block_depth() and conv_is_executable() */
    if (in->page_count < 2u * depth || (in->page_count % depth) != 0u) {
        *out_reason = "CB_IN is off the CONV block grid";
        return 0;
    }
    if (vector->page_count < 2u * depth * taps || (vector->page_count % (depth * taps)) != 0u) {
        *out_reason = "CB_VECTOR is off the CONV weight-plane grid";
        return 0;
    }
    if (state->page_count < 2u * depth * (taps - 1u) || (state->page_count % depth) != 0u) {
        *out_reason = "CB_STATE cannot hold the CONV window";
        return 0;
    }

    /* ---------------------------------------------------------------------------
     * THE CONSUMER'S INDEXED-READ GRID — the half of the rule this suite did not have.
     *
     * `ft_tensix_unpack_ab(cb, ..., tile)` resolves to `read_pointer + tile * page_size`, and
     * a FIFO pointer wraps on an EXACT HIT of the limit and on nothing else. So a compute
     * thread that waits for `n` pages and then names tiles `0..n-1` needs the front low
     * enough that the whole window still fits, and the front is a multiple of `n` only when
     * `n` divides the buffer. Everything above tests CAPACITY and the PRODUCER's push grid;
     * neither of those sees this.
     *
     * MEASURED ON SILICON 2026-08-18, and both were wrong at ORNITH'S OWN SHAPE. CB_VECTOR
     * held 8 pages against a DELTA window of `state_tiles + 3` = 7, so the front walked
     * 0, 7, 6, 5, ... and 24 of 32 heads read their v / decay / beta / scale out of the
     * buffer and into CB_STAGE — exactly the 8 heads whose front was 0 or 1 scored 4e-3
     * against the closed form and the rest scored 1.0. CB_STATE held 8 against a CONV window
     * of `depth*(taps-1)` = 3, so 2 blocks in 8 read past the end; that one is invisible at
     * layer 0 of token 0 only, because the L1 above CB_STATE is CB_SCALER and it is still
     * zero until the first DELTA descriptor writes k/q replicas into it.
     * --------------------------------------------------------------------------- */
    if ((work->page_count % state_tiles) != 0u) {
        *out_reason = "CB_WORK is off the DELTA indexed-read grid (a whole state tile-row)";
        return 0;
    }
    /* THE BATCHED (d)'s OWN GRID. It stages `dst_batch` rank-1 products in CB_PROD and then
     * indexes every one of them off the front, and it rewrites CB_WORK a block at a time.
     * CB_PROD had NO grid before the batch — every access to it named tile 0 — so nothing
     * above this line looks at it, and `resolve_cb_pages()` handed it a flat 2 pages whatever
     * the shape. `delta_is_executable()` in resident_gdn_reader.c refuses the same two
     * clauses on the device; this is the host side of that pair. */
    {
        const uint32_t dst_batch = ftg_dst_batch(state_tiles, FT_TT_OP_GDN_DST_BATCH);
        const ft_tt_launch_cb_entry *prod =
            ft_tt_op_cb_find(entries, FT_TT_OP_GDN_CB_COUNT, FT_TT_OP_GDN_CB_PROD);

        if (!prod) {
            *out_reason = "CB_PROD was not placed";
            return 0;
        }
        if (prod->page_count < 2u * dst_batch) {
            *out_reason = "CB_PROD cannot hold two DST batches of rank-1 products";
            return 0;
        }
        if ((prod->page_count % dst_batch) != 0u) {
            *out_reason = "CB_PROD is off the DELTA DST-batch indexed-read grid";
            return 0;
        }
        if ((work->page_count % dst_batch) != 0u) {
            *out_reason = "CB_WORK is off the DELTA DST-batch rotation grid";
            return 0;
        }
    }
    if ((state->page_count % state_tiles) != 0u) {
        *out_reason = "CB_STATE is off the DELTA indexed-read grid";
        return 0;
    }
    if ((scaler->page_count % (2u * state_tiles)) != 0u) {
        *out_reason = "CB_SCALER is off the DELTA indexed-read grid (k AND q replicas)";
        return 0;
    }
    if ((vector->page_count % (state_tiles + FT_TT_OP_GDN_VECTOR_CONSTANTS)) != 0u) {
        *out_reason = "CB_VECTOR is off the DELTA indexed-read grid (the v columns plus the "
                      "three per-head constants)";
        return 0;
    }
    if ((state->page_count % (depth * (taps - 1u))) != 0u) {
        *out_reason = "CB_STATE is off the CONV indexed-read grid (every tap of a column)";
        return 0;
    }
    if ((vector->page_count % (depth * taps)) != 0u) {
        *out_reason = "CB_VECTOR is off the CONV indexed-read grid (every weight plane)";
        return 0;
    }
    /* THE WRITER'S BARRIER GROUP. CB_OUT has no grid on either side — the producer pushes a
     * page at a time, and the writer addresses its sources with the wrap SPELLED OUT and pops
     * one page at a time — so this clause is purely CAPACITY, and it is the capacity the
     * writer's own clamp reads back off the placed buffer. Two groups: one draining behind a
     * barrier, one filling. A placement short of it does not fail on the card; it silently
     * demotes the writer to the per-tile sequence, which is a slow build nothing refuses. */
    {
        const ft_tt_launch_cb_entry *out =
            ft_tt_op_cb_find(entries, FT_TT_OP_GDN_CB_COUNT, FT_TT_OP_GDN_CB_OUT);

        if (!out) {
            *out_reason = "CB_OUT was not placed";
            return 0;
        }
        if (out->page_count < 2u * ftg_writer_block(state_tiles, FT_TT_OP_GDN_WRITER_RUNG)) {
            *out_reason = "CB_OUT cannot hold two of the WRITER's barrier groups";
            return 0;
        }
    }
    return 1;
}

static void test_cb_placement_serves_the_device(void)
{
    ft_tt_op_gdn_request request;
    ft_tt_launch_cb_entry entries[FT_TT_OP_GDN_CB_COUNT];
    uint64_t mask = 0u;
    uint32_t state_tiles;
    uint32_t taps;

    ft_tt_test_step("the placement satisfies every refusal the READER makes, at EVERY shape "
                    "ft_tt_op_gdn_shape_is_supported() admits");

    for (state_tiles = 1u; state_tiles <= 8u; state_tiles++) {
        for (taps = 2u; taps <= 8u; taps++) {
            const char *reason = "";
            char label[160];
            int rc;

            fill_delta_request(&request);
            request.state_size = state_tiles * FT_TT_OP_TILE_DIM;
            request.conv_taps = taps;
            rc = ft_tt_op_gdn_place_cbs(&request, entries, g_cb_buffer, sizeof(g_cb_buffer),
                                        &mask);
            snprintf(label, sizeof(label), "S=%u (%u state tiles), %u taps — placed and "
                                           "executable",
                     state_tiles * 32u, state_tiles, taps);
            ft_tt_test_check_fmt(
                label,
                rc == FT_TT_OK &&
                    placement_satisfies_device(entries, state_tiles, taps, &reason),
                "rc=%d %s", rc, reason);
        }
    }
}

static void test_tile_format(void)
{
    ft_tt_op_gdn_request request;
    ft_tt_launch_cb_entry entries[FT_TT_OP_GDN_CB_COUNT];
    uint32_t words[FT_TT_OP_GDN_WORDS];
    uint64_t mask = 0u;
    unsigned index;

    ft_tt_test_step("every operand and every circular-buffer page is an fp32 tile — the family "
                    "has exactly one format and one tile size");

    fill_delta_request(&request);
    (void)ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("a DELTA descriptor publishes 4096 as its tile size",
                         words[FT_TT_OP_GDN_TILE_BYTES] == FT_TT_OP_GDN_FP32_TILE_BYTES, "%u",
                         words[FT_TT_OP_GDN_TILE_BYTES]);

    fill_conv_request(&request);
    (void)ft_tt_op_gdn_parameters(&request, words);
    /* CONV took the word from the ACTIVATION and DELTA from the STATE, so the two phases of
     * one binary could publish different page sizes for the same ten buffers. */
    ft_tt_test_check_fmt("a CONV descriptor publishes the SAME tile size, not its own "
                         "operand's",
                         words[FT_TT_OP_GDN_TILE_BYTES] == FT_TT_OP_GDN_FP32_TILE_BYTES, "%u",
                         words[FT_TT_OP_GDN_TILE_BYTES]);

    /* THE ONE THAT WOULD HAVE CORRUPTED L1 ON THE FIRST RUN. The kernels compare
     * kGdnTileBytes against their compiled-in 4096, but that word comes from ONE of five
     * endpoints, so a bf16 activation beside an fp32 state passed the device's check and then
     * had 2048-byte pages placed under 4096-byte tile reads. */
    fill_delta_request(&request);
    request.activation_in.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    ft_tt_test_check("a bf16 activation beside an fp32 state — the device's one tile-size word "
                     "cannot see it, so the host refuses it",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    fill_conv_request(&request);
    request.conv_weight.tile_bytes = FT_TT_OP_TILE_BYTES_BFP8_B;
    ft_tt_test_check("a Bfp8_b conv weight — a block-float page cannot hold an fp32 tile",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    /* THE ONE THAT DID CORRUPT DRAM ON THE FIRST SILICON RUN, and the device could not see it
     * either: `kGdnActivationXy` is ONE packed coordinate for the input AND the output, so a
     * stager that puts them on different DRAM endpoints makes the writer land the answer at
     * the output's offset INSIDE THE INPUT'S BANK. The descriptor retires clean, the counters
     * are right, and the output buffer is never written. Both phases carry the pair. */
    fill_conv_request(&request);
    request.activation_out.x = (uint16_t)(request.activation_in.x + 1u);
    ft_tt_test_check("a CONV output on a different DRAM x from its input — one packed xy "
                     "cannot address two banks",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    fill_delta_request(&request);
    request.activation_out.y = (uint16_t)(request.activation_in.y + 1u);
    ft_tt_test_check("a DELTA output on a different DRAM y from its input",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_EINVAL, NULL);

    fill_delta_request(&request);
    ft_tt_test_check("the activation pair on ONE endpoint is still accepted",
                     ft_tt_op_gdn_parameters(&request, words) == FT_TT_OK, NULL);

    fill_delta_request(&request);
    (void)ft_tt_op_gdn_place_cbs(&request, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    for (index = 0; index < FT_TT_OP_GDN_CB_COUNT; index++) {
        char label[96];

        snprintf(label, sizeof(label), "CB %u holds fp32 pages", entries[index].index);
        ft_tt_test_check_fmt(label, entries[index].page_size_bytes == FT_TT_OP_GDN_FP32_TILE_BYTES,
                             "%u B", entries[index].page_size_bytes);
    }
}

/* ================================================================
 * The DRAM layouts the reader and the writer address
 * ================================================================ */

/**
 * The layout arithmetic, against literals worked out from the ORACLE rather than from the
 * accessors.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS ONE INCLUDES THE DEVICE HEADER INSTEAD OF PARSING IT
 * ---------------------------------------------------------------------------
 * The wire CONSTANTS are declared twice — once for the host and once for the kernels — so a
 * suite that included the device header would compile against the same declaration the kernel
 * does and could not see the two diverge. That is why `test_wire_constants()` parses.
 *
 * The layout ACCESSORS are declared ONCE, in `resident_gdn_protocol.h`, and both dataflow
 * kernels call them. There is no second declaration to drift from — the drift risk is between
 * the accessor and the layout a future stager writes, and the only way to test that is to
 * compute the expected tile index INDEPENDENTLY here and compare. Every literal below is
 * derived from `pn_qwen35moe_ssm_recurrence()` and `ft_tt_tilize_strip()`, not from the
 * function it checks.
 */
static ftg_shape ornith_shape(uint32_t phase)
{
    ftg_shape shape;

    shape.phase = phase;
    shape.state_tiles = FT_TT_OP_GDN_QWEN35_STATE / FT_TT_OP_TILE_DIM;   /* 4  */
    shape.v_heads = FT_TT_OP_GDN_QWEN35_V_HEADS;                         /* 32 */
    shape.k_heads = FT_TT_OP_GDN_QWEN35_K_HEADS;                         /* 16 */
    shape.conv_tile_cols = FT_TT_OP_GDN_QWEN35_CONV_DIM / FT_TT_OP_TILE_DIM; /* 256 */
    shape.conv_taps = FT_TT_OP_GDN_QWEN35_CONV_TAPS;                     /* 4  */
    shape.tile_bytes = FT_TT_OP_GDN_FP32_TILE_BYTES;
    shape.tile_count = 640u;
    return shape;
}

static void test_delta_input_layout(void)
{
    ftg_shape shape = ornith_shape(FT_TT_OP_GDN_PHASE_DELTA);
    ftg_shape wide = shape;

    uint32_t head;
    int strided = 1;
    int block_starts = 1;

    ft_tt_test_step("the DELTA q|k|v layout at kGdnActivationIn — ONE head-major ROW-0 STRIP, "
                    "q and k per K-HEAD");

    /* q and k carry 16 k-heads of `state_tiles` CONSECUTIVE row-0 tiles each — 64 tiles. v
     * carries 32 v-heads the same way, 128. 256 tiles in all, which is the tilized
     * `[1, (2*k_heads + v_heads)*S]` strip PHASE CONV's output already is. */
    ft_tt_test_check_fmt("q is k_heads x state_tiles = 64 row-0 tiles",
                         ftg_delta_stream_tiles(&shape, kGdnStreamQ) == 64u &&
                             ftg_delta_stream_tiles(&shape, kGdnStreamK) == 64u,
                         "%u %u", ftg_delta_stream_tiles(&shape, kGdnStreamQ),
                         ftg_delta_stream_tiles(&shape, kGdnStreamK));
    ft_tt_test_check_fmt("v is v_heads x state_tiles = 128 row-0 tiles",
                         ftg_delta_stream_tiles(&shape, kGdnStreamV) == 128u, "%u",
                         ftg_delta_stream_tiles(&shape, kGdnStreamV));
    ft_tt_test_check_fmt("q | k | v sit at tiles 0, 64 and 128",
                         ftg_delta_stream_base(&shape, kGdnStreamQ) == 0u &&
                             ftg_delta_stream_base(&shape, kGdnStreamK) == 64u &&
                             ftg_delta_stream_base(&shape, kGdnStreamV) == 128u,
                         "%u %u %u", ftg_delta_stream_base(&shape, kGdnStreamQ),
                         ftg_delta_stream_base(&shape, kGdnStreamK),
                         ftg_delta_stream_base(&shape, kGdnStreamV));
    ft_tt_test_check_fmt("the whole staged input is 256 tiles",
                         ftg_delta_input_tiles(&shape) == 256u, "%u",
                         ftg_delta_input_tiles(&shape));

    /* THE STRIDE IS THE GEOMETRY. Consecutive heads are exactly `state_tiles` apart in every
     * block, head 0 of a block IS the block's base, and head 31's v block ENDS on the last
     * tile of the whole input — so the three blocks tile the buffer with no hole and no
     * overlap. Under the old one-row-per-head packing every head of a row block returned the
     * SAME tile and this stride was 0, so the arm is red against it at head 1. */
    for (head = 0; head + 1u < shape.v_heads; head++) {
        if (ftg_delta_head_tile(&shape, kGdnStreamV, head + 1u) -
                ftg_delta_head_tile(&shape, kGdnStreamV, head) !=
            shape.state_tiles)
            strided = 0;
    }
    for (head = 0; head < shape.k_heads; head++) {
        if (ftg_delta_head_tile(&shape, kGdnStreamQ, head) != head * shape.state_tiles ||
            ftg_delta_head_tile(&shape, kGdnStreamK, head) != 64u + head * shape.state_tiles)
            strided = 0;
    }
    ft_tt_test_check_fmt("consecutive heads are state_tiles apart in every block — the stride "
                         "the old one-row-per-head packing made 0",
                         strided && ftg_delta_head_tile(&shape, kGdnStreamV, 1u) -
                                            ftg_delta_head_tile(&shape, kGdnStreamV, 0u) ==
                                        4u,
                         "%u %u", ftg_delta_head_tile(&shape, kGdnStreamV, 0u),
                         ftg_delta_head_tile(&shape, kGdnStreamV, 1u));
    ft_tt_test_check_fmt("head 0 of q, k and v IS that stream's base",
                         ftg_delta_head_tile(&shape, kGdnStreamQ, 0u) ==
                                 ftg_delta_stream_base(&shape, kGdnStreamQ) &&
                             ftg_delta_head_tile(&shape, kGdnStreamK, 0u) ==
                                 ftg_delta_stream_base(&shape, kGdnStreamK) &&
                             ftg_delta_head_tile(&shape, kGdnStreamV, 0u) ==
                                 ftg_delta_stream_base(&shape, kGdnStreamV),
                         "%u %u %u", ftg_delta_head_tile(&shape, kGdnStreamQ, 0u),
                         ftg_delta_head_tile(&shape, kGdnStreamK, 0u),
                         ftg_delta_head_tile(&shape, kGdnStreamV, 0u));
    ft_tt_test_check_fmt("v-head 31's block ENDS exactly at the input's last tile — no hole, "
                         "no overlap, nothing off the end",
                         ftg_delta_head_tile(&shape, kGdnStreamV, shape.v_heads - 1u) +
                                 shape.state_tiles ==
                             ftg_delta_input_tiles(&shape),
                         "%u vs %u",
                         ftg_delta_head_tile(&shape, kGdnStreamV, shape.v_heads - 1u) +
                             shape.state_tiles,
                         ftg_delta_input_tiles(&shape));
    /* And the k block's last head ends exactly where the v block begins. */
    ft_tt_test_check_fmt("k-head 15's block ends exactly where v begins",
                         ftg_delta_head_tile(&shape, kGdnStreamK, shape.k_heads - 1u) +
                                 shape.state_tiles ==
                             ftg_delta_stream_base(&shape, kGdnStreamV),
                         "%u vs %u",
                         ftg_delta_head_tile(&shape, kGdnStreamK, shape.k_heads - 1u) +
                             shape.state_tiles,
                         ftg_delta_stream_base(&shape, kGdnStreamV));
    /* The row inside every one of those tiles is 0 and nothing reads rows 1-31. */
    ft_tt_test_check_fmt("a strip's data is in ROW 0 of every tile — it used to be head % 32",
                         (uint32_t)kGdnStripRow == 0u, "%u", (uint32_t)kGdnStripRow);

    /* q and k are per K-HEAD and v per V-HEAD, so the three blocks are NOT the same size
     * whenever the GQA group is not 1. 64 v-heads over 16 k-heads: q and k are 64 tiles each,
     * v is 256, and the input is 384. The bases do not move, because they are the k blocks. */
    wide.v_heads = 64u;
    ft_tt_test_check_fmt("64 v-heads over 16 k-heads — v is 64 x 4 = 256 tiles",
                         ftg_delta_stream_tiles(&wide, kGdnStreamV) == 256u, "%u",
                         ftg_delta_stream_tiles(&wide, kGdnStreamV));
    ft_tt_test_check_fmt("and the whole input is 384 tiles, not 256",
                         ftg_delta_input_tiles(&wide) == 384u &&
                             ftg_delta_stream_base(&wide, kGdnStreamV) == 128u,
                         "%u %u", ftg_delta_input_tiles(&wide),
                         ftg_delta_stream_base(&wide, kGdnStreamV));

    /* And the mirror case: more k-heads than v-heads cannot happen (the GQA repeat needs
     * k_heads to divide v_heads), but 64 of each must still give three equal blocks of 256. */
    wide.k_heads = 64u;
    ft_tt_test_check_fmt("64 of each — three equal blocks of 256, bases 0, 256, 512",
                         ftg_delta_stream_base(&wide, kGdnStreamQ) == 0u &&
                             ftg_delta_stream_base(&wide, kGdnStreamK) == 256u &&
                             ftg_delta_stream_base(&wide, kGdnStreamV) == 512u &&
                             ftg_delta_input_tiles(&wide) == 768u,
                         "%u %u %u total=%u", ftg_delta_stream_base(&wide, kGdnStreamQ),
                         ftg_delta_stream_base(&wide, kGdnStreamK),
                         ftg_delta_stream_base(&wide, kGdnStreamV),
                         ftg_delta_input_tiles(&wide));
    for (head = 0; head < wide.v_heads; head++) {
        if (ftg_delta_head_tile(&wide, kGdnStreamV, head) !=
            ftg_delta_stream_base(&wide, kGdnStreamV) + head * wide.state_tiles)
            block_starts = 0;
    }
    ft_tt_test_check_fmt("every one of the 64 v-heads starts at its base + h*state_tiles, "
                         "including the ones past 32 that a row block could not hold",
                         block_starts && ftg_delta_head_tile(&wide, kGdnStreamV, 32u) == 640u,
                         "%u", ftg_delta_head_tile(&wide, kGdnStreamV, 32u));

    /* The GQA repeat is INTERLEAVED. v-head 17 reads k-head 1, not k-head 8. */
    ft_tt_test_check_fmt("the GQA repeat is vh mod k_heads, not vh / kv_groups",
                         ftg_k_head_for(17u, 16u) == 1u && ftg_k_head_for(31u, 16u) == 15u,
                         "%u %u", ftg_k_head_for(17u, 16u), ftg_k_head_for(31u, 16u));
    /* And the repeat is now an ADDRESS in the k strip: v-head 17 reads tiles [4, 8). */
    ft_tt_test_check_fmt("v-head 17's k tiles are k-head 1's four, at 64 + 4 = 68",
                         ftg_delta_head_tile(&shape, kGdnStreamK,
                                             ftg_k_head_for(17u, shape.k_heads)) == 68u,
                         "%u",
                         ftg_delta_head_tile(&shape, kGdnStreamK,
                                             ftg_k_head_for(17u, shape.k_heads)));
}

static void test_state_and_output_layout(void)
{
    ftg_shape shape = ornith_shape(FT_TT_OP_GDN_PHASE_DELTA);

    ft_tt_test_step("the carried state, y and the CONV planes — the layouts the WRITER "
                    "addresses");

    /* [v_heads, S, S] fp32, head-major, each head tilized into 4x4 tiles row-major.
     * Head 1's tile-row 2 therefore starts at 1*16 + 2*4 = 24. */
    ft_tt_test_check_fmt("head 1, tile-row 2, tile-column 0 is state tile 24",
                         ftg_state_tile(&shape, 1u, 2u, 0u) == 24u, "%u",
                         ftg_state_tile(&shape, 1u, 2u, 0u));
    ft_tt_test_check_fmt("a head owns state_tiles^2 = 16 tiles",
                         ftg_state_tiles_per_head(&shape) == 16u, "%u",
                         ftg_state_tiles_per_head(&shape));

    /* y is a HEAD-MAJOR ROW-0 STRIP: v-head h's S values are ROW 0 of tiles
     * [h*state_tiles, +state_tiles), so Ornith's y is 32*4 = 128 tiles and head 31 owns
     * 124..127 outright. The gated RMS norm that follows is then a `row_blocks = v_heads,
     * tile_cols = state_tiles` reduce over the strip — the same shape of reduce, over a strip
     * instead of a matrix. */
    ft_tt_test_check_fmt("y is 128 tiles at Ornith, and head 31 owns 124..127 alone",
                         ftg_y_tile(&shape, 0u, 0u) == 0u && ftg_y_tile(&shape, 0u, 3u) == 3u &&
                             ftg_y_tile(&shape, 31u, 0u) == 124u &&
                             ftg_y_tile(&shape, 31u, 3u) == 127u,
                         "%u %u %u %u", ftg_y_tile(&shape, 0u, 0u), ftg_y_tile(&shape, 0u, 3u),
                         ftg_y_tile(&shape, 31u, 0u), ftg_y_tile(&shape, 31u, 3u));

    /* AND IT IS INJECTIVE. This is the arm that is RED against the geometry that shipped: the
     * old `ftg_y_tile()` collapsed every head of a row block onto the same `state_tiles`
     * tiles and told them apart by ROW, so (head, tile_row) -> tile was 32-to-1 and the very
     * first collision below fires at heads 0 and 1. A strip has no row to hide in, so a
     * non-injective y is two heads writing the same DRAM tile. */
    {
        unsigned char seen[128 * 4];
        uint32_t head, row;
        uint32_t collisions = 0u;
        uint32_t highest = 0u;

        memset(seen, 0, sizeof(seen));
        for (head = 0; head < shape.v_heads; head++) {
            for (row = 0; row < shape.state_tiles; row++) {
                const uint32_t tile = ftg_y_tile(&shape, head, row);

                if (tile >= sizeof(seen)) {
                    collisions++;
                    continue;
                }
                if (seen[tile])
                    collisions++;
                seen[tile] = 1u;
                if (tile + 1u > highest)
                    highest = tile + 1u;
            }
        }
        ft_tt_test_check_fmt("(head, tile_row) -> y tile is INJECTIVE over all 32 x 4, and the "
                             "128 tiles it reaches are exactly v_heads * state_tiles",
                             collisions == 0u && highest == shape.v_heads * shape.state_tiles,
                             "collisions=%u highest=%u", collisions, highest);
    }

    /* A 64-head model simply keeps going: head 32 is at 128, not back at the top of a second
     * row block, and the buffer is twice as long. */
    {
        ftg_shape wide = shape;

        wide.v_heads = 64u;
        ft_tt_test_check_fmt("head 32 of a 64-head model is at y tile 128, and head 63 ends "
                             "the 256-tile strip",
                             ftg_y_tile(&wide, 32u, 0u) == 128u &&
                                 ftg_y_tile(&wide, 63u, 3u) + 1u == 256u,
                             "%u %u", ftg_y_tile(&wide, 32u, 0u), ftg_y_tile(&wide, 63u, 3u));
    }

    /* y and the DELTA input are the SAME geometry, which is the whole reason for the change:
     * head h sits at h*state_tiles in both, so PHASE CONV's output strip IS PHASE DELTA's
     * input with no copy and no host round trip. */
    ft_tt_test_check_fmt("y's head stride and the v stream's head stride are the same "
                         "arithmetic — conv_out feeds DELTA with no re-tilization",
                         ftg_y_tile(&shape, 7u, 2u) ==
                             ftg_delta_head_tile(&shape, kGdnStreamV, 7u) + 2u -
                                 ftg_delta_stream_base(&shape, kGdnStreamV),
                         "%u %u", ftg_y_tile(&shape, 7u, 2u),
                         ftg_delta_head_tile(&shape, kGdnStreamV, 7u) + 2u -
                             ftg_delta_stream_base(&shape, kGdnStreamV));

    /* The CONV weight and window are TAP-MAJOR: plane t is `conv_tile_cols` tiles. Tap 3,
     * column 5 is 3*256 + 5. The oracle's own weight is row-major [conv_dim, taps], so a
     * stager that reshapes rather than transposes lands here. */
    ft_tt_test_check_fmt("conv tap 3, column 5 is plane tile 773",
                         ftg_conv_plane_tile(&shape, 3u, 5u) == 3u * 256u + 5u, "%u",
                         ftg_conv_plane_tile(&shape, 3u, 5u));
    ft_tt_test_check_fmt("tap 0 is the first plane, so column 5 of it is tile 5",
                         ftg_conv_plane_tile(&shape, 0u, 5u) == 5u, "%u",
                         ftg_conv_plane_tile(&shape, 0u, 5u));
}

/**
 * What ONE layer costs in card DRAM — the six operand sizes a host stager allocates.
 *
 * ---------------------------------------------------------------------------
 * THE 50.25 MiB THIS ARM USED TO ASSERT WAS A CAPACITY FIGURE READ AS A BUFFER SIZE
 * ---------------------------------------------------------------------------
 * It decomposed as 48 MiB of recurrent state plus 2.25 MiB of conv window, and the second
 * term is `conv_dim * (taps-1) * 4` — the ELEMENT count. The window the reader addresses is
 * `(taps-1)` TAP-MAJOR PLANES of `conv_dim/32` fp32 TILES, and a plane is a ONE-ROW tilized
 * strip whose other 31 rows are the tile quantum. So the DRAM figure is 32x the element one:
 * 3 MiB a layer, not 96 KiB. Allocating the old number would not have been 2.25 MiB of
 * under-reserve, it would have been every window read past plane 0 landing in whatever
 * `ft_tt_buffer_alloc` put above it.
 *
 * Every literal below is worked out from the layout section of `resident_gdn_protocol.h`
 * and Ornith's shape, not from the functions being checked.
 */
static void test_capacity(void)
{
    ft_tt_op_gdn_request request;
    const size_t mib = 1024u * 1024u;
    size_t state, window, weight, stream, input, output, gate;

    ft_tt_test_step("what one layer costs in card DRAM — every operand a stager allocates");

    fill_conv_request(&request);
    request.v_heads = FT_TT_OP_GDN_QWEN35_V_HEADS;
    request.k_heads = FT_TT_OP_GDN_QWEN35_K_HEADS;
    state = ft_tt_op_gdn_state_bytes(&request);
    window = ft_tt_op_gdn_conv_state_bytes(&request);
    weight = ft_tt_op_gdn_conv_weight_bytes(&request);
    stream = ft_tt_op_gdn_conv_stream_bytes(&request);
    input = ft_tt_op_gdn_delta_input_bytes(&request);
    output = ft_tt_op_gdn_delta_output_bytes(&request);
    gate = ft_tt_op_gdn_gate_bytes(&request);

    /* 32 heads x (128/32)^2 = 512 tiles x 4096 B = 2 MiB — and 32*128*128*4 is the same
     * number, because [S,S] is a whole number of tiles with no padding. */
    ft_tt_test_check_fmt("a layer's recurrent state is 2 MiB", state == 2u * mib, "%zu bytes",
                         state);
    /* 3 window planes x 256 tiles x 4096 B. The element count, 8192*3*4, is 96 KiB — the
     * trap this arm exists to pin. */
    ft_tt_test_check_fmt("a layer's conv WINDOW is 3 MiB of tiles, not 96 KiB of elements",
                         window == 3u * mib, "%zu bytes", window);
    /* 4 weight planes x 256 tiles x 4096 B. */
    ft_tt_test_check_fmt("a layer's conv WEIGHT is 4 MiB", weight == 4u * mib, "%zu bytes",
                         weight);
    /* One row block of 256 tiles: qkv in, conv_out out. */
    ft_tt_test_check_fmt("one conv stream is 1 MiB", stream == mib, "%zu bytes", stream);
    /* q|k|v is a ROW-0 STRIP now: 16*4 + 16*4 + 32*4 = 256 tiles, 1 MiB — the same size as the
     * conv stream it IS, since q, k and v are channels of conv_out. It was 12 tiles / 48 KiB
     * as three one-row-per-head matrices; the strip is 32x bigger in DRAM and that is the
     * price of not de-tilizing and re-tilizing 49,408 B a layer over PCIe. */
    ft_tt_test_check_fmt("the staged q|k|v is 256 row-0 tiles = 1 MiB, the same buffer "
                         "conv_out already is",
                         input == mib && input == stream, "%zu bytes", input);
    /* y is v_heads * state_tiles = 128 tiles, 512 KiB — each v-head owns four outright. */
    ft_tt_test_check_fmt("y is 128 tiles = 512 KiB", output == 512u * 1024u, "%zu bytes",
                         output);
    /* 2 x 32 fp32 = 256 B, already a multiple of the 64 B read granule. */
    ft_tt_test_check_fmt("the gate block is 256 B", gate == 256u, "%zu bytes", gate);

    /* THE PER-LAYER TOTAL, AND THE 24-LAYER ONE THE CAPACITY PLAN HAS TO CARRY. Carried
     * state is 5 MiB a layer (2 recurrent + 3 window) = 120 MiB over 24, against the 50.25
     * the plan reserves; the conv weight adds 96 MiB of constant on top. Stated as one
     * expression so the plan's number and the allocation cannot drift apart again. */
    ft_tt_test_check_fmt("carried state over 24 SSM layers is 120 MiB, not 50.25",
                         (state + window) * 24u == 120u * mib, "%.2f MiB",
                         (double)((state + window) * 24u) / (double)mib);
    ft_tt_test_check_fmt("with the conv weights the family owns 216 MiB of card DRAM",
                         (state + window + weight) * 24u == 216u * mib, "%.2f MiB",
                         (double)((state + window + weight) * 24u) / (double)mib);

    /* An odd head count is the shape the 64 B rounding exists for: 2*31*4 = 248 -> 256. */
    request.v_heads = 31u;
    request.k_heads = 31u;
    ft_tt_test_check_fmt("31 v-heads: 248 B of gate rounds UP to the 64 B read granule",
                         ft_tt_op_gdn_gate_bytes(&request) == 256u, "%zu bytes",
                         ft_tt_op_gdn_gate_bytes(&request));
    request.v_heads = 33u;
    request.k_heads = 33u;
    ft_tt_test_check_fmt("33 v-heads: 264 B rounds to 320, so the read has somewhere to land",
                         ft_tt_op_gdn_gate_bytes(&request) == 320u, "%zu bytes",
                         ft_tt_op_gdn_gate_bytes(&request));

    /* Doubling the v-heads doubles y and the v block; q and k do not move, because they are
     * per K-HEAD. 384 input tiles (64 + 64 + 256) and 256 of y. */
    request.v_heads = 64u;
    request.k_heads = 16u;
    ft_tt_test_check_fmt("64 v-heads over 16 k-heads: q|k|v is 384 tiles, y is 256",
                         ft_tt_op_gdn_delta_input_bytes(&request) == 384u * 4096u &&
                             ft_tt_op_gdn_delta_output_bytes(&request) == 256u * 4096u,
                         "%zu %zu", ft_tt_op_gdn_delta_input_bytes(&request),
                         ft_tt_op_gdn_delta_output_bytes(&request));

    /* A request this family will not execute sizes NOTHING, so an unchecked caller allocates
     * nothing rather than something wrong. */
    request.k_heads = 24u; /* does not divide 64 — the GQA repeat is not a repeat */
    ft_tt_test_check_fmt("a shape the family refuses sizes to 0 everywhere",
                         ft_tt_op_gdn_state_bytes(&request) == 0u &&
                             ft_tt_op_gdn_conv_state_bytes(&request) == 0u &&
                             ft_tt_op_gdn_conv_weight_bytes(&request) == 0u &&
                             ft_tt_op_gdn_delta_input_bytes(&request) == 0u &&
                             ft_tt_op_gdn_gate_bytes(&request) == 0u,
                         "%zu", ft_tt_op_gdn_state_bytes(&request));
}

/**
 * THE CONV COLUMN SLICE, encoded by the host and decoded by the kernels' own ftg_decode().
 *
 * The same two bytes are the DELTA head slice, so each phase must read only its own meaning,
 * an unsliced CONV must decode to the whole stream with every head, and every malformed slice
 * must be refused by the ENCODER rather than reinterpreted by a kernel.
 */
static void test_conv_column_slice(void)
{
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];
    ftg_shape shape;
    int rc;

    ft_tt_test_step("PHASE CONV column slice");
    fill_conv_request(&request);
    rc = ft_tt_op_gdn_parameters(&request, words);
    shape = ftg_decode(words);
    ft_tt_test_check_fmt("an unsliced CONV decodes to the whole stream, first 0",
                         rc == FT_TT_OK && shape.col_first == 0u &&
                             shape.col_count == shape.conv_tile_cols && shape.conv_tile_cols > 0u,
                         "rc %d, cols %u+%u of %u", rc, shape.col_first, shape.col_count,
                         shape.conv_tile_cols);
    ft_tt_test_check_fmt("and owns every head (the head slice is not read in CONV)",
                         shape.head_first == 0u && shape.head_count == shape.v_heads,
                         "heads %u+%u of %u", shape.head_first, shape.head_count, shape.v_heads);

    request.col_first = 64u;
    request.col_count = 64u;
    rc = ft_tt_op_gdn_parameters(&request, words);
    shape = ftg_decode(words);
    ft_tt_test_check_fmt("a slice [64, 128) round-trips through the kernels' decode",
                         rc == FT_TT_OK && shape.col_first == 64u && shape.col_count == 64u &&
                             ftg_shape_is_supported(&shape),
                         "rc %d, cols %u+%u", rc, shape.col_first, shape.col_count);
    ft_tt_test_check_fmt("and a sliced CONV still owns every head",
                         shape.head_first == 0u && shape.head_count == shape.v_heads,
                         "heads %u+%u", shape.head_first, shape.head_count);
    ft_tt_test_check_fmt("the plane stride stays the FULL stream: tap 1 column 64 is tile "
                         "conv_tile_cols + 64",
                         ftg_conv_plane_tile(&shape, 1u, 64u) == shape.conv_tile_cols + 64u,
                         "%u", ftg_conv_plane_tile(&shape, 1u, 64u));

    request.col_first = shape.conv_tile_cols - 8u;
    request.col_count = 16u;
    ft_tt_test_check_fmt("a slice that runs off the stream is REFUSED by the encoder",
                         ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "%s", "");
    request.col_first = 8u;
    request.col_count = 0u;
    ft_tt_test_check_fmt("a first column with no count is REFUSED",
                         ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "%s", "");
    fill_conv_request(&request);
    request.head_count = 4u;
    ft_tt_test_check_fmt("a HEAD slice on CONV is still REFUSED",
                         ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "%s", "");
    fill_delta_request(&request);
    request.col_count = 32u;
    ft_tt_test_check_fmt("a COLUMN slice on DELTA is REFUSED, not read as heads",
                         ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "%s", "");
    fill_delta_request(&request);
    request.head_first = 8u;
    request.head_count = 8u;
    rc = ft_tt_op_gdn_parameters(&request, words);
    shape = ftg_decode(words);
    ft_tt_test_check_fmt("a DELTA head slice still decodes as heads, and its CONV range is "
                         "the whole (unused) stream",
                         rc == FT_TT_OK && shape.head_first == 8u && shape.head_count == 8u &&
                             shape.col_first == 0u,
                         "heads %u+%u col_first %u", shape.head_first, shape.head_count,
                         shape.col_first);
}

/** The CONV-taps word an encoder written before the wide slice emits for [first, +count). */
static uint32_t narrow_taps_word(uint32_t first, uint32_t count)
{
    return 4u | (first << 8) | (count << 16) | ((ft_tt_op_gdn_conv_depth() & 0xFFu) << 24);
}

/**
 * THE WIDE CONV COLUMN SLICE: Qwen3.8's stream is 320 tile columns (conv_dim 10240), and an 8-
 * or 16-way split has slices starting past column 255 — the two bytes of the CONV-taps word
 * cannot name them, so the high bytes ride in the tile-column word's high half-word.
 *
 * Every member of both splits must round-trip through the kernels' own ftg_decode() and the
 * decoded slices must PARTITION the stream; a slice under 256 columns (every 4-way slice, and
 * every encoder before this one) must be the same bytes and decode to the same slice.
 */
static void test_conv_wide_column_slice(void)
{
    static const uint32_t splits[] = {8u, 16u, 4u};
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];
    ftg_shape shape;
    uint32_t split, index, column;
    int rc;

    ft_tt_test_step("PHASE CONV wide column slice (Qwen3.8: conv_dim 10240 = 320 tile columns)");
    for (split = 0; split < sizeof(splits) / sizeof(splits[0]); split++) {
        const uint32_t members = splits[split], per = 320u / members;
        uint32_t owners[320];
        bool round_trips = true, narrow_bytes = true, partition = true;

        memset(owners, 0, sizeof(owners));
        for (index = 0; index < members; index++) {
            fill_conv_request(&request);
            request.conv_dim = 10240u;
            request.col_first = index * per;
            request.col_count = per;
            rc = ft_tt_op_gdn_parameters(&request, words);
            shape = ftg_decode(words);
            if (rc != FT_TT_OK || shape.conv_tile_cols != 320u ||
                shape.col_first != index * per || shape.col_count != per ||
                !ftg_shape_is_supported(&shape))
                round_trips = false;
            /* Under 256 the shipped bytes: no high half, and the narrow taps word. */
            if (request.col_first < 256u &&
                (words[FT_TT_OP_GDN_CONV_TILE_COLS] != 320u ||
                 words[FT_TT_OP_GDN_CONV_TAPS] != narrow_taps_word(index * per, per)))
                narrow_bytes = false;
            for (column = shape.col_first; column < shape.col_first + shape.col_count &&
                                           column < 320u; column++)
                owners[column]++;
        }
        for (column = 0; column < 320u; column++)
            if (owners[column] != 1u)
                partition = false;
        ft_tt_test_check_fmt("every member of the split round-trips through ftg_decode()",
                             round_trips, "%u ways of %u columns", members, per);
        ft_tt_test_check_fmt("every slice under column 256 is the shipped bytes", narrow_bytes,
                             "%u ways", members);
        ft_tt_test_check_fmt("the decoded slices partition the 320 columns, each owned once",
                             partition, "%u ways", members);
    }

    fill_conv_request(&request);
    request.conv_dim = 10240u;
    request.col_first = 300u;
    request.col_count = 20u;
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("[300, 320): high bytes in the tile-column word, low in the taps word",
                         rc == FT_TT_OK &&
                             words[FT_TT_OP_GDN_CONV_TILE_COLS] == (320u | (1u << 16)) &&
                             words[FT_TT_OP_GDN_CONV_TAPS] == narrow_taps_word(300u & 0xFFu, 20u),
                         "rc %d, 0x%08X 0x%08X", rc, words[FT_TT_OP_GDN_CONV_TILE_COLS],
                         words[FT_TT_OP_GDN_CONV_TAPS]);

    request.col_first = 0u;
    request.col_count = 320u;
    rc = ft_tt_op_gdn_parameters(&request, words);
    shape = ftg_decode(words);
    ft_tt_test_check_fmt("an explicit [0, 320) decodes to 320 columns (the count's high byte)",
                         rc == FT_TT_OK && shape.col_first == 0u && shape.col_count == 320u &&
                             words[FT_TT_OP_GDN_CONV_TILE_COLS] == (320u | (1u << 24)),
                         "rc %d, cols %u+%u, 0x%08X", rc, shape.col_first, shape.col_count,
                         words[FT_TT_OP_GDN_CONV_TILE_COLS]);

    /* A descriptor an older encoder built: high half zero. It decodes as it always did. */
    fill_conv_request(&request);
    request.conv_dim = 10240u;
    rc = ft_tt_op_gdn_parameters(&request, words);
    words[FT_TT_OP_GDN_CONV_TAPS] = narrow_taps_word(80u, 80u);
    shape = ftg_decode(words);
    ft_tt_test_check_fmt("a narrow-format [80, 160) descriptor decodes unchanged",
                         rc == FT_TT_OK && shape.conv_tile_cols == 320u &&
                             shape.col_first == 80u && shape.col_count == 80u,
                         "rc %d, cols %u+%u of %u", rc, shape.col_first, shape.col_count,
                         shape.conv_tile_cols);
    words[FT_TT_OP_GDN_CONV_TAPS] = narrow_taps_word(0u, 0u);
    shape = ftg_decode(words);
    ft_tt_test_check_fmt("and an unsliced one is still the whole stream",
                         shape.col_first == 0u && shape.col_count == 320u, "cols %u+%u",
                         shape.col_first, shape.col_count);

    fill_conv_request(&request);
    request.conv_dim = 10240u;
    request.col_first = 310u;
    request.col_count = 20u;
    ft_tt_test_check_fmt("a wide slice that runs off the stream is REFUSED",
                         ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "%s", "");
    request.col_first = FT_TT_OP_GDN_COL_MAX + 1u;
    request.col_count = 2u;
    ft_tt_test_check_fmt("a first column past 16 bits is REFUSED",
                         ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "%s", "");
    fill_conv_request(&request);
    request.conv_dim = (FT_TT_OP_GDN_COL_MAX + 1u) * FT_TT_OP_TILE_DIM;
    ft_tt_test_check_fmt("a stream wider than the tile-column half-word is REFUSED",
                         ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "%s", "");
    fill_delta_request(&request);
    request.head_first = 8u;
    request.head_count = 8u;
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("a DELTA descriptor still leaves the tile-column word zero",
                         rc == FT_TT_OK && words[FT_TT_OP_GDN_CONV_TILE_COLS] == 0u, "0x%08X",
                         words[FT_TT_OP_GDN_CONV_TILE_COLS]);
}

/** Host model of where one CONV descriptor of `cols` columns leaves CB_STATE and CB_PROD:
 *  the reader's and the compute thread's pushes, block by block, modulo the placed pages. */
static void model_conv_boundary(uint32_t cols, uint32_t taps, uint32_t state_pages,
                                uint32_t prod_pages, uint32_t *state_at, uint32_t *prod_at)
{
    uint32_t depth = ft_tt_op_gdn_conv_depth(), first, tap, column;

    if (depth > cols)
        depth = cols;
    while (depth > 1u && cols % depth != 0u) /* the kernels' conv_block_depth() */
        depth--;
    *state_at = 0u;
    *prod_at = 0u;
    for (first = 0; first < cols; first += depth) { /* pass A: every window tap */
        *state_at = (*state_at + (taps - 1u) * depth) % state_pages;
        for (column = 0; column < depth; column++)
            *prod_at = (*prod_at + (taps - 1u)) % prod_pages; /* one product a window tap */
    }
    for (tap = 0; tap + 2u < taps; tap++) /* pass B: the window one tap higher */
        for (first = 0; first < cols; first += depth)
            *state_at = (*state_at + depth) % state_pages;
}

/**
 * THE CONV SLICE GRANULE (FT_TT_OP_GDN_CONV_SLICE_GRANULE): the 18-column slice [300, 318) the
 * first `gdn_conv_col_drop` red sent ran its CONV and broke the DELTA after it on silicon. The
 * encoder must refuse it, and admit every legal Qwen3.8 shape — exactly the slices that leave
 * CB_STATE and CB_PROD on PHASE DELTA's grid in the host model above.
 */
static void test_conv_slice_granule(void)
{
    static const uint32_t splits[] = {4u, 8u, 16u};
    ft_tt_op_gdn_request request;
    ft_tt_launch_cb_entry entries[FT_TT_OP_GDN_CB_COUNT];
    uint32_t words[FT_TT_OP_GDN_WORDS];
    uint32_t state_pages = 0u, prod_pages = 0u, state_at, prod_at, cols, split, index;
    const uint32_t tiles = 128u / FT_TT_OP_TILE_DIM;
    const uint32_t batch = ftg_dst_batch(tiles, FT_TT_OP_GDN_DST_BATCH);
    uint64_t mask = 0u;
    bool agrees = true, legal = true;
    int rc;

    ft_tt_test_step("PHASE CONV slice granule (the 18-column slice that broke DELTA on silicon)");
    fill_conv_request(&request);
    request.conv_dim = 10240u;
    request.state_size = 128u;
    request.conv_taps = 4u;
    rc = ft_tt_op_gdn_place_cbs(&request, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    for (index = 0; rc == FT_TT_OK && index < FT_TT_OP_GDN_CB_COUNT; index++) {
        if (entries[index].index == FT_TT_OP_GDN_CB_STATE)
            state_pages = entries[index].page_count;
        if (entries[index].index == FT_TT_OP_GDN_CB_PROD)
            prod_pages = entries[index].page_count;
    }
    ft_tt_test_check_fmt("Qwen3.8's placement: CB_STATE 12 pages, CB_PROD 8, DELTA grids 4 and 4",
                         rc == FT_TT_OK && state_pages == 12u && prod_pages == 8u &&
                             tiles == 4u && batch == 4u,
                         "rc %d state %u prod %u batch %u", rc, state_pages, prod_pages, batch);
    if (rc != FT_TT_OK || state_pages == 0u || prod_pages == 0u)
        return;
    ft_tt_test_check_fmt("the granule at S 128, 4 taps, depth 2 is 4 columns",
                         ft_tt_op_gdn_conv_slice_granule(128u, 4u) == 4u, "%u",
                         ft_tt_op_gdn_conv_slice_granule(128u, 4u));

    model_conv_boundary(18u, 4u, state_pages, prod_pages, &state_at, &prod_at);
    ft_tt_test_check_fmt("the model: 18 columns leave CB_STATE at page 6, CB_PROD at 6 (off "
                         "DELTA's 4-page grid)",
                         state_at == 6u && prod_at == 6u, "state %u prod %u", state_at, prod_at);
    model_conv_boundary(20u, 4u, state_pages, prod_pages, &state_at, &prod_at);
    ft_tt_test_check_fmt("the model: 20 columns (the shipped 16-way slice) leave both on grid",
                         state_at % tiles == 0u && prod_at % batch == 0u, "state %u prod %u",
                         state_at, prod_at);

    request.col_first = 300u;
    request.col_count = 18u;
    ft_tt_test_check_fmt("the 18-column slice [300, 318) is REFUSED before the card sees it",
                         ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "%s", "");
    request.col_first = 298u;
    request.col_count = 20u;
    ft_tt_test_check_fmt("[298, 318), the fixed drop red's shape, is admitted",
                         ft_tt_op_gdn_parameters(&request, words) == FT_TT_OK, "%s", "");

    /* The encoder admits a count exactly when the model leaves both buffers on grid. */
    for (cols = ft_tt_op_gdn_conv_depth(); cols <= 320u; cols += ft_tt_op_gdn_conv_depth()) {
        const bool on_grid =
            (model_conv_boundary(cols, 4u, state_pages, prod_pages, &state_at, &prod_at),
             state_at % tiles == 0u && prod_at % batch == 0u);

        request.col_first = 0u;
        request.col_count = cols;
        if ((ft_tt_op_gdn_parameters(&request, words) == FT_TT_OK) != on_grid)
            agrees = false;
    }
    ft_tt_test_check_fmt("for every count up to 320, admitted == on grid in the model", agrees,
                         "%s", "");

    for (split = 0; split < sizeof(splits) / sizeof(splits[0]); split++)
        for (index = 0; index < splits[split]; index++) {
            request.col_first = index * (320u / splits[split]);
            request.col_count = 320u / splits[split];
            if (ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK)
                legal = false;
        }
    request.col_first = 0u;
    request.col_count = 0u;
    if (ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK)
        legal = false;
    ft_tt_test_check_fmt("every 4/8/16-way member and the whole stream are admitted", legal,
                         "%s", "");
    ft_tt_test_check_fmt("a shape with no CONV placement has no granule",
                         ft_tt_op_gdn_conv_slice_granule(128u, 1u) == 0u &&
                             ft_tt_op_gdn_conv_slice_granule(0u, 4u) == 0u,
                         "%u %u", ft_tt_op_gdn_conv_slice_granule(128u, 1u),
                         ft_tt_op_gdn_conv_slice_granule(0u, 4u));
}

int main(void)
{
    ft_tt_test_step("Gated-Delta-Net op family — off-hardware acceptance (no device is "
                    "opened; the triad does not exist yet)");
    test_wire_constants();
    test_compile_time_arg_contract();
    test_declared_conv_depth();
    test_dst_batch_clamp();
    test_writer_block_rungs();
    test_writer_page_walk();
    test_parameters();
    test_refusals();
    test_conv_column_slice();
    test_conv_wide_column_slice();
    test_conv_slice_granule();
    test_cb_placement();
    test_cb_placement_serves_the_device();
    test_tile_format();
    test_delta_input_layout();
    test_state_and_output_layout();
    test_capacity();
    return ft_tt_test_finish();
}
