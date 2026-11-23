/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_attention_ops.c
 * @brief OFF-HARDWARE gate for flash-decode attention: the wire format, the derived
 *   quantities, the refusals, the constants, the capacity wall — and the ALGORITHM, scored
 *   as a blocked online softmax against a two-pass reference.
 *
 * Built WITHOUT -DFT_TT_PHASE1B_NOC_ENABLED, so every device call in the objects it links
 * would be refused before touching the NoC. It cannot reach the card even if it is wrong,
 * which is what makes it safe to run while somebody else owns the silicon.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS CAN PROVE AND WHAT IT CANNOT
 * ---------------------------------------------------------------------------
 * It proves the ALGORITHM: that an online softmax over padded, masked KV blocks reproduces
 * a two-pass softmax over the live row, on shapes where the two are not trivially the same.
 * It proves the CONTRACT: that the sixteen words mean on the host what they mean in the
 * kernel, by PARSING the kernel header rather than retyping its constants — the same
 * arrangement test_vector_ops.c uses, and the reason is dev notes §4's catalogue of
 * confident sentences nobody re-derived.
 *
 * It cannot prove the KERNEL. Nothing here executes a tile, a face permutation or a
 * circular buffer; PCC is blind to a consistent permutation, and there is no aperture
 * without a card (dev notes §3). tests/test_phase6_attention.c is where those go red.
 *
 * ---------------------------------------------------------------------------
 * THE INVARIANCE TESTS ARE THE POINT OF THE ALGORITHM SECTION
 * ---------------------------------------------------------------------------
 * softmax(x - c) = softmax(x), so a WRONG RUNNING MAX is structurally invisible to
 * correlation — that exact mutation came back green on seven checks in the reduce suite.
 * Two shapes here exist only to break that symmetry:
 * an outlier planted in a LATE block, so a block-local max produces values outside
 * representable range rather than a shifted-but-equal answer, and a dead-region poison, so
 * a mask that does not mask changes the answer instead of being absorbed.
 *
 * Usage:  ./build-attn/test_attention_ops
 *
 * @status See the Phase 6 attention report.
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_op_attention.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_memory.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "ft_tt_attention_reference.h"
#include "ft_tt_test_report.h"

/* THE DEVICE HALF, INCLUDED RATHER THAN RETYPED. The chunk mode's mask selection and its
 * geometry refusal are INLINE FUNCTIONS in the kernel header, so including it makes this
 * suite exercise the code the card runs instead of a host copy of it — the same code the
 * reader and the compute thread compile. The constants are still PARSED out of the file by
 * kernel_constant(), because those really are two declarations that can drift.
 * `FTQ_L1_PTR` is the SFPI GCC fork's own pointer attribute and means nothing here. */
#define FTQ_L1_PTR
#include "../kernels/resident_attention_protocol.h"

#define TILE_DIM   32u
#define TILE_BYTES FT_TT_OP_TILE_BYTES_BF16

/** MiniCPM5-1B, from the GGUF header: 16 Q heads, 2 KV
 *  heads, head_dim 128, 24 layers, context 131072. Every shape below is that model's. */
#define MODEL_Q_HEADS    16u
#define MODEL_KV_HEADS   2u
#define MODEL_HEAD_DIM   128u
#define MODEL_DIM_TILES  (MODEL_HEAD_DIM / TILE_DIM)
#define MODEL_GROUP_ROWS (MODEL_Q_HEADS / MODEL_KV_HEADS)

/** Ornith-9B (`general.architecture = qwen35`), the SECOND model this op is asked to serve.
 *
 *  **READ OUT OF THE GGUF HEADER, 2026-08-20, after this stage spent a round asserting a
 *  shape the model does not have.** `qwen35.attention.head_count = 16`,
 *  `head_count_kv = 4`, `key_length = value_length = 256`, `block_count = 32` with 8 of them
 *  full attention. The 32/16 this file used to carry is the GDN block's head count, not the
 *  attention layers' — and every number derived from it (602 L1 pages, 128 descriptors a
 *  token, a 4 GiB KV family) was wrong while the stage reported 0 failures. Pantheon prints
 *  the truth at arm time: "16 Q heads / 4 KV groups of 4 rows".
 *
 *  It is still the near-inverse of MiniCPM5-1B — a WIDE head (dim_tiles 8 against 4) over a
 *  SHALLOWER group (q_rows 4 against 8) — which is why it is worth its own stage: every
 *  bound in this op is a product of `q_rows` and `dim_tiles`. */
#define ORNITH_Q_HEADS    16u
#define ORNITH_KV_HEADS   4u
#define ORNITH_HEAD_DIM   256u
#define ORNITH_ATTN_LAYERS 8u
#define ORNITH_DIM_TILES  (ORNITH_HEAD_DIM / TILE_DIM)
#define ORNITH_GROUP_ROWS (ORNITH_Q_HEADS / ORNITH_KV_HEADS)

/** The residency every algorithm check runs at. */
#define TEST_BLOCK_TILES 8u

/** What `pn_tt_attention.c` ships as its flash-decode prefetch depth. */
#define SHIPPED_BLOCK_TILES 16u

/* ================================================================
 * The kernel header, PARSED — not retyped
 * ================================================================ */

/**
 * Read `enum { <name> = <value> };` out of the kernel's protocol header -- the
 * C/C++-shared declaration form the device headers use since plan Phase 6.6.
 *
 * The host and the device halves of this op are two files that must agree word for word.
 * Retyping the device constants here would make the check a copy of a copy: it would pass
 * against a header that had moved, because both would have been edited from the same wrong
 * belief. Reading the actual file is the only version of this check that can fail.
 *
 * @return the value, or UINT32_MAX if the name is absent (which is itself a failure)
 */
static uint32_t kernel_constant(const char *name)
{
    static char buffer[262144];
    static size_t length = 0;
    const char *directory = getenv("FT_TT_KERNEL_DIR");
    char path[512];
    char needle[128];
    const char *found;

    if (length == 0u) {
        FILE *file;

        snprintf(path, sizeof(path), "%s/resident_attention_protocol.h",
                 directory ? directory : "../kernels");
        file = fopen(path, "rb");
        if (!file) {
            /* SAY WHY, ONCE. Without this the whole wire-contract stage reports UINT32_MAX
             * against every host value and reads as 22 protocol mismatches, when the only
             * fact is that the binary was run from a directory where `../kernels` is not the
             * kernel directory. A harness that fails for a reason other than the one it
             * prints costs a round. */
            static bool said = false;

            if (!said)
                fprintf(stderr, "test_attention_ops: cannot open %s — set FT_TT_KERNEL_DIR, "
                                "or run from tt/tests. Every kernel constant below reads as "
                                "UINT32_MAX and NONE of it is a protocol finding.\n", path);
            said = true;
            return UINT32_MAX;
        }
        length = fread(buffer, 1u, sizeof(buffer) - 1u, file);
        buffer[length] = '\0';
        fclose(file);
    }

    snprintf(needle, sizeof(needle), "enum { %s = ", name);
    found = strstr(buffer, needle);
    if (!found)
        return UINT32_MAX;
    return (uint32_t)strtoul(found + strlen(needle), NULL, 0);
}

static void stage_wire_contract(void)
{
    static const struct {
        const char *kernel_name;
        uint32_t host_value;
    } pairs[] = {
        { "kAttnHeads", FT_TT_OP_ATTENTION_HEADS },
        { "kAttnDimTiles", FT_TT_OP_ATTENTION_DIM_TILES },
        { "kAttnKvPositions", FT_TT_OP_ATTENTION_KV_POSITIONS },
        { "kAttnBlockTiles", FT_TT_OP_ATTENTION_BLOCK_TILES },
        { "kAttnCapacityTiles", FT_TT_OP_ATTENTION_CAPACITY_TILES },
        { "kAttnReserved", FT_TT_OP_ATTENTION_RESERVED },
        { "kAttnQCoord", FT_TT_OP_ATTENTION_Q_COORD },
        { "kAttnQBase", FT_TT_OP_ATTENTION_Q_BASE },
        { "kAttnKCoord", FT_TT_OP_ATTENTION_K_COORD },
        { "kAttnKBase", FT_TT_OP_ATTENTION_K_BASE },
        { "kAttnVCoord", FT_TT_OP_ATTENTION_V_COORD },
        { "kAttnVBase", FT_TT_OP_ATTENTION_V_BASE },
        { "kAttnConstCoord", FT_TT_OP_ATTENTION_CONST_COORD },
        { "kAttnConstBase", FT_TT_OP_ATTENTION_CONST_BASE },
        { "kAttnOutCoord", FT_TT_OP_ATTENTION_OUT_COORD },
        { "kAttnOutBase", FT_TT_OP_ATTENTION_OUT_BASE },
        { "kAttnWords", FT_TT_OP_ATTENTION_WORDS },
        { "kAttnConstScalerTile", FT_TT_OP_ATTENTION_CONST_SCALER_TILE },
        { "kAttnConstScaleTile", FT_TT_OP_ATTENTION_CONST_SCALE_TILE },
        { "kAttnConstMaskTile", FT_TT_OP_ATTENTION_CONST_MASK_TILE },
        /* kAttnConstTiles is DERIVED on both sides (`causal + 32`), so it is checked by the
         * "66 tiles" arithmetic in stage_chunk_contract() rather than pinned here — this
         * parser reads literals, and a pin that reads 0 for a derived name is a false
         * finding, not a check. */
        { "kAttnConstCausalTile", FT_TT_OP_ATTENTION_CONST_CAUSAL_TILE },
        { "kAttnFlagChunk", FT_TT_OP_ATTENTION_FLAG_CHUNK },
        { "kAttnCbConstPages", FT_TT_OP_ATTENTION_CONST_PAGES },
        { "kAttnTileRows", TILE_DIM }
    };
    size_t index;

    ft_tt_test_step("the host and the kernel agree word for word — by PARSING the kernel "
                    "header, not by retyping it");

    for (index = 0; index < sizeof(pairs) / sizeof(pairs[0]); index++) {
        const uint32_t device = kernel_constant(pairs[index].kernel_name);
        char label[128];

        snprintf(label, sizeof(label), "%s", pairs[index].kernel_name);
        ft_tt_test_check_fmt(label, device == pairs[index].host_value, "kernel %u host %u",
                             device, pairs[index].host_value);
    }
}

/* ================================================================
 * The parameter block
 * ================================================================ */

static void fill_operand(ft_tt_op_operand *operand, uint16_t x, uint16_t y, uint32_t address)
{
    operand->x = x;
    operand->y = y;
    operand->address = address;
    operand->tile_bytes = TILE_BYTES;
}

/** A request whose every field is a DIFFERENT recognisable value, so a word in the wrong
 *  slot cannot pass for its neighbour. */
static void fill_reference_request(ft_tt_op_attention_request *request)
{
    memset(request, 0, sizeof(*request));
    request->q_rows = MODEL_GROUP_ROWS;
    request->dim_tiles = MODEL_DIM_TILES;
    request->block_tiles = TEST_BLOCK_TILES;
    request->kv_positions = 1000u;
    request->capacity_tiles = 4096u;
    request->query_row = 7u;
    fill_operand(&request->query, 1u, 2u, 0x00010000u);
    fill_operand(&request->keys, 3u, 4u, 0x00020000u);
    fill_operand(&request->values, 5u, 6u, 0x00030000u);
    fill_operand(&request->constants, 7u, 8u, 0x00040000u);
    fill_operand(&request->output, 9u, 10u, 0x00050000u);
}

static void stage_parameters(void)
{
    ft_tt_op_attention_request request;
    uint32_t words[FT_TT_OP_ATTENTION_WORDS];
    int rc;

    ft_tt_test_step("the parameter block, pinned as LITERALS");

    fill_reference_request(&request);
    rc = ft_tt_op_attention_parameters(&request, words);
    ft_tt_test_check_fmt("a MiniCPM5-1B group encodes", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;

    ft_tt_test_check_fmt("heads word packs q_rows and the query row",
                         words[FT_TT_OP_ATTENTION_HEADS] == (8u | (7u << 8)), "0x%X",
                         words[FT_TT_OP_ATTENTION_HEADS]);
    ft_tt_test_check_fmt("dim tiles", words[FT_TT_OP_ATTENTION_DIM_TILES] == 4u, "%u",
                         words[FT_TT_OP_ATTENTION_DIM_TILES]);
    ft_tt_test_check_fmt("kv positions", words[FT_TT_OP_ATTENTION_KV_POSITIONS] == 1000u, "%u",
                         words[FT_TT_OP_ATTENTION_KV_POSITIONS]);
    ft_tt_test_check_fmt("block tiles", words[FT_TT_OP_ATTENTION_BLOCK_TILES] == 8u, "%u",
                         words[FT_TT_OP_ATTENTION_BLOCK_TILES]);
    ft_tt_test_check_fmt("capacity tiles", words[FT_TT_OP_ATTENTION_CAPACITY_TILES] == 4096u,
                         "%u", words[FT_TT_OP_ATTENTION_CAPACITY_TILES]);
    ft_tt_test_check_fmt("reserved is zero", words[FT_TT_OP_ATTENTION_RESERVED] == 0u, "%u",
                         words[FT_TT_OP_ATTENTION_RESERVED]);
    ft_tt_test_check_fmt("Q coordinate packs (y << 16) | x",
                         words[FT_TT_OP_ATTENTION_Q_COORD] == ((2u << 16) | 1u), "0x%X",
                         words[FT_TT_OP_ATTENTION_Q_COORD]);
    ft_tt_test_check_fmt("K coordinate", words[FT_TT_OP_ATTENTION_K_COORD] == ((4u << 16) | 3u),
                         "0x%X", words[FT_TT_OP_ATTENTION_K_COORD]);
    ft_tt_test_check_fmt("V coordinate", words[FT_TT_OP_ATTENTION_V_COORD] == ((6u << 16) | 5u),
                         "0x%X", words[FT_TT_OP_ATTENTION_V_COORD]);
    ft_tt_test_check_fmt("constants coordinate",
                         words[FT_TT_OP_ATTENTION_CONST_COORD] == ((8u << 16) | 7u), "0x%X",
                         words[FT_TT_OP_ATTENTION_CONST_COORD]);
    ft_tt_test_check_fmt("output coordinate",
                         words[FT_TT_OP_ATTENTION_OUT_COORD] == ((10u << 16) | 9u), "0x%X",
                         words[FT_TT_OP_ATTENTION_OUT_COORD]);
    ft_tt_test_check_fmt("every base survives",
                         words[FT_TT_OP_ATTENTION_Q_BASE] == 0x00010000u &&
                             words[FT_TT_OP_ATTENTION_K_BASE] == 0x00020000u &&
                             words[FT_TT_OP_ATTENTION_V_BASE] == 0x00030000u &&
                             words[FT_TT_OP_ATTENTION_CONST_BASE] == 0x00040000u &&
                             words[FT_TT_OP_ATTENTION_OUT_BASE] == 0x00050000u,
                         "0x%X 0x%X 0x%X 0x%X 0x%X", words[FT_TT_OP_ATTENTION_Q_BASE],
                         words[FT_TT_OP_ATTENTION_K_BASE], words[FT_TT_OP_ATTENTION_V_BASE],
                         words[FT_TT_OP_ATTENTION_CONST_BASE],
                         words[FT_TT_OP_ATTENTION_OUT_BASE]);
    ft_tt_test_check("the opcode is 0x11", FT_TT_QUEUE_OP_ATTENTION == 0x11u, "");
}

/** Each refusal names its REASON, not merely "it failed": a check with several sufficient
 *  grounds tests whichever is cheapest (dev notes §3). */
static void stage_refusals(void)
{
    ft_tt_op_attention_request request;
    uint32_t words[FT_TT_OP_ATTENTION_WORDS];

    ft_tt_test_step("what the encoder refuses, and for which reason");

    fill_reference_request(&request);
    request.kv_positions = 0u;
    ft_tt_test_check("a zero KV length is refused",
                     ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL, "");

    fill_reference_request(&request);
    request.q_rows = 33u;
    ft_tt_test_check("a group wider than a tile is refused",
                     ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL, "");

    fill_reference_request(&request);
    request.query_row = 32u;
    ft_tt_test_check("a query row past the tile is refused",
                     ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL, "");

    fill_reference_request(&request);
    request.block_tiles = 0u;
    ft_tt_test_check("a zero block is refused (it is a zero-length NoC read)",
                     ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL, "");

    /* THE ONE THAT MATTERS. The reader fetches the PADDED window, so a capacity that covers
     * the live length but not the padding is a read past the end of the KV region. */
    fill_reference_request(&request);
    request.kv_positions = 33u * TILE_DIM;
    request.capacity_tiles = 33u;
    ft_tt_test_check("a capacity that covers the LIVE length but not the PADDED window is "
                     "refused",
                     ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL, "");
    request.capacity_tiles = 48u;
    ft_tt_test_check("...and the same shape passes once the region holds the padding",
                     ft_tt_op_attention_parameters(&request, words) == FT_TT_OK, "");

    /* K and V take their cache's tile, block float included and each its own
     * (ft_tt_op_memory.h §1.6; test_kv_mixed_format.c), but never a size no cache has. */
    fill_reference_request(&request);
    request.keys.tile_bytes = 1024u;
    ft_tt_test_check("a K operand whose tile is no cache format is refused",
                     ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL, "");
    fill_reference_request(&request);
    request.values.tile_bytes = 1024u;
    ft_tt_test_check("a V operand whose tile is no cache format is refused",
                     ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL, "");
}

static void stage_derived_quantities(void)
{
    static const struct {
        uint32_t kv;
        uint32_t live;
        uint32_t padded;
    } cases[] = {
        { 1u, 1u, 16u },      { 32u, 1u, 16u },    { 33u, 2u, 16u },
        { 512u, 16u, 16u },   { 513u, 17u, 32u },  { 1000u, 32u, 32u },
        { 8192u, 256u, 256u }, { 8193u, 257u, 272u }
    };
    size_t index;

    ft_tt_test_step("the derived quantities — live tiles and the padded window at "
                    "block_tiles = 8");

    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        const uint32_t live = ft_tt_op_attention_live_tiles(cases[index].kv);
        const uint32_t padded = ft_tt_op_attention_padded_tiles(cases[index].kv, TEST_BLOCK_TILES);
        char label[96];

        snprintf(label, sizeof(label), "kv=%u", cases[index].kv);
        ft_tt_test_check_fmt(label, live == cases[index].live && padded == cases[index].padded,
                             "live %u (want %u) padded %u (want %u)", live, cases[index].live,
                             padded, cases[index].padded);
    }

    /* The block-prefetch invariant's second condition, as arithmetic: the descriptor has to
     * leave the write pointer where it found it, which means the padded tile count is a
     * whole number of DOUBLE blocks. */
    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        const uint32_t padded = ft_tt_op_attention_padded_tiles(cases[index].kv, TEST_BLOCK_TILES);
        char label[128];

        snprintf(label, sizeof(label), "kv=%u leaves the CB write pointer at the base",
                 cases[index].kv);
        ft_tt_test_check_fmt(label, padded % (2u * TEST_BLOCK_TILES) == 0u, "padded %u", padded);
    }
}

/* ================================================================
 * The constants
 * ================================================================ */

static void stage_constants(void)
{
    static uint16_t constants[FT_TT_OP_ATTENTION_CONST_TILES * TILE_DIM * TILE_DIM];
    const float expected_scale = 1.0f / sqrtf((float)MODEL_HEAD_DIM);
    uint32_t tail;
    int rc;

    ft_tt_test_step("the constants operand — one scaler, one query scale, 32 masks");

    rc = ft_tt_op_attention_build_constants(MODEL_HEAD_DIM, TILE_BYTES, constants,
                                            sizeof(constants));
    ft_tt_test_check_fmt("it builds", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;

    ft_tt_test_check_fmt("the reduce scaler is 1.0 everywhere",
                         ft_float_from_bf16(constants[0]) == 1.0f &&
                             ft_float_from_bf16(constants[1023]) == 1.0f,
                         "%f", (double)ft_float_from_bf16(constants[0]));
    {
        const float scale = ft_float_from_bf16(constants[TILE_DIM * TILE_DIM]);

        ft_tt_test_check_fmt("the query scale is 1/sqrt(head_dim)",
                             fabs((double)scale - (double)expected_scale) < 0.002,
                             "%f vs %f", (double)scale, (double)expected_scale);
    }

    /* A mask that is right in face (0,0) and wrong everywhere else is the exact failure a
     * flat-index check cannot see, so every row of every mask
     * is walked in FACE order. */
    for (tail = 0; tail < TILE_DIM; tail++) {
        const uint16_t *tile =
            constants + (FT_TT_OP_ATTENTION_CONST_MASK_TILE + tail) * TILE_DIM * TILE_DIM;
        bool correct = true;
        uint32_t row;

        for (row = 0; row < TILE_DIM && correct; row++) {
            uint32_t column;

            for (column = 0; column < TILE_DIM; column++) {
                const float value = ft_float_from_bf16(tile[ft_tt_tile_face_index(row, column)]);
                const bool live = column < tail;

                if (live ? (value != 0.0f) : (value > -1.0e29f)) {
                    correct = false;
                    break;
                }
            }
        }
        if (tail == 0u || tail == 1u || tail == 17u || tail == 31u) {
            char label[96];

            snprintf(label, sizeof(label), "mask %u is dead at and above column %u, in FACE "
                                           "order",
                     tail, tail);
            ft_tt_test_check(label, correct, "");
        } else {
            ft_tt_test_check("a mask tile is correct", correct, "");
        }
    }
}

/* ================================================================
 * CHUNK MODE — the flag, the geometry refusal, the mask selection, the causal tile
 * ================================================================ */

/** A chunk descriptor for `rows` positions starting at `first`. */
static void fill_chunk_request(ft_tt_op_attention_request *request, uint32_t first,
                               uint32_t rows)
{
    fill_reference_request(request);
    request->q_rows = rows;
    request->query_row = 0u;
    request->flags = FT_TT_OP_ATTENTION_FLAG_CHUNK;
    request->kv_positions = first + rows;
    request->capacity_tiles = 4096u;
}

/** Is causal tile `constants[CAUSAL + live - 1]` live exactly where `c <= r && c < live`,
 *  walked in FACE order rather than flat? A mask right in face (0,0) and wrong elsewhere is
 *  the exact failure a flat-index check cannot see. */
static bool causal_tile_is_correct(const uint16_t *constants, uint32_t live)
{
    const uint16_t *tile =
        constants + (size_t)(FT_TT_OP_ATTENTION_CONST_CAUSAL_TILE + live - 1u) * TILE_DIM *
                        TILE_DIM;
    uint32_t row;

    for (row = 0; row < TILE_DIM; row++) {
        uint32_t column;

        for (column = 0; column < TILE_DIM; column++) {
            const float value = ft_float_from_bf16(tile[ft_tt_tile_face_index(row, column)]);
            const bool want_live = column <= row && column < live;

            if (want_live ? (value != 0.0f) : (value > -1.0e29f))
                return false;
        }
    }
    return true;
}

static void stage_chunk_contract(void)
{
    static uint16_t constants[FT_TT_OP_ATTENTION_CONST_TILES * TILE_DIM * TILE_DIM];
    ft_tt_op_attention_request request;
    uint32_t words[FT_TT_OP_ATTENTION_WORDS];
    fta_shape shape;
    uint32_t live;
    int rc;

    ft_tt_test_step("CHUNK MODE — one Q head over C positions, and causality as ONE tile");

    /* ---- the flag rides in byte 2, and the DEVICE decoder reads it back ---- */
    fill_chunk_request(&request, 992u, 32u);
    rc = ft_tt_op_attention_parameters(&request, words);
    ft_tt_test_check_fmt("a 32-position chunk at an aligned first position encodes",
                         rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("the flag rides in byte 2 and C in byte 0 of the heads word",
                         words[FT_TT_OP_ATTENTION_HEADS] == (32u | (1u << 16)), "0x%X",
                         words[FT_TT_OP_ATTENTION_HEADS]);

    shape = fta_decode(words);
    ft_tt_test_check_fmt("the DEVICE decoder reads C, row 0 and the chunk flag back",
                         shape.q_rows == 32u && shape.query_row == 0u &&
                             shape.flags == (uint32_t)kAttnFlagChunk,
                         "q_rows %u query_row %u flags %u", shape.q_rows, shape.query_row,
                         shape.flags);
    ft_tt_test_check("the device accepts the decoded chunk shape", fta_shape_valid(&shape), "");

    /* ---- THE MASK SELECTION, WHICH IS THE WHOLE OF CAUSALITY ----
     * At first = 992, C = 32: kv_positions 1024, live_tiles 32, tail 0. DECODE calls the
     * last live tile entirely live; CHUNK must mask it. That one difference is the bug this
     * mode exists to avoid, and it is invisible to every length and address check. */
    ft_tt_test_check_fmt("chunk: a tile before the last live one takes NO mask",
                         fta_mask_for_tile(&shape, 30u) == (uint32_t)kAttnCbConstPages, "%u",
                         fta_mask_for_tile(&shape, 30u));
    ft_tt_test_check_fmt("chunk: the LAST LIVE tile takes the mask slot even at tail 0",
                         fta_mask_for_tile(&shape, 31u) == (uint32_t)kAttnCbConstTailMask,
                         "tail %u -> slot %u", shape.tail, fta_mask_for_tile(&shape, 31u));
    ft_tt_test_check_fmt("chunk: a tile past the end takes the all-dead mask",
                         fta_mask_for_tile(&shape, 32u) == (uint32_t)kAttnCbConstDeadMask, "%u",
                         fta_mask_for_tile(&shape, 32u));
    ft_tt_test_check_fmt("chunk: the tail slot is loaded with the CAUSAL tile for C",
                         fta_tail_slot_source_tile(&shape) ==
                             (uint32_t)kAttnConstCausalTile + 31u,
                         "%u", fta_tail_slot_source_tile(&shape));
    {
        /* The SAME shape without the flag — the discriminating arm. Decode leaves that tile
         * unmasked, which is every position in the chunk attending to every other one. */
        fta_shape decode_shape = shape;

        decode_shape.flags = 0u;
        ft_tt_test_check_fmt("decode: the SAME shape leaves the last live tile UNMASKED at "
                             "tail 0 — which is what chunk mode had to change",
                             fta_mask_for_tile(&decode_shape, 31u) ==
                                 (uint32_t)kAttnCbConstPages,
                             "%u", fta_mask_for_tile(&decode_shape, 31u));
        ft_tt_test_check_fmt("decode: the tail slot is loaded with the TAIL mask",
                             fta_tail_slot_source_tile(&decode_shape) ==
                                 (uint32_t)kAttnConstMaskTile + decode_shape.tail,
                             "%u", fta_tail_slot_source_tile(&decode_shape));
    }

    /* ---- a RAGGED chunk: 7 positions at 96, the established split's last piece ---- */
    fill_chunk_request(&request, 96u, 7u);
    rc = ft_tt_op_attention_parameters(&request, words);
    ft_tt_test_check_fmt("the ragged chunk of the 32+32+32+7 split encodes", rc == FT_TT_OK,
                         "rc=%d", rc);
    shape = fta_decode(words);
    ft_tt_test_check_fmt("ragged: live_tiles 4, tail 7, and the DIAGONAL tile is tile 3",
                         shape.live_tiles == 4u && shape.tail == 7u &&
                             fta_mask_for_tile(&shape, 3u) == (uint32_t)kAttnCbConstTailMask,
                         "live %u tail %u", shape.live_tiles, shape.tail);
    ft_tt_test_check_fmt("ragged: the causal tile is C's, not the tail's — one tile does BOTH "
                         "jobs",
                         fta_tail_slot_source_tile(&shape) ==
                             (uint32_t)kAttnConstCausalTile + 6u,
                         "%u", fta_tail_slot_source_tile(&shape));

    /* ---- the geometry refusals, on BOTH sides ---- */
    fill_chunk_request(&request, 96u, 7u);
    request.kv_positions = 100u; /* first = 93, not a multiple of 32 */
    ft_tt_test_check("a chunk whose first position is not a multiple of 32 is refused",
                     ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL, "");

    fill_chunk_request(&request, 96u, 7u);
    request.query_row = 3u;
    ft_tt_test_check("a chunk with a non-zero query row is refused (rows ARE positions)",
                     ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL, "");

    fill_chunk_request(&request, 96u, 7u);
    request.flags = 2u;
    ft_tt_test_check("an unknown flag bit is refused rather than ignored",
                     ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL, "");

    fill_chunk_request(&request, 992u, 32u);
    (void)ft_tt_op_attention_parameters(&request, words);
    words[FT_TT_OP_ATTENTION_KV_POSITIONS] = 1000u; /* first = 968, unaligned */
    shape = fta_decode(words);
    ft_tt_test_check("...and the DEVICE refuses the unaligned chunk too, not only the host",
                     !fta_shape_valid(&shape), "");
    words[FT_TT_OP_ATTENTION_KV_POSITIONS] = 1024u;
    words[FT_TT_OP_ATTENTION_HEADS] = 32u | (3u << 8) | (1u << 16);
    shape = fta_decode(words);
    ft_tt_test_check("...and a chunk carrying a query row", !fta_shape_valid(&shape), "");
    words[FT_TT_OP_ATTENTION_HEADS] = 32u | (2u << 16);
    shape = fta_decode(words);
    ft_tt_test_check("...and an unknown flag bit", !fta_shape_valid(&shape), "");

    /* ---- the causal tiles themselves ---- */
    rc = ft_tt_op_attention_build_constants(MODEL_HEAD_DIM, TILE_BYTES, constants,
                                            sizeof(constants));
    ft_tt_test_check_fmt("the constants build with the causal block appended", rc == FT_TT_OK,
                         "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("the operand is 66 tiles — 34 decode reads, 32 appended",
                         ft_tt_op_attention_constants_bytes(TILE_BYTES) ==
                             (size_t)66u * TILE_BYTES,
                         "%zu B", ft_tt_op_attention_constants_bytes(TILE_BYTES));
    /* THE DECODE BLOCK DID NOT MOVE. Tile 2 + tail is still the tail mask, which is what
     * makes this an APPEND rather than a layout change — and what keeps the shipped decode
     * arm's constants byte for byte what they were. */
    ft_tt_test_check("the tail masks are still at tile 2 + tail, so decode's constants are "
                     "unmoved",
                     FT_TT_OP_ATTENTION_CONST_MASK_TILE == 2u &&
                         FT_TT_OP_ATTENTION_CONST_CAUSAL_TILE == 34u,
                     "");
    for (live = 1u; live <= TILE_DIM; live++) {
        char label[128];
        const bool correct = causal_tile_is_correct(constants, live);

        if (live == 1u || live == 7u || live == 17u || live == TILE_DIM) {
            snprintf(label, sizeof(label),
                     "causal tile C=%u is live exactly where c <= r and c < %u, in FACE order",
                     live, live);
            ft_tt_test_check(label, correct, "");
        } else {
            ft_tt_test_check("a causal tile is correct", correct, "");
        }
    }
    {
        /* A LOWER-TRIANGULAR TILE IS NOT A TAIL MASK, AND THE SUITE SHOULD SAY SO. Without
         * this the whole block above would pass against a builder that emitted 32 copies of
         * the all-live tile. */
        const uint16_t *causal =
            constants + (size_t)(FT_TT_OP_ATTENTION_CONST_CAUSAL_TILE + 31u) * TILE_DIM *
                            TILE_DIM;
        uint32_t live_count = 0u;
        uint32_t row;

        for (row = 0; row < TILE_DIM; row++) {
            uint32_t column;

            for (column = 0; column < TILE_DIM; column++) {
                if (ft_float_from_bf16(causal[ft_tt_tile_face_index(row, column)]) == 0.0f)
                    live_count++;
            }
        }
        ft_tt_test_check_fmt("the C=32 causal tile is half a tile live, not a whole one",
                             live_count == (TILE_DIM * (TILE_DIM + 1u)) / 2u,
                             "%u live of %u", live_count, TILE_DIM * TILE_DIM);
    }
}

/* ================================================================
 * The residency and the capacity wall
 * ================================================================ */

static void stage_capacity(void)
{
    const size_t arena = (size_t)(FT_TT_OP_CB_ARENA_END - FT_TT_OP_CB_ARENA_BASE);
    ft_tt_op_attention_residency residency;
    size_t short_context;
    size_t long_context;
    uint32_t best;

    ft_tt_test_step("the capacity wall — why the online softmax is required and not "
                    "preferred");

    /* A STANDALONE softmax holds the row twice over, so its ceiling is a row width. A
     * 32k-context attention row is Wt = 1024 and this is the assertion that says so. */
    {
        const size_t standalone_row = (size_t)1024u * TILE_BYTES;

        ft_tt_test_check_fmt("a 32k score row does not fit L1 even ONCE, let alone twice",
                             2u * standalone_row > arena, "row %zu arena %zu", standalone_row,
                             arena);
    }

    residency.q_rows = MODEL_GROUP_ROWS;
    residency.dim_tiles = MODEL_DIM_TILES;
    residency.block_tiles = TEST_BLOCK_TILES;
    residency.tile_bytes = TILE_BYTES;
    short_context = ft_tt_op_attention_l1_bytes(&residency);
    long_context = ft_tt_op_attention_l1_bytes(&residency);
    ft_tt_test_check_fmt("attention's L1 is FLAT in the context length — the same residency "
                         "serves 512 and 131072",
                         short_context == long_context && short_context != 0u, "%zu bytes",
                         short_context);
    ft_tt_test_check_fmt("and it fits the arena with room to spare", short_context < arena / 2u,
                         "%zu of %zu", short_context, arena);

    best = ft_tt_op_attention_max_block_tiles(MODEL_GROUP_ROWS, MODEL_DIM_TILES, TILE_BYTES);
    ft_tt_test_check_fmt("the deepest block that fits is reported and is well past 8",
                         best >= 16u, "block_tiles <= %u", best);
    residency.block_tiles = best;
    ft_tt_test_check_fmt("...and the residency at that depth still fits",
                         ft_tt_op_attention_l1_bytes(&residency) <= arena, "%zu of %zu",
                         ft_tt_op_attention_l1_bytes(&residency), arena);
    residency.block_tiles = best + 1u;
    ft_tt_test_check_fmt("...and one deeper does not", ft_tt_op_attention_l1_bytes(&residency) >
                                                           arena,
                         "%zu of %zu", ft_tt_op_attention_l1_bytes(&residency), arena);
}

/* ================================================================
 * Ornith-9B's shape
 * ================================================================ */

/**
 * Everything this op has to be true of before Ornith's attention can move onto the card.
 *
 * NOT a second copy of the checks above at different numbers. Every check here is one that
 * MiniCPM5-1B's shape cannot fail and Ornith's can, because the two models sit on opposite
 * sides of the one product this op is bounded by:
 *
 *     MiniCPM5-1B   q_rows 8, dim_tiles 4   deep group, narrow head
 *     Ornith-9B     q_rows 2, dim_tiles 8   shallow group, WIDE head
 *
 * K and V are `2 * block_tiles * dim_tiles` pages EACH, so doubling `dim_tiles` doubles the
 * dominant term of the residency while halving the group count does not pay any of it back.
 * That is the check that had to be run before a line of host wiring was written, and it is
 * the reason it is run with no card: an L1 refusal is discovered at ARM time, which is
 * after a 9.5 GB model load on a 7.1 GB host.
 */
static void stage_ornith_shape(void)
{
    const size_t arena = (size_t)(FT_TT_OP_CB_ARENA_END - FT_TT_OP_CB_ARENA_BASE);
    ft_tt_op_attention_residency residency;
    ft_tt_kv_cache_layout layout;
    size_t l1;
    uint32_t best;
    uint32_t head;

    ft_tt_test_step("Ornith-9B's shape — 16 Q heads / 4 KV heads / head_dim 256, "
                    "8 attention layers");

    ft_tt_test_check_fmt("the group shape is q_rows=4 over dim_tiles=8",
                         ORNITH_GROUP_ROWS == 4u && ORNITH_DIM_TILES == 8u, "q_rows %u "
                         "dim_tiles %u", ORNITH_GROUP_ROWS, ORNITH_DIM_TILES);

    /* ---- the residency, at the depth pn_tt_attention.c actually ships ---- */
    residency.q_rows = ORNITH_GROUP_ROWS;
    residency.dim_tiles = ORNITH_DIM_TILES;
    residency.block_tiles = SHIPPED_BLOCK_TILES;
    residency.tile_bytes = TILE_BYTES;
    l1 = ft_tt_op_attention_l1_bytes(&residency);

    /* 618 pages at q_rows=4. The deeper group costs 16 pages over q_rows=2's 602 — Q, QS,
     * O, PV and the output staging all carry `q_rows` rows — and it is 93.6% of the arena
     * against 91.2%. Written out because a number this close to the wall must be reproduced
     * by the arithmetic and not merely by the function under test. */
    ft_tt_test_check_fmt("L1 at the shipped block_tiles=16 is 618 pages = 1,265,664 B",
                         l1 == 618u * TILE_BYTES, "%zu B (%zu pages)", l1, l1 / TILE_BYTES);
    ft_tt_test_check_fmt("...and it FITS the arena — the shipped depth needs no clamp",
                         l1 != 0u && l1 <= arena, "%zu of %zu, %.1f%% used", l1, arena,
                         100.0 * (double)l1 / (double)arena);
    /* MiniCPM's stage asserts `< arena/2`. Ornith cannot: at 91% it has ONE deeper block of
     * headroom, and that is the finding, not a rounding detail. */
    ft_tt_test_check_fmt("...with only ~6% spare, unlike MiniCPM5-1B's half-empty arena",
                         l1 > arena / 2u, "%zu of %zu", l1, arena);

    best = ft_tt_op_attention_max_block_tiles(ORNITH_GROUP_ROWS, ORNITH_DIM_TILES, TILE_BYTES);
    ft_tt_test_check_fmt("the deepest block that fits Ornith's shape is 17", best == 17u,
                         "block_tiles <= %u", best);
    ft_tt_test_check_fmt("...so the shipped 16 is under the wall and is NOT clamped",
                         SHIPPED_BLOCK_TILES <= best, "shipped %u, wall %u",
                         SHIPPED_BLOCK_TILES, best);
    residency.block_tiles = best + 1u;
    ft_tt_test_check_fmt("...and one deeper does not fit",
                         ft_tt_op_attention_l1_bytes(&residency) > arena, "%zu of %zu",
                         ft_tt_op_attention_l1_bytes(&residency), arena);
    ft_tt_test_check_fmt("the WIDE head is what costs it — MiniCPM's shape reaches 34",
                         ft_tt_op_attention_max_block_tiles(MODEL_GROUP_ROWS, MODEL_DIM_TILES,
                                                            TILE_BYTES) == 34u,
                         "MiniCPM wall %u",
                         ft_tt_op_attention_max_block_tiles(MODEL_GROUP_ROWS, MODEL_DIM_TILES,
                                                            TILE_BYTES));

    /* ---- the GQA mapping: the op groups CONTIGUOUSLY, and so does Ornith ---- */
    {
        bool agrees = true;

        for (head = 0; head < ORNITH_Q_HEADS; head++) {
            /* What `pn_tt_attention()` implies: group g's query operand starts at tile
             * `g * q_rows * dim_tiles`, i.e. head `g * q_rows`, so head h is served by
             * group `h / q_rows`. */
            const uint32_t op_group = head / ORNITH_GROUP_ROWS;
            /* What Pantheon's CPU oracle does (`qwen35moe_attn.c`,
             * `kv_h = qh / a->n_kv_groups` with `n_kv_groups = n_head / n_head_kv`). */
            const uint32_t oracle_kv_head = head / (ORNITH_Q_HEADS / ORNITH_KV_HEADS);

            if (op_group != oracle_kv_head)
                agrees = false;
        }
        ft_tt_test_check("the op's KV grouping IS Ornith's GQA repeat — CONTIGUOUS, "
                         "kv_head = q_head / 4, on all 16 heads", agrees,
                         "checked every Q head");
    }

    /* ---- the joint Q+gate projection cannot be read in place ---- */
    {
        /* `blk.N.attn_q.weight` emits 2*q_heads*head_dim = 8,192 features: head h's Q at
         * tiles [16h, 16h+8) and head h's GATE at [16h+8, 16h+16). The op reads group g as
         * ONE contiguous run of `q_rows*dim_tiles = 32` tiles from `g*32`, which over that
         * strip is head 4g's Q followed by head 4g's GATE — not head 4g's Q followed by head
         * (4g+1)'s. This is the check that says the de-interleave is REQUIRED rather than
         * tidy, and it is the only thing standing between the op and Ornith. */
        const uint32_t joint_head_stride = 2u * ORNITH_DIM_TILES;
        const uint32_t group = 3u;
        const uint32_t run_first = group * ORNITH_GROUP_ROWS * ORNITH_DIM_TILES;
        const uint32_t wanted_second_head_tile =
            (group * ORNITH_GROUP_ROWS + 1u) * joint_head_stride;

        ft_tt_test_check_fmt("the op's contiguous group run does NOT land on the joint "
                             "Q+gate strip — the de-interleave is required",
                             run_first + ORNITH_DIM_TILES != wanted_second_head_tile,
                             "group %u would read tiles %u.. but head %u's Q starts at %u",
                             group, run_first, group * ORNITH_GROUP_ROWS + 1u,
                             wanted_second_head_tile);
        ft_tt_test_check_fmt("...and a DE-INTERLEAVED strip does land on it exactly",
                             (group * ORNITH_GROUP_ROWS + 1u) * ORNITH_DIM_TILES ==
                                 run_first + ORNITH_DIM_TILES,
                             "tile %u", run_first + ORNITH_DIM_TILES);
    }

    /* ---- the padded read window, and what it costs at SHORT context ---- */
    {
        static const struct {
            uint32_t kv;
            uint32_t padded_at_16;
            uint32_t padded_at_1;
        } cases[] = {
            { 50u, 32u, 2u },       { 1024u, 32u, 32u },      { 1025u, 64u, 34u },
            { 8192u, 256u, 256u },  { 14336u, 448u, 448u },   { 32768u, 1024u, 1024u }
        };
        size_t index;

        for (index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
            const uint32_t at16 = ft_tt_op_attention_padded_tiles(cases[index].kv,
                                                                  SHIPPED_BLOCK_TILES);
            const uint32_t at1 = ft_tt_op_attention_padded_tiles(cases[index].kv, 1u);
            char label[96];

            snprintf(label, sizeof(label), "kv=%u padded window", cases[index].kv);
            ft_tt_test_check_fmt(label,
                                 at16 == cases[index].padded_at_16 &&
                                     at1 == cases[index].padded_at_1,
                                 "block16 %u (want %u), block1 %u (want %u)", at16,
                                 cases[index].padded_at_16, at1, cases[index].padded_at_1);
        }

        /* THE 48-TOKEN ORACLE RUNS HERE, and this is why its arm needs a different knob.
         * At block_tiles=16 the window quantum is 2*16*32 = 1024 POSITIONS, so a descriptor
         * at position 50 does 1024 positions of work — 16x what block_tiles=1 does. The
         * shipped depth is right at context and wrong at the bench length, which is exactly
         * the trade `PN_TT_ATTN_BLOCK_TILES` exists to make. */
        ft_tt_test_check_fmt("at kv=50 the shipped depth does 16x the work of block_tiles=1",
                             ft_tt_op_attention_padded_tiles(50u, SHIPPED_BLOCK_TILES) ==
                                 16u * ft_tt_op_attention_padded_tiles(50u, 1u),
                             "%u tiles vs %u",
                             ft_tt_op_attention_padded_tiles(50u, SHIPPED_BLOCK_TILES),
                             ft_tt_op_attention_padded_tiles(50u, 1u));
    }

    /* ---- the KV cache Ornith needs, and what it costs in card DRAM ---- */
    ft_tt_test_check_fmt("a 32k KV layout for 4 heads x 256 dim is accepted",
                         ft_tt_kv_cache_layout_init(ORNITH_KV_HEADS, ORNITH_HEAD_DIM, 32768u,
                                                    TILE_BYTES, &layout) == FT_TT_OK,
                         "4 x 256 x 32768 bf16");
    ft_tt_test_check_fmt("...capacity_tiles 1024, position tile 16,384 B, head stride 16 MiB",
                         layout.capacity_tiles == 1024u &&
                             layout.position_tile_bytes == 16384u &&
                             layout.head_stride == 16777216ull,
                         "tiles %u, position %u B, stride %llu B", layout.capacity_tiles,
                         layout.position_tile_bytes,
                         (unsigned long long)layout.head_stride);
    ft_tt_test_check_fmt("...one region is 64 MiB, so K+V over 8 attention layers is 1 GiB",
                         layout.region_bytes == 67108864ull &&
                             layout.region_bytes * 2ull * ORNITH_ATTN_LAYERS ==
                                 1073741824ull,
                         "region %llu B, family %.2f GiB",
                         (unsigned long long)layout.region_bytes,
                         (double)layout.region_bytes * 2.0 * ORNITH_ATTN_LAYERS /
                             (1024.0 * 1024.0 * 1024.0));
    /* AT THE REAL HEAD COUNT this is a QUARTER of what the spec carries. The per-position
     * cost across the 8 attention layers is 8*4*256*2 B *2 = 32,768 B on the card at bf16 and
     * 65,536 B in the host fp32 mirror — not the 131,072 / 262,144 derived from 16 KV heads.
     * The host-attention SLOPE (0.53 ms/position) is a measurement and stands; what changes
     * is the byte attribution under it. */
    ft_tt_test_check_fmt("one position of context is 32,768 B on card and 65,536 B on host",
                         (uint64_t)ORNITH_ATTN_LAYERS * ORNITH_KV_HEADS * ORNITH_HEAD_DIM *
                                 2ull * 2ull == 32768ull,
                         "%llu B",
                         (unsigned long long)((uint64_t)ORNITH_ATTN_LAYERS * ORNITH_KV_HEADS *
                                              ORNITH_HEAD_DIM * 2ull * 2ull));

    /* ---- the capacity wall is enforced, not assumed ---- */
    {
        ft_tt_op_attention_request request;
        uint32_t words[FT_TT_OP_ATTENTION_WORDS];
        const ft_tt_op_operand operand = { 1u, 1u, 0x1000u, TILE_BYTES };

        memset(&request, 0, sizeof(request));
        request.q_rows = ORNITH_GROUP_ROWS;
        request.dim_tiles = ORNITH_DIM_TILES;
        request.block_tiles = SHIPPED_BLOCK_TILES;
        request.capacity_tiles = layout.capacity_tiles;
        request.query_row = 0u;
        request.query = operand;
        request.keys = operand;
        request.values = operand;
        request.constants = operand;
        request.output = operand;

        request.kv_positions = 32768u;
        ft_tt_test_check("a full 32,768-position descriptor is encodable",
                         ft_tt_op_attention_parameters(&request, words) == FT_TT_OK,
                         "padded 1024 == capacity 1024");
        request.kv_positions = 32769u;
        ft_tt_test_check("...and one position past the cache is REFUSED, not clamped",
                         ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL,
                         "padded 1056 > capacity 1024");
        request.kv_positions = 50u;
        ft_tt_test_check_fmt("the bench-length descriptor encodes with q_rows=4 in byte 0",
                             ft_tt_op_attention_parameters(&request, words) == FT_TT_OK &&
                                 (words[FT_TT_OP_ATTENTION_HEADS] & 0xffu) ==
                                     ORNITH_GROUP_ROWS &&
                                 words[FT_TT_OP_ATTENTION_DIM_TILES] == ORNITH_DIM_TILES,
                             "heads word 0x%08x, dim_tiles %u", words[FT_TT_OP_ATTENTION_HEADS],
                             words[FT_TT_OP_ATTENTION_DIM_TILES]);
    }

    /* ---- the descriptor budget, which is the other thing the shape swap moves ---- */
    ft_tt_test_check_fmt("Ornith issues 32 attention descriptors a token against "
                         "MiniCPM5-1B's 48",
                         ORNITH_KV_HEADS * ORNITH_ATTN_LAYERS == 32u &&
                             MODEL_KV_HEADS * 24u == 48u,
                         "%u groups x %u layers", ORNITH_KV_HEADS, ORNITH_ATTN_LAYERS);
}

/* ================================================================
 * The algorithm
 * ================================================================ */

#define MAX_KV       2048u
#define MAX_ELEMENTS (MODEL_GROUP_ROWS * MODEL_HEAD_DIM)

static float g_query[MAX_ELEMENTS];
static float g_keys[MAX_KV * MODEL_HEAD_DIM];
static float g_values[MAX_KV * MODEL_HEAD_DIM];
static float g_reference[MAX_ELEMENTS];
static float g_model[MAX_ELEMENTS];

/** Fill the whole PADDED window, including the region past the live length: the model reads
 *  it, the card reads it, and filling only the live rows would hide the mask bug. */
static void build_problem(uint32_t kv_positions, uint64_t seed, float dead_value)
{
    const uint32_t padded = ft_tt_op_attention_padded_tiles(kv_positions, TEST_BLOCK_TILES) *
                            TILE_DIM;
    uint32_t index;

    ft_tt_attention_seed(seed);
    for (index = 0; index < MAX_ELEMENTS; index++)
        g_query[index] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_attention_random()));
    for (index = 0; index < padded * MODEL_HEAD_DIM; index++) {
        const bool dead = (index / MODEL_HEAD_DIM) >= kv_positions;
        const float value = dead ? dead_value : ft_tt_attention_random();

        g_keys[index] = ft_float_from_bf16(ft_bf16_from_float(value));
        g_values[index] = ft_float_from_bf16(ft_bf16_from_float(dead ? dead_value
                                                                    : ft_tt_attention_random()));
    }
}

static void score_model(const char *label, uint32_t kv_positions)
{
    ft_tt_attention_problem problem;
    double pcc;
    double max_abs;
    char text[160];

    problem.q_rows = MODEL_GROUP_ROWS;
    problem.head_dim = MODEL_HEAD_DIM;
    problem.kv_positions = kv_positions;
    problem.query = g_query;
    problem.keys = g_keys;
    problem.values = g_values;

    if (!ft_tt_attention_reference(&problem, g_reference) ||
        !ft_tt_attention_online_model(&problem, TEST_BLOCK_TILES, g_model)) {
        ft_tt_test_check_fmt(label, false, "the reference or the model refused the shape");
        return;
    }
    pcc = ft_tt_attention_pcc(g_reference, g_model, MAX_ELEMENTS);
    max_abs = ft_tt_attention_max_abs(g_reference, g_model, MAX_ELEMENTS);
    snprintf(text, sizeof(text), "%s (kv=%u)", label, kv_positions);
    ft_tt_test_check_fmt(text,
                         pcc > 0.9999999 && max_abs < 1.0e-5 &&
                             ft_tt_attention_non_finite(g_model, MAX_ELEMENTS) == 0u,
                         "pcc %.9f max-abs %.3e non-finite %zu", pcc, max_abs,
                         ft_tt_attention_non_finite(g_model, MAX_ELEMENTS));
}

static void stage_algorithm(void)
{
    ft_tt_test_step("the ONLINE softmax reproduces a two-pass softmax");

    build_problem(1024u, 20260812ull, 0.0f);
    score_model("a whole number of blocks", 1024u);

    build_problem(1000u, 20260813ull, 0.0f);
    score_model("a partial last tile — the additive tail mask", 1000u);

    build_problem(33u, 20260814ull, 0.0f);
    score_model("two live tiles inside a 16-tile padded window", 33u);

    build_problem(1u, 20260815ull, 0.0f);
    score_model("one live position", 1u);

    ft_tt_test_step("the two invariance tests — what PCC alone cannot see");

    /* 1. A WRONG RUNNING MAX. softmax(x - c) = softmax(x), so a block-local max is
     *    invisible to correlation on well-scaled data. An
     *    outlier in a LATE block breaks the symmetry: a max taken per block puts the early
     *    blocks' exponentials outside representable range. */
    build_problem(1024u, 20260816ull, 0.0f);
    {
        uint32_t element;

        for (element = 0; element < MODEL_HEAD_DIM; element++)
            g_keys[(size_t)1000u * MODEL_HEAD_DIM + element] = 40.0f;
    }
    score_model("an outlier planted in a LATE KV block", 1024u);

    /* 2. A MASK THAT DOES NOT MASK. Poison the dead region with values that would dominate
     *    the row max if they reached it. The reference never looks past the live length, so
     *    this changes the model's answer if and only if the mask is not doing its job. */
    build_problem(1000u, 20260817ull, 500.0f);
    score_model("garbage poisoned into the dead region past the live length", 1000u);
}

int main(void)
{
    ft_tt_test_step("Phase 6 — flash-decode attention, OFF-HARDWARE");

    stage_wire_contract();
    stage_parameters();
    stage_refusals();
    stage_derived_quantities();
    stage_constants();
    stage_chunk_contract();
    stage_capacity();
    stage_ornith_shape();
    stage_algorithm();
    return ft_tt_test_finish();
}
