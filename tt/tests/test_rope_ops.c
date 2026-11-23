/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_rope_ops.c
 * @brief OFF-HARDWARE gate for ROPE — partial NeoX rotary position embedding. Runs without
 *   a card, and is what has to be green before the hardware pass is worth a core.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS CAN PROVE, AND WHAT IT CANNOT
 * ---------------------------------------------------------------------------
 * It CAN prove:
 *   - the host and the KERNEL agree on every wire constant, by PARSING the device header
 *     rather than by a comment claiming they match (dev notes §4);
 *   - the parameter block puts every field in the word the kernel reads;
 *   - the TABLE is tilized with the same face map the activations use — checked element by
 *     element through ft_tt_tile_face_index(), which is the only check that can see it,
 *     because a face-permuted table is a wrong ANGLE and not a permuted answer;
 *   - the ANGLE is right, read directly out of a rotated basis vector rather than inferred
 *     from a correlation;
 *   - the ARRANGEMENT OF PASSES computes the op, and clears PCC >= 0.99 with bf16 at every
 *     circular-buffer boundary — a prediction of the card's score;
 *   - the test DATA can see a swapped pair, asserted rather than assumed.
 *
 * It CANNOT prove the kernel implements the model. Nothing off-hardware can (dev notes §3:
 * "off-hardware suites cannot cover the copy path"), and here it cannot cover the FPU, the
 * packer, the real face order or the CB credit protocol either.
 *
 * ---------------------------------------------------------------------------
 * TWO KINDS OF CHECK, ON PURPOSE
 * ---------------------------------------------------------------------------
 * A rotation preserves the norm, so no norm-based check can see a wrong angle
 * (docs/tt_bringup_log.md §22.2 — the general form of the softmax shift-invariance trap).
 * And PCC is blind to a consistent permutation (§9), which is exactly the error a face-order
 * or partner-order mistake produces. So:
 *   - STRUCTURE is scored by BYTES, against a synthetic table of exactly-representable
 *     values where every intermediate is exact in bf16, so the comparison has no tolerance
 *     to hide in;
 *   - the ANGLE is scored SEPARATELY and explicitly, by rotating (1, 0) and reading (cos,
 *     sin) straight out of the answer.
 *
 * Environment:
 *   FT_TT_KERNEL_DIR  where the device headers live (default ../kernels)
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_rope.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "ft_tt_rope_reference.h"
#include "ft_tt_test_report.h"

#define PCC_ACCEPT 0.99

/** Where the device headers live, so the wire-constant pin reads the real file. */
#define KERNEL_DIRECTORY_ENVIRONMENT "FT_TT_KERNEL_DIR"
#define KERNEL_DIRECTORY_DEFAULT "../kernels"
#define PATH_MAX_BYTES 512u

#define TILE_DIM   FT_TT_OP_TILE_DIM
#define TILE_BYTES FT_TT_OP_TILE_BYTES_BF16

/** MiniCPM5-1B, read from the GGUF header on the card host. */
#define MODEL_HEAD_DIM FT_TT_OP_ROPE_MINICPM5_HEAD_DIM
#define MODEL_ROPE_DIM FT_TT_OP_ROPE_MINICPM5_ROPE_DIM
#define MODEL_THETA    FT_TT_OP_ROPE_MINICPM5_THETA

static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];

/* ================================================================
 * Reading a constant out of a device header
 * ================================================================ */

/** The value of `enum { <name> = <value> };` in `path` -- the C/C++-shared form the
 *  device headers use since plan Phase 6.6, read out of the real header rather than
 *  retyped here. A missing constant is a
 *  FAILURE and never a default — a default would pass against a header that no longer
 *  declares the field. */
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

static void test_wire_constants(void)
{
    ft_tt_test_step("the host and the KERNEL agree on every wire word — PARSED, not claimed");

    check_device_constant("resident_rope_protocol.h", "kRopeTileCount",
                          (uint32_t)FT_TT_OP_ROPE_TILE_COUNT);
    check_device_constant("resident_rope_protocol.h", "kRopeHeadTiles",
                          (uint32_t)FT_TT_OP_ROPE_HEAD_TILES);
    check_device_constant("resident_rope_protocol.h", "kRopeRotTiles",
                          (uint32_t)FT_TT_OP_ROPE_ROT_TILES);
    check_device_constant("resident_rope_protocol.h", "kRopeHeads",
                          (uint32_t)FT_TT_OP_ROPE_HEADS);
    check_device_constant("resident_rope_protocol.h", "kRopeDepth",
                          (uint32_t)FT_TT_OP_ROPE_DEPTH);
    check_device_constant("resident_rope_protocol.h", "kRopeAX", (uint32_t)FT_TT_OP_ROPE_A_X);
    check_device_constant("resident_rope_protocol.h", "kRopeAY", (uint32_t)FT_TT_OP_ROPE_A_Y);
    check_device_constant("resident_rope_protocol.h", "kRopeABase",
                          (uint32_t)FT_TT_OP_ROPE_A_BASE);
    check_device_constant("resident_rope_protocol.h", "kRopeTableX",
                          (uint32_t)FT_TT_OP_ROPE_TABLE_X);
    check_device_constant("resident_rope_protocol.h", "kRopeTableY",
                          (uint32_t)FT_TT_OP_ROPE_TABLE_Y);
    check_device_constant("resident_rope_protocol.h", "kRopeTableBase",
                          (uint32_t)FT_TT_OP_ROPE_TABLE_BASE);
    check_device_constant("resident_rope_protocol.h", "kRopeCX", (uint32_t)FT_TT_OP_ROPE_C_X);
    check_device_constant("resident_rope_protocol.h", "kRopeCY", (uint32_t)FT_TT_OP_ROPE_C_Y);
    check_device_constant("resident_rope_protocol.h", "kRopeCBase",
                          (uint32_t)FT_TT_OP_ROPE_C_BASE);
    check_device_constant("resident_rope_protocol.h", "kRopeWords",
                          (uint32_t)FT_TT_OP_ROPE_WORDS);
    check_device_constant("resident_queue_protocol.h", "kOpRope",
                          (uint32_t)FT_TT_QUEUE_OP_ROPE);
    ft_tt_test_check_fmt("the parameter block fits one queue slot",
                         (uint32_t)FT_TT_OP_ROPE_WORDS == FT_TT_QUEUE_PARAM_WORDS, "%u vs %u",
                         (uint32_t)FT_TT_OP_ROPE_WORDS, FT_TT_QUEUE_PARAM_WORDS);
}

/* ================================================================
 * The parameter block
 * ================================================================ */

/** A request with a distinct value in every field, so a field written into the wrong word is
 *  visible rather than plausible. */
static void fill_request(ft_tt_op_rope_request *request, uint32_t row_blocks, uint32_t heads,
                         uint32_t head_dim, uint32_t rope_dim, uint32_t first_position)
{
    memset(request, 0, sizeof(*request));
    request->row_blocks = row_blocks;
    request->heads = heads;
    request->head_dim = head_dim;
    request->rope_dim = rope_dim;
    request->first_position = first_position;
    request->prefetch_tiles = 2u;
    request->input.x = 11u;
    request->input.y = 12u;
    request->input.address = 0x00100000u;
    request->input.tile_bytes = TILE_BYTES;
    request->table.x = 21u;
    request->table.y = 22u;
    request->table.address = 0x00200000u;
    request->table.tile_bytes = TILE_BYTES;
    request->output.x = 31u;
    request->output.y = 32u;
    request->output.address = 0x00300000u;
    request->output.tile_bytes = TILE_BYTES;
}

static void test_parameter_block(void)
{
    ft_tt_op_rope_request request;
    uint32_t words[FT_TT_OP_ROPE_WORDS];
    uint32_t expected_table;
    int rc;

    ft_tt_test_step("the parameter block puts every field in the word the kernel reads");

    fill_request(&request, 2u, 16u, MODEL_HEAD_DIM, MODEL_ROPE_DIM, 64u);
    rc = ft_tt_op_rope_parameters(&request, words);
    ft_tt_test_check_fmt("MiniCPM5-1B Q, 64 tokens, encodes", rc == FT_TT_OK, "rc=%d", rc);

    ft_tt_test_check_fmt("tile_count = row_blocks * heads * head_tiles",
                         words[FT_TT_OP_ROPE_TILE_COUNT] == 2u * 16u * 4u, "%u",
                         words[FT_TT_OP_ROPE_TILE_COUNT]);
    ft_tt_test_check_fmt("head_tiles = head_dim / 32",
                         words[FT_TT_OP_ROPE_HEAD_TILES] == MODEL_HEAD_DIM / TILE_DIM, "%u",
                         words[FT_TT_OP_ROPE_HEAD_TILES]);
    ft_tt_test_check_fmt("rot_tiles = rot / 32",
                         words[FT_TT_OP_ROPE_ROT_TILES] == MODEL_ROPE_DIM / TILE_DIM, "%u",
                         words[FT_TT_OP_ROPE_ROT_TILES]);
    ft_tt_test_check_fmt("heads travels", words[FT_TT_OP_ROPE_HEADS] == 16u, "%u",
                         words[FT_TT_OP_ROPE_HEADS]);
    ft_tt_test_check_fmt("the activation coordinate travels",
                         words[FT_TT_OP_ROPE_A_X] == 11u && words[FT_TT_OP_ROPE_A_Y] == 12u &&
                             words[FT_TT_OP_ROPE_A_BASE] == 0x00100000u,
                         "(%u,%u)+0x%X", words[FT_TT_OP_ROPE_A_X], words[FT_TT_OP_ROPE_A_Y],
                         words[FT_TT_OP_ROPE_A_BASE]);
    ft_tt_test_check_fmt("the output coordinate travels",
                         words[FT_TT_OP_ROPE_C_X] == 31u && words[FT_TT_OP_ROPE_C_Y] == 32u &&
                             words[FT_TT_OP_ROPE_C_BASE] == 0x00300000u,
                         "(%u,%u)+0x%X", words[FT_TT_OP_ROPE_C_X], words[FT_TT_OP_ROPE_C_Y],
                         words[FT_TT_OP_ROPE_C_BASE]);

    /* The driver ADDS the row-block offset so a caller cannot forget to. Position 64 is row
     * block 2, and a row block is 2*rot_tiles tiles. */
    expected_table = 0x00200000u + 2u * (2u * (MODEL_ROPE_DIM / TILE_DIM)) * TILE_BYTES;
    ft_tt_test_check_fmt("the table base is advanced to the position's row block",
                         words[FT_TT_OP_ROPE_TABLE_BASE] == expected_table, "0x%X vs 0x%X",
                         words[FT_TT_OP_ROPE_TABLE_BASE], expected_table);

    ft_tt_test_check_fmt("the depth divides head_tiles AND rot_tiles",
                         words[FT_TT_OP_ROPE_DEPTH] != 0u &&
                             (words[FT_TT_OP_ROPE_HEAD_TILES] % words[FT_TT_OP_ROPE_DEPTH]) == 0u &&
                             (words[FT_TT_OP_ROPE_ROT_TILES] % words[FT_TT_OP_ROPE_DEPTH]) == 0u,
                         "depth %u", words[FT_TT_OP_ROPE_DEPTH]);
}

static void test_refusals(void)
{
    ft_tt_op_rope_request request;
    uint32_t words[FT_TT_OP_ROPE_WORDS];

    ft_tt_test_step("the shapes the FACE-SPACE argument cannot express are REFUSED");

    /* rot = 96 -> rot_tiles 3, half = 48 features = one and a HALF tiles, so the partner
     * element sits in a different face COLUMN. Buildable with a face-granular gather; not
     * built; refused rather than approximated. */
    fill_request(&request, 1u, 2u, 128u, 96u, 0u);
    ft_tt_test_check_fmt("an odd rot_tiles (half lands mid-tile) is refused",
                         ft_tt_op_rope_parameters(&request, words) == FT_TT_EINVAL, "rot 96");

    /* rot = 48 -> not even a whole number of tiles. */
    fill_request(&request, 1u, 2u, 128u, 48u, 0u);
    ft_tt_test_check_fmt("a rot that is not a multiple of 32 is refused",
                         ft_tt_op_rope_parameters(&request, words) == FT_TT_EINVAL, "rot 48");

    fill_request(&request, 1u, 2u, 100u, 64u, 0u);
    ft_tt_test_check_fmt("a head_dim that is not a multiple of 32 is refused",
                         ft_tt_op_rope_parameters(&request, words) == FT_TT_EINVAL,
                         "head_dim 100");

    /* THE ROW-PHASE CONTRACT: one row block may start anywhere (decode), several may not. */
    fill_request(&request, 1u, 2u, 128u, 128u, 7u);
    ft_tt_test_check_fmt("ONE row block at an unaligned position is ACCEPTED (decode)",
                         ft_tt_op_rope_parameters(&request, words) == FT_TT_OK, "position 7");
    fill_request(&request, 2u, 2u, 128u, 128u, 7u);
    ft_tt_test_check_fmt("TWO row blocks at an unaligned position are refused",
                         ft_tt_op_rope_parameters(&request, words) == FT_TT_EINVAL,
                         "position 7, 2 blocks");

    fill_request(&request, 0u, 2u, 128u, 128u, 0u);
    ft_tt_test_check_fmt("a zero row-block count is refused",
                         ft_tt_op_rope_parameters(&request, words) == FT_TT_EINVAL,
                         "row_blocks 0");

    ft_tt_test_check_fmt("rope_dim 0 means the whole head",
                         ft_tt_op_rope_rotary_width(128u, 0u) == 128u, "%u",
                         ft_tt_op_rope_rotary_width(128u, 0u));
    ft_tt_test_check_fmt("rope_dim past head_dim means the whole head",
                         ft_tt_op_rope_rotary_width(128u, 256u) == 128u, "%u",
                         ft_tt_op_rope_rotary_width(128u, 256u));
    ft_tt_test_check_fmt("rope_dim inside the head is the partial case",
                         ft_tt_op_rope_rotary_width(256u, 64u) == 64u, "%u",
                         ft_tt_op_rope_rotary_width(256u, 64u));
}

/* ================================================================
 * Circular buffers
 * ================================================================ */

static void test_cb_placement(void)
{
    ft_tt_op_rope_request request;
    ft_tt_launch_cb_entry entries[FT_TT_OP_ROPE_CB_COUNT];
    uint64_t mask = 0;
    unsigned i;
    unsigned j;
    bool disjoint = true;
    bool inside = true;
    int rc;

    ft_tt_test_step("the seven circular buffers fit the arena and do not overlap");

    fill_request(&request, 1u, 16u, MODEL_HEAD_DIM, MODEL_ROPE_DIM, 0u);
    request.prefetch_tiles = 4u;
    rc = ft_tt_op_rope_place_cbs(&request, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("the placement succeeds", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;

    for (i = 0; i < FT_TT_OP_ROPE_CB_COUNT; i++) {
        const uint32_t start = entries[i].address;
        const uint32_t end = start + entries[i].size_bytes;

        if (start < FT_TT_OP_CB_ARENA_BASE || end > FT_TT_OP_CB_ARENA_END)
            inside = false;
        for (j = i + 1u; j < FT_TT_OP_ROPE_CB_COUNT; j++) {
            const uint32_t other = entries[j].address;
            const uint32_t other_end = other + entries[j].size_bytes;

            if (start < other_end && other < end)
                disjoint = false;
        }
    }
    ft_tt_test_check("every buffer is inside the arena", inside, "arena 0x26000..0x170000");
    ft_tt_test_check("no two buffers overlap", disjoint, "seven buffers");
    ft_tt_test_check_fmt("the mask names seven buffers",
                         mask == ((1ull << FT_TT_OP_ROPE_CB_IN) | (1ull << FT_TT_OP_ROPE_CB_ROT) |
                                  (1ull << FT_TT_OP_ROPE_CB_COS) | (1ull << FT_TT_OP_ROPE_CB_SIN) |
                                  (1ull << FT_TT_OP_ROPE_CB_PROD_COS) |
                                  (1ull << FT_TT_OP_ROPE_CB_PROD_SIN) |
                                  (1ull << FT_TT_OP_ROPE_CB_OUT)),
                         "0x%llX", (unsigned long long)mask);
}

/* ================================================================
 * The table
 * ================================================================ */

/** The table artifact, read the way the CARD will read it: tile `tile` of the strip, element
 *  (row, column-within-tile), through ft_tt_tile_face_index(). */
static uint16_t table_element(const uint16_t *tiles, uint32_t row, uint32_t column)
{
    const uint32_t tile = column / TILE_DIM;
    const uint32_t within = ft_tt_tile_face_index(row, column % TILE_DIM);

    return tiles[(size_t)tile * TILE_DIM * TILE_DIM + within];
}

static void test_table_layout_and_angles(void)
{
    const uint32_t rot = MODEL_ROPE_DIM;
    const uint32_t rot_tiles = rot / TILE_DIM;
    const uint32_t half = rot / 2u;
    const uint32_t blocks = 2u;
    const size_t bytes = ft_tt_op_rope_table_bytes(blocks * TILE_DIM, rot_tiles, TILE_BYTES);
    uint16_t *table = malloc(bytes);
    uint16_t *cos_plain = malloc((size_t)TILE_DIM * rot * sizeof(uint16_t));
    uint16_t *sin_plain = malloc((size_t)TILE_DIM * rot * sizeof(uint16_t));
    uint32_t block;
    size_t mismatches = 0;
    size_t sign_wrong = 0;
    double worst_angle = 0.0;
    int rc;

    ft_tt_test_step("the TABLE: the face map, the sign, and the angle");

    if (!table || !cos_plain || !sin_plain) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }

    rc = ft_tt_op_rope_build_table(0u, blocks * TILE_DIM, MODEL_HEAD_DIM, MODEL_ROPE_DIM,
                                   MODEL_THETA, table, bytes);
    ft_tt_test_check_fmt("the table builds", rc == FT_TT_OK, "rc=%d, %zu bytes", rc, bytes);
    if (rc != FT_TT_OK)
        goto done;

    /* THE FACE-MAP CHECK, and the only one that can see this failure. Every element of the
     * artifact is compared against an independently computed plain value AT THE OFFSET THE
     * CARD WILL READ IT FROM. A table tilized flat row-major passes no part of this, while a
     * PCC over the op's output would partly absorb it. */
    for (block = 0; block < blocks; block++) {
        const uint16_t *cos_tiles =
            table + (size_t)block * 2u * rot_tiles * TILE_DIM * TILE_DIM;
        const uint16_t *sin_tiles = cos_tiles + (size_t)rot_tiles * TILE_DIM * TILE_DIM;
        uint32_t row;

        ft_tt_rope_reference_table(cos_plain, sin_plain, block * TILE_DIM, rot, MODEL_THETA);
        for (row = 0; row < TILE_DIM; row++) {
            uint32_t column;

            for (column = 0; column < rot; column++) {
                if (table_element(cos_tiles, row, column) != cos_plain[row * rot + column])
                    mismatches++;
                if (table_element(sin_tiles, row, column) != sin_plain[row * rot + column])
                    mismatches++;
            }
        }
    }
    ft_tt_test_check_fmt("every table element is where the FACE map puts it",
                         mismatches == 0u, "%zu mismatches of %u", mismatches,
                         blocks * TILE_DIM * rot * 2u);

    /* THE SIGN IS IN THE TABLE. Below the half it must be the NEGATIVE sine, above it the
     * positive one — that is what removes the branch from the kernel, so it is a property
     * worth a check of its own rather than one buried in an end-to-end score. */
    {
        uint32_t row;

        for (row = 1u; row < TILE_DIM; row++) {
            uint32_t j;

            for (j = 1u; j < half; j++) {
                const float low = ft_float_from_bf16(
                    table_element(table + (size_t)rot_tiles * TILE_DIM * TILE_DIM, row, j));
                const float high = ft_float_from_bf16(table_element(
                    table + (size_t)rot_tiles * TILE_DIM * TILE_DIM, row, j + half));

                if (low != -high)
                    sign_wrong++;
            }
        }
    }
    ft_tt_test_check_fmt("the sin table is NEGATED below the half and positive above",
                         sign_wrong == 0u, "%zu columns disagree", sign_wrong);

    /* THE ANGLE ITSELF, against a second, independently written formula. A rotation
     * preserves the norm, so nothing downstream of here can see a wrong angle. */
    {
        uint32_t row;

        for (row = 0; row < TILE_DIM; row++) {
            uint32_t j;

            for (j = 0; j < half; j += 7u) {
                const double angle = ft_tt_op_rope_angle(row, j, rot, MODEL_THETA);
                const double independent =
                    (double)row * pow(MODEL_THETA, -2.0 * (double)j / (double)rot);
                const double difference = fabs(angle - independent);

                if (difference > worst_angle)
                    worst_angle = difference;
            }
        }
    }
    ft_tt_test_check_fmt("the angle matches an independently written formula",
                         worst_angle == 0.0, "worst |difference| = %.3g", worst_angle);

    /* Position 0 is the identity: every angle is 0, so cos is 1 and sin is 0. It is the
     * control that says the table is not simply constant. */
    {
        const float cosine = ft_float_from_bf16(table_element(table, 0u, 5u));
        const float sine = ft_float_from_bf16(
            table_element(table + (size_t)rot_tiles * TILE_DIM * TILE_DIM, 0u, 5u));
        const float moved = ft_float_from_bf16(table_element(table, 3u, 5u));

        ft_tt_test_check_fmt("position 0 is the identity rotation",
                             cosine == 1.0f && sine == 0.0f, "cos %g sin %g", (double)cosine,
                             (double)sine);
        ft_tt_test_check_fmt("and a later row is NOT the identity", moved != 1.0f, "cos %g",
                             (double)moved);
    }

    ft_tt_test_check_fmt("the whole MiniCPM5-1B context table is 67.1 MB",
                         ft_tt_op_rope_table_bytes(FT_TT_OP_ROPE_MINICPM5_CONTEXT, rot_tiles,
                                                   TILE_BYTES) == 67108864u,
                         "%zu bytes",
                         ft_tt_op_rope_table_bytes(FT_TT_OP_ROPE_MINICPM5_CONTEXT, rot_tiles,
                                                   TILE_BYTES));
    ft_tt_test_check_fmt("the row-block offset indexes by position / 32",
                         ft_tt_op_rope_table_offset(100u, rot_tiles, TILE_BYTES) ==
                             3u * 2u * rot_tiles * TILE_BYTES,
                         "0x%X", ft_tt_op_rope_table_offset(100u, rot_tiles, TILE_BYTES));

done:
    free(table);
    free(cos_plain);
    free(sin_plain);
}

/* ================================================================
 * The op itself, through the device model
 * ================================================================ */

static double pearson(const float *a, const float *b, size_t count)
{
    double mean_a = 0.0;
    double mean_b = 0.0;
    double covariance = 0.0;
    double variance_a = 0.0;
    double variance_b = 0.0;
    size_t index;

    for (index = 0; index < count; index++) {
        mean_a += a[index];
        mean_b += b[index];
    }
    mean_a /= (double)count;
    mean_b /= (double)count;
    for (index = 0; index < count; index++) {
        const double da = a[index] - mean_a;
        const double db = b[index] - mean_b;

        covariance += da * db;
        variance_a += da * da;
        variance_b += db * db;
    }
    if (variance_a == 0.0 || variance_b == 0.0)
        return 0.0;
    return covariance / sqrt(variance_a * variance_b);
}

static double max_abs_difference(const float *a, const float *b, size_t count)
{
    double worst = 0.0;
    size_t index;

    for (index = 0; index < count; index++) {
        const double difference = fabs((double)a[index] - (double)b[index]);

        if (difference > worst)
            worst = difference;
    }
    return worst;
}

static uint64_t g_random_state = 0x9E3779B97F4A7C15ull;

static float random_value(void)
{
    /* Full-mantissa, in [-2, 2). Never small integers: this suite is about layout as much as
     * arithmetic, and a bf16-exact array cannot tell two layouts apart (dev notes §3). */
    g_random_state = g_random_state * 6364136223846793005ull + 1442695040888963407ull;
    return ((float)(int32_t)(g_random_state >> 33) / 1073741824.0f);
}

/**
 * Run the device model over one row block and score it against the independent reference.
 *
 * `swap_partner` computes the WRONG pairing on purpose, so the suite can assert its own data
 * is able to see a swapped pair rather than assuming so.
 */
static void model_one_shape(uint32_t heads, uint32_t head_dim, uint32_t rope_dim,
                            uint32_t position, const char *what)
{
    const uint32_t rot = ft_tt_op_rope_rotary_width(head_dim, rope_dim);
    const uint32_t rows = TILE_DIM;
    const size_t width = (size_t)heads * head_dim;
    const size_t count = (size_t)rows * width;
    float *source = malloc(count * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    float *swapped_actual = malloc(count * sizeof(float));
    uint16_t *in = malloc(count * sizeof(uint16_t));
    uint16_t *out = malloc(count * sizeof(uint16_t));
    uint16_t *swapped_out = malloc(count * sizeof(uint16_t));
    uint16_t *cos_plain = malloc((size_t)rows * rot * sizeof(uint16_t));
    uint16_t *sin_plain = malloc((size_t)rows * rot * sizeof(uint16_t));
    uint16_t *swapped_sin = malloc((size_t)rows * rot * sizeof(uint16_t));
    size_t index;
    size_t untouched_wrong = 0;
    char label[192];

    if (!source || !expected || !actual || !swapped_actual || !in || !out || !swapped_out ||
        !cos_plain || !sin_plain || !swapped_sin) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }

    for (index = 0; index < count; index++) {
        source[index] = random_value();
        in[index] = ft_bf16_from_float(source[index]);
        source[index] = ft_float_from_bf16(in[index]); /* what the card will actually see */
    }
    ft_tt_rope_reference_table(cos_plain, sin_plain, position, rot, MODEL_THETA);

    ft_tt_rope_model_apply(in, cos_plain, sin_plain, out, rows, heads, head_dim, rot);
    for (index = 0; index < count; index++)
        actual[index] = ft_float_from_bf16(out[index]);

    ft_tt_rope_reference_apply(source, expected, rows, heads, head_dim, rope_dim, position,
                               (float)MODEL_THETA);

    snprintf(label, sizeof(label), "%s: the model computes RoPE (PCC vs an fp32 reference)",
             what);
    ft_tt_test_check_fmt(label, pearson(expected, actual, count) >= PCC_ACCEPT,
                         "PCC %.9f, max abs %.6g", pearson(expected, actual, count),
                         max_abs_difference(expected, actual, count));

    /* PARTIAL: everything past `rot` must come through BIT-identical, not merely close. */
    if (rot < head_dim) {
        uint32_t row;

        for (row = 0; row < rows; row++) {
            uint32_t head;

            for (head = 0; head < heads; head++) {
                uint32_t column;

                for (column = rot; column < head_dim; column++) {
                    const size_t at = (size_t)row * width + (size_t)head * head_dim + column;

                    if (out[at] != in[at])
                        untouched_wrong++;
                }
            }
        }
        snprintf(label, sizeof(label), "%s: the tiles past the rotary dimension pass through "
                                       "BIT-identical", what);
        ft_tt_test_check_fmt(label, untouched_wrong == 0u, "%zu of %zu wrong",
                             untouched_wrong,
                             (size_t)rows * heads * (head_dim - rot));
    }

    /* CAN THIS DATA SEE A SWAPPED PAIR? Asserted, not assumed. The wrong pairing is applied
     * by negating the table's sign convention, which is exactly what swapping the two halves
     * of the partner stream does to the answer. If the operands were symmetric or the data
     * were small integers this would come back "no difference" and the whole byte-comparison
     * gate would be worthless. */
    for (index = 0; index < (size_t)rows * rot; index++)
        swapped_sin[index] = ft_bf16_from_float(-ft_float_from_bf16(sin_plain[index]));
    ft_tt_rope_model_apply(in, cos_plain, swapped_sin, swapped_out, rows, heads, head_dim, rot);
    for (index = 0; index < count; index++)
        swapped_actual[index] = ft_float_from_bf16(swapped_out[index]);
    snprintf(label, sizeof(label), "%s: the DATA can see a wrong pairing", what);
    ft_tt_test_check_fmt(label, max_abs_difference(actual, swapped_actual, count) > 0.01,
                         "max abs difference %.6g",
                         max_abs_difference(actual, swapped_actual, count));

done:
    free(source);
    free(expected);
    free(actual);
    free(swapped_actual);
    free(in);
    free(out);
    free(swapped_out);
    free(cos_plain);
    free(sin_plain);
    free(swapped_sin);
}

/**
 * THE ANGLE, READ STRAIGHT OUT OF THE ANSWER.
 *
 * Rotate the basis pair (1, 0): the output IS (cos, sin). No correlation, no norm, no
 * tolerance for a wrong position or a wrong theta to hide in — §22.2's rule applied to the
 * op whose invariance is the rotation itself.
 */
static void test_angle_is_read_directly(uint32_t position)
{
    const uint32_t head_dim = MODEL_HEAD_DIM;
    const uint32_t rot = MODEL_ROPE_DIM;
    const uint32_t half = rot / 2u;
    const uint32_t rows = TILE_DIM;
    const uint32_t row = position % TILE_DIM;
    const uint32_t block_first = position - row;
    uint16_t *in = calloc((size_t)rows * head_dim, sizeof(uint16_t));
    uint16_t *out = calloc((size_t)rows * head_dim, sizeof(uint16_t));
    uint16_t *cos_plain = malloc((size_t)rows * rot * sizeof(uint16_t));
    uint16_t *sin_plain = malloc((size_t)rows * rot * sizeof(uint16_t));
    double worst_cos = 0.0;
    double worst_sin = 0.0;
    uint32_t j;
    char label[160];

    if (!in || !out || !cos_plain || !sin_plain) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }

    /* x[j] = 1, x[j+half] = 0, for every j — one head, one meaningful row. */
    for (j = 0; j < half; j++)
        in[(size_t)row * head_dim + j] = ft_bf16_from_float(1.0f);

    ft_tt_rope_reference_table(cos_plain, sin_plain, block_first, rot, MODEL_THETA);
    ft_tt_rope_model_apply(in, cos_plain, sin_plain, out, rows, 1u, head_dim, rot);

    for (j = 0; j < half; j++) {
        const double frequency = pow(MODEL_THETA, -2.0 * (double)j / (double)rot);
        const double angle = (double)position * frequency;
        const double got_cos = ft_float_from_bf16(out[(size_t)row * head_dim + j]);
        const double got_sin = ft_float_from_bf16(out[(size_t)row * head_dim + j + half]);
        const double cos_error = fabs(got_cos - cos(angle));
        const double sin_error = fabs(got_sin - sin(angle));

        if (cos_error > worst_cos)
            worst_cos = cos_error;
        if (sin_error > worst_sin)
            worst_sin = sin_error;
    }

    /* 2^-8 is bf16's step at 1.0, and the value passes through two bf16 roundings, so ~0.008
     * is the floor a correct implementation can reach. A wrong POSITION or a wrong theta
     * misses by O(1), not by O(2^-8). */
    snprintf(label, sizeof(label), "position %u: rotating (1,0) returns (cos, sin)", position);
    ft_tt_test_check_fmt(label, worst_cos < 0.01 && worst_sin < 0.01,
                         "worst |dcos| %.5f, |dsin| %.5f", worst_cos, worst_sin);

done:
    free(in);
    free(out);
    free(cos_plain);
    free(sin_plain);
}

/**
 * A position ONE row later must produce a different answer.
 *
 * The mutation this exists for is "the table row is always row 0", which every score above
 * would survive at small positions because the angles are nearly equal there — and which is
 * the exact bug the row-phase contract invites.
 */
static void test_position_is_actually_used(void)
{
    const uint32_t rot = MODEL_ROPE_DIM;
    const uint32_t half = rot / 2u;
    uint16_t *cos_plain = malloc((size_t)TILE_DIM * rot * sizeof(uint16_t));
    uint16_t *sin_plain = malloc((size_t)TILE_DIM * rot * sizeof(uint16_t));
    double worst = 0.0;
    uint32_t j;

    ft_tt_test_step("the POSITION reaches the answer — row by row, not just block by block");

    if (!cos_plain || !sin_plain) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }
    ft_tt_rope_reference_table(cos_plain, sin_plain, 0u, rot, MODEL_THETA);
    for (j = 0; j < half; j++) {
        const double row_0 = ft_float_from_bf16(cos_plain[j]);
        const double row_1 = ft_float_from_bf16(cos_plain[rot + j]);
        const double difference = fabs(row_0 - row_1);

        if (difference > worst)
            worst = difference;
    }
    ft_tt_test_check_fmt("consecutive table rows differ", worst > 0.1,
                         "largest row-to-row |dcos| = %.4f", worst);

done:
    free(cos_plain);
    free(sin_plain);
}

/* ================================================================ */

int main(void)
{
    ft_tt_test_step("RoPE — OFF-HARDWARE gate");

    test_wire_constants();
    test_parameter_block();
    test_refusals();
    test_cb_placement();
    test_table_layout_and_angles();

    ft_tt_test_step("the op, through the device model");
    model_one_shape(FT_TT_OP_ROPE_MINICPM5_HEADS_Q, MODEL_HEAD_DIM, MODEL_ROPE_DIM, 0u,
                    "MiniCPM5-1B Q at position 0");
    model_one_shape(FT_TT_OP_ROPE_MINICPM5_HEADS_KV, MODEL_HEAD_DIM, MODEL_ROPE_DIM, 4096u,
                    "MiniCPM5-1B KV at position 4096");
    /* The PARTIAL case MiniCPM5-1B does not exercise: 64 of 256, which is Qwen3.6's shape. */
    model_one_shape(2u, 256u, 64u, 1024u, "partial, 64 of 256");

    ft_tt_test_step("the ANGLE, read out of the answer rather than correlated with it");
    test_angle_is_read_directly(0u);
    test_angle_is_read_directly(7u);
    test_angle_is_read_directly(100u);
    test_angle_is_read_directly(131040u);

    test_position_is_actually_used();

    return ft_tt_test_finish();
}
