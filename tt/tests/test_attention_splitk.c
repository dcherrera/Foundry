/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_attention_splitk.c
 * @brief OFF-HARDWARE gate for SPLIT-K decode attention (resident_attention_protocol.h §7):
 *   the PARTIAL / MERGE wire format, the shipped encodings byte for byte, and the arithmetic —
 *   the kernel's partial + merge replayed at its own rounding points against float64.
 *
 * Stages:
 *   1. the split constants agree with the kernel header (parsed, not retyped);
 *   2. every non-split descriptor encodes to the words origin/main's encoder produced
 *      (k_golden_words, make_attention_golden_words.sh) — the knob-off identity of the words;
 *   3. PARTIAL / MERGE encode as pinned, and every malformed one is refused;
 *   4. the device validity: the shipped programs refuse both flags, the split build accepts
 *      them and refuses the malformed ones, and a MERGE keeps CB_CONST on its 4-page grid;
 *   5. the split model with fp32 storage against float64 (the algorithm: chunking, partials,
 *      merge order) at 1k / 8k / 33k positions, 1 / 2 / 8 verify rows, chunk counts that do
 *      not divide the length, K8V4-rounded K/V and a poisoned stale region;
 *   6. the split model with bf16 storage (the kernel's Float16_b CBs) against float64 and
 *      against today's one-core model: split must not be worse than lanes by more than the bar;
 *   7. negative controls: a dropped partial and malformed chunk lists must fail.
 *
 * `make -f Makefile.attention red-splitk` builds the model with one partial always dropped
 * (-DFT_TT_SPLITK_RED_DROP) and must report failures.
 *
 * Usage:  ./build-attn/test_attention_splitk   (from tt/tests)
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_op_attention.h"
#include "ft_tt_attention_splitk_model.h"
#include "ft_tt_test_report.h"

#define FTQ_L1_PTR
#include "../kernels/resident_attention_protocol.h"

#include "attention_golden_requests.inc"

#define TILE_DIM    32u
#define TILE_BYTES  2048u
#define BLOCK_TILES 16u /* shipped */

/** Qwen3.8-27B attention: 24 Q heads / 4 KV heads (q_rows 6), head_dim 256. */
#define Q38_Q_ROWS   6u
#define Q38_HEAD_DIM 256u
#define Q38_DIM_TILES (Q38_HEAD_DIM / TILE_DIM)

#define MAX_POSITIONS 35840u /* 33,000 live, padded to whole double blocks, plus slack */
#define MAX_ROWS      (Q38_Q_ROWS * 8u)

/** fp32 storage against float64: max |model - reference| / max |reference|. Measured at
 *  most 1.15e-6 over the stage-5 cases (2026-11-16), the same order as today's one-core model
 *  in fp32 (printed beside it): the fp32 score dot products dominate, not the split. The bar
 *  is ~2.6x the worst. A dropped partial measures 0.3-0.9 (stage 7). */
#define FP32_RELATIVE_BAR 3.0e-6
/** bf16 storage: split's worst error may exceed today's one-core worst by at most this
 *  factor. Measured split / lanes 0.46-1.11 (2026-11-16): above 1 only for short chunks
 *  (1k positions over 3 chunks, more merges than blocks); at depth split rounds O fewer times
 *  than the one-core block loop does. */
#define BF16_SPLIT_OVER_LANES_BAR 1.25

/* ===================================================================== words */

/** origin/main df55904's encoder on attention_golden_requests.inc
 *  (bash make_attention_golden_words.sh df55904). */
static const uint32_t k_golden_words[GOLDEN_REQUEST_COUNT][FT_TT_OP_ATTENTION_WORDS] = {
    { 0x00000708u, 0x00000004u, 0x000003E8u, 0x00000008u, 0x00001000u, 0x00000000u,
      0x00020001u, 0x00010000u, 0x00040003u, 0x00020000u, 0x00060005u, 0x00030000u,
      0x00080007u, 0x00040000u, 0x000A0009u, 0x00050000u },
    { 0x00000004u, 0x00000008u, 0x00007000u, 0x00000010u, 0x00002000u, 0x00000000u,
      0x00020001u, 0x00010100u, 0x00040003u, 0x00020200u, 0x00060005u, 0x00030300u,
      0x00080007u, 0x00040000u, 0x000A0009u, 0x00050500u },
    { 0x00000006u, 0x00000008u, 0x00007001u, 0x00000010u, 0x00002000u, 0x00000000u,
      0x00020001u, 0x00010200u, 0x00040003u, 0x00020400u, 0x00060005u, 0x00030600u,
      0x00080007u, 0x00040000u, 0x000A0009u, 0x00050A00u },
    { 0x02020006u, 0x00000008u, 0x00000AFEu, 0x00000010u, 0x00002000u, 0x0003A000u,
      0x00020001u, 0x00010300u, 0x00040003u, 0x00020600u, 0x00060005u, 0x00030900u,
      0x00080007u, 0x00040000u, 0x000A0009u, 0x00050F00u },
    { 0x03020506u, 0x00000008u, 0x00000B05u, 0x00000010u, 0x00002000u, 0x0003A800u,
      0x00020001u, 0x00010400u, 0x00040003u, 0x00020800u, 0x00060005u, 0x00030C00u,
      0x00080007u, 0x00040000u, 0x000A0009u, 0x00051400u },
    { 0x00010020u, 0x00000008u, 0x00000420u, 0x00000010u, 0x00002000u, 0x00000000u,
      0x00020001u, 0x00010500u, 0x00040003u, 0x00020A00u, 0x00060005u, 0x00030F00u,
      0x00080007u, 0x00040000u, 0x000A0009u, 0x00051900u },
    { 0x00010007u, 0x00000008u, 0x00000807u, 0x00000010u, 0x00002000u, 0x00000000u,
      0x00020001u, 0x00010600u, 0x00040003u, 0x00020C00u, 0x00060005u, 0x00031200u,
      0x00080007u, 0x00040000u, 0x000A0009u, 0x00051E00u },
    { 0x00000006u, 0x00000008u, 0x00040000u, 0x00000010u, 0x00002040u, 0x00000000u,
      0x00020001u, 0x00010700u, 0x00040003u, 0x00020E00u, 0x00060005u, 0x00031500u,
      0x00080007u, 0x00040000u, 0x000A0009u, 0x00052300u },
};

/** `enum { name = value };` out of the kernel's protocol header (test_attention_ops.c's
 *  reader, for the split names). */
static uint32_t kernel_constant(const char *name)
{
    static char buffer[262144];
    static size_t length = 0;
    const char *directory = getenv("FT_TT_KERNEL_DIR");
    char needle[128];
    const char *found;

    if (length == 0u) {
        char path[512];
        FILE *file;

        snprintf(path, sizeof(path), "%s/resident_attention_protocol.h",
                 directory ? directory : "../kernels");
        file = fopen(path, "rb");
        if (!file) {
            fprintf(stderr, "test_attention_splitk: cannot open %s — run from tt/tests or set "
                            "FT_TT_KERNEL_DIR\n", path);
            return UINT32_MAX;
        }
        length = fread(buffer, 1u, sizeof(buffer) - 1u, file);
        buffer[length] = '\0';
        fclose(file);
    }
    snprintf(needle, sizeof(needle), "enum { %s = ", name);
    found = strstr(buffer, needle);
    return found ? (uint32_t)strtoul(found + strlen(needle), NULL, 0) : UINT32_MAX;
}

static void stage_wire_contract(void)
{
    static const struct {
        const char *name;
        uint32_t host;
    } pairs[] = {
        { "kAttnFlagPartial", FT_TT_OP_ATTENTION_FLAG_PARTIAL },
        { "kAttnFlagMerge", FT_TT_OP_ATTENTION_FLAG_MERGE },
        { "kAttnFlagRows", FT_TT_OP_ATTENTION_FLAG_ROWS },
        { "kAttnPartialStatTiles", FT_TT_OP_ATTENTION_PARTIAL_STAT_TILES },
        { "kAttnMaxPartials", FT_TT_OP_ATTENTION_MAX_PARTIALS },
    };

    ft_tt_test_step("1. the split constants agree with the kernel header");
    for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); i++) {
        const uint32_t device = kernel_constant(pairs[i].name);

        ft_tt_test_check_fmt(pairs[i].name, device == pairs[i].host, "kernel %u host %u", device,
                             pairs[i].host);
    }
}

static void stage_golden_words(void)
{
    ft_tt_test_step("2. every non-split descriptor encodes to origin/main's words");
    for (uint32_t i = 0; i < GOLDEN_REQUEST_COUNT; i++) {
        ft_tt_op_attention_request request;
        uint32_t words[FT_TT_OP_ATTENTION_WORDS];
        uint32_t differ = 0u;
        char label[96];
        int rc;

        golden_request(i, &request);
        rc = ft_tt_op_attention_parameters(&request, words);
        for (uint32_t w = 0; rc == FT_TT_OK && w < FT_TT_OP_ATTENTION_WORDS; w++)
            differ += words[w] != k_golden_words[i][w];
        snprintf(label, sizeof(label), "golden request %u (flags %u) byte-identical", i,
                 request.flags);
        ft_tt_test_check_fmt(label, rc == FT_TT_OK && differ == 0u, "rc %d, %u word(s) differ",
                             rc, differ);
    }
}

/* ================================================================== encoding */

static void fill_operand(ft_tt_op_operand *operand, uint16_t x, uint16_t y, uint32_t address)
{
    operand->x = x;
    operand->y = y;
    operand->address = address;
    operand->tile_bytes = TILE_BYTES;
}

/** A Qwen3.8 PARTIAL over chunk tiles [64, 128) of a 28,673-position decode: K / V bases
 *  carry the chunk's first tile, the live length and capacity are chunk-relative. */
static void fill_partial(ft_tt_op_attention_request *request)
{
    memset(request, 0, sizeof(*request));
    request->q_rows = Q38_Q_ROWS;
    request->dim_tiles = Q38_DIM_TILES;
    request->block_tiles = BLOCK_TILES;
    request->kv_positions = 64u * TILE_DIM;
    request->capacity_tiles = 8192u - 64u;
    request->flags = FT_TT_OP_ATTENTION_FLAG_PARTIAL;
    fill_operand(&request->query, 1u, 2u, 0x10000u);
    fill_operand(&request->keys, 3u, 4u, 0x20000u + 64u * Q38_DIM_TILES * 1088u);
    request->keys.tile_bytes = 1088u;
    fill_operand(&request->values, 5u, 6u, 0x30000u + 64u * Q38_DIM_TILES * 576u);
    request->values.tile_bytes = 576u;
    fill_operand(&request->constants, 7u, 8u, 0x40000u);
    fill_operand(&request->output, 9u, 10u, 0x50000u);
}

static void fill_merge(ft_tt_op_attention_request *request, uint32_t partials)
{
    memset(request, 0, sizeof(*request));
    request->q_rows = Q38_Q_ROWS;
    request->dim_tiles = Q38_DIM_TILES;
    request->block_tiles = BLOCK_TILES;
    request->capacity_tiles = 8192u;
    request->query_row = 0u;
    request->flags = FT_TT_OP_ATTENTION_FLAG_MERGE | FT_TT_OP_ATTENTION_FLAG_ROWS;
    request->positions = 2u;
    request->partial_count = partials;
    fill_operand(&request->query, 11u, 12u, 0x60000u);
    fill_operand(&request->output, 9u, 10u, 0x50000u);
}

static int encode(const ft_tt_op_attention_request *request, uint32_t *words)
{
    return ft_tt_op_attention_parameters(request, words);
}

static void stage_split_encoding(void)
{
    ft_tt_op_attention_request request;
    uint32_t words[FT_TT_OP_ATTENTION_WORDS];
    int rc;

    ft_tt_test_step("3. PARTIAL and MERGE encode as pinned; malformed ones are refused");
    fill_partial(&request);
    rc = encode(&request, words);
    ft_tt_test_check_fmt("a decode PARTIAL encodes", rc == FT_TT_OK, "rc %d", rc);
    ft_tt_test_check_fmt("word 0: q_rows 6, flags PARTIAL", words[0] == (6u | (4u << 16)),
                         "0x%08X", words[0]);
    ft_tt_test_check_fmt("word 2: the chunk-relative live length", words[2] == 2048u, "%u",
                         words[2]);
    ft_tt_test_check_fmt("word 4: the capacity past the chunk start", words[4] == 8128u, "%u",
                         words[4]);
    ft_tt_test_check_fmt("word 9 / 11: K and V bases carry the chunk start",
                         words[9] == 0x20000u + 64u * 8u * 1088u &&
                             words[11] == 0x30000u + 64u * 8u * 576u,
                         "0x%X 0x%X", words[9], words[11]);
    request.flags |= FT_TT_OP_ATTENTION_FLAG_ROWS;
    request.positions = 2u;
    request.rows_mask_address = 0x3b000u;
    rc = encode(&request, words);
    ft_tt_test_check_fmt("a ROWS PARTIAL encodes: flags 6, positions 2, word 5 the mask",
                         rc == FT_TT_OK && words[0] == (6u | (6u << 16) | (2u << 24)) &&
                             words[5] == 0x3b000u,
                         "rc %d 0x%08X 0x%X", rc, words[0], words[5]);
    request.flags = FT_TT_OP_ATTENTION_FLAG_PARTIAL | FT_TT_OP_ATTENTION_FLAG_CHUNK;
    request.q_rows = 32u;
    ft_tt_test_check("PARTIAL with CHUNK refused", encode(&request, words) == FT_TT_EINVAL, "");

    fill_merge(&request, 8u);
    rc = encode(&request, words);
    ft_tt_test_check_fmt("a ROWS MERGE of 8 encodes", rc == FT_TT_OK, "rc %d", rc);
    ft_tt_test_check_fmt("word 0: q_rows 6, flags MERGE|ROWS, positions 2",
                         words[0] == (6u | (10u << 16) | (2u << 24)), "0x%08X", words[0]);
    ft_tt_test_check_fmt("word 2: the partial count", words[2] == 8u, "%u", words[2]);
    ft_tt_test_check_fmt("Q: the partials; OUT: the output",
                         words[6] == ((12u << 16) | 11u) && words[7] == 0x60000u &&
                             words[14] == ((10u << 16) | 9u) && words[15] == 0x50000u,
                         "0x%X 0x%X 0x%X 0x%X", words[6], words[7], words[14], words[15]);
    ft_tt_test_check_fmt("K, V, constants and word 5 zero",
                         (words[5] | words[8] | words[9] | words[10] | words[11] | words[12] |
                          words[13]) == 0u,
                         "%u", words[5] | words[8] | words[9]);
    fill_merge(&request, 0u);
    ft_tt_test_check("MERGE of 0 partials refused", encode(&request, words) == FT_TT_EINVAL, "");
    fill_merge(&request, FT_TT_OP_ATTENTION_MAX_PARTIALS + 1u);
    ft_tt_test_check("MERGE past the maximum refused", encode(&request, words) == FT_TT_EINVAL,
                     "");
    fill_merge(&request, 4u);
    request.flags |= FT_TT_OP_ATTENTION_FLAG_PARTIAL;
    ft_tt_test_check("MERGE with PARTIAL refused", encode(&request, words) == FT_TT_EINVAL, "");
    fill_merge(&request, 4u);
    request.positions = 6u; /* 36 rows: past a tile */
    ft_tt_test_check("MERGE packing past a tile refused", encode(&request, words) == FT_TT_EINVAL,
                     "");
    ft_tt_test_check_fmt("a Qwen3.8 partial is 10 tiles",
                         ft_tt_op_attention_partial_bytes(Q38_DIM_TILES, TILE_BYTES) == 20480u,
                         "%u", ft_tt_op_attention_partial_bytes(Q38_DIM_TILES, TILE_BYTES));
}

/* ================================================================== protocol */

static fta_shape decoded(const ft_tt_op_attention_request *request)
{
    uint32_t words[FT_TT_OP_ATTENTION_WORDS];

    memset(words, 0, sizeof(words));
    (void)encode(request, words);
    return fta_decode(words);
}

static void stage_device_validity(void)
{
    ft_tt_op_attention_request request;
    fta_shape shape;
    uint32_t bad = 0u;

    ft_tt_test_step("4. the device validity: shipped refuses, split accepts, malformed refused");
    fill_partial(&request);
    shape = decoded(&request);
    ft_tt_test_check("the shipped programs refuse a PARTIAL", !fta_shape_is_valid(&shape), "");
    ft_tt_test_check("the split build accepts it", fta_split_shape_is_valid(&shape), "");
    ft_tt_test_check("a PARTIAL decodes as one", fta_is_partial(&shape) && !fta_is_merge(&shape),
                     "");
    fill_merge(&request, 8u);
    shape = decoded(&request);
    ft_tt_test_check("the shipped programs refuse a MERGE", !fta_shape_is_valid(&shape), "");
    ft_tt_test_check("the split build accepts it", fta_split_shape_is_valid(&shape), "");
    ft_tt_test_check_fmt("a MERGE carries its count in word 2 and its packing in word 0",
                         fta_is_merge(&shape) && shape.kv_positions == 8u && shape.positions == 2u,
                         "%u %u", shape.kv_positions, shape.positions);
    shape.flags |= kAttnFlagChunk;
    ft_tt_test_check("split refuses MERGE|CHUNK", !fta_split_shape_is_valid(&shape), "");
    fill_partial(&request);
    shape = decoded(&request);
    shape.flags |= kAttnFlagChunk;
    ft_tt_test_check("split refuses PARTIAL|CHUNK", !fta_split_shape_is_valid(&shape), "");
    shape.flags = 16u;
    ft_tt_test_check("split refuses an unknown flag", !fta_split_shape_is_valid(&shape), "");
    fill_merge(&request, 8u);
    shape = decoded(&request);
    shape.kv_positions = kAttnMaxPartials + 1u;
    ft_tt_test_check("split refuses a MERGE past kAttnMaxPartials",
                     !fta_split_shape_is_valid(&shape), "");
    ft_tt_test_check_fmt("a partial is dim_tiles + 2 tiles", fta_partial_tiles(8u) == 10u, "%u",
                         fta_partial_tiles(8u));
    /* resident_attention_reader.c read_partials: 2-page pairs, plus a spare pair when P is
     * odd, must be a whole number of CB_CONST's 4-page laps for every count. */
    for (uint32_t p = 1u; p <= kAttnMaxPartials; p++)
        bad += ((((p + 1u) / 2u) * 2u * kAttnPartialStatTiles) % kAttnCbConstPages) != 0u;
    ft_tt_test_check_fmt("a MERGE of every count leaves CB_CONST on its 4-page grid", bad == 0u,
                         "%u counts off grid", bad);
}

/* ================================================================ arithmetic */

static float g_query[MAX_ROWS * Q38_HEAD_DIM];
static float g_keys[MAX_POSITIONS * Q38_HEAD_DIM];
static float g_values[MAX_POSITIONS * Q38_HEAD_DIM];
static float g_reference[MAX_ROWS * Q38_HEAD_DIM];
static float g_model[MAX_ROWS * Q38_HEAD_DIM];
static float g_lanes[MAX_ROWS * Q38_HEAD_DIM];
static uint32_t g_live[MAX_ROWS];

static uint64_t g_state;

static float random_unit(void)
{
    g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((double)(g_state >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

static float round_bf16(float value)
{
    return ft_float_from_bf16(ft_bf16_from_float(value));
}

/**
 * Qwen3.8 shape, `verify` rows over `live` positions (row p*6 + h sees live - verify + p + 1).
 * K and V are rounded to K8V4 when `k8v4`; the stale region past the live length is poisoned
 * with +40 so a mask that leaks shows. A few large keys make the scores spread several units.
 */
static ft_tt_splitk_problem build_problem(uint32_t live, uint32_t verify, bool k8v4,
                                          uint64_t seed)
{
    ft_tt_splitk_problem problem;

    g_state = seed | 1u;
    for (uint32_t i = 0; i < Q38_Q_ROWS * verify * Q38_HEAD_DIM; i++)
        g_query[i] = round_bf16(random_unit() * 2.0f);
    for (uint32_t p = 0; p < MAX_POSITIONS; p++) {
        float *key = g_keys + (size_t)p * Q38_HEAD_DIM;
        float *value = g_values + (size_t)p * Q38_HEAD_DIM;
        const float boost = (p % 997u == 0u) ? 4.0f : 1.0f;

        for (uint32_t e = 0; e < Q38_HEAD_DIM; e++) {
            key[e] = p < live ? round_bf16(random_unit() * boost) : 40.0f;
            value[e] = p < live ? round_bf16(random_unit()) : 40.0f;
        }
        if (k8v4 && p < live) {
            ft_tt_splitk_round_block_float(key, Q38_HEAD_DIM, false);
            ft_tt_splitk_round_block_float(value, Q38_HEAD_DIM, true);
        }
    }
    for (uint32_t p = 0; p < verify; p++)
        for (uint32_t h = 0; h < Q38_Q_ROWS; h++)
            g_live[p * Q38_Q_ROWS + h] = live - verify + p + 1u;
    problem.rows = Q38_Q_ROWS * verify;
    problem.head_dim = Q38_HEAD_DIM;
    problem.live = g_live;
    problem.query = g_query;
    problem.keys = g_keys;
    problem.values = g_values;
    problem.capacity_positions = MAX_POSITIONS;
    return problem;
}

/** `count` chunks over `live_tiles`: whole multiples of `unit` tiles, the remainder in the
 *  last — the shape a plan aligned to double blocks produces. */
static uint32_t make_chunks(uint32_t live_tiles, uint32_t count, uint32_t unit,
                            ft_tt_splitk_chunk *chunks)
{
    const uint32_t units = (live_tiles + unit - 1u) / unit;
    const uint32_t per = (units + count - 1u) / count;
    uint32_t made = 0u;

    for (uint32_t first = 0; first < live_tiles; first += per * unit) {
        chunks[made].first_tile = first;
        chunks[made].tile_count =
            first + per * unit < live_tiles ? per * unit : live_tiles - first;
        made++;
    }
    return made;
}

static double relative_error(const float *expected, const float *actual, size_t count)
{
    double worst = 0.0;
    double scale = 0.0;

    for (size_t i = 0; i < count; i++) {
        const double difference = fabs((double)expected[i] - (double)actual[i]);

        if (!isfinite(actual[i]))
            return INFINITY;
        if (difference > worst)
            worst = difference;
        if (fabs((double)expected[i]) > scale)
            scale = fabs((double)expected[i]);
    }
    return scale > 0.0 ? worst / scale : worst;
}

typedef struct splitk_case {
    uint32_t live;
    uint32_t verify;
    uint32_t chunks;
    bool k8v4;
} splitk_case;

static const splitk_case k_cases[] = {
    { 1024u, 1u, 2u, false }, { 1024u, 2u, 3u, true },   { 8192u, 1u, 4u, true },
    { 8191u, 2u, 3u, true },  { 8200u, 8u, 5u, true },   { 33000u, 1u, 8u, true },
    { 33000u, 2u, 7u, true }, { 33001u, 8u, 6u, false },
};

static void stage_algorithm(double *out_split_over_lanes)
{
    ft_tt_splitk_chunk chunks[FT_TT_OP_ATTENTION_MAX_PARTIALS];
    double worst_ratio = 0.0;

    ft_tt_test_step("5-6. split partial + merge vs float64 (fp32 storage) and vs lanes (bf16)");
    for (size_t i = 0; i < sizeof(k_cases) / sizeof(k_cases[0]); i++) {
        const splitk_case *c = &k_cases[i];
        const ft_tt_splitk_problem problem = build_problem(c->live, c->verify, c->k8v4, 1115u + i);
        const uint32_t live_tiles = (c->live + TILE_DIM - 1u) / TILE_DIM;
        const uint32_t unit = c->live >= 2048u ? 2u * BLOCK_TILES : 4u;
        const uint32_t count = make_chunks(live_tiles, c->chunks, unit, chunks);
        const size_t elements = (size_t)problem.rows * Q38_HEAD_DIM;
        double fp32, split, lanes;
        char label[160];
        bool ok;

        ok = ft_tt_splitk_reference(&problem, g_reference) &&
             ft_tt_splitk_model(&problem, BLOCK_TILES, chunks, count, FT_TT_SPLITK_STORE_FP32,
                                UINT32_MAX, g_model) &&
             ft_tt_splitk_lanes_model(&problem, BLOCK_TILES, FT_TT_SPLITK_STORE_FP32, g_lanes);
        fp32 = ok ? relative_error(g_reference, g_model, elements) : INFINITY;
        lanes = ok ? relative_error(g_reference, g_lanes, elements) : INFINITY;
        snprintf(label, sizeof(label), "fp32: %u positions, %u row(s), %u chunks%s", c->live,
                 c->verify, count, c->k8v4 ? ", K8V4" : "");
        ft_tt_test_check_fmt(label, ok && fp32 <= FP32_RELATIVE_BAR,
                             "rel %.3e (bar %.1e; one-core fp32 %.3e)", fp32, FP32_RELATIVE_BAR,
                             lanes);

        ok = ft_tt_splitk_model(&problem, BLOCK_TILES, chunks, count, FT_TT_SPLITK_STORE_BF16,
                                UINT32_MAX, g_model) &&
             ft_tt_splitk_lanes_model(&problem, BLOCK_TILES, FT_TT_SPLITK_STORE_BF16, g_lanes);
        split = ok ? relative_error(g_reference, g_model, elements) : INFINITY;
        lanes = ok ? relative_error(g_reference, g_lanes, elements) : INFINITY;
        if (lanes > 0.0 && split / lanes > worst_ratio)
            worst_ratio = split / lanes;
        snprintf(label, sizeof(label), "bf16: %u positions, %u row(s), %u chunks%s", c->live,
                 c->verify, count, c->k8v4 ? ", K8V4" : "");
        ft_tt_test_check_fmt(label, ok && split <= BF16_SPLIT_OVER_LANES_BAR * lanes,
                             "split rel %.3e, lanes rel %.3e, ratio %.2f (bar %.2f)", split,
                             lanes, lanes > 0.0 ? split / lanes : 0.0,
                             BF16_SPLIT_OVER_LANES_BAR);
    }
    *out_split_over_lanes = worst_ratio;
}

static void stage_negative_controls(void)
{
    ft_tt_splitk_chunk chunks[FT_TT_OP_ATTENTION_MAX_PARTIALS];
    const ft_tt_splitk_problem problem = build_problem(8192u, 2u, true, 77u);
    const uint32_t count = make_chunks(256u, 4u, 2u * BLOCK_TILES, chunks);
    const size_t elements = (size_t)problem.rows * Q38_HEAD_DIM;
    ft_tt_splitk_chunk broken[FT_TT_OP_ATTENTION_MAX_PARTIALS];
    double error;

    ft_tt_test_step("7. negative controls: the gate can fail");
    (void)ft_tt_splitk_reference(&problem, g_reference);
    for (uint32_t drop = 0; drop < count; drop++) {
        char label[96];

        error = ft_tt_splitk_model(&problem, BLOCK_TILES, chunks, count, FT_TT_SPLITK_STORE_FP32,
                                   drop, g_model)
                    ? relative_error(g_reference, g_model, elements)
                    : INFINITY;
        snprintf(label, sizeof(label), "dropping chunk %u's partial fails the fp32 bar", drop);
        ft_tt_test_check_fmt(label, error > 1.0e3 * FP32_RELATIVE_BAR, "rel %.3e", error);
    }
    memcpy(broken, chunks, sizeof(broken));
    broken[1].first_tile -= 1u; /* overlaps chunk 0 */
    ft_tt_test_check("overlapping chunks refused",
                     !ft_tt_splitk_model(&problem, BLOCK_TILES, broken, count,
                                         FT_TT_SPLITK_STORE_FP32, UINT32_MAX, g_model), "");
    memcpy(broken, chunks, sizeof(broken));
    broken[2].first_tile += 1u; /* a gap */
    broken[2].tile_count -= 1u;
    ft_tt_test_check("a gap between chunks refused",
                     !ft_tt_splitk_model(&problem, BLOCK_TILES, broken, count,
                                         FT_TT_SPLITK_STORE_FP32, UINT32_MAX, g_model), "");
    ft_tt_test_check("chunks short of the live length refused",
                     !ft_tt_splitk_model(&problem, BLOCK_TILES, chunks, count - 1u,
                                         FT_TT_SPLITK_STORE_FP32, UINT32_MAX, g_model), "");
    broken[0].first_tile = 0u;
    broken[0].tile_count = 255u;
    broken[1].first_tile = 255u;
    broken[1].tile_count = 1u;
    ft_tt_test_check("a last chunk without every row's limit is valid only when it holds them",
                     ft_tt_splitk_chunks_are_valid(broken, 2u, 256u, 255u) &&
                         !ft_tt_splitk_chunks_are_valid(broken, 2u, 256u, 254u), "");
}

int main(void)
{
    double ratio = 0.0;

    ft_tt_test_step("split-K decode attention, OFF-HARDWARE");
    stage_wire_contract();
    stage_golden_words();
    stage_split_encoding();
    stage_device_validity();
    stage_algorithm(&ratio);
    stage_negative_controls();
    printf("worst bf16 split / lanes error ratio: %.3f\n", ratio);
    return ft_tt_test_finish();
}
