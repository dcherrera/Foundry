/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_vector.c
 * @brief Implementation of the vector op family's host driver — the elementwise binary and
 *   the reduce binary. See ft_tt_op_vector.h, which carries the decision about which ops
 *   share a binary and why residency, not code size, is what decides it.
 *
 * Everything here is PURE — no device access, no allocation. The enqueue itself is
 * ft_tt_queue_write_op_parameters() plus ft_tt_queue_enqueue(), which the caller does,
 * because a resident op has no launch of its own to own.
 *
 * @status Green off-hardware in tests/test_vector_ops.c (198 checks) and on the card in
 *   tests/test_phase6_vector.c (Tensix 20, 0 checks failed). The ragged-W padding calls
 *   added 2026-08-15 are OFF-HARDWARE ONLY so far: three red proofs (pad value, divisor,
 *   one wrong pad lane) and no card time.
 * @issues Layer norm has NO ragged-W path and cannot get one from a caller — see
 *   ft_tt_op_reduce_pad_value(). It needs the reduce kernel's mask tile.
 * @todo None
 */

#include "ft_tt_op_vector.h"

#include <string.h>

#include "convert/ft_float_convert.h"
#include "ft_tt_device.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_tilize.h"

/** Pages per elementwise input CB, as a multiple of the prefetch depth. TWO blocks: the
 *  reader fills one while the compute thread consumes the other, which is the whole point
 *  of a depth, and it is also the invariant that keeps a reserved block contiguous. */
#define ELTWISE_CB_BLOCKS 2u

/** Pages in the CBs that are produced and consumed a tile at a time. */
#define VECTOR_CB_PAGES_SMALL 2u

/** Elements in one 32x32 tile. */
#define VECTOR_TILE_ELEMENTS (FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM)

bool ft_tt_op_vector_function_is_binary(ft_tt_op_vector_function function)
{
    return function >= FT_TT_OP_VECTOR_FN_ADD;
}

const char *ft_tt_op_vector_function_name(ft_tt_op_vector_function function)
{
    switch (function) {
    case FT_TT_OP_VECTOR_FN_SILU:
        return "SILU";
    case FT_TT_OP_VECTOR_FN_SIGMOID:
        return "SIGMOID";
    case FT_TT_OP_VECTOR_FN_ADD:
        return "ADD";
    case FT_TT_OP_VECTOR_FN_MUL:
        return "MUL";
    case FT_TT_OP_VECTOR_FN_SUB:
        return "SUB";
    case FT_TT_OP_VECTOR_FN_SILU_MUL:
        return "SILU_MUL";
    default:
        return "unknown";
    }
}

/** Whether `function` is one this build knows. An unknown function would otherwise reach
 *  the kernel's switch, match no case, and produce an untouched DST — a zero tile, which
 *  looks like data. */
static bool function_is_known(ft_tt_op_vector_function function)
{
    return function >= FT_TT_OP_VECTOR_FN_SILU && function <= FT_TT_OP_VECTOR_FN_SILU_MUL;
}

/** An operand is usable when it names a tile size. The COORDINATE is not checked here:
 *  it comes from ft_tt_coord and a caller that invented one has a bigger problem than a
 *  range check (spec §7.3). */
static bool operand_is_set(const ft_tt_op_operand *operand)
{
    return operand->tile_bytes != 0u;
}

int ft_tt_op_eltwise_parameters(const ft_tt_op_eltwise_request *request,
                                uint32_t out_words[FT_TT_OP_VECTOR_WORDS])
{
    if (!request || !out_words)
        return FT_TT_EINVAL;
    if (request->tile_count == 0u || !function_is_known(request->function))
        return FT_TT_EINVAL;
    if (!operand_is_set(&request->input_a) || !operand_is_set(&request->output))
        return FT_TT_EINVAL;
    if (ft_tt_op_vector_function_is_binary(request->function) &&
        !operand_is_set(&request->input_b))
        return FT_TT_EINVAL;

    memset(out_words, 0, FT_TT_OP_VECTOR_WORDS * sizeof(out_words[0]));
    out_words[FT_TT_OP_VECTOR_FUNCTION] = (uint32_t)request->function;
    out_words[FT_TT_OP_VECTOR_TILE_COUNT] = request->tile_count;
    out_words[FT_TT_OP_VECTOR_TILE_COLS] = request->row0 ? FT_TT_OP_ELTWISE_FLAG_ROW0 : 0u;
    out_words[FT_TT_OP_VECTOR_DEPTH] =
        ft_tt_op_block_depth(request->tile_count, request->prefetch_tiles);
    out_words[FT_TT_OP_VECTOR_A_X] = request->input_a.x;
    out_words[FT_TT_OP_VECTOR_A_Y] = request->input_a.y;
    out_words[FT_TT_OP_VECTOR_A_BASE] = request->input_a.address;
    out_words[FT_TT_OP_VECTOR_B_X] = request->input_b.x;
    out_words[FT_TT_OP_VECTOR_B_Y] = request->input_b.y;
    out_words[FT_TT_OP_VECTOR_B_BASE] = request->input_b.address;
    out_words[FT_TT_OP_VECTOR_C_X] = request->output.x;
    out_words[FT_TT_OP_VECTOR_C_Y] = request->output.y;
    out_words[FT_TT_OP_VECTOR_C_BASE] = request->output.address;
    return FT_TT_OK;
}

int ft_tt_op_eltwise_place_cbs(const ft_tt_op_eltwise_request *request,
                               ft_tt_launch_cb_entry out_entries[FT_TT_OP_ELTWISE_CB_COUNT],
                               void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    ft_tt_op_cb_request requests[FT_TT_OP_ELTWISE_CB_COUNT];
    uint32_t depth;
    uint32_t input_b_bytes;

    if (!request || !out_entries || !buffer || !out_mask)
        return FT_TT_EINVAL;
    if (request->tile_count == 0u || !function_is_known(request->function))
        return FT_TT_EINVAL;
    if (!operand_is_set(&request->input_a) || !operand_is_set(&request->output))
        return FT_TT_EINVAL;

    depth = ft_tt_op_block_depth(request->tile_count, request->prefetch_tiles);
    /* A unary function still gets a cb_in1, so the kernel alone decides arity. Sizing it
     * from input_a when there is no input_b keeps the page size legal without inventing a
     * format the caller did not name. */
    input_b_bytes = operand_is_set(&request->input_b) ? request->input_b.tile_bytes
                                                      : request->input_a.tile_bytes;

    requests[0].index = FT_TT_OP_ELTWISE_CB_IN0;
    requests[0].page_bytes = request->input_a.tile_bytes;
    requests[0].page_count = depth * ELTWISE_CB_BLOCKS;
    requests[1].index = FT_TT_OP_ELTWISE_CB_IN1;
    requests[1].page_bytes = input_b_bytes;
    requests[1].page_count = depth * ELTWISE_CB_BLOCKS;
    requests[2].index = FT_TT_OP_ELTWISE_CB_SCRATCH;
    requests[2].page_bytes = request->output.tile_bytes;
    requests[2].page_count = VECTOR_CB_PAGES_SMALL;
    requests[3].index = FT_TT_OP_ELTWISE_CB_OUT;
    requests[3].page_bytes = request->output.tile_bytes;
    requests[3].page_count = VECTOR_CB_PAGES_SMALL;

    return ft_tt_op_cb_place(requests, FT_TT_OP_ELTWISE_CB_COUNT, out_entries, buffer,
                             buffer_bytes, out_mask);
}

/* ================================================================
 * The REDUCE family
 * ================================================================ */

/** One member's whole definition, in one row. Adding a fifth member of this family is a
 *  row here plus a reference in the test suite — not a kernel, which is the point of the
 *  merge (kernels/resident_reduce_protocol.h). */
typedef struct reduce_definition {
    ft_tt_op_reduce_function function;
    const char *name;
    uint32_t pass_count;
    uint32_t pool[2];
    uint32_t map[2];
    uint32_t verb[2];
    uint32_t add_eps[2];
    uint32_t source;
    uint32_t chain;
    bool scaler_is_mean;
} reduce_definition;

static const reduce_definition g_reduce_definitions[] = {
    /* softmax: max, then sum of exp(x - max) with a reciprocal; scale the exponentials. */
    { FT_TT_OP_REDUCE_FN_SOFTMAX, "SOFTMAX", 2u,
      { FT_TT_OP_REDUCE_POOL_MAX, FT_TT_OP_REDUCE_POOL_SUM },
      { FT_TT_OP_REDUCE_MAP_NONE, FT_TT_OP_REDUCE_MAP_SUB_EXP },
      { FT_TT_OP_REDUCE_VERB_NONE, FT_TT_OP_REDUCE_VERB_RECIP },
      { 0u, 0u },
      FT_TT_OP_REDUCE_SOURCE_MID, FT_TT_OP_REDUCE_CHAIN_SCALE, false },
    /* RMS norm: one sum of squares with eps folded in and an rsqrt; scale x by gamma. */
    { FT_TT_OP_REDUCE_FN_RMS_NORM, "RMS_NORM", 1u,
      { FT_TT_OP_REDUCE_POOL_SUM, 0u },
      { FT_TT_OP_REDUCE_MAP_SQUARE_SELF, 0u },
      { FT_TT_OP_REDUCE_VERB_RSQRT, 0u },
      { 1u, 0u },
      FT_TT_OP_REDUCE_SOURCE_IN,
      FT_TT_OP_REDUCE_CHAIN_GAMMA | FT_TT_OP_REDUCE_CHAIN_SCALE, true },
    /* layer norm: mean, then the mean of the centred squares with eps and an rsqrt. */
    { FT_TT_OP_REDUCE_FN_LAYER_NORM, "LAYER_NORM", 2u,
      { FT_TT_OP_REDUCE_POOL_SUM, FT_TT_OP_REDUCE_POOL_SUM },
      { FT_TT_OP_REDUCE_MAP_NONE, FT_TT_OP_REDUCE_MAP_SUB_SQUARE },
      { FT_TT_OP_REDUCE_VERB_NONE, FT_TT_OP_REDUCE_VERB_RSQRT },
      { 0u, 1u },
      FT_TT_OP_REDUCE_SOURCE_IN,
      FT_TT_OP_REDUCE_CHAIN_CENTER | FT_TT_OP_REDUCE_CHAIN_GAMMA |
          FT_TT_OP_REDUCE_CHAIN_SCALE | FT_TT_OP_REDUCE_CHAIN_BETA, true },
    /* router normalise: one sum with a reciprocal. Softmax's second pass, alone. */
    { FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE, "ROUTER_NORMALISE", 1u,
      { FT_TT_OP_REDUCE_POOL_SUM, 0u },
      { FT_TT_OP_REDUCE_MAP_NONE, 0u },
      { FT_TT_OP_REDUCE_VERB_RECIP, 0u },
      { 0u, 0u },
      FT_TT_OP_REDUCE_SOURCE_IN, FT_TT_OP_REDUCE_CHAIN_SCALE, false },
    /* L2 norm: RMS norm's program with the scaler left at 1.0 and gamma dropped, so the
     * reduce is a SUM of squares rather than a MEAN and the chain scales x by rsqrt of it.
     * Two table cells apart from RMS_NORM and no kernel — which is the claim the merge in
     * kernels/resident_reduce_protocol.h makes, tested here rather than restated. */
    { FT_TT_OP_REDUCE_FN_L2_NORM, "L2_NORM", 1u,
      { FT_TT_OP_REDUCE_POOL_SUM, 0u },
      { FT_TT_OP_REDUCE_MAP_SQUARE_SELF, 0u },
      { FT_TT_OP_REDUCE_VERB_RSQRT, 0u },
      { 1u, 0u },
      FT_TT_OP_REDUCE_SOURCE_IN, FT_TT_OP_REDUCE_CHAIN_SCALE, false }
};

#define REDUCE_DEFINITION_COUNT \
    (sizeof(g_reduce_definitions) / sizeof(g_reduce_definitions[0]))

/** Fixed pages a reduce residency holds regardless of Wt: scaler, both columns, eps, the
 *  two chain scratch pages, and a double-buffered output. */
#define REDUCE_FIXED_PAGES 8u

/** P3: the deepest CB_IN a residency may ask for, in rows. */
#define REDUCE_INPUT_ROWS_MAX 64u

/** P3 residency extents, with 0 meaning the pre-P3 one. */
static uint32_t reduce_stage_pages(const ft_tt_op_reduce_residency *residency)
{
    return residency->stage_pages ? residency->stage_pages : 1u;
}

static uint32_t reduce_input_rows(const ft_tt_op_reduce_residency *residency)
{
    return residency->input_rows ? residency->input_rows : 1u;
}

/** cb_out: double-buffered, or one chain run deep (P3) when that is more. */
static uint32_t reduce_out_pages(const ft_tt_op_reduce_residency *residency)
{
    const uint32_t stage = reduce_stage_pages(residency);

    return stage > VECTOR_CB_PAGES_SMALL ? stage : VECTOR_CB_PAGES_SMALL;
}

static const reduce_definition *reduce_definition_for(ft_tt_op_reduce_function function)
{
    size_t index;

    for (index = 0; index < REDUCE_DEFINITION_COUNT; index++) {
        if (g_reduce_definitions[index].function == function)
            return &g_reduce_definitions[index];
    }
    return NULL;
}

static uint32_t pack_pass(const reduce_definition *definition, unsigned pass)
{
    return (definition->pool[pass] << (FT_TT_OP_REDUCE_BYTE_POOL * 8u)) |
           (definition->map[pass] << (FT_TT_OP_REDUCE_BYTE_MAP * 8u)) |
           (definition->verb[pass] << (FT_TT_OP_REDUCE_BYTE_VERB * 8u)) |
           (definition->add_eps[pass] << (FT_TT_OP_REDUCE_BYTE_ADD_EPS * 8u));
}

const char *ft_tt_op_reduce_function_name(ft_tt_op_reduce_function function)
{
    const reduce_definition *definition = reduce_definition_for(function);

    return definition ? definition->name : "unknown";
}

uint32_t ft_tt_op_reduce_pass_count(ft_tt_op_reduce_function function)
{
    const reduce_definition *definition = reduce_definition_for(function);

    return definition ? definition->pass_count : 0u;
}

bool ft_tt_op_reduce_uses_gamma(ft_tt_op_reduce_function function)
{
    const reduce_definition *definition = reduce_definition_for(function);

    return definition && (definition->chain & FT_TT_OP_REDUCE_CHAIN_GAMMA) != 0u;
}

bool ft_tt_op_reduce_uses_beta(ft_tt_op_reduce_function function)
{
    const reduce_definition *definition = reduce_definition_for(function);

    return definition && (definition->chain & FT_TT_OP_REDUCE_CHAIN_BETA) != 0u;
}

bool ft_tt_op_reduce_uses_eps(ft_tt_op_reduce_function function)
{
    const reduce_definition *definition = reduce_definition_for(function);
    unsigned pass;

    if (!definition)
        return false;
    for (pass = 0; pass < definition->pass_count; pass++) {
        if (definition->add_eps[pass] != 0u)
            return true;
    }
    return false;
}

bool ft_tt_op_reduce_scaler_is_mean(ft_tt_op_reduce_function function)
{
    const reduce_definition *definition = reduce_definition_for(function);

    return definition && definition->scaler_is_mean;
}

int ft_tt_op_reduce_program(ft_tt_op_reduce_function function, uint32_t *out_pass0,
                            uint32_t *out_pass1, uint32_t *out_final)
{
    const reduce_definition *definition = reduce_definition_for(function);

    if (!definition || !out_pass0 || !out_pass1 || !out_final)
        return FT_TT_EINVAL;

    *out_pass0 = pack_pass(definition, 0u);
    *out_pass1 = (definition->pass_count == 2u) ? pack_pass(definition, 1u) : 0u;
    *out_final = (definition->pass_count << (FT_TT_OP_REDUCE_BYTE_PASS_COUNT * 8u)) |
                 (definition->source << (FT_TT_OP_REDUCE_BYTE_SOURCE * 8u)) |
                 (definition->chain << (FT_TT_OP_REDUCE_BYTE_CHAIN * 8u));
    return FT_TT_OK;
}

int ft_tt_op_reduce_residency_for(const ft_tt_op_reduce_function *functions, size_t count,
                                  uint32_t max_tile_cols, uint32_t tile_bytes,
                                  ft_tt_op_reduce_residency *out_residency)
{
    size_t index;

    if (!functions || count == 0u || !out_residency)
        return FT_TT_EINVAL;
    if (max_tile_cols == 0u || tile_bytes == 0u)
        return FT_TT_EINVAL;

    memset(out_residency, 0, sizeof(*out_residency));
    out_residency->max_tile_cols = max_tile_cols;
    out_residency->tile_bytes = tile_bytes;
    for (index = 0; index < count; index++) {
        if (!reduce_definition_for(functions[index]))
            return FT_TT_EINVAL;
        if (ft_tt_op_reduce_uses_gamma(functions[index]))
            out_residency->carries_gamma = true;
        if (ft_tt_op_reduce_uses_beta(functions[index]))
            out_residency->carries_beta = true;
    }
    if (max_tile_cols > ft_tt_op_reduce_max_tile_cols(out_residency))
        return FT_TT_ENOMEM;
    return FT_TT_OK;
}

uint32_t ft_tt_op_reduce_max_tile_cols(const ft_tt_op_reduce_residency *residency)
{
    uint32_t arena = FT_TT_OP_CB_ARENA_END - FT_TT_OP_CB_ARENA_BASE;
    uint32_t rows;
    uint32_t fixed;

    if (!residency || residency->tile_bytes == 0u)
        return 0u;
    /* P3 extents past these are a caller bug, not a shape, and would overflow below. */
    if (reduce_stage_pages(residency) > FT_TT_OP_REDUCE_CHAIN_CHUNK_MAX ||
        reduce_input_rows(residency) > REDUCE_INPUT_ROWS_MAX)
        return 0u;
    /* cb_in (P3: `input_rows` of them) and cb_mid are always a whole row deep; gamma and beta are too when the
     * residency carries them, and one page each when it does not. ft_tt_op_cb_place()
     * rounds every base up to 64 bytes and every tile size in use is already a multiple of
     * 64, so this arithmetic is exact rather than optimistic. */
    rows = reduce_input_rows(residency) + 1u;
    fixed = REDUCE_FIXED_PAGES - 2u - VECTOR_CB_PAGES_SMALL + 2u * reduce_stage_pages(residency) +
            reduce_out_pages(residency);
    if (residency->carries_gamma)
        rows++;
    else
        fixed++;
    if (residency->carries_beta)
        rows++;
    else
        fixed++;

    if (arena <= fixed * residency->tile_bytes)
        return 0u;
    return (arena - fixed * residency->tile_bytes) / (rows * residency->tile_bytes);
}

int ft_tt_op_reduce_parameters(const ft_tt_op_reduce_request *request,
                               uint32_t out_words[FT_TT_OP_VECTOR_WORDS])
{
    ft_tt_op_reduce_residency residency;
    uint32_t pass0 = 0u;
    uint32_t pass1 = 0u;
    uint32_t final_word = 0u;
    int rc;

    if (!request || !out_words)
        return FT_TT_EINVAL;
    if (request->row_blocks == 0u || request->tile_cols == 0u)
        return FT_TT_EINVAL;
    if (!operand_is_set(&request->input) || !operand_is_set(&request->constants) ||
        !operand_is_set(&request->output))
        return FT_TT_EINVAL;
    rc = ft_tt_op_reduce_program(request->function, &pass0, &pass1, &final_word);
    if (rc != FT_TT_OK)
        return rc;

    /* The ceiling is checked against a residency shaped like THIS member. A caller running
     * several members from one residency has a tighter ceiling still, and gets it from
     * ft_tt_op_reduce_residency_for(), which refuses there. */
    memset(&residency, 0, sizeof(residency));
    residency.max_tile_cols = request->tile_cols;
    residency.tile_bytes = request->input.tile_bytes;
    residency.carries_gamma = ft_tt_op_reduce_uses_gamma(request->function);
    residency.carries_beta = ft_tt_op_reduce_uses_beta(request->function);
    if (request->tile_cols > ft_tt_op_reduce_max_tile_cols(&residency))
        return FT_TT_EINVAL;
    if (request->chain_chunk > FT_TT_OP_REDUCE_CHAIN_CHUNK_MAX)
        return FT_TT_EINVAL;

    memset(out_words, 0, FT_TT_OP_VECTOR_WORDS * sizeof(out_words[0]));
    /* P3: the schedule rides in the FUNCTION slot, which the reduce family never used. */
    out_words[FT_TT_OP_REDUCE_SCHEDULE] =
        request->chain_chunk | (request->stream_input ? FT_TT_OP_REDUCE_SCHED_STREAM : 0u) |
        (request->half_read ? FT_TT_OP_REDUCE_SCHED_HALF_READ : 0u);
    out_words[FT_TT_OP_VECTOR_TILE_COUNT] = request->row_blocks * request->tile_cols;
    out_words[FT_TT_OP_VECTOR_TILE_COLS] = request->tile_cols;
    /* A reduce block may not span two row blocks: the compute thread consumes cb_in a
     * whole row at a time, so the run a block must divide is Wt and not the total. */
    out_words[FT_TT_OP_VECTOR_DEPTH] =
        ft_tt_op_block_depth(request->tile_cols, request->prefetch_tiles);
    out_words[FT_TT_OP_VECTOR_A_X] = request->input.x;
    out_words[FT_TT_OP_VECTOR_A_Y] = request->input.y;
    out_words[FT_TT_OP_VECTOR_A_BASE] = request->input.address;
    out_words[FT_TT_OP_VECTOR_B_X] = request->constants.x;
    out_words[FT_TT_OP_VECTOR_B_Y] = request->constants.y;
    out_words[FT_TT_OP_VECTOR_B_BASE] = request->constants.address;
    out_words[FT_TT_OP_VECTOR_C_X] = request->output.x;
    out_words[FT_TT_OP_VECTOR_C_Y] = request->output.y;
    out_words[FT_TT_OP_VECTOR_C_BASE] = request->output.address;
    out_words[FT_TT_OP_REDUCE_PASS0] = pass0;
    out_words[FT_TT_OP_REDUCE_PASS1] = pass1;
    out_words[FT_TT_OP_REDUCE_FINAL] = final_word;
    return FT_TT_OK;
}

int ft_tt_op_reduce_place_cbs(const ft_tt_op_reduce_residency *residency,
                              ft_tt_launch_cb_entry out_entries[FT_TT_OP_REDUCE_CB_COUNT],
                              void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    ft_tt_op_cb_request requests[FT_TT_OP_REDUCE_CB_COUNT];
    uint32_t page;
    uint32_t row;

    if (!residency || !out_entries || !buffer || !out_mask)
        return FT_TT_EINVAL;
    if (residency->max_tile_cols == 0u || residency->tile_bytes == 0u)
        return FT_TT_EINVAL;
    if (residency->max_tile_cols > ft_tt_op_reduce_max_tile_cols(residency))
        return FT_TT_ENOMEM;

    page = residency->tile_bytes;
    row = residency->max_tile_cols;

    requests[0].index = FT_TT_OP_REDUCE_CB_IN;
    requests[0].page_count = row * reduce_input_rows(residency);
    requests[1].index = FT_TT_OP_REDUCE_CB_SCALER;
    requests[1].page_count = 1u;
    requests[2].index = FT_TT_OP_REDUCE_CB_COL0;
    requests[2].page_count = 1u;
    requests[3].index = FT_TT_OP_REDUCE_CB_COL1;
    requests[3].page_count = 1u;
    requests[4].index = FT_TT_OP_REDUCE_CB_MID;
    requests[4].page_count = row;
    requests[5].index = FT_TT_OP_REDUCE_CB_EPS;
    requests[5].page_count = 1u;
    /* A stream the residency does not carry is still PLACED, at one page: a descriptor that
     * asks for it then blocks in cb_reserve_back and is retired by its bound, rather than
     * the reader writing into a circular buffer the host never configured. */
    requests[6].index = FT_TT_OP_REDUCE_CB_GAMMA;
    requests[6].page_count = residency->carries_gamma ? row : 1u;
    requests[7].index = FT_TT_OP_REDUCE_CB_BETA;
    requests[7].page_count = residency->carries_beta ? row : 1u;
    requests[8].index = FT_TT_OP_REDUCE_CB_STAGE0;
    requests[8].page_count = reduce_stage_pages(residency);
    requests[9].index = FT_TT_OP_REDUCE_CB_STAGE1;
    requests[9].page_count = reduce_stage_pages(residency);
    requests[10].index = FT_TT_OP_REDUCE_CB_OUT;
    /* P3: a chunked chain emits its output in runs of K, so the writer (one page at a time)
     * keeps draining while the next run's first step computes — but only if a run fits. */
    requests[10].page_count = reduce_out_pages(residency);

    {
        size_t index;

        for (index = 0; index < FT_TT_OP_REDUCE_CB_COUNT; index++)
            requests[index].page_bytes = page;
    }
    return ft_tt_op_cb_place(requests, FT_TT_OP_REDUCE_CB_COUNT, out_entries, buffer,
                             buffer_bytes, out_mask);
}

size_t ft_tt_op_reduce_constants_bytes(ft_tt_op_reduce_function function, uint32_t tile_cols,
                                       uint32_t tile_bytes)
{
    size_t tiles = FT_TT_OP_REDUCE_CONST_STREAM_TILE;

    if (!reduce_definition_for(function) || tile_cols == 0u || tile_bytes == 0u)
        return 0u;
    if (ft_tt_op_reduce_uses_gamma(function))
        tiles += tile_cols;
    if (ft_tt_op_reduce_uses_beta(function))
        tiles += tile_cols;
    return tiles * tile_bytes;
}

/**
 * Bytes per element of a constants buffer whose tiles are `tile_bytes` long.
 *
 * THE ONE PLACE THE FAMILY MAPS A PAGE SIZE ONTO AN ELEMENT FORMAT, and the reason it is a
 * function rather than a literal is the defect the GDN family shipped: five circular buffers
 * sized from one operand's `tile_bytes` while the kernels were compiled for another. A
 * constants buffer built at 2048 under a program placed at 4096 is the same failure in this
 * family — the reader issues 4096-byte reads and the scaler tile is half a tile of garbage —
 * and nothing on the card can see it, because a data-movement kernel has no descriptor
 * header. So the builders take the page size the residency was placed at, and this is the
 * only conversion.
 */
static size_t reduce_element_bytes(uint32_t tile_bytes)
{
    if (tile_bytes == FT_TT_OP_TILE_BYTES_BF16)
        return sizeof(uint16_t);
    if (tile_bytes == FT_TT_OP_TILE_BYTES_FP32)
        return sizeof(float);
    return 0u;
}

/** Write one value at a constants buffer's element format. */
static void encode_element(float value, uint32_t tile_bytes, unsigned char *slot)
{
    if (tile_bytes == FT_TT_OP_TILE_BYTES_FP32) {
        memcpy(slot, &value, sizeof(value));
    } else {
        const uint16_t encoded = ft_bf16_from_float(value);

        memcpy(slot, &encoded, sizeof(encoded));
    }
}

/** One tile whose every element is `value`, face-ordered. */
static int build_constant_tile(float value, uint32_t tile_bytes, void *out)
{
    unsigned char plain[VECTOR_TILE_ELEMENTS * sizeof(float)];
    const size_t element_bytes = reduce_element_bytes(tile_bytes);
    size_t index;

    if (element_bytes == 0u)
        return FT_TT_EINVAL;
    for (index = 0; index < VECTOR_TILE_ELEMENTS; index++)
        encode_element(value, tile_bytes, plain + index * element_bytes);
    if (ft_tt_tilize_strip(plain, element_bytes, FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM,
                           FT_TT_OP_TILE_DIM * element_bytes, out) != 0)
        return FT_TT_EINVAL;
    return FT_TT_OK;
}

/**
 * `tile_cols` tiles of a row, each column's value replicated down all 32 rows — which is
 * what turns the gamma/beta multiply into a plain mul_tiles.
 *
 * `value_count` is the REAL length of `values`, which need not be `tile_cols * 32`: a
 * ragged W pads the last tile. Reading past the caller's array instead is exactly the bug a
 * padded width invites, so the count is a parameter rather than derived from the tiles.
 */
static int build_broadcast_row(const float *values, uint32_t value_count, uint32_t tile_cols,
                               float pad, uint32_t tile_bytes, unsigned char *out)
{
    const size_t element_bytes = reduce_element_bytes(tile_bytes);
    uint32_t tile;

    if (element_bytes == 0u)
        return FT_TT_EINVAL;
    for (tile = 0; tile < tile_cols; tile++) {
        unsigned char plain[VECTOR_TILE_ELEMENTS * sizeof(float)];
        uint32_t column;

        for (column = 0; column < FT_TT_OP_TILE_DIM; column++) {
            const uint32_t index = tile * FT_TT_OP_TILE_DIM + column;
            const float value = index < value_count ? values[index] : pad;
            uint32_t row;

            for (row = 0; row < FT_TT_OP_TILE_DIM; row++)
                encode_element(value, tile_bytes,
                               plain + ((size_t)row * FT_TT_OP_TILE_DIM + column) *
                                           element_bytes);
        }
        if (ft_tt_tilize_strip(plain, element_bytes, FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM,
                               FT_TT_OP_TILE_DIM * element_bytes,
                               out + (size_t)tile * tile_bytes) != 0)
            return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

/**
 * Both public builders. `width` is always the REAL W — the scaler and the eps tile are
 * computed from it and never from `tile_cols * 32`, which is the whole of what makes a
 * padded descriptor correct.
 */
static int build_constants_at(ft_tt_op_reduce_function function, uint32_t width,
                              uint32_t tile_cols, float eps, const float *gamma,
                              const float *beta, uint32_t tile_bytes, void *out,
                              size_t out_bytes)
{
    unsigned char *cursor = out;
    size_t needed;
    int rc;

    if (!out || width == 0u || tile_cols == 0u || reduce_element_bytes(tile_bytes) == 0u)
        return FT_TT_EINVAL;
    needed = ft_tt_op_reduce_constants_bytes(function, tile_cols, tile_bytes);
    if (needed == 0u || out_bytes < needed)
        return FT_TT_EINVAL;
    if (ft_tt_op_reduce_uses_gamma(function) && !gamma)
        return FT_TT_EINVAL;
    if (ft_tt_op_reduce_uses_beta(function) && !beta)
        return FT_TT_EINVAL;

    rc = build_constant_tile(ft_tt_op_reduce_scaler_is_mean(function) ? 1.0f / (float)width : 1.0f,
                            tile_bytes, cursor);
    if (rc != FT_TT_OK)
        return rc;
    cursor += tile_bytes;

    /* eps as one more accumulated row sum, and THE VALUE DEPENDS ON THE SCALER because the
     * eps tile goes through the same reduce the activations do. Each row of the tile has 32
     * columns, so the pass contributes `32 * value * scaler`:
     *
     *     scaler 1/W (a MEAN)  ->  32 * (eps*W/32) * (1/W) = eps
     *     scaler 1.0 (a SUM)   ->  32 * (eps/32)   * 1.0   = eps
     *
     * The mean form used to be written unconditionally, which is right for the two members
     * that existed and is wrong by a factor of W for any member that sums — L2_NORM is the
     * first. The slot is written even when it is not read, so the layout never depends on
     * the program. */
    rc = build_constant_tile(ft_tt_op_reduce_uses_eps(function)
                                 ? eps *
                                       (ft_tt_op_reduce_scaler_is_mean(function)
                                            ? (float)width
                                            : 1.0f) /
                                       (float)FT_TT_OP_TILE_DIM
                                 : 0.0f,
                             tile_bytes, cursor);
    if (rc != FT_TT_OK)
        return rc;
    cursor += tile_bytes;

    /* A padded gamma/beta column multiplies a padded ACTIVATION lane, which this member's
     * pad value has already driven to the pool's identity — so the weight there is arbitrary
     * and 0 is chosen because it can never be an inf or a NaN. */
    if (ft_tt_op_reduce_uses_gamma(function)) {
        rc = build_broadcast_row(gamma, width, tile_cols, 0.0f, tile_bytes, cursor);
        if (rc != FT_TT_OK)
            return rc;
        cursor += (size_t)tile_cols * tile_bytes;
    }
    if (ft_tt_op_reduce_uses_beta(function)) {
        rc = build_broadcast_row(beta, width, tile_cols, 0.0f, tile_bytes, cursor);
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

int ft_tt_op_reduce_build_constants(ft_tt_op_reduce_function function, uint32_t width, float eps,
                                    const float *gamma, const float *beta, uint32_t tile_bytes,
                                    void *out, size_t out_bytes)
{
    if (width == 0u || (width % FT_TT_OP_TILE_DIM) != 0u)
        return FT_TT_EINVAL;
    return build_constants_at(function, width, width / FT_TT_OP_TILE_DIM, eps, gamma, beta,
                              tile_bytes, out, out_bytes);
}

/* ================================================================
 * A ragged W, served by padding
 * ================================================================ */

uint32_t ft_tt_op_reduce_padded_tile_cols(uint32_t width)
{
    if (width == 0u)
        return 0u;
    return (width + FT_TT_OP_TILE_DIM - 1u) / FT_TT_OP_TILE_DIM;
}

/** What ONE pass needs a padded lane to be. Kept as three states rather than a float so
 *  that "no constant works" is a value the combining step can carry, not an error code it
 *  has to invent. */
typedef enum pass_pad_requirement {
    PASS_PAD_ANY = 0,  /**< no pass has spoken yet */
    PASS_PAD_ZERO,     /**< the lane must be 0 */
    PASS_PAD_MASK,     /**< the lane must be far below every real value */
    PASS_PAD_NONE      /**< no constant makes this pass's padded lane vanish */
} pass_pad_requirement;

/**
 * What a padded lane has to be for ONE pass to be unaffected by it.
 *
 * The reasoning is in ft_tt_op_vector.h's table; this is that table as code, read off the
 * pass's own pool and map so that a fifth member of the family is classified by its
 * definition row rather than by a name in a switch here.
 */
static pass_pad_requirement pass_pad_for(uint32_t pool, uint32_t map)
{
    switch (map) {
    case FT_TT_OP_REDUCE_MAP_SQUARE_SELF:
        return PASS_PAD_ZERO; /* pad*pad adds nothing to a sum only at pad == 0 */
    case FT_TT_OP_REDUCE_MAP_SUB_EXP:
        return PASS_PAD_MASK; /* exp(pad - col0) is 0 for a pad far below col0 */
    case FT_TT_OP_REDUCE_MAP_SUB_SQUARE:
        /* (pad - col0)^2 is 0 only at pad == col0, and col0 is the ROW mean — a different
         * number for every row, so it cannot be written into the buffer as a constant. And a
         * PER-ROW pad of that mean is no escape either: the pass that PRODUCED col0 is a
         * plain sum over the padded row, so a non-zero pad moves it to m*padded/W. Two
         * passes, one buffer, incompatible requirements — see the header. */
        return PASS_PAD_NONE;
    default:
        return (pool == FT_TT_OP_REDUCE_POOL_MAX) ? PASS_PAD_MASK : PASS_PAD_ZERO;
    }
}

/** The one requirement every pass of `definition` agrees on, or PASS_PAD_NONE. */
static pass_pad_requirement member_pad_requirement(const reduce_definition *definition)
{
    pass_pad_requirement agreed = PASS_PAD_ANY;
    unsigned pass;

    for (pass = 0; pass < definition->pass_count; pass++) {
        const pass_pad_requirement wanted =
            pass_pad_for(definition->pool[pass], definition->map[pass]);

        if (wanted == PASS_PAD_NONE)
            return PASS_PAD_NONE;
        if (agreed == PASS_PAD_ANY)
            agreed = wanted;
        else if (agreed != wanted)
            return PASS_PAD_NONE; /* one pass wants 0 and another wants the mask value */
    }
    return agreed;
}

int ft_tt_op_reduce_pad_value(ft_tt_op_reduce_function function, float *out_value)
{
    const reduce_definition *definition = reduce_definition_for(function);
    pass_pad_requirement requirement;

    if (!definition || !out_value)
        return FT_TT_EINVAL;
    requirement = member_pad_requirement(definition);
    if (requirement == PASS_PAD_ZERO) {
        *out_value = FT_TT_OP_REDUCE_PAD_ZERO_VALUE;
        return FT_TT_OK;
    }
    if (requirement == PASS_PAD_MASK) {
        *out_value = FT_TT_OP_REDUCE_PAD_MASK_VALUE;
        return FT_TT_OK;
    }
    return FT_TT_EINVAL;
}

bool ft_tt_op_reduce_padding_is_exact(ft_tt_op_reduce_function function)
{
    float ignored = 0.0f;

    return ft_tt_op_reduce_pad_value(function, &ignored) == FT_TT_OK;
}

int ft_tt_op_reduce_pad_rows(ft_tt_op_reduce_function function, const float *in, uint32_t rows,
                             uint32_t width, float *out, size_t out_count)
{
    const uint32_t tile_cols = ft_tt_op_reduce_padded_tile_cols(width);
    const uint32_t padded_width = tile_cols * FT_TT_OP_TILE_DIM;
    float pad = 0.0f;
    uint32_t row;
    int rc;

    if (!in || !out || rows == 0u || width == 0u)
        return FT_TT_EINVAL;
    rc = ft_tt_op_reduce_pad_value(function, &pad);
    if (rc != FT_TT_OK)
        return rc;
    if (out_count < (size_t)rows * (size_t)padded_width)
        return FT_TT_EINVAL;

    for (row = 0; row < rows; row++) {
        const float *source = in + (size_t)row * width;
        float *destination = out + (size_t)row * padded_width;
        uint32_t column;

        for (column = 0; column < width; column++)
            destination[column] = source[column];
        for (column = width; column < padded_width; column++)
            destination[column] = pad;
    }
    return FT_TT_OK;
}

int ft_tt_op_reduce_build_constants_ragged(ft_tt_op_reduce_function function, uint32_t width,
                                           float eps, const float *gamma, const float *beta,
                                           uint32_t tile_bytes, void *out, size_t out_bytes)
{
    if (width == 0u)
        return FT_TT_EINVAL;
    /* An exact W needs no padding at all, so every member stays reachable there — the
     * refusal is about the RAGGED case and only about it. */
    if ((width % FT_TT_OP_TILE_DIM) != 0u && !ft_tt_op_reduce_padding_is_exact(function))
        return FT_TT_EINVAL;
    return build_constants_at(function, width, ft_tt_op_reduce_padded_tile_cols(width), eps,
                              gamma, beta, tile_bytes, out, out_bytes);
}
