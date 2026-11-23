/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_sample.c
 * @brief Implementation of the sampling op pair's host driver. See ft_tt_op_sample.h,
 *   which carries the semantics this file assumes and the reference they came from.
 *
 * Everything here is PURE — no device access, no allocation. The enqueue itself is
 * ft_tt_queue_write_op_parameters() plus ft_tt_queue_enqueue(), which the caller does,
 * because a resident op has no launch of its own to own.
 *
 * @status Off-hardware suite green in tests/test_topk_ops.c (132 checks); driving the card
 *   green in tests/test_phase6_topk.c (40 checks).
 * @issues None
 * @todo None
 */

#include "ft_tt_op_sample.h"

#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_op_l1_layout.h"

/** Pages in the logit CB, as a multiple of the prefetch depth. TWO blocks: the reader fills
 *  one while the selector drains the other, which is the whole point of a depth and is also
 *  the invariant that keeps a reserved block contiguous. */
#define SAMPLE_CB_BLOCKS 2u

/** CB indices used as plain arena ALLOCATIONS rather than streams. They carry a page size
 *  and one page so the allocator places them; no kernel ever pushes or pops them. */
#define SAMPLE_CB_RESULT 1u
#define SAMPLE_CB_STATE 2u

/** An operand is usable when it names a tile size. The COORDINATE is not checked: it comes
 *  from ft_tt_coord and a caller that invented one has a bigger problem than a range check
 *  (spec §7.3). */
static bool operand_is_set(const ft_tt_op_operand *operand)
{
    return operand->tile_bytes != 0u;
}

/** float -> its bit pattern, without an aliasing violation. */
static uint32_t float_bits(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/** A bit pattern -> the float it denotes. */
static float bits_float(uint32_t bits)
{
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t ft_tt_op_sample_row_tiles(uint32_t vocab)
{
    if (vocab == 0u || (vocab % 32u) != 0u)
        return 0u;
    return vocab / 32u;
}

int ft_tt_op_sample_parameters(const ft_tt_op_topk_request *request,
                               uint32_t out_words[FT_TT_OP_SAMPLE_WORDS])
{
    if (!request || !out_words)
        return FT_TT_EINVAL;
    if (request->tile_count == 0u || request->row >= 32u)
        return FT_TT_EINVAL;
    if (request->rows > FT_TT_OP_SAMPLE_MAX_ROWS ||
        (request->rows > 1u && (request->row + request->rows > 32u ||
                                (request->flags & FT_TT_OP_SAMPLE_FLAG_TAIL_STATS) != 0u)))
        return FT_TT_EINVAL;
    if (request->select_count == 0u || request->select_count > FT_TT_OP_SAMPLE_MAX_SELECT)
        return FT_TT_EINVAL;
    if (!operand_is_set(&request->logits))
        return FT_TT_EINVAL;
    if (request->result_address == 0u || request->state_address == 0u)
        return FT_TT_EINVAL;
    /* The hand-off is a 16-byte transfer and both ends must be aligned. The kernel refuses it
     * too — this check is the one that gives a CALLER an error code instead of a descriptor
     * that dies on the card (see FT_TT_OP_SAMPLE_TOKEN_OUT_ALIGNMENT). */
    if (request->token_out.address != 0u &&
        (request->token_out.address % FT_TT_OP_SAMPLE_TOKEN_OUT_ALIGNMENT) != 0u)
        return FT_TT_EINVAL;

    memset(out_words, 0, FT_TT_OP_SAMPLE_WORDS * sizeof(out_words[0]));
    out_words[FT_TT_OP_SAMPLE_TILE_COUNT] = request->tile_count;
    out_words[FT_TT_OP_SAMPLE_ROW] = request->row;
    out_words[FT_TT_OP_SAMPLE_SELECT_COUNT] = request->select_count;
    out_words[FT_TT_OP_SAMPLE_DEPTH] =
        ft_tt_op_block_depth(request->tile_count, request->prefetch_tiles);
    out_words[FT_TT_OP_SAMPLE_LOGITS_X] = request->logits.x;
    out_words[FT_TT_OP_SAMPLE_LOGITS_Y] = request->logits.y;
    out_words[FT_TT_OP_SAMPLE_LOGITS_BASE] = request->logits.address;
    out_words[FT_TT_OP_SAMPLE_RESULT_ADDRESS] = request->result_address;
    out_words[FT_TT_OP_SAMPLE_STATE_ADDRESS] = request->state_address;
    out_words[FT_TT_OP_SAMPLE_TOKEN_OUT_X] = request->token_out.x;
    out_words[FT_TT_OP_SAMPLE_TOKEN_OUT_Y] = request->token_out.y;
    out_words[FT_TT_OP_SAMPLE_TOKEN_OUT_BASE] = request->token_out.address;
    out_words[FT_TT_OP_SAMPLE_FLAGS] = request->flags;
    if ((request->flags & FT_TT_OP_SAMPLE_FLAG_TAIL_STATS) != 0u) {
        if (request->tail_bins == 0u || request->tail_bins > FT_TT_OP_SAMPLE_TAIL_MAX_BINS)
            return FT_TT_EINVAL;
        out_words[FT_TT_OP_SAMPLE_TAIL_BINS] = request->tail_bins;
    }
    out_words[FT_TT_OP_SAMPLE_ROWS] = request->rows > 1u ? request->rows : 0u;
    out_words[FT_TT_OP_SAMPLE_HELPER_AREA] = request->helper_area;
    return FT_TT_OK;
}

int ft_tt_op_sample_tail_keys_encode(const uint32_t *keys, uint32_t bins,
                                     uint32_t out_words[FT_TT_OP_SAMPLE_TAIL_MAX_BINS])
{
    uint32_t j;

    if (!keys || !out_words || bins == 0u || bins > FT_TT_OP_SAMPLE_TAIL_MAX_BINS)
        return FT_TT_EINVAL;
    memset(out_words, 0, FT_TT_OP_SAMPLE_TAIL_MAX_BINS * sizeof(out_words[0]));
    for (j = 0u; j < bins; j++) {
        if (keys[j] > 0xFFFFu || (j > 0u && keys[j] >= keys[j - 1u]))
            return FT_TT_EINVAL;
        out_words[j] = keys[j];
    }
    return FT_TT_OK;
}

int ft_tt_op_sample_tail_counts_decode(const uint32_t *area, uint32_t bins, uint32_t *out_counts)
{
    uint32_t j;

    if (!area || !out_counts || bins == 0u || bins > FT_TT_OP_SAMPLE_TAIL_MAX_BINS)
        return FT_TT_EINVAL;
    if (area[FT_TT_OP_SAMPLE_TAIL_STATUS_WORD] != FT_TT_OP_SAMPLE_TAIL_STATUS_MAGIC)
        return FT_TT_EINVAL;
    for (j = 0u; j <= bins; j++)
        out_counts[j] = area[j];
    return FT_TT_OK;
}

int ft_tt_op_sample_place_cbs(const ft_tt_op_topk_request *request,
                              ft_tt_launch_cb_entry out_entries[FT_TT_OP_SAMPLE_CB_COUNT],
                              void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    ft_tt_op_cb_request requests[FT_TT_OP_SAMPLE_CB_COUNT];
    uint32_t depth;

    if (!request || !out_entries || !buffer || !out_mask)
        return FT_TT_EINVAL;
    if (request->tile_count == 0u || !operand_is_set(&request->logits))
        return FT_TT_EINVAL;

    depth = ft_tt_op_block_depth(request->tile_count, request->prefetch_tiles);

    requests[0].index = FT_TT_OP_SAMPLE_CB_LOGITS;
    requests[0].page_bytes = request->logits.tile_bytes;
    requests[0].page_count = depth * SAMPLE_CB_BLOCKS;
    requests[1].index = SAMPLE_CB_RESULT;
    requests[1].page_bytes = FT_TT_OP_SAMPLE_RESULT_BYTES;
    /* One result block a row of a multi-row TOPK, then the helper area. */
    requests[1].page_count = FT_TT_OP_SAMPLE_RESULT_PAGES;
    requests[2].index = SAMPLE_CB_STATE;
    requests[2].page_bytes = FT_TT_OP_SAMPLE_STATE_BYTES;
    requests[2].page_count = 1u;

    return ft_tt_op_cb_place(requests, FT_TT_OP_SAMPLE_CB_COUNT, out_entries, buffer,
                             buffer_bytes, out_mask);
}

int ft_tt_op_sample_state_build(const ft_tt_op_sample_params *params, uint32_t *out,
                                size_t out_words)
{
    uint32_t entry;

    if (!params || !out || out_words < FT_TT_OP_SAMPLE_STATE_WORDS)
        return FT_TT_EINVAL;
    /* xorshift32 has a fixed point at zero: a "random" sampler that returns the same draw
     * forever reads as a temperature bug for a long time before anyone suspects the seed. */
    if (params->rng_seed == 0u)
        return FT_TT_EINVAL;
    if (params->repeat_window > FT_TT_OP_SAMPLE_RECENT_CAPACITY)
        return FT_TT_EINVAL;
    if (params->eos_count > FT_TT_OP_SAMPLE_EOS_CAPACITY)
        return FT_TT_EINVAL;

    memset(out, 0, FT_TT_OP_SAMPLE_STATE_WORDS * sizeof(out[0]));
    out[FT_TT_OP_SAMPLE_STATE_MAGIC] = FT_TT_OP_SAMPLE_STATE_MAGIC_VALUE;
    out[FT_TT_OP_SAMPLE_STATE_RNG] = params->rng_seed;
    out[FT_TT_OP_SAMPLE_STATE_TEMPERATURE_BITS] = float_bits(params->temperature);
    out[FT_TT_OP_SAMPLE_STATE_TOP_P_BITS] = float_bits(params->top_p);
    out[FT_TT_OP_SAMPLE_STATE_REPEAT_PENALTY_BITS] = float_bits(params->repeat_penalty);
    out[FT_TT_OP_SAMPLE_STATE_TOP_K] = params->top_k;
    out[FT_TT_OP_SAMPLE_STATE_REPEAT_WINDOW] = params->repeat_window;
    out[FT_TT_OP_SAMPLE_STATE_TOKEN_BUDGET] = params->token_budget;
    out[FT_TT_OP_SAMPLE_STATE_EOS_COUNT] = params->eos_count;
    for (entry = 0u; entry < params->eos_count; entry++)
        out[FT_TT_OP_SAMPLE_EOS_OFFSET_WORDS + entry] = params->eos_ids[entry];
    return FT_TT_OK;
}

int ft_tt_op_sample_result_decode(const uint32_t *block, uint32_t max_pairs,
                                  uint32_t *out_indices, float *out_values,
                                  uint32_t *out_count, uint32_t *out_token)
{
    const uint32_t *pairs;
    uint32_t count;
    uint32_t entry;

    if (!block || !out_indices || !out_values || !out_count)
        return FT_TT_EINVAL;
    if (block[FT_TT_OP_SAMPLE_RESULT_MAGIC] != FT_TT_OP_SAMPLE_RESULT_MAGIC_VALUE)
        return FT_TT_EINVAL;

    count = block[FT_TT_OP_SAMPLE_RESULT_COUNT];
    if (count > max_pairs || count > FT_TT_OP_SAMPLE_MAX_SELECT)
        return FT_TT_EINVAL;

    pairs = block + FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS;
    for (entry = 0u; entry < count; entry++) {
        out_indices[entry] = pairs[entry * FT_TT_OP_SAMPLE_PAIR_WORDS + 0u];
        out_values[entry] = bits_float(pairs[entry * FT_TT_OP_SAMPLE_PAIR_WORDS + 1u]);
    }
    *out_count = count;
    if (out_token && block[FT_TT_OP_SAMPLE_RESULT_TOKEN_VALID] != 0u)
        *out_token = block[FT_TT_OP_SAMPLE_RESULT_TOKEN];
    return FT_TT_OK;
}
