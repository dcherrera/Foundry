/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * attention_golden_requests.inc
 *   The non-split attention descriptors the knob-off identity is pinned on: every mode the
 *   lanes / rows / chunk paths send (decode, ROWS verify at two strip offsets, CHUNK), at
 *   MiniCPM5-1B, Ornith and Qwen3.8 shapes, bf16 and K8V4 operands. Shared by
 *   test_attention_splitk.c (which encodes them with this tree) and
 *   print_attention_golden_words.c (which encodes them with another tree's encoder, to
 *   produce the literals: make_attention_golden_words.sh). Uses only fields both have.
 */

#define GOLDEN_REQUEST_COUNT 8u

static void golden_operand(ft_tt_op_operand *operand, uint16_t x, uint16_t y,
                           uint32_t address, uint32_t tile_bytes)
{
    operand->x = x;
    operand->y = y;
    operand->address = address;
    operand->tile_bytes = tile_bytes;
}

/** Request `index` of GOLDEN_REQUEST_COUNT, fully written. */
static void golden_request(uint32_t index, ft_tt_op_attention_request *request)
{
    static const struct {
        uint32_t q_rows, dim_tiles, block_tiles, kv_positions, capacity_tiles, query_row;
        uint32_t flags, positions, rows_mask, k_bytes, v_bytes;
    } shapes[GOLDEN_REQUEST_COUNT] = {
        { 8u, 4u, 8u, 1000u, 4096u, 7u, 0u, 0u, 0u, 2048u, 2048u },        /* MiniCPM decode */
        { 4u, 8u, 16u, 28672u, 8192u, 0u, 0u, 0u, 0u, 1088u, 576u },       /* Ornith k8v4 */
        { 6u, 8u, 16u, 28673u, 8192u, 0u, 0u, 0u, 0u, 1088u, 576u },       /* Qwen3.8 decode */
        { 6u, 8u, 16u, 2814u, 8192u, 0u, 2u, 2u, 0x3a000u, 1088u, 576u },  /* Qwen3.8 k1 */
        { 6u, 8u, 16u, 2821u, 8192u, 5u, 2u, 3u, 0x3a800u, 1088u, 576u },  /* verify, 2nd set */
        { 32u, 8u, 16u, 1056u, 8192u, 0u, 1u, 0u, 0u, 1088u, 576u },       /* prefill CHUNK */
        { 7u, 8u, 16u, 2055u, 8192u, 0u, 1u, 0u, 0u, 1088u, 576u },        /* ragged CHUNK */
        { 6u, 8u, 16u, 262144u, 8192u, 0u, 0u, 0u, 0u, 1088u, 576u },      /* 256k decode */
    };
    const uint32_t i = index % GOLDEN_REQUEST_COUNT;

    memset(request, 0, sizeof(*request));
    request->q_rows = shapes[i].q_rows;
    request->dim_tiles = shapes[i].dim_tiles;
    request->block_tiles = shapes[i].block_tiles;
    request->kv_positions = shapes[i].kv_positions;
    request->capacity_tiles = i == 7u ? 8192u + 64u : shapes[i].capacity_tiles;
    request->query_row = shapes[i].query_row;
    request->flags = shapes[i].flags;
    request->positions = shapes[i].positions;
    request->rows_mask_address = shapes[i].rows_mask;
    golden_operand(&request->query, 1u, 2u, 0x00010000u + i * 0x100u, 2048u);
    golden_operand(&request->keys, 3u, 4u, 0x00020000u + i * 0x200u, shapes[i].k_bytes);
    golden_operand(&request->values, 5u, 6u, 0x00030000u + i * 0x300u, shapes[i].v_bytes);
    golden_operand(&request->constants, 7u, 8u, 0x00040000u, 2048u);
    golden_operand(&request->output, 9u, 10u, 0x00050000u + i * 0x500u, 2048u);
}
