/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file test_gdn_multi_row.c
 * @brief Off-hardware acceptance of the gdn family's MULTI-ROW DELTA mode (PM 5): the flags
 *        encoding, the extension block, the refusals, and the placement the reader demands.
 *
 * The device header is #included (ft_tt_op_gdn.c already does), so the host/kernel constant
 * agreement is checked against the SAME enums the kernels compile, not against a copy.
 *
 * @status added 2026-10-07 with the mode.
 * @issues None
 * @todo None
 */

#include <stdint.h>
#include <string.h>

#include "ft_tt_op_gdn.h"
#include "ft_tt_queue.h"
/* The SFPI L1 pointer attribute is the device compiler's; a host reads the header without it. */
#define FTQ_L1_PTR
#include "kernels/resident_gdn_protocol.h"

#include "ft_tt_test_report.h"

static uint8_t g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];

static ft_tt_op_operand fp32_operand(uint16_t x, uint16_t y, uint32_t address)
{
    ft_tt_op_operand out;

    out.x = x;
    out.y = y;
    out.address = address;
    out.tile_bytes = FT_TT_OP_TILE_BYTES_FP32;
    return out;
}

/** Ornith's DELTA shape over `rows` verify rows, row r's state at 0x00500000 + r MiB. */
static void fill_rows_request(ft_tt_op_gdn_request *request, uint32_t rows)
{
    uint32_t r;

    memset(request, 0, sizeof(*request));
    request->phase = FT_TT_OP_GDN_PHASE_DELTA;
    request->state_size = FT_TT_OP_GDN_QWEN35_STATE;
    request->v_heads = FT_TT_OP_GDN_QWEN35_V_HEADS;
    request->k_heads = FT_TT_OP_GDN_QWEN35_K_HEADS;
    request->conv_dim = FT_TT_OP_GDN_QWEN35_CONV_DIM;
    request->conv_taps = FT_TT_OP_GDN_QWEN35_CONV_TAPS;
    request->activation_in = fp32_operand(1u, 2u, 0x00100000u);
    request->activation_out = fp32_operand(1u, 2u, 0x00200000u);
    request->gate = fp32_operand(3u, 4u, 0x00300000u);
    request->state_recurrent = fp32_operand(5u, 6u, 0x00400000u);
    request->row_count = rows;
    for (r = 0; r < rows && r < FT_TT_OP_GDN_MULTI_ROW_MAX; r++)
        request->state_rows_out[r] = fp32_operand(5u, 6u, 0x00500000u + (r << 20));
    if (rows > 0u)
        request->state_out = request->state_rows_out[0];
}

static void test_constants_agree(void)
{
    ft_tt_test_step("multi-row: host and kernel constants are the same numbers");
    ft_tt_test_check_fmt("kGdnCbCarry is FT_TT_OP_GDN_CB_CARRY",
                         (uint32_t)kGdnCbCarry == FT_TT_OP_GDN_CB_CARRY, "%u %u",
                         (unsigned)kGdnCbCarry, (unsigned)FT_TT_OP_GDN_CB_CARRY);
    ft_tt_test_check_fmt("the row-count shift agrees",
                         (uint32_t)kGdnRowCountShift == FT_TT_OP_GDN_ROW_COUNT_SHIFT, "%u %u",
                         (unsigned)kGdnRowCountShift, (unsigned)FT_TT_OP_GDN_ROW_COUNT_SHIFT);
    ft_tt_test_check_fmt("the row cap agrees",
                         (uint32_t)kGdnMultiRowMax == FT_TT_OP_GDN_MULTI_ROW_MAX, "%u %u",
                         (unsigned)kGdnMultiRowMax, (unsigned)FT_TT_OP_GDN_MULTI_ROW_MAX);
    ft_tt_test_check_fmt("the row count sits above the row select and inside the word",
                         (uint32_t)kGdnRowCountShift >= (uint32_t)kGdnRowShift + 5u &&
                             (uint32_t)kGdnRowCountShift + 5u <= 32u,
                         "select %u, count %u", (unsigned)kGdnRowShift,
                         (unsigned)kGdnRowCountShift);
    ft_tt_test_check_fmt("the extension block is the next parameter slot (16 words)",
                         (uint32_t)kGdnWords == FT_TT_QUEUE_PARAM_WORDS, "%u",
                         (unsigned)kGdnWords);
    ft_tt_test_check_fmt("the rows red flag agrees and is above the row count",
                         (uint32_t)kGdnFlagRedRowsRow0 == FT_TT_OP_GDN_FLAG_RED_ROWS_ROW0 &&
                             (uint32_t)kGdnFlagRedRowsRow0 >= (1u << (kGdnRowCountShift + 5)),
                         "0x%08X", (unsigned)kGdnFlagRedRowsRow0);
    ft_tt_test_check_fmt("the gate stride is the host's gate allocation",
                         ftg_gate_stride(32u) == 256u && ftg_gate_stride(33u) == 320u, "%u %u",
                         ftg_gate_stride(32u), ftg_gate_stride(33u));
}

static void test_encoding(void)
{
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];
    uint32_t extension[FT_TT_OP_GDN_WORDS];
    ftg_shape shape;
    int rc;

    ft_tt_test_step("multi-row: the encoding round-trips through the device's own decode");
    fill_rows_request(&request, 4u);
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("a 4-row DELTA encodes", rc == FT_TT_OK, "rc=%d", rc);
    shape = ftg_decode(words);
    ft_tt_test_check_fmt("the device decodes rows [0, 4)",
                         shape.row_first == 0u && shape.row_count == 4u &&
                             ftg_shape_is_supported(&shape),
                         "first %u count %u", shape.row_first, shape.row_count);
    ft_tt_test_check_fmt("the writer is told 4x the CB_OUT pages", words[FT_TT_OP_GDN_TILE_COUNT] == 2560u,
                         "%u", words[FT_TT_OP_GDN_TILE_COUNT]);
    rc = ft_tt_op_gdn_row_extension(&request, extension);
    ft_tt_test_check_fmt("the extension carries each row's state address, the rest zero",
                         rc == FT_TT_OK && extension[0] == 0x00500000u &&
                             extension[3] == 0x00800000u && extension[4] == 0u,
                         "rc=%d 0x%08X 0x%08X 0x%08X", rc, extension[0], extension[3],
                         extension[4]);

    fill_rows_request(&request, 1u);
    request.row_count = 0u;
    rc = ft_tt_op_gdn_parameters(&request, words);
    shape = ftg_decode(words);
    ft_tt_test_check_fmt("row_count 0 encodes the one-row walk byte for byte (count bits 0)",
                         rc == FT_TT_OK && ((words[FT_TT_OP_GDN_FLAGS] >> 21) & 31u) == 0u &&
                             shape.row_count == 1u && words[FT_TT_OP_GDN_TILE_COUNT] == 640u,
                         "rc=%d flags 0x%08X", rc, words[FT_TT_OP_GDN_FLAGS]);
    request.row_count = 1u;
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("row_count 1 is the same bytes", rc == FT_TT_OK &&
                         ((words[FT_TT_OP_GDN_FLAGS] >> 21) & 31u) == 0u, "flags 0x%08X",
                         words[FT_TT_OP_GDN_FLAGS]);
}

static void test_refusals(void)
{
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];

    ft_tt_test_step("multi-row: the refusals (each one is a check that CAN fail)");
    fill_rows_request(&request, 17u);
    ft_tt_test_check("17 rows refused", ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "");
    fill_rows_request(&request, 4u);
    request.row = 30u;
    ft_tt_test_check("rows past the tile refused", ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "");
    fill_rows_request(&request, 4u);
    request.phase = FT_TT_OP_GDN_PHASE_CONV;
    ft_tt_test_check("a multi-row CONV outside strip mode refused",
                     !ft_tt_op_gdn_shape_is_supported(&request), "");
    fill_rows_request(&request, 4u);
    request.state_rows_out[2] = fp32_operand(7u, 6u, 0x00700000u);
    ft_tt_test_check("a row state in another bank refused",
                     ft_tt_op_gdn_row_extension(&request, words) != FT_TT_OK, "");
    fill_rows_request(&request, 4u);
    memset(&request.state_rows_out[3], 0, sizeof(request.state_rows_out[3]));
    ft_tt_test_check("an unset row state refused",
                     ft_tt_op_gdn_row_extension(&request, words) != FT_TT_OK, "");
    fill_rows_request(&request, 1u);
    ft_tt_test_check("a one-row request has no extension",
                     ft_tt_op_gdn_row_extension(&request, words) != FT_TT_OK, "");
    fill_rows_request(&request, 16u);
    request.v_heads = 64u;
    ft_tt_test_check("16 rows of 64-head gate blocks (> one scratch page) refused",
                     !ft_tt_op_gdn_shape_is_supported(&request), "");
}

/** Ornith's CONV over `rows` verify rows in strip mode, row r's window at 0x00600000 + r MiB. */
static void fill_conv_rows_request(ft_tt_op_gdn_request *request, uint32_t rows)
{
    uint32_t r;

    memset(request, 0, sizeof(*request));
    request->phase = FT_TT_OP_GDN_PHASE_CONV;
    request->state_size = FT_TT_OP_GDN_QWEN35_STATE;
    request->v_heads = FT_TT_OP_GDN_QWEN35_V_HEADS;
    request->k_heads = FT_TT_OP_GDN_QWEN35_K_HEADS;
    request->conv_dim = FT_TT_OP_GDN_QWEN35_CONV_DIM;
    request->conv_taps = FT_TT_OP_GDN_QWEN35_CONV_TAPS;
    request->activation_in = fp32_operand(1u, 2u, 0x00100000u);
    request->activation_out = fp32_operand(1u, 2u, 0x00200000u);
    request->conv_weight = fp32_operand(3u, 4u, 0x00300000u);
    request->state_conv = fp32_operand(5u, 6u, 0x00400000u);
    request->flags = FT_TT_OP_GDN_FLAG_CONV_STRIP;
    request->row_count = rows;
    for (r = 0; r < rows && r < FT_TT_OP_GDN_MULTI_ROW_MAX; r++)
        request->state_rows_out[r] = fp32_operand(5u, 6u, 0x00600000u + (r << 20));
    if (rows > 0u)
        request->state_out = request->state_rows_out[0];
}

/** PM 6: the multi-row CONV encodes, round-trips, extends and refuses like the DELTA. */
static void test_conv_rows(void)
{
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];
    uint32_t extension[FT_TT_OP_GDN_WORDS];
    ftg_shape shape;
    int rc;

    ft_tt_test_step("multi-row CONV: encoding, extension and refusals");
    fill_conv_rows_request(&request, 4u);
    rc = ft_tt_op_gdn_parameters(&request, words);
    shape = ftg_decode(words);
    ft_tt_test_check_fmt("a 4-row strip CONV encodes and decodes rows [0, 4)",
                         rc == FT_TT_OK && shape.phase == (uint32_t)kGdnPhaseConv &&
                             shape.row_first == 0u && shape.row_count == 4u &&
                             ftg_shape_is_supported(&shape),
                         "rc=%d first %u count %u", rc, shape.row_first, shape.row_count);
    ft_tt_test_check_fmt("the writer is told 4x the one-row CONV pages",
                         words[FT_TT_OP_GDN_TILE_COUNT] == 4u * (256u + 256u * 3u), "%u",
                         words[FT_TT_OP_GDN_TILE_COUNT]);
    rc = ft_tt_op_gdn_row_extension(&request, extension);
    ft_tt_test_check_fmt("the extension carries each row's window address",
                         rc == FT_TT_OK && extension[0] == 0x00600000u &&
                             extension[3] == 0x00900000u && extension[4] == 0u,
                         "rc=%d 0x%08X 0x%08X", rc, extension[0], extension[3]);
    fill_conv_rows_request(&request, 4u);
    request.state_rows_out[1] = fp32_operand(5u, 7u, 0x00700000u);
    ft_tt_test_check("a row window outside the window's bank refused",
                     ft_tt_op_gdn_row_extension(&request, words) != FT_TT_OK, "");
    fill_conv_rows_request(&request, 17u);
    ft_tt_test_check("17 CONV rows refused", ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK,
                     "");
    fill_conv_rows_request(&request, 1u);
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("a one-row CONV is the shipped bytes (count bits 0, one row's pages)",
                         rc == FT_TT_OK && ((words[FT_TT_OP_GDN_FLAGS] >> 21) & 31u) == 0u &&
                             words[FT_TT_OP_GDN_TILE_COUNT] == 256u + 256u * 3u,
                         "rc=%d flags 0x%08X", rc, words[FT_TT_OP_GDN_FLAGS]);
}

/** The pages the reader's multi-row refusal (delta_is_executable) demands, at Ornith. */
static void test_placement(void)
{
    ft_tt_op_gdn_request request;
    ft_tt_launch_cb_entry entries[FT_TT_OP_GDN_CB_COUNT];
    uint64_t mask = 0u;
    uint32_t index, carry = 0u, stage = 0u, scaler = 0u;
    int rc;

    ft_tt_test_step("multi-row: the placement serves the reader's multi-row checks at Ornith");
    fill_rows_request(&request, 4u);
    rc = ft_tt_op_gdn_place_cbs(&request, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("placed", rc == FT_TT_OK, "rc=%d", rc);
    for (index = 0; index < FT_TT_OP_GDN_CB_COUNT; index++) {
        if (entries[index].index == FT_TT_OP_GDN_CB_CARRY)
            carry = entries[index].page_count;
        if (entries[index].index == FT_TT_OP_GDN_CB_STAGE)
            stage = entries[index].page_count;
        if (entries[index].index == FT_TT_OP_GDN_CB_SCALER)
            scaler = entries[index].page_count;
    }
    ft_tt_test_check_fmt("the mask names the carry", (mask >> FT_TT_OP_GDN_CB_CARRY) & 1u,
                         "0x%016llX", (unsigned long long)mask);
    ft_tt_test_check_fmt("CB_CARRY holds a head's state on its tile-row grid",
                         carry >= 16u && carry % 4u == 0u, "%u pages", carry);
    ft_tt_test_check_fmt("CB_STAGE holds k, q, v and the gate page", stage >= 13u, "%u pages",
                         stage);
    ft_tt_test_check_fmt("CB_SCALER holds two rows' replicas on the 8-page grid",
                         scaler >= 16u && scaler % 8u == 0u, "%u pages", scaler);
}

int main(void)
{
    ft_tt_test_step("Gated-Delta-Net MULTI-ROW DELTA — off-hardware acceptance");
    test_constants_agree();
    test_encoding();
    test_refusals();
    test_placement();
    test_conv_rows();
    return ft_tt_test_finish();
}
