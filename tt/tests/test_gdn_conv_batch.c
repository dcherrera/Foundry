/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_gdn_conv_batch.c
 * @brief Off-hardware acceptance of the gdn family's CONV ROW BATCH: the flags encoding, the
 *        device's own decode and predicate, the one-pass page count, the extension, and every
 *        refusal (G2a bf16 input, rows past face 1, DELTA, outside strip mode, one row).
 *
 * The device header is #included, so the host/kernel constant agreement is checked against the
 * SAME enums and inline predicates the reader and writer compile.
 *
 * @status added with the row batch.
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

/** One-pass CONV pages at Ornith: conv_out (256 tiles) then the 3-plane window. */
#define ORNITH_CONV_PAGES (256u + 256u * 3u)

static ft_tt_op_operand fp32_operand(uint16_t x, uint16_t y, uint32_t address)
{
    ft_tt_op_operand out;

    out.x = x;
    out.y = y;
    out.address = address;
    out.tile_bytes = FT_TT_OP_TILE_BYTES_FP32;
    return out;
}

/** Ornith's strip CONV over `rows` verify rows from row `row`, row r's window at
 *  0x00600000 + r MiB, batched when `batch`. */
static void fill_batch_request(ft_tt_op_gdn_request *request, uint32_t row, uint32_t rows,
                               uint32_t batch)
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
    request->row = row;
    request->row_count = rows;
    request->conv_row_batch = batch;
    for (r = 0; r < rows && r < FT_TT_OP_GDN_MULTI_ROW_MAX; r++)
        request->state_rows_out[r] = fp32_operand(5u, 6u, 0x00600000u + (r << 20));
    if (rows > 0u)
        request->state_out = request->state_rows_out[0];
}

static void test_constants_agree(void)
{
    const uint32_t field = (uint32_t)kGdnConvBatchMask << kGdnConvBatchShift;
    const uint32_t taken = FT_TT_OP_GDN_FLAG_NOC_REPLICATE | FT_TT_OP_GDN_FLAG_RED_NO_COPY |
                           FT_TT_OP_GDN_FLAG_CONV_STRIP | FT_TT_OP_GDN_FLAG_RED_STRIP_HALF |
                           FT_TT_OP_GDN_FLAG_STATE_OUT | FT_TT_OP_GDN_FLAG_TABLE |
                           (31u << FT_TT_OP_GDN_ROW_SHIFT) | (31u << FT_TT_OP_GDN_ROW_COUNT_SHIFT) |
                           FT_TT_OP_GDN_FLAG_RED_ROWS_ROW0 | FT_TT_OP_GDN_FLAG_CONV_BF16_IN;

    ft_tt_test_step("row batch: host and kernel constants are the same numbers");
    ft_tt_test_check_fmt("the batch shift agrees",
                         (uint32_t)kGdnConvBatchShift == FT_TT_OP_GDN_CONV_BATCH_SHIFT, "%u %u",
                         (unsigned)kGdnConvBatchShift, (unsigned)FT_TT_OP_GDN_CONV_BATCH_SHIFT);
    ft_tt_test_check_fmt("the row limit agrees (faces 0-1)",
                         (uint32_t)kGdnConvBatchRowLimit == FT_TT_OP_GDN_CONV_BATCH_ROW_LIMIT &&
                             FT_TT_OP_GDN_CONV_BATCH_ROW_LIMIT == 16u,
                         "%u %u", (unsigned)kGdnConvBatchRowLimit,
                         (unsigned)FT_TT_OP_GDN_CONV_BATCH_ROW_LIMIT);
    ft_tt_test_check_fmt("the batch field holds 16 and overlaps no other flag",
                         (uint32_t)kGdnConvBatchMask >= 16u && (field & taken) == 0u,
                         "field 0x%08X taken 0x%08X", field, taken);
}

static void test_encoding(void)
{
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];
    uint32_t extension[FT_TT_OP_GDN_WORDS];
    ftg_shape shape;
    uint32_t rows;
    int rc;

    ft_tt_test_step("row batch: the encoding round-trips through the device's own decode");
    for (rows = 2u; rows <= 4u; rows++) {
        fill_batch_request(&request, 0u, rows, 1u);
        rc = ft_tt_op_gdn_parameters(&request, words);
        shape = ftg_decode(words);
        ft_tt_test_check_fmt("a batched strip CONV encodes; the compute sees ONE row",
                             rc == FT_TT_OK && shape.row_count == 1u && shape.row_first == 0u &&
                                 ftg_shape_is_supported(&shape),
                             "rows %u rc=%d count %u", rows, rc, shape.row_count);
        ft_tt_test_check_fmt("the reader and writer see the batch",
                             ftg_conv_batch_rows(words[FT_TT_OP_GDN_FLAGS]) == rows &&
                                 ftg_conv_batch_is_supported(&shape, words[FT_TT_OP_GDN_FLAGS]),
                             "rows %u flags 0x%08X", rows, words[FT_TT_OP_GDN_FLAGS]);
        ft_tt_test_check_fmt("the writer is told ONE pass of pages",
                             words[FT_TT_OP_GDN_TILE_COUNT] == ORNITH_CONV_PAGES, "rows %u %u",
                             rows, words[FT_TT_OP_GDN_TILE_COUNT]);
        rc = ft_tt_op_gdn_row_extension(&request, extension);
        ft_tt_test_check_fmt("the extension carries each row's window address",
                             rc == FT_TT_OK && extension[0] == 0x00600000u &&
                                 extension[rows - 1u] == 0x00600000u + ((rows - 1u) << 20) &&
                                 extension[rows] == 0u,
                             "rows %u rc=%d", rows, rc);
    }
    fill_batch_request(&request, 12u, 4u, 1u);
    rc = ft_tt_op_gdn_parameters(&request, words);
    shape = ftg_decode(words);
    ft_tt_test_check_fmt("rows [12, 16) end at face 1's last row and encode",
                         rc == FT_TT_OK && shape.row_first == 12u &&
                             ftg_conv_batch_is_supported(&shape, words[FT_TT_OP_GDN_FLAGS]),
                         "rc=%d first %u", rc, shape.row_first);

    ft_tt_test_step("row batch: off is the multi-row walk, byte for byte");
    {
        uint32_t off[FT_TT_OP_GDN_WORDS];
        uint32_t index, differing = 0u;

        fill_batch_request(&request, 0u, 4u, 0u);
        rc = ft_tt_op_gdn_parameters(&request, off);
        shape = ftg_decode(off);
        ft_tt_test_check_fmt("conv_row_batch 0 leaves the batch bits 0 and four passes",
                             rc == FT_TT_OK && ftg_conv_batch_rows(off[FT_TT_OP_GDN_FLAGS]) == 0u &&
                                 shape.row_count == 4u &&
                                 off[FT_TT_OP_GDN_TILE_COUNT] == 4u * ORNITH_CONV_PAGES,
                             "rc=%d flags 0x%08X", rc, off[FT_TT_OP_GDN_FLAGS]);
        fill_batch_request(&request, 0u, 4u, 1u);
        (void)ft_tt_op_gdn_parameters(&request, words);
        for (index = 0; index < FT_TT_OP_GDN_WORDS; index++)
            if (index != FT_TT_OP_GDN_FLAGS && index != FT_TT_OP_GDN_TILE_COUNT &&
                words[index] != off[index])
                differing++;
        ft_tt_test_check_fmt("on and off differ only in the flags and the page count",
                             differing == 0u, "%u other word(s) differ", differing);
    }
}

static void test_refusals(void)
{
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];
    ftg_shape shape;

    ft_tt_test_step("row batch: the host refuses what the reader and writer cannot run");
    fill_batch_request(&request, 0u, 4u, 1u);
    request.stream_bf16 = fp32_operand(1u, 2u, 0x00700000u);
    request.stream_bf16.tile_bytes = FT_TT_OP_GDN_BF16_TILE_BYTES;
    ft_tt_test_check("a batch over the G2a bf16 stream refused",
                     ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "");
    fill_batch_request(&request, 0u, 17u, 1u);
    ft_tt_test_check("17 batched rows refused", ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK,
                     "");
    fill_batch_request(&request, 13u, 4u, 1u);
    ft_tt_test_check("rows [13, 17) past face 1 refused",
                     ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "");
    fill_batch_request(&request, 0u, 1u, 1u);
    ft_tt_test_check("a one-row batch refused (it is the row-select walk)",
                     ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "");
    fill_batch_request(&request, 0u, 4u, 1u);
    request.flags = 0u;
    ft_tt_test_check("a batch outside strip mode refused",
                     ft_tt_op_gdn_parameters(&request, words) != FT_TT_OK, "");
    fill_batch_request(&request, 0u, 4u, 1u);
    request.phase = FT_TT_OP_GDN_PHASE_DELTA;
    ft_tt_test_check("a DELTA batch refused", !ft_tt_op_gdn_shape_is_supported(&request), "");

    ft_tt_test_step("row batch: the device predicate refuses the same words");
    fill_batch_request(&request, 0u, 4u, 1u);
    (void)ft_tt_op_gdn_parameters(&request, words);
    shape = ftg_decode(words);
    ft_tt_test_check("the bf16 input flag beside a batch refused on the device",
                     !ftg_conv_batch_is_supported(&shape,
                                                  words[FT_TT_OP_GDN_FLAGS] |
                                                      (uint32_t)kGdnFlagConvBf16In),
                     "");
    ft_tt_test_check("a batch without the strip flag refused on the device",
                     !ftg_conv_batch_is_supported(&shape, words[FT_TT_OP_GDN_FLAGS] &
                                                              ~(uint32_t)kGdnFlagConvStrip),
                     "");
    shape.row_first = 13u;
    ft_tt_test_check("rows past face 1 refused on the device",
                     !ftg_conv_batch_is_supported(&shape, words[FT_TT_OP_GDN_FLAGS]), "");
    shape.row_first = 0u;
    shape.row_count = 2u;
    ft_tt_test_check("a batch beside a multi-row count refused on the device",
                     !ftg_conv_batch_is_supported(&shape, words[FT_TT_OP_GDN_FLAGS]), "");
    shape.row_count = 1u;
    shape.phase = (uint32_t)kGdnPhaseDelta;
    ft_tt_test_check("a DELTA descriptor with batch bits refused on the device",
                     !ftg_conv_batch_is_supported(&shape, words[FT_TT_OP_GDN_FLAGS]), "");
    ft_tt_test_check("batch bits 0 always run",
                     ftg_conv_batch_is_supported(&shape, 0u), "");
}

int main(void)
{
    ft_tt_test_step("Gated-Delta-Net CONV ROW BATCH — off-hardware acceptance");
    test_constants_agree();
    test_encoding();
    test_refusals();
    return ft_tt_test_finish();
}
