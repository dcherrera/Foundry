/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file test_gdn_conv_bf16_in.c
 * @brief Off-hardware acceptance of G2a: PHASE CONV reading the bf16 projection strip and
 *        widening it in the reader (kGdnFlagConvBf16In) instead of a render_widen descriptor.
 *
 * Checks: the host/kernel constants agree; the encoding moves only words 4, 8 and the flag
 * bit (and nothing when the operand is unset); the refusals (no strip, DELTA, a non-bf16
 * operand); the widen rule over all 65,536 bf16 patterns against `x + 0` with denormals read
 * as zero; the IN-PLACE face-row widen the reader runs, against an out-of-place copy, for
 * every element position in either lane; and that every half read's two ends are congruent
 * mod 64 (the NoC DRAM-read rule; an even row sits in the low lane). Build with -DFTG_WIDEN_RED_DESCENDING to see the in-place check fail.
 *
 * @status added 2026-11-20 (G2a).
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

static ft_tt_op_operand operand(uint16_t x, uint16_t y, uint32_t address, uint32_t tile_bytes)
{
    ft_tt_op_operand out;

    out.x = x;
    out.y = y;
    out.address = address;
    out.tile_bytes = tile_bytes;
    return out;
}

/** A one-row strip CONV at the qwen35 shape. */
static void fill_conv_request(ft_tt_op_gdn_request *request)
{
    memset(request, 0, sizeof(*request));
    request->phase = FT_TT_OP_GDN_PHASE_CONV;
    request->state_size = FT_TT_OP_GDN_QWEN35_STATE;
    request->v_heads = FT_TT_OP_GDN_QWEN35_V_HEADS;
    request->k_heads = FT_TT_OP_GDN_QWEN35_K_HEADS;
    request->conv_dim = FT_TT_OP_GDN_QWEN35_CONV_DIM;
    request->conv_taps = FT_TT_OP_GDN_QWEN35_CONV_TAPS;
    request->activation_in = operand(1u, 2u, 0x00100000u, FT_TT_OP_TILE_BYTES_FP32);
    request->activation_out = operand(1u, 2u, 0x00200000u, FT_TT_OP_TILE_BYTES_FP32);
    request->conv_weight = operand(3u, 4u, 0x00300000u, FT_TT_OP_TILE_BYTES_FP32);
    request->state_conv = operand(5u, 6u, 0x00400000u, FT_TT_OP_TILE_BYTES_FP32);
    request->flags = FT_TT_OP_GDN_FLAG_CONV_STRIP;
}

static void test_constants_agree(void)
{
    ft_tt_test_step("G2a: host and kernel constants are the same numbers");
    ft_tt_test_check_fmt("kGdnFlagConvBf16In is FT_TT_OP_GDN_FLAG_CONV_BF16_IN",
                         (uint32_t)kGdnFlagConvBf16In == FT_TT_OP_GDN_FLAG_CONV_BF16_IN, "0x%08X",
                         (unsigned)kGdnFlagConvBf16In);
    ft_tt_test_check_fmt("the flag is above the rows red bit and below the top",
                         (uint32_t)kGdnFlagConvBf16In > (uint32_t)kGdnFlagRedRowsRow0, "0x%08X",
                         (unsigned)kGdnFlagConvBf16In);
    ft_tt_test_check_fmt("kGdnBf16TileBytes is FT_TT_OP_GDN_BF16_TILE_BYTES",
                         (uint32_t)kGdnBf16TileBytes == FT_TT_OP_GDN_BF16_TILE_BYTES, "%u",
                         (unsigned)kGdnBf16TileBytes);
}

static void test_encoding(void)
{
    ft_tt_op_gdn_request request;
    uint32_t base[FT_TT_OP_GDN_WORDS], words[FT_TT_OP_GDN_WORDS];
    unsigned w, moved = 0u;
    int rc;

    ft_tt_test_step("G2a: the encoding moves words 4, 8 and the flag, and nothing else");
    fill_conv_request(&request);
    rc = ft_tt_op_gdn_parameters(&request, base);
    ft_tt_test_check_fmt("the strip CONV encodes", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("unset stream_bf16 leaves the flag clear",
                         (base[FT_TT_OP_GDN_FLAGS] & FT_TT_OP_GDN_FLAG_CONV_BF16_IN) == 0u,
                         "0x%08X", base[FT_TT_OP_GDN_FLAGS]);
    request.stream_bf16 = operand(7u, 8u, 0x00700000u, FT_TT_OP_GDN_BF16_TILE_BYTES);
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("with stream_bf16 it encodes", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("word 4 is the bf16 strip's address",
                         words[FT_TT_OP_GDN_ACTIVATION_IN] == 0x00700000u, "0x%08X",
                         words[FT_TT_OP_GDN_ACTIVATION_IN]);
    ft_tt_test_check_fmt("word 8 is its coordinate (x | y << 16)",
                         words[FT_TT_OP_GDN_CONSTANT_GATE] == (7u | (8u << 16)), "0x%08X",
                         words[FT_TT_OP_GDN_CONSTANT_GATE]);
    ft_tt_test_check_fmt("word 3 is still the output's coordinate",
                         words[FT_TT_OP_GDN_ACTIVATION_XY] == base[FT_TT_OP_GDN_ACTIVATION_XY],
                         "0x%08X", words[FT_TT_OP_GDN_ACTIVATION_XY]);
    ft_tt_test_check_fmt("the flag is set",
                         words[FT_TT_OP_GDN_FLAGS] ==
                             (base[FT_TT_OP_GDN_FLAGS] | FT_TT_OP_GDN_FLAG_CONV_BF16_IN),
                         "0x%08X", words[FT_TT_OP_GDN_FLAGS]);
    for (w = 0; w < FT_TT_OP_GDN_WORDS; w++)
        if (words[w] != base[w] && w != FT_TT_OP_GDN_ACTIVATION_IN &&
            w != FT_TT_OP_GDN_CONSTANT_GATE && w != FT_TT_OP_GDN_FLAGS)
            moved++;
    ft_tt_test_check_fmt("no other word moved", moved == 0u, "%u", moved);
}

static void test_refusals(void)
{
    ft_tt_op_gdn_request request;
    uint32_t words[FT_TT_OP_GDN_WORDS];
    int rc;

    ft_tt_test_step("G2a: refusals");
    fill_conv_request(&request);
    request.stream_bf16 = operand(7u, 8u, 0x00700000u, FT_TT_OP_GDN_BF16_TILE_BYTES);
    request.flags = 0u;
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("refused without strip mode", rc != FT_TT_OK, "rc=%d", rc);
    fill_conv_request(&request);
    request.stream_bf16 = operand(7u, 8u, 0x00700000u, FT_TT_OP_TILE_BYTES_FP32);
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("refused for an fp32 operand", rc != FT_TT_OK, "rc=%d", rc);
    fill_conv_request(&request);
    request.phase = FT_TT_OP_GDN_PHASE_DELTA;
    request.gate = operand(3u, 4u, 0x00300000u, FT_TT_OP_TILE_BYTES_FP32);
    request.state_recurrent = operand(5u, 6u, 0x00400000u, FT_TT_OP_TILE_BYTES_FP32);
    request.stream_bf16 = operand(7u, 8u, 0x00700000u, FT_TT_OP_GDN_BF16_TILE_BYTES);
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("refused on a DELTA descriptor", rc != FT_TT_OK, "rc=%d", rc);
}

/** `x + 0` on the FPU: denormals read as zero (DAZ), the sum is IEEE (-0 + +0 = +0). */
static uint32_t widen_by_add(uint32_t h)
{
    uint32_t bits = ((h & 0x7F80u) == 0u ? (h & 0x8000u) : h) << 16;
    float x, sum;

    memcpy(&x, &bits, sizeof(x));
    sum = x + 0.0f;
    memcpy(&bits, &sum, sizeof(bits));
    return bits;
}

static void test_widen_rule(void)
{
    uint32_t h, wrong = 0u, first = 0u;

    ft_tt_test_step("G2a: the widen rule is `x + 0` for every non-NaN bf16 pattern");
    for (h = 0; h < 0x10000u; h++) {
        const int nan = (h & 0x7F80u) == 0x7F80u && (h & 0x7Fu) != 0u;

        if (!nan && ftg_widen_bf16(h) != widen_by_add(h) && wrong++ == 0u)
            first = h;
    }
    ft_tt_test_check_fmt("65,536 patterns agree (NaNs excluded)", wrong == 0u,
                         "%u wrong, first 0x%04X", wrong, first);
    ft_tt_test_check_fmt("-0 widens to +0", ftg_widen_bf16(0x8000u) == 0u, "0x%08X",
                         ftg_widen_bf16(0x8000u));
    ft_tt_test_check_fmt("1.0 widens to 1.0", ftg_widen_bf16(0x3F80u) == 0x3F800000u, "0x%08X",
                         ftg_widen_bf16(0x3F80u));
}

static void test_in_place_face_row(void)
{
    uint32_t row[16], want[16];
    uint16_t halves[16];
    uint32_t trial, i, lane, bad = 0u;

    ft_tt_test_step("G2a: the in-place face-row widen equals an out-of-place one, either lane");
    for (lane = 0; lane <= 32u; lane += 32u)
        for (trial = 0; trial < 64u; trial++) {
            for (i = 0; i < 16u; i++) {
                halves[i] = (uint16_t)(0x3C00u + trial * 257u + i * 4099u);
                want[i] = ftg_widen_bf16(halves[i]);
                row[i] = 0xDEADBEEFu;
            }
            memcpy((uint8_t *)row + lane, halves, sizeof(halves));
            ftg_widen_face_row_at(row, lane);
            if (memcmp(row, want, sizeof(row)) != 0)
                bad++;
        }
    ft_tt_test_check_fmt("2 x 64 face rows widen exactly in place", bad == 0u, "%u rows wrong",
                         bad);
}

/** The reader's bf16 face-row offset (resident_gdn_reader.c ftg_bf16_face_row_offset). */
static uint32_t bf16_face_row_offset(uint32_t row, uint32_t half)
{
    return ((row / 16u) * 2u + half) * (kGdnBf16TileBytes / 4u) +
           (row % 16u) * (kGdnBf16TileBytes / 64u);
}

/** The reader's fp32 face-row offset of row 0 (resident_gdn_reader.c ftg_face_row_offset). */
static uint32_t fp32_face_row0_offset(uint32_t half)
{
    return half * (FT_TT_OP_TILE_BYTES_FP32 / 4u);
}

static void test_read_congruence(void)
{
    /* A 64-aligned strip, and the +32 B red `gdn_widen_row1` moves it to. */
    static const uint32_t bases[2] = { 0x00100000u, 0x00100020u };
    const uint32_t page = 0x00020000u; /* a CB page: 64-aligned */
    uint32_t b, row, tile, half, bad = 0u, even_low = 0u;

    ft_tt_test_step("G2a: every bf16 half read has its two ends congruent mod 64");
    for (b = 0; b < 2u; b++)
        for (row = 0; row < 32u; row++) {
            const uint32_t lane = ftg_bf16_half_lane(bases[b] + bf16_face_row_offset(row, 0u));

            for (tile = 0; tile < 8u; tile++)
                for (half = 0; half < 2u; half++) {
                    const uint32_t source = bases[b] + tile * kGdnBf16TileBytes +
                                            bf16_face_row_offset(row, half);
                    const uint32_t destination = page + tile * FT_TT_OP_TILE_BYTES_FP32 +
                                                 fp32_face_row0_offset(half) + lane;

                    if ((source % 64u) != (destination % 64u) ||
                        destination + kGdnBf16TileBytes / 64u >
                            page + tile * FT_TT_OP_TILE_BYTES_FP32 +
                                fp32_face_row0_offset(half) + 64u)
                        bad++;
                }
            if (b == 0u && (row % 2u) == 0u && lane == 0u)
                even_low++;
        }
    ft_tt_test_check_fmt("2 bases x 32 rows x 8 tiles x 2 halves congruent, inside the face row",
                         bad == 0u, "%u reads not", bad);
    /* The defect this pins: the even rows (decode's row 0) sit at 0 mod 64, not in the top half. */
    ft_tt_test_check_fmt("the 16 even rows of an aligned strip use the low lane", even_low == 16u,
                         "%u", even_low);
}

int main(void)
{
    ft_tt_test_step("Gated-Delta-Net CONV bf16 stream input (G2a) — off-hardware acceptance");
    test_constants_agree();
    test_encoding();
    test_refusals();
    test_widen_rule();
    test_in_place_face_row();
    test_read_congruence();
    return ft_tt_test_finish();
}
