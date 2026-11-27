/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_gdn_state_round.c
 * @brief Off-hardware acceptance of d1, the GDN state round bias (kGdnFlagStateRound in `gdn`,
 *        kGdncDeltaFlagStateRound in `gdnc`): before each state pack the compute thread adds
 *        0x1000 to the fp32 words as integers, so the FPU's truncating Tf32 unpack of the
 *        stored state rounds instead.
 *
 * Checks: the host and kernel constants agree, the SFPU kernel adds the protocol's bias, and
 * neither flag collides with another bit of its word; the encoders move only the flag bit and
 * refuse it on CONV; the rule (stored bits, then the Tf32 truncation) rounds every finite
 * value to the nearer Tf32 with ties away from zero, including the edges: a carry into the
 * exponent, +-0, the largest Tf32 band rounding to inf, inf, NaN and its one wrap; and the
 * stored value is within half a Tf32 ulp of the computed one (what a non-truncating reader
 * sees). dswap (kGdnFlagDecaySwap): the constants agree, it encodes on DELTA only, and the reader's
 * decay rounding (ftg_round_bits9) is nearest even at 9 bits. Build with -DGDN_STATE_ROUND_RED to
 * model a bias of 0x0FFF and see the ties fail.
 *
 * @status added 2026-11-27 (d1).
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_op_gdn.h"
#include "ft_tt_op_gdnc.h"
#include "ft_tt_queue.h"
/* The SFPI L1 pointer attribute is the device compiler's; a host reads the header without it. */
#define FTQ_L1_PTR
#include "kernels/resident_gdn_protocol.h"
#include "kernels/resident_gdnc_protocol.h"

#include "ft_tt_test_report.h"

#ifndef FT_TT_KERNEL_DIR
#define FT_TT_KERNEL_DIR "../kernels"
#endif

/* The Tf32 unpack: 10 of the 23 mantissa bits survive. */
#define TF32_KEEP_MASK 0xFFFFE000u
#define TF32_ULP_BITS 0x2000u

static float bits_float(uint32_t bits)
{
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

/** The fp32 bits the card stores (the red build models a wrong bias). */
static uint32_t stored_bits(uint32_t bits)
{
#ifdef GDN_STATE_ROUND_RED
    return bits + 0x0FFFu;
#else
    return ftg_state_round_stored_bits(bits);
#endif
}

/** What every FPU reader of the stored state sees. */
static uint32_t read_bits(uint32_t bits)
{
    return stored_bits(bits) & TF32_KEEP_MASK;
}

static ft_tt_op_operand operand(uint16_t x, uint16_t y, uint32_t address)
{
    ft_tt_op_operand out;

    out.x = x;
    out.y = y;
    out.address = address;
    out.tile_bytes = FT_TT_OP_TILE_BYTES_FP32;
    return out;
}

/** `#define name value` in a kernel header, read as text: -1 when missing. */
static long header_define(const char *file, const char *name)
{
    char path[512], line[512];
    long value = -1;
    FILE *in;

    snprintf(path, sizeof(path), "%s/%s", FT_TT_KERNEL_DIR, file);
    in = fopen(path, "r");
    if (in == NULL)
        return -1;
    while (fgets(line, sizeof(line), in) != NULL) {
        char word[128];
        char *end;
        long parsed;

        if (sscanf(line, " #define %127s", word) != 1 || strcmp(word, name) != 0)
            continue;
        parsed = strtol(strstr(line, name) + strlen(name), &end, 0);
        if (end != strstr(line, name) + strlen(name))
            value = parsed;
    }
    fclose(in);
    return value;
}

static void test_constants_agree(void)
{
    const uint32_t gdn_taken = (uint32_t)kGdnFlagStateRound | (uint32_t)kGdnFlagNocReplicate | (uint32_t)kGdnFlagRedNoCopy |
                               (uint32_t)kGdnFlagConvStrip | (uint32_t)kGdnFlagRedStripHalf |
                               (uint32_t)kGdnFlagStateOut | (uint32_t)kGdnFlagTable |
                               ((uint32_t)kGdnRowMask << kGdnRowShift) |
                               ((uint32_t)kGdnRowCountMask << kGdnRowCountShift) |
                               (uint32_t)kGdnFlagRedRowsRow0 | (uint32_t)kGdnFlagConvBf16In;
    const uint32_t gdnc_taken = 0xFFu | (uint32_t)kGdncDeltaFlagCompact |
                                (uint32_t)kGdncFlagSerialize | (uint32_t)kGdncDeltaFlagTable;
    const long sfpu_bias = header_define("ft_tensix_sfpu_tf32_round.h", "FT_SFPU_TF32_ROUND_BIAS");

    ft_tt_test_step("d1: host and kernel constants are the same numbers");
    ft_tt_test_check_fmt("kGdnFlagStateRound is FT_TT_OP_GDN_FLAG_STATE_ROUND",
                         (uint32_t)kGdnFlagStateRound == FT_TT_OP_GDN_FLAG_STATE_ROUND, "0x%08X",
                         (unsigned)kGdnFlagStateRound);
    ft_tt_test_check_fmt("kGdncDeltaFlagStateRound is FT_TT_OP_GDNC_DELTA_FLAG_STATE_ROUND",
                         (uint32_t)kGdncDeltaFlagStateRound ==
                             FT_TT_OP_GDNC_DELTA_FLAG_STATE_ROUND,
                         "0x%08X", (unsigned)kGdncDeltaFlagStateRound);
    ft_tt_test_check_fmt("the gdn round bit is one bit and collides with no other field",
                         ((uint32_t)kGdnFlagStateRound & ((uint32_t)kGdnFlagStateRound - 1u)) ==
                                 0u &&
                             ((uint32_t)kGdnFlagStateRound & (gdn_taken &
                                                               ~(uint32_t)kGdnFlagStateRound)) == 0u,
                         "taken 0x%08X", (unsigned)gdn_taken);
    ft_tt_test_check_fmt("kGdnFlagDecaySwap is FT_TT_OP_GDN_FLAG_DECAY_SWAP, one bit, free",
                         (uint32_t)kGdnFlagDecaySwap == FT_TT_OP_GDN_FLAG_DECAY_SWAP &&
                             ((uint32_t)kGdnFlagDecaySwap & ((uint32_t)kGdnFlagDecaySwap - 1u)) ==
                                 0u &&
                             ((uint32_t)kGdnFlagDecaySwap & gdn_taken) == 0u,
                         "0x%08X", (unsigned)kGdnFlagDecaySwap);
    ft_tt_test_check_fmt("the gdnc bit collides with no other field",
                         ((uint32_t)kGdncDeltaFlagStateRound & gdnc_taken) == 0u, "taken 0x%08X",
                         (unsigned)gdnc_taken);
    ft_tt_test_check_fmt("the SFPU kernel adds kGdnStateRoundBias (half a Tf32 ulp)",
                         sfpu_bias == (long)kGdnStateRoundBias &&
                             2u * (uint32_t)kGdnStateRoundBias == TF32_ULP_BITS,
                         "FT_SFPU_TF32_ROUND_BIAS=%ld (FT_TT_KERNEL_DIR=%s)", sfpu_bias,
                         FT_TT_KERNEL_DIR);
}

/** A one-head-row DELTA at the qwen35 shape. */
static void fill_delta_request(ft_tt_op_gdn_request *request)
{
    memset(request, 0, sizeof(*request));
    request->phase = FT_TT_OP_GDN_PHASE_DELTA;
    request->state_size = FT_TT_OP_GDN_QWEN35_STATE;
    request->v_heads = FT_TT_OP_GDN_QWEN35_V_HEADS;
    request->k_heads = FT_TT_OP_GDN_QWEN35_K_HEADS;
    request->conv_dim = FT_TT_OP_GDN_QWEN35_CONV_DIM;
    request->conv_taps = FT_TT_OP_GDN_QWEN35_CONV_TAPS;
    request->activation_in = operand(1u, 2u, 0x00100000u);
    request->activation_out = operand(1u, 2u, 0x00200000u);
    request->gate = operand(3u, 4u, 0x00300000u);
    request->state_recurrent = operand(5u, 6u, 0x00400000u);
}

static void test_gdn_encoding(void)
{
    ft_tt_op_gdn_request request;
    uint32_t base[FT_TT_OP_GDN_WORDS], words[FT_TT_OP_GDN_WORDS];
    unsigned w, moved = 0u;
    int rc;

    ft_tt_test_step("d1: gdn encodes the flag on DELTA only, and moves nothing else");
    fill_delta_request(&request);
    rc = ft_tt_op_gdn_parameters(&request, base);
    ft_tt_test_check_fmt("the DELTA encodes", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("without the flag the bit is clear",
                         (base[FT_TT_OP_GDN_FLAGS] & FT_TT_OP_GDN_FLAG_STATE_ROUND) == 0u,
                         "0x%08X", base[FT_TT_OP_GDN_FLAGS]);
    request.flags = FT_TT_OP_GDN_FLAG_STATE_ROUND;
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("with the flag it encodes", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("word 14 gains exactly the bit",
                         words[FT_TT_OP_GDN_FLAGS] ==
                             (base[FT_TT_OP_GDN_FLAGS] | FT_TT_OP_GDN_FLAG_STATE_ROUND),
                         "0x%08X", words[FT_TT_OP_GDN_FLAGS]);
    for (w = 0; w < FT_TT_OP_GDN_WORDS; w++)
        if (w != FT_TT_OP_GDN_FLAGS && words[w] != base[w])
            moved++;
    ft_tt_test_check_fmt("no other word moved", moved == 0u, "%u", moved);

    memset(&request, 0, sizeof(request));
    request.phase = FT_TT_OP_GDN_PHASE_CONV;
    request.state_size = FT_TT_OP_GDN_QWEN35_STATE;
    request.v_heads = FT_TT_OP_GDN_QWEN35_V_HEADS;
    request.k_heads = FT_TT_OP_GDN_QWEN35_K_HEADS;
    request.conv_dim = FT_TT_OP_GDN_QWEN35_CONV_DIM;
    request.conv_taps = FT_TT_OP_GDN_QWEN35_CONV_TAPS;
    request.activation_in = operand(1u, 2u, 0x00100000u);
    request.activation_out = operand(1u, 2u, 0x00200000u);
    request.conv_weight = operand(3u, 4u, 0x00300000u);
    request.state_conv = operand(5u, 6u, 0x00400000u);
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("the CONV control encodes", rc == FT_TT_OK, "rc=%d", rc);
    request.flags = FT_TT_OP_GDN_FLAG_STATE_ROUND;
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("the flag on CONV is refused", rc != FT_TT_OK, "rc=%d", rc);
    request.flags = FT_TT_OP_GDN_FLAG_DECAY_SWAP;
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("dswap on CONV is refused", rc != FT_TT_OK, "rc=%d", rc);

    fill_delta_request(&request);
    request.flags = FT_TT_OP_GDN_FLAG_STATE_ROUND | FT_TT_OP_GDN_FLAG_DECAY_SWAP;
    rc = ft_tt_op_gdn_parameters(&request, words);
    ft_tt_test_check_fmt("dswap1 (round + swap) encodes both bits and nothing else",
                         rc == FT_TT_OK &&
                             words[FT_TT_OP_GDN_FLAGS] ==
                                 (base[FT_TT_OP_GDN_FLAGS] | FT_TT_OP_GDN_FLAG_STATE_ROUND |
                                  FT_TT_OP_GDN_FLAG_DECAY_SWAP),
                         "0x%08X", words[FT_TT_OP_GDN_FLAGS]);
}

/** The reader's dswap decay: 9 mantissa bits, nearest even. */
static void test_decay_round9(void)
{
    ft_tt_test_step("dswap: the decay is rounded to 9 mantissa bits, nearest even");
#ifdef GDN_STATE_ROUND_RED
#define ROUND9(x) ((x) & 0xFFFFC000u)
#else
#define ROUND9(x) ftg_round_bits9(x)
#endif
    ft_tt_test_check_fmt("below half rounds down", ROUND9(0x3F7F9FFFu) == 0x3F7F8000u, "0x%08X",
                         ROUND9(0x3F7F9FFFu));
    ft_tt_test_check_fmt("above half rounds up", ROUND9(0x3F7FA001u) == 0x3F7FC000u, "0x%08X",
                         ROUND9(0x3F7FA001u));
    ft_tt_test_check_fmt("a tie goes to the even 9-bit mantissa (down)",
                         ROUND9(0x3F7F2000u) == 0x3F7F0000u, "0x%08X", ROUND9(0x3F7F2000u));
    ft_tt_test_check_fmt("a tie goes to the even 9-bit mantissa (up)",
                         ROUND9(0x3F7F6000u) == 0x3F7F8000u, "0x%08X", ROUND9(0x3F7F6000u));
    ft_tt_test_check_fmt("a carry into the exponent (0.99998 -> 1.0)",
                         ROUND9(0x3F7FFFC0u) == 0x3F800000u, "0x%08X", ROUND9(0x3F7FFFC0u));
    ft_tt_test_check_fmt("a 9-bit value is unchanged", ROUND9(0x3F7FC000u) == 0x3F7FC000u,
                         "0x%08X", ROUND9(0x3F7FC000u));
#undef ROUND9
}

static void fill_gdnc_delta(ft_tt_op_gdnc_request *request)
{
    memset(request, 0, sizeof(*request));
    request->phase = FT_TT_OP_GDNC_PHASE_DELTA;
    request->chunk_rows = 16u;
    request->state_tiles = 4u;
    request->v_heads = 32u;
    request->k_heads = 16u;
    request->activation_in = operand(1u, 2u, 0x00100000u);
    request->activation_out = operand(1u, 2u, 0x00200000u);
    request->constant = operand(3u, 4u, 0x00300000u);
    request->state_conv = operand(5u, 6u, 0x00400000u);
    request->gate_address = 0x00380000u;
}

static void test_gdnc_encoding(void)
{
    ft_tt_op_gdnc_request request;
    uint32_t base[FT_TT_OP_GDNC_WORDS], words[FT_TT_OP_GDNC_WORDS];
    unsigned w, moved = 0u;
    int rc;

    ft_tt_test_step("d1: gdnc encodes the flag on DELTA only, and moves nothing else");
    fill_gdnc_delta(&request);
    rc = ft_tt_op_gdnc_parameters(&request, base);
    ft_tt_test_check_fmt("the DELTA encodes", rc == FT_TT_OK, "rc=%d", rc);
    request.delta_flags = FT_TT_OP_GDNC_DELTA_FLAG_STATE_ROUND;
    rc = ft_tt_op_gdnc_parameters(&request, words);
    ft_tt_test_check_fmt("with the flag it encodes", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("the flags word gains exactly the bit",
                         words[FT_TT_OP_GDNC_FLAGS] ==
                             (base[FT_TT_OP_GDNC_FLAGS] | FT_TT_OP_GDNC_DELTA_FLAG_STATE_ROUND),
                         "0x%08X", words[FT_TT_OP_GDNC_FLAGS]);
    for (w = 0; w < FT_TT_OP_GDNC_WORDS; w++)
        if (w != FT_TT_OP_GDNC_FLAGS && words[w] != base[w])
            moved++;
    ft_tt_test_check_fmt("no other word moved", moved == 0u, "%u", moved);

    memset(&request, 0, sizeof(request));
    request.phase = FT_TT_OP_GDNC_PHASE_CONV;
    request.chunk_rows = 32u;
    request.conv_taps = 4u;
    request.conv_dim = 8192u;
    request.state_tiles = 4u;
    request.v_heads = 32u;
    request.k_heads = 16u;
    request.delta_flags = FT_TT_OP_GDNC_DELTA_FLAG_STATE_ROUND;
    ft_tt_test_check_fmt("the flag on CONV is not a supported shape",
                         ft_tt_op_gdnc_shape_is_supported(&request) == 0, "%d",
                         ft_tt_op_gdnc_shape_is_supported(&request));
    request.delta_flags = 0u;
    ft_tt_test_check_fmt("the same CONV without it is",
                         ft_tt_op_gdnc_shape_is_supported(&request) != 0, "%d",
                         ft_tt_op_gdnc_shape_is_supported(&request));
}

/** One rule edge: what the reader sees for `in`. */
static void check_read(const char *what, uint32_t in, uint32_t want)
{
    const uint32_t got = read_bits(in);

    ft_tt_test_check_fmt(what, got == want, "0x%08X -> 0x%08X, want 0x%08X", in, got, want);
}

static void test_rule_edges(void)
{
    ft_tt_test_step("d1: the stored state reads back rounded, at the edges");
    check_read("a tie rounds up (away from zero)", 0x3F801000u, 0x3F802000u);
    check_read("a negative tie rounds away from zero", 0xBF801000u, 0xBF802000u);
    check_read("below the tie rounds down", 0x3F800FFFu, 0x3F800000u);
    check_read("above the tie rounds up", 0x3F801FFFu, 0x3F802000u);
    check_read("a Tf32 value is unchanged", 0x3F802000u, 0x3F802000u);
    check_read("+0 stays +0", 0x00000000u, 0x00000000u);
    check_read("-0 stays -0", 0x80000000u, 0x80000000u);
    check_read("a mantissa carry into the exponent is the right rounding", 0x3FFFF000u,
               0x40000000u);
    check_read("and negative", 0xBFFFF000u, 0xC0000000u);
    check_read("the top Tf32 band below its tie stays finite", 0x7F7FEFFFu, 0x7F7FE000u);
    check_read("the top Tf32 band's tie rounds to inf", 0x7F7FF000u, 0x7F800000u);
    check_read("inf stays inf", 0x7F800000u, 0x7F800000u);
    check_read("-inf stays -inf", 0xFF800000u, 0xFF800000u);
    ft_tt_test_check_fmt("a quiet NaN stays NaN", isnan(bits_float(read_bits(0x7FC00000u))),
                         "0x%08X", read_bits(0x7FC00000u));
    ft_tt_test_check_fmt("a NaN with a low mantissa stays NaN",
                         isnan(bits_float(read_bits(0x7F801000u))), "0x%08X",
                         read_bits(0x7F801000u));
    check_read("the one wrap: a NaN with all ones past bit 12 becomes -0", 0x7FFFF000u,
               0x80000000u);
    ft_tt_test_check_fmt("the bias is undone exactly by subtracting it",
                         stored_bits(0x3F801234u) - (uint32_t)kGdnStateRoundBias == 0x3F801234u,
                         "0x%08X", stored_bits(0x3F801234u));
}

/** Every finite value within the range reads as the nearer Tf32, ties away from zero, and the
 *  stored value is within half a Tf32 ulp of the computed one. */
static void test_rule_sweep(void)
{
    unsigned wrong = 0u, tie_wrong = 0u, stored_far = 0u, checked = 0u;

    ft_tt_test_step("d1: every low-13-bit pattern rounds to the nearer Tf32, ties away");
    for (uint32_t exponent = 1u; exponent < 0xFEu; exponent += 13u) {
        for (uint32_t low = 0u; low < TF32_ULP_BITS; low++) {
            for (uint32_t sign = 0u; sign < 2u; sign++) {
                const uint32_t in = (sign << 31) | (exponent << 23) | (0x2A5u << 13) | low;
                const uint32_t down = in & TF32_KEEP_MASK, up = down + TF32_ULP_BITS;
                const double x = fabs((double)bits_float(in));
                const double lo = fabs((double)bits_float(down));
                const double hi = fabs((double)bits_float(up));
                const uint32_t got = read_bits(in);
                const double stored = fabs((double)bits_float(stored_bits(in)));
                const uint32_t want = (x - lo < hi - x) ? down : up;

                checked++;
                if (x - lo == hi - x && got != up)
                    tie_wrong++;
                else if (got != want)
                    wrong++;
                if (fabs(stored - x) > (hi - lo) * 0.5)
                    stored_far++;
            }
        }
    }
    ft_tt_test_check_fmt("no value rounds the wrong way", wrong == 0u, "%u of %u", wrong,
                         checked);
    ft_tt_test_check_fmt("every tie goes away from zero", tie_wrong == 0u, "%u of %u", tie_wrong,
                         checked);
    ft_tt_test_check_fmt("the stored value is within half a Tf32 ulp", stored_far == 0u,
                         "%u of %u", stored_far, checked);
}

int main(void)
{
    ft_tt_test_step("Gated-Delta-Net state round bias (d1) — off-hardware acceptance");
    test_constants_agree();
    test_gdn_encoding();
    test_gdnc_encoding();
    test_rule_edges();
    test_rule_sweep();
    test_decay_round9();
    return ft_tt_test_finish();
}
