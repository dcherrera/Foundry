/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_convert.c
 * @brief Phase 3 off-device test: the GGUF reader, the Q8_0 dequantizer, the
 *   precision ledger and the streaming artifact writer.
 *
 * NO DEVICE IS TOUCHED and no model file is needed: the test writes its own
 * small GGUF to a temporary path, with tensor shapes chosen to hit every edge
 * the real checkpoint has (a partial tile, a 1-D tensor, an odd vocabulary),
 * converts it, and reads the artifact back.
 *
 * ---------------------------------------------------------------------------
 * THE Q8_0 CLAIM THIS FILE HAS TO PIN
 * ---------------------------------------------------------------------------
 * The whole quality gate rests on Q8_0 dequantizing to fp32 *exactly*
 * (`docs/tt_phase0_decisions.md` §1), so the reference side of every PCC
 * carries no error of its own. That is not an opinion about fp32, it is a
 * claim about int8 x fp16, and it is checked here directly: every one of the
 * 255 x 2048 representable Q8_0 products is reconstructed and compared against
 * the exact rational value.
 *
 * @status Green on macOS/arm64 under
 *   `cc -std=c99 -Wall -Wextra -Werror -pedantic -Wshadow -Wconversion`.
 *   Mutation-proven: see docs/tt_bringup_log.md §12.
 * @issues None
 * @todo None
 */

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_bfp.h"
#include "ft_float_convert.h"
#include "ft_gguf_dequant.h"
#include "ft_gguf_map.h"
#include "ft_gguf_kquant_reference_vectors.h"
#include "ft_tt_precision_ledger.h"
#include "ft_tt_tilize.h"
#include "ft_tt_weight_convert.h"

/* ------------------------------------------------------------------------- */
/* Tiny harness                                                              */
/* ------------------------------------------------------------------------- */

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (condition) {
        printf("  ok    %s\n", what);
    } else {
        checks_failed++;
        printf("  FAIL  %s\n", what);
    }
}

static void check_u64(uint64_t got, uint64_t want, const char *what)
{
    checks_run++;
    if (got == want) {
        printf("  ok    %s = %" PRIu64 "\n", what, got);
    } else {
        checks_failed++;
        printf("  FAIL  %s = %" PRIu64 ", expected %" PRIu64 "\n", what, got, want);
    }
}

static void section(const char *name)
{
    printf("\n== %s ==\n", name);
}

/* ------------------------------------------------------------------------- */
/* float helpers                                                             */
/* ------------------------------------------------------------------------- */

static uint32_t bits_from_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* ------------------------------------------------------------------------- */
/* Q8_0 dequantizes to fp32 exactly                                          */
/* ------------------------------------------------------------------------- */

static void test_q8_0_is_exact_in_fp32(void)
{
    uint64_t checked = 0;
    uint64_t inexact = 0;

    section("Q8_0 dequantizes to fp32 with no rounding at all");

    /*
     * A Q8_0 value is int8 x fp16-scale. Both factors are exact binary
     * rationals, the product needs at most 8 + 11 = 19 significant bits, and
     * fp32 carries 24 — so the product is exact and the reference side of every
     * PCC in this project is exact with it. Checked here over every finite,
     * non-denormal fp16 scale exponent and every int8, using long double as an
     * arbiter that is strictly wider than fp32 on both hosts.
     */
    for (uint32_t scale_exp = 1; scale_exp < 31; scale_exp++) {
        for (uint32_t scale_frac = 0; scale_frac < 1024; scale_frac += 37) {
            uint16_t scale_bits = (uint16_t)((scale_exp << 10) | scale_frac);
            float scale = ft_float_from_fp16(scale_bits);

            for (int quant = -128; quant < 128; quant++) {
                float product = (float)quant * scale;
                long double exact = (long double)quant * (long double)scale;

                checked++;
                if ((long double)product != exact)
                    inexact++;
            }
        }
    }

    check_u64(inexact, 0, "int8 x fp16 products that fp32 could not represent exactly");
    printf("        checked %" PRIu64 " products\n", checked);

    /* And bf16 is NOT exact for them, which is why the ladder starts at fp32. */
    {
        uint64_t bf16_inexact = 0;

        for (int quant = -128; quant < 128; quant++) {
            float scale = ft_float_from_fp16(0x3c01u); /* 1.0009765625 */
            float product = (float)quant * scale;

            if (ft_float_from_bf16(ft_bf16_from_float(product)) != product)
                bf16_inexact++;
        }
        check(bf16_inexact > 0, "bf16 loses Q8_0 products (so it is not a valid reference)");
        printf("        bf16 was wrong for %" PRIu64 " of 256 products at one scale\n", bf16_inexact);
    }
}

/**
 * The Q8_0 block decoder, against a hand-built block with known values.
 *
 * This has to exist separately from the conversion round trip: the round trip
 * dequantizes on BOTH sides of every comparison, so a decoder that reads the
 * scale in the wrong format is perfectly self-consistent and completely
 * invisible to it. The scale below is chosen so that fp16 and bf16 disagree
 * about it by a factor of 128.
 */
static void test_q8_0_block_decode(void)
{
    unsigned char block[FT_GGML_Q8_0_BLOCK_BYTES];
    float decoded[FT_GGML_Q8_0_BLOCK];
    uint16_t scale_bits = 0x3c00u; /* fp16 1.0; the same bits as bf16 are 0.0078125 */
    static const int8_t QUANTS[8] = {1, -1, 127, -128, 0, 64, -64, 3};

    section("the Q8_0 block decoder, on a hand-built block");

    memcpy(block, &scale_bits, sizeof(scale_bits));
    for (int i = 0; i < FT_GGML_Q8_0_BLOCK; i++) {
        int8_t quant = QUANTS[i % 8];

        memcpy(block + 2 + i, &quant, sizeof(quant));
    }

    check(ft_gguf_dequant_to_float(FT_GGML_TYPE_Q8_0, block, FT_GGML_Q8_0_BLOCK, decoded) == 0, "block decodes");
    check_u64(bits_from_float(decoded[0]), bits_from_float(1.0f), "quant 1 x scale 1.0");
    check_u64(bits_from_float(decoded[1]), bits_from_float(-1.0f), "quant -1");
    check_u64(bits_from_float(decoded[2]), bits_from_float(127.0f), "quant 127");
    check_u64(bits_from_float(decoded[3]), bits_from_float(-128.0f), "quant -128");
    check_u64(bits_from_float(decoded[4]), bits_from_float(0.0f), "quant 0");
    check_u64(bits_from_float(decoded[5]), bits_from_float(64.0f), "quant 64");
    check_u64(bits_from_float(decoded[31]), bits_from_float(3.0f), "the last quant of the block");

    check_u64(ft_gguf_type_bytes(FT_GGML_TYPE_Q8_0, 32), FT_GGML_Q8_0_BLOCK_BYTES, "one block is 34 bytes");
    check_u64(ft_gguf_type_bytes(FT_GGML_TYPE_Q8_0, 31), 0, "a partial block has no size");
    check_u64(ft_gguf_type_bytes(FT_GGML_TYPE_F32, 10), 40, "F32 sizing");
    check_u64(ft_gguf_type_bytes(FT_GGML_TYPE_F16, 10), 20, "F16 sizing");
    check_u64(ft_gguf_type_bytes(999, 10), 0, "an unsupported type has no size");
    check(ft_gguf_dequant_to_float(FT_GGML_TYPE_Q8_0, block, 31, decoded) != 0, "a partial block is refused");
}

static void test_fp16_conversion(void)
{
    section("fp16 to fp32");

    check_u64(bits_from_float(ft_float_from_fp16(0x0000u)), 0x00000000u, "+0");
    check_u64(bits_from_float(ft_float_from_fp16(0x8000u)), 0x80000000u, "-0");
    check_u64(bits_from_float(ft_float_from_fp16(0x3c00u)), 0x3f800000u, "1.0");
    check_u64(bits_from_float(ft_float_from_fp16(0xbc00u)), 0xbf800000u, "-1.0");
    check_u64(bits_from_float(ft_float_from_fp16(0x7c00u)), 0x7f800000u, "+inf");
    check_u64(bits_from_float(ft_float_from_fp16(0x0001u)), 0x33800000u, "smallest denormal 2^-24");
    check_u64(bits_from_float(ft_float_from_fp16(0x03ffu)), 0x387fc000u, "largest denormal");
    check_u64(bits_from_float(ft_float_from_fp16(0x7bffu)), 0x477fe000u, "largest finite 65504");

    section("fp32 to bf16 rounds to nearest, ties to even");
    check_u64(ft_bf16_from_float(1.0f), 0x3f80u, "1.0");
    /* 0x3f808000 is exactly halfway between 0x3f80 and 0x3f81; the low bit of
     * 0x3f80 is 0, so it rounds down. 0x3f818000 rounds up to 0x3f82. */
    {
        float halfway_even;
        float halfway_odd;
        uint32_t even_bits = 0x3f808000u;
        uint32_t odd_bits = 0x3f818000u;

        memcpy(&halfway_even, &even_bits, sizeof(float));
        memcpy(&halfway_odd, &odd_bits, sizeof(float));
        check_u64(ft_bf16_from_float(halfway_even), 0x3f80u, "tie with even guard rounds down");
        check_u64(ft_bf16_from_float(halfway_odd), 0x3f82u, "tie with odd guard rounds up");
    }
    {
        float nan_value;
        uint32_t nan_bits = 0x7f800001u; /* NaN whose payload is entirely in the low half */

        memcpy(&nan_value, &nan_bits, sizeof(float));
        check(( ft_bf16_from_float(nan_value) & 0x7fffu) > 0x7f80u, "NaN stays NaN, does not round to infinity");
    }
}

/* ------------------------------------------------------------------------- */
/* Precision ledger                                                          */
/* ------------------------------------------------------------------------- */

static void test_ledger(void)
{
    section("tensor roles from GGUF names");

    check(ft_tt_ledger_role("token_embd.weight") == FT_TT_ROLE_EMBEDDING, "token_embd");
    check(ft_tt_ledger_role("output.weight") == FT_TT_ROLE_LM_HEAD, "output.weight is the head");
    check(ft_tt_ledger_role("output_norm.weight") == FT_TT_ROLE_NORM, "output_norm is a norm, not the head");
    check(ft_tt_ledger_role("blk.7.attn_q.weight") == FT_TT_ROLE_ATTENTION, "attn_q");
    check(ft_tt_ledger_role("blk.7.attn_norm.weight") == FT_TT_ROLE_NORM, "attn_norm is a norm, not attention");
    check(ft_tt_ledger_role("blk.7.attn_q.bias") == FT_TT_ROLE_BIAS, "attn_q.bias is a bias, not attention");
    check(ft_tt_ledger_role("blk.7.ffn_gate.weight") == FT_TT_ROLE_FFN_UP, "ffn_gate");
    check(ft_tt_ledger_role("blk.7.ffn_up.weight") == FT_TT_ROLE_FFN_UP, "ffn_up");
    check(ft_tt_ledger_role("blk.7.ffn_down.weight") == FT_TT_ROLE_FFN_DOWN, "ffn_down");
    check(ft_tt_ledger_role("blk.7.ffn_norm.weight") == FT_TT_ROLE_NORM, "ffn_norm");

    section("the production allocation is spec 6.4's");

    check(ft_tt_ledger_format_for(FT_TT_LEDGER_PRODUCTION, FT_TT_FAMILY_LLAMA, FT_TT_ROLE_FFN_UP, 2) ==
              FT_TT_FORMAT_BFP8_B,
          "gate/up are BFP8 — MEASURED, +9.687% perplexity at BFP4");
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_PRODUCTION, FT_TT_FAMILY_LLAMA, FT_TT_ROLE_FFN_DOWN, 2) ==
              FT_TT_FORMAT_BFP8_B,
          "down is BFP8");
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_PRODUCTION, FT_TT_FAMILY_LLAMA, FT_TT_ROLE_ATTENTION, 2) ==
              FT_TT_FORMAT_BFP8_B,
          "llama attention is BFP8");
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_PRODUCTION, FT_TT_FAMILY_QWEN, FT_TT_ROLE_ATTENTION, 2) ==
              FT_TT_FORMAT_BF16,
          "Qwen attention is bf16 — the spec 6.4 exception");
    /* Demoted from bf16 on a measured perplexity A/B, not on PCC — +0.035% for
     * 383 MiB (bring-up log §13.5). The embedding goes with it: same shape,
     * same score, same argument. */
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_PRODUCTION, FT_TT_FAMILY_LLAMA, FT_TT_ROLE_LM_HEAD, 2) ==
              FT_TT_FORMAT_BFP8_B,
          "the head is BFP8, not bf16");
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_PRODUCTION, FT_TT_FAMILY_LLAMA, FT_TT_ROLE_EMBEDDING, 2) ==
              FT_TT_FORMAT_BFP8_B,
          "the embedding is BFP8 too");
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_PRODUCTION, FT_TT_FAMILY_LLAMA, FT_TT_ROLE_NORM, 1) ==
              FT_TT_FORMAT_FP32,
          "norms stay fp32");
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_PRODUCTION, FT_TT_FAMILY_LLAMA, FT_TT_ROLE_FFN_UP, 1) ==
              FT_TT_FORMAT_FP32,
          "a 1-D tensor stays fp32 whatever its name suggests");
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_FP32, FT_TT_FAMILY_LLAMA, FT_TT_ROLE_FFN_UP, 2) == FT_TT_FORMAT_FP32,
          "the fp32 rung overrides the allocation");
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_BF16, FT_TT_FAMILY_LLAMA, FT_TT_ROLE_FFN_UP, 2) == FT_TT_FORMAT_BF16,
          "the bf16 rung overrides the allocation");
    check(ft_tt_ledger_family_from_architecture("llama") == FT_TT_FAMILY_LLAMA, "llama architecture");
    check(ft_tt_ledger_family_from_architecture("qwen3moe") == FT_TT_FAMILY_QWEN, "qwen architecture");
}

/**
 * A hybrid model's tensors, named exactly as Qwen3.8-27B's GGUF names them.
 *
 * Every one of these fell to ROLE_UNKNOWN before 2026-08-14, which for the 2-D
 * ones meant FT_TT_LAYOUT_SOURCE — a layout nothing would have complained about
 * and every kernel reading N-major would have got wrong. The names are copied
 * off the checkpoint rather than guessed, because the whole class of bug here is
 * a rule that matches a name it was not written for.
 */
static void test_hybrid_roles(void)
{
    section("GDN / SSM tensors are classified, not defaulted");

    check(ft_tt_ledger_role("blk.0.ssm_out.weight") == FT_TT_ROLE_SSM_OUT,
          "ssm_out is a matmul weight with a role");
    check(ft_tt_ledger_role("blk.0.ssm_alpha.weight") == FT_TT_ROLE_SSM_GATE, "ssm_alpha");
    check(ft_tt_ledger_role("blk.0.ssm_beta.weight") == FT_TT_ROLE_SSM_GATE, "ssm_beta");
    check(ft_tt_ledger_role("blk.0.ssm_conv1d.weight") == FT_TT_ROLE_SSM_CONV, "ssm_conv1d");
    check(ft_tt_ledger_role("blk.0.ssm_a") == FT_TT_ROLE_SSM_DECAY, "ssm_a");
    check(ft_tt_ledger_role("blk.0.ssm_dt.bias") == FT_TT_ROLE_BIAS, "ssm_dt.bias is a bias first");
    check(ft_tt_ledger_role("blk.0.ssm_norm.weight") == FT_TT_ROLE_NORM, "ssm_norm is a norm first");
    check(ft_tt_ledger_role("blk.64.nextn.eh_proj.weight") == FT_TT_ROLE_PROJECTION, "nextn.eh_proj");
    /* `.enorm` is not `.norm` and not `_norm`, so these two matched nothing and
     * were only harmless because they are 1-D. */
    check(ft_tt_ledger_role("blk.64.nextn.enorm.weight") == FT_TT_ROLE_NORM, "nextn.enorm");
    check(ft_tt_ledger_role("blk.64.nextn.hnorm.weight") == FT_TT_ROLE_NORM, "nextn.hnorm");

    section("the attn_ PREFIX is not the attention rule");

    check(ft_tt_ledger_role("blk.0.attn_qkv.weight") == FT_TT_ROLE_SSM_IN,
          "attn_qkv is a GDN in-projection, not attention");
    check(ft_tt_ledger_role("blk.0.attn_gate.weight") == FT_TT_ROLE_SSM_IN,
          "attn_gate is a GDN in-projection, not attention");
    check(ft_tt_ledger_role("blk.3.attn_q.weight") == FT_TT_ROLE_ATTENTION, "attn_q still is attention");
    check(ft_tt_ledger_role("blk.3.attn_k.weight") == FT_TT_ROLE_ATTENTION, "attn_k");
    check(ft_tt_ledger_role("blk.3.attn_v.weight") == FT_TT_ROLE_ATTENTION, "attn_v");
    check(ft_tt_ledger_role("blk.3.attn_output.weight") == FT_TT_ROLE_ATTENTION, "attn_output");
    /* The 7.50 GiB: on Qwen3.8 these are 96 tensors, and holding them at bf16
     * for a reason about softmax attention is what made the production rung the
     * worst option that fits the card. */
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_PRODUCTION, FT_TT_FAMILY_QWEN, FT_TT_ROLE_SSM_IN, 2) ==
              FT_TT_FORMAT_BFP8_B,
          "the Qwen bf16 exception does NOT reach a GDN in-projection");
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_PRODUCTION, FT_TT_FAMILY_QWEN, FT_TT_ROLE_ATTENTION, 2) ==
              FT_TT_FORMAT_BF16,
          "and still does reach real attention");

    section("a MoE router is not an up-projection");

    check(ft_tt_ledger_role("blk.0.ffn_gate_inp.weight") == FT_TT_ROLE_ROUTER,
          "ffn_gate_inp is the router, though ffn_gate is a prefix of it");
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_PRODUCTION, FT_TT_FAMILY_QWEN, FT_TT_ROLE_ROUTER, 2) ==
              FT_TT_FORMAT_BF16,
          "the router stays bf16 — its output is an argmax over experts");

    section("every matmul role is N-major, and only the matmul roles");

    {
        static const ft_tt_tensor_role NMAJOR[] = {FT_TT_ROLE_ATTENTION, FT_TT_ROLE_FFN_UP,
                                                   FT_TT_ROLE_FFN_DOWN,  FT_TT_ROLE_LM_HEAD,
                                                   FT_TT_ROLE_SSM_IN,    FT_TT_ROLE_SSM_OUT,
                                                   FT_TT_ROLE_SSM_GATE,  FT_TT_ROLE_PROJECTION,
                                                   FT_TT_ROLE_ROUTER};
        static const ft_tt_tensor_role SOURCE[] = {FT_TT_ROLE_EMBEDDING, FT_TT_ROLE_SSM_CONV};
        ft_tt_tensor_layout layout;

        for (size_t i = 0; i < sizeof(NMAJOR) / sizeof(NMAJOR[0]); i++)
            check(ft_tt_convert_layout_for_checked(NMAJOR[i], 2, &layout) == FT_TT_CONVERT_OK &&
                      layout == FT_TT_LAYOUT_MATMUL_NMAJOR,
                  ft_tt_ledger_role_name(NMAJOR[i]));
        for (size_t i = 0; i < sizeof(SOURCE) / sizeof(SOURCE[0]); i++)
            check(ft_tt_convert_layout_for_checked(SOURCE[i], 2, &layout) == FT_TT_CONVERT_OK &&
                      layout == FT_TT_LAYOUT_SOURCE,
                  ft_tt_ledger_role_name(SOURCE[i]));
    }
}

/**
 * The two uniform rungs, and the caution that has to travel with one of them.
 */
static void test_uniform_rungs(void)
{
    section("the uniform rungs exist and are selectable by name");

    check(ft_tt_ledger_mode_from_name("bfp8") == FT_TT_LEDGER_BFP8, "--ledger bfp8");
    check(ft_tt_ledger_mode_from_name("bfp4") == FT_TT_LEDGER_BFP4, "--ledger bfp4");
    check(ft_tt_ledger_mode_from_name("nonsense") < 0, "an unknown rung is refused");
    check(strcmp(ft_tt_ledger_mode_name(FT_TT_LEDGER_BFP8), "bfp8") == 0, "the name round-trips");
    check(strcmp(ft_tt_ledger_mode_name(FT_TT_LEDGER_PRODUCTION), "production") == 0,
          "and so does production");

    /* Uniform means uniform: no role escapes it, INCLUDING the ones the
     * production rung holds at bf16. A rung with exceptions is not a rung. */
    {
        static const ft_tt_tensor_role EVERY[] = {
            FT_TT_ROLE_EMBEDDING, FT_TT_ROLE_LM_HEAD,  FT_TT_ROLE_ATTENTION, FT_TT_ROLE_FFN_UP,
            FT_TT_ROLE_FFN_DOWN,  FT_TT_ROLE_SSM_IN,   FT_TT_ROLE_SSM_OUT,   FT_TT_ROLE_SSM_GATE,
            FT_TT_ROLE_SSM_CONV,  FT_TT_ROLE_ROUTER,   FT_TT_ROLE_PROJECTION};
        int all_bfp8 = 1;
        int all_bfp4 = 1;

        for (size_t i = 0; i < sizeof(EVERY) / sizeof(EVERY[0]); i++) {
            if (ft_tt_ledger_format_for(FT_TT_LEDGER_BFP8, FT_TT_FAMILY_QWEN, EVERY[i], 2) !=
                FT_TT_FORMAT_BFP8_B)
                all_bfp8 = 0;
            if (ft_tt_ledger_format_for(FT_TT_LEDGER_BFP4, FT_TT_FAMILY_QWEN, EVERY[i], 2) !=
                FT_TT_FORMAT_BFP4_B)
                all_bfp4 = 0;
        }
        check(all_bfp8, "the bfp8 rung takes every 2-D role, Qwen attention included");
        check(all_bfp4, "the bfp4 rung takes every 2-D role");
    }
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_BFP4, FT_TT_FAMILY_LLAMA, FT_TT_ROLE_NORM, 1) ==
              FT_TT_FORMAT_FP32,
          "1-D norms stay fp32 even on the bfp4 rung");

    section("the BFP4 rung carries its measured cost with it");

    check(ft_tt_ledger_mode_caution(FT_TT_LEDGER_BFP4) != NULL,
          "selecting bfp4 produces a caution a CLI must print");
    check(ft_tt_ledger_mode_caution(FT_TT_LEDGER_BFP4) != NULL &&
              strstr(ft_tt_ledger_mode_caution(FT_TT_LEDGER_BFP4), "40.216") != NULL,
          "and the caution names the measured number, not an adjective");
    check(ft_tt_ledger_mode_caution(FT_TT_LEDGER_BFP8) == NULL, "bfp8 has nothing to warn about");
    check(ft_tt_ledger_mode_caution(FT_TT_LEDGER_PRODUCTION) == NULL, "nor does production");
}

/* ------------------------------------------------------------------------- */
/* A synthetic GGUF                                                          */
/* ------------------------------------------------------------------------- */

typedef struct {
    const char *name;
    uint32_t n_dims;
    uint64_t dims[2];
    uint32_t ggml_type;
} synthetic_tensor;

/* Shapes deliberately awkward: a partial tile, a 1-D norm, an odd row count. */
static const synthetic_tensor SYNTHETIC[] = {
    {"token_embd.weight", 2, {64, 100}, FT_GGML_TYPE_Q8_0},
    {"blk.0.attn_norm.weight", 1, {64, 1}, FT_GGML_TYPE_F32},
    {"blk.0.attn_q.weight", 2, {64, 64}, FT_GGML_TYPE_Q8_0},
    {"blk.0.ffn_gate.weight", 2, {64, 33}, FT_GGML_TYPE_Q8_0},
    {"blk.0.ffn_down.weight", 2, {96, 64}, FT_GGML_TYPE_Q8_0},
    /* 33 fp32 values = 132 bytes, which is NOT a multiple of the artifact's
     * 64-byte alignment. Without this the alignment padding is never exercised,
     * because every tiled format's tile size already divides by 64. */
    {"blk.0.ffn_norm.weight", 1, {33, 1}, FT_GGML_TYPE_F32},
    {"output.weight", 2, {64, 100}, FT_GGML_TYPE_F16},
    /* A MoE router. Since the embedding and head were demoted to BFP8 on a
     * perplexity measurement, this is the only bf16 tensor a production-mode
     * artifact still contains — and bf16 is the one format that IS stored
     * face-ordered, so without it the face-order check has no subject. */
    {"blk.0.ffn_gate_inp.weight", 2, {64, 64}, FT_GGML_TYPE_F16},
    /* A BF16 source tensor. It is last so the indices every other check uses
     * are undisturbed. It has to be here rather than only in a dedicated test:
     * BF16 is what the 27B target actually ships as, so the whole pipeline —
     * strip loader, tilizer, packer, round trip — must meet it on the ordinary
     * path, not just the reader. */
    {"blk.0.attn_k.weight", 2, {64, 64}, FT_GGML_TYPE_BF16},
};

#define SYNTHETIC_COUNT ((int)(sizeof(SYNTHETIC) / sizeof(SYNTHETIC[0])))

static void put_u32(FILE *f, uint32_t value)
{
    fwrite(&value, sizeof(value), 1, f);
}

static void put_u64(FILE *f, uint64_t value)
{
    fwrite(&value, sizeof(value), 1, f);
}

static void put_string(FILE *f, const char *text)
{
    uint64_t length = strlen(text);

    put_u64(f, length);
    fwrite(text, 1, (size_t)length, f);
}

static float synthetic_value(uint32_t tensor, uint64_t index)
{
    /* Wide dynamic range on purpose, so block-float has real work to do. */
    uint32_t state = (tensor * 2654435761u) ^ (uint32_t)(index * 40503u);

    state = state * 1664525u + 1013904223u;
    return (float)((int32_t)(state >> 8) - 8388608) * (1.0f / 8388608.0f) *
           (1.0f + (float)(index % 17u) * 3.0f);
}

/** Quantize a row to Q8_0 the way llama.cpp does: scale = max|x| / 127. */
static void write_q8_0_row(FILE *f, const float *row, uint64_t count)
{
    for (uint64_t block = 0; block < count / FT_GGML_Q8_0_BLOCK; block++) {
        const float *values = row + block * FT_GGML_Q8_0_BLOCK;
        float peak = 0.0f;
        float scale;

        for (int i = 0; i < FT_GGML_Q8_0_BLOCK; i++) {
            float magnitude = values[i] < 0 ? -values[i] : values[i];

            if (magnitude > peak)
                peak = magnitude;
        }
        scale = peak / 127.0f;
        {
            /* Narrow the scale to fp16, because fp16 is what Q8_0 stores.
             * Truncating rather than rounding is fine: the test needs a valid
             * checkpoint, not an optimal one. */
            uint32_t bits = bits_from_float(scale);
            uint32_t exp = (bits >> 23) & 0xffu;
            uint16_t half;

            if (scale == 0.0f || exp < 113u || exp > 142u) {
                half = 0x3c00u; /* fall back to 1.0 for degenerate blocks */
            } else {
                half = (uint16_t)(((bits >> 16) & 0x8000u) | ((exp - 127u + 15u) << 10) | ((bits >> 13) & 0x3ffu));
            }
            fwrite(&half, sizeof(half), 1, f);
            scale = ft_float_from_fp16(half);
        }
        for (int i = 0; i < FT_GGML_Q8_0_BLOCK; i++) {
            float scaled = scale != 0.0f ? values[i] / scale : 0.0f;
            int rounded = (int)(scaled < 0 ? scaled - 0.5f : scaled + 0.5f);
            int8_t quant;

            if (rounded > 127)
                rounded = 127;
            if (rounded < -128)
                rounded = -128;
            quant = (int8_t)rounded;
            fwrite(&quant, sizeof(quant), 1, f);
        }
    }
}

/**
 * What a part file should declare about the split it belongs to.
 *
 * `part_count == 0` means "not a split file": no split.* keys at all, which is
 * what an unsplit checkpoint looks like and what every test written before
 * splits existed still exercises.
 */
typedef struct {
    uint32_t part_count;
    uint32_t part_index;     /**< 0-based, as split.no is */
    uint64_t declared_total; /**< what to write as split.tensors.count */
} synthetic_split;

/** Write one tensor's payload, in whichever ggml type the entry declares. */
static void write_synthetic_tensor_data(FILE *f, const synthetic_tensor *t, uint32_t seed, float *row)
{
    uint64_t rows = t->n_dims > 1 ? t->dims[1] : 1;

    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t c = 0; c < t->dims[0]; c++)
            row[c] = synthetic_value(seed, r * t->dims[0] + c);

        if (t->ggml_type == FT_GGML_TYPE_Q8_0) {
            write_q8_0_row(f, row, t->dims[0]);
        } else if (t->ggml_type == FT_GGML_TYPE_F16) {
            for (uint64_t c = 0; c < t->dims[0]; c++) {
                uint32_t bits = bits_from_float(row[c]);
                uint32_t exp = (bits >> 23) & 0xffu;
                uint16_t half = (uint16_t)(((bits >> 16) & 0x8000u) |
                                           ((exp >= 113u && exp <= 142u) ? ((exp - 112u) << 10) : 0x3800u) |
                                           ((bits >> 13) & 0x3ffu));

                fwrite(&half, sizeof(half), 1, f);
            }
        } else if (t->ggml_type == FT_GGML_TYPE_BF16) {
            for (uint64_t c = 0; c < t->dims[0]; c++) {
                uint16_t narrow = ft_bf16_from_float(row[c]);

                fwrite(&narrow, sizeof(narrow), 1, f);
            }
        } else {
            fwrite(row, sizeof(float), (size_t)t->dims[0], f);
        }
    }
}

/** The whole-model metadata, which real `gguf-split` puts in part 0 only. */
static void write_synthetic_model_metadata(FILE *f)
{
    put_string(f, "general.architecture");
    put_u32(f, 8); /* string */
    put_string(f, "llama");

    /*
     * Two arrays, because a real checkpoint's tokenizer is an array and a
     * reader that skips one by the wrong number of bytes does not fail — it
     * silently starts reading the tensor directory from the middle of the
     * array. Both element kinds are covered: fixed-width and string.
     */
    put_string(f, "tokenizer.ggml.token_type");
    put_u32(f, 9); /* array */
    put_u32(f, 5); /* of int32 */
    put_u64(f, 37);
    for (int i = 0; i < 37; i++)
        put_u32(f, (uint32_t)i);

    put_string(f, "tokenizer.ggml.tokens");
    put_u32(f, 9); /* array */
    put_u32(f, 8); /* of string */
    put_u64(f, 5);
    put_string(f, "<s>");
    put_string(f, "</s>");
    put_string(f, "a");
    put_string(f, "");
    put_string(f, "a rather longer token than the others");

    put_string(f, "general.alignment");
    put_u32(f, 4); /* uint32 */
    put_u32(f, 32);
}

/** `split.no`, `split.count`, `split.tensors.count` — in every part. */
static void write_synthetic_split_metadata(FILE *f, const synthetic_split *split)
{
    put_string(f, "split.no");
    put_u32(f, 2); /* uint16, which is what gguf-split writes */
    {
        uint16_t value = (uint16_t)split->part_index;

        fwrite(&value, sizeof(value), 1, f);
    }

    put_string(f, "split.count");
    put_u32(f, 2); /* uint16 */
    {
        uint16_t value = (uint16_t)split->part_count;

        fwrite(&value, sizeof(value), 1, f);
    }

    put_string(f, "split.tensors.count");
    put_u32(f, 5); /* int32 */
    put_u32(f, (uint32_t)split->declared_total);
}

/** Extra metadata keys a caller wants in part 0, and how many there are. */
typedef struct {
    void (*write)(FILE *f);
    uint32_t key_count;
} synthetic_extra_metadata;

/**
 * Write `count` tensors from an arbitrary table into one GGUF file.
 *
 * `seed_base` is added to the table index to seed each tensor's values, so a
 * tensor's payload depends only on its GLOBAL index and never on which part it
 * landed in — which is what lets the split tests compare a split checkpoint and
 * the unsplit one byte for byte instead of asserting something weaker about
 * shapes and counts.
 */
static int write_gguf(const char *path, const synthetic_tensor *tensors, int count, int seed_base,
                      const synthetic_split *split, const synthetic_extra_metadata *extra)
{
    FILE *f = fopen(path, "wb");
    long header_end;
    uint64_t offset = 0;
    uint64_t key_count = split->part_count > 0 ? 3u : 0u;
    int carries_model_metadata = split->part_count == 0 || split->part_index == 0;

    if (!f)
        return -1;

    /* Part 0 carries the model's metadata; later parts carry only split.*,
     * exactly as gguf-split writes them. A reader that looked for
     * general.architecture in the wrong part would find nothing. */
    if (carries_model_metadata)
        key_count += 4u + (extra ? extra->key_count : 0u);

    put_u32(f, 0x46554747u); /* "GGUF" */
    put_u32(f, 3);
    put_u64(f, (uint64_t)count);
    put_u64(f, key_count);

    if (carries_model_metadata) {
        write_synthetic_model_metadata(f);
        if (extra)
            extra->write(f);
    }
    if (split->part_count > 0)
        write_synthetic_split_metadata(f, split);

    for (int i = 0; i < count; i++) {
        const synthetic_tensor *t = &tensors[i];
        uint64_t elements = t->dims[0] * (t->n_dims > 1 ? t->dims[1] : 1);

        put_string(f, t->name);
        put_u32(f, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++)
            put_u64(f, t->dims[d]);
        put_u32(f, t->ggml_type);
        put_u64(f, offset);
        offset += ft_gguf_type_bytes(t->ggml_type, elements);
        offset = (offset + 31u) / 32u * 32u;
    }

    header_end = ftell(f);
    {
        long padded = (header_end + 31) / 32 * 32;

        for (long i = header_end; i < padded; i++)
            fputc(0, f);
    }

    {
        float *row = malloc(4096 * sizeof(float));
        long data_base = ftell(f);

        if (!row) {
            fclose(f);
            return -1;
        }
        offset = 0;
        for (int i = 0; i < count; i++) {
            const synthetic_tensor *t = &tensors[i];
            uint64_t elements = t->dims[0] * (t->n_dims > 1 ? t->dims[1] : 1);

            fseek(f, data_base + (long)offset, SEEK_SET);
            write_synthetic_tensor_data(f, t, (uint32_t)(seed_base + i), row);
            offset += ft_gguf_type_bytes(t->ggml_type, elements);
            offset = (offset + 31u) / 32u * 32u;
        }
        fseek(f, data_base + (long)offset, SEEK_SET);
        fputc(0, f); /* make sure the file covers the last tensor's padding */
        free(row);
    }

    fclose(f);
    return 0;
}

/** Tensors [first, first+count) of the SYNTHETIC table, into one part file. */
static int write_synthetic_gguf_range(const char *path, int first, int count, const synthetic_split *split)
{
    return write_gguf(path, SYNTHETIC + first, count, first, split, NULL);
}

static int write_synthetic_gguf(const char *path)
{
    static const synthetic_split unsplit = {0, 0, 0};

    return write_synthetic_gguf_range(path, 0, SYNTHETIC_COUNT, &unsplit);
}

/* ------------------------------------------------------------------------- */
/* Reader and converter                                                      */
/* ------------------------------------------------------------------------- */

typedef struct {
    uint32_t tensors;
    double worst_pcc;
    uint64_t converted_bytes;
    ft_tt_tensor_format formats[SYNTHETIC_COUNT];
} convert_observer;

static int observe(const ft_tt_convert_tensor_report *report, void *user)
{
    convert_observer *observer = (convert_observer *)user;

    if (observer->tensors < SYNTHETIC_COUNT)
        observer->formats[observer->tensors] = report->format;
    if (report->pcc < observer->worst_pcc)
        observer->worst_pcc = report->pcc;
    observer->converted_bytes += report->converted_bytes;
    observer->tensors++;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Where an element actually lands                                           */
/* ------------------------------------------------------------------------- */

/** Read one directory entry out of a written artifact, by name. */
static int read_artifact_entry(const char *artifact_path, const char *tensor_name, ft_tt_artifact_entry *out)
{
    FILE *f = fopen(artifact_path, "rb");
    ft_tt_artifact_header header;
    int found = 0;

    if (!f)
        return 0;
    if (fread(&header, sizeof(header), 1, f) == 1 && fseek(f, (long)header.directory_offset, SEEK_SET) == 0) {
        for (uint32_t i = 0; i < header.tensor_count && !found; i++) {
            if (fread(out, sizeof(*out), 1, f) != 1)
                break;
            if (strcmp(out->name, tensor_name) == 0)
                found = 1;
        }
    }
    fclose(f);
    return found;
}

/** Pearson correlation of two float runs, for the layout discrimination below. */
static double correlate(const float *a, const float *b, size_t count)
{
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    double n = (double)count;
    double var_a, var_b;

    for (size_t i = 0; i < count; i++) {
        sa += a[i];
        sb += b[i];
        saa += (double)a[i] * a[i];
        sbb += (double)b[i] * b[i];
        sab += (double)a[i] * b[i];
    }
    var_a = n * saa - sa * sa;
    var_b = n * sbb - sb * sb;
    if (var_a <= 0 || var_b <= 0)
        return 0.0;
    return (n * sab - sa * sb) / sqrt(var_a * var_b);
}

/**
 * The one thing a round-trip test cannot catch: whether the converter applied
 * the face reordering to the block-float path, which tt-metal's packer already
 * does internally.
 *
 * Encode and decode would still agree with each other after that mistake — the
 * PCC the converter reports would be unchanged — but the bytes on the card
 * would have every element outside face (0,0) in the wrong place. So this reads
 * the artifact back and asks where element (row, col) of the SOURCE ended up,
 * and checks that the right answer scores clearly better than the face-permuted
 * one. If face order had been applied twice, the two scores would swap.
 */
static void test_artifact_layout(const char *gguf_path, const char *artifact_path)
{
    ft_gguf_map *map = NULL;
    ft_tt_artifact_entry entry;
    static float source[FT_TT_TILE_VALUES];
    static float straight[FT_TT_TILE_VALUES];
    static float permuted[FT_TT_TILE_VALUES];
    static float decoded[FT_TT_TILE_VALUES];
    static unsigned char raw[4096];
    FILE *f;

    section("elements land where the card will look for them");

    if (ft_gguf_map_open(gguf_path, &map) != FT_GGUF_OK) {
        check(0, "reopened the synthetic GGUF");
        return;
    }

    /* blk.0.ffn_gate.weight is BFP8 under the production ledger AND a matmul weight, so
     * it takes FT_TT_LAYOUT_MATMUL_NMAJOR: 33 rows (out) of 64 columns (in) becomes
     * [64 in x 33 out], a Kt=2 by Nt=2 grid stored COLUMN-major.
     *
     * Two independent things are checked and they fail differently. The TRANSPOSE shows
     * up inside tile 0 — the source's first tile with its two axes exchanged. The TILE
     * ORDER shows up in which tile comes second: N-major makes it (nt=0, kt=1), i.e.
     * in-features 32..63, fully populated, where K-major would make it (kt=0, nt=1),
     * i.e. out-feature 32 alone, which is 31/32 zero padding. Correlating tile 1 against
     * the wrong one of those is not a near miss. */
    {
        const ft_gguf_tensor *tensor = ft_gguf_map_find(map, "blk.0.ffn_gate.weight");
        static float row[64];
        static float full[FT_TT_TILE_HEIGHT * 64];
        static float window[FT_TT_TILE_VALUES];

        check(tensor != NULL && read_artifact_entry(artifact_path, "blk.0.ffn_gate.weight", &entry),
              "found ffn_gate in both the GGUF and the artifact");
        if (!tensor) {
            ft_gguf_map_close(map);
            return;
        }
        check_u64(entry.format, FT_TT_FORMAT_BFP8_B, "ffn_gate is stored as BFP8");
        check_u64(entry.layout, FT_TT_LAYOUT_MATMUL_NMAJOR,
                  "a matmul weight is stored [in x out], N-major");
        /* ffn_gate is 33x64, whose two tile counts are both 2, so its grid cannot tell a
         * swap from an identity. ffn_down is 64 out x 96 in: Kt=3, Nt=2, and token_embd
         * is 100 out x 64 in and NOT transposed, so 4 x 2 the other way round. Those two
         * together are what make the field non-vacuous. */
        {
            ft_tt_artifact_entry other;

            if (read_artifact_entry(artifact_path, "blk.0.ffn_down.weight", &other)) {
                check_u64(other.tile_rows, 3, "ffn_down's stored grid is Kt = ceil(96/32) rows");
                check_u64(other.tile_cols, 2, "by Nt = ceil(64/32) columns — the SWAP of GGUF's");
            }
            if (read_artifact_entry(artifact_path, "token_embd.weight", &other)) {
                check_u64(other.tile_rows, 4,
                          "token_embd is untransposed, so its grid stays ceil(100/32) rows");
                check_u64(other.tile_cols, 2, "by ceil(64/32) columns");
            }
        }

        for (uint32_t r = 0; r < FT_TT_TILE_HEIGHT; r++) {
            const unsigned char *src = (const unsigned char *)tensor->data + (size_t)r * (64u / 32u * 34u);

            ft_gguf_dequant_to_float(tensor->ggml_type, src, 64, row);
            memcpy(&full[r * 64u], row, 64u * sizeof(float));
        }

        f = fopen(artifact_path, "rb");
        check(f && fseek(f, (long)entry.data_offset, SEEK_SET) == 0 &&
                  fread(raw, 1, ft_bfp_tile_bytes(FT_BFP_BFP8_B), f) == ft_bfp_tile_bytes(FT_BFP_BFP8_B),
              "read the first BFP8 tile out of the artifact");
        if (f)
            fclose(f);

        check(ft_bfp_unpack_tiles(FT_BFP_BFP8_B, raw, 1, decoded) == FT_BFP_OK, "unpacked it");

        /* `straight` reads the stored tile as [in x out]; `permuted` reads the same bytes
         * untransposed, which is what the artifact held before 2026-08-12. */
        for (uint32_t r = 0; r < FT_TT_TILE_HEIGHT; r++) {
            for (uint32_t c = 0; c < FT_TT_TILE_WIDTH; c++) {
                straight[r * FT_TT_TILE_WIDTH + c] = decoded[c * FT_TT_TILE_WIDTH + r];
                permuted[r * FT_TT_TILE_WIDTH + c] = decoded[r * FT_TT_TILE_WIDTH + c];
            }
        }
        for (uint32_t r = 0; r < FT_TT_TILE_HEIGHT; r++)
            memcpy(&window[r * FT_TT_TILE_WIDTH], &full[r * 64u],
                   FT_TT_TILE_WIDTH * sizeof(float));
        memcpy(source, window, sizeof(window));

        {
            double correct = correlate(window, straight, FT_TT_TILE_VALUES);
            double wrong = correlate(window, permuted, FT_TT_TILE_VALUES);

            printf("        BFP8 tile 0: transposed %.6f, untransposed %.6f\n", correct, wrong);
            check(correct > 0.99, "tile 0 is the source's first tile, TRANSPOSED");
            check(correct - wrong > 0.5, "and reading it untransposed is clearly worse");
        }

        f = fopen(artifact_path, "rb");
        check(f && fseek(f, (long)(entry.data_offset + (uint64_t)ft_bfp_tile_bytes(FT_BFP_BFP8_B)),
                         SEEK_SET) == 0 &&
                  fread(raw, 1, ft_bfp_tile_bytes(FT_BFP_BFP8_B), f) ==
                      ft_bfp_tile_bytes(FT_BFP_BFP8_B),
              "read the SECOND BFP8 tile out of the artifact");
        if (f)
            fclose(f);
        check(ft_bfp_unpack_tiles(FT_BFP_BFP8_B, raw, 1, decoded) == FT_BFP_OK,
              "unpacked the second tile");

        {
            double as_n_major;
            uint32_t zeros = 0;

            /* (nt=0, kt=1) is source columns 32..63 against source rows 0..31, exchanged. */
            for (uint32_t r = 0; r < FT_TT_TILE_HEIGHT; r++)
                for (uint32_t c = 0; c < FT_TT_TILE_WIDTH; c++)
                    window[r * FT_TT_TILE_WIDTH + c] = full[c * 64u + 32u + r];
            for (uint32_t i = 0; i < FT_TT_TILE_VALUES; i++) {
                straight[i] = decoded[i];
                if (decoded[i] == 0.0f)
                    zeros++;
            }
            as_n_major = correlate(window, straight, FT_TT_TILE_VALUES);

            printf("        BFP8 tile 1: as N-major %.6f, %u/%u zero\n", as_n_major, zeros,
                   (unsigned)FT_TT_TILE_VALUES);
            check(as_n_major > 0.99, "tile 1 is (nt=0, kt=1) — the tile grid is COLUMN-major");
            check(zeros * 2u < (uint32_t)FT_TT_TILE_VALUES,
                  "and it is populated, not the 31/32-padded tile K-major would put here");
        }
    }

    /* The MoE router is the one tensor a production artifact still stores as
     * bf16: its output is consumed as an ARGMAX over experts, so a block-float
     * rounding that reorders two near-tied logits routes the token elsewhere.
     * That makes it the only subject available for the face-order check — bf16
     * tiles ARE face-ordered, the opposite convention to block float above. */
    {
        const ft_gguf_tensor *tensor = ft_gguf_map_find(map, "blk.0.ffn_gate_inp.weight");
        static float row[64];
        static float source_transposed[FT_TT_TILE_VALUES];
        static uint16_t tile[FT_TT_TILE_VALUES];

        check(tensor != NULL && read_artifact_entry(artifact_path, "blk.0.ffn_gate_inp.weight", &entry),
              "found the router in both the checkpoint and the artifact");
        if (!tensor) {
            ft_gguf_map_close(map);
            return;
        }
        check_u64(entry.format, FT_TT_FORMAT_BF16, "the router is bf16, not block float");

        check_u64(entry.layout, FT_TT_LAYOUT_MATMUL_NMAJOR,
                  "and it is a matmul weight, so it is transposed like one");

        /* And the two that were demoted really are BFP8 in the written
         * artifact, not merely in the ledger function. */
        if (read_artifact_entry(artifact_path, "output.weight", &entry)) {
            check_u64(entry.format, FT_TT_FORMAT_BFP8_B, "output.weight is stored as BFP8");
            check_u64(entry.layout, FT_TT_LAYOUT_MATMUL_NMAJOR,
                      "output.weight IS a matmul weight, so it is transposed");
            if (read_artifact_entry(artifact_path, "token_embd.weight", &entry)) {
                check_u64(entry.format, FT_TT_FORMAT_BFP8_B, "token_embd.weight is stored as BFP8");
                /* The one that shares the head's shape and must NOT move: an embedding
                 * table is indexed BY ROW, so transposing it turns a lookup into a
                 * gather across the whole tensor. */
                check_u64(entry.layout, FT_TT_LAYOUT_SOURCE,
                          "token_embd.weight is indexed by row and stays in source order");
            }
            (void)read_artifact_entry(artifact_path, "blk.0.ffn_gate_inp.weight", &entry);
        }

        for (uint32_t r = 0; r < FT_TT_TILE_HEIGHT; r++) {
            const unsigned char *src = (const unsigned char *)tensor->data + (size_t)r * 64u * 2u;

            ft_gguf_dequant_to_float(tensor->ggml_type, src, 64, row);
            memcpy(&source[r * FT_TT_TILE_WIDTH], row, FT_TT_TILE_WIDTH * sizeof(float));
        }
        /* Stored tile 0 is (kt=0, nt=0), i.e. the source's first 32x32 block with
         * its axes exchanged. Comparing against the block UNexchanged would make
         * a face-order failure and a transpose failure indistinguishable. */
        for (uint32_t r = 0; r < FT_TT_TILE_HEIGHT; r++)
            for (uint32_t c = 0; c < FT_TT_TILE_WIDTH; c++)
                source_transposed[r * FT_TT_TILE_WIDTH + c] = source[c * FT_TT_TILE_WIDTH + r];

        f = fopen(artifact_path, "rb");
        check(f && fseek(f, (long)entry.data_offset, SEEK_SET) == 0 &&
                  fread(tile, sizeof(uint16_t), FT_TT_TILE_VALUES, f) == FT_TT_TILE_VALUES,
              "read the first bf16 tile out of the artifact");
        if (f)
            fclose(f);

        for (uint32_t r = 0; r < FT_TT_TILE_HEIGHT; r++) {
            for (uint32_t c = 0; c < FT_TT_TILE_WIDTH; c++) {
                straight[r * FT_TT_TILE_WIDTH + c] = ft_float_from_bf16(tile[r * FT_TT_TILE_WIDTH + c]);
                permuted[r * FT_TT_TILE_WIDTH + c] = ft_float_from_bf16(tile[ft_tt_tile_face_index(r, c)]);
            }
        }

        {
            double faced = correlate(source_transposed, permuted, FT_TT_TILE_VALUES);
            double flat = correlate(source_transposed, straight, FT_TT_TILE_VALUES);

            printf("        bf16 tile: face-ordered %.6f, flat %.6f\n", faced, flat);
            check(faced > 0.999, "the bf16 tile IS face-ordered");
            check(faced - flat > 0.5, "and reading it flat is clearly worse");
        }
    }

    ft_gguf_map_close(map);
}

/**
 * The transpose on the FACE-ORDERED path, which the production artifact never exercises.
 *
 * fp32 and bf16 tiles are stored face-ordered — four 16x16 faces, (0,0) (0,1) (1,0) (1,1)
 * — while block float is stored swizzled, so the two take DIFFERENT transposes. A
 * face-ordered tile's transpose is each face transposed AND the two off-diagonal faces
 * exchanged, and doing only the first half is a bijection applied to both sides of any
 * round trip: PCC would not see it (dev notes §3). The only way to catch it is to read
 * the written artifact back and index it the way the card will.
 *
 * The BF16 rung is where it happens, because at the production rung every matmul weight
 * is block float. `blk.0.attn_q.weight` is 64x64, so its stored grid is Kt=2 by Nt=2 and
 * tile 0 is the source's first tile with its axes exchanged.
 */
static void test_face_ordered_transpose(const char *gguf_path)
{
    const char *artifact_path = "build/test_bf16_rung.fttw";
    ft_tt_convert_options options;
    ft_gguf_map *map = NULL;
    ft_tt_artifact_entry entry;
    static float full[FT_TT_TILE_HEIGHT * 64];
    static float window[FT_TT_TILE_VALUES];
    static float faced[FT_TT_TILE_VALUES];
    static float faced_untransposed[FT_TT_TILE_VALUES];
    static float flat[FT_TT_TILE_VALUES];
    static uint16_t tile[FT_TT_TILE_VALUES];
    static float row[64];
    const ft_gguf_tensor *tensor;
    FILE *f;

    section("the face-ordered transpose, on the bf16 rung");

    memset(&options, 0, sizeof(options));
    options.mode = FT_TT_LEDGER_BF16;
    check(ft_tt_convert_gguf(gguf_path, artifact_path, &options) == FT_TT_CONVERT_OK,
          "converted the synthetic checkpoint at the bf16 rung");
    if (ft_gguf_map_open(gguf_path, &map) != FT_GGUF_OK) {
        check(0, "reopened the synthetic GGUF");
        return;
    }
    tensor = ft_gguf_map_find(map, "blk.0.attn_q.weight");
    check(tensor != NULL && read_artifact_entry(artifact_path, "blk.0.attn_q.weight", &entry),
          "found attn_q in both");
    if (!tensor) {
        ft_gguf_map_close(map);
        return;
    }
    check_u64(entry.format, FT_TT_FORMAT_BF16, "attn_q is bf16 on this rung");
    check_u64(entry.layout, FT_TT_LAYOUT_MATMUL_NMAJOR, "and it is still a matmul weight");

    for (uint32_t r = 0; r < FT_TT_TILE_HEIGHT; r++) {
        const unsigned char *src = (const unsigned char *)tensor->data + (size_t)r * (64u / 32u * 34u);

        ft_gguf_dequant_to_float(tensor->ggml_type, src, 64, row);
        memcpy(&full[r * 64u], row, 64u * sizeof(float));
    }
    for (uint32_t r = 0; r < FT_TT_TILE_HEIGHT; r++)
        for (uint32_t c = 0; c < FT_TT_TILE_WIDTH; c++)
            window[r * FT_TT_TILE_WIDTH + c] = full[c * 64u + r]; /* transposed source */

    f = fopen(artifact_path, "rb");
    check(f && fseek(f, (long)entry.data_offset, SEEK_SET) == 0 &&
              fread(tile, sizeof(uint16_t), FT_TT_TILE_VALUES, f) == FT_TT_TILE_VALUES,
          "read attn_q's first bf16 tile out of the artifact");
    if (f)
        fclose(f);

    for (uint32_t r = 0; r < FT_TT_TILE_HEIGHT; r++) {
        for (uint32_t c = 0; c < FT_TT_TILE_WIDTH; c++) {
            faced[r * FT_TT_TILE_WIDTH + c] = ft_float_from_bf16(tile[ft_tt_tile_face_index(r, c)]);
            faced_untransposed[r * FT_TT_TILE_WIDTH + c] =
                ft_float_from_bf16(tile[ft_tt_tile_face_index(c, r)]);
            flat[r * FT_TT_TILE_WIDTH + c] = ft_float_from_bf16(tile[r * FT_TT_TILE_WIDTH + c]);
        }
    }

    {
        double correct = correlate(window, faced, FT_TT_TILE_VALUES);
        double untransposed = correlate(window, faced_untransposed, FT_TT_TILE_VALUES);
        double as_flat = correlate(window, flat, FT_TT_TILE_VALUES);

        printf("        bf16 attn_q tile 0: face-ordered+transposed %.6f, untransposed %.6f, "
               "flat %.6f\n",
               correct, untransposed, as_flat);
        check(correct > 0.999, "the face-ordered tile holds the TRANSPOSED source");
        check(correct - untransposed > 0.5, "reading it untransposed is clearly worse");
        check(correct - as_flat > 0.5, "and reading it flat is clearly worse");
    }

    ft_gguf_map_close(map);
}

static void test_gguf_and_convert(void)
{
    const char *gguf_path = "build/test_synthetic.gguf";
    const char *artifact_path = "build/test_synthetic.fttw";
    ft_gguf_map *map = NULL;
    char architecture[32] = "";

    section("synthetic GGUF round trip");

    check(write_synthetic_gguf(gguf_path) == 0, "wrote a synthetic GGUF");
    check(ft_gguf_map_open(gguf_path, &map) == FT_GGUF_OK, "opened it");
    if (!map)
        return;

    check_u64(ft_gguf_map_version(map), 3, "GGUF version");
    check_u64(ft_gguf_map_tensor_count(map), SYNTHETIC_COUNT, "tensor count");
    check(ft_gguf_map_metadata_string(map, "general.architecture", architecture, sizeof(architecture)) == FT_GGUF_OK &&
              strcmp(architecture, "llama") == 0,
          "general.architecture reads back as llama");
    check(ft_gguf_map_find(map, "blk.0.ffn_down.weight") != NULL, "found a tensor by name");
    check(ft_gguf_map_find(map, "no.such.tensor") == NULL, "a missing tensor is NULL");

    {
        const ft_gguf_tensor *tensor = ft_gguf_map_find(map, "blk.0.attn_q.weight");

        check(tensor && tensor->ggml_type == FT_GGML_TYPE_Q8_0, "attn_q is Q8_0");
        if (tensor)
            check_u64(tensor->byte_count, 64u * 64u / 32u * 34u, "attn_q byte count matches the Q8_0 block size");
    }

    ft_gguf_map_close(map);

    section("conversion under each ladder rung");
    {
        static const struct {
            ft_tt_ledger_mode mode;
            const char *name;
        } RUNGS[] = {{FT_TT_LEDGER_FP32, "fp32"}, {FT_TT_LEDGER_BF16, "bf16"}, {FT_TT_LEDGER_PRODUCTION, "production"}};
        uint64_t sizes[3] = {0, 0, 0};

        for (int r = 0; r < 3; r++) {
            ft_tt_convert_options options;
            convert_observer observer;
            char label[96];

            memset(&options, 0, sizeof(options));
            memset(&observer, 0, sizeof(observer));
            observer.worst_pcc = 2.0;
            options.mode = RUNGS[r].mode;
            options.measure_pcc = 1;
            options.progress = observe;
            options.progress_user = &observer;

            snprintf(label, sizeof(label), "converted at the %s rung", RUNGS[r].name);
            check(ft_tt_convert_gguf(gguf_path, artifact_path, &options) == FT_TT_CONVERT_OK, label);
            check_u64(observer.tensors, SYNTHETIC_COUNT, "every tensor was reported");
            sizes[r] = observer.converted_bytes;

            snprintf(label, sizeof(label), "%s rung worst PCC %.9f", RUNGS[r].name, observer.worst_pcc);
            check(observer.worst_pcc > 0.9, label);

            if (RUNGS[r].mode == FT_TT_LEDGER_FP32)
                check(observer.worst_pcc == 1.0, "the fp32 rung is lossless, so PCC is exactly 1");
            if (RUNGS[r].mode == FT_TT_LEDGER_PRODUCTION) {
                check(observer.formats[3] == FT_TT_FORMAT_BFP8_B, "ffn_gate landed in BFP8");
                check(observer.formats[4] == FT_TT_FORMAT_BFP8_B, "ffn_down landed in BFP8");
                check(observer.formats[1] == FT_TT_FORMAT_FP32, "the 1-D norm stayed fp32");
            }
        }

        check(sizes[1] < sizes[0], "bf16 is smaller than fp32");
        check(sizes[2] < sizes[1], "the production ledger is smaller than bf16");
        printf("        fp32 %" PRIu64 " B, bf16 %" PRIu64 " B, production %" PRIu64 " B\n", sizes[0], sizes[1],
               sizes[2]);
    }

    section("artifact structure");
    {
        FILE *f = fopen(artifact_path, "rb");
        ft_tt_artifact_header header;
        ft_tt_artifact_entry entry;

        check_u64(sizeof(ft_tt_artifact_header), FT_TT_ARTIFACT_HEADER_BYTES, "header size is pinned");
        check_u64(sizeof(ft_tt_artifact_entry), FT_TT_ARTIFACT_ENTRY_BYTES, "directory entry size is pinned");

        check(f != NULL, "artifact opens");
        if (!f)
            return;
        check(fread(&header, sizeof(header), 1, f) == 1, "header reads");
        check_u64(header.magic, FT_TT_ARTIFACT_MAGIC, "magic");
        check_u64(header.version, FT_TT_ARTIFACT_VERSION_DENSE, "version (a dense file stays v2)");
        check_u64(header.tensor_count, SYNTHETIC_COUNT, "tensor count");
        check(header.directory_offset > header.data_offset, "the directory follows the data");
        check(header.data_offset % FT_TT_ARTIFACT_ALIGNMENT == 0, "data starts aligned");

        check(fseek(f, (long)header.directory_offset, SEEK_SET) == 0, "seek to the directory");
        {
            uint64_t previous_end = header.data_offset;
            size_t misordered = 0;
            size_t misaligned = 0;
            size_t needed_padding = 0;

            for (uint32_t i = 0; i < header.tensor_count; i++) {
                if (fread(&entry, sizeof(entry), 1, f) != 1)
                    break;
                if (entry.data_offset < previous_end)
                    misordered++;
                if (entry.data_offset % FT_TT_ARTIFACT_ALIGNMENT != 0)
                    misaligned++;
                if (entry.byte_count % FT_TT_ARTIFACT_ALIGNMENT != 0)
                    needed_padding++;
                previous_end = entry.data_offset + entry.byte_count;
                if (i == 0) {
                    check(strcmp(entry.name, "token_embd.weight") == 0, "first entry is token_embd");
                    check_u64(entry.tile_cols, ft_tt_tile_count_for(64), "token_embd tile columns");
                    check_u64(entry.tile_rows, ft_tt_tile_count_for(100), "token_embd tile rows");
                }
            }
            check_u64(misordered, 0, "entries are in ascending, non-overlapping offset order");
            check_u64(misaligned, 0, "every tensor starts on a 64-byte boundary");
            /* Without this the alignment check above is vacuous: every tiled
             * format's size already divides by 64, so only an odd 1-D tensor
             * makes the padding do any work. */
            check(needed_padding > 0, "at least one tensor's size actually needed padding");
        }
        fclose(f);
    }

    section("measure-only mode writes nothing and scores everything");
    {
        convert_observer observer;

        memset(&observer, 0, sizeof(observer));
        observer.worst_pcc = 2.0;
        check(ft_tt_convert_measure(gguf_path, FT_TT_FORMAT_BFP8_B, observe, &observer) == FT_TT_CONVERT_OK,
              "measured at BFP8");
        check_u64(observer.tensors, SYNTHETIC_COUNT, "every tensor was scored");
        check(observer.worst_pcc > 0.99, "BFP8 PCC on synthetic weights stays above 0.99");
        printf("        BFP8 worst PCC %.9f\n", observer.worst_pcc);

        memset(&observer, 0, sizeof(observer));
        observer.worst_pcc = 2.0;
        check(ft_tt_convert_measure(gguf_path, FT_TT_FORMAT_BFP4_B, observe, &observer) == FT_TT_CONVERT_OK,
              "measured at BFP4");
        printf("        BFP4 worst PCC %.9f\n", observer.worst_pcc);
        check(observer.worst_pcc < 1.0, "BFP4 is lossy, so it cannot score a perfect 1.0");
    }

    test_artifact_layout(gguf_path, artifact_path);
    test_face_ordered_transpose(gguf_path);

    section("tensor file offsets address the same bytes through the filesystem");
    {
        /* The accessor exists so a caller can reach a tensor WITHOUT this
         * mapping, so the check has to leave the mapping: read the bytes back
         * with stdio at the reported offset and compare against the mapped
         * pointer. A check done through `map` alone would only prove the
         * arithmetic is self-consistent. */
        ft_gguf_map *offset_map = NULL;

        if (ft_gguf_map_open(gguf_path, &offset_map) == FT_GGUF_OK) {
            uint32_t matched = 0;
            uint32_t count = ft_gguf_map_tensor_count(offset_map);
            FILE *raw = fopen(gguf_path, "rb");

            for (uint32_t i = 0; raw != NULL && i < count; i++) {
                const ft_gguf_tensor *tensor = ft_gguf_map_tensor(offset_map, i);
                uint64_t offset = ft_gguf_map_tensor_file_offset(offset_map, tensor);
                unsigned char from_file[64];
                size_t want = tensor->byte_count < sizeof(from_file) ? (size_t)tensor->byte_count
                                                                    : sizeof(from_file);

                if (fseek(raw, (long)offset, SEEK_SET) != 0)
                    continue;
                if (fread(from_file, 1, want, raw) != want)
                    continue;
                if (memcmp(from_file, tensor->data, want) == 0)
                    matched++;
            }
            if (raw != NULL)
                fclose(raw);

            check_u64(matched, count, "tensors whose file offset reads back the mapped bytes");
            check(ft_gguf_map_tensor_file_offset(offset_map, NULL) == 0, "a NULL tensor has offset 0");
            check(ft_gguf_map_tensor_file_offset(NULL, ft_gguf_map_tensor(offset_map, 0)) == 0,
                  "a NULL offset_map has offset 0");
            ft_gguf_map_close(offset_map);
        } else {
            check(0, "reopened the synthetic checkpoint for the offset check");
        }
    }

    section("bad input is refused, not misread");
    {
        ft_gguf_map *bad = NULL;

        check(ft_gguf_map_open("build/no_such_file.gguf", &bad) == FT_GGUF_EIO, "a missing file is EIO");
        check(ft_gguf_map_open("Makefile", &bad) == FT_GGUF_ERROR, "a non-GGUF file is rejected");
        check(ft_tt_convert_gguf(NULL, artifact_path, NULL) == FT_TT_CONVERT_EINVAL, "NULL input path rejected");
    }
}

/* ------------------------------------------------------------------------- */
/* Unreadable source formats — refused, and refused legibly                  */
/* ------------------------------------------------------------------------- */

/** A one-tensor 32x32 GGUF of an arbitrary declared type, zero-filled and long
 *  enough for even an fp32 payload. Everything except the type is identical
 *  between the cases below, so the only thing a refusal can be about is the
 *  type — a short file would be refused for running past the end instead, and
 *  that would be a different test passing for the wrong reason. */
static int write_gguf_with_type(const char *path, uint32_t ggml_type)
{
    FILE *f = fopen(path, "wb");
    int i;

    if (!f)
        return -1;

    put_u32(f, 0x46554747u); /* "GGUF" */
    put_u32(f, 3);
    put_u64(f, 1); /* one tensor */
    put_u64(f, 1); /* one metadata key */

    put_string(f, "general.architecture");
    put_u32(f, 8); /* string */
    put_string(f, "llama");

    put_string(f, "blk.0.attn_q.weight");
    put_u32(f, 2);
    put_u64(f, 32);
    put_u64(f, 32);
    put_u32(f, ggml_type);
    put_u64(f, 0);

    /* 32 x 32 at 4 B/element, plus alignment slack. */
    for (i = 0; i < 32 * 32 * 4 + 64; i++)
        fputc(0, f);

    fclose(f);
    return 0;
}

/**
 * F32, F16, BF16, Q8_0 and the five k-quants are readable; everything else is
 * refused BY NAME. The failure mode this guards is not a wrong number, it is a
 * person choosing the *second* target model, hitting this at conversion time,
 * and being told "unsupported ggml type 23". So the type must be named, and the
 * refusal must actually happen.
 *
 * The k-quants moved from the second column to the first in Phase 7.6 because
 * that is what a downloaded GGUF actually is. What did NOT move is the Route A
 * line: that was a dequantizing reader kernel ON THE DEVICE, and this reader
 * runs on the host in an offline converter, which is Route B.
 *
 * Every IQ type, the legacy Q4_0/Q4_1/Q5_0/Q5_1 family, Q8_1, Q8_K, MXFP4,
 * NVFP4 and the TQ ternary formats are still refused, and it matters that the
 * list below keeps naming them — a refusal that says "?" sends the reader to
 * the source for something the tool already knew.
 */
static void test_unreadable_formats_are_named_and_refused(void)
{
    /* type, name, and whether this converter reads it. */
    static const struct {
        uint32_t type;
        const char *name;
        int readable;
    } CASES[] = {
        {0, "F32", 1},    {1, "F16", 1},     {8, "Q8_0", 1},  {30, "BF16", 1},
        {12, "Q4_K", 1},  {11, "Q3_K", 1},   {14, "Q6_K", 1}, {10, "Q2_K", 1},
        {13, "Q5_K", 1},  {2, "Q4_0", 0},    {3, "Q4_1", 0},  {6, "Q5_0", 0},
        {7, "Q5_1", 0},   {9, "Q8_1", 0},    {15, "Q8_K", 0}, {23, "IQ4_XS", 0},
        {16, "IQ2_XXS", 0}, {19, "IQ1_S", 0}, {39, "MXFP4", 0}, {40, "NVFP4", 0},
        {34, "TQ1_0", 0}, {35, "TQ2_0", 0},  {26, "I32", 0},  {28, "F64", 0}
    };
    size_t case_count = sizeof(CASES) / sizeof(CASES[0]);
    int named = 0;
    int classified = 0;
    int sized = 0;

    section("unreadable source formats are named and refused");

    for (size_t i = 0; i < case_count; i++) {
        if (strcmp(ft_gguf_type_name(CASES[i].type), CASES[i].name) == 0)
            named++;
        if (ft_gguf_type_supported(CASES[i].type) == (CASES[i].readable ? 1 : 0))
            classified++;
        /* An unreadable type must size to 0 — that zero is what trips the
         * refusal in ft_gguf_map.c. A readable one must not. */
        if ((ft_gguf_type_bytes(CASES[i].type, 256) != 0) == (CASES[i].readable != 0))
            sized++;
    }

    check_u64((uint64_t)named, (uint64_t)case_count, "type numbers that name themselves");
    check_u64((uint64_t)classified, (uint64_t)case_count, "types classified readable/not correctly");
    check_u64((uint64_t)sized, (uint64_t)case_count, "unreadable types size to zero, readable ones do not");
    check(strcmp(ft_gguf_type_name(0xDEADu), "?") == 0, "a type number nobody has heard of is still '?'");

    /* BF16 is what a modern "unquantized" GGUF release ships as, and it is NOT
     * F16 — a reader that treated the two as interchangeable would produce
     * garbage rather than an error, so the distinction is pinned here. */
    check(ft_gguf_type_supported(30) != 0, "BF16 is readable -- it is what the 'full precision' GGUF ships as");
    check(strcmp(ft_gguf_type_name(30), "BF16") == 0, "BF16 is named BF16, not '?'");
    check(FT_GGML_TYPE_BF16 != FT_GGML_TYPE_F16, "BF16 and F16 are different types, and not aliases");

    {
        ft_gguf_map *map = NULL;

        /* IQ4_XS is a real format a real "small" release ships as, so this is
         * the refusal a person actually hits now that the k-quants are read. */
        check(write_gguf_with_type("build/test_iquant.gguf", 23) == 0, "wrote an IQ4_XS checkpoint");
        check(ft_gguf_map_open("build/test_iquant.gguf", &map) == FT_GGUF_ERROR,
              "an IQ4_XS checkpoint is refused at open, not misread");
        check(map == NULL, "nothing is handed back for a refused checkpoint");

        check(write_gguf_with_type("build/test_kquant.gguf", 12) == 0, "wrote a Q4_K checkpoint");
        check(ft_gguf_map_open("build/test_kquant.gguf", &map) == FT_GGUF_OK,
              "a Q4_K checkpoint IS accepted now -- 32x32 is four whole super-blocks");
        ft_gguf_map_close(map);
        map = NULL;

        check(write_gguf_with_type("build/test_bf16.gguf", 30) == 0, "wrote a BF16 checkpoint");
        check(ft_gguf_map_open("build/test_bf16.gguf", &map) == FT_GGUF_OK, "a BF16 checkpoint IS accepted");
        ft_gguf_map_close(map);
        map = NULL;

        check(write_gguf_with_type("build/test_f16.gguf", FT_GGML_TYPE_F16) == 0, "wrote an F16 checkpoint");
        check(ft_gguf_map_open("build/test_f16.gguf", &map) == FT_GGUF_OK,
              "an F16 checkpoint of the same shape IS accepted -- the refusal is about the type");
        ft_gguf_map_close(map);
    }
}

/* ------------------------------------------------------------------------- */
/* K-quants, against upstream ggml's own answer                              */
/* ------------------------------------------------------------------------- */

/** The five k-quants, with the block size each one's stride depends on. */
static const struct {
    uint32_t type;
    const char *name;
    size_t block_bytes;
} KQUANTS[] = {
    {FT_GGML_TYPE_Q2_K, "Q2_K", FT_GGML_Q2_K_BLOCK_BYTES},
    {FT_GGML_TYPE_Q3_K, "Q3_K", FT_GGML_Q3_K_BLOCK_BYTES},
    {FT_GGML_TYPE_Q4_K, "Q4_K", FT_GGML_Q4_K_BLOCK_BYTES},
    {FT_GGML_TYPE_Q5_K, "Q5_K", FT_GGML_Q5_K_BLOCK_BYTES},
    {FT_GGML_TYPE_Q6_K, "Q6_K", FT_GGML_Q6_K_BLOCK_BYTES}
};

#define KQUANT_COUNT ((int)(sizeof(KQUANTS) / sizeof(KQUANTS[0])))

/**
 * Every fp32 word ggml produced from the frozen super-blocks, reproduced
 * bit-for-bit.
 *
 * The expected words in reference/ft_gguf_kquant_reference_vectors.h came out
 * of upstream ggml's compiled `dequantize_row_qN_K` — not out of this file, not
 * out of ft_gguf_dequant.c, and not out of anyone's reading of ggml's source.
 * That is the whole point: foundry, Pantheon's K80 reader and ggml each hold
 * their own copy of this arithmetic, and a copy that only agrees with a
 * transcription of itself has been checked against nothing.
 *
 * Bit-exact rather than a tolerance, and that is a real claim rather than a
 * convenience: Q2_K, Q4_K and Q5_K end in `scale*q - min`, a subtraction that
 * rounds, so an implementation that computed the same VALUE by a different
 * association would land on a different word and this would catch it.
 *
 * Each type is decoded twice — every vector one super-block at a time, and then
 * its first four vectors as one contiguous run. The second is not redundant: the run is
 * the only thing that exercises the block STRIDE, and a stride wrong by one
 * type's worth of bytes still decodes the first block perfectly.
 */
static void test_kquant_readers_match_ggml(void)
{
    uint64_t exact_single = 0;
    uint64_t exact_run = 0;
    uint64_t run_words = 0;
    uint64_t expected_words = 0;
    uint64_t nonzero = 0;
    uint64_t ordinary_words = 0;
    uint64_t discriminated = 0;
    uint64_t comparisons = 0;

    section("k-quants reproduce upstream ggml's fp32 words exactly");

    for (int k = 0; k < KQUANT_COUNT; k++) {
        unsigned char run[4 * FT_GGML_Q6_K_BLOCK_BYTES];
        float run_decoded[4 * FT_KQUANT_REF_VALUES];
        const uint32_t *run_expected[4];
        size_t in_run = 0;

        for (size_t v = 0; v < FT_KQUANT_REF_COUNT; v++) {
            const ft_kquant_reference_vector *vector = &ft_kquant_reference_vectors[v];
            float decoded[FT_KQUANT_REF_VALUES];

            if (vector->ggml_type != KQUANTS[k].type)
                continue;

            check_u64(vector->byte_count, (uint64_t)KQUANTS[k].block_bytes, "the frozen block is the format's size");

            if (ft_gguf_dequant_to_float(vector->ggml_type, vector->bytes, FT_KQUANT_REF_VALUES, decoded) != 0) {
                check(0, "a frozen super-block decodes");
                continue;
            }
            for (int i = 0; i < FT_KQUANT_REF_VALUES; i++) {
                expected_words++;
                if (bits_from_float(decoded[i]) == vector->expected_bits[i])
                    exact_single++;
                if (strncmp(vector->origin, "edge:", 5) != 0) {
                    ordinary_words++;
                    if (vector->expected_bits[i] != 0u)
                        nonzero++;
                }
            }

            if (in_run < 4) {
                memcpy(run + in_run * KQUANTS[k].block_bytes, vector->bytes, KQUANTS[k].block_bytes);
                run_expected[in_run] = vector->expected_bits;
                in_run++;
            }
        }

        check_u64((uint64_t)in_run, 4u, "four frozen super-blocks for this format");

        if (ft_gguf_dequant_to_float(KQUANTS[k].type, run, (uint64_t)in_run * FT_KQUANT_REF_VALUES, run_decoded) !=
            0) {
            check(0, "a multi-super-block run decodes");
            continue;
        }
        for (size_t b = 0; b < in_run; b++) {
            for (int i = 0; i < FT_KQUANT_REF_VALUES; i++) {
                run_words++;
                if (bits_from_float(run_decoded[b * FT_KQUANT_REF_VALUES + (size_t)i]) == run_expected[b][i])
                    exact_run++;
                /* Block b's answer must differ from block b-1's SOMEWHERE, or
                 * a reader stuck on block 0 would score a perfect run. */
                if (b > 0) {
                    comparisons++;
                    if (run_expected[b][i] != run_expected[b - 1][i])
                        discriminated++;
                }
            }
        }
    }

    check_u64(exact_single, (uint64_t)(FT_KQUANT_REF_COUNT * FT_KQUANT_REF_VALUES),
              "words that match ggml exactly, decoding one super-block at a time");
    check_u64(run_words, (uint64_t)(KQUANT_COUNT * 4 * FT_KQUANT_REF_VALUES), "words decoded as four-block runs");
    check_u64(exact_run, run_words, "words that match ggml exactly, decoding four super-blocks as one run");
    check_u64(expected_words, (uint64_t)(FT_KQUANT_REF_COUNT * FT_KQUANT_REF_VALUES), "words compared");

    /* The comparison has to be capable of saying no. Two guards: the expected
     * words of the non-edge vectors are mostly not zero (so "all zeros" would
     * not pass; the all-zero EDGE block is zero by design and is left out), and
     * adjacent super-blocks genuinely differ (so a reader that ignored its input
     * pointer would not pass either). */
    check(ordinary_words > 0 && nonzero > ordinary_words * 9u / 10u,
          "and the words being matched are overwhelmingly non-zero");
    check(comparisons > 0 && discriminated > comparisons / 2u,
          "adjacent super-blocks disagree, so a reader stuck on one block cannot pass");
}

/**
 * Where the frozen bytes came from, per format. Recorded as a check rather than
 * a comment so the provenance cannot drift silently: every block is synthetic
 * (seeded), and each format has the same mix — two blocks from ggml's own
 * `quantize_row_qN_K_ref` over seeded weights, two of pseudorandom bytes, and
 * four edge cases. Anything else (a block from a model checkpoint) fails.
 */
static void test_kquant_vector_provenance_is_declared(void)
{
    int quantized_blocks[KQUANT_COUNT];
    int random_blocks[KQUANT_COUNT];
    int edge_blocks[KQUANT_COUNT];
    int other_blocks = 0;

    section("where the k-quant reference bytes came from");

    for (int k = 0; k < KQUANT_COUNT; k++) {
        quantized_blocks[k] = 0;
        random_blocks[k] = 0;
        edge_blocks[k] = 0;
    }
    for (size_t v = 0; v < FT_KQUANT_REF_COUNT; v++) {
        const ft_kquant_reference_vector *vector = &ft_kquant_reference_vectors[v];

        for (int k = 0; k < KQUANT_COUNT; k++) {
            if (vector->ggml_type != KQUANTS[k].type)
                continue;
            if (strncmp(vector->origin, "synthetic: ggml quantize_row_", 29) == 0)
                quantized_blocks[k]++;
            else if (strncmp(vector->origin, "pseudorandom", 12) == 0)
                random_blocks[k]++;
            else if (strncmp(vector->origin, "edge:", 5) == 0)
                edge_blocks[k]++;
            else
                other_blocks++;
        }
    }

    for (int k = 0; k < KQUANT_COUNT; k++) {
        check_u64((uint64_t)quantized_blocks[k], 2u, "blocks from ggml's own quantizer over seeded weights");
        check_u64((uint64_t)random_blocks[k], 2u, "pseudorandom super-blocks, for code points a quantizer misses");
        check_u64((uint64_t)edge_blocks[k], 4u, "edge-case super-blocks (zero, saturated, negative, subnormal)");
    }
    check_u64((uint64_t)other_blocks, 0u, "blocks of any other provenance -- NONE, every block is synthetic");
}

/**
 * Sizing, and the boundary a k-quant reader is most likely to get wrong.
 *
 * A partial super-block must size to zero and must be refused, because that
 * zero is what makes ft_gguf_map.c stop instead of running off the end of the
 * mapping. 255 and 257 are both checked: an off-by-one in a `% 256` test tends
 * to survive one of them.
 */
static void test_kquant_sizing_and_partial_blocks(void)
{
    unsigned char block[FT_GGML_Q6_K_BLOCK_BYTES * 2] = {0};
    float decoded[FT_KQUANT_REF_VALUES * 2];

    section("k-quant sizing, and partial super-blocks refused");

    for (int k = 0; k < KQUANT_COUNT; k++) {
        uint32_t type = KQUANTS[k].type;
        uint64_t block_bytes = (uint64_t)KQUANTS[k].block_bytes;

        check(ft_gguf_type_supported(type) != 0, "the format is readable");
        check(strcmp(ft_gguf_type_name(type), KQUANTS[k].name) == 0, "and it still names itself");
        check_u64(ft_gguf_type_bytes(type, FT_GGML_K_SUPERBLOCK), block_bytes, "one super-block's size");
        check_u64(ft_gguf_type_bytes(type, FT_GGML_K_SUPERBLOCK * 7u), block_bytes * 7u, "seven super-blocks");
        check_u64(ft_gguf_type_bytes(type, FT_GGML_K_SUPERBLOCK - 1u), 0u, "one element short sizes to zero");
        check_u64(ft_gguf_type_bytes(type, FT_GGML_K_SUPERBLOCK + 1u), 0u, "one element over sizes to zero");
        check(ft_gguf_dequant_to_float(type, block, FT_GGML_K_SUPERBLOCK - 1u, decoded) != 0,
              "a partial super-block is refused rather than decoded short");
        check(ft_gguf_dequant_to_float(type, NULL, FT_GGML_K_SUPERBLOCK, decoded) != 0, "a NULL source is refused");
        check(ft_gguf_dequant_to_float(type, block, FT_GGML_K_SUPERBLOCK, NULL) != 0,
              "a NULL destination is refused");
    }

    /* The five block sizes are distinct and are the ones ggml's static_asserts
     * pin. Getting one wrong reads the next block from the wrong offset, which
     * is the single most damaging mistake available here. */
    check_u64(FT_GGML_Q2_K_BLOCK_BYTES, 84u, "Q2_K super-block bytes");
    check_u64(FT_GGML_Q3_K_BLOCK_BYTES, 110u, "Q3_K super-block bytes");
    check_u64(FT_GGML_Q4_K_BLOCK_BYTES, 144u, "Q4_K super-block bytes");
    check_u64(FT_GGML_Q5_K_BLOCK_BYTES, 176u, "Q5_K super-block bytes");
    check_u64(FT_GGML_Q6_K_BLOCK_BYTES, 210u, "Q6_K super-block bytes");
}

/**
 * A k-quant checkpoint, through the mapping and out the other side as an
 * artifact — the thing Phase 7.6 is actually asking for.
 *
 * The payload is the frozen reference blocks rather than something this test
 * quantized itself, so the bytes on disk are ggml's and the fp32 the strip
 * loader sees can be checked against ggml's answer for those same bytes.
 */
static int write_kquant_gguf(const char *path, uint32_t type, size_t block_bytes, uint64_t cols, uint64_t rows,
                             const unsigned char *payload, size_t payload_bytes)
{
    FILE *f = fopen(path, "wb");
    long header_end;
    long padded;

    if (!f)
        return -1;

    put_u32(f, 0x46554747u); /* "GGUF" */
    put_u32(f, 3);
    put_u64(f, 1);
    put_u64(f, 4);
    write_synthetic_model_metadata(f);

    put_string(f, "blk.0.attn_q.weight");
    put_u32(f, 2);
    put_u64(f, cols);
    put_u64(f, rows);
    put_u32(f, type);
    put_u64(f, 0);

    header_end = ftell(f);
    padded = (header_end + 31) / 32 * 32;
    for (long i = header_end; i < padded; i++)
        fputc(0, f);

    fwrite(payload, 1, payload_bytes, f);
    /* Pad past the declared extent so a refusal can never be about a short
     * file — the only thing under test here is the type and the block count. */
    for (size_t i = payload_bytes; i < block_bytes * 64u + 64u; i++)
        fputc(0, f);

    fclose(f);
    return 0;
}

static void test_kquant_checkpoint_maps_and_converts(void)
{
    section("a k-quant checkpoint maps, dequantizes and converts");

    for (int k = 0; k < KQUANT_COUNT; k++) {
        unsigned char payload[4 * FT_GGML_Q6_K_BLOCK_BYTES];
        const uint32_t *expected[4];
        size_t blocks = 0;
        char path[128];
        char artifact[128];
        ft_gguf_map *map = NULL;

        for (size_t v = 0; v < FT_KQUANT_REF_COUNT && blocks < 4; v++) {
            const ft_kquant_reference_vector *vector = &ft_kquant_reference_vectors[v];

            if (vector->ggml_type != KQUANTS[k].type)
                continue;
            memcpy(payload + blocks * KQUANTS[k].block_bytes, vector->bytes, KQUANTS[k].block_bytes);
            expected[blocks] = vector->expected_bits;
            blocks++;
        }

        snprintf(path, sizeof(path), "build/test_%s.gguf", KQUANTS[k].name);
        snprintf(artifact, sizeof(artifact), "build/test_%s.fttw", KQUANTS[k].name);

        /* 4 rows of 256 columns: one super-block per row, so the strip loader
         * walks the block stride row by row exactly as it would on a real
         * checkpoint. */
        check(write_kquant_gguf(path, KQUANTS[k].type, KQUANTS[k].block_bytes, FT_GGML_K_SUPERBLOCK, 4, payload,
                                blocks * KQUANTS[k].block_bytes) == 0,
              "wrote a k-quant checkpoint whose payload is the frozen ggml blocks");

        if (ft_gguf_map_open(path, &map) != FT_GGUF_OK) {
            check(0, "the k-quant checkpoint opens");
            continue;
        }
        check(1, "the k-quant checkpoint opens");

        {
            const ft_gguf_tensor *tensor = ft_gguf_map_find(map, "blk.0.attn_q.weight");
            float decoded[4 * FT_KQUANT_REF_VALUES];
            uint64_t exact = 0;

            check(tensor != NULL && tensor->ggml_type == KQUANTS[k].type, "and the tensor kept its type");
            if (tensor) {
                check_u64(tensor->byte_count, (uint64_t)KQUANTS[k].block_bytes * 4u,
                          "the mapping sized it by the format's block size");
                check(ft_gguf_dequant_to_float(tensor->ggml_type, tensor->data, 4u * FT_GGML_K_SUPERBLOCK,
                                               decoded) == 0,
                      "the mapped bytes dequantize");
                for (size_t b = 0; b < 4; b++)
                    for (int i = 0; i < FT_KQUANT_REF_VALUES; i++)
                        if (bits_from_float(decoded[b * FT_KQUANT_REF_VALUES + (size_t)i]) == expected[b][i])
                            exact++;
                check_u64(exact, 4u * FT_KQUANT_REF_VALUES, "and every word is still ggml's, read through the map");
            }
        }
        ft_gguf_map_close(map);

        check(ft_tt_convert_gguf(path, artifact, NULL) == FT_TT_CONVERT_OK,
              "and the converter turns it into an artifact");
    }

    /* A k-quant tensor whose row is not a whole super-block cannot exist in a
     * real checkpoint, but a corrupt or hand-built one can claim it. It must be
     * refused, and — since the type IS readable — refused for the right reason. */
    {
        unsigned char payload[FT_GGML_Q4_K_BLOCK_BYTES * 2] = {0};
        ft_gguf_map *map = NULL;

        check(write_kquant_gguf("build/test_q4k_partial.gguf", FT_GGML_TYPE_Q4_K, FT_GGML_Q4_K_BLOCK_BYTES, 200, 1,
                                payload, sizeof(payload)) == 0,
              "wrote a Q4_K tensor claiming a 200-element row");
        check(ft_gguf_map_open("build/test_q4k_partial.gguf", &map) == FT_GGUF_ERROR,
              "a Q4_K row that is not a whole super-block is refused, not read past");
        check(map == NULL, "and nothing is handed back");
    }
}

/* ------------------------------------------------------------------------- */
/* A BF16 source is read EXACTLY                                             */
/* ------------------------------------------------------------------------- */

/**
 * BF16 is admitted for one reason: widening it to fp32 is exact, so it adds no
 * error term underneath a measurement whose whole job is to attribute error to
 * the OUTPUT format. That claim is checked exhaustively rather than sampled —
 * all 65,536 bit patterns — because "exact" is the entire justification, and a
 * spot check of a few hundred values would pass for a reader that mishandled
 * denormals, infinities or the sign of zero.
 */
static void test_bf16_source_is_exact(void)
{
    uint64_t exact = 0;
    uint64_t finite = 0;
    uint64_t nans = 0;
    uint64_t denormals = 0;

    section("a BF16 source widens to fp32 exactly");

    for (uint32_t pattern = 0; pattern < 65536u; pattern++) {
        uint16_t narrow = (uint16_t)pattern;
        uint32_t expected_bits = (uint32_t)pattern << 16;
        float decoded = 0.0f;
        uint32_t got_bits;

        if (ft_gguf_dequant_to_float(FT_GGML_TYPE_BF16, &narrow, 1, &decoded) != 0)
            continue;
        got_bits = bits_from_float(decoded);
        if (got_bits == expected_bits)
            exact++;

        if ((pattern & 0x7f80u) == 0x7f80u) {
            if ((pattern & 0x007fu) != 0)
                nans++;
        } else {
            finite++;
            if ((pattern & 0x7f80u) == 0)
                denormals++;
        }
    }

    check_u64(exact, 65536u, "bf16 patterns that widen to exactly their own bits");
    check(finite > 60000u, "and most of them were finite values, not a table of NaNs");
    check(nans > 0, "NaN patterns were among them");
    check(denormals > 200u, "so were denormals, which a naive shift-and-scale reader gets wrong");

    /* The harness must be able to say no. A reader that dropped the low
     * mantissa byte would still round-trip a great many patterns, so prove the
     * comparison above actually discriminates by feeding it a wrong answer. */
    {
        uint16_t narrow = 0x3f81u; /* 1.0078125f, whose low mantissa bit is set */
        float decoded = 0.0f;
        uint32_t truncated = 0x3f80u << 16;

        check(ft_gguf_dequant_to_float(FT_GGML_TYPE_BF16, &narrow, 1, &decoded) == 0, "a lone bf16 value decodes");
        check(bits_from_float(decoded) != truncated, "and it is NOT the value a mantissa-dropping reader would give");
        check(bits_from_float(decoded) == ((uint32_t)narrow << 16), "it is the exact widening");
    }

    check_u64(ft_gguf_type_bytes(FT_GGML_TYPE_BF16, 10), 20, "BF16 sizing");
    check(ft_gguf_type_supported(FT_GGML_TYPE_BF16) != 0, "BF16 is readable");
    check(strcmp(ft_gguf_type_name(FT_GGML_TYPE_BF16), "BF16") == 0, "and it is still named BF16");
}

/* ------------------------------------------------------------------------- */
/* A BF16 checkpoint packs to Bfp8 straight from its own values              */
/* ------------------------------------------------------------------------- */

#define DIRECT_SIDE 64u
#define DIRECT_VALUES (DIRECT_SIDE * DIRECT_SIDE)

/** Convert `gguf` at the uniform bfp8 rung; copy blk.0.attn_q.weight's packed bytes out. */
static size_t convert_and_read_bfp8_payload(const char *gguf, const char *artifact, unsigned char *out,
                                            size_t out_bytes)
{
    ft_tt_convert_options options;
    ft_tt_artifact_entry entry;
    FILE *f;
    size_t got = 0;

    memset(&options, 0, sizeof(options));
    options.mode = FT_TT_LEDGER_BFP8;
    if (ft_tt_convert_gguf(gguf, artifact, &options) != FT_TT_CONVERT_OK)
        return 0;
    if (!read_artifact_entry(artifact, "blk.0.attn_q.weight", &entry) || entry.format != FT_TT_FORMAT_BFP8_B ||
        entry.byte_count > out_bytes)
        return 0;
    f = fopen(artifact, "rb");
    if (f && fseek(f, (long)entry.data_offset, SEEK_SET) == 0)
        got = fread(out, 1, (size_t)entry.byte_count, f);
    if (f)
        fclose(f);
    return got == entry.byte_count ? got : 0;
}

/** The Q8_0 rendering of `values` (rows of DIRECT_SIDE), as llama.cpp would quantize it. */
static size_t q8_0_payload(const float *values, unsigned char *out, size_t out_bytes)
{
    FILE *f = fopen("build/test_direct_q8_rows.bin", "w+b");
    size_t got = 0;

    if (!f)
        return 0;
    for (uint32_t r = 0; r < DIRECT_SIDE; r++)
        write_q8_0_row(f, values + r * DIRECT_SIDE, DIRECT_SIDE);
    rewind(f);
    got = fread(out, 1, out_bytes, f);
    fclose(f);
    return got;
}

/**
 * The claim the BF16-sourced Ornith artifact rests on: a BF16 tensor is packed to Bfp8 from
 * its own bf16 values, with no 8-bit stop on the way. Checked on the artifact bytes, three
 * renderings of ONE bf16-exact matrix:
 *   BF16 == F32  — the BF16 path is the exact values (an F32 source has no rounding step);
 *   BF16 != Q8_0 — and a Q8_0 rendering of the same values is NOT, so the comparison can say no.
 */
static void test_bf16_packs_bfp8_from_its_own_values(void)
{
    static float values[DIRECT_VALUES];
    static uint16_t narrow[DIRECT_VALUES];
    static unsigned char q8[DIRECT_VALUES / FT_GGML_Q8_0_BLOCK * 34u];
    static unsigned char from_bf16[2 * DIRECT_VALUES], from_f32[2 * DIRECT_VALUES], from_q8[2 * DIRECT_VALUES];
    size_t n_bf16, n_f32, n_q8;

    section("a BF16 checkpoint packs to Bfp8 from its own values, not via 8-bit");

    for (uint32_t i = 0; i < DIRECT_VALUES; i++) {
        narrow[i] = ft_bf16_from_float(synthetic_value(7u, i));
        values[i] = ft_float_from_bf16(narrow[i]);
    }
    check(write_kquant_gguf("build/test_direct_bf16.gguf", FT_GGML_TYPE_BF16, 2, DIRECT_SIDE, DIRECT_SIDE,
                            (const unsigned char *)narrow, sizeof(narrow)) == 0, "wrote the BF16 checkpoint");
    check(write_kquant_gguf("build/test_direct_f32.gguf", FT_GGML_TYPE_F32, 4, DIRECT_SIDE, DIRECT_SIDE,
                            (const unsigned char *)values, sizeof(values)) == 0, "wrote the same values as F32");
    check(q8_0_payload(values, q8, sizeof(q8)) == sizeof(q8) &&
              write_kquant_gguf("build/test_direct_q8.gguf", FT_GGML_TYPE_Q8_0, 34, DIRECT_SIDE, DIRECT_SIDE, q8,
                                sizeof(q8)) == 0,
          "and as Q8_0");

    n_bf16 = convert_and_read_bfp8_payload("build/test_direct_bf16.gguf", "build/test_direct_bf16.fttw", from_bf16,
                                           sizeof(from_bf16));
    n_f32 = convert_and_read_bfp8_payload("build/test_direct_f32.gguf", "build/test_direct_f32.fttw", from_f32,
                                          sizeof(from_f32));
    n_q8 = convert_and_read_bfp8_payload("build/test_direct_q8.gguf", "build/test_direct_q8.fttw", from_q8,
                                         sizeof(from_q8));
    check(n_bf16 > 0 && n_bf16 == n_f32 && n_bf16 == n_q8, "all three converted to Bfp8 payloads of one size");
    check(n_bf16 > 0 && memcmp(from_bf16, from_f32, n_bf16) == 0,
          "BF16 source == F32 source of the same values, byte for byte");
    check(n_bf16 > 0 && memcmp(from_bf16, from_q8, n_bf16) != 0,
          "and != the Q8_0 rendering of them (the comparison can fail)");
}

/* ------------------------------------------------------------------------- */
/* Split checkpoints                                                         */
/* ------------------------------------------------------------------------- */

/** How the SYNTHETIC tensors are dealt out across the parts written below. */
#define SPLIT_PARTS 3u
static const int SPLIT_FIRST[SPLIT_PARTS + 1u] = {0, 3, 6, SYNTHETIC_COUNT};

static void split_part_path(char *out, size_t out_bytes, const char *stem, uint32_t index, uint32_t count)
{
    snprintf(out, out_bytes, "%s-%05u-of-%05u.gguf", stem, index + 1u, count);
}

/** Write all SPLIT_PARTS parts of `stem`, optionally lying in one of them. */
static int write_split_checkpoint(const char *stem, uint32_t liar_index, const synthetic_split *lie)
{
    for (uint32_t index = 0; index < SPLIT_PARTS; index++) {
        synthetic_split split;
        char path[512];

        split.part_count = SPLIT_PARTS;
        split.part_index = index;
        split.declared_total = SYNTHETIC_COUNT;
        if (lie && index == liar_index)
            split = *lie;

        split_part_path(path, sizeof(path), stem, index, SPLIT_PARTS);
        if (write_synthetic_gguf_range(path, SPLIT_FIRST[index], SPLIT_FIRST[index + 1] - SPLIT_FIRST[index],
                                       &split) != 0)
            return -1;
    }
    return 0;
}

/** Copy `src` to `dst`, keeping at most `keep_bytes`. Returns bytes written. */
static long copy_file_truncated(const char *src, const char *dst, long keep_bytes)
{
    FILE *in = fopen(src, "rb");
    FILE *out = NULL;
    long copied = 0;
    int byte;

    if (!in)
        return -1;
    out = fopen(dst, "wb");
    if (!out) {
        fclose(in);
        return -1;
    }
    while (copied < keep_bytes && (byte = fgetc(in)) != EOF) {
        fputc(byte, out);
        copied++;
    }
    fclose(in);
    fclose(out);
    return copied;
}

static long file_bytes(const char *path)
{
    FILE *f = fopen(path, "rb");
    long bytes;

    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    bytes = ftell(f);
    fclose(f);
    return bytes;
}

/** Flip one bit in the middle of a file. Returns 0 on success. */
static int corrupt_one_byte(const char *path, long offset)
{
    FILE *f = fopen(path, "r+b");
    int byte;

    if (!f)
        return -1;
    if (fseek(f, offset, SEEK_SET) != 0 || (byte = fgetc(f)) == EOF) {
        fclose(f);
        return -1;
    }
    if (fseek(f, offset, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    fputc(byte ^ 0x40, f);
    fclose(f);
    return 0;
}

/**
 * Compare a split reading against the unsplit one, tensor by tensor, byte for
 * byte.
 *
 * @return the number of tensors whose name, shape and PAYLOAD BYTES agree
 */
static uint32_t split_tensors_matching_whole(const ft_gguf_map *split, const ft_gguf_map *whole)
{
    uint32_t matched = 0;

    for (uint32_t i = 0; i < ft_gguf_map_tensor_count(whole); i++) {
        const ft_gguf_tensor *want = ft_gguf_map_tensor(whole, i);
        const ft_gguf_tensor *got = ft_gguf_map_tensor(split, i);

        if (!got || strcmp(got->name, want->name) != 0)
            continue;
        if (got->ggml_type != want->ggml_type || got->byte_count != want->byte_count)
            continue;
        if (memcmp(got->data, want->data, (size_t)want->byte_count) != 0)
            continue;
        matched++;
    }
    return matched;
}

/** Byte-for-byte comparison of two files. */
static int files_are_identical(const char *a, const char *b)
{
    FILE *fa = fopen(a, "rb");
    FILE *fb = fopen(b, "rb");
    int same = 1;

    if (!fa || !fb) {
        if (fa)
            fclose(fa);
        if (fb)
            fclose(fb);
        return 0;
    }
    for (;;) {
        int ca = fgetc(fa);
        int cb = fgetc(fb);

        if (ca != cb) {
            same = 0;
            break;
        }
        if (ca == EOF)
            break;
    }
    fclose(fa);
    fclose(fb);
    return same;
}

/**
 * A split checkpoint reads as one model — and an incomplete one is refused.
 *
 * The failure this exists to make impossible is not a wrong number, it is a
 * conversion that SUCCEEDS while holding two thirds of a model: the artifact
 * would be well-formed, the tool would print a total, and the missing tensors
 * would only surface as a card producing nonsense. So the positive check is the
 * strong one available — the split reading must be byte-identical to the
 * unsplit reading of the same tensors, and the artifacts must be identical
 * files — and every way a part can be absent, stale or lying is checked to
 * REFUSE.
 */
static void test_split_checkpoint(void)
{
    const char *stem = "build/test_split";
    const char *whole_path = "build/test_split_whole.gguf";
    char part_path[SPLIT_PARTS][512];
    ft_gguf_map *whole = NULL;
    ft_gguf_map *split = NULL;

    section("a split checkpoint reads as one model");

    for (uint32_t i = 0; i < SPLIT_PARTS; i++)
        split_part_path(part_path[i], sizeof(part_path[i]), stem, i, SPLIT_PARTS);

    check(write_synthetic_gguf(whole_path) == 0, "wrote the unsplit reference checkpoint");
    check(write_split_checkpoint(stem, 0, NULL) == 0, "wrote the same tensors as 3 parts");
    check(ft_gguf_map_open(whole_path, &whole) == FT_GGUF_OK, "opened the unsplit reference");
    check(ft_gguf_map_part_count(whole) == 1, "an unsplit checkpoint is a one-part set");

    check(ft_gguf_map_open(part_path[0], &split) == FT_GGUF_OK, "opened the split checkpoint by its first part");
    if (!split || !whole) {
        check(0, "the split checkpoint could not be opened at all");
        ft_gguf_map_close(split);
        ft_gguf_map_close(whole);
        return;
    }

    check_u64(ft_gguf_map_part_count(split), SPLIT_PARTS, "parts opened");
    check_u64(ft_gguf_map_tensor_count(split), SYNTHETIC_COUNT, "tensors across the whole set");
    check_u64(split_tensors_matching_whole(split, whole), SYNTHETIC_COUNT,
              "tensors whose bytes match the unsplit reading");

    /* The tensor the brief cares about: one that lives past the first part. */
    {
        const char *late = SYNTHETIC[SYNTHETIC_COUNT - 1].name;
        const ft_gguf_tensor *tensor = ft_gguf_map_find(split, late);

        check(tensor != NULL, "the LAST tensor is found, and it lives in the last part");
        check_u64(ft_gguf_map_tensor_part(split, tensor), SPLIT_PARTS - 1u, "and its part index says so");
        if (tensor) {
            const char *path = ft_gguf_map_part_path(split, ft_gguf_map_tensor_part(split, tensor));

            check(path != NULL && strcmp(path, part_path[SPLIT_PARTS - 1]) == 0,
                  "its bytes are attributed to the right FILE, not to the one that was opened");
        }
    }

    /* Any part may be named; the set is always assembled from part 0 upwards. */
    {
        ft_gguf_map *from_last = NULL;

        check(ft_gguf_map_open(part_path[SPLIT_PARTS - 1], &from_last) == FT_GGUF_OK,
              "opening the LAST part opens the whole set too");
        if (from_last) {
            check_u64(ft_gguf_map_tensor_count(from_last), SYNTHETIC_COUNT, "and yields every tensor");
            check_u64(split_tensors_matching_whole(from_last, whole), SYNTHETIC_COUNT,
                      "in the same order, with the same bytes");
            ft_gguf_map_close(from_last);
        }
    }

    /* Metadata comes from part 0, which is the only part that carries it. */
    {
        char architecture[64] = "";

        check(ft_gguf_map_metadata_string(split, "general.architecture", architecture, sizeof(architecture)) ==
                      FT_GGUF_OK &&
                  strcmp(architecture, "llama") == 0,
              "the model's metadata is read from part 0");
    }

    ft_gguf_map_close(split);
    ft_gguf_map_close(whole);

    /* The end-to-end statement: same tensors in, same artifact out. */
    {
        check(ft_tt_convert_gguf(whole_path, "build/test_split_whole.fttw", NULL) == FT_TT_CONVERT_OK,
              "converted the unsplit checkpoint");
        check(ft_tt_convert_gguf(part_path[0], "build/test_split_parts.fttw", NULL) == FT_TT_CONVERT_OK,
              "converted the split checkpoint");
        check(files_are_identical("build/test_split_whole.fttw", "build/test_split_parts.fttw"),
              "the two artifacts are byte-identical");
        check(file_bytes("build/test_split_parts.fttw") > 0, "and the artifact is not empty");
    }

    /* Now prove that comparison can fail, by making it. One flipped bit in the
     * LAST part must reach the artifact — if it did not, the artifact equality
     * above would be agreeing about a part nobody read. */
    {
        long payload = file_bytes(part_path[SPLIT_PARTS - 1]);

        check(corrupt_one_byte(part_path[SPLIT_PARTS - 1], payload - 40) == 0,
              "flipped one bit inside the last part's tensor data");
        check(ft_tt_convert_gguf(part_path[0], "build/test_split_corrupt.fttw", NULL) == FT_TT_CONVERT_OK,
              "the corrupted set still converts");
        check(!files_are_identical("build/test_split_whole.fttw", "build/test_split_corrupt.fttw"),
              "and the artifact CHANGED -- so the last part really is being read");
    }
}

/**
 * Every way a part can be missing, short or lying, and the refusal for each.
 *
 * Each case rewrites the whole set first, so a failure in one cannot leave a
 * broken file behind and make the next one pass for the wrong reason.
 */
static void test_incomplete_split_is_refused(void)
{
    const char *stem = "build/test_broken";
    char part_path[SPLIT_PARTS][512];
    ft_gguf_map *map = NULL;

    section("an incomplete split checkpoint is refused, not half-converted");

    for (uint32_t i = 0; i < SPLIT_PARTS; i++)
        split_part_path(part_path[i], sizeof(part_path[i]), stem, i, SPLIT_PARTS);

    check(write_split_checkpoint(stem, 0, NULL) == 0, "wrote a good 3-part set");
    check(ft_gguf_map_open(part_path[0], &map) == FT_GGUF_OK, "which opens");
    ft_gguf_map_close(map);
    map = NULL;

    /* 1. A part that is simply not there. */
    check(remove(part_path[1]) == 0, "deleted part 2 of 3");
    check(ft_gguf_map_open(part_path[0], &map) != FT_GGUF_OK, "a missing part is refused at open");
    check(map == NULL, "and nothing is handed back");
    check(ft_tt_convert_gguf(part_path[0], "build/test_broken.fttw", NULL) != FT_TT_CONVERT_OK,
          "so the converter refuses too, rather than writing a two-thirds model");

    /* 2. A part that is there but truncated — the header parses, the tensor
     *    data does not reach the end of the directory's last entry. */
    check(write_split_checkpoint(stem, 0, NULL) == 0, "restored the set");
    {
        long full = file_bytes(part_path[1]);

        check(full > 0, "part 2 has a size");
        check(copy_file_truncated(part_path[1], "build/test_broken_short.gguf", full / 2) == full / 2,
              "made a half-length copy of part 2");
        check(remove(part_path[1]) == 0 && rename("build/test_broken_short.gguf", part_path[1]) == 0,
              "put the truncated part 2 in place");
        check(ft_gguf_map_open(part_path[0], &map) != FT_GGUF_OK, "a truncated part is refused");
        check(map == NULL, "and nothing is handed back");
    }

    /* 3. A part that parses perfectly and belongs to a different model. This is
     *    the case filename parsing cannot see, and the reason split.* is read
     *    from inside the files at all. */
    {
        synthetic_split stale;

        stale.part_count = SPLIT_PARTS;
        stale.part_index = 1;
        stale.declared_total = SYNTHETIC_COUNT + 4u; /* an older, larger build */
        check(write_split_checkpoint(stem, 1, &stale) == 0, "wrote a set whose part 2 came from another model");
        check(ft_gguf_map_open(part_path[0], &map) != FT_GGUF_OK,
              "a part that disagrees about split.tensors.count is refused");
        check(map == NULL, "and nothing is handed back");
    }

    /* 4. A part in the wrong place: right name, wrong split.no. */
    {
        synthetic_split misplaced;

        misplaced.part_count = SPLIT_PARTS;
        misplaced.part_index = 2; /* claims to be part 3 while named part 2 */
        misplaced.declared_total = SYNTHETIC_COUNT;
        check(write_split_checkpoint(stem, 1, &misplaced) == 0, "wrote a set whose part 2 claims to be part 3");
        check(ft_gguf_map_open(part_path[0], &map) != FT_GGUF_OK, "a part whose split.no is wrong is refused");
        check(map == NULL, "and nothing is handed back");
    }

    /* 5. A part that disagrees about how many parts there are. */
    {
        synthetic_split disagrees;

        disagrees.part_count = SPLIT_PARTS + 1u;
        disagrees.part_index = 1;
        disagrees.declared_total = SYNTHETIC_COUNT;
        check(write_split_checkpoint(stem, 1, &disagrees) == 0, "wrote a set whose part 2 says there are 4 parts");
        check(ft_gguf_map_open(part_path[0], &map) != FT_GGUF_OK, "a part that disagrees about split.count is refused");
        check(map == NULL, "and nothing is handed back");
    }

    /* 6. A part renamed out of the convention. The metadata still says it is
     *    one of three, and there is then no way to find the other two. */
    {
        check(write_split_checkpoint(stem, 0, NULL) == 0, "restored the set");
        check(copy_file_truncated(part_path[0], "build/test_broken_renamed.gguf",
                                  file_bytes(part_path[0])) == file_bytes(part_path[0]),
              "copied part 1 to a name outside the -00001-of-00003.gguf convention");
        check(ft_gguf_map_open("build/test_broken_renamed.gguf", &map) != FT_GGUF_OK,
              "a split part whose name cannot locate its siblings is refused");
        check(map == NULL, "and nothing is handed back");
    }

    /* And the whole set, restored, still opens — so every refusal above was
     * about the fault introduced and not about the set being unopenable. */
    check(write_split_checkpoint(stem, 0, NULL) == 0, "restored the set one last time");
    check(ft_gguf_map_open(part_path[0], &map) == FT_GGUF_OK, "the repaired set opens again");
    check_u64(ft_gguf_map_tensor_count(map), SYNTHETIC_COUNT, "with every tensor");
    ft_gguf_map_close(map);
}

/* ------------------------------------------------------------------------- */
/* An unclassified tensor is REFUSED, not laid out on a guess                 */
/* ------------------------------------------------------------------------- */

/**
 * The red proof for the role classifier, and it is the whole point of having one.
 *
 * A tensor whose role no rule recognises has no layout, because BOTH layouts are
 * a guess about what a kernel will do with it and a wrong guess is invisible:
 * every index stays in range and every answer is wrong. `ssm_out` — a real
 * 6144x5120 matmul weight — was shipped that way. So the converter refuses.
 *
 * **What makes this a red proof rather than a green conversion:** the assertion
 * is that a KNOWN-BAD input FAILS. Delete the FT_TT_CONVERT_EUNCLASSIFIED branch
 * from ft_tt_convert_layout_for_checked() — restore the old
 * `default: FT_TT_LAYOUT_SOURCE` — and this test goes red while every other test
 * in the file stays green, because the old behaviour is exactly "convert it
 * quietly in source order".
 */
/** A progress sink for the paths that require one but whose report is not the subject. */
static int ignore_report(const ft_tt_convert_tensor_report *report, void *user)
{
    (void)report;
    (void)user;
    return 0;
}

static void test_unclassified_tensor_is_refused(void)
{
    static const synthetic_tensor MYSTERY[] = {
        {"blk.0.attn_norm.weight", 1, {64, 1}, FT_GGML_TYPE_F32},
        /* Matches no rule. It is 2-D and 64x64, so it is indistinguishable in
         * shape from a weight the card would read N-major. */
        {"blk.0.mystery_proj.weight", 2, {64, 64}, FT_GGML_TYPE_F32},
    };
    static const synthetic_split unsplit = {0, 0, 0};
    const char *gguf_path = "build/test_unclassified.gguf";
    const char *artifact_path = "build/test_unclassified.fttw";
    ft_tt_convert_options options;
    ft_tt_tensor_layout layout = FT_TT_LAYOUT_MATMUL_NMAJOR;

    section("a tensor with no role is refused, not laid out on a guess");

    check(ft_tt_ledger_role("blk.0.mystery_proj.weight") == FT_TT_ROLE_UNKNOWN,
          "the classifier really has no opinion about it");
    check(ft_tt_convert_layout_for_checked(FT_TT_ROLE_UNKNOWN, 2, &layout) == FT_TT_CONVERT_EUNCLASSIFIED,
          "so no layout can be chosen for it");
    check(layout == FT_TT_LAYOUT_MATMUL_NMAJOR,
          "and the caller's variable is left alone rather than filled with a default");
    /* 1-D is not a guess: a norm or a bias is untiled and row-major whatever it
     * is called, so the refusal is scoped to the case that can be wrong. */
    check(ft_tt_convert_layout_for_checked(FT_TT_ROLE_UNKNOWN, 1, &layout) == FT_TT_CONVERT_OK &&
              layout == FT_TT_LAYOUT_SOURCE,
          "a 1-D tensor of unknown role is still fine — it has no layout to get wrong");

    check(write_gguf(gguf_path, MYSTERY, 2, 0, &unsplit, NULL) == 0,
          "wrote a checkpoint containing one unclassifiable 2-D tensor");

    memset(&options, 0, sizeof(options));
    options.mode = FT_TT_LEDGER_PRODUCTION;
    remove(artifact_path);
    check(ft_tt_convert_gguf(gguf_path, artifact_path, &options) == FT_TT_CONVERT_EUNCLASSIFIED,
          "the CONVERTER refuses it, with a status that names the reason");

    /* And it refuses in the measure path too, which is the one that would
     * otherwise report a PCC for a grouping nobody stores. */
    check(ft_tt_convert_measure(gguf_path, FT_TT_FORMAT_BFP8_B, ignore_report, NULL) ==
              FT_TT_CONVERT_EUNCLASSIFIED,
          "and so does the measure-only path");

    /* The other half of the same proof, on the tensor this actually happened to.
     * `ssm_out` has a rule now, so it converts — and it converts N-MAJOR, which
     * is what it was NOT doing when it had none. Delete the `ssm_out` line from
     * ssm_role() and this half goes red as well: the conversion STOPS instead of
     * quietly writing a matmul weight in source order. */
    {
        static const synthetic_tensor GDN[] = {
            {"blk.0.ssm_norm.weight", 1, {64, 1}, FT_GGML_TYPE_F32},
            {"blk.0.ssm_out.weight", 2, {96, 64}, FT_GGML_TYPE_F32},
            {"blk.0.attn_qkv.weight", 2, {64, 96}, FT_GGML_TYPE_F32},
        };
        const char *gdn_gguf = "build/test_gdn.gguf";
        const char *gdn_fttw = "build/test_gdn.fttw";
        ft_tt_artifact_entry entry;

        check(write_gguf(gdn_gguf, GDN, 3, 0, &unsplit, NULL) == 0, "wrote a small GDN checkpoint");
        check(ft_tt_convert_gguf(gdn_gguf, gdn_fttw, &options) == FT_TT_CONVERT_OK,
              "it converts, because every one of its tensors now has a rule");
        check(read_artifact_entry(gdn_fttw, "blk.0.ssm_out.weight", &entry), "found ssm_out");
        check_u64(entry.layout, FT_TT_LAYOUT_MATMUL_NMAJOR,
                  "ssm_out is stored N-MAJOR — the layout it was silently NOT getting");
        check_u64(entry.role, FT_TT_ROLE_SSM_OUT, "and the artifact records the role that decided it");
        /* 64 out-features of 96 in-features, so the stored grid is Kt=3 by Nt=2:
         * the swap of GGUF's, which a tensor left in source order would not show. */
        check_u64(entry.tile_rows, 3, "its stored grid is Kt = ceil(96/32) rows");
        check_u64(entry.tile_cols, 2, "by Nt = ceil(64/32) columns");
        check(read_artifact_entry(gdn_fttw, "blk.0.attn_qkv.weight", &entry), "found attn_qkv");
        check_u64(entry.layout, FT_TT_LAYOUT_MATMUL_NMAJOR, "attn_qkv is N-major too");
        check_u64(entry.role, FT_TT_ROLE_SSM_IN, "and is recorded as a GDN in-projection");
    }
}

/* ------------------------------------------------------------------------- */
/* The rotary row permutation                                                */
/* ------------------------------------------------------------------------- */

/** 2 heads of 64, one KV head, full-width rope: the smallest shape that is not degenerate. */
#define ROPE_TEST_COLS 32u
#define ROPE_TEST_HEAD 64u
#define ROPE_TEST_Q_ROWS 128u
#define ROPE_TEST_KV_ROWS 64u

static const synthetic_tensor ROPE_MODEL[] = {
    {"blk.0.attn_q.weight", 2, {ROPE_TEST_COLS, ROPE_TEST_Q_ROWS}, FT_GGML_TYPE_F32},
    {"blk.0.attn_k.weight", 2, {ROPE_TEST_COLS, ROPE_TEST_KV_ROWS}, FT_GGML_TYPE_F32},
    /* v carries no rotary features, so it is the control: if it moved, the rule
     * would be matching on `attn_` again rather than on the two tensors that
     * actually hold rotated rows. */
    {"blk.0.attn_v.weight", 2, {ROPE_TEST_COLS, ROPE_TEST_KV_ROWS}, FT_GGML_TYPE_F32},
};

static void write_rope_metadata(FILE *f)
{
    put_string(f, "llama.attention.head_count");
    put_u32(f, 4); /* uint32 */
    put_u32(f, 2);
    put_string(f, "llama.attention.head_count_kv");
    put_u32(f, 4);
    put_u32(f, 1);
    put_string(f, "llama.attention.key_length");
    put_u32(f, 4);
    put_u32(f, ROPE_TEST_HEAD);
    put_string(f, "llama.rope.dimension_count");
    put_u32(f, 4);
    put_u32(f, ROPE_TEST_HEAD);
}

/** The same four keys, with a head geometry no tensor in the model matches. */
static void write_wrong_rope_metadata(FILE *f)
{
    put_string(f, "llama.attention.head_count");
    put_u32(f, 4);
    put_u32(f, 3); /* 128 rows is not 3 heads of 64 */
    put_string(f, "llama.attention.head_count_kv");
    put_u32(f, 4);
    put_u32(f, 1);
    put_string(f, "llama.attention.key_length");
    put_u32(f, 4);
    put_u32(f, ROPE_TEST_HEAD);
    put_string(f, "llama.rope.dimension_count");
    put_u32(f, 4);
    put_u32(f, ROPE_TEST_HEAD);
}

/**
 * One bf16 element of an N-major artifact tensor, read out of the file.
 *
 * `n` is the output feature, `k` the input feature. Stored tile index is
 * `nt * Kt + kt`, and inside a tile the element is at face-ordered position
 * (row = k % 32, column = n % 32) — the card's own indexing, so a test written
 * against it fails if any half of the layout moves.
 */
static float read_nmajor_bf16(const char *path, const ft_tt_artifact_entry *entry, uint32_t n, uint32_t k)
{
    uint32_t kt = entry->tile_rows;
    uint64_t tile = (uint64_t)(n / FT_TT_TILE_WIDTH) * kt + (k / FT_TT_TILE_HEIGHT);
    uint64_t element = tile * FT_TT_TILE_VALUES +
                       ft_tt_tile_face_index(k % FT_TT_TILE_HEIGHT, n % FT_TT_TILE_WIDTH);
    uint16_t narrow = 0;
    FILE *f = fopen(path, "rb");

    if (!f)
        return 0.0f;
    if (fseek(f, (long)(entry->data_offset + element * sizeof(narrow)), SEEK_SET) != 0 ||
        fread(&narrow, sizeof(narrow), 1, f) != 1)
        narrow = 0;
    fclose(f);
    return ft_float_from_bf16(narrow);
}

/** Convert ROPE_MODEL at the bf16 rung under one pairing. */
static int convert_rope_model(const char *gguf_path, const char *artifact_path, ft_tt_rope_pairing pairing)
{
    ft_tt_convert_options options;

    memset(&options, 0, sizeof(options));
    options.mode = FT_TT_LEDGER_BF16;
    options.rope_pairing = pairing;
    return ft_tt_convert_gguf(gguf_path, artifact_path, &options);
}

/**
 * The permutation is a relabelling of output rows and nothing else.
 *
 * Pinned at the **bf16 rung on purpose**: bf16 has no shared exponent, so the
 * permuted artifact must be the unpermuted one's rows in a different order,
 * BIT FOR BIT. At a block-float rung the same permutation regroups which 16
 * output features share an exponent, so the values move by a quantization step
 * — real, small, and not what this test is for. Testing the claim at the rung
 * where it is exact is what makes a failure mean something.
 */
static void test_rope_permutation(void)
{
    static const synthetic_split unsplit = {0, 0, 0};
    static const synthetic_extra_metadata rope_metadata = {write_rope_metadata, 4};
    static const synthetic_extra_metadata wrong_metadata = {write_wrong_rope_metadata, 4};
    const char *gguf_path = "build/test_rope.gguf";
    const char *plain_path = "build/test_rope_source.fttw";
    const char *permuted_path = "build/test_rope_split.fttw";
    ft_tt_artifact_entry plain, permuted;
    uint32_t half = ROPE_TEST_HEAD / 2u;
    uint32_t mismatches = 0;

    section("the rotary permutation, at the rung where it is exact");

    check(write_gguf(gguf_path, ROPE_MODEL, 3, 0, &unsplit, &rope_metadata) == 0,
          "wrote a checkpoint with a declared head geometry");
    check(convert_rope_model(gguf_path, plain_path, FT_TT_ROPE_PAIRING_SOURCE) == FT_TT_CONVERT_OK,
          "converted it with the source pairing");
    check(convert_rope_model(gguf_path, permuted_path, FT_TT_ROPE_PAIRING_SPLIT_HALF) == FT_TT_CONVERT_OK,
          "and again with split-half");

    check(read_artifact_entry(plain_path, "blk.0.attn_q.weight", &plain) &&
              read_artifact_entry(permuted_path, "blk.0.attn_q.weight", &permuted),
          "found attn_q in both artifacts");
    check(plain.byte_count == permuted.byte_count,
          "a permutation moves rows, so the two artifacts are the same size");
    check_u64(permuted.layout, FT_TT_LAYOUT_MATMUL_NMAJOR, "and attn_q is still N-major");

    /* Every output row of every head, against every input feature. 128 x 32 is
     * small enough to check exhaustively, and exhaustive is what rules out a
     * permutation that happens to be right on the rows a spot check picks. */
    for (uint32_t n = 0; n < ROPE_TEST_Q_ROWS; n++) {
        uint32_t head = n / ROPE_TEST_HEAD;
        uint32_t within = n % ROPE_TEST_HEAD;
        uint32_t source = head * ROPE_TEST_HEAD +
                          (within < half ? within * 2u : (within - half) * 2u + 1u);

        for (uint32_t k = 0; k < ROPE_TEST_COLS; k++) {
            float got = read_nmajor_bf16(permuted_path, &permuted, n, k);
            float want = read_nmajor_bf16(plain_path, &plain, source, k);

            if (bits_from_float(got) != bits_from_float(want))
                mismatches++;
        }
    }
    check_u64(mismatches, 0, "every attn_q element is the split-half source row, BIT for bit");

    /* The permutation is not the identity — otherwise the check above passes for
     * a converter that ignored the option entirely. */
    {
        uint32_t moved = 0;

        for (uint32_t n = 0; n < ROPE_TEST_Q_ROWS; n++)
            for (uint32_t k = 0; k < ROPE_TEST_COLS; k++)
                if (bits_from_float(read_nmajor_bf16(permuted_path, &permuted, n, k)) !=
                    bits_from_float(read_nmajor_bf16(plain_path, &plain, n, k)))
                    moved++;
        printf("        attn_q: %u of %u elements sit somewhere new\n", moved,
               ROPE_TEST_Q_ROWS * ROPE_TEST_COLS);
        check(moved > ROPE_TEST_Q_ROWS * ROPE_TEST_COLS / 2u,
              "and most of the tensor really did move — the option is not a no-op");
    }

    /* attn_k is permuted with the KV head count, not the Q head count. With 1 KV
     * head of 64 rows the two counts give different answers, which is what makes
     * this check able to see the bug. */
    {
        ft_tt_artifact_entry k_plain, k_permuted;
        uint32_t k_mismatches = 0;

        check(read_artifact_entry(plain_path, "blk.0.attn_k.weight", &k_plain) &&
                  read_artifact_entry(permuted_path, "blk.0.attn_k.weight", &k_permuted),
              "found attn_k in both");
        for (uint32_t n = 0; n < ROPE_TEST_KV_ROWS; n++) {
            uint32_t source = n < half ? n * 2u : (n - half) * 2u + 1u;

            for (uint32_t k = 0; k < ROPE_TEST_COLS; k++)
                if (bits_from_float(read_nmajor_bf16(permuted_path, &k_permuted, n, k)) !=
                    bits_from_float(read_nmajor_bf16(plain_path, &k_plain, source, k)))
                    k_mismatches++;
        }
        check_u64(k_mismatches, 0, "attn_k is permuted under the KV head count");
    }

    /* attn_v holds no rotary features and must be untouched. */
    {
        ft_tt_artifact_entry v_plain, v_permuted;
        uint32_t v_mismatches = 0;

        check(read_artifact_entry(plain_path, "blk.0.attn_v.weight", &v_plain) &&
                  read_artifact_entry(permuted_path, "blk.0.attn_v.weight", &v_permuted),
              "found attn_v in both");
        for (uint32_t n = 0; n < ROPE_TEST_KV_ROWS; n++)
            for (uint32_t k = 0; k < ROPE_TEST_COLS; k++)
                if (bits_from_float(read_nmajor_bf16(permuted_path, &v_permuted, n, k)) !=
                    bits_from_float(read_nmajor_bf16(plain_path, &v_plain, n, k)))
                    v_mismatches++;
        check_u64(v_mismatches, 0, "attn_v is NOT permuted — it carries no rotary features");
    }

    section("the artifact says which pairing it holds");

    {
        unsigned char header[FT_TT_ARTIFACT_HEADER_BYTES];
        FILE *f = fopen(permuted_path, "rb");
        uint32_t flags = 0;

        if (f && fread(header, 1, sizeof(header), f) == sizeof(header))
            memcpy(&flags, header + 12, sizeof(flags));
        if (f)
            fclose(f);
        check((flags & FT_TT_ARTIFACT_FLAG_ROPE_SPLIT_HALF) != 0,
              "the permuted artifact sets FT_TT_ARTIFACT_FLAG_ROPE_SPLIT_HALF");

        flags = 0;
        f = fopen(plain_path, "rb");
        if (f && fread(header, 1, sizeof(header), f) == sizeof(header))
            memcpy(&flags, header + 12, sizeof(flags));
        if (f)
            fclose(f);
        check(flags == 0u, "and the unpermuted one does not — the two files are otherwise identical");
    }

    section("a geometry the permutation cannot express is refused");

    {
        const char *bad_path = "build/test_rope_bad.gguf";
        const char *out_path = "build/test_rope_bad.fttw";

        check(write_gguf(bad_path, ROPE_MODEL, 3, 0, &unsplit, &wrong_metadata) == 0,
              "wrote a checkpoint whose declared head count does not divide attn_q");
        check(convert_rope_model(bad_path, out_path, FT_TT_ROPE_PAIRING_SPLIT_HALF) ==
                  FT_TT_CONVERT_EROPEGEOMETRY,
              "permuting rows whose meaning is not established is refused");
        check(convert_rope_model(bad_path, out_path, FT_TT_ROPE_PAIRING_SOURCE) == FT_TT_CONVERT_OK,
              "and the same checkpoint converts fine when nothing is being permuted");
    }

    {
        /* No head geometry at all. Deriving one from a tensor's own row count
         * would be a guess, so it refuses instead. */
        const char *bare_path = "build/test_rope_bare.gguf";
        const char *out_path = "build/test_rope_nometa.fttw";

        check(write_gguf(bare_path, ROPE_MODEL, 3, 0, &unsplit, NULL) == 0,
              "wrote the same tensors with no head geometry declared at all");
        check(convert_rope_model(bare_path, out_path, FT_TT_ROPE_PAIRING_SPLIT_HALF) ==
                  FT_TT_CONVERT_EROPEGEOMETRY,
              "a checkpoint that declares no head geometry is refused, not guessed at");
    }
}

/**
 * `--rope-pairing split-half-reversed`: the same relabelling drawn a THIRD way.
 *
 * Two claims are load-bearing and each gets its own stage, because a test that only asserted
 * "the bytes moved" would pass for any permutation at all.
 *
 *  1. **It is split-half composed with a reflection of the pair index.** Checked element by
 *     element against the SPLIT-HALF artifact — not against the source — so what is measured
 *     is the reflection alone rather than the split-half undo a second time. At the bf16 rung,
 *     where the permutation is exact and a mismatch is a bug and not a quantization step.
 *  2. **A partner is still exactly `half` rows away.** That is the property the on-card
 *     kernel's tile argument rests on AND the property that makes "apply frequency
 *     `half-1-p` at slot `p`" a complete contract. It is an INDEX claim, so it is checked
 *     over every slot rather than spot-checked.
 *
 * Plus: it must DIFFER from plain split-half. Without that check a converter that ignored the
 * new pairing entirely would satisfy claim 2 and read green.
 */
static void test_rope_pairs_reversed(void)
{
    static const synthetic_split unsplit = {0, 0, 0};
    static const synthetic_extra_metadata rope_metadata = {write_rope_metadata, 4};
    const char *gguf_path = "build/test_rope_rev.gguf";
    const char *split_path = "build/test_rope_rev_split.fttw";
    const char *reversed_path = "build/test_rope_rev_reversed.fttw";
    ft_tt_artifact_entry split, reversed;
    uint32_t half = ROPE_TEST_HEAD / 2u;
    uint32_t mismatches = 0;

    section("the reversed rotary pairing — a third draw over the same two tensors");

    check(write_gguf(gguf_path, ROPE_MODEL, 3, 0, &unsplit, &rope_metadata) == 0,
          "wrote a checkpoint with a declared head geometry");
    check(convert_rope_model(gguf_path, split_path, FT_TT_ROPE_PAIRING_SPLIT_HALF) == FT_TT_CONVERT_OK,
          "converted it with split-half");
    check(convert_rope_model(gguf_path, reversed_path, FT_TT_ROPE_PAIRING_SPLIT_HALF_REVERSED) ==
              FT_TT_CONVERT_OK,
          "and again with split-half-reversed");

    check(read_artifact_entry(split_path, "blk.0.attn_q.weight", &split) &&
              read_artifact_entry(reversed_path, "blk.0.attn_q.weight", &reversed),
          "found attn_q in both artifacts");
    check(split.byte_count == reversed.byte_count,
          "a permutation moves rows, so the two artifacts are the same size");

    for (uint32_t n = 0; n < ROPE_TEST_Q_ROWS; n++) {
        uint32_t head = n / ROPE_TEST_HEAD;
        uint32_t within = n % ROPE_TEST_HEAD;
        uint32_t upper = within / half;
        uint32_t pair = within % half;
        uint32_t from = head * ROPE_TEST_HEAD + upper * half + (half - 1u - pair);

        for (uint32_t k = 0; k < ROPE_TEST_COLS; k++)
            if (bits_from_float(read_nmajor_bf16(reversed_path, &reversed, n, k)) !=
                bits_from_float(read_nmajor_bf16(split_path, &split, from, k)))
                mismatches++;
    }
    check_u64(mismatches, 0, "reversed slot p holds split-half's slot half-1-p, BIT for bit");

    /* Claim 2, read off the artifact rather than off the formula: the row the LOWER half of
     * a slot's pair came from and the row its UPPER half came from must be `half` apart in
     * the split-half artifact, for every slot. Checked by finding each reversed row's source
     * by content, so a formula that agrees with itself cannot carry it. */
    {
        uint32_t broken = 0;

        for (uint32_t p = 0; p < half; p++) {
            uint32_t lower_from = ROPE_TEST_Q_ROWS, upper_from = ROPE_TEST_Q_ROWS;

            for (uint32_t s = 0; s < ROPE_TEST_HEAD; s++) {
                uint32_t same_lower = 1, same_upper = 1;

                for (uint32_t k = 0; k < ROPE_TEST_COLS; k++) {
                    uint32_t source = bits_from_float(read_nmajor_bf16(split_path, &split, s, k));

                    if (source != bits_from_float(read_nmajor_bf16(reversed_path, &reversed, p, k)))
                        same_lower = 0;
                    if (source != bits_from_float(read_nmajor_bf16(reversed_path, &reversed, p + half, k)))
                        same_upper = 0;
                }
                if (same_lower)
                    lower_from = s;
                if (same_upper)
                    upper_from = s;
            }
            if (lower_from >= ROPE_TEST_HEAD || upper_from >= ROPE_TEST_HEAD ||
                upper_from - lower_from != half)
                broken++;
        }
        check_u64(broken, 0, "every reversed slot keeps its partner exactly `half` rows away");
    }

    /* And it is not plain split-half. `half - 1 - p == p` has no solution for an even `half`,
     * so EVERY rotary row must move, and rope_dim == head_dim here so there is no tail. */
    {
        uint32_t moved_rows = 0;

        for (uint32_t n = 0; n < ROPE_TEST_Q_ROWS; n++) {
            uint32_t differs = 0;

            for (uint32_t k = 0; k < ROPE_TEST_COLS; k++)
                if (bits_from_float(read_nmajor_bf16(reversed_path, &reversed, n, k)) !=
                    bits_from_float(read_nmajor_bf16(split_path, &split, n, k)))
                    differs = 1;
            moved_rows += differs;
        }
        printf("        attn_q: %u of %u rows differ from the split-half artifact\n", moved_rows,
               ROPE_TEST_Q_ROWS);
        check_u64(moved_rows, ROPE_TEST_Q_ROWS,
                  "every rotary row differs from split-half — the reversal is not a no-op");
    }

    {
        ft_tt_artifact_entry v_split, v_reversed;
        uint32_t v_mismatches = 0;

        check(read_artifact_entry(split_path, "blk.0.attn_v.weight", &v_split) &&
                  read_artifact_entry(reversed_path, "blk.0.attn_v.weight", &v_reversed),
              "found attn_v in both");
        for (uint32_t n = 0; n < ROPE_TEST_KV_ROWS; n++)
            for (uint32_t k = 0; k < ROPE_TEST_COLS; k++)
                if (bits_from_float(read_nmajor_bf16(reversed_path, &v_reversed, n, k)) !=
                    bits_from_float(read_nmajor_bf16(split_path, &v_split, n, k)))
                    v_mismatches++;
        check_u64(v_mismatches, 0, "attn_v is identical under both pairings — it holds no rotary rows");
    }

    section("the crossed control: reflected, but the partner stays ADJACENT");

    /* `interleaved-reversed` is the arm that separates the two things split-half changed at
     * once. It must be GGUF's own order with the pair index reflected — so slot (p, c) holds
     * GGUF row 2*(half-1-p) + c — and it must NOT claim split-half shape. */
    {
        const char *crossed_path = "build/test_rope_rev_crossed.fttw";
        ft_tt_artifact_entry source, crossed;
        const char *source_path = "build/test_rope_rev_source.fttw";
        uint32_t crossed_mismatches = 0;

        check(convert_rope_model(gguf_path, source_path, FT_TT_ROPE_PAIRING_SOURCE) == FT_TT_CONVERT_OK,
              "converted the same checkpoint with the source pairing");
        check(convert_rope_model(gguf_path, crossed_path, FT_TT_ROPE_PAIRING_INTERLEAVED_REVERSED) ==
                  FT_TT_CONVERT_OK,
              "and again with interleaved-reversed");
        check(read_artifact_entry(source_path, "blk.0.attn_q.weight", &source) &&
                  read_artifact_entry(crossed_path, "blk.0.attn_q.weight", &crossed),
              "found attn_q in both");

        for (uint32_t n = 0; n < ROPE_TEST_Q_ROWS; n++) {
            uint32_t head = n / ROPE_TEST_HEAD;
            uint32_t within = n % ROPE_TEST_HEAD;
            uint32_t from = head * ROPE_TEST_HEAD + (half - 1u - within / 2u) * 2u + within % 2u;

            for (uint32_t k = 0; k < ROPE_TEST_COLS; k++)
                if (bits_from_float(read_nmajor_bf16(crossed_path, &crossed, n, k)) !=
                    bits_from_float(read_nmajor_bf16(source_path, &source, from, k)))
                    crossed_mismatches++;
        }
        check_u64(crossed_mismatches, 0,
                  "interleaved-reversed slot (p,c) is GGUF row 2*(half-1-p)+c, BIT for bit");

        /* THE PROPERTY THE CONTROL EXISTS FOR: the partner is ADJACENT, not +half. Odd slot
         * `2p+1` must come from the row exactly ONE past the row slot `2p` came from. */
        {
            uint32_t broken = 0;

            for (uint32_t p = 0; p < half; p++) {
                uint32_t even_from = (half - 1u - p) * 2u;
                uint32_t odd_from = (half - 1u - p) * 2u + 1u;

                if (odd_from - even_from != 1u)
                    broken++;
            }
            check_u64(broken, 0, "and its partner stays ADJACENT — this arm keeps a pair in ONE "
                                 "block-float exponent group, which is the variable under test");
        }
    }

    section("the four pairings are the four combinations of two independent bits");

    {
        unsigned char header[FT_TT_ARTIFACT_HEADER_BYTES];
        const char *crossed_path = "build/test_rope_rev_crossed.fttw";
        const char *source_path = "build/test_rope_rev_source.fttw";
        struct {
            const char *path;
            uint32_t want;
            const char *what;
        } expect[4] = {{source_path, 0x0u, "source: neither bit"},
                       {split_path, 0x1u, "split-half: SPLIT_HALF only"},
                       {crossed_path, 0x2u, "interleaved-reversed: PAIRS_REVERSED only"},
                       {reversed_path, 0x3u, "split-half-reversed: both"}};
        unsigned index;

        for (index = 0; index < 4u; index++) {
            FILE *f = fopen(expect[index].path, "rb");
            uint32_t flags = 0xFFFFFFFFu;

            if (f && fread(header, 1, sizeof(header), f) == sizeof(header))
                memcpy(&flags, header + 12, sizeof(flags));
            if (f)
                fclose(f);
            check_u64(flags, expect[index].want, expect[index].what);
        }
    }
}

/* ------------------------------------------------------------------------- */
/* A tied checkpoint gets its LM head                                         */
/* ------------------------------------------------------------------------- */

/** One artifact entry's payload bytes, read through its directory offset. */
static size_t read_entry_payload(const char *artifact, const ft_tt_artifact_entry *entry,
                                 unsigned char *out, size_t out_bytes)
{
    FILE *f = fopen(artifact, "rb");
    size_t got = 0;

    if (!f)
        return 0;
    if (entry->byte_count <= out_bytes && fseek(f, (long)entry->data_offset, SEEK_SET) == 0)
        got = fread(out, 1, (size_t)entry->byte_count, f);
    fclose(f);
    return got;
}

static void test_tied_head(void)
{
    /* A: token_embd and no output.weight (tied). B: an untied head holding the SAME values
     * (seed 0 in both), so B's head bytes are what A's derived head must be. */
    static const synthetic_tensor TIED[] = {
        {"token_embd.weight", 2, {64, 100}, FT_GGML_TYPE_Q8_0},
        {"blk.0.attn_q.weight", 2, {64, 64}, FT_GGML_TYPE_Q8_0},
    };
    static const synthetic_tensor UNTIED[] = {
        {"output.weight", 2, {64, 100}, FT_GGML_TYPE_Q8_0},
    };
    static const synthetic_split unsplit = {0, 0, 0};
    const char *tied_gguf = "build/tied.gguf", *tied_art = "build/tied.fttw";
    const char *head_gguf = "build/untied.gguf", *head_art = "build/untied.fttw";
    static unsigned char a[64 * 1024], b[64 * 1024];
    ft_tt_artifact_entry head, embd, ref;
    ft_tt_artifact_header header;
    ft_gguf_map *map = NULL;
    size_t na, nb;
    FILE *f;

    section("a tied checkpoint gets an LM head, converted from token_embd");

    check(write_gguf(tied_gguf, TIED, 2, 0, &unsplit, NULL) == 0, "wrote the tied checkpoint");
    check(write_gguf(head_gguf, UNTIED, 1, 0, &unsplit, NULL) == 0, "wrote the untied head");
    check(ft_gguf_map_open(tied_gguf, &map) == FT_GGUF_OK, "mapped the tied checkpoint");
    check(map && ft_tt_convert_tied_head_source(map) != NULL, "it is recognised as tied");
    ft_gguf_map_close(map);
    map = NULL;
    check(ft_gguf_map_open(head_gguf, &map) == FT_GGUF_OK, "mapped the untied checkpoint");
    check(map && ft_tt_convert_tied_head_source(map) == NULL, "an output.weight means untied");
    ft_gguf_map_close(map);

    check(ft_tt_convert_gguf(tied_gguf, tied_art, NULL) == FT_TT_CONVERT_OK, "tied converts");
    check(ft_tt_convert_gguf(head_gguf, head_art, NULL) == FT_TT_CONVERT_OK, "untied converts");
    f = fopen(tied_art, "rb");
    check(f && fread(&header, sizeof(header), 1, f) == 1, "tied header reads");
    if (f)
        fclose(f);
    check_u64(header.tensor_count, 3u, "the artifact holds the two tensors plus the head");

    check(read_artifact_entry(tied_art, "output.weight", &head), "the head is in the directory");
    check(read_artifact_entry(tied_art, "token_embd.weight", &embd), "the embedding stays");
    check(read_artifact_entry(head_art, "output.weight", &ref), "the reference head exists");
    check(head.role == FT_TT_ROLE_LM_HEAD && head.layout == FT_TT_LAYOUT_MATMUL_NMAJOR,
          "the head is an N-major LM head");
    check(embd.layout == FT_TT_LAYOUT_SOURCE, "the embedding keeps its row-lookup layout");
    check(head.format == ref.format && head.tile_rows == ref.tile_rows &&
              head.tile_cols == ref.tile_cols,
          "the head's format and grid are an untied head's");
    na = read_entry_payload(tied_art, &head, a, sizeof(a));
    nb = read_entry_payload(head_art, &ref, b, sizeof(b));
    check(na > 0 && na == nb && memcmp(a, b, na) == 0,
          "the head's bytes are the untied head's, byte for byte");
}

int main(void)
{
    printf("test_convert — GGUF reader, precision ledger and artifact writer\n");

    test_q8_0_is_exact_in_fp32();
    test_q8_0_block_decode();
    test_fp16_conversion();
    test_bf16_source_is_exact();
    test_bf16_packs_bfp8_from_its_own_values();
    test_kquant_readers_match_ggml();
    test_kquant_vector_provenance_is_declared();
    test_kquant_sizing_and_partial_blocks();
    test_kquant_checkpoint_maps_and_converts();
    test_ledger();
    test_hybrid_roles();
    test_uniform_rungs();
    test_gguf_and_convert();
    test_unclassified_tensor_is_refused();
    test_rope_permutation();
    test_rope_pairs_reversed();
    test_unreadable_formats_are_named_and_refused();
    test_split_checkpoint();
    test_incomplete_split_is_refused();
    test_tied_head();

    printf("\n%d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}
