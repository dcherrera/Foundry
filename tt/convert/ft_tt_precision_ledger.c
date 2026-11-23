/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_precision_ledger.c
 * @brief Implementation of the tensor-role classifier and the precision ladder.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "ft_tt_precision_ledger.h"

#include <string.h>

static int name_contains(const char *name, const char *needle)
{
    return strstr(name, needle) != NULL;
}

/**
 * Classify the GDN / SSM half of a hybrid model, or return UNKNOWN.
 *
 * Split out from ft_tt_ledger_role() for one reason that is not cosmetic: TWO of
 * these names begin with `attn_`, so this MUST run before the real-attention
 * rule, and a reader has to be able to see that ordering at the call site. See
 * the header for what each tensor is and why `attn_qkv`/`attn_gate` are not
 * attention.
 */
static ft_tt_tensor_role ssm_role(const char *name)
{
    /* Longest-first inside each family: `attn_qkv` would otherwise be caught by
     * the `attn_q` rule below, and `ssm_conv1d` shares no prefix but is listed
     * with its siblings so the whole GDN set reads in one place. */
    if (name_contains(name, "attn_qkv") || name_contains(name, "attn_gate"))
        return FT_TT_ROLE_SSM_IN;
    if (name_contains(name, "ssm_out"))
        return FT_TT_ROLE_SSM_OUT;
    if (name_contains(name, "ssm_alpha") || name_contains(name, "ssm_beta"))
        return FT_TT_ROLE_SSM_GATE;
    if (name_contains(name, "ssm_conv1d"))
        return FT_TT_ROLE_SSM_CONV;
    if (name_contains(name, "ssm_a") || name_contains(name, "ssm_dt"))
        return FT_TT_ROLE_SSM_DECAY;
    return FT_TT_ROLE_UNKNOWN;
}

/**
 * Real softmax attention, and ONLY real softmax attention.
 *
 * The Qwen bf16 exception hangs off this, so its scope is the difference between
 * a 27.06 GiB artifact and a 28.14 GiB one. `attn_q` is tested after `attn_qkv`
 * has already been taken by ssm_role(), because `attn_q` is a prefix of it.
 */
static ft_tt_tensor_role attention_role(const char *name)
{
    if (name_contains(name, "attn_q") || name_contains(name, "attn_k") ||
        name_contains(name, "attn_v") || name_contains(name, "attn_output"))
        return FT_TT_ROLE_ATTENTION;
    return FT_TT_ROLE_UNKNOWN;
}

ft_tt_tensor_role ft_tt_ledger_role(const char *tensor_name)
{
    ft_tt_tensor_role role;

    if (!tensor_name)
        return FT_TT_ROLE_UNKNOWN;

    /* Bias before everything else: a bias inherits its parent's name, so
     * `blk.0.attn_q.bias` must not be classified as attention. */
    if (name_contains(tensor_name, ".bias"))
        return FT_TT_ROLE_BIAS;
    /* `enorm`/`hnorm` are the MTP block's two input norms. They matched neither
     * `_norm` nor `.norm` — a dot followed by `norm` is not what `.enorm` is —
     * so they were UNKNOWN, and only their being 1-D kept that harmless. */
    if (name_contains(tensor_name, "_norm") || name_contains(tensor_name, ".norm") ||
        name_contains(tensor_name, "enorm") || name_contains(tensor_name, "hnorm"))
        return FT_TT_ROLE_NORM;

    if (name_contains(tensor_name, "token_embd"))
        return FT_TT_ROLE_EMBEDDING;
    if (strcmp(tensor_name, "output.weight") == 0 || name_contains(tensor_name, "lm_head"))
        return FT_TT_ROLE_LM_HEAD;

    role = ssm_role(tensor_name);
    if (role != FT_TT_ROLE_UNKNOWN)
        return role;
    role = attention_role(tensor_name);
    if (role != FT_TT_ROLE_UNKNOWN)
        return role;

    /* Routed experts before every other FFN rule: `ffn_gate_exps` would otherwise be an
     * ordinary FFN_UP, and the ledger could then not give the routed experts a format the
     * shared expert (`ffn_*_shexp`, a plain FFN) does not get. */
    if (name_contains(tensor_name, "_exps")) {
        if (name_contains(tensor_name, "ffn_down"))
            return FT_TT_ROLE_EXPERT_DOWN;
        if (name_contains(tensor_name, "ffn_gate") || name_contains(tensor_name, "ffn_up"))
            return FT_TT_ROLE_EXPERT_UP;
    }
    /* Before `ffn_gate`, which is a PREFIX of it. A MoE router matching the
     * up-projection rule would be quantized as if a routing decision were a
     * numeric output, and it is not: it is an argmax over experts. */
    if (name_contains(tensor_name, "ffn_gate_inp") || name_contains(tensor_name, "router"))
        return FT_TT_ROLE_ROUTER;
    if (name_contains(tensor_name, "ffn_down"))
        return FT_TT_ROLE_FFN_DOWN;
    if (name_contains(tensor_name, "ffn_gate") || name_contains(tensor_name, "ffn_up"))
        return FT_TT_ROLE_FFN_UP;

    if (name_contains(tensor_name, "eh_proj"))
        return FT_TT_ROLE_PROJECTION;

    return FT_TT_ROLE_UNKNOWN;
}

const char *ft_tt_ledger_role_name(ft_tt_tensor_role role)
{
    switch (role) {
    case FT_TT_ROLE_EMBEDDING:
        return "embedding";
    case FT_TT_ROLE_LM_HEAD:
        return "lm_head";
    case FT_TT_ROLE_ATTENTION:
        return "attention";
    case FT_TT_ROLE_FFN_UP:
        return "ffn_gate_up";
    case FT_TT_ROLE_FFN_DOWN:
        return "ffn_down";
    case FT_TT_ROLE_NORM:
        return "norm";
    case FT_TT_ROLE_BIAS:
        return "bias";
    case FT_TT_ROLE_SSM_IN:
        return "ssm_in";
    case FT_TT_ROLE_SSM_OUT:
        return "ssm_out";
    case FT_TT_ROLE_SSM_GATE:
        return "ssm_gate";
    case FT_TT_ROLE_SSM_CONV:
        return "ssm_conv";
    case FT_TT_ROLE_SSM_DECAY:
        return "ssm_decay";
    case FT_TT_ROLE_PROJECTION:
        return "projection";
    case FT_TT_ROLE_ROUTER:
        return "router";
    case FT_TT_ROLE_EXPERT_UP:
        return "expert_gate_up";
    case FT_TT_ROLE_EXPERT_DOWN:
        return "expert_down";
    default:
        return "unknown";
    }
}

const char *ft_tt_ledger_format_name(ft_tt_tensor_format format)
{
    switch (format) {
    case FT_TT_FORMAT_FP32:
        return "fp32";
    case FT_TT_FORMAT_BF16:
        return "bf16";
    case FT_TT_FORMAT_BFP8_B:
        return "bfp8_b";
    case FT_TT_FORMAT_BFP4_B:
        return "bfp4_b";
    case FT_TT_FORMAT_BFP6_B:
        return "bfp6_b";
    default:
        return "?";
    }
}

ft_tt_model_family ft_tt_ledger_family_from_architecture(const char *architecture)
{
    if (architecture && strncmp(architecture, "qwen", 4) == 0)
        return FT_TT_FAMILY_QWEN;
    return FT_TT_FAMILY_LLAMA;
}

ft_tt_tensor_format ft_tt_ledger_format_for(ft_tt_ledger_mode mode, ft_tt_model_family family, ft_tt_tensor_role role,
                                            uint32_t n_dims)
{
    /* Norms and biases stay fp32 on every rung. They are 1-D, they are a
     * rounding error in the artifact, and a shared exponent across 16
     * normalization scales buys nothing. */
    if (n_dims < 2 || role == FT_TT_ROLE_NORM || role == FT_TT_ROLE_BIAS)
        return FT_TT_FORMAT_FP32;

    if (mode == FT_TT_LEDGER_FP32)
        return FT_TT_FORMAT_FP32;
    if (mode == FT_TT_LEDGER_BF16)
        return FT_TT_FORMAT_BF16;
    /* The two uniform rungs are uniform over 2-D tensors, which is the scope the
     * perplexity A/B in the header measured. 1-D norms and biases stay fp32 on
     * every rung and were already returned above. */
    if (mode == FT_TT_LEDGER_BFP8)
        return FT_TT_FORMAT_BFP8_B;
    if (mode == FT_TT_LEDGER_BFP4)
        return FT_TT_FORMAT_BFP4_B;
    /* The MoE rung is the uniform Bfp8_b rung with TWO exceptions: the routed experts
     * (Bfp6_b) and the router (fp32). The router rule is David's (2026-10-27: as precise as
     * possible while everything fits) and D4 (2026-10-28) applied it after qwen36 Phase 4.3
     * measured a Bfp8_b router costing as much KL as all the Bfp6_b experts and flipping 12 %
     * of routing sets on 4 blocks; fp32 costs +62 MB on Qwen3.6. */
    if (mode == FT_TT_LEDGER_MOE_BFP6) {
        if (role == FT_TT_ROLE_EXPERT_UP || role == FT_TT_ROLE_EXPERT_DOWN)
            return FT_TT_FORMAT_BFP6_B;
        return role == FT_TT_ROLE_ROUTER ? FT_TT_FORMAT_FP32 : FT_TT_FORMAT_BFP8_B;
    }

    switch (role) {
    case FT_TT_ROLE_EMBEDDING:
    case FT_TT_ROLE_LM_HEAD:
        /* §6.4 put these at bf16 because the head feeds sampling directly. That
         * reason has now been measured rather than assumed, and it does not hold:
         * a perplexity A/B over 1067 chunks with only these two tensors demoted
         * to BFP8 moved perplexity by **+0.035%** (18.1926 -> 18.1990), against
         * +5.73% for the same two at BFP4 — a control that proves the measurement
         * responds to perturbing exactly these tensors. See
         * docs/tt_bringup_log.md §13.5. On MiniCPM5-1B they are 59% of the
         * artifact, so this is 765 MiB -> 383 MiB and the whole artifact drops
         * from 1292 MiB to ~910 MiB, below the Q8_0 source it came from. */
        return FT_TT_FORMAT_BFP8_B;
    case FT_TT_ROLE_ATTENTION:
        /* §6.4: Qwen is attention-precision-sensitive where Llama is not. The
         * evidence is about SOFTMAX attention on Qwen2.5-7B, so the rule is
         * scoped to the four real attention projections and does not reach a
         * GDN layer's `attn_qkv`/`attn_gate`, which only share the prefix. On
         * Qwen3.8 that scoping is 96 tensors and 7.50 GiB. */
        return family == FT_TT_FAMILY_QWEN ? FT_TT_FORMAT_BF16 : FT_TT_FORMAT_BFP8_B;
    case FT_TT_ROLE_SSM_IN:
    case FT_TT_ROLE_SSM_OUT:
    case FT_TT_ROLE_PROJECTION:
        /* Ordinary matmul weights. Nothing distinguishes them from ffn_down,
         * which the same measurement puts at Bfp8_b. */
        return FT_TT_FORMAT_BFP8_B;
    case FT_TT_ROLE_ROUTER:
        /* A router's output is not consumed as a number, it is consumed as an
         * ARGMAX over experts, so its error mode is discrete: a block-float
         * rounding that reorders two near-tied logits routes a token to the
         * wrong expert, and no averaging softens that. It is also
         * [n_embd x n_expert] — 256 experts of Qwen3.6 is 2.6 MiB at bf16. */
        return FT_TT_FORMAT_BF16;
    case FT_TT_ROLE_SSM_GATE:
        /* `ssm_alpha`/`ssm_beta` are 5120->48, so bf16 across all 48 GDN layers
         * of Qwen3.8 is ~63 MiB and Bfp8_b would save ~30 MiB of a 27 GiB
         * artifact. Nothing has measured a per-head gate projection under block
         * float, and 30 MiB does not buy the risk. */
        return FT_TT_FORMAT_BF16;
    case FT_TT_ROLE_SSM_CONV:
    case FT_TT_ROLE_SSM_DECAY:
        /* A 4-tap depthwise convolution and the per-head decay parameters. Both
         * are tiny and both feed a recurrence, where an error does not wash out
         * over a sequence the way a per-token projection error does. */
        return FT_TT_FORMAT_FP32;
    case FT_TT_ROLE_FFN_UP:
        /* §6.4 put `ffn_gate`/`ffn_up` at BFP4 assuming the FFN is the cheapest
         * place to spend bits. MEASURED, it is the most expensive one: BFP4 here
         * costs **+9.50% perplexity** (MiniCPM5-1B-Q8_0, 2 MiB enwik8 slice,
         * -c 512, 1067 chunks, paired), and it is the named cause of the Phase 7
         * forward's 0.40-0.44 mean|dlogit| — these two tensors score per-op PCC
         * 0.991-0.993 where every other op scores 0.99988+. The FFN is ~2x MORE
         * BFP4-sensitive per byte than embedding+head (dNLL 0.29 vs 0.15), which
         * is the opposite of what §6.4 assumed.
         *
         * It is also close to free. §18.2 measured BFP4 is NOT faster than BFP8
         * (532.6 vs 534.1 GFLOP/s — halving 1088 -> 576 bytes a tile bought
         * nothing), so no throughput is being traded for the bits. The cost is
         * +162.0 MiB of artifact, 933 -> ~1095 MiB, still under the 1292 MiB this
         * artifact was before §13.5 demoted embedding and head. */
        return FT_TT_FORMAT_BFP8_B;
    case FT_TT_ROLE_FFN_DOWN:
    case FT_TT_ROLE_EXPERT_UP:
    case FT_TT_ROLE_EXPERT_DOWN:
        return FT_TT_FORMAT_BFP8_B;
    default:
        /* An unrecognised 2-D tensor is a tensor we have not reasoned about.
         * Give it the safest format rather than the cheapest. */
        return FT_TT_FORMAT_BF16;
    }
}

int ft_tt_ledger_mode_from_name(const char *name)
{
    if (!name)
        return -1;
    if (strcmp(name, "fp32") == 0)
        return FT_TT_LEDGER_FP32;
    if (strcmp(name, "bf16") == 0)
        return FT_TT_LEDGER_BF16;
    if (strcmp(name, "production") == 0)
        return FT_TT_LEDGER_PRODUCTION;
    if (strcmp(name, "bfp8") == 0)
        return FT_TT_LEDGER_BFP8;
    if (strcmp(name, "bfp4") == 0)
        return FT_TT_LEDGER_BFP4;
    if (strcmp(name, "moe_bfp6") == 0)
        return FT_TT_LEDGER_MOE_BFP6;
    return -1;
}

int ft_tt_ledger_format_from_name(const char *name)
{
    if (!name)
        return -1;
    if (strcmp(name, "fp32") == 0)
        return FT_TT_FORMAT_FP32;
    if (strcmp(name, "bf16") == 0)
        return FT_TT_FORMAT_BF16;
    if (strcmp(name, "bfp8_b") == 0)
        return FT_TT_FORMAT_BFP8_B;
    if (strcmp(name, "bfp4_b") == 0)
        return FT_TT_FORMAT_BFP4_B;
    if (strcmp(name, "bfp6_b") == 0)
        return FT_TT_FORMAT_BFP6_B;
    return -1;
}

const char *ft_tt_ledger_mode_name(ft_tt_ledger_mode mode)
{
    switch (mode) {
    case FT_TT_LEDGER_FP32:
        return "fp32";
    case FT_TT_LEDGER_BF16:
        return "bf16";
    case FT_TT_LEDGER_PRODUCTION:
        return "production";
    case FT_TT_LEDGER_BFP8:
        return "bfp8";
    case FT_TT_LEDGER_BFP4:
        return "bfp4";
    case FT_TT_LEDGER_MOE_BFP6:
        return "moe_bfp6";
    default:
        return "?";
    }
}

const char *ft_tt_ledger_mode_caution(ft_tt_ledger_mode mode)
{
    if (mode == FT_TT_LEDGER_BFP4)
        return "uniform Bfp4_b is a MEASURED quality regression: +40.216% perplexity "
               "(MiniCPM5-1B-Q8_0, 1067 paired chunks) against +0.175% for uniform Bfp8_b, "
               "and Bfp4_b is not faster (532.6 vs 534.1 GFLOP/s). It is selectable for "
               "capacity arithmetic only. Do not ship it.";
    return NULL;
}

const char *ft_tt_ledger_map_refusal(ft_tt_tensor_role role, uint32_t n_dims)
{
    switch (role) {
    case FT_TT_ROLE_NORM:
    case FT_TT_ROLE_BIAS:
    case FT_TT_ROLE_SSM_DECAY:
        return "a norm, bias or SSM decay term: fp32 on every rung";
    case FT_TT_ROLE_SSM_CONV:
        return "ssm_conv1d is a depthwise convolution, not a matmul";
    case FT_TT_ROLE_SSM_GATE:
        return "ssm_alpha / ssm_beta stay where the ledger puts them (tiny, unmeasured under block float)";
    case FT_TT_ROLE_ROUTER:
        return "a MoE router is consumed as an argmax; it stays where the ledger puts it";
    case FT_TT_ROLE_EXPERT_UP:
    case FT_TT_ROLE_EXPERT_DOWN:
        return "routed-expert stacks have their own rung (moe_bfp6)";
    case FT_TT_ROLE_EMBEDDING:
        return "token_embd is a host-side row table: a Bfp4 copy frees no card DRAM and the runtime has no "
               "Bfp4 row type";
    case FT_TT_ROLE_UNKNOWN:
        return "matches no role rule";
    case FT_TT_ROLE_LM_HEAD:
    case FT_TT_ROLE_ATTENTION:
    case FT_TT_ROLE_FFN_UP:
    case FT_TT_ROLE_FFN_DOWN:
    case FT_TT_ROLE_SSM_IN:
    case FT_TT_ROLE_SSM_OUT:
    case FT_TT_ROLE_PROJECTION:
        return n_dims == 2u ? NULL : "not a 2-D matmul weight";
    }
    return "matches no role rule";
}
