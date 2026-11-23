/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_precision_ledger.h
 * @brief Which on-card numeric format each tensor group gets, and the three
 *   rungs of the bring-up precision ladder.
 *
 * This is spec §6.4's ledger — TT's own production per-tensor-group precision
 * map from the tt_transformers model config — made
 * executable, plus the two lower rungs `docs/tt_phase0_decisions.md` §1 asks
 * the bring-up to climb through first.
 *
 * ---------------------------------------------------------------------------
 * THE LADDER
 * ---------------------------------------------------------------------------
 *   FT_TT_LEDGER_FP32        everything fp32. No packer in the path at all,
 *                            so Phase 4's first correct kernel is honest.
 *   FT_TT_LEDGER_BF16        everything bf16. Halves the upload, still no
 *                            block-float term in the error budget.
 *   FT_TT_LEDGER_PRODUCTION  the §6.4 allocation below.
 *   FT_TT_LEDGER_BFP8        every 2-D tensor Bfp8_b, no per-role exceptions.
 *   FT_TT_LEDGER_BFP4        every 2-D tensor Bfp4_b. **MEASURED HARMFUL.**
 *   FT_TT_LEDGER_MOE_BFP6    FT_TT_LEDGER_BFP8, except routed MoE experts
 *                            (`ffn_*_exps`) at Bfp6_b. Qwen3.6-35B-A3B's rung.
 *
 * Each rung is a separate, measurable regression check. The fallback when a
 * tensor group regresses is to move that group UP this ladder — precision, not
 * format. Route A (K-quants) is dropped and is not a fallback.
 *
 * A FORMAT MAP (`--format-map`, ft_tt_format_map.h) is not a rung either: it moves named
 * matmul tensors between Bfp8_b and Bfp4_b on top of one (ft_tt_ledger_map_refusal() says
 * which tensors it may name). Its artifact is named by its map, and the CLI prints
 * ft_tt_format_map_caution(), which names the map, beside any rung caution.
 *
 * ---------------------------------------------------------------------------
 * THE TWO UNIFORM RUNGS ARE NOT SYMMETRIC — ONE IS REFUTED
 * ---------------------------------------------------------------------------
 * They exist because the capacity arithmetic for a model that does not fit is a
 * format question and it has to be ASKABLE from the command line. But they are
 * not two equivalent choices, and the ladder must not present them as if they
 * were. Paired perplexity, MiniCPM5-1B-Q8_0, 2 MiB enwik8 slice, -c 512, 1067
 * chunks (`docs/tt_bringup_log.md` §13.5 and the Phase 9 round that reproduced
 * it four rows for four):
 *
 *   uniform Bfp8_b          +0.175%    the only defensible uniform rung
 *   ffn_gate/up Bfp4_b      +9.687%
 *   embed+head Bfp4_b       +5.729%
 *   uniform Bfp4_b         +40.216%    a quality collapse, not a trade
 *
 * So BFP4 is selectable and is NOT a recommendation. Reconstruction error said
 * the opposite — relRMS spread across tensor types is under 1.4x, with no
 * outlier — and that is exactly the trap: relRMS 15x maps to perplexity 230x.
 * `ft_tt_ledger_mode_caution()` returns the sentence a CLI must print when this
 * rung is chosen, so the cost travels with the choice instead of living in a
 * document nobody opens.
 *
 * ---------------------------------------------------------------------------
 * THE PRODUCTION ALLOCATION, AND THE ONE ENTRY THAT WILL BITE
 * ---------------------------------------------------------------------------
 *   ffn_gate / ffn_up   BFP4   TT uses BFP4 here universally, even in accuracy
 *                              mode: up-projections are the most
 *                              precision-tolerant tensors in a transformer
 *   ffn_down            BFP8   error concentrates on the accumulator side
 *   attn_q/k/v/output   BFP8   ...but bf16 on Qwen-family models. §6.4 records
 *                              that Qwen2.5-7B degrades and needs BF16
 *                              attention where Llama does not. MiniCPM5-1B
 *                              declares `general.architecture = llama`, so it
 *                              takes the BFP8 path; the exception exists in
 *                              this file so the second model does not
 *                              rediscover it. **The exception is about REAL
 *                              ATTENTION and it is scoped to the four real
 *                              attention projections by name** — see below
 *   GDN in/out          BFP8   `attn_qkv`, `attn_gate`, `ssm_out` — a hybrid
 *                              model's linear-attention projections. They are
 *                              matmul weights and they are N-major, and the
 *                              Qwen bf16 exception does NOT reach them
 *
 * ---------------------------------------------------------------------------
 * HYBRID (GDN / SSM) MODELS, AND THE TWO WAYS THIS FILE USED TO GET THEM WRONG
 * ---------------------------------------------------------------------------
 * Qwen3.6 and Qwen3.8 are Gated-DeltaNet hybrids: most layers are GDN and only
 * a few are real attention. Both mistakes were found on 2026-08-14, the first
 * time the classifier met one.
 *
 *  1. **A LAYOUT BUG THAT CANNOT ANNOUNCE ITSELF.** `ssm_out` is a genuine
 *     6144->5120 matmul weight, and `ssm_alpha`/`ssm_beta` are genuine 5120->48
 *     ones. They matched no rule, so they were ROLE_UNKNOWN, so they kept
 *     FT_TT_LAYOUT_SOURCE. Feed a SOURCE-layout tensor to a kernel that reads
 *     N-major and **every index is in range and every answer is wrong** — no
 *     crash, no refusal, no PCC to look at, because nothing is measured at run
 *     time. That is why an unclassified 2-D tensor is now REFUSED by the
 *     converter (`ft_tt_convert_layout_for_checked`) rather than defaulted.
 *
 *  2. **THE QWEN bf16 EXCEPTION MISFIRED ON GDN.** It matched the `attn_`
 *     PREFIX, and a GDN layer's in-projections are named `attn_qkv` and
 *     `attn_gate`. On Qwen3.8 that is **96 tensors held at bf16 for a reason
 *     about softmax attention that does not apply to linear attention**, and it
 *     cost **7.50 GiB** — enough to make the production rung the WORST option
 *     that fits the card. The rule is now scoped to the four real attention
 *     projections by name.
 *
 * The GDN roles and what they are, read off Qwen3.8-27B's own tensors:
 *   attn_qkv    [5120 -> 10240]  q(16x128) k(16x128) v(48x128), fused
 *   attn_gate   [5120 ->  6144]  the output gate z (48x128)
 *   ssm_out     [6144 ->  5120]  the out-projection
 *   ssm_alpha   [5120 ->    48]  per-head gate projections. Small enough that
 *   ssm_beta    [5120 ->    48]  bf16 costs ~30 MiB across 48 layers, so they
 *                                stay bf16: nothing has measured them, and this
 *                                is not where bytes are worth taking
 *   ssm_conv1d  [4 taps x 10240] a depthwise convolution, NOT a matmul. Source
 *                                layout, fp32, and it is 63 MiB — the padding
 *                                waste of a 4-wide tensor in 32-wide tiles is
 *                                real and is not worth a special case yet
 *   ssm_a, ssm_dt.bias           1-D decay parameters, fp32 like every 1-D
 *   nextn.eh_proj [10240->5120]  the MTP draft layer's input projection; a
 *                                matmul like any other
 *   token_embd / output BFP8   §6.4 puts these at bf16 because the head feeds
 *                              sampling directly. MEASURED and demoted
 *                              2026-08-12: a paired perplexity A/B with only
 *                              these two at BFP8 moved perplexity +0.035%
 *                              (against +5.73% for the same two at BFP4, the
 *                              control). They were 59% of the artifact, so
 *                              this is -358.6 MiB and takes the whole thing
 *                              BELOW the Q8_0 source. `docs/tt_bringup_log.md`
 *                              §13.5. This is the one deliberate departure
 *                              from §6.4, and it rests on perplexity, not PCC
 *   norms and biases    fp32   1-D, negligible bytes, and normalization scale
 *                              errors propagate through the whole residual
 *
 * @status The allocation is TT's, unvalidated on our models. The Phase 3
 *   quality gate (tools/ft_tt_pcc_gate.c) is what turns it from inherited to
 *   measured; results are recorded in docs/tt_bringup_log.md §12.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_PRECISION_LEDGER_H
#define FT_TT_PRECISION_LEDGER_H

#include <stdint.h>

/** On-card storage format for a converted tensor. */
typedef enum {
    FT_TT_FORMAT_FP32 = 0,   /**< 4 B/value, face-ordered tiles */
    FT_TT_FORMAT_BF16 = 1,   /**< 2 B/value, face-ordered tiles */
    FT_TT_FORMAT_BFP8_B = 2, /**< 1088 B/tile */
    FT_TT_FORMAT_BFP4_B = 3, /**< 576 B/tile */
    FT_TT_FORMAT_BFP6_B = 4  /**< 832 B/tile; a DRAM format, expanded to Bfp8_b on the card */
} ft_tt_tensor_format;

/** Which rung of the precision ladder to convert at. */
typedef enum {
    FT_TT_LEDGER_FP32 = 0,
    FT_TT_LEDGER_BF16 = 1,
    FT_TT_LEDGER_PRODUCTION = 2,
    FT_TT_LEDGER_BFP8 = 3, /**< uniform Bfp8_b; measured +0.175% perplexity */
    FT_TT_LEDGER_BFP4 = 4, /**< uniform Bfp4_b; measured +40.2% — see the header */
    /**
     * Routed MoE experts Bfp6_b, the router (ffn_gate_inp) fp32 (D4), every other 2-D
     * tensor exactly as FT_TT_LEDGER_BFP8 (embedding included). Qwen3.6-35B-A3B: Bfp6_b
     * experts measured +0.053% enwik8 perplexity, 95.80% agentic top-1. On a model
     * with no routed experts it is byte-identical to FT_TT_LEDGER_BFP8.
     */
    FT_TT_LEDGER_MOE_BFP6 = 5
} ft_tt_ledger_mode;

/** Model families that need a different allocation. */
typedef enum {
    FT_TT_FAMILY_LLAMA = 0, /**< also MiniCPM, Mistral, Phi — BFP8 attention is fine */
    FT_TT_FAMILY_QWEN = 1   /**< attention-precision-sensitive; bf16 attention */
} ft_tt_model_family;

/**
 * Coarse role of a tensor, derived from its GGUF name.
 *
 * The numbering is written into every artifact's directory (`entry->role`), so
 * new roles are APPENDED and existing ones never renumbered.
 */
typedef enum {
    FT_TT_ROLE_UNKNOWN = 0,
    FT_TT_ROLE_EMBEDDING,
    FT_TT_ROLE_LM_HEAD,
    FT_TT_ROLE_ATTENTION, /**< real softmax attention: attn_q/k/v/output only */
    FT_TT_ROLE_FFN_UP,    /**< gate and up projections */
    FT_TT_ROLE_FFN_DOWN,  /**< down projection */
    FT_TT_ROLE_NORM,
    FT_TT_ROLE_BIAS,
    FT_TT_ROLE_SSM_IN,      /**< GDN in-projections: attn_qkv, attn_gate */
    FT_TT_ROLE_SSM_OUT,     /**< GDN out-projection: ssm_out */
    FT_TT_ROLE_SSM_GATE,    /**< per-head gate projections: ssm_alpha, ssm_beta */
    FT_TT_ROLE_SSM_CONV,    /**< depthwise conv1d — NOT a matmul */
    FT_TT_ROLE_SSM_DECAY,  /**< 1-D decay parameters: ssm_a, ssm_dt */
    FT_TT_ROLE_PROJECTION, /**< a plain matmul that belongs to no other group */
    FT_TT_ROLE_ROUTER,     /**< MoE expert router: ffn_gate_inp */
    FT_TT_ROLE_EXPERT_UP,  /**< routed experts' gate and up: ffn_{gate,up}_exps (3-D) */
    FT_TT_ROLE_EXPERT_DOWN /**< routed experts' down: ffn_down_exps (3-D) */
} ft_tt_tensor_role;

/** Roles a ledger can name; sized for per-role tables. */
#define FT_TT_ROLE_COUNT 17

/** Formats a directory entry can carry; sized for per-format tables. */
#define FT_TT_FORMAT_COUNT 5

/**
 * Classify a GGUF tensor name.
 *
 * Never fails; a name this build has not reasoned about gets ROLE_UNKNOWN, and
 * the CONVERTER refuses an unclassified 2-D tensor rather than guessing a
 * layout for it. Returning UNKNOWN is therefore a real answer here, not a
 * fallback — it is what makes the refusal possible.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
ft_tt_tensor_role ft_tt_ledger_role(const char *tensor_name);

/** Human-readable role name, for reports. */
const char *ft_tt_ledger_role_name(ft_tt_tensor_role role);

/** Human-readable format name, for reports and the artifact header. */
const char *ft_tt_ledger_format_name(ft_tt_tensor_format format);

/** Model family from GGUF's `general.architecture` string. */
ft_tt_model_family ft_tt_ledger_family_from_architecture(const char *architecture);

/**
 * The format a tensor gets.
 *
 * 1-D tensors are always fp32 whatever the rung: they are norms and biases,
 * they are a rounding error in the artifact size, and block-floating a
 * 16-element block of a normalization scale buys nothing and costs quality.
 *
 * @param mode         which rung of the ladder
 * @param family       model family, for the Qwen attention exception
 * @param role         from ft_tt_ledger_role()
 * @param n_dims       tensor rank, as GGUF declares it
 * @return             the on-card format
 */
ft_tt_tensor_format ft_tt_ledger_format_for(ft_tt_ledger_mode mode, ft_tt_model_family family, ft_tt_tensor_role role,
                                            uint32_t n_dims);

/** Ladder rung from a command-line word, or -1 if it names none of them. */
int ft_tt_ledger_mode_from_name(const char *name);

/**
 * The inverse of ft_tt_ledger_format_name(): a format from its own report word,
 * or -1 if it names none of them.
 *
 * Exists so a caller that has to NAME one format — an experiment demoting a
 * single tensor family to measure that family's share of the error — spells it
 * the way every report already prints it, instead of growing a second
 * vocabulary for the same four values.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_ledger_format_from_name(const char *name);

/** The command-line word for a rung, for reports and artifact provenance. */
const char *ft_tt_ledger_mode_name(ft_tt_ledger_mode mode);

/**
 * The measured cost of choosing this rung, or NULL when there is nothing to
 * warn about.
 *
 * A CLI MUST print this when it is non-NULL. It exists because BFP4 has to be
 * reachable — the capacity arithmetic for a model that does not otherwise fit
 * is a format question — while being a measured quality regression of +40.2%
 * perplexity, and a menu entry that reads like every other menu entry is how
 * that gets picked by accident.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
const char *ft_tt_ledger_mode_caution(ft_tt_ledger_mode mode);

/**
 * Why a FORMAT MAP (ft_tt_format_map.h, the converter's `--format-map`) may not name a
 * tensor of this role and rank, or NULL when it may.
 *
 * A map moves single tensors between Bfp8_b and Bfp4_b ON TOP OF a rung, and it may name
 * only the dense 2-D matmul weights whose bytes the card reads per token: real attention,
 * the GDN in/out projections, the FFN, the LM head and the MTP `eh_proj`. Everything else
 * is refused, each for its own reason: norms, biases and the SSM decay terms are fp32 on
 * every rung; `ssm_conv1d` is not a matmul; `ssm_alpha`/`ssm_beta` and a MoE router stay
 * where the ledger puts them (tiny, and unmeasured or discrete under block float);
 * `token_embd` is a host-side row table, so a Bfp4 copy frees no card DRAM and the runtime
 * has no Bfp4 row type; routed-expert stacks have their own rung (moe_bfp6).
 *
 * A map built this way is NOT a rung: the artifact it produces is named by its map. This
 * rule lives here, not in the map library, so the converter checks it with nothing more
 * linked than it already links.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
const char *ft_tt_ledger_map_refusal(ft_tt_tensor_role role, uint32_t n_dims);

#endif /* FT_TT_PRECISION_LEDGER_H */
