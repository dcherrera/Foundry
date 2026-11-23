/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_capacity_plan.h
 * @brief How big a GGUF checkpoint's artifact would be at each rung, and WHICH RUNG to
 *   convert it at, from the tensor directory alone.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS A LIBRARY AND NOT ARITHMETIC IN A TOOL
 * ---------------------------------------------------------------------------
 * The size of a converted tensor is `tiles(rows) * tiles(cols) * tile_bytes(format)`,
 * aligned up — it depends on the SHAPE and the FORMAT and on nothing about the values. So
 * the whole fit question is answerable from the GGUF directory in milliseconds, without
 * touching a tensor byte, and `ft_tt_gguf_plan` has answered it that way since Phase 3.
 *
 * It stopped being a tool's business when the product had to answer it: `pantheon serve`
 * gets a `.gguf` and has to pick a rung with nobody to ask. One implementation of the
 * arithmetic, used by both, or the tool and the product can disagree about whether a model
 * fits — and the one that matters would be the unreviewed one.
 *
 * ---------------------------------------------------------------------------
 * THE LADDER IS TWO RUNGS, AND THAT IS A HARDWARE FACT
 * ---------------------------------------------------------------------------
 * `pn_tt_program_for()` holds a matmul program for a 1088-byte (Bfp8_b) and a 576-byte
 * (Bfp4_b) weight tile and for nothing else — a 2048-byte bf16 tile is refused. So the
 * candidate rungs are FT_TT_LEDGER_BFP8 and FT_TT_LEDGER_BFP4, in that order, and the
 * PRODUCTION ledger (which puts a Qwen model's attention at bf16) is NOT among them.
 * Blackhole's own ladder is 8 / 4 / 2 with no 6-bit rung, so there is nothing between them
 * for a DENSE model.
 *
 * A MoE checkpoint has a third rung between them, FT_TT_LEDGER_MOE_BFP6: the routed experts
 * at Bfp6_b (832 B tiles), everything else as Bfp8_b. Its 832 B tile is loadable ONLY on
 * routed-expert stacks, which the expander-fed `rtriad_bfp6` serves (qwen36_blackhole_plan.md
 * Phase 5.1); an 832 B tile anywhere else marks the rung unloadable. On a dense model the
 * rung is byte-identical to bfp8, so it never changes a dense model's answer.
 *
 * A trailing MTP / NextN block is NOT in the trunk artifact (FT_TT_CONVERT_SCOPE_TRUNK) but
 * it IS on the card (`.mtp.fttw`), so its bytes are measured apart (`mtp_bytes`) and charged
 * to the fit.
 *
 * A selection is the FIRST candidate that fits. Highest precision that the card can hold,
 * never a cheaper one for its own sake.
 *
 * A FORMAT MAP (`--format-map`, ft_tt_format_map.h) is priced by NONE of these rows: it
 * moves named tensors between Bfp8_b and Bfp4_b on top of a rung, so a mapped artifact is
 * smaller than its rung's row here and is never selected by this plan. Price it from its
 * map, and name it by its map.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FT_TT_CAPACITY_PLAN_H
#define FT_TT_CAPACITY_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "ft_gguf_map.h"
#include "ft_tt_precision_ledger.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Candidate rungs the card can actually load, best first. */
#define FT_TT_CAPACITY_RUNG_COUNT 3u

/** Room for the refusal arithmetic, which is a whole paragraph on purpose. */
#define FT_TT_CAPACITY_REASON_BYTES 512u

/** Status codes. Distinct values, because the CALLER's response differs for each. */
enum {
    FT_TT_CAPACITY_OK = 0,
    FT_TT_CAPACITY_EINVAL = -3,
    /** No candidate rung fits the budget. `reason` carries the arithmetic. */
    FT_TT_CAPACITY_ENOFIT = -11,
    /** The converter would refuse this checkpoint whatever the budget: it holds 2-D
     *  tensors whose role the classifier does not recognise. A different problem with a
     *  different fix, so it is a different code and never reported as "too big". */
    FT_TT_CAPACITY_EUNCLASSIFIED = -9
};

/** What one rung would cost, measured over the whole tensor directory. */
typedef struct {
    ft_tt_ledger_mode mode;
    uint64_t artifact_bytes;      /**< payload the converter would write, alignment included */
    uint64_t mtp_bytes;           /**< the MTP block's artifact at this rung; 0 without one */
    uint32_t tensor_count;
    uint32_t matmul_tensor_count; /**< tensors stored MATMUL_NMAJOR — the ones a program serves */
    uint32_t matmul_tile_bytes;   /**< the single matmul tile size, or 0 when the rung mixes two */
    uint32_t unclassified_2d;     /**< 2-D tensors the converter would refuse outright */
    uint32_t expert_stacks;       /**< routed-expert stacks in the trunk; 0 = a dense model */
    int loadable;                 /**< every matmul tile is a size pn_tt_program_for() holds */
} ft_tt_capacity_rung;

/**
 * The run-time footprint a selection has to leave room for, in the caller's terms.
 *
 * These are the CALLER's numbers because they are the runtime's, not the checkpoint's: the
 * KV geometry comes from the model config and the scratch from what the forward allocates.
 * They are carried into every message this library prints so a fit is never read without
 * the assumptions that produced it.
 */
typedef struct {
    uint64_t card_bytes;         /**< total card DRAM the allocator hands out */
    uint64_t runtime_state_bytes;/**< scratch, activations, rotary tables, recurrent state */
    uint64_t kv_bytes_per_token; /**< 0 means "do not reserve context" */
    uint32_t context_tokens;     /**< context the run must be able to serve */
} ft_tt_capacity_budget;

/** The answer: which rung, why, and the arithmetic behind it. */
typedef struct {
    int chosen;                  /**< 1 when a rung was selected, 0 when none fits */
    ft_tt_ledger_mode mode;      /**< meaningful only when `chosen` */
    uint32_t chosen_index;       /**< index into `rungs` */
    uint32_t rung_count;
    ft_tt_capacity_rung rungs[FT_TT_CAPACITY_RUNG_COUNT];
    uint64_t reserve_bytes;      /**< runtime_state + kv_bytes_per_token * context_tokens */
    uint64_t spare_bytes;        /**< card - artifact - MTP - reserve, when chosen */
    uint64_t max_context_tokens; /**< what the spare would buy on top of `context_tokens` */
    char reason[FT_TT_CAPACITY_REASON_BYTES]; /**< the refusal, when !chosen; else "" */
} ft_tt_capacity_selection;

/**
 * Stored bytes for one tensor at one format, INCLUDING the artifact's alignment pad.
 *
 * This is the one implementation of the size arithmetic. 1-D tensors are stored untiled as
 * fp32 whatever the rung, which is why the format argument does not reach them.
 *
 * @param tensor  a directory entry; no tensor byte is read
 * @param format  the format the ledger assigned
 * @return        bytes the artifact would spend, or 0 for a format with no tile size
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t ft_tt_capacity_tensor_bytes(const ft_gguf_tensor *tensor, ft_tt_tensor_format format);

/**
 * Measure one rung over a whole checkpoint.
 *
 * @param map     an open GGUF map; only its directory is walked
 * @param family  from ft_tt_ledger_family_from_architecture()
 * @param mode    the rung
 * @param out     zeroed and filled
 * @return        FT_TT_CAPACITY_OK, or FT_TT_CAPACITY_EINVAL
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (touches the mmap)
 */
int ft_tt_capacity_rung_measure(const ft_gguf_map *map, ft_tt_model_family family,
                                ft_tt_ledger_mode mode, ft_tt_capacity_rung *out);

/**
 * Choose the rung to convert this checkpoint at.
 *
 * Walks FT_TT_LEDGER_BFP8 then FT_TT_LEDGER_BFP4 and takes the first whose artifact plus
 * the caller's reserve fits `card_bytes`. Every rung is measured either way, so the caller
 * can print the table it rejected as well as the row it took.
 *
 * @param map     an open GGUF map
 * @param family  from ft_tt_ledger_family_from_architecture()
 * @param budget  the run-time footprint to leave room for
 * @param out     zeroed and filled; `reason` carries the arithmetic on a refusal
 * @return        FT_TT_CAPACITY_OK when a rung was chosen;
 *                FT_TT_CAPACITY_EUNCLASSIFIED when the converter would refuse the file;
 *                FT_TT_CAPACITY_ENOFIT when no rung fits;
 *                FT_TT_CAPACITY_EINVAL on a bad argument
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_capacity_select(const ft_gguf_map *map, ft_tt_model_family family,
                          const ft_tt_capacity_budget *budget, ft_tt_capacity_selection *out);

/**
 * The one line a serve prints about its own rung choice.
 *
 * Names the rung, the artifact size, what it was measured against and what it left — so the
 * choice is auditable from the log without re-running the planner.
 *
 * @param selection  a chosen selection; a refusal writes its `reason` instead
 * @param budget     the same budget the selection was made against
 * @param out        destination
 * @param out_bytes  its size; the line is always NUL-terminated
 * @return           `out`
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
const char *ft_tt_capacity_selection_line(const ft_tt_capacity_selection *selection,
                                          const ft_tt_capacity_budget *budget, char *out,
                                          size_t out_bytes);

#ifdef __cplusplus
}
#endif

#endif /* FT_TT_CAPACITY_PLAN_H */
