/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_capacity_plan.c
 * @brief Implementation of the fit arithmetic and the automatic rung selector.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "ft_tt_capacity_plan.h"

#include <stdio.h>
#include <string.h>

#include "ft_tt_op_l1_layout.h"
#include "ft_tt_weight_convert.h"

/** The rungs the card holds a matmul program for, best precision first. */
static const ft_tt_ledger_mode CAPACITY_LADDER[FT_TT_CAPACITY_RUNG_COUNT] = {
    FT_TT_LEDGER_BFP8, FT_TT_LEDGER_MOE_BFP6, FT_TT_LEDGER_BFP4
};

/** A Bfp6_b tile: 64 B exponents + 768 B of 6-bit datums (ft_bfp.c). DRAM-only. */
#define CAPACITY_TILE_BYTES_BFP6_B 832u

static uint64_t capacity_tile_bytes(ft_tt_tensor_format format)
{
    switch (format) {
    case FT_TT_FORMAT_FP32:
        return FT_TT_OP_TILE_BYTES_FP32;
    case FT_TT_FORMAT_BF16:
        return FT_TT_OP_TILE_BYTES_BF16;
    case FT_TT_FORMAT_BFP8_B:
        return FT_TT_OP_TILE_BYTES_BFP8_B;
    case FT_TT_FORMAT_BFP4_B:
        return FT_TT_OP_TILE_BYTES_BFP4_B;
    case FT_TT_FORMAT_BFP6_B:
        return CAPACITY_TILE_BYTES_BFP6_B;
    default:
        return 0u;
    }
}

/**
 * A weight tile size the card holds a matmul program for, on this tensor: Bfp8_b and Bfp4_b
 * anywhere (pn_tt_program_for()), Bfp6_b only on a routed-expert stack (the expander-fed
 * `rtriad_bfp6`), fp32 only on a MoE router (its own fp32 router program, qwen36 Phase 5.3).
 */
static int capacity_tile_size_is_loadable(uint64_t tile_bytes, int expert_stack, int router)
{
    if (tile_bytes == CAPACITY_TILE_BYTES_BFP6_B)
        return expert_stack;
    if (tile_bytes == FT_TT_OP_TILE_BYTES_FP32)
        return router;
    return tile_bytes == FT_TT_OP_TILE_BYTES_BFP8_B || tile_bytes == FT_TT_OP_TILE_BYTES_BFP4_B;
}

/** The name prefix of the checkpoint's MTP block ("blk.L."), or "" when it has none. */
static void capacity_mtp_prefix(const ft_gguf_map *map, char *out, size_t out_bytes)
{
    uint32_t block;

    out[0] = '\0';
    if (ft_tt_convert_mtp_block(map, &block))
        snprintf(out, out_bytes, "blk.%u.", block);
}

static uint64_t capacity_tiles_for(uint64_t elements)
{
    return (elements + 31u) / 32u;
}

static uint64_t capacity_align_up(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1u) / alignment * alignment;
}

uint64_t ft_tt_capacity_tensor_bytes(const ft_gguf_tensor *tensor, ft_tt_tensor_format format)
{
    uint64_t cols;
    uint64_t rows;

    if (!tensor)
        return 0u;

    cols = tensor->dims[0];
    rows = tensor->dims[1] * tensor->dims[2] * tensor->dims[3];

    if (tensor->n_dims < 2)
        return capacity_align_up(cols * sizeof(float), FT_TT_ARTIFACT_ALIGNMENT);

    /* Transposed or not, the TILE COUNT is the same — which is why the artifact's size does
     * not depend on the layout and this arithmetic does not need to know it.
     *
     * The alignment here is defensive and CANNOT BE OBSERVED today: every tile size on the
     * ladder (4096, 2048, 1088, 576) is already a multiple of 64, so a tiled tensor is
     * never padded. Said plainly because a mutation arm removing it stays green, and a
     * reader deserves to know that is the arithmetic and not a hole in the suite. The 1-D
     * branch above is where alignment actually bites, and that one is gated. */
    return capacity_align_up(capacity_tiles_for(rows) * capacity_tiles_for(cols) *
                                 capacity_tile_bytes(format),
                             FT_TT_ARTIFACT_ALIGNMENT);
}

int ft_tt_capacity_rung_measure(const ft_gguf_map *map, ft_tt_model_family family,
                                ft_tt_ledger_mode mode, ft_tt_capacity_rung *out)
{
    char mtp_prefix[32];
    uint32_t count;
    uint32_t index;
    int mixed = 0;

    if (!map || !out)
        return FT_TT_CAPACITY_EINVAL;

    memset(out, 0, sizeof(*out));
    capacity_mtp_prefix(map, mtp_prefix, sizeof(mtp_prefix));
    out->mode = mode;
    out->loadable = 1;
    count = ft_gguf_map_tensor_count(map);

    for (index = 0; index < count; index++) {
        const ft_gguf_tensor *tensor = ft_gguf_map_tensor(map, index);
        ft_tt_tensor_role role;
        ft_tt_tensor_format format;
        ft_tt_tensor_layout layout;

        if (!tensor)
            return FT_TT_CAPACITY_EINVAL;

        role = ft_tt_ledger_role(tensor->name);
        format = ft_tt_ledger_format_for(mode, family, role, tensor->n_dims);
        if (mtp_prefix[0] && strncmp(tensor->name, mtp_prefix, strlen(mtp_prefix)) == 0) {
            /* The MTP artifact splits eh_proj into two halves of the same tile count, so its
             * bytes are this tensor's bytes. Loadability is the trunk's question. */
            out->mtp_bytes += ft_tt_capacity_tensor_bytes(tensor, format);
            ft_gguf_map_release(map, tensor);
            continue;
        }
        out->tensor_count++;
        out->artifact_bytes += ft_tt_capacity_tensor_bytes(tensor, format);

        if (role == FT_TT_ROLE_UNKNOWN && tensor->n_dims >= 2)
            out->unclassified_2d++;
        if (ft_tt_convert_is_expert_stack(tensor))
            out->expert_stacks++;

        if (ft_tt_convert_layout_for_checked(role, tensor->n_dims, &layout) == 0 &&
            layout == FT_TT_LAYOUT_MATMUL_NMAJOR) {
            uint64_t tile_bytes = capacity_tile_bytes(format);

            out->matmul_tensor_count++;
            if (out->matmul_tile_bytes == 0u)
                out->matmul_tile_bytes = (uint32_t)tile_bytes;
            else if ((uint64_t)out->matmul_tile_bytes != tile_bytes)
                mixed = 1;
            if (!capacity_tile_size_is_loadable(tile_bytes, ft_tt_convert_is_expert_stack(tensor),
                                                role == FT_TT_ROLE_ROUTER))
                out->loadable = 0;
        }
        ft_gguf_map_release(map, tensor);
    }
    /* The tied LM head the converter adds (ft_tt_convert_tied_head_source): the same bytes
     * as the embedding again, at the head's format. */
    const ft_gguf_tensor *tied = ft_tt_convert_tied_head_source(map);
    if (tied) {
        ft_tt_tensor_format format =
            ft_tt_ledger_format_for(mode, family, FT_TT_ROLE_LM_HEAD, tied->n_dims);
        uint64_t tile_bytes = capacity_tile_bytes(format);

        out->tensor_count++;
        out->artifact_bytes += ft_tt_capacity_tensor_bytes(tied, format);
        out->matmul_tensor_count++;
        if (out->matmul_tile_bytes != 0u && (uint64_t)out->matmul_tile_bytes != tile_bytes)
            mixed = 1;
        if (!capacity_tile_size_is_loadable(tile_bytes, 0, 0))
            out->loadable = 0;
    }

    /* A rung that mixes two matmul tile sizes still LOADS — pn_tt_program_for() holds both
     * — but the resident triad is one binary bound to one tile size, so it falls back to
     * one-shot dispatch. Report the mix as 0 rather than one of the two, so a caller that
     * needs a single size sees that it does not have one. */
    if (mixed)
        out->matmul_tile_bytes = 0u;

    return FT_TT_CAPACITY_OK;
}

/** The MoE rung on a dense model is the bfp8 rung again, not a third choice. */
static int capacity_rung_is_distinct(const ft_tt_capacity_rung *rung)
{
    return rung->mode != FT_TT_LEDGER_MOE_BFP6 || rung->expert_stacks > 0u;
}

static double capacity_gib(uint64_t bytes)
{
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

/**
 * Write the refusal: every rung that was measured, what it would have cost, and the budget
 * it was measured against.
 *
 * "Does not fit" with no numbers has been a wrong diagnosis here twice. The arithmetic is
 * the message.
 */
static void capacity_write_refusal(ft_tt_capacity_selection *selection,
                                   const ft_tt_capacity_budget *budget)
{
    size_t used = 0;
    uint32_t i;
    int written;

    written = snprintf(selection->reason, FT_TT_CAPACITY_REASON_BYTES,
                       "no loadable rung fits this card: %.3f GiB of DRAM, minus %.3f GiB "
                       "of run-time state and %.3f GiB of KV for %u context tokens "
                       "(%llu B/token), leaves %.3f GiB for weights.",
                       capacity_gib(budget->card_bytes),
                       capacity_gib(budget->runtime_state_bytes),
                       capacity_gib(budget->kv_bytes_per_token * budget->context_tokens),
                       budget->context_tokens,
                       (unsigned long long)budget->kv_bytes_per_token,
                       budget->card_bytes > selection->reserve_bytes
                           ? capacity_gib(budget->card_bytes - selection->reserve_bytes)
                           : 0.0);
    if (written > 0)
        used = (size_t)written;

    for (i = 0; i < selection->rung_count && used + 1u < FT_TT_CAPACITY_REASON_BYTES; i++) {
        const ft_tt_capacity_rung *rung = &selection->rungs[i];

        if (!capacity_rung_is_distinct(rung))
            continue;
        written = snprintf(selection->reason + used, FT_TT_CAPACITY_REASON_BYTES - used,
                           " %s would be %.2f GiB%s.", ft_tt_ledger_mode_name(rung->mode),
                           capacity_gib(rung->artifact_bytes + rung->mtp_bytes),
                           rung->loadable ? "" : " and is not loadable on this card");
        if (written <= 0)
            break;
        used += (size_t)written;
    }
}

int ft_tt_capacity_select(const ft_gguf_map *map, ft_tt_model_family family,
                          const ft_tt_capacity_budget *budget, ft_tt_capacity_selection *out)
{
    uint32_t i;

    if (!map || !budget || !out)
        return FT_TT_CAPACITY_EINVAL;
    if (budget->card_bytes == 0u)
        return FT_TT_CAPACITY_EINVAL;

    memset(out, 0, sizeof(*out));
    out->rung_count = FT_TT_CAPACITY_RUNG_COUNT;
    out->reserve_bytes =
        budget->runtime_state_bytes + budget->kv_bytes_per_token * budget->context_tokens;

    for (i = 0; i < FT_TT_CAPACITY_RUNG_COUNT; i++) {
        int rc = ft_tt_capacity_rung_measure(map, family, CAPACITY_LADDER[i], &out->rungs[i]);

        if (rc != FT_TT_CAPACITY_OK)
            return rc;
    }

    /* A tensor the classifier does not recognise is refused by the converter itself, so no
     * rung is reachable and "it does not fit" would send the reader to the wrong problem. */
    if (out->rungs[0].unclassified_2d > 0u) {
        snprintf(out->reason, FT_TT_CAPACITY_REASON_BYTES,
                 "%u two-dimensional tensor(s) have no role in the precision ledger, and the "
                 "converter refuses a checkpoint it cannot classify. This is a ledger gap, "
                 "not a capacity one — re-run ft_tt_gguf_plan --per-tensor to name them.",
                 out->rungs[0].unclassified_2d);
        return FT_TT_CAPACITY_EUNCLASSIFIED;
    }

    for (i = 0; i < FT_TT_CAPACITY_RUNG_COUNT; i++) {
        const ft_tt_capacity_rung *rung = &out->rungs[i];
        uint64_t needed;

        if (!rung->loadable || !capacity_rung_is_distinct(rung))
            continue;
        needed = rung->artifact_bytes + rung->mtp_bytes + out->reserve_bytes;
        if (needed > budget->card_bytes)
            continue;

        out->chosen = 1;
        out->chosen_index = i;
        out->mode = rung->mode;
        out->spare_bytes = budget->card_bytes - needed;
        out->max_context_tokens =
            budget->kv_bytes_per_token > 0u
                ? budget->context_tokens + out->spare_bytes / budget->kv_bytes_per_token
                : 0u;
        return FT_TT_CAPACITY_OK;
    }

    capacity_write_refusal(out, budget);
    return FT_TT_CAPACITY_ENOFIT;
}

const char *ft_tt_capacity_selection_line(const ft_tt_capacity_selection *selection,
                                          const ft_tt_capacity_budget *budget, char *out,
                                          size_t out_bytes)
{
    const ft_tt_capacity_rung *chosen;
    const ft_tt_capacity_rung *rejected;

    if (!out || out_bytes == 0u)
        return out;
    if (!selection || !budget) {
        out[0] = '\0';
        return out;
    }
    if (!selection->chosen) {
        snprintf(out, out_bytes, "%s", selection->reason);
        return out;
    }

    chosen = &selection->rungs[selection->chosen_index];

    /* When a higher rung was measured and rejected, SAY which and by how much — "chose
     * bfp4" on its own reads as a preference, and it is not one. */
    rejected = NULL;
    for (uint32_t i = selection->chosen_index; i > 0u && !rejected; i--)
        if (capacity_rung_is_distinct(&selection->rungs[i - 1u]))
            rejected = &selection->rungs[i - 1u];
    if (rejected) {
        snprintf(out, out_bytes,
                 "%s: %.2f GiB of weights, %.2f GiB spare, up to %llu context tokens "
                 "(%s would have been %.2f GiB against a %.3f GiB card and does not fit)",
                 ft_tt_ledger_mode_name(chosen->mode),
                 capacity_gib(chosen->artifact_bytes + chosen->mtp_bytes),
                 capacity_gib(selection->spare_bytes),
                 (unsigned long long)selection->max_context_tokens,
                 ft_tt_ledger_mode_name(rejected->mode),
                 capacity_gib(rejected->artifact_bytes + rejected->mtp_bytes),
                 capacity_gib(budget->card_bytes));
        return out;
    }

    snprintf(out, out_bytes,
             "%s: %.2f GiB of weights, %.2f GiB spare, up to %llu context tokens "
             "(the highest rung this card loads, and it fits)",
             ft_tt_ledger_mode_name(chosen->mode),
             capacity_gib(chosen->artifact_bytes + chosen->mtp_bytes),
             capacity_gib(selection->spare_bytes),
             (unsigned long long)selection->max_context_tokens);
    return out;
}
