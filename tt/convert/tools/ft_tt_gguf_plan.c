/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_gguf_plan.c
 * @brief CLI: what a GGUF checkpoint WOULD convert to, at every rung of the precision
 *   ladder, without converting it.
 *
 * The converter answers "how big is the artifact" by producing one, which for a 50 GiB
 * checkpoint is ~23 minutes of packing to learn a number that is pure arithmetic on the
 * tensor directory: a tensor's stored size is `tile_rows * tile_cols * tile_bytes(format)`
 * and nothing about it depends on the VALUES. This tool reads the directory only — no
 * tensor byte is ever touched, so a split 50 GiB checkpoint plans in well under a second —
 * and prints the whole ladder side by side plus the capacity line that decides whether a
 * model fits on the card at a wanted context.
 *
 * It is also the fastest way to see the ROLE CLASSIFIER's opinion of a model it has never
 * met: `--per-tensor` prints the role and layout every tensor would get, which is how the
 * GDN mis-classification in Phase 3.1 was found and is how the next one will be.
 *
 *   ft_tt_gguf_plan model.gguf                    the ladder, and the fit
 *   ft_tt_gguf_plan model.gguf --per-tensor       every tensor's role/format/layout
 *   ft_tt_gguf_plan model.gguf --ledger bfp8      one rung only
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_gguf_map.h"
#include "ft_tt_capacity_plan.h"
#include "ft_tt_memmap.h"
#include "ft_tt_precision_ledger.h"
#include "ft_tt_weight_convert.h"

/**
 * Card DRAM the allocator actually hands out, taken from the code that hands it out
 * rather than from the box: FT_TT_MEM_DRAM_SIZE per bank across Blackhole's 8 banks.
 * 31.875 GiB, not 32.
 */
#define PLAN_CARD_BUDGET_BYTES ((uint64_t)FT_TT_MEM_DRAM_SIZE * 8ull)

#define PLAN_ROLE_SLOTS FT_TT_ROLE_COUNT
#define PLAN_FORMAT_SLOTS FT_TT_FORMAT_COUNT

typedef struct {
    uint64_t tensors;
    uint64_t bytes;
} plan_census;

typedef struct {
    plan_census by_role[PLAN_ROLE_SLOTS];
    plan_census by_format[PLAN_FORMAT_SLOTS];
    uint64_t total_bytes;
    uint64_t tensors;
    uint64_t unknown_2d;
} plan_totals;

static void plan_accumulate(plan_totals *totals, ft_tt_tensor_role role, ft_tt_tensor_format format,
                            uint32_t n_dims, uint64_t bytes)
{
    totals->tensors++;
    totals->total_bytes += bytes;
    if ((uint32_t)role < PLAN_ROLE_SLOTS) {
        totals->by_role[role].tensors++;
        totals->by_role[role].bytes += bytes;
    }
    if ((uint32_t)format < PLAN_FORMAT_SLOTS) {
        totals->by_format[format].tensors++;
        totals->by_format[format].bytes += bytes;
    }
    if (role == FT_TT_ROLE_UNKNOWN && n_dims >= 2)
        totals->unknown_2d++;
}

static double plan_gib(uint64_t bytes)
{
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

/**
 * Walk every tensor at one rung.
 *
 * @param per_tensor  print a line per tensor — the role classifier's opinion, which is the
 *                    thing worth reading on a model the converter has never seen
 */
static void plan_rung(const ft_gguf_map *map, ft_tt_model_family family, ft_tt_ledger_mode mode,
                      int per_tensor, plan_totals *totals)
{
    uint32_t count = ft_gguf_map_tensor_count(map);

    memset(totals, 0, sizeof(*totals));
    for (uint32_t i = 0; i < count; i++) {
        const ft_gguf_tensor *tensor = ft_gguf_map_tensor(map, i);
        ft_tt_tensor_role role = ft_tt_ledger_role(tensor->name);
        ft_tt_tensor_format format = ft_tt_ledger_format_for(mode, family, role, tensor->n_dims);
        ft_tt_tensor_layout layout;
        uint64_t bytes = ft_tt_capacity_tensor_bytes(tensor, format);
        int layout_known = ft_tt_convert_layout_for_checked(role, tensor->n_dims, &layout) == 0;

        plan_accumulate(totals, role, format, tensor->n_dims, bytes);
        if (per_tensor)
            printf("  %-44s %-14s %-8s %-9s %8llu x %-8llu %12llu B\n", tensor->name,
                   ft_tt_ledger_role_name(role), ft_tt_ledger_format_name(format),
                   layout_known ? (layout == FT_TT_LAYOUT_MATMUL_NMAJOR ? "nmajor" : "source")
                                : "REFUSED",
                   (unsigned long long)(tensor->dims[1] * tensor->dims[2] * tensor->dims[3]),
                   (unsigned long long)tensor->dims[0], (unsigned long long)bytes);
        ft_gguf_map_release(map, tensor);
    }
}

static void plan_print_census(const plan_totals *totals)
{
    printf("    by role:");
    for (uint32_t r = 0; r < PLAN_ROLE_SLOTS; r++) {
        if (totals->by_role[r].tensors == 0u)
            continue;
        printf("  %s %llu/%.2f GiB", ft_tt_ledger_role_name((ft_tt_tensor_role)r),
               (unsigned long long)totals->by_role[r].tensors, plan_gib(totals->by_role[r].bytes));
    }
    printf("\n    by format:");
    for (uint32_t f = 0; f < PLAN_FORMAT_SLOTS; f++) {
        if (totals->by_format[f].tensors == 0u)
            continue;
        printf("  %s %llu/%.2f GiB", ft_tt_ledger_format_name((ft_tt_tensor_format)f),
               (unsigned long long)totals->by_format[f].tensors, plan_gib(totals->by_format[f].bytes));
    }
    printf("\n");
}

/**
 * The capacity line.
 *
 * `kv_bytes_per_token` and `state_bytes` are the caller's model of the run-time footprint;
 * they are printed with the answer so the number is never read without its assumptions.
 */
static void plan_print_fit(uint64_t weights, uint64_t kv_bytes_per_token, uint64_t state_bytes,
                           uint64_t card_bytes)
{
    /* "no KV model was given" and "the weights do not fit" are DIFFERENT ANSWERS, and this
     * line used to print the second for both — so every run without --kv-bytes-per-token
     * read as a model that does not fit, including two that fit with 23 GiB to spare. */
    if (weights + state_bytes >= card_bytes) {
        printf("  fit: DOES NOT FIT (weights %.2f GiB + state against a %.3f GiB budget)\n",
               plan_gib(weights), plan_gib(card_bytes));
        return;
    }
    if (kv_bytes_per_token == 0u) {
        printf("  fit: weights %.2f GiB of %.3f GiB, %.2f GiB spare (no --kv-bytes-per-token, "
               "so no context estimate)\n",
               plan_gib(weights), plan_gib(card_bytes),
               plan_gib(card_bytes - weights - state_bytes));
        return;
    }
    {
        uint64_t spare = card_bytes - weights - state_bytes;

        printf("  fit: weights %.2f GiB, spare %.2f GiB, max context %llu tokens (%.1fk)\n",
               plan_gib(weights), plan_gib(spare), (unsigned long long)(spare / kv_bytes_per_token),
               (double)(spare / kv_bytes_per_token) / 1000.0);
    }
}

/**
 * The rung the product would pick on its own, and the arithmetic behind it.
 *
 * This calls the SAME ft_tt_capacity_select() `pantheon serve` calls, so what this prints
 * is what a serve would do — not a second opinion about it.
 */
static void plan_print_auto_selection(const ft_gguf_map *map, ft_tt_model_family family,
                                      uint64_t kv_bytes_per_token, uint64_t state_bytes,
                                      uint32_t context_tokens, uint64_t card_bytes)
{
    ft_tt_capacity_selection selection;
    ft_tt_capacity_budget budget;
    char line[FT_TT_CAPACITY_REASON_BYTES];
    int rc;

    memset(&budget, 0, sizeof(budget));
    budget.card_bytes = card_bytes;
    budget.runtime_state_bytes = state_bytes;
    budget.kv_bytes_per_token = kv_bytes_per_token;
    budget.context_tokens = context_tokens;

    rc = ft_tt_capacity_select(map, family, &budget, &selection);
    printf("\nauto-selected rung (what a serve would do with no one to ask)\n");
    printf("  %s\n",
           ft_tt_capacity_selection_line(&selection, &budget, line, sizeof(line)));
    for (uint32_t i = 0; i < selection.rung_count; i++)
        printf("  rung %-8s trunk artifact %llu B, MTP artifact %llu B%s%s\n",
               ft_tt_ledger_mode_name(selection.rungs[i].mode),
               (unsigned long long)selection.rungs[i].artifact_bytes,
               (unsigned long long)selection.rungs[i].mtp_bytes,
               selection.rungs[i].loadable ? "" : ", NOT loadable",
               selection.chosen && selection.chosen_index == i ? "  <- chosen" : "");
    if (rc != FT_TT_CAPACITY_OK)
        printf("  REFUSED (rc %d)\n", rc);
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s <model.gguf> [--per-tensor] [--ledger NAME]\n"
            "         [--kv-bytes-per-token N] [--state-mib N]\n"
            "         [--context-tokens N] [--card-gib X]\n"
            "  --per-tensor          one line per tensor: role, format, layout, bytes\n"
            "  --ledger NAME         plan one rung only (default: every rung)\n"
            "  --kv-bytes-per-token  for the capacity line; 0 disables it\n"
            "  --state-mib           constant recurrent state to reserve (GDN/SSM)\n"
            "  --context-tokens      context the auto-selector must leave room for\n"
            "  --card-gib            plan against a different card than this one\n",
            program);
}

int main(int argc, char **argv)
{
    static const ft_tt_ledger_mode rungs[] = {FT_TT_LEDGER_FP32, FT_TT_LEDGER_BF16,
                                              FT_TT_LEDGER_PRODUCTION, FT_TT_LEDGER_BFP8,
                                              FT_TT_LEDGER_MOE_BFP6, FT_TT_LEDGER_BFP4};
    static const char *rung_names[] = {"fp32", "bf16", "production", "bfp8", "moe_bfp6", "bfp4"};
    ft_gguf_map *map = NULL;
    char architecture[64] = "";
    ft_tt_model_family family;
    uint64_t kv_bytes_per_token = 0;
    uint64_t state_bytes = 0;
    uint64_t card_bytes = PLAN_CARD_BUDGET_BYTES;
    unsigned long context_tokens = 0;
    int per_tensor = 0;
    int only_rung = -1;

    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--per-tensor") == 0) {
            per_tensor = 1;
        } else if (strcmp(argv[i], "--ledger") == 0 && i + 1 < argc) {
            only_rung = ft_tt_ledger_mode_from_name(argv[++i]);
            if (only_rung < 0) {
                fprintf(stderr, "%s: unknown ledger rung '%s'\n", argv[0], argv[i]);
                return 2;
            }
        } else if (strcmp(argv[i], "--kv-bytes-per-token") == 0 && i + 1 < argc) {
            kv_bytes_per_token = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--state-mib") == 0 && i + 1 < argc) {
            state_bytes = strtoull(argv[++i], NULL, 10) * 1024ull * 1024ull;
        } else if (strcmp(argv[i], "--context-tokens") == 0 && i + 1 < argc) {
            context_tokens = strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--card-gib") == 0 && i + 1 < argc) {
            card_bytes = (uint64_t)(strtod(argv[++i], NULL) * 1024.0 * 1024.0 * 1024.0);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (ft_gguf_map_open(argv[1], &map) != FT_GGUF_OK) {
        fprintf(stderr, "%s: cannot open %s\n", argv[0], argv[1]);
        return 1;
    }
    (void)ft_gguf_map_metadata_string(map, "general.architecture", architecture, sizeof(architecture));
    family = ft_tt_ledger_family_from_architecture(architecture);

    printf("%s\n  architecture '%s' -> family %s, %u tensors across %u part(s)\n", argv[1],
           architecture, family == FT_TT_FAMILY_QWEN ? "QWEN" : "LLAMA",
           ft_gguf_map_tensor_count(map), ft_gguf_map_part_count(map));
    printf("  card budget %.3f GiB", plan_gib(card_bytes));
    if (kv_bytes_per_token)
        printf(", KV %llu B/token, reserved state %.0f MiB",
               (unsigned long long)kv_bytes_per_token, (double)state_bytes / (1024.0 * 1024.0));
    printf("\n");

    for (size_t r = 0; r < sizeof(rungs) / sizeof(rungs[0]); r++) {
        plan_totals totals;

        if (only_rung >= 0 && rungs[r] != (ft_tt_ledger_mode)only_rung)
            continue;
        printf("\nrung %s\n", rung_names[r]);
        plan_rung(map, family, rungs[r], per_tensor, &totals);
        printf("  %llu tensors, %llu B (%.2f GiB)\n", (unsigned long long)totals.tensors,
               (unsigned long long)totals.total_bytes, plan_gib(totals.total_bytes));
        plan_print_census(&totals);
        if (totals.unknown_2d)
            printf("    WARNING: %llu two-dimensional tensors are ROLE_UNKNOWN — the converter "
                   "REFUSES these\n",
                   (unsigned long long)totals.unknown_2d);
        plan_print_fit(totals.total_bytes, kv_bytes_per_token, state_bytes, card_bytes);
    }

    plan_print_auto_selection(map, family, kv_bytes_per_token, state_bytes,
                              (uint32_t)context_tokens, card_bytes);

    ft_gguf_map_close(map);
    return 0;
}
