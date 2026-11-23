/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_pcc_gate.c
 * @brief CLI: the Phase 3 quality gate. Score every tensor of a checkpoint at
 *   every candidate on-card format and print the table the Route B go/no-go is
 *   decided on.
 *
 * Writes no artifact. For each tensor it dequantizes the source to fp32 —
 * which is EXACT for a Q8_0 checkpoint, so the reference side of the
 * comparison carries no error of its own (`docs/tt_phase0_decisions.md` §1) —
 * encodes it, decodes it, and folds the result into a Pearson correlation,
 * an RMS error and a maximum absolute error.
 *
 * Read the output by role, not by tensor: the decision the ledger makes is per
 * tensor group. The `--summary` mode prints exactly that.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_precision_ledger.h"
#include "ft_tt_weight_convert.h"

#define ROLE_COUNT 8

typedef struct {
    uint32_t tensors;
    double worst_pcc;
    double best_pcc;
    double sum_pcc;
    double worst_rms;
    /* Relative RMS is what ranks tensor types against each other; absolute RMS
     * ranks them by how large their weights happen to be. Both are kept because
     * the absolute one is what a downstream error budget adds up. */
    double worst_relative_rms;
    double sum_relative_rms;
    double sum_source_relative_rms; /**< only meaningful with --reference */
    double worst_max_abs;
    char worst_name[FT_TT_ARTIFACT_NAME_BYTES];
} role_summary;

typedef struct {
    int verbose;
    int has_reference;
    ft_tt_tensor_format format;
    role_summary roles[ROLE_COUNT];
} gate_state;

static int on_tensor(const ft_tt_convert_tensor_report *report, void *user)
{
    gate_state *state = (gate_state *)user;
    role_summary *summary;

    if ((int)report->role >= ROLE_COUNT)
        return 0;

    summary = &state->roles[report->role];
    if (summary->tensors == 0) {
        summary->worst_pcc = report->pcc;
        summary->best_pcc = report->pcc;
    }
    if (report->relative_rms_error > summary->worst_relative_rms) {
        summary->worst_relative_rms = report->relative_rms_error;
        /* The worst tensor is the one with the worst RELATIVE error, which is
         * the one a mixed format assignment would have to move. PCC saturates
         * at 0.9999... over a whole tensor and stops discriminating long before
         * relative RMS does. */
        snprintf(summary->worst_name, sizeof(summary->worst_name), "%s", report->name);
    }
    if (report->pcc < summary->worst_pcc)
        summary->worst_pcc = report->pcc;
    if (report->pcc > summary->best_pcc)
        summary->best_pcc = report->pcc;
    if (report->rms_error > summary->worst_rms)
        summary->worst_rms = report->rms_error;
    if (report->max_abs_error > summary->worst_max_abs)
        summary->worst_max_abs = report->max_abs_error;
    summary->sum_pcc += report->pcc;
    summary->sum_relative_rms += report->relative_rms_error;
    summary->sum_source_relative_rms += report->source_relative_rms_error;
    summary->tensors++;

    if (state->verbose) {
        printf("%-40s %-12s %8llu x %-8llu  pcc %.9f  relrms %.6e  rms %.6e  max %.6e", report->name,
               ft_tt_ledger_role_name(report->role), (unsigned long long)report->rows,
               (unsigned long long)report->cols, report->pcc, report->relative_rms_error, report->rms_error,
               report->max_abs_error);
        if (state->has_reference)
            printf("  src-relrms %.6e", report->source_relative_rms_error);
        printf("\n");
    }

    return 0;
}

static void print_summary(const gate_state *state)
{
    printf("\n%-14s %-8s %6s %13s %13s %13s %13s %13s%s  %s\n", "role", "format", "count", "worst PCC", "mean PCC",
           "mean relRMS", "worst relRMS", "worst maxabs", state->has_reference ? "   src relRMS" : "", "worst tensor");
    for (int role = 0; role < ROLE_COUNT; role++) {
        const role_summary *summary = &state->roles[role];

        if (summary->tensors == 0)
            continue;
        printf("%-14s %-8s %6u %13.9f %13.9f %13.6e %13.6e %13.6e",
               ft_tt_ledger_role_name((ft_tt_tensor_role)role), ft_tt_ledger_format_name(state->format),
               summary->tensors, summary->worst_pcc, summary->sum_pcc / (double)summary->tensors,
               summary->sum_relative_rms / (double)summary->tensors, summary->worst_relative_rms,
               summary->worst_max_abs);
        if (state->has_reference)
            printf(" %13.6e", summary->sum_source_relative_rms / (double)summary->tensors);
        printf("  %s\n", summary->worst_name);
    }
}

static int run_format(const char *path, const char *reference, ft_tt_tensor_format format, int verbose)
{
    gate_state state;
    int rc;

    memset(&state, 0, sizeof(state));
    state.verbose = verbose;
    state.format = format;
    state.has_reference = reference != NULL;

    printf("\n=== %s%s ===\n", ft_tt_ledger_format_name(format),
           reference ? " (scored against the reference checkpoint)" : "");
    rc = ft_tt_convert_measure_against(path, reference, format, on_tensor, &state);
    if (rc != FT_TT_CONVERT_OK) {
        fprintf(stderr, "ft_tt_pcc_gate: measurement failed with status %d\n", rc);
        return rc;
    }
    print_summary(&state);
    return FT_TT_CONVERT_OK;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s <input.gguf> [--format bf16|bfp8|bfp4|all] [--per-tensor]\n"
            "           [--reference <truth.gguf>]\n"
            "  default: --format all, summary only\n"
            "  --reference scores the round trip against ANOTHER checkpoint instead of\n"
            "              against the input's own values -- which is the only honest way\n"
            "              to measure a lossy source, e.g. a Q8_0 input against the BF16\n"
            "              release it was quantized from\n",
            program);
}

int main(int argc, char **argv)
{
    const char *input = NULL;
    const char *reference = NULL;
    const char *which = "all";
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            which = argv[++i];
        } else if (strcmp(argv[i], "--reference") == 0 && i + 1 < argc) {
            reference = argv[++i];
        } else if (strcmp(argv[i], "--per-tensor") == 0) {
            verbose = 1;
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 2;
        } else if (!input) {
            input = argv[i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!input) {
        usage(argv[0]);
        return 2;
    }

    if (strcmp(which, "all") == 0) {
        if (run_format(input, reference, FT_TT_FORMAT_BF16, verbose) != FT_TT_CONVERT_OK)
            return 1;
        if (run_format(input, reference, FT_TT_FORMAT_BFP8_B, verbose) != FT_TT_CONVERT_OK)
            return 1;
        if (run_format(input, reference, FT_TT_FORMAT_BFP4_B, verbose) != FT_TT_CONVERT_OK)
            return 1;
        return 0;
    }
    if (strcmp(which, "bf16") == 0)
        return run_format(input, reference, FT_TT_FORMAT_BF16, verbose) == FT_TT_CONVERT_OK ? 0 : 1;
    if (strcmp(which, "bfp8") == 0)
        return run_format(input, reference, FT_TT_FORMAT_BFP8_B, verbose) == FT_TT_CONVERT_OK ? 0 : 1;
    if (strcmp(which, "bfp4") == 0)
        return run_format(input, reference, FT_TT_FORMAT_BFP4_B, verbose) == FT_TT_CONVERT_OK ? 0 : 1;

    usage(argv[0]);
    return 2;
}
