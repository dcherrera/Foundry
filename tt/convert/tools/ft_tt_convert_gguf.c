/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_convert_gguf.c
 * @brief CLI: convert a GGUF checkpoint into a tiled Blackhole weight artifact.
 *
 * Runs on the Mac (spec §8.4). Streams: peak resident memory is set by the
 * strip buffers, not the model, and the acceptance measurement is
 *
 *     /usr/bin/time -l  ft_tt_convert_gguf model.gguf model.fttw     (macOS)
 *     /usr/bin/time -v  ft_tt_convert_gguf model.gguf model.fttw     (Linux)
 *
 * reading "maximum resident set size".
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "ft_tt_format_map.h"
#include "ft_tt_precision_ledger.h"
#include "ft_tt_weight_convert.h"

typedef struct {
    int verbose;
    int measure_pcc;
    uint32_t tensors;
    uint64_t source_bytes;
    uint64_t converted_bytes;
    double worst_pcc;
    char worst_name[FT_TT_ARTIFACT_NAME_BYTES];
} run_totals;

static int on_tensor(const ft_tt_convert_tensor_report *report, void *user)
{
    run_totals *totals = (run_totals *)user;

    totals->tensors++;
    totals->source_bytes += report->source_bytes;
    totals->converted_bytes += report->converted_bytes;

    if (totals->measure_pcc && report->pcc < totals->worst_pcc) {
        totals->worst_pcc = report->pcc;
        snprintf(totals->worst_name, sizeof(totals->worst_name), "%s", report->name);
    }

    if (totals->verbose) {
        printf("%-40s %-12s %-8s %8llu x %-8llu %10llu B", report->name, ft_tt_ledger_role_name(report->role),
               ft_tt_ledger_format_name(report->format), (unsigned long long)report->rows,
               (unsigned long long)report->cols, (unsigned long long)report->converted_bytes);
        if (totals->measure_pcc)
            printf("  pcc %.9f  rms %.3e  max %.3e", report->pcc, report->rms_error, report->max_abs_error);
        printf("\n");
    }

    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s <input.gguf> <output.fttw> [options]\n"
            "  --ledger NAME                   precision rung (default production):\n"
            "       fp32        every 2-D tensor fp32\n"
            "       bf16        every 2-D tensor bf16\n"
            "       production  the per-role allocation\n"
            "       bfp8        uniform Bfp8_b   (+0.175%% perplexity, measured)\n"
            "       bfp4        uniform Bfp4_b   (+40.2%% perplexity, measured — DO NOT SHIP)\n"
            "       moe_bfp6    bfp8, with routed MoE experts (ffn_*_exps) at Bfp6_b\n"
            "  --scope trunk|mtp|all           which tensors (default trunk):\n"
            "       trunk  everything but a trailing MTP / NextN block\n"
            "       mtp    ONLY that block, as Pantheon's `.mtp.fttw` (mtp.* names,\n"
            "              eh_proj split into eh_proj_e / eh_proj_h along its input)\n"
            "       all    every tensor under its own name\n"
            "  --rope-pairing source|split-half|split-half-reversed|interleaved-reversed\n"
            "       split-half permutes attn_q/attn_k output rows out of GGUF's\n"
            "       interleaved order into the split-half order the on-card RoPE\n"
            "       kernel needs. It changes no math, and the artifact header\n"
            "       records it so a host forward cannot rotate it the wrong way.\n"
            "       The two `-reversed` values are EXPERIMENTS, not checkpoint formats:\n"
            "       each reflects the rotary pair index, which regroups the block-float\n"
            "       exponent blocks over the same two tensors while leaving the score\n"
            "       exact in exact arithmetic. split-half-reversed keeps a pair's two\n"
            "       components in DIFFERENT exponent blocks; interleaved-reversed keeps\n"
            "       them in the SAME one, which is the crossed control that separates\n"
            "       the two things split-half changed at once. The on-card RoPE kernel\n"
            "       refuses both — its table is built in slot order.\n"
            "  --format-override SUB=FORMAT    EXPERIMENT. Encode every 2-D tensor whose\n"
            "       name contains SUB at FORMAT (fp32|bf16|bfp8_b|bfp4_b) instead of the\n"
            "       format the ledger chose, and name each one on stderr as it goes.\n"
            "       For measuring what share of the error ONE tensor family carries —\n"
            "       degrade it alone and read the residual-divergence plateau. The result\n"
            "       is NOT a shipping artifact and has no rung name; confirm what it\n"
            "       actually holds with ft_tt_fttw_list, which reads the written bytes.\n"
            "  --format-map FILE               per-tensor formats on top of the ledger rung.\n"
            "       One '<gguf tensor name> <bfp4_b|bfp8_b>' per line, '#' comments, blank\n"
            "       lines ignored. Only dense 2-D matmul weights may be named (attention,\n"
            "       GDN in/out, FFN, LM head, MTP eh_proj); an unknown or unmappable name\n"
            "       fails before any output is written. The artifact is named by its map,\n"
            "       not by the rung. Cannot be combined with --format-override.\n"
            "  --pcc                           round-trip every tensor and report PCC\n"
            "  --quiet                         totals only\n",
            program);
}

int main(int argc, char **argv)
{
    ft_tt_convert_options options;
    ft_tt_format_map format_map;
    char map_error[FT_TT_FORMAT_MAP_ERROR_BYTES];
    run_totals totals;
    const char *input = NULL;
    const char *output = NULL;
    int rc;

    memset(&options, 0, sizeof(options));
    memset(&totals, 0, sizeof(totals));
    memset(&format_map, 0, sizeof(format_map));
    options.mode = FT_TT_LEDGER_PRODUCTION;
    totals.verbose = 1;
    totals.worst_pcc = 2.0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ledger") == 0 && i + 1 < argc) {
            int mode = ft_tt_ledger_mode_from_name(argv[++i]);

            if (mode < 0) {
                fprintf(stderr, "%s: unknown ledger '%s'\n", argv[0], argv[i]);
                return 2;
            }
            options.mode = (ft_tt_ledger_mode)mode;
        } else if (strcmp(argv[i], "--rope-pairing") == 0 && i + 1 < argc) {
            const char *word = argv[++i];

            if (strcmp(word, "source") == 0) {
                options.rope_pairing = FT_TT_ROPE_PAIRING_SOURCE;
            } else if (strcmp(word, "split-half") == 0) {
                options.rope_pairing = FT_TT_ROPE_PAIRING_SPLIT_HALF;
            } else if (strcmp(word, "split-half-reversed") == 0) {
                options.rope_pairing = FT_TT_ROPE_PAIRING_SPLIT_HALF_REVERSED;
            } else if (strcmp(word, "interleaved-reversed") == 0) {
                options.rope_pairing = FT_TT_ROPE_PAIRING_INTERLEAVED_REVERSED;
            } else {
                fprintf(stderr, "%s: unknown rope pairing '%s'\n", argv[0], word);
                return 2;
            }
        } else if (strcmp(argv[i], "--format-override") == 0 && i + 1 < argc) {
            char *spec = argv[++i];
            char *equals = strchr(spec, '=');
            int format;

            if (!equals || equals == spec) {
                fprintf(stderr, "%s: --format-override wants SUBSTRING=FORMAT, got '%s'\n", argv[0],
                        spec);
                return 2;
            }
            *equals = '\0';
            format = ft_tt_ledger_format_from_name(equals + 1);
            if (format < 0) {
                fprintf(stderr, "%s: unknown format '%s' (fp32|bf16|bfp8_b|bfp4_b)\n", argv[0],
                        equals + 1);
                return 2;
            }
            options.format_override_name_contains = spec;
            options.format_override = (ft_tt_tensor_format)format;
        } else if (strcmp(argv[i], "--format-map") == 0 && i + 1 < argc) {
            if (options.format_map) {
                fprintf(stderr, "%s: --format-map given twice\n", argv[0]);
                ft_tt_format_map_free(&format_map);
                return 2;
            }
            if (ft_tt_format_map_load(argv[++i], &format_map, map_error, sizeof(map_error)) !=
                FT_TT_FORMAT_MAP_OK) {
                fprintf(stderr, "%s: --format-map: %s\n", argv[0], map_error);
                return 2;
            }
            options.format_map = &format_map;
        } else if (strcmp(argv[i], "--scope") == 0 && i + 1 < argc) {
            const char *word = argv[++i];

            if (strcmp(word, "trunk") == 0) {
                options.scope = FT_TT_CONVERT_SCOPE_TRUNK;
            } else if (strcmp(word, "mtp") == 0) {
                options.scope = FT_TT_CONVERT_SCOPE_MTP;
            } else if (strcmp(word, "all") == 0) {
                options.scope = FT_TT_CONVERT_SCOPE_ALL;
            } else {
                fprintf(stderr, "%s: unknown scope '%s' (trunk|mtp|all)\n", argv[0], word);
                return 2;
            }
        } else if (strcmp(argv[i], "--pcc") == 0) {
            options.measure_pcc = 1;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            totals.verbose = 0;
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 2;
        } else if (!input) {
            input = argv[i];
        } else if (!output) {
            output = argv[i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (!input || !output) {
        usage(argv[0]);
        ft_tt_format_map_free(&format_map);
        return 2;
    }
    if (options.format_map && options.format_override_name_contains) {
        fprintf(stderr, "%s: --format-map and --format-override cannot be combined: two overrides of one "
                        "tensor have no single obvious answer\n", argv[0]);
        ft_tt_format_map_free(&format_map);
        return 2;
    }

    totals.measure_pcc = options.measure_pcc;
    options.progress = on_tensor;
    options.progress_user = &totals;

    {
        const char *caution = ft_tt_ledger_mode_caution(options.mode);

        printf("ledger rung '%s', rope pairing '%s'\n", ft_tt_ledger_mode_name(options.mode),
               options.rope_pairing == FT_TT_ROPE_PAIRING_SPLIT_HALF ? "split-half" : "source");
        if (caution)
            fprintf(stderr, "\n*** %s ***\n\n", caution);
        if (options.format_override_name_contains)
            fprintf(stderr, "\n*** EXPERIMENT ARTIFACT: every 2-D tensor matching '%s' is forced "
                            "to %s, so this file is NOT the '%s' rung it would otherwise be and "
                            "must not be shipped or compared as one. ***\n\n",
                    options.format_override_name_contains,
                    ft_tt_ledger_format_name(options.format_override),
                    ft_tt_ledger_mode_name(options.mode));
        if (options.format_map) {
            char map_caution[FT_TT_FORMAT_MAP_SOURCE_BYTES + 512];

            ft_tt_format_map_caution(options.format_map, options.mode, map_caution, sizeof(map_caution));
            fprintf(stderr, "\n*** %s ***\n\n", map_caution);
        }
    }

    rc = ft_tt_convert_gguf(input, output, &options);
    ft_tt_format_map_free(&format_map);
    if (rc != FT_TT_CONVERT_OK) {
        fprintf(stderr, "%s: conversion failed with status %d\n", argv[0], rc);
        return 1;
    }

    printf("\n%u tensors  %.2f MiB source -> %.2f MiB artifact  (%.3fx)\n", totals.tensors,
           (double)totals.source_bytes / 1048576.0, (double)totals.converted_bytes / 1048576.0,
           totals.source_bytes ? (double)totals.converted_bytes / (double)totals.source_bytes : 0.0);
    if (totals.measure_pcc)
        printf("worst per-tensor PCC %.9f on %s\n", totals.worst_pcc, totals.worst_name);
    {
        /* The RSS bound is measured on every run, not only under /usr/bin/time: ru_maxrss is
         * KiB on Linux and bytes on macOS. */
        struct rusage usage;

        if (getrusage(RUSAGE_SELF, &usage) == 0)
#ifdef __APPLE__
            printf("peak RSS %.1f MiB\n", (double)usage.ru_maxrss / (1024.0 * 1024.0));
#else
            printf("peak RSS %.1f MiB\n", (double)usage.ru_maxrss / 1024.0);
#endif
    }

    return 0;
}
