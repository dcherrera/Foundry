/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_fttw_splice.c
 * @brief CLI: splice MXTL tile files into a copy of an FTTW artifact; extract an artifact's
 *   entries as a tile file; print a spliced artifact's record. See ft_tt_artifact_splice.h.
 *
 *   ft_tt_fttw_splice <source.fttw> --tile-file A.mxtl [--tile-file B.mxtl ...] --out <out.fttw>
 *   ft_tt_fttw_splice --extract <artifact.fttw> --tensor NAME [--tensor NAME ...] --out <tiles.mxtl>
 *   ft_tt_fttw_splice --show <spliced.fttw>
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_sha256.h"
#include "ft_tt_artifact_splice.h"
#include "ft_tt_precision_ledger.h"

/** Repeated flags (--tile-file, --tensor) beyond this are refused, not truncated. */
#define SPLICE_CLI_MAX_LIST 1024u

static void usage(const char *program)
{
    fprintf(stderr,
            "usage:\n"
            "  %s <source.fttw> --tile-file A.mxtl [--tile-file B.mxtl ...] --out <out.fttw>\n"
            "      Copy the source with each tile-file tensor's payload and format replaced. Every\n"
            "      other byte up to the directory is laid out as the converter lays it out; a splice\n"
            "      record (names + payload sha256) follows the directory. Refuses unknown names,\n"
            "      roles a format map may not move, formats other than bfp4_b/bfp8_b, a layout or a\n"
            "      size the converter would not write, duplicates, and --out equal to an input.\n"
            "  %s --extract <artifact.fttw> --tensor NAME [--tensor NAME ...] --out <tiles.mxtl>\n"
            "      Write the named entries' payloads, formats and layouts as an MXTL tile file.\n"
            "  %s --show <spliced.fttw>\n"
            "      Print the splice record.\n",
            program, program, program);
}

static int show_record(const char *path)
{
    ft_tt_splice_record record;
    char hex[2u * FT_SHA256_BYTES + 1u];
    int rc;

    memset(&record, 0, sizeof(record));
    rc = ft_tt_splice_record_read(path, &record);
    if (rc != FT_TT_SPLICE_OK) {
        fprintf(stderr, "%s: %s\n", path,
                rc == FT_TT_SPLICE_EABSENT ? "carries no splice record" : "splice record unreadable or inconsistent");
        return 1;
    }
    ft_sha256_hex(record.base_directory_sha256, hex);
    printf("splice record of %s\n  %u tensors replaced; base FTTW %llu B, directory sha256 %s; base provenance "
           "trailer %s\n",
           path, record.entry_count, (unsigned long long)record.base_total_bytes, hex,
           (record.flags & FT_TT_SPLICE_FLAG_BASE_PROVENANCE) ? "kept" : "none");
    for (uint32_t i = 0; i < record.entry_count; i++) {
        ft_sha256_hex(record.entries[i].sha256, hex);
        printf("  %-44s %-7s layout %u %12llu B  %s\n", record.entries[i].name,
               ft_tt_ledger_format_name((ft_tt_tensor_format)record.entries[i].format), record.entries[i].layout,
               (unsigned long long)record.entries[i].n_bytes, hex);
    }
    ft_tt_splice_record_free(&record);
    return 0;
}

/** Append `value` to a bounded list; 0 on success. */
static int list_add(const char **list, uint32_t *count, const char *value)
{
    if (*count >= SPLICE_CLI_MAX_LIST)
        return -1;
    list[(*count)++] = value;
    return 0;
}

int main(int argc, char **argv)
{
    static const char *tiles[SPLICE_CLI_MAX_LIST];
    static const char *tensors[SPLICE_CLI_MAX_LIST];
    char error[FT_TT_SPLICE_ERROR_BYTES] = "";
    const char *source = NULL, *extract = NULL, *output = NULL;
    uint32_t tile_count = 0, tensor_count = 0;
    int rc;

    if (argc == 3 && strcmp(argv[1], "--show") == 0)
        return show_record(argv[2]);
    for (int i = 1; i < argc; i++) {
        int bad = 0;

        if (strcmp(argv[i], "--tile-file") == 0 && i + 1 < argc)
            bad = list_add(tiles, &tile_count, argv[++i]);
        else if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc)
            bad = list_add(tensors, &tensor_count, argv[++i]);
        else if (strcmp(argv[i], "--extract") == 0 && i + 1 < argc && !extract)
            extract = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc && !output)
            output = argv[++i];
        else if (argv[i][0] != '-' && !source)
            source = argv[i];
        else
            bad = 1;
        if (bad) {
            usage(argv[0]);
            return 2;
        }
    }
    if (!output || (extract ? (source || tile_count || !tensor_count) : (!source || !tile_count || tensor_count))) {
        usage(argv[0]);
        return 2;
    }

    if (extract)
        rc = ft_tt_artifact_extract_tiles(extract, tensors, tensor_count, output, error, sizeof(error));
    else
        rc = ft_tt_artifact_splice(source, tiles, tile_count, output, error, sizeof(error));
    if (rc != FT_TT_SPLICE_OK) {
        fprintf(stderr, "%s: refused (%d): %s\n", argv[0], rc, error);
        return 1;
    }
    if (extract)
        printf("wrote %u tensors from %s to %s\n", tensor_count, extract, output);
    else
        return show_record(output);
    return 0;
}
