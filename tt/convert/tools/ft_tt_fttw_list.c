/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_fttw_list.c
 * @brief CLI: report what a WRITTEN .fttw artifact actually contains — per-tensor
 *   format, role, layout and tile grid, plus a per-format and per-layout census.
 *
 * WHY IT PARSES THE BYTES ITSELF. "The converter chose format X" and "the artifact
 * holds format X" are different claims, and only the second one decides which Tensix
 * program directories a run needs — a resident matmul binary is bound to ONE weight
 * tile size. Reading the file back through `ft_tt_fttw_open()` would score our own
 * reader against our own writer, which DEV_NOTES.md §3 calls out as measuring
 * self-consistency rather than correctness. So this walks the documented on-disk
 * layout (a 40-byte header, then fixed 200-byte little-endian directory entries)
 * with nothing but stdio, and the only thing it shares with the converter is the
 * two enum-to-string helpers.
 *
 *   ft_tt_fttw_list <artifact.fttw> [--per-tensor]
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_precision_ledger.h"
#include "ft_tt_weight_convert.h"

/** Formats and layouts are small dense enums; these bound the census arrays. */
#define LIST_FORMAT_SLOTS 8u
#define LIST_LAYOUT_SLOTS 4u

/** One census row: how many tensors, and how many payload bytes they own. */
typedef struct {
    uint64_t tensors;
    uint64_t bytes;
} list_census;

static uint32_t read_u32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t read_u64(const unsigned char *p)
{
    return (uint64_t)read_u32(p) | ((uint64_t)read_u32(p + 4) << 32);
}

static const char *layout_name(uint32_t layout)
{
    switch (layout) {
    case FT_TT_LAYOUT_SOURCE:
        return "source";
    case FT_TT_LAYOUT_MATMUL_NMAJOR:
        return "nmajor";
    default:
        return "?";
    }
}

/** Bytes of one 32x32 tile in `format`, or 0 when the format is not tiled here. */
static uint32_t tile_bytes(uint32_t format)
{
    switch (format) {
    case FT_TT_FORMAT_FP32:
        return 4096u;
    case FT_TT_FORMAT_BF16:
        return 2048u;
    case FT_TT_FORMAT_BFP8_B:
        return 1088u;
    case FT_TT_FORMAT_BFP4_B:
        return 576u;
    case FT_TT_FORMAT_BFP6_B:
        return 832u;
    default:
        return 0u;
    }
}

/**
 * The provenance trailer, parsed the same way everything else here is: by hand, from the
 * documented offsets, with nothing borrowed from the writer.
 *
 * This is what makes "is the cached artifact the right one" answerable from OUTSIDE the
 * cache. A stale artifact and a fresh one are the same size and the same shape; the
 * trailer is the only thing that tells them apart, and reading it through the cache
 * library would be scoring the cache against itself.
 *
 * An artifact with no trailer is not an error — it was produced by hand, or before the
 * cache existed — so this says so and returns.
 */
static void print_provenance(FILE *file, uint64_t file_bytes)
{
    unsigned char trailer[64];

    if (file_bytes < sizeof(trailer) ||
        fseek(file, (long)(file_bytes - sizeof(trailer)), SEEK_SET) != 0 ||
        fread(trailer, 1, sizeof(trailer), file) != sizeof(trailer)) {
        printf("\n  provenance: none (the file is too short to carry a trailer)\n");
        return;
    }
    if (read_u32(trailer + 60) != 0x50545446u) { /* 'FTTP' */
        printf("\n  provenance: none — not a cache-managed artifact\n");
        return;
    }

    printf("\n  provenance trailer (the artifact's own account of where it came from)\n");
    printf("    trailer version %u, %u B; converter behaviour stamp 0x%08x\n",
           read_u32(trailer + 24), read_u32(trailer + 52), read_u32(trailer + 48));
    printf("    source: %llu B, tensor-directory digest 0x%016llx\n",
           (unsigned long long)read_u64(trailer + 0), (unsigned long long)read_u64(trailer + 8));
    printf("    written at rung %s, rope pairing %u, artifact version %u, flags 0x%08x, "
           "%u tensors, %llu B of FTTW\n",
           ft_tt_ledger_mode_name((ft_tt_ledger_mode)read_u32(trailer + 32)),
           read_u32(trailer + 36), read_u32(trailer + 28), read_u32(trailer + 40),
           read_u32(trailer + 44), (unsigned long long)read_u64(trailer + 16));

    /* The one check this tool can make with no checkpoint present: does the file's own
     * length agree with the trailer's account of it. */
    if (read_u64(trailer + 16) + sizeof(trailer) != file_bytes)
        printf("    MISMATCH: %llu B of FTTW plus a %zu B trailer is %llu B, but the file is "
               "%llu B\n",
               (unsigned long long)read_u64(trailer + 16), sizeof(trailer),
               (unsigned long long)(read_u64(trailer + 16) + sizeof(trailer)),
               (unsigned long long)file_bytes);
}

static void print_census(const char *title, const list_census *census, int with_tile)
{
    printf("\n  census — %s\n", title);
    for (uint32_t f = 0; f < LIST_FORMAT_SLOTS; f++) {
        if (census[f].tensors == 0u)
            continue;
        printf("    %-8s %4llu tensors  %12llu B  %8.1f MiB", ft_tt_ledger_format_name((ft_tt_tensor_format)f),
               (unsigned long long)census[f].tensors, (unsigned long long)census[f].bytes,
               (double)census[f].bytes / (1024.0 * 1024.0));
        if (with_tile)
            printf("   tile %u B", tile_bytes(f));
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    list_census by_format[LIST_FORMAT_SLOTS];
    list_census nmajor_by_format[LIST_FORMAT_SLOTS];
    uint64_t by_layout[LIST_LAYOUT_SLOTS];
    unsigned char header[FT_TT_ARTIFACT_HEADER_BYTES];
    unsigned char entry[FT_TT_ARTIFACT_ENTRY_BYTES];
    uint64_t directory_offset, total_bytes = 0;
    uint64_t expert_stacks = 0, expert_bytes = 0, stride_mismatches = 0;
    uint32_t tensor_count, index;
    int per_tensor = 0;
    FILE *file;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <artifact.fttw> [--per-tensor]\n", argv[0]);
        return 2;
    }
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--per-tensor") == 0)
            per_tensor = 1;
    }

    file = fopen(argv[1], "rb");
    if (!file) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    if (fread(header, 1, sizeof(header), file) != sizeof(header)) {
        fprintf(stderr, "%s: short header\n", argv[1]);
        fclose(file);
        return 1;
    }
    tensor_count = read_u32(header + 8);
    directory_offset = read_u64(header + 16);
    printf("artifact %s\n  magic 0x%08x  version %u  %u tensors  %llu B total\n", argv[1],
           read_u32(header), read_u32(header + 4), tensor_count,
           (unsigned long long)read_u64(header + 32));
    {
        /* The flags change what the BYTES MEAN, so they come before anything about the
         * payload: an artifact with the rotary permutation and one without it are the
         * same size, the same shape and the same tensor list, and this line is the only
         * difference visible from outside the file. */
        uint32_t flags = read_u32(header + 12);

        printf("  flags 0x%08x  rope=%s%s%s\n", flags,
               (flags & FT_TT_ARTIFACT_FLAG_ROPE_SPLIT_HALF) ? "split-half" : "interleaved(source)",
               (flags & FT_TT_ARTIFACT_FLAG_ROPE_PAIRS_REVERSED) ? ", pair index REVERSED" : "",
               (flags & ~FT_TT_ARTIFACT_FLAGS_KNOWN) ? "  UNKNOWN BITS SET" : "");
        if (flags & ~FT_TT_ARTIFACT_FLAGS_KNOWN) {
            fprintf(stderr, "%s: header flags 0x%08x carry bits this build does not know; the "
                            "payload may not mean what this tool would print\n",
                    argv[1], flags);
            fclose(file);
            return 1;
        }
    }

    if (fseek(file, (long)directory_offset, SEEK_SET) != 0) {
        fprintf(stderr, "%s: directory at %llu is not reachable\n", argv[1],
                (unsigned long long)directory_offset);
        fclose(file);
        return 1;
    }

    memset(by_format, 0, sizeof(by_format));
    memset(nmajor_by_format, 0, sizeof(nmajor_by_format));
    memset(by_layout, 0, sizeof(by_layout));

    for (index = 0; index < tensor_count; index++) {
        uint32_t format, role, layout, tile_rows, tile_cols, n_dims, experts = 1u;
        uint64_t byte_count, stride;
        const char *name;

        if (fread(entry, 1, sizeof(entry), file) != sizeof(entry)) {
            fprintf(stderr, "%s: directory truncated at entry %u of %u\n", argv[1], index,
                    tensor_count);
            fclose(file);
            return 1;
        }
        name = (const char *)entry;
        format = read_u32(entry + FT_TT_ARTIFACT_NAME_BYTES);
        role = read_u32(entry + FT_TT_ARTIFACT_NAME_BYTES + 4);
        n_dims = read_u32(entry + FT_TT_ARTIFACT_NAME_BYTES + 8);
        tile_rows = read_u32(entry + FT_TT_ARTIFACT_NAME_BYTES + 12);
        tile_cols = read_u32(entry + FT_TT_ARTIFACT_NAME_BYTES + 16);
        layout = read_u32(entry + FT_TT_ARTIFACT_NAME_BYTES + 20);
        byte_count = read_u64(entry + FT_TT_ARTIFACT_NAME_BYTES + 24 + 32 + 8);

        if (format >= LIST_FORMAT_SLOTS || layout >= LIST_LAYOUT_SLOTS) {
            fprintf(stderr, "%s: entry %u (%.*s) has format %u layout %u, which this build "
                            "does not model — refusing rather than counting it as something\n",
                    argv[1], index, FT_TT_ARTIFACT_NAME_BYTES, name, format, layout);
            fclose(file);
            return 1;
        }
        /* v3: a 3-D routed-expert stack's grid is ONE expert's, and expert e is the slice at
         * e * stride. Read back out of the file, so a wrong stride shows here. */
        stride = (uint64_t)tile_rows * tile_cols * tile_bytes(format);
        if (n_dims == 3u && (role == FT_TT_ROLE_EXPERT_UP || role == FT_TT_ROLE_EXPERT_DOWN)) {
            experts = (uint32_t)read_u64(entry + FT_TT_ARTIFACT_NAME_BYTES + 24 + 16);
            expert_stacks++;
            expert_bytes += byte_count;
            if ((uint64_t)experts * stride != byte_count) {
                stride_mismatches++;
                fprintf(stderr, "%s: %.*s: %u experts x stride %llu B != %llu B\n", argv[1],
                        FT_TT_ARTIFACT_NAME_BYTES, name, experts, (unsigned long long)stride,
                        (unsigned long long)byte_count);
            }
        }
        by_format[format].tensors++;
        by_format[format].bytes += byte_count;
        by_layout[layout]++;
        total_bytes += byte_count;
        if (layout == FT_TT_LAYOUT_MATMUL_NMAJOR) {
            nmajor_by_format[format].tensors++;
            nmajor_by_format[format].bytes += byte_count;
        }
        if (per_tensor && experts > 1u)
            printf("  %-44.*s %-8s %-14s %-7s %4u x %-5u tile %4u B  %12llu B  %u experts x %llu B\n",
                   FT_TT_ARTIFACT_NAME_BYTES, name, ft_tt_ledger_format_name((ft_tt_tensor_format)format),
                   ft_tt_ledger_role_name((ft_tt_tensor_role)role), layout_name(layout), tile_rows, tile_cols,
                   tile_bytes(format), (unsigned long long)byte_count, experts, (unsigned long long)stride);
        else if (per_tensor)
            printf("  %-44.*s %-8s %-14s %-7s %4u x %-5u tile %4u B  %12llu B\n",
                   FT_TT_ARTIFACT_NAME_BYTES, name, ft_tt_ledger_format_name((ft_tt_tensor_format)format),
                   ft_tt_ledger_role_name((ft_tt_tensor_role)role), layout_name(layout), tile_rows, tile_cols,
                   tile_bytes(format), (unsigned long long)byte_count);
    }

    {
        uint64_t file_bytes = 0;

        if (fseek(file, 0, SEEK_END) == 0) {
            long end = ftell(file);

            if (end > 0)
                file_bytes = (uint64_t)end;
        }
        print_provenance(file, file_bytes);
    }
    fclose(file);

    print_census("ALL tensors", by_format, 0);
    /* The MATMUL_NMAJOR rows are the ones that decide which Tensix program builds a run
     * needs: only these tensors are ever fed to a matmul, and a resident matmul binary
     * is bound to one weight tile size. */
    print_census("MATMUL_NMAJOR only — this is the program-build requirement", nmajor_by_format, 1);

    printf("\n  layouts: source %llu, nmajor %llu\n",
           (unsigned long long)by_layout[FT_TT_LAYOUT_SOURCE],
           (unsigned long long)by_layout[FT_TT_LAYOUT_MATMUL_NMAJOR]);
    printf("  payload total %llu B (%.1f MiB)\n", (unsigned long long)total_bytes,
           (double)total_bytes / (1024.0 * 1024.0));
    printf("  payload span (directory - data offset, padding included) %llu B\n",
           (unsigned long long)(directory_offset - read_u64(header + 24)));
    if (expert_stacks)
        printf("  routed-expert stacks: %llu, %llu B (%.2f GiB); stride mismatches %llu\n",
               (unsigned long long)expert_stacks, (unsigned long long)expert_bytes,
               (double)expert_bytes / (1024.0 * 1024.0 * 1024.0), (unsigned long long)stride_mismatches);
    return stride_mismatches ? 1 : 0;
}
