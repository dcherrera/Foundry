/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file route_agreement_bfp4.c
 * @brief How often the two routes to a Bfp4_b tile agree, per tensor class: the converter's
 *   Q8_0 -> Bfp4_b and the load-time repack's Q8_0 -> Bfp8_b -> Bfp4_b.
 *
 * The repack knob (Pantheon PN_MIXW_BFP4) builds its Bfp4 tiles from the shipped Bfp8 tiles;
 * a shipped mixed artifact packs them straight from the Q8_0 source. The second rounding can
 * move a value that sat just below a Bfp4 midpoint onto it, and the tie then rounds the
 * other way. This measures how much that happens on real (or realistic) weights.
 *
 *     route_agreement_bfp4                         synthetic Qwen3.8-shaped Q8_0 checkpoint
 *     route_agreement_bfp4 model.gguf [--out-dir DIR] [--keep]
 *
 * It converts the checkpoint at the bfp8 and the bfp4 rungs (scope all) into DIR (default
 * build/), repacks every Bfp8_b tile to Bfp4_b and compares it with the converter's Bfp4_b
 * tile: tiles identical, bytes equal (exponent bytes apart), values equal. The two artifacts
 * are deleted afterwards unless --keep: for a real model they are ~1.06x and ~0.56x the
 * Q8_0 size, so DIR needs that much free disk.
 *
 * Before measuring, it proves the comparison can see both answers: a planted tile both
 * routes encode identically must score 100%, and a planted double-rounding tile must score
 * exactly the values it was built to move. Exit status is non-zero if either fails.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_bfp.h"
#include "ft_tt_precision_ledger.h"
#include "ft_tt_weight_convert.h"
#include "synthetic_hybrid_checkpoint.h"

#define TILE_VALUES 1024u
#define BFP8_TILE_BYTES 1088u
#define BFP4_TILE_BYTES 576u
#define CHUNK_TILES 256u

/** Agreement counters for one class (or the whole model). */
typedef struct {
    uint64_t tensors;
    uint64_t tiles;
    uint64_t tiles_identical;
    uint64_t exponent_bytes_equal; /**< of tiles * 64 */
    uint64_t bytes_equal;          /**< of tiles * 576 */
    uint64_t values_equal;         /**< of tiles * 1024, decoded and compared bit for bit */
} route_stats;

/** Scratch for one chunk of tiles. */
typedef struct {
    float *from_bfp8;
    float *direct;
    unsigned char *repacked;
} route_scratch;

static int scratch_alloc(route_scratch *s)
{
    s->from_bfp8 = malloc((size_t)CHUNK_TILES * TILE_VALUES * sizeof(float));
    s->direct = malloc((size_t)CHUNK_TILES * TILE_VALUES * sizeof(float));
    s->repacked = malloc((size_t)CHUNK_TILES * BFP4_TILE_BYTES);
    return s->from_bfp8 && s->direct && s->repacked ? 0 : -1;
}

static void scratch_free(route_scratch *s)
{
    free(s->from_bfp8);
    free(s->direct);
    free(s->repacked);
}

/**
 * Repack `tiles` (<= CHUNK_TILES) Bfp8_b tiles to Bfp4_b and score them against the
 * converter's own Bfp4_b tiles.
 */
static int compare_route_tiles(const unsigned char *bfp8, const unsigned char *bfp4, size_t tiles,
                               route_scratch *scratch, route_stats *stats)
{
    if (ft_bfp_unpack_tiles(FT_BFP_BFP8_B, bfp8, tiles, scratch->from_bfp8) != FT_BFP_OK ||
        ft_bfp_pack_tiles(FT_BFP_BFP4_B, scratch->from_bfp8, tiles, scratch->repacked) != FT_BFP_OK ||
        ft_bfp_unpack_tiles(FT_BFP_BFP4_B, scratch->repacked, tiles, scratch->from_bfp8) != FT_BFP_OK ||
        ft_bfp_unpack_tiles(FT_BFP_BFP4_B, bfp4, tiles, scratch->direct) != FT_BFP_OK)
        return -1;
    for (size_t t = 0; t < tiles; t++) {
        const unsigned char *a = scratch->repacked + t * BFP4_TILE_BYTES;
        const unsigned char *b = bfp4 + t * BFP4_TILE_BYTES;
        uint64_t equal = 0;

        for (size_t i = 0; i < BFP4_TILE_BYTES; i++) {
            if (a[i] != b[i])
                continue;
            equal++;
            if (i < FT_BFP_TILE_EXPONENT_BYTES)
                stats->exponent_bytes_equal++;
        }
        stats->bytes_equal += equal;
        if (equal == BFP4_TILE_BYTES)
            stats->tiles_identical++;
        stats->tiles++;
    }
    for (size_t v = 0; v < tiles * TILE_VALUES; v++)
        if (memcmp(&scratch->from_bfp8[v], &scratch->direct[v], sizeof(float)) == 0)
            stats->values_equal++;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* The comparison can see both answers                                       */
/* ------------------------------------------------------------------------- */

/** Score one planted fp32 tile: route A packs it to Bfp4 directly, route B via Bfp8. */
static int score_planted_tile(const float *tile, route_scratch *scratch, route_stats *stats)
{
    unsigned char bfp8[BFP8_TILE_BYTES];
    unsigned char bfp4[BFP4_TILE_BYTES];

    if (ft_bfp_pack_tiles(FT_BFP_BFP8_B, tile, 1, bfp8) != FT_BFP_OK ||
        ft_bfp_pack_tiles(FT_BFP_BFP4_B, tile, 1, bfp4) != FT_BFP_OK)
        return -1;
    return compare_route_tiles(bfp8, bfp4, 1, scratch, stats);
}

/**
 * Two planted tiles. Every 16-value block leads with 1.75, so the shared exponent is 2^0
 * and the Bfp4 step is 0.25. EXACT fills the rest with multiples of 0.25 (both routes
 * exact). DOUBLE fills it with 0.875 - 2^-8: direct rounds it to 0.75, while Bfp8 (step
 * 2^-6) rounds it up onto the 0.875 midpoint, whose tie goes to even, 1.0. So DOUBLE agrees
 * on exactly the 64 leading values.
 */
static int self_check(route_scratch *scratch)
{
    float exact[TILE_VALUES], doubled[TILE_VALUES];
    route_stats a, b;
    int failures = 0;

    for (uint32_t v = 0; v < TILE_VALUES; v++) {
        int lead = v % FT_BFP_BLOCK_VALUES == 0;

        exact[v] = lead ? 1.75f : (float)((int)(v % 15u) - 7) * 0.25f;
        doubled[v] = lead ? 1.75f : 0.87109375f;
    }
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    if (score_planted_tile(exact, scratch, &a) != 0 || score_planted_tile(doubled, scratch, &b) != 0)
        return 1;
    printf("self-check: exact tile %" PRIu64 "/1 identical, %" PRIu64 "/1024 values equal\n", a.tiles_identical,
           a.values_equal);
    printf("self-check: double-rounding tile %" PRIu64 "/1 identical, %" PRIu64 "/1024 values equal (want 64)\n",
           b.tiles_identical, b.values_equal);
    failures += a.tiles_identical != 1 || a.values_equal != TILE_VALUES;
    failures += b.tiles_identical != 0 || b.values_equal != TILE_VALUES / FT_BFP_BLOCK_VALUES;
    return failures;
}

/* ------------------------------------------------------------------------- */
/* Two artifacts, entry by entry                                             */
/* ------------------------------------------------------------------------- */

typedef struct {
    FILE *file;
    ft_tt_artifact_header header;
    ft_tt_artifact_entry *entries;
} artifact_stream;

static int artifact_open(const char *path, artifact_stream *a)
{
    memset(a, 0, sizeof(*a));
    a->file = fopen(path, "rb");
    if (!a->file || fread(&a->header, sizeof(a->header), 1, a->file) != 1 ||
        a->header.magic != FT_TT_ARTIFACT_MAGIC)
        return -1;
    a->entries = calloc(a->header.tensor_count, sizeof(*a->entries));
    if (!a->entries || fseek(a->file, (long)a->header.directory_offset, SEEK_SET) != 0 ||
        fread(a->entries, sizeof(*a->entries), a->header.tensor_count, a->file) != a->header.tensor_count)
        return -1;
    return 0;
}

static void artifact_close(artifact_stream *a)
{
    if (a->file)
        fclose(a->file);
    free(a->entries);
    memset(a, 0, sizeof(*a));
}

static int read_tiles(artifact_stream *a, uint64_t offset, size_t bytes, unsigned char *out)
{
    return fseek(a->file, (long)offset, SEEK_SET) == 0 && fread(out, 1, bytes, a->file) == bytes ? 0 : -1;
}

/** Score one tensor present at Bfp8_b in `bfp8` and Bfp4_b in `bfp4`, chunk by chunk. */
static int compare_entry(artifact_stream *bfp8, const ft_tt_artifact_entry *e8, artifact_stream *bfp4,
                         const ft_tt_artifact_entry *e4, route_scratch *scratch, unsigned char *chunk8,
                         unsigned char *chunk4, route_stats *stats)
{
    uint64_t tiles = e8->byte_count / BFP8_TILE_BYTES;

    if (e4->byte_count != tiles * BFP4_TILE_BYTES || e8->byte_count % BFP8_TILE_BYTES != 0u)
        return -1;
    for (uint64_t first = 0; first < tiles; first += CHUNK_TILES) {
        size_t count = (size_t)(tiles - first < CHUNK_TILES ? tiles - first : CHUNK_TILES);

        if (read_tiles(bfp8, e8->data_offset + first * BFP8_TILE_BYTES, count * BFP8_TILE_BYTES, chunk8) != 0 ||
            read_tiles(bfp4, e4->data_offset + first * BFP4_TILE_BYTES, count * BFP4_TILE_BYTES, chunk4) != 0 ||
            compare_route_tiles(chunk8, chunk4, count, scratch, stats) != 0)
            return -1;
    }
    stats->tensors++;
    return 0;
}

static void print_stats_row(const char *label, const route_stats *s)
{
    printf("%-16s %7" PRIu64 " %9" PRIu64 "  %9.4f%%  %9.4f%%  %9.4f%%  %9.4f%%\n", label, s->tensors, s->tiles,
           s->tiles ? 100.0 * (double)s->tiles_identical / (double)s->tiles : 0.0,
           s->tiles ? 100.0 * (double)s->exponent_bytes_equal / (double)(s->tiles * FT_BFP_TILE_EXPONENT_BYTES) : 0.0,
           s->tiles ? 100.0 * (double)s->bytes_equal / (double)(s->tiles * BFP4_TILE_BYTES) : 0.0,
           s->tiles ? 100.0 * (double)s->values_equal / (double)(s->tiles * TILE_VALUES) : 0.0);
}

static void accumulate(route_stats *into, const route_stats *from)
{
    into->tensors += from->tensors;
    into->tiles += from->tiles;
    into->tiles_identical += from->tiles_identical;
    into->exponent_bytes_equal += from->exponent_bytes_equal;
    into->bytes_equal += from->bytes_equal;
    into->values_equal += from->values_equal;
}

/** Compare every Bfp8_b entry of `bfp8` with its Bfp4_b twin, and print the per-class table. */
static int compare_artifacts(const char *bfp8_path, const char *bfp4_path, route_scratch *scratch)
{
    route_stats per_role[FT_TT_ROLE_COUNT], total;
    artifact_stream bfp8, bfp4;
    unsigned char *chunk8 = malloc((size_t)CHUNK_TILES * BFP8_TILE_BYTES);
    unsigned char *chunk4 = malloc((size_t)CHUNK_TILES * BFP4_TILE_BYTES);
    uint64_t expected_tiles = 0;
    int rc = artifact_open(bfp8_path, &bfp8) == 0 && artifact_open(bfp4_path, &bfp4) == 0 && chunk8 && chunk4 ? 0 : -1;

    memset(per_role, 0, sizeof(per_role));
    memset(&total, 0, sizeof(total));
    for (uint32_t i = 0; rc == 0 && i < bfp8.header.tensor_count; i++) {
        const ft_tt_artifact_entry *e8 = &bfp8.entries[i];
        const ft_tt_artifact_entry *e4 = i < bfp4.header.tensor_count ? &bfp4.entries[i] : NULL;

        if (e8->format != FT_TT_FORMAT_BFP8_B)
            continue;
        expected_tiles += e8->byte_count / BFP8_TILE_BYTES;
        if (!e4 || strcmp(e4->name, e8->name) != 0 || e4->format != FT_TT_FORMAT_BFP4_B || e8->role >= FT_TT_ROLE_COUNT)
            rc = -1;
        else
            rc = compare_entry(&bfp8, e8, &bfp4, e4, scratch, chunk8, chunk4, &per_role[e8->role]);
    }
    if (rc == 0) {
        printf("\n%-16s %7s %9s  %10s  %10s  %10s  %10s\n", "class", "tensors", "tiles", "tiles=", "exp bytes=",
               "bytes=", "values=");
        for (uint32_t r = 0; r < FT_TT_ROLE_COUNT; r++) {
            if (!per_role[r].tiles)
                continue;
            print_stats_row(ft_tt_ledger_role_name((ft_tt_tensor_role)r), &per_role[r]);
            accumulate(&total, &per_role[r]);
        }
        print_stats_row("all", &total);
        if (total.tiles != expected_tiles || total.tiles == 0)
            rc = -1;
    }
    artifact_close(&bfp8);
    artifact_close(&bfp4);
    free(chunk8);
    free(chunk4);
    return rc;
}

static int convert_rung(const char *gguf, const char *out, ft_tt_ledger_mode mode)
{
    ft_tt_convert_options options;

    memset(&options, 0, sizeof(options));
    options.mode = mode;
    options.scope = FT_TT_CONVERT_SCOPE_ALL;
    return ft_tt_convert_gguf(gguf, out, &options);
}

int main(int argc, char **argv)
{
    const char *gguf = NULL;
    const char *out_dir = "build";
    char bfp8_path[1024], bfp4_path[1024], synthetic[1024];
    route_scratch scratch;
    int keep = 0, failures, rc = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc)
            out_dir = argv[++i];
        else if (strcmp(argv[i], "--keep") == 0)
            keep = 1;
        else if (argv[i][0] != '-' && !gguf)
            gguf = argv[i];
        else {
            fprintf(stderr, "usage: %s [model.gguf] [--out-dir DIR] [--keep]\n", argv[0]);
            return 2;
        }
    }
    snprintf(bfp8_path, sizeof(bfp8_path), "%s/route_bfp8.fttw", out_dir);
    snprintf(bfp4_path, sizeof(bfp4_path), "%s/route_bfp4.fttw", out_dir);
    memset(&scratch, 0, sizeof(scratch));
    if (scratch_alloc(&scratch) != 0)
        goto done;

    failures = self_check(&scratch);
    if (failures) {
        printf("self-check FAILED (%d): the comparison cannot be trusted\n", failures);
        goto done;
    }
    if (!gguf) {
        snprintf(synthetic, sizeof(synthetic), "%s/route_synthetic.gguf", out_dir);
        if (synthetic_hybrid_checkpoint_write(synthetic) != 0)
            goto done;
        gguf = synthetic;
        printf("synthetic Qwen3.8-shaped Q8_0 checkpoint (realistic values): %s\n", gguf);
    }
    if (convert_rung(gguf, bfp8_path, FT_TT_LEDGER_BFP8) != FT_TT_CONVERT_OK ||
        convert_rung(gguf, bfp4_path, FT_TT_LEDGER_BFP4) != FT_TT_CONVERT_OK) {
        fprintf(stderr, "%s: conversion failed\n", argv[0]);
        goto done;
    }
    rc = compare_artifacts(bfp8_path, bfp4_path, &scratch) == 0 ? 0 : 1;
    if (rc)
        fprintf(stderr, "%s: the two artifacts do not pair up entry for entry\n", argv[0]);
    if (!keep) {
        remove(bfp8_path);
        remove(bfp4_path);
    }
done:
    scratch_free(&scratch);
    return rc;
}
