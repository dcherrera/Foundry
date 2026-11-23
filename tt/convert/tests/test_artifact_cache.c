/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_artifact_cache.c
 * @brief Off-hardware gates for the automatic rung selector, the artifact provenance
 *   trailer and the convert-on-load cache.
 *
 * Nothing here touches a card. Every fixture is a synthetic GGUF written by this file, so
 * the whole suite runs anywhere in under a second and a mutation battery is cheap.
 *
 * WHAT THIS SUITE IS FOR, stated so it is not mistaken for a unit-test formality: the two
 * failures it exists to prevent have both already happened in this project. A green
 * measurement taken on a STALE artifact (twice), and a fallback that shipped without
 * saying it had fallen back (twice). So the assertions that matter most are the ones that
 * prove a stale artifact is DETECTED and that a refusal is a refusal.
 *
 * @status None
 * @issues None
 * @todo None
 */

#define _DEFAULT_SOURCE

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ft_bfp.h"
#include "ft_gguf_dequant.h"
#include "ft_gguf_map.h"
#include "ft_tt_artifact_cache.h"
#include "ft_tt_artifact_provenance.h"
#include "ft_tt_capacity_plan.h"
#include "ft_tt_precision_ledger.h"
#include "ft_tt_weight_convert.h"

static int checks = 0;
static int failures = 0;

static void check(int condition, const char *what)
{
    checks++;
    if (condition) {
        printf("  ok    %s\n", what);
        return;
    }
    failures++;
    printf("  FAIL  %s\n", what);
}

static void check_u64(uint64_t got, uint64_t want, const char *what)
{
    checks++;
    if (got == want) {
        printf("  ok    %s\n", what);
        return;
    }
    failures++;
    printf("  FAIL  %s (got %llu, want %llu)\n", what, (unsigned long long)got,
           (unsigned long long)want);
}

static void section(const char *name)
{
    printf("\n%s\n", name);
}

/* ------------------------------------------------------------------------- */
/* A synthetic checkpoint                                                     */
/* ------------------------------------------------------------------------- */

/**
 * A tensor in the fixture model.
 *
 * The shapes are deliberately large enough that bfp8 and bfp4 differ by megabytes: a
 * budget test whose two rungs are a few kilobytes apart cannot tell a real selection from
 * a rounding accident.
 */
typedef struct {
    const char *name;
    uint32_t n_dims;
    uint64_t dims[2];
    uint32_t ggml_type;
} fixture_tensor;

static const fixture_tensor FIXTURE[] = {
    {"token_embd.weight", 2, {1024, 2048}, FT_GGML_TYPE_Q8_0},
    {"blk.0.attn_norm.weight", 1, {1024, 1}, FT_GGML_TYPE_F32},
    {"blk.0.attn_q.weight", 2, {1024, 1024}, FT_GGML_TYPE_Q8_0},
    {"blk.0.attn_k.weight", 2, {1024, 1024}, FT_GGML_TYPE_Q8_0},
    {"blk.0.attn_v.weight", 2, {1024, 1024}, FT_GGML_TYPE_Q8_0},
    {"blk.0.attn_output.weight", 2, {1024, 1024}, FT_GGML_TYPE_Q8_0},
    {"blk.0.ffn_norm.weight", 1, {1024, 1}, FT_GGML_TYPE_F32},
    {"blk.0.ffn_gate.weight", 2, {1024, 2048}, FT_GGML_TYPE_Q8_0},
    {"blk.0.ffn_up.weight", 2, {1024, 2048}, FT_GGML_TYPE_Q8_0},
    {"blk.0.ffn_down.weight", 2, {2048, 1024}, FT_GGML_TYPE_Q8_0},
    /* 1000 fp32 values = 4000 B, which is NOT a multiple of the artifact's 64-byte
     * alignment. Every tiled format's tile size already divides by 64, so without a
     * 1-D tensor of an awkward length the alignment padding is never exercised and a
     * planner that forgot it would agree with the converter by accident. */
    {"output_norm.weight", 1, {1000, 1}, FT_GGML_TYPE_F32},
    /* 2000 is NOT a multiple of 32. Without a partial tile somewhere in the fixture,
     * the rounding in the size arithmetic is never exercised and a planner that
     * truncated instead of rounding up would agree with the converter by accident. */
    {"output.weight", 2, {1024, 2000}, FT_GGML_TYPE_Q8_0}
};

#define FIXTURE_COUNT ((int)(sizeof(FIXTURE) / sizeof(FIXTURE[0])))

static void put_u32(FILE *f, uint32_t value)
{
    fwrite(&value, sizeof(value), 1, f);
}

static void put_u64(FILE *f, uint64_t value)
{
    fwrite(&value, sizeof(value), 1, f);
}

static void put_string(FILE *f, const char *text)
{
    uint64_t length = strlen(text);

    put_u64(f, length);
    fwrite(text, 1, (size_t)length, f);
}

static float fixture_value(uint32_t tensor, uint64_t index)
{
    uint32_t state = (tensor * 2654435761u) ^ (uint32_t)(index * 40503u);

    state = state * 1664525u + 1013904223u;
    return (float)((int32_t)(state >> 8) - 8388608) * (1.0f / 8388608.0f);
}

/**
 * fp32 -> fp16, for the Q8_0 block scale.
 *
 * Written here rather than borrowed: ft_float_convert.h exposes fp16 -> fp32 and both bf16
 * directions, but nothing the other way, and the fixture only needs normal magnitudes.
 */
static uint16_t fixture_fp16_from_float(float value)
{
    uint32_t bits;
    uint32_t exponent;

    memcpy(&bits, &value, sizeof(bits));
    exponent = (bits >> 23) & 0xffu;
    return (uint16_t)(((bits >> 16) & 0x8000u) |
                      ((exponent >= 113u && exponent <= 142u) ? ((exponent - 112u) << 10) : 0u) |
                      ((bits >> 13) & 0x3ffu));
}

/** One Q8_0 block row, written the way ggml lays it out: fp16 scale then 32 int8. */
static void write_q8_0_row(FILE *f, const float *row, uint64_t count)
{
    for (uint64_t base = 0; base < count; base += 32u) {
        float peak = 0.0f;
        float scale;
        uint16_t half;

        for (uint64_t i = 0; i < 32u && base + i < count; i++) {
            float magnitude = row[base + i] < 0.0f ? -row[base + i] : row[base + i];

            if (magnitude > peak)
                peak = magnitude;
        }
        scale = peak / 127.0f;
        half = fixture_fp16_from_float(scale);
        fwrite(&half, sizeof(half), 1, f);
        for (uint64_t i = 0; i < 32u; i++) {
            float value = (base + i < count) ? row[base + i] : 0.0f;
            int quantized = scale > 0.0f ? (int)(value / scale + (value < 0.0f ? -0.5f : 0.5f)) : 0;
            signed char byte;

            if (quantized > 127)
                quantized = 127;
            if (quantized < -127)
                quantized = -127;
            byte = (signed char)quantized;
            fwrite(&byte, 1, 1, f);
        }
    }
}

/**
 * Write the fixture as a GGUF.
 *
 * `mutate_index` >= 0 renames that tensor, which is how the digest is proven to depend on
 * the directory rather than on the file's length.
 */
static int write_fixture_gguf(const char *path, int mutate_index, const char *mutated_name,
                              uint64_t mutated_rows)
{
    FILE *f = fopen(path, "wb");
    uint64_t offset = 0;
    long header_end;
    long data_base;
    float *row;

    if (!f)
        return -1;

    put_u32(f, 0x46554747u); /* "GGUF" */
    put_u32(f, 3);
    put_u64(f, (uint64_t)FIXTURE_COUNT);
    put_u64(f, 5);

    put_string(f, "general.architecture");
    put_u32(f, 8); /* string */
    put_string(f, "llama");

    /* The four keys the split-half rotary permutation needs. Without them the converter
     * refuses split-half — which is a real product path (the cache falls back to source
     * pairing and says so), but it is not the path this fixture is for. 8 heads of 128
     * rows is the 1024-row attention shape above. */
    put_string(f, "llama.attention.head_count");
    put_u32(f, 4); /* uint32 */
    put_u32(f, 8);
    put_string(f, "llama.attention.head_count_kv");
    put_u32(f, 4);
    put_u32(f, 8);
    put_string(f, "llama.attention.key_length");
    put_u32(f, 4);
    put_u32(f, 128);
    put_string(f, "llama.rope.dimension_count");
    put_u32(f, 4);
    put_u32(f, 128);

    for (int i = 0; i < FIXTURE_COUNT; i++) {
        const fixture_tensor *t = &FIXTURE[i];
        uint64_t rows = t->n_dims > 1 ? t->dims[1] : 1;
        uint64_t elements;

        if (i == mutate_index && mutated_rows > 0u)
            rows = mutated_rows;
        elements = t->dims[0] * rows;

        put_string(f, (i == mutate_index && mutated_name) ? mutated_name : t->name);
        put_u32(f, t->n_dims);
        put_u64(f, t->dims[0]);
        if (t->n_dims > 1)
            put_u64(f, rows);
        put_u32(f, t->ggml_type);
        put_u64(f, offset);
        offset += ft_gguf_type_bytes(t->ggml_type, elements);
        offset = (offset + 31u) / 32u * 32u;
    }

    header_end = ftell(f);
    {
        long padded = (header_end + 31) / 32 * 32;

        for (long i = header_end; i < padded; i++)
            fputc(0, f);
    }

    row = malloc(4096 * sizeof(float));
    if (!row) {
        fclose(f);
        return -1;
    }
    data_base = ftell(f);
    offset = 0;
    for (int i = 0; i < FIXTURE_COUNT; i++) {
        const fixture_tensor *t = &FIXTURE[i];
        uint64_t rows = t->n_dims > 1 ? t->dims[1] : 1;
        uint64_t elements;

        if (i == mutate_index && mutated_rows > 0u)
            rows = mutated_rows;
        elements = t->dims[0] * rows;

        fseek(f, data_base + (long)offset, SEEK_SET);
        for (uint64_t r = 0; r < rows; r++) {
            for (uint64_t c = 0; c < t->dims[0]; c++)
                row[c] = fixture_value((uint32_t)i, r * t->dims[0] + c);
            if (t->ggml_type == FT_GGML_TYPE_Q8_0)
                write_q8_0_row(f, row, t->dims[0]);
            else
                fwrite(row, sizeof(float), (size_t)t->dims[0], f);
        }
        offset += ft_gguf_type_bytes(t->ggml_type, elements);
        offset = (offset + 31u) / 32u * 32u;
    }
    fseek(f, data_base + (long)offset, SEEK_SET);
    fputc(0, f);
    free(row);
    fclose(f);
    return 0;
}

/** A fixture with one 2-D tensor the role classifier does not recognise. */
static int write_unclassified_gguf(const char *path)
{
    FILE *f = fopen(path, "wb");
    float row[64];
    uint64_t offset = 0;
    long header_end;

    if (!f)
        return -1;

    put_u32(f, 0x46554747u);
    put_u32(f, 3);
    put_u64(f, 1);
    put_u64(f, 1);
    put_string(f, "general.architecture");
    put_u32(f, 8);
    put_string(f, "llama");

    put_string(f, "blk.0.mystery_projection.weight");
    put_u32(f, 2);
    put_u64(f, 64);
    put_u64(f, 64);
    put_u32(f, FT_GGML_TYPE_F32);
    put_u64(f, offset);

    header_end = ftell(f);
    {
        long padded = (header_end + 31) / 32 * 32;

        for (long i = header_end; i < padded; i++)
            fputc(0, f);
    }
    memset(row, 0, sizeof(row));
    for (int r = 0; r < 64; r++)
        fwrite(row, sizeof(float), 64, f);
    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* File helpers                                                               */
/* ------------------------------------------------------------------------- */

static long file_bytes(const char *path)
{
    struct stat info;

    if (stat(path, &info) != 0)
        return -1;
    return (long)info.st_size;
}

static int flip_byte(const char *path, long offset)
{
    FILE *f = fopen(path, "r+b");
    int value;

    if (!f)
        return -1;
    if (fseek(f, offset, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    value = fgetc(f);
    if (value == EOF) {
        fclose(f);
        return -1;
    }
    if (fseek(f, offset, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    fputc(value ^ 0x01, f);
    fclose(f);
    return 0;
}

static int truncate_file(const char *path, long keep)
{
    return truncate(path, (off_t)keep);
}

static int copy_file(const char *from, const char *to)
{
    FILE *in = fopen(from, "rb");
    FILE *out;
    char buffer[65536];
    size_t got;

    if (!in)
        return -1;
    out = fopen(to, "wb");
    if (!out) {
        fclose(in);
        return -1;
    }
    while ((got = fread(buffer, 1, sizeof(buffer), in)) > 0)
        fwrite(buffer, 1, got, out);
    fclose(in);
    fclose(out);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Budgets                                                                    */
/* ------------------------------------------------------------------------- */

static ft_tt_capacity_budget budget_of(uint64_t card_bytes, uint64_t kv_per_token,
                                       uint32_t context_tokens)
{
    ft_tt_capacity_budget budget;

    memset(&budget, 0, sizeof(budget));
    budget.card_bytes = card_bytes;
    budget.kv_bytes_per_token = kv_per_token;
    budget.context_tokens = context_tokens;
    return budget;
}

/** The artifact bytes one rung would cost, straight from the library. */
static uint64_t rung_bytes(const ft_gguf_map *map, ft_tt_ledger_mode mode)
{
    ft_tt_capacity_rung rung;

    if (ft_tt_capacity_rung_measure(map, FT_TT_FAMILY_LLAMA, mode, &rung) != FT_TT_CAPACITY_OK)
        return 0u;
    return rung.artifact_bytes;
}

/* ------------------------------------------------------------------------- */
/* 1. The path                                                                */
/* ------------------------------------------------------------------------- */

static void test_cache_path(void)
{
    char path[FT_TT_ARTIFACT_CACHE_PATH_BYTES];
    char narrow[16];

    section("cache path derivation");

    check(ft_tt_artifact_cache_path("/models/ornith-9b-Q8_0.gguf", NULL, FT_TT_LEDGER_BFP8, path,
                                    sizeof(path)) == FT_TT_ARTIFACT_CACHE_OK &&
              strcmp(path, "/models/ornith-9b-Q8_0.bfp8.fttw") == 0,
          "the artifact sits beside the checkpoint, with the rung in its name");

    check(ft_tt_artifact_cache_path("/models/ornith-9b-Q8_0.gguf", NULL, FT_TT_LEDGER_BFP4, path,
                                    sizeof(path)) == FT_TT_ARTIFACT_CACHE_OK &&
              strcmp(path, "/models/ornith-9b-Q8_0.bfp4.fttw") == 0,
          "two rungs of one checkpoint are two files, so neither can shadow the other");

    check(ft_tt_artifact_cache_path("/models/plain", "/var/cache", FT_TT_LEDGER_BFP8, path,
                                    sizeof(path)) == FT_TT_ARTIFACT_CACHE_OK &&
              strcmp(path, "/var/cache/plain.bfp8.fttw") == 0,
          "a cache directory takes the basename, and a name without .gguf keeps all of it");

    check(ft_tt_artifact_cache_path("/models/a-very-long-checkpoint-name.gguf", NULL,
                                    FT_TT_LEDGER_BFP8, narrow,
                                    sizeof(narrow)) == FT_TT_ARTIFACT_CACHE_EINVAL,
          "a path that does not fit is refused, not silently truncated");
}

/* ------------------------------------------------------------------------- */
/* 2. The source digest                                                       */
/* ------------------------------------------------------------------------- */

static void test_source_digest(const char *fixture_path)
{
    ft_gguf_map *map = NULL;
    ft_gguf_map *other = NULL;
    uint64_t base, again, renamed, reshaped;

    section("source tensor-directory digest");

    check(ft_gguf_map_open(fixture_path, &map) == FT_GGUF_OK, "opened the fixture checkpoint");
    base = ft_tt_artifact_source_digest(map);
    again = ft_tt_artifact_source_digest(map);
    ft_gguf_map_close(map);
    check_u64(again, base, "the digest of one checkpoint is stable across calls");

    check(write_fixture_gguf("build/cache_renamed.gguf", 2, "blk.0.attn_q_renamed.weight", 0) == 0,
          "wrote a checkpoint with one tensor renamed");
    check(ft_gguf_map_open("build/cache_renamed.gguf", &other) == FT_GGUF_OK, "opened it");
    renamed = ft_tt_artifact_source_digest(other);
    ft_gguf_map_close(other);
    check(renamed != base, "a renamed tensor changes the digest");

    check(write_fixture_gguf("build/cache_reshaped.gguf", 2, NULL, 512) == 0,
          "wrote a checkpoint with one tensor reshaped");
    check(ft_gguf_map_open("build/cache_reshaped.gguf", &other) == FT_GGUF_OK, "opened it");
    reshaped = ft_tt_artifact_source_digest(other);
    ft_gguf_map_close(other);
    check(reshaped != base, "a reshaped tensor changes the digest");
    check(reshaped != renamed, "and the two mutations do not collide with each other");
}

/* ------------------------------------------------------------------------- */
/* 3. The selector                                                            */
/* ------------------------------------------------------------------------- */

static void test_selector(const char *fixture_path)
{
    ft_tt_capacity_selection selection;
    ft_tt_capacity_budget budget;
    ft_gguf_map *map = NULL;
    uint64_t bfp8_bytes, bfp4_bytes;
    char line[FT_TT_CAPACITY_REASON_BYTES];

    section("automatic rung selection");

    check(ft_gguf_map_open(fixture_path, &map) == FT_GGUF_OK, "opened the fixture checkpoint");
    bfp8_bytes = rung_bytes(map, FT_TT_LEDGER_BFP8);
    bfp4_bytes = rung_bytes(map, FT_TT_LEDGER_BFP4);
    check(bfp8_bytes > bfp4_bytes, "bfp8 is the larger of the two rungs");
    check(bfp8_bytes - bfp4_bytes > 1024u * 1024u,
          "and by megabytes, so a budget between them is not a rounding accident");

    /* Room for everything: take the best rung there is. */
    budget = budget_of(bfp8_bytes * 4u, 0, 0);
    check(ft_tt_capacity_select(map, FT_TT_FAMILY_LLAMA, &budget, &selection) ==
              FT_TT_CAPACITY_OK &&
              selection.chosen && selection.mode == FT_TT_LEDGER_BFP8,
          "a card with room to spare gets bfp8 — the highest rung it can load");
    check(selection.chosen_index == 0u, "and it is the first rung on the ladder, not a search");

    /* Room for one rung only. The model has not changed; only the card has. */
    budget = budget_of(bfp8_bytes - 1u, 0, 0);
    check(ft_tt_capacity_select(map, FT_TT_FAMILY_LLAMA, &budget, &selection) ==
              FT_TT_CAPACITY_OK &&
              selection.chosen && selection.mode == FT_TT_LEDGER_BFP4,
          "a card one byte too small for bfp8 gets bfp4, with nobody asked");
    ft_tt_capacity_selection_line(&selection, &budget, line, sizeof(line));
    check(strstr(line, "bfp4") != NULL && strstr(line, "bfp8") != NULL,
          "and the line names the rung it took AND the one it could not");
    check(strstr(line, "does not fit") != NULL,
          "in the words of a measurement, not of a preference");

    /* The RESERVE has to move the decision, or it is decoration. */
    budget = budget_of(bfp8_bytes + 1024u * 1024u, 0, 0);
    check(ft_tt_capacity_select(map, FT_TT_FAMILY_LLAMA, &budget, &selection) ==
              FT_TT_CAPACITY_OK &&
              selection.mode == FT_TT_LEDGER_BFP8,
          "with a 1 MiB margin and no context reserved, bfp8 still fits");
    budget = budget_of(bfp8_bytes + 1024u * 1024u, 1024u, 4096u);
    check(ft_tt_capacity_select(map, FT_TT_FAMILY_LLAMA, &budget, &selection) ==
              FT_TT_CAPACITY_OK &&
              selection.mode == FT_TT_LEDGER_BFP4,
          "reserving 4 MiB of KV for 4096 tokens on that same card drops it to bfp4");

    /* Nothing fits. This must be a refusal with arithmetic, not a fallback. */
    budget = budget_of(bfp4_bytes / 2u, 128u, 1024u);
    check(ft_tt_capacity_select(map, FT_TT_FAMILY_LLAMA, &budget, &selection) ==
              FT_TT_CAPACITY_ENOFIT,
          "a card too small for either rung refuses");
    check(!selection.chosen, "and does not quietly choose one anyway");
    check(strstr(selection.reason, "bfp8") != NULL && strstr(selection.reason, "bfp4") != NULL,
          "the refusal names both rungs it measured");
    check(strstr(selection.reason, "GiB") != NULL,
          "and carries the arithmetic rather than the verdict alone");
    ft_tt_capacity_selection_line(&selection, &budget, line, sizeof(line));
    check(strcmp(line, selection.reason) == 0,
          "the one-line report of a refusal IS the refusal, not a summary of it");

    ft_gguf_map_close(map);
}

static void test_selector_refuses_an_unclassified_model(void)
{
    ft_tt_capacity_selection selection;
    ft_tt_capacity_budget budget = budget_of(64ull * 1024ull * 1024ull * 1024ull, 0, 0);
    ft_gguf_map *map = NULL;

    section("a checkpoint the ledger cannot classify");

    check(write_unclassified_gguf("build/cache_unclassified.gguf") == 0,
          "wrote a checkpoint with an unrecognised 2-D tensor");
    check(ft_gguf_map_open("build/cache_unclassified.gguf", &map) == FT_GGUF_OK, "opened it");
    check(ft_tt_capacity_select(map, FT_TT_FAMILY_LLAMA, &budget, &selection) ==
              FT_TT_CAPACITY_EUNCLASSIFIED,
          "an unclassified tensor is its own refusal, even on a card with 64 GiB free");
    check(strstr(selection.reason, "not a capacity one") != NULL,
          "and says so, because telling this person to buy a bigger card is the wrong fix");
    ft_gguf_map_close(map);
}

/* ------------------------------------------------------------------------- */
/* 4. The provenance trailer                                                  */
/* ------------------------------------------------------------------------- */

/** Every field of a record, so a comparison mutation can move exactly one. */
static void fill_reference_record(ft_tt_artifact_provenance *record)
{
    memset(record, 0, sizeof(*record));
    record->source_bytes = 123456789ull;
    record->source_digest = 0xfeedfacecafebeefull;
    record->payload_bytes = 4096ull;
    record->provenance_version = FT_TT_PROVENANCE_VERSION;
    record->artifact_version = FT_TT_ARTIFACT_VERSION_DENSE;
    record->ledger_mode = (uint32_t)FT_TT_LEDGER_BFP8;
    record->rope_pairing = (uint32_t)FT_TT_ROPE_PAIRING_SPLIT_HALF;
    record->artifact_flags = FT_TT_ARTIFACT_FLAG_ROPE_SPLIT_HALF;
    record->tensor_count = 12u;
    record->behaviour_stamp = ft_tt_artifact_behaviour_stamp();
    record->magic = FT_TT_PROVENANCE_MAGIC;
}

static void test_provenance_compare(void)
{
    ft_tt_artifact_provenance want, got;
    char why[256];

    section("provenance comparison names the field that disagrees");

    fill_reference_record(&want);
    got = want;
    check(ft_tt_artifact_provenance_compare(&want, &got, why, sizeof(why)) == FT_TT_PROVENANCE_OK,
          "two identical records agree");
    check(why[0] == '\0', "and say nothing");

    /* One field at a time. A comparison that only checked the digest would pass five of
     * these, and every one of them is a real way to serve the wrong artifact. */
    got = want;
    got.source_digest ^= 1ull;
    check(ft_tt_artifact_provenance_compare(&want, &got, why, sizeof(why)) ==
              FT_TT_PROVENANCE_ESTALE && strstr(why, "digest") != NULL,
          "a different source digest is caught and named");

    got = want;
    got.source_bytes += 1u;
    check(ft_tt_artifact_provenance_compare(&want, &got, why, sizeof(why)) ==
              FT_TT_PROVENANCE_ESTALE && strstr(why, "bytes") != NULL,
          "a different source length is caught and named");

    got = want;
    got.tensor_count += 1u;
    check(ft_tt_artifact_provenance_compare(&want, &got, why, sizeof(why)) ==
              FT_TT_PROVENANCE_ESTALE && strstr(why, "tensor count") != NULL,
          "a different tensor count is caught and named");

    got = want;
    got.ledger_mode = (uint32_t)FT_TT_LEDGER_BFP4;
    check(ft_tt_artifact_provenance_compare(&want, &got, why, sizeof(why)) ==
              FT_TT_PROVENANCE_ESTALE && strstr(why, "rung") != NULL,
          "an artifact converted at another rung is caught and named");

    got = want;
    got.rope_pairing = (uint32_t)FT_TT_ROPE_PAIRING_SOURCE;
    check(ft_tt_artifact_provenance_compare(&want, &got, why, sizeof(why)) ==
              FT_TT_PROVENANCE_ESTALE && strstr(why, "rotary") != NULL,
          "a different rotary pairing is caught — the two artifacts are the SAME SIZE");

    got = want;
    got.artifact_flags = 0u;
    check(ft_tt_artifact_provenance_compare(&want, &got, why, sizeof(why)) ==
              FT_TT_PROVENANCE_ESTALE && strstr(why, "flags") != NULL,
          "different header flags are caught and named");

    got = want;
    got.artifact_version = 1u;
    check(ft_tt_artifact_provenance_compare(&want, &got, why, sizeof(why)) ==
              FT_TT_PROVENANCE_ESTALE && strstr(why, "version") != NULL,
          "a stale v1 artifact is caught — and a stale v1 is byte-for-byte the same SIZE");

    got = want;
    got.behaviour_stamp ^= 1u;
    check(ft_tt_artifact_provenance_compare(&want, &got, why, sizeof(why)) ==
              FT_TT_PROVENANCE_ESTALE && strstr(why, "behaviour") != NULL,
          "an artifact from a converter with different behaviour is caught and named");
}

static void test_provenance_on_a_file(const char *artifact_path)
{
    ft_tt_artifact_provenance record;
    char scratch[] = "build/cache_trailer_scratch.fttw";
    char why[256];
    long size;

    section("the trailer, on a real artifact");

    check(copy_file(artifact_path, scratch) == 0, "copied an artifact to work on");
    check(ft_tt_artifact_provenance_read(scratch, &record) == FT_TT_PROVENANCE_OK,
          "the artifact carries a readable trailer");
    check(record.magic == FT_TT_PROVENANCE_MAGIC, "with the expected magic");
    check(ft_tt_artifact_provenance_verify_file(scratch, &record, why, sizeof(why)) ==
              FT_TT_PROVENANCE_OK,
          "and agrees with its own header and length");

    size = file_bytes(scratch);
    check(size > 64, "the artifact is longer than its own trailer");

    /* The last four bytes ARE the magic. Move one bit and the file stops claiming to be
     * cache-managed — which is a reconversion, not a silent serve. */
    check(flip_byte(scratch, size - 1) == 0, "flipped a bit in the trailer magic");
    check(ft_tt_artifact_provenance_read(scratch, &record) == FT_TT_PROVENANCE_EABSENT,
          "a damaged magic reads as 'no trailer', never as a valid one");
    check(copy_file(artifact_path, scratch) == 0, "restored the artifact");

    /* A trailer that survives but describes a different length. */
    check(flip_byte(scratch, size - 64 + 16) == 0, "flipped a bit in the trailer's payload length");
    check(ft_tt_artifact_provenance_read(scratch, &record) == FT_TT_PROVENANCE_OK,
          "the trailer still parses");
    check(ft_tt_artifact_provenance_verify_file(scratch, &record, why, sizeof(why)) ==
              FT_TT_PROVENANCE_ESTALE,
          "but the file's own length contradicts it, and that is caught");
    check(copy_file(artifact_path, scratch) == 0, "restored the artifact");

    /* Truncation. A short artifact whose directory points past its own end. */
    check(truncate_file(scratch, size - 32) == 0, "truncated the artifact by 32 bytes");
    check(ft_tt_artifact_provenance_read(scratch, &record) == FT_TT_PROVENANCE_EABSENT,
          "a truncated artifact has no trailer to find");
    check(copy_file(artifact_path, scratch) == 0, "restored the artifact");

    /* Something appended. The FTTW part is intact and the file is longer than it says. */
    {
        FILE *f = fopen(scratch, "ab");

        if (f) {
            fputc(0, f);
            fclose(f);
        }
    }
    check(ft_tt_artifact_provenance_read(scratch, &record) == FT_TT_PROVENANCE_EABSENT,
          "an artifact something was appended to no longer ends in the magic");

    /* THE ISOLATING ARM for the file-length check. A trailer appended TWICE leaves the
     * last 64 bytes a perfectly valid trailer whose every field agrees with the artifact's
     * header — the only thing wrong is that the file is 64 bytes longer than the trailer
     * says it is, and nothing but the length comparison can see that. */
    check(copy_file(artifact_path, scratch) == 0, "restored the artifact");
    {
        unsigned char tail[64];
        FILE *f = fopen(scratch, "rb");
        FILE *out;

        if (f) {
            fseek(f, -64, SEEK_END);
            if (fread(tail, 1, sizeof(tail), f) != sizeof(tail))
                memset(tail, 0, sizeof(tail));
            fclose(f);
            out = fopen(scratch, "ab");
            if (out) {
                fwrite(tail, 1, sizeof(tail), out);
                fclose(out);
            }
        }
    }
    check(ft_tt_artifact_provenance_read(scratch, &record) == FT_TT_PROVENANCE_OK,
          "an artifact with its trailer appended twice still ends in a valid trailer");
    check(ft_tt_artifact_provenance_verify_file(scratch, &record, why, sizeof(why)) ==
              FT_TT_PROVENANCE_ESTALE,
          "and the file's own length is the only thing that catches it");

    /* A trailer whose self-description is wrong: same magic, impossible size. */
    check(copy_file(artifact_path, scratch) == 0, "restored the artifact");
    check(flip_byte(scratch, size - 64 + 52) == 0, "corrupted the trailer's own length field");
    check(ft_tt_artifact_provenance_read(scratch, &record) == FT_TT_PROVENANCE_ESTALE,
          "a trailer that disagrees with itself is STALE, not absent — so it is reported, "
          "not silently reconverted");
    remove(scratch);
}

/* ------------------------------------------------------------------------- */
/* 5. The planner and the converter must agree                                */
/* ------------------------------------------------------------------------- */

/**
 * The planner's arithmetic against the artifact the converter actually wrote.
 *
 * This is the gate that keeps the two implementations honest. The planner decides whether
 * a model fits without converting it; if it were wrong, the first sign would be a serve
 * that OOMs the card after a twenty-minute conversion.
 */
static void test_planner_matches_the_converter(const char *fixture_path,
                                               const char *artifact_path)
{
    ft_gguf_map *map = NULL;
    uint64_t predicted;
    uint64_t actual = 0;
    unsigned char header[FT_TT_ARTIFACT_HEADER_BYTES];
    FILE *f;

    section("the planner's arithmetic equals the artifact the converter writes");

    check(ft_gguf_map_open(fixture_path, &map) == FT_GGUF_OK, "opened the fixture checkpoint");
    predicted = rung_bytes(map, FT_TT_LEDGER_BFP8);
    predicted += FT_TT_ARTIFACT_ALIGNMENT; /* the header, padded to the payload alignment */
    predicted += (uint64_t)ft_gguf_map_tensor_count(map) * FT_TT_ARTIFACT_ENTRY_BYTES;
    ft_gguf_map_close(map);

    f = fopen(artifact_path, "rb");
    check(f != NULL, "opened the artifact the cache produced");
    if (f) {
        if (fread(header, 1, sizeof(header), f) == sizeof(header))
            actual = (uint64_t)header[32] | ((uint64_t)header[33] << 8) |
                     ((uint64_t)header[34] << 16) | ((uint64_t)header[35] << 24) |
                     ((uint64_t)header[36] << 32) | ((uint64_t)header[37] << 40) |
                     ((uint64_t)header[38] << 48) | ((uint64_t)header[39] << 56);
        fclose(f);
    }
    check_u64(actual, predicted,
              "the artifact's own total_bytes is what the planner predicted, to the byte");
}

/* ------------------------------------------------------------------------- */
/* 6. The cache                                                               */
/* ------------------------------------------------------------------------- */

static ft_tt_artifact_cache_request request_for(const char *gguf_path, uint64_t card_bytes,
                                                int forced_mode)
{
    ft_tt_artifact_cache_request request;

    memset(&request, 0, sizeof(request));
    request.gguf_path = gguf_path;
    request.budget = budget_of(card_bytes, 0, 0);
    request.rope_pairing = -1; /* resolved from the checkpoint, as the product does */
    request.forced_mode = forced_mode;
    request.progress_style = FT_TT_CACHE_PROGRESS_SILENT;
    return request;
}

static void test_cache_produces_then_hits(const char *fixture_path, char *artifact_path_out,
                                          size_t artifact_path_bytes)
{
    ft_tt_artifact_cache_request request = request_for(fixture_path, 64ull * 1024ull * 1024ull *
                                                                        1024ull, -1);
    ft_tt_artifact_cache_result result;
    long first_size;

    section("convert on load, then never again");

    {
        int rc = ft_tt_artifact_cache_ensure(&request, &result);

        check(rc == FT_TT_ARTIFACT_CACHE_OK,
              "a checkpoint with no artifact beside it produces one");
        if (rc != FT_TT_ARTIFACT_CACHE_OK)
            printf("        rc %d: %s\n", rc, result.reason);
    }
    check(result.produced == 1, "and says it converted");
    check(result.selection.mode == FT_TT_LEDGER_BFP8, "at the rung the selector chose");
    check(file_bytes(result.artifact_path) > 0, "the artifact exists");
    first_size = file_bytes(result.artifact_path);
    snprintf(artifact_path_out, artifact_path_bytes, "%s", result.artifact_path);

    check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_OK,
          "the second call succeeds");
    check(result.produced == 0, "WITHOUT converting — this is the whole point of the cache");
    check(result.note[0] == '\0', "and with nothing to explain");
    check_u64((uint64_t)file_bytes(result.artifact_path), (uint64_t)first_size,
              "the artifact is the same file, byte count included");
}

static void test_cache_detects_a_stale_artifact(const char *fixture_path,
                                                const char *artifact_path)
{
    ft_tt_artifact_cache_request request = request_for(fixture_path, 64ull * 1024ull * 1024ull *
                                                                        1024ull, -1);
    ft_tt_artifact_cache_result result;
    char keep[] = "build/cache_keep.fttw";
    long size = file_bytes(artifact_path);

    section("a stale or damaged cached artifact is detected, not served");

    check(copy_file(artifact_path, keep) == 0, "kept a good copy of the artifact");

    /* The exact failure this project has shipped twice: an artifact of the RIGHT SIZE and
     * the right shape that is not the right artifact. */
    check(flip_byte(artifact_path, size - 64 + 8) == 0,
          "changed the source digest the cached artifact claims");
    check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_OK,
          "the cache still answers");
    check(result.produced == 1, "by RECONVERTING rather than serving what it found");
    check(strstr(result.note, "digest") != NULL, "and it names the digest as the reason");

    check(truncate_file(artifact_path, size / 2) == 0, "truncated the cached artifact in half");
    check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_OK &&
              result.produced == 1,
          "a truncated artifact is reconverted");

    /* An artifact from another rung, moved onto this rung's path. Same model, same
     * converter, same everything except the one thing that matters. */
    {
        ft_tt_artifact_cache_request forced = request_for(fixture_path,
                                                          64ull * 1024ull * 1024ull * 1024ull,
                                                          (int)FT_TT_LEDGER_BFP4);
        ft_tt_artifact_cache_result bfp4;

        check(ft_tt_artifact_cache_ensure(&forced, &bfp4) == FT_TT_ARTIFACT_CACHE_OK,
              "produced a bfp4 artifact of the same checkpoint");
        check(strcmp(bfp4.artifact_path, artifact_path) != 0,
              "which lives at its own path, so it could not have overwritten the bfp8 one");
        check(copy_file(bfp4.artifact_path, artifact_path) == 0,
              "copied the bfp4 artifact over the bfp8 one");
        check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_OK &&
                  result.produced == 1 && strstr(result.note, "rung") != NULL,
              "the cache reads the rung out of the file and reconverts");
        remove(bfp4.artifact_path);
    }

    /* An artifact with no trailer at all: what a hand-run ft_tt_convert_gguf produces. */
    {
        FILE *f = fopen(artifact_path, "rb");
        FILE *out = fopen("build/cache_notrailer.fttw", "wb");
        char buffer[65536];
        size_t got;
        long remaining = file_bytes(artifact_path) - 64;

        if (f && out) {
            while (remaining > 0 &&
                   (got = fread(buffer, 1,
                                remaining < (long)sizeof(buffer) ? (size_t)remaining
                                                                 : sizeof(buffer),
                                f)) > 0) {
                fwrite(buffer, 1, got, out);
                remaining -= (long)got;
            }
        }
        if (f)
            fclose(f);
        if (out)
            fclose(out);
        check(copy_file("build/cache_notrailer.fttw", artifact_path) == 0,
              "put a trailer-less artifact at the cache path");
        check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_OK &&
                  result.produced == 1 && strstr(result.note, "provenance") != NULL,
              "an artifact with no provenance is reconverted and says why");
        remove("build/cache_notrailer.fttw");
    }

    /* The source changed under a good artifact. token_embd is the tensor to resize: the
     * attention tensors carry the rope geometry the metadata declares, and shrinking one of
     * those would make the CONVERTER refuse — which would test the refusal path, not the
     * invalidation one. */
    check(write_fixture_gguf(fixture_path, 0, NULL, 1024) == 0,
          "rewrote the checkpoint with one tensor reshaped");
    check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_OK &&
              result.produced == 1,
          "a changed checkpoint invalidates its artifact");
    check(write_fixture_gguf(fixture_path, -1, NULL, 0) == 0, "restored the checkpoint");
    check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_OK &&
              result.produced == 1,
          "and restoring it invalidates the artifact of the changed one");
    check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_OK &&
              result.produced == 0,
          "after which the cache settles and hits again");
    remove(keep);
}

static void test_cache_refuses_honestly(const char *fixture_path)
{
    ft_tt_artifact_cache_result result;
    ft_gguf_map *map = NULL;
    uint64_t bfp4_bytes;

    section("the cache refuses a model that cannot fit, with the arithmetic");

    check(ft_gguf_map_open(fixture_path, &map) == FT_GGUF_OK, "opened the fixture checkpoint");
    bfp4_bytes = rung_bytes(map, FT_TT_LEDGER_BFP4);
    ft_gguf_map_close(map);

    {
        ft_tt_artifact_cache_request request = request_for(fixture_path, bfp4_bytes / 2u, -1);

        check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_ENOFIT,
              "a card too small for the smallest rung is refused");
        check(result.produced == 0, "and nothing is converted");
        check(strstr(result.reason, "GiB") != NULL && strstr(result.reason, "bfp4") != NULL,
              "with the arithmetic and both rungs in the message");
    }

    {
        /* Forcing a rung may pick one; it may not overrule the card. */
        ft_tt_artifact_cache_request request =
            request_for(fixture_path, bfp4_bytes / 2u, (int)FT_TT_LEDGER_BFP8);

        check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_ENOFIT,
              "and forcing a rung does not get around the fit check");
    }

    {
        ft_tt_artifact_cache_request request =
            request_for("build/cache_does_not_exist.gguf", 64ull * 1024ull * 1024ull * 1024ull, -1);

        check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_ESOURCE,
              "a checkpoint that is not there is its own error, not a fit failure");
    }
}

static void test_cache_rope_pairing_is_part_of_the_key(const char *fixture_path,
                                                       const char *artifact_path)
{
    ft_tt_artifact_cache_request request = request_for(fixture_path, 64ull * 1024ull * 1024ull *
                                                                        1024ull, -1);
    ft_tt_artifact_cache_result result;
    long split_half_size;

    section("rotary pairing is part of the cache key, and it has to be");

    request.rope_pairing = (int)FT_TT_ROPE_PAIRING_SPLIT_HALF;
    check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_OK,
          "an artifact exists for split-half");
    split_half_size = file_bytes(result.artifact_path);

    request.rope_pairing = (int)FT_TT_ROPE_PAIRING_SOURCE;
    check(ft_tt_artifact_cache_ensure(&request, &result) == FT_TT_ARTIFACT_CACHE_OK &&
              result.produced == 1,
          "asking for source pairing reconverts, because the bytes MEAN something different");
    check(strstr(result.note, "rotary") != NULL || strstr(result.note, "flags") != NULL,
          "and names the pairing, not a size");
    check_u64((uint64_t)file_bytes(result.artifact_path), (uint64_t)split_half_size,
              "the two artifacts are exactly the same size — which is why a size check "
              "would have served the wrong one");
    (void)artifact_path;
}

/**
 * Delete everything a previous run of this suite left behind.
 *
 * THIS IS NOT HOUSEKEEPING, IT IS THE GATE. The first assertion in the suite is "a
 * checkpoint with no artifact beside it produces one", and an artifact left by the last
 * run makes it a cache HIT — so the suite passed once and then reported a failure on the
 * second run, from state on disk rather than from the code. Same shape as the L1 test
 * pattern that read the previous process's bytes and passed.
 */
static void remove_stale_fixtures(void)
{
    static const char *const LEFTOVERS[] = {
        "build/cache_fixture.bfp8.fttw",     "build/cache_fixture.bfp4.fttw",
        "build/cache_renamed.gguf",          "build/cache_reshaped.gguf",
        "build/cache_unclassified.gguf",     "build/cache_keep.fttw",
        "build/cache_notrailer.fttw",        "build/cache_trailer_scratch.fttw"
    };
    size_t i;

    for (i = 0; i < sizeof(LEFTOVERS) / sizeof(LEFTOVERS[0]); i++)
        remove(LEFTOVERS[i]);
}

int main(void)
{
    char fixture_path[] = "build/cache_fixture.gguf";
    char artifact_path[FT_TT_ARTIFACT_CACHE_PATH_BYTES];

    printf("ft_tt artifact cache, provenance and rung selection\n");
    remove_stale_fixtures();

    if (write_fixture_gguf(fixture_path, -1, NULL, 0) != 0) {
        printf("cannot write the fixture checkpoint; is build/ there?\n");
        return 1;
    }

    test_cache_path();
    test_source_digest(fixture_path);
    test_selector(fixture_path);
    test_selector_refuses_an_unclassified_model();
    test_provenance_compare();

    test_cache_produces_then_hits(fixture_path, artifact_path, sizeof(artifact_path));
    test_planner_matches_the_converter(fixture_path, artifact_path);
    test_provenance_on_a_file(artifact_path);
    test_cache_rope_pairing_is_part_of_the_key(fixture_path, artifact_path);
    test_cache_detects_a_stale_artifact(fixture_path, artifact_path);
    test_cache_refuses_honestly(fixture_path);

    printf("\n%d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}
