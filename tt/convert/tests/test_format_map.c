/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_format_map.c
 * @brief Host tests for the converter's per-tensor format map (`--format-map`): the map
 *   parser, which tensors a map may name, and conversion with a map on a synthetic
 *   Qwen3.8-shaped Q8_0 checkpoint.
 *
 * The conversion gate: mapped tensors carry their map format and exactly the bytes the
 * uniform rung of that format writes for them; every other entry is byte-identical to the
 * `bfp8` rung; an empty map changes nothing. The reds: a typo'd name and every unmappable
 * role are refused before any output exists.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_format_map.h"
#include "ft_tt_precision_ledger.h"
#include "ft_tt_weight_convert.h"
#include "synthetic_hybrid_checkpoint.h"

#define FIXTURE "build/fmap_hybrid.gguf"

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (!condition)
        checks_failed++;
    printf("  %s  %s\n", condition ? "ok  " : "FAIL", what);
}

static void check_u64(uint64_t got, uint64_t want, const char *what)
{
    checks_run++;
    if (got != want)
        checks_failed++;
    printf("  %s  %s = %" PRIu64 "%s\n", got == want ? "ok  " : "FAIL", what, got, got == want ? "" : " (wrong)");
    if (got != want)
        printf("        expected %" PRIu64 "\n", want);
}

static void section(const char *name)
{
    printf("\n== %s ==\n", name);
}

/* ------------------------------------------------------------------------- */
/* Artifact reading                                                          */
/* ------------------------------------------------------------------------- */

typedef struct {
    ft_tt_artifact_header header;
    ft_tt_artifact_entry *entries;
    unsigned char *bytes;
    long size;
} artifact_file;

static int artifact_read(const char *path, artifact_file *a)
{
    FILE *f = fopen(path, "rb");

    memset(a, 0, sizeof(*a));
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    a->size = ftell(f);
    fseek(f, 0, SEEK_SET);
    a->bytes = malloc((size_t)a->size);
    if (!a->bytes || fread(a->bytes, 1, (size_t)a->size, f) != (size_t)a->size) {
        fclose(f);
        return -1;
    }
    fclose(f);
    memcpy(&a->header, a->bytes, sizeof(a->header));
    a->entries = (ft_tt_artifact_entry *)(a->bytes + a->header.directory_offset);
    return 0;
}

static void artifact_free(artifact_file *a)
{
    free(a->bytes);
    memset(a, 0, sizeof(*a));
}

static const ft_tt_artifact_entry *artifact_find(const artifact_file *a, const char *name)
{
    for (uint32_t i = 0; i < a->header.tensor_count; i++)
        if (strcmp(a->entries[i].name, name) == 0)
            return &a->entries[i];
    return NULL;
}

static int same_payload(const artifact_file *a, const ft_tt_artifact_entry *x, const artifact_file *b,
                        const ft_tt_artifact_entry *y)
{
    return x->byte_count == y->byte_count &&
           memcmp(a->bytes + x->data_offset, b->bytes + y->data_offset, (size_t)x->byte_count) == 0;
}

/** Everything but format, offset and size: the geometry a map must never touch. */
static int same_geometry(const ft_tt_artifact_entry *x, const ft_tt_artifact_entry *y)
{
    return strcmp(x->name, y->name) == 0 && x->role == y->role && x->n_dims == y->n_dims &&
           x->tile_rows == y->tile_rows && x->tile_cols == y->tile_cols && x->layout == y->layout &&
           memcmp(x->dims, y->dims, sizeof(x->dims)) == 0;
}

static int files_identical(const char *x, const char *y)
{
    artifact_file a, b;
    int same;

    if (artifact_read(x, &a) != 0 || artifact_read(y, &b) != 0)
        return 0;
    same = a.size == b.size && memcmp(a.bytes, b.bytes, (size_t)a.size) == 0;
    artifact_free(&a);
    artifact_free(&b);
    return same;
}

static int file_exists(const char *path)
{
    FILE *f = fopen(path, "rb");

    if (f)
        fclose(f);
    return f != NULL;
}

static int convert(const char *out, ft_tt_ledger_mode mode, ft_tt_convert_scope scope, const ft_tt_format_map *map)
{
    ft_tt_convert_options options;

    memset(&options, 0, sizeof(options));
    options.mode = mode;
    options.scope = scope;
    options.format_map = map;
    return ft_tt_convert_gguf(FIXTURE, out, &options);
}

static int parse_text(const char *text, ft_tt_format_map *map, char *error)
{
    return ft_tt_format_map_parse(text, strlen(text), "t.map", map, error, FT_TT_FORMAT_MAP_ERROR_BYTES);
}

/* ------------------------------------------------------------------------- */
/* Parser                                                                    */
/* ------------------------------------------------------------------------- */

static void test_parse_accepts(void)
{
    static const char text[] = "# budget -4 GiB\n"
                               "\n"
                               "blk.0.ffn_up.weight bfp4_b\n"
                               "  blk.1.attn_q.weight\tbfp8_b   # trailing comment\r\n"
                               "output.weight bfp4_b";
    ft_tt_format_map map;
    char error[FT_TT_FORMAT_MAP_ERROR_BYTES] = "";
    const ft_tt_format_map_entry *q;

    section("parser accepts the format");
    memset(&map, 0, sizeof(map));
    check(parse_text(text, &map, error) == FT_TT_FORMAT_MAP_OK, "comments, blank line, tabs, CRLF, no final newline");
    check_u64(map.count, 3, "entries");
    q = ft_tt_format_map_find(&map, "blk.1.attn_q.weight");
    check(q && q->format == FT_TT_FORMAT_BFP8_B && q->line == 4, "attn_q is bfp8_b, from line 4");
    check(ft_tt_format_map_find(&map, "output.weight") && ft_tt_format_map_find(&map, "output.weight")->line == 5,
          "the last line counts without a newline");
    check(ft_tt_format_map_find(&map, "blk.0.ffn_up") == NULL, "lookup is exact, not a substring");
    check_u64(ft_tt_format_map_count(&map, FT_TT_FORMAT_BFP4_B), 2, "bfp4_b entries");
    check(parse_text("# only a comment\n\n   \n", &map, error) == FT_TT_FORMAT_MAP_OK && map.count == 0,
          "a map of comments is an empty map");
    ft_tt_format_map_free(&map);
}

static void test_parse_refuses(void)
{
    static const struct {
        const char *text;
        const char *message_has;
        const char *what;
    } cases[] = {
        {"blk.0.ffn_up.weight\n", "t.map:1:", "one field"},
        {"\nblk.0.ffn_up.weight bfp4_b extra\n", "t.map:2:", "three fields"},
        {"blk.0.ffn_up.weight bfp4\n", "'bfp4'", "a rung name is not a format"},
        {"blk.0.ffn_up.weight bf16\n", "'bf16'", "bf16 is not a map format"},
        {"blk.0.ffn_up.weight bfp6_b\n", "'bfp6_b'", "bfp6_b is not a map format"},
        {"a.weight bfp4_b\na.weight bfp8_b\n", "line 1", "a name listed twice"},
    };
    char long_line[FT_TT_FORMAT_MAP_NAME_BYTES + 16];
    ft_tt_format_map map;
    char error[FT_TT_FORMAT_MAP_ERROR_BYTES];

    section("parser refuses everything else, naming the line");
    memset(&map, 0, sizeof(map));
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        error[0] = '\0';
        check(parse_text(cases[i].text, &map, error) == FT_TT_FORMAT_MAP_EPARSE && map.count == 0 &&
                  strstr(error, cases[i].message_has) != NULL,
              cases[i].what);
        printf("        %s\n", error);
    }
    memset(long_line, 'x', FT_TT_FORMAT_MAP_NAME_BYTES);
    snprintf(long_line + FT_TT_FORMAT_MAP_NAME_BYTES, 16, " bfp4_b\n");
    check(parse_text(long_line, &map, error) == FT_TT_FORMAT_MAP_EPARSE, "a 128-byte name");
    check(ft_tt_format_map_load("build/no_such.map", &map, error, sizeof(error)) == FT_TT_FORMAT_MAP_EIO,
          "a missing file is EIO");
    ft_tt_format_map_free(&map);
}

/* ------------------------------------------------------------------------- */
/* Which tensors a map may name                                              */
/* ------------------------------------------------------------------------- */

static void test_map_refusal_rule(void)
{
    static const struct {
        const char *name;
        uint32_t n_dims;
        int allowed;
    } cases[] = {
        {"blk.0.ffn_gate.weight", 2, 1},   {"blk.0.ffn_up.weight", 2, 1},       {"blk.0.ffn_down.weight", 2, 1},
        {"blk.0.attn_qkv.weight", 2, 1},   {"blk.0.attn_gate.weight", 2, 1},    {"blk.0.ssm_out.weight", 2, 1},
        {"blk.1.attn_q.weight", 2, 1},     {"blk.1.attn_k.weight", 2, 1},       {"blk.1.attn_v.weight", 2, 1},
        {"blk.1.attn_output.weight", 2, 1}, {"output.weight", 2, 1},            {"blk.64.nextn.eh_proj.weight", 2, 1},
        {"token_embd.weight", 2, 0},       {"blk.0.attn_norm.weight", 1, 0},    {"blk.0.ssm_a", 1, 0},
        {"blk.0.ssm_conv1d.weight", 2, 0}, {"blk.0.ssm_dt.bias", 1, 0},         {"blk.0.ssm_alpha.weight", 2, 0},
        {"blk.0.ssm_beta.weight", 2, 0},   {"blk.0.ffn_gate_inp.weight", 2, 0}, {"blk.0.ffn_up_exps.weight", 3, 0},
        {"blk.0.ffn_up.weight", 3, 0},     {"blk.0.mystery.weight", 2, 0},
    };
    uint32_t right = 0;

    section("ft_tt_ledger_map_refusal: dense 2-D matmul weights only");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const char *refusal = ft_tt_ledger_map_refusal(ft_tt_ledger_role(cases[i].name), cases[i].n_dims);

        if ((refusal == NULL) == cases[i].allowed)
            right++;
        else
            printf("        wrong: %s (%u-D) %s\n", cases[i].name, cases[i].n_dims,
                   refusal ? refusal : "accepted");
    }
    check_u64(right, sizeof(cases) / sizeof(cases[0]), "roles judged as the spec says");
}

/* ------------------------------------------------------------------------- */
/* Conversion with a map                                                     */
/* ------------------------------------------------------------------------- */

/**
 * Compare a mapped artifact with the bfp8-rung artifact and the bfp4-rung artifact: each
 * entry the map moves to Bfp4 must equal the bfp4 rung's entry, every other must equal the
 * bfp8 rung's. Returns the count of entries that matched their reference.
 */
static uint32_t entries_matching_references(const artifact_file *mixed, const artifact_file *bfp8,
                                            const artifact_file *bfp4, const char *const *bfp4_entries, uint32_t bfp4_entry_count,
                                            uint32_t *moved_out)
{
    uint32_t matched = 0;

    *moved_out = 0;
    for (uint32_t i = 0; i < mixed->header.tensor_count; i++) {
        const ft_tt_artifact_entry *got = &mixed->entries[i];
        const ft_tt_artifact_entry *base = artifact_find(bfp8, got->name);
        int moved = 0;

        for (uint32_t m = 0; m < bfp4_entry_count; m++)
            moved |= strcmp(got->name, bfp4_entries[m]) == 0;
        if (!base || !same_geometry(got, base))
            continue;
        if (moved) {
            const ft_tt_artifact_entry *want = artifact_find(bfp4, got->name);

            *moved_out += 1;
            if (want && got->format == FT_TT_FORMAT_BFP4_B && same_payload(mixed, got, bfp4, want))
                matched++;
        } else if (got->format == base->format && same_payload(mixed, got, bfp8, base)) {
            matched++;
        }
    }
    return matched;
}

static void test_trunk_with_map(const ft_tt_format_map *map)
{
    static const char *const moved[] = {"blk.0.ffn_up.weight", "blk.0.attn_qkv.weight",
                                        "blk.1.attn_output.weight", "output.weight"};
    artifact_file mixed, bfp8, bfp4;
    uint32_t moved_count = 0, matched;

    section("trunk: listed tensors at bfp4_b, every other entry byte-identical to bfp8");
    check(convert("build/fmap_bfp8.fttw", FT_TT_LEDGER_BFP8, FT_TT_CONVERT_SCOPE_TRUNK, NULL) == FT_TT_CONVERT_OK &&
              convert("build/fmap_bfp4.fttw", FT_TT_LEDGER_BFP4, FT_TT_CONVERT_SCOPE_TRUNK, NULL) ==
                  FT_TT_CONVERT_OK &&
              convert("build/fmap_mixed.fttw", FT_TT_LEDGER_BFP8, FT_TT_CONVERT_SCOPE_TRUNK, map) ==
                  FT_TT_CONVERT_OK,
          "converted the bfp8 rung, the bfp4 rung and the bfp8 rung + map");
    if (artifact_read("build/fmap_mixed.fttw", &mixed) != 0 || artifact_read("build/fmap_bfp8.fttw", &bfp8) != 0 ||
        artifact_read("build/fmap_bfp4.fttw", &bfp4) != 0) {
        check(0, "read the three artifacts back");
        return;
    }
    check_u64(mixed.header.tensor_count, bfp8.header.tensor_count, "tensor count unchanged");
    check_u64(mixed.header.version, bfp8.header.version, "artifact version unchanged");
    matched = entries_matching_references(&mixed, &bfp8, &bfp4, moved, 4u, &moved_count);
    check_u64(moved_count, 4, "entries the map moved (the eh_proj entry is outside the trunk)");
    check_u64(matched, mixed.header.tensor_count, "entries equal to their reference rung, byte for byte");
    check(artifact_find(&mixed, "blk.0.ffn_up.weight") &&
              artifact_find(&mixed, "blk.0.ffn_up.weight")->byte_count <
                  artifact_find(&bfp8, "blk.0.ffn_up.weight")->byte_count,
          "a moved Q8_0 tensor is smaller than at bfp8");
    artifact_free(&mixed);
    artifact_free(&bfp8);
    artifact_free(&bfp4);
}

static void test_mtp_with_map(const ft_tt_format_map *map)
{
    static const char *const moved[] = {"mtp.eh_proj_e.weight", "mtp.eh_proj_h.weight"};
    artifact_file mixed, bfp8, bfp4;
    uint32_t moved_count = 0, matched;

    section("MTP scope: both eh_proj halves follow the GGUF name");
    check(convert("build/fmap_bfp8.mtp.fttw", FT_TT_LEDGER_BFP8, FT_TT_CONVERT_SCOPE_MTP, NULL) ==
                  FT_TT_CONVERT_OK &&
              convert("build/fmap_bfp4.mtp.fttw", FT_TT_LEDGER_BFP4, FT_TT_CONVERT_SCOPE_MTP, NULL) ==
                  FT_TT_CONVERT_OK &&
              convert("build/fmap_mixed.mtp.fttw", FT_TT_LEDGER_BFP8, FT_TT_CONVERT_SCOPE_MTP, map) ==
                  FT_TT_CONVERT_OK,
          "converted the MTP block three ways");
    if (artifact_read("build/fmap_mixed.mtp.fttw", &mixed) != 0 ||
        artifact_read("build/fmap_bfp8.mtp.fttw", &bfp8) != 0 ||
        artifact_read("build/fmap_bfp4.mtp.fttw", &bfp4) != 0) {
        check(0, "read the three MTP artifacts back");
        return;
    }
    matched = entries_matching_references(&mixed, &bfp8, &bfp4, moved, 2u, &moved_count);
    check_u64(moved_count, 2, "eh_proj halves moved");
    check_u64(matched, mixed.header.tensor_count, "MTP entries equal to their reference rung");
    artifact_free(&mixed);
    artifact_free(&bfp8);
    artifact_free(&bfp4);
}

static void test_empty_map_changes_nothing(void)
{
    ft_tt_format_map empty;
    char error[FT_TT_FORMAT_MAP_ERROR_BYTES];

    section("an empty map is no map");
    memset(&empty, 0, sizeof(empty));
    check(parse_text("# nothing\n", &empty, error) == FT_TT_FORMAT_MAP_OK &&
              convert("build/fmap_empty.fttw", FT_TT_LEDGER_BFP8, FT_TT_CONVERT_SCOPE_TRUNK, &empty) ==
                  FT_TT_CONVERT_OK,
          "converted with an empty map");
    check(files_identical("build/fmap_empty.fttw", "build/fmap_bfp8.fttw"), "byte-identical to no map");
    ft_tt_format_map_free(&empty);
}

static void test_refusals(const ft_tt_format_map *good)
{
    static const char *const refused[] = {
        "blk.0.ffn_upp.weight bfp4_b\n",    /* the typo */
        "token_embd.weight bfp4_b\n",       "blk.0.attn_norm.weight bfp4_b\n", "blk.0.ssm_a bfp4_b\n",
        "blk.0.ssm_conv1d.weight bfp4_b\n", "blk.0.ssm_dt.bias bfp4_b\n",      "blk.0.ssm_alpha.weight bfp4_b\n",
    };
    ft_tt_format_map map;
    ft_tt_convert_options options;
    char error[FT_TT_FORMAT_MAP_ERROR_BYTES];
    uint32_t refused_cleanly = 0;

    section("red: unknown and unmappable names are refused, and no file is written");
    memset(&map, 0, sizeof(map));
    for (size_t i = 0; i < sizeof(refused) / sizeof(refused[0]); i++) {
        remove("build/fmap_refused.fttw");
        if (parse_text(refused[i], &map, error) == FT_TT_FORMAT_MAP_OK &&
            convert("build/fmap_refused.fttw", FT_TT_LEDGER_BFP8, FT_TT_CONVERT_SCOPE_TRUNK, &map) ==
                FT_TT_CONVERT_EFORMATMAP &&
            !file_exists("build/fmap_refused.fttw"))
            refused_cleanly++;
    }
    check_u64(refused_cleanly, sizeof(refused) / sizeof(refused[0]), "maps refused with EFORMATMAP and no output");
    ft_tt_format_map_free(&map);

    memset(&options, 0, sizeof(options));
    options.mode = FT_TT_LEDGER_BFP8;
    options.format_map = good;
    options.format_override_name_contains = "ffn_down";
    options.format_override = FT_TT_FORMAT_BFP4_B;
    remove("build/fmap_refused.fttw");
    check(ft_tt_convert_gguf(FIXTURE, "build/fmap_refused.fttw", &options) == FT_TT_CONVERT_EINVAL &&
              !file_exists("build/fmap_refused.fttw"),
          "a map together with --format-override is refused");
}

static void test_caution(const ft_tt_format_map *map)
{
    char caution[1024];

    section("the caution names the map");
    ft_tt_format_map_caution(map, FT_TT_LEDGER_BFP8, caution, sizeof(caution));
    printf("        %s\n", caution);
    check(strstr(caution, "'t.map'") != NULL, "names the map");
    check(strstr(caution, "5 tensors at bfp4_b and 1 at bfp8_b") != NULL, "gives its counts");
    check(strstr(caution, "NOT the 'bfp8' rung") != NULL, "says the artifact is not the rung");
}

int main(void)
{
    static const char good_text[] = "blk.0.ffn_up.weight bfp4_b      # Q8_0 source\n"
                                    "blk.0.attn_qkv.weight bfp4_b    # Q8_0, GDN in\n"
                                    "blk.1.attn_output.weight bfp4_b # f32 source\n"
                                    "output.weight bfp4_b            # LM head\n"
                                    "blk.0.ssm_out.weight bfp8_b     # already bfp8: a no-op\n"
                                    "blk.2.nextn.eh_proj.weight bfp4_b\n";
    ft_tt_format_map good;
    char error[FT_TT_FORMAT_MAP_ERROR_BYTES];

    test_parse_accepts();
    test_parse_refuses();
    test_map_refusal_rule();

    memset(&good, 0, sizeof(good));
    section("fixture");
    check(synthetic_hybrid_checkpoint_write(FIXTURE) == 0, "wrote the synthetic Qwen3.8-shaped Q8_0 checkpoint");
    check(parse_text(good_text, &good, error) == FT_TT_FORMAT_MAP_OK, "parsed the gate map");
    test_trunk_with_map(&good);
    test_mtp_with_map(&good);
    test_empty_map_changes_nothing();
    test_refusals(&good);
    test_caution(&good);
    ft_tt_format_map_free(&good);

    printf("\n%d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed ? 1 : 0;
}
