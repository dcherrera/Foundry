/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_convert_moe.c
 * @brief Host tests for the MoE conversion path (qwen36_blackhole_plan.md Phase 2): the
 *   expert roles and the moe_bfp6 rung, FTTW v3 expert stacks (per-expert N-major slices
 *   at a fixed stride), the trunk / MTP scopes with the eh_proj split, the capacity plan's
 *   MoE rung and MTP bytes, and the GGUF writer behind the metadata GGUF.
 *
 * Every gate here has a red beside it: experts interleaved by row, the swapped eh_proj
 * halves, the wrong expert decoded, a tensor read back one byte off.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_bfp.h"
#include "ft_gguf_map.h"
#include "ft_gguf_write.h"
#include "ft_tt_capacity_plan.h"
#include "ft_tt_precision_ledger.h"
#include "ft_tt_weight_convert.h"

#define MOE_K 64u      /* hidden: an expert's input */
#define MOE_N 64u      /* expert ff: an expert's output */
#define MOE_E 4u       /* experts in a stack */
#define MOE_VOCAB 96u
#define GGML_F32 0u

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

/** A deterministic, sign-varying value with a few octaves of magnitude per block. */
static float synthetic_value(uint32_t seed, uint64_t index)
{
    uint32_t h = (uint32_t)(index * 2654435761u) ^ (seed * 0x9E3779B9u);

    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return ((float)(h % 2001u) - 1000.0f) / 1000.0f * (float)(1u << (h % 3u));
}

static float *synthetic_array(uint32_t seed, uint64_t count)
{
    float *values = malloc((size_t)count * sizeof(float));

    for (uint64_t i = 0; values && i < count; i++)
        values[i] = synthetic_value(seed, i);
    return values;
}

static ft_gguf_write_tensor f32_tensor(const char *name, uint32_t n_dims, uint64_t d0, uint64_t d1, uint64_t d2,
                                       const float *data)
{
    ft_gguf_write_tensor t;

    memset(&t, 0, sizeof(t));
    t.name = name;
    t.n_dims = n_dims;
    t.dims[0] = d0;
    t.dims[1] = n_dims > 1u ? d1 : 1u;
    t.dims[2] = n_dims > 2u ? d2 : 1u;
    t.dims[3] = 1u;
    t.ggml_type = GGML_F32;
    t.data = data;
    t.byte_count = d0 * t.dims[1] * t.dims[2] * sizeof(float);
    return t;
}

/** Write a GGUF with `arch` and an optional `<arch>.block_count`. */
static int write_model(const char *path, const char *arch, uint32_t block_count, const ft_gguf_write_tensor *tensors,
                       uint32_t count)
{
    ft_gguf_kv_buffer kv;
    char key[64];
    int rc;

    memset(&kv, 0, sizeof(kv));
    ft_gguf_kv_add_string(&kv, "general.architecture", arch);
    if (block_count) {
        snprintf(key, sizeof(key), "%s.block_count", arch);
        ft_gguf_kv_add_u32(&kv, key, block_count);
    }
    rc = kv.failed ? -1 : ft_gguf_write_file(path, kv.bytes, kv.used, kv.count, tensors, count, 32u);
    ft_gguf_kv_free(&kv);
    return rc;
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

static int artifact_has_prefix(const artifact_file *a, const char *prefix)
{
    for (uint32_t i = 0; i < a->header.tensor_count; i++)
        if (strncmp(a->entries[i].name, prefix, strlen(prefix)) == 0)
            return 1;
    return 0;
}

static const unsigned char *payload(const artifact_file *a, const ft_tt_artifact_entry *e)
{
    return a->bytes + e->data_offset;
}

static int convert(const char *gguf, const char *out, ft_tt_ledger_mode mode, ft_tt_convert_scope scope,
                   const char *override_contains)
{
    ft_tt_convert_options options;

    memset(&options, 0, sizeof(options));
    options.mode = mode;
    options.scope = scope;
    if (override_contains) {
        options.format_override_name_contains = override_contains;
        options.format_override = FT_TT_FORMAT_BFP6_B;
    }
    return ft_tt_convert_gguf(gguf, out, &options);
}

/* ------------------------------------------------------------------------- */
/* Ledger                                                                    */
/* ------------------------------------------------------------------------- */

static void test_moe_ledger(void)
{
    static const struct {
        const char *name;
        ft_tt_tensor_role role;
        ft_tt_tensor_format moe_bfp6;
        ft_tt_tensor_format bfp8;
    } cases[] = {
        {"blk.0.ffn_gate_exps.weight", FT_TT_ROLE_EXPERT_UP, FT_TT_FORMAT_BFP6_B, FT_TT_FORMAT_BFP8_B},
        {"blk.0.ffn_up_exps.weight", FT_TT_ROLE_EXPERT_UP, FT_TT_FORMAT_BFP6_B, FT_TT_FORMAT_BFP8_B},
        {"blk.0.ffn_down_exps.weight", FT_TT_ROLE_EXPERT_DOWN, FT_TT_FORMAT_BFP6_B, FT_TT_FORMAT_BFP8_B},
        {"blk.0.ffn_gate_shexp.weight", FT_TT_ROLE_FFN_UP, FT_TT_FORMAT_BFP8_B, FT_TT_FORMAT_BFP8_B},
        {"blk.0.ffn_down_shexp.weight", FT_TT_ROLE_FFN_DOWN, FT_TT_FORMAT_BFP8_B, FT_TT_FORMAT_BFP8_B},
        {"blk.0.ffn_gate_inp.weight", FT_TT_ROLE_ROUTER, FT_TT_FORMAT_FP32, FT_TT_FORMAT_BFP8_B},
        {"token_embd.weight", FT_TT_ROLE_EMBEDDING, FT_TT_FORMAT_BFP8_B, FT_TT_FORMAT_BFP8_B},
        {"blk.0.attn_qkv.weight", FT_TT_ROLE_SSM_IN, FT_TT_FORMAT_BFP8_B, FT_TT_FORMAT_BFP8_B},
        {"mtp.ffn_up_exps.weight", FT_TT_ROLE_EXPERT_UP, FT_TT_FORMAT_BFP6_B, FT_TT_FORMAT_BFP8_B},
    };
    char what[160];

    section("MoE roles and the moe_bfp6 rung");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        ft_tt_tensor_role role = ft_tt_ledger_role(cases[i].name);
        uint32_t dims = role == FT_TT_ROLE_EXPERT_UP || role == FT_TT_ROLE_EXPERT_DOWN ? 3u : 2u;

        snprintf(what, sizeof(what), "%s is %s, %s at moe_bfp6, %s at bfp8", cases[i].name,
                 ft_tt_ledger_role_name(cases[i].role), ft_tt_ledger_format_name(cases[i].moe_bfp6),
                 ft_tt_ledger_format_name(cases[i].bfp8));
        check(role == cases[i].role &&
                  ft_tt_ledger_format_for(FT_TT_LEDGER_MOE_BFP6, FT_TT_FAMILY_QWEN, role, dims) ==
                      cases[i].moe_bfp6 &&
                  ft_tt_ledger_format_for(FT_TT_LEDGER_BFP8, FT_TT_FAMILY_QWEN, role, dims) == cases[i].bfp8,
              what);
    }
    check(ft_tt_ledger_format_for(FT_TT_LEDGER_MOE_BFP6, FT_TT_FAMILY_QWEN, FT_TT_ROLE_ROUTER, 1u) ==
              FT_TT_FORMAT_FP32,
          "ffn_gate_inp_shexp (1-D) stays fp32");
    check(ft_tt_ledger_mode_from_name("moe_bfp6") == FT_TT_LEDGER_MOE_BFP6 &&
              strcmp(ft_tt_ledger_mode_name(FT_TT_LEDGER_MOE_BFP6), "moe_bfp6") == 0 &&
              ft_tt_ledger_format_from_name("bfp6_b") == FT_TT_FORMAT_BFP6_B,
          "moe_bfp6 and bfp6_b round-trip through their names");
    check(FT_TT_ROLE_EXPERT_DOWN == 16 && FT_TT_FORMAT_BFP6_B == 4 && FT_TT_LEDGER_MOE_BFP6 == 5,
          "the appended enum values are the ones written into artifacts");
}

/* ------------------------------------------------------------------------- */
/* The synthetic MoE checkpoint                                              */
/* ------------------------------------------------------------------------- */

/** Source arrays, kept alive across the tests that read them. */
typedef struct {
    float *embd;      /* [K, VOCAB] */
    float *gate_exps; /* [K, N, E]  */
    float *down_exps; /* [N, K, E]  */
    float *norm;      /* [K]        */
    float *eh_proj;   /* [2K, K]    */
    float *mtp_norm;  /* [K]        */
    float *mtp_exps;  /* [K, N, E]  */
} moe_sources;

static int moe_sources_make(moe_sources *s)
{
    s->embd = synthetic_array(1, (uint64_t)MOE_K * MOE_VOCAB);
    s->gate_exps = synthetic_array(2, (uint64_t)MOE_K * MOE_N * MOE_E);
    s->down_exps = synthetic_array(3, (uint64_t)MOE_N * MOE_K * MOE_E);
    s->norm = synthetic_array(4, MOE_K);
    s->eh_proj = synthetic_array(5, (uint64_t)2u * MOE_K * MOE_K);
    s->mtp_norm = synthetic_array(6, MOE_K);
    s->mtp_exps = synthetic_array(7, (uint64_t)MOE_K * MOE_N * MOE_E);
    return s->embd && s->gate_exps && s->down_exps && s->norm && s->eh_proj && s->mtp_norm && s->mtp_exps ? 0 : -1;
}

static void moe_sources_free(moe_sources *s)
{
    free(s->embd);
    free(s->gate_exps);
    free(s->down_exps);
    free(s->norm);
    free(s->eh_proj);
    free(s->mtp_norm);
    free(s->mtp_exps);
}

/** blk.0 is a trunk layer; blk.1 is the trailing MTP block (it carries nextn.eh_proj). */
static int write_moe_model(const char *path, const moe_sources *s)
{
    ft_gguf_write_tensor t[7];

    t[0] = f32_tensor("token_embd.weight", 2, MOE_K, MOE_VOCAB, 1, s->embd);
    t[1] = f32_tensor("blk.0.ffn_gate_exps.weight", 3, MOE_K, MOE_N, MOE_E, s->gate_exps);
    t[2] = f32_tensor("blk.0.ffn_down_exps.weight", 3, MOE_N, MOE_K, MOE_E, s->down_exps);
    t[3] = f32_tensor("blk.0.attn_norm.weight", 1, MOE_K, 1, 1, s->norm);
    t[4] = f32_tensor("blk.1.nextn.eh_proj.weight", 2, 2u * MOE_K, MOE_K, 1, s->eh_proj);
    t[5] = f32_tensor("blk.1.nextn.enorm.weight", 1, MOE_K, 1, 1, s->mtp_norm);
    t[6] = f32_tensor("blk.1.ffn_up_exps.weight", 3, MOE_K, MOE_N, MOE_E, s->mtp_exps);
    return write_model(path, "qwen35moe", 2u, t, 7u);
}

/**
 * Each expert of blk.0.ffn_gate_exps as its own 2-D tensor (`x_alone{e}`), and the RED: the
 * same shape built from rows INTERLEAVED across experts (`x_inter{e}` takes stack rows
 * n * E + e), which is what a converter striding experts by row would produce.
 */
static int write_alone_model(const char *path, const moe_sources *s, float *interleaved)
{
    static char names[2 * MOE_E][48];
    ft_gguf_write_tensor t[2 * MOE_E];
    const uint64_t expert_values = (uint64_t)MOE_K * MOE_N;

    for (uint32_t e = 0; e < MOE_E; e++) {
        for (uint32_t n = 0; n < MOE_N; n++)
            memcpy(interleaved + e * expert_values + (uint64_t)n * MOE_K,
                   s->gate_exps + ((uint64_t)n * MOE_E + e) * MOE_K, MOE_K * sizeof(float));
        snprintf(names[e], sizeof(names[e]), "blk.0.ffn_up.x_alone%u.weight", e);
        snprintf(names[MOE_E + e], sizeof(names[0]), "blk.0.ffn_up.x_inter%u.weight", e);
        t[e] = f32_tensor(names[e], 2, MOE_K, MOE_N, 1, s->gate_exps + e * expert_values);
        t[MOE_E + e] = f32_tensor(names[MOE_E + e], 2, MOE_K, MOE_N, 1, interleaved + e * expert_values);
    }
    return write_model(path, "qwen35moe", 0u, t, 2u * MOE_E);
}

/* ------------------------------------------------------------------------- */
/* Expert stacks                                                             */
/* ------------------------------------------------------------------------- */

/** Expert `e` of a stack entry == the alone tensor's payload, byte for byte. */
static int expert_slice_equals(const artifact_file *stack, const ft_tt_artifact_entry *entry, uint32_t e,
                               const artifact_file *alone, const char *alone_name)
{
    const ft_tt_artifact_entry *one = artifact_find(alone, alone_name);
    uint64_t stride = (uint64_t)entry->tile_rows * entry->tile_cols * 832u;

    return one && one->byte_count == stride && memcmp(payload(stack, entry) + e * stride, payload(alone, one),
                                                      (size_t)stride) == 0;
}

static void test_expert_stack_layout(const artifact_file *trunk, const artifact_file *alone, const moe_sources *s)
{
    const ft_tt_artifact_entry *gate = artifact_find(trunk, "blk.0.ffn_gate_exps.weight");
    const ft_tt_artifact_entry *down = artifact_find(trunk, "blk.0.ffn_down_exps.weight");
    uint32_t equal = 0, red_equal = 0;
    char name[48];

    section("expert stack: per-expert N-major slices at a fixed stride (FTTW v3)");
    check(gate && down, "both stacks are in the trunk artifact");
    if (!gate || !down)
        return;
    check_u64(trunk->header.version, FT_TT_ARTIFACT_VERSION, "a MoE artifact is version 3");
    check(gate->format == FT_TT_FORMAT_BFP6_B && gate->role == FT_TT_ROLE_EXPERT_UP &&
              down->role == FT_TT_ROLE_EXPERT_DOWN && gate->n_dims == 3u && gate->dims[2] == MOE_E,
          "stored Bfp6_b, expert roles, 3-D with E experts");
    check(gate->layout == FT_TT_LAYOUT_MATMUL_NMAJOR && gate->tile_rows == MOE_K / 32u &&
              gate->tile_cols == MOE_N / 32u,
          "the grid is ONE expert's: Kt x Nt, not Kt x (E*Nt)");
    check_u64(gate->byte_count, (uint64_t)MOE_E * gate->tile_rows * gate->tile_cols * 832u,
              "byte_count == E * stride");
    check(gate->data_offset % 64u == 0u && ((uint64_t)gate->tile_rows * gate->tile_cols * 832u) % 64u == 0u,
          "the stack and every expert in it are 64 B aligned");

    for (uint32_t e = 0; e < MOE_E; e++) {
        snprintf(name, sizeof(name), "blk.0.ffn_up.x_alone%u.weight", e);
        equal += (uint32_t)expert_slice_equals(trunk, gate, e, alone, name);
        snprintf(name, sizeof(name), "blk.0.ffn_up.x_inter%u.weight", e);
        red_equal += (uint32_t)expert_slice_equals(trunk, gate, e, alone, name);
    }
    check_u64(equal, MOE_E, "experts identical to each expert converted alone");
    check_u64(red_equal, 0, "RED: experts interleaved by row match none of them");

    {
        /* Decode expert 2's first tile: transposed, so tile (k, n) is W_2[n][k]. */
        float tile[1024];
        double worst = 0.0, red_worst = 0.0;
        const uint64_t stride = (uint64_t)gate->tile_rows * gate->tile_cols * 832u;

        ft_bfp_unpack_tiles(FT_BFP_BFP6_B, payload(trunk, gate) + 2u * stride, 1, tile);
        for (uint32_t k = 0; k < 32u; k++)
            for (uint32_t n = 0; n < 32u; n++) {
                float want = s->gate_exps[((uint64_t)2u * MOE_N + n) * MOE_K + k];
                float other = s->gate_exps[((uint64_t)3u * MOE_N + n) * MOE_K + k];

                worst = fmax(worst, fabs((double)tile[k * 32u + n] - want));
                red_worst = fmax(red_worst, fabs((double)tile[k * 32u + n] - other));
            }
        printf("        expert 2 tile 0: max |decoded - source| %.4f, vs expert 3 %.4f\n", worst, red_worst);
        check(worst <= 4.0 / 32.0, "decoded expert 2 is the source within a Bfp6 step of the block max (4/32)");
        check(red_worst > 0.5, "RED: decoded expert 2 against expert 3's values is far off");
    }
}

static void test_expert_geometry_refused(void)
{
    float *values = synthetic_array(9, (uint64_t)MOE_K * 48u * MOE_E);
    ft_gguf_write_tensor t = f32_tensor("blk.0.ffn_up_exps.weight", 3, MOE_K, 48u, MOE_E, values);

    section("expert stack geometry refusal");
    check(values && write_model("build/moe_bad_n.gguf", "qwen35moe", 0u, &t, 1u) == 0, "wrote an N = 48 stack");
    check(convert("build/moe_bad_n.gguf", "build/moe_bad_n.fttw", FT_TT_LEDGER_MOE_BFP6, FT_TT_CONVERT_SCOPE_TRUNK,
                  NULL) == FT_TT_CONVERT_EEXPERTGEOMETRY,
          "an expert N that is not a multiple of 32 is refused, not padded across experts");
    free(values);
}

/* ------------------------------------------------------------------------- */
/* Scopes                                                                    */
/* ------------------------------------------------------------------------- */

static void test_trunk_and_mtp_scopes(const char *gguf, const artifact_file *trunk, const artifact_file *mtp)
{
    ft_gguf_map *map = NULL;
    ft_tt_convert_options options;
    uint32_t block = 0;

    section("trunk and MTP scopes");
    check(ft_gguf_map_open(gguf, &map) == FT_GGUF_OK, "opened the synthetic checkpoint");
    check(ft_tt_convert_mtp_block(map, &block) == 1 && block == 1u, "blk.1 is found as the MTP block");
    memset(&options, 0, sizeof(options));
    options.mode = FT_TT_LEDGER_MOE_BFP6;
    check_u64(ft_tt_convert_tensor_count_for(map, &options), trunk->header.tensor_count,
              "trunk entry count (predicted == written)");
    options.scope = FT_TT_CONVERT_SCOPE_MTP;
    check_u64(ft_tt_convert_tensor_count_for(map, &options), mtp->header.tensor_count,
              "MTP entry count (predicted == written)");
    ft_gguf_map_close(map);

    check(!artifact_has_prefix(trunk, "blk.1.") && !artifact_has_prefix(trunk, "mtp."),
          "the trunk holds nothing of the MTP block");
    check(artifact_find(trunk, "output.weight") != NULL, "the trunk still gets the tied head");
    check(!artifact_has_prefix(mtp, "blk.") && artifact_find(mtp, "output.weight") == NULL,
          "the MTP artifact holds only mtp.* names and no head");
    check(artifact_find(mtp, "mtp.enorm.weight") && artifact_find(mtp, "mtp.ffn_up_exps.weight") &&
              !artifact_find(mtp, "mtp.eh_proj.weight"),
          "nextn. is dropped from the names and eh_proj is not written whole");
    {
        const ft_tt_artifact_entry *e = artifact_find(mtp, "mtp.eh_proj_e.weight");
        const ft_tt_artifact_entry *h = artifact_find(mtp, "mtp.eh_proj_h.weight");

        check(e && h && e->dims[0] == MOE_K && e->dims[1] == MOE_K && e->format == FT_TT_FORMAT_BFP8_B,
              "eh_proj_e and eh_proj_h are [n -> n] Bfp8_b");
    }
}

/** The split halves against the same halves converted as their own tensors, and the swap red. */
static void test_eh_proj_split(const artifact_file *mtp, const moe_sources *s)
{
    float *halves = malloc((size_t)2u * MOE_K * MOE_K * sizeof(float));
    ft_gguf_write_tensor t[2];
    artifact_file ref;
    const ft_tt_artifact_entry *e = artifact_find(mtp, "mtp.eh_proj_e.weight");
    const ft_tt_artifact_entry *h = artifact_find(mtp, "mtp.eh_proj_h.weight");

    section("MTP eh_proj split along its input");
    for (uint32_t row = 0; halves && row < MOE_K; row++) {
        memcpy(halves + (uint64_t)row * MOE_K, s->eh_proj + (uint64_t)row * 2u * MOE_K, MOE_K * sizeof(float));
        memcpy(halves + (uint64_t)(MOE_K + row) * MOE_K, s->eh_proj + (uint64_t)row * 2u * MOE_K + MOE_K,
               MOE_K * sizeof(float));
    }
    t[0] = f32_tensor("x.eh_proj_e.weight", 2, MOE_K, MOE_K, 1, halves);
    t[1] = f32_tensor("x.eh_proj_h.weight", 2, MOE_K, MOE_K, 1, halves ? halves + (uint64_t)MOE_K * MOE_K : NULL);
    check(halves && write_model("build/moe_eh_ref.gguf", "qwen35moe", 0u, t, 2u) == 0 &&
              convert("build/moe_eh_ref.gguf", "build/moe_eh_ref.fttw", FT_TT_LEDGER_MOE_BFP6,
                      FT_TT_CONVERT_SCOPE_ALL, NULL) == FT_TT_CONVERT_OK &&
              artifact_read("build/moe_eh_ref.fttw", &ref) == 0,
          "converted the two halves as their own tensors");
    if (e && h && ref.bytes) {
        const ft_tt_artifact_entry *re = artifact_find(&ref, "x.eh_proj_e.weight");
        const ft_tt_artifact_entry *rh = artifact_find(&ref, "x.eh_proj_h.weight");

        check(re && rh && re->byte_count == e->byte_count &&
                  memcmp(payload(mtp, e), payload(&ref, re), (size_t)e->byte_count) == 0 &&
                  memcmp(payload(mtp, h), payload(&ref, rh), (size_t)h->byte_count) == 0,
              "eh_proj_e == input columns [0, n), eh_proj_h == [n, 2n), byte for byte");
        check(re && rh && memcmp(payload(mtp, e), payload(&ref, rh), (size_t)e->byte_count) != 0,
              "RED: the swapped halves differ");
    }
    artifact_free(&ref);
    free(halves);
}

static void test_mtp_scope_refused_without_block(void)
{
    float *values = synthetic_array(11, (uint64_t)MOE_K * MOE_K);
    ft_gguf_write_tensor t = f32_tensor("blk.0.attn_qkv.weight", 2, MOE_K, MOE_K, 1, values);

    section("MTP scope on a checkpoint without an MTP block");
    check(values && write_model("build/moe_dense.gguf", "qwen35moe", 1u, &t, 1u) == 0, "wrote a dense checkpoint");
    check(convert("build/moe_dense.gguf", "build/moe_dense.mtp.fttw", FT_TT_LEDGER_MOE_BFP6,
                  FT_TT_CONVERT_SCOPE_MTP, NULL) == FT_TT_CONVERT_ENOMTP,
          "refused with ENOMTP");
    free(values);
}

/** Ascending float order by bits' value, for a multiset comparison across a transpose. */
static int float_ascending(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;

    return (x > y) - (x < y);
}

/** Whether `count` floats of `got` are `want` as a multiset (the NMAJOR transpose permutes). */
static int same_float_multiset(const unsigned char *got, const float *want, size_t count)
{
    float *a = malloc(count * sizeof(float)), *b = malloc(count * sizeof(float));
    int same = a && b;

    if (same) {
        memcpy(a, got, count * sizeof(float));
        memcpy(b, want, count * sizeof(float));
        qsort(a, count, sizeof(float), float_ascending);
        qsort(b, count, sizeof(float), float_ascending);
        same = memcmp(a, b, count * sizeof(float)) == 0;
    }
    free(a);
    free(b);
    return same;
}

/**
 * D4: the router is fp32 at moe_bfp6 — its tiles hold the source values exactly — and the
 * capacity plan counts an fp32 router as loadable (an fp32 tile anywhere else is not).
 */
static void test_router_fp32(const moe_sources *s)
{
    const uint64_t router_out = 64u, values = (uint64_t)MOE_K * router_out;
    float *router = synthetic_array(23, values);
    ft_gguf_write_tensor t[2];
    artifact_file moe, red;
    ft_tt_capacity_rung rung;
    ft_gguf_map *map = NULL;

    section("router (ffn_gate_inp) fp32 at moe_bfp6 (D4)");
    memset(&moe, 0, sizeof(moe));
    memset(&red, 0, sizeof(red));
    t[0] = f32_tensor("blk.0.ffn_gate_inp.weight", 2, MOE_K, router_out, 1, router);
    t[1] = f32_tensor("blk.0.ffn_gate_exps.weight", 3, MOE_K, MOE_N, MOE_E, s->gate_exps);
    check(router && write_model("build/moe_router.gguf", "qwen35moe", 0u, t, 2u) == 0 &&
              convert("build/moe_router.gguf", "build/moe_router.fttw", FT_TT_LEDGER_MOE_BFP6,
                      FT_TT_CONVERT_SCOPE_ALL, NULL) == FT_TT_CONVERT_OK &&
              artifact_read("build/moe_router.fttw", &moe) == 0 &&
              convert("build/moe_router.gguf", "build/moe_router_bfp8.fttw", FT_TT_LEDGER_BFP8,
                      FT_TT_CONVERT_SCOPE_ALL, NULL) == FT_TT_CONVERT_OK &&
              artifact_read("build/moe_router_bfp8.fttw", &red) == 0,
          "converted a router + one expert stack at moe_bfp6, and at bfp8 (the red)");
    if (moe.bytes && red.bytes) {
        const ft_tt_artifact_entry *e = artifact_find(&moe, "blk.0.ffn_gate_inp.weight");
        const ft_tt_artifact_entry *r = artifact_find(&red, "blk.0.ffn_gate_inp.weight");

        check(e && e->format == FT_TT_FORMAT_FP32 && e->byte_count == values * sizeof(float),
              "the router entry is fp32 tiles, 4 B a weight");
        check(e && same_float_multiset(payload(&moe, e), router, (size_t)values),
              "its tiles hold exactly the source values (a multiset: the K x N transpose permutes)");
        check(r && r->format == FT_TT_FORMAT_BFP8_B, "RED: at bfp8 the router entry is Bfp8_b");
    }
    check(ft_gguf_map_open("build/moe_router.gguf", &map) == FT_GGUF_OK &&
              ft_tt_capacity_rung_measure(map, FT_TT_FAMILY_QWEN, FT_TT_LEDGER_MOE_BFP6, &rung) ==
                  FT_TT_CAPACITY_OK &&
              rung.loadable,
          "moe_bfp6 with an fp32 router is loadable");
    if (map)
        ft_gguf_map_close(map);
    map = NULL;
    t[0] = f32_tensor("blk.0.attn_qkv.weight", 2, MOE_K, router_out, 1, router);
    check(write_model("build/moe_fp32_dense.gguf", "qwen35moe", 0u, t, 2u) == 0 &&
              ft_gguf_map_open("build/moe_fp32_dense.gguf", &map) == FT_GGUF_OK &&
              ft_tt_capacity_rung_measure(map, FT_TT_FAMILY_QWEN, FT_TT_LEDGER_FP32, &rung) == FT_TT_CAPACITY_OK &&
              !rung.loadable,
          "RED: an fp32 tile on a non-router matmul is not loadable");
    if (map)
        ft_gguf_map_close(map);
    artifact_free(&moe);
    artifact_free(&red);
    free(router);
}

/* ------------------------------------------------------------------------- */
/* Capacity                                                                  */
/* ------------------------------------------------------------------------- */

static void test_capacity(const char *gguf, const artifact_file *trunk, const artifact_file *mtp)
{
    ft_tt_capacity_selection selection;
    ft_tt_capacity_budget budget;
    ft_tt_capacity_rung rung, bfp8;
    ft_gguf_map *map = NULL;
    char line[FT_TT_CAPACITY_REASON_BYTES];

    section("capacity plan: the MoE rung and the MTP head");
    check(ft_gguf_map_open(gguf, &map) == FT_GGUF_OK, "opened the synthetic checkpoint");
    ft_tt_capacity_rung_measure(map, FT_TT_FAMILY_QWEN, FT_TT_LEDGER_MOE_BFP6, &rung);
    ft_tt_capacity_rung_measure(map, FT_TT_FAMILY_QWEN, FT_TT_LEDGER_BFP8, &bfp8);
    check_u64(rung.artifact_bytes, trunk->header.directory_offset - trunk->header.data_offset,
              "predicted trunk payload == the written trunk's");
    check_u64(rung.mtp_bytes, mtp->header.directory_offset - mtp->header.data_offset,
              "predicted MTP payload == the written MTP artifact's");
    check(rung.loadable && rung.expert_stacks == 2u, "moe_bfp6 is loadable (832 B only on expert stacks)");

    memset(&budget, 0, sizeof(budget));
    budget.card_bytes = bfp8.artifact_bytes + bfp8.mtp_bytes - 1u;
    check(ft_tt_capacity_select(map, FT_TT_FAMILY_QWEN, &budget, &selection) == FT_TT_CAPACITY_OK &&
              selection.mode == FT_TT_LEDGER_MOE_BFP6,
          "one byte short of bfp8 (trunk + MTP) selects moe_bfp6, not bfp4");
    ft_tt_capacity_selection_line(&selection, &budget, line, sizeof(line));
    printf("        %s\n", line);
    budget.card_bytes = rung.artifact_bytes + rung.mtp_bytes;
    budget.kv_bytes_per_token = 22528u;
    budget.context_tokens = 1u;
    check(ft_tt_capacity_select(map, FT_TT_FAMILY_QWEN, &budget, &selection) == FT_TT_CAPACITY_OK &&
              selection.mode == FT_TT_LEDGER_BFP4,
          "the MTP bytes and one KV position are charged: moe_bfp6 exactly full falls to bfp4");
    budget.card_bytes = 1024u;
    check(ft_tt_capacity_select(map, FT_TT_FAMILY_QWEN, &budget, &selection) == FT_TT_CAPACITY_ENOFIT &&
              strstr(selection.reason, "moe_bfp6") != NULL,
          "nothing fits: the refusal names the moe_bfp6 arithmetic too");
    ft_gguf_map_close(map);
}

/* ------------------------------------------------------------------------- */
/* GGUF writer                                                               */
/* ------------------------------------------------------------------------- */

static void test_gguf_writer_round_trip(const char *gguf)
{
    ft_gguf_map *source = NULL, *copy = NULL;
    const void *kv_a, *kv_b;
    uint64_t bytes_a, bytes_b, count_a, count_b;
    ft_gguf_write_tensor t;
    const ft_gguf_tensor *norm;
    int same = 0, shifted_same = 1;

    section("GGUF writer: verbatim key/value section, tensors read back");
    check(ft_gguf_map_open(gguf, &source) == FT_GGUF_OK, "opened the source");
    norm = ft_gguf_map_find(source, "blk.0.attn_norm.weight");
    ft_gguf_map_metadata_raw(source, &kv_a, &bytes_a, &count_a);
    memset(&t, 0, sizeof(t));
    t.name = norm->name;
    t.n_dims = norm->n_dims;
    memcpy(t.dims, norm->dims, sizeof(t.dims));
    t.ggml_type = norm->ggml_type;
    t.data = norm->data;
    t.byte_count = norm->byte_count;
    check(ft_gguf_write_file("build/moe_meta.gguf", kv_a, bytes_a, count_a, &t, 1u, ft_gguf_map_alignment(source)) ==
              FT_GGUF_WRITE_OK &&
              ft_gguf_map_open("build/moe_meta.gguf", &copy) == FT_GGUF_OK,
          "wrote and reopened a one-tensor metadata GGUF");
    if (copy) {
        const ft_gguf_tensor *got = ft_gguf_map_find(copy, "blk.0.attn_norm.weight");
        uint32_t block = 0;

        ft_gguf_map_metadata_raw(copy, &kv_b, &bytes_b, &count_b);
        check(bytes_a == bytes_b && count_a == count_b && memcmp(kv_a, kv_b, (size_t)bytes_a) == 0,
              "the key/value section is byte-identical");
        same = got && got->byte_count == norm->byte_count && memcmp(got->data, norm->data, (size_t)got->byte_count) == 0;
        check(same, "the kept tensor reads back byte-identical");
        check(ft_tt_convert_mtp_block(copy, &block) == 0, "with eh_proj gone, no MTP block is claimed");
        ft_gguf_map_close(copy);
    }
    t.data = (const unsigned char *)norm->data + 1;
    if (ft_gguf_write_file("build/moe_meta_red.gguf", kv_a, bytes_a, count_a, &t, 1u, 32u) == FT_GGUF_WRITE_OK &&
        ft_gguf_map_open("build/moe_meta_red.gguf", &copy) == FT_GGUF_OK) {
        const ft_gguf_tensor *got = ft_gguf_map_find(copy, "blk.0.attn_norm.weight");

        shifted_same = got && memcmp(got->data, norm->data, (size_t)got->byte_count) == 0;
        ft_gguf_map_close(copy);
    }
    check(!shifted_same, "RED: a tensor written one byte off does not read back equal");
    ft_gguf_map_close(source);
}

int main(void)
{
    moe_sources sources;
    float *interleaved = malloc((size_t)MOE_K * MOE_N * MOE_E * sizeof(float));
    artifact_file trunk, mtp, alone;
    const char *gguf = "build/moe_model.gguf";

    memset(&trunk, 0, sizeof(trunk));
    memset(&mtp, 0, sizeof(mtp));
    memset(&alone, 0, sizeof(alone));
    test_moe_ledger();

    section("synthetic MoE checkpoint");
    check(interleaved && moe_sources_make(&sources) == 0 && write_moe_model(gguf, &sources) == 0,
          "wrote a 2-block checkpoint (blk.1 = MTP)");
    check(convert(gguf, "build/moe_model.fttw", FT_TT_LEDGER_MOE_BFP6, FT_TT_CONVERT_SCOPE_TRUNK, NULL) ==
                  FT_TT_CONVERT_OK &&
              artifact_read("build/moe_model.fttw", &trunk) == 0,
          "converted the trunk at moe_bfp6");
    check(convert(gguf, "build/moe_model.mtp.fttw", FT_TT_LEDGER_MOE_BFP6, FT_TT_CONVERT_SCOPE_MTP, NULL) ==
                  FT_TT_CONVERT_OK &&
              artifact_read("build/moe_model.mtp.fttw", &mtp) == 0,
          "converted the MTP block at moe_bfp6");
    check(write_alone_model("build/moe_alone.gguf", &sources, interleaved) == 0 &&
              convert("build/moe_alone.gguf", "build/moe_alone.fttw", FT_TT_LEDGER_BFP8, FT_TT_CONVERT_SCOPE_ALL,
                      "x_") == FT_TT_CONVERT_OK &&
              artifact_read("build/moe_alone.fttw", &alone) == 0,
          "converted each expert alone (and the interleaved red) at Bfp6_b");

    if (trunk.bytes && mtp.bytes && alone.bytes) {
        test_expert_stack_layout(&trunk, &alone, &sources);
        test_trunk_and_mtp_scopes(gguf, &trunk, &mtp);
        test_eh_proj_split(&mtp, &sources);
        test_capacity(gguf, &trunk, &mtp);
    }
    test_router_fp32(&sources);
    test_expert_geometry_refused();
    test_mtp_scope_refused_without_block();
    test_gguf_writer_round_trip(gguf);

    artifact_free(&trunk);
    artifact_free(&mtp);
    artifact_free(&alone);
    moe_sources_free(&sources);
    free(interleaved);
    printf("\n%d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed ? 1 : 0;
}
