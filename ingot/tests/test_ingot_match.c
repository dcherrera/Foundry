/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_match.c
 * @brief  Match mode (format version 4): contexts against a direct integer reference and a golden
 *         hash (same under -O0 and -O2); planted cross-tensor structure found by the search, coded
 *         smaller than plain and round-tripped bit-identical (row units with a self source, column
 *         units); external refs across files (stream and random access; missing or wrong refs
 *         refused); red tests (corrupt side stream, param hash or block; version 3 still decodes).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lib/test.h"
#include "ingot/codec/ingot_match.h"
#include "ingot/codec/ingot_plain.h"
#include "ingot/codec/ingot_tensor_codec.h"
#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_reader.h"
#include "ingot/container/ingot_stream.h"
#include "ingot/container/ingot_writer.h"
#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"
#include "safetensors/safetensors.h"
#include "ingot/tests/ingot_test_files.h"

#define DIR "build/test_data"

static uint16_t *gauss_bf16(size_t n, double sigma) {
    uint16_t *w = malloc(n * sizeof(*w));
    for (size_t i = 0; i < n; i++)
        w[i] = ingot_f32_to_bf16((float)(sigma * test_gauss()));
    return w;
}

static double val(uint16_t w) { return (double)ingot_bf16_to_f32(w); }

/* ---- contexts: direct reference and golden hash ------------------------------------------ */

typedef struct {
    ingot_match_params p;
    uint16_t *d, *s0, *s1;
} ctx_case;

/* Target rows [U, H]; source 0 rows [N0, H]; source 1 columns [H, N1]; source 2 self. */
static void make_ctx_case(ctx_case *k, uint64_t U, uint64_t H) {
    const uint64_t N0 = 50, N1 = 40;
    memset(k, 0, sizeof(*k));
    k->p = (ingot_match_params){.axis = INGOT_PREDICT_ROWS, .units = U, .unit_len = H,
                                .source_count = 3, .scale_c = 30, .classes = 4,
                                .class_width = 2, .class_ref = -24};
    k->p.source[0] = (ingot_match_source){10, INGOT_PREDICT_ROWS, N0, 13, 0};
    k->p.source[1] = (ingot_match_source){11, INGOT_PREDICT_COLS, N1, 12, 0};
    k->p.source[2] = (ingot_match_source){INGOT_MATCH_SELF_REF, INGOT_PREDICT_ROWS, U, 14, 1};
    ingot_match_params_alloc(&k->p);
    k->d = gauss_bf16(U * H, 0.02);
    k->s0 = gauss_bf16(N0 * H, 0.03);
    k->s1 = gauss_bf16(H * N1, 0.01);
    for (uint64_t u = 0; u < U; u++) {
        k->p.scale[u] = (int8_t)(-26 + (int)(test_next() % 8));
        k->p.k[u] = (uint8_t)(test_next() % (INGOT_MATCH_MAX_K + 1));
        for (uint32_t j = 0; j < k->p.k[u]; j++) {
            size_t at = (size_t)u * INGOT_MATCH_MAX_K + j;
            uint8_t s = (uint8_t)(test_next() % 3);
            s = (s == 2 && u == 0) ? 0 : s;
            k->p.link[at] = s;
            k->p.index[at] = s == 2 ? test_next() % u : test_next() % k->p.source[s].units;
            k->p.coef[at] = (int16_t)((int)(test_next() % 60001) - 30000);
        }
    }
}

static void free_ctx_case(ctx_case *k) {
    ingot_match_params_release(&k->p);
    free(k->d);
    free(k->s0);
    free(k->s1);
}

/* Context of weight (u, h) computed directly from the definition. */
static uint8_t ref_context(const ctx_case *k, uint64_t u, uint64_t h) {
    const ingot_match_params *p = &k->p;
    int64_t sum = 0, t[INGOT_PREDICT_BUCKETS - 1];
    for (uint32_t j = 0; j < p->k[u]; j++) {
        size_t at = (size_t)u * INGOT_MATCH_MAX_K + j;
        const ingot_match_source *s = &p->source[p->link[at]];
        uint64_t i = p->index[at];
        uint16_t x = p->link[at] == 0   ? k->s0[i * p->unit_len + h]
                     : p->link[at] == 1 ? k->s1[h * s->units + i]
                                        : k->d[i * p->unit_len + h];
        sum += (int64_t)p->coef[at] * ingot_predict_fixed(x, p->scale_c - s->coef_exp);
    }
    ingot_predict_thresholds(p->scale[u], p->scale_c, t);
    int64_t mag = sum < 0 ? -sum : sum;
    uint32_t b = 0;
    for (int j = 0; j < INGOT_PREDICT_BUCKETS - 1; j++)
        b += mag >= t[j];
    uint32_t z = INGOT_PREDICT_BUCKETS *
                     ingot_predict_class(p->scale[u], p->classes, p->class_width, p->class_ref) +
                 b;
    return (uint8_t)((sum < 0 ? 0x80 : 0) | z);
}

static void test_contexts_match_reference(void) {
    test_rng = 31;
    ctx_case k;
    make_ctx_case(&k, 64, 300);
    ASSERT_INT_EQ(0, ingot_match_params_check(&k.p));
    ingot_match_model *m = NULL;
    ASSERT_INT_EQ(0, ingot_match_model_create(&k.p, &m));
    uint16_t *s1t = malloc(300 * 40 * sizeof(uint16_t)); /* column units, unit-major */
    ingot_match_transpose(k.s1, 300, 40, s1t);
    const uint16_t *src[3] = {k.s0, s1t, k.d};
    uint8_t *ctx = malloc(64 * 300);
    ingot_match_contexts(m, src, 7, 64 * 300 - 5, ctx);
    int bad = 0;
    for (uint64_t i = 7; i < 64 * 300 - 5; i++)
        bad += ctx[i - 7] != ref_context(&k, i / 300, i % 300);
    ASSERT_INT_EQ(0, bad);
    free(s1t);
    free(ctx);
    ingot_match_model_free(m);
    free_ctx_case(&k);
}

/* Built at -O2 and at -O0; both compare against this constant. */
#define GOLDEN_CONTEXT_HASH 0x1fa5d76c7b4a58d2ull

static void test_contexts_golden_hash(void) {
    test_rng = 4243;
    ctx_case k;
    make_ctx_case(&k, 96, 512);
    ingot_match_model *m = NULL;
    ASSERT_INT_EQ(0, ingot_match_model_create(&k.p, &m));
    uint16_t *s1t = malloc(512 * 40 * sizeof(uint16_t));
    ingot_match_transpose(k.s1, 512, 40, s1t);
    const uint16_t *src[3] = {k.s0, s1t, k.d};
    uint8_t *ctx = malloc(96 * 512);
    ingot_match_contexts(m, src, 0, 96 * 512, ctx);
    uint64_t h = ingot_hash64(ctx, 96 * 512) ^ ingot_match_param_hash(&k.p);
    if (h != GOLDEN_CONTEXT_HASH)
        printf("    context hash 0x%016llxull\n", (unsigned long long)h);
    ASSERT_TRUE(h == GOLDEN_CONTEXT_HASH);
    free(s1t);
    free(ctx);
    ingot_match_model_free(m);
    free_ctx_case(&k);
}

/* ---- codec round trips with planted structure -------------------------------------------- */

enum { H = 1024, U = 200, N0 = 300, N1 = 150 };

typedef struct {
    uint16_t *d, *s0, *s1;
    uint64_t *perm;
    ingot_entry e[3];
} planted;

/* Target row u = 0.8 s0[perm u] + 0.4 s1[:, u % N1] + 0.3 d[u - 1] + 0.3 noise (axis ROWS), or
 * the same per column without the self term (axis COLS: d is [H, U]). */
static void make_planted(planted *pl, ingot_predict_axis axis) {
    test_rng = axis == INGOT_PREDICT_ROWS ? 501 : 502;
    pl->s0 = gauss_bf16((size_t)N0 * H, 0.02);
    pl->s1 = gauss_bf16((size_t)H * N1, 0.02);
    pl->d = malloc((size_t)U * H * sizeof(uint16_t));
    pl->perm = malloc(U * sizeof(uint64_t));
    for (uint64_t u = 0; u < U; u++)
        pl->perm[u] = (u * 7 + 3) % N0;
    for (uint64_t u = 0; u < U; u++)
        for (uint64_t h = 0; h < H; h++) {
            double v = 0.8 * val(pl->s0[pl->perm[u] * H + h]) + 0.4 * val(pl->s1[h * N1 + u % N1]) +
                       0.3 * 0.02 * test_gauss();
            if (axis == INGOT_PREDICT_ROWS && u > 0)
                v += 0.3 * val(pl->d[(u - 1) * H + h]);
            pl->d[axis == INGOT_PREDICT_ROWS ? u * H + h : h * U + u] = ingot_f32_to_bf16((float)v);
        }
    pl->e[0] = (ingot_entry){.dtype = INGOT_DTYPE_BF16, .ndim = 2, .shape = {N0, H},
                             .source_size = (uint64_t)N0 * H * 2};
    pl->e[1] = (ingot_entry){.dtype = INGOT_DTYPE_BF16, .ndim = 2, .shape = {H, N1},
                             .source_size = (uint64_t)H * N1 * 2};
    uint64_t r = axis == INGOT_PREDICT_ROWS ? U : H, c = axis == INGOT_PREDICT_ROWS ? H : U;
    pl->e[2] = (ingot_entry){.dtype = INGOT_DTYPE_BF16, .mode = INGOT_MODE_MATCH, .ndim = 2,
                             .shape = {r, c}, .source_size = (uint64_t)U * H * 2};
}

static void free_planted(planted *pl) {
    free(pl->d);
    free(pl->s0);
    free(pl->s1);
    free(pl->perm);
}

static uint16_t *transpose(const uint16_t *x, uint64_t rows, uint64_t cols) {
    uint16_t *t = malloc((size_t)(rows * cols) * sizeof(*t));
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            t[j * rows + i] = x[i * cols + j];
    return t;
}

/* Search, plan and encode a planted tensor; *found = units whose candidates hold perm u. */
static int encode_planted(planted *pl, ingot_predict_axis axis, ingot_sink *out, int *found,
                          const ingot_match_params **params, ingot_match_plan **plan) {
    uint16_t *s1t = transpose(pl->s1, H, N1);
    uint16_t *dum = axis == INGOT_PREDICT_ROWS ? NULL : transpose(pl->d, H, U);
    const uint16_t *d_units = dum ? dum : pl->d;
    uint32_t ns = axis == INGOT_PREDICT_ROWS ? 3 : 2;
    ingot_match_input src[3] = {
        {0, INGOT_PREDICT_ROWS, N0, pl->s0, pl->s0, 0},
        {1, INGOT_PREDICT_COLS, N1, pl->s1, s1t, 0},
        {INGOT_MATCH_SELF_REF, INGOT_PREDICT_ROWS, U, pl->d, d_units, 1}};
    int16_t *sk[3] = {malloc((size_t)N0 * INGOT_MATCH_SKETCH_DIM * 2),
                      malloc((size_t)N1 * INGOT_MATCH_SKETCH_DIM * 2),
                      malloc((size_t)U * INGOT_MATCH_SKETCH_DIM * 2)};
    ingot_match_sketch(pl->s0, N0, H, 2, sk[0]);
    ingot_match_sketch(s1t, N1, H, 2, sk[1]);
    ingot_match_sketch(d_units, U, H, 2, sk[2]);
    ingot_match_pool pool[3] = {{sk[0], N0, 0}, {sk[1], N1, 0}, {sk[2], U, 1}};
    ingot_match_candidates cand = {16, malloc(U * 16), malloc(U * 16 * sizeof(uint64_t)), 0};
    int rc = ingot_match_search(sk[2], U, pool, ns, 2, &cand);
    *found = 0;
    for (uint64_t u = 0; u < U; u++)
        for (uint32_t j = 0; j < 16; j++)
            if (cand.source[u * 16 + j] == 0 && cand.index[u * 16 + j] == pl->perm[u]) {
                (*found)++;
                break;
            }
    if (!rc)
        rc = ingot_match_plan_create(axis, U, H, pl->d, d_units, src, ns, &cand, 2, plan);
    if (!rc) {
        uint8_t frame[16];
        ingot_store_u64(frame, 16 + ingot_match_plan_prefix_bytes(*plan));
        ingot_store_u64(frame + 8, ingot_match_plan_blocks(*plan));
        rc = ingot_sink_write(out, frame, 16);
    }
    if (!rc)
        rc = ingot_match_encode(*plan, 2, out);
    *params = rc ? NULL : ingot_match_plan_params(*plan);
    if (!rc) {
        const ingot_match_params *p = *params;
        uint32_t refs[INGOT_MATCH_MAX_SOURCES];
        for (uint32_t s = 0; s < p->source_count; s++)
            refs[s] = p->source[s].ref;
        pl->e[2].base_hash = ingot_match_ref_hash(refs, p->source_count);
    }
    for (int k = 0; k < 3; k++)
        free(sk[k]);
    free(cand.source);
    free(cand.index);
    free(s1t);
    free(dum);
    return rc;
}

/* Decode a payload given its plan's sources (ref 0, 1 or self). */
static int decode_planted(const planted *pl, const ingot_match_params *p, const uint8_t *payload,
                          uint64_t size, uint8_t *out) {
    ingot_partner_data partners[INGOT_MATCH_MAX_SOURCES];
    for (uint32_t s = 0; s < p->source_count; s++) {
        uint32_t ref = p->source[s].ref;
        partners[s] = ref == 0   ? (ingot_partner_data){&pl->e[0], (const uint8_t *)pl->s0}
                      : ref == 1 ? (ingot_partner_data){&pl->e[1], (const uint8_t *)pl->s1}
                                 : (ingot_partner_data){&pl->e[2], NULL};
    }
    return ingot_tensor_decode_payload(&pl->e[2], payload, size, 1, NULL, partners, 3, out);
}

static uint64_t plain_bytes(const uint16_t *w, uint64_t n) {
    ingot_plain_plan *pp = NULL;
    ingot_plain_plan_create(w, n, 1, 2, &pp);
    uint64_t b = ingot_plain_plan_bytes(pp) + 16;
    ingot_plain_plan_free(pp);
    return b;
}

static void check_planted(ingot_predict_axis axis) {
    planted pl;
    make_planted(&pl, axis);
    ingot_sink s = {.fd = INGOT_SINK_MEMORY};
    int found = 0;
    const ingot_match_params *p = NULL;
    ingot_match_plan *plan = NULL;
    ASSERT_INT_EQ(0, encode_planted(&pl, axis, &s, &found, &p, &plan));
    ASSERT_TRUE(found >= U * 9 / 10); /* the search finds the planted partner */
    uint64_t plain = plain_bytes(pl.d, (uint64_t)U * H);
    printf("    %s: match %llu bytes, plain %llu (%.3f)\n", axis ? "cols" : "rows",
           (unsigned long long)s.pos, (unsigned long long)plain, (double)s.pos / (double)plain);
    ASSERT_TRUE(s.pos < plain * 92 / 100); /* contexts, not residuals: part of R^2 0.9 */
    uint8_t *out = malloc((size_t)U * H * 2);
    ASSERT_INT_EQ(0, decode_planted(&pl, p, s.mem, s.pos, out));
    ASSERT_INT_EQ(0, memcmp(out, pl.d, (size_t)U * H * 2));
    free(out);
    ingot_match_plan_free(plan);
    free(s.mem);
    free_planted(&pl);
}

static void test_planted_rows_with_self_round_trip(void) { check_planted(INGOT_PREDICT_ROWS); }

static void test_planted_cols_round_trip(void) { check_planted(INGOT_PREDICT_COLS); }

/* Flip byte `at` of the payload and decode: must fail (or, never, decode the same tensor). */
static int decode_flipped(const planted *pl, const ingot_match_params *p, const ingot_sink *s,
                          uint64_t at, uint8_t flip, uint8_t *out) {
    uint8_t *copy = malloc(s->pos);
    memcpy(copy, s->mem, s->pos);
    copy[at] ^= flip;
    int rc = decode_planted(pl, p, copy, s->pos, out);
    if (!rc && memcmp(out, pl->d, (size_t)U * H * 2))
        rc = -INGOT_ECORRUPT; /* decoded something else: the caller's hash check catches it */
    free(copy);
    return rc;
}

static void test_corrupt_side_hash_or_block_fails(void) {
    planted pl;
    make_planted(&pl, INGOT_PREDICT_ROWS);
    ingot_sink s = {.fd = INGOT_SINK_MEMORY};
    int found = 0;
    const ingot_match_params *p = NULL;
    ingot_match_plan *plan = NULL;
    ASSERT_INT_EQ(0, encode_planted(&pl, INGOT_PREDICT_ROWS, &s, &found, &p, &plan));
    uint8_t *out = malloc((size_t)U * H * 2);
    uint64_t header = 16 + ingot_match_header_bytes(p->source_count);
    uint64_t side_tables = ingot_load_u64(s.mem + 16 + 72), side = ingot_load_u64(s.mem + 16 + 80);
    uint64_t side_at = ((header + side_tables + 7) & ~7ull) + 16 + 4 * INGOT_RANS_LANES;
    int caught = 0;
    for (uint64_t k = 0; k < 16 && side_at + k * 7 < side_at - 136 + side; k++)
        caught += decode_flipped(&pl, p, &s, side_at + k * 7, 0x10, out) != 0;
    ASSERT_INT_EQ(16, caught);                                       /* side stream */
    ASSERT_INT_EQ(-INGOT_ECORRUPT, decode_flipped(&pl, p, &s, 16 + 56, 0x01, out)); /* hash */
    ASSERT_TRUE(decode_flipped(&pl, p, &s, s.pos - 40, 0x04, out) != 0);         /* block */
    ASSERT_TRUE(decode_flipped(&pl, p, &s, 16 + 88, 0x01, out) != 0);            /* a ref */
    free(out);
    ingot_match_plan_free(plan);
    free(s.mem);
    free_planted(&pl);
}

/* ---- files: external refs across two shards ---------------------------------------------- */

enum { FH = 512, FI = 768 };

/* Shard A: layer 0 gate/up/down. Shard B: layer 1, gate rows planted on layer 0's gate rows
 * (shifted index), up on its own gate, down columns on its gate/up rows. wrong: a different
 * layer 0 (same names and shapes). */
static int write_shards(const char *a_path, const char *b_path, int wrong) {
    test_rng = wrong ? 9001 : 9000;
    uint16_t *g0 = gauss_bf16((size_t)FI * FH, 0.02), *u0 = gauss_bf16((size_t)FI * FH, 0.02);
    uint16_t *d0 = gauss_bf16((size_t)FH * FI, 0.01);
    test_rng = 9100;
    uint16_t *g1 = malloc((size_t)FI * FH * 2), *u1 = malloc((size_t)FI * FH * 2);
    uint16_t *d1 = malloc((size_t)FH * FI * 2);
    const uint16_t *g0r = g0; /* B is written only from the real A */
    for (size_t r = 0; r < FI; r++)
        for (size_t h = 0; h < FH; h++) {
            g1[r * FH + h] = ingot_f32_to_bf16(
                (float)(0.85 * val(g0r[((r + 5) % FI) * FH + h]) + 0.01 * test_gauss()));
            u1[r * FH + h] = ingot_f32_to_bf16(
                (float)(0.6 * val(g1[r * FH + h]) + 0.015 * test_gauss()));
        }
    for (size_t h = 0; h < FH; h++)
        for (size_t c = 0; c < FI; c++)
            d1[h * FI + c] = ingot_f32_to_bf16(
                (float)(0.3 * val(g1[c * FH + h]) + 0.3 * val(u1[c * FH + h]) +
                        0.006 * test_gauss()));
    int64_t s_gu[2] = {FI, FH}, s_d[2] = {FH, FI};
    fs_writer *w = fs_writer_new();
    fs_writer_add_tensor(w, "m.layers.0.mlp.down_proj.weight", FS_DTYPE_BF16, s_d, 2, d0);
    fs_writer_add_tensor(w, "m.layers.0.mlp.gate_proj.weight", FS_DTYPE_BF16, s_gu, 2, g0);
    fs_writer_add_tensor(w, "m.layers.0.mlp.up_proj.weight", FS_DTYPE_BF16, s_gu, 2, u0);
    int rc = fs_writer_save(w, a_path) ? 0 : -1;
    fs_writer_free(w);
    w = fs_writer_new();
    fs_writer_add_tensor(w, "m.layers.1.mlp.down_proj.weight", FS_DTYPE_BF16, s_d, 2, d1);
    fs_writer_add_tensor(w, "m.layers.1.mlp.gate_proj.weight", FS_DTYPE_BF16, s_gu, 2, g1);
    fs_writer_add_tensor(w, "m.layers.1.mlp.up_proj.weight", FS_DTYPE_BF16, s_gu, 2, u1);
    if (b_path && !rc)
        rc = fs_writer_save(w, b_path) ? 0 : -1;
    fs_writer_free(w);
    uint16_t *all[] = {g0, u0, d0, g1, u1, d1};
    for (size_t k = 0; k < 6; k++)
        free(all[k]);
    return rc;
}

static int encode_with_refs(const char *src, const char *dst, const char *ref, int max_version,
                            ingot_encode_stats *st) {
    ingot_base_set *refs = NULL;
    if (ref && ingot_base_set_open(&ref, 1, 2, &refs))
        return -1;
    ingot_encode_options opt = {.threads = 3, .self_check = 1, .refs = refs,
                                .max_version = max_version};
    int rc = ingot_encode_file(src, dst, &opt, st);
    ingot_base_set_close(refs);
    return rc;
}

static int decode_with_refs(const char *src, const char *dst, const char *ref) {
    ingot_base_set *refs = NULL;
    if (ref && ingot_base_set_open(&ref, 1, 2, &refs))
        return -1;
    ingot_input *in = NULL;
    ingot_stream_options opt = {.threads = 2, .refs = refs};
    int rc = ingot_input_open_path(src, &in);
    if (!rc)
        rc = ingot_stream_decode_to_file(in, dst, &opt, NULL);
    ingot_input_close(in);
    ingot_base_set_close(refs);
    return rc;
}

/* Random access: every entry decodes and matches the source bytes. */
static int random_access_matches(const char *ingot, const char *source, const char *ref) {
    ingot_reader *r = NULL;
    ingot_base_set *refs = NULL;
    if (ingot_reader_open(ingot, &r) || (ref && ingot_base_set_open(&ref, 1, 2, &refs)))
        return -1;
    ingot_reader_set_refs(r, refs);
    FILE *f = fopen(source, "rb");
    int rc = f ? 0 : -1;
    for (size_t i = 0; !rc && i < ingot_reader_count(r); i++) {
        const ingot_entry *e = ingot_reader_entry(r, i);
        uint8_t *a = malloc((size_t)e->source_size + 1), *b = malloc((size_t)e->source_size + 1);
        rc = ingot_reader_decode(r, i, NULL, 2, a);
        if (!rc && (fseek(f, (long)e->source_offset, SEEK_SET) ||
                    fread(b, 1, (size_t)e->source_size, f) != e->source_size ||
                    memcmp(a, b, (size_t)e->source_size)))
            rc = -INGOT_ECORRUPT;
        free(a);
        free(b);
    }
    if (f)
        fclose(f);
    ingot_base_set_close(refs);
    ingot_reader_close(r);
    return rc;
}

static void test_external_refs_round_trip(void) {
    ingot_encode_stats st_ref, st_alone, st_a;
    ASSERT_INT_EQ(0, write_shards(DIR "/mA.safetensors", DIR "/mB.safetensors", 0));
    ASSERT_INT_EQ(0, write_shards(DIR "/wA.safetensors", NULL, 1));
    ASSERT_INT_EQ(0, encode_with_refs(DIR "/mA.safetensors", DIR "/mA.ingot", NULL, 0, &st_a));
    ASSERT_INT_EQ(0, encode_with_refs(DIR "/mB.safetensors", DIR "/mB.ingot",
                                      DIR "/mA.safetensors", 0, &st_ref));
    ASSERT_INT_EQ(0, encode_with_refs(DIR "/mB.safetensors", DIR "/mB.alone.ingot", NULL, 0,
                                      &st_alone));
    printf("    B with refs %llu bytes, alone %llu\n", (unsigned long long)st_ref.output_bytes,
           (unsigned long long)st_alone.output_bytes);
    ASSERT_TRUE(st_ref.output_bytes < st_alone.output_bytes * 98 / 100); /* gate from layer 0 */
    ASSERT_INT_EQ(3, (int)st_ref.tensors[INGOT_MODE_MATCH]);
    uint32_t version = 0;
    ASSERT_INT_EQ(1, ingot_reader_probe(DIR "/mB.ingot", &version));
    ASSERT_INT_EQ(4, (int)version);

    ASSERT_INT_EQ(0, decode_with_refs(DIR "/mA.ingot", DIR "/mA.rebuilt", NULL));
    ASSERT_TRUE(test_files_equal(DIR "/mA.safetensors", DIR "/mA.rebuilt"));
    ASSERT_INT_EQ(0, decode_with_refs(DIR "/mB.ingot", DIR "/mB.rebuilt", DIR "/mA.rebuilt"));
    ASSERT_TRUE(test_files_equal(DIR "/mB.safetensors", DIR "/mB.rebuilt"));
    ASSERT_INT_EQ(0, random_access_matches(DIR "/mB.ingot", DIR "/mB.safetensors",
                                           DIR "/mA.safetensors"));
    ASSERT_INT_EQ(-INGOT_EBASE, decode_with_refs(DIR "/mB.ingot", DIR "/x.st", NULL));
    ASSERT_INT_EQ(-INGOT_EBASE,
                  decode_with_refs(DIR "/mB.ingot", DIR "/x.st", DIR "/wA.safetensors"));
    ASSERT_TRUE(random_access_matches(DIR "/mB.ingot", DIR "/mB.safetensors", NULL) != 0);
}

static void test_version3_writer_still_decodes(void) {
    ingot_encode_stats st;
    ASSERT_INT_EQ(0, encode_with_refs(DIR "/mB.safetensors", DIR "/mB.v3.ingot", NULL, 3, &st));
    uint32_t version = 0;
    ASSERT_INT_EQ(1, ingot_reader_probe(DIR "/mB.v3.ingot", &version));
    ASSERT_INT_EQ(3, (int)version);
    ASSERT_INT_EQ(0, (int)st.tensors[INGOT_MODE_MATCH]);
    ASSERT_INT_EQ(0, decode_with_refs(DIR "/mB.v3.ingot", DIR "/mB.v3.rebuilt", NULL));
    ASSERT_TRUE(test_files_equal(DIR "/mB.safetensors", DIR "/mB.v3.rebuilt"));
}

static void test_corrupt_file_side_stream_fails(void) {
    /* a byte inside B's first match payload prefix, past its header: decode must not succeed */
    ingot_reader *r = NULL;
    ASSERT_INT_EQ(0, ingot_reader_open(DIR "/mB.ingot", &r));
    long at = -1;
    for (size_t i = 0; i < ingot_reader_count(r) && at < 0; i++) {
        const ingot_entry *e = ingot_reader_entry(r, i);
        if (e->mode == INGOT_MODE_MATCH)
            at = (long)e->payload_offset + 16 + (long)ingot_match_header_bytes(2) + 300;
    }
    ingot_reader_close(r);
    ASSERT_TRUE(at > 0);
    ASSERT_INT_EQ(0, test_copy_flipped(DIR "/mB.ingot", DIR "/mB.bad.ingot", at, 0x22, -1));
    ASSERT_TRUE(decode_with_refs(DIR "/mB.bad.ingot", DIR "/x.st", DIR "/mA.rebuilt") != 0);
}

/* "contexts" runs only the integer tests (the -O0 build). */
int main(int argc, char **argv) {
    RUN_TEST(contexts_match_reference);
    RUN_TEST(contexts_golden_hash);
    if (argc > 1 && !strcmp(argv[1], "contexts")) {
        TEST_REPORT();
        return __ft_failed > 0 ? 1 : 0;
    }
    RUN_TEST(planted_rows_with_self_round_trip);
    RUN_TEST(planted_cols_round_trip);
    RUN_TEST(corrupt_side_hash_or_block_fails);
    RUN_TEST(external_refs_round_trip);
    RUN_TEST(version3_writer_still_decodes);
    RUN_TEST(corrupt_file_side_stream_fails);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
