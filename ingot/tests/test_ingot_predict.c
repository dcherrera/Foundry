/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_predict.c
 * @brief  Predict mode (format version 3): the integer predictor against a float-free reference
 *         and a golden context hash (same under -O0 and -O2), planted same-unit correlation for
 *         every rule (down | gate, up; up | gate; z | v; out | v, z; o | v with GQA) chosen and
 *         round-tripped by stream (file, callback, pipe) and random access with the expected
 *         size gain, shuffled partners not chosen, exact v2 sizes, and red tests (changed
 *         coefficient or scale, coefficients that matter, wrong partner field, version 2 header).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "lib/test.h"
#include "ingot/codec/ingot_block.h"
#include "ingot/codec/ingot_predict.h"
#include "ingot/container/ingot_header.h"
#include "ingot/container/ingot_reader.h"
#include "ingot/container/ingot_stream.h"
#include "ingot/container/ingot_writer.h"
#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"
#include "safetensors/safetensors.h"
#include "ingot/tests/ingot_test_files.h"

#define DIR "build/test_data"
#define MODEL DIR "/pred.safetensors"
#define MODEL_INGOT DIR "/pred.safetensors.ingot"
#define MODEL_V2 DIR "/pred.v2.ingot"

enum { H = 2048, I = 1024, V = 512, QK = 512, HD = 64, NH = 8, NKV = 2 };

/* ---- the integer predictor against a reference built from doubles (exact here) ---------- */

static int64_t ref_fixed(uint16_t b, int32_t shift) {
    const double sat = 1099511627776.0; /* 2^40 */
    uint32_t exp = (b >> 7) & 0xFFu;
    double mag = exp == 255 ? INFINITY : fabs((double)ingot_bf16_to_f32(b));
    double v = ldexp(mag, shift);
    int64_t r = v >= sat ? (int64_t)sat : (int64_t)trunc(v);
    return (b & 0x8000u) ? -r : r;
}

static void ref_thresholds(int s, int c, int64_t *out) {
    static const double t[11] = {1.0 / 16, 1.0 / 8, 1.0 / 4, 3.0 / 8, 1.0 / 2, 3.0 / 4,
                                 1.0,      1.25,    1.5,     2.0,     3.0};
    static const double root[4] = {65536, 77936, 92682, 110218};
    int q = (int)floor(s / 4.0), r = s - 4 * q;
    for (int k = 0; k < 11; k++) {
        double v = ldexp(t[k] * 256.0 * root[r], q + c - 24); /* exact: < 2^53 bits */
        out[k] = v >= ldexp(1.0, 62) ? INGOT_PREDICT_THRESHOLD_MAX : (int64_t)floor(v);
    }
}

static void test_fixed_point_matches_reference(void) {
    static const int32_t shifts[] = {-300, -140, -60, -20, 0, 7, 20, 60, 125, 140, 160, 300};
    int bad = 0;
    for (size_t k = 0; k < sizeof(shifts) / sizeof(shifts[0]); k++)
        for (uint32_t b = 0; b < 65536; b++)
            bad += ingot_predict_fixed((uint16_t)b, shifts[k]) != ref_fixed((uint16_t)b, shifts[k]);
    ASSERT_INT_EQ(0, bad);
}

static void test_thresholds_match_reference(void) {
    int bad = 0;
    for (int s = -128; s < 128; s++)
        for (int c = -40; c < 120; c += 3) {
            int64_t a[11], b[11];
            ingot_predict_thresholds((int8_t)s, c, a);
            ref_thresholds(s, c, b);
            bad += memcmp(a, b, sizeof(a)) != 0;
        }
    ASSERT_INT_EQ(0, bad);
}

/* ---- contexts: library vs a per-element reference, and a golden hash ------------------- */

typedef struct {
    ingot_predict_params params;
    uint16_t *partner[2];
    uint64_t partner_rows[2];
} ctx_case;

static uint16_t random_bf16(double sigma) {
    return ingot_f32_to_bf16((float)(sigma * test_gauss()));
}

/* A COLS case with GQA mapping on partner 0 and an offset identity on partner 1 (or ROWS). */
static void make_case(ctx_case *k, ingot_predict_axis axis, uint64_t units, uint64_t len) {
    ingot_predict_params *p = &k->params;
    *k = (ctx_case){0};
    p->geom = (ingot_predict_geometry){.axis = axis, .units = units, .unit_len = len,
                                       .partner_count = 2};
    p->geom.partner[0] = (ingot_predict_partner){.rows = units / 4, .head_dim = 8, .group = 4};
    p->geom.partner[1] = (ingot_predict_partner){.rows = units + 5, .row_offset = 5,
                                                 .head_dim = 1, .group = 1};
    p->scale_c = 22;
    p->coef_exp[0] = 18;
    p->coef_exp[1] = 17;
    p->classes = 4;
    p->class_width = 2;
    p->class_ref = -26;
    p->scale = malloc(units);
    p->coef = malloc(units * 2 * sizeof(int16_t));
    for (uint64_t u = 0; u < units; u++) {
        p->scale[u] = (int8_t)(-26 + (int)(test_next() % 13) - 6);
        p->coef[2 * u] = (int16_t)((int)(test_next() % 20001) - 10000);
        p->coef[2 * u + 1] = (int16_t)((int)(test_next() % 20001) - 10000);
    }
    for (int j = 0; j < 2; j++) {
        k->partner_rows[j] = p->geom.partner[j].rows;
        k->partner[j] = malloc(k->partner_rows[j] * len * sizeof(uint16_t));
        for (uint64_t i = 0; i < k->partner_rows[j] * len; i++)
            k->partner[j][i] = random_bf16(0.02);
    }
}

static void free_case(ctx_case *k) {
    free(k->params.scale);
    free(k->params.coef);
    free(k->partner[0]);
    free(k->partner[1]);
}

/* The context byte of flat element i, straight from the definitions. */
static uint8_t ref_context(const ctx_case *k, uint64_t i) {
    const ingot_predict_params *p = &k->params;
    const ingot_predict_geometry *g = &p->geom;
    uint64_t u = g->axis == INGOT_PREDICT_ROWS ? i / g->unit_len : i % g->units;
    uint64_t h = g->axis == INGOT_PREDICT_ROWS ? i % g->unit_len : i / g->units;
    int64_t sum = 0;
    for (int j = 0; j < 2; j++) {
        const ingot_predict_partner *q = &g->partner[j];
        uint64_t row = q->row_offset + (u / q->head_dim / q->group) * q->head_dim +
                       u % q->head_dim;
        sum += (int64_t)p->coef[2 * u + j] *
               ref_fixed(k->partner[j][row * g->unit_len + h], p->scale_c - p->coef_exp[j]);
    }
    int64_t t[11];
    ref_thresholds(p->scale[u], p->scale_c, t);
    int bucket = 0;
    for (int j = 0; j < 11; j++)
        bucket += llabs(sum) >= t[j];
    int a = p->scale[u] - p->class_ref;
    int cls = (int)floor((double)a / p->class_width) + (int)p->classes / 2;
    cls = cls < 0 ? 0 : cls >= (int)p->classes ? (int)p->classes - 1 : cls;
    return (uint8_t)((sum < 0 ? 0x80 : 0) | (cls * 12 + bucket));
}

static int contexts_match(ingot_predict_axis axis, uint64_t units, uint64_t len) {
    ctx_case k;
    make_case(&k, axis, units, len);
    ingot_predict_model *m = NULL;
    int bad = ingot_predict_model_create(&k.params, &m) ? 1 : 0;
    const uint16_t *partners[2] = {k.partner[0], k.partner[1]};
    uint64_t n = units * len, ranges[4][2] = {{0, n}, {1, n - 3}, {len + 7, 3 * len + 1},
                                             {n / 2 + 1, n / 2 + units + 2}};
    uint8_t *ctx = malloc(n);
    for (int r = 0; !bad && r < 4; r++) {
        ingot_predict_contexts(m, partners, ranges[r][0], ranges[r][1], ctx);
        for (uint64_t i = ranges[r][0]; i < ranges[r][1]; i++)
            bad += ctx[i - ranges[r][0]] != ref_context(&k, i);
    }
    free(ctx);
    ingot_predict_model_free(m);
    free_case(&k);
    return bad;
}

static void test_contexts_match_reference(void) {
    test_rng = 77;
    ASSERT_INT_EQ(0, contexts_match(INGOT_PREDICT_COLS, 96, 300));
    ASSERT_INT_EQ(0, contexts_match(INGOT_PREDICT_ROWS, 64, 257));
}

/* The same bytes on every compiler and optimization level: this suite is built at -O2 and at
 * -O0 and both compare against this constant. */
#define GOLDEN_CONTEXT_HASH 0x6e1fe1b8c1d0d672ull

static void test_contexts_golden_hash(void) {
    test_rng = 4242;
    ctx_case k;
    make_case(&k, INGOT_PREDICT_COLS, 256, 512);
    ingot_predict_model *m = NULL;
    ASSERT_INT_EQ(0, ingot_predict_model_create(&k.params, &m));
    const uint16_t *partners[2] = {k.partner[0], k.partner[1]};
    uint8_t *ctx = malloc(256 * 512);
    ingot_predict_contexts(m, partners, 0, 256 * 512, ctx);
    uint64_t h = ingot_hash64(ctx, 256 * 512);
    if (h != GOLDEN_CONTEXT_HASH)
        printf("    context hash 0x%016llxull\n", (unsigned long long)h);
    ASSERT_TRUE(h == GOLDEN_CONTEXT_HASH);
    free(ctx);
    ingot_predict_model_free(m);
    free_case(&k);
}

/* ---- a Qwen-like file with planted same-unit correlation --------------------------------- */

static const double rho_main = 0.7, rho_up = 0.6;

/* out[n] = sigma (rho * x / rms(x) + sqrt(1 - rho^2) * noise), as BF16. */
static void correlated(uint16_t *out, const double *x, size_t n, size_t stride, double rho,
                       double sigma) {
    double ss = 0;
    for (size_t k = 0; k < n; k++)
        ss += x[k] * x[k];
    double inv = ss > 0 ? 1.0 / sqrt(ss / (double)n) : 0.0;
    for (size_t k = 0; k < n; k++)
        out[k * stride] = ingot_f32_to_bf16(
            (float)(sigma * (rho * x[k] * inv + sqrt(1 - rho * rho) * test_gauss())));
}

static uint16_t *gauss_tensor(size_t n, double sigma) {
    uint16_t *t = malloc(n * sizeof(*t));
    for (size_t i = 0; i < n; i++)
        t[i] = random_bf16(sigma);
    return t;
}

/* Unit u's target from partner rows a (and b): alpha a + beta b; a shift pairs a unit with
 * another unit's partners (the uncorrelated control). */
static void mix_rows(double *x, const uint16_t *a, const uint16_t *b, size_t u, size_t units,
                     size_t shift) {
    size_t v = (u + shift) % units;
    double alpha = 0.5 + (double)(u % 7) / 7.0, beta = b ? 1.0 - (double)(u % 5) / 5.0 : 0.0;
    for (size_t h = 0; h < H; h++)
        x[h] = alpha * ingot_bf16_to_f32(a[v * H + h]) +
               (b ? beta * ingot_bf16_to_f32(b[v * H + h]) : 0.0);
}

typedef struct {
    uint16_t *gate, *up, *down, *qkv, *z, *out, *o, *v;
    uint16_t norm[HD];
} model_data;

static void make_mlp(model_data *d, int shuffled) {
    double x[H];
    d->gate = gauss_tensor((size_t)I * H, 0.02);
    d->up = malloc((size_t)I * H * sizeof(uint16_t));
    d->down = malloc((size_t)H * I * sizeof(uint16_t));
    for (size_t u = 0; u < I; u++) {
        mix_rows(x, d->gate, NULL, u, I, shuffled ? I / 4 + 1 : 0);
        correlated(d->up + u * H, x, H, 1, rho_up, 0.02);
    }
    for (size_t u = 0; u < I; u++) {
        mix_rows(x, d->gate, d->up, u, I, shuffled ? I / 2 + 1 : 0);
        correlated(d->down + u, x, H, I, rho_main, 0.015); /* column u of [H, I] */
    }
}

static void make_attention(model_data *d, int shuffled) {
    double x[H];
    d->qkv = gauss_tensor((size_t)(QK + V) * H, 0.02);
    const uint16_t *vrows = d->qkv + (size_t)QK * H;
    d->z = malloc((size_t)V * H * sizeof(uint16_t));
    d->out = malloc((size_t)H * V * sizeof(uint16_t));
    for (size_t u = 0; u < V; u++) {
        mix_rows(x, vrows, NULL, u, V, shuffled ? V / 4 + 1 : 0);
        correlated(d->z + u * H, x, H, 1, rho_main, 0.02);
    }
    for (size_t u = 0; u < V; u++) {
        mix_rows(x, vrows, d->z, u, V, shuffled ? V / 2 + 1 : 0);
        correlated(d->out + u, x, H, V, rho_main, 0.01);
    }
    d->v = gauss_tensor((size_t)NKV * HD * H, 0.02);
    d->o = malloc((size_t)H * NH * HD * sizeof(uint16_t));
    for (size_t u = 0; u < NH * HD; u++) {
        size_t row = (u / HD / (NH / NKV)) * HD + u % HD;
        mix_rows(x, d->v, NULL, row, NKV * HD, shuffled ? NKV * HD / 2 : 0);
        correlated(d->o + u, x, H, NH * HD, rho_main, 0.01);
    }
    for (int k = 0; k < HD; k++)
        d->norm[k] = ingot_f32_to_bf16(1.0f);
}

/* Tensors in the order safetensors lists them (sorted by name: down before gate and up,
 * o_proj before v_proj). shuffled: every dependent unit is built from other units' partners
 * (shifts chosen so no chain of dependents brings a correlation back). */
static int write_qwen_like(const char *path, int shuffled) {
    model_data d;
    test_rng = 99;
    make_mlp(&d, shuffled);
    make_attention(&d, shuffled);
    fs_writer *w = fs_writer_new();
    int64_t s_down[2] = {H, I}, s_gu[2] = {I, H}, s_qkv[2] = {QK + V, H}, s_z[2] = {V, H},
            s_out[2] = {H, V}, s_o[2] = {H, NH * HD}, s_v[2] = {NKV * HD, H}, s_n[1] = {HD};
    fs_writer_add_tensor(w, "m.layers.0.mlp.down_proj.weight", FS_DTYPE_BF16, s_down, 2, d.down);
    fs_writer_add_tensor(w, "m.layers.0.mlp.gate_proj.weight", FS_DTYPE_BF16, s_gu, 2, d.gate);
    fs_writer_add_tensor(w, "m.layers.0.mlp.up_proj.weight", FS_DTYPE_BF16, s_gu, 2, d.up);
    fs_writer_add_tensor(w, "m.layers.1.linear_attn.in_proj_qkv.weight", FS_DTYPE_BF16, s_qkv, 2,
                         d.qkv);
    fs_writer_add_tensor(w, "m.layers.1.linear_attn.in_proj_z.weight", FS_DTYPE_BF16, s_z, 2, d.z);
    fs_writer_add_tensor(w, "m.layers.1.linear_attn.out_proj.weight", FS_DTYPE_BF16, s_out, 2,
                         d.out);
    fs_writer_add_tensor(w, "m.layers.2.self_attn.o_proj.weight", FS_DTYPE_BF16, s_o, 2, d.o);
    fs_writer_add_tensor(w, "m.layers.2.self_attn.q_norm.weight", FS_DTYPE_BF16, s_n, 1, d.norm);
    fs_writer_add_tensor(w, "m.layers.2.self_attn.v_proj.weight", FS_DTYPE_BF16, s_v, 2, d.v);
    int rc = fs_writer_save(w, path) ? 0 : -1;
    fs_writer_free(w);
    uint16_t *all[] = {d.gate, d.up, d.down, d.qkv, d.z, d.out, d.o, d.v};
    for (size_t k = 0; k < sizeof(all) / sizeof(all[0]); k++)
        free(all[k]);
    return rc;
}

static int encode(const char *src, const char *dst, int no_predict, ingot_encode_stats *st) {
    ingot_encode_options opt = {.threads = 3, .no_predict = no_predict, .self_check = 1,
                                .exact_v2 = 1, .max_version = 3}; /* version 3 writer */
    return ingot_encode_file(src, dst, &opt, st);
}

static int stream_to_file(const char *ingot, const char *out) {
    ingot_input *in = NULL;
    ingot_stream_options opt = {.threads = 2};
    int rc = ingot_input_open_path(ingot, &in);
    if (!rc)
        rc = ingot_stream_decode_to_file(in, out, &opt, NULL);
    ingot_input_close(in);
    return rc;
}

/* Random access: every entry decodes and matches the source bytes at its offset. */
static int random_access_matches(const char *ingot, const char *source) {
    ingot_reader *r = NULL;
    int rc = ingot_reader_open(ingot, &r);
    FILE *f = fopen(source, "rb");
    for (size_t i = 0; !rc && f && i < ingot_reader_count(r); i++) {
        const ingot_entry *e = ingot_reader_entry(r, i);
        uint8_t *out = malloc((size_t)e->source_size + 1);
        uint8_t *ref = malloc((size_t)e->source_size + 1);
        rc = ingot_reader_decode(r, i, NULL, 2, out);
        fseek(f, (long)e->source_offset, SEEK_SET);
        if (!rc && (fread(ref, 1, (size_t)e->source_size, f) != e->source_size ||
                    memcmp(out, ref, (size_t)e->source_size)))
            rc = -INGOT_ECORRUPT;
        free(out);
        free(ref);
    }
    if (f)
        fclose(f);
    ingot_reader_close(r);
    return f ? rc : -INGOT_EIO;
}

/* Mode and payload size of a named entry. */
static int entry_of(const char *ingot, const char *name, ingot_entry *out, size_t *index) {
    ingot_reader *r = NULL;
    size_t i = 0;
    int rc = ingot_reader_open(ingot, &r);
    if (!rc)
        rc = ingot_reader_find(r, name, &i);
    if (!rc) {
        *out = *ingot_reader_entry(r, i);
        out->name = NULL;
        if (index)
            *index = i;
    }
    ingot_reader_close(r);
    return rc;
}

static const struct {
    const char *name;
    double rho;
} DEPENDENTS[] = {
    {"m.layers.0.mlp.down_proj.weight", 0.7},
    {"m.layers.0.mlp.up_proj.weight", 0.6},
    {"m.layers.1.linear_attn.in_proj_z.weight", 0.7},
    {"m.layers.1.linear_attn.out_proj.weight", 0.7},
    {"m.layers.2.self_attn.o_proj.weight", 0.7},
};

static void test_planted_correlation_predicted_and_exact(void) {
    mkdir(DIR, 0755);
    ASSERT_INT_EQ(0, write_qwen_like(MODEL, 0));
    ingot_encode_stats st, st2;
    ASSERT_INT_EQ(0, encode(MODEL, MODEL_INGOT, 0, &st));
    ASSERT_INT_EQ(0, encode(MODEL, MODEL_V2, 1, &st2));
    uint32_t version = 0;
    ASSERT_INT_EQ(1, ingot_reader_probe(MODEL_INGOT, &version));
    ASSERT_INT_EQ(3, (int)version);
    ASSERT_INT_EQ(1, ingot_reader_probe(MODEL_V2, &version));
    ASSERT_INT_EQ(2, (int)version);
    ASSERT_INT_EQ(5, (int)st.tensors[INGOT_MODE_PREDICT]);
    ASSERT_TRUE(st.v2_output_bytes == st2.output_bytes); /* exact v2 sizes */
    for (size_t k = 0; k < sizeof(DEPENDENTS) / sizeof(DEPENDENTS[0]); k++) {
        ingot_entry a, b;
        ASSERT_INT_EQ(0, entry_of(MODEL_INGOT, DEPENDENTS[k].name, &a, NULL));
        ASSERT_INT_EQ(0, entry_of(MODEL_V2, DEPENDENTS[k].name, &b, NULL));
        ASSERT_INT_EQ(INGOT_MODE_PREDICT, (int)a.mode);
        double n = (double)(a.source_size / 2);
        double gain = 8.0 * ((double)b.payload_size - (double)a.payload_size) / n;
        double planted = -0.5 * log2(1 - DEPENDENTS[k].rho * DEPENDENTS[k].rho);
        printf("    %-42s gain %.3f bits/weight (planted %.3f)\n", DEPENDENTS[k].name, gain,
               planted);
        ASSERT_TRUE(gain > 0.5 * planted && gain < planted + 0.05);
    }
}

static void test_round_trips(void) {
    remove(DIR "/pred.rebuilt");
    ASSERT_INT_EQ(0, stream_to_file(MODEL_INGOT, DIR "/pred.rebuilt"));
    ASSERT_TRUE(test_files_equal(MODEL, DIR "/pred.rebuilt"));
    ASSERT_INT_EQ(0, random_access_matches(MODEL_INGOT, MODEL));
    remove(DIR "/pred.v2.rebuilt");
    ASSERT_INT_EQ(0, stream_to_file(MODEL_V2, DIR "/pred.v2.rebuilt"));
    ASSERT_TRUE(test_files_equal(MODEL, DIR "/pred.v2.rebuilt"));
    ASSERT_INT_EQ(0, random_access_matches(MODEL_V2, MODEL));
}

/* The write-callback API (no seekable output) on the reordered file. */
typedef struct {
    FILE *ref;
    int mismatch;
} compare_ctx;

static int compare_write(void *ctx, const void *buf, size_t n) {
    compare_ctx *c = ctx;
    uint8_t *ref = malloc(n);
    c->mismatch |= !ref || fread(ref, 1, n, c->ref) != n || memcmp(ref, buf, n);
    free(ref);
    return 0;
}

static int stream_compare(int fd, const char *source) {
    ingot_input *in = NULL;
    compare_ctx c = {fopen(source, "rb"), 0};
    ingot_stream_options opt = {.threads = 2};
    int rc = c.ref ? ingot_input_open_fd(fd, 1, &in) : -INGOT_EIO;
    if (!rc)
        rc = ingot_stream_decode(in, compare_write, &c, &opt, NULL);
    ingot_input_close(in);
    if (!rc && (c.mismatch || fgetc(c.ref) != EOF))
        rc = -INGOT_ECORRUPT;
    if (c.ref)
        fclose(c.ref);
    return rc;
}

/* A read fd fed by a child copying `path` into a pipe (forward-only input). */
static int pipe_from_file(const char *path, pid_t *pid) {
    int fds[2];
    if (pipe(fds))
        return -1;
    *pid = fork();
    if (*pid == 0) {
        close(fds[0]);
        int in = open(path, O_RDONLY);
        char buf[1 << 16];
        ssize_t n = 0;
        while (in >= 0 && (n = read(in, buf, sizeof(buf))) > 0)
            if (write(fds[1], buf, (size_t)n) != n)
                break;
        _exit(0);
    }
    close(fds[1]);
    return fds[0];
}

static void test_callback_and_pipe_decode(void) {
    ASSERT_INT_EQ(0, stream_compare(open(MODEL_INGOT, O_RDONLY), MODEL));
    pid_t pid = -1;
    ASSERT_INT_EQ(0, stream_compare(pipe_from_file(MODEL_INGOT, &pid), MODEL));
    int status = 0;
    waitpid(pid, &status, 0);
}

static void test_shuffled_partners_not_predicted(void) {
    ASSERT_INT_EQ(0, write_qwen_like(DIR "/shuf.safetensors", 1));
    ingot_encode_stats st;
    ASSERT_INT_EQ(0, encode(DIR "/shuf.safetensors", DIR "/shuf.ingot", 0, &st));
    ASSERT_INT_EQ(0, (int)st.tensors[INGOT_MODE_PREDICT]);
    remove(DIR "/shuf.rebuilt");
    ASSERT_INT_EQ(0, stream_to_file(DIR "/shuf.ingot", DIR "/shuf.rebuilt"));
    ASSERT_TRUE(test_files_equal(DIR "/shuf.safetensors", DIR "/shuf.rebuilt"));
}

/* ---- red tests on the payload and directory ---------------------------------------------- */

/* A whole file in memory (NULL if missing). */
static uint8_t *read_file(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    uint8_t *buf = *size > 0 ? malloc((size_t)*size) : NULL;
    fseek(f, 0, SEEK_SET);
    if (buf && fread(buf, 1, (size_t)*size, f) != (size_t)*size) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    return buf;
}

/* Byte offset in MODEL_INGOT of the down_proj prefix's scale[0] and coef[0]. */
static void prefix_offsets(long *scale_at, long *coef_at, long *prefix_at) {
    ingot_entry e = {0};
    entry_of(MODEL_INGOT, DEPENDENTS[0].name, &e, NULL);
    *prefix_at = (long)e.payload_offset + INGOT_PAYLOAD_FRAME_BYTES;
    *scale_at = *prefix_at + 64 + 40 * 2;
    *coef_at = (*scale_at + I + 1) & ~1L;
}

static int decode_both(const char *path) {
    int a = stream_to_file(path, DIR "/red.rebuilt");
    int b = random_access_matches(path, MODEL);
    return a && b ? a : 0; /* both must fail */
}

static void test_changed_coefficient_or_scale_fails(void) {
    long scale_at = 0, coef_at = 0, prefix_at = 0;
    prefix_offsets(&scale_at, &coef_at, &prefix_at);
    ASSERT_INT_EQ(0, test_copy_flipped(MODEL_INGOT, DIR "/red.ingot", coef_at, 0x01, -1));
    ASSERT_INT_EQ(-INGOT_ECORRUPT, decode_both(DIR "/red.ingot"));
    ASSERT_INT_EQ(0, test_copy_flipped(MODEL_INGOT, DIR "/red.ingot", scale_at + 9, 0x01, -1));
    ASSERT_INT_EQ(-INGOT_ECORRUPT, decode_both(DIR "/red.ingot"));
}

/* With the parameter hash rewritten to match, a changed coefficient changes contexts and the
 * blocks no longer decode: the coefficients really drive the coding. */
static void test_coefficients_drive_contexts(void) {
    long scale_at = 0, coef_at = 0, prefix_at = 0;
    prefix_offsets(&scale_at, &coef_at, &prefix_at);
    long size = 0;
    uint8_t *buf = read_file(MODEL_INGOT, &size);
    ASSERT_NOT_NULL(buf);
    if (!buf)
        return;
    for (int u = 0; u < 8; u++)
        buf[coef_at + 4 * u + 1] ^= 0x40; /* partner 0 of units 0..7: +-16384 */
    ingot_hash64_state h;
    ingot_hash64_begin(&h, I + 4 * I);
    ingot_hash64_update(&h, buf + scale_at, I);
    ingot_hash64_update(&h, buf + coef_at, 4 * I);
    ingot_store_u64(buf + prefix_at + 56, ingot_hash64_end(&h));
    FILE *f = fopen(DIR "/red.ingot", "wb");
    ASSERT_TRUE(f && fwrite(buf, 1, (size_t)size, f) == (size_t)size);
    if (f)
        fclose(f);
    free(buf);
    ASSERT_INT_EQ(-INGOT_ECORRUPT, decode_both(DIR "/red.ingot"));
}

/* Rewrite the directory with entry `index`'s base_hash field replaced. */
static int write_with_field(const char *dst, size_t index, uint64_t field, uint32_t version) {
    long size = 0;
    uint8_t *buf = read_file(MODEL_INGOT, &size);
    FILE *f = NULL;
    int ok = buf != NULL;
    ingot_file_header hdr = {0};
    ingot_entry *entries = NULL;
    ok = ok && !ingot_header_parse(buf, (size_t)size, &hdr) &&
         !ingot_directory_parse(buf + hdr.dir_offset, hdr.dir_size, hdr.entry_count, &entries);
    if (ok) {
        entries[index].base_hash = field;
        uint64_t at = hdr.dir_offset;
        for (size_t i = 0; i < hdr.entry_count; i++)
            at += ingot_entry_write(&entries[i], buf + at);
        ingot_store_u32(buf + 4, version);
        f = fopen(dst, "wb");
        ok = f && fwrite(buf, 1, (size_t)size, f) == (size_t)size;
        if (f)
            fclose(f);
    }
    ingot_entries_free(entries, hdr.entry_count);
    free(buf);
    return ok ? 0 : -1;
}

static void test_wrong_partner_field_or_version_fails(void) {
    ingot_entry e = {0};
    size_t index = 0;
    ASSERT_INT_EQ(0, entry_of(MODEL_INGOT, DEPENDENTS[0].name, &e, &index));
    uint32_t partners[2] = {0};
    ASSERT_INT_EQ(2, (int)ingot_partner_list(e.base_hash, partners));
    uint32_t swapped[2] = {partners[1], partners[0]}, self[2] = {partners[0], (uint32_t)index};
    ASSERT_INT_EQ(0, write_with_field(DIR "/red.ingot", index, ingot_partner_field(swapped, 2), 3));
    ASSERT_INT_EQ(-INGOT_EFORMAT, decode_both(DIR "/red.ingot"));
    ASSERT_INT_EQ(0, write_with_field(DIR "/red.ingot", index, ingot_partner_field(self, 2), 3));
    ASSERT_INT_EQ(-INGOT_EFORMAT, decode_both(DIR "/red.ingot"));
    /* the unchanged directory as version 2: mode 6 was reserved there */
    ASSERT_INT_EQ(0, write_with_field(DIR "/red.ingot", index, e.base_hash, 2));
    ingot_reader *r = NULL;
    ASSERT_INT_EQ(-INGOT_EUNSUPPORTED, ingot_reader_open(DIR "/red.ingot", &r));
    ASSERT_INT_EQ(-INGOT_EUNSUPPORTED, stream_to_file(DIR "/red.ingot", DIR "/red.rebuilt"));
    /* sanity: the rewrite itself is faithful */
    ASSERT_INT_EQ(0, write_with_field(DIR "/red.ingot", index, e.base_hash, 3));
    ASSERT_INT_EQ(0, stream_to_file(DIR "/red.ingot", DIR "/red.rebuilt"));
    ASSERT_INT_EQ(0, random_access_matches(DIR "/red.ingot", MODEL));
}

/* "contexts" runs only the integer predictor tests (the -O0 build). */
int main(int argc, char **argv) {
    RUN_TEST(fixed_point_matches_reference);
    RUN_TEST(thresholds_match_reference);
    RUN_TEST(contexts_match_reference);
    RUN_TEST(contexts_golden_hash);
    if (argc > 1 && !strcmp(argv[1], "contexts")) {
        TEST_REPORT();
        return __ft_failed > 0 ? 1 : 0;
    }
    RUN_TEST(planted_correlation_predicted_and_exact);
    RUN_TEST(round_trips);
    RUN_TEST(callback_and_pipe_decode);
    RUN_TEST(shuffled_partners_not_predicted);
    RUN_TEST(changed_coefficient_or_scale_fails);
    RUN_TEST(coefficients_drive_contexts);
    RUN_TEST(wrong_partner_field_or_version_fails);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
