/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cuda_cases.h
 * @brief  Test helpers (header-only): synthetic BF16 payloads of every mode the GPU decodes,
 *         encoded by the codec itself, with the tensors and sources needed to decode them. Shared
 *         by the CPU-reference test and the GPU test, so both check the same payloads.
 *
 * Cases: plain (2-D, two blocks; 3-D, four groups), delta (identical, near and far rows),
 * predict (row units; column units from two partners; grouped-query map), match (row units with
 * a self source, in order or in 32-row strips; column units with a transposed source).
 */
#ifndef FOUNDRY_INGOT_CUDA_CASES_H
#define FOUNDRY_INGOT_CUDA_CASES_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/codec/ingot_delta.h"
#include "ingot/codec/ingot_match.h"
#include "ingot/codec/ingot_plain.h"
#include "ingot/codec/ingot_predict.h"
#include "ingot/codec/ingot_qtensor.h"
#include "ingot/codec/ingot_tensor_codec.h"
#include "ingot/core/ingot_bf16.h"
#include "ingot/cuda/ingot_cuda_image.h"
#include "ingot/cuda/ingot_cuda_reference.h"
#include "ingot/tests/ingot_test_files.h"

enum { CASE_MAX_ARRAYS = 4 };

/** One payload with what decoding it needs. arrays[i] is the tensor of directory entry i. */
typedef struct {
    const char *name;
    ingot_entry e;              /* the coded tensor (entry index CASE_MAX_ARRAYS) */
    uint8_t *payload;
    uint64_t size;
    uint16_t *data;             /* expected decoded tensor */
    uint16_t *base;             /* delta base, or NULL */
    uint16_t *arrays[CASE_MAX_ARRAYS];
    ingot_entry array_entry[CASE_MAX_ARRAYS];
} cuda_case;

static inline uint16_t *case_gauss(size_t n, double sigma) {
    uint16_t *w = malloc(n * sizeof(*w));
    for (size_t i = 0; i < n; i++)
        w[i] = ingot_f32_to_bf16((float)(sigma * test_gauss()));
    return w;
}

static inline double case_val(uint16_t w) { return (double)ingot_bf16_to_f32(w); }

static inline ingot_entry case_entry(ingot_mode mode, uint64_t rows, uint64_t cols) {
    ingot_entry e = {.dtype = INGOT_DTYPE_BF16, .mode = mode, .ndim = 2, .shape = {rows, cols},
                     .source_size = rows * cols * 2};
    return e;
}

/* Frame (u64 prefix bytes, u64 blocks) then the mode payload in a memory sink. */
static inline void case_frame(ingot_sink *s, uint64_t prefix, uint64_t blocks) {
    uint8_t frame[16];
    ingot_store_u64(frame, 16 + prefix);
    ingot_store_u64(frame + 8, blocks);
    ingot_sink_write(s, frame, 16);
}

static inline void case_finish(cuda_case *k, ingot_sink *s) {
    k->payload = s->mem;
    k->size = s->pos;
    k->e.payload_size = s->pos;
}

static inline void case_plain(cuda_case *k, const char *name, uint64_t rows, uint64_t cols,
                              int groups) {
    memset(k, 0, sizeof(*k));
    k->name = name;
    k->data = case_gauss(rows * cols, 0.02);
    k->e = case_entry(INGOT_MODE_PLAIN, rows, cols);
    if (groups) {
        k->e.ndim = 3;
        k->e.shape[0] = 4;
        k->e.shape[1] = 256;
        k->e.shape[2] = 1024;
        for (uint64_t i = 0; i < rows * cols; i++) /* a different scale per group */
            k->data[i] = ingot_f32_to_bf16((float)(case_val(k->data[i]) * (1 + i / (256 * 1024))));
    }
    ingot_plain_plan *pp = NULL;
    ingot_plain_plan_create(k->data, rows * cols, groups ? 4 : 1, 2, &pp);
    ingot_sink s = {.fd = INGOT_SINK_MEMORY};
    case_frame(&s, ingot_plain_plan_prefix_bytes(pp), ingot_plain_plan_blocks(pp));
    ingot_plain_encode(pp, 2, &s);
    ingot_plain_plan_free(pp);
    case_finish(k, &s);
}

/* Rows: every 7th identical to the base, every 11th unrelated (fallback contexts), else near. */
static inline void case_delta(cuda_case *k) {
    enum { ROWS = 260, COLS = 1024 };
    memset(k, 0, sizeof(*k));
    k->name = "delta";
    k->base = case_gauss((size_t)ROWS * COLS, 0.02);
    k->data = malloc((size_t)ROWS * COLS * 2);
    for (uint64_t i = 0; i < (uint64_t)ROWS * COLS; i++) {
        uint64_t r = i / COLS;
        double b = case_val(k->base[i]);
        double t = r % 7 == 0 ? b : r % 11 == 0 ? 0.2 * test_gauss() : b + 0.0005 * test_gauss();
        k->data[i] = r % 7 == 0 ? k->base[i] : ingot_f32_to_bf16((float)t);
    }
    k->e = case_entry(INGOT_MODE_DELTA, ROWS, COLS);
    ingot_delta_plan *dp = NULL;
    ingot_delta_plan_create(k->data, k->base, ROWS, COLS, 2, &dp);
    ingot_sink s = {.fd = INGOT_SINK_MEMORY};
    case_frame(&s, ingot_delta_plan_prefix_bytes(dp), ingot_delta_plan_blocks(dp));
    ingot_delta_encode(dp, 2, &s);
    ingot_delta_plan_free(dp);
    case_finish(k, &s);
}

/* Partner arrays [rows, H] as entries 0..J-1; D coded given them. */
static inline void case_predict_encode(cuda_case *k, const ingot_predict_geometry *g) {
    const uint16_t *partners[INGOT_PREDICT_MAX_PARTNERS] = {k->arrays[0], k->arrays[1]};
    ingot_predict_plan *pp = NULL;
    ingot_predict_plan_create(g, k->data, partners, 2, &pp);
    ingot_sink s = {.fd = INGOT_SINK_MEMORY};
    case_frame(&s, ingot_predict_plan_prefix_bytes(pp), ingot_predict_plan_blocks(pp));
    ingot_predict_encode(pp, 2, &s);
    ingot_predict_plan_free(pp);
    uint32_t idx[2] = {0, 1};
    k->e.base_hash = ingot_partner_field(idx, g->partner_count);
    case_finish(k, &s);
}

/* kind 0: up | gate (rows); 1: down | gate, up (columns); 2: o | v (columns, grouped heads). */
static inline void case_predict(cuda_case *k, int kind) {
    enum { H = 1024, I = 256, HD = 64, G = 2 };
    memset(k, 0, sizeof(*k));
    k->name = kind == 0 ? "predict-rows" : kind == 1 ? "predict-cols" : "predict-gqa";
    uint64_t prow = kind == 2 ? I / G : I;
    uint32_t np = kind == 1 ? 2 : 1;
    for (uint32_t j = 0; j < np; j++) {
        k->arrays[j] = case_gauss(prow * H, 0.02);
        k->array_entry[j] = case_entry(INGOT_MODE_RAW, prow, H);
    }
    ingot_predict_geometry g = {.axis = kind ? INGOT_PREDICT_COLS : INGOT_PREDICT_ROWS,
                                .units = I, .unit_len = H, .partner_count = np};
    for (uint32_t j = 0; j < np; j++)
        g.partner[j] = (ingot_predict_partner){j, prow, 0, kind == 2 ? HD : 1, kind == 2 ? G : 1};
    k->data = malloc((size_t)I * H * 2);
    for (uint64_t u = 0; u < I; u++)
        for (uint64_t h = 0; h < H; h++) {
            double v = 0.01 * test_gauss();
            for (uint32_t j = 0; j < np; j++)
                v += (0.6 - 0.3 * j) * case_val(k->arrays[j][ingot_predict_map(&g.partner[j], u) * H + h]);
            k->data[kind ? h * I + u : u * H + h] = ingot_f32_to_bf16((float)v);
        }
    k->e = case_entry(INGOT_MODE_PREDICT, kind ? H : I, kind ? I : H);
    case_predict_encode(k, &g);
}

static inline uint16_t *case_transpose(const uint16_t *x, uint64_t rows, uint64_t cols) {
    uint16_t *t = malloc((size_t)(rows * cols) * sizeof(*t));
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            t[j * rows + i] = x[i * cols + j];
    return t;
}

/* Sketch, pick candidates, plan and encode a match target over sources 0 ([N0, H] rows), 1 ([H, N1]
 * columns) and, for row units, itself. */
static inline void case_match_encode(cuda_case *k, ingot_predict_axis axis, uint64_t U,
                                     uint64_t H, uint64_t N0, uint64_t N1, uint64_t strip) {
    uint16_t *s1t = case_transpose(k->arrays[1], H, N1);
    uint16_t *dum = axis == INGOT_PREDICT_ROWS ? NULL : case_transpose(k->data, H, U);
    const uint16_t *d_units = dum ? dum : k->data;
    uint32_t ns = axis == INGOT_PREDICT_ROWS ? 3 : 2;
    ingot_match_input src[3] = {
        {0, INGOT_PREDICT_ROWS, N0, k->arrays[0], k->arrays[0], 0},
        {1, INGOT_PREDICT_COLS, N1, k->arrays[1], s1t, 0},
        {INGOT_MATCH_SELF_REF, INGOT_PREDICT_ROWS, U, k->data, d_units, 1}};
    size_t dim = INGOT_MATCH_SKETCH_DIM;
    int16_t *sk[3] = {malloc(N0 * dim * 2), malloc(N1 * dim * 2), malloc(U * dim * 2)};
    ingot_match_sketch(k->arrays[0], N0, H, 2, sk[0]);
    ingot_match_sketch(s1t, N1, H, 2, sk[1]);
    ingot_match_sketch(d_units, U, H, 2, sk[2]);
    ingot_match_pool pool[3] = {{sk[0], N0, 0}, {sk[1], N1, 0}, {sk[2], U, 1}};
    ingot_match_candidates cand = {16, malloc(U * 16), malloc(U * 16 * sizeof(uint64_t)), 0, strip};
    ingot_match_search(sk[2], U, pool, ns, 2, &cand);
    ingot_match_plan *plan = NULL;
    ingot_match_plan_create(axis, U, H, k->data, d_units, src, ns, &cand, 2, &plan);
    ingot_sink s = {.fd = INGOT_SINK_MEMORY};
    case_frame(&s, ingot_match_plan_prefix_bytes(plan), ingot_match_plan_blocks(plan));
    ingot_match_encode(plan, 2, &s);
    const ingot_match_params *p = ingot_match_plan_params(plan);
    uint32_t refs[INGOT_MATCH_MAX_SOURCES];
    for (uint32_t i = 0; i < p->source_count; i++)
        refs[i] = p->source[i].ref;
    k->e.base_hash = ingot_match_ref_hash(refs, p->source_count);
    ingot_match_plan_free(plan);
    for (int i = 0; i < 3; i++)
        free(sk[i]);
    free(cand.source);
    free(cand.index);
    free(s1t);
    free(dum);
    case_finish(k, &s);
}

/* Unit u = 0.8 s0[perm u] + 0.4 s1[:, u % N1] + 0.3 noise (+ 0.3 d[u - 1] for row units). */
static inline void case_match(cuda_case *k, ingot_predict_axis axis, uint64_t strip) {
    enum { H = 1024, U = 320, N0 = 300, N1 = 150 };
    memset(k, 0, sizeof(*k));
    k->name = axis == INGOT_PREDICT_COLS ? "match-cols" : strip ? "match-rows-strip" : "match-rows-self";
    k->arrays[0] = case_gauss((size_t)N0 * H, 0.02);
    k->arrays[1] = case_gauss((size_t)H * N1, 0.02);
    k->array_entry[0] = case_entry(INGOT_MODE_RAW, N0, H);
    k->array_entry[1] = case_entry(INGOT_MODE_RAW, H, N1);
    k->data = malloc((size_t)U * H * 2);
    for (uint64_t u = 0; u < U; u++)
        for (uint64_t h = 0; h < H; h++) {
            double v = 0.8 * case_val(k->arrays[0][((u * 7 + 3) % N0) * H + h]) +
                       0.4 * case_val(k->arrays[1][h * N1 + u % N1]) + 0.006 * test_gauss();
            if (axis == INGOT_PREDICT_ROWS && u > 0)
                v += 0.3 * case_val(k->data[(u - 1) * H + h]);
            k->data[axis == INGOT_PREDICT_ROWS ? u * H + h : h * U + u] = ingot_f32_to_bf16((float)v);
        }
    k->e = case_entry(INGOT_MODE_MATCH, axis == INGOT_PREDICT_ROWS ? U : H,
                      axis == INGOT_PREDICT_ROWS ? H : U);
    case_match_encode(k, axis, U, H, N0, N1, strip);
}

/* Q8_0 rows x cols (mode 3): ggml quantization of BF16 weights; every 37th block's scale moved
 * off the BF16 lattice (coded through the escape) and every 53rd block without a |code| = 127
 * (pos = 32). data holds the expected Q8_0 bytes. */
static inline void case_q8(cuda_case *k, const char *name, uint64_t rows, uint64_t cols) {
    memset(k, 0, sizeof(*k));
    k->name = name;
    uint8_t *q = test_q8_tensor(rows, cols, 0.02);
    uint64_t blocks = rows * cols / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        uint8_t *blk = q + b * 34;
        if (b % 37 == 5)
            blk[0] ^= 1;
        for (int c = 0; b % 53 == 7 && c < 32; c++)
            blk[2 + c] = (uint8_t)((int8_t)blk[2 + c] / 2);
    }
    k->data = (uint16_t *)q;
    k->e = (ingot_entry){.dtype = INGOT_DTYPE_GGML + 8, .mode = INGOT_MODE_QUANT, .ndim = 2,
                         .shape = {rows, cols}, .source_size = blocks * 34};
    ingot_qtensor_plan *pp = NULL;
    ingot_qtensor_plan_create(INGOT_MODE_QUANT, k->e.dtype, q, NULL, rows, cols, 2, &pp);
    ingot_sink s = {.fd = INGOT_SINK_MEMORY};
    case_frame(&s, ingot_qtensor_plan_prefix_bytes(pp), ingot_qtensor_plan_blocks(pp));
    ingot_qtensor_encode(pp, 2, &s);
    ingot_qtensor_plan_free(pp);
    case_finish(k, &s);
}

/* Weight r, c of a Q8_0 case, dequantized (d q in fp32). */
static inline double case_q8_val(const cuda_case *k, uint64_t r, uint64_t c) {
    const uint8_t *blk = (const uint8_t *)k->data + (r * k->e.shape[1] + c) / 32 * 34;
    return (double)ingot_cuda_f16_f32((uint32_t)(blk[0] | blk[1] << 8)) *
           (int8_t)blk[2 + c % 32];
}

enum { CASE_COUNT = 10 };

/* Build case i (0 .. CASE_COUNT - 1) with a fixed seed. */
static inline void case_build(cuda_case *k, int i) {
    test_rng = 0x5EED0000ull + (uint64_t)i;
    switch (i) {
    case 0: case_plain(k, "plain", 300, 1024, 0); break;
    case 1: case_plain(k, "plain-groups", 1024, 1024, 1); break;
    case 2: case_delta(k); break;
    case 3: case_predict(k, 0); break;
    case 4: case_predict(k, 1); break;
    case 5: case_predict(k, 2); break;
    case 6: case_match(k, INGOT_PREDICT_ROWS, 0); break;
    case 7: case_match(k, INGOT_PREDICT_COLS, 0); break;
    case 9: case_match(k, INGOT_PREDICT_ROWS, 32); break;
    default: case_plain(k, "plain-small", 48, 100, 0); break;
    }
}

static inline void case_free(cuda_case *k) {
    free(k->payload);
    free(k->data);
    free(k->base);
    for (int i = 0; i < CASE_MAX_ARRAYS; i++)
        free(k->arrays[i]);
    memset(k, 0, sizeof(*k));
}

/* The codec's own CPU decode of a case (partners in the payload's source order). */
static inline int case_codec_decode(const cuda_case *k, uint8_t *out) {
    ingot_tensor_decoder *d = NULL;
    uint64_t at = 0;
    int rc = ingot_tensor_decoder_open(&k->e, k->payload + 16, k->size - 16, &d, &at);
    if (rc)
        return rc;
    uint32_t refs[INGOT_MATCH_MAX_SOURCES];
    uint32_t n = ingot_tensor_decoder_partners(d, refs);
    ingot_tensor_decoder_close(d);
    ingot_partner_data partners[INGOT_MATCH_MAX_SOURCES];
    for (uint32_t s = 0; s < n; s++)
        partners[s] = refs[s] < CASE_MAX_ARRAYS
                          ? (ingot_partner_data){&k->array_entry[refs[s]],
                                                 (const uint8_t *)k->arrays[refs[s]]}
                          : (ingot_partner_data){&k->e, NULL};
    return ingot_tensor_decode_payload(&k->e, k->payload, k->size, 1, (const uint8_t *)k->base,
                                       n ? partners : NULL, 2, out);
}

/* Bind every non-self source of a case image to array addresses (host pointers or device
 * addresses of the case arrays, indexed like arrays[]). Returns the first bind error. */
static inline int case_bind_sources(const cuda_case *k, const ingot_cuda_image *img,
                                    ingot_cuda_params *p, const uint64_t *address) {
    for (uint32_t s = 0; p->kind == INGOT_CUDA_KIND_MATCH && s < p->source_count; s++) {
        uint32_t ref = img->shape[s].ref;
        if (img->shape[s].self)
            continue;
        if (ref >= CASE_MAX_ARRAYS || !k->arrays[ref])
            return -1;
        int rc = ingot_cuda_bind_source(img, p, s, address[ref], k->array_entry[ref].shape[0],
                                        k->array_entry[ref].shape[1]);
        if (rc)
            return rc;
    }
    return 0;
}

/** Outputs of one reference run. */
typedef struct {
    uint16_t *out;
    float *y;
    uint32_t bad_blocks;
    int rc;
} ref_result;

static inline ref_result case_reference_run(const cuda_case *k, uint32_t flags,
                                            const float *x) {
    ref_result r = {0};
    ingot_cuda_image img;
    r.rc = ingot_cuda_image_build(&k->e, k->payload, k->size, &img);
    if (r.rc)
        return r;
    ingot_cuda_params p;
    ingot_cuda_image_params(&img, (uint64_t)(uintptr_t)img.image, &p);
    uint64_t addr[CASE_MAX_ARRAYS];
    for (int i = 0; i < CASE_MAX_ARRAYS; i++)
        addr[i] = (uint64_t)(uintptr_t)k->arrays[i];
    r.rc = case_bind_sources(k, &img, &p, addr);
    uint64_t n = p.rows * p.cols;
    r.out = calloc(n, 2);
    r.y = calloc(p.rows, sizeof(float));
    float *partials = calloc(2 * p.block_count, sizeof(float));
    uint32_t *status = calloc(p.block_count, sizeof(uint32_t));
    ingot_cuda_bind_self(&p, (uint64_t)(uintptr_t)r.out);
    p.flags = flags | (img.self_source ? INGOT_CUDA_STORE : 0);
    p.out = (uint64_t)(uintptr_t)r.out;
    p.base = (uint64_t)(uintptr_t)k->base;
    p.x = (uint64_t)(uintptr_t)x;
    p.y = (uint64_t)(uintptr_t)r.y;
    p.partials = (uint64_t)(uintptr_t)partials;
    p.status = (uint64_t)(uintptr_t)status;
    if (!r.rc) {
        ingot_cuda_reference_run(&p);
        if (flags & INGOT_CUDA_MATVEC)
            ingot_cuda_combine_rows(&p);
    }
    for (uint64_t b = 0; b < p.block_count; b++)
        r.bad_blocks += status[b] != 0;
    free(partials);
    free(status);
    ingot_cuda_image_free(&img);
    return r;
}

static inline void ref_free(ref_result *r) {
    free(r->out);
    free(r->y);
}

#endif /* FOUNDRY_INGOT_CUDA_CASES_H */
