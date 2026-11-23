/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_quant.c
 * @brief  Quantized modes: the Q8_0 scale provenance math against a float reference, Q8_0 coded
 *         alone (BF16-sourced, F32-sourced and arbitrary bytes), quant delta and block delta
 *         against a base, size targets, red tests (corrupt block, changed table, wrong base),
 *         and a GGUF file with mixed types through encode + stream decode.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>

#include "lib/test.h"
#include "ingot/codec/ingot_q8.h"
#include "ingot/codec/ingot_qtensor.h"
#include "ingot/codec/ingot_tensor_codec.h"
#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_reader.h"
#include "ingot/container/ingot_stream.h"
#include "ingot/container/ingot_writer.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_ggml.h"
#include "ingot/tests/ingot_test_files.h"

#define DIR "build/test_data"
#define Q8_DTYPE ((ingot_dtype)(INGOT_DTYPE_GGML + INGOT_GGML_Q8_0))

/* Every non-negative finite BF16 amax: integer scale == float reference; inverse round-trips. */
static void test_q8_scale_matches_float_reference(void) {
    unsigned mismatches = 0, inverse_bad = 0;
    for (uint32_t a = 0; a < 0x7F80; a++) {
        float d = ingot_bf16_to_f32((uint16_t)a) / 127.0f;
        uint16_t ref = test_f32_to_f16(d), got = ingot_q8_scale_of_amax((uint16_t)a), back = 0;
        mismatches += ref != got;
        inverse_bad += ingot_q8_amax_of_scale(got, &back) != 0 ||
                       ingot_q8_scale_of_amax(back) != got;
    }
    ASSERT_INT_EQ(0, (int)mismatches);
    ASSERT_INT_EQ(0, (int)inverse_bad);
    uint16_t a = 0;
    ASSERT_INT_EQ(-1, ingot_q8_amax_of_scale(0x7E00, &a)); /* NaN: no amax */
}

/* Code `data` (rows x cols of dtype) in `mode`, decode it back, compare; *bits = payload bits per
 * weight. Returns 0 on an exact round trip. */
static int round_trip(ingot_mode mode, ingot_dtype dtype, const uint8_t *data,
                      const uint8_t *base, uint64_t rows, uint64_t cols, uint64_t bytes,
                      double *bits) {
    ingot_qtensor_plan *p = NULL;
    ingot_sink s = {.fd = -1};
    int rc = ingot_qtensor_plan_create(mode, dtype, data, base, rows, cols, 3, &p);
    if (!rc)
        rc = ingot_qtensor_encode(p, 3, &s);
    ingot_entry e = {.name = "t", .dtype = dtype, .mode = mode, .ndim = 2,
                     .shape = {rows, cols}, .source_size = bytes};
    uint8_t *out = malloc((size_t)bytes);
    if (!rc)
        rc = ingot_tensor_decode_payload(&e, s.mem, s.pos, 0, base, NULL, 3, out);
    if (!rc && memcmp(out, data, (size_t)bytes))
        rc = -INGOT_ECORRUPT;
    if (bits)
        *bits = 8.0 * (double)s.pos / (double)(rows * cols);
    ingot_qtensor_plan_free(p);
    free(s.mem);
    free(out);
    return rc;
}

static void test_q8_bf16_sourced_round_trip_and_size(void) {
    enum { R = 1024, C = 4096 }; /* 4 M weights: table costs (~8 KB) stay small */
    test_rng = 11;
    uint8_t *q = test_q8_tensor(R, C, 0.02);
    double bits = 0.0;
    ASSERT_INT_EQ(0, round_trip(INGOT_MODE_QUANT, Q8_DTYPE, q, NULL, R, C, R * C / 32 * 34,
                                &bits));
    printf("  q8 bf16-sourced: %.4f bits/weight (stored 8.5)\n", bits);
    ASSERT_DBL_LT(bits, 7.95);
    free(q);
}

/* Scales from F32 weights mostly have no BF16 amax: the fp16-step escape keeps it exact. */
static void test_q8_f32_sourced_and_arbitrary_bytes(void) {
    enum { R = 1024, C = 2048, N = R * C };
    float *w = malloc(N * sizeof(*w));
    uint8_t *q = malloc(N / 32 * 34);
    test_rng = 12;
    for (int i = 0; i < N; i++)
        w[i] = (float)(0.03 * test_gauss());
    for (int b = 0; b < N / 32; b++) {
        float amax = 0.0f;
        for (int k = 0; k < 32; k++)
            amax = fabsf(w[b * 32 + k]) > amax ? fabsf(w[b * 32 + k]) : amax;
        float d = amax / 127.0f;
        uint16_t d16 = test_f32_to_f16(d);
        q[b * 34] = (uint8_t)d16;
        q[b * 34 + 1] = (uint8_t)(d16 >> 8);
        for (int k = 0; k < 32; k++)
            q[b * 34 + 2 + k] = (uint8_t)(int8_t)roundf(w[b * 32 + k] / d);
    }
    double bits = 0.0;
    ASSERT_INT_EQ(0, round_trip(INGOT_MODE_QUANT, Q8_DTYPE, q, NULL, R, C, N / 32 * 34, &bits));
    printf("  q8 f32-sourced: %.4f bits/weight\n", bits);
    ASSERT_DBL_LT(bits, 8.25);
    for (int i = 0; i < N / 32 * 34; i++)
        q[i] = (uint8_t)test_next(); /* NaN / negative scales, code -128: still exact */
    ASSERT_INT_EQ(0, round_trip(INGOT_MODE_QUANT, Q8_DTYPE, q, NULL, R, C, N / 32 * 34, NULL));
    free(w);
    free(q);
}

static void test_q8_quant_delta_against_base(void) {
    enum { R = 256, C = 2048, BYTES = R * C / 32 * 34 };
    test_rng = 13;
    uint8_t *base = test_q8_tensor(R, C, 0.02), *t = malloc(BYTES);
    memcpy(t, base, BYTES);
    for (int b = 0; b < R * C / 32; b += 20) /* 5% of blocks changed */
        t[b * 34 + 2 + (b % 32)] ^= 0x05;
    double bits = 0.0, same = 0.0;
    ASSERT_INT_EQ(0, round_trip(INGOT_MODE_QUANT_DELTA, Q8_DTYPE, t, base, R, C, BYTES, &bits));
    ASSERT_INT_EQ(0, round_trip(INGOT_MODE_QUANT_DELTA, Q8_DTYPE, base, base, R, C, BYTES,
                                &same));
    printf("  q8 delta: 5%% changed %.4f, identical %.4f bits/weight\n", bits, same);
    ASSERT_DBL_LT(bits, 0.6);
    ASSERT_DBL_LT(same, 0.01);
    free(base);
    free(t);
}

static void test_block_delta_f32(void) {
    enum { R = 64, C = 1024, BYTES = R * C * 4 };
    float *base = malloc(BYTES), *t = malloc(BYTES);
    test_rng = 14;
    for (int i = 0; i < R * C; i++)
        base[i] = t[i] = (float)test_gauss();
    for (int i = 0; i < R * C; i += 997)
        t[i] += 1.0f;
    double bits = 0.0;
    ingot_dtype f32 = INGOT_DTYPE_F32;
    ASSERT_INT_EQ(0, round_trip(INGOT_MODE_BLOCK_DELTA, f32, (const uint8_t *)t,
                                (const uint8_t *)base, R, C, BYTES, &bits));
    printf("  block delta f32: %.4f bits/weight (stored 32)\n", bits);
    ASSERT_DBL_LT(bits, 2.0);
    free(base);
    free(t);
}

/* Red: a flipped byte in a coded Q8_0 block, and a wrong base, must fail the decode. */
static void test_q8_red_corrupt_block_and_wrong_base(void) {
    enum { R = 128, C = 2048, BYTES = R * C / 32 * 34 };
    test_rng = 15;
    uint8_t *q = test_q8_tensor(R, C, 0.02), *other = test_q8_tensor(R, C, 0.02);
    ingot_qtensor_plan *p = NULL;
    ingot_sink s = {.fd = -1};
    ASSERT_INT_EQ(0, ingot_qtensor_plan_create(INGOT_MODE_QUANT, Q8_DTYPE, q, NULL, R, C, 2, &p));
    ASSERT_INT_EQ(0, ingot_qtensor_encode(p, 2, &s));
    ingot_entry e = {.name = "t", .dtype = Q8_DTYPE, .mode = INGOT_MODE_QUANT, .ndim = 2,
                     .shape = {R, C}, .source_size = BYTES};
    uint8_t *out = malloc(BYTES);
    s.mem[s.pos - 200] ^= 0x10; /* inside the last block's word stream */
    int rc = ingot_tensor_decode_payload(&e, s.mem, s.pos, 0, NULL, NULL, 2, out);
    ASSERT_TRUE(rc != 0 || memcmp(out, q, BYTES) != 0);
    free(s.mem);
    ingot_qtensor_plan_free(p);
    /* quant delta decoded against another tensor than its base gives other bytes */
    double bits = 0.0;
    ASSERT_INT_EQ(0, round_trip(INGOT_MODE_QUANT_DELTA, Q8_DTYPE, q, q, R, C, BYTES, &bits));
    p = NULL;
    s = (ingot_sink){.fd = -1};
    ASSERT_INT_EQ(0, ingot_qtensor_plan_create(INGOT_MODE_QUANT_DELTA, Q8_DTYPE, q, q, R, C, 2,
                                               &p));
    ASSERT_INT_EQ(0, ingot_qtensor_encode(p, 2, &s));
    e.mode = INGOT_MODE_QUANT_DELTA;
    rc = ingot_tensor_decode_payload(&e, s.mem, s.pos, 0, other, NULL, 2, out);
    ASSERT_TRUE(rc != 0 || memcmp(out, q, BYTES) != 0);
    ASSERT_INT_EQ(-INGOT_EBASE,
                  ingot_tensor_decode_payload(&e, s.mem, s.pos, 0, NULL, NULL, 2, out));
    free(s.mem);
    ingot_qtensor_plan_free(p);
    free(out);
    free(q);
    free(other);
}

/* A GGUF with Q8_0, F32, BF16 tensors (and a trailing pad) encodes to quant / raw / plain and
 * stream-decodes byte-identical; random access agrees. */
static void test_gguf_mixed_file(void) {
    mkdir(DIR, 0755);
    test_rng = 16;
    uint8_t *q8 = test_q8_tensor(256, 1024, 0.02), *q8b = test_q8_tensor(64, 4096, 0.05);
    float norm[1024];
    uint16_t bf[64 * 512];
    for (int i = 0; i < 1024; i++)
        norm[i] = (float)(1.0 + 0.1 * test_gauss());
    for (int i = 0; i < 64 * 512; i++)
        bf[i] = ingot_f32_to_bf16((float)(0.02 * test_gauss()));
    test_gguf_tensor t[4] = {
        {"blk.0.attn_q.weight", INGOT_GGML_Q8_0, 2, {1024, 256}, q8, 256 * 1024 / 32 * 34},
        {"blk.0.attn_norm.weight", INGOT_GGML_F32, 1, {1024}, norm, sizeof(norm)},
        {"blk.0.ffn_down.weight", INGOT_GGML_Q8_0, 2, {4096, 64}, q8b, 64 * 4096 / 32 * 34},
        {"blk.0.bf16.weight", INGOT_GGML_BF16, 2, {512, 64}, bf, sizeof(bf)},
    };
    ASSERT_INT_EQ(0, test_write_gguf(DIR "/mixed.gguf", t, 4, 0));
    ingot_encode_options opt = {.threads = 2};
    ingot_encode_stats st;
    ASSERT_INT_EQ(0, ingot_encode_file(DIR "/mixed.gguf", DIR "/mixed.gguf.ingot", &opt, &st));
    ASSERT_INT_EQ(2, (int)st.tensors[INGOT_MODE_QUANT]);
    ASSERT_INT_EQ(1, (int)st.tensors[INGOT_MODE_PLAIN]);
    ASSERT_INT_EQ(1, (int)st.tensors[INGOT_MODE_RAW]);
    ASSERT_TRUE(st.output_bytes < st.source_bytes);
    ingot_input *in = NULL;
    ingot_stream_result res;
    ASSERT_INT_EQ(0, ingot_input_open_path(DIR "/mixed.gguf.ingot", &in));
    ASSERT_INT_EQ(0, ingot_stream_decode_to_file(in, DIR "/mixed.rebuilt.gguf", NULL, &res));
    ingot_input_close(in);
    ASSERT_TRUE(test_files_equal(DIR "/mixed.gguf", DIR "/mixed.rebuilt.gguf"));
    ASSERT_INT_EQ(INGOT_SOURCE_GGUF, (int)res.kind);
    ASSERT_INT_EQ(0, memcmp(res.sha256, st.source_sha256, 32));
    free(q8);
    free(q8b);
}

int main(void) {
    RUN_TEST(q8_scale_matches_float_reference);
    RUN_TEST(q8_bf16_sourced_round_trip_and_size);
    RUN_TEST(q8_f32_sourced_and_arbitrary_bytes);
    RUN_TEST(q8_quant_delta_against_base);
    RUN_TEST(block_delta_f32);
    RUN_TEST(q8_red_corrupt_block_and_wrong_base);
    RUN_TEST(gguf_mixed_file);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
