/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_kquant.c
 * @brief  K-quant coders (Q3_K..Q6_K): field split identity on random bytes and agreement with
 *         ggml's dequantize loops, quantized-tensor round trips on synthetic blocks (with and
 *         without a base), size below raw, and red tests: a corrupted coded block must fail the
 *         decode or change the output.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lib/test.h"
#include "ingot/codec/ingot_kquant_fields.h"
#include "ingot/codec/ingot_qtensor.h"
#include "ingot/codec/ingot_qtype.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_ggml.h"

static const uint32_t KQ_TYPES[4] = {INGOT_GGML_Q3_K, INGOT_GGML_Q4_K, INGOT_GGML_Q5_K,
                                     INGOT_GGML_Q6_K};

/* xorshift64* so the tests are deterministic. */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng_next(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 2685821657736338717ull;
}

static double rng_gauss(void) {
    double u = ((double)(rng_next() >> 11) + 0.5) / 9007199254740992.0;
    double v = ((double)(rng_next() >> 11) + 0.5) / 9007199254740992.0;
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

static int clampi(double v, int lo, int hi) {
    int i = (int)lrint(v);
    return i < lo ? lo : i > hi ? hi : i;
}

static uint8_t *random_bytes(size_t n) {
    uint8_t *p = malloc(n);
    for (size_t i = 0; p && i < n; i++)
        p[i] = (uint8_t)(rng_next() >> 56);
    return p;
}

/* Fields of one quantized-looking block: drifting fp16 d, structured scales, Gaussian codes. */
static void synth_fields(const ingot_kq_layout *l, uint64_t index, ingot_kq_fields *f) {
    memset(f, 0, sizeof(*f));
    f->d = (uint16_t)(0x1C00 + (index % 97) * 3 + (rng_next() >> 60));
    if (!l->has_min && rng_next() % 10 == 0)
        f->d |= 0x8000; /* Q3_K / Q6_K scales can be negative */
    f->dmin = (uint16_t)(0x1800 + (rng_next() >> 58));
    int top = (1 << l->code_bits) - 1;
    for (uint32_t k = 0; k < l->subblocks; k++) {
        double g = rng_gauss();
        if (l->scale_bits == 8)
            f->scale[k] = (uint8_t)(128 + clampi(60.0 + 25.0 * g, -128, 127));
        else if (l->has_min)
            f->scale[k] = (uint8_t)clampi(44.0 + 10.0 * g, 0, 63);
        else
            f->scale[k] = (uint8_t)clampi(48.0 + 8.0 * g, 0, 63);
        f->min[k] = l->has_min ? (uint8_t)clampi(20.0 + 8.0 * rng_gauss(), 0, 63) : 0;
        double spread = (double)(top + 1) / (4.0 + (f->scale[k] & 15) / 8.0);
        uint32_t per = INGOT_KQ_WEIGHTS / l->subblocks;
        for (uint32_t i = 0; i < per; i++)
            f->code[k * per + i] = (uint8_t)clampi(top / 2.0 + spread * rng_gauss(), 0, top);
    }
}

static uint8_t *synth_tensor(const ingot_kq_layout *l, uint64_t blocks) {
    uint8_t *t = malloc(blocks * l->block_bytes);
    ingot_kq_fields f;
    for (uint64_t b = 0; t && b < blocks; b++) {
        synth_fields(l, b, &f);
        ingot_kq_pack(l, &f, t + b * l->block_bytes);
    }
    return t;
}

static void test_fields_bytes_identity(void) {
    enum { BLOCKS = 4096 };
    for (int t = 0; t < 4; t++) {
        const ingot_kq_layout *l = ingot_kq_layout_of(KQ_TYPES[t]);
        ASSERT_NOT_NULL(l);
        uint8_t *in = random_bytes((size_t)BLOCKS * l->block_bytes);
        uint8_t out[256];
        ingot_kq_fields f;
        int bad = 0;
        for (size_t b = 0; b < BLOCKS; b++) {
            const uint8_t *blk = in + b * l->block_bytes;
            ingot_kq_unpack(l, blk, &f);
            memset(out, 0xA5, sizeof(out));
            ingot_kq_pack(l, &f, out);
            bad += memcmp(out, blk, l->block_bytes) != 0;
        }
        free(in);
        ASSERT_INT_EQ(0, bad);
    }
    ASSERT_NULL(ingot_kq_layout_of(INGOT_GGML_Q8_0));
    ASSERT_NULL(ingot_kq_layout_of(15));
}

/* Reference meaning of a block, written from ggml's dequantize_row_q3_K .. q6_K loops: per
 * weight its stored code, scale and min (stored representation), and d / dmin. */
typedef struct {
    uint16_t d, dmin;
    uint8_t code[256], scale[256], min[256];
} ggml_ref;

static uint16_t ref16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static void ref_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

static void ref_q3(const uint8_t *x, ggml_ref *r) {
    const uint8_t *hm = x, *q = x + 32;
    uint32_t aux[4], kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    memcpy(aux, x + 96, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const uint8_t *scales = (const uint8_t *)aux;
    int is = 0, y = 0;
    uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; half++, is++)
                for (int l = 16 * half; l < 16 * half + 16; ++l, ++y) {
                    r->code[y] = (uint8_t)(((q[l] >> shift) & 3) + ((hm[l] & m) ? 4 : 0));
                    r->scale[y] = scales[is];
                }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
    r->d = ref16(x + 108);
}

static void ref_q45(const uint8_t *x, ggml_ref *r, int five) {
    const uint8_t *ql = x + (five ? 48 : 16), *qh = x + 16;
    int is = 0, y = 0;
    uint8_t u1 = 1, u2 = 2, sc1, m1, sc2, m2;
    for (int j = 0; j < 256; j += 64) {
        ref_scale_min_k4(is + 0, x + 4, &sc1, &m1);
        ref_scale_min_k4(is + 1, x + 4, &sc2, &m2);
        for (int l = 0; l < 32; ++l, ++y) {
            r->code[y] = (uint8_t)((ql[l] & 0xF) + (five && (qh[l] & u1) ? 16 : 0));
            r->scale[y] = sc1;
            r->min[y] = m1;
        }
        for (int l = 0; l < 32; ++l, ++y) {
            r->code[y] = (uint8_t)((ql[l] >> 4) + (five && (qh[l] & u2) ? 16 : 0));
            r->scale[y] = sc2;
            r->min[y] = m2;
        }
        ql += 32;
        is += 2;
        u1 = (uint8_t)(u1 << 2);
        u2 = (uint8_t)(u2 << 2);
    }
    r->d = ref16(x);
    r->dmin = ref16(x + 2);
}

static void ref_q6(const uint8_t *x, ggml_ref *r) {
    const uint8_t *ql = x, *qh = x + 128, *sc = x + 192;
    for (int n = 0, y = 0; n < 256; n += 128, y += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            r->code[y + l] = (uint8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4));
            r->code[y + l + 32] = (uint8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4));
            r->code[y + l + 64] = (uint8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4));
            r->code[y + l + 96] = (uint8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4));
            r->scale[y + l] = (uint8_t)(sc[is + 0] ^ 0x80); /* offset-binary, as the fields */
            r->scale[y + l + 32] = (uint8_t)(sc[is + 2] ^ 0x80);
            r->scale[y + l + 64] = (uint8_t)(sc[is + 4] ^ 0x80);
            r->scale[y + l + 96] = (uint8_t)(sc[is + 6] ^ 0x80);
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
    r->d = ref16(x + 208);
}

/* unpack agrees with ggml on what every bit means (codes, which scale / min each weight uses). */
static void test_fields_match_ggml(void) {
    for (int t = 0; t < 4; t++) {
        const ingot_kq_layout *l = ingot_kq_layout_of(KQ_TYPES[t]);
        uint8_t *in = random_bytes((size_t)512 * l->block_bytes);
        int bad = 0;
        for (size_t b = 0; b < 512; b++) {
            const uint8_t *blk = in + b * l->block_bytes;
            ggml_ref r;
            ingot_kq_fields f;
            memset(&r, 0, sizeof(r));
            if (l->ggml_type == INGOT_GGML_Q3_K)
                ref_q3(blk, &r);
            else if (l->ggml_type == INGOT_GGML_Q6_K)
                ref_q6(blk, &r);
            else
                ref_q45(blk, &r, l->ggml_type == INGOT_GGML_Q5_K);
            ingot_kq_unpack(l, blk, &f);
            uint32_t per = INGOT_KQ_WEIGHTS / l->subblocks;
            bad += f.d != r.d || f.dmin != r.dmin;
            for (int i = 0; i < INGOT_KQ_WEIGHTS; i++)
                bad += f.code[i] != r.code[i] || f.scale[i / per] != r.scale[i] ||
                       f.min[i / per] != r.min[i];
        }
        free(in);
        ASSERT_INT_EQ(0, bad);
    }
}

/* Every in-range field value survives pack then unpack (the other direction of the bijection). */
static void test_fields_values_identity(void) {
    for (int t = 0; t < 4; t++) {
        const ingot_kq_layout *l = ingot_kq_layout_of(KQ_TYPES[t]);
        uint8_t blk[256];
        int bad = 0;
        for (int n = 0; n < 2048; n++) {
            ingot_kq_fields f, g;
            memset(&f, 0, sizeof(f));
            f.d = (uint16_t)rng_next();
            f.dmin = l->has_min ? (uint16_t)rng_next() : 0;
            for (uint32_t k = 0; k < l->subblocks; k++) {
                f.scale[k] = (uint8_t)(rng_next() >> (64 - l->scale_bits));
                f.min[k] = l->has_min ? (uint8_t)(rng_next() >> 58) : 0;
            }
            for (int i = 0; i < INGOT_KQ_WEIGHTS; i++)
                f.code[i] = (uint8_t)(rng_next() >> (64 - l->code_bits));
            ingot_kq_pack(l, &f, blk);
            ingot_kq_unpack(l, blk, &g);
            bad += memcmp(&f, &g, sizeof(f)) != 0;
        }
        ASSERT_INT_EQ(0, bad);
    }
}

/* Encode one tensor to memory; returns the payload (prefix + block area). */
static ingot_sink encode_mem(ingot_mode mode, uint32_t type, const uint8_t *data,
                             const uint8_t *base, uint64_t rows, uint64_t cols) {
    ingot_sink s = {.fd = -1};
    ingot_qtensor_plan *plan = NULL;
    if (ingot_qtensor_plan_create(mode, ingot_ggml_dtype(type), data, base, rows, cols, 2,
                                  &plan) == 0 &&
        ingot_qtensor_encode(plan, 2, &s) != 0) {
        free(s.mem);
        s = (ingot_sink){.fd = -1};
    }
    ingot_qtensor_plan_free(plan);
    return s;
}

/* Decode every block into out; 0 or the first error. */
static int decode_mem(ingot_mode mode, uint32_t type, const ingot_sink *s, const uint8_t *base,
                      uint64_t rows, uint64_t cols, uint8_t *out) {
    ingot_qtensor_decoder *d = NULL;
    uint64_t at = 0;
    if (!s->mem)
        return -INGOT_EINVAL;
    int rc = ingot_qtensor_decoder_open(mode, ingot_ggml_dtype(type), rows, cols, s->mem, s->pos,
                                        &d, &at);
    ingot_block_area area;
    if (!rc)
        rc = ingot_block_area_open(s->mem + at, s->pos - at, ingot_qtensor_decoder_blocks(d),
                                   &area);
    for (size_t b = 0; !rc && b < ingot_qtensor_decoder_blocks(d); b++) {
        ingot_block_view v;
        uint64_t lo = 0, hi = 0;
        rc = ingot_block_view_get(&area, b, &v);
        ingot_qtensor_decoder_range(d, b, &lo, &hi);
        if (!rc)
            rc = ingot_qtensor_decoder_block(d, b, &v, base ? base + lo : NULL, out + lo);
    }
    ingot_qtensor_decoder_close(d);
    return rc;
}

/* Round trip; *coded = payload bytes. 1 if exact. */
static int roundtrip_ok(ingot_mode mode, uint32_t type, const uint8_t *data, const uint8_t *base,
                        uint64_t rows, uint64_t cols, uint64_t *coded) {
    const ingot_kq_layout *l = ingot_kq_layout_of(type);
    size_t bytes = (size_t)(rows * cols / INGOT_KQ_WEIGHTS * l->block_bytes);
    ingot_sink s = encode_mem(mode, type, data, base, rows, cols);
    uint8_t *out = calloc(1, bytes);
    int ok = out && decode_mem(mode, type, &s, base, rows, cols, out) == 0 &&
             memcmp(out, data, bytes) == 0;
    if (coded)
        *coded = s.pos;
    free(out);
    free(s.mem);
    return ok;
}

static void test_qtensor_roundtrip_synthetic(void) {
    static const uint64_t SHAPES[3][2] = {{700, 768}, {80, 4096}, {3, 256}};
    for (int t = 0; t < 4; t++) {
        const ingot_kq_layout *l = ingot_kq_layout_of(KQ_TYPES[t]);
        ASSERT_NOT_NULL(ingot_kquant_qtype(KQ_TYPES[t]));
        for (int k = 0; k < 3; k++) {
            uint64_t rows = SHAPES[k][0], cols = SHAPES[k][1];
            uint8_t *data = synth_tensor(l, rows * cols / INGOT_KQ_WEIGHTS);
            ASSERT_TRUE(roundtrip_ok(INGOT_MODE_QUANT, KQ_TYPES[t], data, NULL, rows, cols,
                                     NULL));
            free(data);
        }
    }
    ASSERT_NULL(ingot_kquant_qtype(INGOT_GGML_Q8_0));
    ASSERT_NULL(ingot_kquant_qtype(10)); /* Q2_K */
}

/* Any bytes (NaN / negative d, every scale and code pattern) round trip. */
static void test_qtensor_roundtrip_random_bytes(void) {
    for (int t = 0; t < 4; t++) {
        const ingot_kq_layout *l = ingot_kq_layout_of(KQ_TYPES[t]);
        uint64_t rows = 64, cols = 1024;
        uint8_t *data = random_bytes(rows * cols / INGOT_KQ_WEIGHTS * l->block_bytes);
        ASSERT_TRUE(roundtrip_ok(INGOT_MODE_QUANT, KQ_TYPES[t], data, NULL, rows, cols, NULL));
        free(data);
    }
}

/* Base: half the blocks identical, plus an unrelated base and an identical one. */
static void test_qtensor_roundtrip_with_base(void) {
    for (int t = 0; t < 4; t++) {
        const ingot_kq_layout *l = ingot_kq_layout_of(KQ_TYPES[t]);
        uint64_t rows = 300, cols = 1536, blocks = rows * cols / INGOT_KQ_WEIGHTS;
        uint8_t *data = synth_tensor(l, blocks), *base = synth_tensor(l, blocks);
        uint8_t *same = malloc(blocks * l->block_bytes);
        memcpy(same, data, blocks * l->block_bytes);
        uint64_t coded_unrelated = 0, coded_half = 0, coded_same = 0;
        ASSERT_TRUE(roundtrip_ok(INGOT_MODE_QUANT_DELTA, KQ_TYPES[t], data, base, rows, cols,
                                 &coded_unrelated));
        for (uint64_t b = 0; b < blocks; b++)
            if (rng_next() & 1)
                memcpy(base + b * l->block_bytes, data + b * l->block_bytes, l->block_bytes);
        ASSERT_TRUE(roundtrip_ok(INGOT_MODE_QUANT_DELTA, KQ_TYPES[t], data, base, rows, cols,
                                 &coded_half));
        ASSERT_TRUE(roundtrip_ok(INGOT_MODE_QUANT_DELTA, KQ_TYPES[t], data, same, rows, cols,
                                 &coded_same));
        ASSERT_TRUE(coded_half < coded_unrelated * 6 / 10);
        ASSERT_TRUE(coded_same < blocks * l->block_bytes / 50);
        free(data);
        free(base);
        free(same);
    }
}

static void test_qtensor_size_below_raw(void) {
    for (int t = 0; t < 4; t++) {
        const ingot_kq_layout *l = ingot_kq_layout_of(KQ_TYPES[t]);
        uint64_t rows = 512, cols = 2048, raw = rows * cols / INGOT_KQ_WEIGHTS * l->block_bytes;
        uint8_t *data = synth_tensor(l, rows * cols / INGOT_KQ_WEIGHTS);
        uint64_t coded = 0;
        ASSERT_TRUE(roundtrip_ok(INGOT_MODE_QUANT, KQ_TYPES[t], data, NULL, rows, cols, &coded));
        printf("    %s: %.3f -> %.3f bits/weight\n", ingot_ggml_name(KQ_TYPES[t]),
               8.0 * l->block_bytes / INGOT_KQ_WEIGHTS, 8.0 * coded / (rows * cols));
        ASSERT_TRUE(coded < raw * 97 / 100);
        free(data);
    }
}

/* Flip bytes inside a coded block (raw stream, rANS words, lane states): each must be caught. */
static void test_qtensor_corrupt_block_fails(void) {
    for (int t = 0; t < 4; t++) {
        const ingot_kq_layout *l = ingot_kq_layout_of(KQ_TYPES[t]);
        uint64_t rows = 160, cols = 2048, bytes = rows * cols / INGOT_KQ_WEIGHTS * l->block_bytes;
        uint8_t *data = synth_tensor(l, rows * cols / INGOT_KQ_WEIGHTS);
        uint8_t *out = malloc(bytes);
        ingot_sink s = encode_mem(INGOT_MODE_QUANT, KQ_TYPES[t], data, NULL, rows, cols);
        ASSERT_NOT_NULL(s.mem);
        ASSERT_INT_EQ(0, decode_mem(INGOT_MODE_QUANT, KQ_TYPES[t], &s, NULL, rows, cols, out));
        int missed = 0;
        for (int k = 1; k <= 24; k++) {
            uint64_t at = s.pos - s.pos * (uint64_t)k / 50; /* spread over the block area */
            s.mem[at] ^= (uint8_t)(1u << (k % 8));
            int rc = decode_mem(INGOT_MODE_QUANT, KQ_TYPES[t], &s, NULL, rows, cols, out);
            missed += rc == 0 && memcmp(out, data, bytes) == 0;
            s.mem[at] ^= (uint8_t)(1u << (k % 8));
        }
        ASSERT_INT_EQ(0, missed);
        ASSERT_INT_EQ(0, decode_mem(INGOT_MODE_QUANT, KQ_TYPES[t], &s, NULL, rows, cols, out));
        ASSERT_TRUE(memcmp(out, data, bytes) == 0);
        free(s.mem);
        free(out);
        free(data);
    }
}

int main(void) {
    RUN_TEST(fields_bytes_identity);
    RUN_TEST(fields_values_identity);
    RUN_TEST(fields_match_ggml);
    RUN_TEST(qtensor_roundtrip_synthetic);
    RUN_TEST(qtensor_roundtrip_random_bytes);
    RUN_TEST(qtensor_roundtrip_with_base);
    RUN_TEST(qtensor_size_below_raw);
    RUN_TEST(qtensor_corrupt_block_fails);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
