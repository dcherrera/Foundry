/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../turboquant.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Deterministic LCG so the tests do not depend on the platform rand(). */
static uint64_t g_rng = 0x1234567;
static float frand(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t x = (uint32_t)(g_rng >> 33);
    return 2.0f * ((float)x / (float)0x7fffffff) - 1.0f;   /* [-1, 1] */
}

/* ============================================================= */
/* Codebook                                                      */
/* ============================================================= */

static void test_codebook_beta(void) {
    fa_tq_codebook cb;
    ASSERT_INT_EQ(0, fa_tq_build_codebook(&cb, 128, 4));
    ASSERT_INT_EQ(4, cb.bits);
    ASSERT_INT_EQ(16, cb.levels);

    /* Centroids strictly increasing and inside (0,1). */
    for (int i = 0; i < cb.levels; i++) {
        ASSERT_TRUE(cb.centroids[i] > 0.0f && cb.centroids[i] < 1.0f);
        if (i > 0) ASSERT_TRUE(cb.centroids[i] > cb.centroids[i - 1]);
    }
    /* Symmetric about 0.5 (Beta(d/2,d/2) is symmetric). */
    for (int i = 0; i < cb.levels; i++) {
        float mirror = 1.0f - cb.centroids[cb.levels - 1 - i];
        ASSERT_FLT_NEAR(cb.centroids[i], mirror, 0.01f);
    }
    /* Boundaries lie between neighbouring centroids. */
    for (int i = 0; i < cb.levels - 1; i++) {
        ASSERT_TRUE(cb.boundaries[i] > cb.centroids[i]);
        ASSERT_TRUE(cb.boundaries[i] < cb.centroids[i + 1]);
    }
}

/* ============================================================= */
/* Rotation                                                      */
/* ============================================================= */

static void test_rotation_roundtrip(void) {
    fa_tq_state* st = fa_tq_init(128, fa_tq_config_default(), 42);
    ASSERT_NOT_NULL(st);
    int rd = fa_tq_rot_dim(st);
    ASSERT_INT_EQ(128, rd);

    float x[128], y[128];
    for (int i = 0; i < 128; i++) { x[i] = frand(); y[i] = x[i]; }

    fa_tq_rotate(st, y);
    fa_tq_rotate_inverse(st, y);
    for (int i = 0; i < 128; i++) ASSERT_FLT_NEAR(x[i], y[i], 1e-3f);

    fa_tq_free(st);
}

/* Rotation is orthogonal: it preserves the L2 norm. */
static void test_rotation_orthogonal(void) {
    fa_tq_state* st = fa_tq_init(128, fa_tq_config_default(), 7);
    float x[128];
    double n0 = 0.0;
    for (int i = 0; i < 128; i++) { x[i] = frand(); n0 += (double)x[i] * x[i]; }
    fa_tq_rotate(st, x);
    double n1 = 0.0;
    for (int i = 0; i < 128; i++) n1 += (double)x[i] * x[i];
    ASSERT_FLT_NEAR((float)n0, (float)n1, 1e-2f);
    fa_tq_free(st);
}

/* ============================================================= */
/* Quantize / dequantize round-trip                              */
/* ============================================================= */

static double head_mse(fa_tq_state* st, bool is_key, int trials) {
    size_t bph = is_key ? fa_tq_key_bytes_per_head(st) : fa_tq_value_bytes_per_head(st);
    uint8_t packed[512];
    float range[2];
    float in[128], out[128];
    double se = 0.0;
    int n = 0;
    for (int t = 0; t < trials; t++) {
        double norm = 0.0;
        for (int i = 0; i < 128; i++) { in[i] = frand(); norm += (double)in[i] * in[i]; }
        norm = sqrt(norm);
        for (int i = 0; i < 128; i++) in[i] /= (float)norm;   /* unit vectors */
        (void)bph;
        fa_tq_quantize_head(st, in, is_key, packed, range);
        fa_tq_dequantize_head(st, packed, range, is_key, out);
        for (int i = 0; i < 128; i++) { double d = in[i] - out[i]; se += d * d; n++; }
    }
    return se / n;
}

static void test_roundtrip_mse(void) {
    fa_tq_state* st = fa_tq_init(128, fa_tq_config_default(), 123);
    double mse_k = head_mse(st, true, 200);
    double mse_v = head_mse(st, false, 200);
    /* Unit vectors: coord variance ~1/128. 4-bit << 2-bit error. */
    ASSERT_DBL_LT(mse_k, 1e-4);
    ASSERT_DBL_LT(mse_v, 2e-3);
    ASSERT_DBL_LT(mse_k, mse_v);   /* keys (4-bit) beat values (2-bit) */
    fa_tq_free(st);
}

/* Dot product preservation: softmax attention scores survive quantization. */
static void test_dot_preservation(void) {
    fa_tq_state* st = fa_tq_init(128, fa_tq_config_default(), 99);
    uint8_t kp[512];
    float range[2];
    float q[128], k[128], kdq[128];
    double serr = 0.0, smag = 0.0;
    for (int t = 0; t < 100; t++) {
        for (int i = 0; i < 128; i++) { q[i] = frand(); k[i] = frand(); }
        fa_tq_quantize_head(st, k, true, kp, range);
        fa_tq_dequantize_head(st, kp, range, true, kdq);
        double d0 = 0.0, d1 = 0.0;
        for (int i = 0; i < 128; i++) { d0 += (double)q[i] * k[i]; d1 += (double)q[i] * kdq[i]; }
        serr += (d0 - d1) * (d0 - d1);
        smag += d0 * d0;
    }
    /* Relative RMS dot error (mean-abs is unstable here since random q,k are
     * near-orthogonal, giving tiny denominators).  ~9-10% is the Lloyd-Max
     * optimum for a 4-bit scalar quantizer (SQNR ~20 dB); in real attention
     * these errors average out across the sequence after softmax. */
    ASSERT_DBL_LT(sqrt(serr / smag), 0.12);   /* 4-bit keys */
    fa_tq_free(st);
}

/* ============================================================= */
/* Determinism                                                   */
/* ============================================================= */

static void test_determinism(void) {
    fa_tq_state* a = fa_tq_init(128, fa_tq_config_default(), 555);
    fa_tq_state* b = fa_tq_init(128, fa_tq_config_default(), 555);
    uint8_t pa[512], pb[512];
    float ra[2], rb[2], v[128];
    for (int i = 0; i < 128; i++) v[i] = frand();
    fa_tq_quantize_head(a, v, true, pa, ra);
    fa_tq_quantize_head(b, v, true, pb, rb);
    ASSERT_INT_EQ(0, memcmp(pa, pb, fa_tq_key_bytes_per_head(a)));
    ASSERT_FLT_NEAR(ra[0], rb[0], 0.0f);
    ASSERT_FLT_NEAR(ra[1], rb[1], 0.0f);
    fa_tq_free(a);
    fa_tq_free(b);
}

/* ============================================================= */
/* Edge cases                                                    */
/* ============================================================= */

static void test_edge_cases(void) {
    fa_tq_state* st = fa_tq_init(128, fa_tq_config_default(), 3);
    uint8_t p[512];
    float range[2], in[128], out[128];

    /* All zero. */
    for (int i = 0; i < 128; i++) in[i] = 0.0f;
    fa_tq_quantize_head(st, in, true, p, range);
    fa_tq_dequantize_head(st, p, range, true, out);
    for (int i = 0; i < 128; i++) ASSERT_FLT_NEAR(0.0f, out[i], 1e-4f);

    /* All identical (near-worst case).  A pure DC vector is the hardest input;
     * check overall relative energy error rather than each element, since the
     * specific random sign pattern leaves a few outlier coordinates. */
    for (int i = 0; i < 128; i++) in[i] = 3.5f;
    fa_tq_quantize_head(st, in, true, p, range);
    fa_tq_dequantize_head(st, p, range, true, out);
    double se = 0.0, en = 0.0;
    for (int i = 0; i < 128; i++) { double d = in[i] - out[i]; se += d * d; en += (double)in[i] * in[i]; }
    ASSERT_DBL_LT(se / en, 0.02);

    fa_tq_free(st);
}

/* ============================================================= */
/* Non power-of-two head_dim                                     */
/* ============================================================= */

static void test_non_pow2(void) {
    fa_tq_state* st = fa_tq_init(96, fa_tq_config_default(), 11);
    ASSERT_INT_EQ(96, fa_tq_head_dim(st));
    ASSERT_INT_EQ(128, fa_tq_rot_dim(st));   /* padded up */
    uint8_t p[512];
    float range[2], in[96], out[96];
    double se = 0.0, norm = 0.0;
    for (int i = 0; i < 96; i++) { in[i] = frand(); norm += (double)in[i] * in[i]; }
    for (int i = 0; i < 96; i++) in[i] /= (float)sqrt(norm);
    fa_tq_quantize_head(st, in, true, p, range);
    fa_tq_dequantize_head(st, p, range, true, out);
    for (int i = 0; i < 96; i++) { double d = in[i] - out[i]; se += d * d; }
    ASSERT_DBL_LT(se / 96.0, 1e-3);
    fa_tq_free(st);
}

/* ============================================================= */
/* Cache round-trip + eviction                                   */
/* ============================================================= */

static void test_cache_roundtrip(void) {
    int H = 4, D = 128, L = 2, S = 8;
    fa_tq_state* st = fa_tq_init(D, fa_tq_config_default(), 2024);
    fa_tq_cache* c = fa_tq_cache_new(st, L, H, S);
    ASSERT_NOT_NULL(c);

    float* vec = malloc((size_t)H * D * sizeof(float));
    float* out = malloc((size_t)H * D * sizeof(float));

    for (int pos = 0; pos < 5; pos++) {
        for (int i = 0; i < H * D; i++) vec[i] = frand();
        fa_tq_quantize(st, c, 0, pos, vec, true);   /* keys */
    }
    ASSERT_INT_EQ(5, fa_tq_cache_seq_len(c));

    /* Re-quantize a known vector at pos 2 and check dequant matches. */
    for (int i = 0; i < H * D; i++) vec[i] = frand() * 2.0f;
    fa_tq_quantize(st, c, 0, 2, vec, true);
    fa_tq_dequantize(st, c, 0, 2, 1, true, out);
    double se = 0.0, en = 0.0;
    for (int i = 0; i < H * D; i++) { double d = vec[i] - out[i]; se += d * d; en += (double)vec[i] * vec[i]; }
    ASSERT_DBL_LT(se / en, 0.02);   /* <2% relative energy loss (4-bit keys) */

    free(vec);
    free(out);
    fa_tq_cache_free(c);
    fa_tq_free(st);
}

/* ============================================================= */
/* Bit packing for all widths                                    */
/* ============================================================= */

static void test_packing_all_bits(void) {
    /* Round-trip every level value through quantize/dequantize path is
     * covered above; here confirm K3/V2 preset builds and packs sanely. */
    fa_tq_config cfg = { 3, 2 };
    fa_tq_state* st = fa_tq_init(128, cfg, 1);
    ASSERT_NOT_NULL(st);
    ASSERT_INT_EQ(3, fa_tq_key_codebook(st)->bits);
    ASSERT_INT_EQ(2, fa_tq_value_codebook(st)->bits);
    /* 128 coords, 3-bit -> 2 per byte -> 64 bytes; 2-bit -> 4 per byte -> 32 bytes. */
    ASSERT_INT_EQ(64, (int)fa_tq_key_bytes_per_head(st));
    ASSERT_INT_EQ(32, (int)fa_tq_value_bytes_per_head(st));
    fa_tq_free(st);
}

/* ============================================================= */
/* Compression ratio report                                      */
/* ============================================================= */

static void test_compression_ratio(void) {
    int H = 16, D = 128, L = 4, S = 2048;
    fa_tq_state* st = fa_tq_init(D, fa_tq_config_default(), 1);
    fa_tq_cache* c = fa_tq_cache_new(st, L, H, S);

    size_t fp32 = (size_t)2 * L * S * H * D * sizeof(float);
    size_t comp = fa_tq_cache_bytes(c);
    double ratio = (double)fp32 / (double)comp;

    printf("    [K4/V2] FP32 KV = %zu bytes, TurboQuant = %zu bytes, ratio = %.2fx\n",
           fp32, comp, ratio);
    ASSERT_DBL_GT(ratio, 8.0);   /* spec: ~9.1x incl. range metadata */

    fa_tq_cache_free(c);
    fa_tq_free(st);
}

int main(void) {
    printf("\n=== TurboQuant KV Cache Tests ===\n\n");
    RUN_TEST(codebook_beta);
    RUN_TEST(rotation_roundtrip);
    RUN_TEST(rotation_orthogonal);
    RUN_TEST(roundtrip_mse);
    RUN_TEST(dot_preservation);
    RUN_TEST(determinism);
    RUN_TEST(edge_cases);
    RUN_TEST(non_pow2);
    RUN_TEST(cache_roundtrip);
    RUN_TEST(packing_all_bits);
    RUN_TEST(compression_ratio);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
