/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../hdc.h"
#include "../../tensor/tensor.h"

#include <stdlib.h>
#include <math.h>
#include <unistd.h>

/* ============================================================= */
/* Vector Lifecycle Tests                                        */
/* ============================================================= */

static void test_alloc_free(void) {
    fv_vector *v = fv_alloc(4096);
    ASSERT_NOT_NULL(v);
    ASSERT_INT_EQ(4096, v->dim);
    ASSERT_NOT_NULL(v->bits);
    fv_free(v);

    /* NULL-safe free */
    fv_free(NULL);

    /* Invalid dim */
    fv_vector *bad = fv_alloc(0);
    ASSERT_NULL(bad);
}

static void test_random_near_half(void) {
    fv_vector *v = fv_random(10000, 12345);
    ASSERT_NOT_NULL(v);
    int ones = fv_popcount(v);
    /* Should be roughly 50% set bits (within 5%) */
    ASSERT_TRUE(ones > 4500);
    ASSERT_TRUE(ones < 5500);
    fv_free(v);
}

/* ============================================================= */
/* Core HDC Operations Tests                                     */
/* ============================================================= */

static void test_xor_bind_unbind(void) {
    fv_vector *a = fv_random(4096, 100);
    fv_vector *b = fv_random(4096, 200);
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);

    /* XOR(XOR(a,b), b) should recover a */
    fv_vector *ab = fv_xor(a, b);
    ASSERT_NOT_NULL(ab);
    fv_vector *recovered = fv_xor(ab, b);
    ASSERT_NOT_NULL(recovered);

    ASSERT_INT_EQ(0, fv_hamming(a, recovered));

    fv_free(a);
    fv_free(b);
    fv_free(ab);
    fv_free(recovered);
}

static void test_xor_self_is_zero(void) {
    fv_vector *a = fv_random(4096, 300);
    ASSERT_NOT_NULL(a);

    fv_vector *z = fv_xor(a, a);
    ASSERT_NOT_NULL(z);
    ASSERT_INT_EQ(0, fv_popcount(z));

    fv_free(a);
    fv_free(z);
}

static void test_permute_reversible(void) {
    fv_vector *v = fv_random(4096, 400);
    ASSERT_NOT_NULL(v);

    fv_vector *shifted = fv_permute(v, 17);
    ASSERT_NOT_NULL(shifted);
    fv_vector *restored = fv_permute(shifted, -17);
    ASSERT_NOT_NULL(restored);

    ASSERT_INT_EQ(0, fv_hamming(v, restored));

    fv_free(v);
    fv_free(shifted);
    fv_free(restored);
}

static void test_bundle_majority(void) {
    /* Create 5 vectors: 3 similar to 'target', 2 random */
    fv_vector *target = fv_random(4096, 500);
    ASSERT_NOT_NULL(target);

    fv_vector *vecs[5];
    /* First 3: copies of target (high agreement) */
    vecs[0] = fv_clone(target);
    vecs[1] = fv_clone(target);
    vecs[2] = fv_clone(target);
    /* Last 2: random (low agreement) */
    vecs[3] = fv_random(4096, 600);
    vecs[4] = fv_random(4096, 700);

    fv_vector *bundled = fv_bundle(vecs, 5);
    ASSERT_NOT_NULL(bundled);

    /* Bundled should be very similar to target (majority dominates) */
    float sim = fv_similarity(target, bundled);
    ASSERT_TRUE(sim > 0.8f);

    fv_free(target);
    for (int i = 0; i < 5; i++) fv_free(vecs[i]);
    fv_free(bundled);
}

static void test_bundle_weighted(void) {
    fv_vector *a = fv_random(4096, 800);
    fv_vector *b = fv_random(4096, 900);
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);

    /* Heavy weight on 'a' */
    fv_vector *vecs[2] = { a, b };
    float weights[2] = { 10.0f, 1.0f };
    fv_vector *bundled = fv_bundle_weighted(vecs, weights, 2);
    ASSERT_NOT_NULL(bundled);

    float sim_a = fv_similarity(a, bundled);
    float sim_b = fv_similarity(b, bundled);
    ASSERT_TRUE(sim_a > sim_b);

    fv_free(a);
    fv_free(b);
    fv_free(bundled);
}

/* ============================================================= */
/* Distance Tests                                                */
/* ============================================================= */

static void test_hamming_identical(void) {
    fv_vector *a = fv_random(4096, 1000);
    ASSERT_NOT_NULL(a);
    ASSERT_INT_EQ(0, fv_hamming(a, a));
    fv_free(a);
}

static void test_hamming_opposite(void) {
    fv_vector *a = fv_random(4096, 1100);
    ASSERT_NOT_NULL(a);

    /* Create bitwise complement */
    fv_vector *b = fv_alloc(4096);
    ASSERT_NOT_NULL(b);
    int words = (4096 + 63) / 64;
    for (int i = 0; i < words; i++) {
        b->bits[i] = ~a->bits[i];
    }
    /* Mask unused bits in last word (4096 % 64 == 0, so no masking needed) */

    ASSERT_INT_EQ(4096, fv_hamming(a, b));

    fv_free(a);
    fv_free(b);
}

static void test_hamming_random(void) {
    fv_vector *a = fv_random(10000, 1200);
    fv_vector *b = fv_random(10000, 1300);
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);

    int dist = fv_hamming(a, b);
    /* Two random vectors: hamming ≈ dim/2 (within 10%) */
    ASSERT_TRUE(dist > 4000);
    ASSERT_TRUE(dist < 6000);

    fv_free(a);
    fv_free(b);
}

static void test_similarity_range(void) {
    fv_vector *a = fv_random(4096, 1400);
    fv_vector *b = fv_random(4096, 1500);
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);

    float sim = fv_similarity(a, b);
    ASSERT_TRUE(sim >= 0.0f);
    ASSERT_TRUE(sim <= 1.0f);

    /* Self-similarity = 1.0 */
    ASSERT_FLT_EQ(1.0f, fv_similarity(a, a));

    fv_free(a);
    fv_free(b);
}

/* ============================================================= */
/* Projector Tests                                               */
/* ============================================================= */

static void test_projector_create_free(void) {
    fv_projector *p = fv_projector_new(128, 4096, 42);
    ASSERT_NOT_NULL(p);
    fv_projector_free(p);

    /* NULL-safe free */
    fv_projector_free(NULL);

    /* Invalid dims */
    ASSERT_NULL(fv_projector_new(0, 4096, 42));
    ASSERT_NULL(fv_projector_new(128, 0, 42));
}

static void test_project_deterministic(void) {
    int input_dim = 64;
    int hdc_dim = 2048;

    fv_projector *p1 = fv_projector_new(input_dim, hdc_dim, 42);
    fv_projector *p2 = fv_projector_new(input_dim, hdc_dim, 42);
    ASSERT_NOT_NULL(p1);
    ASSERT_NOT_NULL(p2);

    /* Create a test embedding */
    int64_t shape[1] = { input_dim };
    tm_tensor *emb = tm_ones(shape, 1, TM_F32);
    ASSERT_NOT_NULL(emb);

    fv_vector *v1 = fv_project(p1, emb);
    fv_vector *v2 = fv_project(p2, emb);
    ASSERT_NOT_NULL(v1);
    ASSERT_NOT_NULL(v2);

    /* Same seed + same input = identical output */
    ASSERT_INT_EQ(0, fv_hamming(v1, v2));

    fv_free(v1);
    fv_free(v2);
    tm_free(emb);
    fv_projector_free(p1);
    fv_projector_free(p2);
}

static void test_project_preserves_sim(void) {
    int input_dim = 64;
    int hdc_dim = 4096;

    fv_projector *p = fv_projector_new(input_dim, hdc_dim, 42);
    ASSERT_NOT_NULL(p);

    /* Create two similar embeddings and one different */
    int64_t shape[1] = { input_dim };
    tm_tensor *a = tm_ones(shape, 1, TM_F32);
    tm_tensor *b = tm_ones(shape, 1, TM_F32);
    tm_tensor *c = tm_zeros(shape, 1, TM_F32);
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(c);

    /* Make 'b' slightly different from 'a' */
    float *b_data = (float *)b->data;
    for (int i = 0; i < 5; i++) b_data[i] = 0.9f;

    /* Make 'c' very different */
    float *c_data = (float *)c->data;
    for (int i = 0; i < input_dim; i++) c_data[i] = (i % 2 == 0) ? -1.0f : 1.0f;

    fv_vector *va = fv_project(p, a);
    fv_vector *vb = fv_project(p, b);
    fv_vector *vc = fv_project(p, c);
    ASSERT_NOT_NULL(va);
    ASSERT_NOT_NULL(vb);
    ASSERT_NOT_NULL(vc);

    float sim_ab = fv_similarity(va, vb);
    float sim_ac = fv_similarity(va, vc);

    /* Similar inputs should produce more similar HVs */
    ASSERT_TRUE(sim_ab > sim_ac);

    fv_free(va);
    fv_free(vb);
    fv_free(vc);
    tm_free(a);
    tm_free(b);
    tm_free(c);
    fv_projector_free(p);
}

/* ============================================================= */
/* Tensor Conversion Tests                                       */
/* ============================================================= */

static void test_to_tensor_roundtrip(void) {
    fv_vector *v = fv_random(256, 1600);
    ASSERT_NOT_NULL(v);

    tm_tensor *t = fv_to_tensor(v);
    ASSERT_NOT_NULL(t);
    ASSERT_INT_EQ(1, t->ndim);
    ASSERT_INT_EQ(256, t->shape[0]);

    fv_vector *v2 = fv_from_tensor(t);
    ASSERT_NOT_NULL(v2);

    /* Roundtrip should preserve all bits */
    ASSERT_INT_EQ(0, fv_hamming(v, v2));

    fv_free(v);
    fv_free(v2);
    tm_free(t);
}

/* ============================================================= */
/* Search Tests                                                  */
/* ============================================================= */

static void test_search_finds_exact(void) {
    fv_vector *target = fv_random(4096, 1700);
    ASSERT_NOT_NULL(target);

    /* Corpus: target + 9 random vectors */
    int n = 10;
    fv_vector *corpus[10];
    corpus[5] = fv_clone(target); /* put target at index 5 */
    for (int i = 0; i < n; i++) {
        if (i != 5) corpus[i] = fv_random(4096, 1800 + i);
    }

    int indices[3];
    float scores[3];
    int found = fv_search(corpus, n, target, 3, indices, scores);
    ASSERT_INT_EQ(3, found);

    /* Exact match should be rank 1 */
    ASSERT_INT_EQ(5, indices[0]);
    ASSERT_FLT_EQ(1.0f, scores[0]);

    fv_free(target);
    for (int i = 0; i < n; i++) fv_free(corpus[i]);
}

static void test_search_top_k_order(void) {
    fv_vector *query = fv_random(4096, 1900);
    ASSERT_NOT_NULL(query);

    int n = 20;
    fv_vector *corpus[20];
    for (int i = 0; i < n; i++) {
        corpus[i] = fv_random(4096, 2000 + i);
    }

    int k = 5;
    int indices[5];
    float scores[5];
    int found = fv_search(corpus, n, query, k, indices, scores);
    ASSERT_INT_EQ(k, found);

    /* Results should be in decreasing score order */
    for (int i = 1; i < k; i++) {
        ASSERT_TRUE(scores[i - 1] >= scores[i]);
    }

    fv_free(query);
    for (int i = 0; i < n; i++) fv_free(corpus[i]);
}

/* ============================================================= */
/* Serialization Tests                                           */
/* ============================================================= */

static void test_save_load_roundtrip(void) {
    fv_vector *v = fv_random(4096, 2100);
    ASSERT_NOT_NULL(v);

    const char *path = "/tmp/fv_test_vector.bin";
    int ret = fv_save(v, path);
    ASSERT_INT_EQ(0, ret);

    fv_vector *loaded = fv_load(path);
    ASSERT_NOT_NULL(loaded);
    ASSERT_INT_EQ(v->dim, loaded->dim);
    ASSERT_INT_EQ(0, fv_hamming(v, loaded));

    fv_free(v);
    fv_free(loaded);
    unlink(path);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  HDC Module Tests\n\n");

    RUN_TEST(alloc_free);
    RUN_TEST(random_near_half);
    RUN_TEST(xor_bind_unbind);
    RUN_TEST(xor_self_is_zero);
    RUN_TEST(permute_reversible);
    RUN_TEST(bundle_majority);
    RUN_TEST(bundle_weighted);
    RUN_TEST(hamming_identical);
    RUN_TEST(hamming_opposite);
    RUN_TEST(hamming_random);
    RUN_TEST(similarity_range);
    RUN_TEST(projector_create_free);
    RUN_TEST(project_deterministic);
    RUN_TEST(project_preserves_sim);
    RUN_TEST(to_tensor_roundtrip);
    RUN_TEST(search_finds_exact);
    RUN_TEST(search_top_k_order);
    RUN_TEST(save_load_roundtrip);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
