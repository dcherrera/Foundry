/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_cuda_reference.c
 * @brief  GPU decode, CPU side (runs anywhere): the shared device math against the codec
 *         (partner term over every BF16 value and shift, buckets at every threshold edge, table
 *         lookups over every slot), and the device image decoded by the CPU reference of the
 *         kernels against the codec's decoder on every case (bit-identical, status clean), the
 *         matvec close to a double dot and identical across runs, a corrupt block flagged.
 *
 * Red: build with -DINGOT_CUDA_RED_BUCKET (bucket off by one at equality) and the bucket test
 * must fail.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lib/test.h"
#include "ingot/codec/ingot_predict.h"
#include "ingot/core/ingot_rans.h"
#include "ingot/cuda/ingot_cuda_image.h"
#include "ingot/cuda/ingot_cuda_reference.h"
#include "ingot/tests/ingot_cuda_cases.h"

/* ---- Shared math ---------------------------------------------------------------------------- */

static void test_partner_term_matches_codec(void) {
    static const int32_t q[] = {-32767, -1234, -1, 1, 7, 255, 32767};
    long bad = 0;
    for (int32_t shift = -200; shift <= 200 && !bad; shift++)
        for (uint32_t x = 0; x < 65536; x++)
            for (size_t i = 0; i < sizeof(q) / sizeof(q[0]); i++) {
                int64_t want = (int64_t)q[i] * ingot_predict_fixed((uint16_t)x, shift);
                if (ingot_cuda_partner_term(x, shift, q[i]) != want && !bad++)
                    printf("    x 0x%04x shift %d q %d\n", x, shift, q[i]);
            }
    static const int32_t far[] = {-2048, -1024, -500, 500, 1024, 2048};
    for (size_t f = 0; f < sizeof(far) / sizeof(far[0]); f++)
        for (uint32_t x = 0; x < 65536; x++)
            bad += ingot_cuda_partner_term(x, far[f], -3) !=
                   -3 * ingot_predict_fixed((uint16_t)x, far[f]);
    ASSERT_INT_EQ(0, bad);
}

/* Count of thresholds at or below mag: the codec's definition of the bucket. */
static uint32_t codec_bucket(uint64_t mag, const int64_t *t) {
    uint32_t k = 0;
    for (int j = 0; j < INGOT_PREDICT_BUCKETS - 1; j++)
        k += (int64_t)mag >= t[j];
    return k;
}

static void test_bucket_matches_thresholds(void) {
    uint8_t cells[INGOT_CUDA_BUCKET_SETS * INGOT_CUDA_BUCKET_CELLS];
    uint32_t bases[INGOT_CUDA_BUCKET_SETS * 12];
    ingot_cuda_bucket_tables(cells, bases, 0, 1);
    long bad = 0, checked = 0;
    for (int32_t c = -120; c <= 120; c++)
        for (int32_t s = -128; s <= 127; s++) {
            int64_t t[INGOT_PREDICT_BUCKETS - 1];
            ingot_predict_thresholds((int8_t)s, c, t);
            int32_t sh = ingot_cuda_unit_shift(s, c);
            uint32_t rsel = ingot_cuda_unit_rsel(s);
            uint64_t probes[3 * 11 + 6] = {0, 1, 2, 3, (uint64_t)1 << 57, ((uint64_t)1 << 58) - 1};
            int n = 6;
            for (int k = 0; k < 11; k++)
                for (int d = -1; d <= 1; d++)
                    if (t[k] + d >= 0 && t[k] + d < ((int64_t)1 << 58))
                        probes[n++] = (uint64_t)(t[k] + d);
            for (int i = 0; i < n; i++, checked++)
                if (ingot_cuda_bucket(probes[i], sh, rsel, cells, bases) !=
                        codec_bucket(probes[i], t) && !bad++)
                    printf("    c %d s %d mag %llu\n", c, s, (unsigned long long)probes[i]);
        }
    for (int i = 0; i < 2000000; i++, checked++) { /* random magnitudes, random units */
        int32_t s = (int32_t)(test_next() % 256) - 128, c = (int32_t)(test_next() % 401) - 200;
        uint64_t mag = test_next() >> (6 + test_next() % 58);
        int64_t t[INGOT_PREDICT_BUCKETS - 1];
        ingot_predict_thresholds((int8_t)s, c, t);
        bad += ingot_cuda_bucket(mag, ingot_cuda_unit_shift(s, c), ingot_cuda_unit_rsel(s), cells,
                                 bases) != codec_bucket(mag, t);
    }
    printf("    %ld probes\n", checked);
    ASSERT_INT_EQ(0, bad);
}

/* A table with `used` symbols spread over the alphabet, frequencies from 1 up. */
static void random_table(ingot_rans_table *t, uint32_t alphabet, uint32_t used) {
    uint64_t counts[INGOT_RANS_MAX_ALPHABET] = {0};
    for (uint32_t i = 0; i < used; i++) {
        uint32_t s = used == alphabet ? i : (uint32_t)(test_next() % alphabet);
        counts[s] += 1 + (test_next() % 4 == 0 ? test_next() % 100000 : test_next() % 3);
    }
    ingot_rans_table_build(t, counts, alphabet);
}

static void test_table_lookup_matches_slots(void) {
    static const uint32_t shapes[][2] = {{2, 1}, {2, 2}, {1024, 1}, {1024, 3}, {1024, 150},
                                         {1024, 1024}, {119, 60}, {256, 256}};
    static uint16_t slots[INGOT_RANS_TOTAL];
    long bad = 0;
    for (size_t sh = 0; sh < sizeof(shapes) / sizeof(shapes[0]); sh++)
        for (int rep = 0; rep < 4; rep++) {
            static ingot_rans_table t;
            random_table(&t, shapes[sh][0], shapes[sh][1]);
            uint8_t *img = malloc(ingot_cuda_table_image_bytes(&t));
            ingot_cuda_table_image_write(&t, img);
            ingot_rans_slots_build(&t, slots);
            for (uint32_t slot = 0; slot < INGOT_RANS_TOTAL; slot++) {
                uint32_t start = 0, freq = 0, s = ingot_cuda_table_find(img, slot, &start, &freq);
                bad += s != slots[slot] || start != t.start[s] || freq != t.freq[s];
            }
            free(img);
        }
    ASSERT_INT_EQ(0, bad);
}

/* ---- Image decoded by the CPU reference ----------------------------------------------------- */

static void test_reference_decodes_every_case(void) {
    for (int i = 0; i < CASE_COUNT; i++) {
        cuda_case k;
        case_build(&k, i);
        uint64_t n = k.e.source_size / 2;
        uint8_t *codec = malloc(n * 2);
        ASSERT_INT_EQ(0, case_codec_decode(&k, codec));
        ASSERT_INT_EQ(0, memcmp(codec, k.data, n * 2));
        ref_result r = case_reference_run(&k, INGOT_CUDA_STORE, NULL);
        int same = !r.rc && !memcmp(r.out, codec, n * 2);
        printf("    %-16s %llu bytes  rc %d  bad blocks %u  %s\n", k.name,
               (unsigned long long)k.size, r.rc, r.bad_blocks, same ? "identical" : "DIFFERENT");
        ASSERT_INT_EQ(0, r.rc);
        ASSERT_INT_EQ(0, (int)r.bad_blocks);
        ASSERT_TRUE(same);
        ref_free(&r);
        free(codec);
        case_free(&k);
    }
}

static void test_reference_matvec_close_and_repeatable(void) {
    for (int i = 0; i < CASE_COUNT; i++) {
        cuda_case k;
        case_build(&k, i);
        uint64_t rows = 0, cols = 0;
        ingot_entry_rows_cols(&k.e, &rows, &cols);
        uint32_t g = ingot_cuda_step_weights(k.e.mode == INGOT_MODE_PLAIN   ? INGOT_CUDA_KIND_PLAIN
                                             : k.e.mode == INGOT_MODE_DELTA ? INGOT_CUDA_KIND_DELTA
                                                                            : INGOT_CUDA_KIND_MATCH);
        if (cols % g) {
            case_free(&k);
            continue;
        }
        float *x = malloc(cols * sizeof(float));
        for (uint64_t c = 0; c < cols; c++)
            x[c] = (float)test_gauss();
        ref_result a = case_reference_run(&k, INGOT_CUDA_MATVEC, x);
        ref_result b = case_reference_run(&k, INGOT_CUDA_MATVEC, x);
        double worst = 0;
        for (uint64_t r = 0; r < rows; r++) {
            double want = 0, mag = 0;
            for (uint64_t c = 0; c < cols; c++) {
                want += case_val(k.data[r * cols + c]) * x[c];
                mag += fabs(case_val(k.data[r * cols + c]) * x[c]);
            }
            double err = fabs(a.y[r] - want) / (mag + 1e-30);
            worst = err > worst ? err : worst;
        }
        printf("    %-16s worst rel err %.2e\n", k.name, worst);
        ASSERT_INT_EQ(0, a.rc);
        ASSERT_INT_EQ(0, (int)a.bad_blocks);
        ASSERT_TRUE(worst < 1e-5);
        ASSERT_INT_EQ(0, memcmp(a.y, b.y, rows * sizeof(float)));
        ref_free(&a);
        ref_free(&b);
        free(x);
        case_free(&k);
    }
}

/* A flipped byte in a block's word stream: the block is flagged or the weights differ. */
static void test_corrupt_block_is_flagged(void) {
    cuda_case k;
    case_build(&k, 6); /* match rows with a self source */
    uint64_t prefix = ingot_load_u64(k.payload);
    uint64_t blocks = ingot_load_u64(k.payload + 8);
    uint64_t first = prefix + 8 * (blocks + 1) + ingot_load_u64(k.payload + prefix);
    uint32_t raw = ingot_load_u32(k.payload + first);
    k.payload[first + 8 + 128 + ((raw + 1) & ~1u) + 40] ^= 0x10;
    ref_result r = case_reference_run(&k, INGOT_CUDA_STORE, NULL);
    int differs = r.rc == 0 && memcmp(r.out, k.data, k.e.source_size) != 0;
    printf("    bad blocks %u, output %s\n", r.bad_blocks, differs ? "differs" : "same");
    ASSERT_TRUE(r.bad_blocks > 0);
    ref_free(&r);
    case_free(&k);
}

int main(void) {
    RUN_TEST(partner_term_matches_codec);
    RUN_TEST(bucket_matches_thresholds);
    RUN_TEST(table_lookup_matches_slots);
    RUN_TEST(reference_decodes_every_case);
    RUN_TEST(reference_matvec_close_and_repeatable);
    RUN_TEST(corrupt_block_is_flagged);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
