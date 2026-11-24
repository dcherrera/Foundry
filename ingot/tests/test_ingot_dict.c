/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_dict.c
 * @brief  Format version 5: a dictionary ref (--dict-ref), the first rows of a tensor of another
 *         file (lm_head), joins the match pool. Prefix decode of a ref-free .ingot entry by random
 *         access; planted token-combination rows code smaller and round-trip bit-identical with
 *         the dictionary read from the other file's .ingot (stream and random access) or its
 *         source; reds: missing, wrong (hash) and not ref-free dictionaries refused, a version 5
 *         file relabelled 4 refused; without the dictionary (or at --max-version 4) the writer
 *         writes the same version 4 file as before.
 *         Built with -DINGOT_TEST_RED (no planted rows) the size check must fail (make red).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>

#include "lib/test.h"
#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_reader.h"
#include "ingot/container/ingot_stream.h"
#include "ingot/container/ingot_writer.h"
#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_error.h"
#include "safetensors/safetensors.h"
#include "ingot/tests/ingot_test_files.h"

#define DICT_NAME "lm_head.weight"

/* Dictionary [V, H] (V * H spans 2.3 plain blocks of 2^18 weights); target layer 5 gate
 * and up [I, H]; the dictionary prefix the encoder may use is the first DICT_ROWS rows. */
enum { H = 512, V = 1200, I = 400, DICT_ROWS = 600 };

#ifdef INGOT_TEST_RED
#define PLANT_OFF 1
#define DIR "build/test_data/red" /* its own files: the suite's stay planted */
#else
#define PLANT_OFF 0
#define DIR "build/test_data"
#endif

static uint16_t *gauss_bf16(size_t n, double sigma) {
    uint16_t *w = malloc(n * sizeof(*w));
    for (size_t i = 0; i < n; i++)
        w[i] = ingot_f32_to_bf16((float)(sigma * test_gauss()));
    return w;
}

static double val(uint16_t w) { return (double)ingot_bf16_to_f32(w); }

static int save_one(const char *path, const char *name, int64_t rows, int64_t cols,
                    const uint16_t *data) {
    int64_t shape[2] = {rows, cols};
    fs_writer *w = fs_writer_new();
    fs_writer_add_tensor(w, name, FS_DTYPE_BF16, shape, 2, data);
    int rc = fs_writer_save(w, path) ? 0 : -1;
    fs_writer_free(w);
    return rc;
}

/* The dictionary file (seed picks a different tensor with the same name and shape). */
static int write_dict(const char *path, uint64_t seed) {
    test_rng = seed;
    uint16_t *d = gauss_bf16((size_t)V * H, 0.02);
    int rc = save_one(path, DICT_NAME, V, H, d);
    free(d);
    return rc;
}

/* Layer 5 gate: every other row 0.9 dict[a] + 0.5 dict[b] + 0.3-sigma noise, a, b in the
 * prefix of write_dict(seed)'s tensor (the red build plants nothing); up: noise. */
static int write_target(const char *path, uint64_t seed) {
    test_rng = seed;
    uint16_t *d = gauss_bf16((size_t)V * H, 0.02);
    test_rng = 77;
    uint16_t *gate = gauss_bf16((size_t)I * H, 0.02), *up = gauss_bf16((size_t)I * H, 0.02);
    for (size_t r = 0; r < I && !PLANT_OFF; r += 2) {
        size_t a = (r * 37 + 11) % DICT_ROWS, b = (r * 101 + 5) % DICT_ROWS;
        for (size_t h = 0; h < H; h++)
            gate[r * H + h] = ingot_f32_to_bf16((float)(0.9 * val(d[a * H + h]) +
                                                        0.5 * val(d[b * H + h]) +
                                                        0.3 * 0.02 * test_gauss()));
    }
    int64_t shape[2] = {I, H};
    fs_writer *w = fs_writer_new();
    fs_writer_add_tensor(w, "m.layers.5.mlp.gate_proj.weight", FS_DTYPE_BF16, shape, 2, gate);
    fs_writer_add_tensor(w, "m.layers.5.mlp.up_proj.weight", FS_DTYPE_BF16, shape, 2, up);
    int rc = fs_writer_save(w, path) ? 0 : -1;
    fs_writer_free(w);
    free(d);
    free(gate);
    free(up);
    return rc;
}

/* Encode src with refs = ref (or none); dict: use the dictionary for layers first..last. */
static int encode(const char *src, const char *dst, const char *ref, int dict, long first,
                  long last, int max_version, ingot_encode_stats *st) {
    ingot_base_set *refs = NULL;
    if (ref && ingot_base_set_open(&ref, 1, 2, &refs))
        return -1;
    ingot_encode_options opt = {.threads = 3, .self_check = 1, .refs = refs,
                                .max_version = max_version, .dict_ref = dict ? DICT_NAME : NULL,
                                .dict_rows = DICT_ROWS, .dict_first = first, .dict_last = last};
    int rc = ingot_encode_file(src, dst, &opt, st);
    ingot_base_set_close(refs);
    return rc;
}

static int decode(const char *src, const char *dst, const char *ref) {
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

/* Random access: every entry decodes to the source bytes. */
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

static uint32_t version_of(const char *path) {
    uint32_t version = 0;
    return ingot_reader_probe(path, &version) == 1 ? version : 0;
}

static void test_setup_files(void) {
    ingot_encode_stats st;
    ASSERT_INT_EQ(0, write_dict(DIR "/zdict.safetensors", 4100));
    ASSERT_INT_EQ(0, write_dict(DIR "/zwrong.safetensors", 4101));
    ASSERT_INT_EQ(0, write_target(DIR "/dtarget.safetensors", 4100));
    ASSERT_INT_EQ(0, encode(DIR "/zdict.safetensors", DIR "/zdict.ingot", NULL, 0, 0, 0, 0, &st));
    ASSERT_INT_EQ(0, encode(DIR "/zwrong.safetensors", DIR "/zwrong.ingot", NULL, 0, 0, 0, 0,
                            &st));
    ASSERT_INT_EQ(1, (int)st.tensors[INGOT_MODE_PLAIN]);
    /* not ref-free: the dictionary coded as a delta against the wrong one */
    ingot_base_set *base = NULL;
    const char *bp = DIR "/zwrong.safetensors";
    ASSERT_INT_EQ(0, ingot_base_set_open(&bp, 1, 2, &base));
    ingot_encode_options opt = {.threads = 2, .base = base, .policy = INGOT_POLICY_DELTA};
    ASSERT_INT_EQ(0, ingot_encode_file(DIR "/zdict.safetensors", DIR "/zdelta.ingot", &opt, &st));
    ingot_base_set_close(base);
    ASSERT_INT_EQ(1, (int)st.tensors[INGOT_MODE_DELTA]);
}

/* Bytes [0, n) of the plain dictionary entry, decoded by random access, at block edges. */
static void test_prefix_decode_random_access(void) {
    ingot_reader *r = NULL;
    ASSERT_INT_EQ(0, ingot_reader_open(DIR "/zdict.ingot", &r));
    size_t i = 0;
    ASSERT_INT_EQ(0, ingot_reader_find(r, DICT_NAME, &i));
    const ingot_entry *e = ingot_reader_entry(r, i);
    ASSERT_INT_EQ(INGOT_MODE_PLAIN, (int)e->mode);
    uint8_t *all = malloc((size_t)e->source_size), *part = malloc((size_t)e->source_size);
    ASSERT_INT_EQ(0, ingot_reader_decode(r, i, NULL, 2, all));
    const uint64_t cuts[] = {2, 1000, 2u << 18, (2u << 18) + 10, (4u << 18) + 2, e->source_size};
    for (size_t k = 0; k < sizeof(cuts) / sizeof(cuts[0]); k++) {
        memset(part, 0xA5, (size_t)e->source_size);
        ASSERT_INT_EQ(0, ingot_reader_decode_prefix(r, i, cuts[k], 2, part));
        ASSERT_INT_EQ(0, memcmp(part, all, (size_t)cuts[k]));
        if (cuts[k] < e->source_size)
            ASSERT_TRUE(part[cuts[k]] == 0xA5); /* nothing written past n */
    }
    ASSERT_INT_EQ(-INGOT_EINVAL, ingot_reader_decode_prefix(r, i, e->source_size + 2, 2, part));
    ingot_reader_close(r);
    ASSERT_INT_EQ(0, ingot_reader_open(DIR "/zdelta.ingot", &r));
    ASSERT_INT_EQ(-INGOT_EUNSUPPORTED, ingot_reader_decode_prefix(r, 0, 1000, 2, part));
    ingot_reader_close(r);
    free(all);
    free(part);
}

static void test_dictionary_round_trip(void) {
    ingot_encode_stats with, without;
    ASSERT_INT_EQ(0, encode(DIR "/dtarget.safetensors", DIR "/dtarget.ingot",
                            DIR "/zdict.safetensors", 1, 0, 63, 0, &with));
    ASSERT_INT_EQ(0, encode(DIR "/dtarget.safetensors", DIR "/dtarget.v4.ingot",
                            DIR "/zdict.safetensors", 0, 0, 0, 0, &without));
    printf("    with dictionary %llu bytes, without %llu (%.3f)\n",
           (unsigned long long)with.output_bytes, (unsigned long long)without.output_bytes,
           (double)with.output_bytes / (double)without.output_bytes);
    ASSERT_TRUE(with.output_bytes < without.output_bytes * 975 / 1000); /* planted gate rows */
    ASSERT_INT_EQ(5, (int)version_of(DIR "/dtarget.ingot"));
    ASSERT_INT_EQ(4, (int)version_of(DIR "/dtarget.v4.ingot"));

    /* the dictionary from the other file's .ingot (prefix decoded) or from its source */
    ASSERT_INT_EQ(0, decode(DIR "/dtarget.ingot", DIR "/dtarget.rebuilt", DIR "/zdict.ingot"));
    ASSERT_TRUE(test_files_equal(DIR "/dtarget.safetensors", DIR "/dtarget.rebuilt"));
    remove(DIR "/dtarget.rebuilt");
    ASSERT_INT_EQ(0, decode(DIR "/dtarget.ingot", DIR "/dtarget.rebuilt",
                            DIR "/zdict.safetensors"));
    ASSERT_TRUE(test_files_equal(DIR "/dtarget.safetensors", DIR "/dtarget.rebuilt"));
    ASSERT_INT_EQ(0, random_access_matches(DIR "/dtarget.ingot", DIR "/dtarget.safetensors",
                                           DIR "/zdict.ingot"));
}

static void test_bad_dictionary_refused(void) {
    ASSERT_INT_EQ(-INGOT_EBASE, decode(DIR "/dtarget.ingot", DIR "/x.st", NULL));
    ASSERT_INT_EQ(-INGOT_EBASE, decode(DIR "/dtarget.ingot", DIR "/x.st", DIR "/zwrong.ingot"));
    ASSERT_INT_EQ(-INGOT_EBASE, decode(DIR "/dtarget.ingot", DIR "/x.st", DIR "/zdelta.ingot"));
    ASSERT_TRUE(random_access_matches(DIR "/dtarget.ingot", DIR "/dtarget.safetensors",
                                      DIR "/zwrong.ingot") != 0);
    /* relabelled 4, the row-prefix ref is not allowed */
    ASSERT_INT_EQ(0, test_copy_flipped(DIR "/dtarget.ingot", DIR "/dtarget.as4.ingot", 4, 0x01,
                                       -1));
    ASSERT_INT_EQ(-INGOT_EBASE, decode(DIR "/dtarget.as4.ingot", DIR "/x.st",
                                       DIR "/zdict.ingot"));
}

/* No dictionary in range, or --max-version 4: the version 4 file, byte for byte. */
static void test_without_dictionary_file_unchanged(void) {
    ingot_encode_stats st;
    ASSERT_INT_EQ(0, encode(DIR "/dtarget.safetensors", DIR "/dtarget.off.ingot",
                            DIR "/zdict.safetensors", 1, 6, 63, 0, &st));
    ASSERT_TRUE(test_files_equal(DIR "/dtarget.v4.ingot", DIR "/dtarget.off.ingot"));
    ASSERT_INT_EQ(0, encode(DIR "/dtarget.safetensors", DIR "/dtarget.max4.ingot",
                            DIR "/zdict.safetensors", 1, 0, 63, 4, &st));
    ASSERT_TRUE(test_files_equal(DIR "/dtarget.v4.ingot", DIR "/dtarget.max4.ingot"));
    ASSERT_INT_EQ(0, decode(DIR "/dtarget.v4.ingot", DIR "/dtarget.v4.rebuilt", NULL));
    ASSERT_TRUE(test_files_equal(DIR "/dtarget.safetensors", DIR "/dtarget.v4.rebuilt"));
}

int main(void) {
    mkdir("build", 0755);
    mkdir("build/test_data", 0755);
    mkdir(DIR, 0755);
    RUN_TEST(setup_files);
    RUN_TEST(prefix_decode_random_access);
    RUN_TEST(dictionary_round_trip);
    RUN_TEST(bad_dictionary_refused);
    RUN_TEST(without_dictionary_file_unchanged);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
