/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_container.c
 * @brief  .ingot file round trips on generated safetensors files: byte-identical rebuild (stream
 *         and random access), mixed dtypes, MoE groups, delta against safetensors and .ingot
 *         bases, version 1 files still read, and red tests (corrupt payload byte, wrong base).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>

#include "lib/test.h"
#include "ingot/codec/ingot_block.h"
#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_reader.h"
#include "ingot/container/ingot_stream.h"
#include "ingot/container/ingot_writer.h"
#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_error.h"
#include "safetensors/safetensors.h"
#include "ingot/tests/ingot_test_files.h"

#define DIR "build/test_data"
#define BASE_INGOT DIR "/base.safetensors.ingot" /* original name + ".ingot" */
#define TUNE_INGOT DIR "/tune.safetensors.ingot"

/* A small "model": 2-D BF16, 3-D BF16 experts, BF16 norm vector, F32 tensor. `tune` > 0 makes a
 * fine-tune of the same seed (same base weights plus noise). */
static int write_model(const char *path, uint64_t seed, double tune) {
    enum { E = 4, R = 512, C = 640, H = 1000 };
    uint16_t *experts = malloc(sizeof(uint16_t) * E * R * C);
    uint16_t *proj = malloc(sizeof(uint16_t) * 300 * H);
    uint16_t norm[H];
    float f32[H];
    if (!experts || !proj)
        return -1;
    test_rng = seed;
    for (size_t i = 0; i < (size_t)E * R * C; i++)
        experts[i] = ingot_f32_to_bf16((float)(0.02 * test_gauss()));
    for (size_t i = 0; i < (size_t)300 * H; i++)
        proj[i] = ingot_f32_to_bf16((float)(0.05 * test_gauss()));
    for (int i = 0; i < H; i++) {
        norm[i] = ingot_f32_to_bf16((float)(1.0 + 0.1 * test_gauss()));
        f32[i] = (float)test_gauss();
    }
    test_rng = seed ^ 0xABCDEFull;
    for (size_t i = 0; tune > 0 && i < (size_t)300 * H; i++)
        if (i / H % 3)
            proj[i] = ingot_f32_to_bf16(ingot_bf16_to_f32(proj[i]) + (float)(tune * test_gauss()));
    fs_writer *w = fs_writer_new();
    int64_t se[3] = {E, R, C}, sp[2] = {300, H}, sn[1] = {H};
    fs_writer_add_tensor(w, "model.experts.weight", FS_DTYPE_BF16, se, 3, experts);
    fs_writer_add_tensor(w, "model.proj.weight", FS_DTYPE_BF16, sp, 2, proj);
    fs_writer_add_tensor(w, "model.norm.weight", FS_DTYPE_BF16, sn, 1, norm);
    fs_writer_add_tensor(w, "model.f32", FS_DTYPE_F32, sn, 1, f32);
    int rc = fs_writer_save(w, path) ? 0 : -1;
    fs_writer_free(w);
    free(experts);
    free(proj);
    return rc;
}

/* Stream-decode an .ingot into out (base: a model file / .ingot / NULL). */
static int rebuild_with_base(const char *ingot, const char *base_path, const char *out) {
    ingot_base_set *b = NULL;
    ingot_input *in = NULL;
    const char *paths[1] = {base_path};
    int rc = base_path ? ingot_base_set_open(paths, 1, 2, &b) : 0;
    if (!rc)
        rc = ingot_input_open_path(ingot, &in);
    ingot_stream_options opt = {.threads = 2, .base = b};
    if (!rc)
        rc = ingot_stream_decode_to_file(in, out, &opt, NULL);
    ingot_input_close(in);
    ingot_base_set_close(b);
    return rc;
}

/* Random access: every entry decodes and matches the source bytes at its offset. */
static int random_access_matches(const char *ingot, const char *source, const char *base_path) {
    ingot_reader *r = NULL;
    ingot_base_set *b = NULL;
    const char *paths[1] = {base_path};
    int rc = base_path ? ingot_base_set_open(paths, 1, 2, &b) : 0;
    if (!rc)
        rc = ingot_reader_open(ingot, &r);
    FILE *f = fopen(source, "rb");
    for (size_t i = 0; !rc && f && i < ingot_reader_count(r); i++) {
        const ingot_entry *e = ingot_reader_entry(r, i);
        uint8_t *out = malloc((size_t)e->source_size + 1), *ref = malloc((size_t)e->source_size + 1);
        rc = ingot_reader_decode(r, i, b, 2, out);
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
    ingot_base_set_close(b);
    return f ? rc : -INGOT_EIO;
}

static void test_plain_file_rebuilds_byte_identical(void) {
    mkdir(DIR, 0755);
    ASSERT_INT_EQ(0, write_model(DIR "/base.safetensors", 7, 0.0));
    ingot_encode_options opt = {.threads = 3};
    ingot_encode_stats st;
    ASSERT_INT_EQ(0, ingot_encode_file(DIR "/base.safetensors", BASE_INGOT, &opt, &st));
    ASSERT_TRUE(st.tensors[INGOT_MODE_PLAIN] == 3 && st.tensors[INGOT_MODE_RAW] == 1);
    ASSERT_DBL_LT(8.0 * (double)st.payload_bytes[INGOT_MODE_PLAIN] /
                      (double)st.coded_elements[INGOT_MODE_PLAIN], 11.0);
    ASSERT_INT_EQ(0, rebuild_with_base(BASE_INGOT, NULL, DIR "/base.rebuilt.safetensors"));
    ASSERT_TRUE(test_files_equal(DIR "/base.safetensors", DIR "/base.rebuilt.safetensors"));
    ASSERT_INT_EQ(0, random_access_matches(BASE_INGOT, DIR "/base.safetensors", NULL));
}

static int encode_delta(const char *base_path, const char *out_path, ingot_policy policy,
                        ingot_encode_stats *st) {
    ingot_base_set *b = NULL;
    const char *paths[1] = {base_path};
    int rc = ingot_base_set_open(paths, 1, 2, &b);
    ingot_encode_options opt = {.threads = 2, .base = b, .policy = policy};
    if (!rc)
        rc = ingot_encode_file(DIR "/tune.safetensors", out_path, &opt, st);
    ingot_base_set_close(b);
    return rc;
}

static void test_delta_file_against_safetensors_and_ingot_base(void) {
    ASSERT_INT_EQ(0, write_model(DIR "/tune.safetensors", 7, 0.002));
    ingot_encode_stats st;
    ASSERT_INT_EQ(0, encode_delta(DIR "/base.safetensors", TUNE_INGOT, INGOT_POLICY_AUTO,
                                  &st));
    ASSERT_TRUE(st.tensors[INGOT_MODE_DELTA] >= 3); /* experts + norm identical, proj tuned */
    ASSERT_INT_EQ(0, rebuild_with_base(TUNE_INGOT, DIR "/base.safetensors",
                                       DIR "/tune.rebuilt.safetensors"));
    ASSERT_TRUE(test_files_equal(DIR "/tune.safetensors", DIR "/tune.rebuilt.safetensors"));
    remove(DIR "/tune.rebuilt.safetensors");
    ASSERT_INT_EQ(0, rebuild_with_base(TUNE_INGOT, BASE_INGOT,
                                       DIR "/tune.rebuilt.safetensors"));
    ASSERT_TRUE(test_files_equal(DIR "/tune.safetensors", DIR "/tune.rebuilt.safetensors"));
    ASSERT_INT_EQ(0, random_access_matches(TUNE_INGOT, DIR "/tune.safetensors",
                                           DIR "/base.safetensors"));
}

/* Red: without the base, or with the wrong base, a delta file must not rebuild. */
static void test_delta_wrong_or_missing_base_fails(void) {
    ASSERT_INT_EQ(-INGOT_EBASE, rebuild_with_base(TUNE_INGOT, NULL, DIR "/x.safetensors"));
    ASSERT_INT_EQ(0, write_model(DIR "/other.safetensors", 99, 0.0));
    ASSERT_INT_EQ(-INGOT_EBASE, rebuild_with_base(TUNE_INGOT, DIR "/other.safetensors",
                                                  DIR "/x.safetensors"));
    struct stat st;
    ASSERT_TRUE(stat(DIR "/x.safetensors", &st) != 0); /* partial output removed */
    ASSERT_INT_EQ(-INGOT_EBASE, random_access_matches(TUNE_INGOT, DIR "/tune.safetensors",
                                                      DIR "/other.safetensors"));
}

/* Red: one flipped payload byte must fail exactly one entry (random access) and the stream. */
static void test_corrupt_payload_byte_fails(void) {
    long size = test_file_size(BASE_INGOT);
    ASSERT_INT_EQ(0, test_copy_flipped(BASE_INGOT, DIR "/bad.safetensors.ingot", size / 2, 0x04,
                                       -1)); /* inside the experts payload */
    ingot_reader *r = NULL;
    ASSERT_INT_EQ(0, ingot_reader_open(DIR "/bad.safetensors.ingot", &r));
    int failures = 0;
    for (size_t i = 0; i < ingot_reader_count(r); i++) {
        const ingot_entry *e = ingot_reader_entry(r, i);
        uint8_t *out = malloc((size_t)e->source_size);
        failures += ingot_reader_decode(r, i, NULL, 1, out) != 0;
        free(out);
    }
    ingot_reader_close(r);
    ASSERT_INT_EQ(1, failures);
    int rc = rebuild_with_base(DIR "/bad.safetensors.ingot", NULL, DIR "/bad.safetensors");
    ASSERT_TRUE(rc == -INGOT_ECORRUPT || rc == -INGOT_EFORMAT);
}

/* Write a version 1 file with the same content as a version 2 one (plain / raw entries): 64-byte
 * header, source header, unframed payloads, directory at the end. */
static int write_v1_copy(const char *v2_path, const char *v1_path) {
    ingot_reader *r = NULL;
    int rc = ingot_reader_open(v2_path, &r);
    FILE *in = fopen(v2_path, "rb"), *out = fopen(v1_path, "wb");
    if (rc || !in || !out)
        rc = rc ? rc : -INGOT_EIO;
    uint64_t hsize = 0;
    const uint8_t *src_header = rc ? NULL : ingot_reader_source_header(r, &hsize);
    uint8_t zeros[64] = {0}, *buf = NULL;
    size_t n = rc ? 0 : ingot_reader_count(r);
    ingot_entry *entries = calloc(n ? n : 1, sizeof(*entries));
    if (!rc) {
        fwrite(zeros, 1, 64, out);
        fwrite(src_header, 1, (size_t)hsize, out);
        fwrite(zeros, 1, (8 - (size_t)(64 + hsize) % 8) % 8, out);
    }
    for (size_t i = 0; !rc && i < n; i++) {
        entries[i] = *ingot_reader_entry(r, i);
        uint64_t skip = entries[i].mode == INGOT_MODE_RAW ? 0 : INGOT_PAYLOAD_FRAME_BYTES;
        buf = realloc(buf, (size_t)entries[i].payload_size + 1);
        fseek(in, (long)(entries[i].payload_offset + skip), SEEK_SET);
        size_t want = (size_t)(entries[i].payload_size - skip);
        rc = fread(buf, 1, want, in) == want ? 0 : -1;
        entries[i].payload_offset = (uint64_t)ftell(out);
        entries[i].payload_size -= skip;
        fwrite(buf, 1, (size_t)entries[i].payload_size, out);
        fwrite(zeros, 1, (8 - (size_t)entries[i].payload_size % 8) % 8, out);
    }
    uint64_t dir_offset = out ? (uint64_t)ftell(out) : 0, dir_size = 0;
    for (size_t i = 0; !rc && i < n; i++) {
        uint8_t rec[4096];
        size_t len = ingot_entry_write(&entries[i], rec);
        fwrite(rec, 1, len, out);
        dir_size += len;
    }
    if (!rc) {
        uint8_t h[64] = {0};
        memcpy(h, "INGT", 4);
        ingot_store_u32(h + 4, 1);
        ingot_store_u32(h + 8, (uint32_t)n);
        ingot_store_u64(h + 16, 64);
        ingot_store_u64(h + 24, hsize);
        ingot_store_u64(h + 32, dir_offset);
        ingot_store_u64(h + 40, dir_size);
        ingot_store_u64(h + 48, ingot_reader_source_size(r));
        ingot_store_u64(h + 56, dir_offset + dir_size);
        fseek(out, 0, SEEK_SET);
        fwrite(h, 1, 64, out);
    }
    free(buf);
    free(entries);
    if (in)
        fclose(in);
    if (out)
        fclose(out);
    ingot_reader_close(r);
    return rc;
}

/* Version 1 files (directory at the end, unframed payloads) still decode by random access. */
static void test_version1_file_still_reads(void) {
    ASSERT_INT_EQ(0, write_v1_copy(BASE_INGOT, DIR "/base.v1.ingot"));
    uint32_t version = 0;
    ASSERT_INT_EQ(1, ingot_reader_probe(DIR "/base.v1.ingot", &version));
    ASSERT_INT_EQ(1, (int)version);
    ASSERT_INT_EQ(0, random_access_matches(DIR "/base.v1.ingot", DIR "/base.safetensors", NULL));
    ingot_reader *r = NULL;
    remove(DIR "/base.v1.rebuilt");
    ASSERT_INT_EQ(0, ingot_reader_open(DIR "/base.v1.ingot", &r));
    ASSERT_INT_EQ(0, ingot_reader_rebuild(r, DIR "/base.v1.rebuilt", NULL, 2));
    ingot_reader_close(r);
    ASSERT_TRUE(test_files_equal(DIR "/base.safetensors", DIR "/base.v1.rebuilt"));
    /* the stream decoder refuses version 1 (its directory is at the end) */
    ASSERT_INT_EQ(-INGOT_EUNSUPPORTED, rebuild_with_base(DIR "/base.v1.ingot", NULL,
                                                         DIR "/x.safetensors"));
}

/* Files are recognised by magic + version, not by name; other versions and magics are refused. */
static void test_magic_and_version_not_name(void) {
    uint32_t version = 0;
    ASSERT_INT_EQ(1, ingot_reader_probe(BASE_INGOT, &version));
    ASSERT_INT_EQ(2, (int)version); /* nothing predicted: written as version 2 */
    ASSERT_INT_EQ(0, ingot_reader_probe(DIR "/base.safetensors", NULL));
    ASSERT_INT_EQ(0, memcmp("INGT", INGOT_MAGIC, 4));

    ASSERT_INT_EQ(0, test_copy_flipped(BASE_INGOT, DIR "/renamed.bin", 0, 0, -1));
    ingot_reader *r = NULL;
    ASSERT_INT_EQ(0, ingot_reader_open(DIR "/renamed.bin", &r));
    ingot_reader_close(r);
    /* a renamed .ingot base still serves a delta */
    remove(DIR "/tune.rebuilt.safetensors");
    ASSERT_INT_EQ(0, rebuild_with_base(TUNE_INGOT, DIR "/renamed.bin",
                                       DIR "/tune.rebuilt.safetensors"));
    ASSERT_TRUE(test_files_equal(DIR "/tune.safetensors", DIR "/tune.rebuilt.safetensors"));

    /* a version 2 file is a valid version 3 file (same layout) */
    ASSERT_INT_EQ(0, test_copy_flipped(BASE_INGOT, DIR "/v3.ingot", 4, 0x01, -1)); /* 2 -> 3 */
    remove(DIR "/v3.rebuilt.safetensors");
    ASSERT_INT_EQ(0, rebuild_with_base(DIR "/v3.ingot", NULL, DIR "/v3.rebuilt.safetensors"));
    ASSERT_TRUE(test_files_equal(DIR "/base.safetensors", DIR "/v3.rebuilt.safetensors"));
    ASSERT_INT_EQ(0, test_copy_flipped(BASE_INGOT, DIR "/v6.ingot", 4, 0x04, -1)); /* 2 -> 6 */
    r = NULL;
    ASSERT_INT_EQ(-INGOT_EUNSUPPORTED, ingot_reader_open(DIR "/v6.ingot", &r));
    ASSERT_INT_EQ(-INGOT_EUNSUPPORTED, rebuild_with_base(DIR "/v6.ingot", NULL,
                                                         DIR "/x.safetensors"));
    ASSERT_INT_EQ(1, ingot_reader_probe(DIR "/v6.ingot", &version));
    ASSERT_INT_EQ(6, (int)version);
    /* a version 2 file relabelled 4 or 5 lacks the directory tail */
    ASSERT_INT_EQ(0, test_copy_flipped(BASE_INGOT, DIR "/v4.ingot", 4, 0x06, -1)); /* 2 -> 4 */
    ASSERT_INT_EQ(-INGOT_EFORMAT, ingot_reader_open(DIR "/v4.ingot", &r));
    ASSERT_INT_EQ(0, test_copy_flipped(BASE_INGOT, DIR "/v5.ingot", 4, 0x07, -1)); /* 2 -> 5 */
    ASSERT_INT_EQ(-INGOT_EFORMAT, ingot_reader_open(DIR "/v5.ingot", &r));
    ASSERT_INT_EQ(0, test_copy_flipped(BASE_INGOT, DIR "/nomagic.ingot", 0, 0x20, -1)); /* iNGT */
    ASSERT_INT_EQ(-INGOT_EFORMAT, ingot_reader_open(DIR "/nomagic.ingot", &r));
    ASSERT_INT_EQ(0, ingot_reader_probe(DIR "/nomagic.ingot", NULL));
}

int main(void) {
    RUN_TEST(plain_file_rebuilds_byte_identical);
    RUN_TEST(delta_file_against_safetensors_and_ingot_base);
    RUN_TEST(delta_wrong_or_missing_base_fails);
    RUN_TEST(corrupt_payload_byte_fails);
    RUN_TEST(version1_file_still_reads);
    RUN_TEST(magic_and_version_not_name);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
