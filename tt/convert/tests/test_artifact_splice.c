/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_artifact_splice.c
 * @brief Host tests for SHA-256, the MXTL tile file and the FTTW splice.
 *
 * The gates: the MXTL golden fixture byte for byte; every reader refusal; a splice whose
 * listed entries equal the tile payloads and whose every other entry equals the base; and the
 * equivalence — splice(bfp8 rung, the bfp4 rung's own tiles for X, Y) is byte-identical to
 * convert(bfp8 + map X, Y = bfp4_b) up to the directory's end, followed by exactly the splice
 * record. Every splice refusal is a red case that must leave no output.
 *
 * @status None
 * @issues None
 * @todo None
 */

#define _DEFAULT_SOURCE

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ft_sha256.h"
#include "ft_tt_artifact_provenance.h"
#include "ft_tt_artifact_splice.h"
#include "ft_tt_format_map.h"
#include "ft_tt_precision_ledger.h"
#include "ft_tt_tile_file.h"
#include "ft_tt_weight_convert.h"
#include "synthetic_hybrid_checkpoint.h"

#define FIXTURE "build/splice_hybrid.gguf"
#define BASE "build/splice_bfp8.fttw"
#define RUNG4 "build/splice_bfp4.fttw"
#define MAPPED "build/splice_mapped.fttw"
#define TILES "build/splice_xy.mxtl"
#define OUT "build/splice_out.fttw"
#define REFUSED "build/splice_refused.fttw"
#define X_NAME "blk.1.ffn_gate.weight"
#define Y_NAME "blk.0.ffn_down.weight"

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (!condition)
        checks_failed++;
    printf("  %s  %s\n", condition ? "ok  " : "FAIL", what);
}

static void check_u64(uint64_t got, uint64_t want, const char *what)
{
    checks_run++;
    if (got != want)
        checks_failed++;
    printf("  %s  %s = %" PRIu64 "%s\n", got == want ? "ok  " : "FAIL", what, got, got == want ? "" : " (wrong)");
    if (got != want)
        printf("        expected %" PRIu64 "\n", want);
}

static void section(const char *name)
{
    printf("\n== %s ==\n", name);
}

/* ------------------------------------------------------------------------- */
/* Files                                                                     */
/* ------------------------------------------------------------------------- */

typedef struct {
    unsigned char *bytes;
    size_t size;
} blob;

static int blob_read(const char *path, blob *b)
{
    FILE *f = fopen(path, "rb");
    long size;

    memset(b, 0, sizeof(*b));
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    b->bytes = malloc(size > 0 ? (size_t)size : 1u);
    if (!b->bytes || fread(b->bytes, 1, (size_t)size, f) != (size_t)size) {
        fclose(f);
        return -1;
    }
    b->size = (size_t)size;
    fclose(f);
    return 0;
}

static int blob_write(const char *path, const void *bytes, size_t size)
{
    FILE *f = fopen(path, "wb");
    int ok = f && fwrite(bytes, 1, size, f) == size;

    if (f && fclose(f) != 0)
        ok = 0;
    return ok ? 0 : -1;
}

static void blob_free(blob *b)
{
    free(b->bytes);
    memset(b, 0, sizeof(*b));
}

static int file_exists(const char *path)
{
    return access(path, F_OK) == 0;
}

static const ft_tt_artifact_header *artifact_header(const blob *a)
{
    return (const ft_tt_artifact_header *)(const void *)a->bytes;
}

static const ft_tt_artifact_entry *artifact_entries(const blob *a)
{
    return (const ft_tt_artifact_entry *)(const void *)(a->bytes + artifact_header(a)->directory_offset);
}

static const ft_tt_artifact_entry *artifact_find(const blob *a, const char *name)
{
    for (uint32_t i = 0; i < artifact_header(a)->tensor_count; i++)
        if (strcmp(artifact_entries(a)[i].name, name) == 0)
            return &artifact_entries(a)[i];
    return NULL;
}

static uint32_t tile_bytes_for(uint32_t format)
{
    switch (format) {
    case FT_TT_FORMAT_FP32:
        return 4096u;
    case FT_TT_FORMAT_BF16:
        return 2048u;
    case FT_TT_FORMAT_BFP8_B:
        return 1088u;
    case FT_TT_FORMAT_BFP4_B:
        return 576u;
    default:
        return 0u;
    }
}

/**
 * What a loader checks (ft_tt_fttw_open: magic, version, flags, directory inside the file),
 * plus what the converter guarantees: 64-byte-aligned payloads inside the data region, and a
 * tiled entry's byte count equal to its grid times its format's tile.
 */
static int artifact_well_formed(const blob *a)
{
    const ft_tt_artifact_header *h = artifact_header(a);

    if (a->size < sizeof(*h) || h->magic != FT_TT_ARTIFACT_MAGIC || h->version < FT_TT_ARTIFACT_VERSION_MIN ||
        h->version > FT_TT_ARTIFACT_VERSION || (h->flags & ~FT_TT_ARTIFACT_FLAGS_KNOWN) ||
        h->total_bytes != h->directory_offset + (uint64_t)h->tensor_count * FT_TT_ARTIFACT_ENTRY_BYTES ||
        h->total_bytes > a->size)
        return 0;
    for (uint32_t i = 0; i < h->tensor_count; i++) {
        const ft_tt_artifact_entry *e = &artifact_entries(a)[i];

        if (e->data_offset % FT_TT_ARTIFACT_ALIGNMENT || e->data_offset < h->data_offset ||
            e->data_offset + e->byte_count > h->directory_offset)
            return 0;
        if (e->tile_rows && (uint64_t)e->tile_rows * e->tile_cols * tile_bytes_for(e->format) != e->byte_count)
            return 0;
    }
    return 1;
}

static int convert(const char *out, ft_tt_ledger_mode mode, const ft_tt_format_map *map)
{
    ft_tt_convert_options options;

    memset(&options, 0, sizeof(options));
    options.mode = mode;
    options.format_map = map;
    return ft_tt_convert_gguf(FIXTURE, out, &options);
}

static int splice_one(const char *source, const char *tiles, const char *out, char *error)
{
    const char *paths[1] = {tiles};

    return ft_tt_artifact_splice(source, paths, 1u, out, error, FT_TT_SPLICE_ERROR_BYTES);
}

/* ------------------------------------------------------------------------- */
/* SHA-256                                                                   */
/* ------------------------------------------------------------------------- */

static int sha_hex_is(const void *data, size_t bytes, const char *want)
{
    unsigned char digest[FT_SHA256_BYTES];
    char hex[2u * FT_SHA256_BYTES + 1u];

    ft_sha256(data, bytes, digest);
    ft_sha256_hex(digest, hex);
    return strcmp(hex, want) == 0;
}

static void test_sha256(void)
{
    static const char two_block[] = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    unsigned char *million = malloc(1000000u);
    unsigned char whole[FT_SHA256_BYTES], pieces[FT_SHA256_BYTES];
    ft_sha256_ctx ctx;

    section("SHA-256 against the FIPS 180-4 examples");
    check(sha_hex_is("", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"), "empty message");
    check(sha_hex_is("abc", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "\"abc\"");
    check(sha_hex_is(two_block, strlen(two_block),
                     "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
          "the 448-bit two-block message");
    if (!million) {
        check(0, "allocate the million-'a' message");
        return;
    }
    memset(million, 'a', 1000000u);
    check(sha_hex_is(million, 1000000u, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"),
          "one million 'a'");
    ft_sha256(million, 1000u, whole);
    ft_sha256_init(&ctx);
    ft_sha256_update(&ctx, million, 1u);
    ft_sha256_update(&ctx, million + 1, 63u);
    ft_sha256_update(&ctx, million + 64, 100u);
    ft_sha256_update(&ctx, million + 164, 836u);
    ft_sha256_final(&ctx, pieces);
    check(memcmp(whole, pieces, sizeof(whole)) == 0, "a split update gives the one-shot digest");
    free(million);
}

/* ------------------------------------------------------------------------- */
/* MXTL                                                                      */
/* ------------------------------------------------------------------------- */

#define GOLDEN_NAME "blk.0.ffn_up.weight"
#define GOLDEN_PAYLOAD_BYTES 576u
#define GOLDEN_FILE_BYTES 663u
#define GOLDEN_PAYLOAD_SHA "266d05fcd17ddd9f96f5cb7eae3c9f486fd5697a2b0198586a82af2cdc7c0558"
#define GOLDEN_FILE_SHA "e83db2dcf4c8c611e3f0372245a581731fc67af6c8dea93fdfd459bf12772006"

static void le32(unsigned char *out, uint32_t v)
{
    for (unsigned i = 0; i < 4u; i++)
        out[i] = (unsigned char)(v >> (8u * i));
}

static void le64(unsigned char *out, uint64_t v)
{
    for (unsigned i = 0; i < 8u; i++)
        out[i] = (unsigned char)(v >> (8u * i));
}

/** One MXTL entry built by hand from the contract; returns bytes written into `out`. */
static size_t hand_entry(unsigned char *out, const char *name, uint32_t format, uint32_t layout,
                         const unsigned char *payload, uint64_t n_bytes)
{
    size_t name_len = strlen(name);
    size_t at = 0;

    le32(out, (uint32_t)name_len);
    memcpy(out + 4, name, name_len);
    at = 4u + name_len;
    le32(out + at, format);
    le32(out + at + 4, layout);
    le64(out + at + 8, n_bytes);
    ft_sha256(payload, (size_t)n_bytes, out + at + 16);
    memcpy(out + at + 16 + FT_SHA256_BYTES, payload, (size_t)n_bytes);
    return at + 16u + FT_SHA256_BYTES + (size_t)n_bytes;
}

static size_t hand_header(unsigned char *out, uint32_t n_entries)
{
    memcpy(out, "MXTL", 4);
    le32(out + 4, 1u);
    le32(out + 8, n_entries);
    le32(out + 12, 0u);
    return 16u;
}

static void golden_payload(unsigned char *payload)
{
    for (unsigned i = 0; i < GOLDEN_PAYLOAD_BYTES; i++)
        payload[i] = (unsigned char)(i & 0xffu);
}

static void test_tile_file_golden(void)
{
    unsigned char payload[GOLDEN_PAYLOAD_BYTES];
    unsigned char hand[GOLDEN_FILE_BYTES + 64u];
    ft_tt_tile_file_payload source = {GOLDEN_NAME, 3u, 1u, payload, GOLDEN_PAYLOAD_BYTES};
    char error[FT_TT_TILE_FILE_ERROR_BYTES] = "";
    char hex[2u * FT_SHA256_BYTES + 1u];
    ft_tt_tile_file *file = NULL;
    const ft_tt_tile_file_entry *entry;
    size_t size;
    blob written;

    section("MXTL golden fixture");
    golden_payload(payload);
    size = hand_header(hand, 1u);
    size += hand_entry(hand + size, GOLDEN_NAME, 3u, 1u, payload, GOLDEN_PAYLOAD_BYTES);
    check_u64(size, GOLDEN_FILE_BYTES, "hand-built file bytes");
    check(sha_hex_is(payload, GOLDEN_PAYLOAD_BYTES, GOLDEN_PAYLOAD_SHA), "payload sha256 is the contract's");
    check(sha_hex_is(hand, size, GOLDEN_FILE_SHA), "hand-built file sha256 is the contract's");

    check(ft_tt_tile_file_write("build/golden.mxtl", &source, 1u, error, sizeof(error)) == FT_TT_TILE_FILE_OK,
          "writer wrote the fixture");
    if (blob_read("build/golden.mxtl", &written) != 0) {
        check(0, "read the written fixture");
        return;
    }
    check(written.size == size && memcmp(written.bytes, hand, size) == 0, "writer output equals the contract bytes");
    check(sha_hex_is(written.bytes, written.size, GOLDEN_FILE_SHA), "writer output sha256 is the contract's");
    blob_free(&written);

    check(ft_tt_tile_file_open("build/golden.mxtl", &file, error, sizeof(error)) == FT_TT_TILE_FILE_OK && file,
          "reader accepts it");
    entry = ft_tt_tile_file_entry_at(file, 0);
    check(ft_tt_tile_file_count(file) == 1u && entry && strcmp(entry->name, GOLDEN_NAME) == 0 &&
              entry->format == 3u && entry->layout == 1u && entry->n_bytes == GOLDEN_PAYLOAD_BYTES &&
              entry->payload_offset == GOLDEN_FILE_BYTES - GOLDEN_PAYLOAD_BYTES,
          "reader returns the entry's fields");
    if (entry) {
        unsigned char back[GOLDEN_PAYLOAD_BYTES];

        ft_sha256_hex(entry->sha256, hex);
        check(strcmp(hex, GOLDEN_PAYLOAD_SHA) == 0, "reader returns the payload sha256");
        check(ft_tt_tile_file_read_payload(file, 0, 0, back, sizeof(back)) == FT_TT_TILE_FILE_OK &&
                  memcmp(back, payload, sizeof(back)) == 0,
              "payload reads back");
        check(ft_tt_tile_file_read_payload(file, 0, 1, back, sizeof(back)) == FT_TT_TILE_FILE_EINVAL,
              "a read past the payload is refused");
    }
    ft_tt_tile_file_close(file);
}

/** Write `bytes`, open it, return the reader's status. */
static int open_status(const unsigned char *bytes, size_t size)
{
    char error[FT_TT_TILE_FILE_ERROR_BYTES] = "";
    ft_tt_tile_file *file = NULL;
    int rc;

    if (blob_write("build/bad.mxtl", bytes, size) != 0)
        return 99;
    rc = ft_tt_tile_file_open("build/bad.mxtl", &file, error, sizeof(error));
    if (rc != FT_TT_TILE_FILE_OK)
        printf("        %s\n", error);
    ft_tt_tile_file_close(file);
    return rc;
}

static void test_tile_file_refusals(void)
{
    static const size_t cuts[] = {0, 15, 16, 19, 30, 70, 100, 662};
    unsigned char payload[GOLDEN_PAYLOAD_BYTES];
    unsigned char good[GOLDEN_FILE_BYTES], bad[2u * GOLDEN_FILE_BYTES];
    ft_tt_tile_file_payload twice[2] = {{"a.weight", 3u, 1u, payload, 4u}, {"a.weight", 3u, 1u, payload, 4u}};
    char error[FT_TT_TILE_FILE_ERROR_BYTES];
    uint32_t truncations = 0;
    size_t size;

    section("MXTL reader refusals");
    golden_payload(payload);
    size = hand_header(good, 1u);
    size += hand_entry(good + size, GOLDEN_NAME, 3u, 1u, payload, GOLDEN_PAYLOAD_BYTES);

    memcpy(bad, good, size);
    bad[0] = 'N';
    check(open_status(bad, size) == FT_TT_TILE_FILE_EFORMAT, "bad magic");
    memcpy(bad, good, size);
    le32(bad + 4, 2u);
    check(open_status(bad, size) == FT_TT_TILE_FILE_EFORMAT, "version 2");
    memcpy(bad, good, size);
    le32(bad + 12, 1u);
    check(open_status(bad, size) == FT_TT_TILE_FILE_EFORMAT, "reserved != 0");
    for (size_t i = 0; i < sizeof(cuts) / sizeof(cuts[0]); i++)
        truncations += open_status(good, cuts[i]) == FT_TT_TILE_FILE_ETRUNCATED;
    check_u64(truncations, sizeof(cuts) / sizeof(cuts[0]), "truncations refused as ETRUNCATED");
    memcpy(bad, good, size);
    le32(bad + 8, 2u);
    check(open_status(bad, size) == FT_TT_TILE_FILE_ETRUNCATED, "n_entries larger than the file holds");
    memcpy(bad, good, size);
    bad[size - 1] ^= 0x01u;
    check(open_status(bad, size) == FT_TT_TILE_FILE_ESHA, "one payload bit flipped: sha256 mismatch");
    memcpy(bad, good, size);
    bad[size] = 0;
    check(open_status(bad, size + 1u) == FT_TT_TILE_FILE_EFORMAT, "a byte after the last payload");
    memcpy(bad, good, size);
    le32(bad + 16, 0u);
    check(open_status(bad, size) == FT_TT_TILE_FILE_EFORMAT, "a zero-length name");
    memcpy(bad, good, size);
    bad[22] = '\0';
    check(open_status(bad, size) == FT_TT_TILE_FILE_EFORMAT, "a name holding a NUL");

    size = hand_header(bad, 2u);
    size += hand_entry(bad + size, "a.weight", 3u, 1u, payload, 4u);
    size += hand_entry(bad + size, "a.weight", 3u, 1u, payload + 4, 4u);
    check(open_status(bad, size) == FT_TT_TILE_FILE_EDUPLICATE, "a name listed twice");

    check(ft_tt_tile_file_write("build/bad.mxtl", twice, 2u, error, sizeof(error)) == FT_TT_TILE_FILE_EDUPLICATE,
          "the writer refuses a duplicate name");
    twice[1].name = "";
    check(ft_tt_tile_file_write("build/bad.mxtl", &twice[1], 1u, error, sizeof(error)) == FT_TT_TILE_FILE_EINVAL,
          "the writer refuses an empty name");
}

/* ------------------------------------------------------------------------- */
/* Splice                                                                    */
/* ------------------------------------------------------------------------- */

/** Listed entries equal the tile payloads; every other entry equals the base. */
static void check_splice_payloads(const blob *out, const blob *base, const char *tiles_path)
{
    char error[FT_TT_TILE_FILE_ERROR_BYTES] = "";
    ft_tt_tile_file *tiles = NULL;
    uint32_t replaced_ok = 0, kept_ok = 0;
    uint32_t count = artifact_header(out)->tensor_count;

    if (ft_tt_tile_file_open(tiles_path, &tiles, error, sizeof(error)) != FT_TT_TILE_FILE_OK) {
        check(0, error);
        return;
    }
    for (uint32_t i = 0; i < count; i++) {
        const ft_tt_artifact_entry *got = &artifact_entries(out)[i];
        const ft_tt_artifact_entry *was = artifact_find(base, got->name);
        const ft_tt_tile_file_entry *tile = ft_tt_tile_file_find(tiles, got->name);

        if (!was)
            continue;
        if (tile) {
            unsigned char *payload = malloc((size_t)tile->n_bytes);
            uint32_t index = (uint32_t)(tile - ft_tt_tile_file_entry_at(tiles, 0));

            replaced_ok += payload && got->format == tile->format && got->byte_count == tile->n_bytes &&
                           ft_tt_tile_file_read_payload(tiles, index, 0, payload, (size_t)tile->n_bytes) == 0 &&
                           memcmp(out->bytes + got->data_offset, payload, (size_t)tile->n_bytes) == 0;
            free(payload);
        } else {
            kept_ok += got->format == was->format && got->byte_count == was->byte_count &&
                       memcmp(out->bytes + got->data_offset, base->bytes + was->data_offset,
                              (size_t)got->byte_count) == 0;
        }
    }
    check_u64(replaced_ok, ft_tt_tile_file_count(tiles), "replaced entries byte-equal to the tile payloads");
    check_u64(kept_ok, count - ft_tt_tile_file_count(tiles), "every other entry byte-identical to the base");
    ft_tt_tile_file_close(tiles);
}

static void check_record(const char *path, const blob *base, uint32_t want_flags)
{
    ft_tt_splice_record record;
    unsigned char directory_sha[FT_SHA256_BYTES];
    const ft_tt_artifact_header *h = artifact_header(base);

    memset(&record, 0, sizeof(record));
    check(ft_tt_splice_record_read(path, &record) == FT_TT_SPLICE_OK, "the splice record reads back");
    ft_sha256(base->bytes + h->directory_offset, (size_t)h->tensor_count * FT_TT_ARTIFACT_ENTRY_BYTES,
              directory_sha);
    check(record.entry_count == 2u && strcmp(record.entries[0].name, Y_NAME) == 0 &&
              strcmp(record.entries[1].name, X_NAME) == 0 && record.entries[0].format == FT_TT_FORMAT_BFP4_B &&
              record.entries[1].layout == FT_TT_LAYOUT_MATMUL_NMAJOR,
          "it names both tensors, in directory order, with format and layout");
    check(record.base_total_bytes == h->total_bytes &&
              memcmp(record.base_directory_sha256, directory_sha, FT_SHA256_BYTES) == 0,
          "it identifies the base: total bytes and directory sha256");
    check_u64(record.flags, want_flags, "record flags");
    ft_tt_splice_record_free(&record);
}

static void test_splice_equivalence(void)
{
    char error[FT_TT_SPLICE_ERROR_BYTES] = "";
    const char *xy[2] = {X_NAME, Y_NAME};
    blob out, base, mapped;
    uint64_t record_bytes = 2u * FT_TT_SPLICE_ENTRY_BYTES + FT_TT_SPLICE_FOOTER_BYTES;

    section("splice the bfp4 rung's tiles for X, Y into the bfp8 rung");
    remove(TILES);
    remove(OUT);
    check(ft_tt_artifact_extract_tiles(RUNG4, xy, 2u, TILES, error, sizeof(error)) == FT_TT_SPLICE_OK,
          "extracted X and Y from the bfp4 rung into a tile file");
    check(splice_one(BASE, TILES, OUT, error) == FT_TT_SPLICE_OK, "spliced");
    if (*error)
        printf("        %s\n", error);
    if (blob_read(OUT, &out) != 0 || blob_read(BASE, &base) != 0 || blob_read(MAPPED, &mapped) != 0) {
        check(0, "read the artifacts back");
        return;
    }
    if (!artifact_well_formed(&out)) {
        check(0, "the output is a well-formed artifact (loader + converter invariants)");
        blob_free(&out);
        blob_free(&base);
        blob_free(&mapped);
        return;
    }
    check(1, "the output is a well-formed artifact (loader + converter invariants)");
    check_u64(artifact_header(&out)->tensor_count, artifact_header(&base)->tensor_count, "tensor count");
    check_splice_payloads(&out, &base, TILES);
    check_record(OUT, &base, 0u);
    check(ft_tt_artifact_provenance_read(OUT, &(ft_tt_artifact_provenance){0}) == FT_TT_PROVENANCE_EABSENT,
          "no provenance trailer: the cache will not adopt it as a rung");

    section("equivalence: splice == convert(bfp8 + map X, Y = bfp4_b), then the record");
    check_u64(artifact_header(&out)->total_bytes, mapped.size, "FTTW bytes equal the mapped artifact's size");
    check(out.size == mapped.size + record_bytes && memcmp(out.bytes, mapped.bytes, mapped.size) == 0,
          "bytes [0, total_bytes) identical; the rest is exactly the record");
    blob_free(&out);
    blob_free(&base);
    blob_free(&mapped);
}

static void test_splice_two_files_and_provenance(void)
{
    char error[FT_TT_SPLICE_ERROR_BYTES] = "";
    const char *x[1] = {X_NAME}, *y[1] = {Y_NAME};
    const char *both[2] = {"build/splice_y.mxtl", "build/splice_x.mxtl"};
    blob one, two, base, with_trailer;
    ft_tt_splice_record record;
    int read_ok;

    section("two tile files give the same bytes as one");
    remove(both[0]);
    remove(both[1]);
    remove("build/splice_two.fttw");
    remove("build/splice_from_trailer.fttw");
    check(ft_tt_artifact_extract_tiles(RUNG4, x, 1u, both[1], error, sizeof(error)) == FT_TT_SPLICE_OK &&
              ft_tt_artifact_extract_tiles(RUNG4, y, 1u, both[0], error, sizeof(error)) == FT_TT_SPLICE_OK,
          "one tile file per tensor");
    check(ft_tt_artifact_splice(BASE, both, 2u, "build/splice_two.fttw", error, sizeof(error)) == FT_TT_SPLICE_OK,
          "spliced from two files");
    check(blob_read(OUT, &one) == 0 && blob_read("build/splice_two.fttw", &two) == 0 && one.size == two.size &&
              memcmp(one.bytes, two.bytes, one.size) == 0,
          "byte-identical to the one-file splice");
    blob_free(&one);
    blob_free(&two);

    section("a base with a provenance trailer: kept inside the record, not carried forward");
    if (blob_read(BASE, &base) != 0 || blob_write("build/splice_trailer.fttw", base.bytes, base.size) != 0) {
        check(0, "copy the base");
        return;
    }
    check(ft_tt_artifact_provenance_append("build/splice_trailer.fttw", 1234u, 0xfeedu, FT_TT_LEDGER_BFP8, 0u) ==
              FT_TT_PROVENANCE_OK,
          "appended a provenance trailer to a copy of the base");
    check(splice_one("build/splice_trailer.fttw", TILES, "build/splice_from_trailer.fttw", error) == FT_TT_SPLICE_OK,
          "spliced it");
    read_ok = blob_read("build/splice_from_trailer.fttw", &two) == 0 && blob_read(OUT, &one) == 0 &&
              blob_read("build/splice_trailer.fttw", &with_trailer) == 0 &&
              one.size > FT_TT_SPLICE_FOOTER_BYTES && with_trailer.size == base.size + 64u;
    check(read_ok, "read back");
    check(read_ok && two.size == one.size &&
              memcmp(two.bytes, one.bytes, one.size - FT_TT_SPLICE_FOOTER_BYTES) == 0,
          "same artifact and record entries as from the trailer-less base");
    memset(&record, 0, sizeof(record));
    check(read_ok && ft_tt_splice_record_read("build/splice_from_trailer.fttw", &record) == FT_TT_SPLICE_OK &&
              record.flags == FT_TT_SPLICE_FLAG_BASE_PROVENANCE &&
              memcmp(record.base_provenance, with_trailer.bytes + base.size, 64u) == 0,
          "the record holds the base's trailer verbatim");
    ft_tt_splice_record_free(&record);
    blob_free(&one);
    blob_free(&two);
    blob_free(&base);
    blob_free(&with_trailer);
}

/** A tile file holding one crafted entry; the payload content is irrelevant to the rules. */
static int write_crafted(const char *path, const char *name, uint32_t format, uint32_t layout, uint64_t n_bytes)
{
    unsigned char *payload = calloc(n_bytes ? (size_t)n_bytes : 1u, 1);
    ft_tt_tile_file_payload entry = {name, format, layout, payload, n_bytes};
    char error[FT_TT_TILE_FILE_ERROR_BYTES];
    int rc = payload ? ft_tt_tile_file_write(path, &entry, 1u, error, sizeof(error)) : -1;

    free(payload);
    return rc;
}

/** Splice `tiles` into `source` at REFUSED; 1 when refused with `want` and no file exists. */
static int refused_with(const char *source, const char *const *tiles, uint32_t count, const char *out, int want)
{
    char error[FT_TT_SPLICE_ERROR_BYTES] = "";
    int rc;

    remove(REFUSED);
    rc = ft_tt_artifact_splice(source, tiles, count, out, error, sizeof(error));
    printf("        (%d) %s\n", rc, error);
    return rc == want && !file_exists(REFUSED);
}

static void test_splice_refusals(void)
{
    static const struct {
        const char *name;
        uint32_t format;
        uint32_t layout;
        int64_t size_delta;
        int want;
        const char *what;
    } cases[] = {
        {"blk.0.ffn_upp.weight", 3u, 1u, 0, FT_TT_SPLICE_ENAME, "a name the source lacks"},
        {"token_embd.weight", 3u, 0u, 0, FT_TT_SPLICE_EROLE, "token_embd"},
        {"blk.0.ssm_alpha.weight", 3u, 1u, 0, FT_TT_SPLICE_EROLE, "ssm_alpha"},
        {"blk.0.ssm_beta.weight", 3u, 1u, 0, FT_TT_SPLICE_EROLE, "ssm_beta"},
        {"blk.0.ssm_conv1d.weight", 3u, 0u, 0, FT_TT_SPLICE_EROLE, "ssm_conv1d (not a matmul)"},
        {"blk.0.attn_norm.weight", 3u, 0u, 0, FT_TT_SPLICE_EROLE, "a 1-D norm"},
        {Y_NAME, FT_TT_FORMAT_BFP6_B, 1u, 0, FT_TT_SPLICE_EFORMAT, "bfp6_b is not a map format"},
        {Y_NAME, FT_TT_FORMAT_FP32, 1u, 0, FT_TT_SPLICE_EFORMAT, "fp32 is not a map format"},
        {Y_NAME, 3u, FT_TT_LAYOUT_SOURCE, 0, FT_TT_SPLICE_ELAYOUT, "layout 0 for an N-major entry"},
        {Y_NAME, 3u, 1u, -576, FT_TT_SPLICE_ESIZE, "one tile short"},
        {Y_NAME, 3u, 1u, 1, FT_TT_SPLICE_ESIZE, "one byte long"},
        {Y_NAME, 2u, 1u, 0, FT_TT_SPLICE_ESIZE, "bfp8_b format with a bfp4_b-sized payload"},
    };
    /* blk.0.ffn_down is [512 -> 256]: 16 x 8 tiles; 128 * 576 B at bfp4_b. */
    const uint64_t y_bfp4_bytes = 128u * 576u;
    const char *tiles[2] = {"build/splice_case.mxtl", TILES};
    const char *dup[2] = {TILES, "build/splice_x.mxtl"};
    const char *good[1] = {TILES};
    uint32_t refused = 0;
    blob bad;

    section("red: every refusal, and no output file after any of them");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint64_t size = (uint64_t)((int64_t)y_bfp4_bytes + cases[i].size_delta);
        int ok = write_crafted(tiles[0], cases[i].name, cases[i].format, cases[i].layout, size) == 0 &&
                 refused_with(BASE, tiles, 1u, REFUSED, cases[i].want);

        refused += (uint32_t)ok;
        printf("    %s  %s\n", ok ? "ok  " : "FAIL", cases[i].what);
    }
    check_u64(refused, sizeof(cases) / sizeof(cases[0]), "rule refusals with the right status");

    check(refused_with(BASE, dup, 2u, REFUSED, FT_TT_SPLICE_EDUPLICATE), "one tensor in two tile files");
    check(blob_read(TILES, &bad) == 0 && (bad.bytes[bad.size - 1] ^= 1u, 1) &&
              blob_write("build/splice_corrupt.mxtl", bad.bytes, bad.size) == 0,
          "wrote a tile file with one payload bit flipped");
    blob_free(&bad);
    tiles[0] = "build/splice_corrupt.mxtl";
    check(refused_with(BASE, tiles, 1u, REFUSED, FT_TT_SPLICE_ETILEFILE), "a tile file whose sha256 disagrees");
    check(refused_with(BASE, good, 1u, BASE, FT_TT_SPLICE_ESAMEPATH), "--out equal to the source");
    remove("build/splice_link.fttw");
    check(link(BASE, "build/splice_link.fttw") == 0 &&
              refused_with(BASE, good, 1u, "build/splice_link.fttw", FT_TT_SPLICE_ESAMEPATH),
          "--out a hard link to the source");
    check(refused_with(BASE, good, 1u, TILES, FT_TT_SPLICE_ESAMEPATH), "--out equal to a tile file");
    check(refused_with(OUT, good, 1u, REFUSED, FT_TT_SPLICE_ESOURCE), "a source that is already spliced");
    check(blob_read(BASE, &bad) == 0 && (bad.bytes = realloc(bad.bytes, bad.size + 100u)) != NULL &&
              memset(bad.bytes + bad.size, 0x5a, 100u) &&
              blob_write("build/splice_tail.fttw", bad.bytes, bad.size + 3u) == 0 &&
              blob_write("build/splice_tail100.fttw", bad.bytes, bad.size + 100u) == 0,
          "wrote bases with 3 and 100 stray bytes after the directory");
    check(refused_with("build/splice_tail.fttw", good, 1u, REFUSED, FT_TT_SPLICE_ESOURCE),
          "3 unknown bytes after the directory");
    check(refused_with("build/splice_tail100.fttw", good, 1u, REFUSED, FT_TT_SPLICE_ESOURCE),
          "100 unknown bytes after the directory");
    {
        blob trailer;

        check(blob_read("build/splice_trailer.fttw", &trailer) == 0 && (trailer.bytes[trailer.size - 48u] ^= 1u, 1) &&
                  blob_write("build/splice_bad_trailer.fttw", trailer.bytes, trailer.size) == 0,
              "wrote a base whose trailer's payload_bytes disagrees with the file");
        blob_free(&trailer);
    }
    check(refused_with("build/splice_bad_trailer.fttw", good, 1u, REFUSED, FT_TT_SPLICE_ESOURCE),
          "a provenance trailer that does not describe its file");
    bad.bytes[0] ^= 0xffu;
    check(blob_write("build/splice_magic.fttw", bad.bytes, bad.size) == 0 &&
              refused_with("build/splice_magic.fttw", good, 1u, REFUSED, FT_TT_SPLICE_ESOURCE),
          "a source with a bad magic");
    blob_free(&bad);
}

int main(void)
{
    static const char map_text[] = X_NAME " bfp4_b\n" Y_NAME " bfp4_b\n";
    ft_tt_format_map map;
    char error[FT_TT_FORMAT_MAP_ERROR_BYTES];

    test_sha256();
    test_tile_file_golden();
    test_tile_file_refusals();

    section("fixture: the synthetic Qwen3.8-shaped checkpoint, three ways");
    memset(&map, 0, sizeof(map));
    check(synthetic_hybrid_checkpoint_write(FIXTURE) == 0, "wrote the checkpoint");
    check(ft_tt_format_map_parse(map_text, strlen(map_text), "xy.map", &map, error, sizeof(error)) ==
              FT_TT_FORMAT_MAP_OK,
          "parsed the X, Y map");
    check(convert(BASE, FT_TT_LEDGER_BFP8, NULL) == FT_TT_CONVERT_OK &&
              convert(RUNG4, FT_TT_LEDGER_BFP4, NULL) == FT_TT_CONVERT_OK &&
              convert(MAPPED, FT_TT_LEDGER_BFP8, &map) == FT_TT_CONVERT_OK,
          "converted the bfp8 rung, the bfp4 rung, and bfp8 + map");
    ft_tt_format_map_free(&map);

    test_splice_equivalence();
    test_splice_two_files_and_provenance();
    test_splice_refusals();

    printf("\n%d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed ? 1 : 0;
}
