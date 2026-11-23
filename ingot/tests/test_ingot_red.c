/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_red.c
 * @brief  Red tests on a real .ingot file: for every plain or quant entry, (a) one flipped
 *         byte inside a block and (b) one unit of frequency moved between two entries of its
 *         first non-trivial table must each fail the decode (error or output hash mismatch).
 *         Untouched, every entry decodes.
 *         Usage: test_ingot_red [FILE.ingot]   (default: build/test_data/base.safetensors.ingot)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lib/test.h"
#include "ingot/codec/ingot_block.h"
#include "ingot/container/ingot_reader.h"
#include "ingot/core/ingot_error.h"

static const char *path = "build/test_data/base.safetensors.ingot";

static unsigned char *read_all(const char *p, long *size) {
    FILE *f = fopen(p, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)*size);
    if (buf && fread(buf, 1, (size_t)*size, f) != (size_t)*size) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    return buf;
}

/* Decode entry i of the file image `buf` (written to a sibling path); 1 if it fails. */
static int decode_fails(const unsigned char *buf, long size, size_t i) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s.red", path);
    FILE *f = fopen(tmp, "wb");
    fwrite(buf, 1, (size_t)size, f);
    fclose(f);
    ingot_reader *r = NULL;
    int failed = 1;
    if (ingot_reader_open(tmp, &r) == 0) {
        const ingot_entry *e = ingot_reader_entry(r, i);
        uint8_t *out = malloc((size_t)e->source_size);
        failed = !out || ingot_reader_decode(r, i, NULL, 4, out) != 0;
        free(out);
    }
    ingot_reader_close(r);
    remove(tmp);
    return failed;
}

/* Byte offsets (file) of the first table with >= 2 entries and of the middle of the first block,
 * for a version 2 plain or quant payload: frame (u64 prefix bytes, u64 blocks), mode prefix
 * (plain: 24-byte header; quant: 48-byte header), tables, block area. 0 if none. */
static int coded_offsets(const unsigned char *buf, const ingot_entry *e, uint64_t *table,
                         uint64_t *block_byte) {
    const unsigned char *p = buf + e->payload_offset;
    uint64_t prefix = ingot_load_u64(p), blocks = ingot_load_u64(p + 8);
    uint64_t at = INGOT_PAYLOAD_FRAME_BYTES + (e->mode == INGOT_MODE_PLAIN ? 24 : 48);
    *table = 0;
    while (at + 2 <= prefix && !*table) {
        uint32_t used = p[at] | (p[at + 1] << 8);
        if (used >= 2)
            *table = e->payload_offset + at;
        at += 2 + 4 * (uint64_t)used;
    }
    uint64_t first_end = ingot_load_u64(p + prefix + 8);
    *block_byte = e->payload_offset + prefix + 8 * (blocks + 1) + first_end / 2;
    return *table != 0;
}

static void move_frequency(unsigned char *t) {
    uint32_t used = t[0] | (t[1] << 8), big = 0;
    for (uint32_t k = 1; k < used; k++)
        if ((t[4 + 4 * k] | (t[5 + 4 * k] << 8)) > (t[4 + 4 * big] | (t[5 + 4 * big] << 8)))
            big = k;
    uint32_t small = big ? 0 : 1;
    uint32_t fb = (uint32_t)(t[4 + 4 * big] | (t[5 + 4 * big] << 8)) - 1u;
    uint32_t fs = (uint32_t)(t[4 + 4 * small] | (t[5 + 4 * small] << 8)) + 1u;
    t[4 + 4 * big] = (uint8_t)fb;
    t[5 + 4 * big] = (uint8_t)(fb >> 8);
    t[4 + 4 * small] = (uint8_t)fs;
    t[5 + 4 * small] = (uint8_t)(fs >> 8);
}

static void test_every_coded_entry_goes_red(void) {
    ingot_reader *r = NULL;
    ASSERT_INT_EQ(0, ingot_reader_open(path, &r));
    long size = 0;
    unsigned char *buf = read_all(path, &size);
    ASSERT_NOT_NULL(buf);
    size_t coded = 0, block_red = 0, table_red = 0, clean_ok = 0;
    for (size_t i = 0; buf && i < ingot_reader_count(r); i++) {
        const ingot_entry *e = ingot_reader_entry(r, i);
        uint64_t table = 0, byte = 0;
        if ((e->mode != INGOT_MODE_PLAIN && e->mode != INGOT_MODE_QUANT) ||
            !coded_offsets(buf, e, &table, &byte))
            continue;
        coded++;
        clean_ok += !decode_fails(buf, size, i);
        buf[byte] ^= 0x10;
        block_red += decode_fails(buf, size, i);
        buf[byte] ^= 0x10;
        unsigned char saved[2 + 4 * 1024];
        size_t tsize = 2 + 4 * (size_t)(buf[table] | (buf[table + 1] << 8));
        memcpy(saved, buf + table, tsize);
        move_frequency(buf + table);
        table_red += decode_fails(buf, size, i);
        memcpy(buf + table, saved, tsize);
    }
    printf("  red: %zu coded entries, clean decode %zu, corrupt block fails %zu, "
           "changed table entry fails %zu\n", coded, clean_ok, block_red, table_red);
    ASSERT_TRUE(coded > 0);
    ASSERT_TRUE(clean_ok == coded && block_red == coded && table_red == coded);
    free(buf);
    ingot_reader_close(r);
}

int main(int argc, char **argv) {
    if (argc > 1)
        path = argv[1];
    RUN_TEST(every_coded_entry_goes_red);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
