/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_reader.c
 * @brief  .ingot random-access reader: validation, decode, version 1 rebuild (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_reader.h"

#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ingot/codec/ingot_tensor_codec.h"
#include "ingot/container/ingot_base.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"

struct ingot_reader {
    uint8_t *map;
    uint64_t size;
    ingot_file_header header;
    ingot_entry *entries;
};

static int parse_file(ingot_reader *r) {
    int rc = ingot_header_parse(r->map, (size_t)r->size, &r->header);
    if (rc == -INGOT_ETRUNCATED)
        rc = -INGOT_EFORMAT;
    if (!rc && r->header.file_size != r->size)
        rc = -INGOT_EFORMAT;
    if (!rc)
        rc = ingot_directory_parse(r->map + r->header.dir_offset, r->header.dir_size,
                                   r->header.entry_count, &r->entries);
    for (size_t i = 0; !rc && i < r->header.entry_count; i++)
        rc = ingot_entry_check(&r->entries[i], &r->header);
    return rc;
}

int ingot_reader_open(const char *path, ingot_reader **out) {
    if (!path || !out)
        return -INGOT_EINVAL;
    ingot_reader *r = calloc(1, sizeof(*r));
    if (!r)
        return -INGOT_ENOMEM;
    int fd = open(path, O_RDONLY);
    struct stat st;
    int rc = (fd >= 0 && fstat(fd, &st) == 0) ? 0 : -INGOT_EIO;
    if (!rc) {
        r->size = (uint64_t)st.st_size;
        rc = r->size >= 8 ? 0 : -INGOT_EFORMAT;
    }
    if (!rc) {
        void *m = mmap(NULL, (size_t)r->size, PROT_READ, MAP_PRIVATE, fd, 0);
        r->map = m == MAP_FAILED ? NULL : m;
        rc = r->map ? 0 : -INGOT_EIO;
    }
    if (fd >= 0)
        close(fd);
    if (!rc)
        rc = parse_file(r);
    if (rc) {
        ingot_reader_close(r);
        return rc;
    }
    *out = r;
    return 0;
}

int ingot_reader_probe(const char *path, uint32_t *version) {
    if (!path)
        return -INGOT_EINVAL;
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return -INGOT_EIO;
    uint8_t h[8];
    ssize_t n = read(fd, h, sizeof(h));
    close(fd);
    if (n < 0)
        return -INGOT_EIO;
    if (n < (ssize_t)sizeof(h) || memcmp(h, INGOT_MAGIC, INGOT_MAGIC_BYTES))
        return 0;
    if (version)
        *version = (uint32_t)h[4] | ((uint32_t)h[5] << 8) | ((uint32_t)h[6] << 16) |
                   ((uint32_t)h[7] << 24);
    return 1;
}

void ingot_reader_close(ingot_reader *r) {
    if (!r)
        return;
    ingot_entries_free(r->entries, r->header.entry_count);
    if (r->map)
        munmap(r->map, (size_t)r->size);
    free(r);
}

const ingot_file_header *ingot_reader_header(const ingot_reader *r) {
    return r ? &r->header : NULL;
}

size_t ingot_reader_count(const ingot_reader *r) { return r ? r->header.entry_count : 0; }

const ingot_entry *ingot_reader_entry(const ingot_reader *r, size_t i) {
    return (r && i < r->header.entry_count) ? &r->entries[i] : NULL;
}

int ingot_reader_find(const ingot_reader *r, const char *name, size_t *index) {
    for (size_t i = 0; r && name && i < r->header.entry_count; i++) {
        if (r->entries[i].name[0] && !strcmp(r->entries[i].name, name)) {
            *index = i;
            return 0;
        }
    }
    return -INGOT_ENOTFOUND;
}

uint64_t ingot_reader_file_size(const ingot_reader *r) { return r ? r->size : 0; }

uint64_t ingot_reader_source_size(const ingot_reader *r) {
    return r ? r->header.source_size : 0;
}

const uint8_t *ingot_reader_source_header(const ingot_reader *r, uint64_t *size) {
    *size = r->header.source_header_size;
    return r->map + r->header.source_header_offset;
}

static int is_delta(ingot_mode m) {
    return m == INGOT_MODE_DELTA || m == INGOT_MODE_QUANT_DELTA || m == INGOT_MODE_BLOCK_DELTA;
}

/* The whole base tensor of a delta entry, checked against the stored base hash. */
static int load_base(const ingot_entry *e, ingot_base_set *base, const uint8_t **data) {
    ingot_base_tensor t;
    if (!base || ingot_base_set_find(base, e->name, &t) || t.size != e->source_size ||
        (e->mode != INGOT_MODE_BLOCK_DELTA && t.dtype != e->dtype))
        return -INGOT_EBASE;
    int rc = ingot_base_set_load(base, &t, data);
    if (rc)
        return rc == -INGOT_EUNSUPPORTED ? -INGOT_EBASE : rc;
    return ingot_hash64(*data, (size_t)e->source_size) == e->base_hash ? 0 : -INGOT_EBASE;
}

/* Decode a predict entry: its partners first (each strictly earlier, so this terminates), then
 * the entry against them. */
static int decode_predicted(const ingot_reader *r, const ingot_entry *e, size_t i,
                            ingot_base_set *base, int threads, uint8_t *out) {
    uint32_t index[2] = {0};
    uint32_t n = ingot_partner_list(e->base_hash, index);
    ingot_partner_data partners[2] = {{0}};
    uint8_t *bufs[2] = {NULL, NULL};
    int rc = n ? 0 : -INGOT_EFORMAT;
    for (uint32_t j = 0; !rc && j < n; j++) {
        const ingot_entry *pe = index[j] < i ? &r->entries[index[j]] : NULL;
        bufs[j] = pe ? malloc((size_t)(pe->source_size ? pe->source_size : 1)) : NULL;
        rc = !pe ? -INGOT_EFORMAT : bufs[j] ? 0 : -INGOT_ENOMEM;
        if (!rc)
            rc = ingot_reader_decode(r, index[j], base, threads, bufs[j]);
        partners[j] = (ingot_partner_data){pe, bufs[j]};
    }
    if (!rc)
        rc = ingot_tensor_decode_payload(e, r->map + e->payload_offset, e->payload_size, 1, NULL,
                                         partners, threads, out);
    free(bufs[0]);
    free(bufs[1]);
    return rc;
}

int ingot_reader_decode(
const ingot_reader *r, size_t i, ingot_base_set *base, int threads,
                        uint8_t *out) {
    if (!r || !out || i >= r->header.entry_count)
        return -INGOT_EINVAL;
    const ingot_entry *e = &r->entries[i];
    const uint8_t *payload = r->map + e->payload_offset;
    const uint8_t *base_bytes = NULL;
    int rc = is_delta(e->mode) ? load_base(e, base, &base_bytes) : 0;
    if (!rc && e->mode == INGOT_MODE_RAW)
        memcpy(out, payload, (size_t)e->source_size);
    else if (!rc && e->mode == INGOT_MODE_PREDICT)
        rc = decode_predicted(r, e, i, base, threads, out);
    else if (!rc)
        rc = ingot_tensor_decode_payload(e, payload, e->payload_size, r->header.version >= 2,
                                         base_bytes, NULL, threads, out);
    if (!rc && ingot_hash64(out, (size_t)e->source_size) != e->hash)
        rc = -INGOT_ECORRUPT;
    return rc;
}

/* macOS write(2) rejects counts above INT_MAX; larger tensors go in 1 GiB pieces. */
enum { MAX_WRITE = 1 << 30 };

static int write_all_at(int fd, const uint8_t *p, uint64_t n, uint64_t at) {
    while (n) {
        ssize_t w = pwrite(fd, p, (size_t)(n < MAX_WRITE ? n : MAX_WRITE), (off_t)at);
        if (w <= 0)
            return -INGOT_EIO;
        p += w;
        n -= (uint64_t)w;
        at += (uint64_t)w;
    }
    return 0;
}

int ingot_reader_rebuild(const ingot_reader *r, const char *dst_path, ingot_base_set *base,
                         int threads) {
    if (!r || !dst_path)
        return -INGOT_EINVAL;
    int fd = open(dst_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        return -INGOT_EIO;
    int rc = write_all_at(fd, r->map + r->header.source_header_offset,
                          r->header.source_header_size, 0);
    uint8_t *buf = NULL;
    uint64_t cap = 0;
    for (size_t i = 0; !rc && i < r->header.entry_count; i++) {
        const ingot_entry *e = &r->entries[i];
        if (e->source_size > cap) {
            free(buf);
            cap = e->source_size;
            buf = malloc((size_t)cap);
            rc = buf ? 0 : -INGOT_ENOMEM;
        }
        if (!rc)
            rc = ingot_reader_decode(r, i, base, threads, buf);
        if (!rc)
            rc = write_all_at(fd, buf, e->source_size, e->source_offset);
    }
    if (!rc && ftruncate(fd, (off_t)r->header.source_size))
        rc = -INGOT_EIO;
    free(buf);
    if (close(fd) && !rc)
        rc = -INGOT_EIO;
    if (rc)
        unlink(dst_path);
    return rc;
}
