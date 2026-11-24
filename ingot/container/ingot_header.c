/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_header.c
 * @brief  .ingot header and directory serialization (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_header.h"

#include <stdlib.h>
#include <string.h>

#include "ingot/codec/ingot_block.h"
#include "ingot/codec/ingot_qtensor.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_ggml.h"

size_t ingot_header_size(uint32_t version) {
    return version == 1 ? INGOT_HEADER_BYTES_V1
           : version >= 2 && version <= INGOT_VERSION ? INGOT_HEADER_BYTES : 0;
}

static int check_ranges(const ingot_file_header *h) {
    if (h->source_header_offset > h->file_size ||
        h->source_header_size > h->file_size - h->source_header_offset ||
        h->dir_offset > h->file_size || h->dir_size > h->file_size - h->dir_offset ||
        h->source_header_size > h->source_size)
        return -INGOT_EFORMAT;
    return 0;
}

int ingot_header_parse(const uint8_t *buf, size_t avail, ingot_file_header *h) {
    if (!buf || !h)
        return -INGOT_EINVAL;
    *h = (ingot_file_header){0};
    if (avail < 8)
        return -INGOT_ETRUNCATED;
    if (memcmp(buf, INGOT_MAGIC, INGOT_MAGIC_BYTES))
        return -INGOT_EFORMAT;
    h->version = ingot_load_u32(buf + 4);
    size_t need = ingot_header_size(h->version);
    if (!need)
        return -INGOT_EUNSUPPORTED;
    if (avail < need)
        return -INGOT_ETRUNCATED;
    h->entry_count = ingot_load_u32(buf + 8);
    uint32_t kind = ingot_load_u32(buf + 12);
    h->source_header_offset = ingot_load_u64(buf + 16);
    h->source_header_size = ingot_load_u64(buf + 24);
    h->dir_offset = ingot_load_u64(buf + 32);
    h->dir_size = ingot_load_u64(buf + 40);
    h->source_size = ingot_load_u64(buf + 48);
    h->file_size = ingot_load_u64(buf + 56);
    if (h->version == 1)
        return kind ? -INGOT_EFORMAT : check_ranges(h);
    if (kind != INGOT_SOURCE_SAFETENSORS && kind != INGOT_SOURCE_GGUF)
        return -INGOT_EFORMAT;
    h->kind = (ingot_source_kind)kind;
    memcpy(h->source_sha256, buf + 64, INGOT_SHA256_BYTES);
    memcpy(h->base_sha256, buf + 96, INGOT_SHA256_BYTES);
    h->base_size = ingot_load_u64(buf + 128);
    if (ingot_load_u64(buf + 136))
        return -INGOT_EFORMAT;
    return check_ranges(h);
}

void ingot_header_write(const ingot_file_header *h, uint8_t *out) {
    memset(out, 0, INGOT_HEADER_BYTES);
    memcpy(out, INGOT_MAGIC, INGOT_MAGIC_BYTES);
    ingot_store_u32(out + 4, h->version);
    ingot_store_u32(out + 8, h->entry_count);
    ingot_store_u32(out + 12, (uint32_t)h->kind);
    ingot_store_u64(out + 16, h->source_header_offset);
    ingot_store_u64(out + 24, h->source_header_size);
    ingot_store_u64(out + 32, h->dir_offset);
    ingot_store_u64(out + 40, h->dir_size);
    ingot_store_u64(out + 48, h->source_size);
    ingot_store_u64(out + 56, h->file_size);
    memcpy(out + 64, h->source_sha256, INGOT_SHA256_BYTES);
    memcpy(out + 96, h->base_sha256, INGOT_SHA256_BYTES);
    ingot_store_u64(out + 128, h->base_size); /* out + 136: reserved, zero */
}

int ingot_header_has_base(const ingot_file_header *h) {
    for (int i = 0; i < INGOT_SHA256_BYTES; i++)
        if (h->base_sha256[i])
            return 1;
    return 0;
}

size_t ingot_entry_record_size(const ingot_entry *e) {
    return INGOT_ENTRY_FIXED_BYTES + strlen(e->name) + 8 * (size_t)e->ndim;
}

size_t ingot_entry_write(const ingot_entry *e, uint8_t *out) {
    size_t name_len = strlen(e->name);
    out[0] = (uint8_t)name_len;
    out[1] = (uint8_t)(name_len >> 8);
    memcpy(out + 2, e->name, name_len);
    uint8_t *q = out + 2 + name_len;
    q[0] = (uint8_t)e->dtype;
    q[1] = (uint8_t)e->mode;
    q[2] = (uint8_t)e->ndim;
    q[3] = 0;
    q += 4;
    for (int d = 0; d < e->ndim; d++, q += 8)
        ingot_store_u64(q, e->shape[d]);
    const uint64_t tail[6] = {e->source_offset, e->source_size, e->payload_offset,
                              e->payload_size, e->hash, e->base_hash};
    for (int k = 0; k < 6; k++, q += 8)
        ingot_store_u64(q, tail[k]);
    return (size_t)(q - out);
}

static int parse_entry(const uint8_t *p, uint64_t avail, ingot_entry *e, uint64_t *used) {
    if (avail < 2)
        return -INGOT_EFORMAT;
    size_t name_len = (size_t)p[0] | ((size_t)p[1] << 8);
    if (avail < 2 + name_len + 4)
        return -INGOT_EFORMAT;
    const uint8_t *q = p + 2 + name_len;
    e->dtype = (ingot_dtype)q[0];
    e->mode = (ingot_mode)q[1];
    e->ndim = q[2];
    uint64_t need = INGOT_ENTRY_FIXED_BYTES + name_len + 8 * (uint64_t)e->ndim;
    if (e->ndim > INGOT_MAX_DIMS || q[3] != 0 || avail < need)
        return -INGOT_EFORMAT;
    e->name = calloc(1, name_len + 1);
    if (!e->name)
        return -INGOT_ENOMEM;
    memcpy(e->name, p + 2, name_len);
    q += 4;
    for (int d = 0; d < e->ndim; d++, q += 8)
        e->shape[d] = ingot_load_u64(q);
    e->source_offset = ingot_load_u64(q);
    e->source_size = ingot_load_u64(q + 8);
    e->payload_offset = ingot_load_u64(q + 16);
    e->payload_size = ingot_load_u64(q + 24);
    e->hash = ingot_load_u64(q + 32);
    e->base_hash = ingot_load_u64(q + 40);
    *used = need;
    return 0;
}

int ingot_directory_parse(const uint8_t *dir, uint64_t size, size_t count,
                          ingot_entry **entries) {
    uint64_t used = 0;
    int rc = ingot_directory_parse_used(dir, size, count, entries, &used);
    if (!rc && used != size) {
        ingot_entries_free(*entries, count);
        *entries = NULL;
        rc = -INGOT_EFORMAT;
    }
    return rc;
}

int ingot_directory_parse_used(const uint8_t *dir, uint64_t size, size_t count,
                               ingot_entry **entries, uint64_t *used_bytes) {
    if (!dir || !entries || !used_bytes)
        return -INGOT_EINVAL;
    ingot_entry *list = calloc(count ? count : 1, sizeof(*list));
    if (!list)
        return -INGOT_ENOMEM;
    uint64_t at = 0;
    int rc = 0;
    for (size_t i = 0; !rc && i < count; i++) {
        uint64_t used = 0;
        rc = parse_entry(dir + at, size - at, &list[i], &used);
        at += used;
    }
    if (rc) {
        ingot_entries_free(list, count);
        return rc;
    }
    *used_bytes = at;
    *entries = list;
    return 0;
}

void ingot_entries_free(ingot_entry *entries, size_t count) {
    for (size_t i = 0; entries && i < count; i++)
        free(entries[i].name);
    free(entries);
}

/* Is mode m valid for this entry's dtype? */
static int mode_fits(const ingot_entry *e, uint32_t version) {
    uint32_t type = 0, bw = 0, bb = 0, idx[2];
    switch (e->mode) {
    case INGOT_MODE_RAW:
        return e->payload_size == e->source_size;
    case INGOT_MODE_PLAIN:
    case INGOT_MODE_DELTA:
        return e->dtype == INGOT_DTYPE_BF16 && e->source_size % 2 == 0;
    case INGOT_MODE_QUANT:
    case INGOT_MODE_QUANT_DELTA:
        return version >= 2 && ingot_qtensor_supports(e->dtype);
    case INGOT_MODE_BLOCK_DELTA:
        return version >= 2 && ingot_dtype_ggml_type(e->dtype, &type) == 0 &&
               ingot_ggml_geometry(type, &bw, &bb) == 0;
    case INGOT_MODE_PREDICT:
        return version >= 3 && e->dtype == INGOT_DTYPE_BF16 && e->ndim == 2 && e->shape[0] &&
               e->shape[1] && e->shape[1] <= e->source_size / 2 / e->shape[0] &&
               e->shape[0] * e->shape[1] * 2 == e->source_size &&
               ingot_partner_list(e->base_hash, idx) > 0;
    case INGOT_MODE_MATCH:
        return version >= 4 && e->dtype == INGOT_DTYPE_BF16 && e->ndim == 2 && e->shape[0] &&
               e->shape[1] && e->shape[1] <= e->source_size / 2 / e->shape[0] &&
               e->shape[0] * e->shape[1] * 2 == e->source_size;
    default:
        return 0;
    }
}

int ingot_entry_check(const ingot_entry *e, const ingot_file_header *h) {
    if ((e->mode == INGOT_MODE_PREDICT && h->version == 2) ||
        (e->mode == INGOT_MODE_MATCH && h->version < 4))
        return -INGOT_EUNSUPPORTED; /* reserved in older versions */
    if (e->payload_offset % 8 || e->payload_offset > h->file_size ||
        e->payload_size > h->file_size - e->payload_offset ||
        e->source_offset < h->source_header_size || e->source_offset > h->source_size ||
        e->source_size > h->source_size - e->source_offset)
        return -INGOT_EFORMAT;
    return mode_fits(e, h->version) ? 0 : -INGOT_EFORMAT;
}
