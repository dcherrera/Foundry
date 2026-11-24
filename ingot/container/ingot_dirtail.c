/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_dirtail.c
 * @brief  Version 4 directory tail: refs and match source lists (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_dirtail.h"

#include <stdlib.h>
#include <string.h>

#include "ingot/codec/ingot_block.h"
#include "ingot/codec/ingot_match.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"

enum { MAX_LIST = 1u << 20 };

uint64_t ingot_source_list_hash(const uint32_t *list, uint32_t n) {
    return ingot_match_ref_hash(list, n);
}

size_t ingot_dir_tail_ref_size(const ingot_entry *ref) {
    return 2 + strlen(ref->name) + 4 + 8 * (size_t)ref->ndim + 16;
}

size_t ingot_dir_tail_list_size(uint32_t n) { return 8 + 4 * (size_t)n; }

int ingot_dir_tail_write(const ingot_entry *refs, uint32_t ref_count, const uint32_t *list_entry,
                         const uint32_t *list_n, const uint32_t *const *list_sources,
                         uint32_t list_count, uint8_t *out, size_t size) {
    size_t need = 8;
    for (uint32_t r = 0; r < ref_count; r++)
        need += ingot_dir_tail_ref_size(&refs[r]);
    for (uint32_t l = 0; l < list_count; l++)
        need += ingot_dir_tail_list_size(list_n[l]);
    if (need > size)
        return -INGOT_EINVAL;
    memset(out, 0, size);
    uint8_t *q = out;
    ingot_store_u32(q, ref_count);
    q += 4;
    for (uint32_t r = 0; r < ref_count; r++) {
        const ingot_entry *e = &refs[r];
        size_t n = strlen(e->name);
        q[0] = (uint8_t)n;
        q[1] = (uint8_t)(n >> 8);
        memcpy(q + 2, e->name, n);
        q += 2 + n;
        q[0] = (uint8_t)e->dtype;
        q[1] = (uint8_t)e->ndim;
        q += 4;
        for (int d = 0; d < e->ndim; d++, q += 8)
            ingot_store_u64(q, e->shape[d]);
        ingot_store_u64(q, e->source_size);
        ingot_store_u64(q + 8, e->hash);
        q += 16;
    }
    ingot_store_u32(q, list_count);
    q += 4;
    for (uint32_t l = 0; l < list_count; l++) {
        ingot_store_u32(q, list_entry[l]);
        ingot_store_u32(q + 4, list_n[l]);
        q += 8;
        for (uint32_t k = 0; k < list_n[l]; k++, q += 4)
            ingot_store_u32(q, list_sources[l][k]);
    }
    return 0;
}

static int parse_ref(const uint8_t *p, size_t avail, ingot_entry *e, size_t *used) {
    if (avail < 2)
        return -INGOT_EFORMAT;
    size_t n = (size_t)p[0] | ((size_t)p[1] << 8);
    if (avail < 2 + n + 4)
        return -INGOT_EFORMAT;
    const uint8_t *q = p + 2 + n;
    size_t need = 2 + n + 4 + 8 * (size_t)q[1] + 16;
    if (q[1] > INGOT_MAX_DIMS || q[2] || q[3] || avail < need)
        return -INGOT_EFORMAT;
    e->name = calloc(1, n + 1);
    if (!e->name)
        return -INGOT_ENOMEM;
    memcpy(e->name, p + 2, n);
    e->dtype = (ingot_dtype)q[0];
    e->ndim = q[1];
    q += 4;
    for (int d = 0; d < e->ndim; d++, q += 8)
        e->shape[d] = ingot_load_u64(q);
    e->source_size = ingot_load_u64(q);
    e->hash = ingot_load_u64(q + 8);
    *used = need;
    return 0;
}

static int parse_refs(const uint8_t *tail, size_t size, size_t *at, ingot_dir_tail *t) {
    if (size < 4)
        return -INGOT_EFORMAT;
    t->ref_count = ingot_load_u32(tail);
    *at = 4;
    if (t->ref_count > size / 22)
        return -INGOT_EFORMAT;
    t->refs = calloc(t->ref_count ? t->ref_count : 1, sizeof(*t->refs));
    if (!t->refs)
        return -INGOT_ENOMEM;
    for (uint32_t r = 0; r < t->ref_count; r++) {
        size_t used = 0;
        int rc = parse_ref(tail + *at, size - *at, &t->refs[r], &used);
        if (rc)
            return rc;
        *at += used;
    }
    return 0;
}

/* One source of entry i: earlier entry, ref or self. */
static int source_ok(uint32_t s, size_t i, size_t entry_count, uint32_t ref_count) {
    return s == INGOT_MATCH_SELF_REF || s < i ||
           (s >= entry_count && s - entry_count < ref_count);
}

static int parse_lists(const uint8_t *tail, size_t size, size_t at, const ingot_entry *entries,
                       size_t entry_count, ingot_dir_tail *t) {
    if (size - at < 4)
        return -INGOT_EFORMAT;
    uint32_t lists = ingot_load_u32(tail + at);
    at += 4;
    t->list_start = malloc((entry_count ? entry_count : 1) * sizeof(uint32_t));
    t->list_count = calloc(entry_count ? entry_count : 1, sizeof(uint32_t));
    t->sources = malloc((size - at) / 4 * sizeof(uint32_t) + sizeof(uint32_t));
    if (!t->list_start || !t->list_count || !t->sources)
        return -INGOT_ENOMEM;
    memset(t->list_start, 0xFF, (entry_count ? entry_count : 1) * sizeof(uint32_t));
    for (uint32_t l = 0; l < lists; l++) {
        if (size - at < 8)
            return -INGOT_EFORMAT;
        uint32_t i = ingot_load_u32(tail + at), n = ingot_load_u32(tail + at + 4);
        at += 8;
        if (i >= entry_count || entries[i].mode != INGOT_MODE_MATCH ||
            t->list_start[i] != UINT32_MAX || n > MAX_LIST || (size - at) / 4 < n)
            return -INGOT_EFORMAT;
        t->list_start[i] = (uint32_t)t->source_total;
        t->list_count[i] = n;
        for (uint32_t k = 0; k < n; k++, at += 4) {
            uint32_t s = ingot_load_u32(tail + at);
            if (!source_ok(s, i, entry_count, t->ref_count))
                return -INGOT_EFORMAT;
            t->sources[t->source_total++] = s;
        }
        if (ingot_source_list_hash(t->sources + t->list_start[i], n) != entries[i].base_hash)
            return -INGOT_EFORMAT;
    }
    for (; at < size; at++)
        if (tail[at])
            return -INGOT_EFORMAT;
    for (size_t i = 0; i < entry_count; i++)
        if (entries[i].mode == INGOT_MODE_MATCH && t->list_start[i] == UINT32_MAX)
            return -INGOT_EFORMAT;
    return 0;
}

int ingot_dir_tail_parse(const uint8_t *tail, size_t size, const ingot_entry *entries,
                         size_t entry_count, ingot_dir_tail *out) {
    *out = (ingot_dir_tail){0};
    size_t at = 0;
    int rc = parse_refs(tail, size, &at, out);
    if (!rc)
        rc = parse_lists(tail, size, at, entries, entry_count, out);
    if (rc)
        ingot_dir_tail_release(out);
    return rc;
}

void ingot_dir_tail_release(ingot_dir_tail *t) {
    for (uint32_t r = 0; t->refs && r < t->ref_count; r++)
        free(t->refs[r].name);
    free(t->refs);
    free(t->list_start);
    free(t->list_count);
    free(t->sources);
    *t = (ingot_dir_tail){0};
}

uint32_t ingot_dir_tail_sources(const ingot_dir_tail *t, size_t i, const uint32_t **list) {
    if (!t->list_start || t->list_start[i] == UINT32_MAX)
        return 0;
    *list = t->sources + t->list_start[i];
    return t->list_count[i];
}
