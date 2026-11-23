/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_block.c
 * @brief  Block framing, sink and block-area driver (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_block.h"

#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_parallel.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "ingot reads rANS words in place and needs a little-endian host"
#endif

/* Blocks encoded per thread per batch: enough to keep threads busy, small enough for memory. */
enum { BATCH_PER_THREAD = 4 };

/* macOS write(2) rejects counts above INT_MAX; larger buffers go in 1 GiB pieces. */
enum { MAX_WRITE = 1 << 30 };

uint32_t ingot_load_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

uint64_t ingot_load_u64(const uint8_t *p) {
    return (uint64_t)ingot_load_u32(p) | ((uint64_t)ingot_load_u32(p + 4) << 32);
}

void ingot_store_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++)
        p[i] = (uint8_t)(v >> (8 * i));
}

void ingot_store_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++)
        p[i] = (uint8_t)(v >> (8 * i));
}

static int sink_grow(ingot_sink *s, uint64_t need) {
    if (need <= s->mem_capacity)
        return 0;
    size_t cap = s->mem_capacity ? s->mem_capacity : 4096;
    while (cap < need)
        cap *= 2;
    uint8_t *mem = realloc(s->mem, cap);
    if (!mem)
        return -INGOT_ENOMEM;
    s->mem = mem;
    s->mem_capacity = cap;
    return 0;
}

static int sink_pwrite_all(int fd, const uint8_t *p, size_t n, uint64_t at) {
    while (n) {
        ssize_t w = pwrite(fd, p, n < MAX_WRITE ? n : MAX_WRITE, (off_t)at);
        if (w <= 0)
            return -INGOT_EIO;
        p += w;
        n -= (size_t)w;
        at += (uint64_t)w;
    }
    return 0;
}

int ingot_sink_write(ingot_sink *s, const void *data, size_t n) {
    if (!s || (!data && n))
        return -INGOT_EINVAL;
    if (!n)
        return 0;
    int rc = 0;
    if (s->fd >= 0) {
        rc = sink_pwrite_all(s->fd, data, n, s->pos);
    } else if (s->fd != INGOT_SINK_DISCARD) {
        rc = sink_grow(s, s->pos + n);
        if (!rc)
            memcpy(s->mem + s->pos, data, n);
    }
    if (!rc)
        s->pos += n;
    return rc;
}

int ingot_sink_write_at(ingot_sink *s, uint64_t at, const void *data, size_t n) {
    if (!s || !data || at + n > s->pos)
        return -INGOT_EINVAL;
    if (s->fd == INGOT_SINK_DISCARD)
        return 0;
    if (s->fd >= 0)
        return sink_pwrite_all(s->fd, data, n, at);
    memcpy(s->mem + at, data, n);
    return 0;
}

int ingot_sink_pad(ingot_sink *s, unsigned align) {
    static const uint8_t zeros[16] = {0};
    size_t pad = (size_t)((align - s->pos % align) % align);
    return pad ? ingot_sink_write(s, zeros, pad) : 0;
}

int ingot_block_serialize(const uint32_t *states, const uint8_t *raw, size_t raw_bytes,
                          const uint16_t *words, size_t word_count, ingot_block_buf *out) {
    size_t raw_padded = (raw_bytes + 1) & ~(size_t)1;
    size_t size = INGOT_BLOCK_HEADER_BYTES + raw_padded + 2 * word_count;
    size = (size + 7) & ~(size_t)7;
    uint8_t *p = calloc(1, size);
    if (!p)
        return -INGOT_ENOMEM;
    ingot_store_u32(p, (uint32_t)raw_bytes);
    ingot_store_u32(p + 4, (uint32_t)word_count);
    for (int l = 0; l < INGOT_RANS_LANES; l++)
        ingot_store_u32(p + 8 + 4 * l, states[l]);
    if (raw_bytes)
        memcpy(p + INGOT_BLOCK_HEADER_BYTES, raw, raw_bytes);
    if (word_count)
        memcpy(p + INGOT_BLOCK_HEADER_BYTES + raw_padded, words, 2 * word_count);
    out->data = p;
    out->size = size;
    return 0;
}

typedef struct {
    ingot_block_encode_fn fn;
    void *ctx;
    size_t first;
    ingot_block_buf *bufs;
} batch_job;

static int batch_item(void *ctx, size_t i) {
    batch_job *job = ctx;
    return job->fn(job->ctx, job->first + i, &job->bufs[i]);
}

static int write_batch(ingot_sink *s, batch_job *job, size_t count, uint64_t area_start,
                       uint8_t *offsets) {
    int rc = 0;
    for (size_t i = 0; i < count; i++) {
        if (!rc) {
            ingot_store_u64(offsets + 8 * (job->first + i), s->pos - area_start);
            rc = ingot_sink_write(s, job->bufs[i].data, job->bufs[i].size);
        }
        free(job->bufs[i].data);
        job->bufs[i] = (ingot_block_buf){0};
    }
    return rc;
}

int ingot_block_area_write(ingot_sink *s, size_t block_count, int threads,
                           ingot_block_encode_fn fn, void *ctx) {
    if (!s || !fn)
        return -INGOT_EINVAL;
    size_t table_bytes = 8 * (block_count + 1);
    size_t batch = (size_t)(threads > 1 ? threads : 1) * BATCH_PER_THREAD;
    uint8_t *offsets = calloc(1, table_bytes);
    ingot_block_buf *bufs = calloc(batch, sizeof(*bufs));
    int rc = (offsets && bufs) ? 0 : -INGOT_ENOMEM;
    uint64_t table_at = s->pos;
    if (!rc)
        rc = ingot_sink_write(s, offsets, table_bytes);
    uint64_t area_start = s->pos;
    for (size_t first = 0; !rc && first < block_count; first += batch) {
        size_t count = block_count - first < batch ? block_count - first : batch;
        batch_job job = {fn, ctx, first, bufs};
        rc = ingot_parallel_for(count, threads, batch_item, &job);
        int wrc = write_batch(s, &job, count, area_start, offsets);
        rc = rc ? rc : wrc;
    }
    if (!rc) {
        ingot_store_u64(offsets + 8 * block_count, s->pos - area_start);
        rc = ingot_sink_write_at(s, table_at, offsets, table_bytes);
    }
    free(bufs);
    free(offsets);
    return rc;
}

int ingot_block_area_open(const uint8_t *area, uint64_t avail, size_t block_count,
                          ingot_block_area *out) {
    uint64_t table_bytes = 8 * ((uint64_t)block_count + 1);
    if (!area || !out || avail < table_bytes)
        return -INGOT_EFORMAT;
    out->offsets = area;
    out->blocks = area + table_bytes;
    out->block_count = block_count;
    out->size = ingot_load_u64(area + 8 * block_count);
    if (out->size > avail - table_bytes || ingot_load_u64(area) != 0)
        return -INGOT_EFORMAT;
    uint64_t prev = 0;
    for (size_t b = 1; b <= block_count; b++) {
        uint64_t off = ingot_load_u64(area + 8 * b);
        if (off < prev + INGOT_BLOCK_HEADER_BYTES || off % 8)
            return -INGOT_EFORMAT;
        prev = off;
    }
    return 0;
}

int ingot_block_view_get(const ingot_block_area *a, size_t index, ingot_block_view *v) {
    if (index >= a->block_count)
        return -INGOT_EFORMAT;
    uint64_t lo = ingot_load_u64(a->offsets + 8 * index);
    uint64_t hi = ingot_load_u64(a->offsets + 8 * (index + 1));
    const uint8_t *p = a->blocks + lo;
    uint64_t size = hi - lo;
    v->raw_bytes = ingot_load_u32(p);
    v->word_count = ingot_load_u32(p + 4);
    uint64_t raw_padded = ((uint64_t)v->raw_bytes + 1) & ~(uint64_t)1;
    if (INGOT_BLOCK_HEADER_BYTES + raw_padded + 2 * (uint64_t)v->word_count > size)
        return -INGOT_EFORMAT;
    for (int l = 0; l < INGOT_RANS_LANES; l++)
        v->states[l] = ingot_load_u32(p + 8 + 4 * l);
    v->raw = p + INGOT_BLOCK_HEADER_BYTES;
    v->words = (const uint16_t *)(const void *)(p + INGOT_BLOCK_HEADER_BYTES + raw_padded);
    if ((uintptr_t)v->words % 2)
        return -INGOT_EFORMAT;
    return 0;
}

int ingot_block_check_end(const uint32_t *x, size_t words_read, const ingot_block_view *v) {
    if (words_read != v->word_count)
        return -INGOT_ECORRUPT;
    for (int l = 0; l < INGOT_RANS_LANES; l++)
        if (x[l] != INGOT_RANS_LOW)
            return -INGOT_ECORRUPT;
    return 0;
}
