/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_stream.c
 * @brief  Streaming .ingot decoder (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_stream.h"

#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>

#include "ingot/codec/ingot_predict.h"
#include "ingot/codec/ingot_tensor_codec.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"
#include "ingot/core/ingot_parallel.h"
#include "tt/convert/ft_sha256.h"

/* Blocks decoded per thread per batch, and the raw-copy chunk. */
enum { BATCH_PER_THREAD = 2, COPY_CHUNK = 1 << 20, MAX_WRITE = 1 << 30 };

/* An entry decoded ahead of its place in the output, or kept for the predict entries that
 * need it (version 3). */
typedef struct {
    uint8_t *data;       /* decoded bytes, or NULL */
    uint32_t dependents; /* later predict entries that still need it */
    int verified;        /* its hash checked: only then may a dependent use it */
    int emitted;
} held_entry;

typedef struct {
    ingot_input *in;
    ingot_write_fn write;
    void *write_ctx;
    const ingot_stream_options *opt;
    ingot_file_header header;
    ft_sha256_ctx sha;
    uint64_t out_pos;
    uint64_t coded_bytes;
    uint8_t *chunk;
    const ingot_entry *entries;
    held_entry *held; /* [entry_count] */
} stream_state;

static int emit(stream_state *s, const uint8_t *p, size_t n) {
    int rc = s->write(s->write_ctx, p, n);
    if (rc)
        return rc;
    ft_sha256_update(&s->sha, p, n);
    s->out_pos += n;
    if (s->opt->progress)
        s->opt->progress(s->opt->progress_ctx, s->out_pos, s->header.source_size);
    return 0;
}

static int skip_to(stream_state *s, uint64_t at) {
    uint64_t pos = ingot_input_pos(s->in);
    if (at < pos)
        return -INGOT_EFORMAT; /* not a streamable layout */
    return ingot_input_skip(s->in, at - pos);
}

/* Copy n input bytes to the output, or into dst when given; *hash = ingot_hash64 of them. */
static int copy_through(stream_state *s, uint64_t n, uint8_t *dst, uint64_t *hash) {
    ingot_hash64_state h;
    ingot_hash64_begin(&h, n);
    while (n) {
        size_t step = (size_t)(n < COPY_CHUNK ? n : COPY_CHUNK);
        uint8_t *p = dst ? dst : s->chunk;
        int rc = ingot_input_read(s->in, p, step);
        if (!rc && !dst)
            rc = emit(s, p, step);
        if (rc)
            return rc;
        ingot_hash64_update(&h, p, step);
        n -= step;
        dst = dst ? dst + step : NULL;
    }
    if (hash)
        *hash = ingot_hash64_end(&h);
    return 0;
}

static int read_header(stream_state *s) {
    uint8_t buf[INGOT_HEADER_BYTES];
    int rc = ingot_input_read(s->in, buf, 8);
    if (rc)
        return rc == -INGOT_ETRUNCATED ? -INGOT_EFORMAT : rc;
    rc = ingot_header_parse(buf, 8, &s->header);
    if (rc != -INGOT_ETRUNCATED)
        return rc ? rc : -INGOT_EFORMAT;
    if (s->header.version < 2)
        return -INGOT_EUNSUPPORTED; /* version 1 keeps its directory at the end */
    rc = ingot_input_read(s->in, buf + 8, INGOT_HEADER_BYTES - 8);
    if (!rc)
        rc = ingot_header_parse(buf, sizeof(buf), &s->header);
    if (!rc && s->header.source_header_offset != INGOT_HEADER_BYTES)
        rc = -INGOT_EFORMAT;
    return rc;
}

/* Base tensor of a delta entry: present, same size (and dtype unless block delta). */
static int find_base(stream_state *s, const ingot_entry *e, ingot_base_tensor *t) {
    if (!s->opt->base || ingot_base_set_find(s->opt->base, e->name, t) ||
        t->size != e->source_size ||
        (e->mode != INGOT_MODE_BLOCK_DELTA && t->dtype != e->dtype))
        return -INGOT_EBASE;
    return 0;
}

typedef struct {
    const ingot_tensor_decoder *dec;
    size_t first;
    uint8_t **in;
    ingot_block_view *views;
    uint8_t **base;
    uint8_t **out;
} batch_job;

static int batch_block(void *ctx, size_t i) {
    batch_job *job = ctx;
    return ingot_tensor_decoder_block(job->dec, job->first + i, &job->views[i],
                                      job->base ? job->base[i] : NULL, job->out[i]);
}

/* Per-entry streaming state for the blocks of one coded payload. */
typedef struct {
    const ingot_entry *e;
    size_t index;      /* directory index */
    uint8_t *dst;      /* decode into this buffer instead of the output, or NULL */
    ingot_tensor_decoder *dec;
    size_t blocks;
    uint8_t *offsets;
    ingot_base_tensor base;
    int has_base;
    ingot_hash64_state hash;
    ingot_hash64_state base_hash;
    size_t batch;
    uint8_t *bufs[3];  /* coded, base, out: batch slots each */
    uint64_t slot_bytes[3];
} coded_entry;

static int read_block_view(const uint8_t *p, uint64_t size, ingot_block_view *v) {
    ingot_block_area area;
    uint8_t offsets[16] = {0};
    ingot_store_u64(offsets + 8, size);
    area = (ingot_block_area){.blocks = p, .size = size, .block_count = 1, .offsets = offsets};
    return ingot_block_view_get(&area, 0, v);
}

/* Read the coded blocks [first, first + n) and their base ranges. */
static int read_batch(stream_state *s, coded_entry *c, size_t first, size_t n, uint8_t **in,
                      uint8_t **base, uint8_t **out, ingot_block_view *views) {
    int rc = 0;
    for (size_t i = 0; !rc && i < n; i++) {
        size_t b = first + i;
        uint64_t lo = 0, hi = 0;
        uint64_t size =
            ingot_load_u64(c->offsets + 8 * (b + 1)) - ingot_load_u64(c->offsets + 8 * b);
        ingot_tensor_decoder_range(c->dec, b, &lo, &hi);
        in[i] = c->bufs[0] + i * c->slot_bytes[0];
        base[i] = c->bufs[1] ? c->bufs[1] + i * c->slot_bytes[1] : NULL;
        out[i] = c->dst ? c->dst + lo : c->bufs[2] + i * c->slot_bytes[2];
        if (size > c->slot_bytes[0] || (!c->dst && hi - lo > c->slot_bytes[2]) ||
            hi > c->e->source_size)
            rc = -INGOT_EFORMAT;
        if (!rc)
            rc = ingot_input_read(s->in, in[i], (size_t)size);
        if (!rc)
            rc = read_block_view(in[i], size, &views[i]);
        if (!rc && c->has_base) {
            rc = ingot_base_set_read(s->opt->base, &c->base, lo, base[i], (size_t)(hi - lo));
            if (!rc)
                ingot_hash64_update(&c->base_hash, base[i], (size_t)(hi - lo));
        }
    }
    return rc;
}

/* Read, decode and emit (or store) blocks [first, first + n). */
static int run_batch(stream_state *s, coded_entry *c, size_t first, size_t n) {
    uint8_t *in[64], *base[64], *out[64];
    ingot_block_view views[64];
    int rc = read_batch(s, c, first, n, in, base, out, views);
    batch_job job = {c->dec, first, in, views, c->has_base ? base : NULL, out};
    if (!rc)
        rc = ingot_parallel_for(n, s->opt->threads, batch_block, &job);
    for (size_t i = 0; !rc && i < n; i++) {
        uint64_t lo = 0, hi = 0;
        ingot_tensor_decoder_range(c->dec, first + i, &lo, &hi);
        ingot_hash64_update(&c->hash, out[i], (size_t)(hi - lo));
        rc = c->dst ? 0 : emit(s, out[i], (size_t)(hi - lo));
    }
    return rc == -INGOT_EUNSUPPORTED ? -INGOT_EBASE : rc;
}

/* Offsets table checks; *max_block = the largest coded block. */
static int check_offsets(const uint8_t *offsets, size_t blocks, uint64_t *area,
                         uint64_t *max_block) {
    uint64_t prev = 0;
    *max_block = 0;
    if (ingot_load_u64(offsets) != 0)
        return -INGOT_EFORMAT;
    for (size_t b = 1; b <= blocks; b++) {
        uint64_t off = ingot_load_u64(offsets + 8 * b);
        if (off < prev + INGOT_BLOCK_HEADER_BYTES || off % 8)
            return -INGOT_EFORMAT;
        *max_block = off - prev > *max_block ? off - prev : *max_block;
        prev = off;
    }
    *area = prev;
    return 0;
}

static int alloc_slots(stream_state *s, coded_entry *c, uint64_t max_block) {
    uint64_t max_out = 0;
    for (size_t b = 0; b < c->blocks; b++) {
        uint64_t lo = 0, hi = 0;
        ingot_tensor_decoder_range(c->dec, b, &lo, &hi);
        max_out = hi - lo > max_out ? hi - lo : max_out;
    }
    size_t batch = (size_t)(s->opt->threads > 1 ? s->opt->threads : 1) * BATCH_PER_THREAD;
    c->batch = batch < 64 ? batch : 64;
    c->slot_bytes[0] = max_block;
    c->slot_bytes[1] = c->has_base ? (max_out + 7) & ~(uint64_t)7 : 0;
    c->slot_bytes[2] = c->dst ? 0 : (max_out + 7) & ~(uint64_t)7;
    for (int k = 0; k < 3; k++) {
        if (!c->slot_bytes[k])
            continue;
        c->bufs[k] = malloc((size_t)(c->slot_bytes[k] * c->batch));
        if (!c->bufs[k])
            return -INGOT_ENOMEM;
    }
    return 0;
}

/* A predict payload's partners: earlier entries, held in memory since they were decoded. */
static int bind_partners(stream_state *s, coded_entry *c) {
    uint32_t index[INGOT_PREDICT_MAX_PARTNERS];
    uint32_t n = ingot_tensor_decoder_partners(c->dec, index);
    int rc = 0;
    for (uint32_t j = 0; !rc && j < n; j++) {
        const held_entry *h = index[j] < c->index ? &s->held[index[j]] : NULL;
        rc = h && h->data && h->verified
                 ? ingot_tensor_decoder_bind(c->dec, j, &s->entries[index[j]], h->data)
                 : -INGOT_EFORMAT;
    }
    return rc;
}

/* Frame, mode prefix and offsets table of a coded payload; opens the decoder. */
static int open_coded(stream_state *s, coded_entry *c, uint8_t **prefix) {
    const ingot_entry *e = c->e;
    uint8_t frame[INGOT_PAYLOAD_FRAME_BYTES];
    int rc = ingot_input_read(s->in, frame, sizeof(frame));
    uint64_t prefix_bytes = rc ? 0 : ingot_load_u64(frame), blocks = ingot_load_u64(frame + 8);
    if (!rc && (prefix_bytes < sizeof(frame) || prefix_bytes > e->payload_size ||
                blocks > e->payload_size / INGOT_BLOCK_HEADER_BYTES))
        rc = -INGOT_EFORMAT;
    uint64_t mode_prefix = prefix_bytes - sizeof(frame), used = 0;
    if (!rc && !(*prefix = malloc((size_t)(mode_prefix ? mode_prefix : 1))))
        rc = -INGOT_ENOMEM;
    if (!rc)
        rc = ingot_input_read(s->in, *prefix, (size_t)mode_prefix);
    if (!rc)
        rc = ingot_tensor_decoder_open(e, *prefix, mode_prefix, &c->dec, &used);
    if (!rc && (used != mode_prefix || blocks != ingot_tensor_decoder_blocks(c->dec)))
        rc = -INGOT_EFORMAT;
    if (!rc)
        rc = bind_partners(s, c);
    c->blocks = (size_t)blocks;
    uint64_t table = 8 * (blocks + 1), area = 0, max_block = 0;
    if (!rc && !(c->offsets = malloc((size_t)table)))
        rc = -INGOT_ENOMEM;
    if (!rc)
        rc = ingot_input_read(s->in, c->offsets, (size_t)table);
    if (!rc)
        rc = check_offsets(c->offsets, c->blocks, &area, &max_block);
    if (!rc && prefix_bytes + table + area != e->payload_size)
        rc = -INGOT_EFORMAT;
    if (!rc && ingot_tensor_decoder_needs_base(c->dec)) {
        rc = find_base(s, e, &c->base);
        c->has_base = !rc;
    }
    return rc ? rc : alloc_slots(s, c, max_block);
}

/* Decode coded entry i to the output, or into dst (e->source_size bytes) when given. */
static int decode_coded(stream_state *s, size_t i, uint8_t *dst) {
    const ingot_entry *e = &s->entries[i];
    coded_entry c = {.e = e, .index = i, .dst = dst};
    uint8_t *prefix = NULL;
    int rc = open_coded(s, &c, &prefix);
    ingot_hash64_begin(&c.hash, e->source_size);
    ingot_hash64_begin(&c.base_hash, e->source_size);
    for (size_t first = 0; !rc && first < c.blocks; first += c.batch)
        rc = run_batch(s, &c, first, c.blocks - first < c.batch ? c.blocks - first : c.batch);
    if (!rc && c.has_base && ingot_hash64_end(&c.base_hash) != e->base_hash)
        rc = -INGOT_EBASE;
    if (!rc && ingot_hash64_end(&c.hash) != e->hash)
        rc = -INGOT_ECORRUPT;
    if (!rc)
        s->coded_bytes += e->source_size;
    ingot_tensor_decoder_close(c.dec);
    for (int k = 0; k < 3; k++)
        free(c.bufs[k]);
    free(c.offsets);
    free(prefix);
    return rc;
}

/* Directory, entry checks, and (version 3) what each predict entry will need held. */
static int read_directory(stream_state *s, ingot_entry **entries) {
    const ingot_file_header *h = &s->header;
    int rc = skip_to(s, h->dir_offset);
    uint8_t *dir = rc ? NULL : malloc((size_t)(h->dir_size ? h->dir_size : 1));
    if (!rc && !dir)
        rc = -INGOT_ENOMEM;
    if (!rc)
        rc = ingot_input_read(s->in, dir, (size_t)h->dir_size);
    if (!rc)
        rc = ingot_directory_parse(dir, h->dir_size, h->entry_count, entries);
    free(dir);
    if (!rc && !(s->held = calloc(h->entry_count ? h->entry_count : 1, sizeof(*s->held))))
        rc = -INGOT_ENOMEM;
    for (size_t i = 0; !rc && i < h->entry_count; i++) {
        const ingot_entry *e = &(*entries)[i];
        uint32_t index[2] = {0}, n = 0;
        rc = ingot_entry_check(e, h);
        if (!rc && e->mode == INGOT_MODE_PREDICT)
            n = ingot_partner_list(e->base_hash, index);
        for (uint32_t j = 0; !rc && j < n; j++) {
            rc = index[j] < i ? 0 : -INGOT_EFORMAT; /* partners come first */
            if (!rc)
                s->held[index[j]].dependents++;
        }
    }
    return rc;
}

/* The header names a base file: one must be given, and of the recorded size when known. */
static int check_base_present(stream_state *s) {
    uint64_t size = 0;
    if (!ingot_header_has_base(&s->header))
        return 0;
    if (!s->opt->base)
        return -INGOT_EBASE;
    if (ingot_base_set_single_size(s->opt->base, &size) == 0 && size && s->header.base_size &&
        size != s->header.base_size)
        return -INGOT_EBASE;
    return 0;
}

static int check_base_identity(stream_state *s) {
    uint64_t size = 0;
    uint8_t digest[32];
    if (!ingot_header_has_base(&s->header) ||
        ingot_base_set_single_size(s->opt->base, &size) != 0 ||
        (size && !s->opt->check_base_sha256))
        return 0;
    int rc = ingot_base_set_identity(s->opt->base, &size, digest);
    if (rc)
        return rc;
    return memcmp(digest, s->header.base_sha256, 32) ? -INGOT_EBASE : 0;
}

static void release_if_done(held_entry *h) {
    if (h->emitted && !h->dependents) {
        free(h->data);
        h->data = NULL;
    }
}

/* Emit held entries that now continue the output, in source order. */
static int flush_held(stream_state *s) {
    int found = 1, rc = 0;
    while (!rc && found) {
        found = 0;
        for (size_t i = 0; !rc && i < s->header.entry_count; i++) {
            held_entry *h = &s->held[i];
            if (!h->data || h->emitted || s->entries[i].source_offset != s->out_pos)
                continue;
            rc = emit(s, h->data, (size_t)s->entries[i].source_size);
            h->emitted = 1;
            release_if_done(h);
            found = 1;
        }
    }
    return rc;
}

/* Decode entry i: straight to the output when it continues it and nothing needs it later,
 * else into memory (version 3). */
static int decode_entry(stream_state *s, size_t i) {
    const ingot_entry *e = &s->entries[i];
    held_entry *h = &s->held[i];
    int direct = e->source_offset == s->out_pos && !h->dependents;
    uint64_t hash = 0;
    if (e->source_offset < s->out_pos || (!direct && s->header.version < 3))
        return -INGOT_EFORMAT; /* overlaps, or out of order in a version 2 file */
    int rc = skip_to(s, e->payload_offset);
    if (!rc && !direct && !(h->data = malloc((size_t)(e->source_size ? e->source_size : 1))))
        rc = -INGOT_ENOMEM;
    if (!rc && e->mode == INGOT_MODE_RAW) {
        rc = copy_through(s, e->payload_size, h->data, &hash);
        if (!rc && hash != e->hash)
            rc = -INGOT_ECORRUPT;
    } else if (!rc) {
        rc = decode_coded(s, i, h->data);
    }
    uint32_t index[2] = {0};
    uint32_t n = e->mode == INGOT_MODE_PREDICT ? ingot_partner_list(e->base_hash, index) : 0;
    for (uint32_t j = 0; !rc && j < n; j++) {
        s->held[index[j]].dependents--;
        release_if_done(&s->held[index[j]]);
    }
    h->verified = !rc;
    h->emitted = direct;
    return rc ? rc : flush_held(s);
}

static int decode_entries(stream_state *s) {
    int rc = 0;
    for (size_t i = 0; !rc && i < s->header.entry_count; i++)
        rc = decode_entry(s, i);
    for (size_t i = 0; !rc && i < s->header.entry_count; i++)
        if (!s->held[i].emitted)
            rc = -INGOT_EFORMAT; /* entries do not tile the source */
    return rc;
}

static int stream_run(stream_state *s, ingot_stream_result *result) {
    ingot_entry *entries = NULL;
    int rc = read_header(s);
    if (!rc)
        rc = check_base_present(s);
    if (!rc)
        rc = copy_through(s, s->header.source_header_size, NULL, NULL);
    if (!rc)
        rc = read_directory(s, &entries);
    s->entries = entries;
    if (!rc)
        rc = decode_entries(s);
    for (size_t i = 0; s->held && i < s->header.entry_count; i++)
        free(s->held[i].data);
    free(s->held);
    ingot_entries_free(entries, s->header.entry_count);
    if (!rc && s->out_pos != s->header.source_size)
        rc = -INGOT_EFORMAT;
    if (!rc)
        rc = skip_to(s, s->header.file_size);
    uint8_t digest[32];
    ft_sha256_final(&s->sha, digest);
    if (!rc && memcmp(digest, s->header.source_sha256, sizeof(digest)))
        rc = -INGOT_ECORRUPT;
    if (!rc)
        rc = check_base_identity(s);
    if (!rc && result) {
        *result = (ingot_stream_result){.kind = s->header.kind, .source_bytes = s->out_pos,
                                        .coded_bytes = s->coded_bytes};
        memcpy(result->sha256, digest, sizeof(digest));
    }
    return rc;
}

int ingot_stream_decode(ingot_input *in, ingot_write_fn write, void *write_ctx,
                        const ingot_stream_options *opt, ingot_stream_result *result) {
    if (!in || !write || ingot_input_pos(in))
        return -INGOT_EINVAL;
    ingot_stream_options defaults = {.threads = 1};
    stream_state s = {.in = in, .write = write, .write_ctx = write_ctx,
                      .opt = opt ? opt : &defaults};
    ft_sha256_init(&s.sha);
    s.chunk = malloc(COPY_CHUNK);
    int rc = s.chunk ? stream_run(&s, result) : -INGOT_ENOMEM;
    free(s.chunk);
    return rc;
}

static int write_fd(void *ctx, const void *buf, size_t n) {
    int fd = *(const int *)ctx;
    const uint8_t *p = buf;
    while (n) {
        ssize_t w = write(fd, p, n < MAX_WRITE ? n : MAX_WRITE);
        if (w <= 0)
            return -INGOT_EIO;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

int ingot_stream_decode_to_file(ingot_input *in, const char *dst_path,
                                const ingot_stream_options *opt, ingot_stream_result *result) {
    if (!in || !dst_path)
        return -INGOT_EINVAL;
    int fd = open(dst_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        return -INGOT_EIO;
    int rc = ingot_stream_decode(in, write_fd, &fd, opt, result);
    if (close(fd) && !rc)
        rc = -INGOT_EIO;
    if (rc)
        unlink(dst_path);
    return rc;
}
