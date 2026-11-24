/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_writer.c
 * @brief  Streaming safetensors / GGUF -> .ingot encoder (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_writer.h"

#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>

#include "ingot/codec/ingot_block.h"
#include "ingot/codec/ingot_qtensor.h"
#include "ingot/container/ingot_header.h"
#include "ingot/container/ingot_layout.h"
#include "ingot/container/ingot_predict_rules.h"
#include "ingot/container/ingot_writer_match.h"
#include "ingot/container/ingot_writer_tensor.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"

enum { COPY_CHUNK = 1 << 20 };

/* Scheduling state of one entry (source order). */
typedef struct {
    int has_rule;               /* a dependent: predicted from partners (version 3 rules) */
    int has_match;              /* a match target: waits for its pool entries in this file */
    ingot_predict_geometry geom;
    size_t partner[INGOT_PREDICT_MAX_PARTNERS]; /* entry indices */
    uint32_t dependents;        /* dependents not yet written */
    uint8_t *data;              /* held source bytes (pending, or a partner), else NULL */
    int pending;                /* read, waiting for its partners */
    int emitted;
    uint32_t position;          /* directory index once written */
} entry_plan;

typedef struct {
    ingot_entry *items;
    entry_plan *plans;
    size_t count;
    size_t capacity;
    int open_tail;      /* the last entry is a filler whose size is known only at the end */
    size_t *order;      /* directory position -> entry index */
    size_t emitted;
    size_t pending;
    size_t *tensor_entry; /* layout tensor index -> entry index */
    int64_t v2_saved; /* padded payload bytes predict saved over the best other modes */
    ingot_match_writer *match; /* match mode (version 4), or NULL */
    size_t tail_bytes;         /* reserved for the version 4 directory tail */
} entry_list;

static ingot_entry *entry_push(entry_list *l) {
    if (l->count == l->capacity) {
        size_t cap = l->capacity ? l->capacity * 2 : 64;
        ingot_entry *items = realloc(l->items, cap * sizeof(*items));
        if (!items)
            return NULL;
        l->items = items;
        l->capacity = cap;
    }
    ingot_entry *e = &l->items[l->count++];
    *e = (ingot_entry){0};
    return e;
}

static int push_filler(entry_list *l, uint64_t offset, uint64_t size) {
    ingot_entry *e = entry_push(l);
    if (!e || !(e->name = calloc(1, 1)))
        return -INGOT_ENOMEM;
    e->ndim = 1;
    e->shape[0] = size;
    e->source_offset = offset;
    e->source_size = size;
    return 0;
}

/* Tensors in file order, fillers for the gaps, and a tail filler for trailing bytes (sized at the
 * end when the source size is not known up front). */
static int collect_entries(const ingot_layout *layout, const ingot_input *src, entry_list *l) {
    uint64_t pos = 0;
    ingot_layout_header(layout, &pos);
    size_t tensors = ingot_layout_count(layout);
    l->tensor_entry = calloc(tensors ? tensors : 1, sizeof(*l->tensor_entry));
    if (!l->tensor_entry)
        return -INGOT_ENOMEM;
    for (size_t i = 0; i < tensors; i++) {
        const ingot_layout_tensor *t = ingot_layout_tensor_at(layout, i);
        if (strlen(t->name) > 0xFFFF)
            return -INGOT_EFORMAT;
        if (t->offset > pos && push_filler(l, pos, t->offset - pos))
            return -INGOT_ENOMEM;
        l->tensor_entry[i] = l->count;
        ingot_entry *e = entry_push(l);
        if (!e || !(e->name = strdup(t->name)))
            return -INGOT_ENOMEM;
        e->dtype = t->dtype;
        e->ndim = t->ndim;
        memcpy(e->shape, t->shape, sizeof(e->shape));
        e->source_offset = t->offset;
        e->source_size = t->size;
        pos = t->offset + t->size;
    }
    if (!ingot_input_seekable(src)) {
        l->open_tail = 1;
        return push_filler(l, pos, 0);
    }
    if (ingot_input_size(src) < pos)
        return -INGOT_ETRUNCATED;
    return ingot_input_size(src) > pos ? push_filler(l, pos, ingot_input_size(src) - pos) : 0;
}

/* Scheduling state; with prediction on, every dependent and the partners it waits for. */
static int plan_entries(const ingot_encode_options *opt, const ingot_layout *layout,
                        entry_list *l) {
    l->plans = calloc(l->count ? l->count : 1, sizeof(*l->plans));
    l->order = calloc(l->count ? l->count : 1, sizeof(*l->order));
    if (!l->plans || !l->order)
        return -INGOT_ENOMEM;
    int predict = !opt->no_predict && opt->policy != INGOT_POLICY_PLAIN;
    if (predict && opt->max_version != 3) {
        int rc = ingot_match_writer_create(opt, layout, l->tensor_entry, l->count, &l->match);
        for (size_t i = 0; !rc && i < l->count; i++) {
            const size_t *deps = NULL;
            uint32_t n = ingot_match_writer_deps(l->match, i, &deps);
            l->plans[i].has_match = ingot_match_writer_is_target(l->match, i);
            for (uint32_t k = 0; k < n; k++)
                l->plans[deps[k]].dependents++;
        }
        l->tail_bytes = rc ? 0 : ingot_match_writer_tail_reserve(l->match);
        return rc;
    }
    for (size_t t = 0; predict && t < ingot_layout_count(layout); t++) {
        size_t partner[INGOT_PREDICT_MAX_PARTNERS] = {0};
        entry_plan *p = &l->plans[l->tensor_entry[t]];
        if (ingot_predict_rule(layout, t, &p->geom, partner) != 1)
            continue;
        p->has_rule = 1;
        for (uint32_t j = 0; j < p->geom.partner_count; j++) {
            p->partner[j] = l->tensor_entry[partner[j]];
            l->plans[p->partner[j]].dependents++;
        }
    }
    return 0;
}

/* Will this entry be buffered and coded (vs copied through)? */
static int codable(const ingot_encode_options *opt, const ingot_entry *e) {
    if (!e->name[0] || !e->source_size)
        return 0;
    if (e->dtype == INGOT_DTYPE_BF16 || ingot_qtensor_supports(e->dtype))
        return 1;
    ingot_base_tensor t;
    return opt->base && opt->policy != INGOT_POLICY_PLAIN &&
           ingot_base_set_find(opt->base, e->name, &t) == 0;
}

typedef struct {
    ingot_input *src;
    ingot_sink sink;
    uint8_t *buf;
    uint64_t buf_capacity;
} encode_io;

static int read_tensor(encode_io *io, uint64_t size) {
    if (size > io->buf_capacity) {
        free(io->buf);
        io->buf = malloc((size_t)size);
        io->buf_capacity = io->buf ? size : 0;
        if (!io->buf)
            return -INGOT_ENOMEM;
    }
    return ingot_input_read(io->src, io->buf, (size_t)size);
}

/* Copy an entry through raw in chunks and hash it. */
static int copy_raw(encode_io *io, ingot_entry *e) {
    uint8_t *chunk = malloc(COPY_CHUNK);
    if (!chunk)
        return -INGOT_ENOMEM;
    ingot_hash64_state h;
    ingot_hash64_begin(&h, e->source_size);
    int rc = 0;
    for (uint64_t left = e->source_size; !rc && left;) {
        size_t n = (size_t)(left < COPY_CHUNK ? left : COPY_CHUNK);
        rc = ingot_input_read(io->src, chunk, n);
        if (!rc)
            rc = ingot_sink_write(&io->sink, chunk, n);
        ingot_hash64_update(&h, chunk, n);
        left -= n;
    }
    e->hash = ingot_hash64_end(&h);
    free(chunk);
    return rc;
}

/* The bytes after the last tensor of a source read from a pipe (size known only at its end;
 * in practice a few alignment bytes): read to the end, then hash and write. */
static int copy_tail(encode_io *io, ingot_entry *e) {
    size_t cap = 4096, used = 0;
    uint8_t *tail = malloc(cap);
    int rc = tail ? 0 : -INGOT_ENOMEM;
    while (!rc) {
        if (used == cap) {
            uint8_t *grown = realloc(tail, cap * 2);
            if (!grown) {
                rc = -INGOT_ENOMEM;
                break;
            }
            tail = grown;
            cap *= 2;
        }
        uint64_t before = ingot_input_pos(io->src);
        rc = ingot_input_read(io->src, tail + used, cap - used);
        used += (size_t)(ingot_input_pos(io->src) - before);
        if (rc == -INGOT_ETRUNCATED) {
            rc = 0;
            break;
        }
    }
    if (!rc) {
        e->source_size = used;
        e->shape[0] = used;
        e->hash = ingot_hash64(tail, used);
        rc = ingot_sink_write(&io->sink, tail, used);
    }
    free(tail);
    return rc;
}

/* All partners of entry i written? */
static int ready(const entry_list *l, size_t i) {
    const entry_plan *p = &l->plans[i];
    for (uint32_t j = 0; p->has_rule && j < p->geom.partner_count; j++)
        if (!l->plans[p->partner[j]].emitted)
            return 0;
    const size_t *deps = NULL;
    uint32_t n = p->has_match ? ingot_match_writer_deps(l->match, i, &deps) : 0;
    for (uint32_t k = 0; k < n; k++)
        if (!l->plans[deps[k]].emitted)
            return 0;
    return 1;
}

/* Free a held entry's bytes (and its match caches). */
static void drop_data(entry_list *l, size_t i) {
    free(l->plans[i].data);
    l->plans[i].data = NULL;
    ingot_match_writer_release_entry(l->match, i);
}

/* Entry i's payload is written: give it the next directory position, report it, release
 * partners whose last dependent this was. */
static int finish_emit(const ingot_encode_options *opt, encode_io *io, entry_list *l, size_t i,
                       const ingot_writer_coded *coded) {
    entry_plan *p = &l->plans[i];
    ingot_entry *e = &l->items[i];
    e->payload_size = io->sink.pos - e->payload_offset;
    uint64_t v2 = coded ? coded->v2_bytes : e->payload_size;
    l->v2_saved += (int64_t)((v2 + 7) & ~(uint64_t)7) -
                   (int64_t)((e->payload_size + 7) & ~(uint64_t)7);
    if (opt->entry_done)
        opt->entry_done(opt->entry_ctx, e, v2, coded ? coded->prefix_bytes : 0);
    p->emitted = 1;
    p->position = (uint32_t)l->emitted;
    l->order[l->emitted++] = i;
    for (uint32_t j = 0; p->has_rule && j < p->geom.partner_count; j++) {
        entry_plan *q = &l->plans[p->partner[j]];
        if (--q->dependents == 0)
            drop_data(l, p->partner[j]);
    }
    int rc = 0;
    if (p->has_match) {
        const size_t *deps = NULL;
        uint32_t n = ingot_match_writer_deps(l->match, i, &deps);
        for (uint32_t k = 0; k < n; k++)
            if (--l->plans[deps[k]].dependents == 0)
                drop_data(l, deps[k]);
        rc = ingot_match_writer_note_used(l->match, i, coded ? coded->match_refs : NULL,
                                          coded && e->mode == INGOT_MODE_MATCH
                                              ? coded->match_ref_count
                                              : 0);
    }
    if (!p->dependents)
        drop_data(l, i);
    return rc ? rc : ingot_sink_pad(&io->sink, 8);
}

/* A match target's sources and candidates: every held entry's bytes and position. */
static int prepare_match(entry_list *l, size_t i, const uint8_t *data, ingot_writer_match *m) {
    const uint8_t **held = calloc(l->count ? l->count : 1, sizeof(*held));
    uint32_t *position = calloc(l->count ? l->count : 1, sizeof(*position));
    int rc = held && position ? 0 : -INGOT_ENOMEM;
    for (size_t j = 0; !rc && j < l->count; j++) {
        held[j] = l->plans[j].data;
        position[j] = l->plans[j].position;
    }
    if (!rc)
        rc = ingot_match_writer_prepare(l->match, i, data, l->items,
                                        (const uint8_t *const *)held, position, m);
    free(held);
    free(position);
    return rc;
}

/* Code a buffered tensor (with its partners when it is a dependent) and write the payload. */
static int emit_tensor(const ingot_encode_options *opt, encode_io *io, entry_list *l, size_t i,
                       const uint8_t *data) {
    const entry_plan *p = &l->plans[i];
    ingot_entry *e = &l->items[i];
    ingot_writer_partners partners = {.geom = p->geom};
    for (uint32_t j = 0; p->has_rule && j < p->geom.partner_count; j++) {
        const entry_plan *q = &l->plans[p->partner[j]];
        partners.geom.partner[j].entry = q->position;
        partners.entry[j] = &l->items[p->partner[j]];
        partners.data[j] = q->data;
    }
    ingot_sink payload = {.fd = INGOT_SINK_MEMORY};
    ingot_writer_coded coded;
    ingot_writer_match match = {0};
    e->payload_offset = io->sink.pos;
    e->hash = ingot_hash64(data, (size_t)e->source_size);
    int rc = p->has_match ? prepare_match(l, i, data, &match) : 0;
    partners.match = p->has_match ? &match : NULL;
    if (!rc)
        rc = ingot_writer_code_tensor(opt, e, data,
                                      p->has_rule || p->has_match ? &partners : NULL, &payload,
                                      &coded);
    ingot_match_writer_done(&match);
    if (!rc)
        rc = ingot_sink_write(&io->sink, payload.mem, (size_t)payload.pos);
    free(payload.mem);
    return rc ? rc : finish_emit(opt, io, l, i, &coded);
}

/* Write every pending dependent whose partners are now written (repeat: chains). */
static int drain_pending(const ingot_encode_options *opt, encode_io *io, entry_list *l) {
    int progress = 1, rc = 0;
    while (!rc && progress && l->pending) {
        progress = 0;
        for (size_t i = 0; !rc && i < l->count; i++) {
            entry_plan *p = &l->plans[i];
            if (!p->pending || !ready(l, i))
                continue;
            p->pending = 0;
            l->pending--;
            progress = 1;
            rc = emit_tensor(opt, io, l, i, p->data);
        }
    }
    return rc;
}

/* Read entry i: copy it through, code it now, or hold it (a dependent waiting for partners, or
 * a partner some dependent will need). */
static int process_entry(const ingot_encode_options *opt, encode_io *io, entry_list *l,
                         size_t i) {
    entry_plan *p = &l->plans[i];
    ingot_entry *e = &l->items[i];
    int rc = 0;
    if ((l->open_tail && i + 1 == l->count) || !codable(opt, e)) {
        e->mode = INGOT_MODE_RAW;
        e->payload_offset = io->sink.pos;
        rc = l->open_tail && i + 1 == l->count ? copy_tail(io, e) : copy_raw(io, e);
        return rc ? rc : finish_emit(opt, io, l, i, NULL);
    }
    if (!p->dependents && ready(l, i)) {
        rc = read_tensor(io, e->source_size);
        return rc ? rc : emit_tensor(opt, io, l, i, io->buf);
    }
    p->data = malloc((size_t)e->source_size);
    if (!p->data)
        return -INGOT_ENOMEM;
    rc = ingot_input_read(io->src, p->data, (size_t)e->source_size);
    if (!rc && ready(l, i))
        return emit_tensor(opt, io, l, i, p->data);
    p->pending = 1;
    l->pending++;
    return rc;
}

static int write_directory(ingot_sink *s, const entry_list *l, uint64_t at, uint64_t size) {
    uint8_t *dir = malloc((size_t)(size ? size : 1));
    if (!dir)
        return -INGOT_ENOMEM;
    uint64_t used = 0;
    for (size_t k = 0; k < l->count; k++)
        used += ingot_entry_write(&l->items[l->order[k]], dir + used);
    int rc = used + l->tail_bytes == size ? 0 : -INGOT_EFORMAT;
    uint32_t *position =
        l->tail_bytes ? calloc(l->count ? l->count : 1, sizeof(uint32_t)) : NULL;
    if (!rc && l->tail_bytes && !position)
        rc = -INGOT_ENOMEM;
    for (size_t k = 0; !rc && position && k < l->count; k++)
        position[k] = l->plans[k].position;
    if (!rc && l->tail_bytes)
        rc = ingot_match_writer_tail_write(l->match, position, dir + used, l->tail_bytes);
    free(position);
    if (!rc)
        rc = ingot_sink_write_at(s, at, dir, (size_t)size);
    free(dir);
    return rc;
}

static int uses_base(const entry_list *l) {
    for (size_t i = 0; i < l->count; i++)
        if (l->items[i].base_hash && l->items[i].mode != INGOT_MODE_PREDICT &&
            l->items[i].mode != INGOT_MODE_MATCH)
            return 1;
    return 0;
}

/* The oldest version that holds the file: 5 with a dictionary ref, 4 with match targets (the
 * directory has a tail), 3 with a predicted tensor or payloads out of source order. */
static uint32_t file_version(const entry_list *l) {
    if (l->tail_bytes)
        return ingot_match_writer_uses_dict(l->match) ? 5 : 4;
    for (size_t k = 0; k < l->count; k++)
        if (l->order[k] != k || l->items[k].mode == INGOT_MODE_PREDICT)
            return 3;
    return 2;
}

/* Source digest (the input must be fully consumed) and, for a delta file over one model file,
 * the base identity. */
static int finish_identity(const ingot_encode_options *opt, encode_io *io, const entry_list *l,
                           ingot_file_header *h) {
    uint64_t consumed = ingot_input_pos(io->src);
    int rc = ingot_input_hash_finish(io->src, h->source_sha256);
    if (!rc && ingot_input_pos(io->src) != consumed)
        rc = -INGOT_EFORMAT; /* the source grew past its layout */
    h->source_size = consumed;
    if (rc || !uses_base(l) || !opt->base)
        return rc;
    if (opt->base_sha256) {
        memcpy(h->base_sha256, opt->base_sha256, INGOT_SHA256_BYTES);
        ingot_base_set_single_size(opt->base, &h->base_size); /* 0 when unknown (a pipe) */
        return 0;
    }
    rc = ingot_base_set_identity(opt->base, &h->base_size, h->base_sha256);
    return rc == -INGOT_ENOTFOUND ? 0 : rc;
}

/* Header placeholder, source header, directory placeholder. */
static int write_front(const ingot_layout *layout, const entry_list *l, encode_io *io,
                       ingot_file_header *h) {
    static const uint8_t zeros[INGOT_HEADER_BYTES] = {0};
    const uint8_t *src_header = ingot_layout_header(layout, &h->source_header_size);
    h->kind = ingot_layout_kind(layout);
    h->entry_count = (uint32_t)l->count;
    h->source_header_offset = INGOT_HEADER_BYTES;
    for (size_t i = 0; i < l->count; i++)
        h->dir_size += ingot_entry_record_size(&l->items[i]);
    h->dir_size += l->tail_bytes;
    int rc = ingot_sink_write(&io->sink, zeros, sizeof(zeros));
    if (!rc)
        rc = ingot_sink_write(&io->sink, src_header, (size_t)h->source_header_size);
    if (!rc)
        rc = ingot_sink_pad(&io->sink, 8);
    h->dir_offset = io->sink.pos;
    for (uint64_t left = h->dir_size; !rc && left;) {
        size_t n = (size_t)(left < sizeof(zeros) ? left : sizeof(zeros));
        rc = ingot_sink_write(&io->sink, zeros, n); /* the directory goes here at the end */
        left -= n;
    }
    return rc ? rc : ingot_sink_pad(&io->sink, 8);
}

static int encode_all(const ingot_encode_options *opt, const ingot_layout *layout,
                      entry_list *l, encode_io *io, ingot_file_header *h) {
    int rc = write_front(layout, l, io, h);
    uint64_t total = ingot_input_seekable(io->src) ? ingot_input_size(io->src) : 0;
    for (size_t i = 0; !rc && i < l->count; i++) {
        rc = process_entry(opt, io, l, i);
        if (!rc)
            rc = drain_pending(opt, io, l);
        if (!rc && opt->progress)
            opt->progress(opt->progress_ctx, ingot_input_pos(io->src), total);
    }
    if (!rc && l->emitted != l->count)
        rc = -INGOT_EFORMAT; /* a dependent whose partners never came */
    if (!rc)
        rc = finish_identity(opt, io, l, h);
    if (!rc)
        rc = write_directory(&io->sink, l, h->dir_offset, h->dir_size);
    h->file_size = io->sink.pos;
    h->version = rc ? INGOT_VERSION : file_version(l);
    uint8_t header[INGOT_HEADER_BYTES];
    ingot_header_write(h, header);
    return rc ? rc : ingot_sink_write_at(&io->sink, 0, header, sizeof(header));
}

static void fill_stats(const entry_list *l, const ingot_file_header *h, ingot_encode_stats *st) {
    *st = (ingot_encode_stats){.source_bytes = h->source_size, .output_bytes = h->file_size};
    st->v2_output_bytes = (uint64_t)((int64_t)h->file_size + l->v2_saved);
    memcpy(st->source_sha256, h->source_sha256, sizeof(st->source_sha256));
    for (size_t i = 0; i < l->count; i++) {
        const ingot_entry *e = &l->items[i];
        if (!e->name[0])
            continue;
        uint64_t n = 1;
        for (int d = 0; d < e->ndim; d++)
            n *= e->shape[d];
        st->elements += n;
        st->tensors[e->mode]++;
        st->coded_elements[e->mode] += n;
        st->payload_bytes[e->mode] += e->payload_size;
    }
}

static void list_free(entry_list *l) {
    for (size_t i = 0; i < l->count; i++) {
        free(l->items[i].name);
        if (l->plans)
            free(l->plans[i].data);
    }
    free(l->items);
    free(l->plans);
    free(l->order);
    free(l->tensor_entry);
    ingot_match_writer_free(l->match);
}

int ingot_encode_input(ingot_input *src, const char *dst_path, const ingot_encode_options *opt,
                       ingot_encode_stats *stats) {
    if (!src)
        return -INGOT_EINVAL;
    ingot_encode_options defaults = {.threads = 1};
    opt = opt ? opt : &defaults;
    ingot_layout *layout = NULL;
    entry_list list = {0};
    encode_io io = {.src = src, .sink = {.fd = INGOT_SINK_MEMORY}};
    ingot_file_header h = {0};
    int rc = ingot_input_hash_start(src);
    if (!rc)
        rc = ingot_layout_read(src, &layout);
    if (!rc)
        rc = collect_entries(layout, src, &list);
    if (!rc)
        rc = plan_entries(opt, layout, &list);
    if (!rc && !dst_path) {
        io.sink.fd = INGOT_SINK_DISCARD; /* sizing run: nothing is written */
    } else if (!rc) {
        io.sink.fd = open(dst_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        rc = io.sink.fd >= 0 ? 0 : -INGOT_EIO;
    }
    if (!rc)
        rc = encode_all(opt, layout, &list, &io, &h);
    if (!rc && stats)
        fill_stats(&list, &h, stats);
    if (io.sink.fd >= 0 && close(io.sink.fd) && !rc)
        rc = -INGOT_EIO;
    if (rc && io.sink.fd >= 0)
        unlink(dst_path);
    list_free(&list);
    free(io.buf);
    ingot_layout_free(layout);
    return rc;
}

int ingot_encode_file(const char *src_path, const char *dst_path,
                      const ingot_encode_options *opt, ingot_encode_stats *stats) {
    ingot_input *in = NULL;
    int rc = ingot_input_open_path(src_path, &in);
    if (!rc)
        rc = ingot_encode_input(in, dst_path, opt, stats);
    ingot_input_close(in);
    return rc;
}
