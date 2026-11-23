/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_writer_match.c
 * @brief  The writer's match-mode state: pools, caches, candidate search, directory tail (see
 *         header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_writer_match.h"

#include <stdlib.h>
#include <string.h>

#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_dirtail.h"
#include "ingot/container/ingot_match_pools.h"
#include "ingot/container/ingot_predict_rules.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"

enum {
    MEMBER_ENTRY = 0,
    MEMBER_REF = 1,
    MEMBER_SELF = 2,
    DEFAULT_CANDIDATES = 16,
    MAX_NAME = 2048,
    TRANSPOSE_TILE = 64,
};

static const int DEFAULT_POOL_LAYERS[] = {1, 16};

typedef struct {
    int kind;
    size_t id; /* entry index or ref index */
} pool_member;

/* Per entry: role, pool (targets only), caches. */
typedef struct {
    int is_target;
    ingot_match_role role;
    size_t layout_index;
    pool_member *pool;
    uint32_t pool_count;
    size_t *deps;
    uint32_t dep_count;
    uint32_t *used;      /* source refs of the written payload */
    uint32_t used_count;
    int written;
    uint16_t *unit_major; /* transposed copy (column units), else NULL */
    float *sketch;
} entry_info;

/* An external tensor some target's pool names. */
typedef struct {
    ingot_entry e;        /* name, dtype, ndim, shape, source_size, hash (once loaded) */
    ingot_base_tensor t;
    ingot_match_role role;
    uint8_t *data;
    uint16_t *unit_major;
    float *sketch;
    uint32_t pending;     /* targets not yet written that pool it */
} ref_info;

struct ingot_match_writer {
    const ingot_encode_options *opt;
    const ingot_layout *layout;
    const size_t *tensor_entry;
    size_t entry_count;
    entry_info *info;
    ref_info *refs;
    uint32_t ref_count;
    uint32_t ref_capacity;
};

static int find_ref(ingot_match_writer *w, const char *name, const ingot_match_role *t,
                    size_t *id) {
    for (uint32_t r = 0; r < w->ref_count; r++)
        if (!strcmp(w->refs[r].e.name, name)) {
            *id = r;
            return 1;
        }
    ingot_base_tensor bt;
    ref_info ri = {0};
    if (!w->opt->refs || ingot_base_set_find(w->opt->refs, name, &bt) ||
        ingot_base_set_shape(w->opt->refs, &bt, &ri.e.ndim, ri.e.shape) ||
        !ingot_match_role_of(name, bt.dtype, ri.e.ndim, ri.e.shape, &ri.role) ||
        ri.role.unit_len != t->unit_len || bt.size != ri.e.shape[0] * ri.e.shape[1] * 2)
        return 0;
    if (w->ref_count == w->ref_capacity) {
        uint32_t cap = w->ref_capacity ? 2 * w->ref_capacity : 32;
        ref_info *grown = realloc(w->refs, cap * sizeof(*grown));
        if (!grown)
            return -INGOT_ENOMEM;
        w->refs = grown;
        w->ref_capacity = cap;
    }
    ri.e.name = strdup(name);
    if (!ri.e.name)
        return -INGOT_ENOMEM;
    ri.e.dtype = bt.dtype;
    ri.e.source_size = bt.size;
    ri.t = bt;
    *id = w->ref_count;
    w->refs[w->ref_count++] = ri;
    return 1;
}

/* Pool member k's name: an entry of this file (same unit length), else a ref, else none. */
static int add_member(ingot_match_writer *w, entry_info *ti, const char *name) {
    size_t at = 0, id = 0;
    if (!name[0])
        return 0;
    if (ingot_layout_find(w->layout, name, &at) == 0) {
        size_t e = w->tensor_entry[at];
        entry_info *m = &w->info[e];
        if (!m->role.is_role || m->role.unit_len != ti->role.unit_len)
            return 0;
        ti->pool[ti->pool_count++] = (pool_member){MEMBER_ENTRY, e};
        ti->deps[ti->dep_count++] = e;
        return 0;
    }
    int rc = find_ref(w, name, &ti->role, &id);
    if (rc == 1) {
        ti->pool[ti->pool_count++] = (pool_member){MEMBER_REF, id};
        w->refs[id].pending++;
        rc = 0;
    }
    return rc;
}

static int build_pool(ingot_match_writer *w, size_t e) {
    entry_info *ti = &w->info[e];
    const ingot_layout_tensor *lt = ingot_layout_tensor_at(w->layout, ti->layout_index);
    const int *offs = w->opt->pool_layers ? w->opt->pool_layers : DEFAULT_POOL_LAYERS;
    size_t noff = w->opt->pool_layers ? (size_t)w->opt->pool_layer_count : 2;
    char name[MAX_NAME];
    size_t total = ingot_match_pool_names(lt->name, &ti->role, offs, noff, 0, name, sizeof(name));
    ti->pool = calloc(total + 1, sizeof(*ti->pool));
    ti->deps = calloc(total + 1, sizeof(*ti->deps));
    if (!ti->pool || !ti->deps)
        return -INGOT_ENOMEM;
    int rc = 0;
    for (size_t k = 0; !rc && k < total && ti->pool_count < INGOT_MATCH_MAX_SOURCES - 1; k++) {
        ingot_match_pool_names(lt->name, &ti->role, offs, noff, k, name, sizeof(name));
        rc = add_member(w, ti, name);
    }
    if (ti->role.axis == INGOT_PREDICT_ROWS)
        ti->pool[ti->pool_count++] = (pool_member){MEMBER_SELF, e};
    return rc;
}

int ingot_match_writer_create(const ingot_encode_options *opt, const ingot_layout *layout,
                              const size_t *tensor_entry, size_t entry_count,
                              ingot_match_writer **out) {
    ingot_match_writer *w = calloc(1, sizeof(*w));
    if (!w)
        return -INGOT_ENOMEM;
    *w = (ingot_match_writer){opt, layout, tensor_entry, entry_count, NULL, NULL, 0, 0};
    w->info = calloc(entry_count ? entry_count : 1, sizeof(*w->info));
    int rc = w->info ? 0 : -INGOT_ENOMEM;
    size_t tensors = ingot_layout_count(layout);
    for (size_t t = 0; !rc && t < tensors; t++) {
        const ingot_layout_tensor *lt = ingot_layout_tensor_at(layout, t);
        entry_info *ti = &w->info[tensor_entry[t]];
        ti->layout_index = t;
        ti->is_target = ingot_match_role_of(lt->name, lt->dtype, lt->ndim, lt->shape, &ti->role);
    }
    for (size_t t = 0; !rc && t < tensors; t++)
        if (w->info[tensor_entry[t]].is_target)
            rc = build_pool(w, tensor_entry[t]);
    if (rc) {
        ingot_match_writer_free(w);
        return rc;
    }
    *out = w;
    return 0;
}

int ingot_match_writer_is_target(const ingot_match_writer *w, size_t i) {
    return w && i < w->entry_count && w->info[i].is_target;
}

uint32_t ingot_match_writer_deps(const ingot_match_writer *w, size_t i, const size_t **deps) {
    if (!ingot_match_writer_is_target(w, i))
        return 0;
    *deps = w->info[i].deps;
    return w->info[i].dep_count;
}

/* ---- Caches ---- */

/* Unit-major view of a tensor: data itself for row units, else a transposed copy. */
static int unit_major(const uint8_t *data, const ingot_match_role *r, uint16_t **copy,
                      const uint16_t **view) {
    const uint16_t *src = (const uint16_t *)(const void *)data;
    if (r->axis == INGOT_PREDICT_ROWS) {
        *view = src;
        return 0;
    }
    if (!*copy) {
        uint64_t rows = r->unit_len, cols = r->units; /* stored [H][U] */
        *copy = malloc((size_t)(rows * cols) * sizeof(uint16_t));
        if (!*copy)
            return -INGOT_ENOMEM;
        for (uint64_t r0 = 0; r0 < rows; r0 += TRANSPOSE_TILE)
            for (uint64_t c0 = 0; c0 < cols; c0 += TRANSPOSE_TILE)
                for (uint64_t i = r0; i < rows && i < r0 + TRANSPOSE_TILE; i++)
                    for (uint64_t j = c0; j < cols && j < c0 + TRANSPOSE_TILE; j++)
                        (*copy)[j * rows + i] = src[i * cols + j];
    }
    *view = *copy;
    return 0;
}

static int sketch_of(const uint16_t *view, const ingot_match_role *r, int threads,
                     float **sketch) {
    if (*sketch)
        return 0;
    *sketch = malloc((size_t)r->units * INGOT_MATCH_SKETCH_DIM * sizeof(float));
    if (!*sketch)
        return -INGOT_ENOMEM;
    return ingot_match_sketch(view, r->units, r->unit_len, threads, *sketch);
}

static int load_ref(ingot_match_writer *w, ref_info *r) {
    if (r->data)
        return 0;
    const uint8_t *buf = NULL;
    int rc = ingot_base_set_load(w->opt->refs, &r->t, &buf);
    if (rc)
        return rc == -INGOT_EUNSUPPORTED ? -INGOT_EBASE : rc;
    r->data = malloc((size_t)r->e.source_size);
    if (!r->data)
        return -INGOT_ENOMEM;
    memcpy(r->data, buf, (size_t)r->e.source_size);
    r->e.hash = ingot_hash64(r->data, (size_t)r->e.source_size);
    return 0;
}

static void ref_drop(ref_info *r) {
    free(r->data);
    free(r->unit_major);
    free(r->sketch);
    r->data = NULL;
    r->unit_major = NULL;
    r->sketch = NULL;
}

void ingot_match_writer_release_entry(ingot_match_writer *w, size_t i) {
    if (!w || i >= w->entry_count)
        return;
    free(w->info[i].unit_major);
    free(w->info[i].sketch);
    w->info[i].unit_major = NULL;
    w->info[i].sketch = NULL;
}

/* ---- Prepare ---- */

/* Source s of the prepared candidate from pool member m. */
static int prepare_member(ingot_match_writer *w, const pool_member *m, size_t target,
                          const uint8_t *data, const ingot_entry *entries,
                          const uint8_t *const *entry_data, const uint32_t *position,
                          ingot_writer_match *out, ingot_match_pool *pool) {
    int threads = w->opt->threads;
    uint32_t s = out->source_count;
    ingot_match_input *in = &out->src[s];
    int rc = 0;
    if (m->kind == MEMBER_REF) {
        ref_info *r = &w->refs[m->id];
        rc = load_ref(w, r);
        if (!rc)
            rc = unit_major(r->data, &r->role, &r->unit_major, &in->unit_major);
        if (!rc)
            rc = sketch_of(in->unit_major, &r->role, threads, &r->sketch);
        *in = (ingot_match_input){(uint32_t)(w->entry_count + m->id), r->role.axis,
                                  r->role.units, (const uint16_t *)(const void *)r->data,
                                  in->unit_major, 0};
        out->src_entry[s] = &r->e;
        pool[s] = (ingot_match_pool){r->sketch, r->role.units, 0};
    } else {
        size_t j = m->kind == MEMBER_SELF ? target : m->id;
        entry_info *ei = &w->info[j];
        const uint8_t *bytes = m->kind == MEMBER_SELF ? data : entry_data[j];
        if (!bytes)
            return -INGOT_EINVAL;
        rc = unit_major(bytes, &ei->role, &ei->unit_major, &in->unit_major);
        if (!rc)
            rc = sketch_of(in->unit_major, &ei->role, threads, &ei->sketch);
        int self = m->kind == MEMBER_SELF;
        *in = (ingot_match_input){self ? INGOT_MATCH_SELF_REF : position[j], ei->role.axis,
                                  ei->role.units, (const uint16_t *)(const void *)bytes,
                                  in->unit_major, self};
        out->src_entry[s] = &entries[j];
        pool[s] = (ingot_match_pool){ei->sketch, ei->role.units, self};
    }
    if (!rc)
        out->source_count++;
    return rc;
}

/* The v3 same-unit partners as extra candidates (slots after the searched ones). */
static void add_rule_candidates(ingot_match_writer *w, size_t target, ingot_writer_match *out,
                                uint32_t searched) {
    ingot_predict_geometry g;
    size_t partner[INGOT_PREDICT_MAX_PARTNERS] = {0};
    entry_info *ti = &w->info[target];
    uint32_t per = out->cand.per_unit;
    for (uint64_t u = 0; u < out->units; u++)
        for (uint32_t j = searched; j < per; j++)
            out->cand.source[u * per + j] = 0xFF;
    if (ingot_predict_rule(w->layout, ti->layout_index, &g, partner) != 1 ||
        g.units != out->units)
        return;
    for (uint32_t j = 0; j < g.partner_count && searched + j < per; j++) {
        size_t e = w->tensor_entry[partner[j]];
        uint32_t s = 0;
        while (s < ti->pool_count && !(ti->pool[s].kind == MEMBER_ENTRY && ti->pool[s].id == e))
            s++;
        if (s >= ti->pool_count || s >= out->source_count ||
            out->src[s].axis != INGOT_PREDICT_ROWS)
            continue;
        for (uint64_t u = 0; u < out->units; u++) {
            out->cand.source[u * per + searched + j] = (uint8_t)s;
            out->cand.index[u * per + searched + j] = ingot_predict_map(&g.partner[j], u);
        }
    }
}

static int search(ingot_match_writer *w, size_t target, const ingot_match_pool *pool,
                  ingot_writer_match *out) {
    entry_info *ti = &w->info[target];
    uint32_t m = w->opt->match_candidates > 0 ? (uint32_t)w->opt->match_candidates
                                              : DEFAULT_CANDIDATES;
    uint32_t per = m + INGOT_PREDICT_MAX_PARTNERS;
    ingot_match_candidates found = {m, malloc((size_t)out->units * m),
                                    malloc((size_t)out->units * m * sizeof(uint64_t))};
    out->cand = (ingot_match_candidates){per, malloc((size_t)out->units * per),
                                         malloc((size_t)out->units * per * sizeof(uint64_t))};
    int rc = found.source && found.index && out->cand.source && out->cand.index
                 ? ingot_match_search(ti->sketch, out->units, pool, out->source_count,
                                      w->opt->threads, &found)
                 : -INGOT_ENOMEM;
    for (uint64_t u = 0; !rc && u < out->units; u++) {
        memcpy(out->cand.source + u * per, found.source + u * m, m);
        memcpy(out->cand.index + u * per, found.index + u * m, m * sizeof(uint64_t));
    }
    if (!rc)
        add_rule_candidates(w, target, out, m);
    free(found.source);
    free(found.index);
    return rc;
}

int ingot_match_writer_prepare(ingot_match_writer *w, size_t i, const uint8_t *data,
                               const ingot_entry *entries, const uint8_t *const *entry_data,
                               const uint32_t *position, ingot_writer_match *out) {
    entry_info *ti = &w->info[i];
    *out = (ingot_writer_match){.axis = ti->role.axis, .units = ti->role.units,
                                .unit_len = ti->role.unit_len};
    ingot_match_pool pool[INGOT_MATCH_MAX_SOURCES];
    int rc = unit_major(data, &ti->role, &ti->unit_major, &out->unit_major);
    if (!rc)
        rc = sketch_of(out->unit_major, &ti->role, w->opt->threads, &ti->sketch);
    for (uint32_t k = 0; !rc && k < ti->pool_count; k++)
        rc = prepare_member(w, &ti->pool[k], i, data, entries, entry_data, position, out, pool);
    if (!rc)
        rc = search(w, i, pool, out);
    return rc;
}

void ingot_match_writer_done(ingot_writer_match *m) {
    free(m->cand.source);
    free(m->cand.index);
    m->cand.source = NULL;
    m->cand.index = NULL;
}

int ingot_match_writer_note_used(ingot_match_writer *w, size_t i, const uint32_t *refs,
                                 uint32_t n) {
    entry_info *ti = &w->info[i];
    ti->written = 1;
    if (n) {
        ti->used = malloc(n * sizeof(*ti->used));
        if (!ti->used)
            return -INGOT_ENOMEM;
        memcpy(ti->used, refs, n * sizeof(*refs));
        ti->used_count = n;
    }
    for (uint32_t k = 0; k < ti->pool_count; k++)
        if (ti->pool[k].kind == MEMBER_REF && --w->refs[ti->pool[k].id].pending == 0)
            ref_drop(&w->refs[ti->pool[k].id]);
    return 0;
}

/* ---- Tail ---- */

size_t ingot_match_writer_tail_reserve(const ingot_match_writer *w) {
    size_t n = 8, targets = 0;
    for (uint32_t r = 0; r < w->ref_count; r++)
        n += ingot_dir_tail_ref_size(&w->refs[r].e);
    for (size_t i = 0; i < w->entry_count; i++)
        if (w->info[i].is_target) {
            n += ingot_dir_tail_list_size(w->info[i].pool_count);
            targets++;
        }
    return targets ? (n + 7) & ~(size_t)7 : 0;
}

int ingot_match_writer_tail_write(const ingot_match_writer *w, const uint32_t *position,
                                  uint8_t *out, size_t size) {
    size_t lists = 0;
    for (size_t i = 0; i < w->entry_count; i++)
        lists += w->info[i].used_count > 0;
    ingot_entry *refs = calloc(w->ref_count ? w->ref_count : 1, sizeof(*refs));
    uint32_t *entry = calloc(lists ? lists : 1, sizeof(*entry));
    uint32_t *count = calloc(lists ? lists : 1, sizeof(*count));
    const uint32_t **src = calloc(lists ? lists : 1, sizeof(*src));
    int rc = refs && entry && count && src ? 0 : -INGOT_ENOMEM;
    for (uint32_t r = 0; !rc && r < w->ref_count; r++) {
        refs[r] = w->refs[r].e;
        refs[r].ndim = 2;
    }
    size_t l = 0;
    for (size_t i = 0; !rc && i < w->entry_count; i++) {
        if (!w->info[i].used_count)
            continue;
        entry[l] = position[i];
        count[l] = w->info[i].used_count;
        src[l++] = w->info[i].used;
    }
    if (!rc)
        rc = ingot_dir_tail_write(refs, w->ref_count, entry, count, src, (uint32_t)lists, out,
                                  size);
    free(refs);
    free(entry);
    free(count);
    free(src);
    return rc;
}

void ingot_match_writer_free(ingot_match_writer *w) {
    if (!w)
        return;
    for (size_t i = 0; w->info && i < w->entry_count; i++) {
        free(w->info[i].pool);
        free(w->info[i].deps);
        free(w->info[i].used);
        free(w->info[i].unit_major);
        free(w->info[i].sketch);
    }
    for (uint32_t r = 0; r < w->ref_count; r++) {
        free(w->refs[r].e.name);
        ref_drop(&w->refs[r]);
    }
    free(w->refs);
    free(w->info);
    free(w);
}
