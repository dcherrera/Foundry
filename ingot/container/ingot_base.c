/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_base.c
 * @brief  Delta base set over model files and .ingot files (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_base.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dirent.h>
#include <sys/stat.h>

#include "ingot/container/ingot_layout.h"
#include "ingot/container/ingot_reader.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"
#include "ingot/core/ingot_input.h"
#include "tt/convert/ft_sha256.h"

enum { HASH_CHUNK = 1 << 20 };

typedef struct {
    ingot_input *input;   /* model file (with layout), or NULL for an .ingot file */
    ingot_layout *layout;
    ingot_reader *reader;
} base_file;

struct ingot_base_set {
    base_file *files;
    size_t count;
    size_t capacity;
    int threads;
    uint8_t *buffer;          /* load() target and the decoded .ingot tensor */
    uint64_t buffer_capacity;
    size_t decoded_file;      /* which .ingot tensor the buffer holds (SIZE_MAX: none) */
    size_t decoded_index;
};

static int ends_with(const char *s, const char *suffix) {
    size_t n = strlen(s), m = strlen(suffix);
    return n >= m && !strcmp(s + n - m, suffix);
}

static int open_model_file(const char *path, base_file *f) {
    int rc = ingot_input_open_path(path, &f->input);
    if (!rc && !ingot_input_seekable(f->input))
        rc = ingot_input_hash_start(f->input); /* a pipe can only be hashed as it is read */
    return rc ? rc : ingot_layout_read(f->input, &f->layout);
}

static int add_file(ingot_base_set *b, const char *path) {
    if (b->count == b->capacity) {
        size_t cap = b->capacity ? b->capacity * 2 : 16;
        base_file *files = realloc(b->files, cap * sizeof(*files));
        if (!files)
            return -INGOT_ENOMEM;
        b->files = files;
        b->capacity = cap;
    }
    base_file *f = &b->files[b->count];
    *f = (base_file){0};
    struct stat st;
    int regular = stat(path, &st) == 0 && S_ISREG(st.st_mode);
    int rc = regular ? ingot_reader_probe(path, NULL) : 0; /* by magic, not by name */
    if (rc >= 0)
        rc = rc ? ingot_reader_open(path, &f->reader) : open_model_file(path, f);
    if (rc) {
        ingot_layout_free(f->layout);
        ingot_input_close(f->input);
        return rc;
    }
    b->count++;
    return 0;
}

static int add_directory(ingot_base_set *b, const char *dir) {
    DIR *d = opendir(dir);
    if (!d)
        return -INGOT_EIO;
    int rc = 0;
    struct dirent *ent = NULL;
    while (!rc && (ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.')
            continue;
        char path[4096];
        int n = snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
        if (n < 0 || (size_t)n >= sizeof(path)) {
            rc = -INGOT_EINVAL;
            break;
        }
        if (ends_with(ent->d_name, ".safetensors") || ends_with(ent->d_name, ".gguf") ||
            ingot_reader_probe(path, NULL) == 1)
            rc = add_file(b, path);
    }
    closedir(d);
    return rc;
}

int ingot_base_set_open(const char *const *paths, size_t count, int threads,
                        ingot_base_set **out) {
    if (!out || (count && !paths))
        return -INGOT_EINVAL;
    ingot_base_set *b = calloc(1, sizeof(*b));
    if (!b)
        return -INGOT_ENOMEM;
    b->threads = threads;
    b->decoded_file = SIZE_MAX;
    int rc = 0;
    for (size_t i = 0; !rc && i < count; i++) {
        struct stat st;
        if (stat(paths[i], &st))
            rc = -INGOT_EIO;
        else
            rc = S_ISDIR(st.st_mode) ? add_directory(b, paths[i]) : add_file(b, paths[i]);
    }
    if (rc) {
        ingot_base_set_close(b);
        return rc;
    }
    *out = b;
    return 0;
}

void ingot_base_set_close(ingot_base_set *b) {
    if (!b)
        return;
    for (size_t i = 0; i < b->count; i++) {
        ingot_layout_free(b->files[i].layout);
        ingot_input_close(b->files[i].input);
        ingot_reader_close(b->files[i].reader);
    }
    free(b->files);
    free(b->buffer);
    free(b);
}

int ingot_base_set_find(const ingot_base_set *b, const char *name, ingot_base_tensor *t) {
    if (!b || !name || !t)
        return -INGOT_EINVAL;
    for (size_t f = 0; f < b->count; f++) {
        size_t i = 0;
        const base_file *bf = &b->files[f];
        if (bf->reader && ingot_reader_find(bf->reader, name, &i) == 0) {
            const ingot_entry *e = ingot_reader_entry(bf->reader, i);
            *t = (ingot_base_tensor){e->dtype, e->source_size, f, i};
            return 0;
        }
        if (bf->layout && ingot_layout_find(bf->layout, name, &i) == 0) {
            const ingot_layout_tensor *lt = ingot_layout_tensor_at(bf->layout, i);
            *t = (ingot_base_tensor){lt->dtype, lt->size, f, i};
            return 0;
        }
    }
    return -INGOT_ENOTFOUND;
}

static int reserve(ingot_base_set *b, uint64_t bytes) {
    if (bytes <= b->buffer_capacity)
        return 0;
    free(b->buffer);
    b->decoded_file = SIZE_MAX;
    b->buffer = malloc((size_t)(bytes ? bytes : 1));
    b->buffer_capacity = b->buffer ? bytes : 0;
    return b->buffer ? 0 : -INGOT_ENOMEM;
}

/* Decode an .ingot base tensor into the buffer (cached). */
static int decode_ingot_tensor(ingot_base_set *b, const ingot_base_tensor *t) {
    if (b->decoded_file == t->file && b->decoded_index == t->index)
        return 0;
    const ingot_entry *e = ingot_reader_entry(b->files[t->file].reader, t->index);
    if (e->mode == INGOT_MODE_DELTA || e->mode == INGOT_MODE_QUANT_DELTA ||
        e->mode == INGOT_MODE_BLOCK_DELTA)
        return -INGOT_EUNSUPPORTED; /* no chained deltas */
    int rc = reserve(b, e->source_size);
    if (!rc)
        rc = ingot_reader_decode(b->files[t->file].reader, t->index, NULL, b->threads, b->buffer);
    if (rc)
        return rc;
    b->decoded_file = t->file;
    b->decoded_index = t->index;
    return 0;
}

int ingot_base_set_read(ingot_base_set *b, const ingot_base_tensor *t, uint64_t offset,
                        void *buf, size_t n) {
    if (!b || !t || t->file >= b->count || offset > t->size || n > t->size - offset)
        return -INGOT_EINVAL;
    const base_file *f = &b->files[t->file];
    if (f->layout) {
        const ingot_layout_tensor *lt = ingot_layout_tensor_at(f->layout, t->index);
        return ingot_input_read_at(f->input, lt->offset + offset, buf, n);
    }
    int rc = decode_ingot_tensor(b, t);
    if (!rc)
        memcpy(buf, b->buffer + offset, n);
    return rc;
}

int ingot_base_set_load(ingot_base_set *b, const ingot_base_tensor *t, const uint8_t **data) {
    if (!b || !t || !data || t->file >= b->count)
        return -INGOT_EINVAL;
    int rc = 0;
    if (b->files[t->file].reader) {
        rc = decode_ingot_tensor(b, t);
    } else {
        rc = reserve(b, t->size);
        b->decoded_file = SIZE_MAX;
        if (!rc)
            rc = ingot_base_set_read(b, t, 0, b->buffer, (size_t)t->size);
    }
    if (!rc)
        *data = b->buffer;
    return rc;
}

int ingot_base_set_read_prefix(ingot_base_set *b, const ingot_base_tensor *t, void *buf,
                               size_t n) {
    if (!b || !t || !buf || t->file >= b->count || n > t->size)
        return -INGOT_EINVAL;
    const base_file *f = &b->files[t->file];
    if (f->layout) {
        const ingot_layout_tensor *lt = ingot_layout_tensor_at(f->layout, t->index);
        return ingot_input_read_at(f->input, lt->offset, buf, n);
    }
    return ingot_reader_decode_prefix(f->reader, t->index, n, b->threads, buf);
}

/* Does found tensor t hold ref: its dtype and size, or (allow_prefix) a 2-D tensor with the
 * ref's row length and more rows, the ref being its first rows? */
static int holds_ref(const ingot_base_set *b, const ingot_base_tensor *t, const ingot_entry *ref,
                     int allow_prefix) {
    int ndim = 0;
    uint64_t shape[INGOT_MAX_DIMS] = {0};
    if (t->dtype != ref->dtype || ingot_base_set_shape(b, t, &ndim, shape))
        return 0;
    if (t->size == ref->source_size)
        return 1;
    return allow_prefix && ndim == 2 && ref->ndim == 2 && shape[1] == ref->shape[1] &&
           shape[0] > ref->shape[0] && ref->source_size * shape[0] == t->size * ref->shape[0];
}

int ingot_base_set_load_ref(ingot_base_set *b, const ingot_entry *ref, int allow_prefix,
                            uint8_t **out) {
    ingot_base_tensor t;
    if (!out)
        return -INGOT_EINVAL;
    *out = NULL;
    if (!b || !ref || ingot_base_set_find(b, ref->name, &t) ||
        !holds_ref(b, &t, ref, allow_prefix))
        return -INGOT_EBASE;
    size_t n = (size_t)ref->source_size;
    uint8_t *buf = malloc(n ? n : 1);
    if (!buf)
        return -INGOT_ENOMEM;
    int rc = ingot_base_set_read_prefix(b, &t, buf, n);
    if (!rc && ingot_hash64(buf, n) != ref->hash)
        rc = -INGOT_EBASE;
    if (rc) {
        free(buf);
        return rc == -INGOT_EUNSUPPORTED ? -INGOT_EBASE : rc;
    }
    *out = buf;
    return 0;
}

static const base_file *single_model_file(const ingot_base_set *b) {
    return (b && b->count == 1 && b->files[0].layout) ? &b->files[0] : NULL;
}

int ingot_base_set_single_size(const ingot_base_set *b, uint64_t *size) {
    const base_file *f = single_model_file(b);
    if (!f)
        return -INGOT_ENOTFOUND;
    *size = ingot_input_size(f->input);
    return 0;
}

static int hash_seekable(ingot_input *in, uint8_t sha256[32]) {
    uint8_t *chunk = malloc(HASH_CHUNK);
    if (!chunk)
        return -INGOT_ENOMEM;
    ft_sha256_ctx ctx;
    ft_sha256_init(&ctx);
    uint64_t size = ingot_input_size(in);
    int rc = 0;
    for (uint64_t at = 0; !rc && at < size; at += HASH_CHUNK) {
        size_t n = (size_t)(size - at < HASH_CHUNK ? size - at : HASH_CHUNK);
        rc = ingot_input_read_at(in, at, chunk, n);
        if (!rc)
            ft_sha256_update(&ctx, chunk, n);
    }
    if (!rc)
        ft_sha256_final(&ctx, sha256);
    free(chunk);
    return rc;
}

int ingot_base_set_identity(ingot_base_set *b, uint64_t *size, uint8_t sha256[32]) {
    const base_file *f = single_model_file(b);
    if (!f || !size || !sha256)
        return -INGOT_ENOTFOUND;
    if (ingot_input_seekable(f->input)) {
        *size = ingot_input_size(f->input);
        return hash_seekable(f->input, sha256);
    }
    int rc = ingot_input_hash_finish(f->input, sha256);
    if (!rc)
        *size = ingot_input_pos(f->input);
    return rc;
}

int ingot_base_set_shape(const ingot_base_set *b, const ingot_base_tensor *t, int *ndim,
                         uint64_t *shape) {
    if (!b || !t || t->file >= b->count)
        return -INGOT_EINVAL;
    const base_file *f = &b->files[t->file];
    if (f->layout) {
        const ingot_layout_tensor *lt = ingot_layout_tensor_at(f->layout, t->index);
        *ndim = lt->ndim;
        memcpy(shape, lt->shape, sizeof(lt->shape));
    } else {
        const ingot_entry *e = ingot_reader_entry(f->reader, t->index);
        *ndim = e->ndim;
        memcpy(shape, e->shape, sizeof(e->shape));
    }
    return 0;
}

size_t ingot_base_set_files(const ingot_base_set *b) { return b ? b->count : 0; }
