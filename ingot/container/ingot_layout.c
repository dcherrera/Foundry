/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_layout.c
 * @brief  Source layout: safetensors and GGUF headers parsed from a stream (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_layout.h"

#include <stdlib.h>
#include <string.h>

#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_ggml.h"
#include "safetensors/safetensors.h"

/* GGUF metadata value types (gguf spec). */
enum {
    GGUF_UINT8 = 0, GGUF_INT8 = 1, GGUF_UINT16 = 2, GGUF_INT16 = 3, GGUF_UINT32 = 4,
    GGUF_INT32 = 5, GGUF_FLOAT32 = 6, GGUF_BOOL = 7, GGUF_STRING = 8, GGUF_ARRAY = 9,
    GGUF_UINT64 = 10, GGUF_INT64 = 11, GGUF_FLOAT64 = 12,
};

enum {
    GGUF_DEFAULT_ALIGNMENT = 32,
    MAX_STRING = 1 << 24,      /* sanity bound on one GGUF string */
    MAX_HEADER = 1u << 30,     /* sanity bound on a header region */
    MAX_TENSORS = 1 << 20,
};

struct ingot_layout {
    ingot_source_kind kind;
    uint8_t *header;
    uint64_t header_size;
    uint64_t capacity;
    ingot_layout_tensor *tensors;
    char **names;
    size_t count;
};

/* Read n more header bytes from the input into the header buffer; *at = where they start. */
static int need(ingot_input *in, ingot_layout *l, uint64_t n, uint64_t *at) {
    if (n > MAX_HEADER - l->header_size)
        return -INGOT_EFORMAT;
    if (l->header_size + n > l->capacity) {
        uint64_t cap = l->capacity ? l->capacity : 1 << 16;
        while (cap < l->header_size + n)
            cap *= 2;
        uint8_t *h = realloc(l->header, (size_t)cap);
        if (!h)
            return -INGOT_ENOMEM;
        l->header = h;
        l->capacity = cap;
    }
    int rc = ingot_input_read(in, l->header + l->header_size, (size_t)n);
    if (rc)
        return rc;
    *at = l->header_size;
    l->header_size += n;
    return 0;
}

static int need_u32(ingot_input *in, ingot_layout *l, uint32_t *v) {
    uint64_t at = 0;
    int rc = need(in, l, 4, &at);
    if (!rc)
        memcpy(v, l->header + at, 4);
    return rc;
}

static int need_u64(ingot_input *in, ingot_layout *l, uint64_t *v) {
    uint64_t at = 0;
    int rc = need(in, l, 8, &at);
    if (!rc)
        memcpy(v, l->header + at, 8);
    return rc;
}

/* A GGUF string; *at / *len locate it in the header buffer. */
static int need_string(ingot_input *in, ingot_layout *l, uint64_t *at, uint64_t *len) {
    int rc = need_u64(in, l, len);
    if (!rc && *len > MAX_STRING)
        rc = -INGOT_EFORMAT;
    return rc ? rc : need(in, l, *len, at);
}

static int scalar_bytes(uint32_t type) {
    switch (type) {
    case GGUF_UINT8: case GGUF_INT8: case GGUF_BOOL: return 1;
    case GGUF_UINT16: case GGUF_INT16: return 2;
    case GGUF_UINT32: case GGUF_INT32: case GGUF_FLOAT32: return 4;
    case GGUF_UINT64: case GGUF_INT64: case GGUF_FLOAT64: return 8;
    default: return -1;
    }
}

/* Skip over one metadata value (read into the header buffer); u32 values are returned. */
static int gguf_value(ingot_input *in, ingot_layout *l, uint32_t type, uint64_t *u32_value) {
    uint64_t at = 0, len = 0;
    if (type == GGUF_STRING)
        return need_string(in, l, &at, &len);
    if (type == GGUF_ARRAY) {
        uint32_t elem = 0;
        uint64_t n = 0;
        int rc = need_u32(in, l, &elem);
        if (!rc)
            rc = need_u64(in, l, &n);
        if (!rc && elem == GGUF_STRING) {
            for (uint64_t i = 0; !rc && i < n; i++)
                rc = need_string(in, l, &at, &len);
            return rc;
        }
        int size = scalar_bytes(elem);
        if (!rc && (size < 0 || n > MAX_HEADER / (uint64_t)size))
            rc = -INGOT_EFORMAT;
        return rc ? rc : need(in, l, n * (uint64_t)size, &at);
    }
    int size = scalar_bytes(type);
    if (size < 0)
        return -INGOT_EFORMAT;
    int rc = need(in, l, (uint64_t)size, &at);
    if (!rc && type == GGUF_UINT32 && u32_value) {
        uint32_t v = 0;
        memcpy(&v, l->header + at, 4);
        *u32_value = v;
    }
    return rc;
}

static int gguf_metadata(ingot_input *in, ingot_layout *l, uint64_t n_kv, uint64_t *alignment) {
    *alignment = GGUF_DEFAULT_ALIGNMENT;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t at = 0, len = 0, value = 0;
        uint32_t type = 0;
        int rc = need_string(in, l, &at, &len);
        if (!rc)
            rc = need_u32(in, l, &type);
        int is_alignment = !rc && len == 17 && !memcmp(l->header + at, "general.alignment", 17);
        if (!rc)
            rc = gguf_value(in, l, type, is_alignment ? &value : NULL);
        if (rc)
            return rc;
        if (is_alignment && type == GGUF_UINT32) {
            if (!value || (value & (value - 1)))
                return -INGOT_EFORMAT;
            *alignment = value;
        }
    }
    return 0;
}

/* Size of a tensor of ggml `type` with GGUF dims ne[0..n); 0 with -INGOT_EUNSUPPORTED if the
 * type is unknown. */
static int gguf_tensor_size(uint32_t type, const uint64_t *ne, uint32_t n, uint64_t *size) {
    uint32_t bw = 0, bb = 0;
    *size = 0;
    if (ingot_ggml_geometry(type, &bw, &bb))
        return -INGOT_EUNSUPPORTED;
    if (n == 0 || ne[0] % bw)
        return -INGOT_EFORMAT;
    uint64_t s = ne[0] / bw * bb;
    for (uint32_t d = 1; d < n; d++) {
        if (ne[d] && s > UINT64_MAX / ne[d])
            return -INGOT_EFORMAT;
        s *= ne[d];
    }
    *size = s;
    return 0;
}

static int gguf_tensor_info(ingot_input *in, ingot_layout *l, size_t i, int *unknown) {
    uint64_t at = 0, len = 0, ne[INGOT_MAX_DIMS] = {0}, offset = 0;
    uint32_t ndim = 0, type = 0;
    int rc = need_string(in, l, &at, &len);
    if (!rc && !(l->names[i] = calloc(1, (size_t)len + 1)))
        rc = -INGOT_ENOMEM;
    if (!rc) {
        memcpy(l->names[i], l->header + at, (size_t)len);
        rc = need_u32(in, l, &ndim);
    }
    if (!rc && ndim > INGOT_MAX_DIMS)
        rc = -INGOT_EFORMAT;
    for (uint32_t d = 0; !rc && d < ndim; d++)
        rc = need_u64(in, l, &ne[d]);
    if (!rc)
        rc = need_u32(in, l, &type);
    if (!rc)
        rc = need_u64(in, l, &offset);
    if (rc)
        return rc;
    ingot_layout_tensor *t = &l->tensors[i];
    t->name = l->names[i];
    t->dtype = ingot_ggml_dtype(type);
    t->ndim = (int)ndim;
    for (uint32_t d = 0; d < ndim; d++)
        t->shape[d] = ne[ndim - 1 - d]; /* GGUF lists ne0 (contiguous) first */
    t->offset = offset;                 /* relative to the data start until fixed up */
    rc = gguf_tensor_size(type, ne, ndim, &t->size);
    if (rc == -INGOT_EUNSUPPORTED) {
        *unknown = 1;
        rc = 0;
    }
    return rc;
}

static int by_offset(const void *a, const void *b) {
    const ingot_layout_tensor *x = a, *y = b;
    return x->offset < y->offset ? -1 : (x->offset > y->offset);
}

/* File order; a tensor of unknown type takes the gap to the next one; no overlaps. */
static int finish_tensors(ingot_layout *l, int unknown) {
    qsort(l->tensors, l->count, sizeof(*l->tensors), by_offset);
    for (size_t i = 0; unknown && i < l->count; i++) {
        if (l->tensors[i].size)
            continue;
        if (i + 1 == l->count)
            return -INGOT_EUNSUPPORTED;
        l->tensors[i].size = l->tensors[i + 1].offset - l->tensors[i].offset;
    }
    uint64_t pos = l->header_size;
    for (size_t i = 0; i < l->count; i++) {
        if (l->tensors[i].offset < pos)
            return -INGOT_EFORMAT;
        pos = l->tensors[i].offset + l->tensors[i].size;
    }
    return 0;
}

/* Magic and version are already in the buffer (read to tell the formats apart). */
static int read_gguf(ingot_input *in, ingot_layout *l) {
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0, alignment = 0, at = 0;
    memcpy(&version, l->header + 4, 4);
    int rc = (version < 2 || version > 3) ? -INGOT_EUNSUPPORTED : 0;
    if (!rc)
        rc = need_u64(in, l, &n_tensors);
    if (!rc)
        rc = need_u64(in, l, &n_kv);
    if (!rc && n_tensors > MAX_TENSORS)
        rc = -INGOT_EFORMAT;
    if (!rc)
        rc = gguf_metadata(in, l, n_kv, &alignment);
    if (rc)
        return rc;
    l->count = (size_t)n_tensors;
    l->tensors = calloc(l->count ? l->count : 1, sizeof(*l->tensors));
    l->names = calloc(l->count ? l->count : 1, sizeof(*l->names));
    if (!l->tensors || !l->names)
        return -INGOT_ENOMEM;
    int unknown = 0;
    for (size_t i = 0; !rc && i < l->count; i++)
        rc = gguf_tensor_info(in, l, i, &unknown);
    uint64_t pad = (alignment - l->header_size % alignment) % alignment;
    if (!rc && pad)
        rc = need(in, l, pad, &at);
    for (size_t i = 0; !rc && i < l->count; i++)
        l->tensors[i].offset += l->header_size; /* GGUF offsets start at the data */
    return rc ? rc : finish_tensors(l, unknown);
}

static ingot_dtype dtype_from_fs(fs_dtype d) {
    switch (d) {
    case FS_DTYPE_BF16: return INGOT_DTYPE_BF16;
    case FS_DTYPE_F16: return INGOT_DTYPE_F16;
    case FS_DTYPE_F32: return INGOT_DTYPE_F32;
    case FS_DTYPE_F64: return INGOT_DTYPE_F64;
    case FS_DTYPE_I8: return INGOT_DTYPE_I8;
    case FS_DTYPE_U8: return INGOT_DTYPE_U8;
    case FS_DTYPE_I16: return INGOT_DTYPE_I16;
    case FS_DTYPE_U16: return INGOT_DTYPE_U16;
    case FS_DTYPE_I32: return INGOT_DTYPE_I32;
    case FS_DTYPE_U32: return INGOT_DTYPE_U32;
    case FS_DTYPE_I64: return INGOT_DTYPE_I64;
    case FS_DTYPE_U64: return INGOT_DTYPE_U64;
    case FS_DTYPE_BOOL: return INGOT_DTYPE_BOOL;
    default: return INGOT_DTYPE_OTHER;
    }
}

static int safetensors_tensor(fs_file *fs, ingot_layout *l, size_t i, const char *name) {
    fs_tensor_info *info = fs_get_info(fs, name);
    if (!info || info->ndim < 0 || info->ndim > INGOT_MAX_DIMS) {
        fs_free_info(info);
        return -INGOT_EFORMAT;
    }
    if (!(l->names[i] = strdup(name))) {
        fs_free_info(info);
        return -INGOT_ENOMEM;
    }
    ingot_layout_tensor *t = &l->tensors[i];
    t->name = l->names[i];
    t->dtype = dtype_from_fs(info->dtype);
    t->ndim = info->ndim;
    for (int d = 0; d < info->ndim; d++)
        t->shape[d] = (uint64_t)info->shape[d];
    t->offset = l->header_size + info->data_offset;
    t->size = info->data_size;
    fs_free_info(info);
    return 0;
}

/* The 8-byte length is already in the buffer (it was read to tell the formats apart). */
static int read_safetensors(ingot_input *in, ingot_layout *l) {
    uint64_t json = 0, at = 0;
    memcpy(&json, l->header, 8);
    int rc = need(in, l, json, &at);
    if (rc)
        return rc == -INGOT_ETRUNCATED || json > MAX_HEADER ? -INGOT_EFORMAT : rc;
    fs_file *fs = fs_open_buffer(l->header, (size_t)l->header_size);
    if (!fs)
        return -INGOT_EFORMAT;
    int count = 0;
    const char **names = fs_tensor_names(fs, &count);
    rc = (count < 0 || (count && !names)) ? -INGOT_EFORMAT : 0;
    if (!rc) {
        l->count = (size_t)count;
        l->tensors = calloc(l->count ? l->count : 1, sizeof(*l->tensors));
        l->names = calloc(l->count ? l->count : 1, sizeof(*l->names));
        rc = (l->tensors && l->names) ? 0 : -INGOT_ENOMEM;
    }
    for (size_t i = 0; !rc && i < l->count; i++)
        rc = safetensors_tensor(fs, l, i, names[i]);
    free((void *)names);
    fs_close(fs);
    if (rc)
        return rc;
    return finish_tensors(l, 0);
}

int ingot_layout_read(ingot_input *in, ingot_layout **out) {
    if (!in || !out)
        return -INGOT_EINVAL;
    if (ingot_input_pos(in))
        return -INGOT_EINVAL;
    ingot_layout *l = calloc(1, sizeof(*l));
    if (!l)
        return -INGOT_ENOMEM;
    uint64_t at = 0;
    int rc = need(in, l, 8, &at);
    if (rc == -INGOT_ETRUNCATED)
        rc = -INGOT_EFORMAT;
    if (!rc && !memcmp(l->header, "GGUF", 4)) {
        l->kind = INGOT_SOURCE_GGUF;
        rc = read_gguf(in, l);
    } else if (!rc) {
        l->kind = INGOT_SOURCE_SAFETENSORS;
        rc = read_safetensors(in, l);
    }
    if (rc) {
        ingot_layout_free(l);
        return rc;
    }
    *out = l;
    return 0;
}

void ingot_layout_free(ingot_layout *l) {
    if (!l)
        return;
    for (size_t i = 0; l->names && i < l->count; i++)
        free(l->names[i]);
    free(l->names);
    free(l->tensors);
    free(l->header);
    free(l);
}

ingot_source_kind ingot_layout_kind(const ingot_layout *l) {
    return l ? l->kind : INGOT_SOURCE_UNKNOWN;
}

const uint8_t *ingot_layout_header(const ingot_layout *l, uint64_t *size) {
    *size = l->header_size;
    return l->header;
}

uint64_t ingot_layout_data_end(const ingot_layout *l) {
    if (!l->count)
        return l->header_size;
    const ingot_layout_tensor *t = &l->tensors[l->count - 1];
    return t->offset + t->size;
}

size_t ingot_layout_count(const ingot_layout *l) { return l ? l->count : 0; }

const ingot_layout_tensor *ingot_layout_tensor_at(const ingot_layout *l, size_t i) {
    return (l && i < l->count) ? &l->tensors[i] : NULL;
}

int ingot_layout_find(const ingot_layout *l, const char *name, size_t *index) {
    for (size_t i = 0; l && name && i < l->count; i++) {
        if (!strcmp(l->tensors[i].name, name)) {
            *index = i;
            return 0;
        }
    }
    return -INGOT_ENOTFOUND;
}

uint64_t ingot_layout_elements(const ingot_layout_tensor *t) {
    uint64_t n = 1;
    for (int d = 0; d < t->ndim; d++)
        n *= t->shape[d];
    return n;
}

const char *ingot_dtype_name(ingot_dtype dtype) {
    static const char *const NAMES[] = {"other", "BF16", "F16", "F32", "F64", "I8", "U8",
                                        "I16", "U16", "I32", "U32", "I64", "U64", "BOOL"};
    if ((unsigned)dtype < sizeof(NAMES) / sizeof(NAMES[0]))
        return NAMES[dtype];
    uint32_t type = 0;
    const char *name = NULL;
    if (ingot_dtype_ggml_type(dtype, &type) == 0)
        name = ingot_ggml_name(type);
    return name ? name : "other";
}
