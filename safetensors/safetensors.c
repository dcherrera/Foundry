/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* ============================================================= */
/* Internal structures                                           */
/* ============================================================= */

typedef struct {
    char* name;
    fs_dtype dtype;
    int64_t* shape;
    int ndim;
    size_t offset_start;
    size_t offset_end;
} tensor_entry;

struct fs_file {
    void* data;           /* Memory-mapped data or buffer */
    size_t size;          /* Total file size */
    bool is_mmap;         /* True if mmap'd, false if buffer */
    int fd;               /* File descriptor (if mmap'd) */

    char* header_json;    /* JSON header string */
    size_t header_len;    /* Header length */
    size_t data_offset;   /* Offset to tensor data */

    tensor_entry* tensors;
    int num_tensors;

    char** metadata_keys;
    char** metadata_values;
    int num_metadata;
};

typedef struct {
    char* name;
    fs_dtype dtype;
    int64_t* shape;
    int ndim;
    void* data;
    size_t data_size;
} writer_tensor;

struct fs_writer {
    writer_tensor* tensors;
    int num_tensors;
    int capacity;

    char** metadata_keys;
    char** metadata_values;
    int num_metadata;
    int metadata_capacity;
};

/* ============================================================= */
/* Dtype utilities                                               */
/* ============================================================= */

const char* fs_dtype_name(fs_dtype dtype) {
    switch (dtype) {
        case FS_DTYPE_F32: return "F32";
        case FS_DTYPE_F64: return "F64";
        case FS_DTYPE_F16: return "F16";
        case FS_DTYPE_BF16: return "BF16";
        case FS_DTYPE_I8: return "I8";
        case FS_DTYPE_I16: return "I16";
        case FS_DTYPE_I32: return "I32";
        case FS_DTYPE_I64: return "I64";
        case FS_DTYPE_U8: return "U8";
        case FS_DTYPE_U16: return "U16";
        case FS_DTYPE_U32: return "U32";
        case FS_DTYPE_U64: return "U64";
        case FS_DTYPE_BOOL: return "BOOL";
        default: return "UNKNOWN";
    }
}

size_t fs_dtype_size(fs_dtype dtype) {
    switch (dtype) {
        case FS_DTYPE_F32: return 4;
        case FS_DTYPE_F64: return 8;
        case FS_DTYPE_F16: return 2;
        case FS_DTYPE_BF16: return 2;
        case FS_DTYPE_I8: return 1;
        case FS_DTYPE_I16: return 2;
        case FS_DTYPE_I32: return 4;
        case FS_DTYPE_I64: return 8;
        case FS_DTYPE_U8: return 1;
        case FS_DTYPE_U16: return 2;
        case FS_DTYPE_U32: return 4;
        case FS_DTYPE_U64: return 8;
        case FS_DTYPE_BOOL: return 1;
        default: return 0;
    }
}

fs_dtype fs_dtype_from_string(const char* s) {
    if (!s) return FS_DTYPE_UNKNOWN;
    if (strcmp(s, "F32") == 0) return FS_DTYPE_F32;
    if (strcmp(s, "F64") == 0) return FS_DTYPE_F64;
    if (strcmp(s, "F16") == 0) return FS_DTYPE_F16;
    if (strcmp(s, "BF16") == 0) return FS_DTYPE_BF16;
    if (strcmp(s, "I8") == 0) return FS_DTYPE_I8;
    if (strcmp(s, "I16") == 0) return FS_DTYPE_I16;
    if (strcmp(s, "I32") == 0) return FS_DTYPE_I32;
    if (strcmp(s, "I64") == 0) return FS_DTYPE_I64;
    if (strcmp(s, "U8") == 0) return FS_DTYPE_U8;
    if (strcmp(s, "U16") == 0) return FS_DTYPE_U16;
    if (strcmp(s, "U32") == 0) return FS_DTYPE_U32;
    if (strcmp(s, "U64") == 0) return FS_DTYPE_U64;
    if (strcmp(s, "BOOL") == 0) return FS_DTYPE_BOOL;
    return FS_DTYPE_UNKNOWN;
}

/* ============================================================= */
/* Minimal JSON parser                                           */
/* ============================================================= */

static const char* skip_whitespace(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static char* parse_string(const char** pp) {
    const char* p = *pp;
    if (*p != '"') return NULL;
    p++;

    const char* start = p;
    while (*p && *p != '"') {
        if (*p == '\\') p++;  /* Skip escaped char */
        if (*p) p++;
    }

    size_t len = p - start;
    char* s = malloc(len + 1);
    if (!s) return NULL;

    /* Copy with escape handling */
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        if (start[i] == '\\' && i + 1 < len) {
            i++;
            switch (start[i]) {
                case 'n': s[j++] = '\n'; break;
                case 't': s[j++] = '\t'; break;
                case 'r': s[j++] = '\r'; break;
                case '"': s[j++] = '"'; break;
                case '\\': s[j++] = '\\'; break;
                default: s[j++] = start[i]; break;
            }
        } else {
            s[j++] = start[i];
        }
    }
    s[j] = '\0';

    if (*p == '"') p++;
    *pp = p;
    return s;
}

static int64_t parse_int(const char** pp) {
    const char* p = *pp;
    int64_t sign = 1;
    if (*p == '-') { sign = -1; p++; }

    int64_t val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    *pp = p;
    return sign * val;
}

static int64_t* parse_int_array(const char** pp, int* count) {
    const char* p = *pp;
    *count = 0;

    if (*p != '[') return NULL;
    p++;
    p = skip_whitespace(p);

    /* Count elements first */
    const char* scan = p;
    int n = 0;
    int depth = 0;
    while (*scan && !(*scan == ']' && depth == 0)) {
        if (*scan == '[') depth++;
        else if (*scan == ']') depth--;
        else if (*scan == ',' && depth == 0) n++;
        else if ((*scan >= '0' && *scan <= '9') || *scan == '-') {
            if (depth == 0 && (scan == p || *(scan-1) == ',' || *(scan-1) == '[' ||
                *(scan-1) == ' ' || *(scan-1) == '\t' || *(scan-1) == '\n')) {
                /* Already counted by comma */
            }
        }
        scan++;
    }

    /* Handle empty array */
    if (*p == ']') {
        *pp = p + 1;
        return malloc(1);  /* Non-null but zero elements */
    }

    /* Re-count properly */
    n = 1;
    scan = p;
    while (*scan && *scan != ']') {
        if (*scan == ',') n++;
        scan++;
    }

    int64_t* arr = malloc(n * sizeof(int64_t));
    if (!arr) return NULL;

    for (int i = 0; i < n && *p && *p != ']'; i++) {
        p = skip_whitespace(p);
        arr[i] = parse_int(&p);
        p = skip_whitespace(p);
        if (*p == ',') p++;
    }

    p = skip_whitespace(p);
    if (*p == ']') p++;
    *pp = p;
    *count = n;
    return arr;
}

/* Skip a JSON value (string, number, array, object, bool, null) */
static void skip_value(const char** pp) {
    const char* p = *pp;
    p = skip_whitespace(p);

    if (*p == '"') {
        p++;
        while (*p && *p != '"') {
            if (*p == '\\') p++;
            if (*p) p++;
        }
        if (*p == '"') p++;
    } else if (*p == '[') {
        int depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\') p++;
                    if (*p) p++;
                }
            }
            if (*p) p++;
        }
    } else if (*p == '{') {
        int depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\') p++;
                    if (*p) p++;
                }
            }
            if (*p) p++;
        }
    } else {
        /* Number, bool, null */
        while (*p && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
            p++;
        }
    }
    *pp = p;
}

/* ============================================================= */
/* Header parsing                                                */
/* ============================================================= */

static bool parse_header(fs_file* f) {
    const char* p = f->header_json;

    /* Count tensors and metadata */
    int tensor_count = 0;
    int meta_count = 0;

    p = skip_whitespace(p);
    if (*p != '{') return false;
    p++;

    /* First pass: count entries */
    const char* scan = p;
    while (*scan && *scan != '}') {
        scan = skip_whitespace(scan);
        if (*scan != '"') break;

        char* key = parse_string(&scan);
        if (!key) break;

        scan = skip_whitespace(scan);
        if (*scan == ':') scan++;
        scan = skip_whitespace(scan);

        if (strcmp(key, "__metadata__") == 0) {
            /* Count metadata entries */
            if (*scan == '{') {
                scan++;
                while (*scan && *scan != '}') {
                    scan = skip_whitespace(scan);
                    if (*scan != '"') break;
                    skip_value(&scan);
                    scan = skip_whitespace(scan);
                    if (*scan == ':') scan++;
                    skip_value(&scan);
                    scan = skip_whitespace(scan);
                    if (*scan == ',') scan++;
                    meta_count++;
                }
                if (*scan == '}') scan++;
            }
        } else {
            tensor_count++;
            skip_value(&scan);
        }

        free(key);
        scan = skip_whitespace(scan);
        if (*scan == ',') scan++;
    }

    /* Allocate storage */
    f->tensors = calloc(tensor_count, sizeof(tensor_entry));
    f->num_tensors = 0;

    if (meta_count > 0) {
        f->metadata_keys = calloc(meta_count, sizeof(char*));
        f->metadata_values = calloc(meta_count, sizeof(char*));
    }
    f->num_metadata = 0;

    /* Second pass: parse entries */
    p = f->header_json;
    p = skip_whitespace(p);
    if (*p == '{') p++;

    while (*p && *p != '}') {
        p = skip_whitespace(p);
        if (*p != '"') break;

        char* key = parse_string(&p);
        if (!key) break;

        p = skip_whitespace(p);
        if (*p == ':') p++;
        p = skip_whitespace(p);

        if (strcmp(key, "__metadata__") == 0) {
            /* Parse metadata object */
            if (*p == '{') {
                p++;
                while (*p && *p != '}') {
                    p = skip_whitespace(p);
                    if (*p != '"') break;

                    char* mkey = parse_string(&p);
                    p = skip_whitespace(p);
                    if (*p == ':') p++;
                    p = skip_whitespace(p);
                    char* mval = parse_string(&p);

                    if (mkey && mval && f->num_metadata < meta_count) {
                        f->metadata_keys[f->num_metadata] = mkey;
                        f->metadata_values[f->num_metadata] = mval;
                        f->num_metadata++;
                    } else {
                        free(mkey);
                        free(mval);
                    }

                    p = skip_whitespace(p);
                    if (*p == ',') p++;
                }
                if (*p == '}') p++;
            }
            free(key);
        } else {
            /* Parse tensor entry */
            tensor_entry* t = &f->tensors[f->num_tensors];
            t->name = key;

            if (*p == '{') {
                p++;
                while (*p && *p != '}') {
                    p = skip_whitespace(p);
                    if (*p != '"') break;

                    char* field = parse_string(&p);
                    p = skip_whitespace(p);
                    if (*p == ':') p++;
                    p = skip_whitespace(p);

                    if (strcmp(field, "dtype") == 0) {
                        char* dtype_str = parse_string(&p);
                        t->dtype = fs_dtype_from_string(dtype_str);
                        free(dtype_str);
                    } else if (strcmp(field, "shape") == 0) {
                        t->shape = parse_int_array(&p, &t->ndim);
                    } else if (strcmp(field, "data_offsets") == 0) {
                        int count;
                        int64_t* offsets = parse_int_array(&p, &count);
                        if (offsets && count >= 2) {
                            t->offset_start = (size_t)offsets[0];
                            t->offset_end = (size_t)offsets[1];
                        }
                        free(offsets);
                    } else {
                        skip_value(&p);
                    }

                    free(field);
                    p = skip_whitespace(p);
                    if (*p == ',') p++;
                }
                if (*p == '}') p++;
            }

            f->num_tensors++;
        }

        p = skip_whitespace(p);
        if (*p == ',') p++;
    }

    return true;
}

/* ============================================================= */
/* File operations                                               */
/* ============================================================= */

fs_file* fs_open(const char* path) {
    if (!path) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    size_t size = st.st_size;
    if (size < 8) {
        close(fd);
        return NULL;
    }

    void* data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        return NULL;
    }

    fs_file* f = calloc(1, sizeof(fs_file));
    if (!f) {
        munmap(data, size);
        close(fd);
        return NULL;
    }

    f->data = data;
    f->size = size;
    f->is_mmap = true;
    f->fd = fd;

    /* Read header length (little-endian u64) */
    const uint8_t* bytes = (const uint8_t*)data;
    uint64_t header_len = 0;
    for (int i = 0; i < 8; i++) {
        header_len |= ((uint64_t)bytes[i]) << (i * 8);
    }

    if (8 + header_len > size) {
        fs_close(f);
        return NULL;
    }

    f->header_len = (size_t)header_len;
    f->header_json = malloc(header_len + 1);
    if (!f->header_json) {
        fs_close(f);
        return NULL;
    }
    memcpy(f->header_json, bytes + 8, header_len);
    f->header_json[header_len] = '\0';

    f->data_offset = 8 + header_len;

    if (!parse_header(f)) {
        fs_close(f);
        return NULL;
    }

    return f;
}

fs_file* fs_open_buffer(const void* data, size_t size) {
    if (!data || size < 8) return NULL;

    fs_file* f = calloc(1, sizeof(fs_file));
    if (!f) return NULL;

    f->data = (void*)data;  /* Note: not copying */
    f->size = size;
    f->is_mmap = false;
    f->fd = -1;

    /* Read header length */
    const uint8_t* bytes = (const uint8_t*)data;
    uint64_t header_len = 0;
    for (int i = 0; i < 8; i++) {
        header_len |= ((uint64_t)bytes[i]) << (i * 8);
    }

    if (8 + header_len > size) {
        free(f);
        return NULL;
    }

    f->header_len = (size_t)header_len;
    f->header_json = malloc(header_len + 1);
    if (!f->header_json) {
        free(f);
        return NULL;
    }
    memcpy(f->header_json, bytes + 8, header_len);
    f->header_json[header_len] = '\0';

    f->data_offset = 8 + header_len;

    if (!parse_header(f)) {
        free(f->header_json);
        free(f);
        return NULL;
    }

    return f;
}

void fs_close(fs_file* f) {
    if (!f) return;

    if (f->is_mmap && f->data) {
        munmap(f->data, f->size);
    }
    if (f->fd >= 0) {
        close(f->fd);
    }

    free(f->header_json);

    for (int i = 0; i < f->num_tensors; i++) {
        free(f->tensors[i].name);
        free(f->tensors[i].shape);
    }
    free(f->tensors);

    for (int i = 0; i < f->num_metadata; i++) {
        free(f->metadata_keys[i]);
        free(f->metadata_values[i]);
    }
    free(f->metadata_keys);
    free(f->metadata_values);

    free(f);
}

/* ============================================================= */
/* Tensor queries                                                */
/* ============================================================= */

int fs_num_tensors(fs_file* f) {
    return f ? f->num_tensors : 0;
}

const char** fs_tensor_names(fs_file* f, int* count) {
    if (!f || !count) return NULL;

    const char** names = malloc(f->num_tensors * sizeof(char*));
    if (!names) return NULL;

    for (int i = 0; i < f->num_tensors; i++) {
        names[i] = f->tensors[i].name;
    }
    *count = f->num_tensors;
    return names;
}

bool fs_has_tensor(fs_file* f, const char* name) {
    if (!f || !name) return false;

    for (int i = 0; i < f->num_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            return true;
        }
    }
    return false;
}

static tensor_entry* find_tensor(fs_file* f, const char* name) {
    if (!f || !name) return NULL;

    for (int i = 0; i < f->num_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            return &f->tensors[i];
        }
    }
    return NULL;
}

fs_tensor_info* fs_get_info(fs_file* f, const char* name) {
    tensor_entry* t = find_tensor(f, name);
    if (!t) return NULL;

    fs_tensor_info* info = calloc(1, sizeof(fs_tensor_info));
    if (!info) return NULL;

    info->name = strdup(t->name);
    info->dtype = t->dtype;
    info->ndim = t->ndim;

    if (t->ndim > 0) {
        info->shape = malloc(t->ndim * sizeof(int64_t));
        if (info->shape) {
            memcpy(info->shape, t->shape, t->ndim * sizeof(int64_t));
        }
    }

    info->data_offset = t->offset_start;
    info->data_size = t->offset_end - t->offset_start;

    return info;
}

void fs_free_info(fs_tensor_info* info) {
    if (!info) return;
    free(info->name);
    free(info->shape);
    free(info);
}

const void* fs_get_data(fs_file* f, const char* name) {
    tensor_entry* t = find_tensor(f, name);
    if (!t) return NULL;

    return (const uint8_t*)f->data + f->data_offset + t->offset_start;
}

const float* fs_get_f32(fs_file* f, const char* name) {
    tensor_entry* t = find_tensor(f, name);
    if (!t || t->dtype != FS_DTYPE_F32) return NULL;
    return (const float*)((const uint8_t*)f->data + f->data_offset + t->offset_start);
}

const double* fs_get_f64(fs_file* f, const char* name) {
    tensor_entry* t = find_tensor(f, name);
    if (!t || t->dtype != FS_DTYPE_F64) return NULL;
    return (const double*)((const uint8_t*)f->data + f->data_offset + t->offset_start);
}

const int32_t* fs_get_i32(fs_file* f, const char* name) {
    tensor_entry* t = find_tensor(f, name);
    if (!t || t->dtype != FS_DTYPE_I32) return NULL;
    return (const int32_t*)((const uint8_t*)f->data + f->data_offset + t->offset_start);
}

const int64_t* fs_get_i64(fs_file* f, const char* name) {
    tensor_entry* t = find_tensor(f, name);
    if (!t || t->dtype != FS_DTYPE_I64) return NULL;
    return (const int64_t*)((const uint8_t*)f->data + f->data_offset + t->offset_start);
}

const uint8_t* fs_get_u8(fs_file* f, const char* name) {
    tensor_entry* t = find_tensor(f, name);
    if (!t || t->dtype != FS_DTYPE_U8) return NULL;
    return (const uint8_t*)f->data + f->data_offset + t->offset_start;
}

/* ============================================================= */
/* Metadata                                                      */
/* ============================================================= */

const char* fs_get_metadata(fs_file* f, const char* key) {
    if (!f || !key) return NULL;

    for (int i = 0; i < f->num_metadata; i++) {
        if (strcmp(f->metadata_keys[i], key) == 0) {
            return f->metadata_values[i];
        }
    }
    return NULL;
}

const char** fs_metadata_keys(fs_file* f, int* count) {
    if (!f || !count) return NULL;

    const char** keys = malloc(f->num_metadata * sizeof(char*));
    if (!keys) return NULL;

    for (int i = 0; i < f->num_metadata; i++) {
        keys[i] = f->metadata_keys[i];
    }
    *count = f->num_metadata;
    return keys;
}

/* ============================================================= */
/* Writer                                                        */
/* ============================================================= */

fs_writer* fs_writer_new(void) {
    fs_writer* w = calloc(1, sizeof(fs_writer));
    if (!w) return NULL;

    w->capacity = 16;
    w->tensors = calloc(w->capacity, sizeof(writer_tensor));
    w->metadata_capacity = 8;
    w->metadata_keys = calloc(w->metadata_capacity, sizeof(char*));
    w->metadata_values = calloc(w->metadata_capacity, sizeof(char*));

    return w;
}

void fs_writer_free(fs_writer* w) {
    if (!w) return;

    for (int i = 0; i < w->num_tensors; i++) {
        free(w->tensors[i].name);
        free(w->tensors[i].shape);
        free(w->tensors[i].data);
    }
    free(w->tensors);

    for (int i = 0; i < w->num_metadata; i++) {
        free(w->metadata_keys[i]);
        free(w->metadata_values[i]);
    }
    free(w->metadata_keys);
    free(w->metadata_values);

    free(w);
}

void fs_writer_add_metadata(fs_writer* w, const char* key, const char* value) {
    if (!w || !key || !value) return;

    if (w->num_metadata >= w->metadata_capacity) {
        w->metadata_capacity *= 2;
        w->metadata_keys = realloc(w->metadata_keys, w->metadata_capacity * sizeof(char*));
        w->metadata_values = realloc(w->metadata_values, w->metadata_capacity * sizeof(char*));
    }

    w->metadata_keys[w->num_metadata] = strdup(key);
    w->metadata_values[w->num_metadata] = strdup(value);
    w->num_metadata++;
}

void fs_writer_add_tensor(fs_writer* w, const char* name, fs_dtype dtype,
                          const int64_t* shape, int ndim, const void* data) {
    if (!w || !name || !data) return;

    if (w->num_tensors >= w->capacity) {
        w->capacity *= 2;
        w->tensors = realloc(w->tensors, w->capacity * sizeof(writer_tensor));
    }

    writer_tensor* t = &w->tensors[w->num_tensors];
    t->name = strdup(name);
    t->dtype = dtype;
    t->ndim = ndim;

    if (ndim > 0) {
        t->shape = malloc(ndim * sizeof(int64_t));
        memcpy(t->shape, shape, ndim * sizeof(int64_t));
    } else {
        t->shape = NULL;
    }

    /* Calculate data size */
    size_t elem_size = fs_dtype_size(dtype);
    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        numel *= shape[i];
    }
    t->data_size = numel * elem_size;

    t->data = malloc(t->data_size);
    memcpy(t->data, data, t->data_size);

    w->num_tensors++;
}

void fs_writer_add_f32(fs_writer* w, const char* name,
                       const int64_t* shape, int ndim, const float* data) {
    fs_writer_add_tensor(w, name, FS_DTYPE_F32, shape, ndim, data);
}

static char* build_header_json(fs_writer* w, size_t* offsets) {
    /* Estimate size needed */
    size_t size = 256;
    for (int i = 0; i < w->num_tensors; i++) {
        size += strlen(w->tensors[i].name) + 128;
    }
    for (int i = 0; i < w->num_metadata; i++) {
        size += strlen(w->metadata_keys[i]) + strlen(w->metadata_values[i]) + 16;
    }

    char* json = malloc(size);
    if (!json) return NULL;

    char* p = json;
    char* end = json + size;
    *p++ = '{';

    bool first = true;

    /* Metadata */
    if (w->num_metadata > 0) {
        p += snprintf(p, end - p, "\"__metadata__\":{");
        for (int i = 0; i < w->num_metadata; i++) {
            if (i > 0) *p++ = ',';
            p += snprintf(p, end - p, "\"%s\":\"%s\"", w->metadata_keys[i], w->metadata_values[i]);
        }
        *p++ = '}';
        first = false;
    }

    /* Tensors */
    size_t current_offset = 0;
    for (int i = 0; i < w->num_tensors; i++) {
        writer_tensor* t = &w->tensors[i];
        offsets[i] = current_offset;

        if (!first) *p++ = ',';
        first = false;

        p += snprintf(p, end - p, "\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                      t->name, fs_dtype_name(t->dtype));

        for (int j = 0; j < t->ndim; j++) {
            if (j > 0) *p++ = ',';
            p += snprintf(p, end - p, "%lld", (long long)t->shape[j]);
        }

        p += snprintf(p, end - p, "],\"data_offsets\":[%zu,%zu]}",
                      current_offset, current_offset + t->data_size);

        current_offset += t->data_size;
    }

    *p++ = '}';
    *p = '\0';

    return json;
}

void* fs_writer_to_buffer(fs_writer* w, size_t* size) {
    if (!w || !size) return NULL;

    size_t* offsets = malloc(w->num_tensors * sizeof(size_t));
    if (!offsets) return NULL;

    char* header = build_header_json(w, offsets);
    if (!header) {
        free(offsets);
        return NULL;
    }

    size_t header_len = strlen(header);

    /* Calculate total data size */
    size_t data_size = 0;
    for (int i = 0; i < w->num_tensors; i++) {
        data_size += w->tensors[i].data_size;
    }

    size_t total = 8 + header_len + data_size;
    uint8_t* buf = malloc(total);
    if (!buf) {
        free(header);
        free(offsets);
        return NULL;
    }

    /* Write header length (little-endian u64) */
    for (int i = 0; i < 8; i++) {
        buf[i] = (header_len >> (i * 8)) & 0xFF;
    }

    /* Write header */
    memcpy(buf + 8, header, header_len);

    /* Write tensor data */
    uint8_t* data_ptr = buf + 8 + header_len;
    for (int i = 0; i < w->num_tensors; i++) {
        memcpy(data_ptr + offsets[i], w->tensors[i].data, w->tensors[i].data_size);
    }

    free(header);
    free(offsets);

    *size = total;
    return buf;
}

bool fs_writer_save(fs_writer* w, const char* path) {
    if (!w || !path) return false;

    size_t size;
    void* buf = fs_writer_to_buffer(w, &size);
    if (!buf) return false;

    FILE* f = fopen(path, "wb");
    if (!f) {
        free(buf);
        return false;
    }

    size_t written = fwrite(buf, 1, size, f);
    fclose(f);
    free(buf);

    return written == size;
}

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

void fs_print(fs_file* f) {
    if (!f) {
        printf("(null safetensors file)\n");
        return;
    }

    printf("Safetensors file: %zu bytes, %d tensors\n", f->size, f->num_tensors);

    if (f->num_metadata > 0) {
        printf("Metadata:\n");
        for (int i = 0; i < f->num_metadata; i++) {
            printf("  %s: %s\n", f->metadata_keys[i], f->metadata_values[i]);
        }
    }

    printf("Tensors:\n");
    for (int i = 0; i < f->num_tensors; i++) {
        tensor_entry* t = &f->tensors[i];
        printf("  %s: %s [", t->name, fs_dtype_name(t->dtype));
        for (int j = 0; j < t->ndim; j++) {
            if (j > 0) printf(", ");
            printf("%lld", (long long)t->shape[j]);
        }
        printf("] (%zu bytes)\n", t->offset_end - t->offset_start);
    }
}
