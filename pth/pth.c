/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * PyTorch .pth file reader
 *
 * A .pth file is a ZIP archive containing:
 *   archive/data.pkl       - Pickle: OrderedDict mapping tensor names -> storage refs
 *   archive/data/0         - Raw bytes for storage #0
 *   archive/data/1         - Raw bytes for storage #1
 *   ...
 *
 * Three sub-components:
 *   1. ZIP reader     - parse ZIP central directory, locate entries
 *   2. Pickle parser  - interpret pickle opcodes to extract tensor metadata
 *   3. Storage resolver - map storage IDs to ZIP entries, populate pt_tensor structs
 */

#include "pth/pth.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ============================================================= */
/* ZIP reader                                                     */
/* ============================================================= */

/* ZIP signatures */
#define ZIP_EOCD_SIG  0x06054b50
#define ZIP_CD_SIG    0x02014b50
#define ZIP_LF_SIG    0x04034b50

typedef struct {
    char name[512];
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint32_t local_header_offset;
    const uint8_t *data;   /* pointer to actual file data (after local header) */
    size_t data_size;
} zip_entry;

typedef struct {
    uint8_t *base;         /* mmap'd file */
    size_t file_size;
    zip_entry *entries;
    int n_entries;
} zip_archive;

static uint16_t read_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static zip_archive *zip_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    size_t file_size = (size_t)st.st_size;

    if (file_size < 22) { close(fd); return NULL; } /* min EOCD size */

    uint8_t *base = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return NULL;

    /* Find End of Central Directory (scan backwards) */
    const uint8_t *eocd = NULL;
    size_t search_start = file_size > 65557 ? file_size - 65557 : 0;
    for (size_t i = file_size - 22; i >= search_start; i--) {
        if (read_u32(base + i) == ZIP_EOCD_SIG) {
            eocd = base + i;
            break;
        }
        if (i == 0) break;
    }
    if (!eocd) { munmap(base, file_size); return NULL; }

    uint16_t n_entries = read_u16(eocd + 10);
    uint32_t cd_offset = read_u32(eocd + 16);

    if (cd_offset >= file_size) { munmap(base, file_size); return NULL; }

    /* Parse central directory */
    zip_entry *entries = calloc(n_entries, sizeof(zip_entry));
    if (!entries) { munmap(base, file_size); return NULL; }

    const uint8_t *p = base + cd_offset;
    for (int i = 0; i < n_entries; i++) {
        if (p + 46 > base + file_size) goto fail;
        if (read_u32(p) != ZIP_CD_SIG) goto fail;

        uint16_t name_len = read_u16(p + 28);
        uint16_t extra_len = read_u16(p + 30);
        uint16_t comment_len = read_u16(p + 32);
        entries[i].compressed_size = read_u32(p + 20);
        entries[i].uncompressed_size = read_u32(p + 24);
        entries[i].local_header_offset = read_u32(p + 42);

        if (p + 46 + name_len > base + file_size) goto fail;
        size_t copy_len = name_len < 511 ? name_len : 511;
        memcpy(entries[i].name, p + 46, copy_len);
        entries[i].name[copy_len] = '\0';

        /* Resolve data pointer from local file header */
        uint32_t lf_off = entries[i].local_header_offset;
        if (lf_off + 30 <= file_size) {
            const uint8_t *lf = base + lf_off;
            if (read_u32(lf) == ZIP_LF_SIG) {
                uint16_t lf_name_len = read_u16(lf + 26);
                uint16_t lf_extra_len = read_u16(lf + 28);
                size_t data_off = lf_off + 30 + lf_name_len + lf_extra_len;
                if (data_off <= file_size) {
                    entries[i].data = base + data_off;
                    entries[i].data_size = entries[i].uncompressed_size;
                }
            }
        }

        p += 46 + name_len + extra_len + comment_len;
    }

    zip_archive *z = malloc(sizeof(zip_archive));
    if (!z) goto fail;
    z->base = base;
    z->file_size = file_size;
    z->entries = entries;
    z->n_entries = n_entries;
    return z;

fail:
    free(entries);
    munmap(base, file_size);
    return NULL;
}

static const zip_entry *zip_find(const zip_archive *z, const char *name) {
    for (int i = 0; i < z->n_entries; i++) {
        if (strcmp(z->entries[i].name, name) == 0) return &z->entries[i];
    }
    return NULL;
}

/* Find entry by suffix (e.g. "data.pkl" matches "archive/data.pkl") */
static const zip_entry *zip_find_suffix(const zip_archive *z, const char *suffix) {
    size_t slen = strlen(suffix);
    for (int i = 0; i < z->n_entries; i++) {
        size_t nlen = strlen(z->entries[i].name);
        if (nlen >= slen && strcmp(z->entries[i].name + nlen - slen, suffix) == 0)
            return &z->entries[i];
    }
    return NULL;
}

static void zip_close(zip_archive *z) {
    if (!z) return;
    munmap(z->base, z->file_size);
    free(z->entries);
    free(z);
}

/* ============================================================= */
/* Pickle parser                                                  */
/* ============================================================= */

/* Pickle opcodes we need */
#define PK_PROTO          0x80
#define PK_FRAME          0x95
#define PK_STOP           0x2e  /* . */
#define PK_MARK           0x28  /* ( */
#define PK_POP            0x30  /* 0 */
#define PK_DUP            0x32  /* 2 */
#define PK_NONE           0x4e  /* N */
#define PK_NEWTRUE        0x88
#define PK_NEWFALSE       0x89
#define PK_EMPTY_TUPLE    0x29  /* ) */
#define PK_TUPLE1         0x85
#define PK_TUPLE2         0x86
#define PK_TUPLE3         0x87
#define PK_TUPLE          0x74  /* t */
#define PK_EMPTY_LIST     0x5d  /* ] */
#define PK_EMPTY_DICT     0x7d  /* } */
#define PK_APPEND         0x61  /* a */
#define PK_APPENDS        0x65  /* e */
#define PK_SETITEM        0x73  /* s */
#define PK_SETITEMS       0x75  /* u */
#define PK_SHORT_BINUNICODE 0x8c
#define PK_BINUNICODE     0x58  /* X */
#define PK_BININT1        0x4b  /* K */
#define PK_BININT2        0x4d  /* M */
#define PK_BININT         0x4a  /* J */
#define PK_LONG1          0x8a
#define PK_BINFLOAT       0x47  /* G */
#define PK_BINPERSID      0x51  /* Q */
#define PK_REDUCE         0x52  /* R */
#define PK_GLOBAL         0x63  /* c */
#define PK_STACK_GLOBAL   0x93
#define PK_BUILD          0x62  /* b */
#define PK_SHORT_BINSTRING 0x55  /* U */
#define PK_BINGET         0x68  /* h */
#define PK_LONG_BINGET    0x6a  /* j */
#define PK_BINPUT         0x71  /* q */
#define PK_LONG_BINPUT    0x72  /* r */
#define PK_MEMOIZE        0x94
#define PK_SHORT_BINBYTES 0x42  /* B */
#define PK_BINBYTES       0x43  /* C */

/* Pickle stack item types */
enum pk_type {
    PK_T_NONE,
    PK_T_BOOL,
    PK_T_INT,
    PK_T_FLOAT,
    PK_T_STRING,
    PK_T_TUPLE,
    PK_T_LIST,
    PK_T_DICT,
    PK_T_GLOBAL,       /* module.class reference */
    PK_T_PERSID,       /* persistent ID (storage reference) */
    PK_T_MARK,
    PK_T_BYTES,
};

typedef struct pk_item pk_item;

struct pk_item {
    enum pk_type type;
    union {
        int64_t ival;
        double fval;
        char sval[512];
        struct {
            pk_item *items;
            int count;
        } seq;
    };
};

/* Extracted tensor info from pickle */
typedef struct {
    char name[256];
    char storage_class[64];   /* e.g. "BFloat16Storage" */
    int64_t storage_id;       /* numeric ID */
    char storage_key[64];     /* string key like "0", "1", etc. */
    int64_t shape[8];
    int ndim;
    int64_t offset;           /* byte offset into storage */
} pkl_tensor_info;

typedef struct {
    pkl_tensor_info *tensors;
    int n_tensors;
    int cap_tensors;
} pkl_result;

#define PK_STACK_MAX 2048
#define PK_MEMO_MAX 1024

typedef struct {
    pk_item stack[PK_STACK_MAX];
    int sp;
    pk_item memo[PK_MEMO_MAX];
    int memo_used[PK_MEMO_MAX];
    pkl_result result;
    const uint8_t *data;
    size_t size;
    size_t pos;
} pk_state;

static void pk_push(pk_state *s, pk_item item) {
    if (s->sp < PK_STACK_MAX) s->stack[s->sp++] = item;
}

static pk_item pk_pop(pk_state *s) {
    if (s->sp > 0) return s->stack[--s->sp];
    pk_item empty = {0};
    return empty;
}

static int pk_find_mark(pk_state *s) {
    for (int i = s->sp - 1; i >= 0; i--) {
        if (s->stack[i].type == PK_T_MARK) return i;
    }
    return -1;
}

static void pkl_add_tensor(pkl_result *r, pkl_tensor_info *t) {
    if (r->n_tensors >= r->cap_tensors) {
        r->cap_tensors = r->cap_tensors ? r->cap_tensors * 2 : 64;
        r->tensors = realloc(r->tensors, r->cap_tensors * sizeof(pkl_tensor_info));
    }
    r->tensors[r->n_tensors++] = *t;
}

/* Read a string from the pickle stream */
static void pk_read_string(pk_state *s, char *buf, size_t bufsize, size_t len) {
    size_t copy = len < bufsize - 1 ? len : bufsize - 1;
    if (s->pos + len <= s->size) {
        memcpy(buf, s->data + s->pos, copy);
    }
    buf[copy] = '\0';
    s->pos += len;
}

/* Process BINPERSID: top of stack is a tuple describing a storage.
 * PyTorch persistent ID tuple layout:
 *   ("storage", torch.XxxStorage, "key", "device", size)
 * Items: [0]=STRING "storage", [1]=GLOBAL torch.XxxStorage,
 *        [2]=STRING key ("0","1",...), [3]=STRING device, [4]=INT size
 *
 * We encode into PERSID.sval as "key:StorageClassName" */
static void pk_handle_persid(pk_state *s) {
    pk_item pid = pk_pop(s);
    pk_item result = {0};
    result.type = PK_T_PERSID;
    if (pid.type == PK_T_TUPLE && pid.seq.count >= 3) {
        /* Find the GLOBAL (storage class) and the key string */
        const char *storage_class = NULL;
        const char *storage_key = NULL;
        for (int i = 0; i < pid.seq.count; i++) {
            if (pid.seq.items[i].type == PK_T_GLOBAL && !storage_class)
                storage_class = pid.seq.items[i].sval;
            else if (pid.seq.items[i].type == PK_T_STRING) {
                /* Skip "storage" and "cpu"/"cuda" — key is the short numeric string */
                const char *sv = pid.seq.items[i].sval;
                if (strcmp(sv, "storage") != 0 && strcmp(sv, "cpu") != 0 &&
                    strncmp(sv, "cuda", 4) != 0 && !storage_key)
                    storage_key = sv;
            }
        }
        if (storage_key && storage_class) {
            snprintf(result.sval, sizeof(result.sval), "%s:%s",
                     storage_key, storage_class);
        }
        free(pid.seq.items);
    }
    pk_push(s, result);
}


/* The actual tensor extraction happens during BUILD/REDUCE when we see
 * _rebuild_tensor_v2 calls. The dict gives us name -> storage mapping. */


static int pk_parse(pk_state *s) {
    while (s->pos < s->size) {
        uint8_t op = s->data[s->pos++];

        switch (op) {
        case PK_PROTO: {
            if (s->pos >= s->size) return -1;
            s->pos++; /* skip version byte */
            break;
        }
        case PK_FRAME: {
            s->pos += 8; /* skip frame length */
            break;
        }
        case PK_STOP:
            return 0;

        case PK_MARK: {
            pk_item m = {0};
            m.type = PK_T_MARK;
            pk_push(s, m);
            break;
        }
        case PK_POP:
            pk_pop(s);
            break;

        case PK_DUP:
            if (s->sp > 0) pk_push(s, s->stack[s->sp - 1]);
            break;

        case PK_NONE: {
            pk_item n = {0};
            n.type = PK_T_NONE;
            pk_push(s, n);
            break;
        }
        case PK_NEWTRUE: {
            pk_item b = {0};
            b.type = PK_T_BOOL;
            b.ival = 1;
            pk_push(s, b);
            break;
        }
        case PK_NEWFALSE: {
            pk_item b = {0};
            b.type = PK_T_BOOL;
            b.ival = 0;
            pk_push(s, b);
            break;
        }

        case PK_EMPTY_TUPLE: {
            pk_item t = {0};
            t.type = PK_T_TUPLE;
            t.seq.items = NULL;
            t.seq.count = 0;
            pk_push(s, t);
            break;
        }
        case PK_TUPLE1: {
            pk_item a = pk_pop(s);
            pk_item t = {0};
            t.type = PK_T_TUPLE;
            t.seq.items = malloc(sizeof(pk_item));
            t.seq.count = 1;
            t.seq.items[0] = a;
            pk_push(s, t);
            break;
        }
        case PK_TUPLE2: {
            pk_item b = pk_pop(s);
            pk_item a = pk_pop(s);
            pk_item t = {0};
            t.type = PK_T_TUPLE;
            t.seq.items = malloc(2 * sizeof(pk_item));
            t.seq.count = 2;
            t.seq.items[0] = a;
            t.seq.items[1] = b;
            pk_push(s, t);
            break;
        }
        case PK_TUPLE3: {
            pk_item c = pk_pop(s);
            pk_item b = pk_pop(s);
            pk_item a = pk_pop(s);
            pk_item t = {0};
            t.type = PK_T_TUPLE;
            t.seq.items = malloc(3 * sizeof(pk_item));
            t.seq.count = 3;
            t.seq.items[0] = a;
            t.seq.items[1] = b;
            t.seq.items[2] = c;
            pk_push(s, t);
            break;
        }
        case PK_TUPLE: {
            int mark = pk_find_mark(s);
            if (mark < 0) break;
            int count = s->sp - mark - 1;
            pk_item t = {0};
            t.type = PK_T_TUPLE;
            t.seq.count = count;
            t.seq.items = count > 0 ? malloc(count * sizeof(pk_item)) : NULL;
            for (int i = 0; i < count; i++)
                t.seq.items[i] = s->stack[mark + 1 + i];
            s->sp = mark;
            pk_push(s, t);
            break;
        }

        case PK_EMPTY_LIST: {
            pk_item l = {0};
            l.type = PK_T_LIST;
            pk_push(s, l);
            break;
        }
        case PK_EMPTY_DICT: {
            pk_item d = {0};
            d.type = PK_T_DICT;
            pk_push(s, d);
            break;
        }

        case PK_APPEND:
            pk_pop(s); /* discard appended item, we don't need list contents */
            break;

        case PK_APPENDS: {
            int mark = pk_find_mark(s);
            if (mark >= 0) s->sp = mark; /* pop mark + all items above */
            break;
        }

        case PK_SETITEM: {
            pk_item val = pk_pop(s);
            pk_item key = pk_pop(s);
            /* If this is a state_dict entry, assign name to the last tensor */
            if (key.type == PK_T_STRING && s->result.n_tensors > 0) {
                pkl_tensor_info *last = &s->result.tensors[s->result.n_tensors - 1];
                if (last->name[0] == '\0') {
                    strncpy(last->name, key.sval, sizeof(last->name) - 1);
                }
            }
            (void)val;
            break;
        }
        case PK_SETITEMS: {
            int mark = pk_find_mark(s);
            if (mark < 0) break;
            /* Walk key-value pairs and assign tensor names.
             * Stack between mark+1 and sp has pairs: key, value, key, value, ...
             * But the stack may have overflowed (PK_STACK_MAX=256) if there are
             * many tensors. In that case, items were silently dropped. */
            int name_idx = 0;
            for (int i = mark + 1; i + 1 < s->sp; i += 2) {
                if (s->stack[i].type == PK_T_STRING) {
                    /* Find next unnamed tensor and assign */
                    for (int t = name_idx; t < s->result.n_tensors; t++) {
                        if (s->result.tensors[t].name[0] == '\0') {
                            strncpy(s->result.tensors[t].name, s->stack[i].sval,
                                    sizeof(s->result.tensors[t].name) - 1);
                            name_idx = t + 1;
                            break;
                        }
                    }
                }
            }
            s->sp = mark; /* pop mark + all items */
            break;
        }

        case PK_SHORT_BINUNICODE: {
            if (s->pos >= s->size) return -1;
            uint8_t len = s->data[s->pos++];
            pk_item str = {0};
            str.type = PK_T_STRING;
            pk_read_string(s, str.sval, sizeof(str.sval), len);
            pk_push(s, str);
            break;
        }
        case PK_BINUNICODE: {
            if (s->pos + 4 > s->size) return -1;
            uint32_t len = read_u32(s->data + s->pos);
            s->pos += 4;
            pk_item str = {0};
            str.type = PK_T_STRING;
            pk_read_string(s, str.sval, sizeof(str.sval), len);
            pk_push(s, str);
            break;
        }
        case PK_SHORT_BINSTRING: {
            if (s->pos >= s->size) return -1;
            uint8_t len = s->data[s->pos++];
            pk_item str = {0};
            str.type = PK_T_STRING;
            pk_read_string(s, str.sval, sizeof(str.sval), len);
            pk_push(s, str);
            break;
        }

        case PK_BININT1: {
            if (s->pos >= s->size) return -1;
            pk_item n = {0};
            n.type = PK_T_INT;
            n.ival = s->data[s->pos++];
            pk_push(s, n);
            break;
        }
        case PK_BININT2: {
            if (s->pos + 2 > s->size) return -1;
            pk_item n = {0};
            n.type = PK_T_INT;
            n.ival = read_u16(s->data + s->pos);
            s->pos += 2;
            pk_push(s, n);
            break;
        }
        case PK_BININT: {
            if (s->pos + 4 > s->size) return -1;
            pk_item n = {0};
            n.type = PK_T_INT;
            int32_t v;
            memcpy(&v, s->data + s->pos, 4);
            n.ival = v;
            s->pos += 4;
            pk_push(s, n);
            break;
        }
        case PK_LONG1: {
            if (s->pos >= s->size) return -1;
            uint8_t nbytes = s->data[s->pos++];
            pk_item n = {0};
            n.type = PK_T_INT;
            n.ival = 0;
            if (s->pos + nbytes <= s->size) {
                /* Little-endian, possibly signed */
                for (int i = 0; i < nbytes && i < 8; i++)
                    n.ival |= (int64_t)s->data[s->pos + i] << (i * 8);
                /* Sign extend if negative */
                if (nbytes > 0 && nbytes < 8 && (s->data[s->pos + nbytes - 1] & 0x80))
                    for (int i = nbytes; i < 8; i++)
                        n.ival |= (int64_t)0xFF << (i * 8);
            }
            s->pos += nbytes;
            pk_push(s, n);
            break;
        }
        case PK_BINFLOAT: {
            if (s->pos + 8 > s->size) return -1;
            pk_item f = {0};
            f.type = PK_T_FLOAT;
            /* Big-endian double */
            uint8_t tmp[8];
            for (int i = 0; i < 8; i++) tmp[i] = s->data[s->pos + 7 - i];
            memcpy(&f.fval, tmp, 8);
            s->pos += 8;
            pk_push(s, f);
            break;
        }

        case PK_SHORT_BINBYTES: {
            if (s->pos >= s->size) return -1;
            uint8_t len = s->data[s->pos++];
            pk_item b = {0};
            b.type = PK_T_BYTES;
            pk_read_string(s, b.sval, sizeof(b.sval), len);
            pk_push(s, b);
            break;
        }
        case PK_BINBYTES: {
            if (s->pos + 4 > s->size) return -1;
            uint32_t len = read_u32(s->data + s->pos);
            s->pos += 4;
            pk_item b = {0};
            b.type = PK_T_BYTES;
            pk_read_string(s, b.sval, sizeof(b.sval), len);
            pk_push(s, b);
            break;
        }

        case PK_GLOBAL: {
            /* Read "module\nclass\n" */
            pk_item g = {0};
            g.type = PK_T_GLOBAL;
            char module[256] = {0};
            char cls[256] = {0};
            int mi = 0, ci = 0;
            /* Read module name until newline */
            while (s->pos < s->size && s->data[s->pos] != '\n' && mi < 255)
                module[mi++] = (char)s->data[s->pos++];
            if (s->pos < s->size) s->pos++; /* skip \n */
            /* Read class name until newline */
            while (s->pos < s->size && s->data[s->pos] != '\n' && ci < 255)
                cls[ci++] = (char)s->data[s->pos++];
            if (s->pos < s->size) s->pos++; /* skip \n */
            snprintf(g.sval, sizeof(g.sval), "%s.%s", module, cls);

            /* Security: only allow known torch/collections classes */
            if (strstr(g.sval, "torch") == NULL &&
                strstr(g.sval, "collections") == NULL &&
                strstr(g.sval, "_codecs") == NULL) {
                fprintf(stderr, "pth: blocked GLOBAL: %s\n", g.sval);
                return -1;
            }
            pk_push(s, g);
            break;
        }

        case PK_STACK_GLOBAL: {
            /* Pop class name, pop module name, push global */
            pk_item cls = pk_pop(s);
            pk_item mod = pk_pop(s);
            pk_item g = {0};
            g.type = PK_T_GLOBAL;
            if (mod.type == PK_T_STRING && cls.type == PK_T_STRING)
                snprintf(g.sval, sizeof(g.sval), "%s.%s", mod.sval, cls.sval);
            pk_push(s, g);
            break;
        }

        case PK_REDUCE: {
            /* Pop args tuple, pop callable, call callable(args) */
            pk_item args = pk_pop(s);
            pk_item callable = pk_pop(s);

            if (callable.type == PK_T_GLOBAL &&
                strstr(callable.sval, "_rebuild_tensor_v2")) {
                /* args is a tuple: (storage, offset, shape, stride, ...) */
                /* storage is a PERSID, shape is a tuple of ints */
                if (args.type == PK_T_TUPLE && args.seq.count >= 3) {
                    pkl_tensor_info ti = {0};
                    /* Name will be assigned later by SETITEM */

                    /* Storage info from persistent ID (item 0) */
                    pk_item *storage = &args.seq.items[0];
                    if (storage->type == PK_T_PERSID) {
                        /* sval is "key:torch.XxxStorage" */
                        char *colon = strchr(storage->sval, ':');
                        if (colon) {
                            *colon = '\0';
                            strncpy(ti.storage_key, storage->sval, sizeof(ti.storage_key) - 1);
                            strncpy(ti.storage_class, colon + 1, sizeof(ti.storage_class) - 1);
                            /* Extract just the storage class name */
                            char *dot = strrchr(ti.storage_class, '.');
                            if (dot) {
                                memmove(ti.storage_class, dot + 1, strlen(dot + 1) + 1);
                            }
                        }
                    }

                    /* Offset (item 1) */
                    if (args.seq.items[1].type == PK_T_INT)
                        ti.offset = args.seq.items[1].ival;

                    /* Shape (item 2) - tuple of ints */
                    pk_item *shape = &args.seq.items[2];
                    if (shape->type == PK_T_TUPLE) {
                        ti.ndim = shape->seq.count < 8 ? shape->seq.count : 8;
                        for (int i = 0; i < ti.ndim; i++) {
                            if (shape->seq.items[i].type == PK_T_INT)
                                ti.shape[i] = shape->seq.items[i].ival;
                        }
                    }

                    pkl_add_tensor(&s->result, &ti);
                }
                /* Free tuple items */
                if (args.type == PK_T_TUPLE && args.seq.items) {
                    for (int i = 0; i < args.seq.count; i++) {
                        if (args.seq.items[i].type == PK_T_TUPLE && args.seq.items[i].seq.items)
                            free(args.seq.items[i].seq.items);
                    }
                    free(args.seq.items);
                }
            } else if (callable.type == PK_T_GLOBAL &&
                       strstr(callable.sval, "OrderedDict")) {
                /* OrderedDict() - push an empty dict */
                pk_item d = {0};
                d.type = PK_T_DICT;
                pk_push(s, d);
                if (args.type == PK_T_TUPLE && args.seq.items)
                    free(args.seq.items);
                break;
            } else {
                /* Unknown REDUCE — push None */
                if (args.type == PK_T_TUPLE && args.seq.items)
                    free(args.seq.items);
            }

            /* Push a placeholder result */
            pk_item r = {0};
            r.type = PK_T_NONE;
            pk_push(s, r);
            break;
        }

        case PK_BUILD: {
            /* Pop state dict, pop obj, apply state to obj */
            pk_item state = pk_pop(s);
            /* Leave the object on top of stack */
            (void)state;
            break;
        }

        case PK_BINPERSID:
            pk_handle_persid(s);
            break;

        case PK_BINPUT: {
            if (s->pos >= s->size) return -1;
            int idx = s->data[s->pos++];
            if (idx < PK_MEMO_MAX && s->sp > 0) {
                s->memo[idx] = s->stack[s->sp - 1];
                s->memo_used[idx] = 1;
            }
            break;
        }
        case PK_LONG_BINPUT: {
            if (s->pos + 4 > s->size) return -1;
            uint32_t idx = read_u32(s->data + s->pos);
            s->pos += 4;
            if (idx < PK_MEMO_MAX && s->sp > 0) {
                s->memo[idx] = s->stack[s->sp - 1];
                s->memo_used[idx] = 1;
            }
            break;
        }
        case PK_BINGET: {
            if (s->pos >= s->size) return -1;
            int idx = s->data[s->pos++];
            if (idx < PK_MEMO_MAX && s->memo_used[idx])
                pk_push(s, s->memo[idx]);
            else {
                pk_item n = {0};
                n.type = PK_T_NONE;
                pk_push(s, n);
            }
            break;
        }
        case PK_LONG_BINGET: {
            if (s->pos + 4 > s->size) return -1;
            uint32_t idx = read_u32(s->data + s->pos);
            s->pos += 4;
            if (idx < PK_MEMO_MAX && s->memo_used[idx])
                pk_push(s, s->memo[idx]);
            else {
                pk_item n = {0};
                n.type = PK_T_NONE;
                pk_push(s, n);
            }
            break;
        }
        case PK_MEMOIZE: {
            /* Memoize top of stack at next memo index */
            /* Find next free slot */
            for (int i = 0; i < PK_MEMO_MAX; i++) {
                if (!s->memo_used[i]) {
                    if (s->sp > 0) {
                        s->memo[i] = s->stack[s->sp - 1];
                        s->memo_used[i] = 1;
                    }
                    break;
                }
            }
            break;
        }

        default:
            fprintf(stderr, "pth: unknown pickle opcode 0x%02x at offset %zu\n",
                    op, s->pos - 1);
            return -1;
        }
    }
    return 0;
}

/* ============================================================= */
/* Storage resolver                                               */
/* ============================================================= */

static fs_dtype storage_class_to_dtype(const char *cls) {
    if (strstr(cls, "BFloat16")) return FS_DTYPE_BF16;
    if (strstr(cls, "Float"))    return FS_DTYPE_F32;
    if (strstr(cls, "Half"))     return FS_DTYPE_F16;
    if (strstr(cls, "Double"))   return FS_DTYPE_F64;
    if (strstr(cls, "Byte"))     return FS_DTYPE_U8;
    if (strstr(cls, "Char"))     return FS_DTYPE_I8;
    if (strstr(cls, "Short"))    return FS_DTYPE_I16;
    if (strstr(cls, "Int"))      return FS_DTYPE_I32;
    if (strstr(cls, "Long"))     return FS_DTYPE_I64;
    return FS_DTYPE_UNKNOWN;
}

/* ============================================================= */
/* Public API                                                     */
/* ============================================================= */

struct pt_file {
    zip_archive *zip;
    pt_tensor *tensors;
    int n_tensors;
};

pt_file *pt_open(const char *path) {
    if (!path) return NULL;

    zip_archive *zip = zip_open(path);
    if (!zip) return NULL;

    /* Find the pickle file */
    const zip_entry *pkl_entry = zip_find_suffix(zip, "data.pkl");
    if (!pkl_entry || !pkl_entry->data) {
        zip_close(zip);
        return NULL;
    }

    /* Parse pickle */
    pk_state *pks = calloc(1, sizeof(pk_state));
    if (!pks) { zip_close(zip); return NULL; }
    pks->data = pkl_entry->data;
    pks->size = pkl_entry->data_size;

    /* Single-pass parse: REDUCE adds tensors (unnamed), SETITEM/SETITEMS
     * assigns names retroactively from the dict key strings. */
    if (pk_parse(pks) < 0) {
        free(pks->result.tensors);
        free(pks);
        zip_close(zip);
        return NULL;
    }

    pkl_result *pkl = &pks->result;

    /* Resolve storages: map each tensor's storage_key to a ZIP entry */
    pt_file *f = malloc(sizeof(pt_file));
    if (!f) {
        free(pkl->tensors);
        free(pks);
        zip_close(zip);
        return NULL;
    }

    f->zip = zip;
    f->n_tensors = pkl->n_tensors;
    f->tensors = calloc(pkl->n_tensors, sizeof(pt_tensor));
    if (!f->tensors) {
        free(pkl->tensors);
        free(pks);
        free(f);
        zip_close(zip);
        return NULL;
    }

    for (int i = 0; i < pkl->n_tensors; i++) {
        pkl_tensor_info *ti = &pkl->tensors[i];
        pt_tensor *pt = &f->tensors[i];

        strncpy(pt->name, ti->name, sizeof(pt->name) - 1);
        pt->dtype = storage_class_to_dtype(ti->storage_class);
        pt->ndim = ti->ndim;
        memcpy(pt->shape, ti->shape, sizeof(pt->shape));

        /* Compute total element count and data size */
        size_t n_elements = 1;
        for (int d = 0; d < pt->ndim; d++) n_elements *= (size_t)pt->shape[d];
        size_t elem_size = fs_dtype_size(pt->dtype);
        pt->data_size = n_elements * elem_size;

        /* Find ZIP entry for this storage */
        /* Try multiple naming patterns:
         * "archive/data/KEY", "data/KEY", "archive/data.pkl/../data/KEY" */
        char entry_name[512];
        const zip_entry *ze = NULL;

        /* Pattern 1: archive/data/KEY */
        snprintf(entry_name, sizeof(entry_name), "archive/data/%s", ti->storage_key);
        ze = zip_find(zip, entry_name);

        if (!ze) {
            /* Pattern 2: Try with prefix from pkl entry name */
            /* e.g. if pkl is "RWKV-x070-World-0.1B.../data.pkl",
             * storage is "RWKV-x070-World-0.1B.../data/0" */
            const char *pkl_name = pkl_entry->name;
            const char *pkl_suffix = strstr(pkl_name, "data.pkl");
            if (pkl_suffix) {
                size_t prefix_len = (size_t)(pkl_suffix - pkl_name);
                snprintf(entry_name, sizeof(entry_name), "%.*sdata/%s",
                         (int)prefix_len, pkl_name, ti->storage_key);
                ze = zip_find(zip, entry_name);
            }
        }

        if (ze && ze->data) {
            size_t byte_offset = (size_t)ti->offset * elem_size;
            if (byte_offset + pt->data_size <= ze->data_size)
                pt->data = ze->data + byte_offset;
            else if (byte_offset < ze->data_size)
                pt->data = ze->data + byte_offset;
        }
    }

    free(pkl->tensors);
    free(pks);
    return f;
}

int pt_tensor_count(const pt_file *f) {
    return f ? f->n_tensors : 0;
}

const pt_tensor *pt_tensor_at(const pt_file *f, int index) {
    if (!f || index < 0 || index >= f->n_tensors) return NULL;
    return &f->tensors[index];
}

const char **pt_tensor_names(const pt_file *f, int *count) {
    if (!f) { if (count) *count = 0; return NULL; }
    const char **names = malloc(f->n_tensors * sizeof(char *));
    if (!names) { if (count) *count = 0; return NULL; }
    for (int i = 0; i < f->n_tensors; i++)
        names[i] = f->tensors[i].name;
    if (count) *count = f->n_tensors;
    return names;
}

void pt_close(pt_file *f) {
    if (!f) return;
    zip_close(f->zip);
    free(f->tensors);
    free(f);
}
