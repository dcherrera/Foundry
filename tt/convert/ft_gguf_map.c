/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ft_gguf_map.c
 * @brief  Read-only memory-mapped GGUF reader: header, metadata, tensor directory, splits.
 *
 * Written from a behaviour specification; no ggml or llama.cpp source was consulted.
 *
 * Each part file is mapped once, read-only and private; the header is parsed
 * front to back with a bounds-checked cursor and no tensor byte is touched at
 * open. Tables are allocated up front at the size the header declares, so an
 * absurd count fails as FT_GGUF_ENOMEM rather than as a parse error.
 *
 * @status None
 * @issues Kept as-is: an empty file reports the fstat failure with whatever errno
 *         holds; F32/F16/BF16 sizes that wrap to zero are reported as partial
 *         blocks; a zero-element tensor of any type is accepted; split filename
 *         digits and part 1's split.no are not checked; split tensor totals sum
 *         in 32 bits; a non-integer general.alignment is ignored; data_offset
 *         alignment is not checked.
 * @todo   None
 */

/* Linux hides MADV_DONTNEED under -std=c99 -pedantic without this. */
#define _DEFAULT_SOURCE

#include "ft_gguf_map.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ft_gguf_dequant.h"

#define GGUF_MAGIC 0x46554747u
#define GGUF_DEFAULT_ALIGNMENT 32u
#define GGUF_SPLIT_SUFFIX_BYTES 20u

/** GGUF metadata value type ids. */
enum {
    GGUF_VALUE_U8 = 0,
    GGUF_VALUE_I8 = 1,
    GGUF_VALUE_U16 = 2,
    GGUF_VALUE_I16 = 3,
    GGUF_VALUE_U32 = 4,
    GGUF_VALUE_I32 = 5,
    GGUF_VALUE_F32 = 6,
    GGUF_VALUE_BOOL = 7,
    GGUF_VALUE_STRING = 8,
    GGUF_VALUE_ARRAY = 9,
    GGUF_VALUE_U64 = 10,
    GGUF_VALUE_I64 = 11,
    GGUF_VALUE_F64 = 12
};

/** One retained metadata key. Strings point into the mapping; arrays are not kept. */
typedef struct {
    char key[FT_GGUF_MAX_NAME];
    uint32_t type;
    uint64_t integer;
    const char *text;
    uint64_t text_bytes;
    uint64_t value_offset; /**< the value's first byte, as a file offset */
} gguf_metadata_entry;

/** Where one part sits in a split set, from its own metadata. */
typedef struct {
    int is_split;
    uint32_t count;
    uint32_t index;
    uint64_t tensors;
} gguf_split_layout;

/** One mapped part file. */
typedef struct {
    char path[FT_GGUF_MAX_PATH];
    int fd;
    unsigned char *mapping;
    size_t mapping_bytes;
    uint64_t data_start;
} gguf_part_file;

/** Everything parsed out of one part's header. */
typedef struct {
    uint32_t version;
    gguf_metadata_entry *metadata;
    uint64_t metadata_count;
    ft_gguf_tensor *tensors;
    uint32_t tensor_count;
    gguf_split_layout split;
    uint64_t metadata_raw_begin; /**< file offset of the first key/value pair */
    uint64_t metadata_raw_end;   /**< file offset just past the last one */
    uint64_t alignment;          /**< general.alignment, or the default */
} gguf_part_parse;

struct ft_gguf_map {
    long page_size;
    uint32_t version;
    gguf_part_file single_part;
    gguf_part_file *parts;
    uint32_t part_count;
    ft_gguf_tensor *tensors;
    uint32_t tensor_count;
    uint32_t *tensor_parts;
    gguf_metadata_entry *metadata;
    uint64_t metadata_count;
    uint64_t metadata_raw_begin;
    uint64_t metadata_raw_end;
    uint64_t alignment;
};

/** Bounds-checked reader over a mapping. Once failed, every read fails. */
typedef struct {
    const unsigned char *bytes;
    size_t size;
    size_t position;
    int failed;
} gguf_cursor;

static const unsigned char *cursor_take(gguf_cursor *cursor, uint64_t count)
{
    const unsigned char *at;

    if (cursor->failed || count > cursor->size - cursor->position) {
        cursor->failed = 1;
        return NULL;
    }
    at = cursor->bytes + cursor->position;
    cursor->position += (size_t)count;
    return at;
}

static uint32_t cursor_u32(gguf_cursor *cursor)
{
    const unsigned char *at = cursor_take(cursor, 4);
    uint32_t value = 0;

    if (at)
        memcpy(&value, at, sizeof(value));
    return value;
}

static uint64_t cursor_u64(gguf_cursor *cursor)
{
    const unsigned char *at = cursor_take(cursor, 8);
    uint64_t value = 0;

    if (at)
        memcpy(&value, at, sizeof(value));
    return value;
}

/** A u64 length then that many bytes, no terminator. */
static int cursor_string(gguf_cursor *cursor, const char **text, uint64_t *length)
{
    uint64_t declared = cursor_u64(cursor);
    const unsigned char *at;

    if (cursor->failed)
        return -1;
    at = cursor_take(cursor, declared);
    if (!at)
        return -1;
    *text = (const char *)at;
    *length = declared;
    return 0;
}

/** Copy raw bytes into a name field, truncated to 127 bytes and NUL-terminated. */
static void copy_name(char *name, const char *text, uint64_t length)
{
    size_t copied = length < FT_GGUF_MAX_NAME - 1 ? (size_t)length : FT_GGUF_MAX_NAME - 1;

    memcpy(name, text, copied);
    name[copied] = '\0';
}

/** Byte size of a scalar metadata type; 0 for strings, arrays and unknown ids. */
static uint64_t scalar_value_bytes(uint32_t type)
{
    switch (type) {
    case GGUF_VALUE_U8:
    case GGUF_VALUE_I8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_U16:
    case GGUF_VALUE_I16:
        return 2;
    case GGUF_VALUE_U32:
    case GGUF_VALUE_I32:
    case GGUF_VALUE_F32:
        return 4;
    case GGUF_VALUE_U64:
    case GGUF_VALUE_I64:
    case GGUF_VALUE_F64:
        return 8;
    default:
        return 0;
    }
}

/** Skip an array value. Arrays of arrays are refused. */
static int skip_array_value(gguf_cursor *cursor)
{
    uint32_t element_type = cursor_u32(cursor);
    uint64_t count = cursor_u64(cursor);
    uint64_t element_bytes;
    uint64_t i;

    if (cursor->failed)
        return -1;
    if (element_type == GGUF_VALUE_STRING) {
        for (i = 0; i < count; i++) {
            const char *text;
            uint64_t length;
            if (cursor_string(cursor, &text, &length) != 0)
                return -1;
        }
        return 0;
    }
    if (element_type == GGUF_VALUE_ARRAY)
        return -1;
    element_bytes = scalar_value_bytes(element_type);
    if (element_bytes == 0)
        return -1;
    if (count > (cursor->size - cursor->position) / element_bytes)
        return -1;
    return cursor_take(cursor, count * element_bytes) ? 0 : -1;
}

/** Read one value of `entry->type` into the entry. */
static int parse_metadata_value(gguf_cursor *cursor, gguf_metadata_entry *entry)
{
    const unsigned char *at;
    uint64_t bytes;

    if (entry->type == GGUF_VALUE_STRING)
        return cursor_string(cursor, &entry->text, &entry->text_bytes);
    if (entry->type == GGUF_VALUE_ARRAY)
        return skip_array_value(cursor);
    bytes = scalar_value_bytes(entry->type);
    if (bytes == 0)
        return -1;
    at = cursor_take(cursor, bytes);
    if (!at)
        return -1;

    switch (entry->type) {
    case GGUF_VALUE_U8:
    case GGUF_VALUE_BOOL:
        entry->integer = at[0];
        break;
    case GGUF_VALUE_I8:
        entry->integer = (uint64_t)(int64_t)(int8_t)at[0];
        break;
    case GGUF_VALUE_U16: {
        uint16_t value;
        memcpy(&value, at, sizeof(value));
        entry->integer = value;
        break;
    }
    case GGUF_VALUE_I16: {
        int16_t value;
        memcpy(&value, at, sizeof(value));
        entry->integer = (uint64_t)(int64_t)value;
        break;
    }
    case GGUF_VALUE_U32: {
        uint32_t value;
        memcpy(&value, at, sizeof(value));
        entry->integer = value;
        break;
    }
    case GGUF_VALUE_I32: {
        int32_t value;
        memcpy(&value, at, sizeof(value));
        entry->integer = (uint64_t)(int64_t)value;
        break;
    }
    case GGUF_VALUE_U64:
    case GGUF_VALUE_I64:
        memcpy(&entry->integer, at, sizeof(entry->integer));
        break;
    default:
        break;
    }
    return 0;
}

static int parse_metadata_table(gguf_cursor *cursor, const char *path, uint64_t kv_count,
                                gguf_part_parse *parse)
{
    uint64_t i;

    parse->metadata = calloc(kv_count ? (size_t)kv_count : 1, sizeof(*parse->metadata));
    if (!parse->metadata)
        return FT_GGUF_ENOMEM;

    for (i = 0; i < kv_count; i++) {
        gguf_metadata_entry *entry = &parse->metadata[i];
        const char *key;
        uint64_t key_length;

        if (cursor_string(cursor, &key, &key_length) != 0) {
            fprintf(stderr, "ft_gguf_map: '%s' metadata key %llu is truncated\n", path,
                    (unsigned long long)i);
            return FT_GGUF_ERROR;
        }
        copy_name(entry->key, key, key_length);
        entry->type = cursor_u32(cursor);
        entry->value_offset = cursor->position;
        if (parse_metadata_value(cursor, entry) != 0) {
            fprintf(stderr, "ft_gguf_map: '%s' metadata '%s' (type %u) is malformed\n", path,
                    entry->key, entry->type);
            return FT_GGUF_ERROR;
        }
        parse->metadata_count = i + 1;
    }
    return FT_GGUF_OK;
}

static int parse_tensor_table(gguf_cursor *cursor, const char *path, uint32_t tensor_count,
                              gguf_part_parse *parse)
{
    uint32_t i;

    parse->tensors = calloc(tensor_count ? tensor_count : 1, sizeof(*parse->tensors));
    if (!parse->tensors)
        return FT_GGUF_ENOMEM;

    for (i = 0; i < tensor_count; i++) {
        ft_gguf_tensor *tensor = &parse->tensors[i];
        const char *name;
        uint64_t name_length;
        uint32_t d;

        if (cursor_string(cursor, &name, &name_length) != 0) {
            fprintf(stderr, "ft_gguf_map: '%s' tensor %u name is truncated\n", path, i);
            return FT_GGUF_ERROR;
        }
        copy_name(tensor->name, name, name_length);
        tensor->n_dims = cursor_u32(cursor);
        if (cursor->failed || tensor->n_dims == 0 || tensor->n_dims > 4) {
            fprintf(stderr, "ft_gguf_map: '%s' tensor '%s' has %u dimensions\n", path,
                    tensor->name, tensor->n_dims);
            return FT_GGUF_ERROR;
        }
        for (d = 0; d < 4; d++)
            tensor->dims[d] = d < tensor->n_dims ? cursor_u64(cursor) : 1;
        tensor->ggml_type = cursor_u32(cursor);
        tensor->data_offset = cursor_u64(cursor);
        if (cursor->failed) {
            fprintf(stderr, "ft_gguf_map: '%s' tensor directory is truncated at entry %u\n",
                    path, i);
            return FT_GGUF_ERROR;
        }
        parse->tensor_count = i + 1;
    }
    return FT_GGUF_OK;
}

/** First entry whose key equals `key`, in file order. */
static const gguf_metadata_entry *find_metadata(const gguf_metadata_entry *metadata,
                                                uint64_t count, const char *key)
{
    uint64_t i;

    if (!key)
        return NULL;
    for (i = 0; i < count; i++) {
        if (strcmp(metadata[i].key, key) == 0)
            return &metadata[i];
    }
    return NULL;
}

/** Integer (or bool) metadata lookup; `*value` written only on FT_GGUF_OK. */
static int lookup_integer(const gguf_metadata_entry *metadata, uint64_t count, const char *key,
                          uint64_t *value)
{
    const gguf_metadata_entry *entry = find_metadata(metadata, count, key);

    if (!entry)
        return FT_GGUF_ENOTFOUND;
    if (scalar_value_bytes(entry->type) == 0 || entry->type == GGUF_VALUE_F32 ||
        entry->type == GGUF_VALUE_F64)
        return FT_GGUF_EINVAL;
    *value = entry->integer;
    return FT_GGUF_OK;
}

static int read_split_layout(const char *path, gguf_part_parse *parse)
{
    uint64_t count = 0;
    uint64_t index = 0;
    uint64_t tensors = 0;

    parse->split.is_split = 0;
    parse->split.count = 1;
    parse->split.index = 0;
    parse->split.tensors = 0;
    if (lookup_integer(parse->metadata, parse->metadata_count, "split.count", &count) != FT_GGUF_OK)
        return FT_GGUF_OK;
    if (count == 0 || count > FT_GGUF_MAX_PARTS) {
        fprintf(stderr,
                "ft_gguf_map: '%s' declares split.count = %llu, which is not a part count this "
                "reader accepts (1..256)\n",
                path, (unsigned long long)count);
        return FT_GGUF_ERROR;
    }
    if (lookup_integer(parse->metadata, parse->metadata_count, "split.no", &index) != FT_GGUF_OK) {
        fprintf(stderr,
                "ft_gguf_map: '%s' declares split.count = %llu but no split.no, so its position "
                "is unknown\n",
                path, (unsigned long long)count);
        return FT_GGUF_ERROR;
    }
    if (index >= count) {
        fprintf(stderr,
                "ft_gguf_map: '%s' declares split.no = %llu, out of range for split.count = %llu\n",
                path, (unsigned long long)index, (unsigned long long)count);
        return FT_GGUF_ERROR;
    }
    (void)lookup_integer(parse->metadata, parse->metadata_count, "split.tensors.count", &tensors);

    parse->split.is_split = 1;
    parse->split.index = (uint32_t)index;
    parse->split.count = (uint32_t)count;
    parse->split.tensors = tensors;
    return FT_GGUF_OK;
}

/** Report a tensor whose type/element count has no byte size. */
static void report_unsized_tensor(const char *path, const ft_gguf_tensor *tensor,
                                  uint64_t elements)
{
    const char *type_name = ft_gguf_type_name(tensor->ggml_type);

    if (ft_gguf_type_supported(tensor->ggml_type)) {
        fprintf(stderr,
                "ft_gguf_map: '%s' tensor '%s' is %s with %" PRIu64 " elements, which is not a "
                "whole number of blocks\n",
                path, tensor->name, type_name, elements);
        return;
    }
    fprintf(stderr,
            "ft_gguf_map: '%s' tensor '%s' is %s (ggml type %u), which this converter does not "
            "read\n",
            path, tensor->name, type_name, tensor->ggml_type);
    fprintf(stderr,
            "ft_gguf_map: readable types are F32, F16, BF16, Q8_0 and the k-quants Q2_K, Q3_K, "
            "Q4_K, Q5_K and Q6_K -- re-quantize the source to one of those, or convert the "
            "original weights to a BF16 or F16 GGUF\n");
}

/** Size every tensor and point it into the aligned tensor-data section. */
static int place_tensors(gguf_part_file *part, gguf_part_parse *parse, uint64_t header_bytes,
                         uint64_t alignment)
{
    uint64_t file_size = part->mapping_bytes;
    uint64_t aligned = (header_bytes + alignment - 1) / alignment * alignment;
    uint64_t available;
    const unsigned char *base;
    uint32_t i;

    if (aligned > file_size) {
        fprintf(stderr, "ft_gguf_map: '%s' tensor data starts past the end of the file\n",
                part->path);
        return FT_GGUF_ERROR;
    }
    part->data_start = aligned;
    base = part->mapping + aligned;
    available = file_size - aligned;

    for (i = 0; i < parse->tensor_count; i++) {
        ft_gguf_tensor *tensor = &parse->tensors[i];
        uint64_t elements = tensor->dims[0] * tensor->dims[1] * tensor->dims[2] * tensor->dims[3];
        uint64_t bytes = ft_gguf_type_bytes(tensor->ggml_type, elements);

        if (bytes == 0 && elements != 0) {
            report_unsized_tensor(part->path, tensor, elements);
            return FT_GGUF_ERROR;
        }
        if (tensor->data_offset > available || bytes > available - tensor->data_offset) {
            fprintf(stderr, "ft_gguf_map: '%s' tensor '%s' runs past the end of the file\n",
                    part->path, tensor->name);
            return FT_GGUF_ERROR;
        }
        tensor->byte_count = bytes;
        tensor->data = base + tensor->data_offset;
    }
    return FT_GGUF_OK;
}

/** Parse a mapped part: header, metadata, tensors, alignment, split layout, placement. */
static int parse_part(gguf_part_file *part, gguf_part_parse *parse)
{
    gguf_cursor cursor = {0};
    uint64_t tensor_count;
    uint64_t kv_count;
    uint64_t alignment = GGUF_DEFAULT_ALIGNMENT;
    int status;

    cursor.bytes = part->mapping;
    cursor.size = part->mapping_bytes;
    if (cursor_u32(&cursor) != GGUF_MAGIC) {
        fprintf(stderr, "ft_gguf_map: '%s' is not a GGUF file\n", part->path);
        return FT_GGUF_ERROR;
    }
    parse->version = cursor_u32(&cursor);
    if (parse->version != 2 && parse->version != 3) {
        fprintf(stderr, "ft_gguf_map: '%s' is GGUF version %u, which is not supported\n",
                part->path, parse->version);
        return FT_GGUF_ERROR;
    }
    tensor_count = cursor_u64(&cursor);
    kv_count = cursor_u64(&cursor);
    if (cursor.failed || tensor_count > UINT32_MAX) {
        fprintf(stderr, "ft_gguf_map: '%s' header is truncated or absurd\n", part->path);
        return FT_GGUF_ERROR;
    }

    parse->metadata_raw_begin = cursor.position;
    status = parse_metadata_table(&cursor, part->path, kv_count, parse);
    if (status != FT_GGUF_OK)
        return status;
    parse->metadata_raw_end = cursor.position;
    status = parse_tensor_table(&cursor, part->path, (uint32_t)tensor_count, parse);
    if (status != FT_GGUF_OK)
        return status;

    if (lookup_integer(parse->metadata, parse->metadata_count, "general.alignment", &alignment) ==
            FT_GGUF_OK &&
        (alignment == 0 || (alignment & (alignment - 1)) != 0)) {
        fprintf(stderr, "ft_gguf_map: '%s' general.alignment %llu is not a power of two\n",
                part->path, (unsigned long long)alignment);
        return FT_GGUF_ERROR;
    }
    parse->alignment = alignment;
    status = read_split_layout(part->path, parse);
    if (status != FT_GGUF_OK)
        return status;
    return place_tensors(part, parse, cursor.position, alignment);
}

static void release_part_file(gguf_part_file *part)
{
    if (part->mapping) {
        munmap(part->mapping, part->mapping_bytes);
        part->mapping = NULL;
    }
    if (part->fd >= 0) {
        close(part->fd);
        part->fd = -1;
    }
}

static void free_part_parse(gguf_part_parse *parse)
{
    free(parse->metadata);
    parse->metadata = NULL;
    free(parse->tensors);
    parse->tensors = NULL;
}

/** Map one file and parse it. On failure everything acquired is released. */
static int open_part(const char *path, gguf_part_file *part, gguf_part_parse *parse)
{
    size_t path_length = strlen(path);
    struct stat status_buffer;
    void *mapping;
    int status;

    memset(parse, 0, sizeof(*parse));
    part->fd = -1;
    part->mapping = NULL;
    part->mapping_bytes = 0;
    part->data_start = 0;
    if (path_length >= FT_GGUF_MAX_PATH) {
        fprintf(stderr, "ft_gguf_map: path is longer than %u bytes\n", (unsigned)FT_GGUF_MAX_PATH);
        return FT_GGUF_EINVAL;
    }
    memcpy(part->path, path, path_length + 1);

    part->fd = open(part->path, O_RDONLY);
    if (part->fd < 0) {
        fprintf(stderr, "ft_gguf_map: open('%s'): %s\n", part->path, strerror(errno));
        return FT_GGUF_EIO;
    }
    /* No errno-touching call may sit between open() and this message: for an
     * empty file fstat succeeded and the message prints whatever errno holds. */
    if (fstat(part->fd, &status_buffer) != 0 || status_buffer.st_size <= 0) {
        fprintf(stderr, "ft_gguf_map: fstat('%s'): %s\n", part->path, strerror(errno));
        status = FT_GGUF_EIO;
        goto fail;
    }
    part->mapping_bytes = (size_t)status_buffer.st_size;
    mapping = mmap(NULL, part->mapping_bytes, PROT_READ, MAP_PRIVATE, part->fd, 0);
    if (mapping == MAP_FAILED) {
        fprintf(stderr, "ft_gguf_map: mmap('%s', %zu): %s\n", part->path, part->mapping_bytes,
                strerror(errno));
        status = FT_GGUF_EIO;
        goto fail;
    }
    part->mapping = mapping;
    (void)madvise(part->mapping, part->mapping_bytes, MADV_SEQUENTIAL);

    status = parse_part(part, parse);
    if (status != FT_GGUF_OK)
        goto fail;
    return FT_GGUF_OK;

fail:
    free_part_parse(parse);
    release_part_file(part);
    return status;
}

/** Whether `path` ends in `-DDDDD-of-DDDDD.gguf` with at least one byte before it. */
static int has_split_suffix(const char *path, size_t length)
{
    const char *suffix;
    size_t i;

    if (length < GGUF_SPLIT_SUFFIX_BYTES + 1)
        return 0;
    suffix = path + length - GGUF_SPLIT_SUFFIX_BYTES;
    if (suffix[0] != '-' || memcmp(suffix + 6, "-of-", 4) != 0 ||
        memcmp(suffix + 15, ".gguf", 5) != 0)
        return 0;
    for (i = 1; i <= 5; i++) {
        if (suffix[i] < '0' || suffix[i] > '9' || suffix[i + 9] < '0' || suffix[i + 9] > '9')
            return 0;
    }
    return 1;
}

/** Check part `index` (> 0) against part 1's layout and version. */
static int check_later_part(const gguf_part_file *part, const gguf_part_parse *parse,
                            uint32_t index, uint32_t count, const gguf_split_layout *expected,
                            uint32_t version)
{
    if (!parse->split.is_split) {
        fprintf(stderr, "ft_gguf_map: '%s' should be part %u of %u but declares no split.count\n",
                part->path, index + 1, count);
        return FT_GGUF_ERROR;
    }
    if (parse->split.index != index) {
        fprintf(stderr, "ft_gguf_map: '%s' is named part %u but declares split.no = %u\n",
                part->path, index + 1, parse->split.index);
        return FT_GGUF_ERROR;
    }
    if (parse->split.count != expected->count) {
        fprintf(stderr,
                "ft_gguf_map: '%s' declares split.count = %u, but part 1 declares %u -- these "
                "are not the same model\n",
                part->path, parse->split.count, expected->count);
        return FT_GGUF_ERROR;
    }
    if (parse->split.tensors != expected->tensors) {
        fprintf(stderr,
                "ft_gguf_map: '%s' declares split.tensors.count = %llu, but part 1 declares %llu\n",
                part->path, (unsigned long long)parse->split.tensors,
                (unsigned long long)expected->tensors);
        return FT_GGUF_ERROR;
    }
    if (parse->version != version) {
        fprintf(stderr, "ft_gguf_map: '%s' is GGUF version %u, but part 1 is version %u\n",
                part->path, parse->version, version);
        return FT_GGUF_ERROR;
    }
    return FT_GGUF_OK;
}

/** Concatenate every part's tensor directory, recording the owning part. */
static int concatenate_tensors(ft_gguf_map *map, const gguf_part_parse *parses, uint32_t total)
{
    uint32_t position = 0;
    uint32_t p;

    map->tensors = calloc(total ? total : 1, sizeof(*map->tensors));
    map->tensor_parts = calloc(total ? total : 1, sizeof(*map->tensor_parts));
    if (!map->tensors || !map->tensor_parts)
        return FT_GGUF_ENOMEM;
    for (p = 0; p < map->part_count; p++) {
        uint32_t t;
        for (t = 0; t < parses[p].tensor_count; t++) {
            if (position >= total)
                return FT_GGUF_ERROR;
            map->tensors[position] = parses[p].tensors[t];
            map->tensor_parts[position] = p;
            position++;
        }
    }
    map->tensor_count = total;
    return FT_GGUF_OK;
}

/** Map parts 1..count found by the caller path's name pattern, in order. */
static int open_part_files(ft_gguf_map *map, gguf_part_parse *parses, const char *path,
                           size_t prefix_length, uint32_t count, gguf_split_layout *expected,
                           uint32_t *running_total)
{
    uint32_t index;

    for (index = 0; index < count; index++) {
        char part_path[FT_GGUF_MAX_PATH];
        int written = snprintf(part_path, sizeof(part_path), "%.*s-%05u-of-%05u.gguf",
                               (int)prefix_length, path, index + 1, count);
        int status;

        if (written < 0 || (size_t)written >= sizeof(part_path)) {
            fprintf(stderr, "ft_gguf_map: part %u of %u has a path too long to form\n", index + 1,
                    count);
            return FT_GGUF_ERROR;
        }
        status = open_part(part_path, &map->parts[index], &parses[index]);
        if (status != FT_GGUF_OK) {
            fprintf(stderr,
                    "ft_gguf_map: part %u of %u ('%s') could not be read; the set is incomplete "
                    "and converting it would silently drop its tensors\n",
                    index + 1, count, part_path);
            return status;
        }
        map->part_count = index + 1;

        if (index == 0) {
            *expected = parses[0].split;
            map->version = parses[0].version;
            if (!expected->is_split || expected->count != count) {
                fprintf(stderr,
                        "ft_gguf_map: '%s' does not agree with '%s' about how many parts there "
                        "are\n",
                        map->parts[0].path, path);
                return FT_GGUF_ERROR;
            }
        } else {
            status = check_later_part(&map->parts[index], &parses[index], index, count, expected,
                                      map->version);
            if (status != FT_GGUF_OK)
                return status;
        }
        *running_total += parses[index].tensor_count;
    }
    return FT_GGUF_OK;
}

/** Open the whole split set the caller's part belongs to. */
static int open_split_set(ft_gguf_map *map, const char *path, uint32_t count)
{
    size_t length = strlen(path);
    gguf_part_parse *parses = NULL;
    gguf_split_layout expected = {0};
    uint32_t running_total = 0;
    uint32_t i;
    int status;

    if (!has_split_suffix(path, length)) {
        fprintf(stderr,
                "ft_gguf_map: '%s' declares split.count = %u, but its name does not end in the "
                "'-00001-of-%05u.gguf' form the other parts must be found by\n",
                path, count, count);
        return FT_GGUF_ERROR;
    }
    map->parts = calloc(count, sizeof(*map->parts));
    parses = calloc(count, sizeof(*parses));
    if (!map->parts || !parses) {
        status = FT_GGUF_ENOMEM;
        goto done;
    }

    status = open_part_files(map, parses, path, length - GGUF_SPLIT_SUFFIX_BYTES, count,
                             &expected, &running_total);
    if (status != FT_GGUF_OK)
        goto done;

    map->metadata = parses[0].metadata;
    map->metadata_count = parses[0].metadata_count;
    map->metadata_raw_begin = parses[0].metadata_raw_begin;
    map->metadata_raw_end = parses[0].metadata_raw_end;
    map->alignment = parses[0].alignment;
    parses[0].metadata = NULL;
    if (expected.tensors == 0) {
        fprintf(stderr,
                "ft_gguf_map: '%s' is part of a %u-part split but declares no "
                "split.tensors.count, so no reader can tell whether all the parts are present\n",
                map->parts[0].path, count);
        status = FT_GGUF_ERROR;
        goto done;
    }
    if (expected.tensors != running_total) {
        fprintf(stderr,
                "ft_gguf_map: the %u parts hold %u tensors between them, but "
                "split.tensors.count says %llu\n",
                map->part_count, running_total, (unsigned long long)expected.tensors);
        status = FT_GGUF_ERROR;
        goto done;
    }
    status = concatenate_tensors(map, parses, running_total);

done:
    if (parses) {
        for (i = 0; i < count; i++)
            free_part_parse(&parses[i]);
    }
    free(parses);
    return status;
}

int ft_gguf_map_open(const char *path, ft_gguf_map **out)
{
    ft_gguf_map *map;
    gguf_part_parse probe;
    uint32_t split_count;
    int status;

    if (!path || !out)
        return FT_GGUF_EINVAL;
    *out = NULL;
    map = calloc(1, sizeof(*map));
    if (!map)
        return FT_GGUF_ENOMEM;
    map->page_size = sysconf(_SC_PAGESIZE);

    status = open_part(path, &map->single_part, &probe);
    if (status != FT_GGUF_OK) {
        free(map);
        return status;
    }

    if (!probe.split.is_split) {
        map->parts = &map->single_part;
        map->part_count = 1;
        map->version = probe.version;
        map->tensors = probe.tensors;
        map->tensor_count = probe.tensor_count;
        map->metadata = probe.metadata;
        map->metadata_count = probe.metadata_count;
        map->metadata_raw_begin = probe.metadata_raw_begin;
        map->metadata_raw_end = probe.metadata_raw_end;
        map->alignment = probe.alignment;
        map->tensor_parts = calloc(probe.tensor_count ? probe.tensor_count : 1,
                                   sizeof(*map->tensor_parts));
        if (!map->tensor_parts) {
            ft_gguf_map_close(map);
            return FT_GGUF_ENOMEM;
        }
        *out = map;
        return FT_GGUF_OK;
    }

    split_count = probe.split.count;
    free_part_parse(&probe);
    release_part_file(&map->single_part);
    status = open_split_set(map, path, split_count);
    if (status != FT_GGUF_OK) {
        ft_gguf_map_close(map);
        return status;
    }
    *out = map;
    return FT_GGUF_OK;
}

void ft_gguf_map_close(ft_gguf_map *map)
{
    uint32_t i;

    if (!map)
        return;
    for (i = 0; i < map->part_count; i++)
        release_part_file(&map->parts[i]);
    if (map->parts != &map->single_part)
        free(map->parts);
    free(map->tensors);
    free(map->tensor_parts);
    free(map->metadata);
    free(map);
}

uint32_t ft_gguf_map_tensor_count(const ft_gguf_map *map)
{
    return map ? map->tensor_count : 0;
}

const ft_gguf_tensor *ft_gguf_map_tensor(const ft_gguf_map *map, uint32_t index)
{
    if (!map || index >= map->tensor_count)
        return NULL;
    return &map->tensors[index];
}

const ft_gguf_tensor *ft_gguf_map_find(const ft_gguf_map *map, const char *name)
{
    uint32_t i;

    if (!map || !name)
        return NULL;
    for (i = 0; i < map->tensor_count; i++) {
        if (strcmp(map->tensors[i].name, name) == 0)
            return &map->tensors[i];
    }
    return NULL;
}

uint32_t ft_gguf_map_version(const ft_gguf_map *map)
{
    return map ? map->version : 0;
}

uint32_t ft_gguf_map_part_count(const ft_gguf_map *map)
{
    return map ? map->part_count : 0;
}

const char *ft_gguf_map_part_path(const ft_gguf_map *map, uint32_t part_index)
{
    if (!map || part_index >= map->part_count)
        return NULL;
    return map->parts[part_index].path;
}

uint32_t ft_gguf_map_tensor_part(const ft_gguf_map *map, const ft_gguf_tensor *tensor)
{
    uintptr_t first;
    uintptr_t address;

    if (!map || !tensor || !map->tensors)
        return UINT32_MAX;
    first = (uintptr_t)map->tensors;
    address = (uintptr_t)tensor;
    if (address < first || address >= first + (uintptr_t)map->tensor_count * sizeof(*tensor))
        return UINT32_MAX;
    return map->tensor_parts[(address - first) / sizeof(*tensor)];
}

uint64_t ft_gguf_map_tensor_file_offset(const ft_gguf_map *map, const ft_gguf_tensor *tensor)
{
    uint32_t part = ft_gguf_map_tensor_part(map, tensor);

    if (part == UINT32_MAX)
        return 0;
    return map->parts[part].data_start + tensor->data_offset;
}

int ft_gguf_map_metadata_u64(const ft_gguf_map *map, const char *key, uint64_t *out)
{
    if (!map || !out)
        return FT_GGUF_EINVAL;
    return lookup_integer(map->metadata, map->metadata_count, key, out);
}

int ft_gguf_map_metadata_string(const ft_gguf_map *map, const char *key, char *out, size_t out_bytes)
{
    const gguf_metadata_entry *entry;
    size_t copied;

    if (!map || !out || out_bytes == 0)
        return FT_GGUF_EINVAL;
    entry = find_metadata(map->metadata, map->metadata_count, key);
    if (!entry)
        return FT_GGUF_ENOTFOUND;
    if (entry->type != GGUF_VALUE_STRING)
        return FT_GGUF_EINVAL;
    copied = entry->text_bytes < out_bytes - 1 ? (size_t)entry->text_bytes : out_bytes - 1;
    memcpy(out, entry->text, copied);
    out[copied] = '\0';
    return FT_GGUF_OK;
}

int ft_gguf_map_metadata_raw(const ft_gguf_map *map, const void **bytes, uint64_t *byte_count,
                             uint64_t *kv_count)
{
    if (!map || !bytes || !byte_count || !kv_count || map->part_count == 0)
        return FT_GGUF_EINVAL;
    *bytes = map->parts[0].mapping + map->metadata_raw_begin;
    *byte_count = map->metadata_raw_end - map->metadata_raw_begin;
    *kv_count = map->metadata_count;
    return FT_GGUF_OK;
}

int ft_gguf_map_metadata_value_offset(const ft_gguf_map *map, const char *key, uint32_t *type,
                                      uint64_t *raw_offset)
{
    const gguf_metadata_entry *entry;

    if (!map || !key || !type || !raw_offset)
        return FT_GGUF_EINVAL;
    entry = find_metadata(map->metadata, map->metadata_count, key);
    if (!entry)
        return FT_GGUF_ENOTFOUND;
    *type = entry->type;
    *raw_offset = entry->value_offset - map->metadata_raw_begin;
    return FT_GGUF_OK;
}

uint64_t ft_gguf_map_alignment(const ft_gguf_map *map)
{
    return map ? map->alignment : 0;
}

void ft_gguf_map_release_range(const ft_gguf_map *map, const ft_gguf_tensor *tensor,
                               uint64_t byte_offset, uint64_t byte_count)
{
    uintptr_t page;
    uintptr_t begin;
    uintptr_t start;
    uintptr_t stop;

    if (!map || !tensor || !tensor->data || map->page_size <= 0 ||
        byte_offset >= tensor->byte_count)
        return;
    if (byte_count > tensor->byte_count - byte_offset)
        byte_count = tensor->byte_count - byte_offset;
    page = (uintptr_t)map->page_size;
    begin = (uintptr_t)tensor->data + (uintptr_t)byte_offset;
    start = (begin + page - 1) / page * page;
    stop = (begin + (uintptr_t)byte_count) / page * page;
    if (stop > start)
        (void)madvise((void *)start, stop - start, MADV_DONTNEED);
}

void ft_gguf_map_release(const ft_gguf_map *map, const ft_gguf_tensor *tensor)
{
    uintptr_t page;
    uintptr_t begin;
    uintptr_t start;
    uintptr_t stop;

    if (!map || !tensor || !tensor->data || map->page_size <= 0)
        return;
    page = (uintptr_t)map->page_size;
    begin = (uintptr_t)tensor->data;
    start = (begin + page - 1) / page * page;
    stop = (begin + (uintptr_t)tensor->byte_count) / page * page;
    if (stop > start)
        (void)madvise((void *)start, stop - start, MADV_DONTNEED);
}
