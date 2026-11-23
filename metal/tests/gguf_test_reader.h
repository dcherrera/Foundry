/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gguf_test_reader.h
 * @brief Locates a tensor inside a GGUF v3 container and reads its raw bytes,
 *   so a Metal quant-kernel test can be driven by blocks from the real model
 *   file instead of by synthetic ones. Header-only and test-scope: it parses
 *   only enough of the container to resolve a tensor name to its type, shape
 *   and absolute byte offset, and never dequantizes anything.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_METAL_TESTS_GGUF_TEST_READER_H
#define FOUNDRY_METAL_TESTS_GGUF_TEST_READER_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ggml type ids as they appear in a GGUF tensor info record. */
#define GGUF_GGML_TYPE_Q5_K  13
#define GGUF_GGML_TYPE_Q6_K  14

#define GGUF_MAX_TENSOR_DIMS   4
#define GGUF_MAX_TENSOR_NAME 256

/* Where one tensor lives in the file, and what shape it has.
 * GGUF stores dims fastest-varying first, so for a 2-D weight
 * dims[0] is the column count and dims[1] the row count. */
typedef struct gguf_tensor_location {
    uint32_t ggml_type;
    uint32_t n_dims;
    uint64_t dims[GGUF_MAX_TENSOR_DIMS];
    uint64_t file_offset;      /* absolute offset of the tensor's first byte */
} gguf_tensor_location;

/* An opened container, positioned so tensors can be looked up repeatedly. */
typedef struct gguf_test_file {
    FILE    *fp;
    uint64_t tensor_count;
    uint64_t tensor_info_start;   /* absolute offset of the first tensor info */
    uint64_t tensor_data_start;   /* absolute offset of the tensor data block */
    uint64_t alignment;
} gguf_test_file;

/* ================================================================
 * Primitive reads
 * ================================================================ */

static inline int gguf_read_exact(FILE *fp, void *dst, size_t bytes)
{
    return fread(dst, 1, bytes, fp) == bytes ? 0 : -1;
}

static inline int gguf_read_u32(FILE *fp, uint32_t *out)
{
    return gguf_read_exact(fp, out, sizeof(*out));
}

static inline int gguf_read_u64(FILE *fp, uint64_t *out)
{
    return gguf_read_exact(fp, out, sizeof(*out));
}

/* Reads a GGUF string (u64 length + raw bytes) into `dst`, always
 * NUL-terminating. Oversized strings are consumed but truncated in `dst`,
 * which is fine here — no tensor name we look for is near the cap. */
static inline int gguf_read_string(FILE *fp, char *dst, size_t dst_size)
{
    uint64_t length = 0;
    if (gguf_read_u64(fp, &length) != 0)
        return -1;

    size_t kept = length < dst_size - 1 ? (size_t)length : dst_size - 1;
    if (gguf_read_exact(fp, dst, kept) != 0)
        return -1;
    dst[kept] = '\0';

    if (length > kept && fseeko(fp, (off_t)(length - kept), SEEK_CUR) != 0)
        return -1;
    return 0;
}

/* ================================================================
 * Metadata skipping
 * ================================================================ */

/* Byte width of a fixed-size GGUF metadata value type, or 0 if the type is
 * variable-length (string, array) or unknown. */
static inline size_t gguf_fixed_value_size(uint32_t value_type)
{
    switch (value_type) {
    case 0: case 1: case 7:            return 1;   /* uint8, int8, bool   */
    case 2: case 3:                    return 2;   /* uint16, int16       */
    case 4: case 5: case 6:            return 4;   /* uint32, int32, f32  */
    case 10: case 11: case 12:         return 8;   /* uint64, int64, f64  */
    default:                           return 0;
    }
}

static inline int gguf_skip_value(FILE *fp, uint32_t value_type)
{
    size_t fixed = gguf_fixed_value_size(value_type);
    if (fixed > 0)
        return fseeko(fp, (off_t)fixed, SEEK_CUR);

    if (value_type == 8) {             /* string */
        uint64_t length = 0;
        if (gguf_read_u64(fp, &length) != 0)
            return -1;
        return fseeko(fp, (off_t)length, SEEK_CUR);
    }

    if (value_type == 9) {             /* array */
        uint32_t element_type = 0;
        uint64_t count = 0;
        if (gguf_read_u32(fp, &element_type) != 0 || gguf_read_u64(fp, &count) != 0)
            return -1;
        size_t element_size = gguf_fixed_value_size(element_type);
        if (element_size > 0)
            return fseeko(fp, (off_t)(element_size * count), SEEK_CUR);
        for (uint64_t i = 0; i < count; i++) {
            if (gguf_skip_value(fp, element_type) != 0)
                return -1;
        }
        return 0;
    }

    return -1;                         /* unknown type: cannot skip safely */
}

/* Advances past one tensor info record, optionally reporting it. */
static inline int gguf_read_tensor_info(FILE *fp, char *name, size_t name_size,
                                        gguf_tensor_location *out)
{
    if (gguf_read_string(fp, name, name_size) != 0)
        return -1;

    uint32_t n_dims = 0;
    if (gguf_read_u32(fp, &n_dims) != 0 || n_dims > GGUF_MAX_TENSOR_DIMS)
        return -1;

    gguf_tensor_location location = {0};
    location.n_dims = n_dims;
    for (uint32_t i = 0; i < n_dims; i++) {
        if (gguf_read_u64(fp, &location.dims[i]) != 0)
            return -1;
    }

    uint64_t relative_offset = 0;
    if (gguf_read_u32(fp, &location.ggml_type) != 0 ||
        gguf_read_u64(fp, &relative_offset) != 0)
        return -1;

    location.file_offset = relative_offset;   /* still relative; fixed up by the caller */
    if (out) *out = location;
    return 0;
}

/* ================================================================
 * Public API
 * ================================================================ */

/**
 * Opens a GGUF v3 file and resolves where its tensor data begins.
 *
 * @param path  Path to the .gguf file
 * @param out   Receives the opened handle (close with gguf_test_close)
 * @return      0 on success, -1 if the file is missing or not a GGUF v3
 *
 * Thread-safe: yes (touches only the caller's handle)
 * Reentrant: yes
 * Signal-safe: no
 */
static inline int gguf_test_open(const char *path, gguf_test_file *out)
{
    memset(out, 0, sizeof(*out));

    out->fp = fopen(path, "rb");
    if (!out->fp)
        return -1;

    char magic[4] = {0};
    uint32_t version = 0;
    uint64_t kv_count = 0;
    if (gguf_read_exact(out->fp, magic, 4) != 0 ||
        memcmp(magic, "GGUF", 4) != 0 ||
        gguf_read_u32(out->fp, &version) != 0 || version != 3 ||
        gguf_read_u64(out->fp, &out->tensor_count) != 0 ||
        gguf_read_u64(out->fp, &kv_count) != 0)
        goto fail;

    out->alignment = 32;               /* GGUF default, overridden by metadata */
    for (uint64_t i = 0; i < kv_count; i++) {
        char key[GGUF_MAX_TENSOR_NAME];
        uint32_t value_type = 0;
        if (gguf_read_string(out->fp, key, sizeof(key)) != 0 ||
            gguf_read_u32(out->fp, &value_type) != 0)
            goto fail;
        if (strcmp(key, "general.alignment") == 0 && value_type == 4) {
            uint32_t alignment = 0;
            if (gguf_read_u32(out->fp, &alignment) != 0)
                goto fail;
            out->alignment = alignment ? alignment : 32;
            continue;
        }
        if (gguf_skip_value(out->fp, value_type) != 0)
            goto fail;
    }

    off_t info_start = ftello(out->fp);
    if (info_start < 0)
        goto fail;
    out->tensor_info_start = (uint64_t)info_start;

    for (uint64_t i = 0; i < out->tensor_count; i++) {
        char name[GGUF_MAX_TENSOR_NAME];
        if (gguf_read_tensor_info(out->fp, name, sizeof(name), NULL) != 0)
            goto fail;
    }

    off_t info_end = ftello(out->fp);
    if (info_end < 0)
        goto fail;
    out->tensor_data_start =
        ((uint64_t)info_end + out->alignment - 1) / out->alignment * out->alignment;
    return 0;

fail:
    fclose(out->fp);
    out->fp = NULL;
    return -1;
}

static inline void gguf_test_close(gguf_test_file *file)
{
    if (file && file->fp) {
        fclose(file->fp);
        file->fp = NULL;
    }
}

/**
 * Finds a tensor by name and reports its type, shape and absolute offset.
 *
 * @param file         Handle from gguf_test_open
 * @param tensor_name  Exact tensor name, e.g. "blk.0.attn_qkv.weight"
 * @param out          Receives the location (file_offset is absolute)
 * @return             0 on success, -1 if absent or the file is malformed
 *
 * Thread-safe: no (seeks the shared FILE*)
 * Reentrant: no
 * Signal-safe: no
 */
static inline int gguf_test_find_tensor(gguf_test_file *file, const char *tensor_name,
                                        gguf_tensor_location *out)
{
    if (!file || !file->fp)
        return -1;
    if (fseeko(file->fp, (off_t)file->tensor_info_start, SEEK_SET) != 0)
        return -1;

    for (uint64_t i = 0; i < file->tensor_count; i++) {
        char name[GGUF_MAX_TENSOR_NAME];
        gguf_tensor_location location;
        if (gguf_read_tensor_info(file->fp, name, sizeof(name), &location) != 0)
            return -1;
        if (strcmp(name, tensor_name) == 0) {
            location.file_offset += file->tensor_data_start;
            *out = location;
            return 0;
        }
    }
    return -1;
}

/**
 * Reads raw bytes at an absolute file offset — used to pull a handful of real
 * quantized rows out of a multi-gigabyte model without mapping the whole file.
 *
 * @return 0 on success, -1 on a short or failed read
 *
 * Thread-safe: no (seeks the shared FILE*)
 * Reentrant: no
 * Signal-safe: no
 */
static inline int gguf_test_read_at(gguf_test_file *file, uint64_t offset,
                                    void *dst, size_t bytes)
{
    if (!file || !file->fp)
        return -1;
    if (fseeko(file->fp, (off_t)offset, SEEK_SET) != 0)
        return -1;
    return gguf_read_exact(file->fp, dst, bytes);
}

#endif /* FOUNDRY_METAL_TESTS_GGUF_TEST_READER_H */
