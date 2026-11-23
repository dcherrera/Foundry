/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_gguf_write.c
 * @brief Implementation of the GGUF v3 writer and the key/value section builder.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "ft_gguf_write.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_WRITE_MAGIC 0x46554747u
#define GGUF_WRITE_VERSION 3u
#define GGUF_TYPE_U32 4u
#define GGUF_TYPE_F32 6u
#define GGUF_TYPE_STRING 8u
#define GGUF_TYPE_U64 10u

static void kv_append(ft_gguf_kv_buffer *kv, const void *bytes, size_t count)
{
    if (kv->failed)
        return;
    if (kv->used + count > kv->capacity) {
        size_t capacity = kv->capacity ? kv->capacity : 256u;
        unsigned char *grown;

        while (capacity < kv->used + count)
            capacity *= 2u;
        grown = realloc(kv->bytes, capacity);
        if (!grown) {
            kv->failed = 1;
            return;
        }
        kv->bytes = grown;
        kv->capacity = capacity;
    }
    memcpy(kv->bytes + kv->used, bytes, count);
    kv->used += count;
}

static void kv_append_u32(ft_gguf_kv_buffer *kv, uint32_t value)
{
    unsigned char b[4] = {(unsigned char)value, (unsigned char)(value >> 8), (unsigned char)(value >> 16),
                          (unsigned char)(value >> 24)};

    kv_append(kv, b, sizeof(b));
}

static void kv_append_u64(ft_gguf_kv_buffer *kv, uint64_t value)
{
    kv_append_u32(kv, (uint32_t)value);
    kv_append_u32(kv, (uint32_t)(value >> 32));
}

static void kv_append_string(ft_gguf_kv_buffer *kv, const char *text)
{
    size_t length = strlen(text);

    kv_append_u64(kv, (uint64_t)length);
    kv_append(kv, text, length);
}

static void kv_begin(ft_gguf_kv_buffer *kv, const char *key, uint32_t type)
{
    kv_append_string(kv, key);
    kv_append_u32(kv, type);
    if (!kv->failed)
        kv->count++;
}

void ft_gguf_kv_add_u32(ft_gguf_kv_buffer *kv, const char *key, uint32_t value)
{
    kv_begin(kv, key, GGUF_TYPE_U32);
    kv_append_u32(kv, value);
}

void ft_gguf_kv_add_u64(ft_gguf_kv_buffer *kv, const char *key, uint64_t value)
{
    kv_begin(kv, key, GGUF_TYPE_U64);
    kv_append_u64(kv, value);
}

void ft_gguf_kv_add_f32(ft_gguf_kv_buffer *kv, const char *key, float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    kv_begin(kv, key, GGUF_TYPE_F32);
    kv_append_u32(kv, bits);
}

void ft_gguf_kv_add_string(ft_gguf_kv_buffer *kv, const char *key, const char *value)
{
    kv_begin(kv, key, GGUF_TYPE_STRING);
    kv_append_string(kv, value);
}

void ft_gguf_kv_free(ft_gguf_kv_buffer *kv)
{
    free(kv->bytes);
    memset(kv, 0, sizeof(*kv));
}

static uint64_t write_align_up(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1u) / alignment * alignment;
}

static int write_bytes(FILE *out, const void *bytes, size_t count)
{
    return fwrite(bytes, 1, count, out) == count ? FT_GGUF_WRITE_OK : FT_GGUF_WRITE_EIO;
}

static int write_zeros(FILE *out, uint64_t count)
{
    static const unsigned char zeros[64] = {0};

    while (count > 0u) {
        size_t chunk = count < sizeof(zeros) ? (size_t)count : sizeof(zeros);

        if (write_bytes(out, zeros, chunk) != FT_GGUF_WRITE_OK)
            return FT_GGUF_WRITE_EIO;
        count -= chunk;
    }
    return FT_GGUF_WRITE_OK;
}

/** The tensor table, built in memory (it is small), with each offset placed. */
static int build_table(ft_gguf_kv_buffer *table, const ft_gguf_write_tensor *tensors, uint32_t tensor_count,
                       uint64_t alignment)
{
    uint64_t offset = 0;

    for (uint32_t i = 0; i < tensor_count; i++) {
        const ft_gguf_write_tensor *t = &tensors[i];

        if (!t->name || t->n_dims == 0u || t->n_dims > 4u || (!t->data && t->byte_count))
            return FT_GGUF_WRITE_EINVAL;
        kv_append_string(table, t->name);
        kv_append_u32(table, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++)
            kv_append_u64(table, t->dims[d]);
        kv_append_u32(table, t->ggml_type);
        kv_append_u64(table, offset);
        offset = write_align_up(offset + t->byte_count, alignment);
    }
    return table->failed ? FT_GGUF_WRITE_ENOMEM : FT_GGUF_WRITE_OK;
}

/** Stream every tensor's bytes, each followed by the pad to the next aligned offset. */
static int write_tensor_data(FILE *out, const ft_gguf_write_tensor *tensors, uint32_t tensor_count,
                             uint64_t alignment)
{
    for (uint32_t i = 0; i < tensor_count; i++) {
        const ft_gguf_write_tensor *t = &tensors[i];
        uint64_t padded = write_align_up(t->byte_count, alignment);

        if (t->byte_count && write_bytes(out, t->data, (size_t)t->byte_count) != FT_GGUF_WRITE_OK)
            return FT_GGUF_WRITE_EIO;
        if (write_zeros(out, padded - t->byte_count) != FT_GGUF_WRITE_OK)
            return FT_GGUF_WRITE_EIO;
    }
    return FT_GGUF_WRITE_OK;
}

int ft_gguf_write_file(const char *path, const void *kv_bytes, uint64_t kv_byte_count, uint64_t kv_count,
                       const ft_gguf_write_tensor *tensors, uint32_t tensor_count, uint64_t alignment)
{
    ft_gguf_kv_buffer head;
    ft_gguf_kv_buffer table;
    FILE *out = NULL;
    int rc;

    if (!path || (!kv_bytes && kv_byte_count) || (!tensors && tensor_count) || alignment == 0u ||
        (alignment & (alignment - 1u)) != 0u)
        return FT_GGUF_WRITE_EINVAL;

    memset(&head, 0, sizeof(head));
    memset(&table, 0, sizeof(table));
    kv_append_u32(&head, GGUF_WRITE_MAGIC);
    kv_append_u32(&head, GGUF_WRITE_VERSION);
    kv_append_u64(&head, (uint64_t)tensor_count);
    kv_append_u64(&head, kv_count);
    rc = build_table(&table, tensors, tensor_count, alignment);
    if (rc == FT_GGUF_WRITE_OK && head.failed)
        rc = FT_GGUF_WRITE_ENOMEM;
    if (rc != FT_GGUF_WRITE_OK)
        goto cleanup;

    out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "ft_gguf_write: fopen('%s'): %s\n", path, strerror(errno));
        rc = FT_GGUF_WRITE_EIO;
        goto cleanup;
    }
    {
        uint64_t table_end = (uint64_t)head.used + kv_byte_count + (uint64_t)table.used;

        rc = write_bytes(out, head.bytes, head.used);
        if (rc == FT_GGUF_WRITE_OK && kv_byte_count)
            rc = write_bytes(out, kv_bytes, (size_t)kv_byte_count);
        if (rc == FT_GGUF_WRITE_OK)
            rc = write_bytes(out, table.bytes, table.used);
        if (rc == FT_GGUF_WRITE_OK)
            rc = write_zeros(out, write_align_up(table_end, alignment) - table_end);
    }
    if (rc == FT_GGUF_WRITE_OK)
        rc = write_tensor_data(out, tensors, tensor_count, alignment);

cleanup:
    if (out && fclose(out) != 0 && rc == FT_GGUF_WRITE_OK)
        rc = FT_GGUF_WRITE_EIO;
    if (rc == FT_GGUF_WRITE_EIO)
        fprintf(stderr, "ft_gguf_write: writing '%s' failed\n", path);
    ft_gguf_kv_free(&head);
    ft_gguf_kv_free(&table);
    return rc;
}
