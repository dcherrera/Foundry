/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_tile_file.c
 * @brief Implementation of the MXTL tile file reader and writer.
 *
 * @status None
 * @issues None
 * @todo None
 */

#define _DEFAULT_SOURCE

#include "ft_tt_tile_file.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

/** Bytes of an entry's fixed fields after the name: format, layout, n_bytes, sha256. */
#define TILE_FILE_ENTRY_FIXED_BYTES (4u + 4u + 8u + FT_SHA256_BYTES)

/** Payload bytes hashed per read when verifying. */
#define TILE_FILE_HASH_CHUNK (1u << 20)

struct ft_tt_tile_file {
    int fd;
    char path[1024];
    ft_tt_tile_file_entry *entries;
    uint32_t count;
};

static void tile_file_error(char *error, size_t error_bytes, const char *format, ...)
    __attribute__((format(printf, 3, 4)));

static void tile_file_error(char *error, size_t error_bytes, const char *format, ...)
{
    va_list args;

    if (!error || error_bytes == 0u)
        return;
    va_start(args, format);
    vsnprintf(error, error_bytes, format, args);
    va_end(args);
}

static void put_u32(unsigned char *out, uint32_t value)
{
    for (unsigned i = 0; i < 4u; i++)
        out[i] = (unsigned char)(value >> (8u * i));
}

static void put_u64(unsigned char *out, uint64_t value)
{
    for (unsigned i = 0; i < 8u; i++)
        out[i] = (unsigned char)(value >> (8u * i));
}

static uint32_t get_u32(const unsigned char *in)
{
    return (uint32_t)in[0] | ((uint32_t)in[1] << 8) | ((uint32_t)in[2] << 16) | ((uint32_t)in[3] << 24);
}

static uint64_t get_u64(const unsigned char *in)
{
    return (uint64_t)get_u32(in) | ((uint64_t)get_u32(in + 4) << 32);
}

/* ------------------------------------------------------------------------- */
/* Writer                                                                    */
/* ------------------------------------------------------------------------- */

/** The checks a reader makes on names, made before anything is written. */
static int tile_file_names_checked(const ft_tt_tile_file_payload *entries, uint32_t count, char *error,
                                   size_t error_bytes)
{
    for (uint32_t i = 0; i < count; i++) {
        size_t length = entries[i].name ? strlen(entries[i].name) : 0u;

        if (length == 0u || length >= FT_TT_TILE_FILE_NAME_BYTES) {
            tile_file_error(error, error_bytes, "entry %u: a name must be 1..%u bytes", i,
                            FT_TT_TILE_FILE_NAME_BYTES - 1u);
            return FT_TT_TILE_FILE_EINVAL;
        }
        if (entries[i].n_bytes && !entries[i].payload) {
            tile_file_error(error, error_bytes, "entry %u (%s): no payload", i, entries[i].name);
            return FT_TT_TILE_FILE_EINVAL;
        }
        for (uint32_t j = 0; j < i; j++)
            if (strcmp(entries[i].name, entries[j].name) == 0) {
                tile_file_error(error, error_bytes, "'%s' is listed twice (entries %u and %u)", entries[i].name, j,
                                i);
                return FT_TT_TILE_FILE_EDUPLICATE;
            }
    }
    return FT_TT_TILE_FILE_OK;
}

static int tile_file_write_entry(FILE *out, const ft_tt_tile_file_payload *entry)
{
    unsigned char fixed[TILE_FILE_ENTRY_FIXED_BYTES];
    unsigned char length[4];
    uint32_t name_len = (uint32_t)strlen(entry->name);

    put_u32(length, name_len);
    put_u32(fixed, entry->format);
    put_u32(fixed + 4, entry->layout);
    put_u64(fixed + 8, entry->n_bytes);
    ft_sha256(entry->payload, (size_t)entry->n_bytes, fixed + 16);
    if (fwrite(length, 1, sizeof(length), out) != sizeof(length) ||
        fwrite(entry->name, 1, name_len, out) != name_len || fwrite(fixed, 1, sizeof(fixed), out) != sizeof(fixed))
        return FT_TT_TILE_FILE_EIO;
    if (entry->n_bytes && fwrite(entry->payload, 1, (size_t)entry->n_bytes, out) != entry->n_bytes)
        return FT_TT_TILE_FILE_EIO;
    return FT_TT_TILE_FILE_OK;
}

int ft_tt_tile_file_write(const char *path, const ft_tt_tile_file_payload *entries, uint32_t count, char *error,
                          size_t error_bytes)
{
    unsigned char header[FT_TT_TILE_FILE_HEADER_BYTES];
    FILE *out;
    int rc;

    if (!path || (count && !entries)) {
        tile_file_error(error, error_bytes, "no path or no entries");
        return FT_TT_TILE_FILE_EINVAL;
    }
    rc = tile_file_names_checked(entries, count, error, error_bytes);
    if (rc != FT_TT_TILE_FILE_OK)
        return rc;

    out = fopen(path, "wb");
    if (!out) {
        tile_file_error(error, error_bytes, "%s: cannot create", path);
        return FT_TT_TILE_FILE_EIO;
    }
    put_u32(header, FT_TT_TILE_FILE_MAGIC);
    put_u32(header + 4, FT_TT_TILE_FILE_VERSION);
    put_u32(header + 8, count);
    put_u32(header + 12, 0u);
    rc = fwrite(header, 1, sizeof(header), out) == sizeof(header) ? FT_TT_TILE_FILE_OK : FT_TT_TILE_FILE_EIO;
    for (uint32_t i = 0; i < count && rc == FT_TT_TILE_FILE_OK; i++)
        rc = tile_file_write_entry(out, &entries[i]);
    if (fclose(out) != 0 && rc == FT_TT_TILE_FILE_OK)
        rc = FT_TT_TILE_FILE_EIO;
    if (rc != FT_TT_TILE_FILE_OK) {
        tile_file_error(error, error_bytes, "%s: write failed", path);
        remove(path);
    }
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Reader                                                                    */
/* ------------------------------------------------------------------------- */

/** pread exactly `bytes` at `offset`, or fail. */
static int tile_file_pread(int fd, uint64_t offset, void *buffer, size_t bytes)
{
    unsigned char *cursor = (unsigned char *)buffer;

    while (bytes > 0u) {
        ssize_t got = pread(fd, cursor, bytes, (off_t)offset);

        if (got <= 0)
            return FT_TT_TILE_FILE_EIO;
        cursor += got;
        offset += (uint64_t)got;
        bytes -= (size_t)got;
    }
    return FT_TT_TILE_FILE_OK;
}

/** Hash `bytes` of the file from `offset` and compare with `want`. */
static int tile_file_payload_hash_matches(int fd, uint64_t offset, uint64_t bytes,
                                          const unsigned char want[FT_SHA256_BYTES], unsigned char *buffer)
{
    unsigned char got[FT_SHA256_BYTES];
    ft_sha256_ctx ctx;

    ft_sha256_init(&ctx);
    while (bytes > 0u) {
        size_t chunk = bytes < TILE_FILE_HASH_CHUNK ? (size_t)bytes : TILE_FILE_HASH_CHUNK;

        if (tile_file_pread(fd, offset, buffer, chunk) != FT_TT_TILE_FILE_OK)
            return FT_TT_TILE_FILE_EIO;
        ft_sha256_update(&ctx, buffer, chunk);
        offset += chunk;
        bytes -= chunk;
    }
    ft_sha256_final(&ctx, got);
    return memcmp(got, want, FT_SHA256_BYTES) == 0 ? FT_TT_TILE_FILE_OK : FT_TT_TILE_FILE_ESHA;
}

/** Parse entry `index` at `*cursor` (file `size` bytes long), verifying its payload. */
static int tile_file_read_entry(ft_tt_tile_file *file, uint32_t index, uint64_t size, uint64_t *cursor,
                                unsigned char *buffer, char *error, size_t error_bytes)
{
    ft_tt_tile_file_entry *entry = &file->entries[index];
    unsigned char fixed[TILE_FILE_ENTRY_FIXED_BYTES];
    unsigned char length[4];
    uint32_t name_len;
    int rc;

    if (size - *cursor < sizeof(length) || tile_file_pread(file->fd, *cursor, length, sizeof(length)) != 0)
        goto truncated;
    name_len = get_u32(length);
    *cursor += sizeof(length);
    if (name_len == 0u || name_len >= FT_TT_TILE_FILE_NAME_BYTES) {
        tile_file_error(error, error_bytes, "%s: entry %u: name length %u is not 1..%u", file->path, index, name_len,
                        FT_TT_TILE_FILE_NAME_BYTES - 1u);
        return FT_TT_TILE_FILE_EFORMAT;
    }
    if (size - *cursor < (uint64_t)name_len + sizeof(fixed) ||
        tile_file_pread(file->fd, *cursor, entry->name, name_len) != 0 ||
        tile_file_pread(file->fd, *cursor + name_len, fixed, sizeof(fixed)) != 0)
        goto truncated;
    entry->name[name_len] = '\0';
    if (strlen(entry->name) != name_len) {
        tile_file_error(error, error_bytes, "%s: entry %u: the name holds a NUL", file->path, index);
        return FT_TT_TILE_FILE_EFORMAT;
    }
    *cursor += (uint64_t)name_len + sizeof(fixed);
    entry->format = get_u32(fixed);
    entry->layout = get_u32(fixed + 4);
    entry->n_bytes = get_u64(fixed + 8);
    memcpy(entry->sha256, fixed + 16, FT_SHA256_BYTES);
    entry->payload_offset = *cursor;
    if (entry->n_bytes > size - *cursor)
        goto truncated;
    rc = tile_file_payload_hash_matches(file->fd, *cursor, entry->n_bytes, entry->sha256, buffer);
    if (rc != FT_TT_TILE_FILE_OK) {
        tile_file_error(error, error_bytes, "%s: entry %u (%s): %s", file->path, index, entry->name,
                        rc == FT_TT_TILE_FILE_ESHA ? "payload sha256 mismatch" : "payload unreadable");
        return rc;
    }
    *cursor += entry->n_bytes;
    for (uint32_t j = 0; j < index; j++)
        if (strcmp(file->entries[j].name, entry->name) == 0) {
            tile_file_error(error, error_bytes, "%s: '%s' is listed twice (entries %u and %u)", file->path,
                            entry->name, j, index);
            return FT_TT_TILE_FILE_EDUPLICATE;
        }
    return FT_TT_TILE_FILE_OK;

truncated:
    tile_file_error(error, error_bytes, "%s: entry %u runs past the end of the file (%llu B)", file->path, index,
                    (unsigned long long)size);
    return FT_TT_TILE_FILE_ETRUNCATED;
}

/** Header checks; fills `file->count` and allocates the entry table. */
static int tile_file_read_header(ft_tt_tile_file *file, uint64_t size, char *error, size_t error_bytes)
{
    unsigned char header[FT_TT_TILE_FILE_HEADER_BYTES];
    uint64_t smallest_entry = 4u + 1u + TILE_FILE_ENTRY_FIXED_BYTES;

    if (size < sizeof(header) || tile_file_pread(file->fd, 0, header, sizeof(header)) != 0) {
        tile_file_error(error, error_bytes, "%s: shorter than the %u-byte header", file->path,
                        FT_TT_TILE_FILE_HEADER_BYTES);
        return FT_TT_TILE_FILE_ETRUNCATED;
    }
    if (get_u32(header) != FT_TT_TILE_FILE_MAGIC || get_u32(header + 4) != FT_TT_TILE_FILE_VERSION ||
        get_u32(header + 12) != 0u) {
        tile_file_error(error, error_bytes, "%s: magic 0x%08x version %u reserved %u is not MXTL v%u", file->path,
                        get_u32(header), get_u32(header + 4), get_u32(header + 12), FT_TT_TILE_FILE_VERSION);
        return FT_TT_TILE_FILE_EFORMAT;
    }
    file->count = get_u32(header + 8);
    /* Bound the table by what the file could hold before allocating it. */
    if ((uint64_t)file->count > (size - sizeof(header)) / smallest_entry) {
        tile_file_error(error, error_bytes, "%s: %u entries cannot fit in %llu B", file->path, file->count,
                        (unsigned long long)size);
        return FT_TT_TILE_FILE_ETRUNCATED;
    }
    file->entries = calloc(file->count ? file->count : 1u, sizeof(*file->entries));
    return file->entries ? FT_TT_TILE_FILE_OK : FT_TT_TILE_FILE_ENOMEM;
}

int ft_tt_tile_file_open(const char *path, ft_tt_tile_file **out, char *error, size_t error_bytes)
{
    ft_tt_tile_file *file = NULL;
    unsigned char *buffer = NULL;
    struct stat status;
    uint64_t cursor = FT_TT_TILE_FILE_HEADER_BYTES;
    int rc;

    if (!out)
        return FT_TT_TILE_FILE_EINVAL;
    *out = NULL;
    if (!path)
        return FT_TT_TILE_FILE_EINVAL;
    file = calloc(1, sizeof(*file));
    if (file)
        file->fd = -1;
    buffer = malloc(TILE_FILE_HASH_CHUNK);
    if (!file || !buffer) {
        rc = FT_TT_TILE_FILE_ENOMEM;
        goto fail;
    }
    snprintf(file->path, sizeof(file->path), "%s", path);
    file->fd = open(path, O_RDONLY);
    if (file->fd < 0 || fstat(file->fd, &status) != 0) {
        tile_file_error(error, error_bytes, "%s: cannot open", path);
        rc = FT_TT_TILE_FILE_EIO;
        goto fail;
    }
    rc = tile_file_read_header(file, (uint64_t)status.st_size, error, error_bytes);
    for (uint32_t i = 0; i < file->count && rc == FT_TT_TILE_FILE_OK; i++)
        rc = tile_file_read_entry(file, i, (uint64_t)status.st_size, &cursor, buffer, error, error_bytes);
    if (rc == FT_TT_TILE_FILE_OK && cursor != (uint64_t)status.st_size) {
        tile_file_error(error, error_bytes, "%s: %llu B after the last entry", path,
                        (unsigned long long)((uint64_t)status.st_size - cursor));
        rc = FT_TT_TILE_FILE_EFORMAT;
    }
    if (rc != FT_TT_TILE_FILE_OK)
        goto fail;
    free(buffer);
    *out = file;
    return FT_TT_TILE_FILE_OK;

fail:
    free(buffer);
    ft_tt_tile_file_close(file);
    return rc;
}

void ft_tt_tile_file_close(ft_tt_tile_file *file)
{
    if (!file)
        return;
    if (file->fd >= 0)
        close(file->fd);
    free(file->entries);
    free(file);
}

const char *ft_tt_tile_file_path(const ft_tt_tile_file *file)
{
    return file ? file->path : NULL;
}

uint32_t ft_tt_tile_file_count(const ft_tt_tile_file *file)
{
    return file ? file->count : 0u;
}

const ft_tt_tile_file_entry *ft_tt_tile_file_entry_at(const ft_tt_tile_file *file, uint32_t index)
{
    return file && index < file->count ? &file->entries[index] : NULL;
}

const ft_tt_tile_file_entry *ft_tt_tile_file_find(const ft_tt_tile_file *file, const char *name)
{
    if (!file || !name)
        return NULL;
    for (uint32_t i = 0; i < file->count; i++)
        if (strcmp(file->entries[i].name, name) == 0)
            return &file->entries[i];
    return NULL;
}

int ft_tt_tile_file_read_payload(const ft_tt_tile_file *file, uint32_t index, uint64_t offset, void *buffer,
                                 size_t bytes)
{
    const ft_tt_tile_file_entry *entry = ft_tt_tile_file_entry_at(file, index);

    if (!entry || (bytes && !buffer) || offset > entry->n_bytes || bytes > entry->n_bytes - offset)
        return FT_TT_TILE_FILE_EINVAL;
    return tile_file_pread(file->fd, entry->payload_offset + offset, buffer, bytes);
}
