/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_artifact_splice.c
 * @brief Implementation of the FTTW splice, its record, and tile extraction.
 *
 * @status None
 * @issues None
 * @todo None
 */

#define _DEFAULT_SOURCE

#include "ft_tt_artifact_splice.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ft_bfp.h"
#include "ft_tt_artifact_provenance.h"
#include "ft_tt_precision_ledger.h"
#include "ft_tt_tile_file.h"

/* The directory and header are read and written as raw structs, as the converter writes
 * them; a padded build would produce a file no other reader agrees with. */
typedef char splice_header_size_assert[(sizeof(ft_tt_artifact_header) == FT_TT_ARTIFACT_HEADER_BYTES) ? 1 : -1];
typedef char splice_entry_size_assert[(sizeof(ft_tt_artifact_entry) == FT_TT_ARTIFACT_ENTRY_BYTES) ? 1 : -1];

/** The provenance trailer's on-disk width (ft_tt_artifact_provenance.c). */
#define SPLICE_PROVENANCE_BYTES 64u

/** Bytes per copy step. The only buffer that scales with nothing. */
#define SPLICE_COPY_CHUNK (4u << 20)

/** An open source artifact: header, directory, and whatever followed the directory. */
typedef struct {
    int fd;
    uint64_t size;
    ft_tt_artifact_header header;
    ft_tt_artifact_entry *directory;
    int has_provenance;
    unsigned char provenance[SPLICE_PROVENANCE_BYTES];
    int has_splice_record;
} splice_source;

/** Where each source entry's new payload comes from; `file` NULL = keep the source's. */
typedef struct {
    const ft_tt_tile_file *file;
    uint32_t index;
} splice_replacement;

static void splice_error(char *error, size_t error_bytes, const char *format, ...)
    __attribute__((format(printf, 3, 4)));

static void splice_error(char *error, size_t error_bytes, const char *format, ...)
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

static int splice_pread(int fd, uint64_t offset, void *buffer, size_t bytes)
{
    unsigned char *cursor = (unsigned char *)buffer;

    while (bytes > 0u) {
        ssize_t got = pread(fd, cursor, bytes, (off_t)offset);

        if (got <= 0)
            return FT_TT_SPLICE_EIO;
        cursor += got;
        offset += (uint64_t)got;
        bytes -= (size_t)got;
    }
    return FT_TT_SPLICE_OK;
}

static uint64_t splice_align_up(uint64_t value)
{
    return (value + FT_TT_ARTIFACT_ALIGNMENT - 1u) / FT_TT_ARTIFACT_ALIGNMENT * FT_TT_ARTIFACT_ALIGNMENT;
}

/* ------------------------------------------------------------------------- */
/* Source                                                                    */
/* ------------------------------------------------------------------------- */

static void splice_source_close(splice_source *source)
{
    if (source->fd >= 0)
        close(source->fd);
    free(source->directory);
    memset(source, 0, sizeof(*source));
    source->fd = -1;
}

/** Every entry's name is terminated and its payload lies inside the data region. */
static int splice_source_entries_checked(const splice_source *source, const char *path, char *error,
                                         size_t error_bytes)
{
    const ft_tt_artifact_header *h = &source->header;

    for (uint32_t i = 0; i < h->tensor_count; i++) {
        const ft_tt_artifact_entry *e = &source->directory[i];

        if (memchr(e->name, '\0', sizeof(e->name)) == NULL || e->name[0] == '\0' || e->data_offset < h->data_offset ||
            e->data_offset > h->directory_offset || e->byte_count > h->directory_offset - e->data_offset) {
            splice_error(error, error_bytes, "%s: directory entry %u is unnamed or points outside the data region",
                         path, i);
            return FT_TT_SPLICE_ESOURCE;
        }
    }
    return FT_TT_SPLICE_OK;
}

/** What follows the directory: nothing, a provenance trailer, or a splice record. */
static int splice_source_tail_checked(splice_source *source, const char *path, char *error, size_t error_bytes)
{
    uint64_t tail = source->size - source->header.total_bytes;
    unsigned char magic[4];
    char why[256] = "";
    ft_tt_artifact_provenance record;

    if (tail == 0u)
        return FT_TT_SPLICE_OK;
    if (tail < sizeof(magic) || splice_pread(source->fd, source->size - sizeof(magic), magic, sizeof(magic)) != 0)
        goto unknown;
    if (get_u32(magic) == FT_TT_SPLICE_MAGIC) {
        source->has_splice_record = 1;
        return FT_TT_SPLICE_OK;
    }
    if (get_u32(magic) != FT_TT_PROVENANCE_MAGIC || tail != SPLICE_PROVENANCE_BYTES)
        goto unknown;
    if (ft_tt_artifact_provenance_read(path, &record) != FT_TT_PROVENANCE_OK ||
        ft_tt_artifact_provenance_verify_file(path, &record, why, sizeof(why)) != FT_TT_PROVENANCE_OK) {
        splice_error(error, error_bytes, "%s: its provenance trailer does not describe it: %s", path, why);
        return FT_TT_SPLICE_ESOURCE;
    }
    source->has_provenance = 1;
    return splice_pread(source->fd, source->header.total_bytes, source->provenance, sizeof(source->provenance));

unknown:
    splice_error(error, error_bytes, "%s: %llu B after the directory are neither a provenance trailer nor a splice "
                                     "record", path, (unsigned long long)tail);
    return FT_TT_SPLICE_ESOURCE;
}

/** Open and validate a source artifact. Its trailing record is classified, not judged. */
static int splice_source_open(const char *path, splice_source *source, char *error, size_t error_bytes)
{
    const ft_tt_artifact_header *h = &source->header;
    struct stat status;
    uint64_t directory_bytes;
    int rc;

    memset(source, 0, sizeof(*source));
    source->fd = open(path, O_RDONLY);
    if (source->fd < 0 || fstat(source->fd, &status) != 0) {
        splice_error(error, error_bytes, "%s: cannot open", path);
        return FT_TT_SPLICE_EIO;
    }
    source->size = (uint64_t)status.st_size;
    if (source->size < sizeof(source->header) ||
        splice_pread(source->fd, 0, &source->header, sizeof(source->header)) != 0) {
        splice_error(error, error_bytes, "%s: shorter than an FTTW header", path);
        return FT_TT_SPLICE_ESOURCE;
    }
    directory_bytes = (uint64_t)h->tensor_count * FT_TT_ARTIFACT_ENTRY_BYTES;
    if (h->magic != FT_TT_ARTIFACT_MAGIC || h->version < FT_TT_ARTIFACT_VERSION_MIN ||
        h->version > FT_TT_ARTIFACT_VERSION || (h->flags & ~FT_TT_ARTIFACT_FLAGS_KNOWN) ||
        h->data_offset < sizeof(source->header) || h->directory_offset < h->data_offset ||
        h->total_bytes != h->directory_offset + directory_bytes || h->total_bytes > source->size) {
        splice_error(error, error_bytes, "%s: not a well-formed FTTW v%u..v%u artifact (magic 0x%08x, version %u, "
                                         "flags 0x%08x, directory %llu + %u entries, total %llu, file %llu B)",
                     path, FT_TT_ARTIFACT_VERSION_MIN, FT_TT_ARTIFACT_VERSION, h->magic, h->version, h->flags,
                     (unsigned long long)h->directory_offset, h->tensor_count, (unsigned long long)h->total_bytes,
                     (unsigned long long)source->size);
        return FT_TT_SPLICE_ESOURCE;
    }
    source->directory = calloc(h->tensor_count ? h->tensor_count : 1u, sizeof(*source->directory));
    if (!source->directory)
        return FT_TT_SPLICE_ENOMEM;
    if (splice_pread(source->fd, h->directory_offset, source->directory, (size_t)directory_bytes) != 0) {
        splice_error(error, error_bytes, "%s: cannot read the directory", path);
        return FT_TT_SPLICE_EIO;
    }
    rc = splice_source_entries_checked(source, path, error, error_bytes);
    if (rc == FT_TT_SPLICE_OK)
        rc = splice_source_tail_checked(source, path, error, error_bytes);
    return rc;
}

static int splice_source_find(const splice_source *source, const char *name)
{
    for (uint32_t i = 0; i < source->header.tensor_count; i++)
        if (strncmp(source->directory[i].name, name, FT_TT_ARTIFACT_NAME_BYTES) == 0)
            return (int)i;
    return -1;
}

/* ------------------------------------------------------------------------- */
/* The rules                                                                 */
/* ------------------------------------------------------------------------- */

/**
 * The bytes the converter writes for a tensor of this entry's shape at a block-float
 * `format`: one strip per 32 logical rows, ceil(cols/32) tiles per strip, whatever the
 * layout (ft_tt_weight_convert.c, convert_tensor()). 0 when the stored grid disagrees with
 * the dims, which no converter output does.
 */
static uint64_t splice_converter_bytes(const ft_tt_artifact_entry *e, uint32_t format)
{
    uint64_t row_tiles = (e->dims[1] * e->dims[2] * e->dims[3] + 31u) / 32u;
    uint64_t col_tiles = (e->dims[0] + 31u) / 32u;
    int transposed = e->layout == FT_TT_LAYOUT_MATMUL_NMAJOR;
    ft_bfp_format bfp = format == FT_TT_FORMAT_BFP4_B ? FT_BFP_BFP4_B : FT_BFP_BFP8_B;

    if (e->tile_rows != (transposed ? col_tiles : row_tiles) || e->tile_cols != (transposed ? row_tiles : col_tiles))
        return 0u;
    return row_tiles * col_tiles * ft_bfp_tile_bytes(bfp);
}

/** Every rule one tile entry must pass against the source entry it replaces. */
static int splice_entry_checked(const ft_tt_artifact_entry *target, const ft_tt_tile_file_entry *tile,
                                const char *tile_path, char *error, size_t error_bytes)
{
    const char *refusal = ft_tt_ledger_map_refusal(ft_tt_ledger_role(target->name), target->n_dims);
    uint64_t want;

    if (refusal) {
        splice_error(error, error_bytes, "%s: '%s' may not be spliced: %s", tile_path, tile->name, refusal);
        return FT_TT_SPLICE_EROLE;
    }
    if (tile->format != FT_TT_FORMAT_BFP4_B && tile->format != FT_TT_FORMAT_BFP8_B) {
        splice_error(error, error_bytes, "%s: '%s' has format %u; a splice takes bfp4_b (%u) or bfp8_b (%u)",
                     tile_path, tile->name, tile->format, FT_TT_FORMAT_BFP4_B, FT_TT_FORMAT_BFP8_B);
        return FT_TT_SPLICE_EFORMAT;
    }
    if (tile->layout != target->layout) {
        splice_error(error, error_bytes, "%s: '%s' has layout %u, the artifact stores it in layout %u", tile_path,
                     tile->name, tile->layout, target->layout);
        return FT_TT_SPLICE_ELAYOUT;
    }
    want = splice_converter_bytes(target, tile->format);
    if (want == 0u || tile->n_bytes != want) {
        splice_error(error, error_bytes, "%s: '%s' is %llu B; the converter writes %llu B for [%llu, %llu] at %s",
                     tile_path, tile->name, (unsigned long long)tile->n_bytes, (unsigned long long)want,
                     (unsigned long long)target->dims[0], (unsigned long long)target->dims[1],
                     ft_tt_ledger_format_name((ft_tt_tensor_format)tile->format));
        return FT_TT_SPLICE_ESIZE;
    }
    return FT_TT_SPLICE_OK;
}

/** Map every tile entry onto the source entry it replaces, refusing on the first bad one. */
static int splice_plan(const splice_source *source, ft_tt_tile_file *const *files, uint32_t file_count,
                       splice_replacement *by_entry, uint32_t *replaced, char *error, size_t error_bytes)
{
    *replaced = 0;
    for (uint32_t f = 0; f < file_count; f++) {
        for (uint32_t t = 0; t < ft_tt_tile_file_count(files[f]); t++) {
            const ft_tt_tile_file_entry *tile = ft_tt_tile_file_entry_at(files[f], t);
            const char *path = ft_tt_tile_file_path(files[f]);
            int index = splice_source_find(source, tile->name);
            int rc;

            if (index < 0) {
                splice_error(error, error_bytes, "%s: '%s' is not a tensor of the source artifact", path,
                             tile->name);
                return FT_TT_SPLICE_ENAME;
            }
            if (by_entry[index].file) {
                splice_error(error, error_bytes, "'%s' is in two tile files: %s and %s", tile->name,
                             ft_tt_tile_file_path(by_entry[index].file), path);
                return FT_TT_SPLICE_EDUPLICATE;
            }
            rc = splice_entry_checked(&source->directory[index], tile, path, error, error_bytes);
            if (rc != FT_TT_SPLICE_OK)
                return rc;
            by_entry[index].file = files[f];
            by_entry[index].index = t;
            (*replaced)++;
        }
    }
    return FT_TT_SPLICE_OK;
}

/** Same path, or the same file under another name (a hard link, a relative spelling). */
static int splice_same_file(const char *a, const char *b)
{
    struct stat sa, sb;

    if (strcmp(a, b) == 0)
        return 1;
    return stat(a, &sa) == 0 && stat(b, &sb) == 0 && sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
}

/* ------------------------------------------------------------------------- */
/* Writing                                                                   */
/* ------------------------------------------------------------------------- */

static int splice_write_zeros(FILE *out, uint64_t bytes)
{
    static const unsigned char zeros[FT_TT_ARTIFACT_ALIGNMENT] = {0};

    while (bytes > 0u) {
        size_t chunk = bytes < sizeof(zeros) ? (size_t)bytes : sizeof(zeros);

        if (fwrite(zeros, 1, chunk, out) != chunk)
            return FT_TT_SPLICE_EIO;
        bytes -= chunk;
    }
    return FT_TT_SPLICE_OK;
}

/** Stream one payload, from the source or from a tile file, through `buffer`. */
static int splice_copy_payload(FILE *out, const splice_source *source, const ft_tt_artifact_entry *entry,
                               const splice_replacement *replacement, uint64_t bytes, unsigned char *buffer)
{
    uint64_t done = 0;

    while (done < bytes) {
        size_t chunk = bytes - done < SPLICE_COPY_CHUNK ? (size_t)(bytes - done) : SPLICE_COPY_CHUNK;
        int rc = replacement->file
                     ? ft_tt_tile_file_read_payload(replacement->file, replacement->index, done, buffer, chunk)
                     : splice_pread(source->fd, entry->data_offset + done, buffer, chunk);

        if (rc != 0 || fwrite(buffer, 1, chunk, out) != chunk)
            return FT_TT_SPLICE_EIO;
        done += chunk;
    }
    return FT_TT_SPLICE_OK;
}

static void splice_pack_record_entry(const ft_tt_artifact_entry *entry, const ft_tt_tile_file_entry *tile,
                                     unsigned char out[FT_TT_SPLICE_ENTRY_BYTES])
{
    memset(out, 0, FT_TT_SPLICE_ENTRY_BYTES);
    memcpy(out, entry->name, strnlen(entry->name, FT_TT_ARTIFACT_NAME_BYTES));
    put_u32(out + 128, tile->format);
    put_u32(out + 132, tile->layout);
    put_u64(out + 136, tile->n_bytes);
    memcpy(out + 144, tile->sha256, FT_SHA256_BYTES);
}

/** The record: one entry per replaced tensor in directory order, then the footer. */
static int splice_write_record(FILE *out, const splice_source *source, const splice_replacement *by_entry,
                               uint32_t replaced)
{
    unsigned char footer[FT_TT_SPLICE_FOOTER_BYTES];
    unsigned char packed[FT_TT_SPLICE_ENTRY_BYTES];

    for (uint32_t i = 0; i < source->header.tensor_count; i++) {
        if (!by_entry[i].file)
            continue;
        splice_pack_record_entry(&source->directory[i],
                                 ft_tt_tile_file_entry_at(by_entry[i].file, by_entry[i].index), packed);
        if (fwrite(packed, 1, sizeof(packed), out) != sizeof(packed))
            return FT_TT_SPLICE_EIO;
    }
    memset(footer, 0, sizeof(footer));
    if (source->has_provenance)
        memcpy(footer, source->provenance, SPLICE_PROVENANCE_BYTES);
    ft_sha256(source->directory, (size_t)source->header.tensor_count * FT_TT_ARTIFACT_ENTRY_BYTES, footer + 64);
    put_u64(footer + 96, source->header.total_bytes);
    put_u32(footer + 104, replaced);
    put_u32(footer + 108, source->has_provenance ? FT_TT_SPLICE_FLAG_BASE_PROVENANCE : 0u);
    put_u32(footer + 112, replaced * FT_TT_SPLICE_ENTRY_BYTES + FT_TT_SPLICE_FOOTER_BYTES);
    put_u32(footer + 116, FT_TT_SPLICE_VERSION);
    put_u32(footer + 120, 0u);
    put_u32(footer + 124, FT_TT_SPLICE_MAGIC);
    return fwrite(footer, 1, sizeof(footer), out) == sizeof(footer) ? FT_TT_SPLICE_OK : FT_TT_SPLICE_EIO;
}

/** Write the whole output: the converter's layout, then the record. */
static int splice_write(FILE *out, const splice_source *source, const splice_replacement *by_entry,
                        uint32_t replaced, unsigned char *buffer)
{
    ft_tt_artifact_header header = source->header;
    uint32_t count = header.tensor_count;
    ft_tt_artifact_entry *directory = calloc(count ? count : 1u, sizeof(*directory));
    uint64_t cursor = header.data_offset;
    int rc;

    if (!directory)
        return FT_TT_SPLICE_ENOMEM;
    rc = fwrite(&header, sizeof(header), 1, out) == 1 ? FT_TT_SPLICE_OK : FT_TT_SPLICE_EIO;
    if (rc == FT_TT_SPLICE_OK)
        rc = splice_write_zeros(out, header.data_offset - sizeof(header));
    for (uint32_t i = 0; i < count && rc == FT_TT_SPLICE_OK; i++) {
        const ft_tt_tile_file_entry *tile =
            by_entry[i].file ? ft_tt_tile_file_entry_at(by_entry[i].file, by_entry[i].index) : NULL;

        directory[i] = source->directory[i];
        if (tile) {
            directory[i].format = tile->format;
            directory[i].byte_count = tile->n_bytes;
        }
        directory[i].data_offset = cursor;
        rc = splice_copy_payload(out, source, &source->directory[i], &by_entry[i], directory[i].byte_count, buffer);
        cursor += directory[i].byte_count;
        if (rc == FT_TT_SPLICE_OK)
            rc = splice_write_zeros(out, splice_align_up(cursor) - cursor);
        cursor = splice_align_up(cursor);
    }
    header.directory_offset = cursor;
    header.total_bytes = cursor + (uint64_t)count * FT_TT_ARTIFACT_ENTRY_BYTES;
    if (rc == FT_TT_SPLICE_OK && fwrite(directory, sizeof(*directory), count, out) != count)
        rc = FT_TT_SPLICE_EIO;
    if (rc == FT_TT_SPLICE_OK)
        rc = splice_write_record(out, source, by_entry, replaced);
    if (rc == FT_TT_SPLICE_OK && (fseek(out, 0, SEEK_SET) != 0 || fwrite(&header, sizeof(header), 1, out) != 1))
        rc = FT_TT_SPLICE_EIO;
    free(directory);
    return rc;
}

/** Open every tile file (each verifies itself) and refuse an output that aliases an input. */
static int splice_open_inputs(const char *source_path, const char *const *tile_paths, uint32_t tile_path_count,
                              const char *output_path, ft_tt_tile_file **files, char *error, size_t error_bytes)
{
    char tile_error[FT_TT_TILE_FILE_ERROR_BYTES] = "";

    if (splice_same_file(output_path, source_path)) {
        splice_error(error, error_bytes, "the output %s is the source artifact", output_path);
        return FT_TT_SPLICE_ESAMEPATH;
    }
    for (uint32_t f = 0; f < tile_path_count; f++) {
        if (!tile_paths[f]) {
            splice_error(error, error_bytes, "tile file %u has no path", f);
            return FT_TT_SPLICE_EINVAL;
        }
        if (splice_same_file(output_path, tile_paths[f])) {
            splice_error(error, error_bytes, "the output %s is tile file %u", output_path, f);
            return FT_TT_SPLICE_ESAMEPATH;
        }
        if (ft_tt_tile_file_open(tile_paths[f], &files[f], tile_error, sizeof(tile_error)) != FT_TT_TILE_FILE_OK) {
            splice_error(error, error_bytes, "%s", tile_error);
            return FT_TT_SPLICE_ETILEFILE;
        }
    }
    return FT_TT_SPLICE_OK;
}

/** Write `output_path` through a temp file and a rename. */
static int splice_produce(const splice_source *source, const splice_replacement *by_entry, uint32_t replaced,
                          const char *output_path, char *error, size_t error_bytes)
{
    char temp_path[1100];
    unsigned char *buffer = malloc(SPLICE_COPY_CHUNK);
    FILE *out = NULL;
    int rc;

    if (!buffer)
        return FT_TT_SPLICE_ENOMEM;
    if (snprintf(temp_path, sizeof(temp_path), "%s.tmp.%ld", output_path, (long)getpid()) >= (int)sizeof(temp_path)) {
        free(buffer);
        splice_error(error, error_bytes, "the output path is too long");
        return FT_TT_SPLICE_EINVAL;
    }
    out = fopen(temp_path, "wb");
    if (!out) {
        free(buffer);
        splice_error(error, error_bytes, "%s: cannot create", temp_path);
        return FT_TT_SPLICE_EIO;
    }
    rc = splice_write(out, source, by_entry, replaced, buffer);
    if (fclose(out) != 0 && rc == FT_TT_SPLICE_OK)
        rc = FT_TT_SPLICE_EIO;
    if (rc == FT_TT_SPLICE_OK && rename(temp_path, output_path) != 0)
        rc = FT_TT_SPLICE_EIO;
    if (rc != FT_TT_SPLICE_OK) {
        remove(temp_path);
        splice_error(error, error_bytes, "writing %s failed (%d)", output_path, rc);
    }
    free(buffer);
    return rc;
}

int ft_tt_artifact_splice(const char *source_path, const char *const *tile_paths, uint32_t tile_path_count,
                          const char *output_path, char *error, size_t error_bytes)
{
    splice_source source;
    ft_tt_tile_file **files = NULL;
    splice_replacement *by_entry = NULL;
    uint32_t replaced = 0;
    int rc;

    memset(&source, 0, sizeof(source));
    source.fd = -1;
    if (!source_path || !tile_paths || tile_path_count == 0u || !output_path) {
        splice_error(error, error_bytes, "a splice needs a source, at least one tile file, and an output");
        return FT_TT_SPLICE_EINVAL;
    }
    files = calloc(tile_path_count, sizeof(*files));
    if (!files)
        return FT_TT_SPLICE_ENOMEM;
    rc = splice_open_inputs(source_path, tile_paths, tile_path_count, output_path, files, error, error_bytes);
    if (rc == FT_TT_SPLICE_OK)
        rc = splice_source_open(source_path, &source, error, error_bytes);
    if (rc == FT_TT_SPLICE_OK && source.has_splice_record) {
        splice_error(error, error_bytes, "%s already carries a splice record; splice from its unspliced base",
                     source_path);
        rc = FT_TT_SPLICE_ESOURCE;
    }
    if (rc == FT_TT_SPLICE_OK) {
        by_entry = calloc(source.header.tensor_count ? source.header.tensor_count : 1u, sizeof(*by_entry));
        rc = by_entry ? splice_plan(&source, files, tile_path_count, by_entry, &replaced, error, error_bytes)
                      : FT_TT_SPLICE_ENOMEM;
    }
    if (rc == FT_TT_SPLICE_OK)
        rc = splice_produce(&source, by_entry, replaced, output_path, error, error_bytes);

    free(by_entry);
    for (uint32_t f = 0; f < tile_path_count; f++)
        ft_tt_tile_file_close(files[f]);
    free(files);
    splice_source_close(&source);
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Extraction                                                                */
/* ------------------------------------------------------------------------- */

int ft_tt_artifact_extract_tiles(const char *artifact_path, const char *const *names, uint32_t name_count,
                                 const char *tile_path, char *error, size_t error_bytes)
{
    splice_source source;
    ft_tt_tile_file_payload *payloads = NULL;
    void *mapping = MAP_FAILED;
    int rc;

    memset(&source, 0, sizeof(source));
    source.fd = -1;
    if (!artifact_path || (name_count && !names) || !tile_path)
        return FT_TT_SPLICE_EINVAL;
    rc = splice_source_open(artifact_path, &source, error, error_bytes);
    if (rc == FT_TT_SPLICE_OK) {
        payloads = calloc(name_count ? name_count : 1u, sizeof(*payloads));
        mapping = mmap(NULL, (size_t)source.size, PROT_READ, MAP_PRIVATE, source.fd, 0);
        rc = !payloads ? FT_TT_SPLICE_ENOMEM : mapping == MAP_FAILED ? FT_TT_SPLICE_EIO : FT_TT_SPLICE_OK;
    }
    for (uint32_t i = 0; i < name_count && rc == FT_TT_SPLICE_OK; i++) {
        int index = names[i] ? splice_source_find(&source, names[i]) : -1;

        if (index < 0) {
            splice_error(error, error_bytes, "%s: '%s' is not a tensor of the artifact", artifact_path,
                         names[i] ? names[i] : "(null)");
            rc = FT_TT_SPLICE_ENAME;
            break;
        }
        payloads[i].name = source.directory[index].name;
        payloads[i].format = source.directory[index].format;
        payloads[i].layout = source.directory[index].layout;
        payloads[i].payload = (const unsigned char *)mapping + source.directory[index].data_offset;
        payloads[i].n_bytes = source.directory[index].byte_count;
    }
    if (rc == FT_TT_SPLICE_OK) {
        char tile_error[FT_TT_TILE_FILE_ERROR_BYTES] = "";

        if (ft_tt_tile_file_write(tile_path, payloads, name_count, tile_error, sizeof(tile_error)) !=
            FT_TT_TILE_FILE_OK) {
            splice_error(error, error_bytes, "%s", tile_error);
            rc = FT_TT_SPLICE_ETILEFILE;
        }
    }
    if (mapping != MAP_FAILED)
        munmap(mapping, (size_t)source.size);
    free(payloads);
    splice_source_close(&source);
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Reading the record back                                                   */
/* ------------------------------------------------------------------------- */

void ft_tt_splice_record_free(ft_tt_splice_record *record)
{
    if (!record)
        return;
    free(record->entries);
    memset(record, 0, sizeof(*record));
}

static void splice_unpack_footer(const unsigned char *footer, ft_tt_splice_record *record)
{
    memcpy(record->base_provenance, footer, sizeof(record->base_provenance));
    memcpy(record->base_directory_sha256, footer + 64, FT_SHA256_BYTES);
    record->base_total_bytes = get_u64(footer + 96);
    record->entry_count = get_u32(footer + 104);
    record->flags = get_u32(footer + 108);
}

int ft_tt_splice_record_read(const char *artifact_path, ft_tt_splice_record *record)
{
    unsigned char footer[FT_TT_SPLICE_FOOTER_BYTES];
    unsigned char packed[FT_TT_SPLICE_ENTRY_BYTES];
    ft_tt_artifact_header header;
    struct stat status;
    uint64_t record_bytes;
    int fd, rc = FT_TT_SPLICE_OK;

    if (!artifact_path || !record)
        return FT_TT_SPLICE_EINVAL;
    memset(record, 0, sizeof(*record));
    fd = open(artifact_path, O_RDONLY);
    if (fd < 0 || fstat(fd, &status) != 0) {
        if (fd >= 0)
            close(fd);
        return FT_TT_SPLICE_EIO;
    }
    if ((uint64_t)status.st_size < sizeof(header) + sizeof(footer) ||
        splice_pread(fd, 0, &header, sizeof(header)) != 0 ||
        splice_pread(fd, (uint64_t)status.st_size - sizeof(footer), footer, sizeof(footer)) != 0 ||
        get_u32(footer + 124) != FT_TT_SPLICE_MAGIC) {
        close(fd);
        return FT_TT_SPLICE_EABSENT;
    }
    splice_unpack_footer(footer, record);
    record_bytes = (uint64_t)record->entry_count * FT_TT_SPLICE_ENTRY_BYTES + FT_TT_SPLICE_FOOTER_BYTES;
    /* The record sits right after the directory and says how long it is; both must agree. */
    if (get_u32(footer + 116) != FT_TT_SPLICE_VERSION || get_u32(footer + 120) != 0u ||
        get_u32(footer + 112) != record_bytes || header.total_bytes + record_bytes != (uint64_t)status.st_size)
        rc = FT_TT_SPLICE_ESOURCE;
    if (rc == FT_TT_SPLICE_OK) {
        record->entries = calloc(record->entry_count ? record->entry_count : 1u, sizeof(*record->entries));
        rc = record->entries ? FT_TT_SPLICE_OK : FT_TT_SPLICE_ENOMEM;
    }
    for (uint32_t i = 0; i < record->entry_count && rc == FT_TT_SPLICE_OK; i++) {
        ft_tt_splice_record_entry *entry = &record->entries[i];

        if (splice_pread(fd, header.total_bytes + (uint64_t)i * FT_TT_SPLICE_ENTRY_BYTES, packed, sizeof(packed))) {
            rc = FT_TT_SPLICE_EIO;
            break;
        }
        memcpy(entry->name, packed, FT_TT_ARTIFACT_NAME_BYTES);
        entry->name[FT_TT_ARTIFACT_NAME_BYTES - 1] = '\0';
        entry->format = get_u32(packed + 128);
        entry->layout = get_u32(packed + 132);
        entry->n_bytes = get_u64(packed + 136);
        memcpy(entry->sha256, packed + 144, FT_SHA256_BYTES);
    }
    close(fd);
    if (rc != FT_TT_SPLICE_OK)
        ft_tt_splice_record_free(record);
    return rc;
}
