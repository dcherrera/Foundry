/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_fttw.c
 * @brief Implementation of the FTTW artifact loader. See ft_tt_fttw.h.
 *
 * @status Reads the production MiniCPM5-1B artifact and uploads real tensors.
 * @issues None
 * @todo None
 */

#include "ft_tt_fttw.h"

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "convert/ft_float_convert.h"
#include "ft_tt_device.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_tilize.h"

/** How much of a tensor to hand the accessor per call when the caller says "default". */
#define FTTW_DEFAULT_CHUNK_BYTES 0x10000u

int ft_tt_fttw_open(const char *path, ft_tt_fttw *out_artifact)
{
    struct stat status;
    const unsigned char *base;
    uint64_t directory_bytes;
    int fd;

    if (!path || !out_artifact)
        return FT_TT_EINVAL;
    memset(out_artifact, 0, sizeof(*out_artifact));

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        ft_tt_log_error("fttw: cannot open %s", path);
        return FT_TT_EIO;
    }
    if (fstat(fd, &status) != 0 || (size_t)status.st_size < FT_TT_ARTIFACT_HEADER_BYTES) {
        ft_tt_log_error("fttw: %s is shorter than a header", path);
        close(fd);
        return FT_TT_EIO;
    }

    base = mmap(NULL, (size_t)status.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        ft_tt_log_error("fttw: cannot map %s", path);
        close(fd);
        return FT_TT_EIO;
    }

    /* A RANGE of versions, because v1 and v2 differ only in fields this loader does not
     * interpret: v2 uses the entry's spare word as `layout` and stores a matmul weight
     * transposed. The loader is a byte copy either way; it is the CONSUMER that has to
     * read `layout`, which is 0 on every v1 entry because that word was written as zero
     * padding. A FUTURE version is still refused, which is the direction that could be
     * silently wrong here. */
    memcpy(&out_artifact->header, base, sizeof(out_artifact->header));
    if (out_artifact->header.magic != FT_TT_ARTIFACT_MAGIC ||
        out_artifact->header.version < FT_TT_ARTIFACT_VERSION_MIN ||
        out_artifact->header.version > FT_TT_ARTIFACT_VERSION) {
        ft_tt_log_error("fttw: %s magic 0x%08x version %u is not an FTTW v%u..v%u artifact", path,
                        out_artifact->header.magic, out_artifact->header.version,
                        FT_TT_ARTIFACT_VERSION_MIN, FT_TT_ARTIFACT_VERSION);
        munmap((void *)base, (size_t)status.st_size);
        close(fd);
        return FT_TT_EINVAL;
    }

    /* A flag bit changes what the PAYLOAD MEANS — the rotary one says `attn_q`/`attn_k`
     * hold split-half rows where GGUF had interleaved ones — so an unknown bit is refused
     * rather than ignored. Ignoring it is exactly the silent-wrong-answer shape the
     * `layout` field exists to prevent. */
    if (out_artifact->header.flags & ~FT_TT_ARTIFACT_FLAGS_KNOWN) {
        ft_tt_log_error("fttw: %s header flags 0x%08x carry bits this build does not know", path,
                        out_artifact->header.flags);
        munmap((void *)base, (size_t)status.st_size);
        close(fd);
        return FT_TT_EINVAL;
    }

    /* The directory is written last, so a truncated file has a valid-looking header and
     * no directory. Bound it before anything indexes it. */
    directory_bytes = (uint64_t)out_artifact->header.tensor_count * FT_TT_ARTIFACT_ENTRY_BYTES;
    if (out_artifact->header.directory_offset + directory_bytes > (uint64_t)status.st_size) {
        ft_tt_log_error("fttw: %s directory of %u entries at %llu does not fit %lld bytes", path,
                        out_artifact->header.tensor_count,
                        (unsigned long long)out_artifact->header.directory_offset,
                        (long long)status.st_size);
        munmap((void *)base, (size_t)status.st_size);
        close(fd);
        return FT_TT_EINVAL;
    }

    out_artifact->base = base;
    out_artifact->bytes = (size_t)status.st_size;
    out_artifact->fd = fd;
    out_artifact->directory =
        (const ft_tt_artifact_entry *)(base + out_artifact->header.directory_offset);
    return FT_TT_OK;
}

void ft_tt_fttw_close(ft_tt_fttw *artifact)
{
    if (!artifact)
        return;
    if (artifact->base)
        munmap((void *)artifact->base, artifact->bytes);
    if (artifact->fd > 0)
        close(artifact->fd);
    memset(artifact, 0, sizeof(*artifact));
}

const ft_tt_artifact_entry *ft_tt_fttw_find(const ft_tt_fttw *artifact, const char *name)
{
    uint32_t index;

    if (!artifact || !artifact->directory || !name)
        return NULL;
    for (index = 0; index < artifact->header.tensor_count; index++) {
        const ft_tt_artifact_entry *entry = &artifact->directory[index];

        if (strncmp(entry->name, name, sizeof(entry->name)) == 0)
            return entry;
    }
    return NULL;
}

/** A Bfp6_b tile: 64 B of shared exponents + 768 B of 6-bit datums (convert/ft_bfp.c). A DRAM
 *  format only — no unpacker reads it, so it has no L1 CB size in ft_tt_op_l1_layout.h. */
#define FTTW_TILE_BYTES_BFP6_B 832u

uint32_t ft_tt_fttw_expert_count(const ft_tt_artifact_entry *entry)
{
    if (!entry)
        return 0u;
    return entry->n_dims == 3u && (entry->role == FT_TT_ROLE_EXPERT_UP || entry->role == FT_TT_ROLE_EXPERT_DOWN)
               ? (uint32_t)entry->dims[2]
               : 1u;
}

uint64_t ft_tt_fttw_expert_stride(const ft_tt_artifact_entry *entry)
{
    if (!entry)
        return 0u;
    return (uint64_t)entry->tile_rows * entry->tile_cols * ft_tt_fttw_tile_bytes(entry->format);
}

uint32_t ft_tt_fttw_tile_bytes(uint32_t format)
{
    switch (format) {
    case FT_TT_FORMAT_FP32:
        return FT_TT_OP_TILE_BYTES_FP32;
    case FT_TT_FORMAT_BF16:
        return FT_TT_OP_TILE_BYTES_BF16;
    case FT_TT_FORMAT_BFP8_B:
        return FT_TT_OP_TILE_BYTES_BFP8_B;
    case FT_TT_FORMAT_BFP4_B:
        return FT_TT_OP_TILE_BYTES_BFP4_B;
    case FT_TT_FORMAT_BFP6_B:
        return FTTW_TILE_BYTES_BFP6_B;
    default:
        return 0u;
    }
}

int ft_tt_fttw_upload_range(const ft_tt_fttw *artifact, const ft_tt_artifact_entry *entry,
                            uint64_t source_offset, uint64_t source_bytes,
                            const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                            uint32_t address, uint32_t chunk_bytes)
{
    const unsigned char *source;
    uint64_t remaining;
    uint64_t offset = 0;

    if (!artifact || !entry || !noc)
        return FT_TT_EINVAL;
    if (entry->data_offset + entry->byte_count > (uint64_t)artifact->bytes) {
        ft_tt_log_error("fttw: %s payload runs past the artifact", entry->name);
        return FT_TT_EINVAL;
    }
    /* Bounded against the TENSOR, not merely against the mapping. A run that stayed inside
     * the file but left the tensor would upload a neighbour's bytes, and the card would then
     * compute a plausible wrong answer with no error anywhere. */
    if (source_bytes == 0u || source_offset + source_bytes > entry->byte_count) {
        ft_tt_log_error("fttw: %s range +%llu..%llu is outside its %llu payload bytes",
                        entry->name, (unsigned long long)source_offset,
                        (unsigned long long)(source_offset + source_bytes),
                        (unsigned long long)entry->byte_count);
        return FT_TT_EINVAL;
    }
    if (chunk_bytes == 0u)
        chunk_bytes = FTTW_DEFAULT_CHUNK_BYTES;

    source = artifact->base + entry->data_offset + source_offset;
    remaining = source_bytes;
    while (remaining > 0u) {
        uint32_t bytes = (remaining < chunk_bytes) ? (uint32_t)remaining : chunk_bytes;
        int rc = noc->write(noc->user, x, y, (uint64_t)address + offset, source + offset, bytes);

        if (rc != FT_TT_OK) {
            ft_tt_log_error("fttw: uploading %s at +%llu failed: %s", entry->name,
                            (unsigned long long)(source_offset + offset),
                            ft_tt_error_string(rc));
            return rc;
        }
        offset += bytes;
        remaining -= bytes;
    }
    return FT_TT_OK;
}

int ft_tt_fttw_upload(const ft_tt_fttw *artifact, const ft_tt_artifact_entry *entry,
                      const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t address,
                      uint32_t chunk_bytes)
{
    const unsigned char *source;
    uint64_t remaining;
    uint64_t offset = 0;

    if (!artifact || !entry || !noc)
        return FT_TT_EINVAL;
    if (entry->data_offset + entry->byte_count > (uint64_t)artifact->bytes) {
        ft_tt_log_error("fttw: %s payload runs past the artifact", entry->name);
        return FT_TT_EINVAL;
    }
    if (chunk_bytes == 0u)
        chunk_bytes = FTTW_DEFAULT_CHUNK_BYTES;

    source = artifact->base + entry->data_offset;
    remaining = entry->byte_count;
    while (remaining > 0u) {
        uint32_t bytes = (remaining < chunk_bytes) ? (uint32_t)remaining : chunk_bytes;
        int rc = noc->write(noc->user, x, y, (uint64_t)address + offset, source + offset, bytes);

        if (rc != FT_TT_OK) {
            ft_tt_log_error("fttw: uploading %s at +%llu failed: %s", entry->name,
                            (unsigned long long)offset, ft_tt_error_string(rc));
            return rc;
        }
        offset += bytes;
        remaining -= bytes;
    }
    return FT_TT_OK;
}

int ft_tt_fttw_upload_broadcast_rows(const ft_tt_fttw *artifact,
                                     const ft_tt_artifact_entry *entry,
                                     const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                     uint32_t address, void *scratch, size_t scratch_bytes)
{
    uint16_t replicated[FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM];
    const float *values;
    uint32_t elements;
    uint32_t tile_count;
    uint32_t tile;

    if (!artifact || !entry || !noc || !scratch)
        return FT_TT_EINVAL;
    if (entry->n_dims != 1u || entry->format != FT_TT_FORMAT_FP32) {
        ft_tt_log_error("fttw: %s is not a 1-D fp32 tensor (dims %u format %u)", entry->name,
                        entry->n_dims, entry->format);
        return FT_TT_EINVAL;
    }
    if (scratch_bytes < FT_TT_OP_TILE_BYTES_BF16)
        return FT_TT_EINVAL;
    if (entry->data_offset + entry->byte_count > (uint64_t)artifact->bytes)
        return FT_TT_EINVAL;

    values = (const float *)(const void *)(artifact->base + entry->data_offset);
    elements = (uint32_t)entry->dims[0];
    tile_count = ft_tt_tile_count_for(elements);

    for (tile = 0; tile < tile_count; tile++) {
        uint32_t base_column = tile * FT_TT_OP_TILE_DIM;
        uint32_t column;
        uint32_t row;
        int rc;

        /* One tile of the broadcast: 32 identical rows of 32 bf16 values, zero-padded
         * where the vector ends. Built row-major, then face-ordered by the same tilize
         * the converter uses so there is one layout in the system, not two. */
        for (column = 0; column < FT_TT_OP_TILE_DIM; column++) {
            uint32_t source_index = base_column + column;
            uint16_t value = 0u;

            if (source_index < elements)
                value = ft_bf16_from_float(values[source_index]);
            for (row = 0; row < FT_TT_OP_TILE_DIM; row++)
                replicated[row * FT_TT_OP_TILE_DIM + column] = value;
        }

        if (ft_tt_tilize_strip(replicated, sizeof(uint16_t), FT_TT_OP_TILE_DIM,
                               FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM * sizeof(uint16_t),
                               scratch) != 0)
            return FT_TT_EINVAL;

        rc = noc->write(noc->user, x, y, (uint64_t)address + (uint64_t)tile * FT_TT_OP_TILE_BYTES_BF16,
                        scratch, FT_TT_OP_TILE_BYTES_BF16);
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}
