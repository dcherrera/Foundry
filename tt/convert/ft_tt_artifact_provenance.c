/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_artifact_provenance.c
 * @brief Implementation of the artifact provenance trailer.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "ft_tt_artifact_provenance.h"

#include <stdio.h>
#include <string.h>

#include "ft_tt_weight_convert.h"

/* The trailer is written and read as raw bytes, so a compiler that padded it would produce
 * a file this build cannot read back. Fail the BUILD rather than the cache. */
typedef char ft_tt_provenance_layout_assert[(sizeof(ft_tt_artifact_provenance) == 64u) ? 1 : -1];

#define PROVENANCE_FNV64_OFFSET 1469598103934665603ull
#define PROVENANCE_FNV64_PRIME 1099511628211ull
#define PROVENANCE_FNV32_OFFSET 2166136261u
#define PROVENANCE_FNV32_PRIME 16777619u

static uint64_t provenance_fnv64(uint64_t hash, const void *bytes, size_t count)
{
    const unsigned char *cursor = (const unsigned char *)bytes;
    size_t i;

    for (i = 0; i < count; i++) {
        hash ^= (uint64_t)cursor[i];
        hash *= PROVENANCE_FNV64_PRIME;
    }
    return hash;
}

static uint64_t provenance_fnv64_u64(uint64_t hash, uint64_t value)
{
    unsigned char bytes[8];
    int i;

    for (i = 0; i < 8; i++)
        bytes[i] = (unsigned char)((value >> (8 * i)) & 0xffu);
    return provenance_fnv64(hash, bytes, sizeof(bytes));
}

uint32_t ft_tt_artifact_behaviour_stamp(void)
{
    const char *stamp = FT_TT_CONVERT_BEHAVIOUR_STAMP;
    uint32_t hash = PROVENANCE_FNV32_OFFSET;
    size_t i;

    for (i = 0; stamp[i] != '\0'; i++) {
        hash ^= (uint32_t)(unsigned char)stamp[i];
        hash *= PROVENANCE_FNV32_PRIME;
    }
    return hash;
}

uint64_t ft_tt_artifact_source_digest(const ft_gguf_map *map)
{
    uint64_t hash = PROVENANCE_FNV64_OFFSET;
    uint32_t count;
    uint32_t index;

    if (!map)
        return 0u;

    count = ft_gguf_map_tensor_count(map);
    hash = provenance_fnv64_u64(hash, (uint64_t)count);
    hash = provenance_fnv64_u64(hash, (uint64_t)ft_gguf_map_part_count(map));

    for (index = 0; index < count; index++) {
        const ft_gguf_tensor *tensor = ft_gguf_map_tensor(map, index);
        uint32_t d;

        if (!tensor)
            break;
        /* The NUL is included so "a" and "a\0b" cannot collide by concatenation. */
        hash = provenance_fnv64(hash, tensor->name, strlen(tensor->name) + 1u);
        hash = provenance_fnv64_u64(hash, (uint64_t)tensor->n_dims);
        for (d = 0; d < 4u; d++)
            hash = provenance_fnv64_u64(hash, tensor->dims[d]);
        hash = provenance_fnv64_u64(hash, (uint64_t)tensor->ggml_type);
        /* `byte_count` is NOT hashed: ft_gguf_map derives it from the dimensions and the
         * type, both of which are already in, so a mutation that dropped it could not be
         * seen by any test — and a check nothing can fail is not a check. */
        /* No ft_gguf_map_release() here: the directory is metadata the map already holds,
         * and releasing would only madvise pages this walk never touched. */
    }
    return hash;
}

/* ------------------------------------------------------------------------- */
/* Little-endian field IO — the file format is LE regardless of the host      */
/* ------------------------------------------------------------------------- */

static void provenance_put_u32(unsigned char *out, uint32_t value)
{
    out[0] = (unsigned char)(value & 0xffu);
    out[1] = (unsigned char)((value >> 8) & 0xffu);
    out[2] = (unsigned char)((value >> 16) & 0xffu);
    out[3] = (unsigned char)((value >> 24) & 0xffu);
}

static void provenance_put_u64(unsigned char *out, uint64_t value)
{
    provenance_put_u32(out, (uint32_t)(value & 0xffffffffu));
    provenance_put_u32(out + 4, (uint32_t)(value >> 32));
}

static uint32_t provenance_get_u32(const unsigned char *in)
{
    return (uint32_t)in[0] | ((uint32_t)in[1] << 8) | ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static uint64_t provenance_get_u64(const unsigned char *in)
{
    return (uint64_t)provenance_get_u32(in) | ((uint64_t)provenance_get_u32(in + 4) << 32);
}

static void provenance_pack(const ft_tt_artifact_provenance *record, unsigned char *out)
{
    provenance_put_u64(out + 0, record->source_bytes);
    provenance_put_u64(out + 8, record->source_digest);
    provenance_put_u64(out + 16, record->payload_bytes);
    provenance_put_u32(out + 24, record->provenance_version);
    provenance_put_u32(out + 28, record->artifact_version);
    provenance_put_u32(out + 32, record->ledger_mode);
    provenance_put_u32(out + 36, record->rope_pairing);
    provenance_put_u32(out + 40, record->artifact_flags);
    provenance_put_u32(out + 44, record->tensor_count);
    provenance_put_u32(out + 48, record->behaviour_stamp);
    provenance_put_u32(out + 52, record->trailer_bytes);
    provenance_put_u32(out + 56, record->reserved);
    provenance_put_u32(out + 60, record->magic);
}

static void provenance_unpack(const unsigned char *in, ft_tt_artifact_provenance *record)
{
    record->source_bytes = provenance_get_u64(in + 0);
    record->source_digest = provenance_get_u64(in + 8);
    record->payload_bytes = provenance_get_u64(in + 16);
    record->provenance_version = provenance_get_u32(in + 24);
    record->artifact_version = provenance_get_u32(in + 28);
    record->ledger_mode = provenance_get_u32(in + 32);
    record->rope_pairing = provenance_get_u32(in + 36);
    record->artifact_flags = provenance_get_u32(in + 40);
    record->tensor_count = provenance_get_u32(in + 44);
    record->behaviour_stamp = provenance_get_u32(in + 48);
    record->trailer_bytes = provenance_get_u32(in + 52);
    record->reserved = provenance_get_u32(in + 56);
    record->magic = provenance_get_u32(in + 60);
}

/** On-disk width of the trailer. Not sizeof(): the struct is padded to 8, the record is not. */
#define PROVENANCE_ON_DISK_BYTES 64u

/** Read the artifact's own 40-byte header so the trailer describes the file, not a wish. */
static int provenance_read_artifact_header(FILE *file, ft_tt_artifact_header *out)
{
    unsigned char raw[FT_TT_ARTIFACT_HEADER_BYTES];

    if (fseek(file, 0, SEEK_SET) != 0)
        return FT_TT_PROVENANCE_EIO;
    if (fread(raw, 1, sizeof(raw), file) != sizeof(raw))
        return FT_TT_PROVENANCE_EIO;

    out->magic = provenance_get_u32(raw + 0);
    out->version = provenance_get_u32(raw + 4);
    out->tensor_count = provenance_get_u32(raw + 8);
    out->flags = provenance_get_u32(raw + 12);
    out->directory_offset = provenance_get_u64(raw + 16);
    out->data_offset = provenance_get_u64(raw + 24);
    out->total_bytes = provenance_get_u64(raw + 32);

    if (out->magic != FT_TT_ARTIFACT_MAGIC)
        return FT_TT_PROVENANCE_EINVAL;
    return FT_TT_PROVENANCE_OK;
}

int ft_tt_artifact_provenance_append(const char *artifact_path, uint64_t source_bytes,
                                     uint64_t source_digest, uint32_t ledger_mode,
                                     uint32_t rope_pairing)
{
    ft_tt_artifact_provenance record;
    ft_tt_artifact_header header;
    unsigned char raw[PROVENANCE_ON_DISK_BYTES];
    FILE *file;
    int rc;

    if (!artifact_path)
        return FT_TT_PROVENANCE_EINVAL;

    file = fopen(artifact_path, "r+b");
    if (!file)
        return FT_TT_PROVENANCE_EIO;

    rc = provenance_read_artifact_header(file, &header);
    if (rc != FT_TT_PROVENANCE_OK) {
        fclose(file);
        return rc;
    }

    memset(&record, 0, sizeof(record));
    record.source_bytes = source_bytes;
    record.source_digest = source_digest;
    record.payload_bytes = header.total_bytes;
    record.provenance_version = FT_TT_PROVENANCE_VERSION;
    record.artifact_version = header.version;
    record.ledger_mode = ledger_mode;
    record.rope_pairing = rope_pairing;
    record.artifact_flags = header.flags;
    record.tensor_count = header.tensor_count;
    record.behaviour_stamp = ft_tt_artifact_behaviour_stamp();
    record.trailer_bytes = PROVENANCE_ON_DISK_BYTES;
    record.magic = FT_TT_PROVENANCE_MAGIC;

    provenance_pack(&record, raw);

    if (fseek(file, 0, SEEK_END) != 0 || fwrite(raw, 1, sizeof(raw), file) != sizeof(raw)) {
        fclose(file);
        return FT_TT_PROVENANCE_EIO;
    }
    if (fclose(file) != 0)
        return FT_TT_PROVENANCE_EIO;
    return FT_TT_PROVENANCE_OK;
}

int ft_tt_artifact_provenance_read(const char *artifact_path, ft_tt_artifact_provenance *out)
{
    unsigned char raw[PROVENANCE_ON_DISK_BYTES];
    FILE *file;
    long size;

    if (!artifact_path || !out)
        return FT_TT_PROVENANCE_EINVAL;

    memset(out, 0, sizeof(*out));
    file = fopen(artifact_path, "rb");
    if (!file)
        return FT_TT_PROVENANCE_EIO;

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return FT_TT_PROVENANCE_EIO;
    }
    size = ftell(file);
    if (size < (long)(FT_TT_ARTIFACT_HEADER_BYTES + PROVENANCE_ON_DISK_BYTES)) {
        fclose(file);
        return FT_TT_PROVENANCE_EABSENT;
    }
    if (fseek(file, size - (long)PROVENANCE_ON_DISK_BYTES, SEEK_SET) != 0 ||
        fread(raw, 1, sizeof(raw), file) != sizeof(raw)) {
        fclose(file);
        return FT_TT_PROVENANCE_EIO;
    }
    fclose(file);

    provenance_unpack(raw, out);
    if (out->magic != FT_TT_PROVENANCE_MAGIC)
        return FT_TT_PROVENANCE_EABSENT;
    /* Magic present but the record disagrees with itself: a hand-edited or half-written
     * trailer. That is STALE, not absent — absent would silently reconvert and hide it. */
    if (out->trailer_bytes != PROVENANCE_ON_DISK_BYTES ||
        out->provenance_version != FT_TT_PROVENANCE_VERSION)
        return FT_TT_PROVENANCE_ESTALE;
    return FT_TT_PROVENANCE_OK;
}

/** One field comparison, so every mismatch reports the same way and none is silent. */
#define PROVENANCE_COMPARE_U64(field, label)                                                   \
    do {                                                                                       \
        if (want->field != got->field) {                                                       \
            snprintf(why, why_bytes, "%s: artifact says %llu, this checkpoint needs %llu",     \
                     (label), (unsigned long long)got->field, (unsigned long long)want->field); \
            return FT_TT_PROVENANCE_ESTALE;                                                    \
        }                                                                                      \
    } while (0)

int ft_tt_artifact_provenance_compare(const ft_tt_artifact_provenance *want,
                                      const ft_tt_artifact_provenance *got, char *why,
                                      size_t why_bytes)
{
    if (!want || !got || !why || why_bytes == 0u)
        return FT_TT_PROVENANCE_EINVAL;

    why[0] = '\0';
    PROVENANCE_COMPARE_U64(source_digest, "source tensor directory digest");
    PROVENANCE_COMPARE_U64(source_bytes, "source checkpoint bytes");
    PROVENANCE_COMPARE_U64(tensor_count, "tensor count");
    PROVENANCE_COMPARE_U64(ledger_mode, "precision rung");
    PROVENANCE_COMPARE_U64(rope_pairing, "rotary pairing");
    PROVENANCE_COMPARE_U64(artifact_flags, "artifact header flags");
    PROVENANCE_COMPARE_U64(artifact_version, "FTTW artifact version");
    PROVENANCE_COMPARE_U64(behaviour_stamp, "converter behaviour stamp");
    /* `payload_bytes` is deliberately NOT compared here. The caller cannot know it without
     * converting, and a `want` derived from the planner's arithmetic would turn any future
     * planner/converter drift into a cache that reconverts on EVERY launch and never says
     * why. It is checked against the artifact's own header and length instead — see
     * ft_tt_artifact_provenance_verify_file(). */
    return FT_TT_PROVENANCE_OK;
}

int ft_tt_artifact_provenance_verify_file(const char *artifact_path,
                                          const ft_tt_artifact_provenance *record, char *why,
                                          size_t why_bytes)
{
    ft_tt_artifact_header header;
    FILE *file;
    long size;
    int rc;

    if (!artifact_path || !record || !why || why_bytes == 0u)
        return FT_TT_PROVENANCE_EINVAL;

    why[0] = '\0';
    file = fopen(artifact_path, "rb");
    if (!file) {
        snprintf(why, why_bytes, "cannot open the artifact to verify it");
        return FT_TT_PROVENANCE_EIO;
    }
    rc = provenance_read_artifact_header(file, &header);
    if (rc != FT_TT_PROVENANCE_OK) {
        fclose(file);
        snprintf(why, why_bytes, "artifact header is not an FTTW header");
        return FT_TT_PROVENANCE_ESTALE;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        snprintf(why, why_bytes, "cannot measure the artifact");
        return FT_TT_PROVENANCE_EIO;
    }
    size = ftell(file);
    fclose(file);

    /* A truncated or appended-to artifact. The trailer says how long the FTTW part is and
     * the file says how long the file is; a cache that trusted either alone would serve a
     * file whose directory points past its own end. */
    if ((uint64_t)size != record->payload_bytes + PROVENANCE_ON_DISK_BYTES) {
        snprintf(why, why_bytes,
                 "artifact is %lld B but its trailer describes %llu B of FTTW plus a %u B "
                 "trailer", (long long)size, (unsigned long long)record->payload_bytes,
                 PROVENANCE_ON_DISK_BYTES);
        return FT_TT_PROVENANCE_ESTALE;
    }
    if (header.total_bytes != record->payload_bytes || header.version != record->artifact_version ||
        header.flags != record->artifact_flags || header.tensor_count != record->tensor_count) {
        snprintf(why, why_bytes,
                 "artifact header (version %u, flags 0x%08x, %u tensors, %llu B) disagrees "
                 "with its own trailer (version %u, flags 0x%08x, %u tensors, %llu B)",
                 header.version, header.flags, header.tensor_count,
                 (unsigned long long)header.total_bytes, record->artifact_version,
                 record->artifact_flags, record->tensor_count,
                 (unsigned long long)record->payload_bytes);
        return FT_TT_PROVENANCE_ESTALE;
    }
    return FT_TT_PROVENANCE_OK;
}
