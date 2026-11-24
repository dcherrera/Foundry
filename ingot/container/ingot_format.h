/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_format.h
 * @brief  .ingot container constants: file header, directory entry, tensor modes, dtype codes.
 *         Full layout in ingot/FORMAT.md.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_FORMAT_H
#define FOUNDRY_INGOT_FORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* File magic (4 bytes), then a u32 format version; files are recognised by these, not by name.
 * Name convention: the original name plus ".ingot" (model.safetensors.ingot, model.gguf.ingot). */
#define INGOT_MAGIC "INGT"
#define INGOT_MAGIC_BYTES 4
#define INGOT_SUFFIX ".ingot"

enum {
    INGOT_VERSION = 5,         /* newest version written (a file is written at the oldest version
                                  that holds its features: 2, 3, 4 or 5) */
    INGOT_VERSION_MIN = 1,     /* oldest version read (version 1: random access only) */
    INGOT_HEADER_BYTES = 144,  /* versions 2 to 5 */
    INGOT_HEADER_BYTES_V1 = 64,
    INGOT_SHA256_BYTES = 32,
    INGOT_MAX_DIMS = 8,
    INGOT_ENTRY_FIXED_BYTES = 2 + 4 + 6 * 8, /* plus name and 8 bytes per dim */
    INGOT_PAYLOAD_FRAME_BYTES = 16, /* version 2 coded payloads: u64 prefix size, u64 blocks */
};

/** What kind of file the .ingot rebuilds. */
typedef enum {
    INGOT_SOURCE_UNKNOWN = 0, /* version 1 files (always safetensors) */
    INGOT_SOURCE_SAFETENSORS = 1,
    INGOT_SOURCE_GGUF = 2,
} ingot_source_kind;

/** How a tensor's payload is stored. */
typedef enum {
    INGOT_MODE_RAW = 0,         /* source bytes verbatim */
    INGOT_MODE_PLAIN = 1,       /* BF16, coded alone (ingot_plain.h) */
    INGOT_MODE_DELTA = 2,       /* BF16, coded against a base tensor (ingot_delta.h) */
    INGOT_MODE_QUANT = 3,       /* Q8_0 or K-quant blocks, fields coded (ingot_qtensor.h) */
    INGOT_MODE_QUANT_DELTA = 4, /* as QUANT, blocks identical to the base's are flagged */
    INGOT_MODE_BLOCK_DELTA = 5, /* any block type: identical blocks flagged, others raw */
    INGOT_MODE_PREDICT = 6,     /* BF16, predicted from earlier tensors of the same units
                                   (ingot_predict.h; version 3 and later) */
    INGOT_MODE_MATCH = 7,       /* BF16, per-unit partners from any earlier tensor or an earlier
                                   file (ingot_match.h; version 4; version 5 adds row-prefix
                                   refs, which may be read from a later file's .ingot) */
    INGOT_MODE_COUNT = 8,       /* modes this version decodes */
} ingot_mode;

/** Source dtype (informational except where a mode depends on it). */
typedef enum {
    INGOT_DTYPE_OTHER = 0,
    INGOT_DTYPE_BF16 = 1,
    INGOT_DTYPE_F16 = 2,
    INGOT_DTYPE_F32 = 3,
    INGOT_DTYPE_F64 = 4,
    INGOT_DTYPE_I8 = 5,
    INGOT_DTYPE_U8 = 6,
    INGOT_DTYPE_I16 = 7,
    INGOT_DTYPE_U16 = 8,
    INGOT_DTYPE_I32 = 9,
    INGOT_DTYPE_U32 = 10,
    INGOT_DTYPE_I64 = 11,
    INGOT_DTYPE_U64 = 12,
    INGOT_DTYPE_BOOL = 13,
    INGOT_DTYPE_GGML = 64, /* + ggml type (0..191) for GGUF types without a plain equivalent */
} ingot_dtype;

/** One tensor of a source file, as stored in the directory. */
typedef struct {
    char *name;                     /* "" for filler bytes between tensors */
    ingot_dtype dtype;
    ingot_mode mode;
    int ndim;
    uint64_t shape[INGOT_MAX_DIMS]; /* row-major: the last dimension is contiguous */
    uint64_t source_offset;         /* absolute offset in the source file */
    uint64_t source_size;
    uint64_t payload_offset;        /* absolute offset in the .ingot file (8-aligned) */
    uint64_t payload_size;
    uint64_t hash;                  /* ingot_hash64 of the source bytes */
    uint64_t base_hash;             /* delta modes: ingot_hash64 of the base tensor; predict mode:
                                       the partner entries (ingot_partner_field); match mode:
                                       ingot_hash64 of its source list; else 0 */
} ingot_entry;

/* Predict entries name their (at most two) partner entries in base_hash: directory index + 1 in
 * the low 32 bits, then the second partner's index + 1 (or 0) in the high 32 bits. */
static inline uint64_t ingot_partner_field(const uint32_t *index, uint32_t count) {
    uint64_t field = 0;
    for (uint32_t j = 0; j < count && j < 2; j++)
        field |= ((uint64_t)index[j] + 1) << (32 * j);
    return field;
}

/* Partner count of a base_hash field (index[] filled), 0 if malformed. */
static inline uint32_t ingot_partner_list(uint64_t field, uint32_t index[2]) {
    uint32_t lo = (uint32_t)field, hi = (uint32_t)(field >> 32);
    if (!lo)
        return 0;
    index[0] = lo - 1;
    if (!hi)
        return 1;
    index[1] = hi - 1;
    return 2;
}

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_FORMAT_H */
