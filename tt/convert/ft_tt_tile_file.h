/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_tile_file.h
 * @brief The MXTL tile file: precomputed FTTW tensor payloads (e.g. GPTQ-rounded Bfp4_b
 *   tiles) by GGUF tensor name, for ft_tt_artifact_splice() to put into an artifact.
 *
 * @status None
 * @issues None
 * @todo None
 */

/*
 * ---------------------------------------------------------------------------
 * THE FORMAT, v1 — A FIXED CONTRACT SHARED WITH ANOTHER REPO'S WRITER
 * ---------------------------------------------------------------------------
 * All little-endian.
 *
 *   header   "MXTL" | u32 version = 1 | u32 n_entries | u32 reserved = 0
 *   entry    u32 name_len | name (no NUL) | u32 format | u32 layout | u64 n_bytes |
 *            u8 sha256[32] of the payload | payload (n_bytes)
 *
 * `format` is an ft_tt_tensor_format, `layout` an ft_tt_tensor_layout, and the payload is
 * exactly the bytes the converter would store as that entry's data for that tensor shape,
 * format and layout — unpadded, as the directory's `byte_count` counts them.
 *
 * A reader refuses: a bad magic, another version, reserved != 0, a short file, bytes after
 * the last payload, an empty name, a name with a NUL or of FT_TT_TILE_FILE_NAME_BYTES or
 * more, a payload whose sha256 disagrees, and a name listed twice. It does NOT judge whether
 * a format or layout suits a tensor; that needs the artifact, and is the splice's check.
 *
 * Payloads are never held: the reader hashes them through a fixed buffer and records where
 * each one starts, and ft_tt_tile_file_read_payload() reads a range on demand.
 */

#ifndef FT_TT_TILE_FILE_H
#define FT_TT_TILE_FILE_H

#include <stddef.h>
#include <stdint.h>

#include "ft_sha256.h"

#ifdef __cplusplus
extern "C" {
#endif

/** "MXTL" read as a little-endian u32. */
#define FT_TT_TILE_FILE_MAGIC 0x4c54584du

/** The one version this build reads and writes. */
#define FT_TT_TILE_FILE_VERSION 1u

/** Header bytes: magic, version, n_entries, reserved. */
#define FT_TT_TILE_FILE_HEADER_BYTES 16u

/** Longest name + 1. Matches the FTTW directory's name field, which a name must fit. */
#define FT_TT_TILE_FILE_NAME_BYTES 128u

/** Room for a refusal, which names the file, the entry and what was wrong. */
#define FT_TT_TILE_FILE_ERROR_BYTES 512u

/** Status codes. */
enum {
    FT_TT_TILE_FILE_OK = 0,
    FT_TT_TILE_FILE_ERROR = -1,
    FT_TT_TILE_FILE_ENOMEM = -2,
    FT_TT_TILE_FILE_EINVAL = -3,
    FT_TT_TILE_FILE_EIO = -4,
    FT_TT_TILE_FILE_EFORMAT = -5,   /**< not an MXTL v1 file: magic, version, reserved, a name */
    FT_TT_TILE_FILE_ETRUNCATED = -6, /**< the file ends inside a field or a payload, or runs past one */
    FT_TT_TILE_FILE_ESHA = -7,      /**< a payload's bytes do not hash to its recorded sha256 */
    FT_TT_TILE_FILE_EDUPLICATE = -8 /**< two entries name one tensor */
};

/** One payload to write. `payload` is read once; it may point into an mmap. */
typedef struct {
    const char *name;
    uint32_t format;
    uint32_t layout;
    const void *payload;
    uint64_t n_bytes;
} ft_tt_tile_file_payload;

/** One entry as read: its fields and where its payload starts in the file. */
typedef struct {
    char name[FT_TT_TILE_FILE_NAME_BYTES];
    uint32_t format;
    uint32_t layout;
    uint64_t n_bytes;
    unsigned char sha256[FT_SHA256_BYTES];
    uint64_t payload_offset;
} ft_tt_tile_file_entry;

/** An open, fully verified tile file. Opaque. */
typedef struct ft_tt_tile_file ft_tt_tile_file;

/**
 * Write an MXTL v1 file, truncating `path`. Each payload's sha256 is computed here.
 *
 * Refuses (EINVAL, nothing written) what a reader would refuse: an empty or over-long name,
 * and a name given twice.
 *
 * @param error        receives the refusal or the failing path; may be NULL
 * @return FT_TT_TILE_FILE_OK, EINVAL, EDUPLICATE or EIO
 *
 * Thread-safe: yes (no shared state)    Reentrant: yes    Signal-safe: no
 */
int ft_tt_tile_file_write(const char *path, const ft_tt_tile_file_payload *entries, uint32_t count, char *error,
                          size_t error_bytes);

/**
 * Open a tile file and verify all of it: structure, every payload's sha256, unique names.
 * Reads every payload byte once, through a fixed buffer.
 *
 * @param out          receives the handle on success, NULL otherwise; free with
 *                     ft_tt_tile_file_close()
 * @param error        receives "<path>: <what>" on failure; may be NULL
 * @return FT_TT_TILE_FILE_OK, or EIO/EFORMAT/ETRUNCATED/ESHA/EDUPLICATE/ENOMEM/EINVAL
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_tile_file_open(const char *path, ft_tt_tile_file **out, char *error, size_t error_bytes);

/** Close and free. Safe on NULL. */
void ft_tt_tile_file_close(ft_tt_tile_file *file);

/** The path it was opened from. */
const char *ft_tt_tile_file_path(const ft_tt_tile_file *file);

/** Entries in file order. */
uint32_t ft_tt_tile_file_count(const ft_tt_tile_file *file);

/** Entry `index`, or NULL past the end. Owned by `file`. */
const ft_tt_tile_file_entry *ft_tt_tile_file_entry_at(const ft_tt_tile_file *file, uint32_t index);

/** The entry naming `name`, or NULL. */
const ft_tt_tile_file_entry *ft_tt_tile_file_find(const ft_tt_tile_file *file, const char *name);

/**
 * Read `bytes` bytes of entry `index`'s payload, starting `offset` bytes into it.
 *
 * @return FT_TT_TILE_FILE_OK, EINVAL for a range outside the payload, EIO on a short read
 *
 * Thread-safe: yes (pread)    Reentrant: yes    Signal-safe: no
 */
int ft_tt_tile_file_read_payload(const ft_tt_tile_file *file, uint32_t index, uint64_t offset, void *buffer,
                                 size_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* FT_TT_TILE_FILE_H */
