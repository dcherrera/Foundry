/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_header.h
 * @brief  Serialize and parse the .ingot file header and directory entries (versions 1 to 3),
 *         and check an entry's structure. Shared by the writer, the random-access reader and the
 *         streaming decoder.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_HEADER_H
#define FOUNDRY_INGOT_HEADER_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/container/ingot_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Parsed file header. Version 1 files have no source kind, digests or base fields (zero). */
typedef struct {
    uint32_t version;
    uint32_t entry_count;
    ingot_source_kind kind;
    uint64_t source_header_offset;
    uint64_t source_header_size;
    uint64_t dir_offset;
    uint64_t dir_size;
    uint64_t source_size;
    uint64_t file_size;
    uint8_t source_sha256[INGOT_SHA256_BYTES];
    uint8_t base_sha256[INGOT_SHA256_BYTES]; /* all zero: no base file */
    uint64_t base_size;
} ingot_file_header;

/**
 * Header size of a format version (0 if unknown).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_header_size(uint32_t version);

/**
 * Parse a header from its first bytes.
 *
 * @param buf    At least 8 bytes; the full header size for a known version
 * @param avail  Bytes at buf
 * @param h      Out
 * @return       0, -INGOT_EFORMAT (not an .ingot file, or bad fields), -INGOT_EUNSUPPORTED
 *               (another version; h->version is set), -INGOT_ETRUNCATED (avail too small)
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_header_parse(const uint8_t *buf, size_t avail, ingot_file_header *h);

/**
 * Serialize a version 2 or 3 header (h->version; INGOT_HEADER_BYTES).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_header_write(const ingot_file_header *h, uint8_t *out);

/**
 * 1 if the header names a base file (a non-zero base digest).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_header_has_base(const ingot_file_header *h);

/**
 * Serialized size of one directory entry.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_entry_record_size(const ingot_entry *e);

/**
 * Serialize one entry. @return bytes written (ingot_entry_record_size)
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_entry_write(const ingot_entry *e, uint8_t *out);

/**
 * Parse a whole directory.
 *
 * @param dir      Directory bytes
 * @param size     Directory size (entries must fill it exactly)
 * @param count    Entry count from the header
 * @param entries  Out: array, freed with ingot_entries_free
 * @return         0, -INGOT_EFORMAT or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_directory_parse(const uint8_t *dir, uint64_t size, size_t count,
                          ingot_entry **entries);

/**
 * Free a parsed directory (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_entries_free(ingot_entry *entries, size_t count);

/**
 * Structural check of an entry against the header: ranges in bounds, mode valid for the dtype
 * (and for the file version), payload 8-aligned.
 *
 * @return 0, -INGOT_EFORMAT, or -INGOT_EUNSUPPORTED (the prediction mode in a version 2 file,
 *         where it was reserved)
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_entry_check(const ingot_entry *e, const ingot_file_header *h);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_HEADER_H */
