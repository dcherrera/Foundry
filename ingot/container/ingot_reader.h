/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_reader.h
 * @brief  Random-access .ingot reader (mmap): header, directory, per-tensor decode (hash-checked),
 *         and a byte-identical rebuild for version 1 files. Version 2 and 3 files rebuild through
 *         the streaming decoder (ingot_stream.h). A predict entry decodes its partners first.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_READER_H
#define FOUNDRY_INGOT_READER_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/container/ingot_format.h"
#include "ingot/container/ingot_header.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ingot_reader ingot_reader;
typedef struct ingot_base_set ingot_base_set;

/**
 * mmap and validate a .ingot file (header, directory, payload bounds).
 *
 * @param path  File path
 * @param out   Out: reader, closed with ingot_reader_close
 * @return      0, -INGOT_EIO, -INGOT_EFORMAT, -INGOT_EUNSUPPORTED (unknown format version) or
 *              -INGOT_ENOMEM
 *
 * Thread-safe: yes (read-only after open)   Reentrant: yes   Signal-safe: no
 */
int ingot_reader_open(const char *path, ingot_reader **out);

/**
 * Is `path` a .ingot file? Reads the magic and version only; the file name is not looked at.
 *
 * @param path     File path
 * @param version  Out (optional): the file's format version, when it is a .ingot file
 * @return         1 = .ingot file (any version), 0 = not one, -INGOT_EIO
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_reader_probe(const char *path, uint32_t *version);

/**
 * Close a reader (NULL is a no-op).
 *
 * Thread-safe: no (no other call on the handle may run)   Reentrant: yes   Signal-safe: no
 */
void ingot_reader_close(ingot_reader *r);

/**
 * The parsed file header.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const ingot_file_header *ingot_reader_header(const ingot_reader *r);

/**
 * Directory size and entries (in payload order: source order before version 3; entries with name
 * "" are filler).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_reader_count(const ingot_reader *r);
const ingot_entry *ingot_reader_entry(const ingot_reader *r, size_t i);

/**
 * Find a named entry. @return 0 or -INGOT_ENOTFOUND
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_reader_find(const ingot_reader *r, const char *name, size_t *index);

/**
 * Size of this .ingot file and of the source file it rebuilds.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_reader_file_size(const ingot_reader *r);
uint64_t ingot_reader_source_size(const ingot_reader *r);

/**
 * The stored source header bytes, verbatim.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const uint8_t *ingot_reader_source_header(const ingot_reader *r, uint64_t *size);

/**
 * Decode entry i into out (entry->source_size bytes) and check its hash. Delta entries load the
 * whole base tensor and check its hash first.
 *
 * @param r        Reader
 * @param i        Entry index
 * @param base     Base set for delta entries (NULL if the file has none)
 * @param threads  Worker threads
 * @param out      Output bytes, caller-owned, entry->source_size long, 2-aligned
 * @return         0, -INGOT_EBASE (base missing or wrong), -INGOT_EFORMAT, -INGOT_ECORRUPT
 *
 * Thread-safe: no (the base set is not)   Reentrant: yes   Signal-safe: no
 */
int ingot_reader_decode(const ingot_reader *r, size_t i, ingot_base_set *base, int threads,
                        uint8_t *out);

/**
 * The ref set match entries read their external refs from (version 4; NULL: none). A match
 * entry decodes its sources first (each once per ingot_reader_decode call).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: yes
 */
void ingot_reader_set_refs(ingot_reader *r, ingot_base_set *refs);

/**
 * Write the source file back, byte-identical, to dst_path. Peak memory is the largest tensor
 * (use ingot_stream for bounded memory).
 *
 * @return 0 or the first error (a partial dst file is removed)
 *
 * Thread-safe: as ingot_reader_decode   Reentrant: yes   Signal-safe: no
 */
int ingot_reader_rebuild(const ingot_reader *r, const char *dst_path, ingot_base_set *base,
                         int threads);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_READER_H */
