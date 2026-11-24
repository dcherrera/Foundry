/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_dirtail.h
 * @brief  Version 4 directory tail: the external refs (tensors of earlier files a match entry
 *         reads) and every match entry's source list, so a streaming decoder knows what to hold
 *         before it reaches the payloads.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Layout (after the directory entries, inside dir_size; little-endian):
 *   u32 R, R x { u16 name length, name, u8 dtype, u8 ndim, u16 0, u64 shape[ndim], u64 size,
 *   u64 hash }, u32 L, L x { u32 entry, u32 n, u32 source[n] }, then zero bytes up to dir_size
 *   (the writer reserves the space before it knows the lists).
 * A source is a directory index below the entry, entry_count + r for ref r, or
 * INGOT_MATCH_SELF_REF. A match entry's base_hash is ingot_source_list_hash of its list.
 */
#ifndef FOUNDRY_INGOT_DIRTAIL_H
#define FOUNDRY_INGOT_DIRTAIL_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/container/ingot_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/** The parsed tail. refs are entries with name, dtype, ndim, shape, source_size and hash set. */
typedef struct {
    uint32_t ref_count;
    ingot_entry *refs;
    uint32_t *list_start;  /* [entry_count]: offset into sources, or UINT32_MAX (no list) */
    uint32_t *list_count;  /* [entry_count] */
    uint32_t *sources;
    size_t source_total;
} ingot_dir_tail;

/**
 * ingot_hash64 of a source list (u32 little-endian each).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_source_list_hash(const uint32_t *list, uint32_t n);

/**
 * Serialized size of one ref record, and of a list record with n sources.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_dir_tail_ref_size(const ingot_entry *ref);
size_t ingot_dir_tail_list_size(uint32_t n);

/**
 * Write the tail into out (size bytes, zero-filled past what is written).
 * lists: for each of list_count entries, entry index, count and sources (flattened).
 * @return 0 or -INGOT_EINVAL (does not fit)
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_dir_tail_write(const ingot_entry *refs, uint32_t ref_count, const uint32_t *list_entry,
                         const uint32_t *list_n, const uint32_t *const *list_sources,
                         uint32_t list_count, uint8_t *out, size_t size);

/**
 * Parse and check the tail against the directory: every match entry has exactly one list whose
 * hash is its base_hash; every source is earlier, a ref, or self.
 * @return 0, -INGOT_EFORMAT or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_dir_tail_parse(const uint8_t *tail, size_t size, const ingot_entry *entries,
                         size_t entry_count, ingot_dir_tail *out);

/**
 * Free the tail's arrays (the struct is the caller's).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_dir_tail_release(ingot_dir_tail *t);

/**
 * Sources of entry i (count; *list points into t), 0 when it has none.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint32_t ingot_dir_tail_sources(const ingot_dir_tail *t, size_t i, const uint32_t **list);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_DIRTAIL_H */
