/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_writer_match.h
 * @brief  The writer's match-mode state: every layer role's pool (entries of this file, external
 *         refs from earlier files, itself), per-tensor caches (unit-major copies, sketches), the
 *         candidate search, and the version 4 directory tail.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Per target the writer: waits until its pool entries in this file are written (deps), calls
 * prepare (sources + candidates), codes the tensor, then reports which sources the chosen
 * payload names (note_used). Caches of an entry are dropped with its data (release_entry).
 */
#ifndef FOUNDRY_INGOT_WRITER_MATCH_H
#define FOUNDRY_INGOT_WRITER_MATCH_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/codec/ingot_match.h"
#include "ingot/container/ingot_layout.h"
#include "ingot/container/ingot_writer.h"

#ifdef __cplusplus
extern "C" {
#endif

/** One target's match candidate, as ingot_writer_code_tensor takes it. */
typedef struct {
    ingot_predict_axis axis;
    uint64_t units;
    uint64_t unit_len;
    const uint16_t *unit_major;                          /* the target's units [U][H] */
    uint32_t source_count;
    ingot_match_input src[INGOT_MATCH_MAX_SOURCES];      /* ref = final directory reference */
    const ingot_entry *src_entry[INGOT_MATCH_MAX_SOURCES]; /* shape and size (self: the target) */
    uint8_t no_aligned[INGOT_MATCH_MAX_SOURCES];         /* searched only (the dictionary) */
    ingot_match_candidates cand;                         /* owned; freed by done */
} ingot_writer_match;

typedef struct ingot_match_writer ingot_match_writer;

/**
 * Build pools for every layer role of the layout.
 * @param tensor_entry  Layout tensor index -> entry index
 * @return 0, -INGOT_ENOMEM or a ref-set error
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
int ingot_match_writer_create(const ingot_encode_options *opt, const ingot_layout *layout,
                              const size_t *tensor_entry, size_t entry_count,
                              ingot_match_writer **out);

/**
 * Is entry i a match target; its pool entries in this file (*deps).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_match_writer_is_target(const ingot_match_writer *w, size_t i);
uint32_t ingot_match_writer_deps(const ingot_match_writer *w, size_t i, const size_t **deps);

/**
 * Sources and candidates of target i. entry_data[j] / position[j]: held bytes and directory
 * position of entry j (every dep is written and held). entries: the entry list.
 * @return 0, -INGOT_ENOMEM, -INGOT_EBASE (a ref could not be read)
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int ingot_match_writer_prepare(ingot_match_writer *w, size_t i, const uint8_t *data,
                               const ingot_entry *entries, const uint8_t *const *entry_data,
                               const uint32_t *position, ingot_writer_match *out);

/**
 * Free a prepared candidate's arrays.
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_match_writer_done(ingot_writer_match *m);

/**
 * Target i is written: record its payload's source refs (n = 0 when not match-coded) and drop
 * refs no later target needs.
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
int ingot_match_writer_note_used(ingot_match_writer *w, size_t i, const uint32_t *refs,
                                 uint32_t n);

/**
 * Drop entry i's caches (its data is being freed).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_match_writer_release_entry(ingot_match_writer *w, size_t i);

/**
 * Bytes to reserve for the directory tail (0 when no entry is a target), and the tail itself
 * (position[i]: entry i's directory index).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_match_writer_tail_reserve(const ingot_match_writer *w);
int ingot_match_writer_tail_write(const ingot_match_writer *w, const uint32_t *position,
                                  uint8_t *out, size_t size);

/**
 * 1 when a written payload names the dictionary (--dict-ref): the file is then version 5.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_match_writer_uses_dict(const ingot_match_writer *w);

/**
 * Free (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_match_writer_free(ingot_match_writer *w);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_WRITER_MATCH_H */
