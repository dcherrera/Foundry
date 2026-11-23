/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_writer.h
 * @brief  Encode a safetensors or GGUF file into a .ingot file (version 3 when it predicts a
 *         tensor or reorders payloads, else version 2), reading the source once, in order (a
 *         pipe works: `curl ... | ingot encode - out.gguf.ingot`). Each tensor takes the smallest
 *         of its candidate modes; everything else is stored raw, so the whole file rebuilds
 *         byte-identical. The directory is written before the payloads, so the output decodes
 *         as a stream.
 *
 * A dependent tensor (ingot_predict_rules.h) waits in memory until its partners are read and
 * written, so the payloads (and the directory) are in decode order, not source order; partners
 * stay in memory until their last dependent is coded.
 *
 * Candidates: BF16 -> plain, delta (base tensor), predict (dependents); Q8_0 / K-quants ->
 * quant, quant delta (base tensor of the same type and size); other GGUF types -> block delta
 * (base tensor of the same type and size); always raw.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_WRITER_H
#define FOUNDRY_INGOT_WRITER_H

#include <stdint.h>

#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_format.h"
#include "ingot/core/ingot_input.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Mode policy for coded tensors. */
typedef enum {
    INGOT_POLICY_AUTO = 0,  /* smallest candidate */
    INGOT_POLICY_PLAIN = 1, /* no delta modes (tests, benchmarks) */
    INGOT_POLICY_DELTA = 2, /* a delta mode whenever a matching base tensor exists (tests) */
} ingot_policy;

/** Progress callback: bytes of the source consumed so far, and its size (0 if unknown). */
typedef void (*ingot_progress_fn)(void *ctx, uint64_t done, uint64_t total);

/** Per-entry report, in payload order: the entry as written; v2_bytes = the payload of its best
 * non-predict choice (its own payload unless predicted); prefix_bytes = a predicted payload's
 * mode prefix (parameters + tables), else 0. */
typedef void (*ingot_entry_fn)(void *ctx, const ingot_entry *e, uint64_t v2_bytes,
                               uint64_t prefix_bytes);

typedef struct {
    int threads;
    ingot_base_set *base;        /* NULL: no delta modes */
    ingot_policy policy;
    const uint8_t *base_sha256;  /* optional: known digest of the single base file (skips
                                    hashing it); NULL: hashed */
    int self_check;              /* decode every coded tensor back from the output and compare
                                    it with the source (about one extra decode) */
    int no_predict;              /* no predict mode (and no reordering): a version 2 file;
                                    INGOT_POLICY_PLAIN implies it */
    int exact_v2;                /* also encode a predicted tensor's best other mode (into
                                    nothing) so v2 sizes are exact, not estimates */
    int max_version;             /* newest format version to write: 0 or 4 = match mode (version
                                    4) for layer roles; 3 = same-unit predict mode only */
    ingot_base_set *refs;        /* earlier files of the same model (match sources outside this
                                    file); must not hold this file's own tensors; NULL: none */
    const int *pool_layers;      /* match pool layer offsets (NULL: 1 and 16) */
    int pool_layer_count;
    int match_candidates;        /* search candidates per unit (0: 16) */
    long match_twin_offset;      /* also try units u -+ this offset (grown models copy neurons;
                                    Qwen3.5-27B lineage: 12288); 0: none */
    ingot_entry_fn entry_done;   /* optional */
    void *entry_ctx;
    ingot_progress_fn progress;  /* optional */
    void *progress_ctx;
} ingot_encode_options;

/** Totals for one encoded file. */
typedef struct {
    uint64_t source_bytes;
    uint64_t output_bytes;
    uint64_t elements;                 /* all tensor elements */
    uint64_t coded_elements[INGOT_MODE_COUNT];
    uint64_t payload_bytes[INGOT_MODE_COUNT];
    uint32_t tensors[INGOT_MODE_COUNT];
    uint64_t v2_output_bytes;          /* output_bytes with every predicted tensor in its best
                                          other mode (padding included) */
    uint8_t source_sha256[32];
} ingot_encode_stats;

/**
 * Encode the source read from `src` (position 0) to dst_path.
 *
 * @param src       Source input (file, pipe or callback)
 * @param dst_path  Output file (replaced; removed on failure), or NULL for a sizing run that
 *                  writes nothing (stats still exact)
 * @param opt       Options (NULL: 1 thread, no base, auto)
 * @param stats     Out: totals (may be NULL)
 * @return          0, -INGOT_EIO, -INGOT_EFORMAT, -INGOT_ETRUNCATED, -INGOT_EUNSUPPORTED,
 *                  -INGOT_ENOMEM
 *
 * Thread-safe: yes, with distinct base sets   Reentrant: yes   Signal-safe: no
 */
int ingot_encode_input(ingot_input *src, const char *dst_path, const ingot_encode_options *opt,
                       ingot_encode_stats *stats);

/**
 * ingot_encode_input on a path ("-" = stdin).
 *
 * Thread-safe: yes, with distinct base sets   Reentrant: yes   Signal-safe: no
 */
int ingot_encode_file(const char *src_path, const char *dst_path,
                      const ingot_encode_options *opt, ingot_encode_stats *stats);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_WRITER_H */
