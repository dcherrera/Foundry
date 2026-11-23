/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_writer_tensor.h
 * @brief  The writer's per-tensor coder: plan every candidate mode of one buffered tensor (plain,
 *         delta, quant, quant delta, block delta, predict), keep the smallest, and code its
 *         payload into memory (optionally decoding it back as a self-check).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_WRITER_TENSOR_H
#define FOUNDRY_INGOT_WRITER_TENSOR_H

#include <stdint.h>

#include "ingot/codec/ingot_block.h"
#include "ingot/codec/ingot_predict.h"
#include "ingot/container/ingot_format.h"
#include "ingot/container/ingot_writer.h"
#include "ingot/container/ingot_writer_match.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Partners of a dependent tensor (predict candidate). */
typedef struct {
    ingot_predict_geometry geom;                          /* partner entries = directory indices */
    const ingot_entry *entry[INGOT_PREDICT_MAX_PARTNERS];
    const uint8_t *data[INGOT_PREDICT_MAX_PARTNERS];
    const ingot_writer_match *match;                      /* match candidate, or NULL */
} ingot_writer_partners;

/** What the coder decided, for the stats and the entry report. */
typedef struct {
    uint64_t v2_bytes;     /* payload of the best non-predict choice (= the payload unless
                              predicted; exact with opt->exact_v2, else the estimate) */
    uint64_t prefix_bytes; /* predicted: the mode prefix (parameters + tables), else 0 */
    uint32_t match_refs[INGOT_MATCH_MAX_SOURCES]; /* match: the payload's source refs */
    uint32_t match_ref_count;
} ingot_writer_coded;

/**
 * Code one buffered tensor: sets e->mode and e->base_hash (delta: base tensor hash; predict:
 * partner field) and writes the payload (raw bytes or a framed coded payload) to `payload`.
 *
 * @param opt       Encode options
 * @param e         Entry (name, dtype, shape, source size; hash already set)
 * @param data      The tensor's source bytes
 * @param partners  Predict candidate, or NULL
 * @param payload   Memory sink (fd INGOT_SINK_MEMORY, position 0)
 * @param coded     Out: stats details
 * @return          0, -INGOT_ECORRUPT (self-check failed), -INGOT_ENOMEM, base errors
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_writer_code_tensor(const ingot_encode_options *opt, ingot_entry *e,
                             const uint8_t *data, const ingot_writer_partners *partners,
                             ingot_sink *payload, ingot_writer_coded *coded);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_WRITER_TENSOR_H */
