/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_qtensor.h
 * @brief  Quantized-tensor payloads (modes QUANT, QUANT_DELTA, BLOCK_DELTA): a GGUF tensor of
 *         ggml blocks coded by its type coder (ingot_qtype.h), optionally against the same
 *         tensor of a base model. Plan + encode on the writer side; a block decoder on the reader
 *         side (random access and streaming).
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * A tensor is rows x cols weights (cols = the contiguous dimension, a multiple of the ggml block
 * size). Coded blocks hold whole rows, about 2^18 weights each (at least one row), and are
 * independent. Payload prefix (after the version 2 frame, little-endian):
 *   u32 ggml_type, u32 flags (bit 0: base), u64 rows, u64 cols, u64 rows_per_block,
 *   u64 block_count, u32 context_count, u32 0, context_count rANS tables, pad to 8.
 */
#ifndef FOUNDRY_INGOT_QTENSOR_H
#define FOUNDRY_INGOT_QTENSOR_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/codec/ingot_block.h"
#include "ingot/container/ingot_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 1 if modes QUANT / QUANT_DELTA have a coder for this dtype (Q8_0, Q3_K..Q6_K), else 0.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_qtensor_supports(ingot_dtype dtype);

/**
 * 1 if mode `mode` can code a tensor of this dtype and row length, else 0.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_qtensor_fits(ingot_mode mode, ingot_dtype dtype, uint64_t cols);

/** Histograms and tables for one tensor. Opaque. */
typedef struct ingot_qtensor_plan ingot_qtensor_plan;

/**
 * Count every field and build the tables.
 *
 * @param mode     INGOT_MODE_QUANT, _QUANT_DELTA or _BLOCK_DELTA
 * @param dtype    Tensor dtype
 * @param data     rows x row bytes of the tensor; must outlive the plan
 * @param base     Same layout, the base tensor (delta modes), else NULL; must outlive the plan
 * @param rows     Row count
 * @param cols     Weights per row
 * @param threads  Worker threads
 * @param out      Out: plan, freed with ingot_qtensor_plan_free
 * @return         0, -INGOT_EINVAL (mode does not fit), -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_qtensor_plan_create(ingot_mode mode, ingot_dtype dtype, const uint8_t *data,
                              const uint8_t *base, uint64_t rows, uint64_t cols, int threads,
                              ingot_qtensor_plan **out);

/**
 * Payload size (after the frame) within a few bytes per block; bytes before the block area;
 * block count.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_qtensor_plan_bytes(const ingot_qtensor_plan *plan);
uint64_t ingot_qtensor_plan_prefix_bytes(const ingot_qtensor_plan *plan);
size_t ingot_qtensor_plan_blocks(const ingot_qtensor_plan *plan);

/**
 * Write the payload (prefix + block area) to the sink (8-aligned position).
 * @return 0 or the first error
 *
 * Thread-safe: no (one sink)   Reentrant: yes   Signal-safe: no
 */
int ingot_qtensor_encode(const ingot_qtensor_plan *plan, int threads, ingot_sink *sink);

/**
 * Free (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_qtensor_plan_free(ingot_qtensor_plan *plan);

/** Block decoder for one payload. Opaque. */
typedef struct ingot_qtensor_decoder ingot_qtensor_decoder;

/**
 * Parse the prefix and check it against the directory's mode, dtype and shape.
 *
 * @param prefix_bytes  Out: where the block area starts, relative to prefix
 * @return              0, -INGOT_EFORMAT or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_qtensor_decoder_open(ingot_mode mode, ingot_dtype dtype, uint64_t rows, uint64_t cols,
                               const uint8_t *prefix, uint64_t avail,
                               ingot_qtensor_decoder **out, uint64_t *prefix_bytes);

/**
 * Block count, and the byte range [lo, hi) of block b within the tensor.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_qtensor_decoder_blocks(const ingot_qtensor_decoder *d);
void ingot_qtensor_decoder_range(const ingot_qtensor_decoder *d, size_t b, uint64_t *lo,
                                 uint64_t *hi);

/**
 * Decode block b; base (delta modes) and out cover the block's byte range.
 *
 * @return 0, -INGOT_ECORRUPT or -INGOT_ENOMEM
 *
 * Thread-safe: yes (distinct out buffers)   Reentrant: yes   Signal-safe: no
 */
int ingot_qtensor_decoder_block(const ingot_qtensor_decoder *d, size_t b,
                                const ingot_block_view *v, const uint8_t *base, uint8_t *out);

/**
 * Close (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_qtensor_decoder_close(ingot_qtensor_decoder *d);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_QTENSOR_H */
