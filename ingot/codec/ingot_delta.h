/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_delta.h
 * @brief  Delta mode: a BF16 tensor coded against a base tensor of the same shape (a fine-tune
 *         against its parent). The decoder needs the bit-identical base.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Rows are the last dimension (cols = shape[ndim-1]). Per row, side info row_scale (i8) =
 * round(2 log2(mean |t - b|)), or -128 if the row equals the base (no symbols). Per weight:
 *   e = max(exp(base), 1), k = row_scale - 2 (e - 134)   (expected residual size, half-binades)
 *   k > 12  -> fallback context: plain symbol (exp + top 2 mantissa bits) of the target, 6 raw
 *   else    -> step context max(k, -16) + 16: d = ord(t) - ord(b) (BF16 steps, ingot_bf16.h),
 *              symbol = bit length of |d|, sign and the 2 bits after the leading one; the rest
 *              of |d| raw.
 * Payload (little-endian, starts 8-aligned):
 *   u32 context_count (30), u32 block_log2, u64 rows, u64 cols, u64 block_count,
 *   i8 row_scale[rows], pad to 8, context_count rANS tables (empty if unused), pad to 8,
 *   block area (ingot_block.h). Blocks cover the flat weight index, 2^block_log2 each; positions
 *   in identical rows carry no symbol; the raw stream is the raw bits of the block in order.
 */
#ifndef FOUNDRY_INGOT_DELTA_H
#define FOUNDRY_INGOT_DELTA_H

#include <stdint.h>

#include "ingot/codec/ingot_block.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Row scales, histograms and context tables for one target/base pair. Opaque. */
typedef struct ingot_delta_plan ingot_delta_plan;

/**
 * Compute row scales and build the context tables.
 *
 * @param target   [rows * cols] BF16 target; must outlive the plan
 * @param base     [rows * cols] BF16 base; must outlive the plan
 * @param rows     Row count (> 0)
 * @param cols     Row length (> 0)
 * @param threads  Worker threads
 * @param out      Out: plan, freed with ingot_delta_plan_free
 * @return         0, -INGOT_EINVAL or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_delta_plan_create(const uint16_t *target, const uint16_t *base, uint64_t rows,
                            uint64_t cols, int threads, ingot_delta_plan **out);

/**
 * Payload size the plan will encode to, within a few bytes per block.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_delta_plan_bytes(const ingot_delta_plan *plan);

/**
 * Bytes before the block area (header + row scales + tables + pads), and the block count.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_delta_plan_prefix_bytes(const ingot_delta_plan *plan);
size_t ingot_delta_plan_blocks(const ingot_delta_plan *plan);

/**
 * Write the payload to the sink (sink position must be 8-aligned).
 * @return 0 or the first error
 *
 * Thread-safe: no (one sink)   Reentrant: yes   Signal-safe: no
 */
int ingot_delta_encode(const ingot_delta_plan *plan, int threads, ingot_sink *sink);

/**
 * Free a plan (NULL is a no-op).
 *
 * Thread-safe: no (no other call on the handle may run)   Reentrant: yes   Signal-safe: no
 */
void ingot_delta_plan_free(ingot_delta_plan *plan);

/** Block decoder for one delta payload (row scales copied, tables parsed). Opaque. */
typedef struct ingot_delta_decoder ingot_delta_decoder;

/**
 * Parse the payload prefix. Memory: rows bytes + 30 slot maps of 128 KB.
 *
 * @param prefix        Payload start
 * @param avail         Bytes available at prefix (at least the whole prefix)
 * @param count         Expected weight count
 * @param out           Out: decoder, closed with ingot_delta_decoder_close
 * @param prefix_bytes  Out: where the block area starts, relative to prefix
 * @return              0, -INGOT_EFORMAT or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_delta_decoder_open(const uint8_t *prefix, uint64_t avail, uint64_t count,
                             ingot_delta_decoder **out, uint64_t *prefix_bytes);

/**
 * Block count, and the weight range [lo, hi) of block b.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_delta_decoder_blocks(const ingot_delta_decoder *d);
void ingot_delta_decoder_range(const ingot_delta_decoder *d, size_t b, uint64_t *lo,
                               uint64_t *hi);

/**
 * Decode block b: base and out cover the block's weights [lo, hi).
 *
 * @return 0 or -INGOT_ECORRUPT
 *
 * Thread-safe: yes (distinct out buffers)   Reentrant: yes   Signal-safe: yes
 */
int ingot_delta_decoder_block(const ingot_delta_decoder *d, size_t b, const ingot_block_view *v,
                              const uint16_t *base, uint16_t *out);

/**
 * Close (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_delta_decoder_close(ingot_delta_decoder *d);

/**
 * Decode a delta payload.
 *
 * @param payload  Payload bytes (8-aligned)
 * @param size     Payload size
 * @param base     [count] the base the payload was coded against
 * @param count    Expected weight count
 * @param threads  Worker threads
 * @param out      [count] decoded BF16 bit patterns (must not alias base)
 * @return         0, -INGOT_EFORMAT, -INGOT_ECORRUPT or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_delta_decode(const uint8_t *payload, uint64_t size, const uint16_t *base,
                       uint64_t count, int threads, uint16_t *out);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_DELTA_H */
