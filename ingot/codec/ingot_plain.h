/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_plain.h
 * @brief  Plain mode: one BF16 tensor coded alone. Symbol = exponent + top 2 mantissa bits under
 *         a static 16-bit rANS table per group; sign + low 5 mantissa bits raw (6 bits/weight).
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * A group is a slice with its own table: one per expert for MoE tensors (leading dim of a 3-D
 * tensor), else the whole tensor. Payload (little-endian, starts 8-aligned):
 *   u32 group_count, u32 block_log2, u64 group_weights, u64 block_count,
 *   group_count rANS tables, pad to 8, block area (ingot_block.h).
 * Blocks run group by group, 2^block_log2 weights each (the last block of a group may be short);
 * a block's raw stream is the 6-bit fields of its weights in order.
 */
#ifndef FOUNDRY_INGOT_PLAIN_H
#define FOUNDRY_INGOT_PLAIN_H

#include <stdint.h>

#include "ingot/codec/ingot_block.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Histograms and tables for one tensor, ready to size or encode. Opaque. */
typedef struct ingot_plain_plan ingot_plain_plan;

/**
 * Count symbols and build the group tables.
 *
 * @param weights      BF16 bit patterns; must outlive the plan
 * @param count        Weight count (> 0)
 * @param group_count  Groups (>= 1, divides count)
 * @param threads      Worker threads
 * @param out          Out: plan, freed with ingot_plain_plan_free
 * @return             0, -INGOT_EINVAL or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_plain_plan_create(const uint16_t *weights, uint64_t count, uint32_t group_count,
                            int threads, ingot_plain_plan **out);

/**
 * Payload size the plan will encode to, within a few bytes per block.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_plain_plan_bytes(const ingot_plain_plan *plan);

/**
 * Bytes before the block area (header + tables + pad), and the block count.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_plain_plan_prefix_bytes(const ingot_plain_plan *plan);
size_t ingot_plain_plan_blocks(const ingot_plain_plan *plan);

/**
 * Write the payload to the sink (sink position must be 8-aligned).
 * @return 0 or the first error
 *
 * Thread-safe: no (one sink)   Reentrant: yes   Signal-safe: no
 */
int ingot_plain_encode(const ingot_plain_plan *plan, int threads, ingot_sink *sink);

/**
 * Free a plan (NULL is a no-op).
 *
 * Thread-safe: no (no other call on the handle may run)   Reentrant: yes   Signal-safe: no
 */
void ingot_plain_plan_free(ingot_plain_plan *plan);

/** Block decoder for one plain payload (tables parsed, slot maps built). Opaque. */
typedef struct ingot_plain_decoder ingot_plain_decoder;

/**
 * Parse the payload prefix (header + tables). Memory: one 128 KB slot map per group.
 *
 * @param prefix        Payload start
 * @param avail         Bytes available at prefix (at least the whole prefix)
 * @param count         Expected weight count
 * @param out           Out: decoder, closed with ingot_plain_decoder_close
 * @param prefix_bytes  Out: where the block area starts, relative to prefix
 * @return              0, -INGOT_EFORMAT or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_plain_decoder_open(const uint8_t *prefix, uint64_t avail, uint64_t count,
                             ingot_plain_decoder **out, uint64_t *prefix_bytes);

/**
 * Block count, and the weight range [lo, hi) of block b.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_plain_decoder_blocks(const ingot_plain_decoder *d);
void ingot_plain_decoder_range(const ingot_plain_decoder *d, size_t b, uint64_t *lo,
                               uint64_t *hi);

/**
 * Decode block b into out (hi - lo weights).
 *
 * @return 0, -INGOT_EFORMAT or -INGOT_ECORRUPT
 *
 * Thread-safe: yes (distinct out buffers)   Reentrant: yes   Signal-safe: yes
 */
int ingot_plain_decoder_block(const ingot_plain_decoder *d, size_t b, const ingot_block_view *v,
                              uint16_t *out);

/**
 * Close (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_plain_decoder_close(ingot_plain_decoder *d);

/**
 * Decode a plain payload.
 *
 * @param payload  Payload bytes (8-aligned)
 * @param size     Payload size
 * @param count    Expected weight count
 * @param threads  Worker threads
 * @param out      [count] decoded BF16 bit patterns
 * @return         0, -INGOT_EFORMAT (bad structure), -INGOT_ECORRUPT (a block failed its end
 *                 check) or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_plain_decode(const uint8_t *payload, uint64_t size, uint64_t count, int threads,
                       uint16_t *out);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_PLAIN_H */
