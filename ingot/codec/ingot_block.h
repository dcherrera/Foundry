/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_block.h
 * @brief  Block framing shared by the plain and delta payloads, the output sink, and the
 *         parallel block-encode driver that writes the offset table and blocks.
 *
 * Block layout (little-endian, starts 8-aligned in the file):
 *   u32 raw_bytes, u32 word_count, u32 states[32], u8 raw[raw_bytes], pad to 2,
 *   u16 words[word_count], pad to 8.
 * Block area: u64 offsets[block_count + 1] (relative to the first block; the last one is the area
 * size), then the blocks in order.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_BLOCK_H
#define FOUNDRY_INGOT_BLOCK_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/core/ingot_rans.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    INGOT_BLOCK_LOG2 = 18,
    INGOT_BLOCK_HEADER_BYTES = 8 + 4 * INGOT_RANS_LANES,
};

/** Output sink: a file descriptor (pwrite); with fd == INGOT_SINK_MEMORY (or any other negative fd
 * but INGOT_SINK_DISCARD) a growable memory buffer; with fd == INGOT_SINK_DISCARD nothing is kept,
 * only the position counts (sizing runs). */
#define INGOT_SINK_MEMORY (-1)
#define INGOT_SINK_DISCARD (-2)

typedef struct {
    int fd;
    uint8_t *mem;
    size_t mem_capacity;
    uint64_t pos;
} ingot_sink;

/**
 * Append n bytes. @return 0, -INGOT_EIO or -INGOT_ENOMEM
 *
 * Thread-safe: no (one sink per caller)   Reentrant: yes   Signal-safe: no
 */
int ingot_sink_write(ingot_sink *s, const void *data, size_t n);

/**
 * Overwrite n bytes at absolute offset `at` (< pos). @return 0, -INGOT_EIO or -INGOT_EINVAL
 *
 * Thread-safe: no (one sink per caller)   Reentrant: yes   Signal-safe: no
 */
int ingot_sink_write_at(ingot_sink *s, uint64_t at, const void *data, size_t n);

/**
 * Append zero bytes until pos is a multiple of `align`.
 *
 * Thread-safe: no (one sink per caller)   Reentrant: yes   Signal-safe: no
 */
int ingot_sink_pad(ingot_sink *s, unsigned align);

/** One encoded block, owned by the caller of the encode callback (freed by the driver). */
typedef struct {
    uint8_t *data;
    size_t size;
} ingot_block_buf;

/**
 * Serialize a block. Allocates out->data (free()).
 * @return 0 or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_block_serialize(const uint32_t *states, const uint8_t *raw, size_t raw_bytes,
                          const uint16_t *words, size_t word_count, ingot_block_buf *out);

/** Encode block `index` into *out. */
typedef int (*ingot_block_encode_fn)(void *ctx, size_t index, ingot_block_buf *out);

/**
 * Write the block area: offsets placeholder, every block (encoded in parallel batches, written
 * in order), then the real offsets. Peak memory is one batch of blocks.
 *
 * @return 0 or the first error
 *
 * Thread-safe: no (one sink per caller)   Reentrant: yes   Signal-safe: no
 */
int ingot_block_area_write(ingot_sink *s, size_t block_count, int threads,
                           ingot_block_encode_fn fn, void *ctx);

/** Parsed view of one block (pointers into the payload). */
typedef struct {
    uint32_t states[INGOT_RANS_LANES];
    const uint8_t *raw;
    size_t raw_bytes;
    const uint16_t *words;
    size_t word_count;
} ingot_block_view;

/** Block area located inside a payload. */
typedef struct {
    const uint8_t *blocks;
    uint64_t size;
    size_t block_count;
    const uint8_t *offsets;
} ingot_block_area;

/**
 * Locate and validate a block area at `area` (offsets monotone, within `avail` bytes).
 * @return 0 or -INGOT_EFORMAT
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_block_area_open(const uint8_t *area, uint64_t avail, size_t block_count,
                          ingot_block_area *out);

/**
 * View block `index` of an opened area. @return 0 or -INGOT_EFORMAT
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_block_view_get(const ingot_block_area *a, size_t index, ingot_block_view *v);

/**
 * End-of-block check: every lane back at the encoder start state and every word consumed.
 * @return 0 or -INGOT_ECORRUPT
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_block_check_end(const uint32_t *x, size_t words_read, const ingot_block_view *v);

/**
 * Read a little-endian u32 / u64 from any alignment.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint32_t ingot_load_u32(const uint8_t *p);
uint64_t ingot_load_u64(const uint8_t *p);

/**
 * Write a little-endian u32 / u64.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_store_u32(uint8_t *p, uint32_t v);
void ingot_store_u64(uint8_t *p, uint64_t v);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_BLOCK_H */
