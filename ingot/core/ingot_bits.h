/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_bits.h
 * @brief  LSB-first bit packing for the raw (uncoded) stream of a block: field i starts at the
 *         bit right after field i-1, bit 0 of byte 0 first.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_BITS_H
#define FOUNDRY_INGOT_BITS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Writer over a caller-owned zeroed buffer of `capacity` bytes. */
typedef struct {
    uint8_t *buf;
    size_t capacity;
    uint64_t bitpos;
} ingot_bit_writer;

/**
 * Append the low `nbits` (<= 25) of value; returns 0, or -1 if the buffer is full.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline int ingot_bits_put(ingot_bit_writer *w, uint32_t value, unsigned nbits) {
    if (!nbits)
        return 0;
    uint64_t end = w->bitpos + nbits;
    if ((end + 7) / 8 > w->capacity)
        return -1;
    size_t byte = (size_t)(w->bitpos >> 3);
    unsigned shift = (unsigned)(w->bitpos & 7);
    uint32_t v = (value & ((1u << nbits) - 1u)) << shift;
    for (unsigned b = 0; b * 8 < shift + nbits; b++)
        w->buf[byte + b] |= (uint8_t)(v >> (8 * b));
    w->bitpos = end;
    return 0;
}

/**
 * Bytes written so far (the partial last byte counts).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline size_t ingot_bits_bytes(const ingot_bit_writer *w) {
    return (size_t)((w->bitpos + 7) / 8);
}

/** Reader over `size` bytes. */
typedef struct {
    const uint8_t *buf;
    size_t size;
    uint64_t bitpos;
} ingot_bit_reader;

/**
 * Read `nbits` (<= 25) into *value; returns 0, or -1 past the end.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline int ingot_bits_get(ingot_bit_reader *r, unsigned nbits, uint32_t *value) {
    if (!nbits) {
        *value = 0;
        return 0;
    }
    uint64_t end = r->bitpos + nbits;
    if ((end + 7) / 8 > r->size)
        return -1;
    size_t byte = (size_t)(r->bitpos >> 3);
    unsigned shift = (unsigned)(r->bitpos & 7);
    uint32_t v = 0;
    for (unsigned b = 0; b * 8 < shift + nbits; b++)
        v |= (uint32_t)r->buf[byte + b] << (8 * b);
    *value = (v >> shift) & ((1u << nbits) - 1u);
    r->bitpos = end;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_BITS_H */
