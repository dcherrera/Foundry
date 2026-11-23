/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_hash.h
 * @brief  64-bit content hash stored per tensor, so a decoder can check its output without the
 *         source file. Not cryptographic: it detects corruption, not tampering.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_HASH_H
#define FOUNDRY_INGOT_HASH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Hash `size` bytes. Defined in FORMAT.md: h = 0x6A09E667F3BCC909 ^ size; per little-endian
 * 64-bit word w (the tail zero-padded): h = (h ^ w) * 0x9E3779B97F4A7C15; h ^= h >> 32.
 *
 * @param data  Bytes to hash (any alignment)
 * @param size  Byte count
 * @return      The hash
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
uint64_t ingot_hash64(const void *data, size_t size);

/** Incremental form of ingot_hash64 for data that arrives in pieces (total size known first). */
typedef struct {
    uint64_t h;
    uint8_t tail[8];
    unsigned tail_bytes;
} ingot_hash64_state;

/**
 * Start a hash of `size` bytes in total.
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: yes
 */
void ingot_hash64_begin(ingot_hash64_state *s, uint64_t size);

/**
 * Feed the next n bytes. Any split gives the same result as ingot_hash64 over the whole.
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: yes
 */
void ingot_hash64_update(ingot_hash64_state *s, const void *data, size_t n);

/**
 * Finish: the hash of everything fed.
 *
 * Thread-safe: yes (caller-owned state)   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_hash64_end(ingot_hash64_state *s);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_HASH_H */
