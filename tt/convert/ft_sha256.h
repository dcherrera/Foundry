/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_sha256.h
 * @brief SHA-256 (FIPS 180-4), streaming. Written from the standard; no dependency.
 *
 * Used where a file format names its payloads by content (the MXTL tile file, the FTTW
 * splice record): a collision-resistant digest that another repo's tooling can compute with
 * `sha256sum`, which FNV-1a — what the provenance trailer uses for a cheap identity — is not.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FT_SHA256_H
#define FT_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Digest width in bytes. */
#define FT_SHA256_BYTES 32u

/** Streaming state. Zeroing is not initialising: call ft_sha256_init(). */
typedef struct {
    uint32_t state[8];
    uint64_t total_bytes;
    unsigned char block[64];
    size_t block_used;
} ft_sha256_ctx;

/**
 * Start a digest.
 *
 * Thread-safe: yes (caller-owned state)    Reentrant: yes    Signal-safe: yes
 */
void ft_sha256_init(ft_sha256_ctx *ctx);

/**
 * Feed `bytes` bytes. Any split of the input across calls gives the same digest.
 *
 * Thread-safe: yes (caller-owned state)    Reentrant: yes    Signal-safe: yes
 */
void ft_sha256_update(ft_sha256_ctx *ctx, const void *data, size_t bytes);

/**
 * Finish and write the 32-byte digest. `ctx` must be re-initialised before reuse.
 *
 * Thread-safe: yes (caller-owned state)    Reentrant: yes    Signal-safe: yes
 */
void ft_sha256_final(ft_sha256_ctx *ctx, unsigned char out[FT_SHA256_BYTES]);

/**
 * One-shot digest of a buffer.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
void ft_sha256(const void *data, size_t bytes, unsigned char out[FT_SHA256_BYTES]);

/**
 * Lower-case hex of a digest into `out` (65 bytes: 64 digits and the NUL).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
void ft_sha256_hex(const unsigned char digest[FT_SHA256_BYTES], char out[2u * FT_SHA256_BYTES + 1u]);

#ifdef __cplusplus
}
#endif

#endif /* FT_SHA256_H */
