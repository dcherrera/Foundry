/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_step.h
 * @brief  Step code for a signed integer difference d (|d| < 2^16): one symbol for the bit
 *         length of |d|, its sign and the two bits after the leading one (alphabet 119); the
 *         rest of |d| goes raw. Used by delta mode (BF16 steps) and the quantized coders (scale
 *         steps).
 *
 * Symbols: 0 -> d = 0; 1..2 -> |d| = 1 (+, -); 3..6 -> |d| in {2, 3} with sign;
 * 7 + 8 (L - 3) + 4 sign + next2 for bit length L >= 3, with L - 3 raw bits.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_STEP_H
#define FOUNDRY_INGOT_STEP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { INGOT_STEP_ALPHABET = 119 };

/** A step code: the symbol and the raw low bits of |d|. */
typedef struct {
    uint32_t symbol;
    uint32_t raw;
    unsigned raw_bits;
} ingot_step_code;

/**
 * Code d (|d| < 2^16).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline ingot_step_code ingot_step_encode(int32_t d) {
    uint32_t mag = (uint32_t)(d < 0 ? -d : d), sign = d < 0;
    unsigned b = 0;
    while (b < 17 && (mag >> b))
        b++;
    if (b == 0)
        return (ingot_step_code){0, 0, 0};
    if (b == 1)
        return (ingot_step_code){1 + sign, 0, 0};
    if (b == 2)
        return (ingot_step_code){3 + sign * 2 + (mag & 1u), 0, 0};
    uint32_t next2 = (mag >> (b - 3)) & 3u;
    return (ingot_step_code){7 + (b - 3) * 8 + sign * 4 + next2, mag & ((1u << (b - 3)) - 1u),
                             b - 3};
}

/**
 * Raw bits that follow a symbol (< INGOT_STEP_ALPHABET).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline unsigned ingot_step_raw_bits(uint32_t symbol) {
    return symbol < 7 ? 0u : (symbol - 7) / 8;
}

/**
 * Rebuild d from a symbol and its raw bits.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline int32_t ingot_step_join(uint32_t symbol, uint32_t raw) {
    uint32_t mag = 0, sign = 0;
    if (symbol == 0)
        return 0;
    if (symbol <= 2) {
        mag = 1;
        sign = symbol - 1;
    } else if (symbol <= 6) {
        sign = (symbol - 3) >> 1;
        mag = 2 | ((symbol - 3) & 1u);
    } else {
        uint32_t v = symbol - 7, low_bits = v / 8;
        sign = (v >> 2) & 1u;
        mag = ((4u | (v & 3u)) << low_bits) | raw;
    }
    return sign ? -(int32_t)mag : (int32_t)mag;
}

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_STEP_H */
