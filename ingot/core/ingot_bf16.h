/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_bf16.h
 * @brief  BF16 bit fields, the Ingot plain-mode symbol split, and the ordered-integer map used by
 *         delta mode.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_BF16_H
#define FOUNDRY_INGOT_BF16_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline uint32_t ingot_bf16_sign(uint16_t u) { return (uint32_t)(u >> 15); }
static inline uint32_t ingot_bf16_exp(uint16_t u) { return (uint32_t)((u >> 7) & 0xFFu); }
static inline uint32_t ingot_bf16_mant(uint16_t u) { return (uint32_t)(u & 0x7Fu); }

/**
 * Exact widening BF16 -> f32.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline float ingot_bf16_to_f32(uint16_t u) {
    uint32_t bits = (uint32_t)u << 16;
    float f = 0.0f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/**
 * f32 -> BF16, round to nearest even (NaN kept quiet).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline uint16_t ingot_f32_to_bf16(float f) {
    uint32_t bits = 0;
    memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
        return (uint16_t)((bits >> 16) | 0x40u);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/**
 * Plain-mode entropy-coded symbol: exponent and the top 2 mantissa bits (0..1023).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline uint32_t ingot_bf16_symbol(uint16_t u) { return (uint32_t)((u >> 5) & 0x3FFu); }

/**
 * Plain-mode raw field: sign and the low 5 mantissa bits (0..63).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline uint32_t ingot_bf16_raw6(uint16_t u) {
    return (uint32_t)(((u >> 10) & 0x20u) | (u & 0x1Fu));
}

/**
 * Inverse of the symbol / raw6 split.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline uint16_t ingot_bf16_join(uint32_t symbol, uint32_t raw6) {
    return (uint16_t)(((raw6 & 0x20u) << 10) | (symbol << 5) | (raw6 & 0x1Fu));
}

/**
 * Monotone bijection BF16 bit pattern -> signed int: +0 -> 0, -0 -> -1, larger |w| further from
 * 0. Differences of ordered values count BF16 steps, across zero too.
 */
static inline int32_t ingot_bf16_ordered(uint16_t u) {
    int32_t mag = (int32_t)(u & 0x7FFFu);
    return (u & 0x8000u) ? -mag - 1 : mag;
}

/**
 * Inverse of ingot_bf16_ordered (v in [-32768, 32767]).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline uint16_t ingot_bf16_from_ordered(int32_t v) {
    return v < 0 ? (uint16_t)(0x8000u | (uint32_t)(-(v + 1))) : (uint16_t)v;
}

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_BF16_H */
