/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_q8.h
 * @brief  Q8_0 scale provenance: the fp16 scale ggml stores is fp16(f32(amax / 127)) of a BF16
 *         amax, so a scale is coded as that amax. Exposed for tests; the Q8_0 type coder itself
 *         is ingot_q8_qtype() (ingot_qtype.h).
 *
 * Q8_0 block: fp16 d, int8 codes[32] (34 bytes); ggml quantizes from BF16 with d = amax / 127
 * (f32) and codes = round(w / d).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_Q8_H
#define FOUNDRY_INGOT_Q8_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { INGOT_Q8_BLOCK = 32, INGOT_Q8_BLOCK_BYTES = 34 };

/**
 * fp16 bits of fp16(f32(amax / 127)) for a non-negative finite BF16 amax, computed in integers
 * (exact under any floating-point flags; both roundings to nearest even).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint16_t ingot_q8_scale_of_amax(uint16_t amax_bf16);

/**
 * A BF16 amax with ingot_q8_scale_of_amax(amax) == d16.
 *
 * @return 0 and *amax, or -1 if d16 has no such amax (not quantized from BF16)
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_q8_amax_of_scale(uint16_t d16, uint16_t *amax);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_Q8_H */
