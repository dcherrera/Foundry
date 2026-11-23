/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_ggml.h
 * @brief  ggml tensor types as stored in GGUF files: block geometry (weights and bytes per
 *         block), names, and the mapping to ingot dtype codes.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_GGML_H
#define FOUNDRY_INGOT_GGML_H

#include <stdint.h>

#include "ingot/container/ingot_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/** ggml type ids the coders handle by name (values fixed by the GGUF format). */
enum {
    INGOT_GGML_F32 = 0,
    INGOT_GGML_F16 = 1,
    INGOT_GGML_Q8_0 = 8,
    INGOT_GGML_Q3_K = 11,
    INGOT_GGML_Q4_K = 12,
    INGOT_GGML_Q5_K = 13,
    INGOT_GGML_Q6_K = 14,
    INGOT_GGML_BF16 = 30,
};

/**
 * Block geometry of a ggml type.
 *
 * @param type   ggml type id
 * @param weights Out: weights per block
 * @param bytes   Out: bytes per block
 * @return       0, or -INGOT_EUNSUPPORTED for a type this build does not know
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_ggml_geometry(uint32_t type, uint32_t *weights, uint32_t *bytes);

/**
 * Name of a ggml type ("Q8_0"), or NULL if unknown.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const char *ingot_ggml_name(uint32_t type);

/**
 * ingot dtype code for a ggml type: plain types map to their own code (F32, F16, BF16, I8, ...),
 * the rest to INGOT_DTYPE_GGML + type.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
ingot_dtype ingot_ggml_dtype(uint32_t type);

/**
 * Inverse of ingot_ggml_dtype: the ggml type of a dtype code. @return 0 or -INGOT_EUNSUPPORTED
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_dtype_ggml_type(ingot_dtype dtype, uint32_t *type);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_GGML_H */
