/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_probe.h
 * @brief Diagnostic kernel that reports the function constants its pipeline
 *   was specialized with, so tests can prove specialization reached the shader
 *   compiler and not just the pipeline cache key. Not part of any compute path.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_METAL_KERNEL_PROBE_H
#define FOUNDRY_METAL_KERNEL_PROBE_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Function-constant indices read by the `constant_probe` kernel. */
#define FM_FC_PROBE_REPEAT_COUNT  0   /* int:   compile-time loop bound */
#define FM_FC_PROBE_SCALE         1   /* short: multiplies the loop sum */
#define FM_FC_PROBE_FLAG          2   /* bool:  adds a 1000.0 offset    */

/**
 * Diagnostic kernel that reports the function constants it was compiled
 * with.  Writes, for each element i of `out`:
 *
 *     out[i] = sum(1..repeat_count) * scale + (flag ? 1000 : 0) + i
 *
 * where the sum is an unrolled loop whose bound is the compile-time
 * constant.  Unset constants default to repeat_count 0, scale 1, flag
 * false, so an unspecialized dispatch writes out[i] = i.
 *
 * @param out        Destination buffer, at least count floats
 * @param count      Number of elements to write
 * @param constants  Constant set, or NULL to run with the defaults
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_constant_probe(fm_buffer *out, int count, const fm_fc_values *constants);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_PROBE_H */
