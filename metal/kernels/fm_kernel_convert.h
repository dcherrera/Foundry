/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_convert.h
 * @brief Precision-conversion family: bulk F32 <-> F16 copies over flat
 *   arrays. Support kernels for paths that stage operands in F16 (MPS runs
 *   F16 matmul at roughly twice the F32 throughput on Apple GPUs).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_METAL_KERNEL_CONVERT_H
#define FOUNDRY_METAL_KERNEL_CONVERT_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * out[i] = half(in[i]) for i in [0, n).
 *
 * @param in_f32   Source, at least n floats
 * @param out_f16  Destination, at least n halfs
 * @param n        Element count
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_convert_f32_to_f16(fm_buffer *in_f32, fm_buffer *out_f16, int n);

/* out[i] = float(in[i]) for i in [0, n).  See fm_convert_f32_to_f16. */
void fm_convert_f16_to_f32(fm_buffer *in_f16, fm_buffer *out_f32, int n);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_CONVERT_H */
