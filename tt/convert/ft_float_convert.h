/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_float_convert.h
 * @brief Scalar conversions between fp32, fp16 and bf16 for the weight
 *   conversion path.
 *
 * Pure C99 bit arithmetic. **No SIMD intrinsics of any kind** — the conversion
 * path must build unchanged on Apple Silicon and on the card host's no-AVX Pentium
 * (spec §8.4), so nothing here may reach for `<immintrin.h>` or `<arm_neon.h>`.
 * A compiler is free to vectorise these loops; the source must not assume it.
 *
 * fp32 -> bf16 rounds to nearest, ties to even, matching the block-float
 * packer's rounding so a precision-ladder step does not also change the
 * rounding mode underneath the measurement.
 *
 * @status Used by the converter for the bf16 rung of the precision ladder and
 *   by the GGUF reader for F16 tensors.
 * @issues None
 * @todo None
 */

#ifndef FT_FLOAT_CONVERT_H
#define FT_FLOAT_CONVERT_H

#include <stddef.h>
#include <stdint.h>

/** IEEE binary16 to fp32. Handles denormals, infinities and NaN. */
float ft_float_from_fp16(uint16_t half);

/**
 * fp32 to bfloat16, round to nearest with ties to even.
 * NaN is preserved as a quiet NaN rather than being allowed to round into an
 * infinity.
 */
uint16_t ft_bf16_from_float(float value);

/** bfloat16 back to fp32. Exact — bf16 is fp32 with the low 16 bits cleared. */
float ft_float_from_bf16(uint16_t bf16);

/** Convert `count` fp32 values to bf16. Buffers may not overlap. */
void ft_bf16_from_float_array(const float *src, uint16_t *dst, size_t count);

/** Convert `count` bf16 values to fp32. Buffers may not overlap. */
void ft_float_from_bf16_array(const uint16_t *src, float *dst, size_t count);

#endif /* FT_FLOAT_CONVERT_H */
