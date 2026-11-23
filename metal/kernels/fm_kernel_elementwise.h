/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_elementwise.h
 * @brief Elementwise family: activations and binary ops over flat float32
 *   arrays. Bandwidth-bound, one thread per element (scalar form) or per
 *   four elements (`_4` float4 form).
 *
 * @status Complete: silu, sigmoid, add, mul, each in a scalar and a float4
 *   form. The float4 forms accept any n — the last thread handles a 1-3
 *   element tail scalar-wise.
 * @issues None
 * @todo None
 *
 * FUNCTION-CONSTANT INDICES OWNED BY THIS FAMILY: 120-129 (none used —
 *   these kernels have no shape to specialize on).
 *   Function constants share ONE index space across the whole assembled
 *   shader library, so each family reserves a block and two families
 *   colliding on an index fails the compile of every kernel in the backend,
 *   not just theirs. Allocated so far: 0-2 probe · 30 Q3_K · 50 Q5_K ·
 *   60 Q6_K · 70/80 Q8_0+Q4_K · 100-109 rope · 110-119 norm ·
 *   120-129 elementwise.
 */

#ifndef FOUNDRY_METAL_KERNEL_ELEMENTWISE_H
#define FOUNDRY_METAL_KERNEL_ELEMENTWISE_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Scalar forms — one element per thread.
 * ================================================================ */

/**
 * SiLU activation: out[i] = x[i] / (1 + exp(-x[i])).
 *
 * @param x    Input, n floats
 * @param out  Output, n floats (may alias x)
 * @param n    Element count
 *
 * Thread-safe: no (encodes into the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_silu(fm_buffer *x, fm_buffer *out, int n);

/**
 * Logistic sigmoid: out[i] = 1 / (1 + exp(-x[i])).
 *
 * @param x    Input, n floats
 * @param out  Output, n floats (may alias x)
 * @param n    Element count
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
void fm_sigmoid(fm_buffer *x, fm_buffer *out, int n);

/* Elementwise binary ops: out[i] = a[i] (+|*) b[i], n floats each. */
void fm_add(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n);
void fm_mul(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n);

/* ================================================================
 * float4 forms — four elements per thread.
 *
 * Identical results to the scalar forms, a quarter of the load/store
 * instructions. n need not be a multiple of 4: the grid is rounded up and
 * the final thread falls back to a scalar loop over the 1-3 leftover
 * elements. Every buffer is bound at offset 0 and a Metal allocation is
 * page aligned, so the 16-byte alignment a float4 access needs holds by
 * construction — there is no alignment precondition on the caller.
 *
 * MEASURED (M4, 10 GPU cores, macOS 26): these are NOT faster than the
 * scalar forms — 3.8 vs 4.8 GB/s at n=2048, 74.6 vs 78.9 at n=65536,
 * 117.2 vs 117.4 at n=1M. Fewer, wider loads buy nothing at either end:
 * past ~1M elements the scalar form already sits on the ~120 GB/s memory
 * roof, and below that the kernel is launch- and occupancy-bound, where
 * a quarter as many threads means a quarter as many threadgroups and so
 * fewer of the ten cores fed. Prefer the scalar forms; these exist
 * because the same reduction may pay on a wider GPU, and they are kept
 * correct and tested for that day. Do not reach for them as a speed
 * lever.
 * ================================================================ */

void fm_silu_4(fm_buffer *x, fm_buffer *out, int n);
void fm_sigmoid_4(fm_buffer *x, fm_buffer *out, int n);
void fm_add_4(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n);
void fm_mul_4(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_ELEMENTWISE_H */
