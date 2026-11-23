/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_norm.h
 * @brief Normalization family: row softmax, RMS normalization over one
 *   vector or a batch of rows, and the per-head QK-RMSNorm the attention
 *   block needs. The reductions are cooperative (simd_sum + a 32-slot
 *   threadgroup staging array), except QK-RMSNorm which runs one SIMD group
 *   per head and needs no threadgroup memory at all.
 *
 * @status Complete: softmax, rms_norm, rms_norm_nx, qk_rms_norm. All three
 *   RMS entry points honor weight == NULL.
 * @issues None
 * @todo None
 *
 * OCCUPANCY OF THE ROW MAPPING: rms_norm_nx runs one threadgroup of
 *   min(n, 256) threads per row with consecutive lanes on consecutive
 *   elements, so at n_embd = 2048 every lane of all 8 SIMD groups carries
 *   8 elements — no idle lane, and each load coalesces. Measured on M4:
 *   512 rows x 2048 in 0.053 ms, 238 GB/s counting x read twice plus the
 *   write, i.e. above the ~120 GB/s DRAM roof because a 4 MB working set
 *   stays in cache. Lanes only idle for rows narrower than 32, which no
 *   shape in this model has. qk_rms_norm runs ONE SIMD group per head and
 *   head_dim = 256 gives each lane 8 elements, likewise fully fed; at 16
 *   heads it is dispatch-bound (0.009 ms) rather than bandwidth-bound.
 *
 * FUNCTION-CONSTANT INDICES OWNED BY THIS FAMILY: 110-119 (110 used).
 *   Function constants share ONE index space across the whole assembled
 *   shader library, so each family reserves a block and two families
 *   colliding on an index fails the compile of every kernel in the backend,
 *   not just theirs. Allocated so far: 0-2 probe · 30 Q3_K · 50 Q5_K ·
 *   60 Q6_K · 70/80 Q8_0+Q4_K · 100-109 rope · 110-119 norm ·
 *   120-129 elementwise (claimed, none used).
 */

#ifndef FOUNDRY_METAL_KERNEL_NORM_H
#define FOUNDRY_METAL_KERNEL_NORM_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Softmax over last dimension.
 * x: [rows, cols], out: [rows, cols] */
void fm_softmax(fm_buffer *x, fm_buffer *out, int rows, int cols);

/**
 * RMS normalization over one vector: out = x * rsqrt(mean(x^2) + eps) * weight.
 *
 * @param x       Input, n floats
 * @param weight  Per-element gain, n floats, or NULL for no gain
 * @param out     Output, n floats (may alias x)
 * @param n       Vector length
 * @param eps     Added to mean(x^2) before the rsqrt
 *
 * Thread-safe: no (encodes into the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_rms_norm(fm_buffer *x, fm_buffer *weight, fm_buffer *out,
                 int n, float eps);

/**
 * RMS normalization over a batch of contiguous rows — the prefill form of
 * fm_rms_norm. Each row is normalized independently; the weight is shared
 * by every row, exactly as the single-vector form applies it.
 *
 * @param x       Input, [rows, n] row-major, no padding between rows
 * @param weight  Per-element gain, n floats, or NULL for no gain
 * @param out     Output, [rows, n] (may alias x)
 * @param rows    Number of rows
 * @param n       Elements per row, which is also the row stride
 * @param eps     Added to mean(x^2) before the rsqrt
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
void fm_rms_norm_nx(fm_buffer *x, fm_buffer *weight, fm_buffer *out,
                    int rows, int n, float eps);

/**
 * Per-head QK-RMSNorm: RMS-normalizes each head of a [n_heads, head_dim]
 * activation independently.
 *
 * The weight is SHARED across heads — it is head_dim long, not
 * n_heads * head_dim, and head h reads weight[0 .. head_dim), not
 * weight[h * head_dim ...]. That mirrors the GGUF layout, which stores one
 * pre-baked norm weight per tensor and applies it to every head.
 *
 * @param x         Input, [n_heads, head_dim] row-major
 * @param weight    Shared per-head gain, head_dim floats, or NULL for none
 * @param out       Output, [n_heads, head_dim] (may alias x)
 * @param n_heads   Number of heads
 * @param head_dim  Elements per head
 * @param eps       Added to mean(x^2) before the rsqrt
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
void fm_qk_rms_norm(fm_buffer *x, fm_buffer *weight, fm_buffer *out,
                    int n_heads, int head_dim, float eps);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_NORM_H */
