/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Portions modelled on ggml (MIT License, Copyright (c) 2023-2026 The ggml
 * authors, https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file fm_kernel_matvec_q8_0.h
 * @brief GGUF Q8_0 family: fused dequant + matvec against the standard 34-byte /
 *   32-weight Q8_0 block (one F16 scale per block).
 *
 *   Deliberately separate from fm_kernel_matvec_q8, which is a single-scale
 *   int8 dot — the wrong layout for block-scaled Q8_0.
 *
 *   In the Q3_K_M model this backend targets, Q8_0 carries 94 tensors
 *   (shared-expert projections, attention k/v, ssm_out).
 *
 * @status fm_matvec_q8_0 is implemented and covered by an absolute-correctness
 *   test driven by real blocks from the model file.
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_METAL_KERNEL_MATVEC_Q8_0_H
#define FOUNDRY_METAL_KERNEL_MATVEC_Q8_0_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Function-constant indices reserved for this family in the shared MSL
 * translation unit.  Indices are GLOBAL across the assembled library, so a
 * collision is a whole-library compile failure.  Allocation:
 *   0..2  probe   30  Q3_K   50  Q5_K   60  Q6_K   70  Q8_0   80  Q4_K
 * This family's kernel needs no specialization today — 70..79 stay
 * reserved so it can add one without hunting for a free index. */
#define FM_Q8_0_CONSTANT_INDEX_BASE 70

/* Bytes in one GGUF Q8_0 block: F16 scale + 32 int8 quants. */
#define FM_Q8_0_BLOCK_BYTES   34

/* Weights per GGUF Q8_0 block. */
#define FM_Q8_0_BLOCK_WEIGHTS 32

/**
 * Q8_0 quantized matrix-vector product: out[r] = sum_c dequant(w)[r][c] * x[c].
 *
 * Fused dequant + dot, no scratch: the weight stays GPU-resident as raw
 * 34-byte blocks, so only ~1.0625 bytes per weight are read.  A block
 * decodes as value[i] = d * q[i], with d an F16 scale and q[i] one of 32
 * int8 quants — the standard GGUF layout the loader mmaps, not a repack.
 *
 * Every operand is an fm_bufref because that is the only shape that can
 * read a weight out of an fm_region: no GGUF tensor is page-aligned, so a
 * weight only ever exists at a byte offset inside one big no-copy region
 * buffer.  Pass fm_bufref_at(buf, 0) for a standalone buffer.
 *
 * @param w                 Q8_0 block bytes, [rows, cols]
 * @param x                 [cols] float32 input
 * @param out               [rows] float32 output
 * @param rows              Output rows (one threadgroup each)
 * @param cols              Input length; must be a multiple of 32
 * @param row_stride_bytes  Bytes per weight row, (cols/32)*34
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_matvec_q8_0(fm_bufref w, fm_bufref x, fm_bufref out,
                    int rows, int cols, int row_stride_bytes);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_MATVEC_Q8_0_H */
