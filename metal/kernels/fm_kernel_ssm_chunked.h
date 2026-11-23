/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_ssm_chunked.h
 * @brief Chunked delta-net primitives for prefill: cumulative sum, triangular
 *   mask/diag/fill, and the triangular solve that collapses a chunk's rank-1
 *   updates into one matrix. Turns sequential depth T into T/CS.
 *
 * @status Stub. Declarations and the MSL fragment hook exist and compile;
 *   no kernels implemented yet.
 * @issues None
 * @todo Implement this family. Write every entry point against fm_bufref
 *   (buffer + byte offset) from the start — weights live at nonzero offsets
 *   inside an fm_region segment and a kernel binding at offset 0 cannot read
 *   them, so retrofitting is not an option.
 *
 * FUNCTION-CONSTANT INDICES OWNED BY THIS FAMILY: 220-239.
 *   Function constants share ONE index space across the whole assembled
 *   shader library, so each family reserves a block; two families colliding
 *   on an index fails the compile of every kernel in the backend, not just
 *   theirs. Allocated so far: 0-2 probe / 30 Q3_K / 50 Q5_K / 60 Q6_K / 70 Q8_0 / 80 Q4_K /
 *   100-109 rope / 110-119 norm / 120-129 elementwise / 200-219 ssm /
 *   220-239 ssm_chunked / 240-259 moe / 260-279 attn.
 */

#ifndef FOUNDRY_METAL_KERNEL_SSM_CHUNKED_H
#define FOUNDRY_METAL_KERNEL_SSM_CHUNKED_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Entry points go here. */

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_SSM_CHUNKED_H */
