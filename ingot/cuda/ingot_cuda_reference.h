/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cuda_reference.h
 * @brief  CPU reference of the GPU decode kernels: the same device image, the same integer math
 *         (ingot_cuda_device.h) and the same fp32 operation order for the matvec, so a GPU run
 *         can be compared bit for bit, and the image can be tested on machines without CUDA.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Every address in the params is a host pointer (ingot_cuda_image_params with the host image as
 * base; sources, out, x, y, partials and status in host memory). Symbols are decoded in stream
 * order, lane by lane, which is the order the kernels' ballots reproduce. Matvec: each of the G
 * lanes of a block (G = 32 plain and delta, 16 match) sums its weights with fmaf in step order;
 * at each row end the lanes are added by the shuffle tree (offsets G/2 .. 1).
 */
#ifndef FOUNDRY_INGOT_CUDA_REFERENCE_H
#define FOUNDRY_INGOT_CUDA_REFERENCE_H

#include "ingot/cuda/ingot_cuda_device.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Decode blocks [block_first, block_first + launch_blocks) as the kernels do, writing status[b]
 * (INGOT_CUDA_BAD_* bits) and, per flags, out and y / partials. Matvec needs cols % G == 0.
 * Blocks run in order (self sources read earlier rows of out).
 *
 * Thread-safe: yes (distinct outputs)   Reentrant: yes   Signal-safe: no
 */
void ingot_cuda_reference_run(const ingot_cuda_params *p);

/**
 * Weights a block decodes per step: 32 (plain, delta) or 16 (match). The matvec needs cols to
 * be a multiple of it.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint32_t ingot_cuda_step_weights(uint32_t kind);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_CUDA_REFERENCE_H */
