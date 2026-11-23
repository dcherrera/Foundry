/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cuda_baseline.h
 * @brief  Matvec time of foundry's dense kernels on one die (fc_cuda), for comparing the GPU
 *         decode against: BF16 rows (fc_matvec_bf16_ctx) and Q8_0 rows (fc_matvec_q80_ctx, the
 *         tensor quantized as ggml's Q8_0 and repacked).
 *
 * Its own context on the die (fc_cuda pushes and pops it per call). Create it before the decode
 * runtime's context so that one stays current. Time = mean of 10 launches after a warm-up,
 * wall clock around a stream sync.
 */
#ifndef FOUNDRY_INGOT_CUDA_BASELINE_H
#define FOUNDRY_INGOT_CUDA_BASELINE_H

#include <stdint.h>

typedef struct ingot_cuda_baseline ingot_cuda_baseline;

/** @return 0, or -1 when the driver, the die or the BF16 kernel is missing. */
int ingot_cuda_baseline_open(int device, ingot_cuda_baseline **out);
void ingot_cuda_baseline_close(ingot_cuda_baseline *b);

/**
 * Time both kernels on a rows x cols BF16 tensor (host memory). Q8_0 needs cols % 32 == 0
 * (*q80_ms = -1 otherwise). @return 0 or -1 (allocation or launch failure)
 */
int ingot_cuda_baseline_time(ingot_cuda_baseline *b, const uint16_t *w, uint64_t rows,
                             uint64_t cols, double *bf16_ms, double *q80_ms);

#endif /* FOUNDRY_INGOT_CUDA_BASELINE_H */
