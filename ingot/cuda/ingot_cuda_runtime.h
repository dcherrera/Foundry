/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cuda_runtime.h
 * @brief  Host side of the GPU decode through the CUDA driver API (link -lcuda): load the
 *         kernels' PTX, upload a payload's device image, launch decode / matvec.
 *
 * @status Active
 * @issues None
 * @todo   Pantheon integration goes through foundry's fc_cuda driver table instead of -lcuda.
 *
 * One runtime per die (its own context). Launches run on the context's default stream and
 * return after the work finished (timing included). A payload with a self source is decoded one
 * block per launch, in order, with STORE forced on (later rows read earlier ones).
 */
#ifndef FOUNDRY_INGOT_CUDA_RUNTIME_H
#define FOUNDRY_INGOT_CUDA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/cuda/ingot_cuda_image.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ingot_cuda_runtime ingot_cuda_runtime;

/**
 * Create a context on `device` and load the kernels from a PTX file.
 * @return 0, -INGOT_EIO (driver, device or PTX failure; message on stderr) or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_cuda_runtime_open(int device, const char *ptx_path, ingot_cuda_runtime **out);

/**
 * Destroy the context (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_cuda_runtime_close(ingot_cuda_runtime *rt);

/**
 * Device memory: allocate (zeroed), free, copy in / out. Addresses are CUdeviceptr values.
 * @return 0 or -INGOT_EIO
 *
 * Thread-safe: no (one caller per runtime)   Reentrant: yes   Signal-safe: no
 */
int ingot_cuda_alloc(ingot_cuda_runtime *rt, size_t bytes, uint64_t *address);
void ingot_cuda_free(ingot_cuda_runtime *rt, uint64_t address);
int ingot_cuda_upload(ingot_cuda_runtime *rt, uint64_t address, const void *src, size_t bytes);
int ingot_cuda_download(ingot_cuda_runtime *rt, void *dst, uint64_t address, size_t bytes);

/**
 * Upload a device image; *address = its device copy (ingot_cuda_free). Then
 * ingot_cuda_image_params(img, *address, &p) gives the kernel parameters.
 * @return 0 or -INGOT_EIO
 *
 * Thread-safe: no (one caller per runtime)   Reentrant: yes   Signal-safe: no
 */
int ingot_cuda_image_upload(ingot_cuda_runtime *rt, const ingot_cuda_image *img,
                            uint64_t *address);

/**
 * Decode (and / or matvec, per p->flags) blocks [block_first, block_first + launch_blocks),
 * then add split rows into y when MATVEC covers every block. *elapsed_ms (optional) is the
 * kernels' time from events.
 * @return 0, -INGOT_EUNSUPPORTED (matvec with cols not a multiple of the step's weights; a
 *         self source with rows under 16 weights) or -INGOT_EIO
 *
 * Thread-safe: no (one caller per runtime)   Reentrant: yes   Signal-safe: no
 */
int ingot_cuda_launch(ingot_cuda_runtime *rt, const ingot_cuda_image *img,
                      const ingot_cuda_params *p, float *elapsed_ms);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_CUDA_RUNTIME_H */
