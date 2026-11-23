/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cuda_image.h
 * @brief  Device image of one BF16 payload (modes 1, 2, 6, 7): the block area verbatim, rANS
 *         tables as GPU lookup images, and per-unit predictor records, in one buffer that is
 *         uploaded once and decoded on every use. Pure C (no CUDA): built and tested anywhere.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * The payload is first opened by the codec's own decoder (format checks, the match side stream
 * and its hash), so the image only holds validated values. Image sections start 256-aligned;
 * ingot_cuda_image_params rebases the section offsets to wherever the image lives (a device
 * address, or the host buffer for the CPU reference).
 *
 * Mode 6 (predict) becomes the match layout: unit u has K = J partners, partner j = row
 * map_j(u) of source j with coefficient coef[u][j].
 */
#ifndef FOUNDRY_INGOT_CUDA_IMAGE_H
#define FOUNDRY_INGOT_CUDA_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/container/ingot_format.h"
#include "ingot/core/ingot_rans.h"
#include "ingot/cuda/ingot_cuda_device.h"

#ifdef __cplusplus
extern "C" {
#endif

/** What a source must be bound to (validated at bind). */
typedef struct {
    uint32_t ref;          /* directory entry, entry_count + external ref, or self */
    uint32_t axis;         /* 0 row units ([n, H]), 1 column units ([H, n]) */
    uint64_t units;        /* n; predict: 0 (any row count above min_rows) */
    uint64_t min_rows;     /* predict: rows must exceed the largest mapped row */
    int self;
} ingot_cuda_source_shape;

/** One payload's device image. */
typedef struct {
    ingot_cuda_params params;   /* section addresses are offsets into image until rebased */
    uint8_t *image;
    size_t image_bytes;
    ingot_cuda_source_shape shape[INGOT_CUDA_MAX_SOURCES];
    int serial;                 /* a self source without strips: blocks decode in order */
    int self_source;            /* a self source: the output must be stored (later rows read it) */
    uint64_t table_bytes;       /* lookup images (device memory beyond the payload) */
    uint64_t side_bytes;        /* unit records and partner lists */
} ingot_cuda_image;

/**
 * Build the image of one coded BF16 payload (version 2+ frame).
 *
 * @param e        Directory entry (mode 1, 2, 6 or 7; BF16)
 * @param payload  Payload bytes (frame first)
 * @param size     Payload size
 * @param out      Out: image (ingot_cuda_image_free)
 * @return         0, -INGOT_EUNSUPPORTED (another mode or dtype, or a tensor of 2^32 elements
 *                 or more), -INGOT_EFORMAT / -INGOT_ECORRUPT (payload refused by the codec),
 *                 -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_cuda_image_build(const ingot_entry *e, const uint8_t *payload, uint64_t size,
                           ingot_cuda_image *out);

/**
 * Free (a zeroed image is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_cuda_image_free(ingot_cuda_image *img);

/**
 * Kernel parameters with every section at `base` + its offset (base = the image's device
 * address, or (uintptr_t)img->image for the CPU reference). Sources are left unbound.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_cuda_image_params(const ingot_cuda_image *img, uint64_t base, ingot_cuda_params *out);

/**
 * Bind source s to a BF16 tensor at `address` (rows x cols as stored), checking the shape the
 * payload declares. A self source is bound to the output buffer at launch (no call needed).
 * @return 0 or -INGOT_EFORMAT
 *
 * Thread-safe: yes (caller-owned params)   Reentrant: yes   Signal-safe: yes
 */
int ingot_cuda_bind_source(const ingot_cuda_image *img, ingot_cuda_params *p, uint32_t s,
                           uint64_t address, uint64_t rows, uint64_t cols);

/**
 * Point the self source (if any) at the output buffer.
 *
 * Thread-safe: yes (caller-owned params)   Reentrant: yes   Signal-safe: yes
 */
void ingot_cuda_bind_self(ingot_cuda_params *p, uint64_t out_address);

/**
 * Bytes of one table's lookup image, and the image itself (dst 8-aligned, non-empty table).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_cuda_table_image_bytes(const ingot_rans_table *t);
void ingot_cuda_table_image_write(const ingot_rans_table *t, uint8_t *dst);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_CUDA_IMAGE_H */
