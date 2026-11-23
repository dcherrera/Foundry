/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_qtype.h
 * @brief  Contract between the quantized-tensor driver (ingot_qtensor.c) and one type coder
 *         (Q8_0, K-quants, generic block delta). A type coder declares its contexts and codes the
 *         rows of one block through an ingot_fcoder, in one function used for counting, encoding
 *         and decoding.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_QTYPE_H
#define FOUNDRY_INGOT_QTYPE_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/codec/ingot_fcoder.h"

#ifdef __cplusplus
extern "C" {
#endif

/** The rows of one coded block. */
typedef struct {
    const uint8_t *src;   /* target bytes of the rows (count / encode), else NULL */
    uint8_t *dst;         /* output bytes of the rows (decode), else NULL */
    const uint8_t *base;  /* base bytes of the same rows (delta modes), else NULL */
    uint64_t rows;        /* rows in this block */
    uint64_t cols;        /* weights per row */
    uint32_t block_weights; /* ggml block geometry */
    uint32_t block_bytes;
    void *scratch;        /* scratch_bytes(cols) bytes, owned by the caller */
} ingot_qrows;

/** One type coder. */
typedef struct {
    /** Scratch the code function needs for rows of `cols` weights. */
    size_t (*scratch_bytes)(uint64_t cols);
    /** Declare the contexts (with_base: also the identical-block flags). */
    int (*model)(ingot_fc_model *m, int with_base);
    /** Code the rows (errors go to c->err). With q->base, a block equal to its base block is
     * flagged and costs nothing more; decoding copies it from the base. */
    void (*code)(ingot_fcoder *c, const ingot_qrows *q);
} ingot_qtype;

/**
 * Q8_0 coder (ingot_q8.c). Scales coded as the BF16 amax they came from (provenance), the
 * position + sign of the block's first |code| = 127, and the other codes under a scale context.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const ingot_qtype *ingot_q8_qtype(void);

/**
 * K-quant coder for ggml type Q3_K, Q4_K, Q5_K or Q6_K (ingot_kquant.c), else NULL.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const ingot_qtype *ingot_kquant_qtype(uint32_t ggml_type);

/**
 * Generic block delta (any block type): identical-block flags, differing blocks raw.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const ingot_qtype *ingot_blockdelta_qtype(void);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_QTYPE_H */
