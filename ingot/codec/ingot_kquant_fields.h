/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_kquant_fields.h
 * @brief  Bijective split of one ggml K-quant block (Q3_K, Q4_K, Q5_K, Q6_K; 256 weights) into
 *         its fields: fp16 d (and dmin), sub-block scales (and mins), and one code per weight.
 *         Every bit of the block lands in exactly one field, so pack(unpack(b)) == b for any
 *         bytes b.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Layouts (ggml block_q3_K .. block_q6_K, QK_K = 256), byte offsets:
 *   Q3_K (110): hmask[32] @0, qs[64] @32, scales[12] @96 (16 x 6 bit), d @108
 *   Q4_K (144): d @0, dmin @2, scales[12] @4 (8 scales + 8 mins, 6 bit), qs[128] @16
 *   Q5_K (176): d @0, dmin @2, scales[12] @4, qh[32] @16, qs[128] @48
 *   Q6_K (210): ql[128] @0, qh[64] @128, scales[16] @192 (int8), d @208
 * Codes are in weight order (code[i] is weight i of the block), so sub-block k holds the weights
 * [k * 256 / subblocks, (k + 1) * 256 / subblocks). Code values are the unsigned stored codes:
 * Q3_K low2 + 4 * hmask bit (0..7), Q4_K 0..15, Q5_K 0..31, Q6_K 0..63. Q6_K scales are stored
 * offset-binary (int8 + 128), so their order matches the value order.
 */
#ifndef FOUNDRY_INGOT_KQUANT_FIELDS_H
#define FOUNDRY_INGOT_KQUANT_FIELDS_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/core/ingot_ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { INGOT_KQ_WEIGHTS = 256, INGOT_KQ_MAX_SUBBLOCKS = 16 };

/** Field geometry of one K-quant type. */
typedef struct {
    uint32_t ggml_type;
    uint32_t block_bytes;
    uint32_t subblocks;   /* 16 (Q3_K, Q6_K) or 8 (Q4_K, Q5_K) */
    uint32_t code_bits;   /* 3, 4, 5 or 6 */
    uint32_t scale_bits;  /* 6, or 8 for Q6_K */
    int has_min;          /* Q4_K, Q5_K: dmin and 6-bit mins */
} ingot_kq_layout;

/** The fields of one block. Unused fields (dmin, min[] without has_min) are 0. */
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scale[INGOT_KQ_MAX_SUBBLOCKS];
    uint8_t min[INGOT_KQ_MAX_SUBBLOCKS];
    uint8_t code[INGOT_KQ_WEIGHTS];
} ingot_kq_fields;

/**
 * Layout of ggml type Q3_K (11), Q4_K (12), Q5_K (13) or Q6_K (14), else NULL. Inline so a coder
 * specialized on a constant type folds the geometry into its loops.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
static inline const ingot_kq_layout *ingot_kq_layout_of(uint32_t ggml_type) {
    static const ingot_kq_layout LAYOUTS[4] = {
        {INGOT_GGML_Q3_K, 110, 16, 3, 6, 0},
        {INGOT_GGML_Q4_K, 144, 8, 4, 6, 1},
        {INGOT_GGML_Q5_K, 176, 8, 5, 6, 1},
        {INGOT_GGML_Q6_K, 210, 16, 6, 8, 0},
    };
    if (ggml_type < INGOT_GGML_Q3_K || ggml_type > INGOT_GGML_Q6_K)
        return NULL;
    return &LAYOUTS[ggml_type - INGOT_GGML_Q3_K];
}

/**
 * Split one block (l->block_bytes bytes) into its fields.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_kq_unpack(const ingot_kq_layout *l, const uint8_t *block, ingot_kq_fields *f);

/**
 * Join fields into one block. Fields must be in range (code < 2^code_bits, 6-bit scales and
 * mins below 64); out-of-range bits are dropped.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_kq_pack(const ingot_kq_layout *l, const ingot_kq_fields *f, uint8_t *block);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_KQUANT_FIELDS_H */
