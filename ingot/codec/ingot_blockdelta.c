/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_blockdelta.c
 * @brief  Generic block delta type coder (ingot_qtype.h): per unit of q->block_bytes, a flag
 *         "equal to the base unit" (context = the previous flag); differing units go raw.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include <string.h>

#include "ingot/codec/ingot_qtype.h"

static size_t blockdelta_scratch_bytes(uint64_t cols) {
    (void)cols;
    return 1;
}

static int blockdelta_model(ingot_fc_model *m, int with_base) {
    return with_base ? ingot_fc_model_add(m, 2, 2, NULL) : -INGOT_EINVAL;
}

static void blockdelta_code(ingot_fcoder *c, const ingot_qrows *q) {
    uint64_t units = q->rows * (q->cols / q->block_weights);
    uint32_t prev = 0;
    for (uint64_t u = 0; u < units && !c->err; u++) {
        uint64_t at = u * q->block_bytes;
        uint32_t same = ingot_fc_encoding(c) && !memcmp(q->src + at, q->base + at,
                                                        q->block_bytes);
        same = ingot_fc_sym(c, prev, same);
        prev = same;
        for (uint32_t k = 0; k < q->block_bytes; k++) {
            uint32_t v = same ? q->base[at + k]
                              : ingot_fc_raw(c, 8, ingot_fc_encoding(c) ? q->src[at + k] : 0);
            if (q->dst)
                q->dst[at + k] = (uint8_t)v;
        }
    }
}

static const ingot_qtype BLOCKDELTA_TYPE = {blockdelta_scratch_bytes, blockdelta_model,
                                            blockdelta_code};

const ingot_qtype *ingot_blockdelta_qtype(void) { return &BLOCKDELTA_TYPE; }
