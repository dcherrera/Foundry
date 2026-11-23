/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_kquant.c
 * @brief  K-quant type coders (Q3_K, Q4_K, Q5_K, Q6_K; see ingot_qtype.h). Blocks are split into
 *         fields by ingot_kquant_fields.h and coded one block after another, in row order.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Per block (with a base: an identical-block flag first, context = the previous flag; an
 * identical block is copied from the base and costs nothing more):
 * - d (and dmin): a step of the ordered fp16 pattern from the previous coded block's value in
 *   the coded block; context: first block of a row or not (d and dmin apart).
 * - Scales: context = the previous scale in the block folded to 4 buckets (6-bit) or 8 (Q6_K
 *   int8), the first scale its own.
 * - Mins (Q4_K, Q5_K): context = the same sub-block's scale / 8.
 * - Codes, sub-block by sub-block. Q3_K: context = scale / 2. Q6_K: scale / 16. Q4_K / Q5_K:
 *   the zero point dmin * min / (d * scale) (where weight 0 falls, in 1/16ths of the code range,
 *   computed exactly in integers) x scale / 16.
 * Measured on Qwen3-1.7B Q4_K_M / Q3_K_M: the zero point beats (scale, min) by 0.04 bit/weight;
 * finer scale and min contexts lose more in table bytes (4 per used entry, per tensor) than they
 * gain.
 * Decoding builds the fields in a local struct (codes: one ingot_fc_syms call per sub-block) and
 * packs them into q->dst once per block: byte stores through q->dst would alias the coder state
 * and slow every symbol down.
 */
#include <string.h>

#include "ingot/codec/ingot_kquant_fields.h"
#include "ingot/codec/ingot_qtype.h"
#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_ggml.h"

enum {
    FLAG_CONTEXTS = 2,
    HALF_CONTEXTS = 2, /* fp16 d / dmin: first block of the row, other */
    SCALE6_SHIFT = 4,
    SCALE8_SHIFT = 5,
    MIN_SHIFT = 3,
    Q3_CODE_SHIFT = 1,
    Q6_CODE_SHIFT = 4,
    ZERO_BUCKETS = 16,
    ZERO_SCALE_SHIFT = 4,
};

/* First context of each field for one type and mode. */
typedef struct {
    uint32_t flag, d, dmin, scale, min, code;
} kq_contexts;

static uint32_t scale_contexts(const ingot_kq_layout *l) {
    return ((1u << l->scale_bits) >> (l->scale_bits == 8 ? SCALE8_SHIFT : SCALE6_SHIFT)) + 1;
}

static uint32_t code_contexts(const ingot_kq_layout *l) {
    if (l->has_min)
        return ZERO_BUCKETS * (64 >> ZERO_SCALE_SHIFT);
    return l->scale_bits == 8 ? 256 >> Q6_CODE_SHIFT : 64 >> Q3_CODE_SHIFT;
}

/* Declare the contexts (m may be NULL: offsets only). */
static int kq_declare(const ingot_kq_layout *l, int with_base, ingot_fc_model *m,
                      kq_contexts *x) {
    const struct {
        uint32_t *first, count, alphabet;
        int used;
    } groups[] = {
        {&x->flag, FLAG_CONTEXTS, 2, with_base},
        {&x->d, HALF_CONTEXTS, INGOT_STEP_ALPHABET, 1},
        {&x->dmin, HALF_CONTEXTS, INGOT_STEP_ALPHABET, l->has_min},
        {&x->scale, scale_contexts(l), 1u << l->scale_bits, 1},
        {&x->min, 64 >> MIN_SHIFT, 64, l->has_min},
        {&x->code, code_contexts(l), 1u << l->code_bits, 1},
    };
    uint32_t next = 0;
    *x = (kq_contexts){0};
    for (size_t g = 0; g < sizeof(groups) / sizeof(groups[0]); g++) {
        if (!groups[g].used)
            continue;
        *groups[g].first = next;
        next += groups[g].count;
        int rc = m ? ingot_fc_model_add(m, groups[g].count, groups[g].alphabet, NULL) : 0;
        if (rc)
            return rc;
    }
    return 0;
}

/* fp16 magnitude as mantissa * 2^*e (sign ignored; Inf / NaN patterns just count as large). */
INGOT_FC_INLINE uint64_t half_split(uint16_t h, int32_t *e) {
    uint32_t ex = (h >> 10) & 31u, m = h & 1023u;
    *e = ex ? (int32_t)ex - 25 : -24;
    return ex ? m | 1024u : m;
}

/* Zero point dmin * min / (d * scale) in units of 2^(code_bits - 4) codes, clamped. */
INGOT_FC_INLINE uint32_t zero_bucket(const ingot_kq_layout *l, const ingot_kq_fields *f,
                                     uint32_t scale, uint32_t min) {
    int32_t ed = 0, em = 0;
    uint64_t num = half_split(f->dmin, &em) * min;
    uint64_t den = (half_split(f->d, &ed) * scale) << (l->code_bits - 4);
    int32_t sh = em - ed;
    if (!den || sh > 30)
        return ZERO_BUCKETS - 1;
    if (!num || sh < -30)
        return 0;
    if (sh >= 0)
        num <<= sh;
    else
        den <<= -sh;
    uint64_t z = num / den;
    return z >= ZERO_BUCKETS ? ZERO_BUCKETS - 1 : (uint32_t)z;
}

INGOT_FC_INLINE uint32_t code_context(const ingot_kq_layout *l, const ingot_kq_fields *f,
                                     uint32_t k) {
    uint32_t scale = f->scale[k];
    if (l->has_min)
        return zero_bucket(l, f, scale, f->min[k]) * (64 >> ZERO_SCALE_SHIFT) +
               (scale >> ZERO_SCALE_SHIFT);
    return l->scale_bits == 8 ? scale >> Q6_CODE_SHIFT : scale >> Q3_CODE_SHIFT;
}
/* Per coded block state of one code() call. */
typedef struct {
    const ingot_kq_layout *layout;
    kq_contexts ctx;
    int32_t ref_d;    /* ordered fp16 d of the previous coded block */
    int32_t ref_dmin;
} kq_state;

/* An fp16 value as a step of its ordered pattern from *ref. */
static uint16_t code_half(ingot_fcoder *c, uint32_t ctx, int32_t *ref, uint16_t v) {
    int32_t ord = *ref + ingot_fc_step(c, ctx, ingot_bf16_ordered(v) - *ref);
    if (ord < -32768 || ord > 32767) {
        c->err = c->err ? c->err : -INGOT_ECORRUPT;
        return 0;
    }
    *ref = ord;
    return ingot_bf16_from_ordered(ord); /* fp16 has the same sign-magnitude ordering */
}

/* Scales, each in the context of the previous one (folded); the first has its own context. */
INGOT_FC_INLINE void code_scales(ingot_fcoder *c, const ingot_kq_layout *l, uint32_t ctx0,
                                 ingot_kq_fields *f) {
    unsigned shift = l->scale_bits == 8 ? SCALE8_SHIFT : SCALE6_SHIFT;
    uint32_t ctx = ctx0 + ((1u << l->scale_bits) >> shift);
    for (uint32_t k = 0; k < l->subblocks; k++) {
        f->scale[k] = (uint8_t)ingot_fc_sym(c, ctx, f->scale[k]);
        ctx = ctx0 + (f->scale[k] >> shift);
    }
}

INGOT_FC_INLINE void code_fields(ingot_fcoder *c, kq_state *s, ingot_kq_fields *f, int first) {
    const ingot_kq_layout *l = s->layout;
    f->d = code_half(c, s->ctx.d + !first, &s->ref_d, f->d);
    if (l->has_min)
        f->dmin = code_half(c, s->ctx.dmin + !first, &s->ref_dmin, f->dmin);
    code_scales(c, l, s->ctx.scale, f);
    for (uint32_t k = 0; l->has_min && k < l->subblocks; k++)
        f->min[k] = (uint8_t)ingot_fc_sym(c, s->ctx.min + (f->scale[k] >> MIN_SHIFT), f->min[k]);
    /* one sub-block of codes per ingot_fc_syms call (one context); decoded into a local array so
     * no byte store into f (which escapes to unpack / pack) sits in the symbol loop */
    uint32_t per = INGOT_KQ_WEIGHTS / l->subblocks, sym[INGOT_KQ_WEIGHTS / 8];
    for (uint32_t k = 0; k < l->subblocks; k++) {
        uint8_t *code = f->code + k * per;
        for (uint32_t i = 0; i < per; i++)
            sym[i] = code[i];
        ingot_fc_syms(c, s->ctx.code + code_context(l, f, k), sym, per);
        for (uint32_t i = 0; i < per; i++)
            code[i] = (uint8_t)sym[i];
    }
}

INGOT_FC_INLINE void kq_code(const ingot_kq_layout *l, ingot_fcoder *c, const ingot_qrows *q) {
    kq_state s = {.layout = l};
    if (kq_declare(l, q->base != NULL, NULL, &s.ctx)) {
        c->err = -INGOT_EINVAL;
        return;
    }
    uint64_t blocks = q->cols / INGOT_KQ_WEIGHTS;
    uint32_t bb = l->block_bytes, prev_flag = 0;
    ingot_kq_fields f;
    memset(&f, 0, sizeof(f));
    for (uint64_t r = 0; r < q->rows && !c->err; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            uint64_t at = (r * blocks + b) * bb;
            if (q->base) {
                uint32_t same = ingot_fc_encoding(c) && !memcmp(q->src + at, q->base + at, bb);
                same = ingot_fc_sym(c, s.ctx.flag + prev_flag, same);
                prev_flag = same;
                if (same) {
                    if (q->dst)
                        memcpy(q->dst + at, q->base + at, bb);
                    continue;
                }
            }
            if (ingot_fc_encoding(c))
                ingot_kq_unpack(l, q->src + at, &f);
            code_fields(c, &s, &f, b == 0);
            if (q->dst)
                ingot_kq_pack(l, &f, q->dst + at);
        }
    }
}

static size_t kq_scratch_bytes(uint64_t cols) {
    (void)cols;
    return 1;
}

#define KQ_TYPE(name, ggml)                                                                   \
    static int name##_model(ingot_fc_model *m, int with_base) {                              \
        kq_contexts x;                                                                        \
        return kq_declare(ingot_kq_layout_of(ggml), with_base, m, &x);                       \
    }                                                                                         \
    static void name##_code(ingot_fcoder *c, const ingot_qrows *q) {                         \
        kq_code(ingot_kq_layout_of(ggml), c, q);                                              \
    }                                                                                         \
    static const ingot_qtype name##_TYPE = {kq_scratch_bytes, name##_model, name##_code};

KQ_TYPE(Q3K, INGOT_GGML_Q3_K)
KQ_TYPE(Q4K, INGOT_GGML_Q4_K)
KQ_TYPE(Q5K, INGOT_GGML_Q5_K)
KQ_TYPE(Q6K, INGOT_GGML_Q6_K)

const ingot_qtype *ingot_kquant_qtype(uint32_t ggml_type) {
    switch (ggml_type) {
    case INGOT_GGML_Q3_K:
        return &Q3K_TYPE;
    case INGOT_GGML_Q4_K:
        return &Q4K_TYPE;
    case INGOT_GGML_Q5_K:
        return &Q5K_TYPE;
    case INGOT_GGML_Q6_K:
        return &Q6K_TYPE;
    default:
        return NULL;
    }
}
