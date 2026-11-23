/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_kquant_fields.c
 * @brief  K-quant block <-> fields (see header). Bit placement follows ggml's dequantize_row_q3_K
 *         .. dequantize_row_q6_K and get_scale_min_k4.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_kquant_fields.h"

#include <string.h>

#include "ingot/core/ingot_ggml.h"

static inline uint16_t load16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static inline void store16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

/* Q3_K: 16 scales in 12 bytes: low nibbles in S[0..7], high 2-bit pairs in S[8..11]. */
static void q3_unpack(const uint8_t *blk, ingot_kq_fields *f) {
    const uint8_t *hm = blk, *qs = blk + 32, *s = blk + 96;
    for (int i = 0; i < 4; i++) {
        uint8_t hi = s[8 + i];
        f->scale[i] = (uint8_t)((s[i] & 15) | ((hi & 3) << 4));
        f->scale[i + 4] = (uint8_t)((s[i + 4] & 15) | (((hi >> 2) & 3) << 4));
        f->scale[i + 8] = (uint8_t)((s[i] >> 4) | (((hi >> 4) & 3) << 4));
        f->scale[i + 12] = (uint8_t)((s[i + 4] >> 4) | (((hi >> 6) & 3) << 4));
    }
    for (int h = 0; h < 2; h++)
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 32; l++) {
                uint32_t low2 = (qs[32 * h + l] >> (2 * j)) & 3u;
                uint32_t hbit = (hm[l] >> (4 * h + j)) & 1u;
                f->code[128 * h + 32 * j + l] = (uint8_t)(low2 | (hbit << 2));
            }
    f->d = load16(blk + 108);
}

static void q3_pack(const ingot_kq_fields *f, uint8_t *blk) {
    uint8_t *hm = blk, *qs = blk + 32, *s = blk + 96;
    const uint8_t *sc = f->scale;
    for (int i = 0; i < 4; i++) {
        s[i] = (uint8_t)((sc[i] & 15) | (sc[i + 8] << 4));
        s[i + 4] = (uint8_t)((sc[i + 4] & 15) | (sc[i + 12] << 4));
        s[8 + i] = (uint8_t)(((sc[i] >> 4) & 3) | (((sc[i + 4] >> 4) & 3) << 2) |
                             (((sc[i + 8] >> 4) & 3) << 4) | (((sc[i + 12] >> 4) & 3) << 6));
    }
    for (int l = 0; l < 32; l++) {
        uint32_t hbits = 0;
        for (int h = 0; h < 2; h++) {
            uint32_t low = 0;
            for (int j = 0; j < 4; j++) {
                uint32_t c = f->code[128 * h + 32 * j + l];
                low |= (c & 3u) << (2 * j);
                hbits |= ((c >> 2) & 1u) << (4 * h + j);
            }
            qs[32 * h + l] = (uint8_t)low;
        }
        hm[l] = (uint8_t)hbits;
    }
    store16(blk + 108, f->d);
}

/* Q4_K / Q5_K: 8 scales + 8 mins in 12 bytes (get_scale_min_k4). */
static void scale_min_unpack(const uint8_t *s, ingot_kq_fields *f) {
    for (int k = 0; k < 4; k++) {
        f->scale[k] = s[k] & 63;
        f->min[k] = s[k + 4] & 63;
        f->scale[k + 4] = (uint8_t)((s[k + 8] & 15) | ((s[k] >> 6) << 4));
        f->min[k + 4] = (uint8_t)((s[k + 8] >> 4) | ((s[k + 4] >> 6) << 4));
    }
}

static void scale_min_pack(const ingot_kq_fields *f, uint8_t *s) {
    for (int k = 0; k < 4; k++) {
        s[k] = (uint8_t)((f->scale[k] & 63) | ((f->scale[k + 4] >> 4) << 6));
        s[k + 4] = (uint8_t)((f->min[k] & 63) | ((f->min[k + 4] >> 4) << 6));
        s[k + 8] = (uint8_t)((f->scale[k + 4] & 15) | (f->min[k + 4] << 4));
    }
}

/* Q4_K (five = 0) and Q5_K (five = 1): 4 chunks of 64 weights, low nibbles then high. */
static void q45_unpack(const uint8_t *blk, ingot_kq_fields *f, int five) {
    const uint8_t *qh = blk + 16, *qs = blk + (five ? 48 : 16);
    f->d = load16(blk);
    f->dmin = load16(blk + 2);
    scale_min_unpack(blk + 4, f);
    for (int p = 0; p < 4; p++)
        for (int l = 0; l < 32; l++) {
            uint32_t lo = qs[32 * p + l] & 15u, hi = qs[32 * p + l] >> 4;
            if (five) {
                lo |= ((qh[l] >> (2 * p)) & 1u) << 4;
                hi |= ((qh[l] >> (2 * p + 1)) & 1u) << 4;
            }
            f->code[64 * p + l] = (uint8_t)lo;
            f->code[64 * p + 32 + l] = (uint8_t)hi;
        }
}

static void q45_pack(const ingot_kq_fields *f, uint8_t *blk, int five) {
    uint8_t *qh = blk + 16, *qs = blk + (five ? 48 : 16);
    store16(blk, f->d);
    store16(blk + 2, f->dmin);
    scale_min_pack(f, blk + 4);
    for (int l = 0; l < 32; l++) {
        uint32_t hbits = 0;
        for (int p = 0; p < 4; p++) {
            uint32_t lo = f->code[64 * p + l], hi = f->code[64 * p + 32 + l];
            qs[32 * p + l] = (uint8_t)((lo & 15u) | ((hi & 15u) << 4));
            hbits |= (((lo >> 4) & 1u) << (2 * p)) | (((hi >> 4) & 1u) << (2 * p + 1));
        }
        if (five)
            qh[l] = (uint8_t)hbits;
    }
}

/* Q6_K: per half of 128 weights, ql[64] (nibbles) and qh[32] (2-bit pairs). */
static void q6_unpack(const uint8_t *blk, ingot_kq_fields *f) {
    for (int h = 0; h < 2; h++) {
        const uint8_t *ql = blk + 64 * h, *qh = blk + 128 + 32 * h;
        uint8_t *c = f->code + 128 * h;
        for (int l = 0; l < 32; l++) {
            c[l] = (uint8_t)((ql[l] & 15) | ((qh[l] & 3) << 4));
            c[l + 32] = (uint8_t)((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4));
            c[l + 64] = (uint8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4));
            c[l + 96] = (uint8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4));
        }
    }
    for (int k = 0; k < 16; k++)
        f->scale[k] = (uint8_t)(blk[192 + k] ^ 0x80u);
    f->d = load16(blk + 208);
}

static void q6_pack(const ingot_kq_fields *f, uint8_t *blk) {
    for (int h = 0; h < 2; h++) {
        uint8_t *ql = blk + 64 * h, *qh = blk + 128 + 32 * h;
        const uint8_t *c = f->code + 128 * h;
        for (int l = 0; l < 32; l++) {
            uint32_t c1 = c[l], c2 = c[l + 32], c3 = c[l + 64], c4 = c[l + 96];
            ql[l] = (uint8_t)((c1 & 15u) | ((c3 & 15u) << 4));
            ql[l + 32] = (uint8_t)((c2 & 15u) | ((c4 & 15u) << 4));
            qh[l] = (uint8_t)(((c1 >> 4) & 3u) | (((c2 >> 4) & 3u) << 2) |
                              (((c3 >> 4) & 3u) << 4) | (((c4 >> 4) & 3u) << 6));
        }
    }
    for (int k = 0; k < 16; k++)
        blk[192 + k] = (uint8_t)(f->scale[k] ^ 0x80u);
    store16(blk + 208, f->d);
}

void ingot_kq_unpack(const ingot_kq_layout *l, const uint8_t *block, ingot_kq_fields *f) {
    memset(f->scale, 0, sizeof(f->scale));
    memset(f->min, 0, sizeof(f->min));
    f->dmin = 0;
    switch (l->ggml_type) {
    case INGOT_GGML_Q3_K:
        q3_unpack(block, f);
        break;
    case INGOT_GGML_Q4_K:
    case INGOT_GGML_Q5_K:
        q45_unpack(block, f, l->ggml_type == INGOT_GGML_Q5_K);
        break;
    default:
        q6_unpack(block, f);
        break;
    }
}

void ingot_kq_pack(const ingot_kq_layout *l, const ingot_kq_fields *f, uint8_t *block) {
    switch (l->ggml_type) {
    case INGOT_GGML_Q3_K:
        q3_pack(f, block);
        break;
    case INGOT_GGML_Q4_K:
    case INGOT_GGML_Q5_K:
        q45_pack(f, block, l->ggml_type == INGOT_GGML_Q5_K);
        break;
    default:
        q6_pack(f, block);
        break;
    }
}
