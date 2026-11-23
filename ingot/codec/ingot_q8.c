/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_q8.c
 * @brief  Q8_0 type coder and scale provenance (see ingot_q8.h, ingot_qtype.h).
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Per row of a coded block: first every block's scale (with a base: an identical-block flag
 * first, context = the previous flag), then every coded block's codes.
 * - Scale: the BF16 amax the fp16 scale came from, as a step from the previous coded block's amax
 *   (context: first block of the row or not); for a scale no BF16 amax produces, escape symbol
 *   119 and a step of the fp16 bit pattern from the previous coded block's scale.
 * - Codes: the position of the first |code| = 127 (32 = none) and its sign bit; the other 31
 *   codes in one of 8 contexts = the scale's distance from the row's median scale in half-binade
 *   steps (fp16 bit patterns: 512 per half binade), clamped to [-4, 3].
 */
#include "ingot_q8.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/codec/ingot_qtype.h"
#include "ingot/core/ingot_bf16.h"

enum {
    FLAG_CONTEXTS = 2,
    SCALE_CONTEXTS = 2,
    SCALE_ESCAPE = INGOT_STEP_ALPHABET,
    SCALE_ALPHABET = INGOT_STEP_ALPHABET + 1,
    POS_ALPHABET = INGOT_Q8_BLOCK + 1,
    CODE_CONTEXTS = 8,
    CODE_ALPHABET = 256,
    BF16_FINITE = 0x7F80, /* positive finite BF16 patterns are [0, 0x7F80) */
    NO_AMAX = 0xFFFF,
    HALF_BINADE = 512,
};

static int bit_length(uint64_t v) {
    int n = 0;
    while (v) {
        n++;
        v >>= 1;
    }
    return n;
}

/* Round num/den * 2^e2 (num > 0) to a p-bit significand with leading-bit exponent >= emin
 * (gradual underflow), to nearest even. Result: *m * 2^*lsb. */
static void round_ratio(uint64_t num, uint64_t den, int e2, int p, int emin, uint64_t *m,
                        int *lsb) {
    int s = (p + 2) - (bit_length(num) - bit_length(den));
    uint64_t q = 0, r = 0;
    for (;;) {
        uint64_t n = s >= 0 ? num << s : num, d = s >= 0 ? den : den << -s;
        q = n / d;
        r = n % d;
        int len = bit_length(q);
        if (len == p + 2)
            break;
        s += len < p + 2 ? 1 : -1;
    }
    int lead = e2 - s + p + 1;
    *lsb = (lead < emin ? emin : lead) - (p - 1);
    int shift = *lsb - (e2 - s);
    if (shift >= 64) {
        *m = 0;
        return;
    }
    uint64_t mant = q >> shift, rem = q & ((1ull << shift) - 1), half = 1ull << (shift - 1);
    if (rem > half || (rem == half && (r || (mant & 1))))
        mant++;
    if (mant == 1ull << p) {
        mant >>= 1;
        (*lsb)++;
    }
    *m = mant;
}

static uint16_t compute_scale(uint16_t a) {
    uint32_t e = (a >> 7) & 0xFFu, mant = a & 0x7Fu;
    uint64_t num = e ? 128u + mant : mant;
    if (!num)
        return 0;
    uint64_t m1 = 0, m2 = 0;
    int l1 = 0, l2 = 0;
    round_ratio(num, 127, e ? (int)e - 134 : -133, 24, -126, &m1, &l1); /* f32 */
    if (!m1)
        return 0;
    round_ratio(m1, 1, l1, 11, -14, &m2, &l2); /* fp16 */
    if (!m2)
        return 0;
    int lead = l2 + bit_length(m2) - 1;
    if (lead > 15)
        return 0x7C00;
    if (lead < -14)
        return (uint16_t)m2;
    return (uint16_t)(((uint32_t)(lead + 15) << 10) | (uint32_t)(m2 - 1024));
}

static uint16_t SCALE_OF[BF16_FINITE];
static uint16_t AMAX_OF[65536];
static pthread_once_t TABLES_ONCE = PTHREAD_ONCE_INIT;

static void build_tables(void) {
    memset(AMAX_OF, 0xFF, sizeof(AMAX_OF));
    for (uint32_t a = 0; a < BF16_FINITE; a++) {
        SCALE_OF[a] = compute_scale((uint16_t)a);
        if (AMAX_OF[SCALE_OF[a]] == NO_AMAX)
            AMAX_OF[SCALE_OF[a]] = (uint16_t)a;
    }
}

uint16_t ingot_q8_scale_of_amax(uint16_t amax_bf16) {
    pthread_once(&TABLES_ONCE, build_tables);
    return amax_bf16 < BF16_FINITE ? SCALE_OF[amax_bf16] : 0x7C00;
}

int ingot_q8_amax_of_scale(uint16_t d16, uint16_t *amax) {
    pthread_once(&TABLES_ONCE, build_tables);
    if (AMAX_OF[d16] == NO_AMAX)
        return -1;
    *amax = AMAX_OF[d16];
    return 0;
}

static size_t q8_scratch_bytes(uint64_t cols) {
    return (size_t)(cols / INGOT_Q8_BLOCK) * 5 + 16;
}

static int q8_model(ingot_fc_model *m, int with_base) {
    int rc = with_base ? ingot_fc_model_add(m, FLAG_CONTEXTS, 2, NULL) : 0;
    if (!rc)
        rc = ingot_fc_model_add(m, SCALE_CONTEXTS, SCALE_ALPHABET, NULL);
    if (!rc)
        rc = ingot_fc_model_add(m, 1, INGOT_STEP_ALPHABET, NULL);
    if (!rc)
        rc = ingot_fc_model_add(m, 1, POS_ALPHABET, NULL);
    if (!rc)
        rc = ingot_fc_model_add(m, CODE_CONTEXTS, CODE_ALPHABET, NULL);
    pthread_once(&TABLES_ONCE, build_tables);
    return rc;
}

static inline uint16_t load16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static inline void store16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

typedef struct {
    uint16_t *d16;
    uint16_t *sorted;
    uint8_t *same;
    uint64_t blocks;
    uint32_t ctx_flag, ctx_scale, ctx_fp16, ctx_pos, ctx_code;
    int32_t ref_amax; /* BF16 amax of the previous coded block (reachable ones) */
    int32_t ref_fp16; /* ordered fp16 scale of the previous coded block */
} q8_row;

/* One coded block's scale: a step of its BF16 amax, or (escape) a step of the fp16 pattern. */
static uint16_t code_scale(ingot_fcoder *c, q8_row *w, uint32_t ctx, uint16_t d16) {
    ingot_step_code sc = {SCALE_ESCAPE, 0, 0};
    if (ingot_fc_encoding(c) && AMAX_OF[d16] != NO_AMAX)
        sc = ingot_step_encode((int32_t)AMAX_OF[d16] - w->ref_amax);
    uint32_t sym = ingot_fc_sym(c, ctx, sc.symbol);
    if (sym == SCALE_ESCAPE) {
        int32_t ord = w->ref_fp16 +
                      ingot_fc_step(c, w->ctx_fp16, ingot_bf16_ordered(d16) - w->ref_fp16);
        if (ord < -32768 || ord > 32767) {
            c->err = c->err ? c->err : -INGOT_ECORRUPT;
            return 0;
        }
        w->ref_fp16 = ord;
        return ingot_bf16_from_ordered(ord); /* same sign-magnitude ordering as BF16 */
    }
    int32_t a = w->ref_amax +
                ingot_step_join(sym, ingot_fc_raw(c, ingot_step_raw_bits(sym), sc.raw));
    if (a < 0 || a >= BF16_FINITE) {
        c->err = c->err ? c->err : -INGOT_ECORRUPT;
        return 0;
    }
    w->ref_amax = a;
    w->ref_fp16 = ingot_bf16_ordered(SCALE_OF[a]);
    return SCALE_OF[a];
}

/* Flags and scales of row r; decoding writes the identical blocks and the scales. */
static void code_row_scales(ingot_fcoder *c, const ingot_qrows *q, uint64_t r, q8_row *w) {
    uint32_t prev_flag = 0;
    for (uint64_t b = 0; b < w->blocks; b++) {
        uint64_t at = (r * w->blocks + b) * INGOT_Q8_BLOCK_BYTES;
        uint32_t same = 0;
        if (q->base) {
            if (ingot_fc_encoding(c))
                same = !memcmp(q->src + at, q->base + at, INGOT_Q8_BLOCK_BYTES);
            same = ingot_fc_sym(c, w->ctx_flag + prev_flag, same);
            prev_flag = same;
        }
        w->same[b] = (uint8_t)same;
        if (same) {
            w->d16[b] = load16(q->base + at);
            if (q->dst)
                memcpy(q->dst + at, q->base + at, INGOT_Q8_BLOCK_BYTES);
            continue;
        }
        uint16_t d = ingot_fc_encoding(c) ? load16(q->src + at) : 0;
        w->d16[b] = code_scale(c, w, w->ctx_scale + (b != 0), d);
        if (q->dst)
            store16(q->dst + at, w->d16[b]);
    }
}

static int compare_u16(const void *a, const void *b) {
    uint16_t x = *(const uint16_t *)a, y = *(const uint16_t *)b;
    return (x > y) - (x < y);
}

static uint32_t code_context(uint16_t d16, uint16_t median) {
    int diff = (int)d16 - (int)median;
    int k = diff >= 0 ? diff / HALF_BINADE : -((-diff + HALF_BINADE - 1) / HALF_BINADE);
    k += CODE_CONTEXTS / 2;
    return (uint32_t)(k < 0 ? 0 : k >= CODE_CONTEXTS ? CODE_CONTEXTS - 1 : k);
}

static void code_block_codes(ingot_fcoder *c, const ingot_qrows *q, uint64_t at, uint32_t ctx,
                             const q8_row *w) {
    const int8_t *src = ingot_fc_encoding(c) ? (const int8_t *)(q->src + at + 2) : NULL;
    uint32_t pos = INGOT_Q8_BLOCK;
    for (uint32_t k = 0; src && k < INGOT_Q8_BLOCK && pos == INGOT_Q8_BLOCK; k++)
        if (src[k] == 127 || src[k] == -127)
            pos = k;
    pos = ingot_fc_sym(c, w->ctx_pos, pos);
    uint32_t negative = pos < INGOT_Q8_BLOCK ? ingot_fc_raw(c, 1, src && src[pos] < 0) : 0;
    uint32_t sym[INGOT_Q8_BLOCK]; /* the codes other than the one at pos, as code + 128 */
    uint32_t n = 0;
    for (uint32_t k = 0; k < INGOT_Q8_BLOCK; k++)
        if (k != pos)
            sym[n++] = src ? (uint32_t)(src[k] + 128) : 0;
    ingot_fc_syms(c, w->ctx_code + ctx, sym, n);
    if (!q->dst)
        return;
    uint8_t codes[INGOT_Q8_BLOCK];
    for (uint32_t k = 0, i = 0; k < INGOT_Q8_BLOCK; k++)
        codes[k] = (uint8_t)(k == pos ? (negative ? -127 : 127) : (int)sym[i++] - 128);
    memcpy(q->dst + at + 2, codes, sizeof(codes));
}

static void q8_code(ingot_fcoder *c, const ingot_qrows *q) {
    q8_row w = {.blocks = q->cols / INGOT_Q8_BLOCK};
    w.d16 = q->scratch;
    w.sorted = w.d16 + w.blocks;
    w.same = (uint8_t *)(w.sorted + w.blocks);
    uint32_t off = q->base ? FLAG_CONTEXTS : 0;
    w.ctx_flag = 0;
    w.ctx_scale = off;
    w.ctx_fp16 = off + SCALE_CONTEXTS;
    w.ctx_pos = off + SCALE_CONTEXTS + 1;
    w.ctx_code = off + SCALE_CONTEXTS + 2;
    for (uint64_t r = 0; r < q->rows && !c->err; r++) {
        code_row_scales(c, q, r, &w);
        memcpy(w.sorted, w.d16, w.blocks * sizeof(*w.sorted));
        qsort(w.sorted, w.blocks, sizeof(*w.sorted), compare_u16);
        uint16_t median = w.sorted[w.blocks / 2];
        for (uint64_t b = 0; b < w.blocks; b++) {
            if (w.same[b])
                continue;
            uint64_t at = (r * w.blocks + b) * INGOT_Q8_BLOCK_BYTES;
            code_block_codes(c, q, at, code_context(w.d16[b], median), &w);
        }
    }
}

static const ingot_qtype Q8_TYPE = {q8_scratch_bytes, q8_model, q8_code};

const ingot_qtype *ingot_q8_qtype(void) { return &Q8_TYPE; }
