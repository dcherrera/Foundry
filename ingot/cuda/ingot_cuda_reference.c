/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cuda_reference.c
 * @brief  CPU reference of the GPU decode kernels (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_cuda_reference.h"

#include <math.h>
#include <string.h>

enum { ROW_IDENTICAL = -128, DELTA_FALLBACK = 29, LOW = 1 << 16 };

/** Decoder state of one block. */
typedef struct {
    ingot_cuda_block v;
    uint32_t x[INGOT_CUDA_LANES];
    uint32_t wp;
    uint32_t status;
} block_state;

/** Matvec lane sums of one block. */
typedef struct {
    float acc[INGOT_CUDA_LANES];
    uint32_t group;
    uint64_t b, lo, hi;
} row_sums;

uint32_t ingot_cuda_step_weights(uint32_t kind) {
    return kind == INGOT_CUDA_KIND_MATCH ? INGOT_CUDA_MATCH_GROUP : INGOT_CUDA_LANES;
}

static void block_begin(block_state *s, const ingot_cuda_params *p, uint64_t b) {
    memset(s, 0, sizeof(*s));
    s->v = ingot_cuda_block_at(p, b);
    memcpy(s->x, s->v.states, sizeof(s->x));
}

/* One symbol on a lane (an empty context decodes 0 and flags the block). */
static uint32_t lane_symbol(block_state *s, uint32_t lane, const uint8_t *table) {
    if (!table) {
        s->status |= INGOT_CUDA_BAD_TABLE;
        return 0;
    }
    return ingot_cuda_rans_step(&s->x[lane], table);
}

/* Renormalize a lane in stream order (the kernels' ballot order). */
static void lane_renorm(block_state *s, uint32_t lane) {
    if (s->x[lane] >= LOW)
        return;
    if (s->wp < s->v.word_count)
        s->x[lane] = s->x[lane] << 16 | s->v.words[s->wp];
    else
        s->status |= INGOT_CUDA_BAD_WORDS;
    s->wp++;
}

static uint32_t block_end(const block_state *s, uint64_t raw_bits_used) {
    uint32_t st = s->status;
    for (int l = 0; l < INGOT_CUDA_LANES; l++)
        st |= s->x[l] != LOW ? INGOT_CUDA_BAD_STATE : 0;
    st |= s->wp != s->v.word_count ? INGOT_CUDA_BAD_WORD_COUNT : 0;
    st |= (raw_bits_used + 7) / 8 != s->v.raw_bytes ? INGOT_CUDA_BAD_RAW : 0;
    return st;
}

/* The kernels' shuffle tree over the G lane sums (offsets G/2 .. 1); clears the sums. */
static float row_reduce(row_sums *rs) {
    for (uint32_t off = rs->group / 2; off >= 1; off /= 2)
        for (uint32_t l = 0; l < off; l++)
            rs->acc[l] += rs->acc[l + off];
    float sum = rs->acc[0];
    memset(rs->acc, 0, sizeof(rs->acc));
    return sum;
}

/* Store and/or accumulate weight i (flat position pos) of a block. */
static void emit(const ingot_cuda_params *p, row_sums *rs, uint64_t i, uint32_t w) {
    uint64_t pos = rs->lo + i;
    if (p->flags & INGOT_CUDA_STORE)
        ((uint16_t *)(uintptr_t)p->out)[pos] = (uint16_t)w;
    if (!(p->flags & INGOT_CUDA_MATVEC))
        return;
    const float *x = (const float *)(uintptr_t)p->x;
    uint32_t lane = (uint32_t)(i % rs->group);
    rs->acc[lane] = fmaf(ingot_cuda_bf16_f32(w), x[pos % p->cols], rs->acc[lane]);
    if ((pos + 1) % p->cols == 0 || pos + 1 == rs->hi)
        ingot_cuda_row_store(p, rs->b, rs->lo, rs->hi, pos / p->cols, row_reduce(rs));
}

static uint32_t decode_plain(const ingot_cuda_params *p, uint64_t b) {
    row_sums rs = {{0}, INGOT_CUDA_LANES, b, 0, 0};
    ingot_cuda_block_range(p, b, &rs.lo, &rs.hi);
    uint64_t n = rs.hi - rs.lo;
    block_state s;
    block_begin(&s, p, b);
    const uint8_t *table = ingot_cuda_table(p, (uint32_t)(b / p->blocks_per_group));
    for (uint64_t i = 0; i < n; i++) {
        uint32_t lane = (uint32_t)(i % INGOT_CUDA_LANES);
        uint32_t sym = lane_symbol(&s, lane, table);
        lane_renorm(&s, lane);
        uint32_t raw6 = ingot_cuda_raw_bits(s.v.raw, 6 * i, 6);
        emit(p, &rs, i, (raw6 & 0x20u) << 10 | sym << 5 | (raw6 & 0x1Fu));
    }
    return block_end(&s, 6 * n);
}

/* One delta weight given its base value and row scale; *bit advances over its raw bits. */
static uint32_t delta_weight(const ingot_cuda_params *p, block_state *s, uint32_t lane,
                             int32_t rs, uint32_t base, uint64_t *bit) {
    uint32_t e = (base >> 7) & 0xFFu;
    int32_t k = rs - 2 * ((int32_t)(e ? e : 1u) - 134);
    uint32_t cx = k > 12 ? DELTA_FALLBACK : (uint32_t)((k < -16 ? -16 : k) + 16);
    uint32_t sym = lane_symbol(s, lane, ingot_cuda_table(p, cx));
    lane_renorm(s, lane);
    if (cx == DELTA_FALLBACK) {
        uint32_t raw6 = ingot_cuda_raw_bits(s->v.raw, *bit, 6);
        *bit += 6;
        return (raw6 & 0x20u) << 10 | sym << 5 | (raw6 & 0x1Fu);
    }
    uint32_t nb = ingot_cuda_step_raw_bits(sym);
    int32_t ord = ingot_cuda_bf16_ordered(base) +
                  ingot_cuda_step_join(sym, ingot_cuda_raw_bits(s->v.raw, *bit, nb));
    *bit += nb;
    if (ord < -32768 || ord > 32767) {
        s->status |= INGOT_CUDA_BAD_DELTA;
        return base;
    }
    return ingot_cuda_bf16_from_ordered(ord);
}

static uint32_t decode_delta(const ingot_cuda_params *p, uint64_t b) {
    row_sums rs = {{0}, INGOT_CUDA_LANES, b, 0, 0};
    ingot_cuda_block_range(p, b, &rs.lo, &rs.hi);
    block_state s;
    block_begin(&s, p, b);
    const int8_t *row_scale = (const int8_t *)(uintptr_t)p->row_scale;
    const uint16_t *base = (const uint16_t *)(uintptr_t)p->base;
    uint64_t bit = 0;
    for (uint64_t i = 0; i < rs.hi - rs.lo; i++) {
        uint64_t pos = rs.lo + i;
        int32_t scale = row_scale[pos / p->cols];
        uint32_t w = scale == ROW_IDENTICAL
                         ? base[pos]
                         : delta_weight(p, &s, (uint32_t)(i % INGOT_CUDA_LANES), scale, base[pos],
                                        &bit);
        emit(p, &rs, i, w);
    }
    return block_end(&s, bit);
}

static uint32_t decode_match(const ingot_cuda_params *p, uint64_t b, const uint8_t *cells,
                             const uint32_t *bases) {
    row_sums rs = {{0}, INGOT_CUDA_MATCH_GROUP, b, 0, 0};
    ingot_cuda_block_range(p, b, &rs.lo, &rs.hi);
    uint64_t n = rs.hi - rs.lo;
    block_state s;
    block_begin(&s, p, b);
    for (uint64_t i = 0; i < n; i++) {
        uint64_t pos = rs.lo + i, r = pos / p->cols, c = pos % p->cols;
        uint32_t neg = 0, j = (uint32_t)(i % INGOT_CUDA_MATCH_GROUP);
        uint32_t z = ingot_cuda_match_context(p, p->source, cells, bases, p->axis ? c : r,
                                              p->axis ? r : c, &neg);
        uint32_t flip = lane_symbol(&s, 2 * j, ingot_cuda_table(p, z));
        lane_renorm(&s, 2 * j);
        uint32_t sym = lane_symbol(&s, 2 * j + 1, ingot_cuda_table(p, p->zcount + 2 * z + flip));
        lane_renorm(&s, 2 * j + 1);
        uint32_t low = ingot_cuda_raw_bits(s.v.raw, 5 * i, 5);
        emit(p, &rs, i, ((flip ^ neg) & 1u) << 15 | sym << 5 | low);
    }
    return block_end(&s, 5 * n);
}

void ingot_cuda_reference_run(const ingot_cuda_params *p) {
    uint8_t cells[INGOT_CUDA_BUCKET_SETS * INGOT_CUDA_BUCKET_CELLS];
    uint32_t bases[INGOT_CUDA_BUCKET_SETS * 12];
    ingot_cuda_bucket_tables(cells, bases, 0, 1);
    uint32_t *status = (uint32_t *)(uintptr_t)p->status;
    for (uint64_t b = p->block_first; b < p->block_first + p->launch_blocks; b++) {
        uint32_t st = p->kind == INGOT_CUDA_KIND_PLAIN   ? decode_plain(p, b)
                      : p->kind == INGOT_CUDA_KIND_DELTA ? decode_delta(p, b)
                                                         : decode_match(p, b, cells, bases);
        status[b] = st;
    }
}
