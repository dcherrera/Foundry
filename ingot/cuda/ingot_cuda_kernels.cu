/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cuda_kernels.cu
 * @brief  K80 (sm_37, CUDA 11.4) decode of BF16 .ingot payloads, alone or fused with y = W x:
 *         plain (mode 1), delta (mode 2) and match (modes 6 and 7) block decoders.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Build: nvcc -ptx -arch=sm_37 (see ingot/Makefile, target cuda). Every kernel takes one device
 * pointer to an ingot_cuda_params block (ingot_cuda_device.h).
 *
 * Mapping. A rANS block has 32 interleaved lanes and its symbols use lane n % 32 in stream
 * order. Plain and delta code one symbol per weight, so one warp decodes one block, thread l =
 * lane l, 32 weights per step. Match codes two symbols per weight (sign-agree bit, then the
 * plain symbol), so weight i uses lanes 2(i % 16) and 2(i % 16) + 1: one thread owns both lanes
 * of a weight, a half-warp decodes one block (16 weights per step), a warp two blocks.
 *
 * Renormalization. Words are shared by the lanes in stream order: after every lane of a step
 * has decoded, a ballot of the lanes needing a word and a popcount below each lane give its word
 * (match: two ballots, the sign bit's word before the symbol's within a thread).
 *
 * Matvec (cols a multiple of the step's weights): each thread sums its weights times x with
 * fmaf; at each row end the group's sums are added by a shuffle tree; whole rows go to y, rows
 * split across blocks to partials (ingot_cuda_combine adds them in block order). The CPU
 * reference (ingot_cuda_reference.c) runs the same order, so results match bit for bit.
 *
 * Self sources (match): a block reads earlier rows of the output it writes. Blocks of such a
 * tensor are launched one at a time, in order; reads of the output bypass the read-only cache
 * and every step's stores are fenced. Rows must hold >= 16 weights (an earlier row is then
 * always from an earlier step).
 */
#include "ingot_cuda_device.h"

#define FULL 0xFFFFFFFFu
#define LOW 0x10000u

static __device__ __forceinline__ uint32_t lanemask_lt(void) {
    uint32_t m;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
    return m;
}

/* OR of a value over the lanes [0, width) of each width-aligned group. */
static __device__ __forceinline__ uint32_t group_or(uint32_t v, int width) {
    for (int off = width / 2; off >= 1; off /= 2)
        v |= __shfl_xor_sync(FULL, v, off);
    return v;
}

/* The shuffle tree: lane 0 of each width-aligned group gets the group's sum (offsets width/2..1),
 * in the order ingot_cuda_reference.c replays. */
static __device__ __forceinline__ float group_sum(float v, int width) {
    for (int off = width / 2; off >= 1; off /= 2)
        v += __shfl_down_sync(FULL, v, off, width);
    return v;
}

/* Bucket tables and source descriptors into shared memory (all threads of the block). */
static __device__ void load_shared(const ingot_cuda_params *__restrict__ p, uint8_t *cells,
                                   uint32_t *bases, ingot_cuda_source *sources) {
    ingot_cuda_bucket_tables(cells, bases, threadIdx.x, blockDim.x);
    for (uint32_t s = threadIdx.x; s < p->source_count; s += blockDim.x)
        sources[s] = p->source[s];
    __syncthreads();
}

/* One lane's renormalization within a warp-wide step (plain, delta). */
static __device__ __forceinline__ void warp_renorm(uint32_t *x, bool need, const uint16_t *words,
                                                   uint32_t word_count, uint32_t *wp,
                                                   uint32_t *status) {
    uint32_t mask = __ballot_sync(FULL, need);
    if (need) {
        uint32_t at = *wp + __popc(mask & lanemask_lt());
        if (at < word_count)
            *x = *x << 16 | __ldg(words + at);
        else
            *status |= INGOT_CUDA_BAD_WORDS;
    }
    *wp += __popc(mask);
}

/** Per-thread matvec state: the lane sum and the weight's column / row. */
typedef struct {
    float acc;
    uint64_t row, col;
} lane_rows;

/* Position of weight lo + i, and the step advance by g weights. */
static __device__ __forceinline__ void lane_rows_begin(lane_rows *lr, uint64_t pos, uint64_t cols) {
    lr->acc = 0.0f;
    lr->row = pos / cols;
    lr->col = pos % cols;
}

static __device__ __forceinline__ void lane_rows_advance(lane_rows *lr, uint32_t g, uint64_t cols) {
    lr->col += g;
    while (lr->col >= cols) {
        lr->col -= cols;
        lr->row++;
    }
}

/* End of a step for the matvec: if the group's row ends (or the block does), reduce and store.
 * flush must be uniform per group; any_flush is uniform per warp (keeps the shuffles converged). */
static __device__ __forceinline__ void step_rows(const ingot_cuda_params *__restrict__ p,
                                                 lane_rows *lr, bool flush, bool any_flush,
                                                 int width, uint64_t b, uint64_t lo, uint64_t hi,
                                                 uint64_t row) {
    if (!any_flush)
        return;
    float sum = group_sum(lr->acc, width);
    if (flush) {
        if ((threadIdx.x & (width - 1)) == 0)
            ingot_cuda_row_store(p, b, lo, hi, row, sum);
        lr->acc = 0.0f;
    }
}

/* ---- Plain ---------------------------------------------------------------------------------- */

extern "C" __global__ void ingot_cuda_plain(const ingot_cuda_params *__restrict__ p) {
    const uint32_t lane = threadIdx.x & 31;
    const uint64_t warp = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= p->launch_blocks)
        return;
    const uint64_t b = p->block_first + warp, cols = p->cols;
    const bool store = p->flags & INGOT_CUDA_STORE, matvec = p->flags & INGOT_CUDA_MATVEC;
    uint64_t lo, hi;
    ingot_cuda_block_range(p, b, &lo, &hi);
    const uint32_t n = (uint32_t)(hi - lo);
    const ingot_cuda_block v = ingot_cuda_block_at(p, b);
    const uint8_t *table = ingot_cuda_table(p, (uint32_t)(b / p->blocks_per_group));
    uint32_t x = v.states[lane], wp = 0, status = table ? 0 : INGOT_CUDA_BAD_TABLE;
    uint16_t *out = (uint16_t *)(uintptr_t)p->out;
    const float *xv = (const float *)(uintptr_t)p->x;
    lane_rows lr;
    lane_rows_begin(&lr, lo + lane, cols);
    for (uint32_t step = 0; step < n; step += 32) {
        const uint32_t i = step + lane;
        const bool active = i < n;
        uint32_t sym = active && table ? ingot_cuda_rans_step(&x, table) : 0;
        warp_renorm(&x, active && x < LOW, v.words, v.word_count, &wp, &status);
        if (active) {
            uint32_t raw6 = ingot_cuda_raw_bits(v.raw, 6ull * i, 6);
            uint32_t w = (raw6 & 0x20u) << 10 | sym << 5 | (raw6 & 0x1Fu);
            if (store)
                out[lo + i] = (uint16_t)w;
            if (matvec)
                lr.acc = fmaf(ingot_cuda_bf16_f32(w), __ldg(xv + lr.col), lr.acc);
        }
        if (matvec) {
            const uint64_t end = lo + step + 32 < hi ? lo + step + 32 : hi;
            const uint64_t row = (lo + step) / cols;
            const bool flush = end % cols == 0 || end == hi;
            step_rows(p, &lr, flush, flush, 32, b, lo, hi, row);
            lane_rows_advance(&lr, 32, cols);
        }
    }
    status |= __ballot_sync(FULL, x != LOW) ? INGOT_CUDA_BAD_STATE : 0;
    status |= wp != v.word_count ? INGOT_CUDA_BAD_WORD_COUNT : 0;
    status |= v.raw_bytes != (6ull * n + 7) / 8 ? INGOT_CUDA_BAD_RAW : 0;
    status = group_or(status, 32);
    if (lane == 0)
        ((uint32_t *)(uintptr_t)p->status)[b] = status;
}

/* ---- Delta ---------------------------------------------------------------------------------- */

/* Exclusive prefix of nbits over the warp; *total = the sum. */
static __device__ __forceinline__ uint32_t warp_prefix(uint32_t nbits, uint32_t *total) {
    const uint32_t lane = threadIdx.x & 31;
    uint32_t incl = nbits;
    for (int off = 1; off < 32; off <<= 1) {
        uint32_t t = __shfl_up_sync(FULL, incl, off);
        incl += lane >= (uint32_t)off ? t : 0;
    }
    *total = __shfl_sync(FULL, incl, 31);
    return incl - nbits;
}

/* Context of a delta weight (29 = fallback: plain symbol + 6 raw bits). */
static __device__ __forceinline__ uint32_t delta_context(int32_t rs, uint32_t base) {
    uint32_t e = (base >> 7) & 0xFFu;
    int32_t k = rs - 2 * ((int32_t)(e ? e : 1u) - 134);
    return k > 12 ? 29u : (uint32_t)((k < -16 ? -16 : k) + 16);
}

extern "C" __global__ void ingot_cuda_delta(const ingot_cuda_params *__restrict__ p) {
    const uint32_t lane = threadIdx.x & 31;
    const uint64_t warp = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= p->launch_blocks)
        return;
    const uint64_t b = p->block_first + warp, cols = p->cols;
    const bool store = p->flags & INGOT_CUDA_STORE, matvec = p->flags & INGOT_CUDA_MATVEC;
    uint64_t lo, hi;
    ingot_cuda_block_range(p, b, &lo, &hi);
    const uint32_t n = (uint32_t)(hi - lo);
    const ingot_cuda_block v = ingot_cuda_block_at(p, b);
    const int8_t *row_scale = (const int8_t *)(uintptr_t)p->row_scale;
    const uint16_t *base = (const uint16_t *)(uintptr_t)p->base;
    uint32_t x = v.states[lane], wp = 0, status = 0;
    uint64_t bit = 0;
    lane_rows lr;
    lane_rows_begin(&lr, lo + lane, cols);
    for (uint32_t step = 0; step < n; step += 32) {
        const uint32_t i = step + lane;
        const bool active = i < n;
        const uint32_t bw = active ? __ldg(base + lo + i) : 0;
        const int32_t rs = active ? __ldg(row_scale + lr.row) : -128;
        const bool coded = rs != -128;
        const uint32_t cx = delta_context(rs, bw);
        const uint8_t *table = coded ? ingot_cuda_table(p, cx) : 0;
        status |= coded && !table ? INGOT_CUDA_BAD_TABLE : 0;
        uint32_t sym = table ? ingot_cuda_rans_step(&x, table) : 0;
        warp_renorm(&x, coded && x < LOW, v.words, v.word_count, &wp, &status);
        uint32_t nb = !coded ? 0 : cx == 29 ? 6 : ingot_cuda_step_raw_bits(sym), total;
        uint64_t at = bit + warp_prefix(nb, &total);
        bit += total;
        uint32_t w = bw;
        if (coded && cx == 29) {
            uint32_t raw6 = ingot_cuda_raw_bits(v.raw, at, 6);
            w = (raw6 & 0x20u) << 10 | sym << 5 | (raw6 & 0x1Fu);
        } else if (coded) {
            int32_t ord = ingot_cuda_bf16_ordered(bw) +
                          ingot_cuda_step_join(sym, ingot_cuda_raw_bits(v.raw, at, nb));
            status |= ord < -32768 || ord > 32767 ? INGOT_CUDA_BAD_DELTA : 0;
            w = ord < -32768 || ord > 32767 ? bw : ingot_cuda_bf16_from_ordered(ord);
        }
        if (active && store)
            ((uint16_t *)(uintptr_t)p->out)[lo + i] = (uint16_t)w;
        if (active && matvec)
            lr.acc = fmaf(ingot_cuda_bf16_f32(w), __ldg((const float *)(uintptr_t)p->x + lr.col),
                          lr.acc);
        const uint64_t end = lo + step + 32 < hi ? lo + step + 32 : hi;
        const bool flush = matvec && (end % cols == 0 || end == hi);
        step_rows(p, &lr, flush, flush, 32, b, lo, hi, (lo + step) / cols);
        lane_rows_advance(&lr, 32, cols);
    }
    status |= __ballot_sync(FULL, x != LOW) ? INGOT_CUDA_BAD_STATE : 0;
    status |= wp != v.word_count ? INGOT_CUDA_BAD_WORD_COUNT : 0;
    status |= v.raw_bytes != (bit + 7) / 8 ? INGOT_CUDA_BAD_RAW : 0;
    status = group_or(status, 32);
    if (lane == 0)
        ((uint32_t *)(uintptr_t)p->status)[b] = status;
}

/* ---- Match (modes 6 and 7) ------------------------------------------------------------------ */

/* Both lanes of a match weight renormalized in stream order (sign bit's word first). */
static __device__ __forceinline__ void half_renorm(uint32_t *x0, uint32_t *x1, bool need0,
                                                   bool need1, const uint16_t *words,
                                                   uint32_t word_count, uint32_t *wp,
                                                   uint32_t *status) {
    const uint32_t shift = threadIdx.x & 16;
    const uint32_t m0 = (__ballot_sync(FULL, need0) >> shift) & 0xFFFFu;
    const uint32_t m1 = (__ballot_sync(FULL, need1) >> shift) & 0xFFFFu;
    const uint32_t lt = (lanemask_lt() >> shift) & 0xFFFFu;
    const uint32_t before = *wp + __popc(m0 & lt) + __popc(m1 & lt);
#ifdef INGOT_CUDA_RED_WORDS
    const uint32_t at0 = before + need1, at1 = before; /* planted red: symbol's word first */
#else
    const uint32_t at0 = before, at1 = before + need0;
#endif
    if (need0) {
        if (at0 < word_count)
            *x0 = *x0 << 16 | __ldg(words + at0);
        else
            *status |= INGOT_CUDA_BAD_WORDS;
    }
    if (need1) {
        if (at1 < word_count)
            *x1 = *x1 << 16 | __ldg(words + at1);
        else
            *status |= INGOT_CUDA_BAD_WORDS;
    }
    *wp += __popc(m0) + __popc(m1);
}

/* Sign-agree bit and plain symbol of one weight in context z; the decoded BF16 bits. */
static __device__ __forceinline__ uint32_t match_symbols(const ingot_cuda_params *__restrict__ p,
                                                         uint32_t z, uint32_t neg, uint32_t *x0,
                                                         uint32_t *x1, uint32_t *status) {
    const uint8_t *tf = ingot_cuda_table(p, z);
    uint32_t flip = tf ? ingot_cuda_rans_step(x0, tf) : 0;
    const uint8_t *ts = ingot_cuda_table(p, p->zcount + 2 * z + flip);
    uint32_t sym = ts ? ingot_cuda_rans_step(x1, ts) : 0;
    *status |= !tf || !ts ? INGOT_CUDA_BAD_TABLE : 0;
    return ((flip ^ neg) & 1u) << 15 | sym << 5;
}

extern "C" __global__ void ingot_cuda_match(const ingot_cuda_params *__restrict__ p) {
    __shared__ uint8_t cells[INGOT_CUDA_BUCKET_SETS * INGOT_CUDA_BUCKET_CELLS];
    __shared__ uint32_t bases[INGOT_CUDA_BUCKET_SETS * 12];
    __shared__ ingot_cuda_source sources[INGOT_CUDA_MAX_SOURCES];
    load_shared(p, cells, bases, sources);
    const uint32_t j = threadIdx.x & 15;
    const uint64_t half = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 4;
    if ((half & ~(uint64_t)1) >= p->launch_blocks)
        return; /* whole warp: neither half has a block */
    const bool has = half < p->launch_blocks;
    const uint64_t b = p->block_first + (has ? half : p->launch_blocks - 1), cols = p->cols;
    const bool store = p->flags & INGOT_CUDA_STORE, matvec = p->flags & INGOT_CUDA_MATVEC;
    bool self = false;
    for (uint32_t s = 0; s < p->source_count; s++)
        self |= sources[s].self != 0;
    uint64_t lo, hi;
    ingot_cuda_block_range(p, b, &lo, &hi);
    const uint32_t n = has ? (uint32_t)(hi - lo) : 0;
    const uint32_t steps = max(n, __shfl_xor_sync(FULL, n, 16));
    const ingot_cuda_block v = ingot_cuda_block_at(p, b);
    uint32_t x0 = v.states[2 * j], x1 = v.states[2 * j + 1], wp = 0, status = 0;
    lane_rows lr;
    lane_rows_begin(&lr, lo + j, cols);
    for (uint32_t step = 0; step < steps; step += 16) {
        const uint32_t i = step + j;
        const bool active = i < n;
        uint32_t w = 0;
        if (active) {
            uint32_t neg = 0;
            uint32_t z = ingot_cuda_match_context(p, sources, cells, bases,
                                                  p->axis ? lr.col : lr.row,
                                                  p->axis ? lr.row : lr.col, &neg);
            w = match_symbols(p, z, neg, &x0, &x1, &status);
        }
        half_renorm(&x0, &x1, active && x0 < LOW, active && x1 < LOW, v.words, v.word_count,
                    &wp, &status);
        if (active) {
            w |= ingot_cuda_raw_bits(v.raw, 5ull * i, 5);
            if (store)
                ((uint16_t *)(uintptr_t)p->out)[lo + i] = (uint16_t)w;
            if (matvec)
                lr.acc = fmaf(ingot_cuda_bf16_f32(w),
                              __ldg((const float *)(uintptr_t)p->x + lr.col), lr.acc);
        }
        if (self)
            __threadfence_block();
        const uint64_t end = lo + step + 16 < hi ? lo + step + 16 : hi;
        const bool flush = matvec && step < n && (end % cols == 0 || end == hi);
        step_rows(p, &lr, flush, __any_sync(FULL, flush), 16, b, lo, hi, (lo + step) / cols);
        lane_rows_advance(&lr, 16, cols);
    }
    status |= (__ballot_sync(FULL, x0 != LOW || x1 != LOW) >> (threadIdx.x & 16)) & 0xFFFFu
                  ? INGOT_CUDA_BAD_STATE
                  : 0;
    status |= wp != v.word_count ? INGOT_CUDA_BAD_WORD_COUNT : 0;
    status |= v.raw_bytes != (5ull * n + 7) / 8 ? INGOT_CUDA_BAD_RAW : 0;
    status = group_or(status, 16);
    if (has && j == 0)
        ((uint32_t *)(uintptr_t)p->status)[b] = status;
}

/* ---- Split rows ----------------------------------------------------------------------------- */

extern "C" __global__ void ingot_cuda_combine(const ingot_cuda_params *__restrict__ p) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        ingot_cuda_combine_rows(p);
}
