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

/* Copy `bytes` (a multiple of 8) of a table image to shared memory with n threads from first. */
static __device__ __forceinline__ void table_to_shared(unsigned long long *dst, const uint8_t *src,
                                                       uint32_t bytes, uint32_t first, uint32_t n) {
    const unsigned long long *s = (const unsigned long long *)src;
    for (uint32_t k = first; k < bytes / 8; k += n)
        dst[k] = __ldg(s + k);
}

/* ingot_cuda_rans_step on a table image in shared memory. */
static __device__ __forceinline__ uint32_t shared_rans_step(uint32_t *x,
                                                            const unsigned long long *table) {
    const uint16_t *coarse = (const uint16_t *)table;
    const unsigned long long *entry = table + INGOT_CUDA_TABLE_HEADER / 8;
    const uint32_t slot = *x & 0xFFFFu;
    uint32_t r = coarse[slot >> INGOT_CUDA_COARSE_SHIFT];
    unsigned long long e = entry[r];
    while (slot - (uint32_t)(e & 0xFFFFu) > (uint32_t)((e >> 16) & 0xFFFFu))
        e = entry[++r];
    *x = ((uint32_t)((e >> 16) & 0xFFFFu) + 1u) * (*x >> 16) + slot - (uint32_t)(e & 0xFFFFu);
    return (uint32_t)(e >> 32);
}

/** A warp's window on its block's words: lane l holds words base + l (w0), base + 32 + l (w1)
 *  and base + 64 + l (w2, loaded a window ahead); r words past base are used (r < 32). */
typedef struct {
    uint32_t w0, w1, w2, base, r;
} word_window;

static __device__ __forceinline__ uint32_t window_word(const uint16_t *words, uint32_t at,
                                                       uint32_t count) {
    return at < count ? __ldg(words + at) : 0u;
}

static __device__ __forceinline__ void window_begin(word_window *ww, const uint16_t *words,
                                                    uint32_t count, uint32_t lane) {
    ww->w0 = window_word(words, lane, count);
    ww->w1 = window_word(words, 32 + lane, count);
    ww->w2 = window_word(words, 64 + lane, count);
    ww->base = 0;
    ww->r = 0;
}

/* warp_renorm with the words taken from the window by shuffles (a step uses at most 32). */
static __device__ __forceinline__ void window_renorm(word_window *ww, uint32_t *x, bool need,
                                                     const uint16_t *words, uint32_t count,
                                                     uint32_t lane, uint32_t *status) {
    const uint32_t mask = __ballot_sync(FULL, need);
    const uint32_t at = ww->r + __popc(mask & lanemask_lt());
    const uint32_t lo = __shfl_sync(FULL, ww->w0, at & 31);
    const uint32_t hi = __shfl_sync(FULL, ww->w1, at & 31);
    if (need) {
        if (ww->base + at < count)
            *x = *x << 16 | (at < 32 ? lo : hi);
        else
            *status |= INGOT_CUDA_BAD_WORDS;
    }
    ww->r += __popc(mask);
    if (ww->r >= 32) {
        ww->w0 = ww->w1;
        ww->w1 = ww->w2;
        ww->base += 32;
        ww->r -= 32;
        ww->w2 = window_word(words, ww->base + 64 + lane, count);
    }
}

/* Step k's 6 raw bits of each lane: the step's 192 bits are words 6k .. 6k + 5; lanes 0..6 load
 * words 6k + lane and a lane's bits are shuffled in (they may straddle two words). */
static __device__ __forceinline__ uint32_t step_raw6(const uint32_t *raw, uint32_t raw_words,
                                                     uint32_t k, uint32_t lane) {
    const uint32_t at = 6 * k + lane;
    const uint32_t w = lane < 7 && at < raw_words ? __ldg(raw + at) : 0u;
    const uint32_t j = (6 * lane) >> 5;
    const uint32_t a = __shfl_sync(FULL, w, j), b = __shfl_sync(FULL, w, j + 1);
    return __funnelshift_r(a, b, (6 * lane) & 31) & 0x3Fu;
}

/* One warp per block, the block's table in shared memory (one copy per thread block when its
 * warps share a group, else one per warp). Matvec: cols % 32 == 0 and blocks start at multiples
 * of 32, so a step lies in one row; the warp tracks that row and column (no division per step). */
extern "C" __global__ void ingot_cuda_plain(const ingot_cuda_params *__restrict__ p) {
    extern __shared__ unsigned long long shared_tables[];
    const uint32_t lane = threadIdx.x & 31, wi = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const uint64_t warp = (uint64_t)blockIdx.x * warps + wi;
    const bool live = warp < p->launch_blocks;
    const uint64_t b = p->block_first + (live ? warp : p->launch_blocks - 1);
    const uint8_t *gt = ingot_cuda_table(p, (uint32_t)(b / p->blocks_per_group));
    unsigned long long *table = shared_tables;
    if (ingot_cuda_table_slots(p, warps) == 1) {
        if (gt)
            table_to_shared(table, gt, p->table_slot_bytes, threadIdx.x, blockDim.x);
        __syncthreads();
    } else {
        table += wi * (p->table_slot_bytes / 8);
        if (gt)
            table_to_shared(table, gt, p->table_slot_bytes, lane, 32);
        __syncwarp();
    }
    if (!live)
        return;
    const bool store = p->flags & INGOT_CUDA_STORE, matvec = p->flags & INGOT_CUDA_MATVEC;
    uint64_t lo, hi;
    ingot_cuda_block_range(p, b, &lo, &hi);
    const uint32_t n = (uint32_t)(hi - lo), cols = (uint32_t)p->cols;
    const ingot_cuda_block v = ingot_cuda_block_at(p, b);
    const uint32_t *raw = (const uint32_t *)v.raw, raw_words = (v.raw_bytes + 3) / 4;
    uint32_t x = v.states[lane], status = gt ? 0 : INGOT_CUDA_BAD_TABLE;
    word_window ww;
    window_begin(&ww, v.words, v.word_count, lane);
    uint16_t *out = (uint16_t *)(uintptr_t)p->out + lo;
    const float *xv = (const float *)(uintptr_t)p->x;
    uint64_t row = matvec ? lo / cols : 0;
    uint32_t col = matvec ? (uint32_t)(lo % cols) : 0;
    float acc = 0.0f;
    for (uint32_t step = 0, k = 0; step < n; step += 32, k++) {
        const bool active = step + lane < n;
        const uint32_t sym = active && gt ? shared_rans_step(&x, table) : 0;
        window_renorm(&ww, &x, active && x < LOW, v.words, v.word_count, lane, &status);
        const uint32_t raw6 = step_raw6(raw, raw_words, k, lane);
        const uint32_t w = (raw6 & 0x20u) << 10 | sym << 5 | (raw6 & 0x1Fu);
        if (active && store)
            out[step + lane] = (uint16_t)w;
        if (matvec) {
            if (active)
                acc = fmaf(ingot_cuda_bf16_f32(w), __ldg(xv + col + lane), acc);
            col += 32;
            if (col == cols || step + 32 >= n) {
                const float sum = group_sum(acc, 32);
                if (lane == 0)
                    ingot_cuda_row_store(p, b, lo, hi, row, sum);
                acc = 0.0f;
            }
            if (col == cols) {
                col = 0;
                row++;
            }
        }
    }
    status |= __ballot_sync(FULL, x != LOW) ? INGOT_CUDA_BAD_STATE : 0;
    status |= ww.base + ww.r != v.word_count ? INGOT_CUDA_BAD_WORD_COUNT : 0;
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

/** A half-warp's window on its block's words: lane j holds word base + 16 c + j in wc (c = 0..4;
 *  w3 and w4 are loaded a step ahead); r words past base are used (r < 16). A step uses at most
 *  32 words, so it reads only w0..w2. */
typedef struct {
    uint32_t w0, w1, w2, w3, w4, base, r;
} half_window;

static __device__ __forceinline__ void half_window_begin(half_window *hw, const uint16_t *words,
                                                         uint32_t count, uint32_t j) {
    hw->w0 = window_word(words, j, count);
    hw->w1 = window_word(words, 16 + j, count);
    hw->w2 = window_word(words, 32 + j, count);
    hw->w3 = window_word(words, 48 + j, count);
    hw->w4 = window_word(words, 64 + j, count);
    hw->base = 0;
    hw->r = 0;
}

/* Word base + a of the window (a < 48), by shuffles within the half-warp. */
static __device__ __forceinline__ uint32_t half_window_at(const half_window *hw, uint32_t a) {
    const uint32_t v0 = __shfl_sync(FULL, hw->w0, a & 15, 16);
    const uint32_t v1 = __shfl_sync(FULL, hw->w1, a & 15, 16);
    const uint32_t v2 = __shfl_sync(FULL, hw->w2, a & 15, 16);
    return a < 16 ? v0 : a < 32 ? v1 : v2;
}

/* Both lanes of a match weight renormalized in stream order (sign bit's word first). */
static __device__ __forceinline__ void half_renorm(uint32_t *x0, uint32_t *x1, bool need0,
                                                   bool need1, half_window *hw,
                                                   const uint16_t *words, uint32_t word_count,
                                                   uint32_t j, uint32_t *status) {
    const uint32_t shift = threadIdx.x & 16;
    const uint32_t m0 = (__ballot_sync(FULL, need0) >> shift) & 0xFFFFu;
    const uint32_t m1 = (__ballot_sync(FULL, need1) >> shift) & 0xFFFFu;
    const uint32_t lt = (lanemask_lt() >> shift) & 0xFFFFu;
    const uint32_t before = hw->r + __popc(m0 & lt) + __popc(m1 & lt);
#ifdef INGOT_CUDA_RED_WORDS
    const uint32_t at0 = before + need1, at1 = before; /* planted red: symbol's word first */
#else
    const uint32_t at0 = before, at1 = before + need0;
#endif
    const uint32_t word0 = half_window_at(hw, at0), word1 = half_window_at(hw, at1);
    if (need0) {
        if (hw->base + at0 < word_count)
            *x0 = *x0 << 16 | word0;
        else
            *status |= INGOT_CUDA_BAD_WORDS;
    }
    if (need1) {
        if (hw->base + at1 < word_count)
            *x1 = *x1 << 16 | word1;
        else
            *status |= INGOT_CUDA_BAD_WORDS;
    }
    /* advance by s = 0..2 chunks of 16 (selects, no branch: the halves differ) */
    const uint32_t r = hw->r + __popc(m0) + __popc(m1), s = r >> 4;
    hw->r = r & 15;
    hw->base += 16 * s;
    const uint32_t l48 = s == 2 ? window_word(words, hw->base + 48 + j, word_count) : 0u;
    const uint32_t l64 = s != 0 ? window_word(words, hw->base + 64 + j, word_count) : 0u;
    const uint32_t n0 = s == 0 ? hw->w0 : s == 1 ? hw->w1 : hw->w2;
    const uint32_t n1 = s == 0 ? hw->w1 : s == 1 ? hw->w2 : hw->w3;
    const uint32_t n2 = s == 0 ? hw->w2 : s == 1 ? hw->w3 : hw->w4;
    const uint32_t n3 = s == 0 ? hw->w3 : s == 1 ? hw->w4 : l48;
    hw->w4 = s == 0 ? hw->w4 : l64;
    hw->w0 = n0;
    hw->w1 = n1;
    hw->w2 = n2;
    hw->w3 = n3;
}

/* Sign-agree tables (binary) as the frequency f0 of symbol 0 (65536: only 0; 0: only 1; EMPTY:
 * no table), and the symbol tables' offsets, in shared memory. */
static __device__ void match_tables_to_shared(const ingot_cuda_params *__restrict__ p,
                                              uint32_t *sign_f0, uint32_t *sym_off) {
    for (uint32_t z = threadIdx.x; z < p->zcount; z += blockDim.x) {
        const uint8_t *t = ingot_cuda_table(p, z);
        unsigned long long e =
            t ? __ldg((const unsigned long long *)(t + INGOT_CUDA_TABLE_HEADER)) : 0;
        uint32_t freq = (uint32_t)((e >> 16) & 0xFFFFu) + 1u, sym = (uint32_t)(e >> 32);
        sign_f0[z] = !t ? INGOT_CUDA_TABLE_EMPTY : freq < 65536u ? freq : sym ? 0u : 65536u;
    }
    for (uint32_t c = threadIdx.x; c < 2 * p->zcount; c += blockDim.x)
        sym_off[c] = __ldg((const uint32_t *)(uintptr_t)p->table_offsets + p->zcount + c);
}

/* rANS step of a binary table given f0 (ingot_cuda_rans_step on its image). */
static __device__ __forceinline__ uint32_t binary_rans_step(uint32_t *x, uint32_t f0) {
    const uint32_t slot = *x & 0xFFFFu, one = slot >= f0;
    *x = (one ? 65536u - f0 : f0) * (*x >> 16) + slot - (one ? f0 : 0u);
    return one;
}

/* Sign-agree bit and plain symbol of one weight in context z; the decoded BF16 bits. */
static __device__ __forceinline__ uint32_t match_symbols(const ingot_cuda_params *__restrict__ p,
                                                         const uint32_t *sign_f0,
                                                         const uint32_t *sym_off, uint32_t z,
                                                         uint32_t neg, uint32_t *x0,
                                                         uint32_t *x1, uint32_t *status) {
    const uint32_t f0 = sign_f0[z];
    const uint32_t flip = f0 != INGOT_CUDA_TABLE_EMPTY ? binary_rans_step(x0, f0) : 0;
    const uint32_t off = sym_off[2 * z + flip];
    const uint8_t *ts = (const uint8_t *)(uintptr_t)p->tables + off;
    const uint32_t sym = off != INGOT_CUDA_TABLE_EMPTY ? ingot_cuda_rans_step(x1, ts) : 0;
    *status |= f0 == INGOT_CUDA_TABLE_EMPTY || off == INGOT_CUDA_TABLE_EMPTY
                   ? INGOT_CUDA_BAD_TABLE
                   : 0;
    return ((flip ^ neg) & 1u) << 15 | sym << 5;
}

/** One unit's partners held in registers while its weights decode (row units: a whole row). */
typedef struct {
    uint64_t unit;
    uint32_t rec, k, self_mask;
    const uint16_t *ptr[INGOT_CUDA_MAX_K]; /* the partner unit's first element */
    uint32_t pos_stride[INGOT_CUDA_MAX_K];
    int32_t q[INGOT_CUDA_MAX_K], shift[INGOT_CUDA_MAX_K];
} unit_partners;

/* Load unit u's record and partners (unrolled over the 8 slots, so they stay in registers). */
static __device__ __forceinline__ void unit_load(const ingot_cuda_params *__restrict__ p,
                                                 const ingot_cuda_source *sources, uint64_t u,
                                                 unit_partners *c) {
    c->unit = u;
    c->rec = __ldg((const uint32_t *)(uintptr_t)p->unit_records + u);
    c->k = ingot_cuda_unit_k(c->rec);
    c->self_mask = 0;
    const unsigned long long *part = (const unsigned long long *)(uintptr_t)p->partners +
                                     __ldg((const uint32_t *)(uintptr_t)p->partner_offsets + u);
#pragma unroll
    for (uint32_t j = 0; j < INGOT_CUDA_MAX_K; j++) {
        if (j >= c->k)
            continue;
        unsigned long long pr = __ldg(part + j);
        const ingot_cuda_source *s = &sources[(uint32_t)(pr >> 48)];
        c->ptr[j] = (const uint16_t *)(uintptr_t)s->ptr + (uint32_t)pr * s->unit_stride;
        c->pos_stride[j] = s->pos_stride;
        c->q[j] = (int32_t)(int16_t)(uint16_t)(pr >> 32);
        c->shift[j] = s->shift;
        c->self_mask |= s->self << j;
    }
}

/* Context of the weight at position h of the cached unit (the shared header's
 * ingot_cuda_match_context, with the partners hoisted). */
static __device__ __forceinline__ uint32_t unit_context(const unit_partners *c, uint64_t h,
                                                        const uint8_t *cells,
                                                        const uint32_t *bases, uint32_t *neg) {
    int64_t acc = 0;
#pragma unroll
    for (uint32_t j = 0; j < INGOT_CUDA_MAX_K; j++) {
        if (j >= c->k)
            continue;
        const uint16_t *a = c->ptr[j] + (uint32_t)h * c->pos_stride[j];
        uint32_t x = (c->self_mask >> j) & 1u ? *(const volatile uint16_t *)a : __ldg(a);
        acc += ingot_cuda_partner_term(x, c->shift[j], c->q[j]);
    }
    return ingot_cuda_context_of(acc, c->rec, cells, bases, neg);
}

extern "C" __global__ void ingot_cuda_match(
const ingot_cuda_params *__restrict__ p) {
    __shared__ uint8_t cells[INGOT_CUDA_BUCKET_SETS * INGOT_CUDA_BUCKET_CELLS];
    __shared__ uint32_t bases[INGOT_CUDA_BUCKET_SETS * 12];
    __shared__ ingot_cuda_source sources[INGOT_CUDA_MAX_SOURCES];
    __shared__ uint32_t sign_f0[INGOT_CUDA_MAX_Z], sym_off[2 * INGOT_CUDA_MAX_Z];
    match_tables_to_shared(p, sign_f0, sym_off);
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
    uint32_t x0 = v.states[2 * j], x1 = v.states[2 * j + 1], status = 0;
    half_window hw;
    half_window_begin(&hw, v.words, v.word_count, j);
    const uint32_t *raw = (const uint32_t *)v.raw;
    uint32_t raw_word = (5 * j) >> 5, raw_bit = (5 * j) & 31; /* weight step + j's raw bits */
    lane_rows lr;
    lane_rows_begin(&lr, lo + j, cols);
    unit_partners cache;
    cache.unit = ~(uint64_t)0;
    for (uint32_t step = 0; step < steps; step += 16) {
        const uint32_t i = step + j;
        const bool active = i < n;
        uint32_t w = 0;
        if (active) {
            uint32_t neg = 0;
            const uint64_t u = p->axis ? lr.col : lr.row;
            if (u != cache.unit)
                unit_load(p, sources, u, &cache);
            uint32_t z = unit_context(&cache, p->axis ? lr.row : lr.col, cells, bases, &neg);
            w = match_symbols(p, sign_f0, sym_off, z, neg, &x0, &x1, &status);
        }
        half_renorm(&x0, &x1, active && x0 < LOW, active && x1 < LOW, &hw, v.words, v.word_count,
                    j, &status);
        if (active) {
            w |= __funnelshift_r(__ldg(raw + raw_word), __ldg(raw + raw_word + 1), raw_bit) & 0x1Fu;
            if (store)
                ((uint16_t *)(uintptr_t)p->out)[lo + i] = (uint16_t)w;
            if (matvec)
                lr.acc = fmaf(ingot_cuda_bf16_f32(w),
                              __ldg((const float *)(uintptr_t)p->x + lr.col), lr.acc);
        }
        if (self)
            __threadfence_block();
        raw_bit += 80;
        raw_word += raw_bit >> 5;
        raw_bit &= 31;
        /* matvec: cols % 16 == 0, so the step lies in row lr.row from column lr.col - j */
        const bool flush = matvec && step < n && (lr.col - j + 16 == cols || step + 16 >= n);
        step_rows(p, &lr, flush, __any_sync(FULL, flush), 16, b, lo, hi, lr.row);
        lane_rows_advance(&lr, 16, cols);
    }
    status |= (__ballot_sync(FULL, x0 != LOW || x1 != LOW) >> (threadIdx.x & 16)) & 0xFFFFu
                  ? INGOT_CUDA_BAD_STATE
                  : 0;
    status |= hw.base + hw.r != v.word_count ? INGOT_CUDA_BAD_WORD_COUNT : 0;
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
