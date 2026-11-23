/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cuda_device.h
 * @brief  GPU decode of BF16 .ingot payloads (modes 1, 2, 6, 7): the device image layout, the
 *         kernel parameter block, and the per-weight math shared by the CUDA kernels
 *         (ingot_cuda_kernels.cu) and the CPU reference (ingot_cuda_reference.c), so both run
 *         the same integer code.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Valid C11 and CUDA C++ (nvcc -arch=sm_37, CUDA 11.4). Kepler has no BF16 and slow 32-bit
 * integer multiplies (1/6 rate), so the predictor avoids 64-bit multiplies:
 *
 * Partner term. A match / predict context needs p = sum_k q_k X_k exactly (int64), with
 * X = ingot_predict_fixed(x, F). X is an 8-bit mantissa m shifted by e (truncated for e < 0) or
 * saturated at 2^40, so q X = (q m') << s with q m' < 2^23: one 32-bit multiply and a 64-bit
 * shift per partner (ingot_cuda_partner_term).
 *
 * Bucket. Unit u's thresholds are T_k = B_k 2^sh (floor, clamped at 2^62) with B_k = t_k R and
 * R = 2^((s mod 4) / 4) in Q16, so B_k < 2^27 depends only on s mod 4 (rsel). |p| >= T_k reduces
 * to v >= B_k for one 32-bit v (|p| >> sh for sh >= 0; (|p| + 1) 2^-sh - 1 for sh < 0). Adjacent
 * B_k are >= 2^20 apart, so a 2048-cell table on v >> 16 gives the count up to one compare
 * (ingot_cuda_bucket).
 *
 * Table image (one per rANS context, 8-aligned): u16 coarse[1024] (rank of the symbol holding
 * slot 64 c), then u64 entry[used] = start | (freq - 1) << 16 | symbol << 32. A lookup reads the
 * coarse rank and steps forward while the slot is past the entry (ingot_cuda_table_find).
 */
#ifndef FOUNDRY_INGOT_CUDA_DEVICE_H
#define FOUNDRY_INGOT_CUDA_DEVICE_H

#include <stdint.h>

#if defined(__CUDACC__)
#define INGOT_CUDA_FN __host__ __device__ __forceinline__
#define INGOT_CUDA_LOAD(p) __ldg(p)
#else
#define INGOT_CUDA_FN static inline
#define INGOT_CUDA_LOAD(p) (*(p))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    INGOT_CUDA_COARSE_BITS = 10,
    INGOT_CUDA_COARSE_CELLS = 1 << INGOT_CUDA_COARSE_BITS,
    INGOT_CUDA_COARSE_SHIFT = 16 - INGOT_CUDA_COARSE_BITS,
    INGOT_CUDA_TABLE_HEADER = 2 * INGOT_CUDA_COARSE_CELLS, /* bytes before the entries */
    INGOT_CUDA_THRESHOLDS = 11,
    INGOT_CUDA_BUCKET_CELLS = 2048,       /* v >> 16, clamped */
    INGOT_CUDA_BUCKET_SETS = 4,           /* rsel = scale mod 4 */
    INGOT_CUDA_MAX_SOURCES = 64,
    INGOT_CUDA_MAX_K = 8,
    INGOT_CUDA_LANES = 32,
    INGOT_CUDA_MATCH_GROUP = 16,          /* weights per step of a match / predict block */
    INGOT_CUDA_SHIFT_CLAMP = 64,          /* |sh| beyond this decides nothing more */
};

/* Table offset of a context whose table is empty (decoding it is a corrupt payload). */
#define INGOT_CUDA_TABLE_EMPTY 0xFFFFFFFFu

/** Per-block status bits written by the kernels (0 = block decoded and checked). */
enum {
    INGOT_CUDA_BAD_WORDS = 1,      /* a lane needed a word past the word stream */
    INGOT_CUDA_BAD_TABLE = 2,      /* a symbol in an empty context */
    INGOT_CUDA_BAD_STATE = 4,      /* a lane did not end at 2^16 */
    INGOT_CUDA_BAD_WORD_COUNT = 8, /* words left over */
    INGOT_CUDA_BAD_RAW = 16,       /* raw stream size does not match the bits used */
    INGOT_CUDA_BAD_DELTA = 32,     /* a delta step left the BF16 range */
};

/** Kernel output flags. */
enum {
    INGOT_CUDA_STORE = 1,  /* write the decoded BF16 weights */
    INGOT_CUDA_MATVEC = 2, /* y = W x (fp32; row partials of split rows to `partials`) */
};

/** Payload kinds the device decodes. */
enum {
    INGOT_CUDA_KIND_PLAIN = 1,
    INGOT_CUDA_KIND_DELTA = 2,
    INGOT_CUDA_KIND_MATCH = 7, /* modes 6 and 7: mode 6 is converted to per-unit partner lists */
};

/**
 * One predictor source on the device: element (unit i, position h) is at
 * ptr[i * unit_stride + h * pos_stride] (row units: H, 1; column units: 1, n).
 */
typedef struct {
    uint64_t ptr;          /* device address of the BF16 tensor (self: the output buffer) */
    uint32_t unit_stride;
    uint32_t pos_stride;
    int32_t shift;         /* F = C - E */
    uint32_t self;
} ingot_cuda_source;

/**
 * Kernel parameter block (passed by value; identical layout in C and CUDA C++). Device
 * addresses are u64. Unit records (match): bits 0..7 zbase, 8..9 rsel, 16..23 sh + 64,
 * 24..27 K. Partner records: u64 = index | (u16)q << 32 | source << 48.
 */
typedef struct {
    uint32_t kind;
    uint32_t flags;
    uint64_t rows, cols;            /* the tensor as stored (row-major) */
    uint64_t block_count;
    uint64_t block_first;           /* blocks [block_first, block_first + launch_blocks) */
    uint64_t launch_blocks;
    uint64_t area;                  /* block area: u64 offsets[block_count + 1], then blocks */
    uint64_t tables;                /* table pool */
    uint64_t table_offsets;         /* u32 per context */
    uint64_t group_weights;         /* plain: weights per group; blocks run group by group */
    uint64_t blocks_per_group;
    uint64_t block_weights;         /* 2^18, or R * cols for match strips (R-row blocks) */
    uint64_t row_scale;             /* delta: i8 per row */
    uint64_t base;                  /* delta: base tensor (BF16, same shape) */
    uint64_t units;                 /* match: U */
    uint64_t unit_len;              /* match: H */
    uint32_t axis;                  /* match: 0 row units, 1 column units */
    uint32_t zcount;                /* match: classes * 12 */
    uint64_t unit_records;          /* match: u32 per unit */
    uint64_t partner_offsets;       /* match: u32 per unit + 1 */
    uint64_t partners;              /* match: u64 per partner */
    uint64_t out;                   /* BF16 output (STORE) */
    uint64_t x;                     /* fp32 input vector [cols] (MATVEC) */
    uint64_t y;                     /* fp32 output [rows] (MATVEC; whole rows of a block) */
    uint64_t partials;              /* fp32 [block_count][2]: first / last split row */
    uint64_t status;                /* u32 per block */
    uint32_t source_count;
    uint32_t pad;
    ingot_cuda_source source[INGOT_CUDA_MAX_SOURCES];
} ingot_cuda_params;

/**
 * Weight range [lo, hi) of block b: plain blocks run group by group (2^18 each, the last of a
 * group may be short); delta and match blocks are flat (2^18, or R-row strips).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
INGOT_CUDA_FN void ingot_cuda_block_range(const ingot_cuda_params *p, uint64_t b, uint64_t *lo,
                                          uint64_t *hi) {
    const uint64_t block = p->block_weights;
    uint64_t g = b / p->blocks_per_group, local = b % p->blocks_per_group;
    uint64_t end = (g + 1) * p->group_weights;
    *lo = g * p->group_weights + local * block;
    *hi = *lo + block < end ? *lo + block : end;
}

/* ---- rANS lookup ---------------------------------------------------------------------------- */

/**
 * Find the symbol of `slot` in a table image: *start, *freq and the symbol (returned).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
INGOT_CUDA_FN uint32_t ingot_cuda_table_find(const uint8_t *table, uint32_t slot,
                                             uint32_t *start, uint32_t *freq) {
    const uint16_t *coarse = (const uint16_t *)table;
    const unsigned long long *entry =
        (const unsigned long long *)(table + INGOT_CUDA_TABLE_HEADER);
    uint32_t r = INGOT_CUDA_LOAD(coarse + (slot >> INGOT_CUDA_COARSE_SHIFT));
    unsigned long long e = INGOT_CUDA_LOAD(entry + r);
    while (slot - (uint32_t)(e & 0xFFFFu) > (uint32_t)((e >> 16) & 0xFFFFu))
        e = INGOT_CUDA_LOAD(entry + ++r);
    *start = (uint32_t)(e & 0xFFFFu);
    *freq = (uint32_t)((e >> 16) & 0xFFFFu) + 1u;
    return (uint32_t)(e >> 32);
}

/**
 * One rANS decode step on state *x with a table image; returns the symbol. Renormalization is
 * the caller's (the word order is shared by the lanes): *x < 2^16 afterwards means a word is due.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
INGOT_CUDA_FN uint32_t ingot_cuda_rans_step(uint32_t *x, const uint8_t *table) {
    uint32_t slot = *x & 0xFFFFu, start = 0, freq = 0;
    uint32_t s = ingot_cuda_table_find(table, slot, &start, &freq);
    *x = freq * (*x >> 16) + slot - start;
    return s;
}

/* ---- Raw bits ------------------------------------------------------------------------------- */

/**
 * nbits (0..25) bits at bit offset `bit` of a raw stream starting 4-aligned (LSB-first). May read
 * up to 7 bytes past the last bit (the device image keeps slack after every block area).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
INGOT_CUDA_FN uint32_t ingot_cuda_raw_bits(const uint8_t *raw, uint64_t bit, uint32_t nbits) {
    const uint32_t *w = (const uint32_t *)raw + (bit >> 5);
    uint64_t pair = (uint64_t)INGOT_CUDA_LOAD(w) | ((uint64_t)INGOT_CUDA_LOAD(w + 1) << 32);
    return (uint32_t)(pair >> (bit & 31)) & ((1u << nbits) - 1u);
}

/* ---- BF16 helpers --------------------------------------------------------------------------- */

/** BF16 bits as fp32. */
INGOT_CUDA_FN float ingot_cuda_bf16_f32(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = bits << 16;
    return v.f;
}

/** Monotone BF16 order (+0 -> 0, -0 -> -1), as ingot_bf16_ordered. */
INGOT_CUDA_FN int32_t ingot_cuda_bf16_ordered(uint32_t u) {
    int32_t mag = (int32_t)(u & 0x7FFFu);
    return (u & 0x8000u) ? -mag - 1 : mag;
}

/** Inverse of ingot_cuda_bf16_ordered (v in [-32768, 32767]). */
INGOT_CUDA_FN uint32_t ingot_cuda_bf16_from_ordered(int32_t v) {
    return v < 0 ? (0x8000u | (uint32_t)(-(v + 1))) : (uint32_t)v;
}

/** Step code (ingot_step.h): raw bits after a symbol, and the signed value. */
INGOT_CUDA_FN uint32_t ingot_cuda_step_raw_bits(uint32_t symbol) {
    return symbol < 7 ? 0u : (symbol - 7) / 8;
}

INGOT_CUDA_FN int32_t ingot_cuda_step_join(uint32_t symbol, uint32_t raw) {
    uint32_t mag, sign;
    if (symbol == 0)
        return 0;
    if (symbol <= 2) {
        mag = 1;
        sign = symbol - 1;
    } else if (symbol <= 6) {
        sign = (symbol - 3) >> 1;
        mag = 2 | ((symbol - 3) & 1u);
    } else {
        uint32_t v = symbol - 7;
        sign = (v >> 2) & 1u;
        mag = ((4u | (v & 3u)) << (v / 8)) | raw;
    }
    return sign ? -(int32_t)mag : (int32_t)mag;
}

/* ---- Predictor ------------------------------------------------------------------------------ */

/**
 * q * ingot_predict_fixed(x, shift) exactly (|result| < 2^56), with one 32-bit multiply.
 * |shift| <= 2048, |q| <= 32767.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
INGOT_CUDA_FN int64_t ingot_cuda_partner_term(uint32_t x, int32_t shift, int32_t q) {
    uint32_t exp = (x >> 7) & 0xFFu;
    int32_t m = (int32_t)(exp ? (x & 0x7Fu) | 0x80u : x & 0x7Fu);
    int32_t e = (int32_t)(exp ? exp : 1u) - 134 + shift;
    int32_t a = 0, s = 0;
    int sat = m != 0 && (exp == 0xFFu || (e >= 33 && (e >= 40 || m > (1 << (40 - e)))));
    if (sat) {
        a = q;
        s = 40;
    } else if (e >= 0) {
        a = q * m;
        s = e;
    } else if (e > -8) {
        a = q * (m >> -e);
    }
    a = (x & 0x8000u) ? -a : a;
    return (int64_t)((uint64_t)(int64_t)a << s);
}

/**
 * Unit thresholds as (sh, rsel): T_k = floor(B_k 2^sh) with B_k = t_k R[rsel] (see the file
 * comment); sh clamped to +-INGOT_CUDA_SHIFT_CLAMP (the counts do not change past it).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
INGOT_CUDA_FN int32_t ingot_cuda_unit_shift(int32_t scale, int32_t c) {
    int32_t biased = scale + 128;
    int32_t sh = (biased >> 2) - 32 + c - 24;
    return sh > INGOT_CUDA_SHIFT_CLAMP    ? INGOT_CUDA_SHIFT_CLAMP
           : sh < -INGOT_CUDA_SHIFT_CLAMP ? -INGOT_CUDA_SHIFT_CLAMP
                                          : sh;
}

INGOT_CUDA_FN uint32_t ingot_cuda_unit_rsel(int32_t scale) { return (uint32_t)(scale + 128) & 3u; }

/** B_k = t_k (1/256 units) * 2^(rsel/4) (Q16). */
INGOT_CUDA_FN uint32_t ingot_cuda_threshold_base(uint32_t rsel, uint32_t k) {
    const uint32_t t = k == 0 ? 16 : k == 1 ? 32 : k == 2 ? 64 : k == 3 ? 96 : k == 4 ? 128
                     : k == 5 ? 192 : k == 6 ? 256 : k == 7 ? 320 : k == 8 ? 384 : k == 9 ? 512
                                                                                     : 768;
    const uint32_t r = rsel == 0 ? 65536u : rsel == 1 ? 77936u : rsel == 2 ? 92682u : 110218u;
    return t * r;
}

/** Bucket table cell: the number of B_k <= cell << 16. */
INGOT_CUDA_FN uint32_t ingot_cuda_bucket_cell(uint32_t rsel, uint32_t cell) {
    uint32_t n = 0;
    for (uint32_t k = 0; k < INGOT_CUDA_THRESHOLDS; k++)
        n += ingot_cuda_threshold_base(rsel, k) <= (cell << 16);
    return n;
}

/**
 * The 31-bit v with |p| >= T_k  <=>  v >= B_k for every k (B_k < 2^27).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
INGOT_CUDA_FN uint32_t ingot_cuda_bucket_value(uint64_t mag, int32_t sh) {
    uint64_t t;
    if (sh >= 0)
        t = sh >= 58 ? 0 : mag >> sh;
    else if (-sh >= 28 || mag >= 0x80000000ull)
        t = 0x7FFFFFFFull;
    else
        t = ((mag + 1) << -sh) - 1;
    return t > 0x7FFFFFFFull ? 0x7FFFFFFFu : (uint32_t)t; /* below the 0xFFFFFFFF sentinel */
}

/**
 * Bucket (0..11) of |p| for a unit (sh, rsel), given the cell table (cells[rsel][2048], u8) and
 * the bases (bases[rsel][12], u32, bases[r][11] = 0xFFFFFFFF).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
INGOT_CUDA_FN uint32_t ingot_cuda_bucket(uint64_t mag, int32_t sh, uint32_t rsel,
                                         const uint8_t *cells, const uint32_t *bases) {
    uint32_t v = ingot_cuda_bucket_value(mag, sh);
    uint32_t cell = v >> 16;
    cell = cell < INGOT_CUDA_BUCKET_CELLS ? cell : INGOT_CUDA_BUCKET_CELLS - 1;
    uint32_t k = cells[rsel * INGOT_CUDA_BUCKET_CELLS + cell];
#ifdef INGOT_CUDA_RED_BUCKET
    return k + (v > bases[rsel * 12 + k]); /* planted red: a value equal to a base lands low */
#else
    return k + (v >= bases[rsel * 12 + k]);
#endif
}

/** Unit record fields (match). */
INGOT_CUDA_FN uint32_t ingot_cuda_unit_zbase(uint32_t rec) { return rec & 0xFFu; }
INGOT_CUDA_FN uint32_t ingot_cuda_unit_rsel_of(uint32_t rec) { return (rec >> 8) & 3u; }
INGOT_CUDA_FN int32_t ingot_cuda_unit_sh(uint32_t rec) { return (int32_t)((rec >> 16) & 0xFFu) - 64; }
INGOT_CUDA_FN uint32_t ingot_cuda_unit_k(uint32_t rec) { return (rec >> 24) & 0xFu; }

INGOT_CUDA_FN uint32_t ingot_cuda_unit_record(uint32_t zbase, uint32_t rsel, int32_t sh,
                                              uint32_t k) {
    return zbase | rsel << 8 | (uint32_t)(sh + 64) << 16 | k << 24;
}

/* ---- Blocks --------------------------------------------------------------------------------- */

/** One block of the area (ingot_block.h layout). */
typedef struct {
    const uint8_t *raw;
    const uint16_t *words;
    const uint32_t *states;
    uint32_t raw_bytes;
    uint32_t word_count;
} ingot_cuda_block;

INGOT_CUDA_FN ingot_cuda_block ingot_cuda_block_at(const ingot_cuda_params *p, uint64_t b) {
    const uint8_t *area = (const uint8_t *)(uintptr_t)p->area;
    const uint64_t *offsets = (const uint64_t *)area;
    const uint8_t *blk = area + 8 * (p->block_count + 1) + offsets[b];
    ingot_cuda_block v;
    v.raw_bytes = ((const uint32_t *)blk)[0];
    v.word_count = ((const uint32_t *)blk)[1];
    v.states = (const uint32_t *)(blk + 8);
    v.raw = blk + 8 + 4 * INGOT_CUDA_LANES;
    v.words = (const uint16_t *)(v.raw + ((v.raw_bytes + 1u) & ~1u));
    return v;
}

/** Table image of context cx, or NULL for an empty context. */
INGOT_CUDA_FN const uint8_t *ingot_cuda_table(const ingot_cuda_params *p, uint32_t cx) {
    uint32_t off = INGOT_CUDA_LOAD((const uint32_t *)(uintptr_t)p->table_offsets + cx);
    return off == INGOT_CUDA_TABLE_EMPTY ? (const uint8_t *)0
                                         : (const uint8_t *)(uintptr_t)p->tables + off;
}

/* ---- Matvec rows ---------------------------------------------------------------------------- */

/** 1 when row r lies wholly inside the block [lo, hi). */
INGOT_CUDA_FN int ingot_cuda_row_whole(uint64_t r, uint64_t cols, uint64_t lo, uint64_t hi) {
    return r * cols >= lo && (r + 1) * cols <= hi;
}

/**
 * Store one row sum of block b: y[r] for a whole row, else the block's partial slot (0 for its
 * first row, 1 for its last).
 */
INGOT_CUDA_FN void ingot_cuda_row_store(const ingot_cuda_params *p, uint64_t b, uint64_t lo,
                                        uint64_t hi, uint64_t r, float sum) {
    if (ingot_cuda_row_whole(r, p->cols, lo, hi))
        ((float *)(uintptr_t)p->y)[r] = sum;
    else
        ((float *)(uintptr_t)p->partials)[2 * b + (r == lo / p->cols ? 0 : 1)] = sum;
}

/**
 * Add the split rows' partials into y, block by block in order (one thread; the fixed order
 * makes y identical on every run).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: yes
 */
INGOT_CUDA_FN void ingot_cuda_combine_rows(const ingot_cuda_params *p) {
    float *y = (float *)(uintptr_t)p->y;
    const float *part = (const float *)(uintptr_t)p->partials;
    uint64_t prev = ~(uint64_t)0;
    for (uint64_t b = 0; b < p->block_count; b++) {
        uint64_t lo, hi;
        ingot_cuda_block_range(p, b, &lo, &hi);
        uint64_t first = lo / p->cols, last = (hi - 1) / p->cols;
        for (int slot = 0; slot < 2; slot++) {
            uint64_t r = slot ? last : first;
            if ((slot && last == first) || ingot_cuda_row_whole(r, p->cols, lo, hi))
                continue;
            y[r] = r == prev ? y[r] + part[2 * b + slot] : part[2 * b + slot];
            prev = r;
        }
    }
}

/* ---- Match contexts ------------------------------------------------------------------------- */

/**
 * Context z (and the sign of p in *neg) of the weight at unit u, position h. `sources` holds the
 * descriptors (shared memory on the device); `cells` and `bases` the bucket tables.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
INGOT_CUDA_FN uint32_t ingot_cuda_match_context(const ingot_cuda_params *p,
                                                const ingot_cuda_source *sources,
                                                const uint8_t *cells, const uint32_t *bases,
                                                uint64_t u, uint64_t h, uint32_t *neg) {
    uint32_t rec = INGOT_CUDA_LOAD((const uint32_t *)(uintptr_t)p->unit_records + u);
    uint32_t off = INGOT_CUDA_LOAD((const uint32_t *)(uintptr_t)p->partner_offsets + u);
    const unsigned long long *part = (const unsigned long long *)(uintptr_t)p->partners + off;
    uint32_t k = ingot_cuda_unit_k(rec);
    int64_t acc = 0;
    for (uint32_t j = 0; j < k; j++) {
        unsigned long long pr = INGOT_CUDA_LOAD(part + j);
        const ingot_cuda_source *s = &sources[(uint32_t)(pr >> 48)];
        uint32_t at = (uint32_t)pr * s->unit_stride + (uint32_t)h * s->pos_stride; /* < 2^32 elements */
        const uint16_t *ptr = (const uint16_t *)(uintptr_t)s->ptr + at;
        uint32_t x = s->self ? *(const volatile uint16_t *)ptr : INGOT_CUDA_LOAD(ptr);
        acc += ingot_cuda_partner_term(x, s->shift, (int32_t)(int16_t)(uint16_t)(pr >> 32));
    }
    *neg = acc < 0;
    uint64_t mag = acc < 0 ? (uint64_t)0 - (uint64_t)acc : (uint64_t)acc;
    return ingot_cuda_unit_zbase(rec) +
           ingot_cuda_bucket(mag, ingot_cuda_unit_sh(rec), ingot_cuda_unit_rsel_of(rec), cells,
                             bases);
}

/**
 * Fill the bucket tables: cells[4][2048] (u8) and bases[4][12] (u32, the 12th = 0xFFFFFFFF), by
 * `stride` workers starting at `first` (one worker: 0, 1).
 *
 * Thread-safe: yes (disjoint workers)   Reentrant: yes   Signal-safe: yes
 */
INGOT_CUDA_FN void ingot_cuda_bucket_tables(uint8_t *cells, uint32_t *bases, uint32_t first,
                                            uint32_t stride) {
    for (uint32_t i = first; i < INGOT_CUDA_BUCKET_SETS * INGOT_CUDA_BUCKET_CELLS; i += stride)
        cells[i] = (uint8_t)ingot_cuda_bucket_cell(i / INGOT_CUDA_BUCKET_CELLS,
                                                   i % INGOT_CUDA_BUCKET_CELLS);
    for (uint32_t i = first; i < INGOT_CUDA_BUCKET_SETS * 12; i += stride)
        bases[i] = i % 12 == 11 ? 0xFFFFFFFFu : ingot_cuda_threshold_base(i / 12, i % 12);
}

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_CUDA_DEVICE_H */
