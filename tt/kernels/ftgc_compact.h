/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ftgc_compact.h
 * @brief L11b: the chunkwise GDN's per-head decay-ratio tile and five gate runs, built ON THE
 *        CARD from `decay[rows]` / `beta[rows]` (the gdnc reader's COMPACT staging) instead of
 *        uploaded whole from the host (4 KiB tile + 640 B a head).
 *
 * BIT-IDENTICAL TO THE HOST BUILDERS by construction: `ft_tt_op_gdnc_build_decay(_at)` and
 * `ft_tt_op_gdnc_build_gate(_at)` are product recurrences — multiplications only — so this
 * file replays the same multiplications in the same order with `ftgc_mul`, an integer fp32
 * multiply that reproduces x86 SSE `mulss` under the host's FTZ+DAZ (-ffast-math links
 * crtfastmath: denormal inputs read as zero, a result tiny AFTER rounding is flushed to
 * zero), round to nearest even. `scale` is the host's own `1/sqrt(S)` bits, never recomputed.
 * tests/test_gdnc_compact.c (Pantheon) holds both claims against the hardware and the builders.
 *
 * Integer-only on purpose: the reader's RISC-V core has no FPU, and libgcc's soft-float
 * produces denormals (not the host's flush) and costs 0.7 KiB of a 12 KiB text slab.
 */
#ifndef FTGC_COMPACT_H
#define FTGC_COMPACT_H

#include <stdint.h>

#define FTGC_COMPACT_ONE 0x3F800000u

/** Floats of one head's compact run: decay[32] then beta[32]. */
#define FTGC_COMPACT_HEAD_WORDS 64u
/** The compact block's header (scale bits at word 0), ahead of the heads; 64 B. */
#define FTGC_COMPACT_HEADER_WORDS 16u

/** a*b on fp32 bits, x86 SSE semantics under FTZ+DAZ, round to nearest even. No inf/NaN.
 *
 *  32-bit only (rv32im: one `mul`, one `mulhu`): the mantissas go in pre-shifted by 8 so the
 *  high word of the product carries the 24-bit result plus 7 or 8 round bits, and the low word
 *  is only a sticky bit. ~20 instructions; the first spelling (a uint64_t product and 64-bit
 *  shifts) cost the reader 0.33 s of prefill a 1,629-row prompt (measured 2026-10-06). */
#define FTGC_INLINE static inline __attribute__((always_inline))

FTGC_INLINE uint32_t ftgc_mul(uint32_t a, uint32_t b)
{
    const uint32_t sign = (a ^ b) & 0x80000000u;
    const uint32_t ea = (a >> 23) & 0xFFu, eb = (b >> 23) & 0xFFu;
    uint32_t ma, mb, hi, lo, mant, round, half;
    int32_t e;

    if (ea == 0u || eb == 0u) /* zero, or a denormal read as zero (DAZ) */
        return sign;
    ma = (a << 8) | 0x80000000u; /* 1.m in [2^31, 2^32) */
    mb = (b << 8) | 0x80000000u;
    hi = (uint32_t)(((uint64_t)ma * mb) >> 32); /* mulhu: [2^30, 2^32) */
    lo = ma * mb;                               /* mul: the sticky bits */
    e = (int32_t)ea + (int32_t)eb - 127;
    if (hi & 0x80000000u) { /* product in [2, 4): 24 bits at [31:8], 8 round bits */
        e += 1;
        mant = hi >> 8;
        round = hi & 0xFFu;
        half = 0x80u;
    } else {                /* [1, 2): 24 bits at [30:7], 7 round bits */
        mant = hi >> 7;
        round = hi & 0x7Fu;
        half = 0x40u;
    }
    if (round > half || (round == half && (lo != 0u || (mant & 1u))))
        mant++;
    if (mant == 0x01000000u) {
        mant >>= 1;
        e += 1;
    }
    if (e <= 0) /* tiny after rounding: flushed (FTZ) */
        return sign;
    if (e >= 255)
        return sign | 0x7F800000u;
    return sign | ((uint32_t)e << 23) | (mant & 0x007FFFFFu);
}

/** Float index of element [row][column] inside a 32x32 tile's face layout. */
FTGC_INLINE uint32_t ftgc_tile_index(uint32_t row, uint32_t column)
{
    return ((row >> 4) * 2u + (column >> 4)) * 256u + (row & 15u) * 16u + (column & 15u);
}

/**
 * The decay-ratio tile, face layout, all 1024 words written. `decay[i]` is position
 * `first_row + i`'s decay (fp32 bits). `first_row == 0` is `ft_tt_op_gdnc_build_decay()`
 * (dead columns step by 1.0), otherwise `ft_tt_op_gdnc_build_decay_at()` (rows outside the
 * sub-chunk all zero).
 */
static inline void ftgc_compact_decay_tile(const volatile uint32_t *decay, uint32_t first_row,
                                           uint32_t chunk_rows, volatile uint32_t *tile)
{
    uint32_t row, column;

    if (first_row == 0u) {
        /* Every word written exactly once: above the diagonal zero, the rest the recurrence.
         * The walk LEFT from the diagonal is two runs of consecutive words — the right face's
         * columns (16..row) then the left face's (15..0) — so the index is a pointer
         * decrement, not a face computation a word (the kernel builds at -Os, which left
         * `ftgc_tile_index` and `ftgc_mul` as calls: L11b's +0.33 s).
         *
         * ROWS AT AND ABOVE `chunk_rows` ARE NOT RECOMPUTED. Their steps at columns >= n are
         * 1.0 (exact under FTZ: a normal or zero times 1.0 is itself), so row r >= n holds 1.0
         * at columns n-1..r and then walks from column n-1 EXACTLY as row n-1 walks from its
         * diagonal: columns 0..n-2 are row n-1's, bit for bit. At C = 16 that is 120
         * multiplies a tile instead of 496. */
        const uint32_t live = chunk_rows;

        for (row = 0; row < 32u; row++) {
            volatile uint32_t *left = tile + ftgc_tile_index(row, 0u);   /* columns 0..15 */
            volatile uint32_t *right = tile + ftgc_tile_index(row, 16u); /* columns 16..31 */
            uint32_t value = FTGC_COMPACT_ONE;

            for (column = row + 1u; column < 16u; column++)
                left[column] = 0u;
            for (column = (row + 1u > 16u ? row + 1u : 16u); column < 32u; column++)
                right[column - 16u] = 0u;
            if (row >= live) {
                const volatile uint32_t *src_left = tile + ftgc_tile_index(live - 1u, 0u);
                const volatile uint32_t *src_right = tile + ftgc_tile_index(live - 1u, 16u);

                for (column = live - 1u; column <= row; column++) {
                    if (column < 16u)
                        left[column] = FTGC_COMPACT_ONE;
                    else
                        right[column - 16u] = FTGC_COMPACT_ONE;
                }
                for (column = 0; column + 1u < live; column++) {
                    if (column < 16u)
                        left[column] = src_left[column];
                    else
                        right[column - 16u] = src_right[column - 16u];
                }
                continue;
            }
            /* row < live: every step is a real decay (column <= row <= live - 1). */
            if (row >= 16u) {
                right[row - 16u] = value;
                for (column = row; column > 16u; column--) {
                    value = ftgc_mul(value, decay[column]);
                    right[column - 17u] = value;
                }
                value = ftgc_mul(value, decay[16]);
                left[15] = value;
                column = 15u;
            } else {
                left[row] = value;
                column = row;
            }
            for (; column > 0u; column--) {
                value = ftgc_mul(value, decay[column]);
                left[column - 1u] = value;
            }
        }
        return;
    }
    for (row = 0; row < 32u; row++)
        for (column = 0; column < 32u; column++)
            tile[ftgc_tile_index(row, column)] = 0u;
    for (row = first_row; row < first_row + chunk_rows; row++) {
        uint32_t value = FTGC_COMPACT_ONE;

        tile[ftgc_tile_index(row, row)] = value;
        for (column = row; column > first_row; column--) {
            value = ftgc_mul(value, decay[column - first_row]);
            tile[ftgc_tile_index(row, column - 1u)] = value;
        }
    }
}

/** Gate run indices, mirroring FT_TT_OP_GDNC_GATE_* (P, PS, B, F, G). */
enum { kFtgcGateP = 0, kFtgcGatePS = 1, kFtgcGateB = 2, kFtgcGateF = 3, kFtgcGateG = 4 };

/**
 * The five gate runs (5 x 32 words, run-major) exactly as `ft_tt_op_gdnc_build_gate(_at)`
 * writes them. `scale` is the host's `1/sqrt(state_size)` bits.
 */
static inline void ftgc_compact_gate(const volatile uint32_t *decay, const volatile uint32_t *beta,
                                     uint32_t first_row, uint32_t chunk_rows, uint32_t scale,
                                     volatile uint32_t *out)
{
    volatile uint32_t *p = out + kFtgcGateP * 32u, *ps = out + kFtgcGatePS * 32u,
                      *b = out + kFtgcGateB * 32u, *f = out + kFtgcGateF * 32u,
                      *g = out + kFtgcGateG * 32u;
    uint32_t row, last;

    if (first_row == 0u) {
        for (row = 0; row < 32u; row++) {
            const uint32_t step = (row < chunk_rows) ? decay[row] : FTGC_COMPACT_ONE;

            /* x * 1.0 is x exactly (FTZ: x is normal or zero), so dead rows copy. */
            p[row] = (row == 0u) ? step : (row < chunk_rows ? ftgc_mul(p[row - 1u], step)
                                                            : p[row - 1u]);
        }
        last = p[chunk_rows - 1u];
        f[chunk_rows - 1u] = FTGC_COMPACT_ONE;
        for (row = chunk_rows - 1u; row > 0u; row--)
            f[row - 1u] = ftgc_mul(f[row], decay[row]);
        for (row = 0; row < 32u; row++) {
            ps[row] = (row < chunk_rows || row == 0u) ? ftgc_mul(scale, p[row]) : ps[row - 1u];
            b[row] = (row < chunk_rows) ? beta[row] : 0u;
            if (row >= chunk_rows)
                f[row] = FTGC_COMPACT_ONE;
            g[row] = last;
        }
        return;
    }
    {
        const uint32_t last_row = first_row + chunk_rows - 1u;

        for (row = 0; row < 32u; row++) {
            if (row < first_row || row > last_row) {
                p[row] = 0u;
                continue;
            }
            p[row] = (row == first_row) ? decay[0]
                                        : ftgc_mul(p[row - 1u], decay[row - first_row]);
        }
        last = p[last_row];
        for (row = 0; row < 32u; row++)
            f[row] = FTGC_COMPACT_ONE;
        for (row = last_row; row > first_row; row--)
            f[row - 1u] = ftgc_mul(f[row], decay[row - first_row]);
        for (row = 0; row < 32u; row++) {
            const int live = (row >= first_row && row <= last_row);

            ps[row] = live ? ftgc_mul(scale, p[row]) : 0u; /* scale > 0: scale * +0 = +0 */
            b[row] = live ? beta[row - first_row] : 0u;
            g[row] = last;
        }
    }
}

#endif /* FTGC_COMPACT_H */
