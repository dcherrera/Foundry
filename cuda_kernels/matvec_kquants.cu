/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   matvec_kquants.cu
 * @brief  sm_37 matvecs for repacked Q4_K/Q5_K/Q6_K/Q8_0/Q2_0_g128 and dense F32/F16/BF16 rows.
 *
 * Written from a behaviour specification; no ggml or
 * llama.cpp source was consulted.
 *
 * Every kernel is bit-exact by contract: the order and grouping of every rounding is
 * fixed (see the comments on each body), so a batched kernel reproduces the single-column
 * kernel per column. Single-column kernels are the batched bodies at NC = 1 with a
 * 2048-float x chunk; batched ones use a 512-float chunk per column.
 *
 * @status None
 * @issues Kept as-is on purpose: Q4_K is min-lifted while Q5_K is computed straight;
 *         batched K-quant/Q8_0 kernels address column c at c*512 (cols < 512 is out of
 *         bounds, never launched); partial trailing blocks are ignored; dense batched
 *         trees assume a power-of-two blockDim.x; an odd final row is computed twice
 *         (row 0 duplicated) and the copy discarded.
 * @todo   None
 */

#include "matvec_nx_staging.cuh"

/* Shared prefix (floats) of the single-column Q4_K kernel: one sum per 32 staged x. */
#define FC_KQ_XSUM_FLOATS 64
/* Per-column shared prefix of the batched Q4_K kernels: one sum per 32 staged x. */
#define FC_NX_XSUM_PER_COL 16

#define FC_KQ_THREADS 256
#define FC_KQ_SINGLE_CHUNK 2048
#define FC_KQ_SUPERBLOCK 256

#define FC_KQ_KIND_Q4K 0
#define FC_KQ_KIND_Q5K 1
#define FC_KQ_KIND_Q6K 2

#define FC_Q4K_BLOCK_BYTES 152
#define FC_Q5K_BLOCK_BYTES 184
#define FC_Q6K_BLOCK_BYTES 216
#define FC_Q80_BLOCK_BYTES 36
#define FC_Q2_0_BLOCK_BYTES 36

#define FC_DENSE_F32 0
#define FC_DENSE_F16 1
#define FC_DENSE_BF16 2

extern __shared__ __align__(16) float fc_kq_shared[];

/* ---------------------------------------------------------------------------------------
 * Shared building blocks
 * ------------------------------------------------------------------------------------- */

/* Warp sum, shuffle-down tree 16/8/4/2/1; only lane 0's result is meaningful. */
__device__ __forceinline__ float fc_kq_warp_sum(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v, 8);
    v += __shfl_down(v, 4);
    v += __shfl_down(v, 2);
    v += __shfl_down(v, 1);
    return v;
}

/* 2^e as a float; folds to a constant when e is a compile-time value. */
__device__ __forceinline__ float fc_kq_pow2(int e) {
    return __uint_as_float((unsigned)(127 + e) << 23);
}

/* Exact small-integer decode: OR the masked bits into the mantissa of 2^23, then one FMA
 * scales and re-biases. Saves the shift and the int->float convert per element. */
__device__ __forceinline__ float fc_kq_magic(unsigned word, unsigned mask, float mul, float add) {
    return __fmaf_rn(__uint_as_float(0x4B000000u | (word & mask)), mul, add);
}

/* (word >> pos) & 15, minus bias; pos <= 12 keeps the field inside the mantissa. */
__device__ __forceinline__ float fc_kq_nibble(unsigned word, int pos, float bias) {
    return fc_kq_magic(word, 0xFu << pos, fc_kq_pow2(-pos),
                       -(fc_kq_pow2(23 - pos) + bias));
}

/* 16 * ((word >> pos) & (mask >> pos)): Q5_K fifth bit or Q6_K high pair. */
__device__ __forceinline__ float fc_kq_high16(unsigned word, unsigned mask, int pos) {
    return fc_kq_magic(word, mask, fc_kq_pow2(4 - pos), -fc_kq_pow2(27 - pos));
}

/* Byte k of an 8-byte vector as a float (one byte-select convert). */
__device__ __forceinline__ float fc_kq_byte(uint2 v, int k) {
    const unsigned word = k < 4 ? v.x : v.y;
    return (float)((word >> (8 * (k & 3))) & 0xFFu);
}

/* Single-column x staging: a plain read-only copy of one chunk. */
__device__ __forceinline__ void fc_kq_stage_single(float *sx, const float *x, int cbase,
                                                   int cend) {
    for (int i = threadIdx.x; i < cend; i += FC_KQ_THREADS)
        sx[i] = __ldg(x + cbase + i);
}

/* Stage one chunk of NC columns; the single-column form when CHUNK is the 2048 chunk. */
template <int NC, int CHUNK>
__device__ __forceinline__ void fc_kq_stage(float *sx, const float *x, int col0, int nc,
                                            int cols, int cbase, int cend) {
    if (CHUNK == FC_KQ_SINGLE_CHUNK)
        fc_kq_stage_single(sx, x, cbase, cend);
    else
        FC_NX_STAGE_X(sx, x, col0, cols, cbase, cend, NC, nc, FC_NX_STAGE_KEEP);
}

/* Lane 0 writes column c of row r when the row exists and the column is live. */
template <int NC>
__device__ __forceinline__ void fc_kq_store_row(float *out, const float (&acc)[NC], int rows,
                                                int row, bool active, int col0, int nc) {
    #pragma unroll
    for (int c = 0; c < NC; c++) {
        const float sum = fc_kq_warp_sum(acc[c]);
        if ((threadIdx.x & 31) == 0 && active && c < nc)
            out[(size_t)(col0 + c) * rows + row] = sum;
    }
}

/* Both rows of a pair, column by column (keeps one column's sums live at a time). */
template <int NC>
__device__ __forceinline__ void fc_kq_store_rowpair(float *out, const float (&acc0)[NC],
                                                    const float (&acc1)[NC], int rows, int row,
                                                    bool active0, bool active1, int col0,
                                                    int nc) {
    #pragma unroll
    for (int c = 0; c < NC; c++) {
        const float sum0 = fc_kq_warp_sum(acc0[c]);
        const float sum1 = fc_kq_warp_sum(acc1[c]);
        if ((threadIdx.x & 31) == 0 && c < nc) {
            if (active0)
                out[(size_t)(col0 + c) * rows + row] = sum0;
            if (active1)
                out[(size_t)(col0 + c) * rows + row + 1] = sum1;
        }
    }
}

/* ---------------------------------------------------------------------------------------
 * Q4_K: min-lifted. Per lane and row, P += d * sum(sc*q*x) per super-block, and separately
 * M += (dmin*mn) * xsum over owned (super-block, sub-block) pairs; result W(P - M).
 * ------------------------------------------------------------------------------------- */

/* Sub-block activation sums of the staged chunk. The XOR visit order breaks shared bank
 * conflicts and fixes the summation order (result-relevant). Groups of 4 threads share
 * the shuffles, and the live test is uniform per group. */
template <int NC, int CHUNK>
__device__ __forceinline__ void fc_q4k_xsums(float *xs, const float *sx, int cend) {
    const unsigned groups_per_col = CHUNK / 32;
    const int quarter = threadIdx.x & 3;
    const unsigned live_groups = (unsigned)cend >> 5;

    for (unsigned gi = threadIdx.x >> 2; gi < NC * groups_per_col; gi += FC_KQ_THREADS / 4) {
        const unsigned group = gi & (groups_per_col - 1);
        if (group >= live_groups)
            continue;
        const float *src = sx + CHUNK * (gi / groups_per_col) + 32 * group + 8 * quarter;
        const unsigned swizzle = gi & 7;
        float sum = 0.0f;
        #pragma unroll
        for (unsigned k = 0; k < 8; k++)
            sum += src[k ^ swizzle];
        sum += __shfl_down(sum, 2);
        sum += __shfl_down(sum, 1);
        if (quarter == 0)
            xs[gi] = sum;
    }
}

/* Min half for one chunk. Lane l owns the pairs whose global index is l mod 32; the
 * rotation by cbase/32 keeps that rule for 512-float chunks (and is 0 for 2048 ones). */
template <int NC, int CHUNK>
__device__ __forceinline__ void fc_q4k_min_half(const unsigned char *b0, const unsigned char *b1,
                                                const float *xs, int nsb, int cbase, int lane,
                                                float (&m0)[NC], float (&m1)[NC]) {
    const int pi0 = (lane - (cbase >> 5)) & 31;
    const unsigned char *blk0 = b0 + (pi0 >> 3) * FC_Q4K_BLOCK_BYTES;
    const unsigned char *blk1 = b1 + (pi0 >> 3) * FC_Q4K_BLOCK_BYTES;
    const int sub = 136 + (pi0 & 7);

    /* At most two trips (one when batched): an unroll ladder would only cost registers. */
    #pragma unroll 1
    for (int pi = pi0; pi < 8 * nsb; pi += 32) {
        const float mn0 = __fmul_rn(__ldg((const float *)(blk0 + 148)), (float)__ldg(blk0 + sub));
        const float mn1 = __fmul_rn(__ldg((const float *)(blk1 + 148)), (float)__ldg(blk1 + sub));
        #pragma unroll
        for (int c = 0; c < NC; c++) {
            const float xsum = xs[c * (CHUNK / 32) + pi];
            m0[c] = __fmaf_rn(mn0, xsum, m0[c]);
            m1[c] = __fmaf_rn(mn1, xsum, m1[c]);
        }
        blk0 += 4 * FC_Q4K_BLOCK_BYTES;
        blk1 += 4 * FC_Q4K_BLOCK_BYTES;
    }
}

/* Scale half of one super-block: A = sum(F(sc)*F(q) * x), then P = fma(d, A, P). */
template <int NC, int CHUNK>
__device__ __forceinline__ void fc_q4k_superblock(const unsigned char *b0, const unsigned char *b1,
                                                  const float *sx, int lane,
                                                  float (&p0)[NC], float (&p1)[NC]) {
    const unsigned q0 = __ldg((const unsigned *)b0 + lane);
    const unsigned q1 = __ldg((const unsigned *)b1 + lane);
    const uint2 sc0 = __ldg((const uint2 *)(b0 + 128));
    const uint2 sc1 = __ldg((const uint2 *)(b1 + 128));
    const float d0 = __ldg((const float *)(b0 + 144));
    const float d1 = __ldg((const float *)(b1 + 144));
    const unsigned q0_hi = q0 >> 16, q1_hi = q1 >> 16;
    float a0[NC], a1[NC];

    #pragma unroll
    for (int c = 0; c < NC; c++)
        a0[c] = a1[c] = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const int s = 2 * j + h, pos = 8 * (j & 1) + 4 * h;
            const float t0 = fc_kq_byte(sc0, s) * fc_kq_nibble(j < 2 ? q0 : q0_hi, pos, 0.0f);
            const float t1 = fc_kq_byte(sc1, s) * fc_kq_nibble(j < 2 ? q1 : q1_hi, pos, 0.0f);
            #pragma unroll
            for (int c = 0; c < NC; c++) {
                const float xv = sx[c * CHUNK + 64 * j + 32 * h + lane];
                a0[c] = __fmaf_rn(t0, xv, a0[c]);
                a1[c] = __fmaf_rn(t1, xv, a1[c]);
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < NC; c++) {
        p0[c] = __fmaf_rn(d0, a0[c], p0[c]);
        p1[c] = __fmaf_rn(d1, a1[c], p1[c]);
    }
}

/* ---------------------------------------------------------------------------------------
 * Q5_K: straight form. t = ((d*sc)*q) - (dmin*mn), four roundings; A += t*x; P = P + A.
 * Scale/min/(d, dmin) are broadcast plain global loads on purpose.
 * ------------------------------------------------------------------------------------- */

template <int NC, int CHUNK>
__device__ __forceinline__ void fc_q5k_superblock(const unsigned char *b0, const unsigned char *b1,
                                                  const float *sx, int lane,
                                                  float (&p0)[NC], float (&p1)[NC]) {
    const unsigned q0 = __ldg((const unsigned *)b0 + lane);
    const unsigned q1 = __ldg((const unsigned *)b1 + lane);
    const unsigned h0 = __ldg(b0 + 128 + lane);
    const unsigned h1 = __ldg(b1 + 128 + lane);
    const uint2 sc0 = *(const uint2 *)(b0 + 160), sc1 = *(const uint2 *)(b1 + 160);
    const uint2 mn0 = *(const uint2 *)(b0 + 168), mn1 = *(const uint2 *)(b1 + 168);
    const float2 dd0 = *(const float2 *)(b0 + 176), dd1 = *(const float2 *)(b1 + 176);
    const unsigned q0_hi = q0 >> 16, q1_hi = q1 >> 16;
    float a0[NC], a1[NC];

    #pragma unroll
    for (int c = 0; c < NC; c++)
        a0[c] = a1[c] = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const int s = 2 * j + h, bit = 8 * j + 4 * h;
            const int pos = bit <= 16 ? bit : bit - 16;
            const float n0 = fc_kq_nibble(bit <= 16 ? q0 : q0_hi, pos, 0.0f) +
                             fc_kq_high16(h0, 1u << s, s);
            const float n1 = fc_kq_nibble(bit <= 16 ? q1 : q1_hi, pos, 0.0f) +
                             fc_kq_high16(h1, 1u << s, s);
            const float t0 = __fsub_rn(__fmul_rn(__fmul_rn(dd0.x, fc_kq_byte(sc0, s)), n0),
                                       __fmul_rn(dd0.y, fc_kq_byte(mn0, s)));
            const float t1 = __fsub_rn(__fmul_rn(__fmul_rn(dd1.x, fc_kq_byte(sc1, s)), n1),
                                       __fmul_rn(dd1.y, fc_kq_byte(mn1, s)));
            #pragma unroll
            for (int c = 0; c < NC; c++) {
                const float xv = sx[c * CHUNK + 64 * j + 32 * h + lane];
                a0[c] = __fmaf_rn(t0, xv, a0[c]);
                a1[c] = __fmaf_rn(t1, xv, a1[c]);
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < NC; c++) {
        p0[c] = __fadd_rn(p0[c], a0[c]);
        p1[c] = __fadd_rn(p1[c], a1[c]);
    }
}

/* ---------------------------------------------------------------------------------------
 * Q6_K: A += ((sc - 128) * (q - 32)) * x (exact product), then P = fma(d, A, P).
 * ------------------------------------------------------------------------------------- */

template <int NC, int CHUNK>
__device__ __forceinline__ void fc_q6k_superblock(const unsigned char *b0, const unsigned char *b1,
                                                  const float *sx, int lane,
                                                  float (&p0)[NC], float (&p1)[NC]) {
    const unsigned lo0 = __ldg((const unsigned *)b0 + lane);
    const unsigned lo1 = __ldg((const unsigned *)b1 + lane);
    const unsigned hi0 = __ldg((const unsigned short *)(b0 + 128) + lane);
    const unsigned hi1 = __ldg((const unsigned short *)(b1 + 128) + lane);
    const float d0 = __ldg((const float *)(b0 + 208));
    const float d1 = __ldg((const float *)(b1 + 208));
    const uint2 sc0 = __ldg((const uint2 *)(b0 + 192 + 8 * (lane >> 4)));
    const uint2 sc1 = __ldg((const uint2 *)(b1 + 192 + 8 * (lane >> 4)));
    float a0[NC], a1[NC];

    #pragma unroll
    for (int c = 0; c < NC; c++)
        a0[c] = a1[c] = 0.0f;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const unsigned w0 = half ? lo0 >> 16 : lo0, w1 = half ? lo1 >> 16 : lo1;
        #pragma unroll
        for (int quad = 0; quad < 4; quad++) {
            const int i = 4 * half + quad;
            const int pos = 8 * (quad & 1) + 4 * (quad >> 1), hpos = 8 * half + 2 * quad;
            const float fq0 = fc_kq_nibble(w0, pos, 32.0f) + fc_kq_high16(hi0, 3u << hpos, hpos);
            const float fq1 = fc_kq_nibble(w1, pos, 32.0f) + fc_kq_high16(hi1, 3u << hpos, hpos);
            const float t0 = __fmul_rn(fc_kq_byte(sc0, i) - 128.0f, fq0);
            const float t1 = __fmul_rn(fc_kq_byte(sc1, i) - 128.0f, fq1);
            #pragma unroll
            for (int c = 0; c < NC; c++) {
                const float xv = sx[c * CHUNK + 32 * i + lane];
                a0[c] = __fmaf_rn(t0, xv, a0[c]);
                a1[c] = __fmaf_rn(t1, xv, a1[c]);
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < NC; c++) {
        p0[c] = __fmaf_rn(d0, a0[c], p0[c]);
        p1[c] = __fmaf_rn(d1, a1[c], p1[c]);
    }
}

/* ---------------------------------------------------------------------------------------
 * K-quant row-pair driver: 8 warps x 2 rows, two independent chains per warp sharing one
 * shared-memory x read (Kepler dual-issue). An out-of-range second row re-reads row 0.
 * Warps with no live row still run every barrier, staging, x-sum and the warp sum.
 * ------------------------------------------------------------------------------------- */

template <int KIND, int NC, int CHUNK>
__device__ __forceinline__ void fc_kq_superblock(const unsigned char *b0, const unsigned char *b1,
                                                 const float *sx, int lane,
                                                 float (&p0)[NC], float (&p1)[NC]) {
    if (KIND == FC_KQ_KIND_Q4K)
        fc_q4k_superblock<NC, CHUNK>(b0, b1, sx, lane, p0, p1);
    else if (KIND == FC_KQ_KIND_Q5K)
        fc_q5k_superblock<NC, CHUNK>(b0, b1, sx, lane, p0, p1);
    else
        fc_q6k_superblock<NC, CHUNK>(b0, b1, sx, lane, p0, p1);
}

template <int KIND, int NC, int CHUNK>
__device__ __forceinline__ void fc_kq_rowpair_body(const unsigned char *w, const float *x,
                                                   float *out, int rows, int cols,
                                                   int row_stride, int col0, int nc) {
    const int block_bytes = KIND == FC_KQ_KIND_Q4K ? FC_Q4K_BLOCK_BYTES
                          : KIND == FC_KQ_KIND_Q5K ? FC_Q5K_BLOCK_BYTES : FC_Q6K_BLOCK_BYTES;
    const int xsum_per_col = CHUNK == FC_KQ_SINGLE_CHUNK ? FC_KQ_XSUM_FLOATS : FC_NX_XSUM_PER_COL;
    const int xsum_floats = KIND == FC_KQ_KIND_Q4K ? NC * xsum_per_col : 0;
    float *xs = fc_kq_shared;
    float *sx = fc_kq_shared + xsum_floats;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = 16 * blockIdx.x + 2 * warp;
    const bool active0 = r0 < rows, active1 = r0 + 1 < rows;
    const unsigned char *b0 = w + (size_t)r0 * (size_t)row_stride;
    const unsigned char *b1 = active1 ? b0 + (size_t)row_stride : b0;
    float p0[NC], p1[NC], m0[NC], m1[NC];

    #pragma unroll
    for (int c = 0; c < NC; c++)
        p0[c] = p1[c] = m0[c] = m1[c] = 0.0f;
    for (int cbase = 0; cbase < cols; cbase += CHUNK) {
        const int cend = min(CHUNK, cols - cbase);
        __syncthreads();
        fc_kq_stage<NC, CHUNK>(sx, x, col0, nc, cols, cbase, cend);
        __syncthreads();
        if (KIND == FC_KQ_KIND_Q4K) {
            fc_q4k_xsums<NC, CHUNK>(xs, sx, cend);
            __syncthreads();
        }
        if (!active0)
            continue;
        const int nsb = cend >> 8;
        if (KIND == FC_KQ_KIND_Q4K)
            fc_q4k_min_half<NC, CHUNK>(b0, b1, xs, nsb, cbase, lane, m0, m1);
        const float *xb = sx;
        for (int sb = 0; sb < nsb; sb++) {
            fc_kq_superblock<KIND, NC, CHUNK>(b0, b1, xb, lane, p0, p1);
            b0 += block_bytes;
            b1 += block_bytes;
            xb += FC_KQ_SUPERBLOCK;
        }
    }
    if (KIND == FC_KQ_KIND_Q4K) {
        #pragma unroll
        for (int c = 0; c < NC; c++) {
            p0[c] = __fsub_rn(p0[c], m0[c]);
            p1[c] = __fsub_rn(p1[c], m1[c]);
        }
    }
    fc_kq_store_rowpair<NC>(out, p0, p1, rows, r0, active0, active1, col0, nc);
}

/* ---------------------------------------------------------------------------------------
 * Q8_0: 8 lanes per row, 32 rows per 256-thread block (4 rows per warp).
 * The result is the warp-per-row form's, bit for bit: element e of every 32-weight block
 * still has its own chain t = d * q (own rounding), P = fma(t, x, P) in block order, and
 * the 32 chains meet in the same 16/8/4/2/1 tree. Lane s of a row's group carries chains
 * 4s..4s+3, so one 32-bit load brings 4 weights and one 128-bit shared load their 4 x
 * values (broadcast to the warp's 4 rows); a tile's lanes carry all of its columns, so
 * each weight is loaded and decoded once per tile. sm_37 was bound by one LDG per weight
 * plus one per scale, then by issue slots once all warps computed.
 * ------------------------------------------------------------------------------------- */

#define FC_Q80_LANES_PER_ROW 8
/* 256 threads / 8 lanes: every warp computes 4 rows. fc_cuda.c launches ceil(rows / 32)
 * blocks (FC_Q80_ROWS_PER_BLOCK_HOST must match). */
#define FC_Q80_ROWS_PER_BLOCK (FC_KQ_THREADS / FC_Q80_LANES_PER_ROW)
/* Blocks per unrolled step. Measured on sm_37 (single, 2-, 4- and 8-wide): 4 beats 2, 8
 * and 16 on the large shapes and is within 5% everywhere else. */
#define FC_Q80_UNROLL 4
/* Unroll of the G4 group loop (16 weights a lane a trip) for NC = 1 and NC > 1, and of the
 * row tile's group loop. Trip order and every chain are unchanged, so the result is the
 * same bits; the unrolled trip shares one loop branch and folds the pointer steps into
 * load offsets. Measured on sm_37 at the Qwen3.6 TP2 per-die shapes (bench_matvec_q80_ab,
 * run_q80_unroll_ab.sh): NC = 1 x4 1.05-1.21x (x2 1.04-1.19x; x8 ties x4 but needs more
 * registers), NC = 2 x2 1.00-1.13x (x4 no better), the row tiles x2 no gain. SASS of the
 * NC = 1 G4 loop: 5.56 instructions a weight at x1, 5.22 at x2, 5.08 at x4.
 * Override for an A/B build: make NVCCFLAGS_EXTRA=-DFC_Q80_G4_UNROLL=1. */
#ifndef FC_Q80_G4_UNROLL
#define FC_Q80_G4_UNROLL 4
#endif
#ifndef FC_Q80_G4_UNROLL_NX
#define FC_Q80_G4_UNROLL_NX 2
#endif
#ifndef FC_Q80_TILE_UNROLL
#define FC_Q80_TILE_UNROLL 1
#endif
/* #pragma unroll does not expand macros: the counts as constants. */
constexpr int fc_q80_g4_unroll_single = FC_Q80_G4_UNROLL;
constexpr int fc_q80_g4_unroll_multi = FC_Q80_G4_UNROLL_NX;
constexpr int fc_q80_tile_unroll = FC_Q80_TILE_UNROLL;

/* Byte m of a block word, as a float, exactly. The word arrives XOR 0x80808080, so byte m
 * is q + 128 in [0, 255]; placing it under exponent 0x4B gives the float 2^23 + q + 128,
 * and subtracting 2^23 + 128 is exact. One byte-permute and one add, no conversion. */
__device__ __forceinline__ float fc_q80_byte(unsigned biased_word, int m) {
    return __fsub_rn(__uint_as_float(__byte_perm(biased_word, 0x4B000000u, 0x7440 | m)),
                     8388736.0f);
}

/* The same value from the raw word with one conversion: inline PTX bfe.s32 + cvt.rn.f32.s32,
 * the only spelling ptxas 11.4 lowers to a single I2F.F32.S8 with a byte selector (a C cast
 * costs an extract first). m is a literal at every call, so the branches fold away.
 * FC_Q80_DECODE_I2F picks where it replaces fc_q80_byte: 0 nowhere, 1 the row tiles and the
 * NC > 1 blocks, 2 the NC = 1 blocks too. */
#ifndef FC_Q80_DECODE_I2F
#define FC_Q80_DECODE_I2F 0
#endif
#define FC_Q80_BFE_CVT(POS) "{ .reg .b32 r; bfe.s32 r, %1, " #POS ", 8; cvt.rn.f32.s32 %0, r; }"
__device__ __forceinline__ float fc_q80_byte_i2f(unsigned word, int m) {
    float f;
    if (m == 0)
        asm(FC_Q80_BFE_CVT(0) : "=f"(f) : "r"(word));
    else if (m == 1)
        asm(FC_Q80_BFE_CVT(8) : "=f"(f) : "r"(word));
    else if (m == 2)
        asm(FC_Q80_BFE_CVT(16) : "=f"(f) : "r"(word));
    else
        asm(FC_Q80_BFE_CVT(24) : "=f"(f) : "r"(word));
    return f;
}

/* d * byte m, by the decode FC_Q80_DECODE_I2F selects for NC columns. */
template <int NC>
__device__ __forceinline__ float fc_q80_scaled_byte(unsigned word, float d, int m) {
    if (FC_Q80_DECODE_I2F >= (NC == 1 ? 2 : 1))
        return __fmul_rn(d, fc_q80_byte_i2f(word, m));
    return __fmul_rn(d, fc_q80_byte(word ^ 0x80808080u, m));
}

/* One 32-weight block for this lane's 4 chains and NC columns: word is the lane's 4 bytes
 * of the block, d its scale. */
template <int NC, int CHUNK>
__device__ __forceinline__ void fc_q80_block(unsigned word, float d, const float *xp,
                                             float (&p)[NC][4]) {
    float t[4];

    #pragma unroll
    for (int m = 0; m < 4; m++)
        t[m] = fc_q80_scaled_byte<NC>(word, d, m);
    #pragma unroll
    for (int c = 0; c < NC; c++) {
        const float4 xv = *(const float4 *)(xp + c * CHUNK);
        p[c][0] = __fmaf_rn(t[0], xv.x, p[c][0]);
        p[c][1] = __fmaf_rn(t[1], xv.y, p[c][1]);
        p[c][2] = __fmaf_rn(t[2], xv.z, p[c][2]);
        p[c][3] = __fmaf_rn(t[3], xv.w, p[c][3]);
    }
}

/* nblk blocks of one staged chunk for NC columns, FC_Q80_UNROLL blocks per step.
 * lane_q = row_blk + 4 + 4s; xp = staged x + 4s. */
template <int NC, int CHUNK>
__device__ __forceinline__ void fc_q80_chunk(const unsigned char *row_blk,
                                             const unsigned char *lane_q, const float *xp,
                                             int nblk, float (&p)[NC][4]) {
    int b = 0;

    for (; b + FC_Q80_UNROLL <= nblk; b += FC_Q80_UNROLL) {
        #pragma unroll
        for (int j = 0; j < FC_Q80_UNROLL; j++)
            fc_q80_block<NC, CHUNK>(
                __ldg((const unsigned *)(lane_q + FC_Q80_BLOCK_BYTES * j)),
                __ldg((const float *)(row_blk + FC_Q80_BLOCK_BYTES * j)), xp + 32 * j, p);
        row_blk += FC_Q80_BLOCK_BYTES * FC_Q80_UNROLL;
        lane_q += FC_Q80_BLOCK_BYTES * FC_Q80_UNROLL;
        xp += 32 * FC_Q80_UNROLL;
    }
    #pragma unroll 1
    for (; b < nblk; b++) {
        fc_q80_block<NC, CHUNK>(__ldg((const unsigned *)lane_q), __ldg((const float *)row_blk),
                                 xp, p);
        row_blk += FC_Q80_BLOCK_BYTES;
        lane_q += FC_Q80_BLOCK_BYTES;
        xp += 32;
    }
}

/* nblk (a multiple of 4) blocks of one staged chunk in the G4 layout fc_repack_weight
 * writes when cols % 128 == 0: per 4 blocks (144 B), [f32 d0..d3] then lane s's 16 bytes
 * (bytes 4s..4s+3 of blocks 0..3). One 128-bit quant load and one 128-bit scale load per
 * lane per 4 blocks instead of 4 + 4 32-bit loads; same chains, same block order.
 * grp = row's first group of the chunk; xp = staged x + 4s. */
template <int NC, int CHUNK>
__device__ __forceinline__ void fc_q80_chunk_g4(const unsigned char *grp, int sub,
                                                const float *xp, int nblk,
                                                float (&p)[NC][4]) {
    #pragma unroll (NC == 1 ? fc_q80_g4_unroll_single : fc_q80_g4_unroll_multi)
    for (int b = 0; b < nblk; b += 4) {
        const uint4 q = __ldg((const uint4 *)(grp + 16 + 16 * sub));
        const float4 d = __ldg((const float4 *)grp);
        fc_q80_block<NC, CHUNK>(q.x, d.x, xp, p);
#ifdef FC_Q80_G4_RED
        /* Gate red build only: blocks 1 and 2 accumulated in swapped order. */
        fc_q80_block<NC, CHUNK>(q.z, d.z, xp + 64, p);
        fc_q80_block<NC, CHUNK>(q.y, d.y, xp + 32, p);
#else
        fc_q80_block<NC, CHUNK>(q.y, d.y, xp + 32, p);
        fc_q80_block<NC, CHUNK>(q.z, d.z, xp + 64, p);
#endif
        fc_q80_block<NC, CHUNK>(q.w, d.w, xp + 96, p);
        grp += 4 * FC_Q80_BLOCK_BYTES;
        xp += 128;
    }
}

/* The warp-per-row shuffle tree over 32 chains held 4 per lane: steps 16/8/4 are lane
 * shuffles inside the row's 8-lane group, steps 2/1 are register adds. Lane s == 0 writes. */
template <int NC>
__device__ __forceinline__ void fc_q80_store_row(float *out, float (&p)[NC][4], int rows,
                                                 int row, bool active, int col0, int nc) {
    #pragma unroll
    for (int c = 0; c < NC; c++) {
        float v[4];
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            v[m] = p[c][m];
            v[m] += __shfl_down(v[m], 4);
            v[m] += __shfl_down(v[m], 2);
            v[m] += __shfl_down(v[m], 1);
        }
        const float sum = (v[0] + v[2]) + (v[1] + v[3]);
        if ((threadIdx.x & (FC_Q80_LANES_PER_ROW - 1)) == 0 && active && c < nc)
            out[(size_t)(col0 + c) * rows + row] = sum;
    }
}

template <int NC, int CHUNK>
__device__ __forceinline__ void fc_q80_body(const unsigned char *w, const float *x, float *out,
                                            int rows, int cols, int row_stride, int col0,
                                            int nc) {
    float *sx = fc_kq_shared;
    const int lane = threadIdx.x & 31;
    const int sub = lane & (FC_Q80_LANES_PER_ROW - 1);
    const int row = FC_Q80_ROWS_PER_BLOCK * blockIdx.x + threadIdx.x / FC_Q80_LANES_PER_ROW;
    const bool active = row < rows;
    const unsigned char *row_blk = w + (long long)row * row_stride;
    const bool g4 = (cols & 127) == 0;   /* fc_repack_weight's G4 layout; CHUNKs are x128 */
    float p[NC][4];

    #pragma unroll
    for (int c = 0; c < NC; c++)
        p[c][0] = p[c][1] = p[c][2] = p[c][3] = 0.0f;
    for (int cbase = 0; cbase < cols; cbase += CHUNK) {
        const int cend = min(CHUNK, cols - cbase);
        __syncthreads();
        fc_kq_stage<NC, CHUNK>(sx, x, col0, nc, cols, cbase, cend);
        __syncthreads();
        if (!active)
            continue;
        const int nblk = cend >> 5;
        if (g4)
            fc_q80_chunk_g4<NC, CHUNK>(row_blk, sub, sx + 4 * sub, nblk, p);
        else
            fc_q80_chunk<NC, CHUNK>(row_blk, row_blk + 4 + 4 * sub, sx + 4 * sub, nblk, p);
        row_blk += nblk * FC_Q80_BLOCK_BYTES;
    }
    fc_q80_store_row<NC>(out, p, rows, row, active, col0, nc);
}

/* ---------------------------------------------------------------------------------------
 * Q8_0 prefill tile: R rows x NC columns per thread (G4 layout only).
 *
 * fc_q80_body at NC = 8 gives each thread ONE row: per 32-weight block a lane does 4 FMA
 * per 128-bit shared x load and decodes 4 weights per 32 FMA, and sm_37 runs it issue-
 * and LDS-bound (~0.7 TFLOP/s on the 128-token prefill chunk). Here row slot r of a block
 * (threadIdx.x / 8, 32 slots) carries rows r, r+32, .., r+32(R-1) of a 32R-row tile, so
 * one x load feeds 4R FMA and the block stages x once for 32R rows.
 *
 * BIT-EXACT with fc_q80_body per output: lane s still carries chains 4s..4s+3 of every
 * (row, column), each chain is the same fmaf sequence over the blocks in order (t = d*q
 * rounded, the G4 group loop), and fc_q80_store_row runs the same 4/2/1 shuffle tree and
 * (v0+v2)+(v1+v3) per row. Only which thread holds which outputs changes.
 * Requires cols % 128 == 0 (the launcher falls back otherwise). A row past `rows` reads
 * row 0 and is never stored.
 * ------------------------------------------------------------------------------------- */
template <int R, int NC, int CHUNK>
__device__ __forceinline__ void fc_q80_tile_body(const unsigned char *w, const float *x,
                                                 float *out, int rows, int cols,
                                                 int row_stride, int col0, int nc) {
    float *sx = fc_kq_shared;
    const int sub = threadIdx.x & (FC_Q80_LANES_PER_ROW - 1);
    const int slot = threadIdx.x / FC_Q80_LANES_PER_ROW;
    const int row0 = FC_Q80_ROWS_PER_BLOCK * R * blockIdx.x + slot;
    const unsigned char *grp[R];
    float p[R][NC][4];

    #pragma unroll
    for (int i = 0; i < R; i++) {
        const int row = row0 + FC_Q80_ROWS_PER_BLOCK * i;
        grp[i] = w + (long long)(row < rows ? row : 0) * row_stride;
        #pragma unroll
        for (int c = 0; c < NC; c++)
            p[i][c][0] = p[i][c][1] = p[i][c][2] = p[i][c][3] = 0.0f;
    }
    for (int cbase = 0; cbase < cols; cbase += CHUNK) {
        const int cend = min(CHUNK, cols - cbase);
        __syncthreads();
        /* Same staged values as fc_kq_stage (identity transform), the memory-parallel
         * loop form (matvec_nx_staging.cuh). */
        FC_NX_STAGE_X_EOUTER(sx, x, col0, cols, cbase, cend, NC, nc, FC_NX_STAGE_KEEP);
        __syncthreads();
        const float *xp = sx + 4 * sub;
        #pragma unroll fc_q80_tile_unroll
        for (int b = 0; b < (cend >> 5); b += 4) {
            uint4 q[R];
            float4 d[R];
            #pragma unroll
            for (int i = 0; i < R; i++) {
                q[i] = __ldg((const uint4 *)(grp[i] + 16 + 16 * sub));
                d[i] = __ldg((const float4 *)grp[i]);
                grp[i] += 4 * FC_Q80_BLOCK_BYTES;
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float t[R][4];
                #pragma unroll
                for (int i = 0; i < R; i++) {
                    const unsigned word = j == 0 ? q[i].x : j == 1 ? q[i].y
                                        : j == 2 ? q[i].z : q[i].w;
                    const float dj = j == 0 ? d[i].x : j == 1 ? d[i].y
                                   : j == 2 ? d[i].z : d[i].w;
                    #pragma unroll
                    for (int m = 0; m < 4; m++)
                        t[i][m] = fc_q80_scaled_byte<NC>(word, dj, m);
#ifdef FC_Q80_TILE_RED
                    /* Gate red build only: byte 0 read unsigned in the row tiles. */
                    t[i][0] = __fmul_rn(dj, (float)(word & 0xffu));
#endif
                }
                #pragma unroll
                for (int c = 0; c < NC; c++) {
                    const float4 xv = *(const float4 *)(xp + c * CHUNK + 32 * j);
                    #pragma unroll
                    for (int i = 0; i < R; i++) {
                        p[i][c][0] = __fmaf_rn(t[i][0], xv.x, p[i][c][0]);
                        p[i][c][1] = __fmaf_rn(t[i][1], xv.y, p[i][c][1]);
                        p[i][c][2] = __fmaf_rn(t[i][2], xv.z, p[i][c][2]);
                        p[i][c][3] = __fmaf_rn(t[i][3], xv.w, p[i][c][3]);
                    }
                }
            }
            xp += 128;
        }
    }
    #pragma unroll
    for (int i = 0; i < R; i++) {
        const int row = row0 + FC_Q80_ROWS_PER_BLOCK * i;
        fc_q80_store_row<NC>(out, p[i], rows, row, row < rows, col0, nc);
    }
}

#define FC_Q80_TILE(NAME, R, NC)                                                           \
    extern "C" __global__ void __launch_bounds__(FC_KQ_THREADS)                            \
    NAME(const unsigned char *w, const float *x, float *out, int rows, int cols,          \
         int row_stride, int n) {                                                          \
        const int col0 = (NC) * blockIdx.y;                                                \
        fc_q80_tile_body<R, NC, FC_NX_CHUNK>(w, x, out, rows, cols, row_stride, col0,      \
                                             min((NC), n - col0));                         \
    }

/* The same tile with __launch_bounds__(FC_KQ_THREADS, MINB): ptxas caps the registers so
 * MINB blocks fit a SMX. Same code, so the same bits. */
#define FC_Q80_TILE_MINB(NAME, R, NC, MINB)                                                \
    extern "C" __global__ void __launch_bounds__(FC_KQ_THREADS, MINB)                      \
    NAME(const unsigned char *w, const float *x, float *out, int rows, int cols,          \
         int row_stride, int n) {                                                          \
        const int col0 = (NC) * blockIdx.y;                                                \
        fc_q80_tile_body<R, NC, FC_NX_CHUNK>(w, x, out, rows, cols, row_stride, col0,      \
                                             min((NC), n - col0));                         \
    }

/* Shipped: r4c3 / r4c4 / r4c5 for batches of exactly 3 / 4 / 5 columns (the speculative
 * verify widths: the _nx tiles compute 4 or 8 columns there) and r4c8 for n >= 6, prefill
 * included (fc_cuda.c picks). The sweep set is for cuda/tests/bench_q80_tile.c only
 * (nvcc -DFC_Q80_TILE_SWEEP); measured on sm_37 none beat these, n = 2 stays on nx2, and
 * register prefetch of the next group did not help. */
#ifdef FC_Q80_TILE_SWEEP
FC_Q80_TILE(fc_matvec_q80_r1c8, 1, 8)
FC_Q80_TILE(fc_matvec_q80_r2c8, 2, 8)
FC_Q80_TILE(fc_matvec_q80_r2c12, 2, 12)
FC_Q80_TILE(fc_matvec_q80_r3c8, 3, 8)
FC_Q80_TILE(fc_matvec_q80_r4c1, 4, 1)
FC_Q80_TILE(fc_matvec_q80_r4c2, 4, 2)
FC_Q80_TILE(fc_matvec_q80_r4c6, 4, 6)
FC_Q80_TILE(fc_matvec_q80_r2c2, 2, 2)
FC_Q80_TILE(fc_matvec_q80_r2c3, 2, 3)
FC_Q80_TILE(fc_matvec_q80_r2c5, 2, 5)
FC_Q80_TILE(fc_matvec_q80_r8c2, 8, 2)
FC_Q80_TILE(fc_matvec_q80_r8c3, 8, 3)
#endif
FC_Q80_TILE(fc_matvec_q80_r4c3, 4, 3)
/* r4c4 at 3 blocks/SMX: 168 registers instead of 172 (16 B of spill), so 4096 rows (32
 * blocks) fit one wave of 13 SMX instead of two. 1.27x at 4096x2048, n = 4, on sm_37; the
 * smaller grids where it measured 0.98x run the _nx kernels (fc_cuda.c grid rule). */
FC_Q80_TILE_MINB(fc_matvec_q80_r4c4, 4, 4, 3)
#if FC_Q80_DECODE_I2F
/* The I2F decode takes r4c5 from 167 to 174 registers, 3 -> 2 blocks/SMX (0.83x at 4096
 * rows); held at 3 blocks. */
FC_Q80_TILE_MINB(fc_matvec_q80_r4c5, 4, 5, 3)
#else
FC_Q80_TILE(fc_matvec_q80_r4c5, 4, 5)
#endif
FC_Q80_TILE(fc_matvec_q80_r4c8, 4, 8)

/* ---------------------------------------------------------------------------------------
 * Q8_0 PARITY matvec: one half of a K-split that is bit-exact with the full kernel.
 *
 * fc_q80_store_row sums a row as (V0 + V2) + (V1 + V3), where V_m is the 4/2/1 lane tree
 * over the 8 chains 4s + m, i.e. over exactly the input elements k with k % 4 == m. So
 * rank p in {0, 1} of a two-die split can compute A_p = V_p + V_(p+2) from the elements
 * with k % 2 == p alone, with the same chains (t = d*q rounded, fmaf over the blocks in
 * order) and the same lane tree; the full result is A_0 + A_1, and IEEE add commutes, so
 * either die adding the other's partial gets the full kernel's output bit for bit.
 * (Tensor-parallel FFN down for Qwen3.8 on two K80 dies; the rank's gate/up rows are the
 * k % 2 == p rows, so its SwiGLU output is exactly this kernel's input.)
 *
 * Weight slab (fc_repack_weight_q80_parity): per row, per 4 blocks (128 elements of the
 * full row) 80 B: [f32 d0..d3] then lane s's 8 bytes q_b[4s+p], q_b[4s+p+2] for b = 0..3.
 * x is the rank's compact vector: element k = 2i + p of the full input is x[i], so lane s
 * reads x[16b + 2s] (chain 4s+p) and x[16b + 2s + 1] (chain 4s+p+2) for block b.
 * kh = full cols / 2 (a multiple of 64); out[c * rows + row] = A_p of that row, column c.
 * R rows x NC columns per thread, a block covers 32R rows; a row past `rows` reads row 0
 * and is never stored. CHUNK = compact floats staged per column.
 * ------------------------------------------------------------------------------------- */

#define FC_Q80P_GROUP_BYTES 80

template <int R, int NC, int CHUNK>
__device__ __forceinline__ void fc_q80p_body(const unsigned char *w, const float *x, float *out,
                                             int rows, int kh, int row_stride, int col0,
                                             int nc) {
    float *sx = fc_kq_shared;
    const int sub = threadIdx.x & (FC_Q80_LANES_PER_ROW - 1);
    const int slot = threadIdx.x / FC_Q80_LANES_PER_ROW;
    const int row0 = FC_Q80_ROWS_PER_BLOCK * R * blockIdx.x + slot;
    const unsigned char *grp[R];
    float p[R][NC][2];

    #pragma unroll
    for (int i = 0; i < R; i++) {
        const int row = row0 + FC_Q80_ROWS_PER_BLOCK * i;
        grp[i] = w + (long long)(row < rows ? row : 0) * row_stride;
        #pragma unroll
        for (int c = 0; c < NC; c++)
            p[i][c][0] = p[i][c][1] = 0.0f;
    }
    for (int cbase = 0; cbase < kh; cbase += CHUNK) {
        const int cend = min(CHUNK, kh - cbase);
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < NC; c++)
            for (int i = threadIdx.x; i < cend; i += FC_KQ_THREADS)
                sx[c * CHUNK + i] = c < nc ? x[(size_t)(col0 + c) * kh + cbase + i] : 0.0f;
        __syncthreads();
        const float *xp = sx + 2 * sub;
        for (int g = 0; g < cend; g += 64) {
            uint2 q[R];
            float4 d[R];
            #pragma unroll
            for (int i = 0; i < R; i++) {
                q[i] = __ldg((const uint2 *)(grp[i] + 16 + 8 * sub));
                d[i] = __ldg((const float4 *)grp[i]);
                grp[i] += FC_Q80P_GROUP_BYTES;
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float t[R][2];
                #pragma unroll
                for (int i = 0; i < R; i++) {
                    const unsigned word = j < 2 ? q[i].x : q[i].y;
                    const float dj = j == 0 ? d[i].x : j == 1 ? d[i].y
                                   : j == 2 ? d[i].z : d[i].w;
                    const unsigned biased = word ^ 0x80808080u;
                    t[i][0] = __fmul_rn(dj, fc_q80_byte(biased, 2 * (j & 1)));
                    t[i][1] = __fmul_rn(dj, fc_q80_byte(biased, 2 * (j & 1) + 1));
                }
                #pragma unroll
                for (int c = 0; c < NC; c++) {
                    const float2 xv = *(const float2 *)(xp + c * CHUNK + g + 16 * j);
                    #pragma unroll
                    for (int i = 0; i < R; i++) {
                        p[i][c][0] = __fmaf_rn(t[i][0], xv.x, p[i][c][0]);
                        p[i][c][1] = __fmaf_rn(t[i][1], xv.y, p[i][c][1]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < R; i++) {
        const int row = row0 + FC_Q80_ROWS_PER_BLOCK * i;
        #pragma unroll
        for (int c = 0; c < NC; c++) {
            float v[2];
            #pragma unroll
            for (int m = 0; m < 2; m++) {
                v[m] = p[i][c][m];
#ifdef FC_Q80P_RED
                /* Gate red build only: the tree in the wrong order (1/2/4). */
                v[m] += __shfl_down(v[m], 1);
                v[m] += __shfl_down(v[m], 2);
                v[m] += __shfl_down(v[m], 4);
#else
                v[m] += __shfl_down(v[m], 4);
                v[m] += __shfl_down(v[m], 2);
                v[m] += __shfl_down(v[m], 1);
#endif
            }
            const float sum = v[0] + v[1];
            if (sub == 0 && row < rows && c < nc)
                out[(size_t)(col0 + c) * rows + row] = sum;
        }
    }
}

#define FC_Q80P(NAME, R, NC, CHUNK)                                                        \
    extern "C" __global__ void __launch_bounds__(FC_KQ_THREADS)                            \
    NAME(const unsigned char *w, const float *x, float *out, int rows, int kh,            \
         int row_stride, int n) {                                                          \
        const int col0 = (NC) * blockIdx.y;                                                \
        fc_q80p_body<R, NC, CHUNK>(w, x, out, rows, kh, row_stride, col0,                  \
                                   min((NC), n - col0));                                   \
    }

/* Shipped: the tile fc_cuda.c picks per batch width (fc_q80p_tile_for), measured at the
 * Qwen3.8 FFN down shape on sm_37 (bench_q80_ksplit): r1c1 n = 1, r1c2 n = 2, r2c3 /
 * r2c4 / r2c5 n = 3 / 4 / 5, r2c8 n = 6-8, r4c8 n > 8 (prefill). The sweep set is for
 * the bench only (nvcc -DFC_Q80P_SWEEP); none of it beat the shipped tile at any n. */
#ifdef FC_Q80P_SWEEP
FC_Q80P(fc_matvec_q80p_r2c1, 2, 1, 2048)
FC_Q80P(fc_matvec_q80p_r2c2, 2, 2, 1024)
FC_Q80P(fc_matvec_q80p_r4c3, 4, 3, 512)
FC_Q80P(fc_matvec_q80p_r4c4, 4, 4, 512)
FC_Q80P(fc_matvec_q80p_r4c5, 4, 5, 512)
#endif
FC_Q80P(fc_matvec_q80p_r1c1, 1, 1, 2048)
FC_Q80P(fc_matvec_q80p_r1c2, 1, 2, 1024)
FC_Q80P(fc_matvec_q80p_r2c3, 2, 3, 512)
FC_Q80P(fc_matvec_q80p_r2c4, 2, 4, 512)
FC_Q80P(fc_matvec_q80p_r2c5, 2, 5, 512)
FC_Q80P(fc_matvec_q80p_r2c8, 2, 8, 512)
FC_Q80P(fc_matvec_q80p_r4c8, 4, 8, 512)

/* ---------------------------------------------------------------------------------------
 * Q2_0_g128: activation-quantised integer dot per 32-element sub-block.
 * ------------------------------------------------------------------------------------- */

/* Round a value >= 0 (or +Inf) to IEEE binary16 with round-to-nearest-even and widen it
 * back exactly. Overflow (>= 65520) gives +Inf. Below the binary16 normal range, adding
 * 0.5 rounds to a multiple of 2^-24 (the subnormal step) in one IEEE add, ties to even. */
__device__ __forceinline__ float fc_q2_0_round_via_half(float v) {
    unsigned bits = __float_as_uint(v);

    if (bits >= 0x477FF000u)
        return __uint_as_float(0x7F800000u);
    if (bits < 0x38800000u)
        return __fsub_rn(__fadd_rn(v, 0.5f), 0.5f);
    bits += 0x0FFFu + ((bits >> 13) & 1u);
    return __uint_as_float(bits & ~0x1FFFu);
}

/* Integer dot of four 2-bit codes (value code-1) with q, wrapping mod 2^32. */
__device__ __forceinline__ int fc_q2_0_code_dot(unsigned codes, const int (&q)[4]) {
    unsigned acc = 0;

    #pragma unroll
    for (int k = 0; k < 4; k++)
        acc += (unsigned)((int)((codes >> (2 * k)) & 3u) - 1) * (unsigned)q[k];
    return (int)acc;
}

/* Quantise this lane's four activations against its 8-lane sub-block's absmax/127.
 * Returns the binary16-rounded activation scale; q receives the rounded integers. */
__device__ __forceinline__ float fc_q2_0_quantise(float4 xq, int (&q)[4]) {
    const float xk[4] = { xq.x, xq.y, xq.z, xq.w };
    float amax = 0.0f;

    #pragma unroll
    for (int k = 0; k < 4; k++)
        amax = fmaxf(amax, fabsf(xk[k]));
    amax = fmaxf(amax, __shfl_xor(amax, 1));
    amax = fmaxf(amax, __shfl_xor(amax, 2));
    amax = fmaxf(amax, __shfl_xor(amax, 4));
    const float d_pre = __fmul_rn(amax, 1.0f / 127.0f);
    const float inv = d_pre != 0.0f ? __fdividef(1.0f, d_pre) : 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++)
        q[k] = (int)roundf(__fmul_rn(xk[k], inv));
    return fc_q2_0_round_via_half(d_pre);
}

__device__ __forceinline__ void fc_q2_0_body(const unsigned char *w, const float *x, float *out,
                                             int rows, int cols, int row_stride) {
    float *sx = fc_kq_shared;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = 16 * blockIdx.x + 2 * warp;
    const bool active0 = r0 < rows, active1 = r0 + 1 < rows;
    const unsigned char *b0 = w + (long long)r0 * row_stride;
    const unsigned char *b1 = active1 ? b0 + row_stride : b0;
    float p0[1] = { 0.0f }, p1[1] = { 0.0f };

    for (int cbase = 0; cbase < cols; cbase += FC_KQ_SINGLE_CHUNK) {
        const int cend = min(FC_KQ_SINGLE_CHUNK, cols - cbase);
        __syncthreads();
        fc_kq_stage_single(sx, x, cbase, cend);
        __syncthreads();
        if (!active0)
            continue;
        const int nblk = cend >> 7;
        for (int b = 0; b < nblk; b++) {
            int q[4];
            const float d_act = fc_q2_0_quantise(((const float4 *)sx)[32 * b + lane], q);
            const int dot0 = fc_q2_0_code_dot(__ldg(b0 + 4 + lane), q);
            const int dot1 = fc_q2_0_code_dot(__ldg(b1 + 4 + lane), q);
            const float d0 = __ldg((const float *)b0), d1 = __ldg((const float *)b1);
            p0[0] = __fadd_rn(p0[0], __fmul_rn(d0, __fmul_rn(d_act, (float)dot0)));
            p1[0] = __fadd_rn(p1[0], __fmul_rn(d1, __fmul_rn(d_act, (float)dot1)));
            b0 += FC_Q2_0_BLOCK_BYTES;
            b1 += FC_Q2_0_BLOCK_BYTES;
        }
    }
    fc_kq_store_row<1>(out, p0, rows, r0, active0, 0, 1);
    fc_kq_store_row<1>(out, p1, rows, r0 + 1, active1, 0, 1);
}

/* ---------------------------------------------------------------------------------------
 * Dense F32/F16/BF16: block per row, strided FMA chain per thread, shared halving tree.
 * Plain global loads (no staging). T = blockDim.x must be a power of two for a full sum.
 * ------------------------------------------------------------------------------------- */

template <int WTYPE>
__device__ __forceinline__ float fc_dense_weight(const unsigned char *row, int i) {
    if (WTYPE == FC_DENSE_F32)
        return ((const float *)row)[i];
    const unsigned short bits = ((const unsigned short *)row)[i];
    if (WTYPE == FC_DENSE_BF16)
        return __uint_as_float((unsigned)bits << 16);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

template <int WTYPE, int NC>
__device__ __forceinline__ void fc_dense_body(const unsigned char *w, const float *x, float *out,
                                              int rows, int cols, int row_stride, int col0,
                                              int nc) {
    const int row = blockIdx.x;
    if (row >= rows)
        return;
    const int tid = threadIdx.x, threads = blockDim.x;
    const unsigned char *wrow = w + (size_t)row * (size_t)row_stride;
    float *sd = fc_kq_shared;
    float p[NC];

    #pragma unroll
    for (int c = 0; c < NC; c++)
        p[c] = 0.0f;
    for (int i = tid; i < cols; i += threads) {
        const float wv = fc_dense_weight<WTYPE>(wrow, i);
        #pragma unroll
        for (int c = 0; c < NC; c++) {
            if (c < nc)
                p[c] = __fmaf_rn(wv, x[(size_t)(col0 + c) * cols + i], p[c]);
        }
    }
    #pragma unroll
    for (int c = 0; c < NC; c++)
        sd[c * threads + tid] = p[c];
    __syncthreads();
    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int c = 0; c < NC; c++)
                sd[c * threads + tid] += sd[c * threads + tid + s];
        }
        __syncthreads();
    }
    if (tid != 0)
        return;
    for (int c = 0; c < nc; c++)
        out[(size_t)(col0 + c) * rows + row] = sd[c * threads];
}

/* ---------------------------------------------------------------------------------------
 * Entry points (unmangled). Parameter order is ABI: w, x, out, rows, cols, row_stride[, n].
 * ------------------------------------------------------------------------------------- */

/* Uniform <NC, CHUNK> names per weight type, so one macro can emit each family. */
#define FC_KQ_BODY_ALIAS(NAME, BODY)                                                       \
    template <int NC, int CHUNK>                                                           \
    __device__ __forceinline__ void NAME(const unsigned char *w, const float *x,           \
                                         float *out, int rows, int cols, int row_stride,   \
                                         int col0, int nc) {                               \
        BODY(w, x, out, rows, cols, row_stride, col0, nc);                                 \
    }

FC_KQ_BODY_ALIAS(fc_q4k_body, (fc_kq_rowpair_body<FC_KQ_KIND_Q4K, NC, CHUNK>))
FC_KQ_BODY_ALIAS(fc_q5k_body, (fc_kq_rowpair_body<FC_KQ_KIND_Q5K, NC, CHUNK>))
FC_KQ_BODY_ALIAS(fc_q6k_body, (fc_kq_rowpair_body<FC_KQ_KIND_Q6K, NC, CHUNK>))
FC_KQ_BODY_ALIAS(fc_f32_body, (fc_dense_body<FC_DENSE_F32, NC>))
FC_KQ_BODY_ALIAS(fc_f16_body, (fc_dense_body<FC_DENSE_F16, NC>))
FC_KQ_BODY_ALIAS(fc_bf16_body, (fc_dense_body<FC_DENSE_BF16, NC>))

#define FC_KQ_BATCHED(NAME, BODY, NC)                                                      \
    extern "C" __global__ void NAME(const unsigned char *w, const float *x, float *out,    \
                                    int rows, int cols, int row_stride, int n) {           \
        const int col0 = (NC) * blockIdx.y;                                                \
        BODY<NC, FC_NX_CHUNK>(w, x, out, rows, cols, row_stride, col0, min((NC), n - col0)); \
    }

/* Single-column kernel plus the 2-, 4- and 8-wide batched tiles of one weight type. */
#define FC_KQ_FAMILY(PREFIX, BODY)                                                         \
    extern "C" __global__ void PREFIX(const unsigned char *w, const float *x, float *out,  \
                                      int rows, int cols, int row_stride) {                \
        BODY<1, FC_KQ_SINGLE_CHUNK>(w, x, out, rows, cols, row_stride, 0, 1);              \
    }                                                                                      \
    FC_KQ_BATCHED(PREFIX##_nx2, BODY, 2)                                                   \
    FC_KQ_BATCHED(PREFIX##_nx, BODY, 4)                                                    \
    FC_KQ_BATCHED(PREFIX##_nx8, BODY, 8)

FC_KQ_FAMILY(fc_matvec_q4k, fc_q4k_body)
FC_KQ_FAMILY(fc_matvec_q5k, fc_q5k_body)
FC_KQ_FAMILY(fc_matvec_q6k, fc_q6k_body)
FC_KQ_FAMILY(fc_matvec_q80, fc_q80_body)
FC_KQ_FAMILY(fc_matvec_f32, fc_f32_body)
FC_KQ_FAMILY(fc_matvec_f16, fc_f16_body)
FC_KQ_FAMILY(fc_matvec_bf16, fc_bf16_body)

extern "C" __global__ void fc_matvec_q2_0(const unsigned char *w, const float *x, float *out,
                                          int rows, int cols, int row_stride) {
    fc_q2_0_body(w, x, out, rows, cols, row_stride);
}
