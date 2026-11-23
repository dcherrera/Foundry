/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Portions modelled on ggml (MIT License, Copyright (c) 2023-2026 The ggml
 * authors, https://github.com/ggml-org/ggml), modified. See NOTICE. */
/*
 * matvec_q3k.cu — Q3_K dequant + matrix-vector product for Foundry
 *
 * Target: sm_37 (K80, Kepler)
 * Compile: nvcc -ptx -arch=sm_37 matvec_q3k.cu -o matvec_q3k.ptx
 *
 * Computes: out[row] = dot( dequant_q3k(w[row]), x[cols] )
 *   - w is a Q3_K weight matrix in the REPACKED device layout (see below)
 *   - x is a dense fp32 activation vector
 *   - out is fp32
 *
 * REPACKED WEIGHT LAYOUT (produced host-side by fc_repack_weight at upload;
 * see fc_cuda.h / fc_cuda.c). The raw GGUF Q3_K super-block (110 B, unaligned,
 * packed 6-bit scales) capped this kernel at 10-15 GB/s; the first repack
 * (three quant bytes per u32) reached ~30 GB/s but was bound by the dequant
 * INSTRUCTION stream: sm_37 runs int-mul / int32->f32 cvt at 32/clk and
 * shifts at 64/clk vs 160-192/clk for AND/OR/FADD/FMA. The current 152 B
 * super-block is laid out for a shift-free, cvt-free magic-number decode:
 *
 *   u32 qw[32] @ 0    lane l's 8 elements as nibbles: nibble i (= 4n + j,
 *                     element e = 32i + l) = qv | hbit<<2, so the dequant
 *                     value (qv - hb) is exactly (nibble - 4)
 *   u8  sc[16] @ 128  RAW 6-bit sub-block scales (0..63, value = sc - 32),
 *                     reordered per lane half: sc[8*lh + i] = raw6[2i + lh]
 *                     (each lane's 8 scale bytes in ONE uint2 load)
 *   f32 d      @ 144  super-block scale, half-decoded on the host
 *   pad        @ 148  stride 152 keeps every block 8-byte aligned
 *
 * MAGIC-NUMBER DEQUANT: a small unsigned integer v sitting at bit p of a
 * word becomes the float (v - bias) with AND + OR + one FMA:
 *   __int_as_float(0x4B000000 | (w & (mask << p)))  = 2^23 + v * 2^p
 *   fma(that, 2^-p, -(2^(23-p) + bias))             = v - bias   (exact)
 * No shift, no cvt (the >>16 word-halving preshifts are hoisted per block).
 * fs * fq is an exact small-integer product, so __fmul_rn(fs, fq) is
 * bit-identical to the old (float)(scale * (qv - hb)) cvt and the FMA chain
 * order is unchanged -> results stay bit-identical (verified against the
 * CPU replica in cuda/tests/bench_matvec.c). Value semantics mirror:
 *   Pantheon/src/formats/gguf_dequant.c pn_dequantize_block_q3_k
 *   Pantheon/src/core/qweight.c         dot_q3_k + pn_qmv
 *
 * The kernel is issue-bound on Kepler, so x is STAGED IN SHARED MEMORY in
 * 2048-element chunks by the whole 8-warp block, and each warp computes
 * FC_Q3K_ROWS_PER_WARP output rows with independent accumulators (Kepler
 * needs the ILP to dual-issue; the x LDS traffic is amortized across rows).
 *
 * PTX symbols emitted (for embedding into ../cuda/fc_cuda_ptx.h):
 *   FC_PTX_MATVEC_Q3K  -> kernels  fc_matvec_q3k, fc_matvec_q3k_nx (batched
 *                         N-column variant; see its own header below), and
 *                         fc_matvec_q3k_ps_r{1,2}p{1,2,4,8} (peer-stream
 *                         variants: bit-exact, more loads in flight — see
 *                         "PEER-STREAM variants" below)
 *
 * Grid/block layout: WARP-PER-ROW-PAIR — grid = ceil(rows/16), block = 256
 * (8 warps), warp w handles rows blockIdx.x*16 + w*2 + {0,1}, __shfl_down
 * reduction per row, shared = min(cols, 2048) * sizeof(float) staged x.
 *
 * Kepler notes: pure fp32; __shfl_down is the non-_sync Kepler variant.
 */

/* FC_NX_CHUNK + the batched activation-staging loop, shared with
 * matvec_kquants.cu (macros only — no linkage, so it sits outside extern "C"). */
#include "matvec_nx_staging.cuh"

extern "C" {

/* Repacked Q3_K super-block: 256 elements in 152 bytes (was 110 raw). */
#define FC_Q3K_RBLOCK_BYTES 152
/* Output rows per warp (independent accumulators for dual-issue ILP).
 * The fc_cuda.c launcher grid math must match: grid = ceil(rows / (8*R)). */
#define FC_Q3K_ROWS_PER_WARP 4

/* Magic decodes: exact float (v - bias) from the masked field at bit p.
 * NIB: 4-bit field, BYTE: 8-bit field. p is compile-time after unrolling. */
#define FC_MAGIC_NIB(w, p, bias)                                            \
    __fmaf_rn(__int_as_float(0x4B000000u | ((w) & (0xFu << (p)))),          \
              1.0f / (float)(1u << (p)),                                    \
              -(float)((1u << (23 - (p))) + (bias)))
#define FC_MAGIC_BYTE(w, p, bias)                                           \
    __fmaf_rn(__int_as_float(0x4B000000u | ((w) & (0xFFu << (p)))),         \
              1.0f / (float)(1u << (p)),                                    \
              -(float)((1u << (23 - (p))) + (bias)))

/* Warp sum via __shfl_down (Kepler: the non-_sync variant). */
/* Byte-aligned fields decode fastest through Kepler's 128/clk 8-bit cvt
 * (single I2F.F32.U8 with byte select — cheaper than the magic FMA, which
 * stays for the sub-byte fields). */
#define FC_CVT_BYTE(w, k) ((float)(((w) >> (8 * (k))) & 0xFFu))

__device__ __forceinline__ float fc_q3k_warp_sum(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v, 8);
    v += __shfl_down(v, 4);
    v += __shfl_down(v, 2);
    v += __shfl_down(v, 1);
    return v;
}

/*
 * One lane's share of a repacked Q3_K super-block dot: lane l covers the 8
 * elements e = 32i + l (i in 0..7). One aligned u32 quant load, one uint2
 * scale load, one f32 d load; per element the decode is AND+OR+FMA (fq),
 * AND+OR+FMA (fs), FMUL, FMA — no shift, no cvt (only the three per-block
 * >>16 word-halving preshifts). xb points at the super-block's 256 x values
 * STAGED IN SHARED MEMORY. Returns sum(dequant(e) * xb[e]).
 */
__device__ __forceinline__ float fc_q3k_sb_lane_dot(const unsigned char *blk,
                                                    const float *xb, int lane) {
    unsigned int wv = __ldg((const unsigned int *)blk + lane);
    float d_all = __ldg((const float *)(blk + 144));
    uint2 scw = __ldg((const uint2 *)(blk + 128) + (lane >> 4));
    unsigned int wh = wv >> 16;

    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        /* nibble at bit 4i: direct mask while it stays below the fp32
         * exponent (p <= 16), else from the >>16 half */
        unsigned int qsrc = (i <= 4) ? wv : wh;
        int qp = (i <= 4) ? 4 * i : 4 * i - 16;
        unsigned int ssrc = (i < 4) ? scw.x : scw.y;
        float fq = FC_MAGIC_NIB(qsrc, qp, 4);
        float fs = __fsub_rn(FC_CVT_BYTE(ssrc, i & 3), 32.0f);
        acc = __fmaf_rn(__fmul_rn(fs, fq), xb[32 * i + lane], acc);
    }
    return d_all * acc;
}

/*
 * Interleaved ROW-PAIR lane dot: the two rows' 8-element FMA chains advance
 * in lockstep per element (two independent chains the scheduler can
 * dual-issue, unlike two back-to-back dependent 8-FMA chains), sharing one
 * shared-memory x load per element. Each row's per-lane chain keeps the
 * exact i = 0..7 order of fc_q3k_sb_lane_dot -> bit-exact per row.
 */
__device__ __forceinline__ void fc_q3k_sb_lane_dot2(const unsigned char *blk0,
                                                    const unsigned char *blk1,
                                                    const float *xb, int lane,
                                                    float *p0, float *p1) {
    unsigned int wv0 = __ldg((const unsigned int *)blk0 + lane);
    unsigned int wv1 = __ldg((const unsigned int *)blk1 + lane);
    float d0 = __ldg((const float *)(blk0 + 144));
    float d1 = __ldg((const float *)(blk1 + 144));
    uint2 sw0 = __ldg((const uint2 *)(blk0 + 128) + (lane >> 4));
    uint2 sw1 = __ldg((const uint2 *)(blk1 + 128) + (lane >> 4));
    unsigned int wh0 = wv0 >> 16, wh1 = wv1 >> 16;

    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int qp = (i <= 4) ? 4 * i : 4 * i - 16;
        unsigned int q0 = (i <= 4) ? wv0 : wh0;
        unsigned int q1 = (i <= 4) ? wv1 : wh1;
        unsigned int s0 = (i < 4) ? sw0.x : sw0.y;
        unsigned int s1 = (i < 4) ? sw1.x : sw1.y;
        float xv = xb[32 * i + lane];
        float fq0 = FC_MAGIC_NIB(q0, qp, 4);
        float fs0 = __fsub_rn(FC_CVT_BYTE(s0, i & 3), 32.0f);
        float fq1 = FC_MAGIC_NIB(q1, qp, 4);
        float fs1 = __fsub_rn(FC_CVT_BYTE(s1, i & 3), 32.0f);
        a0 = __fmaf_rn(__fmul_rn(fs0, fq0), xv, a0);
        a1 = __fmaf_rn(__fmul_rn(fs1, fq1), xv, a1);
    }
    *p0 += d0 * a0;
    *p1 += d1 * a1;
}

/*
 * WARP-PER-ROW-PAIR: blockDim = 256 (8 warps); warp w handles rows
 * blockIdx.x * 16 + w*2 + {0,1} with independent accumulators (ILP for
 * Kepler dual-issue; x LDS amortized across the pair). Per 2048-element
 * chunk the whole block first stages x into shared memory (coalesced, one
 * barrier pair per chunk), then each warp runs the lane decode above over
 * the chunk's super-blocks and finally reduces each row with __shfl_down.
 * Inactive tail warps/rows still participate in the staging barriers.
 * Launch: grid = ceil(rows / 16), block = 256,
 * shared = min(cols, 2048) * sizeof(float).
 *
 * w:          [rows * row_stride] REPACKED Q3_K weight bytes (row-major)
 * cols:       inner dimension (must be a multiple of 256)
 * row_stride: bytes per weight row = (cols / 256) * 152
 */
__global__ void fc_matvec_q3k(const unsigned char *w, const float *x,
                              float *out, int rows, int cols, int row_stride) {
    extern __shared__ float sh_x[];        /* [min(cols, 2048)] staged x */
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = (blockIdx.x * 8 + warp) * 2;
    int active0 = row0 < rows;
    int active1 = row0 + 1 < rows;

    const unsigned char *b0 = w + (size_t)row0 * (size_t)row_stride;
    /* tail guard: clamp b1 onto row0 so the interleaved dot stays in
     * bounds; its p1 result is discarded (active1 false) */
    const unsigned char *b1 = active1 ? b0 + (size_t)row_stride : b0;

    float p0 = 0.0f, p1 = 0.0f;
    for (int cbase = 0; cbase < cols; cbase += 2048) {
        int cend = cols - cbase < 2048 ? cols - cbase : 2048;
        __syncthreads();
        for (int i = threadIdx.x; i < cend; i += 256)
            sh_x[i] = __ldg(x + cbase + i);
        __syncthreads();
        if (active0) {
            int nsb = cend >> 8;
            const float *xb = sh_x;
            for (int sb = 0; sb < nsb; sb++) {
                fc_q3k_sb_lane_dot2(b0, b1, xb, lane, &p0, &p1);
                b0 += FC_Q3K_RBLOCK_BYTES;
                b1 += FC_Q3K_RBLOCK_BYTES;
                xb += 256;
            }
        }
    }

    p0 = fc_q3k_warp_sum(p0);
    p1 = fc_q3k_warp_sum(p1);
    if (active0 && lane == 0)
        out[row0] = p0;
    if (active1 && lane == 0)
        out[row0 + 1] = p1;
}

/*
 * ============================================================
 * BATCHED (multi-column) matvec: out[t][r] = dequant(w[r]) . x[t]
 * ============================================================
 *
 * The prefill/batch kernel: N activation columns against the same weight.
 * Each weight block is decoded ONCE and FMA'd into FC_NX_COLS independent
 * per-column accumulators, so the dequant instruction stream (the single-
 * token kernels' issue bound) amortizes over the batch. Weight bytes are
 * streamed once per FC_NX_COLS-column tile (grid.y tiles the batch).
 *
 * BIT-EXACT per column: each column's per-lane chain keeps the exact
 * (super-block, i) order and __shfl_down reduction of fc_matvec_q3k's
 * single-column path, so column t of the batched output is bit-identical
 * to fc_matvec_q3k run on x[t] alone. (Verified: cuda/tests/test_matvec_nx.)
 *
 * x:   [n][cols] row-major (token-major); out: [n][rows] token-major.
 * Launch: grid (ceil(rows/16), ceil(n/FC_NX_COLS)), block 256,
 * shared = FC_NX_COLS * min(cols, FC_NX_CHUNK) floats. Columns beyond n in
 * the last tile stage zeros and skip their writes.
 */
/* L6 (multi-tenant speed-opt): the tile width is now a TEMPLATE parameter
 * with TWO instantiations — fc_matvec_q3k_nx (4 columns, the original) and
 * fc_matvec_q3k_nx8 (8 columns). The 8-wide tile streams the shared dense/
 * head weights ONCE for a K=8 batched-decode step (and any K in 5..8 costs
 * one tile, un-regressing K=5-7), but its extra FMA chains cost ~25% at
 * K<=4 on the issue-bound sm_37 — so the host launcher (fc_cuda.c
 * fc_matvec_nx_launch) dispatches by n: n<=4 -> 4-wide, n>=5 -> 8-wide.
 * 2x8 accumulators/lane + 8 x 512 x 4 B = 16 KB shared staging fits sm_37
 * (48 KB). Per-column FMA order is identical at either width -> bit-exact
 * per column at both (cuda/tests/test_matvec_nx). */
/* FC_NX_CHUNK is defined in matvec_nx_staging.cuh, next to the staging loop
 * that gives it its meaning, because matvec_kquants.cu needs the same value. */

} /* extern "C" — the width-templated nx bodies below need C++ linkage */

/* Interleaved ROW-PAIR x NC lane dot: one decode of both rows' element,
 * 2*NC independent FMA chains. Per column the chain is the exact i = 0..7
 * order of fc_q3k_sb_lane_dot -> bit-exact per column at any NC.
 * xb column c's 256 staged values live at xb[c*FC_NX_CHUNK ..]. */
template <int NC>
__device__ __forceinline__ void fc_q3k_sb_lane_dot2nx(const unsigned char *blk0,
                                                      const unsigned char *blk1,
                                                      const float *xb, int lane,
                                                      float *p0, float *p1) {
    unsigned int wv0 = __ldg((const unsigned int *)blk0 + lane);
    unsigned int wv1 = __ldg((const unsigned int *)blk1 + lane);
    float d0 = __ldg((const float *)(blk0 + 144));
    float d1 = __ldg((const float *)(blk1 + 144));
    uint2 sw0 = __ldg((const uint2 *)(blk0 + 128) + (lane >> 4));
    uint2 sw1 = __ldg((const uint2 *)(blk1 + 128) + (lane >> 4));
    unsigned int wh0 = wv0 >> 16, wh1 = wv1 >> 16;

    float a0[NC], a1[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        a0[c] = 0.0f;
        a1[c] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int qp = (i <= 4) ? 4 * i : 4 * i - 16;
        unsigned int q0 = (i <= 4) ? wv0 : wh0;
        unsigned int q1 = (i <= 4) ? wv1 : wh1;
        unsigned int s0 = (i < 4) ? sw0.x : sw0.y;
        unsigned int s1 = (i < 4) ? sw1.x : sw1.y;
        float fq0 = FC_MAGIC_NIB(q0, qp, 4);
        float fs0 = __fsub_rn(FC_CVT_BYTE(s0, i & 3), 32.0f);
        float fq1 = FC_MAGIC_NIB(q1, qp, 4);
        float fs1 = __fsub_rn(FC_CVT_BYTE(s1, i & 3), 32.0f);
        float t0 = __fmul_rn(fs0, fq0);
        float t1 = __fmul_rn(fs1, fq1);
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float xv = xb[c * FC_NX_CHUNK + 32 * i + lane];
            a0[c] = __fmaf_rn(t0, xv, a0[c]);
            a1[c] = __fmaf_rn(t1, xv, a1[c]);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; c++) {
        p0[c] += d0 * a0[c];
        p1[c] += d1 * a1[c];
    }
}

template <int NC>
__device__ __forceinline__ void fc_matvec_q3k_nx_body(const unsigned char *w,
                                                      const float *x,
                                                      float *out, int rows,
                                                      int cols, int row_stride,
                                                      int n) {
    extern __shared__ float sh_x[];   /* [NC * min(cols, FC_NX_CHUNK)] */
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = (blockIdx.x * 8 + warp) * 2;
    int col0 = blockIdx.y * NC;
    int nc = n - col0;
    if (nc > NC)
        nc = NC;
    int active0 = row0 < rows;
    int active1 = row0 + 1 < rows;

    const unsigned char *b0 = w + (size_t)row0 * (size_t)row_stride;
    const unsigned char *b1 = active1 ? b0 + (size_t)row_stride : b0;

    float p0[NC], p1[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        p0[c] = 0.0f;
        p1[c] = 0.0f;
    }
    for (int cbase = 0; cbase < cols; cbase += FC_NX_CHUNK) {
        int cend = cols - cbase < FC_NX_CHUNK ? cols - cbase : FC_NX_CHUNK;
        __syncthreads();
        FC_NX_STAGE_X(sh_x, x, col0, cols, cbase, cend, NC, nc,
                      FC_NX_STAGE_KEEP);
        __syncthreads();
        if (active0) {
            int nsb = cend >> 8;
            const float *xb = sh_x;
            for (int sb = 0; sb < nsb; sb++) {
                fc_q3k_sb_lane_dot2nx<NC>(b0, b1, xb, lane, p0, p1);
                b0 += FC_Q3K_RBLOCK_BYTES;
                b1 += FC_Q3K_RBLOCK_BYTES;
                xb += 256;
            }
        }
    }

#pragma unroll
    for (int c = 0; c < NC; c++) {
        float s0 = fc_q3k_warp_sum(p0[c]);
        float s1 = fc_q3k_warp_sum(p1[c]);
        if (c < nc && lane == 0) {
            if (active0)
                out[(size_t)(col0 + c) * rows + row0] = s0;
            if (active1)
                out[(size_t)(col0 + c) * rows + row0 + 1] = s1;
        }
    }
}

/*
 * ============================================================
 * PEER-STREAM variants — same math, more memory-level parallelism
 * ============================================================
 *
 * WHY. When a die runs this kernel over ANOTHER die's weights across the K80's
 * on-card PLX peer link, it is not bandwidth-bound, it is LATENCY-bound: the
 * link delivers 9.3-12.4 GB/s but the helper die only achieves 2.7-4.2 GB/s.
 * The bind is bytes IN FLIGHT.
 * Little's law at a ~4 us peer round trip wants ~50 KB outstanding per die;
 * the stock kernel issues one super-block's loads per warp and then consumes
 * them, and the helper's slice is thin, so it never gets there.
 *
 * These variants attack exactly that, and nothing else:
 *   PF  prefetch depth — PF super-blocks' weight loads are ALL issued before
 *       the first is consumed, so bytes in flight scale with PF, and the PF
 *       blocks are contiguous (PF * 152 B) which suits the link.
 *   RPW rows per warp — 1 instead of 2 doubles the warp count for the same
 *       row slice, i.e. more independent load streams per SM.
 *   staged-x is loaded with float4 (128-bit) instead of scalar, quartering
 *       the request count for the activation, which a helper die must also
 *       drag across the link unless it has been staged locally.
 *
 * BIT-EXACT with fc_matvec_q3k at every (RPW, PF). Hoisting a load changes
 * only WHEN a value arrives, and each output row keeps the identical per-lane
 * i = 0..7 / super-block-in-order FMA chain and the identical __shfl_down
 * reduction. The row-pair interleave the stock kernel uses is a scheduling
 * device (dual-issue ILP), not part of the arithmetic, so computing the two
 * rows' chains separately is value-identical. (Gated in
 * cuda/tests/bench_peer_stream.c, which compares every variant's output
 * against the stock kernel's byte for byte.)
 */

/* One super-block's per-lane operands, loaded ahead of use. */
struct fc_q3k_blk {
    unsigned int wv;        /* lane's 8 elements as nibbles */
    float d;                /* super-block scale */
    uint2 sw;               /* lane half's 8 raw 6-bit sub-block scales */
};

__device__ __forceinline__ fc_q3k_blk fc_q3k_blk_load(const unsigned char *blk,
                                                      int lane) {
    fc_q3k_blk b;
    b.wv = __ldg((const unsigned int *)blk + lane);
    b.d  = __ldg((const float *)(blk + 144));
    b.sw = __ldg((const uint2 *)(blk + 128) + (lane >> 4));
    return b;
}

/* Force a preloaded super-block to be LIVE at this point in the program.
 *
 * Without this the prefetch is a lie: ptxas sinks every __ldg back down to its
 * first consumer, and the SASS for PF = 1, 2, 4 and 8 comes out identical (72
 * registers each, max 16 consecutive LDG each — measured 2026-08-09 before
 * this was added). An empty asm volatile with read-write operands is a
 * scheduling barrier for exactly these four values and nothing else, so the
 * loads must complete before it and the whole PF-deep batch is in flight
 * together. It emits no instructions. */
__device__ __forceinline__ void fc_q3k_blk_pin(fc_q3k_blk &b) {
    asm volatile("" : "+r"(b.wv), "+f"(b.d), "+r"(b.sw.x), "+r"(b.sw.y));
}

/* One row's contribution from a PRELOADED super-block. Same i = 0..7 order,
 * same magic decode, same __fmul_rn/__fmaf_rn sequence as
 * fc_q3k_sb_lane_dot -> bit-identical per row. */
__device__ __forceinline__ float fc_q3k_blk_dot(const fc_q3k_blk &b,
                                                const float *xb, int lane) {
    unsigned int wh = b.wv >> 16;
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        unsigned int q = (i <= 4) ? b.wv : wh;
        int qp = (i <= 4) ? 4 * i : 4 * i - 16;
        unsigned int s = (i < 4) ? b.sw.x : b.sw.y;
        float fq = FC_MAGIC_NIB(q, qp, 4);
        float fs = __fsub_rn(FC_CVT_BYTE(s, i & 3), 32.0f);
        a = __fmaf_rn(__fmul_rn(fs, fq), xb[32 * i + lane], a);
    }
    return b.d * a;
}

/* Stage one 2048-element x chunk into shared. float4 when both the chunk and
 * the shared base allow it (4x fewer requests — the point when x is remote). */
__device__ __forceinline__ void fc_q3k_stage_x(float *sh_x, const float *xsrc,
                                               int cend) {
    if (((cend & 3) == 0) && ((((size_t)xsrc) & 15u) == 0) &&
        ((((size_t)sh_x) & 15u) == 0)) {
        const float4 *s = (const float4 *)xsrc;
        float4 *d = (float4 *)sh_x;
        for (int i = threadIdx.x; i < (cend >> 2); i += 256)
            d[i] = __ldg(s + i);
    } else {
        for (int i = threadIdx.x; i < cend; i += 256)
            sh_x[i] = __ldg(xsrc + i);
    }
}

/*
 * RPW rows per warp, PF-deep weight prefetch.
 * Launch: grid = ceil(rows / (8 * RPW)), block = 256,
 * shared = min(cols, 2048) * sizeof(float).
 */
template <int RPW, int PF>
__device__ __forceinline__ void fc_matvec_q3k_ps_body(const unsigned char *w,
                                                      const float *x, float *out,
                                                      int rows, int cols,
                                                      int row_stride) {
    extern __shared__ float sh_xps[];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = (blockIdx.x * 8 + warp) * RPW;
    int active0 = row0 < rows;

    const unsigned char *bp[RPW];
    float p[RPW];
    int act[RPW];
#pragma unroll
    for (int r = 0; r < RPW; r++) {
        act[r] = (row0 + r) < rows;
        /* tail guard: an inactive row aliases row0 so its loads stay in
         * bounds; its result is discarded below */
        bp[r] = w + (size_t)(act[r] ? row0 + r : row0) * (size_t)row_stride;
        p[r] = 0.0f;
    }

    for (int cbase = 0; cbase < cols; cbase += 2048) {
        int cend = cols - cbase < 2048 ? cols - cbase : 2048;
        __syncthreads();
        fc_q3k_stage_x(sh_xps, x + cbase, cend);
        __syncthreads();
        if (active0) {
            int nsb = cend >> 8;
            const float *xb = sh_xps;
            int sb = 0;
            for (; sb + PF <= nsb; sb += PF) {
                fc_q3k_blk bl[RPW][PF];
#pragma unroll
                for (int u = 0; u < PF; u++)
#pragma unroll
                    for (int r = 0; r < RPW; r++)
                        bl[r][u] = fc_q3k_blk_load(
                            bp[r] + u * FC_Q3K_RBLOCK_BYTES, lane);
#pragma unroll
                for (int u = 0; u < PF; u++)
#pragma unroll
                    for (int r = 0; r < RPW; r++)
                        fc_q3k_blk_pin(bl[r][u]);
#pragma unroll
                for (int u = 0; u < PF; u++)
#pragma unroll
                    for (int r = 0; r < RPW; r++)
                        p[r] += fc_q3k_blk_dot(bl[r][u], xb + u * 256, lane);
#pragma unroll
                for (int r = 0; r < RPW; r++)
                    bp[r] += PF * FC_Q3K_RBLOCK_BYTES;
                xb += PF * 256;
            }
            for (; sb < nsb; sb++) {
#pragma unroll
                for (int r = 0; r < RPW; r++) {
                    p[r] += fc_q3k_blk_dot(fc_q3k_blk_load(bp[r], lane), xb,
                                           lane);
                    bp[r] += FC_Q3K_RBLOCK_BYTES;
                }
                xb += 256;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RPW; r++) {
        float s = fc_q3k_warp_sum(p[r]);
        if (act[r] && lane == 0)
            out[row0 + r] = s;
    }
}

extern "C" {

/* Row-pair per warp (the stock shape), prefetch 1/2/4/8. _r2p1 isolates the
 * float4 staged-x change; the rest add weight prefetch on top of it. */
__global__ void fc_matvec_q3k_ps_r2p1(const unsigned char *w, const float *x,
                                      float *out, int rows, int cols,
                                      int row_stride) {
    fc_matvec_q3k_ps_body<2, 1>(w, x, out, rows, cols, row_stride);
}
__global__ void fc_matvec_q3k_ps_r2p2(const unsigned char *w, const float *x,
                                      float *out, int rows, int cols,
                                      int row_stride) {
    fc_matvec_q3k_ps_body<2, 2>(w, x, out, rows, cols, row_stride);
}
__global__ void fc_matvec_q3k_ps_r2p4(const unsigned char *w, const float *x,
                                      float *out, int rows, int cols,
                                      int row_stride) {
    fc_matvec_q3k_ps_body<2, 4>(w, x, out, rows, cols, row_stride);
}
__global__ void fc_matvec_q3k_ps_r2p8(const unsigned char *w, const float *x,
                                      float *out, int rows, int cols,
                                      int row_stride) {
    fc_matvec_q3k_ps_body<2, 8>(w, x, out, rows, cols, row_stride);
}

/* One row per warp: twice the warps (and twice the independent load streams)
 * over the same rows. Costs the row-pair dual-issue ILP, which only matters
 * when the die is issue-bound on its OWN memory — a helper die is not. */
__global__ void fc_matvec_q3k_ps_r1p1(const unsigned char *w, const float *x,
                                      float *out, int rows, int cols,
                                      int row_stride) {
    fc_matvec_q3k_ps_body<1, 1>(w, x, out, rows, cols, row_stride);
}
__global__ void fc_matvec_q3k_ps_r1p4(const unsigned char *w, const float *x,
                                      float *out, int rows, int cols,
                                      int row_stride) {
    fc_matvec_q3k_ps_body<1, 4>(w, x, out, rows, cols, row_stride);
}
__global__ void fc_matvec_q3k_ps_r1p8(const unsigned char *w, const float *x,
                                      float *out, int rows, int cols,
                                      int row_stride) {
    fc_matvec_q3k_ps_body<1, 8>(w, x, out, rows, cols, row_stride);
}

__global__ void fc_matvec_q3k_nx(const unsigned char *w, const float *x,
                                 float *out, int rows, int cols,
                                 int row_stride, int n) {
    fc_matvec_q3k_nx_body<4>(w, x, out, rows, cols, row_stride, n);
}

/* 2-wide tile (L10): launched by the host at n == 2. The tile width is a
 * compile-time constant, so a 4-wide tile fed two live columns still runs
 * four FMA chains per element and stages 4*FC_NX_CHUNK floats -- measured
 * 0.78x TWO single-column launches at 4096x2048 on sm_37, i.e. the "batch"
 * was a net loss at K=2. Bit-exact per column like every other width. */
__global__ void fc_matvec_q3k_nx2(const unsigned char *w, const float *x,
                                  float *out, int rows, int cols,
                                  int row_stride, int n) {
    fc_matvec_q3k_nx_body<2>(w, x, out, rows, cols, row_stride, n);
}

/* 8-wide tile (L6): launched by the host for n >= 5. */
__global__ void fc_matvec_q3k_nx8(const unsigned char *w, const float *x,
                                  float *out, int rows, int cols,
                                  int row_stride, int n) {
    fc_matvec_q3k_nx_body<8>(w, x, out, rows, cols, row_stride, n);
}

} /* extern "C" */
