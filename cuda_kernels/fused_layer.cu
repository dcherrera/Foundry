/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Portions modelled on ggml (MIT License, Copyright (c) 2023-2026 The ggml
 * authors, https://github.com/ggml-org/ggml), modified. See NOTICE. */
/*
 * fused_layer.cu — multi-segment fused per-layer launches for Qwen3.5-MoE
 * (Foundry)
 *
 * Target: sm_37 (K80, Kepler)
 * Compile: nvcc -ptx -arch=sm_37 fused_layer.cu -o fused_layer.ptx
 *
 * Decode on the K80 is LAUNCH-BOUND (~80 us per launch vs ~35 us of work), so
 * these kernels collapse the per-layer launch storm without touching any
 * single matvec's internal accumulation order (bit-exactness is the gate):
 *
 *   fc_fused_multi — up to FC_FUSED_MAX_SEG independent matvec "segments"
 *     that all read the SAME x vector, in ONE launch. A segment is one weight
 *     matrix (any repacked quant type or F32) with its own output buffer and
 *     an optional fused epilogue. The grid is the concatenation of every
 *     segment's blocks (block_base per segment); each block resolves its
 *     segment from blockIdx.x and runs EXACTLY the body its stand-alone
 *     kernel ran:
 *       kind 0 (quant): WARP-PER-ROW-PAIR, 8 warps/block (16 rows/block),
 *               staged-x shared chunks, __shfl_down reduce per row —
 *               duplicated from matvec_q3k.cu / matvec_kquants.cu
 *               (identical fp op order per row; the row pair only adds an
 *               independent accumulator chain for Kepler dual-issue ILP).
 *               Q8_0 instead runs fc_matvec_q80's 8-lanes-per-row body
 *               (32 rows/block; same per-row result, full load width).
 *       kind 1 (F32): one block per row, 256-thread strided partials +
 *               shared-mem tree — duplicated from fc_matvec_f32 (and
 *               fc_moe_shexp_dot_gate's dot, same shape).
 *       kind 2 (quant gate+up pair): one warp computes the gate AND up dots
 *               for its row pair (each dot keeps its own accumulator and the
 *               stand-alone per-row order), writes silu(g)*u — the
 *               fc_moe_gateup_q3k pattern for the shared expert.
 *     Epilogues (applied once to the finished dot by the writing thread —
 *     value-identical to the small kernels they replace):
 *       0 none               out[row] = v
 *       1 sigmoid            out[row] = 1/(1+exp(-v))            (fc_sigmoid)
 *       2 decay gate         out[row] = a[row]*softplus(v+dt[row])
 *                                                          (fc_ssm_decay_gate)
 *       3 residual add       out[row] = v; hid[row] += v            (fc_add)
 *       4 moe combine        bo = moe[row] + v*sgate[0]; out[row] = bo;
 *                            hid[row] += bo   (fc_moe_shexp_dot_gate scale +
 *                                              fc_add combine + fc_add resid)
 *
 *   fc_attn_qknorm_rope — per-head QK-RMSNorm + partial NEOX RoPE for Q and K
 *     in ONE launch (was 4: qk_rms_norm x2 + rope x2). One block per head
 *     (Q heads first, then K heads); the norm body is fc_qk_rms_norm
 *     verbatim, then one barrier, then the fc_rope_neox_partial pair rotation
 *     verbatim. Heads are independent, so fusing preserves per-head math.
 *
 * REPACKED WEIGHT LAYOUTS: see matvec_q3k.cu / matvec_kquants.cu. The lane
 * dots below are duplicated from those files (PTX modules are standalone) and
 * keep the identical fp op order, so results stay bit-identical.
 *
 * PTX symbols emitted (for embedding into ../cuda/fc_cuda_ptx.h):
 *   FC_PTX_FUSED_LAYER -> kernels fc_fused_multi, fc_attn_qknorm_rope
 *
 * Kepler notes: pure fp32; __shfl_down is the non-_sync Kepler variant.
 */

extern "C" {

/* Repacked block sizes (bytes) — must match fc_repack_weight. */
#define FC_FU_Q3K_RB 152
#define FC_FU_Q4K_RB 152
#define FC_FU_Q5K_RB 184
#define FC_FU_Q6K_RB 216
#define FC_FU_Q80_RB  36

/* Magic decodes (duplicated from matvec_q3k.cu / matvec_kquants.cu): exact
 * float (v - bias) from the masked field at bit p, AND + OR + one FMA — no
 * shift, no cvt (sm_37 runs int-mul / int32->f32 cvt at 32/clk, shifts at
 * 64/clk vs AND/OR 160 and FMUL/FMA 192). BIT16/BIT2X16 scale a 1-/2-bit
 * field by 16 (the K-quant high-bit weight). */
#define FC_MAGIC_NIB(w, p, bias)                                            \
    __fmaf_rn(__int_as_float(0x4B000000u | ((w) & (0xFu << (p)))),          \
              1.0f / (float)(1u << (p)),                                    \
              -(float)((1u << (23 - (p))) + (bias)))
#define FC_MAGIC_BYTE(w, p, bias)                                           \
    __fmaf_rn(__int_as_float(0x4B000000u | ((w) & (0xFFu << (p)))),         \
              1.0f / (float)(1u << (p)),                                    \
              -(float)((1u << (23 - (p))) + (bias)))
#define FC_MAGIC_BIT16(w, p)                                                \
    __fmaf_rn(__int_as_float(0x4B000000u | ((w) & (1u << (p)))),            \
              16.0f / (float)(1u << (p)), -(float)(1u << (27 - (p))))
#define FC_MAGIC_BIT2X16(w, p)                                              \
    __fmaf_rn(__int_as_float(0x4B000000u | ((w) & (3u << (p)))),            \
              16.0f / (float)(1u << (p)), -(float)(1u << (27 - (p))))

/* Device-side segment descriptor — layout must match fc_fused_seg_dev in
 * fc_cuda.c (6 x u64 then 8 x s32, 80 bytes, 8-byte aligned). */
typedef struct {
    unsigned long long w;     /* weight bytes (repacked quant or raw F32) */
    unsigned long long w2;    /* kind 2: the up weight; else 0 */
    unsigned long long out;   /* [rows] f32 output */
    unsigned long long aux0;  /* ep 2: dt[rows]; ep 4: moe_out[rows] */
    unsigned long long aux1;  /* ep 2: a[rows]; ep 3/4: hidden[rows] */
    unsigned long long aux2;  /* ep 4: sgate[1] */
    int rows, cols, row_stride;
    int kind;                 /* 0 quant, 1 f32, 2 quant gate+up */
    int qtype;                /* GGUF id: 8, 11, 12, 13, 14 (kind 0/2) */
    int ep;                   /* epilogue code (see header comment) */
    int block_base;           /* first grid block of this segment */
    int pad;
} fc_fused_seg_dev;

/* Byte-aligned fields decode fastest through Kepler's 128/clk 8-bit cvt
 * (single I2F.F32.U8 with byte select — cheaper than the magic FMA, which
 * stays for the sub-byte fields). */
#define FC_CVT_BYTE(w, k) ((float)(((w) >> (8 * (k))) & 0xFFu))

__device__ __forceinline__ float fc_fu_warp_sum(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v, 8);
    v += __shfl_down(v, 4);
    v += __shfl_down(v, 2);
    v += __shfl_down(v, 1);
    return v;
}

/* ---- Lane accumulators, duplicated from matvec_q3k.cu / matvec_kquants.cu.
 *
 * PINNED FP OP ORDER: the stand-alone kernels compiled (nvcc 11.4, default
 * -fmad=true) to specific contraction shapes, and inlining them into this
 * multi-kernel let nvcc re-fold the caller-level `partial += dot` into the
 * element chain (different summation grouping = different bits — caught by
 * the argmax/logit gate). Every accumulation below is therefore written with
 * explicit __fmaf_rn/__fmul_rn/__fsub_rn/__fadd_rn intrinsics that reproduce
 * the stand-alone kernels' EXACT PTX sequences (the magic decodes and the
 * exact small-integer FMULs produce bit-identical term values to the old
 * cvt-based decode):
 *   Q3_K/Q6_K: 8x acc = fma(mul(fs,fq), xb, acc) from 0.0f, then
 *              partial = fma(d, acc, partial)   (mul(fs,fq) = exact int
 *              product, bit-identical to the old cvt(scale*q))
 *   Q4_K/Q5_K: 8x { mul(d,sc), mul(.,nib), mul(dmin,mn), sub,
 *              acc = fma(sub, xb, acc) } from 0.0f, then
 *              partial = add(partial, acc)
 *   Q8_0:      partial = fma(mul(d, (float)q), xb, partial)
 * Each function takes the running partial and returns the updated one. ---- */

__device__ __forceinline__ float fc_fu_q3k_acc(const unsigned char *blk,
                                               const float *xb, int lane,
                                               float partial) {
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
    return __fmaf_rn(d_all, acc, partial);
}

__device__ __forceinline__ float fc_fu_q4k_acc(const unsigned char *blk,
                                               const float *xb, int lane,
                                               float partial) {
    unsigned int qw = __ldg((const unsigned int *)blk + lane);
    uint2 scw = __ldg((const uint2 *)(blk + 128));
    uint2 mnw = __ldg((const uint2 *)(blk + 136));
    float2 dd = __ldg((const float2 *)(blk + 144));
    float d = dd.x, dmin = dd.y;
    unsigned int qwh = qw >> 16;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int si = 2 * j + h;
            int nbit = 8 * j + 4 * h;
            unsigned int qsrc = (nbit <= 16) ? qw : qwh;
            int qp = (nbit <= 16) ? nbit : nbit - 16;
            unsigned int ssrc = (si < 4) ? scw.x : scw.y;
            unsigned int msrc = (si < 4) ? mnw.x : mnw.y;
            float fn = FC_MAGIC_NIB(qsrc, qp, 0);
            float sc = FC_CVT_BYTE(ssrc, si & 3);
            float mn = FC_CVT_BYTE(msrc, si & 3);
            float t = __fsub_rn(__fmul_rn(__fmul_rn(d, sc), fn),
                                __fmul_rn(dmin, mn));
            acc = __fmaf_rn(t, xb[64 * j + 32 * h + lane], acc);
        }
    }
    return __fadd_rn(partial, acc);
}

__device__ __forceinline__ float fc_fu_q5k_acc(const unsigned char *blk,
                                               const float *xb, int lane,
                                               float partial) {
    unsigned int qw = __ldg((const unsigned int *)blk + lane);
    unsigned int qh = __ldg(blk + 128 + lane);
    uint2 scw = __ldg((const uint2 *)(blk + 160));
    uint2 mnw = __ldg((const uint2 *)(blk + 168));
    float2 dd = __ldg((const float2 *)(blk + 176));
    float d = dd.x, dmin = dd.y;
    unsigned int qwh = qw >> 16;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int si = 2 * j + h;
            int nbit = 8 * j + 4 * h;
            unsigned int qsrc = (nbit <= 16) ? qw : qwh;
            int qp = (nbit <= 16) ? nbit : nbit - 16;
            unsigned int ssrc = (si < 4) ? scw.x : scw.y;
            unsigned int msrc = (si < 4) ? mnw.x : mnw.y;
            float fn = FC_MAGIC_NIB(qsrc, qp, 0);
            float fhi = FC_MAGIC_BIT16(qh, si);      /* 16 * high bit */
            float sc = FC_CVT_BYTE(ssrc, si & 3);
            float mn = FC_CVT_BYTE(msrc, si & 3);
            float t = __fsub_rn(__fmul_rn(__fmul_rn(d, sc),
                                          __fadd_rn(fn, fhi)),
                                __fmul_rn(dmin, mn));
            acc = __fmaf_rn(t, xb[64 * j + 32 * h + lane], acc);
        }
    }
    return __fadd_rn(partial, acc);
}

__device__ __forceinline__ float fc_fu_q6k_acc(const unsigned char *blk,
                                               const float *xb, int lane,
                                               float partial) {
    unsigned int qlw = __ldg((const unsigned int *)blk + lane);
    unsigned int qh2 = __ldg((const unsigned short *)(blk + 128) + lane);
    float d = __ldg((const float *)(blk + 208));
    uint2 scw = __ldg((const uint2 *)(blk + 192) + (lane >> 4));
    unsigned int qlh = qlw >> 16;
    float acc = 0.0f;
#pragma unroll
    for (int half = 0; half < 2; half++) {
#pragma unroll
        for (int quad = 0; quad < 4; quad++) {
            int i = 4 * half + quad;
            int nbit = 16 * half + 8 * (quad & 1) + 4 * ((quad >> 1) & 1);
            int hbit = 8 * half + 2 * quad;
            unsigned int qsrc = (nbit <= 16) ? qlw : qlh;
            int qp = (nbit <= 16) ? nbit : nbit - 16;
            unsigned int ssrc = (i < 4) ? scw.x : scw.y;
            float fn = FC_MAGIC_NIB(qsrc, qp, 32);           /* nib - 32 */
            float fhb = FC_MAGIC_BIT2X16(qh2, hbit);         /* 16 * hb */
            float fs = __fsub_rn(FC_CVT_BYTE(ssrc, i & 3), 128.0f);
            float fq = __fadd_rn(fn, fhb);          /* q, exact integer */
            acc = __fmaf_rn(__fmul_rn(fs, fq), xb[32 * i + lane], acc);
        }
    }
    return __fmaf_rn(d, acc, partial);
}

/*
 * Interleaved ROW-PAIR lane dot: the two rows' 8-element FMA chains advance
 * in lockstep per element (two independent chains the scheduler can
 * dual-issue, unlike two back-to-back dependent 8-FMA chains), sharing one
 * shared-memory x load per element. Each row's per-lane chain keeps the
 * exact i = 0..7 order of fc_q3k_sb_lane_dot -> bit-exact per row.
 */
__device__ __forceinline__ void fc_fu_q3k_acc2(const unsigned char *blk0,
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
    *p0 = __fmaf_rn(d0, a0, *p0);
    *p1 = __fmaf_rn(d1, a1, *p1);
}


/* Interleaved ROW-PAIR Q4_K lane dot: both rows' chains advance per element
 * (independent chains for dual-issue), sharing one x load. Each row keeps
 * the exact per-element op order of fc_q4k_sb_lane_dot -> bit-exact. */
__device__ __forceinline__ void fc_fu_q4k_acc2(const unsigned char *blk0,
                                                    const unsigned char *blk1,
                                                    const float *xb, int lane,
                                                    float *p0, float *p1) {
    unsigned int qw0 = __ldg((const unsigned int *)blk0 + lane);
    unsigned int qw1 = __ldg((const unsigned int *)blk1 + lane);
    uint2 sw0 = __ldg((const uint2 *)(blk0 + 128));
    uint2 sw1 = __ldg((const uint2 *)(blk1 + 128));
    uint2 mw0 = __ldg((const uint2 *)(blk0 + 136));
    uint2 mw1 = __ldg((const uint2 *)(blk1 + 136));
    float2 dd0 = __ldg((const float2 *)(blk0 + 144));
    float2 dd1 = __ldg((const float2 *)(blk1 + 144));
    unsigned int qh0 = qw0 >> 16, qh1 = qw1 >> 16;
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int si = 2 * j + h;
            int nbit = 8 * j + 4 * h;
            int qp = (nbit <= 16) ? nbit : nbit - 16;
            unsigned int q0 = (nbit <= 16) ? qw0 : qh0;
            unsigned int q1 = (nbit <= 16) ? qw1 : qh1;
            unsigned int s0 = (si < 4) ? sw0.x : sw0.y;
            unsigned int s1 = (si < 4) ? sw1.x : sw1.y;
            unsigned int m0 = (si < 4) ? mw0.x : mw0.y;
            unsigned int m1 = (si < 4) ? mw1.x : mw1.y;
            float xv = xb[64 * j + 32 * h + lane];
            float t0 = __fsub_rn(
                __fmul_rn(__fmul_rn(dd0.x, FC_CVT_BYTE(s0, si & 3)),
                          FC_MAGIC_NIB(q0, qp, 0)),
                __fmul_rn(dd0.y, FC_CVT_BYTE(m0, si & 3)));
            float t1 = __fsub_rn(
                __fmul_rn(__fmul_rn(dd1.x, FC_CVT_BYTE(s1, si & 3)),
                          FC_MAGIC_NIB(q1, qp, 0)),
                __fmul_rn(dd1.y, FC_CVT_BYTE(m1, si & 3)));
            a0 = __fmaf_rn(t0, xv, a0);
            a1 = __fmaf_rn(t1, xv, a1);
        }
    }
    *p0 = __fadd_rn(*p0, a0);
    *p1 = __fadd_rn(*p1, a1);
}


/* Interleaved ROW-PAIR Q5_K lane dot (see fc_q4k_sb_lane_dot2). */
__device__ __forceinline__ void fc_fu_q5k_acc2(const unsigned char *blk0,
                                                    const unsigned char *blk1,
                                                    const float *xb, int lane,
                                                    float *p0, float *p1) {
    unsigned int qw0 = __ldg((const unsigned int *)blk0 + lane);
    unsigned int qw1 = __ldg((const unsigned int *)blk1 + lane);
    unsigned int qb0 = __ldg(blk0 + 128 + lane);
    unsigned int qb1 = __ldg(blk1 + 128 + lane);
    uint2 sw0 = __ldg((const uint2 *)(blk0 + 160));
    uint2 sw1 = __ldg((const uint2 *)(blk1 + 160));
    uint2 mw0 = __ldg((const uint2 *)(blk0 + 168));
    uint2 mw1 = __ldg((const uint2 *)(blk1 + 168));
    float2 dd0 = __ldg((const float2 *)(blk0 + 176));
    float2 dd1 = __ldg((const float2 *)(blk1 + 176));
    unsigned int qh0 = qw0 >> 16, qh1 = qw1 >> 16;
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int si = 2 * j + h;
            int nbit = 8 * j + 4 * h;
            int qp = (nbit <= 16) ? nbit : nbit - 16;
            unsigned int q0 = (nbit <= 16) ? qw0 : qh0;
            unsigned int q1 = (nbit <= 16) ? qw1 : qh1;
            unsigned int s0 = (si < 4) ? sw0.x : sw0.y;
            unsigned int s1 = (si < 4) ? sw1.x : sw1.y;
            unsigned int m0 = (si < 4) ? mw0.x : mw0.y;
            unsigned int m1 = (si < 4) ? mw1.x : mw1.y;
            float xv = xb[64 * j + 32 * h + lane];
            float n0 = __fadd_rn(FC_MAGIC_NIB(q0, qp, 0),
                                 FC_MAGIC_BIT16(qb0, si));
            float n1 = __fadd_rn(FC_MAGIC_NIB(q1, qp, 0),
                                 FC_MAGIC_BIT16(qb1, si));
            float t0 = __fsub_rn(
                __fmul_rn(__fmul_rn(dd0.x, FC_CVT_BYTE(s0, si & 3)), n0),
                __fmul_rn(dd0.y, FC_CVT_BYTE(m0, si & 3)));
            float t1 = __fsub_rn(
                __fmul_rn(__fmul_rn(dd1.x, FC_CVT_BYTE(s1, si & 3)), n1),
                __fmul_rn(dd1.y, FC_CVT_BYTE(m1, si & 3)));
            a0 = __fmaf_rn(t0, xv, a0);
            a1 = __fmaf_rn(t1, xv, a1);
        }
    }
    *p0 = __fadd_rn(*p0, a0);
    *p1 = __fadd_rn(*p1, a1);
}


/* Interleaved ROW-PAIR Q6_K lane dot (see fc_q4k_sb_lane_dot2). */
__device__ __forceinline__ void fc_fu_q6k_acc2(const unsigned char *blk0,
                                                    const unsigned char *blk1,
                                                    const float *xb, int lane,
                                                    float *p0, float *p1) {
    unsigned int ql0 = __ldg((const unsigned int *)blk0 + lane);
    unsigned int ql1 = __ldg((const unsigned int *)blk1 + lane);
    unsigned int qh20 = __ldg((const unsigned short *)(blk0 + 128) + lane);
    unsigned int qh21 = __ldg((const unsigned short *)(blk1 + 128) + lane);
    float d0 = __ldg((const float *)(blk0 + 208));
    float d1 = __ldg((const float *)(blk1 + 208));
    uint2 sw0 = __ldg((const uint2 *)(blk0 + 192) + (lane >> 4));
    uint2 sw1 = __ldg((const uint2 *)(blk1 + 192) + (lane >> 4));
    unsigned int qlh0 = ql0 >> 16, qlh1 = ql1 >> 16;
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int half = 0; half < 2; half++) {
#pragma unroll
        for (int quad = 0; quad < 4; quad++) {
            int i = 4 * half + quad;
            int nbit = 16 * half + 8 * (quad & 1) + 4 * ((quad >> 1) & 1);
            int hbit = 8 * half + 2 * quad;
            int qp = (nbit <= 16) ? nbit : nbit - 16;
            unsigned int q0 = (nbit <= 16) ? ql0 : qlh0;
            unsigned int q1 = (nbit <= 16) ? ql1 : qlh1;
            unsigned int s0 = (i < 4) ? sw0.x : sw0.y;
            unsigned int s1 = (i < 4) ? sw1.x : sw1.y;
            float xv = xb[32 * i + lane];
            float fq0 = __fadd_rn(FC_MAGIC_NIB(q0, qp, 32),
                                  FC_MAGIC_BIT2X16(qh20, hbit));
            float fq1 = __fadd_rn(FC_MAGIC_NIB(q1, qp, 32),
                                  FC_MAGIC_BIT2X16(qh21, hbit));
            float fs0 = __fsub_rn(FC_CVT_BYTE(s0, i & 3), 128.0f);
            float fs1 = __fsub_rn(FC_CVT_BYTE(s1, i & 3), 128.0f);
            a0 = __fmaf_rn(__fmul_rn(fs0, fq0), xv, a0);
            a1 = __fmaf_rn(__fmul_rn(fs1, fq1), xv, a1);
        }
    }
    *p0 = __fmaf_rn(d0, a0, *p0);
    *p1 = __fmaf_rn(d1, a1, *p1);
}


/* Super-block lane accumulate dispatch (256-elem super-block types). qtype
 * is uniform per block, so the switch never diverges within a warp. */
__device__ __forceinline__ float fc_fu_sb_acc(int qtype, const unsigned char *blk,
                                              const float *xb, int lane,
                                              float partial) {
    switch (qtype) {
    case 11: return fc_fu_q3k_acc(blk, xb, lane, partial);
    case 12: return fc_fu_q4k_acc(blk, xb, lane, partial);
    case 13: return fc_fu_q5k_acc(blk, xb, lane, partial);
    default: return fc_fu_q6k_acc(blk, xb, lane, partial);   /* 14 */
    }
}

/* Element-interleaved pair dispatch (two rows / two matrices per call —
 * two independent accumulator chains for Kepler dual-issue). */
__device__ __forceinline__ void fc_fu_sb_acc2(int qtype,
                                              const unsigned char *blk0,
                                              const unsigned char *blk1,
                                              const float *xb, int lane,
                                              float *p0, float *p1) {
    switch (qtype) {
    case 11: fc_fu_q3k_acc2(blk0, blk1, xb, lane, p0, p1); break;
    case 12: fc_fu_q4k_acc2(blk0, blk1, xb, lane, p0, p1); break;
    case 13: fc_fu_q5k_acc2(blk0, blk1, xb, lane, p0, p1); break;
    default: fc_fu_q6k_acc2(blk0, blk1, xb, lane, p0, p1); break;   /* 14 */
    }
}

__device__ __forceinline__ int fc_fu_sb_bytes(int qtype) {
    switch (qtype) {
    case 13: return FC_FU_Q5K_RB;
    case 14: return FC_FU_Q6K_RB;
    default: return FC_FU_Q3K_RB;                   /* 11, 12 (both 152) */
    }
}

/* Fused epilogue — one store of the finished per-row value v. Each case is
 * value-identical to the stand-alone kernel it replaces (see file header). */
__device__ __forceinline__ void fc_fu_store(const fc_fused_seg_dev *sg,
                                            int row, float v) {
    float *out = (float *)(size_t)sg->out;
    int ep = sg->ep;
    if (ep == 1) {                       /* sigmoid (fc_sigmoid) */
        out[row] = 1.0f / (1.0f + expf(-v));
    } else if (ep == 2) {                /* decay gate (fc_ssm_decay_gate) */
        const float *dt = (const float *)(size_t)sg->aux0;
        const float *a  = (const float *)(size_t)sg->aux1;
        float t = v + dt[row];
        float sp = (t > 20.0f) ? t : log1pf(expf(t));
        out[row] = a[row] * sp;
    } else if (ep == 3) {                /* residual add (fc_add) */
        float *hid = (float *)(size_t)sg->aux1;
        out[row] = v;
        hid[row] = hid[row] + v;
    } else if (ep == 4) {                /* moe combine + residual */
        const float *moe = (const float *)(size_t)sg->aux0;
        float *hid = (float *)(size_t)sg->aux1;
        const float *sgt = (const float *)(size_t)sg->aux2;
        /* Pinned mul-then-add: the stand-alone path was a mul kernel
         * (shexp *= sgate) then an add kernel (moe_out + shexp) — keep the
         * same two ops (no fma contraction). */
        float bo = __fadd_rn(moe[row], __fmul_rn(v, sgt[0]));
        out[row] = bo;
        hid[row] = __fadd_rn(hid[row], bo);
    } else {
        out[row] = v;
    }
}

/* ---- Q8_0 segments: 8 lanes per row, 32 rows per 256-thread block.
 *
 * The fc_matvec_q80 / _nx body (matvec_kquants.cu fc_q80_body), duplicated:
 * lane s of a row's 8-lane group carries element chains 4s..4s+3 of every
 * 32-weight block (t = d * q, P = fma(t, x, P) in block order), one 32-bit
 * quant load and one scale load per block per lane, and the 32 chains meet in
 * the same 16/8/4/2/1 tree as the warp-per-row form below (shuffles 4/2/1
 * inside the group, then (v0 + v2) + (v1 + v3)). Bit-identical per row to
 * both, at the stand-alone kernel's load width. ---- */

#define FC_FU_Q80_LANES 8
#define FC_FU_Q80_ROWS  32    /* rows per block; fc_cuda.c must match */
#define FC_FU_Q80_UNROLL 4

/* Byte m of a block word as an exact float: the word is XORed with
 * 0x80808080 (byte = q + 128), placed under exponent 0x4B, less 2^23 + 128. */
__device__ __forceinline__ float fc_fu_q80_byte(unsigned biased, int m) {
    return __fsub_rn(__uint_as_float(__byte_perm(biased, 0x4B000000u, 0x7440 | m)),
                     8388736.0f);
}

__device__ __forceinline__ void fc_fu_q80_block(const unsigned char *blk, int sub,
                                                const float *xb, float (&p)[4]) {
    const unsigned biased =
        __ldg((const unsigned *)(blk + 4 + 4 * sub)) ^ 0x80808080u;
    const float d = __ldg((const float *)blk);
    const float4 xv = *(const float4 *)(xb + 4 * sub);
    p[0] = __fmaf_rn(__fmul_rn(d, fc_fu_q80_byte(biased, 0)), xv.x, p[0]);
    p[1] = __fmaf_rn(__fmul_rn(d, fc_fu_q80_byte(biased, 1)), xv.y, p[1]);
    p[2] = __fmaf_rn(__fmul_rn(d, fc_fu_q80_byte(biased, 2)), xv.z, p[2]);
    p[3] = __fmaf_rn(__fmul_rn(d, fc_fu_q80_byte(biased, 3)), xv.w, p[3]);
}

/* Four blocks in the G4 layout (cols % 128 == 0, fc_repack_weight): [f32 d0..d3]
 * then lane sub's 16 bytes at 16 + 16*sub. One 128-bit load each, same chains. */
__device__ __forceinline__ void fc_fu_q80_group(const unsigned char *grp, int sub,
                                                const float *xb, float (&p)[4]) {
    const uint4 q = __ldg((const uint4 *)(grp + 16 + 16 * sub));
    const float4 d4 = __ldg((const float4 *)grp);
    const unsigned w[4] = { q.x, q.y, q.z, q.w };
    const float d[4] = { d4.x, d4.y, d4.z, d4.w };
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const unsigned biased = w[j] ^ 0x80808080u;
        const float4 xv = *(const float4 *)(xb + 32 * j + 4 * sub);
        p[0] = __fmaf_rn(__fmul_rn(d[j], fc_fu_q80_byte(biased, 0)), xv.x, p[0]);
        p[1] = __fmaf_rn(__fmul_rn(d[j], fc_fu_q80_byte(biased, 1)), xv.y, p[1]);
        p[2] = __fmaf_rn(__fmul_rn(d[j], fc_fu_q80_byte(biased, 2)), xv.z, p[2]);
        p[3] = __fmaf_rn(__fmul_rn(d[j], fc_fu_q80_byte(biased, 3)), xv.w, p[3]);
    }
}

/* The row's 32 chains -> one sum on every lane of its group (lane 0 used). */
__device__ __forceinline__ float fc_fu_q80_sum(float (&p)[4]) {
#pragma unroll
    for (int m = 0; m < 4; m++) {
        p[m] += __shfl_down(p[m], 4);
        p[m] += __shfl_down(p[m], 2);
        p[m] += __shfl_down(p[m], 1);
    }
    return (p[0] + p[2]) + (p[1] + p[3]);
}

/* One Q8_0 segment block (kind 0, or kind 2 gate+up). x staged in sh by the
 * caller's chunk loop shape (2048 floats). */
__device__ __forceinline__ void fc_fu_q80_seg(const fc_fused_seg_dev *sg, int brel,
                                              const float *x, float *sh) {
    const int rows = sg->rows, cols = sg->cols, kind = sg->kind;
    const bool g4 = (cols & 127) == 0;   /* fc_repack_weight's G4 layout */
    const int sub = threadIdx.x & (FC_FU_Q80_LANES - 1);
    const int row = brel * FC_FU_Q80_ROWS + threadIdx.x / FC_FU_Q80_LANES;
    const bool active = row < rows;
    const size_t roff = (size_t)(active ? row : 0) * (size_t)sg->row_stride;
    const unsigned char *g = (const unsigned char *)(size_t)sg->w + roff;
    const unsigned char *u = (const unsigned char *)(size_t)sg->w2 + roff;
    float pg[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float pu[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (int cbase = 0; cbase < cols; cbase += 2048) {
        int cend = cols - cbase < 2048 ? cols - cbase : 2048;
        __syncthreads();
        for (int i = threadIdx.x; i < cend; i += 256)
            sh[i] = __ldg(x + cbase + i);
        __syncthreads();
        if (!active)
            continue;
        /* FC_FU_Q80_UNROLL blocks per step keep that many loads in flight
         * (the stand-alone kernel's unroll; same per-chain block order). */
        const int nb = cend >> 5;
        int b = 0;
        if (g4) {                        /* nb is a multiple of 4 */
            for (; b < nb; b += 4) {
                fc_fu_q80_group(g, sub, sh + 32 * b, pg);
                if (kind == 2)
                    fc_fu_q80_group(u, sub, sh + 32 * b, pu);
                g += 4 * FC_FU_Q80_RB;
                u += 4 * FC_FU_Q80_RB;
            }
        } else if (kind == 2) {          /* two weights: half the unroll */
            for (; b + FC_FU_Q80_UNROLL / 2 <= nb; b += FC_FU_Q80_UNROLL / 2) {
#pragma unroll
                for (int j = 0; j < FC_FU_Q80_UNROLL / 2; j++) {
                    fc_fu_q80_block(g + FC_FU_Q80_RB * j, sub, sh + 32 * (b + j), pg);
                    fc_fu_q80_block(u + FC_FU_Q80_RB * j, sub, sh + 32 * (b + j), pu);
                }
                g += FC_FU_Q80_RB * (FC_FU_Q80_UNROLL / 2);
                u += FC_FU_Q80_RB * (FC_FU_Q80_UNROLL / 2);
            }
            for (; b < nb; b++) {
                fc_fu_q80_block(g, sub, sh + 32 * b, pg);
                fc_fu_q80_block(u, sub, sh + 32 * b, pu);
                g += FC_FU_Q80_RB;
                u += FC_FU_Q80_RB;
            }
        } else {
            for (; b + FC_FU_Q80_UNROLL <= nb; b += FC_FU_Q80_UNROLL) {
#pragma unroll
                for (int j = 0; j < FC_FU_Q80_UNROLL; j++)
                    fc_fu_q80_block(g + FC_FU_Q80_RB * j, sub, sh + 32 * (b + j), pg);
                g += FC_FU_Q80_RB * FC_FU_Q80_UNROLL;
            }
            for (; b < nb; b++) {
                fc_fu_q80_block(g, sub, sh + 32 * b, pg);
                g += FC_FU_Q80_RB;
            }
        }
    }
    float vg = fc_fu_q80_sum(pg);
    float vu = kind == 2 ? fc_fu_q80_sum(pu) : 0.0f;
    if (active && sub == 0) {
        /* kind 2: silu(gate)*up — identical to fc_moe_silu_mul / gateup. */
        float v = (kind == 2) ? (vg / (1.0f + expf(-vg))) * vu : vg;
        fc_fu_store(sg, row, v);
    }
}

/*
 * Multi-segment fused matvec. All segments read the same x[.] (each segment
 * uses its own cols). Launch: grid = sum of per-segment blocks (kind 0/2:
 * ceil(rows/16), 16 rows per block; kind 1: rows), block = 256, shared =
 * max over segments (kind 0/2: min(cols,2048)*4; kind 1: 1024).
 */
__global__ void fc_fused_multi(const fc_fused_seg_dev *segs, int nseg,
                               const float *x) {
    extern __shared__ float sh[];

    /* Resolve this block's segment (block_base ascending). */
    int s = 0;
    for (int t = 1; t < nseg; t++)
        if (blockIdx.x >= segs[t].block_base)
            s = t;
    const fc_fused_seg_dev *sg = &segs[s];
    const int rows = sg->rows;
    const int cols = sg->cols;
    const int rstride = sg->row_stride;
    const int kind = sg->kind;
    const int brel = blockIdx.x - sg->block_base;

    if (kind == 1) {
        /* F32 block-per-row — fc_matvec_f32 verbatim (256-thread strided
         * partials + tree; also matches fc_moe_shexp_dot_gate's dot). */
        int row = brel;
        if (row >= rows)
            return;
        int tid = threadIdx.x;
        int threads = blockDim.x;
        const float *w_row = (const float *)
            ((const unsigned char *)(size_t)sg->w + (size_t)row * (size_t)rstride);
        float partial = 0.0f;
        for (int i = tid; i < cols; i += threads)
            partial = __fmaf_rn(w_row[i], x[i], partial);  /* fc_matvec_f32's
                                                            * fma chain */
        sh[tid] = partial;
        __syncthreads();
        for (int t = threads / 2; t > 0; t >>= 1) {
            if (tid < t)
                sh[tid] += sh[tid + t];
            __syncthreads();
        }
        if (tid == 0)
            fc_fu_store(sg, row, sh[0]);
        return;
    }

    if (sg->qtype == 8) {
        fc_fu_q80_seg(sg, brel, x, sh);
        return;
    }

    /* Super-block quants: WARP-PER-ROW-PAIR (kind 0) / gate+up pair
     * (kind 2) — staged-x chunk loop duplicated from matvec_q3k.cu /
     * matvec_kquants.cu. Warp w handles rows brel*16 + 2w + {0,1} with
     * independent accumulator chains (dual-issue ILP; staged-x LDS amortized
     * across the pair); each row's per-lane chain and reduction are
     * unchanged -> bit-exact per row. */
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row0 = brel * 16 + warp * 2;
    int active0 = row0 < rows;
    int active1 = row0 + 1 < rows;
    const int qtype = sg->qtype;
    /* strength-reduced running block pointers (advanced per quant block) */
    const unsigned char *w_row0 =
        (const unsigned char *)(size_t)sg->w + (size_t)row0 * (size_t)rstride;
    /* tail: clamp the pair onto row0, duplicate result discarded */
    const unsigned char *w_row1 = active1 ? w_row0 + (size_t)rstride : w_row0;
    const unsigned char *u_row0 =
        (const unsigned char *)(size_t)sg->w2 + (size_t)row0 * (size_t)rstride;
    const unsigned char *u_row1 = active1 ? u_row0 + (size_t)rstride : u_row0;

    float pg0 = 0.0f, pu0 = 0.0f, pg1 = 0.0f, pu1 = 0.0f;
    for (int cbase = 0; cbase < cols; cbase += 2048) {
        int cend = cols - cbase < 2048 ? cols - cbase : 2048;
        __syncthreads();
        for (int i = threadIdx.x; i < cend; i += 256)
            sh[i] = __ldg(x + cbase + i);
        __syncthreads();
        if (active0) {
            int bb = fc_fu_sb_bytes(qtype);
            int nsb = cend >> 8;
            const float *xb = sh;
            for (int sb = 0; sb < nsb; sb++) {
                fc_fu_sb_acc2(qtype, w_row0, w_row1, xb, lane, &pg0, &pg1);
                if (kind == 2)
                    fc_fu_sb_acc2(qtype, u_row0, u_row1, xb, lane,
                                  &pu0, &pu1);
                w_row0 += bb;
                w_row1 += bb;
                if (kind == 2) {
                    u_row0 += bb;
                    u_row1 += bb;
                }
                xb += 256;
            }
        }
    }

    pg0 = fc_fu_warp_sum(pg0);
    pg1 = fc_fu_warp_sum(pg1);
    if (kind == 2) {
        pu0 = fc_fu_warp_sum(pu0);
        pu1 = fc_fu_warp_sum(pu1);
    }
    if (active0 && lane == 0) {
        /* kind 2: silu(gate)*up — identical to fc_moe_silu_mul / gateup. */
        float v = (kind == 2) ? (pg0 / (1.0f + expf(-pg0))) * pu0 : pg0;
        fc_fu_store(sg, row0, v);
    }
    if (active1 && lane == 0) {
        float v = (kind == 2) ? (pg1 / (1.0f + expf(-pg1))) * pu1 : pg1;
        fc_fu_store(sg, row0 + 1, v);
    }
}

/*
 * Fused per-head QK-RMSNorm + partial NEOX RoPE for Q and K (was 4 launches).
 * Blocks [0, n_head) norm+rope Q heads with q_norm_w; blocks [n_head,
 * n_head + n_head_kv) norm+rope K heads with k_norm_w. The norm body is
 * fc_qk_rms_norm verbatim (threads = blockDim.x, tree over blockDim.x); the
 * rope body is fc_rope_neox_partial verbatim (thread j < half rotates the
 * (j, j+half) pair). One barrier between them orders the in-place update.
 *
 * Launch: grid = n_head + n_head_kv, block = min(256, head_dim), shared =
 * blockDim.x * sizeof(float).
 */
__global__ void fc_attn_qknorm_rope(float *q, float *k, const float *q_norm_w,
                                    const float *k_norm_w, int n_head,
                                    int n_head_kv, int head_dim, float eps,
                                    int rope_dim, int pos, float theta_base) {
    extern __shared__ float sdata[];

    int b = blockIdx.x;
    if (b >= n_head + n_head_kv)
        return;
    float *h;
    const float *w;
    if (b < n_head) {
        h = q + (size_t)b * head_dim;
        w = q_norm_w;
    } else {
        h = k + (size_t)(b - n_head) * head_dim;
        w = k_norm_w;
    }

    int tid = threadIdx.x;
    int threads = blockDim.x;

    /* --- fc_qk_rms_norm body (verbatim) --- */
    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += threads) {
        float v = h[i];
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / (float)head_dim + eps);
    float inv_rms = 1.0f / rms;
    for (int i = tid; i < head_dim; i += threads) {
        float v = h[i] * inv_rms;
        if (w)
            v *= w[i];
        h[i] = v;
    }
    __syncthreads();   /* rope reads what other threads normed */

    /* --- fc_rope_neox_partial body (verbatim) --- */
    int rot = (rope_dim > 0 && rope_dim < head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    int j = tid;
    if (j < half) {
        float freq = 1.0f / powf(theta_base, (float)(2 * j) / (float)rot);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float x0 = h[j];
        float x1 = h[j + half];
        /* Pinned to the ORIGINAL rope kernel's SASS (ptxas contracts its
         * PTX mul/mul/sub): h[j] = fma(x1, -sin, x0*cos),
         * h[j+half] = fma(x1, cos, x0*sin) — one pre-rounded mul + one fused
         * fma each, verified against nvdisasm of rope.ptx->cubin. */
        h[j]        = __fmaf_rn(x1, -sin_a, __fmul_rn(x0, cos_a));
        h[j + half] = __fmaf_rn(x1, cos_a, __fmul_rn(x0, sin_a));
    }
}

} /* extern "C" */
