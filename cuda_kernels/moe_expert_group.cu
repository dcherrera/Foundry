/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Portions modelled on ggml (MIT License, Copyright (c) 2023-2026 The ggml
 * authors, https://github.com/ggml-org/ggml), modified. See NOTICE. */
/*
 * moe_expert_group.cu — grouped top-k expert SwiGLU for Qwen3.5-MoE (Foundry)
 *
 * Target: sm_37 (K80, Kepler)
 * Compile: nvcc -ptx -arch=sm_37 moe_expert_group.cu -o moe_expert_group.ptx
 *
 * Replaces the per-expert launch storm in the MoE FFN. The old path launched,
 * per layer, 8 x (gate matvec + up matvec + silu + mul + down matvec +
 * weighted_add) = 48 kernels plus a host sync + top-k read-back to drive the
 * dispatch. These kernels do the whole routed-expert block in TWO launches,
 * reading the router's top_idx/top_wt selection directly from device memory:
 *
 *   1. fc_moe_gateup_q3k — warp-per-row over (ff, k): computes
 *      hid[s*ff+row] = silu(gate[e_s].row . x) * (up[e_s].row . x)
 *      for e_s = top_idx[s]. Q3_K weights, both matrices contiguous 3D
 *      [n_expert, ff, cols] uploads (one fc_buffer per matrix per layer).
 *   2. fc_moe_down_q3k / fc_moe_down_q4k — COOPERATIVE multi-warp-per-row-pair
 *      over n_embd: wpp warps split the k selected experts of one output row
 *      pair (warp wk takes slots wk, wk+wpp, ...), each accumulating weighted
 *      per-lane partials; one __shfl_down tree per warp plus a tiny shared-mem
 *      cross-warp sum then produce
 *      out[row] = sum_s top_wt[s] * (down[e_s].row . hid_s).
 *      Writes out directly (no zero + weighted_add passes). The 512-col down
 *      dot is only 2 super-blocks, so a lone warp per row-pair stalls on
 *      memory latency; wpp cooperating warps multiply the loads in flight.
 *      wpp is a launch arg — tuned default in fc_cuda.h (FC_MOE_DOWN_WPP).
 *
 * Memory paths (measured on a K80, nvprof): weight bytes use PLAIN
 * global loads (L2, 32 B sectors) so the streamed-once weights stay off
 * the read-only/tex cache. The down kernels read the tiny hid vector
 * (16 KB, shared by every block) via __ldg from that now-private RO cache
 * — this replaces the old per-slot shared staging, whose __syncthreads
 * serialized the k-slot loop into latency-bound rounds (0.73 achieved
 * occupancy -> 0.84 after). gateup keeps x staged in shared (each warp
 * re-reads every x element 8x; LDS measured faster than __ldg there).
 *
 * Plus two small fusions for the always-on shared expert:
 *   3. fc_moe_silu_mul       — out = silu(a) * b in one pass (was 2 launches)
 *   4. fc_moe_shexp_dot_gate — gate_logit = gate_w . x on-device, then
 *      shexp *= sigmoid(gate_logit)
 *
 * Weights are in the REPACKED device layouts produced by fc_repack_weight at
 * upload (documented in fc_cuda.h and matvec_q3k.cu / matvec_kquants.cu):
 * Q3_K and Q4_K 152 B aligned super-blocks laid out for the shift-free,
 * cvt-free MAGIC-NUMBER dequant (sm_37 runs int-mul / int32->f32 cvt at
 * 32/clk and shifts at 64/clk; the magic decode is AND + OR + one exact FMA
 * per field — see matvec_q3k.cu for the full derivation). The lane dots
 * below are duplicated from those files (PTX modules are standalone) and
 * keep the identical per-lane fp op order. The gateup kernel stays
 * bit-identical to the cvt-based kernels; the cooperative down kernels
 * reassociate only the final slot-weighted reduction (per-warp slot subsets
 * + cross-warp add) — every op is pinned with explicit __fmaf_rn/adds and
 * reproduced exactly by the CPU replica in cuda/tests/bench_matvec.c, and
 * the end-to-end argmax gate bounds the reassociation drift.
 *
 * QUANT COVERAGE. gate+up and down each come in a Q3_K and a Q4_K flavor, and
 * the HOST dispatches both on the tensor's gguf type (fc_moe_gateup_ctx /
 * fc_moe_down_accum_ctx, ../cuda/fc_cuda.c). Q3_K_M mixes them within one file
 * (its merged ffn_down_exps is Q3_K on half the layers and Q4_K on the other
 * half); a pure Q4_K_M model is Q4_K throughout, gate/up included, which is why
 * fc_moe_gateup_q4k exists. DOWN additionally has a Q6_K flavor, because the
 * Q4_K_M puts Q6_K on half the layers' merged ffn_down_exps. A Q8_0 file has
 * Q8_0 gate/up/down throughout (fc_moe_*_q80, on their own 288-B expert
 * layout — see FC_DEF_Q80_GRP2). The DEDUP gate+up
 * variants are Q3_K-only — see the dedup block comment below.
 *
 * PTX symbols emitted (for embedding into ../cuda/fc_cuda_ptx.h):
 *   FC_PTX_MOE_EXPERT_GROUP -> kernels fc_moe_gateup_q3k, fc_moe_gateup_q4k,
 *     fc_moe_down_q3k, fc_moe_down_q4k, fc_moe_down_q6k, fc_moe_silu_mul,
 *     fc_moe_shexp_dot_gate, plus the batched
 *     _nx variants and the cross-user dedup pair fc_moe_dedup_build /
 *     fc_moe_gateup_q3k_dedup (+ _shx, and + the compacted-grid pair
 *     fc_moe_dedup_build_wl / fc_moe_gateup_q3k_dedup_shxc in a
 *     -DFC_MOE_DEDUP_COMPACT=1 build only)
 *
 * The _shx / _shxc x-staging loop is selected by FC_MOE_SHX_STAGE
 * (moe_dedup_shx_staging.cuh), default 0 = the shipped form; at 0 this module's
 * PTX is byte-identical to the checked-in artifact.
 *
 * Kepler notes: pure fp32; __shfl_down is the non-_sync Kepler variant.
 */

extern "C" {

/* Repacked block sizes (bytes) — must match fc_repack_weight. */
#define FC_Q3K_RBLOCK_BYTES 152
#define FC_Q4K_RBLOCK_BYTES 152
#define FC_Q6K_RBLOCK_BYTES 216

/* Magic decodes (duplicated from matvec_q3k.cu / matvec_kquants.cu):
 * exact float (v - bias) from the masked field at bit p. */
#define FC_MAGIC_NIB(w, p, bias)                                            \
    __fmaf_rn(__int_as_float(0x4B000000u | ((w) & (0xFu << (p)))),          \
              1.0f / (float)(1u << (p)),                                    \
              -(float)((1u << (23 - (p))) + (bias)))
#define FC_MAGIC_BYTE(w, p, bias)                                           \
    __fmaf_rn(__int_as_float(0x4B000000u | ((w) & (0xFFu << (p)))),         \
              1.0f / (float)(1u << (p)),                                    \
              -(float)((1u << (23 - (p))) + (bias)))
/* 2-bit field at bit p, scaled by 16 — Q6_K's high-bit pair. */
#define FC_MAGIC_BIT2X16(w, p)                                              \
    __fmaf_rn(__int_as_float(0x4B000000u | ((w) & (3u << (p)))),            \
              16.0f / (float)(1u << (p)), -(float)(1u << (27 - (p))))

/* Byte-aligned fields decode fastest through Kepler's 128/clk 8-bit cvt
 * (single I2F.F32.U8 with byte select — cheaper than the magic FMA, which
 * stays for the sub-byte fields). */
#define FC_CVT_BYTE(w, k) ((float)(((w) >> (8 * (k))) & 0xFFu))

__device__ __forceinline__ float fc_warp_sum_grp(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v, 8);
    v += __shfl_down(v, 4);
    v += __shfl_down(v, 2);
    v += __shfl_down(v, 1);
    return v;
}

/* x-load flavors for the lane dots below: the down kernels read the tiny
 * hid vector straight from GLOBAL via __ldg (16 KB, read-only-cache
 * resident), while the gateup kernel reads its block-staged x from SHARED
 * (plain load — __ldg is global-only). */
#define FC_XLOAD_LDG(p, off) __ldg((p) + (off))
#define FC_XLOAD_PLAIN(p, off) (p)[off]

/*
 * Interleaved ROW-PAIR lane dot: the two rows' 8-element FMA chains advance
 * in lockstep per element (two independent chains the scheduler can
 * dual-issue, unlike two back-to-back dependent 8-FMA chains), sharing one
 * x load per element. Each row's per-lane chain keeps the exact i = 0..7
 * order of fc_q3k_sb_lane_dot -> bit-exact per row.
 * Q3_K repacked super-block layout (152 B): u32 qw[32] @0 (nibble i =
 * qv | hbit<<2, value = nibble - 4), u8 sc[16] @128 (raw 6-bit, value =
 * sc - 32, reordered per lane half), f32 d @144.
 * Weights use PLAIN global loads (L2, 32 B sectors) — they stream through
 * once, so keeping them off the read-only/tex path leaves that cache to
 * the reused activation vector.
 */
#define FC_DEF_Q3K_GRP2(NAME, XLOAD)                                          \
__device__ __forceinline__ void NAME(const unsigned char *blk0,               \
                                     const unsigned char *blk1,               \
                                     const float *xb, int lane,               \
                                     float *p0, float *p1) {                  \
    unsigned int wv0 = ((const unsigned int *)blk0)[lane];                    \
    unsigned int wv1 = ((const unsigned int *)blk1)[lane];                    \
    float d0 = *(const float *)(blk0 + 144);                                  \
    float d1 = *(const float *)(blk1 + 144);                                  \
    uint2 sw0 = ((const uint2 *)(blk0 + 128))[lane >> 4];                     \
    uint2 sw1 = ((const uint2 *)(blk1 + 128))[lane >> 4];                     \
    unsigned int wh0 = wv0 >> 16, wh1 = wv1 >> 16;                            \
                                                                              \
    float a0 = 0.0f, a1 = 0.0f;                                               \
_Pragma("unroll")                                                             \
    for (int i = 0; i < 8; i++) {                                             \
        int qp = (i <= 4) ? 4 * i : 4 * i - 16;                               \
        unsigned int q0 = (i <= 4) ? wv0 : wh0;                               \
        unsigned int q1 = (i <= 4) ? wv1 : wh1;                               \
        unsigned int s0 = (i < 4) ? sw0.x : sw0.y;                            \
        unsigned int s1 = (i < 4) ? sw1.x : sw1.y;                            \
        float xv = XLOAD(xb, 32 * i + lane);                                  \
        float fq0 = FC_MAGIC_NIB(q0, qp, 4);                                  \
        float fs0 = __fsub_rn(FC_CVT_BYTE(s0, i & 3), 32.0f);                 \
        float fq1 = FC_MAGIC_NIB(q1, qp, 4);                                  \
        float fs1 = __fsub_rn(FC_CVT_BYTE(s1, i & 3), 32.0f);                 \
        a0 = __fmaf_rn(__fmul_rn(fs0, fq0), xv, a0);                          \
        a1 = __fmaf_rn(__fmul_rn(fs1, fq1), xv, a1);                          \
    }                                                                         \
    *p0 += d0 * a0;                                                           \
    *p1 += d1 * a1;                                                           \
}

FC_DEF_Q3K_GRP2(fc_q3k_sb_lane_grp2, FC_XLOAD_LDG)     /* down: global hid */
FC_DEF_Q3K_GRP2(fc_q3k_sb_lane_grp2s, FC_XLOAD_PLAIN)  /* gateup: shared x */


/* Interleaved ROW-PAIR Q4_K lane dot: both rows' chains advance per element
 * (independent chains for dual-issue), sharing one x load. Each row keeps the
 * exact per-element op order of fc_q4k_sb_lane_dot -> bit-exact. Q4_K repacked
 * super-block layout (152 B): u32 qw[32] @0 (byte j = qs[32j+l]), u8 sc[8]
 * @128, u8 mn[8] @136, f32 d @144, f32 dmin @148.
 *
 * Two XLOAD flavors, exactly like the Q3_K pair above: the down kernels read
 * the tiny hid vector from GLOBAL via __ldg, the gate+up kernels read their
 * block-staged x from SHARED. Made a macro (rather than a second copy) when
 * the Q4_K gate+up kernels were added; the _grp2 instantiation expands to the
 * same token stream the hand-written function had, so the down kernels' PTX is
 * unchanged. */
#define FC_DEF_Q4K_GRP2(NAME, XLOAD)                                          \
__device__ __forceinline__ void NAME(const unsigned char *blk0,               \
                                     const unsigned char *blk1,               \
                                     const float *xb, int lane,               \
                                     float *p0, float *p1) {                  \
    /* weights: PLAIN global loads — see fc_q3k_sb_lane_grp2 */               \
    unsigned int qw0 = ((const unsigned int *)blk0)[lane];                    \
    unsigned int qw1 = ((const unsigned int *)blk1)[lane];                    \
    uint2 sw0 = *(const uint2 *)(blk0 + 128);                                 \
    uint2 sw1 = *(const uint2 *)(blk1 + 128);                                 \
    uint2 mw0 = *(const uint2 *)(blk0 + 136);                                 \
    uint2 mw1 = *(const uint2 *)(blk1 + 136);                                 \
    float2 dd0 = *(const float2 *)(blk0 + 144);                               \
    float2 dd1 = *(const float2 *)(blk1 + 144);                               \
    unsigned int qh0 = qw0 >> 16, qh1 = qw1 >> 16;                            \
    float a0 = 0.0f, a1 = 0.0f;                                               \
_Pragma("unroll")                                                             \
    for (int j = 0; j < 4; j++) {                                             \
_Pragma("unroll")                                                             \
        for (int h = 0; h < 2; h++) {                                         \
            int si = 2 * j + h;                                               \
            int nbit = 8 * j + 4 * h;                                         \
            int qp = (nbit <= 16) ? nbit : nbit - 16;                         \
            unsigned int q0 = (nbit <= 16) ? qw0 : qh0;                       \
            unsigned int q1 = (nbit <= 16) ? qw1 : qh1;                       \
            unsigned int s0 = (si < 4) ? sw0.x : sw0.y;                       \
            unsigned int s1 = (si < 4) ? sw1.x : sw1.y;                       \
            unsigned int m0 = (si < 4) ? mw0.x : mw0.y;                       \
            unsigned int m1 = (si < 4) ? mw1.x : mw1.y;                       \
            float xv = XLOAD(xb, 64 * j + 32 * h + lane);                     \
            float t0 = __fsub_rn(                                             \
                __fmul_rn(__fmul_rn(dd0.x, FC_CVT_BYTE(s0, si & 3)),          \
                          FC_MAGIC_NIB(q0, qp, 0)),                           \
                __fmul_rn(dd0.y, FC_CVT_BYTE(m0, si & 3)));                   \
            float t1 = __fsub_rn(                                             \
                __fmul_rn(__fmul_rn(dd1.x, FC_CVT_BYTE(s1, si & 3)),          \
                          FC_MAGIC_NIB(q1, qp, 0)),                           \
                __fmul_rn(dd1.y, FC_CVT_BYTE(m1, si & 3)));                   \
            a0 = __fmaf_rn(t0, xv, a0);                                       \
            a1 = __fmaf_rn(t1, xv, a1);                                       \
        }                                                                     \
    }                                                                         \
    *p0 += a0;                                                                \
    *p1 += a1;                                                                \
}

FC_DEF_Q4K_GRP2(fc_q4k_sb_lane_grp2, FC_XLOAD_LDG)     /* down: global hid */
FC_DEF_Q4K_GRP2(fc_q4k_sb_lane_grp2s, FC_XLOAD_PLAIN)  /* gateup: shared x */


/* Interleaved ROW-PAIR Q6_K lane dot — duplicated from fused_layer.cu's
 * fc_fu_q6k_acc2 / matvec_kquants.cu's fc_q6k_sb_lane_dot (PTX modules are
 * standalone) with the identical per-element op order, so it is bit-exact with
 * both. Q6_K repacked super-block layout (216 B): u32 ql[32] @0 (byte j =
 * ql[32j+l]), u16 qh[32] @128, u8 sc[16] @192 (SIGNED, biased +128 in the
 * repack), f32 d @208.
 *
 * Exists because the Q4_K_M quantization of a Qwen3.5-MoE puts Q6_K on half
 * the layers' merged ffn_down_exps (the other half is Q4_K) — same shape of
 * mixture as the Q3_K_M mini, one quant up. Without it those layers lose the
 * grouped path entirely and fall back to 3*n_expert per-expert buffers a
 * layer, which is what actually makes a Q4_K_M model unloadable on a K80 pair.
 * Down-only: the gate+up matrices are never Q6_K in these files. */
#define FC_DEF_Q6K_GRP2(NAME, XLOAD)                                          \
__device__ __forceinline__ void NAME(const unsigned char *blk0,               \
                                     const unsigned char *blk1,               \
                                     const float *xb, int lane,               \
                                     float *p0, float *p1) {                  \
    unsigned int ql0 = ((const unsigned int *)blk0)[lane];                    \
    unsigned int ql1 = ((const unsigned int *)blk1)[lane];                    \
    unsigned int qh20 = ((const unsigned short *)(blk0 + 128))[lane];         \
    unsigned int qh21 = ((const unsigned short *)(blk1 + 128))[lane];         \
    float d0 = *(const float *)(blk0 + 208);                                  \
    float d1 = *(const float *)(blk1 + 208);                                  \
    uint2 sw0 = ((const uint2 *)(blk0 + 192))[lane >> 4];                     \
    uint2 sw1 = ((const uint2 *)(blk1 + 192))[lane >> 4];                     \
    unsigned int qlh0 = ql0 >> 16, qlh1 = ql1 >> 16;                          \
    float a0 = 0.0f, a1 = 0.0f;                                               \
_Pragma("unroll")                                                             \
    for (int half = 0; half < 2; half++) {                                    \
_Pragma("unroll")                                                             \
        for (int quad = 0; quad < 4; quad++) {                                \
            int i = 4 * half + quad;                                          \
            int nbit = 16 * half + 8 * (quad & 1) + 4 * ((quad >> 1) & 1);    \
            int hbit = 8 * half + 2 * quad;                                   \
            int qp = (nbit <= 16) ? nbit : nbit - 16;                         \
            unsigned int q0 = (nbit <= 16) ? ql0 : qlh0;                      \
            unsigned int q1 = (nbit <= 16) ? ql1 : qlh1;                      \
            unsigned int s0 = (i < 4) ? sw0.x : sw0.y;                        \
            unsigned int s1 = (i < 4) ? sw1.x : sw1.y;                        \
            float xv = XLOAD(xb, 32 * i + lane);                              \
            float fq0 = __fadd_rn(FC_MAGIC_NIB(q0, qp, 32),                   \
                                  FC_MAGIC_BIT2X16(qh20, hbit));              \
            float fq1 = __fadd_rn(FC_MAGIC_NIB(q1, qp, 32),                   \
                                  FC_MAGIC_BIT2X16(qh21, hbit));              \
            float fs0 = __fsub_rn(FC_CVT_BYTE(s0, i & 3), 128.0f);            \
            float fs1 = __fsub_rn(FC_CVT_BYTE(s1, i & 3), 128.0f);            \
            a0 = __fmaf_rn(__fmul_rn(fs0, fq0), xv, a0);                      \
            a1 = __fmaf_rn(__fmul_rn(fs1, fq1), xv, a1);                      \
        }                                                                     \
    }                                                                         \
    *p0 = __fmaf_rn(d0, a0, *p0);                                             \
    *p1 = __fmaf_rn(d1, a1, *p1);                                             \
}

FC_DEF_Q6K_GRP2(fc_q6k_sb_lane_grp2, FC_XLOAD_LDG)     /* down: global hid */


/*
 * Interleaved ROW-PAIR F16 lane dot — gguf type 1, the TOP of this hardware's
 * precision ladder (F32 storage does not fit: 35B x 4 B = 140 GB against 97.6
 * GB of VRAM).
 *
 * sm_37 HAS NO FP16 ARITHMETIC — that starts at sm_53 — so this is not a copy
 * of the fp32 path any more than it is of the quant ones: the weight is LOADED
 * as 2 bytes and converted IN-REGISTER to fp32, and every multiply/add below is
 * the same fp32 op the quant kernels use. cvt.f32.f16 is a STORAGE conversion
 * and has existed since sm_20; it is what `__half2float` compiles to. Written
 * as asm so this file stays header-free (no cuda_fp16.h), which is what keeps
 * the checked-in PTX of every kernel above byte-identical when this block is
 * added. Exact for subnormals and inf/nan — a widening cvt takes no .ftz.
 *
 * There is no block structure and no scale here, so an F16 "super-block" is
 * just the same 256-element step the K-quant kernels already iterate in: 512 B
 * of halves. FC_MOE_GATEUP_BODY / FC_MOE_DOWN_BODY are therefore reused
 * verbatim with BLOCK_BYTES = 512 and row_stride = cols * 2. There is also no
 * fc_repack_weight pass — the GGUF bytes ARE the device layout, which is what
 * keeps a ~70 GB upload from needing a second host copy of every tensor on a
 * 7.8 GB host.
 *
 * LOAD WIDTH. One 2-byte load per element at the same 32*i+lane lane->element
 * map the K-quant dots use, so x staging, x indexing and the reduction tree are
 * untouched. The warp's 32 lanes read 64 CONTIGUOUS bytes per i, so every load
 * is fully coalesced and no DRAM byte is wasted: 16 32-B sectors per 512-B
 * super-block, 100% efficiency.
 *
 * That is NOT the same as saying the load width does not matter, and an earlier
 * version of this comment said it was — that a memory-bound kernel has idle
 * issue slots either way. Coalescing settles DRAM bytes; it says nothing about
 * how many LSU transactions those bytes cost. A warp-wide 16-bit load moves 64
 * B in one transaction against a 128-B transaction budget, so the scalar form
 * spends 8 half-empty transactions per 512-B super-block where a uint2 form
 * spends 4 full ones. Measured: this kernel runs at 139 GB/s and the uint2 form
 * fc_f16_sb_lane_grp2v below runs the same bytes at 181, so this map is leaving
 * 1.3x on the table. See the FC_DEF_VEC16_GRP2 header for the numbers and for
 * the lane-map change that is the price of them.
 */
#define FC_F16_RBLOCK_BYTES 512

__device__ __forceinline__ float fc_f16_to_f32(unsigned short h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

#define FC_DEF_F16_GRP2(NAME, XLOAD)                                          \
__device__ __forceinline__ void NAME(const unsigned char *blk0,               \
                                     const unsigned char *blk1,               \
                                     const float *xb, int lane,               \
                                     float *p0, float *p1) {                  \
    /* weights: PLAIN global loads — see fc_q3k_sb_lane_grp2 */               \
    const unsigned short *h0 = (const unsigned short *)blk0;                  \
    const unsigned short *h1 = (const unsigned short *)blk1;                  \
    float a0 = 0.0f, a1 = 0.0f;                                               \
_Pragma("unroll")                                                             \
    for (int i = 0; i < 8; i++) {                                             \
        int el = 32 * i + lane;                                               \
        float xv = XLOAD(xb, el);                                             \
        a0 = __fmaf_rn(fc_f16_to_f32(h0[el]), xv, a0);                        \
        a1 = __fmaf_rn(fc_f16_to_f32(h1[el]), xv, a1);                        \
    }                                                                         \
    *p0 += a0;                                                                \
    *p1 += a1;                                                                \
}

FC_DEF_F16_GRP2(fc_f16_sb_lane_grp2, FC_XLOAD_LDG)     /* down: global hid */
FC_DEF_F16_GRP2(fc_f16_sb_lane_grp2s, FC_XLOAD_PLAIN)  /* gateup: shared x */


/*
 * Interleaved ROW-PAIR BF16 lane dot — gguf type 30. Same 2-byte storage and
 * same 512 B super-block as F16, and it exists for a reason F16 cannot serve:
 * the model SHIPS as bf16 safetensors, so bf16 is the only format that carries
 * the published weight VALUES. Converting bf16 -> F16 re-rounds every weight
 * (bf16 keeps 7 stored mantissa bits, F16 keeps 10, and the two mantissa fields
 * do not nest), so an F16 run measures a perturbed model. bf16 is therefore the
 * control and F16 is the variant, not the other way round.
 *
 * IT IS NOT CHEAPER TO DECODE THAN F16, which is worth writing down because the
 * arithmetic says it should be. A bf16 value IS the top 16 bits of the fp32 bit
 * pattern, so widening is a shift and a free reinterpret, and the
 * compute-capability 3.7 throughput table prices a 32-bit shift at 64 ops/clk/SM
 * against 32 for "all other type conversions", which is where cvt.f32.f16 lands.
 * Two arguments for bf16, and the SASS refutes both:
 *
 *   fc_moe_gateup_f16    518 instructions,  16 F2F,  0 PRMT
 *   fc_moe_gateup_bf16   518 instructions,   0 F2F, 17 PRMT
 *
 * Same instruction count. ptxas implements the 16-bit shift as one PRMT (byte
 * permute), not the two-op sequence the C suggests, so the widen is one
 * instruction either way; and the throughput difference between the two classes
 * never bills, because this kernel is memory-bound and its issue slots are not
 * the constraint. Measured on die 7 at the per-layer shape: bf16 0.2400 ms vs
 * f16 0.2418 ms on gate+up and 0.1300 vs 0.1304 on down — under 1%, inside the
 * run-to-run spread, and it stays under 1% in the VEC form too even though the
 * vec decodes differ in instruction count (see FC_DEF_VEC16_GRP2). So bf16 is
 * here for FIDELITY — it is the only format holding the shipped values — and
 * not for speed. Do not repeat the shift-beats-cvt argument; it has been
 * measured, twice, and it does not reach the clock.
 *
 * Exactness: every bf16 is representable in fp32 with the mantissa zero-padded,
 * including subnormals (a bf16 subnormal shifts to an fp32 subnormal with the
 * same value — no renormalisation, which is the step F16 widening must do) and
 * inf/nan (the payload rides along in the shifted mantissa). So the host replica
 * is the identical shift and the compare is bit-exact for the dot.
 */
#define FC_BF16_RBLOCK_BYTES 512

__device__ __forceinline__ float fc_bf16_to_f32(unsigned short b) {
    return __int_as_float((unsigned int)b << 16);
}

#define FC_DEF_BF16_GRP2(NAME, XLOAD)                                         \
__device__ __forceinline__ void NAME(const unsigned char *blk0,               \
                                     const unsigned char *blk1,               \
                                     const float *xb, int lane,               \
                                     float *p0, float *p1) {                  \
    /* weights: PLAIN global loads — see fc_q3k_sb_lane_grp2 */               \
    const unsigned short *h0 = (const unsigned short *)blk0;                  \
    const unsigned short *h1 = (const unsigned short *)blk1;                  \
    float a0 = 0.0f, a1 = 0.0f;                                               \
_Pragma("unroll")                                                             \
    for (int i = 0; i < 8; i++) {                                             \
        int el = 32 * i + lane;                                               \
        float xv = XLOAD(xb, el);                                             \
        a0 = __fmaf_rn(fc_bf16_to_f32(h0[el]), xv, a0);                       \
        a1 = __fmaf_rn(fc_bf16_to_f32(h1[el]), xv, a1);                       \
    }                                                                         \
    *p0 += a0;                                                                \
    *p1 += a1;                                                                \
}

FC_DEF_BF16_GRP2(fc_bf16_sb_lane_grp2, FC_XLOAD_LDG)    /* down: global hid */
FC_DEF_BF16_GRP2(fc_bf16_sb_lane_grp2s, FC_XLOAD_PLAIN) /* gateup: shared x */


/*
 * VECTORIZED (uint2) 16-bit lane dot — and it is worth 1.3x.
 *
 * WHY IT PAYS. The scalar dots above are perfectly coalesced but each warp-wide
 * 16-bit load moves only 64 B, and an LSU transaction can carry 128 B. Eight
 * such loads per 512-B super-block therefore cost 8 transactions at 50%
 * occupancy. Loading a uint2 (four 16-bit weights, 8 B/lane) makes the warp read
 * 256 CONTIGUOUS bytes per instruction = 2 full transactions, so a super-block
 * costs 2 instructions / 4 transactions — the same DRAM bytes through half the
 * LSU transactions. uint4 would issue 1 instruction for the same 4 transactions
 * (no further transaction saving) while forcing an 8-float-per-lane stride on
 * the staged x, which is an 8-way shared-memory bank conflict on Kepler's 4-byte
 * banks; uint2 pairs with a float4 x load, the conflict-free pattern. So uint2
 * is the widest useful width here, not a step on the way to uint4.
 *
 * MEASURED, die 7, the per-layer shape (k=8, ff=512, cols=2048), median of
 * three runs that agreed to 0.3%, with the k experts forced DISTINCT so that
 * nominal bytes and DRAM bytes are the same number:
 *
 *   gateup  f16   0.2418 ms  138.8 GB/s  ->  f16-vec   0.1849 ms  181.4 GB/s
 *   gateup  bf16  0.2400 ms  139.8 GB/s  ->  bf16-vec  0.1844 ms  181.9 GB/s
 *   down    f16   0.1304 ms  128.6 GB/s  ->  f16-vec   0.0986 ms  170.2 GB/s
 *   down    bf16  0.1300 ms  129.1 GB/s  ->  bf16-vec  0.0980 ms  171.2 GB/s
 *
 * 1.30-1.33x on all four, and 182 GB/s is 76% of a K80 die's 240 GB/s — the
 * band a well-tuned streaming kernel reaches on Kepler. The scalar form's 58%
 * was the half-empty transactions and nothing else. The SASS confirms the
 * mechanism rather than inferring it — LD.E.U16 x16 + LDS x8 becomes LD.E.64
 * x12 + LDS.128 x6 for gateup, and LD.E.U16 x16 + LDG.E.T becomes LD.E.64 x12 +
 * LDG.E.128 for down. It is paid for in registers, but not enough to matter:
 * gateup 48 -> 52 and down 50 -> 56 (the _nx forms 53 -> 64), all under the 63
 * that would cost a block of occupancy at 256 threads, and 0 bytes of stack
 * frame either way — read the STACK FRAME line, not the spill counters, which
 * hide local arrays.
 *
 * This is the first kernel in this tree where a load-side change converts to
 * wall clock. The standing findings that the K80 is launch-bound and that work
 * reduction does not convert were both learned from the K-quant kernels, which
 * are instruction-bound; they do not transfer to a kernel whose bottleneck is
 * the memory pipe, and this is the counter-example.
 *
 * The two formats stay within 1% of each other in the vec form as well, even
 * though the vec decodes are no longer the same shape: F2F.F32.F16 carries a
 * half-word selector so an F16 pair is 2 F2F, while a bf16 pair is a mask plus
 * a permute — 48 F2F for f16-vec against 39 LOP32I + 25 PRMT for bf16-vec, at
 * the same 595 instructions total. Even at 1.3x the kernel is still short of
 * the memory pipe, so the decode difference does not reach the clock.
 *
 * LANE MAP CONSEQUENCE — this is not a drop-in. The scalar map gives lane l the
 * strided set {32i + l}, which is exactly why it cannot be vectorized: those 8
 * halves are not contiguous. The vec map gives lane l the contiguous runs
 * {4l..4l+3} and {128 + 4l..128 + 4l+3}. Same 256 products per warp, different
 * PARTITION of them across lanes, so the per-lane fmaf chains group differently
 * and the shfl tree sums a different association. The result is therefore NOT
 * bit-identical to the scalar kernel and must not be compared against it — the
 * test carries a second replica built on the vec map and compares bit-exactly
 * against THAT, which is what makes the comparison able to fail.
 *
 */
#define FC_XLOAD4_LDG(p, i)   __ldg((const float4 *)(p) + (i))
#define FC_XLOAD4_PLAIN(p, i) (((const float4 *)(p))[i])

/* Element pair packed little-endian in one 32-bit word: LO is the even element
 * (low half), HI the odd one. The BF16 pair needs no conversion instruction —
 * masking off the low half IS the fp32 value of the odd element, and shifting
 * the low half up IS the even one — but that costs no fewer instructions than
 * the F16 pair, because F2F.F32.F16 carries its own half-word selector and so
 * reads either half of the word directly. Measured, both ways: see the
 * fc_bf16_to_f32 header. */
#define FC_F16_LO(w)  fc_f16_to_f32((unsigned short)((w) & 0xFFFFu))
#define FC_F16_HI(w)  fc_f16_to_f32((unsigned short)((w) >> 16))
#define FC_BF16_LO(w) __int_as_float((w) << 16)
#define FC_BF16_HI(w) __int_as_float((w) & 0xFFFF0000u)

#define FC_DEF_VEC16_GRP2(NAME, CVT_LO, CVT_HI, XLOAD4)                       \
__device__ __forceinline__ void NAME(const unsigned char *blk0,               \
                                     const unsigned char *blk1,               \
                                     const float *xb, int lane,               \
                                     float *p0, float *p1) {                  \
    float a0 = 0.0f, a1 = 0.0f;                                               \
_Pragma("unroll")                                                             \
    for (int h = 0; h < 2; h++) {                                             \
        /* 8 B/lane x 32 lanes = 256 contiguous bytes per warp instruction */ \
        uint2 w0 = ((const uint2 *)(blk0 + 256 * h))[lane];                   \
        uint2 w1 = ((const uint2 *)(blk1 + 256 * h))[lane];                   \
        float4 xv = XLOAD4(xb + 128 * h, lane);                               \
        /* the two rows' chains stay interleaved for dual-issue, exactly as   \
         * in the scalar dots; each row's own order is element-ascending */   \
        a0 = __fmaf_rn(CVT_LO(w0.x), xv.x, a0);                               \
        a1 = __fmaf_rn(CVT_LO(w1.x), xv.x, a1);                               \
        a0 = __fmaf_rn(CVT_HI(w0.x), xv.y, a0);                               \
        a1 = __fmaf_rn(CVT_HI(w1.x), xv.y, a1);                               \
        a0 = __fmaf_rn(CVT_LO(w0.y), xv.z, a0);                               \
        a1 = __fmaf_rn(CVT_LO(w1.y), xv.z, a1);                               \
        a0 = __fmaf_rn(CVT_HI(w0.y), xv.w, a0);                               \
        a1 = __fmaf_rn(CVT_HI(w1.y), xv.w, a1);                               \
    }                                                                         \
    *p0 += a0;                                                                \
    *p1 += a1;                                                                \
}

FC_DEF_VEC16_GRP2(fc_f16_sb_lane_grp2v,   FC_F16_LO,  FC_F16_HI,
                  FC_XLOAD4_LDG)                       /* down: global hid */
FC_DEF_VEC16_GRP2(fc_f16_sb_lane_grp2vs,  FC_F16_LO,  FC_F16_HI,
                  FC_XLOAD4_PLAIN)                     /* gateup: shared x */
FC_DEF_VEC16_GRP2(fc_bf16_sb_lane_grp2v,  FC_BF16_LO, FC_BF16_HI,
                  FC_XLOAD4_LDG)                       /* down: global hid */
FC_DEF_VEC16_GRP2(fc_bf16_sb_lane_grp2vs, FC_BF16_LO, FC_BF16_HI,
                  FC_XLOAD4_PLAIN)                     /* gateup: shared x */


/*
 * Interleaved ROW-PAIR Q8_0 lane dot — gguf type 8, the rung between the
 * K-quants and F16 (~1.06 B/weight in the file, 1.125 on the device).
 *
 * EXPERT LAYOUT, not the dense one. The dense fc_matvec_q80 reads the 36-B
 * [f32 d][32 s8] blocks fc_repack_weight produces, one byte per lane, which is
 * a 32-B warp load per block. The expert tensors are the bulk of the model, so
 * they get their own upload-time layout (fc_moe_repack_weight): every 256
 * elements (8 Q8_0 blocks) become ONE 288-B super-block
 *
 *     [8 x f32 d][256 x s8 q]       (same bytes as 8 x 36, regrouped)
 *
 * which lets a warp read the 256 quants as two fully-used 128-B transactions
 * and the 8 scales as one 32-B sector. Same size as the dense repack.
 *
 * LANE MAP: the uint2-kernel map of FC_DEF_VEC16_GRP2 — lane l owns the runs
 * {4l..4l+3} (block l>>3) and {128+4l..128+4l+3} (block 4+(l>>3)), so each
 * half is one u32 load per lane and one conflict-free float4 x load.
 *
 * OP ORDER (per lane, per super-block, per row): a = sum of the 4 products of
 * the first run as an fmaf chain from 0, b = the same over the second run, then
 * p = fmaf(d_a, a, p); p = fmaf(d_b, b, p). The scale is applied once per
 * 4-element run rather than once per element (dense q80 rounds d*q first);
 * the CPU replica in cuda/tests/test_moe_expert_quants.c reproduces exactly
 * this order, so the dots are bit-exact. q is an exact small integer in fp32
 * and the s8 -> f32 conversion is one I2F with a byte selector.
 */
#define FC_Q80_SB_BYTES 288

/* One lane's share of one 288-B super-block: its two quant words (runs
 * {4l..4l+3} and {128+4l..128+4l+3}) and their two scales. Weights load via
 * __ldg here, unlike the K-quant / 16-bit plain loads (see
 * fc_q3k_sb_lane_grp2): on the Qwen3.6 expert shape (k 8, ff 512, cols 2048,
 * one K80 die) gate+up ran 159.7 -> 118.8 us and down 82.3 -> 79.1 us, same
 * bits (drafts/k80_q8_speedups/moeprobe.cu). */
struct fc_q80_sbl {
    unsigned int w[2];
    float s[2];
};

__device__ __forceinline__ fc_q80_sbl fc_q80_sb_load(const unsigned char *blk,
                                                     int lane) {
    const unsigned int *q = (const unsigned int *)(blk + 32);
    const float *d = (const float *)blk;
    fc_q80_sbl r;
    r.w[0] = __ldg(q + lane);
    r.w[1] = __ldg(q + 32 + lane);
    r.s[0] = __ldg(d + (lane >> 3));
    r.s[1] = __ldg(d + 4 + (lane >> 3));
    return r;
}

/* Signed byte K of w to f32: the value of (float)(signed char)(w >> 8K).
 * Spelled as inline PTX bfe.s32 + cvt.rn.f32.s32, the only spelling ptxas 11.4
 * lowers to one I2F.F32.S8 with a byte selector for all four bytes; the C cast
 * costs BFE + PRMT + I2F.S16 (byte 1) or PRMT + I2F (bytes 2-3).
 * FC_MOE_Q80_RED is a gate-only red build: byte 2 converted unsigned, so the
 * parity tests and the A/B memcmp must fail. k is a literal at every call, so
 * the branches fold away (no template: this file is extern "C"). */
#define FC_Q80_BFE_CVT(TY, POS) \
    "{ .reg .b32 t; bfe." TY "32 t, %1, " #POS ", 8; cvt.rn.f32." TY "32 %0, t; }"
#ifdef FC_MOE_Q80_RED
#define FC_Q80_BYTE2_TY "u"
#else
#define FC_Q80_BYTE2_TY "s"
#endif
__device__ __forceinline__ float fc_q80_s8_to_f32(unsigned int w, int k) {
    float f;
    if (k == 0)
        asm(FC_Q80_BFE_CVT("s", 0) : "=f"(f) : "r"(w));
    else if (k == 1)
        asm(FC_Q80_BFE_CVT("s", 8) : "=f"(f) : "r"(w));
    else if (k == 2)
        asm(FC_Q80_BFE_CVT(FC_Q80_BYTE2_TY, 16) : "=f"(f) : "r"(w));
    else
        asm(FC_Q80_BFE_CVT("s", 24) : "=f"(f) : "r"(w));
    return f;
}

/* One row's 4-weight run: a = fmaf chain of the 4 products from 0, then
 * r = fmaf(s, a, r). Each conversion sits next to its fmaf: this is the form the
 * superoptimizer emitted, and ptxas allocates it in fewer registers than four
 * conversions up front (down kernel 66 vs 76). */
__device__ __forceinline__ float fc_q80_sb_run(unsigned int w, float4 x, float s,
                                               float r) {
    float a = __fmaf_rn(fc_q80_s8_to_f32(w, 0), x.x, 0.0f);
    a = __fmaf_rn(fc_q80_s8_to_f32(w, 1), x.y, a);
    a = __fmaf_rn(fc_q80_s8_to_f32(w, 2), x.z, a);
    a = __fmaf_rn(fc_q80_s8_to_f32(w, 3), x.w, a);
    return __fmaf_rn(s, a, r);
}

/* The row-pair dot of two loaded super-blocks against x halves x0 / x1. The two
 * rows are independent chains, so running them one after the other gives the
 * same bits as interleaving them. */
__device__ __forceinline__ void fc_q80_sb_dot2(const fc_q80_sbl &b0,
                                               const fc_q80_sbl &b1,
                                               const float4 (&xv)[2],
                                               float *p0, float *p1) {
    float r0 = *p0, r1 = *p1;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        r0 = fc_q80_sb_run(b0.w[h], xv[h], b0.s[h], r0);
        r1 = fc_q80_sb_run(b1.w[h], xv[h], b1.s[h], r1);
    }
    *p0 = r0;
    *p1 = r1;
}

#define FC_DEF_Q80_GRP2(NAME, XLOAD4)                                         \
__device__ __forceinline__ void NAME(const unsigned char *blk0,               \
                                     const unsigned char *blk1,               \
                                     const float *xb, int lane,               \
                                     float *p0, float *p1) {                  \
    fc_q80_sbl b0 = fc_q80_sb_load(blk0, lane);                               \
    fc_q80_sbl b1 = fc_q80_sb_load(blk1, lane);                               \
    float4 xv[2] = { XLOAD4(xb, lane), XLOAD4(xb + 128, lane) };              \
    fc_q80_sb_dot2(b0, b1, xv, p0, p1);                                       \
}

FC_DEF_Q80_GRP2(fc_q80_sb_lane_grp2,  FC_XLOAD4_LDG)   /* down: global hid */
FC_DEF_Q80_GRP2(fc_q80_sb_lane_grp2s, FC_XLOAD4_PLAIN) /* gateup: shared x */

/* The same lane dot with PLAIN weight loads, for the batched kernels at
 * prefill token counts (fc_moe_gateup_q80_nxp / fc_moe_down_q80_nxp): there
 * the tokens share experts and __ldg made Qwen3.6 prefill 22% slower end to
 * end, while it speeds up the 1-4 token decode calls. The host picks by
 * token count (fc_moe_q80_ldg_max_tok). Same op order -> same bits. */
#define FC_DEF_Q80_GRP2P(NAME, XLOAD4)                                        \
__device__ __forceinline__ void NAME(const unsigned char *blk0,               \
                                     const unsigned char *blk1,               \
                                     const float *xb, int lane,               \
                                     float *p0, float *p1) {                  \
    const unsigned int *q0 = (const unsigned int *)(blk0 + 32);               \
    const unsigned int *q1 = (const unsigned int *)(blk1 + 32);               \
    const float *d0 = (const float *)blk0;                                    \
    const float *d1 = (const float *)blk1;                                    \
    fc_q80_sbl b0, b1;                                                        \
    b0.w[0] = q0[lane];                                                       \
    b0.w[1] = q0[32 + lane];                                                  \
    b1.w[0] = q1[lane];                                                       \
    b1.w[1] = q1[32 + lane];                                                  \
    b0.s[0] = d0[lane >> 3];                                                  \
    b0.s[1] = d0[4 + (lane >> 3)];                                            \
    b1.s[0] = d1[lane >> 3];                                                  \
    b1.s[1] = d1[4 + (lane >> 3)];                                            \
    float4 xv[2] = { XLOAD4(xb, lane), XLOAD4(xb + 128, lane) };              \
    fc_q80_sb_dot2(b0, b1, xv, p0, p1);                                       \
}

FC_DEF_Q80_GRP2P(fc_q80_sb_lane_grp2p,  FC_XLOAD4_LDG)   /* down: global hid */
FC_DEF_Q80_GRP2P(fc_q80_sb_lane_grp2sp, FC_XLOAD4_PLAIN) /* gateup: shared x */


/*
 * Grouped gate+up SwiGLU over the k selected Q3_K experts.
 *
 * gate_all/up_all: [n_expert * ff * row_stride] REPACKED Q3_K bytes — the
 *                  merged 3D GGUF tensor, one contiguous upload per matrix.
 * x:               [cols] fp32 activation (the post_attention_norm'd hidden)
 * top_idx:         [k] int32 selected expert ids (device — no host read-back)
 * hid:             [k * ff] fp32 output, hid[s*ff + row]
 * cols:            inner dim (n_embd, 2048); row_stride = (cols/256)*152
 *
 * WARP-PER-ROW: blockDim = 256 (8 warps); warp w handles expert-FF row
 * blockIdx.x * 8 + w of slot blockIdx.y, computing the gate AND up dots
 * together against the shared-staged x (staged in 2048-element chunks;
 * measured faster here than __ldg x — this kernel re-reads each x element
 * 8x per warp and LDS keeps that off the tex path) — the two independent
 * accumulator chains give this kernel its dual-issue ILP — then lane 0
 * writes silu(g) * u. Inactive tail warps still hit the staging barriers.
 * Launch: grid (ceil(ff/8), k), block 256,
 * shared min(cols, 2048) * sizeof(float).
 */
/*
 * The gate+up work itself, shared by the Q3_K and Q4_K entry points (and by
 * their batched _nx forms, which differ only in XPTR/HIDOFF). A macro rather
 * than an inline function for the same reason FC_MOE_DEDUP_SHX_BODY is one: it
 * expands to the token stream the hand-written Q3_K kernel had, so that
 * kernel's PTX is untouched by the Q4_K addition.
 *
 * Requires in scope: gate_all, up_all, hid, ff, cols, row_stride (parameters);
 * sh_x, warp, lane, row, active, e (block-local prologue).
 *   XPTR    — the token's x base pointer (x, or x + tok*cols for _nx)
 *   HIDOFF  — the element index this warp writes in hid
 *   BLOCK_BYTES / SB_LANE_DOT2S — the quant's repacked stride and shared-x
 *             row-pair lane dot.
 */
#define FC_MOE_GATEUP_BODY(XPTR, HIDOFF, BLOCK_BYTES, SB_LANE_DOT2S)          \
    size_t expert_stride = (size_t)ff * (size_t)row_stride;                   \
    /* strength-reduced running block pointers (one += per super-block) */    \
    const unsigned char *gb = gate_all + (size_t)e * expert_stride +          \
                              (size_t)row * (size_t)row_stride;               \
    const unsigned char *ub = up_all + (size_t)e * expert_stride +            \
                              (size_t)row * (size_t)row_stride;               \
                                                                              \
    float pg = 0.0f, pu = 0.0f;                                               \
    for (int cbase = 0; cbase < cols; cbase += 2048) {                        \
        int cend = cols - cbase < 2048 ? cols - cbase : 2048;                 \
        __syncthreads();                                                      \
        for (int i = threadIdx.x; i < cend; i += 256)                         \
            sh_x[i] = __ldg((XPTR) + cbase + i);                              \
        __syncthreads();                                                      \
        if (active) {                                                         \
            int nsb = cend >> 8;                                              \
            const float *xb = sh_x;                                           \
            for (int sb = 0; sb < nsb; sb++) {                                \
                /* gate/up element-interleaved: two independent chains */     \
                SB_LANE_DOT2S(gb, ub, xb, lane, &pg, &pu);                    \
                gb += BLOCK_BYTES;                                            \
                ub += BLOCK_BYTES;                                            \
                xb += 256;                                                    \
            }                                                                 \
        }                                                                     \
    }                                                                         \
                                                                              \
    pg = fc_warp_sum_grp(pg);                                                 \
    pu = fc_warp_sum_grp(pu);                                                 \
    if (active && lane == 0)                                                  \
        hid[HIDOFF] = (pg / (1.0f + expf(-pg))) * pu;

/* Prologue shared by the two single-token gate+up entry points. */
#define FC_MOE_GATEUP_PROLOGUE                                                \
    int warp = threadIdx.x >> 5;                                              \
    int lane = threadIdx.x & 31;                                              \
    int row = blockIdx.x * 8 + warp;                                          \
    int slot = blockIdx.y;                                                    \
    int active = row < ff;                                                    \
    int e = __ldg(top_idx + slot);

__global__ void fc_moe_gateup_q3k(const unsigned char *gate_all,
                                  const unsigned char *up_all,
                                  const float *x, const int *top_idx,
                                  float *hid, int ff, int cols,
                                  int row_stride) {
    extern __shared__ float sh_x[];        /* [min(cols, 2048)] staged x */
    FC_MOE_GATEUP_PROLOGUE
    FC_MOE_GATEUP_BODY(x, (size_t)slot * ff + row, FC_Q3K_RBLOCK_BYTES,
                       fc_q3k_sb_lane_grp2s)
}

/*
 * Q4_K gate+up. Same kernel, same launch shape, same shared-x staging as
 * fc_moe_gateup_q3k — only the super-block decode differs, and Q4_K's repacked
 * super-block is the same 152 B, so cols/row_stride/grid arithmetic is
 * unchanged. Exists so a pure Q4_K_M MoE (whose ffn_gate_exps/ffn_up_exps are
 * Q4_K, not just the Q3_K_M mini's half-Q4_K ffn_down_exps) has a gate+up
 * kernel at all; without it the host would feed Q4_K bytes to the Q3_K decode.
 * Accumulates in fp32 like every other kernel here — sm_37 has no fp16 math.
 */
__global__ void fc_moe_gateup_q4k(const unsigned char *gate_all,
                                  const unsigned char *up_all,
                                  const float *x, const int *top_idx,
                                  float *hid, int ff, int cols,
                                  int row_stride) {
    extern __shared__ float sh_x[];        /* [min(cols, 2048)] staged x */
    FC_MOE_GATEUP_PROLOGUE
    FC_MOE_GATEUP_BODY(x, (size_t)slot * ff + row, FC_Q4K_RBLOCK_BYTES,
                       fc_q4k_sb_lane_grp2s)
}

/*
 * F16 gate+up. Same kernel, same launch shape, same shared-x staging as its
 * Q3_K/Q4_K siblings — only the super-block stride (512 B, not 152) and the
 * decode differ, and neither is in the grid arithmetic, which depends on
 * ff/k/cols and not on row_stride. Accumulates in fp32 because sm_37 has no
 * fp16 math (see fc_f16_to_f32).
 */
__global__ void fc_moe_gateup_f16(const unsigned char *gate_all,
                                  const unsigned char *up_all,
                                  const float *x, const int *top_idx,
                                  float *hid, int ff, int cols,
                                  int row_stride) {
    extern __shared__ float sh_x[];        /* [min(cols, 2048)] staged x */
    FC_MOE_GATEUP_PROLOGUE
    FC_MOE_GATEUP_BODY(x, (size_t)slot * ff + row, FC_F16_RBLOCK_BYTES,
                       fc_f16_sb_lane_grp2s)
}

/*
 * BF16 gate+up — fc_moe_gateup_f16 with the shift-widen decode. Identical
 * launch shape and identical 512 B super-block; only the lane dot differs.
 */
__global__ void fc_moe_gateup_bf16(const unsigned char *gate_all,
                                   const unsigned char *up_all,
                                   const float *x, const int *top_idx,
                                   float *hid, int ff, int cols,
                                   int row_stride) {
    extern __shared__ float sh_x[];        /* [min(cols, 2048)] staged x */
    FC_MOE_GATEUP_PROLOGUE
    FC_MOE_GATEUP_BODY(x, (size_t)slot * ff + row, FC_BF16_RBLOCK_BYTES,
                       fc_bf16_sb_lane_grp2s)
}

/*
 * The uint2 gate+up pair. Same launch shape and the same staged x — the vec
 * lane dot reads that x back as float4, so the staging array is declared
 * float4 to pin the dynamic-shared base to a 16-B boundary (every offset
 * inside it, 256 floats per super-block and 128 per half, is already a
 * multiple of 16 B). Their hid is NOT bit-identical to the scalar kernels'
 * (different lane->element partition — see FC_DEF_VEC16_GRP2).
 */
__global__ void fc_moe_gateup_f16_vec(const unsigned char *gate_all,
                                      const unsigned char *up_all,
                                      const float *x, const int *top_idx,
                                      float *hid, int ff, int cols,
                                      int row_stride) {
    extern __shared__ float4 sh_x4[];
    float *sh_x = (float *)sh_x4;
    FC_MOE_GATEUP_PROLOGUE
    FC_MOE_GATEUP_BODY(x, (size_t)slot * ff + row, FC_F16_RBLOCK_BYTES,
                       fc_f16_sb_lane_grp2vs)
}

__global__ void fc_moe_gateup_bf16_vec(const unsigned char *gate_all,
                                       const unsigned char *up_all,
                                       const float *x, const int *top_idx,
                                       float *hid, int ff, int cols,
                                       int row_stride) {
    extern __shared__ float4 sh_x4[];
    float *sh_x = (float *)sh_x4;
    FC_MOE_GATEUP_PROLOGUE
    FC_MOE_GATEUP_BODY(x, (size_t)slot * ff + row, FC_BF16_RBLOCK_BYTES,
                       fc_bf16_sb_lane_grp2vs)
}

/*
 * Q8_0 gate+up over the 288-B expert super-blocks (fc_moe_repack_weight).
 * Same launch shape and staged x as its siblings; x is read back as float4,
 * so the staging array is float4-declared like the vec kernels'.
 */
__global__ void fc_moe_gateup_q80(const unsigned char *gate_all,
                                  const unsigned char *up_all,
                                  const float *x, const int *top_idx,
                                  float *hid, int ff, int cols,
                                  int row_stride) {
    extern __shared__ float4 sh_x4[];
    float *sh_x = (float *)sh_x4;
    FC_MOE_GATEUP_PROLOGUE
    FC_MOE_GATEUP_BODY(x, (size_t)slot * ff + row, FC_Q80_SB_BYTES,
                       fc_q80_sb_lane_grp2s)
}

/*
 * Grouped down-projection + weighted accumulate over the k selected experts.
 *
 * down_all: [n_expert * rows * row_stride] REPACKED quant bytes (merged 3D)
 * hid:      [k * ff] fp32 (from fc_moe_gateup_q3k)
 * top_idx:  [k] int32, top_wt: [k] fp32 (device — no host read-back)
 * out:      [rows] fp32, WRITTEN (not accumulated into)
 * ff:       inner dim (expert_ff, 512); rows = n_embd
 *
 * COOPERATIVE MULTI-WARP-PER-ROW-PAIR: the block's warps are grouped wpp
 * (warps-per-row-pair) at a time; group p handles output rows
 * (blockIdx.x * n_warps/wpp + p) * 2 + {0,1} (row-pair: independent
 * accumulators for dual-issue ILP, one hid load amortized across the pair).
 * Within a group, warp wk takes slots wk, wk+wpp, ... of the k selected
 * experts, chaining t = fma(top_wt[s], dot_s, t) per lane over ITS slots
 * (hid read via __ldg — 16 KB total, read-only-cache resident, no staging
 * barriers, so each warp's k/wpp short 2-super-block dots stream back to
 * back with wpp x the loads in flight of the old one-warp-per-pair form).
 * Each warp then tree-reduces its two lanesums (__shfl_down) and the wpp
 * per-warp partials meet in a tiny static-shared cross-warp add, summed in
 * ascending wk order by one thread per row (deterministic; the only fp
 * reassociation vs. the old kernel — bounded by the argmax gate).
 * wpp must divide the block's warp count; k need not divide wpp (idle-slot
 * warps contribute 0).
 * Launch: grid ceil(rows / (2 * n_warps/wpp)), block <= 512 (tuned: 128
 * threads, wpp=2 — the kernel is instruction-issue bound at ~4.3 IPC / 78%
 * issue-slot utilization, so beyond that extra warps buy nothing), shared 0.
 */
#define FC_MOE_DOWN_BODY(BLOCK_BYTES, SB_LANE_DOT2)                            \
    __shared__ float sred[16][2];    /* per-warp row-pair partials, <=16 w */ \
    int warp = threadIdx.x >> 5;                                              \
    int lane = threadIdx.x & 31;                                              \
    int pairs_per_block = (blockDim.x >> 5) / wpp;                            \
    int pair = warp / wpp;                                                    \
    int wk = warp - pair * wpp;                                               \
    int row0 = (blockIdx.x * pairs_per_block + pair) * 2;                     \
    int active0 = row0 < rows;                                                \
    int active1 = row0 + 1 < rows;                                            \
    size_t expert_stride = (size_t)rows * (size_t)row_stride;                 \
    int n_sb = ff >> 8;                                                       \
    float t0 = 0.0f, t1 = 0.0f;                                               \
    if (active0) {                                                            \
        for (int s = wk; s < k; s += wpp) {                                   \
            int e = __ldg(top_idx + s);                                       \
            const unsigned char *b0 = down_all + (size_t)e * expert_stride +  \
                                      (size_t)row0 * (size_t)row_stride;      \
            /* tail: clamp the pair onto row0, duplicate discarded */         \
            const unsigned char *b1 = active1 ? b0 + (size_t)row_stride : b0; \
            const float *hb = hid + (size_t)s * ff;                           \
            float p0 = 0.0f, p1 = 0.0f;                                       \
            for (int sb = 0; sb < n_sb; sb++) {                               \
                SB_LANE_DOT2(b0, b1, hb, lane, &p0, &p1);                     \
                b0 += BLOCK_BYTES;                                            \
                b1 += BLOCK_BYTES;                                            \
                hb += 256;                                                    \
            }                                                                 \
            float w = __ldg(top_wt + s);                                      \
            t0 = __fmaf_rn(w, p0, t0);                                        \
            t1 = __fmaf_rn(w, p1, t1);                                        \
        }                                                                     \
    }                                                                         \
    t0 = fc_warp_sum_grp(t0);                                                 \
    t1 = fc_warp_sum_grp(t1);                                                 \
    if (lane == 0) {                                                          \
        sred[warp][0] = t0;                                                   \
        sred[warp][1] = t1;                                                   \
    }                                                                         \
    __syncthreads();                                                          \
    if ((int)threadIdx.x < 2 * pairs_per_block) {                             \
        int p = threadIdx.x >> 1;                                             \
        int r = threadIdx.x & 1;                                              \
        int row = (blockIdx.x * pairs_per_block + p) * 2 + r;                 \
        if (row < rows) {                                                     \
            float acc = 0.0f;                                                 \
            for (int w = 0; w < wpp; w++)                                     \
                acc += sred[p * wpp + w][r];                                  \
            out[row] = acc;                                                   \
        }                                                                     \
    }

__global__ void fc_moe_down_q3k(const unsigned char *down_all, const float *hid,
                                const int *top_idx, const float *top_wt,
                                float *out, int k, int rows, int ff,
                                int row_stride, int wpp) {
    FC_MOE_DOWN_BODY(FC_Q3K_RBLOCK_BYTES, fc_q3k_sb_lane_grp2)
}

__global__ void fc_moe_down_q4k(const unsigned char *down_all, const float *hid,
                                const int *top_idx, const float *top_wt,
                                float *out, int k, int rows, int ff,
                                int row_stride, int wpp) {
    FC_MOE_DOWN_BODY(FC_Q4K_RBLOCK_BYTES, fc_q4k_sb_lane_grp2)
}

__global__ void fc_moe_down_q6k(const unsigned char *down_all, const float *hid,
                                const int *top_idx, const float *top_wt,
                                float *out, int k, int rows, int ff,
                                int row_stride, int wpp) {
    FC_MOE_DOWN_BODY(FC_Q6K_RBLOCK_BYTES, fc_q6k_sb_lane_grp2)
}

__global__ void fc_moe_down_f16(const unsigned char *down_all, const float *hid,
                                const int *top_idx, const float *top_wt,
                                float *out, int k, int rows, int ff,
                                int row_stride, int wpp) {
    FC_MOE_DOWN_BODY(FC_F16_RBLOCK_BYTES, fc_f16_sb_lane_grp2)
}

__global__ void fc_moe_down_bf16(const unsigned char *down_all,
                                 const float *hid, const int *top_idx,
                                 const float *top_wt, float *out, int k,
                                 int rows, int ff, int row_stride, int wpp) {
    FC_MOE_DOWN_BODY(FC_BF16_RBLOCK_BYTES, fc_bf16_sb_lane_grp2)
}

/* uint2 down pair. hid comes from global here, so the float4 x load is an
 * __ldg — hid is cudaMalloc'd (256-B aligned) and every offset into it
 * (s * ff, 256 per super-block, 128 per half) is a multiple of 16 B. */
__global__ void fc_moe_down_f16_vec(const unsigned char *down_all,
                                    const float *hid, const int *top_idx,
                                    const float *top_wt, float *out, int k,
                                    int rows, int ff, int row_stride,
                                    int wpp) {
    FC_MOE_DOWN_BODY(FC_F16_RBLOCK_BYTES, fc_f16_sb_lane_grp2v)
}

__global__ void fc_moe_down_bf16_vec(const unsigned char *down_all,
                                     const float *hid, const int *top_idx,
                                     const float *top_wt, float *out, int k,
                                     int rows, int ff, int row_stride,
                                     int wpp) {
    FC_MOE_DOWN_BODY(FC_BF16_RBLOCK_BYTES, fc_bf16_sb_lane_grp2v)
}

/*
 * Q8_0 down: FC_MOE_DOWN_BODY with one change in how a warp walks its slots.
 * At ff 512 (2 super-blocks per row) a warp takes its slots two at a time
 * (s, s + wpp) and issues all 8 super-block loads of both experts' row pair
 * before the first FMA; a lone last slot takes the plain loop. Each slot's
 * dot and the t = fma(top_wt[s], dot_s, t) chain run in the same order as
 * the plain loop, so the result is bit-identical. Qwen3.6 expert shape, one
 * K80 die: 79.1 -> 71.3 us on top of the __ldg loads (moeprobe.cu D1 -> D2).
 */
#define FC_MOE_DOWN_Q80_BODY                                                  \
    __shared__ float sred[16][2];    /* per-warp row-pair partials, <=16 w */ \
    int warp = threadIdx.x >> 5;                                              \
    int lane = threadIdx.x & 31;                                              \
    int pairs_per_block = (blockDim.x >> 5) / wpp;                            \
    int pair = warp / wpp;                                                    \
    int wk = warp - pair * wpp;                                               \
    int row0 = (blockIdx.x * pairs_per_block + pair) * 2;                     \
    int active0 = row0 < rows;                                                \
    int active1 = row0 + 1 < rows;                                            \
    size_t expert_stride = (size_t)rows * (size_t)row_stride;                 \
    size_t row_off = (size_t)row0 * (size_t)row_stride;                       \
    size_t pair_off = active1 ? (size_t)row_stride : 0;                       \
    int n_sb = ff >> 8;                                                       \
    float t0 = 0.0f, t1 = 0.0f;                                               \
    if (active0) {                                                            \
        int s = wk;                                                           \
        if (n_sb == 2) {                                                      \
            for (; s + wpp < k; s += 2 * wpp) {                               \
                const unsigned char *a0 = down_all + row_off +                \
                    (size_t)__ldg(top_idx + s) * expert_stride;               \
                const unsigned char *c0 = down_all + row_off +                \
                    (size_t)__ldg(top_idx + s + wpp) * expert_stride;         \
                fc_q80_sbl la[2][2], lc[2][2];                                \
_Pragma("unroll")                                                             \
                for (int j = 0; j < 2; j++) {                                 \
                    la[j][0] = fc_q80_sb_load(a0 + FC_Q80_SB_BYTES * j, lane);\
                    la[j][1] = fc_q80_sb_load(a0 + pair_off +                 \
                                              FC_Q80_SB_BYTES * j, lane);     \
                    lc[j][0] = fc_q80_sb_load(c0 + FC_Q80_SB_BYTES * j, lane);\
                    lc[j][1] = fc_q80_sb_load(c0 + pair_off +                 \
                                              FC_Q80_SB_BYTES * j, lane);     \
                }                                                             \
                const float *ha = hid + (size_t)s * ff;                       \
                const float *hc = hid + (size_t)(s + wpp) * ff;               \
                float p0 = 0.0f, p1 = 0.0f;                                   \
_Pragma("unroll")                                                             \
                for (int j = 0; j < 2; j++) {                                 \
                    float4 xv[2] = { FC_XLOAD4_LDG(ha + 256 * j, lane),       \
                                     FC_XLOAD4_LDG(ha + 256 * j + 128, lane) };\
                    fc_q80_sb_dot2(la[j][0], la[j][1], xv, &p0, &p1);         \
                }                                                             \
                float w = __ldg(top_wt + s);                                  \
                t0 = __fmaf_rn(w, p0, t0);                                    \
                t1 = __fmaf_rn(w, p1, t1);                                    \
                p0 = 0.0f;                                                    \
                p1 = 0.0f;                                                    \
_Pragma("unroll")                                                             \
                for (int j = 0; j < 2; j++) {                                 \
                    float4 xv[2] = { FC_XLOAD4_LDG(hc + 256 * j, lane),       \
                                     FC_XLOAD4_LDG(hc + 256 * j + 128, lane) };\
                    fc_q80_sb_dot2(lc[j][0], lc[j][1], xv, &p0, &p1);         \
                }                                                             \
                w = __ldg(top_wt + s + wpp);                                  \
                t0 = __fmaf_rn(w, p0, t0);                                    \
                t1 = __fmaf_rn(w, p1, t1);                                    \
            }                                                                 \
        }                                                                     \
        for (; s < k; s += wpp) {                                             \
            const unsigned char *b0 = down_all + row_off +                    \
                (size_t)__ldg(top_idx + s) * expert_stride;                   \
            const unsigned char *b1 = b0 + pair_off;                          \
            const float *hb = hid + (size_t)s * ff;                           \
            float p0 = 0.0f, p1 = 0.0f;                                       \
            for (int sb = 0; sb < n_sb; sb++) {                               \
                fc_q80_sb_lane_grp2(b0, b1, hb, lane, &p0, &p1);              \
                b0 += FC_Q80_SB_BYTES;                                        \
                b1 += FC_Q80_SB_BYTES;                                        \
                hb += 256;                                                    \
            }                                                                 \
            float w = __ldg(top_wt + s);                                      \
            t0 = __fmaf_rn(w, p0, t0);                                        \
            t1 = __fmaf_rn(w, p1, t1);                                        \
        }                                                                     \
    }                                                                         \
    t0 = fc_warp_sum_grp(t0);                                                 \
    t1 = fc_warp_sum_grp(t1);                                                 \
    if (lane == 0) {                                                          \
        sred[warp][0] = t0;                                                   \
        sred[warp][1] = t1;                                                   \
    }                                                                         \
    __syncthreads();                                                          \
    if ((int)threadIdx.x < 2 * pairs_per_block) {                             \
        int p = threadIdx.x >> 1;                                             \
        int r = threadIdx.x & 1;                                              \
        int row = (blockIdx.x * pairs_per_block + p) * 2 + r;                 \
        if (row < rows) {                                                     \
            float acc = 0.0f;                                                 \
            for (int w = 0; w < wpp; w++)                                     \
                acc += sred[p * wpp + w][r];                                  \
            out[row] = acc;                                                   \
        }                                                                     \
    }

/* Q8_0 down over the 288-B expert super-blocks; hid read as float4 via __ldg
 * (every offset into it is a multiple of 16 B, as for the vec pair). */
__global__ void fc_moe_down_q80(const unsigned char *down_all,
                                const float *hid, const int *top_idx,
                                const float *top_wt, float *out, int k,
                                int rows, int ff, int row_stride, int wpp) {
    FC_MOE_DOWN_Q80_BODY
}

/*
 * Batched (prefill) grouped gate+up SwiGLU: grid.z = token t, everything
 * else exactly fc_moe_gateup_q3k on token t's slices — x [n][cols],
 * top_idx [n][k], hid [n][k*ff]. Per token bit-exact (same warp body).
 * Weight reuse across tokens comes from L2 when tokens share experts.
 */
#define FC_MOE_GATEUP_NX_PROLOGUE                                             \
    int warp = threadIdx.x >> 5;                                              \
    int lane = threadIdx.x & 31;                                              \
    int row = blockIdx.x * 8 + warp;                                          \
    int slot = blockIdx.y;                                                    \
    int tok = blockIdx.z;                                                     \
    int active = row < ff;                                                    \
    const float *xt = x + (size_t)tok * cols;                                 \
    int e = __ldg(top_idx + (size_t)tok * k + slot);

__global__ void fc_moe_gateup_q3k_nx(const unsigned char *gate_all,
                                     const unsigned char *up_all,
                                     const float *x, const int *top_idx,
                                     float *hid, int k, int ff, int cols,
                                     int row_stride) {
    extern __shared__ float sh_x[];
    FC_MOE_GATEUP_NX_PROLOGUE
    FC_MOE_GATEUP_BODY(xt, ((size_t)tok * k + slot) * ff + row,
                       FC_Q3K_RBLOCK_BYTES, fc_q3k_sb_lane_grp2s)
}

/* Batched (prefill / multi-user decode) Q4_K gate+up — fc_moe_gateup_q4k with
 * grid.z = token, exactly as _q3k_nx is to _q3k. Per token bit-exact with the
 * single-token Q4_K kernel (same warp body). */
__global__ void fc_moe_gateup_q4k_nx(const unsigned char *gate_all,
                                     const unsigned char *up_all,
                                     const float *x, const int *top_idx,
                                     float *hid, int k, int ff, int cols,
                                     int row_stride) {
    extern __shared__ float sh_x[];
    FC_MOE_GATEUP_NX_PROLOGUE
    FC_MOE_GATEUP_BODY(xt, ((size_t)tok * k + slot) * ff + row,
                       FC_Q4K_RBLOCK_BYTES, fc_q4k_sb_lane_grp2s)
}

/* Batched (prefill / multi-user decode) F16 gate+up — fc_moe_gateup_f16 with
 * grid.z = token, exactly as _q3k_nx is to _q3k. Per token bit-exact with the
 * single-token F16 kernel (same warp body). */
__global__ void fc_moe_gateup_f16_nx(const unsigned char *gate_all,
                                     const unsigned char *up_all,
                                     const float *x, const int *top_idx,
                                     float *hid, int k, int ff, int cols,
                                     int row_stride) {
    extern __shared__ float sh_x[];
    FC_MOE_GATEUP_NX_PROLOGUE
    FC_MOE_GATEUP_BODY(xt, ((size_t)tok * k + slot) * ff + row,
                       FC_F16_RBLOCK_BYTES, fc_f16_sb_lane_grp2s)
}

/* Batched BF16 gate+up — fc_moe_gateup_bf16 with grid.z = token. */
__global__ void fc_moe_gateup_bf16_nx(const unsigned char *gate_all,
                                      const unsigned char *up_all,
                                      const float *x, const int *top_idx,
                                      float *hid, int k, int ff, int cols,
                                      int row_stride) {
    extern __shared__ float sh_x[];
    FC_MOE_GATEUP_NX_PROLOGUE
    FC_MOE_GATEUP_BODY(xt, ((size_t)tok * k + slot) * ff + row,
                       FC_BF16_RBLOCK_BYTES, fc_bf16_sb_lane_grp2s)
}

/* Batched uint2 gate+up. These exist so the vec switch is coherent across the
 * whole MoE block: with only the single-token pair vectorized, a batched run
 * would silently fall back to the scalar lane map and be compared against the
 * wrong replica. Per token bit-exact with fc_moe_gateup_{f16,bf16}_vec. */
__global__ void fc_moe_gateup_f16_vec_nx(const unsigned char *gate_all,
                                         const unsigned char *up_all,
                                         const float *x, const int *top_idx,
                                         float *hid, int k, int ff, int cols,
                                         int row_stride) {
    extern __shared__ float4 sh_x4[];
    float *sh_x = (float *)sh_x4;
    FC_MOE_GATEUP_NX_PROLOGUE
    FC_MOE_GATEUP_BODY(xt, ((size_t)tok * k + slot) * ff + row,
                       FC_F16_RBLOCK_BYTES, fc_f16_sb_lane_grp2vs)
}

__global__ void fc_moe_gateup_bf16_vec_nx(const unsigned char *gate_all,
                                          const unsigned char *up_all,
                                          const float *x, const int *top_idx,
                                          float *hid, int k, int ff, int cols,
                                          int row_stride) {
    extern __shared__ float4 sh_x4[];
    float *sh_x = (float *)sh_x4;
    FC_MOE_GATEUP_NX_PROLOGUE
    FC_MOE_GATEUP_BODY(xt, ((size_t)tok * k + slot) * ff + row,
                       FC_BF16_RBLOCK_BYTES, fc_bf16_sb_lane_grp2vs)
}

/* Batched Q8_0 gate+up — fc_moe_gateup_q80 with grid.z = token. Per token
 * bit-exact with the single-token kernel (same warp body). */
__global__ void fc_moe_gateup_q80_nx(const unsigned char *gate_all,
                                     const unsigned char *up_all,
                                     const float *x, const int *top_idx,
                                     float *hid, int k, int ff, int cols,
                                     int row_stride) {
    extern __shared__ float4 sh_x4[];
    float *sh_x = (float *)sh_x4;
    FC_MOE_GATEUP_NX_PROLOGUE
    FC_MOE_GATEUP_BODY(xt, ((size_t)tok * k + slot) * ff + row,
                       FC_Q80_SB_BYTES, fc_q80_sb_lane_grp2s)
}

/* fc_moe_gateup_q80_nx with plain weight loads (prefill token counts). */
__global__ void fc_moe_gateup_q80_nxp(const unsigned char *gate_all,
                                     const unsigned char *up_all,
                                     const float *x, const int *top_idx,
                                     float *hid, int k, int ff, int cols,
                                     int row_stride) {
    extern __shared__ float4 sh_x4[];
    float *sh_x = (float *)sh_x4;
    FC_MOE_GATEUP_NX_PROLOGUE
    FC_MOE_GATEUP_BODY(xt, ((size_t)tok * k + slot) * ff + row,
                       FC_Q80_SB_BYTES, fc_q80_sb_lane_grp2sp)
}

/*
 * Batched (prefill) grouped down + weighted accumulate: grid.y = token t,
 * the FC_MOE_DOWN_BODY otherwise unchanged on token t's slices —
 * hid [n][k*ff], top_idx/top_wt [n][k], out [n][rows]. Per token bit-exact.
 */
#define FC_MOE_DOWN_NX_PROLOGUE                                               \
    int tok = blockIdx.y;                                                     \
    hid += (size_t)tok * k * ff;                                              \
    top_idx += (size_t)tok * k;                                               \
    top_wt += (size_t)tok * k;                                                \
    out += (size_t)tok * rows;

__global__ void fc_moe_down_q3k_nx(const unsigned char *down_all,
                                   const float *hid, const int *top_idx,
                                   const float *top_wt, float *out, int k,
                                   int rows, int ff, int row_stride, int wpp) {
    FC_MOE_DOWN_NX_PROLOGUE
    FC_MOE_DOWN_BODY(FC_Q3K_RBLOCK_BYTES, fc_q3k_sb_lane_grp2)
}

__global__ void fc_moe_down_q4k_nx(const unsigned char *down_all,
                                   const float *hid, const int *top_idx,
                                   const float *top_wt, float *out, int k,
                                   int rows, int ff, int row_stride, int wpp) {
    FC_MOE_DOWN_NX_PROLOGUE
    FC_MOE_DOWN_BODY(FC_Q4K_RBLOCK_BYTES, fc_q4k_sb_lane_grp2)
}

__global__ void fc_moe_down_q6k_nx(const unsigned char *down_all,
                                   const float *hid, const int *top_idx,
                                   const float *top_wt, float *out, int k,
                                   int rows, int ff, int row_stride, int wpp) {
    FC_MOE_DOWN_NX_PROLOGUE
    FC_MOE_DOWN_BODY(FC_Q6K_RBLOCK_BYTES, fc_q6k_sb_lane_grp2)
}

__global__ void fc_moe_down_f16_nx(const unsigned char *down_all,
                                   const float *hid, const int *top_idx,
                                   const float *top_wt, float *out, int k,
                                   int rows, int ff, int row_stride, int wpp) {
    FC_MOE_DOWN_NX_PROLOGUE
    FC_MOE_DOWN_BODY(FC_F16_RBLOCK_BYTES, fc_f16_sb_lane_grp2)
}

__global__ void fc_moe_down_bf16_nx(const unsigned char *down_all,
                                    const float *hid, const int *top_idx,
                                    const float *top_wt, float *out, int k,
                                    int rows, int ff, int row_stride,
                                    int wpp) {
    FC_MOE_DOWN_NX_PROLOGUE
    FC_MOE_DOWN_BODY(FC_BF16_RBLOCK_BYTES, fc_bf16_sb_lane_grp2)
}

__global__ void fc_moe_down_f16_vec_nx(const unsigned char *down_all,
                                       const float *hid, const int *top_idx,
                                       const float *top_wt, float *out, int k,
                                       int rows, int ff, int row_stride,
                                       int wpp) {
    FC_MOE_DOWN_NX_PROLOGUE
    FC_MOE_DOWN_BODY(FC_F16_RBLOCK_BYTES, fc_f16_sb_lane_grp2v)
}

__global__ void fc_moe_down_bf16_vec_nx(const unsigned char *down_all,
                                        const float *hid, const int *top_idx,
                                        const float *top_wt, float *out, int k,
                                        int rows, int ff, int row_stride,
                                        int wpp) {
    FC_MOE_DOWN_NX_PROLOGUE
    FC_MOE_DOWN_BODY(FC_BF16_RBLOCK_BYTES, fc_bf16_sb_lane_grp2v)
}

__global__ void fc_moe_down_q80_nx(const unsigned char *down_all,
                                   const float *hid, const int *top_idx,
                                   const float *top_wt, float *out, int k,
                                   int rows, int ff, int row_stride, int wpp) {
    FC_MOE_DOWN_NX_PROLOGUE
    FC_MOE_DOWN_Q80_BODY
}

/* The batched Q8_0 down for prefill token counts: plain weight loads, one
 * super-block at a time (FC_MOE_DOWN_BODY), as before the two-slot walk. */
__global__ void fc_moe_down_q80_nxp(const unsigned char *down_all,
                                    const float *hid, const int *top_idx,
                                    const float *top_wt, float *out, int k,
                                    int rows, int ff, int row_stride, int wpp) {
    FC_MOE_DOWN_NX_PROLOGUE
    FC_MOE_DOWN_BODY(FC_Q80_SB_BYTES, fc_q80_sb_lane_grp2p)
}

/*
 * Fused SwiGLU elementwise: out[i] = silu(a[i]) * b[i]. In-place safe
 * (out may alias a or b). Replaces the fc_silu + fc_mul pair for the
 * shared expert. silu matches fc_silu exactly: v / (1 + expf(-v)).
 *
 * Launch: grid ceil(n / blockDim.x), block 256 threads, no shared memory.
 */
__global__ void fc_moe_silu_mul(const float *a, const float *b, float *out,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        out[i] = (v / (1.0f + expf(-v))) * b[i];
    }
}

/*
 * Shared-expert scalar gate, fully on-device:
 *   gate_logit = gate_w . x   (tree-reduced dot over n)
 *   shexp[i] *= sigmoid(gate_logit)
 * Replaces the per-layer host sync + D2H of x + host dot + fc_moe_shared_gate
 * launch. gate_w is the F32 [n_embd] ffn_gate_inp_shexp vector.
 *
 * Launch: 1 block, 256 threads, shared 256 * sizeof(float).
 */
/* Batched (prefill) shared-expert scalar gate: one token per BLOCK
 * (blockIdx.x = t) on x/shexp [n_tok][n]. Same blockDim as the single
 * launch -> same reduction tree -> bit-exact per token. */
__global__ void fc_moe_shexp_dot_gate_nx(const float *gate_w, const float *x,
                                         float *shexp, int n) {
    extern __shared__ float sdata[];
    int tok = blockIdx.x;
    const float *xt = x + (size_t)tok * n;
    float *st = shexp + (size_t)tok * n;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    float p = 0.0f;
    for (int i = tid; i < n; i += threads)
        p += gate_w[i] * xt[i];
    sdata[tid] = p;
    __syncthreads();
    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sg = 1.0f / (1.0f + expf(-sdata[0]));
    for (int i = tid; i < n; i += threads)
        st[i] *= sg;
}

/* Shared-expert scalar gate + combine + residual in ONE launch, per token:
 *   shexp *= sigmoid(gate_w . x);  block_out = moe_out + shexp;
 *   hidden += block_out
 * Replaces fc_moe_shexp_dot_gate_nx + fc_add + fc_add (three launches per MoE
 * layer, ~9 us each under graph replay on the K80). The dot is that kernel's
 * body verbatim (same blockDim -> same tree); the element ops are pinned with
 * __fmul_rn/__fadd_rn so moe_out + shexp*sg can never contract into an FMA
 * the three separate kernels could not form. BIT-EXACT.
 * Launch: grid ntok, 256 threads, shared 256 * sizeof(float). */
__global__ void fc_moe_shexp_gate_combine_nx(const float *gate_w,
                                             const float *x, float *shexp,
                                             const float *moe_out,
                                             float *block_out, float *hidden,
                                             int n) {
    extern __shared__ float sdata[];
    int tok = blockIdx.x;
    const float *xt = x + (size_t)tok * n;
    float *st = shexp + (size_t)tok * n;
    const float *mo = moe_out + (size_t)tok * n;
    float *bo = block_out + (size_t)tok * n;
    float *hd = hidden + (size_t)tok * n;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    float p = 0.0f;
    for (int i = tid; i < n; i += threads)
        p += gate_w[i] * xt[i];
    sdata[tid] = p;
    __syncthreads();
    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sg = 1.0f / (1.0f + expf(-sdata[0]));
    for (int i = tid; i < n; i += threads) {
        const float sv = __fmul_rn(st[i], sg);
        const float bv = __fadd_rn(mo[i], sv);
        st[i] = sv;
        bo[i] = bv;
        hd[i] = __fadd_rn(hd[i], bv);
    }
}

__global__ void fc_moe_shexp_dot_gate(const float *gate_w, const float *x,
                                      float *shexp, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int threads = blockDim.x;

    float p = 0.0f;
    for (int i = tid; i < n; i += threads)
        p += gate_w[i] * x[i];
    sdata[tid] = p;
    __syncthreads();
    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float sgate = 1.0f / (1.0f + expf(-sdata[0]));
    for (int i = tid; i < n; i += threads)
        shexp[i] *= sgate;
}

/* ============================================================
 * MoE expert-union dedup for the batched gate+up
 * ============================================================
 *
 * Q3_K ONLY. Every kernel in this section decodes Q3_K (fc_q3k_decode_sb) and
 * there is deliberately no Q4_K twin: dedup is default-off (QWEN35_MOE_DEDUP=1)
 * and prefill-only, so it is an opt-in speed lever rather than a format the
 * engine must support. The host gates on the gate/up tensor being Q3_K and
 * falls back to fc_moe_gateup_q4k_nx otherwise (qwen35moe_forward_cuda.c,
 * gpu_moe_batch_use_dedup) — it must never reach these with Q4_K bytes.
 *
 * A batched step of n positions selects k experts each — n*k routed-expert
 * applications. The plain _nx gateup kernel reads (and Q3_K-decodes) every
 * (token, slot) expert row separately, so an expert shared by several
 * positions is re-read and re-decoded once per selecting position.
 *
 * Dedup collapses that to ONE weight read+decode per UNIQUE expert per step:
 * the union of routed experts across the batch is materialised as per-expert
 * request lists, then the grouped kernel streams each unique expert's rows
 * once and applies them to every token that routed to it (grouped-by-expert
 * GEMM). A token selects a given expert at most once (its k slots are
 * distinct), so an expert's request count R is bounded by ntok.
 *
 * The saving is n*k / U, where U = |union of routed experts|. That ratio is
 * what decides whether this pays, and it is set entirely by ntok:
 *
 *   - PREFILL chunk, ntok=128, k=8, n_expert=256: 1024 applications over
 *     U ~ 251 experts -> ~4x fewer decodes. This is the intended caller.
 *   - Multi-user DECODE, ntok=8: 64 applications over U ~ 57 -> ~1.1x, which
 *     does not cover the wider grid.y (n_expert blocks instead of k*ntok).
 *     Measured as a 7-9% regression; the Pantheon caller keeps decode on _nx.
 *
 * Bit-exact with fc_moe_gateup_q3k_nx: the per-request hid value reuses the
 * identical per-element FMA order of fc_q3k_sb_lane_grp2s — the Q3_K field
 * products fs*fq are just hoisted out of the request loop (independent of x,
 * so the value is unchanged), and the per-super-block d-scaled accumulate is
 * reproduced verbatim per request. x is read from GLOBAL via __ldg here (vs
 * the nx kernel's shared staging); the loaded float value — and thus every
 * op — is identical, so hid is bit-for-bit the nx result.
 */

/* Request-tile depth: per-request accumulators held in registers. grid.z
 * tiles an expert's request list by this and weights re-stream once per tile,
 * so an expert with R requests is read ceil(R/RT) times.
 *
 * grid.z is sized off maxr (the SAFE bound on R, == ntok), not off the actual
 * max R, because grid dims are host-side and the counts are device-side. At
 * ntok=128 that is 16 z-tiles while the mean R is ~4, so ~15/16 of the blocks
 * find r0 >= R and exit on their first load. That waste is the price of not
 * syncing the counts back to the host; it is bounded by a global load plus a
 * branch per block. Keep in sync with the host wrapper (fc_cuda.h
 * FC_MOE_DEDUP_RT). */
#define FC_MOE_DEDUP_RT 8

/* The x-staging loop of the _shx body, in four forms; FC_MOE_SHX_STAGE selects
 * one and defaults to the shipped FLAT form. Included here rather than at the
 * top of the file because the forms are written in terms of FC_MOE_DEDUP_RT. */
#include "moe_dedup_shx_staging.cuh"

/*
 * COMPACTED-GRID ARM, compile-time, DEFAULT OFF. At 1 it adds two kernels at the
 * end of this file — fc_moe_dedup_build_wl and fc_moe_gateup_q3k_dedup_shxc —
 * that replace the (n_expert x ceil(maxr/RT)) grid above with one axis over a
 * device-built (expert, request-tile) work list. See the block comment there for
 * the defect, the fix and the bit-exactness argument.
 *
 * Compile-time because the kernels ship pre-compiled in cuda/fc_cuda_ptx.h.
 *
 * ★ SHIPPED DEFAULT since 2026-08-11: 1, so the two kernels are IN the
 * regenerated header. This is the compile-time half of a DUAL gate — the header
 * only makes them available, and cuda/fc_cuda.c
 * fc_moe_dedup_compact_enabled_ctx() still requires FC_MOE_DEDUP_COMPACT=1 in
 * the environment to select them, which is where the safe default now lives. A
 * plain build therefore behaves exactly as before unless the env asks for it.
 * Gated at 1.195x in-model (recovered by Amdahl), bit-exact:
 * results/prefill_moe_shx_staging_gate.md.
 *
 * It became a source default because the alternative was worse. Keeping this at
 * 0 meant the DEPLOYED header could not be produced from a clean checkout —
 * the deployed artifact needed
 *   make NVCCFLAGS_EXTRA="-DFC_MOE_DEDUP_COMPACT=1 -DFC_MOE_SHX_STAGE=2"
 * a recipe that lived only in prose, so a plain `make` silently regressed the deployed build
 * by two levers at once and neither failure was loud: this gate's runtime lookup
 * is NULL-tolerant, and FC_MOE_SHX_STAGE only changes a kernel BODY. Both
 * defaults now live in the source and `make` alone reproduces the shipped
 * module. Set -DFC_MOE_DEDUP_COMPACT=0 for a header without the two kernels.
 *
 * Verify MODULE-SCOPED, never header-wide: extract FC_PTX_MOE_EXPERT_GROUP and
 * md5 it. Shipped (compact+VEC4) = cb2e87d100eed7f49fe5eaeb5586fa9a, 14 .entry.
 */
#ifndef FC_MOE_DEDUP_COMPACT
#define FC_MOE_DEDUP_COMPACT 1
#endif

/*
 * Build per-expert request lists from the batch's top_idx.
 *   top_idx: [ntok*k] int32, flat request f = tok*k + slot -> expert id
 *   ecount:  [n_expert] int32 (output) — tokens routing to each expert
 *   ereq:    [n_expert*maxr] int32 (output) — ereq[e*maxr+j] = j-th request f
 * maxr caps the per-expert list (== ntok). Request order within an expert is
 * atomicAdd race order — order-free for the gateup consumer (each request
 * writes its own hid slot). Single block; ntok*k and n_expert are both tiny.
 */
__global__ void fc_moe_dedup_build(const int *top_idx, int *ecount, int *ereq,
                                   int n_expert, int k, int ntok, int maxr) {
    int tid = threadIdx.x;
    int threads = blockDim.x;
    for (int e = tid; e < n_expert; e += threads)
        ecount[e] = 0;
    __syncthreads();
    int total = ntok * k;
    for (int f = tid; f < total; f += threads) {
        int e = top_idx[f];
        if (e < 0 || e >= n_expert)
            continue;
        int slot = atomicAdd(&ecount[e], 1);
        if (slot < maxr)
            ereq[(size_t)e * maxr + slot] = f;
    }
}

/* Decode one Q3_K super-block's 8 field products fs*fq (per lane), returning
 * the block scale d. Reproduces the exact per-i products of the grp2s lane
 * dot; hoisting them out of the request loop is x-independent so the values
 * (and the FMA that consumes them) are unchanged. */
__device__ __forceinline__ float fc_q3k_decode_sb(const unsigned char *blk,
                                                  int lane, float *wprod) {
    unsigned int wv = ((const unsigned int *)blk)[lane];
    float d = *(const float *)(blk + 144);
    uint2 sw = ((const uint2 *)(blk + 128))[lane >> 4];
    unsigned int wh = wv >> 16;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int qp = (i <= 4) ? 4 * i : 4 * i - 16;
        unsigned int q = (i <= 4) ? wv : wh;
        unsigned int s = (i < 4) ? sw.x : sw.y;
        float fq = FC_MAGIC_NIB(q, qp, 4);
        float fs = __fsub_rn(FC_CVT_BYTE(s, i & 3), 32.0f);
        wprod[i] = __fmul_rn(fs, fq);
    }
    return d;
}

/*
 * Grouped gate+up SwiGLU over the UNION of routed experts (dedup path).
 *
 * gate_all/up_all: [n_expert*ff*row_stride] repacked Q3_K (merged 3D)
 * x:               [ntok*cols] fp32 activations (per-token x_norm)
 * ecount/ereq:     per-expert request lists from fc_moe_dedup_build
 * hid:             [ntok*k*ff] fp32 output, hid[f*ff + row] (f = tok*k+slot)
 * cols:            n_embd (2048, multiple of 256); row_stride = cols/256*152
 *
 * Grid: (ceil(ff/8), n_expert, ceil(maxr/RT)); block 256 (8 warps), shared 0.
 * blockIdx.y = expert e; warp w handles FF row blockIdx.x*8+w. Experts with
 * no requests (ecount==0) exit immediately, so the full n_expert grid.y just
 * skips the ~n_expert-U empty experts cheaply. For each super-block the two
 * expert rows are decoded ONCE (fc_q3k_decode_sb) then applied to every
 * request in the tile — one weight stream, many tokens.
 */
__global__ void fc_moe_gateup_q3k_dedup(const unsigned char *gate_all,
                                        const unsigned char *up_all,
                                        const float *x, const int *ecount,
                                        const int *ereq, float *hid, int k,
                                        int ff, int cols, int row_stride,
                                        int maxr) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int e = blockIdx.y;
    int R = ecount[e];
    int r0 = blockIdx.z * FC_MOE_DEDUP_RT;
    if (r0 >= R)
        return;                       /* no requests for e in this tile */
    int rc = R - r0;
    if (rc > FC_MOE_DEDUP_RT)
        rc = FC_MOE_DEDUP_RT;         /* uniform across the block */
    int active = row < ff;

    int fj[FC_MOE_DEDUP_RT];
    int tokj[FC_MOE_DEDUP_RT];
    float pg[FC_MOE_DEDUP_RT], pu[FC_MOE_DEDUP_RT];
#pragma unroll
    for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {
        int f = (j < rc) ? ereq[(size_t)e * maxr + r0 + j] : -1;
        fj[j] = f;
        tokj[j] = (f >= 0) ? f / k : 0;
        pg[j] = 0.0f;
        pu[j] = 0.0f;
    }

    size_t expert_stride = (size_t)ff * (size_t)row_stride;
    const unsigned char *gb = gate_all + (size_t)e * expert_stride +
                              (size_t)row * (size_t)row_stride;
    const unsigned char *ub = up_all + (size_t)e * expert_stride +
                              (size_t)row * (size_t)row_stride;

    if (active) {
        int nsb = cols >> 8;
        for (int sb = 0; sb < nsb; sb++) {
            float wg[8], wu[8];
            float dg = fc_q3k_decode_sb(gb, lane, wg);
            float du = fc_q3k_decode_sb(ub, lane, wu);
#pragma unroll
            for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {
                if (j < rc) {
                    const float *xb =
                        x + (size_t)tokj[j] * cols + (size_t)sb * 256;
                    float ag = 0.0f, au = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        float xv = __ldg(xb + 32 * i + lane);
                        ag = __fmaf_rn(wg[i], xv, ag);
                        au = __fmaf_rn(wu[i], xv, au);
                    }
                    pg[j] += dg * ag;
                    pu[j] += du * au;
                }
            }
            gb += FC_Q3K_RBLOCK_BYTES;
            ub += FC_Q3K_RBLOCK_BYTES;
        }
    }

#pragma unroll
    for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {
        float g = fc_warp_sum_grp(pg[j]);
        float u = fc_warp_sum_grp(pu[j]);
        if (active && lane == 0 && j < rc)
            hid[(size_t)fj[j] * ff + row] = (g / (1.0f + expf(-g))) * u;
    }
}

/*
 * fc_moe_gateup_q3k_dedup_shx — the dedup gate+up with x staged in SHARED,
 * the way fc_moe_gateup_q3k_nx already does it. Bit-exact with both.
 *
 * Two measured problems with the kernel above, fixed together because one
 * change addresses both:
 *
 *  - x traffic. All 8 warps of a block need the SAME rc rows of x; they
 *    differ only in which FF row they own. Reading x with __ldg from global
 *    therefore issues the loads 8x over, and each one carries a TEXDEPBAR.
 *    Counted in SASS, the per-request apply costs 51.5 instructions against
 *    _nx's 26 — the gap is entirely this. Staging one super-block slice per
 *    request (RT*256 floats = 8 KB) and reading LDS collapses it.
 *  - registers. Holding fj/tokj per thread costs 16 registers for values that
 *    are BLOCK-UNIFORM (they depend only on e, r0 and j). Moving them to 64
 *    bytes of shared drops the kernel from 95 registers to 62, which is
 *    8 blocks/SMX instead of 5 — full occupancy.
 *
 * Measured on a K80 at ntok=128, k=8, n_expert=256: +20% on the kernel
 * against fc_moe_gateup_q3k_dedup, bit-exact. Off by default; opt in with
 * FC_MOE_DEDUP_SHX=1.
 *
 * Same grid, same args, same semantics as fc_moe_gateup_q3k_dedup; it just
 * needs 8 KB + 64 B of static shared.
 *
 * The BODY is a macro so the compacted-grid entry point below can reuse it
 * rather than carry a second copy. A macro and not an inline function on
 * purpose: it expands to the same token stream, so the regenerated PTX for THIS
 * kernel stays byte-identical to the checked-in cuda/fc_cuda_ptx.h — that
 * identity is the gate on the refactor. An inline function taking the three
 * __shared__ arrays as pointers would risk generic-address lowering and a
 * different register allocation.
 */

/*
 * The dedup gate+up work for ONE (expert, request-tile) work item.
 *
 * Expands inside a kernel with fc_moe_gateup_q3k_dedup_shx's signature, and
 * requires these names in scope: gate_all, up_all, x, ereq, hid, k, ff, cols,
 * row_stride, maxr (parameters); warp, lane, row, active (block-local); and the
 * three shared arrays sh_x, sh_fj, sh_tok. E/R0/RC are the work item — expert,
 * first request index, request count in the tile (1..FC_MOE_DEDUP_RT).
 *
 * RC gates only `j < rc`; it never appears in an arithmetic expression, and the
 * per-request accumulation over super-blocks is independent of the other
 * requests in the tile. So WHICH tile a request lands in cannot change its
 * value — which is what makes the compacted grid below bit-exact.
 *
 * PRE and STAGE are the x-staging form (moe_dedup_shx_staging.cuh): PRE() emits
 * whatever must be hoisted out of the super-block loop, STAGE(RC, sb) the
 * staging body itself. FC_MOE_DEDUP_SHX_BODY passes the pair FC_MOE_SHX_STAGE
 * selects, so the shipped kernels track the flag; the microbenchmark expands
 * _BODY_S directly, once per form, to get every arm from ONE source in ONE
 * translation unit.
 */
#define FC_MOE_DEDUP_SHX_BODY_S(PRE, STAGE, E, R0, RC)                         \
    do {                                                                       \
        if ((int)threadIdx.x < FC_MOE_DEDUP_RT) {                              \
            int j = threadIdx.x;                                               \
            int f = (j < (RC)) ? ereq[(size_t)(E) * maxr + (R0) + j] : -1;     \
            sh_fj[j] = f;                                                      \
            sh_tok[j] = (f >= 0) ? f / k : 0;                                  \
        }                                                                      \
        __syncthreads();                                                       \
        PRE()                                                                  \
                                                                               \
        float pg[FC_MOE_DEDUP_RT], pu[FC_MOE_DEDUP_RT];                        \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {                            \
            pg[j] = 0.0f;                                                      \
            pu[j] = 0.0f;                                                      \
        }                                                                      \
                                                                               \
        size_t expert_stride = (size_t)ff * (size_t)row_stride;                \
        const unsigned char *gb = gate_all + (size_t)(E) * expert_stride +     \
                                  (size_t)row * (size_t)row_stride;            \
        const unsigned char *ub = up_all + (size_t)(E) * expert_stride +       \
                                  (size_t)row * (size_t)row_stride;            \
                                                                               \
        int nsb = cols >> 8;                                                   \
        for (int sb = 0; sb < nsb; sb++) {                                     \
            __syncthreads();                                                   \
            STAGE(RC, sb)                                                      \
            __syncthreads();                                                   \
            if (active) {                                                      \
                float wg[8], wu[8];                                            \
                float dg = fc_q3k_decode_sb(gb, lane, wg);                     \
                float du = fc_q3k_decode_sb(ub, lane, wu);                     \
                _Pragma("unroll")                                              \
                for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {                    \
                    if (j < (RC)) {                                            \
                        const float *xb = sh_x + j * 256;                      \
                        float ag = 0.0f, au = 0.0f;                            \
                        _Pragma("unroll")                                      \
                        for (int i = 0; i < 8; i++) {                          \
                            float xv = xb[32 * i + lane];                      \
                            ag = __fmaf_rn(wg[i], xv, ag);                     \
                            au = __fmaf_rn(wu[i], xv, au);                     \
                        }                                                      \
                        pg[j] += dg * ag;                                      \
                        pu[j] += du * au;                                      \
                    }                                                          \
                }                                                              \
                gb += FC_Q3K_RBLOCK_BYTES;                                     \
                ub += FC_Q3K_RBLOCK_BYTES;                                     \
            }                                                                  \
        }                                                                      \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {                            \
            float g = fc_warp_sum_grp(pg[j]);                                  \
            float u = fc_warp_sum_grp(pu[j]);                                  \
            if (active && lane == 0 && j < (RC))                               \
                hid[(size_t)sh_fj[j] * ff + row] =                             \
                    (g / (1.0f + expf(-g))) * u;                               \
        }                                                                      \
    } while (0)

/* What the shipped kernels expand: the staging form FC_MOE_SHX_STAGE selects. */
#define FC_MOE_DEDUP_SHX_BODY(E, R0, RC)                                       \
    FC_MOE_DEDUP_SHX_BODY_S(FC_MOE_SHX_STAGE_PRE, FC_MOE_SHX_STAGE_X,          \
                            E, R0, RC)

__global__ void fc_moe_gateup_q3k_dedup_shx(const unsigned char *gate_all,
                                            const unsigned char *up_all,
                                            const float *x, const int *ecount,
                                            const int *ereq, float *hid, int k,
                                            int ff, int cols, int row_stride,
                                            int maxr) {
    __shared__ FC_MOE_SHX_ALIGN float sh_x[FC_MOE_DEDUP_RT * 256];
    __shared__ int sh_fj[FC_MOE_DEDUP_RT];
    __shared__ int sh_tok[FC_MOE_DEDUP_RT];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int e = blockIdx.y;
    int R = ecount[e];
    int r0 = blockIdx.z * FC_MOE_DEDUP_RT;
    if (r0 >= R)
        return;                       /* no requests for e in this tile */
    int rc = R - r0;
    if (rc > FC_MOE_DEDUP_RT)
        rc = FC_MOE_DEDUP_RT;         /* uniform across the block */
    int active = row < ff;

    FC_MOE_DEDUP_SHX_BODY(e, r0, rc);
}

#if FC_MOE_DEDUP_COMPACT
/*
 * ===========================================================================
 * COMPACTED GRID — compiled ONLY into a
 *     cd cuda_kernels && make NVCCFLAGS_EXTRA=-DFC_MOE_DEDUP_COMPACT=1
 * build, so the default fc_cuda_ptx.h is byte-identical without it, and then
 * opted into at runtime with FC_MOE_DEDUP_COMPACT=1. Two gates, both default
 * to the safe state; with the default PTX the runtime flag finds a NULL
 * CUfunction and the wrapper keeps the uncompacted launch.
 * ===========================================================================
 *
 * THE DEFECT. fc_moe_gateup_q3k_dedup{,_shx} launch
 * (ceil(ff/8), n_expert, ceil(maxr/RT)) = (64, 256, 16) = 262,144 blocks, of
 * which only ceil(ff/8) * sum_e ceil(R_e/RT) do any work — 16,448, i.e. 6.3 %,
 * at the shipped prefill shape. The rest load one int, find no work, exit.
 * MEASURED: launched with every
 * ecount zero, so that NO block does any arithmetic at all, the kernel still
 * costs 18.6 % of its full runtime. grid.y * grid.z is 4096 rather than ~257
 * because grid dims are host-side while the per-expert counts are device-side.
 *
 * THE FIX. Fold (expert, request-tile) onto ONE grid axis over a device-built
 * work list, and launch grid.y at the exact host-computable upper bound on the
 * number of work items (fc_moe_dedup_wmax, ../cuda/fc_cuda.h): 352 instead of
 * 4096, i.e. 22,528 blocks instead of 262,144 — 11.6x fewer, and 73 % of them
 * useful instead of 6.3 %. NO device->host sync is introduced: the bound is
 * closed-form in (n_expert, k, ntok, maxr).
 *
 * WHY IT IS BIT-EXACT — the whole argument:
 *   - The SET of (expert, r0, rc) work items is identical. The uncompacted grid
 *     runs (e, z) for every e with z*RT < ecount[e]; the work list holds exactly
 *     those, with the same r0 = z*RT and rc = min(ecount[e] - r0, RT).
 *   - The body is the SAME TOKENS (FC_MOE_DEDUP_SHX_BODY), over the same ereq,
 *     x and weights, so a work item produces the same floats either way.
 *   - Every output hid[f*ff + row] is written by exactly ONE work item, with a
 *     plain store. A request f is in exactly one expert's list exactly once, so
 *     it falls in exactly one tile. No atomic, no accumulation, no
 *     read-modify-write across blocks: the order blocks run in is not
 *     observable. That is stronger than "same accumulation order" — there is no
 *     shared accumulation to order.
 *   - rc is a loop bound, never an operand.
 * The work list is built in expert order by an exclusive scan, so it is also
 * deterministic run to run — which the shipped ereq is not (atomicAdd order),
 * and does not need to be, for the same reason.
 */

/* Expert count the single-block scan below can hold. Beyond it
 * fc_moe_dedup_wmax returns 0 and the caller keeps the uncompacted launch. */
#define FC_MOE_WL_EXPERT_CAP 1024

/* One (expert, request-tile) work item, packed so a gate+up block learns
 * everything from ONE dependent global load — the same single load the
 * uncompacted kernel spends on ecount[e]:
 *   .x = expert, or -1 for "past the end of the list" (the launch window is the
 *        host's upper bound, so its tail is sentinel-filled)
 *   .y = r0 << 16 | rc      (r0 < maxr <= 65535, 1 <= rc <= FC_MOE_DEDUP_RT)
 * Keep the maxr limit in sync with fc_moe_dedup_wmax in ../cuda/fc_cuda.c. */
#define FC_MOE_WL_PACK(r0, rc) (((r0) << 16) | (rc))
#define FC_MOE_WL_R0(y)        ((y) >> 16)
#define FC_MOE_WL_RC(y)        ((y) & 0xFFFF)

/*
 * Compact the per-expert request counts into a (expert, request-tile) work list.
 *   ecount: [n_expert] int32, from fc_moe_dedup_build (which this does NOT
 *           modify — it runs as a separate launch so that kernel's PTX is
 *           untouched)
 *   wlist:  [wmax] int2 (output) — the work items, then a (-1, 0) sentinel tail
 * Single block, like fc_moe_dedup_build: n_expert is small and this is ~0.05 %
 * of the gate+up launch it feeds.
 *
 * Thread 0 scans serially over shared memory. n_expert is at most
 * FC_MOE_WL_EXPERT_CAP, and the alternative — a shuffle or tree scan — buys
 * microseconds on a launch this small while re-importing a runtime-bounded
 * reduction tree, the exact defect class this tree has spent a campaign
 * deleting.
 */
__global__ void fc_moe_dedup_build_wl(const int *ecount, int2 *wlist,
                                      int n_expert, int wmax) {
    __shared__ int sh_wbase[FC_MOE_WL_EXPERT_CAP + 1];
    int tid = threadIdx.x;
    int threads = blockDim.x;

    if (n_expert > FC_MOE_WL_EXPERT_CAP)
        return;                       /* the host guarantees this; be explicit */

    /* Sentinel the whole launch window FIRST, then barrier: the real entries
     * below overwrite a prefix of it and must not race the fill. */
    for (int w = tid; w < wmax; w += threads)
        wlist[w] = make_int2(-1, 0);
    for (int e = tid; e < n_expert; e += threads) {
        int r = ecount[e];
        sh_wbase[e] = (r > 0) ? (r + FC_MOE_DEDUP_RT - 1) / FC_MOE_DEDUP_RT : 0;
    }
    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        for (int e = 0; e < n_expert; e++) {
            int t = sh_wbase[e];
            sh_wbase[e] = acc;
            acc += t;
        }
        sh_wbase[n_expert] = acc;     /* == the real work-item count */
    }
    __syncthreads();

    for (int e = tid; e < n_expert; e += threads) {
        int r = ecount[e];
        int w = sh_wbase[e];
        for (int r0 = 0; r0 < r; r0 += FC_MOE_DEDUP_RT, w++) {
            int rc = r - r0;
            if (rc > FC_MOE_DEDUP_RT)
                rc = FC_MOE_DEDUP_RT;
            if (w < wmax)             /* wmax >= sum ceil(R/RT) by construction */
                wlist[w] = make_int2(e, FC_MOE_WL_PACK(r0, rc));
        }
    }
}

/*
 * fc_moe_gateup_q3k_dedup_shx over the COMPACTED work list.
 *
 * Grid: (ceil(ff/8), wmax, 1); block 256; 8 KB + 64 B static shared — the same
 * shape, block and shared as _shx with grid.y*grid.z collapsed onto grid.y.
 * ecount is not read: rc travels in the work item, so a block still pays
 * exactly ONE dependent load before it can tell whether it has work.
 */
__global__ void fc_moe_gateup_q3k_dedup_shxc(const unsigned char *gate_all,
                                             const unsigned char *up_all,
                                             const float *x, const int *ereq,
                                             const int2 *wlist, float *hid,
                                             int k, int ff, int cols,
                                             int row_stride, int maxr) {
    __shared__ FC_MOE_SHX_ALIGN float sh_x[FC_MOE_DEDUP_RT * 256];
    __shared__ int sh_fj[FC_MOE_DEDUP_RT];
    __shared__ int sh_tok[FC_MOE_DEDUP_RT];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int2 item = wlist[blockIdx.y];
    int e = item.x;
    if (e < 0)
        return;                       /* past the end of the work list */
    int r0 = FC_MOE_WL_R0(item.y);
    int rc = FC_MOE_WL_RC(item.y);
    int active = row < ff;

    FC_MOE_DEDUP_SHX_BODY(e, r0, rc);
}
/*
 * Expert-grouped (dedup) Q8_0 gate+up SwiGLU for prefill token counts: the
 * Q8_0 sibling of fc_moe_gateup_q3k_dedup_shx. Bit-exact with
 * fc_moe_gateup_q80_nx / _nxp per request.
 *
 * gate_all/up_all: [n_expert * ff * row_stride] Q8_0 expert super-blocks
 *                  (288 B per 256 cols); row_stride = cols / 256 * 288
 * x:               [ntok * cols] fp32 (per-token x_norm)
 * ecount/ereq:     per-expert request lists from fc_moe_dedup_build
 * hid:             [ntok * k * ff] fp32, hid[f * ff + row] (f = tok * k + slot)
 *
 * Grid (ceil(ff / 8), wmax) over the COMPACTED work list of
 * fc_moe_dedup_build_wl (item = expert, first request, request count); block
 * 256 (8 warps, warp w owns FF row blockIdx.x * 8 + w); 8 KB + 64 B static
 * shared. The uncompacted (ff/8, n_expert, maxr/8) grid is ~131k blocks at a
 * 128-token chunk, nearly all empty, and measured only 1.11x over nx. Per super-block the block stages the tile's x slices
 * (RT x 256 floats), each warp loads its gate and up super-block lanes ONCE
 * and applies them to every request in the tile. The nx kernels instead load
 * them once per (token, slot): at a 128-token chunk each expert row is read
 * ~4x per chunk from DRAM, which is what bounds Qwen3.6 prefill MoE.
 *
 * EXACTNESS: per request the arithmetic is FC_MOE_GATEUP_BODY's with
 * fc_q80_sb_lane_grp2sp: the same fc_q80_sb_dot2 on the same lane words,
 * scales and x float4s, super-blocks in order, the same fc_warp_sum_grp tree
 * and the same silu(g) * u expression. Requests never mix.
 */
__global__ void fc_moe_gateup_q80_dedup(const unsigned char *gate_all,
                                        const unsigned char *up_all,
                                        const float *x, const int *ereq,
                                        const int2 *wlist, float *hid, int k,
                                        int ff, int cols, int row_stride,
                                        int maxr) {
    __shared__ __align__(16) float sh_x[FC_MOE_DEDUP_RT * 256];
    __shared__ int sh_fj[FC_MOE_DEDUP_RT];
    __shared__ int sh_tok[FC_MOE_DEDUP_RT];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int2 item = wlist[blockIdx.y];
    int e = item.x;
    if (e < 0)
        return;                       /* past the end of the work list */
    int r0 = FC_MOE_WL_R0(item.y);
    int rc = FC_MOE_WL_RC(item.y);
    int active = row < ff;

    if ((int)threadIdx.x < FC_MOE_DEDUP_RT) {
        int j = threadIdx.x;
        int f = (j < rc) ? ereq[(size_t)e * maxr + r0 + j] : -1;
        sh_fj[j] = f;
        sh_tok[j] = (f >= 0) ? f / k : 0;
    }

    float pg[FC_MOE_DEDUP_RT], pu[FC_MOE_DEDUP_RT];
#pragma unroll
    for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {
        pg[j] = 0.0f;
        pu[j] = 0.0f;
    }
    size_t expert_stride = (size_t)ff * (size_t)row_stride;
    const unsigned char *gb = gate_all + (size_t)e * expert_stride +
                              (size_t)row * (size_t)row_stride;
    const unsigned char *ub = up_all + (size_t)e * expert_stride +
                              (size_t)row * (size_t)row_stride;

    int nsb = cols >> 8;
    for (int sb = 0; sb < nsb; sb++) {
        __syncthreads();
        for (int j = 0; j < rc; j++)
            sh_x[j * 256 + threadIdx.x] =
                __ldg(x + (size_t)sh_tok[j] * cols + (size_t)sb * 256 + threadIdx.x);
        __syncthreads();
        if (active) {
            fc_q80_sbl bg = fc_q80_sb_load(gb, lane);
            fc_q80_sbl bu = fc_q80_sb_load(ub, lane);
#pragma unroll
            for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {
                if (j < rc) {
                    const float *xb = sh_x + j * 256;
                    float4 xv[2] = { FC_XLOAD4_PLAIN(xb, lane),
                                     FC_XLOAD4_PLAIN(xb + 128, lane) };
                    fc_q80_sb_dot2(bg, bu, xv, &pg[j], &pu[j]);
                }
            }
            gb += FC_Q80_SB_BYTES;
            ub += FC_Q80_SB_BYTES;
        }
    }

#pragma unroll
    for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {
        float g = fc_warp_sum_grp(pg[j]);
        float u = fc_warp_sum_grp(pu[j]);
        if (active && lane == 0 && j < rc)
            hid[(size_t)sh_fj[j] * ff + row] = (g / (1.0f + expf(-g))) * u;
    }
}

#endif /* FC_MOE_DEDUP_COMPACT */

} /* extern "C" */
