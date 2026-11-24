/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_attention_compute.c
 * @brief Compute third of the resident flash-decode ATTENTION triad. For one query (packed Q,
 *   dim_tiles tiles) against a KV cache delivered in blocks of block_tiles position tiles,
 *   computes softmax(Q K^T * scale + mask) V with an online softmax, never materialising the
 *   whole score row. Per block:
 *
 *     S      = Qs . K_block^T          (Qs = Q * scale, once per descriptor)
 *     S     += mask                    only for position tiles not entirely live
 *     m_new  = max(m_old, rowmax(S))   row-max reduce folded with the previous running max
 *     P      = exp(S - m_new)          in place in the score CB
 *     s      = rowsum(P)
 *     first block:  O = P.V, l = s
 *     later blocks: c = exp(m_old - m_new); l = c*l + s; O = c*O + P.V
 *
 *   and finally out = O * (1/l).
 *
 *   Non-obvious requirements:
 *   - The running max is seeded with the previous block's max (binary max in DST). A
 *     block-local max still gives correct-looking correlation, so nothing else would catch it.
 *   - The mask is additive and applied before the max, so stale cache data past the live end
 *     cannot win the max.
 *   - Every block processes exactly block_tiles tiles, dead ones included, so each CB's
 *     push/pop counts are fixed per descriptor; all trip counts come from the parameter block.
 *   - fp32 DST accumulation is required (the cb table provides it).
 *
 *   Compiled three times (FT_TRISC_UNPACK / MATH / PACK); every hardware statement sits in one
 *   FT_TENSIX_<thread>(...) wrapper. Control flow, waits and the queue protocol run on all.
 *
 *   -DATTN_SPLITK (the `attention_k8v4_split` program) adds split-K's two descriptor kinds
 *   (resident_attention_protocol.h §7): PARTIAL emits O, m and l instead of O / l; MERGE folds
 *   a group's partials through seed_accumulators / merge_block, as more blocks. Without it
 *   the preprocessed source is the shipped one and the images are byte-identical.
 *
 *   FAST-MATH BUILDS (programs attention_k8v4{,_split}_{fexp,lofi,fast}; Pantheon
 *   PN_TT_ATTN_FAST), two switches measured apart, both MATH-thread only:
 *   -DATTN_FAST_EXP  every exp is ft_tensix_sfpu_exp_attention.h's bf16-grade exp for a <= 0
 *                    (12 SFPU instructions an element against 39; max rel error 2^-9.12).
 *   -DATTN_FAST_LOFI the QK^T and PV matmuls at LoFi (one fidelity phase, not the CB table's
 *                    HiFi4); the eltwise mul and the reduces keep the table's fidelity.
 *   Neither is defined in a shipped build, whose preprocessed source is unchanged.
 *
 * Compile-time args: 0 cb_q, 1 cb_k, 2 cb_v, 3 cb_const (scaler, scale, tail mask, dead mask),
 *   4 cb_s, 5 cb_scratch, 6 cb_m_a, 7 cb_m_b, 8 cb_l, 9 cb_lsum, 10 cb_c, 11 cb_o, 12 cb_pv,
 *   13 cb_out, 14 cb_qs, 15 wait_mode (0 upstream, 1 stoppable, 2 bound-only).
 *
 * Build:
 *   CTARGS=0,1,2,3,4,5,6,7,8,9,10,11,12,16,14,1 \
 *       CB_TABLE=kernels/cb_tables/attention_float16b.cbtable \
 *       bash foundry/tt/build_tensix_kernel.sh \
 *           foundry/tt/kernels/resident_attention_compute.c out/ttk_attention
 *
 * The UNPACK image sits near the top of its 0x3000-byte text slab; the two noinline wait
 * wrappers below are what keep it inside (measured). The split build, with its verbs and merge
 * steps out of line, measured 12,044 B (2026-11-16).
 *
 * Split build: as attention_k8v4 plus KERNEL_DEFINES="... -DATTN_SPLITK=1"
 *   (tools/check_splitk_program_gates.sh). Fast-math builds: the base recipe plus
 *   -DATTN_FAST_EXP=1 and/or -DATTN_FAST_LOFI=1 (tools/check_attn_fastmath_program_gates.sh).
 *
 * @status Clean-room rewrite (plan Phase 4.7).
 * @issues None known.
 * @todo None
 */
#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_cb_formats.h"
#include "ft_tensix_compute.h"
#include "ft_tensix_compute_broadcast.h"
#include "ft_tensix_compute_cb.h"
#include "ft_tensix_compute_matmul.h"
#include "ft_tensix_compute_reduce.h"
#include "ft_tensix_kernel_args.h"
#ifdef FT_TRISC_MATH
#include "ft_tensix_compute_math.h"
#include "ft_tensix_compute_sfpu.h"
#include "ft_tensix_math_eltwise_sfpu.h"
#ifdef ATTN_FAST_EXP
#include "ft_tensix_sfpu_exp_attention.h"
#endif
#endif
#ifdef FT_TRISC_UNPACK
#include "ft_tensix_compute_sfpu.h"
#include "ft_tensix_compute_unpack.h"
#endif
#ifdef FT_TRISC_PACK
#include "ft_tensix_compute_pack.h"
#endif

#include "resident_queue_protocol.h"
#include "resident_attention_protocol.h"

FTQ_STATIC_ASSERT(FT_CB_ANY_32BIT_UNPACK == 0,
                  "a CB with 32-bit unpack needs an unpack-to-DST copy path; this kernel only "
                  "has the unpack-then-datacopy path");

#define cb_q ft_tensix_compile_time_arg(0)
#define cb_k ft_tensix_compile_time_arg(1)
#define cb_v ft_tensix_compile_time_arg(2)
#define cb_const ft_tensix_compile_time_arg(3)
#define cb_s ft_tensix_compile_time_arg(4)
#define cb_scratch ft_tensix_compile_time_arg(5)
#define cb_m_a ft_tensix_compile_time_arg(6)
#define cb_m_b ft_tensix_compile_time_arg(7)
#define cb_l ft_tensix_compile_time_arg(8)
#define cb_lsum ft_tensix_compile_time_arg(9)
#define cb_c ft_tensix_compile_time_arg(10)
#define cb_o ft_tensix_compile_time_arg(11)
#define cb_pv ft_tensix_compile_time_arg(12)
#define cb_out ft_tensix_compile_time_arg(13)
#define cb_qs ft_tensix_compile_time_arg(14)
#define wait_mode ft_tensix_compile_time_arg(15)

enum { dst_register = 0, dst_previous = 1 };

/* P0 TIMING PROBE. Undefined in every shipped build, so
 * the preprocessed source -- and every shipped image -- is unchanged. A probe image computes a
 * WRONG answer by design; only its ms-per-descriptor slope is read. Bits: 1 no QK matmul issue,
 * 2 no PV matmul issue, 4 no sub-bcast + exp, 8 no row reduces, 16 LoFi matmuls (now also the
 * real switch ATTN_FAST_LOFI, below), 32 no math at all (reader and CB flow only). */
#ifdef ATTN_PROBE
#define ATTN_PROBE_SKIP(bit) ((ATTN_PROBE) & (bit))
#else
#define ATTN_PROBE_SKIP(bit) 0
#endif

/* MATMUL FIDELITY. HiFi (the CB table's math_fidelity phases) unless ATTN_FAST_LOFI or probe
 * bit 16 asks for LoFi: one phase, the high mantissa bits of each operand only. Only the two
 * matmuls read these; mul, add/sub and the reduces take FT_CB_MATH_FIDELITY directly. The
 * EXP: the accurate fp32 body, or with ATTN_FAST_EXP the bf16-grade one for a <= 0 (every
 * exp here is S - max or m_old - m_new). */
#ifdef FT_TRISC_MATH
#if defined(ATTN_FAST_LOFI) || (defined(ATTN_PROBE) && ((ATTN_PROBE) & 16))
#define MATMUL_HIGH_FIDELITY 0
#else
#define MATMUL_HIGH_FIDELITY (FT_CB_MATH_FIDELITY != 0)
#endif
#define MATMUL_INNER_LOOPS (MATMUL_HIGH_FIDELITY ? (uint32_t)FT_CB_MATH_FIDELITY : 1u)
#define MATMUL_FIDELITY_INCREMENT (MATMUL_HIGH_FIDELITY ? 1u : 0u)
#ifdef ATTN_FAST_EXP
#define ATTENTION_EXP_TILE_INIT() ft_tensix_math_exp_attention_tile_init()
#define ATTENTION_EXP_TILE(dst) ft_tensix_math_exp_attention_tile(dst)
#else
#define ATTENTION_EXP_TILE_INIT() ft_tensix_math_exp_tile_init()
#define ATTENTION_EXP_TILE(dst) ft_tensix_math_exp_tile(dst)
#endif
#endif

/* Encoded FPU words handed to the add/sub init (not issued here). */
#define ATTENTION_ADD_WORD                                                                     \
    FT_BH_ENC_ELWADD(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_0, 0)
#define ATTENTION_SUB_WORD                                                                     \
    FT_BH_ENC_ELWSUB(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, FT_BH_ADDR_MOD_0, 0)

/* ===== Out-of-line waits (noinline keeps the UNPACK image inside its slab) =============== */

static __attribute__((noinline)) bool attention_wait_front(ftq_context *ctx, uint32_t cb,
                                                           uint32_t pages)
{
    return ftq_cb_wait_front_stoppable(ctx, (int32_t)cb, (int32_t)pages, wait_mode);
}

static __attribute__((noinline)) bool attention_reserve_back(ftq_context *ctx, uint32_t cb,
                                                             uint32_t pages)
{
    return ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, (int32_t)pages, wait_mode);
}

/* ===== SrcA format (block-float KV builds only) ========================================== */

/* THE KV CACHE MAY BE BLOCK FLOAT (attention_bfp8 / attention_bfp4: the CB table makes cb_k and
 * cb_v Bfp8_b / Bfp4_b, every other CB Float16_b; ft_tt_op_memory.h §1.6). Only unpacker A
 * ever sees K or V (in1 of both matmuls feeds SrcA), so A is switched LAZILY: to the KV format
 * at a matmul init, back to Float16_b at the first other init. Per block that is a handful of
 * switches, not one per tile. The bf16 build compiles none of this.
 *
 * K AND V MAY DIFFER (attention_k8v4, _k16v4, _k4v16, _k16v8: -DATTN_K_TILE_BYTES and
 * -DATTN_V_TILE_BYTES, the CB table gives cb_k and cb_v their own formats). A matmul init then
 * selects its OWN in1's format: QK^T reads cb_k, PV reads cb_v. */
#if defined(ATTN_K_TILE_BYTES) && defined(ATTN_V_TILE_BYTES)
#define ATTN_MIXED_KV 1
#endif
#if defined(ATTN_KV_TILE_BYTES) || defined(ATTN_MIXED_KV)
#ifdef ATTN_MIXED_KV
/* What A holds: the CB whose format it was last set up for. hw_configure set it up for cb_k. */
enum { kSrcaHoldsBf16 = 0, kSrcaHoldsK = 1, kSrcaHoldsV = 2 };
#endif
#ifdef FT_TRISC_UNPACK
/**
 * Reprogram unpacker A (SrcA) for an operand of another format: its source format, its
 * destination format, the output strides that follow its datum width, and the tile-size GPR:
 * the A half of ft_tensix_unpack_hw_configure_ab that depends on the format (the shared ALU
 * word is left as configured). The stall holds the config writes until unpacker 0 has drained
 * what it was given.
 *
 * Needed by a kernel whose SrcA operands differ in format (attention with a block-float KV
 * cache: K/V Bfp8_b or Bfp4_b, every other operand Float16_b).
 */
static FT_BH_INLINE void attention_unpack_reconfig_srca(uint32_t src_format, uint32_t dst_format,
                                                        uint32_t tile_size)
{
    const uint32_t dst_m = FT_BH_DATA_FORMAT_CONFIG_BITS(dst_format);

    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_CFG, FT_BH_STALLWAIT_COND_UNPACK0);
    /* The output strides follow the destination datum width (1 B block float, 2 B Float16_b),
     * exactly as ft_tensix_unpack_hw_configure_ab sets them for A. */
    FT_TENSIX_CFG_RMW(FT_BH_CFG_UNPACKER0_OUTPUT_Z_STRIDE_WORD,
                      FT_BH_CFG_UNPACKER0_OUTPUT_Z_STRIDE_SHIFT,
                      FT_BH_CFG_UNPACKER0_OUTPUT_Z_STRIDE_MASK, FT_TENSIX_UNPACK_Z_STRIDE(dst_m));
    FT_TENSIX_CFG_RMW(FT_BH_CFG_UNPACKER0_OUTPUT_Y_STRIDE_WORD,
                      FT_BH_CFG_UNPACKER0_OUTPUT_Y_STRIDE_SHIFT,
                      FT_BH_CFG_UNPACKER0_OUTPUT_Y_STRIDE_MASK, FT_TENSIX_UNPACK_Y_STRIDE(dst_m));
    FT_TENSIX_CFG_RMW(FT_BH_CFG_UNPACKER0_TILE_DESCRIPTOR_WORD, 0, 0xFu,
                      FT_BH_DATA_FORMAT_CONFIG_BITS(src_format));
    FT_TENSIX_CFG_RMW(FT_BH_CFG_UNPACKER0_OUTPUT_FORMAT_WORD, 0, 0xFu,
                      FT_BH_DATA_FORMAT_CONFIG_BITS(dst_format));
    FT_BH_PUSH_SETDMAREG(0, FT_TENSIX_LOWER_HALFWORD(tile_size), 0,
                         FT_TENSIX_LO_16(FT_TENSIX_GPR_UNPACK_TILE_SIZE_A));
}

#ifdef ATTN_MIXED_KV
static uint32_t attention_srca_cb_of(uint32_t holds)
{
    return holds == kSrcaHoldsV ? cb_v : holds == kSrcaHoldsK ? cb_k : cb_s;
}

/** Point A at `to_holds`'s CB. Two CBs of ONE format (cb_s and a bf16 K or V) need nothing:
 *  the format fixes the tile size too. */
static FT_BH_INLINE void attention_unpack_select_srca(uint32_t from_holds, uint32_t to_holds)
{
    const uint32_t to = attention_srca_cb_of(to_holds);
    const uint32_t from = attention_srca_cb_of(from_holds);

    if (ft_cb_unpack_src_format[to] != ft_cb_unpack_src_format[from] ||
        ft_cb_unpack_dst_format[to] != ft_cb_unpack_dst_format[from])
        attention_unpack_reconfig_srca(ft_cb_unpack_src_format[to], ft_cb_unpack_dst_format[to],
                                       ft_tensix_cb(to)->fifo_page_size);
}
#endif
#endif

#ifdef ATTN_MIXED_KV
static uint32_t attention_srca_holds = kSrcaHoldsK;

/* NOT inlined: eight init sites call it, and TRISC0's text has a 12 KiB slab. */
static __attribute__((noinline)) void attention_srca_select_mixed(uint32_t want)
{
    if (attention_srca_holds == want)
        return;
    FT_TENSIX_UNPACK(attention_unpack_select_srca(attention_srca_holds, want));
    attention_srca_holds = want;
}
#define ATTENTION_SRCA_KV(in1) \
    attention_srca_select_mixed((in1) == cb_v ? kSrcaHoldsV : kSrcaHoldsK)
#define ATTENTION_SRCA_BF16() attention_srca_select_mixed(kSrcaHoldsBf16)
#else
static uint32_t attention_srca_holds_kv = 1u; /* hw_configure set A up for cb_k */

/* NOT inlined: eight init sites call it, and TRISC0's text has a 12 KiB slab. */
static __attribute__((noinline)) void attention_srca_select(uint32_t kv)
{
    if (attention_srca_holds_kv == kv)
        return;
    FT_TENSIX_UNPACK(attention_unpack_reconfig_srca(
        ft_cb_unpack_src_format[kv ? cb_k : cb_s], ft_cb_unpack_dst_format[kv ? cb_k : cb_s],
        ft_tensix_cb(kv ? cb_k : cb_s)->fifo_page_size));
    attention_srca_holds_kv = kv;
}
#define ATTENTION_SRCA_KV(in1) attention_srca_select(1u)
#define ATTENTION_SRCA_BF16() attention_srca_select(0u)
#endif
#else
#define ATTENTION_SRCA_KV(in1) ((void)0)
#define ATTENTION_SRCA_BF16() ((void)0)
#endif

/* ===== Compute verbs (thread order inside each is fixed) ================================ */

/* THE SPLIT BUILD CALLS THE VERBS, AND THE BLOCK MERGE STEPS, FROM MORE SITES: inline, its
 * UNPACK image measured 15,480 B against the 12,288 B slab. There each is ONE out-of-line copy
 * (noipa: no inlining and no per-constant clones). The shipped builds keep them exactly as
 * they were, because their images must not move. The hot-loop matmul apply stays inline. */
#ifdef ATTN_SPLITK
#define ATTENTION_VERB static __attribute__((noipa))
#define ATTENTION_STEP static __attribute__((noipa))
#else
#define ATTENTION_VERB static FT_BH_INLINE
#define ATTENTION_STEP static
#endif

/** Matmul init; in0 feeds SrcB and in1 feeds SrcA. */
ATTENTION_VERB void attention_matmul_init(uint32_t in0, uint32_t in1, uint32_t transpose)
{
    (void)in0;
    (void)in1;
    (void)transpose;
    ATTENTION_SRCA_KV(in1);
    FT_TENSIX_MATH(ft_tensix_math_matmul_init(transpose, MATMUL_FIDELITY_INCREMENT,
                                              MATMUL_INNER_LOOPS, MATMUL_HIGH_FIDELITY));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul_init(
        transpose,
        (uint32_t)ft_cb_unpack_tile_num_faces[in1] * ft_cb_unpack_tile_face_rows[in1] *
                FT_TENSIX_FACE_C_DIM -
            1u,
        (uint32_t)ft_cb_unpack_tile_num_faces[in0] * ft_cb_unpack_tile_face_rows[in0] *
                FT_TENSIX_FACE_C_DIM -
            1u,
        1u /* kt_dim */));
}

/** Accumulate in0[index0] . in1[index1] into DST. */
static FT_BH_INLINE void attention_matmul_apply(uint32_t in0, uint32_t in1, uint32_t index0,
                                                uint32_t index1)
{
    (void)in0;
    (void)in1;
    (void)index0;
    (void)index1;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_matmul(in0, in1, index0, index1, 1 /* ct_dim */));
    FT_TENSIX_MATH(ft_tensix_math_matmul(dst_register));
}

/** Multiply init: MATH first, then UNPACK. */
ATTENTION_VERB void attention_mul_init(uint32_t a)
{
    (void)a;
    ATTENTION_SRCA_BF16();
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_init(FT_CB_MATH_FIDELITY,
                                                          ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[a],
                                              ft_cb_unpack_num_face_rows[a],
                                              ft_cb_unpack_num_face_cols[a]));
}

/** a[index_a] * b[index_b] into DST. */
ATTENTION_VERB void attention_mul_apply(uint32_t a, uint32_t b, uint32_t index_a,
                                             uint32_t index_b)
{
    (void)a;
    (void)b;
    (void)index_a;
    (void)index_b;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, index_a, index_b));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul(
        dst_register, (uint32_t)ft_cb_unpack_tile_num_faces[a], ft_cb_unpack_tile_face_rows[a]));
}

/** Add/sub init (lowest fidelity): MATH first, then UNPACK. */
ATTENTION_VERB void attention_addsub_init(uint32_t binary_op_word, uint32_t a)
{
    (void)binary_op_word;
    (void)a;
    ATTENTION_SRCA_BF16();
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub_init(
        binary_op_word, (uint32_t)ft_cb_unpack_tile_num_faces[a],
        (uint32_t)ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[a],
                                              ft_cb_unpack_num_face_rows[a],
                                              ft_cb_unpack_num_face_cols[a]));
}

/** a[index_a] +/- b[index_b] into DST (op chosen by the init). */
ATTENTION_VERB void attention_addsub_apply(uint32_t a, uint32_t b, uint32_t index_a,
                                                uint32_t index_b)
{
    (void)a;
    (void)b;
    (void)index_a;
    (void)index_b;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, index_a, index_b));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub(dst_register));
}

/** Subtract-broadcast-column init (lowest fidelity). */
static FT_BH_INLINE void attention_sub_bcast_col_init(uint32_t a)
{
    (void)a;
    ATTENTION_SRCA_BF16();
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_sub_broadcast_col_init(
        (uint32_t)ft_cb_unpack_num_face_cols[a], (uint32_t)ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_broadcast_col_init(ft_cb_unpack_tile_face_rows[a],
                                                            ft_cb_unpack_num_face_rows[a],
                                                            ft_cb_unpack_num_face_cols[a]));
}

/** a - column(b) into DST. */
static FT_BH_INLINE void attention_sub_bcast_col_apply(uint32_t a, uint32_t b)
{
    (void)a;
    (void)b;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, 0, 0));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_sub_broadcast_col(
        dst_register, (uint32_t)ft_cb_unpack_num_face_rows[a]));
}

/** Multiply-broadcast-column init. */
ATTENTION_VERB void attention_mul_bcast_col_init(uint32_t a)
{
    (void)a;
    ATTENTION_SRCA_BF16();
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_broadcast_col_init(
        FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[a]));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_broadcast_col_init(ft_cb_unpack_tile_face_rows[a],
                                                            ft_cb_unpack_num_face_rows[a],
                                                            ft_cb_unpack_num_face_cols[a]));
}

/** a * column(b) into DST. */
ATTENTION_VERB void attention_mul_bcast_col_apply(uint32_t a, uint32_t b)
{
    (void)a;
    (void)b;
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab(a, b, 0, 0));
    FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_broadcast_col(
        dst_register, FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_num_face_rows[a],
        (uint32_t)ft_cb_unpack_num_face_cols[a], (uint32_t)ft_cb_unpack_tile_face_rows[a]));
}

/** Copy init: UNPACK first, then MATH. */
ATTENTION_VERB void attention_copy_init(uint32_t cb)
{
    (void)cb;
    ATTENTION_SRCA_BF16();
    FT_TENSIX_UNPACK(ft_tensix_unpack_a_init(
        ft_cb_unpack_tile_face_rows[cb],
        (uint32_t)ft_cb_unpack_num_face_rows[cb] * ft_cb_unpack_num_face_cols[cb]));
    FT_TENSIX_MATH(ft_tensix_math_datacopy_init((uint32_t)ft_cb_unpack_num_face_rows[cb] *
                                                ft_cb_unpack_num_face_cols[cb]));
}

/** Copy the front tile of `cb` into DST slot `dst_index`. */
ATTENTION_VERB void attention_copy_apply(uint32_t cb, uint32_t dst_index)
{
    (void)cb;
    (void)dst_index;
    FT_TENSIX_UNPACK(ft_tensix_unpack_a(cb, 0));
    FT_TENSIX_MATH(ft_tensix_math_datacopy(dst_index));
}

/* Undo a reduce; the PACK clear is the only thing that removes the reduce packer edge mask.
 * One-line head and column-0 braces so check_tensix_reduce_exit_state.py sees the wrapper. */
static FT_BH_INLINE void attention_reduce_uninit(void)
{
    FT_TENSIX_MATH(ft_tensix_math_reduce_uninit());
    FT_TENSIX_PACK(ft_tensix_pack_reduce_mask_clear());
}

/* ===== DST commits ====================================================================== */

/** Hand DST to PACK and pack it as one page of `cb`. */
static void commit_dst_to(ftq_context *ctx, uint32_t cb)
{
    FT_TENSIX_MATH(ft_tensix_math_dest_section_done());
    (void)attention_reserve_back(ctx, cb, 1);
    FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
    FT_TENSIX_PACK(ft_tensix_pack_tile(dst_register, cb));
    FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, 1));
}

/** As commit_dst_to, over the front tile just consumed: the CB is full, so pop first. */
static void commit_dst_in_place(ftq_context *ctx, uint32_t cb)
{
    FT_TENSIX_MATH(ft_tensix_math_dest_section_done());
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb, 1));
    (void)attention_reserve_back(ctx, cb, 1);
    FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
    FT_TENSIX_PACK(ft_tensix_pack_tile(dst_register, cb));
    FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, 1));
}

/* ===== Steps ============================================================================ */

/** cb_qs <- cb_q * scale, dim_tiles tiles. */
static void scale_query(ftq_context *ctx, const fta_shape *shape)
{
    uint32_t dt;

    for (dt = 0; dt < shape->dim_tiles; dt++) {
        attention_mul_init(cb_q);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        attention_mul_apply(cb_q, cb_const, dt, kAttnCbConstScale);
        commit_dst_to(ctx, cb_qs);
    }
    (void)attention_wait_front(ctx, cb_qs, shape->dim_tiles);
}

/** cb_s <- Qs . K_block^T (+ mask where the position tile is not entirely live). */
static void score_block(ftq_context *ctx, const fta_shape *shape, uint32_t block)
{
    uint32_t pt;
    uint32_t dt;

    for (pt = 0; pt < shape->block_tiles; pt++) {
        const uint32_t mask = fta_mask_for_tile(shape, block * shape->block_tiles + pt);

        attention_matmul_init(cb_qs, cb_k, 1u /* transpose B */);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        for (dt = 0; dt < shape->dim_tiles && !ATTN_PROBE_SKIP(1); dt++)
            attention_matmul_apply(cb_qs, cb_k, dt, pt * shape->dim_tiles + dt);
        if (mask >= (uint32_t)kAttnCbConstPages) {
            commit_dst_to(ctx, cb_s);
            continue;
        }
        commit_dst_to(ctx, cb_scratch);
        (void)attention_wait_front(ctx, cb_scratch, 1);
        attention_addsub_init(ATTENTION_ADD_WORD, cb_scratch);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        attention_addsub_apply(cb_scratch, cb_const, 0, mask);
        commit_dst_to(ctx, cb_s);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_scratch, 1));
    }
}

/** destination <- max(rowmax(S), previous if seed). */
static void running_max(ftq_context *ctx, const fta_shape *shape, uint32_t destination,
                        uint32_t previous, bool seed)
{
    uint32_t pt;

    (void)previous;
    (void)seed;
    ATTENTION_SRCA_BF16();
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_reduce_row_max_init(
        (uint32_t)ft_cb_unpack_tile_num_faces[cb_s], (uint32_t)ft_cb_unpack_tile_face_rows[cb_s]));
    FT_TENSIX_MATH(
        ft_tensix_math_reduce_row_max_init(FT_CB_MATH_FIDELITY != 0 ? 1u : 0u, FT_CB_MATH_FIDELITY));
    FT_TENSIX_PACK(
        ft_tensix_pack_reduce_row_mask_config((uint32_t)ft_cb_pack_tile_face_rows[destination]));
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    for (pt = 0; pt < shape->block_tiles && !ATTN_PROBE_SKIP(8); pt++) {
        FT_TENSIX_MATH(ft_tensix_math_reduce_row_max(dst_register,
                                                     (uint32_t)ft_cb_unpack_num_face_rows[cb_s],
                                                     (uint32_t)ft_cb_unpack_num_face_cols[cb_s]));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_reduce_row_max(cb_s, cb_const, pt,
                                                            kAttnCbConstScaler));
    }
    /* The uninit must precede the copy below. */
    attention_reduce_uninit();
    if (seed) {
        attention_copy_init(previous);
        attention_copy_apply(previous, dst_previous);
        FT_TENSIX_MATH(ft_tensix_math_binary_max_tile_init());
        FT_TENSIX_MATH(ft_tensix_math_binary_max_tile(dst_register, dst_previous, dst_register));
    }
    commit_dst_to(ctx, destination);
    (void)attention_wait_front(ctx, destination, 1);
}

/** cb_s <- exp(S - column(maximum)), in place. */
static void exponentiate_block(ftq_context *ctx, const fta_shape *shape, uint32_t maximum)
{
    uint32_t pt;

    for (pt = 0; pt < shape->block_tiles; pt++) {
        attention_sub_bcast_col_init(cb_s);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        if (!ATTN_PROBE_SKIP(4)) {
            attention_sub_bcast_col_apply(cb_s, maximum);
            FT_TENSIX_MATH(ATTENTION_EXP_TILE_INIT());
            FT_TENSIX_MATH(ATTENTION_EXP_TILE(dst_register));
        }
        commit_dst_in_place(ctx, cb_s);
    }
    (void)attention_wait_front(ctx, cb_s, shape->block_tiles);
}

/** cb_lsum <- rowsum(P). */
static void block_row_sum(ftq_context *ctx, const fta_shape *shape)
{
    uint32_t pt;

    ATTENTION_SRCA_BF16();
    FT_TENSIX_UNPACK(
        ft_tensix_unpack_ab_reduce_row_init((uint32_t)ft_cb_unpack_tile_num_faces[cb_s]));
    FT_TENSIX_MATH(
        ft_tensix_math_reduce_row_init(FT_CB_MATH_FIDELITY != 0 ? 1u : 0u, FT_CB_MATH_FIDELITY));
    FT_TENSIX_PACK(
        ft_tensix_pack_reduce_row_mask_config((uint32_t)ft_cb_pack_tile_face_rows[cb_lsum]));
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    for (pt = 0; pt < shape->block_tiles && !ATTN_PROBE_SKIP(8); pt++) {
        FT_TENSIX_MATH(ft_tensix_math_reduce_row(dst_register));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_reduce_row(cb_s, cb_const, pt, kAttnCbConstScaler));
    }
    attention_reduce_uninit();
    commit_dst_to(ctx, cb_lsum);
    (void)attention_wait_front(ctx, cb_lsum, 1);
}

/** cb_pv <- P . V_block, dim_tiles tiles. */
static void multiply_by_values(ftq_context *ctx, const fta_shape *shape)
{
    uint32_t dt;
    uint32_t pt;

    for (dt = 0; dt < shape->dim_tiles; dt++) {
        attention_matmul_init(cb_s, cb_v, 0u);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        for (pt = 0; pt < shape->block_tiles && !ATTN_PROBE_SKIP(2); pt++)
            attention_matmul_apply(cb_s, cb_v, pt, pt * shape->dim_tiles + dt);
        commit_dst_to(ctx, cb_pv);
    }
    (void)attention_wait_front(ctx, cb_pv, shape->dim_tiles);
}

/** First block: O <- P.V, l <- s. */
ATTENTION_STEP void seed_accumulators(ftq_context *ctx, const fta_shape *shape)
{
    uint32_t dt;

    for (dt = 0; dt < shape->dim_tiles; dt++) {
        attention_copy_init(cb_pv);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        attention_copy_apply(cb_pv, dst_register);
        commit_dst_to(ctx, cb_o);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_pv, 1));
    }
    attention_copy_init(cb_lsum);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    attention_copy_apply(cb_lsum, dst_register);
    commit_dst_to(ctx, cb_l);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_lsum, 1));
    (void)attention_wait_front(ctx, cb_o, shape->dim_tiles);
    (void)attention_wait_front(ctx, cb_l, 1);
}

/** Later blocks: c = exp(m_old - m_new); l = c*l + s; O = c*O + P.V. */
ATTENTION_STEP void merge_block(ftq_context *ctx, const fta_shape *shape, uint32_t maximum,
                        uint32_t previous)
{
    uint32_t dt;

    attention_addsub_init(ATTENTION_SUB_WORD, previous);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    attention_addsub_apply(previous, maximum, 0, 0);
    FT_TENSIX_MATH(ATTENTION_EXP_TILE_INIT());
    FT_TENSIX_MATH(ATTENTION_EXP_TILE(dst_register));
    commit_dst_to(ctx, cb_c);
    (void)attention_wait_front(ctx, cb_c, 1);

    attention_mul_init(cb_l);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    attention_mul_apply(cb_l, cb_c, 0, 0);
    commit_dst_in_place(ctx, cb_l);
    (void)attention_wait_front(ctx, cb_l, 1);

    attention_addsub_init(ATTENTION_ADD_WORD, cb_l);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    attention_addsub_apply(cb_l, cb_lsum, 0, 0);
    commit_dst_in_place(ctx, cb_l);
    (void)attention_wait_front(ctx, cb_l, 1);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_lsum, 1));

    for (dt = 0; dt < shape->dim_tiles; dt++) {
        attention_mul_bcast_col_init(cb_o);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        attention_mul_bcast_col_apply(cb_o, cb_c);
        commit_dst_in_place(ctx, cb_o);
    }
    (void)attention_wait_front(ctx, cb_o, shape->dim_tiles);

    for (dt = 0; dt < shape->dim_tiles; dt++) {
        attention_addsub_init(ATTENTION_ADD_WORD, cb_o);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        attention_addsub_apply(cb_o, cb_pv, 0, 0);
        /* Written out rather than commit_dst_in_place: this commit pops two CBs. */
        FT_TENSIX_MATH(ft_tensix_math_dest_section_done());
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_o, 1));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_pv, 1));
        (void)attention_reserve_back(ctx, cb_o, 1);
        FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
        FT_TENSIX_PACK(ft_tensix_pack_tile(dst_register, cb_o));
        FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
        FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_o, 1));
    }
    (void)attention_wait_front(ctx, cb_o, shape->dim_tiles);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_c, 1));
}

/** out <- O * (1/l), then release every per-descriptor CB. */
ATTENTION_STEP void normalise_and_emit(ftq_context *ctx, const fta_shape *shape, uint32_t maximum)
{
    uint32_t dt;

    (void)maximum;
    attention_copy_init(cb_l);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    attention_copy_apply(cb_l, dst_register);
    FT_TENSIX_MATH(ft_tensix_math_recip_tile_init());
    FT_TENSIX_MATH(ft_tensix_math_recip_tile(dst_register));
    commit_dst_to(ctx, cb_c);
    (void)attention_wait_front(ctx, cb_c, 1);

    for (dt = 0; dt < shape->dim_tiles; dt++) {
        attention_mul_bcast_col_init(cb_o);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        attention_mul_bcast_col_apply(cb_o, cb_c);
        FT_TENSIX_MATH(ft_tensix_math_dest_section_done());
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_o, 1));
        (void)attention_reserve_back(ctx, cb_out, 1);
        FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
        FT_TENSIX_PACK(ft_tensix_pack_tile(dst_register, cb_out));
        FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
        FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, 1));
    }
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_c, 1));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_l, 1));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(maximum, 1));
#ifdef ATTN_SPLITK
    if (fta_is_merge(shape))
        return; /* a MERGE filled none of the decode CBs below */
#endif
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_qs, shape->dim_tiles));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_q, shape->dim_tiles));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_const, kAttnCbConstPages));
}

#ifdef ATTN_SPLITK
/* ===== Split-K (resident_attention_protocol.h §7; the `_split` build only) ================ */

/** Copy the front tile of `from` to cb_out and pop it. */
static __attribute__((noipa)) void emit_front_tile(ftq_context *ctx, uint32_t from)
{
    attention_copy_init(from);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    attention_copy_apply(from, dst_register);
    commit_dst_to(ctx, cb_out);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(from, 1));
}

/** PARTIAL: emit O (un-normalised), m and l, then release the CBs normalise_and_emit does. */
static __attribute__((noipa)) void emit_partial(ftq_context *ctx, const fta_shape *shape,
                                                uint32_t maximum)
{
    uint32_t dt;

    for (dt = 0; dt < shape->dim_tiles; dt++)
        emit_front_tile(ctx, cb_o);
    emit_front_tile(ctx, maximum);
    emit_front_tile(ctx, cb_l);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_qs, shape->dim_tiles));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_q, shape->dim_tiles));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_const, kAttnCbConstPages));
}

/**
 * MERGE, one partial, as ONE MORE BLOCK of the online softmax: the reader has pushed (m_i, l_i)
 * into cb_const and O_i into cb_q. `maximum` <- m_i (first) or max(m_i, previous); then
 * c = e^(m_i - maximum), cb_lsum <- c * l_i, cb_pv <- c * O_i -- which is what a block hands
 * seed_accumulators / merge_block (its l and P.V taken against the new max).
 */
static __attribute__((noipa)) void merge_stage_partial(ftq_context *ctx, const fta_shape *shape,
                                                       uint32_t maximum, uint32_t previous,
                                                       bool seed)
{
    uint32_t dt;

    (void)attention_wait_front(ctx, cb_const, kAttnPartialStatTiles);
    attention_copy_init(cb_const);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    attention_copy_apply(cb_const, dst_register);
    if (seed) {
        attention_copy_init(previous);
        attention_copy_apply(previous, dst_previous);
        FT_TENSIX_MATH(ft_tensix_math_binary_max_tile_init());
        FT_TENSIX_MATH(ft_tensix_math_binary_max_tile(dst_register, dst_previous, dst_register));
    }
    commit_dst_to(ctx, maximum);
    (void)attention_wait_front(ctx, maximum, 1);

    attention_addsub_init(ATTENTION_SUB_WORD, cb_const);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    attention_addsub_apply(cb_const, maximum, 0, 0);
    FT_TENSIX_MATH(ATTENTION_EXP_TILE_INIT());
    FT_TENSIX_MATH(ATTENTION_EXP_TILE(dst_register));
    commit_dst_to(ctx, cb_c);
    (void)attention_wait_front(ctx, cb_c, 1);

    attention_mul_init(cb_const);
    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
    attention_mul_apply(cb_const, cb_c, 1, 0);
    commit_dst_to(ctx, cb_lsum);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_const, kAttnPartialStatTiles));
    (void)attention_wait_front(ctx, cb_lsum, 1);

    (void)attention_wait_front(ctx, cb_q, shape->dim_tiles);
    for (dt = 0; dt < shape->dim_tiles; dt++) {
        attention_mul_bcast_col_init(cb_q);
        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        attention_mul_bcast_col_apply(cb_q, cb_c);
        commit_dst_to(ctx, cb_pv);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_q, 1));
    }
    (void)attention_wait_front(ctx, cb_pv, shape->dim_tiles);
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_c, 1));
}

/**
 * MERGE: the group's P partials folded in order, each through the block merge the one-core
 * loop already runs (seed_accumulators, then merge_block), then out = O * (1/l) through
 * normalise_and_emit. An odd P leaves cb_const two pages short of its 4-page grid, so the
 * reader pushes one spare pair, dropped here (resident_attention_reader.c read_partials).
 */
static __attribute__((noipa)) void merge_partials(ftq_context *ctx, const fta_shape *shape)
{
    uint32_t maximum = cb_m_a;
    uint32_t previous = cb_m_b;
    uint32_t i;

    for (i = 0; i < shape->kv_positions; i++) {
        const uint32_t swap = maximum;

        merge_stage_partial(ctx, shape, maximum, previous, i != 0u);
        if (i == 0u) {
            seed_accumulators(ctx, shape);
        } else {
            merge_block(ctx, shape, maximum, previous);
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(previous, 1));
        }
        maximum = previous;
        previous = swap;
    }
    if ((shape->kv_positions % 2u) != 0u) {
        (void)attention_wait_front(ctx, cb_const, kAttnPartialStatTiles);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_const, kAttnPartialStatTiles));
    }
    normalise_and_emit(ctx, shape, previous);
}
#endif

#if defined(ATTN_PROBE) && ((ATTN_PROBE) & 32)
/** Probe 32: consume every CB the reader fills and emit dim_tiles unwritten output pages. */
static __attribute__((noinline)) void probe_flow_only(ftq_context *ctx, const fta_shape *shape)
{
    const uint32_t depth = shape->block_tiles * shape->dim_tiles;
    uint32_t block;

    (void)attention_wait_front(ctx, cb_q, shape->dim_tiles);
    (void)attention_wait_front(ctx, cb_const, kAttnCbConstPages);
    for (block = 0; block < shape->blocks; block++) {
        (void)attention_wait_front(ctx, cb_k, depth);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_k, depth));
        (void)attention_wait_front(ctx, cb_v, depth);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_v, depth));
    }
    (void)attention_reserve_back(ctx, cb_out, shape->dim_tiles);
    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, shape->dim_tiles));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_q, shape->dim_tiles));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_const, kAttnCbConstPages));
}
#endif

/** Run one ATTENTION descriptor; the two max CBs ping-pong between blocks. */
static bool compute_one_attention(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const fta_shape shape = fta_decode(parameters);
    const uint32_t depth = shape.block_tiles * shape.dim_tiles;
    uint32_t maximum = cb_m_a;
    uint32_t previous = cb_m_b;
    uint32_t block;

    if (!fta_shape_valid(&shape)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
#ifdef ATTN_SPLITK
    if (fta_is_merge(&shape)) {
        merge_partials(ctx, &shape);
        return !ctx->aborting;
    }
#endif
#if defined(ATTN_PROBE) && ((ATTN_PROBE) & 32)
    probe_flow_only(ctx, &shape);
    return !ctx->aborting;
#endif

    (void)attention_wait_front(ctx, cb_q, shape.dim_tiles);
    (void)attention_wait_front(ctx, cb_const, kAttnCbConstPages);
    scale_query(ctx, &shape);

    for (block = 0; block < shape.blocks; block++) {
        const uint32_t swap = maximum;

        (void)attention_wait_front(ctx, cb_k, depth);
        score_block(ctx, &shape, block);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_k, depth));
        (void)attention_wait_front(ctx, cb_s, shape.block_tiles);
        running_max(ctx, &shape, maximum, previous, block != 0u);
        exponentiate_block(ctx, &shape, maximum);
        block_row_sum(ctx, &shape);
        (void)attention_wait_front(ctx, cb_v, depth);
        multiply_by_values(ctx, &shape);
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_v, depth));
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(cb_s, shape.block_tiles));
        if (block == 0u) {
            seed_accumulators(ctx, &shape);
        } else {
            merge_block(ctx, &shape, maximum, previous);
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(previous, 1));
        }
        maximum = previous;
        previous = swap;
    }
    /* After the loop the final max is in `previous`. */
#ifdef ATTN_SPLITK
    if (fta_is_partial(&shape)) {
        emit_partial(ctx, &shape, previous);
        return !ctx->aborting;
    }
#endif
    normalise_and_emit(ctx, &shape, previous);
    return !ctx->aborting;
}

void kernel_main(void)
{
    ftq_context ctx;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);
    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    /* Once, with the source order reversed for the dominant matmul (in0 -> SrcB, in1 ->
     * SrcA): the unpacker pair is cb_k as A, cb_qs as B. */
    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[cb_k], ft_cb_unpack_src_format[cb_qs],
        ft_cb_unpack_dst_format[cb_k], ft_cb_unpack_dst_format[cb_qs],
        ft_cb_unpack_tile_face_rows[cb_k], ft_cb_unpack_tile_face_rows[cb_qs],
        ft_cb_unpack_tile_num_faces[cb_k], ft_cb_unpack_tile_num_faces[cb_qs],
        ft_tensix_cb(cb_k)->fifo_page_size, ft_tensix_cb(cb_qs)->fifo_page_size));
    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[cb_k],
                                               ft_cb_unpack_dst_format[cb_qs],
                                               FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(
        ft_cb_pack_src_format[cb_s], ft_cb_pack_dst_format[cb_s],
        ft_tensix_cb(cb_s)->fifo_page_size, ft_cb_pack_tile_num_faces[cb_s],
        ft_cb_pack_partial_face[cb_s], 0 /* relu */, FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[cb_s], ft_cb_pack_tile_face_rows[cb_s],
                                       ft_cb_pack_tile_num_faces[cb_s], 1));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());
    attention_matmul_init(cb_qs, cb_k, 0u);

    ctx.cursor = ctx.control[kFieldCompletedCount];
    while (running) {
        volatile FTQ_L1_PTR uint32_t *descriptor;
        uint32_t opcode;

        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;
        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);

        descriptor = ftq_descriptor(&ctx);
        opcode = descriptor[kDescOpcode];
        ftq_status_opcode(&ctx, opcode);
        if (opcode == kOpAttention) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            (void)compute_one_attention(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
            if (ctx.aborting)
                running = false;
        } else if (opcode == kOpStop) {
            running = false;
        }
        if (!ctx.aborting)
            ctx.cursor++;
    }
    ftq_status_set(&ctx, kStateExit);
}
