/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_reduce_compute.c
 * @brief Compute third of the RESIDENT REDUCE triad -- ONE binary answering four row-wise
 *   ops over a [rows x W] activation, selected by the pass count, the pool, the SFPU verb
 *   and the final chain, all of them DESCRIPTOR FIELDS (resident_reduce_protocol.h):
 *
 *     softmax        m = max x ; e = exp(x - m) ; y = e / sum e
 *     RMS norm       y = x * gamma * rsqrt( mean(x^2) + eps )
 *     layer norm     y = (x - mean) * gamma * rsqrt( mean((x-mean)^2) + eps ) + beta
 *     router norm    y = x / sum x
 *
 *   at 32x32 tile granularity, so one "row block" is 32 rows by Wt tiles.
 *
 * THIS IS A MERGE, NOT FOUR KERNELS IN A SWITCH. There is one loop over passes and one loop
 * over chain steps; the four ops are four programs through the same code. The differences
 * between softmax and an RMS norm turned out to be a trip count and two verbs.
 *
 * FIVE THINGS THAT ARE LOAD-BEARING, unchanged by the port:
 *
 *   1. THE MAX PASS IS NOT OPTIONAL FOR SOFTMAX, AND A PCC GATE WILL NOT TELL YOU THAT.
 *      exp overflows bf16 at x ~ 88 and random operands in [-1, 1) never come near it, so a
 *      softmax with no max subtraction passes every acceptance in this repo and produces inf
 *      on the first real attention row. Worse, softmax(x - c) = softmax(x) makes a WRONG max
 *      invisible to correlation as well (docs/tt_bringup_log.md 22.2). The suite plants a
 *      +200 outlier so it is not.
 *   2. AND THE SAME TRAP HAS A SECOND SHAPE HERE. RMS norm, layer norm and the router
 *      normalise are all SCALE-related, and PCC is blind to a global scale factor -- so a
 *      reduce that sums the row twice, or a scaler tile off by 2, moves every output by
 *      1/sqrt(2) and moves PCC by nothing. The acceptance asserts MAX-ABS for this family,
 *      not only PCC, and tests/test_phase6_vector.c proves that check red.
 *   3. ONE SCALER TILE SERVES BOTH REDUCES OF A DESCRIPTOR. Upstream's reduce scaler is 1.0
 *      for SUM and for MAX, and the two passes of a member always want the same one (1.0 for
 *      softmax and the router, 1/W for both norms), so there is one scaler operand.
 *   4. eps COSTS NO CODE. reduce_tile ACCUMULATES into DST across calls, so one extra reduce
 *      against a constant tile whose every element is eps*W/32 adds exactly eps to the mean
 *      square -- no broadcast add, no extra pack, no extra CB round trip. Each row of that
 *      tile has 32 columns: 32 * (eps*W/32) * (1/W) = eps.
 *   5. `reduce_uninit()` COMES AFTER THE PACK, as it does in rms_norm_compute.c. Upstream's
 *      own softmax.cpp calls it before, which packs the full DST tile instead of just the
 *      reduce mask's column 0. Both work because only column 0 is ever consumed, and this
 *      file follows the ordering that is green on THIS silicon.
 *
 * WHAT THE PORT MAKES VISIBLE, and every item is an ORDERING that no type enforces:
 *
 *   * `reduce_uninit` IS THE ONLY THING THAT CLEARS THE PACKER EDGE MASK, and MATH's half of
 *     it is an EMPTY BODY -- `ft_tensix_pack_reduce_mask_clear()` is the whole of it. The
 *     state is per-core and outlives the program, so a kernel that EXITS between
 *     `ft_tensix_pack_reduce_row_mask_config()` and the clear POISONS THE NEXT IMAGE on that
 *     core. That is why every abort path here still runs to the end of the descriptor rather
 *     than returning: the drain rule and the mask rule want the same thing.
 *   * AFTER A REDUCE, MATH STILL HOLDS reduce's ADDR_MOD_0..3, its MOP template and (SUM
 *     arm) its eight MVMULs in replay slots 16-23, and UNPACK still has the haloize bit SET.
 *     The next op's init is what reprograms them, and nothing may unpack in between.
 *   * `ft_tensix_math_wait_for_dest_available()` (`tile_regs_acquire`) does NOT zero DST. The
 *     reduce is an accumulation into DST, so what makes each pass start from zero is PACK's
 *     per-tile `TT_ZEROACC` -- which names `ADDR_MOD_1`, a slot the MATH thread programs, so
 *     the zeroing is cross-thread and invisible in this file.
 *   * NON-COLUMN-0 LANES AFTER A REDUCE HOLD ZERO, not garbage: while the
 *     reduce packer mask is programmed every non-column-0 datum is packed as ZERO. So a
 *     column CB is a genuine column vector.
 *
 * AND THE TWO REDUCE ARMS ARE MIRROR IMAGES ON UNPACK, WHICH IS THE TRAP. MAX does NOT swap
 * the operands (data to SrcA, scaler to SrcB), records the face-by-face `AddrMode = 0b01`
 * UNPACR pair with a MOP inner loop of 4, and uses the immediate `SETADCXX`. The SUM arm is
 * the opposite on all three. That is why `reduce_row_sum()` and `reduce_row_max()` below are
 * two functions and not one with a flag -- exactly as the C++ was two template
 * instantiations.
 *
 * Compile-time args (positional):
 *   0: cb_in      activations, Wt pages
 *   1: cb_scaler  one tile of 1.0 or 1/W, pushed and popped once per descriptor
 *   2: cb_col0    one page, the FIRST pass's column (row max / mean / scale)
 *   3: cb_col1    one page, the SECOND pass's column
 *   4: cb_mid     the intermediate stream, Wt pages
 *   5: cb_eps     one tile of eps*W/32, pushed only when a pass asks for it
 *   6: cb_gamma   per-column weight, Wt pages
 *   7: cb_beta    per-column bias, Wt pages
 *   8: cb_stage0  final-chain scratch, 1 page (P3: >= the chain chunk K)
 *   9: cb_stage1  final-chain scratch, 1 page (P3: >= the chain chunk K)
 *  10: cb_out     output, >= 2 pages
 *  11: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build:
 *   CTARGS=0,1,2,3,4,5,6,7,8,9,16,1 CB_TABLE=kernels/cb_tables/reduce_float16b.cbtable \
 *       bash foundry/tt/build_tensix_kernel.sh \
 *           foundry/tt/kernels/resident_reduce_compute.c /tmp/ttk_reduce
 *
 * IT BUILDS WITH `reduce_float16b.cbtable`, i.e. DST_ACCUM_MODE 1. That is not a detail: it
 * selects `exp`'s fp32-accurate Juffa arm and `recip`'s fp32 arms, and it is why neither of
 * those bodies carries a bf16 rounding step. A `--uniform-legacy` build would need
 * `_sfpu_exp_21f_bf16_tti_`, which is deliberately not ported (ft_tensix_sfpu_math.h says so).
 *
 * @status CONVERTED, 2026-08-13 (plan Phase 6.6). Measured against `resident_reduce_
 *   compute.cpp` with tools/diff_tensix_kernel_instructions.sh; see the round report.
 * @issues exp_tile, recip_tile, rsqrt_tile and square_tile run over the whole DST tile, but
 *   only column 0 of a reduce result is meaningful; the other columns are ZERO (see the
 *   packer contract above), and recip(0) is +inf. `*_tiles_bcast_cols` reads column 0 only,
 *   so those lanes are never consumed. Do not "fix" this by zeroing them -- it would cost a
 *   pass. Same note as rms_norm_compute.c, same reason.
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_cb_formats.h"
#include "ft_tensix_compute.h"
#include "ft_tensix_compute_broadcast.h"
#include "ft_tensix_compute_cb.h"
#include "ft_tensix_compute_reduce.h"
#include "ft_tensix_kernel_args.h"
#ifdef FT_TRISC_MATH
#include "ft_tensix_compute_math.h"
#include "ft_tensix_compute_sfpu.h"
#include "ft_tensix_math_eltwise_sfpu.h"
#endif
#ifdef FT_TRISC_UNPACK
#include "ft_tensix_compute_sfpu.h"
#include "ft_tensix_compute_unpack.h"
#endif
#ifdef FT_TRISC_PACK
#include "ft_tensix_compute_pack.h"
#endif

#include "resident_queue_protocol.h"
#include "resident_reduce_protocol.h"

/* One tile at a time, so only destination register 0 is ever live. */
#define DST_REGISTER 0u

/* The C++ arm spelled these as file-scope `constexpr uint32_t`. A macro over the
 * compile-time-argument reader is the C equivalent that keeps them constants at every use
 * rather than locals threaded through twelve parameters. */
#define FTR_CB_IN ft_tensix_compile_time_arg(0)
#define FTR_CB_SCALER ft_tensix_compile_time_arg(1)
#define FTR_CB_COL0 ft_tensix_compile_time_arg(2)
#define FTR_CB_COL1 ft_tensix_compile_time_arg(3)
#define FTR_CB_MID ft_tensix_compile_time_arg(4)
#define FTR_CB_EPS ft_tensix_compile_time_arg(5)
#define FTR_CB_GAMMA ft_tensix_compile_time_arg(6)
#define FTR_CB_BETA ft_tensix_compile_time_arg(7)
#define FTR_CB_STAGE0 ft_tensix_compile_time_arg(8)
#define FTR_CB_STAGE1 ft_tensix_compile_time_arg(9)
#define FTR_CB_OUT ft_tensix_compile_time_arg(10)
#define FTR_WAIT_MODE ft_tensix_compile_time_arg(11)

/* M3 HALF TILE: the face counts the elementwise passes use for THIS descriptor. Full tile
 * (the CB tables' values) unless the descriptor's word-3 bit 23 is set; then faces 0-1 only.
 * Read by UNPACK and MATH; the row reduce, the verbs and PACK never change. */
static uint32_t ftr_nf_rows = 2u;  /* num_faces_r_dim */
static uint32_t ftr_nf_total = 4u; /* num_faces */

/* M7 REDUCE PHASE CLOCKS, UNPACK thread only (control words 57-59, this core's alone):
 * 57 map pass, 58 row reduce, 59 final chain — cycles >> 4, zeroed at boot. The unpacker
 * leads the three threads through every pass, so its clock brackets each pass's issue. */
static volatile FTQ_L1_PTR uint32_t *ftr_clock_control;

static void ftr_clock_add(uint32_t word, uint32_t started)
{
    (void)word;
    (void)started;
    FT_TENSIX_UNPACK(if (ftr_clock_control) ftr_clock_control[word] +=
                     (ft_bh_wall_clock_low() - started) >> 4);
}

/** DST -> `cb`, with the reserve overlapping the math drain (docs/tt_kernel_model.md 3). */
static void commit_dst_to(ftq_context *ctx, uint32_t cb)
{
    FT_TENSIX_MATH(ft_tensix_math_dest_section_done()); /* tile_regs_commit() */

    (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, 1, FTR_WAIT_MODE);

    FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done()); /* tile_regs_wait() */
    FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER, cb));
    FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32)); /* release */
    FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb, 1));
}

/** The SFPU verb a pass applies to its reduce result, in the reduce's own DST section. */
static void apply_verb(uint32_t verb)
{
    if (verb == (uint32_t)kRedVerbRecip) {
        FT_TENSIX_MATH(ft_tensix_math_recip_tile_init());
        FT_TENSIX_MATH(ft_tensix_math_recip_tile(DST_REGISTER));
    } else if (verb == (uint32_t)kRedVerbRsqrt) {
        FT_TENSIX_MATH(ft_tensix_math_rsqrt_tile_init());
        FT_TENSIX_MATH(ft_tensix_math_rsqrt_tile(DST_REGISTER));
    }
}

/**
 * One pass's MAP: build the stream the reduce will run over, out of `cb_in`, into `cb_mid`.
 *
 * A subtracting map always subtracts cb_col0 -- the first pass's output -- which is the
 * family invariant resident_reduce_protocol.h records rather than sending a column index.
 * The two inits sit INSIDE the loop, where upstream's softmax.cpp and rms_norm_compute.c
 * both put theirs; hoisting them is an optimisation with a correctness question attached
 * (does an SFPU init disturb the unpacker's broadcast configuration?) and this round does
 * not need the answer.
 *
 * `wait_step` 0: the caller has already waited for the whole row.
 */
static void map_row(ftq_context *ctx, uint32_t map, uint32_t tile_cols, uint32_t wait_step)
{
    uint32_t wt;

    for (wt = 0; wt < tile_cols; wt++) {
        /* P3 STREAMED: wait for cb_in a block at a time instead of a row at a time, so x*x
         * starts on the first block the reader lands. The waits are cumulative (nothing is
         * popped until the row is done), and `wait_step` divides Wt, so the last one asks
         * for exactly the row. */
        if (wait_step != 0u && (wt % wait_step) == 0u)
            (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)FTR_CB_IN, (int32_t)(wt + wait_step),
                                              FTR_WAIT_MODE);
        if (map == (uint32_t)kRedMapSquareSelf) {
            /* mul_tiles_init(cb_in, cb_in) */
            FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_init(
                FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[FTR_CB_IN]));
            FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[FTR_CB_IN],
                                                      ftr_nf_rows,
                                                      ft_cb_unpack_num_face_cols[FTR_CB_IN]));

            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available()); /* tile_regs_acquire */

            /* mul_tiles(cb_in, cb_in, wt, wt, DST) */
            FT_TENSIX_UNPACK(ft_tensix_unpack_ab(FTR_CB_IN, FTR_CB_IN, wt, wt));
            FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul(
                DST_REGISTER, ftr_nf_total, (uint32_t)ft_cb_unpack_tile_face_rows[FTR_CB_IN]));
        } else {
            /* sub_bcast_cols_init_short(cb_in, cb_col0) -- LoFi, always, see the broadcast
             * header's ELWSUB section note. */
            FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_sub_broadcast_col_init(
                (uint32_t)ft_cb_unpack_num_face_cols[FTR_CB_IN],
                (uint32_t)ft_cb_unpack_tile_face_rows[FTR_CB_IN]));
            FT_TENSIX_UNPACK(ft_tensix_unpack_ab_broadcast_col_init(
                ft_cb_unpack_tile_face_rows[FTR_CB_IN], ft_cb_unpack_num_face_rows[FTR_CB_IN],
                ft_cb_unpack_num_face_cols[FTR_CB_IN]));

            FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());

            /* sub_tiles_bcast_cols(cb_in, cb_col0, wt, 0, DST) */
            FT_TENSIX_UNPACK(ft_tensix_unpack_ab(FTR_CB_IN, FTR_CB_COL0, wt, 0));
            FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_sub_broadcast_col(
                DST_REGISTER, (uint32_t)ft_cb_unpack_num_face_rows[FTR_CB_IN]));

            if (map == (uint32_t)kRedMapSubExp) {
                FT_TENSIX_MATH(ft_tensix_math_exp_tile_init());
                FT_TENSIX_MATH(ft_tensix_math_exp_tile(DST_REGISTER));
            } else {
                FT_TENSIX_MATH(ft_tensix_math_square_tile_init());
                FT_TENSIX_MATH(ft_tensix_math_square_tile(DST_REGISTER));
            }
        }
        commit_dst_to(ctx, FTR_CB_MID);
    }
}

/**
 * One pass's SUM reduce: `tile_cols` tiles of `source` into column 0 of `column`, then the
 * verb.
 *
 * `reduce_tile` ACCUMULATES into DST across calls, which is both how the row is reduced at
 * all and how eps is added for free (header note 4).
 */
static void reduce_row_sum(ftq_context *ctx, uint32_t source, uint32_t column,
                           uint32_t tile_cols, const ftr_pass *pass)
{
    uint32_t wt;

    (void)source; /* UNPACK-only; MATH and PACK compile every use of it away */

    /* reduce_init<SUM, REDUCE_ROW>(source, cb_scaler, column). The SUM arm SWAPS the
     * operands -- the scaler goes to SrcA and the data to SrcB -- which is what makes the
     * row reduce a matrix multiply. */
    FT_TENSIX_UNPACK(
        ft_tensix_unpack_ab_reduce_row_init((uint32_t)ft_cb_unpack_tile_num_faces[source]));
    FT_TENSIX_MATH(ft_tensix_math_reduce_row_init(FT_CB_MATH_FIDELITY != 0 ? 1u : 0u,
                                                  FT_CB_MATH_FIDELITY));
    FT_TENSIX_PACK(
        ft_tensix_pack_reduce_row_mask_config((uint32_t)ft_cb_pack_tile_face_rows[column]));

    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available()); /* tile_regs_acquire */

    for (wt = 0; wt < tile_cols; wt++) {
        FT_TENSIX_MATH(ft_tensix_math_reduce_row(DST_REGISTER));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_reduce_row(source, FTR_CB_SCALER, wt, 0));
    }

    if (pass->add_eps != 0u) {
        FT_TENSIX_MATH(ft_tensix_math_reduce_row(DST_REGISTER));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_reduce_row(FTR_CB_EPS, FTR_CB_SCALER, 0, 0));
    }

    apply_verb(pass->verb);
    commit_dst_to(ctx, column);

    /* reduce_uninit(). MATH restores NOTHING (the body is empty upstream); PACK's edge mask
     * restore is the whole of it, and it is not optional -- see the file header. */
    FT_TENSIX_MATH(ft_tensix_math_reduce_uninit());
    FT_TENSIX_PACK(ft_tensix_pack_reduce_mask_clear());
}

/**
 * The same pass with a MAX pool. A separate function rather than a flag, because upstream
 * makes it a separate template instantiation and the two arms differ on all three threads.
 *
 * MULTI-TILE MAX ACCUMULATES CORRECTLY -- settled on silicon 2026-08-13, both Wt = 2 and
 * Wt = 4, with a one-tile negative control red on 32 of 32 rows. The `tile_cols` loop is not
 * defective and does not need a guard. `add_eps` cannot reach here: eps is one more
 * accumulated row SUM and `ftr_program_is_valid()` refuses it against a MAX.
 */
static void reduce_row_max(ftq_context *ctx, uint32_t source, uint32_t column,
                           uint32_t tile_cols, const ftr_pass *pass)
{
    uint32_t wt;

    (void)source; /* read by UNPACK and by MATH's face counts; unused on PACK */

    /* reduce_init<MAX, REDUCE_ROW>(source, cb_scaler, column). No operand swap here. */
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_reduce_row_max_init(
        (uint32_t)ft_cb_unpack_tile_num_faces[source], (uint32_t)ft_cb_unpack_tile_face_rows[source]));
    FT_TENSIX_MATH(ft_tensix_math_reduce_row_max_init(FT_CB_MATH_FIDELITY != 0 ? 1u : 0u,
                                                      FT_CB_MATH_FIDELITY));
    FT_TENSIX_PACK(
        ft_tensix_pack_reduce_row_mask_config((uint32_t)ft_cb_pack_tile_face_rows[column]));

    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());

    for (wt = 0; wt < tile_cols; wt++) {
        FT_TENSIX_MATH(ft_tensix_math_reduce_row_max(
            DST_REGISTER, (uint32_t)ft_cb_unpack_num_face_rows[source],
            (uint32_t)ft_cb_unpack_num_face_cols[source]));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_reduce_row_max(source, FTR_CB_SCALER, wt, 0));
    }

    if (pass->add_eps != 0u) {
        FT_TENSIX_MATH(ft_tensix_math_reduce_row_max(
            DST_REGISTER, (uint32_t)ft_cb_unpack_num_face_rows[source],
            (uint32_t)ft_cb_unpack_num_face_cols[source]));
        FT_TENSIX_UNPACK(
            ft_tensix_unpack_ab_reduce_row_max(FTR_CB_EPS, FTR_CB_SCALER, 0, 0));
    }

    apply_verb(pass->verb);
    commit_dst_to(ctx, column);

    FT_TENSIX_MATH(ft_tensix_math_reduce_uninit());
    FT_TENSIX_PACK(ft_tensix_pack_reduce_mask_clear());
}

/**
 * One chain step's INIT: program the unpacker and the FPU for `step` over `source`.
 *
 * Run once per step RUN (P3: up to K tiles) rather than once per tile. Re-running an init
 * reprograms the same MOP and address modifiers to the same values, so hoisting it out of a
 * run of identical tile ops is the upstream `*_init(); for (...) *_tiles(...)` shape and
 * changes no arithmetic. At K = 1 it is the pre-P3 init-per-tile order exactly.
 */
static void chain_step_init(uint32_t step, uint32_t source)
{
    (void)source; /* the shapes are read on UNPACK and MATH only */

    if (step == (uint32_t)kRedChainCenter) {
        /* sub_bcast_cols_init_short(source, cb_col0) */
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_sub_broadcast_col_init(
            (uint32_t)ft_cb_unpack_num_face_cols[source],
            (uint32_t)ft_cb_unpack_tile_face_rows[source]));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_broadcast_col_init(
            ft_cb_unpack_tile_face_rows[source], ft_cb_unpack_num_face_rows[source],
            ft_cb_unpack_num_face_cols[source]));
    } else if (step == (uint32_t)kRedChainGamma) {
        /* mul_tiles_init(source, cb_gamma) */
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_init(
            FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[source]));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[source], ftr_nf_rows,
                                                  ft_cb_unpack_num_face_cols[source]));
    } else if (step == (uint32_t)kRedChainScale) {
        /* mul_bcast_cols_init_short(source, scale_column) */
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_broadcast_col_init(
            FT_CB_MATH_FIDELITY, (uint32_t)ft_cb_unpack_tile_face_rows[source]));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_broadcast_col_init(
            ft_cb_unpack_tile_face_rows[source], ftr_nf_rows, ft_cb_unpack_num_face_cols[source]));
    } else {
        /* add_tiles_init(source, cb_beta) */
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub_init(
            FT_BH_ENC_ELWADD(0 /*clr_src*/, 0 /*acc_to_dest*/, FT_BH_ELWADD_SRCB_NO_BCAST,
                            FT_BH_ADDR_MOD_0, 0),
            (uint32_t)ft_cb_unpack_tile_num_faces[source],
            (uint32_t)ft_cb_unpack_tile_face_rows[source]));
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[source],
                                                  ft_cb_unpack_num_face_rows[source],
                                                  ft_cb_unpack_num_face_cols[source]));
    }
}

/** One chain step on one tile, into a fresh DST section. chain_step_init() ran first. */
static void chain_step_tile(uint32_t step, uint32_t source, uint32_t source_index, uint32_t wt,
                            uint32_t scale_column)
{
    (void)source;
    (void)source_index;
    (void)wt;
    (void)scale_column; /* the operand indices are UNPACK's; MATH reads only the shapes */

    FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());

    if (step == (uint32_t)kRedChainCenter) {
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab(source, FTR_CB_COL0, source_index, 0));
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_sub_broadcast_col(
            DST_REGISTER, (uint32_t)ft_cb_unpack_num_face_rows[source]));
    } else if (step == (uint32_t)kRedChainGamma) {
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab(source, FTR_CB_GAMMA, source_index, wt));
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul(
            DST_REGISTER, ftr_nf_total, (uint32_t)ft_cb_unpack_tile_face_rows[source]));
    } else if (step == (uint32_t)kRedChainScale) {
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab(source, scale_column, source_index, 0));
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_mul_broadcast_col(
            DST_REGISTER, FT_CB_MATH_FIDELITY, ftr_nf_rows,
            (uint32_t)ft_cb_unpack_num_face_cols[source],
            (uint32_t)ft_cb_unpack_tile_face_rows[source]));
    } else {
        FT_TENSIX_UNPACK(ft_tensix_unpack_ab(source, FTR_CB_BETA, source_index, wt));
        FT_TENSIX_MATH(ft_tensix_math_eltwise_binary_addsub(DST_REGISTER));
    }
}

/**
 * The final chain: up to four per-tile steps, the last of which writes cb_out.
 *
 * Intermediate steps ping-pong through two scratch CBs, which is rms_norm's cb_xg pattern
 * generalised: a step cannot read and write the same circular buffer. A member with a
 * single-step chain (softmax, the router, L2 norm) touches neither.
 *
 * P3 CHUNK: the tiles go through in runs of `chunk` — step 1 for the run, then step 2 for
 * the run, ... Per tile the steps, their operands and their pack points are the pre-P3 ones;
 * what changes is that the unpacker no longer waits on a full unpack->math->pack round trip
 * between a tile's two steps, and each step's init runs once a run. A scratch tile is always
 * read at index 0 and popped at once, so a run never addresses past the scratch CB's wrap
 * whatever its page count; the READER refuses a chunk larger than that page count, which is
 * the one way a run could wedge PACK. `chunk` = 1 is the pre-P3 order exactly.
 */
static void final_row(ftq_context *ctx, const ftr_program *program, uint32_t scale_column,
                      uint32_t first, uint32_t count, uint32_t chunk)
{
    const uint32_t stream = (program->source == (uint32_t)kRedSourceMid) ? FTR_CB_MID : FTR_CB_IN;
    uint32_t run_first;

    /* L4: this member's slice of the row; CB_GAMMA/CB_BETA hold only the slice, so their
     * index is `wt - first` while the input stream is indexed by the whole-row `wt`. */
    for (run_first = first; run_first < first + count; run_first += chunk) {
        const uint32_t left = first + count - run_first;
        const uint32_t tiles = left < chunk ? left : chunk;
        uint32_t source = stream;
        uint32_t remaining = program->chain;
        bool source_is_scratch = false;
        bool use_stage1 = false;

        while (remaining != 0u) {
            const uint32_t step = remaining & (~remaining + 1u); /* lowest set bit */
            const uint32_t rest = remaining & ~step;
            const uint32_t destination =
                (rest == 0u) ? FTR_CB_OUT : (use_stage1 ? FTR_CB_STAGE1 : FTR_CB_STAGE0);
            uint32_t t;

            chain_step_init(step, source);
            for (t = 0; t < tiles; t++) {
                const uint32_t wt = run_first + t;

                chain_step_tile(step, source, source_is_scratch ? 0u : wt, wt - first,
                                scale_column);
                commit_dst_to(ctx, destination);
                if (source_is_scratch) {
                    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(source, 1));
                }
            }
            if (rest != 0u) {
                (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)destination, (int32_t)tiles,
                                                  FTR_WAIT_MODE);
                source = destination;
                source_is_scratch = true;
                use_stage1 = !use_stage1;
            }
            remaining = rest;
        }
    }
}

/**
 * The compute half of one REDUCE descriptor.
 *
 * Every trip count comes from the parameter block and none from a wait, which is what makes
 * the drain safe (resident_queue_protocol.h rule 3): when a wait gives up, every pass still
 * runs to completion on whatever the CB pages hold, every push/pop and every tile_regs_*
 * pair still happens, and the three TRISC threads stay in lockstep.
 *
 * `reconfig_data_format(cb_scaler, source)` before the SUM reduce is what reduce.h asks for
 * on the REDUCE_ROW SUM path, and it is deliberately absent: it exists because that path
 * swaps the operands onto SrcA/SrcB, and it can only matter when the two operands carry
 * DIFFERENT formats. Every CB in this op's contract is Float16_b (reduce_float16b.cbtable),
 * which is also why rms_norm_compute.c omits it and scored 0.999979 on silicon. If the
 * on-card PCC ever comes back wrong for a shape, this is the first line to add back.
 */
static bool compute_one_reduce(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const ftr_program program = ftr_decode(parameters);
    const uint32_t tile_cols = parameters[kVecTileCols];
    const uint32_t tile_count = parameters[kVecTileCount];
    const uint32_t row_blocks = (tile_cols == 0u) ? 0u : tile_count / tile_cols;
    const uint32_t scale_column = (program.pass_count == 2u) ? FTR_CB_COL1 : FTR_CB_COL0;
    const bool uses_gamma = (program.chain & (uint32_t)kRedChainGamma) != 0u;
    const bool uses_beta = (program.chain & (uint32_t)kRedChainBeta) != 0u;
    const uint32_t schedule = parameters[kRedSchedule];
    /* P3: from the descriptor alone, so all three threads agree — none of them can see the
     * scratch CBs' page count; the reader refuses a chunk they cannot hold. */
    const uint32_t chunk = ftr_chain_chunk(schedule, (uint32_t)kRedSchedChunkMask);
    /* Block-by-block cb_in waits only where pass 0 is a map — the map is then cb_in's first
     * reader. A pass-0 reduce over cb_in (softmax, the router) keeps the whole-row wait. */
    const uint32_t wait_step =
        ((schedule & (uint32_t)kRedSchedStream) != 0u && program.pass[0].map != kRedMapNone)
            ? ftv_block_depth(tile_cols, ftv_depth_word(parameters[kVecDepth]))
            : 0u;
    uint32_t block;
    uint32_t slice_first, slice_count;

    if (!ftr_program_valid(&program) || tile_cols == 0u || row_blocks == 0u) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    ftr_slice(parameters[kRedFinal], tile_cols, &slice_first, &slice_count);
    /* UNPACK and MATH only: the `unpack_*` tables do not exist in the PACK build. Before
     * 2026-10-05 this block was unguarded, the PACK compile FAILED, and
     * build_tensix_kernel.sh linked a STALE kernel_body_trisc2.o (2026-10-04) — which is
     * why every change to the final chain's PACK flow (M7's hoist, M10) wedged PACK. */
#ifndef FT_TRISC_PACK
    if (ftv_half_tile(parameters[kVecDepth])) {
        ftr_nf_rows = 1u;
        ftr_nf_total = 2u;
    } else {
        ftr_nf_rows = (uint32_t)ft_cb_unpack_num_face_rows[FTR_CB_IN];
        ftr_nf_total = (uint32_t)ft_cb_unpack_tile_num_faces[FTR_CB_IN];
    }
#endif

    (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)FTR_CB_SCALER, 1, FTR_WAIT_MODE);
    if (ftr_program_uses_eps(&program))
        (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)FTR_CB_EPS, 1, FTR_WAIT_MODE);

    for (block = 0; block < row_blocks; block++) {
        bool mid_held = false;
        uint32_t index;

        if (wait_step == 0u)
            (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)FTR_CB_IN, (int32_t)tile_cols,
                                              FTR_WAIT_MODE);

        for (index = 0; index < program.pass_count; index++) {
            const ftr_pass *pass = &program.pass[index];
            const uint32_t column = (index == 0u) ? FTR_CB_COL0 : FTR_CB_COL1;
            uint32_t source = FTR_CB_IN;

            if (pass->map != (uint32_t)kRedMapNone) {
                {
                    const uint32_t started = ft_bh_wall_clock_low();

                    map_row(ctx, pass->map, tile_cols, index == 0u ? wait_step : 0u);
                    ftr_clock_add(57u, started);
                }
                (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)FTR_CB_MID, (int32_t)tile_cols,
                                                  FTR_WAIT_MODE);
                source = FTR_CB_MID;
                mid_held = true;
            }
            if (pass->pool == (uint32_t)kRedPoolMax)
                reduce_row_max(ctx, source, column, tile_cols, pass);
            else
            {
                const uint32_t started = ft_bh_wall_clock_low();

                reduce_row_sum(ctx, source, column, tile_cols, pass);
                ftr_clock_add(58u, started);
            }
            (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)column, 1, FTR_WAIT_MODE);

            /* The intermediate is spent unless the final chain is what scales it. Popping
             * it here rather than after the chain is what lets the reader start the next
             * row block while this one is still being written out. */
            if (mid_held && !(program.source == (uint32_t)kRedSourceMid &&
                              index + 1u == program.pass_count)) {
                FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(FTR_CB_MID, tile_cols));
                mid_held = false;
            }
        }

        /* Softmax's last reader of cb_in was the map, so its row can be released before the
         * output pass -- which is what the standalone softmax kernel did, and it is worth
         * keeping. */
        if (program.source == (uint32_t)kRedSourceMid) {
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(FTR_CB_IN, tile_cols));
        }

        /* gamma and beta are waited for HERE, just before the first chain step that reads
         * them, not at the top: a streamed reader moves them behind row 0's input (P3), and
         * the waits are cumulative and never popped before the descriptor ends, so a later
         * row's wait returns at once. */
        if (uses_gamma)
            (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)FTR_CB_GAMMA, (int32_t)slice_count,
                                              FTR_WAIT_MODE);
        if (uses_beta)
            (void)ftq_cb_wait_front_stoppable(ctx, (int32_t)FTR_CB_BETA, (int32_t)slice_count,
                                              FTR_WAIT_MODE);
        {
            const uint32_t started = ft_bh_wall_clock_low();

            final_row(ctx, &program, scale_column, slice_first, slice_count, chunk);
            ftr_clock_add(59u, started);
        }

        if (program.source == (uint32_t)kRedSourceIn) {
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(FTR_CB_IN, tile_cols));
        }
        if (mid_held) {
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(FTR_CB_MID, tile_cols));
        }
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(FTR_CB_COL0, 1));
        if (program.pass_count == 2u) {
            FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(FTR_CB_COL1, 1));
        }
    }

    FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(FTR_CB_SCALER, 1));
    if (ftr_program_uses_eps(&program)) {
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(FTR_CB_EPS, 1));
    }
    if (uses_gamma) {
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(FTR_CB_GAMMA, slice_count));
    }
    if (uses_beta) {
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_pop_front(FTR_CB_BETA, slice_count));
    }
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

    /* compute_kernel_hw_startup(cb_in, cb_scaler, cb_out): default source order, so cb_in
     * goes to SrcA and cb_scaler to SrcB (only matmul reverses it). ONCE, outside the queue
     * loop: the hardware configuration is a property of the binary, and every CB in this
     * op's contract carries the same format, so no descriptor can need a different one. */
    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[FTR_CB_IN], ft_cb_unpack_src_format[FTR_CB_SCALER],
        ft_cb_unpack_dst_format[FTR_CB_IN], ft_cb_unpack_dst_format[FTR_CB_SCALER],
        ft_cb_unpack_tile_face_rows[FTR_CB_IN], ft_cb_unpack_tile_face_rows[FTR_CB_SCALER],
        ft_cb_unpack_tile_num_faces[FTR_CB_IN], ft_cb_unpack_tile_num_faces[FTR_CB_SCALER],
        ft_tensix_cb(FTR_CB_IN)->fifo_page_size, ft_tensix_cb(FTR_CB_SCALER)->fifo_page_size));

    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[FTR_CB_IN],
                                               ft_cb_unpack_dst_format[FTR_CB_SCALER],
                                               FT_CB_DST_ACCUM_FP32));

    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(
        ft_cb_pack_src_format[FTR_CB_OUT], ft_cb_pack_dst_format[FTR_CB_OUT],
        ft_tensix_cb(FTR_CB_OUT)->fifo_page_size, ft_cb_pack_tile_num_faces[FTR_CB_OUT],
        ft_cb_pack_partial_face[FTR_CB_OUT], 0 /*relu_config*/, FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[FTR_CB_OUT],
                                       ft_cb_pack_tile_face_rows[FTR_CB_OUT],
                                       ft_cb_pack_tile_num_faces[FTR_CB_OUT], 1 /*num_tiles*/));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());

    ctx.cursor = ctx.control[kFieldCompletedCount];
    FT_TENSIX_UNPACK(ftr_clock_control = ctx.control; ctx.control[57] = 0u; ctx.control[58] = 0u;
                     ctx.control[59] = 0u);

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
        if (opcode == kOpReduce) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            (void)compute_one_reduce(
                &ctx, (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)descriptor[kDescA0]);
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
