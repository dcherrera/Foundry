/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file mop_expander_probe_compute.c
 * @brief Measurement kernel: makes the MOP expander record its own expansion in DST.
 *
 *   DST is used as a tape. Two uniform input tiles are unpacked once per trial (SrcA = A, SrcB =
 *   B; the host fills them with 5.0 and 2.0). Every FPU op the instruction source issues writes
 *   one 8-row DST slot and address modifier 0 advances the DST row counter by 8, so the k-th op
 *   lands in slot k, and each op family leaves a distinct value (add 7, sub 3 / -3, mul 10).
 *   The packed output tile is therefore the literal sequence of ops that were issued. DST is
 *   zeroed before each trial so unwritten slots read 0.
 *
 *   Trials are runtime arguments, two words each:
 *     word A (shape): bits 0-3 mode (0 run the MOP template, 1 hand-issue `ladder` copies of
 *                     loop_op0, 2 hand-issue `ladder` ops cycling loop_op0 / loop_op1 /
 *                     loop0_last, anything else issue nothing); 4-7 outer loop length;
 *                     8-11 inner loop length; 12-16 ladder length; 17-18 address-modifier
 *                     slot every op names.
 *     word B (ops):   nine 3-bit symbols (0 NOP, 1 ELWADD, 2 ELWSUB, 3 ELWMUL): 0-2 start_op0,
 *                     3-5 end_op0, 6-8 end_op1, 9-11 loop_op0, 12-14 loop_op1,
 *                     15-17 loop0_last_instr, 18-20 loop1_last_instr.
 *
 *   The unpack thread hand-issues exactly one UNPACR for A and one for B per trial instead of
 *   the library's MOP-driven unpack, so the number of operand-valid signals does not depend on
 *   the expansion rule under test (and cannot deadlock under some rules).
 *
 *   Runtime args: 0 trial count T; 1+2i word A and 2+2i word B of trial i. Not in the shipped
 *   program blob; driven by tests/test_phase2_mop_expander.c.
 *
 * Compile-time args: 0 cb_in0 (A input), 1 cb_in1 (B input), 2 cb_out.
 *
 * Build:
 *   CTARGS=0,1,16 bash foundry/tt/build_tensix_kernel.sh \
 *       foundry/tt/kernels/mop_expander_probe_compute.c out/ttk_mop_probe
 *
 * @status Clean-room rewrite (plan Phase 4.7).
 * @issues None known.
 * @todo None
 */
#include <stdint.h>

#include "ft_tensix_cb_formats.h"
#include "ft_tensix_compute.h"
#include "ft_tensix_compute_cb.h"
#include "ft_tensix_kernel_args.h"
#ifdef FT_TRISC_MATH
#include "ft_tensix_compute_math.h"
#endif
#ifdef FT_TRISC_UNPACK
#include "ft_tensix_compute_unpack.h"
#endif
#ifdef FT_TRISC_PACK
#include "ft_tensix_compute_pack.h"
#endif

#define DST_REGISTER 0u

#define PROBE_MODE_MOP 0u
#define PROBE_MODE_LADDER_UNIFORM 1u
#define PROBE_MODE_LADDER_CYCLE 2u
#define PROBE_MODE_SILENT 3u /* documentation only: the "issue nothing" fall-through */

#define PROBE_FIELD(w, shift, width) (((w) >> (shift)) & ((1u << (width)) - 1u))

#ifdef FT_TRISC_MATH

/** Encoded FPU word for one 3-bit op symbol (a value; nothing is issued). Clear-dvalid 0 keeps
 *  the operands valid for the whole tape; add-to-dest 0 makes each op overwrite its slot. */
static uint32_t probe_op_word(uint32_t symbol, uint32_t addr_mod)
{
    switch (symbol) {
    case 1:
        return FT_BH_ENC_ELWADD(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, addr_mod, 0);
    case 2:
        return FT_BH_ENC_ELWSUB(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, addr_mod, 0);
    case 3:
        return FT_BH_ENC_ELWMUL(0, 0, FT_BH_ELWADD_SRCB_NO_BCAST, addr_mod, 0);
    default:
        return FT_BH_ENC_NOP;
    }
}

/** Slot 0 steps DST by one FPU row group (the tape stepper); slots 1-3 step nothing. */
static void probe_configure_addrmod(void)
{
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_0, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(FT_TENSIX_MAX_FPU_ROWS, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0), FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_1, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0), FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_2, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0), FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
    FT_TENSIX_ADDR_MOD_SET(FT_BH_ADDR_MOD_3, FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_SRC_VAL(0, 0, 0),
                           FT_TENSIX_ADDR_MOD_DEST_VAL(0, 0, 0, 0),
                           FT_TENSIX_ADDR_MOD_FIDELITY_VAL(0, 0), FT_TENSIX_ADDR_MOD_BIAS_VAL(0, 0));
}

/** Issue one trial's ops through the MOP template or by hand. */
static void probe_run_trial(uint32_t shape, uint32_t ops)
{
    const uint32_t mode = PROBE_FIELD(shape, 0, 4);
    const uint32_t outer = PROBE_FIELD(shape, 4, 4);
    const uint32_t inner = PROBE_FIELD(shape, 8, 4);
    const uint32_t ladder = PROBE_FIELD(shape, 12, 5);
    const uint32_t addr_mod = PROBE_FIELD(shape, 17, 2);
    const uint32_t sym0 = PROBE_FIELD(ops, 9, 3);
    const uint32_t sym1 = PROBE_FIELD(ops, 12, 3);
    const uint32_t syml0 = PROBE_FIELD(ops, 15, 3);
    uint32_t i;

    if (mode == PROBE_MODE_MOP) {
        ft_tensix_mop_template template;

        template.outer_loop_len = outer;
        template.inner_loop_len = inner;
        template.start_op0 = probe_op_word(PROBE_FIELD(ops, 0, 3), addr_mod);
        template.end_op0 = probe_op_word(PROBE_FIELD(ops, 3, 3), addr_mod);
        template.end_op1 = probe_op_word(PROBE_FIELD(ops, 6, 3), addr_mod);
        template.loop_op0 = probe_op_word(sym0, addr_mod);
        template.loop_op1 = probe_op_word(sym1, addr_mod);
        template.loop0_last_instr = probe_op_word(syml0, addr_mod);
        template.loop1_last_instr = probe_op_word(PROBE_FIELD(ops, 18, 3), addr_mod);
        ft_tensix_mop_template_program(&template);
        ft_tensix_mop_template_run();
    } else if (mode == PROBE_MODE_LADDER_UNIFORM) {
        const uint32_t word = probe_op_word(sym0, addr_mod);

        for (i = 0; i < ladder; i++)
            FT_BH_PUSH(word);
    } else if (mode == PROBE_MODE_LADDER_CYCLE) {
        uint32_t words[3];

        words[0] = probe_op_word(sym0, addr_mod);
        words[1] = probe_op_word(sym1, addr_mod);
        words[2] = probe_op_word(syml0, addr_mod);
        for (i = 0; i < ladder; i++)
            FT_BH_PUSH(words[i % 3u]);
    }
}

#endif /* FT_TRISC_MATH */

#ifdef FT_TRISC_UNPACK

/** Two-operand unpack of one tile pair with the MOP replaced by two directly issued UNPACRs. */
static void probe_unpack_one_face(uint32_t operand_a, uint32_t operand_b)
{
    const uint32_t address_a = ft_tensix_cb(operand_a)->fifo_read_pointer - 1u;
    const uint32_t address_b = ft_tensix_cb(operand_b)->fifo_read_pointer - 1u;
    volatile uint32_t FT_BH_REG_PTR *cfg;

    FT_BH_TTINSN_SETADCZW(0x3, 0, 0, 0, 0, 0xF);
    cfg = ft_tensix_cfg_pointer();

    /* Wait for a free unpacker config context. */
    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_UNPACK_SYNC) >= 2u) {
    }
    if (unp_cfg_context == 0) {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_WORD] = address_a;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT0_WORD] = address_b;
    } else {
        cfg[FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_WORD] = address_a;
        cfg[FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT1_WORD] = address_b;
    }
    ft_tensix_semaphore_post(FT_TENSIX_SEM_UNPACK_SYNC);

    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_UNPACK, FT_BH_STALLWAIT_COND_TRISC_CFG);
    FT_BH_TTINSN_UNPACR(FT_BH_UNPACR_SRC_A, 0x1, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0,
                        0, 0, 0, 1);
    FT_BH_TTINSN_UNPACR(FT_BH_UNPACR_SRC_B, 0x1, 0, 0, 0, 1, 1, FT_BH_UNPACR_RAREFYB_DISABLE, 0,
                        0, 0, 0, 1);
    FT_BH_TTINSN_SEMGET(FT_TENSIX_T6_SEM(FT_TENSIX_SEM_UNPACK_SYNC));

    /* Switch to the other config context. */
    unp_cfg_context = 1u - unp_cfg_context;
    if (unp_cfg_context == 0) {
        FT_BH_TTINSN_SETC16(FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_WORD, 0x0000);
    } else {
        FT_BH_TTINSN_SETC16(FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_WORD, 0x0101);
    }
}

#endif /* FT_TRISC_UNPACK */

void kernel_main(void)
{
    const uint32_t cb_in0 = ft_tensix_compile_time_arg(0);
    const uint32_t cb_in1 = ft_tensix_compile_time_arg(1);
    const uint32_t cb_out = ft_tensix_compile_time_arg(2);
    const uint32_t trial_count = ft_tensix_runtime_arg(0);
    uint32_t i;

    (void)cb_in0;
    (void)cb_in1;
    (void)cb_out;

    FT_TENSIX_UNPACK(ft_tensix_unpack_hw_configure_ab(
        ft_cb_unpack_src_format[cb_in0], ft_cb_unpack_src_format[cb_in1],
        ft_cb_unpack_dst_format[cb_in0], ft_cb_unpack_dst_format[cb_in1],
        ft_cb_unpack_tile_face_rows[cb_in0], ft_cb_unpack_tile_face_rows[cb_in1],
        ft_cb_unpack_tile_num_faces[cb_in0], ft_cb_unpack_tile_num_faces[cb_in1],
        ft_tensix_cb(cb_in0)->fifo_page_size, ft_tensix_cb(cb_in1)->fifo_page_size));
    FT_TENSIX_UNPACK(ft_tensix_unpack_ab_init(ft_cb_unpack_tile_face_rows[cb_in0],
                                              ft_cb_unpack_num_face_rows[cb_in0],
                                              ft_cb_unpack_num_face_cols[cb_in0]));
    FT_TENSIX_MATH(ft_tensix_math_pack_sync_init());
    FT_TENSIX_MATH(ft_tensix_math_hw_configure(ft_cb_unpack_dst_format[cb_in0],
                                               ft_cb_unpack_dst_format[cb_in1],
                                               FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_hw_configure(
        ft_cb_pack_src_format[cb_out], ft_cb_pack_dst_format[cb_out],
        ft_tensix_cb(cb_out)->fifo_page_size, ft_cb_pack_tile_num_faces[cb_out],
        ft_cb_pack_partial_face[cb_out], 0 /* relu */, FT_CB_DST_ACCUM_FP32));
    FT_TENSIX_PACK(ft_tensix_pack_init(ft_cb_pack_src_format[cb_out],
                                       ft_cb_pack_tile_face_rows[cb_out],
                                       ft_cb_pack_tile_num_faces[cb_out], 1));
    FT_TENSIX_PACK(ft_tensix_pack_dest_init());
    FT_TENSIX_MATH(probe_configure_addrmod());
    FT_TENSIX_MATH(FT_BH_TTINSN_SETC16(FT_BH_CFG_SRCA_DVALID_CLEAR_DISABLE_WORD, 0));
    FT_TENSIX_MATH(FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_ABD_F));
    /* Waited for once and never popped: every trial re-reads the same two pages. */
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_in0, 1));
    FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front(cb_in1, 1));

    for (i = 0; i < trial_count; i++) {
        const uint32_t shape = ft_tensix_runtime_arg((int)(1u + 2u * i));
        const uint32_t ops = ft_tensix_runtime_arg((int)(2u + 2u * i));

        (void)shape;
        (void)ops;
        FT_TENSIX_UNPACK(probe_unpack_one_face(cb_in0, cb_in1));

        FT_TENSIX_MATH(ft_tensix_math_wait_for_dest_available());
        FT_TENSIX_MATH(FT_BH_PUSH_ZEROACC(FT_BH_ZEROACC_CLR_HALF, FT_CB_DST_ACCUM_FP32, 0,
                                          FT_BH_ADDR_MOD_1, ft_tensix_dest_offset_id % 2u));
        FT_TENSIX_MATH(FT_BH_PUSH_SETC16(
            FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD,
            ft_tensix_dest_offset_id != 0 ? FT_TENSIX_DEST_REGISTER_HALF_SIZE : 0u));
        /* Rewind the DST row counter so the first op lands in slot 0. */
        FT_TENSIX_MATH(FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_D));
        FT_TENSIX_MATH(probe_run_trial(shape, ops));
        /* Release the one operand pair the unpacker supplied. */
        FT_TENSIX_MATH(FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_AB, FT_BH_SETRWC_CR_AB, 0, 0, 0,
                                           FT_BH_SETRWC_SET_AB));
        FT_TENSIX_MATH(FT_BH_TTINSN_SETRWC(FT_BH_SETRWC_CLR_NONE, 0, 0, 0, 0, FT_BH_SETRWC_SET_D));
        FT_TENSIX_MATH(ft_tensix_math_dest_section_done());

        FT_TENSIX_PACK(ft_tensix_compute_cb_reserve_back(cb_out, 1));
        FT_TENSIX_PACK(ft_tensix_packer_wait_for_math_done());
        FT_TENSIX_PACK(ft_tensix_pack_tile(DST_REGISTER, cb_out));
        FT_TENSIX_PACK(ft_tensix_pack_dest_section_done(FT_CB_DST_ACCUM_FP32));
        FT_TENSIX_PACK(ft_tensix_compute_cb_push_back(cb_out, 1));
    }
}
